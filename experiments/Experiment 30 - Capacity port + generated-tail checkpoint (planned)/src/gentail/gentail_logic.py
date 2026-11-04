"""Pure-Python prototype of the generated-tail checkpoint logic (no vLLM import).

This is the position / fence / selection arithmetic that a Merlin patch would use.
It mirrors these source lines (Merlin v3.0.0, _scratch/e23/p2/merlin-src/vllm):

- get_mamba_prefill_checkpoint_position     v1/kv_cache_interface.py:1111-1123
- FullAttentionManager._cache_partial_tail_block (eagle token_limit)
                                             v1/core/single_type_kv_cache_manager.py:889-924
- MambaManager.find_longest_cache_hit (eagle drop of one hash unit)
                                             v1/core/single_type_kv_cache_manager.py:1813-1839
- HybridKVCacheCoordinator.find_longest_cache_hit (fixed point; attention drop)
                                             v1/core/kv_cache_coordinator.py:827-962
- KVCacheCoordinator.num_reprefillable_tokens = num_prefill_lookahead - 1
                                             v1/core/kv_cache_coordinator.py:119, 368-374
- RecoverSSM commit plan (decode block-boundary fold)
                                             model_executor/layers/mamba/ops/recoverssm.py:76-102
- PP sampled-token ring (commit of step T runs pp_size calls later on non-last ranks)
                                             v1/worker/gpu/pp_utils.py:81-90, 130-160
- execute_model order: update_pp_decode_requests -> finish_requests -> ... -> CoW copies
                                             v1/worker/gpu/model_runner.py:2196-2201, 1491-1497

Nothing here is wired into the engine. It exists so the arithmetic and the PP fence
rule can be unit-tested on CPU before anyone writes the real patch.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 1. Positions
# ---------------------------------------------------------------------------


def mamba_prefill_checkpoint_position(num_tokens: int, unit: int, drop_eagle: bool) -> int:
    """Copy of get_mamba_prefill_checkpoint_position (kv_cache_interface.py:1111)."""
    proof_limit = num_tokens if drop_eagle else num_tokens - 1
    pos = proof_limit // unit * unit
    if drop_eagle:
        pos -= unit
    return max(pos, 0)


def current_prompt_hit(num_prompt_prev: int, unit: int = 64, drop_eagle: bool = True) -> int:
    """Hit the next turn gets today: the previous prompt's Mamba tail checkpoint.

    Attention registers its partial tail at floor(P/unit)*unit (eagle token_limit = P,
    single_type_kv_cache_manager.py:903-908); the coordinator drops one unit for the
    eagle groups, and Mamba's own checkpoint sits one unit lower
    (single_type_kv_cache_manager.py:2467-2475). Result: 64..127 tokens short of P,
    which is what the K100Q data shows in 677 / 677 turns.
    """
    return mamba_prefill_checkpoint_position(num_prompt_prev, unit, drop_eagle)


@dataclass(frozen=True)
class TailPositions:
    valid_tokens: int  # tokens whose KV and recurrent state are final and true
    attn_key: int  # where the attention partial-tail hash is registered
    mamba_target: int  # where the Mamba state must sit for the reader's hit
    hit: int  # the reader's resulting hit length (== mamba_target)


def generated_tail_positions(
    num_computed: int,
    num_tokens: int,
    unit: int = 64,
    num_reprefillable: int = 2,
    drop_eagle: bool = True,
) -> TailPositions:
    """Where a finish-time checkpoint must be keyed.

    num_computed: request.num_computed_tokens at finish (after rejection adjustment).
      With MTP it can EXCEED num_tokens when the stop token was an accepted draft and
      later drafts were also accepted and folded into the state
      (scheduler.py:2800-2817 trims output tokens but not num_computed).
    num_tokens: request.num_tokens (prompt + kept output, stop token included; the
      last sampled token is normally never fed, so num_computed == num_tokens - 1).

    valid = min(num_computed, num_tokens) - num_reprefillable
      (get_num_cacheable_tokens, kv_cache_coordinator.py:368-374 / 810-825).
    attn_key = floor(valid / unit) * unit       (attention partial hash)
    mamba_target = attn_key - unit if eagle      (coordinator drops one unit, and the
                                                  Mamba finder searches <= that)
    """
    valid = max(0, min(num_computed, num_tokens) - num_reprefillable)
    attn_key = valid // unit * unit
    target = attn_key - unit if drop_eagle else attn_key
    target = max(target, 0)
    return TailPositions(valid, attn_key, target, target)


def grid_checkpoint(target: int, mamba_block_size: int) -> int:
    """Option B: the newest decode block-boundary state at or below the target.

    The RecoverSSM commit already folds the exact state at every multiple of
    mamba_block_size a verify window crosses (recoverssm.py ops:93-102)."""
    return target // mamba_block_size * mamba_block_size


def ring_slot(position: int, unit: int, ring_size: int) -> int:
    """Option C: the tail-ring page a unit boundary is folded into."""
    assert position % unit == 0
    return (position // unit) % ring_size


# ---------------------------------------------------------------------------
# 2. PP fence model
# ---------------------------------------------------------------------------


@dataclass
class ProducedState:
    """A recurrent state the commit wrote at `position` in step `seq`."""

    request_id: str
    position: int
    seq: int  # sched_step_seq of the step whose commit writes it


def pp_fence_seq(produced_seq: int, pp_size: int) -> int:
    """processed_step_seq needed before a state produced in `produced_seq` is
    committed on every PP rank.

    The last rank commits right after sampling (same step). A non-last rank pushes
    the step's commit into a FIFO of depth pp world size and pops one entry per
    execute_model call (pp_utils.py:81-90, 130-140), so step T's commit runs at the
    start of that rank's call T + pp_size, before that call's frees, CoW copies and
    forward (model_runner.py:2196-2201). processed_step_seq >= T + pp_size means the
    last rank finished step T + pp_size, which it can only do after the first rank
    ran it. sched/processed_step_seq count non-empty steps only
    (scheduler.py:1845-1847, 2352-2355); an extra empty execute call would only pop
    the entry earlier, never later.
    """
    return produced_seq + pp_size


def state_is_pp_safe(state: ProducedState, request_last_sched_seq: int) -> bool:
    """A state is safe to publish at finish only if it was produced by a step
    strictly before the request's last scheduled step.

    Reason: the non-last-rank commit of step T is skipped for a request whose slot was
    freed before call T + pp_size (pp_utils.py:142-147, the req_idx generation check).
    A finished request is freed on the worker at the call that carries its id in
    finished_req_ids (model_runner.py:1366-1378), which can be call L_last + 1 when the
    engine did not have a second microbatch in flight. For T < L_last the request was
    still live at call T + pp_size <= ... (see test_pp_fence_model) - for T == L_last
    it may not be.
    """
    return state.seq < request_last_sched_seq


@dataclass
class PendingPublishQueue:
    """FIFO of (fence_seq, payload), drained like scheduler.deferred_frees
    (scheduler.py:3097-3111)."""

    items: deque = field(default_factory=deque)

    def push(self, fence_seq: int, payload) -> None:
        self.items.append((fence_seq, payload))

    def drain(self, processed_step_seq: int) -> list:
        out = []
        # Fences are pushed in non-decreasing order per producer but can interleave;
        # scan the whole queue (it holds at most one entry per finished request).
        keep = deque()
        while self.items:
            fence, payload = self.items.popleft()
            if fence <= processed_step_seq:
                out.append(payload)
            else:
                keep.append((fence, payload))
        self.items = keep
        return out


class Rank0CommitModel:
    """Toy model of a non-last PP rank: which (request, step) commits were applied.

    Mirrors PPHandler.queue (depth = pp world size, pre-seeded with None) and the
    execute_model order: pop+commit, then free finished requests, then forward."""

    def __init__(self, pp_size: int) -> None:
        self.queue: deque = deque([None] * pp_size)
        self.live: set[str] = set()
        self.applied: set[tuple[str, int]] = set()
        self.skipped: set[tuple[str, int]] = set()

    def execute(self, seq: int, scheduled: list[str], finished: list[str], new: list[str]) -> None:
        slot = self.queue.popleft()
        self.queue.append(None)
        if slot is not None:
            step_seq, req_ids = slot
            for rid in req_ids:
                if rid in self.live:
                    self.applied.add((rid, step_seq))
                else:
                    self.skipped.add((rid, step_seq))
        for rid in finished:
            self.live.discard(rid)
        for rid in new:
            self.live.add(rid)
        if scheduled:
            # receive() overwrites this call's reserved slot
            self.queue[-1] = (seq, list(scheduled))
