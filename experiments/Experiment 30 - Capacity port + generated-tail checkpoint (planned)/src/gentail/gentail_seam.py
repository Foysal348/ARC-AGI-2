"""E30 gentail_seam (GENTAIL_CKPT, design "Option B", PROTOTYPE 2026-11-01): env-gated source patch that publishes
the decode block-boundary Mamba state the RecoverSSM commit already writes, so a Duck turn's prefix hit reaches past
the previous prompt into the previous generation (K100Q replay: 70 % of the perfect 2.43 M re-prefill tokens).

Merlin v3.0.0 (vLLM 0.32 fork), commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5, tree 07c16a79 (the tree the Kaggle
payload installs). Files (pre-patch line numbers):
  vllm/v1/core/gentail_ckpt.py           NEW: GenTailController + maybe_install_gentail (all the logic)
  vllm/v1/core/sched/scheduler.py        class attr (:113), install after configure_recoverssm_pp_checkpoints
                                         (:392-394), on_schedule in update_after_schedule (:1935-1936), on_step_output
                                         after the rejection adjustment (:2458), drain next to _drain_deferred_frees
                                         (:2354-2355)
  vllm/v1/core/kv_cache_manager.py       class attr _gentail next to MAX_GRACE_PINS (:735), on_free / after_free in
                                         free (:727-733) and pop_blocks_for_free (:1080-1086)
Read, not patched: single_type_kv_cache_manager.py (MambaManager :1592-2521, FullAttentionManager :769-924,
  _retire_state_blocks :1673-1702, remove_skipped_blocks :1947-1978), block_pool.py (cache_partial_block :452-579,
  touch / free_blocks :759-830), kv_cache_coordinator.py (:119, :360-399, :805-962), ops/recoverssm.py (:27-115),
  model_states/recoverssm.py (:114-193), pp_utils.py (:81-175).
NOT touched: pp_utils.py, model_runner.py, gpu_worker.py (E25 nb4_ppack) and every E27 dflash_seam file: the file sets
  are disjoint, so the seams compose in any order (each checks only its own files).

Every inserted line carries "GENTAIL_CKPT". With GENTAIL_CKPT unset (or "0") maybe_install_gentail returns None before
it touches anything, every hook is `if self._gentail is not None`, so the original statements run unchanged (the
only extra work is one attribute test per hook site). The new module is imported once at scheduler init (stdlib
imports only at module level).

What the logic does: see the NEW_MODULE_SRC docstring (vllm/v1/core/gentail_ckpt.py). In short:
  record   per request the newest decode window whose commit plan (exact port of prepare_commit_plan_kernel's align
           branch, fed with the window's schedule-time committed count, query length and sampled count) stored the
           state at a block boundary b; hold that block in every Mamba group (one extra ref, so retirement or free
           drops only the table ref) -> PendingTail(request_id, request_generation, position, block_ids,
           producer_step, valid_tokens, token_hash).
  finish   finished with output, not aborted, same generation, producer_step + pp_size <= last_sched_seq + 1 (pp 2: T < L), b > num_prompt,
           b <= floor(valid/64)*64 - 64 (valid = min(committed, num_tokens) - reprefillable): cache the attention /
           MTP-draft full blocks up to A = floor(valid/64)*64, key A (partial key on block A // 4096, or, at a block
           end, verify the full block A/4096 - 1 is hashed at A), QUEUE the PendingTail behind the fence
           producer_step + pp_size. Never publish here.
  drain    fence passed (processed_step_seq), generation unchanged, b <= limit, token hash unchanged -> key the
           Mamba blocks at b, hand them to the finished turn's grace pin (else LRU), enforce the pin budget.
  release  superseded / skipped / preempted / reused-id holds also wait for the fence.

Env (all unset = the original Merlin behaviour):
  GENTAIL_CKPT=1                    on
  GENTAIL_CKPT_STRICT_TRUNC=0       keep a checkpoint after a stop truncation (default 1 = discard; A/B)
  GENTAIL_CKPT_KEEP_PROMPT_HOLD=1   keep the prompt-tail CoW hold a grid hold supersedes (memory A/B)
  GENTAIL_CKPT_PIN_BUDGET_FRAC=f    cap on all grace-pinned blocks, fraction of the pool (default 0.25)
  GENTAIL_CKPT_LOG_PUBLISH=1        one "publish" line per checkpoint (block ids for the gate-3 state reads)
  GENTAIL_CKPT_NOFENCE=1            NEGATIVE CONTROL ONLY (gate 4): publish at finish, no fence, no producer rule,
                                    no position limit, length-capped / truncated finishes allowed
Log lines (EngineCore stdout): "E30_GENTAIL {json}", events install / inactive / stats.

What this file holds:
  NEW_MODULE_SRC, EDITS {rel: [(anchor, replacement), ...]} (each anchor must occur exactly once before, each
  replacement exactly once after), PRE / POST shas, apply_gentail_patch / verify_gentail_patch / revert_gentail_patch
  (fail-safe: originals restored on any error), tree_state, server_env, parse_log, readback.
"""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

MERLIN_COMMIT = "ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5"
MERLIN_TREE = "07c16a79d077f1e06c19d0eea84efd64f0b63540"
SCH_REL = "vllm/v1/core/sched/scheduler.py"
KVM_REL = "vllm/v1/core/kv_cache_manager.py"
NEW_REL = "vllm/v1/core/gentail_ckpt.py"
SCH_PRE = "ea56781115c350cf0c053912bc855f86d6bb47a19cd914e1d2e045b99b2a4a7f"
SCH_POST = "2b670430460b9a4518009b2998d6802fbf156524a2f7f9ccf98bdf9208b18d1e"
KVM_PRE = "9f7ebfd75099a9ea784c54be97fbc87dfb87d2355b88d6a7caed4c9f8ad1dc93"
KVM_POST = "45a8d99c36d7decab341aa4a61ab112c27526ca0c5488c4ef8b52c7a46f0614f"
NEW_SHA = "db5f3beef28bcab27055f52893576aa61dff394b610e6cbda51925099dc1e597"
MARKER = "GENTAIL_CKPT"
ENV = "GENTAIL_CKPT"
ENV_STRICT_TRUNC = "GENTAIL_CKPT_STRICT_TRUNC"
ENV_KEEP_PROMPT_HOLD = "GENTAIL_CKPT_KEEP_PROMPT_HOLD"
ENV_PIN_BUDGET = "GENTAIL_CKPT_PIN_BUDGET_FRAC"
ENV_NOFENCE = "GENTAIL_CKPT_NOFENCE"
ENV_LOG_PUBLISH = "GENTAIL_CKPT_LOG_PUBLISH"
ENV_STATS_EVERY = "GENTAIL_CKPT_STATS_EVERY"
ALL_ENVS = (ENV, ENV_STRICT_TRUNC, ENV_KEEP_PROMPT_HOLD, ENV_PIN_BUDGET, ENV_NOFENCE, ENV_LOG_PUBLISH, ENV_STATS_EVERY)
NB4_FILES = ("vllm/v1/worker/gpu/pp_utils.py", "vllm/v1/worker/gpu/model_runner.py", "vllm/v1/worker/gpu_worker.py")
DFLASH_FILES = ("vllm/model_executor/models/config.py", "vllm/models/qwen4_exp/nvidia/model.py",
                "vllm/model_executor/models/qwen3_dflash.py", "vllm/v1/worker/gpu/spec_decode/dflash/utils.py",
                "vllm/models/qwen4_exp/common/qsa_cache.py", "vllm/model_executor/layers/attention/attention.py",
                "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py", "vllm/model_executor/models/qwen3_dflash2.py")

NEW_MODULE_SRC = r'''"""GENTAIL_CKPT=1 (E30 gentail_seam, design "Option B", 2026-11-01): publish the decode
block-boundary Mamba state that the RecoverSSM commit already writes, so the next Duck
turn's prefix hit reaches past the previous prompt into the generated tokens.

Unset / "0": nothing here is constructed (the scheduler / KV-manager hooks test
`is not None` and run the original statements only).

Flow (Merlin v3.0.0, ad42e37; paths relative to vllm/):
  schedule      on_schedule: the commit-plan inputs of a pure-decode window (committed
                count `num_computed`, `query_len`, step seq T). Taken only when the request
                has no other window in flight.
  output (T)    on_step_output: exact port of prepare_commit_plan_kernel's ALIGN branch
                (model_executor/layers/mamba/ops/recoverssm.py:64-115) with the worker's
                inputs (num_sampled = len(sampled ids)). If the plan stored a boundary
                state at b = (num_computed // bs + 1) * bs into column b / bs - 1, the
                block of that column in every Mamba group is held (one extra ref: the
                table ref goes at retirement, _retire_state_blocks, or at free; the hold
                keeps the block out of the free queue, so PP0's lagging commit of T
                never lands in a reused block). PendingTail(producer_step = T). A newer
                crossing replaces it (the older hold is released behind the PP fence).
                The plan's final count must equal the scheduler's rejection-adjusted
                count, else the record is dropped (counter plan_mismatch).
  free          on_free (before the table is popped): FINISHED_STOPPED only (vLLM
                PR #50551: aborted / length-capped / other finishes are discarded), same request generation, producer_step + pp_size <= last_sched_seq + 1
                (producer rule), b <= floor(valid / unit) * unit - unit with
                valid = min(committed, num_tokens) - num_reprefillable, b > num_prompt
                (producer rule in general: producer_step + pp_size <= last_sched_seq + 1);
                then the attention (and MTP draft) partial key at A = floor(valid/unit)*unit
                is registered on the table block and the PendingTail is QUEUED with
                fence producer_step + pp_size. Nothing is published here.
  drain         drain (scheduler: next to _drain_deferred_frees; KV manager: right after
                the grace pin is taken): processed_step_seq >= fence (every PP rank ran
                the commit of T), the generation is unchanged and b <= the limit ->
                key every Mamba block at b (block-end key, cache_partial_block) and hand
                it to the finished turn's grace pin (else free it as a hashed block,
                evict-last LRU). Otherwise the hold is released.
  Releases (superseded / skipped / preempted / reused id) also wait for the fence.

The per-step guard stays: configure_recoverssm_pp_checkpoints (scheduler.py:88-110) keeps
publish_decode_checkpoints = False under RecoverSSM + PP, and nothing here changes it; this
finish-time path is separate and fenced.
Token identity: a hit needs the reader's chained 64-token hashes to equal the producer's,
so a next prompt that re-renders the response differently (tool-call formatter, reasoning
tokens, stop truncation) misses (gain lost, never a wrong state). At drain the producer's
own prefix hash at b must still equal the hash recorded at finish (G-2+).

Env:
  GENTAIL_CKPT=1                    on (default off)
  GENTAIL_CKPT_STRICT_TRUNC=0       relax: keep a checkpoint when the stop truncated the
                                    output (committed >= num_tokens, the stop was an
                                    accepted draft); default 1 = discard it (audit rule).
                                    The state at b is produced >= 16 windows before the
                                    end, so the relaxed rule is also exact; the counter
                                    stop_truncated measures what strict costs.
  GENTAIL_CKPT_KEEP_PROMPT_HOLD=1   do not release the prompt-tail CoW hold a grid hold
                                    supersedes (memory A/B; default: release it)
  GENTAIL_CKPT_PIN_BUDGET_FRAC=f    cap on all grace-pinned blocks as a fraction of the
                                    pool (default 0.25): oldest pins go first, the newest
                                    always stays (MAX_GRACE_PINS rises 16 -> max_num_seqs)
  GENTAIL_CKPT_LOG_PUBLISH=1        one "publish" log line per published checkpoint
                                    (request, position, block ids, groups, step, fence)
  GENTAIL_CKPT_NOFENCE=1            NEGATIVE CONTROL ONLY: no fence, no producer rule, no
                                    position limit, length-capped and truncated finishes
                                    allowed (publishes the newest crossing at finish, even
                                    one made by the final window, so gate 4 can read a
                                    block PP0 never committed)
Counters (stats): recorded, superseded, queued, deferred (fence not passed at finish),
  published (+_pinned / _lru / _tokens), skipped (+ skip_<reason>), released, adopted
  (+ adopted_tokens: a later request resumed from a published block), evicted (a
  published block lost its key before any adoption), pin_budget_released.
Log lines: "E30_GENTAIL {json}" (events install / inactive / stats).
"""

from __future__ import annotations

import itertools
import json
import os
from collections import Counter
from dataclasses import dataclass

ENV = "GENTAIL_CKPT"
ENV_STRICT_TRUNC = "GENTAIL_CKPT_STRICT_TRUNC"
ENV_KEEP_PROMPT_HOLD = "GENTAIL_CKPT_KEEP_PROMPT_HOLD"
ENV_NOFENCE = "GENTAIL_CKPT_NOFENCE"
ENV_PIN_BUDGET = "GENTAIL_CKPT_PIN_BUDGET_FRAC"
ENV_LOG_PUBLISH = "GENTAIL_CKPT_LOG_PUBLISH"
PUBLISHED_TRACK = 512
# vLLM PR #50551 (finalized decode checkpoints): publish only a turn that ended on a
# stop; aborted / length-capped / error / ignored / repetition turns are discarded.
PUBLISH_STATUS = "FINISHED_STOPPED"
ENV_STATS_EVERY = "GENTAIL_CKPT_STATS_EVERY"  # a stats line every N finishes (default 25)


def _emit(event: str, **kw) -> None:
    rec = {"event": event, "pid": os.getpid()}
    rec.update(kw)
    print("E30_GENTAIL " + json.dumps(rec, sort_keys=True, default=str), flush=True)


def commit_plan_boundary(
    num_computed: int,
    num_sampled: int,
    query_len: int,
    spec_query_len: int,
    mamba_block_size: int,
) -> tuple[int, int, int]:
    """prepare_commit_plan_kernel, ALIGN_MODE, one row (ops/recoverssm.py:64-115).

    Returns (commit_len, final_num_computed, boundary): boundary = the position whose
    state the plan stores into column boundary // bs - 1, or 0 when it stores none.
    A null source state (no state block) is checked by the caller (table lookup)."""
    commit_len = min(max(num_sampled, 0), query_len)
    commit_len = min(commit_len, spec_query_len)
    final = num_computed + commit_len
    next_boundary = (num_computed // mamba_block_size + 1) * mamba_block_size
    crosses = final >= next_boundary
    if commit_len > 0 and crosses:
        return commit_len, final, next_boundary
    return commit_len, final, 0


@dataclass
class PendingTail:
    request_id: str
    request_generation: int
    position: int  # b, a multiple of the Mamba block size
    block_ids: tuple[int, ...]  # one per Mamba group
    producer_step: int  # sched seq of the window whose commit plan wrote b
    valid_tokens: int = -1  # set at finish
    limit: int = -1  # floor(valid / unit) * unit - unit, set at finish
    token_hash: object = None  # request.block_hashes at `position`, set at finish
    blocks: tuple = ()  # the held KVCacheBlocks, one per Mamba group
    request: object = None  # set at finish: its block hashes key the blocks


class GenTailController:
    def __init__(
        self,
        kv_cache_manager,
        *,
        pp_size: int,
        spec_query_len: int,
        max_num_seqs: int,
        mamba_cls,
        attn_cls,
        base_cache_blocks,
    ) -> None:
        self.kvm = kv_cache_manager
        self.pool = kv_cache_manager.block_pool
        coord = kv_cache_manager.coordinator
        managers = list(coord.single_type_managers)
        self.mamba = [m for m in managers if isinstance(m, mamba_cls)]
        self.attn = [
            m
            for m in managers
            if isinstance(m, attn_cls)
            and type(m).cache_blocks is attn_cls.cache_blocks
            and m.enable_caching
            and getattr(m.kv_cache_spec, "prefix_cacheable", True)
        ]
        self._base_cache_blocks = base_cache_blocks
        sizes = {m.block_size for m in self.mamba}
        if len(sizes) != 1:
            raise RuntimeError(f"GENTAIL_CKPT: Mamba block sizes {sorted(sizes)}")
        self.bs = sizes.pop()
        self.unit = self.pool.hash_block_size
        self.repref = int(getattr(coord, "num_reprefillable_tokens", 0))
        self.pp_size = int(pp_size)
        self.spec_query_len = int(spec_query_len)
        self.nofence = os.environ.get(ENV_NOFENCE, "0") == "1"
        self.strict_trunc = os.environ.get(ENV_STRICT_TRUNC, "1") != "0"
        self.keep_prompt_hold = os.environ.get(ENV_KEEP_PROMPT_HOLD, "0") == "1"
        self.log_publish = os.environ.get(ENV_LOG_PUBLISH, "0") == "1"
        self.stats_every = max(1, int(os.environ.get(ENV_STATS_EVERY, "25")))
        self.processed_seq = 0
        self._sched: dict[str, tuple[int, int, int]] = {}
        self._pending: dict[str, PendingTail] = {}
        self._gen: dict[str, int] = {}
        self._obj: dict[str, int] = {}
        self._gen_counter = itertools.count(1)
        self._queue: list[tuple[int, PendingTail, bool]] = []
        # id(block) -> (block, position, hash): published, not yet adopted / evicted
        self._published: dict[int, tuple[object, int, object]] = {}
        # Future hook (no implementation): an INT8 cached-checkpoint store in the
        # style of SGLang #28185 (Int8CheckpointStore / MambaCheckpointPool) could
        # hold published tail states at half the bytes. If set, it is offered every
        # published PendingTail (`tail_store.offer(pend) -> bool`, True = it took the
        # blocks' refs); unset = the blocks stay in the BlockPool / grace pin.
        self.tail_store = None
        frac = float(os.environ.get(ENV_PIN_BUDGET, "0.25"))
        self.pin_budget = max(1, int(frac * self.pool.num_gpu_blocks))
        self.stats: Counter = Counter()
        self._finishes = 0
        # 21 workers can sit between turns at once; the stock cap is 16.
        if max_num_seqs > kv_cache_manager.MAX_GRACE_PINS:
            kv_cache_manager.MAX_GRACE_PINS = int(max_num_seqs)

    # ------------------------------------------------------------------ helpers
    def _touch(self, request) -> None:
        """Request generation: a new Request object under a known id (index reuse)
        invalidates whatever the old one left."""
        rid = request.request_id
        if self._obj.get(rid) == id(request):
            return
        if rid in self._obj:
            self.stats["index_reuse"] += 1
            self._sched.pop(rid, None)
            old = self._pending.pop(rid, None)
            if old is not None:
                self._release(old)
        self._obj[rid] = id(request)
        self._gen[rid] = next(self._gen_counter)
        self._check_adoption(request)

    def _check_adoption(self, request) -> None:
        """First sight of a request (its first schedule, after allocate): did its
        prefix hit resume from a published grid state? (G-5 'adopted')."""
        if not self._published or request.num_computed_tokens <= 0:
            return
        nc = request.num_computed_tokens
        if nc % self.bs:
            return
        col = nc // self.bs - 1
        found = False
        for m in self.mamba:
            table = m.req_to_blocks.get(request.request_id, ())
            if col < len(table):
                hit = self._published.pop(id(table[col]), None)
                found = found or (hit is not None and hit[1] == nc)
        if found:
            self.stats["adopted"] += 1
            self.stats["adopted_tokens"] += nc

    def _sweep_evicted(self) -> None:
        for key, (blk, pos, _h) in list(self._published.items()):
            if blk.block_hash is None or blk.block_hash_num_tokens != pos:
                del self._published[key]
                self.stats["evicted"] += 1
        while len(self._published) > PUBLISHED_TRACK:
            del self._published[next(iter(self._published))]

    def enforce_pin_budget(self) -> None:
        """G-3: cap every grace-pinned block (a grace pin holds ALL hashed blocks
        of a finished turn, not only the new state). Oldest first; the newest pin
        always stays. Grace pins are also yielded first under allocation
        pressure (KVCacheManager.yield_pinned_hits)."""
        pins = self.kvm._grace_pins
        while len(pins) > 1 and sum(len(b) for b in pins.values()) > self.pin_budget:
            self.kvm.release_grace_pin(next(iter(pins)))
            self.stats["pin_budget_released"] += 1

    def _release(self, pend: PendingTail) -> None:
        self._queue.append((pend.producer_step + self.pp_size, pend, False))
        self.drain()

    def _committed(self, request) -> int:
        return request.num_computed_tokens - request.num_in_flight_tokens

    # ------------------------------------------------------------------ hooks
    def on_schedule(self, request, num_scheduled: int, seq: int) -> None:
        """Before update_after_schedule advances the counts (scheduler.py:1936)."""
        self._touch(request)
        rid = request.request_id
        if request.num_in_flight_tokens != 0:
            # Another window is in flight: its rejections are not known yet.
            self._sched.pop(rid, None)
            self.stats["sched_overlap"] += 1
            return
        before = request.num_computed_tokens
        if before < request.num_prompt_tokens:
            self._sched.pop(rid, None)  # prefill chunk / final prompt token
            return
        self._sched[rid] = (seq, before, num_scheduled)

    def on_step_output(
        self,
        request,
        num_scheduled: int,
        num_sampled: int,
        seq: int,
        output_is_stale: bool,
    ) -> None:
        """update_from_output, after the rejection adjustment."""
        rid = request.request_id
        rec = self._sched.pop(rid, None)
        if rec is None or output_is_stale:
            return
        rseq, before, qlen = rec
        if rseq != seq or qlen != num_scheduled:
            self.stats["seq_mismatch"] += 1
            return
        commit_len, final, b = commit_plan_boundary(
            before, num_sampled, qlen, self.spec_query_len, self.bs
        )
        if final != self._committed(request):
            self.stats["plan_mismatch"] += 1
            return
        if b == 0:
            return
        col = b // self.bs - 1
        blocks = []
        for m in self.mamba:
            table = m.req_to_blocks.get(rid, ())
            if col >= len(table) or table[col].is_null:
                self.stats["no_block"] += 1
                return
            blocks.append(table[col])
        self.pool.touch(blocks)  # the hold's ref
        old = self._pending.get(rid)
        self._pending[rid] = PendingTail(
            request_id=rid,
            request_generation=self._gen[rid],
            position=b,
            block_ids=tuple(blk.block_id for blk in blocks),
            producer_step=seq,
            blocks=tuple(blocks),
        )
        self.stats["recorded"] += 1
        if old is not None:
            self.stats["superseded"] += 1
            self._release(old)
        if not self.keep_prompt_hold:
            # The grid hold replaces the prompt-tail CoW hold (no extra pages in
            # steady state); the released one ages as a cached free block.
            for m in self.mamba:
                held = getattr(m, "_retained_checkpoints", {}).pop(rid, None)
                if held is not None:
                    self.pool.free_blocks([held[1]])
                    self.stats["prompt_hold_released"] += 1

    def _finish_reason(self, request, pend: PendingTail, gen) -> str | None:
        status = getattr(request.status, "name", str(request.status))
        if not request.is_finished():
            return "not_finished"
        if status == "FINISHED_ABORTED":
            return "abort"
        if status != PUBLISH_STATUS and not (
            self.nofence and status == "FINISHED_LENGTH_CAPPED"
        ):
            # (the negative control may use max_tokens to end right after a crossing)
            return "status_" + status.lower()
        if request.num_output_tokens <= 0:
            return "no_output"
        if not self.kvm.enable_caching:
            return "no_caching"
        if gen is None or pend.request_generation != gen:
            return "generation"
        committed = self._committed(request)
        if committed >= request.num_tokens:
            self.stats["stop_truncated"] += 1
            if self.strict_trunc and not self.nofence:
                return "stop_truncation"
        # Producer rule, general pp_size: PP0 runs the commit of step T at the start
        # of its call T + pp_size, and frees a finished request's slot in the call
        # that carries the finish, call L + 1 at the earliest (idle engine): the
        # commit survives only if T + pp_size <= L + 1 (pp 2: T <= L - 1).
        if pend.producer_step + self.pp_size > request.last_sched_seq + 1:
            self.stats["final_window"] += 1
            if not self.nofence:
                return "final_window"
        if pend.position <= request.num_prompt_tokens:
            return "within_prompt"
        valid = max(0, min(committed, request.num_tokens) - self.repref)
        attn_key = valid // self.unit * self.unit
        pend.valid_tokens = valid
        pend.limit = attn_key - self.unit
        if pend.position > pend.limit and not self.nofence:
            return "past_limit"
        return None

    def _attention_targets(self, request, attn_key: int):
        """(manager, block or None) per attention / MTP-draft group. The full blocks
        up to A are cached first (the last allocate_slots cached fewer: the final
        step's tokens arrived after it; their KV is final). G-1: when A is a block
        end the full block A / bs - 1 carries the key (Merlin skips a partial key at
        a full boundary, single_type_kv_cache_manager.py:911-912) and must now be
        hashed at A; else the partial key goes on block A // bs. Raises LookupError
        when a group cannot carry the key; no partial key is written before every
        group passed."""
        rid = request.request_id
        out = []
        for m in self.attn:
            self._base_cache_blocks(
                m, request, attn_key, retention_interval=None, replay_boundaries=()
            )
            table = m.req_to_blocks.get(rid, ())
            if attn_key % m.block_size == 0:
                idx = attn_key // m.block_size - 1
                if idx < 0 or idx >= len(table) or table[idx].is_null:
                    raise LookupError("attn_no_full_block")
                full = table[idx]
                if full.block_hash is None or full.block_hash_num_tokens != attn_key:
                    raise LookupError("attn_full_block_unhashed")
                out.append((m, None))
                continue
            idx = attn_key // m.block_size
            if idx >= len(table) or table[idx].is_null:
                raise LookupError("attn_no_block")
            blk = table[idx]
            if blk.block_hash is not None and blk.block_hash_num_tokens != attn_key:
                raise LookupError("attn_block_hashed")
            out.append((m, blk))
        return out

    def on_free(self, request) -> None:
        """KVCacheManager.free / pop_blocks_for_free, before the table is popped."""
        rid = request.request_id
        self._sched.pop(rid, None)
        pend = self._pending.pop(rid, None)
        gen = self._gen.get(rid) if self._obj.get(rid) == id(request) else None
        if request.is_finished():
            self._gen.pop(rid, None)
            self._obj.pop(rid, None)
            self._finishes += 1
        if pend is None:
            if request.is_finished():
                self._maybe_log()
            return
        reason = self._finish_reason(request, pend, gen)
        if reason is None:
            attn_key = pend.limit + self.unit
            try:
                targets = self._attention_targets(request, attn_key)
            except LookupError as e:
                reason = str(e)
        if reason is None:
            idx = pend.position // self.unit - 1
            if idx >= len(request.block_hashes):
                reason = "no_hash"
            else:
                pend.token_hash = request.block_hashes[idx]
        if reason is not None:
            self._skip(reason)
            self._release(pend)
            self._maybe_log()
            return
        for m, blk in targets:
            if blk is not None:
                self.pool.cache_partial_block(
                    request=request,
                    block=blk,
                    num_tokens=attn_key,
                    kv_cache_group_id=m.kv_cache_group_id,
                    block_size=m.block_size,
                )
        pend.request = request
        fence = pend.producer_step + self.pp_size
        self._queue.append((fence, pend, True))
        self.stats["queued"] += 1
        if fence > self.processed_seq:
            self.stats["deferred"] += 1
        self._maybe_log()

    def _skip(self, reason: str) -> None:
        self.stats["skipped"] += 1
        self.stats["skip_" + reason] += 1

    def after_free(self) -> None:
        """KVCacheManager.free / pop_blocks_for_free, right after the grace pin
        was taken: publish what the fence already allows, then the pin budget."""
        self.drain()
        self.enforce_pin_budget()

    def drain(self, processed_step_seq: int | None = None) -> None:
        if processed_step_seq is not None:
            self.processed_seq = processed_step_seq
        if not self._queue:
            return
        keep = []
        for fence, pend, publish in self._queue:
            if fence > self.processed_seq and not (publish and self.nofence):
                keep.append((fence, pend, publish))
                continue
            if publish:
                self._publish(pend)
            else:
                self.pool.free_blocks(pend.blocks)
                self.stats["released"] += 1
        self._queue = keep

    def _publish(self, pend: PendingTail) -> None:
        rid = pend.request_id
        cur = self._gen.get(rid)
        if cur is not None and cur != pend.request_generation:
            # A new request under the same id was seen before the fence.
            self._skip("generation_at_drain")
            self.pool.free_blocks(pend.blocks)
            return
        if (pend.position > pend.limit and not self.nofence) or pend.position % self.bs:
            self._skip("limit_at_drain")
            self.pool.free_blocks(pend.blocks)
            return
        idx = pend.position // self.unit - 1
        hashes = pend.request.block_hashes
        if idx >= len(hashes) or hashes[idx] != pend.token_hash:
            # G-2+: the state's token hash must still be the request's prefix hash.
            self._skip("hash_at_drain")
            self.pool.free_blocks(pend.blocks)
            return
        for m, blk in zip(self.mamba, pend.blocks):
            if blk.block_hash is None:
                self.pool.cache_partial_block(
                    request=pend.request,
                    block=blk,
                    num_tokens=pend.position,
                    kv_cache_group_id=m.kv_cache_group_id,
                    block_size=m.block_size,
                )
        if self.tail_store is not None and self.tail_store.offer(pend):
            self.stats["published"] += 1
            self.stats["published_store"] += 1
            return
        pins = self.kvm._grace_pins.get(rid)
        if pins is not None:
            pins.extend(pend.blocks)  # the hold's ref becomes the grace ref
            self.kvm._grace_block_ids[rid] = frozenset(
                set(self.kvm._grace_block_ids.get(rid, ()))
                | {id(blk) for blk in pend.blocks}
            )
            self.stats["published_pinned"] += 1
        else:
            self.pool.free_blocks(pend.blocks)  # hashed: evict-last LRU
            self.stats["published_lru"] += 1
        self.stats["published"] += 1
        self.stats["published_tokens"] += pend.position
        if self.log_publish:
            _emit(
                "publish",
                request_id=rid,
                position=pend.position,
                block_ids=list(pend.block_ids),
                group_ids=[m.kv_cache_group_id for m in self.mamba],
                producer_step=pend.producer_step,
                fence=pend.producer_step + self.pp_size,
                processed_seq=self.processed_seq,
                valid_tokens=pend.valid_tokens,
                pinned=pins is not None,
            )
        for blk in pend.blocks:
            self._published[id(blk)] = (blk, pend.position, pend.token_hash)
        self.enforce_pin_budget()

    def _maybe_log(self) -> None:
        self._sweep_evicted()
        if self._finishes % self.stats_every == 0:
            pins = self.kvm._grace_pins
            _emit(
                "stats",
                stats=dict(self.stats),
                queue=len(self._queue),
                pending=len(self._pending),
                grace_pins=len(pins),
                grace_pinned_blocks=sum(len(b) for b in pins.values()),
                pin_budget=self.pin_budget,
            )


def maybe_install_gentail(scheduler):
    """Scheduler.__init__ hook. None unless GENTAIL_CKPT=1 and the config is the one
    the design covers (prefix caching, Mamba align, RecoverSSM, no speculative state
    blocks); then defer_block_free must be on (the fence counts its step seqs)."""
    if os.environ.get(ENV, "0") != "1":
        return None
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager,
        MambaManager,
        SingleTypeKVCacheManager,
    )

    kvm = scheduler.kv_cache_manager
    managers = list(kvm.coordinator.single_type_managers)
    mamba = [m for m in managers if isinstance(m, MambaManager)]
    reasons = []
    if not kvm.enable_caching:
        reasons.append("prefix caching off")
    if not getattr(scheduler.cache_config, "use_recoverssm", False):
        reasons.append("RecoverSSM off")
    if not mamba:
        reasons.append("no Mamba groups")
    if any(m.mamba_cache_mode != "align" for m in mamba):
        reasons.append("mamba_cache_mode != align")
    if any(m.num_speculative_blocks for m in mamba):
        reasons.append("speculative Mamba state blocks")
    if not scheduler.num_spec_tokens:
        reasons.append("no speculative decoding")
    if reasons:
        _emit("inactive", reasons=reasons)
        return None
    if not scheduler.defer_block_free:
        raise RuntimeError(
            "GENTAIL_CKPT=1 needs defer_block_free (its PP fence counts "
            "sched_step_seq / processed_step_seq); unset GENTAIL_CKPT"
        )
    ctl = GenTailController(
        kvm,
        pp_size=scheduler.parallel_config.pipeline_parallel_size,
        spec_query_len=1 + scheduler.num_spec_tokens,
        max_num_seqs=scheduler.scheduler_config.max_num_seqs,
        mamba_cls=MambaManager,
        attn_cls=FullAttentionManager,
        base_cache_blocks=SingleTypeKVCacheManager.cache_blocks,
    )
    kvm._gentail = ctl
    _emit(
        "install",
        pp_size=ctl.pp_size,
        spec_query_len=ctl.spec_query_len,
        mamba_groups=[m.kv_cache_group_id for m in ctl.mamba],
        attn_groups=[m.kv_cache_group_id for m in ctl.attn],
        mamba_block=ctl.bs,
        unit=ctl.unit,
        reprefillable=ctl.repref,
        max_grace_pins=kvm.MAX_GRACE_PINS,
        nofence=ctl.nofence,
        strict_trunc=ctl.strict_trunc,
        keep_prompt_hold=ctl.keep_prompt_hold,
        pin_budget_blocks=ctl.pin_budget,
        num_gpu_blocks=ctl.pool.num_gpu_blocks,
    )
    return ctl
'''

EDITS = {
    SCH_REL: [
        (
            "class Scheduler(SchedulerInterface):\n    def __init__(\n",
            "class Scheduler(SchedulerInterface):\n"
            "    # GENTAIL_CKPT (E30 gentail_seam): the generated-tail checkpoint controller\n"
            "    # (None unless GENTAIL_CKPT=1; set in __init__).\n"
            "    _gentail = None\n"
            "\n"
            "    def __init__(\n",
        ),
        (
            "            configure_recoverssm_pp_checkpoints(\n"
            "                self.kv_cache_manager.coordinator.single_type_managers\n"
            "            )\n",
            "            configure_recoverssm_pp_checkpoints(\n"
            "                self.kv_cache_manager.coordinator.single_type_managers\n"
            "            )\n"
            "        # GENTAIL_CKPT (E30 gentail_seam): publish the decode block-boundary\n"
            "        # Mamba state behind a PP fence; None unless GENTAIL_CKPT=1.\n"
            "        from vllm.v1.core.gentail_ckpt import maybe_install_gentail\n"
            "\n"
            "        self._gentail = maybe_install_gentail(self)\n",
        ),
        (
            "            request = self.requests[req_id]\n"
            "            request.num_computed_tokens += num_scheduled_token\n",
            "            request = self.requests[req_id]\n"
            "            if self._gentail is not None:  # GENTAIL_CKPT: commit-plan inputs\n"
            "                self._gentail.on_schedule(\n"
            "                    request, num_scheduled_token, self.sched_step_seq\n"
            "                )\n"
            "            request.num_computed_tokens += num_scheduled_token\n",
        ),
        (
            "            self.processed_step_seq += 1\n"
            "            self._drain_deferred_frees()\n",
            "            self.processed_step_seq += 1\n"
            "            self._drain_deferred_frees()\n"
            "            if self._gentail is not None:  # GENTAIL_CKPT: fenced publishes\n"
            "                self._gentail.drain(self.processed_step_seq)\n",
        ),
        (
            "            # Free encoder inputs only after the step has actually executed.\n",
            "            if self._gentail is not None:  # GENTAIL_CKPT: after the rejections\n"
            "                self._gentail.on_step_output(\n"
            "                    request,\n"
            "                    num_tokens_scheduled,\n"
            "                    len(generated_token_ids),\n"
            "                    self.processed_step_seq,\n"
            "                    output_is_stale,\n"
            "                )\n"
            "\n"
            "            # Free encoder inputs only after the step has actually executed.\n",
        ),
    ],
    KVM_REL: [
        (
            "    MAX_GRACE_PINS = 16\n",
            "    MAX_GRACE_PINS = 16\n"
            "    # GENTAIL_CKPT (E30 gentail_seam): the controller (set by the scheduler,\n"
            "    # None unless GENTAIL_CKPT=1); it raises MAX_GRACE_PINS on the instance.\n"
            "    _gentail = None\n",
        ),
        (
            "        self.release_pinned_hit(request.request_id)\n"
            "        if not self._grace_eligible(request):\n"
            "            self.coordinator.free(request.request_id)\n"
            "            return\n"
            "        blocks = self.coordinator.pop_blocks_for_free(request.request_id)\n"
            "        self._take_grace_pin(request, blocks)\n"
            "        self.block_pool.free_blocks(blocks)\n",
            "        self.release_pinned_hit(request.request_id)\n"
            "        if self._gentail is not None:  # GENTAIL_CKPT: before the table goes\n"
            "            self._gentail.on_free(request)\n"
            "        if not self._grace_eligible(request):\n"
            "            self.coordinator.free(request.request_id)\n"
            "            return\n"
            "        blocks = self.coordinator.pop_blocks_for_free(request.request_id)\n"
            "        self._take_grace_pin(request, blocks)\n"
            "        self.block_pool.free_blocks(blocks)\n"
            "        if self._gentail is not None:  # GENTAIL_CKPT: after the grace pin\n"
            "            self._gentail.after_free()\n",
        ),
        (
            "        self.release_pinned_hit(request.request_id)\n"
            "        blocks = self.coordinator.pop_blocks_for_free(request.request_id)\n"
            "        # A finished turn whose last step is still in flight: the grace takes\n"
            "        # its ref now; the table refs go when the fence drains.\n"
            "        if self._grace_eligible(request):\n"
            "            self._take_grace_pin(request, blocks)\n"
            "        return blocks\n",
            "        self.release_pinned_hit(request.request_id)\n"
            "        if self._gentail is not None:  # GENTAIL_CKPT: before the table goes\n"
            "            self._gentail.on_free(request)\n"
            "        blocks = self.coordinator.pop_blocks_for_free(request.request_id)\n"
            "        # A finished turn whose last step is still in flight: the grace takes\n"
            "        # its ref now; the table refs go when the fence drains.\n"
            "        if self._grace_eligible(request):\n"
            "            self._take_grace_pin(request, blocks)\n"
            "        if self._gentail is not None:  # GENTAIL_CKPT: after the grace pin\n"
            "            self._gentail.after_free()\n"
            "        return blocks\n",
        ),
    ],
}
PATCHED_RELS = (SCH_REL, KVM_REL)
if set(PATCHED_RELS + (NEW_REL,)) & set(NB4_FILES + DFLASH_FILES):
    raise RuntimeError("E30 gentail_seam overlaps the NB4 / E27 seam files")


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _lf(t):
    return t.replace("\r\n", "\n")


def patch_text(text, edits):
    """Apply anchored edits: every anchor exactly once before, every replacement exactly once after."""
    for anchor, repl in edits:
        anchor, repl = _lf(anchor), _lf(repl)
        n = text.count(anchor)
        if n != 1:
            raise ValueError(f"GENTAIL anchor occurs {n} times (need 1): {anchor[:70]!r}")
        text = text.replace(anchor, repl)
        if text.count(repl) != 1:
            raise ValueError(f"GENTAIL replacement not unique after edit: {repl[:70]!r}")
    return text


def unpatch_text(text, edits):
    for anchor, repl in reversed(edits):
        anchor, repl = _lf(anchor), _lf(repl)
        n = text.count(repl)
        if n != 1:
            raise ValueError(f"GENTAIL replacement occurs {n} times (need 1): {repl[:70]!r}")
        text = text.replace(repl, anchor)
    return text


def new_module_bytes():
    return _lf(NEW_MODULE_SRC).encode("utf-8")


PRE = {SCH_REL: SCH_PRE, KVM_REL: KVM_PRE}
POST = {SCH_REL: SCH_POST, KVM_REL: KVM_POST}


def _file_state(tree, rel):
    f = Path(tree) / rel
    if rel == NEW_REL:
        if not f.exists():
            return "not_applied", None
        h = sha256_bytes(f.read_bytes())
        return ("applied" if h == NEW_SHA else "foreign"), h
    data = f.read_bytes()
    h = sha256_bytes(data)
    if h == POST[rel] and MARKER.encode() in data:
        return "applied", h
    if h == PRE[rel]:
        return "not_applied", h
    return "foreign", h


def verify_gentail_patch(tree):
    """Read-only: 'applied' / 'not_applied' (all three files agree), else 'foreign' / 'mixed'."""
    files = {}
    for rel in PATCHED_RELS + (NEW_REL,):
        st, h = _file_state(tree, rel)
        files[rel] = {"state": st, "sha": h}
    states = {v["state"] for v in files.values()}
    state = states.pop() if len(states) == 1 else ("foreign" if "foreign" in states else "mixed")
    return {"state": state, "files": files}


def tree_state(tree):
    s = verify_gentail_patch(tree)["state"]
    return {"not_applied": "ORIGINAL", "applied": "FULLY_PATCHED"}.get(s, "INVALID_MIXED")


def _py_compile(py, files):
    cp = subprocess.run([py, "-c", "import py_compile, sys\nfor f in sys.argv[1:]: py_compile.compile(f, doraise=True)"]
                        + [str(f) for f in files], capture_output=True, text=True, timeout=300)
    return {"rc": cp.returncode, "err": cp.stderr[-600:]}


def _drop_pyc(f):
    d = f.parent / "__pycache__"
    if d.is_dir():
        for pyc in d.glob(f.stem + ".*.pyc"):
            pyc.unlink()


def _write_atomic(f, data):
    tmp = f.with_name(f.name + ".e30tmp")
    tmp.write_bytes(data)
    os.replace(tmp, f)


def _restore(tree, originals):
    """Fail-safe: put back every original byte string (None = the file did not exist)."""
    for rel, data in originals.items():
        f = Path(tree) / rel
        if data is None:
            if f.exists():
                f.unlink()
        else:
            _write_atomic(f, data)
        _drop_pyc(f)


def apply_gentail_patch(tree, py=None):
    """Patch the installed tree (the folder holding vllm/). Refuses unless every file is at Merlin ad42e37's pre-sha
    (and gentail_ckpt.py absent), or everything is already applied. All new texts are built and sha-checked before
    anything is written; any failure while writing or verifying restores the original bytes."""
    tree = Path(tree)
    py = py or sys.executable
    v = verify_gentail_patch(tree)
    rec = {"tree": str(tree), "before": v}
    if v["state"] == "applied":
        rec["applied"] = "already_applied"
    elif v["state"] != "not_applied":
        raise RuntimeError("PATCH_GENTAIL_PRE_SHA_MISMATCH: " + json.dumps(v)[:900])
    else:
        originals = {rel: (tree / rel).read_bytes() for rel in PATCHED_RELS}
        originals[NEW_REL] = None
        new = {}
        for rel in PATCHED_RELS:
            out = patch_text(originals[rel].decode("utf-8"), EDITS[rel]).encode("utf-8")
            if sha256_bytes(out) != POST[rel]:
                raise RuntimeError(f"PATCH_GENTAIL_POST_SHA_MISMATCH {rel}: {sha256_bytes(out)}")
            new[rel] = out
        new[NEW_REL] = new_module_bytes()
        if sha256_bytes(new[NEW_REL]) != NEW_SHA:
            raise RuntimeError(f"PATCH_GENTAIL_NEW_SHA_MISMATCH: {sha256_bytes(new[NEW_REL])}")
        try:
            for rel, out in new.items():
                _write_atomic(tree / rel, out)
                _drop_pyc(tree / rel)
            after = verify_gentail_patch(tree)
            comp = _py_compile(py, [tree / rel for rel in new])
            if after["state"] != "applied" or comp["rc"]:
                raise RuntimeError("PATCH_GENTAIL_VERIFY_FAILED: " + json.dumps(
                    {"after": after, "py_compile": comp})[:900])
        except BaseException:
            _restore(tree, originals)
            raise
        rec["applied"] = "applied"
    rec["after"] = verify_gentail_patch(tree)
    rec["py_compile"] = _py_compile(py, [tree / rel for rel in PATCHED_RELS + (NEW_REL,)])
    if rec["after"]["state"] != "applied" or rec["py_compile"]["rc"]:
        raise RuntimeError("PATCH_GENTAIL_VERIFY_FAILED: " + json.dumps(
            {k: rec.get(k) for k in ("after", "py_compile")})[:900])
    rec["ok"] = True
    return rec


def revert_gentail_patch(tree, py=None):
    """Back to the exact Merlin files (anchors restored, sha == pre; gentail_ckpt.py removed). Refuses unless the tree
    is fully applied or already original. Fail-safe like apply."""
    tree = Path(tree)
    py = py or sys.executable
    v = verify_gentail_patch(tree)
    rec = {"tree": str(tree), "before": v}
    if v["state"] == "not_applied":
        rec["reverted"] = "not_applied"
    elif v["state"] != "applied":
        raise RuntimeError("REVERT_GENTAIL_SHA_MISMATCH: " + json.dumps(v)[:900])
    else:
        originals = {rel: (tree / rel).read_bytes() for rel in PATCHED_RELS + (NEW_REL,)}
        old = {}
        for rel in PATCHED_RELS:
            out = unpatch_text(originals[rel].decode("utf-8"), EDITS[rel]).encode("utf-8")
            if sha256_bytes(out) != PRE[rel]:
                raise RuntimeError(f"REVERT_GENTAIL_RESULT_MISMATCH {rel}: {sha256_bytes(out)}")
            old[rel] = out
        try:
            for rel, out in old.items():
                _write_atomic(tree / rel, out)
                _drop_pyc(tree / rel)
            (tree / NEW_REL).unlink()
            _drop_pyc(tree / NEW_REL)
            after = verify_gentail_patch(tree)
            if after["state"] != "not_applied":
                raise RuntimeError("REVERT_GENTAIL_VERIFY_FAILED: " + json.dumps(after)[:900])
        except BaseException:
            _restore(tree, originals)
            raise
        rec["reverted"] = "reverted"
    rec["after"] = verify_gentail_patch(tree)
    rec["py_compile"] = _py_compile(py, [tree / rel for rel in PATCHED_RELS])
    if rec["after"]["state"] != "not_applied" or rec["py_compile"]["rc"]:
        raise RuntimeError("REVERT_GENTAIL_VERIFY_FAILED: " + json.dumps(rec)[:900])
    rec["ok"] = True
    return rec


def server_env(on=True, strict_trunc=True, keep_prompt_hold=False, pin_budget_frac=None, nofence=False,
               log_publish=False, stats_every=None):
    """Env for the server (EngineCore inherits it). nofence=True is gate 4's NEGATIVE CONTROL only."""
    env = {ENV: "1" if on else "0"}
    if log_publish:
        env[ENV_LOG_PUBLISH] = "1"
    if stats_every is not None:
        env[ENV_STATS_EVERY] = str(int(stats_every))
    if not strict_trunc:
        env[ENV_STRICT_TRUNC] = "0"
    if keep_prompt_hold:
        env[ENV_KEEP_PROMPT_HOLD] = "1"
    if pin_budget_frac is not None:
        env[ENV_PIN_BUDGET] = str(float(pin_budget_frac))
    if nofence:
        env[ENV_NOFENCE] = "1"
    return env


def parse_log(text):
    out = []
    for line in text.splitlines():
        i = line.find("E30_GENTAIL ")
        if i < 0:
            continue
        try:
            out.append(json.loads(line[i + len("E30_GENTAIL "):]))
        except ValueError:
            continue
    return out


def readback(log_text):
    """Labels for a run: INSTALLED (+ its config), INACTIVE (+ reasons) or ABSENT; the last stats line."""
    recs = parse_log(log_text)
    inst = [r for r in recs if r.get("event") == "install"]
    inact = [r for r in recs if r.get("event") == "inactive"]
    stats = [r for r in recs if r.get("event") == "stats"]
    label = "INSTALLED" if inst else ("INACTIVE" if inact else "ABSENT")
    return {"label": label, "install": inst[-1] if inst else None, "inactive": inact[-1] if inact else None,
            "last_stats": stats[-1] if stats else None}
