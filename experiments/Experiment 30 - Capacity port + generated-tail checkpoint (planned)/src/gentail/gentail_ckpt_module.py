"""GENTAIL_CKPT=1 (E30 gentail_seam, design "Option B", 2026-11-01): publish the decode
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
