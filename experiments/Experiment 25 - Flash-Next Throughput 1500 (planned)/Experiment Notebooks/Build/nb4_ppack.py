"""E25 NB4 arms Q-pack / Q-defer: the stage-0 in-graph stall behind the PP sampled-token broadcast, two env-gated fixes
patched into Merlin v3.0.0's (vLLM 0.32 fork) V2 model runner (VLLM_USE_V2_MODEL_RUNNER=1).

Source (Merlin ad42e37, tree 07c16a79 = the tree the Kaggle payload installs; line numbers pre-patch):
  vllm/v1/worker/gpu/pp_utils.py      PPHandler: receive :219-275, broadcast :277-310, broadcast_drafts :202-217,
                                      get_prev_sampled_outputs :129-175, ring (pp_size deep) :85-90, group :97-100
  vllm/v1/worker/gpu/model_runner.py  execute_model :2177 (update_pp_decode_requests :2196 -> def :1385-1398),
                                      empty step :2207, forward launch :2492-2544, finish_forward :2552,
                                      sample_tokens :2598 (receive :2647, broadcast :2679, broadcast_drafts :2818),
                                      shutdown :2891-2894
  vllm/v1/worker/gpu_worker.py        Worker.profile :1690, Worker.shutdown :1953

The three broadcasts today (group pp_broadcast = sibling of the PP device group, ranks [0,2] and [1,3]; side stream
PPHandler.broadcast_stream on every rank; n = requests in the micro-batch, K = 3):
  #1 int64 [n, K+1] sampled tokens (zero-padded to K+1)   sender: broadcast() in sample_tokens, right after sample()
  #2 int32 [2, n]   (num_sampled, num_rejected)           sender: broadcast(), right after #1
  #3 int64 [n, K]   draft tokens (req_states rows)         sender: broadcast_drafts() after propose()
  receiver (stage 0, non-last): receive() in sample_tokens(T), after execute_model(T) launched graph(T):
    broadcast_stream.wait_stream(main) -> #1, #2, #3 -> record_event; slot T = queue[-1].
  consumer: execute_model(T + pp_size) -> update_pp_decode_requests -> get_prev_sampled_outputs pops slot T,
    main_stream.wait_event(slot.event), draft rows -> req_states.draft_tokens, then post_update.
  graph(T + 1) waits only for slot T - 1: it has NO data dependency on receive(T). The NB1 B1 stall (node #173, 21 ms,
  ends 4-9 us after receive(T-1) #2) is therefore a false dependency (hardware queue / ordering), not a data wait.

Q-pack (NB4_PPACK=1, every PP rank): one int64 message of n * (2K + 3) elements = [sampled n*(K+1) | num_sampled n |
  num_rejected n | drafts n*K]. The sender writes the sampled part + counts at the original #1/#2 stream point
  (broadcast()) and sends the whole message once, after the drafts (broadcast_drafts()). The receiver posts ONE
  broadcast + record_event (nothing else on the side stream behind it) and keeps contiguous views; the int64 -> int32
  count cast runs on the main stream after wait_event (exact for these values). The consumer already waited for #3,
  so the slot's ready time is unchanged.
Q-defer (NB4_PPDEFER=1, non-last ranks only; the sender is untouched, so the wire format is unchanged): receive()
  takes every host decision as before (mask, generation snapshot, rows) but parks the slot (event None). The parked
  receive is posted by flush_deferred_receive() right after the NEXT forward launch (model_runner: after
  finish_forward, and on empty steps), i.e. stream-ordered after graph(T + 1). Fallbacks keep the op order and the
  T - 2 wait: receive() flushes an older parked slot first; get_prev_sampled_outputs posts a still-parked slot before
  it waits on (or skips) it. Graph capture never posts (dummy_run excluded, is_current_stream_capturing checked).
Both may be set together (packed message, posted late). Both unset: the patched files run the original code paths
(extra work = a few attribute checks; no NCCL / stream / tensor op is added, moved or changed).

rev 2 (2026-11-01, review NB4_PR53948_COMPARISON_2026-11-01.md, fixes F1 / F2 / F3 / R1):
  F1  GPUModelRunner.shutdown posts a parked receive (flush site "shutdown") before torch.accelerator.synchronize() and
      logs one last 'stats' line (key "final_shutdown"). Hang H1: the peer's send of the last step stays outstanding.
  F2  vllm/v1/worker/gpu_worker.py (third patched file): Worker.profile() and Worker.shutdown() post a parked receive
      (site "rpc") first, inside the same RPC as the profiler's / teardown's device sync. Hang H2 (/stop_profile while
      the profiler is still active). Guarded: no model_runner / no pp_handler (PP off) / no method -> nothing runs.
  F3  last rank, either env set: the draft rows draft_tokens[idx_mapping] are gathered on the MAIN stream before the
      broadcast stream is entered (record_stream on the gathered tensor; idx_mapping is no longer used on the side
      stream): Q-pack in _nb4_send_packed, Q-defer (original wire format) in a new broadcast_drafts branch (flag
      nb4_any). Wrong data W1: a later propose() could rewrite those rows before a lagging broadcast stream reads them.
  R1  readback_label: Q-defer is ENGAGED only if the LAST 'stats' line of every non-last rank has
      flush_exec >= 0.9 * defer_park and flush_recv + flush_consume <= 0.05 * defer_park, else ENGAGED_FALLBACK
      (deferral collapsed to post-at-consume: a different timing, not a clean engage). 'stats' lines are also written
      at every 1000th count of a key, so the last line of a run is recent.

What this file holds (the notebook embeds or imports it):
  PATCHES {rel: (pre_sha, post_sha, marker, unified diff)}  strict file patch (NB1 #60490 / NB4 P5 style), 3 files
  apply_ppack_patch / verify_ppack_patch / revert_ppack_patch, server_env, parse_log, readback_label
Log lines (stdout of every worker): "NB4_PPQ {json}" with event init / stats.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MERLIN_COMMIT = "ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5"
MERLIN_TREE = "07c16a79d077f1e06c19d0eea84efd64f0b63540"
PP_REL = "vllm/v1/worker/gpu/pp_utils.py"
MR_REL = "vllm/v1/worker/gpu/model_runner.py"
PP_PRE = "6be931deaf8bee6dc9571c3c84d7e92c9f26c618c09dcb58a8b384bff487a662"
PP_POST = "8b7fa4a6e4598ff4d4a56eb6e595efb1e494276b12b187b099973c262e90c69a"
MR_PRE = "05e889afba5dd60b6995b78aaa1f69051770ab02b638680097eccbf4e592d64c"
MR_POST = "2c25fda60dbd5f2506ff989c29ccfe8fc0ff082fbad457fc3ae9c89ed9a185d6"
GW_REL = "vllm/v1/worker/gpu_worker.py"
GW_PRE = "a97d08ef042a53fcd139bae041fff407fce75796650d936c200c699f0d63a489"
GW_POST = "5642b89d918b30e5b70b1067d53776e379613035b7b6e91e93cf0ee4279cfef5"
PP_MARKER = "# E25 NB4 PPACK / PPDEFER: PP sampled-token broadcast variants (env-gated)."
MR_MARKER = "# E25 NB4 PPDEFER: post the previous step's parked receive behind this"
GW_MARKER = "# E25 NB4 PPDEFER: post a parked PP receive inside this RPC, before the"
ENV_PACK = "NB4_PPACK"
ENV_DEFER = "NB4_PPDEFER"
K_DEFAULT = 3

PP_PATCH = r'''# E25 NB4 Q-pack / Q-defer onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5,
# tree 07c16a79d077f1e06c19d0eea84efd64f0b63540 = the tree the Kaggle payload installs).
# PPHandler: env-gated packed PP broadcast (NB4_PPACK=1) and deferred receive (NB4_PPDEFER=1); inert when unset.
# pre-patch sha256  6be931deaf8bee6dc9571c3c84d7e92c9f26c618c09dcb58a8b384bff487a662
# post-patch sha256 8b7fa4a6e4598ff4d4a56eb6e595efb1e494276b12b187b099973c262e90c69a
--- a/vllm/v1/worker/gpu/pp_utils.py
+++ b/vllm/v1/worker/gpu/pp_utils.py
@@ -37,6 +37,11 @@
     # Model-state data the step's postprocess needs once the sampled tokens
     # arrive (RecoverSSM: the snapshot of the step's deferred commits).
     deferred_state: object | None = None
+    # E25 NB4 PPACK / PPDEFER (env-gated; both unused when the env is unset).
+    # PPDEFER: rows of a receive that is parked, not posted yet (event is None).
+    nb4_num_reqs: int = -1
+    # PPACK: the int64 [2, num_reqs] counts view, cast to int32 at consume time.
+    nb4_packed_counts: torch.Tensor | None = None
 
 
 def compute_need_sampled_mask(input_batch: InputBatch) -> np.ndarray | None:
@@ -56,6 +61,50 @@
     return need_sampled_mask if need_sampled_mask.any() else None
 
 
+# E25 NB4 PPACK / PPDEFER: PP sampled-token broadcast variants (env-gated).
+_NB4_STAT_KEYS = (
+    "packed_stash",
+    "packed_send",
+    "packed_recv",
+    "defer_park",
+    "defer_post",
+    "flush_exec",
+    "flush_empty",
+    "flush_recv",
+    "flush_consume",
+    "flush_rpc",
+    "flush_shutdown",
+)
+_NB4_LOG_AT = (1, 10, 100, 1000, 10000, 100000, 1000000)
+# Also every 1000th count of a key, so the last 'stats' line of a run is recent.
+_NB4_LOG_EVERY = 1000
+
+
+def nb4_pack_views(
+    flat: torch.Tensor, num_reqs: int, k: int
+) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
+    """PPACK layout of the one int64 message of num_reqs * (2k + 3) elements:
+    sampled tokens [num_reqs, k + 1] | num_sampled, num_rejected [2, num_reqs] |
+    draft tokens [num_reqs, k]. All three are contiguous views of `flat`."""
+    a = num_reqs * (k + 1)
+    b = a + 2 * num_reqs
+    return (
+        flat[:a].view(num_reqs, k + 1),
+        flat[a:b].view(2, num_reqs),
+        flat[b:].view(num_reqs, k),
+    )
+
+
+def _nb4_log(event: str, **kw) -> None:
+    import json
+    import os
+    import time
+
+    rec = {"event": event, "pid": os.getpid(), "t": round(time.time(), 3)}
+    rec.update(kw)
+    print("NB4_PPQ " + json.dumps(rec, sort_keys=True, default=str), flush=True)
+
+
 class PPHandler:
     """Runs the PP sampled-token broadcast/recv on a side stream so the
     default stream isn't gated by the matching peer call. Step T's recv is
@@ -99,6 +148,40 @@
             group_desc="pp_broadcast"
         )
         self.aux_hidden_state_relay_keys: tuple[str, ...] = ()
+
+        # E25 NB4 PPACK / PPDEFER: both False unless the env is "1"; then the
+        # original code paths run unchanged.
+        # NB4_PPACK=1 (all PP ranks): one packed int64 broadcast per step, sent
+        # after the drafts, instead of three.
+        # NB4_PPDEFER=1 (non-last ranks): the receive is parked and posted after
+        # the next step's forward launch (flush_deferred_receive).
+        # Either env (last rank): the draft rows are gathered on the main stream
+        # before the broadcast stream reads them (broadcast_drafts).
+        import os as _nb4_os
+
+        _nb4_env = (
+            _nb4_os.environ.get("NB4_PPACK", "0"),
+            _nb4_os.environ.get("NB4_PPDEFER", "0"),
+        )
+        self.nb4_ppack = _nb4_env[0] == "1"
+        self.nb4_ppdefer = _nb4_env[1] == "1" and not self.is_last_rank
+        self.nb4_any = "1" in _nb4_env
+        self.nb4_pack_stash: torch.Tensor | None = None
+        self.nb4_stats: dict[str, int] = dict.fromkeys(_NB4_STAT_KEYS, 0)
+        if self.nb4_any:
+            _nb4_log(
+                "init",
+                env_ppack=_nb4_env[0],
+                env_ppdefer=_nb4_env[1],
+                ppack=self.nb4_ppack,
+                ppdefer=self.nb4_ppdefer,
+                is_last_rank=self.is_last_rank,
+                k=num_speculative_steps,
+                ring_depth=ring_depth,
+                device=str(device),
+            )
+        if self.nb4_ppack and num_speculative_steps <= 0:
+            raise RuntimeError("NB4_PPACK=1 needs speculative decoding (K > 0)")
 
     def on_req_idx_freed(self, req_idx: int) -> None:
         self.req_idx_gen_np[req_idx] += 1
@@ -139,6 +222,11 @@
         self.queue.append(None)
         if slot is None:
             return None
+        if slot.event is None:
+            # E25 NB4 PPDEFER: parked and never flushed since its step; post it
+            # now (before any early return: the sender has sent it).
+            slot = self._nb4_post(slot)
+            self._nb4_count("flush_consume")
 
         # Skip requests which did not need sampled output and/or those already
         # finished. The post_update kernel skips the -1 entries.
@@ -154,6 +242,12 @@
             idx_mapping = async_tensor_h2d(idx_mapping_np, device=self.device)
 
         self.main_stream.wait_event(slot.event)
+        if slot.nb4_packed_counts is not None:
+            # E25 NB4 PPACK: int64 -> int32 on the main stream, after the wait.
+            num_sampled, num_rejected = slot.nb4_packed_counts.to(torch.int32).unbind(
+                dim=0
+            )
+            slot = replace(slot, num_sampled=num_sampled, num_rejected=num_rejected)
         if slot.draft_tokens is not None and draft_tokens_to_update is not None:
             draft_tokens = slot.draft_tokens
             draft_idx_mapping = slot.idx_mapping
@@ -204,7 +298,23 @@
     ) -> None:
         """Broadcast draft proposals so non-last ranks can embed real token ids."""
         assert self.is_last_rank
+        if self.nb4_ppack:
+            # E25 NB4 PPACK: the one packed message goes out here.
+            self._nb4_send_packed(draft_tokens, input_batch)
+            return
         if compute_need_sampled_mask(input_batch) is None:
+            return
+        if self.nb4_any:
+            # E25 NB4: gather the rows on the main stream now. draft_tokens is
+            # the persistent table: a later propose() may rewrite these rows
+            # before a lagging broadcast stream would read them.
+            send = draft_tokens[input_batch.idx_mapping].contiguous()
+            with torch.cuda.stream(self.broadcast_stream):
+                self.broadcast_stream.wait_stream(self.main_stream)
+                send.record_stream(self.broadcast_stream)
+                torch.distributed.broadcast(
+                    send, src=self.last_rank, group=self.broadcast_group
+                )
             return
         with torch.cuda.stream(self.broadcast_stream):
             self.broadcast_stream.wait_stream(self.main_stream)
@@ -228,6 +338,9 @@
         # Snapshot the per-slot generation counter so a later free of any of
         # these RequestStates request indices is detectable at consume time.
         gen_at_receive_np = self.req_idx_gen_np[input_batch.idx_mapping_np]
+        if self.nb4_ppack or self.nb4_ppdefer:
+            # E25 NB4 PPACK / PPDEFER.
+            return self._nb4_receive(input_batch, need_sampled_mask, gen_at_receive_np)
 
         num_reqs = input_batch.num_reqs
         with torch.cuda.stream(self.broadcast_stream):
@@ -290,6 +403,11 @@
 
         if current_platform.is_xpu():
             self.main_stream.synchronize()
+        if self.nb4_ppack:
+            # E25 NB4 PPACK: snapshot into the packed buffer now (same stream
+            # point as the original), send it with the drafts.
+            self._nb4_stash_packed(sampled_token_ids, num_sampled, num_rejected)
+            return
 
         with torch.cuda.stream(self.broadcast_stream):
             self.broadcast_stream.wait_stream(self.main_stream)
@@ -308,3 +426,188 @@
             )
             for tensor in (sampled_token_ids, num_sampled, num_rejected):
                 tensor.record_stream(self.broadcast_stream)
+
+    # ---- E25 NB4 PPACK / PPDEFER (env-gated; never reached when both unset) ----
+    def _nb4_count(self, key: str) -> None:
+        v = self.nb4_stats[key] = self.nb4_stats[key] + 1
+        if v in _NB4_LOG_AT or v % _NB4_LOG_EVERY == 0:
+            _nb4_log("stats", key=key, stats=dict(self.nb4_stats))
+
+    def _nb4_stash_packed(
+        self,
+        sampled_token_ids: torch.Tensor,
+        num_sampled: torch.Tensor,
+        num_rejected: torch.Tensor,
+    ) -> None:
+        """Last rank, PPACK: what the original broadcast() sends, written into
+        the packed buffer on the broadcast stream at the same stream point."""
+        if self.nb4_pack_stash is not None:
+            raise RuntimeError("NB4_PPACK: the previous packed message was never sent")
+        k = self.num_speculative_steps
+        num_reqs = sampled_token_ids.shape[0]
+        with torch.cuda.stream(self.broadcast_stream):
+            self.broadcast_stream.wait_stream(self.main_stream)
+            flat = torch.empty(
+                num_reqs * (2 * k + 3), dtype=torch.int64, device=self.device
+            )
+            tokens, counts, _ = nb4_pack_views(flat, num_reqs, k)
+            send_tokens = torch.nn.functional.pad(
+                sampled_token_ids,
+                (0, self.max_sample_len - sampled_token_ids.shape[-1]),
+            )
+            combined = torch.stack((num_sampled, num_rejected), dim=0)
+            if send_tokens.shape != tokens.shape or combined.shape != counts.shape:
+                raise RuntimeError(
+                    "NB4_PPACK: shape mismatch "
+                    f"{tuple(send_tokens.shape)} {tuple(combined.shape)} n={num_reqs}"
+                )
+            tokens.copy_(send_tokens)
+            counts.copy_(combined)
+            for tensor in (sampled_token_ids, num_sampled, num_rejected):
+                tensor.record_stream(self.broadcast_stream)
+        self.nb4_pack_stash = flat
+        self._nb4_count("packed_stash")
+
+    def _nb4_send_packed(
+        self, draft_tokens: torch.Tensor, input_batch: InputBatch
+    ) -> None:
+        """Last rank, PPACK: add the drafts and send the one message."""
+        flat = self.nb4_pack_stash
+        self.nb4_pack_stash = None
+        need = compute_need_sampled_mask(input_batch) is not None
+        if need != (flat is not None):
+            raise RuntimeError(
+                f"NB4_PPACK: drafts step need={need} but stash={flat is not None}"
+            )
+        if flat is None:
+            return
+        k = self.num_speculative_steps
+        num_reqs = flat.numel() // (2 * k + 3)
+        _, _, drafts = nb4_pack_views(flat, num_reqs, k)
+        # Gather the draft rows on the main stream: the persistent table may be
+        # rewritten by a later propose() before the broadcast stream runs.
+        send = draft_tokens[input_batch.idx_mapping]
+        if send.shape != drafts.shape:
+            raise RuntimeError(
+                f"NB4_PPACK: draft shape {tuple(send.shape)} != "
+                f"{tuple(drafts.shape)}"
+            )
+        with torch.cuda.stream(self.broadcast_stream):
+            self.broadcast_stream.wait_stream(self.main_stream)
+            drafts.copy_(send)
+            send.record_stream(self.broadcast_stream)
+            torch.distributed.broadcast(
+                flat, src=self.last_rank, group=self.broadcast_group
+            )
+        self._nb4_count("packed_send")
+
+    def _nb4_receive(
+        self,
+        input_batch: InputBatch,
+        need_sampled_mask: np.ndarray,
+        gen_at_receive_np: np.ndarray,
+    ) -> bool:
+        """Non-last rank: receive() with PPACK and / or PPDEFER. The host-side
+        decisions (mask, generation snapshot, rows) are taken now as before."""
+        slot = PendingRecv(
+            None,  # type: ignore[arg-type]
+            None,  # type: ignore[arg-type]
+            None,  # type: ignore[arg-type]
+            None,  # type: ignore[arg-type]
+            input_batch.idx_mapping,
+            input_batch.idx_mapping_np,
+            need_sampled_mask,
+            gen_at_receive_np,
+            None,
+            nb4_num_reqs=input_batch.num_reqs,
+        )
+        if self.nb4_ppdefer:
+            # Keep the pp_broadcast op order: an older parked receive posts first.
+            self.flush_deferred_receive("recv")
+            self.queue[-1] = slot
+            self._nb4_count("defer_park")
+        else:
+            self.queue[-1] = self._nb4_post(slot)
+        return bool(need_sampled_mask.all())
+
+    def _nb4_post(self, slot: PendingRecv) -> PendingRecv:
+        """Post the receive(s) of a parked slot on the broadcast stream (after
+        everything queued on the main stream so far) and record its event."""
+        num_reqs = slot.nb4_num_reqs
+        k = self.num_speculative_steps
+        packed_counts = None
+        with torch.cuda.stream(self.broadcast_stream):
+            self.broadcast_stream.wait_stream(self.main_stream)
+            if self.nb4_ppack:
+                flat = torch.empty(
+                    num_reqs * (2 * k + 3), dtype=torch.int64, device=self.device
+                )
+                torch.distributed.broadcast(
+                    flat, src=self.last_rank, group=self.broadcast_group
+                )
+                event = self.broadcast_stream.record_event()
+                flat.record_stream(self.main_stream)
+                sampled_tokens, packed_counts, draft_tokens = nb4_pack_views(
+                    flat, num_reqs, k
+                )
+                num_sampled, num_rejected = packed_counts[0], packed_counts[1]
+            else:
+                # The original receive() body.
+                sampled_tokens = torch.empty(
+                    num_reqs, self.max_sample_len, dtype=torch.int64, device=self.device
+                )
+                combined = torch.empty(
+                    2, num_reqs, dtype=torch.int32, device=self.device
+                )
+                torch.distributed.broadcast(
+                    sampled_tokens, src=self.last_rank, group=self.broadcast_group
+                )
+                torch.distributed.broadcast(
+                    combined, src=self.last_rank, group=self.broadcast_group
+                )
+                draft_tokens = None
+                if self.num_speculative_steps > 0:
+                    draft_tokens = torch.empty(
+                        num_reqs,
+                        self.num_speculative_steps,
+                        dtype=torch.int64,
+                        device=self.device,
+                    )
+                    torch.distributed.broadcast(
+                        draft_tokens, src=self.last_rank, group=self.broadcast_group
+                    )
+                event = self.broadcast_stream.record_event()
+                num_sampled, num_rejected = combined.unbind(dim=0)
+                sampled_tokens.record_stream(self.main_stream)
+                combined.record_stream(self.main_stream)
+                if draft_tokens is not None:
+                    draft_tokens.record_stream(self.main_stream)
+        self._nb4_count("packed_recv" if self.nb4_ppack else "defer_post")
+        return replace(
+            slot,
+            event=event,
+            sampled_tokens=sampled_tokens,
+            num_sampled=num_sampled,
+            num_rejected=num_rejected,
+            draft_tokens=draft_tokens,
+            nb4_packed_counts=packed_counts,
+        )
+
+    def flush_deferred_receive(self, site: str = "exec") -> None:
+        """PPDEFER: post the parked receive (at most one) now. A no-op unless
+        NB4_PPDEFER=1 on a non-last rank; never posts during graph capture.
+        site: exec / empty / recv (this file) / rpc / shutdown (counter key)."""
+        if not self.nb4_ppdefer:
+            return
+        if torch.cuda.is_current_stream_capturing():
+            return
+        for i, slot in enumerate(self.queue):
+            if slot is not None and slot.event is None:
+                self.queue[i] = self._nb4_post(slot)
+                self._nb4_count("flush_" + site)
+
+    def nb4_log_final(self, site: str = "shutdown") -> None:
+        """One last 'stats' line with the counters now. A no-op unless an NB4
+        env is set."""
+        if self.nb4_any:
+            _nb4_log("stats", key="final_" + site, stats=dict(self.nb4_stats))
'''

MR_PATCH = r'''# E25 NB4 Q-defer onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5, tree 07c16a79).
# GPUModelRunner.execute_model: flush a parked PP receive after the forward launch and on empty steps
# (PPHandler.flush_deferred_receive is a no-op unless NB4_PPDEFER=1); GPUModelRunner.shutdown: flush before the
# device sync, then the last NB4 stats line (no-op unless an NB4 env is set).
# pre-patch sha256  05e889afba5dd60b6995b78aaa1f69051770ab02b638680097eccbf4e592d64c
# post-patch sha256 2c25fda60dbd5f2506ff989c29ccfe8fc0ff082fbad457fc3ae9c89ed9a185d6
--- a/vllm/v1/worker/gpu/model_runner.py
+++ b/vllm/v1/worker/gpu/model_runner.py
@@ -2206,6 +2206,9 @@
                 )
             if scheduler_output.total_num_scheduled_tokens == 0:
                 # No need to run the model.
+                if self.pp_handler is not None:
+                    # E25 NB4 PPDEFER: post a parked receive (no-op unless set).
+                    self.pp_handler.flush_deferred_receive("empty")
                 empty_output = self.kv_connector.no_forward(scheduler_output)
                 return self._merge_ec_connector_no_forward(
                     scheduler_output, empty_output
@@ -2550,6 +2553,10 @@
             ):
                 raise RuntimeError("PLE test fault: exc_after_release")
         self.kv_connector.finish_forward()
+        if self.pp_handler is not None and not dummy_run:
+            # E25 NB4 PPDEFER: post the previous step's parked receive behind this
+            # forward on the main stream (no-op unless NB4_PPDEFER=1).
+            self.pp_handler.flush_deferred_receive("exec")
 
         if self.is_last_pp_rank:
             if self.use_aux_hidden_state_outputs:
@@ -2891,6 +2898,12 @@
     def shutdown(self) -> None:
         """Release GPU tensors (model weights, KV caches, workspace) so that
         memory is reclaimable when running in the same process."""
+        _nb4_ph = getattr(self, "pp_handler", None)
+        if _nb4_ph is not None:
+            # E25 NB4 PPDEFER: post a parked receive before the device sync
+            # (no-op unless NB4_PPDEFER=1), then the last NB4 stats line.
+            _nb4_ph.flush_deferred_receive("shutdown")
+            _nb4_ph.nb4_log_final("shutdown")
         torch.accelerator.synchronize()
         # The last window of a run is the one a cadence log always
         # misses; emit it before the device state goes away.
'''

GW_PATCH = r'''# E25 NB4 Q-defer onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5, tree 07c16a79).
# Worker.profile / Worker.shutdown: post a parked PP receive first, inside the same RPC as the device sync
# (PPHandler.flush_deferred_receive is a no-op unless NB4_PPDEFER=1; nothing runs without a pp_handler).
# pre-patch sha256  a97d08ef042a53fcd139bae041fff407fce75796650d936c200c699f0d63a489
# post-patch sha256 5642b89d918b30e5b70b1067d53776e379613035b7b6e91e93cf0ee4279cfef5
--- a/vllm/v1/worker/gpu_worker.py
+++ b/vllm/v1/worker/gpu_worker.py
@@ -1695,6 +1695,15 @@
         delay_iterations: int | None = None,
         max_iterations: int | None = None,
     ):
+        # E25 NB4 PPDEFER: post a parked PP receive inside this RPC, before the
+        # profiler can synchronize the device (no-op unless NB4_PPDEFER=1).
+        _nb4_flush = getattr(
+            getattr(getattr(self, "model_runner", None), "pp_handler", None),
+            "flush_deferred_receive",
+            None,
+        )
+        if _nb4_flush is not None:
+            _nb4_flush("rpc")
         # Check if profiling is enabled
         if self.profiler_config is None or self.profiler_config.profiler is None:
             raise RuntimeError(
@@ -1951,6 +1960,18 @@
             self.model_runner.reset_lora_state()
 
     def shutdown(self) -> None:
+        # E25 NB4 PPDEFER: post a parked PP receive before anything below can
+        # synchronize the device (no-op unless NB4_PPDEFER=1).
+        _nb4_flush = getattr(
+            getattr(getattr(self, "model_runner", None), "pp_handler", None),
+            "flush_deferred_receive",
+            None,
+        )
+        if _nb4_flush is not None:
+            try:
+                _nb4_flush("rpc")
+            except Exception:  # noqa: BLE001
+                logger.exception("E25 NB4 PPDEFER: flush at Worker.shutdown failed")
         self._ple_offload_stopping = True
         gc.unfreeze()
 
'''

PATCHES = {PP_REL: (PP_PRE, PP_POST, PP_MARKER, PP_PATCH), MR_REL: (MR_PRE, MR_POST, MR_MARKER, MR_PATCH),
           GW_REL: (GW_PRE, GW_POST, GW_MARKER, GW_PATCH)}

APPLIER_SRC = r'''
def apply_unified(text, patch):
    """Strict unified-diff applier (no fuzz, no offset): every context / removed line must match at its stated line,
    else ValueError. Lines before the first hunk (comments, ---/+++ headers) are ignored. (= NB1 / NB5 / NB4 P5.)"""
    src = text.splitlines(True)
    hunks, cur = [], None
    for ln in patch.splitlines(True):
        if ln.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", ln)
            if not m:
                raise ValueError("bad hunk header: " + ln[:80])
            cur = [int(m.group(1)), int(m.group(2) or 1), []]
            hunks.append(cur)
        elif cur is not None and ln[:1] in (" ", "-", "+"):
            cur[2].append(ln)
        elif cur is not None and ln.startswith("\\"):
            raise ValueError("'No newline at end of file' hunks are not supported")
    if not hunks:
        raise ValueError("patch has no hunks")
    out, pos = [], 0
    for start, n_old, body in hunks:
        i = start - 1
        if i < pos:
            raise ValueError(f"overlapping hunk at line {start}")
        out += src[pos:i]
        old_seen = 0
        for ln in body:
            tag, s = ln[0], ln[1:]
            if tag in " -":
                if i >= len(src) or src[i] != s:
                    raise ValueError(f"context mismatch at line {i + 1}: {s[:60]!r}")
                i += 1
                old_seen += 1
            if tag in " +":
                out.append(s)
        if old_seen != n_old:
            raise ValueError(f"hunk at line {start}: {old_seen} old lines, header says {n_old}")
        pos = i
    return "".join(out + src[pos:])


def reverse_unified(patch):
    """The reverse patch: hunk headers swapped (-a,b +c,d -> -c,d +a,b), '+' and '-' lines swapped."""
    out = []
    started = False
    for ln in patch.splitlines(True):
        if ln.startswith("@@"):
            started = True
            m = re.match(r"@@ -(\d+(?:,\d+)?) \+(\d+(?:,\d+)?) @@(.*)", ln, re.S)
            if not m:
                raise ValueError("bad hunk header: " + ln[:80])
            out.append(f"@@ -{m.group(2)} +{m.group(1)} @@{m.group(3)}")
        elif started and ln[:1] == "+":
            out.append("-" + ln[1:])
        elif started and ln[:1] == "-":
            out.append("+" + ln[1:])
        else:
            out.append(ln)
    return "".join(out)
'''
exec(APPLIER_SRC)  # noqa: S102 - defines apply_unified / reverse_unified here, the notebook embeds the same text


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(t):
    return sha256_bytes(t.encode("utf-8"))


PATCH_SHA = {rel: sha256_text(p[3]) for rel, p in PATCHES.items()}


def _py_compile(py, files):
    cp = subprocess.run([py, "-c", "import py_compile, sys\nfor f in sys.argv[1:]: py_compile.compile(f, doraise=True)"]
                        + [str(f) for f in files], capture_output=True, text=True, timeout=300)
    return {"rc": cp.returncode, "err": cp.stderr[-600:]}


def _drop_pyc(f):
    d = f.parent / "__pycache__"
    if d.is_dir():
        for pyc in d.glob(f.stem + ".*.pyc"):
            pyc.unlink()


def _state(rel, h, data):
    pre, post, marker, _ = PATCHES[rel]
    if h == post and marker.encode() in data:
        return "applied"
    if h == pre:
        return "not_applied"
    return "foreign"


def verify_ppack_patch(tree):
    """Read-only: 'applied' (every file post-sha + marker), 'not_applied' (every file pre-sha), else 'foreign' / 'mixed'."""
    tree = Path(tree)
    files = {}
    for rel in PATCHES:
        data = (tree / rel).read_bytes()
        h = sha256_bytes(data)
        files[rel] = {"sha": h, "state": _state(rel, h, data)}
    states = {v["state"] for v in files.values()}
    state = states.pop() if len(states) == 1 else ("foreign" if "foreign" in states else "mixed")
    return {"state": state, "files": files}


def _write_atomic(f, data):
    tmp = f.with_name(f.name + ".nb4tmp")
    tmp.write_bytes(data)
    os.replace(tmp, f)


def apply_ppack_patch(tree, py=None):
    """Patch pp_utils.py + model_runner.py + gpu_worker.py of the installed tree. Every file must be at Merlin ad42e37's
    pre-sha (or all already applied), else refuse. All new texts are built and sha-checked before anything is written;
    then the post-sha, the marker and py_compile (with `py`, the Merlin venv python) are verified. Any miss raises."""
    tree = Path(tree)
    py = py or sys.executable
    v = verify_ppack_patch(tree)
    rec = {"tree": str(tree), "before": v, "patch_sha": PATCH_SHA}
    if v["state"] == "applied":
        rec["applied"] = "already_applied"
    elif v["state"] != "not_applied":
        raise RuntimeError("PATCH_PPACK_PRE_SHA_MISMATCH: " + json.dumps(v)[:900])
    else:
        new = {}
        for rel, (pre, post, marker, patch) in PATCHES.items():
            raw = (tree / rel).read_bytes()
            out = apply_unified(raw.decode("utf-8"), patch).encode("utf-8")  # noqa: F821 - from APPLIER_SRC
            if sha256_bytes(out) != post:
                raise RuntimeError(f"PATCH_PPACK_POST_SHA_MISMATCH {rel}: {sha256_bytes(out)}")
            new[rel] = out
        for rel, out in new.items():
            _write_atomic(tree / rel, out)
        rec["applied"] = "applied"
    for rel in PATCHES:
        _drop_pyc(tree / rel)
    rec["after"] = verify_ppack_patch(tree)
    rec["py_compile"] = _py_compile(py, [tree / rel for rel in PATCHES])
    if rec["after"]["state"] != "applied" or rec["py_compile"]["rc"]:
        raise RuntimeError("PATCH_PPACK_VERIFY_FAILED: " + json.dumps(
            {k: rec.get(k) for k in ("after", "py_compile")})[:900])
    rec["ok"] = True
    return rec


def revert_ppack_patch(tree, py=None):
    """Back to the exact Merlin files (reverse patches, sha-checked == pre). Refuses unless every file is applied
    (post-sha) or every file already original."""
    tree = Path(tree)
    py = py or sys.executable
    v = verify_ppack_patch(tree)
    rec = {"tree": str(tree), "before": v}
    if v["state"] == "not_applied":
        rec["reverted"] = "not_applied"
    elif v["state"] != "applied":
        raise RuntimeError("REVERT_PPACK_SHA_MISMATCH: " + json.dumps(v)[:900])
    else:
        old = {}
        for rel, (pre, post, marker, patch) in PATCHES.items():
            raw = (tree / rel).read_bytes()
            out = apply_unified(raw.decode("utf-8"), reverse_unified(patch)).encode("utf-8")  # noqa: F821
            if sha256_bytes(out) != pre:
                raise RuntimeError(f"REVERT_PPACK_RESULT_MISMATCH {rel}: {sha256_bytes(out)}")
            old[rel] = out
        for rel, out in old.items():
            _write_atomic(tree / rel, out)
        rec["reverted"] = "reverted"
    for rel in PATCHES:
        _drop_pyc(tree / rel)
    rec["after"] = verify_ppack_patch(tree)
    rec["py_compile"] = _py_compile(py, [tree / rel for rel in PATCHES])
    if rec["after"]["state"] != "not_applied" or rec["py_compile"]["rc"]:
        raise RuntimeError("REVERT_PPACK_VERIFY_FAILED: " + json.dumps(rec)[:900])
    rec["ok"] = True
    return rec


def server_env(arm):
    """Env added to the server for one arm on top of the base env: 'Q-pack', 'Q-defer' or 'Q-pack+defer'.
    The base / other arms run the patched tree with neither variable set (or the reverted tree)."""
    envs = {"Q-pack": {ENV_PACK: "1"}, "Q-defer": {ENV_DEFER: "1"}, "Q-pack+defer": {ENV_PACK: "1", ENV_DEFER: "1"}}
    if arm not in envs:
        raise ValueError(f"unknown arm {arm!r} (expected one of {sorted(envs)})")
    return dict(envs[arm])


_EV_RE = re.compile(r"NB4_PPQ (\{.*\})\s*$")


def parse_log(text):
    """Every 'NB4_PPQ {json}' line of a server log -> list of dicts (bad JSON lines are returned as {'bad': line})."""
    out = []
    for ln in text.splitlines():
        m = _EV_RE.search(ln)
        if not m:
            continue
        try:
            out.append(json.loads(m.group(1)))
        except ValueError:
            out.append({"bad": ln[-200:]})
    return out


DEFER_EXEC_MIN = 0.9         # R1: flush_exec >= 0.9 * defer_park on every non-last rank
DEFER_FALLBACK_MAX = 0.05    # R1: flush_recv + flush_consume <= 0.05 * defer_park on every non-last rank


def readback_label(log_text, arm, pp_size=2, tp_size=2):
    """Read-back gate of a Q arm: every PP worker logged init with the expected flags, and the engaged counters
    moved ('stats' lines). Returns (label, detail); label ENGAGED / ENGAGED_FALLBACK / NOT_ENGAGED / NOT_INTEGRATED.
    Q-defer (R1): the LAST 'stats' line of every non-last rank (matched by the pid of its init line) must show
    flush_exec >= DEFER_EXEC_MIN * defer_park and flush_recv + flush_consume <= DEFER_FALLBACK_MAX * defer_park, else
    ENGAGED_FALLBACK (the deferral collapsed to post-at-consume: a different timing, never a clean engage). A non-last
    rank without a 'stats' line, with defer_park 0, or init lines without distinct pids -> NOT_ENGAGED."""
    want = server_env(arm)
    ev = parse_log(log_text)
    inits = [e for e in ev if e.get("event") == "init"]
    n_ranks = pp_size * tp_size
    detail = {"inits": len(inits), "expect": n_ranks}
    if len(inits) != n_ranks:
        return "NOT_INTEGRATED", detail
    last = [e for e in inits if e.get("is_last_rank")]
    first = [e for e in inits if not e.get("is_last_rank")]
    detail.update(last=len(last), nonlast=len(first))
    ok = len(last) == tp_size and len(first) == n_ranks - tp_size
    pack = want.get(ENV_PACK) == "1"
    defer = want.get(ENV_DEFER) == "1"
    ok = ok and all(bool(e.get("ppack")) == pack for e in inits)
    ok = ok and all(bool(e.get("ppdefer")) == defer for e in first) and not any(e.get("ppdefer") for e in last)
    if not ok:
        return "NOT_INTEGRATED", detail
    stats = [e for e in ev if e.get("event") == "stats"]
    keys = {e.get("key") for e in stats}
    need = set()
    if pack:
        need |= {"packed_send", "packed_recv"}
    if defer:
        need |= {"defer_park", "flush_exec"}
    detail["stat_keys"] = sorted(k for k in keys if k)
    if not need <= keys:
        return "NOT_ENGAGED", detail
    all_pids = [e.get("pid") for e in inits]
    if None in all_pids or len(set(all_pids)) != len(all_pids):
        detail["why"] = f"init lines without distinct pids: {all_pids}"
        return "NOT_ENGAGED", detail
    if pack:
        # GPT audit 2026-11-01: every rank must show its own packed counter (last ranks send, non-last ranks receive)
        pk_last = {}
        for e in stats:
            if e.get("pid") in all_pids and isinstance(e.get("stats"), dict):
                pk_last[e["pid"]] = e["stats"]
        per_pk, bad = {}, []
        for e in inits:
            k = "packed_send" if e.get("is_last_rank") else "packed_recv"
            n = int((pk_last.get(e["pid"]) or {}).get(k) or 0)
            per_pk[str(e["pid"])] = {k: n}
            if n <= 0:
                bad.append(f"{e['pid']}:{k}")
        detail["pack_per_rank"] = per_pk
        if bad:
            detail["why"] = f"rank(s) without their packed counter: {bad}"
            return "NOT_ENGAGED", detail
    if not defer:
        return "ENGAGED", detail
    pids = [e.get("pid") for e in first]
    if None in pids or len(set(pids)) != len(pids):
        detail["why"] = f"non-last init lines without distinct pids: {pids}"
        return "NOT_ENGAGED", detail
    last_stats = {}
    for e in stats:                                   # log order: the last line per pid wins
        if e.get("pid") in pids and isinstance(e.get("stats"), dict):
            last_stats[e["pid"]] = e["stats"]
    per, missing, fallback = {}, [], False
    for pid in pids:
        st = last_stats.get(pid)
        if st is None:
            missing.append(pid)
            continue
        c = {k: int(st.get(k) or 0) for k in ("defer_park", "flush_exec", "flush_empty", "flush_recv", "flush_consume",
                                              "flush_rpc", "flush_shutdown")}
        park = c["defer_park"]
        c["exec_share"] = round(c["flush_exec"] / park, 4) if park else None
        c["fallback_share"] = round((c["flush_recv"] + c["flush_consume"]) / park, 4) if park else None
        c["clean"] = bool(park > 0 and c["flush_exec"] >= DEFER_EXEC_MIN * park
                          and c["flush_recv"] + c["flush_consume"] <= DEFER_FALLBACK_MAX * park)
        if park <= 0:
            missing.append(pid)
        fallback |= not c["clean"]
        per[str(pid)] = c
    detail["defer_per_rank"] = per
    detail["defer_rule"] = f"flush_exec >= {DEFER_EXEC_MIN} * defer_park and flush_recv + flush_consume <= {DEFER_FALLBACK_MAX} * defer_park"
    if missing:
        detail["why"] = f"non-last rank(s) without a stats line / with defer_park 0: {missing}"
        return "NOT_ENGAGED", detail
    return ("ENGAGED_FALLBACK" if fallback else "ENGAGED"), detail


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-tree")
    ap.add_argument("--shas", action="store_true")
    a = ap.parse_args()
    if a.check_tree:
        print(json.dumps(verify_ppack_patch(a.check_tree), indent=1))
    if a.shas or not a.check_tree:
        print(json.dumps({"PP_PRE": PP_PRE, "PP_POST": PP_POST, "MR_PRE": MR_PRE, "MR_POST": MR_POST,
                          "GW_PRE": GW_PRE, "GW_POST": GW_POST, "PATCH_SHA": PATCH_SHA}, indent=1))
