# Design: a generated-tail prefix-cache checkpoint for Duck multi-turn on Merlin (2026-11-01)

- **Scope:** CPU and source work only. No GPU, no Kaggle, no git.
- **Source read:** Merlin v3.0.0 (vLLM 0.32 fork) at `D:/ARC AGI 2/_scratch/e23/p2/merlin-src/vllm/`. All paths below are relative to that folder.
- **Run config:** taken from the K100Q server log, `_scratch/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x/engine/logs/server_merlin_fp8_mtp_pin508.log`:
  - TP2 × PP2, MTP K3, `block_size 4096`, `prefix_match_unit 64`, `mamba_cache_mode align`, `use_replayssm` (the RecoverSSM path is taken), `max_num_batched_tokens 1024`, `max_num_seqs 21`;
  - `prefix_cache_retention_interval` left at its default of 0;
  - boot lines: "RecoverSSM + pipeline parallel: Mamba prefix checkpoints are published for prompt positions only (3 groups)", "retention: on", "grace pins up to 16 turns", "GPU KV cache size 1,652,197 tokens".
- **Measurement basis:** `Experiment Results/prep_0gpu/duck_timeline/REPORT_duck_idle_and_reprefill_2026-11-01.md` and its `out_analysis2/reprefill_per_turn.csv` (677 later turns).
- **Prototype and CPU tests:** `D:/ARC AGI 2/_scratch/gentail_ckpt/` (`gentail_logic.py`, `test_gentail_logic.py`; 11 / 11 pass, see section 8).

---

## 0. Answer in one table

| Option | What it does | Share of the perfect 2.43 M re-prefill saved (K100Q replay) | Run-time value (6.0-8.2 % × share) | Engine change | Risk |
|---|---|---|---|---|---|
| **B: fenced decode grid checkpoint** | At finish, publish the newest decode block-boundary state (a multiple of 4096) that the RecoverSSM commit **already writes**, plus an attention partial key at the generated end. Publish only after a PP fence. | **1,702,336 tokens = 70.0 %** (helps 309 / 677 turns) | **≈ 4.2-5.7 %** | Scheduler and KV manager only (≈ 150-250 lines). **No kernel change.** | Medium |
| **C: 64-unit tail ring** | Option B plus a ring of R tail pages per Mamba group, written at every 64-unit crossing by the commit kernels. At finish, publish the page at the eagle-adjusted end. | **2,429,440 = 99.96 %** with R = 3 (simulated ring coverage 99.9 %; R = 2 gives 96.9 %) | ≈ 6.0-8.2 % | Option B plus 4 Triton kernels (GDN plan, GDN state commit, conv compaction, PLE commit), worker metadata, MambaSpec sizing (≈ 500-800 lines) | High; memory R × 3 pool blocks per running request |
| **D: session park (client-directed)** | Duck sends `session_id` + an opt-in flag. A stopped turn is parked as a resumable request (the streaming-input machinery that already exists). The next turn's prompt is checked against the parked tokens and resumes at exactly S, with no alignment. | **2,494,563 = 102.6 %** (also removes the 64-127 alignment tail) | ≈ 6.2-8.4 % | API server routing + output processor + scheduler park / TTL / evict + guards (≈ 300-500 lines) | Medium-high (the path is untested with Mamba + MTP + PP) |

**Recommendation:**
1. **Now (free):** turn on `return_token_ids` in the next Duck GPU run. That measures gate 1 (token-ID identity) for every option, with no engine change.
2. **Stage 1:** build option B.
3. **Stage 2:** build C or D, depending on gate 1 and on pool pressure. D needs no extra state pages; C needs R × 3 pages per running request in a pool that peaked at 99.4 %.

---

## 1. Why the hit stops 64-127 tokens before the previous prompt's end

**Short answer: both groups stop it, but the Mamba group is the one that binds.** Fixing only the attention side would gain nothing. Fixing only the Mamba side would also gain nothing, because the eagle drop needs an attention key one unit past the Mamba checkpoint.

### 1.1 How block hashes cover generated tokens (refutes half of the premise)

- `Request.append_output_token_ids` calls `update_block_hashes()` for every output token (`v1/request.py:272-288`).
- The hasher chains a 64-token hash over prompt **and** output tokens (`v1/core/kv_cache_utils.py:942-1001`).
- So every generated 64-token unit has a hash.

### 1.2 Attention groups

- The scheduler caches blocks every step: `allocate_slots` caps the count at `request.num_tokens` (`v1/core/kv_cache_manager.py:703-707`), and the coordinator subtracts `num_reprefillable_tokens` (`v1/core/kv_cache_coordinator.py:368-374, 810-825`).
- `FullAttentionManager.cache_blocks` → `SingleTypeKVCacheManager.cache_blocks` → `BlockPool.cache_full_blocks` (`single_type_kv_cache_manager.py:496-553, 870-887`; `block_pool.py:229-302`).
- So **every full 4096-token attention block of generated tokens is hashed during decode.** This confirms the premise's "attention blocks of generated tokens may already be hashed".
- The sub-block partial key is registered **only at the prompt tail**. `_cache_partial_tail_block` keys `floor(P/64)·64` (eagle `token_limit = P`) and nothing else (`single_type_kv_cache_manager.py:889-924`).
- Attention lookup has two phases (`single_type_kv_cache_manager.py:822-868`):
  - phase 1 walks full blocks from the start;
  - phase 2 probes the partial keys inside the first non-full block.
- So over generated tokens the attention hit ends at the **last full 4096 block of the generation**. It never ends at the generated end.

### 1.3 Mamba groups (GDN × 2 + PLE, the "3 groups")

1. Under RecoverSSM + PP, `configure_recoverssm_pp_checkpoints` sets `publish_decode_checkpoints = False` on every MambaManager (`v1/core/sched/scheduler.py:88-110`, called at `:383-391`).
   - `MambaManager.cache_blocks` then caps `num_tokens` at `num_prompt_tokens − 1` (`single_type_kv_cache_manager.py:2323-2330`).
   - Reason given in the source (`single_type_kv_cache_manager.py:1613-1619`): a non-last PP stage commits a verify window's state `pp_size` steps after the forward, and never if the request is cancelled in between.
2. Even without PP, decode states would not be keyed:
   - With `prefix_cache_retention_interval = 0` (the default here, `config/cache.py:160`), `MambaManager.reachable_block_mask` keeps only the `replay_boundaries` (`single_type_kv_cache_manager.py:1868-1925`).
   - `replay_boundaries` are derived from `num_prompt_tokens` only (`kv_cache_coordinator.py:67-88`).
   - `_cache_partial_tail_block` keys only `latest_prompt_hash_boundary` or a junction inside the prompt (`single_type_kv_cache_manager.py:2453-2486`).
3. Prompt checkpoint position: `get_mamba_prefill_checkpoint_position(P, 64, drop_eagle=True)` = `floor(P/64)·64 − 64` (`v1/kv_cache_interface.py:1111-1123`).
   - The drop is on because the coordinator sets `drop_eagle_checkpoint_block` for MTP (`kv_cache_coordinator.py:186-190`).
   - The scheduler stops a prefill chunk exactly there (`scheduler.py:583-615`), so the state at that position exists.

### 1.4 The fixed point

- `HybridKVCacheCoordinator.find_longest_cache_hit` (`kv_cache_coordinator.py:827-962`) works in two steps:
  1. it drops one hash unit from the attention hit (eagle);
  2. it lets Mamba search only up to that reduced length (`single_type_kv_cache_manager.py:1813-1839`, upstream #53912).
- The newest Mamba state at or below that length is the prompt checkpoint at `floor(P/64)·64 − 64`. It is 64-127 tokens before P.
- **Proof against the data:** `current_prompt_hit(P_prev)` in the prototype reproduces `cached_tokens` **exactly in 677 / 677 later turns** (`test_current_hit_matches_k100q_exactly`).

---

## 2. Request finish: where the final state lives and when it is valid

### 2.1 Where the state is

- **Running state:** align mode keeps it in the block of the **last committed token**. The commit plan sets `final_state_col = (num_computed + commit_len − 1) // mamba_block_size` (`model_executor/layers/mamba/ops/recoverssm.py:80-92`). The align postprocess kernel sets the same column (`v1/worker/gpu/model_states/recoverssm.py:200-212`).
- **Its position S** is the exact committed count. It is an arbitrary integer: **not** a multiple of 64, and not block-floored. A 64-unit hit needs the state at a multiple of 64, and the eagle rule needs it at least one unit behind the attention key. **The final running state therefore can never be published as is.**
- **Block-floored state:** it exists for free. When a verify window crosses a multiple of 4096, the plan sets a boundary target (block column `next_boundary/bs − 1`, `recoverssm.py ops:93-102`). The GDN commit kernel stores the exact state after `boundary_recovery_len` tokens into that block (`model_executor/layers/mamba/gdn/recoverssm_gdn.py:348-455`, masked store at `:436-442`). Conv compaction does the same (`:657-676`), and so does PLE (`v1/attention/backends/ple_recoverssm.py:10`).
- **That boundary block is then retired:** `remove_skipped_blocks` → `_retire_state_blocks` (`single_type_kv_cache_manager.py:1947-1978, 1673-1702`). With retention on, the newest **hashed** retired block is kept as the request's checkpoint. Decode boundary blocks are never hashed, so they go back to the pool.

### 2.2 MTP lookahead and reprefillable tokens

- `num_reprefillable_tokens = num_prefill_lookahead − 1` (`kv_cache_coordinator.py:119`).
- For multi-module MTP K3 that is 3 − 1 = 2 (`config/vllm.py:674-694`); single-module MTP gives 0.
- The last `num_reprefillable` tokens are never cached (`kv_cache_coordinator.py:368-374, 815-824`).
- On top of that, the eagle drop removes one 64 unit (`kv_cache_coordinator.py:137-157, 896-906`). It exists because MTP draft-layer KV at the last positions depends on the writer's own continuation.
- **Stop truncation:** if the stop token was an accepted draft and later drafts were also accepted, `_update_request_with_output` trims the output tokens but not `num_computed_tokens` (`scheduler.py:2800-2817`). The committed state has then consumed up to 2 tokens that are not in the sequence.
- So the valid prefix is `min(num_computed, num_tokens) − num_reprefillable`. Option C's target is `floor(valid/64)·64 − 64` (prototype `generated_tail_positions`; `test_tail_positions_stop_was_an_accepted_draft`).

### 2.3 RecoverSSM commit timing

- The verify forward does **not** advance the committed state. It reads the checkpoint and writes per-token replay records (`recoverssm_gdn.py:1-15`; `gdn_recoverssm.py:1-9`).
- The state advances only in the post-sampling commit (`v1/worker/gpu/model_states/recoverssm.py:114-193`, called from `mamba_hybrid.py:513`).
- So the state at any position is valid only after the forward **and** the commit of the step that crossed it.

### 2.4 Under PP2

- **PP1 (the last rank)** commits in the same step, after sampling.
- **PP0** detaches the step's commit (`recoverssm.py:56-77`) into the PP ring. The ring has depth = PP world size, is pre-seeded with `None`, and pops one entry per `execute_model` call (`v1/worker/gpu/pp_utils.py:81-90, 130-160`). So step T's commit runs at the **start of PP0's call T + pp_size**.
- Order inside that call (`model_runner.py:2196-2201`, copies at `:1491-1497`): `update_pp_decode_requests` (deferred commit) → `finish_requests` → `add_requests` → `update_requests` (zeroing, then CoW copies) → forward. So the commit always precedes that call's own frees, copies and reads.
- **The commit is skipped** for a request whose worker slot was freed before call T + pp_size. This is the `req_idx_gen` check (`pp_utils.py:142-147`). A finished request is freed at the call that carries its id in `finished_req_ids` (`model_runner.py:1366-1378`).
- **Consequence:**
  - Busy engine (a second microbatch in flight): the finish reaches the worker at call L + 2, after the commit of step L. Everything is committed.
  - Idle engine (nothing behind step L): the finish can ride in call L + 1, and **the commit of the request's last step L is lost on PP0**.
  - The toy model shows both cases: `test_pp_fence_model_busy_engine` and `test_pp_fence_model_idle_engine_skips_last_commit`.
- **Two rules follow.** Both are necessary, and together they are sufficient:
  1. **Producer rule:** publish only states produced by a step **strictly before** the request's last scheduled step (`state_is_pp_safe`). For T ≤ L_last − 1, the slot is freed no earlier than call L_last + 1 ≥ T + 2, and in that call the pop runs before the free. This also holds for aborts.
  2. **Fence rule:** publish only once `processed_step_seq ≥ T + pp_size` (`pp_fence_seq`).
     - The last rank finishing step T + pp_size implies PP0 ran that call.
     - `sched_step_seq` and `processed_step_seq` count only non-empty steps (`scheduler.py:1845-1847, 2352-2355`). An empty execute would only pop the entry earlier.
     - Here they advance because `defer_block_free` is on (CircularBufferSpec + `max_concurrent_batches > 1`, `scheduler.py:224-233`). The patch must assert this, or keep its own counter.
- **Option C:** the 64-unit target is at least 66 tokens behind the end, i.e. at least 15 windows before the last step, so the producer rule never costs it a hit (`test_target_never_produced_in_final_window`).
- **Option B:** a grid crossing inside the final window is simply skipped.
- **Idle engine:** the fence cannot pass until two more non-empty steps run, so the next turn misses, which is correct. Its first step pops the older ring entry, which is exactly why a publish without the fence would be unsafe there.
- **Rolling publication** (during decode, before finish) needs one more condition: the request must not have been freed between T and T + pp_size. The doc therefore specifies publication **at finish only**.

---

## 3. The minimal correct design

### 3.1 Option B (stage 1): fenced decode-grid checkpoint (no kernel change)

1. **Remember the newest decode grid state per Mamba group** (`MambaManager`, near `_retire_state_blocks` at `:1673`).
   - In `Scheduler.update_from_output`, after the rejection adjustment, compute `b = (num_computed_after // 4096) · 4096` if it is greater than `num_computed_before`.
   - That means step T committed the boundary fold into block column `b/4096 − 1`.
   - Record `(position b, block, T)` in a new `MambaManager._pending_grid[req]`. It holds one entry and supersedes the previous one.
   - In `_retire_state_blocks` / `_remove_blocks_in_range`, when the retired block is the pending one, **keep its table ref as the hold** instead of freeing it. This uses the same mechanism as `_retained_checkpoints`: one per request, and the older one is released.
   - This also stops PP0's lagging fold from writing into a block that was already freed.
2. **At finish** (`KVCacheManager.free` / `pop_blocks_for_free`, `kv_cache_manager.py:718-733, 1064-1086`, before `_take_grace_pin`):
   - for each Mamba group, take the pending grid entry if `T < request.last_sched_seq` (producer rule) and `b ≤ floor(valid/64)·64 − 64`;
   - for each attention (and MTP draft) group, register `cache_partial_block` at `A = floor(valid/64)·64` on block `A // 4096` (`block_pool.py:452-566`). Attention KV is append-only and written in the forward, so it needs no fence.
   - With **A − 64 ≥ b**, the Mamba search finds `b`. Without the attention key, the attention hit ends at the last full block F and `b = F` would be dropped by eagle (see 1.2), so the attention key is required.
3. **Fence:**
   - Push `(T + pp_size, [request id, b, held blocks, the request's block-hash list])` into a new `pending_tail_publishes` queue (`PendingPublishQueue`).
   - Drain it next to `_drain_deferred_frees` (`scheduler.py:2352-2355, 3097-3111`).
   - On drain:
     - key the Mamba block at `b` (a block-end key; the same key `_block_aligned_prompt_tail` already uses, `:2378-2392`);
     - hand the block to the request's grace pin if one is still held (`kv_cache_manager.py:763-792`), else free it as a hashed block (evict-last LRU);
     - drop the queue's extra ref.
   - In this config the fence is normally already past at finish: the request's last step L_last ≥ T + 1, and the finish is processed after L_last.
4. **Gating flag:** `--enable-mamba-generated-tail-checkpoint`, off by default. One info line at boot, plus counters:
   - published;
   - skipped (final-step producer);
   - fence-deferred;
   - hit-through-tail (`cached_tokens > P_prev`).
5. **Not needed:** reader-side changes. A hit at a block boundary is the standard align-mode hit: the reader reads block `b/4096 − 1` and writes its own new running block.

### 3.2 Option C (stage 2): 64-unit tail ring

- Same as B, with one difference: the Mamba entry is the ring page that holds the state at `floor(valid/64)·64 − 64`.
- **Memory:** add R tail pages per decoding request per Mamba group. Size MambaSpec (`kv_cache_interface.py:1080-1094`) and admission (`get_num_blocks_to_allocate`, `:2013-2115`) for them.
- **Worker plumbing:** pass the ring page ids with the align block table (`gdn_recoverssm.py` builder, `mamba_utils.py`).
- **Plan kernel** (`ops/recoverssm.py:28-103`):
  - a second target: `next_unit = (nc // 64 + 1)·64`; crosses if `nc + commit_len ≥ next_unit`; `unit_len = next_unit − nc`; `unit_idx = ring[(next_unit/64) % R]`;
  - a window has at most 4 tokens < 64, so there is at most one crossing per window.
- **GDN state kernel:** a second masked store `t + 1 == unit_len`, next to `recoverssm_gdn.py:436-442`.
- **Conv compaction and PLE window:** the same second destination.
- **At finish:** publish the ring page as a sub-block partial key. The existing prompt-tail partial-hit / CoW reader path (`single_type_kv_cache_manager.py:2193-2250, 2510-2520`) already handles such a hit, so the reader side is unchanged.
- **Ring size:** R = 3 covers 99.9 % of finishes; R = 2 covers 96.9 % (`test_ring_coverage`).
- **Memory cost:** R × 3 groups = 9 pool blocks per running request at R = 3, about 189 blocks at 21 streams. Mamba pages are padded to the attention page size (log: "Padding mamba page size by 21.01 %"). The prompt-tail CoW hold today is 1 page per group per request.

### 3.3 Memory and eviction (both options)

- **Steady state:** option B adds **no extra pages**. The newest grid state supersedes the prompt-tail hold, and the superseded block becomes an ordinary cached free block (`_retire_state_blocks` supersede rule, `:1693-1700`). Between finish and fence it holds 1 block per group per finished request, normally zero steps.
- **Eviction:** do **not** use evict-first junction entries (`evict_groups.junctions.publish`). At a 99.4 % pool peak an evict-first entry would be the first block taken during the 0.3-0.5 s tool gap. Instead, put the published block under the existing grace pin:
  - it is held until the next turn is admitted (`_release_grace_for_hit` releases it once the hit reaches within 2 units of the prompt, `kv_cache_manager.py:805-835`);
  - it is yielded first under allocation pressure (`yield_pinned_hits`, `:1016-1042`);
  - after that it ages LRU.
- **Raise `MAX_GRACE_PINS` (16, `kv_cache_manager.py:735`) to `max_num_seqs` (21).** With 21 workers, more than 16 turns can be between turns at once.

### 3.4 Correctness invariants the patch must keep

1. A key is registered only for a state that equals the cold-prefill state at that exact position, on every TP and PP rank (producer rule + fence).
2. The position is a multiple of 64 and at most `floor(valid/64)·64 − 64`, with `valid` capped by the true token count.
3. The token IDs are guaranteed by the chained hash (a hit implies identical IDs for the whole prefix). A re-render mismatch costs gain, never correctness.
4. A published block is never written again by its producer: it is held out of the table, and the producer is finished.
5. Drafter KV at the tail is never served: the eagle drop and reprefillable margin are kept unchanged.

---

## 4. The four gates and how the test proves them

1. **Token-ID identity of the replayed prefix.**
   - Use `return_token_ids: true` (`entrypoints/openai/chat_completion/protocol.py:423`; output at `serving.py:1123, 1200`).
   - For each turn k, check `prompt_ids_k[:P_{k−1} + G_{k−1} − 1] == prompt_ids_{k−1} + output_ids_{k−1}[:−1]`. Record the first mismatch index; its distribution is the achievable gain.
   - Only tokens up to the checkpoint (at least 66 before the end) must match. The tool-call tail may differ.
   - This needs no engine change and can ride on any Duck GPU run.
2. **Checkpoint position == prefix length.**
   - Per turn: `usage.prompt_tokens_details.cached_tokens` must equal the predicted hit: `grid_checkpoint(...)` for B, `generated_tail_positions(...).hit` for C, `min(S, P − 1)` for D.
   - Engine counters: published position == looked-up position.
3. **KV + recurrent-state numerical equivalence vs a cold prefill.**
   - In-process `LLM(...)` with the server config.
   - Turn chain with greedy decoding. For each turn, prefill the same prefix cold under a different `cache_salt`.
   - Use `llm.collective_rpc` to read on **all 4 workers**: the published Mamba block (GDN SSM state, conv state, PLE window), and the attention KV of the last unit before the hit, against the cold request's blocks at the same position.
   - Pass: GDN/PLE state bit-identical, or max-abs at the fp32 accumulation noise of a cold-vs-chunked prefill (measure that noise as the baseline first); attention KV bit-identical (same kernel and positions).
   - API-level gate as well: the first 64 warm vs cold greedy tokens are identical, and the logprob max-abs is within the E19 parity tolerance. E19 found MTP + prefix-OFF itself flips 4-12, so compare like with like: warm vs cold, both prefix ON.
4. **PP0 / PP1 fence.**
   - Run the chain twice:
     - (a) busy, with 2 background filler streams so steps keep flowing;
     - (b) idle, single stream.
   - Expect a tail hit in (a) and a fallback to the prompt checkpoint in (b), with gate 3 equality in both.
   - Abort test: abort a request in the step right after a grid crossing. Expect no publish (counter "skipped final-step producer").
   - Inject a negative control: a debug env that disables the fence. On PP0 it should produce a measurable state diff in (b), which proves the test can see the hazard.

---

## 5. Client-directed alternatives (Duck knows each turn boundary)

- **`session_id`** (`chat_completion/protocol.py:406`; carried to the scheduler `Request.session_id`, `v1/request.py:197`) and **`vllm_xargs`** (`protocol.py:511`, passed as `SamplingParams.extra_args` at `:746`) already reach the engine. A per-request opt-in (`vllm_xargs: {"tail_checkpoint": 1}`) costs about 10 lines.
  - It does **not** make B or C simpler. The hard part is engine-internal: getting an aligned state at the end of decode under PP.
  - It only limits the memory of C to opted-in requests. Duck cannot know in advance whether a turn is its last, so it would opt in on every turn anyway.
- **`kv_hints`** (`v1/kv_hints/protocol.py`) reaches `Request.kv_hints` (`v1/request.py:198`). Only the offload connector reads it (`v1/kv_offload/tiering/kvcr/manager.py:582`), and the chat API does not expose it. It is not a shortcut.
- **Option D, session park:** the cleanest client-directed path. The scheduler already has resumable streaming-input sessions:
  - `_handle_stopped_request` parks a stopped request in `WAITING_FOR_STREAMING_REQ` and keeps its blocks and worker slot (`scheduler.py:2772-2798`);
  - `_update_request_as_session` appends the next chunk and keeps the computed outputs (`:1957-1997`);
  - `add_request` routes an update to the existing id (`:2924-2943`).
  - Because the worker slot stays live, PP0's deferred commit is never skipped. The state at S is exact, so **no alignment, no ring and no fence** are needed. It saves 102.6 % of (a), because the 64-127 tail also goes.
  - Today the path is only reachable through `AsyncLLM.generate(AsyncGenerator[StreamingInput])` (`v1/engine/async_llm.py:580-676`) and the realtime ASR websocket. It rejects `stop` strings (`:678-692`).
  - **Work needed:**
    1. In the chat server, on a request with `session_id` + opt-in: if a parked session exists and `new_prompt_ids[:num_computed]` equals the parked tokens, send the delta as a session update; otherwise abort the parked session and run normally.
    2. The output processor returns the turn as finished while the engine request stays parked (`output_processor.py:299, 652`).
    3. Park TTL and pressure eviction, so a parked session can be freed into the grace-pin path when the pool needs blocks or when Duck never sends a next turn.
    4. Guard: do not park if `num_computed > num_tokens − 1` (the stop-as-accepted-draft case in 2.2; `_update_request_as_session` would otherwise keep a state that consumed discarded drafts).
  - **Risk:** this path has never run with Mamba align + RecoverSSM + MTP + PP. Gates 3 and 4 apply unchanged.
- **Token identity by construction:** Duck could send `prompt_ids_{k−1} + output_ids_{k−1} + delta_ids` through `/v1/completions` with token-ID prompts, using `return_token_ids`. That makes gate 1 hold by construction for B, C or D. The cost is that tool-call and reasoning parsing would have to move client-side.
- **Upstream RFC #55697** (application-directed checkpoints, open, PRs #55873 / #55875 / #55876):
  - a `<|mamba_checkpoint|>` marker in the **prompt**, stripped before hashing, gives an aligned producer checkpoint and consumer deferral;
  - it covers prompt positions only. A marker cannot be placed inside generated tokens, so it does not address this case.
- **Upstream PR #52789** (merged): internal prefill checkpoints, saving the boundary state inside one forward. At merge it was FlashKDA-only with spec decoding excluded.
- **Upstream draft PR #53803**:
  - "retain align-mode mamba grid and decode-end checkpoints" (rolling registration of the newest frozen boundary block during decode) is the same idea as option B;
  - its author reports that the rolling decode-end part "showed no effect" on main for their append benchmark, and that the grid only materializes where it meets a chunk end;
  - in Merlin the RecoverSSM commit itself writes the decode grid state (2.1), so that limitation does not apply here;
  - the PR has no PP fence.

---

## 6. Effort, risk and the cheapest GPU test

| | Effort | Main risks | Gain (K100Q) |
|---|---|---|---|
| B | 1-2 days code + CPU unit tests on the real classes (BlockPool / MambaManager / coordinator are pure Python) + 1 GPU commit | Holding the grid block through retirement; grace-pin interaction; the fence counter when `defer_block_free` is off | 70 % of 2.43 M → ≈ 4.2-5.7 % of the run (+16-23 tok/s on 385) |
| C | +3-5 days (4 kernels, worker plumbing, spec sizing) | Kernel correctness on both PP ranks; pool pressure (+9 blocks per stream) | 99.96 % → ≈ 6.0-8.2 % |
| D | 2-4 days (server routing, park / TTL / evict, guards) | Untested streaming path on hybrid + MTP + PP; parked KV under pressure | 102.6 % → ≈ 6.2-8.4 % |

### Cheapest GPU test (one ~30-40 min Kaggle GPU commit with fail-fast gates; needs an explicit go)

1. **Boot** the same config with the flag on.
2. **Chain:** replay one recorded 10-turn Duck chain from K100Q, using the saved messages with greedy decoding. Pick a chain with at least 3 turns whose generation is over 4 k tokens, so option B is exercised.
3. **Two passes:**
   - **cold:** every turn with a fresh `cache_salt`;
   - **warm:** a shared salt.
   - Both with `return_token_ids` and `logprobs`.
4. **Gates:**
   - gate 1 and gate 2 from the API;
   - gate 3 through `collective_rpc` block reads in an in-process `LLM` cell;
   - gate 4 with the busy (2 filler streams) and idle variants plus the abort case.
5. **Output:** per-turn `cached_tokens`, predicted vs actual, state max-abs per layer per rank, and TTFT warm vs cold.

---

## 7. What still has to be checked against the source or the GPU (not assumed)

- Whether `defer_block_free` is True in the production config. It should be, because of the QSA CircularBufferSpec with PP2 (`scheduler.py:224-233`). If it is False, `sched_step_seq` does not advance and the fence must use its own counter.
- Whether the decode grid block can be retired and reused before PP0's deferred boundary fold writes it in **today's** code (`_retire_state_blocks` runs on the last-rank-committed count: `allocate_slots` → `remove_skipped_blocks(total_computed − num_in_flight)`, `kv_cache_manager.py:590-596`). Option B removes this hazard for the held block. Gate 3 on PP0 is the check.
- Whether the klee100 MTP is multi-module, which gives `num_reprefillable = 2` instead of 0. It moves option C's target by at most one unit, and the position checks in gate 2 read the real value.
- Grace pins: only hashed blocks are pinned (`kv_cache_manager.py:774`). The option B patch must add the block to the grace pin at publish time, after the fence, not at free.

---

## 8. Prototype and CPU tests

**Folder:** `D:/ARC AGI 2/_scratch/gentail_ckpt/`

- **`gentail_logic.py`** is pure Python with no vLLM import. It contains:
  - the position arithmetic, mirroring the source lines cited in its header;
  - option B and C selection;
  - the PP fence rule, the producer rule and a fence queue;
  - a toy PP0 ring model.
- **`test_gentail_logic.py`:** 11 tests, all PASS (`python -I test_gentail_logic.py`):
  - the current-hit formula reproduces all 677 K100Q `cached_tokens` exactly;
  - tail positions in normal and stop-as-draft finishes;
  - grid and ring math, ring coverage (R = 3: 99.9 %, R = 2: 96.9 %);
  - the busy vs idle PP fence model (the last-step commit is skipped when idle);
  - the fence queue;
  - the K100Q savings replay: perfect 2,430,527; C 2,429,440 (99.955 %); B 1,702,336 (70.04 %, 309 turns); D 2,494,563.
- **No diff against Merlin is included.** None of the three options is a small change that is clearly correct without the GPU gates. Section 3.1 lists each edit for option B with its file:line.

---

## 9. v2 (2026-11-01, later): Option B prototype patch, audit fixes folded in

**Status:** a prototype seam patch with CPU tests exists. Nothing has run on a GPU. The prototype lives in `D:/ARC AGI 2/_scratch/gentail_ckpt/`, with a copy in `Experiment 30/src/gentail/`.
- `gentail_seam.py` patches 2 Merlin files and adds 1 new file, all behind `GENTAIL_CKPT=1`.
- `test_gentail_seam.py`: 31 / 31 pass. `test_gentail_logic.py`: 11 / 11 pass.
- `GPU_TEST_SPEC_gentail.md` is the GPU test spec.

### 9.1 What changed against sections 3.1 and 4

| Item | v1 (sections 3-4) | v2 (patch) | Why |
|---|---|---|---|
| G-1 attention key | partial key at A on block A // 4096 | if A % 4096 == 0, no partial key; the full block A/4096 − 1 must be hashed at A (the full blocks up to A are cached at finish first). Else the partial key goes on block A // 4096 | Merlin skips a partial key at a full boundary (`single_type_kv_cache_manager.py:911-912`); v1 pointed at the next block |
| Producer rule | T < L (pp 2) | **T + pp_size ≤ L + 1** | Found by the CPU PP model with pp_size 24: T < L lets PP0 free the slot before its call T + pp_size. Identical for pp 2 |
| G-2 producer step | arithmetic on committed tokens | `commit_plan_boundary`: an exact port of `prepare_commit_plan_kernel`'s align branch (`ops/recoverssm.py:64-115`). Fed with the window's schedule-time committed count, query length and sampled count; cross-checked against the scheduler's rejection-adjusted count (else `plan_mismatch`, no record) | Bind the producer to what the commit plan did. The worker returns no plan metadata to the scheduler today |
| G-2 generation | — | `PendingTail(request_id, request_generation, position, block_ids, producer_step, valid_tokens, token_hash)`. The generation is bumped on a new Request object under a known id (index reuse); checked at finish and at drain | the scheduler-side mirror of the `pp_utils` slot generation. The worker's own generation check is what skips PP0's commit; the producer rule and the fence make that never reach a published state |
| G-2+ token hash | — | the token hash at b is recorded at finish; at drain, `request.block_hashes[b/64 − 1]` must still equal it | publish-condition invariant |
| Hold | keep the retired block's table ref | one extra ref at record time (`touch`), so `_retire_state_blocks` / free drop only the table ref. Replacing a hold releases the old one behind the fence | same effect, `single_type_kv_cache_manager.py` untouched; tested: retired + held, superseded + released |
| Finish status | finished with output | **FINISHED_STOPPED only** (vLLM PR #50551); abort, length-capped, repetition, error and ignored finishes are discarded | upstream finalized-checkpoint rule |
| Stop truncation | the limit rule only | default **strict**: discard when committed ≥ num_tokens. `GENTAIL_CKPT_STRICT_TRUNC=0` relaxes it (the state at b is ≥ 16 windows older than the end, so relaxed is also exact); the counter `stop_truncated` measures the cost | audit rule; the GPU run measures the cost |
| Publish | at finish if the fence passed | always queued at finish, published only in the drain (scheduler next to `_drain_deferred_frees`; KV manager right after the grace pin) | audit rule. With pp 2 the fence has always passed at finish (b ≤ limit means T ≤ L − 16), so this costs nothing |
| G-3 grace pins | MAX 16 → 21 | MAX_GRACE_PINS = `max_num_seqs` (instance attribute), plus a **pin budget**: all grace-pinned blocks ≤ 25 % of `num_gpu_blocks` (`GENTAIL_CKPT_PIN_BUDGET_FRAC`). Oldest pins are released first; the newest always stays | a pin holds ALL hashed blocks of a turn. K100Q gaps simulation: peak 3 pins / 176,608 tokens (10.7 % of the pool) with gap = finish → next request; 5 pins / 12.7 % with +2 s admission wait. The cap 16 vs 21 never binds there |
| Memory | supersedes the prompt-tail hold | the grid hold releases `_retained_checkpoints[rid]` (the prompt-tail CoW hold) when it is taken. `GENTAIL_CKPT_KEEP_PROMPT_HOLD=1` keeps it | steady state: +0 pages per running request. The released prompt state ages as a cached free block (a sibling branch can still hit it until it is evicted) |
| G-5 telemetry | 4 counters | `recorded, superseded, queued, deferred, published (+_pinned/_lru/_tokens/_store), skipped (+skip_<reason>), released, adopted (+_tokens), evicted, stop_truncated, final_window, pin_budget_released, index_reuse, plan_mismatch, seq_mismatch, no_block, sched_overlap`; "E30_GENTAIL" stats lines with the grace-pin load; an optional per-publish line | GPU gates 2-4 |
| Per-step guard | — | `configure_recoverssm_pp_checkpoints` / `publish_decode_checkpoints = False` stays (tested). The finish-time path is separate and fenced | the GPT research report, section 5.2 |
| Negative control | disable the fence | `GENTAIL_CKPT_NOFENCE=1` disables the fence, the producer rule, the limit, and the stop and length-capped rules | with pp 2 the limit alone already keeps a final-window state out, so the control must lift every rule to show PP0's missing commit. In the CPU PP model it does, in ≥ 5 / 20 seeds; the same seeds with the fence are clean |
| Future store | — | `GenTailController.tail_store` (None): an `offer(pend) -> bool` hook for an INT8 cached-checkpoint store in the style of SGLang #28185. No implementation | cached tail states at about half the bytes |

### 9.2 Gain under the patch's rules (K100Q, 677 turns; the test fails if the CSV is absent)

Saved **1,688,832 of 2,430,527 = 69.5 %** (297 turns):
- 9 turns lose a ≤ 127-token gain to the b > prompt rule (b inside the previous prompt);
- 3 turns follow a length-capped finish and are discarded.

Strict stop truncation and Duck stop strings (which the frontend turns into aborts) cannot be replayed from this table. They are upper-bound effects for the GPU run to measure.

### 9.3 Reference designs

- **vLLM #53803** (draft): rolling align-mode decode / grid checkpoints. Same idea as B. It has no PP fence, and its grid only exists at chunk ends. In Merlin the RecoverSSM commit writes the grid state.
- **vLLM #50551** (open): finalized decode checkpoints at finish, published only on FINISHED_STOPPED. This patch adopts its finish rule and adds the PP fence, the generation and the token hash.
- **vLLM #55697** (RFC, open): application-directed checkpoints (a prompt marker). Prompt positions only; a candidate for a Duck-directed variant later.
- **SGLang `mem_cache/mamba_checkpoint_pool.py` (MambaRadixCache) and #28185** (INT8 store for inactive checkpoints): an alternative architecture, and the future store hook in 9.1.
- **Marconi** (MLSys 2025): admission and eviction for hybrid prefix caches. It is relevant to the pin budget / LRU policy, which v2 keeps simple: grace pin, then LRU.

### 9.4 Open risks (not settled by CPU tests)

1. **Not run on real vLLM classes.** No local venv has vLLM's dependencies, so the logic ran against fakes that mirror BlockPool, the managers and KVCacheManager. Next step: the Kaggle CPU kit check in GPU spec section 0.
2. **Commit-plan inputs.** The port reproduces the kernel exactly, but it assumes the worker's `num_computed` for the window equals the scheduler's committed count at schedule time. It records only windows with nothing else in flight for that request (`sched_overlap` counts the others). The GPU run must show `plan_mismatch` == 0.
3. **Stop strings.** If Duck ends turns by a stop string rather than the EOS / stop token, the engine sees an abort and publishes nothing. The GPU run's `skip_abort` vs `published` shows which.
4. **Strict truncation cost.** It is unknown. The `stop_truncated` counter measures it, and the relaxed mode is one env away.
5. **The released prompt-tail hold.** It may lower hits for sibling branches under pool pressure. `GENTAIL_CKPT_KEEP_PROMPT_HOLD=1` is the A/B.
6. **Hit through the published state on the reader side.** It relies on the standard align-mode block-boundary hit path; gates 2 / 3 check it.
7. **Gate 3 baseline.** The tolerance relies on the measured chunking noise.
