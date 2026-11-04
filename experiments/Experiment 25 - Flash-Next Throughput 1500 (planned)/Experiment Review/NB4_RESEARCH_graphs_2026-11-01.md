# NB4 source research: FULL graphs, commit graph, the PP broadcast, P5 wiring (2026-11-01)

Read-only research for `NB4_TICK_SPEC_2026-11-01.md`. Nothing was run on a GPU, nothing in the repo was changed except this file.

**How claims are marked.**
- Source: `merlin-src` = `D:/ARC AGI 2/_scratch/e23/p2/merlin-src/` (git `ad42e37`, "Merlin 3.0.0").
- Trace numbers: re-measured by me from the raw `*.pt.trace.json.gz` files.
- **UNVERIFIED** = an explanation or prediction nobody has measured yet.
- **(est.)** = arithmetic on measured inputs.

**Source = runtime check.** The server log line numbers match this tree exactly:
- `model_runner.py:1341` "Graph capturing finished";
- `cuda_communicator.py:353` backend list;
- `recoverssm_commit_graph.py:455` / `:533` / `:544` / `:113`;
- `target_dependent_ar/speculator.py:151`.

The only known delta is the #60490 patch (`Build/patches/60490_qsa_indexer_workspace_merlin_ad42e37.patch`), which does not touch any file below.

---

## Lead question (stall): (a)-(d) and arm S

**Two corrections first.**
- **The runtime is the V2 model runner, not `gpu_model_runner.py`.**
  - `VLLM_USE_V2_MODEL_RUNNER=1` is in the boot env.
  - The log's `model_runner.py:1341` is `v1/worker/gpu/model_runner.py`. The V1 file logs the same message at `gpu_model_runner.py:6590`.
  - V1's `_pp_receive_prev_sampled_token_ids_to_input_batch` (`gpu_model_runner.py:4594-4610`) broadcasts ONE int32 `[n,1]` tensor on `pp.device_group`, and V1 waits on it in `_pp_recv_work.wait()` (`:1769-1771`).
  - The trace shows THREE broadcasts (`Long ×16`, `Int ×8`, `Long ×12`) on the group described as `pp_broadcast`. That is the V2 `PPHandler` (`v1/worker/gpu/pp_utils.py:59-310`). `_pp_recv_work` is not used.
- **The stalled graph is FULL, not PIECEWISE**: 1 launch, 1,037 nodes, every decode step.

**(a) Where stage 0 waits, and whether the dependency is real.**
- **Real dependency, outside the graph.**
  - At the start of `execute_model` of step T, `update_pp_decode_requests` (`model_runner.py:2196` call; definition `:1385-1398`) pops the ring entry of step T − pp_size = T − 2, which is the SAME micro-batch.
  - It then does `self.main_stream.wait_event(slot.event)` (`pp_utils.py:137-156`). The event is recorded after all 3 receives of step T−2 (`pp_utils.py:256`).
  - This is before `prepare_inputs` and the eager `_combine_sampled_and_draft_tokens_kernel`. It is real: this micro-batch's input ids and draft tokens come from step T−2.
- **No receive event or wait exists inside the graph.** The graph is captured at boot and cannot reference per-step events (`run_fullgraph`, `cudagraph_utils.py:640-660`).
- **The in-graph stall is a FALSE dependency on the OTHER micro-batch.** NB1 B1 timeline (ms):
  - graph(T−1) ends at 458.08;
  - receive(T−1) #1 (`Long ×16`) starts at 458.10, because `receive()` does `broadcast_stream.wait_stream(main)` right after graph(T−1) (`pp_utils.py:233-234`);
  - graph(T) starts at 460.06. Its real gate, step T−2's drafts, had already arrived at about 450.0;
  - at +4.9 ms (node #173) graph(T) stalls until receive(T−1) #2 (`Int ×8`) ends at 486.07, then resumes 4-9 µs later (39 of 39 graphs, both PP0 ranks).
  - Step T−1 is the other micro-batch, and graph(T) needs nothing from it. The likely mechanism is ordering in the stream / work queue: #2 cannot be dispatched behind the spinning #1 and blocks the graph branch queued behind it (**UNVERIFIED** mechanism; the timing match is measured).

**(b) Batches in flight.**
- `max_concurrent_batches = pp_size + 1 = 3` with async scheduling and V2 (`config/vllm.py:607-617`). Async is on by default for MTP (`:1620-1679`). Engine `batch_queue = deque(maxlen=3)` (`v1/engine/core.py:312-318`; `step_with_batch_queue` `:790-850`).
- **The third slot cannot hold a third decode micro-batch.**
  - Every decode request is re-eligible only `pp_size` = 2 steps later (`async_scheduler.py:52-57`, enforced at `scheduler.py:794-798`).
  - The receive ring is also fixed at `pp_size` deep (`pp_utils.py:85-89`).
  - So the third slot only ever carries prefill work or an empty step (NB2's `generation_0(0)`).
- **It would not hide the stall.** The stall is a false wait on step T−1, not the T−2 dependency, so more depth does not change it.
- **No knob.** The only one is `VLLM_XPU_PP_MICROBATCH`, which is XPU-only (`async_scheduler.py:19-23`, `pp_utils.py:84-87`). Depth 3 would need a patch to both the stagger and the ring.

**(c) Knobs that can remove the false dependency.**

| Knob | Status |
|---|---|
| `CUDA_DEVICE_MAX_CONNECTIONS` (CUDA default 8, max 32: the number of hardware work queues per device) | Not set anywhere: the whole Merlin tree, our builders and the NB1 server log have 0 hits. vLLM and PyTorch do not set it (Megatron/TE set it to 1). NCCL setting it: **UNVERIFIED**, no NCCL binary here; check the env inside the workers. **Best env candidate.** Must be in the env before `vllm serve`, since workers inherit it before CUDA init. Not guaranteed: the graph forks about 96 streams, more than 32 queues. |
| `NCCL_LAUNCH_ORDER_IMPLICIT` (NCCL ≥ 2.26; default 0; orders launches on one device across communicators in host program order) | Name in 2.29.7 **UNVERIFIED** here; set it and look for "set by environment" in the NCCL INFO log. Host order is receive(T−1) then graph(T), so it can only ADD this ordering for eager calls and cannot reorder already-captured graph nodes. **Not a fix; likely neutral or worse.** |
| `NCCL_GRAPH_MIXING_SUPPORT` (default 1) | Concerns mixing graph and eager calls on the SAME communicator. Our stall is between two communicators (TP comm in the graph, `pp_broadcast` eager). Setting it to 0 is unsafe: the TP comm mixes eager prefill calls and graph replays. **Do not use.** |
| Posting the receive later or elsewhere | It is already on its own `broadcast_stream` (`pp_utils.py:78`). The problem is the 3-kernel chain posted before graph(T). Two patches remove it: (i) **pack** the 3 broadcasts into one int64 `[n, 2K+3]` on both sides (sender sends after the drafts, `pp_utils.py:202-217, 277-310`); or (ii) **defer** `pp_handler.receive(...)` (`model_runner.py:2647`) until after the next step's `run_fullgraph` launch (`:2492`), so the graph is queued ahead of it. |
| `VLLM_TARGET_SEQ_LENS_ASYNC` | Not on this path. Its snapshot capture is skipped under PP (`model_runner.py:2764` `... and self.pp_handler is None`; init `:409-416`), and it only removes a host `seq_lens.cpu()` sync. |
| `VLLM_PP_*` | Only `VLLM_PP_LAYER_PARTITION` exists (`envs.py:895`), which changes load, not ordering. No async-output env affects the PP relay. |

**(d) The `Long ×16` broadcast.**
- It is the sampled tokens of step T−1: `int64 [n=4, K+1=4]`, 128 B (`pp_utils.py:235-241`).
- It is on the same stream (stream 31 = `PPHandler.broadcast_stream`) and the same group (`pp_broadcast`, ranks [0, 2] and [1, 3], `pp_utils.py:97-100`) as the following `Int ×8` (`combined` = num_sampled / num_rejected `[2, n]` int32, `:238-244`) and `Long ×12` (drafts `[n, K]` int64, `:245-255`).
- On stage 0 it spins 29.2 ms on average, until stage 1 has sampled.

**Arm S recommendation.**

| Arm | Change | Must log to prove it engaged |
|---|---|---|
| **S** (single env) | `CUDA_DEVICE_MAX_CONNECTIONS=32` in the server env | Worker `os.environ` read-back (all 4 ranks); per stage-0 graph: largest in-graph gap, its node index and stream, and its offset to the nearest `pp_broadcast` kernel end; per-rank tick, busy and idle; stage-1 `SendRecv` wait |
| **S2** (smallest patch; there is no second credible env knob) | Q-pack (one packed broadcast). Q-defer is the alternative. | Same as S, plus 1 `pp_broadcast` kernel per step on each side |

- **Pass for S and S2**: in-graph gap < 1 ms in ≥ 95 % of stage-0 graphs; the tick moves toward the stage-1-bound ≈ 40 ms (est. +22-25 %).
- **Reading S**:
  - the gap moves to another node index → the mechanism is confirmed but not solved, so go to S2;
  - the gap is unchanged → the work-queue theory is wrong.

---

## 0. Headline findings (they change the NB4 arm design)

1. **The decode target forward already runs as ONE FULL CUDA graph. "0 FULL" was a mistake in our analyzer.**
   - NB1 R2 B1: every decode scope on PP0 has exactly 1 `cudaGraphLaunch` with 1,037 GPU nodes; on PP1 it has 1 launch with 951 nodes (40 of 40 scopes on each rank).
   - NB2 SEED2: the 20 "PIECEWISE" scopes are each 1 graph launch with 1,053 nodes. The 20 "EAGER" scopes are `execute_context_0(0)_generation_0(0)`: empty 0-token steps of 0.18 ms.
   - Why the analyzer got it wrong:
     - `_scope_kind` (`e23_lab.py:804-813`, the copy embedded in nb2/nb3 `*_code_export.py`) labels a scope PIECEWISE when any "compute-category" eager kernel appears in it.
     - The eager kernels it found are the RecoverSSM commit `_commit_gdn_state_kernel` (category "gdn"), plus one eager `AllReduce` on NB2.
     - These run after sampling, outside the forward. They are not PIECEWISE pieces.
2. **The draft (MTP) also runs FULL graphs.**
   - The log shows `role=draft cudagraph_mode=prefill:FULL_AND_PIECEWISE,decode:FULL_DECODE_ONLY` (`server_nb1_b1_fp8.log:588-589`).
   - The PP1 trace has 3 extra graph launches per tick (72, 65 and 65 nodes): draft prefill plus 2 draft decode steps, 80 + 40 launches over 40 ticks.
   - So the 27B "decode:NONE" problem does not apply here.
3. **About 21 ms of the ~23.7 ms stage-0 idle time per tick is a stall INSIDE the stage-0 FULL graph.** This is the main new finding.
   - Where: in 39 of 40 graphs, on both PP0 ranks, node #173 (an in-graph TP `AllReduce` on the forked graph stream 2338, right after `triton_poi_fused_add_0`, about 4.9 ms into the graph) starts 20.7-21.6 ms late. Mean 21.2 ms; the first graph stalled 70.8 ms.
   - When it ends: every time, 4-9 µs after the pp_broadcast `Int` broadcast of 8 elements on side stream 31 completes.
   - So the graph is held up by the sampled-token broadcast receive of the PREVIOUS step, which waits on stage 1.
   - Mechanism (**UNVERIFIED**, but it fits every number): head-of-line blocking in a CUDA hardware work queue.
     - Broadcast #1 is dispatched and spins.
     - Broadcast #2 sits behind it, cannot be dispatched, and blocks any graph branch that maps to the same queue.
     - When #2 runs (20 µs) and #3 is dispatched, node #173 starts.
   - Effect: the stage-0 graph wall time is 47 ms for 27.6 ms of kernels, and stage 0 is in-graph back-to-back. The stall is on the critical path: stage 1 then idles 8.4 ms/tick in `SendRecv` waiting for stage 0's hidden states.
4. **NCCL "63.5 %" of kernel time is an accounting artefact.**
   - It is `ncclDevKernel_Broadcast_RING_LL` on stage 0's side stream 31 spinning while it waits for stage 1: 1,713.6 ms of the 2,896.8 ms kernel sum on rank 0.
   - The messages are tiny (n = 4 requests):
     - 16 × int64 (128 B, sampled tokens);
     - 8 × int32 (32 B, num_sampled/num_rejected);
     - 12 × int64 (96 B, draft tokens).
   - On the sender (stage 1) the same kernels take 9-13 µs.
5. **Stage 1 is the heavier stage.**

   | Rank | Tick | GPU compute busy (union, broadcast/SendRecv excluded) | Idle |
   |---|---|---|---|
   | PP0 | 50.65 ms | 27.0 ms | 23.7 ms |
   | PP1 | 48.5 ms | 40.1 ms | 8.4 ms |

   PP1 per tick:
   - target graph 22-23 ms;
   - eager lm-head GEMM 2.58 ms;
   - logits AllGather 1.44 ms;
   - top-k/top-p 0.25 ms;
   - commits 0.87 ms;
   - 3 draft graphs 12.3 ms.

   With the stall removed, the tick becomes stage-1-bound at ≈ 40-41 ms instead of 49.5-50.6 ms: about +22-25 % tok/s (est.).
6. **PP micro-batch split is not balanced by the scheduler.**
   - `AsyncScheduler` sets `next_decode_eligible_step = current_step + pp_size` (`v1/core/sched/async_scheduler.py:52-57`; enforced at `v1/core/sched/scheduler.py:794-798`). There is no balancing.
   - A request's parity is fixed by when its prefill ends. NB2 SEED2 ran all 8 requests in the even steps (`generation_8(24)`) and 0 in the odd ones (`generation_0(0)`): no PP overlap at all.
   - NB1 B1 happened to be 4/4.
7. **With seqs 9 and the CAPS16 list, a 9-request micro-batch runs fully EAGER (target and draft decode).** See §1.3. Our NB5 helper `captures_cover` (`Build/nb5_lib.py:1140-1142`, `max(caps) >= seqs*(K+1)`) reports "covered" here, which is wrong.

---

## 1. FULL vs PIECEWISE for the target and the draft

### 1.1 How the dispatcher picks (V2 model runner, `VLLM_USE_V2_MODEL_RUNNER=1`)

**Uniform-decode detection** (`v1/worker/gpu/model_runner.py:1500-1608`):
- `gather_batch_req_state` returns `uniform_tok_count = max_query_len` only when:
  - every row is decode-like (`decode_graph_eligible`);
  - the query length is uniform (`v1/worker/utils.py:1004-1012`).
- With MTP K3 a decode row is 1 + 3 = 4 tokens.

**Dispatch**: `dispatch_cg_and_sync_dp(...)` (`model_runner.py:2247-2262`) → `CudaGraphManager.dispatch` (`v1/worker/gpu/cudagraph_utils.py:608-637`) → `_is_compatible` (`:113-140`).
- A FULL descriptor needs `desc.uniform_token_count == uniform_tok_count`, `desc.num_reqs >= num_reqs` and `desc.num_tokens >= num_tokens`.
- If nothing matches, it returns `CUDAGraphMode.NONE` (eager).

**Capture set** (`cudagraph_utils.py:266-404`), with `max_decode_tokens = max_num_seqs × decode_query_len` (`:273`):
- **FULL uniform decode**: for each capture size, `rounded = round_up(size, 4)`. It is **skipped when `rounded > max_decode_tokens` or `> max_cudagraph_capture_size`** (`:331-349`).
- **PIECEWISE (mixed)**: with RecoverSSM it is captured only for `num_tokens <= max_decode_tokens` (`:361-379`, "recoverSSM cannot capture a dummy query wider than its workspace").
- **Candidate ranges** run only up to the largest captured size (`:387-404`). A larger batch has no candidate, so it runs NONE.

**Explicit capture lists** are used as given, filtered to `<= max_num_batched_tokens`. `max_cudagraph_capture_size` becomes the last entry (`config/vllm.py:2488-2496` and `~2535-2560`).

**Backend support**: every backend in the forward is `UNIFORM_BATCH`, so FULL is allowed for uniform decode:
- `models/qwen4_exp/nvidia/qsa.py:320`;
- `common/qsa_cache.py:652`;
- `v1/attention/backends/gdn_recoverssm.py:208`;
- `ple_recoverssm.py:246`;
- `short_conv_attn.py:107`;
- `mamba_attn.py:88`.

**PLE**: the host gather is baked into the FULL graph (`cudagraph_utils.py:179-183`, "FULL graphs bake every PLE wait/copy into their native graph"). The trace shows a `Memcpy HtoD (Pinned -> Device)` of 40,960 B inside the stage-0 graph at +1.36 ms.

**Hyper-connections**: the `_hc_*` kernels are graph nodes.

**Conclusion: there is no blocker to FULL, and FULL is what runs.**

### 1.2 Draft model

- `v1/worker/gpu/spec_decode/target_dependent_ar/speculator.py:130-148`:
  - the draft prefill manager uses `decode_query_len = K + 1` (same rules as the target);
  - the draft decode manager uses `FULL_DECODE_ONLY` with `decode_query_len = 1`, so its FULL sizes are capture sizes ≤ `max_num_seqs`.
- The log resolves to `draft.prefill_cudagraph_manager=FULL_AND_PIECEWISE draft.decode_cudagraph_manager=FULL_DECODE_ONLY` (`server_nb1_b1_fp8.log:983-984`).

### 1.3 Coverage at our sizes

| Config | Micro-batch | Tokens | Target graph | Draft decode graph |
|---|---|---|---|---|
| NB1 B1, seqs 8, list up to 32 | 4 req | 16 | FULL 16 (exact) | FULL 4 |
| NB2, K2, seqs 16 | 8 req | 24 | FULL 24 | FULL 8 |
| **NB5/NB4 base, seqs 9, CAPS16** | 4 req | 16 | FULL 16 | FULL 4 |
| same | 5 req | 20 | FULL 24 (padded, 6 req) | FULL 6 (padded) |
| same | 7 req | 28 | FULL 32 (padded) | FULL 8 (padded) |
| same | **9 req** | **36** | **NONE (eager)**: 40/48/64 dropped (> 36); no 36 size; PW only ≤ 32 | **NONE**: 12 > 9 is dropped, no 9 size |

- CAPS16 at seqs 9 gives FULL decode = {4, 8, 12, 16, 24, 32} (6 graphs). PIECEWISE = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}.
- 40, 48 and 64 are captured nowhere.
- Merlin's own ladder `tools/merlin/registry.py:53, 83-89` (`capture_sizes_for(9)` = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 36]) adds 36 but still lacks 9 for the draft decode.
- At level 8 (the spec's 8k@8 / 40k@8 windows) no micro-batch can exceed 8 requests. **So arm G changes nothing at level 8; it must be measured at level 9.**

### 1.4 What gives FULL for every real n (seqs 9, K3)

```
--compilation-config {"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8,9,12,16,20,24,28,32,36]}
```

- Target FULL: 4, 8, 12, …, 36 cover n = 1..9 exactly, with no padding.
- Draft decode FULL: 1..9.
- Draft prefill: same as the target.
- It also gives a RecoverSSM commit graph for every n (§2).
- Cost:
  - 16 PIECEWISE + 9 FULL target graphs;
  - B2 (13 PW + 9 FULL) took 0.44 / 0.56 GiB on PP0 / PP1, against 0.33 / 0.43 GiB for B1's 10 + 6;
  - so expect about +0.12-0.2 GiB per card over base (est.), out of headroom that a fixed `--kv-cache-memory-bytes` leaves.
- No env change is needed. Breakable graphs stay OFF (`VLLM_USE_BREAKABLE_CUDAGRAPH=0`, as now).

---

## 2. RecoverSSM commit graph (`VLLM_RECOVERSSM_COMMIT_GRAPH`)

**Modes** (`v1/worker/gpu/model_states/recoverssm_commit_graph.py:13-21, 64-91`):
- `eager` (default): nothing is captured;
- `staged`;
- `graph` (alias `1`).
- It can be flipped at runtime with the worker RPC `set_recoverssm_commit_mode` (`:106-114`).
- Graphs are captured only when the boot-time mode is not eager (`:229`).

**Bit-identity claim** (`:5-11`): same kernels, grids, arguments, order and stream as the eager chain, rank-local, no collectives. It is by construction, not measured by us (**UNVERIFIED** on GPU).

**What decides which n is captured.**
- `capture_warmup` (`:396-458`) is called from `RecoverSSMState.record_step(for_capture=True)` (`model_states/recoverssm.py:103-111`), reached from `mamba_hybrid.py:399-409` once per FULL forward capture descriptor.
- So **n = `desc.num_reqs` of each FULL uniform-decode descriptor**, i.e. `round_up(size, 4) / 4` for each capture size kept by §1.1.
- NB1 B2 (seqs 16, CAPS16) captured exactly n = 16, 12, 10, 8, 6, 4, 3, 2, 1 (9 graphs) on the PP1 ranks (`server_nb1_b2_fp8.log:373-390`).
- With CAPS16 at seqs 9: n ∈ {1, 2, 3, 4, 6, 8}. A real n of 5, 7 or 9 falls back with `not_captured` (`:538-546`), and a padded FULL step can also fail the `n == num_reqs` test (`not_uniform_decode`, `:281-283`).

**Correction to the spec's reading of NB1 B2.**
- The line "no graph for this step (n=16, 0 captured)" (`server_nb1_b2_fp8.log:366-367`) was logged once (`_logged_uncaptured`), BEFORE the capture phase, during the memory-profile dummy run.
- The capture phase then captured 9 graphs, and the first replay happened (`:510-511`, `captured=9 … first replay n=1`).
- The notebook then flipped the mode to eager at `:518-521` (`nb1_code_export.py:1291`; `commit_mode_at_start`).
- B2 commit stats (from `nb1_results.json` `commit_mode_at_start`):
  - PP1 ranks: captures 9, replays 30, staged 5 (`not_captured` 5), eager 7 (`no_commit` 7 = steps without a spec commit);
  - PP0 ranks: captures 0, `warmup_metadata_type` 18.
- **So B2's oracle FAIL cannot come from the commit graph**: the oracle ran in eager commit mode. That leaves the other bundled B2 changes (`MERLIN_SHARE_DRAFT_WORKSPACE=1` and the DIAG env). The 30 replays during the 4 probes before the flip touched only those requests' states (**UNVERIFIED** that nothing carried over through the prefix cache).

**Stage 0 can never use a commit graph.**
- PP-deferred commits always take the eager path: `recoverssm.py:139` `if not deferred and self._commit_graphs.run(...)`, and the docstring `recoverssm_commit_graph.py:35`.
- Stage 0 also carries the PLE RecoverSSM metadata, which is not `GDNRecoverSSMMetadata`, so it is ineligible (`:275-276`; this is the 18 `warmup_metadata_type` fallbacks).
- **Only the stage-1 commit (0.87 ms of GPU kernels per tick) can be graphed.**
- The GPU kernels are identical either way, so the saving is host launch/Python time only. In the B1 timeline the PP1 commit kernels already run back to back (3 × 0.22-0.33 ms, then the draft graph 70 µs later).
- **Expected gain ≤ 1 % (est.).**

**Setting for all real n at seqs 9**: the §1.4 capture list plus `VLLM_RECOVERSSM_COMMIT_GRAPH=graph`. With the base CAPS16 list, the env alone covers n = 4 and 8 (the level-8 split cases).

---

## 3. The `Broadcast_RING_LL` path, its sizes, and the stage-0 gap

**Who issues it** (V2 runner, PP2):

| Side | Call | Code |
|---|---|---|
| Sender (last PP rank) | `PPHandler.broadcast` (sampled tokens padded [n, K+1] int64 + `stack(num_sampled, num_rejected)` [2, n] int32) | `v1/worker/gpu/pp_utils.py:277-310`, from `model_runner.py:2679` in `sample_tokens` |
| Sender (last PP rank) | `PPHandler.broadcast_drafts` ([n, K] int64) after drafting | `pp_utils.py:202-217`, from `model_runner.py:2817-2820` |
| Receiver (stage 0) | `PPHandler.receive` posts all 3 receives at once on the side `broadcast_stream` | `pp_utils.py:219-275` (created at `:78`), from `sample_tokens`, `model_runner.py:2647` |

- The broadcasts use a sibling communicator, `pp_broadcast` (`pp_utils.py:97-100`; trace `Process Group Description: pp_broadcast`, ranks [0, 2] and [1, 3]).
- The receiver's broadcast stream first `wait_stream(main)`, so it starts after this step's graph.
- The result is consumed `pp_size` steps later: `get_prev_sampled_outputs`, called from `update_pp_decode_requests` at the start of the step (`model_runner.py:1385-1398`), does `main_stream.wait_event(slot.event)` (`pp_utils.py:137-156`).

**It is NOT any of these**:
- the hidden-state hand-off: that is `irecv/isend_tensor_dict` = `ncclDevKernel_SendRecv` of 81,920 bf16 (16 tokens × 2 × 2560, the hc pair), plus gloo metadata (`v1/worker/gpu_worker.py:1638-1650`, `:1676-1680`);
- a TP metadata broadcast;
- the MTP draft itself.

**Message sizes per step** (n requests in the micro-batch, K = 3):
- 32n B (sampled tokens);
- 8n B (num_sampled/num_rejected);
- 24n B (draft tokens).
- For n = 4 that is 128 + 32 + 96 B.

**Waiting.** On the receiver the kernels last 29.2 ms (#1, mean), 72 µs (#2) and 13.5 ms (#3). They spin until stage 1 has sampled (#1) and drafted (#3). This is "waiting for the other stage", not data movement.

**How it creates the stage-0 gap** (NB1 B1, one tick; `tl2.py` timeline below):
1. Stage 0's graph(T) is gated only by slot T−2. It starts about 2 ms after receive(T−1) #1 begins to spin.
2. Its branch at node #173 stalls until receive(T−1) #2 runs, about 21 ms later (after stage 1 has sampled T−1).
3. Graph(T) then finishes about 22 ms later and sends its hidden states.
4. Stage 1 has finished its draft about 8 ms before that and idles in `SendRecv`.

Tick ≈ 22 (rest of the stage-0 graph) + 27.5 (stage-1 target + sampling) ≈ 50 ms. That matches the trace: 50.65 ms on PP0, 48.5 ms on PP1.

**Existing knobs.** None targets this.
- Async scheduling is on by default for MTP with the V2 runner (`config/vllm.py:1620-1679`), so `max_concurrent_batches = pp_size + 1 = 3` (`config/vllm.py:607-617`; engine `batch_queue_size`, `v1/engine/core.py:312`).
- The decode stagger is fixed at `pp_size` (the only knob, `VLLM_XPU_PP_MICROBATCH`, is XPU-only: `async_scheduler.py:19-23`, `pp_utils.py:84-87`).
- NCCL protocol knobs do not matter for 128-B messages.

**Two candidate fixes** (both **UNVERIFIED**):

| Fix | Change | Expected | Risk |
|---|---|---|---|
| Q-env | `CUDA_DEVICE_MAX_CONNECTIONS=32` (CUDA default 8; nothing in Merlin sets it: grep finds no hit) | Fewer streams share a hardware queue, so the stall may vanish or move. It also serves as the mechanism test. | Not a guarantee: the graph has about 96 forked streams |
| Q-pack | Patch `pp_utils.py`: the sender stashes the `broadcast()` tensors and sends ONE int64 tensor `[n, (K+1) + 2 + K]` from `broadcast_drafts`; the receiver posts ONE broadcast of that shape and slices/casts it (int32 counts are exact through int64) | Leaves no not-yet-dispatchable broadcast ahead of the next graph, so no head-of-line block. Stage 0 already waits for the draft part before using the slot, so the latency is unchanged. | Every step must pair one send with one receive (same `compute_need_sampled_mask` test on both sides, as now); fail closed on a mismatch |

If the stall goes away: tick ≈ 40-41 ms (stage-1-bound), **+22-25 % tok/s (est.)**.

**Later levers on the stage-1 load** (40 ms vs 27 ms), noted only:
- `VLLM_PP_LAYER_PARTITION` (now 25,23). Each layer moved to stage 0 adds about 0.63 GB of W4 experts per stage-0 card (est.), so it is memory-limited.
- A parity-balancing scheduler patch, for the NB2 case.
- `draft_lm_head_quant` (`config/speculative.py:610-623`). It is supported by `models/qwen4_exp/nvidia/mtp.py`; NB2 already used int8.

---

## 4. Where P5 plugs in

**Dispatch chain** (`vllm/distributed/device_communicators/cuda_communicator.py:362-460`), in this order:
1. NCCL_SYMM_MEM;
2. QUICK_REDUCE;
3. FLASHINFER_PCIE_IPC;
4. FLASHINFER;
5. AITER;
6. CUSTOM (`ca_comm`);
7. SYMM_MEM;
8. PYNCCL.

The logged list (`:286-360`) is informational.
- **There is no pluggable registry.**
- CUSTOM is off because of `--disable-custom-all-reduce` (`_ENABLE_CUSTOM_ALL_REDUCE`, `:66`; `ca_comm` stays None, `:147-156`).
- So P5 takes that empty slot: insert it before the PYNCCL fallback.

**Call path.** `GroupCoordinator.all_reduce` goes through the custom op `torch.ops.vllm.all_reduce`, then `_all_reduce_out_place`, then `device_communicator.all_reduce` (`distributed/parallel_state.py:198-203`, `:724-750`). Python dispatch runs at FULL-graph capture time, so a P5 kernel launched there is recorded into the target and draft graphs.

**Wiring without forking the wheel.** Monkeypatch `CudaCommunicator.__init__` (create a P5 handle when the group name starts with `tp` and `world_size == 2`) and `CudaCommunicator.all_reduce` (route when bf16, contiguous and bytes ≤ the P5 cap; else fall through). Two places can do this:
- (a) a `vllm.general_plugins` entry point, loaded in every worker (`v1/worker/worker_base.py:292-294`) before `init_device` builds the communicators;
- (b) an import-time side effect of the `--worker-extension-cls` module that NB1 already uses, resolved at worker init (`worker_base.py:314-330`).

**Graph interaction.**
- P5 is one stream-ordered kernel, graph-capturable, with no CPU work per call. The epoch is read from a device counter, so replays need no reset (`Build/nb3_export/p5_src/host_ar.cu:1-17`; ops `host_ar_binding.cpp:308-331`).
- No buffer registration is needed (unlike `ca_comm.capture()`, `parallel_state.py:673-722`), so nothing needs to be added to `graph_capture`.
- Requirements:
  - both TP ranks make the identical sequence of P5 calls, including capture warm-ups;
  - the route decision depends only on the shape;
  - one shm name per TP group (pair 0,1 and pair 2,3).
- `timeout_ms` must be generous: a P5 timeout NaN-fills and sticky-aborts the handle (`host_ar.cu:12-17`), and in-graph skews of 21+ ms exist today.

**Which groups.**
- In-graph `AllReduce` count: 51 per stage-0 forward (2 per layer + 1) and 55 per stage-1 tick.
- Whether the MoE combine goes through `tp` or `ep` (same ranks): **UNVERIFIED**. The wrapper must count P5 calls per group during capture.

**Expected gain.**
- NB3 (graph mode): at T = 16, NCCL 48.96 µs vs P5 30.65 µs (`q27-xhigh-e25-nb3-kernels/nb3/nb3_tables.json` `tables.d7`).
- That saves about 0.9 ms per tick per stage (est.).
- Stage 0's saving is hidden behind the §3 stall. Stage 1's is about 2 % of a 40-50 ms tick (est.).
- At T = 36 the P5 speedup is only 1.32×.

**Alternative engine-native slot.** `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1` (`flashinfer_pcie_ipc_all_reduce.py`) needs FlashInfer `PcieIpcAllReduceWorkspace` and CUDA IPC peer mappings. Whether either exists on Kaggle L4: **UNVERIFIED**. We run `NCCL_P2P_DISABLE=1`, although NCCL logs `isAllCudaP2p 1`.

---

## 5. Recommended arms (base = NB5 config: seqs 9, CAPS16, MTP K3, FP8 KV, pin 214 blocks)

**Prerequisite for every arm: fix the analyzer first.**
- FULL = exactly one graph launch with ≥ 500 nodes inside the `execute_context_*` scope. Eager pre-forward and post-sampling kernels are allowed (commit, input prep).
- A 0-token scope is EMPTY, not EAGER.
- Report the n histogram from `generation_N(T)`.
- Check against the B1 traces: 40/40 FULL on PP0 and PP1, and draft graphs 72/65/65 × 40 on PP1.

**Windows.** Add level-9 windows (8k@9, 40k@9) to all arms, or arm G cannot engage. 9 × 40k ≈ 0.36 M tokens < 214 × 4096 = 0.88 M (est.).

### Arm G: FULL coverage at seqs 9

- **Change**: the capture list `[1,2,3,4,5,6,7,8,9,12,16,20,24,28,32,36]` (§1.4). Nothing else.
- **Must log**:
  - capture lines: `Capturing CUDA graphs (PIECEWISE) …/16`, `(FULL) …/9` and the draft decode count;
  - "Graph capturing finished … took X GiB" per rank, and free MiB after ready (OOM risk);
  - the per-scope n histogram and per-scope graph-launch count;
  - **`--cudagraph-metrics`** (`config/observability.py:82`; table `compilation/cuda_graph.py:42-90`; logged by `v1/metrics/loggers.py:136-247`): a table of unpadded tokens, padded tokens and runtime mode, which must show 36 → FULL. Turn the flag on in every arm, including base, so it is not "the change".
- **Proof that it engaged**: the level-9 window shows FULL for 36 tokens, and no NONE decode rows.

### Arm C: commit graph

- **Change**: `VLLM_RECOVERSSM_COMMIT_GRAPH=graph`. Nothing else.
- **Must log**:
  - the `[recoverssm-commit] captured commit graph n=…` lines: PP1 ranks only, expected n = 1, 2, 3, 4, 6, 8;
  - the `mode=graph captured=K … first replay n=…` line;
  - before and after each window, the counters (`replays`, `staged_steps`, `eager_steps`, `fallbacks`) from the `set_recoverssm_commit_mode` RPC (as in `nb1_code_export.py:1836-1860`; RPC it with the current mode to read without changing it).
- **Proof**: PP1 replays > 0 and growing by about 1 per PP1 step in the window; PP0 captures 0 (expected, deferred).
- **Correctness**: EXACT expected.
- **Expected**: ≤ 1 % (est.). C covers n = 5, 7 and 9 only if combined with the G list, which is not allowed in a one-change arm.

### Arm P: P5 in the TP all-reduce

- **Change**: the P5 wrapper of §4, via plugin or worker-ext import, with the NB3-built `.so`. The cap is ≤ 184 KB (T ≤ 36); fall back to NCCL above it.
- **Fail closed**:
  - `.so` missing, `init` error, a `status` word ≠ 0, or 0 P5 calls recorded during capture → `P5_NOT_INTEGRATED`;
  - NaN output → `CORRECTNESS_FAIL`.
- **Must log**:
  - the P5 init line per rank (group, world, shm name, max_bytes, timeout);
  - P5 calls per group during capture (count while `torch.cuda.is_current_stream_capturing()`);
  - `epoch(handle)` before and after each window (Δ = in-graph calls);
  - `status(handle)` all zero;
  - trace: P5 kernel nodes inside the decode graphs and 0 `ncclDevKernel_AllReduce` there.
- **Expected**: 0-2 % (est.), stage 1 only.

### New arm, proposed: Q (the stall)

- It goes after T0 if the user agrees: it is the largest lever found (est. +22-25 %).
- Q-env (`CUDA_DEVICE_MAX_CONNECTIONS=32`) is a one-boot env test that also tests the mechanism. Q-pack is the robust patch.
- **Must log**:
  - per stage-0 graph: wall time vs node-sum, the largest in-graph gap and its node index/stream;
  - the time between that gap's end and the nearest pp_broadcast kernel end;
  - per-rank tick, busy and idle;
  - stage-1 `SendRecv` wait per tick.
- **Proof**: in-graph gap < 1 ms in ≥ 95 % of graphs, and stage-1 SendRecv wait about 0.

### T0 gap taxonomy

The spec's four buckets need a fifth: **in-graph stall** (GPU idle while a FULL graph is in flight, attributed to the pp_broadcast kernel whose end it waits for).

Values to reproduce offline on B1 rank 0:

| Quantity | Value |
|---|---|
| Tick | 50.65 ms |
| Compute | 27.0 ms |
| In-graph stall | 21.2 ms (39/39) |
| Pre-graph eager | ≈ 2.7 ms span / 2.0 ms busy |

Rank 2 (stage 1): compute 40.1 ms, idle 8.4 ms.

**Scratch scripts used** (`D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/nb4/`):

| Script | What it does |
|---|---|
| `scopes.py` | per-scope graph and eager kernels |
| `hol.py` | in-graph stall vs broadcast end |
| `busy.py` | per-rank tick, busy, idle |
| `tl2.py` | merged two-stage timeline |
| `eager.py` | eager kernel census |
| `ar.py` | NCCL message sizes |
