# NB4 TICK: research from our experiment folders (2026-11-01)

Read-only research for `Experiment Review/NB4_TICK_SPEC_2026-11-01.md`. Nothing was run on Kaggle and nothing was pushed.
No repo file was changed except this one.

Sources: the experiment folders (E11-E26), `PLAN_1500.md`, the `Past work mining/A1…A7` reports, the `GPT research inputs/`
verifications, the NB0-NB3 builders and code exports, and the Merlin v3.0.0 source checkout we already hold
(`_scratch/e23/p2/merlin-src/`, written below as **[MS]**). The real NB1 R2 B1 and NB2 SEED2 torch traces under
`_scratch/e25_gpu_out/` were also **re-analysed offline** (CPU only), because the spec asks for the analysis to be checked
on those traces before any GPU run.

Path legend:
- **[E25R]** `experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Results/`
- **[E25B]** `experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build/`
- **[RC2]** `experiments/Experiment 13 - Flash-Next in the Duck harness on 4xL4 (planned)/Experiment Results/E13_RC2_GPU_v1_2026-10-31.md`
- **[B1T]** `_scratch/e25_gpu_out/q27-xhigh-e25-nb1-capacity-r2/nb1/traces/profile_b1/*.pt.trace.json.gz` (4 ranks, 40 forwards, 8 streams, 8k prompts)
- **[S2T]** `_scratch/e25_gpu_out/q27-xhigh-e25-nb2-smoke-seed2/nb2/profiles/B/*.pt.trace.json.gz` (4 ranks, K2 + INT8 head + seqs 16 + CAPS16)
- **[B2L]** `_scratch/e25_gpu_out/q27-xhigh-e25-nb1-capacity-r2/nb1/logs/server_nb1_b2_fp8.log`
- **[OFF]** offline re-analysis done for this note. Scripts: `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\nb4r\` (`gl.py`, `gap2.py`, `gap6.py`, `s1.py`, `scope.py`). The scratchpad is session-local, so the method is also written out in §3.1.

---

## 0. The five things NB4 most needs to know (short list; details and sources below)

1. **The stage-0 "gap" is not a host launch gap and not a scheduler / IPC wait. It is a stall inside the stage-0 FULL CUDA graph** [OFF on B1T, §1.1].
   - In 39 of 40 stage-0 graph replays, on both TP ranks, the graph runs 4.70 ms (layers 0-3).
   - It then stalls a median **21.70 ms** before the same node: kernel index 172 of 1,036, an in-graph `AllReduce` on graph stream 2338.
   - The stall ends a median **0.03 ms** after the 27.9 ms PP sampled-token `Broadcast_RING_LL` kernel on the side stream ends.
   - That broadcast belongs to the *other* micro-batch, so this looks like a **false dependency**.
   - The host is far ahead. On stage 0, `cudaGraphLaunch` runs a median **62 ms** before the graph starts on the GPU, so launch and scheduling are not on the critical path.
2. **Decode already runs FULL graphs.** "FULL 0 / PIECEWISE 20-40" is a parser artefact [OFF; parsers in §3].
   - Every decode scope has exactly **1** `cudaGraphLaunch`: 1,036 kernels on stage 0, and 951 kernels on stage 1 plus 3 draft graphs of 72 / 65 / 65 kernels.
   - The "eager compute" that makes the parsers say PIECEWISE comes from three sources, all outside the graph:
     - input prep;
     - the PP-deferred RecoverSSM commit chain;
     - `_build_qsa_metadata_kernel`, whose name contains "qsa".
   - So arm G, as written ("why decode ran 0 FULL forwards"), has nothing to fix in the target forward.
3. **On stage 0, the RecoverSSM commit graph can never be used.**
   - On a non-last PP rank the commit is PP-deferred, and the code skips the graph path when it is deferred: `if not deferred and self._commit_graphs.run(` at [MS] `vllm/v1/worker/gpu/model_states/recoverssm.py:139`. The docstring at `recoverssm_commit_graph.py:30-37` says the same.
   - B2 captured commit graphs only on the PP1 workers.
   - So arm C can act only on stage 1, which is not the binding stage. The measured commit / post kernels are about 1.0 ms per tick on stage 0 and about 0.93 ms on stage 1 [OFF].
4. **The PP2 micro-batch split is not enforced. It comes from the decode-parity cadence** (each request is eligible every `pp_size` steps).
   - In the SEED2 profile window all 8 streams were in one micro-batch (`generation_8(24)`) and the other step was empty (`generation_0(0)`) [OFF on S2T].
   - The stage-0 stall did not occur there.
   - An A/B arm can land in a different split from its base by chance, so NB4 must log the split per step.
5. **Arm P has a small upper bound.**
   - In-engine world-2 all-reduces at T 16 are 51 per stage-0 forward, median 51 µs, p90 79 µs: 3.0-4.2 ms of NCCL time per stage-0 tick [OFF on B1T].
   - NB3's P5 is 30.7 vs 49.0 µs at 82 KB, so it saves at most about 0.9-1.1 ms per stage tick (≈ 2 %). That is at the 2 % noise gate.

---

## 1. Measured facts relevant to the tick

### 1.1 Offline re-analysis of the real traces (new, [OFF])

**Method:** kernels are joined to their launching runtime call by `args.correlation`. Every kernel of one graph replay shares the correlation id of its `cudaGraphLaunch`, so graph spans, intra-graph gaps and launch-to-start latency can be measured per replay.

**Stage 0, rank pp0_tp0, B1T, 33 steady periods (pp0_tp1 is the same within 0.1 ms):**

| Quantity | Median | Min-max |
|---|---|---|
| Period between `execute_context_0(0)_generation_4(16)` scopes (= tick) | 49.83 ms | 45.24-51.81 |
| CPU execute scope | 12.23 ms | 12.07-15.69 |
| CPU: scope start → `cudaGraphLaunch` | 10.25 ms | 10.09-13.69 |
| CPU: `cudaGraphLaunch` call itself | 1.7 ms | — |
| `cudaGraphLaunch` (CPU) → graph's first kernel on GPU | **62.0 ms** | 57.2-64.8 |
| Graph GPU span | 47.64 ms | 45.53-49.35 |
| Graph kernel sum | 28.03 ms | 24.80-29.75 |
| Kernels per graph | 1,036 | fixed |
| **In-graph stall** (one gap > 5 ms per replay) | **21.70 ms** | 20.29-22.92 |
| Stall start, measured from graph start | 4.70 ms (after 8 Marlin-MoE calls = 4 layers, 5 QSA kernels, 3 GDN verify) | 3.90-5.06 |
| Stall end − end of the long sampled-token broadcast | **0.03 ms** | 0.02-0.25 |
| Long broadcast start − stall start | −6.7 ms (the broadcast is already spinning while layers 0-3 run) | −7.0 to −5.8 |
| Replays with the stall | 39 / 40, always kernel index 172, always graph stream 2338 | both TP ranks |
| Pre-graph eager kernels per tick | 86 kernels, 1.28 ms sum, ≈ 1.9 ms span | — |
| Eager kernels launched inside the scope | 82 | — |

The 82 eager kernels inside the scope are:
- elementwise / index input prep;
- `prepare_commit_plan` ×4, `compact_conv_state` ×4, `_postprocess_recoverssm_align` ×4, `_commit_gdn_state` ×3 (the deferred commit chain, ≈ 1.0 ms GPU);
- `_build_qsa_metadata` ×2;
- `_post_update`, `_scatter_num_accepted`, `_combine_sampled_and_draft_tokens`, `_gather_block_tables`, `_compute_slot_mappings`, `pre/post/precopy_mamba_align_fused`.

**NCCL on stage 0:**
- The NCCL time inside the graph is 3.02 ms per tick (tp0) and 4.16 ms (tp1).
- That is 51 world-2 `AllReduce_Sum_bf16_RING_LL` calls per forward (2,040 / 40), all graph-launched, never overlapping each other, spread over 49 graph streams.
- Duration p10 / p50 / p90 / max: 47.6 / 51.0 / 79.1 / 312.6 µs.
- One `SendRecv` per tick (hidden-state send), 0.05 ms.

**Side stream 31 (the `pp_broadcast` communicator) runs three `Broadcast_RING_LL` per tick: 27.9 / 0.03 / 13.5 ms.**
- These are the three broadcasts in `PPHandler.receive()`: sampled tokens, the combined counts and the draft tokens ([MS] `vllm/v1/worker/gpu/pp_utils.py:219-260`).
- Their durations are pure spin-wait for stage 1. The 41.6 ms per tick of "broadcast time" is the whole reason NB1 reported "NCCL 63.5 %".
- The 13.5 ms third broadcast is the time stage 1 spends on MTP drafting.

**Stage 1, rank pp1_tp0, B1T, a typical tick (≈ 49.4-51.6 ms):**

| Segment | ms |
|---|---|
| Recv wait for stage 0's hidden state (`SendRecv`) | 7.3-9.0 (median 8.21) |
| Target graph (951 kernels, kernel sum 25.6-25.9) | 23.1-23.7 |
| Post-target eager (sampling / rejection, 102 eager kernels, 4.2 ms of kernels), incl. the target logits `AllGather` (1.44-1.45 ms) → sampled-token broadcast | ≈ 4.6 |
| MTP drafting: 3 draft graphs (72 / 65 / 65 kernels) | 4.40-4.60 + 3.92-4.03 + 3.95-4.00 ≈ **12.4** |
| Draft-token broadcast, then the next recv | — |

- CPU execute scope on stage 1: 8.6 ms.
- Graph CPU launch runs 80-117 ms ahead of the GPU.
- Stage-1 NCCL: 55 all-reduces per tick (2,200 / 40, median 50 µs) and 11 all-gathers (one of 1.46 ms; the draft ones are ≈ 75 µs).

**Reading (DERIVED from the rows above):**
- The tick is a chain: s1 target + sampling of micro-batch A (≈ 28 ms) → the stage-0 stall releases → stage-0 layers 4-24 of micro-batch B (≈ 21.7 ms) + send → s1 receives B.
- So the tick ≈ 23.4 + 4.6 + 21.7 + ~1 ≈ 50 ms.
- If the false dependency is removed:
  - stage 0 is busy about 29 ms per tick and stage 1 about 41-42 ms (23.4 + 4.6 + 12.4 + ~1);
  - so the tick is about **41-42 ms (−16 to −18 %, about +19-21 % tok/s at 8k@8, est.)** and stage 1 becomes binding;
  - MTP drafting (12.4 ms, ≈ 30 % of stage 1) is then the largest movable term.

**SEED2 contrast (S2T, K2 + INT8 head + seqs 16 + CAPS16):**
- Every stage-0 step alternates a full micro-batch (`execute_context_0(0)_generation_8(24)`, period 73.8-77.4 ms) with an **empty** one (`generation_0(0)`, period 1.8 ms). So the 8 streams were parity-locked 8/0.
- The stage-0 graph spans 31.2 ms with a kernel sum of 34.75 ms. **There is no in-graph stall**, because no other micro-batch's broadcast is in flight.
- Graph launch → GPU start is 74.5 ms (the host is still ahead).
- Stage 1 per full step: target graph 27.2 ms (968 kernels), 6.7 ms eager post, 2 draft graphs (3.9 + 3.3 ms, K2), then a recv wait of 33.8 ms.
- Per request cycle: SEED2 8/0 = 77.6 ms for 8 requests, B1 4/4 = 2 × 49.8 = 99.6 ms. In the current engine, the serialised 8/0 layout is *not slower* per cycle than the "overlapped" 4/4 layout. This is consistent with the false-dependency reading.
- NB2's parser reported `gen_reqs_hist {8: 20}` for this window ([E25R] `E25_GPU_wave1_NB3_NB2SMOKE_2026-10-31.md:67-70` covers only the category shares).

**Not shown by the traces (limits):**
- There is no EngineCore / scheduler trace: `ignore_frontend: true`, and only the 4 worker files exist.
- The `gloo:send` annotations (2 per tick on stage 0, `gloo:recv` on stage 1) have durations of about 149 ms. These are async work-object spans, not CPU blocking.
- Cross-process clocks are aligned through `baseTimeNanoseconds`, so a skew at the ms level is possible ([E25B] `nb2_code_export.py:2502` note).

### 1.2 Decode tick composition and step model (earlier runs)

- **RC2 steady windows (FP8, K3, 90 s)** ([RC2] §6.1, lines 341-366):
  - 40k@1 / @2 / @4 / @5: step 44.1 / 30.3 / 37.5 / 40.3 ms;
  - 8k@8: 48.2 ms;
  - 90k@4 / 116k@3 / @4: 37.8 / 34.4 / 37.6 ms.
  - 1 stream: 44.1 ms per traversal. 2 streams: 30.3 ms per step, 1 request per step (2 streams = 1.53× one stream).
- **Step fit** ([RC2] §6.2, lines 425-431; PLAN §1.1 lines 164-168):
  - step_ms = 25.60 + 5.79·r, or 24.21 + 0.1758·D(4r), with r = C/2;
  - context length costs almost nothing (r = 2 at 40k / 90k / 116k gives 37.5 / 37.8 / 37.6 ms);
  - the 24 ms floor is FITTED and "may contain stage idle and hand-off, not only compute" (PLAN:166-167).
  - **[OFF] gives a concrete reading of that floor:** at 8k@8 about 21.7 ms of the 49.8 ms tick is the in-graph stall.
- **RC2 stage-imbalance estimate** ([RC2] §2.6 lines 143-144, finding 15): light stage idle ≈ 16.5 ms per step at 1 request per micro-batch (44.1 vs 30.3). The log could not say which stage is heavy. **[OFF] shows stage 1 is the heavier compute stage, while stage 0 is the critical-path stage because of the stall.**
- **NB1 R2** ([E25R] `E25_GPU_NB1_R2_2026-10-31.md`):
  - steady 8k@8 B1 261.0 tok/s, tick 48.2 ms, accept 3.15; B3 254.6 / 48.3 / 3.08;
  - 40k@8 B3 249.5 tok/s, tick 49.9 ms;
  - "the step time is the same in both boots, so the pin changes capacity only" (lines 16-25).
  - Trace (lines 48-73): stage 0 critical path 50.6, busy union 27.2, summed 29.5 ms; stage 1 31.6 / 23.3 / 25.8.
  - **Corrections from [OFF]:**
    - stage 0's 50.6 ms "critical path" = the ≈ 1.9 ms of pre-graph eager kernels + the 47.6 ms graph span including the 21.7 ms stall;
    - stage 1's 31.6 ms counts only the kernels launched *inside its scope* (target graph + eager post), not the 3 draft graphs or the recv wait;
    - the result doc calls the window "40k-class", but `pp2_profile` uses `PROMPTS["8k"]` ([E25B] `nb1_code_export.py:2982`).
- **NB1 wave 1** ([E25R] `E25_GPU_wave1_NB0_NB1_2026-10-31.md:39-40`): 8k@8 280.0 tok/s with "step 90.2", a cycle number (2 ticks), not a tick. **Tick vs cycle labels differ between docs; NB4 must name both.**
- **NB2** ([E25R] `E25_GPU_wave1_NB3_NB2SMOKE_2026-10-31.md`):
  - K2 + INT8 head 8k@8: 242.6 / 240.9 tok/s, tick 42.7 / 43.0 ms, vs CTRL K3 259.1 tick 48.9 (lines 47-49, 58-63);
  - per-position acceptance: K2 0.865 / 0.723; K3 0.867 / 0.716 / 0.586;
  - tokens per step 10.35 vs 12.68.
  - **Caveat from §1.1:** "tokens per step ÷ a" cannot tell a 4/4 split from 8/0 if empty steps count as iterations. Whether `vllm:iteration_tokens_total` counts the empty step is UNVERIFIED.
- **PLAN per-tick estimate** (PLAN:103-120; A7 §5.2 lines 213-237), now superseded by measurement:
  - MoE 4.7-21; dense 6.6-12; vocab heads 10.2 (BF16) / 6.4 int8 / 4.5 int4; TP2 AR 1.5-3.8; logits gather 2-3; PP hand-off 1-3; QSA 2-5; GDN + RecoverSSM 1-3; PLE < 0.5; guard 0.3 ms.
  - Measured now: TP2 AR 3.0-4.2 ms per stage tick (incl. skew); logits gather ≈ 1.45 ms target + ≈ 0.8 ms drafts; MTP drafting 12.4 ms.
- **NB3 D5 Marlin MoE** ([E25R] `E25_GPU_wave1_NB3_NB2SMOKE_2026-10-31.md:8-14`):
  - T 16: 771 µs per layer;
  - "24 MoE layers per stage ≈ 18.5 ms of a 42.7-48.2 ms tick";
  - 204-252 GB/s;
  - config tuning 0.4-1.1 % (not a lever).
- **Merlin 3090 references:**
  - stage 0 7.0 + stage 1 11.0 + 2.0 ms "neither computes" per step, single stream (A7:68; A3:74);
  - v2 history "serial (stage 0 ~9.4 ms then waits; stage 1 waits ~7.8 ms then ~13.3 ms)" (A3:80);
  - stage-1 split: dense 3.9 ms, BF16 vocab head 2.25 ms (5 calls), Marlin MoE 1.3, custom AR 0.6 (A7:88).

### 1.3 PP2 bubbles, overlap and micro-batching

- **Interleave mechanics** (A7 GOOD 6, lines 138-140; PLAN:129):
  - `max_concurrent_batches` = pp + 1 ([MS] `vllm/config/vllm.py:607-617`);
  - a request is decoded only every `pp_size` steps ([MS] `vllm/v1/core/sched/scheduler.py:792-796`).
  - RC2: "only 2 micro-batches are ever in flight" ([RC2]:143).
- **The split comes only from parity** ([MS] `vllm/v1/core/sched/async_scheduler.py:18-23, 52-57`: `next_decode_eligible_step = current_step + decode_stagger`, `decode_stagger = pp_size`).
  - Nothing caps a micro-batch at C/2.
  - Which half a request lands in depends on the step where it first decoded (prefill end / prefix-hit timing).
  - Observed 4/4 in B1T and 8/0 in S2T.
- **Sampled-token / draft broadcast design** ([MS] `pp_utils.py:59-110, 202-260`):
  - a dedicated `pp_broadcast` sibling communicator ("so it does not serialize on the wire with the inter-stage hidden-state p2p send/recv");
  - a side `broadcast_stream`;
  - step T's recv is consumed at step T + pp_size (a FIFO pre-seeded with pp_size `None` entries);
  - the main stream waits on the recv event only at consume time;
  - the last rank broadcasts sampled tokens and counts after sampling, and the drafts after the speculator (`broadcast_drafts`).
- **PP hand-off:**
  - lazy 2-rank NCCL communicators at the first request ([RC2] §2.10 line 183);
  - hidden × hc_count = 10,240 bf16 = 20,480 B per token, so 328 KB per micro-batch at T 16 (A7:222);
  - measured `SendRecv` kernel 0.05 ms on the sender; the stage-1 recv waits 7-9 ms [OFF].
  - The PP metadata goes over gloo (`gloo:send` / `gloo:recv` annotations [OFF]); A3:223 shows the deadlock (`irecv_tensor_dict`) when async scheduling is off.
- **3090 "serial stages"** was single-stream; the multi-stream overlap question was RC2's (PLAN:129-130). RC2: overlap holds in the step counts. [OFF]: in the 4/4 case the overlap is broken by the stall.
- **PP2 structural costs (est.):**
  - two half-batches touch +31 % distinct experts at C 16 (PLAN:101; A7:247);
  - single-batch vs PP2 MoE ceilings (06 verification:58-61).

### 1.4 NCCL, all-reduce and broadcast costs

All measured on Kaggle 4×L4 unless noted.
- **Environment** (A7:29-34; [RC2] §2.10 lines 178-185):
  - NCCL 2.29.7 (not the container's 2.25.1);
  - PCIe gen 3 ×16, all PHB, one NUMA node, 46-48 vCPU;
  - tp / pp / ep groups all `['PYNCCL']`;
  - symm-mem "capability 8.9 not supported"; FlashInfer AR not supported at world 2; custom AR off; `NCCL_P2P_DISABLE=1`, `VLLM_SKIP_P2P_CHECK=1`;
  - the physical transport is not logged without `NCCL_DEBUG=INFO`. NB1 B1 set `NCCL_DEBUG=INFO, NCCL_DEBUG_SUBSYS=INIT,NET` ([E25B] `nb1_code_export.py:1290`); the parsed result is in `nb1_results.json` (key `nccl`).
- **World-4 numbers (27B era)** (A7 §1.1 lines 51-69; PLAN §2.4 lines 243-252):
  - 983 KB in a graph: 343.9 µs (E14) / 336.2 µs (E23);
  - 64 KB 49.7 µs; 240 KB 134 µs;
  - eager 8 KiB 90.1, 80 KiB 107.0;
  - P2P 5.5× slower;
  - `NCCL_SHM_USE_CUDA_MEMCPY=1` slower and hangs in capture;
  - CTA 4/8/16/32 all slower;
  - 2 channels.
- **World-2 in-engine (new, [OFF] §1.1):** 82 KB TP2 all-reduce median 51 µs, p90 79 µs, in-graph, 51-55 per stage tick.
- **NB3 D7 world-2 microbenchmark** ([E25R] `E25_GPU_wave1_NB3_NB2SMOKE_2026-10-31.md:27-30`):
  - P5 vs NCCL 1.39-1.68× at 5-123 KB; 1.25× at 246-328 KB; 1.06× at 0.5-0.66 MB;
  - T 16 (82 KB): NCCL 49.0 vs P5 30.7 µs;
  - bit-exact vs an ordered fp32 reference;
  - status **PASS_BOUND_ONLY**: host-arm overlap 0.43-0.97 < 0.5 when both pairs run (concurrency unverified).
- **NCCL kernel time = wait + transfer.** Use min-over-ranks for the transfer and max − min for skew (E14 `Notes/15 … NCCL bench and trace summarizer (2026-10-23).md` §2 "Why the minimum over ranks"; A1:281; [E25R] wave-1 NB2 line 69: "an upper bound on communication cost, not a measured cost").
- **The 27B comm share** (A7:66-67): SGLang TP4 51 %; vLLM 35.6-35.7 % (unchanged by graph mode).
- **LL128:**
  - +6.2 % on jon-nielsen's 4×3090 no-P2P kit (PLAN:251);
  - **but forced `NCCL_PROTO=LL128` was dropped from NB3: NVIDIA warns of data corruption on unsupported platforms** (`GPT research inputs/03_RC2_AUDIT_AI2_VERIFICATION.md:14`; 06 verification:12).
- **Other notes:**
  - `NCCL_CUMEM_ENABLE=0` (todi), `NCCL_NTHREADS=512` untested (PLAN:251);
  - SGLang forces `CUDA_DEVICE_MAX_CONNECTIONS=8`; vLLM leaves the CUDA default (8) (E14 Notes/15:59).

### 1.5 CUDA graph modes (FULL / PIECEWISE / eager) and capture

- **Flash-Next on Merlin (RC2 §2.3 lines 109-115; NB1 `nb1_results.json` `sections.B*.graphs`):**
  - `target=FULL_AND_PIECEWISE`;
  - captures: FP8 PIECEWISE 10/10 + FULL 6/6 (BF16 7 + 3);
  - draft `prefill_cudagraph_manager=FULL_AND_PIECEWISE`, `decode_cudagraph_manager=FULL_DECODE_ONLY`;
  - no downgrade line;
  - `breakable=OFF`, `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, V2 runner.
  - B2 (seqs 16, captures to 64): FULL 9 + PIECEWISE 13.
- **FULL capture sets** ([RC2] §2.5 lines 123-128):
  - FP8 `[1,2,3,4,6,8,12,16,24,32]`;
  - FULL uniform-decode graphs at multiples of 1 + K = 4: {4,8,12,16,24,32} = 1/2/3/4/6/8 requests per forward;
  - so up to 16 streams with a balanced split;
  - `compile_ranges_endpoints [1024]`.
- **Capture sizes count tokens** (A7 F11 line 188; A1:322):
  - [MS] `cudagraph_utils.py:268-347`;
  - capped at `max_num_reqs × (1+K)`.
  - Merlin `registry.capture_sizes_for(16, K=3)` → `[1,2,3,4,6,8,12,16,24,32,40,48,64]` ([E25B] `nb1_code_export.py:1283`).
- **Measured graph residency [OFF]:** 1 graph launch per decode scope on both stages (+3 draft graphs on stage 1). The NB1 parser itself recorded `per_forward_median_graph_launches 1.0`, `eager_kernel_share 0.08` and verdict PIECEWISE (`nb1_results.json` `sections.B1.nb1_b1_fp8.profile.trace`).
- **Routing hook confirms FULL replays** ([E25B] `nb1_code_export.py:2985-2988`): "FULL-graph decode replays are not visible to a Python hook". The NB1 routing ids therefore come only from prefill / uncaptured steps; decode routing needs `--enable-return-routed-experts`.
- **PIECEWISE breaks at every QSA layer** (QSA is a split op) (PLAN:240; A1:321). In our traces that would show as about 7 graph launches per stage forward; we see 1.
- **27B lessons** (A1:137-138, 190-208; A7:99-102):
  - FlashInfer downgrade to PIECEWISE under spec decode;
  - Merlin 27B draft `decode:NONE`;
  - RecoverSSM commit "no graph … (n=24, 0 captured); running staged-eager";
  - PR #60532 G3 FULL: +3.0 % SMOKE, +2 % FULL vs A/A;
  - Triton FULL G1 / G2: −7 % ([E23R] `E23_FULL_esfersami50_v1_2026-10-31.md:13`; PLAN:522-524);
  - eager share 25.5 % → 5.2 %, graph launches 1,809 → 116.
- **External** (E24 `Resources/sources/research/agent_reports/pcie_parallelism_and_graphs.md:142-143`):
  - MTP k=1 + FULL_AND_PIECEWISE deadlock at a bonus-token-only shape (#42271, v0.20.1, not with FULL_DECODE_ONLY);
  - silent corruption with MTP + DCP + FULL_AND_PIECEWISE;
  - GB10: a drafter FULL prototype saved only ~1.3 ms of 18 ms.
- **#57759:** one FULL capture failure (`cudaErrorNotPermitted`) on 4×L20 sm_89 under host load; workaround PIECEWISE; the author retracted the PLE analysis (`01_VERIFICATION_by_agent.md:35, 111`).
- **#58489 / #58441:** pinned PLE prefetch ids corrupted by graph-pool reuse. Fixed in Merlin (`01_VERIFICATION_by_agent.md:34`; PLAN:223 lists #58489 as included).
- **Graph capture cost:**
  - 4-6 s per boot;
  - 0.33 / 0.43 GiB (B1, B3) vs 0.44 / 0.56 GiB with 13 + 9 captures (B2) (`nb1_results.json` `pool.rank_graph_gib`).
- **Other:**
  - torch.compile per boot: PP0 13.8-14.2 s, PP1 12-13 s, eagle 3.3-4.9 s ([RC2] §4.3);
  - `Not enough SMs to use max_autotune_gemm mode` ([RC2]:128).

### 1.6 RecoverSSM commit

- **Modes** ([MS] `recoverssm_commit_graph.py:1-50`):
  - `eager` (default; no graphs captured at boot), `staged` (control arm), `graph`;
  - runtime flip through `set_recoverssm_commit_mode` ([MS] `vllm/v1/worker/gpu_worker.py:1781-1799`, via dev-mode `/collective_rpc`);
  - counters `eager_steps / staged_steps / replays / captures / fallbacks` (`recoverssm_commit_graph.py:118-130`);
  - bit-identical by design;
  - graphs are captured only during the engine's capture phase, "from the dummy uniform-decode metadata of each FULL forward capture";
  - **graph-eligible only when uniform all-decode, all groups the same n, n equals the batch's request count, spec rows 0..n-1. PP-deferred commits, prefill / mixed batches and uncaptured keys run eager.**
- **Deferred = non-last PP rank** ([MS] `model_states/recoverssm.py:18-28, 54-86, 139`; `pp_utils.py:38`):
  - stage 0 detaches its commits and runs them pp_size steps later when the accepted counts arrive;
  - the graph path is skipped (`if not deferred and …`).
- **B2 log facts** ([B2L]):
  - 366-367: `mode=graph: no graph for this step (n=16, 0 captured); running staged-eager` (PP1, 16:41:00, *before* capture: profile / dummy run).
  - 373-390: **9 captures on PP1 only: n = 16, 12, 10, 8, 6, 4, 3, 2, 1**.
  - 510-511: `mode=graph captured=9 (device cuda:3/2, first replay n=1)`.
  - 518-521: all 4 workers `mode=eager (was graph)` at 16:41:17, flipped by NB1's own `/collective_rpc` call: by design, "flipped to eager at once" ([E25B] `nb1_code_export.py:1291, 3075`).
  - **So the B2 oracle (16:41:17 onward) ran with the commit in eager mode. The commit graph is not implicated in the B2 FAIL_HIGH_MARGIN.**
  - The spec's "captured only n=16" and the result doc's "only n=16 was captured" ([E25R] `E25_GPU_NB1_R2_2026-10-31.md:45`) are wrong: 9 keys were captured on PP1, none on PP0 (by design).
- **The runtime A/B harness exists but never ran:** `commit_ab()` ([E25B] `nb1_code_export.py:1842-1866`): eager → staged → graph 8k@8 windows, per-arm counter deltas, greedy graph-vs-eager row, reset to eager. It was skipped because B2's oracle failed.
- **Measured commit cost [OFF]:**
  - stage 0 deferred chain ≈ 1.0 ms GPU per tick (15 kernels, eager, inside the pre-graph prep);
  - stage 1 `commit/post` ≈ 0.93 ms GPU per tick.
  - NB1 category "recoverssm_commit 2.4 %"; NB2 SEED2 `commit_eager_per_scope 11` on stage 0 and 0 on stage 1 (the commit eager kernels land outside the stage-1 scope).
- **RC2 log:**
  - `GDN RecoverSSM commit context: 7/6/6 layers` (PP0), `5/6/6` (PP1);
  - `RecoverSSM + pipeline parallel: Mamba prefix checkpoints are published for prompt positions only (3 groups)`;
  - `GDN spec decode kernel: triton`;
  - `PLE RecoverSSM commit context: 1 layers, history 9` ([RC2] §2.2 line 95, §2.3 line 114).
- **27B Merlin:** commit staged-eager n=24 / n=4 with 0 captured (A7:101).

### 1.7 MTP drafting costs

- **[OFF] B1T stage 1:** 3 draft graphs ≈ 12.4 ms per tick (4.5 / 3.9 / 4.0 ms), plus the post-target sampling ≈ 4.6 ms. Consistent with the stage-0 third broadcast spin of 13.5 ms.
- **[OFF] S2T (K2):** 2 draft graphs 3.9 + 3.3 ms; post-target eager 6.7 ms (top kernel `_topp_sb_step_kernel`).
- **Acceptance** ([RC2] §2.8 lines 152-168):
  - FP8 whole boot 0.834 / 0.660 / 0.516 (MAL 3.010);
  - natural row alone 0.772 / 0.558 / 0.390 (MAL 2.72);
  - decay 3.18 → 2.69 over 33 min.
  - B2 early probes 0.760 / 0.570 / 0.430 ([B2L]:527).
- **`max_num_scheduled_tokens is set to 1024`** is advisory; the 1,024 budget is shared by prefill and verify tokens ([RC2] §2.9 lines 171-176).
- **K2:** −7 % at 8 streams (tokens per step −18 %, tick −12 %). The INT8 head does not lower acceptance ([E25R] wave-1 NB2 lines 58-63).
- **DRAFT_HEAD int8:** −6.5-7.2 % ms/token on 3090 (A3:72, 126; A7:90). The draft head is full-vocab BF16 in 3.0.0; vocab heads are estimated at 10.2 ms per tick at BF16 (A7:219).
- **Other:**
  - MTP layer FP8 KV uses the default scale 1.0 (draft only) ([RC2]:447; `03_RC2_AUDIT_AI2_VERIFICATION.md:7`);
  - "Enabling num_speculative_tokens > 1 will run multiple times of forward on same MTP layer" ([RC2]:449).

### 1.8 Scheduler, async scheduling and host overhead

- **The host is not the bottleneck at 8k@8:** stage-0 CPU leads the GPU by 62 ms, stage 1 by 80-117 ms. CPU time per tick: 12.2 ms (stage 0), 8.6 ms (stage 1) [OFF].
- **Stage-0 CPU split:**
  - 10.25 ms prep before `cudaGraphLaunch`, including about 20 `cudaMemcpyAsync` (H2D input staging) and 82 eager launches;
  - `cudaGraphLaunch` 1.7 ms CPU.
  - If the tick drops to ≈ 41 ms, stage-1 CPU (8.6 ms + draft launches) and stage-0 CPU (12.2 ms) still fit.
- **Async scheduling:**
  - on by default; `--no-async-scheduling` **deadlocks PP** (gloo timeout after 30 min, 36/36 fail; stage 2 blocks in `irecv_tensor_dict`) (A3:223; PLAN:380);
  - on 27B / 0.19, async off gave no benefit (`Experiment 19 …/Experiment Results/E19_GPU_run1_results_2026-10-27.md:18, 41`).
- **Merlin host switches** (A3:164-168; PLAN:340): `VLLM_GDN_DECODE_FASTPATH` (slices instead of CPU-mask indexing + H2D), `VLLM_GDN_SHARED_ALIGN_INDEX`, `VLLM_TARGET_SEQ_LENS_ASYNC` (no seq_lens device sync). `VLLM_DRAFT_BOUNDED_PLAN` and `VLLM_GDN_BA_AUX_STREAM` are dropped by the launcher. Each needs a bitwise or greedy check.
- **Guard reader:**
  - ~0.3 ms per decode step on 3090 (A7:91);
  - **`GUARD=off` is refused by the launcher** (A3:136: "raise, warn (off refused)");
  - PLAN D8's `VLLM_E2_GUARD_MODE=off` is therefore not a plain env arm.
- **Scheduler retention / admission:**
  - the cost-aware cold-miss defer (up to 120 s) plus 16 retained finished turns stalled 90k@3 ([RC2] §2.2 lines 98-107, finding 9);
  - `kv_cache_usage_perc` is not live demand.
  - This is irrelevant to the tick, but it does affect whether a window has "real running == level".
- **Engine-core timing:** no per-step scheduler timing was ever recorded in our runs. The only step clock is Δ`vllm:iteration_tokens_total_count` (RC2 `steps source iteration_histogram`, [RC2]:357).

### 1.9 Launch overhead

- Stage-0 eager launches: 82 per tick inside the scope plus the graph launch; about 1.9 ms of GPU span for the pre-graph kernels [OFF].
- 27B G0 PIECEWISE: 65 graphs + 160 kernels per forward (144 eager compute); G3 FULL: 1 graph + 16 kernels (A1:137-138).
- Archive "graphs 48 → 129 tok/s at 8 seqs" vs "#59548 PIECEWISE ≈ 1.01× eager" (A1:353): conflicting, neither on Merlin.

### 1.10 Profiler and trace methods: what worked and what failed on Kaggle

- **Worked: the vLLM torch profiler over HTTP:**
  - `--profiler-config {"profiler":"torch","torch_profiler_dir":…,"torch_profiler_with_stack":false,"ignore_frontend":true}` ([E25B] `nb1_code_export.py:1286-1287`);
  - `POST /start_profile {"delay_iterations":20,"max_iterations":40}`, sleep, `POST /stop_profile`, wait 20 s ([E23] `src/e23_lab.py:902-919`);
  - 4 rank files `dp0_pp{p}_tp{t}_dcp0_ep{e}_rank{r}.<id>.pt.trace.json.gz`, about 2.0-2.3 MB each for 40 iterations;
  - the profiled window runs as its own window after settling (`profiled_load`, `e23_lab.py:922-990`; streams T 1.0, seeds 4321+);
  - `/start_profile` needs dev endpoints (log: "Profiler with mode 'torch' is enabled in the API server", [B2L]:431).
- **nsys is NOT on the image.** NB1 `profiler_tools()` searched `ncu`, `nsys`, `/usr/local/cuda/bin/{ncu,nsys}` and the Merlin CUDA home: only `ncu` 2025.1.1 was found (`nb1_results.json` `sections.B1.nb1_b1_fp8.profiler_tools`; [E25B] `nb1_code_export.py:2387-2407`).
  - "ncu / nsys cannot attach to a running vLLM server": DRAM bytes need a boot launched under ncu.
  - Whether ncu is allowed the DRAM counters (`ERR_NVGPUCTRPERM`) is unknown ([E25R] `E25_NB3_CPU_kits_v1_2026-10-31.md`, finding 2).
  - **The wave-1 note "NB4 needs a timeline (nsys)" ([E25R] wave-1 NB2 line 70) cannot be met as written. The torch trace is enough (see §1.1).**
- **Parser bugs, known:**
  - The first E23 parser matched `gpu_model_runner` scopes, giving 0 forwards. Fixed in rev 3h with `execute_context_N(N)_generation_N(N)` (A1:280; `e23_lab.py:795-801`). NB2 also matches the `execute_N_context_*(sqN…)` variant ([E25B] `nb2_code_export.py:2367-2368`).
  - GDN was filed as "other" (6.1 % real vs 0.2 %). Fixed by stripping `(anonymous namespace)::` and matching the base name (A7:86; `e23_lab.py:786-791`).
  - **New:** the FULL / PIECEWISE verdict counts out-of-graph PP comm (`nccl`, `nccl_sendrecv`), the deferred commit (`recoverssm_commit`) and `_build_qsa_metadata` (`qsa`) as "eager model compute". So FULL forwards read PIECEWISE:
    - NB2 `NB2_COMPUTE` includes `nccl` and `qsa` ([E25B] `nb2_code_export.py:2366, 2458-2471`);
    - NB1 adds `nccl_sendrecv`, `recoverssm_commit`, `qsa`, `moe_marlin`, `topk` to `L.COMPUTE_CATS` (`nb1_code_export.py:1349-1354`);
    - fix: count graph launches per scope (1 = FULL, > 1 = PIECEWISE), or treat only in-forward model kernels as compute.
  - **New:** NB2's per-scope graph count looks only at the scope's own thread. Stage-1 draft graphs are launched outside the execute scope, so `draft_graph_launches_est` = 0 ([E25B] `nb2_code_export.py:2518-2520`).
  - **New:** `pp2` busy percentages include the spinning NCCL kernels, so S2T shows 98.9 % / 98.7 % busy ([OFF]: the S2T `pp2` block). The busy union must exclude NCCL wait or report it apart.
  - `L.trace_summary` reads only the rank-0 file (`e23_lab.py:820`). Its "NCCL 63.5 %" is rank-0 stage-0 spin time.
  - NB1 `microbatch_costs` (`nb1_code_export.py:2774-2819`) gives a correct "critical path" only for kernels launched inside the scope. The stage-1 drafts and recv waits are invisible to it.
- **E14 SGLang trace summariser** (E14 `src/e14_trace_summary.py`; Notes/15 §2): streaming parser, union busy, min-over-ranks collectives, category regexes. It is built for SGLang file names (`TP-<n>`), so it needs adapting for vLLM `dp0_pp*_tp*` names.

### 1.11 P5: implementation and integration status

- **Source:** E23 `prep/P5_host_allreduce/` (README, `src/host_ar.cu`, `host_ar_binding.cpp`, `build_host_ar.py`, `bench_host_ar.py`, `tests/sim_protocol.py`, `tests/emulate_numerics.py`). Copied into NB3 (`[E25B] nb3_export/p5_src/`, `nb3_export/bench_host_ar.py`, `nb3_common.py`).
- **Ops:**
  - `init(rank, world, name, max_bytes, device, timeout_ms)`; `all_reduce(handle, inp, out, mode, blocks)`, stream-ordered and graph-capturable; `status`, `epoch`, `destroy`;
  - a Meta kernel for torch.compile;
  - worlds 2 / 4 / 8;
  - mode 0 native fp32 accumulate, mode 1 INT8 (README:29-40).
- **Build:** done on the Merlin env with GCC 13 / nvcc 13.0, SASS sm_89 only, in 28 s ([E25R] `E25_NB3_CPU_kits_v1_2026-10-31.md`, merlin kit row).
- **Measured:**
  - world 4, 983 KB: 0.93-0.94× (slower); INT8 1.41-1.43×;
  - world 4, 64 / 240 KB: 1.42× / 1.34× ([E23R] SMOKE rev3g lines 50-60; E23 FULL line 15);
  - world 2: see §1.4. **No engine integration exists anywhere** (no vLLM communicator hook in any builder).
- **Integration point (SOURCE)** [MS] `vllm/distributed/device_communicators/cuda_communicator.py`:
  - backend list and log at 289-360: `FLASHINFER_PCIE_IPC, FLASHINFER, NCCL_SYMM_MEM, QUICK_REDUCE, AITER_CUSTOM, CUSTOM, SYMM_MEM, PYNCCL`;
  - the `all_reduce(self, input_)` dispatch chain at 362-440 (out-of-place returns; PyNCCL is the last branch);
  - a P5 branch before the PyNCCL branch, limited to the tp group and sizes ≤ ~330 KB, is the natural hook.
- **PLAN gate** (PLAN §4.1 item 7, lines 449-455; `PENDING_BATCH_FIXES.md:36` N3-9): P5 adoption only after a full-engine A/B (2 repeats) + 40k / 90k correctness rows + 1000+ graph replays + both-pair contention evidence. `PASS_BOUND_ONLY` scores 0 (N3-11, line 42).

---

## 2. Levers and negatives the NB4 spec does not cover or contradicts

1. **Missing cause category: the in-graph stall.**
   - The spec splits the stage-0 tick into GPU busy / NCCL wait / host launch gap / scheduler-IPC wait. The measured 21.7 ms is none of these: the GPU queue holds the graph (launched 62 ms earlier) and the graph itself stops at a fixed node until the sampled-token broadcast kernel on the side stream ends [OFF §1.1].
   - The analysis module needs a fifth class, "**GPU idle inside a graph replay span, released by event X**". It must report the stall start node (kernel index, stream) and the nearest preceding kernel end on any other stream.
2. **Candidate lever (new, config only): `CUDA_DEVICE_MAX_CONNECTIONS`.**
   - Leading hypothesis for the stall: hardware work-queue aliasing (default 8 connections). The graph spreads over ~75 streams, plus the side `broadcast_stream`, plus the PP send stream, so a later graph branch (stream 2338) queues behind the long-spinning broadcast kernel.
   - Evidence for:
     - the release is immediate (0.03 ms);
     - the stall node is the same in 39/40 replays on both ranks;
     - the stall is absent when no other micro-batch's broadcast is in flight (S2T 8/0).
   - Alternatives, UNVERIFIED:
     - NCCL cross-communicator ordering (tp vs pp_broadcast). Weakened: 8 in-graph all-reduces ran *during* the spinning broadcast before the stall;
     - the PLE in-graph `cuStreamWaitValue32` ([MS] `vllm/model_executor/layers/ple_offload_layer.py:1-25`). Weakened: the PLE kernels run at 1.75 ms, before the stall, and a host round trip in 0.03 ms is implausible.
   - Test: one arm `CUDA_DEVICE_MAX_CONNECTIONS=32` exported before launch (Merlin pops only `MERLIN_*`, A3:115). Success = the stall is gone in the trace and the tick drops toward ≈ 41-42 ms.
   - If that fails, the patch options are:
     - post the receive broadcasts later, or on a stream / mechanism that does not hold a GPU queue: host-polled or gloo CPU broadcast, or a P5-style host-mapped flag;
     - check NCCL launch-order env (`NCCL_LAUNCH_ORDER_IMPLICIT`; its semantics are UNVERIFIED here).
   - **Expected size (est.): −16 to −18 % tick, +19-21 % tok/s at 8 streams.** This is larger than G, C and P combined.
3. **Arm G as written has no target.**
   - Target and draft decode are already FULL graphs: 1 graph launch per scope; draft `FULL_DECODE_ONLY` ([OFF]; [RC2] §2.3).
   - What is still eager is outside any model graph:
     - stage-0 input prep + the deferred commit (≈ 1.9 ms GPU span, 82 launches);
     - stage-1 post-sampling (≈ 4.2 ms of eager kernels across ≈ 18.6 ms incl. drafts).
   - A useful G arm would instead:
     - (a) fix the verdict parser;
     - (b) assert FULL coverage at the base's real micro-batch token counts (seqs 9, a 5/4 split → 20 and 16 tokens; 20 is not in the FP8 base list `[…16, 24, 32]` → the 5-request micro-batch pads to 24, which is FULL but wasteful; CAPS16 `[…16, 24, 32, 40, 48, 64]` also has no 20).
   - E23: a FULL backend can be slower (Triton FULL −7 %, PLAN:522-524).
4. **Arm C cannot touch the binding stage.**
   - Stage-0 commits are PP-deferred, so they always run eager ([MS] `recoverssm.py:139`). Commit graphs exist only on PP1.
   - Stage 1 is not on the stall-bound critical path today, so the expected engine gain ≈ 0 (NO_GAIN).
   - It becomes worth measuring only after item 2 makes stage 1 binding: then stage-1 commit/post (≈ 0.93 ms GPU + eager launches) is on the critical path.
   - **Graph eligibility needs n = the micro-batch request count.** n must be one of the FULL capture sizes ÷ 4. For seqs 9 with a 5/4 split, the n = 5 key needs capture size 20 in the list.
   - **Cheaper method:** the runtime A/B in one boot (`VLLM_SERVER_DEV_MODE=1` + `VLLM_RECOVERSSM_COMMIT_GRAPH=graph` at boot, then eager / staged / graph windows through `set_recoverssm_commit_mode`, `commit_ab()` in [E25B] `nb1_code_export.py:1842-1866`) gives a same-boot A/B, the per-arm `replays / fallbacks` counters and a greedy bit-identity row. It also saves a boot.
5. **Arm P's ceiling is ≈ 2 %.** 51-55 in-graph all-reduces per stage tick × (49 − 31 µs) ≈ 0.9-1.0 ms per stage tick [OFF; NB3 D7].
   - The all-reduce shares the critical path with the stall, so before item 2 even that may not show.
   - P5's numerics differ from NCCL: ordered fp32 sum vs ring bf16 rounding (E23 P5 README Risks item 5). The oracle vs base will not be EXACT, so NEAR_TIE / TF rows decide.
   - Expect `NO_GAIN` under the "> A/A spread and > 2 %" gate unless the stall is gone first.
6. **Micro-batch parity is uncontrolled.**
   - A/B arms can differ in split (4/4 vs 5/4 vs 8/0) by admission timing ([MS] `async_scheduler.py:52-57`; [OFF] S2T 8/0).
   - NB4 must record the per-step `gen_reqs` of every scope (the scope name has it) or a worker counter, and label windows `IMBALANCED`.
   - "tokens per step ÷ a" is not enough (§1.2 caveat).
   - The steady-window seeds and the prompt staggering decide the split. Prefix-cached identical prompts are a likely cause of parity lock (all prefills finish on the same step).
7. **Spin kernels steal nothing measurable, but they mislead.**
   - The 41.6 ms per tick of broadcast spin is reported as "NCCL 63.5 %" (stage 0) and "59.5 %" (S2T).
   - Any "NCCL share" headline must be min-over-ranks transfer time and must exclude the pp_broadcast stream (E14 Notes/15 §2).
8. **After item 2, the next levers are on stage 1:**
   - MTP drafting 12.4 ms (DRAFT_HEAD int8, PLAN D2: −6.5-7.2 % on 3090);
   - sampling + logits gather ≈ 4.6 ms (sharded sampling, PLAN D6 / A7 §5.3 item 3; target all-gather 1.45 ms);
   - target graph 23.4 ms (MoE 18.5 ms per stage by D5);
   - PP partition 25/23 → shift layers to stage 0 (stage 0 ≈ 29 vs stage 1 ≈ 41 ms busy); the capacity trade is in PLAN D6 and A3:150.
9. **Negatives NB4 must not repeat:**
   - P2P on, PCIe-IPC AR, NCCL CTA tuning, `NCCL_SHM_USE_CUDA_MEMCPY` (slower; memcpy hangs in capture) (PLAN:377; A7 §3);
   - forced `NCCL_PROTO=LL128` (corruption risk, dropped);
   - `--no-async-scheduling` with PP (deadlock);
   - prefix caching off with MTP (E19: 40k fell to 3 tok/s; PLAN:379);
   - `COUNTERS=1` (disables the fused QSA writer; PLAN:385; A3:132);
   - `--enable-hc-sp` (PP = 1 only), FlashInfer PCIe IPC (needs P2P) (PLAN:386);
   - `NCCL_GRAPH_MIXING_SUPPORT=0` while a communicator is used both eagerly and in graphs (E14 Notes/15:131). The tp group runs eager prefill and graph decode.
   - `NCCL_GRAPH_REGISTER=1` (SGLang keeps 0: deadlock risk in its LM-head all-to-all, Notes/15:132);
   - Marlin MoE config tuning (≤ 1.1 %; NB3 D5);
   - K2 / INT8 head as a speed lever at 8 streams (−7 %);
   - SSM bf16 (0 capacity at block 4096, plus the B2 oracle FAIL; parked C3);
   - counting microbenchmark gains as engine gains (PLAN §0, §4.1-7).
10. **Env that changes graph or engine behaviour:**
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=0` and `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0` are set by the plan (`nb1_results.json` `setup.merlin_plans.*.env_keys`);
    - `VLLM_USE_V2_MODEL_RUNNER=1`;
    - `MERLIN_FULL_K/QSA_SORT/TIE_DET/TIE_RECENT/TOPK_SORTED_EMIT=1` (e2-toggles lines, [B2L]:395-401);
    - `build()` pops every `MERLIN_*` env, so re-export after build (A3:114-116, 159);
    - trailing CLI args win (`Plan.argv()`, A3:116);
    - `MERLIN_SHARE_DRAFT_WORKSPACE=1` had no effect ([E25R] NB1 R2 line 44).
11. **Boot facts that bound NB4's budget:**
    - RC2 FP8 ready 999 s with a 747 s straggler from active page cache; fix = `POSIX_FADV_DONTNEED` before each boot, verified on NFS mounts (28,412 → 222 MiB) ([RC2] §4.2; [E25R] NB3 CPU kits finding 1);
    - NB1 API → KV ready ≈ 6.3 min; load 239-254 s per worker (wave-1 line 40);
    - B2 init engine 35.8 s ([B2L]:414).
    - Five boots in 150 min is tight. The offline analysis in §1.1 already answers T0's main question, so T0's profiled window can be short and C can be a runtime A/B inside T0's boot.
12. **External hints not in the spec:**
    - "CPU starvation can stall communication and launches even with CUDA graphs" (arXiv 2603.22774; E24 `pcie_parallelism_and_graphs.md:194`). Our trace shows the host ahead, so this is low risk at 8 streams.
    - Issue #63593 forced-rejection GDN divergence (sm89 PASS 6/6) as a test idea (05 verification:16).

---

## 3. Reusable files, code and tools

### 3.1 Trace analysis (validated offline on B1T and S2T)

- **Method used in [OFF]**, needing no new data. For each rank file:
  1. Load the JSON. For each event, `T = ts + baseTimeNanoseconds / 1000`.
  2. Graph replays: runtime events whose name contains `GraphLaunch`. Kernels with the same `args.correlation` are that replay's kernels, giving the GPU span, the kernel sum, the kernel count, and gaps > 5 ms with the kernel index and stream at the gap.
  3. Scopes: `user_annotation` events whose names start with `execute_`; the name gives ctx/gen request and token counts. Period = the difference between consecutive scope starts. CPU launch lead = graph GPU start − `cudaGraphLaunch` ts.
  4. NCCL: per name (`AllReduce`, `AllGather`, `Broadcast`, `SendRecv`) and per stream. Broadcast triplets per tick (sampled / combined / drafts) on the pp_broadcast stream. Gap-end − broadcast-end alignment.
  5. Stage 1: the target graph (largest kernel count), the draft graphs (small kernel counts) and the eager post-sampling in each period.
- **Scripts:**
  - `gl.py`: per replay: launch lead, span, kernel sum, NCCL-in-graph, `SendRecv`;
  - `gap2.py`, `gap6.py`: the stall and its alignment to broadcasts;
  - `s1.py`: the stage-1 period timeline;
  - `scope.py`: the eager kernels inside a scope.
  - All are in `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\nb4r\`. **Copy them into `[E25B]` before the session ends** if NB4's CPU kit should reuse them.
- **Existing parsers (to fix, not to trust as is):**
  - `e23_lab.trace_summary` / `_cat` / `CATS` / `COMPUTE_CATS` / `_scope_kind` / `profile` / `profiled_load`: `experiments/Experiment 23 - Six levers vLLM 0.32 (planned)/src/e23_lab.py:776-990` (graph proof, rev 3h scope regex, GDN fix);
  - NB1 `pp2_trace`, `microbatch_costs`, `pp2_profile`, `profiler_tools`, classifier override: [E25B] `nb1_code_export.py:2836-2911, 2774-2819, 2976-3002, 2387-2407, 1349-1354`;
  - NB2 `nb2_trace`, `_rank_trace`, `_mode`, `NB2_CATS`, `SCOPE_NB2`: [E25B] `nb2_code_export.py:2352-2523`;
  - E14 `src/e14_trace_summary.py` (+ `test_e14_nccl_bench_and_trace.py`, 33 tests): streaming parser, union busy, min-over-ranks, category regexes;
  - E23 trace regression on real traces: `_scratch/e23/gt3h/trace_regression.json`, `gate_trace/` (A1:30).
- **Fixtures for the CPU kit:**
  - B1T and S2T (above);
  - [E25B] `nb1_fixture_b1_gpu_2026-10-31.json`;
  - `nb1_results.json` / `nb2_results.json` of those runs.

### 3.2 Engine control, measurement and harness

- **Merlin dev endpoints** (`VLLM_SERVER_DEV_MODE=1`, 127.0.0.1): `/collective_rpc`, `/reset_prefix_cache`, `/start_profile`, `/stop_profile` ([B2L] route list 440-495; /start_profile 449, /collective_rpc 461).
  - NB1 helpers `rpc(port, method, *args)`, `commit_mode`, `commit_ab`, `commit_counter_delta`, `rpc_cache_info`: [E25B] `nb1_code_export.py:1798-1870`.
  - Worker extension `nb1_worker_ext.NB1WorkerExt` via `--worker-extension-cls` (`NB1_EXT_SRC`, line 1386): cache tensors, free memory, routing capture (`nb1_routing`).
- **Commit-graph boot retry** (a failed boot retries without `VLLM_RECOVERSSM_COMMIT_GRAPH`): [E25B] `nb1_code_export.py:3187-3193`.
- **Steady windows, admission, census, oracle, TF, garbage checks, request logging:**
  - `e23_lab.py` (Stream, steady, prom, wait_idle);
  - NB1 `nb1_ladder`, `window`, `quality_gate` (`nb1_code_export.py:3005-3017`), `config_diff` (3033+);
  - NB2 `graph_plan` / `microbatch_cover` / `readback` (`nb2_code_export.py:1412-1445`, 2557-2567);
  - RC2's long-row and TF code (`build_e13_rc2.py`, `e23_lab.py:502-557`).
- **Config constants:** `CAPTURE_HI`, `COMPILE_HI`, `PROFILER_CFG`, `EXT_FLAG`, `DIAG_ENV`, `NCCL_DIAG`, `COMMIT_GRAPH_ENV`, `RC2_FP8` argv tail ([E25B] `nb1_code_export.py:1283-1292`); `BOOTS` with the B3 pin `KV_FP8 + KV_STEP` (3060-3080).
- **NCCL world-2 bench:** E14 `src/e14_nccl_bench.py` (graph-captured 140-call replay, per-rank medians, transport parse; `--world`, `--arm-env`); NB3 D7 pair / solo runs (`[E25B] build_e25_nb3.py`, `nb3_common.py`; outputs `_scratch/e25_gpu_out/q27-xhigh-e25-nb3-kernels/nb3/d7_pair01_work`, `d7_pair23_work`, `d7_solo01_work`).
- **P5:**
  - E23 `prep/P5_host_allreduce/src/*` + `build_host_ar.py` (built in the NB3 merlin kit in 28 s);
  - `bench_host_ar.py` (1000-replay bit-exact test, INT8 patterns, never-hang timeouts);
  - `tests/sim_protocol.py` (co-residency deadlock model);
  - integration hook [MS] `cuda_communicator.py:362-440`.
- **Patch helpers:** the #60490 backport embedded with pre/post sha (`P60490`, `P60490_PRE` a23d5f90…, `P60490_POST` 88fe0a8f…, `P60490_REL` `vllm/models/qwen4_exp/nvidia/qsa.py`) ([E25B] `nb1_code_export.py:1382-1385`, applied at 1440-1447); `[E25B] patches/` (`60490_qsa_indexer_workspace_merlin_ad42e37.patch`, `60490_upstream.diff`).
- **Platform:**
  - page-cache drop + Merlin host gate (NB_BUILD_SPEC; NB3 kit);
  - `nb1_gpu_util.jsonl`, `telemetry.jsonl`, `hostmem/*` snapshots (NB1 outputs);
  - load monitor `load_monitor_*.jsonl`.

---

## 4. Risks for NB4's arms

1. **Wrong attribution (highest risk).**
   - The stage-0 NCCL time is spin-wait.
   - The current FULL / PIECEWISE verdicts are wrong for this model.
   - NB1's "critical path" includes the stall and misses the stage-1 drafts.
   - `pp2` busy percentages include spin kernels.
   - Rank-0-only summaries.
   - Fix all of these before any label is derived (§1.10, §3.1).
2. **Micro-batch split confound.** The split (4/4 vs 8/0 vs 5/4) can change between arms or windows and changes both the tick and the stall. Gate every window on the recorded split, and compare only equal splits (§2 item 6).
3. **Arm C:**
   - expected NO_GAIN (stage-0 commits are always eager);
   - a key miss when n is not captured (the n = 5 micro-batch with seqs 9 needs capture 20);
   - extra commit-graph memory (B2: graph memory 0.44 / 0.56 GiB vs 0.33 / 0.43);
   - the boot fails loudly on a capture error (docstring), with a retry path in NB1;
   - the runtime flip is process-wide on all 4 workers (log 518-521), so A/B windows must not overlap other traffic.
4. **Arm G:**
   - extra capture sizes add about 0.1-0.13 GiB per rank of graph memory (B2 vs B1);
   - **with the B3 pin the tightest worker has 191 MiB free (ranks 2-3)** ([E25R] NB1 R2 line 14, decisions line 79), so a larger capture list can OOM at boot;
   - FULL capture failure under host load (#57759) → PIECEWISE fallback;
   - #42271-class deadlock at bonus-token-only shapes (FULL_AND_PIECEWISE + MTP);
   - a FULL backend can be slower (E23 Triton −7 %).
5. **Arm P:**
   - **co-residency:** P5 needs its whole grid resident; the stage-0 broadcast kernel spins for 27.9 ms on the same GPU, and NB3 saw host-arm overlap 0.43-0.97 when both pairs ran (PASS_BOUND_ONLY);
   - one handle per TP group per stream (README Risks item 4): in-graph all-reduces run on 45-49 different graph streams, although they never overlap [OFF];
   - the 10 s default timeout must rise, and an abort poisons the handle (all-NaN) until a collective re-init (README Risks item 3);
   - the result is not bit-equal to NCCL (an oracle NEAR_TIE / drift is expected, so TF rows decide);
   - `/dev/shm` counts in the cgroup (A6) and the binding needs `cudaHostRegister` on a MAP_SHARED file;
   - graph pool reuse: the handle must outlive every captured graph (`destroy` after the graphs are dropped);
   - if it cannot be wired into `CudaCommunicator.all_reduce`, it is `NOT_INTEGRATED` (spec), not faked.
6. **The new `CUDA_DEVICE_MAX_CONNECTIONS` arm (if added):**
   - it must reach every worker and the PLE offload process (spawned workers inherit env; Merlin pops only `MERLIN_*`);
   - it must be read back in the worker env (worker extension) because CUDA reads it at context creation;
   - NCCL behaviour with more connections is UNVERIFIED, so run the oracle + TF gates as for any arm;
   - a value of 1 would serialise all streams (control only, not production).
7. **Crash and deadlock risks:**
   - `--no-async-scheduling` with PP deadlocks (30-min gloo timeout = a wasted GPU session);
   - PP long-prefill illegal memory access (#63505 class seen on stock PP4; never on Merlin PP2 so far: RC2 to 121k);
   - NCCL watchdog hangs if one stage dies;
   - P5 / commit-graph fail-loud boots.
   - NB1's crash pattern: a post-processing exception lost the B1 record (NB1-BUG-1, list-valued `steady` entries, wave-1 lines 41-46). Keep partial records on post-processing errors.
8. **OOM / host:**
   - the B3 pin leaves 191 MiB on the worker of ranks 2-3; the GPU peak is 22,377 of 23,034 MiB;
   - "do not go higher without a 120k first-prefill check" ([E25R] NB1 R2 line 79);
   - the profiler adds host and GPU memory (trace buffers) and slows decode, so profiled windows are never tok/s windows (e23 `profiled_load` docstring);
   - the cgroup sat at its limit through RC2 boots and serving (avail 2.0-5.0k MiB): drop the page cache between boots.
9. **Noise and gates:**
   - A/A spread ±0.6-1 % on 27B (PLAN:525-526), NB1 B1 vs B3 at 8k@8 261.0 vs 254.6 (−2.5 %) with the same step time;
   - the run-to-run acceptance drift (3.18 → 2.69 over 33 min) changes tok/s without any tick change, so compare tick ms and per-position acceptance, not tok/s alone;
   - T0b last (spec) is correct for drift.
10. **Profiler side effects:**
    - the `/start_profile` window uses dev endpoints (security warning) and must be off in T0b and all speed windows;
    - `torch_profiler_with_stack` false and `record_shapes` off keep traces at about 2 MB per rank for 40 iterations;
    - `ignore_frontend: true` means no EngineCore / scheduler events, so the scheduler time per step needs another source: an env-gated timer patch or the iteration histogram. Nothing in our folders records it yet.
