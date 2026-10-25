# C — Engine analysis of the three EVAL120G4B commit runs (2026-10-24)

Scope: read-only. Sources, per run folder under `experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Experiment Results/Kaggle Runs/`:
`bundle/logs/sglang_server_try0.log` (the only server log; there is no `q27x/sglang_server_*` other than the identical copy in the LOW folder), `e14_results.json`, `e14_state.json`, `e14_heartbeat.log`, `bundle/e12_run_report.json` (engine, e11, rev6 metrics, `rev6_scheduler.gauges` = 30-s client samples, per-branch `usage` and `rev5_events.prompt_estimates`), the notebook log `q27-*.log`, and the Duck transcripts from `Q27X_E12_DUCK_BUNDLE.zip` (extracted to `_scratch/eval120_analysis/C_extract/{xe16,xno,low}`; 1,590 / 1,569 / 2,544 responses scanned).
Scripts: scratchpad `parse_runs.py`, `analyze.py`, `adm.py`, `scan_tx.py` (session scratchpad `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\`); full numeric dump `analysis.txt` there.

Run labels: **XE16** = "XHIGH E16 (hosen42)", **XNO** = "XHIGH noE16 (hosen42)", **LOW** = "LOW E16 (foysal)" (Duck `E12_REASONING_EFFORT='low'`; the other two `''`).
All three: identical server flags (`e12_run_report.json` → `engine.sglang.launch_flags`): TP4, FP8 KV, context 155,648, mrr 13, chunked prefill 4096, fp16 SSM, DFLASH draft width 6 (`--speculative-num-draft-tokens 6`), graphs bs 1..13, `extra_buffer_lazy`, mamba cache 56, mem 0.83, `lpm`, decode-log-interval 40. No E14 SGLang patch was applied (the `e14_sglang_patches.py` module is only listed with its hash; `e14_results.json` `phases {}`, `decisions {}`).

Method notes (so every number can be re-derived):
- "Decode interval" = the time between two consecutive `Decode batch` lines (= 40 decode iterations plus any prefill run in between). `gen throughput` is `num_generated_tokens / gap` over exactly that interval (`metrics_reporter.py:898-900`), so Σ(tput × dt) = server-generated tokens. A second estimate, Σ(40 × running × accept len), agrees to 0.02-0.05 % in all three runs.
- "Pure decode" = an interval with no `Prefill batch` line inside and the same `#running-req` at both ends. Step time = running × accept / tput.
- `#full token` / `full token usage` / `mamba num` count only **in-use** (non-evictable) entries (`pool_stats_observer.py:271-276`); the radix cache's evictable KV and mamba states are not visible in these logs.
- Solve window (wall clock) = RUN_ID anchor + `started_sec` + `shift_sec` … + `deadline_in_sec`: XE16 00:11:12 → 06:01:16 (21,004 s), XNO 00:11:28 → 06:01:23 (20,995 s), LOW 00:11:02 → 06:00:34 (20,972 s).

---

## 1. Startup

| | XE16 | XNO | LOW |
|---|---|---|---|
| INT4 staging 19.47 GB: copy / total incl. validate | 25.1 s / 90.9 s | 26.6 s / 91.9 s | 39.6 s / 105.5 s |
| INT4 copy rate (ramp 4/8/16/24 workers, MB/s) | **774 MB/s** (277/807/1006/1048, final 16) | 732 (241/696/922/901) | **492** (227/562/692/696) |
| DFlash2 3.85 GB: copy / total, rate | 6.6 / 18.4 s, 582 MB/s | 7.6 / 19.4 s, 506 MB/s | 9.4 / 21.2 s, 409 MB/s |
| single-stream read probe (nfs) | 57.4 MB/s | (same probe) | (same probe) |
| venv install (`engine.sglang.install.sec`) | 140.4 s | 153.4 s | 178.2 s |
| STAGING_JOIN = LAUNCH (e14_state t) | 180.4 s | 192.6 s | 233.6 s |
| LAUNCH → READY | 527.4 s | 532 s | 537 s |
| READY (e14_state t) | **707.9 s** (11.8 min) | 725.1 s | 771.0 s (12.9 min) |
| solve start (started_sec + shift) | 716 s | 733 s | 780 s |
| Server: load_weight / KV alloc | 13.76 / 4.07 s | 13.93 / 4.13 | 14.31 / 4.10 |
| **target-verify graph capture (bs 1..13)** | **375.38 s** (bs 13 alone 363.6 s) | 387.98 s (bs 13: 376.2 s) | 381.92 s (bs 13: 370.1 s) |
| draft-verify graph capture | 10.00 s | 9.98 s | 10.11 s |
| scheduler_e2e / tokenizer_e2e | 450.15 / 461.34 s | 463.75 / 474.89 | 457.67 / 468.73 |
| serving-time Triton compiles after "fired up" | 24 × `chunk_gated_delta_rule_fwd_kkt_solve_kernel` (1.74-1.94 s each, ~44 s of stall in the first minute) | 24 (1.79 s) | 24 (1.75 s) |

Memory (identical in all three, server log lines ~80-189): per GPU 21.72 GB avail at start; target weights 4.62 GB → 17.10 GB; DFlash2 1.49 GB → 15.58 GB; mamba cache 56 slots = conv 0.04 + ssm 1.00 + intermediate ssm 1.48 + conv window 0.03 GB (≈ 46 MB/slot/GPU; line 111/112); **KV 917,816 tokens** (target K 3.50 + V 3.50 GB, draft K 1.09 + V 1.09 GB; `max_total_num_tokens=917816`, line 180/183); avail after pools 3.74 GB, after target graphs 2.48, after draft graphs 2.09, final `available_gpu_mem=1.78 GB`. At runtime the "device-loaded after serving started" notices report free device memory 0.80-0.83 GiB (XE16, LOW) and 0.97-0.98 GiB (XNO). Client gate `E11 caps` PASS, `effective_running 12`, `slots_per_request 4` (the client's assumption; the server actually uses ~3 mamba slots per running request, see §4).

Warnings (all three; excluding `Error in sitecustomize: No module named 'wrapt'` ×7 and the Triton "avoid CUDA OOM" notices, which number 12 / 30 / 15):
- `DFLASH block size mismatch: using speculative_num_draft_tokens=6 but draft config block_size=8` (expected for width 6).
- `Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0` (target and draft).
- `CustomAllReduceV2 / CustomAllreduce is disabled ... more than two PCIe-only GPUs` → NCCL all-reduce (51 % of the step per Notes/30).
- `Not enough SMs to use max_autotune_gemm mode`; `multimem all-gather disabled (no multicast for world_size=4)`; `tl.make_block_ptr is deprecated`; FastAPI ORJSON deprecation; `Acceleration for non-quantized schemes is not supported by Compressed Tensors. Falling back to UnquantizedLinearMethod` (BF16 linear_attn parts, expected).
- 4-5 × `GET /health 503` during startup (normal).
- Smoke test OK (16 tokens `length`, abort `abort`, 2.2 s); isolation preflight PASS 12/12; parity control 2/2 identical.

Startup cost: ~12-13 min before the first Duck token. **~6.1-6.3 min of it is the first verify-graph capture (bs 13)**, i.e. JIT compile, the same symptom as Notes/30's "JIT cache tar IndexError". Fixing the compile cache would save ~6 min (1.7 % of this 6 h window; 0.8 % of a 12 h hidden run). Staging is no longer a problem (90-105 s, in parallel with the 140-178 s venv install), but LOW's 492 MB/s shows the copy rate varies 1.6× between sessions.

---

## 2. Running-request occupancy and the 11-of-12 loss

### 2.1 Histogram over the solve window (time-weighted over decode intervals)

| running | XE16 | XNO | LOW |
|---|---|---|---|
| 12 | 5,245 s = **25.0 %** (queue>0: 0 s) | 5,345 s = **25.5 %** | 5,567 s = **26.6 %** |
| 11 | 15,363 s = **73.2 %** (queue>0: 15,050 s) | 15,274 s = **72.8 %** (14,977 s) | 14,879 s = **71.0 %** (14,473 s) |
| 10 | 191 s = 0.9 % | 301 s = 1.4 % | 364 s = 1.7 % |
| < 10 | 194 s = 0.9 % (9: 37, 8: 123, 7: 31, 3: 3) | 63 s = 0.3 % | 147 s = 0.7 % (end drain) |
| time-weighted mean running | **11.21** | **11.23** | **11.21** |
| queue>0 time share | **71.7 %** (max queue 1) | **71.4 %** (max 1) | **69.1 %** (max 1) |
| decode lines `running 12` / total | 564 / 4,229 | 610 / 4,195 | 419 / 3,777 |

Client 30-s gauges (`rev6_scheduler.gauges`, 696 / 696 / 695 samples) agree: (server running, queue) = (11,1) 442 / 435 / 359, (10,1) 115 / 109 / 172 (transient: during a chunked prefill), (12,0) 83 / 79 / 71, (11,0) 25 / 35 / 31, (10,0) 24 / 35 / 52; `workers_running = 12` in 688 / 689 / 687 samples. **12 running with a non-empty queue never occurs, and 13 never occurs.** The client always had 12 requests in flight; the server ran 11 and parked the 12th.

### 2.2 Cause (source, not a guess)

`managers/min_free_slots_delayer.py:20-21`: for DFLASH with mrr ≥ 8 and no explicit `--min-free-slots-delay`, `min_free_slots = min(4, max(2, (mrr + 5) // 6))` = **3 for mrr 13**. `should_delay()` (line 38) holds all fresh prefill while `num_allocatable_reqs < 3`, called at `scheduler.py:3856`. With 13 slots, 11 running leaves 2 free → the 12th request waits until a running request finishes (running → 10, 3 free), then the waiting request (and any second one that has arrived) is admitted together → 11, sometimes 12. This exactly reproduces the pattern: every admission coincides with a completion (1,248 / 2,572 prefill lines in XE16 are ≤ 2 s after a `POST ... 200`; the rest are continuation chunks), and the log shows `Prefill batch ... #running-req: 10, #queue-req: 1, #pending-token: 0` after each completion (e.g. XE16 lines around 01:49:31-01:52:49), i.e. the queue empties and the finished worker's next request immediately re-fills it. This matches Notes/30 ("delayer confirmed on the GPU; #44401 does not fix it").

### 2.3 Cost in tok/s and tokens

Pure-decode throughput at 12 vs 11 (§3): XE16 434.5 vs 423.0 (×1.027), XNO 434.9 vs 422.9 (×1.028), LOW 445.2 vs 434.8 (×1.024). The step grows 100.0 → 105.8 ms (XE16), so the 12th stream adds only ~2.5-2.8 % decode throughput (per-stream 38.5 → 36.2 tok/s).

| | XE16 | XNO | LOW |
|---|---|---|---|
| tokens lost while a request was parked (Σ over queued intervals of (tput₁₂ − tput_own level) × dt) | **173,075 = 2.4 %** | **179,350 = 2.5 %** | **151,966 = 2.3 %** |
| end-to-end model (decode + fixed prefill cost, §3.4) 11 → 12 | 341.9 → 349.4 tok/s (+2.2 %) | +2.2 % | 315.9 → 322.2 (+2.0 %) |
| mean queue wait per request (Little: 0.717 waiting × window / requests) | **9.5 s** (1,590 req) | 9.5 s (1,569) | 5.7 s (2,544) |
| continuous "1 waiting" stretches: median / p90 / max | 57 / 197 / 933 s | 62 / 202 / 611 s | 47 / 152 / 582 s |

So the 11-of-12 loss is **~2.2-2.5 % of generated tokens** (≈150-180k tokens per 6 h run, ≈1.3k tokens per task), not 9 %: one more stream at this batch size buys only +2.7 % decode, because the step time rises ~5.2 ms per added request. (The naive "+26-40 % if 12 all the time at the pure-decode rate" figure in `analysis.txt` is not a valid estimate: it ignores the 19-27 % of wall time that prefill takes in every case.)

---

## 3. Throughput

### 3.1 Pure-decode tok/s, accept length and step time by running level

| run | XE16 tok/s p10/p50/p90 (n) | accept p50 | step ms p50 | XNO tok/s p50 (n), acc, step | LOW tok/s p50 (n), acc, step |
|---|---|---|---|---|---|
| 12 | 401 / **434.5** / 469 (413) | 3.8 | **105.8** | **434.9** (468), 3.8, 105.6 | **445.2** (219), 4.0, 107.0 |
| 11 | 390 / **423.0** / 460 (2,559) | 3.9 | **100.0** | **422.9** (2,509), 3.8, 100.0 | **434.8** (1,764), 4.0, 100.9 |
| 10 | — | — | — | 357.3 (17), 3.4, 95.9 | 424.4 (5), 4.1, 96.1 |
| 9 | 416.4 (4) | 4.0 | 85.6 | 360.6 (4), 3.6, 90.6 | — |
| 8 | 377.5 (32) | 3.8 | 80.0 | 377.4 (2), 3.8, 81.3 | — |
| 7 | 361.7 (6) | 3.9 | 75.3 | 340.0 (6), 3.7, 75.5 | 391.9 (11), 4.3, 76.3 |
| 6 / 5 / 4 | — | | | — | 322.7 / 277.1 / 239.7; step 72.1 / 66.0 / 61.0 |

Step time vs batch is close to linear: ~61 ms at 4, 75 at 7, 80 at 8, 86 at 9, 96 at 10, 100 at 11, 106 at 12 → **≈5.2 ms per added request** at 25-30k context per request.

All decode intervals including prefill interference (time-weighted mean tok/s): XE16 run 12: 195.6, run 11: 394.6; XNO 207.3 / 388.6; LOW 139.3 / 384.1. The 12-running intervals look slow only because 12 is reached right after a double admission, i.e. during a large prefill.

### 3.2 Totals

| | XE16 | XNO | LOW |
|---|---|---|---|
| server-generated tokens in window Σ(tput·dt) / Σ(40·run·acc) | **7,220,819** / 7,219,404 | **7,163,446** / 7,162,988 (whole log 7,163,773) | **6,660,788** / 6,661,625 |
| mean tok/s over the window | **344.0** | **341.4** | **317.8** |
| report `usage_all.completion_tokens` (= Σ branch generated_tokens) | 7,144,122 | 7,091,227 | 6,606,525 |
| server − report (deadline-cut in-flight + preflight/smoke/diag) | **76,697 (1.06 %)** | **72,546 (1.01 %)** | **54,263 (0.82 %)** |
| requests / prompt / cached (cached %) | 1,590 / 29.12M / 22.78M (78.2 %) | 1,569 / 28.04M / 21.53M (76.8 %) | 2,544 / 51.00M / 42.00M (82.4 %) |
| generated tokens per task (120 tasks) | **59.5k** | 59.1k | 55.1k |
| completion per request p50 / p90 / max | 1,305 / 15,229 / 24,576 | 1,214 / 15,686 / 24,576 | 1,053 / 6,988 / 24,576 |
| request latency p50 / p90 / max | 54.4 / 484 / 1,017 s | 54.1 / 515 / 1,096 s | 45.4 / 253 / 1,056 s |

### 3.3 Accept length

Overall (all decode lines): XE16 p10 3.6 / p50 3.9 / p90 4.1 / mean 3.9, min 3.13; XNO 3.6 / 3.8 / 4.1 / 3.8, min 3.05; LOW 3.7 / 4.0 / 4.3 / 4.0, min 3.14. Histograms (bins of 0.5): XE16 3.0-3.5: 190, 3.5-4.0: 2,969, 4.0-4.5: 1,056, 4.5-5.0: 13; XNO 243 / 2,952 / 993 / 5; LOW 74 / 1,793 / 1,796 / 111 / 5.0-5.5: 2. Accept rate p50 0.54-0.60 (XHIGH), 0.58-0.69 (LOW). For reference, Notes/30 measured width 6 at 4.24 on the short (~6k) R2 band; production accept is ~0.35-0.45 lower at 15-40k.

### 3.4 Prefill share of tokens and time

| | XE16 | XNO | LOW |
|---|---|---|---|
| prefill lines in window | 2,527 | 2,582 | 3,908 |
| computed (new) prefill tokens / cached tokens | **6,409,466** / 22,908,009 | **6,583,886** / 21,645,761 | **9,036,428** / 42,088,072 |
| hit rate | 78.1 % | 76.7 % | 82.3 % |
| new prefill tokens per generated token (r) | **0.888** | 0.919 | **1.357** |
| full 4,096-token chunks | 1,126 | 1,201 | 1,642 |
| prefill lines with `#cached-token: 0` | 1,045 | 1,127 | 1,578 |
| regression dt = a·(40·step) + c·newtok → c | 0.635 ms/token (**1,575 tok/s**), a = 0.996 | 0.635 ms (1,574 tok/s) | 0.638 ms (1,568 tok/s) |
| prefill time ≈ c × new tokens | **4,069 s = 19.4 % of the window** | 4,183 s = 19.9 % | **5,764 s = 27.5 %** |
| excess time of intervals that contain prefill | 3,954 s (18.8 %) | 4,113 s (19.6 %) | 5,701 s (27.2 %) |
| decode intervals with tput < 100 at running ≥ 8 (all contain prefill) | 106 intervals, 3,230 s | 110, 3,450 s | 149, 4,707 s |

Decode stops while a 4,096-token chunk runs (mixed chunk is off), so prefill is pure lost decode time. A two-term model reproduces the observed mean throughput: per generated token = 1/tput_decode + r·c → XE16 1/423 + 0.888 × 0.635 ms = 2.928 ms → **341.6 tok/s (observed 344.0)**; LOW 1/434.8 + 1.357 × 0.638 = 3.166 ms → 315.9 (observed 317.8).

**Where the prefill tokens come from** (per-request `prompt_estimates` × (1 − cached %)):

| | XE16 | XNO | LOW |
|---|---|---|---|
| total uncached | 6,333,849 | 6,517,848 | 8,992,696 |
| first request of a branch | 199 req → 713,980 (11 %) | 196 → 694,546 | 203 → 831,595 |
| later requests, cached ≥ 50 % (normal increments) | 1,224 → 1,414,705 (22 %) | 1,191 → 1,291,859 | 2,136 → 1,940,259 |
| **later requests, cached < 50 % (misses)** | **167 → 4,205,164 (66 %)** | **182 → 4,531,444 (70 %)** | **205 → 6,220,842 (69 %)** |
| of which: first request of a stage-2 run (resume of a paused branch), `runs[].first_uncached` | 166 runs, **3,656,622** (100 runs > 10k: 3,242,913) | 162, **3,663,635** | 174, **4,315,749** |

So **~58-66 % of all prefill is re-prefilling branches resumed in stage 2** after their KV/mamba state was evicted (they were paused at 28k exposure hours earlier, behind ~200 other branches; the full histories, ~6-10M tokens, cannot fit a 917,816-token pool). Per-request cached % p10 is 6.1 / 5.5 / 6.9 %; 267 / 282 / 326 requests are < 50 % cached. The cache diagnostic itself (`cache_diag`) is healthy: exact replay 9,024 / 9,032 cached, divergent 8,192-8,960 / 9,038, including after churn.

### 3.5 Throughput as the run progresses (hourly, from the solve start)

XE16:

| hour | tok/s (all) | accept | mean running | KV used p50 / max | `#full token` p50 | mamba p50 | prefill hit % (new / cached) | queue>0 |
|---|---|---|---|---|---|---|---|---|
| 0 | 373.5 | 3.81 | 11.06 | 0.26 / 0.38 | 236,837 | 0.59 | 86.0 % (0.75M / 4.58M) | 89 % |
| 1 | 387.7 | 3.87 | 11.02 | 0.28 / 0.39 | 253,605 | 0.59 | 89.0 % (0.58M / 4.72M) | 96 % |
| 2 | 351.7 | 3.91 | 11.15 | 0.27 / 0.47 | 250,993 | 0.61 | 83.2 % (1.11M / 5.50M) | 80 % |
| 3 | 314.7 | 3.88 | 11.39 | 0.35 / 0.46 | 323,246 | 0.61 | **68.8 %** (1.45M / 3.21M) | 58 % |
| 4 | 323.8 | 3.79 | 11.47 | 0.36 / 0.47 | 327,903 | 0.61 | **61.8 %** (1.27M / 2.06M) | 51 % |
| 5 (50 min) | 306.3 | 3.84 | 11.18 | 0.37 / 0.47 | 339,735 | 0.59 | 69.5 % (1.25M / 2.84M) | 53 % |

XNO: 386.1 / 389.6 / 346.4 / 304.3 / 307.6 / 309.0 tok/s; hit 87.6 / 88.7 / 84.3 / 63.7 / 56.1 / 60.9 %; mean running 11.07 / 11.01 / 11.11 / 11.46 / 11.48 / 11.26.
LOW: 390.3 / 378.8 / 325.2 / 272.5 / 261.6 / 271.1 tok/s; hit 91.5 / 91.2 / 83.4 / 75.2 / 70.3 / 76.2 %; prefill new tokens per hour 0.86M → 2.15M.

Throughput falls **~18-30 % from hours 0-1 to hours 3-5**. Accept is flat (3.8-3.9; LOW 3.9-4.1), and context growth explains only ~3 % (step +3 ms). The fall is the stage-2 resume prefill: computed prefill tokens per hour double (0.6-0.9M → 1.3-2.1M) while the hit rate falls from 86-92 % to 56-75 %. The occupancy actually improves (mean running 11.0 → 11.5) because completions are more frequent, so the delayer releases more often.

### 3.6 Throughput / step / accept vs context

Pure decode, bucketed by `#full token / running` (pool average per request; the shared prefix is counted once, so this underestimates the per-request length):

| ctx/req | XE16 run 11: tok/s, accept, step | XE16 run 12 | LOW run 11 | LOW run 12 |
|---|---|---|---|---|
| < 10k | 479.7, 4.02, 91.9 ms (25) | — | 527.8, 4.38, 92.0 (11) | — |
| 10-20k | 440.9, 3.87, 96.7 (355) | 451.5, 3.79, 100.8 (20) | 472.1, 4.17, 96.9 (164) | 492.9, 4.14, 101.0 (12) |
| 20-30k | 424.1, 3.84, 99.6 (1,584) | 439.8, 3.85, 105.2 (246) | 439.9, 4.00, 100.1 (928) | 455.2, 4.01, 105.8 (99) |
| 30-40k | 409.7, 3.82, 102.6 (595) | 423.6, 3.79, 107.2 (147) | 419.8, 3.92, 102.9 (653) | 430.7, 3.89, 107.8 (108) |
| 40-60k | — | — | 409.9, 3.94, 105.6 (8) | — |

By total `#full token` (run 10-12): XE16 0-100k 478.8 tok/s / 91.9 ms → 100-200k 447.2 / 96.2 → 200-300k 428.9 / 99.0 → 300-400k 417.8 / 102.3 → 400-500k 400.8 / 104.9 ms; XNO and LOW the same within ±2 %. **≈+3 ms of step per +100k tokens of live context** (−3 to −4 % tok/s per +100k).

---

## 4. KV, mamba and failure signs

| | XE16 | XNO | LOW |
|---|---|---|---|
| KV in use (`full token usage`) p50 / p90 / p99 / max | 0.31 / **0.40** / 0.45 / **0.47** | 0.32 / 0.40 / 0.45 / 0.49 | 0.33 / 0.41 / 0.48 / **0.52** |
| peak in-use tokens (of 917,816) | 431,237 | 450,507 | 475,268 |
| client gauge kv p50 / p90 / max | 0.316 / 0.401 / 0.461 | 0.325 / 0.404 / 0.491 | 0.343 / 0.412 / 0.502 |
| mamba in use p50 / p90 / max (`mamba num` max of 56) | 0.59 / 0.64 / 0.70 (39) | 0.61 / 0.64 / 0.70 (39) | 0.59 / 0.64 / 0.70 (39) |
| per-request prompt length p50 / p90 / max | 13,559 / 40,798 / **51,637** | 13,132 / 38,851 / 51,454 | 17,168 / 40,495 / 52,453 |

Mamba: 33 slots at 11 running and 36 at 12 → **~3 in-use slots per running request** (state + 2 ping-pong; the client's `slots_per_request 4` is conservative). The in-use KV never exceeded 52 % at 12 streams; the rest of the pool is evictable radix cache (not visible in the logs).

Failure signs (server logs fully grepped):
- Traceback 0, `out of memory` 0, retract / preempt 0 (the only matches are inside `server_args`), watchdog fired 0, `KV cache pool is full` 0, NaN 0, stall messages 0 except the 24 start-up compile notices.
- HTTP: 5,882 / 5,859 / 6,828 × 200; 503 only during startup (5 / 4 / 4). One non-streaming `create_error_response` (`serving_chat.py:2250`, a ValueError path) per run in the last 2 minutes (06:01:13 XE16, 06:01:21 XNO, 06:00:32 LOW), i.e. the deadline aborts.
- Longest gaps between decode lines: 41-57 s (XE16), 43-55 s (XNO), 47-59 s (LOW), all inside big chunked prefills (13-14 × 4,096 chunks at ~2.6 s each = 34-37 s for a ~55k-token resume). No engine stall.
- Client: 8 / 7 / 4 `analyzer request timed out` at the deadline (21,726.8 s, 21,733.4 s, 21,764.1 s); `timeouts_clamped` 66 / 54 / 158 (request timeouts clamped to the deadline); `infra_errors` 2 / 4 / 3; `verify_skipped_deadline` 0; kv holds 0, stragglers 0.
- Malformed tool calls (server log): `Tool 'submit' is not defined in the tools list` **22 / 28 / 23** (the model calls the python helper `submit` as a tool); `Tool 'code'` 1 / 0 / 6; `'function_calls'` 1 / 0 / 0; `'function'` 0 / 2 / 1; `Parsed parameter 'python' is not defined` 0 / 1 / 0; **`Tool call marker present but no complete call parsed ... dropping the incomplete call` 1 / 0 / 1** (XE16 line 1780 at 00:49:46; LOW line 9428 at 03:40:28). Client side, `tool_call_argument_issues: invalid JSON arguments (Unterminated string ...)` 0 / 2 / 2.

---

## 5. Width 6 at long context

Accept len vs pool-average context per request (all decode lines with running ≥ 10):

| ctx/req | XE16 p10 / p50 / p90 (n) | XNO p50 (n) | LOW p10 / p50 / p90 (n) |
|---|---|---|---|
| 0-5k | 3.84 / 4.02 / 4.24 (11) | 3.91 (15) | 4.33 / 4.46 / 4.57 (11) |
| 5-10k | 3.74 / 4.02 / 4.30 (34) | 3.95 (30) | 3.98 / 4.39 / 4.65 (25) |
| 10-15k | 3.59 / 3.79 / 4.09 (50) | 3.92 (98) | 4.03 / 4.27 / 4.50 (40) |
| 15-20k | 3.62 / 3.87 / 4.13 (584) | 3.87 (505) | 3.81 / 4.14 / 4.46 (344) |
| 20-25k | 3.59 / 3.86 / 4.16 (1,210) | 3.87 (963) | 3.76 / 4.06 / 4.37 (780) |
| 25-30k | 3.58 / 3.85 / 4.12 (1,347) | 3.84 (1,629) | 3.68 / 3.99 / 4.30 (1,262) |
| 30-35k | 3.56 / 3.82 / 4.12 (803) | 3.80 (797) | 3.61 / 3.92 / 4.23 (1,025) |
| 35-40k | 3.55 / 3.72 / 4.02 (130) | 3.73 (137) | 3.63 / 3.97 / 4.35 (226) |
| 40-50k | 3.83 (1) | — | 3.95 (12), 4.40 (1) |

**No collapse.** Accept falls slowly: XE16 4.02 → 3.72 (−7.5 %) from < 10k to 35-40k; LOW 4.46 → 3.92-3.97 (−11 %). Linear extrapolation gives ~3.4-3.6 at 80-100k, still far from 1. **Caveat: the 40-100k band was not reached.** The largest single prompt was 51.6k / 51.5k / 52.5k tokens (rev6 pauses at 28k exposure, window 76,484), and the pool average never exceeded ~45k per request. Width 6 at 60-100k per request is therefore still unmeasured.

---

## 6. Numerical health

- Accept len never below 3.05 on any 40-step line (no line < 2.5 in any run). There is no drift toward 1.
- Finish reasons (transcript META = request records): XE16 tool_calls 1,491, length 98, stop 1; XNO 1,485 / 84 / 0; LOW 2,524 / 12 / 8. No `abort` or other finish reasons.
- `length` = the reasoning hit the 24,576 cap with no tool call (98 / 84 / 12 = rev6 `length_cut_no_call` 98 / 82 / 10), reasoning p50 55-58k chars. These are budget cuts, not EOS problems.
- EOS mid-thinking / empty content after long reasoning: `finish stop + content 0 + no tool call` occurs once in XE16 (9bbf930d_b0, reasoning 76 chars, request sent 00:49:29) and once in LOW (9bbf930d_b0, reasoning 7,898 chars, sent 03:37:51). Both coincide with the server's `dropping the incomplete call` line (00:49:46 and 03:40:28), so they are **dropped malformed tool calls, not EOS inside thinking**. LOW's other 7 `stop`s are one branch (58490d8a_b0) ending normally after a verified 3/3 solution, with content 135-1,333 chars. XNO has none.
- Repetition scan of all thinking blocks: 110 / 100 / 96 blocks contain a token repeated 16+ times in a row, and in every example checked the token is a grid digit (`0 0 0 …`, hand-read grid rows; rev6 `hand_reading_grid_row_share` 0.0064 / 0.0061 / 0.0086, `reasoning_repeat_share` 0.0101 / 0.0051 / 0.0047). 7 / 5 / 7 blocks repeat the same line 6-12 times (grid art such as `.....#` ×12). No single-character runs > 40, and no `!!!!` garbage: the 3 / 2 / 0 hits are emphatic prose ("✓✓✓!!!!", "Observed: right ✗!!!!").
- Raw-reasoning switch (E14-P): sampled branch 0934a4d8_b0 shows `kept 21, same 0, mismatch 0, markup 0`.
- Conclusion: FP8 KV with unit scales, fp16 SSM and width 6 show **no sign of numerical trouble up to ~52k tokens per request**.

---

## 7. End of run

| | XE16 | XNO | LOW |
|---|---|---|---|
| "tail fade" message | 280.6 min (69.5 min left) | 282.3 min (67.6 left) | 282.3 min (67.2 left) |
| running in the last 15 min | 11-12 until 05:58:10, then 10 → 9 → 8 → 7 at 06:01 | 11-12 until 05:58:57, then 10 → 7 | 11 until 05:57:15, then 10 → 7 → 4 by 05:59:57 |
| deadline / last decode line | 06:01:16 / 06:01:16 | 06:01:23 / 06:01:24 | 06:00:34 / 06:00:34 |
| client timeouts at the deadline | 8 requests | 7 | 4 |
| post-deadline server work | `cache_diag` (12-request churn, prompt 108,398, ~66 s) = 35 prefill lines, 118,448 new + 44,160 cached; last prefill 06:02:40 | same 35 lines; last 06:02:46 | same; last 06:01:58 |
| SIGTERM / report REPORT phase (e14 t) | 06:02:42 / 21,805.7 s | 06:02:48 / 21,813.8 s | 06:02:00 / 21,837.9 s |
| vs TOTAL_BUDGET_SEC 23,400 | 1,594 s (26.6 min) unused | 1,586 s | 1,562 s |
| server − counted tokens (§3.2) | 76,697 (1.06 %) | 72,546 (1.01 %) | 54,263 (0.82 %) |

Drain losses are small: about 3 minutes at 7-10 streams (~0.1-0.2 % of tokens) plus ~1 % of tokens generated by requests that were cut at the deadline and never counted. Branch status at the end: paused 165 / 156 / 170, done 37 / 43 / 33; never checked 146 / 144 / 134 of 202 / 199 / 203 branches. Scores: eval 46 / 172 (task-norm 30.0 %), 41 / 172 (25.83 %), 31 / 172 (18.89 %).

---

## 8. Conclusions

### 8.1 What limits throughput and coverage now (in order of size)

1. **Decode step cost**: 100 ms at 11 and 106 ms at 12 streams (≈ +5.2 ms per stream, +3 ms per +100k live tokens). Decode-only rate is 423-445 tok/s; per Notes/30 about half of the step is NCCL all-reduce (PCIe, custom all-reduce disabled).
2. **Prefill takes 19-20 % (XHIGH) and 27.5 % (LOW) of the solve window**, at ~1,575 tok/s, with decode fully stopped (no mixed chunk). 58-66 % of the computed prefill tokens (3.66M / 3.66M / 4.32M) are **stage-2 resumes of paused branches whose cache was evicted**. This is why throughput falls 18-30 % after hour 2. If those resumes hit cache, the model in §3.4 gives XE16 r 0.888 → 0.381 and 341.6 → **~384 tok/s (+12 %)**, LOW 1.357 → 0.709 and 316 → **~363 tok/s (+15 %)**.
3. **Occupancy 11.21 of 12** (delayer): 2.2-2.5 % of tokens.
4. Startup 12-13 min (≈ 6 min of it the first verify-graph compile); deadline / drain ~1 %.
- Coverage result: ~59.5k generated tokens per task (XHIGH), i.e. at the 60k floor on average, while 146 of 202 branches never reached a `check`. Every extra % of tokens goes straight into coverage.

### 8.2 Expected gains

**(a) Real 12/12** (`--min-free-slots-delay 1` with graphs 1..13: the Notes/30 crash came from buffers sized for 12, and graphs are now 1..13 while the client offers only 12, but this combination is untested; or the E15 `w6_mrr14` arm, where mrr 14 → threshold 3 → admission at ≤ 11 running): decode +2.7 % (423 → 434.5 tok/s), **end-to-end ≈ +2.2 %**, about +155-180k tokens per 6 h window, plus ~9.5 s less queue wait per request. Real but small.

**(b) 14-20 streams.** Model: step(n) ≈ 100 + 5.2·(n − 11) ms at 25-30k context, accept 3.85. The verify batch stays ≤ 126 tokens up to 21 streams, so the INT4 GEMMs plausibly stay in the same M-tile class (an assumption, not measured). Prefill cost per generated token is held fixed (XE16).

| streams | decode tok/s | Δ decode | end-to-end tok/s | Δ e2e | in-use KV p90 / max (scaled from 0.40 / 0.47) | mamba in use (3/stream) |
|---|---|---|---|---|---|---|
| 11 (now) | 423 | — | 342 | — | 0.40 / 0.47 | 33 |
| 12 | 434.5 (measured) | +2.7 % | 349 | +2.2 % | 0.40 / 0.47 | 36 |
| 14 | ~466 | +10 % | ~369 | +8 % | 0.47 / 0.55 | 42 |
| 16 | ~489 | +15 % | ~383 | +12 % | 0.53 / 0.63 | 48 |
| 20 | ~525 | +24 % | ~405 | +18 % | 0.67 / 0.78 | 60 (> 56) |

Caveats: (i) these are linear extrapolations and must be measured (E15 arms, real-concurrency asserts); (ii) more in-use KV leaves less evictable cache, so the resume-miss prefill (item 2) grows; at 20 streams only ~20-30 % of the pool would remain for cache; (iii) mamba cache must be ≥ ~4 × mrr above ~16 streams (46 MB/slot/GPU, i.e. 80 slots cost ~1.1 GB/GPU ≈ 105k KV tokens); (iv) the delayer threshold scales (mrr 17 → 3, mrr 20-24 → 4), so mrr must be ≥ streams + threshold − 1 (or disable the delayer); (v) LOW gains are smaller (16 streams ≈ +9 % e2e) because of its heavier prefill. KV itself is not the binding constraint at 14-16 streams.

**(c) MTP (Notes/35)**: predicted KV ~1,198k tokens at 12 streams (+31 %; 100k per stream) and ~1.2 GB extra headroom; the only measured data are E11-03 rev3b (fp32 SSM, capped at 10): accept 3.05 / 4, 374 tok/s at 10 running versus DFlash2 W8 477-489 at 12-13 in the same run. Here DFlash W6 delivers 3.8-4.0 accepted tokens per step at 100-106 ms. MTP would need a ≤ ~85 ms step at 12 streams just to tie (12 × 3.05 / 0.085 ≈ 430 tok/s), and it runs 3 sequential draft passes with full-vocab heads. **Expected decode change vs DFlash W6 at 12 streams: about −10 % to +5 %.** Its +31 % KV does not help at 12-16 streams, because in-use KV is ≤ 0.52 now. It helps only (i) at ≥ 20 streams, or (ii) as extra radix cache for paused branches. ~280k extra tokens hold only ~7 more 40k-token histories out of ~100 resumed, so it would remove maybe ~5-10 % of the resume misses. MTP is not the next lever on throughput; measure it in E15 as planned.

### 8.3 Ranked recommendations from this data

1. **Attack the stage-2 resume re-prefill** (+12-15 % e2e if fully avoided; it is also the cause of the late-run slowdown). Options:
   - Schedule each branch's stage-2 continuation while its cache is still warm, or avoid deep pauses.
   - Host-memory KV + mamba cache for paused branches. About 43 KB/token across 4 GPUs means 100 GB of host RAM holds ~2.3M tokens; SGLang 0.5.21 hicache with hybrid mamba + DFLASH is unverified (`hicache_attached=False`).
   - `--enable-mixed-chunk`, so decode continues during prefill (DFLASH compatibility unknown, needs a smoke test).
2. **14-16 real streams with the delayer fixed** (+8-12 % e2e predicted; KV and mamba fit at 16 with mamba ≥ 64-68).
3. 12/12 fix alone: +2.2 %. It comes for free with item 2.
4. JIT/compile cache for the verify graphs: −6 min startup.
5. Width 6 is safe up to ~52k per request (accept −7 to −11 % from short to 40k, no numerical issues). Keep it; measure 60-100k separately if the harness will ever go there.
