# A2: past work mining, SGLang and speculative decoding (for Experiment 25, Flash-Next 1500)

**Author:** agent A2, 2026-10-31. Read-only mining; this file is the only output.
**Scope:** Experiment 11 (SGLang 0.5.21 + RedHat INT4 + DFlash2), Experiment 14 SGLang parts (E14A, E14S, EMS engine matrix), Experiment 15 (Duck config B and E15P on SGLang), Experiment 17 (SGLang silent accuracy), Experiment 24 (SGLang speed track), and the related `_scratch` folders. Cross-references to E19/E23/VMS (vLLM) appear only where they bear directly on speculative decoding.
**Model in all measured rows:** Qwen3.8-27B (dense GDN hybrid: 48 GDN + 16 full-attention layers), not Flash-Next, unless a row says FN.
**Hardware in all measured rows:** Kaggle 4×L4 (sm_89), TP4, PCIe, unless a row says otherwise.

Abbreviations:
- **E11R** = `experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/`
- **E14** = `experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/`
- **E15N** = `experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Notes/`
- **E24** = `experiments/Experiment 24 - SGLang speed track (planned)/`
- **C_engine** = `E14/Notes/EVAL120 analysis 2026-10-24/C_engine.md`
- **EMS** = `E24/Base Engine/results/ems_report.md` (same data as `E14/Experiment Results/EMS_SAFE_RISKY_VMS_rev2g_GPU_results_2026-10-26.md`)

Labels: **[M]** measured on our Kaggle GPU; **[S]** read in engine source by us; **[C]** third-party or vendor claim; **[I]** my inference, unmeasured.

---

## 1. Measured numbers

### 1.1 Engine sweeps (synthetic prompts, steady windows)

tok/s is the aggregate generated (accepted) tokens per second over all running streams. "Streams" is the real running count read from the server log, unless marked "requested".

| Experiment | Config | Context @ streams | tok/s | Accept / width | KV | Failures / notes | Evidence |
|---|---|---|---|---|---|---|---|
| E11-03 A | INT4, no spec, fp32 SSM, extra_buffer, auto mem 0.62 | ~4.3k @1/4/8/12/16 | 45.6 / 145 / 229 / 289 / 347 | — | 604,274 tok fp8 | 24/32 requested: 293/349 (only 22 can run: mamba cap 113) | `E11R/Kaggle Runs/E11-03 rev3b run1 (foysal, 2026-10-17)/e11_03/report.md` (A_int4 table); `E11R/E11-03 run1 analysis (2026-10-17).md` §1-2 |
| E11-03 A long | same | 12 × 24,576 out | 282.4 | — | usage max 0.53 | A had 56.7k of 57.1k prompt tokens cached, B/C ~1k (unfair compare) | same report; analysis §6 |
| E11-03 B | INT4 + NEXTN MTP 3/1/4, fp32 SSM, auto | ~4.3k @1/4/8, then capped at **10** | 55.1 / 205 / 297; "12"=263.6, "16"=314.5 (really 10) | **3.17 / 3.10 / 3.09 / 3.11 / 3.17** of 4 | 474,203 | `max_running_requests is capped to 10 by the mamba state cache (max_mamba_cache_size=50, 5 state slots per request)` | report B table; `E14/Notes/35 Claude agent - E15 MTP preflight (2026-10-24).md` §0 |
| E11-03 B long | same | 12 (10 real) × 24,576 | 262.3 | 3.02 | usage 0.80 | — | report |
| E11-03 C | INT4 + DFlash2 W8, fp32 SSM, auto | ~4.3k @1/4, then capped at **6** | 93.7 / 250; "8-32" ≈ 268-307 (really 6) | 3.89 / 4.37 / 4.74 / 4.70 / 4.95 | 394,559 | capped to 6 (mamba 33) | report C table |
| E11-03 C longctx | same | 4 × ~85.5k ctx | server decode median ~238 tok/s at 4 running | 4.58 | usage 0.87 | only ~4 × 85k fit in 395k | report extra tests; analysis §3 |
| E11-03 C temp | same, 12 × 4096 | capped 6 | T1.0 285.8 / T0.6 241.7 | T1.0 4.34 / T0.6 4.60 | — | inconclusive (both capped) | report extra tests |
| E11-03 D | DFlash2 W8 + **fp16 SSM** | ~4.3k @1/4/8/**12**/16(13) | 107.5 / 278 / 356 / **404** / 381 | 4.31 / 4.51 / 4.85 / 4.62 / 4.78 | 386,563 | cap 13 (mamba 68); D at 12 = 1.39× A | report D table; analysis §2 |
| E11-04 W1 | DFlash2 W8, fp16 SSM, **extra_buffer_lazy, mamba 56, no prefill graph, mem 0.83** | short 4k @1/4/8/12(11) | 110.6 / 257.5 / 363.8 / 325.9 | 4.50 / 4.49 / 4.67 / 4.51 | **881,336** | sweep at 12 reached only 11 (staggered prefill) | `E11R/Kaggle Runs/E11-04 rev2b run1 (esfersami50, 2026-10-18)/e11_04/report.md` |
| E11-04 W1 PROD | same, closed loop ARC prompts, T 1.0 | 12 real, ctx ≤ 30k | **416.5** (log by ctx: <10k 458, 10-20k 413, 20-40k 388) | ~4.5 | 881,336 | ≥ 40k not reached (PROD cut to 10 min) | same; `E11R/E11-04 run1 analysis (2026-10-18).md` |
| E11-04 W2 | same with bf16 SSM | 12 real | 412.1 (466 / 402 / 375 by ctx) | — | 881,336 | sanity 7/8 vs W1 8/8 | same |
| E11-04 W3 | same with fp32 SSM | 1/4/8 | 100 / 273.5 / 337.9 | 4.15-4.79 | **598,712** | fp32 state costs 32 % of the KV pool vs fp16 | same |
| E14A SLIM R2 W1 | DFlash2 W8, 12 streams | ~5.9k | **505.0**, step 114.0 ms | 4.86 | — | long band (~40.9k median prompt per Notes/37) 449.8, 123.1 ms, 4.62 | `E14/Notes/30 Claude - E14A SLIM R2 run results (2026-10-24).md`; `E14/README_FINDINGS.md` §1.1 |
| E14A R2 width 6 | DFlash2 W6 | ~5.9k @12 | **531.4 (+6.3 %)**, 96.4 ms | 4.24 | — | adopted | same |
| E14A R2 width 4 | DFlash2 W4 | ~5.9k @12 | 526.8 (+5.3 %), 74.5 ms | 3.27 | — | — | same |
| E14A R2 target-only | no spec | ~5.9k @12 | 339.6, **35.3 ms** | 1 | — | DFlash2-8 = 1.49× target-only | same |
| E14A R2 #44401 | backport | ~5.9k @12 | 508.1 | 4.84 | — | no occupancy gain | same |
| E14A R2 nodelay | `--min-free-slots-delay 1`, graph max 12 | — | 490.3 | 4.63 | — | **crash at 13 running**: `tensor a (12) must match b (13)` in `dflash_worker_v2.py:273` | same |
| E14A R2 graphs13 | graphs 1..13 | — | 500.1 | 4.79 | — | neutral | same |
| E14A R2 int8 mamba | int8 Mamba checkpoint | — | — | — | — | **OOM** (`allocate 200 MiB, free 89 MiB`) | same |
| EMS SAFE ref | DFlash2 W6, 16 streams, lazy, mamba 68, mem 0.83, lpm, page 1 | 8k / 40k / XL 90k | **583 / 521 / 370 (8 streams)** | 4.21 / 4.11 / 4.01; step 115.7 / 127.3 / 86.8 ms | 852,152 | gate PASS (BOUNDARY), XL parity PASS, canary 0 | EMS table row `ref_dflash_w6_s16` |
| EMS RISKY ref | same, second measurement | 8k / 40k / XL | 572 / 544 / 403 (8) | 4.14 / 4.33 | — | run-to-run spread ≈ ±2-4 % | `E14/Experiment Results/EMS_SAFE_RISKY_VMS_rev2g_GPU_results_2026-10-26.md` |
| EMS **best** | DFlash2 W6, **20 streams**, mrr 21, mamba 84 | 8k / 40k / XL | **597 / 536 / 364 (7)** | 4.17 / 4.21; step 140.3 / 157.1 ms | 779,192 | +2-3 % vs ref, inside the spread | EMS `dflash_w6_s20`; `E24/README.md` |
| EMS W4 | DFlash2 W4, 16 | 8k / 40k / XL | 592 / 526 / 363 (9) | 3.30 / 3.31; step **89.2 / 100.9** | 917,816 | = ref | EMS |
| EMS W8 | DFlash2 W8, 16 | 8k / 40k / XL | 573 / 477 / 363 (7) | 5.00 / 4.47; 138.8 / 149.9 | 786,488 | worse at 40k | EMS |
| EMS W6 S24 | DFlash2 W6, 24 | — | — | — | 706,232 | **out of memory + traceback, CAPPED** | EMS |
| EMS MTP 3/1/4 | EAGLE MTP, 16 | 8k / 40k / XL | 478 / 402 / 296 (11) | 2.88 / 2.94; 96.4 / 117.3 | 1,142,688 | 0.82× DFlash | EMS |
| EMS MTP 3/1/4 + #45151 | duplicate embed/lm_head freed | 8k / 40k / XL | 492 / 394 / 295 (12) | 2.98 / 2.90 | **1.29M vs 1.14M (+13 %)** | no speed gain | EMS RISKY |
| EMS MTP 2/1/3 | 16 / 20 | 8k / 40k / XL | 509 / 403 / 269 (16); **542 / 413 / 263 (20)** | 2.47 / 2.45; step 77.7 / 97.6 (16) | 1,183,245 / 1,120,156 | best MTP | EMS |
| EMS MTP mem 0.86 | MTP 3/1/4 | 8k / 40k | 483 / 408 | 2.98 / 3.00 | 1,223,053 | — | EMS |
| EMS MTP ReplaySSM fp16 / fp32 / fp32 S20 | no_buffer replay | 8k / 40k / XL | 497/410/313; 501/415/288; 543/428/275 | 2.9-3.0 | — | not adoptable (#43387/#43749 risk) | EMS RISKY |
| EMS NTN flags | INT4 / FP8 | 8k / 40k / XL | 444/363/230; 400/343/151 (4) | ~2.9 | FP8 only 419k | 0.78× / 0.70× | EMS RISKY |
| EMS target-only | no spec, 16 | 8k / 40k / XL | **405 / 354 / 275 (15 streams)** | —; step **39.5 / 45.1 / 54.6 ms** | **1,576,170** | oracle | EMS |
| EMS W6 + PCIe-IPC AR | FlashInfer 0.7 | 8k / 40k / XL | 230 / 202 / 233 | 4.32 / 3.91; step 299.6 / 309.9 | 852,152 | **0.39×** | EMS |
| EMS MTP + PCIe-IPC | — | 8k / 40k | 248 / 205 | — | — | 0.42× | EMS |
| EMS 2×TP2 DFlash W6 | 8 / 3 per replica | 8k / 40k | 658 (sum) / 308 (sum) | — | **152k per replica** | 1.15× at 8k, 0.57× at 40k | EMS RISKY |
| VMS (vLLM 0.32) ref | FP8 MTP3, 16 | 8k / 40k / XL | 434 / 399 / 281 (10) | 2.96 / 3.05 / 3.08 | — | PASS (BOUNDARY) | `E24/Base Engine/e22_speedlab_sglang/ENGINE_FINDINGS_vLLM032_SGLang.md` §1 |
| VMS | FP8 MTP2 / MTP4, 16 | 8k / 40k / XL | 448/397/260; 399/381/267 | — | — | MTP3 best at XL | same |
| VMS | FP8 MTP3 SSM fp16, 16 | 8k / 40k / XL | 450 / 414 / **312 (11)** | — | — | more streams fit at XL | same |
| VMS | INT4 MTP3, 20 | 8k / 40k / XL | 518 / 439 / 332 | — | — | best vLLM | same |
| E23 SMOKE (vLLM, Merlin) | Merlin W4A16 27B, TP4, MTP4 | 40k@8 / 120k@8 | **421.5 / 329.0** | **4.18 / 4.23**; step 79.4 / 102.8 ms | 1,789,034 | server default reasoning_effort "medium" (must be matched to xhigh) | `experiments/Experiment 23 - Six levers vLLM 0.32 (planned)/Experiment Results/smoke_rev3g/SMOKE_REV3G_RESULTS_2026-10-30.md` |

### 1.2 Production runs (real ARC workloads, whole-run averages)

| Experiment | Config | Streams | Whole-run tok/s | Accept | KV | Failures / notes | Evidence |
|---|---|---|---|---|---|---|---|
| E11-05 EVAL30 (S49 Poetiq) | W1 (DFlash2 W8, fp16 SSM, lazy, mamba 56, mem 0.83) | 12 set, **11 real** | **328.9** (decode by ctx <10k 429, 10-20k 372, 20-40k 340, 40-60k 323) | median 4.55 | 881,336 | 2,358 decode lines at 11+1 queued vs 114 at 12; no retractions | `E11R/E11-05 run1 analysis (2026-10-18).md` |
| R5 EVAL120 (vLLM reference) | vLLM 0.19 FP8 + MTP-3 | 16 | 280.9 | — | — | — | same file |
| E11-06 P1-60K EVAL14 | W1 rev 3c | max 12 | 379.9; 87.2 % cached prompt tokens | — | — | 0 errors | `E11R/E11-06 P1-60K EVAL14 run1 analysis (2026-10-19).md` |
| E11-05 S60 6H (cancelled at ~35 min) | W1, 12 running | 12 | 325-405 (partial log); READY 512 s | — | — | redundant run | `E15N/17 Claude - Duck vs Poetiq vs Direct oracle and overlap (2026-10-25).md` §8 |
| EVAL120G4B XE16 / XNO / LOW (Duck) | DFlash2 W6, mrr 13, graphs 1..13, mamba 56, delayer default | 12 offered, mean **11.21** | **344.0 / 341.4 / 317.8** (pure decode at 12: 434.5 / 434.9 / 445.2) | p50 3.9 / 3.8 / 4.0 | 917,816 | 11-of-12 delayer (73 % of time), prefill 19-27.5 % of window, throughput falls 18-30 % after hour 2 | C_engine §2-3 |
| E15B 45K30K commit (Duck, EVAL20) | W6, delayer off, mrr 17, graphs 1..17, mamba 68 | 16 | gen p50 by running: 16: **451**, 15: 470, 14: 433, 12: 383, 8: 312, 2: 144, 1: 78 | p50 3.8 | 852,152 | queue 0, 0 retract/OOM; cache hit 94 %; KV peak 0.71 | `E15N/11 Claude - 45K30K and 40K35K GPU commit results (2026-10-25).md` |
| E15B 2X35K run (partial) | same | 16 | 257-311 first, then 430-442 | — | 852,152 | "16 streams give no aggregate decode gain" over 12-stream E14S (415-434) in that sample | `E15N/10 Claude - 2X35K GPU run findings and rev 3h (2026-10-25).md` lines 19-25 |
| E15P Poetiq EVAL120 6 h | same engine, 15 streams (KV/req 56k; 16 = retraction risk) | 15 for 3,743 of 3,790 samples | **389** over 20,127 s | **median 3.4** | peak token_usage 0.83, mamba 0.72 | 0 retractions; 99/120 iteration-1 force-closed at 60k; only 18 tasks reached iteration 2 | `E15N/20 Claude - E15P rev 2 EVAL120 6-h result (2026-10-26).md` |

### 1.3 Step-time and memory decomposition

| Item | Value | Evidence |
|---|---|---|
| Profiler, DFlash2 W8, 12 streams, ~5.9k ctx, TP rank 0 | kernel sum 116.1 ms: NCCL all-reduce 140 calls **50.9 ms (44 %)**, all-gather 8.1 (7 %), Marlin INT4 GEMM 25.3 (22 %), other GEMM (drafter, lm_head) 14.5 (12.5 %), GDN Triton 8.0 (7 %), attention 2.7, sampling 2.4 | `E14/Notes/30`; `E14/README_FINDINGS.md` §1.2 |
| NCCL bench | 343.9 µs per 983 KB all-reduce in a graph (SHM, 2 channels); forced P2P 1,880 µs (5.5× slower); 64 MB in 20.9 ms | `E14/README_FINDINGS.md` §1.2; E11-03 report preflight |
| Custom all-reduce | disabled at TP4 on PCIe-only GPUs; at world size 2 it turns on only if the P2P test passes | `E14/README_FINDINGS.md` §2.1 [S] `custom_all_reduce_utils.py:442-512` |
| Step fit over widths 4/6/8 | step ≈ 26 ms + 0.94 ms per verify row (±3 ms); weights only ~16-22 ms of a 107-125 ms step | `E24/Base Engine/e22_speedlab_sglang/ENGINE_FINDINGS_vLLM032_SGLang.md` §2 |
| Marginal cost per stream (production, W6) | ≈ +5.2 ms per added running request at 25-30k ctx; +3 ms per +100k live tokens | C_engine §3.1, §3.6 |
| Marginal cost per verified token (E11 logs) | ~0.85-1.3 ms; no-spec 21.9 ms @1 → 37.9 @12; MTP 40.3 @1 → 79.9 @10; DFlash2-8 39.9 @1 → 115.5 @12 | `E14/Notes/00 Claude - step-time analysis and TP2 gain (2026-10-23).md` §1 |
| Prefill rate | ~1,575 tok/s (regression), decode fully stopped during a 4,096 chunk (mixed chunk off) | C_engine §3.4 |
| E11-03 D memory per GPU | target weights 4.62, drafter 1.49, mamba 3.26 (ssm 1.21 + intermediate 1.97 + conv 0.08), target KV 2.94, draft KV 0.92, prefill graph 2.60, verify graphs 0.50 + 0.38, free 4.39 GB | `E11R/E11-03 run1 analysis (2026-10-17).md` §6 |
| E14S/E15 W6 memory per GPU | mamba 56 slots ≈ 46 MB/slot/GPU (ssm 1.00 + intermediate 1.48 GB); KV 917,816 (target K+V 7.0 GB, draft K+V 2.18 GB); final free 1.78 GB | C_engine §1 |
| KV bytes per token per GPU | target 8,192 B (16 attn layers × 1 KV head × 256 × fp8 × K+V); DFlash2 draft +2,560 B (+31 %); MTP draft +512 B | `E14/Notes/35` §3 |
| Mamba slots per request (SGLang 0.5.21) | radix off 1; no_buffer 3; extra_buffer + overlap 5; extra_buffer_lazy 4; −1 with `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK`; in production ~3 in use per running request | [S] `kv_cache_configurator.py:2232-2261`; C_engine §4 |
| Startup | first target-verify graph capture (bs 1..13) 375 s, bs 13 alone 364 s; LAUNCH → READY ~530 s; READY ~12-13 min | C_engine §1 |

---

## 2. GOOD (what worked, and why)

1. **Speculative decoding still pays at 16-20 streams on L4.** DFlash2 W6 gave 1.44× (8k), 1.47-1.51× (40k) and 1.35× (XL, at fewer streams) over target-only (EMS 583/521/370 vs 405/354/275). In E14A R2 it was 1.49×. Why: the L4 step is not weight-bandwidth bound. A target-only step is already 35-45 ms of mostly fixed overhead (NCCL, launch). Extra verify rows cost only ~0.94 ms each, so accepted tokens per step grow faster than the step. This is the "MagicDec regime" the research predicted (`E14/Notes/Research - Qwen 27B fastest serving setup/speculative_decoding.md` Q2 inferences). Our measurements confirm it.
2. **Acceptance does not fall with concurrency.** DFlash2 W8 accept was 4.31 / 4.51 / 4.85 / 4.62 / 4.78 at c1/4/8/12/16, and NEXTN MTP 3/1/4 was 3.17 / 3.10 / 3.09 / 3.11 / 3.17 (E11-03 report). What falls is the speed-up, because the verify cost scales with streams × width. Acceptance is a per-request property; concurrency enters only through the step time.
3. **Acceptance decays only slowly with context, and never collapsed.** W6 in production: 4.02 → 3.72 (−7.5 %) from <10k to 35-40k; LOW 4.46 → 3.92-3.97. No 40-step line was ever below 3.05, with no drift toward 1 (C_engine §5-6). E15P at 60k thinking: median 3.4. So the literature's "acceptance collapses to 1-2 after 10-25k tokens" (arXiv 2605.09329, cited in `speculative_decoding.md` Q1) did not happen for DFlash2 or MTP up to ~60k on our stack.
4. **Narrower verify width is free or better at high batch.** W4 = W6 = W8 at 8k, and W8 loses at 40k (477 vs 521-526). W4 has the shortest step (89-101 ms) and a larger KV pool (917,816 vs 852,152), because the per-request reserve is 2 × width (`E14/Notes/Research - 4xL4 engine bottleneck validation/dflash_width_correctness.md` Q1, [S] `dflash_info_v2.py:141-147`).
5. **Setting the state/memory flags explicitly unlocked real concurrency.** fp16 SSM, `extra_buffer_lazy`, an explicit `--max-mamba-cache-size`, no prefill CUDA graph and mem 0.83 together took the DFlash config from 6 running / 395k KV (E11-03 C) to 12 running / 881k KV (E11-04), and later to 16-20 running. Each auto default hid a cap (`verify-real-concurrency-in-benchmarks` memory; E11-03 analysis §1).
6. **fp16 SSM was faster and showed no measured quality loss on our probes.** E11-03 D fp16 404 vs C fp32 307 at 12, because fp32 capped at 6. In E11-04 fp32 cost 32 % of the KV pool (598k vs 881k). The fp16 W1 sanity was 8/8 with marker PASS, and C_engine §6 found no numerical trouble up to ~52k tokens per request.
7. **Delayer off plus graphs/mrr sized above the stream count gave real 15-16 of 16.** `--min-free-slots-delay 1` with mrr 17 and graphs 1..17 removed the 11-of-12 loss. E15P ran at 15 running in 3,743 of 3,790 samples, with 0 retractions (`E14/README_FINDINGS.md` §1.4).
8. **The radix prefix cache on the hybrid model works.** Production hit rates were 76-94 %. Short forced-close continuations hit ≥ 0.965-0.979 and ran 10-11× faster than cold. Forced-close phase 2 hit 98.1 % (E11-03/04/05; `E14/README_FINDINGS.md` §1.3b).
9. **Correctness gates on spec decoding passed** under a margin oracle: every EMS/VMS arm PASS (BOUNDARY), 0 large-margin flips vs target-only, XL (~90k) parity PASS, canary leaks 0 (EMS findings 7). The E11-04 W1 marker test passed 8/8 with no cross-request leak. E11-03 "contam" was batch non-determinism, not contamination (E11-03 analysis §3).
10. **The two-term production model explains the bench-to-run gap.** Per generated token cost = 1/tput_decode + r × c, where r is new prefill tokens per generated token and c ≈ 0.635 ms/token. XE16: 1/423 + 0.888 × 0.635 ms → 341.6 tok/s predicted vs 344.0 observed (C_engine §3.4). This is a ready-made planning tool for FN.

## 3. BAD (what failed or hurt, and why)

1. **Silent caps from the mamba state pool.** With spec decode, SGLang needs 4-5 state slots per request. The auto pool capped MTP at 10 and DFlash fp32 at 6, while `/get_server_info` still said 32. Every sweep level above the cap was invalid, and I first mis-explained it (CUDA graphs, step cost) before reading the logs (E11-03 analysis §1; memory `verify-real-concurrency-in-benchmarks`).
2. **The 11-of-12 loss.** SGLang's DFlash-only min-free-slots delayer holds fresh prefill while fewer than 3 slots are free: threshold min(4, max(2, (mrr+5)//6)) = 3 at mrr 13 ([S] `managers/min_free_slots_delayer.py:20-21`, `scheduler.py:3856`). The run spent 71-73 % of the time at 11 + 1 queued, costing 2.2-2.5 % of tokens (C_engine §2).
3. **Disabling the delayer exposed a crash.** At 13 running, DFlash buffers sized by `--cuda-graph-max-bs-decode 12` raised `tensor a (12) must match b (13)` (`dflash_worker_v2.py:273`). Lesson: graph/buffer capture sizes must cover the maximum running count (E14/Notes/30).
4. **Spec decoding costs KV and state memory, which cuts residency at long context.** KV pool: target-only 1,576,170; MTP 1,142,688; DFlash W6 852,152 (EMS). In the XL band DFlash ran only 7-8 streams vs target-only 15, which shrank its gain to 1.35×. W6 at 24 streams hit OOM / CAPPED. 2×TP2 had only 152k KV per replica, i.e. 3 streams at 40k.
5. **The drafter weight footprint.** DFlash2 BF16 is 1.49 GB per GPU. The EAGLE/MTP head loads a duplicate BF16 embed + lm_head (~1.19 GiB/GPU at TP4) before the KV pool is sized (#44145 / PR #45151, [S] `scheduler.py:1049-1075`, `eagle_worker_v2.py:226-241`). The #45151 patch recovered +13 % KV but no speed (EMS RISKY).
6. **MTP (EAGLE/NEXTN) on SGLang lost to DFlash2 on our 27B:** 0.82-0.95× at 16-20 streams (EMS). MTP 3/1/4 was 0.91× of no-spec at c12 in E11-03, although that run was capped at 10 with fp32 SSM, so it was not a fair comparison (Notes/35 §0). MTP runs k sequential draft passes, each with a full-vocabulary lm_head.
7. **Communication dominates the step,** and every "faster all-reduce" path we tried was slower: FlashInfer 0.7 PCIe-IPC 0.39-0.42× (SGLang), 0.68× (vLLM); forced P2P 5.5× slower; `NCCL_SHM_USE_CUDA_MEMCPY=1` slower and timing out in graphs; NCCL env arm no gain (`E24/README.md` "Measured negatives").
8. **Prefill blocks decode.** With mixed chunk off, a 4,096 chunk stops all decoding. Prefill was 19-27.5 % of the window, and 58-66 % of it was re-prefill of evicted paused branches. Throughput fell 18-30 % after hour 2 (C_engine §3.4-3.5, §8.1). The bench-to-production ratio was ~0.65-0.79 (583-597 bench at 16-20 vs 344-389 whole-run).
9. **LB accuracy gap on SGLang, cause still unknown.** Poetiq on SGLang INT4 + DFlash2: P1-60K **29.17**, E15P **26.94**. vLLM FP8: S1 family 34.58 / 32.08 / 31.81 / 31.94 / 31.53 (`experiments/Experiment 21 - Duck config B on vLLM 0.19 + 0.32 (planned)/README.md:11-13`; memory `experiment-20-s1-drain15`). Every SGLang / INT4 / DFlash submission scored 17.50-29.17 (`experiments/Experiment 17 - SGLang silent-accuracy tests (planned)/Experiment Results/README_E17_BUGHUNT_RESTART_FOR_GPT_2026-10-27.md` §1). Eval per-request quality looked equal: R5 vs E15P iteration 1 gave 23 vs 22 train-verified on 26 tasks (E21 README:15-16), and EVAL120 E15P 39.31 % = Duck R3K2 39.03 %. The gap is confounded by prompt, cap and coverage, and E17 never isolated it.
10. **E17's bug hunt never reached its target.** All four arms crashed with `Can not alloc mamba cache` (the #43559 path) before the #43567 stale-checkpoint path could occur. Verdict NOT_EXPOSED_OR_INCONCLUSIVE (`.../Experiment 17 .../Experiment Results/README_E17_SHORT_GPU_RUN1_FOR_GPT_2026-10-26.md`).
11. **FP8 KV ran with unit scales.** The log says "Using FP8 KV cache but no scaling factors provided. Defaulting to 1.0", although the RedHat checkpoint carries 32 calibrated k/v scales (0.021-0.254). Runtime use was never verified (E11-04 report header; E17 README §"Our reading" 3).
12. **SGLang on Flash-Next failed outright on 4×L4.**
    - E3 (Pennyroyal 0.5.19): "!!!!" at ~6 tok/s.
    - E13 SMOKE1 (0.5.21): OOM in `gptq_marlin_moe_repack`.
    - LOADPROBE: "!!!!" (token 0) at 19.10 GB per rank, KV 94,528.
    - RC1: EP4 GPTQ-Marlin MoE gives wrong expert outputs from layer 1 and NaN at layer 2.
    - Sources: memory `e13-flashnext-smoke1-oom`; `PLAN_1500.md` §2.
13. **Hybrid + spec bugs in our class, open upstream:**
    - #36548: DFlash2 + concurrency gives 36.5 % wrong answers on NVFP4; width 4 / fp32 SSM / no decode graphs each left 37/304; a BF16 lm_head gave 1/304; closed stale 10-27.
    - #35150: GDN TARGET_VERIFY vs DECODE drift.
    - #38009: greedy divergence with thinking on, and across restarts.
    - #43514: accept → 0 with "!" output under grammar + multi-chunk prefill.
    - For Flash-Next: #37326 (NEXTN acceptance decays to ~0 over uptime; only a restart restores it) and #37111 (QSA + NEXTN decode graph silently corrupts output to "!" on GB10).
    - Sources: `dflash_width_correctness.md` Q4; `E24/Resources/sources/research/agent_reports/specdecode_and_kernels.md` Q2; `.../engines_flashnext_duck.md:36-45`.

## 4. Speculative decoding lessons

### 4.1 Acceptance vs concurrency
- Accept length is flat across c1-c16 for both DFlash2 and MTP (E11-03). The design variable is therefore the step-time cost of verify rows (streams × (1+k)) and the memory each request's draft state takes, not acceptance.
- Context: −7 to −11 % from <10k to 35-40k for W6 (C_engine §5). Long single-thread thinking (Poetiq 60k) median 3.4 for W6 vs 3.8-4.0 in the Duck's shorter requests (E15P vs C_engine §3.3). Plan with production accept, not bench accept: W6 bench 4.1-4.2, production 3.4-3.9.
- Sampling temperature changes acceptance: T0.6 4.60 vs T1.0 4.34 (E11-03; that comparison was capped, so speed was inconclusive).
- vLLM MTP3 on 27B: accept 2.96-3.08 (VMS). Merlin MTP4 on 27B: **4.18-4.23** at 8 streams, 40-120k (E23 SMOKE). Merlin's MTP acceptance was much higher than SGLang NEXTN 3/1/4 (≈3.0). Caveat: the server default was reasoning_effort "medium".

### 4.2 Drafter cost
- **DFlash2 (27B):**
  - 5 layers, BF16, 1.49 GB per GPU;
  - draft KV +31 % per token;
  - "other GEMM (drafter + lm_head)" 14.5 ms of 116 ms (12.5 %);
  - verify width W means W−1 drafts + 1 bonus, max accept W.
- **MTP head (27B):**
  - 1 full-attention layer, about 0.29 GB, plus a duplicate embed and lm_head (~1.2 GB per GPU until #45151);
  - k sequential draft passes, each with a full-vocab head;
  - draft KV +512 B per token.
  - Notes/35 §1.5 and §3.
- **Step model on L4:** step ≈ 26 ms + 0.94 ms per verify row (27B INT4 TP4). Target-only at 16 rows ≈ 40 ms; W6 at 16 × 6 = 96 rows ≈ 116 ms.
  - So 1500 tok/s at 20 streams × 4.2 needs a ~56 ms step, and at 7 resident streams (XL) ~20 ms (`E24/README.md` "What 1500 tok/s requires").
- **Best width at batch 12-20:** about 4-6 (W4 = W6 > W8), matching DFlash paper advice for compute-bound verification.
- **Best MTP depth at batch 16-20:** MTP2 ≈ MTP3 > MTP4 at 8k-40k; MTP3 is best at XL (VMS).
- **Third-party Flash-Next data [C]** (vLLM, 3×4090 48 GB, NVFP4, MTP depth at 16 streams): K0 559.0, K1 636.4, K2 **644.5**, K3 541.1, i.e. "MTP=3 loses at 16-way" (`engines_flashnext_duck.md:63`).
- **SGLang FN cookbook [C]** (RTX PRO 6000, 1024 in / 256 out):
  - MTP vs no-MTP at 16 streams: 613 vs 524 tok/s;
  - no-MTP at 64 streams: 861;
  - MTP accept 3.3 of 4 (2.9 on long-form ShareGPT, ~2.5 on thinking output per the TP2 cell);
  - with MTP only ~20 requests fit by state-pool arithmetic vs ~70 without;
  - source: `_scratch/e13_validate/cookbook/flashnext_main.mdx:42,54,58`.

### 4.3 Concurrency and memory interplay
- Speculation multiplies per-request recurrent state: intermediate SSM states per draft position. On SGLang, intermediate state = (min(mrr, K//4)+1) × D × ssm ([S] `kv_cache_configurator.py:2462-2482`).
- A 1.4-1.5× spec gain is cancelled if it halves residency: XL 7-8 streams with DFlash vs 15 without.
- FN cookbook: fp32 FN state is 0.109 GiB per slot at TP1 (bf16 0.055). The default `extra_buffer` allowed 3 requests with MTP vs 12 without on 96 GB.
- `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` gives 3 slots per request ("trades cache retention, not numerics").

### 4.4 Correctness
- Never require token identity across a server restart: DFlash2 greedy output differs across restarts (#38009). Use a target-only oracle with a top-1/top-2 margin rule (large-margin flip > ~0.5-1 nat = bug), an A/A baseline, and a cross-request canary (E14 Build 2 rules in memory `experiment-14-optimized-engine`; EMS gate definitions).
- Silent-failure signatures to monitor:
  - accept 1.00 + token-0 "!" runs (#43514, and FN #37111/#37326);
  - accept ≈ 1.03 from a quantized drafter: SGLang must get `--speculative-draft-model-quantization unquant` for DFlash, #39087;
  - accept ≈ 1.06 from a layout shift (#40144);
  - accept 1.00 + endless output with repetition_penalty (#28180).
  - Source: `dflash_width_correctness.md` Q5.
- vLLM: MTP with prefix caching OFF failed (4-12 nat flips), while MTP + prefix ON passed (E19, memory `experiment-19-vllm-correctness`). Never turn prefix caching off as a "safety" flag while MTP is on.
- Short gates (256-1,024 tokens, 75-s windows, XL parity) do not certify 60k-120k reasoning. The SGLang LB gap shows that passing short gates is not enough (`E14/README_FINDINGS.md` §7; `E24/README.md` "Known accuracy problem").

## 5. Transfer to Flash-Next (Merlin v3.0.0, `fn-quad-24g-sm89`, MTP K3)

| # | Lesson from the 27B work | Applies / not / adapt | How, for E25 |
|---|---|---|---|
| T1 | Read the real running count, KV pool, state pool and graph sizes from the boot log; never trust requested levels | **Applies** | Already in the PLAN_1500 §4 notebook rule. Add: assert running == level in every steady window, and record `num_running`, `num_waiting`, preemptions, KV blocks free per window. |
| T2 | Capture/buffer sizes must cover max running (DFlash crash at 13 with graphs sized 12) | **Adapt** | Merlin captures 1..32 with seqs 8. If vLLM capture sizes count tokens under spec (8 × (1+3) = 32), then seqs 10/12 (lever L7) need capture ≥ 40/48 or the batch falls back to PIECEWISE/eager. PLAN open question 4 must be answered from the RC2 log before L7. **UNVERIFIED** for Merlin. For SGLang, verify graphs are captured per request batch with `num_tokens_per_req` = draft tokens (Notes/35 §5 item 6). |
| T3 | Speculation still pays at high concurrency on L4 (1.4-1.5×) | **Adapt (likely stronger for MoE)** | [I] Flash-Next is ultra-sparse MoE (512 experts, 10 routed + 1 shared; `engines_flashnext_duck.md:92`). Each verify row routes to new experts until the union saturates. Under uniform routing, the expert fraction touched per layer per step is 1 − (1 − 10/512)^T, for T tokens in the step: **T = 12: 21 %; 32: 47 %; 48: 61 %; 80: 79 %; 128: 92 %**. So extra rows are expensive at low T and become almost free once most experts are read. This argues for high concurrency × MTP depth at once, which is the opposite of the dense 27B where every row cost ~0.94 ms. It must be measured: real routing is skewed (fewer unique experts), and the MTP head's own MoE layer runs k sequential passes. |
| T4 | Expert-weight read sets a step floor | **New for MoE** | [I] With on-GPU main weights of ~16-18.5 GB per rank (UNVERIFIED; take it from the RC2 boot log), reading 61 % of the expert bytes at T = 48 costs ~30-38 ms at 300 GB/s. The PLAN §1 example (12 streams × 3.0 / 24 ms) assumes a step below that floor. Equivalent 1500-tok/s combinations above the floor: 20 streams × 3.0 / 40 ms, or 24 × 3.0 / 48 ms, or 16 × 3.5 / 37 ms. So more resident streams (L1, L5, L6, L7) is the lever that scales, because the MoE step grows sublinearly with T. |
| T5 | MTP depth: at 16-20 streams MTP2 ≈ MTP3 > MTP4 on the 27B (VMS); FN 3×4090: K2 644 > K1 636 > K0 559 > K3 541 at 16 streams | **Applies** to lever L3 | Sweep K1/K2/K3 at each stream count. Do not assume K3 is best at ≥ 12 streams. Expected accept on thinking output ≈ 2.5 of 4 for FN (cookbook), and Merlin's 27B MTP4 reached 4.2, so measure on xhigh ARC prompts, not the default "medium". |
| T6 | Draft head memory: the duplicate embed/lm_head (#44145) and the k full-vocab passes | **Adapt** | Merlin `DRAFT_HEAD=int8/int4` (lever L4) is the matching lever. FN's MTP layer is a MoE layer: albucino's drafter is `mtp-dense` + `mtp-routed-experts-int4`, 4,139,535,872 B (`experiments/Experiment 13 .../PLAN.md:110`, `VALIDATION.md:13`). Franzen also used an FR-Spec 64k hot-token map to cut draft lm_head cost (`E14/Notes/Speed review 2026-10-24/part_B_fastest_serving_research.md:746-760`). Check whether Merlin ties the draft embed/head to the target's or loads copies; measure the per-rank VRAM delta. |
| T7 | "Possible DFlash drafter for Flash-Next by albucino (4.2 GB)" | **Correction: not a DFlash drafter** | The Kaggle model `dfranzen/albucino-qwen3-8-flash-next-drafter` holds `mtp-dense.safetensors` and `mtp-routed-experts-int4.safetensors` (`_scratch/e13_validate/inputs_listing.json`). Franzen ran it as `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`, draft quantization compressed-tensors (part_B :750-760). It is an **INT4-expert MTP head**, a candidate replacement for Merlin's own draft head, not a DFlash block drafter. Also, vLLM issue #56088 "qwen4_exp cannot serve DeepSpec DFlash/DSpark drafters: five blockers" is open (`engines_flashnext_duck.md:29`), so DFlash on Flash-Next is blocked on vLLM/Merlin regardless. The 4.2 GB naming in PLAN/prompt (`00_RESEARCH_PROMPT_2026-10-31.md:17`) should be fixed. |
| T8 | DFlash width lessons (W4-W6 best, W8 worse at long ctx) | **Not applicable** directly | No FN DFlash drafter is known to load on Merlin (T7). Keep as a lesson only: narrower verification is better at high batch. |
| T9 | fp16 SSM gave +residency and passed our probes; fp32 cost 32 % of KV | **Adapt** to lever L5 | On vLLM 0.32 the fused GDN spec-verify kernel needs BF16 or FP32 state; fp16 falls back to Triton, yet it was still faster at XL because more streams fit (ENGINE_FINDINGS §3; VMS 312 vs 281). For Merlin: read the active SSM dtype and GDN kernel from the RC2 log; prefer bf16 if it keeps the fused kernel; gate on long-reasoning correctness (SGLang cookbook: "treat bf16 as an accuracy gate"). |
| T10 | Mamba slots per request multiply with spec (SGLang 4-5 per request; FN fp32 0.109 GiB per slot at TP1) | **Adapt** | Merlin uses vLLM `--mamba-cache-mode align`. How vLLM stores per-draft GDN states under MTP K3 (×(1+k)?) is an open gap (`speculative_decoding.md` Q5 gaps). Measure state bytes per stream from the RC2 boot log at K3 vs K1. |
| T11 | Prefill stops decode; re-prefill of evicted contexts was 58-66 % of prefill; prefill 19-27.5 % of the window | **Applies (stronger)** | FN prefill rate on L4 is unknown. FN cookbook: 100k prefill at 2,400-2,840 tok/s on another GPU; Franzen P90 10.5k on RTX PRO 6000. Lever L2 (chunk 512/1024/2048) should measure decode TPOT p95 during prefill and the r × c term. Use the two-term model: sustained = 1 / (1/decode + r × c). For 1500 sustained at r ≈ 0.9, the decode peak must be well above 1500 (e.g. c = 0.4 ms/token → 1/1500 − 0.36 ms ≈ 0.31 ms → decode ≈ 3,200 tok/s). Prefix reuse (keep resumed contexts resident, lever L10) is therefore as important as decode speed. |
| T12 | All-reduce = 51 % of the step at TP4 on PCIe; IPC/P2P/memcpy tricks are slower; NCCL SHM ~344 µs per ~1 MB | **Applies** to lever L8 | TP2×PP2 halves the TP world. At TP2, custom all-reduce may switch on only if the P2P test passes, and P2P was 5.5× slower on this VM: keep it disabled unless measured. EP adds all-to-all over PCIe. Profile the step once (as E14A did) before tuning NCCL. Do not retry PCIe-IPC, forced P2P or `NCCL_SHM_USE_CUDA_MEMCPY`. |
| T13 | The delayer (SGLang DFlash-only) | **Not applicable** | Merlin is vLLM, but check vLLM's admission behaviour: running < seqs while waiting > 0. A request counts as running once its first prefill chunk is scheduled (`E14/README_FINDINGS.md` §2.2). |
| T14 | Prefix caching with MTP: OFF fails (E19 vLLM 0.19); 0.32 align mode hit only 55-56 % in Duck vs 69-82 % on 0.19 | **Applies** | Keep prefix caching on (Merlin default). Measure the hit rate per band. Do not use prefix-off as a control while MTP is on. |
| T15 | Startup: first verify-graph capture ~6 min; READY 12-13 min | **Applies** | Merlin boot time per arm adds directly to the 12 h. Budget it and reuse compile caches only if they are valid. Under spec decode a capture list of 1..32 can be long. |
| T16 | Correctness gating for spec | **Applies** | Margin oracle vs MTP-off (and BF16 for the RC2 FP8 KV arm), A/A, canary, non-finite logprob gate (RC2 H2). Add the FN silent-failure signatures: accept trending to 1.0 over uptime (#37326), token-0 runs (#37111). Plot acceptance per 10-minute window over the whole 12 h run. |
| T17 | Bench vs production ratio 0.65-0.79 | **Applies** | Report the three denominators separately (saturated decode, mixed serving window, agent end to end), as the E24 plan required. |
| T18 | SGLang as an FN engine | **Not now** | 0.5.21 EP4 GPTQ-Marlin MoE is wrong on L4 (RC1). The QSA SM89 path needs classic FA2 (guard PR #36968 open). Revisit only with a fixed MoE path. |

## 6. Open questions

1. **Merlin capture semantics under MTP:** are capture sizes counted in tokens (seqs × (1+k)) or in sequences? What happens at seqs 10/12 with capture 1..32: a PIECEWISE fallback or eager? (PLAN Q4, lever L7; T2.)
2. **The real MoE step floor on L4:** step ms vs tokens per step (T) for FN, from the RC2 decode lines at several stream counts and K. Is the unique-expert curve close to the uniform-routing estimate (T3/T4)?
3. **Per-draft GDN state under vLLM align mode:** bytes per stream at K1/K2/K3 and SSM fp32/bf16/fp16. Does fp16 or bf16 keep the fused verify kernel in Merlin? (T9/T10.)
4. **FN MTP acceptance on xhigh ARC thinking at 60-120k:** is it ~2.5 of 4 (cookbook) or ~4.2 of 5 (Merlin on the 27B)? Does it decay over uptime (SGLang #37326 analogue)?
5. **Does `DRAFT_HEAD=int8/int4` change acceptance on L4?** Merlin's CHANGELOG claims −6.5-7.2 % ms/token at the same acceptance; not measured on L4. Can the albucino INT4-expert MTP head be loaded into Merlin, and does it beat Merlin's own head? (T6/T7.)
6. **Best K per stream count** (K1/K2/K3 at 5/8/12/16 resident streams), given the 3×4090 FN result that K3 loses at 16-way.
7. **The FN prefill rate on 4×L4 under TP2×PP2** and the decode stall per chunk, which set the r × c term (T11).
8. **The SGLang LB gap** (29.17 / 26.94 vs ~32) is still unexplained. It matters for E25 only as a warning: a speed winner must pass a paired ARC accuracy run (MTP on vs off, same tasks), not only short parity probes.
9. **FP8 KV scales:** our SGLang runs logged unit scales despite calibrated checkpoint scales. Which scales does Merlin's FP8 QSA KV use, and is it gated against BF16 KV on long contexts (RC2 compares BF16 and FP8 boots)?
10. **Delayer analogue in vLLM:** does the Merlin scheduler keep running == seqs under churn (agent turns finishing and new prefills), or does it park requests the way the 11-of-12 pattern did?

---

### Source index (all read for this report)
- `E11R/E11-03 run1 analysis (2026-10-17).md`, `E11-04 run1 analysis (2026-10-18).md`, `E11-05 run1 analysis (2026-10-18).md`, `E11-06 P1-60K EVAL14 run1 analysis (2026-10-19).md`; `E11R/Kaggle Runs/E11-03 rev3b run1 (foysal, 2026-10-17)/e11_03/report.md`; `E11R/Kaggle Runs/E11-04 rev2b run1 (esfersami50, 2026-10-18)/e11_04/report.md`; `experiments/Experiment 11 .../README.md`.
- `E14/README_FINDINGS.md`; `E14/Experiment Results/EMS_SAFE_RISKY_VMS_rev2g_GPU_results_2026-10-26.md`; `E14/Notes/00`, `/30`, `/35`; C_engine; `E14/Notes/Research - 4xL4 engine bottleneck validation/dflash_width_correctness.md`; `E14/Notes/Research - Qwen 27B fastest serving setup/speculative_decoding.md`; `E14/Notes/Speed review 2026-10-24/part_B_fastest_serving_research.md` (lines 740-760).
- `E15N/11`, `/10` (grep), `/17` §8, `/20`.
- `experiments/Experiment 17 .../README.md`; `.../Experiment Results/README_E17_SHORT_GPU_RUN1_FOR_GPT_2026-10-26.md`; `.../README_E17_BUGHUNT_RESTART_FOR_GPT_2026-10-27.md`.
- `E24/README.md`; `E24/Base Engine/README_sglang.md`; `E24/Base Engine/results/ems_report.md`; `E24/Base Engine/e22_speedlab_sglang/ENGINE_FINDINGS_vLLM032_SGLang.md`; `E24/Resources/sources/research/agent_reports/specdecode_and_kernels.md` (Q1-Q2); `.../engines_flashnext_duck.md` (Q1-Q4 excerpts).
- `experiments/Experiment 23 .../Experiment Results/smoke_rev3g/SMOKE_REV3G_RESULTS_2026-10-30.md`; `experiments/Experiment 21 .../README.md`; `experiments/Experiment 13 .../PLAN.md:100-120`, `VALIDATION.md:8-20`.
- `_scratch/e13_validate/cookbook/flashnext_main.mdx` (lines 40-58); `_scratch/e13_validate/inputs_listing.json`.
- Memory files: `sglang-int4-dflash2-plan.md`, `experiment-14-optimized-engine.md`, `experiment-15-config-b.md`, `experiment-17-silent-accuracy.md`, `experiment-24-sglang-track.md`, `e15p-poetiq-duck-engine.md`, `verify-real-concurrency-in-benchmarks.md`, `experiment-19-vllm-correctness.md`, `experiment-23-six-levers.md`, `flashnext-vllm032-support.md`, `e13-flashnext-smoke1-oom.md`.
