# E23 FULL (esfersami50 v1, sha 674961ee) — 659 min, no GPU fault, 2026-10-31

Output: `_scratch/e23_full_sami/out/q27-xhigh-e23-six-levers-full/` (`e23/e23_results.json`, `report_md` inside).
Inputs: FP8 27B repacked, Merlin W4A16 27B, Merlin env, sm89 kits; **INT4 = None** (not mounted on this account) → the
INT4-dependent arms were skipped (PP4 e/f, section 4 INT4 arms).

| lever | arm | 40k | 90k | 120k | note |
|---|---|---|---|---|---|
| 1 Merlin TP1 ×4 replicas | x4 (rep1 / rep2) | 503.8 / 506.8 / 506.2 @16 | 262.4 / 264.9 / 266.7 @8 | — | best 27B aggregate; A/A ±0.6 % |
| 1 Merlin TP4 MTP4 | d | 456 @16, 472 @20, 503 @24 | 364 @16 | 324 @12 | accept 3.6-3.76; 90k@20 / 120k@16 stalled stream (1 preemption) |
| 6 stock vLLM FP8 MTP3 G0 PIECEWISE | mtp3 + 3 A/A reps | 359 / 355 / 353 / 361 @16; 367-370 @20 | 261-275 @12 | 202-203 @8 | A/A ±1 % |
| 6 MTP depth @16 40k | MTP0 / MTP1 / MTP3 | 222.5 / 319.2 / 359.1 | — | — | @20: 248 / 354 / 367 |
| 3 FULL graph @16 40k | G1 / G2 / G3 (PROVEN_FULL) | 330.7 / 334.9 / 364.7 | — | — | G3 ≈ +2 % vs G0 A/A mean 357; G1 / G2 −7 % |
| 5 NVFP4 KV target-only | 90k@12 | — | 169.8 (vs FP8 MTP0 142.6) | — | KV tokens 2.36M vs 1.41M; MTP3 arm SKIPPED_CORRECTNESS_GATE |
| 2 P5 host AR world 4, 983 KB | p5_bench | host 362-377 us vs NCCL 341 (0.94×); int8 238 us (1.43×) | | | BENCH_ERROR only from one sub-step `nccl_ctas16` (EADDRINUSE port clash); every correctness check true |
| 4 sm89 FP8 GEMM (P4) | p4_bench | Marlin W8A16 faster than CUTLASS W8A8 at M 1-8 (0.88-0.94×) | | | not a lever at decode M |

NVFP4 gate (40k): NVFP4 rank-1 0.9766, mean 0.139, p99 2.451 vs FP8 rank-1 0.9805, mean 0.060, p99 0.62 → p99 over the
3× limit (1.86) → fail; 90k worse (rank-1 0.945).
Cross-track insight: on the 27B, FP8 KV vs BF16 KV teacher-forced drift is small (rank-1 0.98, mean 0.06 at 40k), while
Flash-Next RC2 FP8 QSA KV showed rank-1 0.367 / mean 1.88 → the Flash-Next FP8 problem is specific to its QSA FP8 path (NB0).
