# 30 — E14A SLIM R2 run results (hosen42, 2026-10-23 17:33 → 19:13 UTC)

## The run

| Item | Value |
|---|---|
| Kernel | `hosen42/q27-xhigh-e14a-slim-optimized-engine-sglang` v2 (v1 hit a mount error, no quota used) |
| Notebook | sha 2298d827 (R2) |
| Status | **COMPLETE**, `terminal_reason: done`, 5,985 s (~100 min) |
| Quota | 36.58 → 40.53 = 3.95 h |
| Outputs | `Experiment Results/Kaggle Runs/E14A SLIM R2 run1 (hosen42, 2026-10-23)/` |

**Every phase ran.** Startup 502 s again. Relaunches: p44401 71 s, nodelay 71, graphs13 71, duck 61, a2 71, int8 66, target 76, width6 66, width4 76.

## Engine measurements (8k "short" band ≈ 5.9k context / request; "long" band ≈ 8.7k)

| Arm | tok/s (short) | step ms | accept | Long band | Decision |
|---|---|---|---|---|---|
| W1 baseline (TP4, DFlash2-8) | **505.0** | 114.0 | 4.86 | 449.8 tok/s, 123.1 ms, 4.62 | reference |
| #44401 backport | 508.1 | 114.3 | 4.84 | 455.2 | **not adopted**: no occupancy gain (still 11); files reverted |
| nodelay (`--min-free-slots-delay 1`) | 490.3 | 114.4 | 4.63 | 481.0 | **not adopted, and the server crashed** (see below) |
| graphs13 (cuda graphs 1..13) | 500.1 | 114.7 | 4.79 | 436.9; 13 not reachable (measured 12) | adopted (neutral 1.005) |
| target-only (no spec) | 339.6 | 35.3 | — | — | DFlash2-8 = **1.49×** target-only; parity PASS |
| **width 6** | **531.4 (+6.3 %)** | 96.4 | 4.24 | not measured | **adopted**, gate PASS |
| **width 4** | **526.8 (+5.3 %)** | 74.5 | 3.27 | not measured | adopted, gate PASS |
| int8 Mamba checkpoint | — | — | — | — | **not adopted: OOM** (`allocate 200 MiB, free 89 MiB`), 192 exceptions |
| NCCL env arm | — | — | — | — | none: the bench repeated 343.9 µs/call baseline |

**Width notes:**
- The width gain is small, because accept falls (4.79 → 4.24 → 3.27) while the step falls (114 → 96 → 74.5 ms).
- Width was measured only at ~6k context. Long-context behaviour is unknown.

**Latch (churn test; 14 offered against a server cap of 13):**
- W1: running_hist `{11: 114}`, i.e. **stuck at 11 in 100 % of samples**, queued 97 %.
- #44401: identical (95.6 %).
- **The delayer is confirmed on the GPU; #44401 does not fix it.**
- Note: the steady windows reached 12 because all 12 requests start together. The 11-of-12 loss appears only with churn (Duck).

**nodelay crash:**
- It reached 12 in 36 % of samples. Then, at **13 running**, every TP rank raised `RuntimeError: The size of tensor a (12) must match the size of tensor b (13)` in `dflash_worker_v2.py:273`. The cause is a buffer sized by `--cuda-graph-max-bs-decode 12`.
- **The delayer has been hiding a crash at 13 running with DFlash.** The gate caught it (Traceback × 4 among the log signs).
- Fix candidates:
  - nodelay **together with graphs13** (buffers sized 13);
  - or nodelay with `--max-running-requests 12`.
  - Neither is tested.

## Profiler (A1-2, 20 decode steps, TP rank 0; the same within 0.35 ms on all ranks)

| Category | ms/step | Share |
|---|---|---|
| NCCL all-reduce (140 calls) | **50.9** | 44 % |
| NCCL all-gather (3 calls) | 8.1 | 7 % |
| Marlin INT4 GEMM (512 calls) | 25.3 | 22 % |
| Other GEMM (133 calls, probably the BF16 drafter + lm_head) | 14.5 | 12.5 % |
| GDN (Triton) | 8.0 | 7 % |
| elementwise | 2.9 | 2.5 % |
| attention | 2.7 | 2.3 % |
| sampling | 2.4 | 2 % |
| norm | 1.1 | 1 % |
| **Kernel sum** | **116.1** (busy 113.0, wall 116.7) | |

- **Communication is 59 ms of 116 ms (51 %).** The bench predicted 345 µs × 140 = 48.3 + 8 = 56.3 ms, so the profiler confirms it.
- GDN is 8 ms (< 15 ms), so the GDN-kernel lever is small.

## Duck run (rev 6g, first GPU run; G4B 12 tasks; 25-min window, fresh launch of graphs13)

| Metric | R2 | G4B rev 6d (10-21) |
|---|---|---|
| Eval queries solved | **6 / 19** | 2 / 19 |
| Task-normalised | **33.33 %** | — |
| Branches started / done / accepted | 17 / 10 / 3 | — |
| Cached prompt share | **82.0 %** (1,266,937 / 1,545,883) | 75.0 % |
| Raw-reasoning switch (E14-P) | **115 kept, 0 mismatch** | off |
| Isolation preflight | PASS (12 / 12 own marker), 12 workers | |
| KV / window | 866,744 / 72,228 per stream | |

Agent replay (A2-1, default arm): 120 follow-ups → reply_reused 67, lost_at_prefill_end 26, partial_reply 27, deeper_miss 0.

## Bugs found

1. **JIT cache tar:** `IndexError('list index out of range')`; no cache was saved. Must be fixed before the compile-cache lever.
2. **nodelay crash at 13 running** (above). This is an SGLang behaviour, not ours. Keep the delayer, or pair nodelay with graphs13 and test that.
3. **int8 Mamba checkpoint does not fit** at mem 0.83 (OOM).

## For the submission (rerun mode)

- Rerun runs no measurements, so the arm must be **set in the profile**: W1 + **width 6** + graphs13 + raw reasoning (+ the E14 load path).
- Width 6 is chosen over 4 because it has the highest tok/s and the higher accept. It is the lower-risk change, but long-context behaviour is unmeasured.
