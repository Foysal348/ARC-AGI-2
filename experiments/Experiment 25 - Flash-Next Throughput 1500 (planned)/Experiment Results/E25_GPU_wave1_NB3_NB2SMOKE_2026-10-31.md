# E25 GPU wave 1: NB3 kernel bench + NB2 SMOKE (2026-10-31)

Outputs: `_scratch/e25_gpu_out/q27-xhigh-e25-nb3-kernels/`, `_scratch/e25_gpu_out/q27-xhigh-e25-nb2-decode-smoke/`.

## NB3 (foysalemonshanto, sha f57935a9, 15.2 min)
Steps: install DONE, sha gate DONE, bench env DONE, D5 core DONE, D5 rest PARTIAL_BUDGET (stopped before the ncu units,
186 s left vs 240 s reserve), D7 build / pairs / solo DONE, D4 UNSUPPORTED_SM89 (CuTe candidates failed to compile; cuBLAS
baselines measured). NB4 top: D5 config only, 0.007 ms/tick (= no gain).

**D5 Marlin W4 MoE, one layer, layer-0 local experts (EP2), synthetic independent routing, graph us:**
T 1 37 | 4 216 | 8 346 | 16 771 | 32 1,199 | 64 1,838 | 128 2,411; derived 204-252 GB/s (L4 peak 300).
Config sweep: best = base within 0.4-1.1 % (bitwise equal) → Marlin config tuning is not a lever.
At the RC2 point (T 16 per micro-batch) 24 MoE layers per stage ≈ 18.5 ms of a 42.7-48.2 ms tick.

**C6 humming (alignment inside the timed forward, indexed EP; W3 / W2 synthetic RTN requant), ratio vs Marlin W4 graph:**
| arm | T 8 | T 16 | T 32 | T 64 |
|---|---|---|---|---|
| W4 g128 same weights (control) | 1.06 | 1.20 | 1.07 | 1.07 |
| W3 g128 | 0.84 | 0.98 | 0.84 | 0.85 |
| W2 g128 | 0.60 | 0.70 | 0.60 | 0.61 |
| W3 g64 | 0.87 | 0.97 | 0.87 | 0.88 |
All own-weight references PASS; graph PASS everywhere. Strict gate (W3 ≤ 0.8, W2 ≤ 0.6) only W2@T32 PASS; capacity reading
NET_POSITIVE_POSSIBLE for all. Humming is 6-20 % slower than Marlin on the same W4 weights; T 16 is humming's worst point
(sm89 heuristic tile choice) → a tuning target. W2 g64 (klee100's middle layers) was not measured.
Rough projection (klee100 mix per stage = 12 W3 + 12 W2): T 16 saves ~3 ms of 18.5 ms MoE; T 32 saves ~8 ms of 28.8 ms.

**D7 world-2 all-reduce (both TP2 pairs, graph us):** P5 host-staged AR vs NCCL (LL, SHM): 1.39-1.68× faster for
5-123 KB (T 1-24 × 2560 × bf16), 1.25× at 246-328 KB, 1.06× at 0.5-0.66 MB. Numeric PASS (bit-exact vs ordered fp32 ref,
identical ranks, graph replays bit-exact, int8 within bound). Status PASS_BOUND_ONLY: concurrency unverified (host arm overlap
0.43-0.97 < 0.5 at small sizes when both pairs run). Example T 16 (82 KB): NCCL 49.0 vs P5 30.7 us.

**D4 (cuBLAS BF16 baselines, us, M 1-16):** LM head TP2 shard 124160×2560: 2,492-2,562 (the single biggest dense item);
QSA fused QKV/gate 137-138; QSA o_proj 65; HC up 29; HC down 31; shared expert gate/up 14-18, down 8-9; router 12-15;
GDN in_proj_ba 3-15. Candidate kernels: compile failures (UNSUPPORTED_SM89).

## NB2 SMOKE (foysalemonshanto, sha ad8a28a5, 21.2 min) = SMOKE_FAIL
Boot B = K2 + INT8 draft head + seqs 16 + CAPS16. Probes / sampled / short oracle PASS; config read-back PASS (K, head,
seqs, captures); admission 16 ADMITTED_ALL (0 preemptions); dev-mode reset OK; trace parsed.
Fail: 8k@8 window `FAIL_GARBAGE_UNDER_LOAD` → boot `NO_GO_GARBAGE_UNDER_LOAD`. One of 8 streams (trace row 37, temp 0.6,
top_k 20, seed 1234, ignore_eos) entered a "row? row? row? …" loop at char 3,057 of 6,111 (~1,400 tokens), after coherent
grid analysis; the other 7 streams are coherent. One loop event cannot separate a K2 / INT8-head / CAPS16 effect from the
target's own repetition at temp 0.6 (spec decoding with probabilistic verification is lossless in principle).
Speed (early reasoning, ~2.7k tokens per stream): 242.6 tok/s at 8k@8, tick 42.7 ms (cycle 85.4), a 2.592 (K2 max 3),
per position 0.860 / 0.731, real running == 8. RC2 K3 BF16 head 8k@8: 261.8 tok/s, tick 48.2 ms, a 3.154.
→ K2 −7 % at 8 streams; tick −11 %. Rule: NB2 FULL only after SMOKE_PASS.

## NB2 diagnostic SMOKEs (foysalemonshanto, 2026-10-31): both DIAG_DONE, 0 loops — the failed-run loop was NOT reproduced
| run | config | seeds | streams clean | tokens per stream | 8k@8 tok/s | accept / draft | preemptions |
|---|---|---|---|---|---|---|---|
| SMOKE_SEED2 (e9504bbb, 20.4 min) | K2 + INT8 head + seqs 16 + CAPS16 | 4321-4328 | 8 / 8 | 2,817-3,067 | 240.9 | 1.587 | 0 |
| SMOKE_CTRL (b643b304, 21.0 min) | exact RC2 FP8 (K3, BF16 head, seqs 8) | 1234-1241 (= failed run) | 8 / 8 | 3,027-3,356 | 259.1 | 2.17 | 0 |
All integrity checks true (prompt identity / order, 8 streams, seeds, trace, no duplicates; CTRL exact RC2 diff empty).
Reading (interpretation table row "both clean"): the original stream-0 loop (seed 1234, onset ≈1,400 tokens) is rare /
non-deterministic, not reproduced at the same horizon; no configuration is implicated. Speed repeats the earlier finding: K2 +
INT8 head is −7 % vs RC2 K3 BF16 head at 8k@8 (240.9 vs 259.1; failed SMOKE 242.6 vs RC2 261.8) → K2 / INT8 head is not a speed
lever at 8 streams; NB2 FULL deprioritised; the RC2 K3 config stays.

Deep-analysis additions (2026-10-31):
- Same seed is not a replay: the failed run's row 37 and CTRL seed 1234 (same prompt sha) share only the first 4 of 2,857 /
  3,356 tokens. Batch composition changes the trajectory, so CTRL tested the same configuration and seeds, not the same text.
- K2 loss is the missing third draft position only: per-position acceptance 0.865 / 0.723 (SEED2) vs 0.867 / 0.716 / 0.586
  (CTRL); the INT8 head does not lower acceptance. Tokens per step 10.35 vs 12.68 (−18 %) outweigh the tick saving 43.0 vs
  48.9 ms (−12 %).
- Peak KV usage in the server logs: SEED2 96.6 % with 16 running (admission-16 test, 0 preemptions) → 16 streams at that
  length sit at the pool edge; CTRL 87.9 % with 8 running.
- Model load 327-349 s (SEED2) and 415-433 s (CTRL) per worker; graph capture 5 s; 0 preemption lines in both logs.
- SEED2 trace (40 forwards): FULL 0 / PIECEWISE 20 / EAGER 20; kernel time nccl 59.5 % (top ncclDevKernel_Broadcast_RING_LL
  905 ms), Marlin 22.2 %, other GEMM 8.2 %, GDN 3.9 %. NCCL kernel time includes the time a kernel waits for its peer stage, so
  it is an upper bound on communication cost, not a measured cost. Which forwards ran eager is not recorded → NB4 needs a
  timeline (nsys) to split wait from transfer before a PP-broadcast lever is sized.
