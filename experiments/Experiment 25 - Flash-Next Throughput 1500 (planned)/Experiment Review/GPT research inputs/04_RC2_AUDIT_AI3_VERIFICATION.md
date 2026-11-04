# Verification of external audit #3 (04_RC2_AUDIT_AI3_2026-10-31.md)

Checked by the lead on 2026-10-31. Only claims not already covered by 02/03 verifications are listed.

| Claim | Evidence | Verdict |
|---|---|---|
| The 116k natural answer for 0934a4d8 is wrong (15/27 cells; rows 14-17) | Local `arc-agi_evaluation_solutions.json` (E10 data copy): `779 729 729 779 447 447 661 666 161` vs model `913 992 992 913 447 447 661 666 161` | VERIFIED (our RC2 analysis agent found the same; external AI #2's "consistent" was wrong) |
| `fp8_gate()` PASSes on short greedy output; the TF divergence found later leaves status CLEAN | `build_e13_rc2.py` fp8_gate / tf_last order (TF runs last) | VERIFIED. RC2's CLEAN = short-output diagnostic pass only |
| "step_ms is about half the full iteration latency; 48.2 ms would predict 524 tok/s" | 8k@8: tokens_per_step 12.62 = (8/2) × 3.154; 12.62 / 48.2 ms = 261.8 tok/s | **REFRAMED, not an error.** step_ms is one PP tick (one micro-batch of C/2 requests). A request advances once per 2 ticks, so the per-request cycle = 2 × step_ms = 96.4 ms. PLAN §1 already uses t_cycle = 2 ticks. Builders must report `tick_ms` and `cycle_ms` explicitly to avoid this misreading |
| F = 7 compatible with usage | Our RC2 analysis measured F = 6 (3 mamba groups × 2) and found retention of finished sessions in kv_usage | SUPERSEDED by the analysis (F = 6; usage includes retained sessions) |
| FP8 isolation matrix: BF16/FP8 × MTP off/K3 × prefix fresh/repeat, raw per-position rows | — | ACCEPTED, moved into a dedicated short notebook NB0 (prefix caching off is allowed only with MTP off; never with MTP on) |
| Production gate example (NO_GO if mean > 1.0 or rank-1 < 0.5; else PENDING_LONG_ARC_ACCURACY) with negative tests | — | ACCEPTED as a triage gate; NB1's screening gate (0.90 / 0.15) stays the stricter screening rule |
| Long-output quality A/B on a frozen public subset (BF16 MTP off, BF16 MTP3, FP8 MTP3; exact grid match) | — | ACCEPTED as the accuracy gate; placed in the FN-Duck paired evaluation (NB5), cost to be decided with the user |
| PLE log pinned=False then host-mapped pull | log | NOTED (residency not proven) |
