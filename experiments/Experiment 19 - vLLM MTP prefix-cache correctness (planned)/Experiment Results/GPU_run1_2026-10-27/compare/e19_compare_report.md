# E19 cross-account verdicts

| arm | what | final | in-arm | churn (mode) | rows ok / planned | serial A/A ident/cmp (flips) | cache serial | churn vs serial | cont8k | cont30k | corrupt rows | MTP accept min / median | cache warm hits | launch s | KV tokens | block |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | reference: MTP OFF, prefix caching OFF | **PASS** | PASS | PASS (uncalibrated) | 60/60 | 6/6 (0) | - | 0/24 (0) | 5/6 (0) | 6/6 (0) | 0 | None / None | 0/18 | 285.0 | 376320 | 1568 |
| B | MTP 3, prefix caching OFF | **FAIL** | FAIL | PASS (calibrated) | 60/60 | 6/6 (0) | - | 0/24 (0) | 4/6 (2) | 4/6 (2) | 0 | 0.582 / 0.786 | 0/18 | 475.0 | 337600 | 1600 |
| C | MTP OFF, prefix caching ON | **PASS** | PASS | PASS (calibrated) | 66/66 | 6/6 (0) | 6/6 (0) | 0/24 (0) | 6/6 (0) | 6/6 (0) | 0 | None / None | 18/18 | 290.0 | 376320 | 1568 |
| D | S1 exactly: MTP 3 + prefix caching ON, fp8 KV | **PASS** | PASS | PASS (calibrated) | 66/66 | 6/6 (0) | 6/6 (0) | 0/24 (0) | 6/6 (0) | 6/6 (0) | 0 | 0.575 / 0.795 | 18/18 | 495.1 | 337600 | 1600 |
| E | S1 + --no-async-scheduling | **PASS** | PASS | PASS (calibrated) | 66/66 | 6/6 (0) | 6/6 (0) | 0/24 (0) | 6/6 (0) | 6/6 (0) | 0 | 0.57 / 0.763 | 18/18 | 500.1 | 337600 | 1600 |
| K | S1 with bf16 KV (--kv-cache-dtype auto) | **PASS** | PASS | PASS (calibrated) | 66/66 | 6/6 (0) | 6/6 (0) | 0/24 (0) | 6/6 (0) | 6/6 (0) | 0 | 0.593 / 0.818 | 18/18 | 325.0 | 176800 | 800 |

## Reasons

- **A** (PASS): none
- **B** (FAIL): cont_long8k: 2 large-margin flips > the arm's serial A/A 0; cont_long30k: 2 large-margin flips > the arm's serial A/A 0
- **C** (PASS): none
- **D** (PASS): trailing Mamba block over generated tokens NOT_EXERCISED (no continuation hit reached into the server's generated tokens): no safety claim for that path
- **E** (PASS): trailing Mamba block over generated tokens NOT_EXERCISED (no continuation hit reached into the server's generated tokens): no safety claim for that path
- **K** (PASS): trailing Mamba block over generated tokens NOT_EXERCISED (no continuation hit reached into the server's generated tokens): no safety claim for that path

## Secondary: token comparison against arm A (informational; MTP on vs off may differ at low margins)

- B: churn/churn 0/24 (0); long30k/cont 6/6 (0); long30k/nat 4/6 (2); long8k/cont 4/6 (2); long8k/nat 6/6 (0); serial/cold 0/6 (0); serial/warm 0/6 (0)
- C: churn/churn 0/24 (0); long30k/cont 6/6 (0); long30k/nat 6/6 (0); long8k/cont 5/6 (0); long8k/nat 6/6 (0); serial/cold 0/6 (0); serial/warm 0/6 (0)
- D: churn/churn 0/24 (0); long30k/cont 6/6 (0); long30k/nat 6/6 (0); long8k/cont 5/6 (1); long8k/nat 6/6 (0); serial/cold 0/6 (0); serial/warm 0/6 (0)
- E: churn/churn 0/24 (0); long30k/cont 6/6 (0); long30k/nat 6/6 (0); long8k/cont 5/6 (1); long8k/nat 6/6 (0); serial/cold 0/6 (0); serial/warm 0/6 (0)
- K: churn/churn 1/24 (0); long30k/cont 6/6 (0); long30k/nat 6/6 (0); long8k/cont 5/6 (1); long8k/nat 6/6 (0); serial/cold 0/6 (0); serial/warm 0/6 (0)

Cells: identical / compared (large-margin flips). A large-margin flip = the first token divergence where a known top-1 minus top-2 logprob margin is > 1 nat. In-arm FAIL: corrupt rows with clean partners, cache-hit or continuation flips above the arm's serial A/A, MTP acceptance collapse. Churn FAIL: flips vs serial above the calibrators' (A / B) churn flips, or above the arm's own serial A/A when no calibrator ran ('uncalibrated'). Rows not run in the budget make an arm NOT_TESTED_BUDGET, never PASS.

Inputs: A <- D:/ARC AGI 2/_scratch/e19_gpu/p2; B <- D:/ARC AGI 2/_scratch/e19_gpu/p3; C <- D:/ARC AGI 2/_scratch/e19_gpu/p1; D <- D:/ARC AGI 2/_scratch/e19_gpu/p1; E <- D:/ARC AGI 2/_scratch/e19_gpu/p2; K <- D:/ARC AGI 2/_scratch/e19_gpu/p3
