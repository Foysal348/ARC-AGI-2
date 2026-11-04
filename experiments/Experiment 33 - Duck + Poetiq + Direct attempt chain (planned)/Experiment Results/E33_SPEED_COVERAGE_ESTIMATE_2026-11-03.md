# E33 speed and coverage estimate (2026-11-03, before any GPU run)

## Method
- **Simulation:** `e33_coverage_sim_2026-11-03.py`; full output in `e33_coverage_sim_2026-11-03_output.txt`.
- **Input:** the real E28 K100 6-h EVAL120 per-attempt data, `e28_k100_report.json`:
  - A tokens, kinds and order per task;
  - B2 tokens and outcomes;
  - C tokens for the train-pass tasks.
- **Model:** 24 slots at a constant per-stream speed.
- **Priorities and gates, as in the E33 design:**
  - A, then B, then C, then D, then E inside an attempt;
  - attempt 1 before attempt 2;
  - attempt-2 group 1 (attempt-1 A not a train pass) before group 2 (the rest, ascending A tokens);
  - Poetiq starts only with ≥ 68,192 tokens of time left;
  - Direct starts only with ≥ 112,192 tokens of time left.
- **Calibration:** the same sim in E28 mode gives:
  - A queue empty at 225 min (real 233);
  - 62 C starts (real 53);
  - all 6 attempt_2-hit tasks finished (real 6).

  It is slightly optimistic.
- **Assumptions not yet measured:**
  - Flash-Next Poetiq tokens per iteration: 64k assumed. It has never run on GPU.
  - Direct: 104k.
  - Hidden rerun: 240 tasks = the eval set twice, in a ~670-min window.

## Speed
- **E33 has no engine change.** It runs the E29 DFlash2 D3s engine.
- **Expected speed:**
  - E28 K100 (MTP) measured 419.5 tok/s with 23.4 running streams.
  - NB-S1 measured DFlash2 at +4-16 % over MTP.
  - So E33 should give ≈ 436-487 tok/s, i.e. 18.6-20.8 tok/s per stream.
- **E29 smoke:** 262 tok/s at 7.4 mean streams (12 tasks only). Not a throughput number.

## Coverage (sim)

| Run | Speed vs E28 | A1 queue empty | 1C / 1D / 1E done | Attempt-2 A′ done (hard / pass) | E28 attempt_2-hit tasks reached (of 6) |
|---|---|---|---|---|---|
| EVAL120 6 h | ×1.0 | 225 min | 0 / 0 / 0 (7 C skipped for time) | 5 / 9 | 0 |
| EVAL120 6 h | ×1.1 | 205 min | 0 / 0 / 0 | 9 / 27 | 4 |
| EVAL120 6 h | ×1.5 | 150 min | 8 / 8 / 0 | 16 / 76 | 6 |
| EVAL120 6 h | ×2.0 | 113 min | 8 / 8 / 7 | 16 / 104 | 6 |
| Hidden 12 h (240) | ×1.0 | 479 min | 8 / 0 / 0 | 0 / 0 | 0 |
| Hidden 12 h (240) | ×1.1 | 435 min | 15 / 8 / 0 | 5 / 0 | 0 |
| Hidden 12 h (240) | ×1.5 | 319 min | 16 / 16 / 15 | 32 / 100 | 6 |
| Hidden 12 h (240) | ×2.0 | 239 min | 16 / 16 / 15 | 32 / 208 | 6 |

- **E28 scheduler, same sim, hidden 12 h:** 51 (×1.0) and 100 (×1.1) attempt-2 Duck runs on train-pass tasks.
- **E28's real result:** 107 / 172 queries (pass1 100, pass2 7), task-norm 65.0 %.
  - All 7 attempt_2 hits came from those runs on train-pass tasks.
  - They started at 262-313 min.

## Reading
- **At E29 speed (×1.0-1.1), the chain rarely gets past B.**
  - Attempt-1 A takes 205-225 min (EVAL120) or 435-479 min (hidden).
  - B2 then starts only when the A queue is empty (E28 rule: A before B), so it ends at ~308-360 min.
  - Poetiq / Direct start on 0 tasks (6 h) or 8-15 C / 0-8 D / 0 E (12 h).
- **Attempt 2 starts late.**
  - Group 1 = A′ on the hard tasks at ~104k, and its runs are mostly cut by the window.
  - So group 2 (where E28's attempt_2 hits came from) is reached for 0-4 of the 6 tasks.
  - Expected EVAL120 at ×1.0-1.1: ≈ 101-106 / 172 queries (task-norm ≈ 60-64 %), i.e. about E28 level.
- **From ×1.5 speed (≈ 630 tok/s)** the whole chain runs:
  - C and D on every hard task, E on most;
  - attempt 2 on most tasks;
  - all 6 attempt_2-hit tasks reached.
- **The gain from Poetiq / Direct** then depends on their unmeasured Flash-Next hit rate on Duck-fail tasks. The guess is +1-3 queries per 120 tasks.
- **Speed is the lever that opens E33's coverage** (E32 lever bundle / 1500+ track).

## Addendum (2026-11-03): the built notebook `Q27-XHIGH-E33_CHAIN.ipynb` (sha 0429b589, finalize_e33.py, sched patch 4aea0104)
- **What the code does.** I read the built notebook's code.
  - The scheduler priority is as in the design: per attempt, waiting B2, then A, B, C, D, E, then group, then task. Attempt 1 comes first.
  - The gates use a fixed 18 tok/s (the lowest of 18 and the measured rate):
    - Poetiq: 60 s + 60,000 / 18 + drain 9 min = 65.6 min before the solve end;
    - Direct: 60 s + 60,000 / 18 + drain 26 min = 82.6 min before the solve end.
  - So a faster engine does not loosen the gates.
- **The answer order differs from the design.** The notebook ships `E33_A1_ORDER = "e28"` (train pass, best train, Direct). The user had accepted `direct_first`.
- **Re-run of the sim** with these exact gates: `e33_coverage_sim_nb_gates_2026-11-03.py` / `_output.txt`. Expected E29 speed is ×1.04-1.16 of E28.

| Run | Speed | A1 empty | C / D / E done | A′ done (hard / pass) | E28 a2-hit tasks reached (of 6) | E28 scheduler: A′-on-pass done |
|---|---|---|---|---|---|---|
| EVAL120 6 h | ×1.0 | 225 | 0 / 0 / 0 | 5 / 9 | 0 | 39 (6) |
| EVAL120 6 h | ×1.04 | 216 | 0 / 0 / 0 | 7 / 16 | 2 | 45 (6) |
| EVAL120 6 h | ×1.1 | 205 | 0 / 0 / 0 | 9 / 27 | 4 | 59 (6) |
| EVAL120 6 h | ×1.16 | 194 | 2 / 0 / 0 | 11 / 38 | 6 | 65 (6) |
| Hidden 12 h, 670 min | ×1.0 | 479 | 6 / 0 / 0 | 0 / 0 | 0 | 51 (4) |
| Hidden 12 h, 670 min | ×1.04 | 460 | 10 / 0 / 0 | 0 / 0 | 0 | 74 (6) |
| Hidden 12 h, 670 min | ×1.1 | 435 | 14 / 7 / 0 | 5 / 0 | 0 | 100 (6) |
| Hidden 12 h, 670 min | ×1.16 | 413 | 16 / 13 / 0 | 8 / 0 | 0 | 122 (6) |
| Hidden 12 h, 690 min | ×1.16 | 413 | 16 / 14 / 0 | 10 / 0 | 0 | 126 (6) |

- **Expected EVAL120 6 h:** ≈ 100-108 / 172 queries (task-norm ≈ 58-66 %; E28 65.0 %). Poetiq / Direct contribute ~0.
- **Hidden 12 h:**
  - Poetiq C runs on 6-16 of the ~32 hard tasks and D on 0-14. Direct never runs.
  - Attempt-2 runs on train-pass tasks (E28's attempt_2 source) never start.
  - Expected vs the E28 scheduler: the Poetiq hits on hard tasks (unmeasured on Flash-Next; guess +1-3 / 240) against the lost attempt_2 hits (E28 rate ≈ 6 tasks / 120). Net ≈ -2 to -4 task-norm points.
