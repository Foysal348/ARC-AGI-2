# Poetiq R5 EVAL120 with ORDER2 instead of alphabetical order: counterfactual replay (Claude, 2026-10-15)

**Question (user):** what would Experiment 10's Poetiq R5 run have scored with the ORDER2 ordering?

**Method** (session scratchpad `order2/r5_order2_replay.py`): a discrete-event replay of R5's scheduler.
- 16 slots and a FIFO queue; a failed iteration goes to the back.
- Admission only before 694 min; the scheduler starts at 14.4 min.
- **Every request keeps its measured duration and outcome** (`poetiq_iterations.jsonl`).
- An iteration that never ran in the real run gets the task's own iteration-1 duration and counts as "not passed" for scheduling.
- For the score, an iteration that never ran counts 0 (lower bound) or the measured iteration-2 rate (expected): 0.173 task points per iteration-2 request.

**Validation:** replaying the real (alphabetical) order reproduces the real run within about 1 request and 0.8 points.

| | Replay | Real run |
|---|---|---|
| Second iterations | 51 | 52 |
| Failed tasks never retried | 14 | 13 |
| Task-norm | 47.78 % | 48.61 % |

## 1. With the full EVAL120 clock (what really happened, re-ordered)

| | Alphabetical (replay) | **ORDER2** |
|---|---|---|
| Requests per iteration | 120 / 51 | 120 / **58** |
| Round 1 finished | 542 min | 552 min |
| Failed iteration-1 tasks never retried | 14 | **7** |
| Final task-norm | 47.78 % | **48.19 % (lower) / 49.64 % (expected)** |
| Reached by 2 / 4 / 6 / 8 / 10 h | 6.2 / 14.6 / 24.6 / 34.4 / 40.3 % | **8.3 / 19.2 / 28.8** / 34.0 / 42.8 % |

- **The final effect is small: +0.4 to +1.9 points.** All 120 tasks got iteration 1 in both orders. ORDER2 only changes which failed tasks get iteration 2.
- **11 tasks gain an iteration 2:** `d59b0160`, `d8e07eb2`, `da515329`, `de809cff`, `e12f9a14`, `e376de54`, `e87109e9`, `edb79dae`, `eee78d87`, `f931b4a8`, `faa9f03d`.
  - All are order_score 1 or 2, and late in the alphabet.
  - Their iteration-2 outcome is unknown, so it counts at the measured rate.
  - `d59b0160` and `e376de54` are DIRECT-only solves; `edb79dae` was solved only by ARC Prize.
- **4 tasks lose iteration 2:** `4c7dc4dd`, `4e34c42c`, `5dbc8537`, `6e4f6532`, all order_score 0.
  - `6e4f6532` was really solved by its iteration 2, so −0.5 points.

## 2. When the clock ends earlier (time pressure; not a hidden-run projection)

Task-norm by admission end:

| Admission ends at | Alphabetical | ORDER2 | Δ | Perfect hindsight order (upper bound) |
|---|---|---|---|---|
| 240 min | 19.58 | **22.92** | +3.3 | 36.94 |
| 300 min | 25.42 | **28.75** | +3.3 | 36.94 |
| 360 min | 29.17 | **32.08** | +2.9 | 36.94 |
| 420 min | 35.28 | 34.03 | −1.3 | 36.94 |
| 480 min | 36.94 | 36.94 | 0 | 39.44 |
| 540 min | 40.28 | **41.94** | +1.7 | 48.61 |
| 600 min | 41.94 | **44.44** | +2.5 | 48.61 |
| 694 min | 47.78 | **48.19** | +0.4 | 48.61 |

- **ORDER2 is worth about +3 points while the clock is tight, and less as time grows.** It is not monotone; at 420 min it was −1.3.
- The perfect-hindsight order is 8–14 points higher at 240–360 min. **Most of the value of ordering is still unclaimed**, because ORDER2's predictor is weak (AUC ≈ 0.55–0.60).

## 3. Why ORDER2 helps only a little for Poetiq

| order_score | Tasks | Iteration-1 train pass | Median iteration-1 minutes | Final task score |
|---|---|---|---|---|
| 2 | 52 | 29 (56 %) | 66.0 | 59.6 % |
| 1 | 53 | 21 (40 %) | 72.9 | 42.1 % |
| 0 | 15 | 5 (33 %) | 69.4 | 33.3 % |

- The ranking is right in direction, but the gaps are moderate.
- Score-2 tasks are only about 10 % faster.
- Of the first 60 tasks to finish round 1: ORDER2 34 Poetiq-solved vs alphabetical 32.

## Conclusions

1. **In the real Experiment 10 run, ORDER2 would have moved the score from 48.6 % to about 48.2–49.6 %. That is almost the same.** Every task got its first iteration anyway.
2. **ORDER2 matters when time runs short:** about +3 points at 4–6 h. Even then it is far from what a better predictor could give (up to +14 at 4 h).
3. **The lever with headroom is a better predictor of which tasks Poetiq solves**, from observable inputs only. ORDER2's two rules capture a small part of it.
