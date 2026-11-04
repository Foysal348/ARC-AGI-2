# GPU EVAL120 Poetiq R5 vs DIRECT ORDER2: deep analysis (Claude, 2026-10-15)

**Run:** `hosen42/q27-xhigh-e10-poetiq-r5-eval120`. Its code cells are identical to `Q27-XHIGH-v1_4_4_E10_86K_POETIQ_R5_EVAL120.ipynb` (sha `498109b7…`).

**Setup:**
- 77,824 thinking + 8,192 answer per iteration.
- 1 expert, up to 10 iterations, solver temperature 1.0.
- Alphabetical order, 16 sequences.
- The hidden-run clock.

**Outputs:** `Kaggle Runs/GPU EVAL120 Poetiq R5 (hosen42 q27-xhigh-e10-poetiq-r5-eval120)/`. Script: session scratchpad `order2/r5_analysis.py`.

## Headline

| | Poetiq R5 | DIRECT ORDER2 | Union (both runs) |
|---|---|---|---|
| Test inputs hit | **78/172 (45.35 %)** | 58/172 (33.72 %) | 92/172 (53.5 %) |
| Task-normalised | **48.61 %** | 36.94 %| **56.81 %** |
| Wall time / tokens / speed | 691 min / 11.65 M / 281 tok/s | 534 min / 10.45 M / 333 tok/s | — |

- Overlap: both 44, Poetiq only 34, DIRECT only 14.
- Poetiq's attempt_2 added only 1 hit (pass1 77).

## 1. Iterations and time

| Iteration | Requests | Forced at cap | Train passed | Median minutes per request |
|---|---|---|---|---|
| 1 | 120 | 67 (56 %) | 55 | 70.9 |
| 2 | 52 | 37 | 16 | 69.9 |

- **Round 1 alone took until about 540 min.** Iteration-2 results came between 536 and 703 min.
- At the end, 47 entries were still waiting in the queue, and 13 failed tasks never got a second iteration.
- No task reached iteration 3.
- Final status: 71 tasks train-verified, 49 still queued.

## 2. Token allocation per iteration (the user's question)

The thinking of each solving iteration was counted from the saved thinking files.

| | Value |
|---|---|
| Solves through a verified program | 56 tasks |
| Thinking at the solve, p25 / median / p75 / max | 48.3k / **59.9k** / 73.5k / 77.8k |
| Solves with thinking > 49,152 | **39 of 56 (70 %)** |
| Solves with thinking > 60k / > 70k | 26 / 16 |
| Iteration-2 solves | 10 tasks, at 39.3k, 44.4k, 48.7k, 49.1k, 49.9k, 56.9k, 65.8k, 77.8k, 77.8k, 77.8k: **6 of 10 above 49,152** |
| Forced closes in iteration 1 that still passed train | 13 |

**Verdict:** **keep 77,824 for every Poetiq iteration.** A 49k cap would remove most Poetiq solves. This confirms the user's objection to a 49k Poetiq cap.

## 3. How reliable is the train check (no test answer is known)

- **71 train-verified tasks:** **49 fully right on test (69 %)**, 7 partly right, **15 wrong (21 %)**.
  - n_train ≤ 2: 13 of 20 right, 5 wrong.
  - n_train ≥ 3: 36 of 51 right, 10 wrong.
  - So low_train is only slightly worse here, unlike R4.
- **Verified but wrong:** `20270e3b`, `221dfab4`, `271d71e2`, `2b83f449`, `3dc255db`, `446ef5d2`, `78332cb0`, `800d221b`, `8698868d`, `97d7923e`, `9bbf930d`, `a25697e4`, `b10624e5`, `dbff022c`, `dfadab01`.
- **Not verified, yet the best partial program was right:** 10 inputs in 7 tasks.

## 4. Where Poetiq gains

| Group | Units | Poetiq | DIRECT |
|---|---|---|---|
| Single test | 71 | 40 (56 %) | 32 (45 %) |
| **Multi test** | 101 | **38 (38 %)** | 26 (26 %) |
| n_train ≤ 2 | 49 | 19 (39 %) | 13 (27 %) |
| n_train ≥ 3 | 123 | 59 (48 %) | 45 (37 %) |

**Multi-test tasks:**
- Poetiq solves 14 fully and 9 partly; DIRECT solves 5 fully and 15 partly.
- One program per task applies one rule to every test input. This is the "one reasoning per task" effect measured for real.

**Oracle tiers** (built before R5), task score:

| Tier | Tasks | Poetiq | DIRECT |
|---|---|---|---|
| A | 26 | 22.0 | 24.0 |
| B | 21 | 13.8 | 11.0 |
| C | 25 | 11.0 | 9.3 |
| **D (never solved before)** | 48 | **11.5** | 0 |

**13 tasks were never solved before by any method:** `142ca369`, `28a6681f`, `2c181942`, `45a5af55`, `62593bfd`, `6e4f6532`, `88bcf3b4`, `8b7bacbf`, `8b9c3697`, `a251c730`, `aa4ec2a5`, `c7f57c3e`, `fc7cae8d`. **A better method moved the "never solved" set.**

**6 tasks solved by ARC Prize but never by us before:** `0934a4d8`, `7b3084d4`, `7c66cb00`, `8e5c0c38`, `cb2d8a2c`, `d35bdbdc`.

## 5. DIRECT as the complement (where it adds)

| Poetiq group | Tasks | Poetiq units | DIRECT units | DIRECT-only units |
|---|---|---|---|---|
| Verified at iteration 1 | 55 | 56 | 33 | **1** |
| Failed iteration 1, got iteration 2 | 52 | 20 | 21 | **10** |
| Failed iteration 1, never retried (time) | 13 | 2 | 4 | 3 |
| Verified but wrong | 15 | 0 | 2 | 2 |
| Never verified (end) | 49 | 10 | 15 | **11** |

- **DIRECT adds almost nothing where Poetiq's program passed train.**
- **It adds a lot where Poetiq failed train.** On the tasks that failed iteration 1, one DIRECT trace (21 units) matched Poetiq's iteration 2 (20 units), and the overlap is small: the union is 30.

**Policies computed from the two runs (task-normalised):**

| Policy | Score |
|---|---|
| Poetiq only | 48.61 % |
| **Poetiq if train-verified, otherwise attempt_1 = Poetiq best + attempt_2 = DIRECT** | **55.14 %** |
| Poetiq if verified, otherwise DIRECT only | 52.22 % |
| Both full runs, attempt_2 = DIRECT | 56.81 % |

The policy that runs DIRECT only on unverified tasks keeps 97 % of the full-union score (55.14 of 56.81). The gate is train verification, which is known without the answer.

## 6. Hidden-run risk (not a projection)

- **The hidden test has about 240 tasks; this eval run had 120,** and Poetiq round 1 alone ran until about 540 min.
- Hidden puzzles are easier, so eval time is not projected.
- **Measurable calibration:**
  - Poetiq requests take about 1.4× DIRECT's here: median 70.9 vs 51.0 min per request.
  - Hidden has 240 Poetiq requests in round 1 vs 259 DIRECT units.
  - So the E9 86k DIRECT LB run's hidden timing bounds Poetiq round-1 time at roughly 1.3–1.4× of DIRECT's.
- **R5 has no ordering.** Unadmitted tasks keep the identity fallback.

## Decisions this supports

1. **Poetiq (program + train verification) is our strongest method:** +11.7 task-norm points over DIRECT on all 120 eval tasks, and it opens new tasks.
2. **Keep 77,824 thinking for every Poetiq iteration.**
3. **Best next design: Poetiq round 1 → for tasks not train-verified, a DIRECT trace as the second attempt** (55.1 % on these data), instead of only Poetiq iteration 2.
4. **For the hidden run:** add ORDER2 to Poetiq (unadmitted tasks are lost), and check time feasibility with the E9 LB result.
