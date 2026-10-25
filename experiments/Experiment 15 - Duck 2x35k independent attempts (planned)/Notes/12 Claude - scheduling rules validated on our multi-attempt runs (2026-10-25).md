# 12 — The E15B scheduling rules, validated on our multi-attempt Duck runs

**Date:** 2026-10-25. **User request:** "validate these rules, or find different ones".

**Code and full output:** `Experiment Results/Analysis/scheduling rules (2026-10-25)/` (`load_attempts.py`, `analyze.py`, `analysis_out.md`).

## Data

648 Duck attempts (`branch_result.json`), scored against the evaluation solutions.

| Run | Attempts | Note |
|---|---|---|
| G4B XHIGH E16 EVAL120 | 120 b0 + 82 b1 | b1 = indexed view, stage 2 |
| G4B XHIGH noE16 EVAL120 | 120 b0 + 79 b1 | an independent repeat of the run above |
| G4B LOW E16 EVAL120 | 120 b0 + 83 b1 | effort low |
| E15B 45K30K, E15B 40K35K | 20 b0 + 20 b1 each | b1 = fresh, facts view |
| E12 EVAL12R5 LOW | 12 + 12 | loaded, not used |

**Classes:** the scheduler's `stage2_class` on attempt 1:
- 0 = partial progress, no accept;
- 1 = nothing;
- 2 = one accepted answer;
- 3 = two or more distinct accepted answers.

**Credit:** task-normalised, from the accepted candidates. **Gain:** the credit that a fresh attempt adds to attempt 1.

## A. Stage 2: which attempt 2 first

### A4. Pooled pairs

Two sources:
- cross-run: the b0 of one XHIGH run vs the b0 of the other, both directions, 240 pairs;
- E15B: b0 → b1, 40 pairs.

| Class of attempt 1 | Pairs | Attempt-1 credit | Fresh credit | **Gain per attempt** | Fresh tokens | Gain / 100k tokens |
|---|---|---|---|---|---|---|
| 0 partial progress | 19 | 0 | 0.079 | **0.079** | 41,480 | 0.190 |
| 1 nothing | 145 | 0 | 0.141 | **0.141** | 48,101 | 0.294 |
| 2 one accepted | 106 | 0.733 | 0.533 | **0.055** | 34,058 | 0.162 |
| 3 two+ accepted | 10 | 0.700 | 0.450 | **0.100** | 34,247 | 0.292 |
| 2, flagged | 16 | 0.688 | 0.500 | 0.000 | 37,054 | |
| 2, not flagged | 90 | 0.741 | 0.539 | 0.065 | 33,526 | |

### Checks against the current rules

| Current rule | Verdict |
|---|---|
| Attempts with no accept before attempts with an accepted answer | **Supported:** 0.141 / 0.079 vs 0.055. |
| Class 0 before class 1 | **Not supported:** 0.079 vs 0.141. n = 19, so weak evidence, but it points the wrong way. Partial progress at the end of an attempt is mostly a wrong program. |
| Flagged first inside class 2 | **Not supported:** gain 0.000 (n = 16) vs 0.065. The overfit flag does not find the wrong accepts: P(wrong \| flagged) 31 % vs 26 %. |
| Class 3 last | Not supported (0.100, n = 10); too few to say. |
| Fresh attempt 2 on the facts view | **Supported.** The old indexed-view b1 (G4B, 161 attempts) gave 0 credit in every class. The facts view adds 0.14-0.20 on class 1. |

### A6. Simulated stage-2 queue

Credit (tasks, out of 280 pairs) when only part of the attempt-2 tokens fits:

| Order | 25 % | 50 % | 75 % | 100 % |
|---|---|---|---|---|
| current: 0, 1, 2 flagged first, 3 | 7.9 | 16.3 | 22.9 | 28.8 |
| 1, 0, 3, 2 | 8.6 | 17.1 | 24.1 | 28.8 |
| no-accept first (0 = 1), then 3, then 2 | 8.2 | 16.6 | 24.1 | 28.8 |

The better order is worth about +1 task, and only when time binds. At 45K30K (8.8 h of 11.6 h) every attempt 2 runs, so the order changes nothing.

### A5. A third attempt

The k-th independent xhigh attempt, on tasks where every earlier attempt had no correct accepted answer. Pool: 2 G4B XHIGH b0 + 4 E15B attempts.

| k | P(adds credit) | Mean tokens | Gain / 100k tokens | Tasks with ≥ k attempts |
|---|---|---|---|---|
| 1 | 0.270 | 43,615 | 0.618 | 120 |
| 2 | 0.134 | 46,946 | 0.286 | 120 |
| 3 | 0.156 | 39,105 | 0.400 | 20 |
| 4 | 0.119 | 39,071 | 0.305 | 20 |
| 5 | 0.103 | 39,214 | 0.263 | 20 |

- **A 3rd attempt on a still-unsolved task (0.12-0.16) is worth 2-3 × an attempt 2 on a task that already has an accepted answer (0.055).**
- k ≥ 3 is measured only on the 20 E15B tasks, which are hard-biased.
- **Hidden-run estimate (45K30K):**
  - spare ≈ 11.6 − 8.8 = 2.8 h ≈ 113 attempt-3 runs (39k tokens × 36.6 stream-s per 1k tokens, 16 streams);
  - × 0.10-0.156 ≈ **11-18 tasks = +4.7 to +7.3 task-norm points**.
  - Today this time is idle: the solve stops when the queue is empty.

## B. Stage 1: ORDER2 + cells

**AUC for "attempt 1 got credit":**

| Key | XHIGH E16 | XHIGH noE16 | LOW | ARC Prize xhigh | ARC Prize low | E10 direct |
|---|---|---|---|---|---|---|
| ORDER2 + cells (current) | 0.573 | 0.566 | 0.547 | 0.627 | 0.654 | 0.572 |
| ORDER2 only | 0.575 | 0.583 | 0.564 | 0.603 | 0.627 | 0.551 |
| fewer total cells | 0.498 | 0.473 | 0.472 | 0.572 | 0.568 | 0.567 |
| fewer colours | **0.645** | 0.594 | **0.676** | **0.425** | 0.560 | 0.494 |

- **ORDER2 is positive on all six label sets** (0.55-0.65): supported.
- **The cells tie-break is mixed.** It hurts on Duck (B2, 30 % of tokens: 13 / 8 / 7 tasks with cells vs 16 / 12 / 10 without) and helps on ARC Prize / E10 direct.
- **"Fewer colours" is the best key on our 3 Duck runs, but it fails on ARC Prize xhigh (0.425).**
  - The 3 Duck runs use the same 120 tasks, so a leave-one-run-out test (0.60-0.67) is not generalisation to new tasks.
  - Not adopted.
- **Stage-1 order matters only if attempt 1 cannot finish on every task.** It finishes in ~5-5.5 h, so it does not matter for the score; the order is insurance (an engine death). **No change.**

## Recommendation (needs the user's go; not built)

1. **Stage 3, the biggest lever:**
   - after the stage-2 queue empties, a fresh attempt 3 on every task with no accepted answer (the facts view, the attempt-2 budget), in the stage-1 order, until the solve deadline;
   - when time binds, an attempt 3 ranks above an attempt 2 of class 2;
   - pass@2 slots: an accepted attempt-3 answer goes first; partial answers fill the rest.
2. **Stage-2 order:** 1, 0, 3, 2, without "flagged first". About +1 task only when time binds; zero risk.
3. **Stage 1:** unchanged (ORDER2 + cells).

## C. Replay: skip attempt 2 where attempt 1 has an accepted answer (user request)

**Policies:**
- **P0 = now:** attempt 2 on every task.
- **P1 = skip attempt 2 on tasks with an accepted answer;** the saved tokens buy attempt 3 on tasks still unsolved. Same total tokens as P0.
- **P2 = P0 + the same extra tokens for attempt 3,** i.e. what the spare hidden time allows without skipping anything.

**Attempt 3** = an unused, real, independent xhigh attempt on the same task (G4B XHIGH b0 or E15B b0/b1), random over 500 shuffles.

**Script:** `Experiment Results/Analysis/scheduling rules (2026-10-25)/replay_skip_policy.py`.

| Replayed run | Tasks | P0 credit | Attempt 2s skipped | Tokens saved | Lost by skipping | Attempt 3s bought | **P1** (vs P0) | **P2** (vs P0) |
|---|---|---|---|---|---|---|---|---|
| E15B 45K30K | 20 | 13.00 | 14 | 316,328 | 0.33 (1ae2feb7) | 7.9 | 13.34 (+0.34) | 13.71 (+0.71) |
| E15B 40K35K | 20 | 7.50 | 10 | 226,884 | 0 | 5.9 | 9.60 (+2.10) | 9.53 (+2.03) |
| EVAL120: a1 = XHIGH E16, a2 = XHIGH noE16 | 120 | 46.50 | 46 | 1,705,261 | 1.50 (6e453dd6, 7ed72f31) | 22.6* | 46.00 (−0.50)* | 47.50 (+1.00)* |
| EVAL120: a1 = XHIGH noE16, a2 = XHIGH E16 | 120 | 46.50 | 46 | 1,704,173 | 5.00 (221dfab4, 31f7f899, 58490d8a, 5961cc34, 80a900e0) | 26.5* | 42.50 (−4.00)* | 47.50 (+1.00)* |

\* Real attempt-3 data exists only for the 20 E15B tasks, so EVAL120 could spend only part of the saved tokens.

**Rate-based estimate for EVAL120:** 1.70M tokens / 39k ≈ 43.6 attempt 3s, at the A5 rate 0.10 / 0.134 / 0.156.

| Direction | P1 (vs P0) | P2 (vs P0) |
|---|---|---|
| a1 = XHIGH E16 | +2.9 / +4.3 / +5.3 | +4.4 / +5.8 / +6.8 |
| a1 = XHIGH noE16 | −0.6 / +0.8 / +1.8 | +4.4 / +5.8 / +6.8 |

**Reading:**
- **Skipping costs real tasks.** An accepted answer is wrong ~26-31 % of the time, and attempt 2 is the fix: 0.33-5 tasks lost per run.
- At equal compute, P1 ranges from −4 to +5: the sign depends on the run. It is a weak, noisy gain.
- **P2 (keep every attempt 2, add attempt 3 in the spare time) is the best or tied in all 4 replays, and never loses credit.**
  - The hidden run at 45K30K has ~2.8 h spare, so P2 is possible there.
  - P1 only makes sense when time binds. The proposed priority (attempt 3 for unsolved tasks above attempt 2 of class 2) does exactly that, only when the time runs out.
- **Recommendation:** do not skip attempt 2. Add stage 3 (attempt 3 on unsolved tasks) in the spare time.

## D. Time-sliced replay at hidden scale (user request: the rule on attempt 2, solve count by time)

**Setup:**
- 240 synthetic tasks = the 120 EVAL120 tasks × 2 directions. Attempt 1 = the b0 of one XHIGH G4B run; attempt 2 = the b0 of the other (a real, fresh xhigh attempt).
- 16 streams; 36.6 stream-s per 1k tokens (measured, E15B 45K30K); solve window 11.6 h.
- A correct answer counts when its first accepted candidate appeared.
- The scheduler is as in E15B: stage 1 in ORDER2 + cells order; stage 2 after the last attempt 1 starts, in class order.
- **Attempt 3 is an estimate:** a Bernoulli p, with the timing taken from real fresh solves.
  - p is adjusted for the 30k cap: only 43.5 % of the fresh solves came within 30k, so A5's 0.10-0.156 becomes 0.043-0.068 (0.10 is shown as the high case).
- **Files:** `time_slice_sim.py`, `capped_peff.py`, `time_slice_out.md`.

**Policies:**
- **P0** = now: attempt 2 on every task.
- **R** = attempt 2 only where attempt 1 has no accepted answer.
- **+3** = then attempt 3 on tasks with no accepted answer, until the window ends.

### RAW: the attempts' real token counts

All attempt 1s need 6.8 h and all attempt 2s 6.8 h, i.e. 13.6 h > 11.6 h.

| Policy | 3 h | 6 h | 9 h | 10 h | 11 h | 11.6 h |
|---|---|---|---|---|---|---|
| P0 | 37.5 | 59.5 | 77.5 | 81.5 | 85.5 | 86.5 |
| R | 37.5 | 59.5 | 77.5 | 81.5 | 85.5 | 86.5 |
| R + 3 (p 0.156) | 37.5 | 59.5 | 77.5 | 81.5 | 85.5 | 87.3 |

**P0 = R:** when time binds, the class order already runs the attempt 2s of accepted tasks last, and the window ends before it reaches them.

### CAPPED to the 45K30K budgets

Attempt 1s need 5.9 h and attempt 2s 4.3 h: 10.2 h of 11.6 h.

| Policy | 3 h | 6 h | 9 h | 10 h | 11 h | 11.6 h |
|---|---|---|---|---|---|---|
| P0 | 33.5 | 58.5 | 62.5 | 63.5 | 64.5 | **64.5** |
| R | 33.5 | 58.5 | 62.5 | 62.5 | 62.5 | **62.5** (−2: wrong accepts that attempt 2 fixed) |
| P0 + 3, p 0.043 / 0.068 / 0.10 | | | 62.5 | 63.5 | 66.3 / 67.1 / 68.6 | **67.7 / 69.5 / 71.7** |
| R + 3, p 0.043 / 0.068 / 0.10 | | | 62.6 | 64.8 / 66.1 / 67.8 | 67.1 / 69.6 / 73.1 | **68.5 / 71.7 / 76.5** |

**Reading:**
- Up to ~9 h every policy is identical (stage 1 + the first attempt 2s).
- **R alone loses 2 tasks.** Skipping without using the freed time is worse.
- **R + 3 > P0 + 3** (+0.8 / +2.2 / +4.8) when the time left after all attempt 2s is short (here 1.4 h).
- Gain over today's 64.5:
  - P0 + 3: +3.2 to +7.2 tasks;
  - R + 3: +4.0 to +12.0 tasks of 240, i.e. +1.7 to +5.0 task-norm points.
- **Correction of section A's estimate:** an attempt 3 at the 30k cap is worth less (p 0.043-0.068) than A5's uncapped rate. The earlier "+4.7 to +7.3 points" was optimistic for 45K30K.
- **Best order** (it combines both): attempt 2 on no-accept tasks → attempt 3 on still-unsolved tasks → attempt 2 on accepted tasks, if time remains. It is ≥ R + 3, and it equals P0 + 3 when there is enough time.

## E. Thinking budget grid at hidden scale (user request: the gain from more thinking)

**Setup:** the same replay as D (240 synthetic tasks from the two XHIGH EVAL120 runs, 16 streams, 11.6 h). Each attempt is capped at the given budget. An attempt counts as correct only if its first correct accept came within the cap.

**Files:** `budget_grid.py`, `budget_grid_out.md`.

**Share of the 69 correct solves within a budget:**

| Budget | 30k | 35k | 40k | 45k | 50k | 55k | 60k |
|---|---|---|---|---|---|---|---|
| Share | 0.435 | 0.58 | 0.725 | 0.87 | 0.913 | 0.986 | 1.0 |

No solve came after 55,126 tokens, though the attempts ran to 62k (p90) and 77k (max).

### P0 (attempt 2 on every task)

Task credit at 11.6 h (out of 240), and the hours both attempts would need:

| Attempt 1 \ attempt 2 | 30k | 35k | 45k | 50k | 60k |
|---|---|---|---|---|---|
| 35k | 47.0 (9.2 h) | 53.0 (9.8 h) | | | |
| 40k | 55.5 (9.8 h) | 61.5 (10.4 h) | | | |
| 45k | 64.5 (10.2 h) | 70.5 (10.8 h) | 79.5 (11.8 h) | | |
| 50k | 67.0 (10.6 h) | 73.0 (11.2 h) | 81.0 (12.2 h) | 83.5 (12.5 h) | |
| 55k | 71.0 (10.8 h) | 77.0 (11.4 h) | 84.0 (12.4 h) | 86.5 (12.8 h) | |
| 60k | 72.0 (11.0 h) | 78.0 (11.6 h) | 85.0 (12.6 h) | 86.5 (12.9 h) | 87.5 (13.3 h) |
| 70k | 72.0 | 77.0 | 85.0 | 86.5 | 86.5 |

**R + stage 3** (estimate) is about +2 to +6 at small budgets and about +0.5 to +1 at large ones; the top is 88.2 at 60k / 60k.

**Reading:**
- **The biggest lever is attempt 2's budget:**
  - 30k → 45k at attempt 1 = 45k: +15 tasks (64.5 → 79.5), i.e. +6.3 task-norm points;
  - 50k / 30k → 50k / 45k: +14.
- **Attempt 1:** 45k → 50k = +2.5; 50k → 60k = +4 (at a 45k attempt 2).
- **The plateau is at about 55-60k per attempt;** 70k adds nothing.
- **More thinking wins even when both attempts need more than 11.6 h.** The cut part is the lowest-value tail: the attempt 2s of tasks that already have an accepted answer run last.
- **Caveats:**
  - The token speed is held constant, but long contexts are ~5-10 % slower, so the large budgets are slightly optimistic.
  - The data are G4B attempts with E14S settings (a 28k pause and resume), not B mode.
  - It is the same 120 tasks twice.
  - On the real 20-task commits, 50K45K (62.5 %) ≈ 45K30K (65 %). The difference the sim predicts on 20 tasks (~1.4) is inside the noise.


## Correction (agent 3, 2026-10-25)

- The 20 E15B tasks are **easier** than the EVAL120 average (G4B correct-accept rate 0.50 vs 0.22 on the other 100), not harder. So A5's attempt-3 rate (0.156) is biased upward.
- On the 59 tasks with no accept in two G4B attempts, attempt 3 is worth 0.057-0.094 per attempt, and 27 of the 59 were never solved by anything.
- At hidden scale the time binds right after the attempt-2 phase. Stage 3 therefore runs only when the attempt 2s of accepted tasks are deferred past it or skipped.
- Sections C-D overstate the gain. Expected R3J order vs rev 3h: −1 to +0.6 tasks out of 240 (Notes/16).
