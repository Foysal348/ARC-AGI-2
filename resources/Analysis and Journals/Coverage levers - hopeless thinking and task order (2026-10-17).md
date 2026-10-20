# Coverage levers without a faster engine: hopeless-thinking stop and task order (Claude, 2026-10-17)

**Question (user):** can we gain coverage by (1) recognising hopeless thinking early and stopping it, or (2) running the solvable tasks first?

**Data, all local (no GPU):**
- 446 Poetiq requests with full thinking and outcomes: R5, D66, L66.
- ORDER2 DIRECT per unit (172 units: tokens and hit).
- R5 per-task tokens and final hits.
- ARC Prize xhigh per task.

**Scripts and outputs:** `_scratch/session_1c6db7aa/lever_analysis/` (`hopeless_signals.py`, `task_ordering.py`, `ordering_variants.py`, `pause_resume.py`, `out/*.md`).

## Starting facts

**At the 77.8k cap, capped requests are expensive and rarely succeed:**

| run | capped requests' share of tokens | capped train-pass | natural-close train-pass |
|---|---|---|---|
| R5 | 64 % | 20 % | 75 % |
| D66 | 76 % | test-correct 8 % | test-correct 26 % |

**But the long thinking is where most solves come from:**
- R5 iteration 1: only 13 of 120 requests close naturally before 49k (11 train-pass); 37 close between 49k and 77.8k (29 train-pass).
- ORDER2 DIRECT: 58 hits, of which 39 have thinking ≤ 49k and 44 ≤ 60k.
- So shortening thinking would lose solves.

**The queue is already coverage-first.** S1 and R5 run a FIFO queue in ORDER2 order: every task gets iteration 1 before any iteration 2. On hidden, where about 30 % of tasks are never reached, retries hardly run, so there are no retry tokens to save.

## Lever 1: recognise hopeless thinking (NEGATIVE)

**Features measured** at cutoffs of 30k, 40k, 49,152 and 60k, on the last 6k tokens and the prefix:
- zlib compressibility;
- duplicated lines;
- grid-dump lines;
- churn words (wait / actually / hmm / recheck …);
- distinct 8-grams;
- `def transform` count;
- code fences.

**Results:**
- **No loop signal.** Failing requests are not more repetitive: zlib AUC 0.51–0.56. Successful ones even repeat lines more, because they re-check their work.
- **Best signal = churn** (self-correction words per 1k characters), AUC 0.61–0.64. The best stop rule (losing at most 5 % of later solves) saves only **2.7–4.0 % of tokens**.
- **Out of sample** (fitted on R5, applied to D66 + L66 at 49k): it loses **12 of 65 train-passes and 7 of 32 test-correct** for 1.8M tokens saved.
- **Code written by the cutoff is a positive sign** (fences, `def transform`), but too weak to stop on.

**Decision:** no text-heuristic early stop. It loses solves faster than it saves tokens. An LLM self-assessment probe would be a separate idea and is not tested.

## Lever 2: task order under a coverage budget

**Method:**
- Simulate walking an order and spending each task's real tokens until the budget (50–80 % of the total) runs out.
- Score = task-normalised hits / 120, the LB metric.
- The feature model is scored leave-one-out.

**Results:**

| order | ORDER2-DIRECT 50 % / 70 % budget | R5-Poetiq 50 % / 70 % | AUC vs ARC Prize xhigh |
|---|---|---|---|
| random (mean of 500) | 19.0 / 26.5 | 24.5 / 34.2 | — |
| **ORDER2 (S1 now)** | **25.8 / 32.5** | **29.2 / 37.9** | 0.603 |
| ORDER2 + smaller total cells first (tie-breaker) | 29.9 / 33.2 | 31.5 / 40.3 | 0.627 |
| ORDER2 + smaller test input first | 28.2 / 32.8 | 30.3 / 39.9 | 0.649 |
| logistic model P(solve), LOO | 19.0 / 29.4 | 24.6 / 34.2 | 0.600 |
| ORACLE (value / real cost) | 36.9 / 36.9 | 48.6 / 48.6 | — |

- **ORDER2 is already 4–7 points better than random.** The fitted feature model does not beat it (LOO AUC about 0.5).
- **Cheap tasks are the solvable ones** (solved vs actual cost AUC 0.73), **but cost cannot be predicted from the task JSON**: LOO r = 0.62 for ORDER2-DIRECT and **−0.08 for R5-Poetiq**. The oracle's big lead comes from knowing cost in advance, which we cannot.
- **A simple, unfitted tie-breaker helps a little on both labels:** "inside each ORDER2 group, smaller total grid cells first".
  - +0.7 (DIRECT) and +2.4 (R5) points at a 70 % budget; +4.0 / +2.4 at 50 %.
  - It also agrees better with the independent ARC Prize label (AUC 0.603 → 0.627).
  - Caveat: chosen among a handful of variants on the same 120 tasks; the expected LB effect is about +1, within the noise (sd about 1.5).
- **Pause-and-resume is NEGATIVE.** Pausing every request at P (30k, 40k, 49k or 60k) and resuming after all tasks had pass 1 is worse than the straight run at every budget, e.g. at 70 %: straight 31.7 vs 25–27. Most solves need the long thinking, and pausing spreads the tokens thin.

## Conclusions

1. **No large free coverage lever** in early-stop or ordering. Tokens spent on hard requests are expensive, but they are not identifiable early.
2. **Cheap, small gain:** add the tie-breaker "smaller total cells first within ORDER2 groups". It is a one-line change. If adopted, change only this at once and measure it on the LB.
3. **The main coverage levers remain:**
   - **(a) a faster engine** (E11);
   - **(b) the per-request cap**: the LB went 21.39 at 77.8k → 34.58 at 60k. 49,152 is the floor (the user's rule), and the "49k + S1 fixes" LB test is still open.
