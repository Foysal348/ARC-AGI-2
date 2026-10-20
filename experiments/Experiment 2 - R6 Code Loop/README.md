# Experiment 2 — R6 Code Loop

**Status:** closed (2026-10-06).

## Idea

Think at xhigh, write `solve(grid)`, run it on every training pair, get feedback, repair. In other words, DIRECT reasoning plus code verification.

## Result (run 5, EVAL12)

**2/22**, the same as DIRECT v1.4, at **92.7k tokens per task** (a projected 28 h for the hidden set).

**What happened in the turns:**
- 46 of 47 turns were force-closed: xhigh never ends within a 12k repair cap.
- 35 repair turns led to 1 verify.
- 3 of 4 train-verified programs were wrong on test.

## Lessons (carried into Experiments 4 and 5)

- xhigh always uses the full budget it is given.
- Train verification is a weak selector on 2–5 pairs.
- Repair turns without the previous reasoning restart from scratch.

## Folders

- **Experiment Notebooks/:** `Q27-R6-CODE-LOOP-v1_0/1.ipynb`. The builder is `make_q27x_r6.py` in Experiment 1's `Build/Q27X_build`.
- **Experiment Results/Kaggle Runs/:** `Q27X_run5_R6`, `q27-r6-code-loop.ipynb`.
