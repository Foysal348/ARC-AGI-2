# NVARC ceiling: findings from arc-2026d-with-codex v25 and v2 (2026-11-04)

Sources:
- **v25** = `provided notebooks/arc-2026d-with-codex_v25.ipynb` (user's file; 12 cells with outputs; run 17,110 s = 4.75 h; 120 eval puzzles / 172 test inputs).
- **v2** = `source notebook arc-2026d-with-codex/` (the 2026-10-19 download; 35 cells, 34 numbered experiments; outputs only in the run log, extracted to `output/arc-2026d-with-codex_stdout.txt`; run 17,116 s = 4.75 h).
- Model: `sorokin/qwen3_4b_grids15_sft139` (NVARC 2025 Qwen3-4B), test-time LoRA training per puzzle, `turbo_dfs` beam search, augmentation scoring, 2 attempts.

## 1. Numbers of the two runs

| measure | v2 | v25 |
|---|---|---|
| submission (pass@2), task-norm / 120 | 33.50 (27.92 %) | 37.67 (31.39 %) |
| test inputs right (of 172) | 45 | 51 |
| puzzles with >= 1 test input right | n/a | 40 |
| candidate pool contains the answer | 55 / 172 test inputs, oracle 41.17 task-norm | 48 puzzles |
| correct candidates / all candidates | 398 / 2,416 | 426 / 2,466 |
| puzzles with no valid grid | 14 test inputs with only a fallback grid | 0 |

## 2. Selection is close to its ceiling

- v2 Experiment 13: dual selection 33.50, oracle selection 41.17. **Perfect selection adds at most +7.67 task-norm.**
- v2 Experiment 14 (top-k coverage of the answer): top-1 36, top-2 45, top-3 51, top-5 54, **top-10 = top-20 = top-100 = 55 / 172**. Past rank 10 there is nothing left to select.
- v2 Experiment 4: probmul and kgmon put the answer at the same rank in 48 / 55 cases (87.3 %), rank correlation 0.961. The two scorers are nearly the same signal, so a second scorer cannot recover much.
- v2 Experiment 10: every re-weighting of beam vs augmentation score was **worse** than the shipped scorer (−4.67 to −9.33 task-norm).
- v25 CHECK 9: 8 puzzles are in the pool but not in the 2 attempts (`142ca369`, `20270e3b`, `3dc255db`, `4c7dc4dd`, `62593bfd`, `800d221b`, `8b7bacbf`, `dfadab01`): 40 -> 48 puzzles with perfect selection.

## 3. The real wall is generation

- **117 / 172 test inputs (68 %) never get a correct candidate** (v2 Experiments 5, 14, 19).
- v2 Experiment 18 classifies those 117 by `avg_sim`, the MEAN cell agreement with the answer over ALL candidates of the
  test input (a candidate of another shape counts as 0; notebook cell 30). Corrected 2026-11-04: an earlier version of this
  file said "closest candidate", which is not what the code computes. Cut points are the notebook's: > 90, > 50, else.
  - 48 "close" (mean agreement > 90 %, but no candidate exact);
  - 33 "partial" (50-90 %);
  - 22 "far off" (<= 50 %);
  - 14 only a fallback grid (no valid grid produced at all).
  - v1: 47 / 32 / 22 / 15 (`scripts/paper_stats.py`).
- **Diversity collapse** (v2 Experiments 2, 21, 22, 23, 26, 27, 32, 33): on average 14.0 candidates per test input but only 7.3 unique (duplicate ratio 0.378).
  - Several test inputs have 1-3 unique grids out of 16-18 candidates, e.g. `38007db0_1` 1 / 16, `38007db0_0` 2 / 16, `45a5af55_0` 2 / 18.
  - In many test inputs 100 % of candidate pairs are > 95 % similar.
  - The code has no diversity term in `turbo_dfs` (Experiments 24, 30).
- Systematic near misses: `142ca369` best candidates are 72.8 % / 93.2 % similar. Every wrong cell is a coloured cell predicted as 0 (88 / 88 and 26 / 26 cells; Experiments 6, 17, 25, 28, 29). This is a generation error, not a parsing or selection error.
- v25 CHECK 9: of the 72 unsolved puzzles, 24 are dynamic-shape, 11 high size-ratio, 3 tier-3, and 8 are "easy_static". The model misses easy same-size puzzles too.

## 4. More runs hit the same subset

- Our 23 saved NVARC runs (April-September 2026; all 64 come from the arc-2026d family) together reach **64 / 172 test inputs, task-norm 46.50 = 38.75 %**. The best single run reaches 37.83.
- Re-running with other seeds or versions mostly re-finds the same puzzles: v3 adds 4 new test inputs over 9 earlier runs, v25 adds 2.
- LB: best 32.64. The same version scored 27.36-32.64 over 10 submissions (noise about ±2.5).

## 4b. Second run with full pool data: arc-2026d-with-codex v1 (added 2026-11-04)

- **Download.** Fetched with `version_label="v1"` / `"v2"` into `kaggle versions arc-2026d-with-codex/`.
  - v2's submission.json is byte-identical to our 2026-10-19 copy (sha 17034cca).
  - v1 is a separate run: 17,164 s, all 120 puzzles trained, Experiments 1-19.
- **v1 numbers:**
  - dual 35.83 (49 / 172 test inputs), oracle 40.50 (pool has the answer for 56 / 172), selection headroom +4.67;
  - top-k: 39 / 49 / 52 / 54 / 55 / 56 / 56 (k = 1 / 2 / 3 / 5 / 10 / 20 / 100);
  - probmul vs kgmon: same rank in 51 / 56, correlation 0.991;
  - 14.4 candidates per test input, 7.6 unique (duplicate ratio 0.381).
- **Pool union v1 + v2:** 61 test inputs, task-norm 44.50. 50 are shared, 6 only in v1, 5 only in v2.
- **All NVARC evidence together** (23 submissions + 2 pools): 70 / 172, task-norm 50.83 = 42.36 %.

## 4c. Every saved version of all 34 notebooks (added 2026-11-04)

- **Fetched.** 58 saved versions (`kaggle versions/`, `manifest.json`).
  - 41 unique submissions are scored (`data/run_scores.csv`).
  - Per-version wall time, puzzles trained, printed score and OOM are in `data/version_log_summary.csv`.
- **The union does not move.** Over all 41 unique submissions it stays at 64 / 172 (task-norm 46.50). The 18 extra versions added 0 new right units.
- **Six full runs (all 120 puzzles trained, 4.7-4.8 h each):**

  | run | units | task-norm |
  |---|---|---|
  | arc-2026d v4 | 51 | 37.67 |
  | arc-2026d-with-codex v1 | 49 | 35.83 |
  | arc-2026d-with-codex v2 | 45 | 33.50 |
  | fork-of-arc-2026d-32-22lb-updated-v3 | 51 | 37.83 |
  | arc-2026d-with-codex v25 | 51 | 37.67 |
  | experiment-with-arc-agi-2 v10 | 47 | 33.67 |

- **Union growth over five of the full runs** (v10 adds 0 new units):
  - cumulative 51 -> 57 -> 58 -> 61 -> 62 units;
  - in task-norm 37.67 -> 42.00 -> 42.50 -> 44.00 -> 45.00;
  - new units per added run: 6, 1, 3, 1.
- **Core.** 37 units (task-norm 27.50) are right in all five. Pairwise, two full runs share 41-45 units out of a union of 52-59.
- **A third selector test.** experiment-with-arc-agi-2 v10 (full run, 4.8 h) benchmarks 3 selectors on the same pool, counting the actual 2 returned grids:

  | selector | task-norm | units |
  |---|---|---|
  | CURRENT | 33.67 | 47 |
  | IMBUE_EXACT | 33.67 | 47 |
  | IMBUE_QWEN_DIVERSE | 34.67 | 48 |

  Changing the selector moves the score by at most 1.0. The log is in `kaggle versions/experiment-with-arc-agi-2/v10/output/`, and the stdout is extracted to `*_stdout.txt`.
- **Partial runs.** arc-2026d v7 and v8 hit CUDA OOM before the end (108 and 107 puzzles trained).

## 5. What the data does NOT prove (keep out of the post, or state as limits)

- That no NVARC variant can ever go past this: the data covers our runs of one model with one search procedure.
  - Union growth is small but not zero (+4, +2 per new run).
  - Nobody measured more samples, a diversity penalty (Experiment 34 is only a proposal; never run) or a larger model.
- v2 Experiment 20 "lowering max_score" is circular. It can only see candidates that already survived the real threshold, so "55 survive at 1.609" says nothing about pruned answers.

## 6. Misleading outputs inside these two notebooks (do not quote as findings)

1. **Experiment 5** "Lost at ranking stage: 117 (68.0 %)". Those 117 are not in the pool at all: that is a generation loss, not a ranking loss.
2. **Experiment 8** labels are inverted. Example `b6f77b65_0`:
   - GT aug_mean 12.78 vs rank-1 4.18, and GT beam 1.31 vs 1.18 (lower is better);
   - the cell prints "GT has BETTER beam" and "BETTER aug";
   - so the GT is actually worse on both.
3. **Experiment 10** prints "WINNER" and "RECOMMENDATION" for a variant that is 4.67 below the baseline. Every variant lost.
4. **Experiment 16** "Meta-Selector Score: 0.00": a broken simulation (it gives 0 against a 33.50 baseline), not a measurement.
5. **Experiment 18** "FALLBACK exists -> Worker crashed or timed out" (14 test inputs), and **Experiment 19** "Pruned by max_score" for beam 999. Both are the fallback grid written when no valid grid was produced, not crashes or pruning.
   - Same pattern as v3: there, all 9 no-output puzzles finished normally (137-774 s).
6. **Experiment 20** "Beam Width" rows are just top-k of the existing pool, not a beam-width sweep.
7. **v25 CHECK 7 and CHECK 11** did not run ("No timing data found").
