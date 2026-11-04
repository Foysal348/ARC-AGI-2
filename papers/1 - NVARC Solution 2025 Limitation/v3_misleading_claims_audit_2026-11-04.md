# fork-of-arc-2026d-32-22lb-updated-v3: claims vs code vs run output (2026-11-04)

Source: `provided notebooks/fork-of-arc-2026d-32-22lb-updated-v3.ipynb` (11 cells, saved outputs, run 16,868 s = 4.69 h, 120 eval puzzles).
Every item below was checked against the cell source and the saved cell outputs.

## A. Markdown (cell 0) claims that the code does not support

1. **"Fully reproducible runs" (fix 1).** Train and eval augmentation seeds use `hashlib.md5`, but the scoring augmentation
   still uses Python's randomized hash: `aug_seed = hash(bk) % 1024**2` (arc_solver.py, cell 5, line 459). So the augmented
   scores (`score_aug`) that both rankers use still change between runs. The run is not reproducible.
2. **"Worker Sync Timeout: 300-second timeout on worker sync files" (fix 9).** There are no sync files and no 300 s timeout in
   starter.py (cell 6). Workers share only an import lock. This fix does not exist.
3. **"Dual Selection: combine score_full_probmul_3 and score_kgmon, pick top 1 from each" (fix 8).** The code builds the pool as
   `ranked_a[:3] + ranked_b[:3]`; attempt 2 = `pool[1]`, which is probmul rank 2 whenever probmul has 2 or more candidates.
   kgmon's top 1 is used only when probmul has a single candidate. In practice dual selection = probmul top-2. The run proves it:
   `score_full_probmul_3` 37.8 and `run_dual_selection` 37.8 (cell 8).
4. **"Dynamic baseline: all scores are positive; ranking remains stable" (fix 4).** The getter is called per candidate
   (per unique solution group in `score_sum`), so each candidate gets its own baseline = its own max beam score + 1.
   Candidates are no longer scored on a common scale, so the ranking changes versus the fixed baseline; "remains stable" was not
   measured. `score_aug` values larger than that baseline still give negative terms.
5. **"DEBUG_PUZZLES: 14 puzzles. Manual run: only these 14 puzzles."** The list has 49 entries with 3 duplicates
   (`136b0064`, `291dc1e1`, `271d71e2`), i.e. 46 unique, and `RUN_FULL_EVAL = True`, so the manual run queued all 120
   (log: `Queued 120 puzzles (rerun_mode=False, RUN_FULL_EVAL=True)`).
6. **"EDA ordering: more puzzles solved within the 12-hour limit" (fix 7).** The run finished all 120 puzzles in 4.69 h, so the
   order could not change what was solved. No measurement backs the claim. The weights are fixed, not "dynamic".
7. **"Status: all critical fixes applied and verified."** Items 1-4 above show they were not.

## B. Scores and diagnostics that read better than the run

8. **"acc: 37.8/111".** The denominator is the number of puzzles with any decoded output (111), not 120. The real score is
   37.83/120 = 31.53 % (51/172 units), not 34 %.
9. **"FINAL SUMMARY: TIER vs SOLVE RATE" (Tier 1 27/60 = 45.0 %, Tier 2 19/47 = 40.4 %, Tier 3 1/4).** The denominators drop the
   9 no-output puzzles (the starter counted 65 / 51 / 4). With the full tiers: 27/65 = 41.5 %, 19/51 = 37.3 %, 1/4.
   The numerators count a puzzle as solved if any candidate in the pool is correct, not the 2 submitted attempts.
10. **"SUMMARY: 47/120 puzzles solved in decoded candidates"** is a pool oracle (any candidate anywhere, any one test input),
    not the submission. Example: `3dc255db`, `58490d8a`, `8b7bacbf`, `4c7dc4dd` are "SOLVED" but the correct grid is at rank 5-7,
    so the submission misses it.
11. **"34/120 puzzles solved by DUAL SELECTION (our final method)" is wrong in the other direction.** The check marks a puzzle
    as dual-solved only when the correct grid is rank 1 in either ranker (cell 10, line 188); it ignores attempt 2. The submitted
    file actually solves at least one test input in 43 puzzles (33 fully). Nine puzzles shown as "SOLVED / dual No" at probmul
    rank 2 (`2ba387bc`, `31f7f899`, `4c3d4a41`, `7b0280bc`, `800d221b`, `a6f40cea`, `eee78d87`, part of `62593bfd`, `7ed72f31`)
    are solved by the real submission.
12. **The rank columns are per puzzle but computed per test input.** For multi-query puzzles the loop overwrites
    `rank_probmul` / `rank_kgmon` with the last solved test input, so the shown rank does not describe the puzzle.
13. **"No output (likely timeout/queue skip): 9".** None of the 9 timed out or was skipped. All 9 finished normally
    (137-774 s each, 4-8 decode batches) and produced zero valid grids (0 scoring calls), so no candidate file was written:
    `136b0064` 137 s, `13e47133` 774 s, `291dc1e1` 157 s, `4e34c42c` 203 s, `5dbc8537` 212 s, `6e4f6532` 489 s,
    `6ffbe589` 191 s, `b10624e5` 634 s, `fc7cae8d` 276 s. The only per-puzzle timeout in the log is `d8e07eb2` (1,732 s),
    and that puzzle is solved.
14. **CHECK 4 shows 7 "NO OUTPUT" rows, CHECK 5 lists 9.** The two checks disagree; `eee78d87` is printed twice in CHECK 4.
15. **CHECK 1 "SEED DETERMINISM" and CHECK 3 "TRAIN/EVAL SEED SEPARATION"** only recompute the md5 seeds, so they always pass.
    They do not test the `hash(bk)` scoring seed (item 1), which is the non-deterministic one.
16. **CHECK 2 is titled "FOR DEBUG_PUZZLES"** but the run used all 120; CHECK 7 (time vs tier) did not run
    ("No timing data found").

## C. Real numbers of this run

| measure | value |
|---|---|
| wall time | 16,868 s (4.69 h); sum of per-puzzle times 65,622 s over 4 workers |
| puzzles trained / finished | 120 / 120 |
| puzzles with no valid grid | 9 |
| submission (pass@2) | 51/172 units, task-norm 37.83/120 = 31.53 % |
| puzzles with >= 1 test input right / fully right | 43 / 33 |
| pool oracle (any candidate) | 47 puzzles |
| subkey candidates correct | 396 / 2,427 |
