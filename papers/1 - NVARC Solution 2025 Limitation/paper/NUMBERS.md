# Where every number in `main.tex` comes from

Run from the paper folder (`papers/1 - NVARC Solution 2025 Limitation/`):

```
python -B scripts/score_all.py     # S: unions over all submissions and pools
python -B scripts/paper_stats.py   # P: everything else; writes data/paper_numbers.json and paper/figures/data/*.dat
```

`score_all.py` rewrites `data/run_scores.csv` and `data/pool_units.json`. On a non-Windows machine the content is the same,
but the line endings and the order of two rows differ. Restore them with `git checkout --` if nothing else changed.

R1 = `arc-2026d-with-codex` v1 (`kaggle versions arc-2026d-with-codex/v1/output/`), R2 = v2 (`source notebook arc-2026d-with-codex/output/`).
Log line numbers refer to the `*_stdout.txt` files.

| number(s) in the paper | source |
|---|---|
| 120 tasks, 172 test inputs | `data/arc-agi_evaluation_solutions.json` |
| 34 notebooks; 58 saved versions | `kaggle versions/manifest.json`; `data/version_log_summary.csv` (58 rows) |
| 41 distinct results = 39 submission files from 20 notebooks + 2 printed sets (B, C) | S: `runs scored (unique): 41`; `data/run_scores.csv` |
| six full runs: units, task-norm, wall h (Table 1) | P: `[full]` lines; wall from `data/version_log_summary.csv`, papermill metadata for B and C |
| R1 / R2: 49 / 45 units, 35.83 / 33.50 | P `[v1]` / `[v2]`: `submitted ...` |
| pool 56 (40.50) / 55 (41.17); headroom +4.67 / +7.67; +7 / +10 units | P; also S `pool ...` lines; R2 log `Oracle Score: 41.17 (55/172 puzzles)`, R1 log line 8693 |
| no correct candidate 116 (67.4 %) / 117 (68.0 %) | P `no correct candidate` |
| top-k 39/49/52/54/55/56 and 36/45/51/54/55/55 | P `top-k` (from the EXPERIMENT 1 tables); R2 log Experiment 14, R1 log lines 8709-8715 |
| submitted set = probmul rank <= 2 | P `submitted == top-2: True` |
| same rank 51/56, 48/55; correlations 0.991 / 0.961 | P `probmul = kgmon rank`; logs `Same rank in both scorers`, `Rank correlation` |
| kgmon top 2 instead: 48 (34.17, -1.67) / 47 (34.83, +1.33) | P `kgmon top-2` (EXPERIMENT 1 rank_b <= 2); logs `acc: 34.2/120 ('score_kgmon')` (R1), `acc: 34.8/120 ('score_kgmon')` (R2) |
| Pearson rank correlation 0.991 / 0.961; duplicate ratio 0.381 / 0.378 | P `rank Pearson`, `duplicate ratio` (recomputed; equal to the printed values) |
| R1 = R2 program: 240 identical seed lines; shared 42, union 52; pools shared 50, union 61 (44.50) | P `[R1 vs R2]`; code diff R1 vs R2: `arc_solver.py` line 59 only (MAX_BEAM_CANDIDATES 5 vs 10) |
| training losses 0.018519 / 0.018989 (271d71e2), 128 steps | R1 / R2 logs: `training stats for puzzle 271d71e2: TrainOutput(global_step=128, training_loss=...)` |
| longest search call + scoring 246.9 s (R2) / 227.1 s (R1) | P `longest search call + scoring` (JSON log timestamps) |
| Exp 14 "recover at most 18" (R2) / 15 (R1); real gain to top 5: 9 / 5 | R2 log line 8688, R1 log line 8718; P top-k (54 - 45, 54 - 49) |
| Exp 10: seven variants -4.67 to -9.33, four at 26.50 | R2 log, EXPERIMENT 10 (as printed); code: notebook cell 28 (`custom_getter([cand], ...)`) |
| R1 142ca369_0: 95.68 %, 14 cells 2 instead of 6 | R1 log, EXPERIMENT 17 |
| reproduction differences (baseline 3, kgmon top 2, 1200 s, `hash()` seeds) | `NVARC family (...)/reproduce-nvarc-2025-results/output/arc_decoder.py` l.21, `arc_solver.py` l.405 and l.443, notebook cell 7; run A: `kaggle versions/arc-2026d/v4/output/arc_decoder.py` l.23, `arc_solver.py` l.44, l.411, l.448 |
| selectors 33.67 / 33.67 / 34.67 (run D) | D log lines 14185-14187 (`experiment-with-arc-agi-2_v10_stdout.txt`); 33.67 = P `[full] experiment-with-arc-agi-2 v10` |
| re-weightings -4.67 to -9.33 | R2 log, EXPERIMENT 10 (as printed; not recomputed) |
| classes close / partial / far / placeholder 48/33/22/14 (R2), 47/32/22/15 (R1) | P `no correct candidate ... {...}` (EXPERIMENT 18 tables, cut points of notebook cell 30) |
| placeholders in 11 tasks, 182-802 s / 134-873 s, 0 time-outs, 0 exceptions | P `fallback reasons ... in 11 tasks ...` |
| 142ca369: 72.84 % / 93.16 %, 88 / 26 cells, all predicted 0 | R2 log, EXPERIMENT 17 |
| 14.0 / 14.4 candidates, 7.3 / 7.6 unique, duplicate ratio 0.378 / 0.381 | logs, EXPERIMENT 2; P recount 14.05 / 7.29 and 14.36 / 7.56 |
| non-placeholder 15.2 / 7.85 and 15.64 / 8.19; n = 158 / 157; max 27 | P `candidates / unique` |
| <= 3 distinct: 33 / 26, answer absent 19 / 14; 5961cc34_0 1/16, 95.9 % | P `candidates / unique` |
| union growth 51 / 57 / 58 / 61 / 62; 37.67 / 42.00 / 42.50 / 44.00 / 45.00 | P `[growth]` |
| all orders: after 2 runs 52-59 | P `[growth] all 120 orders` |
| core 37 (27.50); in exactly one run 4; pairwise 41-45 of 52-59; D adds 0 | P `[core]`, `[union6]` |
| union of all submissions 64 (46.50 = 38.75 %); pools 61 (44.50); all evidence 70 (50.83 = 42.36 %); 102 = 172 - 70 | S |
| LB = per-task averaging (percent): 18 distinct values all multiples of 100/720, no N in 100-399 fits a plain average over N outputs | P `[lb]` (`fit_task_average_100_720`, `unit_average_N_fitting_100_399`); wording of the overview: Kaggle competition page, Evaluation section |
| LB best 32.64; 10 submissions of one version (331102752) 27.36-32.64, mean 29.36; 63 submissions, 19 notebooks | P `[lb]` (`data/lb_by_notebook.csv` incl. `scored_versions`, from the NVARC family README table) |
| run B: 37.8/111, 47/120, same 37.8 for probmul and dual | `v3_misleading_claims_audit_2026-11-04.md` items 3, 8, 10, 11 (cell outputs of the provided v3 notebook) |
| pitfalls 1-5 (R2 diagnostics) | R2 log Experiments 5, 8, 10, 16, 18-20; `v25_v2_ceiling_findings_2026-11-04.md` section 6 |
| pipeline settings | R2 `arc_solver.py` (LoRA r 256, alpha 32, rsLoRA, lr 5e-5, 16 train augmentations, `augment(n=2)` for test views, `max_score = -ln 0.2`, `MAX_BEAM_CANDIDATES = 10`, 540 s and 1600 s caps, placeholder), `arc_loader.py` (`augment`: transpose x rot90 x colour permutations), `arc_decoder.py` (probmul, kgmon, dual selection) |
| abstract (arXiv version): 7-10 lost in selection, 116-117 without a correct candidate | the rows above (+7 / +10 units; 116 / 117); `paper/arxiv/METADATA.md` holds the same abstract |
| a third attempt would add 6 and 3 units (Section 4.1) | P top-k: rank <= 3 minus rank <= 2 (51 - 45, 52 - 49) |
| resubmissions vary by up to 5.28 LB points (Conclusion) | P `[lb]`: 32.64 - 27.36 |
