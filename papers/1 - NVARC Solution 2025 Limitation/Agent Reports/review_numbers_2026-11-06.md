# Numbers review of Paper 1 (lens: NUMBERS), 2026-11-06

Paper: `paper/main.tex` (line numbers below are lines of this file). Reviewer: 1 of 4, numbers lens.
Work folder: `D:\ARC AGI 2\_scratch\paper1_review\numbers\` (`copy/` = full copy of the paper folder; `indep.py`, `check2.py`, `check3.py` = my own recomputation, which does not import the team's scripts; `page03/09/10/11.png` = rendered figure pages). Nothing in the paper folder was changed except this new file.

## Verdict

**I found no wrong number.** Every core number (S, O, H, C(k), the 117 / 116 no-candidate counts, classes, union growth, unions, LB range) reproduces, both by rerunning the team's scripts and by my own recomputation from the raw submissions, logs and solutions. Scoring conventions are correct and no run is mislabelled in its score or double-counted.

The findings below are about what the correct numbers are said to mean, and about statements of method that are stronger than the data. Four should be fixed before arXiv. The fixes are all small text edits; no script or table needs to change except for optional new numbers.

---

## MUST FIX before arXiv

### M1. Run-to-run noise is understated: "differ by 4" is a net count

- **Where:** `main.tex` l.76 ("differ by 4 correct test inputs"), l.542 ("The two runs differ by 4 correct units, so single-run numbers carry run-to-run noise of that order").
- **Paper says:** the noise between two runs of one program is about 4 units.
- **Measured:** 4 is only the difference of the totals (49 - 45). The runs disagree on **10 units**: 7 correct only in R1, 3 only in R2 (shared 42, union 52). In task-norm the score difference is 2.33, but the units they disagree on carry 6.33. Their pools disagree on **11 units** (6 only in R1, 5 only in R2) while the totals differ by 1. The headroom H itself differs by 3.00 between them (+4.67 vs +7.67).
- **How:** set operations on `submission.json` of both runs and on the EXPERIMENT 1 tables (`indep.py`, `check2.py`). The paper's own l.472 ("share 42 ... together answer 52") already implies 10.
- **Why it matters:** l.542 draws a conclusion from the number, and the conclusion is too low by a factor of 2.5.
- **Correction:**
  - l.76: "...that agree on 42 correct test inputs and disagree on 10".
  - l.542: "The two runs differ by 4 correct units in total but disagree on 10 units (7 only in R1, 3 only in R2) and on 11 pool units, and their headroom differs by 3.00 points, so single-run numbers carry run-to-run noise of that order."

### M2. The "close" class is mostly explained by a trivial baseline

- **Where:** abstract l.53, Fig. 1 box l.125, l.388, l.552 ("often while the candidates are more than 90 % correct"); also `paper/kaggle_post_draft.md`, and the phrase "near misses" at l.394 follows this list.
- **Paper says:** 48 / 47 units without a correct candidate have candidates above 90 % mean cell agreement. The count is right (see "Verified"). The framing presents this as candidates coming close to the answer.
- **Measured:** copying the test input unchanged, with no model at all, already agrees above 90 % with the answer for **27 of the 48 close units in R2 and 26 of the 47 in R1**. Of the close units, 47 of 48 and 47 of 47 have the same shape as the test input. Over all 172 units, 123 have an answer of the input's shape and 39 pass the 90 % bar by copying the input.
- **How:** `check2.py`, function `copyag`: cell agreement of `challenges[task]["test"][i]["input"]` with the solution grid, for units whose shape matches. Same metric as a_u with one candidate.
- **Why it matters:** this is one of the three headline measurements in the abstract, and a reader who knows ARC will test exactly this. l.545 says the measure is coarse, but that does not reach the abstract or the conclusion. The units do differ in how hard they are; the point is that >90 % does not by itself show near-correctness.
- **Correction:** keep the number and add the baseline at l.388, e.g. "For 27 of the 48 (R2) and 26 of the 47 (R1), simply copying the test input also exceeds 90 %, so this class does not by itself show that the candidates are near the answer." Change l.552 to drop "often while the candidates are more than 90 % correct", or qualify it the same way. In the abstract, replace the clause with something like "...and 48 and 47 of the test inputs without a correct candidate exceed 90 % mean cell agreement, a bar that copying the test input alone clears for 27 and 26 of them". The abstract has 20 characters left (1,900 of 1,920), so another clause must shrink. Add the numbers to `paper_stats.py` and `NUMBERS.md`.

### M3. "58 saved versions" and "all saved results" are not what was measured

- **Where:** l.71 ("all saved results of our NVARC notebooks"), l.251 ("34 notebooks ... with 58 saved versions"), Fig. 1 l.137, abstract l.49 and l.476 ("41 distinct results of these notebooks"), `kaggle_post_draft.md` ("re-scored all 41").
- **Measured:**
  - `kaggle versions/manifest.json`: 58 is the number of versions the API returned. They come from **24 notebooks**. The other 10 of the 34 returned nothing (7 with 404, 3 with 403 on v1-v3). The manifest holds 93 "404" and 9 "403" entries.
  - `scripts/fetch_versions.py` stops a notebook after 3 consecutive misses, so it cannot see any version beyond a gap. `arc-2026d-with-codex` returned v1, v2 and then 404 for v3-v5.
  - Run C is labelled "v25" of that same notebook (Table 1, l.306). It is not v1 or v2 (51 vs 49 / 45 units, other cell count: 12 vs 35), so at least one full run of this notebook exists outside the "58 saved versions". This contradicts l.251 as written. I could not establish which Kaggle version C really is; "v25" is only a file name.
  - Run B (`fork-of-arc-2026d-32-22lb-updated-v3`) is not one of the 34 notebooks (the slug is absent from the family folder and manifest; PLAN.md says the same). So "41 distinct results of these notebooks" is not accurate; l.251 itself treats B and C as extra.
- **How:** `check2.py`, `check3.py`; reading `fetch_versions.py`; `ls` of the family folder.
- **Correction:**
  - l.251: "Our team's Kaggle account holds 34 notebooks that use the NVARC solver. For 24 of them the Kaggle API returned saved outputs, 58 versions in all." Then "Two more runs (B and C), from notebooks downloaded by hand with their cell outputs (B's notebook is not among the 34; "v25" of C is the file name of the download)".
  - l.71: "all the results we could retrieve".
  - l.137: "34 notebooks, 58 retrieved versions".
  - l.49: "41 distinct results of our NVARC runs".
  - Table 1 row C: footnote that "v25" is a file name, not a verified Kaggle version number.

### M4. Figure 1 caption says every number comes from the two scripts

- **Where:** l.146 ("every number is recomputed from saved outputs by two scripts that need no GPU"), l.141 ("no notebook-printed score used unchecked"), and l.248 ("All scores are recomputed from the saved submission files").
- **Measured:** this is false by the paper's own l.271, l.513 and l.543. These are quoted as printed and cannot be recomputed: Exp 10 (-4.67 to -9.33, 26.50), run D's 34.67 (only the 33.67 baseline matches a re-score), the 142ca369 and 95.68 % cell figures, the training losses, the audit items for run B. Also, B and C are scored from a printed set, as l.248 goes on to say.
- **Correction:** l.146: "...the measurement pipeline: scores and statistics are recomputed from saved outputs by two scripts that need no GPU; the few values that cannot be recomputed are quoted from run logs and marked as such." l.141: "printed scores checked or marked as quoted".

---

## SHOULD FIX

### S1. "All 18 distinct values" are per-notebook minima and maxima, not the leaderboard values

- **Where:** l.246; also README section 3, NUMBERS.md, PLAN.md.
- **Measured:** `paper_stats.py` tests `{lb_min, lb_max}` of each notebook, which gives 18 distinct positive values. I reproduced: all fit 100/720; no N in 100-399 fits; 100/360 and 100/240 do not fit; over denominators 100-2000 only 720 and 1440 fit. A value fits by chance with probability 7.2 %, so the joint evidence is strong (about 3e-21 under a uniform model). The conclusion stands.
- **Problem:** the wording implies all values in the records. The 63 individual scores are not in the paper folder. At least one other value exists in the same README (26.81 for `arc-2026d`, which also fits: 193.03 steps). The statement that the LB has 120 tasks is an assumption of the argument and has no citation in the paper; team memory records it as a fact, and the fit alone cannot show it.
- **Correction:** "all 18 distinct per-notebook minimum and maximum values". Add "assuming the public leaderboard, like the public evaluation set, has 120 tasks with one to three test inputs".

### S2. Leaderboard records need their scope

- **Where:** l.267, l.479.
- **Measured:** the source table (family README, l.14-17) is a snapshot of 2026-10-19, matched by notebook slug in the submission URL, and says that submissions from a raw file are not matched. The competition ended 2026-11-02. The 63 include one 0.00 (`arc-2026d` min). So "the best NVARC submission of our team" holds only among matched submissions of that snapshot.
- **Correction:** l.267: "...our team's submission list as of 2026-10-19, matched by notebook: 63 scored submissions (one of them 0.00) from 19 notebooks; submissions uploaded as a raw file are not included."

### S3. "Selection headroom is small" needs the relative figure and the sample size

- **Where:** abstract l.52 ("only"), section title l.315, l.552 ("only +4.67 to +7.67"), Conclusion.
- **Measured:** the numbers are right, but perfect selection is worth **+22.9 % (R2) and +13.0 % (R1)** of the score, and 10 of 55 and 7 of 56 in-pool answers (18 % and 12.5 %) are lost. A third attempt alone gives R2 +4.67 of its +7.67 (rank 3 task-norm 38.17 vs 33.50), R1 +1.17. The range rests on 2 runs of one program and its width (3.00) is itself run noise (M1). The conclusion "bounded mainly by generation" still holds: 117 / 116 units against 10 / 7.
- **Correction:** add the relative figures once at l.319-320, and say "in the two runs measured" next to the range at l.552.

### S4. "Repeated runs" mixes code versions; true repeats are only R1 and R2

- **Where:** section title l.460, l.468, abstract l.54, l.553 ("Repeated runs mostly re-solve the same test inputs").
- **Measured:** A, R1 / R2, B, C, D differ in code (A: fixed baseline, 1200 s, `hash()`; R2: cap 10; B: `hash()` for scoring). l.252 says so, but the abstract and conclusion read as if the same pipeline was simply repeated. The growth is therefore re-running plus code change. The only true repeat grows 49 -> 52 units and the pool 56 -> 61. Also, 51 is the start of one chosen order; over all orders the first run is 45-51.
- **Correction:** abstract: "Across five full runs of closely related code versions, ...". l.553: "Repeated runs and code versions mostly re-solve ...".

### S5. Run A is not the only exception to the MD5 seeds

- **Where:** l.92, l.95.
- **Measured:** l.92 lists MD5 seeds as a difference of "our notebooks", and l.95 names only A as an exception. Run B still has `aug_seed = hash(bk) % 1024**2` for the scoring augmentations (source of the provided notebook; also audit item 1). R1, R2, C and D use MD5 everywhere I looked. So B's scoring seeds were random between runs.
- **Correction:** add after l.95: "Run B still derives its scoring-augmentation seed from \code{hash()}."

### S6. "41 distinct results" means distinct files, and most are small

- **Where:** l.49, l.71, l.251, l.476.
- **Measured:** the 39 files are distinct by SHA-256 (reproduced), from 20 notebooks (reproduced). They contain only **20 distinct correct-unit sets**, 22 with B and C. Of the 41 results, 24 have at most 4 correct units, 9 have 5-20, 2 have 21-44, 6 are the full runs. The 2 units the partial runs add over the six full runs are `78332cb0_0` and `c7f57c3e_0`. l.251 says "most of them partial", so nothing is false.
- **Correction:** optional, at l.251: "(22 distinct sets of correct units)".

---

## MINOR

- **m1. Duplicate ratio, l.427-432.** 0.378 / 0.381 are correct as the mean of d_u over 172 units, including placeholders with d_u = 0. But the sentence before gives 14.0 and 7.3, which suggests 0.48. Pooled 1 - sum|P| / sum|C| is 0.481 / 0.473; the mean over non-placeholder units is 0.412 / 0.418. One clause would prevent the question.
- **m2. Terminology, l.433.** "The largest pool has 27 candidates": pool is defined at l.255 as distinct grids. 27 is the largest |C_u|; the largest |P_u| is 19 in both runs. Say "The largest candidate multiset has 27 candidates (19 distinct grids at most)".
- **m3. Cap list, l.162 and l.209.** "10 in R2, 5 in R1, A and D" is true but incomplete: B and C also have 5. R2 is the only run with 10.
- **m4. Order of runs.** The abstract and text give R2 first ("55 and 56"), the tables give R1 first, and l.54 gives R1 first ("49 and 45"). All pairs are correct, but the abstract flips order inside one paragraph.
- **m5. Basis of the 68 %.** 67.4-68.0 % is per unit. In task-norm the no-candidate share is 65.69 % (R2) and 66.25 % (R1). "About two thirds" at l.552 covers both.
- **m6. Five vs six runs, l.468-470.** No reason is given for leaving D out. It has no effect: D adds 0 units, and the core is 37 (27.50) with or without D.
- **m7. Stale companion file.** `v25_v2_ceiling_findings_2026-11-04.md` section 4 still says "Our 23 saved NVARC runs" and "+4, +2 per new run", which disagrees with the paper's 41 and +6, +1, +3, +1. l.557 publishes this folder.

---

## Verified as correct

### Rerun

`score_all.py` and `paper_stats.py` run on Python 3.14 in the scratch copy. All outputs are byte-identical (md5) to the paper folder: `data/paper_numbers.json`, `lb_by_notebook.csv`, `run_scores.csv`, `union_units.json`, `pool_units.json`, all four `.dat` files.

### Independently recomputed (own code, raw files)

- **Data set:** 120 tasks, 172 units, n_t = 1 / 2 / 3 for 71 / 46 / 3 tasks; challenges match solutions in keys and test counts. "One to three" is right.
- **Table 1 scores:** A 51 / 37.67, R1 49 / 35.83, R2 45 / 33.50, B 51 / 37.83, C 51 / 37.67, D 47 / 33.67. Submissions hold 120 tasks and 172 units. As % of 120: 31.39, 29.86, 27.92, 31.53, 31.39, 28.06. Weighting is task-normalised (1 / n_t) throughout; correct / 172 is reported separately and the two are never mixed.
- **B and C:** `sets[2]` is really the `run_dual_selection` set (label printed right after it); the notebooks' "Reload score" 37.83 and 37.67 match. The benchmark uses `v[:n_guesses]`, and `n_guesses=2` in the reproduction cell, so l.248 holds.
- **Pool:** R2 55 / 41.17, R1 56 / 40.50; H +7.67 / +4.67; +10 / +7 units; EXPERIMENT 1 and 18 pools equal; submitted subset of pool; submitted set equals probmul rank <= 2 in both runs. Log lines "Oracle Score" agree.
- **Top-k:** R2 36/45/51/54/55/55, R1 39/49/52/54/55/56 at k = 1/2/3/5/10/20. Max rank 10 (R2), 15 (R1). Third attempt +6 / +3. Log Exp 14 agrees (R1 lines 8709-8715 read; "at most 18 / 15" at R2 l.8688, R1 l.8718).
- **No correct candidate:** 117 (68.02 %), 116 (67.44 %). Classes 48/33/22/14 and 47/32/22/15, summing to 117 / 116 and, with submitted and in-pool units, to 172. No unit sits exactly on a cut point.
- **Rankers:** same rank 48/55, 51/56; Pearson 0.961, 0.991; kgmon top 2: 47 / 34.83 (+1.33), 48 / 34.17 (-1.67); printed 34.8 and 34.2.
- **Diversity:** candidates 2416 / 2470 = 14.05 / 14.36; distinct 7.29 / 7.56; dup 0.378 / 0.381; non-placeholder 15.20 / 7.85 (n = 158), 15.64 / 8.19 (n = 157); max 27; <= 3 distinct 33 / 26, absent 19 / 14; 5961cc34_0 1 of 16, 95.88 %, in both runs (27 x 27).
- **Placeholders:** FALLBACK_SAVE units equal the flagged units exactly (14 / 15, all NOT_PROCESSED, 11 tasks each); 181.7-801.9 s, 133.6-872.9 s; 120 tasks finished; 0 `timeout after`, 0 PUZZLE / DFS / DECODE / SCORE exceptions. Code path checked: NOT_PROCESSED means no view returned a beam, and invalid grids would log EMPTY_DECODED_RESULT, of which there are none. So l.390 holds. Longest search call 246.9 / 227.1 s (own implementation). One task in each run ran past 1600 s (1723.3 / 1705.6 s) without a time-out line; this fits the check-before-batch logic of l.231.
- **R1 vs R2:** shared 42, union 52; pools shared 50, union 61 (44.50); 240 seed lines identical; 120 of 120 tasks at 128 steps; losses 0.018519 / 0.018989. Code diff: `arc_solver.py` line 59 only; loader and decoder identical; `starter.py` differs only in commented lines. Scoring seed is MD5-derived in both (l.531 of the solver), so "identical seeds" is fair.
- **Union:** 51/57/58/61/62; 37.67/42.00/42.50/44.00/45.00; all 120 orders min 45/52/57/60/62, max 51/59/61/62/62; D adds 0; core 37 (27.50); 4 units in one run only; pairs shared 41-45, union 52-59.
- **All results:** 39 SHA-distinct files from 20 notebooks; 62 units / 45.50 from files; with B and C 64 / 46.50 = 38.75 %. All evidence 70 / 50.83 = 42.36 %; 172 - 70 = 102. Family folder has 34 notebooks.
- **LB:** 63 submissions, 19 notebooks, best 32.64, range 27.36-32.64, difference 5.28, as extracted from the README table; 18 distinct values; fit as in S1.
- **Wall time:** B 16,868 s = 4.69, C 17,110 s = 4.75 (papermill), R1 and R2 from the summary; log spans A 4.673, R1 4.765, R2 4.752, D 4.794 h are consistent. The range 4.68-4.80 is right.

### Pipeline constants checked against code (`arc_solver.py`, `arc_loader.py`, `arc_decoder.py` of R2)

Rank 256, alpha 32, rsLoRA, target modules, lr 5e-5, cosine, batch 1, 1 epoch, max length 8192, 8 x 16 = 128 training samples, 8 x 2 = 16 views, batches of 4, threshold -ln 0.2, strict inequality, log-softmax over the full vocabulary (so Eq. 1 holds), 540 s, 1600 s with training included, 8 further augmentations, placeholder 999. Eq. 2 and 3 match `getter_full_probmul_3` and `getter_kgmon`. Dual selection logic matches l.176-177 and Algorithm 1. Run A: baseline 3, 1200 s, `hash()`, cap 5. D: cap 5, 1600 s, MD5. Reproduction notebook: baseline 3, 1200 s, `hash()`, `run_selection_algo()` default kgmon with `n_guesses=2`.

a_u (Eq. 6) matches cell 30 (mean over all candidates, other shapes count 0).

### Quoted values found in the logs as stated

- Pitfall 1: 117 (68.0 %), R2 l.7708.
- Pitfall 2: 12.78 vs 4.18, R2 l.7461.
- Pitfall 3: seven variants, four at 26.50, -4.67 to -9.33, "WINNER" at -4.67.
- Pitfall 5: meta-selector 0.00 vs 33.50.
- Pitfall 7: 37.8/111.
- Pitfall 8: 47/120, 34/120 (source: rank 1 in either ranker only).
- Pitfall 9: 37.8 twice.
- 142ca369: 72.84 / 93.16 %, 88 / 26 cells, all candidate values 0 (Counters at R2 l.9254, l.9280); arithmetic checks with 18 x 18 and 20 x 19. R1: 95.68 %, 14 cells 2 for 6.
- Run D: 33.67 / 33.67 / 34.67.

### Figures and tables

- Table 1 and Table 2: every cell matches.
- Figures 2-5 read the `.dat` files, which match my recomputation. Rendered pages 3, 9, 10, 11 inspected: bars, curves and legends show those values and nothing is clipped (distinct grids go to 19, xmax 19.7). Fig. 2 y axis starts at 30, labelled.
- Fig. 1 box numbers match.

### Package

`arxiv/main.tex` equals `main.tex` without comments; tarball `main.tex` is identical; `.dat` copies identical; abstract 1,900 characters; PDF 14 pages; 5 figures, 2 tables, 3 algorithms; every decimal in the tex that is a paper number appears in the PDF text (the 9 absent ones are layout parameters).

---

## Could not verify

- **Individual LB scores, the mean 29.36, the 63 count against Kaggle.** The only source in the folder is the README summary table. I did not query Kaggle.
- **How many versions really exist on Kaggle, and which version C is.** See M3.
- **Candidate-level claims.** Candidate grids were not saved, so ranks, a_u, counts and pools come from printed tables. I checked them for internal consistency and against submissions, not from candidates.
- **B and C scores** rest on printed sets; no submission file exists.
- **Exp 10 and run D selector scores** are printed values; not recomputable.
- **Pitfall 5 sweeps and the "Pruned by max_score" label:** I confirmed the 0.00 figure but did not re-derive the threshold and beam-width claims from cell code. NUMBERS.md and the findings file are the source.
- **Competition overview wording** (l.246): I did not open the Kaggle page.
- **"Four NVIDIA L4 GPUs":** ranks 0-3 and accelerator `nvidiaL4` are in the logs and metadata, which supports it. I did not see a device listing.
