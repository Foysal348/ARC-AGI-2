# Review of Paper 1 against the team's own record (2026-11-06)

Reviewer lens: OUR OWN RECORD. Paper checked: `paper/main.tex` (line numbers below refer to it), `paper/NUMBERS.md`, `paper/kaggle_post_draft.md`, `paper/arxiv/METADATA.md`.
Times in transcript quotes are UTC as stored in the transcript; add 6 h for BDT.

Short verdict: the title, tone and banned claims all follow the user's decisions, and no number I rechecked is wrong. But three statements conflict with runs already in our folder, and all three are easy to fix. The main argument survives every fix, and two of the fixes make it stronger.

## Summary of findings

| id | rank | finding |
|---|---|---|
| M1 | MUST FIX | Run C is a third run of the R1 program with the same seeds (51 units). The paper says only R1 and R2 are the same program, so it understates run-to-run noise: 45-51 units, not 45-49. |
| M2 | MUST FIX | "We did not test ... a lower probability threshold ... longer test-time training" and "All runs use ... the same test-time training recipe and the same thresholded search" are false for our notebooks. Threshold 0.15, 24 views and 24 TTT colour permutations were all run and scored on the LB. |
| M3 | MUST FIX | The near-miss showcase 142ca369 ("No ranking of such candidates can produce the answer") has a correct candidate at rank 7 in run C, the same program. This also adds at least one unit to the "70 units / remaining 102" figures. |
| S1 | SHOULD FIX | "The other four full runs provide submission-level data only" is false for B and C, which print task-level pools (47 and 48 tasks). |
| S2 | SHOULD FIX | The LB best (32.64) is not one of the analysed runs. That version has only a 35-task eval run, and the notebook with pool data scored 27.64-28.89. |
| S3 | SHOULD FIX | /120 points and LB percentages sit side by side. 33.50 is 27.92 %, and the oracles 41.17 / 40.50 are 34.31 % / 33.75 %. Our own record has already flagged this trap. |
| S4 | SHOULD FIX | Direct evidence on NVARC variants is missing: the recolour data bug (LB 28.47 to 35.97) and the lower-threshold plus more-epochs port. Both are in `resources/Kaggle Writeups`. |
| S5 | SHOULD FIX | Run C's label "v25" is a browser download suffix, not a Kaggle version. |
| S6 | SHOULD FIX | "Built 34 notebooks" and "41 distinct results of these notebooks" are loose: B is not among the 34, and authorship is unverified. |
| S7 | SHOULD FIX | `kaggle_post_draft.md` repeats M1, M2, S3 and S6. |
| S8 | SHOULD FIX | Files that will go public still carry superseded statements (findings file, family README copies). |
| S9 | SHOULD FIX | Submission status is unclear: the user said on 2026-11-06 that the NVARC work was submitted, while PLAN and METADATA say it was not. The repo returns 404 without login. |
| N1-N5 | MINOR | Seed wording for run B, LB sd, two more pitfalls, reproduction LB. |
| W1-W6 | WORTH ADDING | Three-run table, B/C pools, units other methods reach, LB table, selector tests, LB/eval agreement. |

## MUST FIX before arXiv

### M1. Run C is a third run of the R1 program; the paper says only R1 and R2 are the same program

Paper lines:
- l.253: "Most of them are versions of the pipeline with code changes between them ... The exception is \R{1} and \R{2}, versions 1 and 2 of one notebook."
- l.54 (abstract): "two runs of the same program with identical seeds answer 49 and 45."
- l.76: "two runs of the same program with identical seeds that differ by 4 correct test inputs".
- l.542: "The two runs differ by 4 correct units, so single-run numbers carry run-to-run noise of that order."

Evidence (recomputed for this review):
- I extracted the code cells of `provided notebooks/arc-2026d-with-codex_v25.ipynb` and diffed them with trailing whitespace and blank lines normalised.
  - `arc_solver.py`, `arc_loader.py` and `arc_decoder.py` match R1 with 0 differing lines.
  - Against R2 the only difference is `MAX_BEAM_CANDIDATES` 5 vs 10, the cap the paper shows never binds.
- `starter.py` differs from R1 in one hunk only: the `DEBUG_PUZZLES` list, which is commented out in R1 and active in C. C still trained all 120 tasks (findings file header; `Queued 120` behaviour as in the v3 audit item 5 was not separately confirmed for C, since C has no log).
- Seeds: all 120 `eval_seed` values printed by C's CHECK 1 equal those in the R1 log. All 118 parsed `train_seed` keys also match. My parse keyed on seed value, so seed collisions drop 2 keys; that is a parsing limit, not a mismatch.
- Results of the three runs:

| run | units | task-norm |
|---|---|---|
| R2 | 45 | 33.50 |
| R1 | 49 | 35.83 |
| C | 51 | 37.67 |
| shared by all three | 41 | 31.00 |
| union of the three | 57 | 41.67 |

- 5 of C's correct units are in neither R1 nor R2: `4a21e3da_0`, `a395ee82_0`, `b5ca7ac4_0`, `b6f77b65_2`, `fc7cae8d_0`.
- 4 of those 5 are outside the R1 + R2 pool union as well (all except `b5ca7ac4_0`).

Why it matters:
- A reader with our public data can find this. The noise is 6 units and 4.17 points, against the paper's 4 units and 2.33 points.
- The spread of the full runs in Table 1 (33.50-37.83) is almost entirely covered by this one program. So code changes among the full runs show no clear effect.
- The union of same-program runs (41.67) already exceeds either single-run oracle (41.17, 40.50).
- The limitation at l.541 ("how many is open") gains a data point: three same-program runs put at least 65 distinct units in a pool or a submission (61 + 4).

Proposed correction:
- l.253: "Three of the six runs (\R{1}, \R{2}, C) use the same loader, decoder and solver and the same seeds; \R{2} differs only in the non-binding cap. They are three runs of one program, and we use them to measure run-to-run variation."
- Abstract l.54: "three runs of the same program with identical seeds answer 45, 49 and 51." Check the 1,920-character metadata limit (1,900 is used now) and trim elsewhere if needed. `paper/arxiv/METADATA.md` must change with it.
- l.76 and l.472-474: add the three-run figures from the table above.
- l.542: "The three same-program runs span 6 units (4.17 points)."
- Keep the caveat that C has no saved submission file and no run log, only the printed correct-unit set.

### M2. Generation-side variants were run; the paper says they were not

Paper lines:
- l.541: "We did not test more views or samples, a lower probability threshold, stochastic sampling, a diversity term in the search, longer test-time training or a larger model."
- l.540: "All runs use the same 4B model, the same test-time training recipe and the same thresholded search".
- l.85: "All runs in this study use the same model and the same solver code family."
- l.56 (abstract): "list variants we did not measure: more samples, diversity-promoting search and other models."

Evidence (saved `arc_solver.py` of every version in the paper folder, 29 distinct files):

| notebook | versions | threshold | test views | TTT colour permutations | LB (slug level) |
|---|---|---|---|---|---|
| `2025-winning-solution-updated-for-2026` | v1 (`0805c056`) | -ln 0.15 | `augment(n=3)`, 24 views | n=24, 192 samples | 23.75-26.25, mean 24.86, 3 subs |
| same | v2, v3 | -ln 0.15 | n=2 | n=24 | same row |
| `arc-2026d-8ab47d` | v1 (`5f57d251`) | -ln 0.2 | n=2 | n=24 | 28.06-32.22, mean 29.54, 3 subs |
| `arc-2026d-8ab47d-7beb3e` | v1 | -ln 0.2 | n=2 | n=24 | none (0 tasks trained) |
| reproduction notebooks, the six full runs | all | -ln 0.2 | n=2 | n=16 | reference |

Further details:
- `2025-winning-solution...` also has a 700 s search budget instead of 540 s.
- `arc-2026d` v7 and v8 have other beam caps (10, 8). The family copy has a 2000 s task cap and a 1000 s search budget.
- `fork-of-arc-2026d-32-22lb-updated-v2` saved an `arc_dsl_sidecar.py` rule module (LB 27.22). I did not confirm that the run calls it.

What is still true, and should be said precisely:
- All six full runs use threshold 0.2, 16 views and 128 TTT samples.
- The variants ran only 0-4 eval tasks in their saved versions (`data/version_log_summary.csv`), so no pool or coverage was measured for them.
- A diversity term, stochastic sampling and a larger model were indeed not run inside this pipeline.
- The LB evidence is weak: 3 submissions each, noise of about 5 points on one version, and only slug-level mapping. The saved versions may not be the scored ones.

The README rule (no "more samples would not help") is still respected. These LB figures must not be turned into that claim either.

Proposed correction:
- l.85: "All six full runs ...".
- l.540: "All six full runs use the same 4B model, the same test-time training recipe and the same thresholded search. A few other notebooks changed these settings; see below."
- l.541: "No full-run or pool measurement exists for a generation-side variant. Three of our notebooks did change generation settings (threshold 0.15, 24 views, 24 colour permutations in test-time training), but their saved runs cover at most 4 evaluation tasks. Their only full evidence is leaderboard scores (means 24.86 and 29.54 over 3 submissions each), which are too few and too noisy to conclude anything. We did not test a diversity term, stochastic sampling or a larger model in this pipeline."
- The abstract and `kaggle_post_draft.md` need the same narrowing.

### M3. The 142ca369 showcase is contradicted by run C

Paper lines:
- l.394-396: "A single task shows what these near misses look like. For task \code{142ca369} ... No ranking of such candidates can produce the answer."
- l.477: "Submissions and pools together cover 70 units (50.83 points, 42.36\,\%). For the remaining 102 units, no submission and neither logged pool contains the answer."

Evidence:
- Run C, cell 10, CHECK 4: `142ca369  SOLVED  No  7  7  1.07869  2`, meaning answer in the pool at rank 7 under both rankers, not submitted.
- Run C, CHECK 9: `SOLVED_NOT_DUAL 8: ['142ca369', ...]`.
- Independent trace: the commented block in `starter.py` of R1 reads `"142ca369",  # rank 7 / 7` under "Same-rank selection candidates".
- `data/pool_units.json` and `data/union_units.json` contain no 142ca369 unit, so this task is outside the 70 units. Across B's and C's CHECK 4 tables, it is the only in-pool task not already covered by the 70-unit evidence.
- The CHECK 4 row is task level, so I cannot say which of the two test inputs it is.

Note on trust: the paper warns against notebook diagnostics, but the defect in this "SOLVED" column is reading it as "submitted". As a pool-presence flag it is what the v3 audit item 10 describes ("a pool oracle"), and the rank 7 comment supports it.

Why it matters: the sentence is literally about R2's candidates, but the paper uses the task as its example of a generation failure, and the same program generated the answer in another run. It is a selection loss in C and a generation loss in R2.

Proposed correction:
- After l.396 add: "Generation of this task varies between runs of the same program: in run C a correct grid for this task is in the pool at rank 7 (as printed by the run)."
- Or replace the example with one stable across runs. `5961cc34` is a candidate: it is in no pool or submission of ours, and it is unsolved in both B's and C's tables.
- l.477: "... at most 101 units; run C's printed task-level table adds at least one (a test input of 142ca369)." Figure 5's dotted line can stay if the caption says it excludes B's and C's task-level pools.

## SHOULD FIX

### S1. Four full runs are not all submission-only

Paper l.542: "Pool-level statistics come from two runs of one program. The other four full runs provide submission-level data only."

Evidence:
- `README.md` section 3 itself says: "v25 and v3 print puzzle-level counts only (48 and 47 puzzles in pool)."
- Findings section 2: "40 -> 48 puzzles with perfect selection".
- v3 audit section C: pool oracle 47 puzzles, submission right on 43 puzzles.
- I recounted CHECK 4: 47 (B) and 48 (C) tasks, 120 rows each.

Correction: "A and D provide submission-level data only. B and C print task-level pool counts: 43 to 47 and 40 to 48 tasks with at least one correct unit under perfect selection." These two extra runs support the selection-headroom claim, so adding them helps. R2 and R1 have 47 and 45 tasks in the pool (`data/pool_units.json`), a consistent range.

### S2. The LB best is not one of the analysed runs

Paper l.479: "the best NVARC submission of our team scored 32.64\,\% ... Ten submissions of that same notebook version".

Evidence:
- `NVARC family .../README.md`: `arc-2026d-32-22lb-updated-v2` has 35 puzzles trained, a "subset (the 32.64 LB reference)". It is not in Table 1, and its saved submission scores 13.67 of 120 (`data/run_scores.csv`).
- The notebook with pool data, `arc-2026d-with-codex`, scored LB 27.64 / 28.89 / 28.89 (same README; assistant 2026-10-19T10:27:07).
- A reader will assume 32.64 belongs to the analysed pipeline. The paper never gives the LB of R1 and R2.
- Seed nature of the noise: version 331102752's saved solver sets `aug_seed = hash(bk) % 1024**2` (`arc_solver.py` l.458). So its LB spread can include Python hash randomness in scoring as well as GPU non-determinism. R1, R2 and C fixed that seed.
- `arc-2026d-8ab47d` has 3 LB-scored versions but only 1 saved version with outputs. "All saved results" is correct, but it does not cover every scored version.

Correction: add to l.479-480: "That version was never run on all 120 evaluation tasks. The notebook of \R{1} and \R{2} scored 27.64--28.89 over 3 submissions." State that the LB-best version still used a randomised scoring seed.

### S3. Two scales read as one

Paper l.52, l.319-320, l.552 (points of 120) next to l.479 (percent). l.267 warns once: "Leaderboard scores are percentages, not points out of 120."

Evidence of the trap in our own record:
- `CANONICAL_BASELINE_EVAL120_2026-10-03.md`: 'Note that NVARC's printed "33.50" is a task count, not a percentage.'
- User, 2026-10-19T10:29:28: "আমরা জানি যে এই NVARC এর সেলিং ৪০" ("we know this NVARC's ceiling is 40"). The reply at 10:30:19 repeats "ছাদ 40". I could not trace which number was meant, so this is not a correction. It does show that a bare "40" circulates in the team, and the writeup citing this paper should not repeat it without a unit.

Conversions: 33.50 = 27.92 %, 35.83 = 29.86 %, 41.17 = 34.31 %, 40.50 = 33.75 %, 37.83 = 31.53 %, 46.50 = 38.75 %, 50.83 = 42.36 %. The LB 32.64 % equals 39.17 of 120.

Correction: give percentages next to the oracle numbers in the abstract and Table 2, or use percentages throughout and keep points in tables. A casual reader of the abstract would otherwise think the LB best 32.64 is close to R2's score of 33.50.

### S4. Published evidence on NVARC variants is omitted

Paper l.541 and l.554 treat generation changes as untested in general. Our `resources` folder holds direct results, added after draft v1:
- `resources/Kaggle Writeups/writeup_deepseek-v4-1-flash_2026-11-05.md` (Nguyen, Chan Kha Vu, Anastasia, public 13th). "NVARC's Qwen3-4B, as released" LB 28.47. "4,000 more steps on our fixed synthetic data; inference in vLLM": eval 39, LB 35.97. The section "Colors that merge in NVARC's synthetic data" says: "About 95% of the training data was affected."
- `resources/Kaggle Writeups/writeup_755564_darrenamadeusmartin_FlashNext_AllYouNeed_2026-11-04.md`: the vLLM port "allowed a lower DFS threshold (so more candidate answers are explored) and more training epochs, but it did not translate well to the leaderboard."
- NVARC family README: Anil Thomas "35 -> 45" by ensembling an NVARC solution with Qwen3.8-27B (discussion 750546).

Why it matters:
- LB 35.97 is above our best (32.64) and above the single-run oracles of R1 and R2 as percentages (33.75-34.31). Those are eval and LB on different task sets, so this is not a strict comparison. But it is the best available evidence for the paper's own limits: continued base-model training on fixed data moved the score, while per the second writeup a lower threshold and more epochs "did not translate well".
- The recolour bug gives a reported candidate cause for "generation collapse" in the model weights, which every run in this paper uses.
- A reviewer who knows these writeups will ask why they are absent.

Correction: add both to Related Work and to the limitation at l.541, cited as external and not verified by us. Keep l.554's wording ("we did not measure them").

### S5. Run C's label

Paper Table 1 l.306: `arc-2026d-with-codex, "v25"$^\dagger$`.

Evidence:
- User, 2026-11-04T04:45:39: `"D:\arc-2026d-with-codex (25).ipynb"`. The "(25)" looks like a browser download suffix. The user's later shorthand "v25" (05:34:22) stuck.
- The R2-era copy has the same pattern: `31_experiment Notebook_arc-2026d-with-codex (22).ipynb` (`nvarc_eval120_frozen.json`).
- `manifest.json` has only v1 and v2 with outputs for this notebook.
- papermill start time of C: 2026-07-10T19:26:24.

Correction: label it "`arc-2026d-with-codex`, run of 2026-07-10 (notebook copy)". Add a note in README that "v25" is a file name. Ask the user or check Kaggle which version number this was, if any.

### S6. "Built" and "of these notebooks"

Paper l.48: "our team built 34 ARC Prize 2026 notebooks on it"; l.49: "We re-score 41 distinct results of these notebooks"; l.65; post draft: "We built 34 notebooks".

Evidence:
- PLAN (draft v1 audit): "41 = 39 files from 20 notebooks + 2 printed sets (B is not one of the 34 notebooks)".
- A session summary of 2026-10-05 calls the team's score "an NVARC-type public-notebook score". The markdown of run C says "Changes From the 32.22 LB Notebook". The family README says 10 of the 34 have no saved outputs.
- The user's scope order covered one account: 2026-11-04T04:50:17 "শুধু আমার একাউন্ট দেখবে" ("look only at my account"). Nothing was surveyed on other team accounts.

Correction: "our team ran or adapted 34 notebooks", and "41 distinct results of these and one related notebook". Only the user can confirm how many were forks of public notebooks. At l.251 add that this is one team member's account.

### S7. Kaggle post draft inherits the same errors

`paper/kaggle_post_draft.md`:
- "Two runs of the same code with identical seeds solved 49 and 45" (M1).
- "What we did *not* measure: more samples or views ..." (M2).
- "We built 34 notebooks" (S6).
- "from 33.50 to 41.17" with no percent (S3).

Fix these before the post goes out. The post still needs Hamid's approval.

### S8. Superseded statements in files that will become public

The paper says scripts and data are in the repo, so readers will open these:
- `v25_v2_ceiling_findings_2026-11-04.md` section 2: "every re-weighting of beam vs augmentation score was **worse**". PLAN records: "Exp 10 is not a re-weighting of the shipped ranker".
- Same file section 4: "Our 23 saved NVARC runs", superseded by 41. Section 5 still says "+4, +2 per new run", superseded by +6, +1, +3, +1.
- Same file section 1: "puzzles with no valid grid ... v25: 0". C's CHECK 9 prints `NO_OUTPUT 0`. This looks like an artefact of C's fallback writer, since v3 has 9 and R1 and R2 have 11 tasks. I did not confirm it because C has no log.
- `NVARC family .../README.md` and `related data/NVARC family README.md`: "`arc-2026d-with-codex` ... the only full 120 run", superseded by six full runs.

Correction: add a dated "superseded" note at the top of each. Do not rewrite them.

### S9. Submission status

- User, 2026-11-06T08:36:19: "NVARC এর কাজ ভাইয়েরা জমা দিছে" ("the brothers submitted the NVARC work").
- `paper/arxiv/METADATA.md`: "Status: DRAFT. Do not submit." The `TODO(authors)` comment is still in `main.tex` l.39.
- `git log --all` shows no paper commit after 2026-11-05.
- `https://github.com/Foysal348/ARC-AGI-2` returned HTTP 404 without login today, while l.557 gives it as the data location.

Action: find out what was submitted and where. If it is already on arXiv, these fixes need a replacement version, and the writeup should cite that version. The repo must be public before the paper's link works.

## MINOR

- **N1.** l.92 says seeds are "derived from MD5 hashes instead of Python's randomised hash()". Run B still uses `aug_seed = hash(bk) % 1024**2` for scoring (v3 audit item 1, confirmed in the notebook source). Run A is called out at l.95; add B.
- **N2.** l.480 gives the LB range only. Our record has sd 1.65 for this version and sd 0.7-1.5 for five other repeated notebooks (`Qwen3.8-27B LB and run deep analysis (2026-10-16).md`). Adding the sd is fairer than the extreme range alone.
- **N3.** A further pitfall for l.513-526: the R1 log's "SELECTOR BENCHMARK" prints "WINNER: Rank 4 with score 30.50" for a policy 5.33 below the baseline (R1 stdout l.7249 on). It is the same defect as pitfall 3.
- **N4.** Another pitfall candidate: R2's Experiment 33 "Logits Audit" reads no logits ("inferred from candidate similarity"). The paper correctly does not quote it.
- **N5.** The reproduction differences at l.87-95 are stated correctly, but the reproduction LB (28.06-29.31) is not given. It would show that our changes did not clearly beat the base (see W4).

## WORTH ADDING

- **W1. Three-run table (from M1).** It turns a weakness into a stronger noise measurement: same program, same seeds, 45-51 units, 41 shared, 57 union.
- **W2. Pool evidence from B and C (from S1).** Perfect selection adds 4 and 8 tasks, so four runs agree.
- **W3. The missing units are not out of reach for other methods.** From `resources/Puzzle Capability Ledger/oracle_units_by_family_2026_11_02.csv`, recomputed here:
  - of the 102 units with no NVARC evidence, our other methods answered 78 (80 with the ARC Prize run);
  - of the 70 NVARC units, 65 were also answered by our other methods;
  - 5 units are NVARC-only against our other runs: `2b83f449_0`, `4c7dc4dd_1`, `7b80bb43_0`, `b6f77b65_2`, `d35bdbdc_0`.
  - Also usable, quoted from the 2026-10-19 comparison file (not recomputed): R2 against the union of two Qwen3.8-27B runs shares 36 units, with 9 NVARC-only and 56 Qwen-only. Using NVARC's attempt 1 as attempt 2 of Poetiq R5 gives 89 units, 54.72 % (+6.1).
  - This supports "the limit is this pipeline's generation", not task impossibility, and it bears on "or a larger model" at l.541. Keep it clearly separate as a different pipeline, and note that those eval runs lacked the hidden-run coverage limit (comparison file, Caveat).
- **W4. LB by notebook.** `data/lb_by_notebook.csv` is already extracted. A small table (reproduction 28.06-29.31, best version mean 29.36, with-codex 28.47, threshold-0.15 notebook 24.86) would show the measured spread across variants within the tone rule. Few submissions per notebook, so no claim of effect.
- **W5. More selection evidence, as printed.** R1's "RANK 3-5 RESCUE" proof says 4 units at most. R1's selector benchmark shows six alternative attempt-2 policies at -5.33 to -7.33 against the current 35.83.
- **W6. LB and eval agree for this pipeline.** Eval 27.92-31.53 % against LB means around 28-29 (assistant 2026-10-19T10:27:07; comparison file: "LB/eval about 1.0: it finishes"). It is a slug-level comparison, so state it loosely.

Optional, needs the user's go: PLAN step 3 is still open. In the 2026-11-04 messages I found no answer to the question of 06:27:14 about a GPU run. One full 4.75 h run at a lower threshold with candidate logging could replace the limitation, but it is not needed for correctness once M2 is fixed.

## Checked and consistent

- **Title:** matches the user's decision word for word (2026-11-04T06:24:04).
- **Tone:** the user's "৮ মাস হাজার হাজার প্রতিযোগীরা সময় নষ্ট করেছে ... এর থেকে বেশি সলভ সম্ভব ছিলো না" (05:34:22; thousands wasted 8 months, no more solves were possible) stayed out of paper and post, as the README tone rule requires. No opinion wording found.
- **Banned claims:** "no NVARC variant can ever do better" and "more samples would not help" do not appear. l.79 and l.540-541 state the opposite limits. "Generation collapse" comes from the approved title, and l.437 says "consistent with".
- **Experiment 34 not presented as run:** `main.tex` does not mention it. Notebook Experiment 34 is printed proposal code only (R2 stdout l.14999 on). `turbo_dfs_with_diversity` and "diversity penalty" have no hit in `.md`, `.py` or `.txt` files under `experiments/`, `resources/` or `archive/`.
- **E34 outcome:** there is none to add. Repo folders "Experiment 34 - SSM bf16 + block 2048 capacity" and "Experiment 33 - Duck + Poetiq + Direct attempt chain" have no NVARC text hit, and their subjects are engine capacity and an attempt chain. Commit d0098df only happens to bundle them with the paper folder. Notebook Experiments 33 and 34 are not repo experiments E33 and E34; keep the names apart in any writeup.
- **Retractions already handled in the paper:** Exp 10 moved to pitfalls (l.517-518); `a_u` is mean agreement, not closest candidate (l.381-388); fallback units are not crashes (l.519).
- **Six full runs:** this replaces the older "only one full run" and "four runs over 4 h" (assistant 05:22:27). `version_log_summary.csv` agrees for A, R1, R2, D.
- **Partial OOM runs:** `arc-2026d` v7, v8 (39 and 43 units) are inside the 41 and rightly not in Table 1.
- **Run selection:** R1 and R2 were picked by log availability, not score. Both score below C, so the choice does not flatter the claim. I found no selection by outcome. The omitted evidence is M1, M3 and S1.
- **Reproduced numbers:** R1 49 / 35.83, R2 45 / 33.50, C 51 / 37.67, union 64, pools 61, all-evidence 70, the frozen map's rule ("query solved iff Probmul rank in {1,2}", 33.5, 27.92 %), LB rows sum to 63 submissions over 19 notebooks. Run A has the 1200 s cap, as l.95 says.
- **Time:** 4.68-4.80 h per 120 tasks is our measure. The user's challenge of "~4 h for all 240" (2026-10-19T08:37:50) was resolved as Kris Dreemurr's claim, and it is not in the paper.
- **The user's "31 tests":** corrected to 34 at 2026-11-04T04:35. The paper quotes neither.
- **LB as percent over 120 tasks:** matches `Qwen3.8-27B LB and run deep analysis (2026-10-16).md`.

## What I could not check

- Which saved version produced which LB score. `lb_by_notebook.csv` holds script version ids, and the manifest holds vN labels, with no map between them. M2's and S2's LB points are slug level.
- Kaggle-side data: I made no Kaggle calls. The real version of run C is unresolved.
- Run C's raw log, since only the notebook exists. M1 rests on cell source and printed sets. C's environment (image, date of the other runs) is not proven identical to R1.
- Whether `arc_dsl_sidecar.py` was actually called. `starter.py` and `arc_decoder.py` have no reference to it; notebook cells were not checked.
- Authorship and fork origin of the 34 notebooks.
- NVARC runs on team accounts other than foysalemonshanto.
- Transcripts:
  - I scanned 293 `.jsonl` files by keyword on text blocks only. Tool results were not scanned, and my extract truncated long pasted content to 6,000 characters.
  - In the windows that matter (2026-11-04 03:00-08:00 and 14:30-16:00, 2026-11-06 08:30-09:00) I read every user message regardless of keyword.
  - Outside those windows I read keyword hits only, condensed. Pasted GPT audits and subagent hand-backs were sampled, not read in full.
- 224 repo files mention NVARC. Most are logs, tokenizer or build files, reviewed by keyword hit only. I read the notes, writeups and reports quoted above. The E12, E14 and E15 notes were not read beyond keyword lines.
- The paper's other numbers, the bibliography and the LaTeX build were not rechecked; they are outside this lens.
