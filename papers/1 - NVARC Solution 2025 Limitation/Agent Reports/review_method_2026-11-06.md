# Method-accuracy review of Paper 1 (2026-11-06)

Lens: does the paper describe what the code really does, and do the conclusions follow from the mechanism?

STATUS: complete. Nothing in the paper folder was edited except this file. No GPU work, no Kaggle or git push.

Conventions:
- `main.tex` line numbers are from `paper/main.tex`. `paper/arxiv/main.tex`, `paper/arxiv/METADATA.md` and `paper/kaggle_post_draft.md` carry the same text, so any fix must reach them too.
- R1 = `kaggle versions arc-2026d-with-codex/v1/output/`. R2 = `source notebook arc-2026d-with-codex/output/`.
- External sources fetched for this review:
  - NVARC write-up `nvarc_2025.pdf`, README, `ARChitects/run_sft_4b.sh`, `SDG/scripts/build_datasets.py` and `ARC-AGI1/pip-install-unsloth-flash-patch.ipynb` from github.com/1ytic/NVARC;
  - the Unsloth 2025.9.7 wheel from PyPI. This is the version printed in the R1 and R2 logs.
  - All are stored in `D:\ARC AGI 2\_scratch\review_unsloth\`, and were read but never executed.

## Summary

The Methods section is mostly faithful. The equations, Algorithms 1-3, the bound of 4 / 64, the non-binding K cap, the 1600 s and 540 s logic, dual selection = probmul top 2, placeholders and augmentation inversion all match the code.

The serious problems are about what the measurements mean, not about the mechanics:
- an undisclosed leak of the evaluation puzzles into the model's training data;
- a "close to the answer" measure that is about as high for simply copying the input;
- a third replicate of the same program that the paper treats as a code variant;
- a false "deterministic" statement;
- a false "one search procedure" claim;
- "saturates / ceiling / collapse" wording that the data does not establish;
- NVARC described as if our runs were the winning pipeline.

## MUST FIX before arXiv

### M1. The evaluation set is not clean for this model, and the paper does not say so
- **Paper:** L49, L62-65, L151, L544. The set is presented as "the public ARC-AGI-2 evaluation set". The only limitation listed is that the leaderboard uses hidden tasks.
- **Source:**
  - NVARC write-up section 2.1: the authors labelled 29 evaluation puzzles by hand and generated summaries for "the remaining 91 evaluation puzzles", and these summaries feed the synthetic data.
  - Table 1: "NVARC full subset uses both training and evaluation puzzles from ARC-AGI-2". That subset is 54.9 % of the 3.2M samples.
  - Section 3.5: "we 'leaked' the descriptions of these puzzles to our synthetic dataset".
- **Link to this exact model:** `ARChitects/run_sft_4b.sh` has `EXP_NAME="qwen3_4b_grids15_sft139"` and `max_num_steps=12716` at global batch 256, which is 3,255,296 samples. That matches the 3,255,481 total of Table 1. So our model is the one trained on that mixture (inferred from the step count; the script does not name the dataset).
- **Why it matters:** every pool, oracle, top-k and union number is measured on tasks whose descriptions shaped the training data. The numbers may overstate behaviour on unseen tasks.
- **Scope, to state accurately:** the leak is indirect (LLM descriptions turned into synthetic puzzles). I found no evidence that the eval answers were trained on directly. `arc2_evaluation6` (`build_datasets.py` L249-250) is built from the eval puzzles, but the write-up describes validation on "120 evaluation puzzles", so it looks like the validation set. I did not read `sft_mg.yaml`, so this is unconfirmed.
- **Correction:** add a Limitations item and a clause in the Data section, citing NVARC sections 2.1 and 3.5. Example: "The model's synthetic training data were derived in part from descriptions of these 120 evaluation tasks (NVARC, Table 1), so pool coverage here may not transfer to unseen tasks." I did not measure the size of this effect.

### M2. "Close" near misses are about as high as copying the test input
- **Paper:** L53 (abstract), L125 (figure), L388, L552 ("often while the candidates are more than 90 % correct"), L396. Figure 3 uses the label "close".
- **Measured for this review** (test input vs answer, a different shape counts as 0, the same rule as `avg_sim`; unit lists from `data/paper_numbers.json`):

| class | run | units | copy-input mean agreement | units where copy-input alone exceeds 90 % |
|---|---|---|---|---|
| close | R2 | 48 | 88.3 % | 27 |
| close | R1 | 47 | 90.3 % | 26 |
| partial | R2 | 33 | 54.4 % | 1 |
| partial | R1 | 32 | 57.4 % | 1 |
| far | R2 | 22 | 5.8 % | 0 |
| far | R1 | 22 | 10.0 % | 1 |

- **Why it matters:** for over half of the close units, a no-op scores above 90 %. The classes therefore mostly sort tasks by how similar output is to input, not by how close the model came. The Limitations item at L545 does not cover this.
- **Correction:** report the copy-input baseline next to `a_u`. Either replace the class with a change-aware measure (agreement on the cells where the answer differs from the input), or drop the near-miss reading from the abstract, the figure and the conclusion. A change-aware measure needs the candidate grids, which were not saved, so a GPU rerun or a rewording is required. Rewording is the option available now.

### M3. Run C is a third run of the same program, so run-to-run variation is understated
- **Paper:** L252-253 ("The exception is R1 and R2"), L54 (abstract), L76, L472, L542 ("run-to-run noise of that order"), Table 1.
- **Code:** compared with R1, run C (`provided notebooks/arc-2026d-with-codex_v25.ipynb`) has:
  - identical `arc_solver.py`, `arc_decoder.py` and `arc_loader.py` (0 differing lines);
  - `starter.py` differing in the debug-puzzle list, in the lines I inspected (`RUN_FULL_EVAL = True`, as in R1). I did not diff the whole file line by line;
  - 240 of 240 seed lines identical to R1;
  - the same Unsloth 2025.9.7 and Transformers 4.55.4;
  - a different training loss on `271d71e2`: 0.018372 (R1 0.018519, R2 0.018989).
- **Consequence:** there are three replicates, not two:

| run | units | task-norm |
|---|---|---|
| R1 | 49 | 35.83 |
| R2 | 45 | 33.50 |
| C | 51 | 37.67 |

  - The spread is 6 units and 4.17 points, not 4 units and 2.33 points.
  - The union of the three is 57 units (41.67). 41 units are correct in all three.
  - Four units C answered correctly are in neither logged pool (`4a21e3da_0`, `a395ee82_0`, `b6f77b65_2`, `fc7cae8d_0`). So the same program's pool union is at least 65 units, not 61.
  - Strictly, R2 differs from R1 and C in the non-binding K only.
- **Why it matters:** the headroom range and the claim that repeated runs add little both rest on this.
- **Correction:**
  - Rewrite L252-253 and L472 around three replicates.
  - Update L54, L76, L474, L541 and L542.
  - Add a note to Table 1 that C equals R1's code.
  - The label "v25" (L306) is not one of the two saved versions (v1, v2) of that notebook. Say what it is. It may be a file name, a Kaggle version number or a third saved run; `PLAN.md` and the data do not settle this.

### M4. "The search is deterministic" is contradicted by NVARC and by our own runs
- **Paper:** L437.
- **Source:** NVARC write-up section 3.3: the batch DFS "became nondeterministic", which "could play a critical role" because decisions depend on logit values; the fix was not used in the final submission. The three same-seed runs in M3 show different results. Their training losses differ too, so training is non-reproducible as well.
- **Correction:** "The search draws no random samples, but batched inference is numerically non-deterministic (NVARC, section 3.3), and training losses also differ between runs with identical seeds." Use this as the explanation at L253, which currently gives none.

### M5. "One model with one search procedure" and "we did not test" are false for the pooled data
- **Paper:** L79, L540 ("the same test-time training recipe and the same thresholded search"), L541 ("We did not test more views or samples, a lower probability threshold"), L267.
- **Code:** settings across all 53 saved `kaggle versions/*/v*/output/arc_solver.py` files:
  - 47 have threshold 0.2, 16 train colour permutations and 2 view colour permutations.
  - `2025-winning-solution-updated-for-2026` v1: threshold 0.15, n = 24, 3 view colour permutations (24 views).
  - The same notebook, v2 and v3: threshold 0.15, n = 24, 2 view colour permutations.
  - `arc-2026d-8ab47d` v1 and `-7beb3e` v1: threshold 0.2, n = 24.
  - `arc-agi-enhanced-solution` v1: my regex could not parse the values, so that solver is not confirmed as equivalent.
- **Also across the family:** per-task caps are 1200, 1450, 1600 or 2000 s, and K is 5, 8 or 10.
- **Where this reaches the numbers:**
  - Four partial results from these notebooks are among the 41 (`data/run_scores.csv`), with few tasks each: `2025-winning` 1-4 tasks, `8ab47d` 4 tasks.
  - The LB table pools these notebooks: `2025-winning` 23.75-26.25 and `8ab47d` 28.06-32.22 (`data/lb_by_notebook.csv`). These are the ranges for the notebooks' scored versions. I could not map scored version ids to saved vN, so I cannot say which settings each LB value used.
- **What remains true:** all six full runs do use threshold 0.2, n = 16 and 16 views.
- **Correction:** say that the six full runs and the two pool runs share these settings, that a few partial results and LB submissions use threshold 0.15, 24 colour permutations or 24 views, and that no full public-evaluation run or pool measurement exists for those variants.

### M6. "Saturates", "ceiling" and "generation collapse" are not established
- **Paper:** title L36-38, abstract L48-55, L377, L551-554.
- **"Saturates":** no sweep of any resource was run. Saturation means more input stops helping, and nothing here varies the input.
- **"Ceiling" is a property of the settings measured, not of the pipeline:**
  - Pool size is capped by construction by threshold 0.2 and 16 views (eq. 1). The search keeps only outputs with p > 0.2, and the mean is 14.05-14.36 candidates over 16 views, about 0.9 per view.
  - The 14-15 no-output units exist only because of that threshold (`NOT_PROCESSED`).
  - The time budget did not bind: 4.68-4.80 h of 12 h on the evaluation set, no task time-out in R1 or R2, longest call 246.9 s of 540 s. So the ceiling is not a Kaggle time limit on this set. I did not measure runtime on the hidden set, so the spare budget there is unknown.
  - The selection conclusion (L554) inherits this. "Small ceiling" holds at threshold 0.2. A lower threshold would enlarge the pool and could change the selection problem.
- **"Generation collapse":**
  - The term appears only in the title and in abstract item (iii). There is no definition, and the body never uses it.
  - No decline is shown: "collapse" implies a process, but the paper shows a static count.
  - Its two supporting numbers are weak. M2 undermines the 90 % figure. Duplicates are not by themselves a failure, because the rankers are built to reward agreement between views (see S8).
- **Correction:**
  - The README says the user fixed this title, so the decision is the user's.
  - If the title stays: define "candidate-pool ceiling" as "at the measured settings", and define "generation collapse" in the body or remove it from abstract item (iii).
  - State in the conclusion that the ceiling is tied to threshold 0.2, 16 views and 1-epoch TTT, and that it is not the time budget.
  - A safer title would be "Where the NVARC Qwen3-4B Pipeline Loses Test Inputs on ARC-AGI-2: Candidate-Pool Coverage vs Selection".

### M7. NVARC is misdescribed, and credit is incomplete
- **Paper:** L48 ("The pipeline that won ARC Prize 2025 (NVARC)"), L64-65, L531.
- **Source:**
  - NVARC README: three components, namely synthetic data generation, an improved ARChitects solution and improved Tiny Recursive Models (Jolicoeur-Martineau).
  - Write-up abstract: "The core idea of our solution is the novel synthetic dataset".
  - Sections 4.4 and 3.5 discuss a Qwen3 + TRM ensemble and Qwen3 alone. Best score was 27.64 % in the competition, 29.72 % in a 12 h post-deadline variant.
- **What we ran:** only the ARChitects-derived Qwen3-4B path, with changes. Notably, `arc_decoder.py` submits probmul top 2 with a per-grid baseline, while the write-up (section 3.4) names the count-plus-log-likelihood score (kgmon) as the better method.
- **Missing credit:** the ARChitects as the origin of the DFS and rescoring design (L64 cites Franzen et al. only for scoring), NVARC for the batch DFS (section 3.3), TRM, and the Qwen3-4B-Thinking-2507 base.
- **Correction:**
  - Abstract: "the Qwen3-4B component of NVARC, as modified in our notebooks".
  - L64: list the three components and state the portion studied.
  - Credit DFS and rescoring to the ARChitects, and the batch implementation to NVARC.
  - Use "the NVARC-derived pipeline" in the title or define it in the first paragraph.

## SHOULD FIX

### S1. The embeddings and output head are not LoRA-adapted (L152; Figure 1, L112)
`arc_solver.py` L258 lists `embed_tokens` and `lm_head` in `target_modules`. Unsloth 2025.9.7 `models/llama.py` L2484-2503 moves both to `modules_to_save`, so they are trained directly as full copies. Both logs print "Training embed_tokens in mixed precision". Correction: "LoRA on attention and MLP projections; embeddings and output head trained directly."

### S2. The effect of the per-grid baseline is not stated (L89, L170-173)
- `arc_decoder.py` L22-31 sets the baseline per distinct grid. For a grid found once, probmul = 9 + 8b - sum(a_j). So a worse search NLL raises the score with weight 8, and grids are not on a common scale.
- With n equal occurrences it is n + 8(b + 1) - sum(a_j).
- The eq. 2 formula (`eq:probmul`) itself is correct, but a reader cannot see this. It also undoes the comparability NVARC sought in section 3.4.
- Correction: state it in one sentence. The oracle numbers are unaffected. The kgmon comparison is the only ranker swap measured on these pools; the fixed baseline was not measured on them.

### S3. Views also differ in demonstration order (L437)
`arc_loader.py` L266 ends `augment` with `d.shuffle_ex()`. "Differ only by symmetry and colour relabelling" is therefore false. Correction: add "and demonstration order". The same applies to the training samples at L152.

### S4. Colour permutations are independent draws, not a crossed factor (L113, L152, L155, L186, L189)
`arc_loader.py` L180 calls the descriptor per key, so there are 128 and 16 independent permutations. "8 x 2" reads as two shared permutations. Correction: "2 (16) independently drawn colour permutations for each dihedral transform". They permute all ten colours, including 0.

### S5. "Every number is recomputed" conflicts with the quoted numbers (L141, L146, L513)
Figure 1 says "no notebook-printed score used unchecked", and the caption says every number is recomputed. L328, L394-395 and L543 quote values that could not be recomputed. B and C use a printed correct-unit set (L248). L271 and L513 word this correctly. Correction: align the figure and caption with L271.

### S6. The K cap list is incomplete (L162, L209)
The saved code gives K = 5 for A, R1, B, C and D, and 10 for R2. The reproduction notebook has no cap (the slice is commented out). Correction: "10 in R2, 5 in the other full runs, none in the reproduction notebook".

### S7. The list of differences from the reproduction notebook is incomplete, and Run B is not covered (L87-95)
- Diff R2 vs reproduction also shows:
  - `adamw_torch` changed to `adamw_torch_fused`;
  - an attention mask added in `calc_scores`;
  - stricter grid validation;
  - the beam cap;
  - flash SDP settings.
- L92 says MD5 replaces `hash()`, and L95 names only A as the exception. But Run B still uses `hash(bk)` for the scoring augmentation seed (its solver cell; `v3` audit item 1).
- B's solver differs from R1 by 344 changed lines (including comments and formatting), and the diff shows `save_fallback_result` removed. Audit item 13 reports no candidate file for its 9 no-output tasks, so L94 and L164 do not hold for B: it would submit the default `[[0]]`. I inferred this from the diff and the audit, and did not trace B's output.
- Correction: add these, or say "B also differs" with the two facts.

### S8. Duplicates are framed as a defect (L53, L427-438)
Agreement across views is what both rankers reward (`eq:probmul`, `eq:kgmon`), and `known_scores` and the count term rely on it. The averages of 14.0 and 7.3 also include the 14-15 placeholder units, which L433 corrects but the abstract does not. Correction: present the distinct-grid count as a coverage limit, and use the non-placeholder averages in the abstract, or state that the placeholders are included.

### S9. The patched Unsloth inference path is not mentioned (L151, L229-231)
All 34 notebooks load `sorokin/pip-install-unsloth-flash-patch` (`kernel-metadata.json`). Search calls go through Unsloth's cached inference path, patched to call `flash_attn_func`. Correction: mention and cite it, since it is part of the method as run and it is linked to the non-determinism in M4.

### S10. Licences and provenance need a human decision (L557, L65)
- GitHub API reports `license: null` for 1ytic/NVARC, and no LICENSE file exists at main. Reproducing the NVARC-derived solver in the public repo (`arc_solver.py`, `arc_decoder.py`, `arc_loader.py`, notebooks) needs a check, or permission from the authors. I give no legal opinion.
- A text search of the 34 notebooks for fork, public-notebook, licence and credit markers found nothing. R2's first cell refers to "the 32.22 LB Notebook", whose origin I could not determine. Kaggle metadata has no parent field. The team should confirm that "our team built" is accurate for every notebook, and whether any public notebook by someone else was forked.

## MINOR

1. L390: "including scoring". `max_decode_gap` stops at the worker's next line, normally the first `scoring` line, so the figure is usually search only. Harmless: there are 0 `split uneven` lines, so no second call in a batch was missed. Say "longest logged interval after a search call starts".
2. L151: name the base model, Qwen3-4B-Thinking-2507 (`run_sft_4b.sh`).
3. L152: add 10 % warmup and fused AdamW. "Fresh" adapter means the same initial weights restored per task (`arc_solver.py` L311-312, L372).
4. L152: `cut_to_len` drops demonstration pairs when the text exceeds the length limit. This is not mentioned. I did not count how often it happens.
5. Alg. 1 L190: the code also stops at the global `end_time`.
6. `paper/kaggle_post_draft.md` repeats the "two runs of the same code" and ">90 % cell-correct" claims. `NUMBERS.md` needs new rows for the changes above.

## Verified as correct

- Model path `sorokin/qwen3_4b_grids15_sft139`, bfloat16, loaded through Unsloth.
- LoRA r 256, alpha 32, rsLoRA, lr 5e-5, cosine schedule, batch 1, 1 epoch, max length 8192.
- 8 x 16 = 128 samples, 128 steps (log line for `271d71e2`). Test inputs are not in the TTT text (`arc_loader.py` `get`, with no replies loaded).
- 16 views in 4 batches of 4, with the batches of different test inputs interleaved. There were no uneven-length splits in R1 or R2.
- 1600 s check before each batch, counted from before training. 540 s per search call.
- `turbo_dfs`: threshold -ln 0.2, EOS handling, `m > 1` condition, ordering, K cap; Algorithm 2 matches.
- Eq. 1: outputs are prefix-free, so at most 4 per view and 64 per test input. K never binds, because at most 4 tokens can exceed conditional probability 0.2.
- KV cache: `turbo_dfs` passes the cache with no clone, which looked like a possible bug. It is safe: Unsloth's path writes to slot `seq_len` of a paged buffer and returns views, and the buffer is reset on the next full forward (`llama.py` L462, L1072). Checked by reading, not by running.
- Eq. 2 and 3 (`eq:probmul`, `eq:kgmon`) match `getter_full_probmul_3` and `getter_kgmon`. Augmentation NLLs are computed once per grid (`known_scores`).
- Dual selection: attempt 2 is probmul rank 2 whenever at least 2 grids exist; kgmon never changes the pair. `submitted_equals_top2` is True in both runs. The submission is written by `run_dual_selection` (notebook cell 8).
- Placeholders: all-zero grid of test-input shape, scores 999. All are `NOT_PROCESSED`. There are 0 `EMPTY_DECODED_RESULT`, `SCORE_EXCEPTION`, `DFS_EXCEPTION` and `DECODE_EXCEPTION` lines in R1 and R2, so no placeholder is mixed into a real pool.
- Augmentation inversion: `invert_mod` reverses the operations, including the colour permutation.
- Eq. 6 (`eq:agree`) matches notebook cell 30 (`avg_sim`, other shapes count as 0). Algorithm 3 matches `paper_stats.py`.
- R1 vs R2: only `arc_solver.py` L59 differs (K 5 vs 10). Loader and decoder are identical; `starter.py` differs only in commented-out lines.
- Run A: fixed baseline 3, 1200 s cap, `hash()` seed. Reproduction notebook: baseline 3, 1200 s, `hash()`. B and C print with `n_guesses=2`.
- Author names and repo URL of the NVARC citation.

## Could not verify

- The size of the leak's effect (M1), and whether `arc2_evaluation6` was only a validation set (`sft_mg.yaml` not read).
- The model's Kaggle page. Its identity rests on `run_sft_4b.sh`.
- The reproduction notebook against official code: the NVARC repo tree has no solver or decoder file, so "the 2025 release" is only as good as our notebook. A notebook named `ARC-AGI1/002_ivan_arc1.ipynb` exists there and was not read.
- Mapping of scored Kaggle version ids to saved vN (M5).
- Runtime on the hidden set.
- `arc-agi-enhanced-solution` and non-full-run family notebooks were checked by regex only, not read. Run D's decoder was not read line by line.
- The patched Unsloth file was read in part, and Transformers was not read. No search code was executed.
- Competition rules and licence terms (S10). "What the code does" for the PDF figures was checked from the LaTeX source; the rendered PDF was not inspected.
