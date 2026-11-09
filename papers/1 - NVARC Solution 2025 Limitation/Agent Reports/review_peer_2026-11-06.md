# Hostile peer review and arXiv readiness: Paper 1 (NVARC saturation)

Reviewer lens: hostile peer reviewer, an ARC expert or the NVARC authors trying to prove the paper wrong; plus arXiv readiness.
Date: 2026-11-06. Line numbers refer to `paper/main.tex` unless stated.
Nothing in the paper folder was edited; this file is the only addition. No GPU work, no submission, no push.
Scratch work: `D:\ARC AGI 2\_scratch\paper1_review\` (`ci.py`, `close.py`, `fit.py`, `pkg2.py`, `clone/`, `arxiv/`, `web/`).

## Verdict

**Not ready for arXiv.** The measurements are sound and reproduce exactly. The problem is what is claimed from them.

- The title says "Saturates", but no input to the pipeline was ever varied.
- The word "ceiling" is contradicted by the paper's own 55 -> 61 -> 70 units.
- NVARC's own write-up contains two facts a reader from that team will use at once: the batch DFS is nondeterministic (the paper says deterministic), and descriptions of all 120 public evaluation puzzles went into the synthetic training data (the paper never mentions this).

All of these can be fixed by editing text and adding small CPU analyses. No GPU run is required, although one is proposed as optional.

## MUST FIX before arXiv

### M1. "Saturates" is not earned: nothing was varied
- **Where:** title l.36-38; abstract l.49 ("where this pipeline stops improving"), l.50 ("locate the limit"); conclusion l.551.
- **Weakness:** saturation means output stops growing as input grows. The paper measures one operating point (threshold 0.2, 16 views, one epoch of TTT, 8 re-scoring augmentations). No threshold, view count, sample count, training length or time budget is varied. L.541 admits this, so the title claims what the limitations section disclaims. The README (section 1) lists "more samples / diversity would not help" as unsupported, yet the title implies it.
- **Fix:** retitle to what was measured, for example *"Decomposing the Score of an NVARC-Derived Pipeline on ARC-AGI-2: Candidate-Pool Coverage, Selection Headroom and Candidate Diversity"*. The title was fixed by the user, so this needs their decision. If "Saturates" stays, it needs a dose-response result: a GPU sweep of threshold or views, which requires an explicit go. Rewrite l.49-50 as "We decompose where test inputs are lost". The title also propagates to `METADATA.md` l.11, and `make_arxiv.py` checks that they match.

### M2. "Candidate-Pool Ceiling" is contradicted by the paper's own numbers
- **Where:** title; abstract l.51, l.55; conclusion l.551; versus l.473-477.
- **Weakness:** one run's pool holds 55-56 units; two pools hold 61; submissions plus pools hold 70 (50.83 points). The best single-run oracle is 41.17, so the evidence union is +9.66 points above it. That is larger than either selection headroom (+4.67, +7.67). The abstract reports the union of submissions (62) but omits 61 and 70, the numbers that hurt the ceiling claim. My task-clustered bootstrap gives a union of two pools minus one pool of +3.33 points (95 % CI 1.00 to 6.33), so the growth is not noise.
- **Fix:** call it "single-run pool coverage", not a ceiling. Put 55-56 -> 61 -> 70 in the abstract. State the correct finding: the single-run pool bounds that run's score; it does not bound the pipeline.

### M3. The generation/selection split depends on the p > 0.2 threshold
- **Where:** l.67, l.155, l.380, l.392, l.396, l.554, abstract l.55.
- **Weakness:** the threshold is itself a filter on model probability. An answer the model rates at p = 0.15 is counted as a "generation" failure, although that is arguably a ranking decision made earlier. So "68 % never receive a correct candidate" holds only conditional on this threshold. The paper never measured the answer's probability under the adapted model. The same paper criticises a notebook sweep for being unable to see pruned answers (l.521), and it has the same blind spot. "The limit is generation" (section heading, l.377) is therefore stronger than the data.
- **Fix:** define "decoded pool at the threshold used" in l.255 and state the conditioning in the abstract. Add to Limitations: "We cannot separate answers pruned by the threshold from answers the model does not favour." Optional, and only with an explicit GPU go: force-score the gold answer under the adapted model for each view. That one measurement would split the 116-117 units into pruned versus disfavoured.

### M4. Public-evaluation leakage is not disclosed
- **Where:** l.238 and l.544. The word "leak" does not appear anywhere in the paper.
- **Weakness:** NVARC's write-up section 3.5 says: "Despite the fact that we 'leaked' the descriptions of these puzzles to our synthetic dataset". Section 2 gives 29 hand-labelled plus 91 LLM-labelled evaluation puzzles (= 120). Table 1 says "NVARC full subset uses both training and evaluation puzzles" (54.9 % of samples). So on the set this paper uses, the model has probably seen synthetic derivatives of the very tasks. Every pool, rank and closeness figure may be optimistic, and an NVARC author will spot this first.
- **Mitigating evidence:** NVARC report a good correlation with the public leaderboard. Our six full runs score 27.9-31.5 % on eval (33.50-37.83 of 120), against a leaderboard mean of 29.36 % for one version. So the inflation may be small, but it has to be argued.
- **Fix:** add a threats-to-validity item with that comparison, citing NVARC section 3.5. Confirm that `qwen3_4b_grids15_sft139` was trained on the full subset; I could not verify this from the model page metadata.

### M5. "The search is deterministic" conflicts with NVARC and with the paper itself
- **Where:** l.437 versus l.253 and l.472.
- **Weakness:** NVARC section 3.3 says of the batched DFS: "It became nondeterministic", cites Thinking Machines Lab, and says the batch-invariant fix was dropped as about 17 % slower. The paper presents identical-seed divergence as a puzzle and gives no cause. It reports differing training losses, which also points to numerical nondeterminism in training.
- **Fix:** l.437: "Given fixed weights and batch composition, the search has no sampling step; batched inference is not deterministic in practice (NVARC section 3.3)". In l.253, state that seeds do not fix outcomes on this stack, cite NVARC and the Thinking Machines post, and say that we did not isolate the source. Also state that the two runs may differ in Kaggle image or GPU assignment: "two runs of one program" (l.253) is not shown to mean one environment.

### M6. "The pipeline that won ARC Prize 2025" misdescribes NVARC's scope
- **Where:** abstract l.48; intro l.64; title; related work l.531.
- **Weakness:** the NVARC repo lists three components: synthetic data generation, an improved ARChitects solution, and improved TRM. The word "ARChitects" appears zero times in the paper, and TRM is never mentioned. What was run is the Qwen3-4B ARChitects branch, modified further by us (l.87-94). None of the five union runs is an unmodified reproduction: l.95 shows only run A keeps some original settings, and the `reproduce-nvarc-2025-results*` notebooks (leaderboard 28.06-29.31) are not in Table 1.
  - NVARC's own best 4B run took 12 h for the rerun; ours extrapolates to about 9.4 h. Settings therefore probably differ, which I could not confirm.
  - In fairness to the paper, NVARC report choosing "Qwen3 4B alone" for one final submission and that TRM gave no gain on 4B (27.22 both ways). So the TRM omission can be defended, but the paper must say so.
- **Fix:** name it "the Qwen3-4B (ARChitects-derived) component of NVARC, as modified in our notebooks" in the abstract and the first paragraph, and put "NVARC-Derived" in the title. Say that TRM was not run and why this is defensible. State NVARC's scores as the anchor: 24.0 % private, 27.64 % public, 29.72 % post-deadline.

### M7. No uncertainty is reported on any headline number
- **Where:** abstract; l.319-327; Table 2; conclusion l.552.
- **Weakness:** 172 units in 120 tasks, with no interval anywhere. My task-clustered bootstrap (20,000 resamples, seed 1):

| quantity | value | 95 % CI |
|---|---|---|
| S, R1 | 35.83 | 26.67 to 45.33 |
| S, R2 | 33.50 | 24.50 to 43.00 |
| O, R1 | 40.50 | 31.00 to 50.33 |
| O, R2 | 41.17 | 31.67 to 50.83 |
| H, R1 | +4.67 | 1.33 to 8.67 |
| H, R2 | +7.67 | 3.17 to 12.83 |
| S(R1) - S(R2) | +2.33 | -1.67 to 6.50 |
| O(R1) - O(R2) | -0.67 | -5.50 to 3.83 |

  These intervals describe sampling of tasks, not run noise. Wilson intervals for pool coverage on units are 26.0-39.9 % (56) and 25.5-39.3 % (55), which ignore clustering. R1 versus R2: exact McNemar p = 0.34 on submissions (7 vs 3) and p = 1.0 on pools (6 vs 5). In other words, the two runs are statistically indistinguishable.
- **"Only" is spin:** +7.67 is +22.9 % over 33.50, and +4.67 is +13.0 %. Against a noise level of a few points, a hostile reader will call this substantial, not "small". The H interval's upper end (12.83) exceeds a third of the base score.
- **Fix:** add the table above (a CPU job; scripts in scratch), drop "only" at l.51-52 and l.552, and change the heading "Selection headroom is small" (l.315) to give the numbers, for example "Selection headroom: 13-23 % of the score".

### M8. Run-to-run noise is understated
- **Where:** l.542 ("differ by 4 correct units, so single-run numbers carry run-to-run noise of that order"); also l.76.
- **Weakness:** 4 is the net difference. The runs disagree on 10 units in 10 tasks (7 + 3), and on 11 pool units (6 + 5). Net difference hides the churn, and churn is the relevant quantity for a decomposition by unit.
- **Fix:** report the symmetric difference (10 of 52 submitted-right units in the union), with a binomial sd of about 3.2 units on the net.

### M9. "Close" (> 90 %) is largely a property of the tasks, not of the candidates
- **Where:** abstract l.53; l.388; "near misses" l.394; conclusion l.552; Fig 3 legend.
- **Weakness:** ARC outputs often resemble inputs. I computed the trivial copy-the-test-input baseline: it exceeds 90 % cell agreement on 39 of 172 units, 27 of which are in neither pool. Of the "close" units, 27 of 48 (R2) and 26 of 47 (R1) are also above 90 % for copying the input. So more than half of the headline "close" class is met by a solver that does nothing. The candidates do beat copy on most of these units (copy matches or beats them on only 6 in each run), so the signal is not zero, but the abstract's framing misleads. An ARC expert will raise this immediately. Also, l.394 calls 142ca369 a near miss with a 72.84 % example, which is not "close" by the paper's own cut, and it uses the closest candidate while the classes use the mean.
- **Fix:** report agreement relative to the copy baseline (for example improvement over copy, or agreement on the cells that change). Drop "near miss". State in the abstract that 27 and 26 of these units also exceed 90 % under copying.

### M10. "Generation collapse" is undefined, and the diversity story is unsupported
- **Where:** title; abstract l.53; l.437-439; conclusion l.552.
- **Weakness:** nothing collapses; there is no process and no comparison. The term is defined nowhere. A duplicate ratio of 0.38 across symmetric views can equally be read as agreement, which is what the rankers exploit. Using the paper's own counts, low diversity is associated with the answer being present, not absent:
  - R2: units with at most 3 distinct grids have the answer in the pool in 14 of 33 (42.4 %), others 41 of 125 (32.8 %).
  - R1: 12 of 26 (46.2 %), others 44 of 131 (33.6 %). Both exclude placeholder units.
  This is the opposite of what the framing implies. Note also that no causal claim was tested, though l.533 says diverse decoding "targets the low diversity we observe".
- **Fix:** replace the term with "low candidate diversity", define it (eq. 8), and report this association. Remove the implication that diversity is the limiter.

### M11. Union figures are unattainable under the rules, and the runs are not replicates
- **Where:** l.463-477; Fig 5; section title l.460; abstract l.54; conclusion l.553.
- **Weakness:** the union is an oracle over runs; a real entry still submits 2 attempts. L.252 says the runs have code changes, so the curve mixes code changes with noise, and "repeated runs" (l.460, l.553) is true only for R1 and R2. The order A, R1, R2, B, C is arbitrary, and the sixth full run (D) is left out of the curve, though it adds 0 units. This curve should not be read as a saturation result.
- **Fix:** label it "oracle union over code variants". Keep the 120-order band as the main display. Add a capture-recapture lower bound as the honest saturation estimate: Chao1 = 62.7 from frequencies {1:4, 2:12, 3:2, 4:7, 5:37}; Lincoln-Petersen on the pools = 61.6. Both are lower bounds, because task heterogeneity biases them low, and the 70-unit evidence union already exceeds the pool estimate.

### M12. Citation error in the bibliography
- **Where:** `refs.bib` l.22-27; l.64, l.531.
- **Weakness:** arXiv:2411.07279 is currently titled "The Surprising Effectiveness of Test-Time Training for **Few-Shot Learning**" with **eight** authors: Akyürek, Damani, Zweiger, Qiu, Guo, Pari, Kim, Andreas. The bib has the old title and six authors, omitting Adam Zweiger and Jyothish Pari. arXiv's policy makes authors responsible for incorrect references.
- **Fix:** use the current title and all eight authors.

### M13. Missing obvious references, and required content
- **Where:** `refs.bib`; related work l.531-533.
- **Add:**
  - ARC Prize 2025 Technical Report, arXiv:2601.10904 (Chollet, Knoop, Kamradt, Landers), which `PLAN.md` used but the paper does not cite;
  - TRM, "Less is More: Recursive Reasoning with Tiny Networks", arXiv:2510.04871 (Jolicoeur-Martineau);
  - the Thinking Machines Lab nondeterminism post that NVARC cites;
  - the public ARC-AGI-2 repository as the data source.
  Also say explicitly that `franzen2025poe` is the ARChitects 2024 paper. The ARC Prize paper page asks that prior work note similarities and differences; three sentences do not. NVARC's "1-2 points variability" should be cited beside the 5.28 range at l.479.

### M14. No AI-use disclosure
- **Where:** nowhere in `main.tex` (0 matches); `README.md` shows the draft was prepared by an AI agent.
- **Weakness:** arXiv's moderation page requires authors to report significant use of text-to-text generative AI, and bars AI tools from authorship. Signing authors carry full responsibility.
- **Fix:** add an acknowledgements note, and have the human authors verify the text and references.

### M15. Authors, repository and draft residue
- **Where:** l.39-40; l.557; `paper/arxiv/main.tex` l.34.
- **Weakness:**
  - "Hack2Publish team" is not an acceptable sole author line. arXiv wants named authors; the format is "Hack2Publish Collaboration: Author One, Author Two", with the full list in the paper.
  - The code link returned 404 today (checked 2026-11-06). The paper states the data "are available", which is currently false.
  - The tarball's `main.tex` still holds the `TODO(authors)` comment. arXiv publishes source, so a `--draft` package must not be uploaded.
- **Fix:** see the human checklist below.

### M16. Do not publish the whole private repository to satisfy the data link
- **Where:** l.557; `METADATA.md` open item 2.
- **Weakness:** making `Foysal348/ARC-AGI-2` public exposes every experiment and all history, including personal emails in commit metadata. There is no `LICENSE` at the repo root, so a public repo grants readers no reuse rights. The link path contains spaces. A URL with no version pin is also not an archival record.
- **Fix:** publish a dedicated minimal repository (the paper folder only, 39 MB as tracked), with a licence, a tagged release and ideally a Zenodo DOI; cite the tag. Exclude what the licence check below flags.

## SHOULD FIX

| # | Where | Weakness | Fix |
|---|---|---|---|
| S1 | l.230-231 | 4.68-4.80 h sits beside "12-hour limit" for 120 tasks. Readers will infer 7 h spare and say the budget was left unused. The rerun is 240 tasks (Kaggle data page, NVARC section 1), so a run projects to about 9.4 h of 12 h. | State the 240-task rerun and the projected budget, and label it a projection. Say the budget was not the binding limit in R1 and R2 (no time-outs, l.390), and that this is a design choice of the settings. |
| S2 | l.246 | "All 18 distinct values" are only per-notebook min and max (`distinct_min_max_values` in `paper_stats.py`), not all 63 scores; the csv holds 19 distinct values including 0.0. The fit itself is robust: in my check 720 is the smallest N that fits, and 120/240/360/480 do not. But "one to three test inputs" conflicts with Kaggle's data page, which says one, with a small number having two. | Say "the minimum and maximum per notebook". Note that 100/480 does not fit, so at least some tasks have 3, or the denominators differ, and flag this as an inference. Cite that the public leaderboard covers half of the 240 tasks. |
| S3 | l.479, l.553 | Mean and range of 10 only; no sd, and raw scores are not in the repo. 32.64 is the maximum over 63 submissions (a selection effect), and 10 resubmissions of a version are not independent of the notebook choice. | Add the 10 raw values and sd to `data/`. Lead with mean 29.36 rather than best 32.64. Say that best-of-N is biased upward. |
| S4 | l.328, l.517-518 | Run D's selector gain (+1.00) is one comparison among several, with no noise estimate, and is smaller than R1-R2 noise. Same for Exp 10 and kgmon top 2. | State that none exceeds run noise; these are not independent tests. |
| S5 | l.327 | "Changing the selector moves the score little" rests on one alternative, within a pool that is already filtered by the rankers' own model. | Limit the claim to the kgmon swap. Add that selection and generation are coupled: a larger pool makes selection harder, so H at this pool does not bound H at a bigger pool. |
| S6 | l.536-546 | The section is named "Limitations" and omits: leakage (M4), threshold dependence (M3), code-version mix, environment drift, the 2-attempt limit on unions, Kaggle image and package versions, and the fact that hidden results rest on scores alone. | Rename it "Threats to validity" and add these items. |
| S7 | l.90 vs NVARC 3.4 | NVARC's post-deadline `score_agg` is the kgmon form. So the shipped NVARC rule is the one the notebooks replaced, while l.90 presents it as an internal detail. | Say so, and compare the top-2 rule against NVARC's rule explicitly. |
| S8 | l.65, l.48 | "Our team built 34 notebooks"; I found no credit or fork notice in any notebook (0 matches), yet slugs such as `32-22lb-updated`, `fork-of-...` and hash-suffixed copies suggest some lineage. I could not establish whether any began as another participant's public notebook. | An author must confirm the origin; credit any public source. |
| S9 | Paper track | Kaggle's Paper Track evaluates a Writeup that "documents your solution" for ARC-AGI-2. This paper analyses a baseline, not the team's best entry (54+ by project records). Accuracy, Completeness and Novelty would score low. The rules: writeup up to 1,500 words, cover image required, public notebook, submission ID. A PDF may go in the Public Project Link. | Cite Paper 1 as supporting evidence and do not use it as the track entry. Team must match the ARC-AGI-2 team. |
| S10 | Deadline | Kaggle lists the Paper Track deadline as 2026-11-09 23:59 UTC; an arcprize.org overview snippet from web search says November 8. | Treat 8 November as the working date. |
| S11 | Data licence | Kaggle Rule 4a allows any use and the data licence is Apache 2.0, but Rule 4b forbids publishing Competition Data to non-participants. The bundled files match `arcprize/ARC-AGI-2` (Apache-2.0): all 120 IDs, and 10 of 10 sampled tasks identical. | Do not ship Kaggle's JSON. Have `score_all.py` download from the GitHub repo at a pinned commit, and cite that as the source. |
| S12 | Third-party code | `1ytic/NVARC` has no LICENSE file (GitHub API: none; file 404). `kaggle versions/` and `NVARC family/` contain solver code derived from it. Rule 6b allows public sharing only if it does not violate third-party IP. The model itself is Apache 2.0 (verified on Kaggle). | Publish logs and submissions, and keep solver sources out of the public release or ask the authors. Cite the source instead. Contact Sorokin and Puget for a courtesy pre-read as well: this is the highest-value single step against public error. |
| S13 | l.267, l.479 | The paper gives no private-leaderboard or hidden-task content, which is correct. But "public leaderboard (hidden tasks)" omits that it is the 120-task half. | Add it. |
| S14 | Reproducibility | Scripts reproduce the analysis, but no Kaggle image, package versions or notebook links are given, and candidate grids were not saved. GPU results cannot be regenerated. | Add an environment table and say plainly: analysis reproducible, runs not. |
| S15 | arXiv pacing | Since 2026-10-01 arXiv limits each submitter to 2 submissions per calendar month, and rejections count. This matters for a 7-paper series. | Have different submitters, and do not risk a rejection on this one. |
| S16 | `METADATA.md` l.72 | The endorsement note predates the current rules. | Update it: since 21 January 2026 an institutional email alone is not enough (secondary sources; the official page says automatic endorsement needs a claimed paper and an institutional email). Plan on a personal endorsement. |

## MINOR

| # | Where | Weakness | Fix |
|---|---|---|---|
| N1 | l.68 | "actually has" is rhetorical. | Delete "actually". |
| N2 | l.377 | "The limit is generation", and 23 uses of "only", read as argument. | Use neutral headings. |
| N3 | l.70, `refs.bib` l.57 | pass@k credited to Codex, which popularised rather than originated it. | Say "as used in". |
| N4 | `refs.bib` | arXiv ids sit in `journal` under `@article`, and some have published versions. | Use `@misc` with `eprint`, and venue details where they exist. |
| N5 | `refs.bib` l.95 vs l.10 | "Greg" versus "Gregory" Kamradt. | Keep the Kaggle citation as printed or unify. |
| N6 | `refs.bib` l.73 | "Ramprasaath" differs from arXiv's metadata ("Ramprasath"). | Keep it; the arXiv listing appears to be the misspelling, from memory. |
| N7 | l.306 | Run C is labelled "v25" but Kaggle has only v1 and v2 of that notebook, per the README. | Explain the label. |
| N8 | l.158 | `eq:bound` is unused. | Reference it at l.433 or drop the label. |
| N9 | l.41 | Empty date; PDF title and author metadata are blank. | Set `pdftitle` and `pdfauthor`. |
| N10 | Git | `.DS_Store` is tracked at the root, in `papers/` and in the paper folder; none is ignored. | `git rm --cached` and ignore them. |

## Repo hygiene: `04efc47`

- **Does `settings.json` contain anything sensitive? No.** Path: `.claude/settings.json`. It has one key, `permissions`, with one allow entry; a pattern scan found no credential, token or home-path match. No value is reproduced here.
- The commit also changed `papers/1 - NVARC Solution 2025 Limitation/.DS_Store` (see N10).
- `.DS_Store` files can leak local folder names. I did not decode them.

Secret scan of the wider repo: 0 hits for known token formats at HEAD; limits under "Could not verify".

## Already strong

- **Reproducibility of the analysis.** A clean `git archive` of tracked files ran `score_all.py` and `paper_stats.py` to exit 0 on CPU, and regenerated data and figure tables match with no differences after CR normalisation. This is unusual and worth keeping.
- **Headline counts verified independently.** My own scoring against submissions gives 49/35.83, 45/33.50, pools 56/40.50 and 55/41.17, with submitted subset of pool in both runs.
- **The core separation** of right, in-pool-unsubmitted and absent units is exhaustive and exact.
- **Honest statements exist already:** l.79, l.541, and "We do not claim that no variant can do better". Most fixes extend them to the title and abstract.
- **Section 5 (pitfalls)** is the most original content and needs no change in my reading.
- **Compliance:** only public-leaderboard scores appear. No private score or hidden-task content. The public evaluation set is stated clearly (l.238, l.544). Model licence Apache 2.0.
- **Citations:** of the 11 arXiv entries, checked in the arXiv API, all IDs are real and correctly attributed, and 10 match on title and authors; the one error is M12. NVARC attribution (Sorokin, Puget) is right against the repo and PDF, and 1st place in 2025 is confirmed. The Kaggle page exists. The Unsloth URL was not fetched.
- **Package mechanics (static):** tarball equals `paper/arxiv/`, 14 of 14 citation keys resolve, no unresolved labels, figure tables present, pure ASCII, abstract in `METADATA.md` 1,900 characters (limit 1,920 verified), TeX Live 2025 default verified, 14-page PDF matches the Comments line.
- **Style:** no em dashes in `main.tex`, `main.pdf` or the Kaggle post draft. Categories cs.LG plus cs.AI suit the content. CC BY 4.0 matches the competition's winner licence type.
- **Type of paper:** an empirical analysis with measurements, so the opinion-piece exclusion should not apply once the title and framing are neutral. arXiv's content-type page lists opinion pieces as typically not accepted; it does not name CS position papers.

## Human steps to submit

1. Decide the title (M1) and apply MUST items. Rerun `paper_stats.py` and update `NUMBERS.md`.
2. Name authors, affiliations and contact; name the submitter, who remembers the 2-per-month limit.
3. Add the AI-use note. Every author reads and approves the paper.
4. Create the minimal public repo with a licence and tag; update l.557; check it opens without login.
5. Obtain endorsement in the cs domain.
6. Run `python -B scripts/make_arxiv.py` without `--draft` on a machine with TeX Live and read the log.
7. Upload, choose the licence, and check arXiv's PDF preview. arXiv submission is a human action.
8. Optional but valuable: send Sorokin and Puget a pre-read.

## Could not verify

- **LaTeX build:** no `pdflatex`, `latexmk` or `tectonic` on this machine. I did not install one, so all package checks are static. Overfull boxes and bibliography rendering are **not** verified; the "0 overfull" claim in `PLAN.md` is unconfirmed by me. `main.log` is not kept. The PDF was built by pdfTeX 1.40.29.
- **Whole-repo secret scan, partial:** `git grep` over all tracked text files at HEAD for Kaggle, Hugging Face, GitHub and AWS token formats and Kaggle key JSON found 0 files. A broader Python scan (generic `api_key=`-style patterns, tracked emails, filenames) did not finish, and git history was not scanned. Run a history scanner such as gitleaks before any repository goes public.
- **Kaggle citation** authors: the competition's citation block was not returned by the API; `PLAN.md` says it was taken from the page.
- **Which NVARC data** trained `qwen3_4b_grids15_sft139`, and which submission scored 24.0 % privately: not stated in what I read. NVARC's "Evaluation score 30" (Table 2) has unclear units, so I did not compare it with ours.
- **Whether NVARC's public notebook matches our p > 0.2 search settings:** I did not read `sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue`.
- **Notebook origin (S8)** and whether Kaggle notebooks are public.
- **Endorsement post of 21 January 2026:** read through secondary sources and the official help page, not the post itself.
- **Environment of R1 and R2:** whether the Kaggle image differed between runs.
- **Published venues (N4)** and the Selvaraju spelling (N6): from memory only.
- **Bootstrap scope:** CIs resample tasks with both runs held fixed; they do not include run noise.

## Sources

- [NVARC repository](https://github.com/1ytic/NVARC) and write-up `nvarc_2025.pdf`
- [ARC Prize 2025 Technical Report](https://arxiv.org/abs/2601.10904)
- [ARC Prize 2026 paper page](https://arcprize.org/competitions/2026/paper)
- [Kaggle ARC-AGI-2 competition](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) and [Paper Track](https://www.kaggle.com/competitions/arc-prize-2026-paper-track)
- [arXiv submission metadata rules](https://info.arxiv.org/help/prep.html)
- [arXiv endorsement](https://info.arxiv.org/help/endorsement.html)
- [arXiv moderation](https://info.arxiv.org/help/moderation/index.html)
- [arXiv content types](https://info.arxiv.org/help/policies/content-types.html)
- [arXiv TeX Live](https://info.arxiv.org/help/faq/texlive.html)
- [arXiv rate-limit policy, 2026-10-01](https://blog.arxiv.org/2026/10/01/updated-rate-limit-policy/)
