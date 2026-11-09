# Paper 1 review: consolidated fix list (2026-11-06)

Merged from the four reports in this folder: numbers (N), method (M), record (R), peer (P). IDs refer to each report's own findings. Line numbers are `paper/main.tex`. No fix has been applied yet.

Overall: no wrong measured number. The analysis reproduces. The problems are claims stronger than the data, one wrong mechanism statement, statements contradicted by our own runs, missing disclosure and credit, and publication blockers.

## A. Must fix: claims and facts
1. Title and abstract: "Saturates", "ceiling", "generation collapse" are not established. No sweep was run, the ceiling comes from threshold 0.2 and 16 views, 4.7 h of 12 h shows time did not bind on eval, and "generation collapse" is undefined and shows no decline. Evidence union rises 55-56 -> 61 -> 70, and the abstract omits 61 and 70. (M-M6, P-M1, P-M2, P-M10). Needs the user's decision, see section D.
2. Run C is a third run of the R1 program (same solver, loader, decoder, seeds). Replicates give 45 / 49 / 51 units, 33.50 / 35.83 / 37.67, 41 shared, union 57 (41.67). Noise is 6 units and 4.17 points, not 4 units. Fix l.54, 76, 252-253, 472, 542. (M-M3, R-M1, with N-M1, P-M8: R1 and R2 disagree on 10 submitted units and 11 pool units.)
3. "We did not test a lower threshold / longer TTT" and "all runs use the same recipe" are false for the pooled data. Variants exist: `-ln 0.15`, 24 views, n=24. Correct wording: all six full runs share the settings; the variants have no pool data. Fix l.56, 79, 85, 267, 540, 541. (M-M5, R-M2)
4. Near-miss showcase 142ca369 is in run C's pool at rank 7. l.396 "No ranking..." and l.477 "remaining 102" (at most 101) must change. (R-M3)
5. "The search is deterministic" is wrong, l.437. NVARC section 3.3 says batched DFS is nondeterministic, and our same-seed runs differ. (M-M4, P-M5)
6. "Close, >90 %" is largely a task property. Copying the test input exceeds 90 % on 27 of 48 (R2) and 26 of 47 (R1) close units, mean 88.3 / 90.3 %. Report the baseline, drop "near miss". Fix l.53, 125, 388, 394, 552. (N-M2, M-M2, P-M9)
7. "68 % never get a correct candidate" and the generation/selection split are conditional on the p > 0.2 filter. Fix the definition and abstract. (P-M3)
8. Low diversity is not a failure signal: at most 3 distinct grids gives 42.4 vs 32.8 % (R2), 46.2 vs 33.6 % (R1). Remove the causal reading. (P-M10, M-S8)
9. Undisclosed public-eval leakage. NVARC says descriptions of the 120 eval puzzles fed the synthetic data ("NVARC full" 54.9 % of samples). The link to this model is inferred from the step count, unconfirmed. Add as a threat, with the mitigating eval 27.9-31.5 % vs LB mean 29.36 %. (M-M1, P-M4)
10. "The pipeline that won" and incomplete credit. NVARC has three parts; we ran the modified Qwen3-4B / ARChitects path. Credit ARChitects, NVARC batch DFS, TRM, Qwen3-4B-Thinking-2507. Add NVARC's own scores. (M-M7, P-M6)
11. "58 saved versions / all saved results": 58 is what the API returned from 24 of 34 notebooks; run C and run B are outside that set. Reword, footnote row C. (N-M3)
12. "Every number recomputed by two scripts" contradicts l.271, 513, 543. (N-M4, M-S5)
13. No uncertainty. Add task-clustered bootstrap CIs (P-M7 has values), S(R1)-S(R2) +2.33 [-1.67, 6.50], McNemar p = 0.34. Drop "only" and "small"; headroom is +13 % to +23 % of the score.
14. Unions are oracles that mix code variants, not replicates. Relabel, add Chao1 62.7 and Lincoln-Petersen 61.6. (P-M11)

## B. Must fix: publication
15. Citation error: arXiv:2411.07279 title and authors (8 authors; Zweiger and Pari missing). (P-M12)
16. Missing references: ARC Prize 2025 Technical Report (2601.10904), TRM (2510.04871), Thinking Machines nondeterminism post, public ARC-AGI-2 repo. (P-M13)
17. AI-use disclosure. (P-M14)
18. Named authors; `TODO(authors)` in the package; repo link is 404. (P-M15, R-S9)
19. Publish a minimal licensed repo, never the whole private one. Run gitleaks, including git history. (P-M16)

## C. Should fix (see each report for detail)
- /120 points beside LB percentages; state the unit each time (R-S3).
- LB data: 2026-10-19 snapshot, public 120-task half, best-of-63 biased, lead with the mean and sd 1.65; 32.64 is not an analysed run (N-S2, P-S3, P-S13, R-S2, R-N2).
- "other four full runs provide submission-level data only" is false: B and C print task-level pools (R-S1).
- Relabel "v25": a download suffix, not a Kaggle version (R-S5, P-N7).
- Run B details: `hash(bk)` scoring seed, placeholders (N-S5, M-S7).
- Method detail: `modules_to_save`, probmul per-grid baseline, `shuffle_ex`, independent colour draws, K list, patched Unsloth (M-S1 to S4, S6, S9, N-m3).
- 4.7 h next to the 12-hour limit: 240-task rerun projects to about 9.4 h (P-S1).
- Add threats section, environment table, omitted external NVARC evidence (P-S6, P-S14, R-S4).
- Licences: Kaggle rule 4b forbids publishing Competition Data, use `arcprize/ARC-AGI-2`; `1ytic/NVARC` has no licence; pre-read for the NVARC authors (P-S11, P-S12, M-S10).
- Stale files going public: findings file, README copies, `kaggle_post_draft.md`, `NUMBERS.md` (N-m7, R-S7, R-S8).
- Paper Track judges the submission Writeup (1,500 words); cite this paper as support. Work to 8 Nov (P-S9, P-S10).
- arXiv limit: 2 submissions per calendar month per submitter, rejections count (P-S15).

## D. Decisions for the user
1. Title and main claim:
   - Option 1, retitle, no GPU, e.g. "Decomposing the Score of an NVARC-Derived Pipeline on ARC-AGI-2: ...".
   - Option 2, keep the title and earn it with a GPU sweep (threshold, views, force-scored gold answers). Needs a per-run go and time.
2. Authors and affiliations.
3. Submission status: METADATA says "DRAFT. Do not submit"; where was the work submitted?
4. Public repo contents.

## E. Worth adding
Three-run noise table; B and C pool agreement on selection headroom; ledger overlap (other methods answered 78 of the 102 no-evidence units, 65 of the 70 NVARC units, 5 NVARC-only); LB-by-notebook table; R1 selector benchmark.

## F. Not verified
LaTeX build (no LaTeX on this machine); LB scores against Kaggle; which data trained this exact model; run C's Kaggle version; git history secret scan.
