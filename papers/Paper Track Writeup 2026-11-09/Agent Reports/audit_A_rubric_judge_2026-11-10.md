# Audit A (independent rubric judge), 2026-11-10 ~04:35 BDT
Verbatim agent report.

JUDGE REPORT: Paper Track package (main PDF 16 pp, companion 18 pp, WRITEUP.md 1,142 words, which is under the 1,500 limit)

Statistics recomputed: Fisher 61/79 vs 50/79 p=0.081; 58/72 vs 49/76 p=0.043; 61/79 vs 65/104 p=0.037; 24/26 vs 22/28 p=0.25; all Wilson intervals match; Section 6 token arithmetic correct (7.2-9.5 M, 30-40k per task).

SCORES: Accuracy 2, Universality 3, Progress 2, Theory 3, Completeness 4, Novelty 2; average about 2.7.

TOP 5 TEXT CHANGES
1. Table 4 scope (Sec 5.2 para 2, Contribution 1, Abstract, Writeup item 2): say the 15 classified false verifications come from the two quality runs (15 and 30 tasks); the 18 and 29 false first-attempt verifications of the 6-hour runs were not classified.
2. Which 6-hour builds were submitted: Table 3 caption, Limitations bullet 2, Writeup Limits bullet 3 -> "The 6-hour 3-bit run is the build submitted as 57896498; the 6-hour W4A16 run differs from the submitted W4A16 build".
3. Quantified path (Sec 7 Next steps + Writeup): 3-bit reach (104 verified on 120) at W4A16 precision (77.2 %) -> about 80 correct verified first attempts instead of 65, roughly +12 task-normalised eval points (projection, not measurement); union oracle 161/172 bounds portfolio gain.
4. Writeup overclaims: "A stronger model more often picks the rule a person would pick" -> hypothesis wording; "the run loses more than it gains" -> "a 12-session W4A16 run on a pool sized for 9 had 2,028 preemptions in 6 hours"; Sec 6 "false verification concentrates there" -> "7 of the 13 3-bit errors have a test grid larger than every demonstration (base rate not measured)".
5. Portable procedure (Sec 8 + Writeup): same tasks, same time per task; accept C only if correct answers rise and a one-sided Fisher test of verified precision on shared tasks shows no drop (e.g. p > 0.1); our 3-bit build fails (61/79 vs 50/79).

CONTRADICTIONS AND OVERCLAIMS
1. "classifies all false verifications" vs Table 4 caption (15 from two quality runs); Table 3 implies 18+29 unclassified.
2. Table 3 caption / Limitations "6-hour builds differ from the submitted ones" vs Sec 5.4 "3-bit run is the build submitted as 57896498" and Appendix A.8.
3. Sec 6 "concentrates there" vs Sec 5.2 "we did not measure how often correct verifications share these features".
4. Writeup "A stronger model more often picks the rule a person would pick" states a hypothesis as fact.
5. Writeup "loses more than it gains": no controlled session-count comparison.
6. Writeup negative results "thinking caps above or below 60k" lacks the "separate notebooks, not controlled" caveat.
7. "About 70 submissions ... scored 23.75-32.64" vs companion "63 scored submissions (one 0.00)"; suggest "mostly scored".
8. Figure numbering: Writeup Fig 1 = pipeline, Fig 2 = LB history; PDF Fig 1 = LB history, Fig 2 = pipeline.
9. Verify: KV pool 738,142 tokens vs 214 FP8 blocks in Sec 5.3 (different pin?); union 161/172 vs E37 notes 162/172 and 155; "W4A16 solved 63" vs 53.75 % x 120 = 64.5 (add "fully solved").
10. Rules/credibility: no open-source licence statement for own code (rules: CC0 or MIT-0); companion "scripts available on request"; team naming "Hack2Publish" vs Paper Track team "FOYSAL" (4 members) and Hack2Publish disqualified.

Files reviewed (nothing edited): pdf5/all.txt, p1pdf2/all.txt, WRITEUP.md, PAPER_TRACK_AND_WRITEUP_RULES_2026-11-09.md
