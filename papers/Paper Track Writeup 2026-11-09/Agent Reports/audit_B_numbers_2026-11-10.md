# Audit B (independent numbers fact-check), 2026-11-10 ~05:20 BDT
Verbatim agent report (nothing edited).

No wrong numbers found. Every count, percentage, p-value and Wilson interval checked matches the source data (recomputed in Python: scipy Fisher exact two-sided; Wilson z = 1.96). 4 wording issues, 1 sourcing caveat.

Discrepancies (none is a wrong number):
1. "These builds are not the submitted ones" contradicts itself: Sec 5.4 says the 3-bit run is the build submitted as 57896498; Table 3 caption and Limitations repeat the blanket claim. Per A.8 the 6-hour 3-bit run is the same notebook (esfersami50/q27-xhigh-e28-k100-eval120-6h) in a 6-hour eval profile; 57896498 is its 12-hour RERUN_PROD profile. Fix: "the 3-bit 6-hour run used the notebook later submitted as 57896498, in a 6-hour eval profile; the W4A16 6-hour run differs from 57851995."
2. Figure numbers differ between Writeup (Fig 1 pipeline, Fig 2 LB history) and PDF (Fig 1 LB history, Fig 2 pipeline); Fig 3 matches.
3. Writeup "75-88 at 600-1,240 tok/s" looser than Table 1 (75.69-87.22, about 680-1,240; fit-the-whale different model).
4. Writeup "easy/medium" vs PDF "usually-solved/sometimes-solved" (source: solid 6,344 and sometimes 23,603 medians of correct attempt-1 tokens).
5. 738,142-token pool measured in the 15-task NB5 FP8 run at the same pin (+1.0 GiB) and #60490; hidden-run pool not in code; add "(measured at the same pin)".

Checked exactly: matched 92 tasks (79/61 vs 79/50; accepted 72/58 vs 76/49; p 0.0812 and 0.0426; Wilson 66.8-85.1 and 52.3-73.1); all 120 tasks 65/104 (52.9-71.2), p 0.0371; 15 matched 24/26 vs 22/28 p 0.2531, 41/54 (63.1-85.4); p range 0.04-0.08 consistent; false verifications 15 = W4A16 2 (b) + 3-bit 13 (a 2, b 9, c 1, d 1), 9 of 13 wrong in <= 8 cells, 7 of 13 every test larger than every demo; coverage tokens 9,062,803 vs 3,939,455 = 2.30x, 419.4 vs 182.3 tok/s, 120 vs 92 started, 107 vs 90 outputs, 65.0 vs 53.75, 63 vs 57 on positions 1-92, 57 + 17 = 74, 44 vs 32 at 3.0M tokens, +15 in last hour, 2,028 preemptions, later attempts 867,304 tokens / 9 rescues, 598,479 after correct first; 15-task 1.44x, 34.5 % vs 18.1 %, 30.5 % vs 45 %; hidden budget 7.22-9.50 M, 30.1-39.6k per task; FP8 vs BF16 13/15 each, 18/20, 86.67 vs 90.00, McNemar 1.0, 214 vs 112 blocks, 9 vs 5 sessions, 169.2 -> 225 tok/s, window 158 -> 172; KV pool 738,142, peak 363,807 = 49.3 %, 3 of 270 queued, 0 preemptions; headroom 107 / 143 (85.69 %) / 161 (95.28 %), 11 outputs in 8 tasks (ledger 162 includes ARC Prize runs, correctly excluded); Domdolus 139 and nisshokuitsuki 151 in 5 h 23 min, LB 83.47 vs 77.22-78.06; companion 116/117 and +4.67/+7.67 match, 18 pages; LB/27B numbers agree; appendix arithmetic (3,553,580,524 B; 120,384; 5,120; 717/715/712; 9/8/8/6) correct.

Not checked (time): ~70 TTT submissions 23.75-32.64; replay 208 vs 177; first check at 55k / 25-31k; early stopping 2.7-4.0 %; ordering 3.7-6.8; int8 draft head in W4A16 6-hour run; 3-bit pin ladder.
