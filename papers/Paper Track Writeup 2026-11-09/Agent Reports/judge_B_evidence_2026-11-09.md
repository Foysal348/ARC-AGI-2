# Judge B (evidence check) report, 2026-11-09 (condensed; all must-fix items and every non-VERIFIED row kept)

Raw data used: `_scratch/e25_gpu_out/...nb5-quality/q27x/phase_Q_FP8_K3|BF16_K3`, `_scratch/e26_gpu_out/...k100-quality/q27x/phase_Q_K100_FP8_K3`, `_scratch/e28_w4_out/...`; gold `papers/1 - .../data/arc-agi_evaluation_solutions.json`.

## Must-fix
1. One definition of "correct": W4 FP8 26 accepted, 24 correct on every output (16b78196 b0 wrong; 247ef758 b0 right q0 wrong q1). "25" = any-output rule. 3-bit 41/54 uses all-outputs rule (any-output: 44/54). Use 24/26 (92 %) vs 41/54 (76 %), or 25/26 vs 44/54. Fix Fig. 3 (make_figs right=[25,41], wrong=[1,13]) and WRITEUP.
2. "8 vs 1 on the same 15 tasks" WRONG: all-outputs rule 6 (3-bit: 16b78196 a1, 247ef758 a1+a2, 2b83f449 a1, b5ca7ac4 a2, f931b4a8 a2) vs 2 (W4); any-output 3 vs 1. "8" only in E28 README l.54, does not reproduce.
3. 6-hour W4A16 run is not the submitted build: 12 workers (not 9), Q-defer, I8 int8 draft head, pin 0.75 GiB -> pool 197 blocks / 679,504 tokens (vs 214 / 738,142), A/B/C scheduler with 120k B2 re-runs, --language-model-only, plain alphabetical order (not ORDER2); running_check FAIL_RUNNING_NE_INFLIGHT (agree 0.20, KV-starved); 182.3 tok/s window; 90/172 outputs. "Runs twice as long" wrong; eval-vs-LB 74 vs 63 / 49.58 vs 54.17 compares a never-submitted eval build with a never-EVAL120 submitted build (only 3-bit side matches: E28 K100 = LB 49.58).
4. "Hard tasks used 90k and more without success" wrong: never group attempt-1 tokens 16b78196 86,756 (accepted wrong), 2b83f449 103,761 (cut), 64efde09 90,477 (accepted correct). 6.3k/32.7k/90.5k are attempt-1 medians of all attempts per group, not correct ones.
5. 27B 34.58 solver (S1-60K) is Poetiq Code-Verify (E10 README l.208): also verifies code on train pairs. "direct reasoning", "Poetiq-style direct solver", "direct / refine" mislabel; reword "typed grid carries no such evidence ... 27B agent could not use this filter" and novelty sentence.
6. 162/172 includes ARC Prize's official 27B run (ledger l.406-407, 419; E37 RESULT l.81-89: 143 ours -> 146 +ARC Prize -> 162 +E37). Ours + E37 without ARC Prize = 161/172, task-norm 95.28, 11 outputs in 8 tasks unsolved. 95.56 is task-normalised; 162/172 = 94.19 % of outputs.
7. Paper 1 qualifier: "no correct candidate above the p > 0.2 threshold".
8. Prior public work: notnitsuj 755321 (LB 75.69, 1,100 tok/s, FP8 KV, "Memory is throughput"), darrenamadeusmartin 755564, YiChiaChen 755801, mikelou1 and andy2709 notes, NI and Domdolus notebooks (both verify code). WRITEUP "no public reports" false; cite for novelty and memory finding.
9. "Attempt 2 rescued one task in 30" = 3-bit 30-task run (20270e3b). In W4 15-task run attempt 2 rescued one output (247ef758#1) where attempt 1 had a train-passing wrong program -> next-step 3 would have lost the observed W4 rescue.
10. "only submits training-checked program output" wrong: 2b83f449 attempt_1 = output of transform2 with train_exact [False, True] (not accepted). Non-passing program outputs are fallback answers.

## Other non-VERIFIED rows
- PLE named "per-layer-embedding" vs "n-gram embedding table": use one name (writeups: n-gram embedding table).
- NVARC-lineage attribution loose (fig legend "mostly"); 32.64 not an analysed run (Paper 1 review).
- Duck: notes call Franzen "author of the Duck fork we vendor"; bib title unsourced; checker/submit addition undocumented.
- "Marlin-style 4-bit MoE kernels": no source.
- "Longer cap reached fewer hidden tasks": simulation only (E10 README l.218: ~208 vs ~177 reach iteration 1).
- First check ~55k: from early E12 runs (36 branches, 49,152 per-response cap); later EVAL120 Duck runs median 24.9-30.7k (E14 Notes/44 l.277, Notes/45 l.128). Contradicts 35-45k budgets.
- "pinned-memory ladder (3.55 GB, then 0.75/0.5/0)" wrong: KV-cache-size ladder (--kv-cache-memory-bytes) RC2 KV + 1.0 GiB = 3,553,580,524 B, then +0.75/+0.5/+0 GiB; <=3 tries so 0 step never reached.
- "QSA prefill-workspace patch": #60490 backport reserves QSA indexer workspace at profile time.
- 225/172 tok/s are from the 15-task NB5 run, not the hidden run.
- Sandbox fail-closed: exact behaviour undocumented.
- NB5 has fixed seed 20261005; NB6 has none; NB5 vs NB6 also differ in B1 accept mode (2 vs "") and fixed list vs ORDER2.
- "3-bit passed train more often per hour": ~29/h (54 in 6,600 s, 30 tasks) vs ~23/h (26 in 4,050 s, 15 tasks); task sets differ.
- FP8 task-norm 86.67 vs BF16 90.00: worth stating.
- 27B agent "rarely reached a passing program": no count.
- "Teacher-forced divergence does not show this": only TF evidence is NB0 for FP8 KV (NO_GO).
- MTP distribution: theory only.
- Budget arithmetic ignores ~13-17 min boot (NB5 boot overhead 781 s, start 1,019 s); NB5 projection 215 tok/s x 41,000 s ~ 8.8M.
- 77.8k vs 60k 13 points: causal claim rests on simulation.
- Ordering 4-7 points: replay simulation on 27B logs (gains 3.7-6.8).
- Caps 49k/77.8k: different notebooks, not a clean sweep.
- "Gap is mostly selection and budget": unsupported (union of 50+ runs, different models).
- NB5 vs NB6 differences understated.
- Fig 2: 54.17 dated 2026-10-31 (Kaggle submission date 2026-10-31T22:22Z = 2026-11-01 BDT; fine in UTC); 22.92 appears in both D27 (10-14, E9 Direct) and A27 (10-24, E15B Duck) - two separate submissions with equal score (check).
- WRITEUP: "385 vs 225" compares 3-bit window average with W4 steady median; like-for-like 385 vs 172 (window means) or 471.6/413.6 vs 225 (medians); 385 is K100Q (21 workers, Q-defer), not the LB build. "subsets (15 and 120 tasks)" omits the 30-task run. "76 %" vs 24/26 = 92 %.
