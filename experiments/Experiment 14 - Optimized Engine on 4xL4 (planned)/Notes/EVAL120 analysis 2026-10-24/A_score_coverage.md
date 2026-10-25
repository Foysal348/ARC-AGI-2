# EVAL120 G4B Duck runs: score, coverage, references, Poetiq complementarity, hidden projection (2026-10-24)

Read-only analysis. No project file was changed and Kaggle was not touched.

## Inputs

**Runs** (all under `D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Experiment Results/Kaggle Runs/`):

| Label | Folder | Effort | E16 | Duck sha | Run id |
|---|---|---|---|---|---|
| XH-E16 | `EVAL120G4B XHIGH E16 (hosen42, 2026-10-24)` | xhigh | yes | 673618ed | 20261023T235916Z |
| XH | `EVAL120G4B XHIGH noE16 (hosen42, 2026-10-24)` | xhigh | no | 758bddb5 (the E14S adapter) | 20261023T235915Z |
| LOW-E16 | `EVAL120G4B LOW E16 (foysal, 2026-10-24)` | low | yes | 673618ed | 20261023T235802Z |

**Where the data came from:**
- Per-branch data (`branch_result.json` + `transcript.txt`, all 202 / 199 / 203 branches) comes from each folder's `Q27X_E12_DUCK_BUNDLE.zip`.
- Score, configuration and rev6 metrics come from `bundle/e12_run_report.json`.
- The server log is `Q27X_E14A_BUNDLE.zip:logs/sglang_server_try0.log`.

**References:**
- **Poetiq R5:** `Experiment 10 .../Experiment Results/Kaggle Runs/GPU EVAL120 Poetiq R5 (hosen42 q27-xhigh-e10-poetiq-r5-eval120)/q27x/poetiq_run_report.json` (`eval_unit_records`) and `poetiq_iterations.jsonl`.
- **DIRECT ORDER2 86k:** `.../GPU EVAL120 ORDER2 86k DIRECT (hosen42 q27-xhigh-e10-order2)/q27x/run_report.json` (`eval_hit_units`).
- **Solutions:** `Experiment 10 .../Experiment Review/E10_POETIQ_R5_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_solutions.json`. It is still at that path.

**Scripts and outputs** (scratch only):
- Comparison script, run first: `compare_eval_runs.py`. Its full stdout is in Appendix A, and its JSON is `D:/ARC AGI 2/_scratch/eval120_analysis/cmp_A.json`.
- `D:/ARC AGI 2/_scratch/eval120_analysis/scripts/stage_detail.py` → `stage_detail.json`: stage timing and solve source. Its in-flight list is superseded by the next script; see section 9.
- `D:/ARC AGI 2/_scratch/eval120_analysis/scripts/inflight_m.py`: in-flight tokens of deadline-cut requests, and accept times.
- `D:/ARC AGI 2/_scratch/eval120_analysis/scripts/hybrid.py` → `per_task_hits_5runs.json`: Duck vs R5 vs DIRECT, per query.

**Run setup, the same for all 3** (`configuration.e12`, `e12_run.g4b`):
- 120 eval tasks, 172 test queries.
- 12 streams and 12 workers; KV 917,816 tokens.
- Gate 4b:
  - `sec_per_task` 175.0 / 175.0 / 174.8;
  - `window_sec` 21,003 / 20,995 / 20,971;
  - `hidden_solve_end_sec` 42,720;
  - `solve_start_sec` 713 / 730 / 777.
- Branches: E12_BRANCHES 2. Stage-1 first exposure `E12_R6_FIRST_EXPOSURE` 28,000 tokens; b1 first cap 24,576.
- Caps: E12_MAX_GEN_PER_BRANCH 100,000; E12_DUCK_MAX_OUTPUT 12,288; temperature 1.0.
- All three ended cleanly:
  - e14_state is `done (REPORT)`;
  - `e12_run.error` is null;
  - 0 OOM / Traceback / CUDA errors in the server log.

---

## 1. Headline

| | XH-E16 | XH (= E14S agent) | LOW-E16 |
|---|---|---|---|
| Query hits | **46/172** | 41/172 | 31/172 |
| pass1 / pass2 hits | 45 / 1 | 41 / 0 | 31 / 0 |
| attempt_2-only hits | 1 (5961cc34#0) | 0 | 0 |
| Task-norm % (report = recomputed = re-scored vs solutions) | **30.00** | 25.83 | 18.89 |
| Tasks with ≥ 1 hit / fully solved | 37 / 35 | 32 / 30 | 25 / 20 |
| Tasks with a train-verified program | 47 | 46 | 41 |
| ... of those right on test (any / full) | 37 (78.7 %) / 35 | 32 (69.6 %) / 30 | 25 (61.0 %) / 20 |
| Hit tasks WITHOUT a verified program | 0 | 0 | 0 |
| Tokens generated (counted / server-log estimate) | 7,144,122 / 7,219,975 | 7,091,227 / 7,161,583 | 6,606,525 / 6,653,238 |
| Effective tok/s over the window (counted / corrected) | 340.2 / 343.3 | 337.9 / 341.1 | 315.5 / 316.8 |

**Every hit comes from a train-verified program.** In all three runs, 100 % of hit tasks have one, either an accepted submit or a harvested check(). The unverified fallback contributed 0 hits.

So the score is:

`score = (tasks reaching a verified program) × (verify precision)`

| | Tasks verified | Precision | Tasks solved |
|---|---|---|---|
| XH-E16 | 47 | 79 % | 37 |
| XH | 46 | 70 % | 32 |
| LOW-E16 | 41 | 61 % | 25 |

The E16 gain over XH is almost entirely precision: +1 verified task, but 4 fewer wrong ones.

---

## 2. Solved tasks per run (* = partial)

**XH-E16 (37):** 135a2760, 136b0064, 16de56c4, 1818057f, 20270e3b\*, 221dfab4, 2ba387bc, 31f7f899, 36a08778, 38007db0, 3e6067c3, 409aa875, 45a5af55, 4a21e3da, 58490d8a, 58f5dbd5, 5961cc34, 5dbc8537, 67e490f4, 7491f3cf, 7ed72f31\*, 80a900e0, 898e7135, 8b9c3697, 8f215267, 8f3a5a89, 981571dc, 9aaea919, a251c730, a395ee82, aa4ec2a5, bf45cf4b, c4d067a0, cb2d8a2c, db695cfb, dd6b8c4b, eee78d87

**XH (32):** 135a2760, 16de56c4, 1818057f, 1ae2feb7, 247ef758\*, 269e22fb, 2ba387bc, 332f06d7, 38007db0, 3e6067c3, 45a5af55, 4a21e3da, 58f5dbd5, 67e490f4, 6e453dd6, 7c66cb00, 7ed72f31, 8b9c3697, 8f215267, 981571dc, 9aaea919, a395ee82, aa4ec2a5, b0039139\*, b5ca7ac4, bf45cf4b, c4d067a0, db0c5428, db695cfb, dd6b8c4b, de809cff, edb79dae

**LOW-E16 (25):** 16de56c4\*, 1818057f, 1ae2feb7\*, 247ef758\*, 269e22fb, 2ba387bc, 36a08778\*, 38007db0, 3e6067c3, 45a5af55, 4a21e3da, 58490d8a, 58f5dbd5, 6e453dd6, 7c66cb00, 7ed72f31, 8f215267, 8f3a5a89, 9aaea919, a395ee82, aa4ec2a5, b10624e5, bf45cf4b, c4d067a0, dd6b8c4b\*

### Source of each solved task

From `branch_result.json` in the Duck zip: the accepted branch and its stage run, or a harvested verified check() without a submit.

| Source | XH-E16 | XH | LOW-E16 |
|---|---|---|---|
| b0 accepted in stage 1 (≤ the 28k pause) | 11: 1818057f, 221dfab4, 2ba387bc, 31f7f899, 38007db0, 45a5af55, 58490d8a, 7ed72f31, 981571dc, 9aaea919, bf45cf4b | 13: 16de56c4, 1818057f, 2ba387bc, 38007db0, 45a5af55, 67e490f4, 8f215267, 981571dc, 9aaea919, a395ee82, aa4ec2a5, bf45cf4b, db695cfb | 13: 1818057f, 1ae2feb7, 2ba387bc, 38007db0, 45a5af55, 58490d8a, 58f5dbd5, 6e453dd6, 7ed72f31, 8f215267, 9aaea919, aa4ec2a5, bf45cf4b |
| b0 accepted in its stage-2 resume | 14: 136b0064, 16de56c4, 3e6067c3, 4a21e3da, 58f5dbd5, 5961cc34, 67e490f4, 7491f3cf, 80a900e0, 8b9c3697, 8f3a5a89, a251c730, aa4ec2a5, c4d067a0 | 15: 247ef758, 269e22fb, 332f06d7, 4a21e3da, 58f5dbd5, 6e453dd6, 7c66cb00, 7ed72f31, 8b9c3697, b5ca7ac4, c4d067a0, db0c5428, dd6b8c4b, de809cff, edb79dae | 7: 16de56c4, 247ef758, 269e22fb, 36a08778, 3e6067c3, 4a21e3da, 8f3a5a89 |
| Harvested verified check(), no submit | 12: 135a2760, 20270e3b, 36a08778, 409aa875, 5dbc8537, 898e7135, 8f215267, a395ee82, cb2d8a2c, db695cfb, dd6b8c4b, eee78d87 | 4: 135a2760, 1ae2feb7, 3e6067c3, b0039139 | 5: 7c66cb00, a395ee82, b10624e5, c4d067a0, dd6b8c4b |
| b1 (any) | 0 | 0 | 0 |

### Wrong accepts and wrong verified programs (train-overfit)

**b0 accepted, 0 hits:**
- XH-E16: 53fb4810, 6e453dd6, 7b5033c1, 89565ca0, e376de54.
- XH: 221dfab4, 31f7f899, 58490d8a, 5961cc34, 7b5033c1, dbff022c, dfadab01, e376de54.
- LOW-E16: 221dfab4, 31f7f899, 3dc255db, 53fb4810, 7b5033c1, 80a900e0, 97d7923e, d35bdbdc, db695cfb.

**Accepted, partial:**
- XH-E16: 7ed72f31.
- XH: 247ef758.
- LOW-E16: 16de56c4, 1ae2feb7, 247ef758, 36a08778.

**All verified-but-wrong tasks** (`rev6.metrics.verified_programs_per_task` minus hit tasks):

| Run | Count | Tasks |
|---|---|---|
| XH-E16 | 10 | 0934a4d8, 1ae2feb7, 4c3d4a41, 53fb4810, 6e453dd6, 78332cb0, 7b5033c1, 89565ca0, dfadab01, e376de54 |
| XH | 14 | 221dfab4, 31f7f899, 58490d8a, 5961cc34, 71e489b6, 78332cb0, 7b5033c1, 7b80bb43, 80a900e0, d35bdbdc, dbff022c, dfadab01, e376de54, fc7cae8d |
| LOW-E16 | 16 | 0934a4d8, 221dfab4, 31f7f899, 3dc255db, 4c3d4a41, 53fb4810, 67e490f4, 7491f3cf, 7b5033c1, 80a900e0, 97d7923e, a6f40cea, b5ca7ac4, d35bdbdc, db695cfb, e376de54 |

**Overfit flags** (`rev6.metrics.overfit_flags`, counted on unique programs):

| Flag | XH-E16 | XH | LOW-E16 |
|---|---|---|---|
| test_output_equals_input | 2 | 3 | 4 |
| test_shape_outside_train_pattern | 1 | 2 | 1 |
| fixed_offset | 6 | 8 | 4 |
| test_palette_outside_train | — | — | 1 |
| offset_tied_to_train_dims | — | — | 1 |

### Why attempt_2 is about 0 (pass1 45/41/31, pass2 1/0/0)

1. **One candidate per task.** `tasks_with_two_branch_candidates` = 0 in all three runs. Branch 1 never produced a second verified program for a task that branch 0 had verified.
   - `branch1_yield_by_stage2_class` gives b1 "new_distinct_verified": 1 / 0 / 0.
   - b1 accepted nothing: 0 / 0 / 0.
2. **Few multi-prediction tasks.** `distinct_predictions_per_task` > 1 in only a few tasks:
   - XH-E16: 3e6067c3 (2), 5961cc34 (3), 898e7135 (2);
   - XH: 1ae2feb7 (3), 38007db0 (2), 5961cc34 (2), 7b5033c1 (2), 7ed72f31 (2);
   - LOW-E16: 9aaea919 (2), db695cfb (2).
   - These second grids come from the same branch's other verified versions. They are almost never right when the first is wrong. The single success is XH-E16 5961cc34 (3 distinct predictions; attempt_2 hit).
3. **The unverified fallback contributes 0 hits.** Attempt_2 for tasks without a verified program is therefore effectively empty.
4. **Why branch 1 is empty: time.** Branch 1 only starts in stage 2:
   - first b1 start at 10,400 / 10,314 / 10,131 s, i.e. 49.5 / 49.1 / 48.3 % of the window;
   - each b1 got exactly ONE 24,576-token chunk, then paused and was never resumed (all b1 end classes are P or M);
   - b1 consumed 1,846,945 / 1,730,457 / 1,462,022 tokens = 25.9 / 24.4 / 22.1 % of all tokens, for 1 / 0 / 0 new verified programs (`verified_yield.b1.per_100k_for_task` 0.054 / 0 / 0).
   - Its first transform came at a censored median of 24,576, i.e. never inside the chunk. Observed b1 transforms: 1 / 3 / 8.

   So **attempt_2 needs an independent candidate source that finishes inside its budget.** A b1 cut at 24.6k is not one.

---

## 3. Complementarity between the Duck runs

| Pair | Both | Only A | Only B | Union | Jaccard | Sign test on the discordant tasks |
|---|---|---|---|---|---|---|
| XH-E16 vs XH | 21 | 16 | 11 | 48 | 0.438 | 16 vs 11, p = 0.44 (**not significant**) |
| XH-E16 vs LOW-E16 | 19 | 18 | 6 | 43 | 0.442 | 18 vs 6, p = 0.023 |
| XH vs LOW-E16 | 21 | 11 | 4 | 36 | 0.583 | 11 vs 4, p = 0.12 |

- **XH-E16 vs XH:**
  - Only XH-E16: 136b0064, 20270e3b, 221dfab4, 31f7f899, 36a08778, 409aa875, 58490d8a, 5961cc34, 5dbc8537, 7491f3cf, 80a900e0, 898e7135, 8f3a5a89, a251c730, cb2d8a2c, eee78d87.
  - Only XH: 1ae2feb7, 247ef758, 269e22fb, 332f06d7, 6e453dd6, 7c66cb00, b0039139, b5ca7ac4, db0c5428, de809cff, edb79dae.
- **XH-E16 vs LOW-E16:**
  - Only XH-E16: 135a2760, 136b0064, 20270e3b, 221dfab4, 31f7f899, 409aa875, 5961cc34, 5dbc8537, 67e490f4, 7491f3cf, 80a900e0, 898e7135, 8b9c3697, 981571dc, a251c730, cb2d8a2c, db695cfb, eee78d87.
  - Only LOW-E16: 1ae2feb7, 247ef758, 269e22fb, 6e453dd6, 7c66cb00, b10624e5.
- **XH vs LOW-E16:**
  - Only XH: 135a2760, 332f06d7, 67e490f4, 8b9c3697, 981571dc, b0039139, b5ca7ac4, db0c5428, db695cfb, de809cff, edb79dae.
  - Only LOW-E16: 36a08778, 58490d8a, 8f3a5a89, b10624e5.

**Union of the 3 Duck runs:** 62/172 queries, task-norm **39.58 %**, 49 tasks.
- All 3 solve 16: 16de56c4, 1818057f, 2ba387bc, 38007db0, 3e6067c3, 45a5af55, 4a21e3da, 58f5dbd5, 7ed72f31, 8f215267, 9aaea919, a395ee82, aa4ec2a5, bf45cf4b, c4d067a0, dd6b8c4b.
- Only one run:
  - XH-E16 (13): 136b0064, 20270e3b, 221dfab4, 31f7f899, 409aa875, 5961cc34, 5dbc8537, 7491f3cf, 80a900e0, 898e7135, a251c730, cb2d8a2c, eee78d87;
  - XH (6): 332f06d7, b0039139, b5ca7ac4, db0c5428, de809cff, edb79dae;
  - LOW-E16 (1): b10624e5.

**Reading:** two xhigh runs of nearly the same agent overlap at Jaccard 0.44. This is the same "failure is mostly sampling" pattern as D66/L66 (union 50 vs 28 / 34).
- The E16 vs no-E16 difference (+5 queries, +5 tasks, +4.17 task-norm) is inside the run-to-run sampling noise. The paired sign test gives p = 0.44, and the binomial sd of a 120-task task-norm near 26-30 % is about 4.0-4.2 points.
- XH-E16 beats LOW-E16 significantly (p = 0.023).

---

## 4. Coverage

| | XH-E16 | XH | LOW-E16 |
|---|---|---|---|
| Tasks with stage-1 (b0) first exposure | **120/120** | **120/120** | **120/120** |
| Tasks never started | 0 | 0 | 0 |
| b0 stage-1 outcome: paused at ≥ 28k / accepted before 28k | 105 / 15 | 103 / 17 | 101 / 19 |
| b0 reached 28k tokens or finished (script "first exposure budget") | 120/120 | 120/120 | 120/120 |
| Stage-1 tokens per task p10 / p50 / p90 | 26.6k / 32.1k / 42.1k | 24.0k / 31.1k / 46.4k | 22.0k / 30.2k / 40.0k |
| Stage-1 tokens total (share of all) | 3,860,676 (54.0 %) | 3,908,576 (55.1 %) | 3,653,995 (55.3 %) |
| Last task's first exposure (task) | **9,537.5 s = 45.4 %** of window (e3721c99) | 9,592.6 s = 45.7 % (e3721c99) | 9,029.8 s = 43.1 % (e3721c99) |
| First-exposure p50 / p90 | 4,498.6 / 8,591.6 s | 4,255.2 / 8,469.9 s | 4,168.1 / 8,183.2 s |
| Stage 1 complete (last b0 stage-1 run ends) | 10,904.5 s (51.9 %) | 10,937.3 s (52.1 %) | 10,456.7 s (49.9 %) |
| b0 finished (A/C/E), whole run | 30 | 36 | 29 |
| b0 accepted in stage 1 / stage 2 | 15 / 15 | 17 / 19 | 19 / 10 |
| b0 stage-2 resumes (tasks; exactly 1 chunk each) | 88 (1,436,501 tok) | 87 (1,452,194 tok) | 91 (1,490,508 tok) |
| b1 started | 82 (3 with 0 counted tokens) | 79 (3 with 0) | 83 (0 with 0) |
| b1 finished (A/C/E) | **0** | **0** | **0** |
| b1 start times, first → last | 10,400 → 20,668 s | 10,314 → 20,633 s | 10,131 → 20,704 s |
| b1 tokens | 1,846,945 | 1,730,457 | 1,462,022 |
| Tasks with any stage-2 work (b0 resume or b1) | 88 | 87 | 92 |
| End classes | b0: A 30, M 4, P 86; b1: M 3, P 79 | b0: A 36, M 3, P 81; b1: M 4, P 75 | b0: A 29, M 3, P 88; b1: M 1, P 82 |
| Branches cut by the deadline (P+D+M) / paused, never resumed | 172 / 165 | 163 / 156 | 174 / 170 |
| b0 accepts by 25 / 50 / 75 / 100 % of the window | 10 / 18 / 26 / 30 | 13 / 20 / 32 / 36 | 14 / 25 / 27 / 29 |

### Tokens per task

All branches. "Counted" = `generated_tokens`. "Corrected" adds the estimated in-flight tokens of the deadline-cut (class M) requests: in-flight seconds × the measured per-stream rate, capped at 24,576.

| | XH-E16 | XH | LOW-E16 |
|---|---|---|---|
| Counted p10 / p50 / p90 / mean | 26.6k / 66.7k / 87.2k / 59.5k | 24.0k / 68.5k / 85.8k / 59.1k | 22.0k / 59.7k / 81.5k / 55.1k |
| Corrected p10 / p50 / p90 / mean | 26.6k / 67.1k / 87.2k / 60.1k | 24.0k / 69.3k / 85.8k / 59.7k | 22.0k / 60.2k / 81.8k / 55.4k |
| Per-stream rate p50 (requests > 120 s, > 3k tok) | 30.6 tok/s (n 502) | 30.8 (n 498) | 28.3 (n 582) |
| M branches: in-flight s / est. tokens | 195c6913 b0 172/5.3k, b1 705/21.6k; 446ef5d2 b0 341/10.4k, b1 414/12.7k; 5dbc8537 b0 6/0.2k, b1 336/10.3k; d8e07eb2 b0 157/4.8k | 16b78196 b1 720/22.2k; 195c6913 b0 284/8.8k, b1 362/11.2k; 3e6067c3 b0 12/0.4k, b1 655/20.2k; a32d8b75 b0 217/6.7k, b1 9/0.3k | 4c7dc4dd b1 33/0.9k; 4e34c42c b0 631/17.8k; 8e5c0c38 b0 507/14.3k; cbebaa4b b0 166/4.7k |
| In-flight tokens estimated | 65,229 | 69,596 | 37,787 |
| Corrected total vs server-log decode estimate | 7,209,351 vs 7,219,975 (99.85 %) | 7,160,823 vs 7,161,583 (99.99 %) | 6,644,312 vs 6,653,238 (99.87 %) |
| Tokens per task: solved p50 / unsolved p50 | 40.3k / 71.1k | 42.4k / 71.4k | 27.0k / 63.8k |

- The report under-counts by only about 1 %: 7 / 7 / 4 cut requests. The correction closes the gap to the server log almost exactly.
- **Counted tokens per window hour** (h0 → h5):
  - XH-E16: 1.29 / 1.37 / 1.30 / 1.05 / 1.16 / 0.97 M;
  - XH: 1.30 / 1.42 / 1.25 / 1.04 / 1.08 / 0.99 M;
  - LOW-E16: 1.37 / 1.36 / 1.14 / 0.99 / 0.94 / 0.81 M.

  Throughput falls about 25-40 % in the second half (longer contexts plus more prefill).

### Engine (server log, TP0 decode lines in the window)

| | XH-E16 | XH | LOW-E16 |
|---|---|---|---|
| Decode tok/s p50 at ≥ 11 running | 415.3 | 415.9 | 416.6 |
| Decode tok/s p50 at 12, clean | 434.5 | 434.9 | 445.2 |
| Running histogram | 11: 3,571 lines, 12: 564 | 11: 3,503, 12: 610 | 11: 3,236, 12: 419 |
| Accept length p50 | 3.85 | 3.84 | 4.00 |
| Effective / decode p50 | 340/415 = 82 % | 81 % | 76 % |
| Uncached prefill tokens (log) | 6.41 M | 6.58 M | 9.04 M |
| Prefill share = uncached / (prefill + gen) | 0.470 | 0.479 | 0.576 |
| Decode lines right after a prefill | 25.5 % | 24.6 % | 42.0 % |
| Cached % (log / report) | 78.1 / 78.2 | 76.7 / 76.8 | 82.3 / 82.4 |
| KV max / mamba max / queue max | 0.47 / 0.70 / 1 | 0.49 / 0.70 / 1 | 0.52 / 0.70 / 1 |

- LOW is not cheaper per task: stage 1 always pauses at 28k. Its 859 "continue" resumes (vs 524 and 497) cost +2.6 M uncached prefill tokens, and its effective rate is the lowest (315 tok/s).
- The run still sits at 11 running most of the time (3,571 vs 564 lines). The "11 of 12" delayer is still visible.

### Coverage verdict

At the hidden time share:
- **all 120 tasks got stage 1 (≥ 28k or an accept) by 45 % of the window**;
- 73-77 % got one stage-2 b0 chunk;
- **no task got more than one stage-2 chunk**;
- **no branch 1 ever finished.**

Coverage of a first look is solved. Depth on unsolved tasks is what binds: unsolved tasks got a median of about 71k tokens, of which about 25k is the dead b1 chunk.

### E16 behaviour (Section 4 of the script output, Appendix A)

| | XH-E16 | XH | LOW-E16 |
|---|---|---|---|
| E16 early nudges: branches (token p50) | 90 (15,442) | — | 168 (14,160) |
| ... → transform / accepted / task solved | 36 / 19 / 24 | — | 41 / 18 / 20 |
| 24k nudges: branches | 21 | 92 | 9 |
| ... → transform / accepted / solved | 10 / 4 / 7 | 25 / 15 / 15 | — |
| "≥ 24k, no transform, no nudge" | 61 | 55 | 1 |
| Length-cut replies with no call | 98 | 82 | 10 |

---

## 5. Reference points on eval and the LB/eval ratio

| Config | Eval (EVAL120) | Per-task time in that eval run | LB | LB / eval |
|---|---|---|---|---|
| Poetiq R5 (`THINK_BUDGET` **77,824**, not 86k; vLLM FP8; T 1.0; 691 min, 11.65 M tok, 281 tok/s) | 78/172, task-norm **48.61 %** (pass1 77) | Full 12 h for 120 tasks ≈ **2× the hidden share** | 21.39 | **0.44** |
| DIRECT ORDER2 (this one is the **86k**) | 58/172, 36.94 % | About 2× the hidden share (534 min) | (E9 86k: 22.92) | 0.62 |
| 49k DIRECT | ~25.4 % | — | 26.94-27.92 | 1.06-1.10 |
| **S1-60K** (Poetiq, 60k, vLLM) | **No EVAL120 result exists.** `Q27-XHIGH-v1_4_4_E10_S1_60K_EVAL120.ipynb` was built but never run; no run folder exists in Exp 10, 11 or 14. | — | **34.58** | n/a |
| E11-06 P1-60K (SGLang W1, P1 prompts) | EVAL14 only: 5/14 tasks (35.7 %) | — | 29.17 | n/a |
| Duck rev 5 LOW (6 live) | EVAL12 7/12 tasks (not G4B) | — | 17.50 | n/a |
| **E14S = XH agent** (adapter 758bddb5, width 6 + graphs) | **41/172, 25.83 %** (this run, gate 4b) | **= hidden share** | **22.78** | **0.88** |

Proxies for S1-60K on eval:
- R5's iteration-1 records re-capped at 60k: mean 58.3k tokens; only 26/120 close naturally at ≤ 60k (`poetiq_iterations.jsonl`).
- Notes/36: 14.1-14.6 M tokens for 240 tasks.
- Memory simulation: about 208 of 240 hidden tasks reach iteration 1.

### How to read 0.88 vs the old ratios

- The old ratios (1.06 / 0.62 / 0.44) compare a hidden LB with an eval run that had about 2× the hidden seconds per task. They therefore contain the hidden-coverage loss.
- The 0.88 compares with a gate-4b eval that already had the hidden per-task share. It is the residual factor: difficulty difference, start-up differences, and sampling noise.
  - Eval task-norm sd ≈ 4.0 points (binomial, 120 tasks).
  - The public LB is a different 120-task sample: about 4 points of task sampling plus 1.0-1.65 points of run noise.
  - So 22.78 vs 25.83 (−3.05) is about 0.5-0.6 combined sd, and the ratio is 0.88 ± about 0.25.

  **Gate 4b did what it was built for:** the Duck LB/eval is near 1, unlike the 86k and Poetiq methods.

---

## 6. Per task: Poetiq R5 vs Duck (complementarity)

| Pair (task level) | Both | Only R5 | Only Duck | Union |
|---|---|---|---|---|
| R5 vs XH-E16 | 30 | **33** | **7** | 70 |
| R5 vs XH | 26 | 37 | 6 | 69 |
| R5 vs LOW-E16 | 22 | 41 | 3 | 66 |
| R5 vs Duck 3-run union (49) | — | 26 | 12 | 75 |
| R5 vs DIRECT 86k | 39 | 24 | 13 | 76 |
| DIRECT 86k vs XH-E16 | 26 | 26 (DIRECT) | 11 (XH-E16) | 63 |

- **R5 solved, XH-E16 did not (33).** Grouped by R5's iteration records; the 60k column is how each group behaves under an S1-60K cap.

  | R5 group | Tasks | Under a 60k cap |
  |---|---|---|
  | Iteration 1 passed train, natural close ≤ 60k | 9: 1ae2feb7, 291dc1e1, 53fb4810, 7b5033c1, 7c66cb00, 8e5c0c38, b0039139, b99e7126, db0c5428 | same request |
  | Iteration 1 passed, but > 60k or forced at 77.8k | 13: 0934a4d8, 247ef758, 28a6681f, 4c3d4a41, 62593bfd, 65b59efc, 7666fa5d, 88bcf3b4, b5ca7ac4, c7f57c3e, d35bdbdc, e8686506, fc7cae8d | uncertain |
  | Iteration 2 passed | 5: 332f06d7, 6e4f6532, 71e489b6, 7b0280bc, 7b3084d4 | needs a second request |
  | No train pass (best partial program was right) | 6: 142ca369, 2c181942, 8b7bacbf, a6f40cea, d8e07eb2, f931b4a8 | — |

- **XH-E16 solved, R5 did not (7):** 135a2760, 20270e3b, 221dfab4 (R5 verified it but was wrong), 31f7f899, 5dbc8537, 67e490f4, eee78d87.
- **Duck 3-run union solved, R5 did not (12):** 135a2760, 20270e3b, 221dfab4, 269e22fb, 31f7f899, 5dbc8537, 67e490f4, 6e453dd6, b10624e5, de809cff, edb79dae, eee78d87.
- **R5 solved, no Duck run did (26):** 0934a4d8, 142ca369, 28a6681f, 291dc1e1, 2c181942, 4c3d4a41, 53fb4810, 62593bfd, 65b59efc, 6e4f6532, 71e489b6, 7666fa5d, 7b0280bc, 7b3084d4, 7b5033c1, 88bcf3b4, 8b7bacbf, 8e5c0c38, a6f40cea, b99e7126, c7f57c3e, d35bdbdc, d8e07eb2, e8686506, f931b4a8, fc7cae8d.
- **Query level:**
  - R5 has more hits than XH-E16 on 34 tasks: the 33 above, plus 7ed72f31 (R5 2/2, Duck 1/2).
  - XH-E16 has more hits than R5 on 8 tasks: 135a2760, 20270e3b, 221dfab4 (2/2), 31f7f899, 5dbc8537 (2/2), 67e490f4, cb2d8a2c (Duck 2/2 vs R5 1/2), eee78d87.
  - R5 vs XH-E16 on discordant tasks: 33 vs 7, sign test p = 4e-5. With 2× the time, R5 is clearly stronger.
- **Two Duck failure patterns on R5-only tasks:**
  - **Train-overfit accepts.** 53fb4810 and 7b5033c1 were accepted wrong in XH-E16. R5 solved both at iteration 1 with ≤ 60k.
  - **Timeouts at the pause.** Most of the others are P branches with 42-77k tokens and no verified program.

### Union / hybrid potential on EVAL120 (no time constraint, so these are UPPER BOUNDS)

| Combination | Queries | Task-norm % | Tasks |
|---|---|---|---|
| XH-E16 ∪ R5 (OR per query) | 88/172 | **54.44** | 70 |
| a1 = XH-E16 attempt_1, a2 = R5 attempt_1 (a real pass@2 layout) | 87/172 | 54.03 | 69 |
| XH ∪ R5 | 85 | 53.61 | 69 |
| LOW-E16 ∪ R5 | 82 | 51.11 | 66 |
| Three Duck runs ∪ R5 | 94 | 58.61 | 75 |
| XH-E16 ∪ DIRECT 86k | 76 | 47.36 | 63 |
| R5 ∪ DIRECT 86k | 92 | 56.81 | 76 |
| XH-E16 ∪ R5 ∪ DIRECT 86k | 99 | **60.56** | 80 |

- 36 tasks were solved by none of the 5 runs.
- What Duck adds on top of R5: about 7 tasks, +5.8 task-norm (48.61 → 54.44).
- What R5 adds on top of Duck: about 33 tasks, +24.4 task-norm. The average solved R5-only task is worth 0.74 task-norm points: 24.44 / 33.
- These unions need R5's 2×-share run AND the Duck run, about 3× the hidden time. They are **not** reachable in one 12 h hidden run.

---

## 7. Hidden 240-task LB projection (ESTIMATES, clearly not predictions)

**Basis:**
- The eval runs used the hidden seconds per task (gate 4b: 175 s/task = (42,720 − 713) / 240), so the eval coverage is the hidden coverage estimate.
- Hidden difficulty ≈ eval difficulty (user fact).
- Two methods:
  - **A**, uncalibrated: LB ≈ eval task-norm;
  - **B**, calibrated: × 0.88, the factor observed for E14S vs this run's XH.
- ±1 sd ≈ ±4 LB points (eval sampling, plus a different LB task sample, plus 1-1.65 run noise).

| Config | Eval task-norm | Est. LB, A | Est. LB, B (× 0.88) | Note |
|---|---|---|---|---|
| XH-E16 | 30.00 | 30.0 | **26.5** | +3.7 vs E14S under B; the paired difference vs XH has p = 0.44 |
| XH (= E14S) | 25.83 | 25.8 | 22.8 | Observed LB **22.78** |
| LOW-E16 | 18.89 | 18.9 | **16.7** | Duck rev 5 LOW LB was 17.50 |

### Hybrid estimate

Duck XH-E16 stage 1 + b0 stage-2 resumes are kept; **branch 1 is replaced by an S1-60K plain request on Duck-unverified tasks**, as in the Notes/36 design.

**Budget freed:**
- b1 used 1.85 M tokens per 120 eval tasks at the hidden share.
- Each plain request costs ≈ 60.5k tokens (Notes/36: R5 iteration 1 re-capped at 60k is 58.3-58.7k; 60-61k on harder tasks).
- So 25-30 plain requests per 120 tasks, i.e. about 50-60 per 240 hidden. The late ones are lost to the plain admit rule (≥ ~2,070 s before the deadline), which is why the range is below 30.5.

**Yield per plain request on a Duck-unsolved task:**
- R5 solved 33 of XH-E16's 83 unsolved tasks with up to 2 iterations at 77.8k.
- One 60k iteration counts:
  - the 9 natural ≤ 60k at full;
  - 30-70 % of the 13 + 6 that needed > 60k or a partial.
- That gives (9 + 0.3·19) / 83 = 17.7 % to (9 + 0.7·19) / 83 = 26.9 % tasks per request, at about 0.74 points each.

**Result:**
- Gain +3.3 to +6.0 eval points.
- **Hybrid eval estimate ≈ 33-36 % → LB ≈ 33-36 (A) or ≈ 29-32 (B).**

Yield per 100k tokens, on which the "replace b1, not the b0 resumes" rule rests (`verified_yield`):

| Source | New verified per 100k tokens | Basis |
|---|---|---|
| b0 stage 1 | 0.60 | measured |
| b0 stage 2 | ≈ 1.9 | measured; 27 of the 28 stage-2 programs come from b0 |
| b1 | 0.054 | measured |
| Plain | ≈ 0.29-0.44 tasks | estimate |

### Pure Poetiq at the hidden share is MEASURED, not estimated

- S1-60K LB 34.58 (vLLM FP8).
- P1-60K on our SGLang W1 engine with P1 prompts: 29.17. The gap is unexplained (Notes/37 item 1).

**Under method B, every Duck configuration here and the Duck + plain hybrid project at or below S1-60K's 34.58.** Only the uncalibrated top of the hybrid range (36) is above it.

The complementarity data says the larger lever is the other direction: Poetiq as the base and Duck as the second method. On eval:
- R5-only = 33 tasks;
- Duck-only = 7 (XH-E16) or 12 (3-run union).

This is a structural observation, not a projection of a built notebook.

---

## 8. Key findings

1. **Score.** XH-E16 46/172 (30.00 %), XH 41/172 (25.83 %), LOW-E16 31/172 (18.89 %).
   - E16 vs no-E16: +4.2 points, but p = 0.44 (sampling noise; Jaccard 0.44 between the two xhigh runs).
   - xhigh vs low is real (p = 0.023).
2. **All hits come from train-verified programs.** Precision 79 / 70 / 61 %. Wrong accepts number 5 / 8 / 9; verified-but-wrong tasks 10 / 14 / 16.
3. **attempt_2 ≈ 0 because there is never a second independent candidate:**
   - b1 got one 24.6k chunk after 48-50 % of the window;
   - 0 b1 accepts in all runs; 1 / 0 / 0 new verified from 1.46-1.85 M tokens (22-26 % of all tokens).
4. **Coverage:**
   - 120/120 tasks got stage 1 in all runs;
   - the last first exposure came at 43-46 % of the window, and stage 1 completed at 50-52 %;
   - 87-91 b0 got exactly one stage-2 chunk, and 156-170 branches were paused, never resumed.
   - Tokens per task p10 / p50 / p90 ≈ 27k / 67k / 87k (XH-E16, corrected).
   - Effective rate 340 / 338 / 315 tok/s vs 415 decode p50; prefill share 47-58 %.
5. **LB/eval for the gate-4b Duck is 0.88** (22.78 / 25.83). It is not comparable to the old 1.06 / 0.62 / 0.44, because those eval runs had 2× the hidden time.
6. **Poetiq R5 vs Duck.** R5 is stronger with 2× the time: 33 vs 7 discordant tasks. The upper bound XH-E16 ∪ R5 = 54.44 %; adding DIRECT gives 60.56 %.
7. **Projections:**
   - XH-E16 LB ≈ 26.5 (B) to 30 (A);
   - LOW-E16 ≈ 16.7-18.9;
   - the hybrid (b1 → S1 plain) ≈ 29-32 (B) to 33-36 (A);
   - S1-60K measured 34.58.

## 9. Data checks and caveats

- **Script data checks**, all three runs:
  - "3 / 3 / 0 b1 runs started with 0 counted tokens" (these are M branches, corrected above);
  - check() callers, events vs transcripts: 56 vs 54, 55 vs 56, 69 vs 63;
  - `e11.decode_summary.retract_lines` = 1 is a false positive (the 'retraction_policy' server arg).
- `branch_result.t1` is the finalisation time (≈ the deadline) for every branch, so accept times were taken from the last request of each accepted branch.
- **My first in-flight estimate was wrong, and I superseded it.** The first `stage_detail.py` list ("3.97 M in-flight") wrongly counted the idle time of paused branches as generation. `inflight_m.py` restricts it to class-M branches (65k / 70k / 38k), which matches the server log within 0.15 %.
- R5's iteration file has no per-iteration test correctness, so the "60k would keep it" split is by train pass and token count, not by a 60k re-run.
- **ARC-AGI-2 eval vs hidden:** the projections assume equal difficulty (user fact). The 0.88 calibration rests on a single pair (E14S).

---

## Appendix A: full output of compare_eval_runs.py

Command: `python -I compare_eval_runs.py <XH-E16 dir> <XH dir> <LOW-E16 dir> --labels XH-E16,XH,LOW-E16 --out D:/ARC AGI 2/_scratch/eval120_analysis/cmp_A.json`

### Duck eval run comparison (script output)

Generated 2026-10-24 19:09 by compare_eval_runs.py. 'n/a' = field or file missing. Estimates are labelled.

- **XH-E16**: `D:\ARC AGI 2\experiments\Experiment 14 - Optimized Engine on 4xL4 (planned)\Experiment Results\Kaggle Runs\EVAL120G4B XHIGH E16 (hosen42, 2026-10-24)`
- **XH**: `D:\ARC AGI 2\experiments\Experiment 14 - Optimized Engine on 4xL4 (planned)\Experiment Results\Kaggle Runs\EVAL120G4B XHIGH noE16 (hosen42, 2026-10-24)`
- **LOW-E16**: `D:\ARC AGI 2\experiments\Experiment 14 - Optimized Engine on 4xL4 (planned)\Experiment Results\Kaggle Runs\EVAL120G4B LOW E16 (foysal, 2026-10-24)`

## 1. Run setup
|  | XH-E16 | XH | LOW-E16 |
|---|---|---|---|
| profile / run id | E14SEVAL120G4B / 20261023T235916Z | E14SEVAL120G4B / 20261023T235915Z | E14SEVAL120G4B / 20261023T235802Z |
| build variant | E14S | E14S | E14S |
| reasoning effort | xhigh | xhigh | low |
| E16 overlay | yes | no | yes |
| duck sha | 673618ed4cf3 | 758bddb5aeeb | 673618ed4cf3 |
| engine | sglang-0.5.21-int4-dflash2-w1 | sglang-0.5.21-int4-dflash2-w1 | sglang-0.5.21-int4-dflash2-w1 |
| streams / workers | 12 / 12 | 12 / 12 | 12 / 12 |
| KV tokens | 917,816 | 917,816 | 917,816 |
| engine load s | 670.1 | 688 | 717.9 |
| g4b sec/task | 175 | 175 | 174.8 |
| g4b window s | 21,003 | 20,995 | 20,971 |
| solve start s (notebook) | 713 | 730 | 777 |
| deadline in s | 21,004 | 20,995 | 20,972 |
| e14_state terminal | done (REPORT) | done (REPORT) | done (REPORT) |
| e12_run.error | none | none | none |

## 2. Score
|  | XH-E16 | XH | LOW-E16 |
|---|---|---|---|
| query hits | 46/172 | 41/172 | 31/172 |
| pass1 (attempt_1 hits) | 45 | 41 | 31 |
| pass2 (attempt_2 hits) | 1 | 0 | 0 |
| attempt_2-only hits (contribution) | 1 | 0 | 0 |
| task-norm % (report) | 30.00 | 25.83 | 18.89 |
| task-norm % (recomputed) | 30.00 | 25.83 | 18.89 |
| re-scored hits (solutions) | 46/172 | 41/172 | 31/172 |
| tasks with >= 1 hit | 37 | 32 | 25 |
| tasks fully solved | 35 | 30 | 20 |
| hit queries w/o accepted program | 16 | 7 | 5 |

- **XH-E16** solved (37): 135a2760, 136b0064, 16de56c4, 1818057f, 20270e3b*, 221dfab4, 2ba387bc, 31f7f899, 36a08778, 38007db0, 3e6067c3, 409aa875, 45a5af55, 4a21e3da, 58490d8a, 58f5dbd5, 5961cc34, 5dbc8537, 67e490f4, 7491f3cf, 7ed72f31*, 80a900e0, 898e7135, 8b9c3697, 8f215267, 8f3a5a89, 981571dc, 9aaea919, a251c730, a395ee82, aa4ec2a5, bf45cf4b, c4d067a0, cb2d8a2c, db695cfb, dd6b8c4b, eee78d87  (* = partial)
- **XH** solved (32): 135a2760, 16de56c4, 1818057f, 1ae2feb7, 247ef758*, 269e22fb, 2ba387bc, 332f06d7, 38007db0, 3e6067c3, 45a5af55, 4a21e3da, 58f5dbd5, 67e490f4, 6e453dd6, 7c66cb00, 7ed72f31, 8b9c3697, 8f215267, 981571dc, 9aaea919, a395ee82, aa4ec2a5, b0039139*, b5ca7ac4, bf45cf4b, c4d067a0, db0c5428, db695cfb, dd6b8c4b, de809cff, edb79dae  (* = partial)
- **LOW-E16** solved (25): 16de56c4*, 1818057f, 1ae2feb7*, 247ef758*, 269e22fb, 2ba387bc, 36a08778*, 38007db0, 3e6067c3, 45a5af55, 4a21e3da, 58490d8a, 58f5dbd5, 6e453dd6, 7c66cb00, 7ed72f31, 8f215267, 8f3a5a89, 9aaea919, a395ee82, aa4ec2a5, b10624e5, bf45cf4b, c4d067a0, dd6b8c4b*  (* = partial)

## 3. Coverage
|  | XH-E16 | XH | LOW-E16 |
|---|---|---|---|
| tasks | 120 | 120 | 120 |
| b0 started | 120/120 (100.0 %) | 120/120 (100.0 %) | 120/120 (100.0 %) |
| b0 finished (A/C/E) | 30/120 (25.0 %) | 36/120 (30.0 %) | 29/120 (24.2 %) |
| b0 got first exposure budget | 120/120 (100.0 %) | 120/120 (100.0 %) | 120/120 (100.0 %) |
| b1 started | 82/120 (68.3 %) | 79/120 (65.8 %) | 83/120 (69.2 %) |
| b1 started, 0 tokens counted | 3 | 3 | 0 |
| b1 finished (A/C/E) | 0/120 (0.0 %) | 0/120 (0.0 %) | 0/120 (0.0 %) |
| tasks never reached | 0 | 0 | 0 |
| branches started (found / report) | 202 / 202 | 199 / 199 | 203 / 203 |
| tasks with response (report) | 120 | 120 | 120 |
| end classes | b0:A=30, b0:M=4, b0:P=86, b1:M=3, b1:P=79 | b0:A=36, b0:M=3, b0:P=81, b1:M=4, b1:P=75 | b0:A=29, b0:M=3, b0:P=88, b1:M=1, b1:P=82 |
| branches cut by deadline (P+D+M) | 172 | 163 | 174 |
| .. paused never resumed (P) | 165 | 156 | 170 |
| .. in-flight request cut (M) | 7 | 7 | 4 |
| in-flight seconds cut (M) | 13,134.4 | 13,409 | 1,337.2 |
| window start (UTC) | 2026-10-24 00:11:10 | 2026-10-24 00:11:26 | 2026-10-24 00:10:59 |
| window planned / observed s | 21,003 / 20,998.5 | 20,995 / 20,986.2 | 20,971 / 20,939.1 |
| first exposure of last task s | 9,537.5 (e3721c99) | 9,592.6 (e3721c99) | 9,029.8 (e3721c99) |
| .. as share of window | 0.454 | 0.457 | 0.431 |
| exposure p50 / p90 s | 4,498.6 / 8,591.6 | 4,255.2 / 8,469.8 | 4,168.1 / 8,183.2 |
| tokens/task p10/p50/p90 | 26.6k / 66.7k / 87.2k | 24.0k / 68.5k / 85.8k | 22.0k / 59.7k / 81.5k |
| total generated (counted) | 7,144,122 | 7,091,227 | 6,606,525 |
| log-estimated generated (window) | 7,219,975 | 7,161,583 | 6,653,238 |
| counted / log tokens | 0.989 | 0.990 | 0.993 |
| eff tok/s (counted / observed window) | 340.2 | 337.9 | 315.5 |
| eff tok/s (counted / planned window) | 340.1 | 337.8 | 315 |

End classes: A accepted (stopped on an accepted submit), C token cap, E error, P paused never resumed, D deadline cut, M deadline cut mid-request (transcript ends in request_timeout; its tokens are not counted).

- **XH-E16** never reached (0): none
- **XH** never reached (0): none
- **LOW-E16** never reached (0): none

## 4. Behaviour
|  | XH-E16 | XH | LOW-E16 |
|---|---|---|---|
| branches started | 202 | 199 | 203 |
| ever called check() (events) | 56/202 (27.7 %) | 55/199 (27.6 %) | 69/203 (34.0 %) |
| ever called check() (transcripts) | 54/202 (26.7 %) | 56/199 (28.1 %) | 63/203 (31.0 %) |
| defined transform (transcripts) | 57/202 (28.2 %) | 55/199 (27.6 %) | 72/203 (35.5 %) |
| tool calls / branch p50 | 7.0 | 8 | 11 |
| branches with a verified program | 47/202 (23.3 %) | 46/199 (23.1 %) | 41/203 (20.2 %) |
| first transform tokens (all) | obs 30,658 (n=56/202) · naive-cens 28,681 · KM 56,412 · <=20k 14 | obs 28,469 (n=55/199) · naive-cens 24,576 · KM 71,523 · <=20k 17 | obs 24,951 (n=65/203) · naive-cens 24,820 · KM 59,456 · <=20k 26 |
| first transform tokens (b0) | obs 30,682 (n=55/120) · naive-cens 42,159 · KM 55,126 · <=20k 13 | obs 29,278 (n=52/120) · naive-cens 45,699 · KM 71,523 · <=20k 16 | obs 26,398 (n=57/120) · naive-cens 43,574.5 · KM 59,456 · <=20k 19 |
| first transform tokens (b1) | obs 11,036 (n=1/82) · naive-cens 24,576 · KM >39,559 · <=20k 1 | obs 21,241 (n=3/79) · naive-cens 24,576 · KM >24,576 · <=20k 1 | obs 8,859.5 (n=8/83) · naive-cens 16,147 · KM >36,698 · <=20k 7 |
| first check tokens (all) | obs 30,658 (n=56/202) · naive-cens 28,681 · KM 56,412 · <=20k 14 | obs 28,469 (n=55/199) · naive-cens 24,576 · KM 71,523 · <=20k 17 | obs 24,951 (n=69/203) · naive-cens 24,632 · KM 59,456 · <=20k 27 |
| first check tokens (b0) | obs 30,682 (n=55/120) · naive-cens 42,159 · KM 54,157 · <=20k 13 | obs 29,278 (n=52/120) · naive-cens 45,511.5 · KM 71,523 · <=20k 16 | obs 26,630.5 (n=60/120) · naive-cens 43,149.5 · KM 50,877 · <=20k 19 |
| nudged branches | 111 | 92 | 177 |
| nudge 24k | 21 (tok p50 26,701, seen 17, via tool:21) -> transform 10 (+8,310 tok p50), check 10, accepted 4, verified 7, task solved 7 | 92 (tok p50 27,486, seen 78, via tool:90/opener:2) -> transform 25 (+8,084 tok p50), check 25, accepted 15, verified 22, task solved 15 | 9 (tok p50 25,059, seen 6, via tool:9) -> transform 3 (+7,742 tok p50), check 3, accepted 1, verified 2, task solved 1 |
| nudge E16 early | 90 (tok p50 15,442, seen 82, via tool:89/opener:1) -> transform 36 (+16,366.5 tok p50), check 36, accepted 19, verified 31, task solved 24 | 0 | 168 (tok p50 14,159.5, seen 117, via tool:167/opener:1) -> transform 41 (+13,214 tok p50), check 44, accepted 18, verified 28, task solved 20 |
| nudge early (no reason field) | 0 | 0 | 0 |
| >= 24k, no transform, no nudge | 61 | 55 | 1 |
| e16 inspection calls hist | {"0": 63, "1": 19, "2": 3, "3": 3, "4": 6, "5": 8, "6": 11, "7": 13, "8": 16, "9": 8, "10": 5, "11": 3, "12": 5, "13": 6, "14": 5, "15": 4, "16": 5, "17": 5, "18": 1, "19": 5, "20": 1, "21": 1, "22": 3, "23": 2, "29": 1} | n/a | {"1": 4, "2": 10, "3": 10, "4": 9, "5": 28, "6": 14, "7": 15, "8": 15, "9": 6, "10": 10, "11": 5, "12": 10, "13": 12, "14": 6, "15": 6, "16": 6, "17": 10, "18": 2, "19": 4, "20": 3, "21": 4, "22": 4, "23": 3, "24": 1, "25": 1, "27": 2, "29": 1, "32": 1, "44": 1} |
| e16 raw content kept/same/mism/markup | 0/0/0/298 | n/a | 7/0/0/250 |
| raw reasoning kept/same/mism/markup | 1590/0/0/0 | 1569/0/0/0 | 2544/0/0/0 |
| length cut, no call | 98 | 82 | 10 |
| infra errors | 2 | 4 | 3 |

## 5. Engine (server log lines inside the solve window)
|  | XH-E16 | XH | LOW-E16 |
|---|---|---|---|
| log used | `Q27X_E14A_BUNDLE.zip:logs/sglang_server_try0.log` | `Q27X_E14A_BUNDLE.zip:logs/sglang_server_try0.log` | `Q27X_E14A_BUNDLE.zip:logs/sglang_server_try0.log` |
| decode / prefill lines | 4,228 / 2,534 | 4,192 / 2,589 | 3,764 / 3,915 |
| running-req histogram | 7:10 8:38 9:10 10:35 11:3571 12:564 | 7:8 8:3 9:6 10:62 11:3503 12:610 | 4:4 5:14 6:5 7:13 9:1 10:72 11:3236 12:419 |
| accept len p50 (all) | 3.85 | 3.84 | 4.00 |
| accept len p50 (running >= full-1) | 3.85 | 3.84 | 4.00 |
| gen tok/s p50 at full (= streams) | 423.2 (n=564, level 12) | 424.8 (n=610, level 12) | 389.7 (n=419, level 12) |
| gen tok/s p50 at full, clean | 434.5 | 434.9 | 445.2 |
| gen tok/s p50 at running >= full-1 | 415.3 (n=4,135) | 415.9 (n=4,113) | 416.6 (n=3,655) |
| gen tok/s p50 (all lines) | 414.5 | 415.0 | 415.5 |
| decode lines right after a prefill | 0.255 | 0.246 | 0.420 |
| prefill share (uncached prefill / (prefill + gen)) | 0.470 | 0.479 | 0.576 |
| prefill uncached / cached tokens (log) | 6,409,981 / 22,908,009 | 6,584,401 / 21,645,761 | 9,036,943 / 42,088,072 |
| cached % (log prefill) | 78.1 | 76.7 | 82.3 |
| cached % (report usage_all) | 78.2 | 76.8 | 82.4 |
| cached % branch p50 | 65.6 | 65.3 | 84.0 |
| queue max / KV usage max / mamba max | 1 / 0.47 / 0.70 | 1 / 0.49 / 0.70 | 1 / 0.52 / 0.70 |
| OOM lines (all logs / in window) | 0 | 0 | 0 |
| Traceback lines | 0 | 0 | 0 |
| Retract lines | 0 | 0 | 0 |
| CUDA/NCCL error lines | 0 | 0 | 0 |
| other *Error:/*Exception: lines | 0 | 0 | 0 |
| kernel log Tracebacks | 0 | 0 | 0 |

## 6. Per-task table
Cell = query hits/queries · b0 class/k tokens · b1 class/k tokens. '-' = not started, blank = no branch.

| task | XH-E16 | XH | LOW-E16 | solved by |
|---|---|---|---|---|
| 0934a4d8 | 0/1 · P/34k ·   | 0/1 · P/51k · P/17k | 0/1 · P/30k ·   |  |
| 135a2760 | **1/1** · P/55k · P/25k | **1/1** · P/51k ·   | 0/1 · P/47k · P/13k | XH-E16, XH |
| 136b0064 | **1/1** · A/36k · P/25k | 0/1 · P/50k · P/25k | 0/1 · P/60k · P/25k | XH-E16 |
| 13e47133 | 0/2 · P/61k · P/29k | 0/2 · P/48k · P/25k | 0/2 · P/50k · P/15k |  |
| 142ca369 | 0/2 · P/54k · P/25k | 0/2 · P/45k · P/25k | 0/2 · P/49k · P/22k |  |
| 16b78196 | 0/1 · P/52k · P/25k | 0/1 · P/44k · M/0k | 0/1 · P/53k · P/18k |  |
| 16de56c4 | **2/2** · A/43k · P/25k | **2/2** · A/28k ·   | **1/2** · A/43k ·   | XH-E16, XH, LOW-E16 |
| 1818057f | **1/1** · A/17k ·   | **1/1** · A/6k ·   | **1/1** · A/3k ·   | XH-E16, XH, LOW-E16 |
| 195c6913 | 0/2 · M/41k · M/0k | 0/2 · M/50k · M/0k | 0/2 · P/47k · P/12k |  |
| 1ae2feb7 | 0/3 · P/43k · P/12k | **3/3** · P/28k ·   | **2/3** · A/26k ·   | XH, LOW-E16 |
| 20270e3b | **1/2** · P/30k ·   | 0/2 · P/61k · P/25k | 0/2 · P/57k · P/14k | XH-E16 |
| 20a9e565 | 0/2 · P/45k · P/25k | 0/2 · P/55k · P/25k | 0/2 · P/61k · P/16k |  |
| 21897d95 | 0/2 · P/56k · P/15k | 0/2 · P/67k · P/16k | 0/2 · P/46k · P/13k |  |
| 221dfab4 | **2/2** · A/27k ·   | 0/2 · A/21k ·   | 0/2 · A/23k ·   | XH-E16 |
| 247ef758 | 0/2 · P/42k · P/25k | **1/2** · A/38k · P/25k | **1/2** · A/33k ·   | XH, LOW-E16 |
| 269e22fb | 0/2 · P/46k · P/16k | **2/2** · A/35k · P/16k | **2/2** · A/36k · P/13k | XH, LOW-E16 |
| 271d71e2 | 0/1 · P/49k · P/25k | 0/1 · P/46k · P/25k | 0/1 · P/66k · P/26k |  |
| 28a6681f | 0/1 · P/71k · P/32k | 0/1 · P/53k · P/25k | 0/1 · P/53k · P/22k |  |
| 291dc1e1 | 0/1 · P/44k · P/25k | 0/1 · P/47k · P/25k | 0/1 · P/59k · P/20k |  |
| 2b83f449 | 0/1 · P/59k · P/25k | 0/1 · P/72k · P/15k | 0/1 · P/57k · P/22k |  |
| 2ba387bc | **1/1** · A/4k ·   | **1/1** · A/13k ·   | **1/1** · A/5k ·   | XH-E16, XH, LOW-E16 |
| 2c181942 | 0/1 · P/48k · P/24k | 0/1 · P/59k · P/25k | 0/1 · P/52k · P/14k |  |
| 2d0172a1 | 0/2 · P/57k · P/13k | 0/2 · P/49k · P/25k | 0/2 · P/48k · P/13k |  |
| 31f7f899 | **1/1** · A/17k ·   | 0/1 · A/24k ·   | 0/1 · A/12k ·   | XH-E16 |
| 332f06d7 | 0/1 · P/59k · P/25k | **1/1** · A/43k ·   | 0/1 · P/52k · P/22k | XH |
| 35ab12c3 | 0/1 · P/75k · P/25k | 0/1 · P/53k · P/25k | 0/1 · P/59k · P/22k |  |
| 36a08778 | **2/2** · P/43k ·   | 0/2 · P/53k · P/25k | **1/2** · A/42k ·   | XH-E16, LOW-E16 |
| 38007db0 | **2/2** · A/20k ·   | **2/2** · A/21k ·   | **2/2** · A/16k ·   | XH-E16, XH, LOW-E16 |
| 3a25b0d8 | 0/2 · P/37k ·   | 0/2 · P/32k ·   | 0/2 · P/44k · P/19k |  |
| 3dc255db | 0/1 · P/62k · P/40k | 0/1 · P/63k · P/25k | 0/1 · A/55k ·   |  |
| 3e6067c3 | **2/2** · A/42k · P/24k | **2/2** · M/44k · M/0k | **2/2** · A/32k ·   | XH-E16, XH, LOW-E16 |
| 409aa875 | **1/1** · P/57k · P/25k | 0/1 · P/59k · P/24k | 0/1 · P/63k · P/20k | XH-E16 |
| 446ef5d2 | 0/2 · M/41k · M/0k | 0/2 · P/50k ·   | 0/2 · P/45k · P/19k |  |
| 45a5af55 | **1/1** · A/19k ·   | **1/1** · A/28k ·   | **1/1** · A/7k ·   | XH-E16, XH, LOW-E16 |
| 4a21e3da | **2/2** · A/38k ·   | **2/2** · A/42k ·   | **2/2** · A/31k · P/18k | XH-E16, XH, LOW-E16 |
| 4c3d4a41 | 0/2 · P/30k ·   | 0/2 · P/29k ·   | 0/2 · P/30k ·   |  |
| 4c416de3 | 0/1 · P/63k · P/15k | 0/1 · P/46k · P/18k | 0/1 · P/44k · P/13k |  |
| 4c7dc4dd | 0/2 · P/29k ·   | 0/2 · P/29k ·   | 0/2 · P/30k · M/7k |  |
| 4e34c42c | 0/2 · P/33k ·   | 0/2 · P/34k ·   | 0/2 · M/48k · P/19k |  |
| 53fb4810 | 0/1 · A/27k ·   | 0/1 · P/43k · P/21k | 0/1 · A/35k ·   |  |
| 5545f144 | 0/1 · P/58k · P/21k | 0/1 · P/64k · P/17k | 0/1 · P/62k · P/20k |  |
| 581f7754 | 0/2 · P/53k · P/20k | 0/2 · P/58k · P/24k | 0/2 · P/47k · P/17k |  |
| 58490d8a | **1/1** · A/13k ·   | 0/1 · A/51k · P/16k | **1/1** · A/27k ·   | XH-E16, LOW-E16 |
| 58f5dbd5 | **1/1** · A/36k · P/15k | **1/1** · A/34k ·   | **1/1** · A/22k ·   | XH-E16, XH, LOW-E16 |
| 5961cc34 | **1/1** · A/53k ·   | 0/1 · A/43k ·   | 0/1 · P/54k · P/20k | XH-E16 |
| 5dbc8537 | **2/2** · M/37k · M/0k | 0/2 · P/45k ·   | 0/2 · P/42k · P/15k | XH-E16 |
| 62593bfd | 0/2 · P/33k ·   | 0/2 · P/52k · P/25k | 0/2 · P/30k ·   |  |
| 64efde09 | 0/1 · P/58k · P/25k | 0/1 · P/75k · P/25k | 0/1 · P/42k · P/24k |  |
| 65b59efc | 0/2 · P/60k · P/25k | 0/2 · P/61k · P/25k | 0/2 · P/46k · P/13k |  |
| 67e490f4 | **1/1** · A/37k · P/25k | **1/1** · A/25k ·   | 0/1 · P/44k · P/12k | XH-E16, XH |
| 6e453dd6 | 0/1 · A/19k ·   | **1/1** · A/48k · P/25k | **1/1** · A/20k ·   | XH, LOW-E16 |
| 6e4f6532 | 0/2 · P/32k ·   | 0/2 · P/36k ·   | 0/2 · P/43k · P/13k |  |
| 6ffbe589 | 0/1 · P/50k · P/25k | 0/1 · P/51k · P/25k | 0/1 · P/46k · P/18k |  |
| 71e489b6 | 0/2 · P/59k · P/23k | 0/2 · P/43k ·   | 0/2 · P/45k · P/13k |  |
| 7491f3cf | **1/1** · A/34k · P/25k | 0/1 · P/46k · P/25k | 0/1 · P/31k ·   | XH-E16 |
| 7666fa5d | 0/1 · P/45k · P/25k | 0/1 · P/55k · P/22k | 0/1 · P/46k · P/26k |  |
| 78332cb0 | 0/2 · P/64k · P/25k | 0/2 · P/46k · P/25k | 0/2 · P/58k · P/15k |  |
| 7b0280bc | 0/1 · P/52k · P/19k | 0/1 · P/72k · P/25k | 0/1 · P/63k · P/21k |  |
| 7b3084d4 | 0/1 · P/56k · P/25k | 0/1 · P/43k · P/25k | 0/1 · P/71k · P/18k |  |
| 7b5033c1 | 0/1 · A/5k ·   | 0/1 · A/12k ·   | 0/1 · A/5k ·   |  |
| 7b80bb43 | 0/1 · P/53k · P/18k | 0/1 · P/29k ·   | 0/1 · P/46k · P/13k |  |
| 7c66cb00 | 0/1 · P/49k · P/25k | **1/1** · A/36k · P/13k | **1/1** · P/33k ·   | XH, LOW-E16 |
| 7ed72f31 | **1/2** · A/24k ·   | **2/2** · A/52k ·   | **2/2** · A/14k ·   | XH-E16, XH, LOW-E16 |
| 800d221b | 0/1 · P/41k · P/18k | 0/1 · P/46k · P/25k | 0/1 · P/42k · P/18k |  |
| 80a900e0 | **1/1** · A/40k ·   | 0/1 · P/43k ·   | 0/1 · A/27k ·   | XH-E16 |
| 8698868d | 0/1 · P/45k · P/25k | 0/1 · P/62k · P/25k | 0/1 · P/46k · P/18k |  |
| 88bcf3b4 | 0/2 · P/48k · P/25k | 0/2 · P/61k · P/25k | 0/2 · P/57k · P/34k |  |
| 88e364bc | 0/2 · P/46k · P/25k | 0/2 · P/54k · P/14k | 0/2 · P/43k · P/16k |  |
| 89565ca0 | 0/1 · A/59k ·   | 0/1 · P/48k · P/17k | 0/1 · P/56k · P/14k |  |
| 898e7135 | **1/1** · P/59k ·   | 0/1 · P/47k · P/25k | 0/1 · P/44k · P/14k | XH-E16 |
| 8b7bacbf | 0/2 · P/45k · P/15k | 0/2 · P/46k · P/25k | 0/2 · P/61k · P/13k |  |
| 8b9c3697 | **1/1** · A/39k · P/25k | **1/1** · A/49k · P/22k | 0/1 · P/49k · P/23k | XH-E16, XH |
| 8e5c0c38 | 0/2 · P/34k ·   | 0/2 · P/30k ·   | 0/2 · M/40k · P/13k |  |
| 8f215267 | **1/1** · P/29k ·   | **1/1** · A/12k ·   | **1/1** · A/15k ·   | XH-E16, XH, LOW-E16 |
| 8f3a5a89 | **1/1** · A/34k · P/20k | 0/1 · P/48k · P/25k | **1/1** · A/49k ·   | XH-E16, LOW-E16 |
| 9385bd28 | 0/2 · P/50k · P/25k | 0/2 · P/62k · P/25k | 0/2 · P/43k · P/20k |  |
| 97d7923e | 0/1 · P/50k · P/25k | 0/1 · P/52k · P/22k | 0/1 · A/35k ·   |  |
| 981571dc | **1/1** · A/27k ·   | **1/1** · A/11k ·   | 0/1 · P/56k · P/18k | XH-E16, XH |
| 9aaea919 | **1/1** · A/12k ·   | **1/1** · A/18k ·   | **1/1** · A/24k ·   | XH-E16, XH, LOW-E16 |
| 9bbf930d | 0/1 · P/41k · P/25k | 0/1 · P/58k · P/25k | 0/1 · P/45k · P/16k |  |
| a251c730 | **1/1** · A/38k · P/25k | 0/1 · P/45k · P/21k | 0/1 · P/54k · P/19k | XH-E16 |
| a25697e4 | 0/2 · P/61k · P/25k | 0/2 · P/54k · P/25k | 0/2 · P/48k · P/16k |  |
| a32d8b75 | 0/2 · P/43k · P/25k | 0/2 · M/31k · M/11k | 0/2 · P/47k · P/13k |  |
| a395ee82 | **1/1** · P/54k · P/25k | **1/1** · A/23k ·   | **1/1** · P/28k ·   | XH-E16, XH, LOW-E16 |
| a47bf94d | 0/1 · P/53k · P/25k | 0/1 · P/58k · P/25k | 0/1 · P/56k · P/13k |  |
| a6f40cea | 0/1 · P/57k · P/25k | 0/1 · P/54k · P/25k | 0/1 · P/56k ·   |  |
| aa4ec2a5 | **1/1** · A/30k ·   | **1/1** · A/15k ·   | **1/1** · A/14k ·   | XH-E16, XH, LOW-E16 |
| abc82100 | 0/2 · P/55k · P/25k | 0/2 · P/50k · P/24k | 0/2 · P/51k · P/13k |  |
| b0039139 | 0/2 · P/48k · P/25k | **1/2** · P/47k · P/25k | 0/2 · P/46k · P/18k | XH |
| b10624e5 | 0/1 · P/71k · P/25k | 0/1 · P/47k · P/25k | **1/1** · P/43k ·   | LOW-E16 |
| b5ca7ac4 | 0/1 · P/65k · P/24k | **1/1** · A/55k · P/25k | 0/1 · P/46k · P/13k | XH |
| b6f77b65 | 0/3 · P/46k · P/15k | 0/3 · P/57k · P/17k | 0/3 · P/52k · P/15k |  |
| b99e7126 | 0/1 · P/64k · P/25k | 0/1 · P/47k · P/24k | 0/1 · P/66k · P/17k |  |
| b9e38dc0 | 0/1 · P/46k · P/25k | 0/1 · P/68k · P/25k | 0/1 · P/45k · P/20k |  |
| bf45cf4b | **1/1** · A/9k ·   | **1/1** · A/13k ·   | **1/1** · A/13k ·   | XH-E16, XH, LOW-E16 |
| c4d067a0 | **1/1** · A/31k · P/25k | **1/1** · A/32k · P/25k | **1/1** · P/61k · P/17k | XH-E16, XH, LOW-E16 |
| c7f57c3e | 0/1 · P/77k · P/25k | 0/1 · P/76k · P/25k | 0/1 · P/52k · P/21k |  |
| cb2d8a2c | **2/2** · P/32k ·   | 0/2 · P/60k · P/25k | 0/2 · P/65k · P/25k | XH-E16 |
| cbebaa4b | 0/2 · P/42k ·   | 0/2 · P/28k ·   | 0/2 · M/40k · P/18k |  |
| d35bdbdc | 0/3 · P/66k · P/25k | 0/3 · P/61k · P/25k | 0/3 · A/19k ·   |  |
| d59b0160 | 0/1 · P/46k · P/25k | 0/1 · P/48k · P/25k | 0/1 · P/44k · P/19k |  |
| d8e07eb2 | 0/2 · M/38k · P/13k | 0/2 · P/38k ·   | 0/2 · P/43k · P/14k |  |
| da515329 | 0/1 · P/46k · P/19k | 0/1 · P/44k · P/20k | 0/1 · P/44k · P/16k |  |
| db0c5428 | 0/1 · P/49k · P/25k | **1/1** · A/42k · P/25k | 0/1 · P/69k · P/14k | XH |
| db695cfb | **1/1** · P/29k ·   | **1/1** · A/24k ·   | 0/1 · A/26k ·   | XH-E16, XH |
| dbff022c | 0/1 · P/61k · P/25k | 0/1 · A/46k · P/25k | 0/1 · P/67k · P/28k |  |
| dd6b8c4b | **2/2** · P/51k · P/25k | **2/2** · A/55k · P/19k | **1/2** · P/46k · P/13k | XH-E16, XH, LOW-E16 |
| de809cff | 0/1 · P/42k · P/25k | **1/1** · A/43k ·   | 0/1 · P/43k · P/14k | XH |
| dfadab01 | 0/1 · P/29k ·   | 0/1 · A/51k ·   | 0/1 · P/56k · P/20k |  |
| e12f9a14 | 0/2 · P/62k · P/25k | 0/2 · P/62k · P/25k | 0/2 · P/46k · P/17k |  |
| e3721c99 | 0/2 · P/32k ·   | 0/2 · P/29k ·   | 0/2 · P/30k ·   |  |
| e376de54 | 0/1 · A/22k ·   | 0/1 · A/17k ·   | 0/1 · P/29k ·   |  |
| e8686506 | 0/1 · P/51k · P/25k | 0/1 · P/47k · P/22k | 0/1 · P/47k · P/37k |  |
| e87109e9 | 0/1 · P/65k · P/25k | 0/1 · P/67k · P/25k | 0/1 · P/47k · P/12k |  |
| edb79dae | 0/1 · P/42k · P/25k | **1/1** · A/33k · P/25k | 0/1 · P/48k · P/16k | XH |
| eee78d87 | **1/1** · P/50k · P/25k | 0/1 · P/70k · P/25k | 0/1 · P/54k · P/24k | XH-E16 |
| f560132c | 0/2 · P/34k ·   | 0/2 · P/28k ·   | 0/2 · P/45k · P/23k |  |
| f931b4a8 | 0/2 · P/48k · P/36k | 0/2 · P/59k · P/25k | 0/2 · P/49k · P/18k |  |
| faa9f03d | 0/2 · P/77k · P/25k | 0/2 · P/57k · P/25k | 0/2 · P/48k · P/17k |  |
| fc7cae8d | 0/1 · P/46k · P/25k | 0/1 · P/51k · P/25k | 0/1 · P/43k · P/13k |  |

## 7. Union and complementarity
- union (query OR across runs): 62/172 queries, task-norm 39.58 %
- tasks with a hit in any run (49): 135a2760, 136b0064, 16de56c4, 1818057f, 1ae2feb7, 20270e3b, 221dfab4, 247ef758, 269e22fb, 2ba387bc, 31f7f899, 332f06d7, 36a08778, 38007db0, 3e6067c3, 409aa875, 45a5af55, 4a21e3da, 58490d8a, 58f5dbd5, 5961cc34, 5dbc8537, 67e490f4, 6e453dd6, 7491f3cf, 7c66cb00, 7ed72f31, 80a900e0, 898e7135, 8b9c3697, 8f215267, 8f3a5a89, 981571dc, 9aaea919, a251c730, a395ee82, aa4ec2a5, b0039139, b10624e5, b5ca7ac4, bf45cf4b, c4d067a0, cb2d8a2c, db0c5428, db695cfb, dd6b8c4b, de809cff, edb79dae, eee78d87
- tasks with a hit in every run (16): 16de56c4, 1818057f, 2ba387bc, 38007db0, 3e6067c3, 45a5af55, 4a21e3da, 58f5dbd5, 7ed72f31, 8f215267, 9aaea919, a395ee82, aa4ec2a5, bf45cf4b, c4d067a0, dd6b8c4b
- only XH-E16 (13): 136b0064, 20270e3b, 221dfab4, 31f7f899, 409aa875, 5961cc34, 5dbc8537, 7491f3cf, 80a900e0, 898e7135, a251c730, cb2d8a2c, eee78d87
- only XH (6): 332f06d7, b0039139, b5ca7ac4, db0c5428, de809cff, edb79dae
- only LOW-E16 (1): b10624e5

| pair | both | only A | only B | union | Jaccard |
|---|---|---|---|---|---|
| XH-E16 vs XH | 21 | 16: 136b0064, 20270e3b, 221dfab4, 31f7f899, 36a08778, 409aa875, 58490d8a, 5961cc34, 5dbc8537, 7491f3cf, 80a900e0, 898e7135, 8f3a5a89, a251c730, cb2d8a2c, eee78d87 | 11: 1ae2feb7, 247ef758, 269e22fb, 332f06d7, 6e453dd6, 7c66cb00, b0039139, b5ca7ac4, db0c5428, de809cff, edb79dae | 48 | 0.438 |
| XH-E16 vs LOW-E16 | 19 | 18: 135a2760, 136b0064, 20270e3b, 221dfab4, 31f7f899, 409aa875, 5961cc34, 5dbc8537, 67e490f4, 7491f3cf, 80a900e0, 898e7135, 8b9c3697, 981571dc, a251c730, cb2d8a2c, db695cfb, eee78d87 | 6: 1ae2feb7, 247ef758, 269e22fb, 6e453dd6, 7c66cb00, b10624e5 | 43 | 0.442 |
| XH vs LOW-E16 | 21 | 11: 135a2760, 332f06d7, 67e490f4, 8b9c3697, 981571dc, b0039139, b5ca7ac4, db0c5428, db695cfb, de809cff, edb79dae | 4: 36a08778, 58490d8a, 8f3a5a89, b10624e5 | 36 | 0.583 |

## 8. Projection to the hidden test (ESTIMATE)
|  | XH-E16 | XH | LOW-E16 |
|---|---|---|---|
| gate 4b window | yes | yes | yes |
| eval s/task (planned window / tasks) | 175 | 175 | 174.8 |
| hidden s/task (window / 240) | 175 | 175 | 174.8 |
| task-norm over reached tasks % | 30.00 | 25.83 | 18.89 |
| coverage (b0 started) | 1.000 | 1.000 | 1.000 |
| coverage (b0 got first-exposure budget) | 1.000 | 1.000 | 1.000 |
| hidden coverage estimate | 1.000 | 1.000 | 1.000 |
| LB proxy % (estimate) | 30.00 | 25.83 | 18.89 |

- XH-E16: gate 4b: the eval run had the hidden run's seconds per task, so its coverage is the hidden estimate; LB proxy = task-norm(reached) x coverage (= the all-task task-norm)
- XH: gate 4b: the eval run had the hidden run's seconds per task, so its coverage is the hidden estimate; LB proxy = task-norm(reached) x coverage (= the all-task task-norm)
- LOW-E16: gate 4b: the eval run had the hidden run's seconds per task, so its coverage is the hidden estimate; LB proxy = task-norm(reached) x coverage (= the all-task task-norm)
- LB is scored on 120 of the 240 hidden tasks (noise sd ~1.5 points); past LB/eval ratios were coverage-driven. The proxy is an estimate, not a prediction.

## 9. Data checks (anything that looks wrong)
**XH-E16**
- 3 branch-1 runs were started but count 0 generated tokens (their only request was cut at the deadline: in-flight tokens are not counted)
- check() callers: events 56 vs transcripts 54
- report e11.decode_summary.retract_lines = 1 but the log has no real retract line (the counter matches 'retraction_policy' in server_args)

**XH**
- 3 branch-1 runs were started but count 0 generated tokens (their only request was cut at the deadline: in-flight tokens are not counted)
- check() callers: events 55 vs transcripts 56
- report e11.decode_summary.retract_lines = 1 but the log has no real retract line (the counter matches 'retraction_policy' in server_args)

**LOW-E16**
- check() callers: events 69 vs transcripts 63
- report e11.decode_summary.retract_lines = 1 but the log has no real retract line (the counter matches 'retraction_policy' in server_args)

