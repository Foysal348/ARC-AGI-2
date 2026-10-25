# B: Agent behaviour in the three EVAL120G4B runs (2026-10-24)

## Data and method

The three runs:
- **run 1 = XHIGH + E16.** 46/172 queries (pass1 45, pass2 1), task-norm 30.00 %.
- **run 2 = XHIGH, no E16.** 41/172 (41 + 0), task-norm 25.83 %.
- **run 3 = LOW + E16.** 31/172 (31 + 0), task-norm 18.89 %.

Sources:
- `bundle/e12_run_report.json`: `branches[]` (`rev6_events`, `runs`, `status`, `accepted`, `progress_best`, `usage`), `eval_unit_records`, `usage_all`, `order`.
- `q27x/e12_generated_attempts.json`.
- `duck/<task>_b<k>/branch_result.json` and `transcript.txt`. All three runs are complete: I extracted every `Q27X_E12_DUCK_BUNDLE.zip` to `D:/ARC AGI 2/_scratch/eval120_analysis/B_extract/{xe16,xno,low}`. That is 202, 199 and 203 transcripts plus `branch_result.json`. Nothing was missing.

How correctness was scored:
- Each candidate program is scored against the gold answers in `E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_solutions.json`. A candidate is any `checked[]` or `submissions[]` entry in `branch_result.json`, using its `test_outputs`.
- My numbers agree with `eval_query_hits` in all three runs.

Settings:
- Every run used the same task order (`order` is identical).
- E16 flags, from `build_e14.py` L158: `E12_E16_EARLY_NUDGE=1`, `INSPECTION_CALLS=2`, `NUDGE_TOKENS=12000`, `RAW_CONTENT=1`.
- The LOW run used `E12_REASONING_EFFORT=low`. `configuration.REASONING_EFFORT` still says `xhigh`, but that key belongs to the direct path, not Duck.

Scripts are in `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\b\` (`table.py`, `a1.py` … `a15.py`).

Counts:

| run | branches | b0 | b1 | b0 done / paused | b1 done / paused | completion tok | prompt tok (cached %) |
|---|---|---|---|---|---|---|---|
| XHIGH E16 | 202 | 120 | 82 | 34 / 86 | 3 / 79 | 7,144,122 | 29.1 M (78.2) |
| XHIGH noE16 | 199 | 120 | 79 | 39 / 81 | 4 / 75 | 7,091,227 | 28.0 M (76.8) |
| LOW E16 | 203 | 120 | 83 | 32 / 88 | 1 / 82 | 6,606,525 | 51.0 M (82.4) |

No branch reached `E12_MAX_GEN_PER_BRANCH` = 100k. The largest b0 totals were 77,373, 75,989 and 71,174 tokens. When a branch ended "paused", it was still alive at the deadline, so time ended it, not a token cap.

---

## 1. E16 effect (run 1 vs run 2)

### 1.1 Share of branches that ever called check(), and when

The table gives `rev6_events.first_check_tokens` and `first_transform_tokens`. Branches that never called check() are right-censored at their final `generated_tokens`.

| | XHIGH E16 b0 | XHIGH noE16 b0 | LOW E16 b0 |
|---|---|---|---|
| ever check() | **55/120 (45.8 %)** | **52/120 (43.3 %)** | 60/120 (50.0 %) |
| ever one-grid transform | 55 | 52 | 57 |
| first_check p10/25/50/75/90 (observed) | 11.7k / 23.9k / 30.7k / 37.5k / 43.3k | 10.6k / 17.6k / 29.7k / 40.8k / 47.2k | 8.2k / 16.5k / 26.9k / 40.8k / 46.4k |
| censored (no check): final tokens p10/50/90 | 37.0k / 49.2k / 65.1k (n=65) | 31.6k / 50.0k / 63.0k (n=68) | 42.0k / 47.4k / 61.4k (n=60) |
| KM P(check by 12k / 24k / 36k / 48k / 60k) | .050 / .125 / .321 / .450 / .529 | .067 / .167 / .308 / .420 / .475 | .067 / .208 / .334 / .487 / .545 |
| first transform ≤16k / ≤24k / ≤32k / ≤40k | 7 / 15 / 28 / 43 | 12 / 20 / 31 / 38 | n/a |

Branch 1 almost never calls check():
- XHIGH E16: 1/82. It was 1ae2feb7 at 11.0k tokens, and the program was accepted but wrong.
- XHIGH noE16: 3/79, none accepted.
- LOW: 9/83, none accepted.

**E16 did not move the first check earlier.** At 12k and 24k, run 1 is slightly *behind* run 2 (5.0 vs 6.7 % and 12.5 vs 16.7 %). It is ahead only beyond 36k, and the share that ever called check() differs by 3 branches.

### 1.2 Nudges by reason

The nudge reason is `e16_nudge_reason` and the timing is `provisional_nudge_tokens`. In run 2 the base adapter has no reason key, so its nudges are the 24k nudge.

| | inspection (12k, ≥2 inspection-only calls) | existing 24k | no nudge |
|---|---|---|---|
| XHIGH E16 b0 | **89** (nudge at p10/50/90 = 12.4k / 15.5k / 21.2k) | 20 (24.9k / 27.0k / 37.8k) | 11 |
| XHIGH noE16 b0 | n/a | **90** (24.8k / 27.5k / 39.1k) | 30 |
| LOW E16 b0 | **100** (12.3k / 14.3k / 19.9k) | 6 | 14 |

Other counters:
- The request carrying the nudge was actually sent (`provisional_nudge_seen_request` set) in 82/89 inspection branches in run 1, 78/90 in run 2 and 99/100 in LOW.
- How the nudge was delivered (`provisional_nudge_via`): run 1 tool 108, opener 1; run 2 tool 88, opener 2.
- Inspection-call counts (`e16_inspection_calls`) in run 1 b0: median 8, up to 29.
- **The raw-content half of E16 was a no-op on XHIGH.** `e16_raw_content_kept` = 0 and `e16_raw_content_markup` = 292 in run 1, because every assistant message with content also carried tool-call markup. In LOW, kept = 7 and markup = 140. So on XHIGH, E16 means the early nudge and nothing else.

### 1.3 What branches did after the nudge (b0)

| group | n | transform after nudge | within 2k / 4k / 8k / 16k tok | nudge→transform gap p10/50/90 | accepted | correct (accepted and right on all test inputs) | task hit |
|---|---|---|---|---|---|---|---|
| E16 inspection | 89 | 36 (40 %) | 2 / 3 / 7 / 17 | 5.9k / 16.4k / 30.3k | 31 | 23 | 24 |
| E16 existing 24k | 20 | 10 | 0 / 1 / 5 / 9 | 4.4k / 7.6k / 13.1k | 7 | 6 | 7 |
| E16 no nudge | 11 | 9 (before any nudge) | – | – | 8 | 6 | 6 |
| noE16 24k nudge | 90 | 25 (28 %) | 5 / 7 / 12 / 19 | 1.2k / 8.1k / 27.2k | 22 | 13 | 15 |
| noE16 no nudge | 30 | 27 (before 24k) | – | – | 24 | 17 | 17 |
| LOW inspection | 100 | 40 | 7 / 7 / 8 / 21 | 1.3k / 14.2k / 37.6k | 28 | 12 | 17 |

**The model ignores the nudge.** Only 3 of 89 branches wrote a transform within 4k tokens of the 12k nudge. The median gap was 16.4k tokens, which looks like "when it was ready" rather than a response to the nudge.

Across all b0 thinking, only 6 run-1 transcripts mention "provisional" outside the opener text. Exactly one says "following the harness note".

### 1.4 Effect on the population E16 actually targeted

I grouped tasks by the nudge reason of run 1's branch 0 and counted task-normalised hits (`a11.py`):

| run-1 b0 group | tasks | run 1 task-norm | run 2 task-norm | run 1 q-hits | run 2 q-hits |
|---|---|---|---|---|---|
| inspection-nudged | 89 | **23.50** | **23.50** | 29 | 31 |
| existing 24k | 20 | 6.50 | 3.50 | 11 | 6 |
| no nudge | 11 | 6.00 | 4.00 | 6 | 4 |

**On the 89 tasks where the early nudge fired, the effect is exactly zero.** The whole +5-query / +4.17 pp gap comes from the 31 tasks the early nudge did not touch. On those tasks the two runs differ only by sampling, so the gap is noise.

Nudge timing makes no difference either. For nudges at 12–14k, 14–17k, 17–20k and 20–24k, run 1 solved 7/6/6/5 tasks against 7/10/6/1 for run 2.

### 1.5 Branches possibly hurt

11 tasks were solved in run 2 but not in run 1. In run 1, 10 of them were inspection-nudged early, against a base rate of 89/120 = 74 %; the 11th, b0039139, got only the 24k nudge.
- **Never wrote a transform in run 1 (8):** 1ae2feb7, 247ef758, 269e22fb, 332f06d7, 7c66cb00, b5ca7ac4, db0c5428, de809cff.
  - In run 2, three of these needed no nudge: 1ae2feb7 wrote a transform at 17.6k, de809cff at 23.8k and b5ca7ac4 at 54.0k.
  - The other five got run 2's 24k nudge and wrote transforms 4–10k later.
- **edb79dae:** the transform came only at 41.3k and was not accepted.
- **6e453dd6:** the only case with a plausible nudge-caused mechanism. Nudge at 17,162 → transform at 18,343 → accepted on 3/3 train pairs but wrong on test → `submit` → branch "done" at 19,197 tokens. In run 2 the branch went on to 47–48k and found the correct program.

Going the other way, 10 of the 16 tasks solved only in run 1 were also inspection-nudged. I see no systematic harm. I also see no systematic benefit. The one real failure mode is that early pressure can produce a quick train-exact but wrong program, and the branch then stops.

### 1.6 Is +5 queries beyond noise?

- **Query level.** Both runs: 26. Run 1 only: 20. Run 2 only: 15. Union: 61. Sign test p = 0.50.
  - If each query is solved with its own probability p_i, the 35 discordant queries ≈ Σ2p(1−p). That gives a per-run sd of about 4.2 queries and an sd of the difference of about **5.9 queries**.
  - +5 is therefore about 0.85 sd. Clustering by task gives sd 7.1 over 28 discordant tasks.
- **Task level.** Any hit in both: 21. Run 1 only: 16. Run 2 only: 11. Task-norm difference +5.0 against a null sd of 5.0 (z = 1.0). Sign test p = 0.44.
- **Verdict: not beyond noise.** The mechanism analysis (1.4) also shows zero effect where E16 acted.
- Run-to-run overlap is only 26 of 61 queries. Each XHIGH run solves about 43 queries, but two runs together reach 61. **Independent attempts are highly complementary.** Today attempt_2 does not exploit that (section 3).

---

## 2. LOW vs XHIGH (run 3 vs runs 1 and 2)

| per branch 0 | XHIGH E16 | XHIGH noE16 | LOW E16 |
|---|---|---|---|
| generated tokens mean (p50 / p90) | 44,143 (45.4k / 61.9k) | 44,673 (46.8k / 61.6k) | 42,871 (45.9k / 59.6k) |
| requests p50 / p90 | 12 / 19 | 11 / 18 | 15 / 24 |
| tool calls mean | 13.0 | 12.7 | **16.4** |
| tokens per tool call | 3,389 | 3,522 | **2,610** |
| completion per request p50 / p75 / p90 | 1,129 / 4,332 / 10,206 | 1,017 / 4,489 / 11,297 | 1,015 / 3,310 / **6,800** |
| request finishes: tool_calls / length / stop (EOS without a tool call) | 1461 / 39 / 1 | 1448 / 32 / 0 | 1987 / 10 / **8** |
| reasoning chars / gen token | 2.27 | 2.28 | 2.37 |
| digit share of reasoning | .108 | .107 | .087 |
| branches with context trimmed (prompt > 51.4k) | 32 | 41 | **64** |
| accepted programs: right / partly right / wrong | 59 / 4 / 20 | 57 / 7 / 24 | **36 / 9 / 27** |
| tasks whose accepted programs are all wrong | 12/47 (26 %) | 16/46 (35 %) | **21/41 (51 %)** |
| "wrote transform, never passed train" tasks | 9 | 7 | **22** |
| branches with partial progress (best k<n) and no accept | 9 | 8 | 27 |

**How LOW fails:**
- **It does not save tokens.** The budget is set by time, so branch 0 spends about the same 43k tokens.
- **Shorter steps, more of them.** It thinks less per step (p90 request 6.8k vs 10.2k) and makes about 25 % more tool calls.
- **Expensive context.** Prompt tokens are 51.0M vs 29.1M and uncached prefill 9.0M vs 6.3M. The context fills faster, so 64 of 120 branches get trimmed. Trimmed LOW branches end correct 5 % of the time (3/64), against 30 % for untrimmed ones.
- **Earlier checks, worse programs.** P(check by 24k) is .208 vs .125, but the programs are worse:
  - only 50 % of LOW's accepted programs are right on test, vs 71 % (XHIGH E16);
  - first checks after 36k almost never pay: 1 correct from 20 checks;
  - 22 tasks wrote a transform that never passed train.
- **EOS mid-thinking is rare** (8 `stop` finishes in branch 0). That is not the failure mode.
- On branch 1, LOW does call tools (6.4 per branch), unlike XHIGH, but gets 0 accepted programs.

**Solved-set overlap (queries):**
- LOW ∩ XHIGH E16 = 22: LOW-only 9, XHIGH-E16-only 24. Sign test p = 0.014; task level z = 2.78, p = 0.023.
- LOW ∩ XHIGH noE16 = 27: LOW-only 4, noE16-only 14. p = 0.031.
- **Only one query is solved by LOW and by neither XHIGH run: b10624e5 q0.**
- The other LOW-only hits are hits that one XHIGH run also had: 36a08778, 58490d8a and 8f3a5a89 (solved by run 1 but not run 2); 1ae2feb7, 247ef758, 269e22fb, 6e453dd6, 7c66cb00 and 7ed72f31 (solved by run 2 but not run 1).
- All three runs together reach 62 queries over 49 tasks.

Verdict: LOW is clearly worse, by 10–15 queries, and adds about 1 unique query.

---

## 3. attempt_2 and branch 1

### 3.1 How many tasks got branch 1, and what it produced

| | b1 branches | made ≥1 request | first b1 request (s from start) p10/50/90 | b1 tokens | share of stream-seconds | accepted / correct |
|---|---|---|---|---|---|---|
| XHIGH E16 | 82 | 79 | 11,335 / 15,120 / 19,101 | 1,846,945 (25.9 % of all completion) | **26.7 %** | 1 / **0** |
| XHIGH noE16 | 79 | 76 | 11,555 / 15,517 / 19,297 | 1,730,457 (24.4 %) | 25.9 % | 0 / **0** |
| LOW E16 | 83 | 83 | 11,029 / 15,523 / 19,648 | 1,462,022 (22.1 %) | 25.9 % | 0 / **0** |

Branch 1 gets `stage2_class` 1 ("nothing yet") in 80/82, 77/79 and 80/83 cases. It starts after stage 1 of every branch 0 is finished; the latest stage-1 end was 10,905 s in run 1.

On XHIGH, branch 1 uses the `indexed` grids view, and its **first request runs to the 24,576 cap while thinking, with no tool call**:
- first-request completion has median 24,576 and p25 22.5k; median request time is 831 s;
- request finishes are length 59, tool_calls 30 in run 1, and length 52, tool_calls 37 in run 2;
- tool calls per branch: 0.3 in run 1 and 0.5 in run 2.

The model is doing direct, in-head solving of the indexed grids. An example is `247ef758_b1`: the thinking is verifying a stamping rule with ✓ marks when the cap cuts it. The run then stops at the 12,288 chunk (`runs[0].pause_at` = 12288, status paused). It gets "stop_requested" in the transcript and never receives a second round, because the round-robin `rank()` puts rounds=1 behind every rounds=0 item.

Across 244 branch-1 runs in three runs: **0 correct programs, and one accepted program, which was wrong.**

By contrast, branch 0's stage 2 (24.0 % of stream-seconds in run 1) produced 18 of the 32 *correct* first accepts in run 1, 15/27 in run 2 and 5/19 in LOW (`a10.py`).

### 3.2 Why pass2 ≈ 0

The table classifies the sources of `e12_generated_attempts.json` attempts against the candidates (`a5.py`):

| | a1 = accepted program | a1 = test-input fallback | a2 when a1 accepted: test input / 2nd accepted | queries with ≥2 distinct accepted outputs | pass2 hits |
|---|---|---|---|---|---|
| XHIGH E16 | 60 | 112 | 58 / 2 | 3 | 1 (5961cc34 q0, two distinct accepted b0 programs) |
| XHIGH noE16 | 61 | 111 | 57 / 4 | 6 | 0 |
| LOW E16 | 52 | 120 | 50 / 2 | 2 | 0 |

There are three causes:
1. **Branch 1 produces no candidates** (3.1).
2. **Branch 0 almost never holds two distinct train-exact programs.** After the first accepted `submit` the branch ends ("done"). That happened in 34 of 120 run-1 branch-0 runs, 25 of them correct.
3. **Partial programs are never run on test.** Only train-exact check() calls are signalled for host verification (`checks_signalled` ≈ `checks_accepted`: 53/53, 54/52, 54/43). So `select_attempts_r5`'s "partial" tier is empty (1, 3 and 1 non-accepted candidates per run). attempt_2 then falls back to the test input. Branches with partial progress (best k<n) and no accept that could supply an a2: 9, 8 and 27.

`select_attempts_r5` itself works correctly: when two distinct accepted outputs existed, attempt_2 got the second one, and that produced the one pass2 hit.

How much attempt_2 could gain:
- Queries where attempt_1 was an accepted but wrong program: 16 in run 1, 21 in run 2, 25 in LOW.
- Another run solved 5, 6 and 12 of these.
- Across all unsolved queries, another run solved 16 of 126 (run 1), 21 of 131 (run 2) and 31 of 141 (LOW).

---

## 4. Wrong accepts (overfit on test)

| | accepted programs | right on all tests | partly right | all wrong | branches whose only accepts are wrong (done / paused) | tasks with accepts, all wrong |
|---|---|---|---|---|---|---|
| XHIGH E16 | 83 | 59 | 4 | 20 | 12 (6 / 6) | **12/47**: 0934a4d8, 1ae2feb7, 20270e3b, 4c3d4a41, 53fb4810, 6e453dd6, 78332cb0, 7b5033c1, 7ed72f31, 89565ca0, dfadab01, e376de54 |
| XHIGH noE16 | 88 | 57 | 7 | 24 | 16 (9 / 7) | **16/46**: 221dfab4, 247ef758, 31f7f899, 58490d8a, 5961cc34, 71e489b6, 78332cb0, 7b5033c1, 7b80bb43, 80a900e0, b0039139, d35bdbdc, dbff022c, dfadab01, e376de54, fc7cae8d |
| LOW E16 | 72 | 36 | 9 | 27 | 21 (13 / 8) | **21/41**: 0934a4d8, 16de56c4, 1ae2feb7, 221dfab4, 247ef758, 31f7f899, 36a08778, 3dc255db, 4c3d4a41, 53fb4810, 67e490f4, 7491f3cf, 7b5033c1, 80a900e0, 97d7923e, a6f40cea, b5ca7ac4, d35bdbdc, db695cfb, dd6b8c4b, e376de54 |

Notes on the table:
- Overfit is stochastic: 221dfab4, 31f7f899, 58490d8a, 5961cc34 and 80a900e0 overfit in run 2 but were right in run 1.
- Branches whose only accepts were wrong ended at these token counts:
  - run 1: 4.8k, 12.4k, 19.2k, 22.3k, 24.4k, 27.1k, 28.6k, 30.3k, 30.4k, 34.0k, 59.2k, 64.1k;
  - run 2: 12.1k … 61.5k.

  Many stopped with budget left.
- **The `overfit_flags` heuristics do not separate right from wrong programs.**
  - Run 1: 12 flagged programs, 5 wrong and 7 right.
  - Run 2: 20 flagged, 6 wrong and 14 right.
  - LOW: 16 flagged, 10 wrong and 6 right.
  - Flag types are mostly `fixed_offset`, `test_output_equals_input` and `test_shape_outside_train_pattern`.

---

## 5. Failure taxonomy of unsolved tasks

Tasks with any query missed (`a15.py`):

| class | XHIGH E16 (85) | XHIGH noE16 (90) | LOW (100) |
|---|---|---|---|
| A never wrote a one-grid transform (either branch) | **64** (b0 paused after 2 runs: 54; after only stage 1: 8; done: 2) | **67** (53 / 12 / 2) | **57** (51 / 3 / 3) |
| B wrote a transform, never passed train | 9 (best 0/2, 0/3×3, 1/2, 1/3, 2/3, 2/5, 4/5) | 7 | 22 |
| C passed train, wrong test | 12 (2 of them partial 1/2) | 16 (2 partial) | 21 (5 partial) |

More detail:
- **Class A branches, cut by the deadline.** Final branch-0 tokens in run 1: p10/50/90 = 37.0k / 49.6k / 65.1k. Reasoning chars: median 113k. Tool calls: median 12. Nudges: 52 inspection, 10 at 24k.
  - All but 2 (run 1), 2 (run 2) and 3 (LOW) were still paused at the deadline.
  - 8 tasks in run 1 and 12 in run 2 never got a stage-2 resume at all.
  - The last ORDER2 quartile scores lowest: task-norm Q1–Q4 is 11.0 / 10.5 / 9.0 / 5.5 in run 1 and 7.0 / 11.5 / 9.5 / 3.0 in run 2.
- **Request cap hits** (a request reaching 24,576 tokens, `finish=length`, `length_cut_no_call`):
  - branch 0: 39, 32 and 10;
  - branch 1: 59, 52 and 2.
  - Branch-level cap (100k): 0 everywhere.
- **Yield of first checks once they happen (XHIGH).** About 85 % of first checks are accepted, and about 70 % of accepted programs are right. The model writes code late but well. By first-check token bin, the run-1 counts for checked / accepted / right are:

  | first check at | checked | accepted | right |
  |---|---|---|---|
  | <12k | 6 | 5 | 3 |
  | 12–24k | 9 | 9 | 6 |
  | 24–28k | 7 | 7 | 4 |
  | 28–36k | 16 | 14 | 13 |
  | 36–44k | 12 | 7 | 6 |
  | 44–60k | 5 | 4 | 3 |

  The hazard of a first check per 8k tokens is .07 / .12 / .17 / .14 / .09 / .06 over 16–64k (run 1), and .07 / .11 / .09 / .14 / .08 / .00 (run 2). The bottleneck is time to the first transform, i.e. coverage.

### 5.1 Two context-loss mechanisms (found in the transcripts and requests)

**(a) Duck turn rollback.** This comes from the vendored Duck code, `vendor/inference/agent/tool_agent.py`, L7094–7101 and L7183–7186.
- The setting is `yield_tokens: 2048` per turn.
- The model sometimes emits two parallel tool calls. If the turn-token yield (`turn_token_budget`) or `stop_requested` fires after the first call, `preserve_history = False`. The *whole turn* is then rolled back to `previous_history_messages`.
- I detected rollbacks as requests whose prompt is smaller than the previous prompt without the trimmer flag:

  | run | events | branch-0 runs affected | context tokens lost |
  |---|---|---|---|
  | run 1 | 56 | 35 | 522,561 |
  | run 2 | 47 | 29 | 460,421 |
  | LOW | 6 | 6 | 57,697 |

- Two-call responses in branch 0: 204, 165 and 35.
- Example: `xe16/duck/1ae2feb7_b0`.
  - Every turn ended "Yielded control to solver: turn_token_budget" with `history_messages: 0`.
  - The prompt dropped to 2,358 tokens at requests 4, 8, 12 and 15, so the model re-analysed train 0 from scratch four times in 43k tokens.
  - Run 2 solved this task at 17.6k.
- Outcome association is mixed. Correct among affected vs unaffected runs: 29 % vs 29 % in run 1, 14 % vs 29 % in run 2. The waste itself is unambiguous.

**(b) Context trimming at `context_budget_tokens` 51,396.** That is the window of 76,484 minus the 24,576 reply reserve minus 512.

| run | branches trimmed | tokens trimmed | correct, trimmed vs untrimmed |
|---|---|---|---|
| run 1 | 32 | 883,826 | 16 % vs 34 % |
| run 2 | 41 | 1,064,713 | 32 % vs 22 % |
| LOW | 64 | 1,614,519 | 5 % vs 30 % |

Other agent defects:
- "Unknown tool: submit" (the model calls `submit` as a tool instead of through python): 27 responses in 23 branches, 34 in 28, and 33 in 28. Each costs one request.

---

## 6. Agent-level levers, ranked by evidence

1. **Drop the current branch 1 (indexed view) and give its time to branch 0 continuation.**
   - Evidence: about 26 % of stream-seconds (1.85M tokens in run 1) produced 0 correct programs in 244 runs. Branch-0 stage 2, using a similar 24 %, produced 18, 15 and 5 correct first accepts.
   - KM P(first check) rises from .45 at 48k to about .53–.61 at 60–72k. In run 1, 63 branch-0 runs with no check were still alive (paused) at the deadline.
   - Moving the branch-1 tokens raises the average branch-0 budget from about 44k to about 59k.
   - Estimate: +16–25 extra first transforms × ~0.85 accepted × ~0.70 right ≈ **+8 to +12 queries**. This is my extrapolation from the hazard; trimming above 51k may reduce it.
2. **Fix the Duck turn rollback.**
   - Evidence: 56 events, 35 branches and 0.52M context tokens lost in run 1 (5.1).
   - Fix: when a yield fires mid-batch, execute or keep the reply's remaining tool calls, or drop only the unexecuted call. Never discard the turn.
   - This is a bug fix with no token cost.
3. **Second attempt for attempt_2 from an *independent* branch-0-style sample (facts view, tool-first), not from the indexed view.** Use it where branch 0 finished early with one accepted program (stage-2 class 2).
   - Evidence: the two XHIGH runs overlap on only 26 of 61 queries.
   - Of the queries with an accepted-but-wrong attempt_1, another run solved 5/16, 6/21 and 12/25.
   - 6, 9 and 13 overfit branches stopped "done", often below 30k tokens.
   - Today 58 of the 60 accepted attempt_1s in run 1 have the test input as attempt_2.
4. **Avoid context trimming at 51.4k:** a smaller reply reserve after stage 1, or trim old tool outputs before reasoning.
   - Evidence: trimmed branches end correct 16 % vs 34 % (run 1) and 5 % vs 30 % (LOW). Run 2 points the other way: 32 % vs 22 %. So the run-1 and LOW gap may be a selection effect (long, hard branches), but it matters more once lever 1 makes branches longer.
5. **attempt_2 from the best partial program.** Host-run the best k<n function on the test inputs and use its output as the attempt_2 fallback instead of the test input.
   - Covers 9, 8 and 27 branches.
   - Low expected yield, but free.
6. **Accept `submit` as a tool name** (23–28 branches per run waste a request). Small.
7. **E16 early nudge: neutral. Keep it or drop it, but do not count the +5 as a gain, and do not lower it further.**
   - Compliance is 3/89 within 4k tokens.
   - Zero effect on the 89 tasks it touched (23.50 vs 23.50).
   - The first transform is not earlier.
   - The raw-content part never fired on XHIGH (kept = 0).
   - One plausible premature-commit loss (6e453dd6).
8. **Do not use LOW effort:** −15 and −10 queries, p = 0.014 and 0.031; it adds 1 unique query (b10624e5); it has 2× the wrong-accept rate and 2× the trimming.
9. **Not useful: `overfit_flags` as a selection or rerun signal.** Precision is 42 %, 30 % and 63 % in the three runs.
