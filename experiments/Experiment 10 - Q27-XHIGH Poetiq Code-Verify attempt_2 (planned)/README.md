# Experiment 10: Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)

**Status (2026-10-14):**
- The base notebook is built and has passed CPU validation.
- The Poetiq part is **not built yet**. It will be built only after the user approves the plan.
- Nothing has been pushed or run on Kaggle.

## Base: `Experiment Notebooks/Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb`

**The base is Experiment 9 without the tail resample.** In other words, it is v1.4.4 DIRECT (LB 26.94) with the 86k budget and no puzzle ordering.

| | |
|---|---|
| Builder | `Build/build_e10_base.py` (deterministic; built twice with an identical sha) |
| Source | `Build/source/Q27-XHIGH-v1_4_4_LB26.94.ipynb` (sha `959959806170960c`) |
| Output sha256 | `530cb119e3028994…` |
| Changes from v1.4.4 | thinking 77,824 + answer 8,192; `MAX_MODEL_LEN` 106,496; `EXPECTED_LIVE` 43,000; PROXY56 profile available (commit default is EVAL12); `t_admit_min` / `t_done_min` in the records |
| Removed from E9 | the tail resample. On EVAL12 it used 41.3 % of the generated tokens for 0 gain. Its report fields went with it |
| Diff against the E9 notebook | only the resample code and comments (checked cell by cell) |

**Why this base:**
- 86k was tested in E9: 4/22 on EVAL12 (our best), KV max 91.4 %, 0 preemptions.
- Poetiq's iterations take over the tail slot that the resample used.
- The E7 puzzle ordering is **not** included (one change at a time).

**Caution:** 86k has never been submitted to the LB. The only LB-proven notebook is v1.4.4 at 49k (26.94).

**CPU validation** (`Experiment Review/CPU validation base (2026-10-14)/`): `harness_e10.py`, which is `harness_e9.py` with the resample checks replaced by no-resample checks.

| Scenario | Result |
|---|---|
| mixed | 22/22, `max_model_len` 106,496, no resample fields, `projection_valid` True, record times present |
| drain | drain and hard stop handled; projection marked invalid |
| rerun | 259/259 hidden-test inputs; model grids written to `submission.json` |

Run it with:
```
set Q27X_FIXTURES=<Experiment 1>/Experiment Notebooks/Build/Q27X_build/fixtures/fake_input
python harness/harness_e10.py ../Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb mixed|drain|rerun
```

## RULE20 variant (2026-10-14): `Experiment Notebooks/Q27-XHIGH-v1_4_4_E10_BASE_86K_RULE20.ipynb`

The same base notebook, with one difference: the default commit profile is **RULE20**.
- **What RULE20 is:** 20 evaluation tasks that meet both ordering rules (at least 3 training examples, and exactly 1 test input).
- **How they were drawn:** randomly (`random.Random(20261014)`) from the 52 tasks that meet both rules.
- **Tasks:** `332f06d7 3dc255db 6ffbe589 7491f3cf 7b0280bc 7b3084d4 8f215267 8f3a5a89 981571dc 9bbf930d a395ee82 a47bf94d a6f40cea b5ca7ac4 bf45cf4b c4d067a0 da515329 dbff022c dfadab01 eee78d87`.
- **Overlap with other sets:** none of these tasks is in EVAL12; 8 are in PROXY48.
- **Prior solves:** Qwen3.8-27B (any run, any level) has solved 12 of the 20; ARC Prize xhigh solved 9.
- **Settings:** 20 test inputs, a 3 h budget, admit at 0.80 and drain at 0.90 of the budget, the same as EVAL12. Set `Q27X_PROFILE=EVAL12` to get the old default back.
- **Build:** `python Build/build_e10_base.py --rule20`. Two builds gave the same sha256, `2168cfb20ce1cb8b…`.
- **Base unchanged:** the base notebook keeps sha `530cb119…`.
- **CPU validation** (`Experiment Review/CPU validation RULE20 (2026-10-14)/`): the mixed, drain and rerun scenarios all PASS. With the real tokenizer, the median prompt is 5,000 tokens and the longest is 16,400, so no task's thinking budget is reduced.

## The method: Poetiq (chosen by the user on 2026-10-14)

**Source:** [poetiq-ai/poetiq-arc-agi-solver](https://github.com/poetiq-ai/poetiq-arc-agi-solver) (MIT). A copy for reading only is in `Build/poetiq_reference/`. It is never executed.

**Why Poetiq rather than Imbue or Agentic Coding:**
- **Cost.** An Imbue run costs about $2.67 per task with Kimi K2.5, roughly 20× our budget. Poetiq has a configuration for an open-weight model (GPT-OSS-120B) at under one cent per problem.
- **Size.** Poetiq is about 600 lines with one LLM-call loop, so it ports directly to our offline vLLM engine.
- **Our history with REPL agents.** Experiment 4 (a REPL agent) scored 0/22, because Qwen tool calls failed under vLLM 0.19.

### What Poetiq does (from `solve_coding.py` and `solve_parallel_coding.py`)

1. The model writes a `transform(grid)` function instead of a grid.
2. The code runs in a subprocess on every training pair, with a timeout.
3. If every training pair passes, the loop stops early and the code produces the test output.
4. Otherwise it builds feedback:
   - the wrong cells, shown as `pred/correct`;
   - a shape mismatch, shown as "your shape … correct shape …";
   - the error trace;
   - an accuracy score per example.
5. The next iteration is a fresh prompt. It includes the best ≤ 5 earlier solutions, each with its code, feedback and score, in improving order. The training examples are shuffled each time.
6. If no program passes, the best program by train score is returned.
7. Several experts run in parallel, and their results are voted on:
   - outputs are grouped by identical test output;
   - groups that pass training come first, ordered by vote count;
   - failing groups follow, ordered by soft score;
   - one result is taken per group, and the first two become the two attempts.

### What we adapt (plan; not built)

| # | Poetiq part | In our notebook |
|---|---|---|
| 1 | Code instead of a grid | After DIRECT closes its thinking, an **answer-only branch** starts from the same prompt + thinking prefix (prefix cache): "write this rule as `transform()`". No new thinking. At most 8,192 answer tokens. |
| 2 | Run the code on the training pairs | A subprocess sandbox with a timeout. The Experiment 4 audit-hook guard can be reused. |
| 3 | Feedback: `pred/correct` diff, shape mismatch, error | Ported as is. |
| 4 | Iterations with the best ≤ 5 solutions, shuffled examples | **In the tail slot** (where E9 ran the resample). Each iteration gets the full thinking budget (never below 49,152). |
| 5 | Early stop when all training pairs pass | No further iterations for that task. |
| 6 | Experts + voting | The test-input units of the same task act as the experts. Each program runs on every test input of its task, and the votes are counted per output. |
| 7 | Choosing the two attempts | **attempt_1 = DIRECT, never changed.** attempt_2 = the best-voted code output, if it differs from attempt_1. In v1.4.4, attempt_2 is a copy of the test input. |
| 8 | Best train score when nothing passes | Still used as attempt_2. |

**Not adapted:**
- hosted APIs (offline);
- 10 iterations × 8 experts (too much compute);
- temperature 1.0 (the primary stays at 0.6).

### Evidence and risks

- **For:**
  - **`247ef758#1` (E9).** The model's reversed layer order already fails training example 0, so running the code on the training pairs would have caught it.
  - **Shape and jagged-grid errors.** Code-rendered grids cannot drop rows. This applies to 4 units in E9 and 5 in E8, as long as the rule itself is right.
- **Against:**
  - **Experiment 2 (R6 code loop) scored 2/22**, the same as DIRECT. Its repair turns were capped at 12k and were force-closed. Here every iteration gets the full budget.
  - **Training verification is a weak selector.** In R6, 3 of 4 train-verified programs were wrong on test. That is why attempt_1 stays DIRECT.
- **Unknown:**
  - the number of extra hits, which needs a GPU run;
  - the cost of the answer branch (prefill of the prefix if it was evicted from the cache).

## Folders

- **Experiment Notebooks/:**
  - the base notebook;
  - `Build/build_e10_base.py`;
  - `Build/source/`;
  - `Build/harness/harness_e10.py`;
  - `Build/poetiq_reference/` (Poetiq source, for reading only).
- **Experiment Review/:** the CPU validation logs, and any review bundles later.
- **Experiment Results/:**
  - `Analysis/`;
  - `Kaggle Runs/`.

## Poetiq R4 runs (2026-10-14)

- **R4 itself:** GPT's notebook `Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R4.ipynb`, sha `596bf694…`.
  - Audit and Kaggle sandbox probe: PASS (`Experiment Review/R4 audit (Claude, 2026-10-14).md`).
  - Committed as `foysalemonshanto/q27-xhigh-e10-poetiq-r4` v1 on EVAL12; the user is cancelling that run to free their own GPU.
- **EVAL22 variant:** `Q27-XHIGH-v1_4_4_E10_86K_POETIQ_R4_EVAL22.ipynb`, sha `cf7341f2…`, built by `Build/build_e10_r4_eval22.py`.
  - Differs from R4 in cells 0 and 2 only.
  - Default commit profile `EVAL22`: the 12 EVAL12 tasks + 10 RULE20 tasks (`7b0280bc 7b3084d4 981571dc a395ee82 a6f40cea b5ca7ac4 c4d067a0 da515329 dbff022c dfadab01`). That is 22 tasks / 32 test inputs, 3 h.
  - CPU harness (sandbox stubbed, think 77,824): 22 tasks, 210 iterations, solver complete, no error.

## ORDER2 variant (2026-10-14): `Experiment Notebooks/Q27-XHIGH-v1_4_4_E10_86K_ORDER2.ipynb`

- **What it is:** the E10 BASE 86K notebook (DIRECT, LB-26.94 code + 86k, no resample) with a **Qwen3.8-27B-specific queue order** in every mode, commit and hidden rerun.
  - **Score:** `[>= 3 training examples] + [1 test input]`.
  - **Order:** higher score first; ties keep the original sorted-id order; the units of one task stay together.
- **Build:** `python Build/build_e10_order2.py`. Two builds gave the same sha256, `6f4b2e7df458fcd0…`.
- **Changes:**
  - cell 4: `order_score`, `ORDER_SCORE`, `ORDERED_UNITS`, with asserts;
  - cell 7: `queue = ORDERED_UNITS`; records gain `order_score` and `admit_rank`;
  - cell 0: the banner.
- **Evidence for the rule:** `resources/Puzzle Capability Ledger/derive_ordering_rules.py`.
  - Nested-CV AUC 0.57, against 0.51–0.53 for the NVARC tier/fine order.
  - Solve rate by score: 2 → 58 %, 1 → 43 %, 0 → 20 %.
- **CPU validation** (`Experiment Review/CPU validation ORDER2 (2026-10-14)/`): mixed, drain and rerun all PASS.
  - Rerun with the placeholder test file: 259/259 units. Admission follows the ORDER2 rank, and scores never increase along the queue.
  - Hidden placeholder: 182 units with score 2, 69 with 1, 8 with 0.
  - EVAL12 tasks by score 2/1/0 = 1/8/3.
- **Default commit profile: EVAL12,** so the result is directly comparable with E9 (4/22).
- **Update (same day):** the ORDER2 commit-run default is now **EVAL20F**, sha `5205bbd0df3e950c…`.
  - It is the 12 EVAL12 tasks plus the 8 tasks whose thinking closed naturally with the fewest tokens in E8 R5 (PROXY48, 49k): `e376de54` 17.6k, `b99e7126` 18.7k, `31f7f899` 31.0k, `dbff022c` 35.3k, `5961cc34` 37.0k, `c4d067a0` 39.5k, `3dc255db` 39.7k, `7491f3cf` 41.8k.
  - E8 solved 6 of those 8.
  - The profile has 20 tasks / 30 inputs and a 3 h budget. Set `Q27X_PROFILE=EVAL12` to get the old default back.
  - CPU mixed, drain and rerun all PASS. Prompts: median 3,832 tokens, max 11,962; no thinking budget reduced.
- **Update 2 (same day): EVAL20F is replaced by MIX20**, sha `08f0741ae8d6eb91…`. EVAL12 was dropped because it has been run many times already.
  - MIX20 = FAST8 (above) + FRESH12. FRESH12 are 12 tasks we have never run: `db0c5428 b5ca7ac4 58f5dbd5 bf45cf4b edb79dae 80a900e0 d35bdbdc 8e5c0c38 58490d8a b10624e5 8b9c3697 8698868d`.
  - FRESH12 was drawn with seed 20261014 by stratum:
    - 4 ARC-Prize-xhigh-solved tasks with order score 2;
    - 4 xhigh-solved tasks with score 1 or 0;
    - 2 solved only at another level;
    - 2 never solved (one with score 2, one with score 1 or 0).
  - MIX20 has 20 tasks / 23 inputs, with scores 2/1/0 = 14/5/1, and a 3 h budget.
  - CPU mixed, drain and rerun all PASS: 23/23 and 259/259 finished; admission follows the order rank; prompts median 4,530 tokens, max 11,962; no thinking budget reduced.
- **Update 3 (same day): the ORDER2 default commit profile is now `EVAL120`, the full run** (sha `b62ae567aa4d8af2…`).
  - Scope: all 120 evaluation tasks / 172 inputs, scored.
  - Clock: the same as the hidden run. Budget 11 h 50 min; admit until 694 min (11:34), drain at 696, hard stop at 704.
  - Order scores 2/1/0 = 52/53/15 tasks. Prompts: median 4,336 tokens, max 16,400; no thinking budget reduced.
  - MIX20 and EVAL12 remain available through `Q27X_PROFILE`.
  - CPU mixed (172/172, admission in rank order), drain and rerun (259/259) all PASS.
  - Estimated wall time: ~9–10.5 h. Worst case, with every unit hitting the cap: ~11.3 h.
- **GPU result (`hosen42/q27-xhigh-e10-order2`, 2026-10-14): 58/172 = 33.72 %, task-normalised 36.94 %.**
  - All 172 units finished; last done at 534 min. 0 preemptions; KV max 87.3 %.
  - 19 of the 58 hits came after 49,152 thinking tokens.
  - Analysis: `Experiment Results/Analysis/GPU EVAL120 ORDER2 86k analysis (2026-10-14).md`.

## R4T (2026-10-14): R4 + full thinking saved

- **What it is:** `Experiment Notebooks/Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R4T.ipynb`, sha `1f8fd03e…`, built by `Build/build_e10_r4t.py`.
  - Cell 7 now writes the complete generation of every iteration to `q27x/poetiq_thinking/<task>_e<expert>_i<iteration>.txt`. That is the thinking, the forced-close stop text and the answer/code, plus a header line with the tokens and mode.
  - **Logging only:** generation, prompts, verification and voting are unchanged. A write failure is logged and never stops the solver.
- **EVAL22 variant rebased on R4T:** `Q27-XHIGH-v1_4_4_E10_86K_POETIQ_R4T_EVAL22.ipynb`, sha `d9d446ad…`.
- **CPU harness** (sandbox stubbed, thinking 77,824): 120 iterations and 120 thinking files, each containing `</think>` and the code block. Solver complete, no error.

## ORDER2 + LOOPSTOP (2026-10-15): `Experiment Notebooks/Q27-XHIGH-v1_4_4_E10_86K_ORDER2_LOOPSTOP.ipynb`

- **Build:** sha `8f8c732d765eb1fd…`, from `Build/build_e10_order2_loopstop.py`.
- **One change vs ORDER2:** a degenerate exact-repetition thinking loop is force-closed early, and its slot is freed. The check runs only after 49,152 thinking tokens; grid rows are never flagged.
- **Offline:** 5/5 real loops found in 333 saved tails; 0 false positives; 0 flags in 41 full traces.
- **CPU harness:** loop, loopoff, mixed, drain and rerun all PASS.
- **Details:** `Experiment Review/CPU validation ORDER2 LOOPSTOP (2026-10-15)/README.md`.
- **Status:** not pushed; a GPU run needs the user's go.

## ORDER2 + COVER-FIRST (2026-10-15): `Experiment Notebooks/Q27-XHIGH-v1_4_4_E10_ORDER2_COVERFIRST.ipynb`

- **Build:** sha `de9b98e82ea15e95…`, from `Build/build_e10_coverfirst.py`.
- **Why:** E9 (86k for every unit) scored LB 22.92, below 49k (26.94 / 27.92). Most likely the hidden units did not finish in time.
- **One change vs ORDER2:**
  - pass 1 gives every unit 49,152 thinking;
  - after every unit is admitted, traces that hit 49,152 are resumed token-exact from their own thinking up to 77,824, in ORDER2 rank order;
  - a KV guard prevents preemption;
  - the deeper answer becomes attempt_1 and the pass-1 answer is kept as attempt_2.
- **CPU harness:** cover, coveroff, coverdrain, coverrerun, mixed, drain and rerun all PASS. Details: `Experiment Review/CPU validation ORDER2 COVERFIRST (2026-10-15)/README.md`.
- **Status:** not pushed; a GPU run or submission needs the user's go.

## S1-60K (2026-10-16): Poetiq R5 + fixes + ORDER2 queue + 60k thinking (submission build, Experiment 11 base) and Run 1 (77.8k at T 0.6 on EVAL120)

The builder is `Experiment Notebooks/Build/build_e10_s1.py`. Its source is the exact R5 notebook, which is the version
submitted for LB 21.39.

| Notebook | Thinking | Commit run | Temperature | sha256 |
|---|---|---|---|---|
| `Q27-XHIGH-v1_4_4_E10_S1_60K_POETIQ_FIX_ORDER2.ipynb` | **60,000** + 8,192 | `S1COMMIT4`: 4 quick eval tasks, about 1 h, then submit | 1.0 | `ed530071…` |
| `Q27-XHIGH-v1_4_4_E10_S1_T06_EVAL120.ipynb` | 77,824 + 8,192 | EVAL120 on the hidden-run clock | **0.6** | `7231dded…` |

**Why 60k (user, 2026-10-16):** S1 is the base for Experiment 11, and its LB measures Poetiq at a 60,000-token threshold against R5's 77,824 (LB 21.39), to find the best LB token threshold. 60,000 is above the 49,152 floor. Simulation from R5's per-request records: about 208 of the 240 hidden tasks get iteration 1 at 60k, against about 177 at 77.8k. The old 77.8k S1 file (`..._E10_S1_POETIQ_FIX_ORDER2.ipynb`, `328a078f…`, no F2b) is superseded.

**Unchanged from R5:**
- 8,192 answer tokens (thinking: 60,000 in S1, 77,824 in R1);
- 1 expert, up to 10 iterations;
- prompts, voting, sandbox and engine;
- full thinking saved;
- the hidden-run clock.

**Changes:**
- **Q:** iteration 1 runs in ORDER2 order. Failed tasks still go to the back, so every task gets iteration 1 before any iteration 2. Iteration-1 score does not predict iteration-2 success: 4 of the 12 iteration-2 solves had an iteration-1 score of 0.
- **F1:** the last ```python block that defines `transform` is executed.
- **F3:** a train-passing program that fails on one test input is kept, and attempts are picked per input. Latent: R5 had 0 such programs among its 71 train-passing programs, so the earlier "5 inputs" figure was wrong.
- **A2:** attempt_2 never repeats attempt_1. It is the next distinct grid, otherwise the test input or its mirror. R5 repeated attempt_1 on 134 of 172 inputs.
- **F2:** verifications still pending at the hard stop are harvested.
- **F2b (GPT audit, 2026-10-16):** after the harvest window, still-queued verifications are cancelled (`pool.shutdown(wait=False, cancel_futures=True)`), so leaving the thread pool waits only for the ≤ 4 running jobs. Before, it waited for every queued job.
- **C1:** a bad output closes only its own unit; the solver keeps running.
- **KV admission gate:** no new request above **0.95** KV usage while ≥12 are running. The user set 0.95 on 2026-10-16 because 0.85 is too strict. In R5, KV was above 0.85 for 18 % of the run and above 0.95 for only 3 %; the 9 preemptions happened at about 0.995.
- **Weight prefetch:** model weights are read while pip runs.
- **CLOCK (user, 2026-10-16: "you are wasting 17 min").**
  - R5 ended at 11:43:40, only 16 s after its solver, so 16 min of the 12 h went unused.
  - The clock now starts at kernel start (R5's cell 0 began 7 s after it).
  - The hidden and EVAL120 clocks are: last admission **11:45**, drain **11:47** (the 8-minute drain is kept), hard stop **11:55**, verify harvest until **11:56**. This is +11 min of solving, with a margin of about 3.5 min.

**CPU harness** (`Build/harness/harness_s1.py`) runs the real cells 0, 1, 3, 4 and 6 with the real tokenizer and eval data, a fake vLLM engine and a fake sandbox. Scenarios: mixed, kvgate, hardstop, rerun, slowverify (8 × 4 s verifications on 4 workers pending at the hard stop). All pass for both notebooks.
- **F2b control:** the old S1 without F2b ends its solver after 8.4 s in slowverify (FAIL); S1-60K ends after 4.4 s.
- **Negative control (R5):** it crashes on an unexpected request id, does not follow ORDER2, and **loses all 8 verifications pending at the hard stop**. S1 harvests all 8. This bug was latent in the real R5 run: its solver ended 43 s before the hard stop with 0 pending.

**GPT audit (2026-10-16), validated:**
- **F3A (ranking regression):** true in a synthetic case, but 0 such cases in R5 data. GPT's fix (non-passing programs first) would lose more, because 77 of R5's 78 hits came from the top program. Not adopted.
- **F2A (unbounded pool wait):** true. Fixed by F2b.
- **CLOCK:** R5's log clock matched Kaggle's runtime within 2 s. Finalization took 16 s for 120 tasks and is estimated at ≤ 40 s for 240. The margin after the 11:55 hard stop is about 4 min. Kept.


## D66 deep run (2026-10-16, user go): the 66 eval tasks R5 did not fully solve

**Notebook:** `Q27-XHIGH-v1_4_4_E10_D66_HARD_DEEP.ipynb` (sha `49c5c67f…`; earlier builds `f35cfae3…` and `97d96429…`), built by `build_e10_s1.py` (variant D66). It is not a submission: its commit run is the 12 h measurement.

**Per-iteration evidence (GPT audit F1, 2026-10-16).** Each record in `poetiq_iterations.jsonl` also stores:
- `test_grids`, `test_errors` and `test_usable`;
- `train_soft_scores`;
- `test_correct`: per input against the eval solutions, written after selection; `None` in a rerun;
- `t_done_sec`.

These fields are logging only. R5 kept only its final attempts.

**Independent review (2026-10-16).** Verdict: GO AFTER FIXES, no blocker. Added, D66 only:
- `t_admit` and `p1_finish` (drained vs cap);
- `E10_SCHED['unharvested']`;
- `poetiq_attempt_curve.jsonl`: the deployable two-attempt score after every commit;
- parser `except Exception`.

Not adopted: re-queueing on `unit_error` (loop risk). The harness slowverify check was made robust (6 s jobs; F2b 7.8 s vs the old S1 14.2 s). GPT's F2 (sandbox fail-fast) was already in the code: cell 6 sets `_LADDER = []` when the self-test fails, so no engine loads.

**What changes against R5** (the S1 fixes are included):
- Task list `HARD66`: 57 zero-score + 9 partial tasks, 103 test inputs. The R5 fully solved tasks are left out.
- Fresh seeds: `SEED` 20261016 (R5 20261005) and `E10_POETIQ_SEED` 1 (R5 0, the example order).
- Everything else is as in R5: 77,824 + 8,192 thinking, T 1.0, up to 10 iterations, 1 expert, the hidden-run clock (admit 11:45, drain 11:47, hard stop 11:55).

**Why (new information R5 cannot give):**
- In R5 no task got more than 2 iterations; 21 zero-score tasks got only 1.
- Questions:
  1. How many R5-failed tasks does a fresh attempt solve? Is the failure a model limit or sampling?
  2. What is the value of iterations 3+ per slot-hour on hard tasks?
  3. Do the 15 train-passing but test-wrong tasks get a correct program on a new attempt?
- Together with R5 this gives the compute → score curve: the eval score with k attempts per task. That curve decides:
  - how far Poetiq can go when Experiment 11 gives 2–3× speed;
  - whether spare time goes to more Poetiq iterations or to DIRECT attempt_2 (DIRECT data exists).
- Expected volume: about 170 requests (1.07 slot-hours each in R5), about 2.6 iterations per task.

**Checks:**
- `scratchpad/s1_build/check_profile_d66.py` runs the real config and task cells with the default profile. It verifies the profile, the 66 ids, the seeds, the budget, the temperature and the clock. PASS.
- The harness passes mixed, kvgate, hardstop, rerun and slowverify.

## L66 long-thinking run (2026-10-16, user go): D66 with 122,880 thinking on the same 66 tasks

**Notebook:** `Q27-XHIGH-v1_4_4_E10_L66_HARD_120K.ipynb` (sha `545a7e8e…`), builder variant L66. It is not a submission.

**Why:** in R5, 74 % of the hard tasks' iteration-1 requests were cut at the 77,824 cap (49 of 66), against 33 % for the solved tasks. We have never run above 78k. Paired with D66 (77,824) on the same tasks with fresh seeds, the two runs compare more attempts against longer thinking per attempt.

**Differences from D66** (cell 2 only, plus the header):

| Setting | D66 | L66 |
|---|---|---|
| `THINK_BUDGET` | 77,824 | 122,880 |
| `MAX_MODEL_LEN` | 106,496 | 155,648 (otherwise `phase1_cap` clips to about 93k) |
| `MAX_NUM_SEQS` | 16 | 12 |
| KV gate floor | 12 | 8 |
| `SEED` / `E10_POETIQ_SEED` | 20261016 / 1 | 20261017 / 2 |

**Checks:**
- `check_profile_l66.py`: PASS. The phase-1 cap is the full 122,880 for prompts up to 24k, and the engine formula gives 12 slots.
- The harness passes mixed, kvgate, hardstop, rerun and slowverify.
- D66, S1-60K, R1 and A60 are byte-unchanged.

## D66 / L66 analysis, ready before the runs end (2026-10-16)

**Pull the outputs:**
- Script: `scratchpad/push159/pull_run_outputs.py <owner/slug> <dest>`.
- Slugs: `foysalemonshanto/q27-xhigh-e10-d66-hard-deep` and `…-l66-hard-120k`.

**Analyse:** `Experiment Notebooks/Build/analysis/analyze_d66_l66.py --r5 <R5 q27x> --d66 <D66 q27x> --l66 <L66 q27x> --solutions <eval solutions> --out <dir>`. It reports:
- per run:
  - requests, tokens, slot-hours and finish types;
  - the deployable two-attempt result;
  - the oracle result (any program output correct), split into all inputs and the 94 R5-wrong inputs;
  - a per-iteration table (new correct inputs per slot-hour);
  - deployable score by hour;
  - outcome categories for the 15 R5 train-pass/test-wrong tasks;
- D66 vs L66, paired at iteration 1:
  - a 2×2 table;
  - L66 natural closes between 77,824 and 122,880 tokens and their correctness;
  - deployable inputs per M tokens.

**Tested:** `make_fake_runs.py` synthetic runs in the exact record format. The reported request, oracle and deployable counts match the fixture's known values.

## D66 / L66 results (2026-10-17)

Both runs finished cleanly: no error, 0 unit errors, 0 unharvested verifications, KV gate never blocked. Outputs: `Experiment Results/Kaggle Runs/GPU HARD66 D66 77.8k …` and `… L66 122.9k …`. Analysis: `Experiment Results/Kaggle Runs/D66_L66_analysis/`.

All numbers are on the 66 tasks / 103 inputs R5 did not fully solve; R5 had 9 of these inputs right.

| | D66 (77,824, 16 slots) | L66 (122,880, 12 slots) |
|---|---|---|
| Inputs right (deployable = oracle) | **28** | **34** |
| Of the 94 R5-wrong inputs | 22 | 29 |
| Tokens / wall / decode speed | 10.80 M / 42,231 s / 256 tok/s | 10.66 M / 41,968 s / 254 tok/s |
| Requests (iterations 1 / 2 / 3) | 151 (66 / 52 / 33) | 123 (66 / 41 / 16) |
| Iteration-1 requests cut at the cap | 49 / 66 | 9 / 66 |
| Iteration-1 fully correct tasks | 7 | 11 |
| Train-passing programs that are test-correct | 14 / 33 | 15 / 43 |
| Inputs right per M tokens | 2.59 | 3.19 |

Per-input comparison: only D66 16, only L66 22, both 12. McNemar exact p = 0.42, so L66 > D66 is **not significant**. L66 is at least as good per GPU-hour.

**Main finding: independent attempts are highly complementary.**
- D66 ∪ L66 = **50 / 103** inputs, against 28 and 34 alone; R5 ∪ D66 ∪ L66 = 51.
- On hard tasks, failure is mostly sampling, not a model limit: a fresh attempt recovered 22–29 of the 94 R5-wrong inputs.
- A second independent attempt per task, used as attempt_2, would collect most of this union. R5's attempt_2 added 1 input.

**Other findings:**
- **Train pass is not test correct on hard tasks.** Only 35–42 % of train-passing programs were test-correct.
  - Of the 15 R5 train-pass/test-wrong tasks, 6 (D66) and 9 (L66) overfit again.
  - Poetiq stops a task at the first train pass, so an overfit program ends it.
- **Iteration 2 (with feedback) is about as productive as iteration 1:**

  | Correct inputs per slot-hour | Iteration 1 | Iteration 2 |
  |---|---|---|
  | D66 | 0.14 | 0.22 |
  | L66 | 0.27 | 0.25 |

  Iteration 3 gave about 0, because the drain cut it.
- **Cap-forced answers rarely succeed:**
  - D66: 5 of 49 cut at 77.8k.
  - L66: 1 of 9 cut at 122.9k.
  - L66's 39 natural closes between 78k and 123k gave only 3 fully correct tasks.

## M2H medium-effort probe (2026-10-17, user go: "a 2-hour short version on our last submitted notebook")

**Notebook:** `Q27-XHIGH-v1_4_4_E10_M2H_MEDIUM_2H.ipynb` (sha `95d8bacd…`), builder variant M2H.
- It is the submitted S1-60K (LB 34.58) with `REASONING_EFFORT = "medium"` (env `Q27X_REASONING_EFFORT` overrides).
- Profile `MED2H`: all 120 eval tasks on a 2 h clock (last admission 1:48, drain 1:50, hard stop 1:58).
- D66-style per-iteration logging (`test_correct`, attempt curve).
- Not a submission.

**The Qwen3.8 chat template:**

| Effort | System instruction added |
|---|---|
| xhigh (default) | "Please think carefully …, validate key assumptions, consider plausible alternatives" |
| **medium** | **none**: the model's natural thinking |
| low | "Keep your thinking brief" |

The idea comes from the public delongyang notebook (medium effort; claimed 4 h, but measured on the placeholder training tasks; no published eval score; non-standard submission format).

**Measures:**
- thinking tokens per request;
- tasks reached in 2 h;
- iteration-1 accuracy, paired by task with R5 (xhigh 77.8k).

**Checks:**
- `check_profile_m2h.py`: profile, medium effort, 120 tasks, 60,000 thinking; the rendered prompt has no xhigh instruction and still opens `<think>`. The clock check's expectation was wrong (1:55 instead of 1:58), not the notebook.
- The harness passes mixed, kvgate, hardstop, rerun and slowverify.

**Update (user, 2026-10-17): 3 h, not 2 h.**
- Profile `MED3H`: last admission 2:48, drain 2:50, hard stop 2:58.
- File: `Q27-XHIGH-v1_4_4_E10_M3H_MEDIUM_3H.ipynb` (sha `83e48f54…`). The 2 h file was removed.
- `check_profile_m2h.py` and the harness (5 scenarios) PASS.

## W3H low-effort probe (2026-10-17): replaces M3H (medium)

- **Why medium was dropped:** ARC Prize's verified results for Qwen3.8-27B on ARC-AGI-2 semi-private are xhigh 42.4 %, **medium 13.2 %**, low 22.8 % and none 1.5 %.
  - Medium even used more tokens than low. The chat template adds no instruction for medium.
  - The user chose low: "low দিয়ে চালাই চলো" (let's run it with low).
- **Notebook:** `Q27-XHIGH-v1_4_4_E10_W3H_LOW_3H.ipynb`, sha `7446340e…`. Built by `build_e10_s1.py`, variant W3H; the medium builder is backed up as `build_e10_s1.m3h_backup.py`.
- **Settings:** S1-60K (LB 34.58) with `reasoning_effort` "low". The template adds "Keep your thinking brief and focused…".
- **Run:** profile `LOW3H` (all 120 eval tasks); clock admit 2:48, drain 2:50, hard stop 2:58; 60,000 thinking cap; T 1.0.
- **Checks:** profile check and all 5 harness scenarios (mixed, kvgate, hardstop, rerun, slowverify) PASS, with the same check list as M3H.
- **Pushed:** `foysalemonshanto/q27-xhigh-e10-w3h-low-3h` v1, SAVE_AND_RUN_ALL (the commit is the measurement), 2026-10-17. Not a submission.
- **Compare against R5 (xhigh), task by task:** tokens per request, tasks reached in 3 h, iteration-1 accuracy, and solves that xhigh misses. On ARC Prize's public eval, low solved 8 tasks that xhigh did not.
- The M3H medium notebook was moved to `Experiment Notebooks/superseded/`.
- **2026-10-17 update:** the user cancelled W3H v1 about 13 minutes after it started, before any results (status CANCEL_ACKNOWLEDGED). It may be rerun later.
- **Re-committed on esfersami50** (user): `esfersami50/q27-xhigh-e10-w3h-low-3h` v1, SAVE_AND_RUN_ALL, same sha `7446340e`, 2026-10-17.

### W3H result (esfersami50 v1, COMPLETE, 2026-10-17)
Outputs and a comparison script are in `Experiment Results/Kaggle Runs/GPU EVAL120 W3H low 3h (...)`.

- **Score:** **17/172 query hits (9.88 %)**; 11 tasks fully solved, 15 with any hit.
  - R5 (xhigh, 12 h): 78/172 on 63 tasks.
  - ARC Prize low (public eval): 19/120 tasks.
- **Coverage:** iteration 1 reached all 120 tasks; the last admit was at 2.79 h.
  - 125 requests, 3.91M tokens, 400 tok/s (16 slots, shorter contexts).
- **Thinking length:** median 30.3k tokens per request, against R5's 77.8k cap. **0 of 120 hit the 60k cap**; R5 had 70 of 120 at its cap.
  - **Correction (code check, 2026-10-17):** low does NOT finish its thinking.
    - In 93 of 120 requests (78 %) the model emitted **EOS in the middle of its thinking**, with no `</think>`, at a median of about 30k tokens. 83 of the 93 stopped mid-sentence or mid-grid.
    - The harness handles this like a cap hit: it strips the EOS, appends the forced-close text and asks for the answer in phase 2 (`e10_start_phase2`). So most low answers came from unfinished thinking.
    - With xhigh this never happens (D66: 0 of 151 forced_close+stop; all its forced closes were cap hits).
    - Effort, sampling and scoring were applied correctly (`reasoning_effort` low in `e10_render_request`), so this is not a settings bug. It is a low-mode behaviour the harness was not adapted to.
    - A fairer low test would resume thinking after a premature EOS. That was not run, because the ceiling (ARC Prize low 19.6 % public vs xhigh 38.75 %) stays far below xhigh.
- **Train-pass at iteration 1:** low 16 (13 %), xhigh R5 55 (46 %). Low's train-pass precision is 10/16.
- **Complementarity: none.** All 15 W3H-solved tasks are inside R5's 63.
- On the 16 tasks low train-verified, low used 0.42M tokens against R5's 0.90M.
- **Decision:** low gives no new solves and only about 10 % accuracy, so xhigh stays. Low is not useful as a replacement or as a complement. An end-of-run fallback is about break-even with xhigh time, so it is not pursued.

## S49: S1-60K with thinking 49,152 (built 2026-10-18, user "প্রার্থী ২"; not pushed yet)

- **File:** `Q27-XHIGH-v1_4_4_E10_S1_49K_POETIQ_FIX_ORDER2.ipynb`, sha `1cae1d59…`, builder variant `S49`.
- **Update 2026-10-18 (user):** the ORDER2 tie-breaker "smaller total grid cells first" was added too, so S49 = **two changes** (49,152 cap + cells tie-break). An LB change cannot be split between them. The queue order equals the lever-analysis order (verified).
- **One change from the submitted S1-60K (LB 34.58):** `THINK_BUDGET` 60,000 → **49,152** (the user's floor). The notebook diff is exactly 2 lines: the note and THINK_BUDGET.
- **Commit:** `S1COMMIT4`, then submit.
- **Checks:** the S1 harness (mixed, kvgate, hardstop, rerun, slowverify) PASS with the same check list as S1-60K.
- **Why:** the LB test of the next cap step. 77.8k → 60k gave 21.39 → 34.58, but together with the S1 fixes.
- **Prior from simulation:** the ORDER2 DIRECT cap simulation predicts only about ±1–1.5 between 49k and 60k. That is DIRECT data, not Poetiq, so the LB result is the real answer.
- **2026-10-18: Quick-saved** to `foysalemonshanto/q27-xhigh-e10-s1-49k-poetiq` v1 (sha `1cae1d59`). The save took about 1 minute; no GPU run. The user commits (Save & Run All) and submits from the editor.
- **Not added (decided 2026-10-18):** the second-attempt policy (near-miss feedback first; a fresh attempt after a train pass).
  - Iteration-1 tokens: 7.18M at 60k vs 6.13M at 49k for eval-120, i.e. 49k is 0.853 of 60k.
  - For hidden 240 that is about 12.3M needed against an about 11.7M budget, so about 96 % coverage and no spare time. Retries would hardly run, so the policy gives about 0 LB.
  - It belongs to the E11-speed build, where about 2.9M tokens would be spare.
