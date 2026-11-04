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
