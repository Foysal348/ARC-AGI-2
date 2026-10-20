# Q27-XHIGH v1.4 RUN4 → R5 Safe Patch: Bug Fixes & Score Experiments

**Date:** 2026-10-06  
**Original:** `Q27X_v1_4_RUN4_FOR_GPT_2026-10-06.zip`  
**Patched notebook:** `Q27X_v1_4_R5_SAFE_PATCH.ipynb`  
**Status:** CPU OFFLINE TESTED; **GPU / Kaggle rerun NOT VALIDATED**. This is not a claim of leaderboard improvement.

## প্রথমে কোন সমস্যা কোথায় ও কী সমাধান

| Priority | Original location | Evidence | R5 concrete fix | CPU test / GPU validation |
|---|---|---|---|---|
| **P0** | `CELL 4 -> make_attempts()` | attempt_2 usually test input / mirrored input; 0 incremental hits in RUN4 | Added `merge_second_candidate()`; an opt-in low-effort independent model reviewer (`run_optional_audit()`) in CELL 6 generates a distinct possible `attempt_2` and never overwrites `attempt_1`. Default **OFF** | CPU simulated step API PASS; actual Kaggle pass@2 **unknown** |
| **P0 operational** | `CELL 0`, `CELL 6 -> write_submission()` | RUN4 commit-mode `submission.json` is identity, NOT the 22 EVAL12 predictions | Explicit `mode_contract.json` and `q27x/eval_predictions.json` for non-rerun EVAL. Real competition rerun still writes to `submission.json` as before | CPU write behavior PASS |
| **P1** | `CELL 4 -> parse_grid()`, `CELL 6 -> finish()` | 247ef758#0 15 rows with one width 21; 6e4f6532#0 **31 rows**, one width 29 | Added raw-grid diagnostics to each record; no blind dimension guessing; optional model regeneration available | Both archived malformed examples PASS. **No automatic semantic repair** |
| **P1 logging** | `CELL 5` | Archive `vllm_frontend.log` was 0 bytes | Import vLLM before log handler; emit and flush `Q27X_LOG_PROBE_FRONTEND` marker | Syntax PASS; actual vLLM logger probe needs Kaggle check |
| **P1 metric integrity** | `CELL 7` | If a second candidate is added later, first-pass `RECORDS` score would be stale | Evaluate `FINAL` merged attempts; add `eval_incremental_pass2_queries` | Static compile PASS; test score with GT after actual run |
| **P1** | `CELL 6` | Long first-pass reasoning / 18 force-closes | Existing first-pass model pipeline kept intact; second-pass prototype can test alternate solution on leftover time; controlled budget ablation proposed below | **Not solved by this patch** |
| **P1** | `CELL 3 + CELL 6` | Multi-query tasks repeat 49k thinking independently | See independent, train-verified task-level Python solver experiment design below | **Architecture experiment; not implemented in R5** |
| **P1** | `CELL 7 projection`, `CELL 5 ENGINE_CONFIG` | 22-query throughput projected across 259-query hidden batch | Keep current 16 slots / MTP3 / FP8 KV; compare sustained throughput and 12h coverage experimentally | **Not GPU verified** |

## কীভাবে patch ব্যবহার করবে

1. Notebook `Q27X_v1_4_R5_SAFE_PATCH.ipynb` Kaggle-এ upload করো, original GPU, datasets, vLLM setup অপরিবর্তিত রাখো।
2. প্রথম রান শুধু `EVAL12` mode-এ original baseline defaults-এ করো; `Q27X_ENABLE_AUDIT=0` (default)। Output-এর `q27x/mode_contract.json`, `q27x/eval_predictions.json`, `q27x/run_report.json`, `q27x/records.jsonl` মিলিয়ে দেখো। `q27x/vllm_frontend.log`-এ `Q27X_LOG_PROBE_FRONTEND` আছে কি না দেখো।
3. দ্বিতীয় local EVAL run-এ optional reviewer test করার জন্য **CELL 1-এর আগে** environment variables set করো (e.g. first code cell at the beginning):

```python
import os
os.environ['Q27X_ENABLE_AUDIT'] = '1'
os.environ['Q27X_AUDIT_MAX_QUERIES'] = '8'
os.environ['Q27X_AUDIT_THINK_BUDGET'] = '4096'
os.environ['Q27X_AUDIT_ANSWER_MAX'] = '4096'
os.environ['Q27X_AUDIT_CONCURRENCY'] = '2'
```

The audit queue starts **only after** the original scheduler finishes its first-pass queue. Therefore this does *not* guarantee a second pass in the 259-query hidden workload; it is a deliberately conservative experiment. It may add runtime and is intentionally disabled by default. Changing the four env variables is an experiment, not a verified performance upgrade.

4. Compare the **same pre-registered hold-out task set** under baseline vs opt-in reviewer. Measure `eval_pass1_queries`, `eval_incremental_pass2_queries`, `eval_score_pct`, `audit_stats`, and wall-time. Promote only if there is a measurable improvement, with no serious coverage regression. Do not treat the 12-task EVAL12 as a sufficient generalization benchmark.
5. Only after those tests, run a Kaggle Save & Run All competition-rerun smoke. Confirm printed `MODE: COMPETITION RERUN (hidden test)`, `mode_contract.json: MODEL_PREDICTIONS_WHEN_FINISHED`, and final `submission.json` contains every challenge task ID, with test query order and both attempt keys correct. **Do not submit `run4/output/submission.json` as model output.**

## The highest-impact unsolved issue: a train-verified Python-rule solver (R6)

Run4 has **2/22** exact answers; two naturally finished outputs missed by only one cell and 18 of 22 hit forced close. The intended scoring fix is to stop relying exclusively on manual text-to-grid transcription.

**Implement as a separate branch, not a silent change to the successful GPU engine:**

1. For each task, ask model to emit a small `def solve(grid): ...` function, strictly transforming one input grid to one output grid. Give model training pairs; do NOT show public evaluation solution keys or hidden ground truth.
2. Parse the function with `ast`, restrict constructs/imports/builtins, enforce program length and loop/time limits; execute in a low-privilege **isolated subprocess/container** with CPU/memory caps. **A regular subprocess and AST checks are not a full security sandbox.**
3. Test on *every train pair*, using exact grid equality. Accept a rule only when all train examples are matched, all outputs are rectangular 1–30 cells per dimension, and runtime is within cap. Train verification is a necessary filter, not proof of correct test prediction.
4. Reuse the same verified solver across all test queries of the task. Any unverified or failing rule must retain the model-only fallback; do not propagate an unverified pattern to other test queries.
5. Keep two attempts independent: program-generated grid as one candidate if train verified; model-only reasoning output as a distinct candidate, rather than running two full 49k rollouts by default.
6. Score on a held-out *training-derived* task set with known outputs never exposed during rule generation. Track: exact pass@1, incremental pass@2, overfitting-to-train, failure/timeout rate, total wall time, completed queries under the 12 h deadline.

**Estimated potential:** high but *unmeasured*. A rule-program loop may improve accuracy, may be too slow, or may overfit small training examples. It is not included in the R5 notebook.

## Prioritized validation matrix

| Test | Variant A | Variant B | Acceptance criteria |
|---|---|---|---|
| Parser/reviewer | Original | R5+optional review | Valid outputs; unchanged attempt_1; positive incremental pass@2 on holdout |
| Reasoning budget | 49,152 xhigh | 16,384 / 24,576 / 32,768 xhigh | Higher `exact solved per GPU-hour`, not token/s alone |
| Reasoning effort | xhigh | medium / low | Score and number of completed queries within identical wall-clock budget |
| Candidate generation | Text grid only | Train-verified Python solver | Higher test exact accuracy with no invalid/risky program execution |
| KV quality | fp8_e4m3 | auto/BF16 KV | No quality drop outweighing throughput cost; same prompts, seeds |
| Engine | 16 slots MTP3 cache-off | 18/20 slots; MTP1; cache-on (one variable at a time) | No OOM/preemption spirals; higher complete-batch solved count |

## CPU reproducibility

```bash
python Q27X_R5_OFFLINE_TESTS.py
```

Expected: **6/6 PASS**. It checks notebook syntax, archived malformed-grid diagnostics, parser, second-candidate merge, commit vs rerun output, and an entirely mocked two-phase vLLM reviewer. **It does not load Qwen or benchmark GPU accuracy**.

## Non-negotiable submission guardrails

- Kaggle rules require `submission.json` containing exactly two attempts per test query, with every test task ID present; GPU notebook run time at most 12 hours and internet off. Refer to the official competition overview and rules.
- Do not use EVAL ground truth to construct input-specific inference rules, judge hidden cases or correct grid cells. Use it only for offline evaluation; train-pair verification is safe but can still overfit.
- Never blindly remove a malformed grid's extra digit based on test-set ground truth.
- Don't trust a throughput projection until a sustained nearly-full 259-query benchmark demonstrates coverage margin.
- This R5 patch should not be assumed superior to the team's separate NVARC 32.64 baseline without a leaderboard-comparable evaluation.
