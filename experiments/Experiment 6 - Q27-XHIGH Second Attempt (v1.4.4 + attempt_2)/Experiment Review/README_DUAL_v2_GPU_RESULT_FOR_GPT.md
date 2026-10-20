# Experiment 6: dual-answer v2 GPU EVAL12 result, for review (2026-10-12)

**Request: a deep, unrestricted review.** Audit the notebook, the run outputs and Claude's analysis. Every conclusion and suggestion below is Claude's suggestion only; challenge or replace any of it.

**Standing user rules:**
- never less than 49,152 continuous xhigh thinking + 8,192 answer per test input;
- v1.4.4 (LB 26.94) is the base;
- nothing is pushed or submitted without the user's approval.

## What ran

| Item | Value |
|---|---|
| Notebook | `notebook/Q27-XHIGH-v1_4_4_DUAL_ANSWER_EVAL_v2.ipynb` (sha256 `1f8d7ea4…`), audited GO by Claude before the run (`notebook/FINAL_AUDIT_DUAL_ANSWER_v2_2026-10-12.md`) |
| Kaggle | `foysalemonshanto/q27-xhigh-dual` version 1: 4× L4, internet off, pinned docker image, the same inputs as the LB kernel |
| Mode | commit run, EVAL12 (12 tasks / 22 test outputs); second branch ON, policy `protect_supported` |
| Time | 18:47 → 20:14 (87 min): setup 705 s, primary scheduler 3,608 s (288.5 tok/s, 16 running, 0 preemptions, MTP acceptance 0.664), second stage 880 s; prefix-cache hit rate 0.844 |

## Result

| | Hits / 22 |
|---|---|
| Primary (attempt_1) | **1** (136b0064#0) |
| Dual (attempt_1 + new attempt_2) | 1 |
| Marginal from the second answer | **0** (no fallback hit lost; `primary_unchanged_all` = true) |
| Counterfactual policies: keep_fallback / replace_any / protect_supported | 1 / 1 / 1 |

**Second-answer stage:**
- 22 submitted and completed.
- 3 duplicated the primary grid (natural closes).
- **12 were invalid (ragged rows inside the fenced grid)**, mostly after forced closes on large grids.
- 7 were valid and novel; they replaced attempt_2 and all were wrong.
- 20a9e565#0 used the full 8,192-token answer cap, which took most of the 880 s.

## Comparison with run 7 (the same v1.4.4 primary code and seeds, 3/22)

| Unit | Run 7 | This run |
|---|---|---|
| 136b0064#0 | natural 25,934, **hit** | natural 25,113, **hit** |
| 247ef758#0 | natural 36,770, **hit** | forced close, ragged grid (no grid) |
| 4c3d4a41#0 | forced close, **hit** | natural 26,669, wrong |
| 4c3d4a41#1 | forced close, wrong | natural 41,012, wrong |
| 247ef758#1 | natural 46,470, wrong | natural 38,748, wrong |
| Primary answers without a valid grid | 0/22 | **5/22** (ragged) |

**EVAL12 primary results across identical-method runs:** run 4 = 2, run 6 = 2, run 7 = 3, this run = 1. The union of correct units is {136b0064#0, 4c3d4a41#0, 247ef758#0}.

## Claude's observations (suggestions; please verify)

1. Re-sampling only the answer from the same reasoning gave no value: duplicates on natural closes; ragged or wrong grids on forced closes.
2. **Run-to-run variance is large** (GPU batching and MTP nondeterminism with the same seeds). One EVAL12 run cannot resolve ±1–2 hits.
3. **Independent full samples** (a new seed and the full 49,152 thinking) are the only evidence-backed source of second-attempt hits: two runs together beat each run alone.
4. **Ragged grids** are a distinct failure mode (5 primary and 12 second answers). Repairing them by padding or truncation would be a guess, so it is not proposed.
5. **Proposed next experiment (Experiment 7, plan included as context):**
   - A: NVARC easy-first ordering (seeds unchanged);
   - B: a KV-aware admission controller instead of a fixed 16;
   - **C (proposed addition):** attempt_2 from an **independent second sample**, run only on spare capacity (the primary queue empty, KV guard), low priority;
   - evaluated on EDA PROXY-48 (56 outputs) to reduce noise.

**Questions for the reviewer:**
- Is the diagnosis right?
- Is something wrong in the notebook or the analysis?
- What is the best next experiment for attempt_2 and for the overall LB, given this evidence?
- Is the Experiment 7 plan (with Part C) sound?

## Files

| Path | What |
|---|---|
| `notebook/Q27-XHIGH-v1_4_4_DUAL_ANSWER_EVAL_v2.ipynb` | the notebook that ran |
| `notebook/FINAL_AUDIT_DUAL_ANSWER_v2_2026-10-12.md` | Claude's pre-run audit |
| `run_output/` | everything Kaggle returned: the full log, `q27x/run_report.json`, `records.jsonl`, `records_final.json`, `second_branch_report.json`, `second_branch_candidates.json`, `eval_generated_dual_submission.json`, `answers/*` (primary: thinking tail + answer), `second_answers/*`, the vLLM metrics and logs, `submission.json`, the bundle |
| `analysis/GPU EVAL12 dual v2 analysis (2026-10-12).md` | Claude's analysis |
| `reference_run7_v1_4_4/` | run 7 `records.jsonl` and `run_report.json`, for the comparison |
| `context/Experiment7_plan_README.md` | the Experiment 7 plan (ordering + KV controller; Part C proposed) |
