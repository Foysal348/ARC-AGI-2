# Experiment 6 — Q27-XHIGH Second Attempt (v1.4.4 + attempt_2)

**Status:** opened 2026-10-12. **Only scope:** make Kaggle's second attempt useful on top of the LB-26.94 notebook.
- GPT built the dual-answer notebook; **v2** (`Experiment Notebook from GPT/Q27-XHIGH-v1_4_4_DUAL_ANSWER_EVAL_v2.ipynb`, sha256 1f8d7ea4) passed Claude's final audit: `Experiment Review/FINAL_AUDIT_DUAL_ANSWER_v2_2026-10-12.md`.
- **GPU EVAL12 result (foysal `q27-xhigh-dual` v1):** primary 1/22, dual 1/22, so the second answer added **0**.
  - 12/22 second answers were ragged or invalid.
  - The same code and seeds as run 7 (3/22) show large run-to-run variance.
  - See `Experiment Results/Analysis/GPU EVAL12 dual v2 analysis (2026-10-12).md`.
- Nothing else changes without the user's permission: no ordering, KV controller, concurrency, temperature, prompt or budget changes.
- GPT is working on the design.

## Problem (P1 in Experiment 1's issue review)

Q27-XHIGH v1.4.4 (LB 26.94) sets:
- `attempt_1` = the model's grid;
- `attempt_2` = **the test input, or its mirror when the model returned the input**.

So one of the two attempts Kaggle allows earns almost nothing.

## Evidence so far

**Independent samples add hits.** On the same 22 EVAL12 test outputs:
- run 4 (v1.4) got 136b0064#0 and 4c3d4a41#0;
- run 6 (v1.4.1) got 136b0064#0 and 247ef758#0.
- Each run alone scores 2/22. Two attempts from the two runs would have scored **3/22**.

**There is no free second grid in today's answers.** In run 7 (v1.4.4), every one of the 22 answers contains exactly **1** grid, and the thinking tails hold no correct alternative grid.

**Hits come mostly from natural thinking closes.** Across runs 4, 6 and 7, natural closes were correct 6/12; forced closes 1/54.

**Constraint:** with MTP ×3, vLLM 0.19 silently ignores `min_p`, `logit_bias` and the server thinking budget (vLLM #39573). A fix must not depend on them.

**Note:** the hidden test puzzles are much easier than evaluation puzzles, so eval-based run-time projections do not apply to the hidden run.

## Fixed from v1.4.4

These are unchanged unless the attempt_2 fix itself requires it:
- Qwen3.8-27B-FP8;
- offline vLLM 0.19.0 engine, MTP ×3, fp8 KV, prefix cache;
- xhigh with 49,152 thinking + 8,192 answer, host-forced close;
- temperature 0.6 / top_p 0.95 / top_k 20;
- the russcore prompt and the grid parser;
- admission until 11:34, drain 11:36, hard stop 11:44, end 11:50;
- the pinned docker image; dual mode.

## Folders

| Folder | Contents |
|---|---|
| `Experiment Notebooks/` | notebooks; `Build/` holds the builder, tests and harness (to be derived from Experiment 1's `make_q27x.py` v1.4.4) |
| `Experiment Review/` | zips and documents for GPT |
| `Experiment Results/` | GPT outputs and Kaggle runs |

## Base files

| What | Where |
|---|---|
| v1.4.4 notebook | `../Experiment 1 - …/Experiment Notebooks/Q27-XHIGH-v1_4_4.ipynb` |
| Builder | `../Experiment 1 - …/Experiment Notebooks/Build/Q27X_build/make_q27x.py` (byte-identical rebuild, sha 95995980) |
| v1.4.4 issue review | `../Experiment 1 - …/v1.4.4 vs all experiments - remaining issues (2026-10-12).md` |
