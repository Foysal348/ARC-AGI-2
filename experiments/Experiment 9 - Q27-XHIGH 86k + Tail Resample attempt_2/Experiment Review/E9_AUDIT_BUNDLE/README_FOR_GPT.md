# Experiment 9: request for a pre-GPU audit (2026-10-13)

**Request:** audit this notebook deeply before its first GPU run. Report only **blockers and serious bugs** (anything that can crash the run, lose primary results, corrupt `submission.json`, or silently change behaviour away from the spec). Skip minor style issues. Verify claims against the code; do not trust this README.

## Fixed spec (user decisions; do not redesign)

1. **Base:** `source/Q27-XHIGH-v1_4_4_LB26.94.ipynb`, i.e. v1.4.4 DIRECT, LB 26.94 (sha256 `95995980…`).
   - **No puzzle ordering** (original order).
   - Prompt, sampling (0.6 / 0.95 / 20), MTP×3, fp8 KV, prefix caching, the host-forced close, the parser and `make_attempts` are unchanged.
2. **86k budget:** thinking **77,824** + answer **8,192**; `MAX_MODEL_LEN` **106,496**.
   - User rule: never less than 49,152 + 8,192 per test input.
3. **attempt_2 = tail resample.**
   - Once the primary queue is empty, free slots run **one independent new-seed sample** of a finished unit, with the same prompt and budget, before `ADMIT_DEADLINE` and before the drain.
   - Priority: no valid grid > forced close / no room > natural close with thinking ≥ 0.9 × THINK_BUDGET.
   - A valid grid that differs from attempt_1 (and from the current attempt_2) becomes attempt_2. **attempt_1 is never changed.**
   - No candidate mining and no likelihood scoring: Experiment 8 showed `prompt_logprobs` OOM kills the engine, and mined candidates contain 0 correct answers.
4. **Commit-run set PROXY56:** PROXY48 + 8 EVAL12 tasks = 56 tasks / 72 test inputs, 6 h budget (admit 288 min, drain 324, hard stop 335). The hidden rerun (`RERUN_ALL`) keeps v1.4.4's deadlines (admit 11:34, drain 11:36, hard stop 11:44, end 11:50).
5. Dual mode as v1.4.4: the commit run evaluates and scores; the competition rerun writes the hidden `submission.json`.

## Exact changes (all made by `build/build_e9.py`; each replacement is asserted to hit once)

| Cell | Change |
|---|---|
| 0 | banner |
| 2 (config) | `THINK_BUDGET` 77,824; `MAX_MODEL_LEN` 106,496; `EXPECTED_LIVE_TOKENS_PER_SEQ` 43,000; new asserts `phase1_cap(12800) == phase1_cap(20000) == 77824`, `phase1_cap(30000) == 68048`; profile `PROXY56` (default when not a rerun) |
| 4 (tasks) | `task_list` support (ids asserted to exist in the evaluation file) |
| 7 (scheduler) | the E9 block: `_e9_priority`, `_e9_next_candidate`, `admit_resample`, `_record_resample`; `finish()` dispatches resample units; the tail-admission loop after the primary admission loop; records carry `t_admit_min` / `t_done_min`; resample answer files `*_r1.txt` |
| 8 (report) | primary-only rows for the existing aggregates; `eval_query_hits` = **final pairs** (after resamples); `eval_primary_only_hits`, `eval_resample_gain`, `eval_resample_hit_units`, `resample` summary |

**Resample unit mechanics:**
- id `"<tid>#<q>~r1"`.
- `UNIT_TASK`, `UNIT_Q` and `UNIT_INDEX` are registered at admission; `UNIT_INDEX` = base + 100,000, so seed = `SEED + 10·(idx+100000) + 1/2`.
- It uses the base unit's `PROMPT_IDS` and `P1_CAPS`.
- It shares `STATE`, `start_phase2`, drain and hard stop with primary units.
- `FINAL[base]` is updated only through `_record_resample`.

## Evidence behind the design (our runs)

- Forced-close answers hit **2/98**; natural closes **16/31**. In the E8 PROXY48 run, 6 of 15 natural closes came between 44k and 49k.
- Two independent EVAL12 runs of the same code solved different units: single-run mean 2.0 hits/22; best-of-2 mean 2.67.
- The old attempt_2 (test input or mirror) hit 0/88 (runs) and 0/172 (whole evaluation set).
- E8 R5 GPU run at 49k (PROXY48, 16 slots): 353 tok/s; KV max 73.6 %; 0 preemptions; MTP acceptance 0.663; only about 0.4 GB GPU memory free beside the KV pool.

## CPU validation (Claude)

Real Qwen tokenizer and chat template, fake vLLM step engine. Logs are in `validation/`.

| Scenario | Result |
|---|---|
| mixed (PROXY56, budgets overridden to 3,000 / 40,000 by the harness for speed) | PASS: 24 resamples (16 no-grid + 8 forced), 16 attempt_2 changes, primary-only 48 → final 64. The harness checks the arithmetic independently |
| drain (tight deadlines) | PASS: no resample, because the queue was never empty before the admission deadline |
| rerun (hidden, 259 units) | PASS: attempt_1 identical to v1.4.4. 85 resamples, 0 changes (harness artifact: the fake hidden answer equals the mirror fallback) |

`harness/harness_e9.py` is Experiment 1's harness. Its only adaptations: resample units answer like their base unit; the primary-only checks skip resample rows; `max_model_len` is 106,496; the E9 score arithmetic is checked.

## Questions (please answer each)

1. **Memory and KV at 86k on 4×L4:**
   - Is 16 concurrent sequences safe?
   - Our rough fit: KV-equivalent ≈ 1.25 × live tokens + about 11k per sequence.
   - Any risk beyond KV preemption, e.g. activation or CUDA-graph memory at `max_model_len` 106,496 with MTP and prefix caching?
2. **Tail resample correctness:**
   - Can a resample ever modify attempt_1, or overwrite a correct attempt_2?
   - Can it run before the primary queue is empty, or after the drain or admission deadline?
   - Can it duplicate request ids, or break `STATE` / drain / hard-stop handling?
3. **Submission integrity in the hidden rerun:** `write_submission()` uses `FINAL[base]`. Any path where a resample leaves `submission.json` invalid or partially written?
4. **Report correctness:** do the final versus primary-only scores and the projection fields stay consistent?
5. **Any behaviour change for primary units** versus v1.4.4, other than the budget and `MAX_MODEL_LEN`?

## Files

| Path | What |
|---|---|
| `notebook/Q27-XHIGH-v1_4_4_E9_86K_RESAMPLE.ipynb` | the notebook to audit (sha256 `806a2088…`) |
| `build/build_e9.py`, `build/source/Q27-XHIGH-v1_4_4_LB26.94.ipynb` | the builder and the base |
| `harness/harness_e9.py` | the CPU harness used |
| `validation/mixed.txt`, `drain.txt`, `rerun.txt` | harness logs |
| `context/E8_FINAL_REPORT.md` | why E8 (mining + scoring) was closed |
| `context/EXPERIMENT_9_README.md` | the experiment README |
