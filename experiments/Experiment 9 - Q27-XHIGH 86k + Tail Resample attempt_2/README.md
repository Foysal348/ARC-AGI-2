# Experiment 9: Q27-XHIGH 86k budget + tail resample (attempt_2 from a new independent sample)

**Result (2026-10-13): EVAL12 4/22 (18.18 %), our best EVAL12 score; resample gain 0.** See `Experiment Results/Analysis/GPU EVAL12 E9 result (2026-10-13).md`. Run: `esfersami50/q27-xhigh-e9` v1.

**Earlier status: RUNNING on Kaggle.** `foysalemonshanto/q27-xhigh-e9` v1, SAVE_AND_RUN_ALL, commit run = **EVAL12** (12 tasks / 22 inputs, 3 h). Notebook sha256 `845ff050…`.

- Audit fixes 5 and 6 are applied (task-normalised score from the final pairs; `projection_valid = False` once resamples ran).
- The commit-run default is changed to EVAL12 per the user; PROXY56 stays available.
- CPU harness v2: mixed, drain and rerun PASS (logs in `Experiment Review/CPU validation v2 EVAL12 + fixes (2026-10-13)/`). Hidden attempt_1 is identical to v1.4.4.

**Earlier:** notebook v1 built (sha256 `806a2088…`) and CPU-validated.

**CPU harness** (real Qwen tokenizer + chat template, fake vLLM; logs in `Experiment Review/CPU validation (2026-10-13)/`):

| Scenario | Result |
|---|---|
| **mixed** (PROXY56, 72 units) | PASS. `reduced thinking budget: 0`. 24 resamples ran (all 16 no-grid units + 8 forced closes; short natural closes were not resampled), 16 changed attempt_2, primary-only 48 → final 64 hits; the report's arithmetic is checked independently |
| **drain** (tight deadlines) | PASS. No resample, because the queue was not empty before the admission deadline (as designed) |
| **rerun** (hidden, 259 units) | PASS. 85 resamples ran. attempt_1 is **identical** to v1.4.4's. 0 attempt_2 changes is a harness artifact: the fake hidden answer equals the mirror fallback |

## User decisions (fixed spec)

- **Base:** v1.4.4 DIRECT (LB 26.94), with **no puzzle ordering** (original v1.4.4 order). The Experiment 7 ORDERING_ONLY notebook is the same notebook with the Codex order removed.
- **86k budget:** thinking **77,824** + answer **8,192**; `MAX_MODEL_LEN` **106,496**.
- **attempt_2 = an independent new-seed sample run in the free slots at the end of the run (tail resample).** No mining, no scoring (Experiment 8 closed).
- **One experiment, one notebook.** Commit-run set: **PROXY48 + the EVAL12 tasks not in PROXY48** = 56 tasks / 72 test inputs.

## What changes versus v1.4.4 (builder `Experiment Notebooks/Build/build_e9.py`, deterministic)

| Cell | Change |
|---|---|
| 0 | banner |
| 1 (config) | `THINK_BUDGET` 49,152 → 77,824; `MAX_MODEL_LEN` 81,920 → 106,496; `EXPECTED_LIVE_TOKENS_PER_SEQ` 30,000 → 43,000 (CONCURRENCY stays 16); `phase1_cap` asserts for the new values (`phase1_cap(12800) == phase1_cap(20000) == 77,824`); new profile **PROXY56** (default for the commit run) |
| 3 (tasks) | fixed task-list support for PROXY56 (every id asserted to exist) |
| 6 (scheduler) | **tail resample** (below); records gain `t_admit_min` / `t_done_min` |
| 7 (report) | scores the **final** pairs (after resamples) and keeps the primary-only score as `eval_primary_only_hits`; adds `eval_resample_gain`, `eval_resample_hit_units` and a `resample` summary |

**Unchanged:** prompt, sampling (0.6 / 0.95 / 20), seeds of primary units, MTP×3, fp8 KV, prefix caching, the host-forced close (stop text), the parser, `make_attempts`, the submission writer, and the hidden-rerun deadlines (admit 11:34, drain 11:36, hard stop 11:44, end 11:50).

### Tail resample (cell 6)

1. **When it starts:** only when the primary queue is empty (every unit admitted), a slot is free (`active < CONCURRENCY`), and it is before `ADMIT_DEADLINE` and before the drain. These are the same rules as for primary units, so it never takes a slot from a primary unit.
2. **Which unit is resampled:** at most one resample per unit, chosen by priority, earliest finished first within a priority:
   - **0:** no valid grid (attempt_1 is the test-input fallback);
   - **1:** forced close (thinking hit the cap), or no room for an answer;
   - **2:** natural close with long thinking (≥ 0.9 × THINK_BUDGET ≈ 70k);
   - natural closes shorter than that are never resampled.
3. **How it runs:** the same prompt and the same budget, but a different seed (`SEED + 10 × (UNIT_INDEX + 100000) + 1/2`). It goes through the same phase-1 / forced-close / phase-2 path, drain and hard stop.
4. **Result:** if the resample grid is valid and differs from attempt_1 and from the current attempt_2, it becomes **attempt_2** (`src_2 = "resample"`). **attempt_1 is never changed.** The submission is rewritten immediately.
5. **Evidence for this design** (our runs):
   - forced closes hit 2/98, natural closes 16/31;
   - two independent EVAL12 runs together average 2.67 hits, against 2.0 for one run;
   - the old attempt_2 (input or mirror) never hits.

## Commit-run profile PROXY56

- **Tasks:** PROXY48 (EDA LB mirror) + `21897d95 247ef758 4c3d4a41 6e4f6532 88bcf3b4 898e7135 8b7bacbf b6f77b65`, i.e. 56 tasks / 72 test inputs.
- **Time:** budget 6 h; admit until 288 min, drain 324 min, hard stop 335 min (notebook clock).
- **Time estimate** (evaluation data, valid for this evaluation set only): primary about 72 × 66.5k ≈ 4.8M tokens ≈ 3.8 h at 353 tok/s, plus 13 min setup. The rest of the window up to 288 min goes to tail resamples.
- **GPU quota:** foysal had 23.08 / 30 h used (2026-10-13), i.e. about 6.9 h left, so a 6 h run fits only just. Another account, or waiting for the reset, is an option.

## What the run must measure

1. **86k:**
   - natural closes between 49k and 77.8k, and their hit rate;
   - forced-close share;
   - KV max and `num_preemptions` at 16 sequences (expect occasional preemptions);
   - mean tokens per unit.
2. **Tail resample:**
   - how many resamples were admitted and finished;
   - how many changed attempt_2;
   - **resample gain** (final minus primary-only hits);
   - how many resample grids equal attempt_1.
3. **Ledger:** add the run to `resources/Puzzle Capability Ledger/build_ledger.py`.

## Files

| Path | What |
|---|---|
| `Experiment Notebooks/Q27-XHIGH-v1_4_4_E9_86K_RESAMPLE.ipynb` | the notebook (built) |
| `Experiment Notebooks/Build/build_e9.py`, `Build/source/` | the deterministic builder and the v1.4.4 base (sha `95995980…`) |
| `Experiment Notebooks/Build/harness/harness_e9.py` | the Experiment 1 CPU harness (real tokenizer, fake vLLM) adapted for resample units (`tid#q~r1`) |
| `Experiment Review/`, `Experiment Results/` | reviews and run outputs (later) |
