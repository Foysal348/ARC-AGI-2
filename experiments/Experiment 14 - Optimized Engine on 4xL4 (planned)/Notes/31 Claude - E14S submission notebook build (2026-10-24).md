# 31 — E14S submission notebook: W1 + DFlash width 6 + CUDA graphs 1..13 (Claude, 2026-10-24)

**Why:** R2 (Notes/30) adopted width 6 (+6.3 %) and graphs13. But the competition rerun runs no measurements, so the
R2 notebook would submit with plain W1 (width 8, graphs up to 12). The arm therefore has to be set in the launch flags.

## Build

| Item | Value |
|---|---|
| Builder | `Build/build_e14.py` (backup `build_e14.py.pre_e14s_backup`) |
| Notebook | `Q27-XHIGH-E14S_SUBMIT_W6G13_DUCK_SGLANG.ipynb`, sha **fc53adbf**006c9fa3dd370e8a1fe04c2b6fcd26573e970ceaa556f81f7b74f8f9, 227 kB |
| Payload | unchanged from R2: `q27x_e14_modules_78029e146cd3.bin`, `q27x_e12_duck_6f48b81af59a.bin`. Already in `hosen42/q27-e14a-payload` |
| R2 notebook | untouched (2298d827, now listed in PUSHED) |

## Changes

1. **The submission arm in `e11_flags`.**
   - When `E14_DO_MEASURE` is False, i.e. in the rerun and in the E14S commit:
     - `--speculative-num-draft-tokens 6`;
     - `--cuda-graph-bs-decode 1..13`;
     - `--cuda-graph-max-bs-decode` dropped.
   - The E14A measurements still start from W1.
   - **Check:** the rerun flags were diffed against R2's real `width6` launch (`e14_results.json` launches). They are identical; only the harness stub paths and seed differ. The R2 width6 arm was built on the kept graphs13 arm, which is the same set.
2. **Profile `E14S` (the new default commit).**
   - The same 25-min Duck window on the 12 G4B tasks as R2, with the submission arm and no measurements.
   - Time budget 55 min.
3. **JIT cache tar only in measurement runs** (`E14_DO_MEASURE`, and > 600 s left).
   - R2 hit the IndexError and nothing was saved.
   - In the rerun, its subprocesses (up to ~6 min) could have run past the 12 h limit.
4. **Markdown** describes E14S.

## Validation

| Check | Result |
|---|---|
| Local harness E14S commit (Py 3.12 venv, scale 0.1) | **11/11** |
| Local harness E14S rerun (`KAGGLE_IS_COMPETITION_RERUN=1`, 5 test tasks, clock 25 min) | **13/13** |
| Kaggle CPU harness (kit 59f5c6ffc927, notebook 3bddd117, foysal v5) | see below |

- The new harness is `src/harness/harness_e14s_nb.py`. It checks:
  - the submission schema;
  - the mode and profile;
  - the server flags (width 6, graphs 1..13, no max-bs, no nodelay);
  - that no measurement ran;
  - the payload and staging;
  - the phase record;
  - that the Duck ran with no error;
  - isolation PASS with 12 workers;
  - in the rerun, that submission = the solver's generated attempts for every test task.
- **My first harness check was wrong.** It treated attempt_1 = input as "identity only", but the fake's programs return the input for test tasks. I replaced it with the comparison against `e12_generated_attempts.json`.
- `build_cpu_kit.py` now runs E14S in commit + rerun mode, and E14A in default + corrupt.

## Update: two notebooks (user 2026-10-24)

The user asked for a 12 h-style evaluation run to see coverage at "the hidden run's time per task", plus a short
(~30 min) submit commit. Both are built from the same builder; their rerun path is identical (RERUN_ALL, same flags).

| Notebook | sha | Commit profile | Commit length |
|---|---|---|---|
| `Q27-XHIGH-E14S_SUBMIT_W6G13_DUCK_SGLANG.ipynb` (`build_e14.py`) | **3dde5cbd**1fa88e0f07d690929ebdef4639d15b43d5798ba6d4adfdcf5b96b237 | E14S: the 12 G4B tasks, 8-min Duck window | **36 min** |
| `Q27-XHIGH-E14S_EVAL120G4B_W6G13_DUCK_SGLANG.ipynb` (`build_e14.py --eval120g4b`) | **44e11cf6**741780f4c12d81c8a2366e8f8b06d015c97afe07121b2af5df923d81 | E14SEVAL120G4B: 120 eval tasks / 172 inputs, gate 4b (hidden_equiv) | **6.5 h** budget |

- **Why 36 min and not 30.**
  - R2 was READY at about 12.5 min: staging done at 124 s, venv install about 3 min, first launch 502 s.
  - The SGLang give-up is `min(T0 + 75 min, admit - 5 min)`. A 30-min budget puts it at 13 min, and a late launch would then go to the vLLM fallback.
  - A 36-min budget puts it at **19 min** (`give_up_after_sec` 1140, read in the harness).
- **EVAL120 G4B window:**
  - per task = (42,720 − start) / 240 ≈ 175 s at a start of about 800 s;
  - 120 tasks → about 5.8 h of solving;
  - the 6.5 h budget holds start + window + the 3 + 2 min stop margins.
- **Earlier shas fc53adbf and 5fb3fddd are superseded** (not pushed).
- **Validation (local, Py 3.12):**
  - submit 3dde5cbd: commit 11/11, rerun 13/13;
  - eval120: harness mode `eval120` (checks the 120/172 plan, the gate 4b window formula, that the budget holds the window, and the scoring).
- **Quota for EVAL120 at R2's rate** (1.66 h wall → 3.95 h quota, 2.37×): about 15 h; hosen42 has 4.47 h left.

## GPU plan (needs the user's go)

| Item | Value |
|---|---|
| Account | hosen42 (inputs checked: models, wheelhouse, payload 2/2; quota 40.53 of 45 h) |
| Commit | `push_e14s.py --hamid --sha fc53adbf… --commit`. Profile E14S, worst case ~55-58 min wall, ≈1.9 h quota at 1.95× |
| Stop point | the notebook hard stop at 53 min |
| Then | submit this version (rerun = 240 hidden tasks, 12 h) |
| Risk | width 6 is measured only at ~6k context; the commit's Duck window is the first long-context check (R2 graphs13 + width 8: 6/19) |

## E14S SUBMIT commit 1 (hosen42 v1, sha 3dde5cbd), 2026-10-24 05:03 → 05:34 local, and the submission

Outputs: `Experiment Results/Kaggle Runs/E14S SUBMIT commit1 (hosen42, 2026-10-24)/`.

| Item | Value |
|---|---|
| Phases | staging done at 123 s; LAUNCH at 260 s (no NCCL bench); **READY at 783 s** (give-up was 1,140 s); REPORT at 1,344 s; terminal `done` |
| Server flags | width 6 (log: "DFLASH block size mismatch: using speculative_num_draft_tokens=6 but draft config block_size=8", the same warning as R2's width6 arm); verify graphs bs 1..13 captured; max running 13 |
| Health | 0 "out of memory", 0 Traceback, 0 RuntimeError, 0 Retract (the 10 "OOM" strings are Triton "Pre-load … to avoid CUDA OOM" warnings; free 0.98 GiB) |
| Engine in the Duck window | accept p50 4.07 (111 decode logs), running max 12, gen tok/s p50 **483** at 12 running |
| Duck (8-min window, 12 G4B tasks) | 12/12 branches started and done, isolation PASS, 12 workers, cached 83.5 %; **1/19 correct** (bf45cf4b, pass1) |
| Submission | **submitted 2026-10-24, ref 57574781** (`_scratch/e14s_submit/submit_e14s.py`, hosen42; no other submission pending) |

- 1/19 in 8 min is a smoke result, not a score: R2's 6/19 needed 25 min.
- Four branches show < 300 generated tokens because their first 12,288-token request had not finished when the window closed.
