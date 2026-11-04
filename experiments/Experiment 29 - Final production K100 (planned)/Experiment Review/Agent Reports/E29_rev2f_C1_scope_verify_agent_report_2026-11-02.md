**GO.** No BLOCKER or HIGH findings in the rev 2f C1-scope change. My verdict is limited by what I could not check (see the last section), mainly whether the Kaggle dataset version has the new payload files.

## Findings, ranked

### BLOCKER
None.

### HIGH
None in code.

### MEDIUM
**M1 – The new payload files must be uploaded before any GPU commit (I could not check this).**
- The production notebook now pins `e29_nbx.py` 9d77f854… and `e29d_sched_patch.py` f2c60690… (notebook line 539 `E29D_FILES`).
- The old Kaggle dataset version would have 1388aa90… and 4cc807f2… instead.
- If the dataset version is stale, two things fail:
  - the CELL 3 sha check fails, so the DIRECT diff is not applied and DIRECT is off;
  - the `e29_nbx.py` exec in `e29_extras` fails, so DIRECT and LIVE are both off.
- The notebook does not crash. It prints `!!!!!!!!!!` and `effective()` reports `DIRECT_OFF`. But the whole run would quietly be "no DIRECT".
- The local folder `D:\ARC AGI 2\_scratch\push_e25\ds_q27-e29-final-payload\` is correct (all shas match, details under item 4).

### LOW
- **L1 – Counter label.** `e28["direct"]["duck_c"]` now counts the by-design C2 routes (why `c2_duck`) together with late / no_hook / hook_error routes.
  - `e29_lib.py:346`: the `run_summary` docstring calls them "Duck-C late routes".
  - `e29_lib.py:385`: the [SMOKE SUMMARY] prints `duck_c` and `duck_c_why`, so the breakdown by reason is visible. Read `duck_c_why`, not `duck_c`.
  - Nothing gates on this counter (searched the notebook, `e29_nbx.py`, `e29_lib.py`).
- **L2 – LIVE line wording.** `e29_live.py:75-76`: "attempt_2 hits" is printed only on C-DIRECT lines, but its value adds up attempt_2-only hits from every source over all tasks. Now that Duck C runs on C2, it includes Duck C hits, so it is not DIRECT-only. Print-only. Labels themselves are right: `phase()` (`e29_live.py:24-28`) gives "C-DIRECT" for `e29_direct` records and "C" for Duck C records. Mixed C items do not crash (checked the code; the tests exercise mixed records).
- **L3 – Stale doc text still says "every C item is DIRECT".**
  - The notebook header markdown (builder `build_e29_final.py:1159`) says "phase C runs one DIRECT request per task".
  - The `e29d_sched_patch.py` block comment at lines 162-165 says the same (the rev 2f note right after it is correct).
- **L4 – `Build\SCHED_DONE.json` is stale.** It still records rev 2d (4cc807f2… / post 937e6cac…), and the assert message at `build_e29_final.py:1256` refers to it. `SCHED_PIN` itself is correct (f2c60690… = the module sha). Record-only.
- **L5 – A phase typo turns DIRECT off for all C items.** `e29_lib.direct2_cfg` (lines 495, 501): `'c'`, `'C2'`, `None` etc. make the whole config invalid, so DIRECT is off, not just reduced to C1. This is as specified, and it is logged (CELL 1 "E29 DIRECT2 … invalid", state `OFF_SWITCH (invalid…)`, validity `DIRECT_OFF`). Separately, `_e29d_scope()` treats any env value other than "C" as C1, which is the safe direction.
- **L6 – Older behaviour, not new in 2f.** In `e29_extras` (`build_e29_final.py:700-712`), an exception in `e29_live_setup()` after a successful `e29_direct_setup()` removes `E29_DIRECT` and the hooks. A LIVE failure would therefore switch DIRECT off, while `E29_DIRECT_RT["state"]` still says "ON" (`extras_error` is recorded). `e29_live_setup` is very unlikely to raise.
- **L7 – Queue-order observation, unchanged from E28.** C1 items always sort before C2 (`e28_c_key` gives (1, …) vs (2, …)). So the Duck C items on C2 tasks, which produced all 7 attempt_2-only hits in E28, run after every C1 DIRECT item. If time runs short, the C2 items are the ones cut. This is the same order E28 used, not a bug.

## Item-by-item verification

**1. Scheduler (I rebuilt arc2_duck.py from the Duck payload through E28, E29 and DIRECT in scratch; the post sha matches POST 27bc0ca7…)**
- **Generator.** `e29_direct_build.py --regen-sched --dry` reproduces `e29d_sched_patch.py` byte-for-byte (sha f2c60690…).
- **Only rev 2f code change.** Against the backup patch, the only logic change is in `e29d_sched_patch.py`:
  - line 42: the `E29D_PHASE_ENV` constant;
  - line 114: the call is now `_e29d_can_start(deadline, p)`;
  - lines 177-191: the new `_e29d_scope()` and the `c2_duck` early return.
  - Everything else is hunk offsets and comments.
- **C2 path.** The `c2_duck` return falls into the existing F1 branch (patched duck lines 4326-4337). That calls `_e29d_duck_c` (dict counters only) and then the unmodified `s = sess(tid, p, qpos)` Duck C code. So a C2 item gets exactly what the E29 post file does:
  - one `started["C"]` / `done["C"]`;
  - `route()`;
  - `E28_RELEASE((tid, 1))` once, at line 4386;
  - `row_of` and `publish`.
- **Reservation.** `_e28_try_reserve` (notebook line 1004) is called in `pick()` before routing and does not depend on DIRECT. The test helper `_per_item` asserts one start, one completion and one `(task, 1)` release per item, with no double start. `_duck_parity` asserts C2 records and submissions equal the E29 post run.
- **c_priority.**
  - `push_c` (patched duck lines 3987-3993) sets `c_priority = e28_c_key(full, …)[0]`, which is 2 when full and 1 otherwise (lines 3912-3915).
  - It has only two callers, both in `route()`: A TRAIN_PASS always pushes full=True, so priority 2; a B end pushes full = (B kind == TRAIN_PASS) (lines 4220 and 4226).
  - `e29_sched_patch.py` (supply) adds no `push_c`. Supply items are phases S1/S2 (never "C"), so the DIRECT branch never sees them. Session-setup FAIL on a C item: `route()` ignores phase C.
  - Result: every C item has `c_priority`, it is 1 exactly when the task has no full train match after A/B, and never 1 for a full-match item.
- **Lock.** `_e29d_can_start` now only adds an `os.environ.get` and a dict `.get` under the scheduler lock. It does no I/O, no waiting and cannot raise there. For C2 items it returns before calling the hook.

**2. Answer selection**
- `select_attempts_e28` sends a task to `select_attempts_e29d` only when a DIRECT record exists (patched duck line 3866).
- A C2 task under C1 scope has no DIRECT record, so the plain E28/E29 rule applies (attempt_2 = Duck C's answer). The parity tests confirm the submissions are identical.
- On a C1 task, `select_attempts_e29d` computes attempt_1 from the non-DIRECT records, through the same E29 supply rule. attempt_1 never changes; `test_c1_default_direct_on_train_fail_only` checks this.

**3. Environment variable `E29_DIRECT_PHASE`** (`build_e29_final.py:785,797`; `e29_nbx.py:211,223`)
- It is removed first thing in `e29_direct_setup`, and set only after both hooks are set on the ON path. It stays unset on switch off, invalid switch, and "diff not applied" (that raise comes before the set).
- On the extras-exec failure and setup-exception paths a value may remain, but `E29_DIRECT` is removed and the hooks are None, so `_e29d_on()` is False and the leftover value has no effect.
- There is no path where DIRECT is on and the env is unset. If it ever were unset, the patch default is C1 (tested). A stale 'C' cannot matter, because the ON path always overwrites it.

**4. Consistency**
- **Payload vs Build vs SHA256SUMS:** identical — `e29_direct.py` 8b42802c, `e29_nbx.py` 9d77f854, `e29d_sched_patch.py` f2c60690, fixture 46ef4839.
- **Notebook pins:** `E29D_FILES` matches those files; `SCHED_PIN` = module sha.
- **Embedded modules:** `E28_SRC` (notebook line 513) decompresses with its sha OK; the embedded `e29_lib.py` and `e29_sched_patch.py` equal the Build files.
- **Production vs backup:** exactly 3 changed lines (59 switch, 513 `E28_SRC`, 539 `E29D_FILES`). The packed engine blob is unchanged.
- **Smoke vs production:** exactly the one `E29_PROFILE` line.
- **Sizes:** 396,614 and 396,612 bytes, both under 400,000.
- **Notebook shas:** d3054fa1… / c7317614…, also recorded in the build facts.
- **Other variants:** E29_DEFAULT, E29_MTP_V64 and ALL_OFF are byte-identical to the backup (they contain no DIRECT code).
- **Code export:** matches the notebook (only the packed-blob cells differ, by design).

**5. LIVE / reports** — see L1 and L2. Nothing crashes on mixed DIRECT / Duck C items.

**6. Tests** — see the next section.

**7. Other** — no new crash, answer-loss or silent-disable path beyond M1, L5 and L6.

## Test commands and results
All run with `PYTHONIOENCODING=utf-8`.
- **`cd Build; python e29d_direct_tests.py`** → 28 run, 0 failures, 0 errors, 0 skipped (16.2 s). `-v` on TestC1Scope plus TestDirect2Cfg → 8/8 OK.
- **`python e29_rev2d_adversarial_tests.py`** → 59 cases: 59 PASS, 0 FAIL (C1.1 PASS, F1.x PASS).
- **`python e29_sched_tests.py`** (untouched) → 29 run, all OK.
- **`e29_kit_lib_tests.py`**
  - Run from `Build` directly it fails at import: `e28k100_lib` is missing (it is a kit file).
  - Re-run from `scratchpad\verify_c1\kitlib\` with `e28k100_lib.py` from `_kit_local\dflash-d3s` (identical to E28's), `E29LT_NB` = the production notebook, `E29LT_FACTS` = `e29_dflash_d3s_build_facts.json`, `E29LT_KF` = `_kit_local\dflash-d3s\e29_kit_facts.json`, `E29LT_NB5LIB` = `_kit_local\dflash-d3s\nb5_lib.py`, `E29LT_PAYLOAD_DIR` = the dataset folder → **44/44 PASS** (R11, R12 PASS).
  - The same run on the SMOKE notebook gives 43/44: only R29 fails. That is expected, because R29 asserts the production profile line and checks the smoke sha by replacing that line (it passed on production).
- **Negative control 1 (scheduler):** `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\verify_c1\negctl\` = a copy of the Build files with the backup `e29d_sched_patch.py` (4cc807f2…). Command: `python -m unittest -v e29d_direct_tests.TestC1Scope e29d_direct_tests.TestDirect2Cfg`.
  - `test_can_start_scope`: ERROR (no `_e29d_scope`).
  - `test_c1_default_direct_on_train_fail_only`, `test_c1_multi_worker`, `test_c1_real_solver`: FAIL (DIRECT also ran on t000/t001/t003/t005).
  - `test_c1_late_all_duck`: FAIL (`{'late': 6}` instead of `{'late': 2, 'c2_duck': 4}`).
  - `test_c1_switch_off_parity`, `test_scope_c_is_rev2e` and TestDirect2Cfg pass, as expected (the old behaviour is identical there, and TestDirect2Cfg uses `e29_lib`).
- **Negative control 2 (lib):** the backup `e29_lib.py` in the same copy → `test_phase_key` ERROR, `KeyError: 'phase'`.
- No Build files were modified; the harness temp dirs were cleaned up.

## What I could not check
- Whether the Kaggle dataset version has the new payload files (M1); no Kaggle API was allowed.
- Real GPU / Kaggle runtime behaviour: DIRECT and Duck C side by side under real KV pressure, and real timing of C2 Duck items behind C1 DIRECT items.
- The Kaggle CPU kit on Py3.12 (I ran locally only).
- The other variants' kits (their notebooks are unchanged, so not re-run).
