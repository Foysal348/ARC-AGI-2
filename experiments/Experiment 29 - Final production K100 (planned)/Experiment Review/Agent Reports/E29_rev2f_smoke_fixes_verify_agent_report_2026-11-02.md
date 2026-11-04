GO for the three rev 2f smoke fixes. I found no blocker in the ladder change, the submission_check retarget or the LIVE dedupe. One caveat: a separate rev 2g change ("early DIRECT on train-fail tasks") was rebuilt at 08:09 BDT while I was reviewing. It moved the notebook shas, and I did not review its logic (see HIGH-1).

## Findings

**HIGH-1: the build changed during the review; rev 2g is not reviewed by me**
- Evidence: `_backup_rev2g_pre_early_0806\` appeared. At 08:09 BDT these were rewritten: `Build\e29d_sched_patch.py` (sha f2c60690 -> b88efa86), `e29_direct_build.py`, `e29d_direct_tests.py`, `build_e29_final.py:1245` (SCHED_PIN only), both notebooks, both push folders and the payload dataset.
- The shas you gave are stale. Current production notebook is `9a0978474fb2...`, current smoke = SUBMIT notebook is `19928629ac35...`. Sizes are unchanged: 396,918 and 396,916 bytes.
- I re-checked the consistency items on the current rev 2g bytes, and they all hold:
  - production and smoke differ only on the `E29_PROFILE` line (cell 2 line 3);
  - E29D_FILES (direct 8b42802c, nbx 531d18ec, sched b88efa86) == Build == `ds_q27-e29-final-payload` == SHA256SUMS (all 9 files OK);
  - the embedded e29_lib text == `Build\e29_lib.py` (339dd2ac).
- The rev 2f content is unchanged in 2g: e29_lib and e29_nbx shas are the same, and the builder diff against the 2f backup is only line 1245.
- Not reviewed, and worth a decision: in 2g, a task that got an early DIRECT item queues no Duck C at its B end (`e29d_sched_patch.py` "early_b_end_no_c"). If that early DIRECT item never produces a record (an error, or not admitted after the drain point), the task appears to get no second-branch attempt at all. Please confirm that is intended.

**MEDIUM: none.**

**LOW-1: the ladder change is correct; only notes**
- `e29_lib.py:86-100`: `df_tries(True)` gives exactly 7,665,090,560 (rung 2) -> 7,396,655,104 (rung 3) -> 6,859,784,192 (rung 4), all with EMBED_HOST. `df_tries(False)` is unchanged.
- Every user takes its tries from the same `df_tries(...)`, so nothing assumes the first D try is D_PIN0 or that there are 4 D rungs:
  - boot ladder: `build_e29_final.py:1073` (export 4156);
  - in-run plan: `build_e29_final.py:619` (export 2453);
  - `df_relaunch_plan` matches by rung id (`e29_lib.py:156-166`);
  - `next_try_index` checks positions.
- OOM on the last rung:
  - at boot: `next_try_index` returns the first E28 try. Then `e29_drop("df")` and `mtp_top(tries, j, K100_PIN_STEPS_GIB[0])` (top_kv defaults to D_PIN0 7,933,526,016, `e29_lib.py:276`; it does not duplicate the E28 top rung, which is also K100_PINS[0]).
  - in-run: rung 4 is the last of 3 rungs, so the plan is "mtp". `e29_mtp_boot` (`build_e29_final.py:474`, export 2308) sets kv_bytes = D_PIN0 and pin_gib = K100_PIN_STEPS_GIB[0], keeps the booted `--max-num-seqs 24`, and `e29_drop` restores I8. So the MTP top pin, 24 seqs and I8 are unchanged.
- Seqs on the D rungs: `top_rung_seqs` no longer fires on the first D try, because its gib differs from step 0. Seqs comes from `E29_CFG["workers"]`, which is 24, the same value. The smoke shows "workers 24" on `df3r2`.
- Boot-try names now start at `_df3r2` (smoke: `merlin_fp8_mtp_pin483_df3r2`). Nothing parses `r0`.
- Docstring nit: the `df_tries` text still describes the top rung first (`e29_lib.py:87-88`).
- E29_DEFAULT, MTP_V64 and ALL_OFF embed `e29_lib_v1_0.py` (d9fba907; builder lines 157/168). Their notebooks are not rebuilt (dated 11/1). They are unaffected.
- Margin evidence is one GPU sample: 903 MiB free vs the 512 floor at 7,665,090,560.

**LOW-2: the LIVE dedupe is correct; no real case loses a count**
- Changed code: `e29_live.py:18,60-67`, same in `e29_nbx.py:20,62-69`.
- The re-publish is the window-end `publish(fr, final=True)` over every session (e28_sched_patch ~L803-815). With the dedupe, smoke lines 148-161 would show A 12/12 and C 2.
- No legitimately new record can share a key:
  - **B1 vs B2:** both are branch 2 but have different phases. They also share session (tid, 2), so one task cannot run both.
  - **Supply:** S attempts get branch next_b = branch + 1 (3, 4, ...), so each has its own key.
  - **DIRECT vs Duck C on branch 1:** the phases differ ("C-DIRECT" vs "C").
  - **Stragglers:** a straggler never published by its worker is counted once, on the final publish.
  - **No re-queue:** nothing re-queues the same (task, branch).
- Theoretical only: a record without `e28.phase` falls back to the branch map. `row_of` sets the phase at attempt start, so I found no path where this happens.
- Small print-only gap: a duplicate does not update the accepted/exact counts it carried first. The selection, queries and task-norm do update.

**LOW-3: submission_check retarget is right in both modes and cannot raise into the run**
- `test_challenges` is the CELL 0 global (cell 1 line 31, `arc-agi_test_challenges.json`). It is never reassigned in cells 2-8.
- Commit: the smoke `submission.json` has 240 tasks / 259 rows. The old check against the 12-task solve set printed "missing 12". The new check compares against the 240 the file is written for.
- Rerun: `test_challenges` is the hidden file, the same set as CHALLENGES.
- Call sites:
  - `build_e29_final.py:660` / `e29_nbx.py:115`: e29_nbx is exec'd into the notebook globals, so `globals()` resolves; wrapped in `try/except Exception`.
  - `build_e29_final.py:1174` (cell 8 line 158): inside `try/except BaseException`.
- `submission_check` also catches its own exceptions (`e29_lib.py:336-347`).

**LOW-4: rerun profile and SUBMIT folder are fine; two notes**
- Rerun: `PROFILE = "RERUN_PROD" if RERUN else ...` (cell 2 line 78, export 126), whatever the `E29_PROFILE` line says. The deadman is armed only `if not RERUN` (cell 2 lines 66-74). The smoke task list / window is only used through the E29SMOKE profile row.
- The SUBMIT commit run itself is the 70-min smoke, with the deadman `os._exit(0)` at 70 min. The smoke finished at about 61.7 min (3,700 s) including the 6.5-min failed rung, so it now has more margin.
- SUBMIT folder: id `foysalemonshanto/q27-xhigh-e29-dflash-d3s-submit`. Its datasets, models, competition, docker image and machine are the same as production; only id, title and code_file differ.
- The `q27-xhigh-e29-dflash-d3s-smoke__hosen42` folder still holds old bytes (4b6bfd63, 396,383 B). Do not re-push it by mistake.

## Tests (from Build, PYTHONIOENCODING=utf-8)
- `python e29_rev2d_adversarial_tests.py`: exit 0, 59/59 PASS.
- `python e29_sched_tests.py`: exit 0, 29 run, 0 failures.
- `python e29d_direct_tests.py`:
  - First run: 3 failures (`TestC1Scope` c1_default / c1_late_all_duck / c1_real_solver, plus `E29_DIRECT_STATE_MISMATCH ... OTHER 468f43a6`). That run overlapped the rev 2g rewrite of `e29d_sched_patch.py` and `e29d_direct_tests.py`, so I count it as transient.
  - Rerun: 28 run, 0 failures.
  - Rerun after the 2g test file landed: 34 run, 0 failures.
- `e29_kit_lib_tests.py`: 44/44 PASS, exit 0.
  - Run from `Build\_kit_local\dflash-d3s` (cwd there; the file is identical to Build's, 3604bd50).
  - Env: E28KT_NB = E29LT_NB = the current production notebook. E28KT_FACTS = E29LT_FACTS = an LF copy of `e29_dflash_d3s_build_facts.json` in scratch.
  - E28KT_NB5LIB / E29LT_NB5LIB, E29KT_E28_NB and E29LT_KF from `_kit_local\dflash-d3s`.
  - E28KT_K100Q_FIXTURE and E29LT_PAYLOAD_DIR from the payload folder.
  - Scratch outputs: `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\verify_smokefix\`.

## Could not check
- The rev 2g logic (early DIRECT scheduling; `e29d_sched_patch.py` b88efa86, `e29_direct_build.py`) and its effect on coverage.
- Real GPU behaviour of rungs 3 and 4 and of the in-run OOM relaunch (not exercised in the smoke).
- Whether the payload dataset and notebooks on Kaggle match the local folders (no Kaggle API by rule).
- The Kaggle CPU kits for rev 2f/2g: `e29_kit_facts.json` and `_kit_local\facts` are from 07:59-08:00, before the 2g rebuild.
