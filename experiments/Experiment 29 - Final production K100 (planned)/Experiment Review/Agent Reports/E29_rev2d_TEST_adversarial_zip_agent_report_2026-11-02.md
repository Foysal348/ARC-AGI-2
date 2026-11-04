## TEST agent report: E29 rev 2d adversarial tests + self-contained audit zip (2026-11-02)

I created only these two new files and changed nothing else. Nothing was pushed or uploaded. I did not open any token files or thinking traces. The K100Q/W4 `branch_result.json` files were only inspected for their key names and sizes; their largest `raw_reasoning` value is 51 chars, so they hold no trace text.
- `D:\ARC AGI 2\experiments\Experiment 29 - Final production K100 (planned)\Build\e29_rev2d_adversarial_tests.py` (LF)
- `D:\ARC AGI 2\experiments\Experiment 29 - Final production K100 (planned)\Build\mk_e29_selfcontained_audit_zip.py` (LF)

### Results
- **rev 2c** (the REV2C FLAT audit zip, notebook ed2e23a7, plus the deps in `D:\ARC AGI 2\_scratch\e29_rev2d_adv\rev2c_dir`, run with `--no-fallback`): 58 cases, **13 PASS, 25 FAIL, 20 SKIP_NOT_IMPLEMENTED**, exit 1. All 58 match the expected rev 2c column. The FAILs are the audit bugs reproduced.
- **Current Build at 05:18** (already rebuilt: e29_lib 99f97226, e29d_sched_patch 4cc807f2 = facts sha, e29_nbx 1388aa90, code export 01787a21, notebook 80d4aeff), run with `--strict`: **58/58 PASS**, exit 0.
- Both results are written in the file's header.

### The adversarial test file
Usage: `python e29_rev2d_adversarial_tests.py [--dir DIR] [--export FILE] [--only F1,F10,...] [--strict] [--no-fallback] [--json-out F] [-v]`
- It loads the real modules by path and AST-extracts the notebook functions from the code export, as `audit_repro_e29.py` does. Module-level literal state the functions need (e.g. `E29_SEAM_BYTES`) comes along. Only the environment is stubbed.
- The F1 scheduler cases run the real arc2_duck from the Duck payload with the E28 → E29 → DIRECT patches applied, using the `e29_sched_tests` harness.
- Each case reports PASS, FAIL, SKIP_NOT_IMPLEMENTED (a missing rev 2d function, looked up dynamically) or SKIP_NO_INPUT.
- Exit code is 1 on any FAIL. With `--strict`, any SKIP also gives 1. **Use `--strict` for the post-rebuild check; the expected result is 58/58 PASS.**

Case list (current result on rev 2c in brackets):
- **F10, in-run relaunch plan**
  - [FAIL] F10.1 XID alone → `mtp`.
  - [FAIL] F10.2 XID plus OOM on every rung, and illegal-access plus OOM → `mtp`.
  - [PASS] F10.3 plain OOM steps to the next rung, EMBED_HOST kept.
  - [PASS] F10.4 last-rung OOM → `mtp`.
  - [PASS] F10.5 healthy death → None.
  - [FAIL] F10.6 notebook `e29_relaunch_plan` on XID → MTP top rung and a GPU_DEVICE_FAULT record.
  - [PASS] F10.7 per-rung in-run OOM keeps the pin, EMBED_HOST and seam/QSA env; the last rung goes to MTP.
  - [PASS] F10.8 healthy death leaves the config unchanged.
- **F4, MTP relaunch and its boot gates**
  - [FAIL] F4.1 `e29_mtp_boot` gives the top pin 7,933,526,016, 24 seqs, I8, no DFlash/QSA env, and `mtp_relaunch` armed.
  - [FAIL] F4.2 zero-accept canary kills the server once, then MTP top plus armed gates.
  - [FAIL] F4.3 OOM chain 0→2→3→4→MTP ends with gates armed.
  - [SKIP_NOT_IMPLEMENTED] F4.4 `e29_relaunch_gates` all OK (real `E28L.i8_readback` on an I8_ENGAGED log).
  - [SKIP_NOT_IMPLEMENTED] F4.5 INT8 head not engaged → kill, BF16 head at the same pin, gates stay armed.
  - [SKIP_NOT_IMPLEMENTED] F4.6 C-11 or method-map fatal → kill.
  - [SKIP_NOT_IMPLEMENTED] F4.7 static check: the relaunch path calls the gates.
- **F11, seam revert**
  - [SKIP_NOT_IMPLEMENTED] F11.1 finish raises after a partial revert → re-restored, SEAM_PRISTINE.
  - [SKIP_NOT_IMPLEMENTED] F11.2 a file that cannot be restored (replaced by a directory) → SEAM_REVERT_UNVERIFIED, seam_unverified, the run continues.
  - [SKIP_NOT_IMPLEMENTED] F11.3 clean finish.
  - [SKIP_NOT_IMPLEMENTED] F11.4 `seam_inert` labels.
  - [SKIP_NOT_IMPLEMENTED] F11.5 static: snapshot before apply, restore in drop and in setup failure, inert gate in `nb5_boot_try`, E29_SEAM_NOT_INERT treated as a try failure.
- **F3 / F14, QSA read-back**
  - [FAIL] F3.1–F3.3 read-back labels: 4 ranks OK / one rank at 256 / one rank missing.
  - [FAIL] F3.4 `e29_qsa_rb` falls back to the worker RPC.
  - [FAIL] F3.5 `e29_df_readback` returns QSA_ENV_MISMATCH; with `qsa_off` it is ENGAGED.
  - [FAIL] F3.6 the notebook's own `nb5_boot`: a mismatch relaunches the same rung at 256/256 with DFlash kept.
  - [FAIL] F3.7 a second mismatch goes to MTP top, no loop.
- **F2, first-prefill probe**
  - [FAIL] F2.1 LOW_FREE → one ladder step.
  - [PASS] F2.2 PROBE_FAILED with OOM → step; without OOM → MTP.
  - [SKIP_NOT_IMPLEMENTED] F2.3 probe constants, `worker_free_mib`, probe call in `nb5_boot_try`.
- **F12, `submission_check`** (all SKIP_NOT_IMPLEMENTED)
  - F12.1 valid submission.
  - F12.2 missing task.
  - F12.3 wrong row count.
  - F12.4 bad cells: 10, -1, True, '3', 1.0, None, ragged, empty, 31 rows, missing attempt_2.
  - F12.5 the real CELL 0 fallback submission → OK.
  - F12.6 truncated or missing file → ok False, no raise.
  - F12.7 static: the notebook calls it.
- **F5, effective algorithm**
  - [SKIP_NOT_IMPLEMENTED] F5.1 `effective()` gives DIRECT_OFF / OK correctly.
  - [PASS] F5.2 a missing `e29_nbx` payload → no raise, `extras_error` recorded, RUN None, env removed.
  - [SKIP_NOT_IMPLEMENTED] F5.3 static: `effective` is called.
- **F1, late phase C**
  - [FAIL] F1.7 notebook hook on a fake clock: 31 and 27.02 min left → True; exactly 27.00, 26.5, 25, 20, 1 min → False.
  - [PASS] F1.8a switch OFF → no hook.
  - [FAIL] F1.8b setup failure clears stale RUN and CAN_START hooks.
  - [PASS] F1.1 real scheduler, hook True → DIRECT for all 6 phase-C items, C started/done once, `E28_RELEASE((t,1))` exactly once, valid attempt_2.
  - [FAIL] F1.1b the hook is called once per item with the scheduler's deadline.
  - [FAIL] F1.2–F1.5 hook False / no hook / returns None / raises → the original Duck C session: one start, one completion, one release, a valid attempt_2, `duck_c` recorded.
  - [PASS] F1.6 E29_DIRECT env unset → Duck.
  - [SKIP_NOT_IMPLEMENTED] F1.9 integration: the real nbx hook with drain 1 on the real scheduler, 400 s → DIRECT, 100 s → Duck C.
- **A4's 7 repro scenarios**
  - [PASS] A4-1, A4-2, A4-3 (unchanged correct behaviour).
  - [FAIL] A4-4 XID plus OOM, inverted.
  - [FAIL] A4-5 solver precondition at T-20 plus the patch gated on CAN_START.
  - [FAIL] A4-6 lib XID, inverted.
  - [FAIL] A4-7 a raising finish must produce a verified `seam_restore`; with the F11 decision, DFlash is dropped and I8 is on.

Not covered:
- The "server down" case from the F1 test column: the expected routing behaviour is not specified.
- F6 / F13 ledger line format.
- The body of `e29_prefill_probe`, which needs a GPU stream.

### The zip builder
Usage: `python mk_e29_selfcontained_audit_zip.py [--date 2026-11-02] [--kit rev2d=DIR ...] [--no-rev2c-kits] [--out ZIP] [--no-verify] [--verify-dir DIR] [--keep-verify-dir] [--rev2c-zip ZIP]`
- Default output: `...\Experiment Review\Audit Zips\E29_DFLASH_D3S_REV2D_audit_FLAT_<date>.zip`.
- The rev 2c kits from `D:/ARC AGI 2/_scratch/kits_e29v2c` are always added. Add the rev 2d kits with `--kit rev2d=<dir>`; repeat for more.

Contents (flat):
- The notebook and push metadata: `PUSH__` (kernel-metadata, payload dataset-metadata, SHA256SUMS) and `KITNB__` (kit notebooks and their metadata).
- All Build modules and the 4 test files, plus the builder itself.
- Dependencies: `e28_sched_patch.py`, `e28k100_lib.py`, `nb5_lib.py`, `dflash_seam.py`, and the Duck payload `q27x_e15b_duck_9e4b89285136.bin` (found under Experiment 15 Build/payload, sha checked).
- Payload-dataset files matched by the facts sha; it warns if a sha does not match.
- `e29_kit_lib_facts_dflash-d3s.json`.
- `records_k100q_w4.zip` (`branch_result.json` only) and `arc-agi_evaluation_challenges.json`, for the real-record selection tests.
- The audits: `AUDIT_A1__`, `AUDIT_A2A3__`, `AUDIT_A4__` (reports, `audit_repro`, logs only, not the rev 2c code copies).
- Both fix lists, all agent reports (`REPORT__`), kit reports (`KIT_<label>__`).
- `DELTA_rev2c_to_rev2d.diff`.
- `00_RUN_TESTS.py`, `00_GPT_PROMPT.md` (GO / NO-GO, blockers and severe issues only, on the rev 2d delta; it explains the F11 decision), `00_MANIFEST_SHA256.json`, `SHA256SUMS`.

Verification: the builder extracts the zip into an empty folder, re-checks every manifest sha, then runs `00_RUN_TESTS.py`. All paths point inside that folder and the adversarial tests run with `--no-fallback --strict`. It also flags any log line mentioning "ARC AGI 2".
- **`python -I` is not possible here.** `requests` sits in the user site, which `-I` hides, so my first run failed (sched tests: "No module named requests"). The runner now tries `-I` and falls back to `-E -P`, and records which flags it used.

### Self-containment check
- Trial zip: `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\trial_audit.zip` (2.0 MB, 82 files).
- Extracted to `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\zipcheck2` (logs in `_logs`). 0 sha mismatches, 0 warnings, flags `-E -P`, Python 3.14.6.
  - `e29_sched_tests.py`: 29/29 OK (42.9 s).
  - `e29d_direct_tests.py`: 20/20 OK (18.8 s).
  - `e29_kit_lib_tests.py`: 42/42 PASS.
  - `e29_rev2d_adversarial_tests.py`: 58/58 PASS.
  - 0 log lines mention the repo path. **SELF-CONTAINED CHECK: PASS.**
- I did **not** write the final zip into Audit Zips. The Build may still change, and the rev 2d kit dirs are not available yet.
- Final command: `python mk_e29_selfcontained_audit_zip.py --kit rev2d=<kit dir>` (add one `--kit` per kit dir). It writes the zip and runs the same check.
