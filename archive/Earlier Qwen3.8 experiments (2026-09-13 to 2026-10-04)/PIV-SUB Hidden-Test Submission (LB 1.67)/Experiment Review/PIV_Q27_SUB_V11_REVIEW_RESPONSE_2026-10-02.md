# Response to the independent review of PIV-SUB v1.0 — and what v1.1 changes

Audience: the reviewer who audited `PIV_Q27_REVIEW_BUNDLE_2026-10-02.zip`, and the project owner.

Every finding was re-checked against the code before anything was changed. **All seven findings are real.** Two needed
their impact restated; both restatements are below with the evidence. Everything is fixed in
`PIV-Q27-SUB-v1_1-HIDDEN-TEST-SUBMISSION.ipynb`, and each control fix was proven by failure injection rather than by
inspection.

---

## 1. Verification of each finding

| # | Finding | Verified | Note |
|---|---|---|---|
| P0 | fallback submission written too late | **yes** | Before the old CELL 2B ran: model/wheelhouse/data existence checks, model `config.json` validation, offline pip install, runtime version parity, then CELL 2's five JSON loads plus `assert set(test_challenges) == set(sample_submission)` and the Proxy-48 membership check. Any of these raising left `/kaggle/working/submission.json` absent. The README's narrow claim ("from this point on") was right; its overall claim ("no execution path can leave the file missing") was wrong and has been corrected. |
| P1a | `ADAPTIVE_SMOKE_COMPLETED = True` unconditional | **yes** | Scheduler cell, directly after the stage-B `for _arm in ARMS: run_arm(_arm)` loop. `run_real_piv()` can `break` out of its batch loop on the safety deadline, and tasks in batches that never started get no record at all, so `run_report.json` could claim `"status": "completed"` with `tasks_with_terminal_record < tasks_offered_to_solver`. |
| P1b | no projected-deadline check before the finalizer | **yes** | The per-batch wave loop has a hard `if time.perf_counter() >= deadline` check; the exploration wave additionally has a projected check (`_projected_wave_sec`); the terminal-finalizer branch had neither. |
| P2a | sanitizer does not reject floats/bools as documented | **yes** | Measured on the shipped function: `[[1.0]] → [[1]]`, `[[np.float64(1.0)]] → [[1]]`, `[[np.bool_(True)]] → [[1]]`, `[[np.int64(2)]] → [[2]]`; only Python `bool` was rejected. `1 == 1.0` and `1 == np.bool_(True)` defeated the `value != cell` test. |
| P2b | repair rows lost after a deadline stop | **yes** | The deadline branch appended rows for the current chunk only, then `break`. |
| P2c | CPU pass budget is per task, not per program/query | **yes** | The budget was checked once per task; inside, up to `SUB_MAX_PROGRAMS_PER_TASK` programs × N queries × 10 s could still run. |
| P3 | `SHA256SUMS.txt` not standard; absolute Windows paths | **yes** | The file had a trailing `  (N bytes)` column *and* CRLF line endings, so `sha256sum -c` failed on both counts. `harness_c1.py` pinned the scratch root and a dead v2.4E notebook default; the build scripts pinned `D:\ARC AGI 2`. |

### Two impact restatements

**P0 — the severity, as stated, over-reads what the file does for scoring.** A Kaggle code-competition submission is
scored from the re-run against hidden data; a notebook that raises is marked as having thrown an exception and scores
nothing, whether or not `submission.json` sits on disk. So moving the write earlier does **not** rescue a run whose
environment checks fail — and deliberately *not* raising would be worse: it would spend one of the day's submissions
on a guaranteed zero while hiding the breakage. v1.1 therefore does both of the things that are actually useful:
the fallback is written in the **first cell** (standard library + the attached test file only), and the environment
checks still raise loudly. The gain is a correct file for interactive inspection, a guarantee that the file on disk is
never a stale invalid one, and a run report that states whether the bootstrap succeeded.

**P2a — real as a contract violation, but with no reachable path in this pipeline.** Every grid that reaches
`sub_add_candidate` comes from the verifier subprocess, and the verifier's own `clean()` already raises
`non_integer_cell` for `isinstance(x, bool) or not isinstance(x, numbers.Integral)` (CELL 6). Results also cross a
JSON boundary, which cannot carry NumPy scalars at all. So no float or `np.bool_` could have reached the submission
file; the defect was the docstring plus a missing defence-in-depth check. Fixed anyway, strictly.

---

## 2. What v1.1 changes

`Research Base Experiment/PIV-Q27-SUB-v1_1-HIDDEN-TEST-SUBMISSION.ipynb`
sha256 `2042c2914bd6621b673930c9c10a6e0aad23713fba8fc9301480c51a9b793386`
(solver notebook unchanged: `PIV-Q27-C1-v2_5A-...` `2188b999efeb6e286fc914bb8c1d876a5c9bdab99537842cd8c94360cd47d5ff`)

1. **New first cell — fallback bootstrap.** Standard library only. Locates
   `arc-agi_test_challenges.json` (exact path, else a unique `rglob` hit), writes an identity `attempt_1`/`attempt_2`
   per test input through a temp file + `os.replace`, and never raises: a failure is recorded in
   `BOOTSTRAP_SUBMISSION_ERROR` and the full machinery in CELL 2B rewrites the file later anyway.
2. **Completion derived from coverage.** After stage B the notebook reads the per-arm records and compares against
   `ACTIVE_TASK_IDS`; missing tasks set `ADAPTIVE_SMOKE_COMPLETED = False`,
   `ADAPTIVE_STOP_REASON = "incomplete_terminal_coverage"`, and `run_report.json` gains `coverage_missing_tasks`.
3. **Finalizer projected against the hard session limit.** New constant `SESSION_HARD_MARGIN_SEC = 10 * 60`. The
   limit is derived from the live deadline (`deadline + SAFETY_RESERVE_SEC - SESSION_HARD_MARGIN_SEC`), which equals
   `notebook_start + TOTAL_RUNTIME_SEC - margin` in a real run and keeps the scheduler self-tests (which override
   `deadline`) unaffected. Deliberately checked against the **session** limit and not against the solver's safety
   reserve: the finalizer is a task's last chance to emit a program, so it must not be dropped merely because the
   reserve is near. Skipped tasks are recorded as `global_deadline_projected` with an infrastructure flag.
4. **Strict grid types.** `bool`, `numpy.bool_` and every float (including `1.0`) are rejected;
   Python and NumPy integers in 0–9 are accepted. The docstring now matches, and the submission self-test asserts all
   of these cases.
5. **Complete repair reporting.** A deadline stop emits `repair_skipped_deadline` rows for **every** remaining job and
   prints how many were skipped.
6. **One absolute stop line for the CPU pass.** `_stop_at = min(deadline, start + SUB_UNVERIFIED_BUDGET_SEC)` is
   checked before each task, before each program within a task, and before each query inside
   `sub_run_program_on_test_inputs`.
7. **P3.** `SHA256SUMS.txt` is now `sha256sum -c`-compatible (hash, two spaces, path, LF endings), with sizes moved to
   a separate `MANIFEST.csv`. `harness_c1.py` honours `PIV_HARNESS_SCRATCH` and requires the notebook path as
   `argv[2]`; both build scripts honour `PIV_PROJECT_ROOT`.

---

## 3. How the control fixes were proven (not just inspected)

The harness gained injection hooks (`HARNESS_SHRINK_PRIOR`, `_TOTAL`, `_RESERVE`, `_MARGIN`, `_DEADLINE`) so the time
guards can be driven into their failure corners on CPU.

| injection | observed |
|---|---|
| `PRIOR=0.0005 DEADLINE=200` on `SUBMIT` (240 tasks) | stage B truncated at 62/240 → `STAGE B INCOMPLETE - tasks without a terminal record: {'ACT-LOW': 178}`, `"status": "stopped_early"`, `"stop_reason": "incomplete_terminal_coverage"`, `coverage_missing_tasks` populated. Pre-fix this reported `completed`. |
| same run, repair round | `repair: stopping before a 1.5 min chunk would cross the deadline; 45 job(s) skipped` — all remaining jobs reported, not just one chunk. |
| `PRIOR=0.0005 TOTAL=600 RESERVE=20 MARGIN=610` on `SUBMIT_SMOKE` | `finalizer skipped for 1 task(s): projected 1.5 min would cross the hard session limit`; the notebook's own REAL-scheduler terminal-commit self-test still **passes**, the interface gate then correctly fails, and `submission_valid` stays `true`. |
| `PRIOR=0.0005 TOTAL=300 RESERVE=60 MARGIN=20` | realistic ordering: the solver's safety deadline stops work first and the finalizer guard never fires — i.e. the new guard is a last-resort net, not a new early-stop. |
| strict types | asserted inside `run_submission_selftests()`, which runs in every profile before the model is allocated. |

Normal (non-injected) runs of `SUBMIT_SMOKE`, `SUBMIT_FAST` and `SUBMIT` on v1.1 complete end-to-end; logs are in the
bundle. `submission.json` is re-validated outside the notebook after every run (ids identical to
`sample_submission.json`, one answer per test input, only `attempt_1`/`attempt_2`, zero illegal grids).

---

## 4. Your remaining open items, answered

* **Concurrency 12 vs the ~160–168k-token KV cache.** Unresolved by design: it has never been run. With the
  8,192-token budget a sequence's context stays near 12–15k, so 12 sequences sit at the edge and vLLM may preempt and
  recompute (slower, not fatal). `SUBMIT_SMOKE` on real hardware is the cheap way to find out; if its wave timings
  look bad, dropping to 8 is a one-line profile change.
* **`SUBMIT_FAST` leaves only 2,048 exploration tokens.** Agreed, and stated in the notebook: it degenerates to
  "think briefly, then commit a program through the finalizer", i.e. most of PIV's mechanism is removed. It exists
  only because the owner wants a submission inside one working day. `SUBMIT` is the profile that keeps the mechanism.
* **Repair cost in `SUBMIT_FAST`.** Fair: the leftover of a 4,096-token budget rarely admits a repair, while the
  round still re-verifies every candidate (~1–3 s each). It is deadline-guarded and now fully reported, so the run
  report will show exactly how many were `repair_admission_failed`; if that is most of them, the next version should
  skip the round for that profile.
* **Runtime provenance / `language_model_only=True`.** Both accepted as future work, not changed here: swapping the
  wheel source or an `LLM(...)` flag on the eve of a run would invalidate the only GPU evidence we have (v2.4G).
* **transformers 4.57.6 vs the model config's `5.8.0.dev0`.** Your conclusion matches ours, and we have direct
  evidence: v2.4G ran this exact pin on 4×L4 for 78 minutes with 16/16 structured-output parses and a passing
  positive control.
