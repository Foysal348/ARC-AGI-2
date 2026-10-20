# Response to the second review (v1.1 → v1.2)

Audience: the reviewer who audited `PIV_Q27_REVIEW_BUNDLE_V11_2026-10-02.zip`, and the project owner.

**All six findings confirmed in code. All six fixed in v1.2.** The P0 fix and the P1 fix are each demonstrated by a
run, not by inspection. Your two mandatory-before-submission items are both done.

`PIV-Q27-SUB-v1_2-HIDDEN-TEST-SUBMISSION.ipynb` — sha256 `0e0d21414455da23618d0c062c9fbb8efb570f121de902719b78c3033e1202af`
(solver notebook unchanged: `PIV-Q27-C1-v2_5A-...` `2188b999efeb6e286fc914bb8c1d876a5c9bdab99537842cd8c94360cd47d5ff`)

---

## 1. Verification of each finding

| # | Finding | Verified | Evidence |
|---|---|---|---|
| P0 | repair stage not fully deadline-guarded | **yes** | In the v1.1 repair cell: the candidate-1 loop called `fresh_verify_solution(code1, eval_challenges[tid], VERIFIER_TIMEOUT_SEC)` with no time check, and after `arc_llm.generate(...)` each job called `fresh_verify_solution(code2, ..., VERIFIER_TIMEOUT_SEC)` with no check either. Only the `generate()` call was guarded. Worst case for 240 tasks: 2 × 240 × 20 s ≈ 160 min against a 75-minute reserve. |
| P1 | bootstrap failure can leave a stale `submission.json` | **yes** | The v1.1 CELL 0 wrote through a temp file and `os.replace`, but never touched an existing file before the `try`, so a failure left whatever was there. Your `{"stale": true}` simulation is correct. |
| P2 | CPU fallback can overshoot by one verifier timeout | **yes** | `_stop_at` was checked, then `_run_verifier_stage(..., 10.0)` started regardless of how little was left. |
| P2/P3 | build scripts still not POSIX-portable | **yes** | My own earlier "fix" produced `PROJECT_ROOT / (r"Research Base Experiment\PIV-...ipynb")` — one string containing a backslash, which is a literal character on Linux. |
| P3 | `competitor_baseline/arc_qwen_phamtruongdat.py` syntax-broken | **yes** | `py_compile` fails; the file ends with `', encoding='utf-8`. My extraction cut the embedded literal at the last `')` instead of at `', encoding=`. |
| P3 | repair `global_waves` inaccurate | **yes** | `"global_waves": 1 if jobs else 0`, while the round can now run many chunks. |

No impact restatements this round: your severities match what the code does. On P0 I agree it is the one that must
not ship, for exactly the reason you give — with one submission per day, a run that dies at hour 11 costs a day.

---

## 2. What v1.2 changes

**P0 — repair stage is absolute-deadline aware end to end.** New helpers in the repair cell:

```python
REPAIR_MIN_SAFE_REMAINING_SEC = 60.0
REPAIR_VERIFIER_MARGIN_SEC = 5.0

def _v25_repair_time_left():
    return float(deadline) - time.perf_counter()

def _v25_repair_verifier_timeout():
    left = _v25_repair_time_left() - REPAIR_VERIFIER_MARGIN_SEC
    if left <= 1.0:
        return None
    return min(float(VERIFIER_TIMEOUT_SEC), left)
```

* candidate-1 loop: before each verifier subprocess, if the timeout is `None` or less than
  `REPAIR_MIN_SAFE_REMAINING_SEC` is left, **every remaining task** is reported `repair_skipped_deadline` and the loop
  breaks — no further subprocess is started;
* candidate-2 (post-generation): the same timeout helper; out of time means keeping candidate #1 with status
  `candidate2_verification_skipped_deadline`, which is the quality-preserving choice anyway, since candidate #2 may
  only replace #1 on verified evidence;
* both calls now pass the shrunk timeout, so no verifier can run past the deadline by a full
  `VERIFIER_TIMEOUT_SEC`. A grep of the repair cell shows zero remaining raw-timeout calls:
  `fresh_verify_solution(code1, ..., _v1_timeout)` and `fresh_verify_solution(code2, ..., _v2_timeout)` are the only
  two.

**P1 — the bootstrap quarantines first.** CELL 0 now renames any existing `submission.json` to
`submission.stale.<pid>.json` *before* the `try` block that reads the test file, and records the path in
`BOOTSTRAP_STALE_QUARANTINED` (reported in `run_report.json`). Renamed rather than deleted so the evidence stays in
the working directory, but out of the submission path either way.

**P2 — CPU fallback timeout shrinks.** `sub_run_program_on_test_inputs` computes
`effective_timeout = max(0.5, min(timeout_sec, stop_at - now))` per query and skips the query entirely below 0.5 s.

**P2/P3 — build scripts.** `PROJECT_ROOT / "Research Base Experiment" / "…ipynb"` as real components. Checked under a
POSIX root: `PIV_PROJECT_ROOT=/tmp/piv_posix_test` resolves to
`/tmp/piv_posix_test/Research Base Experiment/PIV-Q27-C1-v2_5-…ipynb` with no backslash.

**P3 — competitor baseline.** Re-extracted by cutting the embedded literal at `', encoding=`; the file now passes
`py_compile` (20,257 chars).

**P3 — wave count.** `_repair_waves_run` is reset per arm and incremented per generated chunk;
`global_waves` reports it.

---

## 3. Proof, not inspection

| test | result |
|---|---|
| unit test of the new timeout helper against a controlled `deadline` | `deadline` in the past → `None`; `+3 s` → `None`; `+30 s` → `20.0`; `+1 h` → `20.0`. Exactly the intended shrink-and-refuse behaviour. |
| injection `PRIOR=0.0005 DEADLINE=170` on `SUBMIT` (240 tasks) | `repair: 53s left before the safety deadline; skipping candidate re-verification for the remaining 240 task(s)` — the prep loop refuses to start a single verifier. |
| injection `DEADLINE=160` | same, with 33 s left. |
| injection `DEADLINE=200` | the prep loop still had time (≈70 s), completed, and the chunk guard then skipped the 10 jobs: `repair_summary` = 226 `missing_candidate1_record`, 10 `repair_skipped_deadline`, 4 `candidate1_not_eligible` — complete reporting, no silent gaps. |
| normal `SUBMIT_SMOKE` | `Quarantined a pre-existing submission.json -> …/submission.stale.10748.json`, then the bootstrap wrote 240 tasks / 259 queries; self-tests pass; gate passes. |
| normal `SUBMIT_FAST`, `SUBMIT` | see `harness_logs/` in the bundle. |
| build scripts under a POSIX `PIV_PROJECT_ROOT` | backslash-free paths (above). |
| `py_compile` on every packaged `.py` | passes, including the re-extracted competitor baseline. |

---

## 4. Where I agree with your framing, and what is still open

* **Fake-engine caveat.** Agreed and stated in the bundle: these runs prove control flow, schema, state handling,
  fallback and reporting. They say nothing about Qwen-27B behaviour, real vLLM stability on 4×L4, KV pressure at
  concurrency 12, or ARC accuracy. Only `SUBMIT_SMOKE` on real hardware starts answering those.
* **Your recommended sequence is the one I recommend too**: real `SUBMIT_SMOKE` → read the wave timings, memory and
  any preemption warnings in the log → then the full run.
* **One submission per day** is exactly why the deadline work matters more than the solver work right now; the
  expected score is ~0 either way (v2.4G produced 0/16 train-exact programs).
* Still open, unchanged from the last round: concurrency 12 vs the ~160–168k-token KV cache (never run);
  whether `SUBMIT_FAST`'s 2,048 exploration tokens are worth a session at all, given it removes most of the REPL
  mechanism; the repair round's cost in that profile; owning the runtime wheel bundle instead of the third-party
  wheelhouse; and `language_model_only=True` as a possible KV saving. None of these were changed on the eve of a run.
