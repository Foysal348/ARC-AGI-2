# E28 SCHED: API for the notebook builders (NB-W4 from NB6Q, NB-K100 from K100Q)

Owner: the E28 scheduler agent. Files: `e28_sched_patch.py` (the module), `e28_sched_tests.py` (local units).
Final shas, hunks and test counts are in section 6.

## 1. What the module patches, and where it runs

- **Target = `arc2_duck.py` of the Duck payload** (`q27x_e15b_duck_9e4b89285136.bin`, manifest sha
  `b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128`). The scheduler (`solve_all_r6`), the
  per-attempt budget / window (`Arc2ToolAgent.__init__`) and the answer selection (`build_submission`) all live there,
  so the patch goes there, not into the notebook cell text (the cells have no scheduler of their own).
- **When:** in CELL 3, right AFTER the manifest check `assert not _bad, ("E12 package hash mismatch", _bad)` and
  BEFORE `import arc2_duck as E12A`. The manifest check keeps verifying the ORIGINAL payload bytes; the module then
  checks the pre-sha itself, writes the patched file atomically, verifies the post-sha + marker + `py_compile`.
- **Env-gated:** with `E28_SCHED` unset (or not `"1"`), the patched file behaves exactly like the original
  (same scheduler, same budgets, same selection; tested on the rev 6 scheduler, the real agent class and the real
  K100Q / W4 records). `E28_SCHED=1` turns on the A / B / C scheduler and the E28 answer slots. The env is read at
  call time (solve start / agent creation / every submission build).

## 2. Functions and constants (names are fixed)

```python
import e28_sched_patch as E28S          # or exec the embedded source into the notebook namespace

E28S.ENV_NAMES        # ("E28_SCHED", "E28_T1", "E28_B2_TOKENS", "E28_B2_WINDOW", "E28_B2_REQUEST_TIMEOUT")
E28S.PATCHES          # {"arc2_duck.py": (pre_sha, post_sha, marker, unified_diff_text)}
E28S.E28_SCHED_SHA    # {"arc2_duck.py": {"pre": ..., "post": ...}, "patch": sha256(diff text), "applier": sha256(applier)}
E28S.T1, E28S.B2_TOKENS, E28S.MAX_MODEL_LEN, E28S.B2_WINDOW   # 104000, 120000, 131072, 130560 (user 2026-11-01)
E28S.server_env()     # {"E28_SCHED": "1", "E28_T1": "104000", "E28_B2_TOKENS": "120000", "E28_B2_WINDOW": "130560"}
                      # = the env of the NOTEBOOK process (Duck runs in it); nothing goes to the vLLM server env.
                      # E28_B2_REQUEST_TIMEOUT is optional (default: base request timeout x 120000 / 104000, B2 only;
                      # the notebooks set it to 9,000 s and E12_DUCK_REQUEST_TIMEOUT to 7,200 s explicitly).

rec = E28S.apply_e28_sched(E12_DIR)            # a directory holding arc2_duck.py (or the file path) -> patches the file
rec = E28S.apply_e28_sched(arc2_duck_text)     # the source text of arc2_duck.py -> {"text": patched, ...}
ok, why = E28S.verify_e28_sched(E12_DIR)       # ok only when FULLY_PATCHED (post sha + marker); why = state + shas
rec = E28S.revert_e28_sched(E12_DIR)           # back to the exact original bytes (pre sha); text in -> {"text": ...}
E28S.tree_state(E12_DIR)                       # "ORIGINAL" | "FULLY_PATCHED" | "INVALID_MIXED"
E28S.NOTEBOOK_EDITS                            # [(label, old, new)] N1 / N2 / N3 / N4 (section 4)
E28S.apply_notebook_edits(text, labels=None)   # -> (new_text, applied_labels); each old must occur exactly once
```

`apply_e28_sched` raises `RuntimeError("E28_SCHED_PRE_SHA_MISMATCH ...")` on an unknown file (nothing written; CRLF is
refused too) and `RuntimeError("E28_SCHED_VERIFY_FAILED ...")` if the result is not the post sha. Already patched =
no-op (`rec["applied"] == "already_applied"`). `rec` keys: mode, file, before, sha_before, applied, after, sha_after,
py_compile, ok, e28_sched_sha (+ text in text mode).

## 3. What changes at runtime (E28_SCHED=1)

- `E12A.solve_all_r6(...)` keeps its signature; with E28 on it runs the A -> B -> C scheduler. Ignored (recorded in
  `SCHED_LAST["e28"]["ignored_args"]`): `branches`, `max_generated_tokens(_b1)`, `first_exposure_tokens`,
  `min_chunk_tokens`, `fair_share`, `share_warmup_sec`, `stage2_p90_cutoff`, `skip_b1_if_accepted`, `extra_attempts`,
  `max_attempts`, `b1_accepted_mode`. Kept: `workers`, `deadline`, `min_admit_sec`, `capacity_probe` +
  `kv_admit_limit` / `kv_poll_sec` / `kv_stale_sec` (the KV gate: waits, never preempts; idle override when nothing
  runs), `join_grace_sec`, `on_branch_done`, `analyzer_timeout`, `b1_view`, `save_request_logs`, `weak_commit`.
- `order` is used as given (no re-sort). Pass the plain order (N3).
- Branch numbers: **b0 = Phase A (attempt 1)**, **b2 = the Phase-B re-run** (fresh agent, new conversation, counts
  as an attempt-1 candidate), **b1 = Phase C (attempt 2)**. No other branches (no stage 3).
- Classification of an attempt (`E12A.e28_classify(rec, budget, at_deadline, window_end_epoch)`), first match wins
  (rev 3, GPT audits 2026-11-01):
  1. TRAIN_PASS = a host-verified full train match (accepted submission or accepted check), however it ended;
  2. B2 = cap verdict `nb5_verdict` in CONTEXT_EXHAUSTED / INCONCLUSIVE_CONTEXT_LENGTH / BUDGET_LIMITED;
  3. TIME = an admission-window note ("nb5: no KV reservation before the admit deadline", SMOKE early exit);
  4. B2 = explicit budget evidence, even with a note: `b_events.tiny_request_skipped` (commit_done_tiny_rest),
     `b_events.commit_at_tokens` set (forced_commit);
  5. TIME = time cut: `status == "paused"`, or at the deadline (live: within 5 s), or the run end within 90 s of
     `window_end_epoch` (records);
  6. FAIL = any other verdict / note (re-run as B1);
  7. B2 = the near-budget heuristic, tokens >= budget - 1,024 (budget_spent);
  8. FAIL = no tokens; else B1 = ended on its own, train < 100 % (self_ended).
- Queues: A = FIFO in `order`; B = task order (B1 and B2 mixed); C = (1, task order) for tasks still without a full
  train match after B, then (2, tokens of the full-match attempt (A, or the B re-run), task order). A free worker
  A B / C item is queued as soon as its evidence exists.
- **Admission, no idle worker (user 2026-11-01; the earlier "hold" is REMOVED):** at every admission check (a worker
  frees, KV frees) the candidates are tried in priority order: B2 items that already waited (did not fit before)
  FIRST, then A (task order), then B (task order), then C1 (task order), then C2 (tokens ascending). The first that
  fits starts at once. A head that does not fit stays at the head and is re-checked at every admission check; a
  lower-priority item that fits starts instead (backfill). A worker waits only when nothing fits at all or the KV gate
  holds; with nothing running the head is admitted anyway (never a deadlock).
  - Fit = `E12A.E28_TRY_RESERVE(key, plan)` (installed by N4): a non-blocking reservation in `NB5_RES`, taken now or
    refused now; the attempt's own `res.acquire` in `_nb5_session_run` then finds its key held. Unset = everything
    fits (= the plain A -> B -> C order). `E12A.E28_RELEASE(key)` is called after every run (idempotent).
  - **Slots mode (the notebooks' mode, K100Q):** need = 1 slot, capacity = workers, so a free worker always fits:
    every free worker takes the next item in priority order immediately (A, then B with B2 first, then C1, then C2);
    no fit failure, no backfill, no hold. The KV gate (reservation fraction, limit 1.0, 5-s probe) can no longer hold
    a freed worker on a stale sample: a sample taken before the latest attempt end is ignored (`kv_superseded`).
  - Projected mode: B2 needs 32 + F blocks (T1: 30 + F); smaller T1 items backfill while it waits; it starts at the first check
    where its blocks fit (not FIFO-starvation-proof: it waits as long as backfilled items keep the pool full).
- Budgets: A / B1 / C = `E28_T1`; B2 = `E28_B2_TOKENS` with the Duck window `E28_B2_WINDOW`; session cap = the
  budget; reserve `E12_B_RESERVE` (4,096) unchanged. A / B1 / C keep the notebook's `E12_DUCK_CONTEXT_WINDOW`.
  B2 request timeout = `E28_B2_REQUEST_TIMEOUT` or base x B2 / T1.
- Every BranchSession carries `s.e28_plan` and every agent `agent._e28_plan` =
  `{"task", "branch", "phase": "A"|"B1"|"B2"|"C", "budget", "window": int|None, "timeout", "b_class", "b_reason",
  "c_priority", "c_key_tokens", "c_from"}`. `E12A.E28_PLANS[(task, branch)]` holds the same dict.
- Answer slots (`E12A.build_submission` -> `select_attempts_e28`, same output format): attempt_1 = best-train
  candidate of A / B (a full B replaces A, the old A answer becomes the attempt_2 fallback; neither full -> higher
  exact-train fraction, a tie keeps A), attempt_2 = Phase C's candidate (accepted first, else its best partial) if it
  exists and differs, else the displaced attempt-1 answer, else the next distinct select_attempts_r5 candidate, else
  the test input.
- Report: `E12A.SCHED_LAST["e28"]` (CELL 6 copies SCHED_LAST into `E12_RUN["rev6_scheduler"]` -> `e12_run_report.json`):
  config (t1, b2_tokens, b2_window, timeouts, workers, min_admit_sec, tasks), ignored_args, warnings (T1 != the
  notebook's E12_B_ATTEMPT_TOKENS; a fixed LOCAL_ANALYZER_SEED), started / done / queued / left per phase,
  first_start_sec / last_start_sec / queue_empty_sec, `idle_worker_seconds` (= nothing_fits + kv_gate; about 0
  except when nothing fits at all) and `idle_sec_by_reason` {nothing_fits, kv_gate, no_runnable (queues empty, B / C
  not yet queued)}, backfill_starts, backfill_log (<= 500: started item + the skipped keys), forced_admits,
  admission_checks, fit_checks, fit_fails, fit_errors, fit_hook, a_class / b_class per task, setup_failed, slots
  (per task, per test input: attempt_1 / attempt_2 source A / B / C / r5 / input / same), and `attempts` = one row per
  attempt: task, branch, phase, b_class, b_reason, c_priority, c_key_tokens, budget, window_plan, window (the agent's),
  agent_budget, session_cap, request_timeout, tokens, best_train {full, k, n, frac, acc, source}, end_reason, kind,
  worker_slot, start_seq, start_sec, end_sec, queues_at_start [A, B, C], and for B / C items queued_sec,
  first_tried_sec, wait_sec (runnable = first tried -> start), fit_fails, backfill_starts_while_waiting; answer_slots.
  `SCHED_LAST["kv_superseded"]` counts gate samples ignored as stale. Each published branch record
  also carries `rec["e28"]` (the same row, without answer_slots). All other SCHED_LAST keys of the rev 6 scheduler
  are present (CELL 7 reads them unchanged).

## 4. Notebook-side items (your builders)

`E28S.NOTEBOOK_EDITS` (verified count == 1 in `nb6q_code_export.py` and `e26_k100q_code_export.py`; the edited exports
compile; values unit-tested):
1. **N1 `_nb5_peak_len`** (CELL 3 NB5 patch): projected peak from the plan budget: `min(MAX_LEN, 8,192 + budget +
   8,192)` -> B2 131,072 (capped by MAX_LEN); T1 120,384 (= NB5's peak_b0 / peak_b1 with T1).
2. **N2 `need` in `_nb5_session_run`**: `cdiv(that peak, block) + F` for an E28 attempt (B2: 32 + F, T1: 30 + F); live mode and
   non-E28 attempts unchanged.
3. **N3 CELL 6**: `E12_ORDER = list(TASK_IDS)` when `E28_SCHED=1`, else the ORDER2 sort unchanged; the builder's
   `task_order` override below it is kept.
4. **N4 CELL 3** (after `E12A.BranchSession.run = _nb5_session_run`): defines `_e28_try_reserve(key, plan)` /
   `_e28_release(key)` and sets `E12A.E28_TRY_RESERVE`, `E12A.E28_RELEASE`. Need = 1 (slots), `start` blocks (live),
   N2's blocks (projected); same grant rules as `NB5L.Reservation.acquire` (idle grant, live hold, a need above the
   capacity is left to C-7); never waits; counts `stats["e28_try_grants"]`. **Required** (without it the scheduler
   treats everything as fitting, and in projected mode a worker would block inside `res.acquire`).

Not offered as edits (engine / report side):
- The server stays at `--max-model-len 131072` (`MAX_LEN`, `E21_VLLM_MAX_MODEL_LEN` unchanged): there is no 262k
  path any more. `nb5_attempt={"b0": 104000, "b1": 104000}` (= `E28_T1`; a mismatch is a warning).
- `E12_DUCK_REQUEST_TIMEOUT`: a T1 request may think 99,904 tokens and a B2 request 115,904; at ~20-30
  tok/s per stream that is over 3,600 s (B2 gets the scaled timeout by default, T1 does not).
- No fixed `LOCAL_ANALYZER_SEED` (else the B1 re-run's first request can repeat attempt 1; a warning is recorded).
- CELL 7 report code treats branch >= 1 as attempt 2 (`_bud` uses `E12_B_ATTEMPT_TOKENS_B1`; `eval_b_attempts`
  counts b2 as "attempt 3+"): b2 is the Phase-B re-run (an attempt-1 candidate, 120k for B2). Use
  `SCHED_LAST["e28"]["attempts"]` / `rec["e28"]` for phase-correct budgets and slots.
- Admission mode: in "slots" mode only C-7 uses N2's `need` (the engine schedules KV: B2 contexts up to 131k raise the
  preemption risk on the W4 pool); in "projected" mode N2 / N4 size the reservation (never preempts).

## 5. Recommended CELL 3 use

```python
E28_SCHED_REC = E28S.apply_e28_sched(E12_DIR)        # after the manifest assert, before `import arc2_duck`
ok, why = E28S.verify_e28_sched(E12_DIR)
os.environ.update(E28S.server_env())                  # turns the scheduler on
import arc2_duck as E12A
```

## 6. Status (final, 2026-11-01, rev 3: classification priority + INCONCLUSIVE_CONTEXT_LENGTH; T1 104k / B2 120k)

- `e28_sched_patch.py` sha256 `108bb75cab0e3a4499621c0e615299b48ccbc783b1eb59cd630f6ed1f8d16730` (69,195 B, LF)
- `arc2_duck.py` pre `b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128` -> post
  `35b8d8577ab05be60e4e66f80400e57ccec3005dfad9b5290d9fa6f53adee57f`; diff sha
  `8089ee65a67c589afb22ac51035c2c181e6a8d6ff954612404374e8c5af9dae1` (8 hunks, +744 / -3 lines); applier sha
  `b871f0e409c5f44ad22a44cb61051138efc220657010b3a85ba4b53a72340e88`.
- Hunks (original line numbers): @@ -1444 (agent plan + prompt budget), @@ -1456 (B2 window), @@ -1464 (attempt
  budget), @@ -2769 (select_attempts_r5 signature), @@ -2829 (r5 candidate lists out), @@ -2839 (build_submission
  dispatch), @@ -3247 (solve_all_r6 dispatch), @@ -3659 (the E28 block appended at the end of the file, +723).
- `e28_sched_tests.py` sha256 `0439d632c6b3995bccb75babb11515d02952f0bb6432378d7c09126518a616a0`: 33 tests, all pass
  on Python 3.12 (7 repeated full runs). Builders: re-embed the module (NOTEBOOK_EDITS = N1-N4, unchanged texts).
