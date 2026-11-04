# E28 scheduler agent: final report, verbatim

## E28 SCHED: the scheduler module is done. All 27 local tests pass on Python 3.12, across 3 repeated runs.

### Files (all in `D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Build/`, LF, no CR)
- `e28_sched_patch.py`: sha256 `1ff9943ab8f36654268fdcd655cdbdff18c2608ad78c67a6bc31d063430f9efe` (59,308 B, 1,062 lines).
- `e28_sched_tests.py`: sha256 `b1bf3e8728fc999fddd68835d0752c1d0f29b6d0e2c73251b8cea223a7d35e30` (826 lines, 27 tests).
- `E28_SCHED_API.md`: sha256 `db20d1776c79901b03c9fa1babd6a5f19f1e5a79ea3bc14a6ebada73b4fa59b3`.
  - I wrote it first. Final version: API, runtime behaviour, notebook-side items, status.
- Scratch only (not deliverables), under `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/gen/`:
  - the generators `make_patch.py`, `make_module.py`, `module_template.py`, `e28_block.py`;
  - `out/arc2_duck.py`, the patched file (sha `d69ff849…`);
  - `out/arc2_duck_e28.diff`.

### How the harness gets into NB6Q / K100Q, and what I chose
- Both notebooks load `arc2_duck.py` the same way:
  1. CELL 3 loads the payload zip `q27x_e15b_duck_9e4b89285136.bin` (sha `9e4b8928…`, from the payload dataset) with `q27x_payload`.
  2. It extracts it to `E12_DIR=/tmp/e12_duck` and checks the manifest: `arc2_duck.py` = `b9f67010…`.
  3. It runs `import arc2_duck as E12A`.
  4. Export lines: nb6q_code_export.py L389-425, e26_k100q_code_export.py L581-583.
  5. CELL 6 (`e21_cell6`) calls `E12A.solve_all_r6` (nb6q L3489, k100q L4121).
- The scheduler, the per-attempt budget and window (`Arc2ToolAgent.__init__`) and the answer selection (`build_submission`) all live in `arc2_duck.py`. The cell text has no scheduler of its own.
- **Choice:** patch `arc2_duck.py` itself, after the manifest assert and before the import.
- Three notebook-text lines that need the per-attempt plan are offered as build-time edits (`NOTEBOOK_EDITS` N1-N3).
- The notebook agents are already wiring it this way (build_e28_k100.py K5/K6/K7/K11 assert my constants and apply / verify / revert).

### arc2_duck.py patch
- pre `b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128` → post `d69ff849e42235242799d5004744455aa6f152e0f840889eb7f5914238de9cac`.
- Diff sha `af11ac178dcf4ced33acea7d57da9c343f515bfafe24c2b12323e970050241dc`; applier sha `b871f0e409c5f44ad22a44cb61051138efc220657010b3a85ba4b53a72340e88`.
- 8 hunks, +638 / −3 lines. Line numbers are original → patched:
  1. `@@ -1444,10 +1444,11`: `__init__` sets `self._e28_plan = _e28_plan_for(task_id, branch)`; the prompt budget becomes `_e28_budget(self._e28_plan, branch)`.
  2. `@@ -1456,6 +1457,8`: the B2 Duck window (`_arc2_window = plan["window"]`).
  3. `@@ -1464,7 +1467,7`: `_arc2_b_budget = _e28_budget(...)`.
  4. `@@ -2769,7 +2772,7`: `select_attempts_r5(..., _e28_lists=None)` signature.
  5. `@@ -2829,6 +2832,8`: r5 appends its ordered grids per test input when asked.
  6. `@@ -2839,6 +2844,8`: `build_submission` uses `select_attempts_e28` when E28 is on.
  7. `@@ -3247,6 +3254,20`: `solve_all_r6` hands over to `_e28_solve_all` when E28 is on (ignored args recorded).
  8. `@@ -3659,3 +3680,617`: the E28 block appended at the end of the file (patched L3703-4296). It contains:
     - constants `E28_PLANS`, `E28_BRANCH`, `E28_RANK`;
     - `_e28_on`, `_e28_plan_for`, `_e28_budget`, `e28_config`, `e28_train`, `e28_end_epoch`;
     - `e28_classify` (L3766), `e28_b_phase`, `_e28_phase_of`, `_e28_cand`;
     - `select_attempts_e28` (L3847), `e28_slot_table`, `e28_c_key`;
     - `_e28_solve_all` (L3901: pop_item L3971, route L4088, worker L4115, order hold L4138, SCHED_LAST L4262).
- With `E28_SCHED` unset, every path is the original one.

### API (names as requested)
- `apply_e28_sched(target)`: target is the dir, the file path, or the source text (text in → `{"text": ...}` out).
- `verify_e28_sched(target) -> (ok, why)`, `revert_e28_sched(target)`, `tree_state(target)`.
  - States: ORIGINAL / FULLY_PATCHED / INVALID_MIXED. CRLF counts as INVALID_MIXED, and nothing is written.
- `server_env()` = `{"E28_SCHED": "1", "E28_T1": "120000", "E28_B2_TOKENS": "200000", "E28_B2_WINDOW": "261632"}`.
  - This is the notebook-process env; nothing goes to the server.
- `ENV_NAMES` = the 4 above + optional `E28_B2_REQUEST_TIMEOUT`.
- `PATCHES`, `E28_SCHED_SHA`, `T1` / `B2_TOKENS` / `MAX_MODEL_LEN` / `B2_WINDOW`.
- `NOTEBOOK_EDITS` and `apply_notebook_edits(text)`:
  - **N1** `_nb5_peak_len`: the plan's projected peak (B2 = 216,384).
  - **N2** the reservation `need`: cdiv(peak, block) + F (B2 = 53 + F).
  - **N3** no ORDER2 when `E28_SCHED=1`.
- Pattern copied from nb4_ppack / dflash_seam: strict no-fuzz applier, exact pre/post sha, atomic write, py_compile.

### Behaviour (your rule)
- **Branches:** b0 = A, b2 = the Phase-B re-run (new agent, new conversation; counts as an attempt-1 candidate), b1 = C.
- **Budgets:** A / B1 / C get T1 = 120,000. B2 gets 200,000 with window 261,632.
- **Phase B classification (from attempt 1's record):**
  - TRAIN_PASS = a host-verified accepted submission or check.
  - B2 = CONTEXT_EXHAUSTED or BUDGET_LIMITED verdict, `tiny_request_skipped`, tokens ≥ budget − 1,024, or a forced commit.
  - FAIL = a failure note; it is re-run as B1.
  - TIME = cut by the solve window or the admission window; no re-run.
  - B1 = ended on its own with train < 100 %.
- **Queues:** A plain order, then B (task order), then C. C1 = still no full train match, in task order. C2 = by the tokens of the full-match attempt, ascending.
- **Safeties kept:** KV gate (waits only), min_admit_sec, deadline clamp, join grace, final freeze.
- **Slots:**
  - attempt_1: a full B replaces A. If neither is full, the higher train match wins; a tie keeps A.
  - attempt_2: C's answer, else the displaced A answer, else the next distinct r5 candidate.
- **Records:** per attempt in `rec["e28"]` and `SCHED_LAST["e28"]["attempts"]`. Each row has phase, budget, window, best train, tokens, end_reason, kind, worker slot, start_seq, queue lengths and answer slots. Phase counts, timeline, slots and warnings are recorded too.

### Tests (27, all OK)
- **Patch:** apply / verify / revert in dir and text form, shas, INVALID_MIXED refused, strict diff (1-line drift and offset rejected), hunk count.
- **Real records (60 klee100 + 30 W4 attempts):**
  - B2 = exactly {klee100 16b78196 a2 (CONTEXT_EXHAUSTED 95,825), klee100 20270e3b a1 (commit_done_tiny_rest 103,427), W4 2b83f449 a1 (103,761)}.
  - B1 = ∅; FAIL = ∅.
  - TIME = the 5 window cuts that had no train pass (the other 2 cuts had a train pass).
  - klee100: 54 accepted + 2 train-pass checks; W4: 26.
  - end_reason matches k100_quality_report on all 90 attempts.
  - Real C2 order is ascending by tokens.
- **Synthetic edge cases** for every rule.
- **Single-worker exact order:** A as given, then B, then C1, then C2 ascending; plus budgets, windows and slots.
- **Simulation:** 120 tasks × 21 workers: no deadlock (finishes in under 60 s), A drains before B and B before C (checked by the queues seen at each start), budgets per phase, slots.
- **Other scheduler tests:** KV gate (always-high and temporarily-high), time cut (no B / C), admission window, C never overtaking a waiting B2.
- **Env-off identity:** the rev 6 scheduler gives identical results; selection is identical on the real K100Q and W4 records; the real `Arc2ToolAgent` prompt, budget and window are identical.
- **E28 on:** the real agent shows 200,000 in the prompt, window 261,632, first cap 195,904.
- **N1-N3** apply to both exports, the edited exports compile, and the edited values are checked.

### Decisions I made (beyond the rule as written)
- Failed attempts (a note or verdict) are re-run as B1, labelled `b_class` FAIL.
- C2 is sorted by the tokens of the full-match attempt: the B re-run's tokens when B produced it.
- B2 request timeout defaults to base × 200k/120k.
- A later phase is held while an earlier-phase attempt still waits for its NB5 reservation (agent not yet created). This prevents B2 starvation, because projected mode's reservation is not first-come-first-served.

### Open risks
1. The notebooks must set server `MAX_LEN` and `E21_VLLM_MAX_MODEL_LEN` to 262144 and T1 profile = 120000. Otherwise B2 is clamped at 131k.
2. The T1 request timeout of 3,600 s may be too short for a 115,904-token request at ~20-30 tok/s per stream. B2 is scaled; T1 is not.
3. A fixed `LOCAL_ANALYZER_SEED` would make the B1 re-run not independent. Only a warning is recorded.
4. In "slots" mode the engine schedules KV, so 200k B2 contexts raise the preemption risk on the W4 pool. In projected mode N2 sizes the reservation.
5. 200k-context correctness on Flash-Next has not been validated.
6. CELL 7's existing metrics treat b2 as "attempt 3+" and use the b1 budget. Use `SCHED_LAST["e28"]` instead.
7. The attempt_2 rule is applied literally: C's best partial beats another train-exact candidate held by A. This is rare.
8. Tests are local units only. The Kaggle CPU kit with the real e2e is still needed.
