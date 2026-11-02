# Subagent a232801895039f281 hand-back (verbatim)

## E29 SCHED supply (scheduler S-1): done. All 29 tests pass on Python 3.12 and 3.14.

The two Build files are written, plus this report saved as `D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)/Experiment Review/Agent Reports/E29_REPORT_sched_supply_2026-11-01.md` (11,032 B). No git, push, Kaggle or GPU. Scratch is in `D:/ARC AGI 2/_scratch/e29_sched_build/`.

### Files and shas (all LF only)
- **`D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)/Build/e29_sched_patch.py`**
  - sha256 `c52211fed9acd27db5073c913260c7b1bd12edf4f70de7c3cba474972cc8cb0d`
  - 35,467 B, 617 lines
- **`D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)/Build/e29_sched_tests.py`**
  - sha256 `165bdc8b9e23f2002499d200d2f30e087f8686c662aebbd88dd1d7759caff37c`
  - 59,687 B, 1,051 lines
- **arc2_duck.py chain:**
  - pre `b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128`
  - E28 post `35b8d8577ab05be60e4e66f80400e57ccec3005dfad9b5290d9fa6f53adee57f`
  - **E29 post `d8603109f9530896e9e4a56e52806202fc2cd4c58bd664d9a25222fe6f7592c2`**
- **E29_PATCH** (E28 post -> E29 post):
  - sha `c438e55b70e45ade9230e170e43b77bd84742db25a2eb2cbaeb0c9da190ba8c6`
  - **12 hunks, +214 / -11 lines**, 325 patch lines including a 3-line header
  - All hunks are inside the E28 block (E28-post lines 3860-4435); the last one appends the E29 block.
  - The E28 diff (sha 8089ee65) and the applier (sha b871f0e4) are reused unchanged.
- **Generator:** `gen_e29.py` makes exact-once edits plus an appended block on `E28S.patched_text(orig)`, then runs `difflib` with n=3. `fill_e29.py` fills `e29_sched_patch.template.py`. Re-running both rebuilds the module byte for byte (same sha).

### Tests
- **Results:** 29 tests, 0 failures. `py -3.12` (3.12.10) exit 0; `python` (3.14.6) exit 0. The final count is printed and the exit code is 0 / 1.
- **Stress:** 20 of 20 passed with 10 processes in parallel (5 per Python, two rounds). That was after hardening three load-sensitive tests:
  - the no-idle test now has a 6 s admission window;
  - the deadline-guard test uses deadline 8 s with min_admit 6.5 s;
  - the time-cut off-identity check compares only the deterministic parts.
- **Do not run them with `python -I`.** `-I` hides the user site-packages, which hold `requests`, and the Duck vendor `tool_agent` needs it.
- **e28_sched_patch.py location:** the tests import it from the E28 Build folder (env `E29_TEST_E28_DIR`). In the notebook TMP both modules must sit side by side, because e29_sched_patch imports `e28_sched_patch` from its own directory.

**What the tests cover:**
- **Patch handling:** apply / verify / revert from ORIGINAL and from E28_PATCHED; INVALID_MIXED, marker-less and CRLF input refused with nothing written; text mode; py_compile; strict diff; hunk counts.
- **E29_SUPPLY unset == E28 post file:**
  - single worker: strict comparison of records, SCHED_LAST["e28"] minus timing fields, the other SCHED_LAST keys and the submission;
  - 40 tasks × 6 workers;
  - the real NB5 Reservation backfill scenario;
  - the time-cut and admission-window scenarios;
  - E28 off == the original rev 6 scheduler;
  - real K100Q / W4 records: same selection and slot tables, including E29_SUPPLY=1 with no S records.
- **Supply behaviour:**
  - S1 when the queues are empty, with exact order and views;
  - S2 only after S1 is exhausted, with 120,000 tokens, window 130,560 and the scaled B2 timeout;
  - A / B / C priority over supply, checked with lock-ordered timestamps from patched `_e29_take` / `e28_c_key`;
  - the per-task cap (default 16, and the env override);
  - supply stops after a TRAIN_PASS, and S4 then fills attempt_1;
  - the deadline guard;
  - idle time is 0 while supply exists, and all 6 workers run supply at once;
  - 3 supply attempts of one task run at the same time;
  - fewest-attempts-first ordering;
  - the projected and slots Reservation modes, with no leaks;
  - no KeyError on any phase counter.
- **Answer rule:** 15 synthetic cases, all the E28 cases (identical), and fake-run records (identical).
- **Real Arc2ToolAgent with branches 3 / 4 / 5:** budget, window, prompt budget and first request cap.

### API (names as the builder will call them)
- **`E29S.apply_e29_sched(E12_DIR)`**
  - ORIGINAL -> E28 diff, then E29 diff, with one atomic write.
  - E28_PATCHED -> E29 diff.
  - FULLY_PATCHED -> `"already_applied"`.
  - Anything else -> `RuntimeError("E29_SCHED_PRE_SHA_MISMATCH ...")`.
  - A bad result raises `E29_SCHED_VERIFY_FAILED`.
  - Record keys are E28's plus `e28_step` and `e29_sched_sha`.
- **`E29S.verify_e29_sched`** returns `(ok, why)`.
- **`E29S.revert_e29_sched`** restores the exact ORIGINAL bytes from FULLY_PATCHED or E28_PATCHED.
- **`E29S.tree_state`** returns ORIGINAL / E28_PATCHED / FULLY_PATCHED / INVALID_MIXED.
- **`E29S.server_env(supply=True)`** = E28 env + `E29_SUPPLY=1`; with `supply=False` it is exactly E28's env.
- **Other names:**
  - `E29S.ENV_NAMES` = E28's names + `E29_SUPPLY` and `E29_SUPPLY_MAX_ATTEMPTS`;
  - `E29S.E29_SCHED_SHA` = pre, e28_post, post, patch, e28_patch and applier shas, plus a nested `arc2_duck.py` entry;
  - `E29S.NOTEBOOK_EDITS` and `apply_notebook_edits` are E28S's objects;
  - `T1` / `B2_TOKENS` / `MAX_MODEL_LEN` / `B2_WINDOW` are re-exported.

### Design decisions
1. **The cap counts the C still to come.** E28 always queues phase C, and supply can start while that task's B is still running. So the cap counts attempts started (running ones included) plus the pending C. A + B + C + supply then never exceeds `E29_SUPPLY_MAX_ATTEMPTS` (valid for caps >= 3). Without this, a concurrent run could go one over the cap.
2. **Supply follows the same admission rules as other items.** It goes through the same `_fits` hook, failed-budget skip, KV gate and forced head admit (when nothing runs). Each pick counts in `admission_checks` / `forced_admits`.
3. **Supply only when all queues are empty.** It is considered only when qa, qb and qc are all empty. In projected mode, an A / B / C head that does not fit still waits as `nothing_fits`; supply does not backfill it, as the spec says.
4. **Plans and report fields:**
   - supply plans carry a `view` key, which `sess()` uses;
   - rows add `s_index`, `s_reason` and `view_plan`;
   - the supply report also has `attempts_by_task` and `s_class`;
   - slot rows get `s_full` only when S records exist, so they stay identical to E28's otherwise.
5. **Dispatch priority:** supply attempts get queue positions 300000 + n, so the engine gate ranks them after A / B / C.

### Duck-code findings (arc2_duck.py b9f67010)
- **No `branch == 1` case reaches branches >= 3.** The budget, window and prompt budget come from `E28_PLANS[(task, branch)]` through `_e28_plan_for`, which works for any branch number. Without a plan, `b_attempt_budget(branch >= 1)` would give the attempt-2 budget, but every supply attempt has one.
- **The only other branch-related difference is view-based:** `first_b1` (24,576) applies to any view other than "facts".
- **Views:** odd s_index -> `b1_view`, even -> "facts". Both notebooks pass `E12_R6_B1_VIEW="facts"`, so in production every supply attempt runs "facts".
- **Notebook side:** N1 / N2 / N4 size the reservation from `plan["budget"]`, so S2 gets B2-sized blocks. The CELL 7 report counts branch >= 2 as "extra" attempts; that is report-only, so use `SCHED_LAST["e28"]["attempts"]` for correct per-phase numbers.
- **Late supply attempts can end as TIME.** The unchanged E28 live rule marks any attempt that ends within 5 s of the deadline as TIME. So a supply attempt still running when the deadline cuts it is classified TIME.
