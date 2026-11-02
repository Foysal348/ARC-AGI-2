# E29 SCHED supply (scheduler S-1): agent report, 2026-11-01

No push, no git, no Kaggle and no GPU were used. The only files written in the repository are the two Build files and this report. Scratch work is in `D:/ARC AGI 2/_scratch/e29_sched_build/`.

## Deliverables

- `D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)/Build/e29_sched_patch.py`
  - sha256 `c52211fed9acd27db5073c913260c7b1bd12edf4f70de7c3cba474972cc8cb0d`
  - 35,467 B, 617 lines, LF only
- `D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)/Build/e29_sched_tests.py`
  - sha256 `165bdc8b9e23f2002499d200d2f30e087f8686c662aebbd88dd1d7759caff37c`
  - 59,687 B, 1,051 lines, LF only
- **arc2_duck.py chain:**
  - pre (ORIGINAL): `b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128`
  - E28 post: `35b8d8577ab05be60e4e66f80400e57ccec3005dfad9b5290d9fa6f53adee57f`
  - **E29 post: `d8603109f9530896e9e4a56e52806202fc2cd4c58bd664d9a25222fe6f7592c2`**
- **E29_PATCH:** a strict unified diff, E28 post -> E29 post.
  - sha `c438e55b70e45ade9230e170e43b77bd84742db25a2eb2cbaeb0c9da190ba8c6`
  - **12 hunks, +214 / -11 lines**, 325 patch lines including a 3-line comment header.
  - Every hunk lies inside the E28 block (E28-post line numbers 3860-4435). The last hunk appends the E29 block (+171 in that hunk).
  - The E28 diff (sha 8089ee65...) and the applier (sha b871f0e4...) are reused from e28_sched_patch.py.
- **Tests:** 29 tests, all pass.
  - Python 3.12.10 (`py -3.12`): exit 0.
  - Python 3.14.6 (`python`): exit 0.
  - Stress: 20 / 20 passed with 10 processes in parallel (5 per Python, two rounds), after three load-sensitive tests were hardened.
  - Do not run them with `python -I`: that hides the user site-packages, where `requests` lives (the Duck vendor `tool_agent` needs it).
- **Generator (scratch, reproducible):**
  - `gen_e29.py` builds the E29 post text: exact-once edits plus an appended block onto `E28S.patched_text(original)`, then `difflib.unified_diff(n=3)`.
  - `fill_e29.py` embeds the diff into `e29_sched_patch.template.py` and writes the module (LF, asserts no CR).
  - Re-running both reproduces the module byte for byte, same sha.

## API (fixed names, as the notebook builder calls them)

- **Import:** `import e29_sched_patch as E29S`. It imports `e28_sched_patch` from its own directory, so both files must sit side by side in the notebook TMP.
- **`E29S.apply_e29_sched(E12_DIR)`**
  - ORIGINAL -> E28 diff, then E29 diff, in memory; one atomic write.
  - E28_PATCHED -> E29 diff.
  - FULLY_PATCHED -> no-op, `"already_applied"`.
  - Anything else -> `RuntimeError("E29_SCHED_PRE_SHA_MISMATCH ...")`, nothing written.
  - File mode also drops the pyc, runs py_compile and checks the post sha + marker; a failure raises `E29_SCHED_VERIFY_FAILED`.
  - Text mode returns `{"text": ...}`.
  - Record keys are E28's (mode, file, before, sha_before, applied, after, sha_after, py_compile, ok, e28_sched_sha), plus `e28_step` ("applied" / "already_applied") and `e29_sched_sha`.
- **`E29S.verify_e29_sched(E12_DIR)`** -> `(ok, why)`; ok only when FULLY_PATCHED.
- **`E29S.revert_e29_sched(E12_DIR)`**
  - From FULLY_PATCHED or E28_PATCHED: back to the exact ORIGINAL bytes, every step sha-checked.
  - ORIGINAL: `"not_applied"`.
  - INVALID_MIXED: `RuntimeError("E29_SCHED_REVERT_SHA_MISMATCH ...")`.
- **`E29S.tree_state`** -> ORIGINAL / E28_PATCHED / FULLY_PATCHED / INVALID_MIXED. CRLF text and a missing file count as INVALID_MIXED.
- **`E29S.server_env(supply=True)`** = `E28S.server_env()` + `{"E29_SUPPLY": "1"}`. With `supply=False` it is exactly `E28S.server_env()`.
- **`E29S.ENV_NAMES`** = `E28S.ENV_NAMES + ("E29_SUPPLY", "E29_SUPPLY_MAX_ATTEMPTS")`.
- **`E29S.E29_SCHED_SHA`** = {pre, e28_post, post, patch, e28_patch, applier, "arc2_duck.py": {pre, e28_post, post}}.
- **Re-exported:**
  - `E29S.NOTEBOOK_EDITS` is `E28S.NOTEBOOK_EDITS` (N1-N4, unchanged) and `E29S.apply_notebook_edits` is E28S's.
  - `E29S.T1` / `B2_TOKENS` / `MAX_MODEL_LEN` / `B2_WINDOW` = 104000 / 120000 / 131072 / 130560.
  - Also `E29S.SUPPLY_MAX_ATTEMPTS = 16`, `E29S.patched_text`, and a CLI (`apply` / `verify` / `revert` / `state`).

## Runtime (only with E28_SCHED=1 and E29_SUPPLY=1; env read at the solve call / submission build)

### When supply starts

When the qa, qb and qc queues are ALL empty and the existing `left >= min_admit_sec` guard passes, a free worker takes a supply item instead of waiting.

### Supply kinds

- **S1:** the task's phase-A attempt has ended (in `a_class`, not TIME), it has no TRAIN_PASS among its ended attempts, and not all of its ended attempts are B2.
  - Budget T1, window None, base request timeout.
  - `s_reason` `"no_train_pass"`.
- **S2:** only when no S1 candidate exists. All of the task's ended attempts are B2.
  - Budget `b2_tokens`, `b2_window`, B2 request timeout.
  - `s_reason` `"all_cap_cut"`.

### Order, cap and plan

- **Priority:** fewest attempts started (running ones count), then task order. Several supply attempts of one task run at once when no other eligible task has fewer.
- **Stop:** a task stops getting supply at its first TRAIN_PASS, or at the cap.
- **Cap (design decision):** the cap counts attempts started plus the task's phase-C attempt while it is still to come. E28 always queues C, even while supply runs during that task's B, so the total A + B + C + supply never exceeds `E29_SUPPLY_MAX_ATTEMPTS` (default 16, valid for caps >= 3). Without this, concurrency could overshoot the cap by the pending C.
- **Plan:** phase S1 / S2, branch 3, 4, ... (next unused per task), budget, window, timeout, `s_index` (1, 2, ... per task), `s_reason`, view.
- **Dispatch:** queue position 300000 + n. The engine priority gate ranks supply after A / B / C.

### Admission

- Supply uses the same fit hook `_fits` (`E28_TRY_RESERVE`; slots mode always fits), the failed-budget skip, and the forced head admit when nothing runs. It also counts admission_checks / forced_admits.
- It sits behind the same KV gate.
- A / B / C always come first. At every admission check `has = qa or qb or qc` is evaluated before supply, and supply is only considered when has is False.
- Rows of supply attempts always show `queues_at_start == [0, 0, 0]` (tested).

### Counters and idle accounting

- `e28["started"]`, `["done"]` and `["queued"]` get S1 / S2 keys; `plan()` uses `E28_BRANCH.get(phase)`, so there is no KeyError.
- `route()` notes every ended attempt kind per task, records the S class, and queues nothing for S.
- Idle: with supply on, the "no queued item" wait goes to `idle_sec_by_reason["no_supply"]` and `no_runnable` stays 0. With supply off it is exactly E28.

### Answers (`select_attempts_e28` -> `select_attempts_e29`; the E15B writer is unchanged)

- **attempt_1:** the best-train candidate of A, B, then S in branch order. A candidate replaces the current best only when it is full and the best is not, or when neither is full and its fraction is strictly higher.
- **Displaced:** the best of the rest by (full, frac, earlier).
- **attempt_2:** C, except when C is not full and the displaced is full (then the displaced first). After that come the r5 lists, then the input.
- The slot table uses sources "S<branch>". `answer_slots` maps S rows through `_e29_src_branch`.
- Slot rows gain `s_full` only when S records exist, so with no S records the rows are identical to E28's.

### Report

`SCHED_LAST["e28"]["supply"]` = {enabled, max_attempts, started {S1, S2}, done {S1, S2}, first_start_sec, by_task {task: supply attempts}, attempts_by_task {task: all attempts started}, train_pass_from_supply, slots_from_supply, s_class}. S rows add `s_index`, `s_reason` and `view_plan`.

## Off-identity (E29_SUPPLY unset)

All E29 lines are behind `sup is None` / `_e29_on()`; `plan()` gives identical values for A / B / C. Tested by running the same scenarios on the E28 post and E29 post files:

- **Single worker:** strict comparison of records, SCHED_LAST["e28"] (minus timing fields), the other SCHED_LAST keys, and the submission.
- **40 tasks, 6 workers:** order-insensitive comparison.
- **The real NB5 Reservation backfill scenario.**
- **The time-cut / admission-window scenarios:** deterministic parts only.
- **E28 off (with or without E29_SUPPLY=1):** E29 post == the original rev 6 scheduler.
- **Real K100Q / W4 records:** selection and slot tables are identical (E28 on and off; and E29_SUPPLY=1 with no S records == the E28 rule).

## Duck-code findings for branches >= 3 (arc2_duck.py b9f67010)

- **Budget, window, prompt budget:** `Arc2ToolAgent.__init__` reads the plan with `_e28_plan_for(task_id, branch)` from `E28_PLANS[(task, branch)]`, which works for any branch number.
  - Without a plan, `b_attempt_budget(branch >= 1)` would give the attempt-2 budget, but every supply attempt has a plan.
  - Verified on the real agent: S1 b3 has 104,000 and window 130,560; S2 b4 has 120,000, window 130,560, and its first request cap is 115,904.
- **No `branch == 1` special case reaches a branch >= 3 attempt.**
  - The only other agent difference is view-based, not branch-based: the first cap `first_b1` (24,576) applies to any view other than "facts".
  - Run dirs are `<task>_b<branch>`; publish replaces per (task, branch); `select_attempts_r5` ties on the lowest branch.
- **View:** odd `s_index` -> `b1_view`, even -> "facts".
  - Both notebooks pass `E12_R6_B1_VIEW = "facts"` (E15B #3), so in production every supply attempt runs "facts".
  - The alternation only matters if `b1_view` changes. The "indexed" view was tested to construct correctly on a b5 agent.
- **Notebook side (not patched):**
  - N1 / N2 / N4 size the reservation from `plan["budget"]`, so S2 gets B2 blocks.
  - The CELL 7 report counts branch >= 2 as "extra" attempts. That is report-only; use `SCHED_LAST["e28"]["attempts"]` for phase-correct numbers.
  - The live classifier marks any attempt that ends within 5 s of the deadline as TIME (unchanged E28 rule). Supply attempts started late in the window and cut at the deadline are TIME, and so is their slot data.

## Test notes

- **Harness:** copied from e28_sched_tests.py (the real arc2_duck.py from the payload, the real BranchSession, a fake agent, ResKit with the real NB5 Reservation and the N4 hooks).
- **Fake agent additions:**
  - kind "partial": ends on its own after a partial check, which gives FAIL rather than B2;
  - a per-branch script key `(task, "b<branch>")`;
  - a RunLog wrapper for peak concurrency per phase.
- **The A/B/C-priority test uses lock-ordered timestamps.** `_e29_take` and `e28_c_key` are patched to log times under the scheduler lock. The test asserts that every supply item taken after t001's C item was queued starts after that C.
- **Hardened for load:**
  - the no-idle test uses a 6 s admission window and checks peak S1 = 6;
  - the deadline-guard test uses deadline 8 s with min_admit 6.5 s, because the live TIME rule is deadline - 5 s;
  - the time-cut identity test compares only the deterministic parts.
