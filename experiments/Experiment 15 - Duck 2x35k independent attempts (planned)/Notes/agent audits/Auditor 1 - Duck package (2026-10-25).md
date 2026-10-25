# Auditor 1: Duck package (E15B), 2026-10-25

**Scope:**
- `src/duck_b/arc2_duck.py` (LF), diffed against Exp 12 `duck_arc2/arc2_duck.py`;
- `src/duck_b/tool_agent.py` and `src/duck_b/python_tool_sandbox.py`, diffed against Exp 12 `vendor/inference/agent/`.

The audit was read-only. No project file was changed.

**Scratch scripts and reproductions:** `D:\ARC AGI 2\_scratch\e15b\agent1\`
- `repro_common.py`: the `test_duck_b.py` preamble, including the Windows sandbox shim.
- `repro1_harvest_overwrites_partial.py`
- `repro2_commit_after_rejected_submit.py`
- `repro3_sandbox_escape.py`
- `repro4_deep_recursion.py`
- `repro5_harvest_sets_commit.py`
- Real-data scripts: `p_minus_g.py`, `harvest_real.py`, `harvest_vs_partial.py`, `multi_call.py`, `def_forms.py`.

**Real data used:** the Exp 14 runs `EVAL120G4B XHIGH noE16`, `EVAL120G4B XHIGH E16` and `E14S SUBMIT commit1`. Between them they hold 395 + 12 branch records and about 3,160 model replies.

**Tests:**
- `python test_duck_b.py` gives ALL PASS (0 SKIP).
- `python verify_items.py` gives ALL CODE ITEMS PRESENT.

Neither test covers findings 1-3.

---

## Findings, ranked by impact

### 1. The end-of-attempt harvest overwrites the partial-best program, so the partial test runs a different function

- **Severity:** MEDIUM.
- **Status:** CONFIRMED (reproduced, and seen in real-run data).

**Where:**
- `arc2_duck.py:2903-2911`: `BranchSession.run` calls `arc2_b_reasoning_harvest` first, then `arc2_b_partial_test`.
- `arc2_duck.py:2241`: the harvest runs the harvested module through the normal `_run_python_tool`.
- `arc2_duck.py:2026`: `_run_python_tool` calls `_record_retained_functions`, which replaces `_kept_functions[name]` with the harvested source.
- `arc2_duck.py:2278`: `arc2_b_partial_test` then calls `_verify_submission(best["name"])`, which uses the retained sources at that moment.

**Failure scenario:**
1. An attempt ends at 35k with no verified candidate. Its progress best is `transform`: the executed version scored k/n.
2. Its thinking also contains a different `def transform(grid)`: an earlier draft or a variant that never ran.
3. The harvest re-defines `transform` from the thinking. The draft is not exact, so `new_accepted` is 0.
4. The partial test (#7) then host-runs the draft, not the program that earned the progress score.
5. The candidate stored in `checked` (the attempt_2 tier, or attempt_1 for an unverified task) is the output of a program that was never checked. The real partial program is never run on test.

**Evidence (reproduction, `repro1_harvest_overwrites_partial.py`):**

```
progress best after the executed version: {'name': 'transform', 'k': 1, 'n': 2, ...}
kept transform (executed): 'def transform(grid):\n    return [r[::-1] for r in grid]'
harvest: {'ran': True, 'texts': 1, 'names': ['transform'], 'chars': 37, 'new_accepted': 0}
kept transform after the harvest: 'def transform(grid):\n    return [[0]]'
partial_test verdict: train_exact [False, False] test_outputs [[[0]]]
BUG REPRODUCED
```

**Evidence (real runs, `harvest_vs_partial.py`):** "Partial-tier" branches are branches with no accepted candidate and a progress best with k > 0 or acc > 0.

| Run | Partial-tier branches | Best name re-defined by the harvest | Re-defined with a source different from the last executed one |
|---|---|---|---|
| XH noE16 | 7 | 5 | 4 |
| XH E16 | 8 | 1 | 1 |

- One example is `0934a4d8_b0`. Its best `transform` is exact on 4/4 train pairs but gave no valid test grid.
- Its last executed definition is 1,411 characters long; the harvested one is 1,544 characters.
- In B mode more attempts end at the budget without an accept, so this path is hit more often.

**Minimal fix** (either one):
- **(a)** Run `arc2_b_partial_test()` before `arc2_b_reasoning_harvest()` in `BranchSession.run`.
  - When the partial program is really train-exact it becomes accepted, and the harvest is then skipped as intended.
- **(b)** In `arc2_b_reasoning_harvest`, save `saved = dict(self._kept_functions or {})` before the `_run_python_tool` call. Restore `self._kept_functions = saved` when `out["new_accepted"] == 0`.

Option (b) keeps today's time order (the harvest first, within the grace).

### 2. No commit message after a rejected `submit` inside the reserve; the harvest also records a commit the model never saw

- **Severity:** LOW-MEDIUM.
- **Status:** CONFIRMED (reproduced).

**Where:**
- `arc2_duck.py:2095`: the tool-result note is only added `if self._arc2_pending_submit is None and self._arc2_b_in_commit()`.
- `arc2_duck.py:1565-1567`: a turn after an executed step (Duck sets `_resume_after_yield = not step_executed`, `tool_agent.py` around line 7247) takes the `"rejected"` branch. The commit check (line 1577) is only in the resume branch.

**Failure scenario:**
1. In the last 5,120 tokens the model calls `submit(transform)` and the submit is rejected. The submit is an executed step, so the turn ends.
2. That tool result has no commit note.
3. The next opener is "Your submission ... was not accepted ... fix the rule or the code, check() it, then submit again". The commit text is missing.
4. That request gets cap = left (≤ 5,120). The model is pushed to "fix" rather than commit.
5. If that request is cut, the next opener carries the commit text, but only about 256 tokens are left by then.
6. So the planned commit turn of #48 is effectively lost for this attempt.

**Evidence (`repro2_commit_after_rejected_submit.py`, session initialized as `analyze()` does):**

```
in commit: True
step_executed (turn ends): True | submission: `transform` is wrong on training pair(s) [0] of 1 ...
commit note in the tool result: False
next opener: Your submission `transform` was not accepted: ... Continue fro...
commit text in the opener: False | commit_sent: False | next request cap: 4000
```

**Side effect on the B metrics (`repro5_harvest_sets_commit.py`):**
- The end-of-attempt harvest calls `_run_python_tool`. Inside the reserve, line 2096 then calls `_arc2_b_commit_text("tool")`.
- That records `commit_at_tokens` / `commit_via="tool"` for an attempt that never sent a commit request:

```
before harvest: {'commit_at_tokens': None, 'commit_via': None, 'commit_requests': 0}
after harvest:  {'commit_at_tokens': 34990, 'commit_via': 'tool', 'commit_requests': 0}
```

`b_summary.commit_turns` (builder line 459) counts `commit_at_tokens is not None`, so it over-counts.

**Minimal fix:**
- **Opener:** in `_arc2_r6_opener`, test `self._arc2_b and self._arc2_b_in_commit()` before the `"rejected"` branch. Use the commit text there, with the rejection summary added in front.
- **Tool result:** at line 2095, add the commit note also when a submit was verified and not accepted. That case is set after the verdict at line 2102, so the note has to go after it.
- **Harvest:** set a flag such as `self._arc2_b_harvesting = True` around the harvest's `_run_python_tool` call, and skip both notes while it is set.

### 3. Yield-after-batch runs the remaining calls after a deadline stop with unclamped 30-s sandbox calls

- **Severity:** LOW.
- **Status:** PLAUSIBLE (from the code; not reproduced).

**Where:**
- `tool_agent.py:7108`: `continue` for any yield reason, including `stop_requested` at the deadline.
- `arc2_duck.py:2009`: `_run_python_tool` uses `timeout_seconds=self._python_timeout` (30 s) with no deadline clamp. `_arc2_autocheck` and `_verify_submission` are clamped; this call is not.

**Failure scenario:**
1. A reply with 3 python calls arrives just before the deadline.
2. The deadline passes after call 1. Calls 2-3 still run, up to 30 s each.
3. The end-of-attempt harvest then sees `left = deadline + 30 − now < 20` and is skipped.
4. The partial test's verification is skipped too: past deadline + 29 it is "skipped: the run deadline is reached".

**Exposure:**
- 7-9 % of real replies have more than one tool call (`multi_call.py`: 112 / 1,569 and 139 / 1,590).
- The window is only the last seconds before the deadline, so it affects only the attempts cut by the deadline.
- The join grace (90 s) still covers 2 calls.

**Minimal fix:** clamp the sandbox timeout in `_run_python_tool` to the deadline plus grace, as `_arc2_autocheck` already does:

```python
timeout_seconds=max(1, min(self._python_timeout, int(self._arc2_deadline + getattr(self, "_arc2_verify_grace", 0.0) - time.monotonic())))
```

Use it when `_arc2_deadline` is set. Stopping after a budget stop is not needed: tool calls generate no tokens.

---

## Checked, no issue

### Scheduler (`solve_all_r6`)

- **The start barrier holds.**
  - `stage1_left` is decremented when a stage-1 item is popped.
  - Stage-1 items sort first in the heap ((0, …) < (1, …)), so a stage-2 item can only be popped once no stage-1 item is queued, that is when `stage1_left == 0`.
  - The re-push / `cond.wait(5)` branch is therefore unreachable in practice, and there is no spin.
  - Each task's attempt 2 is queued by `stage2_items(tid)` after its own attempt 1 ends.
- **No pause, chunk or resume.**
  - `E12_R6_FIRST_EXPOSURE` is 0, so `pause = None`.
  - `E12_R6_FAIR_SHARE` is 0, so `share_tokens()` is `None` and `pause = None`.
  - `BranchSession.run` therefore always ends with `status = "done"`, `g412_defer_b1` is always False (it needs a paused b0), and `E12_R6_P90_CUTOFF` is 0.
- **The running count is always released.**
  - `publish` is wrapped (#18).
  - The decrement shares one critical section with `stage2_items`.
  - The `worker()` restart wrapper only re-enters `_worker_body`.
  - A setup exception in `s.run` is caught, and the session is set to done.
- **No deadlock on the KV gate.**
  - It is overridden when nothing is running.
  - A stale or missing sample admits.
  - `min_admit_sec=300` makes every worker exit when less than 300 s are left.
  - The join grace of 90 s exceeds the bounded end-of-attempt work (about 35 s: one 30-s sandbox call plus verifications clamped to deadline + 29 s).
- **Freeze and late callbacks are safe.**
  - `publish` drops non-final records after `closed`.
  - `write()` is a no-op once frozen.
  - The `_agent_state` fallback ("counters only") updates `self.record` in place, so the older candidate lists stay in the record.

### Caps and context (`_arc2_apply_cap`)

- **Caps:** the first request gets 30,904; inside the reserve the cap is `max(256, left)`. An attempt can overshoot 35,000 by at most 255 tokens.
- **The cap is re-derived before every request,** through `_trim_messages_for_context`, both at turn start and per request.
- **No trim inside an attempt.**
  - The trimmer fires only when (estimated prompt − generated so far) > 68,584.
  - Real data (`p_minus_g.py`) puts this value at: p50 8.8k, max 19.7k over 395 EVAL120 branches, and 0 branches above 50k.
  - The history turn cap is 150 assistant turns; the real maximum is 45 requests per branch.
  - The rolling summary is off (`ARC3_SUMMARY_INTERVAL_TOKENS` = 0) and `ARC3_STUB_DEAD_REASONING` is off.
  - The prompt grows with generated tokens in the real data, so the reasoning really is re-sent.
- **The commit turn is delivered** (outside finding 2):
  - With the 2,048-token yield, every long request is its own turn.
  - The next opener carries the commit text in all resume cases: length cut, cut call, bare yield and timeout.
  - A tool result inside the reserve also carries it.

### `SERVER_DOWN`

- **The wait counts nothing and loses nothing.**
  - The wait is bounded by the deadline and retries the same step.
  - Duck rolls the turn's tokens back on a request failure (`tokens_at_turn_start`).
  - `_post_with_retries` retries connection errors 3 × 5 s.
- **A dead process is caught quickly:** the watchdog polls every 10 s, well before 10 consecutive failures (about 25 s each).
- **Read timeouts:** they are not retried, so a deadline-clamped request ends at the deadline.

### Tool routing and the autocheck pattern

- **`_dispatch_tool`:** submit, check and check_many are routed to python. A non-identifier argument falls back to `transform`. Other unknown tools behave as before.
- **The autocheck regex:**
  - In 401 real transcripts the only definitions the widening matches are 2 × `trans(g)` (wanted) and `candidates(g, colors, rule)` (3 arguments, so excluded by `_r6_single_grid_arg`).
  - `transpose` and `translate` are excluded.

### Reasoning harvest on real thinking (`harvest_real.py`)

- 46 branches produce harvest names.
- 0 contain an import outside the allowed set, so no harvest dies on `import numpy`.
- 0 re-define a prelude or helper name (for example `check`).
- The harvested module is at most 7.6k characters, so the 24,000-character cut never applies.

### Selection

- **`select_attempts_r5` and the rule library:**
  - Rules come after every accepted and partial program.
  - The rule library runs inside its own try (A6, tested), so a rule error cannot cost a program.
  - `build_submission` falls back per task.
  - In pass@2 the a1 / a2 order between two attempts' bests does not matter: both are in the two slots.
- **A submit in the middle of a batch** (rollback through `step_executed`) occurred in 0 of about 3,160 real replies.

### Sandbox (`python_tool_sandbox.py`)

- **The shim is not an escape regression.**
  - The sandbox was already escapable from Python with the flag off: `print.__self__.__import__('os')` works (`repro3_sandbox_escape.py`).
  - The shim's `setrecursionlimit.__globals__` route gives nothing new.
  - The isolation is the OS layer: rlimits plus the notebook's process limits.
- **`time` is bounded:** `time.sleep` is ended by the host deadline kill.
- **The import messages** only change the error text.
- **Deep recursion works:** recursion 19,000 deep, both pure Python and through `any(generator)`, works at the 20,000 clamp on local Python 3.14 (`repro4_deep_recursion.py`). Python 3.12 is not verified here: run it on the Kaggle CPU kit.

### Vendor patch (`ARC3_YIELD_AFTER_BATCH`)

Both test cases behave as intended:
- **ON:** both tool results are kept.
- **OFF:** the upstream rollback applies.
