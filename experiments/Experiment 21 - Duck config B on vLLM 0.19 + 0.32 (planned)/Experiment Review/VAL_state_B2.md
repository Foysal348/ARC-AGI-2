# VAL state B2: Experiment 21 wrapper, per-half isolation and submission (2026-10-28)

**Reviewer:** independent second reviewer, scope B. Read-only: no project file was edited and Kaggle was not touched.

**Scope:**
1. Does `build_e21.wrap` / `module_bindings` preserve the behaviour of E15B CELL 6 / CELL 7? Proved by execution.
2. Per-half isolation: what half A leaves to half B, and whether both halves get the same Duck configuration.
3. `submission.json` in commit mode and in hidden-rerun mode.

**Scripts and raw outputs:** `D:/ARC AGI 2/_scratch/e21_val_B2/`, scripts `s1`-`s7` with their `*_py312.json` / `*_py314.json`.
- Python 3.12.10 is the version of the pinned Kaggle image. Every script also ran on 3.14.6, with the same results.
- The Duck code used is the real package `_scratch/donor_map/duck_9e4b` (`arc2_duck.py` + vendor `tool_agent.py`).

## Verdict

- **The wrapper preserves behaviour for these two cells.** It was proved by execution: every global name, every file
  and every bundle member, stdout, exceptions, server calls, HTTP calls and the solver arguments are identical between
  module-level and wrapped runs. The only difference is the `__qualname__` of three lambdas, which nothing uses.
- **The submission is safe in both modes.**
  - Commit: always the complete identity submission, never a half-B-only or empty file.
  - Rerun: half A only, built from half A's records, complete.
- **Isolation has one HIGH gap.** Duck's process-wide priority gate in `tool_agent.py` is not reset between halves.
  - Half B inherits half A's slot count and half A's (expired) clock.
  - Effect: half B is silently capped at half A's stream count whenever half A ran fewer streams than half B.
  - This is plausible in NB1: half A falls back to FP8 with 16 streams (INT4 on vLLM 0.19 is untested, README Finding
    5) while half B runs INT4 with 20.
- **The Duck configuration is otherwise identical in both halves:** budgets 50k/45k, before3, weak commit, extra
  attempts 8, window 100,000, reasoning key `reasoning`, KV gate 0.95, join grace 90 s (s4).
- **Two conditions can still make the halves differ:**
  - the stream count, which is derived per half from vLLM's own KV line;
  - the solve window, which is not guaranteed equal when a launch is slow or falls back (MEDIUM).
- **No BLOCKER.**

## Findings

### HIGH-1: Duck's priority gate survives into half B (slot count + clock of half A)

**Location:**
- `tool_agent.py` 2139-2154 (`_PRIORITY_GATE` singleton, "built on first use") and 2013-2023 (`configure_clock`:
  the earliest start and the earliest deadline win).
- `arc2_duck.py` 2992-2995 (`BranchSession.run` uses `TA._priority_gate()`, `configure_clock(started_at, deadline)`,
  `acquire`) and 3256 (`started = time.monotonic()` per `solve_all_r6` call).
- `e21_engine_new.py` `e21_begin_half` (lines 164-212). It resets `E12A.SERVER_DOWN / SERVER_WAITS / SERVER_LAST /
  SCHED_LAST` but not `tool_agent._PRIORITY_GATE`.
- The comment in CELL 3 says the stream count "is set again from the engine's concurrency before the gate is first
  used (read at call time)". That is true for half A only.

**Evidence (s5_duck_gate.py, real tool_agent 9e4b, Py 3.12):**
- Half A: `ARC3_MAX_ACTIVE_STREAMS=16`, then the first `_priority_gate()` gives a gate with 16 slots.
- Half B: CELL 6 sets `ARC3_MAX_ACTIVE_STREAMS=20`, then:
  - `_priority_gate()` returns **the same object** (`same_gate_object_as_A: true`) with **16 slots**;
  - the clock start and deadline are still **half A's** (`clock_deadline_minus_now_sec: -0.09`, i.e. already past).
- With 20 half-B workers all entering `run()`, **only 16 hold a slot** (`concurrent_holders_while_all_20_want_a_slot:
  16`, `_GATE_STATS.blocked 4`).
- `DUCK_ENV` turns on `ARC3_PRIORITY_TAIL_FADE=1` (fraction 0.2) and `ARC3_PRIORITY_REFRESH_QUEUE=1`. On the shared
  gate, the priority of the same snapshot is **94**; on a fresh gate with half B's clock it is **853**. Tail fade is
  fully on for all of half B ("0.0 minutes remaining").
- Resetting the singleton (the proposed fix) gives a new gate with 20 slots and half B's own clock (`fix_reset_singleton`).

**Failure scenario:**
1. NB1 half A: the INT4 primary on vLLM 0.19 fails, so the FP8 fallback runs with 16 seqs and 16 streams.
2. Half B: INT4 on 0.32, with 20 seqs, 20 streams and 20 workers.
3. Half B never runs more than 16 Duck sessions at a time, yet its report says streams 20 / workers 20.
4. So "INT4 + 20 streams on vLLM 0.32", the variant's main measurement, actually runs at 16. The only trace is
   `server_running_mean` ≈ 16 and the "mean server running < streams - 0.5" WARNING.

The same happens without any fallback:
- whenever `e21_capacity()` gives half A fewer streams than half B (vLLM 0.19 and 0.32 log different KV figures);
- whenever a stuck half-A straggler still holds a slot.

Waiting branches are then ordered by tail-fade priorities computed on half A's expired clock.

The existing CPU harness cannot see this:
- EVAL6 has 6 tasks, which is fewer than 16 slots.
- In `run_NB1_fallbackA_210428` the fake 0.32 server's `max_inflight 20` comes from the 20-request isolation preflight,
  which does not go through the gate. The server stats lines in that run never exceed 6 running.

**Fix:**
1. In `e21_begin_half`, before CELL 6 runs:
   `_TA = E12A._import_duck()[0]; with _TA._PRIORITY_GATE_LOCK: _TA._PRIORITY_GATE = None`. Optionally also reset
   `_TA._GATE_STATS`.
2. Record the gate's slot count per half (for example `E12_RUN["gate_slots"]`) after the solve.
3. Add a harness check with ≥ 20 tasks, or a unit check: after half B, the gate's slot count equals half B's streams.

### MEDIUM-1: The two halves' solve windows are equal only when both launches are fast

**Location:**
- `e21_engine_new.py` 172-174: half A's end is
  `E21_HALF_END_PC = HARD_STOP − 180 − E21_HALF_SOLVE_SEC − E21_B_SETUP_RESERVE_SEC`.
- CELL 6 (`build_e21.cell6` rep at lines 382-385): `_end = min(HARD_STOP − 180, E21_HALF_END_PC)`, then
  `min(_end, now + window)`, then the preflight shift, capped by the same bound.
- The CELL 1 assert (`build_e21.py` 175-176) checks only `2 × window + reserve < budget`, with no setup time.

**Evidence (real profile numbers):**
- `E21X2EVAL120`: budget 43,020 s, hard margin 120, so `HARD_STOP − 180` = T0 + 42,720 s.
- Half A's end = 42,720 − 19,800 − 1,500 = **T0 + 21,420 s**. Half A gets the full 5.5 h only if its post-preflight
  solve starts by **1,620 s (27 min)**.
- Half B gets 5.5 h only if everything from half A's end to half B's solve start takes ≤ 25 min. That includes CELL 7,
  `e21_end_half`, the 0.32 install, the launch, the smoke and the preflight.
- On the normal path, both fit:
  - E18 smoke: 0.19 ready at 668-733 s; 0.32 install 175 s + launch 550 s;
  - R3K2's solve started at 830 s and its CELL 7 took about 10 s (R3K2 Kaggle log).
- Any primary failure eats into that half's window first:
  - the cost is up to `E18_SILENCE_SEC` 600 s, or the 30-min launch timeout, plus the fallback launch;
  - NB1's FP8 fallback weights are not staged, because `E21_STAGE_PATHS` holds only the variant's weights;
  - example: half A starting at 50 min gets about 5.1 h while half B gets 5.5 h.
- s4 confirms that half A's end is fixed in absolute time (`half_end_pc_in_sec` A vs B).

**Failure scenario:** the full-score comparison between halves (vLLM 0.19 vs 0.32) is biased by unequal compute.
- The attempt-1-only score is fairly robust, because stage 1 covers every task first.
- The "full" score and "attempt 2 started" are not.

**Fix (pick one):**
- Cap half B's window at half A's realized window, from `E21_STATE["halves"]["A"]["e12_run"]["deadline_in_sec"]`, so
  both get the same seconds.
- Or record `effective_window_sec` per half in `e21_report.json`, with a `windows_equal` flag and a WARNING line, so
  the comparison is normalized explicitly.

### LOW-1: A half-A thread that is busy when half A sets `_E12_STOP` can survive into half B

**Location:**
- Donor CELL 6 line 427 (`_E12_STOP = threading.Event()`), lines 522 / 315 (watchdog
  `while not _E12_STOP.wait(_WD_POLL)`) and line 719 (`_E12_STOP.set()`).
- After wrapping, these are all *global* names, looked up again on every loop iteration.
- `e21_relaunch` / `e21_launch` read `E21_HALF`, `ENGINE_INFO`, `E11_SERVER` and `E21_HALF_END_PC` as globals.
  `e21_launch`'s failure path does `E11_SERVER = None; _e18_kill_gpu_pids()`.

**Evidence (s7_thread_rebind.py):** a thread started by half A's wrapped cell is still busy when half A sets the event.
It re-checks only after half B's cell has re-bound `_E12_STOP`. It then **keeps running and acts on half B's globals**
(`acted_on: [["B", "server_B"]]`). The `started_in` field prints "B" only because it reads the global; the thread is
started only in half A.

**Failure scenario:**
1. Half A's server dies at least 20 min before the solve end, and the watchdog starts `e21_relaunch()`.
2. Its `wait_ready` can take up to 30 min.
3. The solver returns early (all work done), so CELL 7, `e21_end_half(A)` and half B's begin run.
4. The relaunch then fails (its server was killed by `e21_end_half`). The thread then:
   - calls `_e18_kill_gpu_pids()`, which kills **half B's** vLLM server;
   - sets `E11_SERVER = None`;
   - keeps watching half B.

Arithmetic shows a relaunch always ends before the half's own deadline (duration ≤ about `_left − 1,290 s` when
`E21_HALF_END_PC ≈ _end`). So this needs the solver to finish early during a relaunch. That is improbable with
extra attempts up to 8. Its effect would be severe.

**Fix:**
- Give each half its own stop event captured by value (for example `_stop_ev = _E12_STOP` as a default argument of
  `_e15b_watchdog`), or check `E21_HALF == <half at start>` in the loop.
- In `e21_end_half`, join the watchdog thread (name `e15b_watchdog`) with a timeout before continuing.

### LOW-2: Stale `ENGINE_INFO` if `e21_begin_half` raises between `OUT = …` and `ENGINE_INFO = …`

**Location:** `e21_engine_new.py` 169-180, and RUNNER `build_e21.py` 413-423, which only sets `ENGINE_READY = False`
on a begin error.

**Evidence:** an incomplete stub in my first s4 run made `e21_begin_half` raise at line 177 (`E12A.SERVER_LAST`).
Half B's `e12_run_report.json` then carried `"engine": {"half": "A", …}`, and `e21_end_half(B)` stored half A's
`ENGINE_INFO` under half B. Real raise points in that span are few: `OUT.mkdir` on a full disk, or an env write.

**Fix:**
- Assign `ENGINE_READY = False` and `ENGINE_INFO = {"half": H, …}` first in `e21_begin_half`.
- Or set `ENGINE_INFO = {"half": _H, "fatal": "begin failed"}` in the RUNNER's begin-error branch.

### LOW-3: The `/v1/models` fallback model id never matches E21's served names

**Location:** donor CELL 6 lines 442-445: `E12_RUN["model_id"] = str(SGL_MODEL_PATH)` when `/v1/models` fails.
E21 serves `Qwen/Qwen3.8-27B-INT4` / `-FP8` (`E21_SERVED`).

**Failure scenario:**
- One transient failure of that request (30 s timeout), just after the launch smoke succeeded, makes every Duck request
  of the half use a model name the server does not serve (404).
- The half's branches then burn their failure budgets.
- R3K2 / SGLang served the path, so there the fallback was harmless.

**Fix:** in CELL 6 (`build_e21.cell6` rep), fall back to `E21_SERVED[ENGINE_INFO["cfg"]["weights"]]`, or to
`ENGINE_INFO["attempts"][-1]["models"][0]`.

### LOW-4: CELL 7's final `e11_kill_server(wait=False)` is a silent no-op

**Location:** donor CELL 7 line 403. It calls `E11_SERVER.kill(wait=False)`, but `E18Server.kill(self, fast=False)` has
no `wait` parameter. The `TypeError` is swallowed by `except Exception: pass`.

**Evidence:** s3 `server_kill_calls: []` in every scenario.

**Effect:** none on results. `e21_end_half` kills the server, and keeping it alive until then is what makes
`decode_summary` work. The donor's intent ("stop the server after the report") is not what runs.

**Fix:** none needed. Optionally comment it, or replace it with `e11_kill_server` → `E11_SERVER.kill(fast=False)`
only in the last half.

### LOW-5: Hidden rerun: the post-solve tail now includes `e21_end_half` GPU waits

**Location:** `e21_end_half` (`e21_engine_new.py` 225-237), which runs in RERUN too:
- `kill`: SIGTERM wait ≤ 30 s, then SIGKILL wait ≤ 30 s;
- `_e18_kill_gpu_pids`: nvidia-smi, ≤ 20 s;
- `_wait_gpu_release`: ≤ 180 s.

**Evidence:** the solve deadline is `HARD_STOP − 180`. `HARD_STOP` is 5 min before the 12-h limit, so there are
8 min between the solve deadline and the Kaggle kill. R3K2's post-solve tail (CELL 6 summary → CELL 7 bundle) took
about 11 s (Kaggle log 21,301 → 21,312). The worst case of the added steps is about 4.5 min. Typical is well under
1 min. A run that exceeds 12 h scores 0, even with `submission.json` already written.

**Fix:** when `H == E21_HALVES[-1]`, skip `_e18_kill_gpu_pids` / `_wait_gpu_release`, or use `kill(fast=True)`
(nothing follows the last half).

### LOW-6: Half B is launched without checking that half A's GPU memory was released

**Location:** `e21_end_half` records `st["gpu_release"]`. `e21_begin_half` does not look at it.

**Failure scenario:** `released: False` after 180 s, for example from a zombie engine core. Half B's primary then
likely fails or OOMs and falls to the fallback, which shifts half B's config and window (MEDIUM-1).

**Fix:** in `e21_begin_half(B)`, if half A's `gpu_release.released` is False, run `_e18_kill_gpu_pids()` and
`_wait_gpu_release()` again, and record the outcome.

### LOW-7 (latent): blind spots of `wrap()` / `module_bindings()`; none occurs in the donor cells

**Location:** `build_e21.py` 64-128.

**Evidence (s1 / s2, executed on 3.12 and 3.14):** 31 synthetic cells were run. The following behave **differently**
when wrapped:

| Construct | What happens when wrapped |
|---|---|
| `del <name from another cell>` | `UnboundLocalError`; `ast.Delete` is not handled |
| `match` capture patterns | the names stay local; `match` cases are never walked |
| `type X = …` | `ast.TypeAlias` is not handled; the name stays local |
| `exec("x = …")` | writes into the function's locals |
| `locals()` / `vars()` / `dir()` | return the function's scope |
| a leading docstring | no module `__doc__` |
| annotated assignments (`x: int = 1`, `x: int`) | `SyntaxError` "annotated name can't be global"; caught at build time by `compile()` |

`wrap()`'s multi-line-string assertion checks only `STRING` tokens. On Python ≥ 3.12, f-strings are
`FSTRING_START/MIDDLE/END`, so a triple-quoted f-string with a newline passes the assertion. Its content then gains 4
spaces per line: `"a\n1\nb"` becomes `"a\n    1\n    b"` (s1 `fstring_multiline_probe`). The builder runs on 3.14
here.

**The donor cells contain none of these.** AST scan: no `Match`, `Delete`, `TypeAlias`, `TryStar`, `exec`, `eval`,
`locals`, `vars` or `dir` (one `globals()`, which is fine). No multi-line token of any kind. No odd line-break
characters (`\x0c`, `\u2028`, …).

**Fix:** make the builder fail closed.
- Assert that none of these AST nodes / calls / multi-line tokens (any type) is present.
- Or check with `symtable` that every local of the wrapped function is a comprehension target. That is what s1 does:
  on 3.12+ the only "locals" are the PEP 709 inlined comprehension variables `i k r t v x` (CELL 6) / `f s u x`
  (CELL 7), which never leak.

### LOW-8 (cosmetic): `CONFIG["e12"]["E12_SOLVE_WINDOW_SEC"]` records `None`

**Location:** `CONFIG["e12"]` is built at notebook source line 433, before the E21 block re-defines
`E12_SOLVE_WINDOW_SEC` (line 456). `CONFIG["e21"]["half_solve_sec"]` holds the real value.

**Fix:** none needed, or refresh `CONFIG["e12"]` after the E21 block.

### INFO (checked, no action)

- **Qualnames.** Only lambdas assigned to globals (`_bud`, `_f`, `_ok`) get `e21_cell6.<locals>.<lambda>`. `def` /
  `class` names keep their plain `__qualname__`, because CPython uses the bare name for names declared global (s2
  `qualname` case). Nothing pickles these objects: no multiprocessing or pickle use in the cells or `arc2_duck.py`.
- **The RUNNER catches `BaseException`,** including `SystemExit` and `KeyboardInterrupt`. An exception escaping a
  wrapped cell no longer stops the notebook, as it would at module level in R3K2. This is intended and more robust.
  An escaping exception inside the cell gives the same exception and the same partial state (s3
  `escape_before_solve`).
- **`arc2_duck._RULE_CACHE`** (keyed by `id(task)`) is reused by half B, because it gets the same `CHALLENGES`
  objects. The rule outputs are deterministic, so there is no behavioural effect.
- **Other Duck state that survives a half,** all harmless:
  - `_GATE_STATS` and `_DIAG_STATE` (diagnostics only);
  - `SANDBOX_LIMIT` counters (not reported per half);
  - the `_TOKENIZER` cache.
- **Import-time `tool_agent` constants** are fixed by `DUCK_ENV` in CELL 3 before the import, and are identical for
  both halves:
  - `_LOCAL_ANALYZER_BASE_URL` = `http://127.0.0.1:30000/v1`;
  - `_LOCAL_ANALYZER_MODEL_ID` = `pending`;
  - context window 131,072;
  - `max_output` from `DUCK_ENV` / `E12_DUCK_MAX_OUTPUT`.
  The stale URL / model are **not used**: a real Duck agent built per half uses the explicit `base_url` / model (s6:
  A `127.0.0.1:1234`, B `127.0.0.1:8011`). `E12_DUCK_CONTEXT_WINDOW` (agent init) and `ARC3_REASONING_HISTORY_KEY`
  (`_reasoning_history_keys()` at call time) are read per half.
- **Straggler records cannot pollute half B.**
  - `solve_all_r6.publish` drops every non-final record after `state["closed"]` (C6-4).
  - `probe_loop` exits on `state["stop"]`.
  - Half A's heartbeat and watchdog exit on `_E12_STOP.set()`, apart from LOW-1.
  - s3 shows no thread alive after either cell.

## Scope (1): the wrapper preserves behaviour (proved by execution)

**s1_static.py:**
- The donor files `E15B_R3K2_c09.py` / `c10.py` are byte-equal to the builder's `E15B[9]` / `E15B[10]`.
- Both notebooks' CELL 6/7 cell equals `wrap(cell6())` / `wrap(cell7())` of the current builder. CELL 1, 1b, 3, the
  engine cell, RUNNER and REPORT are also equal, so the notebooks on disk are current.
- symtable of the wrapped functions:
  - names declared global: 53 (CELL 6) and 56 (CELL 7);
  - **no declared-but-unbound extras**;
  - the only function locals are the comprehension variables (PEP 709 inlining artefact; isolation proven in s2
    `comprehension_isolation`).
- Un-indenting the wrapped body gives the cell back exactly.

**s2_synthetic.py:** 31 cells, one per binding construct, run at module level and wrapped from the same starting
globals. Compared: namespace, stdout and exception.
- **23 identical:** every assignment form (chained, nested and starred unpacking, augmented), for/while/else, if/elif,
  `with … as (a, b)`, try/except/else/finally, `except*`, every import form (dotted, as, from … as), decorated
  def/async def/class with `super()`, closures and late binding, `global` inside an inner def, walrus at top level, in a
  comprehension and in a lambda, comprehension isolation with a global of the same name, module-level `global`, `del`
  of an own name, `globals()` read/write, an exception mid-cell (same partial state), a name read before binding, an
  earlier cell's name re-bound, a thread reading globals, `__name__`, `sys.exit` (`SystemExit` in both).
- **8 different, all expected and absent from the donors:** listed in LOW-7.

**s3_real_cells.py:** the real E21 CELL 6 + CELL 7, run at module level vs wrapped.
- Setup:
  - the real `arc2_duck` 9e4b for `build_submission` (rev 5/6 selection + B-mode rules), `workers_for_isolation` and
    `isolation_verdict`;
  - a scripted `solve_all_r6` that publishes records through `on_branch_done` and writes transcripts;
  - a fake clock, fake HTTP (`/v1/models`, the 20-request isolation preflight, `/health`) and a fake server with vLLM
    stats lines.
- Six scenarios: commit, rerun, engine not ready, solver raises (caught in the cell), an exception escaping before
  solving (`E12_OUT.mkdir`), and `SOLUTIONS=None`.
- Result, identical in every scenario:
  - all of the 121-163 resulting globals (after normalizing object addresses and the temp root);
  - all files: `submission.json`, `e12_run_report.json`, `e12_generated_attempts.json`, `sglang_decode_summary.json`,
    and the per-half bundle zip member by member;
  - stdout, exceptions, server kill calls, the HTTP call sequence, and every `solve_all_r6` argument.
- The only difference: the qualname of 3 lambdas. Same on 3.14.

## Scope (2): per-half isolation and configuration

**s4_two_halves.py** runs CELL 8 verbatim (`build_e21.RUNNER`) with the real `e21_begin_half` / `e21_end_half` and
`e21_cfgs` (extracted from the engine cell) and the wrapped CELL 6 / 7 in **one** namespace. Only `e21_launch`, the GPU
helpers and the solver are stubs. Five scenarios: commit normal, rerun, half-B launch fails (both configs), half-A
fallback, and KV capacity A 19 / B 20.

**Records and outputs are separated by half:**
- records: half A holds only A's tags and half B only B's;
- reports: `q27x/halfA|halfB/e12_run_report.json` each name their own half's engine;
- Duck output: `q27x/half<H>/duck`;
- server logs: `vllm_server_half<H>_<tag>.log` under the half's `OUT`;
- bundles: `Q27X_E21_<V>_HALF<H>_DUCK_BUNDLE.zip`.

**Per-half state is reset:**
- `E12_RECORDS` / `E12_RUN` / `_E12_STOP` / `_E12_LOCK` / `E12_OUT` are re-created at the top of CELL 6 before
  anything fallible;
- `E15B_SOLVE_END_PC = inf`, `SGL_URL`, `ENGINE_READY` / `CONCURRENCY` / `ENGINE_INFO`, `E12_STREAMS` and
  `E12_VLLM_CONCURRENCY` are reset by `e21_begin_half`;
- `E12A.SERVER_DOWN / WAITS / LAST / SCHED_LAST` are cleared.

**Names half A leaves behind** (`names_not_rebound_by_half_B`): for example `_e15b_watchdog`, `_WD_POLL`,
`_e15b_solve_end`, `_workers` and `_end` in the B-launch-fails case. No later code reads them:
- CELL 7 reads only names re-created by CELL 6;
- CELL 9 reads `E21_STATE`, the report files and `SOLUTIONS`.

**Same Duck configuration in both halves** (`solve_all_r6` arguments plus env at solve time):

| Setting | Value in both halves |
|---|---|
| `max_generated_tokens` | 50,000 |
| `max_generated_tokens_b1` | 45,000 |
| `b1_accepted_mode` | `before3` |
| `weak_commit` | true |
| `extra_attempts` | true |
| `max_attempts` | 8 |
| `branches` | 2 |
| `b1_view` | `facts` |
| `first_exposure_tokens` | None |
| `min_admit_sec` | 300 |
| `kv_admit_limit` | 0.95 |
| `min_chunk_tokens` | 12,288 |
| `fair_share` | off |
| `p90_cutoff` | off |
| `join_grace_sec` | 90 |
| `E12_DUCK_CONTEXT_WINDOW` | 100,000 |
| `ARC3_REASONING_HISTORY_KEY` | `reasoning` |

- Base URL: port 1234 in half A, port 8011 in half B.
- `ARC3_MAX_ACTIVE_STREAMS` and the workers follow each half's capacity: A 16 / B 20 with the A fallback, A 19 / B 20
  in the KV case.
- The client's streams are honoured only if the gate is reset (HIGH-1).
- The window: see MEDIUM-1.

**Config dicts:** `E15B` / `E12_R6` / the env set in CELL 1 are not mutated by CELL 6 or by Duck. A grep found no
`os.environ` writes in `arc2_duck.py` / vendor besides `duck_env`'s returned dict. The only per-half env writes are
`ARC3_MAX_ACTIVE_STREAMS`, `E12_DUCK_CONTEXT_WINDOW` and `ARC3_REASONING_HISTORY_KEY`.

**Threads:** see LOW-1 and the INFO bullet on stragglers.

**Shared folders:**
- `submission.json` is shared, and is identity in commit (scope 3).
- `q27x/e14` and `e14_state.json` are notebook-wide by design.
- The Duck package dir `/tmp/e12_duck` is read-only after CELL 3.

## Scope (3): `submission.json`

**Commit:** CELL 0 writes the identity of `test_challenges`. CELL 6's `e12_write_submission` (non-RERUN branch) and
CELL 7 rewrite **the same complete identity** in every half. Writes are atomic: a unique temp file + `os.replace`
under `_E12_WRITE_LOCK`.
- s4 commit / B-fails / A-fallback: identical full-schema identity, every task, one `{attempt_1, attempt_2}` per test
  input.
- A half-B-only or empty submission cannot arise.
- The evaluation score lives in `q27x/half<H>/e12_run_report.json` and `q27x/e21_report.json`.

**Rerun:** `E21_HALVES = ["A"]` and `E21_HALF_END_PC = HARD_STOP − 180`. `P` (`RERUN_ALL`) has no `solve_window_sec`,
so `E12_SOLVE_WINDOW_SEC = None` and half A gets the full window.
- The submission is identity overlaid with `E12A.build_submission` of half A's records, only where the row counts
  match.
- It is written after every branch until `E15B_SOLVE_END_PC`, then finally at the end of CELL 6 and in CELL 7.
- s3/s4 rerun: complete, with records-based attempts (for example `t1 attempt_1 = [[2]]`, correct).
- Nothing after half A rewrites it: `e21_end_half` and CELL 9 only read it.
- Engine not ready / solver error: the identity or last-written submission stays valid (s3 `not_ready`,
  `solver_raises`).
- Residual risk: the time tail only (LOW-5).

## What was not verified (out of this scope or not executable here)

- **vLLM metric names and units** (`vllm:kv_cache_usage_perc`, 0-1) on the real 0.19 / 0.32 servers. If one version
  differs, the KV admission gate is fail-open in that half only. This is an engine-scope check for the CPU kit with the
  real packages.
- **Real GPU timings:**
  - how long half A's setup and half B's switch take on the 2026-10-28 image (MEDIUM-1 margins);
  - vLLM's KV figures per version (the streams per half).
