# VAL_state_B1: E21 wrapped cells, two-half state, runner isolation, end/report reads

Validator B1, 2026-10-28. Read-only: no project file edited, nothing on Kaggle.

**Inputs**
- `Experiment Notebooks/Build/build_e21.py`: `module_bindings()`, `wrap()`, `cell6()`, `cell7()`, `RUNNER`, `REPORT`.
- `Build/e21_engine_new.py`: `e21_launch`, `e21_relaunch`, `e21_begin_half`, `e21_end_half`.
- Built notebooks, cells extracted:
  - NB1 `Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20.ipynb`, sha ad5b2d1f.
  - NB2 `..._NB2_FP8_SSM16_S16.ipynb`, sha b968e322.
- Donor cells: `_scratch/donor_map/E15B_R3K2_c09.py` / `c10.py`.
- Duck: `_scratch/donor_map/duck_9e4b/arc2_duck.py` and `vendor/inference/agent/tool_agent.py`.

**Experiments:** in `D:/ARC AGI 2/_scratch/e21_val_B1/`, run with Python 3.12.10 (= the Kaggle major version) and 3.14.6.

| Script | What it checks |
|---|---|
| `extract.py` | extracts the cells |
| `codeobj.py` | code objects |
| `symcheck.py`, `astscope.py` | the binding lists |
| `clash.py`, `undefined.py` | name clashes, undefined names |
| `comp_global.py` | a comprehension variable that is also declared global |
| `wrap_forms.py` | binding forms not used by the cells |
| `gate_test.py`, `gate_fix.py` | the Duck priority gate |
| `stale_thread.py` | a stale thread loop |
| `rebuild_cmp.py` | builder output == notebook |

## Summary

| # | Severity | Finding |
|---|---|---|
| H1 | HIGH | Duck's process-wide priority gate (`tool_agent._PRIORITY_GATE`) is built once, in half A, with half A's stream count and clock. `e21_begin_half` never resets it, so half B runs with at most half A's slot count. Proven. |
| M1 | MEDIUM | The watchdog relaunch timeout differs from the donor. It is 60 s whenever <= ~31 min are left to the half's end, so a relaunch in the last ~31 min of a half (also in the hidden rerun) is almost certain to fail. |
| M2 | MEDIUM | The two halves do not get equal solve windows. Half A is capped at T0 + 21,420 s, so it gets 5.95 h minus its setup time (typically ~5.0-5.4 h). Half B gets up to 5.5 h. |
| L1 | LOW | Half-A threads look up the `_E12_STOP` global on every loop. A half-A thread still busy when half B re-creates the event keeps running in half B. Not reachable on the normal timeline (computed); no join or guard. |
| L2 | LOW | `e21_end_half` uses `_E12_LOCK` before it kills the server and frees the GPUs. If half A's CELL 6 failed before creating it: NameError, then no GPU kill or release before half B. |
| L3 | LOW | `e21_begin_half` does not reset `E12_RECORDS` / `E12_RUN`, and `cell6_ran` is set before CELL 6 runs. If half B's CELL 6 failed in its first 20 lines, half A's records would be stored and scored as half B's. |
| L4 | LOW | `e21_begin_half` launches half B without checking that half A's GPU release succeeded. |
| L5 | LOW | Hidden rerun: `e21_end_half` adds up to ~200 s (GPU-pid kill + `_wait_gpu_release`) inside the last minutes, although no next half follows. |
| L6 | LOW | CELL 6: when `/v1/models` fails, the fallback model id is `str(SGL_MODEL_PATH)`. vLLM serves `E21_SERVED[...]`, so every request would 404. |
| L7 | LOW | e21_launch does not cache an `e18_install` failure. If the 0.32 install fails, half B's fallback runs the installer again (up to 2,400 s), with the same result. |
| L8 | LOW | CELL 7's `_r6e_concurrency()` parses SGLang `TP0]` decode lines. On vLLM the field is always `{}`. Informational. |
| L9 | LOW (latent) | Builder gaps: `module_bindings` misses `match` captures and a bare `del NAME`. `wrap`'s multi-line-string assert does not see f-strings on Python >= 3.12. None of these forms occurs in the two cells (verified). |
| L10 | LOW | Every function defined in the wrapped cells has `__qualname__` `e21_cell6.<locals>.…` (cannot be pickled; tracebacks point into cell 8). Nothing pickles them. Informational. |
| L11 | LOW | The runner and CELL 6 use `except BaseException`. In an interactive session a KeyboardInterrupt is swallowed and the next step or half starts. Commit and rerun are unaffected. |

No BLOCKER. On (1), the wrapped functions are semantically identical to the cells run at module level (proven, details below). The hidden rerun path is correct: half A only, its submission is not overwritten, and every step is isolated. H1 affects only half B's measurement (commit), not the LB.

---

## (1) Wrapped function vs module level

### Verified correct

1. **The `global` lists are complete and exact.** Checked with an independent AST scope walk (`astscope.py`), under 3.12.10. It counts Name stores outside function, lambda and class bodies and outside comprehensions; walrus targets in comprehensions; import aliases; except-handler names; def and class names; match captures; del targets.

   | Wrapped function | Module-level bindings | Missing from `global` | Extra |
   |---|---|---|---|
   | `e21_cell6` | 53 | 0 | 0 |
   | `e21_cell7` | 56 | 0 | 0 |

   Neither cell has a `del` target.
2. **No closure over the wrapper's scope.** In the compiled code objects (`codeobj.py`), both `e21_cell6` and `e21_cell7` have `co_cellvars == ()` and `co_freevars == ()`. Every free variable of a nested def, lambda or genexpr is bound by an intermediate nested scope (`_ask`, `steps`, `t_end`, `_rnd`, `probes`, `_st`, `ev`, …), exactly as at module level. So `_e15b_watchdog`, `_e12_heartbeat`, `_e12_branch_done_r6` and the rest resolve every name as a module global, as in R3K2.
3. **The only fast locals are inlined comprehension variables.** These are `i, t, k, v, r, x` (cell 6) and `s, f, x, u` (cell 7), PEP 709 inlined comprehensions; symtable on 3.12 lists them as locals. They are isolated exactly as at module level. `comp_global.py` checks a name that is both declared global and used as a comprehension or genexpr iteration variable, including lambdas capturing it. Output, identical on 3.12.10 and 3.14.6:

   ```
   function: ('outer', [0, 1, 2], [2, 2, 2], {'a': 'a', 'b': 'b'}, [0, 1]) global t = outer
   module:   ('outer', [0, 1, 2], [2, 2, 2], {'a': 'a', 'b': 'b'}, [0, 1])
   ```

   The global `t` is not clobbered by `[t for t in …]`.
4. **Binding forms used in the two cells.** Counts:

   | Form | e21_cell6 | e21_cell7 |
   |---|---|---|
   | Import | 4 | 3 |
   | With | 7 | 2 |
   | AugAssign | 5 | 14 |
   | Lambda | 7 | 18 |
   | ListComp | 20 | 36 |
   | DictComp | 14 | 11 |
   | SetComp | 1 | 7 |
   | GeneratorExp | 47 | 53 |

   The cells also contain tuple targets, for / with / except targets and defs. They contain no walrus, `del`, `match`, `nonlocal`, class, `locals()`, `vars()`, `dir()`, `exec` or `eval`.

   `globals()` appears twice:
   - `_e12_branch_done_r6`: `globals().get("E15B_SOLVE_END_PC", ...)`
   - cell 7: `"E15B" in globals()`

   Both read the same notebook namespace dict as at module level.

   There are 0 multi-line string tokens (`mlstr.py`), so the re-indentation changes no string.
5. **Name used before assignment.** With a `global` declaration, such a read resolves to the existing global, exactly as at module level. Names assigned only on some paths (`_workers`, `_shift`, `_first`, `_ra`, `_hid_end`, `_per`, `_left`, `_kv`, `hits`, `_gh`, …) are only read on the same path. No unconditional read of a conditionally assigned name exists, so half A values cannot leak into half B's logic through them.
6. **Clobbering by short names** (`t, r, a, b, k, v, z, fp, _, _e, _run, _pack, report, branches, …`):
   - `clash.py` uses symtable over all 10 code cells. No function in cells 1-7, the runner or the report reads any of the 109 names, except the intended `E15B_SOLVE_END_PC` (written by `e21_begin_half`) and `_E12_LOCK` (read by `e21_end_half`).
   - Earlier top-level temporaries that the cells overwrite: `k` (CELL 1); `_e` (CELL 1b, 2, 3); `_` (CELL 3, rebound only in the gate-4b profile); `t` (the order cell); `k`, `v` (the engine cell). No later code reads any of them.
   - The runner's own names (`_H`, `_step`, `_fn`, `_e21_cell6_of`) are in neither list.
   - The report cell binds its own `_e, _pack, _run, fp, z, k, r, t, v, a` before reading them.
   - This is the same clobbering R3K2 had at module level.
7. **No name is undefined except dead code.** `undefined.py` finds a single name that is read and never bound in the notebook: `E11Server`, in `_e12_parity` (parity profile only, never called).
8. **The notebooks match the builder and the donors** (`rebuild_cmp.py`; imports the builder without calling `build()`):
   - the wrapped text equals `wrap(cell6())` / `wrap(cell7())`;
   - cells 9 and 10 equal `RUNNER` / `REPORT`;
   - the donor files equal `E15B[9]` / `E15B[10]`;
   - NB1 and NB2 differ only in CELL 1's E21 block (`E21_VARIANT`, `E21_HALF_SPECS`, `E21_STAGE_PATHS`).
9. **IPython specifics.** `_` (the display hook name) is assigned only in the gate-4b branch, and neither cell ends in an expression, so the display hook changes nothing. The functions' `__globals__` is the user namespace, as at module level.

### L9 (LOW, latent): binding forms the builder does not handle

**Where:** `build_e21.py` `module_bindings()` lines 64-115 and `wrap()` lines 118-128.

**Evidence** (`wrap_forms.py`, same on 3.12.10 and 3.14.6):

```
match                  bindings=[] | equivalent=False | module=['y', 'z'] wrapped=[]
fstring_multiline      bindings=['v', 's'] | equivalent=False | s_mod='a\n2\nb' s_fun='a\n    2\n    b'
```

- `walk()` never descends into `ast.Match.cases`, and never adds a `MatchAs` / `MatchStar` name.
- A bare `del X` of a global that is not otherwise assigned would make `X` local, giving UnboundLocalError.
- The assert `tok.type == tokenize.STRING and "\n" in tok.string` does not see multi-line f-strings: since Python 3.12 they are FSTRING_START / FSTRING_MIDDLE / FSTRING_END tokens. They would be silently re-indented. The builder runs on 3.14.

**Impact today:** none. None of these forms is in the two cells (verified above).

**Fix (for reuse):**
1. Also assert on `FSTRING_MIDDLE` (and `TSTRING_MIDDLE` on 3.14) tokens that span lines.
2. Handle `ast.Match` (walk `case.body`; add `MatchAs.name`, `MatchStar.name`, `MatchMapping.rest`) and `ast.Delete` Name targets.
3. Best: a final assert in `wrap()`. Compile the wrapped text and assert that the function's `co_cellvars == ()` and that `co_varnames` holds only comprehension iteration variables (symtable), which is the check used here.

### L10 (LOW, informational): qualified names

Functions defined in the cells have `__qualname__` `e21_cell6.<locals>._e12_heartbeat` and so on. They cannot be pickled. Nothing pickles them: Duck runs them as thread targets and callbacks, `ThreadPoolExecutor` is thread-based, and `copy.deepcopy` copies records only. Traceback line numbers point into cell 8. No fix needed.

---

## (2) State that survives from half A into half B

### H1 (HIGH): the Duck priority gate keeps half A's slot count and clock

**Where:**
- `vendor/inference/agent/tool_agent.py` 2139-2154: `_PRIORITY_GATE` and `_priority_gate()`;
- `configure_clock`, 2013-2023;
- `arc2_duck.py` 2991-2994, where `BranchSession.run` acquires the gate;
- `e21_engine_new.py` line 177, where `e21_begin_half` resets the other arc2_duck state but not this.

**Evidence:**

```python
def _priority_gate() -> _PriorityGate | None:
    """Process-wide, built on first use. ..."""
    global _PRIORITY_GATE
    slots = _max_active_streams()                 # reads ARC3_MAX_ACTIVE_STREAMS now ...
    ...
        if _PRIORITY_GATE is None:
            _PRIORITY_GATE = _PriorityGate(slots) # ... but only the first call builds the gate
        return _PRIORITY_GATE
...
    def configure_clock(self, started_at, deadline):
        self._clock_start = min(self._clock_start, started_at) if self._clock_start is not None else started_at
        if deadline is not None:
            self._clock_deadline = min(self._clock_deadline, deadline) if ... else deadline   # earliest wins
```

```python
E12A.SERVER_DOWN.clear(); E12A.SERVER_WAITS.update(waits=0, sec=0.0); E12A.SERVER_LAST.clear(); E12A.SCHED_LAST.clear()
```

That begin_half line resets every other piece of arc2_duck state, but no `_PRIORITY_GATE`. CELL 6 sets `os.environ["ARC3_MAX_ACTIVE_STREAMS"] = str(E12_RUN["streams"])` in each half. Half B's value is ignored, because the gate already exists.

**Proof** (`gate_test.py`, 3.12, the real vendor `tool_agent`):

```
half A gate 1641828787376 free 16
half B gate 1641828787376 same object: True free slots: 16 clock_start == half A: True clock_deadline == half A: True
admitted of 20 workers: 16 | blocked: 4
```

**Failure scenario (NB1, plausible):**
- Half A's primary, INT4 on vLLM 0.19, is untested (README Finding 5; R-table). If it falls back to FP8 with 16 seqs, or the capacity check gives fewer than 20 streams, half A builds the gate with 16 slots.
- Half B's INT4 on 0.32 is ready with 20 streams, so the scheduler starts 20 workers.
- Only 16 hold a gate slot. Four workers block in `gate.acquire` inside `BranchSession.run`, after the scheduler has already counted them as running.
- Half B therefore really runs 16 streams. Yet its report says `streams 20` and `workers 20`, and the "INT4 20 streams on vLLM 0.32" measurement is silently a 16-stream one. That is exactly the "real concurrency == level" trap; `server_running_mean` would show it only if someone looks.

**Second effect, the stale clock:**
- DUCK_ENV sets `ARC3_PRIORITY_TAIL_FADE=1` and `ARC3_PRIORITY_REFRESH_QUEUE=1`.
- Half B's gate keeps half A's (already past) `_clock_deadline`, so `remaining = 0` and `tail_fraction = 0` from half B's first second.
- This only matters when there are waiters, that is in the scenario above.

**Other cases:**
- NB2 runs 16 in both halves; it is affected only if half A's capacity check gives fewer than 16 streams.
- S_B < S_A has no effect, because the workers bound the concurrency.
- The hidden rerun, with half A only, is not affected.

**Why the harness misses it:** the CPU harness runs EVAL6 (at most ~12 concurrent sessions, never more than 16), and its check is `max_running <= max_num_seqs`. That passes in fallbackA mode.

**Fix**, in `e21_begin_half` after the SERVER_* resets. It is proven in `gate_fix.py`: a new gate with 20 slots, and a half-A straggler releasing to the old gate does not touch the new one.

```python
_TA = E12A._import_duck()[0]
with _TA._PRIORITY_GATE_LOCK:
    _TA._PRIORITY_GATE = None      # rebuilt on half H's first BranchSession.run with this half's streams and clock
```

Optionally add a report check per half: `server_running_mean_client_full` against `streams`.

### L1 (LOW): half-A threads re-read the re-created `_E12_STOP`

**Where:** CELL 6 (wrapped):
- `_E12_STOP = threading.Event()` (donor line 427);
- `_e12_heartbeat`: `while not _E12_STOP.wait(every)`;
- NEW_WATCHDOG: `while not _E12_STOP.wait(_WD_POLL)`.

**Mechanism:** each loop looks up the module global. Half A sets its event at the end of CELL 6. Half B's CELL 6 re-binds the name to a new, unset event.

A thread that is busy when its event is set and then replaced keeps running in half B, with half B's globals (`E11_SERVER`, `SGL_URL`, `E12_RUN`, `_e15b_solve_end`). The busy states are `metrics_snapshot()` and `/health`, ≤10 s each, and above all `e21_relaunch()`.

`stale_thread.py` shows it:

```
half-A thread still alive in half B: True | ticks after replacement: 3
stops only on half B's event: True
```

A stale half-A watchdog would mean two watchdogs on half B. Its failure path in `e21_launch` (`srv.kill(); E11_SERVER = None; _e18_kill_gpu_pids()`) kills every GPU process and clears the global server.

**Reachability, computed:**
- A relaunch starts only with >= 20 min left (`_WD_MIN_LEFT`).
- Its wait is `max(60, min(1800, E21_HALF_END_PC - now - 1800))`.
- Worst-case total work is wait + ~850 s: kill 60, GPU-pid kill 20, GPU release 180, `/v1/models` 30, smoke 300, failure kill / release 260.
- So it ends at least ~290 s before `E21_HALF_END_PC`.
- When the window binds before `E21_HALF_END_PC` (half A setup < 27 min), a relaunch can still be in flight when `e21_end_half` kills the server. Its failure path then runs within seconds of that kill: while `e21_end_half` waits for the GPUs, and long before half B's server exists (`e18_install` runs first).
- Half B's event is created only after cell 7, end, the 0.32 install and the launch.

So this is not reachable on the normal timeline, but nothing enforces it.

**Fix** (cheap), in `e21_end_half` before the GPU kill:

```python
for th in threading.enumerate():
    if th.name == "e15b_watchdog" or "_e12_heartbeat" in th.name:
        th.join(timeout=60)
st["threads_left"] = [th.name for th in threading.enumerate()
                      if th.name == "e15b_watchdog" or "_e12_heartbeat" in th.name]
```

and in `e21_launch`, set `E11_SERVER = None` only `if E11_SERVER is srv`.

### L3 (LOW): `E12_RECORDS` / `E12_RUN` are not reset per half; `cell6_ran` is set before CELL 6

**Where:**
- RUNNER: `E21_STATE["halves"].setdefault(H, {})["cell6_ran"] = True     # CELL 6 re-creates E12_RECORDS / E12_RUN first`, then `e21_cell6()`;
- `e21_end_half`: `_recs, _run = globals().get("E12_RECORDS") or {}, globals().get("E12_RUN") or {}` and `if st.get("cell6_ran"):`.

**Scenario:** half B's CELL 6 raises before donor lines 20-21, for example in `E12_ORDER = sorted(...)`. Then:
- `e21_end_half` stores half A's records and E12_RUN as half B's;
- half B's CELL 7 builds `halfB/e12_run_report.json` from half A's `E12_RECORDS`;
- the E21 report shows half A's scores under half B.

The same holds for `ENGINE_INFO` if begin_half failed before its line 180 (not realistic).

**Probability:** tiny. Lines 5-19 depend only on `TASK_IDS` and `CHALLENGES`, which half A already used.

**Fix:** in `e21_begin_half`, add `global E12_RECORDS, E12_RUN` and set `E12_RECORDS = {t: [] for t in TASK_IDS}` and `E12_RUN = {"error": "CELL 6 not reached", "streams": None}`. Alternatively, set `cell6_ran` at the end of `e21_cell6` instead.

### Verified correct (2)

- **`e21_begin_half` resets** OUT (= `q27x/half<H>`, created before any launch, so every server log goes into the half's folder), `E21_HALF`, `E21_HALF_T0`, `E21_HALF_END_PC` and `SGL_URL` (own port 1234 / 8011). It also resets:
  - `E15B_SOLVE_END_PC = inf`;
  - `SERVER_DOWN.clear()`, `SERVER_WAITS` zeroed, `SERVER_LAST.clear()`, `SCHED_LAST.clear()` (so a half whose `solve_all_r6` raises never copies half A's SCHED_LAST);
  - `ARC3_REASONING_HISTORY_KEY=reasoning`;
  - `ENGINE_READY`, `CONCURRENCY`, `engine`, `llm`, a new `ENGINE_INFO`, `E12_STREAMS`, `E12_VLLM_CONCURRENCY`.
- **CELL 6 re-creates** `E12_ORDER`, `E12_RECORDS`, `E12_RUN`, `_E12_LOCK`, `_E12_WRITE_LOCK`, `_E12_STOP` and `E12_OUT` (= OUT/duck, per half). It re-sets `ARC3_MAX_ACTIVE_STREAMS` and `E12_DUCK_CONTEXT_WINDOW` before solving.
- **Half-A scheduler stragglers cannot write into half B:**
  - `publish()` returns once `state["closed"]` is set (`arc2_duck.py` 3366: "a straggler after shutdown never rewrites the records");
  - the final snapshots are published before `solve_all_r6` returns;
  - `probe_loop` stops on `state["stop"]`;
  - straggler requests go to half A's port (killed) and are clamped to half A's deadline;
  - a `SERVER_DOWN` wait is bounded by their own (past) deadline;
  - two concurrent submission writers would be safe anyway (unique temp file + `os.replace`).
- **Other Duck module state is harmless across halves:**

  | State | Why harmless |
  |---|---|
  | `_TOKENIZER` | same tokenizer both halves (README Finding 4); content-hash cache |
  | `_RULE_CACHE` | pure function of the task object |
  | `SANDBOX_LIMIT` | counters, not read by cells 6 / 7 |
  | `_GATE_STATS`, `_DIAG_STATE` | diagnostics |
  | retained functions and runtime state | per branch dir `OUT/half<H>/duck/<task>_b<n>`, so fresh per half |
  | tool_agent import-time constants (`_LOCAL_ANALYZER_*`) | from CELL 3's `duck_env`, identical for both halves |
  | the agent's server | its `base_url` is passed explicitly per half (`tool_agent.py` 3483-3489, `arc2_duck.py` 2956), so CELL 3's `LOCAL_ANALYZER_BASE_URL` (port 30000) is unused |
  | `model_id` | per half from that half's `/v1/models` |

- **Open servers:**
  - CELL 7 ends with `e11_kill_server(wait=False)`, the E21 override `E11_SERVER.kill(fast=True)`: SIGTERM, then always SIGKILL of the process group.
  - `e21_end_half` kills again (idempotent; a dead group raises and is caught), then `_e18_kill_gpu_pids()` and `_wait_gpu_release()`, and sets `E11_SERVER = None`.
  - The two halves use different ports, so there is no TIME_WAIT clash.
- **Environment:** `LIBRARY_PATH` (libcuda shim) and the CELL 3 Duck env carry into half B on purpose. `e18_install` copies `os.environ` into the 0.32 env.

---

## (3) Runner: error isolation and lambdas

### Verified correct

- Each of the four steps (begin, cell6, cell7, end) of each half has its own `try/except BaseException`. The error is appended to `E21_STATE["halves"][H]["errors"]` and printed, and the next step and half run.
  - A begin failure sets `ENGINE_READY = False` at module level (the runner is module-level code).
  - CELL 6 then takes its "not run" path: fresh `E12_RECORDS`, `E12_RUN["error"]`, a final identity / records submission write. CELL 7 and end still run.
  - Statements of CELL 6 / 7 outside the donor's own try (defs, `E12_OUT.mkdir`, the final write) are now caught by the runner too. That is more isolation than R3K2, where an exception there would stop the notebook.
- **Lambda late-binding:** `lambda: e21_begin_half(_H)` and its siblings read the global `_H` when called. Every lambda is called in the same loop iteration that created it, and nothing called in between binds `_H`: it is not in either global list, and begin and end take it as the parameter `H`. `("cell7", e21_cell7)` passes the function object. Correct.
- **Hidden rerun:**
  - `E21_HALVES = ["A"] if RERUN else ["A", "B"]`;
  - `E21_HALF_END_PC = _stop` because `len(E21_HALVES) == 1`;
  - `PROFILE = "RERUN_ALL"` has no `solve_window_sec`, so `E12_SOLVE_WINDOW_SEC = None`;
  - the two-window assert runs only `if E12_SOLVE_WINDOW_SEC and not RERUN`.
  - The submission is built from half A's records by CELL 6 (`if RERUN:` branch) and again by CELL 7. Nothing later (end, E21 report, bundle) writes `SUBMISSION_PATH`.
  - The E21 report cell's top-level statements outside its try (`def _e21_score`, the `E21_REPORT = {...}` literal) read only names defined in CELL 0 / 1.

### L11 (LOW): KeyboardInterrupt is swallowed

**Where:** the runner, `except BaseException:`, and the donor CELL 6 `except BaseException:`.

**Scenario:** an interactive editor session. Pressing interrupt during half A's solve ends that step only; cell7, end and then the whole of half B start. Each step needs another interrupt.

Commit and rerun are unaffected.

**Fix (optional):** `except KeyboardInterrupt: raise` before `except BaseException` in the runner, only when not in a Kaggle batch run.

---

## (4) `e21_end_half`, the E21 report, and other reads

### L2 (LOW): `_E12_LOCK` is used before the server kill and GPU release

**Where:** `e21_end_half`:

```python
_recs, _run = globals().get("E12_RECORDS") or {}, globals().get("E12_RUN") or {}
if st.get("cell6_ran"):
    with _E12_LOCK:                       # NameError if CELL 6 never reached donor line 23 in any half
        st["records"] = ...
...
E11_SERVER.kill() ... _e18_kill_gpu_pids() ... _wait_gpu_release()   # only after the line above
```

**Scenario:** half A's CELL 6 raises before `_E12_LOCK = threading.Lock()`.
1. `end` raises NameError.
2. The runner records it, and half B begins.
3. CELL 7 already fast-killed the server group, but no `_e18_kill_gpu_pids()` or `_wait_gpu_release()` ran.
4. Half B may launch while memory is still being freed; the primary, and maybe the fallback, could OOM.

Unlikely: lines 5-22 depend only on `TASK_IDS` / `CHALLENGES`.

**Fix:** do the kill and release in a `try/finally` before the records block, and use `_lk = globals().get("_E12_LOCK") or threading.Lock()`.

### L4 (LOW): half B does not check that half A's GPU release succeeded

**Where:** `e21_begin_half` never looks at `E21_STATE["halves"][prev]["gpu_release"]["released"]`. `_wait_gpu_release` returns `released: False` after 180 s, and nothing acts on it.

**Scenario:** an engine-core process of half A survives the group SIGKILL (another session or group) and `nvidia-smi` cannot kill it in time. Both of half B's configs then fail, and half B is lost.

**Fix:** for `H != E21_HALVES[0]`, if the previous half shows `released` False, run `_e18_kill_gpu_pids(); _wait_gpu_release()` again before the first launch, and record it in `ENGINE_INFO`.

### L5 (LOW): post-solve time in the hidden rerun

**Where:** `e21_end_half` always runs `_e18_kill_gpu_pids()` (`nvidia-smi` timeout 20 s) and `_wait_gpu_release()` (up to `Q27X_GPU_WAIT_MAX` = 180 s).

**Timeline in the rerun:**
1. The solve stops at `HARD_STOP - 180` = T0 + 11 h 52 min.
2. Then come the join grace (≤90 s), the b_summary and log parse, and CELL 7's report plus the zip of 240 tasks' transcripts.
3. Then `e21_end_half` adds up to ~3.5 min that half B needs but the rerun does not.
4. The E21 report zip follows.

The submission is already written in step 1, but a kernel that exceeds 12 h is a failed run. The margin still fits, but this spends it for nothing.

**Fix:** skip `_wait_gpu_release()` (and the pid kill) when `H == E21_HALVES[-1]`.

### L6 (LOW): CELL 6's `/v1/models` fallback model id is a path

**Where:** donor CELL 6, line 444: `E12_RUN["model_id"] = str(SGL_MODEL_PATH)` on a `/v1/models` exception.

**Scenario:** vLLM serves only `--served-model-name` (`Qwen/Qwen3.8-27B-INT4` or `-FP8`). For NB2 that path is not even the weights it loaded. If `/v1/models` fails once right after a verified launch, every Duck request returns 404 and the half is lost.

**Fix:** use `E21_SERVED[ENGINE_INFO["cfg"]["weights"]]` as the fallback. That is a one-line `rep()` in `cell6()`.

### L7 (LOW): an `e18_install` failure is not cached

**Where:** `e21_launch`:

```python
if E21_STATE.get("v032_install") is None:
    E21_STATE["v032_install"] = e18_install()        # on an exception nothing is stored
```

**Scenario:** the 0.32 install fails or times out (subprocess timeout 2,400 s). Half B's fallback config calls `e18_install()` again, which costs up to another 40 min and fails the same way. Half B is lost either way, so only time is wasted.

**Fix:** catch the exception, store `{"error": repr(e)}`, and on later calls raise immediately when that key is present.

### L8 (LOW, informational): CELL 7 concurrency field is empty on vLLM

**Where:** CELL 7 `_r6e_concurrency()` keeps only lines with `"TP0]"` and `"Decode batch"` (SGLang). On vLLM `report["rev6"]["metrics"]["concurrency"]` is always `{}`.

It is not an error. The vLLM equivalent is `b_summary.server_log` (NEW_LOGSUM) and `e11_decode_summary()`.

**Fix:** none required. Optionally a note in the README.

### Verified correct (4)

- **`e21_end_half`:**
  - reads `E12_RECORDS` / `E12_RUN` through `globals().get(...) or {}`;
  - copies the records under the half's lock (lists copied, record dicts shared, but half A's dicts are no longer mutated after `solve_all_r6` returns);
  - `ENGINE_INFO`, `E11_SERVER` and `NOTEBOOK_T0` always exist (engine cell);
  - `e11_decode_summary()` is in try/except and reads the killed server's log file, which still exists;
  - the kill is idempotent; `_e18_kill_gpu_pids` and `_wait_gpu_release` catch everything.
- **E21 report:** every per-half read is a `.get(...) or {}` chain (`records`, `engine`, `e12_run`, `b_summary`, `server_log`, `attempts`, `usage_all`).
  - A missing `half<H>/e12_run_report.json` gives `{"error": ...}`, and then `score_full` is all None.
  - `_e21_score` returns None when `SOLUTIONS is None` (rerun).
  - The table print uses only keys it built itself.
  - `E21_REPORT` holds only JSON-native values; CONFIG["e21"] paths are `str()`.
  - The report and the bundle each have their own `try/except BaseException`.
- **CELL 7 reads** `engine` (None), `E11_SERVER`, `e11_decode_summary` (the E12-V vLLM version), `e11_kill_server` (the E21 override with `fast=`), `_E12_REV6`, `SUBMISSION_PATH`, `WORKING` and `E21_HALF`. All exist. The bundle name is per half and `E12_OUT` is per half.
- **The "attempt-1-only" score** passes branch 0's records only to `build_submission`. Both attempt slots can come from branch 0's candidates, which matches the report comment ("branch 0 only").

---

## Other findings (outside the four items, found on the way)

### M1 (MEDIUM): the watchdog relaunch timeout is not the donor's

**Where:** `e21_engine_new.py`:
- `e21_launch` line 107: `left = E21_HALF_END_PC - time.perf_counter() - 30 * 60          # keep >= 30 min of solving`;
- line 109: `srv.wait_ready(max(60.0, min(E21_LAUNCH_TIMEOUT_SEC, left)))`;
- `e21_relaunch` line 150, which reuses `e21_launch`.

The donor E15B watchdog: `_new.wait_ready(min(SGL_RELAUNCH_TIMEOUT_SEC, max(60.0, _left - _WD_MIN_LEFT / 2)))`, that is `min(20 min, left − 10 min)`.

The watchdog relaunches only with at least 20 min left (`_WD_MIN_LEFT`). Wait time given to a relaunch, by time left to the half's end:

| Time left | Donor | E21 |
|---|---|---|
| 20-31 min | 10-20 min | 60 s |
| 40 min | 20 min | 10 min |
| ≥ 50 min | 20 min | ≥ 20 min |

**Scenario:** the vLLM server dies with 20-31 min left to the half's end. This also applies to the last half hour of the 12-h hidden rerun (`E21_HALF_END_PC = _stop`).
1. The relaunch gets 60 s to load TP4 weights and capture graphs.
2. It fails, retries once (another 60 s) and gives up.
3. The remaining 20-30 min of every branch are lost: requests fail and burn the consecutive-failure budget.

The donor's relaunch would have had 10-20 min.

**Fix:** give `e21_launch` a `timeout=None` parameter. `e21_relaunch` passes the donor rule:

```python
min(SGL_RELAUNCH_TIMEOUT_SEC, max(60.0, (_e15b_solve_end - time.perf_counter()) - 10 * 60))
```

To avoid making `_e15b_solve_end` a dependency, `E15B_SOLVE_END_PC` (the same value) can be used. `e21_begin_half` keeps the current 30-min rule.

### M2 (MEDIUM): the halves' solve windows are not equal

**Where:**
- `e21_begin_half`: `E21_HALF_END_PC = ... _stop - E21_HALF_SOLVE_SEC - E21_B_SETUP_RESERVE_SEC`;
- CELL 6 `_end = min(HARD_STOP - E12_STOP_BEFORE_HARD_SEC, E21_HALF_END_PC)`;
- the shifted end is capped by the same value.

**Numbers:** `time_budget_sec` is 43,200 − 180, `hard_margin_sec` 120 and `E12_STOP_BEFORE_HARD_SEC` 180.
- `HARD_STOP` = T0 + 42,900. `_stop` = T0 + 42,720.
- Half A must end by 42,720 − 19,800 − 1,500 = **T0 + 21,420 s (5.95 h)**.
- Half A's window is therefore 21,420 s minus its solve start (CELL 2 install, staging join up to 55 min, the launch, the preflight).
- Half A gets the full 5.5 h only if its setup takes under 27 min. With a typical 40-60 min setup it gets about 4.95-5.3 h.
- Half B gets min(5.5 h, `_stop` − its start). That is 5.5 h when its setup fits the 25-min reserve.

**Effect:** the README states "a solve window of 5.5 h from each half's own solve start". In fact A vs B is compared at unequal compute, by up to about 30 min in B's favour. The report has the data to see this: `solve_start_sec` and `deadline_in_sec` per half, in JSON only, not in the printed table.

**Fix:** in `e21_begin_half` for half B, set a global `E12_SOLVE_WINDOW_SEC` to half A's achieved window (`E21_STATE["halves"]["A"]["e12_run"]["deadline_in_sec"]` plus its `window_shift.shift_sec`) when half A solved. CELL 6 reads `E12_SOLVE_WINDOW_SEC` at module level. Also print both windows in the E21 table.

---

## Files of this validation

All in `D:/ARC AGI 2/_scratch/e21_val_B1/`:

- `extract.py`: extracts the cells as `NB1_INT4_S20_cNN.py` and `NB2_FP8_SSM16_S16_cNN.py`.
- `codeobj.py`, `astscope.py`, `symcheck.py`: the global lists, code objects and scopes.
- `clash.py`, `undefined.py`: name clashes and undefined names.
- `comp_global.py`: a comprehension variable declared global (3.12 / 3.14).
- `mlstr.py`, `wrap_forms.py`: multi-line strings, and the binding forms `wrap` does not handle.
- `gate_test.py`, `gate_fix.py`: H1, proof and fix.
- `stale_thread.py`: L1 mechanism.
- `rebuild_cmp.py`: notebook == builder output == donors.
