# VAL2: cells 8, 9 and 10 of E21 (rev 2d) — bug hunt (2026-10-28)

**Scope.** I reviewed the built notebook cells, not only `build_e21.py`:
- cell 8: `e21_cell6()`, the patched E15B CELL 6, and `e21_cell7()`, the patched CELL 7;
- cell 9: the runner;
- cell 10: the E21 report.

**Notebooks.** I extracted the cells from the rev 2d shas named in the task and checked them: NB1 `dddb5d0c`, NB2 `3ce943a2`, NB1_SMOKE `f063fea7` and NB2_SMOKE `18285c9c`. Cells 7, 8, 9 and 10 are byte-identical in all four notebooks.

**Method.** Read only. No project file was edited. No server or port was used: every experiment stubs Duck and the network. The scripts are in `D:/ARC AGI 2/_scratch/val2_cells/`, and the rev 2d cell texts are in `rev2d/`.

**The build changed during this review.** At 23:34, `build_e21.py` and all four notebooks were rebuilt by another process as rev 2e:
- NB1 `80d747c4`, NB2 `e909e9d8`, NB1_SMOKE `a5d1ad62`, NB2_SMOKE `571de316`;
- the rev 2e cells are in `cur/`, plus the snapshot `cur/build_e21_snapshot.py`.

Cells 1-9 of rev 2e are byte-identical to rev 2d. Only cell 10 changed: the prefix-cache key and `config_status`. So every finding below about cells 8 and 9 also applies to rev 2e. Findings R1 and R2 are rev 2d bugs that rev 2e already changes; I checked the rev 2e versions too.

## Findings

### F1 — MEDIUM: the rev 2c watchdog "binding" does nothing; a half-A watchdog can act in half B

**Location.** Cell 8, `e21_cell6()`, from build_e21.py `NEW_WATCHDOG`:
```python
_wd_stop, _wd_run, _wd_half = _E12_STOP, E12_RUN, E21_HALF     # E21 rev 2c: this half's objects
def _e15b_watchdog():
    ...
    while not _wd_stop.wait(_WD_POLL):
        if E21_HALF != _wd_half:          # rev 2c: a half-A watchdog never acts in half B
            return
```

**Why it fails.** The assignment line is module-level code in the donor cell. `wrap()` declares every module-level binding `global`; the global list of `e21_cell6` contains `_wd_stop, _wd_run, _wd_half`. So these three are not captured per half. They are module globals, and half B's `e21_cell6()` rebinds them:
- `_wd_stop` becomes half B's `_E12_STOP`;
- `_wd_run` becomes half B's `E12_RUN`;
- `_wd_half` becomes `"B"`.

After that, the half-A thread's check `E21_HALF != _wd_half` compares `"B"` with `"B"` and never returns. The thread also reads these as globals at call time: `_e15b_solve_end`, `_WD_POLL`, `_WD_FAIL`, `_WD_MIN_LEFT`, `E11_SERVER`, `SGL_URL` and `e21_relaunch`. Only `_wd = _wd_run["watchdog"]`, taken once at thread start, stays half A's.

This is the same behaviour as the unbound donor. The heartbeat is different: it is bound correctly, through the default argument `_stop=_E12_STOP`.

**Evidence.**
- **`exp8_scopes.py`** (bytecode of the built cell). `_e15b_watchdog` has no parameters and no free variables. It reads `_wd_half`, `_wd_run`, `_wd_stop`, `_e15b_solve_end`, `E21_HALF`, `E11_SERVER` and `e21_relaunch` with LOAD_GLOBAL. By contrast:
  - `_e12_heartbeat` has params `('every', '_stop')`;
  - `_e12_branch_done_r6` has params `('rec', '_recs', '_run', '_lock', '_half')`.
- **`exp1_cell6_halves.py`** runs the built `e21_cell6` twice under Python 3.12.10, with stubs:
  1. The half-A watchdog is held inside a relaunch past the end of half A.
  2. Half B's CELL 6 runs: `_wd_half = B`, `_wd_stop is B's True`.
  3. The relaunch is released. The half-A watchdog then made 11 health probes during half B (`HALF-A watchdog health probes during half B: 0 -> 11`).
- **`exp1b_halfB_dead.py`** (half B's server dead). The half-A thread called `e21_relaunch()` with `E21_HALF == "B"`. Relaunch calls: `[('A','A-thread'), ('B','B-thread'), ('B','B-thread'), ('B','A-thread')]`.

**Failure scenario.**
1. Half A's watchdog starts a relaunch at least 20 min before the end of half A.
2. The relaunch stays inside `e21_relaunch` (wait_ready, smoke, contract and cache probes, GPU release) until after half B's CELL 6 has started.
3. When it returns, the loop continues in half B. The thread is now a second watchdog for half B, with its own relaunch budget, because it counts on half A's `_wd["relaunches"]`.
4. It can kill and relaunch half B's server (`e21_relaunch` kills `E11_SERVER`), and race half B's own watchdog.
5. It sets and clears `E12A.SERVER_DOWN` under half B's branches.

The rev 2c note "a half-A watchdog never acts in half B" does not hold. The tracker row "GPT BLOCKER 1 … FIXED: both threads are bound to their half's event, run state and half id" is wrong for the watchdog.

**Probability (honest).** Low. The window is mostly closed by timing:
- the stop event is set at the end of half A's CELL 6;
- CELL 7 kills the server, so a running relaunch fails fast (`wait_ready` returns on process exit);
- `e21_end_half` joins the thread for 30 s;
- `e21_begin_half("B")` takes 10+ min before half B's CELL 6 rebinds the globals.

The thread therefore has to be blocked for more than about 10 minutes after half A ends.

**Fix.** Bind by default arguments (the only binding that `wrap()` keeps), and drop the module-level line:
```python
def _e15b_watchdog(_wd_stop=_E12_STOP, _wd_run=E12_RUN, _wd_half=E21_HALF, _solve_end=_e15b_solve_end):
```
- Use `_solve_end` instead of `_e15b_solve_end` inside.
- Re-check `if E21_HALF != _wd_half: return` right before `E12A.SERVER_DOWN.set()` / `e21_relaunch()`.
- Optionally, give `e21_relaunch` the half as an argument and have it refuse when `H != E21_HALF`.

Then correct the tracker row.

### F2 — MEDIUM: `stage1.attempt1_cut_by_deadline` is measured against the last attempt-1 end, not the deadline

**Location.** Cell 10, `_e21_stage1(recs, run)`. The same code is in rev 2d and rev 2e.
```python
dl = max(ends) if ends else None
"attempt1_cut_by_deadline": sum(1 for e in ends if dl is not None and dl - e < 90) if ends else None,
```

**Why it fails.**
- `dl` is the latest attempt-1 request end, so the attempt that ended last always counts. The value is always ≥ 1 whenever any attempt 1 has requests, even when stage 1 finished hours before the deadline.
- The parameter `run`, which carries the half's deadline, is never used.

**Evidence.** In `exp5_report.py`, scenario S2, stage 1 finished at hour 2 of a 5.45-h window in both halves. Both rev 2d and rev 2e report `"attempt1_cut_by_deadline": 1` for each half.

**Failure scenario.** This metric exists (rev 2b, VAL_timing_C2 LOW-4) to decide whether the "attempt-1-only" score is comparable between the halves. As built, it always says that at least one attempt 1 was cut in both halves. Either nobody reads it, or someone reads it wrongly.

**Fix.**
1. In CELL 6, next to `planned_solve_end_sec`, record the deadline on Duck's clock:
   `E12_RUN["solve_end_epoch"] = round(time.time() + (_e15b_solve_end - time.perf_counter()), 1)`. Request `t` and record `t0` are `time.time()` (arc2_duck.py 2040, 2960).
2. In `_e21_stage1`, take `dl = run.get("solve_end_epoch")` and count the attempt-1 records whose last request ended within 90 s of it.
3. Keep `attempt1_all_done_h` as it is.

### F3 — LOW: the KV-probe outage evidence is lost, not per half, and pollutes Duck's KV telemetry

**Location.** Cell 8, `_e12_capacity_probe` (build_e21.py `NEW_PROBE`) and cell 10.

**Evidence.**
1. **The counters are never reported.** `E21_STATE["probe_failures"]` and `["probe_fail_open"]` are top-level keys of `E21_STATE`. Nothing writes them out: not `e21_report.json` (cell 10 reads only `E21_STATE["halves"]`), not `e12_run_report.json` (CONFIG / ENGINE_INFO / E12_RUN), and not a bundle. I checked with a grep of every cell for `probe_fail`. They also add up over both halves.
2. **The outage clock carries over to half B.** `probe_fail_since` is never reset in `e21_begin_half`. If half A's last probe failed, for example during a final outage, half B's first failed probe goes to fail-open at once, without the 120-s hold. In `exp4_probe.py`: `half B first failed probe: ('ADMIT', None)`.
3. **The synthetic 1.0 looks like real KV pressure.** The 1.0 passes Duck's `_finite01`, so it becomes a real sample:
   - `state["gauges"]` records `kv = 1.0`;
   - `SERVER_LAST` feeds every request's `meta["kv"]`;
   - the scheduler's `kv_holds` / `kv_hold_sec` count the hold (arc2_duck.py 3390-3410, 3456-3466).

   Duck copies only four keys from the snapshot, so the extra `probe_failed` key is dropped. In the report, outage holds therefore cannot be told apart from real KV-pressure holds.
4. **A flapping /metrics keeps admissions on hold most of the time without ever failing open.** Each success resets the 120-s clock. In `exp4`, with one success every 100 s, 190 of 200 probes held. This is bounded, not a deadlock: a free worker admits within about 100 s.

**Fix.**
- Keep the counters per half, e.g. `E21_STATE["halves"][E21_HALF].setdefault("probe", {})`, with failures, fail-open count, outage seconds and synthetic-full samples.
- Pop `probe_fail_since` in `e21_begin_half`.
- Copy the probe dict into `E21_REPORT["halves"][H]`.
- Document that `kv_holds` includes outage holds, or subtract the synthetic-full samples.

The gate logic itself is correct (see Verified, item 5).

### F4 — LOW: `late_cross_half_callbacks` can never show a non-zero value in the E21 report

**Location.** Cell 8 callback, cell 7 `e21_end_half`, cell 10.

**Why it fails.**
- A cross-half callback (`_half != E21_HALF`) can only happen after `e21_begin_half("B")` has set `E21_HALF = "B"`.
- That is after `e21_end_half("A")` stored `st["e12_run"] = dict(_run)`, which is a copy.
- The callback increments half A's live `E12_RUN`, not the copy that the report reads (`_run.get("late_cross_half_callbacks")`).

**Evidence.** In `exp1`: `late_cross_half_callbacks: live A dict = 1 | stored snapshot (what the E21 report reads) = None`.

The same applies to `stale_callbacks_dropped` and `late_writes_skipped` arriving after the copy.

**Fix.** Count into the live per-half state, e.g. `E21_STATE["halves"][_half]["late_cross_half_callbacks"]` (bind `_st=E21_STATE["halves"][E21_HALF]` as one more default argument). Or keep a reference to the live `E12_RUN` in `e21_end_half` and read the counter at report time.

### F5 — LOW: `tool_calls_with_content` is summed over the whole notebook, not per half

**Location.** Cell 10: `E21_REPORT["tool_calls_with_content"] = globals().get("E21_CALL_WITH_CONTENT")`.

**Why it matters.** The counter measures a vLLM 0.32 parser behaviour (D1 F4: a tool call with visible text). The report gives one number for both halves, so a 0.32 effect cannot be attributed to its half. `length_restored` is already split per half, by subtraction in `e21_end_half`.

**Fix.** Snapshot `E21_CALL_WITH_CONTENT["n"]` in `e21_end_half` the same way as `length_restored`, and report it per half.

### F6 — LOW (hardening): the callback's binding parameters are positional

**Location.** `_e12_branch_done_r6(rec, _recs=E12_RECORDS, _run=E12_RUN, _lock=_E12_LOCK, _half=E21_HALF)`.

**Today.** This is not a bug. Duck 9e4b calls `on_branch_done(copy.deepcopy(rec))` with exactly one positional argument (arc2_duck.py 3377; also `solve_all` at 2893).

**Risk.** A future caller passing a second positional argument (for example a `final` flag) would silently replace `_recs` and write into the wrong object.

**Fix.** Make them keyword-only: `def _e12_branch_done_r6(rec, *, _recs=..., _run=..., _lock=..., _half=...)`.

### F7 — LOW (hardening, unverified): the deadline-overrun check uses one clock only

**Location.** Cell 8: `solve_overrun_sec = perf_counter() - _e15b_solve_end`.

**Why it matters.** `perf_counter` is CLOCK_MONOTONIC on Linux. Whether a Kaggle host pause (the NB2 rerun case in the tracker) advances it depends on the hypervisor; I could not verify this. The check would miss a pause that stops the monotonic clock.

**Fix (cheap).** Also record a wall-clock overrun:
- store `time.time()` at the planned end;
- report `wall_overrun_sec`;
- flag `deadline_violation` on either clock.

## rev 2d bugs already changed in the working build (rev 2e, cell 10)

These are not in the tracker. Rev 2e attributes them to "Kaggle CPU kit H3". I report them because the task's notebooks (rev 2d) contain them, and I checked the rev 2e fixes.

### R1 — MEDIUM in rev 2d: `config_status` under option ka

**Location.** Cell 10.

**Why it fails.** NB1/NB2 run with `E21_B_FOLLOWS_A = True`. When half A falls back, `e21_begin_half("B")` merges the fallback into half B's spec, so `e21_cfgs()` tags the identical config `"primary"`. Half A is then `"fallback"` and half B `"primary"`, which gives `CONFOUNDED_FALLBACK` and `comparison_valid = False`, although both halves ran the same weights, seqs and SSM.

**Evidence.** In `exp5`, scenario S1: rev 2d gives `CONFOUNDED_FALLBACK / False`; rev 2e gives `FALLBACK_BOTH / True`.

**Also.** In rev 2d, `_cfgs` was computed and never used (dead code). The `json.loads(v[1])` guard for `"null"` itself was correct.

**Rev 2e.** It compares the configs that ran (`weights` / `seqs` / `ssm`) and adds `SWITCHED_MIDHALF`. I checked: correct for S1-S4. `relaunch_cfg_switched` is set before the switched relaunch is attempted, so a failed switched relaunch also marks the half as SWITCHED_MIDHALF. That is conservative and acceptable.

### R2 — MEDIUM in rev 2d: `prefix_cache_hit_pct_last` always None

**Location.** Cell 10. The rev 2 log summary (`e21_logsum.py`) names the key `prefix_cache_hit_pct_last_rolling`, but rev 2d read `prefix_cache_hit_pct_last`.

**Evidence.** In `exp5`, rev 2d prints `prefix_last={'A': None, 'B': None}`; rev 2e prints `66.6`.

**Rev 2e.** It reads the `_rolling` key and adds `prefix_hit_pct_whole_run` from `final_counters`. Correct.

## Verified correct (with evidence)

1. **Callback binding.** The default arguments are evaluated when the def runs inside `e21_cell6`, after `E12_RECORDS`, `E12_RUN` and `_E12_LOCK` were re-created (cell 8, lines 26-29 before line 65).
   - `exp1`: `cb defaults: recs is A's True | half A`.
   - A half-A straggler callback released during half B wrote only into half A's live records (`A records: ['A-straggler'] | B records: ['B']`). It hit the half check before any `e12_write_submission`, so it never wrote half B's submission.
   - Duck calls it with one positional argument (arc2_duck.py 3377), so nothing collides.
2. **Rerun submission.** `e12_write_submission` resolves `E12_RECORDS` and `e12_write_submission` as globals at call time; with one half that is half A. `exp7_rerun_sub.py`: `build_submission` saw half A's records, the submission has the generated attempt for `t1`, and identity for `t2`, which has no record.
3. **Heartbeat.** It is bound to its half's stop event (`_stop=_E12_STOP`, defined after `_E12_STOP = threading.Event()`). It is named `e12_heartbeat` and joined in `e21_end_half`. Its print reads `E12_RUN` as a global, which is cosmetic only.
4. **Watchdog `return` placement.** It is at the top of the loop, after the stop wait, and the donor's other returns are unchanged. A diff against the E15B R3K2 donor watchdog (`donor_e15b_cell6.py`) shows only the intended changes: `e21_relaunch` instead of the SGLang E11Server relaunch, `_wd` for the counters, and the half check. The binding issue is F1.
5. **KV probe against `solve_all_r6`** (`exp4_probe.py`, using the built probe and Duck's `_finite01`):
   - healthy: `kv 0.40` admits;
   - a failed `/metrics` read gives `kv_usage 1.0`, which is a valid `_finite01` sample. It holds when work is running; with nothing running, Duck's idle override admits, so there is no deadlock;
   - after 120 s, `kv_usage None` gives Duck's own fail-open (`kv_missing`, warned once);
   - recovery clears `probe_fail_since`;
   - the extra key `probe_failed` is ignored: Duck copies only `kv_usage / mamba_usage / running / queue` (arc2_duck.py 3406), and the cache diagnostic, which would read it, is disabled;
   - metric names and units are right for both versions: `vllm:kv_cache_usage_perc` is a 0..1 gauge (E18 AUDIT_A #9 and l.246, AUDIT_B l.309, and recorded `vllm_metrics.json` of Exp 1 run 7). There is one row per engine, so `max` over rows is correct.
6. **`_e15b_solve_end` / `solve_overrun_sec` / `planned_solve_end_sec`.**
   - `_e15b_solve_end` is assigned in the same `try` before the watchdog starts and before `solve_all_r6`, so it is defined on every path that reaches the SCHED_NEW block (the solve path and the parity path). The not-run path and an exception before it never reach that block.
   - In `exp1`, both halves record `planned_solve_end_sec` and `solve_overrun_sec`.
   - The 120-s threshold against `join_grace_sec = 90` will not raise false alarms. In R3K2, the solver returned about 1 s after its deadline: started 830, window starting at 833, deadline_in 20,467, finished_sec 21,301, summary at log time 21,308.
7. **Log summary against its consumers.**
   - CELL 7 reads only `e11_decode_summary()` keys: `running_hist` (string keys, `int(k)` works) and `accept_len_median` (the alias exists).
   - The E21 report reads `gen_tok_s_p50`, `gen_tok_s_mean`, `kv_usage_pct_max` and `preempt_lines`, which all exist, plus the prefix key (R2).
   - The b_summary keys the report reads all exist.
   - The stats regex matched every stats line of the real logs: 46 of 46 in the E12-V run-2 vLLM 0.19 log and 122 of 122 in an E19 log (`exp6_statre.py`).
8. **Robustness of the report cell** (`exp5`, rev 2d and rev 2e). It does not crash:
   - when half A's begin failed before `E21_STATE["halves"]["A"]` existed (S3: NOT_RUN, `comparison_valid False`);
   - in the hidden rerun without solutions (S4);
   - on `"null"` configs.

   `windows_equal`, the 1 % window rule and the deadline-violation gate are applied as documented.
9. **Runner.**
   - The lambdas are called within their own iteration, so `_H` is current.
   - No name in the `e21_cell6` / `e21_cell7` global lists collides with `_H`, `_step` or `_fn`.
   - Each step is isolated by `except BaseException`. `ENGINE_READY = False` after a failed begin is a module global (the runner is not wrapped).
   - `_e21_cell6_of` marks `cell6_ran` before CELL 6. `e21_end_half` then stores the half's re-created records, or `{}` from begin.
10. **`wrap()`.**
    - Re-wrapping the bodies of the built `e21_cell6` / `e21_cell7` reproduces both functions byte-identically under Python 3.12.10 and 3.14.6 (`exp3_wrap.py`).
    - The symtable check only asserts; it returns `out` unchanged.
    - Comprehension variables whose names are declared `global` (`t`, `r`, `v`, …) do not overwrite the globals in 3.12 or 3.14 (`exp2_comp_global.py`).
    - An annotated assignment fails closed: symtable raises a SyntaxError at build time.
    - The wrapped code behaves like module-level code (same `globals()`, the same names bound). That is exactly why F1's "binding" does nothing.
11. **Cells 7-10 are identical** in NB1, NB2 and both SMOKE notebooks.

## Experiment files

All in `D:/ARC AGI 2/_scratch/val2_cells/`:
- `extract.py`: extracts the cells and checks the shas;
- `rev2d/`, `cur/`: the rev 2d and rev 2e cell texts;
- `exp1_cell6_halves.py`, `exp1b_halfB_dead.py`: the built `e21_cell6` run as two halves (watchdog, callback, late counter);
- `exp2_comp_global.py`: comprehension variables against `global`;
- `exp3_wrap.py`: wrap determinism and blind-spot probes;
- `exp4_probe.py`: the KV probe against Duck's gate;
- `exp5_report.py`: cell 10 of rev 2d and rev 2e on scenarios S1-S4;
- `exp6_statre.py`: the stats regex on real vLLM logs;
- `exp7_rerun_sub.py`: the rerun submission;
- `exp8_scopes.py`: name resolution in the bytecode of the nested functions;
- `donor_e15b_cell6.py`: the R3K2 donor CELL 6, for the watchdog diff.
