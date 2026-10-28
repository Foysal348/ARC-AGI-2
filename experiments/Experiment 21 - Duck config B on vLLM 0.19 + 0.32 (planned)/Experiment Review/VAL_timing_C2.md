# VAL timing C2: Experiment 21 run modes on the clock and on outputs (independent second reviewer)

Date: 2026-10-28. This was a read-only review: no project file was edited and nothing was pushed or run on Kaggle. The only file written is this report.
Reviewed: the two built notebooks `Q27-XHIGH-E21_DUCKB_VLLM_NB1_INT4_S20.ipynb` (sha 85677597) and `..._NB2_FP8_SSM16_S16.ipynb` (sha a1a9e3c7). I read each cell after extracting it. Apart from the variant block in CELL 1 (spec, `E21_STAGE_PATHS`), the two notebooks are identical. I also read `Build/build_e21.py`, `Build/e21_engine_new.py`, the README, the harness (`src/harness/harness_e21_nb.py`), the Duck package that R3K2 ran (`_scratch/donor_map/duck_9e4b`: `arc2_duck.py`, `vendor/inference/agent/tool_agent.py`) and the local harness outputs in `_scratch/e21/harness_runs/`.
Real timings used:
- the R3K2 run (report, log, `e14_state.json`);
- the E18 vLLM 0.32 smoke;
- the E12-V vLLM 0.19 Duck smoke (report, log, server log).

To keep this review independent, I did not read `VAL_timing_C1.md` or `PROVENANCE_AND_BUGS_E21_2026-10-28.md`.

Line numbers are given as `NB cell index : line` of the extracted cells (cell 7 = CELL 5 engine, cell 8 = CELL 6/7, cell 9 = CELL 8 runner, cell 10 = CELL 9 report), or as builder or package file lines.

---

## 1. Real timings used

| Source | Fact | Value |
|---|---|---|
| R3K2 `e12_run_report.json` | `started_sec` / `deadline_in_sec` / `finished_sec` | 830 / **20,467** / 21,301 |
| R3K2 | `window_shift` | 2.9 s (preflight), window start 833 |
| R3K2 | budget 6 h → HARD_STOP − 180 = 21,300 | the solve ended on it; `finished` 1 s later |
| R3K2 log | after the solve: report 21,309, bundle 21,311, notebook written 21,322 | **tail ≈ 20 s** (SGLang, no kill wait) |
| R3K2 `e14_state.json` | staging of 19.5 GB INT4 + drafter to /dev/shm | copy done 114 s (678 MB/s); join at 301 s; READY 824 s |
| R3K2 | `trimmed_requests` | 0 (no Duck trims, so no gate handover) |
| R3K2 | stage1_left / attempts / attempt2_started | 0 / 198 / 78 |
| E12-V run 2 (vLLM 0.19 FP8 Duck server, port 1234, 135,168 ctx) | install done / tokenizer / engine ready + smoke / solve start | 94 s / 115 s / 656 s / `started_sec` 649 |
| E12-V server log | KV: 337,600 tokens; "Maximum concurrency for 135,168 tokens: 8.46x" | effective 1,143,521, so 22 streams at 50k |
| E12-V server log | longest log silence during startup | **65 s** (stall rule 600 s) |
| E18 smoke (vLLM 0.32) | CELL 2 (0.19 install) done / 0.32 venv install / launch to ready / solve start | 161 s / **175 s** / **550 s** / 14.8 min |
| E18 | longest startup silence; KV | 60 s; 1,110,415 (with `--language-model-only`) |
| S1-60K on 0.19 (E18 note) | ready | 668-733 s |

---

## 2. The clock constants in the notebooks

All values are seconds from `NOTEBOOK_T0` (the kernel process start).

| Name | Value | Where |
|---|---|---|
| `TOTAL_BUDGET_SEC` (E21X2EVAL120 and RERUN_ALL) | 43,020 (12 h − 3 min) | cell 2:118-120, 128 |
| `HARD_STOP` | 42,900 | cell 2:187 |
| `_stop` = HARD_STOP − E12_STOP_BEFORE_HARD_SEC | **42,720** | `e21_engine_new.py:172` |
| `E21_HALF_SOLVE_SEC` | 19,800 (5.5 h) | cell 2:408-410 |
| `E21_B_SETUP_RESERVE_SEC` | 1,500 | cell 2:411 |
| Half A end cap `E21_HALF_END_PC` (commit) | 42,720 − 19,800 − 1,500 = **21,420** | `e21_engine_new.py:174` |
| Half B end cap (commit), half A end cap (RERUN) | **42,720** | same line |
| Solve end | min(cap, solve start + 19,800), then + preflight shift, capped again (commit); cap only (RERUN) | cell 8:419-421, 460-472 |
| CELL 1 guard | 2 × 19,800 + 1,500 = 41,100 < 42,900 | cell 2:422-423 |
| Time left to 12 h after the last solve end | 43,200 − 42,720 = **480 s** (about 473 s against Kaggle log time 0, since T0 ≈ log 7 s) | |

---

## 3. Expected timelines

### 3.1 Commit (two halves, 120 eval tasks each), nominal path

| Phase | Expected (s from T0) | Basis |
|---|---|---|
| CELL 0-3: payloads, env, read probe, staging start, vLLM 0.19 pip install (94-161 s), tokenizer, Duck package, sandbox self-test | 0 → ~200 | E12-V 115 s, E18 161 s |
| Staging (INT4 19.5 GB for NB1, FP8 for NB2) to /dev/shm, in parallel | done ~100-150 | R3K2 114 s |
| Engine cell: staging join, libcuda probe; CELL 8 → `e21_begin_half("A")` → vLLM 0.19 launch → ready | ~200 → ~740 | E12-V 541 s launch; S1 ready 668-733 |
| /v1/models + capacity + smoke + CELL 6 preflight (window starts after it) | ~740 → **s_A ≈ 750** | R3K2 shift 2.9 s |
| **Half A solve** | 750 → **20,553** (= s_A + 19,800 + shift) | cap 21,420 leaves **867 s of slack** |
| Join grace (≤ 90 s; R3K2 1 s), CELL 7 report + bundle (~10 s), `e21_end_half` (kill + GPU release ~15-60 s) | → ~20,600 | R3K2 tail; E18Server.kill |
| `e21_begin_half("B")`: `e18_install` (175 s) + vLLM 0.32 launch (550 s) + smoke + preflight | → **s_B ≈ 21,335** | E18 |
| **Half B solve** | 21,335 → **41,138** | cap 42,720 leaves **1,582 s of slack** |
| Tail: join, CELL 7, end, CELL 9, nbconvert | → **≈ 41,200 s = 11.44 h** | about 33 min under 12 h |

**Result.** On the nominal path both halves get the same window: 19,800 s plus their own few-second preflight shift. The same tasks run in both halves, in the same order (`E12_ORDER`), with fresh records and fresh sessions. Each half writes to its own folder.

### 3.2 How much extra delay each window can absorb

| Extra delay | Effect |
|---|---|
| At A's start, up to **867 s (14.5 min)** | A keeps the full 5.5 h |
| At A's start, beyond that | A's window = 21,420 − s_A. B is not shortened to match. |
| A tail + B setup, up to **1,582 s (26 min)** beyond nominal | B keeps the full 5.5 h |
| A tail + B setup, beyond that | B's window = 42,720 − s_B |
| A capped at 21,420 | B has 1,500 s minus A's tail, against a nominal need of ~12-13 min |

Concrete failure paths:

| Path | s_A or s_B | Window |
|---|---|---|
| NB1 INT4 on 0.19 crashes fast (≤ 2 min), then FP8 fallback | s_A ≈ 950-1,000 | A full |
| Primary hits the 600-s stall rule, then fallback | s_A ≈ 1,900 | **A = 19,520 (−1.4 %)** |
| Primary hits the 1,800-s ready timeout, then fallback | s_A ≈ 2,710 | **A = 18,710 (−5.5 %)** |
| Staging stalls to its join limit (T0 + 55 min), then launch | s_A ≈ 3,850 | **A = 17,570 (−11 %)** |
| B primary times out at 1,800 s, then fallback | s_B ≈ 23,900 | **B = 18,820 (−5 %)** |

### 3.3 Hidden rerun (half A only, 240 tasks)

| Phase | Expected | Note |
|---|---|---|
| Setup + vLLM 0.19 ready | s_A ≈ 750 | same as the commit |
| Solve (no window, no shift; absolute end) | 750 → **42,720** | 41,970 s = 11.66 h for 240 tasks = **174.9 s per task** |
| Tail, nominal: join (~1 s), CELL 6 summary + final submission write, CELL 7 (~15-30 s), `e21_end_half` (~15-40 s), CELL 9 (~5 s), nbconvert (~12 s) | → ≈ 42,790-42,810 | **~6.5-7 min under 12 h** |
| Tail, worst case: join 90 + CELL 7 30 + kill 60 + nvidia-smi 20 + GPU release ≤ 200 + CELL 9 10 + nbconvert 15 | ≈ 425 s → 43,145 | **~55 s under 12 h**, before any Kaggle clock offset (see M2) |

`submission.json` holds the final real attempts from the moment CELL 6 ends (≈ 42,720-42,810). CELL 7 rewrites it identically. Nothing after that changes it.

---

## 4. Findings

### HIGH-1: Duck's priority gate is a process-wide singleton. Half B silently inherits half A's slot count (and A's clock).

- **Location.**
  - `vendor/inference/agent/tool_agent.py:2139-2154` (`_PRIORITY_GATE`, built once from `ARC3_MAX_ACTIVE_STREAMS` at first use, then always returned).
  - `arc2_duck.py:2992-2995` and `3044` (every `BranchSession.run` acquires and releases that gate).
  - Notebook cell 8:417-418: CELL 6 sets `os.environ["ARC3_MAX_ACTIVE_STREAMS"]` for each half. That has no effect once the gate exists.
  - `e21_begin_half` (`e21_engine_new.py:164-212`) resets `SERVER_DOWN`, `SERVER_WAITS`, `SERVER_LAST` and `SCHED_LAST`, but not the gate.
- **Evidence.**
  - I ran the vendor's own `_PriorityGate` / `_priority_gate` source in a probe, with the env at 16 (half A) and then 20 (half B). Result: `half B gets the same gate object: True | slots: 16` and `20 half-B workers -> peak concurrent slot holders: 16`.
  - `configure_clock` keeps the minimum start and the minimum deadline, so half B's gate clock is half A's (start and an already-past deadline). With `ARC3_PRIORITY_TAIL_FADE=1` this only affects snapshot priorities after a trim. R3K2 had `trimmed_requests` 0, so this part is inert today.
- **Scenario.** NB1 half A's primary (INT4 on vLLM 0.19) is untested. The README says "NB1's fallback covers that". If A falls back to FP8 with 16 streams, the gate is built with 16 slots. Half B then becomes ready on INT4 0.32 with 20 streams and 20 workers, but only 16 attempts can hold a gate slot at once.
  - The B measurement ("INT4, 20 streams on vLLM 0.32") is then actually a 16-stream run.
  - Half B's report would still say `streams` 20 and `workers` 20. Only `server_running_mean` (about 16) would show it.
  - The same cap applies whenever A's capacity check gives fewer streams than B's.
  - Stragglers of half A (threads stuck in a native call after the join grace) also keep their slots into B.
- **Why the harness missed it.** In the `run_NB1_fallbackA_210428` run, B's fake server shows `max_running` 20. But that peak comes from the 20 concurrent isolation-preflight requests (`preflight` 24 = 20 + 4 parity), which bypass the gate. B had only 18 Duck conversations (`new_convs` 18), so the 16-slot cap could never bind with EVAL6.
- **Fix.**
  - In `e21_begin_half`, under `_E12_TA._PRIORITY_GATE_LOCK`, set `_E12_TA._PRIORITY_GATE = None` and zero `_E12_TA._GATE_STATS`. The next `BranchSession.run` in that half then builds a gate with that half's `ARC3_MAX_ACTIVE_STREAMS`. Old stragglers release into the old object, which is harmless.
  - Add a check at each half's solve start: `_E12_TA._priority_gate()._free == E12_RUN["streams"]`, recorded in the report.
  - Add a harness case in which half B has more Duck attempts than A's stream count, and assert that B's Duck-only concurrency (tool requests in flight) exceeds A's streams.

### MEDIUM-1: The halves get equal windows only on the nominal path. When one half is cut, the other is not matched.

- **Location.** `e21_engine_new.py:174` (A cap = `_stop − window − reserve`); cell 8:419-421 and 465 (window = min(cap, start + 19,800), plus shift, capped); `e21_launch` ready timeout (`e21_engine_new.py:107-109`); staging join (`e21_engine_new.py:246`, up to T0 + 55 min).
- **Evidence.** Section 3.2:
  - A loses window as soon as its solve start is later than 1,617 s. The nominal start is ~750 s, so the slack is only **14.5 min**.
  - The README's NB1 failure mode (primary stalls or times out, then fallback) costs 1.4-5.5 % of A's window.
  - A slow staging join costs up to 11 %.
  - B is cut only after ≥ 26 min of extra A-tail + B-setup time.
  - The E21 report shows each half's `deadline_in_sec`. Nothing equalizes the windows or warns about them, and the README states "a solve window of 5.5 h from each half's own solve start" without that condition.
- **Scenario.** NB1: INT4 on 0.19 stalls (no log line for 600 s) or never becomes ready within 1,800 s. The fallback is ready at ~1,900-2,700 s. A solves 18,710-19,520 s and B solves 19,800 s. The A/B comparison is then biased toward B by the window, on top of the weights/streams difference (MEDIUM-3).
- **Fix.**
  - When A's achieved `deadline_in_sec` is shorter than `E21_HALF_SOLVE_SEC`, give B the same window: `E12_SOLVE_WINDOW_SEC = A's deadline_in_sec` for half B.
  - Add `windows_equal` and both windows to `e21_report.json` and the printed table.
  - Optionally run `e18_install()` in a background thread during half A (it is CPU/disk only). That moves ~175 s out of B's setup.
  - If B is cut (it cannot be compensated afterwards), flag it in the report.

### MEDIUM-2: The worst-case RERUN tail leaves about 1 min under the 12-h limit. The new end-of-half GPU waits are pure overhead in the last half.

- **Location.** `e21_end_half` (`e21_engine_new.py:215-238`): `E11_SERVER.kill()` (SIGTERM, wait ≤ 30 s, SIGKILL, wait ≤ 30 s; cell 7:213-238), `_e18_kill_gpu_pids()` (nvidia-smi ≤ 20 s), `_wait_gpu_release()` (≤ 180 s, plus a final nvidia-smi call ≤ 20 s). CELL 9 follows.
- **Evidence.**
  - After the solve end at 42,720 there are 480 s to 12 h, about 473 s against Kaggle log time 0. R3K2 and S1 had a ~20 s tail with no kill wait: R3K2's CELL 7 used `e11_kill_server(wait=False)`, a non-blocking SGLang kill.
  - E21's worst tail is about 425 s (section 3.3). Nominal is ~60-90 s. In the last half, `_wait_gpu_release` and `_e18_kill_gpu_pids` serve no purpose: no further launch follows.
  - A hidden run that exceeds 12 h scores nothing, even though `submission.json` was already written.
  - It is unknown how much earlier than the kernel process the 12-h clock starts. Kaggle log time 0 ≈ T0 − 7 s; container start may be earlier still.
- **Scenario.** RERUN with stuck stragglers (join 90 s), a slow vLLM shutdown (the full 30 + 30 s), and GPU memory reported as still held for a while (the 180-s poll). The notebook ends within ~1 min of the limit.
- **Fix.** For the last half (`H == E21_HALVES[-1]`; in RERUN that is half A), call `E11_SERVER.kill(fast=True)` and skip `_e18_kill_gpu_pids` and `_wait_gpu_release`. Alternatively, skip the kill in RERUN entirely, since the kernel exits anyway. Keep the full release wait only before half B's launch.

### MEDIUM-3: NB1 with half A on its fallback confounds the A/B comparison.

- **Location.** cell 2:401-404 (NB1 spec: A primary INT4/20, fallback FP8/16; B primary INT4/20, fallback FP8/16); `e21_begin_half` loop (`e21_engine_new.py:189-205`).
- **Evidence.** B's primary is tried independently of what A ended up using.
- **Scenario.** INT4 on 0.19 fails and A runs FP8 with 16 streams; B runs INT4 with 20 streams. The halves then differ in vLLM version, weights and stream count (B is additionally capped to 16 until HIGH-1 is fixed). The result can no longer be read as "0.19 vs 0.32". README Findings 5 expects this branch.
- **Fix (user decision).** Either (a) when A used its fallback, start B on the same config (B spec = A's `cfg` with `vllm` 0.32.0), which keeps a clean engine-version comparison; or (b) keep B on INT4/20 but label the comparison in `e21_report.json` as `"confounded": ["weights", "streams"]`.

### MEDIUM-4: The launch smoke accepts any non-empty text. A numerically broken engine passes, and the RERUN has no other guard.

- **Location.** `e21_launch` smoke (`e21_engine_new.py:117-129`): `ok = bool(gen) and prompt_tokens > 0 and completion_tokens > 0`.
- **Evidence.**
  - This project has already seen an engine that loads and then emits garbage (memory: E13 Flash-Next on SGLang 0.5.21 produced `"!!!!"`).
  - INT4 W4A16 with a BF16 MTP head on vLLM 0.19 is untested (README Findings 5).
  - The isolation preflight passes only if each reply contains its marker (`isolation_verdict`). On FAIL it reduces workers to 6 but does not change the engine, so a garbage engine still runs.
- **Scenario.**
  - Commit: half A runs 5.5 h on a broken engine and scores ~0. This is visible, and mitigated because the user would not submit.
  - RERUN: if the hidden-run node behaves differently, nothing makes the notebook use the proven FP8 fallback.
- **Fix.** Make the smoke strict. At temperature 0 with thinking off, require `"4"` in `gen`, or a second fixed-answer probe. On failure, raise so that the existing fallback path runs. For RERUN, also treat an isolation `FAIL` with 0 own markers as an engine failure and move to the fallback.

### LOW-1: `e11_kill_server(wait=False)` at the end of CELL 7 is a silent no-op.

- **Location.** cell 8:1081-1084 calls `e11_kill_server(wait=False)` (E12-V donor, cell 7:145-147), which calls `E11_SERVER.kill(wait=wait)`. `E18Server.kill` takes `fast=`, not `wait=` (cell 7:213), so it raises a `TypeError` that `except Exception: pass` swallows.
- **Effect.** The server is still killed by `e21_end_half`, so the outcome is unchanged. The dead `_e12_parity` path has the same call plus `E11Server` (undefined) and is never reached.
- **Fix.** Remove the call, or make `e11_kill_server` call `E11_SERVER.kill(fast=not wait)`.

### LOW-2: A watchdog relaunch late in a half gets a ready timeout too short to succeed.

- **Location.** `e21_launch`: `left = E21_HALF_END_PC − now − 30 min`, then `wait_ready(max(60, min(1800, left)))` (`e21_engine_new.py:107-109`), also used by `e21_relaunch`. The watchdog allows a relaunch while ≥ 20 min of solving remain (cell 8:523).
- **Evidence.** For half A (cap − solve end = 867 s), a death with fewer than ~29 min of solving left gives a ready timeout under ~540 s. That is shorter than vLLM 0.19's ~540 s start (E12-V) and 0.32's 550 s (E18). Branches wait on `SERVER_DOWN` during that futile attempt, and it uses up one of the two relaunches.
- **Note.** The relaunch duration is bounded: it always ends before the half's solve end. So no path extends the notebook.
- **Fix.** For a relaunch, use `min(E21_LAUNCH_TIMEOUT_SEC, solve_end − now − 5 min)`, and skip the relaunch (`action: none`) when that is under 10 min.

### LOW-3: Commit compute per task is below both R3K2 and the hidden run.

- **Evidence.**

| Run | Seconds per task | Difference |
|---|---|---|
| Commit half: 19,800 / 120 | 165.0 | |
| R3K2: 20,467 / 120 | 170.6 | commit is −3.3 % |
| Hidden RERUN: ≈ 41,970 / 240 | 174.9 | commit is −5.7 % |

- **Notes.**
  - The README mentions "R3K had about 5.7 h", but not the hidden-run share.
  - With this reserve and a ~750-s start, the largest equal window that fits is 2W ≤ 42,720 − 1,500 − 750, so W ≈ 20,235 s (5.62 h).
  - Comparisons against R3K2 (39.03 / 33.47) and LB projections from these halves are therefore slightly pessimistic.
- **Fix.** None required. State it in the report or README. Optionally raise the window to ~5.6 h.

### LOW-4: The "attempt-1-only" score is window-independent only when stage 1 finishes in both halves, and its definition should be stated.

- **Location.** `build_e21.py` REPORT `_e21_score` (lines 430-441, 468).
- **Evidence.**
  - It builds the normal two-slot submission from branch-0 records only (`select_attempts_r5`). Attempt_1 is branch 0's best. Attempt_2 is branch 0's other accepted or partial candidates, then the CPU rule outputs, then the test input. So it is "both slots from attempt 1", not slot-1-only.
  - On R3K2 data (section 5) it gives **46 / 172, task-norm 27.92 %**, with pass2 0. Strict slot-1-only is also 46, so on that run the two definitions agree.
  - It is unaffected by attempt 2 only if every task's attempt 1 ended by budget or acceptance rather than by the deadline. At ~400 tok/s, stage 1 of 120 × 50k takes ~4.2 h of a 5.5-h window. A slower engine could cut late attempt-1s.
  - The report does not show when stage 1 finished or how many branch-0 attempts the deadline cut.
- **Fix.** Add per half: the stage-1 completion time, `b0_cut_by_deadline`, and a strict `pass1_b0`. Rename or define the field in the README ("branch-0 records, both slots").

### LOW-5: `e21_begin_half` does not reset the half's record state.

- **Location.** `e21_begin_half`. `E12_RECORDS`, `E12_RUN` and `_E12_LOCK` are recreated only by CELL 6 itself (cell 8:26-29).
- **Scenario.** If B's `e21_cell6` raised before its line 26, B's CELL 7 and `e21_end_half` would score and store **A's** records as B's. Those lines are deterministic and already ran in A, so this is practically impossible, but cheap to rule out.
- **Fix.** Reset `E12_RECORDS = {t: [] for t in TASK_IDS}` and a fresh `E12_RUN` in `e21_begin_half`.

### LOW-6: A non-default commit profile breaks the halves silently.

- **Evidence.** With a profile that has no `solve_window_sec` (for example `Q27X_PROFILE=EVAL12`), `E21_HALF_SOLVE_SEC` = 0. A's cap becomes 42,720 − 1,500, B gets ≤ 25 min, and the CELL 1 guard is skipped (cell 2:422).
- **Fix.** When `len(E21_HALVES) == 2`, assert `E12_SOLVE_WINDOW_SEC`.

### LOW-7: Logs and bundles

- **Location.** CELL 7's per-half bundle (cell 8:1067-1077).
- **Evidence.** It holds only the current server's log. A failed primary's log and relaunch logs are in `q27x/half<H>/` and in CELL 9's `Q27X_E21_<V>_BUNDLE.zip` (glob `half*/vllm_server_*.log`), but not in the per-half zip.
- **Fix.** None required. Mention it in the README.

### LOW-8: A dead half-A server can hold GPU memory into half B.

- **Location.** `_e18_kill_gpu_pids` (cell 7:242-258) relies on `nvidia-smi --query-compute-apps`, which often lists no pids inside Kaggle containers. `killpg` on the group covers vLLM's normal children.
- **Scenario.** If GPU memory is still held after the 180-s wait, B's launch fails with OOM. B's failure path repeats the kill and wait before the fallback; if that also fails, B is lost and the notebook still completes.
- **Fix.** Record the `gpu_release` result of A's end in B's capacity row. Optionally skip B's primary while A's release reported `released: false`.

### LOW-9: Harness coverage gaps

- **Location.** `harness_e21_nb.py`.
- **Evidence.**
  - No harness check compares the two halves' `deadline_in_sec`.
  - No check would catch HIGH-1.
  - The profile check uses `time_budget − hard_margin` (42,900) instead of the notebook's real stop (42,720). It passes either way.
- **Fix.** Add a window-equality check, a gate-slots check, and the real `_stop`.

---

## 5. Commit-mode scoring checked against the solutions (real R3K2 data)

I re-ran the CELL 7 formula against `arc-agi_evaluation_solutions.json`, using the Experiment 10 copy, the same 120 tasks. Inputs: R3K2's `q27x/e12_generated_attempts.json` and, separately, a rebuild from its 198 `duck/*/branch_result.json` files with the real `arc2_duck.build_submission` (env rev5, rev6, B mode, weak commit).

| Computation | Hits / queries | pass1 / pass2 | Task-norm |
|---|---|---|---|
| CELL 7 logic on R3K2's generated attempts | **65 / 172** | 64 / 1 | **39.03 %** (= R3K2's report) |
| Rebuild from branch_result.json, all branches | 65 / 172 (identical dict to generated) | 64 / 1 | 39.03 % |
| E21 `_e21_score` logic, branch 0 only | **46 / 172** | 46 / 0 | **27.92 %** |
| No records (rule library + test input only) | 0 / 172 | 0 / 0 | 0.00 % |

What this shows:
- The per-half CELL 7 score (any of the two attempts equal to the gold grid per test input; task-norm = mean over tasks of the fraction of inputs hit) is correct against the solutions.
- The E21 attempt-1-only score is correct for its definition (LOW-4).
- **R3K2's branch-0-only reference is 46 / 172 (27.92 %).** That is the number to compare each E21 half's `score_attempt1_only` against.
- A half without an engine scores 0, so CELL 7 shows no false credit from the CPU rules.

Other checks:
- `score_full` in `e21_report.json` is read from `q27x/half<H>/e12_run_report.json`, which that half's CELL 7 wrote from its own `E12_RECORDS`.
- `score_attempt1_only` uses the copy that `e21_end_half` takes under `_E12_LOCK`.
- Straggler publishes after the scheduler closes are dropped (`solve_all_r6.publish`: `if state.get("closed") and not final: return`), so no record of half A can reach half B.

---

## 6. Outputs per mode

**Commit:**
- `submission.json` = CELL 0's identity of `arc-agi_test_challenges.json` (240 tasks in the commit data, full schema). `e12_write_submission` rewrites the same identity at every branch publish, at the end of CELL 6 and in CELL 7, for both halves (cell 8:53-62). That is the evaluation fallback, as intended.
- Per half: `q27x/half<H>/` holds:
  - `e12_run_report.json` (with `eval_*`);
  - `e12_generated_attempts.json`;
  - `sglang_decode_summary.json` (vLLM stats, file name kept);
  - `vllm_server_half<H>_<tag>.log` for every launch;
  - `duck/<task>_b<k>/` (full transcripts).
- Bundles:
  - per half `Q27X_E21_<V>_HALF<H>_DUCK_BUNDLE.zip` (submission, report, attempts, decode summary, transcripts, branch results, current server log);
  - `Q27X_E21_<V>_BUNDLE.zip` (`q27x/e21_report.json`, `submission.json`, both half reports, every half server log, decode summaries, `q27x/e14/*`, `e14_state.json`).
  - The harness `run_NB1_fallbackA_210428` folder shows exactly these files.
- Size: R3K2's `q27x` was 799 files / 68 MB, so two halves come to about 1,600 files / ~140 MB. That is far below the 20 GB working limit.

**RERUN:**
- `E21_HALVES = ["A"]`, `PROFILE = RERUN_ALL`, `TASK_IDS` = all 240 test tasks, `SOLUTIONS = None`. No window and no shift: the absolute end is 42,720.
- `submission.json` = CELL 0's identity, overwritten after every branch publish (until the solve end) and finally at the end of CELL 6 and in CELL 7. The rows come from `build_submission` over half A's records, which are the real attempts. A task without a record keeps its identity row (the row count per task is checked).
- Nothing after CELL 7 writes it. CELL 9 only reads and zips.
- The harness `rerun` mode checks `submission.json == q27x/halfA/e12_generated_attempts.json` for every fixture test task.

---

## 7. Half B when half A ends early, fails, or its server dies

| Event in half A | What happens | Half B |
|---|---|---|
| A finishes its work before its deadline | CELL 6 returns, CELL 7, end; B begins at once | full 5.5 h from its own start; the notebook ends earlier; windows equal |
| No A config becomes ready (both fail; worst ~2 × 30 min + kills) | `fatal`, CELL 6 "not run", CELL 7 writes a 0-score half report and bundle, end | begins at ~1-1.2 h, full 5.5 h, notebook ends ~6.7 h; no comparison possible |
| CELL 6 / CELL 7 raise | caught inside (BaseException) or by CELL 8's per-step `try`; `errors` recorded per half | unaffected |
| A's server dies, relaunch OK | branches wait on `SERVER_DOWN`; lost minutes are not compensated (as in R3K2) | unaffected (but see MEDIUM-1 if the windows matter) |
| A's server dies, no relaunch (limit, < 20 min left, or LOW-2) | requests fail until A's deadline; A's score drops | unaffected, unless GPU memory stays held (LOW-8) |
| A relaunch in progress near A's end | bounded: ready timeout ≤ cap − now − 30 min, so it finishes before A's solve end; the watchdog then sees A's stop event and exits | unaffected; no zombie watchdog reaches B |
| A stragglers | daemon threads; their publishes are dropped | each one still holds a gate slot until it ends (HIGH-1) |

---

## 8. Can any path exceed 12 h?

**Commit:** no, with margin. Half B's solve end is absolute (≤ 42,720). Every wait before it is bounded:
- `e18_install` ≤ 2,400 s, at most twice;
- `wait_ready` ≤ 1,800 s;
- smoke ≤ 300 s;
- preflight requests ≤ 180 s each;
- kill ≤ 60 s;
- GPU release ≤ 200 s.

None of these can push solving past the cap. In the worst case B simply solves less. The tail after 42,720 is the same as in RERUN (≤ ~425 s worst). Nominal end ≈ 41,200 s (11.44 h).

**RERUN:** no path is unbounded. The worst-case tail is the risk: ≈ 425 s of the ~473 s left (MEDIUM-2). The nominal end is ≈ 42,800 s (11.89 h).

**No unbounded step found.** The watchdog relaunch, launch, install, smoke, preflight, join and release waits all have explicit timeouts. The vLLM stall rule (600 s of log silence) is well above the measured startup silences (65 s on 0.19, 60 s on 0.32), so it will not kill a healthy launch.

---

## 9. What is correct

1. **Half caps.** Half A's cap 21,420 = `_stop` − window − reserve. Half B's cap and RERUN's end are `_stop` = 42,720. The CELL 1 guard (41,100 < 42,900) holds.
2. **Window start.** Each half's window starts after its own preflight (shift, capped). RERUN keeps the absolute end, with no window and no shift.
3. **Nominal timeline.** Both halves get exactly 19,800 s (+ seconds of shift). The commit ends ≈ 11.44 h. B keeps 26 min of slack.
4. **Same tasks and order.** `TASK_IDS` = all 120 eval tasks in both halves (sorted sample of 120 of 120). `E12_ORDER` is deterministic. Each half has fresh `E12_RECORDS` / `E12_RUN` / sessions and its own `OUT` and `duck/` folder.
5. **Shared state reset per half.** `SERVER_DOWN` / `SERVER_WAITS` / `SERVER_LAST` / `SCHED_LAST` are reset, `E15B_SOLVE_END_PC = inf` until the solve starts, and the reasoning key is reset. The only shared module state left is the gate (HIGH-1), the tokenizer and the deterministic `_RULE_CACHE`.
6. **Same Duck settings in both halves.** Env `E15B` (50k / 45k, before3, weak commit), the fixed 100k window, workers by isolation, and temperature 1.0. Streams come from each half's capacity check (intended; see HIGH-1 and MEDIUM-3).
7. **Server separation.** Distinct ports (1234 / 8011). Each half waits for ready on its own `/health` plus its own KV line. One server at a time: half B begins after A's end, checked by the harness. Explicit `base_url` overrides Duck's env URL (`tool_agent.py:3483-3489`).
8. **Scoring.** CELL 7 scoring is exact against the solutions (R3K2: 65 / 172, 39.03 %, reproduced twice). The E21 branch-0 score logic is correct for its definition (R3K2 reference 46 / 172, 27.92 %).
9. **Submissions.** Commit `submission.json` is the full-schema identity fallback. RERUN `submission.json` holds half A's real attempts for every task that has records, and identity for the rest. Nothing after CELL 7 changes it.
10. **Per-half outputs.** Folders, the per-half bundle names and the final bundle are correct and not overwritten between halves. A failed launch keeps its own log (`vllm_server_half<H>_primary.log`).
11. **Isolation of failures.** Each step is isolated in CELL 8, so a failed half never stops the next. A begin error sets `ENGINE_READY = False` at module level.
12. **Bounded waits.** Every wait is bounded. No path makes the commit exceed 12 h; RERUN's only risk is the tail margin (MEDIUM-2).

---

## 10. Limits of this review

- The vLLM 0.32 INT4 launch time is not measured anywhere. I assumed it is about equal to E18's FP8 550 s; VMS ran INT4 0.32 at 20 streams but did not record a ready time.
- How long Kaggle's 12-h clock runs before the kernel process starts is unknown. Previous LB runs with these constants finished with ~20-s tails.
- I did not run the notebooks or the harness. The gate behaviour was reproduced with the vendor's own gate source, and the scoring with the real `arc2_duck.build_submission` on R3K2 data.
