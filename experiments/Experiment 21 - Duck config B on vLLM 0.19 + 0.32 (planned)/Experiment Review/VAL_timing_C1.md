# VAL timing C1: the E21 clock (commit and hidden rerun)

**Validator:** C1 (clock only). Read-only: no project file was edited and nothing was sent to Kaggle.
**Date:** 2026-10-28.

**Inputs read in full (the parts that touch time):**
- Both built notebooks, extracted cell by cell. NB1 and NB2 differ only in CELL 1: the variant, the half specs and `E21_STAGE_PATHS`. Every other cell is byte-identical.
- `Build/build_e21.py` and `Build/e21_engine_new.py`.
- The README.
- The donor cells `E15B_R3K2_c01/c02/c09/c10`.
- `arc2_duck.py` b9f67010 (the Duck package R3K2 ran): `solve_all_r6`, the join, the final grace, `publish` and `freeze`.
- The R3K2 commit evidence (`hosen42`, c058d58b): its Kaggle log, `e14_state.json` and `e14_results.json`.
- The harness README.

All times below are seconds after `NOTEBOOK_T0`, also shown as h:mm:ss.

**Where Kaggle's 12-h limit falls on this clock:**
- R3K2's log printed CELL 0's line at log time 9.2 s, with the clock "anchored 1.8 s before cell 0". So the Kaggle log clock starts about 7 s before `NOTEBOOK_T0`.
- The 12-h limit is therefore at about **T0 + 43,193 s**. It could be a little earlier if Kaggle counts any time before its log starts; that is unknown.

---

## 1. Every deadline actually used

### 1.1 CELL 0 and CELL 1 (identical in both modes unless noted)

| Name | Formula (code) | Commit (`E21X2EVAL120`) | Rerun (`RERUN_ALL`) |
|---|---|---|---|
| `NOTEBOOK_T0` | `perf_counter()` at CELL 0 minus the kernel-process age (psutil; capped at 600 s; 0 if psutil fails) | anchor | anchor |
| `TOTAL_BUDGET_SEC` | `P["time_budget_sec"]` = 12 h − 3 min | 43,020 (11:57:00) | 43,020 (11:57:00) |
| `VERIFY_DEADLINE` | T0 + budget − 60 (reporting only) | 42,960 (11:56:00) | 42,960 |
| `HARD_STOP` | T0 + budget − `hard_margin_sec` 120 | **42,900 (11:55:00)** | **42,900 (11:55:00)** |
| `DRAIN_DEADLINE` | HARD_STOP − 480 (Poetiq path; never read by the Duck run) | 42,420 | 42,420 |
| `ADMIT_DEADLINE` | DRAIN − 120 (only feeds `SGL_GIVE_UP_AT`) | 42,300 | 42,300 |
| `SGL_GIVE_UP_AT` | min(T0 + (240 if RERUN else 75) min, ADMIT − 300) | 4,500 (1:15:00) | 14,400 (4:00:00) |
| Staging deadline (CELL 1b) | max(T0 + 600, min(T0 + 2,700, GIVE_UP − 1,200)) | 2,700 (0:45:00) | 2,700 (0:45:00) |
| Staging join bound (CELL 5) | min(GIVE_UP − 1,200, T0 + 3,300) | 3,300 (0:55:00) | 3,300 (0:55:00) |
| `E12_STOP_BEFORE_HARD_SEC` | 180 | 180 | 180 |
| `_stop` = HARD_STOP − 180 | the notebook's solve end | **42,720 (11:52:00)** | **42,720 (11:52:00)** |
| `E12_SOLVE_WINDOW_SEC` / `E21_HALF_SOLVE_SEC` | `float(env E21_HALF_SOLVE_SEC or P["solve_window_sec"])` if the row has a window, else None / 0.0 | 19,800 (5:30:00) per half | **None / 0.0** (no window) |
| `E21_B_SETUP_RESERVE_SEC` | env or 1,500 | 1,500 | 1,500 (unused: one half) |
| CELL 1 assertion | `2W + R < HARD_STOP − T0` (commit only) | 41,100 < 42,900 (spare 1,800) | skipped |

### 1.2 Per half (`e21_begin_half`, `e21_launch`, CELL 6)

| Name | Formula | Commit A | Commit B | Rerun A |
|---|---|---|---|---|
| `E21_HALF_END_PC` (half cap) | `_stop` if H ≠ A or one half, else `_stop − W − R` | **21,420 (5:57:00)** | 42,720 (11:52:00) | 42,720 (11:52:00) |
| Ready timeout per launch | `max(60, min(1,800, cap − now − 1,800))` | 1,800 | 1,800 | 1,800 |
| Stall (log silence) | `E18_SILENCE_SEC` 600 | 600 | 600 | 600 |
| `/v1/models` and smoke | 30 s and 300 s timeouts | ≤ 330 | ≤ 330 | ≤ 330 |
| Fallback skipped when | now > cap − 1,800 | after 19,620 | after 40,920 | after 40,920 |
| Failed-launch teardown | kill (SIGTERM, wait ≤ 30; SIGKILL, wait ≤ 30) + `nvidia-smi` pid kill (≤ 20) + `_wait_gpu_release` (≤ 180, overshoot ≤ 25) | ≤ ~285 | ≤ ~285 | ≤ ~285 |
| `e18_install` (vLLM 0.32 venv) | `subprocess.run(..., timeout=2400)`, once per notebook if it succeeds | — | ~175 (≤ 2,400) | — |
| Solve end, fixed at the solve start | min(`_stop`, cap); then min(·, now + W) when there is a window | min(21,420, t_fix + W) | min(42,720, t_fix + W) | 42,720 |
| Window shift after the preflight | `if not RERUN and W`: end += preflight seconds, capped at min(`_stop`, cap) | yes → **min(21,420, S_A + W)** | yes → **min(42,720, S_B + W)** | no (absolute end, as in R3K2) |
| `E15B_SOLVE_END_PC` | = solve end; per-branch submission writes stop here (B1) | D_A | D_B | 42,720 |
| Scheduler `deadline` | monotonic + (end − perf_counter) | D_A | D_B | 42,720 |
| `min_admit_sec` | `E12_R6_MIN_ADMIT_SEC` 300: no new attempt in the last 5 min | D_A − 300 | D_B − 300 | 42,420 |
| KV gate | no admission while `vllm:kv_cache_usage_perc` ≥ 0.95 (unless nothing runs), polled every 5 s; `/metrics` timeout 10 s | same | same | same |
| Per-request clamp | every Duck request clamped to the deadline (R6-S6); `E12_DUCK_REQUEST_TIMEOUT` 3,600 is only the ceiling | — | — | — |
| Final grace per attempt | `E12_B_FINAL_GRACE_SEC` 30 (harvest + partial test after the deadline) | inside the join | inside the join | inside the join |
| Join grace | `join_until = deadline + 90` (one absolute cutoff; stragglers are left as daemon threads) | D_A + 90 | D_B + 90 | 42,810 |
| Watchdog | poll 10 s; fail after 180 s of `/health` failing with a silent log; ≤ 2 relaunches; none with < 1,200 s left | same | same | same |
| Cache diagnostic | `E12_R6_CACHE_DIAG = 0` (forced off in CELL 1) | not run | not run | not run |
| CELL 7 (report + per-half bundle) | R3K2: about 3 s (summary 21,308.3 → bundle 21,311.0) | ~5 | ~5 | ~5-20 (240 tasks) |
| `e21_end_half` | decode summary + kill (≤ 60) + pid kill (≤ 20) + GPU release (≤ ~205) | typical 20-50, worst ~285 | same | same |
| CELL 9 (E21 report + bundle) | zips the JSON files, the logs and the submission | ~5 | — | ~5 |

### 1.3 Setup times used in the timelines

| Step | Value | Source |
|---|---|---|
| CELL 0 / 1 / 1b | ~10 s | R3K2 log 9.2 → 9.8 s |
| Staging copy (runs during the install) | INT4 19.47 GB in 93.5 s (R3K2), 102 s (E14A); FP8 ~30 GB, ~150 s (estimate) | R3K2 `e14_results.json` |
| vLLM 0.19 install into the kernel (CELL 2) | ~160 s | task brief (S1) |
| Tokenizer, CELL 3 (Duck + sandbox self-test), order cell, CELL 5 (staging join ≈ 0 s, libcuda probe) | ~25 s | R3K2 log |
| vLLM 0.19 ready | 550-1,200 s | task brief; R3K2 SGLang: 522 s |
| `/v1/models` + capacity + smoke | ~10 s | — |
| Preflight (isolation 16-20 parallel, 24 tokens; 4 serial parity requests) | ~3-10 s | R3K2: 2.9 s |
| Join after the deadline | R3K2: 8 s; ≤ 90 s | R3K2 log |
| vLLM 0.32 venv install + ready | ~175 s + ~550 s | task brief (VMS) |
| GPU release | up to 180 s (+ `nvidia-smi` overshoot) | `_wait_gpu_release` |

---

## 2. Timelines

### 2.1 Commit, the typical case (both servers ready in 550 s)

| Event | T0 + s | Clock |
|---|---|---|
| CELL 0 (fallback `submission.json` written) | 0-10 | 0:00:10 |
| Staging starts (copy ~95-150 s, under the install) | ~10 | |
| CELL 2: vLLM 0.19 install | 15 → 175 | 0:02:55 |
| CELL 3, order cell, CELL 5 (join instant) | → ~200 | 0:03:20 |
| Half A launch (0.19) → ready (550) → smoke | 200 → 760 | 0:12:40 |
| Solve end fixed, preflight, shift → **S_A** | ~770 | 0:12:50 |
| **D_A** = min(21,420, S_A + 19,800) | 20,570 | 5:42:50 (full 5.5 h) |
| Join (≤ 90), summary, final write, CELL 7 | → ~20,595 | |
| `e21_end_half(A)`: kill, pid kill, GPU release | → ~20,625 | |
| Half B: `e18_install` (175) → launch → ready (550) → smoke | → ~21,360 | |
| Preflight, shift → **S_B** | ~21,370 | 5:56:10 |
| **D_B** = min(42,720, S_B + 19,800) | 41,170 | 11:26:10 (full 5.5 h) |
| Join, CELL 7, `e21_end_half(B)`, CELL 9 | → ~41,230 | **11:27:10 end** (about 32 min unused) |

### 2.2 Commit, other cases (computed with the same model)

| Case | S_A | Window A | D_A | S_B | Window B | D_B | Notebook end |
|---|---|---|---|---|---|---|---|
| Typical (ready 550 / 550) | 770 | 19,800 | 5:42:50 | 21,370 | 19,800 | 11:26:10 | 11:27:10 |
| Slow ready (1,200 / 1,200) | 1,420 | 19,800 | 5:53:40 | 22,670 | 19,800 | 11:47:50 | 11:48:50 |
| A ready 1,400; B ready 1,200; worst join (90) and teardown (285) | 1,620 | 19,800 | 5:57:00 | 23,215 | **19,505** | 11:52:00 | **11:58:50** |
| A primary times out (1,800 s), fallback ready in 1,200 | 3,505 | **17,915** | 5:57:00 | 22,220 | 19,800 | 11:40:20 | 11:41:20 |
| A typical; B primary times out (1,800 s), fallback ready in 550 | 770 | 19,800 | 5:42:50 | 23,455 | **19,265** | 11:52:00 | 11:53:00 |
| A capped (S_A 1,720); B primary times out | 1,720 | 19,700 | 5:57:00 | 24,305 | **18,415** | 11:52:00 | 11:53:00 |
| A capped; B primary stalls (fails at ~900 s), fallback ready in 550 | 1,720 | 19,700 | 5:57:00 | 23,160 | **19,560** | 11:52:00 | 11:53:00 |

### 2.3 Hidden rerun (`RERUN_ALL`, half A only, 240 tasks)

| Event | T0 + s | Clock |
|---|---|---|
| CELL 0: identity `submission.json` for all 240 hidden tasks | 0-10 | |
| Install, staging, half A launch, ready, smoke, preflight → S_A | 770-1,420 | 0:12:50-0:23:40 |
| Solve (no window, no shift): end = `_stop` | **42,720** | **11:52:00** |
| Join (R3K2: 8 s; ≤ 90) | ≤ 42,810 | |
| Post-solve summary + final `submission.json` write | ≤ ~42,825 | ≤ 11:53:45 (before HARD_STOP 11:55:00) |
| CELL 7 (report, bundle, rewrites `submission.json`) | ≤ ~42,845 | |
| `e21_end_half(A)`: typical 20-50 s / worst ~285 s | ~42,770-42,895 / ≤ 43,130 | |
| CELL 9 → notebook end | typical ~42,780-42,900 (≤ 11:55:00); worst **~43,140 (11:59:00)** | Kaggle limit ≈ 43,193: about 53 s spare at worst |

**Compute per task:**

| Run | Seconds of solving per task |
|---|---|
| Rerun | ≈ (42,720 − S_A) / 240 ≈ **172-175** |
| Commit, per half | 19,800 / 120 = **165** |
| R3K2 | 20,467 / 120 = 170.6 |

So a commit half gets about 95 % of the hidden run's per-task compute: a slightly conservative proxy for the hidden run.

---

## 3. Answers

### (1) Does half B always get its full window?

**No, but it does in every typical case.**

**Exact condition.** B gets the full 19,800 s if and only if its post-preflight solve start is no later than `_stop − W`:

> **S_B ≤ T0 + 22,920 (6:22:00).**

**The same condition as a gap.** Let G be the time from D_A to S_B:
- join;
- summary;
- CELL 7;
- `e21_end_half(A)`;
- `e18_install`;
- B's launch (including any failed primary and its teardown);
- smoke;
- B's preflight.

Then B is full if and only if:

> **G ≤ max(1,500, 3,120 − S_A)**

The 3,120 − S_A term applies when A was not capped (S_A ≤ 1,620). The bare 1,500 s applies when A ran to its cap of 21,420.

**Typical G** is about 800-1,000 s with B ready in 550 s, and about 1,450 s with B ready in 1,200 s. Both fit. With A ready in 550 s the allowance is about 2,350 s.

**B is cut when G exceeds the allowance.** The cut is `S_B − 22,920` seconds, taken from B's window (its end is pinned at 11:52:00). Concretely:
- **B's primary fails slowly:**
  - by the 30-min ready timeout: B loses about 8-23 min (19,265 s with A typical; 18,415 s with A capped);
  - by a stall that ends at about 900 s: up to about 6 min when A was capped.
- **`e18_install` fails by its 2,400-s timeout.** The failure is not cached (`E21_STATE["v032_install"]` stays None), so the fallback runs the install again. That costs up to another 40 min, and B usually gets no solve at all.
- **Everything is at its worst at once** (A capped at 21,420; join 90; end_half 285; B ready 1,200). B is cut by 295 s.
- **B's preflight hits its timeouts.** Each preflight request has a 180-s timeout, and an INCONCLUSIVE result is retried once. Worst case about 1,800 s, taken directly from B's window. A's shift cannot rescue B here because B's end is already at `_stop`.

### (2) Can anything run past the 12-h limit?

**No path found that exceeds 12 h, but the worst case is close.** Nothing is guarded after the outputs are written.

**What is bounded:**
- Every solve end is ≤ 42,720 and every Duck request is clamped to it.
- The join is ≤ +90 s.
- The cache diagnostic is off.
- Watchdog relaunches always finish before their solve end in realistic configurations; the proof is in finding L1.

**The notebook's end after the last solve:**
- join ≤ 90;
- summary ≤ 15 (one `/metrics` call with a 10-s timeout);
- final write;
- CELL 7 about 5-20 s;
- `e21_end_half`: kill ≤ 60 + pid kill ≤ 20 + `_wait_gpu_release` ≤ ~205;
- CELL 9 about 5-10 s.

**Result:**
- Worst case: **T0 + ~43,140 (11:59:00)**, against a limit of about T0 + 43,193. About 53 s spare.
- Typical: 11:52:58 (rerun) or earlier (commit).

The worst case needs a capped last half (the rerun always is) plus a server or GPU that does not release; a stuck process after an Xid or NCCL hang is exactly that case. See finding M1.

### (3) Rerun: the solve end, and `submission.json` validity

**Solve end: yes, it is R3K2's design.**
- `RERUN_ALL` has no `solve_window_sec`, so `E12_SOLVE_WINDOW_SEC` is None and the window code is skipped.
- `RERUN` skips the shift.
- `E21_HALVES = ["A"]`, so `E21_HALF_END_PC = _stop` and the B reserve is not applied. The CELL 1 assertion is skipped.
- So `_end = min(HARD_STOP − 180, HARD_STOP − 180) = 42,720 (11:52:00)`. This is exactly R3K2's `_end = HARD_STOP − E12_STOP_BEFORE_HARD_SEC` with the same `RERUN_ALL` row.
- The final `submission.json` is written at ≤ ~42,825, before HARD_STOP (11:55:00).
- Per-branch writes stop at `E15B_SOLVE_END_PC`. Records published after that still update `E12_RECORDS`, inside the lock and before the time check, so the final write contains them.

**What is new compared with R3K2:** E21 adds `e21_end_half` after the outputs (up to ~285 s). R3K2's CELL 7 killed the server without waiting and ended about 11 s after the solve. This teardown is the only part that brings the rerun near the limit (M1).

**`submission.json` when the engine never starts: complete and valid.**
- CELL 0 writes `identity_submission(test_challenges)`: every task, every test input, with attempt_1 = attempt_2 = the input grid.
- `e12_write_submission` always starts from that identity and only replaces a task whose row count matches.
- When neither config becomes ready (worst case about 1,800 + 285 + 1,800 + 285 s ≈ 1 h 12 min), CELL 6 takes its "not run" branch. It still performs the final write (identity), and CELL 7 rewrites it once more (identity).
- Every step of CELL 8 and CELL 9 is wrapped in try/except, so the notebook ends cleanly about 1.2 h in, with a valid identity submission (it scores about 0).

**One caveat outside the clock (inherited, info only).** An exception at module level in CELL 1b, 2 or 3 stops Run All. Examples: the payload dataset missing, the wheelhouse missing, or a failed `pip`. Kaggle then reports a notebook error whatever the CELL 0 file contains. R3K2 behaved the same way.

### (4) Is the half-A cap formula right? What if half A's engine fails at minute 10?

**The formula is right for what it protects.** `E21_HALF_END_PC(A) = _stop − W − R = 42,720 − 19,800 − 1,500 = 21,420 (5:57:00)`, and B's end is `_stop`.

**What the cap guarantees.** When A runs to its cap, B gets its full W whenever G ≤ R = 1,500 s. With an assumed ready time of 550-1,200 s, typical G is 800-1,450 s, so it fits.

**What the cap does not guarantee:**
1. **A full window for A.** A is full only if S_A ≤ 21,420 − 19,800 = **1,620 s (27 min)**. Any longer setup is taken from A, never from B. That happens when:
   - A is ready only after more than ~1,400 s;
   - a fallback is needed, unless the primary fails fast;
   - staging is slow (the join may wait until 0:45-0:55).

   The two halves are then **asymmetric** (example above: A 17,915 s against B 19,800 s). That biases the 0.19-vs-0.32 comparison against A. See M2.
2. **A full window for B.** R has only about 50-700 s of slack for the realistic G, and less than zero in the worst case (G ≈ 1,800). See M3.

**"Half A's engine fails at minute 10", three readings:**
- **(a) The A launch fails at T0 + 600** (the primary exits).
  - The teardown takes about 5-285 s, then the fallback is launched.
  - The fallback is ready at about 1,155-2,010 s. NB1's FP8 fallback loads unstaged from `/kaggle/input`, so it sits at the slow end (finding M4).
  - S_A ≈ 1,175-2,030, so A's window is min(19,800, 21,420 − S_A) = **19,390-19,800 s** (at most about 7 min short).
  - B is unaffected.
- **(b) The A server dies 10 min into the solve.**
  - The watchdog notices within 10 s (dead process), or after 180 s (hung, silent log).
  - It relaunches the same config: teardown ≤ 285 + ready 550-1,200 + smoke. Branches wait on `SERVER_DOWN`.
  - D_A does not move, so A loses about 10-25 min of solving inside its window. B is unaffected.
  - If both relaunches fail, the branches burn their failure budgets and the scheduler drains early. A ends early, B starts early and gets its full W, and the run ends early.
- **(c) Both A configs fail** (for example, the primary at 10 min and the fallback by its timeout).
  - CELL 6 runs nothing; CELL 7 and `end_half` take seconds.
  - B begins at about T0 + 3,000-4,300 and gets exactly W (5.5 h). The notebook ends at about 6.5-7 h.
  - The A arm is lost and about 5 h of budget goes unused. That follows from the fixed-window design, not from a bug; it is listed as L5.

### (5) The CELL 1 assertion `2W + R < HARD_STOP − NOTEBOOK_T0`

**It is correct arithmetic, but it is static and too loose to mean "both windows fit".**
- `HARD_STOP − NOTEBOOK_T0` is the constant 42,900, so the check is 41,100 < 42,900 on every run (1,800 s spare). It never sees the real setup time.
- It compares against HARD_STOP, but both solve ends are at `HARD_STOP − 180`.
- It leaves out half A's pre-solve setup: install, staging join, ready, smoke and preflight, about 13-24 min in the typical range.

**What it actually implies.** The spare 1,800 − 180 = **1,620 s is the implicit budget for A's setup**. Passing the assertion only proves that A could get W − 180 s if it started at T0.

**The real condition for two full windows** is S_A + G ≤ 42,720 − 2W = **3,120 s (52 min)**.

**A meaningful guard**, for example:

```python
E21_A_SETUP_BUDGET_SEC = 25 * 60     # install + staging join + ready (<= 1,200) + smoke + preflight
assert 2 * W + R + E12_STOP_BEFORE_HARD_SEC + E21_A_SETUP_BUDGET_SEC <= HARD_STOP - NOTEBOOK_T0
```

That gives 42,780 ≤ 42,900, which holds with 120 s spare: the configuration is feasible but tight. Add a runtime line at A's solve start that prints the predicted latest S_B (22,920), and record whether A's window was capped.

---

## 4. Findings

No BLOCKER was found: no path loses the outputs, runs past 12 h, or yields an invalid submission.

### M1: MEDIUM. The teardown after the last half can bring the notebook within about 1 min of 12 h

- **Location:**
  - CELL 8 runner: `("end", lambda: e21_end_half(_H))` for every half, including the last one.
  - `e21_end_half`: `E11_SERVER.kill()` (≤ 60 s), `_e18_kill_gpu_pids()` (≤ 20 s), `_wait_gpu_release()` (`max_sec` 180, plus up to ~25 s of `nvidia-smi` overshoot).
- **Evidence:**
  - The rerun solve ends at 42,720 and the join at ≤ 42,810. The outputs are written by about 42,845.
  - Then `end_half` runs, with a worst case of ~285 s, followed by CELL 9. The end is ≈ 43,140, against the Kaggle limit ≈ 43,193.
  - R3K2 had no wait: its CELL 7 called `e11_kill_server(wait=False)` and finished about 11 s after the solve.
  - The harness sets `Q27X_GPU_WAIT_MAX=1`, so it never exercises this wait.
- **Scenario:**
  - A hidden rerun, or a commit whose half B was cut, where the server or a worker is stuck (Xid, NCCL hang, D-state).
  - SIGTERM waits 30 s, SIGKILL waits 30 s, and the GPU memory never drops below 1 GB, so the full 180 s plus overshoot is spent.
  - Any further delay (CELL 7's bundle of 240 tasks of transcripts, slow `/kaggle/working` I/O, or Kaggle's clock starting earlier than its log) pushes the run over 12 h. A timed-out rerun is not scored, even though `submission.json` is on disk.
- **Fix:**
  - For the last half (`_H == E21_HALVES[-1]`), skip `_wait_gpu_release`: use `E11_SERVER.kill(fast=True)` and no wait, as R3K2 effectively did.
  - Or bound the wait by the clock: `_wait_gpu_release(max_sec=max(0.0, min(180.0, HARD_STOP - time.perf_counter())))`.

### M2: MEDIUM. The windows are asymmetric when A's setup exceeds 27 min, which biases the comparison against vLLM 0.19

- **Location:** `e21_begin_half`, line `E21_HALF_END_PC = _stop if (...) else _stop - E21_HALF_SOLVE_SEC - E21_B_SETUP_RESERVE_SEC`, together with CELL 6's `_end = min(..., E21_HALF_END_PC)`.
- **Evidence:**
  - A's window is min(19,800, 21,420 − S_A).
  - B's window is measured from B's own start.
  - So any A setup beyond 1,620 s is taken from A only. Example: the A primary times out and the fallback is ready in 1,200 s. Then S_A = 3,505, A gets 17,915 s and B gets 19,800 s, which is 9.5 % less compute for A.
- **Triggers:**
  - A ready only after more than ~1,400 s (the 2026-10-21 GCSFUSE slowdown gave 1,197 s on SGLang).
  - Any half-A fallback except a fast exit. NB1's INT4 on vLLM 0.19 is untested (README Finding 5).
  - Slow staging: the join may wait until 2,700-3,300 s.
- **Scenario:** NB1 half A falls back to FP8 after a stall. A then runs about 5.0 h and B 5.5 h, and the "0.19 vs 0.32" delta is partly a time delta.
- **Fix, either of:**
  - (a) Give B the window A actually got. At A's solve start, store `W_eff = E12_RUN["deadline_in_sec"]` after the shift, and use `min(W, W_eff)` for B.
  - (b) Keep the code and compare the halves post hoc at equal time: truncate both to the shorter effective window using the attempts' t0/t1. `deadline_in_sec` is already in `e21_report.json`. Add `window_shift` there as well.

### M3: MEDIUM. The 25-min half-B reserve has little slack, and a slow or failed B launch is taken straight from B's window

- **Location:**
  - `E21_B_SETUP_RESERVE_SEC = 1,500`.
  - `e21_launch`: B's ready timeout of 1,800 s per config.
  - `e18_install`: 2,400-s timeout, failure not cached.
  - The B preflight request timeouts of 180 s.
- **Evidence:** G (D_A → S_B) breaks down as:

  | Component | Typical (s) | Worst (s) |
  |---|---|---|
  | Join | 8 | 90 |
  | Summary | ~5 | ~15 |
  | CELL 7 | ~5 | ~5 |
  | `end_half` | 20-50 | 285 |
  | Install | 175 | 175 |
  | Ready | 550-1,200 | 1,200 |
  | Smoke + preflight | ~20 | ~20 |
  | **Total G** | **≈ 800-1,450** | **≈ 1,800** |

  The allowance is max(1,500, 3,120 − S_A).
- **Scenarios:**
  - The B primary stalls or times out: B is cut by 4-25 min (table 2.2).
  - The install times out: it is retried for the fallback (`E21_STATE["v032_install"]` is still None), adding up to 40 min more and usually leaving no B solve.
  - The B preflight retry hits its timeouts: worst ~1,800 s off B's window.
- **Fix:**
  - Cache an install failure: store the exception, and make the fallback skip a second `e18_install` when the first one timed out.
  - Lower B's primary ready timeout to about 900-1,200 s (0.32 is ready in ~550 s), so a hung primary still leaves the fallback inside R.
  - Either raise R to about 30 min, or shorten W slightly (5.45 h). The feasibility bound is S_A + G ≤ 3,120.
  - Bound the preflight request timeout to about 60 s.

### M4: MEDIUM. NB1 stages only the INT4 weights; its FP8 fallback (both halves) loads from GCSFUSE

- **Location:** CELL 1: `E21_STAGE_PATHS = [SGL_MODEL_PATH]` (NB1); `e21_model_path("fp8")` → `MODEL_PATH` (`/kaggle/input`).
- **Evidence:**
  - Staging copies only the INT4 folder, 19.47 GB in about 95 s.
  - The NB1 fallback (FP8, 16 streams) reads about 30 GB through GCSFUSE. The CELL 1b single-stream read probe in R3K2 measured 34 MB/s; under the 10-21 slowdown, ready took 1,197 s.
- **Scenario:** INT4 fails on vLLM 0.19 (untested). The FP8 fallback is then slow to load, so S_A passes 1,620 and A is cut (M2). In half B the same fallback pushes G past R (M3).
- **Fix:** stage FP8 too for NB1.
  - Cost: about 30 GB more in `/dev/shm` (R3K2 headroom was 185 GB) and roughly 150 s of copying, which runs under the 160-s install.
  - Alternative: start a second staging of `MODEL_PATH` as soon as the primary fails.

### L1: LOW. The relaunch ready timeout uses the half cap, not the solve end

- **Location:** `e21_launch`: `left = E21_HALF_END_PC - time.perf_counter() - 30 * 60`. This is used for `e21_relaunch` too.
- **Evidence:**
  - The watchdog relaunches only with at least 1,200 s left before `_e15b_solve_end`.
  - A relaunch then takes at most about 285 + (L + gap − 2,085) + 335 + 285 = **L + gap − 1,180**, where gap = cap − solve end.
  - In a realistic half A, gap = max(0, 1,620 − S_A) ≤ ~850 s, so every relaunch ends before D_A. That is the proof behind answer (2).
  - When half B started early (A failed or finished early), gap can be large (for example, about 19,900 s).
- **Scenario:**
  - B's server dies 20-30 min before D_B. The relaunch thread waits up to 1,800 s and outlives the solve.
  - Meanwhile CELL 7 and `end_half(B)` run. If `end_half` runs before the thread assigns `E11_SERVER`, the newly started server is never killed.
  - On failure the thread sets `E11_SERVER = None` and runs another teardown, possibly while CELL 9 runs.
  - Harmless for the outputs, because B is the last half and D_B is then early. The same race in half A would need gap > 1,180, which is not realistic.
- **Fix:** `left = min(E21_HALF_END_PC, globals().get("E15B_SOLVE_END_PC", float("inf"))) - time.perf_counter() - 30 * 60` (and keep `max(60, ...)`).

### L2: LOW. The CELL 1 assertion is static and does not prove both windows fit

- See answer (5) for the analysis and the suggested guard.

### L3: LOW. Setting `E21_HALF_SOLVE_SEC=0` silently disables the windows and starves half B

- **Location:** CELL 1: `E12_SOLVE_WINDOW_SEC = float(env ...)`.
- **Evidence:**
  - 0.0 is falsy, so the assertion is skipped and CELL 6 sees no window.
  - A's cap becomes 42,720 − 0 − 1,500 = 41,220 (11:27:00).
  - B then gets about 1,500 − G ≈ 500 s, or nothing.
- **Fix:** `assert E21_HALF_SOLVE_SEC > 0 or RERUN`.

### L4: LOW. CELL 7's server kill is a silent no-op

- **Location:** E15B CELL 7 tail: `e11_kill_server(wait=False)` → `E11_SERVER.kill(wait=False)`.
- **Evidence:** `E18Server.kill(self, fast=False)` has no `wait` keyword. The call raises a TypeError, which `except Exception: pass` swallows, so the server keeps running until `e21_end_half`.
- **Clock effect:** none. `end_half` kills it. This only matters together with M1, if you choose to drop the wait for the last half: CELL 7's kill would not happen either.
- **Fix:** call `E11_SERVER.kill(fast=True)` there, or drop the line.

### L5: LOW (design note). When half A fails completely, about 5 h of the commit is unused

- B keeps its fixed 5.5 h window from its own start, at about T0 + 3,000-4,300, and the notebook ends at about 6.5-7 h.
- This is the right choice for comparability. If you also want an A data point after a total failure, a later relaunch of A after B would fit in the time left.

### L6: LOW (info). E21's windows are 3.3 % shorter than R3K2's

- Each E21 half gets 19,800 s; R3K2 got 20,467 s (from the log: "stops in 20467 s").
- Per task, a commit half gets 165 s, the rerun about 172-175 s, and R3K2 170.6 s.
- E21 commit scores are therefore a slightly conservative comparison to both R3K2 (39.03 %) and the hidden run.
- Minor detail: `CONFIG["e12"]["E12_SOLVE_WINDOW_SEC"]` is recorded before the E21 block redefines the window. It shows the stale value only if the env override is used. `CONFIG["e21"]["half_solve_sec"]` is correct.

---

## 5. What is correct

- **CELL 0 anchoring** (kernel-process age, capped at 600 s) is verbatim R3K2. The builder asserts it is identical across all donors. Every deadline derives from the same T0, and the perf_counter → monotonic conversion for the Duck deadline is consistent.
- **Deadline chain.** The `ADMIT < DRAIN < HARD_STOP < VERIFY < budget end` assertion holds: 42,300 < 42,420 < 42,900 < 42,960 < 43,020. The Duck run uses only HARD_STOP − 180. DRAIN and ADMIT are not read by the Duck path.
- **Rerun solve end** = 42,720 (11:52:00), no window and no shift: identical to R3K2's design. The final outputs are written before HARD_STOP (≤ ~11:53:45).
- **Commit windows:**
  - Each half's window starts after its own preflight (shift).
  - Each window is capped by the half cap and by `_stop`.
  - Half A's cap leaves B its W whenever G ≤ R. Typical cases give both halves the full 5.5 h, and the notebook ends at about 11:27-11:49.
- **Launches:**
  - The ready timeout is `max(60, min(1,800, cap − now − 1,800))`.
  - The fallback is skipped when less than 30 min remain.
  - A failed launch kills its process group and the GPU pids and waits for release before the next config.
- **Watchdog.** No relaunch with less than 20 min left, at most 2 relaunches, and branches wait on `SERVER_DOWN` (bounded by the deadline). Each relaunch is proven to finish before the solve end in half A (L1 proof).
- **Scheduler:**
  - `min_admit_sec` 300 (no new attempt in the last 5 min), KV gate 0.95 polled every 5 s, every request clamped to the deadline, final grace 30 s.
  - One absolute join cutoff (deadline + 90). After the scheduler closes, stragglers cannot publish (`publish` returns when `state["closed"]`; sessions are frozen).
  - So half A's stragglers cannot touch half B's `E12_RECORDS` or its score.
  - Half A's heartbeat and watchdog threads wait on half A's `_E12_STOP` object and exit when it is set, before half B rebinds the name.
- **Staging:**
  - It is bounded: the internal deadline is 2,700 and the join bound 3,300.
  - It is fail-open: the original paths are used if it is not finished.
  - In practice the copy (about 95-150 s) finishes under the 160-s install, so the join costs about 0 s.
- **Cache diagnostic** is forced off, so no post-solve time is spent before the outputs.
- **`submission.json`** is complete and valid in every engine outcome:
  - the identity fallback is written in CELL 0;
  - every write starts from that identity;
  - the final write happens even when nothing ran;
  - per-branch writes stop at the solve end and the final write includes the last records.
- **Nothing runs past 12 h.** Worst case ≈ 11:59:00 (M1); typical end 11:27-11:53.
