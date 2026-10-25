# 08 — Rev 3: the user's decisions of 2026-10-25 and the three agent audits

## User decisions (2026-10-25)

1. Take A1-A7 (Notes/07).
2. **D-A:** attempt 2 starts once every attempt 1 has **started** (as built). It is not a finish barrier.
3. **The commit run = 20 eval tasks, with BOTH attempts finishing.** The rerun = all 240 hidden tasks, as before.
   - Built: the 16 G4B tasks + the first 4 by id of the 29 tasks solved in 1-2 of the 3 EVAL120 G4B runs (2026-10-24): 16de56c4, 1ae2feb7, 221dfab4, 269e22fb. That is 20 tasks, 40 attempts, 32 test inputs.
   - Solve window 78 min (auditor 3 R4), budget 100 min.
4. Three independent agent audits of the notebook. The audits must also read the earlier notebook analyses, to confirm that the old bugs are fixed and that no new problem is missed.
5. A Kaggle CPU test.
6. A GPT audit zip.

## The three agent audits

The reports are in `Notes/agent audits/`. Each finding below was checked in the code before acting on it.

### Auditor 1 — Duck package

| # | Finding | Verdict | Change |
|---|---|---|---|
| F1 | The end-of-attempt harvest re-defines the retained functions from the thinking text before the partial test. The partial test then runs an unchecked draft (reproduced; EVAL120 data: 5 of 7 and 1 of 8 such branches). | **taken** | The partial test runs first. Also, inside `arc2_b_reasoning_harvest`, the retained functions are restored when the harvest accepts nothing (so any caller is safe). Unit-tested. |
| F2 | After a rejected submit inside the commit zone, neither the tool result nor the next opener carries the commit text (reproduced). The harvest's tool call records a commit that was never sent (reproduced). | **taken** | Inside the commit zone, the opener gives the commit text plus the rejection summary. The tool-result note no longer requires "no pending submit". The harvest sets `_arc2_b_harvesting`, so it never marks a commit. Unit-tested. |
| F3 | The remaining calls of a batch after a deadline stop run with an unclamped 30-s sandbox timeout. | **taken** | `_run_python_tool` clamps the timeout to the deadline + grace. |

### Auditor 2 — notebook wiring and engine

| # | Finding | Verdict | Change |
|---|---|---|---|
| F1 | ~480 `submission.json` rebuild + rewrites after the hidden solve end: 0.27-0.56 s each = 2.2-4.5 min against the 12-h limit, which could mean a 0 score. | **taken** | No per-branch write after the solve end (`E15B_SOLVE_END_PC`; counted in `late_writes_skipped`); the final write after the solve stays. The rule-library outputs are cached per task. |
| F2 | After a relaunch, the prefill storm (~400k tokens) makes `/health` fail for > 180 s, so the watchdog kills its own new server. | **taken** | Liveness = server-log growth first; `/health` only when the log is silent. |
| F3 | A relaunch reuses the same 16-stream command, and a failed relaunch ends the watchdog. | **taken** | After an OOM / RuntimeError in the dead log, or on the 2nd relaunch → the 12-stream flags. A failed relaunch retries while one is left. Harness `kill2`. |
| F4 | Any engine failure ends in the identity submission (~0). The give-up is at T0 + 75 min with 3 launches. | **taken (partly)** | RERUN: give-up T0 + 4 h; 4 launches (16 → 12 → diagnosed → W1); the staging join keeps its T0 + 55 min bound. **Not built:** the case where the gate-fallback 12-stream relaunch fails (that would be a double failure). |
| F5 | The W1 last-resort launch drops `--min-free-slots-delay 1`. | **rejected** | W1 sizes its buffers with `--cuda-graph-max-bs-decode 12` and mrr 13. The nodelay crash (Exp 14 Notes/30) was exactly a 13th request on buffers sized 12. W1 is safe only because its delayer stays on. |
| F6 | The engine fields are wrong after a relaunch (counter reset; only the last log is read; token max over label rows). | **taken** | All server logs are read. Tokens are summed per server (max over TP ranks, summed over the other label rows). Accept length p50 and gen tok/s p50 per running level are added. |
| F7 | Both attempts finish in the commit only at 16 streams. | **taken** | Window 78 min (with auditor 3 R4). |
| F8 | The `/health` probe can take the spare server slot. | **taken** | `/health` only when the log is silent (F2). |
| F9 | Stale docs / labels (Notes/08 missing, Notes/03 old). | **taken** | This note; Notes/03 updated. |

### Auditor 3 — regression vs past runs (135-item ledger)

**Ledger:** 91 fixed, 12 still present, 27 N/A, 4 need the GPU, 1 low not verified.

| # | Item | Verdict | Change |
|---|---|---|---|
| L26 | A read timeout loses the in-flight reply | by design | It bites only at the deadline. |
| L39 / R10 | A non-timeout request error discards the current turn | **not changed** | Vendor turn logic; it happens only when the server dies (a watchdog event). At most one turn per in-flight attempt. |
| L41 | Unknown tool names `code`, `function`, `function_calls` | **taken** | An unknown tool with a `code` string runs as python. Unit-tested. |
| L65 | Every launch fails → identity | partly | Auditor 2 F4 (4 h, 4 launches). |
| L85 / L125 / L126 | Accept length, per-server counters, per-level tok/s | **taken** | Auditor 2 F6. |
| L105 | The CPU kit was not run on the final sha | **this rev** | The Kaggle CPU kit runs on the rev 3 shas. |
| L111 / R5 | `kill()` does nothing when the main process is dead → leftover TP workers (+3.3 min per crash) | **taken** | `killpg` also for a dead main process. |
| L112 / R1 | The watchdog relaunches the same 16-stream flags | **taken** | Auditor 2 F3. |
| L116 | Duck re-sends on connection errors / 5xx | **not changed** | Rare; low. |
| L127 | Stale docs | **taken** | |
| R2 | KV peak at 16 streams (~0.90 of the pool late in the run) | GPU smoke | Read `retract_lines`, `preempt_lines` and the KV peak. If they are bad → `E12_R6_KV_ADMIT=0.85` or 14 streams. |
| R3 | The commit turn without a tool call | GPU smoke | Read `commit_with_tool_call / commit_turns`. If < ~50 % → `E12_B_RESERVE=8192`. |
| R4 | The 20-task commit window is marginal | **taken** | 78 min. |
| R6 | The 16-stream gain is unmeasured | GPU smoke | `gen_tok_s_p50_by_running`. If 16 is not faster than 12 → the 12-stream flags. |
| R7 | The isolation preflight can lock 6 workers for 12 h | **not changed** | 6 is the approved safeguard; changing it needs the user. |
| R8 | `min_admit` 300 loses ~1-2 % of tokens at the deadline | accepted | A5. |
| R9 | A blocked import aborts the reasoning harvest | **taken** | Imports are wrapped in `try`. Unit-tested. |
| R11 | The build was behind the working copy | **this rev** | Rebuilt; all tests re-run. |

## Tests of rev 3

- **Unit tests:** `test_duck_b.py`, now including the auditor regressions.
- **Item verifier:** `verify_items.py`, with items 1-55, A1-A7 and the audit fixes.
- **CPU harness modes:**
  - commit;
  - rerun;
  - eval120;
  - fault kill;
  - fault kill2 (two kills → the 2nd relaunch at 12 streams);
  - fault slow.
- **Kaggle CPU kit:** the same, on the pinned image (Py3.12).
