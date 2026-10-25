# Auditor 2: notebook wiring and engine (2026-10-25)

**Scope:** the two built E15B notebooks (`Q27-XHIGH-E15B_SUBMIT_2X35K_S16_DUCK_SGLANG.ipynb`, `..._EVAL120G4B_...`), `build_e15b.py`, the payload, and the base `Q27-XHIGH-E14S_EVAL120G4B_W6G13_DUCK_SGLANG.ipynb`. I checked the code against the local SGLang 0.5.21 source and the real Exp 14 Kaggle logs.

**Read-only.** No project file was changed. Scratch scripts are in `D:\ARC AGI 2\_scratch\e15b\agent2\`:
- `extract.py`: the cells, as `sub/`, `ev/` and `base/` `cNN.py`;
- `flags_sim.py`: the launch flags along the fallback chain;
- `bench.py`, `time_final_loop*.py`: the cost of each submission write;
- `test_logsum.py`: the server-log summary run on a real log.

**Line references:**
- "CELL 6 line N" is the line inside that cell's source (`sub/c09.py`). The same scheme holds for CELL 1 = `c02`, CELL 1b = `c03`, CELL 5 = `c07` and CELL 7 = `c10`.
- `arc2_duck.py` lines refer to the payload copy (sha `2941e8cf`, identical to `src/duck_b/arc2_duck.py`).

The two notebooks differ only in the default profile (CELL 1 line 2) and the markdown cell. Every finding applies to both.

---

## Findings (ranked by impact)

### F1. HIGH (outcome) / PLAUSIBLE (timing): after the hidden-run deadline, `submission.json` is rebuilt and rewritten once per attempt, with no clock guard. E15B makes each rewrite about 2-5x more expensive.

**Location:**
- `arc2_duck.py` 3375-3377: the final `publish(s.freeze(), final=True)` for every session.
- `arc2_duck.py` 3187-3191: `on_branch_done`.
- CELL 6 lines 59-75: `_e12_branch_done_r6`, which calls `e12_write_submission()`.
- CELL 6 lines 47-56: in RERUN, `build_submission` over every task, then an atomic `json.dump`.
- `arc2_duck.py` 2658-2662 / 2683-2687: the rule library runs again for every task on every build.

**Scenario: the hidden run (RERUN_ALL).**
1. The solve deadline is T0 + 712 min (CELL 1 line 117 budget 717 min, hard margin 2 min, `E12_STOP_BEFORE_HARD_SEC` 180 s).
2. Of those 180 s, E15B uses up to 120 s itself: the final harvest grace is 30 s and the join grace is 90 s (builder line 365).
3. After the join, the scheduler publishes about 480 final records (240 tasks x 2 attempts; the design starts nearly all of them).
4. Each publish rebuilds the whole submission (240 tasks, rule library included) and rewrites it with the pure-Python `json.dump` encoder.
5. Only after all 480 writes does `solve_all_r6` return. CELL 6's tail, CELL 7 (report, a second `build_submission`, the zip of about 480 transcripts) and CELL 7b follow.
6. Everything must end before the 12 h limit (T0 + 720 min, minus any time before the T0 anchor). A notebook that ends late scores 0.

**Evidence** (Python 3.12.10, i5-10310U, min of n runs; scripts `bench.py`, `time_final_loop*.py`):

| Cost per final publish | E14S (rules off) | E15B (`E12_B_MODE=1`) |
|---|---|---|
| `build_submission`, 240 tasks, real-size records (from the EVAL120 run) | 0.036 s | **0.186 s** (rule library) |
| `json.dump` of the submission: public-placeholder size (387 kB, = E14S output) | 0.08 s | 0.08 s |
| `json.dump`: eval-sized (1 MB) | 0.37 s | 0.37 s |
| **Per publish** | 0.12-0.41 s | **0.27-0.56 s** |
| **x 480 publishes** | (E14S had fewer: about 400) | **2.2-4.5 min** |

On Kaggle, the EVAL120 commit's final loop (199 publishes of the 387 kB identity file) took about 16 s, which matches these per-call numbers (log: deadline 21,728 s, cache diag printed at 21,819 s).

**Estimated end of the hidden run:**
- 712:00 deadline;
- about 712:40 when the workers finish their harvest and publish (16 serialized writes);
- plus 2.2-4.5 min (longer if the Kaggle CPU or the GIL is slower);
- plus about 0.5-1 min for CELL 6, 7 and 7b.

That is about **715.5-718 min**. E14S (LB 22.78) ended about 1.5-2 min earlier on the same structure, and we have no measurement of its real end margin.

Neither the harness nor the EVAL120G4B commit can show this: the commit writes the identity file only (the `RERUN` branch is skipped), and the harness rerun uses 5 tasks.

**Minimal fix** (any one of these removes most of the cost; the first removes all of it):
- In `_e12_branch_done_r6`, skip `e12_write_submission()` once the solve deadline has passed (`if time.perf_counter() < _end: e12_write_submission()`). CELL 6 already writes once after `_E12_STOP.set()` (line 634), and E12_RECORDS is still updated.
- Memoize `rule_library_outputs` per task (it is a pure function of the task).
- In `_e12_write_json_atomic`, use `f.write(json.dumps(obj))` (the C encoder): 0.013 s instead of 0.08 s for 387 kB.
- In RERUN, guard CELL 7's report and bundle with `time.perf_counter() < HARD_STOP` (the submission is already final then).

### F2. MEDIUM-HIGH / PLAUSIBLE (mechanism confirmed in the source): the watchdog kills the server it has just relaunched, during that server's re-prefill storm.

**Location:** CELL 6 lines 486-534 (builder 312-361), the `/health` failure rule (`_WD_FAIL` = 180 s).

**Scenario:**
1. The SGLang server dies mid-solve, and the watchdog relaunches it (about 2-3 min with the JIT cache).
2. `SERVER_DOWN` is cleared, and the 16 waiting attempts resend their full histories within about 2 s. The radix cache is empty after the relaunch.
3. SGLang 0.5.21 schedules prefill first (`scheduler.py:3746-3748` "Run prefill first if possible"; `prefill_decode_interval` defaults to 0 for this model, `validation_hook.py:492-502`). So no decode step runs until the whole queue is prefilled.
4. Duck requests are non-streaming (`openai_compat.py`, provider vllm), and a non-stream request sends output only every 50 tokens (`output_streamer.py`, `SGLANG_FORCE_STREAM_INTERVAL`). The tokenizer manager therefore gets no message during the storm.
5. The `/health` probe (input `[0]`) waits behind the queue (lpm with equal 0-prefix ties = FCFS). The server's handler gives up after 20 s; the watchdog's client gives up after 10 s.
6. The storm lasts the summed contexts divided by about 1,575 prefill tok/s (C_engine §3.4). Sixteen attempts at an average of about 25k tokens each is about 400k tokens, about 250 s, which is more than 180 s.
7. So `/health` fails for 180 s, and the healthy new server is killed. Relaunch 2 then repeats the storm.
8. On relaunch 2's storm the limit (2) is reached, and the watchdog returns without killing; the server survives.

**Net cost:** one wasted relaunch plus storm (about 6-8 min of all 16 streams), and the watchdog is gone for the rest of the run. E14S never polled `/health` during a solve, and the harness's fake server has no prefill storm, so the A7 fault test cannot show this.

**Minimal fix:** count server progress as liveness. In the watchdog, before counting a health failure, check whether `_srv.log_path` grew since the last poll (SGLang logs every prefill batch and every 40 decode steps); if it grew, reset `_bad` and continue. Alternatives: a grace of about 15 min after a relaunch, or `/metrics` counters that advance.

### F3. MEDIUM / PLAUSIBLE: the watchdog relaunches the identical 16-stream command after a crash, and a failed relaunch ends the watchdog.

**Location:** CELL 6 line 518: `E11Server(list(_srv.cmd), _srv.env, ...)`. Line 533: `if not _rec.get("ok"): return`.

**Scenario:** the server crashes at runtime because of something 16-stream-specific, for example a CUDA OOM at peak.
- Real evidence at 12 streams: `available_gpu_mem=1.78 GB` after init and only **0.98 GiB free at runtime**. SGLang itself warns: "Triton kernel ... device-loaded after serving started (free device mem: 0.98 GiB). Pre-load it during engine init to avoid CUDA OOM" (EVAL120 XNO log 00:11:32; C_engine §1 reports 0.80-0.98 GiB in all three runs).
- 16 streams add graphs 1..17 (about +0.3 GB) and a larger mamba pool, so runtime free memory may be about 0.5-0.7 GiB.

With the same command, the crash can repeat: both relaunches are spent, and the rest of the run talks to a dead server. Separately, if a relaunch does not become ready (for example the GPU was not freed: the result of `_e11_gpu_free()` is ignored), the watchdog returns at once, even though a second relaunch is still allowed.

**Minimal fix:**
- When the dead server's log tail has an `E11_OOM_MARKERS` match, relaunch with the 12-stream flags: `e14_edit_flags_b(cmd, {"--max-running-requests": ["13"], "--cuda-graph-bs-decode": 1..13, "--max-mamba-cache-size": ["56"]})`. The client's 16 streams then queue in the server; this is safe.
- On a failed relaunch, `continue` while fewer than 2 relaunches have been used and at least 20 min are left, instead of `return`.

### F4. MEDIUM / CONFIRMED (inherited, made worse): every SGLang-path failure ends in the identity submission, the give-up is fixed at T0 + 75 min in a 12 h run, and the new 16→12 step uses one of the 3 launches.

**Location:**
- CELL 1 line 224: `SGL_GIVE_UP_AT = min(T0 + 75 min, ADMIT - 5 min)`.
- CELL 5 lines 652-677: `for i in range(3)`.
- CELL 5 lines 245-249: `e11_next`, where any first failure goes to `streams12`.
- CELL 5 lines 683-699: the gate fallback, whose failure raises.
- CELL 2 line 104: a failed tokenizer check sets `ENGINE_BACKEND = "vllm"`.

**Evidence:** no vLLM engine code is left in this notebook (no `from vllm`, no LLM engine; `E11_ALLOW_VLLM_FALLBACK` is dead config). So every one of the following ends at CELL 0's identity file (about 0 LB):
- `E11Fail`;
- a capacity-gate failure;
- a failed tokenizer check;
- a failed relaunch after the gate fallback.

The give-up of T0 + 75 min came from the time when vLLM was the fallback. In a 717-min run, a launch that is READY at T0 + 120 min would still have about 9.5 h of solving.

The order of the chain is now:

| Launch | Configuration |
|---|---|
| try0 | 16 streams |
| try1 | always `streams12`, whatever the cause |
| try2 | the first diagnosed fix (oom / lazy / prefill), else W1 |

So two diagnosed causes (for example OOM, then a lazy-buffer error) now exhaust the 3 launches (`flags_sim.py`). In E14S they had a launch each.

**Minimal fix (RERUN only):**
- Give-up at about T0 + 4 h (any solving beats identity).
- `range(4)` launches.
- On a diagnosed cause, apply the fix together with `streams12` in one relaunch.
- When the gate-fallback relaunch fails, continue into the `e11_next` chain instead of raising.

### F5. LOW / CONFIRMED: the W1 last-resort launch turns the DFlash delayer back on.

**Location:** CELL 5 line 204: `if E14_DO_MEASURE or cfg.get("w1"): return f`. This returns the plain W1 flags without `--min-free-slots-delay 1`.

**Evidence:** `flags_sim.py`, retry3 w1 gives width 8, mrr 13, `--cuda-graph-max-bs-decode 12`, and no `--min-free-slots-delay`. `resolve_min_free_slots` (`min_free_slots_delayer.py`) then gives 3 for DFLASH at mrr 13, which is the measured 11-of-12 running (C_engine §2).

**Minimal fix:** append `["--min-free-slots-delay", "1"]` in the w1 return path.

### F6. LOW / CONFIRMED: the `b_summary` engine fields are wrong after a watchdog relaunch.

**Location:** CELL 6 lines 91-98, 478, 596-597, 601.

**Evidence:**
- `server_gen_tokens` is the new server's counter (which restarts at 0) minus the old server's starting value, so it is wrong or negative after any relaunch.
- `server_log` summarizes `E11_SERVER.log_text()` only, so the crashed server's OOM, Traceback and retract lines (the reason for the relaunch) are not counted. They are still bundled by CELL 7b (`sglang_server_*.log`).
- `_e15b_gen_tokens_total` takes the max over rows that differ by the `is_streaming` label (`metrics_collector.py:1558-1561`), not over TP ranks as its comment says. It also counts finished requests only.

**Minimal fix:**
- Keep a per-server baseline, and sum the deltas over every server of the solve.
- Run `_e15b_server_log_summary` over every `OUT/sglang_server_*.log` started after the solve began.
- Sum across the label rows.

### F7. LOW / PLAUSIBLE: the commit profile's goal ("BOTH attempts of 20 tasks finish") holds at 16 streams only.

**Location:** CELL 1 lines 99-105.

**At 16 streams** (per stream about 26-30 tok/s, from the measured step time of about 5.2 ms per added request, C_engine §3.1): a full 35k attempt takes about 21-23 min with tool time. The 40 attempts run in three waves of 16, 16 and 8, about 23 + 23 + 13-15 = **59-61 min of the 65-min window**. This is tight but fits.

**After the 12-stream fallback:** there are 4 waves, about 75 min, so the last attempt-2s are cut at the deadline. That is 4 tasks or more if the waves stay aligned.

**Minimal fix:** none needed for safety. If the goal matters, add about 10 min to the window when `SGL_MAX_RUNNING == 12` after bring-up. The hard stop leaves room: the solve ends by T0 + 95 at the latest.

### F8. LOW / PLAUSIBLE: the `/health` probe takes the spare server slot.

**Location:** CELL 6 line 494.

`/health` is a real 1-token generation (`http_server.py:699-777`), sent every 10-20 s for about 11.6 h. With 16 client streams and mrr 17, a probe can occupy the "+1" slot that exists for the chunked-prefill double count (CELL 1 lines 218-221). It can briefly park a Duck request; the A1 `queue>0` fields would show it.

**Fix:** with F2's log-growth rule, call `/health` only when the log has been silent (for example 60 s).

### F9. LOW / CONFIRMED: documentation and report labels.

- The builder line 134 and CELL 1 line 95 cite "Exp 15 Notes/08", which does not exist. Notes/03 still describes the old 45-min, 16-task smoke.
- `CONFIG["e12"]["E12_LIVE_BRANCHES"]` = 12 is reported, but `solve_all_r6` never uses it. It is misleading next to 16 streams.
- `CONFIG["e11"]` is printed with 16/17 before any fallback, and nothing updates it after one. The backend label `-s12` and `caps` are correct.

**Fix:** correct the reference; drop or annotate the unused field.

---

## Time math

### Commit E15B (100-min budget, HARD_STOP T0 + 98, solve end at most T0 + 95)

| Step | Time |
|---|---|
| READY (past runs 707-771 s, C_engine §1) | 12-13 min |
| Graphs 1..17 (the 6-min verify-graph capture is one JIT compile on the first, largest batch size; the other 12 batch sizes took 12 s together, EVAL120 log 00:03:53 → 00:10:09 → 00:10:21) | + about 5 s |
| Smoke + preflight | about 1 min |
| Solve start | about T0 + 14 |
| Solve end (65-min window) | T0 + 79 |
| Harvest + join | ≤ 2 min |
| Cache diag | ≤ 4 min |
| CELL 7 | < 1 min |
| **Done** | **about T0 + 86** |

- With a 16→12 relaunch (+8-10 min): the window still starts by about T0 + 25 and ends about T0 + 90. This fits.
- Staging deadline: `max(T0 + 10, min(T0 + 45, give-up − 20 min))` = T0 + 45, and the join waits until T0 + 55. This is consistent.

### Hidden RERUN_ALL (budget 717 min)

| Step | Time |
|---|---|
| READY | about 12-13 min (about 20 min on a slow day; first launch allowed until T0 + 75 under the stall rule) |
| Solve | about T0 + 14..22 → **712** |
| Post-deadline work | see F1: about 3.5-6 min, against an 8-min outer margin minus the pre-T0 offset |

**Capacity:** 480 x 35k = 16.8 M tokens. The supply is 16 streams x about 27 tok/s x about 41,500 s = about 17.9 M, before tool time. So the last attempt-2s may be cut by the deadline. That is the design (attempt 1 first), not a bug.

---

## Checked, no issue

**Delayer:**
- `--min-free-slots-delay 1` disables it: `resolve_min_free_slots` returns None for a user value ≤ 1 (`min_free_slots_delayer.py`, `scheduler.py:1183-1193`). The flag is applied in try0, streams12 and lazy.
- No `--enable-prefill-delayer` (asserted).

**Flags** (`flags_sim.py`; no duplicated flag in any configuration):

| Launch | Flags |
|---|---|
| try0 | mrr 17, graphs 1..17, mamba 68, width 6, `extra_buffer_lazy`, mem 0.83; `--cuda-graph-max-bs-decode` dropped |
| streams12 | mrr 13, graphs 1..13, mamba 56; the globals become 12/13 before the capacity gate and CELL 6's `streams = min(CONCURRENCY, 16)` |
| lazy | `extra_buffer`, mamba 5 x 13 = 65 |

**Mamba cap:** 68 // 4 (lazy + overlap = base 3 + 1, `kv_cache_configurator.py:2232-2261, 181-184`) = 17 ≥ mrr 17, so mrr is not capped. 56 // 4 = 14 ≥ 13.

**Capacity gate at 16:**
- KV estimate 850-865k tokens (917,816 at mamba 56 minus about 0.55-0.68 GB of extra mamba state). 850k // 50k = 17, so the concurrency is 16.
- Verify-graph batch sizes ≤ 16 give 16.
- Headroom: about 1.75 GB estimated (2.09 GB at 13 minus the larger graphs) ≥ 1.0 GB.

**KV fit:** in the EVAL120 run, the context of branches with ≥ 30k generated tokens was p50 36k / p90 46k / max 51k. 16 x 46k = 736k, which is below the pool. A fixed window of 100k plus a request cap of ≤ 30.9k is below `SGL_CONTEXT_LEN` 155,648.

**Request timeout 3,000 s:** a 30.9k-token request at about 27-30 tok/s takes about 1,000-1,150 s, so the margin is about 2.7x.

**CELL 1: every setting is applied once and is consistent:**
- `E15B` block plus the assert (attempt budget = session cap = 35,000);
- `E12_R6` keys exported to env:
  - first exposure 0 → None;
  - B1 view facts;
  - min admit 300;
  - fair share 0;
  - p90 cutoff 0;
- request timeout 3,000;
- `ARC3_MAX_ACTIVE_STREAMS` set in CELL 6 before the priority gate is built (built on first use, `tool_agent.py:2141-2151`);
- workers 16 on an isolation PASS;
- `E12_DUCK_CONTEXT_WINDOW` set before any agent is created.

**Install timeouts:** venv 300 s, pip 1,800 s, sysconfig 120 s. A timeout raises into CELL 5's `except`, so there is no hang.

**Watchdog (apart from F2, F3, F6 and F8):**
- `/health` is a decode-progress check (`http_server.py:699-777`); a busy server in steady state passes. The longest silence at 12 streams was 41-57 s, inside stage-2 resumes, which do not exist in config B.
- A dead process is seen within ≤ 10 s; Duck's 3 x 5 s retries mean at most about 1 failure is counted before `SERVER_DOWN`.
- Waits are bounded by the deadline; the step is retried with nothing counted (`arc2_duck.py` 2865-2876).
- A relaunch happens only with ≥ 20 min left, and its ready wait ≤ left − 10 min, so no relaunch overlaps the solve end.
- Exceptions are caught; `SERVER_DOWN` is cleared in `finally`.
- The thread is a daemon and exits on `_E12_STOP`.
- `kill()` uses the process group (`start_new_session=True`); `_e11_gpu_free` falls back to `pkill`.

**Report code** (CELL 6 `b_summary`, `server_log`; CELL 7 A2) is wrapped in try/except, runs after the solve, and cannot touch `submission.json`. `_e15b_server_log_summary` on the real EVAL120 log matches the C_engine numbers (`test_logsum.py`):
- running mean 11.11;
- queue>0 share 0.821;
- capture 387.98 s;
- KV 917,816;
- retract / preempt / OOM / Traceback all 0.

**CELL 7:**
- The scoring is unchanged. In A2, a task counts for an attempt only if it is right on every test input. `_f` is evaluated inside its own loop iteration, so there is no late-binding bug.
- Dual mode: the eval block is skipped when `SOLUTIONS is None`.
- The final `e12_write_submission()` is the same as CELL 6's.

**Submission:**
- CELL 0's identity fallback covers every task and every test input, with both attempts.
- RERUN rows replace a task only when the number of test inputs matches.
- Writes use a unique temp file + `os.replace` under a lock.
- Straggler publishes after `closed` are dropped (`arc2_duck.py` 3178).

**Python 3.12:**
- Every code cell, the Duck package and the E14 modules compile under 3.12.10 (the builder compiled them under 3.14).
- No `Thread._stop` shadowing.

**Kaggle limits:**
- Size: 254,737 B (SUBMIT) and 254,717 B (EVAL120G4B), both < 400 kB.
- No secrets or URLs.
- The payload .bin shas (`4831ed01`, `55c1eb79`) match CELL 1b / CELL 3. The manifest has 3 files changed; the zipped `arc2_duck.py` equals `src/duck_b`.

**Profile task list:** 20 unique ids, all in the evaluation set, 32 test inputs.

**`verify_items.py`:** "ALL CODE ITEMS PRESENT" (A7 = NOTE).

**Residual risk (inherited, not new):** an isolation preflight that is INCONCLUSIVE twice drops the whole run to 6 workers (C6-2, by design).
