## E31 four-build comparison: OURS vs G1 / G2 / G3 (read-only; no build files edited)

**Recommendation:** keep OURS as the base. It is the only build that passed every cross-test except one minor cap deviation. Of the GPT builds, G2 is the cleanest. G1 has severe solver bugs.

There are two things in OURS to fix before the GPU commit:
- **Must fix:** a phase-1 request error goes straight to forced close with whatever thinking exists, even zero tokens.
- **Should fix:** the phase-1 cap uses `answer` instead of `answer × N`.

### Test results (each run on a fresh copy)
| Build | Own tests | Under `python -I test_e31_direct.py` |
|---|---|---|
| OURS `e31_unit_tests.py` (run as is) | **80/80 PASS** | — |
| G1 | 8/8 groups PASS when the build dir is put on sys.path | **FAIL**: `ModuleNotFoundError: e31_direct` (-I drops the script dir) |
| G2 | 10/10 PASS (includes the donor E28 `e21_cell7` scorer, eval 0/6 by design) when the build dir is put on sys.path | **FAIL**, same import error |
| G3 | **PASS, 52 assertions** (its test inserts its own path) | PASS |

G1's own tests never exercise a natural `</think>` close, which is why its main bug below went unnoticed.

### Cross-test harness
Location: `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\cmp_harness\` (`harness_common.py`, `harness_static.py`, `harness_dyn.py`; outputs `static_out.json`, `dyn_all.json`, `dyn_down.json`).

Setup:
- the real klee100 tokenizer fixture and the real 30 eval tasks and solutions;
- the ORDER2 functions extracted from the E10 notebook;
- a fake streaming token-id `/v1/completions` server on port 8001;
- the E28 `_e12_branch_done_r6` record store, followed by an E28-style `E12A.build_submission(CH, records)` call.

Scenarios:
- **MIX:** real budgets 104k/8192, 24 workers, 8 behaviours.
- **ORDER:** 1 worker.
- **DRAIN / HARDSTOP:** compressed clock, 1 s per minute.
- **ERROR:** phase-1 kill, HTTP 500, phase-2 kill.
- **DOWN:** server down for 3 s, `/health` 503.
- **URL:** wrong port 8011.

### Component table
| Component | OURS | G1 | G2 | G3 |
|---|---|---|---|---|
| Prompt, single test | correct, 15/15 byte-equal to ORDER2 | correct, 15/15 | correct, 15/15 | correct, 15/15 |
| Prompt, multi test | correct: FINAL INPUT k, every ```grid k block shown, "Output nothing after the last block." | correct | correct | **bug (minor):** a literal `...` line between blocks; for 3 tests `grid 2` is never shown (`e31_direct.py:158`) |
| Multi-test parser | correct on all 25 cases; also accepts `grid1` as a label | correct | correct, **better:** also reads `grid K:` and a `grid K:` line just before the fence; positional only when there are exactly N fences in total | correct |
| make_attempts | correct (5/5) | correct | correct | correct |
| Adapter (DIRECT wrong + CPU rule / Duck b1 right) | correct: stays wrong; no record → identity (ORDER2) | correct; no record → input + mirror | correct; input + mirror | correct; input + mirror (reads its own state, not E28's records) |
| Phase 1 / phase 2 / ids | correct: suffix token-exact, EOS stripped, continue mode, phase-2 cap and seeds right. **Bug (minor):** phase-1 cap = min(think, 131072 − P − answer − 256), not answer × N: b6f77b65 sent 104,000 vs design 102,234 (`e31_direct.py:563`) | **3 bugs, severe**, see G1 bugs 1–3 | correct | correct. **Deviation:** seed = SEED + 10 × ORDER2 position, not the sorted index (39 mismatches; record-only) |
| Drain | correct: 24 aborted, 24 forced-close requests, all gold, returned 1.4 s before T | correct in the test | correct | correct |
| Hard stop (T − 1) | correct: partial grid used 24/24, no orphan stream | correct | correct | correct |
| Early return | correct, 3.7 s | correct | correct | correct |
| Client / streaming | correct: socket shutdown on abort, no idle timeout, every sampling field explicit | **risk:** `response.close()` with no socket shutdown (a cancel race not reproduced on Windows); urlopen timeout **180 s** | correct: shutdown, no timeout; sends T / top_p / top_k only (the server's override values equal ORDER2's) | shutdown watcher thread; urlopen timeout **120 s** (risk) |
| Error handling | best: admission gated on `SERVER_DOWN` and `/health`; recovered 15/15 error tasks. **Bug:** a phase-1 error → forced close immediately. On HTTP 500 that is a forced close with **zero** thinking (2 requests per task); on a kill it keeps only what had streamed | **bug, high:** no wait, no retry; 15/15 error tasks lost; DOWN scenario: **all 30 tasks burned in 3 s** | same as G1: 15/15 lost, 30/30 burned | same as G1, and the ids of the failed request are lost (`stream_ids` raises before returning ids) |
| Records / SCHED_LAST | correct: full contract, `SCHED_LAST` updated in place with Duck keys | **bug:** records lack `t0` / `t1` (30/30); `publish_seq` always 1; skipped tasks get no callback | correct; stragglers is a list; returns the report, not records (E28 ignores the return value) | correct, minimal record; `SCHED_LAST` has no gauges; returns the report |
| Report fields | best: per group main / partial / control / multi / single (hits, fully solved, task-norm), coverage 60k / 78k / 104k, speed (server counter deltas, TTFT, per-request tok/s), E28 Duck reference and per-task comparison | basic; comparison comes from cfg, empty | reference table; coverage uses 77,824, not 78k | basic, no reference |
| Thinking save | correct, 30/30 | 30/30 | 30/30 | 30/30, but lost on a stream error |
| CELL 1 / budget | drain 23; budget 205 follows `E31_CFG`; deadman follows the budget; RERUN_PROD = DIRECT | drain 26; deadman literal 12300 (does not follow cfg); `E31_CFG` grows to several lines; rerun = E28 Duck unchanged | drain 24; budget 205 fixed; `E12_BRANCHES=1`; `quality_ab` False; rerun = Duck **without** `e28_sched` (not the proven E28) | drain 24; budget 205; rerun = E28 unchanged; raises if the window is too short |
| Notebook size | 394,400 B, self-contained | 398,556 B + a separate Kaggle dataset | **388,767 B, self-contained** | 398,819 B + a dataset found by globbing `/kaggle/input/**` (slow on GCSFUSE; fails if more than one copy is found) |
| Workers | min(E28 `_workers`, cfg), so Duck's isolation fail-safe (6 workers) would also cap DIRECT | respects the 6-worker fail-safe | **better:** ignores the Duck sandbox floor (uses cfg 24) | min(cfg, arg) |
| Port check | in CELL 7b (unit test T6) | asserts `:8001` and that `base_url` matches (by code; my harness call itself failed) | refuses 8011 | refuses 8011 |
| Tests | 80/80; plus the real CELL 7 scorer | 8/8 | 10/10, including the real E28 `e21_cell7` | 52/52 |

### G1 bugs (severe; do not use its solver)
1. **Natural finish still gets a phase 2.** `e31_direct.py:404-427` always sends a "continue" request, then `:428` parses only p2. In MIX, 18 natural tasks got an extra request and 22 queries came out wrong, including gold answers.
2. **No EOS strip before a forced close** (`:408`). The suffix contains 248046 before the STOP_IDS (4/4 eos tasks).
3. **Parses the full decode, not the text after the last `</think>`** (`:428`, `skip_special_tokens=False`). A grid inside the thinking can be taken.
4. No `t0` / `t1` in records, plus the error cascade and the timeout described in the table.

### G2 / G3 bugs
- **Error cascade (both):** no wait or retry on a request error or a dead server. A relaunch window turns every admitted task into an error.
- **Stream timeouts:** G3 120 s, G1 180 s. A preempted request waiting longer becomes an error.
- **G3 multi prompt** (`:158`): the `...` line.
- **G3 tie order:** depends on E28's `E12_ORDER` (with `plain_order` False, ties go by cell count, not sorted id).
- **G2 rerun:** runs Duck without `e28_sched`.

### MERGE LIST: base = OURS
**Must fix before GPU (OURS bug):**
1. **Phase-1 error must not force-close.** Today, when `pending == "p1"` and the server recovers, `start_phase2(resend=False)` runs (`e31_direct.py:774-778`).
   - Fix: if time < DRAIN, resend phase 1 with prompt ids + kept p1 ids (EOS stripped), cap = P1_CAP − len(p1_ids), same seed. Forced close only at or after DRAIN.
   - At the very least, a phase-1 error with 0 tokens must resend phase 1.
   - Without this, an E15B relaunch truncates the thinking of all 24 live tasks, and an HTTP 500 gives a forced close with zero thinking.

**Should fix:**
2. **Phase-1 cap.** `phase1_cap(len(ids), think, answer * NTEST[tid], max_model_len)` (`e31_direct.py:563`). This is the design formula; G1, G2 and G3 all do it. Only b6f77b65 changes, to 102,234.
3. **Workers.** In CELL 7b, use `_w = min(int(E31_CFG["workers"]), int(E12_RUN["streams"]))` instead of E28's isolation-gated `workers` (taken from G2). DIRECT runs no Python sandbox, so its 6-stream fail-safe would only cost coverage.
4. **Fail closed on the stop ids.** In production, assert `STOP_IDS == FN_STOP_IDS` and `THINK_CLOSE_ID == 248069` before any request (taken from G2 / G3). OURS currently only records `stop_ids_equal_spec`.

**Optional grafts:**
5. G2's label extension: `grid K:` in the info string and a `grid K:` line just before the fence. It only matters when blocks come out of order.
6. A task with no record → `make_attempts(input, None)` = input + mirror instead of the identity (G1 / G2 / G3). This is a free extra guess, but it departs from ORDER2's identity.
7. `drain_min` 24–26 instead of 23. 3 × 8192 / 18 tok/s is 22.76 min, so 23 leaves 0.24 min of margin. This is an `E31_CFG` value, no rebuild needed.
8. `E12_BRANCHES = 1` for the E31 profile (G2), so the E28 report's "branches done x/N" counts against 30.
9. A per-record `phase2_prefix_sha256` (G1), as an audit field.

**Do not graft:**
- the separate dataset module (G1, G3);
- the 120 / 180 s stream timeouts;
- the GPT error handling.

**For your decision:** what the hidden rerun runs.
- OURS: DIRECT on every test task, per the design.
- G1 / G3: the unchanged E28 Duck K100.
- G2: Duck without `e28_sched`.

### Other observations (all builds, not bugs)
- At drain, each build drops the ids still in flight (24/24 phase-2 prefixes are a shorter exact prefix of what was sent).
- In the compressed hard-stop test, OURS returned 0.01 s before T, because it writes the transcripts synchronously. The real margin is 60 s, so the risk is low.
