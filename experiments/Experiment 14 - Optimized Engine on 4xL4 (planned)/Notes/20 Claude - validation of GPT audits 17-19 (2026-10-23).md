# 20 — Validation of the three GPT audits of E14-A `a9dad9fc` (Claude, 2026-10-23)

**Inputs:**
- Notes/17 (audit 1), Notes/18 (audit 2), Notes/19 (audit 3). Each is saved verbatim from the user's paste.
- All three say **NO-GO for the exact SHA `a9dad9fc`**, with targeted fixes and no redesign.

**Method:**
- Every claim was checked against our code and against the SGLang 0.5.21 source (`_scratch/session_1c6db7aa/sgl_src`).
- Claims marked "reproduced" were run on this machine.
- **No code has been changed yet** (rule: validate, record, change nothing until the user says so).

**Result:**
- **Every finding is true.** None was found false.
- One finding, audit 2's model-staging collision, is a **real P0 that would have broken the GPU run at launch**. The fake-server harness could not catch it, because its fake paths had different basenames.

## 1. Claim-by-claim verdicts

| ID (audit) | Claim | Verdict | Evidence |
|---|---|---|---|
| **A2-P0 staging collision** (18) | Both Kaggle model dirs end in `/1`; `parallel_copy` uses `dst_root / src.name`, so the draft copy deletes the target copy and both map to the same dir | **TRUE — reproduced** | `stage_models([…/int4/1, …/dflash/1])` mapped both to `dst/e14-weights/1`, which held `draft.bin` + `{"m":"draft"}` with `error: None`. Real paths: CELL 1 `…/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1` and `…/qwen3-8-27b-dflash2/pytorch/bf16/1`. On GPU, SGLang would load the drafter as the target. |
| **Duck 25-min reserve** (17 HIGH-2, 18 P0, 19 HIGH-E) | `e14_phase` only keeps HARD_STOP − 8 min; A1 can eat the Duck window | **TRUE** (code) | `e14_phase`: `left = HARD_STOP - now - 8*60`; `e14_launch`: only the "< 15 min before hard stop" check. CELL 6 `_end = min(HARD_STOP - 180, now + 1500)`. A slow first launch (up to the 75-min give-up) + A1 caps (69 min) + relaunches can leave the Duck window a few minutes long. |
| **NCCL graph_correct** (18 P0) | `build_summary` candidates ignore `graph_correct` and `status` | **TRUE** (code) | `e14_nccl_bench.py:459`: `cands = {… if v["graph_per_call_us"] is not None and v["correct"] is True}`. |
| **Per-arm correctness gate** (17, 18 M5, 19 P0-A) | #44401 / nodelay / graphs13 are adopted on speed + latch only; parity runs only on W1 | **TRUE** (code) | CELL 5b: `_keep = bool(_ok and _pp and _speed["adopt"] and _better)`; nodelay / graphs13 likewise. A1-4 parity ran only on W1. |
| **Zero-gain adoption** (18 E14-9, 19 P0-C) | Equal speed and zero latch shares adopt an arm | **TRUE — seen in harness run 5** | Run 5 decisions: p44401, nodelay and graphs13 all `adopt: true` at speed ratio 1.000 (fake server). `adopt_speed` with `min_gain=0` accepts down to −2 %; the latch rule accepts ties. |
| **Latch metric misses 11 + queue** (17 §5, 19 P0-C) | `want = min(14, 13) = 13`; `share_stuck_one_below` counts running == 12 with queue > 0, not the real pathology 11 + queue | **TRUE** (code) | `latch_test`. Worse: a nodelay arm that goes from 11 to 12 running + queue would raise `stuck_one_below` and could be **rejected**. My earlier note (Notes/16 §4b) said the rule was fine; that was wrong in this case. |
| **parity_ok fails open** (17 HIGH-1, 18 E14-6, 19 HIGH-D) | PASS with compared = 1 of 12; PASS with 0/6 identical when the A/A is also 0; missing margins count as "no flip" | **TRUE** (code; the GPTs reproduced it) | `parity_ok` checks `compared` only for non-zero, no expected count; in `compare_passes` the margin `m = None` gives no flip. Margins do exist under DFlash in 0.5.21 (`compute_spec_logprobs` with top-k, `meta_info["output_top_logprobs"]` tokenizer_manager.py:2787), so missing margins mean an error case that must fail closed. |
| **adopt_speed validity** (17 §6.1, 19 HIGH-D) | No check of measured level, context, line count | **TRUE** (code) | `adopt_speed` reads only `tok_s_p50`. The bands use the same prompts, so context is matched by construction in normal runs, but nothing enforces it or flags a bad window. |
| **post_json past the deadline** (17 HIGH-3, 18 E14-4) | With `ctx.left() <= 0` a request is still sent with a 5 s timeout | **TRUE** (code) | `timeout = max(5.0, min(timeout, ctx.left() + 5.0))`. |
| **Cumulative cleanup** (18 E14-5, 19 P0-B) | Per-thread `join(60)` after abort: 12 → 720 s, 14 → 840 s; aborts are sequential | **TRUE** (code; worst case on a hung server) | `steady_window` / `latch_test` `g.join(60)` per generator; `agent_replay` per-thread join; `abort()` serial with 30 s each. |
| **flush failure not classified** (17 HIGH-4, 18 E14-10, 19 §3) | "Flush cache failed." becomes `{"text": …}` and the Duck runs anyway | **TRUE** (code) | CELL 5b stores the reply only. The real endpoint refuses while requests run (http_server.py:987-1000). |
| **int8 cross-arm parity** (17 §6.2, 18 E14-7, 19 §4) | Cold vs cached within each arm only | **TRUE** (code) | `_cache_hit_parity` compares two passes of the same arm. |
| **A2 controls from old W1** (18 E14-8) | Width / target-only compare against `E14_PAR_D` from W1, while A2 arms run on the adopted arm | **TRUE** (code) | `E14_PAR_D` = A1-4 serial on W1; A2 launches use `_base_arm` (adopted set). |
| **graphs13 unparsed list fails open** (17, 18) | Missing capture list → speed decides | **TRUE** (by design) | `_gcov = not _gcap or …`. The caps parser reads real E11 logs, so fail-closed is safe. |
| **Staging keeps copying after fallback** (17 LOW) | If the join times out, the copy thread keeps reading GCSFUSE while the server loads from it | **TRUE** (code) | `Staging` has no stop; CELL 5 hook joins with a timeout and then launches from the original paths. |
| **full_reply_text_match off-by-one** (18 E14-11) | Last-token divergence counts as a match | **TRUE** (code) | `k >= len(exp) - 1`. It was meant to tolerate the stop token; it must become an explicit stop-token rule. |
| **Tests not portable** (17, 18 E14-12, 19 §7) | Patches test uses `D:/ARC AGI 2/...` REF; mounts test patches `os.name = "nt"`; needs PYTHONPATH | **TRUE** | `test_e14_sglang_patches.py:25`; `test_e14_loadpath.py:447-448`. Not a runtime bug. |
| Width 6/4 (Q5, all) | A runtime override is supported; record the warning, acceptance and max accept ≤ width; matched W8 control | Agree | `set_block_size()`; `is_causal: false`. |
| Quota order (Q6) | Drop int8 and width 4 first if quota is short | Agree | All three agree. |

**Claims I could not check here:** public GitHub issue #33783 (flush with paused requests), #40144 (DFlash layout) and the "PR #44401 merged" note. They are cited by the GPTs as context and are not needed for any fix.

## 2. Consolidated patch set (proposed; not built yet)

**Must (P0 / HIGH):**
1. **M1 staging:** a unique destination per model, `<role>-<sha1(src path)[:8]>`, e.g. `target-…` / `draft-…`. `stage_models` asserts that no two sources map to one destination. Test: two sources both named `1`, so the dirs, files and checksums stay separate.
2. **M2 Duck reserve:**
   - A1 phases and A1 relaunches start only if `HARD_STOP − now − (1500 + 180 + preflight 300 + report 300)` ≥ the phase's minimum useful time (and a launch-cost allowance for relaunches).
   - Otherwise skip, with the reason recorded.
   - If the window cannot be given, the Duck result is labelled `solver_window_unavailable` / the real window seconds.
   - Test with a fake clock (early / late start).
3. **M3 deadlines:**
   - `post_json` raises `TimeoutError` when `left <= 0`, and timeout = min(given, left) with no 5 s floor;
   - cleanup uses one shared absolute cutoff (≈ 20 s): concurrent aborts, one total join budget;
   - if requests are still running on the server after cleanup, the next arm relaunches the server (no measurement on a dirty server).
   - Test: 14 hung generators, aborts that fail → cleanup ≤ 20 s.
4. **M4 NCCL:** candidates need `status == ok`, `correct`, `graph_correct is True`, per-call present. Test: eager OK + graph wrong + 90 % faster → rejected.
5. **M5 correctness:**
   - `parity_ok` fails closed: expected count (all prompts valid), errors = 0, missing margins = 0, A/A whole-sequence identity 0 → INCONCLUSIVE (no adoption).
   - Every candidate arm (#44401, nodelay, graphs13) gets a short gate before adoption: serial 6 × 256 vs the baseline serial (exact identity expected at bs 1, else large-margin check) + concurrent-12 × 256 vs the baseline c12 A/A (complete, 0 large-margin flips).
   - No gate time → no adoption.
   - Test: fake "faster but wrong" arm → rejected.
6. **M6 latch:**
   - new metrics `share_running_ge12` and `share_stuck11_queued` (`share_at_13` kept as diagnostic);
   - nodelay / #44401 are adopted only with a real gain: ge12 up, stuck11 down, no new log signs, speed not worse at the same band.
   - Test: synthetic 11 + queue baseline vs 12-running candidate → improvement; identical → no adoption.

**Same patch (MEDIUM):**

7. `adopt_speed` validity: require `lines` ≥ a minimum, `real_running_reached` / `measured_level` recorded, and `ctx_per_req_p50` within ±25 %, else INCONCLUSIVE. The nodelay 11→12 case is labelled an occupancy-gain comparison.
8. `/flush_cache`:
   - bounded wait for running = queue = 0;
   - classify the reply (`Cache flushed.` → ok);
   - on failure record `cold_start_unverified` (the Duck still runs; no restart just for this);
   - log the first Duck requests' cached tokens as the cold check.
9. A2:
   - a W8 serial + c12 control on the A2 base launch for width / target-only;
   - int8 cross-arm parity (default cold vs int8 cold, same ids).

**LOW (same patch, cheap):**

10. graphs13 fails closed on an unparsed list.
11. A staging stop event, set when the launch falls back to the original paths.
12. An explicit stop-token rule in `full_reply_text_match`.
13. Portable tests: `E14_SGL_REF` env for the patches fixture (skip when absent), the mounts test made portable, a conftest-free `PYTHONPATH=src` note.

**Unchanged (all three audits agree):** SGLang 0.5.21, TP4, RedHat INT4, DFlash2, rev 6g solver, host verifier, raw-reasoning switch, sandbox, output caps, backports, no new scheduler/HiCache/TP2.

**New fake-server scenarios** (audit 3):
- a faster-but-wrong candidate;
- a hung abort with 14 active requests;
- an 11-running / queued baseline with a 12-running nodelay candidate;
- plus the basename-`1` staging case.

## 3. What this says about my process

- The staging bug shows the same class as before: **the CPU harness used fixture names (`int4`, `dflash`) that differ from the real Kaggle paths**.
- New rule for the harness: the fixture directory names must copy the real Kaggle path shape (`…/pytorch/<variant>/1`).

---

## 4. Re-read addendum (user: "আবার পড়ে আসো 3 ta deeply, I think you miss something")

All three audits were re-read line by line. **My first validation kept the big findings but dropped:**
- many of the proposed tests;
- the alternative fixes;
- the audits' conflicts;
- the acceptance conditions;
- one finding entirely (the profiler-window semantics).

Items marked **M** were missed entirely; items marked **P** were only partly captured. Section references give the audit (17/18/19) and its section.

### 4.1 Correctness gate (parity)
1. **P (17 HIGH-1):** at a divergence, inspect **both** runs' top-2 margins (baseline and candidate). Today only run a's margin is used.
2. **M (17 HIGH-1):** the required gate test set:
   - stable A/A;
   - unstable A/A 0/12;
   - missing margins;
   - missing requests;
   - a large-margin flip;
   - a low-margin benign divergence (must not fail).
3. **P (18 E14-6):** use the same prompt indices and the same sample count. Today the serial `parity_pass` can `break` when time runs low, and `compare_passes` silently zips fewer rows.
4. **P (18 E14-6):** a large-margin divergence that also appears in the A/A is numeric variation, not corruption. So the arm's flips must be judged against the A/A's flips, not against zero alone.
5. **M (18 Q4):** add a **known-answer test** (fixed prompt → expected greedy prefix recorded on the baseline) as an admission condition, next to parity.
6. **M (19 P0-A):** if the gate fails, **revert to the previous verified arm**. Use one small shared probe suite for all arms; no 512 × 12 × repeats per arm.
7. **M (my own, from re-reading 19 §6 "transport safety evidence"):** the **NCCL env arm** has a speed rule (≥ 5 %, builder lines 602-604) but **no output parity**. A different transport can change the reduction order. Add it to the same gate.
8. **M (17 §13):** the harness has **never exercised the patch-revert path**. #44401 was adopted in every fake run, so `E14P.revert` → `W1_back` never ran. It must run in a new scenario.
9. **Process (17 HIGH-1, 18 M5, 19 P0-A):** my README said "adopts only those that pass a speed rule and a correctness gate". **That was not true** for #44401 / nodelay / graphs13 / NCCL. My claim was wrong, not just the code.

### 4.2 Deadlines and cleanup
10. **M (17 HIGH-3):** after the bounded aborts, **check the server's active-request count**. If requests are still running, do a controlled server termination/relaunch.
    - Required tests: deadline before the call; deadline during the response; 12 outstanding requests; abort error; server shutdown fallback; the final hard-stop boundary.
11. **M (18 E14-5):** a **carry-over check**: no request alive from one phase may reach the next phase's window. Record it, and invalidate the window if it happens.
12. **M (19 HIGH-E):** the Duck reserve must use a **conservative relaunch cost** (not the warm median) and a **final artifact margin of about 10 min** (CELL 7 + cache diag + 7b).

### 4.3 Latch / nodelay / speed
13. **M (18 Q2):** extra latch metrics:
    - max simultaneous running;
    - queue-drain latency;
    - retraction count;
    - committed tok/s at the same offered load.
14. **M (18 Q2):** a held-row correctness test for #44401 kept separate. Only the source-hash guard and the CPU patch tests exist; a scheduler-level test is not possible on CPU (recorded as a limit).
15. **M (19 §5):** record each arm's contribution and adoption reason separately, and weigh cache cost.
16. **M (17 §6.1):** mandatory per-window metadata in the report: `measured_level`, `ctx_per_req_p50`, `window_sec`, `real_running_share`.
17. **M (18 nice-to-have):** **profiler-window duration semantics.**
    - In `steady_window(..., during=profile_steps)`, the timed window starts only after `profile_steps` returns (it can wait up to 240 s for traces).
    - So the `w1prof` window's `window_sec` and tok/s include profiler overhead.
    - It must be labelled diagnostic and never used as a speed reference (today it is not used, but nothing marks it).

### 4.4 Flush / clean state
18. **M (17 HIGH-4 vs 18 Q3, conflict):** what to do when the flush fails.
    - Audit 1: relaunch the adopted arm once if time allows.
    - Audit 2: no automatic costly restart.
    - **Resolution proposed:** a warm relaunch in the same session costs ≈ 12–14 s plus warm graph capture (E11-04), so **start the Duck on a fresh relaunch of the adopted arm**, which guarantees a clean radix + mamba state. Use `/flush_cache` only as the fallback when the relaunch would break the reserve.
19. **M (17 Q3, 19 §3):** after the flush (or relaunch), take a **cold-prefix observation**: the first Duck requests' cached tokens are logged as proof of the clean state.

### 4.5 NCCL bench
20. **M (18 P0):** NCCL test cases: missing rank, timeout and missing `graph_correct` → rejected (not only "graph wrong").

### 4.6 A2 arms
21. **M (18 E14-7):** int8 needs its own A/A control as well as the cross-arm comparison.
22. **P (18 E14-8):** the matched W8 control is **the final adopted width-8 arm**, measured on the A2 base launch.
23. **M (17 Q5, 18 Q5, 19 §4):** per-width checks:
    - the resolved width / mismatch warning from the server log;
    - the acceptance distribution;
    - **max accept ≤ width** (W4 > 4 = instrumentation fault);
    - KV/graph capacity;
    - errors;
    - committed tok/s;
    - a small concurrent parity as well as the serial one;
    - memory/step time at 40k.
24. **M (19 §6):** target-only correctness is the highest A2 priority, and audit 3 suggests a **small pre-Duck target-only gate**. Cost: one extra relaunch (target-only graphs differ) plus 6 serial prompts. Proposed as optional: only if the Duck reserve allows.

### 4.7 Quota order (conflict)
25. **M:**
    - Audit 1: drop graphs13 first, then int8, then NCCL; keep #44401 short.
    - Audits 2 and 3: drop width 4 / int8 first; keep target-only and agent replay.
    - **Proposed:** follow 2 and 3 (two of three, and both keep target-only). Graphs13 stays because it is a launch flag on the Duck arm with a cheap check.
26. **M (18 Q6):** no 200-min run without a **live quota check**. Already in my plan; now it is a written precondition.

### 4.8 Records and tests
27. **M (17 §9):** keep a final record-integrity regression and a SHA check for the frozen rev 6g records.
28. **M (17 §8):** a restored JIT cache still needs the known-answer check. It is not used in this first run (save only), but it must stay wired before any restore.
29. **M (17 §7):** fix #10 is a substring diagnostic, not a structured error classifier. Labelled as such in the report.
30. **M (18 §5):** the audit lists fixes whose acceptance invariants were incomplete: **1, 2, 4, 7, 8, 11, 12**. Each must get a negative test in the new build.

## 5. Revised patch set (replaces section 2; still not built, awaiting the user's go)

**Must:**
1. **M1 staging collision:** destinations `target-<hash>` / `draft-<hash>`; assert distinct; a basename-`1` test. A **harness fixture with real Kaggle path shapes** (`…/pytorch/w4a16-g128/1`, `…/pytorch/bf16/1`).
2. **M2 Duck reserve:**
   - reserve = 1500 + 180 + 300 preflight + 600 artifacts + a conservative relaunch cost (cold capture 400 s);
   - applies to A1 phases and A1 relaunches; skipped items are recorded;
   - the Duck records its real window, or `solver_window_unavailable`;
   - fake-clock early/medium/late tests.
3. **M3 deadlines:**
   - strict `post_json` (raise when no time);
   - one shared cleanup cutoff (≈ 20 s, concurrent aborts, total join budget);
   - an active-request check after cleanup → relaunch if dirty;
   - a carry-over check;
   - the 6 deadline tests + 14 hung requests.
4. **M4 NCCL bench:** `status ok` + `correct` + `graph_correct is True` + per-call present; tests for graph-wrong, missing-rank, timeout, missing graph flag.
5. **M5 correctness gate for every adopted arm** (#44401, nodelay, graphs13, NCCL env, width):
   - complete and index-aligned samples;
   - both-run margins;
   - arm flips judged against the A/A flips;
   - missing margins → fail;
   - A/A 0 identity → INCONCLUSIVE;
   - a known-answer prefix;
   - a shared small probe suite (serial 6 × 256 + c12 12 × 256);
   - gate fail → revert to the previous verified arm.
   - The 6 gate tests + a fake "faster but wrong" arm + **the revert path exercised**.
6. **M6 latch / occupancy:**
   - `share_running_ge12`, `share_stuck11_queued`, max running, queue-drain latency, retractions, committed tok/s;
   - adopt only on a real gain;
   - the reason and contribution recorded per arm;
   - an 11 + queue → 12 test, and identical → no adoption.

**Same build:**

7. Speed validity: min lines, measured level, ctx within ±25 % → else INCONCLUSIVE; the nodelay 11→12 case labelled; mandatory window metadata; the profiler window labelled diagnostic.
8. Clean Duck start: a fresh relaunch of the adopted arm when the reserve allows, else `/flush_cache` with idle wait + reply classification; `cold_start_unverified` flag; the cold-prefix observation from the first Duck requests.
9. A2:
   - a matched W8 control on the A2 base launch;
   - int8 cross-arm parity + int8 A/A;
   - per-width checks (resolved width, acceptance distribution, max ≤ width, capacity, errors, c12 parity);
   - drop order int8 → width 4 → width 6 → NCCL if time is short;
   - an optional small pre-Duck target-only gate if the reserve allows.

**Low:**

10. graphs13 fails closed on an unparsed list.
11. A staging stop event on fallback.
12. An explicit stop-token rule in `full_reply_text_match`.
13. Portable tests: env-configurable SGLang REF (skip if absent), the mounts test made portable, a PYTHONPATH note.
14. A record-integrity SHA check kept; the substring log signs labelled diagnostic; the JIT known-answer check kept wired.

**New harness scenarios:**
- the basename-`1` staging case;
- a faster-but-wrong arm, rejected with the patch reverted;
- 14 hung requests with failing aborts;
- an 11 + queue baseline vs a 12-running nodelay candidate;
- a flush-failure path;
- the late-start reserve case.

## 6. Build 2 done (2026-10-23): what was built

**User decision ("এগুলো sob e ki blocker?" → "Ok"):** category A + B were built; category C was deferred.

**Notebook:** `a5318ddab1ee692535a7417c9452f9e758c4fb3c290b668cd8c1b0ca0a54c6ea`.

**Modules:**

| Module | sha |
|---|---|
| loadpath | 7c66c179 |
| measure | 68eb69e3 |
| patches | 98da9521 |
| nccl | 355adda0 |
| trace | 8d89ca10 |
| arc2_duck | 758bddb5 |

**GPT bundle:** `Experiment Review/E14A_BUILD2_AUDIT_FOR_GPT_2026-10-23.zip` (43 files; its README holds the item-by-item response to audits 17-19 and the round-2 table).

**Found while building (not in the audits):**
1. **Gate-first.** The gate placed after the windows was starved of phase time and rejected graphs13 as INCOMPLETE in a harness run. The gate now runs first; the windows run only after a pass.
2. **E14_GOOD bug.** It tracked the last *launched* arm, so a failure restored a rejected arm. Replaced by `E14_KEPT`.

**Independent review round 2 (24 findings, all checked; README §2.5):** the key one, R1, was that the A1 gate demanded bs-1 identity across a server restart.
- DFlash2 greedy output is deterministic within one process but differs across restarts (#38009, confirmed in our research notes).
- The A1 gates therefore use `identity=False`; large-margin flips vs the A/A decide.
- Also fixed:
  - relaunch waits capped by the reserve;
  - a non-forced dirty relaunch;
  - kept-arm restore only after a failed forced launch;
  - A2 built on `E14_KEPT`;
  - every unadopted patch reverted, and the revert checked;
  - steps/s speed basis (accept swings per boot);
  - a real cold pass for int8;
  - only a valid NCCL baseline;
  - latch validity;
  - a dirty A1-4 reference invalidated;
  - module-level try/except;
  - a flush between the gate and the windows;
  - the Duck window needs the preflight too;
  - int8 / target-only through `_gated`;
  - queue-drain metric fixed;
  - warm request cleaned up;
  - test stub robustness;
  - the fake answers 400 on a refused flush.

**Tests:**
- unit 165/165 ×3 (measure 32, loadpath 58, patches 36, nccl/trace 39);
- harness 31/31 in the `default` scenario (nodelay + graphs13 adopted, #44401 reverted, Duck on a fresh launch of `graphs13`);
- harness 31/31 in the `corrupt` scenario (nodelay rejected by the gate with 6/6 + 12/12 large-margin flips, windows skipped).

**Still not exercised in the harness:** the flush fallback at the Duck start, a late start, a dirty relaunch, a failed arm launch.

**Status:** not pushed; waiting for the GPT audit of `a5318dda` and the user's go.
