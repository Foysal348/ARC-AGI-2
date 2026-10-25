# 03 — E15B build record (rev 3 on top; rev 2 below): every item built and checked

**Newer: rev 3f / 3g / 3h, see Notes/09 and Notes/10.**

## Rev 3 (2026-10-25)

**What rev 3 adds:**
- Notes/07 A1-A7 (the two AI proposals).
- The user's 2026-10-25 decisions (Notes/08):
  - D-A = start barrier as built;
  - commit = 20 eval tasks with both attempts.
- The fixes from the three agent audits (Notes/08, `Notes/agent audits/`).

| File | sha256 (first 16) | Size |
|---|---|---|
| `Experiment Notebooks/Q27-XHIGH-E15B_SUBMIT_2X35K_S16_DUCK_SGLANG.ipynb` | `0f0d6f675dfcd44a` | 260 kB |
| `Experiment Notebooks/Q27-XHIGH-E15B_EVAL120G4B_2X35K_S16_DUCK_SGLANG.ipynb` | `30653201e11f0466` | 260 kB |

- **SUBMIT notebook:** commit = 20 eval tasks, both attempts (profile E15B: 100 min, 78-min window), scored; rerun = all 240 hidden tasks.
- **EVAL120G4B notebook:** commit = 120 eval tasks at the hidden share.

**Payloads:**
- `Build/payload/q27x_e15b_duck_19192e8006e0.bin`: arc2_duck `140663b2`, tool_agent `f7454201`, sandbox `d29d807e`.
- `q27x_e14_modules_55c1eb7911c6.bin`: unchanged.
- The rev 2 payload `…42dedf221479.bin` was removed (never uploaded).

**Checks on the rev 3 shas:**

| Check | Result |
|---|---|
| `src/test_duck_b.py` | **77/77 PASS, 0 SKIP** (incl. A4 / A6 and every auditor regression) |
| `src/verify_items.py` (items 1-55 + A1-A7 + the audit fixes, both notebooks) | **ALL CODE ITEMS PRESENT** (0 MISS, manifest mismatches 0) |
| harness commit | 25/25 |
| harness rerun | 28/28 |
| harness fault kill | 29/29 |
| harness fault kill2 (2nd relaunch at 12 streams) | 27/27 |
| harness fault slow (no relaunch) | 27/27 |
| harness eval120 | **30/30** (run alone: the 6-parallel run hit a local out-of-memory error in the tokenizer library, as did the first kill run; both pass alone) |

**Kaggle CPU kit (2026-10-25):** run on foysalemonshanto, `q27-e15b-cpu-harness` v1, kit `q27x_e15b_cpu_kit_78073ae3d005.bin`.
- **Image:** the pinned image, Python 3.12.13, on the exact rev 3 shas.
- **Result:** **ALL OK** in 882 s.
  - unit 77/77;
  - commit 25/25;
  - rerun 28/28;
  - eval120 30/30;
  - fault kill 29/29;
  - fault kill2 27/27;
  - fault slow 27/27.
- **Output:** `Experiment Results/Kaggle Runs/E15B CPU kit run1 (foysal, 2026-10-25, kit 78073ae3)/`.

**Not yet done:** the GPU run (needs the user's go; the payload upload to esfersami50 comes first).

---

## Rev 2 (2026-10-24) — superseded

**Date:** 2026-10-24.

**User instructions:**
- "build it first, then the audit";
- "fix everything completely and check each item one by one";
- do not commit yet.

Nothing was pushed, run or committed (git or Kaggle).

## Outputs

| File | sha256 (first 16) | Size |
|---|---|---|
| `Experiment Notebooks/Q27-XHIGH-E15B_SUBMIT_2X35K_S16_DUCK_SGLANG.ipynb` (commit = 45-min smoke at 16 streams, profile E15B; rerun = all hidden tasks) | `750419209f94f9cb` | 247 kB (< 400 kB) |
| `Experiment Notebooks/Q27-XHIGH-E15B_EVAL120G4B_2X35K_S16_DUCK_SGLANG.ipynb` (commit = 120 eval tasks at the hidden per-task share) | `fde1b220032a383d` | 247 kB |
| `Experiment Notebooks/Build/payload/q27x_e15b_duck_42dedf221479.bin` (the B Duck package; manifest verified) | `42dedf221479…` | |
| `Experiment Notebooks/Build/payload/q27x_e14_modules_55c1eb7911c6.bin` (unchanged E14 modules) | `55c1eb7911c6…` | |
| `Experiment Notebooks/Build/payload/dataset-metadata.json` (id `esfersami50/q27-e15b-payload`; **not uploaded**) | | |

**Builder:** `Experiment Notebooks/Build/build_e15b.py`.
- It executes `build_e14.py` (E14S) in memory, which runs the E12 builder in memory.
- Nothing in Exp 12 or Exp 14 is written; the E14 notebooks keep their PUSHED shas.
- The build is deterministic (rebuilding gives the same sha).

**Package:** 3 files replaced from `src/duck_b/`, each with its upstream line ending.

| File | Line ending | sha | Diff |
|---|---|---|---|
| `arc2_duck.py` | LF | 758bddb5 → b7ea4d24 | 597 lines |
| `tool_agent.py` | CRLF | 485ef8a9 → f7454201 | 28 lines |
| `python_tool_sandbox.py` | CRLF | c2208578 → d29d807e | 62 lines |

The notebook diff vs E14S EVAL120G4B is 520 lines (all in `Experiment Review/diffs/`).

## Verification

| Check | Result | Output |
|---|---|---|
| `src/verify_items.py`: every Notes/02 item 1-55 against the BUILT notebook + package (exact evidence text) | **ALL CODE ITEMS PRESENT**: 48 OK, 7 NOTE (non-code), 0 MISS, manifest mismatches 0 | `Experiment Review/verify_items_output.txt` |
| `src/test_duck_b.py` (64 checks; the Windows sandbox shim of the harness makes the sandbox checks run here too) | **ALL PASS, 0 SKIP** | `Experiment Review/test_duck_b_windows_output.txt` |
| `src/harness/harness_e15b_nb.py` commit (SUBMIT, final build) | **23/23** | `_scratch/e15b/harness_commit2.log` |
| harness rerun (SUBMIT, final build) | **25/25** | `_scratch/e15b/harness_rerun2.log` |
| harness eval120 (EVAL120G4B, final build) | **27/27** | `_scratch/e15b/harness_eval120b.log` |

**Harness E15B checks** (beyond the E14S ones):
- the E15B flags (width 6, graphs 1..17, delayer 1, mrr 17, mamba 68, no prefill delayer);
- streams and workers 16; isolation PASS;
- `E15B` block and CONFIG;
- first exposure 0, facts view, fair share 0;
- window 100,000;
- **every attempt ran in one run (no pause / chunk / resume);**
- **attempt 2 started only after every attempt 1;**
- the per-attempt records;
- `b_summary` with the server-log block and the watchdog;
- no worker error and no stale server-down state;
- the self-test on the fixed task.

**A harness finding, fixed:** #27's staging deadline (the launch-join point) was in the past under a 25-min budget, so the copy aborted at once (rerun 24/25). A floor of 10 min was added; rerun then passed 25/25. In the real runs the deadline is T0 + 55 min (hidden, EVAL120) or T0 + 15.5 min (45-min smoke); the copy takes ~1.5 min.

**Harness limits** (the fake server, not our code):
- The fake server has no `sglang:generation_tokens_total` (so `server_gen_tokens` = null).
- Its 30-s gauges show low "running" because fake requests are short and the workers spend most time in the sandbox.
- One socket Traceback in the fake server's log is its client-abort handler.

Real values come from the GPU smoke.

## Item status (Notes/02 numbering): all built

| # | Item | Status | Evidence (in the built notebook / package) |
|---|---|---|---|
| 1 | Two independent attempts | done | `E12_R6_B1_VIEW` facts, 2 branches; Duck sends no seed |
| 2 | Attempt 1 on all tasks first | done (existing) | `stage1_left` gate; harness: attempt 2 starts after every attempt 1 |
| 3 | 35k, no pause / chunk / resume | done | `E12_MAX_GEN` 35000 = `E12_B_ATTEMPT_TOKENS` (asserted); first exposure 0; harness: one run per attempt |
| 4 | No 12k / 24k / 32k caps | done | cap = remaining − 4,096, then the remaining tokens (unit-tested) |
| 5 | No trim inside an attempt | done | reserve = cap; fixed window 100k; harness `trimmed_requests` 0 |
| 6 / 49 | min_admit 600 | done | |
| 7 | a1 / a2 + the partial-program test run (attempt_2 tier) | **done (rev 2)** | `arc2_b_partial_test` at the attempt end; tested in the selection (partial before rules) |
| 8 | End-of-attempt harvest | done | autocheck + check harvest + #50 |
| 9 | xhigh | done | |
| 10 | Turn rollback | done | `ARC3_YIELD_AFTER_BATCH`; fake-reply test ON 2/2 results kept, OFF = upstream rollback |
| 11 | submit / check as tools → python | done | sandbox-tested (code argument and name argument) |
| 12 | Tool contract in the prompt | done | |
| 13 | Autocheck names | done | |
| 14 | Sandbox: sys shim (setrecursionlimit capped at 20,000), time, import messages | done | **8 sandbox tests pass** |
| 15 | **Server watchdog** | **done (rev 2)** | the notebook thread relaunches a dead SGLang (or one with /health failing for 180 s), at most 2 times and only with ≥ 20 min left; while it is down, branches wait on `arc2_duck.SERVER_DOWN` (no failures counted); recorded in `b_summary.watchdog` / `server_down_waits` |
| 16 | Launch fallback 16 → 12 → W1, and a gate fallback | **done (rev 2: any 16-stream failure → 12 first; W1 last)** | `e11_next`, `e11_bring_up` |
| 17 | Self-test + tokenizer probe on fixed tasks | **done (rev 2)** | `E15B_SELFTEST_TASK`; tokenizer probe fragment fixed |
| 18 | Worker restarts; running count released | done | |
| 19 | Attempt-2 order | done (existing `rank()`) | |
| 20 | **CPU rule library** | **done (rev 2)** | `rule_library_outputs`: rotations, flips, transposes, bbox crop, scales, tilings, mirrors, global color map; only rules exact on every train pair; after every program candidate, before the test input; tested |
| 21 | Defined-functions inventory | done | |
| 22 | Bounded-search guidance | done (prompt) | |
| 23 | No overfit_flags selection | unchanged (correct) | |
| 24 | **pip / venv / sysconfig timeouts** | **done (rev 2)**: 300 / 1,800 / 120 s | |
| 25 | Keep the payload .bin until the hidden run ends | process | |
| 26 | submission.json per branch | kept on purpose (crash safety) | |
| 27 | **Staging copy stops at the launch join** | **done (rev 2)**: deadline = max(T0 + 10 min, min(T0 + 45 min, give-up − 20 min)) | |
| 28 | Labels | done (variant, backend `…-w6-s16` / `-w1`, arm label, markdown) | |
| 29 | Per-attempt record | done | `b_events`, `b_budget`, `b_candidate` |
| 30 | B-risk metric | done | `at_budget_with_progress` |
| 31 | **Candidate fields** | **done (rev 2)** | `b_candidate`: written / retained / check attempted / executed / timeout / train exact / host verified / harvested / submitted |
| 32 | **Server-side generated tokens** | **done (rev 2)** | `server_gen_tokens` vs `counted_gen_tokens` |
| 33 | **Retract counter** | **done (rev 2)** | server_args / retraction_policy excluded |
| 34 | **Accepted vs harvested** | **done (rev 2)** | `accepted_submit`, `harvested_only` |
| 35 | **Effective effort** | **done (rev 2)** | `effective_effort` |
| 36 | Delayer off | done | |
| 37 | 16 streams (mrr 17, graphs 1..17, mamba 68) | done | harness flag check |
| 38 | Gates allow 16 | done | KV per request 50k; harness: gate concurrency 16 |
| 39 | Real-concurrency check | done | gauges + server-log histogram |
| 40 / 41 | KV fit, READY time | **GPU smoke** | |
| 42 | **Engine block** | **done (rev 2)** | `server_log`: running histogram, queue > 0 share, gen tok/s, retract / preempt / OOM / Traceback (regex checked on the real EVAL120 log) |
| 43 | Harness + CPU kit | **harness done (3 modes)**; the Kaggle CPU kit needs a push → the user's go | |
| 44 | Smoke profile | done | E15B, 45 min, 15-min window, 16 tasks |
| 45-48 | Fair share off, fixed window, timeout 3,000 s, commit turn | done | harness: commit turns 32/32 and 240/240 with a tool call (fake) |
| 50 | Reasoning-code harvest | done | tested |
| 51 | Yield 2,048 | kept on purpose | harmless after #10 |
| 52 | No new scheduler | done | |
| 53 | **Retract / preempt = 0 check** | **done (rev 2)** | in `server_log` |
| 54 | **Deadline harvest** | **done (rev 2)** | final harvest + partial test within a 30-s grace after the deadline (`E12_B_FINAL_GRACE_SEC`); the verification honours it; scheduler join grace 90 s; tested |
| 55 | Facts cap / size / traces | **checked** | 1,000 training tasks: facts p50 721 / p90 784 / max 800 tokens, never over the bound (char max 1,519 < 3,600): no change needed; notebook 247 kB; transcripts kept |

## Known risks for the audit / smoke

1. **The commit turn may produce no code** (thinking on, ≈ 4-5k tokens). #50 harvests code in that thinking. Measured by `commit_with_tool_call`.
2. **Startup and memory at 16 streams are unmeasured:**
   - graphs 1..17;
   - mamba 68 (≈ 0.55 GB / GPU);
   - the headroom gate is ≥ 1.0 GB.

   The fallbacks cover a failure at the cost of one relaunch.
3. **Per-stream rate at 16 ≈ 26-30 tok/s:** one attempt ≈ 20-22 min; the request timeout is 3,000 s.
4. **The watchdog relaunch** costs ≈ 10-20 min (graph capture). Branches wait meanwhile. At most 2 relaunches, only with ≥ 20 min left.
5. **D2 = yes by default:** attempt 2 also runs on accepted tasks, last.
