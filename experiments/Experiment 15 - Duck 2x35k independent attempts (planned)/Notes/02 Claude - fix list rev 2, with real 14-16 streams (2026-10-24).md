# 02 — Prioritized fix list for config B, rev 2: adds the 11-of-12 fix and real 14-16 streams

**Date:** 2026-10-24.

**Why rev 2:** Notes/01 was found empty on disk on 2026-10-24, so this file is self-contained. It contains every item of rev 1 plus the new engine items. Rev 1 numbering is kept; the new items are 36-44.

**User decisions:**
- config B: 2 × 35k independent attempts per task; attempt 1 on every task first, then attempt 2 (final);
- the request of 2026-10-24: "fix the 11-running problem too", and **14-16 streams**;
- the turn rollback and the `submit` tool bug stay in (items 10-11).

**Sources** (all re-read in full on 2026-10-24):
- Exp 14 `Notes/EVAL120 analysis 2026-10-24/`: A_score_coverage, B_behaviour, C_engine;
- GPT Notes/44, 45, 46;
- Notes/43: merged list and my verifications.

**Legend:**
- [A] / [B] / [C] = agent reports;
- [G1] / [G2] / [G3] = GPT Notes/44 / 45 / 46;
- [V] = my verification (Notes/43);
- [S] = SGLang 0.5.21 source (`_scratch/session_1c6db7aa/sgl_src/src/sglang/srt/managers/min_free_slots_delayer.py`).

**Base:** E14S (`build_e14.py` variant E14S; XH 41/172 = 25.83 %, LB 22.78) with Duck adapter rev 6g.

---

## Correction to the 11-of-12 figure

[C] §2.1 measured the share of time with 12 running:
- **time-weighted: 25.0 / 25.5 / 26.6 %** of the window; 11 running 73 / 73 / 71 %;
- by count of decode log lines: 13-16 % (564 / 4,229, 610 / 4,195, 419 / 3,777);
- mean running 11.21 / 11.23 / 11.21.

The "~14 %" is the line share. The loss is the same either way:
- **2.2-2.5 % of the tokens** at 12 streams ([C] §2.3);
- the 12th stream adds only +2.7 % decode, because the step grows ≈ 5.2 ms per added request.

The large gain is from more streams (14-16), not from the 12th alone.

---

## P0-A — core of config B

| # | Change | Where / how | Evidence |
|---|---|---|---|
| 1 | **Two independent attempts per task.** Attempt 2 = a fresh Duck run (facts view, its own seed). Not a resume, not the indexed-view branch 1. | `solve_all_r6(branches=2, b1_view="facts")` and a per-attempt seed. Check that a facts-view b1 gets its own agent and history. | [A] §2: b1 gave 0 correct in 244 runs. [B] §1.6: two xhigh runs overlap on 26 of 61 queries. [G1-G3] |
| 2 | Attempt 1 on every task before any attempt 2; order ORDER2 + cells. | stage 1 = attempt 1, stage 2 = attempt 2 | [A] §4 |
| 3 | **35,000 generated tokens per attempt, then it ends.** No 28k pause, no chunks, no resumes. | `max_generated_tokens=35000`, `first_exposure_tokens=None`; remove the resume classes | [C] §3.4: 58-66 % of prefill was resume re-prefill; −18-30 % throughput after hour 2 |
| 4 | No 12,288 / 24,576 / 32,768 request caps; each request may use the remaining attempt budget. | `_arc2_apply_cap` | [B] §3.1, [C] §6: 84-98 tool-less length cuts |
| 5 | **No context trim inside an attempt.**<br>• reply reserve = the remaining attempt budget;<br>• Duck window ~100k (server context 155,648).<br>Today trim = 76,484 − reserve − 512. A 35k reserve would trim at ~41k. | `arc2_duck.py:1530-1542` | [B] §5.1b: correct 16 % vs 34 % / 5 % vs 30 %. [V] trimmed branches 34 / 41 / 64 |
| 6 | ~~Admission guard per attempt: start only if ≥ ~1,300 s remain.~~ **Superseded by item 49:** a worker exits when the time left < `min_admit_sec`, so keep ≈ 240-600 s and rely on the deadline clamp plus harvest. | `min_admit_sec` (240 s today) | [A] §4: per-stream p50 30.6 tok/s. At 16 streams the per-stream rate drops (≈ 489 / 16 ≈ 30 tok/s decode), so re-check |
| 7 | attempt_1 / attempt_2 assembly:<br>• a1 = attempt 1's verified program;<br>• a2 = attempt 2's verified program if distinct;<br>• else the other attempt's best partial program on test;<br>• test input last.<br>Keep `select_attempts_r5()`. | | [A] §2, [B] §3.2: pass2 = 0; 58 of 60 a2 slots were the test input. [V] 46 = 46 |
| 8 | End-of-attempt harvest (`check()` on a retained one-grid transform) | proven, ON | [A] §2: 12 / 4 / 5 solved tasks |
| 9 | Effort xhigh only | | [A], [B] §2 |

## P0-B — bugs and robustness

| # | Fix | Where | Evidence |
|---|---|---|---|
| 10 | **Duck turn rollback:** keep the turn and drop only the unexecuted call when the 2,048-token yield or a stop fires after the first of two parallel tool calls. | `vendor/inference/agent/tool_agent.py` L7094-7101, L7183-7186 | [B] §5.1a: 56 / 47 / 6 events, 0.52 / 0.46 / 0.06 M context tokens lost; 1ae2feb7 fell back to a 2,358-token prompt 4 times (another run solved it at 17.6k) |
| 11 | **`submit` called as a tool** → route it to the python `submit` helper. | Duck tool dispatch | [C] §4: 22 / 28 / 23. [V] 22 / 27 / 23 branches |
| 12 | Tool contract in the prompt (one tool, `python`; `check` / `submit` are functions inside it) + a unit test. | `arc2_r6_system_prompt()` | [G3] P0-3 |
| 13 | Autocheck names: add `candidate` and `trans` to `_R6_AUTOCHECK_RE`, with 4 tests. | `arc2_duck.py:1177`, `src/e16/arc2_duck_e16.py:1177` | [G1], [V] |
| 14 | Sandbox:<br>• a safe `sys` shim (`setrecursionlimit`, `maxsize`);<br>• allow `time`;<br>• a helper-import message;<br>• `numpy` stays blocked, and the error says so. | `python_tool_sandbox.py:281`, `ARC2_R6_ALLOWED_IMPORTS` | [V] `sys` blocked 21 / 21 / 13 times |
| 15 | Server watchdog (restart if SGLang dies; more retries) | engine cell | code audit H1 |
| 16 | Launch fallback: if the new flags fail, relaunch with the E14S flags (12 streams); if those fail, the W1 flags. | engine cell | code audit H2. **Now also covers items 36-38** |
| 17 | Sandbox self-test and tokenizer probe on a fixed task | notebook | code audit M4 |
| 18 | Worker loop fully in try/except | `solve_all_r6` | code audit M5 |

## P0-C — engine: real 14-16 streams (new, user request)

| # | Change | Where / how | Evidence |
|---|---|---|---|
| 36 | **Disable the DFlash min-free-slots delayer:** `--min-free-slots-delay 1`. A value ≤ 1 disables it ([S] `resolve_min_free_slots`). Without it, with mrr 13 the threshold is min(4, max(2, (13+5)//6)) = 3, so the 12th request waits while 11 run. | server flags | [C] §2.2 (source and log), [S] |
| 37 | **Streams N = 16** (fallback 14; D9):<br>• client workers / streams = N;<br>• `--max-running-requests` N + 1;<br>• `--cuda-graph-bs-decode` 1..N + 1;<br>• `--max-mamba-cache-size` ≥ 4 × (N + 1) (16 → 68; 14 → 60).<br>The +1 is a margin for smoke / diagnostic requests: Notes/30's nodelay crash came from a 13th running request on buffers sized for 12. | server flags + `E12_STREAMS`, `workers` | [C] §8.2b: e2e +8 % at 14, +12 % at 16 vs 11 running (linear model, **not measured**). The server uses ~3 mamba slots per running request; 46 MB per slot per GPU |
| 38 | **Client capacity gate must allow N.** Check:<br>• the E11 caps gate (`effective_running`, `slots_per_request 4` → mamba 68 / 4 = 17 ≥ 16);<br>• the KV-only admission (`kv_admit_limit` 0.95);<br>• the isolation preflight at N, not 12;<br>so the client really offers N. | `capacity_probe`, E11 gate, preflight | memory: verify real concurrency (E11-03 cap miss) |
| 39 | **Real-concurrency assert in the run:** log time-weighted running, and warn and report if the mean running < N − 0.5. | heartbeat + report | memory: verify real concurrency |
| 40 | **KV fit at 16 × B context:**<br>• each attempt grows to ≈ 40-45k context (prompt + 35k generated + tool outputs);<br>• the pool shrinks by ≈ 0.55 GB / GPU (≈ 50k tokens) for mamba 68 → ≈ 865k tokens;<br>• average in use ≈ 16 × 25k ≈ 400k (46 %);<br>• worst case, all 16 near the end: ≈ 720k (83 %).<br>The KV admission gate covers the tail. | capacity check | [C] §4: in-use p90 0.40 at 12 today; [C] §8.2b caveat ii-iii |
| 41 | **Startup time with graphs 1..17:** today READY at 708-771 s, of which the bs-13 verify-graph capture alone takes 363-376 s (JIT). More batch sizes add capture time. It must stay well under the 30-min launch timeout (memory: GCSFUSE slowdown, READY 1,197 s once). | measured in the smoke (item 44) | [C] §1 |

## P1 — same build, smaller

| # | Fix | Where / notes | Evidence |
|---|---|---|---|
| 19 | Attempt-2 order: partial-progress tasks first, then no progress, accepted last (D2) | stage-2 rank, gold-free | [G2], [V] §7.2 |
| 20 | CPU rule-library fallback instead of the test input | assembly | 4-day review D, [G1] |
| 21 | Retained-state inventory in each tool result | tool formatter | [G3]: NameError 19 / 22 / 23 branches |
| 22 | Bounded search guidance + one-train-input pre-check | prompt + helper | [G1] e8686506 |
| 23 | No `overfit_flags` selection | | [B] §4 |
| 24 | pip / venv timeouts | notebook | M6 |
| 25 | Keep the payload `.bin` files until the hidden run ends | process | M6 |
| 26 | `submission.json` written once | notebook | L2 |
| 27 | Abort the staging copy on timeout | notebook | L7 |
| 28 | Fix the stale labels | notebook | L5 / L7 |

## P1-R — reporting and measurement

| # | Item | Why |
|---|---|---|
| 29 | Per-attempt record:<br>• seed, tokens, end reason;<br>• first transform and check tokens;<br>• best train k/n;<br>• harvested or not. | To score B |
| 30 | **The B-risk metric:** attempts reaching 35k with a transform or partial progress, and the attempt-2 recovery on those tasks. | [V]: under rev 6, 21/46 and 17/41 correct programs first appeared after 35k |
| 31 | Candidate fields (written / retained / check attempted / executed / timeout / train exact / host verified), reasoning-only code, blocked imports | [G1] |
| 32 | Server-side count of deadline-cut tokens | [A], [C] |
| 33 | Fix the retract counter | [A] §9 |
| 34 | Accepted vs harvested | [A] |
| 35 | Effective request effort | [G3], [V] |
| 42 | **Engine block in the report:**<br>• the running-level histogram (time-weighted);<br>• queue > 0 time;<br>• decode tok/s at each level;<br>• step ms;<br>• KV and mamba max;<br>• READY time;<br>• graph-capture time. | Items 36-41 must be read from the run |

## Validation steps

| # | Step | Details |
|---|---|---|
| 43 | **Unit tests + harness + Kaggle CPU kit** on the exact sha | The harness checks the new flags:<br>• delayer 1;<br>• mrr N + 1;<br>• graphs 1..N + 1;<br>• mamba;<br>• streams N;<br>• the gate allows N. |
| 44 | **Short GPU smoke, ~30 min hard stop, before EVAL120** (memory: short GPU smoke over audits). | Launch with the 16-stream flags.<br>Pass rule:<br>• READY time recorded;<br>• no crash or OOM;<br>• real running ≥ 15.5 mean over ≥ 10 min of Duck load;<br>• decode tok/s by level recorded;<br>• no wrong output in a 12-request target-only parity probe.<br>If it fails, re-run the smoke at 14; if that fails too, keep 12 with only item 36. **Needs the user's go.** |

---

## Open decisions

| # | Decision | Recommendation |
|---|---|---|
| D2 | Attempt 2 also on accepted tasks, last? | Yes (fits at 16 streams; see below) |
| D3 | E16 off? | Off (flag kept) |
| D4 | Isolation INCONCLUSIVE → 6 workers? | Keep (approved safeguard) |
| D6 | S1-60K (LB 34.58) stays the fallback submission | Yes |
| D7 | Validation: EVAL120 G4B or EVAL12 / 24? | Smoke (item 44), then EVAL120 G4B |
| D9 | ~~16 streams with a fallback to 14, or 14 directly?~~ | **Resolved by the user (2026-10-24): 16.** See Rev 2c |

Resolved: D1 = B; D8 = independent Duck attempt; the delayer fix and 14-16 streams are IN; **D9 = 16 streams, and requests must not wait in the server queue again** (user, 2026-10-24).

## Held — not in this build (kept, not dropped)

| Item | Why |
|---|---|
| Rest of E15: MTP, width 4, chunked prefill 2048, NCCL, FP8 drafter, triton attention, `SKIP_DECODE_LOCK`, 2×TP2, the 65k window, the E15 audit fixes (N43 #35-53) | User: E15 off. Items 36-41 move the stream part into B |
| #43567 backport, JIT / compile cache | Engine scope kept to items 36-41 |
| E17 accuracy tests (N43 #54-63, #77, #90) | Separate experiment |
| Poetiq-style attempt 2; candidate-first prompt; generalization review / soft sanity score; view A/B; hybrid; S1 EVAL120; SGLang vs vLLM S1; EVAL12 / 24 paired sets; raw-prefix diagnostic; DSL | Later arms (N43 #29-34, #73-81, #84-89) |
| Process:<br>• wheelhouse as a Kaggle Model;<br>• 12 h margin;<br>• cancel API;<br>• **10-26 rules acceptance (user)**;<br>• **11-02 deadline check**;<br>• write-up `[PENDING]` | Process |

---

## Capacity check (estimate; recompute with the built notebook before any GPU run)

| | 12 streams, no resumes | 16 streams, no resumes |
|---|---|---|
| Effective tok/s | ≈ 380 ([C] §3.4) | ≈ 380 × 1.12 ≈ **425** ([C] §8.2b; unmeasured) |
| Attempt 1, 240 tasks, ≈ 7.6 M tokens | 20.0 ks | 17.9 ks |
| Attempt 2, ≈ 185 unverified tasks, ≈ 6.2 M | 16.3 ks | 14.6 ks |
| Subtotal (window ≈ 42 ks) | 36.3 ks | **32.5 ks** |
| D2: attempt 2 on ≈ 55 accepted tasks, ≈ 1.9 M | +5.0 ks → 41.3 ks (tight) | +4.5 ks → **37.0 ks (fits)** |

**Per-stream rate at 16:** ≈ 30 tok/s decode (489 / 16), less with prefill, so a 35k attempt ≈ 20-22 min of wall time. The admission guard (item 6) uses the measured rate.

**KV at 16:** see item 40 (average ≈ 46 %, worst ≈ 83 % of ≈ 865k).

**EVAL120 G4B:** 120 tasks at the hidden per-task share (≈ 21 ks window), so the same shares hold.

---

## Rev 2b — gaps found in the deep re-check of the code (2026-10-24)

Checked against:
- `arc2_duck.py` (`solve_all_r6` L2686-2960, `_arc2_apply_cap` L1530-1542, `DUCK_ENV` L40-80);
- `vendor/inference/agent/tool_agent.py` (seed L204 / L5242, drain L205-262 / L6455+);
- the E14S notebook source (`E12_R6` block, `rev6_window`, `E12_DUCK_REQUEST_TIMEOUT`).

### Would break B (P0)

| # | Gap | Code fact | Fix |
|---|---|---|---|
| 45 | **Fair share would chunk attempt 2.** Every stage-2 run pauses after max(`min_chunk_tokens` 12,288, share) tokens and is re-queued: exactly how the old b1 ended up with one 12-25k chunk. | `arc2_duck.py:2928-2934`; `E12_R6_FAIR_SHARE` = 1, `E12_R6_MIN_CHUNK` = 12288 | For B, `pause_at_tokens=None` for both attempts (`fair_share=False`). `max_generated_tokens=35000` is the only stop. |
| 46 | **The context-window formula shrinks with more streams.** `rev6_window = min(131072, 1.0 × KV / streams)` = 917,816 / 12 = 76,484 today. At 16 streams with mamba 68 (KV ≈ 865k) it would be ≈ 54k, so the trim would fire *earlier* than today. | notebook CELL 6 (`E12_RUN["rev6_window"]`), `os.environ["E12_DUCK_CONTEXT_WINDOW"]` | Item 5 implemented as a fixed window (≈ 100k, ≤ the 131,072 cap, ≤ the server context 155,648), not KV / streams. The KV admission gate (0.95) and item 40 cover memory. |
| 47 | **The request timeout is too short for 35k requests.** `E12_DUCK_REQUEST_TIMEOUT` = 1,800 s. Today's max request latency is 1,017-1,096 s for 24,576 tokens (worst ≈ 22 tok/s per stream). At 16 streams one 35k request can take ≈ 1,600-1,900 s → timeout → the reasoning is lost. | notebook `E12_DUCK_REQUEST_TIMEOUT`, `solve_all_r6` `timeout` | Raise it to ≈ 3,000 s; the deadline clamp (R6-S6) still applies. Report the request latency p99. |
| 48 | **The final request ends mid-thinking at 35k, with no chance to write code.** Qwen emits tool calls only after `</think>`. Under rev 6 a length cut was followed by a resume ("do not restart: write transform + check()"). B has no resume, so a cut at the budget is pure loss. | `_arc2_apply_cap`; `R6_TEXT["length"]` L1153 | **Reserve the last ≈ 3k:**<br>• the request cap = remaining − 3,000;<br>• when the remaining budget is ≤ 3,000, send one short commit turn ("budget almost spent: write your best `transform` now and call `check(transform)`").<br>Also state the attempt budget in the opener (≈ 35k tokens), so the model can plan.<br>Evidence: 84-98 tool-less length cuts; 64-67 no-transform tasks; harvest gave 12 / 4 / 5 hits. |
| 49 | **Item 6 corrected: `min_admit_sec` makes workers exit.** A worker *returns* when the time left < `min_admit_sec` (`arc2_duck.py:2868`). At 1,300 s all 16 workers would stop ≈ 22 min before the deadline. | `solve_all_r6` worker loop | Keep `min_admit_sec` ≈ 240-600 s and rely on the deadline clamp plus harvest. A late, cut attempt still has value: 9-11 of the 41-46 correct programs came by 20k tokens. |

### Important (P1)

| # | Gap | Fix |
|---|---|---|
| 50 | **Code written only in the reasoning text** is never run (db0c5428: `def transform` inside thinking, never executed). | At the end of an attempt, extract complete `def transform…` blocks from the reasoning, run them in the sandbox and host-verify (gold-free). |
| 51 | **The `yield_tokens` 2,048 per-turn yield** is what triggers the turn rollback (item 10), and B no longer needs mid-attempt pause checks. | After fix 10, check whether budget enforcement needs the yield; if not, raise or disable it (`LOCAL_ANALYZER_YIELD_TOKENS`) for B. |
| 52 | The stage-2 rank already orders by class (0 progress → 1 nothing → 2 one verified → 3 two or more), which is item 19's order. | No new scheduler: remove the resume kind and keep `rank()`. D2 decides whether classes 2 and 3 get attempt 2. |
| 53 | **Precedent:** D66 / L66 direct reasoning had 0 preemptions at 12 slots but 66 at 16 (contexts up to 86k). B's contexts are ≤ ≈ 50k, so this is likely fine, but it must be measured. | Item 44's smoke pass rule adds: 0 retract / preempt lines in the server log (today 0). |
| 54 | Deadline harvest of in-flight attempts (7 / 7 / 4 requests were cut mid-request at the deadline). | Check that the pending checks of cut attempts are verified and enter the final assembly (`verify_skipped_deadline` was 0). |

### Checklist items (memory rules)

| # | Item |
|---|---|
| 55 | **The build:**<br>• commit notebook < 400 kB, payloads in the dataset;<br>• full thinking traces saved for every request (≈ 480 attempts; size fine);<br>• the hidden test is sized like training (larger grids), so re-check the facts cap (3,600 chars / 800 tokens) and the prompt size. |

### Verified, no fix needed

- **Attempt 2 really is independent.** The Duck sends no per-request seed (`LOCAL_ANALYZER_SEED` = −1 → not sent; `openai_compat.py:95`), so every request samples freshly. Attempt 2 has its own `BranchSession` (own agent and history).
- **The `ARC3_PRIORITY_TAIL_FADE` message** appeared at 69.5 min left, but running stayed at 11-12 until the last 3 min ([C] §7): our scheduler controls admission. Harmless.
- **The history turn cap** (`ARC3_HISTORY_ASSISTANT_TURNS` 150) is far above B's ≈ 15-25 requests per attempt.

---

## Rev 2c — the 16-stream configuration (user D9 = 16; "requests must not queue again")

### Why a request waits in the server queue (SGLang 0.5.21 source), and how each cause is removed at 16

| Cause | Source | Today (12 streams) | At 16 |
|---|---|---|---|
| **DFlash min-free-slots delayer:** holds a fresh prefill while free slots < min(4, max(2, (mrr + 5) // 6)) | `managers/min_free_slots_delayer.py:20-38`; called at `managers/scheduler.py:3856` | mrr 13 → threshold 3 → the 12th waits while 11 run (72 % of the time queue > 0) | **`--min-free-slots-delay 1`** → `resolve_min_free_slots` returns None → the delayer is off |
| **Request slots:** `get_num_allocatable_reqs` = min(`pp_max_micro_batch_size` − running, `req_to_token_pool` free); `pp_max_micro_batch_size` = mrr | `scheduler.py:3784-3802`, `1194-1200` | mrr 13 | **mrr 17** = 16 client streams + 1 margin (Notes/30's nodelay crash was a 13th request on buffers sized for 12) |
| **Mamba slots:** a new request needs `rem_mamba_slots` > 0 (free + evictable mamba states) | `managers/schedule_policy.py:705-716`, `850-858` | 56 slots; in use p50 33 at 11 running, max 39 ([C] §4: ~3 per running request) | **`--max-mamba-cache-size 68`** (16 × 3 = 48 in use + 20 for radix states) |
| **KV tokens:** `NO_TOKEN` when the pool cannot hold the prefill + the decode reserve. The reserve per running request is min(max_new − out, 4,096) × new_token_ratio, so it is small. | `schedule_policy.py:785-798`, `1095` | in-use KV max 0.47-0.52 | ≈ 865k pool (mamba 68 costs ≈ 50k); average ≈ 46 %, worst ≈ 83 % (item 40). The client KV gate holds new attempts at 0.95. |
| **Generic prefill delayer** | `scheduler.py:1394-1410`: only with `--enable-prefill-delayer` | off | stays off (the flag is not set; the harness asserts it) |
| **Chunked prefill:** one chunked request at a time; others wait for its 4,096-token chunks (≈ 2.6 s each) | `scheduler.py:3866` | 55k resume re-prefills took 34-37 s ([C] §4) and caused the (10, 1) samples | B has no resumes. The only big prefill is an attempt's first request (≈ 3-8k). Transient waits stay seconds long. |

### The full 16-stream setting (every dependent value)

| Area | Setting | Today | B / 16 |
|---|---|---|---|
| **Server** | `--min-free-slots-delay` | unset (→ 3) | **1** (off) |
| | `--max-running-requests` | 13 | **17** |
| | `--cuda-graph-bs-decode` | 1..13 | **1..17** (drop `--cuda-graph-max-bs-decode`) |
| | `--max-mamba-cache-size` | 56 | **68** |
| | `--enable-prefill-delayer` | off | off (asserted) |
| | context / mem fraction / chunked prefill / width 6 / FP8 KV / fp16 SSM | 155,648 / 0.83 / 4096 / 6 | unchanged |
| **Client** | `E12_STREAMS` (`ARC3_MAX_ACTIVE_STREAMS`) and `solve_all_r6(workers=…)` | 12 | **16** |
| | E11 caps gate (`effective_running`, `slots_per_request` 4) | 56 / 4 = 14 → 12 | 68 / 4 = 17 ≥ 16 (the harness asserts `effective_running` = 16) |
| | Isolation preflight | 12/12 | **16/16** |
| | Isolation INCONCLUSIVE fallback (D4) | 6 workers | 6 workers (unchanged unless the user decides) |
| | KV admission gate | 0.95 | 0.95 |
| | Duck context window (item 46) | KV / streams = 76,484 | **fixed 100,000** |
| | Request timeout (item 47) | 1,800 s | **3,000 s**. One 35k request at ≈ 26 tok/s effective per stream ≈ 1,350 s; × 1.4 worst ≈ 1,900 s |
| | `min_admit_sec` (item 49) | 240 s | **600 s** |
| | fair share / chunks (item 45) | on, 12,288 | **off** |
| | Per-attempt budget | — | 35,000 generated, with the last ≈ 3,000 reserved for the commit turn (item 48) |
| **Fallbacks** (item 16) | if the 16-stream launch fails | — | relaunch with the E14S 12-stream flags (with the delayer still off and mrr 13), then the W1 flags |
| **Report / heartbeat** (items 39, 42) | running-level histogram, queue > 0 share | — | **targets: mean running ≥ 15.5; queue > 0 ≤ 5 % of the time** (only chunked-prefill transients) |

### The smoke's pass rule, updated (item 44; ~30-min hard stop; needs the user's go)

- READY time recorded, and below the 30-min launch timeout with margin. Graphs 1..17 add capture time.
- No crash, OOM, retract or preempt line.
- Over ≥ 10 min of Duck load:
  - mean running ≥ 15.5;
  - queue > 0 for ≤ 5 % of the time;
  - decode tok/s and step ms recorded per level.
- A 12-request target-only parity probe: no wrong output.
- **If it fails:** first look at the server log for the cause (rule: read server logs before explaining). Then fix and re-smoke. A 14-stream re-smoke is the emergency fallback only.

### Capacity at 16 (from the table above)

- **Tokens:** attempt 1 + attempt 2 ≈ 13.8 M in ≈ 32.5 ks at ≈ 425 tok/s (unmeasured).
- **With D2:** ≈ 37 ks of the ≈ 42 ks window.
- **Per-stream:** ≈ 26-30 tok/s, so one attempt ≈ 20-22 min of wall time.
