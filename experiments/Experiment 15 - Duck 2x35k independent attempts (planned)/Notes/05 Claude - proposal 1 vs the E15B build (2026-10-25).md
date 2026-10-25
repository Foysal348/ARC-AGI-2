# 05 — AI fix proposal 1 (Notes/04) compared with the E15B rev 2 build

**Date:** 2026-10-25.

**Scope:** every item of Notes/04 compared with the built code (`src/duck_b/*.py`, the built SUBMIT notebook sha `750419209f94`). Nothing was changed. Proposal 2 (there are two, not three) and the final adopt list: Notes/06, Notes/07.

**Verdicts:**
- **same** — the build already does it;
- **ours better** — the build covers it in a way that loses less;
- **adopt?** — a small change worth taking;
- **no** — not taken, with the reason.

## Phase A — attempts

| # | Proposal 1 | The build | Verdict |
|---|---|---|---|
| 1 | Two agents with their own histories, facts view; a seed is optional | The same; Duck sends no seed (−1), so each attempt is a fresh sample | same |
| 2 | **Completion barrier:** attempt 2 only after every attempt 1 has *finished* | Attempt 2 starts after every attempt 1 has *started* (`stage1_left`). Attempt 2 of a task is queued only when its own attempt 1 ends. | **no** (see D-A below). At the end of stage 1, up to 15 of 16 slots would idle while the last attempt 1 runs (≈ 20 min at ~28 tok/s): ≈ 5 slot-hours ≈ 3 % of the run. |
| 3 | 35k per attempt; no overshoot at the end | cap = remaining − 4,096, then the remaining tokens; never above the remaining tokens | same |
| 4 | No 12k / 24k caps; final form with the commit reserve | the same | same |
| 5 | 100k window; keep emergency trim and rejection handling | window 100k; the trimmer still exists (budget = 100k − cap − 512); harness `trimmed_requests` 0 | same |
| 6 / 49 | `min_admit_sec` 300 | 600 | **no change** (neutral). A new attempt with 5 min left gets ≈ 8k tokens; few solves come that early. The user can pick 300; the cost is nil either way. |
| 7 / 57 | New `select_attempts_b`: b0 best → b1 best → other accepted → partial → test input; per test input; `valid_grid` | `select_attempts_r5`: every branch's best before any alternate. Bests are grouped by identical output (agreement first), then explicit submit, then the latest. Then the other accepted candidates, the partial programs, the CPU rules and the test input. Per test input, with `valid_grid`. | **ours better.** Proposal 1's own refinement asks for exactly this order. Ours also ranks agreement between the two attempts first. |
| 8 | Harvest at the end with the existing verifier | the same (`_arc2_harvest_r6` + #50), no model request | same |
| 9 | xhigh; log the effective effort | the same (`effective_effort`) | same |

## Phase B — tools, sandbox, recovery

| # | Proposal 1 | The build | Verdict |
|---|---|---|---|
| 10 | On a mid-batch yield: keep the first result and add "skipped" results for the rest | On a mid-batch yield (`ARC3_YIELD_AFTER_BATCH`), the remaining calls of the batch **run**, then the turn yields. Every call has its real result. | **ours better** (no work lost, and the pairing stays valid) |
| 11 | `submit` routed to python with an identifier check | `submit` / `check` / `check_many` are routed. A name argument is checked with the same identifier regex. A code argument runs as the python tool (it is sandboxed either way). | same |
| 12 | Tool contract in the prompt | the same (`B_PROMPT_EDITS`) | same |
| 13 | Autocheck regex incl. `trans`, `candidate` | `^(transform\|solve\|solution\|predict\|candidate)\w*$\|^trans(\d*\|_\w*)$\|^\w+_(alt\w*\|v\d+)$` + `_r6_single_grid_arg` | same (= the safer variant) |
| 14 | sys shim; `setrecursionlimit` 100-3,000, raises above | sys shim (maxsize, recursion get/set, stdout/stderr; no `modules`). `setrecursionlimit` is **clamped** to 20,000 (no exception); time allowed; numpy / helper messages. | **no.** A raise breaks the model's code (it writes `setrecursionlimit(10**6)`); Py3.12 raises RecursionError instead of overflowing the C stack; the child is isolated and limited to 30 s. |
| 15 | Watchdog; 1 restart | Watchdog; at most 2 relaunches, only with ≥ 20 min left; branches wait on `SERVER_DOWN` with no failure counted. The history is client-side, so a lost request is just re-sent. | same (2 vs 1: keep 2) |
| 16 | Fallback 16 → 14 → 12 | 16 → 12 (delayer still off) → the diagnosed fixes → W1 | **no.** A 14 step costs one more cold launch (graph capture ≈ 5-15 min) for ≈ +2 streams. The user chose 16. |
| 17 | Fixed self-test task | the same (`E15B_SELFTEST_TASK`, tokenizer probe) | same |
| 18 | Worker cleanup in `finally` | a worker wrapper that restarts on an error; the running count is released; `worker_errors` reported | same |

## Phase C — smaller fixes

| # | Proposal 1 | The build | Verdict |
|---|---|---|---|
| 19 | Attempt-2 order by `stage2_class` | the same (`rank()`) | same |
| 20 | Rule library optional, never ahead of a verified program | train-exact rules only, after every program candidate | same |
| 21 | Inventory of the defined functions, 20 names | the same, 40 names | same |
| 22 | Bounded-search guidance | prompt | same |
| 23 | `overfit_flags` not used for selection | unchanged (not used) | same |
| 24 | pip / venv timeouts, clamped to the time left | 300 / 1,800 / 120 s (fixed) | **no** (install runs in the first minutes; the give-up timer governs) |
| 25 | Keep the payloads | process | same |
| 26 | One fallback write + one final atomic write | a fallback at start + an atomic write (unique tmp + `os.replace`) after each attempt, kept for crash safety | ours covers it (each write is atomic) |
| 27 | Staging copy stops at the deadline | `parallel_copy(deadline=…)`; the join falls back to the original paths | same |
| 28 | Labels from the actual launch | backend label from the chosen launch | same |

## Phase D — reporting

| # | Proposal 1 | The build | Verdict |
|---|---|---|---|
| 29 | Per-attempt record incl. first transform / first check tokens | `b_events` / `b_budget` / `b_candidate`; `first_transform_tokens`, `first_check_tokens` | same |
| 30 | Cutoff with partial; **rescued by attempt 2** | `at_budget_with_progress`; **no per-attempt correctness** in the eval score | **adopt?** (A2) |
| 31 | Candidate stages | `b_candidate` | same |
| 32 | Cancelled tokens unknown, never estimated | client counts usage only; `server_gen_tokens` (Prometheus) shown beside it | same |
| 33 | Retract counter from real events | `server_args` / `retraction_policy` lines excluded | same |
| 34 | Accepted vs harvested | `accepted_submit`, `harvested_only` | same |
| 35 | Effective effort | `effective_effort` | same |

## Phase E — engine

| # | Proposal 1 | The build | Verdict |
|---|---|---|---|
| 36 | `--min-free-slots-delay 1` | the same | same |
| 37 | mrr 17, graphs 1..17, mamba 68 | the same | same |
| 38 | Gates allow 16; KV gate 0.95; record telemetry loss | the same; a stale KV sample means no hold (the server manages KV itself); **stale samples not counted** | **adopt?** (A4, a count only) |
| 39 | Running histogram; under backlog separately | histogram of the server-log decode lines + the queue > 0 share; **no histogram limited to queue > 0 lines** | **adopt?** (A1) |
| 40 | KV fit report | GPU smoke (`kv_usage`, `mamba_usage` peaks in the gauges) | same |
| 41 | Startup split (load / graphs / ready) | ready time only | **adopt?** (A3) |
| 42 | Engine block | `server_log` block | same |
| 43 | Tests for 12 / 14 / 16 | 16 and 12 (14 not built) | same (scope) |
| 44 | 30-min smoke + a 12-request parity gate | 45-min smoke, 16 tasks; no parity gate | **no.** Parity at 12 requests does not test 16; the smoke's solves, the accept length and Traceback / OOM counts are the gate. |

## Phase F

| # | Proposal 1 | The build | Verdict |
|---|---|---|---|
| 45 | Fair share off; one run per attempt | the same; harness: one run per attempt | same |
| 46 | Fixed window 100k | the same | same |
| 47 | Timeout 3,000 s; **no request after the deadline** (`max(0.5, …)` wrong) | 3,000 s; per-request clamp `max(0.5, min(given, left))` | **no.** After the deadline, a request lives ≤ 0.5 s and ends the attempt as a timeout would; only the CPU harvest follows. A raise path would change the exception handling for ≈ 0 gain. |
| 48 | 3,000-token commit reserve | 4,096 | **keep 4,096** (thinking is on; the commit turn needs ≈ 4-5k) |
| 50 | Code harvest from the reasoning, in the sandbox | the same | same |
| 51 | Keep yield 2,048 in the first build | kept | same |
| 52 | Reuse `stage2_class` | the same | same |
| 53 | Retractions = 0 as a GO criterion | counted in `server_log` | same |
| 54 | Bounded harvest after the deadline | 30-s grace + 90-s join | same |
| 55 | Notebook < 400 kB; facts cap | 247 kB; facts max 800 tokens on 1,000 tasks | same |
| 56 | Build order | done | — |
| 58 | One central config | `E15B` block + `CONFIG["e15b"]` | same |

## Changes worth adopting from proposal 1

All four are **report-only**; they do not change the solver or the engine.

| ID | Item | Change | Why |
|---|---|---|---|
| A1 | #39 | `running_hist_queue_gt0`: the running histogram over the decode lines with queue > 0 | Shows directly whether the 11-of-12 problem is gone: with the fix, no line should have queue > 0 and running < 16. |
| A2 | #30 | In the eval score per task: `b0_correct`, `b1_correct`, `b1_only` (rescued by attempt 2), `b0_at_budget_partial_then_b1` | Measures what attempt 2 adds, the first question of config B. |
| A3 | #41 | `graph_capture_sec` and the load time parsed from the server log, next to the ready time | Graphs 1..17 are new; the smoke must show their cost. |
| A4 | #38 | `kv_stale_samples` count | Shows if the KV gate ran blind. |

## Not adopted, with reasons

| Item | Reason |
|---|---|
| #2 | Completion barrier (idle slots); **D-A** below |
| #6 / #49 | 300 s (neutral) |
| #10 | Skipped results (ours runs the calls) |
| #14 | Recursion raise at 3,000 |
| #16 | 14-stream step |
| #24 | Timeout clamp |
| #44 | Parity gate |
| #47 | No-request-after-deadline |
| #48 | 3k reserve |

**D-A (the user's to confirm):** "attempt 1 on all tasks first" is built as *all started first*. Proposal 1 reads it as *all finished first*. Built = started.
