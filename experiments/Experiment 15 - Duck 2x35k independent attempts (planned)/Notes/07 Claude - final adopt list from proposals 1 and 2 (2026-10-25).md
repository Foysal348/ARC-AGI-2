# 07 — Final adopt list from the two AI proposals (Notes/04, Notes/06)

**Date:** 2026-10-25.

**Background:**
- There are two proposals, not three (user).
- Notes/05 compares proposal 1 item by item.
- Proposal 2 covers the same 1-55 items. Where it differs from proposal 1 or from the build, the difference is checked below against the code or measured.

Nothing has been changed yet.

## Proposal 2 points checked against the code / measured

| Point (proposal 2) | Check | Result |
|---|---|---|
| #5: window ≤ the server context length | `SGL_CONTEXT_LEN = 155648` (notebook line 260) | 100k fits; no change |
| #15: a long non-streaming request must not look like a dead server | SGLang 0.5.21 `http_server.py:699-777`: `/health` = a 1-token generation check. It passes when any detokenizer output arrived in the last 20 s (`SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` default True). Non-streaming requests emit an output every 50 tokens (`SGLANG_FORCE_STREAM_INTERVAL` 50). The watchdog needs 180 s of consecutive failures. | Already a decode-progress check; a busy healthy server passes. No change. **A fault-injection harness test is still missing** (A7). |
| #20: measure the rule library first; drop it if the gain is 0 | `rule_library_outputs` on the public data | **eval 120 tasks: 0 fired, 0 solved. Training 1,000: 24 fired, 23 solved.** Hidden ≈ eval difficulty, so the expected gain ≈ 0. A risk was found: `_rl_color_map` runs outside the per-rule `try`, and an exception in `select_attempts_r5` makes `build_submission` fall back to the test inputs for the whole task, **losing its program candidates**. |
| #10: drop the unexecuted calls from the assistant message | the build runs the remaining calls | ours keeps more; no change |
| #1: per-attempt seeds | Duck sends no seed | not needed for independence; no change |
| #16: B14 tier | — | both proposals ask for it. Not taken in-run; if the 16-stream smoke fails, the next build is 14 (smoke-driven). |
| #2: completion barrier | — | both proposals ask for it → **D-A** (the user's call) |
| #49: min_admit 300 | — | both proposals → taken (A5) |

## Final adopt list

| ID | From | Change | Kind |
|---|---|---|---|
| A1 | P1 #39, P2 #39 | A running histogram over the decode lines with queue > 0. A time-weighted mean of `running` from the 30-s gauge samples where running + queue ≥ 16 (the time share at 16). | report |
| A2 | P1 #30, P2 #30 | Eval score per task: `b0_correct`, `b1_correct`, `b1_only` (rescued by attempt 2), `b0_at_budget_partial` → rescued | report |
| A3 | P1 #41, P2 #41 | `model_load_sec`, `graph_capture_sec` from the server log, beside the ready time | report |
| A4 | P1 #38 | `kv_stale_samples` count | report |
| A5 | P1 #49, P2 #49 | `min_admit_sec` 600 → 300 | config |
| A6 | P2 #20 (measured) | The rule library inside its own `try` in `select_attempts_r5` (an error → no rules, never a lost program). Docstring corrected with the measurement (eval 0/120, training 23/1000). | hardening |
| A7 | P2 #15, step 10 | Harness fault injection: (a) kill the fake server mid-solve → watchdog relaunch → run ends with a valid submission; (b) a slow but healthy server → no relaunch | test |

## Not taken

| Proposal | Reason |
|---|---|
| Completion barrier (#2) | D-A: idle slots at the stage boundary |
| Seeds (#1) | Not needed for independence |
| Skipped-call results / dropping calls (#10) | Ours runs the calls |
| Recursion raise (#14) | Clamp to 20,000 instead |
| B14 in-run tier (#16) | Smoke-driven instead |
| Timeout clamp for setup (#24) | Setup runs early |
| Parity gate (#44) | The smoke is the 16-request stress test |
| No-request-after-deadline (#47) | ≈ 0 gain |
| 3k reserve (#48) | Keep 4,096 |
| SearchBudget class (#22) | Prompt guidance kept |
| HARVEST_ONLY phase (#54) | The 30-s grace + 90-s join do this |
| Central dataclass config | The `E15B` block + `CONFIG["e15b"]` exist |

**D-A:** both proposals read "attempt 1 on all tasks first" as *all attempt 1 finished*. The build uses *all attempt 1 started*; each task's attempt 2 is queued only after its own attempt 1 ends.
- **Cost of a finish barrier:** at the stage boundary, up to 15 of 16 slots idle while the last attempt 1 runs (≈ 20 min at ~28 tok/s). That is ≈ 5 slot-hours, ≈ 3 % of the run.
- **Recommendation:** keep the start barrier.

**After the go:**
1. Build A1-A7 (+ D-A as decided).
2. Re-run the unit tests, `verify_items.py` and the harness in the 3 modes plus the new fault scenarios.
3. New shas; Notes/03 updated.
4. No commit or push.
