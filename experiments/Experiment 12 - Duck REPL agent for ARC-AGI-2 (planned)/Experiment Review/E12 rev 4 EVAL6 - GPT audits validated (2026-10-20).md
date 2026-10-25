# E12 rev 4 EVAL6: three GPT audits validated (2026-10-20)

Audits:
- **A**: "Strict Pre-GPU Audit, NO-GO".
- **B**: "Strict Pre-run Audit, NO-GO: blocking issues".
- **C**: "Strict Pre-run Audit, conditional GO".

All three reviewed `E12_REV4_EVAL6_AUDIT_FOR_GPT_2026-10-20.zip`. Every claim below was checked in the vendored Duck source.

The user's rule: ARC-AGI-2 is not ARC-AGI-3, so change what this competition needs. Not so few changes that the test is worthless, not so many that it breaks.

## Claims and verdicts

| # | Claim (by) | Code check | Verdict |
|---|---|---|---|
| 1 | The 900 s read timeout is too short for a 60k reply (A, B, C) | **True.** Non-streaming (`openai_compat.py:52`, `stream=False`). A ReadTimeout is not retried (`tool_agent.py:628`); Duck yields (`:7106`): completed exchanges are kept, the in-flight reply is lost. Decode of 6 x 60k at 368 tok/s takes 978 s | **Fixed**: EVAL6 only, 1,800 s (the audits suggested 1,500; extra margin for slower long-context decode). `min(timeout, remaining)` still bounds it by the solve deadline |
| 2 | The 100k per-branch cap is soft, up to about 160k (A, B, C) | **True.** Checked after each request | Kept soft. The report uses actual tokens; 1.2 M is not a ceiling |
| 3 | B: clamp `max_tokens` to the remaining budget (second request 40k) | Would work | **Rejected.** It cuts reasoning, the very thing this test measures (EVAL30: 29/30 first requests cut at 12,288). A and C accept the soft cap |
| 4 | Input budget 70,560, effective drain 35,280 at window 131,072 (A, B, C) | **True** (`tool_agent.py:3505, 6471`) | Accepted as a caveat (row 6) |
| 5 | B: drain 11,168, so the trim target is 59,392 | Duck's own rationale (`tool_agent.py:228-242`): a small drain trims on almost every turn, and each trim invalidates the whole prefix cache. Re-prefill was the main EVAL30 loss (9.6 M tokens). With 60k replies, the block size dominates: the older reply is dropped at either target | **Rejected** |
| 6 | Trimming is an interpretation caveat, not a code change (C, A similar) | The newest block is always kept (`preserve_recent=1`) and the task is never dropped. Two 60k replies fit no budget the server allows (155,648) | Report the trims |
| 7 | All 12 branches may not get compute in 60 min (A, B, C) | **True.** 12 x ~120k = 1.44 M tokens; decode at 368 tok/s takes 65 min, plus about 14 min prefill. With 6 live there are two waves, and the second wave got half the time | **Changed**: solve window 90 min, total budget 125 min (+ about 1 h of quota). The audits kept 60 min and only flagged exposure; but an unexposed half makes a 0-1 result ambiguous, which would force a rerun |
| 8 | `max_tokens` 60,000 is a ceiling, not a minimum thinking (A) | True | Wording fixed (no "60k floor" claim) |
| 9 | The 6 tasks were selected, so the result is not an unbiased accuracy (A, B) | True | Stated in the analysis |
| 10 | Count fully-solved tasks from `eval_unit_records` (B) | There is no such field in the report | Analysis only |
| 11 | Mark branches that never ran (A, C) | Turns and tokens are in the branch records | Analysis only |
| 12 | Gate + live pool deadlock (A, B, C) | No deadlock; setup failure is counted; held slots are released in `finally` | PASS |
| 13 | Server context and 6 streams (A, B, C) | C's 881,336 is the E11-04 KV figure; EVAL30 measured 866,744. 6 x 131,072 = 786k fits either | PASS |
| 14 | The fake harness does not prove long requests (A, B, C) | True | Accepted. No new slow-request test. A unit check proves that 1,800 reaches `run_branch` |

## Checked here, not covered by the audits

- **Yield at 2,048 tokens** (request question 2). The check runs only between requests (`tool_agent.py:6718-6731`), so a 60k request is never cut. Behaviour is as at 12,288: a yield after each request.
- **`ARC3_HTTP_RETRY_INITIAL_SECONDS` 900** is the cold-start connect grace, not a read timeout.
- **Rolling summaries** are off (`ARC3_SUMMARY_INTERVAL_TOKENS` unset = 0), so B's summary-token note does not apply.
- **SGLang abort on client disconnect** was not verifiable locally. Either way, a longer timeout is the safe side.

## Reverted (my own earlier change, not asked by any audit)

- **Change:** Duck window 154,624 with 5 live.
- **Now:** back to the milestone 131,072 with 6 live / 6 streams, the audited state.
- **Why:**
  - with 60k replies it keeps no extra reply;
  - it cost one live branch;
  - it left only about 1k of margin to the server context.
- The `E12_DUCK_CONTEXT_WINDOW` hook was removed from `arc2_duck.py`.

## Final EVAL6 settings vs the audited zip

| Knob | Zip | Now |
|---|---|---|
| Read timeout | 900 s | **1,800 s** (`E12_DUCK_REQUEST_TIMEOUT`, EVAL6 only; unset = 900) |
| Solve window / total budget | 60 / 100 min | **90 / 125 min** |
| Output per request, cap, live, streams, window | 60,000; 100k soft; 6; 6; 131,072 | unchanged |

## Local evidence (after the changes)

- **EVAL6 harness: 25/25**:
  - knobs 6/6/60000/1800;
  - Duck env window 131,072;
  - at most 6 in flight;
  - first 6 conversations = first 3 tasks x 2.
- **SMOKE1 regression** (knobs unset): 22/22.
- **Adapter unit checks: 41 OK**, including two new ones:
  - timeout 1,800 reaches both branches;
  - unset gives no override, with the default staying 900.
- **Notebook shas:**
  - EVAL6: `53a66856f330aa07a9634059a4d9c0e1d54b6a19743d6ba35a1e411da0f44b74`;
  - SMOKE1 `0fea4d0c…` and EVAL30 `b65fb9dc…` are untouched.

## Analysis plan (no notebook change)

1. **Fully solved per task** (all test inputs hit), from `eval_unit_records`, plus input-level hits.
2. **Exposure per branch:**
   - turns and generated tokens;
   - branches that never started or ran under 1 request: incomplete exposure, not model failure.
3. **Timeouts:** count `request_timeout (yielding` in the transcripts.
4. **Trims:**
   - from `usage`, a prompt that is smaller than the previous one;
   - count `context_overflow_recovered`.
5. **Actual tokens per branch** against the 100k soft cap. 1.2 M is not reported as a ceiling.
6. **Decision rule** on fully-solved tasks among the exposed tasks:
   - 2-3 or more of 6: full rev 4 EVAL30;
   - 0-1 with full exposure: back to the Poetiq line;
   - 0-1 with poor exposure: inconclusive (state why).
7. **Serializer tampering check** (final audit, GPT 3): grep every accepted program for `json` / `builtins` / `sys.modules` patches.
   - A forged acceptance can only stop a branch early.
   - It cannot raise the score, which is checked against the truth.

## Final audit (2026-10-20): 2 GO, 1 NO-GO

**GPT 3's blocker:** a `json.dumps` patch forges the verifier result.

**The same finding was validated on 2026-10-19** (`E12 step 3.5 rev 2 - GPT re-audits validated`): valid and reproduced, but no new capability. It stays **not fixed**, for these reasons:
1. The score is checked against the ground truth, so a forged acceptance never adds a solve.
2. It gains nothing over a train-output lookup function, which is accepted the same way because the train outputs are in the prompt.
3. Measured in E12 EVAL30 (30 tasks, 31 branch transcripts with code): the model never imported or used `json`, and never touched `builtins`, `sys.modules` or `HOST_STDOUT`.
4. The proposed guard covers `json.dumps` only; `json.encoder` and `JSONEncoder` stay reachable.

**Instead:** analysis item 7 above.
