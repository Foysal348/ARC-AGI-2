# Fix list: E31 FN DIRECT30 and NB-S2 rebuild (design audits, 2026-11-01)

Rule (user): collect all 3 audits, then build one list, then dispatch ONCE when the user says. Fix blockers only.

## Audit #1: E31 design, NO-GO (GO after 3 fixes)
| # | Severity | Item | Verified | Fix | Tests |
|---|---|---|---|---|---|
| E31-1 | HIGH | Multi-test parser: the fallback "last N valid blocks positional" can put a grid on the wrong test when labeled and unlabeled blocks are mixed | TRUE: design section 4 | Labeled blocks present: map by label only, with label in 1..N and a valid grid, last occurrence wins. No labels: positional only if the count of valid blocks == N. Any test still missing gets the ORDER2 fallback (test input / mirror) for that test only. Never guess | 2-test / 3-test, reordered labels, duplicate labels, missing labels, extra example blocks, truncated final block |
| E31-2 | HIGH | DIRECT records go through E28's Duck `build_submission` / `select_attempts_r5`, which treats accepted=False as partial candidates; under `E12_B_MODE=1` it also adds train-exact CPU rule outputs. So attempt_1 = model grid is not guaranteed | TRUE: E28 hook-point report section 7 | Keep the E28 writer and CELL 7. Write the exact record (branch 0, submissions[{test_outputs, accepted False, train_exact []}]) and assert `E12A.build_submission({tid: ch}, {tid: [record]})[tid] == expected_direct_attempts`. If it differs, add a small E31 selector adapter (ORDER2 make_attempts per test) | single-test, 2-test, 3-test, partial grid, no grid |
| E31-3 | HIGH | Drain timing: phase 1 cut at the solve end leaves no time for phase 2 | TRUE: `e21_cell6` `_end` is the solve end; ORDER2 had a separate DRAIN = HARD_STOP - 8 min | `DRAIN_START = SOLVE_END - answer_reserve`. The reserve comes from ORDER2's own forced-close timing: 8,192 tokens at ~18 tok/s per stream is about 7.6 min; N tests take longer. At DRAIN_START, abort phase 1 (keeping its ids) and start a forced-close phase 2 that finishes before SOLVE_END | fake streaming server with 24 simultaneous phase-1 streams: ids kept, prefix token-exact, phase 2 admitted, phase 2 done before the end, grids in the submission and the report |
| E31-4 | info | Timeline: 104k x 24 streams at about 17.9 tok/s per stream is about 97 min per trace, so second-wave coverage is uncertain | — | Report how many tasks reach 60k / 78k / 104k (record-only) | — |

## Audit #2: E31 design, NO-GO (GO after the same 3 fixes)
Same three HIGH items as audit #1, which confirms E31-1, E31-2 and E31-3. New details:
| # | Item | Verified | Fix / test |
|---|---|---|---|
| E31-1+ | If any recognized label is present, the positional fallback is forbidden. `grid K` goes to test K only. Positional parsing applies only when there is no label and exactly N blocks. Single-test keeps the ORDER2 parser unchanged | TRUE (design section 4) | Regressions: N=2 and N=3; missing first label; missing last label; duplicate labels; reversed labels; partial truncated block; extra intermediate block. Test K's grid never goes to another test |
| E31-2+ | The adapter must also give `E12A.SCHED_LAST` a compatible scheduler summary (E28 reads it after solve_all_r6), plus `on_branch_done` with one completed branch record per task. The E31 report scores the parsed DIRECT grids directly. Attempt semantics = ORDER2 (model grid, test input, mirror) | TRUE: e28k100_code_export:4274-4292 (call), `E12_RUN["rev6_scheduler"] = dict(E12A.SCHED_LAST)` | 3-test task with mixed success, one missing grid, one model grid equal to the input. Run the real `E12A.build_submission` and the CELL 7 scorer; expected == actual |
| E31-3+ | Phased drain inside the solve window: no new phase-1 admission from T-12 min; abort and drain running phase 1 from T-8 min; phase-2 answer deadline T-1 min; the last minute is for writing. Set the reserve from a 24-stream phase-2 timing in the fake e2e. Never cut the 60k floor as a quality gate | TRUE (ORDER2 has ADMIT = DRAIN - 2 min, DRAIN = HARD_STOP - 8 min) | 24 phase-1 streams running until the deadline, then 24 forced-close requests: answers persist, no orphan requests, the budget is respected |
| E31-5 | Token-exact continuation: correctness must not depend on a prefix-cache hit. Never decode then re-tokenize. Keep the streamed ids as they are; test stripping of a trailing EOS, `</think>` handling and the forced-close prefix rebuild | TRUE: ORDER2 is id-based | Unit test on ids only; a no-cache fake still gives the same prefix |
| E31-4 | Info: 30 x 104k = 3.12 M tokens / 419.5 tok/s is about 124 min of pure generation, which fits 170 min in theory but is UNVERIFIED. Preemptions are expected (24 x 104k does not fit in KV) and accepted, as in E28 | — | Report coverage at 60k / 78k / 104k (record-only) |

## Audit #3: E31 design, NO-GO (GO after 4 fixes)
| # | Severity | Item | Verified | Fix / test |
|---|---|---|---|---|
| E31-6 (NEW) | HIGH | The E18 shim targets its own server: `E18_PORT` defaults to 8011 (e18_engine_cell.py:17-18), and `E18Engine._ensure_server` (246-273) relaunches the server itself. E28's server is `SGL_URL` (:8001), and E28 owns the lifecycle (E15B watchdog + `e21_relaunch`). Used as is, the shim sends requests to the wrong port and two owners would relaunch the server | TRUE (source lines read) | Take from E18 only the streaming token-id client behaviour (`/v1/completions`, `return_token_ids`, abort by closing the stream). `base_url = SGL_URL`, asserted to equal the booted server's URL. The E18 relaunch is not used: one lifecycle owner, which is E28. A request error is never counted as a finish. Fake-server tests: the request goes to 8001, never 8011; streamed ids are exact; abort closes the connection; on a server-kill injection, E28 relaunches once at most and the error is not reported as success. vLLM 0.32 supports streaming `return_token_ids` (docs v0.32.0 completion protocol) |
| E31-2++ | HIGH | Stronger form of E31-2: install a DIRECT-only adapter `e31_build_submission`, with `E12A.build_submission` rebound inside the E31 notebook only. It uses only the DIRECT solver's recorded per-query attempt_1 / attempt_2 (ORDER2 make_attempts per test). No train-exact CPU rule, no harvested program, no Duck candidate. `E12A.SCHED_LAST` is populated by the direct scheduler | TRUE: audit #3 shows that `score_from_generated` scores whatever selector output it is given (0 % vs 100 % when injected) | Regression: DIRECT wrong while the CPU train-exact candidate is right -> the E31 score must show DIRECT wrong. Per query, the attempt source and the final grid agree |
| E31-3++ | HIGH | Drain reserve: a 3-test answer budget of 8,192 x 3 tokens at about 18 tok/s per stream is about 23 min, so ORDER2's 8 min is not enough for every multi-test task. Phase 2 that has not finished by hard_end is aborted and its partial text parsed (ORDER2 hard-stop) | TRUE | `drain_at = hard_end - drain_reserve`. Put `'drain_min'` in the E31_CFG line; the default covers the largest live answer. The build-time assertion checks the window math. Regression: a fake stream never emits `</think>`; the drain sends STOP_IDS exactly; scoring never starts before phase 2 ends or aborts |
| E31-7 (NEW) | severe (budget) | 15 min boot + 170 min window = 185 min before any reserve, so "3 h at most" is wrong. `nb5_phase` keeps extra margin, so a 180-min profile shrinks the window or gives SKIPPED_BUDGET | TRUE (nb5_phase window = min(fixed, HARD_STOP-180 - now - 60)) | Notebook budget of about 205 min (deadman to match) for a 170-min window, or a smaller window if a hard 3 h limit is wanted. Build-time assertion on the final window |
| E31-1++ | HIGH | Parser, same as audits #1 and #2. Each record also carries `query_index`, `mapping_source` (label / positional / fallback) and `fallback_reason` | TRUE | Cases: 2 and 3 tests, reordered labels, repeated labels, missing label 2, extra reasoning fence, truncated final block, duplicate grids, malformed grid. Before gold scoring, assert query_index -> grid |
| E31-8 (NEW) | measurement | The "5 uniquely DIRECT tasks" were partly partial: 446ef5d2 and cbebaa4b were ◐ (partial) for 27B-Direct; only 7b5033c1, a47bf94d and a6f40cea were fully solved | TRUE (table rows 33, 60, 85, 86, 99) | The E31 report keeps query_hits, fully_solved_tasks and task_normalized separate, per group |

## CONSOLIDATED E31 FIX LIST (to dispatch once, to the E31 builder agent, when the user says)
1. **E31-1 parser:** label-only mapping when any label exists. Positional mapping only with no labels and exactly N valid blocks. Per-test ORDER2 fallback. Record `query_index` / `mapping_source` / `fallback_reason`. Full regression set.
2. **E31-2 DIRECT-only adapter:** `E12A.build_submission` -> `e31_build_submission`, using only the DIRECT attempts (ORDER2 make_attempts per test). Populate `E12A.SCHED_LAST`. One completed branch record per task through `on_branch_done`. The real CELL 7 scorer runs over the adapter. Regressions include "DIRECT wrong + CPU rule right -> scored wrong".
3. **E31-3 phased drain inside the window:**
   - no admission after T-12 min;
   - `drain_at = T - drain_min`: abort phase 1 and keep its ids, then forced-close phase 2;
   - abort phase 2 at T-1 min and parse its partial text;
   - the last minute is for writing.
   `'drain_min'` goes in the E31_CFG line, sized for N x 8,192 tokens at the measured rate. No 60k-floor quality gate.
4. **E31-5 ids only:** never decode then re-tokenize; continuation does not depend on a prefix-cache hit (test with a no-cache fake).
5. **E31-6 client:** only the E18 streaming token-id client behaviour. `base_url = SGL_URL` (asserted). No E18 relaunch; E28 owns the lifecycle. A request error is never a finish.
6. **E31-7 budget:** about 205 min total for a 170-min window (deadman to match), with a build-time assertion on the window.
7. **E31-4 / E31-8 report:** coverage at 60k / 78k / 104k; query_hits, fully_solved and task_norm per group; multi-test tasks shown separately.
8. **e2e kit:** a fake streaming server with 24 parallel streams covering:
   - all 30 tasks;
   - partial grids;
   - thinking that never closes;
   - the forced close;
   - a server kill;
   - the real E28 writer / CELL 7 / report.

## NB-S2 rebuild design
No audit has been received yet (the zip went out at the same time). Pending.
