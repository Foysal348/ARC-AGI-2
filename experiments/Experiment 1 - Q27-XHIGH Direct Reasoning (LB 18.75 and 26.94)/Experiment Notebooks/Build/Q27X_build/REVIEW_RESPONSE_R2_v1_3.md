# Q27-XHIGH v1.3 — response to GPT review R2 (which reviewed v1.2), with a true / false ledger

v1.2.1 (re-checked against the discussion: 65,536 total tokens, `fp8_e4m3`) came between v1.2 and this version. All of
R2's points also applied to v1.2.1. Each claim was checked before changing anything.

- Notebook: `notebooks/Q27-XHIGH-v1_3.ipynb`.
- Test outputs: `build/harness_output_v1_3.txt`. These pass:
  - the production-defaults check;
  - 9 scenarios: mixed, drain, rerun, crash, eos, vdeadline, allfail, smokefail, norelease;
  - the engine ladder.

| # | R2 claim | verdict | evidence | fix |
|---|---|---|---|---|
| 1 | EVAL12 drain (162 min) came after the hard stop (155 min), so the drain never ran | **TRUE** | 0.90 × 180 = 162 > 180 − 25 = 155. The harness missed it because it overrides the deadlines | Deadlines derived in order: `DRAIN = min(frac, HARD − 8 min)`, `ADMIT = min(frac, DRAIN − 7 min)`, with an assert in cell 1. Production values: EVAL12 140/147/155/172 min, RERUN 630/666/695/712 min (`check_defaults.py`) |
| 2 | A ladder attempt counts after the constructor only; a fp8 / MTP failure at the first decode is not caught | **TRUE** | v1.2 cell 5 | Each attempt runs a real 16-token decode (`_engine_smoke`) before use; a failed attempt's engine core is shut down. Harness `smokefail`: fp8 engine starts, decode fails → falls back to bf16 KV + MTP |
| 3 | `_wait_gpu_release` result ignored | **TRUE** | v1.2 returned seconds only | Returns `{released, sec, last_used_mb}`; if not released, no further attempt (`fatal`). Harness `norelease` |
| 4 | All attempts failing → `raise` → failed notebook | **TRUE** | v1.2 cell 5 | No raise: `ENGINE_READY=False`, the scheduler is skipped, the identity submission is kept, the report is written. Harness `allfail` |
| 5 | Projection undercounts setup; failed-attempt time was missing; a projection could come from an incomplete run | **TRUE** | v1.2 `load_sec` summed only successful attempts | Setup = measured `scheduler start − notebook start`; every attempt records `sec`; `projection_valid` only if the run is complete with no errors and no skipped tasks |
| 6 | Concurrency heuristic 40k vs ~61–65k trajectories | **Reasonable** (performance risk, inferred) | — | 55k per live sequence (GPT: 55–60k). With fp8 KV the KV capacity still allows 16. The report gives live-token p50/p75/p90/max to tune the hidden run |
| 7 | Helper defined after `solution()` is lost | **TRUE** (reproduced) | v1.2 variants used only earlier blocks | Added an "all code blocks" variant; self-test layout `helper_after` (9 layouts now) |
| 8 | MTP acceptance not persisted | **TRUE** | v1.2 bundle had no vLLM stats | `llm.get_metrics()` snapshots every 2 min + final → `q27x/vllm_metrics.json`. The report holds MTP drafted / accepted / acceptance rate and preemptions. The vLLM front-end log goes to `q27x/vllm_frontend.log`. Source: `v1/engine/llm_engine.py` 318–329, 374–376; metric names in `v1/spec_decode/metrics.py` 155–173 |

## Accepted from R2 without change
Per-query scoring, adaptive thinking cap, verification deadline, temperature 1.0, MTP 3, fp8 KV ladder concept.

## Still deferred (scope)
Verifier-feedback repair turn, NVARC fusion, import blocklist. Decide after the commit-run measurements.

## Gate for the hidden run (as R2 proposed)
The commit run (EVAL12) must show all of the following:
- 12/12 tasks finished;
- no engine or scheduler error;
- the fp8 decode smoke passed (or a recorded fallback);
- MTP acceptance;
- aggregate tok/s;
- p50/p90 tokens;
- preemptions;
- the per-query score;
- a valid end-to-end 240-task projection under ~11.3 h.

## Addendum v1.3.1 — the same R2 review re-sent for the v1.2.1 bundle (points 1–6 and 9 were already fixed in v1.3)
| # | claim | verdict | fix |
|---|---|---|---|
| 7 (refined) | the "all code blocks" variant lets the LAST `solution()` win, so an earlier target never runs with its helpers | TRUE | variant = target `solution()` + every non-solution block in original order; self-test `two_solutions_helper_after` (10 layouts) |
| 8 | the rerun harness never proved that a NON-identity prediction reaches submission.json (fake answers equalled the inputs) | TRUE | fake hidden-test answers are row-reversed grids; the harness asserts that every model-sourced first attempt is written to submission.json (169 non-identity queries) |
| 10 | "EAGLE 3 steps = MTP 3" was worded as an exact equivalence | TRUE (wording) | DISCUSSION_CHECK now says: closest vLLM-native counterpart, not an exact equivalent |

## Addendum v1.3.2 — GPT final review (verdict: GO for EVAL12)
| claim | verdict | fix | test |
|---|---|---|---|
| projection_valid ignored hard-stop aborts, drained / unadmitted tasks and cut verifications; hours were computed even when invalid | TRUE | all four now make it invalid; hours set to None when invalid | harness `drain` and `vdeadline` assert projection invalid; `mixed` asserts valid |
| a train-exact variant that crashes on a test input stopped the search, so a later working variant was never tried | TRUE | prefer a train-exact variant that also runs on every test input; stop only on train-exact + test-valid | cell-4 self-test `_v6` (crash-on-test variant A, working variant B → B used) |
| re-prefill after a forced close (prefix caching off) | Acknowledged, deliberately unchanged | measured in EVAL12 (forced-close rate, tok/s) before any change | — |
