# Claude's validation of the dual-answer notebook (2026-10-12)

**Notebook:** `Q27-XHIGH-v1_4_4_DUAL_ANSWER_EVAL.ipynb` (sha256 `42124213…`), built by GPT from the v1.4.4 base.

## Diff against the base (cell by cell)

**6 of 9 cells are byte-identical** to v1.4.4:
- markdown header;
- cell 0 (mode + fallback);
- cell 2 (install);
- cell 3 (prompts);
- cell 4 (parser);
- cell 5 (engine).

**Changed cells:**
- **Cell 1 (configuration):** adds `SECOND_BRANCH_*` settings. ON by default in EVAL12; **OFF by default in the hidden rerun**.
- **Cell 6 (scheduler):** the original scheduler is unchanged. After it ends, a new `run_second_answer_stage()`:
  1. builds a token-exact prefix for each finished unit: natural close = prompt + phase-1 text up to `</think>`; forced close = prompt + phase 1 + `STOP_IDS`;
  2. samples one new answer per unit (seed = `SEED + 10*UNIT_INDEX + 3`, ≤ 8,192 tokens, at most 8 at a time, ending 90 s before HARD_STOP);
  3. replaces **attempt_2 only**, and only when the grid is valid and different from attempt_1.
- **Cell 7 (report):** adds `primary_baseline_hits`, `dual_answer_hits`, `second_answer_marginal_hits`, `second_answer_only_units`, `baseline_fallback_lost_units` and `primary_unchanged_all`.
- **Scoring note:** `eval_query_hits`, `eval_score_pct` and the "EVAL SCORE" line now report the **dual** score. The v1.4.4-comparable figure is `primary_baseline_hits`.

## Tests run by Claude

| Test | Result |
|---|---|
| GPT's `test_q27_dual_cpu.py` (paths adapted) | 8/8 categories PASS |
| pyflakes over all code cells joined (shared kernel globals) | no real errors: only unused imports, the same as v1.4.4, plus one unused `global` |
| Experiment 1 harness (`harness_q27x.py`: real Qwen tokenizer + chat template, fake vLLM step-API engine), scenario `mixed` | PASS; primary 16/22, the same as v1.4.4 |
| same harness, `drain` (deadline drain + hard stop) | PASS; second stage correctly skipped (`already_past_second_answer_deadline`); primary unchanged |
| same harness, `rerun` (259 hidden units) | PASS; second branch disabled; the result is identical to v1.4.4 behaviour |
| harness with phase-3 support added (`harness_q27x_dual.py`: second answer = the correct grid) | 22/22 second answers generated; 6 marginal hits; `primary_unchanged_all` true; `baseline_fallback_lost_units` empty. The harness's final assert fails **by design**: it expects the primary-only score, while `eval_query_hits` is now dual. |

**Harness limitation found:** the original harness fake engine has no phase-3 script. The unmodified harness therefore produced 14 `add_request` `KeyError`s, which the notebook caught safely (primary intact). The patched harness fixes this.

## Caveat on expected benefit (not a bug)

- The second answer is re-sampled from the **same** reasoning; only the text after `</think>` is new.
  - **Natural closes:** the answer is largely determined, so the second grid will often equal the first.
  - **Forced closes** (19/22 on EVAL12): the second grid may differ, but forced-close answers were about 2 % correct in runs 4, 6 and 7.
- The cross-run evidence (run 4 ∪ run 6 = 3/22) came from **independent full reasoning**, which this design does not do.
- The Codex-notebook gain (+7.5) comes from another model and method (Qwen3-4B, beam candidates).
- **Expected EVAL12 marginal hits: about 0–1.** The run still measures how often the same reasoning yields distinct grids, and at what GPU cost.

## Minor

- The markdown header still says v1.4.4.
- Push to a **new kernel slug** (for example `q27-xhigh-dual`) so the LB-26.94 `q27-xhigh` kernel is not overwritten.
