# Q27-XHIGH v1.4 / RUN4 — Independent Deep Audit
**Review date:** 2026-10-06  
**Bundle:** `Q27X_v1_4_RUN4_FOR_GPT_2026-10-06.zip`  
**Sequence:** `README.md` → source notebook → executed notebook / records / output → RUN3 comparison → public scoring rules and task GT for limited diagnostics → **only then** `OUR_ANALYSIS.md`.

## Executive verdict
**NO-GO as a performance-upgrade candidate for a 60% leaderboard target without additional experiments.** The notebook successfully executes its 22 EVAL12 queries and implements a usable fast engine, but it obtains only **2/22 = 9.09% exact query hits**; both hits are among the four requests in which thinking ended naturally. The dominant problems are solver quality, use of almost all compute on unsuccessful 49k-token reasoning, hand-transcribing grids, and no real second candidate. A successful execution is not proof that it is competitive against the team's reported 32.64 LB baseline (different sample and leaderboard).

## Scope, verified facts, and limits
- `notebook/Q27-XHIGH-v1_4.ipynb` SHA-256 matches README: `aa27d6e1ac27320f676337d6a8ca05bdb8ac3f1123ba60c9ac9d93e0c5669137`.
- RUN4 `run_report.json`, 22 `records.jsonl` entries, and 22 answer text files are mutually consistent: 12 tasks; 22 queries; 4 natural, 18 forced; 20 parsed; 2 hits, both `attempt_1`; 0 engine preemptions; no scheduler error; compiled Python source passes syntax parsing.
- RUN3 is **per task** (12 requests, 22 answers); RUN4 is **per query** (22 requests). Other changes include sampling temperature 1.0→0.6 and code+JSON→grid-only. Therefore **1/22→2/22 is not an isolated causal A/B result**.
- EVAL12 query-count distribution: three tasks with 1 query, eight with 2, one with 3. The hidden test in the bundle is 240 tasks / 259 queries. A 12-task sample with 22 queries is not task-count representative, and has a large uncertainty range.
- Actual source of Kaggle competition metric: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/evaluation — exact output **per test query**, max of two attempts, 12 h GPU, internet off. The `eval_score_pct` (query-weighted) is aligned with the metric; `eval_task_normalized_pct` is secondary, not the Kaggle metric.
- External public evaluation ground truth was consulted solely to inspect diagnostic cell differences, **not for model prompting or test-time rule construction**. It must never be injected into the competition submission logic.
- The referenced ~60 score with this model is **not independently established by this bundle**. NguyenThanhNhan's 23.33/22.50 and russcore's 10.14/7.78/8.61 are supported by included screenshots; no causal equivalence of hardware, inference serving, prompting, task distribution, or attempt strategy is established.

## Root-cause ranking (expected accuracy impact, qualitative)

### P0 — Reasoning spends ~49k tokens without a validated transformation
**Evidence:** 18/22 forced closes generated full 49,152-token first phase. Only 4 naturally terminated; these scored 2/4. All 18 forced closes scored 0/18. Some thinking tails still debate hypotheses at the cap; others are calculating exact grid coordinates. This is association and **not** proof that cutting reasoning shorter would improve accuracy (harder tasks may simply be more likely to hit the cap).

**Fix:** Implement a staged, train-verified solver mode before making it the default: (1) infer a transformation as a pure function, (2) execute it against **every training input** and compare exact outputs, (3) apply to test inputs if verified, (4) reserve a smaller independent second candidate / repair path. Avoid asking the model to manually write 300–900 cells when it can produce an algorithm. Require reproducibility and time/CPU/memory limits for execution; never run arbitrary generated code in the notebook's GPU process. Test this against untouched hold-out tasks, not by iteratively tuning to the same public evaluation answers.

**Acceptance gate:** fraction train-verified; actual **test exact pass@1 and pass@2**; seconds per attempted task/query; number of task outputs completed by 12h; malicious/infinite-loop code sandbox resilience. Fit-on-train alone is not proof of a general rule.

### P0 — The second allowed prediction is not a model attempt
**Evidence:** `CELL 4 -> make_attempts()` picks model, then input grid or its mirror. RUN4 shows `eval_query_hits == eval_pass1_queries == 2`; none of 22 second attempts adds a correct solution. Kaggle explicitly allows two distinct predictions for each output.

**Fix:** Keep a per-query candidate bank. Always preserve the first answer, even if a later pass fails. Schedule distinct second model outputs only after sufficient first-pass coverage, and validate the second candidate's grid format. Do not replace a valid first prediction with a speculative second. For cheap second passes, do a targeted output audit/reconstruction or second seed/strategy; do not assume that an independent 49k second rollout fits.

**Acceptance gate:** count of distinct valid second predictions, pass@1, **incremental** pass@2 hits, token/time overhead, coverage. A second answer identical to first or identical to input is not a meaningful improvement except where the transformation really is identity.

### P0 — Hand transcription errors are real, not just parser errors
Cross-checked actual model output against public official test outputs for a few targeted cases:

| ID | Wrong cells | Coordinates, 0-based | Other |
|---|---:|---|---|
| `4c3d4a41#1` | 1 of 160 | (row 1, col 17): predicted 7, actual 5 | natural finish |
| `247ef758#1` | 1 of 216 | (row 6, col 12): predicted 4, actual 6 | natural finish |
| `2b83f449#0` | 6 of 323 | (2,1), (8,15), (14,15), (18,1), (18,6), (18,15) | forced close |

**Fix:** Generate the full grid from an executable rule where possible. If model-only, add a fast **independent row/cell audit** as candidate_2; do not mutate candidate_1 in-place without verification. Self-consistency checks (dimensions, structural repeats, copies of train-derived patterns) are useful but cannot establish ground-truth correctness. One-cell near-misses are compelling for the value of checks, but NOT proof that a cheap reviewer will fix them.

**Acceptance gate:** among originally one-cell-away errors, how many convert to exact matches without losing original hits; compare incremental pass@2.

### P1 — Two malformed grids, one violates height restriction too
- `247ef758#0`: 15 rows, 14 have width 20 and one width 21.
- `6e4f6532#0`: **31 rows**, 30 have width 28 and one width 29. Removing the extra cell alone still leaves an invalid 31-row grid.
- `CELL 4 -> parse_grid()` rejects ragged grids and falls back to identity; this prevents invalid format but loses candidate content. Some shape corrections are ambiguous, and many malformed grids may also have wrong semantic content.

**Fix:** Parser must retain raw fenced candidate, explicit `{height, row_lengths, invalid_rows}`, and a recovery attempt (e.g. model asked to regenerate only an exact N×M grid where shape is independently justified). If `n_rows>30`, reject rather than blindly clipping. Never auto-drop a cell unless a deterministic rule, justified independently of evaluation ground truth, pins the location uniquely. Re-validate any repaired grid.

**Acceptance gate:** 22/22 format recovery on archived outputs where feasible, **and** exact hit delta. Format recovery alone isn't accuracy gain.

### P0 operational — RUN4 `submission.json` is deliberately NOT its EVAL12 predictions
**Evidence:** `CELL 6 -> write_submission()` immediately writes an identity submission and returns when `RERUN=False`. RUN4 `run_report.json` explicitly has `rerun: false`, and its archived `run4/output/submission.json` has all 240 hidden challenge task IDs; it therefore **does not store the 22 EVAL12 model answers**. Those answers are in `answers/` and `records.jsonl`. This is normal for a commit-mode artifact, but a serious interpretation/deployment trap.

**Fix:** To obtain a genuine scored Kaggle leaderboard result, submit the committed **notebook** for the competition rerun, not the RUN4 `submission.json` artifact. Assert the expected competition-rerun environment flag at runtime and log the resolved mode prominently. Add a mode-contract test: with `RERUN=False`, archived submission must be labeled `IDENTITY_FLOOR_NOT_MODEL_OUTPUT`; with `RERUN=True`, final output must be validated against all hidden input task IDs, test counts, and two rectangular attempts; fail visibly if rerun mode silently remains false. Do not change the current safety floor into a missing file.

### P1 — Multi-query tasks re-learn the same rule from scratch
**Evidence:** 9/12 EVAL12 tasks have multiple test inputs, yet `CELL 3 -> render_prompt` and `CELL 6 -> admit()` treat every `(task,query)` independently, with fresh ~49k thinking. Example `b6f77b65` has 3 separately force-closed requests, none hit. `4c3d4a41` has one exact and one one-cell miss.

**Fix:** Keep a **task-level verified solver or verified abstract rule** if obtained from training demonstrations; reuse it for all test inputs in that task. If it is not train-verified, keep independent candidate paths and do not propagate one false rule to every test query. Gains are disproportionate on this EVAL12 because it has 10 extra query slots; the hidden split has only 19 extra slots, so be cautious extrapolating.

**Acceptance gate:** per-task test-query exact score, mean extra-query tokens, wrong-rule propagation rate, aggregate coverage on a representative test-count distribution.

### P1 — 12-hour projection is neither a hard failure nor a guarantee
**Evidence:** RUN4 report quotes 13.86 h using **248.9 tok/s whole-22-query average**. From 390 approximately 10-second vLLM logging windows in executed notebook, **215 windows** have 16 requests active and generation mean **356.8 tok/s** (median 369.9); 33.3% windows have ≤6 active. KV usage peaked at 74.2% and preemptions remained zero. A 259-query steady-load run can perform better than the 22-query tail. Simple optimistic scenario using the observed steady-load mean: `259 * 46,946.3 / 356.8 / 3600 + (856.5 / 3600) + 0.05 ≈ 9.76 h`. **This is an extrapolation, not a measured 259-query result.** Longer contexts, different completion patterns, FP8 caching, and higher concurrency can lower throughput.

The hidden scheduler stops admitting at ~10.5h, starts draining ~11.1h, and hard-stops ~11h35m after the notebook starts; a naive 13.86h workload would leave a substantial tail unattempted. `report["projection_valid"]` means only "complete, error-free small-sample measurement," not "reliable hidden duration prediction".

**Fix:** Run a **representative stress benchmark** at ~16 slots long enough to see sustained throughput with a queue always full; log distribution of request length, prefill/second-phase stalls, KV, preemptions, and completion times. Fit optimistic/central/pessimistic coverage bounds. Do not call 16→20 concurrency a free gain: 20/16 × 74.2% ≈ 92.8% simple KV extrapolation, close to capacity and not accounting for variance. Test 16/18/20 as separate speed-memory experiments.

**Acceptance gate:** sustained `actual_generated_tokens / scheduler_wall_seconds`, no OOM, bounded preemptions, >98% predicted query completion by 12h with reserve; improvement in **total exact solutions under a time budget**, not peak tokens/sec.

### P1/P2 — FP8 KV scale warning can be a quality variable
**Evidence:** Executed engine log includes: `Checkpoint does not provide a q scaling factor. Setting it to k_scale.` and `Using KV cache scaling factor 1.0 for fp8_e4m3. If this is unintended, verify that k/v_scale scaling factors are properly set in the checkpoint.` This does **not prove** that the quantized cache caused 0/18 forced-close hits. It does expose a plausible unmeasured quality/runtime trade-off.

**Fix:** First verify FP8 checkpoint provenance, available scale metadata, and template/tokenizer parity. Controlled fixed-prompt/fixed-seed comparisons on **training-derived holdout** tasks: FP8 KV vs BF16/auto KV, same model weights. BF16 KV may reduce concurrency, so measure both exact solve rate and total solves per 12h. Do not change scales by guesswork.

### P2 — Prefix cache and logging need validated experiments
**Evidence:** `CELL 5 ENGINE_CONFIG` sets `enable_prefix_caching=False`, so forced close of 49k tokens requires another prefill. `run4/output/q27x/vllm_frontend.log` is **0 bytes**, despite stats available in executed notebook outputs. Thus attached standalone metrics file omits important text logs. In vLLM 0.19 for hybrid GDN/Mamba, prefix caching is **experimental** and may be ineffective when cache block alignment prevents reuse; the vLLM config docs and issue reports warn about this.

**Fix:** Enable cache only in a short stress run; inspect `prompt_tokens_cached`, prefill sec per forced close, GPU KV, preemptions, and `tok/s`. Try a short-sample `MTP3` vs `MTP1` comparison (vLLM advises MTP1 may optimize latency, while high concurrency throughput changes). Install the frontend logging handler **after** importing vLLM, and smoke-test that a marker reaches the file. Worker logs in independent processes may still require explicit output capture; do not claim complete logging from the frontend handler alone.

## Review of `OUR_ANALYSIS.md`— what is correct and what needs qualification

| Existing claim | Reviewer verdict |
|---|---|
| "Rule search out of budget" | **Supported** by 18 force-closes, thinking tails. Hard/easy selection effect means correlation, not confirmed causality. |
| "Some rules found, model writes grid manually" | **Supported** for individual traces; likely meaningful. |
| "Two naturally ended grids are one cell wrong" | **Verified** exact public GT comparison (see table). |
| "2b83f449 is six cells wrong" | **Verified**, exact positions above. |
| "Two ragged grids" | **True but incomplete**: `6e4f6532#0` also has **31 rows**, outside legal ARC dimensions. |
| "attempt_2 unused" | **Correct**: 0 incremental pass@2 hits. |
| "only 16 concurrent although KV max 74%" | **True observation, unproven recommendation** to raise to 20. |
| "projected 13.86h too pessimistic" | **Plausible, not a guarantee**: sustained 16-request windows perform much better, but 259-query regime is unmeasured. |
| "sort short prompts first" | **Weak potential benefit**: prompt size is not a reliable proxy for reasoning difficulty, and request lengths dominate. Better schedule explicit coverage and second-pass priority. |
| "submit v1.4 now" | **Only as one measured baseline**, if submission budget allows and actual hidden-time reliability is tested first. Do not treat it as candidate to replace a reported 32.64 LB system. |
| "Python-tool harness is top priority" | **Reasonable research direction, high implementation cost**. Must first show train verification and test gains on an untouched hold-out, plus secure execution and wall-time accounting. |

## Concrete code-level patch sketches (for the next notebook, not silently applied)

### 1. Candidate bank, compatible with current `make_attempts` semantics
Add to `CELL 4` and adapt `CELL 6` so multiple passes don't overwrite a successful first candidate. Existing `finish(st,...)` / `_record()` always overwrite `FINAL[uid]`; add an explicit pass identifier and merge rather than overwrite:

```python
def merge_candidate(uid, candidate, pass_idx):
    """Valid grid only; keeps first good answer and a different second candidate."""
    inp = CHALLENGES[UNIT_TASK[uid]]["test"][UNIT_Q[uid]]["input"]
    previous = FINAL.get(uid)
    if previous is None:
        previous = make_attempts(uid, candidate)
        FINAL[uid] = previous
        return
    if pass_idx >= 2 and valid_grid(candidate):
        if candidate != previous["attempt_1"]:
            previous["attempt_2"] = candidate
            previous["src_2"] = "model_2"
        # otherwise keep previous distinct fallback untouched
```

**Warning:** this snippet is a merge utility; it does not magically launch a second generation. Update the scheduler with a separate pass-two queue and request IDs, phase state, token budget and deadline. Keep `write_submission()` atomic. Tests: model2 different updates only attempt2; duplicate and malformed do not alter attempt1; a valid first hit is never lost; report per pass.

### 2. Preserve invalid-grid diagnostics instead of guessing repairs
Immediately after receiving the answer text:

```python
def raw_grid_diagnostics(answer_text):
    matches = _FENCE_RE.findall(answer_text or "")
    if not matches:
        return {"reason": "no_closed_fence", "row_lengths": []}
    body = matches[-1][1]
    raw_rows = [s.strip() for s in body.splitlines() if s.strip()]
    lens = [len(re.findall(r"\d", row)) for row in raw_rows]
    return {"n_rows": len(raw_rows), "row_lengths": lens,
            "invalid_shape": len(raw_rows) > 30 or len(set(lens)) != 1}
```

Only diagnostic. For a parser that supports JSON and free-form answers, branch by fence label instead of applying the digit-regex blindly to JSON syntax; if final fenced answer is malformed, request regeneration with a verifiable shape hint rather than promoting unrelated earlier text.

### 3. Add a long-run throughput estimator, not `projected_valid=True` as a promise
Compute and store separate `{pessimistic, central, optimistic}` projections from sustained active-set windows and an observed lower-throughput period. Include dynamic remaining-time admission gating; preserve a submission after any failure. Benchmark 16 vs 18 vs 20 slots at same work distribution and identical time cap; measure solved queries, not throughput alone.

### 4. Fix frontend log attachment order
In `CELL 5`, import `vllm` and set the handler **after** that import; emit `logging.getLogger("vllm").info("Q27X_LOG_PROBE")` and require `Q27X_LOG_PROBE` in the file. The executed notebook output is still needed for multiprocessing engine worker diagnostics unless redirected explicitly.

### 5. Formal GPU-quality A/B
Hold all except one variable constant per test, e.g. 16 concurrency, MTP3, 49k thinking; compare 0.6 vs 0.6 fixed across experimental branches first. Do not simultaneously change sampling, phase budgets, tool execution, prefix caching and concurrency and then attribute score differences to one change.

## Proposed experiment sequence, ranked for real 12h/4×L4 feasibility
1. **R5 diagnostic patch (low cost):** real pass@2 bookkeeping, save bad grids + row profiles, log after vLLM import, report optimistic/conservative throughput, verify per-query scoring. No huge GPU retraining; should be done first.
2. **R6 solver-stage budget ablation (medium cost):** same hold-out task set under (A) 49k single, (B) shorter reasoning + distinct second model candidate, (C) xhigh + small direct-output auditing second pass. Compare marginal hits/time. Keep a true 12h resource envelope.
3. **R7 prototype train-verified Python solver (high cost/high upside):** implement on public training holdout only; execute candidates in restricted subprocess, check train exact, make tasks with multiple queries share a verified rule. Integrate only if wins under the same budget.
4. **R8 engine microbenchmark (low-medium):** prefix cache off/on and `MTP1`/`MTP3` separately; throughput and cached-prefix token counters; 16/18/20 concurrency with preemption/OOM gate.
5. **R9 controlled FP8 KV vs auto quality parity (medium):** same model and prompt, fixed seed, fit-to-time speed/score; investigate absent KV scaling factors and compare with reference logits where available.
6. **R10 representative shadow 12h run:** count completed real-format queries, active mix over time, exact accuracy, runtime margin; only then consider formal LB submission.

## Go/no-go criteria
- Every submission output is valid JSON, all challenge task IDs and test query counts match, and both `attempt_1`,`attempt_2` are valid rectangular 1–30 grids in digits 0–9.
- No scheduler exception, OOM, preemption spiral, or missing output; robust atomic checkpoint after each completed candidate; no use of evaluation GT in inference logic.
- Meaningful incremental exact **pass@2** gains and verified task-level sharing gains on an independent holdout; measure score per wall-second, not only answer parsing.
- Good 12h completion margin under a full-size 259-query workload; 12h includes installation and model loading.
- For the 60% goal, a 2/22 small-sample signal alone is far from sufficient. Use a larger, stable, pre-registered evaluation protocol without optimizing repeatedly to a fixed public eval12 set.

## Source and evidence locations
- Main notebook: `notebook/Q27-XHIGH-v1_4.ipynb` — **CELL 0** fallback, **CELL 1** budgets/deadlines, **CELL 3** prompt/unit construction, **CELL 4** parser and attempts, **CELL 5** engine/FP8 cache, **CELL 6** scheduler and overwrite behavior, **CELL 7** scoring/projection.
- RUN4: `run4/output/q27x/records.jsonl`, `run4/output/q27x/answers/`, `run4/output/q27x/run_report.json`, `run4/run4_executed.ipynb` (vLLM logs in stream output, NOT the empty `vllm_frontend.log`).
- RUN3: `previous_run3/README.md`, `previous_run3/q27x/records.jsonl`, `previous_run3/q27x/run_report.json`.
- External benchmark context: Kaggle scoring/rules; vLLM v0.19 model config docs and experimental hybrid prefix caching; Qwen3.8 chat template `reasoning_effort` is a text-injected instruction, not a guaranteed compute budget; model effort comparisons require actual rendered prompt assertions.

**Audit limitation:** All checks here use the supplied archive plus public data and docs; I did **not** execute the 27B model on an L4 GPU, verify accuracy against the entire 120-task public eval, or reproduce a hidden Kaggle rerun. Hypotheses about future score improvement remain experiments, not established gains.
