# PIV-Q27-C1 v2.5 — fake-engine / failure-injection audit (Claude, 2026-09-18)

Notebook: `PIV-Q27-C1-v2_5-CANDIDATE-FIRST-VERIFIER-FEEDBACK-REPAIR.ipynb`
SHA-256: `eb7d65ba6b0f1f0da56975ee467b3eb1b10f41539059561332403b376bff7ee6`
Harness: `harness_c1.py` (in-process cell exec, real Qwen tokenizer + chat template, real ARC-AGI-2 data, scripted `_FakeLLM`,
real subprocess verifier). Profiles run: PREFLIGHT, TERMINAL_PILOT, ADAPTIVE_SMOKE, ADAPTIVE_SMOKE + `HARNESS_OVERCAP=1`.

## Verdict: DO NOT RUN ON GPU as-is — two deterministic P0 blockers

### P0-1 — cell 6 self-test asserts a false condition → AssertionError in EVERY profile (before model load)
`run_v25_repair_helper_selftests()` (cell 6, called at cell import: `V25_REPAIR_HELPER_SELFTEST=run_v25_repair_helper_selftests()`):

```python
_dim_identity=_v25_pair_diag({"index":1,"exact":False,"pred":fake_task["train"][1]["input"],"error":None},fake_task["train"][1])
assert _dim_identity["kind"]=="dims_mismatch" and _dim_identity["identity"],_dim_identity
```
`fake_task["train"][1]` = input `[[1,2,3]]` (1x3), output `[[3,2,1]]` (1x3). `pred = input` → got_dims (1,3) == expected_dims (1,3),
so `_v25_pair_diag` correctly returns `kind="cells_wrong"` (identity=True). The assertion demands `dims_mismatch` → fails.
Reproduced with plain Python outside the harness (pure function, no stubs involved):
```
{'pair': 1, 'kind': 'cells_wrong', 'got_dims': (1, 3), 'expected_dims': (1, 3), 'wrong_cells': 2, 'total_cells': 3, 'mismatches': [(0,0,3,1),(0,2,1,3)], 'identity': True}
```
None of the 4 fake train pairs has input dims ≠ output dims, so the test cannot pass with any pred==input.
The helper is right; the self-test is wrong. This proves cell 6 was never executed before delivery.

Fix (either): add a pair whose input/output dims differ (e.g. input `[[1,2,3]]`, output `[[1],[2],[3]]`) and use that pair for the
dims+identity check; or assert `kind=="cells_wrong" and identity` on the existing pair and test `dims_mismatch` separately with
`pred=[[1],[2],[3]]` against output `[[3,2,1]]` (that case already exists as `fake_v` index 1 and is covered via `fb["text"]`).

### P0-2 — cell 9 (CELL 8B) `NameError: _mk_finalizer_params` in every real (non-PREFLIGHT) profile
Cell 8: `_mk_finalizer_params` is defined at line 615 **nested inside** `def run_real_piv():` (line 589). Cell 9 line 159:
```python
params=[_mk_finalizer_params(j["cap"],j["seed"]) for j in jobs]
```
runs at module level inside `run_v25_repair_round_for_arm` → `NameError`. Reproduced (after patching P0-1) in ADAPTIVE_SMOKE:
```
File "cell9", line 264, in <module>
File "cell9", line 159, in run_v25_repair_round_for_arm
NameError: name '_mk_finalizer_params' is not defined
```
Timing on GPU: this fires AFTER the full v2.4G scheduler has run both arms (≈78 min) and BEFORE scoring/bundle → the run would
crash with no v25 bundle. PREFLIGHT cannot catch it (repair is skipped when `RUN_PROFILE=="PREFLIGHT" or not USE_MODEL`), and the
line is outside the `try/except BaseException` that wraps `arc_llm.generate`, so it is not converted to `repair_engine_exception`.

Fix: move `_mk_finalizer_params` (and `_mk_params`) to module level in cell 8 (they only use globals: `SamplingParams`,
`StructuredOutputsParams`, `FINALIZER_JSON_SCHEMA`, `FINALIZER_SAMPLING`), or define an identical module-level copy at the top of
cell 9. Add a PREFLIGHT-executed assertion `assert callable(globals().get("_mk_finalizer_params"))` in cell 9 so the interface
is checked before model load.

## With both P0s patched locally (scratch copy only; original untouched): all 4 profiles complete end-to-end
| profile | result |
|---|---|
| PREFLIGHT | completes; repair correctly skipped; helper self-tests pass; 48/48 ×3 arms |
| TERMINAL_PILOT | completes; repair round runs on 2 tasks/arm |
| ADAPTIVE_SMOKE | completes; interface gate pass; repair 7 eligible / 7 attempted per arm; bundle written |
| ADAPTIVE_SMOKE + OVERCAP | completes; repair over-cap caught (`repair_output_exceeded_max_tokens`, infra flag, candidate #1 preserved) |

Repair-stage failure injections exercised (per SMOKE8 position) and their observed handling — all correct:
- identity re-emitted → `repair_verified`, delta 0, `selected_candidate_index=1`
- dims-mismatch program → `repair_verified`, selected 1
- truncated JSON (finish=length at cap) → `repair_output_parse_failed`, selected 1
- program raising IndexError → `repair_verified`, c2=0, selected 1
- `time.sleep(15)` program → candidate #2 `train_timeout`, selected 1, no infra flag
- `import os` program → `final_guard_rejected:blocked_import`, selected 1
- verified-correct program (lookup-table cheat, 28a6681f) → `candidate2_all_train_exact=1`, selected 2, record_stage
  `v2.5_candidate_first_repair`; **scoring (cell 11 `latest_map` = last record per task) picks it up: train_perfect 0.125, tier usable_quality**
- candidate #1 with `final_code=None` (136b0064 BAD_JSON terminal) → `candidate1_not_eligible`, untouched
- transport assertion (stripped presence + SHA marker) true on all 14 rendered repair prompts; `repair_feedback.jsonl`,
  `repair_summary.csv/json`, `deferred_phase2_768_diagnostic.json`, `agent_loop_runtime.json` (per-arm) written as designed.

## P1 — pre-register before the GPU run (not crashes, but they shape the result)
**P1-1 Repair cap is the leftover of the 12,288 budget, and for most tasks it is smaller than 4,096.** Using the real v2.4G
records (`cap = min(4096, 12288 − generated_tokens)` vs candidate #1 JSON size in tokens):

| arm | task | leftover cap | cand#1 JSON tokens |
|---|---|---|---|
| LOW | 271d71e2 | 1,841 | 2,254 ← cap < candidate |
| LOW | 2c181942 | 1,229 | 2,866 ← cap < candidate |
| LOW | 16b78196 | 1,305 | 2,790 ← cap < candidate |
| LOW | cb2d8a2c / 136b0064 / 28a6681f / 291dc1e1 | 2,403 / 3,107 / 2,888 / 3,256 | 1,692 / 988 / 1,207 / 839 |
| BUDGET | 291dc1e1 / 271d71e2 / 2c181942 | 2,175 / 2,299 / 2,661 | 1,920 / 1,795 / 1,621 |
| BUDGET | others | 2,728–4,096 | 206–1,522 |

If v2.4G behaviour repeats, 3/8 LOW tasks cannot even re-emit a program of candidate #1's size → structurally
`repair_output_parse_failed` (finish=length), and 13/16 tasks give the repair ≤1.5× the candidate's own length. The v2.5 result
"repair helped k/16" must be read against this ceiling. Either pre-register it as a known limit, or give the repair round its
own budget (e.g. `REPAIR_MAX_TOKENS` independent of the 12,288 exploration cap, accounting for it separately in
`generated_tokens`) — the latter changes the budget contract, so it should be a stated decision, not silent.

**P1-2 Feedback loses the real error when the verifier returns no per-pair results.** For `verifier_status` in
{`train_timeout`, `train_code_error`} (`train=[]`, e.g. v2.4G BUDGET 136b0064 timeout) `_v25_feedback_text` emits
`pair i: EXCEPTION missing verifier pair result` for every pair and drops `train_error` (the timeout message / the SyntaxError
traceback). The model is told nothing actionable. Fix: when `train` is empty, emit one line from
`verifier_status` + `_v25_exception_last_line(verifier.get("train_error"))` (e.g. `VERIFIER: train_timeout after 20s` /
`VERIFIER: train_code_error SyntaxError: ...`). Cheap, no contract change.

## P2 / notes (no action required for the run)
- `_v25_update_record_from_selected` and `_v25_append_infra_invalid` overwrite `stop_reason="final_program"`; for eligible
  candidates it is already `final_program`, so gate diagnostics are unchanged, but a repair infra failure record also reads
  `final_program` (infra flag still blocks the gate — verified in OVERCAP run).
- On a resume/re-run, `_v25_latest_records_for_active_arm` takes the last record per task, which after one repair round is the
  repaired record → a second execution of cell 9 would repair the repaired candidate (double round). Guard with
  `record_stage!="v2.5_candidate_first_repair"` if resume is ever used.
- `cells_wrong` feedback line prints `dims got HxW / expected HxW` with identical dims — harmless.
- Harness caveats: fake engine does not model real decode latency; the lookup-table "successful repair" only tests the
  plumbing (selection → record → scoring), not model ability.

## Requested changes to get a runnable v2.5A (minimal)
1. Fix the cell 6 self-test (P0-1).
2. Hoist `_mk_finalizer_params` / `_mk_params` to module level (P0-2) + PREFLIGHT-time callable assertion.
3. P1-2 one-line feedback fix for empty `train`.
4. Pre-register P1-1 in the notebook header/manifest (or decide a separate repair budget).
No other logic change. I will re-run the same harness on the v2.5A SHA before the GPU run.
