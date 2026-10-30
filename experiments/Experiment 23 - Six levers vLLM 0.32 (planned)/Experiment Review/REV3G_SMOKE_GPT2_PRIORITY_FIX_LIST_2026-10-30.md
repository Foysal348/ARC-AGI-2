# E23 rev 3g SMOKE: priority fix list from two GPT audits (2026-10-30, no fixes applied yet)

Both audits:
- GO on the SMOKE results;
- NO-GO on running the current FULL notebook unchanged;
- GO on a FULL exploratory run after the fixes, without PP4, with the NVFP4 gate fail-closed.

Each claim below was checked against the code, the raw traces and the bench source before it was listed.

## Blockers for FULL

| # | Sev | Where | Problem (verified) | Fix | Validation |
|---|---|---|---|---|---|
| 1 | HIGH | `src/e23_lab.py` trace_summary, lines 768-771 | Looks for `user_annotation` names that contain `forward` and `gpu_model_runner`. The real traces have 0 such events. All annotations are `execute_context_N(N)_generation_N(N)`: G3 has 29, G0 (mtp3) has 27. This is why `forwards` = 0 and the verdict was UNPROVEN. | Match `^execute_context_.*_generation_\d+\(\d+\)$`. Keep per-scope graph and kernel counts. Classify verify-core coverage (attention, GDN, Marlin, spec verify inside a graph) separately from auxiliary eager kernels. Do not require "1 graph + 0 eager". | Regression test on the two SMOKE_B traces. Expect about 65 graph / 160 eager per scope for G0 and about 1 / 16 for G3 (both audits). Attention and GDN must be graph-resident in G3. Can run on CPU now. |
| 2 | HIGH | `src/e23_lab.py` compare_tf, line 522 | `rank1_agreement` = `(a[1]==1) == (b[1]==1)`. It measures whether the forced token is rank 1 on both sides, not whether both engines predict the same top-1 token. Example: ranks 2 and 3 count as "agreement". The SMOKE NVFP4 value of 0.9648 is therefore not a top-1 agreement. | `teacher_forced()` keeps the rank-1 token id of every position from `prompt_logprobs`. compare_tf uses `a[2] == b[2]`. A missing id gives INCONCLUSIVE_TOP1_MISSING, never PASS. | Different ids -> 0, same ids -> 1, missing id -> INCONCLUSIVE (gate tests). |
| 3 | HIGH | build nvfp4_gate, lines 1104-1136 | Iterates only over the bands present in the NVFP4 row. A run with 40k only, or with 90k missing, can PASS. FULL runs both bands (`SM_TF` only trims SMOKE), but the gate does not enforce it. | Require {"40k", "90k"} in both `tf_fp8kv_vs_bf16` and `tf_nvfp4_vs_bf16`, each MEASURED with n == 256. Otherwise STOP (`MISSING_REQUIRED_TF_BANDS` / `INVALID_TF_BAND`). Apply together with #2. | Gate tests: 40k only -> STOP, one incomplete band -> STOP, both complete + corrected top-1 -> eligible. |
| 4 | HIGH | build CELL3, `_HEAD` / `_TAIL` / `prompt()`, lines 378-394 | Load prompts are hand-built (`<|im_start|>user ... <|im_start|>assistant\n<think>\n`) with no reasoning effort. Production renders with `tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=True, reasoning_effort="xhigh")` (Experiment 1 make_q27x.py:343). A server flag alone cannot change raw `/v1/completions` ids. Merlin's own default is `reasoning_effort: "medium"` + the froggeric v22.5 template. | Build PROMPTS with the production tokenizer chat template, xhigh, enable_thinking. Record prompt hash, rendered template, effort and tokenizer sha. Launch Merlin with our production template and `--default-chat-template-kwargs '{"enable_thinking": true, "reasoning_effort": "xhigh"}'`. Add one chat-path probe that checks the rendered prompt equals the production render. Keep same-token speed rows and xhigh-agent rows as separate results. | A medium vs xhigh render differs as expected. The same hash is used on Merlin TP1 / TP4 and the stock arms. |
| 5 | MED | `prep/P5_host_allreduce/bench_host_ar.py:969-974` (in the kits dataset, sha-checked) | `ok = nan_ok and within_bound_frac == 1.0`. `within_bound_frac` is a float mean, giving 0.9999999999999999 on the two FP32 cases, while `worst_err_over_bound` is 0.959 / 0.941. All 32 cases are below 0.9884. This is a false CHECK_FAILED. | `ok = status_ok and identical_all_ranks and nonfinite_groups_all_nan and isfinite(worst) and worst <= 1.0`, or an elementwise `(err <= bound).all()`. Re-upload the kits dataset and update the sha manifest. | 0.99 -> PASS, 1.01 -> FAIL, NaN or rank mismatch -> FAIL. FP32/BF16, outlier, NaN/Inf and graph replay still pass. INT8 stays a microbench result, not production GO. |

## Decisions (no code risk)

| # | Decision |
|---|---|
| 6 | PP4 out of FULL. Stock vLLM 0.32 + INT4 eager hits an illegal memory access at 8k (#63505). It returns only with an upstream or code fix. |
| 7 | No direct stacking of Merlin + NVFP4 KV + FULL graph: first prove that the Merlin fork supports NVFP4 KV and the FULL graph mode, as a single arm each. |
| 8 | The FULL oracle checks 128 generated tokens. Before any production-accuracy claim, a separate long-context gate is needed: 90k-120k context, long MTP decode, forced rejection and multi-turn state. This is not a blocker for the exploratory FULL. |
| 9 | Health probe: Kaggle's nvidia-smi has no `reset_status.reset_required`. Treat the CUDA probe as the source and record the field as unsupported, not null. |

## Order after the fixes (both audits agree)
1. Apply fixes 1-5 and 9; add gate tests; then one parallel Kaggle CPU kit (regression + preflight).
2. FULL exploratory run on hosen42 (user go needed), in this order:
   - Merlin TP4 MTP4 at 8/12/16/20/24 streams, at 40k/90k/116k, on xhigh prompts;
   - the corrected NVFP4 vs FP8 calibration, then NVFP4 + MTP3 only if the gate passes;
   - the G3 graph proof and A/A repeats;
   - the INT4 MTP3 control.
3. INT8 all-reduce integrated in vLLM as an isolated arm with a teacher-forced drift check.
4. The winning configuration as an agent replay: sustained throughput and ARC quality.
5. In parallel: E24 SGLang, to find the cause of the INT4 + DFlash2 accuracy loss.
