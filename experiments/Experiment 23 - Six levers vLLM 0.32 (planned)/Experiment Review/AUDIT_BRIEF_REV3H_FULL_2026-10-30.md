# E23 rev 3h: last audit before the FULL GPU run (2026-10-30)

## What to audit
The FULL notebook `notebooks/q27-xhigh-e23-six-levers-full.ipynb` (sha 674961ee) is ready to push to Kaggle hosen42. It
needs 690 min of the 12 h session on 4x L4, with vLLM 0.32 and Qwen3.8-27B.

The user decided to skip the TARGET run and go straight to FULL. This is the last check.

Please look for:
1. Any fix from the three rev 3g SMOKE audits (list below) that is missing or wrong.
2. New bugs that the fixes introduced.
3. Anything that would waste the 690-minute run: a crash, a gate that blocks everything, a wrong number, a budget problem.

Point to the exact file and line. A fix proposal is welcome.

## Fix map (rev 3g audits -> rev 3h code)
| # | Audit finding | rev 3h change | Evidence |
|---|---|---|---|
| 1 | Graph parser looked for `gpu_model_runner` annotations; the real traces have only `execute_context_N(N)_generation_N(N)` | `src/e23_lab.py`: `SCOPE_RE`, `_is_scope`, `_scope_kind`, per-scope [graphs, kernel launches, eager compute, unresolved]. FULL = >= 1 graph and no eager model compute (attention / GDN / GEMM / NCCL). Eager input-prep kernels are allowed and listed in `aux_eager_kernels_in_scopes`. Unresolved launches are never FULL. All scopes must agree, else MIXED | Real SMOKE traces: G0 = PIECEWISE in 27/27 scopes (65 graphs, 128 eager compute per scope); G3 = FULL in 29/29 (1 graph, 0 eager compute, 16 prep kernels). Gate tests `scope_*` |
| 1b | (found while fixing) `_cat` reduced `void (anonymous namespace)::gdn_decode_post_conv_mtp_kernel<...>` to "void " -> other | Strip `(anonymous namespace)::` and leading `void`; `reshape_and_cache` counts as attention | The GDN share of decode kernel time is 6.3 % (it was reported as 0.2 %) |
| 2 | `rank1_agreement` measured "forced token rank 1 on both or neither": ranks 2 vs 3 read 1.0 | `teacher_forced` keeps the rank-1 token id of each position (`_tok_id`). `compare_tf` = the share of positions where the two engines have the same top-1 id. A missing id gives `INCONCLUSIVE_TOP1_MISSING`. `tf_complete` requires the id. The old metric is kept as `forced_token_rank1_both_or_neither` (info only) | Gate tests: `tf_rank2_vs_rank3_diff_top1_is_0`, `tf_same_top1_is_1`, `tf_top1_missing_inconclusive`, `tf_parse_vllm_shape` (vLLM-shaped response) |
| 3 | `nvfp4_gate` could pass with only 40k | `NV_BANDS = {"40k", "90k"}` is required on both `tf_fp8kv_vs_bf16` and `tf_nvfp4_vs_bf16`, each MEASURED with n == 256, else `MISSING_REQUIRED_TF_BANDS` / `INVALID_TF_BAND` | Gate tests: `nv_40k_only_stops`, `nv_fp8_calibration_40k_only_stops`, `nv_90k_short_band_stops`, `nv_90k_top1_missing_stops`, `nv_both_bands_pass` |
| 4 | The load prompts were hand-built without effort; production renders xhigh | CELL3 renders head and tail with `AutoTokenizer.from_pretrained(FP8).apply_chat_template(..., add_generation_prompt=True, enable_thinking=True, reasoning_effort="xhigh")`, split at a placeholder. It asserts that the xhigh instructions are present and records the render, the template sha and the tokenizer sha in `R["prompt_render"]`. Merlin is launched with `--chat-template <production template>` and `--default-chat-template-kwargs {"enable_thinking": true, "reasoning_effort": "xhigh"}` (appended after the profile flags; argparse takes the last). Every oracle records a `/tokenize` template probe (server render vs production xhigh ids) | Local render with the production template: xhigh adds `<|im_start|>system\nReasoning effort is set to xhigh. ...<|im_end|>`; the medium render equals the rev 3g prompt exactly, so the rev 3g SMOKE tok/s are medium-prompt numbers. The Merlin checkpoint template has the same md5 as the FP8 one |
| 5 | P5 `ok` used `within_bound_frac == 1.0` | `prep/P5_host_allreduce/bench_host_ar.py` int8_eval: `ok = nan_ok and all_within_bound (elementwise all()) and isfinite(worst) and worst <= 1`. The kits dataset has a new version; manifest sha f9588c8f… (the notebook checks it at start). The 0.9999999999999999 comes from torch's double mean (sum x 1/n) | Gate tests `p5_eval_*`, including the SMOKE fp32 size 245761 |
| 9 | Kaggle nvidia-smi has no `reset_status.reset_required` | Recorded as `reset_status_field: unsupported`; health = 4 GPUs + the CUDA matmul probe | - |
| 10 | Merlin TP4 levels too short for the long-context sweep | 40k [8, 16, 20, 24], 90k [8, 12, 16, 20], 120k [8, 12, 16]; a KV-bound band stops at its fit | Static gate test |
| 11 | PP4 ran before the A/A fill | `pp4()` runs after the A/A loop; the A/A loop keeps `PP4_RESERVE` = 25 min | Static gate test `pp4_after_aa_fill`; kit p3 order ends with `e_pp4_int4_eager` |
| 6-8 | PP4 de-prioritised; no direct Merlin + NVFP4 + FULL graph stack; long-context accuracy gate before any production claim | Decisions, unchanged code paths (the FULL plan measures each lever as its own arm) | - |

## Validation done
- **Local gate suite:** 103 cases, all OK, run against the generated FULL notebook code (`results/gate_tests_local.txt`).
- **Kaggle CPU kit v8,** on the pinned image with the real /kaggle/input mounts and a fake vLLM server:
  - p2 (TARGET_B + competition rerun): ALL OK, rc 0, 0 tracebacks.
  - p3 (FULL with a 30-min budget, order and budget guard): ALL OK, rc 0, 0 tracebacks.
  - p1 (unit + TARGET_A + the gates on Kaggle): still running when this zip was made.
  - The fake server has no prompt_logprobs, so the TF steps read INCONCLUSIVE_TF in the kit (as in rev 3g). The TF parser is covered by `tf_parse_vllm_shape`.
- **SMOKE notebooks** (the ones that ran on Kaggle) are untouched: 106a37f2 / bc6215ef.

## FULL plan (unchanged except the items above)
P5 -> Merlin TP1 (+P4) -> TP1 x2 -> TP1 x4 -> TP4 -> FP8 MTP0 -> INT4 MTP0 ref -> INT4 MTP3 s20 control -> MTP1 -> MTP3 / G0
(profiled) -> G1 -> G2 -> G3 (+#60532, profiled) -> TF BF16 ref -> TF FP8 vs BF16 -> NVFP4 target-only -> TF NVFP4 vs BF16 ->
NVFP4 + MTP3 (only if the gate passes) -> A/A repeats (time fill) -> PP4 eager (-> compiled only if eager is proven).

The main sequence is estimated at 651 of 690 min (build assert <= 670). There is no EAGLE: `E23_EAGLE_FULL` is not set, and
no EAGLE inputs are attached.

## Files
- `notebooks/q27-xhigh-e23-six-levers-full.ipynb` + `kernel-metadata.json`: the push copy (id `hosen42/...`).
- `notebooks/build/build_e23_notebook.py`, `notebooks/build/build_cpu_kit_e23.py`.
- `src/e23_lab.py`, `src/harness/gate_tests.py`, `src/harness/fake_vllm_server.py`, `src/bench_host_ar.py` (kits copy), `src/q27_e23_kernels_manifest.json`.
- `results/`:
  - the local gate output;
  - the kit v8 p2 / p3 reports;
  - `trace_regression.json` (the new parser on the two real SMOKE traces);
  - the earlier fix lists.
- `git_diff_rev3g_to_rev3h.patch`: every source change since the SMOKE run.
