# E23 rev 3g SMOKE: outputs + proposed next steps (audit brief, 2026-10-30)

## Goal and context
- Target: 1500+ tok/s sustained aggregate decode for Qwen3.8-27B on Kaggle 4x NVIDIA L4 (sm_89, 24 GB each, PCIe, no NVLink,
  offline), with long agent reasoning (thinking up to 120k, max-model-len 131072, 16-24 streams), without accuracy loss.
- Engines in scope: vLLM 0.32 and the best SGLang (INT4 + DFlash2 W6).
- Please audit: (1) whether the SMOKE conclusions below follow from the outputs in `outputs/`, (2) whether the next-step plan
  is the right order to reach 1500+, (3) anything missing, wrong, or risky. Point to exact files / fields.

## Earlier baselines (4xL4, TP4, fp8 KV, prefix cache on; tok/s at 8k / 40k / 90k context)
| setup | streams | 8k | 40k | 90k |
|---|---|---|---|---|
| SGLang 0.5.21 INT4 + DFlash2 W6 | 20 | 597 | 536 | 364 (about 6-7 LB points lower accuracy, cause unknown) |
| vLLM 0.32 INT4 MTP3 | 20 | 518 | 439 | 332 |
| vLLM 0.32 FP8 MTP3 SSM fp16 | 16 | 450 | 414 | 312 |
| vLLM 0.19 FP8 MTP3, real agent runs, whole run | 16 | about 256-320 | | |

## SMOKE runs
- SMOKE_A (Kaggle hosen42 `q27-xhigh-e23-six-levers-smoke-a`, notebook sha 106a37f2): 50.5 of 105 min, no abort.
- SMOKE_B (`q27-xhigh-e23-six-levers-smoke-b`, bc6215ef): 78.9 of 135 min, no abort.
- Build: `notebooks/build/build_e23_notebook.py` (rev 3g) generates SMOKE_A, SMOKE_B and FULL; lab code `src/e23_lab.py`.
- All load requests go through `/v1/completions` with our own chat-formatted token ids (thinking on), so server chat templates
  are not applied there. Only the arithmetic probes use `/v1/chat/completions`, with thinking off.

## Measured (30 s steady windows; running_min == running_max == level, waiting 0, preemptions 0)
| lever | arm | window | tok/s | per stream | accept len | step ms | KV tokens |
|---|---|---|---|---|---|---|---|
| #1 Merlin W4A16 MTP4 | TP1 GPU0 | 40k@2 | 93.9 | 47.0 | 4.338 | 92.4 | 220,897 |
| #1 Merlin | TP1 GPU0 | 120k@1 | 41.0 | 41.0 | 3.924 | 95.6 | 220,897 |
| #1 Merlin | TP4 | 40k@8 | 421.5 | 52.7 | 4.184 | 79.4 | 1,789,034 |
| #1 Merlin | TP4 | 120k@8 | 329.0 | 41.1 | 4.227 | 102.8 | 1,789,034 |
| #1 Merlin | TP1 x4 replicas | 40k, 2 each | 359.5 | 44.9 | - | - | 220,897 each |
| #6 FP8 MTP3 G0 PIECEWISE | TP4 | 40k@16 | 435.7 | 27.2 | 3.569 | 131.1 | 1,159,379 |
| #3 FP8 MTP3 FlashInfer + PR 60532 (FULL_AND_PIECEWISE) | TP4 | 40k@16 | 448.9 | 28.1 | 3.584 | 127.7 | 1,110,733 |
| #5 NVFP4 KV, MTP off | TP4 | 120k@8 | 133.0 | 16.6 | - | 60.2 | 2,356,868 |

Correctness:
- Every DONE arm: probes OK, oracle without garbage, correctness CLEAN.
- Merlin TP4 vs TP1: greedy 90k EXACT; teacher-forced 40k mean |dlogprob| 0.077, rank-1 0.988.
- NVFP4 KV vs BF16 KV (teacher-forced 40k, n 256): mean 0.379, p99 3.95, max 13.2, rank-1 0.965. The FP8 KV calibration arm is
  FULL-only, so there is no `nvfp4_gate` verdict yet.

Graph proof (#3): the trace had 0 annotated forwards, so the verdict is UNPROVEN / INCONCLUSIVE_GRAPH. Launch sources:
- G0: 25.5 % eager kernels, attention and GDN launched eager.
- G3: 5.2 % eager kernels, attention and GDN graph-launched.
- Decode kernel time (both): NCCL 35.7 %, Marlin GEMM 27 %, attention 19.8 %, other GEMM 9 %, rest 8 %.

Kernels:
- #4 FP8 GEMM, MLP only. Marlin W8A16 is 9-11 % faster than CUTLASS W8A8+quant at M 1-16; CUTLASS is 6 % faster at M 80.
- #2 all-reduce, 983,040 B:
  - NCCL graph 336.23 us.
  - Host-staged 361.68 us (0.93x).
  - INT8 graph 237.73 us (1.41x). Its CHECK_FAILED comes only from the float32 cases, where `within_bound_frac` =
    0.9999999999999999 and `worst_err_over_bound` < 1. We read this as a float-summation artefact of an `== 1.0` test.

PP4 (stock vLLM 0.32 + RedHat INT4, eager):
- 2k EXACT, then `illegal memory access` at 8k.
- No Xid; health OK.
- Compiled PP4 skipped.

## Interpretation (to be audited)
- We are at about 36 % of the target at 40k (536 SGLang, 449 vLLM) and about 20 % in whole-run agent throughput.
- Merlin is the largest open lever: per-stream speed is about 1.9x FP8 and KV fits about 13.6 x 131k at TP4, but it was only
  measured at 8 streams.
- NCCL is 36 % of the decode step, so INT8 all-reduce (1.41x microbench) could cut about 10 % of the step if it works inside
  vLLM without accuracy loss.
- NVFP4 KV gives 1.67x KV (more streams), but its drift needs the FP8 calibration verdict.
- FULL graph gives +3 %.
- Host-staged all-reduce, PP4 on 0.32, and TP1 x4 (KV-bound at 2 streams per replica) are not paths to the target.

## Proposed next steps (order to audit)
1. Fix the harness before the E23 FULL run:
   - (a) forward annotations, so a graph proof is possible;
   - (b) the P5 int8 check, using `worst_err_over_bound <= 1` instead of `within_bound_frac == 1.0`;
   - (c) the Merlin server: launch with our production chat template and `--default-chat-template-kwargs`
     `{"enable_thinking": true, "reasoning_effort": "xhigh"}` (its profile default is "medium" with the froggeric v22.5
     template), plus one chat-path probe that checks that the rendered prompt carries xhigh.
2. E23 FULL (690 min budget) on hosen42 after the user's go. Key measurements:
   - Merlin TP4 at 16/20/24 streams at 90k and 120k;
   - the NVFP4 gate with FP8 calibration, then NVFP4 + MTP3;
   - the INT4 MTP3 control;
   - A/A repeats.
3. Stack the levers that pass (candidates: Merlin TP4 MTP4 + NVFP4 KV + FULL graph) and measure them as one configuration.
4. Run INT8 all-reduce inside vLLM (custom all-reduce path) with a teacher-forced drift check.
5. E24 SGLang track (folder ready, commit 4482c4e): find the cause of the INT4 + DFlash2 accuracy loss, then start from the
   597/536 tok/s base.

## Files in this zip
- `notebooks/`: the SMOKE_A / SMOKE_B notebooks exactly as pushed (with kernel-metadata.json), the FULL notebook, and the
  build script.
- `src/`: e23_lab.py, harness (gate_tests.py, fake_vllm_server.py), patches, DESIGN.md.
- `outputs/SMOKE_A`, `outputs/SMOKE_B`: e23_results.json, p4/p5 bench JSON, notebook logs, every server log, profiler
  traces (SMOKE_B), and SMOKE_REV3G_RESULTS_2026-10-30.md (our summary).
