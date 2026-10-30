# E23 rev 3g SMOKE results (2026-10-30, hosen42, 4xL4, vLLM 0.32.0)

SMOKE_A 106a37f2: 50.5 min of the 105 min budget, no abort. SMOKE_B bc6215ef: 78.9 min of 135, no abort.
Prompts are the same in both (2k/8k/40k/90k/116k, 24 each, distinct first task). The tok/s values come from 30 s steady
windows with the real number of running streams (running_min == running_max == level, waiting 0, 0 preemptions).

## Throughput (steady windows)

| lever | arm | window | tok/s | per stream | accept len | step ms | KV tokens |
|---|---|---|---|---|---|---|---|
| #1 Merlin W4A16 | TP1 GPU0 | 40k@2 | 93.9 | 47.0 | 4.338 (MTP4) | 92.4 | 220,897 |
| #1 Merlin W4A16 | TP1 GPU0 | 120k@1 | 41.0 | 41.0 | 3.924 | 95.6 | 220,897 |
| #1 Merlin W4A16 | TP4 | 40k@8 | 421.5 | 52.7 | 4.184 | 79.4 | 1,789,034 |
| #1 Merlin W4A16 | TP4 | 120k@8 | 329.0 | 41.1 | 4.227 | 102.8 | 1,789,034 |
| #1 Merlin W4A16 | TP1 x4 replicas | 40k, 2 per replica | 359.5 (92.0/91.8/90.0/85.7) | 44.9 | - | - | 220,897 each |
| #6 FP8 MTP3 (G0 PIECEWISE control) | TP4 | 40k@16 | 435.7 | 27.2 | 3.569 | 131.1 | 1,159,379 |
| #3 FP8 MTP3 FlashInfer + PR 60532 (G3 FULL_AND_PIECEWISE) | TP4 | 40k@16 | 448.9 (+3.0 % vs G0) | 28.1 | 3.584 | 127.7 | 1,110,733 |
| #5 NVFP4 KV, MTP off | TP4 | 120k@8 | 133.0 | 16.6 | - | 60.2 | 2,356,868 (1.67x FP8 MTP0 1,412,664) |

Not measured in SMOKE (by design, rev 3g trims): Merlin at 16+ streams, FP8 MTP0 throughput, INT4 MTP3 control,
NVFP4 + MTP3, TF FP8 KV calibration, EAGLE, compiled PP4.

## Correctness

- Every DONE arm: probes OK, oracle garbage-free, correctness CLEAN.
- Merlin TP4 vs TP1: greedy 90k EXACT; oracle 3 EXACT + 1 NEAR_TIE (first diff at token 24, margin 0.125); teacher
  forced 40k (n 256): mean |dlogprob| 0.0772, p99 1.0, max 3.0, rank-1 0.9883 (same engine and weights, numerics only).
- Merlin TP1 x4 replicas: greedy EXACT on all 4.
- Merlin server default chat template kwargs: `enable_thinking: True, reasoning_effort: "medium"` and the froggeric
  v22.5 template. Production uses xhigh, so this must be matched before any accuracy use.
- FP8 G0, G3, NVFP4 MTP0: oracle NEAR_TIE vs fp8_mtp0 (and fp8_g0 for G3).
- NVFP4 KV teacher-forced vs BF16 KV (40k, n 256): mean 0.3787, p99 3.946, max 13.187, rank-1 0.9648. The FP8 KV
  calibration arm is FULL-only, so `nvfp4_gate` has no verdict from this SMOKE (for scale: the TP4-vs-TP1 numerics
  drift above is mean 0.077 / rank-1 0.988).
- INT4 RedHat MTP0 reference: CLEAN, oracle saved, KV 1,788,404.

## Graph proof (#3)

Both profiled arms: `forwards 0` (no forward annotations recorded), so the verdict is UNPROVEN and graph_proof is
INCONCLUSIVE_GRAPH for G0 and G3. The kernel launch sources still show the difference:

| arm | graph launches | kernel launches | eager kernel share | attention launches | GDN launches |
|---|---|---|---|---|---|
| G0 PIECEWISE | 1,809 | 7,155 | 25.5 % | 1,026 eager | 1,296 eager |
| G3 FULL_AND_PIECEWISE | 116 | 1,566 | 5.2 % | 1,102 graph | 1,392 graph |

In both, NCCL all-reduce/all-gather is 35.6-35.7 % of kernel time, Marlin GEMM 27 %, attention 19.8 %.
The PR 60532 patch applied cleanly (rc 0) and was reverted (restored sha 343b949d62db7fc0).

## Kernels

- #4 sm89 FP8 GEMM (GPU3, real weights, gate_up_proj + down_proj only; qkv/o/gdn_in missing, so the end-to-end speedup
  fields are null). MLP time per forward: M=1 Marlin W8A16 17,011 us vs CUTLASS W8A8+quant 18,690 us (Marlin 9 % faster);
  M=16 17,136 vs 19,153 (Marlin 11 % faster); M=80 24,294 vs 22,932 (CUTLASS 6 % faster). BF16 cuBLAS 33,101-39,030 us.
  All gates passed.
- #2 host all-reduce P5 (983,040 B, 4 ranks): NCCL graph 336.23 us, host graph best 361.68 us (12 blocks) = 0.93x;
  NCCL tuned channels 4/8/16/32 were all slower (351-429 us). INT8 graph 237.73 us (1.41x vs NCCL). Status CHECK_FAILED
  only on `d_int8_within_bound_and_nonfinite_groups_nan` for the two float32 cases: `within_bound_frac`
  0.9999999999999999 with `worst_err_over_bound` 0.959 / 0.941 (< 1, so no element exceeded the bound): a float
  summation artefact in the check (exact == 1.0 test), not an error over the bound. All bf16 cases pass.

## PP4 (#1)

Stock vLLM 0.32 + RedHat INT4, eager: 2k stage EXACT, 8k stage crashed with `CUDA error: an illegal memory access`
(NCCL watchdog, rank 0); no Xid; GPU health probe OK on 4 GPUs; class APP_RESTART. Compiled PP4 skipped
(SKIPPED_EAGER_NOT_PROVEN). Consistent with the open issue #63505 noted in the build.

## Harness notes for FULL

- Forward annotations never fired (forwards 0) -> no graph proof is possible until fixed.
- P5 int8 bound check needs a tolerance (or `worst_err_over_bound <= 1`) instead of `within_bound_frac == 1.0`.
- nvidia-smi on Kaggle has no `reset_status.reset_required` field (health then relies on the CUDA probe).
- `ModuleNotFoundError: wrapt` from sitecustomize in every venv subprocess (harmless here).
