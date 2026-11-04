# A1: vLLM engine speed, mined from E14 / E18 / E19 / E20 / E22 / E23 (for E25 Flash-Next 1500)

Agent A1, 2026-10-31. Read-only mining. All numbers are on Kaggle 4x L4 with Qwen3.8-27B (dense GDN hybrid). The exceptions are labelled.
- **Measured** means a number from a GPU log or results JSON.
- **Derived** means my own arithmetic on measured numbers.
- **Claim** means an upstream or research statement that we did not measure.

## Evidence keys

The paths are relative to `D:\ARC AGI 2\`.

| Key | Path |
|---|---|
| E14R | `experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Experiment Results/EMS_SAFE_RISKY_VMS_rev2g_GPU_results_2026-10-26.md` |
| VMS | `_scratch/ems_out/vms/q27x/vms/vms_report.md`. The server logs `vllm_server_<arm>.log` are in the same folder; the run JSON is `_scratch/ems_out/vms/e14_results.json` |
| E14F | `experiments/Experiment 14 .../README_FINDINGS.md` |
| N00 / N30 | `experiments/Experiment 14 .../Notes/00 Claude - step-time analysis and TP2 gain (2026-10-23).md` and `.../Notes/30 Claude - E14A SLIM R2 run results (2026-10-24).md` |
| N15 | `experiments/Experiment 14 .../Notes/15 Claude agent - NCCL bench and trace summarizer (2026-10-23).md` |
| PC | `experiments/Experiment 14 .../Notes/Speed review 2026-10-24/part_C_measured_runs_and_insights.md`, the vLLM 0.19-era table §1.13, rows 178-195 |
| E18S | `experiments/Experiment 18 - S1-60K on vLLM 0.32 (planned)/Experiment Results/E18_SMOKE_GPU_v1_2026-10-27.md`. Raw output: `_scratch/e18_gpu_v1/q27x/vllm_server_s1.log` |
| E18R2 | `experiments/Experiment 18 .../Experiment Results/GPU_commit_R2_2026-10-29/`, files `vllm_server_s1.log` and `e18_engine_stats.json`. The log is byte-identical to `_scratch/e18_gpu_v2/q27x/vllm_server_s1.log` (checked with `cmp`) |
| E18v3 | `_scratch/e18_gpu_v3/q27x/`, files `vllm_server_s1.log` and `e18_engine_stats.json` (SMOKE30, 2026-10-30) |
| E19 | `experiments/Experiment 19 - vLLM MTP prefix-cache correctness (planned)/Experiment Results/E19_GPU_run1_results_2026-10-27.md` |
| E20 | `experiments/Experiment 20 - S1-60K drain fix (planned)/Experiment Results/E20_LB_analysis_2026-10-29.md` |
| E21 | `experiments/Experiment 21 - Duck config B on vLLM 0.19 + 0.32 (planned)/Experiment Results/E21_SMOKE_GPU_run1_2026-10-29.md`, plus the final-run table in memory `experiment-21-duckb-vllm.md`. E21 is outside the A1 scope; it is used only for its prefix and SSM facts |
| E22F | `experiments/Experiment 22 - Speed lab vLLM 0.32 + SGLang (planned)/ENGINE_FINDINGS_vLLM032_SGLang.md` |
| E22K | `_scratch/e22_kit2b/flags/e22_flags_report.json`. E22 was a CPU flag kit only and never ran on the GPU |
| E23S | `experiments/Experiment 23 - Six levers vLLM 0.32 (planned)/Experiment Results/smoke_rev3g/SMOKE_REV3G_RESULTS_2026-10-30.md` |
| E23A / E23B | `.../smoke_rev3g/SMOKE_A/e23_results.json` and `.../SMOKE_B/e23_results.json`. Logs are in `.../SMOKE_A/logs/` and `.../SMOKE_B/logs/` |
| E23TR | `_scratch/e23/gt3h/trace_regression.json`: the rev 3h graph parser run on the real SMOKE_B traces |
| E23FIX / E23H | `experiments/Experiment 23 .../Experiment Review/REV3G_SMOKE_GPT2_PRIORITY_FIX_LIST_2026-10-30.md` and `.../AUDIT_BRIEF_REV3H_FULL_2026-10-30.md` |
| E23FULL | `_scratch/e23_full_sami/live_stream.txt`, the E23 FULL run on esfersami50. It is still running; no results are downloaded |

---

## 1. Measured numbers

### Column conventions
- **Context@streams:** the 8k / 40k / XL bands of VMS are prompts of about 3.85k / 40.9k / 90.6k tokens (VMS header line "prompts").
- **tok/s:** aggregate decode tok/s in a steady window, unless the row says "whole run" or "e2e".
- **Per-stream:** tok/s ÷ the real running count.
- **Accept:** the mean acceptance length, i.e. tokens per verify, including the bonus token.
- **KV:** the server-logged "GPU KV cache size" in tokens.

### Sampling used by each benchmark
- **VMS windows:** T = 1.0, top_p 0.95, top_k 20, ignore_eos, 75-s windows after the prefill settled (`Experiment Notebooks/Build/build_vms.py:217-229, 318-345, 863-871`).
- **E23 windows:** T = 1.0, top_p 0.95, top_k 20, ignore_eos, 30-s windows. Each window requires running_min == running_max == level, waiting 0 and 0 preemptions (`src/e23_lab.py:316-322, 569-596`). The SMOKE prompts used the **medium** render, i.e. without the xhigh system line (E23H, fix #4).
- **E18:** real Poetiq reasoning, T = 1.0.

### 1a. vLLM 0.32.0, stock

| Experiment | Config | Context@streams | tok/s | Per-stream | Accept | KV tokens (max usage) | Preempt | Evidence |
|---|---|---|---|---|---|---|---|---|
| E14 VMS | **ref** FP8 repack, TP4, MTP3, fp8_e4m3 KV, prefix on (align), FlashInfer, chunk 4096, seqs 16, util 0.90, custom AR off. Graphs requested FULL_AND_PIECEWISE, **downgraded to PIECEWISE** | 8k@16 / 40k@16 / 90k@10 | 434 / 399 / 281 | 27.1 / 24.9 / 28.1 | 2.96 / 3.05 / 3.08 | 1,019,504 (0.33 / 0.40) | 0 / 0 | VMS row `ref_fp8_mtp3_s16`; `vllm_server_ref_fp8_mtp3_s16.log:9` (args), `:227` (downgrade), `:251` (KV) |
| E14 VMS | same, e2e rate (submission to window end) | 8k / 40k | 322 / 195 | — | — | — | — | VMS column "e2e tok/s" |
| E14 VMS | FP8 MTP2 | 8k@16 / 40k@16 / 90k@10 | 448 / 397 / 260 | — | 2.47 / 2.50 / 2.54 | 1,056,871 | 0 | VMS `fp8_mtp2` |
| E14 VMS | FP8 MTP4 | 8k@16 / 40k@16 / 90k@9 | 399 / 381 / 267 | — | 3.38 / 3.53 / 3.45 | 962,267 | 0 | VMS `fp8_mtp4` |
| E14 VMS | FP8 MTP3, seqs 20 | 8k@20 / 40k@20 / 90k@10 | 452 / 423 / 281 | 22.6 / 21.2 / 28.1 | 2.96 / 3.14 / 3.00 | 1,006,517 (0.41 / 0.48) | 0 | VMS `fp8_mtp3_s20` |
| E14 VMS | FP8 MTP3, seqs 24 | 8k@24 / 40k@24 / 90k@9 | 472 / 424 / 256 | 19.7 / 17.7 / 28.4 | 2.96 / 3.04 / 2.92 | 992,231 (0.49 / 0.57) | 0 | VMS `fp8_mtp3_s24` |
| E14 VMS | FP8 MTP3, `--linear-backend triton` (TritonFp8BlockScaledMM, real W8A8) | 8k@16 / 40k@16 / 90k@10 | 438 / 398 / 283 | — | 3.05 / 3.07 / 3.18 | 1,028,595 | 0 | VMS `fp8_mtp3_lb_triton` and its kernel line |
| E14 VMS | FP8 MTP3, `--mamba-ssm-cache-dtype float16` | 8k@16 / 40k@16 / 90k@**11** | 450 / 414 / **312** | — | 2.95 / 3.04 / 3.14 | **1,123,048** | 0 | VMS `fp8_mtp3_ssm_fp16` |
| E14 VMS | FP8 MTP3 + FlashInfer PCIe-IPC all-reduce | 8k@16 / 40k@16 / 90k@10 | 297 / 278 / 206 (**0.68x**) | — | 2.98 | 1,019,504 | 0 | VMS `fp8_mtp3_pcie` |
| E14 VMS | FP8 MTP3, **prefix caching off** | 8k@16 / 40k@**2** | 435 / **3** | — | 2.97 / 2.59 | 1,058,219 | 0 | VMS `fp8_mtp3_noprefix`; E14R:52 |
| E14 VMS | FP8 target-only, prefix off (the oracle) | — | no speed row | — | — | **1,342,449** | — | VMS `fp8_target_only_noprefix` |
| E14 VMS | RedHat INT4 W4A16 (Marlin) MTP3, seqs 16 | 8k@16 / 40k@16 / 90k@12 | 495 / 434 / 321 | — | 3.02 / 2.98 / 2.83 | 1,297,432 | 0 | VMS `int4_mtp3_s16` |
| E14 VMS | INT4 MTP3, seqs 20 | 8k@20 / 40k@20 / 90k@12 | **518 / 439 / 332** | 25.9 / 22.0 / 27.7 | 2.97 / 2.90 / 2.92 | 1,284,445 | 0 | VMS `int4_mtp3_s20`; E14R:54 |
| E14 VMS | 2x TP2 FP8 MTP3 (DP2) | 8k@14 / 40k@2 | 191 / 22 | — | 2.79 / 2.91 | **146,756 per replica** (1.00 / 0.98) | **4** / 0 | VMS `tp2x2_fp8_mtp3`; E14R:55 |
| E14 VMS | 2x TP2 INT4 MTP3 | 8k@16 / 40k@2 | 483 / 92 | — | 2.95 / 2.80 | 422,087 | 0 | VMS `tp2x2_int4_mtp3` |
| E18 smoke v1 (10-27) | S1 flags + `--language-model-only`, FP8 MTP3, seqs 20, max_len 106,496, real Poetiq T 1.0 | ~7k → ~30k @20 | 457-466 → 345-353 | ~23 → ~17.5 | 2.5-2.9 (log median 2.72) | **1,110,415** | 0 | E18S:29-35 and the table above it; v1 log |
| E18 smoke v1 | same, steady phase 16.6-34.6 min / whole solver | @20 | **382** steady / **284** whole | 19.1 | — | — | 0 | E18S:37 |
| E18 R2 commit (10-29) | same engine | KV usage 38.9 % → 69.1 % @20 | 450 → 357; mean of 111 steady lines **382.8** | 19.1 | **2.69** (= 1 + 288,968 / 170,863); per position 0.732 / 0.543 / 0.416 | 1,110,415 | 0 | E18R2 `vllm_server_s1.log`; `e18_engine_stats.json` → metrics_end `spec_decode_*` |
| E18 R2 commit | whole solver (`scheduler_tok_per_sec`) | — | **279.2** | — | — | — | 0 | `e18_engine_stats.json` → scheduler_tok_per_sec |
| E18 R2 commit | drain phase (phase-2 re-prefill) | Running 5 / Waiting 14 | gen **4.9** while prompt 2,416.7 tok/s | — | — | — | 0 | E18R2 log, 10-29 23:08:59 |
| E18 R2 commit | prefix-cache hit over the run | — | 46,400 / 666,697 = **7.0 %** | — | — | — | — | `e18_engine_stats.json` → `vllm:prefix_cache_hits_total` / `queries_total` |
| E18 SMOKE30 v3 (10-30) | same | KV 38.8 → 46 % @20 | 479 → 404; mean of 25 lines **410.9**; whole run 241.8 | 20.5 | median 2.72 | 1,110,415 | 0 | E18v3 log; `e18_engine_stats.json` |
| E23 SMOKE_B | **G0**: FP8 MTP3, FlashInfer, PIECEWISE (control) | 40k@16 | **435.7** | 27.2 | 3.569 (accept_rate 0.856) | 1,159,379 (0.71) | 0 | E23B `sections.6_mtp_depth.mtp3_eq_G0_piecewise.live_windows["40k@16"]`: step_ms **131.1**, tokens_per_step 57.1, prefill 1,435 tok/s |
| E23 SMOKE_B | **G3**: FP8 MTP3, FlashInfer + PR #60532 backport, FULL_AND_PIECEWISE | 40k@16 | **448.9 (+3.0 %)** | 28.1 | 3.584 | 1,110,733 (0.74) | 0 | E23B `sections.3_full_graph.G3_flashinfer_pr60532`: step 127.7 ms, 57.34 tok/step |
| E23 SMOKE_B | FP8 target-only (MTP0), FULL + PIECEWISE captured | — | not measured | — | — | **1,412,664** | — | E23B `mtp0_target_only.servers.fp8_mtp0` |
| E23 SMOKE_B | **NVFP4 KV**, MTP0, FP8 weights | 120k@8 | 133.0 | 16.6 | — | **2,356,868 (1.67x)** (0.371) | 0 | E23B `5_nvfp4_kv.nvfp4_target_only`: step 60.2 ms, 8 tok/step, prefill 1,305 tok/s |
| E23 SMOKE_B | TF reference, BF16 KV (`auto`) | — | — | — | — | 619,202 | — | E23B `tf_bf16kv_reference.servers.tf_auto` |
| E23 SMOKE_B | RedHat INT4 MTP0 reference | — | — | — | — | 1,788,404 | — | E23B `4_sm89_fp8_gemm.int4_mtp0_target_only_ref` |
| E23 SMOKE_B | **PP4** (TP1xPP4), RedHat INT4, eager | 2k EXACT; **8k: CUDA illegal memory access** | — | — | — | 1,374,799 | — | E23B `1_tp1x4_pp4.e_pp4_int4_eager`, status NO_GO_8k |

### 1b. Merlin v3.0.0 (vLLM 0.32.0 fork), Qwen3.8-27B W4A16-Merlin checkpoint, 27B profiles

The TP4 server flags, from `SMOKE_A/logs/server_merlin_tp4.log:8`:
- block_size **1024**; gpu util 0.96;
- **`kv_cache_memory_bytes` 16,301,686,784** (a KV pin: "skipped memory profiling", log :196);
- fp8_e4m3 KV, prefix caching, **`mamba_ssm_cache_dtype` float16** (it overrides the model's fp32, :19), `mamba_cache_mode` align, `prefix_match_unit` 16, `enable_mamba_shared_prefix_checkpoint`, `use_replayssm`, mamba_backend triton;
- max_num_batched_tokens 4096, **max_num_seqs 24**, long_prefill_token_threshold 832;
- MTP **4** with `draft_sample_method` probabilistic;
- cudagraph_mode FULL_AND_PIECEWISE, with capture sizes [1, 2, 4, 5, 8, 10, 15, 20, 25, 30, 35, 40, 60, 80, 100, 120];
- server default chat kwargs `reasoning_effort: "medium"` with the froggeric v22.5 template.

| Experiment | Config | Context@streams | tok/s | Per-stream | Accept | KV tokens (usage) | Preempt | Evidence |
|---|---|---|---|---|---|---|---|---|
| E23 SMOKE_A | Merlin **TP1** on GPU0 (seqs 4, capture [1, 2, 5, 10, 15, 20]) | 40k@2 | 93.9 | 47.0 | **4.338** | **220,897** (0.816) = 1.69 x 131k | 0 | E23A `a_merlin_tp1_gpu0.live_windows["40k@2"]`: step 92.4 ms, 8.68 tok/step, prefill **622** tok/s, GPU peak 21,616 MiB |
| E23 SMOKE_A | Merlin TP1 | 120k@1 | 41.0 | 41.0 | 3.924 | 220,897 (0.752) | 0 | same, `["120k@1"]`: step 95.6 ms, prefill 731 tok/s |
| E23 SMOKE_A | Merlin **TP4** | **40k@8** | **421.5** | **52.7** | **4.184**; per position 0.91-0.94 / 0.82-0.83 / 0.73-0.76 / 0.65-0.66 | **1,789,034** (0.267) = 13.65 x 131k | 0 | E23A `d_merlin_tp4.live_windows["40k@8"]`: step **79.4 ms**, 33.47 tok/step, prefill **1,642** tok/s, peak 21,681 MiB on all 4. Per-position rates from `server_merlin_tp4.log` SpecDecoding lines at 05:51:13-05:51:33 |
| E23 SMOKE_A | Merlin TP4 | **120k@8** | **329.0** | 41.1 | 4.227 | 1,789,034 (0.588) | 0 | same, `["120k@8"]`: step **102.8 ms**, 33.82 tok/step, prefill 2,035 tok/s |
| E23 SMOKE_A | Merlin **TP1 x4 replicas** | 40k, 2 per replica (8 total) | **359.5** (92.0 / 91.8 / 90.0 / 85.7) | 44.9 | 4.28 (rep0) | 220,897 each | 0 | E23A `c_merlin_tp1_x4["40k"]`; E23S table |
| E23 SMOKE_A | Merlin TP4 vs TP1 correctness | 90k greedy EXACT; oracle 3 EXACT + 1 NEAR_TIE; teacher-forced (TF) 40k mean \|dlogprob\| 0.0772, rank-1 0.9883 | — | — | — | — | — | E23A `d_merlin_tp4.greedy_90k_vs_tp1`, `teacher_forced_vs_tp1` |

### 1c. vLLM 0.19.0, production era (context for scaling and acceptance)

| Run | Config | Context@streams | tok/s | Accept | KV | Preempt | Evidence |
|---|---|---|---|---|---|---|---|
| S1-60K / V production | FP8, MTP3, fp8 KV, prefix (align), FlashInfer, seqs 16 | by real running count: 1 / 5 / 10 / 12 / 16 | **44 / 163 / 268 / 293 / 352** decode; 333.3 whole run; ≈22 per stream at 16 | 3.0 (T 0.6), **2.67 (T 1.0)**, 3.45 (low effort) | — | — | PC rows 180-181 (`part_C...md:292-293`) |
| Poetiq R5 EVAL120 | V, 16 | whole solver | 281 | — | KV 99.5 % | **9** | PC row 178 |
| D66 / L66 | V, 16 at 77.8k / 12 at 155,648 | whole run | 255.8 / 254.1 | — | 99.9 % / 90 % | **66** / 0 | PC row 179; memory `d66-l66-hard-task-findings.md` §6 |
| archive (older vLLM) | CUDA graphs on vs eager | 8 sequences | **48 → 129** | — | — | — | PC row 190 (`:302`). Same row: custom AR at TP4 0.9995x; prefix caching at 64 sequences killed the engine; TP2 had 17k KV and failed |
| E20 commit (S1COMMIT4) | S1 engine, 0.19 | — | — | — | **1,353,600** | 0 | E20 §3 |
| E19 (0.19 correctness) | S1 with bf16 KV vs fp8 KV | — | — | MTP accept min / median 0.58 / 0.80 per position | 176,800 (bf16) vs 337,600 (fp8) at the E19 settings | — | E19 table |
| E21 Duck (12-20 streams, real agent) | 0.19 FP8 s18 SSM16 / 0.32 INT4 s20 / 0.19 FP8 s16 SSM16 / 0.32 FP8 s16 SSM16 | ~11.5 running | p50 **329 / 252 / 327 / 278** | MTP 1.9 accepted per draft | blocks 855 x 1600 (fp32 SSM) vs 1627 x 832 (fp16 SSM) | 0 | E21:11-16; memory `experiment-21-duckb-vllm.md`, final table (prefix hit 89.0 / 61.6 / 88.6 / 68.3 %) |

### 1d. Communication and kernel microbenchmarks (4x L4)

| What | Number | Evidence |
|---|---|---|
| NCCL all-reduce, ~983 KB (the 27B TP4 decode size), inside a CUDA graph | **343.9 µs** per call (E14). E23 P5 baseline: NCCL graph **336.23 µs** | E22F:47; N30 "NCCL env arm"; E23S "#2 host all-reduce" |
| 64 MiB eager all-reduce | 20.9 ms = 3.21 GB/s algbw, 4.82 busbw; the SHM transport with 2 channels; PCIe link gen max 3; all GPUs PHB on one NUMA node; NCCL 2.29.7 | N15:19; E22F:47-49 |
| P2P (all pairs `can_device_access_peer`) | **1,880 µs** per call = **5.5x slower** than NCCL SHM | E14F:41 |
| `NCCL_SHM_USE_CUDA_MEMCPY=1` | slower, and times out in graphs | E22F:48 |
| NCCL tuned channels 4 / 8 / 16 / 32 | all slower: 351-429 µs vs 336 | E23S |
| Host-staged 4-rank all-reduce (P5, bf16) | 361.68 µs = **0.93x** NCCL | E23S |
| INT8-payload all-reduce (P5) | 237.73 µs = **1.41x** (microbenchmark only; never run inside vLLM) | E23S |
| FlashInfer 0.7 PCIe-IPC all-reduce inside the engine | vLLM 0.68-0.73x; SGLang 0.39-0.42x | VMS `fp8_mtp3_pcie`; E14R:24 |
| symm-mem; FlashInfer all-reduce at world size 2 / 4 | unsupported on sm_89 | E22F:57 |
| FP8 MLP GEMM per forward (gate_up + down, real weights) | M=1: Marlin W8A16 17,011 µs vs CUTLASS W8A8+quant 18,690 (Marlin 9 % faster). M=16: 17,136 vs 19,153 (Marlin 11 % faster). M=80: 24,294 vs 22,932 (CUTLASS 6 % faster). BF16 cuBLAS 33,101-39,030 | E23S "#4"; E23A `4_sm89_fp8_gemm.p4_bench_gpu3.summary_per_m` |

### 1e. Where a decode step goes (TP4 27B)

| Engine | Breakdown | Evidence |
|---|---|---|
| SGLang INT4 + DFlash2-8, 12 running, 116 ms step | NCCL all-reduce 140 calls **50.9 ms (44 %)**; all-gather 8.1 (7 %); Marlin INT4 25.3 (22 %); other GEMM 14.5; GDN 8.0; attention 2.7; sampling 2.4. **Communication 51 %** | N30 "Profiler"; E14F §1.2 |
| vLLM 0.32 FP8 MTP3, 16 running, 40k (G0 and G3) | NCCL **35.6-35.7 %**, Marlin GEMM 27 %, attention 19.8 %, other GEMM 9 %, GDN 6.3 %, sampling 0.7 %. NCCL time includes rank-skew waiting | E23TR `g0.by_category_pct`, `g3.by_category_pct` |
| Graph residency, vLLM G0 (PIECEWISE) | 1,809 graph launches vs 7,155 kernel launches; **25.5 % eager kernels**; per forward 65 graphs + 160 kernels, 144 of them eager compute. Attention (1,539) and GDN (2,592) all eager | E23TR `g0` |
| Graph residency, vLLM G3 (FULL, PR #60532) | 116 graph launches vs 1,566 kernel launches; **5.2 % eager**; per forward 1 graph + 16 kernels, 0 eager compute; FULL in 29 of 29 scopes. Sampling stays eager (203) | E23TR `g3` |
| Step-time fits | SGLang: step ≈ 26 ms + 0.94 ms per verify row (E22F:50). Derived from the vLLM 0.19 by-running table at accept 2.67: 1 → 60.7 ms, 5 → 82, 10 → 100, 12 → 109, 16 → 121 ms, i.e. **≈ 57 ms fixed + ≈ 4 ms per stream (≈ 1 ms per verify row)**. Derived for Merlin TP4: 79.4 ms for 40 verify rows (8 x 5); stock FP8 G0: 131.1 ms for 64 rows | E22F:50; PC:292; E23A / E23B |

### 1f. SGLang 0.5.21 reference rows (same hardware; only to compare engines)

| Config | tok/s at 8k / 40k / XL | Accept | Evidence |
|---|---|---|---|
| INT4 + DFlash2 W6, seqs 16 | 572-583 / 521-544 / 370-403 (8) | 4.1-4.3 | E14R first rows |
| DFlash2 W6, seqs 20 (best) | **597 / 536 / 364 (7)** | 4.17 / 4.21 | E14R:20 |
| target-only | 405 / 354 / 275 | — | E14R |
| EAGLE MTP 3/1/4 | 478-492 / 396-402 / 292-296 | 2.9-3.0 | E14R |
| 2x TP2 DFlash W6 | 658 (sum) / 308 (sum): 152k KV per replica, 3 streams at 40k | — | E14R |
| Real run E15P Poetiq EVAL120 6 h (15 streams) | **389 whole run** | — | E14F §1.3 |
| Width sweep at ~6k (E14A R2) | W8 505.0 (114.0 ms) / W6 531.4 (96.4 ms) / W4 526.8 (74.5 ms) / target-only 339.6 (35.3 ms) | 4.86 / 4.24 / 3.27 | N30 table |

---

## 2. What worked (GOOD), and why

1. **Speculative decoding is the biggest single lever at the concurrency we can afford.**
   - SGLang DFlash2-8 = 1.49x target-only (N30).
   - On vLLM, Merlin MTP4 at 120k@8 gave 41.1 per stream. NVFP4 MTP0 at 120k@8 gave 16.6 per stream, on the same engine family but with different weights (E23A / E23B). So MTP gave about 2.5x per stream while the step grew only 1.7x (102.8 vs 60.2 ms).
   - **Why:** the step is dominated by fixed cost (communication + weight read + launches). Extra verify rows are cheap until the row count gets large.
2. **Merlin on the 27B gave the best per-stream vLLM speed.**
   - 52.7 tok/s per stream at 40k@8 with MTP4 accept 4.18. Stock FP8 MTP3 gives 27.2 at 40k@16 (E23A / E23B).
   - **Why:**
     - acceptance is higher: MTP4 with probabilistic draft sampling, 4.18-4.34 vs 3.0-3.6;
     - the weights are INT4;
     - the KV pool is large: 1.79M tokens at TP4 thanks to the KV pin, fp16 SSM and INT4 weights. That is 13.65 x 131k.
   - Correctness held: TP4 vs TP1 greedy 90k EXACT; TF rank-1 0.988.
3. **More streams help on 27B, but sub-linearly.**
   - FP8 16 → 20 → 24 streams: 434 → 452 → 472 at 8k and 399 → 423 → 424 at 40k (VMS).
   - vLLM 0.19 by running count: 44 → 163 → 268 → 293 → 352 tok/s for 1 → 5 → 10 → 12 → 16 streams (PC:292).
   - **Why:** the fixed cost of ≈ 57 ms is amortised; each extra stream adds ≈ 4 ms (derived, §1e).
4. **SSM fp16 bought capacity.**
   - +10 % KV tokens (1,123,048 vs 1,019,504), one more stream at XL (11 vs 10), XL 312 vs 281 (+11 %) (VMS).
   - The vLLM Mamba-aligned block falls from 1600 to 832 (E21:11, :38), so more streams fit and prefix granularity is finer.
   - No measured Duck accuracy loss: NB2 B 33.19 vs NB2 A 33.33 task-norm (memory E21 final table).
   - Merlin's own 27B profile uses fp16 SSM (`server_merlin_tp4.log:19`).
   - **Cost:** the GDN verify falls back from the fused CUDA kernel to Triton (E22F:55). That was net positive here.
5. **`--language-model-only` freed KV.** 1,006,517 → 1,110,415 tokens (**+10.3 %**) at seqs 20, the same max_len and util (VMS `fp8_mtp3_s20` vs E18S "KV").
6. **Prefix caching ON (align) with MTP is both correct and necessary.**
   - E19: arm D (MTP3 + prefix on) PASS on 66/66 rows.
   - Prefix off makes the 40k band recompute the shared system block. It fell to 2 running and 3 tok/s (VMS `fp8_mtp3_noprefix`).
7. **NCCL SHM default is the fastest collective on this VM.** Every alternative lost:
   - P2P 5.5x slower;
   - PCIe-IPC 0.68x (vLLM) / 0.39x (SGLang);
   - tuned channels 351-429 µs;
   - host-staged 0.93x;
   - `NCCL_SHM_USE_CUDA_MEMCPY` slower;
   - custom AR at TP4 0.9995x.
   Keeping `--disable-custom-all-reduce` (stock and Merlin profiles) is right (§1d).
8. **FULL graph replay for MTP verify works** with PR #60532 (G3).
   - FULL in 29/29 scopes; eager kernel share 25.5 % → 5.2 %; attention and GDN graph-resident (E23TR).
   - Speed +3.0 % at 40k@16 (448.9 vs 435.7) for −4 % KV (1,110,733 vs 1,159,379).
   - Small but real. It is small because NCCL (36 %) and Marlin (27 %) dominate the step at 16 streams, not launch overhead.
9. **Weight prefetch and engine-start work paid off.** Weight load 64 s → 4.8 s and engine start 600 → 532 s (memory `d66...` lesson 7). The vLLM 0.32 venv install is 175-210 s (E18S; VMS header).
10. **INT4 weights are faster than FP8 on the same engine.** +14-19 % at 8k, +9-10 % at 40k, +14-18 % at XL (VMS int4 vs fp8 rows).
    - **Why:** decode at 16-20 streams is weight-bandwidth bound in its GEMM part.
    - **But see BAD #9:** INT4 lost about 5 task-norm points in Duck.

## 3. What failed or hurt (BAD), and why

1. **FULL_AND_PIECEWISE with spec decode is downgraded to PIECEWISE on sm_89 + FlashInfer**, in every stock and Merlin vLLM 0.32 log:
   - `vllm_server_ref_fp8_mtp3_s16.log:227`;
   - E18R2 log :223-229;
   - `server_merlin_tp4.log:206-209`: "FlashInferBackend support: UNIFORM_SINGLE_TOKEN_DECODE".

   In the Merlin 27B profile, the **draft** ran with `cudagraph_mode=prefill:PIECEWISE,decode:NONE` (:211-213), and the log says "FLASHINFER cannot update draft decode metadata in place; falling back to rebuilding attention metadata eagerly". The RecoverSSM commit said "no graph for this step (n=24, 0 captured); running staged-eager" (:240-245).

   So Merlin's "uniform multi-token graph" patch did **not** give FULL verify graphs for the 27B on L4 with FlashInfer.
2. **PP4 on stock vLLM 0.32 + INT4 (eager) crashed** with an illegal memory access at 8k, after 2k was EXACT (E23B `e_pp4_int4_eager`, upstream #63505). Compiled PP4 was never tried.
3. **TP1 replicas and 2x TP2 are KV-bound on 27B.**
   - Merlin TP1: 220,897 tokens per GPU = 1.69 x 131k. Four replicas at 2 streams each gave 359.5, against TP4 at 8 streams 421.5 (E23A).
   - 2x TP2 FP8: 146,756 tokens per replica, KV max 1.00, 4 preemptions, 22 tok/s at 40k (VMS).
   - **Why:** each replica holds a full weight copy, so the KV that remains per GPU is small.
4. **Deeper MTP at 16 streams lost on stock vLLM.**
   - MTP4: 399 / 381 / 267, against MTP3 434 / 399 / 281.
   - MTP4 also used 6 % less KV: 962k vs 1.02M (VMS).
   - **Why:** verify rows grow as seqs x (1+k), while acceptance gains shrink (accept 3.53 vs 3.05).
   - MTP2 won only at 8k (+3 %).
   - The trade-off depends on concurrency. Merlin's MTP4 won at only 8 streams.
5. **MTP + prefix caching OFF diverged** in vLLM 0.19: 4 flips of 4-12 nat (E19 arm B). Never use "prefix off" as a safety flag with MTP.
6. **Mamba align mode loses almost all reuse of generated tokens.**
   - Poetiq: 7.0-7.7 % prefix hit (E18S:50; E18R2 stats). The drain re-prefilled at 2.4k tok/s while decode fell to 4.9 tok/s.
   - Duck: 0.32 gave 54-68 % against 82-89 % on 0.19, so real Duck runs on 0.32 were **slower** than on 0.19: p50 252-278 vs 327-329 (E21).
   - **Why:** align mode caches the GDN state only when a step ends exactly at a block boundary, and MTP steps of 2-5 tokens rarely land there (E18S:52, vLLM `config/cache.py:189-195`).
7. **Whole-run throughput is about 0.73x of the steady window.**
   - E18: 279-284 whole vs 382 steady.
   - SGLang E15P: 389 whole vs 521-583 steady.
   - **Why:**
     - prefill stalls: a new request blocks decode, and the shared step makes decode nearly zero during long prefills;
     - tail drain;
     - context growth: 466 → 345 tok/s from 7k to 30k (E18S); Merlin 421 → 329 from 40k to 120k.
8. **NVFP4 KV is not validated.**
   - 1.67x tokens, but the TF drift vs BF16 KV was mean 0.379 / max 13.2. For scale, TP-numerics drift is mean 0.077 (E23S).
   - The rank-1 metric used then was buggy (E23FIX #2).
   - vLLM captures only single-token decode graphs with it, and a fork measured −13-17 % (E22F §4.6, claim).
   - It was only measured with MTP0.
9. **INT4 RedHat weights cost accuracy.** Duck task-norm was 28.19 (0.32 INT4) against 33.19 (0.32 FP8) (memory E21 final table). INT4 + MTP3 on vLLM 0.19 generated garbage (E21 smoke 1).
10. **More streams hurt at XL.** FP8 seqs 24 at 90k fell to 256 (9 streams) against 281 at 16, because the KV fit cut the running count anyway (VMS).
11. **vLLM 0.32's profiled KV pool is smaller than 0.19's at the same flags.**
    - 0.32 at util 0.90: 1.02-1.11M; 0.19: 1.35M (E20 §3 vs VMS / E18S).
    - This is derived across runs; the 0.32 rows also differ in seqs and `--language-model-only`. The direction is about −18 to −25 %.
    - On top of that:
      - MTP3 costs 18-24 % of the pool (1,342,449 → 1,019,504 in VMS; 1,412,664 → 1,159,379 in E23B);
      - FULL graphs cost about 4 %.
12. **Two other lost levers:**
    - Linear-kernel swaps gave nothing: Triton W8A8 = Marlin W8A16 (438 vs 434), and CUTLASS W8A8 loses at M ≤ 16.
    - SGLang int8 Mamba checkpoints OOMed (N30).

## 4. Measurement-method lessons

1. **Measure only after the prefill has settled.**
   - vLLM counts a request as "running" from its first scheduled prefill chunk (E14F:140).
   - Seen in Merlin TP4: Running 8 for about 3 minutes at 0.5-1.7 gen tok/s during the 8 x 40k prefill (`server_merlin_tp4.log`, 05:48:03-05:50:53).
   - VMS requires 3 equal prompt-token readings. E23 requires running_min == running_max == level, waiting 0, preemptions 0.
2. **Assert the real running count and read the server's caps at launch.** These come from the server log, not from the requested level (memory `verify-real-concurrency-in-benchmarks.md`):
   - KV tokens;
   - max concurrency;
   - capture sizes;
   - graph mode;
   - kernels selected.

   History: SGLang auto-capped at 6/10 in E11-03, and the delayer gave 11 of 12 (N30 latch: 100 % of samples at 11).
3. **Fake fixtures must come from real logs.** The E12 rev 6d controller stepped down on queue = 1, which is SGLang's normal state; my made-up "healthy" fixture had queue = 0. Controllers must be replayed on recorded series.
4. **Acceptance in a benchmark is inflated against real reasoning, all at T = 1.0:**
   - E23 synthetic ARC-grid prompts: MTP3 accept **3.57**;
   - VMS prompts: 3.05;
   - E18 real Poetiq reasoning: **2.69**, with per-position rates 0.73 / 0.54 / 0.42.

   The E23 sample tails show grid-row copying, which MTP predicts easily. Any 1500 projection must use real xhigh-reasoning acceptance from `vllm:spec_decode_num_accepted_tokens_per_pos_total`.
5. **Keep the units separate:**
   - steady decode;
   - e2e from submission (VMS: 322 / 195 vs 434 / 399);
   - whole-run scheduler tok/s (E18: 279);
   - task throughput.

   The memory note for E18 says the same: "not in the same unit".
6. **The "Avg prompt throughput" log line is bursty** (9,000 and 27,999 tok/s spikes in `server_merlin_tp4.log`). Compute prefill rate from the counter deltas over the admission time, as E23's `admission.prefill_tok_s` does: 622-2,035 tok/s.
7. **A graph mode needs proof in two places:**
   - **The server log:** a downgrade warning is decisive; `e2-identity` / `cudagraph_mode` lines appear per role.
   - **A profiler trace, per scope:** the annotation pattern is `execute_context_N(N)_generation_N(N)`. The first parser looked for `gpu_model_runner` and found 0 forwards (E23FIX #1). Expect 1 graph and 0 eager compute per FULL scope (E23TR).
8. **Rank skew hides inside NCCL time** in profiler traces (E23TR note; N15:239-242). A faster transport removes only the min-over-ranks part.
9. **A microbenchmark gain is not an engine gain.** P5 INT8 1.41x and P4 CUTLASS W8A8 are microbenchmarks only (E23 rev 3c fix list P2 #17).
10. **Synthetic engine gains did not carry over to agent runs.** 0.32 beat 0.19 in VMS but lost in Duck because of prefix reuse (E21). Every engine change needs a real-harness replay.
11. **Correctness gates that worked:**
    - greedy parity vs a target-only oracle, with classes EXACT / BOUNDARY (near-tie) / large-margin flips > 1 nat;
    - an A/A, because output is never identical across restarts (#38009);
    - a canary for cross-request leaks;
    - an XL (~85-90k) parity row.

    **Bugs found in our own gates:**
    - `rank1_agreement` measured "forced token is rank 1 on both sides", not the same top-1 id (E23FIX #2);
    - the NVFP4 gate passed with one band (E23FIX #3);
    - a distinct-character degenerate check flagged ARC grids (E14F §4).
12. **Prompt rendering matters.** The E23 SMOKE prompts were medium renders (no xhigh system line), and Merlin's server defaults to `reasoning_effort: "medium"` with the froggeric template (E23H fix #4). Speed rows and accuracy rows must use the production render.
13. **Context bands matter.** Report every number with its context and running count. Context moves throughput by 25-30 %:
    - 7k → 30k: 466 → 345 (E18S);
    - 40k → 120k: 421.5 → 329 (Merlin TP4).
14. **DP / multi-replica metrics:**
    - `--max-num-seqs` is per DP rank;
    - `/metrics` is labelled per engine, so KV usage is the per-engine max, not the sum (E14F §2.2).
15. **Two run-level practices:**
    - **Readiness:** wait for the server's own KV line, kill the process group with SIGKILL, and arm a 10-min silence watchdog (E18 rev 2).
    - **Validation:** a Kaggle CPU kit on the pinned Py3.12 image catches crashes such as `Thread._stop` that a local Py3.14 run misses (E14F §4).

## 5. Transfer to Flash-Next on Merlin (fn-quad-24g-sm89: TP2xPP2+EP, FP8 QSA KV, MTP K3, prefix caching, mamba align, block 4096, chunk 1024, seqs 8)

**Derived arithmetic for the target:**
- At seqs 8, 1500 tok/s needs 187.5 tok/s per stream.
- With 3.0 accepted per step that is 24 tokens per step, i.e. a **16 ms step**.
- Our best measured per-stream number on L4 is 52.7: the 27B Merlin TP4 at 40k@8, with a 79.4 ms step.
- So FN needs about 5x shorter steps than 27B TP4, or more streams with long accepts. PLAN example: 12 x 3.0 / 24 ms.

The table below maps each past lever to FN.

| Lever (27B evidence) | Applies to FN? | How to adapt |
|---|---|---|
| **MTP depth** (stock: MTP3 best at 16 streams, MTP2 best at 8k, MTP4 −5-8 %; Merlin MTP4 at 8 streams accept 4.18) | **Applies; re-measure** | At seqs 8 the verify rows are few (8 x 4 = 32), so deeper K is cheaper than on the 27B at 16-24 streams. Sweep K 1/2/3 (4 if the FN MTP head allows it), at real running 4/6/8 and contexts 8k/40k/90k. Score = per-position accept from `vllm:spec_decode_num_accepted_tokens_per_pos_total` on **real xhigh reasoning at T = 1.0**, not synthetic grids. Use 2.69 (E18) vs 3.57 (E23 synthetic) as the inflation warning. K also changes graph capture sizes and the GDN spec-state slots (state x (1+k)). |
| **Draft sampling** (Merlin 27B: `draft_sample_method probabilistic`) | Applies | Read the FN profile value from the RC2 log `non-default args`; record it in the telemetry. |
| **KV dtype FP8** (E19: FP8 = BF16 in correctness, 2x pool) | Applies (already FP8 QSA KV) | Check the FP8 KV scales. Memory says scale 1.0 everywhere except RC2. Keep the BF16 KV boot as the oracle (RC2 does both). |
| **NVFP4 KV** (1.67x tokens, drift unvalidated, single-token graphs only) | **Probably does not apply** | The FN QSA KV path is Triton and accepts bf16 / fp8_e4m3 (memory `flashnext-vllm032-support.md`). NVFP4 would need a source check of the QSA KV store first. Not a near-term lever. |
| **FULL graphs for MTP verify** (stock: downgraded on FlashInfer; PR #60532 +3 % at 16 streams; Merlin 27B profile still PIECEWISE, draft decode NONE) | **Applies, and probably worth more for FN** | FN at seqs 8 with EP-MoE (many small kernels) and the PLE fetch is likely launch-bound. PR #60532's author reported +36-47 % at batch 1 on SM80 (claim, E22F §4). QSA is a split op (`vllm::qwen4_exp_qsa_with_output` is in `splitting_ops`, `vllm_server_ref_fp8_mtp3_s16.log:27`), so PIECEWISE breaks at every QSA layer. Steps: (1) grep the RC2 log for the "not supported with spec-decode" downgrade line and the `e2-identity ... cudagraph_mode=` lines (target and draft); (2) profile one decode window and use the E23 rev 3h per-scope parser to count graph vs eager per forward; (3) only then pursue a FULL path for the QSA backend. |
| **Capture sizes** (Merlin 27B: seqs 24 x MTP4 → captures to 120; TP1 seqs 4 → [1, 2, 5, 10, 15, 20]) | **Applies** | The sizes are counted in **tokens = seqs x (1+k)**; the FN profile's "capture 1..32" = 8 x 4. This answers PLAN open question 4 from config evidence (still check the FN log). L7 (seqs 10/12) **must** extend capture to 40/48 tokens, or the larger batches fall back to eager/piecewise. Also check the FN log for "no graph for this step ... staged-eager" (the RecoverSSM commit graph, `server_merlin_tp4.log:240-245`). |
| **TP1 replicas / 2x TP2 DP** (27B: KV-bound, 0.85x at equal streams; 2x TP2 146-422k per replica) | **Does not apply** (TP1 impossible; DP2xTP2 very doubtful) | FN weights (~115 GiB checkpoint, PLE in host) cannot fit TP1, and DP2xTP2 doubles the weights per GPU. The 27B lesson is that KV per replica binds first. Run PLAN Q5 only as a memory census (weights per GPU + PLE per rank + KV) before any boot. |
| **TP4 vs TP2xPP2** (27B TP4: comm 36-51 % of the step; NCCL SHM 336-344 µs per ~1 MB) | **Applies as L8** | TP2 all-reduces have half the participants. EP adds all-to-all / all-gather, and PP adds activation send/recv over the same slow PCIe (P2P 5.5x slower; NCCL picks SHM). PP2 at seqs 8 has a bubble unless vLLM keeps ≥ 2 batches in flight. Telemetry: NCCL transport lines per group (tp / pp / ep), per-op time in one profiler trace, and running per stage. |
| **PP stability** (stock 0.32 PP4 INT4: illegal memory access at 8k) | **Risk flag** | The FN profile is Merlin's own PP2 path, but the 27B PP failure was in stock 0.32 compiled/eager PP. Keep RC2's fail-fast and long rows (8k, 40k, 90k) before any speed arm, and stop at the first illegal-access or NCCL-watchdog line. |
| **Custom all-reduce** (TP4: refused / no effect; at world size 2 it turns on if the P2P test passes, and P2P is 5.5x slower) | **Applies (critical at TP2)** | With TP2 the legacy custom AR **can** turn on (SGLang `custom_all_reduce_utils.py:442-512`; vLLM behaves similarly, unverified). Confirm that the FN profile passes `--disable-custom-all-reduce` (the 27B Merlin profile does, :8) and that the log's all-reduce backend line is PYNCCL. |
| **NCCL env tuning, PCIe-IPC, host-staged AR** (all lost) | **Does not apply**; re-test only INT8 AR later | Keep the NCCL defaults. INT8 AR (1.41x microbench) would need an engine integration and a TF drift gate. It is low priority until a trace shows comm dominating the FN step. |
| **Async scheduling** (E19: no correctness difference on 0.19; never speed-tested) | Applies as the default | Keep the default (on). Do not spend an arm on it. |
| **Chunk size** (27B: 4096; Merlin warns "max_num_scheduled_tokens is set to 4096 based on the speculative decoding settings ... suboptimal"; prefill 1.4-2.0k tok/s at TP4, 0.6-0.7k at TP1) | **Applies (PLAN L2)** | FN chunk 1024 means 40 steps per 40k prompt; every step pays the full fixed cost (MoE weights + PLE fetch). Measure prefill tok/s from the counters (not the bursty log line) and the decode tok/s of the other streams during prefill. If decode collapses (E18 drain: 4.9 tok/s), test 2048 together with Merlin's `long_prefill_token_threshold` (832 on 27B). Look for the same "max_num_scheduled_tokens" warning in the FN log. |
| **Prefix caching mode** (align: 7 % Poetiq hit, 55-68 % Duck; prefix OFF + MTP diverges; Merlin adds shared prefix checkpoints, CoW tail retention, finished-turn retention, cost-aware admission, `server_merlin_tp4.log` 05:46:01 lines) | **Applies (keep ON); measure per turn** | Block 4096 means a GDN checkpoint only every 4096 tokens, which is coarser than 27B's 832-1600. Check the FN log for the same Merlin retention lines. Measure the hit per turn in a 2-turn replay: a previous reply reused, or a continuation after an abort. Merlin's cost-aware admission ("a cold miss waits up to 120 s") can hold requests back, so assert the real running count. |
| **max-num-seqs** (27B: 16 → 24 is +9 % at 8k, 0 at 40k, −9 % at XL) | **Applies (L7), bounded by KV** | The PLAN KV pin gives ≈ 5 streams at 70k, so seqs above the KV fit just queue. Sweep the seqs that KV allows at the measured context mix. Report the real running count and KV usage max. Extend the capture sizes together with seqs (see the capture-sizes row). |
| **GPU util / KV pin** (Merlin 27B: `kv_cache_memory_bytes` 16.3 GB skips profiling; peak 21,681 of ~23,030 MiB; 0.32 profiled pool 18-25 % smaller than 0.19) | **Applies (L1)** | The pin is the right mechanism. It removes the profiler's conservative reserve, which on 27B was the 0.32 vs 0.19 gap. Grow it in +0.25 GiB steps with the boot-OOM retry. Record free VRAM after graphs and during the peak prefill chunk (prefill activations were 1.5-2.6 GB on another GPU, PC row 214). |
| **`--language-model-only`** (+10.3 % KV on 27B) | **Applies (L6)** | Same mechanism: the vision tower becomes `StageMissingLayer` (PLAN §2). Read the VRAM delta from the boot log. |
| **SSM state dtype** (fp16: +10 % KV, block 1600 → 832, XL +11 %, no Duck accuracy loss; Merlin 27B profile uses fp16; the fused GDN verify falls back to Triton) | **Applies (L5)** | First read the active dtype in the RC2 log (the "mamba_ssm_dtype ... user-specified" line). If FN runs fp32 SSM, fp16 shrinks the state x (1+k) slots and the aligned block. It needs the long-reasoning correctness gate, since Nemotron reported verbosity +40 % from a straight FP16 cast (PC row 206, claim). |
| **Weight quantization / linear kernels** (INT4 Marlin +9-19 % vs FP8 Marlin W8A16; CUTLASS W8A8 loses at M ≤ 16) | Partly applies | FN is already INC W4A16 on the Marlin MoE path, so there is no FP8 GEMM lever. Watch the Marlin MoE kernels at small M per expert (8 seqs x 4 tokens spread over EP experts). |
| **DRAFT_HEAD int8** (L4) | No 27B evidence | Only Merlin's CHANGELOG claim. Measure it on the FN with acceptance per position. |
| **Engine start / load** (prefetch 64 → 5 s; capture 51-101 s; relaunch ~8 min) | Applies | Budget every relaunch in the FN notebooks (the 115 GiB checkpoint + PLE pin are far larger than the 27B's). One boot per entry point (PLAN §4). |
| **Harness-level run average** (0.73x of steady: prefill stalls, tail, context growth) | **Applies (L10)** | The 1500 target must be checked as whole-run scheduler tok/s, not as a 30-s window. Counter-measures: prefix reuse on second turns, staggered admission, keeping the tail full of new attempts. |

## 6. Open questions

1. Does the FN target forward on Merlin fn-quad get FULL graphs with MTP K3? QSA is a split op, and on 27B with FlashInfer it stayed PIECEWISE. Is the draft decode graph NONE again? What is the eager-kernel share per forward? (Next: RC2 log grep, then one profiled window with the rev 3h parser.)
2. What is the FN decode-step breakdown (MoE GEMM / EP collective / TP2 all-reduce / PP send-recv / PLE host fetch / QSA / GDN / sampling)? The 27B step had a ≈ 57 ms fixed part; FN needs ≈ 16 ms at seqs 8 for 1500.
3. What is the real FN MTP acceptance per position on xhigh reasoning at T = 1.0, versus synthetic prompts? And how does it change K's optimum at 4/6/8 running?
4. Does Merlin's PP2 keep ≥ 2 batches in flight at seqs 8? How large is the bubble? Is PP2 stable at 40k-120k? The 27B stock PP4 crashed at 8k.
5. With TP2, is the custom all-reduce disabled in the FN profile? Which NCCL transport does each group (tp / pp / ep) use?
6. What prefix-hit rate do Merlin's align retention features give at block 4096 on a 2-turn replay and after an abort? Does the cost-aware admission (≤ 120 s wait) reduce the real running count?
7. What is the active SSM dtype in the FN profile? What do fp16 vs fp32 do to KV tokens, the aligned block size and long-reasoning correctness?
8. Prefill: what are FN prefill tok/s at chunk 1024 vs 2048, and decode tok/s during prefill? Does the "max_num_scheduled_tokens set to 4096 based on spec decoding" warning appear?
9. Does seqs 10/12 need explicit capture sizes to 40/48 tokens? Does the boot log show a fallback for the larger batches?
10. E23 FULL (esfersami50) is still running (E23FULL). P5 ended BENCH_ERROR; Merlin TP1 / x2 / x4 / TP4 and MTP0 / MTP1 are DONE; the INT4 arms were SKIPPED_REQUIRED_ARTIFACT (INT4 input None). Its Merlin TP4 sweep at 16/20/24 streams and 90k/120k, plus the FULL-graph G1/G2 rows (TRITON_ATTN), are the best remaining 27B evidence on how Merlin scales with streams. Read them when the output lands.
11. NVFP4 KV on the FN QSA path: is it supported at all (source check)? Only after that, a correctness gate with both bands and a fixed top-1 metric.
12. The archive "graphs 48 → 129 tok/s at 8 seqs" (PC row 190) contradicts the upstream "#59548 PIECEWISE ≈ 1.01x eager" claim (E22F:56). Neither was measured on 0.32 / Merlin. An FN eager-vs-graph A/B would settle how much launch overhead matters on this model.
