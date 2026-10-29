# Findings: vLLM 0.32 and SGLang 0.5.21 on Kaggle 4× L4 (Qwen3.8-27B), 2026-10-29

Target: **1500+ tok/s sustained aggregate decode** (done on this hardware by the NVIDIA team). The sections below are the facts we have for the two engines we optimize, and the levers E22 measures.

## 1. Measured speed (TP4, fp8_e4m3 KV, prefix caching on; tok/s at 8k / 40k / XL ~90k context)

### vLLM 0.32.0 (VMS matrix, hosen42, 2026-10-26; `vms_report.md`)
| Arm | Streams | tok/s | Accept length | Gate |
|---|---|---|---|---|
| ref FP8 MTP3 | 16 | 434 / 399 / 281 | 2.96 / 3.05 / 3.08 | PASS (BOUNDARY) |
| FP8 MTP3 | 20 | 452 / 423 / 281 | | PASS |
| FP8 MTP3 | 24 | 472 / 424 / 256 | | PASS |
| FP8 MTP3 + SSM fp16 | 16 | 450 / 414 / 312 | | PASS |
| FP8 MTP2 / MTP4 | 16 | 448 / 397 / 260 and 399 / 381 / 267 | | PASS |
| FP8 MTP3, `--linear-backend triton` | 16 | 438 / 398 / 283 | | PASS |
| FP8 MTP3 + PCIe-IPC all-reduce | 16 | 297 / 278 / 206 (0.68×) | | PASS |
| INT4 W4A16 MTP3 | 16 | 495 / 434 / 321 | | diagnostic |
| **INT4 W4A16 MTP3** | **20** | **518 / 439 / 332** | | diagnostic (coherent at 135k in E21 smoke 2) |
| 2×TP2 | | KV-limited (0.57× at 40k) | | |
| prefix caching off | | 40k recomputes the prompt | | MTP + prefix off drifts (E19) |

### vLLM 0.32 under the Duck harness (E21 smokes, 20-minute windows, ~12 running)
| Config | p50 tok/s | Prefix hit | Preemptions |
|---|---|---|---|
| FP8 SSM16 s16 | 329 | 55 % | 0 |
| INT4 s20 | 322 | 56 % | 0 |

On 0.19 the same prompts hit 69-82 %. 0.32 caches fewer turns (align mode).

### SGLang 0.5.21 (EMS SAFE, hosen42, 2026-10-26; `ems_report.md`)
| Arm | Streams | tok/s (8k / 40k) | Accept length | Step ms (8k / 40k) | Gate |
|---|---|---|---|---|---|
| ref DFlash2 W6 | 16 | 583 / 521 | 4.21 / 4.11 | 115.7 / 127.3 | PASS |
| **DFlash2 W6** | **20** | **597 / 536 (XL 364)** | 4.17 / 4.21 | 140.3 / 157.1 | PASS |
| DFlash2 W4 | 16 | 592 / 526 | 3.30 / 3.31 | **89.2 / 100.9** | PASS |
| DFlash2 W8 | 16 | 573 / 477 | 5.00 / 4.47 | 138.8 / 149.9 | PASS |
| EAGLE MTP 3/1/4 | 16 | 478 / 402 | 2.88 / 2.94 | 96.4 / 117.3 | PASS |
| EAGLE MTP 2/1/3 | 20 | 542 / 413 | 2.48 / 2.45 | 92.0 / 118.2 | PASS |
| target only (no speculation) | 16 | 405 / 354 | | **39.5 / 45.1** | oracle |
| DFlash W6 + PCIe-IPC | 16 | 230 / 202 (0.39×) | | 299.6 / 309.9 | PASS |
| DFlash W6 | 24 | capped by KV | | | |

Under the Duck harness: 45K30K reached 451 tok/s at 16 running. EVAL120 6 h scored 39.31 % task-norm (R3K2).

## 2. Where the step time goes (measured)
- **All-reduce.** The SGLang TP4 DFlash profile: all-reduce 50.9 ms + all-gather 8.1 ms of a 116 ms step = **51 % communication**.
- **NCCL path.** Default NCCL on this VM is SHM with 2 channels: **343.9 µs** per ~1 MB all-reduce inside a CUDA graph; 64 MiB reaches 4.90 GB/s busbw.
- **P2P is slower.** P2P works (`can_device_access_peer` True on all pairs) but is **5.5× slower** (1,880 µs). This is why PCIe-IPC lost. `NCCL_SHM_USE_CUDA_MEMCPY=1` is slower and times out in graphs.
- **Topology.** The VM shows PCIe link gen max = 3. All four GPUs are PHB on one NUMA node. NCCL 2.29.7, driver 580.178.04.
- **Weights are a small part.** Reading the weights is only ~16-22 ms of a ~107-125 ms step. Step ≈ 26 ms + 0.94 ms per verify row (fit over four widths, ±3 ms). The target of 1500 tok/s at 16-20 streams needs a ~30-37 ms step at today's accept lengths (more streams or longer accepts relax it).

## 3. Kernels really selected (from server logs)
- **vLLM 0.32 FP8 linear** = `MarlinFP8ScaledMMLinearKernel`: weight-only FP8 (W8A16), no FP8 tensor cores on sm_89. CUTLASS block-FP8 needs SM90+.
- **vLLM 0.32 INT4 linear** = `MarlinLinearKernel` for CompressedTensorsWNA16, plus mixed-precision linear.
- **vLLM GDN spec-decode verify:** the fused **cuda** kernel with the default SSM state. With `--mamba-ssm-cache-dtype float16` it falls back to **triton** ("requires BF16 or FP32 recurrent state"). SSM fp16 was still faster at XL (312 vs 281) because more streams fit.
- **vLLM CUDA graphs:** with spec decode on FlashInfer, FULL_AND_PIECEWISE falls back to PIECEWISE (FlashInfer supports only UNIFORM_SINGLE_TOKEN_DECODE on SM < 90). PIECEWISE ≈ 1.01× eager on L4 (vLLM #59548).
- **Unsupported on sm_89:** symm-mem ("Device capability 8.9 not supported"); FlashInfer all-reduce at world size 4 / 2.
- **SGLang:** linear-attention backend triton (FlashInfer GDN needs SM90+, CuTe DSL GDN SM100+); attention flashinfer; `mamba_radix_cache_strategy extra_buffer_lazy`, `max_mamba_cache_size 84`, `mem_fraction_static 0.83`, `max_running_requests 21`, `chunked_prefill_size 4096`, schedule lpm.

## 4. Levers for 1500+ (from the 2026-10-29 research; E22 measures the first block)

### Measured in E22 now
| Lever | Engine | Arm(s) | Expected |
|---|---|---|---|
| FULL CUDA graphs with MTP | vLLM | `--attention-backend TRITON_ATTN --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'` or `FULL_DECODE_ONLY` | +3-20 % (the author of PR #60532 measured +36-47 % at batch 1 on SM80) |
| NCCL channels | both | `NCCL_MIN_CTAS` / `NCCL_MAX_CTAS` = 4 or 8, `NCCL_NTHREADS=512` | 0 to +17 % |
| More streams | both | vLLM INT4 24 streams; SGLang W4 24 streams (W4's shorter steps leave KV room) | |
| Bigger prefill chunk | both | vLLM `--max-num-batched-tokens 8192`; SGLang `--chunked-prefill-size 8192` (fewer prefill steps interrupting decode) | |
| Combination | vLLM | INT4 s24 + FULL graph + NCCL 8 | |
| Combination | SGLang | W4 s24 + NCCL 8 | |

### Next, needs code
1. **A 4-rank host-staged all-reduce** (pinned, mapped host memory, two-shot, graph-capturable): est. +14-17 % in bf16, +30-35 % with an INT8 payload behind a logprob gate. Base: SGLang PR #39605 (TP2, open).
2. **Sharded sampling:** per-shard argmax / Gumbel-max instead of the vocab all-gather, ~5-7 ms per step.
3. **vLLM PR #60532:** cherry-pick onto 0.32.0 (Python only, FlashInfer fa2 path for multi-token verify, then FULL graphs with FlashInfer).
4. **vLLM PR #61224:** Ada CUTLASS block-FP8 W8A8 (+42 % on L4 with Qwen3-8B, mostly prefill). Needs an SM89 source build.
5. **SSM bf16 / fp32:** keeps the fused GDN kernel; compare with fp16 at the stream count each allows.
6. **NVFP4 KV** (`--kv-cache-dtype nvfp4`, vLLM PR #46963): 1.75× KV. SM89 untested; captures only single-token decode graphs; a fork measured −13-17 %.
7. **Mamba decode checkpoint** (vLLM PR #50551, open): prefix hit 63.6 → 94.7 % in agent turns; check compatibility with MTP.
8. **Harness:** refill the tail with more attempts / branches so concurrency stays high to the end.
9. **TP2×PP2:** cuts all-reduce volume ~3×. vLLM V2 runner only for MTP + PP (#46994); SGLang DFlash requires pp_size 1.

## 5. The SGLang accuracy gap (must be closed to use its speed)
- **Symptom:** the same Poetiq method scored ~6-7 LB points lower on SGLang INT4 + DFlash2 than on vLLM FP8 (E15P 26.94, E11-06 29.17 vs S1 ~32).
- **Lead: SGLang #36548.** Spec decode at concurrency 8 gave 36.5 % wrong answers vs 0 % without it; it was closed stale with no merged fix.
  - Width 4, fp32 SSM and decode graphs off still left 37/304 wrong.
  - Only a BF16 lm_head target export brought it to 1/304. RedHat INT4 keeps lm_head BF16 (it is in the ignore list).
- **Cheap rule-outs:**
  - **#42917:** a stale `~/.cache/sglang/nv` CUDA compute cache made a W4A16 perplexity gap of 9.98 vs 6.05. Clear it before the launch.
  - **#43514:** DFlash acceptance falls to 0 with silent "!" output under grammar-constrained requests plus multi-chunk prefill.
- **Separating test:** SGLang INT4 without speculation vs vLLM FP8 without speculation on the same ARC tasks.
- **Newer version:** SGLang 0.5.22 (10-16) is out. Check its DFlash / mamba / accuracy fixes.

## 6. Not in scope for these two tracks
- vLLM 0.19: kept only as the LB-proven fallback.
- RedHat INT4 on vLLM 0.19 produces garbage.
- Qwen3.8-Flash-Next on 4× L4: 4-bit weights ~74 GB leave a KV pool of 94-118k tokens, i.e. 1-2 long streams.

**Sources:**
- `research/agent_reports/*.md` (5 research notes with URLs);
- `research/final_report_bn.md`;
- `engine_pack/` (notebooks, `vms_report.md`, `ems_report.md`, server logs);
- `experiments/Experiment 14 .../Kaggle Runs/E14A SLIM R2 run1 (hosen42, 2026-10-23)/bundle/e14/nccl_bench.json`.
