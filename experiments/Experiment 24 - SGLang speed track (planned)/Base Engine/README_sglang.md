# SGLang 0.5.21: best configuration for Qwen3.8-27B on Kaggle 4× L4 (2026-10-29)

## Install
SGLang 0.5.21 (cu13) goes into a venv at `/tmp/sglang-venv`, offline, from `foysalemonshanto/sglang-0-5-21-cu13-wheelhouse`. The installer is the E11 / E14 engine cell inside the notebooks. The weights are staged to `/dev/shm` with parallel ranged readers (E14 load path, `payload/q27x_e14_modules_*`).

## The best measured configuration (`dflash_w6_s20`, server args read from `server_logs/sglang_server_ems_dflash_w6_s20.log`)
- **Model:** `--model-path <RedHat INT4 w4a16-g128>`, `--tp-size 4`, `--context-length 155648`.
- **Cache:** `--kv-cache-dtype fp8_e4m3`, `--mamba-ssm-dtype float16`, `--mamba-radix-cache-strategy extra_buffer_lazy`, `--max-mamba-cache-size 84`, page size 1.
- **Speculative decoding:** `--speculative-algorithm DFLASH`, `--speculative-draft-model-path <qwen3-8-27b-dflash2 bf16>`, `--speculative-draft-model-quantization unquant`, `--speculative-num-draft-tokens 6` (width 6; num_steps 1, topk 1).
- **Scheduling:** `--max-running-requests 21` (20 client streams), `--chunked-prefill-size 4096`, `--mem-fraction-static 0.83`, schedule policy lpm, `--min-free-slots-delay 1`.
- **Kernels:** attention flashinfer, linear-attention backend triton, `--disable-flashinfer-autotune`.
- **Other:** `--enable-metrics`, the seed.

The production notebook (`notebooks/Q27-XHIGH-E15B_EVAL120_50K45K_S16_R3K2_DUCK_SGLANG.ipynb`, sha `c058d58b`) runs the same engine at 16 streams. Its flag builder is `e11_server_flags(cfg)`, and its relaunch, OOM and watchdog logic come from E14S / E15B.

## The engine matrix (`results/ems_report.md`, EMS SAFE `8f01847c`, run on hosen42 2026-10-26)

| Arm | Streams | 8k tok/s | 40k tok/s | Accept length 8k / 40k | Step ms 8k / 40k | Gate |
|---|---|---|---|---|---|---|
| ref DFlash W6 | 16 | 583 | 521 | 4.21 / 4.11 | 115.7 / 127.3 | PASS |
| **DFlash W6** | **20** | **597** | **536** | 4.17 / 4.21 | 140.3 / 157.1 | PASS |
| DFlash W4 | 16 | 592 | 526 | 3.30 / 3.31 | 89.2 / 100.9 | PASS |
| DFlash W8 | 16 | 573 | 477 | 5.00 / 4.47 | 138.8 / 149.9 | PASS |
| MTP (3, 1, 4) | 16 | 478 | 402 | 2.88 / 2.94 | 96.4 / 117.3 | PASS |
| MTP (2, 1, 3), 20 streams | 20 | 542 | 413 | 2.48 / 2.45 | 92.0 / 118.2 | PASS |
| target only (no speculation) | 16 | 405 | 354 | — | **39.5 / 45.1** | reference |
| DFlash W6 + PCIe-IPC all-reduce | 16 | 230 | 202 | | 299.6 / 309.9 | 0.39× |
| DFlash W6, 24 streams | 24 | | | | | CAPPED by KV |

The XL band (~90k prompts) reached 364 tok/s with W6 at 20 streams.

## Open questions for this track
1. **The LB accuracy gap (most important).** SGLang issue #36548 reports wrong answers under DFlash at concurrency 8: 36.5 % against 0 % without speculation. Only a BF16 lm_head target export fixed it; width 4, fp32 SSM and decode graphs off did not.
   - Check: RedHat INT4 keeps lm_head BF16 (it is in the ignore list).
   - Test: SGLang INT4 without speculation against vLLM FP8 without speculation, on the same tasks.
   - Also check #42917 (a stale `~/.cache/sglang/nv` compute cache gave a W4A16 perplexity gap) and #43514 (DFlash acceptance falls to 0, silent "!" output).
2. **Speculation costs most of the step.** Target-only steps are 39-45 ms; DFlash W6 steps are 116-157 ms. DFlash W4 gives the same tok/s with shorter steps.
3. **All-reduce:** sweep NCCL channels (`NCCL_MIN_CTAS` / `NCCL_MAX_CTAS` 4-8, `NCCL_NTHREADS=512`); consider a host-staged all-reduce (SGLang PR #39605, TP2 only, open).
4. **Newer SGLang:** 0.5.22 was released on 10-16. Check its DFlash, mamba and accuracy fixes.
