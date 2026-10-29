# Deep research: maximum aggregate decode throughput for Qwen3.8 on Kaggle 4×L4

Search the whole public internet: GitHub (code, issues, PRs, commits, gists), Hugging Face (model cards, discussions,
Spaces), the vLLM / SGLang / TensorRT-LLM / FlashInfer / llama.cpp / ExLlama / LMDeploy repos and forums, Reddit
(r/LocalLLaMA, r/MachineLearning), Kaggle discussions and notebooks (ARC Prize 2026 ARC-AGI-2 and ARC-AGI-3), X/Twitter,
blogs, Discord or Slack summaries, papers and arXiv. Read the primary sources, not summaries. Go deep: follow the links
inside issues and PRs.

## Goal
Find every concrete, verifiable way to raise **sustained aggregate decode throughput (tok/s summed over all concurrent
requests)** for Qwen3.8-27B (and Qwen3.8-Flash-Next) on **4× NVIDIA L4**. The workload: long reasoning requests of
50k-130k tokens, 16-24+ concurrent requests, multi-turn agent turns with tool calls, and no accuracy loss.
Target: 700-1,000 tok/s sustained. We are at about 300.

## Hard environment (cannot change)
- Kaggle notebook, 4× L4: Ada sm_89, 24 GB each, 300 GB/s per GPU, **PCIe only, no NVLink, no P2P guarantee**.
- 48 vCPU, ~197 GB RAM, 92 GB /dev/shm. **Offline at run time**: every wheel and kernel must be prebuilt and attached as
  a dataset.
- Pinned Docker image: Python 3.12, torch 2.10.0+cu128, CUDA 12.8 user space; the driver supports CUDA 13 via compat.
  We can install other stacks into a venv: vLLM 0.32.0 (cu13) already works this way.
- 12 h hard limit for 240 hidden tasks. The engine load must stay under ~10 min.

## The model
- Qwen3.8-27B: `Qwen3_5ForConditionalGeneration`, hybrid **Gated DeltaNet (linear attention, 48 layers) + full
  attention every 4th layer (16 layers)**, head_dim 256, hidden 5120, built-in **MTP head** (1 layer).
- Checkpoints we have:
  - FP8 (block FP8 repack);
  - RedHat compressed-tensors **W4A16 g128** (pack-quantized, actorder static, compressed-tensors 0.18.1, calibrated
    fp8 KV scales; GDN in_proj_qkv / in_proj_z / out_proj are INT4, in_proj_a / b, the MTP head and lm_head are BF16);
  - a BF16 DFlash2 drafter.
- Also research **Qwen3.8-Flash-Next**: the leading teams reportedly use it with Daniel Franzen's "Duck" agent harness.

## What we measured (all TP4, fp8_e4m3 KV, prefix caching ON)
| stack | 8k ctx | 40k ctx | 90k ctx | notes |
|---|---|---|---|---|
| SGLang 0.5.21 INT4 + DFlash2 width 6, 20 streams | 597 | 536 | 364 | fastest, but the same method scores ~6-7 points lower on LB than vLLM FP8 (cause unknown) |
| vLLM 0.32 INT4 MTP-3, 20 streams | 518 | 439 | 332 | coherent; accuracy not certified |
| vLLM 0.32 FP8 MTP-3 + mamba-ssm-cache-dtype fp16, 16 streams | 450 | 414 | 312 | correctness gate PASS |
| vLLM 0.32 FP8 MTP-3, 16 / 20 / 24 streams | 434 / 452 / 472 | 399 / 423 / 424 | 281 / 281 / 256 | |
| vLLM 0.19 FP8 MTP-3, 16 streams (our LB engine) | — | — | — | whole-run average ~256-300 tok/s in real agent runs |

**Things that hurt or failed:**
- PCIe-IPC all-reduce (FlashInfer 0.7): 0.68× on vLLM, 0.4× on SGLang.
- 2×TP2 data parallel: the KV no longer fits; only 2 streams at 40k.
- MTP-2 and MTP-4 are no better than MTP-3.
- RedHat INT4 on vLLM 0.19: garbage output.
- MTP with prefix caching OFF: output drift.
- Custom all-reduce is disabled (PCIe).
- With spec decode on FlashInfer, vLLM falls back to PIECEWISE CUDA graphs.
- Mamba "align" prefix-cache mode never caches the previous assistant reply, so each agent turn re-prefills it (prefix
  hit 55 % on vLLM 0.32, 69-82 % on 0.19).

## Public evidence to explain or reproduce
- Kaggle ARC-AGI-2 discussion 750546, Darren Amadeus Martin (3rd on the LB):
  - "with lots of optimizations by Claude Code ... top speed 900 tok/s";
  - "980 tok/s on 1-minute harness work incl. tool calls and prefill";
  - "average decode 710 tok/s over 6 h, 2 passes of the public eval (tail starved of concurrency)";
  - "1 single stream on L4 is very slow, high speed only with high concurrency";
  - "the secret sauce is how to have high concurrency when 1 problem itself can reason for more than 100k tokens";
  - "it works on RTX PRO 6000 too, much faster than Franzen's setup".
- Anil Thomas, same thread: leading teams use Qwen3.8-Flash-Next + the Duck harness; ~365 tok/s needed for 1 pass at
  64k/task, ~440 for 2 passes.
- Others report 250-500 tok/s. nvbanana scored 76.67 in ~7.7 h.

## Questions: answer each with sources
1. **High concurrency with 100k+ token requests on 4×24 GB.** KV and state capacity tricks:
   - KV quantization below fp8 (int4 / fp4 KV, per-head scales);
   - KV CPU offload or tiered KV (LMCache, vLLM KV offload, SGLang HiCache, Mooncake), at what speed;
   - Mamba / GDN state sizing;
   - `mamba_ssm_cache_dtype`;
   - sliding or partial KV for hybrid models;
   - prefix sharing across agent turns;
   - pausing or preempting idle agent requests during tool calls;
   - eviction policies;
   - max-model-len tricks;
   - context compaction in agent harnesses.
2. **The PCIe TP4 all-reduce bottleneck.**
   - Fastest all-reduce for 4× L4 without NVLink: NCCL tuning (NCCL_ALGO / PROTO / NCHANNELS, LL / LL128), one-shot vs
     two-shot custom all-reduce over PCIe, QuickReduce, sglang `--enable-custom-all-reduce` behavior without P2P.
   - Pipeline parallel vs tensor parallel vs expert or other layouts on PCIe.
   - Disaggregated or hybrid layouts, e.g. TP2×PP2, or DP with a smaller quantized model per GPU pair.
   - Measured numbers from anyone.
3. **Speculative decoding at high batch:**
   - MTP vs EAGLE-3 vs DFlash / DFlash2 vs n-gram / suffix / lookahead for reasoning text;
   - acceptance vs concurrency;
   - full CUDA graphs with spec decode (FlashInfer, FlashAttention-3 on Ada?, Triton attention);
   - draft-tree settings that pay off at 16-32 concurrent requests.
4. **Kernels on Ada sm_89:**
   - best W4A16 / W8A8 / FP8 GEMM kernels at batch 16-64 (Marlin, Machete, CUTLASS, AllSpark, QQQ, fp8 Marlin, ExLlamaV3);
   - Gated DeltaNet / FLA kernels (fused recurrent vs chunked, spec-decode verify kernels);
   - FlashInfer vs FlashAttention vs Triton attention for long-context decode with fp8 KV;
   - torch.compile and CUDA graph settings;
   - known L4-specific regressions.
5. **Which engine and version is fastest today** for the Qwen3.5/3.8 hybrid architecture (vLLM latest, SGLang latest,
   TensorRT-LLM, LMDeploy, llama.cpp / ik_llama, ExLlamaV3), with links to benchmarks on L4 or other Ada GPUs (L40S,
   RTX 4090 / 6000 Ada).
6. **Qwen3.8-Flash-Next:** architecture, size, quantizations, speed on 4×L4 vs 27B, the best serving config, and
   Franzen's Duck harness setup (ARC-AGI-3 milestone 2 solution: repo, configs, flags).
7. **Agent-harness-level throughput:**
   - keeping concurrency high in the tail (task ordering, more parallel attempts, branch splitting);
   - avoiding re-prefill between turns;
   - batching tool calls;
   - prefill/decode separation;
   - chunked prefill size;
   - max-num-batched-tokens.
8. Any Kaggle notebook, GitHub repo or post that reports **≥ 600 tok/s on 4×L4** (any model of 14B or more), with
   its exact config.

## Output format
- A ranked list of levers, biggest expected gain first. For each lever:
  - (a) what it is;
  - (b) the exact flags, commands, versions or PRs;
  - (c) the evidence, with URLs and the numbers from the source;
  - (d) the expected tok/s gain for OUR workload (state your assumptions);
  - (e) accuracy risk;
  - (f) how to verify it in under 30 min on Kaggle;
  - (g) whether it works offline from prebuilt wheels on sm_89.
- A separate "unverified / rumor" section. Never present a guess as a fact.
- A table of every throughput number found for L4 or other Ada GPUs, with links.
- Finally, a concrete plan: the 3 configurations most likely to reach 700+ tok/s sustained for our workload, with the
  exact launch commands.
