# Speculative decoding for Qwen3.8-27B (hybrid GDN + full attention, built-in MTP) on 4x L4 TP4 at 6-16+ streams, ~60k-token thinking outputs

Labels used: **[vendor]** = reported by the drafter or engine authors; **[indep]** = independent third-party measurement; **[user-report]** = a GitHub issue or community post; **[academic]** = peer-reviewed or arXiv paper; **[snippet]** = taken from a search-result summary, page not fully fetched. All of my own arithmetic is in the "Inferences" subsections and is labeled as such.

## Q1. Acceptance lengths and speedups per method on Qwen3.x, especially in thinking mode

### Takeaway
On short outputs (up to 4k tokens), DFlash/DFlash2 have the highest reported acceptance length (AL) on Qwen3.x. For Qwen3.8-27B the vendor reports DFlash2 AL of about 4.1-5.5 against about 3.7-5.0 for the built-in 7-token MTP. Native MTP beats EAGLE-3 and n-gram on Qwen3.5/3.6/Next. In thinking mode, AL drops for every method. MTP falls to about 2.5 of 4 on thinking output for a Qwen3.8 hybrid model. DFlash, EAGLE-3 and PARD collapse toward an AL of 1-2 after roughly 10-25k output tokens. No published source measures AL across a 60k-token thinking trace on Qwen3.8-27B.

### Cited Findings
**DFlash2 vs MTP on Qwen3.8-27B (the exact target)**
- Qwen3.8-27B-DFlash2 drafter: about 2B params in BF16, block size 8 (7 draft tokens per verify step), two-tap dynamic convolutions plus a path selector. The card claims lossless decoding (greedy output matches the target, sampling preserves its distribution). [vendor] — [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2); mirrored at [z-lab/Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
- Eval setup: SGLang on 1x H200 with FA3, temperature 1.0, top-p 0.95, top-k 20, **xhigh reasoning effort, max_new_tokens 4096**. Baselines are the built-in **seven-token MTP** and RadixArk DSpark. [vendor] — [z-lab card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
- Acceptance length, MTP / DSpark / DFlash2: GSM8K 5.02 / 4.36 / 5.46; MATH-500 4.72 / 3.92 / 5.28; HumanEval 3.91 / 3.30 / 4.39; MBPP 3.99 / 3.51 / 4.79; MT-Bench 3.74 / 3.01 / 4.10. [vendor] — [incoai card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
- The Inco blog Table 4 (Qwen3.8-27B, default sampling, block 8) gives mean AL of MTP 4.28, DSpark 3.62, DFlash2 4.80. The blog claims DFlash2 is "close to 3x" faster than autoregressive decoding and 2.7-3.4x at batch 1 in SGLang. [vendor] — [inco.ai/blog/dflash2](https://inco.ai/blog/dflash2/)
- On Qwen3.5-4B with thinking on and temperature 1.0, mean AL is MTP 4.54, DFlash 4.92, DSpark 5.49, DFlash2 5.97. [vendor] — [inco.ai/blog/dflash2](https://inco.ai/blog/dflash2/)
- Independent single-stream test on 1x RTX 3090, llama.cpp, Qwen3.8-27B UD-Q4_K_XL, greedy, 400-token window. Baseline 41.55 tok/s. MTP n-max 2 reached 66.39 tok/s (+59.8%), MTP n-max 3 +52.3%, DFlash2 n-max 4 +51.9%, MTP n-max 5 +32.1%, DFlash2 n-max 7 +22.6%. **The best MTP arm beat the best DFlash2 arm** on this bandwidth-limited consumer GPU. DFlash2 n-max 7 was negative on 3 of 5 prompt classes. [indep] — [HackMD thc1006](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090). This contradicts the vendor's batch-1 ordering of 3.43x (DFlash2) vs 2.59x (MTP) on H200 SGLang.
- Jetson AGX Thor, vLLM, Qwen3.8-27B: MTP depth 3 raised decode from about 13 to about 35 tok/s (about 2.7x). [vendor: NVIDIA] — [Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/speculative-decoding/)
- DGX Spark, Qwen3.8-27B, independent:

  | Setup | c=1 decode | c=10 aggregate |
  |---|---|---|
  | vLLM NVFP4 | 11.5 tok/s | 84.3 tok/s |
  | vLLM NVFP4 + MTP-3 | 22.0 tok/s | 105.8 tok/s |
  | SGLang FP8 | 7.7 tok/s | 54.3 tok/s |
  | SGLang FP8 + NEXTN | 13.4 tok/s | 71.0 tok/s |

  [indep] — [kubesimplify](https://blog.kubesimplify.com/qwen3-8-27b-on-dgx-spark)

**Thinking mode lowers MTP acceptance**
- SGLang Qwen3.8-Flash-Next cookbook (same model family, MTP with 4 draft tokens): "accept length 3.5-3.7 of 4 on non-thinking output, **about 2.5 on thinking output**" (DGX Spark). On RTX PRO 6000 it reports about 3.3 of 4 on GSM8K and random prompts and 2.9 on ShareGPT. [vendor: SGLang] — [SGLang Qwen3.8-Flash-Next cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
- Qwen3.6-27B with MTP in llama.cpp on an M2 Max: enabling thinking on coding tasks dropped Q8_0 draft acceptance from 87% to 73%. Speedup stayed at +94%. Acceptance by task: code 79-89%, factual 62-70%, analysis 48-56%, creative 39-48%. [indep] — [froggeric/Qwen3.6-27B-MTP-GGUF](https://huggingface.co/froggeric/Qwen3.6-27B-MTP-GGUF)
- DFlash paper, Qwen3-4B/8B with thinking enabled (speedup / τ): at T=0, MATH-500 4.59x/5.74 (4B) and 4.64x/5.82 (8B); at T=1, MATH-500 3.93x/4.89 and 4.03x/5.06, AIME25 3.64x/4.68 and 3.70x/4.69. These ran with short max lengths. [academic] — [DFlash arXiv HTML](https://arxiv.org/html/2602.06036v2)

**Long outputs: acceptance decays as generation grows (critical for 60k thinking)**
- Test-Time Speculation paper, Qwen3-8B, MATH-500, across DFlash, EAGLE-3 and PARD: "the average acceptance length is 3.7 for the first 10K output tokens, but sharply decreases to 1.5 for the last 10K output tokens". EAGLE-3 "reduces to 1.1 for all the tasks after generating ~20K tokens". On Qwen3.6-35B with DFlash and AIME-2025, AL falls "from 15 to 1.7 ... within a span of 25K tokens". The authors' explanation is that drafters are trained offline on short sequences. MTP was not tested. Their fix, online distillation from the verify step, improves AL by up to 72% (41% on average). [academic] — [arXiv 2605.09329](https://arxiv.org/html/2605.09329)
- DFlash paper Table 4: the Qwen3.5-27B base drafter was trained on 4K context. Its τ degrades with context: hotpotqa 4.91 (1K), 4.46 (8K), 3.61 (16K); gov_report 4.53 (1K), 3.32 (8K), 2.67 (16K), 2.09 (32K). A long-context fine-tuned drafter (1.6K LongAlign samples) holds up: gov_report 3.56 at 32K and hotpotqa 6.05 at 16K. [academic] — [DFlash arXiv HTML](https://arxiv.org/html/2602.06036v2)
- Conflict: the current z-lab Qwen3.5-27B-DFlash card says the drafter was trained at **40k** sequence length, with a sliding-window long-context variant. That differs from the paper's 4K base drafter, so the card has probably been updated. [vendor] — [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash). The Qwen3.8-27B DFlash2 cards do not state a training context length.
- The "attention drift" paper finds drift in both EAGLE3 drafters and **MTP heads**: drafter hidden-state magnitude grows with chain depth. Its normalization fix gives 1.18x AL on long-context tasks. [academic] — [arXiv 2605.09992](https://arxiv.org/abs/2605.09992)

**EAGLE-3 on Qwen3.x**
- Qwen3.6-27B EAGLE-3 (community, SpecForge/Model-Optimizer): accept length 2.2 on stock Qwen3.6-27B. [snippet] — [Ex0bit/Qwen3.6-27B-PRISM-EAGLE3](https://huggingface.co/Ex0bit/Qwen3.6-27B-PRISM-EAGLE3)
- On Qwen3.5-397B, the EAGLE-3 drafter is below MTP AL on every benchmark (e.g., MT-Bench 3.03 vs 3.28, CEval 1.77 vs 2.34), though it wins throughput on some tasks. [snippet] — [AQ-MedAI/Qwen3.5-397B-A17B-eagle3](https://huggingface.co/AQ-MedAI/Qwen3.5-397B-A17B-eagle3)
- Qwen3.6-35B-A3B in SGLang on B200: native MTP AL 3.715 vs EAGLE3 1.251. [indep] — [shreyansh26 RESULTS.md](https://github.com/shreyansh26/qwen-spec-decode-benchmarking/blob/main/RESULTS.md)
- DFlash paper Table 1 (Qwen3-8B, thinking off, T=0): EAGLE-3 (16) averages 1.76x/τ 2.96 vs DFlash (16) 4.86x/τ 6.49. [academic] — [DFlash arXiv HTML](https://arxiv.org/html/2602.06036v2)

**MTP on Qwen3-Next (SPEED-Bench, BS=32, draft length 3, T=0)**
- Mean AL is 2.81: coding 3.34, math 3.13, reasoning 2.89, writing 2.46, roleplay 2.09. Mean speedup is 1.20x (1.18x at T=1). "Qwen3-Next's pre-trained MTP head maintains higher ALs than the post-trained setups like EAGLE3." [academic, NVIDIA] — [SPEED-Bench HTML](https://arxiv.org/html/2604.09557v2)

**n-gram / suffix decoding**
- SPEED-Bench n-gram mean AL is 1.41 (0.88x, a slowdown) on Llama 3.3 70B and 1.31 (**0.29x**) on GPT-OSS 120B. [academic] — [SPEED-Bench](https://arxiv.org/html/2604.09557v2)
- Snowflake SuffixDecoding: 1.96-3.12x end-to-end on BlazeEdit, but only 1.0-1.28x on SpecBench. It reports up to 5.3x on agentic workloads (AgenticSQL, SWE-Bench) and no reasoning-workload numbers. [vendor] — [Snowflake blog](https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/)
- An MLSys 2026 talk reports up to 2.13x over vLLM n-gram for Qwen3 1.7B/8B/14B reasoning serving and RL rollouts on H100. [snippet] — [MLSys 2026 slides](https://mlsys.org/media/mlsys-2026/Slides/3733_usZ8H1U.pdf)

### Inferences
- The vendor DFlash2-vs-MTP gap of about +0.4-0.8 AL was measured with outputs capped at 4096 tokens. For 60k-token thinking, the long-output decay results suggest any externally trained drafter (DFlash/DFlash2/EAGLE-3) may lose most of its edge after about 10-25k tokens unless it was trained on long sequences. MTP is trained jointly with the target and reads its final hidden state, so it is plausibly more robust. This is not measured.
- On thinking output, a realistic MTP AL at 4 draft tokens is about 2.5 (Flash-Next evidence). Plan around 2-2.5, not the 3.7-5 benchmark figures.

### Gaps
- No source measures AL per output-position window (0-10k, 10-30k, 30-60k) for MTP vs DFlash2 on Qwen3.8-27B in xhigh thinking.
- No published DFlash2 training context length for Qwen3.8-27B.
- No ARC-like (grid-transcription) workload data for suffix/n-gram decoding. Grid copying could raise its acceptance, but no evidence was found.

## Q2. How do gains change with concurrency (batch 1-64)?

### Takeaway
Every method's speedup shrinks with batch size. On H200 at concurrency 32, Qwen3.8-27B MTP is at or below 1.0x (0.77-1.04x), while DFlash2 keeps 1.01-1.45x (vendor numbers, 4k outputs). Wide EAGLE trees go negative by batch 16-32. Those curves come from compute-rich GPUs with short contexts. On L4 with long contexts, decoding stays memory-bound to a higher batch (MagicDec regime), so gains should shrink less, but there are no L4 measurements.

### Cited Findings
- **Qwen3.8-27B, H200, SGLang**, speedup vs autoregressive:

  | Concurrency | MTP (7 tokens) | DFlash2 |
  |---|---|---|
  | 1 | 1.96-2.59x | 2.67-3.43x |
  | 8 | 1.74-2.19x | 2.27-2.85x |
  | 32 | **0.77-1.04x** | **1.01-1.45x** |

  Absolute GSM8K numbers: c=32 autoregressive 1,329.8 tok/s, MTP 1,381.1, DFlash2 1,922.5. MATH-500 at c=32: autoregressive 1,505.8, MTP 1,415.6 (0.94x), DFlash2 1,951.8 (1.30x). [vendor] — [incoai card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
- **Qwen3.5-27B DFlash (v1), B200, TP1**: c=1 block 8 gives 3.48-4.76x and block 16 gives 3.77-6.20x; c=32 block 8 gives 2.14-3.01x. The card says it beats MTP at every matched setting. [vendor] — [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash)
- DFlash paper Table 3 (B200 SGLang, concurrency 1/4/8/16/32): Qwen3-8B Math500 5.1x / 4.5x / 4.5x / 3.9x / 2.8x; HumanEval 4.2x / 3.6x / 3.6x / 3.0x / 2.4x. Qwen3-Coder-30B-A3B is flat at about 3.1-3.5x. [academic] — [DFlash arXiv HTML](https://arxiv.org/html/2602.06036v2)
- DFlash paper Table 5 (LLaMA-3.1-8B, B200, concurrency 1/4/8/16/32): EAGLE-3 with 10 draft tokens goes 1.6x → 1.0x (GSM8K). **EAGLE-3 with a 60-token tree goes 1.9x → 0.6x.** DFlash (10) goes 2.4x → 1.6x. [academic] — [DFlash arXiv HTML](https://arxiv.org/html/2602.06036v2)
- EAGLE-3 paper: 1.38x throughput at batch 64 in SGLang (H100). [academic] — [arXiv 2503.01840](https://arxiv.org/abs/2503.01840)
- Qwen3.8-Flash-Next on RTX PRO 6000 (1024 in / 256 out): with MTP, 148 tok/s at 1 request and 613 tok/s at 16. Without MTP, 83 tok/s at 1, 524 tok/s at 16 (**+17% at 16**), and 861 tok/s at 64. MTP could not reach 64 because the mamba cache caps concurrency (see Q5). ShareGPT: 785 tok/s at 16-way with MTP vs 1,258 tok/s at 64-way without. [vendor: SGLang] — [Flash-Next cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
- Single DGX Spark: MTP 71.7 tok/s at 8 concurrent vs no-MTP 83 tok/s at 24 concurrent. MTP admitted only 8 requests (40 slots) against 24 without. [vendor: SGLang] — [Flash-Next cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
- Qwen3.5-397B SGLang cookbook (H200, TP8, NEXTN 3/1/4): accept length 3.12 at c=1 (222.84 tok/s) and 3.31 at c=100 (1,767 tok/s). No baseline without MTP is given. [vendor] — [SGLang Qwen3.5 cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.5)
- vLLM recipe: MTP-1 is for latency-sensitive, low-concurrency work. "Under high load, it lowers text throughput because speculative tokens consume KV cache capacity and shrink the effective batch size." Throughput configs do not use MTP. [vendor] — [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html)
- B200, c=8, 8K input / 128 output, Qwen3.6-35B-A3B. vLLM: MTP +3.5% (within launch noise of 1,024-1,140 tok/s). SGLang FP8: MTP +23.1%, EAGLE3 −3.6%, DFlash V1 block 4 −53.3%, DFlash V2 block 4 −46.3% (AL 1.2-1.4). [indep] — [shreyansh26 RESULTS.md](https://github.com/shreyansh26/qwen-spec-decode-benchmarking/blob/main/RESULTS.md)
- Dre Dyson guide (Qwen3.6-27B, llama.cpp): at concurrency 4 the non-MTP setup pulled ahead. [snippet] — [dredyson.com](https://dredyson.com/how-i-mastered-mtp-speculative-decoding-with-llama-cpp-on-qwen3-6-27b-the-complete-advanced-configuration-guide-that-pros-dont-want-you-to-know/)
- MagicDec: at long sequence lengths the bottleneck shifts from weights to KV loading, so speculative decoding still helps at large batch. Up to 2.51x for LLaMA-3.1-8B at batch 32-256. [academic] — [MagicDec arXiv 2408.11049](https://arxiv.org/pdf/2408.11049)
- SuffixDecoding: about 10% overhead at concurrency 64, and it beats both n-gram configs at all concurrency levels. [vendor] — [Snowflake blog](https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/)
- SPEED-Bench: "At lower batch sizes, where the system is memory-bound, longer drafts are preferred"; at higher batch sizes "DL=1 is preferred". Synthetic inputs overestimate throughput by about 23% on average. [academic] — [SPEED-Bench](https://arxiv.org/html/2604.09557v2)

**Long context (closest hardware analog: 4x RTX 4090, Ada, TP4)**
- vLLM 0.28.0, Qwen3.5-family hybrid + DFlash2, single stream on 4x RTX 4090 TP4: at **185k context, no spec runs at about 71 tok/s while DFlash DT4 runs at 16.0 tok/s and DT2 at 12.1** (a 4.4x slowdown). At short context, DT4 is about 190 and DT8 about 218 tok/s. The draft-side cost grows with context. Open, no fix. [user-report] — [vLLM #54691](https://github.com/vllm-project/vllm/issues/54691)
- vLLM PR #61261 proposes an opt-in `draft_context_window` to bound DFlash/DSpark draft attention. [user-report] — [vLLM PR #61261](https://github.com/vllm-project/vllm/pull/61261)
- GLM-5.3-Flash DFlash2 k=7 on 2x GB10: the original report saw acceptance fall from 70-73% (short) to 16-32% at about 172k. A follow-up ladder found no context decay (0.31-0.36 accept rate from 4K to 170K) but a slower step at 170K. The two sets of data conflict. [user-report] — [MiaAI issue #73](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/issues/73)

### Inferences
- **L4 roofline (my arithmetic).** L4 has 300 GB/s and about 121 dense FP16 TFLOPS ([NVIDIA L4](https://www.nvidia.com/en-au/data-center/l4); the [datasheet](https://bxin.cisco.com/c/dam/en/us/products/collateral/servers-unified-computing/ucs-c-series-rack-servers/nvidia-l4-gpu.pdf) says the 242 TFLOPS figure assumes sparsity).
  - Model dimensions from the [Qwen3.8-27B card](https://huggingface.co/Qwen/Qwen3.8-27B): 16 full-attention layers, 4 KV heads, head dimension 256. That gives 64 KiB/token of KV in BF16 (32 KiB in FP8), or 16 KiB (8 KiB) per GPU at TP4.
  - At 16 streams and about 35k mean context with FP8 KV, one decode step per GPU reads about 4.3 GiB of KV (about 15 ms), about 4-5 GB of INT4 weights plus BF16 embed/lm_head shards (about 15 ms), and reads and writes about 1.2 GB of GDN state (about 4 ms). That is about 35 ms per step, an upper bound of about 450 tok/s without speculation.
  - Verifying 16 × 4 = 64 tokens needs about 0.86 TFLOP per GPU of GEMM, roughly 7-14 ms. That is below the memory time, so the verify step should stay memory-bound at 16 streams. This is the MagicDec regime: speculation should still pay at 16+ streams on L4, unlike the H200 c=32 numbers.
  - Overheads that could eat the gain: intermediate SSM-state writes per draft position (about 16 × 4 × 37.7 MB ≈ 2.4 GB per step in fp32, about 8 ms; half that in fp16), PCIe all-reduce latency for each sequential draft step, and draft-model KV reads at long context (the vLLM #54691 effect).
- Rough expected gain (my estimate, unvalidated): MTP at AL 2-2.5 → about 1.4-1.7x aggregate at 16 streams on 4x L4, provided the mamba-slot cap does not cut concurrency. DFlash2 could be higher at short context but carries the long-context draft-cost risk.

### Gaps
- No measured spec-decode concurrency curve on L4, or on any Ada TP4 setup with 6-16 streams and 30-70k contexts, for any method.
- INT4 Marlin/W4A16 GEMM efficiency on sm_89 at M = 64-128 (verify width × streams) is not measured.

## Q3. Optimal steps / topk / draft-token settings at high concurrency

### Takeaway
Use topk=1 chains, never trees, at high batch. For MTP the evidence clusters at 2-3 steps / 3-4 draft tokens: SGLang's default is steps 3, topk 1, draft tokens 4, and independent sweeps peak at 3-5 on B200 and 2-3 on consumer GPUs. For DFlash, the vendor recommends block 8 over 16 for concurrent serving, and block 4-5 for quantized setups. Long contexts favor shorter drafts.

### Cited Findings
- SGLang Qwen3.5/Qwen3-Next cookbook: `--speculative-algorithm EAGLE` (or NEXTN) `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`. Tune with `bench_speculative.py`. [vendor] — [Qwen3.5 cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.5); [Qwen3-Next cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3-Next)
- vLLM: `{"method":"mtp","num_speculative_tokens":N}`. Recipes use N=1 (latency), 2 (Qwen3.6), 3 (MI300X) and 5 (DGX Spark), and suggest tuning within 1-5. [vendor] — [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html); [vLLM Qwen3.6-27B recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-27B)
- RedHat INT4 Qwen3.8-27B card: MTP `num_speculative_tokens` 3, or DSpark with 7. [vendor] — [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)
- Sweep at c=8 on B200 (32 requests). SGLang MTP: steps1 649.9, steps2 703.1, steps3 736.2, steps4 725.0, steps5 740.5, steps7 706.6 tok/s. vLLM MTP: tokens1 937.7, tokens2 880.3, tokens3 1,024.0, tokens5 1,033.5, tokens7 1,032.3. EAGLE3 was best at steps1_k1_d2. [indep] — [shreyansh26](https://github.com/shreyansh26/qwen-spec-decode-benchmarking/blob/main/RESULTS.md)
- RTX 3090, batch 1, Qwen3.8-27B: MTP n-max 2 > 3 > 5 (+59.8% / +52.3% / +32.1%). [indep] — [HackMD](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090)
- M2 Max: "N=3 is optimal" for all quants except F16, where N=4 is slightly better. [indep] — [froggeric GGUF](https://huggingface.co/froggeric/Qwen3.6-27B-MTP-GGUF)
- Jetson starting points: MTP 3, DFlash 15 (block-16 checkpoint), DSpark 4. Lower DFlash depth can win when acceptance is low. [vendor] — [Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/speculative-decoding/)
- Qwen3.5-27B DFlash: block 8 recommended "for higher-concurrency serving", block 16 for concurrency 1. Mean AL at c=1 for block 4 / 8 / 16 is 3.53 / 5.64 / 7.51 on GSM8K and 3.22 / 4.63 / 5.62 on MT-Bench. [vendor] — [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash)
- DFlash2 for Qwen3.8-27B is trained at block 8. Flags: SGLang `--speculative-num-draft-tokens 8`, vLLM `num_speculative_tokens: 7`, llama.cpp `--spec-draft-n-max 7`. For quantized targets the z-lab repo recommends `block_size <= 5` (MLX quantized-matmul cost). [vendor] — [inco blog](https://inco.ai/blog/dflash2/); [z-lab/dflash GitHub](https://github.com/z-lab/dflash)
- DFlash paper Table 8: a drafter trained at b16 and run at b8 loses τ (Math500 6.33 → 5.09). Run the block size the drafter was trained with. [academic] — [DFlash arXiv HTML](https://arxiv.org/html/2602.06036v2)
- DFlash2 on GLM (GB10): k=4 gave 47% acceptance and 28.7 tok/s, k=7 gave 27% and 20.6 tok/s. [user-report] — [MiaAI #73](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks/issues/73)
- SGLang's adaptive step size (`--speculative-adaptive`) **crashes on hybrid GDN Qwen3.5** (RTX 4090, sm89): the MTP draft mutates the shared config. Fix PR #23331; the issue was closed by an inactivity bot, so the fix is unconfirmed. [user-report] — [SGLang #23330](https://github.com/sgl-project/sglang/issues/23330)

### Inferences
- For 6-16 streams on L4: MTP steps 2-3, topk 1, draft tokens 3-4 is the evidence-backed band. Fewer draft tokens also mean fewer mamba slots per request (Q5), which matters on 24 GB.
- If DFlash2 is tried, it must run at its trained block of 8, which means 8 verify positions and 1+8 mamba slots per request, a heavier memory footprint than MTP-4.

### Gaps
- No setting sweep for MTP on Qwen3.8-27B at 16 streams with long thinking on any GPU.

## Q4. Engine support for hybrid (mamba/GDN) models with speculative decoding

### Takeaway
Both engines support MTP on Qwen3.5/3.6/3.8-class hybrids. SGLang handles mamba-state rollback by giving each draft token its own state slot, and it needs `extra_buffer` (FLA kernels, NVIDIA-only) for overlap scheduling and radix caching. Correctness is the main risk. There is an open SGLang bug where DFlash2 corrupts state on Qwen3.8-27B under concurrency, and vLLM has repeated MTP + prefix-cache corruption on hybrids. DFlash2 in vLLM still requires an unmerged PR. The FP8 draft KV path crashes on sm_89 in vLLM.

### Cited Findings
**SGLang**
- Mamba radix strategies: `no_buffer` (V1) has no overlap scheduler and uses less memory. `extra_buffer` (V2) enables overlap scheduling and branching-point caching, requires `--page-size 64` (page size must divide or be a multiple of FLA_CHUNK_SIZE 64), and needs FLA kernels (NVIDIA only). [vendor] — [Qwen3.5 cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.5)
- The flag is being renamed `--mamba-scheduler-strategy` → `--mamba-radix-cache-strategy` [snippet] — [SGLang #27418](https://github.com/sgl-project/sglang/issues/27418). "auto" now resolves to extra_buffer on CUDA with radix cache enabled. [snippet] — [SGLang PR #27220](https://github.com/sgl-project/sglang/pull/27220)
- A changelog entry adds spec decoding with `extra_buffer_lazy` and cuts spec-v2 host overhead in hybrid-linear MTP decode. [snippet] — [SGLang releases](https://github.com/sgl-project/sglang/releases)
- Rollback mechanism: SSM states update in place, so each draft token gets its own cache slot and the last accepted slot is promoted. [snippet] — [PyTorch blog](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)
- Spec V2 overlap: Qwen3-8B on B200 at c=32 went from about 11.4k to about 15.3k tok/s (+33%) with `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`. [vendor] — [LMSYS blog](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- Verified Spec V2 attention backends: TRTLLM MLA/MHA, FA3, Ascend, Triton. FlashInfer works, but its plan stream adds a sync point that limits overlap. Ada (sm_89) defaults to FlashInfer, else Triton. [snippet] — [SGLang attention backend docs](https://docs.sglang.io/docs/advanced_features/attention_backend)
- The DFlash recipes use `--speculative-draft-attention-backend fa4` and `--attention-backend trtllm_mha`, both Blackwell-only. A community fork runs DFLASH on sm89 through FlashInfer. [vendor] — [z-lab/Qwen3.5-27B-DFlash](https://huggingface.co/z-lab/Qwen3.5-27B-DFlash); [snippet] — [agens-sglang fork](https://github.com/BlockWayz/agens-sglang)
- **Open correctness bug, Qwen3.8-27B + DFlash2 + concurrency.** The context gets mixed across requests and benchmark accuracy drops to about 50% vs about 90% expected. The reporter could not reproduce it on the 0.5.18 release with MTP/EAGLE. The first divergence is in the SSM transition at layer 0, possibly a DECODE vs TARGET_VERIFY kernel mismatch. A dense BF16 lm_head cut wrong answers from 111/304 to 1/304 but did not remove the drift. No fix PR. [user-report] — [SGLang #36548](https://github.com/sgl-project/sglang/issues/36548)
- The mamba cache silently caps `max_running_requests`. With DFLASH, each request needs 1 + num_draft_tokens state slots. The cap is logged at info level only, and /get_server_info still reports the requested value. A fix to `--max-mamba-cache-size 40 --mamba-ssm-dtype bfloat16` gave +75% at c=12. Open. [user-report] — [SGLang #36889](https://github.com/sgl-project/sglang/issues/36889); [SGLang #38846](https://github.com/sgl-project/sglang/issues/38846) [snippet]
- Hybrid models + speculative decoding + `--disable-radix-cache` can fail to start (default-48 treated as a user-set limit). [snippet] — [SGLang #44355](https://github.com/sgl-project/sglang/issues/44355); [PR #44372](https://github.com/sgl-project/sglang/pull/44372)
- A spec + structured-output fix addressed resuming from wrong mamba states. [snippet] — [SGLang releases](https://github.com/sgl-project/sglang/releases)

**vLLM (current series 0.28-0.31 as of the issues found)**
- MTP method name is `mtp`, minimum vLLM 0.17.0 for Qwen3.6-27B. Mamba "align" prefix caching is experimental. A `num_cache_lines >= batch` assertion fires when the CUDA graph capture size exceeds the mamba cache size; reduce `--max-cudagraph-capture-size`. [vendor] — [vLLM Qwen3.6-27B recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-27B); [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html)
- MTP + prefix caching costs about 20% accuracy on Qwen3.6-35B-A3B. A follow-up says it still corrupts output in v0.28.0 on hybrid GDN (repeated punctuation, empty responses). [user-report] — [vLLM #43559](https://github.com/vllm-project/vllm/issues/43559); [vLLM #53912](https://github.com/vllm-project/vllm/issues/53912)
- MTP and DFlash silently drop prefix-cache hits to 0 for hybrid GDN on nightly. It worked on 0.24.0. [snippet] — [vLLM #54360](https://github.com/vllm-project/vllm/issues/54360)
- DFlash2 for Qwen3.8-27B requires `vllm @ git+...@refs/pull/52816/head`, an unmerged PR. [vendor] — [z-lab card](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2)
- PR #55601 (DFlash2 prefix-hit mamba state index) was reportedly merged 2026-10-07, after 0.31.0. [snippet, third-party tracker] — [HyperQwen #344](https://github.com/syv-ai/HyperQwen/issues/344)
- DFlash + pipeline parallelism is unsupported for Qwen3.5 hybrids (v0.30.0). [snippet] — [vLLM #60765](https://github.com/vllm-project/vllm/issues/60765)
- **FP8 draft KV crashes on SM89** (4x RTX 4090, TP4, DFlash, vLLM 0.28.0). There is first a `Unrecognized dtype: auto` error, then a native worker crash. No workaround. [user-report] — [vLLM #54690](https://github.com/vllm-project/vllm/issues/54690)
- Suffix decoding: `"enable_suffix_decoding": true` in Arctic Inference/vLLM. It can be combined with a model drafter (it uses suffix candidates if their score beats the draft length, else the draft model). LSTM+Suffix reaches 209 tok/s on a mixed workload vs 154-155 for either alone (Llama 3.1 70B). MTP + suffix is not documented. No hybrid-model data. [vendor] — [Snowflake Arctic blog](https://www.snowflake.com/en/engineering-blog/fast-speculative-decoding-vllm-arctic/)

**"Lossless" in practice**
- On llama.cpp with Qwen3.8-27B, MTP and DFlash2 outputs diverged from serial greedy decoding in 92-100% of requests at 1,600 tokens. The divergence was deterministic. Possible causes are floating-point path dependence, quantized kernels, batch shapes or KV handling. Semantic quality was not measured. [indep] — [HackMD](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090)

### Inferences
- SGLang MTP (EAGLE/NEXTN, topk 1) on extra_buffer is the most-trodden path for this architecture. DFlash2 on SGLang has an unresolved concurrency-correctness bug on this exact model. Any run should gate on an accuracy A/B (spec vs no-spec on the same tasks) rather than trust "lossless".
- On sm_89, the Blackwell-tuned DFlash recipes (fa4, trtllm_mha) cannot be used as-is. Expect FlashInfer or Triton backends, which have the weaker Spec V2 overlap.

### Gaps
- No upstream SGLang doc confirms DFLASH draft attention works on sm_89.
- No official statement on how vLLM stores per-draft-token GDN states (memory multiplier).

## Q5. Memory cost of each drafter and its effect on KV capacity on 24 GB GPUs

### Takeaway
The weights are not the main cost: about 0.8 GB for MTP (estimate) and 3.8 GB for DFlash2 in BF16. The bigger cost on hybrid models is the mamba-state multiplier: SGLang reserves 1 + draft tokens SSM slots per request, which can silently cut max running requests by 3-4x. Draft KV and the draft tokens' KV also reduce the effective batch.

### Cited Findings
- DFlash2 drafter file sizes: BF16 3.8 GB (2B params), Q8_0 2.0 GB, Q4_K_M 1.1 GB. The GGUF quants are for llama.cpp only. [vendor] — [incoai GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)
- MTP layers ship inside the main checkpoint, so no separate draft model is needed [vendor] — [Qwen3-Next cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3-Next). MTP "can need almost no extra model memory" [vendor] — [Jetson AI Lab](https://www.jetson-ai-lab.com/tutorials/speculative-decoding/)
- A community count says Qwen3.6-27B has 1 MTP layer (15 tensors) that shares embed/lm_head with the backbone. Estimated about 0.4B params, about 0.8 GB BF16; this is an estimate, not measured. [snippet] — [originalGeek MTP head-only](https://huggingface.co/originalGeek/Qwen3.6-27B-unsloth-MTP-Q8_0-HEAD-ONLY)
- Mamba state slots per request with MTP (4 draft tokens): **extra_buffer 5, extra_buffer_lazy 4, radix disabled 1, lazy + `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` 3**. `--mamba-ssm-dtype bfloat16` halves the slot size (0.109 → 0.055 GiB per slot for Flash-Next). Set `--max-mamba-cache-size` = requests × slots. On RTX PRO 6000 96 GB the default extra_buffer allows **about 3 requests with MTP vs 12 without**; tuning raised that to 16 with MTP and 64 without. Ceilings were about 20 with MTP and about 70 without. [vendor: SGLang] — [Flash-Next cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
- DFLASH needs 1 + speculative_num_draft_tokens slots per request. One report saw concurrency capped to 2, where fixing it raised throughput 48.8 → 85.4 tok/s at c=12. bf16 SSM state cost about 6% single-stream decode through shorter accepts. [user-report] — [SGLang #36889](https://github.com/sgl-project/sglang/issues/36889)
- In a separate report, 4 slots were held per request but the cap was computed for 5, leaving about 20% of the cache unreachable. [snippet] — [jpezzulli/sglang-rtxpro6000 #5](https://github.com/jpezzulli/sglang-rtxpro6000/issues/5)
- `--mamba-full-memory-ratio` (default 0.9) sets the ratio of mamba memory to KV memory. Raising `--max-mamba-cache-size` reduces KV space. In KV-bound cases the benefit of V2 overlap must be weighed against lower max concurrency. [vendor] — [Qwen3-Next cookbook](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3-Next)
- vLLM: speculative tokens consume KV cache capacity and shrink the effective batch. [vendor] — [vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3.5.html)
- DFlash's KV-injection design avoids storing target latents to save KV space. [vendor] — [LMSYS blog](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- On RTX 3090 24 GB, a vLLM MTP comparison for Qwen3.8-27B could not start because of extra BF16 allocations. [indep] — [HackMD](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090)

### Inferences (my arithmetic from the Qwen3.8-27B card dimensions)
- **GDN state:** 48 GDN layers × 48 V heads × 128 × 128 × 4 B ≈ **151 MB per request in fp32**, about 37.7 MB per GPU at TP4 (18.9 MB in fp16/bf16). For 16 streams on SGLang extra_buffer:
  - MTP (5 slots/request): about 3.0 GB per GPU in fp32, 1.5 GB in fp16.
  - DFlash2 (1+8 = 9 slots): about 5.4 GB per GPU in fp32, 2.7 GB in fp16.
  - No spec: about 0.6-1.2 GB per GPU.

  On a 24 GB L4 that is a large share of the memory left for KV.
- **Full-attention KV:** 64 KiB/token in BF16, 16 KiB/GPU at TP4. A 70k-token request (prompt plus 60k thinking) needs about 1.07 GiB/GPU in BF16 or 0.53 GiB in FP8. For 16 such streams that is about 17 GiB/GPU in BF16 (does not fit) or about 8.5 GiB in FP8.
- **MTP draft KV:** one attention layer adds about 1/16 (6%) of target KV, assuming the same KV shape.
- **DFlash2 weights:** about 0.95 GB/GPU if TP-sharded (3.8 GB if replicated), plus its own context KV (layer count unpublished).
- **Net:** on 4x L4, MTP costs roughly 2-3 GB per GPU at 16 streams in fp32 SSM (about 1-1.5 GB in fp16). DFlash2 costs roughly 4-6.5 GB per GPU. The DFlash2 overhead alone could cost several 70k-token streams of KV capacity per GPU.

### Gaps
- Exact MTP tensor sizes for Qwen3.8-27B (check the safetensors index).
- DFlash2 layer count and draft-KV size.
- Whether SGLang shards the DFLASH drafter under TP4.
- vLLM's per-request GDN state multiplier with `num_speculative_tokens`.

## Q6. DFlash2 vs MTP head-to-head, and interaction with INT4/FP8 target quantization

### Takeaway
The only same-model head-to-head is the vendor's Qwen3.8-27B table: BF16 target, H200, 4k outputs, DFlash2 ahead at every concurrency. The single independent same-model test (RTX 3090, Q4 target, batch 1) found MTP n-max 2 ≥ DFlash2. No source measures acceptance for an INT4 (W4A16) target vs BF16 target for either method. Indirect evidence suggests quantized targets lose little acceptance but shrink speedups. Whether the RedHat INT4 checkpoint keeps the MTP head in BF16 is not stated on its card.

### Cited Findings
- Head-to-head on Qwen3.8-27B (BF16 target, H200, SGLang): see the Q1/Q2 tables. DFlash2 AL is +0.4-0.8 over MTP, and throughput at c=32 is 1.01-1.45x (DFlash2) vs 0.77-1.04x (MTP). [vendor] — [incoai card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
- Head-to-head at batch 1 on RTX 3090 with a Q4_K_XL target (llama.cpp): MTP n-max 2 +59.8% vs DFlash2 n-max 4 +51.9%. "No paired test establishes which method wins," and the ordering reversed at some long-context settings. [indep] — [HackMD](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090)
- DFlash2 with a **Q4_K_M target** (llama.cpp, xhigh, first 8 GSM8K examples, 2048 tokens): AL 5.28 with the BF16 draft, 5.13 with Q8_0 and 5.39 with Q4_K_M. Compare 5.46 on GSM8K with a BF16 target in SGLang. Different engine and a tiny sample. [vendor] — [incoai GGUF](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF)
- MTP speedup shrinks as the target quant gets coarser (Qwen3.6-27B, M2 Max, N=3, 2.5k tokens):

  | Quant | Code | Creative |
  |---|---|---|
  | F16 | +171% | — |
  | Q8_0 | +123% | — |
  | Q6_K | +50% | — |
  | Q4_K_M | +31% | −9% |

  Mainline llama.cpp gives Q4_K_M code +18%. Acceptance per quant is not given. The faster quantized baseline is a confound. [indep] — [froggeric GGUF](https://huggingface.co/froggeric/Qwen3.6-27B-MTP-GGUF)
- MTP head precision. The NVIDIA NVFP4 Flash-Next export ships FP8 block-scaled MTP. The RadixArk BF16 draft measured about the same: accept 3.33 vs 3.31, TPOT at 16 requests 18.5 vs 19.1 ms. [vendor: SGLang] — [Flash-Next cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)
- **RedHatAI/Qwen3.8-27B-INT4** (W4A16 GPTQ with AWQ smoothing, FP8 KV): unquantized parts are the vision tower, lm_head, embed_tokens and GDN `in_proj_a`/`in_proj_b`. The ignore list does not mention MTP layers. The card serves with `{"method":"mtp","num_speculative_tokens":3}` or DSpark with 7. Accuracy recovery is 98.5-101%. No acceptance numbers. [vendor] — [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)
- Third-party INT4 variants: jksdfosd `-mtpq` ships `model_mtp.safetensors` (precision not stated) and keeps lm_head and embed in bf16. A `-both-mtp` variant also exists. [user-report] — [jksdfosd/Qwen3.8-27B-INT4-mtpq](https://huggingface.co/jksdfosd/Qwen3.8-27B-INT4-mtpq); [jksdfosd/Qwen3.8-27B-INT4-both-mtp](https://huggingface.co/jksdfosd/Qwen3.8-27B-INT4-both-mtp) [snippet]
- The Kaitchup Qwen3.5-27B INT4/NVFP4 variants leave the MTP head unquantized. [indep] — [Kaitchup](https://kaitchup.substack.com/p/qwen35-27b-latency-and-throughput). In llama.cpp, IQ2_M keeps MTP tensors at Q4_K. [indep] — [froggeric](https://huggingface.co/froggeric/Qwen3.6-27B-MTP-GGUF)
- A quantized-target study trained a block-diffusion drafter in two stages (high-precision target, then the quantized target) for Qwen3.5-4B. It reports 6.978x average speedup on A10G. This implies quantization-induced drafter/target mismatch is worth adapting to. [academic] — [arXiv 2607.04244](https://arxiv.org/abs/2607.04244)
- In the DFlash2 corruption bug, a dense BF16 lm_head reduced wrong answers 111/304 → 1/304. Quantized output layers interact badly with that verify path. [user-report] — [SGLang #36548](https://github.com/sgl-project/sglang/issues/36548)
- Unsloth says MTP gives about 1.4-2.2x with no accuracy change; about 160 tok/s for 27B MTP on RTX 6000. [snippet] — [Unsloth MTP guide](https://unsloth.ai/docs/models/mtp)

### Inferences
- Quantizing the target probably lowers acceptance only slightly. The drafter predicts the quantized model's own distribution, and the DFlash2 Q4-target AL is about 5.3 vs 5.46. But the speedup shrinks because the cheaper target reduces the share of time that speculation saves.
- Keeping the MTP head in BF16 (or FP8) appears to cost little memory (about 0.2 GB/GPU at TP4) and is the safe choice. Inspect the RedHat INT4 safetensors index for `mtp.*` tensors and their dtype before relying on it.

### Gaps
- No INT4-target vs BF16-target acceptance measurement for MTP or DFlash2 on Qwen3.8-27B.
- No head-to-head under 60k thinking or at 16 streams on a bandwidth-limited GPU.
- The RedHat INT4 card does not state whether the MTP head is present or its precision.

## Q7 (objective). Which method gives the largest aggregate throughput gain on 4x L4 TP4 at 6-16+ streams with ~60k thinking tokens?

### Takeaway
The evidence favors **built-in MTP via SGLang EAGLE/NEXTN with topk 1, 2-3 steps and 3-4 draft tokens** as the best expected aggregate gain with acceptable risk. DFlash2 has the higher vendor numbers, but they come from short outputs on H200, and it has these against it:
- an open concurrency-correctness bug on this exact model ([SGLang #36548](https://github.com/sgl-project/sglang/issues/36548));
- documented acceptance and draft-cost decay at long outputs and long contexts ([arXiv 2605.09329](https://arxiv.org/html/2605.09329); [vLLM #54691](https://github.com/vllm-project/vllm/issues/54691), on 4x RTX 4090 TP4);
- about 2x the mamba-slot memory of MTP-4;
- Blackwell-only recommended attention backends.

EAGLE-3 trails native MTP on Qwen3.x. N-gram/suffix decoding is a slowdown on diverse text, and its hybrid-model and reasoning support is unproven. It is only interesting as a complement for repetitive grid output (vLLM/Arctic only). No speculation is the baseline. On big GPUs it wins above about c=32, but L4 at long context is more memory-bound, so speculation should still pay at 16 streams. This last point is my inference and is unmeasured.

### Cited Findings
- MTP is ahead of EAGLE-3 on Qwen3.x AL ([SPEED-Bench](https://arxiv.org/html/2604.09557v2); [shreyansh26](https://github.com/shreyansh26/qwen-spec-decode-benchmarking/blob/main/RESULTS.md); [AQ-MedAI](https://huggingface.co/AQ-MedAI/Qwen3.5-397B-A17B-eagle3)).
- DFlash2 is ahead of MTP on short outputs ([incoai card](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2), vendor), but not on a consumer GPU at batch 1 ([HackMD](https://hackmd.io/@thc1006/qwen3-8-27b-speculative-decoding-rtx-3090), indep).
- MTP AL on thinking output is about 2.5 of 4 ([Flash-Next cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next)).
- Speculation still helps at large batch when KV loading dominates ([MagicDec](https://arxiv.org/pdf/2408.11049)).
- Without tuning, the mamba slots for spec decoding cap concurrency 3-4x lower ([Flash-Next cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next); [SGLang #36889](https://github.com/sgl-project/sglang/issues/36889)).

### Inferences
- Practical recipe to test first (my synthesis, not measured on L4):
  - SGLang with `--speculative-algorithm EAGLE` (or NEXTN), `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`, then also try 2/1/3.
  - `extra_buffer_lazy` (4 slots) or the skip-decode-lock variant (3 slots).
  - Half-precision SSM state, with an acceptance check because bf16 cost about 6% in #36889.
  - `--max-mamba-cache-size` set explicitly to streams × slots.
  - FP8 KV.
  - Confirm the effective `max_running_requests` in the startup log or with `sglang:num_running_reqs`.
  - A/B accuracy on the same tasks with and without spec.
- Expected gain (my rough model): about 1.4-1.7x aggregate tokens/s at 16 streams if AL is about 2-2.5 on thinking and concurrency is not capped. If spec forces fewer concurrent streams, the net gain can vanish, as in the Flash-Next RTX PRO 6000 case (785 tok/s at 16 with MTP vs 1,258 at 64 without).
- DFlash2 is worth a bounded trial only after (a) #36548 is confirmed fixed in the chosen build, (b) a per-position AL log across 60k thinking shows no collapse, and (c) the slot and weight memory still fit the target stream count.

### Gaps
- No measurement of any method on L4, and none at 60k-token thinking at 6-16 streams. A short in-situ benchmark that logs AL per 10k-token window is the only way to settle MTP vs DFlash2 for this workload.
