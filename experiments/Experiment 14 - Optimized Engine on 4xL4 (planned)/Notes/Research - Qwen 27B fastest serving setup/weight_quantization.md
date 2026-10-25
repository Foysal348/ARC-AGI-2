# Weight and KV-cache quantization for Qwen3.8-27B (hybrid Gated DeltaNet + gated attention, MTP) on 4x NVIDIA L4 (sm_89)

Scope note: the notes use Qwen3.8-27B data wherever it exists. RedHatAI publishes INT4, NVFP4 and MXFP4 cards for Qwen3.8-27B; Qwen publishes an official FP8 checkpoint. Where Qwen3.8 data is missing, the notes use Qwen3.5/3.6-27B (same 64-layer hybrid layout) or Qwen3/QwQ/R1-distill papers, and say so on each line.

"Internal" sources are this project's own Kaggle 4x L4 runs, cited by local file path. They are real measurements but have confounds: engine, speculative method and format all change at once.

Architecture facts used throughout ([Qwen/Qwen3.8-27B-FP8 card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)):
- hidden 5120;
- 64 layers = 16 x (3 x [Gated DeltaNet -> FFN] + 1 x [Gated Attention -> FFN]);
- GDN: 48 V heads and 16 QK heads, head dim 128;
- attention: 24 Q heads, 4 KV heads, head dim 256;
- FFN 17,408;
- vocab 248,320;
- MTP head;
- 262k native context;
- thinking sampling T=1.0, top_p 0.95, top_k 20.

---

## Q1. Which GEMM kernels serve each format on sm_89 in vLLM 0.19+ and SGLang 0.5.x?

### Takeaway
On Ada every format except FP4 has a real kernel path:
- W4A16 (GPTQ/AWQ/compressed-tensors) runs on Marlin. Machete is Hopper-only.
- Per-channel/per-tensor FP8 W8A8 runs on CUTLASS FP8 (Ada FP8 tensor cores).
- Block-128 FP8, the official Qwen checkpoint format, falls back to a Triton block-FP8 GEMM on sm_89. Our own L4 logs show it running with default, untuned configs.
- INT8 W8A8 runs on CUTLASS INT8.
- NVFP4 and MXFP4 have no native path. They run only as weight-only W4A16 through FP4-Marlin.

### Cited Findings
- **vLLM support matrix (latest docs, dated 2026-09-14):**
  - Ada (SM 8.9) supports "Marlin (GPTQ/AWQ/FP8/FP4)", "llm-compressor INT8 (W8A8)" and "llm-compressor FP8 (W8A8)".
  - FP8 W8A8 is not supported on Ampere or Turing.
  - Turing does not support Marlin MXFP4.
  - [vLLM quantization docs (latest)](https://docs.vllm.ai/en/latest/features/quantization/)
- **Older vLLM v0.9.0 matrix:** the same Ada support, with FP8 (W8A8) on Ada and Hopper only and "Marlin (GPTQ/AWQ/FP8)" on Ampere, Ada and Hopper. [vLLM v0.9.0 supported hardware](https://docs.vllm.ai/en/v0.9.0/features/quantization/supported_hardware.html)
- **vLLM `Fp8LinearMethod`:**
  - Delegates kernel choice to `init_fp8_linear_kernel`.
  - Uses a weight-only Marlin path only "for GPUs that lack FP8 hardware support".
  - `block_quant = weight_block_size is not None`.
  - The minimum capability for the fp8 method is 75 (Marlin covers pre-Ada).
  - [vLLM fp8.py (main)](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/layers/quantization/fp8.py)
- **vLLM block-FP8 dispatch:**
  - Order: DeepGEMM first if supported, then CUTLASS block-scaled if `cutlass_block_fp8_supported`, then AITER (ROCm only).
  - Otherwise the Triton `w8a8_block_fp8_matmul`.
  - The only capability checks visible are SM90-specific CUTLASS block handling.
  - [vLLM fp8_utils v0.11.0](https://docs.vllm.ai/en/v0.11.0/api/vllm/model_executor/layers/quantization/utils/fp8_utils.html)
- **Ada block vs per-channel FP8 (third-party, vLLM 0.19.0 source):**
  - On Ada (SM 8.9), block-128 FP8 falls back to a Triton kernel. The CUTLASS block-scaled path needs SM90+ and DeepGEMM needs SM90/SM100.
  - Per-channel FP8 uses the CUTLASS FP8 GEMM; vLLM selects `CutlassFP8ScaledMMLinearKernel` on SM 8.9, which needs CUDA >= 12.4.
  - The source cites `csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu`.
  - [AzatAI/Qwen3.8-27B-FP8-dynamic card](https://friendli.ai/models/AzatAI/Qwen3.8-27B-FP8-dynamic)
- **Internal 4x L4 log, vLLM serving the block-128 Qwen3.8-27B-FP8 checkpoint:**
  - Prints "Selected CutlassFP8ScaledMMLinearKernel for Fp8LinearMethod".
  - Then repeatedly: "Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal! Config file not found at …/configs/N=4096,K=5120,device_name=NVIDIA_L4,dtype=fp8_w8a8,block_shape=[128,128].json".
  - The same warning appears for the shapes N=5120,K=1536; N=8704,K=5120; N=5120,K=4352; N=3584,K=5120.
  - The block-FP8 Triton path therefore runs with untuned tile configs on L4.
  - [Internal: E12-V vLLM SMOKE1 run2 server log](D:/ARC%20AGI%202/experiments/Experiment%2012%20-%20Duck%20REPL%20agent%20for%20ARC-AGI-2%20(planned)/Experiment%20Results/Kaggle%20Runs/E12-V%20vLLM%20SMOKE1%20run2%20(esfersami50,%202026-10-19)/q27x/vllm_openai_server.log)
- **Contradicting claim:** one model card says "Ada and Ampere GPUs serve the model through weight-only FP8-Marlin kernels". This conflicts with the vLLM matrix and with the internal L4 log, which shows native FP8 kernels on Ada; Marlin-FP8 is the pre-Ada (Ampere) path. [bottlecapai/ThinkingCap-Qwen3.6-27B-FP8](https://friendli.ai/models/bottlecapai/ThinkingCap-Qwen3.6-27B-FP8)
- **Machete** is "a mixed-input GEMM kernel optimized for NVIDIA Hopper GPUs" (w4a16), meant to beat Marlin in compute-bound, high-load serving on H100. It is not an Ada option. [Red Hat: Introducing Machete](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel) (search snippet; page not fetched)
- **vLLM FP4-Marlin:**
  - `is_fp4_marlin_supported()` returns `has_device_capability(80)`.
  - It handles NVFP4 (group 16, has `weight_scale_2`) and MXFP4 (group 32).
  - It logs "Your GPU does not have native support for FP4 computation … Weight-only FP4 compression will be used leveraging the Marlin kernel. This may degrade performance for compute-heavy workloads."
  - [vLLM marlin_utils_fp4 v0.10.2](https://docs.vllm.ai/en/v0.10.2/api/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.html)
- **SGLang `--fp8-gemm-backend`** (blockwise FP8):
  - `deep_gemm`: SM90/SM100. `flashinfer_trtllm`: SM100. `flashinfer_cutlass`: SM100/120. `flashinfer_deepgemm`: SM90. `cutlass`: SM120. `triton`: all hardware, the fallback.
  - `auto` order: DeepGEMM, FlashInfer TRTLLM, CUTLASS (SM120), AITER, then Triton. On sm_89 that order lands on Triton.
  - [SGLang quantization docs (md)](https://docs.sglang.io/docs/advanced_features/quantization.md)
- **SGLang other paths:**
  - `--quantization w8a8_fp8` / `w8a8_int8` "invoke the corresponding CUTLASS int8_kernel or fp8_kernel in sgl-kernel" for per-channel + per-token dynamic models.
  - `awq_marlin` / `gptq_marlin` are CUDA Marlin kernels.
  - For NVFP4, `--fp4-gemm-backend auto` selects `marlin` on SM80–SM90, as a weight-only W4A16 fallback.
  - [SGLang quantization docs (md)](https://docs.sglang.io/docs/advanced_features/quantization.md); [SGLang quantization page](https://docs.sglang.io/advanced_features/quantization.html)
- **GDN (linear attention) layers** are not GEMM-quantization-bound. vLLM on L4 logs "Using Triton/FLA GDN prefill kernel" for Qwen3_5ForConditionalGeneration. [Internal: E12-V server log](D:/ARC%20AGI%202/experiments/Experiment%2012%20-%20Duck%20REPL%20agent%20for%20ARC-AGI-2%20(planned)/Experiment%20Results/Kaggle%20Runs/E12-V%20vLLM%20SMOKE1%20run2%20(esfersami50,%202026-10-19)/q27x/vllm_openai_server.log)
- **TP4 on 4 PCIe L4s:** SGLang disables custom allreduce ("not supported on more than two PCIe-only GPUs"), so TP4 uses NCCL. vLLM was run with `disable_custom_all_reduce=True` in our notebooks. [Internal: E11-03 analysis](D:/ARC%20AGI%202/experiments/Experiment%2011%20-%20SGLang%200.5.21%20+%20RedHat%20INT4%20+%20DFlash2%20(planned)/Experiment%20Results/E11-03%20run1%20analysis%20(2026-10-17).md)

### Inferences
- With the official block-128 FP8 checkpoint on L4, the FP8 GEMMs probably run on the Triton block kernel with default configs, not CUTLASS.
- Two cheap levers follow:
  - Generate L4-tuned block-FP8 configs for the five shard shapes the log lists.
  - Switch to a per-channel FP8_DYNAMIC checkpoint (llm-compressor), which takes the CUTLASS FP8 path on sm_89.
- Neither lever has been measured on L4.
- INT4 W4A16 (RedHat compressed-tensors) on Ada runs on Marlin in both engines. There is no Machete alternative on sm_89.

### Gaps
- No public sm_89 benchmark of Triton block-FP8 vs CUTLASS per-channel FP8 at decode-sized M (8–128). The speed gap on L4 is unknown.
- I did not find the exact vLLM 0.19+ source of `cutlass_block_fp8_supported` or the scaled-mm kernel selector (`init_fp8_linear_kernel`). The sm_89 Triton fallback is taken from the third-party card plus our log.
- No confirmed statement of which SGLang 0.5.x kernel serves per-channel FP8 from a compressed-tensors checkpoint on sm_89, without `--quantization w8a8_fp8`.

---

## Q2. How does decode throughput scale with batch for each format? (published + internal benchmarks)

### Takeaway
No public benchmark of a 27B model on 4x L4 exists. The evidence instead comes from three places: Ampere/Ada papers, a Qwen3.8-27B comparison on a bandwidth-limited DGX Spark, and our own 4x L4 runs.

All of it says weight-only 4-bit has the highest per-step throughput while decode is memory-bound. Two things erode its edge as M grows: KV/state reads and, at very large M, Marlin's FP16-tensor-core compute ceiling. W8A8 formats catch up at high concurrency, and with speculative decoding, where M = streams x draft tokens.

On L4 (300 GB/s, 121 dense FP16 TFLOPS, 242 dense FP8 TFLOPS), decode stays memory-bound to a larger M than on A10/A100. INT4 should still lead FP8 at 8–32 streams, by a margin that shrinks with context length.

### Cited Findings
- **L4 hardware:**
  - 24 GB, 300 GB/s, 72 W, PCIe Gen4 x16 64 GB/s, no NVLink.
  - Tensor rates "with sparsity": FP16/BF16 242 TFLOPS, FP8 485 TFLOPS, INT8 485 TOPS, TF32 120.
  - Dense rates are half: about 121 BF16 and about 242 FP8/INT8.
  - [PNY L4 page](https://www.pny.com/nvidia-l4); dense = half per [NVIDIA L4 datasheet (Cisco mirror)](https://bxin.cisco.com/c/dam/en/us/products/collateral/servers-unified-computing/ucs-c-series-rack-servers/nvidia-l4-gpu.pdf) (search snippet)
- **Marlin batch scaling (A10):**
  - The kernel is near the ideal 3.87x vs FP16 at batch 1 and stays close up to batch 16–32, then falls toward about 1.5x at batch 128.
  - The paper says the kernel is memory-bound below about batch 50–64.
  - [Marlin paper (arXiv 2408.11743)](https://arxiv.org/html/2408.11743)
- **Marlin end-to-end vLLM speedup vs FP16** (Table 2, batch 1/8/16/32/64/128):

  | Model / GPU | 1 | 8 | 16 | 32 | 64 | 128 |
  |---|---|---|---|---|---|---|
  | Llama-2-7B / A10 | 2.93 | 2.90 | 2.74 | 2.26 | 1.78 | 1.20 |
  | Llama-2-70B / 4x A6000 | 2.84 | 2.84 | 2.67 | 2.02 | 1.74 | 1.23 |
  | Yi-34B / A100 | 2.90 | 2.79 | 2.69 | 2.36 | 1.71 | 1.18 |

  [Marlin paper](https://arxiv.org/html/2408.11743)
- **"Give Me BF16 or Give Me Death"** (A6000/A100/H100 only, no Ada):
  - In synchronous (low-concurrency) serving, W4A16 is best: 2–3x lower cost per query and 1.5–2.5x lower latency for 8B/70B.
  - In asynchronous continuous batching, "W8A8 formats (INT8 and FP8) give the highest throughput", while W4A16 "remains competitive and sometimes beats W8A8".
  - QPS speedups vs BF16 on code completion:

    | Setup | INT8 | INT4 | FP8 |
    |---|---|---|---|
    | 70B / 4x A6000 | 1.91x | 1.92x | – |
    | 70B / 4x A100 | 1.87x | 1.64x | – |
    | 70B / 4x H100 | – | 1.55x | 1.77x |

  - [arXiv 2411.02355](https://arxiv.org/html/2411.02355)
- **Qwen3.8-27B on DGX Spark, vLLM 0.27.1, 16 concurrent:**
  - unsloth NVFP4 (23.4 GB; "4-bit MLP + 8-bit attention + FP8 KV") vs official FP8 (30.9 GB, block 128).
  - Output throughput +29–34%: decode-heavy 1k->8k 99.47 -> 132.07 tok/s; balanced 104.44 -> 134.41.
  - TPOT 160.19 -> 120.60 ms (decode-heavy).
  - [NVIDIA forum](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dgx-spark-using-vllm-nvfp4-vs-fp8-performance/380258)
  - A separate GB10 thread says vLLM "still falls back to weight-only FP4 via Marlin on GB10". [NVIDIA forum GB10 thread](https://forums.developer.nvidia.com/t/state-of-native-nvfp4-kernel-support-on-gb10/372559) (search snippet)
- **Qwen3.8-27B per-channel FP8-dynamic on 2x RTX 6000 Ada (TP2, vLLM 0.19.0, single stream):**
  - Decode 38.7 tok/s without MTP, 63.3 with MTP k=3.
  - MTP mean acceptance 2.56 tokens/step on prose and 3.6 on code.
  - Decode is "memory-bandwidth-bound on Ada" at about 18.4 GB of weights per GPU per token.
  - [AzatAI card](https://friendli.ai/models/AzatAI/Qwen3.8-27B-FP8-dynamic)
- **Qwen3.6-27B on RTX PRO 6000 Blackwell** (not Ada; vLLM 0.25, batch 16): "FP8 decodes about 55% faster than bf16" (36.4 vs 23.4 tok/s). MTP adds about 1.8–2.0x. [bottlecapai card](https://friendli.ai/models/bottlecapai/ThinkingCap-Qwen3.6-27B-FP8)
- **Third-party Qwen3.8-27B NVFP4 on 4x RTX PRO 2000 Blackwell (TP4):** aggregate 69.0 tok/s at c=1, 202.3 at c=4, 377.8 at c=8 (self-reported). [sakamakismile/Qwen3.8-27B-MTP-NVFP4](https://friendli.ai/models/sakamakismile/Qwen3.8-27B-MTP-NVFP4) (search snippet only)
- **L40S (Ada), unverified:** FP8 cut chat TPOT from 23 ms to about 15 ms, and concurrency-16 throughput rose from 633 (bf16) to 949–971 tok/s. [Towards AI L40S playbook](https://pub.towardsai.net/the-vllm-optimization-playbook-for-l40s-backed-by-83-experiments-ae5da228f514) (search snippet only; page returned 403, model unknown)
- **A10G (Ampere, so FP8 there is weight-only FP8-Marlin), vLLM 0.30.0, Qwen3.5-9B, 200 requests, 1024 in / 128 out:**
  - BF16 117 tok/s, FP8-dynamic 338, GPTQ W4A16 363, AWQ W4A16 363.
  - GSM8K strict: FP8 87.5 vs GPTQ 81.8.
  - [computingforgeeks](https://computingforgeeks.com/?p=172295) (search snippet only)
- **Internal 4x L4, SGLang 0.5.21, RedHat Qwen3.8-27B-INT4 (W4A16 Marlin)** — total tok/s at real running counts (short prompts, about 4.3k):

  | Real running | A: INT4, no speculation | D: INT4 + DFlash2 + fp16 SSM | vLLM FP8 + MTP-3 (earlier runs) |
  |---|---|---|---|
  | 1 | 45.6 | 107.5 | – |
  | 4 | 145 | 278 | – |
  | 8 | 229 | 356 | – |
  | 12 | 289 | 404 | about 307 |
  | 16 | 347 | – | about 373 |

  - Long runs at 12 x 24,576 (capped): A 282 tok/s.
  - 4 x 85.5k contexts: about 238 tok/s decode at 4 running.
  - [Internal: E11-03 analysis](D:/ARC%20AGI%202/experiments/Experiment%2011%20-%20SGLang%200.5.21%20+%20RedHat%20INT4%20+%20DFlash2%20(planned)/Experiment%20Results/E11-03%20run1%20analysis%20(2026-10-17).md)
- **Internal production-like run, SGLang INT4 + DFlash2, real 12 running:**
  - 416.5 tok/s with fp16 SSM state, 412.1 with bf16.
  - By mean context: <10k 458, 10–20k 413, 20–40k 388 tok/s.
  - vLLM FP8 + MTP-3 production is "about 255–300 on long contexts".
  - [Internal: E11-04 analysis](D:/ARC%20AGI%202/experiments/Experiment%2011%20-%20SGLang%200.5.21%20+%20RedHat%20INT4%20+%20DFlash2%20(planned)/Experiment%20Results/E11-04%20run1%20analysis%20(2026-10-18).md)

### Inferences
- **Roofline for one L4 decode step at TP4, from the architecture above.** These are upper bounds that ignore allreduce, sampling and kernel inefficiency; measured numbers run at about 50% of them.
  - Weight bytes read per GPU per step: BF16 about 13.7 GB = 46 ms; FP8 7.15 GiB = 7.7 GB = 26 ms (vLLM measured load size, includes MTP); INT4 about 4.9 GB = 16 ms (19.45 GB / 4).
  - FP8-KV reads: 8 KiB per token per GPU (16 attention layers x 1 KV head x 256 x 2 x 1 B). 16 streams x 30k average context = 3.9 GB = 13 ms per step.
  - GDN state read+write (fp32): about 36 MiB per stream per GPU, so 16 streams = about 1.2 GB = 4 ms.
  - Resulting ceilings at B=16, 30k average context:

    | Format | Ceiling (tok/s) | vs FP8 |
    |---|---|---|
    | BF16 | about 254 | – |
    | FP8 | about 376 | 1.0x |
    | INT4 | about 482 | 1.28x |

  - At B=32 the ceilings are about 400 / 537 / 637, so INT4 leads FP8 by 1.19x. At B=8 they are 147 / 235 / 324, so INT4 leads by 1.38x.
  - INT4's lead over FP8 shrinks as KV reads grow with context and batch. It does not reverse in the memory-bound regime.
- **Compute ridge on L4:**
  - Marlin W4A16 computes on FP16 tensor cores (about 121 TFLOPS dense). Ridge = 121e12 / 300e9 ≈ 400 FLOP/byte; intensity ≈ 4M FLOP/byte; so it is memory-bound up to M ≈ 100.
  - FP8 W8A8 (242 TFLOPS dense) is memory-bound up to M ≈ 400.
  - The A10 has about 2x the bandwidth for similar FP16 compute, so Marlin's falloff, which starts near batch 32–64 on A10, should start at a larger M on L4.
  - With speculative decoding, M per GEMM = streams x (draft + 1). For example, 16 x 4 = 64 (MTP-3) and 16 x 8 = 128 (DFlash block 8).
  - Under speculation at more than 12–16 streams, INT4 Marlin approaches its compute ceiling, and FP8 W8A8's 2x higher tensor rate begins to matter. This is the regime where "Give me BF16" finds W8A8 ≥ W4A16.
- **Measured vs ceiling:** the internal SGLang-INT4 vs vLLM-FP8 numbers (+30–60% for INT4 + DFlash2 at 12 streams) agree in direction with the roofline. They are not an apples-to-apples format comparison, because engine and speculator also differ.
- **DGX Spark analog:** the device is a single bandwidth-bound part, as L4 is. Weight-only 4-bit (FP4-Marlin) beat block-FP8 by about 30% at 16 concurrency, close to the roofline-predicted 1.2–1.3x INT4-vs-FP8 gap on L4.

### Gaps
- No same-engine, same-speculator, 4x L4 comparison of BF16 vs FP8 (block or per-channel) vs INT4 at 8/16/24/32 streams with 30–60k contexts. This is the measurement that would settle the question.
- No public Ada (L4/L40S/4090/RTX 6000 Ada) batch sweep comparing FP8 W8A8 and W4A16 Marlin on a dense 27–32B model.
- NCCL allreduce cost per step on 4 PCIe L4s (128 allreduces per decode step for 64 layers) was not quantified in any source. It is a fixed per-step cost that favors larger batches for every format.

---

## Q3. Accuracy: FP8 vs INT4 W4A16 vs AWQ on long-CoT reasoning; thinking length; hybrid-specific sensitivities

### Takeaway
For Qwen3.8-27B, FP8 is effectively lossless; Qwen calls it "nearly identical" to BF16.

RedHat's INT4 (GPTQ with AWQ smoothing, W4A16) loses about 1.3 points on AIME25 and GPQA-Diamond (98.5–98.7% recovery). NVFP4 W4A4 with FP8 attention is similar, and MXFP4 W4A4 is worse (AIME25 95.7% recovery).

Papers find W8A8 and W4A16 near-lossless for 32B reasoners, with harder tasks degrading more. They disagree on whether 4-bit inflates thinking length: one study says no, another finds +4% (GPTQ) to +14% (AWQ) tokens on Qwen3-30B.

Hybrid-specific practice is consistent across sources: keep the GDN `in_proj_a/b` gates, lm_head, embeddings, MTP and vision in BF16, and do not drop the SSM state to bf16 without measuring.

### Cited Findings
- **Official FP8:** Qwen3.8-27B-FP8 is fine-grained FP8 with block size 128, and its performance metrics are "nearly identical to those of the original model". [Qwen3.8-27B-FP8 card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8)
- **RedHatAI/Qwen3.8-27B-INT4:** INT4 weights via the W4A16 GPTQ scheme with AWQ smoothing (`duo_scaling="both"`), activations unquantized, KV cache FP8 static per-tensor. Results, BF16 -> INT4:

  | Benchmark | BF16 | INT4 | Recovery |
  |---|---|---|---|
  | AIME 2025 | 95.42 | 94.17 | 98.69% |
  | GPQA-Diamond | 89.23 | 87.88 | 98.49% |
  | MATH-500 | 83.73 | 83.33 | 99.52% |
  | MMLU-Pro | 84.46 | 83.45 | 98.81% |
  | IFEval | 92.24 | 91.93 | 99.67% |
  | GSM8K-Platinum | 95.75 | 96.77 | 101.07% |

  3 seeds (8 for AIME). [RedHatAI/Qwen3.8-27B-INT4](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)
- **RedHatAI/Qwen3.8-27B-NVFP4:**
  - Scheme: MLP gate/up/down in NVFP4 (W4A4). Attention q/k/v/o, linear-attention `in_proj_qkv`, `in_proj_z`, `out_proj`, and the MLP of layers 56–63 in FP8 dynamic. FP8 KV. About 24.7 GB on disk vs about 54 GB BF16.
  - Results: AIME25 96.67 -> 95.00 (98.27%), GPQA 89.56 -> 89.22 (99.62%), MATH-500 100.72%.
  - [RedHatAI/Qwen3.8-27B-NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4)
- **RedHatAI/Qwen3.8-27B-MXFP4** (W4A4 MXFP4, GPTQ + AWQ smoothing): AIME25 96.67 -> 92.50 (95.7%), GPQA 89.56 -> 87.71 (97.9%), MATH-500 101.2%. [RedHatAI/Qwen3.8-27B-MXFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-MXFP4)
- **Baseline variance:** RedHat's two BF16 baselines for the same model differ (AIME25 95.42 vs 96.67; GPQA 89.23 vs 89.56). Run-to-run noise is about 1 point, the same size as the INT4 deltas. [INT4 card](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4); [NVFP4 card](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4)
- **Closest-architecture data, NVIDIA Qwen3.6-27B-NVFP4 vs Qwen3.6-27B-FP8:** AIME25 93.1 -> 92.7, GPQA 86.0 -> 85.5, HLE 21.7 -> 21.8, MMLU-Pro 86.1 -> 86.3, SciCode 44.8 -> 44.5. [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)
- **Community FP8-dynamic of Qwen3.6-27B** (n=1, 30 problems, noisy): AIME-24 83.33 -> 73.33; AIME-25 73.33 -> 73.33; MATH-500 86.6 -> 86.2. [soyrsoyr/Qwen3.6-27B-FP8-dynamic](https://friendli.ai/models/soyrsoyr/Qwen3.6-27B-FP8-dynamic) (search snippet only)
- **"Quantization Hurts Reasoning?" (v2)** — average over AIME-120 / MATH-500 / GSM8K / GPQA-D / LiveCodeBench:

  | Config | QwQ-32B (BF16 78.7) | R1-Distill-Qwen-32B (BF16 74.8) | Qwen3-8B (BF16 75.8) |
  |---|---|---|---|
  | W4 AWQ / GPTQ | −0.8 / −0.9 | −0.4 / −1.9 | −1.3 / −1.5 |
  | W3 | −4.4 to −6.9 | – | down to −14.5 |
  | W8A8KV8 (SmoothQuant / QuaRot / FlatQuant) | +0.3 / −0.1 / −0.2 | – | – |
  | KV4 | about −0.3 to +0.3 | – | – |

  - Harder benchmarks (AIME) degrade "up to 4x" more than easy ones (GSM8K).
  - Quantized models "do not generally think longer" at W4, W8A8KV8 and KV4. W3 and W4A4 can lengthen outputs, mostly in small models.
  - The authors hypothesize that error accumulates over long CoT, and that quantized models gain less from longer test-time reasoning.
  - The paper has no FP8 rows.
  - [arXiv 2504.04823v2](https://arxiv.org/html/2504.04823v2)
- **"Quantization Inflates Reasoning" (arXiv 2606.25519)** — CoT token inflation (CTIR) vs BF16, INT4 g128:

  | Model | GPTQ | AWQ | RTN | HQQ |
  |---|---|---|---|---|
  | Qwen3-30B | +3.7% | +14.4% | +43.0% | +3.9% |
  | Qwen3-4B | +12.0% | +8.2% | – | – |

  - Qwen3-30B INT3: GPTQ +19.0%, AWQ +32.8%.
  - Agentic BFCLv4: Qwen3-30B GPTQ-INT4 CoT tokens 5351 -> 5488.
  - Serving example: "1.1x CTIR increase raises p90 TTFT from 1050 s to 1400 s".
  - The paper has no 8-bit results.
  - [arXiv 2606.25519v2](https://arxiv.org/html/2606.25519v2)
  - This conflicts with arXiv 2504.04823's "no increased output length" finding for W4. The methods and models differ, so the question is open.
- **"Give Me BF16 or Give Me Death":**
  - W8A8-FP is "essentially lossless".
  - W8A8-INT shows a 1–3% per-task drop.
  - W4A16-INT recovers 99.36% (Leaderboard V1) and ≥96% (V2).
  - DeepSeek-R1-Distill reasoning recovers above 99%, except small INT4 models (Llama-8B 97.2%, Qwen-1.5B 93.5%).
  - RULER long-context recovery is ≥98% for all formats.
  - GPTQ beat AWQ by 2.9 points on Arena-Hard (8B).
  - [arXiv 2411.02355](https://arxiv.org/html/2411.02355)
- **Red Hat R1-distill quantized blog:** FP8 and INT8 recover near-perfectly on AIME24 / MATH-500 / GPQA-D. INT4 recovers 97%+ for ≥7B models. [Red Hat developers blog](https://developers.redhat.com/articles/2025/03/03/deployment-ready-reasoning-quantized-deepseek-r1-models) (search snippet)
- **Modules kept high-precision in the hybrid model:**
  - Official Qwen FP8 (as described for Qwen3.6-27B-FP8-style checkpoints) keeps lm_head, the MTP head, the vision tower and the Gated-DeltaNet input gates `linear_attn.in_proj_a/b` in bf16, "48-wide output doesn't tile a 128 block". [bottlecapai card](https://friendli.ai/models/bottlecapai/ThinkingCap-Qwen3.6-27B-FP8)
  - RedHat INT4 and MXFP4 ignore `visual.*`, `lm_head`, `embed_tokens`, `linear_attn.in_proj_a`, `linear_attn.in_proj_b`. [INT4 card](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4); [MXFP4 card](https://huggingface.co/RedHatAI/Qwen3.8-27B-MXFP4)
  - The AzatAI FP8-dynamic recipe ignores all of `linear_attn`. [AzatAI card](https://friendli.ai/models/AzatAI/Qwen3.8-27B-FP8-dynamic)
  - The RedHat Qwen3.5-9B W4A16 card reportedly leaves "linear attention layers" unquantized (19.3 GB -> 11 GB). [llm-compressor Qwen3.5 docs](https://docs.vllm.ai/projects/llm-compressor/en/stable/key-models/qwen3.5/) (search snippet)
- **SSM state precision:** SGLang's Qwen3-Next cookbook says `--mamba-ssm-dtype` is float32 by default; bfloat16 reduces mamba cache memory while float32 is more accurate. [SGLang Qwen3-Next cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3-Next) (search snippet; the page 404'd on fetch)
- **Internal Qwen3.8-27B SSM-state data (4x L4):**
  - fp16 SSM: 8/8 sanity and 416.5 tok/s. bf16 SSM: 7/8 and 412.1 tok/s.
  - Greedy drift with fp16 SSM (166-token common prefix) is no worse than INT4 + speculation alone (152–155).
  - INT4 + DFlash2 sanity was 7/8 against an FP8 expectation of 6.74, with "thinking tokens similar to FP8". One task thought 44.7k vs 23.2k.
  - These are 8 easy tasks, so this is not proof of eval parity.
  - [Internal: E11-03](D:/ARC%20AGI%202/experiments/Experiment%2011%20-%20SGLang%200.5.21%20+%20RedHat%20INT4%20+%20DFlash2%20(planned)/Experiment%20Results/E11-03%20run1%20analysis%20(2026-10-17).md); [Internal: E11-04](D:/ARC%20AGI%202/experiments/Experiment%2011%20-%20SGLang%200.5.21%20+%20RedHat%20INT4%20+%20DFlash2%20(planned)/Experiment%20Results/E11-04%20run1%20analysis%20(2026-10-18).md)

### Inferences
- **Accuracy ordering for Qwen3.8-27B long-CoT:** BF16 ≈ FP8 (block or per-channel) ≥ INT4 W4A16 GPTQ+AWQ ≈ NVFP4 (W4A4 MLP + FP8 attention) > MXFP4 W4A4.
- **Size of the INT4 gap:** about 1–1.5 points on AIME/GPQA, which is inside one-seed noise but is consistently negative. Papers say hard tasks amplify losses (about 4x), so on ARC-AGI-2-hard puzzles at 60k thinking the INT4 penalty could exceed the card's 1.3 points.
- **Expected token inflation:** for a GPTQ-based INT4, probably single-digit percent (+3.7% for GPTQ on Qwen3-30B). At equal accuracy this removes part of INT4's 1.2–1.4x per-token speed advantage, but not all of it.
- **On L4 the NVFP4/MXFP4 W4A4 checkpoints would run as W4A16** (Q6), so activations would not be quantized. Their accuracy on L4 should be at least the card value, but this is unverified.
- **For hybrid models,** the GDN state recurrence (not the weights) is the long-context-sensitive part. Keep the SSM state in fp32, or fp16 after measurement (internal evidence shows fp16 is safe and bf16 slightly worse). Keep `in_proj_a/b` (decay/beta gates) in BF16; every reputable recipe does.

### Gaps
- No published Qwen3.8-27B (or Qwen3.5/3.6-27B) eval of FP8 vs INT4 on LiveCodeBench or on very long (>50k) thinking traces. Card evals do not state max tokens.
- No study isolates SSM/GDN-state quantization sensitivity, or the effect of quantizing GDN `in_proj_qkvz`/`out_proj` to INT4 versus keeping them BF16.
- No study of whether 4-bit increases thinking length specifically for Qwen3.5+/3.8 hybrids at xhigh effort.
- No ARC-AGI-2 accuracy comparison by format beyond our own 8-task sanity checks.

---

## Q4. Memory math on 4x L4 (TP4): weights, KV, mamba state, concurrency at about 60k contexts

### Takeaway
At TP4 the per-GPU weights are about 13.5 GB (BF16), about 7.2 GiB (FP8, measured) and about 4.9 GB (INT4).

With FP8 KV, that leaves roughly 5 / 11 / 13.5 GiB per GPU for KV and state. Each 60k-token stream costs about 0.47 GiB FP8 KV (0.94 GiB BF16 KV) plus about 36 MiB fp32 GDN state per GPU. Speculative decoding adds extra state slots per stream.

Approximate worst-case concurrency at 60k, with FP8 KV:

| Weights | Streams at 60k (FP8 KV) |
|---|---|
| BF16 | about 10 |
| FP8 | about 20–22 |
| INT4 | about 25–28 |

BF16 KV halves all of these. BF16 weights cannot sustain 16–32 x 60k.

### Cited Findings
- **Checkpoint sizes:**
  - BF16 is about 54 GB ([RedHat NVFP4 card](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4)), or 55.6 GB including the vision encoder and MTP ([AzatAI card](https://friendli.ai/models/AzatAI/Qwen3.8-27B-FP8-dynamic)).
  - Official FP8 is 30.9 GB ([NVIDIA forum](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dgx-spark-using-vllm-nvfp4-vs-fp8-performance/380258)); Qwen3.6-27B-FP8 is about 29 GB ([bottlecapai](https://friendli.ai/models/bottlecapai/ThinkingCap-Qwen3.6-27B-FP8)).
  - RedHat INT4 is 19.45 GB = 18.6 GB model + 0.85 GB BF16 MTP ([internal memory note on the E11 model mirror](C:/Users/hp/.claude/projects/d--ARC-AGI-2/memory/sglang-int4-dflash2-plan.md); INT4 scheme per the [RedHat INT4 card](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)).
  - NVFP4 is about 24.7 GB ([RedHat NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4)); unsloth NVFP4 is 23.4 GB ([NVIDIA forum](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dgx-spark-using-vllm-nvfp4-vs-fp8-performance/380258)).
- **Internal vLLM on 4x L4, Qwen3.8-27B FP8 + MTP, gpu_memory_utilization 0.90:**
  - "Model loading took 7.15 GiB memory" per GPU.
  - "Available KV cache memory: 10.98 GiB".
  - "GPU KV cache size: 337,600 tokens".
  - "Maximum concurrency for 135,168 tokens per request: 8.46x".
  - vLLM set the attention block size to 1600 tokens so attention pages ≥ mamba pages, with Mamba cache mode 'align' under prefix caching.
  - [Internal: E12-V server log](D:/ARC%20AGI%202/experiments/Experiment%2012%20-%20Duck%20REPL%20agent%20for%20ARC-AGI-2%20(planned)/Experiment%20Results/Kaggle%20Runs/E12-V%20vLLM%20SMOKE1%20run2%20(esfersami50,%202026-10-19)/q27x/vllm_openai_server.log)
  - Our analysis records the vLLM FP8 pool as 1.35M tokens. [Internal: E11-03](D:/ARC%20AGI%202/experiments/Experiment%2011%20-%20SGLang%200.5.21%20+%20RedHat%20INT4%20+%20DFlash2%20(planned)/Experiment%20Results/E11-03%20run1%20analysis%20(2026-10-17).md)
- **Internal SGLang INT4 KV pools depend on configuration:**
  - 604k (INT4 alone), 474k (+MTP), 395k (+DFlash2, fp32 SSM) and 387k (+DFlash2, fp16 SSM) with auto mem_fraction 0.62. That left 4.39 GB per GPU unused.
  - The DFlash2 config's mamba cache was 3.26 GB per GPU = ssm 1.21 + intermediate ssm 1.97 + conv 0.08.
  - Speculative decoding needs 5 state slots per request, which capped running requests at 6–13.
  - [Internal: E11-03](D:/ARC%20AGI%202/experiments/Experiment%2011%20-%20SGLang%200.5.21%20+%20RedHat%20INT4%20+%20DFlash2%20(planned)/Experiment%20Results/E11-03%20run1%20analysis%20(2026-10-17).md)
  - With mem 0.83, mamba cache 56 and extra_buffer_lazy, the pool reached 881,336 KV tokens and 12 real running. [Internal: E11-04](D:/ARC%20AGI%202/experiments/Experiment%2011%20-%20SGLang%200.5.21%20+%20RedHat%20INT4%20+%20DFlash2%20(planned)/Experiment%20Results/E11-04%20run1%20analysis%20(2026-10-18).md)
- **Production need:** about 920k KV tokens. S1-60K peaked at 68% of 1.35M with 12 x 60k. [Internal: E11-03](D:/ARC%20AGI%202/experiments/Experiment%2011%20-%20SGLang%200.5.21%20+%20RedHat%20INT4%20+%20DFlash2%20(planned)/Experiment%20Results/E11-03%20run1%20analysis%20(2026-10-17).md)
- **vLLM recipe guidance (Qwen3.5-27B):** FP8 on "a single 40 GB GPU (H100/H200/L40S)"; Int4 on "a single 24 GB GPU"; `--language-model-only` skips the vision encoder. [vLLM recipe Qwen3.5-27B](https://recipes.vllm.ai/Qwen/Qwen3.5-27B)

### Inferences
These are derived from the architecture in the [Qwen card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) and the internal measurements.

- **KV per token.** Only the 16 full-attention layers hold KV, so 2 x 4 KV heads x 256 x 16 layers = 32,768 elements per token. That is 64 KiB (BF16) or 32 KiB (FP8) in total. At TP4 (1 KV head per GPU) it is 16 KiB or 8 KiB per GPU.
- **One 60k-token stream:**
  - KV: 3.66 GiB BF16 / 1.83 GiB FP8 in total, or 0.92 / 0.46 GiB per GPU.
  - GDN state: 48 layers x 48 heads x 128 x 128 = 37.7M elements. That is 151 MB fp32 / 75 MB fp16 per stream in total, or about 36 / 18 MiB per GPU, independent of context length.
  - Cross-check: the internal ssm pool of 1.21 GB over 68 slots ≈ 17.8 MB per slot fp16, which matches.
  - Conv state is about 0.7 MB per GPU per stream (negligible).
  - SGLang's 5 slots per request under speculation turn this into about 180 MiB fp32 / 90 MiB fp16 per stream per GPU. At 32 streams that is about 5.6 / 2.8 GiB, a real cost.
- **KV budget per GPU.** Keep vLLM's measured non-KV overhead (0.9 x about 22.5 GiB − 7.15 − 10.98 ≈ 2.1 GiB) and swap the weights:

  | Weights | Weights per GPU | KV + state budget | FP8-KV tokens | BF16-KV tokens |
  |---|---|---|---|---|
  | BF16 | about 12.8 GiB | about 5.3 GiB | about 0.69M | about 0.35M |
  | FP8 | 7.15 GiB (measured) | 10.98 GiB (measured) | about 1.35–1.44M | about 0.7M |
  | INT4 | about 4.5 GiB | about 13.6 GiB | about 1.78M | about 0.89M |

  GDN state per stream comes out of the same budget.
- **Worst case (every stream at 60k):**
  - FP8 KV: BF16 weights ≈ 10–11 streams, FP8 ≈ 21–22, INT4 ≈ 26–28 (fewer with speculative state slots).
  - BF16 KV: about 5 / 11 / 14 streams.
  - Average residency is lower than worst case: thinking grows from the prompt to 60k. With prefix caching the real number of concurrent streams is higher, but 16–32 x 60k needs FP8 weights or INT4 plus FP8 KV.
- **TP4 shards weights** (vocab-parallel embedding/lm_head included), so per-GPU weights ≈ total / 4. The vision encoder can be skipped with `--language-model-only` (vLLM) to save memory.
- **The 1.35M figure** is 4 x the printed 337,600 (4 KV-cache groups: 3 GDN groups + 1 attention group, each of 16 layers). The printed "max concurrency 8.46x at 135k" implies about 1.14M usable context tokens, so the printed per-group figure understates capacity. This interpretation is inferred from the numbers, not from vLLM documentation.

### Gaps
- No measured per-GPU load size for INT4 or BF16 under vLLM/SGLang TP4 on L4 (the INT4 4.5 GiB is computed). Activation and CUDA-graph overhead also differs by engine.
- The exact vLLM hybrid KV accounting (block sharing between GDN and attention groups in 'align' mode) was not confirmed from documentation.

---

## Q5. FP8 (e4m3) KV cache: accuracy impact on long reasoning, and capacity/throughput gain

### Takeaway
fp8_e4m3 KV is close to lossless for this model family. vLLM's FP8-KV study measured at most 0.7 points of loss on Qwen3.5-27B reasoning (lowest recovery 99%, AIME25) and full MRCR AUC parity to 1M context, using uncalibrated per-tensor scales (the worst case).

It halves KV bytes: 8 KiB vs 16 KiB per token per GPU here. Capacity doubles, and per-step KV read time halves. At 16 x 30k contexts on L4, that is worth about +25–30% decode throughput by roofline.

Caveat: those accuracy results are on Hopper FA3 / Blackwell FlashInfer kernels. On Ada, FP8 KV runs through FlashInfer or Triton attention; FA2 cannot do FP8 KV. So validation on L4 kernels is still needed.

### Cited Findings
- **vLLM FP8 KV blog, accuracy** (2026-04-22; uncalibrated, per-tensor scale 1.0 = worst case):
  - Qwen3.5-27B reasoning: "at most 0.7 points; lowest recovery was 99% (AIME25)".
  - Qwen3.5-27B MRCR up to 1M: aggregate AUC@1M "fully matches baseline".
  - Qwen3-30B-A3B-Thinking: at most 1–2 points, lowest recovery 97% (GPQA-D).
  - Before an FA3 two-level-accumulation fix, 128k needle-in-a-haystack fell from 91% to 13%; after the fix it was 89%.
  - [vLLM blog: State of FP8 KV-Cache](https://vllm.ai/blog/2026-04-22-fp8-kvcache)
- **vLLM FP8 KV blog, performance (H100):**
  - FP8 KV halves KV storage.
  - Decode ITL slope is 54–80% of BF16.
  - Break-even is about 7k tokens for Llama-3.1-8B after the fixes. The blog recommends staying BF16 below about 7k contexts.
  - At concurrency 8 with about 20k in / 2k out, output throughput rose 450.3 -> 517.5 tok/s (+14.9%).
  - The blog does not cover Ada, Triton or FA2.
  - [vLLM blog](https://vllm.ai/blog/2026-04-22-fp8-kvcache)
- **vLLM quantized-KV docs:**
  - fp8_e4m3 and fp8_e5m2 on CUDA 11.8+.
  - Default scales are 1.0 (uncalibrated); calibration via llm-compressor is recommended.
  - Per-head scales work only with the Flash Attention backend.
  - `--kv-cache-dtype-skip-layers` can keep sensitive layers in native dtype.
  - [vLLM quantized KV cache docs](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache.html)
- **FA backend gate:** vLLM's FlashAttention backend raises "FlashAttention does not support fp8 kv-cache on this device" when FP8 KV is unsupported. [vLLM flash_attn backend docs](https://docs.vllm.ai/en/v0.18.0/api/vllm/v1/attention/backends/flash_attn/) (search snippet)
- **Qwen3-Next user report:** switching from FlashInfer to Flash_attn fixed bad output, but that backend "doesn't support KV-Cache FP8". [HF discussion](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/discussions/35) (search snippet)
- **Internal L4 practice:**
  - The vLLM harness asserts `kv_cache_dtype == "fp8_e4m3"` with backend `TRITON_ATTN`, versus `FLASH_ATTN` when KV is auto. [Internal harness_e9.py](D:/ARC%20AGI%202/experiments/Experiment%209%20-%20Q27-XHIGH%2086k%20+%20Tail%20Resample%20attempt_2/Experiment%20Notebooks/Build/harness/harness_e9.py)
  - A later vLLM run selected the FLASHINFER backend and warned "Checkpoint does not provide a q scaling factor. Setting it to k_scale. This only matters for FP8 Attention backends". [Internal: E12-V server log](D:/ARC%20AGI%202/experiments/Experiment%2012%20-%20Duck%20REPL%20agent%20for%20ARC-AGI-2%20(planned)/Experiment%20Results/Kaggle%20Runs/E12-V%20vLLM%20SMOKE1%20run2%20(esfersami50,%202026-10-19)/q27x/vllm_openai_server.log)
- **Scale calibration:**
  - The RedHat INT4 card ships a static FP8 KV scheme (per-tensor, static min-max). [RedHat INT4 card](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)
  - The checkpoint has 32 calibrated scale tensors (0.021–0.254), yet SGLang warned "no scaling factors provided -> 1.0". Whether the scales are used at runtime is unverified. [Internal: E11-03](D:/ARC%20AGI%202/experiments/Experiment%2011%20-%20SGLang%200.5.21%20+%20RedHat%20INT4%20+%20DFlash2%20(planned)/Experiment%20Results/E11-03%20run1%20analysis%20(2026-10-17).md); [Internal: E11-04](D:/ARC%20AGI%202/experiments/Experiment%2011%20-%20SGLang%200.5.21%20+%20RedHat%20INT4%20+%20DFlash2%20(planned)/Experiment%20Results/E11-04%20run1%20analysis%20(2026-10-18).md)
- **Papers:** KV4 stays within about 1 point on reasoning (QwQ-32B −0.3 / +0.3), and within 1% on AIME-120 even at outputs >10k tokens. W8A8KV8 is about lossless (QwQ-32B −0.2 to +0.3). [arXiv 2504.04823v2](https://arxiv.org/html/2504.04823v2)

### Inferences
- **Capacity:** for this hybrid, KV is only on 16 of 64 layers, which already makes KV 4x cheaper than a dense 64-layer GQA model. Even so, FP8 KV doubles token capacity (Q4 table), and without it FP8 weights would hold only about 11 streams at 60k.
- **Throughput:** by the roofline (Q2), at 16 streams x 30k average, FP8 KV saves about 13 ms of a 55 ms step, about +30% decode throughput. At 32 streams the saving is larger. Below about 7k contexts the gain is negligible, per the vLLM blog.
- **Accuracy:** the Qwen3.5-27B ≤0.7-point result transfers well to Qwen3.8-27B (same attention layout). The risk on L4 is kernel-specific: FA3's accumulation bug did not apply to FlashInfer on Blackwell, and FlashInfer/Triton on Ada were not tested in the blog.
- **Scales:** use calibrated k/v scales if the engine actually loads them. Uncalibrated scale = 1.0 is the tested worst case and was still fine.

### Gaps
- No published accuracy study of FP8 KV on Ada attention kernels (FlashInfer sm_89 or vLLM Triton attention) for >32k reasoning traces.
- No L4-measured A/B of fp8_e4m3 vs bf16 KV throughput for this model.
- Unclear whether SGLang 0.5.21 loads compressed-tensors static KV scales for Qwen3.5-arch models.

---

## Q6. NVFP4 / MXFP4 on sm_89: native support? emulation path?

### Takeaway
Ada has no FP4 tensor cores. The L4 spec lists FP8, INT8, FP16/BF16 and TF32 tensor rates only. So NVFP4 and MXFP4 W4A4 cannot run natively on L4.

Both vLLM (FP4-Marlin, capability ≥ 8.0) and SGLang (`--fp4-gemm-backend marlin` on SM80–SM90) provide a weight-only W4A16 fallback. FP4 weights are dequantized in-kernel and multiplied on FP16 tensor cores, which works like INT4 Marlin.

This is an emulation with no speed advantage over INT4 W4A16. On Qwen3.8-27B the NVFP4/MXFP4 checkpoints are larger than INT4 (24.7 GB, partly FP8). MXFP4 also has worse published accuracy. Neither format beats RedHat INT4 on L4.

### Cited Findings
- **L4 tensor-core rates:** FP16/BF16, FP8, INT8 and TF32 only; no FP4 rate listed. [PNY L4](https://www.pny.com/nvidia-l4)
- **vLLM FP4-Marlin:** `is_fp4_marlin_supported = has_device_capability(80)`, supports NVFP4 (group 16) and MXFP4 (group 32), and warns that "Weight-only FP4 compression will be used leveraging the Marlin kernel. This may degrade performance for compute-heavy workloads." [vLLM marlin_utils_fp4](https://docs.vllm.ai/en/v0.10.2/api/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.html)
- **vLLM latest support matrix:** "Marlin (GPTQ/AWQ/FP8/FP4)" is ✅ on Ada; Turing lacks Marlin MXFP4. [vLLM quantization docs](https://docs.vllm.ai/en/latest/features/quantization/)
- **SGLang:** `--fp4-gemm-backend auto` "selects marlin on SM80-SM90" as a weight-only W4A16 fallback for NVFP4. The native FP4 backends need SM100/SM120. [SGLang quantization md](https://docs.sglang.io/docs/advanced_features/quantization.md)
- **Same fallback on other FP4-less parts:**
  - Hopper serves NVFP4 "through the Marlin WNA16 path". [vLLM issue 31085](https://github.com/vllm-project/vllm/issues/31085) / [GB10 forum](https://forums.developer.nvidia.com/t/state-of-native-nvfp4-kernel-support-on-gb10/372559) (search snippets)
  - An RTX 5090 logs the same Marlin warning when a backend-selection gap applies. [vLLM issue 47749](https://github.com/vllm-project/vllm/issues/47749) (search snippet)
- **Conflicting, dated report (Dec 2025):** a user tried every kernel on sm89 and "looks like sm100 is required … no fallback". This predates the current docs listing FP4-Marlin on Ada. [Reddit snapshot](https://reddit.sentinel-team.org/posts/1pe4xm4/snapshots/2025-12-05T08%3A30%3A58.532537Z) (search snippet)
- **Third-party Ada FP4 projects exist** ("AdaLLM … NVFP4-first inference runtime for Ada Lovelace"; [NVFP4-on-4090-vLLM repo](https://github.com/BenChaliah/NVFP4-on-4090-vLLM)). Not evaluated (search snippets only).
- **gpt-oss MXFP4:** the vLLM recipe says it runs on Ampere via the "Marlin MXFP4 MoE kernel" with TRITON_ATTN, and that Ada support was "actively being worked on" at the recipe's writing. [vLLM GPT-OSS recipe](https://docs.vllm.ai/projects/recipes/en/latest/OpenAI/GPT-OSS.html) (search snippet)
- **The Qwen3.8-27B FP4 checkpoints are W4A4:**
  - RedHat MXFP4: weights and activations FP4.
  - RedHat NVFP4: W4A4 MLP + FP8-dynamic attention / linear-attention / last-8-layer MLP.
  - nvidia/Qwen3.6-27B-NVFP4 lists Hopper and Blackwell as targets.
  - [RedHat MXFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-MXFP4); [RedHat NVFP4](https://huggingface.co/RedHatAI/Qwen3.8-27B-NVFP4); [nvidia/Qwen3.6-27B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4)
- **Measured speed on a fallback device:** on DGX Spark (vLLM, where FP4 reportedly falls back to Marlin), unsloth NVFP4 was +29–34% output tok/s vs official FP8 at 16 concurrency. [NVIDIA forum](https://forums.developer.nvidia.com/t/qwen3-8-27b-on-dgx-spark-using-vllm-nvfp4-vs-fp8-performance/380258)

### Inferences
- On L4, an NVFP4 or MXFP4 checkpoint is functionally a W4A16 checkpoint with FP4 (E2M1) weights and FP8/E8M0 group scales. Throughput should be about INT4-Marlin. The RedHat NVFP4 checkpoint should be slower than INT4, because its attention and linear-attention layers are FP8 (more bytes).
- There is no accuracy reason to prefer it on L4 either. RedHat INT4 (AIME25 98.7% recovery) already matches NVFP4 (98.3%) and beats MXFP4 (95.7%).
- Running a W4A4 checkpoint through the Marlin fallback, which skips activation quantization, should not be less accurate than the card. Whether vLLM/SGLang take the Marlin path for compressed-tensors NVFP4 W4A4 checkpoints on sm_89 (rather than erroring) has not been verified on L4.

### Gaps
- No L4 or sm_89 run of a Qwen3.8 NVFP4 or MXFP4 checkpoint in vLLM ≥0.19 or SGLang 0.5.x was found. Support is inferred from the code and docs.
- No throughput comparison of FP4-Marlin vs INT4-Marlin on Ada.

---

## Q7. INT8 W8A8 and BF16 on Ada, and the overall verdict for 8–32 long-reasoning streams

### Takeaway
INT8 W8A8 works on Ada (CUTLASS INT8 in vLLM and sgl-kernel) at the same byte size and tensor rate as FP8. It offers nothing over FP8 here: it needs calibration, has a slightly worse accuracy record, and no Qwen3.8-27B INT8 checkpoint is published.

BF16 is ruled out by capacity (about 10 x 60k streams even with FP8 KV) and by the slowest decode.

For the stated goal, the evidence ranks the configurations as follows:
1. **INT4 W4A16 (RedHat GPTQ+AWQ, Marlin) + fp8_e4m3 KV:** highest aggregate decode throughput and capacity, at a small (about 1–1.5 point), task-difficulty-amplified accuracy cost and possible single-digit token inflation.
2. **FP8 + fp8 KV:** near-lossless. About 0.75–0.85x INT4's ceiling at 8–32 streams, and relatively better under heavy speculative decoding. On L4 it has an untuned Triton block-FP8 path (block-128 checkpoint) or a CUTLASS path (per-channel FP8-dynamic).

### Cited Findings
- **INT8 support:** "llm-compressor INT8 (W8A8)" is ✅ on Ada in vLLM ([vLLM docs](https://docs.vllm.ai/en/latest/features/quantization/)). SGLang `w8a8_int8` uses the sgl-kernel CUTLASS int8 kernel ([SGLang md](https://docs.sglang.io/docs/advanced_features/quantization.md)).
- **L4 rates:** INT8 485 TOPS sparse = FP8 485 TFLOPS sparse. [PNY L4](https://www.pny.com/nvidia-l4)
- **INT8 accuracy:** W8A8-INT "modest drop of about 1–3% per task"; W8A8-FP "essentially lossless". [arXiv 2411.02355](https://arxiv.org/html/2411.02355)
- **No Qwen3.8-27B W8A8 or INT8 checkpoint** appears among the top-60 Hugging Face results for "Qwen3.8-27B". Available formats are Qwen FP8, RedHat INT4 / NVFP4 / MXFP4, cyankiwi AWQ-INT4, a W4A16 AutoRound-GPTQ, many NVFP4 and GGUF builds, and exl3. [HF API search](https://huggingface.co/api/models?search=Qwen3.8-27B&sort=downloads&limit=60); [RedHatAI listing](https://huggingface.co/api/models?search=Qwen3.8-27B&author=RedHatAI&limit=50)
- **Deployment recommendations in the literature:** W4A16 for synchronous / low-concurrency serving; W8A8 for high-throughput continuous batching, with a "crossover point [that] depends on model size, hardware, and task". [arXiv 2411.02355](https://arxiv.org/html/2411.02355)
- **Internal best measured config on 4x L4:**
  - SGLang + RedHat INT4 + DFlash2 + fp16 SSM + fp8 KV: about 404–417 tok/s at 12 real streams.
  - vLLM FP8 + MTP-3: about 255–300 on long contexts.
  - [Internal: E11-04](D:/ARC%20AGI%202/experiments/Experiment%2011%20-%20SGLang%200.5.21%20+%20RedHat%20INT4%20+%20DFlash2%20(planned)/Experiment%20Results/E11-04%20run1%20analysis%20(2026-10-18).md)

### Inferences
- **Highest aggregate decode throughput at 8–32 streams on 4x L4:** INT4 W4A16 Marlin. The margin over FP8 is about 1.2–1.4x by roofline and +30–60% measured, though the measurement is confounded by engine and speculator. The margin narrows at long contexts and very high speculative M.
- **INT4 also has the most KV/state headroom.** On L4, capacity rather than per-step speed is often the binding constraint at 60k.
- **Cost of INT4:** about 1–1.5 points on AIME/GPQA-class tasks, possibly more on the hardest tasks, and possible small thinking-length inflation. It needs a paired eval against FP8 on the target task set to confirm it is not a net loss.
- **Best accuracy-safe alternative:** FP8 weights + FP8 KV. On L4 a per-channel FP8_DYNAMIC checkpoint (CUTLASS sm_89 path) or L4-tuned block-FP8 Triton configs could narrow the gap to INT4. Both are unmeasured.
- **Not worth pursuing on L4:** INT8 W8A8 (no advantage, no checkpoint), BF16 (capacity), NVFP4/MXFP4 (Marlin emulation, larger or less accurate than INT4).
- **Mixed precision:** the RedHat NVFP4 layout (FP8 attention + 4-bit MLP + FP8 last 8 layers) is the published template for trading a little speed for accuracy. On L4 it would execute as FP8 W8A8 attention + FP4-Marlin MLP.

### Gaps
- No same-engine FP8 vs INT4 A/B on 4x L4 at fixed speculative configuration and 16–32 streams with 30–60k contexts.
- No paired ARC-AGI-2 eval of FP8 vs INT4 for Qwen3.8-27B at about 60k thinking.
- No data on whether per-channel FP8-dynamic (CUTLASS) is faster than block-128 FP8 (Triton) on L4.
