# Inference engine choice and engine configuration for Qwen3.8-27B (hybrid GDN + full attention, built-in MTP) on 4x NVIDIA L4 24 GB (sm_89, PCIe Gen4, no NVLink)

Scope note: research as of 2026-10-23. "Measured" means a number someone ran and reported; "claimed" means a vendor/doc statement without a shown run. Project-internal measurements on the real Kaggle 4xL4 box are cited as local files (they are the only Ada/L4 numbers for this exact model found anywhere). Current versions seen in sources: SGLang 0.5.21 (image `lmsysorg/sglang:v0.5.21-cu130`), vLLM 0.30.0 (Winder.AI, Sept 2026), TensorRT-LLM 1.3.0rc15, ExLlamaV3 1.x.

Hardware facts used throughout: L4 = 24 GB GDDR6, 300 GB/s memory bandwidth, FP8 485 TFLOPS with sparsity (about 242 dense), PCIe Gen4 x16 64 GB/s, 72 W — [Cisco copy of NVIDIA L4 datasheet](https://bxin.cisco.com/c/dam/en/us/products/collateral/servers-unified-computing/ucs-c-series-rack-servers/nvidia-l4-gpu.pdf); [PNY L4](https://www.pny.com/nvidia-l4).

Model facts used throughout:
- 64 layers: 16 full attention, 48 linear attention (Gated DeltaNet) with constant recurrent state; `full_attention_interval: 4`; architecture class `Qwen3_5ForConditionalGeneration` (same class as Qwen3.5 dense) with a vision tower; built-in MTP head; native context 262,144 — [vLLM recipe Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- Per-token KV (whole model): 32.8 KB at fp8, 65.5 KB at bf16. GDN state slot: 153.9 MB at float32 (the checkpoint's declared precision), 78.4 MB at bfloat16. FP8 weights are about 28.5 GB; NVFP4 about 16.5 GB — [SGLang cookbook Qwen3.8-27B](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- RedHat INT4 (W4A16 g128) checkpoint: 19.45 GB, of which 18.6 GB body plus 0.85 GB BF16 MTP head — project memory note, verified on HF by the project (local, `sglang-int4-dflash2-plan.md`)

---

## 1. Support: which engines run Qwen3.5/3.6/3.8 hybrids on sm_89, with which kernels; are TensorRT-LLM or llama.cpp realistic offline?

### Takeaway
Only SGLang and vLLM are realistic for 8–24 concurrent 60k-token streams on 4xL4. Both run the GDN layers on Triton (FLA-derived) kernels on Ada, because FlashInfer's GDN kernels need SM90+. Both have already been run on the project's Kaggle 4xL4. TensorRT-LLM lists no dense Qwen3.5/3.8 architecture. llama.cpp, ExLlamaV3 and LMDeploy can load the model family but have no high-concurrency evidence; llama.cpp tensor-split does not scale.

### Cited Findings
**SGLang**
- Cookbook pages exist for Qwen3.5, Qwen3-Next and Qwen3.8-27B. The Qwen3.8-27B page validates only H200, RTX PRO 6000, RTX 5090, DGX Spark and GB300; Ada/L4/sm_89 is not mentioned anywhere on it — [SGLang cookbook Qwen3.8-27B](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- Day-0 Qwen3.8 support blog (2026-08-12) covers the 2.4T MoE on GB300/B300. GDN decode/MTP uses FlashInfer CuTe-DSL kernels; Unified Radix Cache has a `FULL` component for KV and a `MAMBA` component for GDN checkpoints — [LMSYS blog Qwen3.8 day-0](https://lmsys.org/blog/2026-08-12-qwen3-8-day0-support/)
- Attention-backend defaults: for "Other Architectures (Ampere, Ada, etc.)" SGLang uses flashinfer if available, else triton; FA3 is the Hopper default. flashinfer and triton both support FP8 KV and speculative topk=1 and topk>1 — [SGLang attention backend docs](https://docs.sglang.io/docs/advanced_features/attention_backend)
- GDN backends: `--linear-attn-backend` defaults to triton, with per-phase overrides `--linear-attn-decode-backend` and `--linear-attn-prefill-backend`. The FlashInfer GDN backend is "CUDA, SM90/SM100/SM103"; CuTe DSL is CUDA, decode only — [SGLang attention backend docs](https://docs.sglang.io/docs/advanced_features/attention_backend)
- FlashInfer's `gated_delta_rule_decode` requires SM90; `chunk_gated_delta_rule` prefill requires SM90 or SM100 — [FlashInfer GDN decode docs](https://docs.flashinfer.ai/generated/flashinfer.gdn_decode.gated_delta_rule_decode.html); [FlashInfer GDN prefill docs](https://docs.flashinfer.ai/generated/flashinfer.gdn_prefill.chunk_gated_delta_rule.html)
- Measured on Kaggle 4xL4 (project): SGLang 0.5.21 + RedHat INT4 + DFlash2, TP4. All configs launched first try with 0 failed requests. Linear decode backend was triton. First launch took 601 s (prefill-graph JIT 447 s); later launches took 136–166 s — [E11-03 run1 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-03 run1 analysis (2026-10-17).md>)
- Offline wheel details for SGLang 0.5.21 (project, verified): torch 2.13.0, flashinfer_python 0.6.18 cu13, transformers 5.12.1. `cuda-tile==1.6.0rc5` is sdist-only and needs an override. `flashinfer-jit-cache 0.6.18+cu130` comes from flashinfer.ai, and Kaggle strips the '+' from dataset file names. The wheel set is about 220 wheels, about 5 GB — project memory `sglang-int4-dflash2-plan.md` (local)
- club-3090 (RTX 3090 community) says SGLang is "currently blocked on Ampere" for Qwen3.6-27B. This is a secondary summary; the reason is not stated and it does not apply to Ada directly — [search summary of club-3090 / qwen36-dual-3090](https://github.com/noonghunna/qwen36-dual-3090)

**vLLM**
- Qwen3.5-27B needs vLLM >= 0.17.0. The Qwen3.8-27B recipe also needs transformers >= 5.8.0. DFlash2 needs vLLM >= 0.28.0 (PR #52816) — [vLLM recipe Qwen3.5-27B](https://recipes.vllm.ai/Qwen/Qwen3.5-27B); [vLLM recipe Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- vLLM "integrates Triton kernels from Flash Linear Attention" for GDN and uses the hybrid KV cache manager. Full CUDA graph mode is on by default (blog 2025-09-11) — [vLLM blog Qwen3-Next](https://vllm.ai/blog/2025-09-11-qwen3-next)
- vLLM's Blackwell FlashInfer GDN prefill kernel gives 1.02x–5.78x over FLA/Triton. It is Blackwell-only, so Ada stays on FLA/Triton — [vLLM blog 25K TPS Qwen3.5](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps)
- Measured on Kaggle 4xL4 (project production): vLLM 0.19, TP4, FP8 block weights, fp8 KV, MTP-3, FLASHINFER attention, prefix caching, 16 seqs, max_model_len 106,496, KV 1.35M tokens. On L4 all 5 block-FP8 GEMM shapes use vLLM's Triton w8a8 kernel with the *default* config ("Performance might be sub-optimal"). FP8 Marlin is disabled on cc >= 8.9 unless `VLLM_TEST_FORCE_FP8_MARLIN=1` — [E12 FP8 kernel tuning README (local)](<D:/ARC AGI 2/experiments/Experiment 12 - FP8 kernel tuning on L4 (planned)/README.md>)

**TensorRT-LLM**
- The PyTorch-backend supported-models list (1.3.0rc15) has `Qwen3NextForCausalLM` (Qwen3-Next-80B-A3B) and `Qwen3_5MoeForCausalLM`, the latter "Supported via the AutoDeploy backend". There is no dense Qwen3.5/3.6/3.8 row (`Qwen3_5ForConditionalGeneration` absent). The feature matrix shows MTP = No and DFlash = No for both. The Qwen3Next footnote says it "exhibits relatively low accuracy on the SciCode-AA-v2 benchmark" — [TRT-LLM supported models 1.3.0rc15](https://nvidia.github.io/TensorRT-LLM/1.3.0rc15/models/supported-models.html)
- The TRT-LLM support matrix lists Ada SM89 with FP8/INT8/INT4, with a note that FP8 and quantized types are not implemented for every model — [TRT-LLM support matrix 1.2.0rc1](https://nvidia.github.io/TensorRT-LLM/1.2.0rc1/legacy/reference/support-matrix.html)

**llama.cpp**
- Qwen3.8-27B GGUF (Q4_K_XL/Q6_K plus MTP GGUF) runs on CUDA (1x RTX 3090, WSL) at about 42.9 tok/s single-stream with MTP. An older build (221f0f6) produced garbage on the CUDA path for GDN layers; it was fixed by updating to ece963f (about build 10450) and replacing the stale `libggml-cuda.so` (Aug–Oct 2026) — [llama.cpp discussion #27164](https://github.com/ggml-org/llama.cpp/discussions/27164)
- Measured (4x3090 PCIe): llama.cpp tensor-split "did not scale" — Q4_K_M went from 76.4 tok/s at 2 GPUs to 78.7 at 4 GPUs — [club-3090 discussion #773](https://github.com/noonghunna/club-3090/discussions/773)
- Dual 3090: llama.cpp Q4_K_M + MTP reached about 51/60 TPS against vLLM dual 89/127 TPS (single stream) — [club-3090 repo summary via search](https://github.com/noonghunna/club-3090)

**ExLlamaV3**
- The README lists Qwen 3-Next (`Qwen3NextForCausalLM`), Qwen 3.5 (`Qwen3_5ForConditionalGeneration`), Qwen 3.5 MoE and "Qwen 3.8-Flash-Next". It also lists "flexible tensor-parallel and expert-parallel inference" and "continuous, dynamic batching, speculative decoding", with PyTorch CUDA >= 12.4. No throughput figures are given — [ExLlamaV3 README](https://raw.githubusercontent.com/turboderp-org/exllamav3/master/README.md)
- A secondary summary of v1.0.0 says it "added a graph path for all attn/GDN modules" and extended TP to most models — [scrapbox ExLlamaV3 notes](https://scrapbox.io/work4ai/ExLlamaV3)

**LMDeploy**
- TurboMind lists Qwen3.5 (0.8B–397B) with FP16/BF16, KV INT8 and W4A16, but not KV INT4, and no vision encoder. The PyTorch engine lists Qwen3.5 and Qwen3-Next with BF16 only. Qwen3.6/3.8 are not listed, and there is no FP8 column — [LMDeploy supported models](https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html)

### Inferences
- Qwen3.8-27B uses the same HF architecture class as Qwen3.5 dense (`Qwen3_5ForConditionalGeneration`). Engines that load Qwen3.5-27B (ExLlamaV3, LMDeploy TurboMind) probably load it too, but none publishes an Ada high-concurrency number. Their batching stacks are not built for 12–24 x 60k-token streams.
- On L4, every engine's GDN path is Triton (FLA-style). The FlashInfer/CuTe GDN speedups reported for Hopper/Blackwell (vLLM 5.78x prefill, SGLang ReplaySSM CuTe MTP kernels) do not transfer. Kernel-level differences between SGLang and vLLM on L4 are therefore mostly in scheduling, spec-decode verify, all-reduce and FP8/INT4 GEMMs, not in GDN.
- TensorRT-LLM is not a realistic option: no dense Qwen3.5/3.8 model class and no MTP for the related classes.
- llama.cpp is not realistic for aggregate throughput: no multi-GPU scaling, and no high-concurrency data.

### Gaps
- No public source runs SGLang or vLLM on L4/L40S/4090 with Qwen3.5/3.6/3.8 at high concurrency; only the project's Kaggle runs exist.
- I did not verify TensorRT-LLM wheel availability for offline install (py3.12/cu13), since the model class is missing anyway.
- I found no source on llama.cpp `-np` parallel-slot throughput for GDN models.
- I found no confirmation of ExLlamaV3 on Qwen3.8-27B specifically, or of its multi-stream throughput.

---

## 2. Engine speed: measured SGLang vs vLLM throughput for Qwen3-Next / 3.5 / 3.6 / 3.8

### Takeaway
On hybrid models the two engines are within about ±10–35 % of each other, and the winner flips with configuration. SGLang wins (+7 %) only once its mamba state cache is sized for the concurrency; with defaults it silently caps running requests and loses badly. vLLM won at high concurrency on Qwen3-Coder-Next when SGLang needed workaround flags. On the real 4xL4, SGLang INT4 + DFlash2 beat vLLM FP8 + MTP-3 by +17 % (production) to +39 % (synthetic). That comparison is confounded by quantization and drafter, so it is not an engine-only result.

### Cited Findings
- **Measured, H100 SXM, Qwen3.8-27B FP8, 2026-09-25, SGLang 0.5.20 vs vLLM 0.30.0, about 900 in / 256 out** — [Winder.AI comparison](https://winder.ai/vllm-vs-ollama-vs-sglang-llm-inference-comparison/):

  | concurrency | SGLang tok/s | vLLM tok/s |
  |---|---|---|
  | 1 | 81 | 76 |
  | 10 | 627 | 589 |
  | 50 | 1,725 | 1,610 |

  - The SGLang number at 50 required `--max-mamba-cache-size 256 --mem-fraction-static 0.9`. On defaults SGLang "capped itself at 20 running requests" and served 982 tok/s.
  - vLLM refused to start until `--max-num-seqs 256` was set.
  - Median TTFT at 50 was 1.5 s for SGLang and 2.1 s for vLLM.
- **Claimed (cited by Winder.AI from HelixML):** on 8x RTX PRO 6000 at 192 concurrent, SGLang Qwen3.8 throughput rose from 3,833 to 7,883 tok/s after the state cache was stored in bfloat16 — [Winder.AI](https://winder.ai/vllm-vs-ollama-vs-sglang-llm-inference-comparison/)
- **Measured, 1x RTX PRO 6000 Blackwell, Qwen3-Coder-Next-FP8 (hybrid GDN MoE), 2026-03-05** — [CloudRift](https://www.cloudrift.ai/blog/optimizing-qwen3-coder-rtx5090-pro6000):

  | MCR | SGLang tok/s | vLLM tok/s |
  |---|---|---|
  | 8 | 510 | 495 |
  | 16 | 733 | 779 |
  | 24 | 808 | 846 |
  | 32 | 886 | 988 |
  | 40 | 886 | 1,207 |
  | 48 | 864 | 1,130 |

  - SGLang's best was 898 at MCR 28; vLLM's best was 1,207 at MCR 40, 34 % higher.
  - SGLang needed `--attention-backend triton --disable-radix-cache --kv-cache-dtype bf16 --fp8-gemm-backend triton`, flags the author says reduce performance.
  - At low concurrency (MCR 4) SGLang led: 330.52 vs 276.50 tok/s, TTFT 1,480 vs 5,647 ms.
- **Measured, project Kaggle 4xL4, TP4, short (about 4.3k) prompts, total tok/s at REAL running count** — [E11-03 run1 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-03 run1 analysis (2026-10-17).md>):

  | real running | SGLang INT4 alone | SGLang INT4 + MTP | SGLang INT4 + DFlash2, fp16 SSM | vLLM 0.19 FP8 + MTP-3 |
  |---|---|---|---|---|
  | 1 | 45.6 | 55.1 | 107.5 | — |
  | 4 | 145 | 205 | 278 | — |
  | 8 | 229 | 297 | 356 | — |
  | 12 | 289 | capped at 10 | 404 | ~307 |
  | 16 | 347 | — | 381 (capped at 13) | ~373 |

- **Measured, project Kaggle 4xL4, closed-loop ARC prompts, T 1.0, 12 real running** — [E11-04 run1 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-04 run1 analysis (2026-10-18).md>):
  - SGLang INT4 + DFlash2: 416.5 tok/s with fp16 SSM, 412.1 with bf16 SSM.
  - By mean context: 458 (<10k), 413 (10–20k), 388 (20–40k).
  - vLLM FP8 + MTP-3 production runs: about 255–300 tok/s.
- **Measured, project Kaggle 4xL4 production solver runs** — [E11-05 run1 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-05 run1 analysis (2026-10-18).md>); [E11-06 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-06 P1-60K EVAL14 run1 analysis (2026-10-19).md>):

  | run | engine | tok/s | concurrency |
  |---|---|---|---|
  | E11-05 | SGLang INT4 + DFlash2 | 328.9 | 11 real running (of 12 set) |
  | E11-06 | SGLang INT4 + DFlash2 | 379.9 | max 12, 87.2 % cached prompt tokens |
  | R5 | vLLM FP8 + MTP-3 | 280.9 | 16 |
  | D66 / L66 | vLLM FP8 + MTP-3 | 255.8 / 254.1 | 16 / 12 |

  - E11-05 decode by context: 429 (<10k), 372 (10–20k), 340 (20–40k), 323 (40–60k). DFlash2 accept length median 4.55.
- **Measured, H200 x8, TP8, Qwen3.5-397B-A17B with MTP (SGLang main):** 222.84 tok/s at concurrency 1 (accept 3.12) and 1,767.45 output tok/s at concurrency 100 — [SGLang cookbook Qwen3.5](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5)
- **Measured, H200, SGLang v0.5.5, Qwen3-Next-80B-A3B-FP8 TP2, bs=1 MTP:**
  - 2-token window: 257.20 tok/s (accept 2.709)
  - 3-token window: 306.94 (3.413)
  - topk 4 / 8 draft tokens: 324.57 (4.231)
  - [PyTorch blog: Hybrid Models Meet SGLang](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/)
- **Context (non-hybrid):** Llama 3.1 8B on H100, vLLM and SGLang within 2 % — [Winder.AI](https://winder.ai/vllm-vs-ollama-vs-sglang-llm-inference-comparison/). Qwen2.5-7B on RTX 4090 (Ada) at concurrency 64: SGLang 0.5.15 1,718 vs vLLM 0.25.1 1,740 output tok/s, "a near tie" — [ComputingForGeeks](https://computingforgeeks.com/sglang-vs-vllm/)

### Inferences
- The decisive variable for SGLang on hybrids is whether `max_running_requests` is really reached: the mamba cache cap, and the 0.9 default `mamba-full-memory-ratio`. A comparison that does not assert the real running count is invalid. Winder's 982 vs 1,725 shows a 1.76x swing from this alone.
- On 4xL4 the measured SGLang advantage, about +17–29 % on real runs, comes from a bundle: INT4 weights (half the bytes per decode step on a 300 GB/s card), DFlash2 (accept about 4.5), and fp16 SSM state (more slots). An engine-only A/B (same quant and drafter on both engines) has not been run. vLLM >= 0.28 also supports DFlash2 and INT4, so a vLLM INT4 + DFlash2 arm would isolate the engine.
- On Ada, both engines run the same class of GDN Triton kernels, so the 34 % vLLM win (PRO 6000) and the 7 % SGLang win (H100) are both plausible outcomes. Neither should be taken as a prior for L4.

### Gaps
- No same-quant, same-spec SGLang vs vLLM measurement on any Ada GPU for Qwen3.5/3.6/3.8.
- No vLLM 0.28+ INT4 + DFlash2 run on L4 (the vLLM baseline in the project is 0.19 FP8 + MTP-3).
- The HelixML bf16-state 2x claim is second-hand (via Winder); I did not find the primary post.

---

## 3. Parallelism on 4 PCIe L4: TP4 vs 2x TP2 data-parallel (vs PP), and memory fit for FP8 / INT4

### Takeaway
On PCIe-only consumer/Ada boxes, community measurements show TP4 wins single-stream decode slightly and prefill/TTFT clearly (+55–62 %). At saturation, 2x TP2 replicas win aggregate throughput by up to 2.4x for a 27B hybrid INT4. Custom all-reduce is disabled at TP4 on >2 PCIe GPUs but allowed at TP2. For this workload (12+ streams x about 76k context), memory decides first: FP8 at TP2 leaves too little KV per replica, while INT4 at TP2 plausibly fits about 6 streams per replica. 2x TP2 INT4 is the arm to measure; FP8 should stay TP4.

### Cited Findings
- **Measured, 4x RTX 3090 PCIe Gen3, no NVLink, P2P patched, Qwen3.6-27B, vLLM 0.23–0.25.1** — [club-3090 discussion #773](https://github.com/noonghunna/club-3090/discussions/773):
  - Matched recipe (MTP n=3 both arms), TP2 → TP4:
    - narrative decode 79.26 → 82.44 (+4.0 %)
    - code decode 106.50 → 103.73 (−2.6 %)
    - prefill at 10k 1,556 → 2,407 (+54.6 %); at 90k 1,202 → 1,948 (+62.0 %)
    - TTFT at 90k 73.1 s → 45.6 s
  - AutoRound INT4 peak aggregate at c=64: one TP2 instance 473.1 tok/s vs one TP4 instance 294.8 tok/s.
  - With `--max-num-seqs 256 --enable-chunked-prefill`: Qwen3.6-27B INT4 TP4 gave 307 tok/s vs 737 for 2x TP2 instances (2.40x). gemma-4-12B AWQ gave 1,007 (TP4) vs 3,425 (4x TP1, 3.4x).
  - vLLM log at TP4: "Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs". TP2 keeps it active.
  - Forcing custom all-reduce on at TP4 (8 MiB cap): decode at c=1 80.95 → 92.92 (+14.8 %); c=16 output 144.06 → 164.66 (+14.3 %). But greedy outputs diverged on 4/5 prompts ("not eval-clean"), and prefill-sized tensors still paid registration overhead (about +9 % TTFT at c16).
  - TP2 with custom AR + verified P2P at 220 W beat a no-P2P reference by about +15.7 %.
- **Measured, RTX 3090, Qwen3.8-27B int4-AutoRound, vLLM 0.27.1, MTP 1 token, fp8 KV, 512-token responses, aggregate tok/s** — [biMEMO model card](https://huggingface.co/biMEMO/Qwen3.8-27B-int4-AutoRound):

  | concurrent | 1 GPU | TP2 | TP4 |
  |---|---|---|---|
  | 1 | 55.6 | 90.1 | 102.4 |
  | 2 | 111.4 | 176.1 | 187.4 |
  | 4 | 223.6 | 323.3 | 322.3 |
  | 8 | — | 518.6 | 469.4 |

  - The author calls 2 GPUs the "sweet spot" and says TP4 "did not scale as well at higher concurrency".
- vLLM custom all-reduce source: it returns early when world_size > 2 and the GPUs are not fully NVLink-connected; it is also disabled if the P2P test fails — [vLLM custom_all_reduce API docs](https://docs.vllm.ai/api/vllm/distributed/device_communicators/custom_all_reduce.html)
- On an 8x L20 (Ada sm_89) box, vLLM also logged that SymmMemCommunicator is not supported on device capability 8.9 — [vLLM forum 8xL20 GLM5](https://discuss.vllm.ai/t/8-l20-glm5-fp8/2470) (from search snippet)
- On a PCIe multi-NUMA Blackwell box, vLLM hung at worker init unless `--disable-custom-all-reduce`; `NCCL_P2P_DISABLE=1` was also needed — [vLLM forum](https://discuss.vllm.ai/t/vllm-hangs-during-worker-initialization-on-blackwell-pcie-gpus-unless-disable-custom-all-reduce-is-used/2540)
- An NCCL P2P toggle on dual consumer GPUs gave only about +2–5 % (sometimes negative); SHM is not always slower than P2P — [vLLM forum dual 5090 SHM vs P2P](https://discuss.vllm.ai/t/dual-rtx-5090-tp-2-shm-vs-patched-bar1-p2p-cumem-single-pass-2-7-mean-point-estimate/2870)
- **Measured, project Kaggle 4xL4:** SGLang TP4 log also says custom allreduce "not supported on more than two PCIe-only GPUs", so TP4 uses NCCL. Preflight NCCL all-reduce of 64 MB took 20.9 ms — [E11-03 run1 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-03 run1 analysis (2026-10-17).md>)
- vLLM data parallel "will work with both dense and MoE models". DP x TP GPUs are used (`--data-parallel-size 2 --tensor-parallel-size 2` = 4 GPUs). Internal load balancing uses the running/waiting queues. `--max-num-seqs` is per DP rank, and each DP engine has an independent KV cache — [vLLM data parallel deployment docs](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment.html)
- SGLang: native `--dp-size` is "highly not recommended for use right now"; use the SGLang Model Gateway (`python -m sglang_router.launch_server ... --dp-size N`). The tuning guide says "when there is enough GPU memory, always favor data parallelism" — [SGLang DP/DPA/SMG guide](https://docs.sglang.io/advanced_features/dp_dpa_smg_guide.html)
- vLLM docs mention pipeline parallelism as possibly preferable when GPUs lack NVLink (L40S example) — [vLLM tensor parallelism docs mirror](https://www.mintlify.com/vllm-project/vllm/features/tensor-parallelism) (from search snippet; no numbers)
- **Memory reference points**
  - vLLM 2x RTX 5090 TP2, 262,144 context — [vLLM recipe Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B):

    | precision | weights per GPU | KV tokens |
    |---|---|---|
    | FP8 | 14.28 GiB | 377,456 |
    | NVFP4 (Inferact) | 12.02 GiB | 445,875 |
    | NVFP4 (unsloth) | 10.64 GiB | 920,517 |

  - Project 4xL4 SGLang TP4 INT4 + DFlash2, per GPU — [E11-03 run1 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-03 run1 analysis (2026-10-17).md>):

    | item | GB |
    |---|---|
    | target weights | 4.62 |
    | drafter | 1.49 |
    | mamba (fp16) | 3.26 |
    | target KV | 2.94 |
    | draft KV | 0.92 |
    | prefill graph | 2.60 |
    | verify graphs | 0.88 |
    | free | 4.39 |

    Target + draft KV is about 9.99 KB per token per GPU.
  - After tuning (extra_buffer_lazy, mamba cache 56, no prefill graph, mem 0.83): 881,336 KV tokens at 12 real running, 2.2 GB free — [E11-04 run1 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-04 run1 analysis (2026-10-18).md>)
  - Production needs about 920k KV tokens for 12 x (prompt + 60k thinking + answer). The vLLM FP8 TP4 pool is 1.35M — [E11-03 run1 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-03 run1 analysis (2026-10-17).md>)

### Inferences
- **Communication.** 64 MB in 20.9 ms is about 3.1 GB/s algorithm bandwidth, or about 4.6 GB/s bus bandwidth at n=4, far below PCIe Gen4 x16. This suggests the Kaggle L4s are not using direct P2P (host-staged/SHM path). Inferred; not verified with `nvidia-smi topo -m` or `NCCL_DEBUG=INFO`.
  - Decode all-reduces are small and latency-bound: about 2 per layer x 64 layers per step, plus larger verify batches with speculative decoding. TP4 therefore pays NCCL latency about 128x per step without custom AR.
  - At TP2 the custom all-reduce gate is passed only if the P2P test succeeds. If Kaggle L4s lack P2P, TP2 also falls back to NCCL. Check the log line.
- **Bandwidth model (inference).** Per decode step, TP4 reads 1/4 of the weights per GPU, while 2x TP2 reads 1/2 per GPU per replica but runs two replicas. With KV reads per GPU equal, 2x TP2 beats TP4 only when TP4's all-reduce overhead per step exceeds about one quarter-model weight read: about 15 ms for INT4 (4.6 GB / 300 GB/s), about 24 ms for FP8.
  - At 12 streams with speculative verify on PCIe without P2P, that is plausible but unproven.
  - The 3090 data (2.4x at max-num-seqs 256; 518.6 vs 469.4 at c=8) leans toward DP. The 3090's PCIe Gen3 and a different NCCL path make this indicative only.
- **Memory fit (inference from cited per-token and per-slot sizes).**
  - KV per token per GPU: 8.2 KB at TP4, 16.4 KB at TP2 (fp8). The 32.8 KB/token implies 16 layers x 2 x 4 KV heads x 256 dim x 1 B. With only 4 KV heads, TP4 is the largest TP that shards KV without replication.
  - **FP8 at TP2:** about 14.3 GiB weights per GPU (vLLM 5090 figure) leaves at most about 6 GiB per GPU on an L4 at 0.9 utilization for KV + GDN state + graphs + activations. That gives an upper bound of about 390k tokens per replica before state and graphs, realistically about 150–300k. Two replicas then hold about 0.3–0.6M tokens, well short of about 920k. FP8 2x TP2 cannot hold 12 x 76k streams.
  - **INT4 at TP2:** about 9.3 GB weights + 0.4 GB MTP per GPU. Without an external drafter that leaves about 10 GiB per GPU, i.e. about 500k+ KV tokens per replica after a 6-request fp16 GDN state budget. That gives about 1M total, plausibly fitting 12 x 76k.
    - With DFlash2 (3.85 GB BF16 → about 1.9 GB per GPU at TP2, plus draft KV), capacity drops toward about 0.7–0.8M. This is tight.
- **PP.** It splits weights like TP but needs no per-layer all-reduce. With only 12–24 streams, PP bubbles and the per-stage GDN state make it unattractive. No measurement found.

### Gaps
- No measurement of 2x TP2 vs TP4 on L4 for this model, in either engine. The project planned the arm (E11-03 §5) but never ran it.
- Kaggle L4 P2P capability, and whether TP2 gets custom all-reduce, are unverified.
- No public PP benchmark for Qwen3.5/3.8-27B on PCIe GPUs.
- Exact per-GPU GDN state at TP2 (state shards by heads) is not measured; the estimates above scale the cited per-slot sizes.

---

## 4. Hybrid-model settings: SSM state dtype, radix strategies, mamba cache vs concurrency, prefix caching, mem-fraction, CUDA graphs, chunked prefill, backends, all-reduce, overlap

### Takeaway
In SGLang, the mamba (GDN) state pool, not KV, usually caps concurrency. Each request takes S state slots (radix off 1, no_buffer 3, extra_buffer_lazy 4, extra_buffer 5) plus D verify slots for speculative decoding. The cap is silent, so set `--max-mamba-cache-size = target_concurrency x S` (plus verify) and assert the real running count. fp32 state is the checkpoint default; bf16/fp16 halves state memory. fp16 has passed small project quality checks, while bf16 is treated as an accuracy gate by SGLang. vLLM's hybrid prefix caching is experimental `align` mode with a large (about 528-token) block.

### Cited Findings
**SGLang radix strategies and slots**
- Strategies: `extra_buffer` (S=5, the Qwen3.8-27B page's default), `extra_buffer_lazy` (S=4, "no accuracy cost stated"), `no_buffer` (S=3), radix cache disabled (S=1).
  - `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` frees one slot for the extra_buffer strategies. `extra_buffer` frees one more with the overlap scheduler off.
  - Verify term D: EAGLE/MTP 4, DFLASH2 8, DSPARK block+1, and 0 with `--enable-linear-replayssm-spec`.
  - `--mamba-full-memory-ratio` default 0.9 "over-provisions the KV pool and clamps concurrency". Formula: `ratio = (S + D) x state_bytes / (L x kv_bytes_per_token)`. `--max-mamba-cache-size = target_concurrency x S` overrides it.
  - [SGLang cookbook Qwen3.8-27B](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- The Qwen3.5 page instead calls `no_buffer` (V1) the default: "No overlap scheduler, lower memory usage". `extra_buffer` (V2) "enables overlap scheduling and branching-point caching", needs `--page-size 64` and the FLA kernel backend, and is NVIDIA-only. The page-size rule is `FLA_CHUNK_SIZE % page_size == 0` (chunk 64). The flag name appears both as `--mamba-radix-cache-strategy` and as `--mamba-scheduler-strategy` across doc versions — [SGLang cookbook Qwen3.5](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5); conflicting default with the [Qwen3.8-27B page](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B). The 0.5.21 source also has `auto`.
- **Verified in SGLang 0.5.21 source (project).**
  - `--mamba-radix-cache-strategy {auto,no_buffer,extra_buffer,extra_buffer_lazy}`, `--max-mamba-cache-size`, `--disable-prefill-cuda-graph`, `--cuda-graph-max-bs-decode`.
  - Slots per request: radix off 1, no_buffer 3, extra_buffer + overlap 5, lazy 4.
  - `/get_server_info` still reports `max_running_requests 32` when the real cap is lower; the cap is visible only in a log warning.
  - Measured caps on 4xL4 TP4: INT4 22; + MTP 10; + DFlash2 fp32 6; + DFlash2 fp16 13.
  - [E11-03 run1 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-03 run1 analysis (2026-10-17).md>)
- Measured on 4xL4: 12 set gave 11 really running (2,358 decode lines show 11 running + 1 queued). The likely cause is the PrefillAdder mamba-slot gate (unverified), costing about 8 %; candidate fix `--max-mamba-cache-size 64` — [E11-05 run1 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-05 run1 analysis (2026-10-18).md>)
- SGLang hybrid design (v0.5.5 era) — [PyTorch blog: Hybrid Models Meet SGLang (2025-12-03)](https://pytorch.org/blog/hybrid-models-meet-sglang-more-than-full-attention/):
  - Separate fixed-size Mamba pool (per request) and KV pool (per token); the "elastic memory pool" resizes them via CUDA VMM.
  - MambaRadixCache copies state on match and forks a checkpoint on insert, with separate LRU lists.
  - Speculative decoding gives each draft token a private state slot, and the last accepted slot is promoted. This is why D slots exist.
- ReplaySSM spec-verify (`--enable-linear-replayssm-spec`) replaces per-draft state snapshots with an input window replayed into an fp32 checkpoint (D=0). It requires GDN/KDA, topk ≤ 1, a triton/flashinfer linear decode backend and `SGLANG_RAGGED_VERIFY_MODE=static`. Project estimate: it frees about 1.8–2 GB per GPU on 4xL4; untested — [E11-03 run1 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-03 run1 analysis (2026-10-17).md>). Explicit non-fp32 with ReplaySSM "logs a state-drift warning at boot" — [SGLang cookbook Qwen3.8-27B](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)

**SSM state dtype**
- SGLang Qwen3.8-27B: fp32 153.9 MB per slot vs bf16 78.4 MB. Speed depends on the quant: NVFP4 + EAGLE 152.9 (fp32) vs 144.5 (bf16) tok/s/user; FP8 + EAGLE 106.3 vs 116.1. The page says to "treat bf16 as an accuracy gate". GSM8K across 202 cells was 93.18–95.15 % — [SGLang cookbook Qwen3.8-27B](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- AMD recipe sets `SGLANG_MAMBA_SSM_DTYPE=bfloat16` — [SGLang cookbook Qwen3.5](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5)
- vLLM 25K-TPS recipe uses `--mamba-ssm-cache-dtype bfloat16`, "significantly" increasing effective KV capacity (no number); GSM8K 88 % — [vLLM blog 25K TPS](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps)
- vLLM with `mamba_ssm_cache_dtype=auto` sets the temporal state dtype from the conv-state/model dtype (i.e. bf16, not the checkpoint's fp32) — [vLLM mamba_utils docs](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/mamba/mamba_utils/)
- Nemotron 3 Super: SSM cache stored FP32 by default. Directly casting it to FP16 "can result in up to 40% increase in verbosity when combined with W8A8" — [Nemotron 3 Super paper](https://arxiv.org/pdf/2604.12374) (search snippet). Nemotron 3 Ultra: FP16 with stochastic rounding preserves FP32-cache accuracy and verbosity; FP8 E4M3 degrades — [Nemotron 3 Ultra paper](https://arxiv.org/pdf/2606.15007) (search snippet). Different models.
- Single-user claim: Qwen3.6 outputs "identical with and without" bf16 mamba cache on their harness — [Friendli model card](https://friendli.ai/models/Twu31/Qwen3.6-35B-A3B-GPTQ-INT4-W4A16-LowLatency)
- **Measured on project 4xL4 (SGLang INT4 + DFlash2):**
  - fp16 vs fp32 state raised the mamba cap from 6 to 13 at the same memory. fp16 greedy drift was no worse than INT4 + speculation already causes (common prefix 166 vs 152–155 tokens of 512) — [E11-03 run1 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-03 run1 analysis (2026-10-17).md>)
  - fp16 416.5 tok/s with sanity 8/8; bf16 412.1 with 7/8 (8 easy puzzles; FP8 expectation 6.74) — [E11-04 run1 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-04 run1 analysis (2026-10-18).md>)

**Prefix caching for hybrids**
- vLLM V1 unified allocator: attention block size is raised to align with the mamba page size (e.g. 672 tokens on Nemotron-Nano-12B-v2), and mamba pages are padded. Prefix caching for Mamba-2 hybrids is "experimental" (2025-11-05) — [PyTorch blog: Hybrid Models as First-Class Citizens in vLLM](https://pytorch.org/?p=5797)
- The vLLM Qwen3.5 recipe says prefix caching with Mamba is experimental in "align" mode — [vLLM recipe Qwen3.5-27B](https://recipes.vllm.ai/Qwen/Qwen3.5-27B). LMCache: `--mamba-cache-mode align` is required for GDN ("does not support the all mode"), and align snapshots state only at scheduler-step ends — [LMCache Qwen3.5 recipe](https://docs.lmcache.ai/recipes/qwen3_5.html)
- vLLM 0.19.1 on Qwen3.5-4B: block size 528, so prompts under 528 tokens get 0 % hits. Measured hit rates: 479 tokens about 0 %, 552 tokens 95.4 %, 979 tokens 53.7 %. Open, marked stale; a maintainer called it a design limitation — [vLLM issue #40696](https://github.com/vllm-project/vllm/issues/40696)
- **Measured on project 4xL4 SGLang** — [E11-05 run1 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-05 run1 analysis (2026-10-18).md>):
  - forced-close continuation 98.1 % cached (4.77M prompt tokens)
  - phase-1 iteration ≥ 2: 10.6 % (example order shuffled)
  - earlier contcache test warm was 11x faster than cold
  - aborted requests appear not to be inserted unless the stream is drained first

**mem-fraction / gpu-memory-utilization**
- SGLang Qwen3.8-27B pins: DGX Spark 0.80 (0.85 caused earlyoom kills); RTX 5090 0.88–0.94 depending on drafter and SSM dtype — [SGLang cookbook Qwen3.8-27B](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- Project 4xL4 SGLang TP4: auto `mem_fraction_static` 0.62 left 4.39 GB per GPU (about 19 %) unused. 0.83 + lazy + mamba 56 + prefill graph off gave 881k KV with 2.2 GB free — [E11-03](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-03 run1 analysis (2026-10-17).md>); [E11-04](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-04 run1 analysis (2026-10-18).md>) (local)
- vLLM community TP2/TP4 Qwen3.8-27B INT4 on 3090 used `--gpu-memory-utilization 0.93`, `--max-num-seqs 32`, fp8 KV — [biMEMO card](https://huggingface.co/biMEMO/Qwen3.8-27B-int4-AutoRound)

**CUDA graphs**
- vLLM hybrids use `FULL_AND_PIECEWISE` by default (full graphs for decode, piecewise for mixed batches). It is needed because Triton GDN kernels have CPU launch overhead — [PyTorch blog vLLM hybrids](https://pytorch.org/?p=5797)
- vLLM troubleshooting: "CUDA graph / Mamba cache size error" → reduce `--max-cudagraph-capture-size` (default 512), PR #34571 — [vLLM recipe Qwen3.5-27B](https://recipes.vllm.ai/Qwen/Qwen3.5-27B). 1x RTX 5090 OOMs during graph capture without `--enforce-eager` — [vLLM recipe Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- SGLang: the prefill (piecewise) CUDA graph cost 2.60 GB per GPU and a 447 s first-launch JIT on 4xL4. Disabling it (`--disable-prefill-cuda-graph`) was one of the levers to reach 881k KV. Decode/verify graphs were captured up to the mamba cap — [E11-03 run1 analysis (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-03 run1 analysis (2026-10-17).md>)

**Chunked prefill**
- SGLang Qwen3.8-27B tip: `--chunked-prefill-size 2048`, "since 8192-token chunks stall decode about 600ms at a time". H200 uses 32768; RTX 5090 draft rows 1024/512 — [SGLang cookbook Qwen3.8-27B](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- vLLM: `--max-num-batched-tokens 16384` on prefill nodes gave about +8 % total TPS/GPU (GB200 disagg) — [vLLM blog 25K TPS](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps)

**KV cache dtype**
- SGLang recipes pin `--kv-cache-dtype fp8_e4m3`. Qwen checkpoints carry no calibrated KV scales, so the scale defaults to 1.0, which "may noticeably reduce accuracy on reasoning-heavy tasks" — [SGLang cookbook Qwen3.5](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5); [Qwen3.8-27B page](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- Project: both SGLang and vLLM logged "Using FP8 KV cache but no scaling factors" (1.0). The RedHat INT4 checkpoint contains 32 calibrated KV scale tensors (0.021–0.254); whether they are used at runtime is unverified — [E11-03](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-03 run1 analysis (2026-10-17).md>); [E11-04](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-04 run1 analysis (2026-10-18).md>) (local)
- vLLM on 3090: the FP8 KV path selects FlashInfer, while the INT8 KV path selects Triton. INT8 was +76.5 % on short context but −71.9 % at about 58k prompt (crossover about 7–10k) — [club-3090 #773](https://github.com/noonghunna/club-3090/discussions/773)

**Speculative decoding settings**
- SGLang MTP: `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` (NEXTN is an alias). DFlash2: `--speculative-algorithm DFLASH --speculative-draft-model-path ... --speculative-num-draft-tokens 8`. MTP with the FlashInfer attention backend needs FlashInfer newer than 0.6.15.post1, otherwise use `--attention-backend triton` — [SGLang cookbook Qwen3.8-27B](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- vLLM MTP: `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'`. DFlash2: method `dflash`, 7 tokens. MTP acceptance at TP2 was 0.771 (FP8) and 0.897/0.788 (NVFP4) — [vLLM recipe Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)

### Inferences
- For 12 streams with DFlash2 on SGLang, budget about 12 x (S + D) state slots. With lazy (S=4) + D=8 that is up to 144 slot-equivalents, though the intermediate verify cache is sized differently in the source.
  - At fp32 (153.9 MB per slot whole model) this is infeasible on 4x24 GB. That is why fp16 state, the lazy strategy and ReplaySSM (D=0, fp32 checkpoint) are the levers.
  - ReplaySSM is the only option that both frees memory and keeps fp32 state, but it is untested on L4.
- `no_buffer` saves 2 slots per request but disables the overlap scheduler. `extra_buffer_lazy` is the measured working compromise on 4xL4.
- For this workload (prompt about 8k, output about 60k), prefill is a small share of tokens. Small chunks (1–2k) protect decode at the cost of TTFT, and dropping the prefill CUDA graph frees memory for KV at negligible decode cost.

### Gaps
- No public accuracy study of fp16 vs fp32 GDN state on long (60k-token) Qwen thinking. Project evidence is 8 easy puzzles plus greedy-prefix drift.
- vLLM's per-request GDN state footprint and its interaction with `max_num_seqs` on L4 were not found in sources.
- The overlap scheduler on/off has not been measured for hybrids on Ada.

---

## 5. Known bugs and regressions (hybrid models, L4/Ada, FP8 KV, speculative decoding + CUDA graphs)

### Takeaway
The most damaging known failures are silent ones:
- SGLang running-request caps from the mamba cache
- DFlash acceptance collapsing to 0 under grammar plus mixed-chunk batches
- a quantized or inherited-quant DFlash2 drafter giving about 0 % acceptance
- a stale SGLang CUDA compute cache degrading quality
- vLLM hybrid prefix caching missing short prompts

Forcing custom all-reduce on TP4 PCIe is fast but not bit-clean.

### Cited Findings
- **SGLang #43514** (opened 2026-10-10, open): with DFlash + Qwen3.8-27B on v0.5.19/0.5.20 (4x H20 TP4), the accept rate dropped to 0.00 and every request in the batch emitted token 0 ("!") until max_tokens, with no error.
  - Trigger: grammar-constrained requests plus multi-chunk prefill admissions, with `--enable-mixed-chunk` and `extra_buffer`. About 30 % per round on a used instance, 0/10 on a fresh one.
  - Candidate PR #43518 ("dflash verify stal…") is not confirmed merged.
  - [sglang #43514](https://github.com/sgl-project/sglang/issues/43514)
- **SGLang #39087** (2026-09-11, open): a quantized DFlash2 draft silently gives about 0 % acceptance (accept length 1.03, decode about 38 tok/s, vs 3.71 and about 171 tok/s with the BF16 drafter) on 2x3090 TP2.
  - If `--speculative-draft-model-quantization` is unset, the drafter inherits the target's quant. Use `unquant`.
  - [sglang #39087](https://github.com/sgl-project/sglang/issues/39087)
- **SGLang #38009** (2026-09-04, open): with DFlash2 and thinking enabled, greedy output diverges from target-only output at token 30 on 4x RTX 3090 TP4. Another user saw accepted length vary across container restarts — [sglang #38009](https://github.com/sgl-project/sglang/issues/38009)
- **SGLang #42917** (2026-10-07): Qwen3.8-27B W4A16 PPL 9.98 vs vLLM 6.05 on 0.5.19–0.5.21 (RTX 5090). Root cause was a stale CUDA driver compute cache under `SGLANG_CACHE_DIR` (`~/.cache/sglang/nv`); clearing it gave 6.2784 vs 6.2753 — [sglang #42917](https://github.com/sgl-project/sglang/issues/42917)
- **SGLang silent concurrency cap:** defaults capped running at 20 (982 vs 1,725 tok/s at 50 concurrent) — [Winder.AI](https://winder.ai/vllm-vs-ollama-vs-sglang-llm-inference-comparison/). The default `mamba-full-memory-ratio` "silently clamps concurrency" — [SGLang cookbook Qwen3.8-27B](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B). Project: `/get_server_info` reports 32 while the real cap was 6–22 — [E11-03 (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-03 run1 analysis (2026-10-17).md>)
- **SGLang extra_buffer OOM in disaggregated decode** for the 397B model before a fix — [sglang #19045](https://github.com/sgl-project/sglang/issues/19045) (from search snippet)
- **SGLang #20774:** Qwen3.5-35B-A3B with CPU offload fails with a Mamba conv_states dtype mismatch — [sglang #20774](https://github.com/sgl-project/sglang/issues/20774)
- **SGLang on RTX PRO 6000 FP8 Qwen3-Coder-Next** needed triton attention, triton FP8 GEMM, radix cache off and bf16 KV to run — [CloudRift](https://www.cloudrift.ai/blog/optimizing-qwen3-coder-rtx5090-pro6000)
- **vLLM #40696:** hybrid prefix caching gives 0 % hits for prompts < 528 tokens (0.19.1) — [vllm #40696](https://github.com/vllm-project/vllm/issues/40696)
- **vLLM CUDA graph / Mamba cache size error** → lower `--max-cudagraph-capture-size` (PR #34571) — [vLLM recipe Qwen3.5-27B](https://recipes.vllm.ai/Qwen/Qwen3.5-27B). Graph-capture OOM on 1x5090 — [vLLM recipe Qwen3.8-27B](https://recipes.vllm.ai/Qwen/Qwen3.8-27B)
- **vLLM async scheduling** had race conditions; "accuracy was zero without the fixes" (fixed in the nightly used) — [vLLM blog 25K TPS](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps)
- **vLLM MTP with num_speculative_tokens 2 errored** for a user in March 2026; the maintainer said it should work — [vLLM forum Qwen3.5-27B-FP8 spec decoding](https://discuss.vllm.ai/t/qwen3-5-27b-fp8-speculative-decoding/2447.md) (search snippet)
- **vLLM MTP × TurboQuant CUDA graph capture bug** #40880, fixed by #40914 — [club-3090 README](https://github.com/noonghunna/club-3090)
- **vLLM Qwen3.5-4B with dtype float32** hit GDN-kernel warmup errors — [vLLM forum](https://discuss.vllm.ai/t/float32-qwen3-5-4b-load/2858)
- **vLLM custom all-reduce on PCIe:**
  - disabled at >2 GPUs
  - forcing it at TP4 gave +14.8 % but greedy divergence on 4/5 prompts — [club-3090 #773](https://github.com/noonghunna/club-3090/discussions/773)
  - init hang on multi-NUMA PCIe until disabled — [vLLM forum](https://discuss.vllm.ai/t/vllm-hangs-during-worker-initialization-on-blackwell-pcie-gpus-unless-disable-custom-all-reduce-is-used/2540)
  - SymmMem not supported on cc 8.9 — [vLLM forum 8xL20](https://discuss.vllm.ai/t/8-l20-glm5-fp8/2470)
- **Ada FP8 GEMM:** vLLM 0.19 block-FP8 on L4 runs untuned default Triton configs; FP8 Marlin is disabled on cc ≥ 8.9 unless forced — [E12 README (local)](<D:/ARC AGI 2/experiments/Experiment 12 - FP8 kernel tuning on L4 (planned)/README.md>)
- **FP8 KV scale 1.0** default (no calibrated scales) may hurt reasoning — [SGLang cookbook Qwen3.5](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5)
- **llama.cpp:** older CUDA builds give garbage on GDN layers — [llama.cpp #27164](https://github.com/ggml-org/llama.cpp/discussions/27164)

### Inferences
- **Mandatory guards for an unattended 12 h run:**
  - log-parse the real running count against target
  - log accept length per window and alarm on a collapse to about 1.0
  - use a fresh `SGLANG_CACHE_DIR` / `CUDA_CACHE_PATH`
  - set `--speculative-draft-model-quantization unquant` for a BF16 drafter
  - avoid grammar-constrained requests and `--enable-mixed-chunk` with DFLASH, since #43514's trigger needs grammar
- Do not force custom all-reduce at TP4 for evaluation runs (not bit-clean). Prefer TP2 replicas if all-reduce cost matters.

### Gaps
- No L4/Ada-specific SGLang bug reports were found. Ada issues may be under-reported because few users run hybrids on L4.
- #43514/#43518 fix status could not be confirmed.

---

## 6. Gains to expect from CUDA graphs, chunked prefill size, the overlap scheduler (and related levers) at high concurrency

### Takeaway
Published gains are modest and mostly measured on H100/GB200:
- vLLM full + piecewise CUDA graphs: +2–18 % (dense hybrid) and up to +91 % (small MoE hybrid) vs V0
- SGLang overlap scheduler: about 1.1x
- larger prefill batch: about +8 %

On the real 4xL4, the large levers measured were different:
- speculative decoding (DFlash2 +39 % over no-spec at 12 running)
- INT4 weights
- fixing the mamba-cache concurrency cap (up to 1.76x elsewhere)
- fp16/bf16 state (claimed 2x at 192 concurrency on PRO 6000)

### Cited Findings
- **vLLM CUDA graphs** (H100, v0.10.2, 32k in / 128 out, V1 vs V0): Nemotron-Nano-12B-v2 throughput gains "ranged from 2 percent to 18 percent"; granite-4.0-h-tiny "up to 91 percent". PIECEWISE alone was worse than V0 at low concurrency — [PyTorch blog vLLM hybrids](https://pytorch.org/?p=5797)
- **SGLang overlap (zero-overhead) scheduler:** 1.1x throughput (v0.4 blog, 2024-12-04, not hybrid-specific) — [LMSYS SGLang v0.4](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4). For hybrids, overlap requires `extra_buffer` / `extra_buffer_lazy` (`no_buffer` runs without it) — [SGLang cookbook Qwen3.5](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.5)
- **vLLM `--async-scheduling`** is named a key driver of 25K TPS/GPU, with no isolated number. `--max-num-batched-tokens 16384` gave about +8 %. A raised `--max-cudagraph-capture-size` gave no isolated gain — [vLLM blog 25K TPS](https://vllm.ai/blog/2026-08-06-qwen35-25k-tps)
- **Chunked prefill:** 8192 chunks stall decode about 600 ms each; tip 2048 — [SGLang cookbook Qwen3.8-27B](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- **Speculative decoding on 4xL4 (project, SGLang TP4, short prompts)** — [E11-03 (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-03 run1 analysis (2026-10-17).md>):
  - at 8 running: MTP 297 vs no-spec 229 (+30 %); DFlash2 fp16 356 (+55 %)
  - at 12: DFlash2 404 vs 289 (+39 %)
  - at 1: DFlash2 107.5 vs 45.6 (2.4x)
- **Spec decoding elsewhere:** MTP on the Qwen3.8 dual-PP6 NVFP4 backbone gave +10.0 % throughput and 2.33x per-user speed (GB300) — [LMSYS Qwen3.8 day-0](https://lmsys.org/blog/2026-08-12-qwen3-8-day0-support/). DFlash2 on RTX 5090 bf16 gave 4.92 ms median TPOT at accept 4.29 — [SGLang cookbook Qwen3.8-27B](https://lmsysorg.mintlify.app/cookbook/autoregressive/Qwen/Qwen3.8-27B)
- **State-cache sizing:** SGLang default-capped 982 → sized 1,725 tok/s at 50 concurrent (H100); bf16 state 3,833 → 7,883 tok/s at 192 concurrent (8x PRO 6000, cited) — [Winder.AI](https://winder.ai/vllm-vs-ollama-vs-sglang-llm-inference-comparison/). On 4xL4, 11 vs 12 really running cost about 8 % — [E11-05 (local)](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-05 run1 analysis (2026-10-18).md>)
- **Custom all-reduce** (when allowed): about +14–15 % decode on 3090 TP4 (forced) or TP2 — [club-3090 #773](https://github.com/noonghunna/club-3090/discussions/773)
- **Context-length decay** on 4xL4 SGLang at 11–12 running: 429–458 tok/s (<10k) → 323 tok/s (40–60k), i.e. about −25–30 % — [E11-04](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-04 run1 analysis (2026-10-18).md>); [E11-05](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/E11-05 run1 analysis (2026-10-18).md>) (local)

### Inferences
- **Remaining levers on 4xL4, in likely order of size:**
  1. 2x TP2 INT4 replicas, if memory fits and TP2 gets custom AR or the NCCL latency dominates at TP4. Plausibly +20–100 % from the 3090 analogues, unproven on L4.
  2. Real 12/12 running via `--max-mamba-cache-size` (about +8 %).
  3. ReplaySSM, to free verify-state memory for more streams or fp32 state.
  4. Tuned FP8 Triton configs, if staying on vLLM FP8.
- CUDA graphs and the overlap scheduler are already on in the measured configs, so their remaining gain is near zero. Turning the overlap scheduler off (`no_buffer`) to save state slots would cost about 10 %.
- **Long decode dominates at 60k-token outputs.** At about 330–380 tok/s total, 12 h yields about 14–16M tokens, i.e. about 240–270 requests of 60k. Each +10 % throughput is about +25 requests per 12 h.

### Gaps
- No isolated measurement of CUDA graph, overlap scheduler or chunk size gains on Ada for hybrid models.
- No measured gain for 2x TP2 or for ReplaySSM on L4.
- The SGLang 1.1x overlap figure is from 2024 and non-hybrid models.
