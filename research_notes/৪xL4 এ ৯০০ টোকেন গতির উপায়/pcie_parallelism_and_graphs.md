# Multi-GPU communication, parallel layout and CUDA graphs for Qwen3.8-27B decode on Kaggle 4x L4 (PCIe, no NVLink)

Research date 2026-10-29. Labels: **[M-ours]** = our own Kaggle measurement (file path given), **[M]** = a published measurement, **[C]** = a claim, doc statement or source reading without data, **[E]** = my inference.

This note builds on two earlier folders and does not repeat them in full: `research_notes/৪xL4 ইঞ্জিন বটলনেক যাচাই/nccl_p2p_l4.md` (NCCL SHM vs P2P literature, env-var catalogue, bench recipes) and `tp2_replicas.md` (2xTP2 vs TP4 literature, SGLang custom AR gates, DFlash PP restriction). The biggest change since those notes: **our own Kaggle runs answered the P2P question.**

### Headline facts from our own Kaggle runs (read these first)
- **Topology:** all four L4s are `PHB` to each other, one NUMA node, CPU affinity 0-45; `nvidia-smi topo -p2p r` = OK for every pair; `torch.cuda.can_device_access_peer` = True for every pair; NCCL 2.29.7+cuda13.2, torch 2.13.0+cu130, driver 580.178.04. nvidia-smi reports `pcie.link.gen.max = 3` and width 16 for every L4 inside the VM (L4 is a Gen4 part, so the guest sees a Gen3-class virtual link, or the field is misreported) **[M-ours]** — `experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Experiment Results/Kaggle Runs/E14A SLIM R2 run1 (hosen42, 2026-10-23)/bundle/e14/nccl_bench.json`
- **P2P works functionally but is 5.5x slower than NCCL's SHM path.** CUDA-graph all-reduce of 983,040 B (96 tokens x 5120 x bf16), 140 calls per graph, slowest-rank median **[M-ours]** (same file):

  | arm | transport | 983,040 B per call (graph) | eager 983,040 B | 64 MiB busbw | all-gather 11.9 MB/rank |
  |---|---|---|---|---|---|
  | baseline (NCCL default) | `SHM/direct/direct`, 2 coll channels | **343.9 µs** (algbw 2.86, busbw 4.29 GB/s) | 382 µs | 4.90 GB/s | 8.02 ms |
  | `NCCL_P2P_LEVEL=SYS` | `P2P/IPC` (isAllDirectP2p 1) | **1,880.5 µs** (busbw 0.78 GB/s) | 1,939 µs | **0.42 GB/s** | 79.7 ms |
  | `NCCL_SHM_USE_CUDA_MEMCPY=1` | `SHM/CE/direct` | **timed out** in the graph phase (240 s, killed) | 503 µs | 4.61 GB/s | — |

  8 KiB eager: SHM 90 µs, P2P 134 µs, SHM-CE 101 µs. All arms passed the correctness check (sum = 10.0).
- **In-engine profile (SGLang 0.5.21, TP4 INT4 + DFlash2, 12 streams x 8 verify tokens, 20 decode steps, TP rank 0):** NCCL all-reduce **50.9 ms of a 116 ms step (44 %)**, 140 calls; NCCL all-gather 8.1 ms (7 %, 3 calls); Marlin INT4 GEMM 25.3 ms (22 %); other GEMM 14.5 ms; GDN 8.0 ms; attention 2.7 ms; sampling 2.4 ms. **Communication = 59 ms of 116 ms (51 %)**; the bench predicted 56.3 ms **[M-ours]** — `experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/30 Claude - E14A SLIM R2 run results (2026-10-24).md`
- **FlashInfer 0.7.0.post1 PCIe-IPC all-reduce** (it maps peer memory over CUDA IPC, so it rides the slow P2P path): SGLang DFlash W6 0.39x (230 vs 583 tok/s at 8k), SGLang MTP 0.42x, vLLM 0.32 FP8 MTP3 0.68x (297 vs 434) — all greedy-gate PASS **[M-ours]** — `experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Experiment Results/EMS_SAFE_RISKY_VMS_rev2g_GPU_results_2026-10-26.md`
- **vLLM 0.32 logs on our box:** "FULL_AND_PIECEWISE is not supported with spec-decode for attention backend FlashInferBackend (support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE); setting cudagraph_mode=PIECEWISE" (52 occurrences); "SymmMemCommunicator: Device capability 8.9 not supported, communicator is not available"; "FlashInfer All Reduce is disabled because it is not supported for world_size=4" (and for world_size=2); `disable_custom_all_reduce=True` in every arm **[M-ours]** — `_scratch/ems_out/vms/` logs
- **2xTP2 measured:** SGLang DFlash W6 658 tok/s (1.15x) at 8k but 308 (0.57x) at 40k (152k KV tokens per replica, 3 streams at 40k); vLLM 2xTP2 14-16 / 2 streams, unusable. Custom AR was disabled on the TP2 replicas (`--disable-custom-all-reduce`, because P2P is slow) **[M-ours]** — same results file; `Experiment Notebooks/Build/build_ems.py` L1104-1108

**Consequence [E]:** on this VM every all-reduce that moves data GPU-to-GPU over P2P (vLLM/SGLang custom AR, FlashInfer PCIe-IPC, lukealonso one-shot, TRT-LLM one-shot/two-shot, Flash Communication's peer-DMA kernel) is structurally slower than NCCL SHM. The P2P ceiling we measured (busbw 0.42-0.78 GB/s) is far below GPU-to-host bandwidth (13.2 GB/s GPU→host, 12.4 GB/s host→GPU on g2-standard-48 per Spare Cores, cited in `nccl_p2p_l4.md`). Any faster all-reduce here has to go **through host memory** (like NCCL SHM, but better), **move fewer bytes**, or **be removed by a different layout**.

---

## Q1. NCCL tuning for PCIe-only 4-GPU decode (ALGO, PROTO, channels, P2P/SHM, buffers)

### Takeaway
On our box the two "big" NCCL levers are already measured and both lose: forcing P2P is 5.5x slower, and CE-memcpy SHM is slower in eager mode (503 vs 382 µs) and hung under CUDA-graph capture. What remains untested is **channel count** (our NCCL used only 2 collective channels, and SHM runs at ~35 % of the GPU↔host link floor), plus small effects from PROTO/ALGO/NTHREADS/BUFFSIZE. No published 4-rank SHM data shows a big win from those; expect single-digit percent unless more channels unlock host-link bandwidth.

### Cited Findings
- Our NCCL init on Kaggle: rings 0-1-2-3 on 2 channels, `Pattern 4, nChannels 1, bw 10.0/10.0, type PHB/PIX`, `coll channels 2, p2p 2`, `isAllDirectP2p 0 directMode 0 isAllCudaP2p 1`, transport `via SHM/direct/direct`, "Connected all rings, use ring PXN 0 GDR 1"; env in the bench: `NCCL_CUMEM_ENABLE=0 NCCL_NVLS_ENABLE=0 NCCL_GRAPH_REGISTER=0 CUDA_DEVICE_MAX_CONNECTIONS=8` **[M-ours]** — `.../E14A SLIM R2 run1 (hosen42, 2026-10-23)/bundle/e14/nccl_bench.json` (`arms.baseline.nccl`)
- `NCCL_P2P_LEVEL=SYS` switches the transport to `P2P/IPC` and costs 5.5x at 983 KB and 11.7x in 64 MiB busbw **[M-ours]** — same file. On a bare-metal-like 4-GPU PHB KVM box (Gen3 x8 5060 Ti) the same flag *helped* (−6 % at 1.31 MB, −34 % at 10 KB) **[M]** — [HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254). The difference is the host: GCP's virtualized P2P path is slow, which matches NCCL's warning that IO virtualization can redirect "all PCI point-to-point traffic to the CPU root complex" **[C]** — [NCCL GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html)
- `NCCL_SHM_USE_CUDA_MEMCPY=1`: on our box 983 KB eager 503 µs vs 382 µs baseline, and the CUDA-graph phase never finished within 240 s **[M-ours]**. Elsewhere: 1.9 → 12.5 GB/s at 20 MB but 10 KB latency 39 → 81 µs (2x 5070 Ti, no P2P) **[M]** — [SGLang PR #39605](https://github.com/sgl-project/sglang/pull/39605); 2x slower at 128 KB-1 MB on 8x A10, maintainer: designed for "tens of MB to GB" **[M+C]** — [NVIDIA/nccl #922](https://github.com/NVIDIA/nccl/issues/922)
- NCCL 2.29.2 release notes: built-in **symmetric kernels are enabled only on fully NVLink-connected systems** "because PCIe systems do not perform as well"; other 2.29 items are Blackwell 256M-512M AllReduce tuning and all-to-all channel defaults — nothing for small SHM all-reduce **[C]** — [NCCL 2.29.2 release notes](https://docs.nvidia.com/deeplearning/nccl/release-notes/rel_2-29-2.html)
- `NCCL_PROTO`: LL128 is not enabled on PCIe and forcing it "can corrupt data"; on SHM the 5070 Ti PR saw the same 1.9 GB/s "under every NCCL_PROTO"; NCCL picked LL up to 64 KB and Simple from 512 KB on Kaggle T4 SHM **[C/M]** — [NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html), [SGLang PR #39605](https://github.com/sgl-project/sglang/pull/39605), [nccl-allreduce-bench](https://github.com/sxwangsxwang1/nccl-allreduce-bench)
- `NCCL_ALGO=Tree` on SHM (2 ranks, T4): +14.5 % at large sizes, slower at 8 KB-1 MB (1 MB 3.39 vs 3.56 GB/s busbw) **[M]** — [nccl-allreduce-bench](https://github.com/sxwangsxwang1/nccl-allreduce-bench)
- Channels: `NCCL_MIN_CTAS`/`NCCL_MAX_CTAS` (the newer names for MIN/MAX_NCHANNELS); "excessive CTAs can hurt small-message performance" **[C]** — [NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html). `NCCL_MIN_NCHANNELS=8` described as a "significant bandwidth improvement" on RTX PRO 6000 PCIe, no numbers **[C]** — [rtx6kpro sglang.md](https://github.com/local-inference-lab/rtx6kpro/blob/master/inference-engines/sglang.md). `NCCL_MIN_CTAS=1 NCCL_MAX_CTAS=1` recovered a 2.20 regression on 4x A10G (weak, nvtop) **[M]** — [NVIDIA/nccl #1298](https://github.com/NVIDIA/nccl/issues/1298)
- `NCCL_SHM_DISABLE=1` falls back to sockets: α 14-16 → 60 µs, bandwidth −47 % **[M]** — [nccl-allreduce-bench](https://github.com/sxwangsxwang1/nccl-allreduce-bench)
- NCCL versions regress: 2.20.5 halved all-reduce bandwidth vs 2.18.5 on 8x A10G PHB **[M]** — [NVIDIA/nccl #1298](https://github.com/NVIDIA/nccl/issues/1298)

### Inferences
- [E] **Link-floor arithmetic for our message.** A ring all-reduce at n = 4 moves 2(n−1)/n = 1.5 x S out of and into each GPU. For S = 983,040 B that is 1.47 MB each way. At 12.4-13.2 GB/s GPU↔host that is a floor of about 115-120 µs, versus 344 µs measured. NCCL SHM therefore runs at about **35 % of the host-link floor**. That gap is the only real headroom left in pure NCCL tuning.
- [E] The likeliest NCCL knob to recover part of it is **more channels**: `NCCL_MIN_CTAS=4` or `8` (with `NCCL_MAX_CTAS` equal), and `NCCL_NTHREADS=512`. Each SHM channel is one CTA doing PCIe loads/stores into mapped host memory; two CTAs probably cannot saturate a x16 link in both directions. This is unmeasured on our box and could also lose (more CTAs = more SMs stolen from GEMMs, and more host-memory flows, which hurt on AMD hosts per NVIDIA/nccl #814). It is a 2-minute A/B in the existing `e14_nccl_bench.py` harness.
- [E] Second-tier knobs to run in the same sweep (each costs seconds): `NCCL_PROTO=Simple` vs `LL` (diagnostic only), `NCCL_ALGO=Ring` vs `Tree`, `NCCL_BUFFSIZE=1048576` / `8388608`, `NCCL_CUMEM_ENABLE=1` (our bench forced 0; engine default may differ). Do **not** use `NCCL_P2P_LEVEL`, `NCCL_SHM_USE_CUDA_MEMCPY`, or `NCCL_SHM_DISABLE` on this VM.
- [E] Even a perfect NCCL (115-120 µs) would cut communication from ~51 to ~17 ms per 116 ms step. So the realistic NCCL-only upside is bounded by about 1.4x step time; channel tuning will capture some fraction of that, not all.

### Gaps
- No published nccl-tests sweep of `NCCL_MIN_CTAS` for 4-rank SHM at 0.5-1 MB on any GPU.
- Why CE-memcpy SHM hung under graph capture on our box (NCCL 2.29.7) is unknown; no matching issue was found.
- Whether the guest's "Gen3 max" reading means the virtual root port really caps GPU↔host at Gen3 x16 (~13 GB/s, matching Spare Cores) is unconfirmed.

---

## Q2. Custom all-reduce without NVLink/P2P: vLLM custom AR, one-shot vs two-shot, QuickReduce, SGLang custom AR, FlashInfer PCIe-IPC, TRT-LLM, symm-mem

### Takeaway
Every in-tree custom all-reduce for CUDA needs P2P (CUDA IPC peer mappings) or NVLink multicast, so on this VM they are either gated off (TP4) or slower than NCCL (measured 0.39-0.68x). QuickReduce is ROCm-only. Symm-mem and FlashInfer trtllm AR are Hopper/Blackwell-gated. The only design that fits a "P2P exists but is slow" VM is a **host-staged all-reduce through pinned, mapped host memory**; the one public implementation (SGLang PR #39605) is TP2-only, unmerged, and does not cover our 0.5-1 MB graph-captured decode messages. A 4-rank, graph-capturable, host-staged two-shot kernel (optionally with INT8 payload) would have to be written by us.

### Cited Findings
- **vLLM custom AR**: disabled with a warning at world_size > 2 without full NVLink; at world_size 2 only `_can_p2p` gates it **[C, source description]** — [vLLM RFC #51513](https://github.com/vllm-project/vllm/issues/51513). On our box P2P passes the functional test, so at TP2 it would turn on and run over the slow P2P path; our VMS builder therefore always passes `--disable-custom-all-reduce` **[M-ours]** — `Experiment Notebooks/Build/build_vms.py` L131-134
- **vLLM dispatch order** (cuda_communicator): NCCL symm-mem → QuickReduce (ROCm) → FlashInfer PCIe IPC (`VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`) → FlashInfer AR → AITER → Custom → torch symm-mem (`VLLM_ALLREDUCE_USE_SYMM_MEM`) → PyNCCL **[C]** — [vLLM cuda_communicator](https://docs.vllm.ai/en/stable/api/vllm/distributed/device_communicators/cuda_communicator.html)
- **QuickReduce**: "supported only on the ROCm MI300 series"; on CUDA it is disabled because the quick-reduce library is missing; modes FP / INT8 / INT6 / INT4 with symmetric group-32 quantization, `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` (default NONE) **[C]** — [vLLM quick_all_reduce docs](https://docs.vllm.ai/en/latest/api/vllm/distributed/device_communicators/quick_all_reduce/), [vLLM PR #16804](https://github.com/vllm-project/vllm/pull/16804), [ROCm vLLM optimization](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html). No sign of a CUDA port was found.
- **Torch symm-mem in vLLM**: "Device capability 8.9 not supported, communicator is not available" on our L4s **[M-ours]**; its table lists only "9.0" and "10.0" with multicast **[C]** — [vLLM symm_mem docs](https://docs.vllm.ai/en/v0.16.0/api/vllm/distributed/device_communicators/symm_mem/)
- **FlashInfer AR / AR+RMSNorm fusion**: "FlashInfer All Reduce is disabled because it is not supported for world_size=4" (and 2) on our box **[M-ours]**; vLLM's `fuse_allreduce_rms` support matrix lists SM90 / SM100 only, needs TP > 1 and FlashInfer **[C]** — [vLLM fusion passes](https://docs.vllm.ai/en/stable/design/fusions/), [vllm.config.vllm](https://docs.vllm.ai/en/latest/api/vllm/config/vllm/)
- **FlashInfer PCIe-IPC** (`flashinfer.comm.PcieIpcAllReduceWorkspace`): bf16/fp16, fixed `max_numel`, `max_blocks` 128, an optional `profile` label "rootcplx" or "pcieswitch" that "does not pick a kernel", `tune()` picks among kernels; one workspace per stream; every rank must issue the same calls **[C]** — [FlashInfer docs](https://docs.flashinfer.ai/generated/flashinfer.comm.PcieIpcAllReduceWorkspace.html). SGLang authors measured TP4 DeepSeek-V4-Flash TPOT −10 % (bs1) / −5 % (bs4) on SM120 switch-free hosts **[M]** — [SGLang PR #34528](https://github.com/sgl-project/sglang/pull/34528). On our L4 VM: 0.39-0.68x **[M-ours]**.
- **SGLang custom AR**: V2 requires full NVLink; legacy V1 runs at world_size 2 (8 MiB cap) and is skipped at 4 PCIe GPUs ("not supported on more than two PCIe-only GPUs"); SGLang skips the real P2P test unless `--enable-p2p-check` **[C, source]** — see `tp2_replicas.md` (this repo) and [SGLang custom_all_reduce.py](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/distributed/device_communicators/custom_all_reduce.py)
- **SGLang host-staged all-reduce, PR #39605** ("host-staged all-reduce for two GPUs without peer access", author 05yuki): **still open** as of the snapshot; opened 2026-09-15, LL commit 09-25, race fix 09-26, retitled 10-17, cross-referenced by #46623 on 10-28; no approving review; CI listed as failing (may be stale) **[C]** — [SGLang PR #39605](https://github.com/sgl-project/sglang/pull/39605). Design and numbers **[M]** (same source):
  - `SGLANG_HOST_STAGED_ALLREDUCE=1`: for messages ≥ `SGLANG_HOST_STAGED_ALLREDUCE_MIN_BYTES` (default 1 MiB), **outside CUDA-graph capture**. Shared `/dev/shm` file, pinned mapped memory, 4 MB pieces on a D2H stream, `cuStreamWriteValue32` / `cuStreamWaitValue32` counters, double-buffered slots, no CPU blocking, sum = a + b on both ranks (bit-identical to NCCL).
  - `SGLANG_LL_ALLREDUCE=1`: bf16 ≤ `SGLANG_LL_ALLREDUCE_MAX_BYTES` (64 KiB), one kernel, NCCL LL format into host memory both GPUs map (8-byte stores = two bf16 + sequence number), spin on peer words, **CUDA-graph capturable**.
  - Latency: 20.97 MB 1,494 µs vs NCCL SHM 10,859 µs; **1.31 MB 142 µs**; 12.90 MB 946 µs; 62.9 MB 4,385 µs; LL 4096 bf16 6.5 µs vs NCCL 41.5 µs eager (22-25 µs in a graph).
  - End to end (TP2, 2x 5070 Ti, PCIe 4.0 x16, EPYC, no P2P): Qwen3.8-27B NVFP4 TTFT 3,308 tok 2.55 → 0.56 s, decode unchanged at 67 tok/s; Qwen3.5-35B-A3B decode with graphs 64.7 → 71.0 tok/s (+10 %).
  - Scope: "world_size == 2 only"; ">2 ranks" is listed under "Not this PR"; the author says the layout could extend but the summing logic would not.
- **lukealonso PCIe one-shot (SGLang fork)**: requires P2P ("without the config, custom AR runs about 15x slower than NCCL"); 4x RTX PRO 6000 TP4: 1 MB 95.6 µs vs NCCL 85.2 µs (NCCL wins at 1 MB) **[M]** — [rtx6kpro pcie-oneshot-allreduce](https://github.com/local-inference-lab/rtx6kpro/blob/master/optimization/pcie-oneshot-allreduce.md)
- **TensorRT-LLM**: `AllReduceStrategy` has NCCL, ONESHOT, TWOSHOT, AUTO and LOWPRECISION; the API page does not explain LOWPRECISION or its hardware requirements **[C]** — [TensorRT-LLM functional API](https://nvidia.github.io/TensorRT-LLM/python-api/tensorrt_llm.functional.html). A release note warns that low-precision fused residual-norm + all-reduce can lose "a severe amount of accuracy" on some low-precision MoE models **[C]** — [TensorRT-LLM releases](https://github.com/NVIDIA/TensorRT-LLM/releases). No PCIe/L40S/L4 benchmark of these strategies was found.
- **Quantized all-reduce research**: Flash Communication (arXiv 2412.04964) — 4x L40 PCIe TP4 and A100: two-step all-reduce (quantize → All2All → dequant + ReduceSum → requantize → All-Gather), INT8/INT6/INT4 with group 128; "more than 3x" intra-node communication speed-up, INT4 up to 3.18x kernel latency at > 64 MB; TTFT up to 2.06x on LLaMA-3-8B L40 TP4; **decode/TPOT not evaluated**; accuracy LLaMA-3-70B zero-shot FP16 80.00 / INT8 80.00 / INT4 78.29, LLaMA-3-8B 72.76 / 72.53 / 71.11; C4 ppl 8B INT8 8.89 vs INT4 9.68; **uses GPU peer DMA (P2P)**, fused custom kernel, no code link **[M]** — [arXiv 2412.04964](https://arxiv.org/html/2412.04964). FlashCommunication V2 (arXiv 2508.03760): any-bit (down to 2-bit) communication, up to 3.2x AllReduce **[M, abstract only]** — [arXiv 2508.03760](https://arxiv.org/abs/2508.03760)

### Inferences
- [E] **Why PCIe-IPC lost on our box:** PCIe-IPC reads/writes peer memory over the same P2P path that NCCL P2P/IPC measured at 0.42-0.78 GB/s busbw. Its gain on SM120 bare metal comes from fast P2P, which this VM does not have. The loss is structural, not a tuning problem.
- [E] **Host-staged two-shot, 4 ranks, graph-capturable — the one custom design that fits.** Per GPU for S = 983 KB: write own S to host (≈ 75 µs at 13.2 GB/s), read 3 peer shards of S/4 (737 KB, ≈ 60 µs), reduce, write S/4 (≈ 20 µs), read 3 reduced shards (≈ 60 µs), plus 2 flag syncs (LL-style, a few µs each). Total ≈ **220-250 µs vs 344 µs (−30 %)**, if four GPUs writing/reading host DRAM concurrently do not saturate host memory (≈ 50 GB/s aggregate, within one Cascade Lake socket's DRAM bandwidth, but unverified). Saving ≈ 140 x 0.1-0.12 ms ≈ **14-17 ms per 116 ms step (about +14-17 % tok/s)**.
- [E] **Add INT8 group-quantized payload** (Flash Communication shows INT8 ≈ lossless on LLaMA-3 zero-shot; QuickReduce uses INT8 group-32 on ROCm) and the bytes halve, so ≈ 130-150 µs per call, saving ≈ 28-30 ms per step (+30-35 % tok/s). Accuracy risk is real for a 64-layer reasoning model generating 60k+ tokens: residual-stream activations have outliers; it must pass our greedy-parity / logprob gate, and non-bit-exact reduction changes outputs vs NCCL. Treat INT8 as a measured arm, not a default.
- [E] Implementation path (our code, nobody has published it): start from PR #39605's LL kernel (graph-capturable, host-mapped, sequence-numbered 8-byte stores) and generalise it to 4 ranks with a reduce-scatter + all-gather layout; register it as a custom op in SGLang `parallel_state` (the slot where `pcie_ipc_ar` is dispatched before NCCL) or in vLLM's `cuda_communicator` dispatch list. LL format halves effective bandwidth (half of each 8-byte store is the flag), so for 1 MB messages a Simple-style "data then flag" protocol per chunk is better.
- [E] A cheaper experiment first: NCCL channels (Q1). If more CTAs bring NCCL close to the 120 µs floor, the custom kernel is not needed.

### Gaps
- No public 4-rank host-staged all-reduce for CUDA (graph-capturable or not) was found.
- Accuracy of INT8/INT4 all-reduce on Qwen3.x hybrid models or on long reasoning traces: no data.
- TRT-LLM LOWPRECISION strategy: hardware requirements and PCIe results not found; TRT-LLM support for Qwen3.5/3.8 hybrid GDN on sm_89 not researched here.
- Host DRAM bandwidth and contention on Kaggle's g2 VM when four GPUs stream concurrently: unmeasured.

---

## Q3. Pipeline parallel vs tensor parallel vs TP2xPP2 vs data-parallel replicas for decode on PCIe; engine support for hybrid Qwen3.5/3.8; measured tok/s

### Takeaway
DP replicas are ruled out by KV (measured 0.57x at 40k). Pipeline parallelism is the only layout that **keeps TP4's KV and weight footprint per GPU while removing most per-layer all-reduces**, but: SGLang's DFlash requires `pp_size == 1`; vLLM gained MTP-under-PP only in the V2 model runner (PR #46994, merged 2026-09-11; first release unverified); no one has published batched PP decode numbers on PCIe for a 27B hybrid; the only Qwen3.8-27B PP number is single-stream and PP loses there (32.9 vs TP2 48.9 tok/s). PP4 also re-reads weights once per micro-batch and puts the 248k-vocab lm_head + MTP head on the last stage, so its upside is uncertain. TP2xPP2 is the more balanced candidate on paper.

### Cited Findings
- **Our measurements:** 2xTP2 SGLang DFlash W6: 658 tok/s at 8k (1.15x), 308 at 40k (0.57x), 152k KV tokens per replica; 2xTP2 MTP ReplaySSM fp32: 511 (0.89x) / 279 (0.51x); vLLM 2xTP2 FP8/INT4 MTP3: 191/483 at 8k, 22/92 at 40k (2 streams) **[M-ours]** — `EMS_SAFE_RISKY_VMS_rev2g_GPU_results_2026-10-26.md`
- **Qwen3.8-27B on 2x RTX 4090 48 GB, vLLM 0.26.0, single stream, 29k-token prompt:** TP1 34.4, **TP2 48.9, PP2 32.9 tok/s**; Llama-3.3-70B short context TP2 35.3 vs PP2 21.8; quant, custom AR, NCCL env and graph mode not reported; multi-stream not tested **[M]** — [Houtini dual-4090 runbook](https://houtini.com/articles/the-dual-4090-96gb-vllm-benchmark-runbook/)
- **SGLang DFlash:** "Currently DFLASH speculative decoding only supports pp_size == 1" (`speculative_hook.py` L280-283) **[C, source]** — `research_notes/৪xL4 ইঞ্জিন বটলনেক যাচাই/tp2_replicas.md`
- **vLLM MTP under PP:**
  - RFC #44697: MTP under PP > 1 crashes or "silently diverges"; names Qwen3.5 + PP; for hybrid GDN/Mamba models `num_accepted_tokens` (drives conv/SSM state rollback) is set on the last rank only **[C]** — [vLLM #44697](https://github.com/vllm-project/vllm/issues/44697)
  - PR #44698 (V1 runner fix): **closed unmerged 2026-10-14**; on 2x 16 GB GPUs PP=2, Qwen3.5-27B-AWQ (hybrid GDN) MTP k=1 1.89x vs no-spec (offload-bound, ratio only), MiMo 1.68x (38.16 vs 22.75 tok/s); greedy 5/5 identical at k = 1-3 including `mamba_cache_mode=align` and fp8 KV; V2 runner deadlocked for MiMo **[M]** — [vLLM PR #44698](https://github.com/vllm-project/vllm/pull/44698)
  - **PR #46994 ([Spec][V2] MTP under PP): merged 2026-09-11 (commit 6fe67cb)**; tested with `VLLM_USE_V2_MODEL_RUNNER=1`; depends on #55745 (`record_stream` guard) under async launches; draft runs on the last PP stage only; example `--tensor-parallel-size 4 --pipeline-parallel-size 2` with MTP k=1; acceptance at PP=2 matches PP=1 (Qwen3.6-35B-A3B-AWQ K=1 1.9375 at PP1 vs 1.9354-1.9380 at PP2); Qwen3.5-35B-A3B-GPTQ-Int4 PP2 TP1 mean accept 1.95 / 2.82 / 3.59 at K = 1/2/3; **no tok/s numbers**; CUDA graph mode not stated; release containing it unverified (third-party commits claim 0.30.0 or 0.32.0, conflicting) **[M/C]** — [vLLM PR #46994](https://github.com/vllm-project/vllm/pull/46994)
- **Hybrid GDN + MTP scheduling collapse (vLLM, any parallelism):** Qwen3.8-27B GPTQ INT4 on an RTX 4090 D (sm_89), main 2026-09-06: MTP k=1 schedules only ~3 sequences per step at bs ≥ 4 (bs 8: no-spec 367.1 vs MTP 197.0 tok/s; bs 20: 649.0 vs 210.9); hypothesis: Mamba block accounting reserves 1+k blocks per group per request; open, diagnostic PR #55617 **[M]** — [vLLM #55533](https://github.com/vllm-project/vllm/issues/55533). Our vLLM 0.32 runs reached a real 16-24 running, so this did not bind us at our pool sizes **[M-ours]** (results file).
- **Generic PP-on-PCIe guidance** (secondary, unverified): one guide claims 2-GPU PCIe TP scaling caps at ~1.5x and PP reaches 1.7-1.8x, unclear whether decode or batched throughput **[C]** — [GigaGPU TP vs PP](https://gigagpu.com/tensor-vs-pipeline-parallelism/); an older 2x 3090 test: at batch 8 TP 52 vs PP 48 tok/s **[M, thin method]** — (search summary; [databasemart guide](https://www.databasemart.com/blog/vllm-distributed-inference-optimization-guide))
- **Published 2xTP2 wins** are all short-context with KV to spare (1.3-2.4x) — see `tp2_replicas.md`.

### Inferences
- [E] **Per-GPU memory:** PP4 and TP2xPP2 hold the same weight bytes per GPU as TP4 (W/4) and the same KV/GDN-state bytes per token per GPU (TP4: 16 full-attn layers x 1/4 of KV heads; PP4: 4 layers x all heads). So **PP keeps our ~880k-token KV pool**, unlike 2xTP2. Exceptions: the embedding sits on stage 0 and the lm_head (248k x 5120 bf16 ≈ 2.5 GB) + MTP head sit on the last stage unsharded, which unbalances memory and time on that GPU.
- [E] **Communication per step (verify batch 96 tokens, S = 0.98 MB):** TP4 ≈ 140 all-reduces x 1.5 S per GPU ≈ 206 MB moved per GPU per step (≈ 51 ms measured). TP2xPP2 with 2 micro-batches: each GPU all-reduces only its 32 layers for half the tokens, 2-rank ring moves 1.0 x S/2 → 2 x 64 x 0.49 MB ≈ 63 MB (≈ 3.3x less), plus 2 stage-to-stage sends of ~0.5 MB. PP4: just 3 sends of ~1 MB per micro-batch.
- [E] **The PP cost is weight re-reads and bubbles.** With m micro-batches each stage streams its weights m times per round. TP4 Marlin GEMM time is 25 ms at M = 96; at M = 24-48 Marlin on L4 is close to weight-bandwidth-bound, so PP4 (m = 4) could spend ~60 ms on GEMMs per round and TP2xPP2 (m = 2) ~35-45 ms, vs 25 ms. Rough round time: TP4 116 ms; TP2xPP2 ≈ 116 − 51 (TP4 AR) + ~15-20 (2-rank AR over SHM) + ~15-20 (extra weight reads) ≈ **95-105 ms (≈ 1.1-1.2x)**; PP4 ≈ 116 − 59 + ~35 (weight re-reads) + last-stage lm_head/MTP imbalance ≈ **95-115 ms (≈ 1.0-1.2x)**. Both are paper estimates with wide error; the measured single-stream PP2 loss (Houtini) shows PP gives nothing without enough concurrent micro-batches.
- [E] **Engine reality:** our fastest stack (SGLang DFlash) cannot run PP at all. On vLLM, PP + MTP exists only via the V2 runner (#46994) and has never been timed on PCIe or with hybrid 27B; `num_accepted_tokens` handling for GDN rollback under PP must be re-verified with our greedy gate. The 2-rank all-reduce inside TP2xPP2 must stay on NCCL SHM (`--disable-custom-all-reduce`; P2P is slow); PR #39605's host-staged paths are TP2-only, which fits TP2xPP2, but its graph-capturable LL path stops at 64 KiB and the bulk path is ≥ 1 MiB and not graph-capturable, so our ~0.5 MB decode messages fall between them.
- [E] The vLLM 4-way layout to try, if anything: `--tensor-parallel-size 2 --pipeline-parallel-size 2 --disable-custom-all-reduce` with `VLLM_USE_V2_MODEL_RUNNER=1`, MTP k=3, fp8 KV, 16-24 streams, greedy gate vs TP4. Verify the release includes #46994 and #55745 first (`pip show vllm`, grep the runner source for the PP + MTP path).

### Gaps
- No batched decode measurement of PP or TP2xPP2 vs TP4 on PCIe GPUs for any 27-35B model at ≥ 8 streams and ≥ 32k context.
- SGLang PP support for Qwen3.5/3.8 hybrid with MTP/EAGLE (not DFlash) was not checked.
- Whether vLLM 0.32.0 contains #46994 and #55745 is unverified.
- Whether vLLM PP can use FULL CUDA graphs with spec decode is not documented.

---

## Q4. CUDA graphs: FULL vs PIECEWISE with speculative decoding; which attention backend allows FULL graphs with MTP on Ada; overhead numbers

### Takeaway
vLLM's FlashInfer backend only supports full graphs for single-token decode, so MTP verify batches (query length 1 + k) fall back to PIECEWISE — we see this in our logs. Backends that declare `UNIFORM_BATCH` or `ALWAYS` (FlashAttention 2, Triton attention) would allow `FULL_AND_PIECEWISE` with spec decode on Ada; with fp8 KV on sm_89 the practical candidate is `TRITON_ATTN`. The hybrid model takes the minimum capability across its backends, so the GDN backend's level also matters (not documented). The expected gain is small: in PIECEWISE mode the all-reduces are already inside captured pieces, and the one published spec-decode graph study recovered only ~1.3 % end to end.

### Cited Findings
- Our vLLM 0.32 log: "FULL_AND_PIECEWISE is not supported with spec-decode for attention backend FlashInferBackend (support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE); setting cudagraph_mode=PIECEWISE" **[M-ours]** — `_scratch/ems_out/vms/` logs (`vms_flags(... backend="FLASHINFER")` in `build_vms.py`)
- vLLM CUDA-graph support table **[C]** — [vLLM CUDA Graphs design](https://docs.vllm.ai/en/stable/design/cuda_graphs/):
  - FlashAttention v2: `UNIFORM_BATCH` ("actually ALWAYS, but falls back to FULL_AND_PIECEWISE for performance");
  - FlashAttention v3: `ALWAYS` (Hopper only, so not for L4);
  - Triton attention: `ALWAYS` (FULL_AND_PIECEWISE preferred);
  - FlashInfer: `UNIFORM_SINGLE_TOKEN_DECODE` (UNIFORM_BATCH only with TRTLLM attention on Blackwell);
  - Mamba attention: `UNIFORM_SINGLE_TOKEN_DECODE`; GDN is not listed;
  - `UNIFORM_BATCH` "can cover speculative decoding, where decodes use 1 + num_speculative_tokens";
  - for hybrid models "the model uses the minimum capability across all backends"; modes are downgraded to the closest supported one;
  - modes: `NONE`, `PIECEWISE`, `FULL`, `FULL_DECODE_ONLY`, `FULL_AND_PIECEWISE` (default with piecewise compilation); set via `--compilation-config '{"cudagraph_mode": "..."}'`.
- Known bugs with MTP and full graphs: MTP k=1 + `FULL_AND_PIECEWISE` deadlock at a bonus-token-only batched-decode shape on v0.20.1, not with `FULL_DECODE_ONLY` **[M]** — [vLLM #42271](https://github.com/vllm-project/vllm/issues/42271); silent output corruption with MTP + DCP + `FULL_AND_PIECEWISE` when the backend lacks varlen decode under DCP (PIECEWISE or eager avoids it) **[M]** — [vLLM #45425](https://github.com/vllm-project/vllm/issues/45425)
- Drafter graphs: on GB10 (sm_121), Gemma-4 26B-A4B NVFP4 + MTP γ=3, vLLM 0.22.1.dev: the drafter chain ran PIECEWISE at every shape and was 18 % (conc 1) / 15.5 % (conc 16) of the step; target verify ran as a FULL graph (79.5 ms of 100.6 ms at conc 16). An out-of-tree FULL-graph prototype saved only **~1.3 ms of an 18 ms chain at conc 16 (~1.3 % e2e), ~0 at batch 1**; the author retracted the earlier +11 % estimate ("kernel-bound"). Issue closed as stale 2026-10-19; related Eagle PR #34880 still open as of June 2026 **[M]** — [vLLM #45258](https://github.com/vllm-project/vllm/issues/45258)
- vLLM's FA2 full-graph refactor (separate attention routines, FA2 + FlashInfer full-graph support) **[C]** — [vLLM PR #20059](https://github.com/vllm-project/vllm/pull/20059)
- SGLang captures the whole decode/verify step (target_verify graph took 372-385 s to capture on Kaggle cold start) **[M-ours]** — `tp2_replicas.md` "Other costs"

### Inferences
- [E] On vLLM with fp8 KV on sm_89, `--attention-backend TRITON_ATTN` is the candidate for FULL verify graphs (FA2 in vLLM does not take fp8 KV on Ada, as far as I know; unverified). The GDN backend's declared support decides whether the hybrid gets full graphs; if GDN is `UNIFORM_SINGLE_TOKEN_DECODE` like Mamba, the hybrid stays PIECEWISE regardless. Check the startup log line, which names the limiting backend.
- [E] Size of the prize: in PIECEWISE mode only the attention ops (16 full-attention layers here, plus GDN if it is a splitting op) run eagerly between graph pieces. At 116 ms steps dominated by GEMM + all-reduce, launch overhead is a few ms at most. Expect **≤ 3-5 %**, consistent with the #45258 result. Not a path to 900 tok/s, but cheap to A/B: `--attention-backend TRITON_ATTN --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE"}'` vs the current FlashInfer PIECEWISE, with the greedy gate (watch for the #42271 deadlock; `FULL_DECODE_ONLY` is the fallback).
- [E] SGLang already runs whole-step graphs with DFlash, so this lever only applies to the vLLM path.

### Gaps
- vLLM's `AttentionCGSupport` for the GDN (`gdn_attn`) metadata builder in 0.32 is not documented; needs a source read.
- No measured FULL vs PIECEWISE comparison for MTP on Ada, or for Qwen3.5/3.8 hybrids anywhere.
- Triton attention vs FlashInfer decode-kernel speed for fp8 KV at 40-90k context on L4 is unmeasured (a slower kernel could cancel the graph gain).

---

## Q5. How much of decode step time is all-reduce on 4x L4 / 4x 4090 / 4x L40S PCIe at batch 16-64

### Takeaway
Our own profile is the best evidence available anywhere: **44 % all-reduce + 7 % all-gather = 51 % of a 116 ms TP4 verify step** on Kaggle 4x L4 (SGLang, 96 verify tokens). No published profile for 4x 4090 or 4x L40S PCIe TP4 decode was found; published shares are 12 % on H200 NVLink and "up to 20 %" in the TokenWeave paper (NVLink). Communication cost scales almost linearly with verify tokens per step, so rejected draft tokens cost all-reduce time too.

### Cited Findings
- Kaggle 4x L4, SGLang 0.5.21 TP4 INT4 + DFlash2, 12 x 8 verify tokens: all-reduce 50.9 ms (140 calls, 364 µs each), all-gather 8.1 ms (3 calls), kernel sum 116.1 ms **[M-ours]** — `Notes/30 Claude - E14A SLIM R2 run results (2026-10-24).md`; bench 343.9 µs per 983 KB call in a graph **[M-ours]** — `nccl_bench.json`
- 4x RTX 5060 Ti (Gen3 x8, KVM, SHM, CUDA graph, hidden 5120): 64 tokens 267.6 µs, 128 tokens 498.1 µs per all-reduce, i.e. nearly linear in bytes **[M]** — [HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254)
- H200 NVLink: all-reduce "about 12 % of decode time" **[M]** — [SGLang PR #30364](https://github.com/sgl-project/sglang/pull/30364); TokenWeave: TP inference "continues to pay up to 20 % communication costs" (NVLink-class systems) **[C]** — [TokenWeave arXiv 2505.11329](https://arxiv.org/html/2505.11329v5)
- A5000 PCIe study: for INT4 decode the fixed per-collective overhead can dominate, so a bandwidth gap need not show in runtime **[C, abstract]** — [arXiv 2607.11368](https://arxiv.org/pdf/2607.11368)
- 2x 5070 Ti, no P2P: a 2048-token prefill chunk of a 64-layer model spends 733 ms of an 802 ms step in 128 NCCL SHM all-reduces of 20 MB **[M]** — [SGLang PR #39605](https://github.com/sgl-project/sglang/pull/39605) (prefill, not decode, but shows SHM's bandwidth wall)

### Inferences
- [E] **vLLM MTP-3 at 16 streams** verifies 16 x 4 = 64 tokens → 655 KB per call. Scaling our 344 µs by bytes (with ~20-30 µs fixed cost) gives ≈ 240 µs x 140 ≈ **34 ms per step** of all-reduce, plus the logits all-gather. DFlash W6 at 16-20 streams verifies 112-140 tokens → 1.1-1.4 MB per call ≈ 55-70 ms per step. Communication per *accepted* token = (140 x per-call time) / (accepted tokens per step), so a wider draft with low acceptance pays all-reduce on wasted rows.
- [E] **The 8.1 ms all-gather is a separate, cheaper lever.** It is 3 calls of 11.9 MB per rank — consistent with gathering vocab-parallel logits (≈ 96 rows x 248k / 4 x fp32 ≈ 9.5 MB, plus drafter logits). Greedy verify only needs per-shard (max value, index) pairs, and temperature sampling can be done exactly per shard with the Gumbel-max trick, then all-gather a few bytes per row. That would remove most of the 8 ms (≈ 7 % of the step). Needs an engine patch; no upstream feature found in this pass.
- [E] Putting Q1-Q5 together, the communication share is large enough (51 %) that the comm levers ranked by expected value are: (1) host-staged 4-rank two-shot kernel, with an INT8 arm (−14 to −30 ms/step); (2) NCCL channel sweep (unknown, 0 to −20 ms); (3) sharded sampling instead of logits all-gather (−5 to −7 ms); (4) layout change to TP2xPP2 on vLLM (uncertain, ±; large engineering and correctness risk).

### Gaps
- No public nsys profile of TP4 decode on 4x 4090 / 4x L40S / 4x L4 PCIe from anyone else.
- The exact content of the 3 all-gathers in SGLang's DFlash verify step was not traced to code.
- No profile of our vLLM 0.32 MTP path exists; the 34 ms figure is an estimate.

---

## Q6. Overlapping communication with compute (async TP, sequence parallel, micro-batch overlap) on Ada

### Takeaway
None of the off-the-shelf overlap features run on our setup: vLLM's AR+RMSNorm fusion and async TP are Hopper/Blackwell-oriented (symm-mem/multicast, FlashInfer trtllm), TokenWeave's fused kernel needs NVSHARP/multimem, and SGLang/vLLM two-batch overlap targets MoE expert parallelism. Overlap on Ada PCIe would need a custom two-micro-batch schedule (NanoFlow-style). It is physically plausible because NCCL SHM uses only 2 CTAs plus the PCIe link while GEMMs use SMs and DRAM, but splitting the batch doubles weight reads, so the net gain is limited.

### Cited Findings
- vLLM fusion passes: `fuse_allreduce_rms` supported on SM90 (FP16/BF16, FP8 static) and SM100 (+NVFP4), on by default at O2 on Hopper/Blackwell with TP > 1; sequence parallelism (`enable_sp`, formerly `enable_sequence_parallelism`) is off by default and is a prerequisite for async TP (`fuse_gemm_comms`, formerly `enable_async_tp`) **[C]** — [vLLM fusion passes](https://docs.vllm.ai/en/stable/design/fusions/), [vLLM RFC #27995](https://github.com/vllm-project/vllm/issues/27995). `fuse_allreduce_rms` breaks determinism under TP > 1 with `VLLM_BATCH_INVARIANT=1` (H100) **[M]** — [vLLM #51290](https://github.com/vllm-project/vllm/issues/51290)
- On our L4s vLLM logs "FlashInfer All Reduce is disabled because it is not supported for world_size=4" and symm-mem "Device capability 8.9 not supported" **[M-ours]** — `_scratch/ems_out/vms/` logs
- TokenWeave: fused AllReduce-RMSNorm plus at most two-way, GPU-wave-aware token splits so one split's communication overlaps the other's compute, on separate compute/communication streams; the fused kernel relies on NVSHARP/Multimem (Hopper/Blackwell); for very small batches only the fused kernel is enabled **[C]** — [TokenWeave arXiv 2505.11329](https://arxiv.org/html/2505.11329v5), [MLSys 2026 paper](https://proceedings.mlsys.org/paper_files/paper/2026/file/73ba81c7b25134a559c8a9c39ec1a4c3-Paper-Conference.pdf)
- NanoFlow: splits batches into nano-batches within one GPU so a GEMM of one overlaps attention or a collective of another; requires framework-level changes **[C]** — [NanoFlow summary](https://www.abhik.ai/papers/nanoflow), [TokenWeave](https://arxiv.org/html/2505.11329v5)
- Two-batch overlap in DynaFlow's comparisons is the MoE case (attention on the combined batch, MoE split across two) **[C]** — [DynaFlow arXiv 2605.21603](https://arxiv.org/pdf/2605.21603)
- CPU starvation can stall communication and launches in multi-GPU inference even with CUDA graphs **[C, abstract]** — [arXiv 2603.22774](https://arxiv.org/abs/2603.22774)

### Inferences
- [E] **DIY two-micro-batch overlap at TP4:** split the 96-row verify batch into two halves on two streams; while half A all-reduces layer i, half B runs layer i's GEMMs. Communication is bandwidth-bound, so each half's all-reduce takes ≈ half the time; ideal hiding is up to ~50 ms per step. Costs: Marlin GEMMs at M = 48 re-read the weights twice (≈ +15-25 ms), SM contention with NCCL's CTAs, and a non-trivial engine change (both SGLang and vLLM capture the whole step as one graph on one stream). Net estimate **−10 to −30 ms per 116 ms step**, with high implementation risk. Combines poorly with PP (which already micro-batches).
- [E] A cheaper partial overlap is possible inside one layer: the attention/GDN output all-reduce can overlap the next layer-norm-independent work only if the model has parallel branches; Qwen3.8 is sequential (residual → norm → mixer → AR → norm → MLP → AR), so there is nothing to overlap without splitting the batch.
- [E] Ranking: a faster all-reduce (Q2) or fewer bytes beats overlap on effort and risk; overlap is a last resort.

### Gaps
- No public overlap implementation for dense TP decode on Ada/PCIe was found.
- Whether NCCL SHM kernels and Marlin GEMMs actually run concurrently on L4 (SM and PCIe/DRAM contention) is unmeasured.
