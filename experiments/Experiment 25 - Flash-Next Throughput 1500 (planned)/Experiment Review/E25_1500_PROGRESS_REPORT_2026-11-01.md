# Flash-Next 1500 tok/s: progress report (2026-11-01). The baseline for the next research round is NB-K100.

## 0. Setup
- **Hardware and engine:** Kaggle 4×L4 (24 GB each, sm_89, PCIe, P2P off, NCCL over SHM). Merlin v3.0.0 (a vLLM 0.32.0 fork, commit ad42e37).
- **Topology:** TP2 × PP2 + EP, partition 25 / 23, V2 runner, async scheduling, FULL_AND_PIECEWISE graphs, RecoverSSM.
- **Model:** Qwen3.8-Flash-Next: 48 layers (36 GDN + 12 QSA), 512 experts top-10, hyper-connections hc 4, PLE pinned on the host as FP8.
- **Two checkpoint families:**
  - **W4:** Merlin W4A16, pool 214 blocks at a 3.55 GB KV pin.
  - **klee100:** 3bpw-MTP; routed experts W3 g128 / W2 g64 on humming kernels, W4 MTP; pool 479 blocks at a 7.93 GB pin, F 6.
- **KV:** FP8 E4M3 QSA KV with static scales. Block 4096 tokens.
- **Speculation:** MTP K3, probabilistic.
- **Workload:** the Duck agentic harness on ARC-AGI-2 (multi-turn tool use, xhigh reasoning). Attempts run up to 104k generated tokens with max-model-len 131,072; contexts are long (mean ~33k, p90 ~79k).
- **Target:** whole-run average ≥ 1500 tok/s.

## 1. Baseline for research: NB-K100 (klee100 + Q-defer)
**K100Q, measured on GPU (30 real ARC tasks, real Duck, 21 workers):**
- window-average **385 tok/s**; median 414 with all workers busy;
- 0 preemptions (KV max 99.4 %);
- prefix hit **85.4 %** over the whole run;
- MTP acceptance length 2.8;
- quality 23 / 30 tasks, task-norm 80 %.

**Synthetic (K3_ONLY v2, without Q-defer):** 460 tok/s at 20k × 16 streams, 562 at 20k × 24, 445 at 40k × 16, 431 at 60k × 16. Tick 55.8 ms at 16 streams and 68.7 ms at 24.

**Running now:** NB-K100 EVAL120 6 h (`Q27-XHIGH-E28_K100_EVAL120_6H.ipynb`).
- klee100 + Q-defer + INT8 draft head (I8, activation gated) + 24 workers;
- slots admission, T1 104k / B2 120k;
- the A/B/C attempt scheduler.

**So 1500 is about 3.9× the measured real-Duck window average, or about 2.7× the best synthetic.**

## 2. Levers: measured and adopted
| Lever | Measured gain | Where |
|---|---|---|
| Q-defer: deferred PP receive of the sampled tokens; removes the stage-0 in-graph stall (19.8 → 0.035 ms) | +21.1 % at 8k, +19.8 % at 40k; oracle EXACT (NB4S) | NB-K100, NB-W4 |
| More streams | W4 8 → 12 streams: +28 % / +30 % at 8k / 40k (NB4T). klee100 16 → 24 at 20k: +22 % | NB-W4 C12, NB-K100 24 |
| INT8 draft head at K3 | +3.4 % at 8k (noisy), +11.4 % at 40k; acceptance unchanged (NB4T) | both |
| Pin +1.0 GiB on W4 | 149 → 214 blocks | W4 |
| FP8 QSA KV | 1.9× the pool; ARC quality 13 / 15 = BF16 13 / 15 (NB5) | both |
| `--language-model-only`, vLLM #60490 backport | free memory on PP0 / safety | both |
| Prefix caching (align, unit 64) | 85.4 % reuse | both |
| Page-cache drop before load | load 873 → 239-254 s | both |

## 3. Levers: measured negative or no gain (do NOT re-propose without new evidence)
| Lever | Result |
|---|---|
| MTP K2 (+ INT8 head) | −7 % at 8 streams (the third draft position is lost) |
| PP partition 26 / 22 | −2.5 % at 8k, +4.9 % at 40k, OOM at the base pin |
| SSM state bf16 | 0 capacity (page padding) and an oracle FAIL |
| Marlin MoE config tuning | best = base within 0.4-1.1 % |
| sm_89 CuTe GEMM plans | do not compile on sm_89 |
| P2P on, PCIe IPC all-reduce, NCCL CTA tuning, `NCCL_SHM_USE_CUDA_MEMCPY` | slower / hangs |
| TP4, DP, PP4 | lose pool / crash |
| Prefix caching off with MTP | wrong outputs / collapse |
| KV offload of live streams; admitting beyond residency; time-slicing | collapse |
| `--enable-hc-sp` | PP = 1 only |
| SGLang EP4 for Flash-Next; NVFP4 KV on QSA | wrong outputs / unsupported |
| `MERLIN_SHARE_DRAFT_WORKSPACE` | 0 B effect |

## 4. Not yet measured: candidates
- **DFlash2 drafter** (`incoai/Qwen3.8-Flash-Next-DFlash2`), being measured now on W4 (Exp 27 NBDF).
  - The seam, the PP2 aux relay and the FULL-graph fix (vLLM #63587 / PR #64059) are ported.
  - BF16 drafter KV at block 4096 collapses the pool to 71 blocks. A drafter SW block of 1024 gives 205 blocks, and is the primary arm.
  - Estimated ×1.24-1.34 vs MTP K3.
- **Stage-1 compute.** Stage 1 binds after Q-defer (33 ms vs 21.6 ms on stage 0). Candidates:
  - sharded sampling (no full-vocab AllGather, 2.9 ms per tick);
  - graphing the eager post-target kernels (~4.2 ms);
  - moving the MTP / LM head off stage 1.
- **Fixed blocks per request** (F = 6, about 25-38 % of the pool); Mamba page padding 21 %.
- **Merlin runtime switches:** `VLLM_GDN_DECODE_FASTPATH`, `VLLM_GDN_SHARED_ALIGN_INDEX`, `VLLM_TARGET_SEQ_LENS_ASYNC`, `VLLM_E2_GUARD_MODE`, local argmax reduction.
- **Humming sm_89 tile tuning**; W2 g64 never benchmarked. humming is 1.06-1.20 × Marlin on the same W4 weights, worst at T 16.
- **Re-prefill of the previous turn's generated tokens.** This is 58 % of uncached tokens, because Mamba checkpoints are published only at prompt positions under RecoverSSM + PP. Estimated +4.6 %.
- **Prefill chunk 2048 / long-prefill threshold** (1024 budget now; a prefill step is about 233 ms).
- **Boot caches:** FlashInfer JIT 108 s, compile cache, prebuilt venv.
- **NCCL LL128** (dropped for corruption risk); `NCCL_CUMEM_ENABLE=0`; MoE aux-stream / expandable_segments.
- **MTP-layer KV scale calibration.** FP8 lowers MTP acceptance: AL 2.833 vs 2.957 BF16; the MTP layer uses the default scale 1.0.
- **klee100 above 24 streams** (resident bound 29 at 40k); a KV gate on running blocks.

## 5. Measured profile facts (for sizing)
- W4 tick at 8 streams ~41-48 ms. Stage 1 (after Q-defer) ~33 ms: target 20.2 ms, MTP drafts ~10.5 ms (7.5 ms of it three lm_head reads), AllGather 2.9 ms, AllReduce 3.2 ms.
- MoE per layer: 172 µs at 4 tokens, 645 µs at 28.
- Per-stream rate: 25 tok/s at 9 streams (W4); 23.4 at 24 streams (klee100).
- Synthetic sweeps overstate real Duck tok/s by about 12 % (acceptance).
