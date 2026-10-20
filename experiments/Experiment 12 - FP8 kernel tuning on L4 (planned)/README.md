# Experiment 12: FP8 kernel tuning on L4 (planned, 2026-10-16)

**Goal:** more decode speed for the same Qwen3.8-27B FP8 weights on 4×L4, through kernel choice only.

## Finding (R5 log, verified in the vLLM 0.19.0 source)

All 5 main block-FP8 GEMM shapes of our model run vLLM's Triton `w8a8` block-FP8 kernel with the **default config**. The R5 log warns "Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!" for these TP4 per-rank (N, K) shapes:

| N | K |
|---|---|
| 8704 | 5120 |
| 5120 | 4352 |
| 5120 | 1536 |
| 4096 | 5120 |
| 3584 | 5120 |

What the source shows:
- **Default config** (`fp8_utils.py`): BLOCK_M 64, N 128, K 128, GROUP_M 32, 4 warps, 2 stages.
- **Config lookup:** configs are read only from `vllm/model_executor/layers/quantization/utils/configs/N=…,K=…,device_name=NVIDIA_L4,dtype=fp8_w8a8,block_shape=[128,128].json`, through an `lru_cache` per process. In 0.19 no env var redirects this folder for block FP8, so the files must be copied there before the engine starts.
- **FP8 Marlin:** first in vLLM's CUDA FP8 kernel list, but disabled on compute capability ≥ 8.9 unless `VLLM_TEST_FORCE_FP8_MARLIN=1`. It supports block-128 weights (turned into group-128); activations stay BF16 (weight-only FP8).
- **Compile cache:** that env var is part of vLLM's torch.compile cache key (it is not in `ignored_factors`), so a Marlin run cannot reuse a default run's compiled graph. The tuned config is chosen at runtime inside an opaque custom op, so it is safe with compile caches.
- **vLLM's tuning script** (`benchmarks/kernels/benchmark_w8a8_block_fp8.py`) hard-codes DeepSeek shapes, so we use our own tuner around the same production kernel.

## Notebook E12-K1

`Experiment Notebooks/E12_K1_fp8_kernel_probe_on_KAGGLE.ipynb`, built by `Build/build_e12_k1.py` (sha `8cd9bb87…`).

| Stage | What |
|---|---|
| 1 | Offline install of vLLM 0.19.0 from R5's wheelhouse and R5's libcuda shim. Both are copied verbatim from the R5 notebook by the builder |
| 2 | `k1_tune_block_fp8.py`: the 5 shapes on the 4 GPUs in parallel, with vLLM's own `_w8a8_triton_block_scaled_mm`. Pruned search per M (192 configs for M ≤ 128, 576 above); decode M first (64, 16, 48, 32, …), then prefill chunks. A 40 min budget, with partial results kept. Each best config is checked against the default output (rel. err ≤ 1e-2, else the default is kept) |
| 3 | `k1_engine_bench.py`: one fresh process per variant with R5's exact engine (TP4, fp8 KV, MTP-3, FLASHINFER, prefix caching, 16 seqs, `max_model_len` 106,496). The variants are default, tuned (configs copied in) and marlin (env). Workloads: 16 × short ARC prompts → 3,072 tokens; 16 × about 36k-token prompts → 1,024 tokens (prefill first, so the timed run is mostly decode); 4 × 192 greedy tokens |
| 4 | `report.md` / `results.json`: decode tok/s and speed-up vs default, MTP acceptance, preemptions, kernel-path evidence counted in each log, greedy common prefix vs default, per-shape tuning speed-ups. `configs_l4/` holds the files to ship |

**Run:**
- 4×L4, Internet OFF, the pinned Q27 image (sha 37c64f7d).
- Inputs: FP8 model `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`, wheelhouse `mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3`, competition data.
- Expected time is about 1.5–2 h. A guard stops new stages after 4 h. Not a submission.

**Local check:** `Build/harness/run_dryrun.py` executes the real cells with `K1_DRYRUN=1` (fake tuner and engine results) against a fake vLLM tree. All 10 orchestration checks pass:
- stages;
- 5 config files;
- configs installed only for the tuned variant, then removed, with vLLM's own files untouched;
- Marlin env only for marlin;
- speed-ups and greedy agreement computed;
- report written.

**Not testable locally:** the GPU paths (Triton compile and timing, vLLM engine) and the real compile time per config.

**If a variant wins:** ship the 5 JSON files (inline or as a dataset) and copy them into vLLM's configs directory in the production notebooks before the engine starts; or set `VLLM_TEST_FORCE_FP8_MARLIN=1`. Any change goes into a production notebook only as one change at a time, judged by solved tasks per hour.
