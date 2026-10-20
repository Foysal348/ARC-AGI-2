# Q27-XHIGH run 1 — Kaggle 4xL4, version 355535601 (foysalemonshanto/q27-xhigh), notebook v1.3.2, EVAL12

Result: 0/12 tasks ran. The notebook still finished cleanly: report, bundle and identity submission were written (the
v1.3 safety paths worked).

Engine ladder:
1. **Attempt 1: `kv_cache_dtype=fp8_e4m3` + MTP3** failed after 792 s with
   `EngineDeadError: EngineCore encountered an issue`. The root-cause trace is only in the notebook stdout; the run was
   saved without a captured log, so it is not in this bundle.
2. **GPU release check:** after 181 s every GPU still held ~20.7 GB (20,699 / 20,973 / 20,699 / 20,701 MiB).
   - Cause: `LLM()` itself raised, so there was no object to shut down, and the vLLM worker processes stayed alive.
   - The notebook therefore stopped the ladder (`gpu_memory_not_released_after_failed_engine`).
3. **Setup to scheduler:** 935 s (~0.5 quota-hours).

Fix in v1.3.3:
- Kill every child process of the kernel after a failed attempt, then check GPU release (harness scenario `orphan`).
- Start with bf16 KV + MTP3; fp8 KV only with `Q27X_KV_DTYPE=fp8_e4m3` until the trace is known.
- Run as Save & Run All so that the log is captured.

## Correction (from the full run-1 log, seen by GPT)
The real root cause was **not** fp8 KV. Model weights (~7.15 GiB/GPU), MTP and the fp8 KV cache (10.14 GiB, 312,000
tokens) all loaded. With fp8 KV, vLLM selected FlashInfer on sm89. FlashInfer JIT-compiled its kernels and then failed
to link them: `/usr/bin/ld: cannot find -lcuda`, because Kaggle has only `libcuda.so.1`. That failure surfaced as
`EngineDeadError`.

v1.3.4 changes:
- A `libcuda.so` shim on `LIBRARY_PATH` plus a real `-lcuda` link probe, both before vLLM starts.
- An fp8-first ladder with explicit `attention_config.backend`:
  FlashInfer (if the probe passes) → Triton → bf16 + FlashAttention-2 → bf16 without MTP.
- Concurrency sized from the measured KV: 312k // 30k = 10 slots.
