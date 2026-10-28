# E13 LOADPROBE, GPU run 1 (2026-10-27)

- Kernel: `foysalemonshanto/q27-xhigh-e13-flashnext-loadprobe` v1, scriptVersionId 362636813.
- Notebook sha 107aa182, 4x L4, total 29.5 min.
- Outputs are in `LOADPROBE_GPU_run1_2026-10-27/`; the top-k `.pt` fixtures were dropped.
- The memory trace (`q27x/e13_memtrace/<arm>/trace_<pid>.jsonl`) came from our sitecustomize hook. It does not edit SGLang.

## Results

| Arm | Model | Outcome | GPU memory per rank | Smoke |
|---|---|---|---|---|
| bf16ple (rev 6b exactly) | Franzen's Intel package, BF16 PLE | **ready** after 1,132 s: load 582 s, decode graph capture 420 s (TileLang JIT) | after load 19.10 GB used, 2.51 GB free; KV fp8 **94,528 tokens**; 1.05 GB free after graphs | **garbage**: count `!!!!!!!!…` (32 tokens); chat empty, 4,096 reasoning tokens |
| fp8ple (E3 v4.2.1 merge) | 15 Intel shards + 10 FP8 PLE shards (merge OK: 25 files, FP8 config) | **OOM** on TP1 after 235 s, in the Marlin MoE repack of `model.layers.8.mlp.experts` | — | not reached |

Other results:
- The top-k test gave the same result as rev 6b: t1 2 / 98 failures (both uniform), t2 0 / 98.
- Host-RAM guard: bf16ple needed 122.4 GB, fp8ple 71.2 GB, against 155 GB cgroup free.
- The GPUs were free before and after each arm.

## Finding 1: the OOM is a delayed free in the post-load repack, not the PLE package

The trace has one row per TP rank and arm:

| | bf16ple (all 4 ranks) | fp8ple TP1 (the OOM rank) |
|---|---|---|
| allocated at `load_weights_only` start | 18.45 GiB (the parameters are allocated at model init) | 18.45 GiB |
| after `load_weights_only` | 18.45 GiB; CUDA tensors 18.35 GiB | 18.45 GiB; CUDA tensors 18.35 GiB |
| per MoE layer inside `process_weights_after_loading` | +309 MiB (the new Marlin buffer) | +309 MiB |
| allocated after each MoE layer | **flat, about 18.6 GiB**: the old buffer is freed when the module's staging context exits | **grows by 0.30 GiB per layer**: 20.26 → 20.56 → 20.86 GiB after layers 5, 6 and 7, then the 200 MiB allocation for layer 8 fails |
| end | 18.30 GiB, CUDA tensors 18.21 GiB, OK | OOM |

Notes on the evidence:
- Before post-load both arms are byte-identical on the GPU (18.35 GiB of tensors). The PLE package only changes host memory: 23.84 vs 11.92 GiB host tensors.
- The pre-repack buffers are freed in one launch and kept in the other. The code path and the weights are the same. This points to a reference cycle that only Python's cyclic GC frees, so the timing depends on when GC runs. **(Inference; UNVERIFIED which object holds the cycle.)**
- rev 6b (2026-10-26) failed the same way on TP0 at the same 20.86 GiB. E3 on 0.5.19 loaded; that was probably luck rather than a version difference. The repack and staging code is identical in 0.5.19 and 0.5.21.
- Candidate fix: from the same sitecustomize, call `gc.collect()` after each module's post-load step whenever allocated memory has grown more than about 256 MiB since the last check. This is source-free, and the effect is measurable in the trace.

## Finding 2: once loaded, the engine produces garbage on 4x L4 (the E3 symptom, again)

- Greedy `"1, 2, 3, 4, 5, 6, 7, 8,"` continues as `!` × 32 (token 0, typically NaN or Inf logits). The chat question "2 + 2" thinks 4,096 tokens and returns no content.
- The setup that produced it:
  - SGLang 0.5.21; classic FA2 for the QSA decode; Franzen patch #3;
  - TP4 + EP4; Triton GDN; fp8 KV (no KV scales: "Defaulting to scaling factors of 1.0");
  - decode CUDA graphs; mamba bf16.
- E3 got the same `!!!!` on 0.5.19 with a bf16 KV cache, the Triton attention backend and no graphs. So the cause is not the KV dtype, not FA2 and not CUDA graphs.
- What remains common to both: sm89 L4, TP4 + EP4, the Marlin W4A16 MoE kernels with EP4, the TileLang QSA kernels, Triton GDN and the pinned-PLE gather. Franzen's working reference is TP1 on sm120.
- The log shows TileLang warnings "Data race detected: `Logits(bx, position)` is written by multiple threads" during the decode-graph capture (the QSA indexer kernel). This is a candidate, **not verified**.
- Decode speed of the garbage stream: 40.8 tok/s at 1 stream with graphs. It is recorded only; it means nothing for a correct model.

## Next probe (proposal, not built)

A load + NaN-localisation probe, about 35 min, one arm (bf16ple):
1. Apply the `gc.collect()` fix in the hook, so the load succeeds deterministically. The trace shows whether memory stays flat.
2. Run eager (`--disable-cuda-graph`), so forward hooks see every module.
3. Add forward hooks, registered from the hook after `postprocess_weights`. For the first prefill and the first 4 decode steps, they record per rank the first module whose output has NaN or Inf (or |x| > 1e4), with the layer index and type: GDN, QSA attention, MoE / Marlin, dense MLP, PLE, norm, lm_head.
4. Then repeat the smoke.

This names the broken component on sm89 before any further engine work.
