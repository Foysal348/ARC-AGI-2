# Experiment 23: six levers in one notebook (vLLM 0.32, Qwen3.8-27B, 4x L4)

Source: the user's "six most important experiments" plan (2026-10-29). There is one GPU notebook with six sections. Each
section runs only the test sequence the plan names, and adds no extra arms. If a section fails, the failure is recorded
and the next section runs. One GPU session must fit in 12 h; the time column below estimates each section.

| # | Experiment | Steps in the notebook (in order, the plan's own sequence) | Launches | Est. min |
|---|---|---|---|---|
| 1 | Parallelism redesign: TP1x4 / PP4 | (a) Merlin W4A16 checkpoint on one L4: load, memory fit (peak VRAM, KV pool, host RAM). (b) 8k / 40k / 90k correctness on that replica. (c) 2 independent TP1 replicas. (d) 4 TP1 replicas. (e) TP4 with the same checkpoint at a matched output budget. Separate branch: TP1xPP4 with our INT4, run target-only short, then 2k, then 40k; stop at the first garbage or illegal-memory step (issue #63505). | 1+1+1+1+1 (+PP4) | 110 |
| 2 | Four-rank host-staged all-reduce | Standalone 4-process benchmark at 983,040 B (our measured decode size). (a) BF16 / FP32 numerical oracle against NCCL. (b) Two-shot reduction through a pinned host buffer, latency per call against NCCL SHM (343.88 us measured). (c) CUDA graph capture of the host-staged path. (d) INT8 payload compression as its own arm. vLLM model integration comes in a later notebook; the vLLM trace in section 3 measures how much of the step the collective takes. | 0 (no server) | 15 |
| 3 | FULL graph speculative verification | G0 FlashInfer + PIECEWISE (control); G1 TRITON_ATTN + FULL_AND_PIECEWISE; G2 TRITON_ATTN + FULL_DECODE_ONLY; G3 FlashInfer + #60532 backport (python-only; `git apply --check` is clean on v0.32.0 flashinfer.py). Each arm records a short torch-profiler trace. The proof is that the MTP-3 verify step replays as a FULL CUDA graph (graph launches vs individual kernel launches in the trace), not only that the server starts. | 4 | 60 |
| 4 | Native SM89 FP8 GEMM + Merlin low-M patches | (a) #61224 CUTLASS block-FP8 W8A8 (SM89) built as a standalone torch extension, against the existing MarlinFP8ScaledMMLinearKernel, with the same weights and scales on our actual TP4 GEMM shapes (qkv / o / gate_up / down / GDN in_proj, M = 1..80); numerical match first. (b) Merlin patches: they come with the Merlin engine in section 1 (in_proj_ba split-K, FlashInfer plan cache, uniform multi-token graph, second draft no-sync), so section 1's TP4 Merlin row against stock vLLM is their combined A/B. INT4 and FP8 paths are reported separately. | 0 (microbench) | 20 |
| 5 | NVFP4 KV + active-stream residency | FP8 KV control (= G0) -> NVFP4 KV target-only (correctness) -> NVFP4 + MTP3 -> 90k memory census (KV tokens, actual running streams at 90k) -> matched throughput against the control. `--kv-cache-dtype nvfp4` on SM8x is in vLLM 0.32.0 (#46963, release notes); SM89 itself was untested upstream, so step 2 is the gate. | 2 | 45 |
| 6 | Better speculation: MTP OFF / K1 / K3 | One server per depth (MTP off, K1, K3), each measured at 8 / 12 / 16 / 20 actual active streams. Score: accepted tokens per target verify, draft + verify time, effective active streams, final accepted tok/s. EAGLE-4 is a later, separate branch (draft checkpoint and memory fit). | 3 | 75 |

Total is about 5.5 h of GPU time, plus about 30 min of setup (vLLM 0.32 venv, weight staging, Merlin venv).

## Prep before the GPU notebook (CPU kernels on foysal, internet on where noted)

- P1 (internet): Merlin checkpoint `halt95/Qwen3.8-27B-W4A16-Merlin` at revision `7a0af39f63dd4fb0733c23d3ec126762074a503f` (~16 GiB, Apache-2.0), downloaded into a kernel's output and attached to the GPU notebook through `kernel_sources`.
- P2 (internet): Merlin engine environment. CPython 3.13 standalone, the `merlin-v3.0.0` wheel `vllm-0.32.0+merlin.7002f56537-cp313`, and `release/requirements-0.32.lock` as a cp313 wheelhouse. The Kaggle image is Ubuntu 22.04 (glibc 2.35) and the README asks for glibc 2.38, so the CPU kit runs `install-env.sh --verify` and imports the engine on the pinned image. If the import fails, the fallback is a source build of the Merlin checkout for Py3.12 and glibc 2.35 in the same kit.
- P3: the #60532 backport, `patches/pr60532.diff` (flashinfer.py only), applied to the vLLM 0.32 venv at runtime.
- P4 (internet): the #61224 kernel files (`patches/pr61224.diff`) plus CUTLASS headers, built for sm_89 into a `.so` on the CPU kit (nvcc from the CUDA 13 wheelhouse), with a Python binding for the microbenchmark.
- P5: the host-staged all-reduce, a small CUDA extension (pinned shared host buffer, GPU-side flags so it can be graph-captured), built into a `.so` with P4.

Payloads stay in a dataset; the notebook stays under 400 kB.
