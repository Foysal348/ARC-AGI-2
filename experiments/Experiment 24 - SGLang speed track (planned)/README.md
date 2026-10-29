# Experiment 24: SGLang speed track (planned)

**Goal.** Reach 1500+ tok/s sustained aggregate accepted-output decode for Qwen3.8-27B on Kaggle 4x L4 (sm_89, 24 GB, PCIe, no NVLink, offline, 12 h) with SGLang. The SGLang accuracy gap must be closed before any speed winner is used in a submission.

This is the SGLang half of the two engine tracks the user chose on 2026-10-29: vLLM 0.32 is Experiment 23, and SGLang is this experiment.

- **Research basis:** `Resources/Qwen3_8_27B_SGLang_0_5_21_Research_BN_v5_FINAL.pdf` (v5.0, 48 pages). Its final priority queue is AA1 and its campaign is AA8.
- **Sources:** all URLs and local evidence are in `Resources/sources/SOURCES.md`.

Status: **planned**. No notebook has been built and no GPU has run. The notebook is built only after the user's go, and every GPU run needs a separate go.

## Base engine (the reference, not a frozen configuration)
### Speed reference: EMS arm `dflash_w6_s20`
Source: `Base Engine/notebooks/Q27-XHIGH-EMS_SAFE_ENGINE_MATRIX_SGLANG.ipynb`, sha `8f01847c`. The argv is in `Base Engine/server_logs/sglang_server_ems_dflash_w6_s20.log`.

```
python -m sglang.launch_server --model-path <RedHat INT4 W4A16 g128> --tp-size 4 --context-length 155648 \
  --kv-cache-dtype fp8_e4m3 --mamba-ssm-dtype float16 --mamba-radix-cache-strategy extra_buffer_lazy --max-mamba-cache-size 84 \
  --speculative-algorithm DFLASH --speculative-draft-model-path <qwen3-8-27b-dflash2 bf16> \
  --speculative-draft-model-quantization unquant --speculative-num-draft-tokens 6 \
  --max-running-requests 21 --chunked-prefill-size 4096 --mem-fraction-static 0.83 \
  --attention-backend flashinfer --linear-attn-backend triton --disable-flashinfer-autotune
  # plus page size 1, schedule policy lpm, --min-free-slots-delay 1, --enable-metrics, seed (see the server log)
```

- **Engine:** SGLang 0.5.21, cu13, in the venv `/tmp/sglang-venv`. It installs offline from the dataset `foysalemonshanto/sglang-0-5-21-cu13-wheelhouse`.
- **Load path:** weights are staged to `/dev/shm` with parallel ranged readers.
- **Engine modules:** `Base Engine/src_e14_modules/`, extracted from payload `q27x_e14_modules_55c1eb7911c6`:
  - `e14_loadpath.py`
  - `e14_sglang_patches.py`
  - `e14_measure.py`
  - `e14_nccl_bench.py`
  - `e14_trace_summary.py`
- **Inputs:**
  - Models: `foysalemonshanto/qwen3-8-27b-int4-redhat` (w4a16-g128), `foysalemonshanto/qwen3-8-27b-dflash2` (bf16), and `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1` (hf-fp8, for the vLLM oracle arms).
  - Pinned image: `gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`.

### Production reference: E15B R3K2 Duck
Source: `Base Engine/notebooks/Q27-XHIGH-E15B_EVAL120_50K45K_S16_R3K2_DUCK_SGLANG.ipynb`, sha `c058d58b`.

The production run uses the same engine at 16 client streams, with these differences from the EMS benchmark arm:
- Mamba cache about 68;
- `--max-running-requests 17`;
- decode CUDA graphs 1-17;
- the final override is DFlash width 6. The early `e11_flags()` text mentions width 8.

Comparisons use the effective final argv. The Duck harness is payload `q27x_e15b_duck_9e4b89285136`, kept in `Experiment 15 .../Build/payload/` and the Kaggle payload dataset.

On EVAL120 the 6 h run scored 39.31 % task-norm.

### Measured baseline (`Base Engine/results/ems_report.md`, TP4, fp8 KV, prefix cache on)
| Arm | Streams | 8k tok/s | 40k tok/s | XL ~90k tok/s (actual running) | Accept / step 8k / 40k | Step ms 8k / 40k |
|---|---|---|---|---|---|---|
| **DFlash W6** | **20** | **597** | **536** | 364 (7) | 4.17 / 4.21 | 140.3 / 157.1 |
| DFlash W6 (ref) | 16 | 583 | 521 | 370 (8) | 4.21 / 4.11 | 115.7 / 127.3 |
| DFlash W4 | 16 | 592 | 526 | 363 (9) | 3.30 / 3.31 | 89.2 / 100.9 |
| DFlash W8 | 16 | 573 | 477 | 363 (7) | 5.00 / 4.47 | 138.8 / 149.9 |
| MTP (3,1,4) | 16 | 478 | 402 | 296 (11) | 2.88 / 2.94 | 96.4 / 117.3 |
| Target only | 16 | 405 | 354 | 275 (15) | 1 / 1 | 39.5 / 45.1 |
| W6 + PCIe-IPC all-reduce | 16 | 230 | 202 | 233 (8) | 4.32 / 3.91 | 299.6 / 309.9 |

**Step breakdown** (E14A profile: W8, 12 streams, about 5.9k context; not W6 S20). The step is about 116 ms:

| Component | ms per step |
|---|---|
| NCCL all-reduce (~140 × 983,040 B) | 50.9 |
| All-gather | 8.1 |
| Marlin INT4 GEMM | 25.3 |
| BF16 drafter / lm_head GEMM | 14.5 |
| GDN Triton | 8.0 |
| Rest | ~9.1 |

**What 1500 tok/s requires.**
- **NCCL:** raw NCCL SHM costs 343.88 µs per 983 KB all-reduce in a graph. Forced P2P is 5.5× slower.
- **Step time:** 1500 tok/s needs a step of about 56 ms at 20 active × 4.21 accepted. At the 7 actual running streams of the 90k band, it needs about 19.7 ms. So the critical path and real residency must improve together.

## Known accuracy problem (P0, must be explained first)
On the same Poetiq method, the SGLang INT4 + DFlash2 runs scored about 6-7 LB points below vLLM FP8 (E15P 26.94 and E11-06 29.17, vs about 32). The cause is unknown.

Candidates:
- INT4 vs FP8 weights;
- the engine's INT4 numerics;
- DFlash speculative state (#36548, #35150, #38009, #43514);
- Mamba checkpoint bugs (#43567, #43559, #43029, #22935);
- DFlash tap contamination (#41736) and NaN rows (#41737);
- the stale CUDA driver compute cache (#42917);
- harness, coverage or drain differences.

Earlier probes passed: EMS "BOUNDARY PASS" and DFlash parity against target-only. These were short probes and do not certify 60k-120k reasoning.

## Plan: six sections in ONE notebook, SMOKE before FULL
The structure follows E23: one notebook, a few measured arms per section, a SMOKE first, and FULL using the 12 h session (hard stop about 11.5 h). It applies the user's rules:
- thinking up to 120k, so the 120k band is measured (`--context-length` ≥ 131072, already 155648);
- no single-change LB ablations;
- one fresh server process per arm.

The measurement code is E23's `e23_lab.py`:
- distinct ARC prompts per stream;
- a steady window after prefill settles;
- the margin oracle (FAIL at margin ≥ 0.5);
- the per-GPU release gate;
- the watchdog;
- fail-closed statuses.

It gets an SGLang adapter for `/metrics`, `/start_profile` and the server args.

| # | Section (PDF ID) | Arms | Decisive number | STOP / NO-GO |
|---|---|---|---|---|
| 1 | **Accuracy-gap decomposition** (S-P0-01, N2, AA5, W8; SG-TARGET-ORACLE, SG-CACHE-CLEAN) | The four-way oracle on identical token ids. (a) vLLM 0.32 FP8 target-only; (b) vLLM 0.32 INT4 target-only; (c) SGLang INT4 target-only; (d) SGLang INT4 + DFlash W6 and W4. Every arm runs with a fresh isolated `CUDA_CACHE_PATH`, `TRITON_CACHE_DIR` and `SGLANG_CACHE_DIR` per arm, with the resolved path logged in every rank. Also: forced-reject, serial vs c16, and a long continuation (≥ 30k). | First high-margin divergence and which pair it falls between. Then a paired ARC sample (24-64 tasks, same budget). | A high-margin flip in (c) vs (d) means DFlash state; in (b) vs (c), the engine INT4 path. If the cache path is not honoured, the result is INVALID. |
| 2 | **DFlash width × streams** (S-P1-02, P, W2, AA4; SG-W4-S20, SG-W5-W7) | W4 S20 (never measured; the cheapest), W4 S16, W6 S16, W6 S20. W5 and W7 only if the curve says so. Bands 8k / 40k / 90k / 120k. Max-running and Mamba slots stay consistent (21 / 84 for S20). | Accepted-output tok/s, actual running p50, verify step ms. | XL active collapses, net tok/s is not better, or parity fails. |
| 3 | **TP4 collective critical path** (S-P1-01, Q1, W6, Y8; SG-AR-TRACE, SG-CTA) | Profile first: torch trace / NCCL_DEBUG=INFO to get the real critical path of about 140 × 983 KB all-reduce plus 3 × 11.9 MB gather. Then fresh launches: NCCL_MIN/MAX_CTAS 2 / 4 / 8, NCCL_ALGO Ring / Tree, NCCL_PROTO Simple / LL, and NCCL_NTHREADS 512 only if this NCCL version accepts it. | Verify-step p95 and 40k / 90k tok/s. The microbenchmark alone does not count. | Slower end to end, a hang, or graph timeouts. Never forced P2P or PCIe-IPC again. |
| 4 | **Capacity at 90k / 120k** (S-P1-03, W3, Y5, AA4; SG-CAPACITY) | Mamba slots: the source-derived minimum, then 68 / 84 (with 88 / 92 only if the minimum allows). Matching max-running 17 / 21. mem-fraction-static 0.80 / 0.83 / 0.86. 12 / 16 / 20 / 24 logical streams, shared vs distinct prefixes. | Actual running at 90k and 120k, KV pool, Mamba available / evictable, retractions. | "Can not alloc mamba cache", OOM, or prefix recompute. Never cut Mamba slots only to gain KV. |
| 5 | **Prefill / decode interference** (S-P2-01, R1, W5, Y7; SG-PREFILL) | `--chunked-prefill-size` 2048 / 4096 / 8192 under four loads: D0 = 20 resident; D1 = 19 resident + 1 warm 90k; D2 = 19 + 1 cold 90k; D3 = 16 + 4 cold 90k. | Decode TPOT p95, output tok/s over the whole window, prefill seconds. | Worse mixed-wall throughput, or decode starvation. |
| 6 | **Correctness backports** (S-P0-02/03, O, W4; SG-COMMITTED-CKPT, SG-DRAFT-CONV, SG-E14-MAMBA) | One hunk at a time on 0.5.21, each with a source sha and a negative test: #43567 + #43559 together (never #43559 alone), with forced optional-alloc failure → stale donation → restore → KL; #43029 with grammar truncation where accepted > committed; #41736 with per-request sentinels at 1 / 8 / 16 concurrency; #41737 with injected all-NaN rows. | Patched arm clean, unpatched arm shows the causal chain. If the chain is missing, the result is INCONCLUSIVE, not PASS. | Any wrong restore, assertion, or slot leak. |

**SMOKE** covers:
- section 1 (a)-(d) at 8k / 40k, short;
- the W4 S20 arm;
- one NCCL CTA arm, with the profile;
- the 90k capacity census at 84 slots;
- one D2 interference load;
- the #43567 / #43559 forced-fail check.

**FULL** runs everything, then A/A repeats, as in E23.

## Later branches (after the six sections; each one is a separate go)
- **Host-staged 4-rank all-reduce in SGLang** (S-P3-03, AA3; SG-HSAR-BF16 / SG-HSAR-INT8). It reuses the E23 P5 kernel (`Experiment 23/prep/P5_host_allreduce/`, in the dataset `foysalemonshanto/q27-e23-sm89-kernels`) and adapts the ideas of PR #39605. BF16 comes first; the INT8 payload only behind the long-context logprob gate.
- **Checkpoint retention for agent turns** (Y2, Y3, AA6; SG-RADIX-LRU, SG-LIVE-LEASE).
  - #40865: retention policy, not replay.
  - Franzen's Duck: retained prefill checkpoints, LRU refresh, larger trim blocks.
  - #22935: radix-split test (repeat 50k, then change the last 1 / 16 / 64 / 512 tokens).
  - A bounded tool-return lease, which is a CPU replay first.
- **Low-M kernels** (Y4, AA7; SG-LOWM-SPLITK): Marlin / GDN `in_proj_ba` split-K and LM-head sharded sampling, only where the trace shows the hot spot.
- **EAGLE-4** (S1): a patched-SGLang build of eagle-harness. The license decision is the user's. The vLLM fork variant is already checked in E23 P6.
- **PP4 / TP1x4** (S-P3-02, S2): SGLang PP support (#43885) with the Merlin 16 GiB checkpoint. Sizing comes from the E23 section 1 results.
- **ReplaySSM** (S-P3-01, Y9): only with an extra_buffer_lazy-compatible cursor; blind no_buffer is NO-GO.
- **SGLang 0.5.22 or newer** (S-P3-04): a separate engine migration, not a 0.5.21 flag change.

## Fail-closed gates (PDF G, U3)
An arm is NO-GO if any of these holds:
- **Provenance.** Patch provenance is missing; a flag is unsupported; the run needs internet; or the model differs from the declared checkpoint.
- **Launch.** Startup exceeds 600 s, or a CUDA kernel is invalid on SM89.
- **Correctness.**
  - stale Mamba donation or restore;
  - target-only logprob drift with high margin;
  - grammar or tool-call corruption;
  - repeated punctuation or empty output;
  - output caps or truncation that go unreported;
  - missing cold / warm controls.
- **Coverage.** Submission coverage falls.

INCOMPLETE and NOT_EXERCISED never count as PASS.

The three throughput denominators (Y11) are always reported separately:
- saturated decode;
- mixed serving window;
- agent end to end.

Adoption is decided by correct ARC outputs per GPU-hour, not by raw tok/s.

## Measured negatives (not repeated unless the code changes; PDF H)
- FlashInfer PCIe-IPC all-reduce: 0.39×.
- Forced NCCL P2P: 5.5× slower.
- DFlash W8 everywhere.
- W6 S24 blindly: it is KV-capped.
- MTP (3,1,4) as a speed upgrade.
- FlashInfer GDN on sm89: needs SM90+.
- CuTe DSL GDN verify on sm89.
- ReplaySSM with no_buffer.
- The 4-slot #43567 "natural" test: it hits the #43559 assert first.
- `NCCL_SHM_USE_CUDA_MEMCPY=1`: slower, and it times out in graphs.
- 2xTP2: the KV does not fit at 40k+.

## Folder layout
- `Resources/`: the SGLang research PDF (v5.0 FINAL).
- `Resources/sources/`:
  - `SOURCES.md` (every URL and local evidence file);
  - `research/` (Claude E22 report, 5 agent notes, GPT prompt);
  - `measurements/E14A_nccl_bench.json`.
- `Base Engine/`:
  - `README_sglang.md`;
  - `notebooks/` (EMS SAFE `8f01847c`, E15B R3K2 `c058d58b`);
  - `src_e14_modules/` (engine modules);
  - `results/ems_report.md`;
  - `server_logs/` (7 EMS server logs);
  - `e22_speedlab_sglang/` (the planned, never-run E22 SGLang speedlab notebook, its builder, and the E22 engine findings).
- `Experiment Notebooks/`, `Experiment Results/`: empty until the build.
