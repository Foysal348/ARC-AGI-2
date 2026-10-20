# Concurrency 16 → 25–30 on 4×L4 (vLLM 0.19, Qwen3.8-27B FP8): math and research (Claude, 2026-10-15)

**Sources:**
- Our own measurements and a replay of the KV pool: session scratchpad `order2/concurrency_math.py`.
- One research agent, which read the vLLM v0.19.0 source, issues and benchmarks.

## 1. Memory geometry (vLLM 0.19 source + our numbers)

- **Pool:** 846 blocks × 1,600 tokens = 1,353,600 tokens, about 11.2 GiB per GPU.
  - `kv_cache_usage_perc` **includes the GDN recurrent-state blocks**.
- **Fixed cost per running request:**
  - Agent (from source): 12 blocks at MTP k=3 (3 GDN groups × (1+k)) = **19.2k token-equivalents**, about 640 MiB in total.
  - Our fit to the measured KV time series: **21.2k**. The two agree.
- **Average live context:** about 38–40k tokens per running request (DIRECT); about 45k (Poetiq).

## 2. Can 25–30 fit as things are? No.

KV demand with our real request mix (replay calibrated on ORDER2, slightly optimistic):

| N | DIRECT mean / p99 | Poetiq mean / p99 |
|---|---|---|
| 16 | 65 % / 79 % | 71 % / 85 % |
| 20 | 82 % / 98 % | 89 % / 106 % |
| 25 | 104 % / 122 % | 113 % / 132 % |
| 30 | 125 % / 146 % | 136 % / 159 % |

The safe maximum today is about **18 (DIRECT) / 16 (Poetiq)**.

## 3. Levers (agent's table; N at ≤ 80 % mean usage)

| Config | DIRECT N | Poetiq N |
|---|---|---|
| Now (fp32 state, k=3, mem 0.90) | 18.6 | 16.7 |
| mem 0.95 | 20.9 | 18.8 |
| mem 0.95 + k=2 | 22.8 | 20.3 |
| mem 0.95 + **fp16 SSM state** (`mamba_ssm_cache_dtype="float16"`; bf16 is not available in 0.19) | **25.1** | **22.1** |
| + INT4 W4A16 weights (model-quantization change, needs approval) | 31.2 | 27.5 |

Notes:
- Lowering k loses 15–40 % of the tokens per step (3.01 → 2.48 → 1.82). It only pays when memory is the sole limit.
- CPU KV offload does not work for Qwen3.5-family models in 0.19.

## 4. Throughput gain (inference)

| | N≈25 | N=30 |
|---|---|---|
| Batching (russcore-shaped fit) | +12–18 % | about +23 % |
| Possibly from fp16 state I/O (unmeasured) | +5–15 % | — |
| Fixed-overhead step model (upper bound) | up to +42 % | — |

**Realistic: about 380–430 tok/s vs 333 now.**

## 5. Required changes (one at a time)

1. **Admission control in our scheduler.** Admit a new request only when KV usage < 0.75 and running < `MAX_NUM_SEQS`.
   - Today, `CONCURRENCY = kv_tokens // 43000` = 31 is capped only by `MAX_NUM_SEQS` = 16, so raising that cap alone would over-admit.
2. `gpu_memory_utilization` 0.95, or `kv_cache_memory_bytes`; `max_num_seqs` 24–28; `max_cudagraph_capture_size` 128.
3. `mamba_ssm_cache_dtype="float16"`, only after an A/B on long thinking traces.

## 6. Risks

- OOM at 0.95.
- MTP illegal-memory-access reports under load (#36613, unconfirmed on 0.19).
- Quality drift from the fp16 state on 60–78k-token traces.
- **Prefix caching + MTP silent output corruption on hybrid GDN models** (#43559: 94 % → 75 % on 0.21; #53912 open). This is untested on 0.19, and **we run exactly this combination**. Worth an EVAL12 A/B with prefix caching off: it could even raise accuracy.
