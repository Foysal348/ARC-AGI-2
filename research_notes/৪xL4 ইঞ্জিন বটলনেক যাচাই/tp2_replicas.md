# 2×TP2 replicas (DP2) vs one TP4 server: Qwen3.8-27B INT4 + DFlash2 on Kaggle 4x L4, SGLang 0.5.21

Research date 2026-10-23. Every number is tagged:
- **[M]** measured by the cited source;
- **[C]** claimed by the source without a measurement shown;
- **[D]** arithmetic I derived from cited numbers;
- **[E]** my estimate or model.

**Local paths:**
- SGLang source: `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang/srt/`, abbreviated `SRC/`.
- Our measurements: `D:/ARC AGI 2/research_notes/Qwen 27B দ্রুততম সার্ভিং সেটআপ/our_measurements.md`, abbreviated `OURS`.
- W1 server log: `D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/Kaggle Runs/E11-04 rev2b run1 (esfersami50, 2026-10-18)/e11_04/server_W1_lazy_fp16_try0.log`, abbreviated `W1LOG`.
- Model config: `D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures/tok_int4/config.json`, abbreviated `CFG`.

## Measured comparisons: TP2 vs TP4 vs 2×TP2 on PCIe/no-NVLink, and extrapolation to our workload

### Takeaway
Every published 2×TP2 (or DP2) win over TP4 comes from **short contexts (128–8k tokens) at high concurrency, with KV memory to spare**. In that regime the ratio is 1.3–2.4×. At total concurrency of 4 or less, TP4 ties or wins (0.7–1.0×).

No source tests our regime: 12 streams, 60–76k contexts, an 8-token speculative verify, and KV as the binding resource.

For us the decisive factor is memory, not speed. A TP2 replica with DFlash2 holds only about **170k KV tokens** (about 340k across both replicas), against 881k for TP4 W1. So 2×TP2 + DFlash2 cannot hold 12 × 60–76k streams; it holds about 5–8 [E]. My estimate of aggregate throughput on our workload is **about 0.7–1.0× of TP4 W1** [E]. Even if memory were not binding, I estimate only about **0.9–1.2×** [E].

### Cited Findings

**club-3090 discussion #773** (alesha-pro, opened 2026-07-26; 4× RTX 3090 24 GB, PCIe Gen3 x16, no NVLink, P2P enabled with the aikitoria patch and transfer-verified 12/12; 220 W; vLLM 0.23.0) — [club-3090 #773](https://github.com/noonghunna/club-3090/discussions/773)

- **Saturation ladder** (`--max-num-seqs 256 --enable-chunked-prefill`, no spec decoding, short prompts) [M]. Peak aggregate:

  | Model | TP4 | 2×TP2 | Ratio |
  |---|---|---|---|
  | Qwen3.6-27B INT4 AutoRound | 307 tok/s | **737** | 2.40× |
  | Qwen3.6-27B INT8-W8A8 | 358 | 741 | 2.07× |
  | gemma-4-12B AWQ, TP4 vs 4×TP1 | 1,007 | 3,425 | 3.4× |

  The routing between instances is not described.
- **Same sitting, AutoRound INT4** [M]:
  - single stream: TP2 53.80 vs TP4 69.26 wall tok/s (TP4 +29 %);
  - aggregate peak at c=64: a **single** TP2 instance (2 GPUs) 473.1 vs TP4 (4 GPUs) 294.8 (−38 %).
- **Matched pair, MTP n=3 on both arms** (INT4 AutoRound, fp8_e4m3 KV, vLLM 0.25.1, single stream) [M]:
  - decode: TP2 79.26 / 106.50 vs TP4 82.44 / 103.73 (narrative / code), +4.0 % / −2.6 %;
  - prefill at 10k: 1,556 vs 2,407 tok/s (+54.6 %); prefill at 90k: 1,202 vs 1,948 (+62.0 %);
  - TTFT at 90k: 73.1 s vs 45.6 s;
  - the TP2 arm ran vLLM custom all-reduce, and the TP4 arm had it disabled.
- **Custom all-reduce forced on at TP4** (gate patched) [M]:
  - decode at c=1: 80.95 → 92.92 tok/s (+14.8 %);
  - output at c=16: 144.06 → 164.66 (+14.3 %);
  - TPOT at c=16: 82.6 → 66.1 ms;
  - TTFT at c=16: +9 %;
  - greedy output diverged on 4 of 5 prompts.
- **Without spec decoding**, single-stream TP2→TP4 scaling was +38–62 % decode. With MTP the decode gain disappears [M].

**biMEMO/Qwen3.8-27B-int4-AutoRound model card** — [HF biMEMO](https://huggingface.co/biMEMO/Qwen3.8-27B-int4-AutoRound)
- Setup: RTX 3090 24 GB (interconnect and P2P not stated), vLLM 0.27.1, MTP n=1 (acceptance 89.8 %, mean accepted length 1.90), 512-token responses, `--max-model-len 262144`, `--max-num-seqs 32`, fp8_e4m3 KV.
- Aggregate tok/s [M]:

  | Concurrency | TP2 (single instance) | TP4 |
  |---|---|---|
  | c1 | 90.1 | 102.4 |
  | c2 | 176.1 | 187.4 |
  | c4 | 323.3 | 322.3 |
  | c8 | **518.6** | **469.4** |

- These are one TP2 instance on 2 GPUs vs TP4 on 4 GPUs, not 2×TP2.
- The card gives no mechanism: "TP4 did not scale as well at higher concurrency in our testing" [C].
- **Derived:** 2×TP2 at total c8, assuming an even split (2 × 323.3 at c4), ≈ 647 vs TP4 469 → about 1.38× [D].

**NVIDIA forum, "Comprehensive Qwen3.8-27B Study on DGX Sparks"** (2026-08-24) — [NVIDIA forum 381102](https://forums.developer.nvidia.com/t/comprehensive-qwen3-8-27b-study-on-dgx-sparks-quantization-speculative-decoding-and-tp-dp-scaling/381102)
- Setup:
  - model RadixArk/Qwen3.8-27B-NVFP4 on SGLang (image pinned by SHA);
  - DSpark speculative decoding, block 7;
  - 128 tokens in / 128 out;
  - nodes linked by CX-7 at 100 Gbps (network, not PCIe);
  - TP2+DP2 runs two TP2 replicas behind an SGLang round-robin router.
- Per-user TPS [M]:

  | Total concurrency | 1 | 2 | 4 | 8 | 16 | 32 |
  |---|---|---|---|---|---|---|
  | TP4 | 77.3 | 61.9 | 48.1 | 27.0 | 18.6 | 11.0 |
  | TP2+DP2 | 54.6 | 52.9 | 46.8 | 36.0 | 24.2 | 16.8 |

- Aggregate (TPS × C) [D]:
  - C4: 192 vs 187 (0.97×);
  - C8: 216 vs 288 (1.33×);
  - C16: 298 vs 387 (1.30×);
  - C32: 352 vs 538 (1.53×);
  - C1: 0.71×.
- Author's words [C]: "TP=4 the better use of four Sparks when the priority is maximum single-user or low-concurrency speed"; "two TP=2 replicas are the more effective configuration" for more users; "the round-robin layer therefore adds negligible observable overhead".

**vllm-topology-bench** — [mtecnic/vllm-topology-bench](https://github.com/mtecnic/vllm-topology-bench), summarized by [wz-it.com](https://wz-it.com/en/knowledge/ki/multi-gpu-llm-inference/)
- Setup: 4× RTX 3090, no NVLink, Qwen3.6-35B-A3B-AWQ (MoE), vLLM 0.21.0, 512 in / 256 out, max-model-len 8192, prefix cache off, round-robin across replicas.
- Aggregate tok/s, TP4 vs 2×TP2 [M]:

  | Concurrency | TP4 | 2×TP2 | Gain |
  |---|---|---|---|
  | c1 | 111 | 116 | +4 % |
  | c8 | 375 | 583 | +56 % |
  | c32 | 527 | 1,026 | +95 % |
  | c64 | 580 | 1,210 | +109 % |
  | c128 | 610 | 1,437 | +136 % |

- TTFT at c16: 3.9 vs 1.4 s.
- The author's conclusion is "interconnect-bound, not KV-bound" [C]. This is an MoE model at short context, so it is the most favourable regime for DP.

**Openzeka whitepaper** (DGX Spark, Qwen3.6-27B-NVFP4, vLLM + Ray, no spec, 128/128) — [Openzeka](https://whitepapers.openzeka.com/papers/qwen3.6-27b-dgx-spark-scaling/)
- Per-user TPS at c16 [M]: TP2 14.96, TP4 19.21.
- The 2×TP2 aggregate (~867 tok/s) vs TP4 (~508) is **computed from single-instance runs, not measured** [C/D].
- "Marginal TP2→TP4 efficiency 1.17× under load" [C].

**HyperQwen issue #254** (4× RTX 5060 Ti, PCIe Gen3 x8, all PHB, TP4, vLLM 0.30, custom AR off) — [HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254)
- NCCL left P2P off by default (`via SHM`) even though the driver reported peer access.
- `NCCL_P2P_LEVEL=PHB` turned P2P on [M]:
  - decode +6.1 % at c1, +10.1 % at c2, +13.3 % at c4, +2.0 % at c8 (greedy);
  - 28k prefill unchanged.
- This is a TP4 lever, not a 2×TP2 result.

**Generic guidance (not measurements):**
- vLLM forum: replicas are "generally better for maximizing tokens/s" when the model fits [C] — [vLLM forum](https://discuss.vllm.ai/t/data-parallel-or-tensor-parallel/1608).
- wz-it: "Model fits on one GPU but KV cache is too small … use TP=2" [C] — [wz-it](https://wz-it.com/en/knowledge/ki/multi-gpu-llm-inference/).

**Our own TP4 W1 baseline** [M], from `OURS` §A:
- decode at 12 running by context: <10k 500.1, 10–20k 460.3, 20–40k 409.0, 40–60k 389.4 (E11-06);
- whole runs: 379.9 tok/s (E11-06) and 328.9 (E11-05);
- per stream about 35 tok/s at 12 running;
- DFlash2 vs no-spec at TP4 with 12 running: 404 vs 289 tok/s.

### Inferences
- **Which conditions give which ratio** [D/E]:
  - Total c ≤ 4: TP4 ≥ DP (0.7–1.0×).
  - c = 8–32 at short context: DP 1.3–1.95×.
  - c ≥ 64 at short context: DP 2.1–2.4×.
  - With MTP, the single-stream TP2→TP4 decode gain disappears (club-3090), so spec decoding shifts the break-even toward DP.
  - Long context and KV pressure were never tested by any source. All DP wins had "44–117× more KV than needed" (vllm-topology-bench) or 128-token prompts.
- **Why DP wins at high concurrency:** each decode step has a fixed per-layer all-reduce whose message grows with batch tokens. On PCIe without NVLink, TP4 runs those all-reduces on NCCL. TP2 uses 1-peer reductions, and custom AR is allowed at TP2.
- **Our verify step resembles "high concurrency" in token count:** 12 streams × 8 draft tokens = 96 rows per verify, about 0.98 MB per all-reduce (hidden 5120, bf16) [D from `CFG`]. That favours DP. But:
  - each GPU in a TP2 replica reads **twice the weights per step**: about 11.4 GiB target + drafter, vs about 6.1 GiB at TP4 (`W1LOG`: target 4.62 + drafter 1.49 GB per GPU);
  - per-GPU KV bytes per step are the same in both layouts at equal per-stream context.
- **Step-time model for our workload** [E]:
  - TP4 W1 verify step ≈ 128 ms: 416 tok/s at 12 running with accept ~4.3–4.5 [D from `OURS`].
  - Floor: weights about 22 ms plus KV about 17 ms (12 × 40k × 10.5 KiB, at L4's 300 GB/s spec).
  - All-reduce about 35–55 ms: about 130 all-reduces × 0.3–0.4 ms over SHM. This matches the sibling note `nccl_p2p_l4.md` in this folder.
  - Residual compute and other costs: about 35–50 ms.
  - A TP2 replica at 6 streams (48 rows) ≈ weights about 41 + KV about 17 + residual about 40–47 + comm about 18 (SHM, 2 ranks, 0.49 MB) ≈ 115–123 ms for 6 streams. That is about 215–230 tok/s per replica, **about 430–460 total, 1.0–1.1× of TP4**.
  - With working P2P custom AR at TP2 (comm about 3–5 ms), about 1.15–1.2×.
  - This is the "if KV fit" case.
- **Realistic case with DFlash2** [E]:
  - KV allows only about 3–4 streams of 55–60k per replica (see the memory question).
  - Each replica runs at roughly 150–180 tok/s, about **300–360 total (0.75–0.9×)**, with half the concurrent Duck branches.
  - Range: **0.7–1.0×**.
- **No-spec 2×TP2:** it fits 12 × 76k, but it loses DFlash2's +39 % at 12 streams. I estimate about 150–250 tok/s total (0.4–0.6×) [E].

### Gaps
- No source measured 2×TP2 vs TP4 at contexts ≥32k with ≥8 streams, or with 8-token speculative verify. All published DP wins are short-context.
- No source measured L4s; all used 3090s, 5060 Ti or DGX Spark.
- club-3090 and vllm-topology-bench do not say how requests were split. biMEMO compares a single TP2 instance, not 2×TP2.
- The breakdown of our 128 ms step is inferred; no profile exists (see `nccl_p2p_l4.md`).
- No Winder.AI or CloudRift TP-vs-DP benchmark turned up in searches. The arXiv SSM-TP paper ([arXiv 2602.21144](https://arxiv.org/abs/2602.21144)) reports TP gains of 1.6–2.1× on 2 GPUs and 2.6–4.0× on 4 GPUs for Mamba/Zamba. Its abstract gives no DP baseline or interconnect, so it does not settle the question.

## Memory per TP2 replica on 24 GB L4s: how many 60–76k requests fit

### Takeaway
SGLang's own accounting in `kv_cache_configurator.py` reproduces our TP4 W1 pool to within 0.6 % (886,801 predicted vs 881,336 measured). Applied to TP2, it gives these KV pools per replica:
- **DFlash2: about 170k tokens** (about 340k system-wide) at mem 0.83, and about 224k at 0.88;
- **MTP: about 235k**;
- **No-spec: about 500k**.

We need about 700–920k tokens system-wide (12 × 58–76k). So **2×TP2 fits 12 long streams only without speculation**. With DFlash2 it fits about 2–3 streams of 60–76k per replica (about 4–6 total), or 4 streams of about 54k each (8 total).

### Cited Findings

**Model dimensions** from `CFG` [M]:
- hidden 5120, 64 layers, `full_attention_interval` 4 (so 16 full-attention layers), 4 KV heads, head_dim 256;
- GDN: 16 key heads, 48 value heads, dim 128, conv kernel 4;
- vocab 248,320, intermediate 17,408, `tie_word_embeddings` false.

**KV per token** [D]:
- 16 × 4 × 256 × 2 (K and V) × 1 B (fp8) = **32 KiB total**: 8 KiB per GPU at TP4, **16 KiB per GPU at TP2**.
- W1 check: 3.36 GiB K per GPU / 881,336 tokens = 4.09 KB, so K+V ≈ 8 KiB [D from `W1LOG`: "KV Cache is allocated … #tokens: 881336, K size: 3.36 GB, V size: 3.36 GB"].

**DFlash2 draft KV** [M/D]:
- `W1LOG` reports "DFLASH fused KV materialization enabled. n_layers=5, num_kv_heads=2, head_dim=128" at TP4, with draft K 1.05 GB per GPU = 1,280 B/token = 5 × 2 × 128 × 1 B.
- If that is 8 total heads split 4 ways, TP2 has 4 heads per GPU, so **5 KiB/token/GPU K+V** [E].

**GDN state per request:**
- 48 value heads × 128 × 128 × 2 B × 48 GDN layers = **75.5 MB total, fp16**, or 37.7 MB per GPU at TP2 [D from `CFG`].
- W1 measured: "max_mamba_cache_size: 56, conv_state size: 0.04GB, ssm_state size: 1.00GB intermediate_ssm_state_cache size: 1.83GB" [M `W1LOG`]. That is 1.04 GiB / 57 slots = 18.7 MiB per slot per GPU at TP4, including conv; at TP2 it is **37.4 MiB per slot per GPU** [D].

**SGLang sizing rules** in `SRC/mem_cache/kv_cache_configurator.py` [M, source]:
- KV budget = free memory after weights − `pre_model_load_memory × (1 − mem_fraction_static)` − the multimodal reservation (W1 log: "Reserving 0.10 GB") − mamba state (`_profile_available_bytes`, L2177–2230).
- With `--max-mamba-cache-size` set, the intermediate scratch = `per_req × (capped_reqs + 1) × speculative_num_draft_tokens`, where `capped_reqs = min(max_running_requests, max_mamba_cache_size // ratio)` (L2462–2480).
- Main state = `(max_mamba_cache_size + 1) × per_req` (L2563–2569).
- The ratio for `extra_buffer_lazy` with overlap is 4 slots per request (`_calculate_mamba_ratio`, L2232–2261; `OURS` "extra_buffer_lazy 4").
- So each running request costs 4 main slots plus 8 intermediate slots (DFlash2, 8 draft tokens), i.e. about 12 × 37.4 MiB ≈ 449 MiB per GPU at TP2 [D].
- The W1 intermediate check: 13 × 8 × 18.0 MiB = 1.83 GiB matches the log [D]. E11-06, with max-running 13, logged 1.97 GB = 14 × 8 × per_req [M `OURS`].

**W1 TP4 per-GPU budget** [M `W1LOG`]:
- avail at start 21.72 GB; after target 17.10 GB (target 4.62); after drafter 15.58 GB (drafter 1.49);
- KV 6.72 GB target + 2.10 GB draft = 881,336 tokens;
- target-verify graph 0.68 GB, draft-verify graph 0.35 GB;
- final `available_gpu_mem=1.84 GB`.
- The model above predicts 886,801 tokens for W1 [D], so the formula is validated.

**MTP (NEXTN) weights at TP4:** "Load weight end … type=Qwen3_5ForCausalLMMTP … mem usage=1.45 GB". Draft KV is 0.11 GB K for 474,203 tokens, about 256 B/token K = 1 layer × 1 head × 256 [M; `E11-03 …/server_B_int4_mtp_try0.log`].

**TP2 per-replica table** [E]. Assumptions:
- target 9.24 GiB per GPU (2 × 4.62);
- DFlash2 2.19 GiB per GPU (fitted: replicated 0.79 + sharded 2.79 GiB, from 1.49 at TP4 and a ~3.85 GB file);
- MTP about 2.8 GiB (vocab-parallel embed and lm_head are about 2.37 GiB at TP2);
- 6 running requests per replica, mamba cache 28;
- fp8 KV; mem 0.83 unless noted.

| Arm | rest after slack | mamba main | intermediate | KV GiB/GPU | Bytes/token/GPU | **KV tokens per replica** | System (×2) |
|---|---|---|---|---|---|---|---|
| DFlash2, R=6, mem 0.83 | 6.50 | 1.06 | 2.04 | 3.40 | 21,504 | **~170k** | ~340k |
| DFlash2, R=6, mem 0.88 | 7.58 | 1.06 | 2.04 | 4.48 | 21,504 | ~224k | ~448k |
| DFlash2, R=4 (K=19), mem 0.83 | 6.50 | 0.73 | 1.46 | 4.31 | 21,504 | ~215k (4 × 54k) | ~430k (8 streams) |
| DFlash2 weight sensitivity (9.0–9.6 + 1.9–3.0 GiB) | — | — | — | 2.2–3.9 | — | 111k–196k | 222k–392k |
| MTP (4 draft tokens), R=6, mem 0.83 | 5.89 | 1.06 | 1.02 | 3.81 | 17,408 | ~235k | ~470k |
| MTP, mem 0.88 | — | — | — | 4.89 | 17,408 | ~302k | ~604k |
| No-spec, R=6, mem 0.83 | 8.69 | 1.06 | 0 | 7.63 | 16,384 | **~500k** | ~1.0M |
| No-spec, mem 0.88 | — | — | — | 8.72 | 16,384 | ~571k | ~1.14M |
| **TP4 W1, measured** | — | 1.04 | 1.83 | 8.82 | 10,752 | — | **881,336** |

**What fits** [D]:
- 6 × 76k = 456k and 6 × 60k = 360k per replica.
- With DFlash2 at about 170k, a replica holds **2 streams of 76k or 2–3 of 60k**.
- No-spec holds 6 × 76k (456k of 500k) with little headroom for the Duck prefix cache.
- Our production need: "Production needs about 920k KV tokens" [M `OURS`, E113A L78]. Rev 5 Duck sizing was "12 streams × 58k = 697k" at the median, with p90 about 890k [M `OURS`, E12R5A L70-71].

### Inferences
- **Why TP2 loses so much KV:** going TP4 → 2×TP2 adds about **5.3 GiB of weights per GPU**, because each GPU now holds half of the target and drafter instead of a quarter. GDN state per GPU stays the same (half the requests, twice the heads per GPU). The extra weights come directly out of KV.
  - System KV falls from 881k to about 340k with DFlash2, about 60 % less [D].
- **Raising mem 0.83 → 0.88** buys about 54k tokens per replica. It would leave about 1.1 GiB after TP2 graphs. TP2 graphs are larger than TP4's 1.03 GiB because activations are not sharded; I estimate 1.3–1.6 GiB [E]. That is risky for chunked-prefill activations.
- **Duck prefix reuse gets worse.** Each replica has 170k instead of 881k for radix-cached prefixes. E12 EVAL30 already lost prefixes on 144 of 423 requests at 870k [M `OURS`]. 2×TP2 would thrash far more, and each replica's prefill is slower (club-3090: TP2 prefill at 90k is 38 % slower than TP4) [M].
- **The only 12-stream-capable 2×TP2 arm is no-spec,** and its throughput is far below W1 (see the first question).

### Gaps
- The exact TP2 weight footprint (replicated vision tower and norms) and DFlash2's real KV-head count are unverified. Launching one TP2 replica and reading the "Load weight end" and "KV Cache is allocated" lines would settle both in under 10 minutes.
- The TP2 CUDA-graph memory (target-verify and draft-verify) is unmeasured.
- Whether DFlash2 accepts `--speculative-num-draft-tokens` below its block size of 8 was not checked. Fewer tokens would shrink the intermediate scratch linearly.

## Implementation in SGLang 0.5.21: two servers vs `--dp-size 2` vs the Model Gateway; sticky routing; metrics

### Takeaway
The simplest correct setup is **two independent `sglang.launch_server` processes** (`CUDA_VISIBLE_DEVICES=0,1` on port 30000 and `2,3` on 30001), with **client-side sticky routing**: the Duck harness maps branch_id to a port. That needs no router and gives per-replica logs and `/metrics`.

Native `--dp-size 2` also supports per-request pinning through the `routed_dp_rank` field, but the docs call it "highly not recommended". The SGLang Model Gateway (`sglang-router` / `smg`) is a separate Rust wheel that is **not in our wheelhouse**. Its defaults include 5 retries, which is dangerous for 60k-token requests.

### Cited Findings
- **Native DP** is "highly not recommended for use right now". Its limits [C]:
  - built-in balancing only (`round_robin`, `total_requests`, `total_tokens`);
  - "no cache-aware routing, limited observability, no fault tolerance".
  - Source: [SGLang DP/DPA/SMG guide](https://docs.sglang.io/advanced_features/dp_dpa_smg_guide.html).
- **Native DP source** [M, source]:
  - `SRC/managers/data_parallel_controller.py` L87–91 defines `ROUND_ROBIN`, `FOLLOW_BOOTSTRAP_ROOM`, `TOTAL_REQUESTS`, `TOTAL_TOKENS`.
  - `maybe_external_dp_rank_routing` (L731–745) sends a request whose `routed_dp_rank` is set directly to that rank.
  - The OpenAI protocol accepts `routed_dp_rank` (`SRC/entrypoints/openai/protocol.py` L393, L974); `data_parallel_rank` is deprecated.
  - So sticky routing is possible inside one `--dp-size 2 --tp-size 2` server.
- **Metrics in DP:** `--enable-metrics-for-all-schedulers` makes "schedulers on all TP ranks (not just TP 0) record request metrics separately" [M, source `SRC/arg_groups/fields/observability.py` L82–85]. Without it, per-replica scheduler metrics may be missing in single-server DP mode [E].
- **Router commands** [C, docs]:
  - `python -m sglang_router.launch_router --worker-urls http://…:30000 http://…:30001 --policy cache_aware` (install `pip install sglang-router`);
  - co-launch: `python -m sglang_router.launch_server … --dp-size N`;
  - Prometheus default `0.0.0.0:29000/metrics`.
  - Sources: [SGLang DP guide](https://docs.sglang.io/advanced_features/dp_dpa_smg_guide.html); [SGLang Model Gateway docs](https://docs.sglang.io/advanced_features/sgl_model_gateway.html).
- **cache_aware defaults** [C]:
  - `--cache-threshold 0.3` ("Minimum prefix match ratio for cache hit");
  - `--balance-abs-threshold 64`, `--balance-rel-threshold 1.5`;
  - `--eviction-interval-secs 120`, `--max-tree-size 67108864`.
  - Retries default to `--retry-max-retries 5` on 408/429/500/502/503/504; `--disable-retries` turns them off.
  - Source: [SGLang Model Gateway docs](https://docs.sglang.io/advanced_features/sgl_model_gateway.html).
- **Gateway packaging** [M, PyPI page]: the gateway is also published as `smg`, latest 1.13.0 (2026-10-15). The wheel is `cp38-abi3`, manylinux_2_17 x86-64, so one wheel works on Python 3.12. Source: [PyPI smg](https://pypi.org/project/smg/).
- **Gateway policies** [C]: eight policies: cache_aware, round_robin, power_of_two, consistent_hashing, prefix_hash, manual, random, bucket. Source: [docs.rs smg](https://docs.rs/crate/smg/1.3.2).
- **Header-based stickiness** [C, secondary source]: the `X-SMG-Routing-Key` header belongs to the `manual` policy, and `consistent_hashing` gives "session affinity via hash ring". Source: [DeepWiki smg config](https://deepwiki.com/lightseekorg/smg/8.1-router-configuration-reference).
- **Our wheelhouse:** `D:/ARC AGI 2/_scratch/e13_zip/wheelhouse_0_5_21/SHA256SUMS` (225 lines) and `sglang_0_5_21_requirements.lock` contain **no** `sglang-router` or `smg` wheel [M, local grep].
- **Two-server ports:**
  - Each server builds its ZMQ channels from fresh temp-file IPC names when DP attention is off.
  - The NCCL port comes from `get_free_port()` unless `--nccl-port` is set (`SRC/server_args.py` L677–681, L709–721).
  - So two servers on different `--port` values do not collide by design. A race on `get_free_port` is possible, and pinning `--nccl-port` per server avoids it [E].
- **Router overhead:** the DGX Spark study measured "negligible observable overhead" for the SGLang round-robin router [C] — [NVIDIA forum 381102](https://forums.developer.nvidia.com/t/comprehensive-qwen3-8-27b-study-on-dgx-sparks-quantization-speculative-decoding-and-tp-dp-scaling/381102).

### Inferences
- **For the Duck agent, stickiness belongs in the client** [E]. The harness already knows each branch's identity. Assigning `branch_id % 2` (or the replica with the fewest resident tokens at branch creation) to one port keeps every turn of a branch on the replica that holds its radix prefix.
  - The router's `cache_aware` would likely do the same through prefix matching, since Duck turns share long prefixes.
  - But it adds a wheel, a process, and default retries that could silently re-run a 60k-token generation after a 5xx.
  - If the router is used: `--disable-retries` and `--policy manual` with `X-SMG-Routing-Key`, or `cache_aware`.
- **Per-replica monitoring:** two `/metrics` endpoints (`--enable-metrics` on each), each with its own `#running-req`, `#queue-req` and token usage. The real-concurrency assertion (memory note "verify real concurrency") must be applied per replica.
  - `/get_server_info` misreports `max_running_requests` [M `OURS`], so read the startup "capped" line from each log.
- **Admission bug:** the 11-of-12 bug (a chunked prefill counted twice) applies per replica. A replica sized for 6 would run 5 unless the server cap is set one higher, as with today's 13-for-12 workaround [E from `OURS`].

### Gaps
- Not verified that `routed_dp_rank` works with the hybrid mamba radix cache in `--dp-size` mode on 0.5.21.
- No source states how `cache_aware` treats requests longer than the tree's prefix granularity, or the `X-SMG-Routing-Key` header name, from primary docs.

## Custom all-reduce at TP2 on PCIe-only L4s (source reading)

### Takeaway
At TP2, SGLang 0.5.21 rejects **CustomAllReduceV2**, which requires full NVLink. It falls back to the **legacy CustomAllreduce**, which *is* allowed on 2 PCIe-only GPUs, with an 8 MiB size cap.

The legacy path needs working P2P. But **by default SGLang skips the real P2P test** (`--enable-p2p-check` is False, which monkey-patches the check to return True). On a Kaggle VM where P2P may not work, TP2 would therefore try custom AR blindly. **Use `--enable-p2p-check`** (or `--disable-custom-all-reduce`) on any TP2 launch.

The PCIe-IPC FlashInfer all-reduce in 0.5.21 needs a FlashInfer newer than our 0.6.18.

### Cited Findings (all [M, source] unless marked)
- **Class dispatch** (`SRC/distributed/device_communicators/custom_all_reduce.py` L344–369): on CUDA with `SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2` (default True, `SRC/environ.py` L1347), V2 is tried first. V2 is used only if `can_use_custom_all_reduce_v2` returns `full_nvlink is True` (`custom_all_reduce_v2.py` L484–497). Otherwise SGLang returns the legacy `CustomAllreduce`.
- **Gate** (`custom_all_reduce_utils.py` L466–512, `can_use_custom_all_reduce_with_nvlink`):
  - "if world_size > 2 and not full_nvlink: … disabled because it's not supported on more than two PCIe-only GPUs";
  - then "if not _is_hip and not can_p2p(rank, world_size): … disabled because your platform lacks GPU P2P capability or P2P test failed".
  - At TP2 PCIe the gate returns `False` (not None), so the legacy CA is constructed.
- **Use at TP2** (`custom_all_reduce.py` L258–273): "for 4 or more non NVLink-capable GPUs, custom allreduce provides little performance improvement over NCCL"; `if self.world_size == 2 or self.full_nvlink: return inp_size <= self.max_size`. `_MAX_CAR_SIZE = 8192 * 1024` (L42). Our ~0.5 MB decode/verify all-reduces at TP2 qualify [D].
- **P2P check default:**
  - `SRC/distributed/bootstrap.py` L121–122: `if not get_parallel().enable_p2p_check: monkey_patch_p2p_access_check()`;
  - `SRC/utils/common.py` L2400–2408: "We assume the p2p access is always allowed, which can be wrong for some setups" sets `gpu_p2p_access_check` to `lambda: True`;
  - the field help says "Enable P2P check for GPU access, otherwise the p2p access is allowed by default" (`SRC/arg_groups/fields/parallel.py` L236–239);
  - our W1 launch had `'enable_p2p_check': False` [M `W1LOG` L8].
- **Why a real test exists:** `can_actually_p2p` exists because `torch.cuda.can_device_access_peer` "returns True even if P2P access is not actually possible". Custom AR combines "p2p and cuda IPC … That's why we need p2p access" (`custom_all_reduce_utils.py` L149–175, citing vLLM issue 2728).
- **Our TP4 logs**, as expected: "CustomAllReduceV2 is disabled …" and "CustomAllreduce is disabled because it's not supported on more than two PCIe-only GPUs", on every rank [M `W1LOG` L33–40; the same in E3 Flash-Next logs].
- **FlashInfer PCIe-IPC all-reduce** (`SRC/distributed/device_communicators/pcie_ipc_ar.py`):
  - enabled by `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE` (default False, `environ.py` L1354);
  - world sizes (2, 4, 8), "tp" group only, bf16 only;
  - workspace sized `max_bs × draft_tokens × hidden` for decode;
  - autotune skipped when `--disable-flashinfer-autotune` is set (our flag);
  - docstring measurement [M by SGLang authors, 8× SM120]: 12 KiB 5.3 µs vs NCCL 205.1 µs; TP8 TPOT 21.14 → 13.62 ms.
  - It fails with "this FlashInfer build has no pcie_ipc_comm" if FlashInfer lacks the module.
- **Our FlashInfer:** `flashinfer_python-0.6.18` (`D:/ARC AGI 2/_scratch/e13_validate/flashinfer_src/fi.whl`) has **no** `pcie_ipc` code. `flashinfer/comm/` contains `cuda_ipc.py`, `vllm_ar.py`, `trtllm_ar.py` and others, but no PCIe-IPC module [M, local zip scan]. The lock pins `flashinfer-python==0.6.18` [M, lockfile].
- **Kaggle P2P evidence:**
  - Kaggle 2×T4 VM: GPUs `PHB`, `intraNodeP2pSupport 0`, NCCL "via SHM/direct/direct", busBW plateau 3.97 (ring) to 4.58 (tree) GB/s, small-message α about 14–16 µs [M] — [nccl-allreduce-bench](https://github.com/sxwangsxwang1/nccl-allreduce-bench).
  - Our TP4 preflight "NCCL at 64 MB in 20.9 ms" ≈ 3.06 GB/s algbw ≈ 4.6 GB/s busbw [M `OURS`]. That matches SHM, not P2P.
  - The sibling note `nccl_p2p_l4.md` cites a GKE 2×L4 node at PHB with P2P OK, and nvbandwidth on g2-standard-48 at about 10 GB/s P2P [C, see that note]. So L4 P2P on GCP exists, but not proven inside Kaggle's VM.
- **Measured custom-AR gains on 3090s** [M]:
  - TP2 with custom AR and verified P2P at 220 W beat a no-P2P TP2 reference by about +15.7 % decode;
  - forced CA at TP4 gave +14–15 % decode.
  - Source: [club-3090 #773](https://github.com/noonghunna/club-3090/discussions/773).

### Inferences
- If Kaggle's L4 VM has working P2P, TP2 gets custom AR automatically, which should be worth roughly +10–15 % per replica decode [E from club-3090].
- If P2P is unavailable, the blind default could:
  - (a) raise during IPC setup and fall back to NCCL, with the warning "Setup Custom allreduce failed …";
  - (b) at worst hang or corrupt reductions (vLLM #2728's rationale) [E].
- `--enable-p2p-check` makes the test real; it is cached per `CUDA_VISIBLE_DEVICES` string.
- PCIe-IPC needs P2P as well, because it relies on CUDA IPC peer mapping like custom AR [E]. On Kaggle it also needs a newer FlashInfer wheel. It applies equally to TP4, so it is a TP4 lever too.

### Gaps
- Kaggle 4×L4 `nvidia-smi topo -m` / `-p2p r` output and `NCCL_DEBUG=INFO` transport lines have never been captured.
- I did not verify which FlashInfer version first ships `PcieIpcAllReduceWorkspace`, or whether it supports sm_89.

## Other costs: cold starts, weight reads, host RAM

### Takeaway
Two replicas cost about **one extra cached launch (~1.3 min)** if they start staggered: the second replica reuses the first replica's JIT cache. Run in parallel, they likely both pay the full ~6 min graph JIT [E].

Weight reads do not double if the files are pre-warmed into the page cache, which is shared across processes. Our vLLM prefetch already showed a cut from 64 s to 4.8 s. Host RAM (202.6 GB) easily holds both the target (~19.5 GB) and the drafter (~3.85 GB).

### Cited Findings
- **SGLang W1 cold start on Kaggle** [M `OURS` Startup]:
  - ready in about 520–680 s; target-verify CUDA graph about 372–385 s; weight load 13–88 s;
  - one slow-disk run: load_weight 669.96 s, ready 1,197 s (G4B);
  - later launches in the same session with a warm kernel cache: W2 71 s, W3 76 s (target_verify 11.94 s); E11-03 B/C/D 136–166 s.
- **vLLM weight prefetch:** "weight load fell from 64 s to 4.8 s, and engine start from 600 s to 532 s" [M `OURS`, JDL L136].
- **Triton serving-time compile stalls:** "Triton kernel 'chunk_gated_delta_rule_fwd_kkt_solve_kernel' took 1.77 s to compile after serving started" (6 lines) [M `W1LOG`]. The kernels are shape-keyed.
- **P2P cache file** is keyed by `CUDA_VISIBLE_DEVICES` (`gpu_p2p_access_cache_for_{cuda_visible_devices}.json`, `custom_all_reduce_utils.py` L240–270) [M, source]. Each replica writes its own.

### Inferences
- **TP2 kernel shapes differ from TP4's** (twice the per-GPU N/K), so existing TP4 JIT caches do not help [E]. Replica B started after A's capture (about 6–7 min) should behave like our cached relaunches (about 70–80 s).
  - Staggered total ≈ 8–9 min, vs about 9.5 min for TP4 W1.
  - Duck can start sending to replica A as soon as it is ready.
- **Parallel start:** two processes JIT-compiling the same kernels may duplicate work and contend for CPU. Without file locking guarantees in the Triton or FlashInfer caches, staggering is safer [E].
- **Weights:** both replicas read the same safetensors. At TP4, 4 ranks already read the same files, so 2×TP2 does not add reads. Warming the page cache first (`cat` the files to `/dev/null`) makes the second replica's reads RAM-speed [E].

### Gaps
- Whether `/kaggle/input` is a GCSFUSE mount or a local or persistent disk is unverified. The G4B 670 s load shows the variance, not the mechanism.
- No measurement exists of two simultaneous SGLang JIT captures on Kaggle.

## Alternatives: PP2×TP2, DP-attention, TP4 with cheaper all-reduce or a smaller verify width

### Takeaway
Two alternatives are blocked by source checks:
- **PP2×TP2 with DFlash2** — "Currently DFLASH speculative decoding only supports pp_size == 1";
- **DP-attention with DFlash2** — not supported on non-NPU devices, and DP-attention targets MLA models anyway.

The better-evidenced levers keep TP4's KV pool and attack the all-reduce:
- (1) verify and enable P2P (`NCCL_P2P_LEVEL`): +2–16 % decode measured on PHB consumer cards;
- (2) a newer FlashInfer with PCIe-IPC all-reduce at TP4, if P2P exists;
- (3) a patched custom AR at TP4: +14–15 % on 3090s, with greedy divergence.

I found no evidence on a smaller DFlash2 verify width.

### Cited Findings
- **DFlash2 restrictions** [M, source `SRC/arg_groups/speculative_hook.py`]:
  - L274–278: "Currently DFLASH speculative decoding does not support dp attention on non-NPU devices";
  - L280–283: "Currently DFLASH speculative decoding only supports pp_size == 1".
  - DSpark PP requires PD-prefill disaggregation (L629–634).
- **PP memory, if it were allowed** [E]: per-GPU KV per token would equal TP4 (8 KiB, half the layers × half the heads), and the all-reduce would be 2-rank. It would be memory-equivalent to TP4 with cheaper communication, but DFlash2 cannot use it.
- **DP-attention** is meant for MLA, where "TP duplicates the KV cache across GPUs". For standard GQA models, "standard DP or TP is typically recommended" [C] — [SGLang DP guide](https://docs.sglang.io/advanced_features/dp_dpa_smg_guide.html).
- **NCCL P2P on PHB** (TP4): +6.1 % (c1), +10.1 % (c2), +13.3 % (c4), +2.0 % (c8) [M] — [HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254).
- **Forced custom AR at TP4** (3090s with P2P): +14.8 % decode at c1, +14.3 % at c16, TTFT +9 % at c16, greedy divergence on 4/5 prompts [M] — [club-3090 #773](https://github.com/noonghunna/club-3090/discussions/773).
- **FlashInfer PCIe-IPC** decode-sized workspace at TP8, 8k context: TPOT 21.14 → 13.62 ms and output 35.02 → 48.05 tok/s; a prefill-sized workspace "costs 66 % on TTFT" [M by SGLang authors, docstring `SRC/distributed/device_communicators/pcie_ipc_ar.py` L1–60].

### Inferences
- **Ranking by evidence and fit** [E]:
  1. Capture topology and NCCL transport on Kaggle 4×L4. If P2P works, set `NCCL_P2P_LEVEL=SYS`/`PHB` at TP4 (no memory cost) and expect about +5–15 %.
  2. If P2P works, try PCIe-IPC (FlashInfer upgrade) or a patched custom AR at TP4, with a correctness gate (greedy and logprob parity).
  3. 2×TP2 only if the stream count drops to 6–8 per run or contexts shrink. Neither is planned: the 60k thinking floor is fixed (memory note "No token cuts below 60k").
- **A smaller DFlash2 verify width** would cut intermediate state and all-reduce bytes per step at TP4 alike. Whether DFlash2 supports it and what it does to acceptance are open (see `dflash_width_correctness.md`).

### Gaps
- No published TP4 vs PP2×TP2 measurement for hybrid GDN models.
- No published DFlash2 result at a verify width below 8.
- No published sm_89 (L4) result for PCIe-IPC or forced custom AR.
