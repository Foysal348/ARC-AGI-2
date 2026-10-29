# Serving engines for Qwen3.5/3.8 hybrids on Ada, the Qwen3.8-Flash-Next profile, and Daniel Franzen's Duck harness (state as of 2026-10-29)

Scope note: this file adds to, and does not repeat in full, the earlier notes in `D:/ARC AGI 2/research_notes/Qwen 27B দ্রুততম সার্ভিং সেটআপ/` (mainly `community_evidence.md` §2–4, `speculative_decoding.md`, `engines_and_config.md`) and `D:/ARC AGI 2/research_notes/৪xL4 ইঞ্জিন বটলনেক যাচাই/` (mainly `newer_engine_versions.md`, `agent_effective_throughput.md`, `startup_load_environment.md`). Where an older fact is needed for the argument, it is restated in one line with its source. Labels: **[V]** verified from a primary source (code, issue, model card, API); **[C]** claim by a named person, not independently checked; **[R]** rumor / opinion; **[I]** my inference.

---

## Q1. Engine landscape: which engine/version supports the Qwen3.5/3.8 hybrid (Gated DeltaNet + gated attention, and Flash-Next's QSA + MoE + PLE) and its MTP, and what Ada (L4 / L40S / 4090 / RTX 6000 Ada) benchmarks at concurrency 16+ exist?

### Takeaway
As of 2026-10-29, the newest releases are vLLM 0.32.0 (10-18), SGLang 0.5.22 (10-16), TensorRT-LLM 1.4.0rc2.post1 (10-28), ExLlamaV3 1.6.5 (10-25), LMDeploy 0.18.0 (09-28), and llama.cpp builds after b10660. vLLM and SGLang are the only engines with mature support for both the 27B and Flash-Next, including MTP. TensorRT-LLM lists both, but its Flash-Next multimodal path is "Untested". No primary source gives a Qwen3.8 throughput number on an L4 or L40S at concurrency 16 or more. The only Ada multi-GPU number at c16 is a community vLLM 0.30 recipe on 3x RTX 4090 48 GB: Flash-Next NVFP4 at 635 tok/s with 16 streams. For the 27B on our own 4x L4, the best measured numbers are still our own: about 373–431 tok/s at 16–20 streams.

### Cited Findings
**Release versions (PyPI upload dates, read 2026-10-29) [V]**
- vLLM: 0.29.0 (09-09), 0.30.0 (09-22), 0.31.0 (10-05), 0.32.0 (10-18) — [PyPI vllm](https://pypi.org/project/vllm/).
- SGLang: 0.5.19 (09-04), 0.5.20 (09-18), 0.5.21 (10-01), 0.5.22 (10-16) — [PyPI sglang](https://pypi.org/project/sglang/).
- TensorRT-LLM: 1.3.0rc29 (09-29), 1.4.0rc0 (10-13), 1.4.0rc1 (10-21), 1.4.0rc2.post1 (10-28). Release candidates only — [PyPI tensorrt-llm](https://pypi.org/project/tensorrt-llm/).
- ExLlamaV3: 1.6.2 (10-14) through 1.6.5 (10-25) — [PyPI exllamav3](https://pypi.org/project/exllamav3/).
- LMDeploy: 0.18.0 (09-28) — [PyPI lmdeploy](https://pypi.org/project/lmdeploy/).

**vLLM [V]**
- The Flash-Next recipe requires "vLLM 0.30.0+". It says: "Stable ≥0.30.0 runs the model, but this recipe is tuned and verified on nightly." — [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next).
- The recipe's hardware list is GB300, GB200, 8x H200, 8x H100 and MI355X. Ada, L4, L40S, RTX 4090 and sm89 are not mentioned. The only throughput table is MXFP4 on MI355X: 665 / 1,831 tok/s at concurrency 8 / 32 (TEP8) — [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next).
- Recommended Flash-Next flags: `--enable-expert-parallel`, `--enable-prefix-caching`, `--kv-cache-dtype fp8`, `--attention-config.indexer_kv_dtype fp8`, MTP `{"method":"mtp","num_speculative_tokens":3}`, `--tool-call-parser qwen3_coder --reasoning-parser qwen3`. PLE CPU offload is on by default (`--engram-config '{"cpu_offload":true}'`) and needs ≥ 51 GB of host RAM. Plain TP8 is incompatible with the FP8 checkpoint's 128-wide blocks — [vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next).
- v0.32.0 regression, now closed: Flash-Next checkpoints "with quantized q/k/v and an unquantized QSA indexer fail to load since #59533 (v0.32.0)". Opened 10-18, closed 10-27 — [vllm#62308](https://github.com/vllm-project/vllm/issues/62308).
- Open on v0.32.0 and nightly: "single prefill crossing the 262,144 native boundary crashes the engine with CUDA illegal memory access (… OK on v0.31.1)" — [vllm#62383](https://github.com/vllm-project/vllm/issues/62383).
- Closed 10-27: the QSA indexer prefill workspace (512 MiB, `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB`) was not seen by the memory profiler, so the first long prefill OOMed. This was on 4x RTX 3090 TP4+EP with `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`. Workaround: `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128` — [vllm#60489](https://github.com/vllm-project/vllm/issues/60489).
- Closed 10-05: AutoRound/INC Flash-Next checkpoints failed to load because "PLE rejects INCConfig even though PLE is unquantized" — [vllm#59798](https://github.com/vllm-project/vllm/issues/59798).
- Closed 10-06: "PLE pinned-host FP8 lookup will not compile on sm_86", so `--engram-config cpu_offload` was unusable on consumer Ampere. SM89 status is not stated — [vllm#59868](https://github.com/vllm-project/vllm/issues/59868).
- Open: a tracking issue for Flash-Next optimizations ([vllm#55922](https://github.com/vllm-project/vllm/issues/55922)), and "qwen4_exp cannot serve DeepSpec DFlash/DSpark drafters: five blockers" ([vllm#56088](https://github.com/vllm-project/vllm/issues/56088)). DFlash2 is therefore a 27B lever, not a Flash-Next lever on vLLM.
- For the 27B on Ada: the MTP scheduler cap ([vllm#55533](https://github.com/vllm-project/vllm/issues/55533), RTX 4090 D: bs20 649.0 tok/s without spec vs 210.9 with MTP) and the FP8 draft-KV block on SM89 (#54690) are restated from `newer_engine_versions.md` §Q2. Our E19 test showed MTP with prefix caching off fails correctness (memory `experiment-19-vllm-correctness.md`).

**SGLang [V]**
- 0.5.22 is the latest release. Release-note numbers for Flash-Next are B200 only (#42175: 242 → 271 tok/s at c1 on 4x B200). No entry targets SM89 for W4A16 Marlin, Triton GDN or QSA (restated from `newer_engine_versions.md` §Q1).
- The SGLang Flash-Next cookbook says NVFP4 is Blackwell-only and needs SGLang ≥ 0.5.22. On 1x RTX PRO 6000: 613 tok/s at 16 requests with MTP, 861 at 64 without (restated from `community_evidence.md` §2) — [cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-Flash-Next).
- The PR "[Fix] qsa: Fail early on SM89 without classic FA2" is still OPEN, rebased 10-26. It says the main-branch resolver "still accepts an importable FA4 fallback on SM89 when classic FA2 is missing". The failure chain on four L20s (SM89) with the FP8 checkpoint: "FlashInfer TRTLLM resolver -> unavailable on SM89 / classic flash_attn / FA2 -> not importable / flash_attn.cute / FA4 -> importable, so the resolver selects it / first sparse decode -> CuTe MLIR operation creation failure" — [sglang#36968](https://github.com/sgl-project/sglang/pull/36968).
- A list of Flash-Next QSA issues still OPEN or recently closed (GitHub search, 10-29):
  - QSA + NEXTN decode graph silently corrupts output into `!` on GB10 ([#37111](https://github.com/sgl-project/sglang/issues/37111));
  - NEXTN acceptance decays to ~0 over uptime and is restored only by a restart ([#37326](https://github.com/sgl-project/sglang/issues/37326));
  - a chunked-prefill + radix-insert race corrupts KV pages ([#38319](https://github.com/sgl-project/sglang/issues/38319));
  - an IMA in QSA extend at 8 concurrent requests on H20 ([#37633](https://github.com/sgl-project/sglang/issues/37633));
  - FP8 KV crashes QSA (`tl.dot` rejects bf16 q x fp8 k) on SM120 ([#36545](https://github.com/sgl-project/sglang/issues/36545) OPEN; duplicate [#43741](https://github.com/sgl-project/sglang/issues/43741) closed 10-12, fixed by #36644);
  - PD decode resumes with another request's QSA indexer state after retraction ([#45864](https://github.com/sgl-project/sglang/issues/45864), 10-22);
  - W4A16 on A100 TP4: Marlin MoE "Invalid thread config" for MKN [64, 2560, 320], then the Triton MoE path hits the QSA FA-CuTe capture failure, with Triton warnings "Data race detected: `Logits(bx, position)`" ([#37089](https://github.com/sgl-project/sglang/issues/37089); Marlin N=64 fix PR #37946).

**TensorRT-LLM [V]**
- The supported-models page lists `Qwen4ExpForCausalLM` ("Qwen3.8-Flash-Next (text)") with CUDA Graph, Chunked Prefill, KV Cache Reuse and Speculative Decoding = MTP all "Yes". `Qwen4ExpForConditionalGeneration` has Overlap Scheduler, CUDA Graph and Chunked Prefill "Untested". `Qwen3_5MoeForCausalLM` covers Qwen3.8-MoE / 3.5-MoE. No GPU-architecture restriction or per-model quantization support is listed — [TRT-LLM supported models](https://nvidia.github.io/TensorRT-LLM/models/supported-models.html).
- A third-party RTX 5090 test of the 27B NVFP4 found vLLM 57% faster than TRT-LLM at 32 concurrent requests [C] — [computingforgeeks](https://computingforgeeks.com/tensorrt-llm-vs-vllm/).

**ExLlamaV3 [V/C]**
- The README lists Qwen 3.5, Qwen 3.5 MoE and "Qwen 3.8-Flash-Next", with TP/EP, continuous batching and speculative decoding. No throughput numbers (restated from `engines_and_config.md`) — [ExLlamaV3 README](https://raw.githubusercontent.com/turboderp-org/exllamav3/master/README.md).
- A community EXL3 card says qwen3_5 MTP works upstream, but DFlash2/DSpark need a fork [C] — [Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw](https://huggingface.co/Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw).
- Flash-Next EXL3 quants exist: `doth4580/Qwen3.8-Flash-Next-EXL3-4.05bpw` and `-3.05bpw` [V] (HF API search, 10-29).

**LMDeploy [V]**
- The README lists Qwen3.5 (added Feb 2026). I found no Qwen3.8 entry — [LMDeploy README](https://github.com/InternLM/lmdeploy).

**llama.cpp / ik_llama.cpp [V/C]**
- Mainline llama.cpp added the `qwen4exp` architecture in PR #27742 (release b10660). It is described as "initial … optimization improvements are still pending" — [b10660 release](https://newreleases.io/project/github/ggml-org/llama.cpp/release/b10660).
- ik_llama.cpp reportedly merged qwen4exp MTP (PR #2369) [C, Reddit via aggregator] — [prismix summary](https://prismix.dev/news/6bf14fe7cba2).
- Single-RTX-4090 llama.cpp-family numbers are single-slot only, at about 72–92 tok/s decode (see Q3). No c16 data exists.

**Ada benchmarks at concurrency ≥ 16 [V for the source, C for the numbers]**
- 3x RTX 4090 48 GB (SM89), vLLM 0.30.0 with about 15 patches, TP3 + EP3, Flash-Next NVFP4 experts via MARLIN, FP8 side layers, FP8 KV, PLE in pinned host RAM. FP8 MTP=2: **124.3 / 387.4 / 617.6 / 635.3 tok/s at 1 / 4 / 8 / 16 streams**. MTP depth at 16 streams: K0 559.0, K1 636.4, K2 644.5, K3 541.1 ("MTP=3 loses at 16-way"). It needs about 28 GiB per card, and "24GB cards: OOM at load" — [chenxin199305/qwen38-flash-next-3x4090](https://github.com/chenxin199305/qwen38-flash-next-3x4090).
  - The patches fixed real Flash-Next defects that cause garbage. "MARLIN multiplied top-k weights inside the kernel even with `expert_map`", which produced "garbage or repeated tokens"; relative error went from 0.47–1.16 to 0.0016 after patch 02k. A Mamba slot seeded with the wrong block size gave wrong output on prefix hits (02h). "No MTP speedup" because draft metadata was rebuilt every step (02g, ports #58449).
- I found no Qwen3.8 benchmark on an L40S or an RTX 6000 Ada under concurrent load (search 10-29).
- The 27B on Ada at c≥16, restated from earlier notes:
  - vllm#55533, RTX 4090 D INT4: bs20 649 tok/s without spec;
  - HyperQwen, RTX 3090 (Ampere), patched vLLM 0.31: ~1,035 tok/s at 64 concurrent;
  - club-3090 #1288, 4x 3090 SGLang: c16 927.5 narrative / 1,811 code tok/s.

**Our own 4x L4 measurements (local, [V])**
- 27B on SGLang: about 373 tok/s at 16 running; range 321–431 (`Qwen 27B দ্রুততম সার্ভিং সেটআপ/our_measurements.md`).
- 27B on vLLM 0.32 (E18 smoke v1): 382 tok/s steady at 20 streams, 0 preemptions (memory `experiment-18-s1-vllm032.md`).

### Inferences
- [I] For Flash-Next on Ada, the engine choice comes down to vLLM nightly/0.32 with patches versus SGLang main with the QSA FA2 path. Neither has an Ada-validated Flash-Next path upstream:
  - In vLLM, the only working Ada Flash-Next proof (3x 4090) needed patches to MARLIN EP, the Mamba slot size, the MTP metadata and FP8 side layers.
  - In SGLang, SM89 QSA decode must go through classic FA2, and the guard PR is not merged.
- [I] The bandwidth of 3x 4090 48 GB (~3 x 1.0 TB/s) is about 2.5x that of 4x L4 (4 x 300 GB/s). The memory is 144 GB vs 96 GB. Scaling 635 tok/s at c16 by bandwidth gives very roughly 250 tok/s on 4x L4. That is before the L4's much smaller KV pool (Q3) cuts concurrency. This is a rough estimate, not a measurement.
- [I] TensorRT-LLM, LMDeploy and ExLlamaV3 are not practical drop-ins for our Kaggle 4x L4 image within days: TRT-LLM has only release candidates, LMDeploy has no Qwen3.8 entry, and ExLlamaV3 has no batch-serving numbers. vLLM 0.32 and SGLang 0.5.22/main stay the only realistic choices.

### Gaps
- No primary-source Qwen3.8 (27B or Flash-Next) throughput on L4, L40S or RTX 6000 Ada at c ≥ 16 in any engine.
- I did not verify whether ik_llama.cpp implements QSA and PLE itself, or ExLlamaV3's PCIe TP performance.
- TRT-LLM Ada support for `Qwen4Exp` kernels (QSA, NVFP4/W4A16 MoE) is not stated anywhere I found.

---

## Q2. Qwen3.8-Flash-Next profile: architecture, total/active parameters, layer types, context, quantizations, memory footprint, recommended serving flags

### Takeaway
Flash-Next (`Qwen4ExpForConditionalGeneration`, model_type `qwen4_exp`) is an ultra-sparse MoE: 125B main parameters with about 6B active, plus a separate 51B n-gram (PLE) embedding table and a 4B MTP head (≈176B on disk). It has 48 layers in a 3:1 pattern: Gated DeltaNet (linear attention) then Qwen Sparse Attention (QSA, a full-attention layer with a top-k indexer), each followed by a 512-expert MoE (10 routed + 1 shared). Native context is 262,144. Every 4-bit checkpoint is 100–181 GB on disk because the PLE table dominates. On GPU, the W4A16 main weights take about 74 GB.

### Cited Findings
**Architecture, from `config.json` [V]** — [Qwen/Qwen3.8-Flash-Next config](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/config.json), fetched 10-29.
- Core shape:
  - `architectures`: `Qwen4ExpForConditionalGeneration`; text `model_type` `qwen4_exp_text`;
  - `num_hidden_layers` 48; `full_attention_interval` 4; `layer_types` = 3 x `linear_attention` then 1 x `full_attention`, repeated 12 times;
  - `hidden_size` 2560; `vocab_size` 248,320.
- Full-attention (QSA) layers:
  - `num_attention_heads` 24, `num_key_value_heads` **2**, `head_dim` 256, `partial_rotary_factor` 0.25, `output_gate_type` sigmoid;
  - sparse indexer: `indexer_n_heads` 4, `indexer_kv_heads` 1, `indexer_head_dim` 128, `indexer_compress_ratio` 4, `indexer_budget` 2048.
- Linear-attention (GDN) layers: `linear_num_key_heads` 16, `linear_num_value_heads` 48, key/value head dim 128, conv kernel 4; `mamba_ssm_dtype` float32 (the default state dtype).
- MoE: `num_experts` 512, `num_experts_per_tok` 10, `moe_intermediate_size` 640, `shared_expert_intermediate_size` 640.
- PLE n-gram embedding: `ngram_size` 3, `ngram_vocab_size_base` 20,000,000, `heads_per_ngram` 8, `ple_embed_dim` 2560, `ple_layer_ids` [2].
- Other: `hc_count` 4 and `hc_lowrank` 320 (hyper-connections; vLLM/SGLang issues call them "mHC"); `mtp_num_hidden_layers` 1; `max_position_embeddings` 262,144.

**Model card claims (restated from `community_evidence.md` §3, [C] Qwen)**
- "125B with 6B activated, plus 51B n-gram embedding and 4B MTP".
- Benchmarks vs the 27B: GPQA 91.7 vs 89.2; HLE 35.9 vs 30.8.
- vLLM's announcement: "125B params, 6B active, 262K native, 1M via YaRN … a separate 51B N-gram table you can offload" — [vLLM on X](https://x.com/vllm_project/status/2092600887873286157).
- The 3x4090 repo describes the NVFP4 model as "~176B total parameters, ~6B active, 512 experts, 48 layers, 1 MTP layer" — [chenxin repo](https://github.com/chenxin199305/qwen38-flash-next-3x4090).

**Official and community checkpoints, with on-disk safetensors size (HF API, 10-29) [V]**
- `Qwen/Qwen3.8-Flash-Next` (BF16; 335.28 GiB per the vLLM recipe); 3.64 M downloads.
- `Qwen/Qwen3.8-Flash-Next-FP8`: 185.5 GB, 131 shards (172.78 GiB per the vLLM recipe).
- `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound`: 181.2 GB, 17 shards. Experts-only int4, BF16 PLE kept. 99.64% average recovery [C].
- `albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE`: 128.9 GB, 27 shards. W4A16 with an FP8 PLE table. This is Franzen's drafter source.
- `RadixArk/Qwen3.8-Flash-Next-NVFP4`: 135.2 GB, 206 shards.
- `nvidia/Qwen3.8-Flash-Next-NVFP4` (B200/B300 target).
- `nvidia/Qwen3.8-Flash-Next-FP4-Local`: 102.2 GB (2026-10-08/15). QAD-trained NVFP4 routed experts plus MXFP8 attention. "upstream vLLM support for this format has not yet landed". **Blackwell only** (RTX PRO 6000, DGX Spark). GPQA-D 91.5 vs BF16 92.2 [C] — [card](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local).
- `davetha/Qwen3.8-Flash-Next-DERISKED-W4A16-AWQ`: 179.8 GB.
- Also present: `Inferact/…-NVFP4`, `primitive-ai/…-NVFP4` (mixed NVFP4-FP8; lordhansolo's M2 entry), `local-inference-lab/…-NVFP4`, `amd/…-Quark-MXFP4`, many GGUF builds (unsloth, bartowski, ggml-org, ISTA-DASLab GSQ-RCO with 7.4 M downloads), REAP-pruned `sh0wie/Qwen3.8-Flash-Next-REAP-288-GGUF` (288 of 512 experts), and EXL3 4.05 / 3.05 bpw.
- Distilled or efficient variants: `ukisai/Swift-1.5-Qwen3.8-Flash-Next-*`, `bottlecapai/ThinkingCap-Qwen3.8-Flash-Next` (restated from `token_efficient_variants.md`).

**GPU-resident footprint [V, measured]**
- Franzen W4A16 + BF16 PLE: 73.64 GB on GPU, 95.37 GB offloaded to host.
- lordhansolo mixed NVFP4-FP8: 71.94 GB GPU, 96.55 GB host.
- Restated from `community_evidence.md` §2 (Tong Hui Kang's table, ARC-AGI-3 topic 744792).
- On our 4x L4, Intel W4A16 loads at 19.06–19.10 GB per rank (E3, E13 LOADPROBE). The KV pool is then only **94,528 fp8 tokens** (LOADPROBE bf16ple arm) to **118k** (E3) — memory `e13-flashnext-smoke1-oom.md`; `community_evidence.md` §4.

**Serving flag recommendations**
- vLLM: see Q1. TEP (TP + `--enable-expert-parallel`), FP8 KV, `indexer_kv_dtype fp8`, MTP 3, PLE CPU offload.
- SGLang (Franzen, single RTX PRO 6000):
  - `--quantization auto-round --kv-cache-dtype fp8_e4m3 --page-size 64 --mamba-ssm-dtype bfloat16 --max-mamba-cache-size 60 --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64 --moe-runner-backend auto --ple-offload-embedding`;
  - `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`;
  - `--schedule-policy lpm --reasoning-parser qwen3 --tool-call-parser qwen3_coder`;
  - full command in Q4.
- SGLang hard limits:
  - "`speculative_num_draft_tokens > 4` is rejected by the compress ratio guard" ([sglang#36558](https://github.com/sgl-project/sglang/issues/36558));
  - draft tokens greater than the compress ratio, or topk > 1, crash late in init ([sglang#44099](https://github.com/sgl-project/sglang/issues/44099));
  - in vLLM, "QSA ring assert makes num_speculative_tokens 5..8 unreachable" ([vllm#54552](https://github.com/vllm-project/vllm/issues/54552)).
- Parallelism: "EP4 needed: TP4 MoE split 640/4 = 160 is not a multiple of group 128" (our E13 VALIDATION row 5, restated). The 3x4090 repo uses EP3 for the same reason (640 is not 128-aligned for 3-way TP).

### Inferences
- [I] Activated compute is small (about 6B), but the expert set touched per decode step grows with batch: about 27% of each layer's experts at batch 16 (restated from `community_evidence.md` §4). So at ARC-style 10–20 streams, the bytes read per step per GPU are of the same order as the 27B INT4 under TP4. The "6B active" speed advantage mostly holds at low concurrency or on high-bandwidth single GPUs.
- [I] The KV need per token is small (only 12 of 48 layers hold KV, with 2 KV heads). On 4x L4 the KV pool is still tiny because about 74 GB of the 96 GB goes to weights. With 94–118k pool tokens, only about 1–2 ARC-AGI-2 streams of 60k+ tokens fit at once, against the 12–20 streams the 27B INT4 sustains. This is the binding constraint, more than kernel speed.

### Gaps
- No official Qwen statement on the PLE/hyper-connection design beyond the config and the model card. I did not fetch the full model-card text again (an earlier subagent did; see `community_evidence.md` §3).
- No measured accuracy of W4A16 vs BF16 on ARC tasks. Intel's 99.64% recovery is on general benchmarks.

---

## Q3. Known Flash-Next problems on L4/Ada (Triton/kernels, NVFP4 on sm89, garbage "!!!!" output), measured tok/s, and speed vs Qwen3.8-27B on 4x L4

### Takeaway
On SM89, upstream SGLang has no validated QSA sparse-decode kernel. The TRTLLM path is gated to SM100/SM120, classic FA2 is the only correct SM89 path, and the FA4/CuTe fallback fails. Every known "!!!!" (token id 0) report on Flash-Next comes from a kernel defect that silently produces NaN or zero logits:
- the QSA decode fallback on SM120/SM121;
- a PDL race in the MoE router bias (SM ≥ 9.0 only);
- MARLIN EP top-k scaling in vLLM;
- QSA + NEXTN decode CUDA graphs on GB10.

Our 4x L4 "!!!!" (E3 and E13 LOADPROBE) matches this failure class, but its exact cause on SM89 is not identified. NVFP4 does run on Ada in a patched vLLM (weight-only via MARLIN). But it needs about 28 GiB per card, so it cannot fit 24 GB L4s. No one has published a correct, fast Flash-Next run on 4x L4. One ARC-AGI-2 competitor reports Flash-Next and the 27B at "almost breakeven in speed and accuracy on L4x4".

### Cited Findings
**Root causes of "!" / token-0 output found upstream [V]**
- **SM120 QSA decode fallback:**
  - PR #36806 "route exact SM120 to FlashInfer sparse decode" (merged 08-28 into the qwen4 branch) fixed #36537. The bug: thinking + tools + `qwen3_coder` looped "on token ID 0 … `!`".
  - The PR states: "The returned token IDs are already zero before `qwen3_coder` post-processing, so the parser exposes the symptom but does not generate the corrupt tokens", and "SM120 -> resolver rejects the arch -> packed FA4 varlen fallback -> CuTe MLIR failure".
  - The reporter later said the same fallback also caused "Intermittent (5–19%/generation) mid-generation collapse under CUDA-graph serving".
  - Sources: [sglang#36806](https://github.com/sgl-project/sglang/pull/36806), [sglang#36537](https://github.com/sgl-project/sglang/issues/36537).
- **SM121 (GB10), when the TRTLLM path is forced:** "Past a few thousand tokens of context, generations start coming back as long unbroken runs of token id 0, which decodes to "!". Deterministic at temperature 0 … the engine does not recover … `/flush_cache` does not clear it. Only a restart does." Token-0 corruption rose "from 1/4 runs at 120K tokens to 4/4 at 210K" — [sglang#36558 comment](https://github.com/sgl-project/sglang/issues/36558); [sglang#36806](https://github.com/sgl-project/sglang/pull/36806).
- **MoE router PDL race:**
  - "Two PDL-launched routing kernels load their `bias` before the programmatic-dependent-launch wait … (Triton, `launch_pdl=True` on compute capability >= 9)". This "is the mechanism behind #37111 … output collapsing to repeated `!`" (merged 09-07).
  - Workaround on older images: `SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0`, which gave 100 min clean versus a collapse at 18 min.
  - Sources: [sglang#38290](https://github.com/sgl-project/sglang/pull/38290), [sglang#37111](https://github.com/sgl-project/sglang/issues/37111).
  - [I] PDL is enabled only at CC ≥ 9, so this mechanism should not apply to L4 (CC 8.9).
- **QSA + NEXTN decode CUDA graphs on GB10:** "can return HTTP 200 … while silently producing semantically invalid output"; the eager control passes — [sglang#37111](https://github.com/sgl-project/sglang/issues/37111).
- **vLLM MARLIN with EP:** "MARLIN multiplied top-k weights inside the kernel even with `expert_map`" → "Garbage or repeated tokens"; fixed by local patch 02k on 3x 4090 SM89 — [chenxin repo](https://github.com/chenxin199305/qwen38-flash-next-3x4090). (I did not find whether this patch is upstream.)
- **SM89 QSA:** "QSA sparse decode on NVIDIA SM89 requires classic FlashAttention / FA2". The guard PR is still OPEN (10-26) — [sglang#36968](https://github.com/sgl-project/sglang/pull/36968).
- **The QSA kernel's own warning:** "Data race detected: `Logits(bx, position)` is written by multiple threads" on the Triton MoE + QSA path on A100 — [sglang#37089](https://github.com/sgl-project/sglang/issues/37089). Our E3 log had a similar "TileLang data-race warning in the QSA kernel" (restated, `community_evidence.md` §4).
- **A ruled-out claim:** vllm#54252 had claimed that compressed-tensors W4A16 configs collapse GDN. It was withdrawn as invalid; the cause was a naive RTN int4 checkpoint with "~190% relative error" — [vllm#54252](https://github.com/vllm-project/vllm/issues/54252).

**Our runs on 4x L4 [V, local]**
- **E3** (Pennyroyal SGLang 0.5.19, Intel W4A16 + FP8 PLE, TP4/EP4): ready at 862 s, KV 118k, output "!!!!", about 6 tok/s single stream, with a TileLang data-race warning in QSA (`community_evidence.md` §4).
- **E13 SMOKE1** (SGLang 0.5.21): OOM in `gptq_marlin_moe_repack` at 20.86 GiB per rank. LOADPROBE traced it to a delayed free of +0.30 GiB per MoE layer, likely freed only by GC.
- **E13 bf16ple arm:** loaded at 19.10 GB per rank, KV 94,528 fp8 tokens, ready in 1,132 s, but the smoke output was **"!!!!" (token 0)**. This was "independent of KV dtype, FA2 and CUDA graphs" (memory `e13-flashnext-smoke1-oom.md`).
- Proposed next step (not run): `gc.collect()` fix plus forward hooks that find the first NaN/Inf module.

**NVFP4 on Ada**
- SGLang cookbook: NVFP4 is Blackwell-only (restated). `nvidia/…-FP4-Local` is Blackwell-only [V].
- Contradicting evidence for vLLM: the 3x RTX 4090 recipe serves NVFP4 routed experts on SM89 through MARLIN (weight-only dequant). It needs about 28 GiB per card ("24GB cards: OOM at load") — [chenxin repo](https://github.com/chenxin199305/qwen38-flash-next-3x4090). Our memory note "NVFP4 does not run on L4" is therefore true for SGLang and native FP4 GEMMs, but not strictly for patched vLLM Marlin. The binding blocker on L4 is memory.

**Measured Flash-Next speeds on Ada and similar hardware**
- 3x RTX 4090 48 GB, vLLM, NVFP4 + MTP2: 124 / 387 / 618 / 635 tok/s at 1 / 4 / 8 / 16 streams [C] ([chenxin repo](https://github.com/chenxin199305/qwen38-flash-next-3x4090)).
- 4x RTX 3090 (Ampere), vLLM TP4 + EP, W4A16: 109.5–145.8 tok/s single stream, 252.6 aggregate at 6 agents (restated, [tonyd2wild](https://github.com/tonyd2wild/Qwen38-Flash-Next-4x3090)).
- Single RTX 4090 48 GB, Strata (llama.cpp-derived), UD-Q4_K_XL, 1 slot: 72.4 tok/s decode on chat, 85.8 on agent turns; prefill 1,104 tok/s at 4K and 2,062 at 16K. On a 24 GB 4090 with expert cache in VRAM, totals were 64.6 / 69.4 / 71.0 / 71.2 tok/s at 1 / 2 / 4 / 8 clients. "Speed only: no correctness check" — [Strata #2668](https://github.com/Niko1221/Strata/issues/2668).
- Other single-4090 claims of ~100–164 tok/s use 2–3-bit quants (Strata IQ3_S, 164 tok/s) [C] — [gist](https://gist.github.com/ryan4yin/48617bbddacc7067f10799770b7cc33f); [HN](https://news.ycombinator.com/item?id=49953495).

**Flash-Next vs 27B on 4x L4, competitor statements**
- Adarsh (10-27): "my qwen 3.8 flash-next experiments are showing 27b and fn are almost breakeven in speed and accuracy on L4x4. On a singular GPU though, yes flash-next is faster. My experiments are based on franzen's duck harness setup" [C] — [750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546).
- Adarsh (10-18): Flash-Next "hasn't worked out that well for me as L4 GPUs based on Triton architectures aren't that fast as hopper ones" [C] — same topic.
- TheDivine Aurora's reply (10-28): "your local evals are faulty. qwen FN has a lot more solve % on the public eval dataset than 27b. burned weeks on 27b". He tested through OpenRouter (not L4) on "16 task dataset … 4-5 attempts per task. it solved all of them … quantization hurts quality". He reports his own harness at "250-400tps" [C/R] — same topic.
- Darren Amadeus Martin, earlier: "4 L4 seems to be very slow when running these models"; "flash next might have a better shot at 4 L4 since we can offload its embedding layer into ram" [C] (restated, topics 745787 / 746149).
- Hiểu Vy: 26.81 LB with the 27B vs 8.61 with "Qwen Flash" (restated, [748082](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082)).

### Inferences
- [I] **The "!!!!" on our L4 runs most likely comes from the QSA decode backend or the MoE kernel under EP, not from tokenization or the parser.** Every documented Flash-Next token-0 collapse traces to QSA decode kernels or MoE routing/scaling kernels producing NaN/zero logits.
  - On SM89, SGLang's resolver will select FA4/CuTe when FA2 is missing. The memory note says the E13 arm was "independent of FA2", so if FA2 truly was the decode path, the next suspects are the Marlin/EP MoE path (the vLLM 02k bug class: top-k weights applied with `expert_map`) and the GPTQ-Marlin repack itself (the same function OOMed).
  - The planned NaN-locator hook run is the right discriminator. Rule out the other cause by comparing EP4 against TP-only if memory allows, or MoE runner `triton` against `marlin`.
- [I] **Memory is the decisive constraint on 4x L4.** Even with every kernel fixed, a 94–118k-token KV pool supports about 1–2 concurrent 60–100k-token ARC-AGI-2 streams. Single-stream Flash-Next decode on L4 (bandwidth-bound) would be far below what is needed. Reaching Darren-type 700–980 tok/s needs ≥ 12–16 resident streams, which needs a far smaller GPU footprint:
  - a 2.5–3.5-bit expert quant (GGUF IQ3, EXL3 3.05 bpw), a REAP-pruned 288-expert model, or a lighter checkpoint like nvidia FP4-Local (Blackwell-only, so unavailable);
  - or the 27B (19.5 GB INT4 → ~70 GB KV).
  This is my arithmetic from our measured pools, not a measured result.
- [I] The vLLM 3x4090 recipe is the only evidence of correct Flash-Next decode on SM89. If we pursue Flash-Next on L4, vLLM 0.32 or nightly plus those patch classes (MARLIN EP top-k, the `mamba_block_size` seed, the indexer workspace accounting) is the better-evidenced route than SGLang 0.5.21.

### Gaps
- No public report of correct Flash-Next output on L4 or L20 in any engine. The L20 report (#36968) only reached HTTP-ready before the CuTe failure.
- I did not verify whether SGLang 0.5.22 or main has any SM89-specific QSA decode fix beyond the open guard PR. The `compare` API reported PRs #36806/#38290 as "diverged" from the v0.5.21/v0.5.22 tags: they were merged to the `qwen4-main-squashed` branch, so their presence in the release tags is unconfirmed.
- No measured tok/s for Flash-Next on 4x L4 from anyone.

---

## Q4. Daniel Franzen's ARC-AGI-3 Milestone 2 solution: links, engine config, harness design (Python REPL tool, turns, caps, concurrency, token budgets)

### Takeaway
Franzen scored 27.89 on the ARC-AGI-3 public LB (Milestone 2, about 09-30). His solution is Tufa Labs' Duck harness (Python-tool agent) with Qwen3.8-Flash-Next (Intel W4A16 AutoRound + BF16 PLE in host RAM, albucino INT4 MTP drafter) on a single RTX PRO 6000. Serving is Pennyroyal SGLang v2.5.3 (`d00d88e`) with 5 patches. The harness runs 10 streams, a 128 Ki harness context with 12 Ki max output per request, trims back to about 57 Ki, yields at 2,048 generated tokens, uses about 3,072-token tool output, temp 0.7, and a priority scheduler. Measured: 93.43% prefix reuse, P90 aggregate decode 836.6 tok/s, harness-side generation 609 tok/s.

### Cited Findings
**Links [V]**
- Notebook: [kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution](https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution).
- Repo: [github.com/da-fr/arc-agi-3-solution](https://github.com/da-fr/arc-agi-3-solution); local clone `D:/ARC AGI 2/_scratch/ext/dafr`, HEAD `10882e33`, 2026-09-30, "fix framework readme". Mirror: [tonghuikang/daniel-franzen-arc-agi-3](https://github.com/tonghuikang/daniel-franzen-arc-agi-3).
- Write-up: `WRITEUP.md` in the repo (read in full for this note).
- Base: [Tufalabs/duck-harness](https://github.com/Tufalabs/duck-harness) at commit `7652836` (Milestone 1 winner; technical write-up [ARC-AGI-3 topic 717133](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/717133)).
- Kaggle inputs, as reused by a public fork's metadata: datasets `dfranzen/pennyroyal-v253` and `dfranzen/taaf-kaggle-source-bundle-copy`; models `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/Transformers/default/1` and `dfranzen/albucino-qwen3-8-flash-next-drafter/Transformers/default/1`; machine `NvidiaRtxPro6000` (kernel-metadata of [laimerbotrolenk/lb-score-31-21](https://www.kaggle.com/code/laimerbotrolenk/lb-score-31-21), pulled 10-29).
- Winner comparison: [ARC-AGI-3 topic 744792](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/744792).

**Model path and serving [V, WRITEUP §1]**
- Model path: "Qwen3.6-27B → Qwen3.8-27B → Qwen3.8-Flash-Next. Each model switch gave a substantial improvement".
- Quant history: RadixArk NVFP4 first, then Intel W4A16 AutoRound ("similar quality and throughput, while leaving more VRAM available for the KV cache"). MTP drafter from albucino; "retain the main model's unquantized BF16 PLE table" in host RAM.
- Engine: Pennyroyal (John Pezzulli's SGLang fork) v2.5.3 at `d00d88e`. Patches:
  1. low-M BF16 GEMM (from Gabriel Olympie's SM120 patch 0004);
  2. spec-state memory-budget correction (from Mamy Ratsimbazafy's patch 0002);
  3. Marlin scale-dtype BF16/FP16 fix;
  4. prefix-cache patch (keep only the final-prefill Mamba checkpoint plus a limited branch checkpoint; refresh its LRU position when decode ends);
  5. checkpoint-prefetch patch (loader-order parallel reads, one shard ahead) plus an external prefetcher. The model was resharded.
- Settings table: single RTX PRO 6000; 10 streams; harness context 131,072; max output per request 12,288; server context 139,264; drain 59,392; FP8 E4M3 KV; Mamba cache 60 slots; mem fraction 0.96; chunked prefill 8,192; 3 speculative steps; temp 0.7 / top-p 0.95 / top-k 20.
- Exact launch (pulled notebook cell 12, restated from `community_evidence.md` §2):
  - Core: `sglang serve … --tensor-parallel-size 1 --dtype bfloat16 --quantization auto-round --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.96 --context-length 139264 --page-size 64 --max-running-requests 10 --chunked-prefill-size 8192 --max-prefill-tokens 16384`.
  - CUDA graphs: `--cuda-graph-max-bs-decode 10 --cuda-graph-bs-decode 1 2 4 7 8 9 10`.
  - Mamba and linear attention: `--mamba-ssm-dtype bfloat16 --max-mamba-cache-size 60 --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64 --mamba-backend flashinfer --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer`.
  - Model and parsing: `--moe-runner-backend auto --ple-offload-embedding --reasoning-parser qwen3 --tool-call-parser qwen3_coder --default-chat-template-kwargs '{"preserve_thinking":true}'`.
  - Scheduling and monitoring: `--schedule-policy lpm --enable-cache-report --enable-metrics`.
  - Loading: `--weight-loader-prefetch-checkpoints --weight-loader-drop-cache-after-load`.
  - Speculation: NEXTN 3/1/4, draft quantization compressed-tensors, FR-Spec 64k hot-token map.
  - Env and startup: `SGLANG_MAMBA_CONV_DTYPE=bfloat16`; soft startup deadline `SERVER_STARTUP_TIMEOUT = 12*60` (the harness is released while the server is still loading).
- Measured (earlier demo run, slightly different config): prefix-cache reuse 93.43% of prompt tokens; P90 effective prefill 10,552.6 tok/s; **P90 aggregate decode 836.6 tok/s**; harness-side generation 609.0 tok/s; prefill time fraction 15.25%; long-prompt cache-miss rate 0.03% (1/3,248). He "reached Kaggle's compute limit before I could do a full demo run with the final submission configuration".
- Peak decode of the three M2 winners: dfranzen 946, lordhansolo 1,135, sirikilohit 1,159 tok/s; ready at 531 / 544 / 615 s (restated, topic 744792).

**Harness design [V, WRITEUP §2–6, repo README/CONFIGURATION.md]**
- **Core loop:** "a language model plays the games through a Python tool, using code to inspect the board, test mechanics, and execute actions".
  - History = system prompt + growing interaction history. User prompts, the model's **thinking** (deliberately retained: "removing it leads to a large regression"), responses, tool calls and outputs are all kept.
  - Trimming happens at the start of history, right after the system prompt.
- **Token accounting:** a calibrated text-token estimate (fit to backend usage) plus an image-token estimate from patch/merge geometry.
- **Admission:** "limited admission to ten game streams, matching the server's request limit, to keep their cached prefixes resident". A game "normally keeps its admission slot across tool calls and resumptions"; it re-enters the queue only at context eviction.
- **Hysteresis trim:** max context 128 Ki, 12 Ki reserved for output, trim to about 57 Ki retained.
- **Scheduler:** all games start at once. Priority P = (A + B)·C. A = level gain with an action-efficiency penalty; B = continuation value by levels remaining (8 / 7 / 5 / 0); C = Gaussian fade in actions (115) and tokens (62,000). B fades over the last 20% of runtime.
- **Python tool:** each call starts with a fresh namespace. Model-defined functions are retained for the whole game (`ARC3_PERSISTENT_FUNCTIONS_SCOPE=game`); ordinary variables are discarded. Helpers include `frame_diff()` (object-level changes over 4-connected same-colour components) and animation frames as REPL variables.
- **Tool output:** about 3,072 tokens, middle truncation, action echo.
- **Yield:** "Yielding is now triggered by generated tokens (2,048 in this configuration)", checked between completions, not as a hard mid-stream cutoff (`LOCAL_ANALYZER_YIELD_TOKENS`). There is also an assistant-message history cap of 30 by default (`ARC3_HISTORY_ASSISTANT_TURNS`).
- **Guards:** batch no-op, stale-state and terminal-state.
- **Things that did not help:** structured world-model updates (disabling them was best), summarization, resume-prompt variants, composite animation images, more prescriptive verification.
- **Development compute:** mostly Kaggle, plus about 150 rented RTX PRO 6000 GPU-hours. No training or fine-tuning. "Most of the code for my modifications was written with Claude and ChatGPT."

### Inferences
- [I] Franzen's throughput comes from a single 1.8 TB/s GPU with no all-reduce, 10 resident streams and about 1 M KV tokens. On 4x L4 the transferable parts are the cache policy (slot retention, hysteresis trim, end-of-prefill Mamba checkpoint, calibrated token estimate, LPM, page 64) and the soft startup deadline. The ARC-AGI-3 caps (12 Ki output, 2,048-token yield, 57 Ki trim) are game-loop choices. Our memory rule `arc-agi2-needs-long-thinking.md` says not to import them as ARC-AGI-2 defaults.

### Gaps
- Final-config demo-run metrics were never published (his own statement).
- I did not re-read the Kaggle notebook cells for this note; the launch line is restated from the earlier pull (`_scratch/ext/dfranzen_cells_4_12_16.txt`).

---

## Q5. Claims in Kaggle discussion 750546 and the evidence that ARC-AGI-2 leaders use Flash-Next + Duck; how ARC-AGI-2 teams adapt Duck

### Takeaway
The claim that "many of the leading teams are using Qwen3.8-Flash-Next along with the Duck harness" is Anil Thomas's inference (10-27) from the timing of the score surge. No top team has confirmed it publicly. Darren Amadeus Martin's numbers (900 → 980 tok/s peak, 710 tok/s average over 6 h, 2 passes on public eval, about 250 tok/s under heavy prefill) are real statements, but he never names his model or hardware details. That his team is "3rd place" is not supported by the thread. The public LB on 10-29 is led by **Tufa Labs (the Duck authors) at 88.06**. No public ARC-AGI-2 notebook using Flash-Next + Duck was found.

### Cited Findings
**Topic 750546** "Qwen3.8 or something else?" by Anil Thomas, posted 2026-10-18, 38 comments. Read from the local API dump `D:/ARC AGI 2/_scratch/kaggle_disc/topic_750546.json` — [750546](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546).

*Anil Thomas*
- 10-18: "the sudden flood of 55+ scores makes me think that there is an easy recipe out there. I was able to go from 35 to 45 by ensembling my NVARC-based solution with a Qwen3.8-27B run … a harness that includes a python REPL can help … but squeezing it into a 12 hour window on 4x L4 is hard."
- 10-23: "almost convinced that the recipe involves distilling". mikelou1 replied: "I'm no where near the top but I don't use distillation."
- 10-27 [R, inference by the poster]: "the cat is finally out of the bag! it is clear that many of the leading teams are using Qwen3.8-Flash-Next alongwith the Duck harness … the beginning of the surge coincides with the publication of Daniel Franzen's milestone 2 solution".
  - His model: "Flash-Next can get to a score around 75% with one pass and 80% with two passes".
  - Throughput need: "240*64k/(12*60*60) ~ 365 tok/s" for one pass; "~ 440 tok/s" with a 20% second pass.

*Darren Amadeus Martin* [C]
- 10-27: "With lots and lots of optimizations by claude code, our team manage to get a top speed of 900 tok/s."
- 10-27: "top speed of 980 tok/s on 1 minute harness work which includes tool call and prefill … 1 single stream on L4 is very slow, so we can only have high speed when we increase the concurrency. Now the secret sauce is how to have high concurrency when 1 problem itself can reason for more than 100k tokens."
- 10-28, definition: "if in 1 minute the model on harness generated 60000 and prefill of 5000 token, then the 'decode speed under harness' is 1000 tok/s. … pure decode only with same concurrency … like 1100 tok/s".
- 10-28: "The peak is 980 tok/s … When there's much prefill, it could drop to 250 tok/s. But on 6 hours and 2 pass of the public evaluation, the average decode speed is 710 tok/s (due to the tail being starved of concurrency)".
- 10-28: on a PRO 6000, "it works there too, the speed itself is much faster than franzen's setup but it doesn't translate to the leaderboard"; his ARC-AGI-3 submission "only to get a score of 4"; "I can't tell you more than this as I would spill my team's method".
- 10-19: "our cv also got scores easily above 75% but it doesn't translate that well to LB"; his team validates on "the public CV 120 problems and time limit of 6 hours".
- **He never states the model (27B or Flash-Next) in this thread.**

*Other comments*
- mikelou1 [C]: "With time constraints at around 4xL4 speed I got Qwen to 87% (and from reasoning traces it got the rule to almost every problem)" (10-20); "around 79% ± 5% in CV runs" (10-19). LB 69.72 on 10-23 (leaderboard below).
- TheDivine Aurora [C]: Flash-Next solves more than the 27B on public eval; "quantization hurts quality"; his harness is at 250–400 tps. Earlier: "at around 300-500 tps i had ~19-20 % score. only attempted half the dataset … it just takes 128k - 256k context window".
- Kris Dreemurr [C]: his CV harness at xhigh, "median 70k tokens for pass@2 (so 35k per slot) gets %45 solves or about 52~ out of 120 … I cant fit it in the box".
- Adarsh: see Q3 (FN ≈ 27B on L4x4 with Franzen's Duck setup).

**Licence topic 752709** (Darren, 10-26, no replies): asks whether Flash-Next solutions (Qwen Community License 1.0) are prize-eligible. He states "Many teams in both competitions have spent a lot of time building pipelines around this model" and that "Milestone 2 winners reportedly include solutions using Qwen3.8-Flash-Next" [C] — local `D:/ARC AGI 2/_scratch/kaggle_disc/topic_752709.json`; [752709](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/752709).

**Public LB top 20, ARC-AGI-2** (Kaggle API `competition_leaderboard_view`, read 2026-10-29) [V]:
| # | Team | Score | Last sub |
|---|---|---|---|
| 1 | Tufa Labs | 88.06 | 10-27 |
| 2 | Yi-Chia Chen @ Fields Model | 84.72 | 10-27 |
| 3 | Domdolus Tolus | 83.47 | 10-28 |
| 4 | nvbanana | 81.39 | 10-25 |
| 5 | rabbithole | 80.83 | 10-27 |
| 6 | Ravi Teja Chilakala | 76.94 | 10-27 |
| 7 | Majkel1337 | 76.81 | 10-27 |
| 8 | ALbert RUehlman | 75.56 | 10-25 |
| 9 | Dawid Kopiczko | 75.14 | 10-27 |
| 10 | guoqin gu | 73.89 | 10-27 |
| 11 | ｶﾎｳｼｭｳｾｲ | 73.47 | 10-27 |
| 12 | _hans | 71.81 | 10-27 |
| 13 | ảc | 71.53 | 10-27 |
| 14 | Youssef Nader | 70.56 | 10-27 |
| 15 | artificialagencylab.com | 70.56 | 10-27 |
| 16 | matu997 | 70.14 | 10-27 |
| 17 | mikelou1 | 69.72 | 10-23 |
| 18 | この遥かなる旅路に祝福を | 69.72 | 10-27 |
| 19 | Sakayori Lab | 69.58 | 10-27 |
| 20 | Lashen Learning | 69.44 | 10-27 |

- Darren's team name is not visible in this data. Whether he is "3rd place" cannot be confirmed: #3 is "Domdolus Tolus", and the API returned only the top 20.

**Public notebooks (Kaggle API search "flash-next", "flash next duck", "arc-agi-2 duck", 10-29) [V]**
- All Duck/Franzen forks found are ARC-AGI-3 RTX PRO 6000 notebooks. Examples: `laimerbotrolenk/lb-score-31-21` (ARC-AGI-3 source, Franzen datasets/models, `NvidiaRtxPro6000`); `juliancamilovilla/arc-agi3-franzen-m2`; `guoxiangwangs/arc-agi-3-dfranzen-m2-fork`; `liaojiaxuan/lingjing-flashnext-shortbench-logs` (RTX PRO 6000, Franzen stack, ready about 562 s, 8 games x 900 s, mean 30.03).
- None of the found notebooks is an ARC-AGI-2 4x L4 Flash-Next notebook.
- Top ARC-AGI-2 teams have not published code (competition still running; final deadline 11-02 per memory, unverified).

**How ARC-AGI-2 teams adapt Duck (evidence found)**
- Adarsh: "My experiments are based on franzen's duck harness setup" on L4x4 with both models [C].
- Our own adaptation [V, local]:
  - E12 (Duck adapter for ARC-AGI-2, code-first lessons: memory `code-first-duck-lessons.md`);
  - E15 config B on SGLang: EVAL120 6-h R3K2 = 64/172 test outputs, task-normalized 39.31% (memory `experiment-15-config-b.md`, `e15p-poetiq-duck-engine.md`);
  - E21 Duck config B on vLLM 0.19/0.32 (memory `experiment-21-duckb-vllm.md`).
- Our ARC-AGI-2 rule: keep long uninterrupted thinking. Do not import Duck's 12,288 cap or chunking as defaults (memory `arc-agi2-needs-long-thinking.md`).

### Inferences
- [I] On 4x L4, Darren's "high concurrency when 1 problem itself can reason for more than 100k tokens" points to a large resident KV pool. That is far easier with the 27B INT4 (about 70 GB free for KV) than with Flash-Next W4A16 (94–118k-token pool measured by us). So either his team uses the 27B, or they found a much smaller Flash-Next footprint (sub-4-bit or pruned experts, PLE fully on host). His 710 tok/s 6-h average at 2 passes on 120 tasks is about 15.3 M output tokens per 6 h, roughly twice our best 27B steady 373–431 tok/s. This is an inference, not evidence of the model.
- [I] Tufa Labs leading ARC-AGI-2 at 88.06 is consistent with "Duck-style harness + strong Qwen model" being the leading recipe. Tufa's ARC-AGI-2 model and hardware use are not public, so this is consistent with Anil's claim but not proof of it.
- [I] Anil's coverage arithmetic (365–440 tok/s for 240 tasks at 64k) matches our memory finding that coverage binds. His 64k mean per task is below our 60k-floor plus multi-turn practice, so the real requirement for our pipeline is higher.

### Gaps
- No primary source names the model or engine of any top-10 ARC-AGI-2 team.
- Darren's team name and rank are unverified. The "3rd place" premise in the brief is not supported by the data I could read.
- The leaderboard API returned only 20 rows. Ranks beyond 20, and any team-to-person mapping, were not retrieved.
- TheDivine Aurora's OpenRouter test is not reproducible from the post: no task list and no per-task results.
