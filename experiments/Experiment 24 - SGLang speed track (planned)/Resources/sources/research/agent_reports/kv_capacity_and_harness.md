# KV / recurrent-state capacity levers and harness-level throughput levers for 16-32+ live long-reasoning requests on Kaggle 4x L4 (Qwen3.8-27B hybrid GDN, Qwen3.8-Flash-Next), as of 2026-10-29

Scope note: this file adds to the earlier notes in `research_notes/Qwen 27B দ্রুততম সার্ভিং সেটআপ/` and `research_notes/৪xL4 ইঞ্জিন বটলনেক যাচাই/` and does not repeat them. Those notes already cover SGLang HiCache flags and the hybrid host-tier bugs (#39830, #44735, #45400, #36935), the SGLang mamba radix strategies and Franzen's refresh-on-unlock patch, SGLang eviction policies (`tlru`, `priority`, session radix cache, streaming sessions), the Continuum and Marconi papers, the vLLM 0.30-0.32 release items (`--max-num-active-seqs` #56758, waiting-queue rework #58947, the `max_num_seqs` not-multiple-of-8 cliff #57355, `max_tokens` treated as an upper bound #57035, `all` mode removed), and the MTP + prefix-cache bugs (#53912, #54360, #55533). See `৪xL4 ইঞ্জিন বটলনেক যাচাই/agent_effective_throughput.md` §2-5 and `newer_engine_versions.md`.

Evidence tags: [V] = verified on the primary page in this session. [S] = only in a search-engine summary; the primary page was not opened. [F] = fork or community report, not upstream. [D] = my own arithmetic, with the inputs stated.

Our baseline (from the brief): TP4, fp8 KV, prefix caching on. vLLM 0.32 FP8 + MTP3, SSM fp16, 16 streams: 450 / 414 / 312 tok/s at 8k / 40k / 90k context. vLLM 0.32 INT4 + MTP3, 20 streams: 518 / 439 / 332. SGLang 0.5.21 INT4 + DFlash2: 597 / 536 / 364. KV is about 1.35M tokens (FP8 weights) or 1.71M (INT4 weights).

---

## 1. Sub-fp8 KV cache (NVFP4 / FP4 / INT4 / TurboQuant) on Ada sm_89: support, accuracy, capacity and speed

### Takeaway
There is one upstream sub-fp8 KV path that runs on sm_89. It is vLLM `--kv-cache-dtype nvfp4`, through FlashInfer FA2, merged on 2026-10-07 (#46963) and in v0.32.0. It gives about 1.75x more KV tokens than fp8. Two things count against it. First, the only sm_89 measurement (a fork, on vLLM 0.32) shows aggregate throughput 13-17 % lower. Second, on SM8x the NVFP4 groups capture only single-token decode CUDA graphs, so MTP verify steps probably run without CUDA graphs. Accuracy on Qwen3.8-27B looks close to fp8, but that evidence comes from a different kernel (the Blackwell kernel in SGLang), not the FA2 path. SGLang sub-fp8 KV is Blackwell-only or broken. TurboQuant (vLLM, LMDeploy) and LMDeploy INT4 KV exist, but they have no hybrid + MTP + sm_89 evidence and they cost throughput.

### Cited Findings
**vLLM NVFP4 KV on SM8x (upstream)**
- [V] PR #46963 "Support NVFP4 KV cache on SM8x and SM12x with FlashInfer" (by lesj0610), merged into main on 2026-10-07 and approved by mgoin. — [vLLM #46963](https://github.com/vllm-project/vllm/pull/46963)
  - It needs no CUDA kernel change. Writes go through FlashInfer `nvfp4_quantize_append_paged_kv_cache_with_slot_mapping`, and reads use "the native fa2 prefill and decode wrappers". The page refers only to "block scales"; it gives no per-token or per-head scale detail.
  - The PR says: "SM86, SM89, SM120 and performance are untested". Its accuracy and capacity tests ran on sm_80 and SM121.
  - **CUDA graphs: "on SM12x and SM8x, NVFP4 groups capture only single-token decode graphs."**
  - Refused cases: head_size 64 (fa2 prefill misreads NVFP4 by "10 to 15 percent", flashinfer#5965) and attention sinks (flashinfer#5966). Head_dim 256, which our model uses, is not on the refused list.
  - Accuracy, GSM8K, sm_80, flashinfer 0.7.0.post1:

    | Model | bf16 KV | nvfp4 KV |
    |---|---|---|
    | Qwen3-0.6B, all 1319 questions | 0.384 | 0.174 |
    | Qwen3-4B, 100 questions | 0.920 / 0.910 | 0.880 / 0.880 |

    MRCR on Qwen3.5-4B (a hybrid): auto 0.8523, nvfp4 0.8452.
  - Capacity on sm_80: "1,446,932 to 4,805,405 tokens (3.32x)". The baseline dtype is not named; it is probably bf16.
  - MTP on SM120: a commenter got about 15 t/s decode with MTP. The follow-up #60452 adds XQA decode for SM12x only. The page reports no SM8x MTP result.
  - Prefix caching on SM120: warm prefill was 5-11 % slower, which the commenter later attributed to flashinfer#6204.
- [V] The vLLM 0.32.0 release notes list "NVFP4 KV cache now runs on SM8x and SM12x (#46963)". Cited in the earlier note `newer_engine_versions.md`. — [v0.32.0 notes](https://github.com/vllm-project/vllm/releases/tag/v0.32.0)

**The only sm_89 measurement (fork, vLLM 0.32.0)**
- [V][F] jsboige/vllm PR #123, "NVFP4-KV eval on SM89 — rejected", merged 2026-10-19. The model is not named; a 27B is implied by the profile name `medium-swift15-27b.yml` in #128. Setup: FlashInfer FA2 path, uint8 storage. — [jsboige #123](https://github.com/jsboige/vllm/pull/123), [jsboige #128](https://github.com/jsboige/vllm/pull/128)
  - KV went from 429,319 to 750,747 tokens (x1.75).
  - Aggregate decode: N=1 41.1 → 33.9 t/s; N=48 886.4 → 767.5 t/s (-13..-17 %). Prefill -3..-6 %.
  - Quality gates 11/11; needle HIT at 25 % / 75 %.
  - Reason for rejection: "Capacity is not the scarce resource (observed occupancy 2-7%, fp8+tier = 1.11M tok)". The fork's earlier TurboQuant k8v4 trial (#110) was rejected for the "same signature". The author would reopen it if occupancy rises above about 50 % or the SM89 kernels mature.
  - #128 later switched the profile default to nvfp4 anyway, together with a 32 GiB CPU tier.

**Accuracy of NVFP4 KV on Qwen3.8-27B (Blackwell kernel, SGLang)**
- [V] LMSYS, "Accelerating Long-Context and Agentic Inference with NVFP4 KV Cache" (2026-09-16). — [LMSYS NVFP4 KV blog](https://www.lmsys.org/blog/2026-09-16-nvfp4-kv-cache/)
  - Format: E4M3 scale per 16-value block plus an FP32 tensor scale. The global scale was 1.0 in their runs.
  - Architectures: SM12x and SM100/SM103 only. No Ada path is mentioned.
  - Qwen3.8-27B results against fp8 KV:

    | Benchmark | Difference vs fp8 KV |
    |---|---|
    | GSM8K | -0.31 pt |
    | GPQA-Diamond | -1.01 pt |
    | AIME 2025, thinking/xhigh | equal at 98.33 % |
    | SWE-bench Verified | 381/500 vs 389/500 (-1.60 pt) |

  - Capacity is ideally about 1.78x fp8. Measured concurrency on RTX PRO 6000: 32K context 44 → 70 requests; 160K 10 → 15.
  - Decode: about +26-30 % at equal concurrency; +37 % / +58 % / +78 % at equal capacity for 32K / 160K / 1M.
  - In an AgentX run (Qwen3.5-397B, TP8), fp8 throughput "dropped sharply" beyond C = 12, while NVFP4 throughput kept rising.
  - Flags: `--kv-cache-dtype nvfp4 --prefill-attention-backend flashinfer --decode-attention-backend trtllm_mha`. The feature is marked experimental.
- [V] An ARC-AGI-3 competitor (xz) said there is "very little NLL penalty to use nvfp4 kv". This is in the earlier `community_evidence.md` (local `_scratch/webresearch/d_744792.md`).

**SGLang sub-fp8 KV**
- [S] `--kv-cache-dtype fp4_e2m1` was added for MLA (#10078) and extended to MHA such as Qwen (#12612; uint8 storage plus per-block scale buffers). — [SGLang #12612](https://github.com/sgl-project/sglang/pull/12612)
- [S] Issue #17365: on 4x B200 with Qwen3-32B, `fp4_e2m1` decode fell to about 90 tok/s, versus about 5,000 with `auto`. The reporter suspects the whole buffer is dequantized. — [SGLang #17365](https://github.com/sgl-project/sglang/issues/17365)
- [S] FP4 KV for SM120 (#21601). MXFP4 KV for Qwen3.5/3.8-27B on SM120 (#38724, `--kv-cache-dtype mxfp4`, block 32) is work in progress. No sm_89 path was found. — [SGLang #21601](https://github.com/sgl-project/sglang/pull/21601), [SGLang #38724](https://github.com/sgl-project/sglang/pull/38724)

**TurboQuant (vLLM)**
- [V] PR #39931 "TurboQuant: support hybrid models and uniform quantization", merged 2026-05-05. — [vLLM #39931](https://github.com/vllm-project/vllm/pull/39931)
  - It applies TurboQuant only to `full_attention` layers of Qwen3.5 and Qwen3-Next; before this, those models raised `NotImplementedError` on Mamba layers.
  - The PR page has no benchmark numbers.
  - Downstream reports: 27B dense + K8V4 gave 516K vs 322K tokens for fp8 (+60 %), with gains "within sampling noise" and the change rejected by that fork. A Qwen3.6 deployment reported GSM8K 0.860 and NIAH 100 %.
- [S] The base backend is #38479. The preset `turboquant_3bit_nc` is named by a secondary blog. — [vLLM #38479](https://github.com/vllm-project/vllm/pull/38479)
- [S] Known crashes:
  - #40420: OOM in `_continuation_prefill` at about 185K tokens.
  - #41726: crash on large chunked continuation prefill, which hits agent re-prefills directly.
  — [vLLM #40420](https://github.com/vllm-project/vllm/issues/40420), [vLLM #41726](https://github.com/vllm-project/vllm/issues/41726)
- [S] A tweet by Benjamin Marie: vLLM TurboQuant on Qwen3.5-27B shows "similar accuracy - 4x smaller KV cache". This is a social-media claim and was not verified. — [X post](https://x.com/bnjmn_marie/status/2043950978538279336)

**LMDeploy**
- [V] Online KV quantization, `quant_policy` 4 (INT4) or 8 (INT8), "asymmetric, applied per head and per token". Listed GPU support includes sm_89. — [LMDeploy KV quant docs](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html)
  - internlm2.5-7b GSM8K: 85.67 (fp16 KV) / 85.44 (INT8) / 83.78 (INT4).
  - RPS gain for llama2-7b: x1.27 (INT8) / x1.39 (INT4).
  - `quant_policy` 42 = TurboQuant. It is PyTorch-engine only, with "no speculative decoding". On H200 at concurrency 64 it lowered output throughput by about 7.3 %.
- [V] From the earlier `engines_and_config.md`: TurboMind lists Qwen3.5 with KV INT8 but not KV INT4. Qwen3.6/3.8 are not listed. — [LMDeploy supported models](https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html)

### Inferences
- [D] **NVFP4 KV and MTP on sm_89 probably conflict.** MTP3 verify steps process 1 + 3 tokens per request. Single-token-only decode graphs mean those steps run without CUDA graphs, and eager overhead on PCIe TP4 L4 is large (see the earlier `per_step_overheads.md`). The fork's -13..-17 % was measured without MTP being stated, at 2-7 % occupancy. Test NVFP4 + MTP on L4 before planning around it; the cost could be larger than the fork's figure.
- [D] Only the 16 full-attention layers have KV, so NVFP4 shrinks just that part. The GDN state (fixed per request) is unchanged. On our pool that is 1.35M → about 2.36M tokens (FP8 weights) or 1.71M → about 3.0M (INT4 weights), using the 1.75x the fork measured.
- The accuracy evidence that applies to our model (LMSYS: AIME equal, GPQA -1.0, SWE -1.6) uses the Blackwell `trtllm_mha` kernel, not FlashInfer FA2. The sm_80 FA2 results on small models are bad (Qwen3-0.6B GSM8K roughly halved), and bigger models degrade less (Qwen3-4B -4 pt). An ARC-AGI-2 eval check at T=0 against fp8 KV is needed before any LB use.
- TurboQuant, as the workload goes: the known crashes are in continuation prefill, which is exactly an agent re-prefill. One fork's 27B test showed no gain. That makes it a lower-priority arm than NVFP4.

### Gaps
- There is no measurement of NVFP4 KV on L4 or sm_89 with MTP, at long context (40-130k), or on a hybrid GDN model. There are also no ARC-type accuracy numbers.
- I found no evidence on whether the per-16-block scales in the FA2 NVFP4 path handle head_dim 256 without the error seen at head_size 64.
- TensorRT-LLM sub-fp8 KV on Ada was not researched in this session (no TRT-LLM page was fetched). Earlier notes consider TRT-LLM unrealistic offline for this model.
- Whether 0.32.0 itself contains #46963, or only main after 2026-10-07, rests only on the release-note line in the earlier note.

---

## 2. KV CPU offload / tiered KV (vLLM OffloadingConnector, LMCache, SGLang HiCache, Mooncake): speed over PCIe, hybrid GDN support, spec decode

### Takeaway
Offload does not raise the number of **live** decoding requests. A request that is decoding needs its KV on the GPU. Offload helps only for parked or idle sessions and for requests that get preempted. For hybrids the support is new and fragile:
- Hits happen only at the stored Mamba replay boundary (#63565).
- An explicit `--block-size` silently turns offload into write-only (#58653).
- The LMCache multiprocess (MP) connector supports Qwen3.8-27B (block 784, align mode) but is not bit-exact.

No source covers offload combined with MTP or other spec decode on hybrids. Transfers are cheap compared with re-prefill (about 1 GB per GPU for a 90k session). But since align mode does not store the state at the end of the reply, offload cannot fix our main miss, which is the previous reply being re-prefilled each turn.

### Cited Findings
**vLLM native OffloadingConnector**
- [V] Usage guide. — [vLLM KV offloading usage](https://docs.vllm.ai/en/latest/features/kv_offloading_usage/)
  - Config: `--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"block_size":64,"cpu_bytes_to_use":1000000000}}'`. `cpu_bytes_to_use` is the total over all workers, and the guide advises setting it "larger than the aggregate GPU KV cache".
  - Multi-tier: `TieringOffloadingSpec` with `secondary_tiers`, e.g. fs.
  - Supported platforms: CUDA, ROCm and XPU only.
  - "Per-token-head KV-cache quantization isn't supported" for the canonical layout.
  - The guide says nothing about hybrid or SSM support or spec decode.
- [V] vLLM blog, 2026-01-08. — [vLLM blog: KV offloading connector](https://vllm-project.github.io/2026/01/08/kv-offloading-connector.html)
  - Setup: H100 with 500 GB DRAM.
  - Transfer: up to about 50 GB/s in one direction; bidirectional DMA 83.4 GB/s vs 68.5 GB/s for the custom kernel, at 2 MB blocks.
  - Results: TTFT reduced x2-x22; throughput up to x9 as the CPU hit rate rises; "minimal effect on TTFT for cache misses".
  - Preemption: KV is moved to CPU "before the request is pre-empted". Loading preempted requests back from CPU is planned in PR #29870.
  - Hybrids: "vLLM also supports hybrid models, which are currently not optimized for the offloading connector".
  - Newer flags: `--kv_offloading_backend native --kv_offloading_size <GB>`.
- [S] A homelab PR on vLLM v0.32.0 enables CPU offload for Qwen models with `--mamba-cache-mode align --kv-offloading-backend native --kv-offloading-size 8`. — [HomeLab #781](https://github.com/ArthurVardevanyan/HomeLab/pull/781)

**Hybrid offload bugs and work in progress**
- [S] #58653: on hybrid GDN models with the SimpleCPU offload connector, an explicit `--block-size` that does not align with the Mamba page makes offload "silently write-only". Leaving the flag out works. — [vLLM #58653](https://github.com/vllm-project/vllm/issues/58653)
- [V] #63565 (vLLM 0.30.0 and main as of 2026-10-23; Qwen3.6-35B-A3B; 4x Intel Arc Pro B70 at TP4; `--kv-offloading-backend native --kv-offloading-size 16`; 576-token blocks). — [vLLM #63565](https://github.com/vllm-project/vllm/issues/63565)
  - "each request keeps one Mamba/GDN state, at its replay boundary in the last block of the prompt".
  - A revisit that diverges inside the last block gets 0/3 offload hits (TTFT 1.49 s). A revisit that diverges after it gets 3/3.
  - The fix is PR #63566 (open). MooncakeStore PR #62400 already sets the junction for the Mooncake connector.
- [S] #62914 enables canonical cross-TP offload for GDN hybrids; validated on Qwen3.5-0.8B. — [vLLM #62914](https://github.com/vllm-project/vllm/pull/62914)
- [V] #38230 (open, stale and reopened): the stock offload path needs the LCM of all group block sizes, which is impractical for hybrids. The proposed fix is a HybridOffloadPlanner plus MultiConnector (PR #38261, open). — [vLLM #38230](https://github.com/vllm-project/vllm/issues/38230)
  - On Qwen3.5-4B-FP8 (RTX 4080 Super, `max_model_len` 98304): "97% same-session hit rate" (APC + LMCache CPU); "50 concurrent requests stable at 1061 tok/s, 95% external hit rate".
  - A user reported "still happening on nightly" on Aug 20.
  - It links LMCache#2879 "Fix garbled output on hybrid models".

**LMCache**
- [V] LMCache MP connector, hybrid models. — [LMCache hybrid models](https://docs.lmcache.ai/mp/hybrid_models.html)
  - Validated: Qwen3.5 / 3.6 / 3.8 and Qwen3.8-Flash-Next.
  - Server flag: `--separate-object-groups`. vLLM flags: `--enable-prefix-caching --mamba-cache-mode align` (align is mandatory) plus `--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}'`.
  - Unified block size: 784 for Qwen3.8-27B on 1 GPU.
  - Recurrent state is stored as an opaque page.
  - "Output is not bit-exact between cached and fresh runs."
  - No versions, throughput or spec-decode statement.

**Measured tier use on sm_89 (fork)**
- [V][F] jsboige/vllm #128 (vLLM 0.32, SM89): fp8 KV + RAM tier = 1.11M tokens; 32 GiB CPU tier. — [jsboige #128](https://github.com/jsboige/vllm/pull/128)
  - Three "wedges" in one night: sessions grew to about 260K tokens, GPU KV hit 100 %, the KV fetch from the tier blocked, and global decode fell to 0.3-0.9 t/s.
  - "Each restart emptied the cache and made things worse."
  - The fs tier has no eviction.

**SGLang HiCache / Mooncake**
- Covered in the earlier `agent_effective_throughput.md` §2: the hybrid host-tier correctness bugs; zero host hits on Qwen3.8-27B once more than about 3 long prefixes are evicted (#45400); HiCache shrinks the device Mamba pool (#36935); no int8 Mamba checkpoints with HiCache.
- Milestone-2 ARC-AGI-3 data from that note: sirikilohit's 48 GB HiCache + Mamba host cache had the lowest LB of the three top solutions (22.53).

### Inferences
- [D] Reload vs re-prefill, using about 10 KB of fp8 KV per token per GPU (the earlier SGLang figure, 8.68 GB / 866,744 tokens):
  - A 90k-token parked session is about 0.9 GB per GPU of KV, plus one Mamba slot (about 18 MB per GPU per the earlier note).
  - Assumed link speed: about 20-25 GB/s effective. This is my assumption, not a measured number; L4 is PCIe Gen4 x16 per the NVIDIA L4 spec (not opened this session).
  - Reload is then about 40-50 ms per GPU, against about 60 s to re-prefill 90k at 1,500 tok/s.
  - The blog's ~50 GB/s one-direction figure is for H100 on PCIe Gen5.
  - **Offload would therefore pay only for parked sessions.** Examples: many paused stage-2 branches, Duck sessions waiting in a tool, or preempted requests once load-back (#29870) is in our version.
- **Offload does not raise live concurrency.** For 16-32 live reasoning streams the capacity levers are KV bytes per token (§1), weight bytes (INT4) and Mamba-slot reservations (§3). The host tier is not one of them.
- For append-only agent turns, a hit needs a Mamba state at the end of the previous reply. Align mode stores the state only at the last block boundary of the prompt (see §3). So the reply is re-prefilled even with offload. Offload helps a session that is evicted and later resumed with its prompt unchanged.
- Two risks for Kaggle: the jsboige wedges show that a full GPU KV plus blocking tier fetches can freeze decode; and correctness needs a gate (LMCache says it is not bit-exact; LMCache#2879 fixed garbled hybrid output).

### Gaps
- There is no published measurement of OffloadingConnector or LMCache with a hybrid GDN model plus MTP or DFlash. There is also no host↔GPU bandwidth measurement on a Kaggle L4.
- I could not confirm whether PR #29870 (load preempted requests back from CPU) is in vLLM 0.32.
- I could not confirm whether `--kv-offloading-backend native` handles MambaSpec with spec-decode blocks (MTP reserves 1+k Mamba blocks per group per request, #55533).

---

## 3. Mamba/GDN state cache sizing and prefix caching for hybrids (vLLM align mode, the reply re-prefill, SGLang equivalents)

### Takeaway
In vLLM align mode a request keeps one Mamba state, at the last block boundary of the prompt. **The state at the end of the generated reply is not kept**, so every agent turn re-prefills the previous reply. The fix is PR #50551 `--enable-mamba-decode-checkpoint`. In its 5-turn test the hit rate went from 63.6 % to 94.7 %. But the PR is open, needs a rebase, and **rejects hidden-state speculative decoding**, so it probably does not work with MTP. Align mode also costs 11-16 % throughput even with zero hits (#60008). Our E19 found that prefix caching cannot be turned off together with MTP (it corrupts output), so for now that cost cannot be avoided on vLLM. On SGLang, streaming sessions (covered earlier) remain the only shipped feature that keeps per-session KV and Mamba state outside eviction and makes the next turn token-identical.

### Cited Findings
**Align-mode semantics and the decode-checkpoint fix**
- [S] vLLM engine docs: align mode caches "the mamba state of the last token of each scheduler step when that token sits at a block-size multiple". `--enable-mamba-shared-prefix-checkpoint` adds a checkpoint at the shared-prefix junction; it is off by default and applies only with EAGLE on the Mamba group and a smaller prefix match unit. — [vLLM engine args](https://docs.vllm.ai/en/stable/configuration/engine_args/)
- [V] PR #50551 "Retain finalized Mamba decode checkpoints for prefix caching" (qianlihuang). — [vLLM #50551](https://github.com/vllm-project/vllm/pull/50551)
  - Status: open, label `needs-rebase`, with merge conflicts on Aug 17, Sep 11 and Oct 13.
  - Problem: with `--prefix-cache-retention-interval 0`, PR #45845 made retention follow replay boundaries, which dropped the finalized decode endpoint. A follow-up turn that replays the full assistant message cannot resume there.
  - Fix: pins the latest scheduler-aligned decode state privately and publishes it only on `FINISHED_STOPPED`. Aborted and length-capped requests discard it.
  - Flags: `--enable-mamba-decode-checkpoint --enable-prefix-caching --mamba-cache-mode align --prefix-cache-retention-interval 0`. The test also used `--prefix-match-unit 16`.
  - Test: Qwen3.6-35B-A3B NVFP4, TP2, 5-turn dialog. Cached tokens over all turns 23,056 → 32,736; hit rate 63.6 % → 94.7%; turn 1 cached 48 → 3,168.
  - **It rejects hidden-state speculative decoding as unsupported.** That support is future work in draft PR #60603.
  - Related drafts:
    - #53803: align grid plus decode-end checkpoints.
    - #60760: match-unit-aligned checkpoints.
    - #63808: `mamba_aware_lru` eviction.
- [S] #45238: hybrid prefix caching silently drops to 0 % when the align checkpoint lands in request-unique tokens. — [vLLM #45238](https://github.com/vllm-project/vllm/issues/45238)
- [S] #46384 "[2/N] support partial prefix cache hit for hybrid model". — [vLLM #46384](https://github.com/vllm-project/vllm/pull/46384)
- [S] #63830 proposes removing `mamba_cache_mode` entirely, leaving align or none through `enable_prefix_caching`. — [vLLM #63830](https://github.com/vllm-project/vllm/pull/63830)
- [V] #58997 (removes the deprecated `all` mode) is merged. Source: [vLLM #60008](https://github.com/vllm-project/vllm/issues/60008)

**Throughput cost of align mode (#60008)**
- [V] Setup: Nemotron-3.5-Lightning-30B-A3B-NVFP4, 4x GB200, random 8K in / 1K out, zero hits. — [vLLM #60008](https://github.com/vllm-project/vllm/issues/60008)
  - Output tok/s, prefix caching off → on:

    | Concurrency | Off | On | Change |
    |---|---|---|---|
    | c=1 | 317 | 277 | -14.9 % |
    | c=8 | 2,178 | 1,879 | -13.5 % |
    | c=32 | 7,174 | 5,476 | -16.2 % |
    | c=64 | 10,371 | 9,109 | -13.4 % |

  - Causes:
    - 28 extra eager launches per decode step (4 Mamba groups x 7 ops).
    - Align pre-process, pre-copy and post-process kernels outside the CUDA graph (+0.55-0.75 ms GPU idle per step).
    - A block-boundary split of prefills (8192 → 4176 + 4016).
  - Fix candidates: #58737 / #59726 (fused gather), #58851 (shared metadata), #57329 (internal prefill checkpoints). All open. #58851 + #57329 together cause silent corruption. The author "could not show that any combination fully closes the throughput gap".
  - A second data point (A100, Qwen3.6-35B-A3B-FP8) shows prefix caching on is about +25 % when hits exist (2,048 shared + 256 random).

**Lossy alternative**
- [V] Tail-Replay (arXiv 2608.30310, v2 2026-10-15) caches only full-attention KV and rebuilds GDN states by replaying a short suffix. With a 5-10 % replay budget it retains 92.8-99.9 % of full-prefill quality, with TTFT 9.1-14.3x faster at 32K. It is approximate, and no engine integration is named. — [arXiv 2608.30310](https://arxiv.org/abs/2608.30310)

**State sizing and concurrency**
- [S] Hybrids keep one Mamba state block per running sequence. With full decode CUDA graphs, vLLM 0.30.0 requires `max_num_seqs` ≤ the number of Mamba blocks. vLLM #49064 reports a startup failure when available Mamba blocks are fewer than `max_num_seqs`. — [vLLM #49064](https://github.com/vllm-project/vllm/issues/49064), [unsloth #13235](https://github.com/unslothai/unsloth/pull/13235)
- From the earlier notes:
  - vLLM's MTP window formula is `floor((num_gpu_blocks − 1) / (1 + G·(1+k)))` with G = 3 Mamba groups. MTP reserves 1+k Mamba blocks per group per request, and the MTP head takes about 0.88 GiB (#55533).
  - `--mamba-ssm-cache-dtype` fp16 vs fp32: the vLLM 25K-TPS blog uses bf16 and says it "significantly" increases effective KV capacity. We measured fp16 (not bf16) as the correctness-safe choice.
  - SGLang: about 18 MB per fp16 Mamba slot per GPU; `--max-mamba-cache-size`; `--enable-int8-mamba-checkpoint` (about 2x checkpoint capacity); `--mamba-radix-cache-strategy extra_buffer_lazy`.

### Inferences
- [D] For our vLLM 0.32 + MTP3 runs, each agent turn re-prefills the whole previous reply (up to 50-130k tokens). This continues until a decode-checkpoint feature supports MTP. On the SGLang side, streaming sessions or a patched refresh-on-unlock do the equivalent job.
- One custom option is to backport #50551 into the Kaggle wheel and test whether its hidden-state rejection really covers MTP (MTP drafts from target hidden states, so it probably does).
- Another option needs no engine change: end each request at a natural boundary (e.g. `</think>` or the tool call) and send the next turn as raw `prompt + generated ids` through `/v1/completions` with token IDs. That keeps the prefix token-identical. But the align state is still at the old prompt end, so it saves only the full-attention KV part, not the GDN replay.
- [D] Mamba state cost at 32 streams with MTP3: about 32 x 4 slots x about 18 MB ≈ 2.3 GB per GPU, which is about 230k fp8 KV tokens per GPU. The inputs are rough: the 18 MB SGLang slot size and the 1+k reservation per group from #55533. Spec-decode state reservations become a real share of memory at 32 streams. Dropping MTP state reservations, or moving SSM to bf16 if accuracy allows, frees concurrency. bf16 SSM was flagged as risky in our tests, so it needs a gate.

### Gaps
- No upstream vLLM release keeps the decode-end Mamba state together with MTP. I did not check whether 0.32's #59759 / #60797 ("Mamba prefill checkpoints with MTP and prefix caching") change the reply-reuse picture.
- No measurement of #60008-type align overhead on Qwen3.8-27B or L4.

---

## 4. Freeing capacity while an agent request is idle (pause, park, swap, pin), preemption policy, max-model-len, context compaction

### Takeaway
Neither vLLM nor SGLang has a shipped "park this request during a tool call" API.
- vLLM: a finished request's KV stays in the prefix cache only until LRU evicts it. Preemption is RECOMPUTE, and swap is gone in V1. The closest shipped tools are admission caps (`--max-num-active-seqs`; `max_tokens` treated as an upper bound) and, for parked sessions, CPU offload (§2).
- SGLang: streaming sessions hold KV and Mamba state outside eviction, with a 300 s idle reaper (earlier note).
- TTL pinning (Continuum) and retention-priority APIs (vLLM RFC #37003, vLLM Router #285) are research or RFC only.
- For long reasoning, the documented failure is late-run preemption when long traces pile up. The fix is an admission cap, not more streams.

### Cited Findings
- [V] vLLM V1: "the default preemption mode is `RECOMPUTE` rather than `SWAP`"; "Preempted requests are recomputed when sufficient KV cache space becomes available again". To reduce preemptions, lower `max_num_seqs` or `max_num_batched_tokens`, or raise `gpu_memory_utilization`. — [vLLM optimization docs](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [S] `max_num_active_seqs` "limit[s] how many requests can be admitted into RUNNING", while `max_num_seqs` sizes the runner and CUDA graphs. It must be ≤ `max_num_seqs`. — [vLLM scheduler config](https://docs.vllm.ai/en/latest/api/vllm/config/scheduler/)
- [S] Priming (arXiv 2605.08301) on hybrid test-time scaling: short correct traces finish early while long traces "accumulate and saturate the KV budget, forcing the scheduler to preempt". A lower cap trades early throughput for less late-run preemption, and the best cap depends on the architecture. — [arXiv 2605.08301](https://arxiv.org/pdf/2605.08301)
- [S] Continuum keeps a tool-calling request's KV on the GPU for a TTL estimated from tool-duration statistics, and evicts it when the TTL expires. It is built on vLLM. Reported delay reduction: 1.12x-3.66x. — [arXiv 2511.02230v5](https://arxiv.org/html/2511.02230v5)
- [S] vLLM RFC #37003 ("Context-Aware KV-Cache Retention API (Prioritized Evictions)") discusses the same tool-pause eviction problem; still an RFC. — [vLLM #37003](https://github.com/vllm-project/vllm/issues/37003)
- [S] vLLM Router #285: program-level Progress-TTL scheduling. — [vllm-project/router #285](https://github.com/vllm-project/router/issues/285)
- [V] The vLLM blog says KV is offloaded to CPU before preemption, and load-back is planned (#29870). — [vLLM blog](https://vllm-project.github.io/2026/01/08/kv-offloading-connector.html)
- [V][F] An example of what happens without admission control: jsboige's sessions grew to about 260K tokens, GPU KV hit 100 % and decode collapsed to 0.3-0.9 t/s. — [jsboige #128](https://github.com/jsboige/vllm/pull/128)
- From the earlier notes:
  - The SGLang streaming-session details and the Dynamo agent page (session KV "invisible to eviction"; 300 s idle timeout).
  - Franzen's KV-resident admission and trim hysteresis (128 Ki context → about 57 Ki).
  - rellik13's finding that the KV pool was the limit, not decode.
  - Our EVAL30 thrash (8.39 M re-prefill tokens) vs G4B (12 fixed workers, KV max 0.44).

### Inferences
- [D] With RECOMPUTE preemption on a hybrid, a preempted 100k-token reasoning request costs a full re-prefill: about 65 s at 1,500 tok/s, during which every live decode slows. One preemption per tail is worth several minutes of aggregate decode. Keep the planning rule `Σ(expected peak context of admitted requests) ≤ 0.9 x KV pool` (rev 6 already does this). Set `--max-num-active-seqs` (vLLM ≥ 0.31) to the admission cap and keep `max_num_seqs` larger for graph capture.
- For ARC-AGI-2 single-turn long thinking, "idle in a tool call" barely happens. Parking matters only for Duck-style multi-turn agents. There, the cheapest shipped option on vLLM is a per-session `max_tokens` budget plus admission at the KV limit. On SGLang it is streaming sessions with a longer idle timeout.
- Context compaction (trim or summarize) for ARC-AGI-2 conflicts with the "long uninterrupted thinking" rule in our memory. It applies only to tool-loop history, not to the reasoning stream.

### Gaps
- I found no source that confirms whether vLLM fully recomputes the Mamba state for a preempted hybrid request, or how `--max-num-active-seqs` interacts with MTP Mamba-block reservations.
- No shipped pin/TTL API was found in vLLM 0.32 or SGLang 0.5.21 beyond streaming sessions.

---

## 5. Harness-level throughput: keeping concurrency high in the tail, chunked-prefill size, prefill/decode interference

### Takeaway
The engine docs say little here. vLLM batches all pending decodes before any prefill, so a smaller `max_num_batched_tokens` keeps decode latency (ITL) low but makes the 50-130k re-prefills of agent turns take many steps. The 25K-TPS recipe found +8 % at 16384 on prefill nodes. For long reasoning, the tail is the main loss: as short traces finish, the live batch shrinks and aggregate tok/s falls with it. The harness has to refill the batch by issuing more parallel work (extra attempts, branch splits, task ordering) while respecting the KV admission rule. The engine cannot do this for us.

### Cited Findings
- [V] vLLM V1 "batches all pending decode requests before scheduling any prefill operations". "Smaller values (e.g., 2048) achieve better ITL because there are fewer prefills slowing down decodes"; higher values give better TTFT; "For optimal throughput, we recommend setting `max_num_batched_tokens > 8192` especially for smaller models on large GPUs". The page does not mention `--max-num-partial-prefills` or `--long-prefill-token-threshold`. — [vLLM optimization docs](https://docs.vllm.ai/en/stable/configuration/optimization/)
- [V] Align mode splits prefills at block boundaries (an 8192 prompt became 4176 + 4016), so each step pays the fixed host cost twice. This is the main TTFT cost at c=1 (about x1.8). — [vLLM #60008](https://github.com/vllm-project/vllm/issues/60008)
- [S] Priming paper: in long-reasoning batches the late run is dominated by the long traces that remain, which saturate KV. — [arXiv 2605.08301](https://arxiv.org/pdf/2605.08301)
- From the earlier notes:
  - Our vLLM config uses `max_num_batched_tokens 4096` (our_measurements.md, R5 log).
  - vLLM 0.31 schedules requests that already hold KV first (#58947).
  - vLLM 0.31 has a throughput cliff when `max_num_seqs` is not a multiple of 8 (#57355).
  - The 25K-TPS blog found `--max-num-batched-tokens 16384` gave +8 %.
- From our memory (not external): ORDER2 ordering gave +4-7 vs random; independent attempts are complementary (D66/L66 union 50 vs 28/34); attempt_2 was unused (+1).

### Inferences
- [D] Tail refill is the harness lever with the most evidence behind it. When live requests fall below the target (e.g. 24), start the next task's attempts or a second independent attempt of an unfinished task immediately. Never leave slots idle while work is queued. Our own D66/L66 data says extra independent attempts add solves, so tail filling raises both tok/s and accuracy.
- [D] At 4,096 tokens per step, a 90k re-prefill takes about 22 mixed steps, and each one slows every decoding stream. Option: raise `max_num_batched_tokens` to 8192-16384 so re-prefills finish in fewer steps. Weigh that against ITL; measure aggregate tok/s, not ITL, since nothing is interactive. With MTP, the verify tokens of 32 streams (32 x 4 = 128) also count against this budget.
- [D] Use `max_num_seqs` in multiples of 8 (16, 24, 32) because of the #57355 cliff.

### Gaps
- No public measurement of chunk size vs aggregate tok/s for hybrids at 50-130k contexts on PCIe GPUs.
- I did not verify that `--max-num-partial-prefills`, `--max-long-partial-prefills` and `--long-prefill-token-threshold` exist in vLLM 0.32, or how they behave; the stable optimization page does not list them.

---

## 6. Measured concurrency vs tok/s for Qwen3.5/3.6/3.8 hybrids on 24 GB-class GPUs, and what bounds 4x L4

### Takeaway
Public multi-stream numbers for 27B hybrids on 24 GB-class GPUs are scarce. The best comparable one is club-3090: 4x RTX 3090, SGLang, Qwen3.8-27B W8A8 + DFlash2, c=16, about 1,811 tok/s decode, from the earlier note. Scaled by memory bandwidth, that roughly matches our 597 tok/s on 4x L4. Simple bandwidth arithmetic shows that at 40-130k contexts, **reading the KV dominates each decode step on L4.** Going from about 300 to 700-1,000 tok/s therefore needs fewer bytes read per generated token: more streams per weight pass, high spec acceptance, and above all fewer KV bytes (NVFP4 KV). More streams alone give sublinear gains at long context.

### Cited Findings
- From the earlier `agent_effective_throughput.md` (club-3090 #1288): 4x RTX 3090 (220 W cap), SGLang 0.5.19, Qwen3.8-27B W8A8 + DFlash2, TP4, `max_running_requests` 16, HiCache ratio 1.5. Prefill 4,000-4,300 tok/s at 10k context and 3,029 at 90k; c=16 decode about 1,811 tok/s. The context at that decode figure is not stated. — [club-3090 #1288](https://github.com/noonghunna/club-3090/issues/1288)
- [V][F] jsboige SM89 fork (vLLM 0.32, fp8 KV; model and GPU count not stated): N=48 aggregate 886.4 tok/s; N=1 41.1. — [jsboige #123](https://github.com/jsboige/vllm/pull/123)
- [V] #38230: Qwen3.5-4B-FP8 on an RTX 4080 Super, 50 concurrent at 1,061 tok/s with a 95 % external hit rate (small model). — [vLLM #38230](https://github.com/vllm-project/vllm/issues/38230)
- [S] Single-stream only: Qwen3.6-27B-FP8 on an RTX 4090D 48 GB, vLLM about 48 tok/s. The LLMKube 2x16 GB TP2 bakeoff says vLLM is about 3-4x llama.cpp at high concurrency, but its high-concurrency cells timed out. — [HF discussion](https://huggingface.co/Qwen/Qwen3.6-27B-FP8/discussions/11), [LLMKube](https://llmkube.com/blog/qwen3-6-27b-bakeoff)
- [V] Iso-capacity NVFP4 vs fp8 on Blackwell: +37 % / +58 % / +78 % decode at 32K / 160K / 1M. fp8 collapses beyond C = 12 in AgentX while NVFP4 keeps rising. — [LMSYS NVFP4 KV blog](https://www.lmsys.org/blog/2026-09-16-nvfp4-kv-cache/)
- Hardware spec, not fetched this session (from NVIDIA product pages): L4 has about 300 GB/s of memory bandwidth per GPU and PCIe Gen4 x16; RTX 3090 has about 936 GB/s. — [NVIDIA L4](https://www.nvidia.com/en-us/data-center/l4/), [RTX 3090](https://www.nvidia.com/en-us/geforce/graphics-cards/30-series/rtx-3090-3090ti/)

### Inferences
[D] Per-GPU decode-step byte model. These are my assumptions: weights read once per step; each stream reads its full KV; Mamba state read and written once; effective bandwidth about 250 GB/s per L4.

**Inputs**
- Weights per GPU: about 7 GB for FP8 (our log: 7.15 GiB with MTP) and about 4-4.5 GB for INT4. The INT4 figure is my estimate.
- fp8 KV: about 10 KB per token per GPU.
- Mamba state: about 18 MB per stream per GPU, read and written each step (about 36 MB of traffic).

**Step time and throughput**

| Case | KV read | Total read | Step time | tok/s at 2.0-2.5 accepted tokens per stream per step |
|---|---|---|---|---|
| 16 streams x 65k avg, INT4 | 10.4 GB | ≈ 4.5 + 10.4 + 0.6 ≈ 15.5 GB | ≈ 62 ms | 16 x 2.0-2.5 / 0.062 ≈ 520-650 |
| 32 streams x 65k, fp8 KV | 20.8 GB | ≈ 26.5 GB | ≈ 106 ms | ≈ 600-750 |
| 32 streams x 65k, NVFP4 KV (≈ 0.56x bytes) | ≈ 11.7 GB | ≈ 17.4 GB | ≈ 70 ms | ≈ 910-1,140 |

- The 16-stream row is the same order as our SGLang 536 at 40k and vLLM INT4 439, so the model is plausible.
- The 32-stream fp8 row only fits if KV holds 2.08M tokens per GPU, more than the 1.71M INT4 pool. It is not reachable without KV compression or a shorter average context.
- The NVFP4 row fits in about 3.0M tokens (1.71M x 1.75). That is the only path in this arithmetic that reaches 900+ at long context, and only if the FA2 NVFP4 kernel reads at near bandwidth, keeps CUDA graphs and keeps MTP acceptance. The sm_89 evidence (§1: -13..-17 %, single-token graphs only) says it does not yet.

**Ranked levers, by this arithmetic and the cited evidence**
1. A KV compression that keeps graphs and MTP. NVFP4 is today's candidate; it needs an L4 test with MTP.
2. Tail refill and admission, so live streams stay at the KV-feasible maximum all run (§5).
3. Cutting per-turn re-prefill: decode checkpoints (#50551) or SGLang streaming sessions, plus a larger `max_num_batched_tokens` (§3, §5).
4. Lower Mamba-slot reservations (§3).
5. CPU offload, only for parked sessions (§2).

The ratio check: 4x3090 has about 3.1x the bandwidth of 4x L4, and 1,811 / 3.1 ≈ 584, close to our 597. This suggests our SGLang engine is already near bandwidth parity per byte, and further gains must come from reading fewer bytes, not from engine overhead. It is only suggestive: the club-3090 context length, W8A8 weight size and DFlash acceptance differ from ours.

### Gaps
- No public concurrency sweep (8/16/24/32 streams at 40-130k) for Qwen3.5/3.6/3.8-27B on L4, 4090 or L40S was found.
- The byte model ignores attention compute, all-reduce over PCIe (covered in the earlier `nccl_p2p_l4.md`) and MTP draft cost. Treat it as an upper bound, not a forecast.
- Qwen3.8-Flash-Next: no sub-fp8 KV, offload or concurrency measurement on Ada was found. The LMCache page lists it as validated, on unknown hardware. Our own E13 found its 0.5.21 output is garbage on L4 (memory note).
