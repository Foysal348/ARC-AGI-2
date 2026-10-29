# Speculative decoding at high batch and the fastest sm_89 (L4) kernels for Qwen3.8-27B TP4 (state as of 2026-10-29)

Scope note: this file adds to the two earlier note sets and does not repeat them:
- `research_notes/Qwen 27B দ্রুততম সার্ভিং সেটআপ/speculative_decoding.md` (AL per method, MTP vs DFlash2 vendor curves, mamba-slot memory)
- `research_notes/Qwen 27B দ্রুততম সার্ভিং সেটআপ/weight_quantization.md` (kernel matrix per format)
- `research_notes/৪xL4 ইঞ্জিন বটলনেক যাচাই/dflash_width_correctness.md`, `newer_engine_versions.md`, `per_step_overheads.md` and `profiling_and_levers.md` (DFlash2 width, #36548, vLLM 0.30-0.32 release scope, lm_head and verify cost)

Labels: **[V]** = verified (measured by a named party, or read directly in vLLM 0.32.0 source or our own logs). **[C]** = claim (vendor, blog or unreplicated report). **[I]** = my inference.

The brief's own measurements are cited as "[Internal: task brief]". In these numbers the 8k/40k/90k triplets are aggregate output tok/s at each context depth.

---

## Q1. Which spec-decode method raises aggregate tok/s at 16-32 concurrent 50k-130k reasoning requests? Acceptance vs batch, where it stops paying, adaptive/dynamic speculation, tree vs chain

### Takeaway
At 16-32 streams on 4xL4, MTP and DFlash2 still pay. Our own numbers show DFlash2 W6 (SGLang) is the fastest and MTP 2-3 is the best vLLM setting. The decisive finding is that our step time is about 107-125 ms at 8k context, while the weight-read floor is only about 12-23 ms. That makes per-step overhead (PIECEWISE graphs, GDN verify kernel, PCIe all-reduce, lm_head) the binding cost, not the draft method. The off-the-shelf "turn spec off at high batch" features in vLLM (dynamic SD, `num_speculative_tokens_per_batch_size`) are broken or regress throughput in every 2026 report I found, so they should not be used. Chain drafts (topk 1) remain the only sensible shape at this batch size.

### Cited Findings
**Our baseline (from the brief)**
- **vLLM 0.32 FP8, 16 streams:** MTP3 434/399/281 tok/s, MTP2 448/397/260, MTP4 399/381/267 — [Internal: task brief]
- **vLLM 0.32 INT4, MTP3, 20 streams:** 518/439/332 — [Internal: task brief]
- **SGLang INT4 + DFlash2 W6, 20 streams:** 597/536/364 — [Internal: task brief]
- **MTP acceptance:** mean accepted about 1.9 of 3 drafts per step in real agent runs — [Internal: task brief]

**Dynamic / adaptive speculation in vLLM (all broken or regressing)**
- **vLLM #49548 (OPEN, opened 2026-07-23):** `num_speculative_tokens_per_batch_size` `[[1,4,2],[5,512,0]]` (MTP off at batch ≥5) on Qwen3.5-122B INT4 cut 8-concurrent aggregate throughput from about 232 tok/s (static k=2) to 24-157 tok/s. Wall time rose from 6.3 s to 40-60 s **[V, reporter]**. — [vLLM #49548](https://github.com/vllm-project/vllm/issues/49548)
  - Each candidate root cause (H1-H3) was later retracted or disputed. A commenter called the remaining gap "currently unidentified" (2026-08-09).
  - On a ~49k-token prefix-cache workload (ThinkingCap-Qwen3.6-27B-FP8): no-spec about 178 t/s, stock always-K=0 dynamic 89.4 t/s, and 147-151 t/s with the EAGLE cache-hit block drop disabled. Greedy output identity is lost under those patches **[C, single rep]**.
  - **H100 tax of dynamic SD at K=0 vs no-spec (vLLM 0.27.1, gemma-4-31B):** TPOT +6.56 % at concurrency 2 and +21.58 % at concurrency 189, and a 10.6 % smaller KV pool **[C]**.
  - **DSpark dynamic vs static K=5 on 4xH20:** −68.8 % at c5, +22.8 % at c20, +7.6 % at c60. Step time rose from 121 ms to 423 ms at c5 **[C]**.
  - Fix PRs #52070, #53426 (`skip_draft_when_k0`) and #56136 (per-query-length CUDA-graph ladder) are proposed and not merged. #51466 was closed as unsupported.
- **vLLM #58692:** Dynamic SD plus an MTP drafter on the V2 model runner crashes with `ZeroDivisionError` in `SpeculatorCudaGraphManager._init_candidates` (nightly, about late September 2026) **[C]**. — [vLLM #58692](https://github.com/vllm-project/vllm/issues/58692)
- **vLLM #25112:** the older `disable_by_batch_size` setting had no effect and was not referenced in code in v0.10.1 **[C]**. — [vLLM #25112](https://github.com/vllm-project/vllm/issues/25112)
- **vLLM spec-decode README (main, late 2026):** lists "Dynamic Speculative Decoding" and "Adaptive Verification". It says adaptive verification "[s]izes verification per request from drafter confidence; currently **DSpark only**", rates dynamic SD "Higher than base SD method" at high QPS, and rates n-gram and suffix "Medium" at high QPS. — [vLLM spec decode README](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md)
  - **Conflict:** the v0.30.0 release notes (cited in the earlier note `newer_engine_versions.md`) describe MRV2 "adaptive verification for every draft-model speculator through an online acceptance estimator (#52228)". The README's "DSpark only" wording contradicts "every draft-model speculator". Check the installed version's code before relying on either.
  - The v0.32.0 notes also list "dynamic K in autoregressive speculators (#57053)" for MRV2 — [Internal: _scratch/eng_research/vllm_v0.32.0.md](D:/ARC%20AGI%202/_scratch/eng_research/vllm_v0.32.0.md)

**Papers and batched-SD evidence**
- **D-cut (arXiv 2607.14647, 2026-07-16):** prunes draft depth jointly across the batch, using draft confidence and a runtime cost model. Under high concurrency, average speedup rose from 1.26x to 1.65x, and dense-model configurations where long drafts were slower than AR "are restored to acceleration". No batch threshold, model names, hardware or code are given **[C]**. — [arXiv 2607.14647](https://arxiv.org/abs/2607.14647)
- **DFlash paper (vLLM, Qwen3.5-9B):** about 1.3-2.1x over AR at concurrency 32. The z-lab Gemma-4-31B DFlash card claims 1.9-3.8x at c32. Neither covers long contexts. (Search-snippet level; the tables were not fetched.) — [DFlash arXiv 2602.06036](https://arxiv.org/pdf/2602.06036); [z-lab/gemma-4-31B-it-DFlash](https://huggingface.co/z-lab/gemma-4-31B-it-DFlash)

**Fused verify kernel gains by concurrency (Qwen3.8-Flash-Next NVFP4, MTP3, 8K in / 1K out; vLLM PR #59366, Draft)**
- The PIECEWISE + Triton GDN verify path loses 35-45 % at c1-c2 against native and is roughly even by c8-c32. FULL graphs plus the fused CUDA replay kernel recover native speed at low c and beat it at c32-64 (for example 1295 → 1519 tok/s at c32 on 2x RTX PRO 6000) **[C, author]**. — [vLLM PR #59366](https://github.com/vllm-project/vllm/pull/59366)

**Suffix / n-gram decoding**
- **Mechanism:** pattern-matches the prompt and prior generations, with an adaptive number of speculated tokens per step. Keys: `{"method":"suffix","num_speculative_tokens":N}`, `suffix_decoding_max_tree_depth` (24), `suffix_decoding_max_cached_requests` (10000), `suffix_decoding_max_spec_factor` (1.0), `suffix_decoding_min_token_prob` (0.1). — [vLLM suffix docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/); [vLLM spec decode README](https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/README.md)
- **Target workloads:** the docs name repetitive work ("code-editing, agentic loops (e.g. self-reflection, self-consistency), RL rollouts"), not free-form reasoning. — [vLLM suffix docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/suffix/)
- **Snowflake:** SuffixDecoding beats n-gram in vLLM at all concurrency levels. An earlier version's CPU speculation overhead "began to dominate" at about 64 concurrent requests and was later optimized **[C]**. — [Snowflake engineering blog](https://www.snowflake.com/en/engineering-blog/suffixdecoding-arctic-inference-vllm/)
- **vLLM v0.32.0** includes "correctness fixes for ngram on hybrid recurrent models (#56531)" and an MRV2 "GPU n-gram speculator (#40704)". — [Internal: vllm_v0.32.0.md](D:/ARC%20AGI%202/_scratch/eng_research/vllm_v0.32.0.md)

### Inferences
- **[I] Step-time arithmetic from the brief's numbers.** MTP3 with about 1.9 accepted drafts gives about 2.9 tokens per stream-step.
  - FP8 16 streams at 8k: 434 / (16 × 2.9) ≈ 9.4 steps/s, so about **107 ms per step**.
  - INT4 20 streams: 518 / (20 × 2.9) ≈ 8.9 steps/s, so about **112 ms**.
  - DFlash2 W6 (taking the earlier note's L6 ≈ 3.5-3.95): about 597 / (20 × 3.75) ≈ 8 steps/s, so about **125 ms**.
  - Weight-read floor per step at 300 GB/s per L4: FP8 about 6.75 GB/GPU, or about 22 ms; INT4 about 4.9 GB, or about 16 ms.
  - Halving weight bytes (FP8 → INT4) changed step time by about 5 %. The step is therefore not weight-bandwidth-bound at 8k. Roughly 80 ms per step is overhead that no draft method removes.
  - Reaching 900 tok/s at 16 streams × 2.9 tokens needs about 52 ms per step. That is a kernel and graph problem, not an acceptance problem.
- **[I] Draft length.** At 16-20 streams, MTP2 ≈ MTP3 > MTP4 at 8k-40k, and MTP3 wins at 90k. This matches the "shorter drafts at high batch" evidence in the earlier notes. More draft tokens will not help until the per-step overhead drops.
- **[I] Static per-workload K is the safe policy.** Pick one K per phase from an offline sweep. Do not use vLLM's dynamic SD until #49548 has a merged fix and a reproduction on hybrids.
- **[I] Suffix/n-gram is unlikely to beat MTP on free-form ARC reasoning, but it could be added on top.** ARC traces repeat grid rows and code. Suffix decoding's CPU cost at 16-32 streams and its untested acceptance on Qwen3.8 thinking text make it a measured arm only. vLLM cannot stack MTP and suffix in one config, as far as I found.

### Gaps
- No published acceptance-vs-batch curve for any method at 50k-130k context on a hybrid 27B model. No L4 or Ada spec-decode throughput curve beyond our own.
- No source measures EAGLE-3 for Qwen3.8-27B. No public EAGLE-3 head for Qwen3.8-27B was found in this pass.
- Lookahead decoding: no 2026 vLLM or SGLang support for hybrids was found.
- Whether #52228 adaptive verification applies to MTP in 0.32 (README conflict above) could not be settled without reading `vllm/v1/worker/gpu/spec_decode/*` in full.

---

## Q2. Correctness of spec decode on hybrids (GDN/Mamba state rollback) and known 2026 bugs causing silent accuracy loss; why SGLang INT4 + DFlash2 scores 6-7 points below vLLM FP8

### Takeaway
The strongest public match for our 6-7 point SGLang gap is still **SGLang #36548**: DFlash at concurrency 8 on Qwen3.8-27B gave 36.5 % wrong answers on a greedy probe, against 0 % without speculation. The bot closed it as **stale on 2026-10-27, with no fix merged**. The ablations that helped (width 4, fp32 SSM, no decode graphs) all left about 37/304 wrong. The only near-cure was a **BF16 (dense) lm_head target export**, which brought it to 1/304. Two other SGLang reports must be ruled out as cheap checks:
- **#42917:** a stale `~/.cache/sglang/nv` CUDA compute cache once made a W4A16 Qwen3.8-27B serve with PPL 9.98 against 6.05 on vLLM.
- **#43514:** DFlash accept rate collapses to 0 under grammar plus multi-chunk prefill, with silent "!" output.

### Cited Findings
**#36548: DFlash on Qwen3.8-27B under concurrency**
- **Status:** opened 2026-08-26, closed "due to inactivity" by github-actions on 2026-10-27, not by a fix.
- **Wrong answers (greedy, fixed seeds):**
  - DFlash8 serial 0/100.
  - DFlash8 at c8: 7/100, then 111/304 (36.5 %).
  - No spec at c8: 0/304.
  - Width-one TARGET_VERIFY (all drafts rejected) at batch 1 diverges from plain decode in 19/32 cases.
  - fp32 SSM, decode CUDA graphs off, draft depth 4 and the #36014 beta hunk each gave 37/304.
- **lm_head finding:** switching the target to `RadixArk/Qwen3.8-27B-NVFP4-BF16-LMHead` (dense BF16 lm_head) gave 1/304, described as "useful in practice but not a fix".
- **Root-cause analysis:** the first differing value is the "selected-step SSM transition output" in layer 0. The analyst suspects a DECODE vs TARGET_VERIFY kernel difference amplified by batch composition.
- **Other report:** the original reporter's benchmark scored about 50 % against about 90 % public. The reporter could not reproduce on 0.5.18 with MTP/EAGLE.
- The target was NVFP4; INT4 was not tested. — [SGLang #36548](https://github.com/sgl-project/sglang/issues/36548)

**#43514: accept rate collapses to 0 (opened 2026-10-10; v0.5.19/0.5.20; 4xH20 TP4; Qwen3.8-27B + DFlash2)**
- Every request in a "poisoned" batch emits token id 0 ("!") until max_tokens, with accept length 1.00 and no error.
- Trigger: grammar-constrained requests (`tool_choice:"required"`) plus multi-chunk prefill reshaping the batch, with about 30 requests of about 50k-token prompts. Failure rate about 30 % per round on a used instance; about 12 % of production traffic affected.
- An unmerged upstream PR (#43518) is referenced. — [SGLang #43514](https://github.com/sgl-project/sglang/issues/43514)

**#42917: W4A16 PPL gap that turned out to be a stale cache (opened 2026-10-07)**
- Checkpoint: `dbirks/Qwen3.8-27B-W4A16-AutoRound` (compressed-tensors pack-quantized, symmetric, group 128).
- PPL 9.63-9.98 on SGLang 0.5.19-0.5.21 against 6.05 on vLLM 0.30.0 (KLD 0.55-0.59 vs 0.042).
- **Root cause (reporter, 2026-10-10):** a stale CUDA driver compute cache at `~/.cache/sglang/nv` (`CUDA_CACHE_PATH=$SGLANG_CACHE_DIR/nv`) persisted across containers. After emptying it, SGLang matched vLLM (PPL 6.28 on 5 chunks).
- The reporter concluded "there is no problem in the compressed-tensors WNA16 path". — [SGLang #42917](https://github.com/sgl-project/sglang/issues/42917)

**#38009: thinking-mode greedy divergence**
- DFlash2 greedy output diverges from target-only Qwen3.8-27B when thinking is on, at completion token 30 (BF16, no KV quant). Non-thinking output matched. — [SGLang #38009](https://github.com/sgl-project/sglang/issues/38009)

**SGLang cookbook for Qwen3.8-27B (validated on v0.5.19)**
- GDN state slot: 153.9 MB fp32 vs 78.4 MB bf16.
- "Treat bf16 as an accuracy gate and validate it for your workload."
- With speculation, fp32 vs bf16 speed goes both ways: NVFP4+EAGLE 152.9 vs 144.5 tok/s/user, FP8+EAGLE 106.3 vs 116.1.
- `--enable-linear-replayssm-spec` auto-selects fp32 state when `--mamba-ssm-dtype` is unset. An explicit non-fp32 value logs a state-drift warning at boot.
- `extra_buffer` = 5 state slots per request, `extra_buffer_lazy` = 4, `no_buffer` = 3.
- The cookbook lists no INT4 checkpoint (only BF16, block-FP8 and NVFP4).
- Recommended MTP: `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`.
- Recommended DFlash2: `--speculative-algorithm DFLASH --speculative-draft-model-path incoai/Qwen3.8-27B-DFlash2 --speculative-num-draft-tokens 8`. — [SGLang Qwen3.8-27B cookbook](https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B)

**fp16 SSM evidence**
- An NVIDIA SGLang PR (#41968, Mamba2 compact speculative-state replay) ran with `--mamba-ssm-dtype float16` and measured GSM8K 95.91 % against 96.21 % for the stock snapshot path. That is Mamba2, not GDN, and uses a stochastic-rounding flag **[C]**. — [SGLang PR #41968](https://github.com/sgl-project/sglang/pull/41968) (search snippet)

**vLLM hybrid spec state status (0.32)**
- Still open: MTP + prefix-cache corruption #53912 and spec-decode prefix-hit loss #54360 (detail in the earlier `newer_engine_versions.md`).
- Our own E19 A/B found that only "MTP + prefix caching OFF" fails (4-12 natural flips). The S1 engine config passed — [Internal memory: E19](C:/Users/hp/.claude/projects/d--ARC-AGI-2/memory/MEMORY.md)
- **RecoverSSM** (vLLM PR #58863, open, needs-rebase 2026-10-24) replaces per-draft state snapshots with a read-only checkpoint plus fp32 per-token replay records and a single commit.
  - Author numbers (GB10): KV tokens in 4 GiB 75,678 → 103,953 (+37 %); c4 verify cycle −5.2 %; agent loop −16 %/turn; prefix hit 77.4 → 82.1 % **[C]**.
  - — [vLLM PR #58863](https://github.com/vllm-project/vllm/pull/58863)

### Inferences
- **[I] Most likely cause of our gap: SGLang's DFlash verify-path numeric drift under concurrency (#36548 class).** It is not fixed in any SGLang release as of 2026-10-29, and it was amplified in the reporter's setup by a quantized lm_head. Two things to check:
  - Does the RedHat INT4 checkpoint keep lm_head in BF16? Most compressed-tensors W4A16 exports leave lm_head unquantized; the dbirks card above lists `lm_head` among unquantized modules.
  - Our fp16 SSM override adds rounding at every committed step. The SGLang cookbook gates even bf16, and #36548 shows fp32 did not cure the drift. fp16 is therefore not proven to be the cause, but it is unvalidated.
- **[I] Cheap rule-outs before any GPU A/B:**
  1. Clear `$SGLANG_CACHE_DIR/nv` (and any persisted `CUDA_CACHE_PATH`) in the Kaggle dataset or wheelhouse — the #42917 cause.
  2. Confirm no grammar/tool_choice traffic in Duck turns — #43514.
  3. Run SGLang INT4 **without** DFlash2 against vLLM FP8 without spec on the same EVAL subset. That separates "quant/engine" from "spec verify drift".
- **[I] The vLLM path is the safer accuracy base.** Its 0.32 MTP verify kernel is fixed-function (see Q4), and E19 verified that the S1 config holds accuracy.

### Gaps
- No public report of RedHat `Qwen3.8-27B-INT4` accuracy on SGLang 0.5.21/0.5.22 vs vLLM.
- No public GDN fp16-SSM accuracy study for Qwen3.5/3.8 with spec decode.
- No fix PR for #36548 has a merge or release; its fate after the stale closure is unknown.

---

## Q3. GEMM kernels on sm_89 at batch 16-64 (M = streams × verify tokens ≈ 64-200)

### Takeaway
On vLLM 0.32 with the block-128 FP8 checkpoint, our L4 log shows **`MarlinFP8ScaledMMLinearKernel`**, meaning **weight-only FP8 Marlin (W8A16)**, not FP8 tensor-core CUTLASS. This changed from vLLM 0.19, where it was the Triton block kernel. INT4 also runs on Marlin (W4A16) and reads half the weight bytes, which explains INT4's 7-19 % edge in our runs.
- An open PR (#61224, 2026-10-12) adds a CUTLASS block-FP8 W8A8 kernel for SM89 and claims +42 % end-to-end throughput on L4 (Qwen3-8B-FP8). The gain is in compute-bound shapes; it is about 3.6 % slower at batch 1.
- **Machete** is Hopper-only. **AllSpark** is Ampere-only (it rejects SM90/100; SM89 status was not confirmed).
- W4A8 INT on Marlin was being fixed in #38066 (tested on L40/SM89). No W4A8 Qwen3.8-27B checkpoint is in our stack.

### Cited Findings
- **Our L4 log (E21 smoke run 1, NB2, vLLM 0.32, FP8):** `Selected MarlinFP8ScaledMMLinearKernel for Fp8LinearMethod` **[V]**. — [Internal: E21 SMOKE_GPU_run1 nb2 halfB server log](D:/ARC%20AGI%202/experiments/Experiment%2021%20-%20Duck%20config%20B%20on%20vLLM%200.19%20+%200.32%20(planned)/Experiment%20Results/SMOKE_GPU_run1_2026-10-29/nb2/q27x/halfB/vllm_server_halfB_primary.log)
  - Earlier vLLM (0.19, E12-V log) selected CUTLASS for per-channel FP8 and ran block-FP8 through untuned Triton configs (see the earlier `weight_quantization.md`). The kernel path changed between versions.
- **vLLM PR #61224 "CUTLASS blockwise FP8 GEMM for SM89 (Ada) with per-token activation scales" (OPEN, opened 2026-10-12, awaiting review):**
  - It states that on SM89, block-FP8 checkpoints "currently run through Marlin weight-only".
  - It selects `CutlassFp8BlockScaledMMLinearKernel` once `cutlass_scaled_mm_supports_block_fp8` is true on SM89 (CUDA ≥ 12.4). It needs a build with `TORCH_CUDA_ARCH_LIST=8.9`, so a source build is required.
  - **L4, Qwen3-8B-FP8 vs Marlin:** 1929 → 2739 tok/s (1.42x); batch-32 latency 20.16 → 15.11 s; batch-1 latency 4.77 → 4.94 s (3.6 % slower).
  - **Microbench (µs):** gate_up M=4096 Marlin 26,559 → 8,902; down_proj M=4096 9,003 → 6,159 (Triton 5,584); gate_up M=1 399 → 432.
  - GSM8K 0.892 → 0.904/0.896 **[C, author]**. — [vLLM PR #61224](https://github.com/vllm-project/vllm/pull/61224)
- **vLLM 0.32.0 source refs** include kernel modules `scaled_mm/{cutlass, triton, marlin, flashinfer, deep_gemm, b12x, humming, pytorch}` and `mixed_precision/{marlin, machete, cutlass}`. The v0.32.0 notes say "Humming preferred over Marlin on SM90 (#56997)" (SM90 only). — [Internal: vllm_0.32.0 source refs](D:/ARC%20AGI%202/_scratch/engine_audit_pack_rev2e/source_refs/vllm_0.32.0); [Internal: vllm_v0.32.0.md](D:/ARC%20AGI%202/_scratch/eng_research/vllm_v0.32.0.md)
- **Machete:** a mixed-input GEMM "optimized for NVIDIA Hopper GPUs" — [Red Hat Machete article](https://developers.redhat.com/articles/2024/10/14/introducing-machete-mixed-input-gemm-kernel)
- **W4A8 status:**
  - llm-compressor #2693 logs show the CUTLASS W4A8 kernel accepts only FP8 activations (Hopper), and AllSpark rejects capability 90 and 100 **[C]**. — [llm-compressor #2693](https://github.com/vllm-project/llm-compressor/issues/2693)
  - vLLM PR #38066 fixes W4A8-INT activation quantization in Marlin (tested on L40 SM89, Qwen2.5-7B). Without it, Marlin "silently runs W4A16" — [vLLM PR #38066](https://github.com/vllm-project/vllm/pull/38066)
  - Community `Qwen3.8-27B W4A8 + DFlash2 on SGLang v0.5.19 (TP4)` patch for 3090s exists (club-3090 PR #1226; not fetched) — [club-3090 PR #1226](https://github.com/noonghunna/club-3090/pull/1226)
- **Batch crossover:** on an A6000 (Llama-3.1-8B), W4A16 wins at low concurrency and W8A8 at high concurrency — ["Give Me BF16 or Give Me Death", arXiv 2411.02355](https://arxiv.org/pdf/2411.02355)
- **L4 hardware:** about 300 GB/s; FP8 485 TFLOPS sparse (about 242 dense); BF16 242 sparse (about 121 dense) — [Thunder Compute L40/L40S/L4 spec article](https://www.thundercompute.com/blog/nvidia-l40-vs-l40s)
- **L4 72 W power limit, SM clock max 2040 MHz.** vLLM #59548 measured spec-decode throughput CV of up to 13.92 % across boots on L4 with vLLM 0.29 (CUDA-graph default at 0.49-0.77x eager). On 0.30.0 the ratio was about 1.01 with CV 1-1.6 %. Power-cap clamping is an unconfirmed hypothesis. — [vLLM #59548](https://github.com/vllm-project/vllm/issues/59548)

### Inferences
- **[I] FP8 adds no compute advantage over INT4 at our decode M on 0.32.** On 0.32 our FP8 is W8A16 Marlin, so it dequantizes into BF16 tensor-core math exactly like INT4 Marlin, while reading 2x the bytes.
- **[I] Where #61224 would help.** The +42 % is on compute-bound prefill-heavy shapes. At M ≈ 64-128 per GEMM (16-32 streams × 4 verify tokens), Ada FP8 tensor cores could help the verify GEMM once dequant cost dominates. That is plausible but unmeasured. Building #61224 into our wheel (SM89-only build) is a measurable arm for prefill (TTFT at 50k+ contexts) more than for decode.
- **[I] INT4 Marlin remains the best decode GEMM on L4 today.** Expected gain from any GEMM change is bounded by the weight share of step time: about 16-22 ms of a 107-112 ms step (Q1).

### Gaps
- No ExLlamaV3 tensor-parallel batched (16-64) throughput for 27B hybrids. ExLlamaV3 has no Qwen3.8 GDN + MTP serving path found.
- No AllSpark or QQQ status for SM89 in vLLM 0.32 found. No published Marlin W4A16 vs FP8-Marlin microbench at M = 64-200 on L4.
- Whether `VLLM_MARLIN_INPUT_DTYPE` (`marlin_input_dtype` in `fp8.py`) can switch FP8-Marlin to FP8 activations on SM89 was not resolved.

---

## Q4. Gated DeltaNet / FLA kernels: fused recurrent vs chunked, the "GDN spec decode kernel: cuda" path in vLLM 0.32, Triton vs CUDA, L4 slowness

### Takeaway
In vLLM 0.32.0 the fused CUDA MTP verify kernel (`fused_gdn_decode_post_conv_mtp`) **is** eligible for Qwen3.8-27B. The head-ratio gate allows ratios 1/2/3/4/8; 48V/16K = 3, so the August "unreachable" report is obsolete. It requires:
- **BF16 or FP32 recurrent (SSM) state**
- BF16 conv state and a BF16 model dtype
- head dims K = V = 128
- ≤ 8 spec tokens
- **pure spec-decode batches** (no prefills in the batch)

Our own E21 run with `--mamba-ssm-cache-dtype float16` logged **"Falling back to the Triton GDN spec decode path"**. fp16 SSM therefore silently costs the fused kernel. Every step that contains a prefill chunk also falls back to Triton.

### Cited Findings
- **vLLM 0.32.0 `qwen_gdn_linear_attn.py` [V, source read]:**
  - `MAX_FUSED_GDN_MTP_TOKENS = 8` and `FUSED_GDN_STATE_DTYPES = (torch.float32, torch.bfloat16)`.
  - `_fused_gdn_decode_unsupported_reason()` requires: head_k_dim = head_v_dim = 128; silu/sigmoid gating; non-interleaved GQA; `model_config.dtype == bfloat16`; BF16 conv state; recurrent state in FUSED_GDN_STATE_DTYPES; capability ≥ 8.0; and the op built.
  - On failure it logs "Falling back to the Triton GDN spec decode path: …". `VLLM_GDN_DECODE_KERNEL=cuda` then raises an error.
  - `_can_use_fused_gdn_mtp_decode()` additionally requires `num_decodes == 0 and num_spec_decodes > 0`, `num_v_heads // num_k_heads in (1, 2, 3, 4, 8)` and `state_indices.size(1) <= 8`. The fused call is taken only when `num_prefills == 0`.
  - Startup log line: `GDN spec decode kernel: %s`.
  - — [Internal: vLLM 0.32.0 qwen_gdn_linear_attn.py](D:/ARC%20AGI%202/_scratch/engine_audit_pack_rev2e/source_refs/vllm_0.32.0/vllm_model_executor_layers_mamba_gdn_qwen_gdn_linear_attn.py)
- **Our logs [V]:**
  - E21 smoke run 1, NB2 (FP8, fp16 SSM): `WARNING … mamba_ssm_dtype='float32' in its config, but --mamba-ssm-cache-dtype='float16' was passed`, then `Falling back to the Triton GDN spec decode path` and `GDN spec decode kernel: triton`.
  - Other 0.32 runs (2026-10-28 20:12 and 2026-10-29 00:03) logged `GDN spec decode kernel: cuda`.
  - — [Internal: E21 run1 nb2 halfB log](D:/ARC%20AGI%202/experiments/Experiment%2021%20-%20Duck%20config%20B%20on%20vLLM%200.19%20+%200.32%20(planned)/Experiment%20Results/SMOKE_GPU_run1_2026-10-29/nb2/q27x/halfB/vllm_server_halfB_primary.log)
- **Obsolete claim:** an HF discussion (Aug 2026, vLLM 0.27.2 nightly) reported the gate as `num_v_heads == 8 * num_k_heads`, making the fused kernel unreachable for Qwen3.8-27B with MTP.
  - Single-request 60-100K context: 19.9 tok/s with MTP on vs 72.3 with MTP off (RTX 5090); MTP cost about 4x per step on the Triton path.
  - This is superseded by the 0.32 source above, which has the ratio list. — [HF Qwen3.8-27B discussion #51](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/51)
  - [I] It still shows how expensive the Triton spec path can be at long context.
- **vLLM #59520 (closed 2026-10-02, fixed by #59735):** the default `VLLM_GDN_DECODE_KERNEL=cuda` wrapper regressed **non-spec** decode by about 30-35 % on H200 (vLLM 0.29.0). The fix keeps non-spec on the standard path **[V, closed]**. — [vLLM #59520](https://github.com/vllm-project/vllm/issues/59520)
- **vLLM PR #62961 (OPEN, opened 2026-10-21, conflicts 2026-10-29):**
  - Folds `in_proj_ba` into `in_proj_qkvz`, removes the gate copy, does an in-place spec gated RMSNorm, adds an early PDL trigger for the fused all-reduce + RMSNorm (FlashInfer ≥ 0.7.2), and shares MRV2 attention metadata across the 6 full-attention groups.
  - Qwen3.8-27B TP4 B300 batch 1: no-spec +5.8 %, MTP3 503.8 → 533.2 tok/s (+5.6 %). Spec gated norm 6.36 → 2.73 µs at 16 rows.
  - Flags: `VLLM_USE_V2_MODEL_RUNNER=1` **[C, author]**. — [vLLM PR #62961](https://github.com/vllm-project/vllm/pull/62961)
- **vLLM PR #63239:** warms up the speculative variant of the Triton GDN recurrent kernel, because pure spec steps never reach it and the first mixed step JIT-compiles it while serving. — [vLLM PR #63239](https://github.com/vllm-project/vllm/pull/63239)
- **vLLM PR #59366 (Draft):** adds the RecoverSSM replay mode (`fused_gdn_decode_post_conv_mtp_replay`, `--use-replayssm`).
  - The verify is 21.3 µs/layer vs 36.9 µs for the native snapshot kernel at c8 on H200.
  - Mixed prefill batches still use the Triton verify. — [vLLM PR #59366](https://github.com/vllm-project/vllm/pull/59366)
- **ROCm/other:**
  - Triton port of the fused MTP kernel (#61855), limited to 16/48 heads, TP1, verify ≤ 4 — [vLLM PR #61855](https://github.com/vllm-project/vllm/pull/61855)
  - Deferred state stores in the Triton MTP kernel (ROCm, #64325); "the CUDA op does not implement this mode" — [vLLM PR #64325](https://github.com/vllm-project/vllm/pull/64325)

### Inferences
- **[I] Keep SSM state BF16 or FP32 on vLLM** if the fused verify kernel is wanted. fp16 trades the fused kernel for memory. The E21 FP8 SSM16 arm measures Triton-GDN speed, not the CUDA path.
- **[I] Mixed prefill + decode steps all fall back to Triton**, and they are frequent in a 16-32-stream agent loop with chunked prefill of 50k prompts. Tighter `--max-num-batched-tokens` or a decode-priority prefill schedule increases the share of pure spec steps. This is a measurable lever with no accuracy risk.
- **[I] #62961's +5.6 % is from B300 at batch 1.** On PCIe L4 TP4, the all-reduce PDL saving (1.4 µs/all-reduce) is negligible next to NCCL PCIe latency. Only the GEMM fold and norm parts transfer.

### Gaps
- No L4 or SM89 measurement of the CUDA vs Triton GDN spec kernel exists anywhere. Our logs show which path ran but not its per-layer time. An nsys trace is needed.
- FLA chunked vs fused-recurrent choice for the GDN prefill path on SM89 (logs say "Triton/FLA GDN prefill kernel") was not benchmarked publicly.

---

## Q5. Attention backends for long-context decode with fp8 KV on Ada (head_dim 256)

### Takeaway
On SM89, vLLM's FlashAttention backend is FA2, and FA2 does **not** accept fp8 KV. So with `--kv-cache-dtype fp8` the choice is **FlashInfer (native/fa2 path)** or **TRITON_ATTN**. Both support head size 256. FlashInfer reports only `UNIFORM_SINGLE_TOKEN_DECODE` CUDA-graph support on non-XQA archs, which forces spec decode to PIECEWISE (our L4 logs confirm this).
- **PR #60532 was merged on 2026-10-22**, after v0.32.0 (released 2026-10-18). It routes uniform multi-token verify queries through FlashInfer's fa2 decode path and reports `UNIFORM_BATCH` on pre-SM90 GPUs. That enables FULL decode graphs with MTP.
- Its author measured 73.6 → 108.1 tok/s (+47 %) for Qwen3.8-27B MTP3 at TP1 and 61.0 → 83.1 (+36 %) at TP2, at batch 1, with identical output tokens.

### Cited Findings
- **vLLM attention backend table (latest docs):**
  - `FLASH_ATTN` FA2: KV dtypes auto/float16/bfloat16 only, CC ≥ 8.0. FA3 (CC 9.x) and FA4 (≥ 10.0) add fp8.
  - `FLASHINFER` Native: KV fp8/fp8_e4m3/fp8_e5m2/nvfp4_4over6; head sizes 64/128/256/512; CC 8.x-9.x; block sizes 16-1024. XQA is CC 9.0 only.
  - `TRITON_ATTN`: KV fp8 variants plus int4/int8/fp8 per-token-head; any head size; any CC.
  - Auto order for SM 8.x-9.x: FLASH_ATTN, FLASHINFER, TRITON_ATTN, FLEX_ATTENTION, TURBOQUANT.
  - FA defaults to FA2 on non-SM90/100. — [vLLM attention backends](https://docs.vllm.ai/en/latest/design/attention_backends/)
- **vLLM CUDA-graph design docs:** FlashInfer = `UNIFORM_SINGLE_TOKEN_DECODE`; Triton attention = `ALWAYS`, with FULL_AND_PIECEWISE preferred because it uses separate prefill and pure-decode kernels — [vLLM CUDA Graphs docs](https://docs.vllm.ai/en/stable/design/cuda_graphs/) (via search summary)
- **vLLM 0.32.0 FlashInfer `get_cudagraph_support()` [V, source read]:** returns `UNIFORM_BATCH` only when every attention spec can use trtllm attention, and checks XQA head-dim/nvfp4 on `is_device_capability(90)` or family 120. Otherwise it returns `UNIFORM_SINGLE_TOKEN_DECODE`. — [Internal: vLLM 0.32.0 flashinfer.py](D:/ARC%20AGI%202/_scratch/engine_audit_pack_rev2e/source_refs/vllm_0.32.0/vllm_v1_attention_backends_flashinfer.py)
- **Our L4 log (vLLM 0.32, MTP):** `CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention backend FlashInferBackend (support: AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE); setting cudagraph_mode=PIECEWISE` on all 4 ranks **[V]**. — [Internal: E21 run1 nb2 halfB log](D:/ARC%20AGI%202/experiments/Experiment%2021%20-%20Duck%20config%20B%20on%20vLLM%200.19%20+%200.32%20(planned)/Experiment%20Results/SMOKE_GPU_run1_2026-10-29/nb2/q27x/halfB/vllm_server_halfB_primary.log)
- **vLLM PR #60532 "Decode FlashInfer spec queries with fa2 on pre-SM90 GPUs" (MERGED 2026-10-22, commit `71e9320`):**
  - Tested on SM80 only, with FlashInfer 0.7.0.post1 and Qwen3.8-27B NVFP4 weights plus NVFP4 KV.
  - Batch-1 MTP3: TP1 73.6 → 108.1 tok/s, TP2 61.0 → 83.1. No-spec unchanged.
  - Output tokens identical in every row.
  - Limitations: DCP and non-causal draft attention stay single-token; no varlen decode on pre-SM90; batch 1 only.
  - fp8 KV and head_dim 256 are not mentioned.
  - New config field `use_fa2_native_decode` (reviewer rename) — [vLLM PR #60532](https://github.com/vllm-project/vllm/pull/60532)
- **vLLM #59548:** the same PIECEWISE warning on L4 with DFlash on 0.29 and 0.30 — [vLLM #59548](https://github.com/vllm-project/vllm/issues/59548)
- **vLLM FP8-KV blog (2026-04-22, H100/B200 only):** fp8 KV at head_dim 256 lowers decode ITL (slope 68 % of BF16 on gemma-4-E2B) but raises prefill TTFT about 1.6x from two-level accumulation in FA3. Qwen3.5-27B reasoning loses ≤ 0.7 points with uncalibrated scales — [vLLM blog: State of FP8 KV-Cache](https://vllm.ai/blog/2026-04-22-fp8-kvcache)
- **FlashInfer PR #6987:** SM89 fp8 KV decode for **MLA only**; SM89's 99 KiB shared memory forced FP8→BF16 tile conversion — [FlashInfer PR #6987](https://github.com/flashinfer-ai/flashinfer/pull/6987)
- **vLLM PR #63642 (Triton attention):** applies per-tensor fp8 KV scales to scores and output for 16-bit queries. Tests are gated to SM89+ and cover head sizes 128/256/512 — [vLLM PR #63642](https://github.com/vllm-project/vllm/pull/63642) (search summary)

### Inferences
- **[I] The single largest documented, merged-but-unreleased lever for our vLLM MTP stack is #60532.** On L4 TP4 at 16-32 streams, the PIECEWISE overhead is a smaller share than at batch 1, so expect well under +47 %. Even a +10-20 % cut in per-step overhead is larger than any draft-tuning gain in Q1. It requires a vLLM nightly after 2026-10-22, or a cherry-pick onto 0.32.0 (Python-only change in `flashinfer.py`).
  - Check that the FlashInfer fa2 decode path accepts **fp8 e4m3 KV with head_dim 256** on SM89. The PR tested NVFP4 KV on SM80.
- **[I] TRITON_ATTN is the alternative.** It already supports FULL graphs (`ALWAYS`) and fp8 KV at head 256 on any GPU. A `--attention-backend TRITON_ATTN` arm is a no-build A/B on 0.32.0 today. Its long-context decode speed vs FlashInfer on Ada is unmeasured.
- **[I] XQA/trtllm-gen do not exist on SM89**, so no FlashInfer FP8-KV fast path beyond fa2 is coming for L4.

### Gaps
- No public L4 or SM89 benchmark of FlashInfer vs Triton attention decode at 50k-130k context with fp8 KV and head 256.
- Whether #60532 interacts with fp8 KV on SM89 is untested.
- SGLang's attention backend choice on SM89 for this model was not re-researched here (cookbook: flashinfer needs > 0.6.15.post1 for MTP; otherwise triton).

---

## Q6. torch.compile, CUDA graph capture sizes and max-num-seqs effects

### Takeaway
On vLLM 0.32 with FlashInfer plus spec decode we run PIECEWISE graphs (Q5). The known capture-size pitfalls are:
- a throughput cliff when `max_num_seqs` is not a multiple of 8 (#57355, fixed in 0.31);
- dynamic-SD batch sizes with no uniform-decode graph, which fall to eager (#56136, open);
- boot-to-boot dispersion on L4 in 0.29, gone in 0.30.

Keep `max_num_seqs` a multiple of 8, make sure every `streams × (1+K)` token count used has a captured size, and benchmark with an eager control arm.

### Cited Findings
- **v0.31.0 fixes "a throughput cliff when `max_num_seqs` is not a multiple of 8 (#57355)".** It also adds `--max-num-active-seqs` (#56758) and reworks the waiting queue so requests holding KV blocks go first (#58947). — [vLLM v0.31.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.31.0) (via the earlier note `newer_engine_versions.md`)
- **v0.30.0:** gc frozen during graph capture cut capture from 12 s to 2 s and engine init from 28.9 s to 8.2 s on H200 (#54646) — [vLLM v0.30.0 release](https://github.com/vllm-project/vllm/releases/tag/v0.30.0) (via the earlier note)
- **vLLM PR #56136 (proposed):** in dynamic SD, some batch sizes in a K tier (3, 6, 9, 11) have no uniform-decode CUDA graph and fall back to eager — [vLLM PR #56136](https://github.com/vllm-project/vllm/pull/56136); context in [vLLM #49548](https://github.com/vllm-project/vllm/issues/49548)
- **vLLM PR #23679:** spec-decode cudagraph refactor with "uniform_alignment of cudagraph sizes", aligning capture sizes to multiples of 1+K — [vLLM PR #23679](https://github.com/vllm-project/vllm/pull/23679) (title only)
- **vLLM #59548 (L4):** with the same PIECEWISE fallback, default (graphs) vs eager throughput was 0.49-0.77x with CV up to 13.92 % on 0.29.0, and about 1.01x with CV 1-1.6 % on 0.30.0.
  - Recommendations: an eager control arm, a fixed prompt set, `/version` check, sampling SM clocks/power/throttle reasons, and measuring only after healthy. — [vLLM #59548](https://github.com/vllm-project/vllm/issues/59548)
- **On L4 graph-mode decode, PIECEWISE ≈ eager (1.01x on 0.30).** At batch 1 on SM80, FULL graphs gave +36-47 % for MTP3 (#60532 above).

### Inferences
- **[I] On 0.30+ with spec decode on L4, our PIECEWISE graphs give essentially no gain over eager** (#59548's 1.01x). The step overhead estimated in Q1 is consistent with launch- and host-bound execution. FULL graphs (via #60532 or TRITON_ATTN) are the matching fix.
- **[I] Capture sizes.** With 16-32 streams and MTP K=3, the decode token counts are 64-128 in steps of 4. Check `cudagraph_capture_sizes` covers every multiple of (1+K) up to `max_num_seqs × (1+K)`. A missing size runs eager.

### Gaps
- No public measurement of `max_num_seqs` 16 vs 20 vs 24 vs 32 on L4 for a hybrid 27B.
- `torch.compile` mode effects (`STOCK_TORCH_COMPILE`, #59079 in 0.32) on L4 were not found.
