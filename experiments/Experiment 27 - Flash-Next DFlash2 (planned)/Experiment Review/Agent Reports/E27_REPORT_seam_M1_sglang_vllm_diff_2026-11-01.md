# E27 seam agent (aefbfc6004f4dc34f): M1 SGLang / vLLM DFlash2 fix diff, verbatim

M1 is done. I ported one change, an env-gated A/B for the DFlash2 grouped convolution, as an 8th seam file. Everything else was either already in Merlin ad42e37 or does not apply to our path. The seam changed, so NBDF must re-embed it. Tests: 591/591 pass (was 555).

NEW dflash_seam.py: sha256 7356d4738962bdbab4091264861c4bfbbdcd7098d6c75b9c48db69eeb73e6937 (1760 lines, LF, ASCII).
dflash_seam_tests.py: sha256 ae856fef6a0eb4e0a969a5f4cfb427d6b8a823d95dd34dd638e68428f869f22b (2000 lines).

New file in PATCHES (in CORE_RELS): vllm/model_executor/models/qwen3_dflash2.py
- pre 11ac6ba8c6db702375970e68cd042b98783a5c43b87f9bfa27e88463316bf499
- post 4039da670a5f6a91ef5e5931465a652d4145207c90b87ff44fdb7b1f61d04402

CORRECTION to my first report. Two of the post shas I gave there were stale: I captured them from the test run just before the final fix round (the inline E27 comments on the model.py import lines, and released_bytes in dflash/utils.py). Since the 520-check version, the correct values are:
- nvidia/model.py post = 787161118bb49790fb5b8dd99158e029eed3e855092dfb081e3173c1b74e2f1c (not fee65643...)
- dflash/utils.py post = fb76c074fdc5ecc94148a7efdeddaec426e4aaf286e812d6e715237a2dd7b5a3 (not 5b3cd014...)
The dflash_seam.py shas I reported each time were correct. The module checks its own constants, so apply / verify were never affected. Full current table (pre / post):
- config.py: 1a16138305f395f10e87b1ec9de40d501eba2a3c0f1379a6c787bb5c2d2e61ac / 0f6f63d355a269743cf8b6c0c2ef2b43aad066f5c1d202e534b02d3f44ff2711
- nvidia/model.py: 068adee6c0e062342765d10e1e22da5967c428059197067138064b137f973397 / 787161118bb49790fb5b8dd99158e029eed3e855092dfb081e3173c1b74e2f1c
- qwen3_dflash.py: 5375543288e72219abc1fc777219a640f0d173b3e8ac531718d666483a46ea7a / 774201928124445e72052ead3df482ce95a414e5467b3a8db3caca838fd14e24
- dflash/utils.py: f0584417149c0a6fa67e700b2a725e40a41f4546a806b26e6cc64b64c59cfd66 / fb76c074fdc5ecc94148a7efdeddaec426e4aaf286e812d6e715237a2dd7b5a3
- qsa_cache.py: 43f323bb347e2152e47acb36a99d0682eaebc5cdafe9b5ba168b5c4b4e9acf13 / 07b0b61f9947d36ab2b3f87958830aaf591e899a93ab61f67c4debc50c442bde
- attention.py: bde77677fd0e95ea09c6a580dfaf09b57741e8183e00cb106e2d1cc64d8e03f2 / 5817d764b870409ef0ef5b0c9e619f532a6410cece82833bd464dcea69181a5c
- dflash/speculator.py: b731b09c92b93063d89851a997e1de70e51d512bc40b2a4dd68274a3a999e187 / 85f57f0ee9bc7018f052f59d0610a91c78d35cd018ef711c6f7ecbef298d47ea
- qwen3_dflash2.py: 11ac6ba8c6db702375970e68cd042b98783a5c43b87f9bfa27e88463316bf499 / 4039da670a5f6a91ef5e5931465a652d4145207c90b87ff44fdb7b1f61d04402

Sources, all read-only: the SGLang releases v0.5.10..v0.5.22; gh pr diff for SGLang 41736 / 41737 / 37818 / 42002 / 45142 / 47311 / 38351 / 36261 / 36683; gh search on vllm-project/vllm "dflash2" (40 PRs); the diffs or bodies of vLLM 55960 / 58250 / 53978 / 60252 / 61537 / 53628 / 56552 / 53662 / 60365 / 55294 / 52883 / 55601. Saved in D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/m1/

FINDINGS, item by item:

A. Grouped convolution (GPT area 1).
- SGLang #41736 (merged, v0.5.22): keep grouped-conv taps inside each request's block, so NaN/Inf cannot leak across requests (torch.where instead of multiplying by the mask).
- Merlin: the CUDA path is the fused Triton kernel from vLLM #55960 (qwen3_dflash2.py:25-141, run on every CUDA call at :153-154). The kernel loads tap rows under tap_mask (:57-70), so it does not leak; same effect as #41736. Merlin's torch fallback (:156-167) still multiplies by the mask, so it leaks NaN, but it runs on CPU only.
- vLLM #58250 (open, stale): reverts #55960 because the DFlash acceptance-length nightly fell 3.195 -> 3.060 after the merge (accuracy unchanged). Merlin carries #55960 unchanged.
- PORTED as an A/B arm, not a default change:
  - with DFLASH_SEAM=1, DFLASH_CONV_IMPL=torch routes the conv through the pre-#55960 torch path on CUDA too, with the #41736 torch.where masking;
  - default (fused) and env unset: Merlin's code runs unchanged;
  - read once at import; one 'conv_impl' log line per drafter worker;
  - server_env(conv_impl=...); readback_label(conv_impl="fused" by default; it checks conv_impl lines when they are present, and requires them for torch).
- Recommend a teacher-forced acceptance A/B (fused vs torch) in the first GPU notebook.

B. Candidate selector NaN (GPT area 2).
- SGLang #41737 (merged, v0.5.22): their greedy walk picked index = min(where(scores == max)). An all-NaN row gave index top_k, out of range.
- Merlin does not have that code. _selector_walk_kernel (dflash2/speculator.py:15-75) picks through gumbel_noised_argmax -> tl.max(..., return_indices=True) (sample/gumbel.py:244-302), which always returns a lane index in [0, BLOCK_K). BLOCK_K = next_pow2(selector_top_k=16) = 16 = top_k (:142), so there are no masked lanes and the index is always valid.
- Not applicable, no port. Residual note: a NaN row would still give a garbage, but in-bounds, draft, which rejection removes.
- vLLM #53978 (open): clamp the selector codebook gathers against -1 / uninitialised warmup ids. Merlin's anchor ids come from input_ids, which are zero-initialised (input_batch.py:26), and the candidates come from a top-k over a 248320-row head with no padding at TP2. Not applicable.

C. Hybrid recurrent state with speculation (GPT area 3).
- vLLM #60174 (all-NaN logits) is fixed by #55601 (merged 2026-10-07): seed the hybrid mamba state column with mamba_block_size, because a DFlash drafter group lowers cache_config.block_size. Merlin ALREADY HAS IT (mamba_hybrid.py:158-166).
- SGLang #37818 (mamba checkpoint tracking post-verify), #36261 (DFlash2 PP) and #36683 (ReplaySSM for DFlash) are SGLang-engine code with no Merlin source equivalent. Merlin's RecoverSSM / V2 verify are method-agnostic (the same path MTP uses). Not ported.
- NEW CAUTION (documented in the seam): our DFLASH_DRAFT_SW_BLOCK=<n> option makes the engine set cache_config.block_size = min over the prefix-cacheable groups = n (v1/engine/core.py:436-448). The mamba seeding is safe (#55601). Every other reader of a lowered cache_config.block_size is untested, so the kit must gate that option before any GPU arm. The default is off.

D. Other DFlash2 fixes checked:
- SGLang #45142 (open), qwen4_exp capture for Flash-Next. It captures layer k's completed streams with hc_contract (the stream mean) and states incoai's drafter is trained on the mean. This confirms our default tap (combine). Their "no +1" is the same as our ids = target_layer_ids + 1 in vLLM semantics. Nothing to port.
- SGLang #47311 (muP logit multiplier): the incoai config has no output_multiplier / input_embedding_scale / final_logit_softcapping, and Merlin applies them when present (qwen3_dflash2.py:398-404, DFlash2Qwen3Model.input_embedding_scale). Not applicable.
- vLLM #61537 (bidirectional full-attention layers): the incoai layers are all sliding_attention with is_causal false. Not applicable.
- vLLM #60252 (SW draft block divisor of the primary block): FA2 on sm_89 runs the primary 4096 block unsplit (attention.py:619-666), and our DFLASH_DRAFT_SW_BLOCK requires a divisor. Not applicable.
- vLLM #53628: ROCm prefix-attention NaN, ROCm only.
- vLLM #56552 (SM120 PP2 relay, last-block drop scoping): our seam has its own PP2 relay. Merlin's eagle-group annotation already excludes DFlash (kv_cache_utils.py:2063-2119). Not applicable.
- vLLM #53662 (no eager allocation of shared embed / lm_head): memory only. Covered by our draft lm_head / embed release (B5).
- vLLM #60365 (reduced vocabulary): draft vocab = target vocab 248320. Not applicable.
- vLLM #55294 (FP16 fast-fail): our drafter is BF16. Not applicable.
- vLLM #52883 (unquantized linear lm_head in the selector): our heads are ParallelLMHead BF16. Not applicable.
- vLLM #55959 / #64916: performance only, not ported.
- SGLang #42002 / #38351 and the NPU / AMD / ROCm items: engine- or platform-specific.

New tests:
- T12 (21 checks):
  - AST: the flag branch comes first and the rest of _grouped_conv is unchanged; the fused kernel loads under tap_mask;
  - the real gating function: env off, the fused default, torch, and an unknown value;
  - the real E27 torch path is bitwise equal to Merlin's torch fallback on finite input at Flash-Next shapes (160 groups x 16, taps 2, blocks 4 / 5 / 8), and within bf16 tolerance of the #55960 fp32 reference;
  - the flag routes around the fused op;
  - NaN / Inf in request 0: Merlin's fallback leaks it into request 1 (negative control), the E27 path keeps requests 1-2 finite and equal to a run without the NaN rows.
- T9: +4 conv_impl readback / server_env checks.

Counts by section: T1 85, T2 40, T3 90, T4 17, T5 209, T6 29, T7 26, T8 12, T9 24, T10 19, T11 17, T12 21 = 591, all pass. The Merlin source copy is still clean.
