# DFlash / DFlash2 for Qwen3.8-Flash-Next on Merlin v3.0.0 (4×L4): feasibility check

## Verdict
**Yes, this is feasible on Merlin, and it should take days, not weeks.**

- Merlin already ships the V2-runner DFlash and DFlash2 speculators, the draft model classes, PP relay of auxiliary hidden states, and hybrid/GDN support for drafters.
- What is missing is on the target side: the allowlist entry and the aux-hidden-state seam for Qwen4Exp. A working seam patch for vLLM 0.32.0 already exists publicly (roozich "patch 23", about 120 lines). Its text anchors match Merlin's files line for line. I checked that with grep only.
- On top of that patch we need three things of our own:
  - **PP2 relay of the taps.** About 25 lines, using DeepSeek-V4's HC + PP code in Merlin as the template.
  - **An embedding copy for the drafter on PP rank 1.** Without it the drafter silently produces garbage.
  - **Optionally, the QSA ring widening.** Only needed for K ≥ 5.
- No kernel work is needed.
- Expected gain at K = 3 is about +9 % (range −8 % to +28 %, my estimate). The larger upside is the drafter's trained block size of 8 (K = 7), which Merlin can try and SGLang cannot. That arm is unmeasured.
- The earlier estimate in `STRUCTURAL_TRACK_FEASIBILITY_2026-10-31.md` §4/§6 S3 ("1-2 weeks") is out of date.

**Corrections to the facts you gave me:**
- Upstream vLLM 0.32.0 **does not** serve Flash-Next with dflash out of the box.
- Humming low-bit is **not** Merlin-only.
- The "albucino DFlash drafter 4.2 GB" in the memory notes is actually albucino's INT4 MTP/NEXTN drafter.

Details for all three are under Q2 and Q3.

---

## Q1. What the Merlin source contains

Paths below are under `D:/ARC AGI 2/_scratch/e23/p2/merlin-src/vllm/` (Merlin 3.0.0, commit ad42e37). Every item was read in source.

### DFlash code that already exists
- **Draft models.** `model_executor/models/qwen3_dflash.py` (966 lines) and `qwen3_dflash2.py` (416 lines: DFlashGroupedConv, CandidateSelector, compute_candidates).
- **Registry.** `registry.py:643-645` maps `DFlashDraftModel → qwen3_dflash` and `DFlash2DraftModel → qwen3_dflash2`.
- **Config.** `config/speculative.py`:
  - `DFlashModelTypes = Literal["dflash"]` at `:70`;
  - method auto-detection at `:1357-1373`;
  - `parallel_drafting = True` at `:1498-1499`;
  - the DFlash slot rule (K extra slots) at `:1960-1975`;
  - `use_dflash()` at `:2017`.
- **V2 runner speculators.** `v1/worker/gpu/spec_decode/__init__.py` dispatches `method == "dflash"`: architecture DFlash2DraftModel goes to `dflash2/speculator.py` (265 lines), and anything else goes to `dflash/speculator.py` (815 lines, plus `cudagraph.py` and `utils.py`).
- **Engine rules.** `config/vllm.py:661` (DFlash lookahead K+1), `:771-795` (`_is_dflash_candidate_draft`, multi-KV-group check), `:1604` (async scheduling allowed for dflash).
- **Hybrid state.** `v1/worker/gpu/model_states/mamba_hybrid.py:158-166` already handles "a drafter group with its own smaller block (e.g. DFlash)".

### Where the allowlist is enforced
`model_executor/models/config.py:1049-1058`, inside `Qwen4ExpForConditionalGenerationConfig.verify_and_update_config`. Any method outside `{"mtp", "ngram", "ngram_gpu"}` raises `NotImplementedError("Qwen4Exp speculative decoding supports only its native MTP checkpoint and linear n-gram proposers")`.

### What `method: dflash` needs
It is not an allowlist-only change, and it is not a kernel port. It is a target-side seam plus PP and embedding fixes.

1. **Allowlist.** Add `"dflash"` at `config.py:1049`.
2. **Aux-hidden-state seam on Qwen4Exp.**
   - Today `Qwen4ExpModel(nn.Module)` (`models/qwen4_exp/nvidia/model.py:567`) has no EagleModelMixin, and `Qwen4ExpForCausalLM` (`:866-876`) has no SupportsEagle3.
   - So `set_eagle3_aux_hidden_state_layers` raises "Model does not support EAGLE3 interface" (`v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:15-20`). It is called from `v1/worker/gpu/model_runner.py:524-526`, because `use_aux_hidden_state_outputs = True` for dflash (`:386-393`).
   - roozich's patch adds exactly this seam:
     - the EagleModelMixin;
     - the protocol members, mounted at the module tail;
     - a capture at each completed tap layer through `layer.mlp_hyper_connection.combine(...)`, averaged over the hc_count = 4 streams;
     - a `(hidden, [taps])` return.
   - That capture recipe matches incoai's own `patches/qwen38-dflash2-capture.patch` ("completed, zero-based layer outputs, averaged over the four hyper-connection streams").
   - Anchor check, grep only: every anchor exists exactly once in Merlin's `model.py` (`:42-43`, `:55`, `:567`, `:723-724`, `:752-754`, `:819-820`, and the `__all__` tail) and in `common/qsa_cache.py` (`:24`, `:77`, `:889`, `:955-965`).
   - I could not dry-run the patch script: the auto-mode classifier blocked running downloaded code. It is saved, unexecuted, at `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/dflash/dl_roozich/patch_dflash_qwen4exp.py`.
3. **PP2 relay (our addition; the roozich patch is single GPU).**
   - Under PP, `verify_supports_aux_hidden_states_over_pp` raises "`<model>` does not support dflash with pipeline parallelism" unless the decoder sets `supports_aux_hidden_states_over_pp` (`eagle3_utils.py:44-50`; called at `model_runner.py:528-533`).
   - Qwen4Exp does not set it. Only deepseek_v4, kimi_k3, minimax_m3, llama, qwen2 and sarvam do.
   - Template: `models/deepseek_v4/nvidia/model.py:1288` and `:1466-1527`. That code is an HC model too: `collect_remote_aux_hidden_states` at entry, `pack_local_aux_hidden_states` in the non-last-rank `IntermediateTensors`, and `remote_aux + aux` at the end. The helpers are in `interfaces.py:1571-1640` (EagleModelMixin).
   - The relay plumbing (`pp_handler.configure_aux_hidden_state_relay`) already exists.
   - Tap placement: the incoai taps are [5, 15, 25, 35, 45], which become layer ids [6, 16, 26, 36, 46] after the helper's +1. With the profile partition [25, 23], taps 5 and 15 sit on PP0 and must ride the PP hand-off.
   - If HC sequence parallelism is on, each tap needs an `sp_all_gather`, as deepseek_v4 does at `:1488-1489`.
4. **Drafter embedding under PP: a silent-garbage risk.**
   - The incoai checkpoint has no `embed_tokens` and no `lm_head`. I measured this from the safetensors header: 603,065,600 BF16 parameters (layers, fc 2560×12800, norms, candidate_selector codebooks 2 × 248320×256).
   - Under PP > 1, `maybe_share_target_embed` returns early (`v1/worker/gpu/spec_decode/eagle/utils.py:56-59`). The target's embedding is a PPMissingLayer on rank 1 (`nvidia/model.py:600-610`).
   - The draft's own `VocabParallelEmbedding` (`qwen3_dflash.py:461`) is then mapped away at load (`:918-919`) and stays uninitialized. The likely result is near-zero acceptance with no error.
   - Fix options:
     - append `embed_tokens.weight`, copied from the target (1.27 GB BF16), to the drafter safetensors;
     - or a 5-line loader patch.
   - The lm_head is shared correctly on the last rank (`dflash/utils.py`, `get_target_lm_head`).
5. **QSA ring, only for K ≥ 5.**
   - `common/qsa_cache.py:960-965` asserts that `block_size % (4·ceil((4+K)/4)) == 0`, where `indexer_compress_ratio` = 4 (from config).
   - K = 3 and K = 4 give capacity 8, which divides 4096: OK.
   - K = 5-7 give 12, which does not divide 4096, so boot asserts.
   - roozich part (c) ports vLLM PR #54912's widening (12 → 16) verbatim. This is the "QSA limit" behind incoai's block 4 in SGLang; in Merlin it is a 40-line patch.
6. **Already OK:**
   - draft attention: FA2 on sm_89 supports non-causal (`flash_attn.py:396`) and UNIFORM_BATCH cudagraphs (`:614-618`);
   - `sample_from_anchor` is absent from the incoai config, so the raise at `dflash/speculator.py:80-85` is not reached;
   - the None MTP-hidden getter is already handled (`v1/worker/gpu/spec_decode/utils.py:24-39`).

### Compatibility with V2 runner + PP2 + FULL graphs + RecoverSSM/GDN
- **V2 runner:** yes. The DFlash and DFlash2 speculators are V2 classes.
- **PP2:** yes, after item 3. The speculator lives on the last PP rank only (`model_runner.py:381`), exactly like MTP today.
- **FULL graphs:**
  - The draft decode gets FULL_DECODE_ONLY when the draft attention backend is UNIFORM_BATCH (`dflash/speculator.py:124-148`).
  - PIECEWISE is not supported for dflash (`:139`), so draft prefill runs eagerly. MTP in RC2 had draft prefill FULL_AND_PIECEWISE, so there is a small prefill cost.
  - The target keeps FULL_AND_PIECEWISE.
- **RecoverSSM:** method-agnostic. `use_recoverssm = num_speculative_tokens > 0 and arch in Qwen4Exp archs` (`config/vllm.py:3355-3375`). The GDN verify path checks target tokens regardless of the drafter (`mamba_hybrid.py:310-360`).
- **Adaptive verification:** a separate optional flag, default False (`config/speculative.py:553`). It is rejected under PP for every method (`config/vllm.py:3055-3063`). DFlash does not need it, so it is not a blocker.
- **Remaining unknowns:**
  - how the drafter's sliding-window KV group is sized in the hybrid allocator (page unification with 4096-token / 15.77 MiB blocks);
  - prefix-cache hits for draft KV in multi-turn Duck;
  - whether async scheduling holds with the PP relay.

## Q2. Drafter weights

### For Flash-Next
- **`incoai/Qwen3.8-Flash-Next-DFlash2`** (https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) is the only usable one.
  - Files (HF API, measured): `model.safetensors` 1,206,140,008 B, `config.json`, `patches/qwen38-dflash2-capture.patch`.
  - Config:
    - `DFlash2DraftModel`, `model_type` qwen3, 5 layers, hidden 2560, intermediate 7680;
    - 32 heads / 8 KV heads, head_dim 128;
    - sliding window 2048 on all layers, `is_causal` false;
    - `block_size` **8**, `target_layer_ids` [5, 15, 25, 35, 45] of 48;
    - `mask_token_id` 248077, `selector_top_k` 16, `selector_rank` 256;
    - `conv_group_size` 16, `conv_kernel_size` 2;
    - vocab 248320.
  - Served at block 4 only in SGLang (commit 3c560089 + capture patch, TP1, GB300). The card says that is "limited by the current QSA implementation".
  - Apache-2.0.
- **`PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash`** (https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash) is DeepSpec with `sample_from_anchor`. Merlin rejects that (`dflash/speculator.py:80-85`), and it needs a `query_zero_predicts_next` path. Its served AL was 2.883, worse than native MTP k4's 3.367 (CLAIMED). Not worth porting.
- **`ggml-org/Qwen3.8-Flash-Next-GGUF`** has DFlash2 sidecars (Q4_0 / Q8_0 / BF16 1.22 GB). These are GGUF only.
- **z-lab:** none found for Flash-Next (search on 2026-11-01). It only publishes DFlash for other models, such as Qwen3.6-35B-A3B at block 8/16.
- **Our Kaggle models:** no Flash-Next DFlash drafter. The memory note's "albucino DFlash drafter (4.2 GB)" contradicts the E13 listings, which call albucino's drafter an INT4 MTP/NEXTN head (`mtp-routed-experts-int4`, `mtp-dense`). The memory note should be corrected.
- **Work needed:** upload incoai's 1.2 GB drafter, plus the 1.27 GB embedding copy, as a Kaggle Model.

### Reusing the 27B DFlash2: no
- Kaggle handle: `foysalemonshanto/qwen3-8-27b-dflash2/pytorch/bf16/1`. The config is mirrored at `_scratch/session_1c6db7aa/sgl_src/dflash2_config.json`.
- Comparison:

  | | 27B DFlash2 | Flash-Next |
  |---|---|---|
  | Hidden size | 5120 | 2560 |
  | Target layers | 64, taps [5, 19, 33, 47, 61] | 48 |
  | Mask token | 248070 | 248077 (incoai drafter) |
  | Vocab | 248320 | 248320 |

  Flash-Next numbers come from the local Merlin checkpoint `config.json` (hidden_size 2560, num_hidden_layers 48, vocab_size 248320, hc_count 4).
- Only the vocab matches. The fc input would be 25600 against an expected 12800, which `qwen3_dflash.py:873-880` rejects. Tap 61 also does not exist in Flash-Next.

## Q3. Upstream vLLM 0.32.0 without Merlin: what we would lose

**Upstream gives no DFlash shortcut.**
- Upstream `Qwen4ExpForCausalLM` (`_scratch/vllm032_qwen4exp_src/vllm/models/qwen4_exp/nvidia/model.py:693-701`) also lacks SupportsEagle3.
- The allowlist text is the same: roozich says it matches 0.32.0 byte for byte, and vLLM issue #56088 is still open with no linked PRs.
- So upstream needs the same patch 23, and the same PP relay and embedding fixes, on top.
- roozich validated patch 23 only on a single GB10 (PP1/TP1). Reported numbers: +26 % single-request decode, 57.0 vs 45.2 tok/s against MTP = 2 (CLAIMED, no acceptance figures).

**What we would lose by leaving Merlin:**
1. **RecoverSSM/ReplaySSM.** Upstream Qwen4Exp does not declare SupportsReplaySSM (upstream `model.py:693-701` vs Merlin `:866-876`). So Merlin's GDN RecoverSSM verify/commit path (`config/vllm.py:3355-3375`) is gone, along with RC2's measured F = 6 fixed-block behaviour.
2. **FP8 PLE format.** Merlin has `nvidia/ple_format_guard.py:25` (Qwen4ExpPLEFp8EmbeddingMethod, float8_e4m3fn) and `ple_pull.py`, with a 47.7 GiB pinned FP8 table. Our partial upstream copy shows no FP8 PLE path, so upstream would mean a BF16 PLE of about 95-102 GB in host RAM. RC2 already sat at the 177 GiB cgroup edge.
3. **The `fn-quad-24g-sm89` profile** (`tools/merlin/profiles/flash-next.json`, profiles[7]): TP2×PP2 + EP with the 25/23 partition, fp8 KV pin of 4.1e9, capture sizes 1-32, `max_num_seqs` 8, `max_num_batched_tokens` 1024, P2P-off NCCL exports, `disable_custom_all_reduce`. Upstream would need all of this hand-built.
4. **The Merlin checkpoint and its load hooks.** `halt95/Qwen3.8-Flash-Next-W4A16-Merlin` (compressed-tensors W4A16, INT8 GDN, FP8 PLE) is mirrored on Kaggle as `foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p1..p5`. On upstream we would fall back to Intel AutoRound + BF16 PLE (`dfranzen` mirror).
5. **Merlin MTP extras**: `draft_lm_head_quant` int4/int8 (`config/speculative.py:610`), the draft-vocab file, `MERLIN_SHARE_DRAFT_WORKSPACE`. These are moot once DFlash replaces MTP.
6. **RC2 validation:** BF16 and FP8 both clean, 120k PASS. Upstream on L4 is unvalidated, and SGLang 0.5.21 previously produced "!!!!" garbage.

**Not lost:**
- **Humming low-bit** is already in upstream 0.32.0 (`_scratch/vllm032_qwen4exp_src/.../inc_wna16_scheme.py:26-28, 141-142`, `CUDA_HUMMING_SUPPORTED_BITS = {2,3,5,6,7}`).
- **#60490** (QSA indexer prefill workspace) is missing from Merlin per PLAN_1500 `:223` / D9. Merlin is a 0.32 rebase carrying later PRs, so upstream 0.32.0 lacks it too. Either way it has to be backported.

## Q4. Expected gain (estimates unless marked)

**Inputs:**
- Our Duck MTP K3: AL 2.83 = 1 + 0.795 + 0.599 + 0.439 (your figures; I could not find the source file in the workspace).
- RC2 natural row: 0.772 / 0.558 / 0.390, MAL 2.72 (measured, `E13_RC2_GPU_v1_2026-10-31.md` item 17).
- incoai published acceptance length, MTP → DFlash2, both drafting 3 tokens, xhigh, T 1.0, 4k cap, c1 (CLAIMED):

  | Task | MTP | DFlash2 | Ratio |
  |---|---|---|---|
  | GSM8K | 3.08 | 3.26 | ×1.058 |
  | MATH-500 | 2.93 | 3.17 | ×1.082 |
  | HumanEval | 2.52 | 2.94 | ×1.167 |
  | MBPP | 2.65 | 3.06 | ×1.155 |
  | MT-Bench | 2.50 | 2.83 | ×1.132 |

  The mean ratio is ×1.12. There is no long-context data.

**Draft cost on PP stage 1** (it already binds at 33 ms):
- **DFlash2, one pass per step.** Per card on PP1 (TP2):
  - 5 layers ≈ 0.43 GB;
  - fc 0.065 GB;
  - selector codebooks ≈ 0.25 GB, if replicated;
  - one lm_head candidate pass ≈ 0.64 GB.
  - Total ≈ 1.4 GB → 5.6-8 ms at 250-173 GB/s, plus about 10 TP all-reduces over SHM (≈ 2-4 ms). About 8-12 ms.
- **MTP K3, three sequential passes.** Each pass is one MoE layer plus lm_head, so about 6-10 ms or more depending on Merlin's draft lm_head quantization.
- **Net tick change:** about −3 to +5 ms.
- **PP hand-off:** adds two 2560-wide taps per token. That is trivial in decode, but about 84 MB more per 8k prefill chunk (about 8-28 ms per chunk over SHM).
- **Memory:** about neutral on PP1. The MTP weights are freed (≈ 0.65 GB of experts plus 0.64 GB of embedding per card), and the drafter plus its embedding take ≈ 1.3-1.4 GB. The draft KV is about 21 MB per request per card in BF16 (2048 window), but allocator rounding could cost 1-2 extra 15.77 MiB blocks per request on top of F = 6. That is unverified and must be read from the boot's "Maximum concurrency" line.

**Gains:**

| Arm | AL | Net throughput vs MTP K3 |
|---|---|---|
| K = 3 (block 4, as published) | 2.83 → 2.99-3.30 (central 3.17) | central about +9 %; range −8 % (3.0 AL, +5 ms) to +28 % (3.3 AL, −3 ms) |
| K = 7 (block 8, as trained; needs the ring patch) | must reach ≈ 3.8-3.9 to break even | break-even to about +15 %; unmeasured |

- **Why K = 7 is the arm to test:**
  - Verify tokens per request go 4 → 8, so routed-expert reads per micro-batch rise by roughly +50-65 % at C ≈ 9 (D(T) saturation). That adds roughly 6-12 ms on both stages. Break-even then needs AL ≥ 2.83 × 45/33 ≈ 3.9.
  - Our 27B DFlash2 at block 8 measured AL 4.43 → 3.77 over context on the ARC workload (E11-03, `Experiment 11/.../e11_03/decode_rows_C_int4_dflash2_long.json`).
  - K = 4 needs no ring patch (capacity 8) and may be the sweet spot.
- **Ceiling:** this lever cannot change the PLAN's capacity bind. It is a multiplier on a ≈ 2.7-2.8.

## Q5. Verdict and next step

**Feasible: about 1 day of build + CPU kit, then one short GPU notebook (about 1.5-2 h, fail-fast gates, no separate smoke). About 2-4 days to a measured go/no-go.**

### Exact next step
1. **Build an "E25 DFlash2 seam" patch set for Merlin.** These are new files in an E25 Build folder; nothing is edited in `merlin-src`.
   - (a) roozich parts a/b/c, adapted;
   - (b) the PP relay: `supports_aux_hidden_states_over_pp = True`, `collect_remote`/`pack_local` in `Qwen4ExpModel.forward` following `deepseek_v4/nvidia/model.py:1466-1527`, plus `sp_all_gather` if HC-SP is on;
   - (c) a drafter repack: incoai `model.safetensors` + `embed_tokens.weight` copied from the Merlin checkpoint, uploaded as a Kaggle Model.
2. **Run a Kaggle CPU kit** on the RC2 Merlin venv:
   - patch-apply and AST parse;
   - meta-device config resolution with `method=dflash`: resolves to DFlash2DraftModel, the fc input is 12800 = 5×2560, and the aux layers are (6, 16, 26, 36, 46) with 2 on PP0;
   - assert the relay keys;
   - assert the draft embedding is loaded, not zero/uninitialized;
   - the QSA ring capacity for K = 3/4/7.
3. **Run one GPU notebook** (needs your go):
   - Teacher-forced per-position acceptance on recorded RC2 xhigh traces, comparing MTP K3 against DFlash2 K = 3 / 4 / 7.
   - Then a served ladder at 40k with 4 and 8 streams and at 90k with 3 streams, per arm, in the same boot family.
   - Gates: real running == level, 0 garbage, and the "Maximum concurrency" / F line read.
   - Adopt DFlash2 only if steady tok/s beats MTP K3 at equal resident streams.

Sources:
- [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (plus its config.json, HF tree API and capture patch)
- [PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash](https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash)
- [vllm-project/vllm#56088](https://github.com/vllm-project/vllm/issues/56088)
- [roozich/Qwen3.8-Flash-Spark](https://github.com/roozich/Qwen3.8-Flash-Spark) (CHANGELOG 2026-10-28; `src/patch_dflash_qwen4exp.py`)
- [z-lab/Qwen3.6-35B-A3B-DFlash](https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash/blob/main/README.md)
- [ggml-org/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/ggml-org/Qwen3.8-Flash-Next-GGUF)

Files in `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/dflash/` (nothing was edited in the project, nothing pushed, no git):
- `dl_roozich/patch_dflash_qwen4exp.py` — downloaded, not run
- `dl_roozich/test_dflash_patch_cpu.py` — downloaded, not run
- `capture.patch` — incoai's SGLang capture patch
- `fn_dflash2_head.bin` — the first 200 KB of the drafter safetensors, used to read the tensor header
