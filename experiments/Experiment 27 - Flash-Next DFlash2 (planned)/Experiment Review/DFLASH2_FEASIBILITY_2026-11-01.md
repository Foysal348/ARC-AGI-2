# DFlash2 for Qwen3.8-Flash-Next on Merlin v3.0.0 (4×L4): feasibility check (agent report, 2026-11-01)

**Verdict:** feasible on Merlin.
- Merlin already ships the V2-runner DFlash / DFlash2 speculators, the draft model classes, the PP relay of auxiliary hidden states, and hybrid / GDN drafter support.
- Missing: the Qwen4Exp allowlist entry and the target-side aux-hidden-state seam. A public seam patch for vLLM 0.32.0 exists (roozich "patch 23", about 120 lines), and its anchors match Merlin's files (checked with grep only; the downloaded patch was NOT executed).
- Ours to add: the PP2 relay of the taps (about 25 lines, using Merlin's DeepSeek-V4 HC+PP code as the template), the drafter embedding on PP rank 1 (otherwise the drafts are silently garbage), and the QSA ring widening for K ≥ 5.
- No kernel work.

## Q1. Merlin source (`_scratch/e23/p2/merlin-src/vllm/`, ad42e37)
**What already exists:**
- **Draft models:** `model_executor/models/qwen3_dflash.py` (966 lines) and `qwen3_dflash2.py` (416 lines).
- **Registry:** `registry.py:643-645`.
- **Config:** `config/speculative.py:70` (DFlashModelTypes), `:1357-1373` (method auto-detection), `:1498-1499` (parallel_drafting), `:1960-1975` (slot rule), `:2017` (`use_dflash()`).
- **V2 speculators:** `v1/worker/gpu/spec_decode/__init__.py` dispatches `dflash` to `dflash2/speculator.py` (265 lines) when the architecture is DFlash2DraftModel, else to `dflash/speculator.py` (815 lines).
- **Engine rules:** `config/vllm.py:661` (lookahead K+1), `:771-795` (`_is_dflash_candidate_draft`), `:1604` (async scheduling allowed).
- **Hybrid state:** `v1/worker/gpu/model_states/mamba_hybrid.py:158-166` handles a drafter group with its own smaller block.

**Allowlist:** `model_executor/models/config.py:1049-1058` (`Qwen4ExpForConditionalGenerationConfig.verify_and_update_config`) accepts only {mtp, ngram, ngram_gpu}.

**Needed:**
1. **Allowlist:** add `"dflash"`.
2. **Seam:** `Qwen4ExpModel` (`models/qwen4_exp/nvidia/model.py:567`) needs EagleModelMixin, and `Qwen4ExpForCausalLM` (`:866-876`) needs SupportsEagle3.
   - Otherwise `set_eagle3_aux_hidden_state_layers` raises (`v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:15-20`). It is called at `model_runner.py:524-526`, because `use_aux_hidden_state_outputs` is True for dflash (`:386-393`).
   - Capture at each tap layer: `layer.mlp_hyper_connection.combine(...)`, averaged over hc_count = 4 streams. This matches incoai's `qwen38-dflash2-capture.patch`.
3. **PP2 relay:**
   - `verify_supports_aux_hidden_states_over_pp` (`eagle3_utils.py:44-50`, called at `model_runner.py:528-533`) needs `supports_aux_hidden_states_over_pp`.
   - Template: `models/deepseek_v4/nvidia/model.py:1288`, `:1466-1527`, with the helpers in `interfaces.py:1571-1640`.
   - Taps [5, 15, 25, 35, 45] become layer ids [6, 16, 26, 36, 46]. With the 25/23 partition, 2 taps are on PP0 and must ride the PP hand-off (plus `sp_all_gather` if HC-SP is on).
4. **Drafter embedding under PP (silent-garbage risk):**
   - The incoai checkpoint has no `embed_tokens` / `lm_head`.
   - Under PP > 1, `maybe_share_target_embed` returns early (`v1/worker/gpu/spec_decode/eagle/utils.py:56-59`), and the target embed is a PPMissingLayer on rank 1 (`nvidia/model.py:600-610`). So the draft `VocabParallelEmbedding` (`qwen3_dflash.py:461`, mapped away at `:918-919`) stays uninitialised.
   - Fix: append `embed_tokens.weight` (1.27 GB BF16) to the drafter, or patch the loader.
   - The lm_head is shared correctly on the last rank.
5. **QSA ring (`common/qsa_cache.py:960-965`):** `block_size % (4·ceil((4+K)/4)) == 0`. K = 3/4 work; K = 5-7 assert. The fix is the widening from vLLM PR #54912 (about 40 lines).
6. **Already OK:**
   - FA2 non-causal on sm_89 (`flash_attn.py:396`), with UNIFORM_BATCH cudagraphs (`:614-618`);
   - `sample_from_anchor` is absent;
   - the None MTP-hidden getter is handled.

**Compatibility:**
- **V2:** yes.
- **PP2:** yes, after item 3.
- **Graphs:** draft decode runs FULL_DECODE_ONLY; draft prefill runs eager (PIECEWISE is not supported for dflash, `dflash/speculator.py:139`).
- **RecoverSSM:** method-agnostic (`config/vllm.py:3355-3375`).
- **Adaptive verification:** optional and off by default.

**Unknowns:**
- the draft sliding-window KV group sizing in the hybrid allocator;
- draft-KV prefix-cache hits in multi-turn Duck;
- async scheduling with the PP relay.

## Q2. Drafter weights
- **`incoai/Qwen3.8-Flash-Next-DFlash2`** (HF, Apache-2.0):
  - `model.safetensors` 1,206,140,008 B: 5 layers, hidden 2560, block_size 8, target_layer_ids [5, 15, 25, 35, 45], mask_token_id 248077, vocab 248320.
  - About 603M BF16 params; no embed / lm_head.
  - Served at block 4 in SGLang only. The model card says this is limited by the QSA implementation.
- **`PixelML/...-NVFP4-DFlash`:** DeepSpec with `sample_from_anchor`, which Merlin rejects; claimed AL 2.883, worse than MTP k4. Not useful.
- **`ggml-org` GGUF sidecars:** GGUF only.
- **z-lab:** none for Flash-Next.
- **Our 27B DFlash2** (`foysalemonshanto/qwen3-8-27b-dflash2`) cannot be reused: hidden 5120 vs 2560, 64 vs 48 layers, mask token 248070 vs 248077.

## Q3. Upstream vLLM 0.32.0 without Merlin
Upstream also lacks the seam (`Qwen4ExpForCausalLM` without SupportsEagle3; issue #56088 open). Leaving Merlin would lose:
- RecoverSSM / ReplaySSM;
- the FP8 PLE format (BF16 PLE is about 95-102 GB host RAM, against the 177 GiB cgroup);
- the `fn-quad-24g-sm89` profile;
- the Merlin W4A16 checkpoint path;
- Merlin's MTP extras;
- the RC2 validation.

Humming low-bit is NOT Merlin-only: it is in upstream 0.32.0.

## Q4. Expected gain (estimates)
- **incoai published AL, MTP → DFlash2 at 3 drafted tokens:** ×1.06-1.17, mean ×1.12 (short context, c1).
- **Ours, K = 3:** AL 2.83 → about 3.17. Net about +9 % (range −8 % to +28 %).
  - DFlash2 is one pass of about 1.4 GB per card on PP1, about 8-12 ms with all-reduces, vs MTP K3's 3 sequential passes. Net tick change about −3 to +5 ms.
- **K = 7** (block 8, as trained; needs the ring patch): break-even needs AL ≥ about 3.9. Our 27B DFlash2 measured AL 4.43 → 3.77 on ARC over context. Unmeasured here.
- **K = 4** needs no ring patch and may be the sweet spot.

## Q5. Next steps
1. Build the seam patch set: allowlist, seam, PP relay, drafter embedding, optional ring.
2. Run a Kaggle CPU kit: patch apply; meta-device config resolution with `method=dflash` (fc in 12800, aux layers (6, 16, 26, 36, 46)); embedding non-zero.
3. Run one GPU notebook, MTP K3 vs DFlash2 K = 3 / 4 / 7: teacher-forced acceptance, then served ladders. Gates: running == level, no garbage, the max-concurrency / F line.

**Sources:**
- https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2
- https://github.com/roozich/Qwen3.8-Flash-Spark
- https://github.com/vllm-project/vllm/issues/56088
- https://huggingface.co/PixelML/Qwen3.8-Flash-Next-NVFP4-DFlash
