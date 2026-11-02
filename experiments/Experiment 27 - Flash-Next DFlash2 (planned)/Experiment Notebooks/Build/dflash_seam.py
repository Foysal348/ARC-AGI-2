"""E27 DFlash2 seam: env-gated source patch that lets Merlin v3.0.0 (vLLM 0.32 fork, V2 model runner) serve
Qwen3.8-Flash-Next (Qwen4Exp, served as Qwen4ExpForConditionalGeneration --language-model-only) with a DFlash /
DFlash2 drafter (incoai/Qwen3.8-Flash-Next-DFlash2), under TP2 x PP2 (25 / 23 layers).

Files and pre-patch line numbers (Merlin ad42e37, tree 07c16a79 = the tree the Kaggle payload installs):
  vllm/model_executor/models/config.py               Qwen4ExpForConditionalGenerationConfig allowlist :1049-1058
  vllm/models/qwen4_exp/nvidia/model.py              interfaces import :42-54, ops import :101, Qwen4ExpModel :567-568,
                                                     embed_input_ids :670-671, forward :723-727 / :754-787 / :817-820,
                                                     Qwen4ExpForCausalLM bases :873-876,
                                                     Qwen4ExpForConditionalGeneration bases :1164-1167
  vllm/model_executor/models/qwen3_dflash.py         DFlashQwen3ForCausalLM :771-772, load_weights :886-888 / :918-925
  vllm/v1/worker/gpu/spec_decode/dflash/utils.py     imports :13-16, load_dflash_model :55-60 / :70-76
  vllm/models/qwen4_exp/common/qsa_cache.py          QSAKeyStateCache :886-889, get_kv_cache_spec :960-962
  vllm/model_executor/layers/attention/attention.py  set_default_quant_scales :133, Attention.get_kv_cache_spec :637
  vllm/v1/worker/gpu/spec_decode/dflash/speculator.py  DFlashSpeculator :35-36, set_attn end :251-254,
                                                     propose in-graph metadata branch :496-507
  vllm/model_executor/models/qwen3_dflash2.py        _grouped_conv :144-154 (fused kernel since vLLM #55960)
Read, not patched: interfaces.py EagleModelMixin :1571-1643 / SupportsEagle3 :1676-1735, eagle3_utils.py :15-137,
  deepseek_v4/nvidia/model.py :1287-1288 / :1466-1527, model_runner.py :210-260 / :386-393 / :524-533 / :669 /
  :2442-2456 / :2554-2591, eagle/utils.py :42-74, speculator.py :119-165, vocab_parallel_embedding.py :541-580 /
  :636-744, nvidia/hyperconnection.py :205-211, nvidia/ops/hc.py :235-266, config/speculative.py :375-385 / :426-430.
NOT touched: pp_utils.py, model_runner.py, gpu_worker.py (E25 nb4_ppack's strict shas stay valid).

What the patch does (every new line is marked "E27"; with every env unset the original statements run):
  1 Allowlist (config.py): "dflash" joins {mtp, ngram, ngram_gpu} ONLY when DFLASH_SEAM=1. Unset: the original set and
    the original error message.
  2 Seam (nvidia/model.py): Qwen4ExpModel(nn.Module, EagleModelMixin); SupportsEagle3 mounted explicitly on BOTH
    Qwen4ExpForCausalLM and Qwen4ExpForConditionalGeneration (the latter also inherits it through
    Qwen3VLForConditionalGeneration); the protocol's default set_aux_hidden_state_layers resolves the decoder
    (self.model, resp. get_language_model().model). Qwen4ExpModel._set_aux_hidden_state_layers is overridden: it
    raises unless DFLASH_SEAM=1, checks 0 <= id <= num_hidden_layers, reads DFLASH_TAP_MODE once, calls
    EagleModelMixin's (PP layout cache) and records this stage's tap ids; one 'aux_layers' log line.
    Tap id L (the runner passes dflash target_layer_ids + 1, eagle3_utils.py:108-111) = the completed output of
    zero-based layer L - 1: the pending HC tuple (hidden_states, block_output, injection) materialized with the pure
    layer.mlp_hyper_connection.combine (ops/hc.py:235-266 allocates a new tensor; the PP / deepstack paths use the
    same call; the pending tuple is untouched), then .unflatten(-1, (hc_count, hidden_size)).mean(-2) -> [T, 2560]:
    post-combine, pre-norm, mean over the 4 hyper-connection streams (incoai's capture.patch: completed zero-based
    layer outputs; = roozich's port; = DeepseekV4Model). Captured after the deepstack addition (Qwen3VL order);
    id 0 = the embedding. DFLASH_TAP_MODE for a GPU A/B only (default combine):
      combine      as above (the trained definition);
      pre_combine  the state before the pending MLP combine (verify report B's A/B);
      norm_input   the completed state after the NEXT HC module's per-stream Gemma RMSNorm (layer L's
                   attn_hyper_connection, or the final mixer for L = num_layers); refused at set time when that module
                   is on another PP stage or layer L adds PLE before its mix (PLE is at layer id 2 only).
    Last rank returns (hidden, remote_aux + local_aux) when the layers are set; aux_hidden_state_layers == () (the
    mixin default, so unless a DFlash drafter set them) runs exactly the original statements.
  3 PP2 relay (DeepseekV4Model template): supports_aux_hidden_states_over_pp = True; non-last ranks return
    IntermediateTensors({"hidden_states": h, **pack_local_aux_hidden_states(local)}) (keys aux_hidden_states_<slot>,
    the keys reserve_aux_intermediate_tensor_slots reserves on rank 1 as [B, 2560], so the taps are hc-averaged before
    the hand-off); the last rank returns collect_remote_aux_hidden_states(it) + local, REMOTE FIRST: ids (6, 16) from
    PP0, then (26, 36, 46); slot base bisect_right((6,16,26,36,46), 25) = 2. Both ends check the taps: count ->
    RuntimeError("DFLASH_AUX_COUNT_MISMATCH ..."), width != hidden_size -> "DFLASH_AUX_WIDTH_MISMATCH ..." (never a
    short or permuted list). The setter's one 'aux_layers' line per rank logs the local ids ([6, 16] on PP0, [26, 36,
    46] on PP1), slot_base and upstream_total (the remote count, 2 on PP1). HC-SP: each tap is sp_all_gather'ed to the full token axis (HC-SP is off here and
    config.py:1002-1014 refuses it with PP > 1; kept for PP1 + HC-SP).
  4 Drafter embedding / head (deterministic either way):
    Primary = repacked drafter with embed_tokens.weight (the target's BF16 [248320, 2560], byte-copied), NO loader
    code: qwen3_dflash.py :903-904 keeps it out of the :918-919 drop, process_eagle_weight (models/utils.py
    :1144-1145) sets has_own_embed_tokens = True, maybe_share_target_embed (eagle/utils.py:61-69) keeps the copy on PP
    rank 1, and on a PP1 boot _should_share finds it equal and shares.
    Fallback (env-gated, only without the repack): DFLASH_EMBED_SIDECAR (a safetensors file, or a model dir via
    model.safetensors.index.json; tensor DFLASH_EMBED_KEY, else the single '*embed_tokens.weight' key without 'mtp');
    appended as embed_tokens.weight under PP > 1 through the same stock path; set together with a checkpoint that
    ships embed_tokens it raises.
    Loud: DFlashQwen3ForCausalLM.has_own_embed_tokens = False is explicit on the draft class (= the SupportsEagleBase
    default, so behaviour is unchanged): maybe_share_target_embed's silent PP early return (needs the attribute
    absent) can never be taken, a missing embedding raises "needs the target input embedding" at load. Merlin
    ad42e37 already raises there (verify report B was right, A's "silent" was not; tests T7 run the real function).
    Load checks (qwen3_dflash.py load_weights + dflash/utils.py _e27_finish_draft, DFLASH_SEAM=1):
      every drafter parameter got a tensor (the default loader's strict check is off because this load_weights
      returns None), except the ones mapped away on purpose and an lm_head the checkpoint lacks (DFLASH_LOAD_CHECK=0
      turns this one off); _e27_embed_loaded records whether embed_tokens really came with the weights;
      embed_tokens = the target's module (shared) or own AND loaded; weight 2-D, not meta, a 4096-row sample finite
      and not all zero; lm_head = the target's or own; every drafter linear UnquantizedLinearMethod (the fused
      context-KV path needs it); sha256 + norm of embedding row mask_token_id (248077) on the TP rank holding it,
      compared with DFLASH_MASK_ROW_SHA when set (embed_row_sha() below computes it from the target checkpoint).
    Memory: the draft-owned lm_head (and, on a PP1 share, the draft embed_tokens) the share replaced is released
    (speculator.py _release_replaced_draft_head pattern: clear _parameters / _buffers, gc.collect, empty_cache):
    636 MiB per TP2 card each. DFLASH_DRAFT_EMBED_HOST=1 moves an own drafter embedding to pinned host memory read
    through a UVA view (model_runner.py _host_embed_tables pattern; graphs capture it), 0.59 GiB per PP1 card;
    drafter only: VLLM_HOST_EMBED_TABLE / the target and MTP tables are not touched (the event records host_mib).
  5 QSA ring (qsa_cache.py, separate env DFLASH_QSA_RING_WIDEN=1): qsa_ring_capacity(ratio, K, block) in the style of
    vLLM PR #54912 (own text): the minimal whole-group ring when it divides the block (byte-identical to the original
    formula, e.g. K = 3 / 4 -> 8), else the next multiple of the ratio that divides it, at most 2x the minimal ring,
    else a ValueError naming the sizes. Ratio 4, block 4096: K = 5..8 -> 12 -> 16 (K = 7 needs it). Unset: the
    original code including the assert.
  7 In-graph draft decode metadata (dflash/speculator.py, DFLASH_SEAM=1; vLLM issue #63587, fix = vLLM PR #64059,
    head 7618884b, ported in own text): with FULL graphs and supports_decode_metadata_update, propose() skips
    _build_uniform_attn_metadata (:499-507), the only call that maps the current block tables into the attention
    groups' persistent kernel block tables (AttentionGroup.map_to_kernel_block_table, v1/worker/utils.py:415); with
    kernel blocks smaller than the 4096 manager block the replayed graph reads stale draft KV (issue: acceptance
    0.531 -> 0.004). The patch adds the PR's `elif` branch: every attention group's kernel block table is re-mapped
    from self.block_tables.input_block_tables[group.kv_cache_group_id] before run_fullgraph. Gated by the class
    default _e27_kernel_map_refresh = False, set True at the end of set_attn (_e27_configure_ingraph_meta, before
    capture) only with DFLASH_SEAM=1 and in-graph updates on. DFLASH_NO_INGRAPH_META=1 (A/B fallback, the issue's
    workaround, default off): supports_decode_metadata_update = False for this speculator, so the metadata is rebuilt
    eagerly every step. One 'ingraph_meta' line per drafter worker (native / final flag, forced_off,
    kernel_map_refresh, per-group manager / kernel block).
  8 Grouped convolution A/B (qwen3_dflash2.py, DFLASH_CONV_IMPL=torch with DFLASH_SEAM=1; default fused): Merlin
    carries vLLM #55960's fused Triton DFlash2 grouped conv (:25-141, "adapted from LightSeek TokenSpeed PR #1399")
    and runs it on every CUDA call (:153-154). vLLM #58250 (open, stale) reverts it because the DFlash
    acceptance-length nightly fell 3.195 -> 3.060 after it landed (accuracy unchanged). DFLASH_CONV_IMPL=torch routes
    the conv through the torch path on CUDA too, with SGLang #41736's fix: the cross-request taps are masked with
    torch.where, so NaN / Inf in one request's rows cannot leak through `0 * NaN` (Merlin's CPU-only torch fallback
    :156-167 still multiplies by the mask; the fused kernel masks its loads, :57-70, and is not affected). One
    'conv_impl' line per drafter worker at import. Unset: the fused kernel, unchanged.
  6 Drafter sliding-window KV block (attention.py, separate env DFLASH_DRAFT_SW_BLOCK=<n>): the SW layers marked
    is_draft_layer use block n (a divisor of the attention block size, so the scheduler LCM 4096 is unchanged)
    instead of the primary 4096. Unset: the original block choice. Its effect on pool / F is NOT measured (kit gate
    below). Caution: the engine then sets cache_config.block_size = min over the prefix-cacheable group blocks
    (v1/engine/core.py:436-448), i.e. n; Merlin already seeds the mamba state column with mamba_block_size (the vLLM
    #55601 fix for #60174's all-NaN logits, mamba_hybrid.py:158-166), but every other reader of
    cache_config.block_size under a lowered value is untested: gate it on the kit before any GPU arm.

Env (all unset = the original Merlin behaviour):
  DFLASH_SEAM=1                    allowlist, aux setter, load checks, head / embed release (engine + every worker)
  DFLASH_TAP_MODE=combine|pre_combine|norm_input   tap definition, default combine (the others: A/B only)
  DFLASH_EMBED_SIDECAR=<path>      fallback only (no repacked drafter), PP > 1, needs DFLASH_SEAM=1
  DFLASH_EMBED_KEY=<name>          optional tensor name inside the sidecar
  DFLASH_MASK_ROW_SHA=<hex>        expected sha256 of embedding row 248077 (embed_row_sha of the target checkpoint)
  DFLASH_DRAFT_EMBED_HOST=1        drafter embedding in pinned host memory (UVA); for W4's 191 MiB PP1 headroom
  DFLASH_LOAD_CHECK=0              disables the every-parameter-loaded check (default on with DFLASH_SEAM=1)
  DFLASH_QSA_RING_WIDEN=1          K >= 5 ring widening (independent of DFLASH_SEAM)
  DFLASH_DRAFT_SW_BLOCK=<n>        drafter SW KV block, e.g. 1024 (independent of DFLASH_SEAM)
  DFLASH_NO_INGRAPH_META=1         A/B fallback: no in-graph draft metadata update (needs DFLASH_SEAM=1)
  DFLASH_CONV_IMPL=fused|torch     DFlash2 grouped conv, default fused; torch = A/B (needs DFLASH_SEAM=1)
Launch (no patch needed), --speculative-config '{"method": "dflash", "model": <drafter dir>, "num_speculative_tokens": K,
  "draft_sample_method": "probabilistic", "kv_cache_dtype": "auto", "attention_backend": "FLASH_ATTN"}':
  "method" explicit (the Kaggle path may not contain "dflash"); "kv_cache_dtype": "auto" = BF16 drafter KV, because the
  inherited fp8_e4m3 is rejected by FA2 on sm_89 (fallback FLASHINFER -> eager drafter) and would run with unit scales
  (config/speculative.py:380-385 / :426-430, dflash/utils.py:34-41); "attention_backend": "FLASH_ATTN" (non-causal,
  UNIFORM_BATCH full graphs). Capture sizes: the target / drafter decode graphs round to multiples of 1 + K; set
  cudagraph_capture_sizes up to max_seqs_per_microbatch x (1 + K) (W4 9 streams: K3 36, K4 45, K7 72).
Kit gates this file does not cover: Merlin's get_kv_cache_configs on the real per-stage specs with the drafter group
  (pool / F / groups) for each K and each DFLASH_DRAFT_SW_BLOCK value: the GDN / RecoverSSM / PLE state shapes use
  num_speculative_tokens (nvidia/model.py:985-1040), so the mamba page, stride and pool change with K (K = 7 vs 3).
Log lines (stdout of every worker): "E27_DFLASH {json}", events aux_layers / draft_load_check / embed_sidecar /
  embed_sidecar_skipped / embed / qsa_ring / draft_sw_block / ingraph_meta / conv_impl.

Composition with E25 NB4 Q-pack / Q-defer (nb4_ppack.py): disjoint files (NB4: pp_utils.py, model_runner.py,
  gpu_worker.py), so the two compose in either order; each module verifies only its own files and every file's pre /
  post sha is the same whatever the order (tests T10). The aux taps ride the PP isend_tensor_dict of the device group,
  not the pp_broadcast sibling group Q-pack / Q-defer change. NB4_FILES lists NB4's files; an overlap is refused at
  import.

What this file holds (the notebook embeds or imports it):
  PATCHES {rel: (pre_sha, post_sha, marker, unified diff)}  strict file patch (nb4_ppack style), 8 files;
  CORE_RELS (1-4, 7, 8) + RING_RELS (5, ring=True) + DRAFT_BLOCK_RELS (6, draft_block=True) for apply / verify / revert;
  apply_dflash_patch / verify_dflash_patch / revert_dflash_patch, tree_state / combined_tree_state (ORIGINAL /
  FULLY_PATCHED / INVALID_MIXED, with nb4_ppack), server_env, embed_row_sha, parse_log, readback_label.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

MERLIN_COMMIT = "ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5"
MERLIN_TREE = "07c16a79d077f1e06c19d0eea84efd64f0b63540"
CFG_REL = "vllm/model_executor/models/config.py"
MDL_REL = "vllm/models/qwen4_exp/nvidia/model.py"
DFL_REL = "vllm/model_executor/models/qwen3_dflash.py"
DFU_REL = "vllm/v1/worker/gpu/spec_decode/dflash/utils.py"
QSA_REL = "vllm/models/qwen4_exp/common/qsa_cache.py"
ATT_REL = "vllm/model_executor/layers/attention/attention.py"
SPC_REL = "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"
DF2_REL = "vllm/model_executor/models/qwen3_dflash2.py"
CFG_PRE = "1a16138305f395f10e87b1ec9de40d501eba2a3c0f1379a6c787bb5c2d2e61ac"
CFG_POST = "0f6f63d355a269743cf8b6c0c2ef2b43aad066f5c1d202e534b02d3f44ff2711"
CFG_MARKER = '# E27 DFLASH_SEAM: "dflash" (DFlash / DFlash2 drafters) is admitted only'
MDL_PRE = "068adee6c0e062342765d10e1e22da5967c428059197067138064b137f973397"
MDL_POST = "787161118bb49790fb5b8dd99158e029eed3e855092dfb081e3173c1b74e2f1c"
MDL_MARKER = '# E27 DFLASH_SEAM: DFlash taps (aux hidden states) for a DFlash / DFlash2'
DFL_PRE = "5375543288e72219abc1fc777219a640f0d173b3e8ac531718d666483a46ea7a"
DFL_POST = "774201928124445e72052ead3df482ce95a414e5467b3a8db3caca838fd14e24"
DFL_MARKER = "# E27 DFLASH_SEAM: the drafter's input embedding from a sidecar safetensors file."
DFU_PRE = "f0584417149c0a6fa67e700b2a725e40a41f4546a806b26e6cc64b64c59cfd66"
DFU_POST = "fb76c074fdc5ecc94148a7efdeddaec426e4aaf286e812d6e715237a2dd7b5a3"
DFU_MARKER = '# E27 DFLASH_SEAM: refuse an uninitialised drafter embedding / LM head, free'
QSA_PRE = "43f323bb347e2152e47acb36a99d0682eaebc5cdafe9b5ba168b5c4b4e9acf13"
QSA_POST = "07b0b61f9947d36ab2b3f87958830aaf591e899a93ab61f67c4debc50c442bde"
QSA_MARKER = '# E27 DFLASH_QSA_RING_WIDEN: the ring widening of vLLM PR #54912 (open upstream),'
ATT_PRE = "bde77677fd0e95ea09c6a580dfaf09b57741e8183e00cb106e2d1cc64d8e03f2"
ATT_POST = "5817d764b870409ef0ef5b0c9e619f532a6410cece82833bd464dcea69181a5c"
ATT_MARKER = "# E27 DFLASH_DRAFT_SW_BLOCK=<n>: a smaller KV block for the drafter's sliding-"
SPC_PRE = "b731b09c92b93063d89851a997e1de70e51d512bc40b2a4dd68274a3a999e187"
SPC_POST = "85f57f0ee9bc7018f052f59d0610a91c78d35cd018ef711c6f7ecbef298d47ea"
SPC_MARKER = '# E27 DFLASH_SEAM: refresh the kernel block tables on the in-graph metadata'
DF2_PRE = "11ac6ba8c6db702375970e68cd042b98783a5c43b87f9bfa27e88463316bf499"
DF2_POST = "4039da670a5f6a91ef5e5931465a652d4145207c90b87ff44fdb7b1f61d04402"
DF2_MARKER = '# E27 DFLASH_CONV_IMPL=torch (with DFLASH_SEAM=1; A/B, default = the fused'
GRD_REL = "vllm/v1/e2_guards.py"
GRD_PRE = "aa25244647523c74b43d6fe93d61de965bbbe810c92a20f7d5bd4c2dd516e5c3"
GRD_POST = "d50d754e8479cf77b577f283ef6456a8a8db550b179ca9aa6a4c26d3126bde11"
GRD_MARKER = "# E27 DFLASH_SEAM: the DFlash / DFlash2 drafter's graph manager (query_cudagraph_manager)"
ENV_SEAM = "DFLASH_SEAM"
ENV_TAP_MODE = "DFLASH_TAP_MODE"
ENV_EMBED = "DFLASH_EMBED_SIDECAR"
ENV_EMBED_KEY = "DFLASH_EMBED_KEY"
ENV_MASK_SHA = "DFLASH_MASK_ROW_SHA"
ENV_EMBED_HOST = "DFLASH_DRAFT_EMBED_HOST"
ENV_LOAD_CHECK = "DFLASH_LOAD_CHECK"
ENV_RING = "DFLASH_QSA_RING_WIDEN"
ENV_SW_BLOCK = "DFLASH_DRAFT_SW_BLOCK"
ENV_NO_INGRAPH = "DFLASH_NO_INGRAPH_META"
ENV_CONV_IMPL = "DFLASH_CONV_IMPL"
TAP_MODES = ("combine", "pre_combine", "norm_input")
ALL_ENVS = (ENV_SEAM, ENV_TAP_MODE, ENV_EMBED, ENV_EMBED_KEY, ENV_MASK_SHA, ENV_EMBED_HOST, ENV_LOAD_CHECK, ENV_RING,
            ENV_SW_BLOCK, ENV_NO_INGRAPH, ENV_CONV_IMPL)
MASK_TOKEN_ID = 248077                                        # incoai dflash_config.mask_token_id
NB4_FILES = ("vllm/v1/worker/gpu/pp_utils.py", "vllm/v1/worker/gpu/model_runner.py", "vllm/v1/worker/gpu_worker.py")
DFLASH_TARGET_LAYER_IDS = (5, 15, 25, 35, 45)                 # incoai drafter config
AUX_IDS = tuple(i + 1 for i in DFLASH_TARGET_LAYER_IDS)       # the runner's +1 shift (eagle3_utils.py:108-111)

CFG_PATCH = r'''# E27 DFLASH_SEAM onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5, tree 07c16a79).
# Qwen4ExpForConditionalGenerationConfig.verify_and_update_config: admit "dflash" only with DFLASH_SEAM=1.
# pre-patch sha256  1a16138305f395f10e87b1ec9de40d501eba2a3c0f1379a6c787bb5c2d2e61ac
# post-patch sha256 0f6f63d355a269743cf8b6c0c2ef2b43aad066f5c1d202e534b02d3f44ff2711
--- a/vllm/model_executor/models/config.py
+++ b/vllm/model_executor/models/config.py
@@ -1047,14 +1047,20 @@
         if multimodal_config is not None and multimodal_config.language_model_only:
             _strip_qwen4_exp_mrope(vllm_config.model_config)
         spec_config = vllm_config.speculative_config
-        if spec_config is not None and spec_config.method not in {
-            "mtp",
-            "ngram",
-            "ngram_gpu",
-        }:
+        # E27 DFLASH_SEAM: "dflash" (DFlash / DFlash2 drafters) is admitted only
+        # with DFLASH_SEAM=1, which also enables the aux-hidden-state seam of
+        # models/qwen4_exp/nvidia/model.py. Unset: the original allowlist.
+        import os as _e27_os
+
+        _e27_seam = _e27_os.environ.get("DFLASH_SEAM", "0") == "1"
+        _e27_methods = {"mtp", "ngram", "ngram_gpu"}
+        if _e27_seam:
+            _e27_methods = _e27_methods | {"dflash"}
+        if spec_config is not None and spec_config.method not in _e27_methods:
             raise NotImplementedError(
                 "Qwen4Exp speculative decoding supports only its native MTP "
                 "checkpoint and linear n-gram proposers"
+                + (" (and DFlash drafters, DFLASH_SEAM=1)" if _e27_seam else "")
             )
 
 
'''

MDL_PATCH = r'''# E27 DFLASH_SEAM onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5, tree 07c16a79).
# Qwen4ExpModel + EagleModelMixin (DFlash taps = completed layer outputs averaged over the hc streams, PP
# relay, HC-SP gather, setter refused unless DFLASH_SEAM=1); Qwen4ExpForCausalLM + SupportsEagle3.
# pre-patch sha256  068adee6c0e062342765d10e1e22da5967c428059197067138064b137f973397
# post-patch sha256 787161118bb49790fb5b8dd99158e029eed3e855092dfb081e3173c1b74e2f1c
--- a/vllm/models/qwen4_exp/nvidia/model.py
+++ b/vllm/models/qwen4_exp/nvidia/model.py
@@ -40,11 +40,13 @@
     should_use_host_embed_table,
 )
 from vllm.model_executor.models.interfaces import (
+    EagleModelMixin,  # E27 DFLASH_SEAM
     HasInnerState,
     IsHybrid,
     MambaStateShapes,
     MixtureOfExperts,
     MultiModalEmbeddings,
+    SupportsEagle3,  # E27 DFLASH_SEAM
     SupportsHCSP,
     SupportsLoRA,
     SupportsMRoPE,
@@ -99,6 +101,7 @@
 from .hyperconnection import GatedResidual, HyperConnectionConfig
 from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
 from .ops.cute_dsl.hc_down_silu import request_hc_down_silu_warmup
+from .ops.hc import grouped_gemma_rmsnorm  # E27 DFLASH_SEAM: tap mode norm_input
 from .ple_layer import Qwen4ExpPLELayer
 from .qsa import Qwen4ExpQSAAttention, load_qsa_static_kv_scales
 
@@ -564,8 +567,17 @@
         "deepstack_input_embeds": 0,
     }
 )
-class Qwen4ExpModel(nn.Module):
+class Qwen4ExpModel(nn.Module, EagleModelMixin):
     hf_to_vllm_mapper = Qwen3_5Model.hf_to_vllm_mapper | _EXTRA_WEIGHTS_MAPPER
+    # E27 DFLASH_SEAM: DFlash taps (aux hidden states) for a DFlash / DFlash2
+    # drafter, relayed over PP like DeepseekV4Model. aux_hidden_state_layers
+    # stays () (the EagleModelMixin default) unless DFLASH_SEAM=1 and the runner
+    # sets the drafter's layers; with () forward runs the original path only.
+    supports_aux_hidden_states_over_pp = True
+    # The tap ids this PP stage captures and the tap mode (DFLASH_TAP_MODE,
+    # read once by _set_aux_hidden_state_layers).
+    _e27_local_aux_ids: tuple[int, ...] = ()
+    _e27_tap_mode: str = "combine"
 
     def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
         super().__init__()
@@ -669,6 +681,152 @@
 
     def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
         return self.embed_tokens(input_ids)
+
+    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
+        """E27 DFLASH_SEAM: EagleModelMixin's setter, refused unless
+        DFLASH_SEAM=1. Ids use the runner's +1-shifted semantics: id L is the
+        completed output of zero-based layer L - 1 (id 0 is the embedding)."""
+        if os.environ.get("DFLASH_SEAM", "0") != "1":
+            raise RuntimeError(
+                "Qwen4Exp aux hidden states (DFlash taps) need DFLASH_SEAM=1 "
+                "(E27 dflash_seam patch)"
+            )
+        ids = tuple(sorted({int(i) for i in layers}))
+        num_layers = int(self.config.num_hidden_layers)
+        bad = [i for i in ids if not 0 <= i <= num_layers]
+        if not ids or bad:
+            raise ValueError(
+                f"Qwen4Exp DFlash tap ids {tuple(layers)} must be non-empty and "
+                f"within 0..{num_layers} (bad: {bad})"
+            )
+        mode = os.environ.get("DFLASH_TAP_MODE", "combine").strip() or "combine"
+        if mode not in ("combine", "pre_combine", "norm_input"):
+            raise ValueError(
+                f"DFLASH_TAP_MODE={mode!r}: expected combine, pre_combine or "
+                "norm_input"
+            )
+        local = tuple(
+            i
+            for i in ids
+            if self.start_layer < i <= self.end_layer
+            or (i == 0 and self.start_layer == 0)
+        )
+        self._e27_tap_mode = mode
+        if mode == "norm_input":
+            for i in local:
+                self._e27_next_hc(i)  # raises if the next HC module is not here
+        EagleModelMixin._set_aux_hidden_state_layers(self, ids)
+        self._e27_local_aux_ids = local
+        import json as _e27_json
+
+        print(
+            "E27_DFLASH "
+            + _e27_json.dumps(
+                {
+                    "event": "aux_layers",
+                    "pid": os.getpid(),
+                    "ids": list(ids),
+                    "local": list(self._e27_local_aux_ids),
+                    "start_layer": int(self.start_layer),
+                    "end_layer": int(self.end_layer),
+                    "num_layers": num_layers,
+                    "slot_base": int(self._aux_slot_base_cached),
+                    "upstream_total": int(self._aux_upstream_total_cached),
+                    "hc_sp": bool(self.use_hc_sequence_parallel),
+                    "tap_mode": mode,
+                },
+                sort_keys=True,
+            ),
+            flush=True,
+        )
+
+    def _e27_next_hc(self, tap_id: int) -> nn.Module:
+        """E27 DFLASH_SEAM (tap mode norm_input): the HC module that consumes
+        tap id L next: layer L's attn_hyper_connection, or the final mixer for
+        L == num_hidden_layers. It must live on this PP stage and layer L must
+        not add PLE before its mix (the PLE term is not part of the tap)."""
+        if tap_id == int(self.config.num_hidden_layers):
+            if self.hyper_connection_mixer is None:
+                raise ValueError(f"tap {tap_id}: the final HC mixer is not here")
+            return self.hyper_connection_mixer
+        if not self.start_layer <= tap_id < self.end_layer:
+            raise ValueError(
+                f"DFLASH_TAP_MODE=norm_input: tap {tap_id} feeds layer {tap_id}, "
+                f"not on this PP stage ({self.start_layer}..{self.end_layer - 1})"
+            )
+        layer = self.layers[tap_id]
+        if getattr(layer, "ple", None) is not None:
+            raise ValueError(
+                f"DFLASH_TAP_MODE=norm_input: layer {tap_id} adds PLE before its "
+                "HC mix; tap mode norm_input is not defined there"
+            )
+        return layer.attn_hyper_connection
+
+    def _e27_dflash_tap(
+        self,
+        layer: nn.Module | None,
+        hidden_states: torch.Tensor,
+        block_output: torch.Tensor | None,
+        injection: torch.Tensor | None,
+        full_num_tokens: int,
+        tap_id: int,
+    ) -> torch.Tensor:
+        """E27 DFLASH_SEAM: one DFlash tap [T, H]. Tap mode combine (default) =
+        the completed multi-stream state [T, hc_count * H] (the pending HC
+        combine materialized with the pure combine the PP / deepstack paths
+        use; the pending tuple the loop threads on is left as it was), averaged
+        over the hc_count streams (incoai's DFlash2 capture: the completed
+        layer output, before any norm). A/B modes only: pre_combine = the state
+        before the pending MLP combine; norm_input = the completed state after
+        the next HC module's per-stream Gemma RMSNorm. Under HC-SP the local
+        rows are gathered to the full token axis (DeepseekV4Model)."""
+        if block_output is not None and self._e27_tap_mode != "pre_combine":
+            assert layer is not None
+            hidden_states = layer.mlp_hyper_connection.combine(
+                hidden_states, block_output, injection
+            )
+        if self._e27_tap_mode == "norm_input":
+            hc = self._e27_next_hc(tap_id)
+            hidden_states = grouped_gemma_rmsnorm(
+                hidden_states,
+                hc.hc_norm.weight,
+                hc.config.rms_norm_eps,
+                hc.hc_count,
+            )
+        tap = hidden_states.unflatten(
+            -1, (self.config.hc_count, self.config.hidden_size)
+        ).mean(dim=-2)
+        if self.use_hc_sequence_parallel:
+            tap = sp_all_gather(tap)[:full_num_tokens]
+        return tap
+
+    def _e27_check_aux(
+        self,
+        local: list[torch.Tensor],
+        remote: list[torch.Tensor] | None,
+    ) -> None:
+        """E27 DFLASH_SEAM: every configured tap is captured exactly once and
+        is [T, hidden_size] (remote: the last rank's upstream taps)."""
+        if len(local) != len(self._e27_local_aux_ids):
+            raise RuntimeError(
+                f"DFLASH_AUX_COUNT_MISMATCH: captured {len(local)} DFlash taps on "
+                f"this PP stage (layers {self.start_layer}..{self.end_layer - 1}), "
+                f"expected {len(self._e27_local_aux_ids)} {self._e27_local_aux_ids}"
+            )
+        if remote is not None and len(remote) + len(local) != len(
+            self.aux_hidden_state_layers
+        ):
+            raise RuntimeError(
+                f"DFLASH_AUX_COUNT_MISMATCH: {len(remote)} upstream + {len(local)} "
+                f"local DFlash taps != {len(self.aux_hidden_state_layers)} "
+                f"configured {self.aux_hidden_state_layers}"
+            )
+        for tap in list(remote or ()) + list(local):
+            if tap.shape[-1] != self.config.hidden_size:
+                raise RuntimeError(
+                    f"DFLASH_AUX_WIDTH_MISMATCH: a DFlash tap has width "
+                    f"{tap.shape[-1]}, expected hidden_size {self.config.hidden_size}"
+                )
 
     @staticmethod
     def _start_layer_ple_prefetch(
@@ -724,6 +882,19 @@
         block_output = None
         injection = None
         last_layer = None
+        # E27 DFLASH_SEAM: DFlash taps; nothing runs unless the layers are set.
+        aux_hidden_states: list[torch.Tensor] = []
+        remote_aux_hidden_states: list[torch.Tensor] = []
+        if self.aux_hidden_state_layers:
+            remote_aux_hidden_states = self.collect_remote_aux_hidden_states(
+                intermediate_tensors
+            )
+            if self.start_layer == 0 and 0 in self.aux_hidden_state_layers:
+                aux_hidden_states.append(
+                    self._e27_dflash_tap(
+                        None, hidden_states, None, None, full_num_tokens, 0
+                    )
+                )
         if self.start_layer < self.end_layer:
             self._start_layer_ple_prefetch(
                 self.layers[self.start_layer],
@@ -776,6 +947,22 @@
                 block_output = None
                 injection = None
                 hidden_states = hidden_states + deepstack_embed
+            if (
+                self.aux_hidden_state_layers
+                and layer_idx + 1 in self.aux_hidden_state_layers
+            ):
+                # E27 DFLASH_SEAM: tap id layer_idx + 1 = the completed output
+                # of layer layer_idx, averaged over the hc_count streams.
+                aux_hidden_states.append(
+                    self._e27_dflash_tap(
+                        layer,
+                        hidden_states,
+                        block_output,
+                        injection,
+                        full_num_tokens,
+                        layer_idx + 1,
+                    )
+                )
 
         if not get_pp_group().is_last_rank:
             # PP transports one tensor, not the delayed HC tuple. Materialize
@@ -783,6 +970,16 @@
             if last_layer is not None and block_output is not None:
                 hidden_states = last_layer.mlp_hyper_connection.combine(
                     hidden_states, block_output, injection
+                )
+            if self.aux_hidden_state_layers:
+                # E27 DFLASH_SEAM: this stage's taps ride the PP hand-off
+                # (keys aux_hidden_states_<slot>, the EagleModelMixin layout).
+                self._e27_check_aux(aux_hidden_states, None)
+                return IntermediateTensors(
+                    {
+                        "hidden_states": hidden_states,
+                        **self.pack_local_aux_hidden_states(aux_hidden_states),
+                    }
                 )
             return IntermediateTensors({"hidden_states": hidden_states})
 
@@ -817,6 +1014,10 @@
             # this tensor is needed by the final mixer regardless).
             num_tokens = multi_hidden.shape[0]
             self._mtp_hidden_buffer[:num_tokens].copy_(multi_hidden)
+        if self.aux_hidden_state_layers:
+            # E27 DFLASH_SEAM: upstream PP taps first, then this stage's.
+            self._e27_check_aux(aux_hidden_states, remote_aux_hidden_states)
+            return sample_hidden_states, remote_aux_hidden_states + aux_hidden_states
         return sample_hidden_states
 
     def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
@@ -873,6 +1074,9 @@
     Qwen4ExpMixtureOfExperts,
     IsHybrid,
     SupportsReplaySSM,
+    # E27 DFLASH_SEAM: the EAGLE-3 aux-hidden-state interface (the protocol's
+    # default set_aux_hidden_state_layers resolves self.model, Qwen4ExpModel).
+    SupportsEagle3,
 ):
     packed_modules_mapping = {
         "qkv_proj": ["q_proj", "k_proj", "v_proj", "indexer.index_qk_proj"],
@@ -1163,6 +1367,10 @@
     SupportsHCSP,
     Qwen4ExpMixtureOfExperts,
     SupportsReplaySSM,
+    # E27 DFLASH_SEAM: explicit (it is also inherited through
+    # Qwen3VLForConditionalGeneration); the protocol's default setter resolves
+    # get_language_model().model, Qwen4ExpModel. This is the served class.
+    SupportsEagle3,
 ):
     """Qwen3-VL vision tower backed by the Qwen4Exp language model."""
 
'''

DFL_PATCH = r'''# E27 DFLASH_SEAM onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5, tree 07c16a79).
# DFlashQwen3ForCausalLM.load_weights: embed_tokens from DFLASH_EMBED_SIDECAR under PP > 1 (DFLASH_SEAM=1).
# pre-patch sha256  5375543288e72219abc1fc777219a640f0d173b3e8ac531718d666483a46ea7a
# post-patch sha256 774201928124445e72052ead3df482ce95a414e5467b3a8db3caca838fd14e24
--- a/vllm/model_executor/models/qwen3_dflash.py
+++ b/vllm/model_executor/models/qwen3_dflash.py
@@ -768,8 +768,197 @@
         )
 
 
+# E27 DFLASH_SEAM: the drafter's input embedding from a sidecar safetensors file.
+_E27_EMBED_FLOAT_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
+
+
+def _e27_log(event: str, **kw) -> None:
+    import json
+    import os
+
+    rec = {"event": event, "pid": os.getpid()}
+    rec.update(kw)
+    print("E27_DFLASH " + json.dumps(rec, sort_keys=True, default=str), flush=True)
+
+
+def _e27_mark_embed_loaded(model: nn.Module, includes_embed_tokens: bool) -> None:
+    """E27 DFLASH_SEAM: _e27_embed_loaded tells load_dflash_model's check whether
+    embed_tokens really came with the weights (has_own_embed_tokens alone is a
+    class default that can be True). A no-op unless DFLASH_SEAM=1."""
+    import os
+
+    if os.environ.get("DFLASH_SEAM", "0") == "1":
+        model._e27_embed_loaded = bool(includes_embed_tokens)
+
+
+def _e27_check_loaded(
+    model: nn.Module, loaded, dropped: dict, model_weights: dict
+) -> None:
+    """E27 DFLASH_SEAM: raise unless every drafter parameter got a tensor from
+    the weights, except the ones the loader maps away on purpose (the
+    orig_to_new_substr keys: no embed_tokens / d2t / fc / mask_embedding in the
+    checkpoint) and an lm_head the checkpoint does not carry (shared from the
+    target later). DFLASH_LOAD_CHECK=0 turns it off. A no-op unless
+    DFLASH_SEAM=1."""
+    import os
+
+    if os.environ.get("DFLASH_SEAM", "0") != "1":
+        return
+    if os.environ.get("DFLASH_LOAD_CHECK", "1") == "0":
+        _e27_log("draft_load_check", skipped=True)
+        return
+    if loaded is None:
+        raise RuntimeError(
+            "E27 DFLASH_SEAM: the drafter loader returned no loaded-weights set"
+        )
+    has_head = any("lm_head" in n for n in model_weights)
+    expected = {
+        n
+        for n, _ in model.named_parameters()
+        if not any(s in n for s in dropped)
+        and not (n.startswith("lm_head.") and not has_head)
+    }
+    missing = sorted(expected - set(loaded))
+    if missing:
+        raise RuntimeError(
+            f"E27 DFLASH_SEAM: {len(missing)} drafter parameter(s) got no tensor "
+            f"from the checkpoint: {missing[:12]}"
+        )
+    _e27_log(
+        "draft_load_check",
+        expected=len(expected),
+        loaded=len(set(loaded)),
+        dropped=sorted(dropped),
+    )
+
+
+def _e27_with_embed_sidecar(
+    weights: Iterable[tuple[str, torch.Tensor]], config
+) -> Iterable[tuple[str, torch.Tensor]]:
+    """E27 DFLASH_SEAM: under PP > 1 the target's embed_tokens lives on PP rank
+    0 only and the incoai DFlash2 checkpoint ships none, so the drafter on the
+    last rank needs its own copy. DFLASH_EMBED_SIDECAR names a safetensors file,
+    or a model directory (model.safetensors.index.json, else every
+    *.safetensors), holding the target's input embedding; DFLASH_EMBED_KEY
+    optionally names the tensor. It is appended as embed_tokens.weight, so the
+    stock path (includes_embed_tokens -> has_own_embed_tokens) loads it.
+    DFLASH_SEAM or DFLASH_EMBED_SIDECAR unset: the weights pass through."""
+    import os
+
+    path = os.environ.get("DFLASH_EMBED_SIDECAR", "").strip()
+    if os.environ.get("DFLASH_SEAM", "0") != "1" or not path:
+        return weights
+    return _e27_embed_sidecar_iter(weights, config, path)
+
+
+def _e27_embed_sidecar_iter(
+    weights: Iterable[tuple[str, torch.Tensor]], config, path: str
+) -> Iterable[tuple[str, torch.Tensor]]:
+    from vllm.distributed import get_pp_group
+
+    ships_embed = False
+    for name, loaded_weight in weights:
+        if "embed_tokens" in name:
+            ships_embed = True
+        yield name, loaded_weight
+    if get_pp_group().world_size <= 1:
+        _e27_log(
+            "embed_sidecar_skipped",
+            reason="pp_size 1: the target embedding is shared",
+            path=path,
+        )
+        return
+    if ships_embed:
+        raise ValueError(
+            "E27 DFLASH_SEAM: the drafter checkpoint ships embed_tokens and "
+            "DFLASH_EMBED_SIDECAR is set as well; unset one of them"
+        )
+    yield "embed_tokens.weight", _e27_read_embed_sidecar(path, config)
+
+
+def _e27_read_embed_sidecar(path: str, config) -> torch.Tensor:
+    """The target's input embedding [vocab_size, hidden_size] from `path`;
+    any problem (missing file / key, ambiguous key, shape, dtype, non-finite
+    or all-zero rows) raises."""
+    import glob
+    import json
+    import os
+
+    from safetensors import safe_open
+
+    want = os.environ.get("DFLASH_EMBED_KEY", "").strip()
+    located: dict[str, str] = {}
+    if os.path.isdir(path):
+        index = os.path.join(path, "model.safetensors.index.json")
+        if os.path.isfile(index):
+            with open(index, encoding="utf-8") as f:
+                weight_map = json.load(f)["weight_map"]
+            located = {k: os.path.join(path, v) for k, v in weight_map.items()}
+        else:
+            for file in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
+                with safe_open(file, framework="pt", device="cpu") as f:
+                    for k in f.keys():
+                        located.setdefault(k, file)
+    elif os.path.isfile(path):
+        with safe_open(path, framework="pt", device="cpu") as f:
+            located = dict.fromkeys(f.keys(), path)
+    else:
+        raise FileNotFoundError(
+            f"E27 DFLASH_SEAM: DFLASH_EMBED_SIDECAR={path!r} does not exist"
+        )
+    if want:
+        if want not in located:
+            raise KeyError(
+                f"E27 DFLASH_SEAM: DFLASH_EMBED_KEY={want!r} is not in {path!r}"
+            )
+        key = want
+    else:
+        cands = sorted(
+            k for k in located if k.endswith("embed_tokens.weight") and "mtp" not in k
+        )
+        if len(cands) != 1:
+            raise KeyError(
+                f"E27 DFLASH_SEAM: {len(cands)} embed_tokens candidates in "
+                f"{path!r} ({cands[:8]}); set DFLASH_EMBED_KEY"
+            )
+        key = cands[0]
+    with safe_open(located[key], framework="pt", device="cpu") as f:
+        tensor = f.get_tensor(key)
+    want_shape = (int(config.vocab_size), int(config.hidden_size))
+    if tuple(tensor.shape) != want_shape:
+        raise ValueError(
+            f"E27 DFLASH_SEAM: sidecar {key!r} has shape {tuple(tensor.shape)}, "
+            f"the drafter needs {want_shape}"
+        )
+    if tensor.dtype not in _E27_EMBED_FLOAT_DTYPES:
+        raise TypeError(
+            f"E27 DFLASH_SEAM: sidecar {key!r} has dtype {tensor.dtype}; a "
+            "BF16 / FP16 / FP32 embedding is required"
+        )
+    step = max(1, tensor.shape[0] // 4096)
+    sample = tensor[::step].float()
+    if not bool(torch.isfinite(sample).all()) or float(sample.abs().amax()) == 0.0:
+        raise ValueError(
+            f"E27 DFLASH_SEAM: sidecar {key!r} rows are non-finite or all zero"
+        )
+    _e27_log(
+        "embed_sidecar",
+        path=path,
+        file=located[key],
+        key=key,
+        shape=list(tensor.shape),
+        dtype=str(tensor.dtype),
+        sample_abs_mean=round(float(sample.abs().mean()), 6),
+    )
+    return tensor
+
+
 class DFlashQwen3ForCausalLM(Qwen3ForCausalLM):
     model_cls = DFlashQwen3Model
+    # E27 DFLASH_SEAM: explicit, = the SupportsEagleBase default (identical
+    # behaviour). maybe_share_target_embed's silent PP early return needs the
+    # attribute to be absent; with it, a missing embedding raises at load.
+    has_own_embed_tokens = False
 
     def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
         nn.Module.__init__(self)
@@ -884,6 +1073,9 @@
         return result
 
     def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
+        # E27 DFLASH_SEAM: append the embed_tokens sidecar (a no-op unless
+        # DFLASH_SEAM=1 and DFLASH_EMBED_SIDECAR is set).
+        weights = _e27_with_embed_sidecar(weights, self.config)
         model_weights = {}
         includes_draft_id_mapping = False
         includes_embed_tokens = False
@@ -917,13 +1109,18 @@
             orig_to_new_substr["draft_id_to_target_id"] = None
         if not includes_embed_tokens:
             orig_to_new_substr["embed_tokens"] = None
+        # E27 DFLASH_SEAM: record whether the weights carried embed_tokens.
+        _e27_mark_embed_loaded(self, includes_embed_tokens)
         if not self.model.use_aux_hidden_state:
             orig_to_new_substr["fc."] = None
         if not self.model.has_separate_mask_embedding:
             orig_to_new_substr["mask_embedding"] = None
         mapper = WeightsMapper(orig_to_new_substr=orig_to_new_substr)
         loader = AutoWeightsLoader(self)
-        loader.load_weights(model_weights.items(), mapper=mapper)
+        _e27_loaded = loader.load_weights(model_weights.items(), mapper=mapper)
+        # E27 DFLASH_SEAM: every expected parameter received a tensor (the
+        # default loader's strict check is off: this method returns None).
+        _e27_check_loaded(self, _e27_loaded, orig_to_new_substr, model_weights)
         self.model._build_fused_kv_buffers()
 
     def _read_mask_embedding(self) -> torch.Tensor | None:
'''

DFU_PATCH = r'''# E27 DFLASH_SEAM onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5, tree 07c16a79).
# load_dflash_model: raise at load on an uninitialised drafter embed_tokens / lm_head, free the replaced
# draft modules, checks, DFLASH_DRAFT_EMBED_HOST=1 drafter host embed (all DFLASH_SEAM=1).
# pre-patch sha256  f0584417149c0a6fa67e700b2a725e40a41f4546a806b26e6cc64b64c59cfd66
# post-patch sha256 fb76c074fdc5ecc94148a7efdeddaec426e4aaf286e812d6e715237a2dd7b5a3
--- a/vllm/v1/worker/gpu/spec_decode/dflash/utils.py
+++ b/vllm/v1/worker/gpu/spec_decode/dflash/utils.py
@@ -11,6 +11,208 @@
     maybe_share_target_embed,
 )
 from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config
+
+
+def _e27_release_module(owner: nn.Module, mod: nn.Module | None, keep) -> int:
+    """E27 DFLASH_SEAM: free a draft module that a share replaced (the pattern of
+    speculator.py _release_replaced_draft_head). Its parameters sit in a
+    reference cycle (Parameter.weight_loader is a bound method of the module),
+    so dropping the attribute frees nothing until a GC pass: clear the
+    parameter / buffer dicts, collect, return the blocks. Never touches a
+    module in `keep` or one that is still a submodule of `owner`. Returns the
+    bytes released."""
+    import gc
+
+    import torch
+
+    if mod is None or any(mod is k for k in keep if k is not None):
+        return 0
+    if any(m is mod for m in owner.modules()):
+        return 0
+    nbytes = 0
+    on_cuda = False
+    for p in list(mod.parameters()) + list(mod.buffers()):
+        nbytes += p.numel() * p.element_size()
+        on_cuda = on_cuda or p.device.type == "cuda"
+    for sub in mod.modules():
+        sub._parameters.clear()
+        sub._buffers.clear()
+    gc.collect()
+    if on_cuda:
+        torch.accelerator.empty_cache()
+    return nbytes
+
+
+def _e27_finish_draft(
+    draft_model: nn.Module,
+    draft_inner: nn.Module,
+    target_inner: nn.Module,
+    target_lm_head: nn.Module | None,
+    replaced_embed: nn.Module | None,
+    replaced_head: nn.Module | None,
+) -> None:
+    """E27 DFLASH_SEAM (a no-op unless DFLASH_SEAM=1), after the embed / lm_head
+    share:
+      1 refuse (raise at load) a drafter whose input embedding or LM head is
+        neither shared from the target nor loaded: such weights are
+        uninitialised and the drafts silent garbage. "Loaded" = the stock
+        has_own_embed_tokens flag AND, when the DFlash loader recorded it,
+        _e27_embed_loaded (the checkpoint / sidecar really carried the tensor);
+      2 release the draft-owned embed_tokens / lm_head the share replaced
+        (636 MiB per TP2 card each for a 248320 x 2560 BF16 table);
+      3 assert every drafter linear uses UnquantizedLinearMethod (the fused
+        context-KV path, qwen3_dflash.py _build_context_kv_buffers, needs it);
+      4 sha256 of the embedding row mask_token_id (248077, an unused padding
+        row the drafter was trained with) on the TP rank that holds it; with
+        DFLASH_MASK_ROW_SHA=<hex> (dflash_seam.embed_row_sha of the target
+        checkpoint) a mismatch raises;
+      5 DFLASH_DRAFT_EMBED_HOST=1: move an own (not shared) drafter embedding
+        to pinned host memory read through a UVA view (the model_runner.py
+        _host_embed_tables pattern), freeing its VRAM. Drafter only: the
+        target / MTP tables and VLLM_HOST_EMBED_TABLE are not touched.
+    One 'embed' log line."""
+    import json
+    import os
+
+    if os.environ.get("DFLASH_SEAM", "0") != "1":
+        return
+    import torch
+
+    from vllm.lora.layers.base import BaseLayerWithLoRA
+    from vllm.model_executor.models.utils import PPMissingLayer
+
+    target_embed = getattr(target_inner, "embed_tokens", None) or getattr(
+        target_inner, "embedding", None
+    )
+    if isinstance(target_embed, BaseLayerWithLoRA):
+        target_embed = target_embed.base_layer
+    if isinstance(target_embed, PPMissingLayer):
+        target_embed = None
+    embed = getattr(draft_inner, "embed_tokens", None)
+    name = type(draft_model).__name__
+    if embed is None or isinstance(embed, PPMissingLayer):
+        raise RuntimeError(f"E27 DFLASH_SEAM: {name} has no input embedding")
+    loaded = getattr(draft_model, "_e27_embed_loaded", None)
+    if target_embed is not None and embed is target_embed:
+        embed_source = "shared_target"
+    elif getattr(draft_model, "has_own_embed_tokens", False) and loaded is not False:
+        embed_source = "own"
+    else:
+        raise RuntimeError(
+            f"E27 DFLASH_SEAM: {name}.embed_tokens was neither shared from the "
+            f"target nor loaded (has_own_embed_tokens "
+            f"{getattr(draft_model, 'has_own_embed_tokens', None)}, loaded "
+            f"{loaded}): its weights are uninitialised. Under PP > 1 ship "
+            "embed_tokens.weight with the drafter (or set DFLASH_EMBED_SIDECAR)."
+        )
+    weight = getattr(embed, "weight", None)
+    if (
+        weight is None
+        or weight.numel() == 0
+        or weight.device.type == "meta"
+        or weight.dim() != 2
+    ):
+        raise RuntimeError(f"E27 DFLASH_SEAM: {name}.embed_tokens has no weight")
+    sample = weight[:: max(1, weight.shape[0] // 4096)].float()
+    if not bool(torch.isfinite(sample).all()) or float(sample.abs().amax()) == 0.0:
+        raise RuntimeError(
+            f"E27 DFLASH_SEAM: {name}.embed_tokens rows are non-finite or all "
+            f"zero (source {embed_source})"
+        )
+    lm_head = getattr(draft_model, "lm_head", None)
+    if lm_head is not None and target_lm_head is not None and lm_head is target_lm_head:
+        head_source = "shared_target"
+    elif lm_head is not None and (
+        getattr(draft_model, "has_own_lm_head", False)
+        or getattr(getattr(draft_model, "config", None), "has_own_lm_head", False)
+    ):
+        head_source = "own"
+    else:
+        raise RuntimeError(
+            f"E27 DFLASH_SEAM: {name}.lm_head was neither shared from the target "
+            "nor loaded: its weights are uninitialised"
+        )
+    quantized = sorted(
+        f"{n}:{type(m.quant_method).__name__}"
+        for n, m in draft_inner.named_modules()
+        if "Linear" in type(m).__name__
+        and getattr(m, "quant_method", None) is not None
+        and type(m.quant_method).__name__ != "UnquantizedLinearMethod"
+    )
+    if quantized:
+        raise RuntimeError(
+            f"E27 DFLASH_SEAM: {name} linears must use UnquantizedLinearMethod "
+            f"(fused context-KV path): {quantized[:8]}"
+        )
+    import hashlib
+
+    mask_id = getattr(draft_inner, "mask_token_id", None)
+    mask_row = {"id": mask_id, "on_this_rank": False}
+    shard = getattr(embed, "shard_indices", None)
+    start = int(getattr(shard, "org_vocab_start_index", 0) or 0)
+    end = int(getattr(shard, "org_vocab_end_index", 0) or weight.shape[0])
+    if mask_id is not None and start <= int(mask_id) < end:
+        row = weight[int(mask_id) - start].detach().to("cpu").contiguous()
+        mask_row.update(
+            on_this_rank=True,
+            sha256=hashlib.sha256(row.view(torch.uint8).numpy().tobytes()).hexdigest(),
+            norm=round(float(row.float().norm()), 6),
+        )
+        want = os.environ.get("DFLASH_MASK_ROW_SHA", "").strip().lower()
+        if want and mask_row["sha256"] != want:
+            raise RuntimeError(
+                f"E27 DFLASH_SEAM: embedding row {mask_id} (mask_token_id) sha "
+                f"{mask_row['sha256']} != DFLASH_MASK_ROW_SHA {want}"
+            )
+    keep = (embed, lm_head, target_embed, target_lm_head)
+    released = {
+        "embed_tokens": _e27_release_module(draft_model, replaced_embed, keep),
+        "lm_head": _e27_release_module(draft_model, replaced_head, keep),
+    }
+    host = os.environ.get("DFLASH_DRAFT_EMBED_HOST", "0") == "1"
+    host_bytes = 0
+    if host and embed_source == "own":
+        if weight.device.type != "cuda":
+            raise RuntimeError(
+                f"E27 DFLASH_DRAFT_EMBED_HOST: {name}.embed_tokens is on "
+                f"{weight.device}"
+            )
+        if type(embed.quant_method).__name__ != "UnquantizedEmbeddingMethod":
+            raise ValueError(
+                "E27 DFLASH_DRAFT_EMBED_HOST: needs an unquantized embedding"
+            )
+        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
+
+        host_bytes = weight.numel() * weight.element_size()
+        cpu_data = weight.detach().to(device="cpu").pin_memory()
+        embed._host_embed_weight = cpu_data
+        weight.data = get_accelerator_view_from_cpu_tensor(cpu_data)
+        weight._vllm_is_uva_offloaded = True
+        embed.use_fused_embedding = False
+        torch.accelerator.empty_cache()
+    print(
+        "E27_DFLASH "
+        + json.dumps(
+            {
+                "event": "embed",
+                "pid": os.getpid(),
+                "draft": name,
+                "embed_source": embed_source,
+                "head_source": head_source,
+                "shape": list(weight.shape),
+                "dtype": str(weight.dtype),
+                "device": str(weight.device),
+                "sample_abs_mean": round(float(sample.abs().mean()), 6),
+                "released_mib": {k: round(v / 2**20, 1) for k, v in released.items()},
+                "released_bytes": released,
+                "host": bool(host and embed_source == "own"),
+                "host_mib": round(host_bytes / 2**20, 1),
+                "mask_row": mask_row,
+            },
+            sort_keys=True,
+        ),
+        flush=True,
+    )
 
 
 def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
@@ -56,6 +258,8 @@
     # no .model of its own.
     target_inner = getattr(target_language_model, "model", target_language_model)
     draft_inner = dflash_model.model
+    # E27 DFLASH_SEAM: the draft's own embedding, released below if replaced.
+    _e27_draft_embed = getattr(draft_inner, "embed_tokens", None)
 
     maybe_share_target_embed(dflash_model, draft_inner, target_inner)
 
@@ -72,4 +276,14 @@
             del dflash_model.lm_head
         dflash_model.lm_head = target_lm_head
 
+    # E27 DFLASH_SEAM: refuse an uninitialised drafter embedding / LM head, free
+    # the replaced draft modules, optional host embed (no-op unless DFLASH_SEAM=1).
+    _e27_finish_draft(
+        dflash_model,
+        draft_inner,
+        target_inner,
+        target_lm_head,
+        _e27_draft_embed,
+        draft_lm_head,
+    )
     return dflash_model
'''

QSA_PATCH = r'''# E27 DFLASH_SEAM onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5, tree 07c16a79).
# QSAKeyStateCache.get_kv_cache_spec: ring widening for K >= 5 (vLLM PR #54912), DFLASH_QSA_RING_WIDEN=1.
# pre-patch sha256  43f323bb347e2152e47acb36a99d0682eaebc5cdafe9b5ba168b5c4b4e9acf13
# post-patch sha256 07b0b61f9947d36ab2b3f87958830aaf591e899a93ab61f67c4debc50c442bde
--- a/vllm/models/qwen4_exp/common/qsa_cache.py
+++ b/vllm/models/qwen4_exp/common/qsa_cache.py
@@ -886,6 +886,67 @@
         return QSAStateBackend
 
 
+# E27 DFLASH_QSA_RING_WIDEN: the ring widening of vLLM PR #54912 (open upstream),
+# own text, env-gated. The QSA raw-key ring holds the open group's committed
+# keys plus every row a speculative step stores before acceptance is known
+# (span = ratio + K), rounded up to whole groups; it joins the scheduler-block
+# LCM, so it must divide the attention block size. For K >= 5 at ratio 4 the
+# minimal ring (12) does not divide a power-of-two block. A wider ring is
+# slack (every row a step can write still fits), so the next multiple of the
+# ratio that divides the block is used, bounded by QSA_RING_MAX_WIDENING times
+# the minimal ring (every request holds one ring block for its lifetime).
+QSA_RING_MAX_WIDENING = 2
+
+
+def qsa_ring_widen_enabled() -> bool:
+    import os
+
+    return os.environ.get("DFLASH_QSA_RING_WIDEN", "0") == "1"
+
+
+def qsa_ring_capacity(
+    compress_ratio: int, num_speculative_tokens: int, block_size: int
+) -> int:
+    """Rows of the QSA raw-key ring for this attention block size: the minimal
+    whole-group ring when it divides block_size (then identical to the
+    original formula), else the smallest wider multiple of compress_ratio that
+    divides it, up to QSA_RING_MAX_WIDENING x the minimal ring; else a
+    ValueError that names the sizes."""
+    span = compress_ratio + num_speculative_tokens
+    minimal = compress_ratio * cdiv(span, compress_ratio)
+    limit = min(block_size, QSA_RING_MAX_WIDENING * minimal)
+    capacity = minimal
+    while capacity <= limit:
+        if block_size % capacity == 0:
+            if capacity != minimal:
+                import json
+                import os
+
+                print(
+                    "E27_DFLASH "
+                    + json.dumps(
+                        {
+                            "event": "qsa_ring",
+                            "pid": os.getpid(),
+                            "minimal": minimal,
+                            "capacity": capacity,
+                            "block_size": block_size,
+                            "num_speculative_tokens": num_speculative_tokens,
+                        },
+                        sort_keys=True,
+                    ),
+                    flush=True,
+                )
+            return capacity
+        capacity += compress_ratio
+    raise ValueError(
+        f"QSA ring needs {minimal} rows for num_speculative_tokens="
+        f"{num_speculative_tokens} (compress ratio {compress_ratio}), and no "
+        f"multiple of {compress_ratio} up to {limit} divides the attention "
+        f"block size {block_size}; choose a depth whose ring divides it"
+    )
+
+
 class QSAKeyStateCache(_QSAStateCache):
     """Raw BF16 key followed by the int64 ring tail (pos_t, pos_h, pos_w, tag).
 
@@ -959,6 +1020,14 @@
         # overwrite a committed key the next step needs to close the group.
         span = self.compress_ratio + vllm_config.num_speculative_tokens
         capacity = self.compress_ratio * cdiv(span, self.compress_ratio)
+        if qsa_ring_widen_enabled():
+            # E27 DFLASH_QSA_RING_WIDEN=1 (vLLM PR #54912): the same capacity
+            # whenever the minimal ring divides the block, else the widening.
+            capacity = qsa_ring_capacity(
+                self.compress_ratio,
+                vllm_config.num_speculative_tokens,
+                self.cache_config.block_size,
+            )
         assert self.cache_config.block_size % capacity == 0, (
             f"QSA ring capacity {capacity} must divide the attention block "
             f"size {self.cache_config.block_size}"
'''

ATT_PATCH = r'''# E27 DFLASH_SEAM onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5, tree 07c16a79).
# Attention.get_kv_cache_spec: DFLASH_DRAFT_SW_BLOCK=<n> drafter sliding-window block (divisor of the block).
# pre-patch sha256  bde77677fd0e95ea09c6a580dfaf09b57741e8183e00cb106e2d1cc64d8e03f2
# post-patch sha256 5817d764b870409ef0ef5b0c9e619f532a6410cece82833bd464dcea69181a5c
--- a/vllm/model_executor/layers/attention/attention.py
+++ b/vllm/model_executor/layers/attention/attention.py
@@ -130,6 +130,47 @@
     return max(fitting) if fitting else smallest
 
 
+# E27 DFLASH_DRAFT_SW_BLOCK=<n>: a smaller KV block for the drafter's sliding-
+# window layers (is_draft_layer, set by DraftModelSpeculator.load_model). n must
+# divide the attention block size, so the scheduler-block LCM is unchanged.
+# Unset: the original block choice.
+_E27_DRAFT_SW_LOGGED: set[str] = set()
+
+
+def _e27_draft_sw_block(layer: nn.Module, block_size: int) -> int:
+    import os
+
+    raw = os.environ.get("DFLASH_DRAFT_SW_BLOCK", "").strip()
+    if not raw or not getattr(layer, "is_draft_layer", False):
+        return block_size
+    n = int(raw)
+    if n <= 0 or block_size % n != 0:
+        raise ValueError(
+            f"DFLASH_DRAFT_SW_BLOCK={raw}: must be a positive divisor of the "
+            f"attention block size {block_size}"
+        )
+    name = str(getattr(layer, "layer_name", "?"))
+    if name not in _E27_DRAFT_SW_LOGGED:
+        import json
+
+        _E27_DRAFT_SW_LOGGED.add(name)
+        print(
+            "E27_DFLASH "
+            + json.dumps(
+                {
+                    "event": "draft_sw_block",
+                    "pid": os.getpid(),
+                    "layer": name,
+                    "block_size": block_size,
+                    "draft_block": n,
+                },
+                sort_keys=True,
+            ),
+            flush=True,
+        )
+    return n
+
+
 def set_default_quant_scales(layer: nn.Module, register_buffer: bool = False) -> None:
     """Sets default quantization scales for the layer."""
     if register_buffer:
@@ -634,6 +675,8 @@
             # B/token MLA target: 1728 vs 1536 gives LCM 13824). Backends that
             # cannot run the primary block start from their smallest block and
             # ``unify`` scales it up by an integer ratio.
+            # E27 DFLASH_DRAFT_SW_BLOCK: the drafter's SW block (no-op unless set).
+            block_size = _e27_draft_sw_block(self, block_size)
             shared_page = vllm_config.cache_config.skip_page_size_padded
             # The backend owns its packing
             kv_cache_spec = self.attn_backend.customize_spec(
'''

SPC_PATCH = r'''# E27 DFLASH_SEAM onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5, tree 07c16a79).
# DFlashSpeculator: refresh the kernel block tables on the in-graph draft metadata path (vLLM #63587,
# PR #64059), DFLASH_NO_INGRAPH_META=1 fallback (DFLASH_SEAM=1).
# pre-patch sha256  b731b09c92b93063d89851a997e1de70e51d512bc40b2a4dd68274a3a999e187
# post-patch sha256 85f57f0ee9bc7018f052f59d0610a91c78d35cd018ef711c6f7ecbef298d47ea
--- a/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
+++ b/vllm/v1/worker/gpu/spec_decode/dflash/speculator.py
@@ -34,6 +34,9 @@
 
 class DFlashSpeculator(DraftModelSpeculator):
     _speculator_name = "DFlash"  # For logging, so we can share methods with subclasses
+    # E27 DFLASH_SEAM: refresh the kernel block tables on the in-graph metadata
+    # path (vLLM #63587 / PR #64059); set by _e27_configure_ingraph_meta.
+    _e27_kernel_map_refresh = False
 
     def __init__(self, vllm_config: VllmConfig, device: torch.device):
         parallel_config = vllm_config.parallel_config
@@ -252,6 +255,59 @@
                         layer_names, self.model.get_draft_attn_causal()
                     )
                 }
+        # E27 DFLASH_SEAM: in-graph draft decode metadata (no-op unless set).
+        self._e27_configure_ingraph_meta()
+
+    def _e27_configure_ingraph_meta(self) -> None:
+        """E27 DFLASH_SEAM (vLLM issue #63587, fix = PR #64059): with FULL
+        graphs and in-graph draft decode-metadata updates, propose() skips
+        _build_uniform_attn_metadata, the only call that maps the current block
+        tables into the attention groups' persistent kernel block tables; with a
+        kernel block smaller than the manager block (4096 here) the replayed
+        graph reads stale draft KV (acceptance about 0). DFLASH_SEAM=1: propose()
+        refreshes the kernel block tables on that path. DFLASH_NO_INGRAPH_META=1
+        (A/B fallback, the issue's workaround): this speculator never updates
+        the metadata in the graph (it is rebuilt eagerly every step). Runs
+        before graph capture. One 'ingraph_meta' line. A no-op unless
+        DFLASH_SEAM=1."""
+        import json
+        import os
+
+        if os.environ.get("DFLASH_SEAM", "0") != "1":
+            return
+        native = bool(self.supports_decode_metadata_update)
+        forced_off = os.environ.get("DFLASH_NO_INGRAPH_META", "0") == "1"
+        if forced_off:
+            self.supports_decode_metadata_update = False
+        self._e27_kernel_map_refresh = bool(self.supports_decode_metadata_update)
+        groups = [
+            {
+                "gid": int(group.kv_cache_group_id),
+                "manager_block": int(group.kv_cache_spec.block_size),
+                "kernel_block": group.kernel_block_size,
+            }
+            for attn_groups in self.attn_groups
+            for group in attn_groups
+        ]
+        print(
+            "E27_DFLASH "
+            + json.dumps(
+                {
+                    "event": "ingraph_meta",
+                    "pid": os.getpid(),
+                    "speculator": self._speculator_name,
+                    "native_supports_decode_metadata_update": native,
+                    "supports_decode_metadata_update": bool(
+                        self.supports_decode_metadata_update
+                    ),
+                    "forced_off": forced_off,
+                    "kernel_map_refresh": self._e27_kernel_map_refresh,
+                    "groups": groups,
+                },
+                sort_keys=True,
+            ),
+            flush=True,
+        )
 
     @torch.inference_mode()
     def _run_model(
@@ -505,6 +561,15 @@
                 step=self.num_query_per_req,
                 causal=self._group_causal,
             )
+        elif self._e27_kernel_map_refresh:
+            # E27 DFLASH_SEAM (vLLM #63587, PR #64059): the graph's metadata
+            # reads the kernel block tables, which only the skipped build maps
+            # from the current block tables.
+            for groups in self.attn_groups:
+                for group in groups:
+                    group.map_to_kernel_block_table(
+                        self.block_tables.input_block_tables[group.kv_cache_group_id]
+                    )
 
         # DFlash processes all speculative tokens in one forward pass,
         # so the real token count is num_query_tokens.
'''

DF2_PATCH = r'''# E27 DFLASH_SEAM onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5, tree 07c16a79).
# _grouped_conv: DFLASH_CONV_IMPL=torch A/B (vLLM #58250 revert of #55960) with the SGLang #41736
# NaN-safe tap masking (DFLASH_SEAM=1).
# pre-patch sha256  11ac6ba8c6db702375970e68cd042b98783a5c43b87f9bfa27e88463316bf499
# post-patch sha256 4039da670a5f6a91ef5e5931465a652d4145207c90b87ff44fdb7b1f61d04402
--- a/vllm/model_executor/models/qwen3_dflash2.py
+++ b/vllm/model_executor/models/qwen3_dflash2.py
@@ -141,6 +141,60 @@
     return torch.ops.vllm.dflash2_grouped_conv(x, delta, base, block_size, group_size)
 
 
+# E27 DFLASH_CONV_IMPL=torch (with DFLASH_SEAM=1; A/B, default = the fused
+# Triton kernel): run the grouped convolution through the torch path on CUDA
+# too. vLLM #58250 (open) reverts the fused kernel (#55960, which Merlin
+# carries) because the DFlash acceptance-length nightly fell from 3.195 to
+# 3.060 after it landed. The torch path masks the cross-request taps with
+# torch.where (SGLang #41736), so NaN / Inf in one request's rows (e.g. graph
+# padding) cannot reach the next request through `0 * NaN`; for finite inputs
+# it is the pre-#55960 torch path. Read once at import.
+def _e27_conv_torch_enabled() -> bool:
+    import json
+    import os
+
+    if os.environ.get("DFLASH_SEAM", "0") != "1":
+        return False
+    impl = os.environ.get("DFLASH_CONV_IMPL", "fused").strip() or "fused"
+    if impl not in ("fused", "torch"):
+        raise ValueError(f"DFLASH_CONV_IMPL={impl!r}: expected fused or torch")
+    print(
+        "E27_DFLASH "
+        + json.dumps(
+            {"event": "conv_impl", "pid": os.getpid(), "impl": impl}, sort_keys=True
+        ),
+        flush=True,
+    )
+    return impl == "torch"
+
+
+_E27_CONV_TORCH = _e27_conv_torch_enabled()
+
+
+def _e27_grouped_conv_torch(
+    hidden_states: torch.Tensor,
+    delta: torch.Tensor,
+    base: torch.Tensor,
+    block_size: int,
+    num_groups: int,
+    group_size: int,
+    taps: int,
+) -> torch.Tensor:
+    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
+    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
+    output = coefficients[:, 0] * blocks
+    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
+    if block_size & (block_size - 1) == 0:
+        position = position & (block_size - 1)
+    else:
+        position = position % block_size
+    for tap in range(1, taps):
+        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
+        shifted = torch.where((position >= tap).view(-1, 1, 1), shifted, 0)
+        output = output + coefficients[:, tap] * shifted
+    return output.flatten(-2)
+
+
 def _grouped_conv(
     hidden_states: torch.Tensor,
     delta: torch.Tensor,
@@ -150,6 +204,11 @@
     group_size: int,
     taps: int,
 ) -> torch.Tensor:
+    if _E27_CONV_TORCH:
+        # E27 DFLASH_CONV_IMPL=torch: the torch path on every device.
+        return _e27_grouped_conv_torch(
+            hidden_states, delta, base, block_size, num_groups, group_size, taps
+        )
     if hidden_states.is_cuda:
         return dflash2_grouped_conv(hidden_states, delta, base, block_size, group_size)
 
'''
GRD_PATCH = r'''# E27 DFLASH_SEAM onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5, tree 07c16a79).
# e2_guards.assert_resolved_cudagraph_modes: admit the DFlash / DFlash2 drafter's query_cudagraph_manager (E27 NBDF-K100
# GPU run 2026-11-01: every D arm LAUNCH_FAILED with '<unrecognised speculator DFlash2Speculator>'). The guard stays
# unconditional; it now expects FULL_DECODE_ONLY for that manager (a NONE / eager drafter still raises).
# pre-patch sha256  aa25244647523c74b43d6fe93d61de965bbbe810c92a20f7d5bd4c2dd516e5c3
# post-patch sha256 d50d754e8479cf77b577f283ef6456a8a8db550b179ca9aa6a4c26d3126bde11
--- a/vllm/v1/e2_guards.py
+++ b/vllm/v1/e2_guards.py
@@ -227,11 +227,15 @@
 EXPECTED_RESOLVED_MODES = {
     "target": REQUIRED_CUDAGRAPH_MODE_NAME,
     "draft.prefill_cudagraph_manager": REQUIRED_CUDAGRAPH_MODE_NAME,
     "draft.decode_cudagraph_manager": "FULL_DECODE_ONLY",
     "draft.cudagraph_manager": "FULL_DECODE_ONLY",
+    # E27 DFLASH_SEAM: the DFlash / DFlash2 drafter's graph manager (query_cudagraph_manager)
+    # is FULL_DECODE_ONLY when the decode mode is FULL, else NONE (dflash/speculator.py:126-150).
+    "draft.query_cudagraph_manager": "FULL_DECODE_ONLY",
 }
 _DRAFT_MANAGER_ATTRS = (
     "prefill_cudagraph_manager",
     "decode_cudagraph_manager",
     "cudagraph_manager",
+    "query_cudagraph_manager",
 )
'''
PATCHES = {CFG_REL: (CFG_PRE, CFG_POST, CFG_MARKER, CFG_PATCH), MDL_REL: (MDL_PRE, MDL_POST, MDL_MARKER, MDL_PATCH),
           DFL_REL: (DFL_PRE, DFL_POST, DFL_MARKER, DFL_PATCH), DFU_REL: (DFU_PRE, DFU_POST, DFU_MARKER, DFU_PATCH),
           QSA_REL: (QSA_PRE, QSA_POST, QSA_MARKER, QSA_PATCH), ATT_REL: (ATT_PRE, ATT_POST, ATT_MARKER, ATT_PATCH),
           SPC_REL: (SPC_PRE, SPC_POST, SPC_MARKER, SPC_PATCH), DF2_REL: (DF2_PRE, DF2_POST, DF2_MARKER, DF2_PATCH),
           GRD_REL: (GRD_PRE, GRD_POST, GRD_MARKER, GRD_PATCH)}
CORE_RELS = (CFG_REL, MDL_REL, DFL_REL, DFU_REL, SPC_REL, DF2_REL, GRD_REL)
RING_RELS = (QSA_REL,)
DRAFT_BLOCK_RELS = (ATT_REL,)
if set(PATCHES) & set(NB4_FILES):
    raise RuntimeError("E27 dflash_seam overlaps the NB4 Q-pack / Q-defer files")

APPLIER_SRC = r'''
def apply_unified(text, patch):
    """Strict unified-diff applier (no fuzz, no offset): every context / removed line must match at its stated line,
    else ValueError. Lines before the first hunk (comments, ---/+++ headers) are ignored. (= NB1 / NB5 / NB4 P5.)"""
    src = text.splitlines(True)
    hunks, cur = [], None
    for ln in patch.splitlines(True):
        if ln.startswith("@@"):
            m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", ln)
            if not m:
                raise ValueError("bad hunk header: " + ln[:80])
            cur = [int(m.group(1)), int(m.group(2) or 1), []]
            hunks.append(cur)
        elif cur is not None and ln[:1] in (" ", "-", "+"):
            cur[2].append(ln)
        elif cur is not None and ln.startswith("\\"):
            raise ValueError("'No newline at end of file' hunks are not supported")
    if not hunks:
        raise ValueError("patch has no hunks")
    out, pos = [], 0
    for start, n_old, body in hunks:
        i = start - 1
        if i < pos:
            raise ValueError(f"overlapping hunk at line {start}")
        out += src[pos:i]
        old_seen = 0
        for ln in body:
            tag, s = ln[0], ln[1:]
            if tag in " -":
                if i >= len(src) or src[i] != s:
                    raise ValueError(f"context mismatch at line {i + 1}: {s[:60]!r}")
                i += 1
                old_seen += 1
            if tag in " +":
                out.append(s)
        if old_seen != n_old:
            raise ValueError(f"hunk at line {start}: {old_seen} old lines, header says {n_old}")
        pos = i
    return "".join(out + src[pos:])


def reverse_unified(patch):
    """The reverse patch: hunk headers swapped (-a,b +c,d -> -c,d +a,b), '+' and '-' lines swapped."""
    out = []
    started = False
    for ln in patch.splitlines(True):
        if ln.startswith("@@"):
            started = True
            m = re.match(r"@@ -(\d+(?:,\d+)?) \+(\d+(?:,\d+)?) @@(.*)", ln, re.S)
            if not m:
                raise ValueError("bad hunk header: " + ln[:80])
            out.append(f"@@ -{m.group(2)} +{m.group(1)} @@{m.group(3)}")
        elif started and ln[:1] == "+":
            out.append("-" + ln[1:])
        elif started and ln[:1] == "-":
            out.append("+" + ln[1:])
        else:
            out.append(ln)
    return "".join(out)
'''
exec(APPLIER_SRC)  # noqa: S102 - defines apply_unified / reverse_unified here, the notebook embeds the same text


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_text(t):
    return sha256_bytes(t.encode("utf-8"))


PATCH_SHA = {rel: sha256_text(p[3]) for rel, p in PATCHES.items()}


def _rels(ring, draft_block=True):
    return CORE_RELS + (RING_RELS if ring else ()) + (DRAFT_BLOCK_RELS if draft_block else ())


def _py_compile(py, files):
    cp = subprocess.run([py, "-c", "import py_compile, sys\nfor f in sys.argv[1:]: py_compile.compile(f, doraise=True)"]
                        + [str(f) for f in files], capture_output=True, text=True, timeout=300)
    return {"rc": cp.returncode, "err": cp.stderr[-600:]}


def _drop_pyc(f):
    d = f.parent / "__pycache__"
    if d.is_dir():
        for pyc in d.glob(f.stem + ".*.pyc"):
            pyc.unlink()


def _state(rel, h, data):
    pre, post, marker, _ = PATCHES[rel]
    if h == post and marker.encode() in data:
        return "applied"
    if h == pre:
        return "not_applied"
    return "foreign"


def verify_dflash_patch(tree, ring=True, draft_block=True):
    """Read-only: 'applied' (every file post-sha + marker), 'not_applied' (every file pre-sha), else 'foreign' / 'mixed'.
    ring=False leaves qsa_cache.py, draft_block=False leaves attention.py out of the verdict (the four core files are
    always checked)."""
    tree = Path(tree)
    files = {}
    for rel in _rels(ring, draft_block):
        data = (tree / rel).read_bytes()
        h = sha256_bytes(data)
        files[rel] = {"sha": h, "state": _state(rel, h, data)}
    states = {v["state"] for v in files.values()}
    state = states.pop() if len(states) == 1 else ("foreign" if "foreign" in states else "mixed")
    return {"state": state, "files": files, "ring": bool(ring), "draft_block": bool(draft_block)}


TREE_STATES = ("ORIGINAL", "FULLY_PATCHED", "INVALID_MIXED")


def tree_state(tree, ring=True, draft_block=True):
    """ORIGINAL (every E27 file at Merlin's pre-sha), FULLY_PATCHED (every file post-sha + marker) or INVALID_MIXED
    (anything else: abort). Read-only."""
    s = verify_dflash_patch(tree, ring, draft_block)["state"]
    return {"not_applied": "ORIGINAL", "applied": "FULLY_PATCHED"}.get(s, "INVALID_MIXED")


def combined_tree_state(tree, nb4=None, ring=True, draft_block=True):
    """E27 and (when the nb4_ppack module is passed) E25 NB4 Q-pack / Q-defer states of one tree, each ORIGINAL /
    FULLY_PATCHED / INVALID_MIXED; valid = neither is INVALID_MIXED (the four combinations of the two valid states are
    all legal, the file sets are disjoint)."""
    out = {"e27": tree_state(tree, ring, draft_block)}
    if nb4 is not None:
        s = nb4.verify_ppack_patch(tree)["state"]
        out["nb4"] = {"not_applied": "ORIGINAL", "applied": "FULLY_PATCHED"}.get(s, "INVALID_MIXED")
    out["valid"] = "INVALID_MIXED" not in out.values()
    return out


def _write_atomic(f, data):
    tmp = f.with_name(f.name + ".e27tmp")
    tmp.write_bytes(data)
    os.replace(tmp, f)


def apply_dflash_patch(tree, py=None, ring=True, draft_block=True):
    """Patch the installed tree (config.py, nvidia/model.py, qwen3_dflash.py, dflash/utils.py; with ring=True
    qsa_cache.py; with draft_block=True attention.py). Every file must be at Merlin ad42e37's pre-sha (or all already applied), else refuse. All new texts
    are built and sha-checked before anything is written; then the post-sha, the marker and py_compile (with `py`, the
    Merlin venv python) are verified. Any miss raises."""
    tree = Path(tree)
    py = py or sys.executable
    rels = _rels(ring, draft_block)
    v = verify_dflash_patch(tree, ring, draft_block)
    rec = {"tree": str(tree), "before": v, "patch_sha": {r: PATCH_SHA[r] for r in rels}}
    if v["state"] == "applied":
        rec["applied"] = "already_applied"
    elif v["state"] != "not_applied":
        raise RuntimeError("PATCH_DFLASH_PRE_SHA_MISMATCH: " + json.dumps(v)[:900])
    else:
        new = {}
        for rel in rels:
            pre, post, marker, patch = PATCHES[rel]
            raw = (tree / rel).read_bytes()
            out = apply_unified(raw.decode("utf-8"), patch).encode("utf-8")  # noqa: F821 - from APPLIER_SRC
            if sha256_bytes(out) != post:
                raise RuntimeError(f"PATCH_DFLASH_POST_SHA_MISMATCH {rel}: {sha256_bytes(out)}")
            new[rel] = out
        for rel, out in new.items():
            _write_atomic(tree / rel, out)
        rec["applied"] = "applied"
    for rel in rels:
        _drop_pyc(tree / rel)
    rec["after"] = verify_dflash_patch(tree, ring, draft_block)
    rec["py_compile"] = _py_compile(py, [tree / rel for rel in rels])
    if rec["after"]["state"] != "applied" or rec["py_compile"]["rc"]:
        raise RuntimeError("PATCH_DFLASH_VERIFY_FAILED: " + json.dumps(
            {k: rec.get(k) for k in ("after", "py_compile")})[:900])
    rec["ok"] = True
    return rec


def revert_dflash_patch(tree, py=None, ring=True, draft_block=True):
    """Back to the exact Merlin files (reverse patches, sha-checked == pre). Refuses unless every file is applied
    (post-sha) or every file already original."""
    tree = Path(tree)
    py = py or sys.executable
    rels = _rels(ring, draft_block)
    v = verify_dflash_patch(tree, ring, draft_block)
    rec = {"tree": str(tree), "before": v}
    if v["state"] == "not_applied":
        rec["reverted"] = "not_applied"
    elif v["state"] != "applied":
        raise RuntimeError("REVERT_DFLASH_SHA_MISMATCH: " + json.dumps(v)[:900])
    else:
        old = {}
        for rel in rels:
            pre, post, marker, patch = PATCHES[rel]
            raw = (tree / rel).read_bytes()
            out = apply_unified(raw.decode("utf-8"), reverse_unified(patch)).encode("utf-8")  # noqa: F821
            if sha256_bytes(out) != pre:
                raise RuntimeError(f"REVERT_DFLASH_RESULT_MISMATCH {rel}: {sha256_bytes(out)}")
            old[rel] = out
        for rel, out in old.items():
            _write_atomic(tree / rel, out)
        rec["reverted"] = "reverted"
    for rel in rels:
        _drop_pyc(tree / rel)
    rec["after"] = verify_dflash_patch(tree, ring, draft_block)
    rec["py_compile"] = _py_compile(py, [tree / rel for rel in rels])
    if rec["after"]["state"] != "not_applied" or rec["py_compile"]["rc"]:
        raise RuntimeError("REVERT_DFLASH_VERIFY_FAILED: " + json.dumps(rec)[:900])
    rec["ok"] = True
    return rec


def server_env(embed_sidecar=None, embed_key=None, ring=False, tap_mode=None, mask_row_sha=None, embed_host=False,
               draft_sw_block=None, no_ingraph_meta=False, conv_impl=None):
    """Env added to the server (engine + workers inherit it) for a DFlash arm. embed_sidecar: fallback only, when the
    drafter is not repacked with embed_tokens.weight (the target checkpoint dir or shard). ring: K >= 5.
    tap_mode: None = combine (the default), else one of TAP_MODES (A/B). mask_row_sha: embed_row_sha() of the target
    checkpoint (asserted at load). embed_host: drafter embedding in pinned host memory. draft_sw_block: e.g. 1024."""
    env = {ENV_SEAM: "1"}
    if mask_row_sha:
        env[ENV_MASK_SHA] = str(mask_row_sha).lower()
    if embed_host:
        env[ENV_EMBED_HOST] = "1"
    if draft_sw_block:
        env[ENV_SW_BLOCK] = str(int(draft_sw_block))
    if no_ingraph_meta:
        env[ENV_NO_INGRAPH] = "1"
    if conv_impl is not None:
        if conv_impl not in ("fused", "torch"):
            raise ValueError(f"conv_impl {conv_impl!r} not in ('fused', 'torch')")
        env[ENV_CONV_IMPL] = conv_impl
    if tap_mode is not None:
        if tap_mode not in TAP_MODES:
            raise ValueError(f"tap_mode {tap_mode!r} not in {TAP_MODES}")
        env[ENV_TAP_MODE] = tap_mode
    if embed_sidecar:
        env[ENV_EMBED] = str(embed_sidecar)
    if embed_key:
        env[ENV_EMBED_KEY] = str(embed_key)
    if ring:
        env[ENV_RING] = "1"
    return env


def embed_row_sha(path, row=MASK_TOKEN_ID, key=None):
    """sha256 (hex) of one BF16 row of the target's input embedding, read with safetensors get_slice (one row, no full
    load). path: a safetensors file, or a model dir (model.safetensors.index.json, else every *.safetensors). key:
    the tensor name, else the single '*embed_tokens.weight' key without 'mtp'. Returns (sha, key, file, dtype, norm).
    Same bytes as the in-engine 'embed' event's mask_row.sha256 (row bytes in the tensor's own dtype)."""
    import glob

    import torch
    from safetensors import safe_open

    path = str(path)
    located = {}
    if os.path.isdir(path):
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.isfile(index):
            with open(index, encoding="utf-8") as f:
                located = {k: os.path.join(path, v) for k, v in json.load(f)["weight_map"].items()}
        else:
            for file in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
                with safe_open(file, framework="pt", device="cpu") as f:
                    for k in f.keys():
                        located.setdefault(k, file)
    else:
        with safe_open(path, framework="pt", device="cpu") as f:
            located = dict.fromkeys(f.keys(), path)
    if key is None:
        cands = sorted(k for k in located if k.endswith("embed_tokens.weight") and "mtp" not in k)
        if len(cands) != 1:
            raise KeyError(f"{len(cands)} embed_tokens candidates in {path!r}: {cands[:8]}")
        key = cands[0]
    with safe_open(located[key], framework="pt", device="cpu") as f:
        r = f.get_slice(key)[row:row + 1][0].contiguous()
    sha = hashlib.sha256(r.view(torch.uint8).numpy().tobytes()).hexdigest()
    return sha, key, located[key], str(r.dtype), round(float(r.float().norm()), 6)


_EV_RE = re.compile(r"E27_DFLASH (\{.*\})\s*$")


def parse_log(text):
    """Every 'E27_DFLASH {json}' line of a server log -> list of dicts (bad JSON lines are returned as {'bad': line})."""
    out = []
    for ln in text.splitlines():
        m = _EV_RE.search(ln)
        if not m:
            continue
        try:
            out.append(json.loads(m.group(1)))
        except ValueError:
            out.append({"bad": ln[-200:]})
    return out


def readback_label(log_text, pp_size=2, tp_size=2, ids=AUX_IDS, embed_source=None, ring_k=None, sidecar=False,
                   tap_mode="combine", mask_row_sha=None, embed_host=False, no_ingraph_meta=False, conv_impl="fused"):
    """Read-back gate of a DFlash arm. ENGAGED when: pp_size * tp_size 'aux_layers' lines (distinct pids) carry `ids`
    and `tap_mode`; every PP stage's tp_size workers report the same local ids and the union of the stages' local ids
    is `ids`; exactly tp_size 'embed' lines (the drafter lives on the last PP rank) with distinct pids, embed_source ==
    embed_source when given (else 'own' for PP > 1, 'shared_target' for PP 1), head_source 'shared_target';
    sidecar=True (fallback b): tp_size 'embed_sidecar' lines; ring_k given: >= 1 'qsa_ring' line with that K; exactly
    one embed line holds the mask row (mask_row.on_this_rank), with sha == mask_row_sha when given; embed_host: every
    embed line has host True; a 'draft_load_check' line per drafter worker (tp_size) unless skipped; tp_size
    'ingraph_meta' lines: kernel_map_refresh == the final supports_decode_metadata_update (the #63587 fix is on
    whenever the graph updates the metadata), and with no_ingraph_meta=True forced_off and the flag False; every
    'conv_impl' line (one per drafter worker; DFlash2 drafters only) reports conv_impl.
    Otherwise NOT_INTEGRATED with the reason. Returns (label, detail)."""
    ev = parse_log(log_text)
    detail = {"events": len(ev)}
    bad = [e for e in ev if "bad" in e]
    aux = [e for e in ev if e.get("event") == "aux_layers"]
    emb = [e for e in ev if e.get("event") == "embed"]
    side = [e for e in ev if e.get("event") == "embed_sidecar"]
    ring = [e for e in ev if e.get("event") == "qsa_ring"]
    n = pp_size * tp_size
    detail.update(aux=len(aux), embed=len(emb), sidecar=len(side), ring=len(ring), bad=len(bad))
    why = []
    if bad:
        why.append(f"{len(bad)} unparsable E27_DFLASH line(s)")
    if len(aux) != n or len({e.get("pid") for e in aux}) != n:
        why.append(f"aux_layers lines {len(aux)} (distinct pids {len({e.get('pid') for e in aux})}), want {n}")
    if any(tuple(e.get("ids") or ()) != tuple(ids) for e in aux):
        why.append(f"aux ids differ from {list(ids)}: {sorted({tuple(e.get('ids') or ()) for e in aux})}")
    if any(e.get("tap_mode") != tap_mode for e in aux):
        why.append(f"tap modes {sorted({str(e.get('tap_mode')) for e in aux})}, want {tap_mode!r}")
    stages = {}
    for e in aux:
        stages.setdefault((e.get("start_layer"), e.get("end_layer")), set()).add(tuple(e.get("local") or ()))
    detail["stages"] = {f"{k[0]}..{k[1]}": sorted(v) for k, v in sorted(stages.items(), key=lambda x: str(x[0]))}
    if len(stages) != pp_size or any(len(v) != 1 for v in stages.values()):
        why.append(f"{len(stages)} PP stages with consistent local ids, want {pp_size}")
    union = sorted({i for v in stages.values() for t in v for i in t})
    if union != sorted(ids):
        why.append(f"union of local tap ids {union} != {sorted(ids)}")
    want_src = embed_source or ("own" if pp_size > 1 else "shared_target")
    if len(emb) != tp_size or len({e.get("pid") for e in emb}) != tp_size:
        why.append(f"embed lines {len(emb)}, want {tp_size} (distinct pids)")
    if any(e.get("embed_source") != want_src or e.get("head_source") != "shared_target" for e in emb):
        why.append(f"embed / head sources {[(e.get('embed_source'), e.get('head_source')) for e in emb]}, "
                   f"want ({want_src!r}, 'shared_target')")
    if sidecar and len(side) != tp_size:
        why.append(f"embed_sidecar lines {len(side)}, want {tp_size}")
    holders = [e for e in emb if isinstance(e.get("mask_row"), dict) and e["mask_row"].get("on_this_rank")]
    detail["mask_row"] = [e["mask_row"] for e in holders]
    if emb and len(holders) != 1:
        why.append(f"{len(holders)} embed lines hold the mask row, want 1")
    if mask_row_sha and any(e["mask_row"].get("sha256") != str(mask_row_sha).lower() for e in holders):
        why.append(f"mask row sha {[e['mask_row'].get('sha256') for e in holders]} != {mask_row_sha}")
    if embed_host and any(not e.get("host") for e in emb):
        why.append("embed_host requested but an embed line has host False")
    igm = [e for e in ev if e.get("event") == "ingraph_meta"]
    detail["ingraph_meta"] = [{k: e.get(k) for k in ("supports_decode_metadata_update", "forced_off",
                                                      "kernel_map_refresh")} for e in igm]
    if len(igm) != tp_size:
        why.append(f"ingraph_meta lines {len(igm)}, want {tp_size}")
    for e in igm:
        sup = bool(e.get("supports_decode_metadata_update"))
        if bool(e.get("kernel_map_refresh")) != sup:
            why.append(f"ingraph_meta: kernel_map_refresh {e.get('kernel_map_refresh')} with in-graph update {sup}")
        if no_ingraph_meta and (not e.get("forced_off") or sup):
            why.append("no_ingraph_meta requested but the in-graph update is still on")
        if not no_ingraph_meta and e.get("forced_off"):
            why.append("ingraph_meta forced_off without no_ingraph_meta")
    conv = [e for e in ev if e.get("event") == "conv_impl"]
    detail["conv_impl"] = sorted({str(e.get("impl")) for e in conv})
    if conv and (len(conv) != tp_size or any(e.get("impl") != conv_impl for e in conv)):
        why.append(f"conv_impl lines {[(e.get('pid'), e.get('impl')) for e in conv]}, want {tp_size} x {conv_impl!r}")
    if not conv and conv_impl == "torch":
        why.append("conv_impl torch requested but no conv_impl line")
    chk = [e for e in ev if e.get("event") == "draft_load_check"]
    detail["load_check"] = len(chk)
    if len(chk) != tp_size:
        why.append(f"draft_load_check lines {len(chk)}, want {tp_size}")
    if ring_k is not None and not any(e.get("num_speculative_tokens") == ring_k for e in ring):
        why.append(f"no qsa_ring line for K = {ring_k}")
    if why:
        detail["why"] = why
        return "NOT_INTEGRATED", detail
    return "ENGAGED", detail


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-tree")
    ap.add_argument("--no-ring", action="store_true")
    ap.add_argument("--no-draft-block", action="store_true")
    ap.add_argument("--shas", action="store_true")
    a = ap.parse_args()
    if a.check_tree:
        print(json.dumps(verify_dflash_patch(a.check_tree, ring=not a.no_ring, draft_block=not a.no_draft_block), indent=1))
    if a.shas or not a.check_tree:
        print(json.dumps({rel: {"pre": p[0], "post": p[1], "patch_sha": PATCH_SHA[rel]} for rel, p in PATCHES.items()},
                         indent=1))
