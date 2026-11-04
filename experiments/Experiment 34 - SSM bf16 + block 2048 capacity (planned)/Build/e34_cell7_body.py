# ==================================================================== CELL 7 body (exec'd by the CELL 7 stub into the notebook globals)
KV_KIT_GATE = json.loads(r"""{"passed": false, "checks": {"cell0_identity_submission": true, "inputs_found": true, "merge_ok": true, "sha256_all_ok": true, "drafter_pins_ok": true, "merlin_install_ok": true, "merlin_flag_sources_equal": true, "patch_60490_ok": true, "seam_apply_verify_revert_ok": true, "host_phase_all_5_ok": true, "launch_check_all_5_ok": true, "env_in_spawned_child_all_5": true, "tree_check_all_5": true, "argv_fields_ok": true, "spec_config_preflight_ok": true, "kv_cache_configs_ok": true, "drafter_loaded_weights_ok": true, "e2_guard_dflash_ok": true}, "per_k": {"D4s": true, "D3s": true, "D4": true, "D4t": true, "D7": true, "D4b": true}, "kv_summary": {"D4s": {"num_blocks": [540, 540, 540, 540], "kv_group_count": 6, "drafter_group_count": 1, "groups": [{"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 5, "block_size": 4096, "page_size_bytes": 10485760, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 6, "block_size": 4096, "page_size_bytes": 12582912, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 6, "block_size": 4096, "page_size_bytes": 12582912, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["CircularBufferSpec"], "layers": 6, "block_size": 8, "page_size_bytes": 13824, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["FullAttentionSpec", "MLAAttentionSpec"], "layers": 12, "block_size": 4096, "page_size_bytes": 14155776, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["SlidingWindowSpec"], "layers": 5, "block_size": 1024, "page_size_bytes": 10485760, "is_eagle_group": false, "drafter_layers": 5}], "max_concurrency_131072": 12.833333333333334, "wasted_bytes_per_block": 5597216, "attn_metadata_builders": 7}, "D3s": {"num_blocks": [540, 540, 540, 540], "kv_group_count": 6, "drafter_group_count": 1, "groups": [{"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 5, "block_size": 4096, "page_size_bytes": 10485760, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 6, "block_size": 4096, "page_size_bytes": 12582912, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 6, "block_size": 4096, "page_size_bytes": 12582912, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["CircularBufferSpec"], "layers": 6, "block_size": 8, "page_size_bytes": 13824, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["FullAttentionSpec", "MLAAttentionSpec"], "layers": 12, "block_size": 4096, "page_size_bytes": 14155776, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["SlidingWindowSpec"], "layers": 5, "block_size": 1024, "page_size_bytes": 10485760, "is_eagle_group": false, "drafter_layers": 5}], "max_concurrency_131072": 12.833333333333334, "wasted_bytes_per_block": 6190720, "attn_metadata_builders": 7}, "D4": {"num_blocks": [189, 189, 189, 189], "kv_group_count": 6, "drafter_group_count": 1, "groups": [{"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 5, "block_size": 4096, "page_size_bytes": 10485760, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 6, "block_size": 4096, "page_size_bytes": 12582912, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 6, "block_size": 4096, "page_size_bytes": 12582912, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["CircularBufferSpec"], "layers": 6, "block_size": 8, "page_size_bytes": 13824, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["FullAttentionSpec", "MLAAttentionSpec"], "layers": 12, "block_size": 4096, "page_size_bytes": 14155776, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["SlidingWindowSpec"], "layers": 5, "block_size": 4096, "page_size_bytes": 41943040, "is_eagle_group": false, "drafter_layers": 5}], "max_concurrency_131072": 4.947368421052632, "wasted_bytes_per_block": 5597216, "attn_metadata_builders": 7}, "D4t": {"num_blocks": [540, 540, 540, 540], "kv_group_count": 6, "drafter_group_count": 1, "groups": [{"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 5, "block_size": 4096, "page_size_bytes": 10485760, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 6, "block_size": 4096, "page_size_bytes": 12582912, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 6, "block_size": 4096, "page_size_bytes": 12582912, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["CircularBufferSpec"], "layers": 6, "block_size": 8, "page_size_bytes": 13824, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["FullAttentionSpec", "MLAAttentionSpec"], "layers": 12, "block_size": 4096, "page_size_bytes": 14155776, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["SlidingWindowSpec"], "layers": 5, "block_size": 1024, "page_size_bytes": 10485760, "is_eagle_group": false, "drafter_layers": 5}], "max_concurrency_131072": 12.833333333333334, "wasted_bytes_per_block": 5597216, "attn_metadata_builders": 7}, "D7": {"num_blocks": [189, 189, 189, 189], "kv_group_count": 6, "drafter_group_count": 1, "groups": [{"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 5, "block_size": 4096, "page_size_bytes": 10485760, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 6, "block_size": 4096, "page_size_bytes": 12582912, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 6, "block_size": 4096, "page_size_bytes": 12582912, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["CircularBufferSpec"], "layers": 6, "block_size": 8, "page_size_bytes": 13824, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["FullAttentionSpec", "MLAAttentionSpec"], "layers": 12, "block_size": 4096, "page_size_bytes": 14155776, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["SlidingWindowSpec"], "layers": 5, "block_size": 4096, "page_size_bytes": 41943040, "is_eagle_group": false, "drafter_layers": 5}], "max_concurrency_131072": 4.947368421052632, "wasted_bytes_per_block": 3816704, "attn_metadata_builders": 7}, "D4b": {"num_blocks": [378, 378, 378, 378], "kv_group_count": 6, "drafter_group_count": 1, "groups": [{"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 5, "block_size": 4096, "page_size_bytes": 10485760, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 6, "block_size": 4096, "page_size_bytes": 12582912, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["MambaSpec"], "layers": 6, "block_size": 4096, "page_size_bytes": 12582912, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["CircularBufferSpec"], "layers": 6, "block_size": 8, "page_size_bytes": 13824, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["FullAttentionSpec", "MLAAttentionSpec"], "layers": 12, "block_size": 4096, "page_size_bytes": 14155776, "is_eagle_group": false, "drafter_layers": 0}, {"spec": "UniformTypeKVCacheSpecs", "inner": ["SlidingWindowSpec"], "layers": 5, "block_size": 4096, "page_size_bytes": 20971520, "is_eagle_group": false, "drafter_layers": 5}], "max_concurrency_131072": 9.921052631578947, "wasted_bytes_per_block": 5597216, "attn_metadata_builders": 7}}, "drafter_loaded_weights": true, "require_gpu_load_check": true, "provenance": {"kit_notebook_sha256": "e21982855d6430616f277784f162cad27cac94f7bcd8488fed97f123645ba730", "matches": ["closed_rebuild", "pre_rebuild_file"]}, "identity": {"tp": "2", "pp": "2", "partition": "25,23", "merlin_commit": "ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5"}, "why": ["E34 closed build: the E34 real CPU kit has not passed (build with --kv-gate-report)"], "per_k_success": {"D3s": {"kv_configs_ok": true, "k100_r9_run_ok": true, "run": "K3_swblock1024", "blocks": 540, "predicted": {"blocks_about_540": true, "pp1_page_14155776": true, "role": "record-only capacity prediction (research A)"}, "success": true}, "D4s": {"kv_configs_ok": true, "k100_r9_run_ok": true, "run": "K4_swblock1024", "blocks": 540, "predicted": {"blocks_about_540": true, "pp1_page_14155776": true, "role": "record-only capacity prediction (research A)"}, "success": true}}, "k100_kit_checks": {"k100_setup_scales_sidecar_ok": true, "k100_r9_q480_dsw_kv_configs_ok": true, "k100_dsw_F_10_14": true, "k100_block_drop_level_for_band_ok": true, "k100_mtp_vs_drafter_bytes_found": true, "k100_r12_klee100_draft_unquantized_67": true, "k100_scan_empty_method_map_pass": true, "k100_tokenizer_mask_row_ok": true, "k100_captures_argv_packviews_seqs24_ok": true, "k100_seam_ppack_60490_apply_verify_revert_ok": true}, "report": "D:/ARC AGI 2/_scratch/kits_wave3/q27-e27-nbdf-k100-kit-real/nbdf_kit_real_report.json", "report_sha256": "7417a56be3971edda30284182b448bef01e9133192dd66ac55aafa2dcd74fe77", "closed_build_sha256": "e21982855d6430616f277784f162cad27cac94f7bcd8488fed97f123645ba730", "pre_rebuild_sha256": "e21982855d6430616f277784f162cad27cac94f7bcd8488fed97f123645ba730", "e34_kit_checks": {"e34_arm_plan_ok": null, "e34_launch_path_all_arms_ok": null, "e34_help_flag_gate_real_ok": null, "e34_argparse_change_per_arm": null, "e34_kv_probes_ok": null, "e34_launch_env_ok": null, "e34_captures_argv_packviews_ok": null, "e34_switch_default_ok": null, "e34_neverdo_lift_exact": null, "e34_merlin_source_ok": null, "e34_tf_long_ok": null}, "e34_provenance": null}""")   # the real CPU kit's get_kv_cache_configs gate (build time); not passed -> D arms skipped
_EMB = [(NBDF_LIB_SRC, NBDF_LIB_SHA, "nbdf_lib"), (NB4T_LIB_SRC, NB4T_LIB_SHA, "nb4t_lib"), (TL_SRC, TL_SHA, "nb4_timeline"),
        (PPACK_SRC, PPACK_SHA, "nb4_ppack")] + ([(SEAM_SRC, SEAM_SHA, "dflash_seam")] if SEAM_SHA else [])
for _src, _sha, _n in _EMB:
    if hashlib.sha256(_src.encode("utf-8")).hexdigest() != _sha:
        raise RuntimeError(f"EMBED_SHA_MISMATCH: {_n}")
    (TMP / f"{_n}.py").write_text(_src, encoding="utf-8")
if str(TMP) not in sys.path:
    sys.path.insert(0, str(TMP))
T = importlib.import_module("nb4t_lib")
N = importlib.import_module("nbdf_lib")
TL = importlib.import_module("nb4_timeline")
P = importlib.import_module("nb4_ppack")
DS, SEAM_API, SEAM_LOAD = None, None, {"embedded": bool(SEAM_SHA), "sha256": SEAM_SHA or None}
if SEAM_SHA:
    try:
        DS = importlib.import_module("dflash_seam")
        SEAM_API = N.seam_api(DS)
        SEAM_LOAD["caps"] = SEAM_API["caps"]
        if not SEAM_API["ok"]:
            SEAM_LOAD["error"], DS, SEAM_API = "dflash_seam lacks " + ", ".join(SEAM_API["missing"]), None, None
    except Exception as e:
        SEAM_LOAD["error"], DS, SEAM_API = f"{type(e).__name__}: {e}"[:400], None, None
else:
    SEAM_LOAD["error"] = "SEAM_NOT_EMBEDDED: the builder had no dflash_seam.py"
NBDF = R.setdefault("nbdf", {})
# ---- E27 B2 (user rule 2026-11-01): only hard breakage stops an arm; every heuristic is recorded. The lab's degenerate
# check is reduced to hard garbage (EMPTY / REPEATED_CHAR / BANG_GARBAGE) for every gate (lab steady, smoke, oracle, long
# oracle, sampled, windows); nbdf_lib.garbage_review keeps the full check (.e27_full) and records LOW_VARIETY & co. as the
# window's labels. The cells' probe / sampled / oracle / quality gates are rebound to their hard-only nbdf_lib forms.
if not hasattr(L.degenerate, "e27_full"):
    L.degenerate = N.hard_only(L.degenerate)


def short_gate_failed(o):
    """B2: stop only on a probe REQUEST error or a hard sampled failure (error, dead server, no token, hard garbage)."""
    return N.short_gate_hard(o.get("probes"), o.get("sampled"), FAKE)


def quality_gate(o, ref):
    """B2: FAIL_* only for hard breakage; FAIL_HIGH_MARGIN / wrong probe answers / a short sampled request are recorded."""
    return N.quality_gate_hard(o, ref, FAKE, CLEAN)


def oracle_gate(o, need=(), extra=()):
    """B2: NO_GO only for request errors / hard garbage (oracle or a window) / a probe request error."""
    return N.oracle_gate_hard(o, need, extra, _windows(o), FAKE, CLEAN)


def evidence_gate(o, expected_rows, require_sampled=True):
    """B2 (fail-closed evidence kept): a missing piece is INCONCLUSIVE; NO_GO only for a probe request error, a hard sampled
    failure, oracle request errors or hard oracle garbage; wrong probe answers / a short sampled request are recorded."""
    pr = o.get("probes")
    if not isinstance(pr, dict):
        return "INCONCLUSIVE_PROBES_MISSING"
    if not FAKE and N._req_err(pr):
        return "NO_GO_SHORT_PROBE"
    if require_sampled:
        sm = o.get("sampled")
        if not isinstance(sm, dict):
            return "INCONCLUSIVE_NO_SAMPLED_TEST"
        if N.sampled_hard_fail(sm):
            return "NO_GO_SAMPLED_PATH"
    if expected_rows is None:
        return None
    orc = o.get("oracle")
    if not isinstance(orc, dict):
        return "INCONCLUSIVE_NO_ORACLE"
    if any(orc.get("errors") or []):
        return "NO_GO_ORACLE_REQUEST_ERROR"
    if any(g in N.HARD_GARBAGE for g in (orc.get("garbage") or [])):
        return "NO_GO_ORACLE_GARBAGE"
    if orc.get("row_count") != expected_rows or orc.get("complete") is not True:
        return "INCONCLUSIVE_ORACLE_INCOMPLETE"
    return None


NBDF.update(question="does DFlash2 (incoai drafter, K = 4 / 3 / 7) beat MTP K3 on the Q-defer base at 8k, 40k, 90k and 116k, "
                     "with greedy token ids EXACT / NEAR_TIE vs Q, no prefix-hit collapse and no concurrency contamination?",
            lib=N.LIB_VERSION, lib_sha256=NBDF_LIB_SHA, nb4t_lib_sha256=NB4T_LIB_SHA, timeline_sha256=TL_SHA, ppack_sha256=PPACK_SHA,
            seam=SEAM_LOAD, kv_kit_gate=KV_KIT_GATE, nb4t_notebook_sha256="675b1b9db76d49db1c71bcb271d9a7e9433abfb9c0536b93dbdb70543d073fd0", e18_r2_cell0_sha256="abe694be2a35b4dc00bfbf18fe2fdb0916bff15d74f502b5f5836db30cd46ec0",
            merlin_flags=json.loads(r"""{"Q": {"change": "none: NB4T's Q (nb4_ppack rev 2 + NB4_PPDEFER=1, Merlin's own spec + captures)", "merlin": ["tools/merlin/launch.py:441 (spec = {method mtp, num_speculative_tokens R.MTP_K, draft_sample_method probabilistic})", "tools/merlin/registry.py:49 (MTP_K = 3)", "tools/merlin/launch.py:459-460 (--compilation-config FULL_AND_PIECEWISE + capture_sizes)", "tools/merlin/registry.py:83-89 (capture_sizes_for(8, 3) = [1,2,3,4,6,8,12,16,24,32])"]}, "D_common": {"change": "seam env (DFLASH_SEAM=1 [+ DFLASH_DRAFT_EMBED_HOST=1 when the seam has it] [+ DFLASH_QSA_RING_WIDEN=1 at K >= 5]) + trailing --speculative-config {\"method\":\"dflash\",\"model\":<drafter>,\"num_speculative_tokens\":K,\"draft_sample_method\":\"probabilistic\",\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"FLASH_ATTN\"} + trailing --compilation-config with the multiples of 1 + K up to 8 x (1 + K)", "override": ["tools/merlin/launch.py:110-111 (argv = prefix + vllm serve ckpt + args + EXTRA: extra args come last, the last occurrence wins; Merlin has no knob for the spec method / model: :441 hard-codes mtp)", "tools/merlin/launch.py:452 (Merlin's own --speculative-config), :459-460 (its --compilation-config)"], "spec_fields": ["vllm/config/speculative.py:395 num_speculative_tokens", ":398 model", ":401 method (explicit: the mount path detection at :1357-1373 needs 'dflash' in the path)", ":409 draft_tensor_parallel_size (unset -> the target TP 2, :1579-1583 + :1744-1773)", ":426 attention_backend (:1832-1839 parses the name)", ":430 kv_cache_dtype (None = inherit fp8_e4m3; 'auto' = BF16)", ":380-385 + :1820-1830 (the draft config overrides)", ":602 draft_sample_method (DFlash2: fp32 draft logits, dflash2/speculator.py:200-204; speculator.py:246-258)", ":1498-1499 parallel_drafting = True", ":1968-1974 DFlash: K new slots", ":2017-2018 use_dflash()"], "engine": ["vllm/config/vllm.py:661-664 (DFlash lookahead K + 1 slots)", ":697-715 (uniform_decode_query_len = 1 + K)", "vllm/v1/worker/gpu/spec_decode/__init__.py:27-39 (dflash + DFlash2DraftModel -> DFlash2Speculator)", "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:126-153 (draft graphs FULL_DECODE_ONLY, no PIECEWISE; eager warning when the backend lacks full graphs; 'Capturing model for DFlash2 speculator...'), :170 (progress bar)", "vllm/platforms/cuda.py:446 ('Using <selected> backend.' = the drafter's attention_backend), :507 (auto selection)", "vllm/v1/worker/gpu/cudagraph_utils.py:332-357 (sizes rounded up to multiples of 1 + K; above the max dropped)", "vllm/v1/worker/gpu/model_runner.py:386-393 + :524-533 (aux hidden states for dflash + PP relay check)", "vllm/v1/worker/gpu/spec_decode/eagle/eagle3_utils.py:100-111 (target_layer_ids + 1 -> (6, 16, 26, 36, 46))", "vllm/model_executor/models/config.py:1049-1058 (the Qwen4Exp allowlist: the seam adds dflash)", "vllm/models/qwen4_exp/common/qsa_cache.py:955-973 (QSA ring 4 x ceil((4 + K) / 4); the seam widens K >= 5)", "vllm/v1/core/kv_cache_utils.py:2063-2119 (the drafter group is not an eagle group: no trailing-block drop on hits)"], "draft_kv": "BF16 (auto) + FLASH_ATTN: FP8 drafter KV makes FA2 refuse on sm_89 (fa_utils.py:306-311) -> FlashInfer -> eager drafter (verify A B6); D4b = FP8 + TRITON_ATTN (triton_attn.py:302-308, 327-328), last + optional (upstream #54690)", "captures": "the multiples of 1 + K up to 8 x (1 + K): K3 4..32, K4 5..40, K7 8..64 (every decode batch an exact graph for the target AND the FULL_DECODE_ONLY drafter); Q keeps Merlin's ladder (b = 5 / 7 padded to 6 / 8)", "memory": "PP1 per card (verify A B9 / B section 3): drafter 0.727 GiB + embed 0.592 GiB (unless on the host) + fp32 draft_logits 8 x K x 248320 x 4 B (32 MB at K4) - the MTP module (~1.0-1.3 GiB); D pins on the ladder 3,285,145,068 / 3,016,709,612 (Q 3,553,580,524); 512 MiB free per rank after load / KV / graphs / first decode, else one rung down", "readback": ["config/model.py:735 'Resolved architecture: DFlash2DraftModel'", "seam events E27_DFLASH {json}: aux_layers (4 ranks, ids, local [6, 16 | 26, 36, 46], slot base 2), embed (PP1, sources, abs mean), embed_sidecar, qsa_ring", "v1/spec_decode/metrics.py:121-138 SpecDecoding lines (per-position acceptance)", "nb1_cache_info: kv_cache_specs (block sizes, CircularBufferSpec = the QSA ring capacity, the drafter's SlidingWindowSpec dtype), kv_cache_config groups", "kv_cache_utils 'Maximum concurrency for 131,072 tokens per request' -> F, pool blocks vs Q", "'Model loading took' / 'Graph capturing finished' per rank"]}, "D4s": {"k": 4, "sw_block": "DFLASH_DRAFT_SW_BLOCK=1024: kit v4 R9 205 blocks at 3,016,709,612 B (BF16 drafter page 10.5 MB < the target's 14.2 MB stride) vs 71 at block 4096 (page 41.9 MB); the seam lowers cache_config.block_size engine-wide (v1/engine/core.py:436-448): read back, SW_BLOCK_LOWERED_ENGINE_BLOCK recorded; pin: the D pin 3,016,709,612 first (the kit's 205 blocks), one step-down to 2,748,274,156 on OOM / the 512 MiB gate", "ring": "capacity 8 divides 4096 and 1024"}, "D3s": {"k": 3, "sw_block": "as D4s at K = 3 (kit R9 K3_swblock1024)"}, "D4": {"k": 4, "role": "the 71-block BF16 control (block 4096): smoke + oracle + 8k x 2 only"}, "D7": {"k": 7, "ring": "NEEDED: 4 x ceil(11 / 4) = 12 does not divide 4096 (qsa_cache.py:962); widened to 16 (DFLASH_QSA_RING_WIDEN=1); runs only when the seam's ring patch applied AND verified"}, "D4b": {"k": 4, "kv": "fp8_e4m3 + TRITON_ATTN (optional, last)"}, "ISO": "the first failing D arm reruns once with NB4_PPDEFER unset (same patched tree): QDEFER_INTERACTION / DFLASH_ITSELF", "long_bands": {"rule": "user 2026-11-01: every speed and correctness test up to ~120k context", "oracle": "greedy 256 tokens on PROMPTS['90k'][7] + PROMPTS['116k'][7] (Q = reference; D vs Q EXACT / NEAR_TIE)", "windows": "40k / 90k / 116k at the arm's resident level (pool + F of the boot; running == level confirmed, one step down on a KV-bound window); Q runs 90k and 116k twice (A/A noise)", "reset": "/reset_prefix_cache (VLLM_SERVER_DEV_MODE=1) before the first window of each long band"}, "K100": {"boot": "klee100 LOAD rev 3 (K3_ONLY 7773e63f) / K100Q / E28 K100", "cfg": {"arms": ["Q", "D3s", "D4s"], "seqs": 24, "conc_ladder": [2, 8, 23], "q_pin": 7933526016, "q_ladder": [7933526016, 7665090560], "d_retry": [[7933526016, {"DFLASH_DRAFT_EMBED_HOST": "1"}], [7665090560, {"DFLASH_DRAFT_EMBED_HOST": "1"}], [7396655104, {"DFLASH_DRAFT_EMBED_HOST": "1"}], [6859784192, {"DFLASH_DRAFT_EMBED_HOST": "1"}]], "pin_ladder": [7933526016, 7665090560, 7396655104, 6859784192], "memory_gate": "flag", "q_ref": "k100_none", "method_map_need_mtp": "q_only", "q_i8": true, "q_captures": [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96], "wkeys": ["8k@16#1", "8k@16#2", "profile_8k@16", "40k@16"], "windows": [["8k", 16, "#1"], ["8k", 16, "#2"], ["20k", 24, ""], ["40k", 16, ""], ["90k", 8, ""], ["116k", 6, ""]], "profile_level": 16, "tf_bands": ["2k", "8k", "40k"], "tf_mean_margin": 0.05, "tf_rank1_margin": 0.03, "long_thr_floor": {"90k": 0.037, "116k": 0.054}, "shorten_arm": "D4s", "shorten_min_left_min": 25, "tap_mode_alt": "pre_combine", "tap_est_min": 15, "tap_cap_min": 10, "prefill_admit_floor_tok_s": 3000}, "method_map": "need_mtp only on MTP arms: the DFlash2 drafter is dense (5 layers, no MoE) -> no MTP / drafter expert rows", "scales": "the sidecar has the 12 target QSA layers only; the MTP attention layer is DEFAULT 1.0 and recorded (never an error)"}, "NBD1": {"base": "B = E29_DFLASH production = NBDF-K100 D3s (nbdf_lib.arm_plan D3s after configure(SEQS=24, ...)) + DFLASH_DRAFT_EMBED_HOST=1 from the first try (dflash_seam host embed; the D ladder then skips its same-pin EH rung) + QSA 128 / 64", "QSA": ["tools/merlin/launch.py:419-420 (build() sets VLLM_SPARSE_INDEXER_MAX_LOGITS_MB / VLLM_E2_QSA_LOGITS_GRANULE_MB = 256)", "the notebook's merlin_fn_server: env.update(NB1_ENV) after build() (NB1_ENV wins)", "qsa_indexer.py:797 'QSA prefill logits backing: granule <n> MiB' (logged at the first prefill)", "E29_DFLASH_D3S rev 2c report (the 128 / 64 decision)"], "D4s_DYN": ["vllm/config/speculative.py:493-498 num_speculative_tokens_per_batch_size", "v1/spec_decode/dynamic/utils.py:7-150", "config/vllm.py:1068-1086 (PIECEWISE override only without the V2 runner)", "v1/worker/gpu/model_runner.py:2786-2803"], "GDN": ["vllm/merlin/envs.py:544-555 (strict switches), :822-823 (not in the compile hash)"], "IDX": ["vllm/config/attention.py:88-92", "vllm/models/qwen4_exp/nvidia/indexer_qsa.py:154-166", "vllm/engine/arg_utils.py:1856 --attention-config (JSON)"], "LP": ["vllm/config/scheduler.py:79-90", "vllm/engine/arg_utils.py:1735-1740"], "S28_S32": ["launch.py:453 Merlin's --max-num-seqs (the trailing one wins)", "registry.py:83-89 (multiples of 1 + K here)"], "SPC": ["vllm/config/cache.py:198-202", "vllm/v1/core/kv_cache_manager.py:185-198"], "dyn_source_check": "Merlin ad42e37 config/speculative.py:493 num_speculative_tokens_per_batch_size = [(start, end, K)] inclusive ranges, clamped to num_speculative_tokens (v1/spec_decode/dynamic/utils.py); the scheduler (v1/core/sched/scheduler.py:348) picks K per batch size. config/vllm.py:1068-1085 overrides FULL cudagraphs to PIECEWISE unless the V2 model runner is used (NBDF runs the V2 GPU runner: v1/worker/gpu); v1/worker/gpu/cudagraph_utils.py:283-300 then captures the decode graphs of every tier. -> num_speculative_tokens 4 + [[1, 8, 4], [9, 24, 3]], captures = every b x 5 for b <= 8 and every b x 4 for b 9-24. Per-batch K reaches the drafter: v1/worker/gpu/model_runner.py:2786-2803 passes num_speculative_tokens=<this batch's K> to speculator.propose() and sets draft_tokens[:, K:] = -1 when K < the max; DFlashSpeculator.propose (spec_decode/dflash/speculator.py:343-369, the DFlash2 base) accepts the argument and drafts its full fixed block (num_query_per_req = 1 + num_speculative_steps, :62), so the drafter cost stays K4 at every batch size and only the target's verify length shrinks at 9-24 running. Real-failure read-back: an 'Overriding cudagraph_mode' or 'Disabling num_speculative_tokens_per_batch_size' log line."}, "E34": {"ssm_bf16": ["vllm/engine/arg_utils.py:1396 --mamba-ssm-cache-dtype (config/cache.py:63 MambaDType incl. bfloat16, :186 default auto)", "model_executor/models/config.py:908-934 Qwen3_5 hook (Qwen4ExpForConditionalGenerationConfig :966-973 inherits): auto -> the HF mamba_ssm_dtype (float32); an explicit value wins + the warning 'Qwen3.5 model specifies mamba_ssm_dtype=...' (:928)", "models/qwen4_exp/nvidia/model.py:1016-1027 GDN state dtypes = (conv, ssm) from cache_config + the fp32 replay record (RecoverSSM)", "model_executor/layers/mamba/mamba_utils.py:99-112 ssm dtype = cache_config.mamba_ssm_cache_dtype", "config/vllm.py:3417-3423 RecoverSSM accepts bf16 / fp32 SSM caches (the only refusal: stochastic rounding); :3477-3491 fp32 is required only for NemotronH Mamba-2", "model_executor/layers/mamba/gdn/recoverssm_gdn.py:254 only the replay record must be fp32; :440 / :445 the state store casts to the cache dtype; :495 every layer the same state dtype"], "block_2048": ["platforms/interface.py:940-1140 _align_hybrid_block_size: kernel alignment = max(min backend block, block_size) (:1077-1086), attention block = alignment x cdiv(mamba page, alignment x 512 B) (:1087-1093), raised + 'Setting attention block size' when smaller (:1095-1101), align mode mamba_block_size = block_size (:1103-1104), the Mamba page padded to block x 512 B (:1107-1122 'Padding mamba page size'): bf16 946,560 B <= 1,048,576 B at 2048 (10.78 %); fp32 1,732,992 B -> raised to 4096 (the CPU planner control)", "v1/attention/backends/flash_attn.py:354-365 FA2 on sm_89: MultipleOf(16); models/qwen4_exp/nvidia/qsa.py:377-379 QSA MultipleOf(16)", "models/qwen4_exp/common/qsa_cache.py:954-970 QSA ring capacity 4 x ceil((4 + K) / 4) = 8 must divide the block (8 | 2048)", "models/qwen4_exp/common/qsa_cache.py:976-986 the indexer compressed-key cache = an MLAAttentionSpec at the cache block, tokens_per_state 4 (2048 / 4 = 512 rows); nvidia/indexer_qsa.py:187-200 page_cap = blocks x block / 4 (= 32,768 rows at 4096 and at 2048)", "v1/core/kv_cache_utils.py:805-905 scheduler block = LCM of the group blocks (2048), hash unit = prefix_match_unit 64 (divides 2048 / 512, :884), the compressed key's 4-token alignment (64 % 4, :904)", "v1/engine/core.py:431-450 the engine block = min over the prefix-cacheable groups (the SW drafter block: 1024 on B, 512 on S16B2); config/vllm.py:3289-3318 validate_block_size (align: chunked MM input only)", "v1/kv_cache_interface.py:1077-1083 MambaSpec align: 2 pages a request (running + CoW checkpoint), num_speculative_blocks 0 under RecoverSSM", "v1/core/kv_cache_utils.py:1251-1262 + :1304 one pool block = the largest group page of the rank", "dflash_seam ATT_PATCH (Exp 27 dflash_seam.py:1116-1180): DFLASH_DRAFT_SW_BLOCK must divide the attention block (512 | 2048)", "tools/merlin/launch.py:334 BLOCK_SIZE default 4096 -> :474 --block-size; :475 --prefix-match-unit 64; :110-111 the notebook's extra args come last (the last --block-size wins)"], "sw_512_why": "CPU planner (e34_capacity_probe.json): block 2048 with SW 1024 -> the drafter page 10,485,760 B sets the PP1 stride, pool 756 x 2048 (mc 9.93 < B's 12.25); SW 512 -> 5,242,880 B, pool 1,080 x 2048 (mc 13.32)", "never_do": "nb4t_lib.NEVER_DO_ARG_VALUES {'--mamba-ssm-cache-dtype': ('bfloat16',)} ('C3 SSM bf16: parked (0 capacity + B2 FAIL)', E25 NB1 R2): lifted for S16 / S16B2 / S16B2W only (E34.launch_check / check_arm_plan drop exactly 'argv --mamba-ssm-cache-dtype bfloat16')"}}"""))
B3_PIN = int(K100_CFG["q_pin"])             # E27 K100: every arm starts at the K100Q / E28 top K3 rung (Q pool 479, F 6)
if B3_PIN != 7933526016:
    raise RuntimeError(f"K100 Q pin {B3_PIN} != 7933526016")
Q_EXTRA = EXT_FLAG + ["--language-model-only", "--profiler-config", PROFILER_CFG]   # = NB4T Q_EXTRA
Q_BASE_ENV = dict(DIAG_ENV)                                                      # = NB4T Q_BASE_ENV
N.configure(SEQS=int(K100_CFG["seqs"]), CONC_LADDER=tuple(K100_CFG["conc_ladder"]), Q_CAPTURES=list(K100_CFG["q_captures"]),
            Q_TAIL=["--compilation-config", N.compilation_json(K100_CFG["q_captures"])], WKEYS=tuple(K100_CFG["wkeys"]),
            PREFIX_RECORD_ONLY=True, SHORT_EXIT_ARMS=(), LONG_THR_FLOOR=dict(K100_CFG["long_thr_floor"]),
            TF_MEAN_MARGIN=float(K100_CFG["tf_mean_margin"]), TF_RANK1_MARGIN=float(K100_CFG["tf_rank1_margin"]), TF_REQUIRED=True)   # E27 K100
Q_EXTRA = Q_EXTRA + ["--max-num-seqs", str(K100_CFG["seqs"])]               # E27 K100: every arm at 24 sequences (E28)
Q_BASE_ENV["MERLIN_SHARE_DRAFT_WORKSPACE"] = "1"                              # E27 K100: = E28 / K100Q (inert with DFlash)
RESP_DIR = OUT / "responses"                # memory rule: the complete response of every request, gzipped jsonl per arm
RESP_DIR.mkdir(parents=True, exist_ok=True)
RESP = {"arm": "setup", "rows": None, "counts": {}}
_RESP_LOCK = threading.Lock()


def _resp_write(st):
    r = st.row()
    pr = st.body.get("prompt") or []
    rec = dict(r, arm=RESP["arm"], item=CUR.get("item"), port=st.port, prompt_len=len(pr),
               prompt_sha=hashlib.sha256(json.dumps(pr).encode()).hexdigest()[:16],
               tokens_sha=hashlib.sha256(json.dumps(r.get("tokens") or []).encode()).hexdigest()[:16],
               sampling={k: v for k, v in st.body.items() if k != "prompt"},
               ttft=round(st.t_first - st.t_start, 3) if st.t_first and st.t_start else None,
               sec=round(st.t_end - st.t_start, 3) if st.t_end and st.t_start else None, t=time.time())
    with _RESP_LOCK:
        with gzip.open(RESP_DIR / f"responses_{RESP['arm']}.jsonl.gz", "at", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
        RESP["counts"][RESP["arm"]] = RESP["counts"].get(RESP["arm"], 0) + 1
        if RESP["rows"] is not None:
            RESP["rows"].append(rec)


ACTIVE_STREAMS = set()                      # every running L.Stream (the admission watchdog stops them on a wedge)


def _mk_nbdf_run(prev):
    """CELL 1's _traced_run (requests.jsonl) + the per-arm gz file; wrapped once (nbdf_lib.wrap_once)."""
    def _nbdf_run(self):
        ACTIVE_STREAMS.add(self)
        try:
            prev(self)
        finally:
            ACTIVE_STREAMS.discard(self)
            try:
                _resp_write(self)
            except Exception as e:
                R.setdefault("errors", []).append(f"response save: {e!r}"[:200])
    return _nbdf_run


N.wrap_once(L.Stream, "run", _mk_nbdf_run)
CUR = {"item": None, "kv_bytes": None}


# ---- the drafter Kaggle Model: one dir (glob, any framework case), pinned shas from the header + tensor bytes (no model load)
def drafter_resolve():
    """Exactly one dir under the Model mount (any framework-dir case) with config.json + model.safetensors on a bf16-embed path."""
    base = MD / N.DRAFTER_BASE
    cands = sorted({p.parent for p in base.rglob("config.json") if (p.parent / "model.safetensors").is_file()
                    and N.DRAFTER_VARIATION in p.parent.relative_to(base).parts}) if base.is_dir() else []
    if len(cands) != 1:
        return None, (f"DRAFTER_MISSING: no {N.DRAFTER_VARIATION} dir with config.json + model.safetensors under {base} (attach "
                      f"{N.DRAFTER_MODEL})" if not cands else f"DRAFTER_AMBIGUOUS: {[str(c) for c in cands]}")
    return cands[0], "ok"


def _sha_file(p, lo=0, n=None, rows=None):
    """sha256 of p[lo:lo+n] (whole file when n None); rows = row bytes: also counts all-zero rows."""
    h, zero, left_, buf = hashlib.sha256(), 0, n, b""
    zrow = b"\0" * rows if rows else None
    with open(p, "rb") as fh:
        fh.seek(lo)
        while left_ is None or left_ > 0:
            b = fh.read(1 << 24 if left_ is None else min(1 << 24, left_))
            if not b:
                break
            h.update(b)
            if left_ is not None:
                left_ -= len(b)
            if rows:
                buf += b
                k = len(buf) // rows * rows
                mv = memoryview(buf)
                zero += sum(1 for i in range(0, k, rows) if mv[i:i + rows] == zrow)
                buf = buf[k:]
    return h.hexdigest(), zero


def drafter_facts():
    d, why = drafter_resolve()
    if d is None:
        return {"ok": False, "why": why, "status": "DRAFTER_MISSING"}
    t0 = time.time()
    files, shas = {}, {}
    for f in sorted(p for p in d.rglob("*") if p.is_file()):
        rel = str(f.relative_to(d))
        files[rel] = f.stat().st_size
        shas[rel] = _sha_file(f)[0]
    st = d / "model.safetensors"
    keys = N.safetensors_keys(st)
    with open(st, "rb") as fh:
        hlen = int.from_bytes(fh.read(8), "little")
    emb = [k for k in keys if k.endswith("embed_tokens.weight")]
    facts = {"files": {k: [files[k], shas[k]] for k in files}, "tensors": len(keys), "embed": {}, "mask_row": {}, "zero_rows": None}
    if len(emb) == 1:
        e = keys[emb[0]]
        a, b = e["offsets"]
        row = int(e["shape"][1]) * 2
        es, zr = _sha_file(st, 8 + hlen + a, b - a, rows=row)
        facts["embed"] = {"name": emb[0], "shape": e["shape"], "dtype": e["dtype"], "nbytes": b - a, "sha256": es}
        facts["zero_rows"] = zr
        with open(st, "rb") as fh:
            fh.seek(8 + hlen + a + N.MASK_TOKEN_ID * row)
            mr = fh.read(row)
        facts["mask_row"] = {"sha256": hashlib.sha256(mr).hexdigest(), "norm": round(N.bf16_row_norm(mr), 8)}
    pins_ok, bad = N.drafter_pins_check(facts)
    cfg = json.loads((d / "config.json").read_text())
    man = None
    mf = next((d / n for n in ("dflash2_manifest.json", "E27_DRAFTER_MANIFEST.json") if (d / n).is_file()), None)
    if mf is not None:
        try:
            m = json.loads(mf.read_text())
            man = {k: v for k, v in (m.get("files") or {}).items() if k in files}
        except Exception:
            man = None
    chk = N.drafter_check(cfg, files, {"model.safetensors": keys}, shas, man)
    ok = pins_ok and chk["ok"]
    return {"ok": ok, "status": "OK" if ok else "DRAFTER_SHA_MISMATCH", "why": "ok" if ok else "; ".join(bad + ([chk["why"]] if not chk["ok"] else []))[:900],
            "dir": str(d), "pins_ok": pins_ok, "pin_mismatches": bad, "check": {k: chk.get(k) for k in ("ok", "why", "embed_in_checkpoint",
                                                                                                        "tensors", "selector_top_k", "block_size",
                                                                                                        "sliding_window", "manifest")},
            "facts": facts, "sec": round(time.time() - t0, 1),
            "config": {k: cfg.get(k) for k in ("architectures", "hidden_size", "num_hidden_layers", "num_key_value_heads", "head_dim",
                                               "vocab_size", "layer_types", "sliding_window", "dflash_config", "dtype")}}


DRAFTER = {"ok": False, "why": "not checked yet"}
try:
    DRAFTER = drafter_facts() if not FAKE else {"ok": True, "why": "fake", "dir": str(TMP / "fake_drafter"), "check": {"embed_in_checkpoint": True}}
except Exception as e:
    DRAFTER = {"ok": False, "status": "DRAFTER_CHECK_ERROR", "why": f"DRAFTER_CHECK_ERROR {type(e).__name__}: {e}"[:400]}
NBDF["drafter"] = DRAFTER
DRAFTER_DIR = DRAFTER.get("dir") or str(MD / N.DRAFTER_REL)
EMBED_IN_CKPT = bool((DRAFTER.get("check") or {}).get("embed_in_checkpoint"))
SEAM_ENVS, SEAM_NOTES = {}, {}
for _k in N.D_ARMS:
    SEAM_ENVS[_k], SEAM_NOTES[_k] = N.seam_env_for(SEAM_API, N.K_OF[_k], EMBED_IN_CKPT, sidecar_path=str(CKPT), host_embed=False,
                                                   mask_row_sha=N.DRAFTER_PINS["mask_row"]["sha256"],
                                                   conv_impl="torch" if _k in N.CONV_TORCH_ARMS else None,
                                                   draft_sw_block=N.SW_BLOCK if _k in N.SW_BLOCK_ARMS else None)
NBDF["seam_envs"], NBDF["seam_env_notes"] = SEAM_ENVS, SEAM_NOTES
ARMS = N.arm_plan(Q_EXTRA, Q_BASE_ENV, B3_PIN, DRAFTER_DIR, q_env=P.server_env("Q-defer"), seam_envs=SEAM_ENVS)
N.restrict_arms(K100_CFG["arms"])                # E27 K100: Q > D3s > D4s
ARMS = {k: ARMS[k] for k in N.ORDER}
N.check_arm_plan(ARMS)                      # never-do list + "each D arm = Q + exactly the DFlash change" (raises: no boot)

# ==================================================================== E25 NB-D1: B = E29_DFLASH production (NBDF-K100 D3s + EMBED_HOST + QSA 128/64), levers on B
NBDF_ARMS = ARMS                            # NBDF-K100's own plan (Q, D3s, D4s), checked by nbdf_lib.check_arm_plan
NBD1_LIB_SHA = "af0f795f685fe63f6169e5171f25f5d35f25bf66cfc6538c1272554d3190bad3"
NBD1_LIB_SRC = _nbd1_ds("nbd1_lib.py", NBD1_LIB_SHA)   # the payload dataset (sha-pinned)
if hashlib.sha256(NBD1_LIB_SRC.encode("utf-8")).hexdigest() != NBD1_LIB_SHA:
    raise RuntimeError("EMBED_SHA_MISMATCH: nbd1_lib")
(TMP / "nbd1_lib.py").write_text(NBD1_LIB_SRC, encoding="utf-8")
D1 = importlib.import_module("nbd1_lib")
D1_CFG = json.loads(r"""{"base": "E29_DFLASH = NBDF-K100 D3s + DFLASH_DRAFT_EMBED_HOST=1 (first try) + QSA 128/64", "base_key": "B", "arms": ["B", "Q256", "D4s", "DYN", "G1", "G2", "IDX", "LP", "S28", "S32", "SPC"], "switch_keys": ["B", "BAA", "PROF", "Q256", "D4s", "DYN", "G1", "G2", "IDX", "LP", "S28", "S32", "SPC"], "copies": {"a": {"B": true, "BAA": true, "PROF": true, "Q256": true, "D4s": true, "DYN": true, "G1": false, "G2": false, "IDX": false, "LP": false, "S28": false, "S32": false, "SPC": false}, "b": {"B": true, "BAA": false, "PROF": false, "Q256": false, "D4s": false, "DYN": false, "G1": true, "G2": true, "IDX": true, "LP": true, "S28": false, "S32": false, "SPC": false}, "c": {"B": true, "BAA": false, "PROF": false, "Q256": false, "D4s": false, "DYN": false, "G1": false, "G2": false, "IDX": false, "LP": false, "S28": true, "S32": true, "SPC": true}}, "owners": {"a": "foysalemonshanto", "b": "hosen42", "c": "foysalemonshanto"}, "budget_min": 160, "deadman_min": 165, "sibling": {"prefix_tokens": 6900, "suffix_tokens": 800, "gen": 16}, "qsa_b": {"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "128", "VLLM_E2_QSA_LOGITS_GRANULE_MB": "64"}, "qsa_merlin": {"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "256", "VLLM_E2_QSA_LOGITS_GRANULE_MB": "256"}, "prof_windows": [["8k", 16], ["20k", 24]], "prof_secs": 30.0, "lib_sha256": "af0f795f685fe63f6169e5171f25f5d35f25bf66cfc6538c1272554d3190bad3"}""")
_nbd1_on = globals().get("NBD1_ON")          # CELL 1 arm switches (E25 NB-D1): absent -> every key on, a missing key -> off
NBD1_SWITCH = D1.switch_map(_nbd1_on)
N.restrict_arms(("Q", "D3s"))                # N.SW_BLOCK_ARMS = (D3s,): every NB-D1 arm has parent D3s (launch_check, KV summary)
ARMS = D1.arm_plan(N, NBDF_ARMS["D3s"], seam_env_k4=SEAM_ENVS.get("D4s"), switch=NBD1_SWITCH, pin=int(K100_CFG["q_pin"]),
                   seqs=int(K100_CFG["seqs"]))
D1.check_arm_plan(N, T, ARMS, NBDF_ARMS["D3s"])   # B = D3s + EMBED_HOST + QSA 128/64; each lever = B + exactly its change (raises: no boot)
NBD1 = R.setdefault("nbd1", {})
NBD1.update(lib=D1.LIB_VERSION, lib_sha256=NBD1_LIB_SHA, cfg=D1_CFG, switch=D1.switch_record(_nbd1_on),
            walk=D1.copy_walk(ARMS, NBD1_SWITCH, BUDGET / 60), need_min=D1.copy_need_min(ARMS, NBD1_SWITCH),
            question="which 1500+ speed lever (QSA workspace, K4, dynamic K, GDN switches, indexer fp8, long prefill, seqs 28 / 32, "
                     "shared-prefix checkpoint) beats the DFlash2 production base B at 8k-116k (raw numbers, labels record-only)?",
            seam_env_d3s_vs_d4s_equal=SEAM_ENVS.get("D3s") == SEAM_ENVS.get("D4s"), dyn_source_check=D1.DYN_SOURCE_CHECK, b_ref=D1.B_REF,
            arms={k: {x: a.get(x) for x in ("name", "port", "d1_change", "spec", "captures", "streams", "windows_fixed", "prof_windows",
                                             "est_min", "cap_min", "time_model", "title", "d1_items", "env", "extra")} for k, a in ARMS.items()})
print("E25 NB-D1 arm switches:", NBD1_SWITCH, "| walk:", json.dumps(NBD1["walk"])[:700], flush=True)

# ==================================================================== E34: SSM bf16 + block 2048 (B / S16 / S16B2 / S16B2W on NB-D1's B)
E34_LIB_SHA = "65703276bf9bd93a11d4a088687efb98f62310ca80e8150968acca5df98d7d79"
E34_LIB_SRC = _nbd1_ds("e34_lib.py", E34_LIB_SHA)      # the payload dataset (sha-pinned)
if hashlib.sha256(E34_LIB_SRC.encode("utf-8")).hexdigest() != E34_LIB_SHA:
    raise RuntimeError("EMBED_SHA_MISMATCH: e34_lib")
(TMP / "e34_lib.py").write_text(E34_LIB_SRC, encoding="utf-8")
E34 = importlib.import_module("e34_lib")
E34_CFG = json.loads(r"""{"what": "E34 SSM bf16 + block 2048", "base": "NB-D1 B (E29_DFLASH)", "arms": ["B", "S16", "S16B2", "S16B2W"], "switch_keys": ["B", "S16", "S16B2", "S16B2W", "BMAX", "SMOKE"], "copies": {"full": {"B": true, "S16": true, "S16B2": true, "S16B2W": true, "BMAX": true, "SMOKE": false}, "smoke": {"B": false, "S16": false, "S16B2": true, "S16B2W": false, "BMAX": false, "SMOKE": true}}, "owners": {"full": "foysalemonshanto", "smoke": "foysalemonshanto"}, "budgets_min": {"full": 245, "smoke": 35}, "plan": {"B": {"ssm": "float32", "block": 4096, "sw": 1024}, "S16": {"ssm": "bfloat16", "block": 4096, "sw": 1024}, "S16B2": {"ssm": "bfloat16", "block": 2048, "sw": 512}, "S16B2W": {"ssm": "bfloat16", "block": 2048, "sw": 512}}, "w_rule": {"band_tokens": 40000, "min": 24, "max": 40, "pred": 29}, "cpu_planner": {"B": {"pool": 540, "stride": [14680064, 14680064, 14155776, 14155776], "F_adm": 12, "F_steady": 10, "mc_131072": 12.25}, "S16": {"pool": 540, "stride": [14680064, 14680064, 14155776, 14155776], "F_adm": 12, "F_steady": 10, "mc_131072": 12.25}, "S16B2": {"pool": 1080, "stride": [7340032, 7340032, 7077888, 7077888], "F_adm": 17, "F_steady": 12, "mc_131072": 13.32}, "S16B2_SW1024": {"pool": 756, "stride": [7340032, 7340032, 10485760, 10485760], "F_adm": 12, "F_steady": 10, "mc_131072": 9.93}, "fp32_block2048": "BLOCK_RAISED (the 1,732,992 B fp32 Mamba state page > the 1,048,576 B attention page at 2048)"}, "tf_long_sha256": "bd7c1db3abebb5a88633a5f72a543aceeb5d1402957ee0c58199bc49e09ecb20", "tf_long_file": "e34_tf_long.json", "tf_long_dataset": "foysalemonshanto/q27-e34-tf-long", "lib_sha256": "65703276bf9bd93a11d4a088687efb98f62310ca80e8150968acca5df98d7d79", "never_do_lifted": "argv --mamba-ssm-cache-dtype bfloat16"}""")
_e34_on = globals().get("E34_ON")            # CELL 1 arm switches (E34): absent -> E34.SWITCH_DEFAULT, a missing key -> off
NBD1_SWITCH = E34.switch_map(_e34_on)        # the name nbd1_main / prof_plan / the verdict read
ARMS = E34.arm_plan(D1, N, NBDF_ARMS["D3s"], seam_env_k4=SEAM_ENVS.get("D4s"), switch=NBD1_SWITCH, pin=int(K100_CFG["q_pin"]),
                    seqs=int(K100_CFG["seqs"]))
E34.check_arm_plan(D1, N, T, ARMS, NBDF_ARMS["D3s"], NBD1_SWITCH)   # B = NB-D1's B; each arm = B + its change (raises: no boot)
E34R = R.setdefault("e34", {})
E34R.update(lib=E34.LIB_VERSION, lib_sha256=E34_LIB_SHA, cfg=E34_CFG, switch=E34.switch_record(_e34_on),
            walk=E34.copy_walk(ARMS, NBD1_SWITCH, BUDGET / 60), need_min=E34.copy_need_min(ARMS, NBD1_SWITCH),
            never_do_lifted={"entry": E34.LIFTED, "arms": list(E34.SSM_ARMS), "nb4t_lib_value": T.NEVER_DO_ARG_VALUES.get(E34.SSM_FLAG)},
            cpu_planner=E34.CPU_PLANNER, w_pred=E34.W_PRED,
            question="does the GDN SSM state in bf16 (S16), with block 2048 + drafter SW 512 (S16B2) and more sequences (S16B2W), keep "
                     "B's accuracy (teacher-forced logprobs on 69k-113k recorded thinking traces, greedy oracle, long oracle) and add "
                     "resident streams / throughput at 8k-116k (raw numbers, labels record-only)?",
            arms={k: {x: a.get(x) for x in ("name", "port", "d1_change", "e34", "spec", "captures", "streams", "windows_fixed", "est_min",
                                             "cap_min", "time_model", "title", "d1_items", "env", "extra", "e34_smoke")} for k, a in ARMS.items()})
NBD1.update(switch=E34R["switch"], walk=E34R["walk"], need_min=E34R["need_min"], arms=E34R["arms"], e34=True)


def _e34_tf_long():
    """The recorded thinking traces (dataset foysalemonshanto/q27-e34-tf-long, sha-pinned); None when absent (recorded)."""
    for d in (INP / "datasets/foysalemonshanto/q27-e34-tf-long", INP / "q27-e34-tf-long", Path(os.environ.get("E34_TF_LONG_DIR") or "/nonexistent")):
        f = d / E34.TF_LONG_FILE
        if f.is_file() and hashlib.sha256(f.read_bytes()).hexdigest() == E34_CFG["tf_long_sha256"]:
            return json.loads(f.read_bytes().decode("utf-8"))
    return None


print("E34 arm switches:", NBD1_SWITCH, "| walk:", json.dumps(E34R["walk"])[:700], flush=True)
ISO = {}                                    # "<K>_ISO": the Q-defer isolation rerun (built when a D arm fails)
_AKEYS = ("name", "port", "env", "change", "expect", "expect_cfg", "extra", "kv_bytes", "k", "spec", "captures", "coverage",
          "ring_needed", "ring_capacity_expected", "streams", "windows", "long_bands", "long_aa", "est_min", "cap_min", "title", "optional")

def _k100_next_pin(kvb):
    """The next lower rung of the K100 pin ladder (None below the last one)."""
    low = [x for x in [int(x) for x in K100_CFG["pin_ladder"]] if x < int(kvb)]
    return max(low) if low else None


def _k100_retry(arm, is_d, n_oom):
    """A's B3 (coordinator decision 4): the n-th launch OOM of an arm -> (pin, env added) or None. D: DFLASH_DRAFT_EMBED_HOST=1
    at the SAME pin first, then 7,665,090,560 -> 7,396,655,104 -> 6,859,784,192 (never the -1 GiB step first); Q: one rung."""
    if is_d:
        plan = D1.retry_plan(K100_CFG["d_retry"], int(K100_CFG["q_pin"]), bool(arm.get("d1_eh_first")))   # E25 NB-D1: EMBED_HOST from the first try
        return (int(plan[n_oom - 1][0]), dict(plan[n_oom - 1][1])) if 1 <= n_oom <= len(plan) else None
    ql = K100_CFG["q_ladder"]
    return (int(ql[n_oom]), {}) if 1 <= n_oom < len(ql) else None


# ---- E27 K100 (coordinator 2026-11-01, final design): Q = E28 production (I8), Q's A/A, the TF envelope, deferred reruns
Q_I8 = {"want": bool(K100_CFG["q_i8"]), "on": bool(K100_CFG["q_i8"]), "dropped": None}
NB4T_BUILD_ENV = {}                         # per launch name: DRAFT_HEAD=int8 into Merlin's build() (Q only, while Q_I8 is on)
Q_AA = {}                                   # Q's own A/A: greedy oracle twice (cache reset between) + its ladder
Q_TF_AA = {}                                # Q's teacher-forced A/A per band (fresh vs repeat on the same boot)
PENDING = []                                # (kind, arm) reruns deferred to the end: nometa / iso / tap (D4s always runs first)
NBDF.update(k100_cfg=K100_CFG, q_i8=Q_I8, q_aa=Q_AA, q_tf_aa=Q_TF_AA, pending=PENDING)
if "20k" not in BANDS:                      # the production-level window 20k@24 (K3_ONLY baseline 562 tok/s)
    BANDS["20k"] = 20000
    PROMPTS["20k"] = [prompt(BANDS["20k"], v) for v in range(int(K100_CFG["seqs"]))]
    MAX_NEW["20k"] = MAX_LEN - BANDS["20k"] - 512
TF_P = {b: PROMPTS[b][5] for b in K100_CFG["tf_bands"]}     # teacher-forced 2k / 8k / 40k (RC2's TF variant 5)

if len(PROMPTS["20k"]) < 32:                # E25 NB-D1: S32 at 20k needs 32 distinct prompts (variants 24-31 added)
    PROMPTS["20k"] = list(PROMPTS["20k"]) + [prompt(BANDS["20k"], v) for v in range(len(PROMPTS["20k"]), 32)]

for _b, _n in (("20k", E34.W_MAX), ("40k", E34.W_MAX), ("90k", 24), ("116k", 24)):   # E34: distinct prompts for the W windows
    if len(PROMPTS[_b]) < _n:
        PROMPTS[_b] = list(PROMPTS[_b]) + [prompt(BANDS[_b], v) for v in range(len(PROMPTS[_b]), _n)]
E34_TF_LONG = _e34_tf_long()
TF_P = E34.tf_set(PROMPTS, E34_TF_LONG, smoke=bool(NBD1_SWITCH.get("SMOKE")))   # E34: NBDF's TF code on this set
K100_CFG["tf_bands"] = list(TF_P)            # N.tf_gate reads the band names (recorded only)
E34R["tf"] = {"bands": {b: len(ids) for b, ids in TF_P.items()}, "long_dataset": E34_TF_LONG is not None,
              "long_traces": {k: {x: v.get(x) for x in ("task", "prompt_tokens", "p1_tokens", "len")}
                              for k, v in ((E34_TF_LONG or {}).get("traces") or {}).items()},
              "why_missing": None if E34_TF_LONG is not None else "TF_LONG_DATASET_ABSENT: dataset " + E34.TF_LONG_DS + " not mounted / sha mismatch"}
print("E34 TF set:", E34R["tf"]["bands"], "| long traces mounted:", E34R["tf"]["long_dataset"], flush=True)


def _k100_residents(o):
    """Capacity facts (coordinator decision 8): the boot's resident bound at 40k / 64k / 90k / 120k with the target block 4096
    (the MIXED path when the engine block is 1024), F at the boot's own block, the pool; Q's reference 29 / 21 / 17 / 13."""
    cap = o.get("capacity_after_ready") or {}
    pool = cap["num_gpu_blocks"][0] if len(cap.get("num_gpu_blocks") or []) == 1 else None
    blk = o.get("engine_block") or (((o.get("kv_layout") or {}).get("block_sizes") or {}).get("FullAttentionSpec") or [N.BLOCK_TOKENS])[0]
    out = {"pool": pool, "engine_block": blk, "mc_131072": cap.get("max_concurrency_131072"), "q_reference": {"40k": 29, "64k": 21, "90k": 17, "120k": 13}}
    for nm, tok, band in (("40k", 40000, "40k"), ("64k", 65536, "90k"), ("90k", 90000, "90k"), ("120k", 120000, "116k")):
        try:
            lv, inf = N.level_for_band(band, tok, pool, cap.get("max_concurrency_131072"), int(blk or N.BLOCK_TOKENS), o.get("kv_layout"),
                                       1e9, 1.0, seqs=64)
            out[nm] = {"level": lv, "F": inf.get("F"), "block": inf.get("block"), "source": inf.get("source"), "mixed": inf.get("mixed")}
        except Exception as e:
            out[nm] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return out


def _k100_tap_arm(a):
    """The tap-mode diagnostic (research B): the D arm with DFLASH_TAP_MODE=<alt>, smoke + oracle + 8k only (end of run)."""
    return N.variant_arm(a, "_TAP", "_tap", 9, env_add={"DFLASH_TAP_MODE": K100_CFG["tap_mode_alt"]},
                         title_add=f"DFLASH_TAP_MODE={K100_CFG['tap_mode_alt']} (diagnostic smoke, budget permitting)",
                         short_only=True, est_min=int(K100_CFG["tap_est_min"]), cap_min=int(K100_CFG["tap_cap_min"]))


def _k100_run_pending(runner):
    """Coordinator decision 1: no rerun may starve D4s; the nometa / Q-defer-off / tap-mode reruns run after every arm, each
    only when its full minimum still fits (else SKIPPED_BUDGET, recorded)."""
    isos = 0
    for kind, k in list(PENDING):
        a = ARMS[k]
        if kind == "iso":
            if isos >= N.ISO_MAX:
                continue
            isos += 1
            na = N.iso_arm(a)
        elif kind == "nometa":
            na = N.nometa_arm(a)
        else:
            na = _k100_tap_arm(a)
        if na["key"] in ISO:
            continue
        ISO[na["key"]] = na
        NBDF.setdefault("variants", {})[na["key"]] = {x: na.get(x) for x in _AKEYS}
        if N.arm_fits(left(), na):
            step(na["key"], na["name"], na["est_min"], lambda kk=na["key"]: runner(kk))
        else:
            skip_arm(na["key"], f"left {left() / 60:.1f} min < {na['est_min']} + 5 min", "SKIPPED_BUDGET")
NBDF["arms"] = {k: {x: a.get(x) for x in _AKEYS} for k, a in ARMS.items()}
NBDF["order"] = list(ARMS)                  # E25 NB-D1
NBDF["oom_policy"] = ("Q: NB4T's (a launch OOM -> one retry 0.25 GiB lower); D arms: start on the pin ladder below Q (pin_plan, at most "
                      "2 rungs), a launch OOM or a rank below 512 MiB free (after ready / after the first decode) -> one rung lower, "
                      f"up to {N.D_OOM_RETRIES} more; a later D arm starts at the pin that worked; a body OOM never retries")
NBDF["never_do"] = {"env": sorted(T.NEVER_DO_ENV), "argv": list(T.NEVER_DO_ARGS), "argv_values": T.NEVER_DO_ARG_VALUES}
NBDF["budget"] = {"hard_stop_min": BUDGET / 60, "start_reserve_sec": N.START_RESERVE_SEC, "body_reserve_sec": N.BODY_RESERVE_SEC,
                  "priority": " > ".join(ARMS) + " (switched-on arms; an arm starts only with est_min x 60 + 300 s left)",   # E25 NB-D1
                  "planned_walk": E34.copy_walk(ARMS, NBD1_SWITCH, BUDGET / 60)}   # E34
ARM_OF = {}                                 # launch name (retries included) -> arm
PIN = {"kv_bytes": B3_PIN, "source": "E27 K100: K100Q / E28 top K3 rung"}
D_PIN = {"kv_bytes": None, "plan": None, "source": None}
SW_PIN = {"kv_bytes": None, "source": None}   # SW arms (D4s / D3s / D4t): the D pin (Q's pin - 2 rungs) first, then one rung down
NBDF_READY_SEC = 20 * 60                    # readiness gate per boot (NB4T: 627 s first compiling boot, 264-282 s later)
_HIST = {}
LONG_ORACLE_P = [PROMPTS["90k"][N.LONG_VARIANT], PROMPTS["116k"][N.LONG_VARIANT]]
PREFIX_PROMPTS = {v: prompt(N.PREFIX_P_TOKENS, v) for v in N.PREFIX_VARIANTS}   # 2 prompts x 3 turns (no window uses them)
PREFIX_FILLER = prompt(BANDS["8k"], 42)     # the turn-2 user message body, sized so turn 2's generation completes a block
Q_ACC = {}                                  # Q's greedy oracle acceptance (a diagnostic reference only, GPT audit B2)
Q_REC = {}                                  # Q's record (the 8k throughput rule of SHORT_EXIT_ARMS reads its windows)


def _tree_py():
    eng = merlin_install()
    return Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python"


def _ppack_apply():
    tree, py = _tree_py()
    return P.apply_ppack_patch(tree, py)


def _ppack_revert():
    tree, py = _tree_py()
    rec = P.revert_ppack_patch(tree, py)
    v = P.verify_ppack_patch(tree)
    if v["state"] != "not_applied":
        raise RuntimeError("REVERT_NOT_PRISTINE: " + json.dumps(v)[:400])
    rec["verify_after"] = v
    return rec


def _seam_apply(ring):
    tree, py = _tree_py()
    fn = SEAM_API["fns"]["apply"]
    rec = fn(tree, py, ring=ring) if SEAM_API["caps"]["ring_kw"] else fn(tree, py)
    return rec if isinstance(rec, dict) else {"returned": repr(rec)[:200]}


def _seam_verify(ring):
    tree, _ = _tree_py()
    fn = SEAM_API["fns"]["verify"]
    v = fn(tree, ring=ring) if SEAM_API["caps"]["ring_kw"] else fn(tree)
    return v if isinstance(v, dict) else {"state": str(v)}


def _seam_revert(ring):
    tree, py = _tree_py()
    fn = SEAM_API["fns"]["revert"]
    rec = fn(tree, py, ring=ring) if SEAM_API["caps"]["ring_kw"] else fn(tree, py)
    rec = rec if isinstance(rec, dict) else {"returned": repr(rec)[:200]}
    v = _seam_verify(ring)
    if N.tree_state(v) != "ORIGINAL":
        raise RuntimeError("SEAM_REVERT_NOT_PRISTINE: " + json.dumps(v, default=str)[:400])
    rec["verify_after"] = v
    rec["ppack_after"] = P.verify_ppack_patch(tree)["state"]       # nb4_ppack must be intact (still applied) after the seam revert
    return rec


PSEQ = T.PatchSeq(_ppack_apply, _ppack_revert)
SSEQ = N.SeamSeq(_seam_apply, _seam_verify, _seam_revert)


def tree_check(arm):
    """Three-state check before every boot (GPT analysis 2): nb4_ppack FULLY_PATCHED, dflash_seam FULLY_PATCHED (or ORIGINAL when
    it never applied: Q only); INVALID_MIXED anywhere aborts the D arms."""
    if FAKE:
        return {"ok": True, "fake": True}
    tree, _ = _tree_py()
    pp = N.tree_state(P.verify_ppack_patch(tree))
    if SEAM_API is not None and SEAM_API["caps"].get("tree_state") and SSEQ.state == "applied":
        try:
            NBDF.setdefault("combined_tree_state", {})[arm["key"]] = DS.combined_tree_state(tree, P, ring=SSEQ.ring)
        except Exception as e:
            NBDF.setdefault("combined_tree_state", {})[arm["key"]] = repr(e)[:200]
    sm = None
    if SEAM_API is not None and SSEQ.state == "applied":
        try:
            sm = N.tree_state(_seam_verify(SSEQ.ring))
        except Exception as e:
            sm = "INVALID_MIXED"
            R["errors"].append(f"seam verify: {e!r}"[:200])
    ok = pp == "FULLY_PATCHED" and (sm in (None, "FULLY_PATCHED")) and (arm["key"] == "Q" or sm == "FULLY_PATCHED")
    return {"ok": ok, "ppack": pp, "seam": sm, "seam_seq": SSEQ.state}


def _prom_nbdf(port):
    """e23_lab.prom (same dict, one HTTP read) + the iteration-token histogram buckets with their read time (NB4T)."""
    txt = L.U.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=15).read().decode("utf-8", "replace")
    agg, buckets = T.prom_parse(txt)
    h = _HIST.setdefault(port, [])
    h.append((time.perf_counter(), buckets))
    del h[:-600]
    return agg


L.prom = _prom_nbdf


def _hist_at(port, t):
    best = None
    for ts, b in _HIST.get(port, []):
        if ts <= t + 0.01:
            best = (ts, b)
    return best if best and t - best[0] < 1.0 else None


def _prefix_counters(port):
    try:
        m = L.prom(port)
        return {"queries": m.get("vllm:prefix_cache_queries_total"), "hits": m.get("vllm:prefix_cache_hits_total"),
                "prompt": m.get(L.PROMPT)}
    except Exception:
        return {}


def _prefix_delta(c0, c1):
    if not c0 or not c1 or None in (c0.get("queries"), c1.get("queries"), c0.get("hits"), c1.get("hits")):
        return {"rate": None}
    q, h = c1["queries"] - c0["queries"], c1["hits"] - c0["hits"]
    return {"queries": q, "hits": h, "rate": round(h / q, 4) if q > 0 else None}


def _mk_merlin_fn_server(prev):
    """prev = RC2 CELL 2 + NB1's per-boot hooks (+ NB4T's build-knob line, unused here); wrapped ONCE (nbdf_lib.wrap_once:
    a second execution of this cell must not wrap the wrapper)."""
    def merlin_fn_server(name, kv, port, kv_bytes):
        """E27 NBDF: NB1's Merlin plan, then (fail closed: no server is started) nbdf_lib.launch_check on the FULL final env /
        argv of THIS launch: never-do, watched env (DFLASH_* only on D arms), the effective spec + captures, seqs, pin."""
        return _nbdf_launch_check(prev(name, kv, port, kv_bytes), name, port, kv_bytes)
    return merlin_fn_server


def _nbdf_launch_check(s, name, port, kv_bytes):
    arm = ARM_OF.get(name)
    popped = {}
    for w, want in ((arm or {}).get("expect") or {}).items():
        if want is None and w in s.env:
            popped[w] = s.env.pop(w)
    bad = E34.launch_check(N, T, arm, s.env, s.argv, kv_bytes) if arm else [f"launch {name} has no arm"]   # E34: NBDF's check minus exactly the lifted SSM bf16 entry + the SSM / block / SW plan
    NBDF.setdefault("launch", {})[name] = {
        "arm": (arm or {}).get("key"), "port": port, "kv_bytes": kv_bytes, "never_do_and_plan": bad,
        "spec": T.spec_from_argv(s.argv), "captures": N.captures_from_argv(s.argv), "max_num_seqs": T.last_value(s.argv, "--max-num-seqs"),
        "env": {k: s.env[k] for k in sorted(s.env) if k.startswith(("CUDA_", "NCCL_", "NB4_", "VLLM_SERVER_DEV", "VLLM_USE_V2",
                                                                    "VLLM_PP_LAYER_PARTITION", "DFLASH", "P2P"))},
        "argv_tail": s.argv[-16:], "help": HELP.get(name), "popped_unplanned_env": popped}
    save()
    if bad:
        raise RuntimeError("NBDF_LAUNCH_CHECK: " + "; ".join(bad))
    return s


merlin_fn_server = N.wrap_once(globals(), "merlin_fn_server", _mk_merlin_fn_server)


def free_min(port):
    smi = L.gpu_mem() or []
    w = rpc_cache_info(port)
    wf = w.get("free_mib_per_rank") if isinstance(w, dict) else None
    sf = [t_ - u for _, u, t_ in smi] if smi else None
    mins = None
    if sf and wf and len(sf) == len(wf):
        mins = [min(a, b) if isinstance(b, (int, float)) else a for a, b in zip(sf, wf)]
    return {"nvidia_smi": sf, "worker": wf, "min": mins or sf or wf}


def proc_environ():
    blobs = {}
    for pid, _ in (engine_procs() or [])[:24]:
        try:
            blobs[pid] = Path(f"/proc/{pid}/environ").read_bytes()
        except Exception:
            blobs[pid] = None
    return blobs


def _live(o, key, val):
    o[key] = val
    rec = LIVE.get("rec")
    if rec is not None:
        rec.setdefault("live", {})[key] = val
    e34_item_live(o, key, val)                    # E34: a [LIVE] line after every test (print only)
    save()
    return val


def q_readback(s, arm):
    """nb4_ppack's 4 'NB4_PPQ init' lines + the Q-defer exec / fallback share per rank (every arm but the isolation rerun)."""
    if not arm.get("q_label"):
        return {"label": "NOT_APPLICABLE", "detail": {"why": "Q-defer off (isolation rerun)"}}
    try:
        lab, det = P.readback_label(s.log() or "", arm["q_label"])
    except Exception as e:
        lab, det = "NOT_INTEGRATED", {"error": f"{type(e).__name__}: {e}"[:300]}
    shares = {pid: {"exec_share": v.get("exec_share"), "fallback_share": v.get("fallback_share")}
              for pid, v in ((det or {}).get("defer_per_rank") or {}).items()}
    return {"label": lab, "detail": det, "shares": shares}


def _wall(perf_t):
    """perf_counter time -> the server log's 'MM-DD HH:MM:SS' clock (same container, local time)."""
    return time.strftime("%m-%d %H:%M:%S", time.localtime(time.time() - (time.perf_counter() - perf_t)))


def hist_window(port, s, band, n, arm):
    """NB1's window() + the micro-batch split (iteration-token histogram, the arm's K), the SpecDecoding lines inside the
    window, the prefix-hit rate, and the garbage review of the window's COMPLETE rows (windowed check gates)."""
    c0 = _prefix_counters(port)
    RESP["rows"] = []
    try:
        with D1Ledger(port) as _d1l:              # E25 NB-D1: the window's KV ledger samples (records only)
            r = window(port, s, band, n)
        r["_d1_samples"] = _d1l.samples
    finally:
        rows, RESP["rows"] = RESP["rows"] or [], None
    c1 = _prefix_counters(port)
    h0, h1 = (_hist_at(port, r["t0"]) if r.get("t0") else None), (_hist_at(port, r["t1"]) if r.get("t1") else None)
    d = T.hist_delta(h0[1], h1[1]) if h0 and h1 else None
    r["parity"] = T.parity_from_hist(d, n, PP, arm["k"])
    r["level"] = r.get("level") or n
    pd = _prefix_delta(c0, c1)
    r["prefix_hits"], r["prefix_hit_rate"] = pd, pd.get("rate")
    if r.get("t0") and r.get("t1"):
        r["wall_t0"], r["wall_t1"] = _wall(r["t0"]), _wall(r["t1"])
        r["spec_log"] = N.spec_summary(N.spec_decoding_lines(s.log() or ""), arm["k"], r["wall_t0"], r["wall_t1"])
    N.apply_windowed_gate(r, N.garbage_review(rows, TOK.decode, L.degenerate), n)
    r["responses_saved"] = len(rows)
    return r


def _trace_spans(d):
    """Graph node-count modes + median span per rank (the drafter's FULL_DECODE_ONLY replay vs the target's verify graph)."""
    by = {}
    for k_, f in enumerate(sorted(Path(d).rglob("*.pt.trace.json*"))):
        tr = T._open(f)
        rank, _, _ = T._rank_of(f.name, tr, k_)
        by[rank] = T.rank_graphs(tr)[0]
    return N.graph_spans(by)


def profile_window(port, arm, band="8k", n=None, secs=6.0):   # E25 NB-D1: secs (PROF 30 s)
    """A profiled window (NB4T): lab profiled_load, bounded POST /start_profile, traces kept per arm and band."""
    n = n or arm["streams"]
    shutil.rmtree(PROF_DIR, ignore_errors=True)
    PROF_DIR.mkdir(parents=True, exist_ok=True)
    rec = L.profiled_load(port, PROMPTS[band], n, PROF_DIR, secs=float(secs), max_new=MAX_NEW[band], spec=True, prefill_tok_s=PREFILL["tok_s"])
    rec["profile_request"] = {"delay_iterations": 20, "max_iterations": 40, "secs": float(secs), "level": n, "band": band}
    if "trace" in rec:
        rec["lab_trace_summary"] = rec.pop("trace")
    kept = TRACES / f"profile_{arm['key'].lower()}_{band}"
    shutil.rmtree(kept, ignore_errors=True)
    kept.mkdir(parents=True, exist_ok=True)
    rec["trace_files"] = []
    for f in sorted(PROF_DIR.rglob("*.pt.trace.json*")):
        if f.stat().st_size < 120e6:
            shutil.copy2(f, kept / f.name)
            rec["trace_files"].append([f.name, f.stat().st_size])
    rec["trace_dir"] = str(kept)
    shutil.rmtree(PROF_DIR, ignore_errors=True)
    return rec


def win_need(band, n):
    if band.startswith("profile"):
        b = band.replace("profile_", "")
        return 360 + 10 * max(0, n - 8) + n * BANDS.get(b, 8192) / max(500.0, PREFILL["tok_s"])
    return WIN + 120 + n * BANDS[band] / max(500.0, PREFILL["tok_s"])


def reset_prefix(port):
    """POST /reset_prefix_cache (VLLM_SERVER_DEV_MODE=1) after the engine is idle."""
    out = {"idle": L.wait_idle(port, 90)}
    try:
        req = L.U.Request(f"http://127.0.0.1:{port}/reset_prefix_cache", data=b"", method="POST")
        r = L.U.urlopen(req, timeout=120)
        out.update(ok=200 <= r.status < 300, status=r.status)
    except Exception as e:
        out.update(ok=False, error=f"{type(e).__name__}: {e}"[:200])
    try:
        out["usage_after"] = kvu(port)
    except Exception:
        pass
    return out


def _one(port, ids, n_tok, temperature=0.0, logprobs=2, seed=None, timeout=900):
    g = L.Stream(port, ids, n_tok, temperature=temperature, seed=seed, logprobs=logprobs, ignore_eos=True)
    g.start()
    g.join(timeout)
    if g.is_alive():
        g.stop()
        g.join(10)
        r = g.row()
        r["error"] = r["error"] or "TIMEOUT"
        return r
    return g.row()


def smoke(port, arm):
    """Fail-fast smoke right after the boot: 2 sampled streams x 2,048 tokens (acceptance > ~0, windowed garbage) + the
    concurrency contamination ladder: one greedy probe alone, then next to 2 / 4 / 7 other sampled requests (7 + the probe = 8
    = max_num_seqs): identical token ids, else NEAR_TIE (recorded) or CONCURRENCY_CONTAMINATION. Per step: the probe's
    token sha, the drafted / accepted deltas and the prefix-hit delta."""
    k = arm["k"]
    out = {"streams": N.SMOKE_STREAMS, "max_new": N.SMOKE_TOKENS}
    a0, c0 = accept_pos(port), _prefix_counters(port)
    RESP["rows"] = []
    ss = [L.Stream(port, PROMPTS["2k"][2 + i], N.SMOKE_TOKENS, temperature=TEMP, seed=600 + i, ignore_eos=True) for i in range(N.SMOKE_STREAMS)]
    for g in ss:
        g.start()
    for g in ss:
        g.join(600)
    for g in ss:
        if g.is_alive():
            g.stop()
            g.join(10)
    time.sleep(1.0)
    rows, RESP["rows"] = RESP["rows"] or [], None
    out["acceptance"] = N.acceptance_delta(a0, accept_pos(port), k)
    out["prefix"] = _prefix_delta(c0, _prefix_counters(port))
    out["tokens"] = [len(g.tokens) for g in ss]
    out["errors"] = sorted({g.err for g in ss if g.err})[:3]
    out["garbage"] = N.garbage_review(rows, TOK.decode, L.degenerate)
    lad = {}
    probe = PROMPTS["2k"][0]
    a0, c0 = accept_pos(port), _prefix_counters(port)
    alone = _one(port, probe, N.CONC_TOKENS)
    lad["alone"] = {"error": alone.get("error"), "tokens_sha": hashlib.sha256(json.dumps(alone.get("tokens")).encode()).hexdigest()[:16],
                    "acceptance": N.acceptance_delta(a0, accept_pos(port), k), "prefix": _prefix_delta(c0, _prefix_counters(port))}
    for c in (N.CONC_LADDER if D1.item_on(arm, "ladder") else ()):   # E25 NB-D1 light: no contamination ladder
        others = [L.Stream(port, PROMPTS["2k"][1 + (i % 7)], 1024, temperature=TEMP, seed=700 + 10 * c + i, ignore_eos=True) for i in range(c)]
        for g in others:
            g.start()
        t_ = time.time()
        while time.time() - t_ < 60 and sum(g.t_first is not None for g in others) < c:
            time.sleep(0.5)
        a0, c0 = accept_pos(port), _prefix_counters(port)
        row = _one(port, probe, N.CONC_TOKENS)
        a1, c1 = accept_pos(port), _prefix_counters(port)
        for g in others:
            g.stop()
        for g in others:
            g.join(10)
        L.wait_idle(port, 60)
        lad[str(c)] = {"others": c, "others_decoding": sum(g.t_first is not None for g in others), "error": row.get("error"),
                       "tokens_sha": hashlib.sha256(json.dumps(row.get("tokens")).encode()).hexdigest()[:16],
                       "vs_alone": L.compare([alone], [row]), "acceptance": N.acceptance_delta(a0, a1, k), "prefix": _prefix_delta(c0, c1)}
    out["ladder"] = lad
    out["ladder_gate"] = N.contamination_gate(lad)
    out["ladder_brief"] = {kk: {"sha": v.get("tokens_sha"), "status": (v.get("vs_alone") or {}).get("status")} for kk, v in lad.items()}
    return out


def _measured(port, ids, n_tok, k):
    a0, c0 = accept_pos(port), _prefix_counters(port)
    r = _one(port, ids, n_tok)
    a1, c1 = accept_pos(port), _prefix_counters(port)
    pd = _prefix_delta(c0, c1)
    return r, {"acceptance": N.acceptance_delta(a0, a1, k), "prefix": pd, "prefix_hits": (pd.get("hits") or 0) > 0,
               "error": r.get("error"), "tokens": len(r.get("tokens") or []),
               "tokens_sha": hashlib.sha256(json.dumps(r.get("tokens")).encode()).hexdigest()[:16]}


def prefix_test(port, arm, block=4096):
    """Cold vs warm acceptance, 2 prompts x 3 turns (GPT analyses 2 / 3; upstream #47930 / #47926 / #58894). Per prompt (6,144
    tokens): WARM = the cache reset, turn 1 (greedy 2,048: block 2 completes with KV the speculative steps wrote), turn 2 =
    prompt + turn 1 + the commit suffix + a filler user message sized so its 512 greedy tokens complete the next block,
    turn 3 = turn 2 + its output + the suffix (greedy 256: reuses that block). COLD = the cache reset, then turn 2 alone;
    the cache reset, turn 3 alone. Per turn: AL / per position, drafted / accepted, prefix hits, token ids (cold vs warm)."""
    k = arm["k"]
    out = {"block": block, "prompts": []}
    for v, P0 in PREFIX_PROMPTS.items():
        pr = {"variant": v, "turns": {}}
        try:
            pr["reset_warm"] = reset_prefix(port)
            t1, m1 = _measured(port, P0, N.PREFIX_T1, k)
            if t1.get("error") or not t1.get("tokens"):
                pr["error"] = f"turn 1: {t1.get('error')}"
                out["prompts"].append(pr)
                continue
            head = list(P0) + list(t1["tokens"]) + list(SUFFIX_IDS)
            need = max(0, (-(-len(head) // block) + 1) * block - N.PREFIX_T2 - len(head))
            turn2 = head + list(PREFIX_FILLER[:need]) + list(SUFFIX_IDS)
            t2, m2 = _measured(port, turn2, N.PREFIX_T2, k)
            turn3 = turn2 + list(t2.get("tokens") or []) + list(SUFFIX_IDS)
            t3, m3 = _measured(port, turn3, N.PREFIX_T3, k)
            pr.update(turn1=m1, prompt_tokens={"1": len(P0), "2": len(turn2), "3": len(turn3)})
            pr["turns"]["2"], pr["turns"]["3"] = {"warm": m2}, {"warm": m3}
            for t, ids, n_tok, wr in (("2", turn2, N.PREFIX_T2, t2), ("3", turn3, N.PREFIX_T3, t3)):
                pr["turns"][t]["reset_cold"] = reset_prefix(port)
                cr, mc = _measured(port, ids, n_tok, k)
                pr["turns"][t]["cold"] = mc
                pr["turns"][t]["warm_vs_cold"] = L.compare([cr], [wr])
        except Exception as e:
            pr["error"] = f"{type(e).__name__}: {e}"[:300]
        out["prompts"].append(pr)
    return out


def long_oracle(port, key=None, ref=None, k=3):
    """Greedy LONG_NTOK tokens (ignore_eos) on the 90k + 116k prompts (variant 7). Q stores the reference (complete +
    garbage-free only); a D arm is compared with it by token ids (L.compare: EXACT / NEAR_TIE / FAIL_HIGH_MARGIN)."""
    a0 = accept_pos(port)
    rows = L.greedy(port, LONG_ORACLE_P, n_tok=N.LONG_NTOK, timeout=max(300, min(1500, boot_left() - 120)))
    garbage = [L.degenerate(r.get("text")) for r in rows]
    complete = len(rows) == len(LONG_ORACLE_P) and all(not r.get("error") and len(r.get("tokens") or []) == N.LONG_NTOK for r in rows)
    if key and complete and not any(garbage):
        ROWS[key] = rows
    elif key:
        R["errors"].append(f"long oracle {key}: reference not stored (incomplete / error / garbage)")
    res = {"bands": ["90k", "116k"], "prompt_tokens": [len(p) for p in LONG_ORACLE_P], "n_tok": N.LONG_NTOK, "garbage": garbage,
           "errors": [r.get("error") for r in rows], "tokens": [len(r.get("tokens") or []) for r in rows], "complete": complete,
           "text_head": [(r.get("text") or "")[:160] for r in rows], "acceptance": N.acceptance_delta(a0, accept_pos(port), k)}
    if ref:
        res["vs_" + ref] = L.compare(ROWS.get(ref), rows) if ROWS.get(ref) else {"status": "INCONCLUSIVE_NO_ORACLE"}
    return res


RPC_KEYS = ",".join(N.WATCHED) + ",NCCL_P2P_DISABLE,P2P," + ",".join(D1.D1_ENV_KEYS)   # E25 NB-D1: + the QSA / GDN keys (4-worker read-back)


def band_level(o, band):
    """nbdf_lib.level_for_band on THIS boot's pool (num_gpu_blocks), max concurrency (its log) and block size."""
    cap = o.get("capacity_after_ready") or {}
    pool = cap["num_gpu_blocks"][0] if len(cap.get("num_gpu_blocks") or []) == 1 else None
    blk = o.get("engine_block") or (((o.get("kv_layout") or {}).get("block_sizes") or {}).get("FullAttentionSpec") or [N.BLOCK_TOKENS])[0]
    return N.level_for_band(band, BANDS[band], pool, cap.get("max_concurrency_131072"), int(blk or N.BLOCK_TOKENS), o.get("kv_layout"),
                            PREFILL["tok_s"], WIN, seqs=int(o.get("streams") or N.SEQS))   # E25 NB-D1: the arm's own max_num_seqs


class _watched:
    """The admission watchdog around one item of a D arm: a daemon thread polls /metrics every 5 s; running == 0 with
    waiting > 0 for > 120 s -> ADMISSION_DEADLOCK on the arm record (running / waiting / KV usage) and every live stream
    of the item is stopped (a bounded stop); the arm's remaining long items are then skipped."""

    def __init__(self, o, port, label, active=True):
        self.o, self.port, self.label, self.active = o, port, label, active
        self.watch, self.stop_ev, self.th = None, threading.Event(), None

    def _fire(self, info):
        self.o["admission_deadlock"] = dict(info, arm=self.o.get("arm"))
        for g in list(ACTIVE_STREAMS):
            try:
                g.stop()
            except Exception:
                pass

    def _loop(self):
        t0 = time.perf_counter()
        while not self.stop_ev.wait(5.0):
            try:
                m = L.prom(self.port)
            except Exception:
                continue
            if self.watch.feed(time.perf_counter() - t0, int(m.get(L.RUN, 0)), int(m.get(L.WAIT, 0)), m.get(L.KVU)):
                break

    def __enter__(self):
        if self.active:
            self.watch = N.AdmissionWatch(on_fire=self._fire, label=self.label)
            self.th = threading.Thread(target=self._loop, daemon=True, name=f"nbdf_admission_{self.label}")
            self.th.start()
        return self

    def __exit__(self, *a):
        self.stop_ev.set()
        if self.th is not None:
            self.th.join(10)
        return False


def admission_probes(o, port):
    """D4s smoke: one single request each at 16k and 40k prompt tokens (64 greedy tokens), each under the watchdog."""
    out = {}
    for n_tok in N.ADMISSION_PROBE_TOKENS:
        if o.get("admission_deadlock"):
            out[str(n_tok)] = {"status": "SKIPPED_ADMISSION_DEADLOCK"}
            continue
        ids = PROMPTS["40k"][6] if n_tok == 40000 else prompt(n_tok, 51)
        t0 = time.perf_counter()
        with _watched(o, port, f"admission_probe_{n_tok}"):
            r = _one(port, ids, 64, timeout=900)
        out[str(n_tok)] = {"prompt_tokens": len(ids), "tokens": len(r.get("tokens") or []), "error": r.get("error"),
                           "sec": round(time.perf_counter() - t0, 1),
                           "status": "WEDGE" if (o.get("admission_deadlock") or {}).get("item") == f"admission_probe_{n_tok}" else
                           ("PASS" if len(r.get("tokens") or []) == 64 and not r.get("error") else "FAIL")}
    return out


def run_window(o, s, arm, band, n, tag=""):
    """One steady window (gated by due()); a KV-bound window steps the level down once (residency confirmed, recorded)."""
    key = f"{band}@{n}{tag}"
    need = win_need(band, n)
    if not due(need):
        o["steady"][key] = {"status": "SKIPPED_BOOT_BUDGET", "need_sec": round(need), "boot_left_sec": round(boot_left())}
        o["items_skipped"].append(key)
        d1_live(o, arm, key, o["steady"][key])     # E25 NB-D1 (print only)
        return None
    if not s.alive():
        o["steady"][key] = {"status": "SKIPPED_SERVER_DEAD"}
        return None
    if o.get("admission_deadlock"):
        o["steady"][f"{band}@{n}{tag}"] = {"status": "SKIPPED_ADMISSION_DEADLOCK", "deadlock": o["admission_deadlock"]}
        o["items_skipped"].append(f"{band}@{n}{tag}")
        return None
    if n <= 0:                                    # a valid F with no room for one stream: recorded, never a fallback level
        o["steady"][f"{band}@0{tag}"] = {"status": "SKIPPED_CAPACITY", "level": 0, "long_level": (o.get("long_levels") or {}).get(band)}
        o["items_skipped"].append(f"{band}@0{tag}")
        return None
    CUR["item"] = key
    try:
        with _watched(o, s.port, f"window_{band}@{n}{tag}", active=arm.get("method") == "dflash"):
            r = hist_window(s.port, s, band, n, arm)
        try:
            r["kv_ledger"] = d1_window_ledger(o, r, band, n)   # E25 NB-D1 (records only)
        except Exception as e:                     # records only: never voids the measured window
            r.pop("_d1_samples", None)
            r["kv_ledger"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        if (o.get("admission_deadlock") or {}).get("item") == f"window_{band}@{n}{tag}":
            r = dict(r, status="ADMISSION_DEADLOCK", deadlock=o["admission_deadlock"])
    except Exception as e:
        r = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1200:]}
    N.capacity_window_status(r, ((o.get("long_levels") or {}).get(band) or {}).get("source") == "CAPACITY_UNVERIFIED")
    o["steady"][key] = r
    _live(o, "steady", o["steady"])
    d1_live(o, arm, key, r)                       # E25 NB-D1: [LIVE <min>] <arm> <window> tok/s | AL | running | preempt | vs B (print only)
    if N.kv_bound(r) and n > 1 and (band in ("40k",) + N.LONG_BANDS or arm.get("lever")) and due(win_need(band, n - 1)):   # E25 NB-D1: a lever steps down on any band (never skipped)
        r["residency"] = f"KV-bound at {n}: one step down to {n - 1}"
        o.setdefault("residency_retry", {})[band] = [n, n - 1]
        return run_window(o, s, arm, band, n - 1, tag)
    return key


def dflash_rb(arm, t):
    """nbdf_lib.dflash_readback + dflash_seam.readback_label (ENGAGED required: aux 4 ranks, embed own + mask row sha, load
    check lines, ring line at K >= 5)."""
    rb = N.dflash_readback(arm, t)
    if SEAM_API and SEAM_API["caps"].get("readback_label"):
        try:
            sig = str(inspect.signature(DS.readback_label))
            kw = {"no_ingraph_meta": arm["env"].get("DFLASH_NO_INGRAPH_META") == "1"} if "no_ingraph_meta" in sig else {}
            if "conv_impl" in sig:
                kw["conv_impl"] = arm["env"].get("DFLASH_CONV_IMPL") or "fused"
            lab = DS.readback_label(t, ring_k=arm["k"] if arm["ring_needed"] else None,
                                    mask_row_sha=arm["env"].get("DFLASH_MASK_ROW_SHA"),
                                    embed_host=arm["env"].get("DFLASH_DRAFT_EMBED_HOST") == "1", **kw)
        except Exception as e:
            lab = ("NOT_INTEGRATED", {"error": repr(e)[:200]})
        rb = N.merge_seam_label(rb, lab)
    return rb


MEM_RX = re.compile(r"Model loading took|Initial free memory|Available KV cache memory|peak|Memory profiling|CUDA graph pool|"
                    r"Graph capturing finished|non-torch|GPU KV cache size|Maximum concurrency")


def _can_step(arm):
    """USER RULE 2026-11-01: the 512 MiB memory gate steps a D arm one rung down only while another launch try remains and
    the budget holds it; otherwise the low reading is recorded and the arm measures on (only a real OOM / crash stops it)."""
    return False                              # E27 K100: the memory gate is a flag (coordinator decision 4)


def _mem_gate(p, o, key):
    """nbdf_lib.memory_gate (fail-closed, 4 ranks); a None reading is retried once after 5 s, then recorded as None
    (no step-down; recorded: MEMORY_GATE_LOW_OR_MISSING flag, never a verdict override)."""
    mg = N.memory_gate((o.get(key) or {}).get("min"))
    if mg[0] is None:
        time.sleep(5)
        o[key] = free_min(p)
        mg = N.memory_gate((o[key] or {}).get("min"))
        o[key + "_retried"] = True
    return mg


def arm_body(arm, ref, store):
    """Every arm, the same items in the same order: facts + read-backs -> memory gate -> smoke (+ ladder) -> memory gate (first
    decode) -> oracle (+ AL, diagnostic) -> long oracle -> prefix cold / warm -> 8k x 2 -> 40k -> 90k -> 116k (Q x 2) ->
    profiled 8k (+ 116k when due); every item gated by due() on the boot deadline."""
    def body(ss):
        s = ss[0]
        p = s.port
        n = arm["streams"]
        is_q, is_d = arm["key"] == D1.BASE_KEY, arm.get("method") == "dflash"   # E25 NB-D1: B (a D arm) is the reference arm
        o = {"arm": arm["key"], "k": arm["k"], "items_skipped": [], "streams": n}
        CUR["item"] = "facts"
        o = boot_facts(s, dict(o, census=kv_census(s), probes=L.chat_probes(p), sampled=sampled_check(s)), s.name)
        LIVE["out"] = o
        CUR["item"] = "readback"
        t = s.log() or ""
        _live(o, "env_readback_raw", rpc(p, "nb4s_env", RPC_KEYS))
        _live(o, "env_readback", T.parse_env_readback(o["env_readback_raw"], arm["expect"]))
        _live(o, "proc_environ", T.parse_proc_environ(proc_environ(), arm["expect"]))
        _live(o, "config_readback", N.config_readback(arm, t, CUR["kv_bytes"]))
        _live(o, "q_readback", q_readback(s, arm))
        _live(o, "kv_layout", N.kv_layout_readback(o.get("workers"), arm))
        _live(o, "graph_counts", N.graph_counts(t))
        if not is_d:                               # E25 NB-D1: every arm is a D arm (B included)
            _live(o, "seam_inert", N.q_seam_inert(t))
        else:
            _live(o, "dflash_readback", dflash_rb(arm, t))
            _live(o, "ingraph_meta", N.ingraph_meta(t))
            o["kv_kit_summary"] = (KV_KIT_GATE.get("kv_summary") or {}).get(arm.get("parent") or arm["key"])
        o["fine_grained_disabled"] = N.FINE_GRAINED_RX in t
        o["engine_block"] = N.engine_block(t, o.get("kv_layout"))
        o["kv_draft_group_warning"] = N.DRAFT_GROUP_WARN_RX in t
        o["stage_graphs"] = N.stage_graphs(t)
        _live(o, "free_after_ready_min", free_min(p))
        _live(o, "capacity_after_ready", T.capacity_facts(dict(o, pin=CUR["kv_bytes"])))
        o["k100_residents"] = _k100_residents(o)              # E27 K100: residents at 40k / 64k / 90k / 120k (block 4096)
        e34_capacity_rec(o, arm, t)                           # E34: boot capacity + dtype / block / SW read-back (records, printed)
        o["memory_log_lines"] = [l[:260] for l in t.splitlines() if MEM_RX.search(l)][:40]
        _live(o, "memory_stages", N.memory_stages(o))
        mg = _mem_gate(p, o, "free_after_ready_min")
        o["memory_gate_ready"] = {"ok": mg[0], "why": mg[1]}
        if is_d and mg[0] is False:
            if _can_step(arm):
                o.update(status="MEMORY_GATE_STEP_DOWN", early_stop="after ready: " + mg[1])
                return o
            o.setdefault("recorded_gates", []).append("memory gate after ready (no rung / budget left, arm continues): " + mg[1])
        if short_gate_failed(o):
            o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s),
                     quality_gate="FAIL_SAMPLED" if (o["sampled"] or {}).get("ok") is not True else "FAIL_PROBES")
            return o
        sm_ = N.spec_mismatch(is_d, o.get("config_readback"), o.get("dflash_readback"))
        if sm_[0]:                                # spec method / K / drafter not the plan: this arm only (the others run)
            o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop="SPEC_MISMATCH: " + sm_[1][:400])
            return o
        try:                                       # E25 NB-D1: the arm's planned settings read back (a real failure stops this arm only)
            _d1f = d1_readback_failure(o, arm, t)
        except Exception as e:                     # a read-back code error is recorded, never a stop
            _d1f, o["d1_readback_error"] = None, f"{type(e).__name__}: {e}"[:300]
        if _d1f:
            o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop=_d1f)
            return o
        for nm in ("env_readback", "config_readback", "dflash_readback"):
            if (o.get(nm) or {}).get("definitive"):      # recorded (USER RULE 2026-11-01): the arm still measures
                o.setdefault("recorded_gates", []).append(f"{nm} contradicts the plan: " + str(o[nm].get("why"))[:300])
                log("NBDF", arm["key"], nm, "contradiction recorded (arm continues):", str(o[nm].get("why"))[:200])
        if arm.get("q_label") and (o["q_readback"] or {}).get("label") == "NOT_INTEGRATED":
            o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop="nb4_ppack init lines: " + json.dumps(o["q_readback"])[:300])
            return o
        if is_d and KV_KIT_GATE.get("require_gpu_load_check", True) and \
                not ((o.get("dflash_readback") or {}).get("checks") or {}).get("draft_load_check"):
            o.update(status="NO_GO_DRAFT_LOAD_CHECK", early_stop="no draft_load_check line on both PP1 workers (the seam's "
                     "loaded-weights check) before any window: " + json.dumps(((o.get("dflash_readback") or {}).get("rec") or {}).get("draft_load_check"))[:300])
            return o
        if is_q and not is_d:                     # E28 K100 I8 (never on a D arm): INT8 requested -> I8_ENGAGED required, else the same pin + BF16 head
            o["i8_readback"] = (i8_readback(t, list(getattr(s, "argv", None) or []), arm.get("draft_head")) if not FAKE
                                else {"label": "FAKE_SERVER"})
            if arm.get("draft_head") == "int8" and o["i8_readback"].get("label") not in ("I8_ENGAGED", "FAKE_SERVER"):
                o.update(status="I8_READBACK_UNVERIFIED", early_stop="I8: " + json.dumps(o["i8_readback"], default=str)[:300])
                return o
        _live(o, "k100_gates", k100_boot_gates(p, s, arm, t) if not FAKE else {"real_failure": False, "fake": True})
        if o["k100_gates"].get("real_failure"):    # E27 K100: wrong quant kernels / FP8 KV without its static scales
            o.update(status="NO_GO_K100_BOOT_GATES", early_stop=str(o["k100_gates"].get("why"))[:400])
            return o
        if due(240):
            CUR["item"] = "smoke"
            _live(o, "smoke", smoke(p, arm))
            o["smoke_gate"] = N.smoke_gate(o["smoke"], is_q, q_nondet=Q_AA.get("clean") is not True)   # E27 K100: ladder vs Q's A/A
            o["smoke_pos0"] = N.pos0_label((o["smoke"] or {}).get("acceptance"))                        # STRUCTURAL / TAP_SUSPECT (recorded)
            _live(o, "free_after_smoke_min", free_min(p))
            mg = _mem_gate(p, o, "free_after_smoke_min")
            o["memory_gate_first_decode"] = {"ok": mg[0], "why": mg[1]}
            if is_q and o["smoke_gate"] in ("FAIL_SMOKE_CONCURRENCY_CONTAMINATION", "INCONCLUSIVE_SMOKE_CONCURRENCY_CONTAMINATION"):   # E25 NB-D1 (NB-S2): recorded
                o["smoke_gate_raw"], o["smoke_gate"] = o["smoke_gate"], "BASE_LADDER_NONDET_RECORDED"
            if is_d and o["smoke_gate"].startswith("FAIL"):       # errors / garbage / short / zero accept / contamination
                o.update(status="NO_GO_SMOKE", early_stop=f"smoke {o['smoke_gate']}", census_end=kv_census(s))
                return o
            if o["smoke_gate"] != "PASS":
                o.setdefault("recorded_gates", []).append(f"smoke {o['smoke_gate']} (recorded, arm continues)")
            if is_d and mg[0] is False:
                if _can_step(arm):
                    o.update(status="MEMORY_GATE_STEP_DOWN", early_stop="after the first decode: " + mg[1])
                    return o
                o.setdefault("recorded_gates", []).append("memory gate after the first decode (no rung / budget left, arm continues): " + mg[1])
        else:
            o["items_skipped"].append("smoke")
        if due(300) and (D1.item_on(arm, "prefill_probe") or not PREFILL.get("measured")):   # E25 NB-D1: B (a lever reuses B's rate)
            CUR["item"] = "prefill_probe"
            _live(o, "prefill", prefill_probe(p))
        if D1.item_on(arm, "oracle"):           # E25 NB-D1 light: the oracle only when the arm's plan has it (never)
            if not due(240):
                o["items_skipped"].append("oracle")
                o["status"] = "SKIPPED_BOOT_BUDGET_ORACLE"
                return o
            CUR["item"] = "oracle"
            a0 = accept_pos(p)
            _live(o, "oracle", oracle(p, key=store, refs=(ref,)))
            o["oracle_acceptance"] = N.acceptance_delta(a0, accept_pos(p), arm["k"])
            if is_q:
                Q_ACC.update(o["oracle_acceptance"] or {})
            else:
                o["al_vs_q"] = N.al_vs_q(o["oracle_acceptance"], Q_ACC)          # diagnostic only (GPT audit B2)
            if is_q:                                  # E27 K100: Q's A/A = the greedy oracle again after a cache reset + Q's own ladder
                CUR["item"] = "oracle_aa"
                o["oracle_aa_reset"] = reset_prefix(p)
                _live(o, "oracle_aa", oracle(p, key=None, refs=("q",)))
                Q_AA.update(oracle=(o["oracle_aa"].get("vs_q") or {}).get("status"),
                            ladder=N.contamination_gate((o.get("smoke") or {}).get("ladder")))
                Q_AA["clean"] = N.q_aa_clean(Q_AA)
                o["q_aa"] = dict(Q_AA)
            elif isinstance(o["oracle"].get("vs_" + ref), dict):   # D: token ids judged against Q's A/A (k100_rel)
                _vs = o["oracle"]["vs_" + ref]
                _vs["status_raw"] = _vs.get("status")
                _vs["status"] = N.k100_rel(_vs.get("status"), Q_AA.get("clean") is True)
            o["oracle_vs"] = (o["oracle"].get("vs_" + ref) or {}).get("status")
            o["quality_gate"] = quality_gate(o, ref)
            if o["quality_gate"].startswith("FAIL"):              # oracle errors / garbage / FAIL_HIGH_MARGIN / probes / sampled
                o["load_tests_skipped"] = f"oracle gate vs {ref}: {o['quality_gate']}"
                o["status"] = "NO_GO_QUALITY"
                o["census_end"] = kv_census(s)
                return o
            if o["quality_gate"] != "PASS":
                o.setdefault("recorded_gates", []).append(f"oracle gate vs {ref}: {o['quality_gate']} (recorded, arm continues)")
        else:
            o["items_skipped"].append("oracle")
            o["oracle_vs"], o["quality_gate"] = None, "NOT_IN_PLAN_NB_D1"   # recorded only
        if arm.get("short_only"):                 # D4t: smoke + greedy oracle + the two 8k@8 windows (vs D4), nothing else
            o["steady"], o["long_levels"], o["long_gate"], o["prefix_gate"] = {}, {}, "NOT_MEASURED_SHORT_ARM", "NOT_MEASURED_SHORT_ARM"
            if due(150):
                CUR["item"] = "warmup"
                _live(o, "warmup", warmup(p, n))
            run_window(o, s, arm, "8k", n, "#1")
            run_window(o, s, arm, "8k", n, "#2")
            CUR["item"] = "end"
            t = s.log() or ""
            o.update(census_end=kv_census(s), free_end_min=free_min(p), q_readback_end=q_readback(s, arm),
                     spec_log_all=N.spec_summary(N.spec_decoding_lines(t), arm["k"]), dflash_readback=dflash_rb(arm, t),
                     ingraph_meta=N.ingraph_meta(t), stage_graphs=N.stage_graphs(t), responses_saved=RESP["counts"].get(RESP["arm"]))
            o["memory_stages"] = N.memory_stages(o)
            return o
        if is_d and (arm.get("parent") or arm["key"]) in N.ADMISSION_PROBE_ARMS and due(240):
            CUR["item"] = "admission_probes"
            _live(o, "admission_probes", admission_probes(o, p))
        if due(300) and not o.get("admission_deadlock") and D1.item_on(arm, "long_oracle"):   # E25 NB-D1 light: off
            CUR["item"] = "long_oracle"
            with _watched(o, p, "long_oracle", active=is_d):
                _live(o, "long_oracle", long_oracle(p, key="q_long" if is_q else None, ref=None if is_q else "q_long", k=arm["k"]))
            if not is_q and isinstance((o.get("long_oracle") or {}).get("vs_q_long"), dict):   # E27 K100: vs Q's A/A
                _lv = o["long_oracle"]["vs_q_long"]
                _lv["status_raw"] = _lv.get("status")
                _lv["status"] = N.k100_rel(_lv.get("status"), Q_AA.get("clean") is True)
            o["long_gate"] = N.long_oracle_gate(o, is_q)
            if o["long_gate"].startswith(("NO_GO", "FAIL")):
                o.update(status="NO_GO_LONG_ORACLE", load_tests_skipped=f"long oracle {o['long_gate']}", census_end=kv_census(s))
                return o
        else:
            o["items_skipped"].append("long_oracle")
            o["long_gate"] = ("INCONCLUSIVE_ADMISSION_DEADLOCK" if o.get("admission_deadlock") else
                              "NOT_IN_PLAN_NB_D1" if not D1.item_on(arm, "long_oracle") else "INCONCLUSIVE_LONG_ORACLE_SKIPPED_BUDGET")
        if due(240) and not o.get("admission_deadlock") and D1.item_on(arm, "prefix"):   # E25 NB-D1 light: off
            CUR["item"] = "prefix_cache"
            try:
                with _watched(o, p, "prefix_cache", active=is_d):
                    _live(o, "prefix_cache", prefix_test(p, arm, block=o.get("engine_block") or N.BLOCK_TOKENS))
            except Exception as e:
                o["prefix_cache"] = {"error": f"{type(e).__name__}: {e}"[:300]}
            o["prefix_gate"] = N.prefix_gate(o["prefix_cache"])
        else:
            o["items_skipped"].append("prefix_cache")
            o["prefix_gate"] = "NOT_IN_PLAN_NB_D1" if not D1.item_on(arm, "prefix") else "INCONCLUSIVE_PREFIX_SKIPPED_BUDGET"
        if D1.item_on(arm, "sibling") and due(240) and not o.get("admission_deadlock"):   # E25 NB-D1: SPC's activation read-back (recorded)
            CUR["item"] = "sibling_probe"
            try:
                _live(o, "sibling_probe", sibling_probe(p, o))
            except Exception as e:
                o["sibling_probe"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        _live(o, "q_readback", q_readback(s, arm))
        qf_ = N.qdefer_real_failure(o["q_readback"]) if arm.get("q_label") else (False, "Q-defer off")
        o["qdefer_check"] = {"real_failure": qf_[0], "why": qf_[1]}
        if qf_[0]:                                # only a REAL Q-defer failure stops (USER RULE 2026-11-01); labels are recorded
            lab_ = (o["q_readback"] or {}).get("label")
            o.update(status="NOT_INTEGRATED" if lab_ == "NOT_INTEGRATED" else "QDEFER_FAILED", not_integrated=lab_ == "NOT_INTEGRATED",
                     early_stop="nb4_ppack read-back: " + qf_[1] + " | " + json.dumps(o["q_readback"])[:300])
            return o
        if due(150) and arm.get("windows_fixed") != []:   # E34: no warmup when the arm has no window (the smoke plan)
            CUR["item"] = "warmup"
            _live(o, "warmup", warmup(p, n))
        else:
            o["items_skipped"].append("warmup")
        o["steady"] = {}                      # RC2's _windows() gate reads "steady": garbage under load (windowed) is NO_GO here too
        o["long_levels"] = {}
        long_ok = True                            # E27 K100: the fixed window list (no short exit)
        for band, lvw, tag in (arm["windows_fixed"] if arm.get("windows_fixed") is not None else K100_CFG["windows"]):   # E25 NB-D1 / E34: an empty set = no window
            lvw = int(lvw)
            if band in D1.CAPPED_BANDS:            # E25 NB-D1: 20k too (S32 x 20k can exceed the resident bound)
                lv, why = band_level(o, band)
                use = min(lvw, lv) if isinstance(lv, int) and lv > 0 else lvw
                o["long_levels"][band] = dict(why, level=use, fixed=lvw, resident=lv)
                if band in N.LONG_BANDS:
                    CUR["item"] = f"reset_{band}"
                    o["long_levels"][band]["reset"] = reset_prefix(p) if due(win_need(band, use) + 60) else "SKIPPED"
                lvw = use
            _pf = PREFILL.get("tok_s")
            PREFILL["tok_s"] = min(float(_pf or 3000.0), float(K100_CFG["prefill_admit_floor_tok_s"]))   # the admit limit scales up
            try:
                run_window(o, s, arm, band, lvw, tag)
            finally:
                PREFILL["tok_s"] = _pf
        lv116 = (o["long_levels"].get("116k") or {}).get("level") or 0
        for band, lvb in D1.prof_plan(arm, NBD1_SWITCH):   # E25 NB-D1 PROF: 8k@16 / 20k@24 inside B's boot, 30 s each
            pk = f"profile_{band}@{lvb}"
            if due(win_need(pk.split("@")[0], lvb)) and s.alive() and not o.get("admission_deadlock"):
                CUR["item"] = pk
                try:
                    o["steady"][pk] = profile_window(p, arm, band, lvb, secs=float(D1.PROF_SECS))   # E25 NB-D1
                except Exception as e:
                    o["steady"][pk] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:400]}
            else:
                o["steady"][pk] = {"status": "SKIPPED_BOOT_BUDGET"}
                o["items_skipped"].append(pk)
        _live(o, "steady", o["steady"])
        o["measured_before_tf"] = True            # E27 K100: a server death in the TF below keeps the measured verdict (flag)
        if due(240) and s.alive() and D1.item_on(arm, "tf"):   # E25 NB-D1 light: no TF
            CUR["item"] = "tf_k100"
            try:
                if is_q or arm.get("e34_smoke"):    # E34: the smoke plan stores its own TF reference + A/A (the request path itself)
                    _live(o, "tf_ref", tforced(p, key="q_tf"))
                    _live(o, "tf_aa", tforced(p, ref="q_tf") if s.alive() else {"error": "server died"})
                    Q_TF_AA.clear()
                    Q_TF_AA.update({b_: v_ for b_, v_ in (o["tf_aa"] or {}).items() if isinstance(v_, dict)})
                else:
                    _live(o, "tf_vs_q", tforced(p, ref="q_tf"))
                    o["tf_gate"] = N.tf_gate(o["tf_vs_q"], Q_TF_AA, tuple(K100_CFG["tf_bands"]))
            except Exception as e:
                o["tf_error"] = f"{type(e).__name__}: {e}"[:300]
            o["tf_server_died"] = not s.alive()
        else:
            o["items_skipped"].append("tf_k100")
            if not D1.item_on(arm, "tf"):
                o["tf_plan"] = "NOT_IN_PLAN_NB_D1"
        CUR["item"] = "end"
        t = s.log() or ""
        o["census_end"] = kv_census(s)
        o["workers_end"] = rpc_cache_info(p)
        o["free_end_min"] = free_min(p)
        o["memory_stages"] = N.memory_stages(o)
        o["q_readback_end"] = q_readback(s, arm)
        o["ppq_stats_last"] = [e for e in P.parse_log(t) if e.get("event") == "stats"][-8:]
        o["jit_after_warmup"] = jit_kernels(t)[-12:]
        o["config_readback_end"] = N.config_readback(arm, t, CUR["kv_bytes"])
        o["spec_log_all"] = N.spec_summary(N.spec_decoding_lines(t), arm["k"])
        if is_d:                                   # E25 NB-D1: B included
            o["dflash_readback"] = dflash_rb(arm, t)               # the whole log: the SpecDecoding lines now exist
        else:
            o["seam_inert"] = N.q_seam_inert(t)
        o["responses_saved"] = RESP["counts"].get(RESP["arm"])
        o["prefix_reuse"] = N.prefix_reuse(o, t)
        o["stage_graphs"] = N.stage_graphs(t)
        o["kv_draft_group_warning"] = N.DRAFT_GROUP_WARN_RX in t
        sl_ = N.spec_decoding_lines(t)
        o["preempt_accept"] = {wk: N.preempt_accept(w, arm["k"], sl_) for wk, w in (o.get("steady") or {}).items()
                               if isinstance(w, dict) and (w.get("preemptions") or 0) > 0}
        if not is_d:
            o["ingraph_meta"] = None
        else:
            o["ingraph_meta"] = N.ingraph_meta(t)
        try:
            o["d1_records"] = d1_records(o, arm, t)     # E25 NB-D1: K_eff, KV ledgers, QSA over the whole log (recorded)
        except Exception as e:                     # records only: never fails the arm
            o["d1_records"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        return o
    return body


def arm_watched(body, cap_min, name):
    def b(ss):
        cap = T.body_cap_sec(cap_min, left())
        BOOT.update(deadline=time.time() + cap, fired=None)

        def fire():
            BOOT["fired"] = name
            for x in ss:
                try:
                    x.kill()
                except Exception:
                    pass
        tm = threading.Timer(cap + 300, fire)
        tm.daemon = True
        tm.start()
        try:
            o = body(ss)
            LIVE["out"] = o if isinstance(o, dict) else None
        finally:
            tm.cancel()
            BOOT["deadline"] = None
        o["body_cap_sec"] = round(cap)
        if BOOT["fired"]:
            o["boot_watchdog"] = f"FIRED after {cap + 300:.0f} s: server killed"
        return o
    return b


def arm_boot(arm, body):
    """NB4T's boot (the gate, one Merlin launch, the /dev/shm cleanup) + the three-state tree check; Q: one OOM retry 0.25 GiB
    lower; D arms: start on D_PIN, one rung lower on a launch OOM or the 512 MiB memory gate, up to D_OOM_RETRIES more."""
    is_d = arm.get("method") == "dflash"
    is_sw = (arm.get("parent") or arm["key"]) in N.SW_BLOCK_ARMS
    kvb = int(K100_CFG["q_pin"])              # E27 K100: every arm starts at 7,933,526,016 B
    port, tries, o = arm["port"], [], {}
    ref = K100_CFG["q_ref"] if arm["key"] == D1.BASE_KEY else "q"   # E27 K100 / E25 NB-D1: B
    max_tries = 1 + (len(K100_CFG["d_retry"]) if is_d else len(K100_CFG["q_ladder"]))   # E27 K100: D 4 OOM retries; Q 1 OOM + 1 I8 drop
    n_oom = 0
    while True:
        lname = arm["name"] if not tries else f"{arm['name']}_t{len(tries)}"
        tc = tree_check(arm)
        if not tc["ok"]:
            return {"status": "TREE_INVALID_MIXED", "tree_check": tc, "tries": tries, "launch_name": lname, "pin": kvb}
        if arm["key"] == "Q":                     # E27 K100: Q = E28 production, the INT8 draft head while Q_I8 is on
            arm["draft_head"] = "int8" if Q_I8["on"] else None
            arm["expect_cfg"]["draft_head"] = arm["draft_head"]
            NB4T_BUILD_ENV[lname] = {"DRAFT_HEAD": "int8"} if Q_I8["on"] else {}
        NB1_EXTRA[lname], NB1_ENV[lname], NB1_KV[lname] = list(arm["extra"]), dict(arm["env"]), kvb
        ARM_OF[lname] = arm
        CUR["kv_bytes"] = kvb
        CUR["tries_left"] = max_tries - len(tries) - 1
        BOOT["tag"] = lname
        RESP["arm"] = lname
        gate = boot_gate()
        if gate["ok"] is not True:
            return {"status": "GPU_NOT_EMPTY", "boot_gate": gate, "tries": tries, "launch_name": lname, "pin": kvb}
        shm0 = shm_names()
        mon = LoadMonitor(lname)
        mon.start()
        srv, CUR["item"] = None, "launch"
        try:
            srv = merlin_fn_server(lname, "fp8", port, kvb)
            o = with_servers([srv], NBDF_READY_SEC, arm_watched(body, arm["cap_min"], lname))
        finally:
            mon.stop_ev.set()
        try:
            full = srv.log() or ""
        except Exception:
            full = ""
        lines = full.splitlines()
        o.update(kv_bytes=kvb, pin=kvb, launch_name=lname, port=port, boot_gate=gate, tree_check=tc, shm_cleanup=shm_cleanup(shm0),
                 ready_sec=getattr(srv, "ready_sec", None), load_monitor_last=mon.last, oom_in_log=bool(OOM_ERR.search(full)),
                 xid_in_log=bool(XID.search(full)), oom_lines=[l[:240] for l in lines if OOM_ERR.search(l)][:6],
                 xid_lines=[l[:240] for l in lines if XID.search(l)][:6], died_in=CUR["item"])
        if srv is not None:
            o["q_readback_final"] = q_readback(srv, arm)
            o.setdefault("config_readback", N.config_readback(arm, full, kvb))
            if is_d:
                o["dflash_readback_final"] = dflash_rb(arm, full)
                o.setdefault("dflash_readback", o["dflash_readback_final"])
            o["launch_error_lines"] = [l[:300] for l in lines if re.search(r"Error|Traceback|NotImplementedError|assert", l)][-12:]
        oom = bool(o["oom_in_log"]) and not o["xid_in_log"]   # E25 NB-D1 (= NB-S1 GPU v1 fix: an OOM at ANY stage, not a device fault) -> the ladder
        memstep = o.get("status") == "MEMORY_GATE_STEP_DOWN"
        tries.append({"launch_name": lname, "port": port, "kv_bytes": kvb, "status": o.get("status"), "oom": oom, "memory_step": memstep,
                      "device_fault": o["xid_in_log"], "ready_sec": o["ready_sec"]})
        i8_drop = arm["key"] == "Q" and Q_I8["on"] and (o.get("status") == "I8_READBACK_UNVERIFIED"
                                                        or (o.get("status") == "LAUNCH_FAILED" and not o["xid_in_log"]))
        if i8_drop:                               # E28 K100 I8: the SAME pin with the BF16 head (I8_UNSUPPORTED_ON_K100)
            Q_I8.update(on=False, dropped={"try": lname, "status": o.get("status"), "readback": (o.get("i8_readback") or {}).get("label")})
            tries[-1]["i8_drop"] = True
        n_oom += 1 if (oom and not i8_drop) else 0
        if not (oom or i8_drop) or o["xid_in_log"] or len(tries) >= max_tries or left() < NBDF_READY_SEC + arm["cap_min"] * 60:
            break
        if not i8_drop:                           # A's B3: D -> EMBED_HOST at the same pin, then 0.25-GiB rungs; Q -> one rung
            nxt = _k100_retry(arm, is_d, n_oom)
            if nxt is None:
                break
            kvb = nxt[0]
            if nxt[1]:
                arm["env"].update(nxt[1])
                arm["expect"] = dict(arm["expect"], **{w: arm["env"].get(w, N.WATCHED_DEFAULT.get(w)) for w in N.WATCHED})   # E25 NB-D1: keep the QSA / GDN keys
                arm.setdefault("retry_env", {}).update(nxt[1])
        port += 1
        log("boot", "I8 drop" if i8_drop else "OOM", lname, "-> KV_BYTES", kvb, "env", arm.get("retry_env"), "port", port)
    o["tries"] = tries
    if o.get("status") == "MEMORY_GATE_STEP_DOWN":
        o["status"] = "MEMORY_GATE_FAILED"
    if o.get("status") == "SERVER_DIED":
        o["death"] = ("BOOT_CAP_KILL" if o.get("boot_watchdog") else
                      f"BODY_OOM_{o.get('died_in')}" if o["oom_in_log"] and not o["xid_in_log"] else
                      f"DEVICE_FAULT_{o.get('died_in')}" if o["xid_in_log"] else f"SERVER_DIED_IN_{o.get('died_in')}")
        o["status"] = o["death"]
    o["capacity"] = T.capacity_facts(o)
    if not D1.item_on(arm, "oracle"):         # E25 NB-D1 light: no oracle in the plan -> the hard-only evidence (probes, sampled, windows)
        return gated(o, need=(), expected_rows=None)
    return gated(o, need=(ref,), expected_rows=4)


def analyze_arm(o, key):
    """After the boot (GPUs idle): graph stalls + nb4_timeline on the 8k profile; graph spans (drafter vs target replay) of
    every profile (8k, 116k)."""
    st_ = o.get("steady") or {}
    o["graph_spans"] = {}
    for pk, w in st_.items():
        if pk.startswith("profile") and isinstance(w, dict) and w.get("trace_dir") and list(Path(w["trace_dir"]).glob("*.pt.trace.json*")):
            try:
                o["graph_spans"][pk] = _trace_spans(w["trace_dir"])
            except Exception as e:
                o["graph_spans"][pk] = {"error": repr(e)[:200]}
    pk = next((k_ for k_ in st_ if k_.startswith("profile_8k")), None)
    d = (st_.get(pk) or {}).get("trace_dir") if pk else None
    if not d or not list(Path(d).glob("*.pt.trace.json*")):
        o["stall"] = {"error": "NO_TRACE_FILE"}
        o["timeline"] = {"error": "NO_TRACE_FILE"}
        return o
    t = time.time()
    try:
        st = T.graph_stalls(d)
        (OUT / f"stall_{key.lower()}.json").write_text(json.dumps(st, indent=1, default=str))
        o["stall"] = {k: v for k, v in st.items() if k != "by_rank"}
        o["stall"]["by_rank"] = {r: {x: y for x, y in v.items() if x != "rows"} for r, v in (st.get("by_rank") or {}).items()}
    except Exception as e:
        o["stall"] = {"error": f"{type(e).__name__}: {e}"[:400]}
    try:
        rep = TL.strip_private(TL.analyze(d))
        (OUT / f"timeline_{key.lower()}.json").write_text(json.dumps(rep, indent=1, default=str))
        o["timeline"] = T.summarize_timeline(rep)
        o["stage_compute"] = T.stage_compute(o["timeline"])
        try:
            (OUT / f"timeline_{key.lower()}.md").write_text(TL.format_report(rep))
        except Exception as e:
            o["timeline"]["format_report_error"] = repr(e)[:200]
    except Exception as e:
        o["timeline"] = {"error": f"{type(e).__name__}: {e}"[:400]}
    o["parity_record"] = T.parity_of(o)
    o["analysis_sec"] = round(time.time() - t, 1)
    return o


def _arm(key):
    return ARMS.get(key) or ISO.get(key)


def run_arm(key):
    arm = _arm(key)
    o = arm_boot(arm, arm_body(arm, K100_CFG["q_ref"] if key == D1.BASE_KEY else "q", "q" if key == D1.BASE_KEY else None))   # E25 NB-D1
    LIVE["out"] = o
    try:
        o["arm"], o["planned"] = key, {x: arm.get(x) for x in _AKEYS}
        o["free_mib_per_card"] = free_per_card(o, o.get("launch_name"))
        if left() > 120:
            analyze_arm(o, key)
    except Exception as e:
        keep_after_error(o, e)
    return o


def _arm_rec(key):
    a = _arm(key)
    r = (R["sections"].get(key) or {}).get(a["name"]) if a else None
    return r if isinstance(r, dict) else None


def arm_clean(o):
    return isinstance(o, dict) and o.get("status") == "DONE" and o.get("correctness") == "CLEAN" and o.get("quality_gate") == "PASS"


def skip_arm(key, why, status="SKIPPED_DEPENDENCY", **kw):
    R["sections"].setdefault(key, {})[_arm(key)["name"]] = dict({"status": status, "why": why}, **kw)
    save()


def finalize():
    try:
        recs = {}
        for k in list(ARMS) + list(ISO):
            r = _arm_rec(k)
            if r is not None and "stall" not in r and any(str(x).startswith("profile") and (v or {}).get("trace_dir")
                                                          for x, v in (r.get("steady") or {}).items() if isinstance(v, dict)):
                try:
                    analyze_arm(r, k)
                except Exception as e:
                    r["stall"] = {"error": repr(e)[:300]}
            recs[k] = r
        v = N.verdict({k: r for k, r in recs.items() if r is not None}, ARMS)
    except Exception as e:
        v = {"verdict": "ERROR_IN_VERDICT", "why": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1500:]}
    v["arms_status"] = {k: (_arm_rec(k) or {}).get("status") for k in list(ARMS) + list(ISO)}
    v["patch"], v["seam"], v["drafter_ok"] = NBDF.get("patch"), NBDF.get("seam_seq"), (NBDF.get("drafter") or {}).get("ok")
    v["d_pin"], v["kv_kit_gate"], v["responses_saved"] = D_PIN, KV_KIT_GATE.get("passed"), RESP["counts"]
    v["minutes"] = round((time.time() - T_START) / 60, 1)
    NBDF["verdict"] = v
    try:
        (OUT / "nbdf_verdict.json").write_text(json.dumps(v, indent=1, default=str))
    except Exception:
        pass
    save()
    log("NBDF VERDICT", v.get("verdict"), "|", str(v.get("why"))[:400])
    return v


def _patch_rec(pseq):
    return {"state": pseq.state, "log": pseq.log, "error": pseq.error, "apply": pseq.apply_rec, "revert": pseq.revert_rec}


def _seam_rec(sseq):
    return {"state": sseq.state, "ring": sseq.ring, "log": sseq.log, "error": sseq.error, "apply": sseq.apply_rec,
            "verify": sseq.verify_rec, "revert": sseq.revert_rec}


def nbdf_main(runner=None, pseq=None, sseq=None):
    """install -> #60490 + extension -> sha gate -> nb4_ppack -> dflash_seam (after nb4_ppack; a failure = every D arm
    NOT_INTEGRATED, Q still runs) -> Q -> D4 (AL early exit) -> D3 -> D7 (ring verified) -> D4b; the first failing D arm reruns
    once with Q-defer off; every D arm needs the drafter pins, the KV kit gate, a clean engaged Q and its full minimum of time.
    The seam is reverted first, then nb4_ppack; the verdict is written in the finally block whatever happens."""
    runner = runner or run_arm
    pseq = pseq or PSEQ
    sseq = sseq or SSEQ
    try:
        m = R["sections"]["A0"]["merge"]
        if not m.get("ok"):
            NBDF["stop"] = "MERGE_FAILED: no boot started"
            for k in ARMS:
                skip_arm(k, NBDF["stop"])
            return
        step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
        pt = step("A0", "patch_60490", 3, apply_60490)
        if pt.get("status") != "DONE" or not pt.get("ok"):
            for k in ARMS:
                skip_arm(k, f"#60490 patch / worker extension not verified: {pt.get('error') or pt.get('status')}")
            return
        ks = step("A0", "k100_setup", 3, k100_setup)          # E27 K100: overlay merge + K6, the QSA static scales
        if ks.get("status") != "DONE" or ks.get("ok") is not True:
            for k in ARMS:
                skip_arm(k, f"K100 setup: {ks.get('why') or ks.get('error') or ks.get('status')}", "NOT_INTEGRATED", not_integrated=True)
            return
        if "rc2_fp8" not in ROWS and not FAKE:     # recorded (USER RULE 2026-11-01): Q's oracle vs RC2 is then INCONCLUSIVE_NO_ORACLE
            NBDF["rc2_rows"] = "MISSING: RC2's recorded FP8 oracle rows do not match this notebook's prompts (Q runs; its gate vs RC2 is recorded)"
            log("NBDF", NBDF["rc2_rows"])
        ok, why = sha_gate(min(1500, left() - 900))
        R["setup"]["sha_gate"] = {"ok": ok, "why": why}
        if not ok:
            for k in ARMS:
                skip_arm(k, "sha gate: " + why, "SKIPPED_SHA_GATE")
            return
        ok_p, why_p = pseq.before_arm(ARMS["Q"])
        NBDF["patch"] = _patch_rec(pseq)
        q_tree_ok = ok_p
        if not ok_p:                              # only Q (coordinator 2026-11-01): each D arm runs its own tree check below
            skip_arm("Q", why_p, "NOT_INTEGRATED", not_integrated=True)
        if SEAM_API is None and sseq is SSEQ:
            seam_ok, seam_why = False, "NOT_INTEGRATED: " + str(SEAM_LOAD.get("error"))
        else:
            seam_ok, seam_why = sseq.apply()
        ring = N.ring_status(sseq) if seam_ok else (False, False, "no seam")
        NBDF["seam_seq"], NBDF["seam_ok"], NBDF["seam_why"] = _seam_rec(sseq), seam_ok, seam_why
        NBDF["ring"] = {"available": ring[0], "verified": ring[1], "why": ring[2]}
        try:
            NBDF["ppack_after_seam"] = P.verify_ppack_patch(_tree_py()[0])["state"] if not FAKE else "fake"
        except Exception as e:
            NBDF["ppack_after_seam"] = repr(e)[:200]
        save()
        log("NBDF seam:", seam_ok, seam_why, "| ring:", NBDF["ring"], "| drafter:", DRAFTER.get("status"), DRAFTER.get("why"))
        q = (step("Q", ARMS["Q"]["name"], ARMS["Q"]["est_min"], lambda: runner("Q")) if q_tree_ok
             else R["sections"]["Q"][ARMS["Q"]["name"]])
        bq = N.base_usable(q)                      # USER RULE 2026-11-01: only a REAL Q failure skips the D arms
        bs = N.base_check(q)
        NBDF["base"] = {"usable": bq[0], "label": bq[1], "why": bq[2], "strict_nb4t_base_check": {"ok": bs[0], "label": bs[1], "why": bs[2]}}
        save()
        if not bq[0]:                              # E27 K100: recorded only; the D arms run anyway (no cross-arm skips)
            NBDF["base_not_usable"] = f"{bq[1]}: {bq[2]}"
            log("NBDF K100: Q not a usable base (recorded, the D arms run):", NBDF["base_not_usable"])
        PIN.update(kv_bytes=q.get("pin") or PIN["kv_bytes"], source="Q's final pin")
        Q_REC.clear()
        Q_REC.update(q)
        D_PIN.update(kv_bytes=int(K100_CFG["q_pin"]), plan=None, source="E27 K100: every arm starts at 7,933,526,016 B")
        NBDF["d_pin"] = D_PIN
        stop_d, isos = None, 0                   # stop_d only for a broken tree (never for AL / speed: GPT audit B2)
        for k in N.D_ARMS:
            a = ARMS[k]
            if stop_d:                             # E27 K100: recorded; each arm's own tree check decides
                NBDF.setdefault("tree_notes", []).append(stop_d)
                stop_d = None
            if not DRAFTER.get("ok"):
                skip_arm(k, f"{DRAFTER.get('status') or 'DRAFTER'}: {DRAFTER.get('why')}", "NOT_INTEGRATED", not_integrated=True)
                continue
            if not KV_KIT_GATE.get("passed"):
                skip_arm(k, "the real CPU kit's get_kv_cache_configs gate did not pass: " + str(KV_KIT_GATE.get("why")), "SKIPPED_KV_KIT_GATE")
                continue
            if not seam_ok:
                skip_arm(k, seam_why, "NOT_INTEGRATED", not_integrated=True)
                continue
            if not (KV_KIT_GATE.get("per_k") or {}).get(k):   # recorded (USER RULE 2026-11-01): the boot's own KV read-back plans it
                NBDF.setdefault("recorded_gates", {})[k] = f"no per-K KV summary from the real kit for {k} ({N.KV_RUN_OF[k]}): runs on its boot read-back"
            if a["ring_needed"] and not ring[1]:
                skip_arm(k, f"K {a['k']} needs the QSA ring widening: {ring[2]}", "SKIPPED_NO_RING")
                continue
            if not N.arm_fits(left(), a) and not (k == K100_CFG["shorten_arm"] and left() >= K100_CFG["shorten_min_left_min"] * 60):
                skip_arm(k, f"left {left() / 60:.1f} min < its full minimum {a['est_min']} + {N.START_RESERVE_SEC // 60} min "
                            "(priority Q > D4 > D3 > D7 > D4b)", "SKIPPED_BUDGET")
                continue
            ok_s, why_s = sseq.check(k)
            ok_p, why_p = pseq.before_arm(a)
            NBDF["patch"], NBDF["seam_seq"] = _patch_rec(pseq), _seam_rec(sseq)
            if not ok_s or not ok_p:
                skip_arm(k, why_s if not ok_s else why_p, "TREE_INVALID_MIXED" if "INVALID" in (why_s + why_p) else "NOT_INTEGRATED",
                         not_integrated=True)
                stop_d = "dflash_seam / nb4_ppack tree not FULLY_PATCHED: " + (why_s if not ok_s else why_p)
                continue
            r = step(k, a["name"], a["est_min"], lambda k=k: runner(k))
            if r.get("status") == "DONE" and r.get("pin"):
                if k in N.SW_BLOCK_ARMS:
                    SW_PIN.update(kv_bytes=r["pin"], source=f"{k}'s final pin")
                else:
                    D_PIN.update(kv_bytes=min(D_PIN["kv_bytes"], r["pin"]), source=f"{k}'s final pin")
            if r.get("status") == "TREE_INVALID_MIXED":
                stop_d = "TREE_INVALID_MIXED before " + k
            if N.needs_nometa(r):                  # E27 K100: every rerun is deferred to the end (D4s always runs first)
                PENDING.append(("nometa", k))
            if N.needs_isolation(r) and not str(r.get("smoke_gate", "")).endswith("ZERO_ACCEPT"):
                PENDING.append(("iso", k))
            if (r.get("smoke_pos0") or {}).get("label") in ("STRUCTURAL", "TAP_SUSPECT"):
                PENDING.append(("tap", k))
        _k100_run_pending(runner)
    finally:
        try:
            sseq.finish()
        except Exception as e:
            sseq.error = f"finish: {type(e).__name__}: {e}"[:300]
        try:
            pseq.finish()
        except Exception as e:
            pseq.error = f"finish: {type(e).__name__}: {e}"[:300]
        NBDF["patch"], NBDF["seam_seq"] = _patch_rec(pseq), _seam_rec(sseq)
        finalize()



# ==================================================================== E25 NB-D1: ledger, SPC probe, read-backs, main, records
class D1Ledger:
    """The KV ledger of one window: /metrics every 2 s in a daemon thread (perf_counter time, the clock of steady()'s t0 /
    t1): KV usage, running / waiting, waiting by reason, preemptions, prompt tokens by source (records only)."""

    def __init__(self, port, every=2.0):
        self.port, self.every, self.samples, self.stop_ev, self.th = port, every, [], threading.Event(), None

    def _one(self):
        try:
            txt = L.U.urlopen(f"http://127.0.0.1:{self.port}/metrics", timeout=15).read().decode("utf-8", "replace")
            self.samples.append((time.perf_counter(), D1.parse_metrics(txt)))
        except Exception:
            pass

    def _loop(self):
        while not self.stop_ev.wait(self.every):
            self._one()

    def __enter__(self):
        if not FAKE:
            self._one()
            self.th = threading.Thread(target=self._loop, daemon=True, name=f"nbd1_ledger_{self.port}")
            self.th.start()
        return self

    def __exit__(self, *a):
        self.stop_ev.set()
        if self.th is not None:
            self.th.join(10)
        if not FAKE:
            self._one()
        return False


def d1_window_ledger(o, r, band, n):
    """The window's KV ledger with THIS boot's pool and F (records only)."""
    smp = r.pop("_d1_samples", None) or []
    res = o.get("k100_residents") or {}
    f_ = ((res.get("40k") or {}).get("F"))
    blk = ((res.get("40k") or {}).get("block")) or N.BLOCK_TOKENS
    return {"window": D1.ledger_summary(smp, r.get("t0"), r.get("t1"), res.get("pool"), n, BANDS.get(band), f_, int(blk),
                                        r.get("gen_tokens") or 0),
            "with_admission": D1.ledger_summary(smp, None, None, res.get("pool"), n, BANDS.get(band), f_, int(blk), r.get("gen_tokens") or 0)}


def d1_live(o, arm, key, r):
    """The [LIVE] line after every window (print only; never raises)."""
    try:
        b_rec = None if arm["key"] == D1.BASE_KEY else _arm_rec(D1.BASE_KEY)
        print(E34.live_line(D1, (time.time() - T_START) / 60.0, arm["key"], key, r, b_rec), flush=True)   # E34: + prefix hit, W vs B's W
    except Exception as e:
        print(f"[LIVE] {arm.get('key')} {key}: line error {type(e).__name__}: {e}"[:200], flush=True)


def sibling_probe(port, o):
    """SPC activation read-back (recorded): after a cache reset, three siblings A / B / C = P + a different suffix (P =
    D1_CFG sibling prefix_tokens, not a block multiple); C's prefix-hit tokens show whether the shared-prefix junction was
    checkpointed (D1.sibling_label)."""
    c = D1_CFG["sibling"]
    block = int(o.get("engine_block") or N.BLOCK_TOKENS)
    p0 = prompt(int(c["prefix_tokens"]), 80)
    out = {"prefix_tokens": len(p0), "block": block, "reset": reset_prefix(port), "hits": {}, "rows": {}}
    for nm, v in (("A", 81), ("B", 82), ("C", 83)):
        ids = list(p0) + list(prompt(int(c["suffix_tokens"]), v))
        c0 = _prefix_counters(port)
        r = _one(port, ids, int(c["gen"]))
        d = _prefix_delta(c0, _prefix_counters(port))
        out["hits"][nm] = d.get("hits")
        out["rows"][nm] = {"prompt": len(ids), "queries": d.get("queries"), "tokens": len(r.get("tokens") or []), "error": r.get("error")}
    out["label"] = D1.sibling_label(out, block)
    return out


def d1_readback_failure(o, arm, t):
    """NB-D1 real failures of THIS arm (its own one change read back with another value, or another lever's change present:
    the arm would measure the wrong thing): Q256's QSA env / the GDN switch in the 4 workers, the indexer cache dtype, the
    logged seqs / batched tokens / long-prefill / shared-prefix flags, the logged K / dynamic schedule, the DYN fallback lines
    (NB-S1). B's base settings (QSA 128 / 64, EMBED_HOST) read back differently are recorded only. None = no failure."""
    o["d1_readback"] = D1.readback(arm, t, o.get("env_readback_raw"), o.get("config_readback"), o.get("workers"), o.get("dflash_readback"))
    o["e34_readback"] = E34.readback(arm, t, o.get("workers"), o.get("dflash_readback"))   # E34: SSM dtype / block / SW per worker
    if o["d1_readback"].get("base_mismatch"):   # B's base settings read back differently: recorded, the arm measures on
        o.setdefault("recorded_gates", []).append("NB-D1 base settings (recorded, arm continues): " + "; ".join(o["d1_readback"]["base_mismatch"])[:400])
    if FAKE:
        return None
    if (o.get("e34_readback") or {}).get("real_failure"):   # E34: the planned SSM dtype / block / SW not engaged = this arm only
        return "E34 read-back: " + str(o["e34_readback"].get("why"))[:400]
    if o["d1_readback"].get("real_failure"):
        return "NB-D1 read-back: " + str(o["d1_readback"].get("why"))[:400]
    return None


def _k_eff(w):
    a, r = (w or {}).get("accept_per_draft"), (w or {}).get("accept_rate")
    if isinstance(a, (int, float)) and isinstance(r, (int, float)) and r > 0:
        return round(a / r, 3)
    return None


def d1_records(o, arm, t):
    """Record-only facts of one arm: K_eff per window, the KV ledgers, the pool / residents, the QSA read-back over the whole
    log (Merlin's logged granule at the first prefill), the logged flags, the indexer dtype at the end."""
    st_ = o.get("steady") or {}
    rec = {"k_eff": {k: _k_eff(w) for k, w in st_.items() if isinstance(w, dict) and not k.startswith("profile")},
           "kv_ledger": {k: w.get("kv_ledger") for k, w in st_.items() if isinstance(w, dict) and w.get("kv_ledger")},
           "residents": o.get("k100_residents"), "pool": (o.get("k100_residents") or {}).get("pool"),
           "engine_block": o.get("engine_block"), "qsa_end": D1.qsa_readback(arm, o.get("env_readback_raw"), t),
           "idx_end": D1.idx_readback(o.get("workers_end") or o.get("workers")),
           "log_flags": {f: (f in t) for f in ("long_prefill_token_threshold", "enable_mamba_shared_prefix_checkpoint",
                                               "indexer_kv_dtype", "num_speculative_tokens_per_batch_size")},
           "dyn_fallback_lines": [l.strip()[:200] for l in t.splitlines() if any(x in l for x in D1.DYN_FAIL_RX)][:4]}
    return rec


def nbd1_finalize():
    try:
        recs = {k: _arm_rec(k) for k in ARMS}
        v = D1.verdict(T, {k: r for k, r in recs.items() if r is not None}, ARMS, NBD1_SWITCH, N.LONG_THR_FLOOR)
        v["e34"] = dict(E34.records({k: r for k, r in recs.items() if r is not None}, ARMS, NBD1_SWITCH), w=E34R.get("w"), tf=E34R.get("tf"))   # E34
    except Exception as e:
        v = {"verdict": "ERROR_IN_RECORDS", "why": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1500:]}
    v["arms_status"] = {k: (_arm_rec(k) or {}).get("status") for k in ARMS}
    v["patch"], v["seam"], v["drafter_ok"] = NBDF.get("patch"), NBDF.get("seam_seq"), (NBDF.get("drafter") or {}).get("ok")
    v["kv_kit_gate"], v["responses_saved"] = KV_KIT_GATE.get("passed"), RESP["counts"]
    v["minutes"] = round((time.time() - T_START) / 60, 1)
    R["sections"].setdefault("PROF", {})["nbd1_prof"] = v.get("prof") or D1.prof_record(_arm_rec(D1.BASE_KEY), NBD1_SWITCH)
    NBDF["verdict"] = NBD1["verdict"] = v
    try:
        (OUT / "nbd1_verdict.json").write_text(json.dumps(v, indent=1, default=str))
        (OUT / "e34_verdict.json").write_text(json.dumps(v.get("e34"), indent=1, default=str))   # E34
    except Exception:
        pass
    save()
    log("NB-D1 RECORDS", v.get("verdict"), "|", str(v.get("why"))[:600])
    return v


def nbd1_main(runner=None, pseq=None, sseq=None):
    """NBDF-K100's launch path, B first: merge -> install -> #60490 + extension -> k100_setup (merge + K6, the QSA scales)
    -> sha gate -> nb4_ppack (PSEQ.before_arm(B)) -> dflash_seam (SSEQ.apply) -> every switched-on arm in order with its own
    drafter / KV-gate / seam / budget / tree checks. A failure stops only that arm (no cross-arm skip); B not usable is
    recorded and the levers run. An arm switched off in CELL 1 (NBD1_ON) is recorded SKIPPED_SWITCH_OFF before any other
    check and never booted. The seam is reverted first, then nb4_ppack; the records in the finally."""
    runner = runner or run_arm
    pseq = pseq or PSEQ
    sseq = sseq or SSEQ
    try:
        for k in ARMS:                           # E25 NB-D1: CELL 1 arm switch off -> not booted (recorded; the other arms unaffected)
            if not NBD1_SWITCH.get(k, True):
                skip_arm(k, "switched off in CELL 1 (NBD1_ON)", "SKIPPED_SWITCH_OFF")
        RUN_KEYS = [k for k in ARMS if NBD1_SWITCH.get(k, True)]
        NBD1["run_keys"] = list(RUN_KEYS)
        if not RUN_KEYS:
            NBDF["stop"] = "ALL_ARMS_SWITCHED_OFF: no setup, no boot"
            return
        m = R["sections"]["A0"]["merge"]
        if not m.get("ok"):
            NBDF["stop"] = "MERGE_FAILED: no boot started"
            for k in RUN_KEYS:
                skip_arm(k, NBDF["stop"])
            return
        step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
        pt = step("A0", "patch_60490", 3, apply_60490)
        if pt.get("status") != "DONE" or not pt.get("ok"):
            for k in RUN_KEYS:
                skip_arm(k, f"#60490 patch / worker extension not verified: {pt.get('error') or pt.get('status')}")
            return
        ks = step("A0", "k100_setup", 3, k100_setup)
        if ks.get("status") != "DONE" or ks.get("ok") is not True:
            for k in RUN_KEYS:
                skip_arm(k, f"K100 setup: {ks.get('why') or ks.get('error') or ks.get('status')}", "NOT_INTEGRATED", not_integrated=True)
            return
        ok, why = sha_gate(min(1500, left() - 900))
        R["setup"]["sha_gate"] = {"ok": ok, "why": why}
        if not ok:
            for k in RUN_KEYS:
                skip_arm(k, "sha gate: " + why, "SKIPPED_SHA_GATE")
            return
        ok_p, why_p = pseq.before_arm(ARMS[D1.BASE_KEY])
        NBDF["patch"] = _patch_rec(pseq)
        NBD1["ppack_first_apply"] = {"ok": ok_p, "why": why_p}
        if SEAM_API is None and sseq is SSEQ:
            seam_ok, seam_why = False, "NOT_INTEGRATED: " + str(SEAM_LOAD.get("error"))
        else:
            seam_ok, seam_why = sseq.apply()
        ring = N.ring_status(sseq) if seam_ok else (False, False, "no seam")
        NBDF["seam_seq"], NBDF["seam_ok"], NBDF["seam_why"] = _seam_rec(sseq), seam_ok, seam_why
        NBDF["ring"] = {"available": ring[0], "verified": ring[1], "why": ring[2]}
        try:
            NBDF["ppack_after_seam"] = P.verify_ppack_patch(_tree_py()[0])["state"] if not FAKE else "fake"
        except Exception as e:
            NBDF["ppack_after_seam"] = repr(e)[:200]
        save()
        log("NB-D1 seam:", seam_ok, seam_why, "| drafter:", DRAFTER.get("status"), DRAFTER.get("why"), "| walk:", NBD1.get("walk"))
        for k, a in ARMS.items():
            if k not in RUN_KEYS:                # E25 NB-D1: switched off (recorded above, before any other check)
                continue
            if k in E34.W_ARMS:                  # E34: S16B2W's max_num_seqs from S16B2's measured pool (its boot log)
                a = e34_w_replan(k, a)
            if not DRAFTER.get("ok"):
                skip_arm(k, f"{DRAFTER.get('status') or 'DRAFTER'}: {DRAFTER.get('why')}", "NOT_INTEGRATED", not_integrated=True)
                continue
            if not KV_KIT_GATE.get("passed"):
                skip_arm(k, "the NB-D1 real-kit gate did not pass: " + str(KV_KIT_GATE.get("why")), "SKIPPED_KV_KIT_GATE")
                continue
            if not seam_ok:
                skip_arm(k, seam_why, "NOT_INTEGRATED", not_integrated=True)
                continue
            if a.get("ring_needed") and not ring[1]:
                skip_arm(k, f"K {a['k']} needs the QSA ring widening: {ring[2]}", "SKIPPED_NO_RING")
                continue
            if not N.arm_fits(left(), a):
                skip_arm(k, f"left {left() / 60:.1f} min < its full minimum {a['est_min']} + {N.START_RESERVE_SEC // 60} min "
                            "(order " + " > ".join(RUN_KEYS) + ")", "SKIPPED_BUDGET")
                continue
            ok_s, why_s = sseq.check(k)
            ok_p, why_p = pseq.before_arm(a)
            NBDF["patch"], NBDF["seam_seq"] = _patch_rec(pseq), _seam_rec(sseq)
            if not ok_s or not ok_p:
                skip_arm(k, why_s if not ok_s else why_p, "TREE_INVALID_MIXED" if "INVALID" in (why_s + why_p) else "NOT_INTEGRATED",
                         not_integrated=True)
                NBD1.setdefault("tree_notes", []).append(f"{k}: " + (why_s if not ok_s else why_p))
                continue
            r = step(k, a["name"], a["est_min"], lambda k=k: runner(k))
            if k == D1.BASE_KEY:
                bq = N.base_usable(r)
                NBD1["base"] = {"usable": bq[0], "label": bq[1], "why": bq[2], "role": "recorded only: the levers run anyway"}
                R["sections"].setdefault("PROF", {})["nbd1_prof"] = D1.prof_record(r, NBD1_SWITCH)
                if not bq[0]:
                    log("NB-D1: B not a usable base (recorded, the levers run):", bq[1], bq[2])
            save()
    finally:
        try:
            sseq.finish()
        except Exception as e:
            sseq.error = f"finish: {type(e).__name__}: {e}"[:300]
        try:
            pseq.finish()
        except Exception as e:
            pseq.error = f"finish: {type(e).__name__}: {e}"[:300]
        NBDF["patch"], NBDF["seam_seq"] = _patch_rec(pseq), _seam_rec(sseq)
        nbd1_finalize()



# ==================================================================== E34: boot capacity / read-back records, [LIVE] lines, W re-plan, records
def e34_capacity_rec(o, arm, t):
    """Records only (never stops): the boot capacity (pool, block, F + token-equivalent, mc, KV tokens, group sizes, padding),
    the dtype / block / SW read-back per worker, printed explicitly ([E34 READBACK] + the [LIVE] boot line)."""
    try:
        o["e34_capacity"] = E34.capacity(o, t, o.get("workers"))
        rb = E34.readback(arm, t, o.get("workers"), o.get("dflash_readback"))
        o["e34_readback_boot"] = rb
        print("[E34 READBACK]", arm["key"], json.dumps({"cache_config.mamba_ssm_cache_dtype per rank": rb.get("ssm_dtype_per_rank"),
                                                       "cache_config.block_size per rank": rb.get("block_per_rank"),
                                                       "drafter SW block per rank": rb.get("sw_block_per_rank"),
                                                       "GDN state dtypes": [w.get("gdn_state_dtypes") for w in rb.get("workers") or []],
                                                       "GDN unpadded / padded page": [(w.get("gdn_unpadded_page"), w.get("gdn_page")) for w in rb.get("workers") or []],
                                                       "log": {k: v for k, v in (rb.get("log") or {}).items() if k != "lines"},
                                                       "unreadable": rb.get("unreadable"), "real_failure": rb.get("why")}, default=str)[:3000], flush=True)
        for ln in ((rb.get("log") or {}).get("lines") or [])[:12]:
            print("[E34 BOOTLOG]", arm["key"], ln, flush=True)
        if arm["key"] in E34.W_ARMS:
            o["e34_w"] = E34R.get("w")
        print(E34.boot_line((time.time() - T_START) / 60.0, arm["key"], o["e34_capacity"], rb), flush=True)
    except Exception as e:
        o["e34_capacity"] = {"error": f"{type(e).__name__}: {e}"[:300]}


def e34_item_live(o, key, val):
    """A [LIVE] line after every test (print only; never raises)."""
    try:
        if key in E34.LIVE_ITEMS:
            print(E34.item_line((time.time() - T_START) / 60.0, (o or {}).get("arm"), key, val), flush=True)
    except Exception as e:
        print(f"[LIVE] {key}: line error {type(e).__name__}: {e}"[:200], flush=True)


def e34_w_replan(k, a):
    """S16B2W's max_num_seqs from S16B2's measured boot (num_gpu_blocks + the max-concurrency line, the static admission bound
    at 40k, 24 .. 40); S16B2 not readable -> the CPU planner's W (recorded). The re-planned arm is checked (E34.check_arm_plan)
    before it replaces the planned one; a failed check keeps the planned arm (recorded)."""
    r = _arm_rec("S16B2") or {}
    w, info = E34.w_from_capacity(r.get("e34_capacity"))
    src = info.get("source")
    if w is None:
        w, src = int(a["streams"]), "CPU planner W (S16B2's boot capacity unreadable or S16B2 not run)"
    rec = dict(info, w=w, planned=int(a["streams"]), source=src)
    if w != int(a["streams"]):
        try:
            na = E34.lever_arm(D1, N, ARMS[D1.BASE_KEY], k, NBD1_SWITCH, w)
            tmp = dict(ARMS)
            tmp[k] = na
            E34.check_arm_plan(D1, N, T, tmp, NBDF_ARMS["D3s"], NBD1_SWITCH)
            ARMS[k] = na
            rec["replanned"] = True
        except Exception as e:
            rec.update(replanned=False, error=f"{type(e).__name__}: {e}"[:300], w=int(a["streams"]))
    E34R["w"] = rec
    NBD1.setdefault("arms", {})[k] = {x: ARMS[k].get(x) for x in ("name", "port", "streams", "captures", "windows_fixed", "est_min", "cap_min", "extra")}
    log("E34 W:", json.dumps(rec, default=str)[:600])
    print(f"[LIVE {(time.time() - T_START) / 60.0:.1f}] {k} W = {ARMS[k]['streams']} ({src}; S16B2 pool {info.get('pool')}, F {info.get('F')}, "
          f"block {info.get('attn_block')})", flush=True)
    save()
    return ARMS[k]


def _gpu_util():
    while not _gpu_util_stop.is_set():
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20)
            if r.returncode == 0:
                with open(OUT / "nbdf_gpu_util.jsonl", "a") as fh:
                    fh.write(json.dumps({"t": round(time.time() - T_START, 1), "gpus": r.stdout.strip().splitlines()}) + "\n")
        except Exception:
            pass
        _gpu_util_stop.wait(5)


_gpu_util_stop = threading.Event()
if not RERUN and not os.environ.get("RC2_UNIT"):
    if NVSMI_OK:
        threading.Thread(target=_gpu_util, daemon=True).start()
    nbd1_main()                                  # E25 NB-D1
    _gpu_util_stop.set()
    R["host_phase"] = HOST_PHASE
    R["prefill"] = PREFILL
    R["minutes"] = round((time.time() - T_START) / 60, 1)
    save()
save()
