# Domdolus notebook — cell10 (QSA DCP + improvements patch/config cell) — full inventory

Source: `Build/domdolus_cells/cell10.txt` (144,188 bytes, 339 lines; read completely, every line, including the long
edit-list lines 45-71). Date 2026-11-04. Read-only analysis; nothing was run.

What cell10 is: it is NOT the server launch itself. It is the "QSA DCP" patch + config cell. It (a) defines ~37 config
constants, (b) holds 25 anchor-replace edit lists plus one new file (`ops/qsa_dcp.py`) for stock vLLM 0.31.0 unpacked at
`/tmp/site-vllm/vllm`, (c) applies them idempotently with `.qsa_dcp.bak` backups and `py_compile`, (d) exports ~30
`FLASHNEXT_*` / vLLM env vars that the patched vLLM reads at server start, (e) builds `MODEL_DIR` (optionally an FP8 overlay
symlink folder), (f) prepares a background import check `IMPORT_CHECK = (code, env)` that step 6 (cell43) runs. The actual
`vllm.entrypoints.openai.api_server` command is built in cell43 (`build_cmd`), which also reads cell10's
`CUDAGRAPH_CAPTURE_SIZES` global.

---

## 1. Config constants (lines 1-36) — every one

| Constant | Value | What it does (evidence) |
|---|---|---|
| `DCP_SIZE` | `2` | Decode-context-parallel size. 2 → patches applied and `FLASHNEXT_DCP_SIZE=2` exported, which becomes the default of `--decode-context-parallel-size` (ARGS_EDITS). 1 → all patched files restored from `.bak`, `ops/qsa_dcp.py` removed, all env vars popped. `assert DCP_SIZE == 2, "Flash-Next has 2 KV heads at TP 4: the only valid DCP size is 2"`. |
| `COMPRESSED_BLOCK_FACTOR` | `8` | Exported as `FLASHNEXT_QSA_COMPRESSED_BLOCK_FACTOR`. Under DCP, the QSA compressed-key (indexer) cache block = `block_size * FACTOR`. Allowed 1/2/4/8 (8 is the max). Comment: default block made "21 full-width blocks per 32K request for ~1/8 of their bytes (measured on 4 x L4: 13.85x instead of 16.73x)", factor 8 → "3 blocks per request". |
| `GPU_UTIL_OVERRIDE` | `0.945` | Replaces `GPU_UTIL` (step-1 KIND value for flashnext is `gpu_util=0.92`, cell06 line 9) → `--gpu-memory-utilization 0.945` in cell43. Assert `0.5 < x < 0.99`. |
| `SPARSE_INDEXER_MAX_LOGITS_MB` | `128` | Exported as `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128` (cap on the sparse-indexer FP32 logits workspace; read by vLLM `envs`, printed by the import check). |
| `DCP_COMM_BACKEND` | `"a2a"` | Exported `FLASHNEXT_DCP_COMM=a2a` → default of `--dcp-comm-backend` (vLLM default is `ag_rs`). Allowed `None/"ag_rs"/"a2a"`. Used by `dcp_combine` (`dcp_a2a_lse_reduce`). |
| `MAX_CUDAGRAPH_CAPTURE_SIZE` | `512` | Exported `FLASHNEXT_MAX_CUDAGRAPH=512` → default of `--max-cudagraph-capture-size`. |
| `CUDAGRAPH_CAPTURE_SIZES` | `[1, 2, 4, 8, 16, 32, 48, 64, 72, 128, 256, 512]` | Asserted max == 512. cell43 passes it as `--cudagraph-capture-sizes ...` (`if globals().get("CUDAGRAPH_CAPTURE_SIZES")`). Note the explicit 72 (≈ harness CONCURRENCY 70). |
| `MAX_BATCHED_TOKENS` | `1024` | Exported `FLASHNEXT_MAX_BATCHED=1024`; applied AFTER CLI parse in `EngineArgs.from_cli_args`, so it overrides cell43's explicit `--max-num-batched-tokens 4096` (`MAX_BATCHED = 4096`). Prefill chunk = 1024 tokens. Also = trim "chunk" in the admission cap. |
| `MM_MAX_PIXELS` | `2097152` | `FLASHNEXT_MM_PROCESSOR_KWARGS='{"max_pixels": 2097152}'` merged into `engine_args.mm_processor_kwargs` (image token cap; = 2048×1024 px). |
| `DISABLE_VIDEO` | `True` | `FLASHNEXT_LIMIT_MM='{"video": 0}'` merged into `limit_mm_per_prompt` (images stay on). |
| `EXPANDABLE_SEGMENTS` | `True` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. With KV offload, the patch forces `enable_cumem_allocator=True` (vLLM refuses a KV connector with expandable segments otherwise). |
| `FP8_DENSE_OVERLAY` | `False` | If True: build `/tmp/flashnext_fp8` of symlinks to MODEL_PATH with 5 files replaced from an overlay dataset (`dense-00000/1/2.safetensors`, `config.json`, `model.safetensors.index.json`). Off → model used as attached. |
| `FP8_OVERLAY_PATH` | `/kaggle/input/datasets/darrenamadeusmartin/qwen38-flash-next-fp8-dense` | Overlay (FP8 dense, no HC) — only used if overlay on and `FP8_HC_LMHEAD` False. |
| `FP8_HC_LMHEAD` | `True` | Exports `FLASHNEXT_FP8_HC=1`; applies HC_EDITS + LMHEAD_EDITS: hyper-connection mixers (`input_mix_weight_down_block_inject`, `input_mix_weight_down`, `input_mix_weight_up`) and `lm_head` get the checkpoint's `quant_config` instead of `None` (FP8 instead of BF16). With overlay off, the cell verifies the attached model's `config.json` has `re:` targets matching `language_model.lm_head`, `...layers.0.attn_hyper_connection.input_mix_weight_down`, `...layers.47.mlp_hyper_connection.input_mix_weight_up`, `...hyper_connection_mixer.input_mix_weight_down`; else error "attach VERSION 2 of qwen38-flash-next-q2-gsq-vllm (FP8 merged in) ... `.../pytorch/default/2`". |
| `FP8_HC_OVERLAY_PATH` | `/kaggle/input/datasets/darrenamadeusmartin/qwen38-flash-next-fp8-dense-hc` | Overlay with FP8 dense + HC + lm_head (used only if `FP8_DENSE_OVERLAY` True). |
| `MAMBA_SSM_FP16_SR` | `True` | `FLASHNEXT_MAMBA_SSM_DTYPE=float16` → `engine_args.mamba_ssm_cache_dtype="float16"` (post-parse override) + GDN_EDITS: FP16 recurrent-state stores use stochastic rounding. Default is "checkpoint (FP32)" per the final print. |
| `MAMBA_SSM_BF16` | `False` | Alternative: `bfloat16` plain. Mutually exclusive with FP16_SR (assert). |
| `PROFILE` | `False` | If True `FLASHNEXT_PROFILER_DIR=/kaggle/working/prof` → torch profiler config (CLI default + post-parse). |
| `CUSTOM_ALL_REDUCE` | `False` | Must stay False: `assert not CUSTOM_ALL_REDUCE, ("...vLLM refuses custom all-reduce on more than two PCIe-only GPUs, and forced on it measured 5x slower than NCCL...")`. |
| `MEM_TRIMS` | `True` | `FLASHNEXT_MEM_TRIMS=1` + QSA_TRIM_EDITS + HUMMING_TRIM_EDITS (rotary cache to max_model_len, one shared top-k buffer, Humming pad 64 for HC layers). |
| `PREFIX_MATCH_UNIT` | `None` | → `FLASHNEXT_PREFIX_MATCH_UNIT = PREFIX_MATCH_UNIT or (800 if PREFIX_REUSE else None)` = **800** → default of `--prefix-match-unit` (fine-grained prefix-hash unit, tokens). |
| `PREFIX_RETENTION` | `None` | → `FLASHNEXT_PREFIX_RETENTION = "dense"` (because PREFIX_REUSE) → `--prefix-cache-retention-interval` default `None` ("dense" maps to None; an int would set an interval). |
| `PREFIX_REUSE` | `True` | Enables PREFIX_REUSE_EDITS + `FLASHNEXT_DECODE_PARTIAL=1` + the 800 / dense defaults above. |
| `QAR` | `None` | Compressed all-reduce mode (`"a2a"` / `"rs"`); **off**. Patch QAR_EDITS is always installed but inert (`_FN_QAR_MODE = os.environ.get("FLASHNEXT_QAR") or None`). |
| `QAR_CODEC` | `"int8"` | Exported only if QAR set. int8 (absmax/127) or fp8 e4m3 (absmax/448). |
| `QAR_BLOCK` | `32` | Values per FP32 scale; must divide 128. Exported only if QAR. |
| `QAR_MIN_MB` | `1.5` | Messages below this stay BF16 (all decode steps); assert ≥ 1.0. Exported only if QAR. |
| `SINK_WINDOW` | `None` | Sliding mode (sink, window), window ≥ 2,052 ("the QSA budget 2,048 + 4"). Off. |
| `SINK_JUMP` | `("auto", 49152, 65536)` | Jump mode: `FLASHNEXT_QSA_SINK=auto`, `FLASHNEXT_QSA_JUMP_KEEP=49152`, `FLASHNEXT_QSA_JUMP_LIMIT=65536`. Sink = all tokens before the first `<\|im_start\|>assistant` of the prompt. When QSA attention would see 65,536 tokens it keeps sink + most recent so that it sees 49,152; regrows to 65,536; cut every 16,384 tokens. DeltaNet layers keep the whole conversation. |
| `TRIM_FREE` | `True` | `FLASHNEXT_TRIM_FREE=1` + TF_* edits: the cut-away middle's KV blocks of QSA groups are actually FREED (FnTrimGapManager), admission cap lowered, gap-aware prefix hits. Requires SINK_JUMP auto (assert). Final print: "set MAX_LEN = 262144 at the top of step 6 (else the harness trims first and nothing changes)". |
| `QSA_SKIP_GAP` | `True` | `FLASHNEXT_QSA_SKIP_GAP=1` (only if SINK_JUMP/SINK_WINDOW): indexer scoring kernels skip the cut-away tiles (decode: COMPACT column remap; prefill: tile jump) — "Same selection, bit for bit". |
| `KV_OFFLOAD_GIB` | `30` | `FLASHNEXT_KV_OFFLOAD_GIB=30` → `engine_args.kv_offloading_size=30.0` + `enable_cumem_allocator=True`; `FLASHNEXT_OFFLOAD_LAZY=1`. Pinned host RAM KV tier. |
| `OFFLOAD_EVICT` | `True` | With KV_OFFLOAD → `_EV=True`, `FLASHNEXT_OFFLOAD_EVICT=1`: connector forced to `SimpleCPUOffloadConnector` with `lazy_offload=True` ("copy only before eviction"), segment-ordered freeing, DeltaNet state holding. Needs TRIM_FREE (assert). RAM check: `MemAvailable >= KV_OFFLOAD_GIB + 8` GiB else error. |
| `DIAG_ENCODER_MISS` | `True` | Encoder-cache store/free history + error log on "Encoder cache miss" (log only). |
| `DIAG_WEIGHTS` | `True` | `FLASHNEXT_DIAG_WEIGHTS=1`: after model load logs per-module GPU memory ("WEIGHTS DIAG", top 30) in both V1 and V2 model runners (log only). |

Derived: `_EV = bool(KV_OFFLOAD_GIB and OFFLOAD_EVICT)` = True. `VROOT = Path(os.environ.get("QSA_DCP_VLLM_ROOT", "/tmp/site-vllm/vllm"))`, `BAK = ".qsa_dcp.bak"`, `OPS = VROOT/"models/qwen4_exp/nvidia/ops/qsa_dcp.py"`.

### 1b. Effective environment exported with the shipped values

| Env var | Value | Consumer |
|---|---|---|
| `FLASHNEXT_DCP_SIZE` | `2` | `EngineArgs.decode_context_parallel_size` default + CLI default |
| `FLASHNEXT_QSA_COMPRESSED_BLOCK_FACTOR` | `8` | `qsa_cache.py get_kv_cache_spec` |
| `FLASHNEXT_DCP_COMM` | `a2a` | CLI `--dcp-comm-backend` default |
| `FLASHNEXT_MAX_CUDAGRAPH` | `512` | CLI `--max-cudagraph-capture-size` default |
| `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB` | `128` | vLLM envs |
| `FLASHNEXT_MAX_BATCHED` | `1024` | post-parse override of `max_num_batched_tokens` |
| `FLASHNEXT_MM_PROCESSOR_KWARGS` | `{"max_pixels": 2097152}` | post-parse merge |
| `FLASHNEXT_LIMIT_MM` | `{"video": 0}` | post-parse merge |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | torch |
| `FLASHNEXT_MAMBA_SSM_DTYPE` | `float16` | post-parse `mamba_ssm_cache_dtype` |
| `FLASHNEXT_FP8_HC` | `1` | hyperconnection.py / model.py |
| `FLASHNEXT_DIAG_WEIGHTS` | `1` | model runners |
| `FLASHNEXT_MEM_TRIMS` | `1` | qsa.py, humming linear.py |
| `FLASHNEXT_PREFIX_MATCH_UNIT` | `800` | CLI `--prefix-match-unit` default |
| `FLASHNEXT_PREFIX_RETENTION` | `dense` | CLI `--prefix-cache-retention-interval` default → None |
| `FLASHNEXT_DECODE_PARTIAL` | `1` | single_type_kv_cache_manager |
| `FLASHNEXT_QSA_SINK` | `auto` | qsa_indexer.py, model_state.py, trim |
| `FLASHNEXT_QSA_JUMP_KEEP` | `49152` | qsa_indexer.py, trim |
| `FLASHNEXT_QSA_JUMP_LIMIT` | `65536` | qsa_indexer.py, trim |
| `FLASHNEXT_TRIM_FREE` | `1` | trim manager |
| `FLASHNEXT_QSA_SKIP_GAP` | `1` | scoring kernels |
| `FLASHNEXT_KV_OFFLOAD_GIB` | `30` | post-parse `kv_offloading_size` |
| `FLASHNEXT_OFFLOAD_LAZY` | `1` | OffloadingConnector scheduler patch |
| `FLASHNEXT_OFFLOAD_EVICT` | `1` | config/vllm.py, block pool, coordinator, simple offload manager |
| unset (popped): `FLASHNEXT_PROFILER_DIR`, `FLASHNEXT_CUSTOM_ALL_REDUCE`, `FLASHNEXT_QAR`, `FLASHNEXT_QAR_CODEC`, `FLASHNEXT_QAR_BLOCK`, `FLASHNEXT_QAR_MIN_MB`, `FLASHNEXT_QSA_WINDOW` | — | — |

Env vars read by the patches but never set by cell10 (defaults apply): `FLASHNEXT_QSA_SW_CHECK` (selection check, off),
`FLASHNEXT_TRIM_CHECK` (null-block check, off), `FLASHNEXT_OFFLOAD_PRESSURE` (default `0.8`), `FLASHNEXT_OFFLOAD_AHEAD`
(default `4`), `VLLM_USE_SIMPLE_KV_OFFLOAD` (stock), `QSA_DCP_VLLM_ROOT` (VROOT override).

CLI-default vs override semantics (important for rebuild): DCP size, DCP comm, max cudagraph, prefix match unit,
retention, profiler are only CLI *defaults* (an explicit flag in cell43 would win). `FLASHNEXT_MAX_BATCHED`,
`MM_PROCESSOR_KWARGS`, `MAMBA_SSM_DTYPE`, `LIMIT_MM`, `PROFILER_DIR`, `CUSTOM_ALL_REDUCE`, `KV_OFFLOAD_GIB` are applied
after `from_cli_args` and win over explicit flags ("overrides that must win over flags the notebook's step 6 passes explicitly").

---

## 2. Techniques / patches (target file → edit list → controlling flag)

`TARGETS` (applied in this order; each anchor must occur exactly once, else `RuntimeError(... "Is this vLLM 0.31.0? Nothing was changed in this file.")`; already-patched text is skipped):

| # | vLLM file (under `/tmp/site-vllm/vllm/`) | Edits | Flag |
|---|---|---|---|
| 1 | `models/qwen4_exp/nvidia/qsa.py` | QSA_EDITS (+ QSA_TRIM_EDITS) | DCP_SIZE=2 (+MEM_TRIMS) |
| 2 | `models/qwen4_exp/common/qsa_cache.py` | CACHE_EDITS | DCP |
| 3 | `engine/arg_utils.py` | ARGS_EDITS | DCP (always) |
| 4 | `v1/worker/gpu/mm/encoder_cache.py` | DIAG_CACHE_EDITS | DIAG_ENCODER_MISS |
| 5 | `v1/worker/gpu/mm/encoder_runner.py` | DIAG_RUNNER_EDITS | DIAG_ENCODER_MISS |
| 6 | `third_party/flash_linear_attention/ops/fused_recurrent.py` | GDN_EDITS | MAMBA_SSM_FP16_SR |
| 7 | `models/qwen4_exp/nvidia/hyperconnection.py` | HC_EDITS | FP8_HC_LMHEAD |
| 8 | `models/qwen4_exp/nvidia/model.py` | LMHEAD_EDITS | FP8_HC_LMHEAD |
| 9 | `v1/worker/gpu/model_runner.py` | DIAG_WEIGHTS_V2_EDITS | DIAG_WEIGHTS |
| 10 | `v1/worker/gpu_model_runner.py` | DIAG_WEIGHTS_V1_EDITS | DIAG_WEIGHTS |
| 11 | `model_executor/layers/quantization/utils/humming/linear.py` | HUMMING_TRIM_EDITS | MEM_TRIMS |
| 12 | `v1/core/single_type_kv_cache_manager.py` | PREFIX_REUSE_EDITS + TF_SINGLE_EDITS + EV_SINGLE_EDITS | PREFIX_REUSE / TRIM_FREE / _EV |
| 13 | `distributed/device_communicators/cuda_communicator.py` | QAR_EDITS | always installed; active only if `FLASHNEXT_QAR` |
| 14 | `models/qwen4_exp/nvidia/ops/qsa_indexer.py` | SW_EDITS | always installed; active if SINK_* env |
| 15 | `models/qwen4_exp/nvidia/indexer_qsa.py` | SW_INDEXER_EDITS | always |
| 16 | `models/qwen4_exp/nvidia/model_state.py` | SW_STATE_EDITS | always; active if `FLASHNEXT_QSA_SINK=auto` |
| 17 | `v1/core/kv_cache_coordinator.py` | TF_COORD_EDITS + EV_COORD_EDITS | TRIM_FREE / _EV |
| 18 | `v1/core/kv_cache_manager.py` | TF_KVM_EDITS | TRIM_FREE |
| 19 | `v1/core/sched/scheduler.py` | TF_SCHED_EDITS | TRIM_FREE |
| 20 | `distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py` | OFF_EDITS | KV_OFFLOAD_GIB |
| 21 | `v1/core/block_pool.py` | EV_POOL_EDITS | _EV |
| 22 | `v1/simple_kv_offload/manager.py` | EV_SIMPLE_EDITS | _EV |
| 23 | `config/vllm.py` | EV_CONFIG_EDITS | _EV |
| + | NEW `models/qwen4_exp/nvidia/ops/qsa_dcp.py` | `QSA_DCP_PY` written + compiled | DCP |

### 2.1 QSA decode-context-parallel (DCP=2) — the main KV-capacity lever
- Model facts stated: "Flash-Next has 2 KV heads at TP 4" → each KV head is shared by 2 TP ranks; DCP group = that pair. Each rank stores only its share of every request's QSA KV tokens: "logical position t is owned by rank (t // I) % W at local position (t // (I*W)) * I + t % I", `I = cp_kv_cache_interleave_size` (read from `parallel_config`, not set in cell10).
- The indexer still selects over the whole sequence (its caches are NOT sharded). Each rank filters the selected positions it owns (`_dcp_filter_indices_kernel`, Triton, compacts kept indices to the front, `-1` elsewhere, writes count column `TOPK`), attends with all-gathered query heads (`group.all_gather(query[:num_tokens].contiguous(), dim=1)`), produces FP32 normalized partial + log2 LSE via stock `_qsa._qsa_sparse_paged_gqa_splitk_kernel` (forced `splits = max(2, min(num_splits, num_tiles))`) and new `_dcp_local_merge_kernel`; then vLLM `self.dcp_combine(partial, lse, group, is_lse_base_on_e=False)`; output gate applied: `merged.to(torch.bfloat16).float() * torch.sigmoid(gate.float())` ("Same rounding as the unpatched kernel").
- Split-K tuned constants (in `qsa.py` impl): `_QSA_DCP_SPLITS = 4`, `_QSA_DCP_BLOCK_N = 32`, `_QSA_DCP_WARPS = 8` — "12 query heads over its tokens, measured on L4 (sm_89)". Local merge `num_warps=2, num_stages=1`; split-K `num_stages=2`; filter `num_warps=4`.
- Edits in `qsa.py`: `supports_dcp` False→True on `Qwen4ExpQSAFlashAttentionImpl` and on layer class `Qwen4ExpQSAAttention`; removes the `NotImplementedError("Qwen4Exp QSA does not support decode context parallelism")`; `forward_qsa` branches to `_forward_qsa_dcp` when `dcp_world_size > 1`; layer-level refusal kept only for prefill CP.
- `qsa_cache.py`: ring buffer `CircularBufferSpec(... dcp_sharded=False)`; compressed keys `MLAAttentionSpec(block_size=block_size*FACTOR, ..., tokens_per_state=self.compress_ratio, dcp_sharded=False)` under DCP — they land in their own KV-cache group; widening blocks fixes the packed-layout waste.
- Comm backend `a2a` (vLLM `dcp_a2a_lse_reduce`) instead of default `ag_rs`.
- `qsa_dcp_combine` = single-process reference for 1-GPU tests ("Verified on one L4 by simulating W=2").

### 2.2 FP16 DeltaNet (GDN) recurrent state with stochastic rounding (`fused_recurrent.py`)
- Adds `_qsa_sr_hash` (PCG-style integer hash) and `_qsa_sr_fp16`: "Unbiased FP32 -> FP16 rounding of the recurrent state: add 13 random bits below FP16's mantissa, clear them" (`bits = (bits + (hash & 0x1FFF)) & 0xFFFFE000`). Key mixes element index, state bits and `b_v`.
- Applied at 3 store sites: fused_recurrent fwd in-place state, fused_recurrent fwd per-token states, packed decode — only `if p_ht.dtype.element_ty == tl.float16`.
- Combined with `mamba_ssm_cache_dtype="float16"` (default would be FP32 per final print) → state cache bytes halved.

### 2.3 FP8 hyper-connection mixers + FP8 lm_head (`hyperconnection.py`, `model.py`)
- `_flashnext_fp8_hc_quant_config()` returns `get_current_vllm_config().quant_config` when `FLASHNEXT_FP8_HC=1`; used for `input_mix_weight_down_block_inject`, `input_mix_weight_down`, `input_mix_weight_up` (were `quant_config=None`). `ParallelLMHead(... quant_config=vllm_config.quant_config if FLASHNEXT_FP8_HC=="1")`.
- Requires a checkpoint whose `quantization_config.config_groups` targets those modules (model v2 or the `-hc` overlay).

### 2.4 Memory trims (`qsa.py`, humming `linear.py`)
- Rotary cache: `max_position = min(config.max_position_embeddings, max_model_len)`.
- One shared `topk_indices_buffer` (`torch.empty(max_tokens, packed_output_width, int32)`) for all QSA layers via `_FLASHNEXT_SHARED_TOPK` dict keyed by shape — "each QSA layer writes and reads its selection before the next layer runs".
- Humming linear: `pad_n_to_multiple=64 if _flashnext_small_pad(layer) else 256`, `pad_k_to_multiple=64 ... else 128` for layers whose prefix contains `hyper_connection` ("FP8 hyper-connection mixers (336 x 10,240 and 10,240 x 320)").

### 2.5 Prefix reuse inside generated text (`single_type_kv_cache_manager.py`)
- With `FLASHNEXT_DECODE_PARTIAL=1`, the partial cache entry boundary becomes `max(prompt boundary, latest hash boundary of generated tokens)`: `latest = min(num_tokens, len(request.block_hashes) * hash_block_size) // hash_block_size * hash_block_size`, so "a follow-up call can resume inside the previous reply (the Mamba groups keep those states under dense retention...)".
- Paired with `--prefix-match-unit 800` and dense retention (Mamba states kept at every boundary).

### 2.6 Compressed all-reduce "QAR" (`cuda_communicator.py`) — installed, OFF
- Hook: first line of `CudaCommunicator.all_reduce`: `_fn_out = _fn_qar_maybe(self, input_)`.
- Active only for `tp:` groups, BF16/FP16 CUDA tensors ≥ `FLASHNEXT_QAR_MIN_MB` (1.5 MB), not during CUDA-graph capture; raises if PyNccl missing ("no silent fallback").
- Modes: `a2a` = int8-quantize partials, grouped send/recv, sum own exact shard + dequantized peers in FP32, requantize, all-gather → "~56 % of the BF16 bytes (int8, block 32)"; `rs` = BF16 NCCL reduce-scatter then 8-bit all-gather → "~78 %". 3 fused Triton kernels (`_fn_qar_pack_kernel`, `_fn_qar_sum_repack_kernel`, `_fn_qar_unpack_kernel`, `_FN_QAR_ROWS = 64`), bit-exact vs torch reference `flashnext_qar_ref`.
- Measured context in comments: "Kaggle's 4 x L4 move NCCL data through host memory at ~4.9 GB/s", "3.4 ms at 10.5 MB vs 85 us latency", "a ~1.7 ms (5.2 MB) all-reduce"; torch-op version cost "~430-700 us per call". Quality: "fp8 / 128 added 2-4x vLLM's own run-to-run prompt-logprob noise, int8 / 32 ~1.5-2x".

### 2.7 QSA sink + window / jump selection mask (`ops/qsa_indexer.py`, `indexer_qsa.py`, `model_state.py`)
- `_fn_sw_gap` computes per row (sink group, first recent group) in compressed groups. Jump mode: `lim_c, keep_c = LIMIT//r, KEEP//r; period = lim_c - keep_c; n_cuts = floor((v - lim_c)/period) + 1; start = lim_c + (n_cuts-1)*period - (keep_c - sink_c)`; always keeps ≥ `token_topk // compress_ratio + 1` recent groups; never cuts into the sink.
- `_fn_sink_window_mask` sets cut-away scores to -1 before `_topk` (decode and prefill selection). Sink "auto": `model_state.py` `add_request` finds the first `<|im_start|>` followed by `assistant` tokens in the prompt; `prepare_inputs` copies per-request sinks to a per-token buffer (`repeat_interleave`, buffer `max_num_tokens + 1024`). `indexer_qsa.py`: prefill rows start after decode rows (`_fn_row_start=num_decode_tokens`).
- SKIP_GAP: decode scoring kernel gets `COMPACT=True` with `gap_lo_ptr`/`gap_shift_ptr` (column j → group j or j+shift; `visible_blocks - shift`; selected indices shifted back with `block_indices.add_(...)`); prefill scoring kernel jumps over tiles inside every row's gap (`SKIP_GAP=True`, `gap_lo_ptr`/`gap_hi_ptr`); warm-up compiles the compact variant (`TritonWarmupTensor`).
- Constraint values: QSA budget (`indexer_budget`) = 2,048, compress ratio = 4 (from "window >= 2,052 (the QSA budget 2,048 + 4)").

### 2.8 Trim-free: actually freeing the cut-away KV (`single_type_kv_cache_manager.py`, coordinator, kv_cache_manager, scheduler)
- `FnTrimGapManager(FullAttentionManager)` replaces FullAttentionManager for all full-attention groups when `FLASHNEXT_TRIM_FREE=1` (`get_manager_for_kv_cache_spec`).
- `remove_skipped_blocks`: frees blocks in the gap computed at one cache-hit boundary BEFORE the last (`max(0, processed // a * a - a)`; "session 11 fix" — earlier freeing made follow-ups recompute 50,725 of 50,725 tokens).
- Admission cap: `max_admission_blocks_per_request = cdiv(sink, bs) + 2 + cdiv(LIMIT + chunk, bs)` — "a trimmed request holds at most the sink + LIMIT + one chunk (vLLM would reserve the whole prompt)".
- `find_longest_cache_hit` reimplemented: hits may have missing blocks only inside the gap (bisect over missing list), returns null blocks in the gap; fine-grained tail handling; `get_num_common_prefix_blocks` returns 0. `allocate_external_computed_blocks` puts null blocks in the gap for RAM restores.
- Coordinator: gap manager excluded from the downward-closed shortcut (`and not getattr(manager_cls, "fn_trim_gap", False)`). KV manager sets `_FN_TRIM["cur_sink"]` around `find_longest_cache_hit` and registers the sink before `remove_skipped_blocks`. Scheduler calls `fn_trim_configure(vllm_config)` (reads `indexer_budget`, `indexer_compress_ratio`, `max_num_batched_tokens`) and publishes `usage_fn` = kv_cache_manager.usage.

### 2.9 KV offload to host RAM
- (a) OFF_EDITS on `OffloadingConnector` scheduler: gap-aware lookup (`_fn_gap_lookup`), restore only real (non-null) blocks, never store gap chunks, DeltaNet checkpoints only at `lcm(tokens_per_chunk)` boundaries, lazy (`running requests copied only above 80 % GPU KV use, finished calls always`), re-check hit at final length (`_fn_lookup`, up to 256 shrink iterations). Requires `blocks_per_chunk == 1`.
- (b) EV_* "evict offload" (active, because `OFFLOAD_EVICT=True`): `config/vllm.py` forces `SimpleCPUOffloadConnector` + `"lazy_offload": True` with `cpu_bytes_to_use = 30 GiB`. So with shipped values the OffloadingConnector edits (a) are installed but the SimpleCPUOffload path is the one used.
  - `block_pool.py`: `_fn_on_evict` callback on eviction.
  - Cut-away gap blocks get their cache entry dropped (go to FRONT of free queue, never copied to RAM; counter `gap_dropped`).
  - DeltaNet states retired from running requests: `_fn_retire` holds the 2 newest segment-boundary states + newest other state; older ones dropped.
  - Coordinator `_fn_seg_order`: frees a finished call newest-segment-first, all groups of a segment together; only the newest DeltaNet state per group is copied ("one per segment would add ~80 % to a conversation's RAM on Kaggle").
  - Simple manager: `_target_free *= FLASHNEXT_OFFLOAD_AHEAD (4)` ("Colab E1: 650 blocks lost in the first 10 min with 20"); copies whole segments only near the eviction point (`_fn_pick`, cursor reset each step); RAM lookup knows sink + start token (`cur_off`, must be a multiple of the QSA block); restore skips GPU null blocks; 60-s report lines "flashnext evict offload (last 60 s | total)" + census of parked GPU blocks.

### 2.10 Diagnostics (log only)
- Encoder cache: `_QsaDcpTrackedDict` records set/pop history per mm hash; on miss logs "ENCODER MISS DIAG ...".
- WEIGHTS DIAG: per-module GPU bytes (deduplicated storages, `.N` layer collapse), top 30, plus torch allocated/reserved.

---

## 3. External dependencies (as visible in cell10)

- Stock **vLLM 0.31.0** source unpacked at `/tmp/site-vllm/vllm` by step 2 (anchor error text names 0.31.0; `assert (VROOT / "models/qwen4_exp/nvidia/qsa.py").exists()`). Requires the Qwen4Exp (Flash-Next) model code with QSA, `common/qsa_cache.py`, Humming quantization utils, `third_party/flash_linear_attention`, `v1/simple_kv_offload/manager.py`, `SimpleCPUOffloadConnector`, `--prefix-match-unit`, `--prefix-cache-retention-interval`, `--dcp-comm-backend`, `cp_kv_cache_interleave_size`.
- Model: attached `MODEL_PATH` (step 1/cell05). With `FP8_HC_LMHEAD=True` and no overlay, it must be **VERSION 2 of `qwen38-flash-next-q2-gsq-vllm`** ("FP8 merged in", folder `.../pytorch/default/2`).
- Optional datasets: `/kaggle/input/datasets/darrenamadeusmartin/qwen38-flash-next-fp8-dense` and `.../qwen38-flash-next-fp8-dense-hc` (5 overlay files: `dense-00000.safetensors`, `dense-00001.safetensors`, `dense-00002.safetensors`, `config.json`, `model.safetensors.index.json`); auto-discovery fallback `find /kaggle/input -maxdepth 8 -name dense-00000.safetensors` + `"group_1" in config_groups`.
- Host resources: `/proc/meminfo` MemAvailable ≥ 30 + 8 GiB (evict path); non-evict path also checks `/dev/shm` free ≥ 1.05 × GiB.
- Output paths: `/kaggle/working/vllm-server-0.log`, `/kaggle/working/prof` (profiling off), `/tmp/flashnext_fp8` (overlay off).
- Uses `vllm_env()` from earlier cells if present, else `PYTHONPATH=/tmp/site-vllm`.
- No pip wheels, tar files or kernel caches are referenced in cell10 itself (they live in earlier cells). Triton is used via `vllm.triton_utils`.
- Internal dev references in comments (not shipped): `dev/patches/qsa_dcp/README.md`, `qar.py`, `qar_edits.py`, `tools/qar_bench_cell.py`, `sink_window_edits.py`, `trim_free_edits.py`, `offload_edits.py`, `evict_offload_edits.py`, `docs/10`; dev sessions 8, 9, 11, 17 mentioned.

---

## 4. Memory accounting data available from cell10

Numbers stated in cell10:
- GPU util **0.945** of each L4 (step-1 default 0.92).
- QSA attention KV: DCP=2 → each rank of a KV-head pair stores half the tokens (interleave I) → ~2x QSA-KV token capacity. Indexer compressed keys and the ring buffer stay whole on every rank (`dcp_sharded=False`), in their own KV group.
- Compressed-key block = block_size × 8; "21 full-width blocks per 32K request" → "3 blocks per request"; measured max concurrency "13.85x instead of 16.73x" (vLLM's `kv_cache_max_concurrency` at a 32K-token request, i.e. roughly 16.73 × 32K ≈ 548K tokens of pool in that test config — derived, the test's max_len/util are not stated in cell10).
- QSA budget 2,048 selected tokens, compress ratio 4; "12 query heads" per DCP rank partial; "2 KV heads at TP 4"; layer index 47 exists in target regex (48 layers). Neighbor cell15 says 12 QSA attention layers.
- HC mixer shapes 336 × 10,240 and 10,240 × 320 (padding trimmed to 64).
- Per-request QSA KV upper bound with trim: `sink + LIMIT (65,536) + chunk (1,024)` tokens + 2 blocks (admission formula). Attention window in steady state oscillates 49,152 → 65,536 tokens beyond the sink; DeltaNet keeps all.
- DeltaNet state cache FP16 (vs FP32 default) → half.
- Shared top-k buffer: one `max_tokens × (TOPK+1)` int32 for all QSA layers instead of one per layer.
- RAM KV tier: 30 GiB pinned host memory (`cpu_bytes_to_use = kv_offloading_size * (1 << 30)`), RAM check needs +8 GiB headroom.
- Prefill chunk 1,024 tokens (activation/workspace size), indexer logits cap 128 MB, CUDA graphs up to 512.
- From cell43 (context only): `MAX_LEN = 1048576` with `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `MAX_SEQS = 128`, `KV_DTYPE` = `fp8` (KIND), `--mamba-cache-mode align`, `MTP_TOKENS = 0`, `--engram-config {"cpu_offload": true}`, expert parallel on; cell10 prints "set MAX_LEN = 262144" for trim mode (cell43 ships 1,048,576, i.e. even larger).

Not in cell10 (must be measured from the logs): weights GiB per GPU, KV GiB, num_blocks/tokens. Cell10 tells the user to grep: `GPU KV cache size|kv cache group sizes|kv lcm block sizes|Available KV cache memory|fine-grained prefix`, `Model loading took|WEIGHTS DIAG`, and cell09 adds `FLASHNEXT KVDIAG` (per-group block/page/pages-per-task lines and pool num_blocks/tokens).

---

## 5. Launch sequence

cell10 order:
1. Constants; resolve `VROOT`; assert QSA file exists.
2. Define `QSA_DCP_PY` + 25 edit lists; `_EV`; `TARGETS`.
3. If `DCP_SIZE == 1`: restore all `.bak`, delete `qsa_dcp.py`, pop env, reset `MODEL_DIR`/`GPU_UTIL`, done.
4. Else: asserts (DCP=2, factor, comm backend, util range, FP16/BF16 exclusive, TRIM_FREE needs auto SINK_JUMP, retention, QAR args, SINK_WINDOW ≥ 2052, SINK_JUMP keep−sink ≥ 2052 & limit > keep, not both modes, EVICT needs TRIM_FREE); RAM check for offload; custom-AR assert.
5. For each target: read original (from `.bak` if exists → patches always re-applied onto pristine text), apply edits, back up, write if changed, `py_compile`.
6. Write + compile `ops/qsa_dcp.py`.
7. Export env vars (table 1b); `GPU_UTIL = 0.945`.
8. MODEL_DIR: overlay symlink dir or attached model + FP8-target verification.
9. Build `IMPORT_CHECK` (parses CLI with `--tensor-parallel-size 4 --max-num-batched-tokens 4096`, prints effective dcp, comm, cudagraph max, logits MB, batched tokens, mm kwargs, limits, alloc conf, mamba dtype, profiler, custom AR, prefix unit, retention, and asserts/prints every patch installed + active). Runs in background at step 6 and must pass before the server counts as ready.
10. Print summary and grep instructions.

Server (cell43 `build_cmd`, context): `python -m vllm.entrypoints.openai.api_server --model MODEL_DIR --served-model-name Qwen/Qwen3.8-Flash-Next --host 127.0.0.1 --port 8000 --tensor-parallel-size 4 --max-model-len 1048576 --gpu-memory-utilization 0.945 --max-num-seqs 128 --max-num-batched-tokens 4096 (→1024 by env) --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --default-chat-template-kwargs {"preserve_thinking": true} --cudagraph-capture-sizes 1 2 4 8 16 32 48 64 72 128 256 512 --disable-custom-all-reduce --enable-prefix-caching --mamba-cache-mode align --kv-cache-dtype fp8 --engram-config {"cpu_offload": true} --enable-expert-parallel`, plus env-driven DCP 2 / a2a / max cudagraph 512 / prefix unit 800 / dense retention / KV offload 30 GiB (+cumem allocator) / mm limits. Harness (cell46): `CONCURRENCY = 70`, `MAX_GEN_TOKENS = 200_000`, `REASONING_EFFORT = "xhigh"`.

---

## 6. Tunable observations (with line evidence)

1. **QAR is built but off** (`QAR = None`). Comments measure prefill all-reduce as bandwidth-bound on host-bridged NCCL ("~4.9 GB/s", "3.4 ms at 10.5 MB", "~1.7 ms (5.2 MB)") and a2a int8/32 sends "~56 % of the BF16 bytes" with "~1.5-2x" logprob noise. Decode unaffected (`QAR_MIN_MB = 1.5`, graph capture excluded). Prefill-speed lever, opt-in.
2. **Prefill chunk forced to 1024** (`MAX_BATCHED_TOKENS = 1024` overrides step 6's 4096). It also sets the trim admission term (`LIMIT + chunk`) and all-reduce message size (5.2 MB ↔ 1024 tokens). Bigger chunk = fewer launches/longer messages; trade-off with memory.
3. **Compressed-key block factor at its max 8** (`assert COMPRESSED_BLOCK_FACTOR in (1, 2, 4, 8)`); the measured gain was 13.85x→16.73x. Larger factors are blocked by the assert, not by measurement shown here.
4. **GPU util 0.945**, assert allows up to <0.99.
5. **CUDA graph sizes go to 512** while cell43 `MAX_SEQS = 128` and harness `CONCURRENCY = 70` with `MTP_TOKENS = 0`: sizes 256/512 exceed the decode batch; capture memory/time for them may be unused (whether vLLM clips them is not visible in cell10).
6. **Sink-jump 49,152 / 65,536** (cut every 16,384). Lower LIMIT/KEEP → smaller per-request QSA KV (admission `sink + LIMIT + chunk`) and less scoring; constraint only `keep - sink >= 2052`, `limit > keep`. SKIP_GAP keeps scoring cost "at sink + recent part".
7. **RAM tier 30 GiB**; only constraint is `MemAvailable >= KV_OFFLOAD_GIB + 8`. Eviction look-ahead `FLASHNEXT_OFFLOAD_AHEAD` (default 4) and `FLASHNEXT_OFFLOAD_PRESSURE` (default 0.8, lazy OffloadingConnector path) are not set by cell10.
8. **OffloadingConnector edits (OFF_EDITS) are dormant** with `OFFLOAD_EVICT=True` because `config/vllm.py` switches to `SimpleCPUOffloadConnector`; they only matter with `OFFLOAD_EVICT=False`.
9. **DCP split-K constants** `4 / 32 / 8` are hard-coded class attributes "measured on L4"; `cp_kv_cache_interleave_size` (I) is left at vLLM's value.
10. **Custom all-reduce** explicitly rejected (5x slower than NCCL on this PCIe box) — do not spend time there.
11. **FP8 lm_head + HC mixers** and **FP16 SR DeltaNet state** are memory reducers already on; BF16 plain state is the alternative switch.
12. Diagnostics (`DIAG_ENCODER_MISS`, `DIAG_WEIGHTS`) are on; log-only, the encoder dict wrapper adds per-set Python string ops.
13. `PREFIX_MATCH_UNIT` effective 800 and dense retention are what make follow-up REPL calls hit inside the previous reply; changing them affects hit rate (cell10 tells the user to confirm no "Disabling fine-grained" log line).
