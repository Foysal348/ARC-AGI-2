# Domdolus notebook (public LB 83.47): cells 00-09 and 11-20, technical inventory

Date: 2026-11-04. Source: `D:/ARC AGI 2/experiments/Experiment 35 - Public-best rebuild (planned)/Build/domdolus_cells/cellNN.txt`.
Every cell in scope was read in full. Cell 15 holds four JSON-encoded source files; they were decoded (read-only, into the session scratchpad) and read in full:
`v4_edits.py` (431 lines), `qsa_v4.py` (105), `k4_edits.py` (280), `qsa_k4.py` (139). Cells 10 and 30 belong to other agents; where my cells depend on
names defined in cell 10, this is marked "(defined in cell 10)". I checked only these names in cell 10: `VROOT =` (line 41), `def fn_trim_sink_of` (line 61), `_FN_SEG =` (line 68), plus `KV_OFFLOAD_GIB` mentions.

General patch pattern (all patch cells): read a file under `/tmp/site-vllm/vllm` (or `VROOT`), assert that the anchor occurs exactly once, `str.replace`, append helper code at the end of the file,
`compile()` to check syntax, write the file. A marker comment such as `# flashnext ... (session 17)` makes each cell idempotent ("[already]"). Each feature is behind an
environment variable that the cell sets at its end; with the variable unset, the edited code takes the original path. The variables must be set before the server starts.

---

## Cell 00 (markdown)
`# ARC-AGI-2: Qwen 3.8 Flash-Next on 4 x L4 (Public LB 83.47)`

## Cell 01 (markdown)
`# Competition Switch and Timer`

## Cell 02: session id
Prints `SESSION_ID` parsed from `KAGGLE_CONTAINER_NAME` with `re.search(r"-(\d+)-", ...)`. Diagnostic only.

## Cell 03: competition switch, deadline timer, submission writer
Purpose: decide rerun vs. save, set the deadline, write `submission.json` (also mid-run), and kill everything when the deadline is reached.

Constants:
- `TIME_BUDGET_MIN = 11 * 60 + 55` (11 h 55 min after the kernel start)
- `VALIDATION_MODE = False`, `VALIDATION_TIME_MIN = 6 * 60 - 5` (5 h 55 min), `VALIDATION_TASKS = 0` (0 = all), `CHECK_ATTEMPTS = True`
- `IS_RERUN = os.getenv("KAGGLE_IS_COMPETITION_RERUN") is not None`
- `SUBMISSION = "/kaggle/working/submission.json"`, `WORK = "/tmp/arc_run"`

Mechanics:
- `_kernel_start()`: kernel start time = `btime` from `/proc/stat` + field 22 (starttime ticks) of `/proc/self/stat` / `SC_CLK_TCK`. `DEADLINE = T0 + _BUDGET_MIN * 60`. So the budget counts from the real process start, not from cell execution.
- Test file: finds exactly one `arc-agi_test_challenges.json` under `/kaggle/input`. Validation mode uses `arc-agi_evaluation_challenges.json` and requires `_solutions.json` beside it.
- `write_submission()`: under a lock, merges every `/tmp/arc_run/sub_*/submission.json` (last good parse kept in `_LAST_GOOD`) and, with `CHECK_ATTEMPTS`, the records in `out_*/*/*/check_attempts.json` and `parked/*/*/*/check_attempts.json` (with `metadata.json` two folders up giving `puzzle_id`).
- Variant ids: `_split_id` splits `<task>_<var>` with `var in ("P","T","F","2","3")`. `_BACK` maps answers back: `"T"` = transpose (`_TP`), `"F"` = horizontal flip (`_FL`), `""`, `"P"`, `"2"`, `"3"` = identity.
  `_ROUND_ORDER = {"": 0, "2": 1, "P": 1, "T": 2, "3": 2, "F": 3}` orders final answers by round.
- Candidate order per test input: final answers (in round order), then check() records sorted by `(r["ok"] / max(1, r["n"]), r["pct"])` descending. `attempt_1` = first candidate,
  `attempt_2` = first candidate different from attempt_1. Missing = `[[0]]`. `_grid_ok` validates 1..30 x 1..30, rectangular, ints 0..9 (bools rejected). Atomic write via `.tmp` + `os.replace`.
- `_deadline_watch` thread (`deadline-watch`): polls every 15 s; at the deadline sets `DEADLINE_HIT`, writes the submission, installs an IPython input transformer that turns every later cell into `pass`,
  then for 240 x 15 s runs `pkill -9 -f` on `STOP_EVERYTHING = ("arcagi2.evaluation.evaluate", "stage_manager.py", "vllm.entrypoints", "VLLM::", "/kaggle/working/prefetch.py")`, re-writing the submission once more at iteration 1.
- Not rerun and not validating: writes a placeholder submission and skips all later cells (the "Save Version" path).

Dependencies: competition data under `/kaggle/input`. Harness names that appear: `arcagi2.evaluation.evaluate`, `stage_manager.py`.

## Cell 04 (markdown)
`# Environment and Model`

## Cell 05: locate utility and model; background model prefetch + n-gram table to /dev/shm
Purpose: find the utility dataset (folder with `UTILITY_MANIFEST.json`) and the model (folder with `config.json` + `.safetensors`, outside the utility), then start a detached prefetch process.

Constants: `PREFETCH_READERS = 24`, `PREFETCH_LOG = "/kaggle/working/prefetch.log"`, `NGRAM_SHM = "/dev/shm/flashnext_ngram.bin"`, `PREFETCH_AFTER_STEP2 = False` (so prefetch starts in this cell, before the utility tars are unpacked).

Prefetch script (`/kaggle/working/prefetch.py`, argv = root, readers, log, shm file):
- Files sorted in vLLM's load order (`weight_utils._natural_sort_key` equivalent). Comment, session 11: reading "first pieces of every file first" made Linux drop starts of files vLLM still needed, causing network re-reads; so each file is read by many readers at once, in load order.
- Pieces `piece = 256 << 20` (256 MiB), `os.pread` in 16 MiB chunks, `ThreadPoolExecutor(24)`.
- n-gram table: parses the safetensors headers of `ple-*.safetensors`, collects tensors `ngram_embedding.shard_N.weight`, checks shards 0..N-1, same dtype, 1-D row dim, equal rows (last may be shorter).
  Writes the rows once into `/dev/shm/flashnext_ngram.bin` in global order (size `(total_rows + 65536) * row` bytes: "spare rows (zeros, never touched) for vLLM's per-GPU padding"), in 64 MiB jobs via `os.pwrite`,
  then `posix_fadvise(DONTNEED)` on the source range ("the rows are in /dev/shm now: do not keep the network copy"). Metadata in `<file>.json` (`torch_dtype`, `row_bytes`, `rows`, `shard_rows`, `shards`, `file_bytes`, `complete`).
  Skips the rebuild if a complete matching json exists. The `ple-*` files are then excluded from page-cache prefetch ("vLLM reads only the tiny non-table tensors there itself").
- Log lines `NGRAM SHM DONE: ...` and `PREFETCH DONE: ... GiB in ... s (... MB/s)`. Step 6 waits for it (stated: "step 6 waits for it").
- dtype table supports `F8_E4M3, F8_E5M2, U8, I8, BF16, F16, F32`.

Facts: comment "so the 51 GB are not held twice" (n-gram table size 51 GB).

## Cell 06: model kind, GPU/RAM checks, parallel config
Constants (`KINDS`):
- `"flashnext": dict(vllm="0.31.0", tp=4, replicas=1, kv_dtype="fp8", gpu_util=0.92, expert_parallel=True, min_gpus=4)` (selected: `MODEL_KIND = "flashnext"`)
- `"gsq27": dict(vllm="0.27.1", tp=2, replicas=2, kv_dtype="fp8", gpu_util=0.95, expert_parallel=False, min_gpus=2)` (alternative 27B path)
- `EXPERT_PARALLEL = KIND["expert_parallel"] and TP > 1`
Checks: Python must be 3.12 ("The utility is built for Python 3.12"); compute capability >= 8.9 unless `ALLOW_OTHER_GPU`; driver CUDA < 13.0 prints "torch 2.13 in the utility is built for CUDA 13".
RAM note: "Flash-Next keeps its 51 GB n-gram table in pinned host memory (split over the GPUs' processes): this needs about 70 GB of free RAM." (warns if `MemAvailable < 70`). Warns if `/dev/shm < 2e9`.
Note: `TP`, `REPLICAS`, `KV_DTYPE`, `GPU_UTIL` are set here; DCP2 and the a2a backend are not set in this cell (cell 10 or later).

## Cell 07: unpack utility, CUDA_HOME, driver stub, vLLM env
Purpose: materialize the vLLM and harness environments without pip, find nvcc, link libcuda, build the server environment.
- `MANIFEST` keys printed: `created, vllm, torch, humming, layout, pack, pivotools_commit`; asserts `MANIFEST["vllm"] == "0.31.0"`.
- Layout `"nopip-sites"`: `site-vllm.tar` -> `/tmp/site-vllm`, `site-arc.tar` -> `/tmp/site-arc`. `_warm()` first reads the tar into page cache with 16 readers, 64 MiB pieces (16 MiB preads), then `tar -xf`. `.ready` marker. Older layout: `pip install --no-index --find-links` from `requirements.lock` and `arc/requirements.arc.lock` with `--target`.
- chmod +x on `bin`, `nvidia/*/bin`, `nvidia/*/nvvm/bin`, `triton/backends/nvidia/bin`.
- `CUDA_HOME` = parent of `site-vllm/nvidia/*/bin/nvcc`; creates `lib64 -> lib` symlink and unversioned `.so` symlinks.
- `vllm_env()`: `PYTHONPATH = site-vllm/nvidia_cutlass_dsl/dsl_packages : site-vllm`; `CUDA_HOME`/`CUDA_PATH`; `PATH` adds CUDA bin + site bin; `LD_LIBRARY_PATH` adds CUDA lib + `torch/lib`;
  `HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, VLLM_NO_USAGE_STATS=1, DO_NOT_TRACK=1, TOKENIZERS_PARALLELISM=false, VLLM_ENGINE_READY_TIMEOUT_S=3600`; `LIBRARY_PATH` gets `/tmp/cuda-driver-stub`.
- Driver stub: `/tmp/cuda-driver-stub/libcuda.so -> real libcuda` (direct lookup or a torch probe of `/proc/self/maps`), test-link `gcc -lcuda` (for kernels that compile at first use).
- `STEP2_CHECK`: import check `torch, vllm, humming` deferred to step 6 (runs in the background at server start).
Dependencies: utility dataset (UTILITY_MANIFEST.json, `site-vllm.tar`, `site-arc.tar`), vLLM 0.31.0, torch 2.13 / CUDA 13 (from the warning text), `humming`, `nvidia_cutlass_dsl`, `pivotools` (commit recorded in the manifest).

## Cell 08: model folder check
`MTP_PRESENT = False`. Requires `config.json`, `model.safetensors.index.json`, `tokenizer.json` (`patch_vllm_qwen35_embedding.py` only for gsq27). Prints file count and GB, `architectures`, `quant_method`.
Asserts `Qwen4Exp` in `architectures`; prints `ple_embedding_dtype` (n-gram table dtype). No patch.

## Cell 09 (markdown)
`# vLLM Patches: Parallel Layout and KV-Cache Capacity`

---

## Cell 11: KV layout diagnostics + attention group size override
Target: `vllm/v1/core/kv_cache_utils.py`.
1. Diag (marker `# flashnext KV layout diag (session 17)`). Anchor: `vllm_config.cache_config.kv_cache_max_concurrency = max_concurrency`. Adds, after it, log lines per KV group:
   `FLASHNEXT KVDIAG group %d | spec type | layers | block tokens | page B | real_page_size_bytes | pages per task at max_len | mamba_cache_mode | prefill ckpt blocks | spec blocks | tp_replicated | host_resident | shapes | dtypes`
   and `FLASHNEXT KVDIAG pool | num_blocks | tokens | max_len | groups`. Pure logging (the place to read pool size from server logs).
   Note: the attributes `real_page_size_bytes`, `num_prefill_checkpoint_blocks`, `tp_replicated`, `host_resident` are read with `getattr`; they come from this fork of vLLM / cell 10.
2. Group-size override (marker `# flashnext group-size override (session 17)`). Anchor: `def num_groups_for(spec: UniformTypeKVCacheSpecs, balanced: bool) -> int:` + `if balanced and repeats_per_group is not None:`.
   With `FLASHNEXT_KV_GROUP_LAYERS` set, `repeats_per_group = int(env)` (layers per attention KV group), but the balanced split is not applied to `MambaSpec` / `CircularBufferSpec` groups.
   Flag value set: `FLASHNEXT_KV_GROUP_LAYERS = "4"`.

## Cell 12: hold-states switch (DeltaNet state retention modes)
Targets: `v1/core/single_type_kv_cache_manager.py`, `v1/core/block_pool.py`, `v1/core/kv_cache_coordinator.py`. The anchors in `single_type_kv_cache_manager.py`
(`_fn_h`, `held.append((idx, block, aligned))`, `pool._fn_dropping`, `_FN_SEG["states_dropped"]`) and in the coordinator (`return [e[3] for e in entries]`) are code that cell 10 installed (error text: "patch cell changed?").
Context (from the edited code): during a call, retired DeltaNet (Mamba-type) state blocks at block boundaries are kept in a list `held` of `(idx, block, aligned)`; the original rule keeps the last 2 aligned / 1 unaligned and frees older ones; at the end of the call held blocks are put back into null slots of the request's block list.

Edits (marker `# flashnext hold-states switch (session 17)`), controlled by `FLASHNEXT_HOLD_STATES` = `pin | pinnew | cached | protect | park` (default `pin`):
- "take back at the end of the call": for entries with a 4th field (block hash), the block was freed at retire time; it is reclaimed only if its hash still matches (`ref_cnt == 0` -> remove from free queue, `ref_cnt += 1`); counters `lost`, `back`.
- "retire: cached instead of pinned": in non-pin modes, the held entry stores the hash, older states are evicted from the prefix cache only when still unused (`_maybe_evict_cached_block`).
- "retire: free the kept state at once in cached mode": the new state is freed immediately (`pool.free_blocks([block])`) so it is a normal cached block that can be reclaimed under pressure; in `protect` it is marked `prot[block_id] = inf`.
- "retire: pinnew keeps only the newest state pinned": all older held states are freed to the cache (normal cached blocks), only the newest stays pinned.
- `block_pool.py` (marker `# flashnext hold-states second chance (session 17)`), anchor `ret: list[KVCacheBlock] = self.free_block_queue.popleft_n(num_blocks)` + `if self._reuse_watchers:`:
  `_fn_second_chance` moves protected blocks taken from the free-queue head back to the tail, scanning up to `_FN_SCAN = 256` extra blocks, then forces the oldest protected ones (`forced` counter).
- `kv_cache_coordinator.py` (marker `# flashnext hold-states park (session 17)`): `park` mode protects a finished call's hashed blocks for `FLASHNEXT_PARK_S` seconds (default `"60"`), only for the GPU pool (pool whose `_fn_on_evict` is `_fn_gpu_evicted`); prunes expired entries when `len(prot) > 4 * pool.num_gpu_blocks`.
Flag value set: `FLASHNEXT_HOLD_STATES = "pinnew"`.
Effect (from code): fewer GPU blocks pinned by DeltaNet states per running request (only the newest), older ones become evictable cache.

## Cell 13: batch-sharded sampling
Targets: `models/qwen4_exp/nvidia/model.py` and `engine/arg_utils.py` (marker `# flashnext batch-sharded sampling (session 17)`).
- Adds `Qwen4ExpForCausalLM.compute_logits_local(hidden_states)` = `self.logits_processor(self.lm_head, hidden_states, skip_gather=True)` (logits stay vocab-sharded per TP rank; no all-gather of full-vocab logits).
- `arg_utils.py`, anchor `engine_args.enable_cumem_allocator = True` + `return engine_args`: with `FLASHNEXT_BATCH_SHARDED_SAMPLING == "1"` sets `engine_args.enable_batch_sharded_sampling = True` ("--enable-batch-sharded-sampling").
  The engine arg itself and the sampler that uses `compute_logits_local` are not defined in this cell (they exist in the utility's vLLM tree or cell 10).
Flag value set: `FLASHNEXT_BATCH_SHARDED_SAMPLING = "1"`.

## Cell 14: token embedding in host RAM (UVA)
Target: `models/qwen4_exp/nvidia/model.py`, anchor `loaded = loader.load_weights(weights, mapper=mapper,)` + `return loaded` (marker `# flashnext token embedding in host RAM (session 17)`).
After weight load, `_flashnext_embed_to_host(model)` (if `FLASHNEXT_EMBED_HOST == "1"`): requires `is_uva_available()`; copies `model.embed_tokens.weight` into a pinned CPU tensor, replaces `emb.weight.data`
with `get_accelerator_view_from_cpu_tensor(host)` (GPU-addressable view of host memory over PCIe), verifies the first 4096 rows equal, `torch.cuda.empty_cache()`; prints MiB freed per GPU.
Why: frees the embedding table from each GPU (the embedding lookup reads only the rows of the current tokens through UVA). The MiB value is printed at runtime; not stated in the cell.
Flag value set: `FLASHNEXT_EMBED_HOST = "1"`.

## Cell 15: QSA attention KV in 4 bits (V4, then K4)
Code lines: decode `V4_EDITS` and `V4_MODULE` (JSON strings), `exec(..., {'VROOT': VROOT, 'V4_MODULE_SRC': V4_MODULE})`, set `FLASHNEXT_QSA_V4='1'`;
`assert FLASHNEXT_PREFIX_MATCH_UNIT == '800'` ("expected the patch cell prefix match unit 800") -> set `'1024'`; `assert FLASHNEXT_MAX_BATCHED == '1024'` ("expected the patch cell prefill chunk 1024") -> set `'2048'`;
then `FLASHNEXT_K4_UNDO='0'`, exec `K4_EDITS` with `K4_MODULE`, set `FLASHNEXT_QSA_K4='1'`, `FLASHNEXT_PREFIX_MATCH_UNIT='1504'`. Final print: `QSA KV: K4 (keys + values 4-bit) | prefix match unit 1504 | prefill chunk 2048`.
`VROOT` is defined in cell 10 (line 41).

### 15a. V4 (values 4-bit), marker `# flashnext V4 (session 17)`; scope "the 12 QSA attention layers"
New module `models/qwen4_exp/nvidia/ops/qsa_v4.py`:
- Row layout per (token, KV head), `V4_ROW_BYTES = round_up(D + D/2 + 8, 16)`; D = 256 -> 400 B (FP8: 2*D = 512): `[0,D)` K FP8 e4m3 (scaled by 1/k_scale like `reshape_and_cache_flash`), `[D, D+D/2)` V 4-bit (byte i = dim 2i low nibble | dim 2i+1 high nibble), then V scale f32, V zero (row min) f32, zero padding.
  "The row lives in vLLM's packed-slot page ([blocks, heads, tokens, V4_ROW_BYTES] uint8, AttentionSpec.state_content_bytes), so every block-level mechanism (prefix cache, trim, RAM offload, DCP split) moves K, V and the scales together."
- Kernel `_v4_store_kernel` (Triton, grid `(tokens, heads)`, `num_warps=2`): K -> FP8 with clamp +-448; V asymmetric per-row 4-bit: `scale = (vmax - vmin) / 15`, `code = clamp(floor((v - vmin)/scale + 0.5), 0, 15)`, pack two per byte, store scale and min. Skips slots < 0 or >= num slots.
- `v4_views`: K FP8 view and packed V uint8 view over the same bytes, `[block, token, head, dim]`.
- Why faster/bigger: 400 vs 512 B per token per head (-21.9 %) -> more tokens per page; the decode kernel reads fewer bytes.

Edits by `v4_edits.py`:
- `platforms/interface.py` (hybrid block-size alignment), anchor `attn_page_size_1_token = backend_cls.customize_spec(attn_spec).page_size_bytes`: with V4, `attn_page_size_1_token = num_kv_heads * V4 row`. Comment: "attention blocks of 1,024 tokens at the same page size (409,600 B on Kaggle), the DeltaNet pages stay padded as today".
- `models/qwen4_exp/common/qsa_cache.py` (indexer compressed-key cache `MLAAttentionSpec`): block = attention base block x factor, factor = `(page // unit) // dcp * dcp`, `alignment=page` (pad to page). Second edit caps it:
  `_fn_factor = max(_fn_dcp, min(_fn_factor, 6400 // _fn_base // _fn_dcp * _fn_dcp))` with the evidence comment: "RTX V4 run 1 with page-filling 24,576-token blocks recomputed 7-14K prompt tokens/s from minute ~29 (prefix hits fell, decode 2,200 -> 380 tok/s)". Header comment: "Kaggle: 6 x 1,024 = 6,144 tokens".
- `models/qwen4_exp/nvidia/qsa.py`: impl flag `self._fn_v4` (requires `--kv-cache-dtype fp8`); views via `v4_views`; `v4=` passed to the single-GPU call and the DCP call; new `do_kv_cache_update` override -> `v4_store(key, value, kv_cache, slot_mapping, layer._k_scale_float)`; owner KV spec gets `state_content_bytes=_fn_v4_spec_bytes(...)`.
- `models/qwen4_exp/nvidia/ops/qsa.py` (split-K sparse GQA kernel over paged K/V): constexpr `V4`; V decode in-kernel: load D/2 packed bytes, load scale/zero, `tl.join(packed & 0xF, packed >> 4)`, `codes * scale + zero` -> fp16. Wrapper accepts uint8 V of half width; `output_scale = 1.0` for V4 (per-row scales). Tile cap: `if v4 and BLOCK_N > 64: BLOCK_N = 64` — "128-token tiles need 106-128 KB shared memory with 4-bit K/V (RTX/L4 limit 99 KB)". Warmup (`TritonWarmupTensor`) edited to V4 shapes.
- `models/qwen4_exp/nvidia/ops/qsa_dcp.py` (DCP local partial): `v4` arg, `output_scale = 1.0`, `V4=v4` at launch.
- Undo: `FLASHNEXT_V4_UNDO=1`. An older install is reversed and re-installed.

### 15b. K4 (keys 4-bit too, Hadamard-rotated), marker `# flashnext K4 (session 17)`
New module `ops/qsa_k4.py`:
- Row `K4_ROW_BYTES = round_up(D/2 + D/2 + 16, 16)`; D = 256 -> 272 B (V4 400, FP8 512). Layout: `[0,D/2)` rotated K codes, `[D/2,D)` V codes, `[D,D+8)` V scale/zero f32, `[D+8,D+16)` K scale/zero f32.
- `hadamard(D)`: orthonormal Sylvester Hadamard, f32, cached per (size, device). "(qH).(kH) = q.k, so the scores are unchanged in exact arithmetic, while the rotation spreads outlier channels over all dimensions so a per-row 4-bit grid loses less. Values are NOT rotated."
- `k4_store`: `k_rot = (key[:n].float() @ hadamard(D)).contiguous()` (torch f32 GEMM), then `_k4_store_kernel` (grid `(n, heads)`, `num_warps=2`) quantizes K and V per row with `_quant_row` (same asymmetric 4-bit as V4) and stores 4 floats. FP8 `k_scale` not used.
- `k4_rotate(query)` = `(x.float() @ H).to(x.dtype)` applied to the query in the impl before attention.
- `k4_check_enabled()`: K4 without V4 raises.
Edits by `k4_edits.py` (on top of V4 lines): `interface.py` and `qsa_cache.py` use the K4 row for the page; `qsa.py` spec bytes = K4 row, `self._fn_k4`, `k4_views` + `query = k4_rotate(query)`, no FP8 key view, dtype check relaxed, `k4=` to both calls, cache write via `k4_store`;
`ops/qsa.py` constexpr `K4`, in-kernel K decode (`k_meta = k_row + HEAD_DIM + 8`, decode `[BLOCK_N, D]` then `tl.trans`), head check `q.shape[2] == k_cache.shape[3] * 2`, `is_fp8 = ... or k4`, `softmax_scale = head_dim**-0.5` ("absolute per-row key scales (no FP8 k_scale)"), warmup; `ops/qsa_dcp.py` `k4` arg, `softmax_scale` without k_scale, `K4=k4`.
Why: 272 vs 512 B per token per head (-46.9 %): about 1.88x tokens per page vs FP8, and fewer bytes read per decode step. Since K4 requires V4, the `BLOCK_N <= 64` cap stays active.

## Cell 16: DeltaNet state to RAM ("state RAM v2", session 18b)
Marker `# flashnext state RAM v2 (session 18b)`; refuses a tree with the first version (`# flashnext state RAM (session 18b)`, "restore too late, session 18b bug").
Requires: `FLASHNEXT_HOLD_STATES=pinnew` (asserted), and the RAM offload with `FLASHNEXT_OFFLOAD_EVICT=1` (`KV_OFFLOAD_GIB` set; from cell 10), otherwise `RuntimeError`.
Targets and edits:
- `v1/core/single_type_kv_cache_manager.py`: in `pinnew` with `_fn_sr` present, retire goes to `_fn_sr_retire`: free older held states, queue one GPU -> RAM copy of the newest state (`_fn_sr_store`), record it with its hash, and free it as a normal cached block (taken back at call end if still on GPU).
  Lookups (`get_cached_block` for fine-grained and normal DeltaNet hits) go through `_fn_sr_cached`: GPU hit as before; GPU miss -> `_fn_sr_restore` from RAM (only for aligned boundaries: `block_size == alignment_tokens or (i + 1) * block_size % alignment_tokens == 0`).
- `v1/simple_kv_offload/manager.py` (connector, scheduler side): `_fn_sr_init` (flags `FLASHNEXT_STATE_RAM`, `FLASHNEXT_STATE_RAM_FORCE` test-only, `FLASHNEXT_STATE_RAM_CHECK` test-only, `FLASHNEXT_STATE_RAM_RESERVE` default `"64"` blocks).
  `_fn_sr_store`: one RAM copy per (request, group); drops the request's previous RAM copy if unused; skips if already in RAM or RAM pool full; queues `TransferMeta([gpu], [cpu])` in `_pending_finished_stores`. `_fn_sr_last` capped at 200,000 entries (drops 100,000).
  `_fn_sr_restore`: needs `free >= groups + reserve`; takes new GPU blocks, `_insert_block_hash` under the same hash, queues RAM -> GPU loads. `_fn_sr_take` passes block ids to the worker in the connector metadata and sets `has_sync_kv_loads = True`.
  Also: a state copy whose GPU block lost its hash is skipped instead of asserting. Report line every 60 s: `flashnext state RAM (last 60 s | total): stored, already, ram_full, no_hash, ram_dropped, restored, no_room, forced; hold freed / taken back / not on the GPU at the call end`.
- `v1/simple_kv_offload/metadata.py`: fields `fn_sr_gpu_blocks`, `fn_sr_cpu_blocks`, `fn_sr_check`.
- `v1/simple_kv_offload/worker.py`: restore copy in `handle_preemptions` (the first connector call of the step), `g[d].copy_(c[s], non_blocking=True)` per layer cache per block on the compute stream, then `torch.cuda.synchronize()` ("steps with a restore are rare"); guard that raises if restores reach `start_load_kv` unseen. Disk mode unsupported.
- `v1/worker/gpu/model_runner.py` (V2 runner): before the Mamba "align" pre-copy (`preprocess_state`), calls `get_kv_transfer_group().handle_preemptions(md)` when restores exist, so the resumed DeltaNet state is in place before it is copied into the running block.
Flag value set: `FLASHNEXT_STATE_RAM = "1"`.
Effect: the newest DeltaNet state of each running task is not pinned on GPU; it lives in RAM and returns on a prefix hit -> more free GPU blocks.

## Cell 17: indexer keys FP8 + two layers per compressed-key row ("ckpack", session 18b)
Target: `models/qwen4_exp/nvidia/indexer_qsa.py` (marker `# flashnext ckpack (session 18b)`), flag `FLASHNEXT_QSA_CKPACK == "1"`:
- "indexer keys FP8": after `self.indexer_kv_dtype = vllm_config.attention_config.resolve_indexer_kv_dtype("bf16")`, sets `self.indexer_kv_dtype = "fp8"` (half the bytes of BF16 compressed keys).
- "two layers per compressed-key row": `_fn_ckp_make` — the first indexer layer creates a `QSACompressedKeyCache` with `head_size = 2 * index_head_dim` (owner); the next layer gets a `_FnCkPartner` with no cache of its own, whose `kv_cache` is `owner.kv_cache[..., w:2w]`. Requires FP8 (`torch.float8_e4m3fn`). Prints `pairs so far N`.
- "forward uses this layer's half of the row": `compressed_key_cache = _fn_ckp_view(self.compressed_key_cache)` (owner -> `[..., :w]`).
Why (from code): one cache spec and page for two layers instead of two; with page padding of compressed-key blocks (cell 15: "padded to the page (vLLM MLA alignment)") this halves the padded pages. Flag value set: `FLASHNEXT_QSA_CKPACK = "1"`.

## Cell 18 (markdown)
`# vLLM Patches: Recompute Prevention`

## Cell 19: exact resume (session 18b)
Targets: `v1/core/kv_cache_manager.py`, `v1/core/single_type_kv_cache_manager.py` (marker `# flashnext exact resume (session 18b)`), flag `FLASHNEXT_EXACT_RESUME == "1"`.
Purpose: when a call finishes with STOP, park all its blocks (including the non-hashed partial tail block, the DeltaNet state at position L, and the circular-buffer ring block) so that a follow-up prompt that starts with exactly the same L tokens resumes with zero recompute (normal prefix caching only hits full hashed blocks and aligned states).
- `kv_cache_manager.py`: after `find_longest_cache_hit`, `_fn_er_adopt` may replace the result with the parked entry: `(blocks per group, L, 0)`; after `allocate_new_blocks`, `_fn_er_done` releases the pins; in `free()` and `pop_blocks_for_free()`, `_fn_er_park` runs first.
- `single_type_kv_cache_manager.py`: `CircularBufferManager.add_local_computed_blocks` hands the parked ring block (`KpoolTailManager` is the next class) to the request if pending.
- Park rules: only `FINISHED_STOPPED` (test flag `FLASHNEXT_EXACT_RESUME_TEST_ANY_FINISH` also length-capped); `L = num_computed_tokens >= 1024`; key = `hash(tuple(ids[:sink + 200]))` with `sink = fn_trim_sink_of(ids)` (defined in cell 10). Per group: ring -> pin `blocks[0]`; Mamba -> pin `blocks[(L-1)//block_size]`; attention -> first `cdiv(L, block_size)` blocks, hashed ones (except the last) only checked by hash, others pinned (`ref_cnt += 1`).
- Adopt rules: entry L > current hit and L <= max; `ids[:L]` must equal the stored tokens; any hashed block whose hash changed drops the entry.
- Housekeeping: `FLASHNEXT_EXACT_RESUME_TTL` default `"600"` s, `cap 512` entries, `FLASHNEXT_EXACT_RESUME_MIN_FREE` default `"64"` blocks (entries dropped under pressure). Report line `FLASHNEXT EXACT RESUME (last 60 s | total): parked, parked blocks pinned, adopted, tokens gained, ...`.
Flag value set: `FLASHNEXT_EXACT_RESUME = "1"`.

## Cell 20: token splice (session 18b)
Targets: `v1/engine/output_processor.py` and `v1/engine/async_llm.py` (marker `# flashnext token splice (session 18b)`), flag `FLASHNEXT_TOKEN_SPLICE == "1"`.
Purpose: a follow-up prompt re-tokenizes the previous answer text, which can give different token ids than the model generated, breaking the prefix-cache hit. The splice replaces the prefix of the new prompt with the model's own token ids where the bytes are identical.
- `output_processor.py`: on finish with `FinishReason.STOP`, `_fn_tsp_record` stores `prompt ids + output ids` (as `array('i')`) under key = hash of everything before the first `<|im_start|>assistant`; keeps the last 3 non-prefix sequences + the new one per key; at most 600 keys (LRU).
- `async_llm.py`: in `add_request` after `assign_request_id`, `_fn_tsp_apply`: builds a byte table for every token id (GPT-2 byte-level alphabet; added tokens as UTF-8), finds the first differing position d between stored S and new P, checks that the bytes of `S[d:L]` (L = len(S) - 1) are a byte prefix of P's bytes; a straddling token is re-encoded and verified; new prompt = `S[:L] + rest_ids + P[j:]`. Picks the candidate with the longest reuse.
  Moves multimodal offsets (checks images inside the spliced span); lowers `max_tokens` if `len(new) + max_tokens > max_model_len`. Report line every 60 s with `recorded, spliced, tokens re-used, already exact, no key, no candidate, bytes differ, straddle ...`.
Flag value set: `FLASHNEXT_TOKEN_SPLICE = "1"`.

---

## Flag summary (values set by cells 11-20)
| Variable | Value | Cell |
|---|---|---|
| FLASHNEXT_KV_GROUP_LAYERS | 4 | 11 |
| FLASHNEXT_HOLD_STATES | pinnew | 12 |
| FLASHNEXT_PARK_S | default 60 (only used in park mode) | 12 |
| FLASHNEXT_BATCH_SHARDED_SAMPLING | 1 | 13 |
| FLASHNEXT_EMBED_HOST | 1 | 14 |
| FLASHNEXT_QSA_V4 | 1 | 15 |
| FLASHNEXT_PREFIX_MATCH_UNIT | 800 (cell 10) -> 1024 -> 1504 | 15 |
| FLASHNEXT_MAX_BATCHED | 1024 (cell 10) -> 2048 | 15 |
| FLASHNEXT_K4_UNDO | 0 | 15 |
| FLASHNEXT_QSA_K4 | 1 | 15 |
| FLASHNEXT_STATE_RAM | 1 (RESERVE default 64) | 16 |
| FLASHNEXT_QSA_CKPACK | 1 | 17 |
| FLASHNEXT_EXACT_RESUME | 1 (TTL 600, MIN_FREE 64, cap 512) | 19 |
| FLASHNEXT_TOKEN_SPLICE | 1 | 20 |

Engine-level constants from cell 06: vLLM 0.31.0, TP 4, 1 replica, `kv_dtype="fp8"`, `gpu_util=0.92`, expert parallel on.

## Memory / capacity facts
- n-gram (PLE) table: 51 GB; kept in pinned host memory split over the GPU processes; "needs about 70 GB of free RAM" (cell 06). Built once in `/dev/shm/flashnext_ngram.bin` with `+65536` spare rows (cell 05).
- Attention (QSA) bytes per token per KV head (D = 256): FP8 512 B, V4 400 B, K4 272 B (`qsa_v4.py` / `qsa_k4.py` header lines). 12 QSA attention layers (v4_edits header).
- Page size on Kaggle: "attention blocks of 1,024 tokens at the same page size (409,600 B on Kaggle)" (v4_edits header). 409,600 / 400 = 1,024 tokens -> this implies 1 KV head per rank in the attention page.
  My arithmetic, consistent with the cell's prefix-unit values: 409,600 / 512 = 800 (the cell 10 value `FLASHNEXT_PREFIX_MATCH_UNIT=800`), / 400 = 1,024 (V4 value), / 272 = 1,505.9 -> 1,504 (K4 value, multiple of 16). So the prefix match unit equals the attention block size in tokens at a fixed ~409.6 kB page.
- Compressed indexer-key blocks: "Kaggle: 6 x 1,024 = 6,144 tokens" (V4), capped by `6400 // base // dcp * dcp`. With base 1,504 and DCP 2 this formula gives 4 x 1,504 = 6,016 tokens (my arithmetic from the line, assuming base = 1,504).
- Indexer keys: FP8 instead of BF16 and two layers per row (cell 17).
- Embedding table moved off each GPU to pinned host RAM (cell 14; MiB printed at runtime, not in the cell).
- DeltaNet states: only the newest per running request is held (pinnew), and with state RAM even that one goes to RAM (cells 12, 16). RAM restore needs `groups + 64` free GPU blocks.
- Pool size itself: printed at runtime by the KVDIAG lines (`num_blocks`, `tokens`, `max_len`, per-group page bytes) from cell 11; no number in these cells.
- Budget: run deadline 11 h 55 min from kernel start; validation 5 h 55 min (cell 03).

## Tunable for more speed or bigger pool (observations with line evidence)
1. `gpu_util=0.92` (cell 06, line 9) is the vLLM memory fraction; the 27B kind uses `0.95` (line 10). Any change needs an OOM check; the cells give no measured headroom.
2. QSA split-K tile cap: `if v4 and BLOCK_N > 64: BLOCK_N = 64` because "128-token tiles need 106-128 KB shared memory with 4-bit K/V (RTX/L4 limit 99 KB)" (v4_edits, "config: tile cap"). The cap is keyed on `v4`, so it also applies under K4.
3. K4 rotation runs outside the store kernel as a torch f32 GEMM: `k_rot = (key[:n].float() @ hadamard(head_dim, key.device)).contiguous()` (qsa_k4.py line 108) and per forward `query = k4_rotate(query)` = `(x.float() @ H).to(x.dtype)` (line 46). These are unfused f32 matmuls + casts per QSA layer per step; a fused rotate-and-quantize (or a fast Walsh-Hadamard transform) is the visible fusion target.
4. Store kernels use `num_warps=2`, grid `(tokens, heads)` (qsa_v4.py line 87, qsa_k4.py line 114).
5. Do not enlarge compressed-key blocks: the cap to ~6,400 tokens is backed by "page-filling 24,576-token blocks recomputed 7-14K prompt tokens/s ... decode 2,200 -> 380 tok/s" (v4_edits, "compressed-key blocks stay ~6,400 tokens").
6. Prefill chunk `FLASHNEXT_MAX_BATCHED` raised 1024 -> 2048 together with V4/K4 (cell 15, lines 6-8); it is a single env value.
7. `FLASHNEXT_KV_GROUP_LAYERS = "4"` (cell 11 line 48) sets layers per attention KV group; changing it changes page grouping (the KVDIAG lines show the result).
8. State RAM restore does a full `torch.cuda.synchronize()` and a Python loop of per-layer, per-block `copy_` calls (cell 16, worker edits: `for name, g in self.gpu_kv_caches.items(): ... g[d].copy_(c[s], non_blocking=True)` then `_fn_t.cuda.synchronize()`); the comment says restores are rare, so this matters only if the restore count in the 60 s report is high.
9. Retention knobs with defaults: `FLASHNEXT_STATE_RAM_RESERVE` 64, `FLASHNEXT_EXACT_RESUME_TTL` 600 s, `FLASHNEXT_EXACT_RESUME_MIN_FREE` 64, exact-resume `cap 512`, token-splice 600 keys x 4 sequences, `_FN_SCAN = 256`, `FLASHNEXT_PARK_S` 60. The 60 s report lines (`FLASHNEXT EXACT RESUME`, `flashnext state RAM`, `FLASHNEXT TOKEN SPLICE`) give the hit/miss counters needed to tune them.
10. Exact resume parks only `FINISHED_STOPPED` calls with `L >= 1024`; length-capped calls are parked only with the test flag `FLASHNEXT_EXACT_RESUME_TEST_ANY_FINISH` (cell 19). Token splice records only `FinishReason.STOP` (cell 20).
11. Startup: `PREFETCH_READERS = 24` and `PREFETCH_AFTER_STEP2 = False` (cell 05); tar warm-up with 16 readers (cell 07 `_warm`). The comment on read order (session 11) is the evidence that load-order prefetch avoids network re-reads.
12. `VLLM_ENGINE_READY_TIMEOUT_S = 3600` (cell 07) — startup may take up to an hour without failing.

## Dependencies (all cells in scope)
- Kaggle inputs: ARC Prize 2026 competition data (`arc-agi_test_challenges.json`; evaluation challenges + solutions for validation); the utility dataset (`UTILITY_MANIFEST.json`, `site-vllm.tar`, `site-arc.tar`, layout `nopip-sites`); one model folder (`config.json` with `Qwen4Exp` architecture, `model.safetensors.index.json`, `tokenizer.json`, `ple-*.safetensors` with `ngram_embedding.shard_N.weight`).
- Software: Python 3.12, vLLM 0.31.0 (with the `qwen4_exp` model and `simple_kv_offload` modules), torch 2.13 / CUDA 13 (warning text), `humming`, `nvidia_cutlass_dsl`, Triton (`vllm.triton_utils`), `transformers` AutoTokenizer (token splice), harness `arcagi2.evaluation.evaluate` / `stage_manager.py` in `site-arc`.
- Hardware: compute capability >= 8.9, at least 4 GPUs, about 70 GB free RAM.
- Cross-cell: `VROOT`, `fn_trim_sink_of`, `_FN_SEG`, `_fn_h`/`held` logic, RAM offload (`KV_OFFLOAD_GIB`, `FLASHNEXT_OFFLOAD_EVICT`), `FLASHNEXT_PREFIX_MATCH_UNIT=800`, `FLASHNEXT_MAX_BATCHED=1024` all come from cell 10.
