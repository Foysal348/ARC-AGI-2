# Domdolus notebook (public LB 83.47): cells 21-29 and 31-46, full inventory

Date: 2026-11-04 (BDT). Source: `D:/ARC AGI 2/experiments/Experiment 35 - Public-best rebuild (planned)/Build/domdolus_cells/cellNN.txt`.
I read every line of cells 21-29 and 31-46. cell46 (716 lines) and cell25 (368 lines) were read in chunks. cell30 is covered by another agent.
Read-only work. No GPU or Kaggle runs.

Notes on conventions:
- All vLLM patches are **anchor-replace**: `assert _s.count(_old) == 1`, then `replace`, then `compile()`, then write. If `_new` is already in the file, the edit is skipped (`[already]`). The vLLM root is `VROOT` (cell10) or `/tmp/site-vllm/vllm`.
- Every switch is an env var read **at import time** in the vLLM worker. It must be set before the server starts. Unset means stock behaviour.
- Most patches carry the marker comment `(session 22)`.

---

## Cell 21: markdown
`# vLLM Patches: Decode Kernels` (section header only).

## Cell 22: top-k candidate exchange in sharded sampling
**Purpose.** With batch-sharded sampling (`FLASHNEXT_BATCH_SHARDED_SAMPLING=1`, cell13), each TP rank normally sends the full vocab-shard logits to the rank that owns each request (`all_to_all_logits`). This patch sends only each rank's top-K `(value, token id)` pairs. The sampler then runs on a compact `[n, tp*K]` logits tensor.

**Target.** `v1/worker/gpu/model_runner.py`. Marker `# flashnext top-k exchange (session 22)`.

**Edits:**
1. **helpers**, inserted before `class GPUModelRunner(LoRAModelRunnerMixin):`:
   - `_FN_TOPK = os.environ.get("FLASHNEXT_TOPK_EXCHANGE") == "1"`
   - `_FN_TOPK_K = int(os.environ.get("FLASHNEXT_TOPK_EXCHANGE_K", "64"))`
   - `_fn_topk_reason(runner, batch, grammar_output)`: returns None (the fast path is allowed) or the reason for a full exchange. It falls back for any of: no sampler, grammar, speculative tokens (`batch.num_draft_tokens`), logit rows != requests, logprobs, sampling mask / trace / nans, `top_k outside 1..K` (it requires `temp == 0` or `1 <= top_k <= K` for every request), logit bias, penalties, bad words, any custom logits processor, thinking budget.
   - `_fn_topk_count`: fast/full counters. Every 60 s rank 0 prints `flashnext top-k exchange: fast N | full M {reasons}`.
   - `_fn_topk_all_to_all(logits_shard, metadata, vocab_size, tp_rank)`: `torch.topk(logits_shard.float(), k)`. Global ids = `ids + tp_rank * shard_w`. Values and ids are packed as float32 `[n, 2k]` (the ids are carried as float32 and later `.round()`ed back to int64). `torch.distributed.all_to_all_single` over the TP device group with `output_split_sizes=[num_local_logits]*tp_size` and `input_split_sizes=metadata.num_logits_per_rank`. Then a reshape to `[n_local, tp*k, 2]`. Ids `>= vocab_size` (padding) are masked to `-inf`.
2. **candidate ids variable**: `_fn_ids = None` after `global_input_batch = input_batch`.
3. **exchange**: replaces `logits = all_to_all_logits(local_logits, shard_metadata); logits = logits[:, :self.vocab_size]` with the conditional fast path.
4. **map drawn positions to token ids**: after `sampler_output = self.sampler(logits, input_batch)`, the sampled position is mapped back to the real token: `sampler_output.sampled_token_ids = torch.gather(_fn_ids, 1, sampled.view(-1,1))`.

**Flag.** `FLASHNEXT_TOPK_EXCHANGE=1` (needs `FLASHNEXT_BATCH_SHARDED_SAMPLING=1`). K comes from `FLASHNEXT_TOPK_EXCHANGE_K` (default 64).
**Why faster.** It cuts the all-to-all payload from vocab/tp floats per request to 2*K floats, and the sampler works on tp*K = 256 columns instead of the full vocabulary.
**Caveat (from the code).** It is exact only when top_k <= K (or greedy). Any request with top_k > 64, penalties, logprobs, or MTP draft tokens disables it for the whole step.

## Cell 23: HC-down split-K kernel (`fn_hc_down.py`) + Humming linear patch
**Purpose.** This replaces Humming's stream-K GEMM for the hyper-connection "down + inject" projection, which is FP8 weight 336 x 10240. The kernel docstring explains why: Humming uses "only 6 output tiles: 27 of 58 SMs, ~33 us at M 60 on an L4 (~120 GB/s ...): the long K loop per tile is the limit".

**New module.** `models/qwen4_exp/nvidia/ops/fn_hc_down.py`, written when missing or different.
- `_fn_hcd_partial` (Triton, grid `(cdiv(N,BN), SPLITS, cdiv(M,BM))`): each program computes one output tile over one K slice (`K_PER_SPLIT`). The FP8 weight is loaded and converted to bf16, then `tl.dot` with fp32 accumulation. Scales are either per row (`ROW_SCALE`, applied once after the loop) or per (row, K-block). It stores fp32 partials to a workspace `part[pid_k, M, N]`. `do_not_specialize=["M"]` is set, and "K_PER_SPLIT stays specialized: its alignment lets Triton vectorize the loads".
- `_fn_hcd_reduce`: sums the SPLITS partials in a fixed order ("deterministic") and writes bf16. BLOCK 1024, 4 warps.
- `_workspace`: a cached fp32 buffer per device, at least `1<<20` elements, grown on demand.
- `plan(M)` returns `(BM, BN, BK, SPLITS, warps)`:
  - M<=64: BM 16/32/64 (for M<=16 / <=32 / else), BN 32, BK 128, splits 5, warps 8
  - M<=128: 64, 32, 128, 5, 8
  - M<=256: 64, 128, 128, 4, 4
  - else: 64, 128, 64, 2, 4
  - Measured numbers in the docstring: "M 60 23.5 vs 28.7 us (weight in L2 16.4 vs 27.6), M 128 25.6 vs 44.0, M 512 ~67 vs 69.6, M 1024 ~125 vs 112.6 (the only case slower: prefill chunks)". `num_stages=3`.
- `hc_down(x, w8, scale, out=None, plan_fn=None)`: asserts `K % (bk*splits) == 0`.

**Target patch.** `model_executor/layers/quantization/utils/humming/linear.py`. Marker `# flashnext HC-down split-K (session 22)`.
1. **helpers**, before `def _flashnext_small_pad(layer) -> bool:`:
   - `_fn_hcd_target(layer)`: `FLASHNEXT_HC_DOWN == "1"` and the layer prefix ends with `input_mix_weight_down_block_inject`.
   - `_fn_hcd_keep(layer, weight_schema, tensors)`: keeps the checkpoint FP8 `[N,K]` weight and fp32 scales **before** Humming re-lays them out. It raises on a bias or extra tensors, or when the weight is not FP8 e4m3. It handles scale shapes: scalar, per-row, 2-D per (row, K-block) with K-group % 64 == 0, and 2-D block scales via `weight_scale_group_size_n` (`repeat_interleave`). The pad rows (output partitions `[lora rank, hc count, pad]`) are zeroed, because "the pad rows get no checkpoint weight (uninitialized memory, can be NaN)". It stores `layer._fn_hcd_pending`.
   - `_fn_hcd_check_and_switch`: a self-check on a random 16 x K input (seed 1234) against `apply_humming_linear`. It requires a relative max diff below `2e-2` on the real rows. Then it **frees Humming's copy** (each param is replaced by an empty tensor) and sets `layer._fn_hcd = (w8, s32)`. It prints `flashnext HC-down split-K: ON | first layer ...` once.
2. **keep the FP8 weight before the transform**: after `layer.weight_schema = weight_schema`.
3. **self-check, free, switch**: before `return config` at the end of the param-setting loop.
4. **routing**: at the top of `apply_humming_linear` (before `from vllm.utils.humming import humming_forward`), when `layer._fn_hcd` exists it calls `hc_down` on the flattened x.

**Flag.** `FLASHNEXT_HC_DOWN=1`.
**Observation.** The route is unconditional on M once a layer is switched. Every M goes to split-K, including M 1024 prefill chunks, where the docstring says it is slower (125 vs 112.6 us).

## Cell 24: DCP communication (QSA attention)
**Target.** `models/qwen4_exp/nvidia/qsa.py`. Marker `# flashnext DCP comm (session 22)`.
1. **query all-gather started before the indexer**: before `selected = self.indexer(projected_qk, positions, self.topk_indices_buffer[:num_tokens])`. When `_FN_DCP_QUERY_EARLY` is set and `impl.dcp_world_size > 1`, the query is K4-rotated (`k4_rotate`) if `impl._fn_k4`. Then, on a side stream (`_fn_dcp_side_stream()`, one lazily created `torch.cuda.Stream`), it runs `get_dcp_group().all_gather(_q, dim=1)`, with `record_stream`. The result is stored in `impl._fn_query_all = (_qa, _side)`.
2. **the attention uses the early all-gather**: replaces `query_all = group.all_gather(query[:num_tokens].contiguous(), dim=1)`. It pops `_fn_query_all` and waits on the side stream; otherwise it falls back to the original.
3. **BF16 partials**: before `merged = self.dcp_combine(partial, lse, group, is_lse_base_on_e=False)`, `partial = partial.to(torch.bfloat16)` when `_FN_DCP_BF16_PARTIAL`.
4. **switches + side stream**: before `class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):`. Prints `flashnext DCP comm: query all-gather early ON` / `BF16 partials ON`.

**Flags.** `FLASHNEXT_DCP_QUERY_EARLY=1`, `FLASHNEXT_DCP_BF16_PARTIAL=1`.
**Why faster.** The query all-gather of the DCP2 pair overlaps with the indexer compute. The BF16 partials halve the bytes of the DCP combine (fp32 to bf16).

## Cell 25: decode trims (9 switches) + 2 new kernel modules
Marker `# flashnext decode trims (session 22)`. Asserted files: qsa.py, `ops/qsa_indexer.py`, model_runner.py, `ops/qsa_k4.py`, `model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, `nvidia/hyperconnection.py`, `ops/fn_hc_down.py` (from cell23), `v1/worker/mamba_utils.py`. **Order dependency.** It must run after cells 22, 23 and 24, because its anchors contain their inserted lines.

**New module `ops/fn_hcd_silu.py`** (`hc_down_silu`). This is the same `_fn_hcd_partial` with a new reduce, `_fn_hcd_reduce_silu`. It uses one program per row, BLOCK = next_pow2(lora+hc), 4 warps. It sums the splits, rounds to bf16, then computes `x = v/HC; y = x*sigmoid(x)`. It writes `lora_out [M, lora_rank]` and `inj_out [M, hc_count]` as two contiguous tensors and drops the pad. The header says it is "Bit-identical to fn_hc_down + split + vLLM's _hc_silu_kernel" and gives "One kernel less per HC mixer (96 per decode step)". It also explains why: "vLLM's fused down + SiLU GEMM needs sm_90".

**New module `ops/fn_qsa_gate.py`** (`qsa_gate_store`). This is one Triton kernel, grid (T, H), 2 warps: `out = bf16(float(bf16(merged)) * sigmoid(float(gate)))`, using `libdevice.exp` and `div_rn`. It replaces "~6 small kernels per QSA layer". The header says it is "Bit-identical (Colab L4 check over all 65,536 bf16 gate values)".

**Edits per file:**
- **qsa.py**
  - (a) Switches `_FN_TRIM_K4_ROT`, `_FN_TRIM_QSA_GATE`, inserted after cell24's `_FN_DCP_QUERY_EARLY` line.
  - (b) **K4_ROT**: at `key_cache, value_cache = k4_views(...); query = k4_rotate(query)`, the rotation is skipped when `_FN_TRIM_K4_ROT and "_fn_query_all" in self.__dict__`, because "the early DCP copy is already rotated". This only works together with DCP_QUERY_EARLY.
  - (c) **QSA_GATE**: at `gate = output_gate[:num_tokens].reshape(...)`, it calls `qsa_gate_store(merged, gate, output[:num_tokens]); return output`.
- **qsa_indexer.py**
  - (a) **GAP_ONCE**: the original `_fn_sw_gap` is renamed `_fn_sw_gap_compute`. A new `_fn_sw_gap` wraps it in `_fn_step_memo`, which is a per-forward-context memo. The key includes `torch.cuda.is_current_stream_capturing()`, the visible_blocks pointer/shape/stride/dtype, token_topk, compress_ratio, row_start and the sink pointer. It is reset when `get_forward_context()` changes. It returns None when `_FN_SW_WINDOW <= 0 and _FN_SW_LIMIT <= 0` (the SINK_JUMP window is off).
  - (b) **compact arguments once**: `_fn_lo`, `_fn_shift`, `_fn_vis` (int32 casts plus a subtraction) are memoized per forward pass.
  - (c) **QSA_SHIFT** switch + kernel `_fn_shift_back_kernel`: grid (rows, cdiv(cols,1024)), 4 warps. It computes `b >= lo ? b+sh : b` in place.
  - (d) It replaces `block_indices.add_(torch.where(block_indices >= _fn_lo[:, None], _fn_shift[:, None], 0)...)` with `_fn_shift_back`. This is one kernel instead of compare + where + cast + add.
- **model_runner.py**: **TOPK_BF16**. `torch.topk(logits_shard, k)` runs on the BF16 logits, then `.float()` on the K values ("BF16 -> FP32 is exact, no full FP32 copy").
- **qsa_k4.py**: **K4_FWHT**
  - `k4_rotate` uses `_fn_fwht_rotate` (head size 256 only) instead of `x.float() @ hadamard(D)`.
  - `_fn_fwht256` does 8 butterfly stages (Sylvester order) via reshape/permute/split/join, then `* 0.0625`.
  - `_fn_fwht256_kernel` uses BLOCK_M=2 and `num_warps=1`.
  - `_k4_store_kernel` gets a new constexpr `ROTATE: tl.constexpr = False`. When it is set, the key is rotated in-kernel, and the store call passes the BF16 key directly with `ROTATE=True, num_warps=2`, replacing `k_rot = (key[:n].float() @ hadamard(...))`.
  - Why faster: an O(D log D) butterfly in registers instead of an fp32 256x256 matmul plus casts, and for the keys it is fused into the cache store.
- **qwen_gdn_linear_attn.py**: **GDN_AB**. The call site of `torch.ops.vllm.qwen_gdn_attention_core` passes strided `b`/`a` views (no `.contiguous()`). The copy moves into the mixed path (`b = b[:num_actual_tokens].contiguous()`), where it is a no-op when the views are already contiguous.
- **hyperconnection.py**: **HCD_SILU**. `_FN_HCDS_MAX_M = 128` ("larger M keeps split-K + vLLM's SiLU (M 1024: +6.6 us fused)"). At `down_and_injection = self.input_mix_weight_down_block_inject(xn)`, it returns `hc_down_silu(...)` directly. It raises if `_fn_hcd` is missing ("needs ... FLASHNEXT_HC_DOWN=1").
- **mamba_utils.py**: **MAMBA_TILES**. `_TEMPORAL_TILES = 1 if FLASHNEXT_TRIM_MAMBA_TILES == "1" else 16` (the DeltaNet align copies use 1 tile).

**Flags.** `FLASHNEXT_TRIM_{K4_ROT, GAP_ONCE, TOPK_BF16, K4_FWHT, GDN_AB, HCD_SILU, MAMBA_TILES, QSA_GATE, QSA_SHIFT}=1`.

## Cell 26: markdown
`# Kernel Cache Restore`.

## Cell 27: startup cache restore
- It looks for `startup_caches*.tar.gz` or `startup_caches*.tar.gz.bin` at fixed `/kaggle/input` depths (`/kaggle/input/*`, `*/*`, `datasets/*/*`, `datasets/*/*/*`), falling back to a recursive search. It extracts to `/` with `filter="tar"`.
- Fallback: an unpacked `root/.humming|.triton|.cache` folder, copied with 32 threads.
- If there is neither, it raises ("Attach the fast-startup dataset").
- It prints file counts for `/root/.cache/flashinfer`, `/root/.humming`, `/root/.triton`, `/root/.cache/vllm`.
- FlashInfer AOT: for each `/root/.cache/flashinfer/*/*/cached_ops/<name>/<name>.so` it copies to `/tmp/site-vllm/flashinfer/data/aot/<name>/<name>.so`, "loaded as prebuilt, never rebuilt". Without any, it prints "the sampling kernel will be compiled during the start, ~2 min".
- Sets `CACHE_RESTORED_T`.

## Cell 28: markdown
`# Harness`.

## Cell 29: harness repo + Docker-free sandbox + memory guard
- It copies `WHEELHOUSE/'pivotools'` to `REPO=/tmp/arc-agi-2-submission` (WHEELHOUSE and ASITE come from cell07).
- Constants: `KERNEL_MEM_GIB = 8`, `RAM_BRAKE_GIB = 12`, `MEM_GUARD_LOG = /kaggle/working/memory_guard.log`.
- `ARC_ENV`: PYTHONPATH = ASITE + REPO/src, `HF_HUB_OFFLINE=1`, `TOKENIZERS_PARALLELISM=false`, plus the ARC_KERNEL_MEM_GIB / ARC_RAM_BRAKE_GIB / ARC_MEM_GUARD_LOG env vars.
- Import check: `arcagi2, jupyter_client, ipykernel, numpy, scipy, shapely, networkx, skimage, ortools, z3, openai`.
- It writes `src/arcagi2/sandbox/local_sandbox.py`:
  - `LocalKernelSandbox` / `LocalKernelREPL` run one real IPython kernel subprocess per REPL through `AsyncKernelManager`.
  - An own kernelspec (`_OwnSpecManager`) with a launcher that empties `SYSTEM_CONFIG_DIRS` / `ENV_CONFIG_DIRS`, plus `IPYTHONDIR` per workdir and `JUPYTER_NO_CONFIG=1`, and a temp working directory.
  - `wait_for_ready(timeout=60)`.
  - Result handling matches ipybox: stream text plus the text/plain of results, and the error = traceback lines.
- **Memory guard** thread (1 s loop):
  - It sums the RSS of the kernel and its direct children from `/proc`.
  - It kills (SIGKILL) a kernel above 8 GiB.
  - When `MemAvailable < 12 GiB`, it kills the largest live kernel above 0.5 GiB, then sleeps 5 s.
  - The killed REPL restarts its kernel, and the model gets `MemoryError: this python process ... replaced by a fresh one`.
- Timeout path: `interrupt_kernel`, then wait_for_ready 5 s / 8 s; otherwise it replaces the kernel and raises `ExecutionTimeoutError`.
- Self-check subprocess: state, errors, and the guard kill test (`b'x' * (lim+1) GiB`).

## Cell 30
Covered by another agent.

## Cell 31: harness fixes "step 5v" (clients_v3.py + config)
Targets: `src/arcagi2/api/clients_v3.py`, `src/arcagi2/solver/config/qwen3_8_flash_next_free.py`. Marker `# session 18b: harness fixes (step 5v)`. It asserts that step 5u was not run (`"an empty reply keeps the conversation" not in _c`).
1. **tool output cut**: the middle cut is now at whole lines, with the message `... [output lines h+1-n-t of n not shown (k lines, c characters)] ...` (head = limit - limit//3, tail = limit//3).
2. **helpers bg='auto'**:
   - New `_bg_of(g, bg)` = the most common colour (ties: the smallest colour).
   - `bbox(grid, bg="auto")` and `components(grid, bg="auto", ...)` replace `bg=0`, both in the session code and in the escaped `helper_source` copy.
   - `_GRID_DEPS` gains `"bbox": ["_bg_of"], "components": ["_bg_of"]`.
   - The prompt edits in the config: `bbox(g, bg='auto')` and the `components(...)` description.
3. **recovery keeps the conversation**: after a rejected final answer, the old code did `head = base_head + self._recovery_pair(...); entries.clear(); st.rebase = True`. It now appends the assistant entry and the feedback user entry. Empty content (thinking only) is set to `""`, because "vLLM refuses content None without tool calls (HTTP 400)".

Live check: the import, `bbox` / `components` on a test grid, the `helper_source` copy, and that the prompt contains `bg='auto'`.

## Cell 32: harness fixes "step 5z" (sandbox robustness)
Targets: `local_sandbox.py`, `api/clients.py`, `api/clients_v3.py`.
- **(a) reader stops.** `_run_guarded` gets a `finally` that cancels the `_run` reader if it is not done: "before, the reader kept going, took the output of later calls and made some fail with an empty error (-> the whole run restarted)".
- **(b) broken kernel replaced + setup re-run.**
  - New `self.setup_code = []`. clients.py sets `repl.setup_code = list(initial_code)` after "Initial code in REPL executed successfully".
  - New `_replace_kernel(why)`, `_rerun_setup()` (each setup cell with a 120 s timeout) and `_fresh_note()` ("`puzzle`, check() and the helper functions are loaded again...").
  - An interrupt failure becomes `ExecutionError("TimeoutError: ... replaced by a fresh one")`.
  - A memory stop re-runs the setup.
  - Any other exception becomes `PythonProcessError` with a kernel replace "instead of a whole-run restart". Before, it was `SandboxInfrastructureError`, so the solver restarted the WHOLE run.
- **(c) empty reply.** All None-valued keys are deleted from the assistant message, and content is set to `""`.

Live check: 3 timeouts then 45 quick calls must all be correct; a call that ignores the interrupt; a memory stop; `queue.Empty` on iopub ("what the 64K run's broken calls raised").

## Cell 33: prompt and protection fixes "step 5p"
It asserts `SINK_JUMP[0] == "auto"` (from cell10: `SINK_JUMP = ("auto", 49152, 65536)`). `_RECENT` is computed but not used in the edits.
- **repl_tool.py**:
  - The redefine check becomes `(?<![\w.])puzzle\s*=(?!=)` (so `my_puzzle = ...` and `puzzle == ...` are allowed).
  - The print check refuses only a whole-dict `print( puzzle )`.
  - New message: "Nothing in this cell was run. Print only the part you need".
- **system_prompt_free.txt**:
  - Rule 1: the final answer runs in a fresh session and must define everything.
  - New rule 4: a cell printing the whole `puzzle` or assigning to it is not run.
- **agentic_coding_solver_free.txt**:
  - The memory line is replaced by "You always see this first message; older parts of the conversation fade ...{{BUDGET_LINE}}" (no token numbers, "nonum").
  - The thinking line "Do not think for long ..." becomes "Think carefully, and do the validating with code".
  - The output format: "the LAST one is used", self-contained, a fresh session.
- **config `qwen3_8_flash_next_free.py`**: `{{BUDGET_LINE}}` is filled when `ARC_MAX_GEN_TOKENS` is set: "The work may stop before you give a final answer; then the functions you ran through `check()` are used instead (most training examples right first), so run `check()` on every promising `solution`." The check() part appears only when `ARC_CHECK_ATTEMPTS` and `ARC_CHECK_HELPER` are set.

The check asserts the removed and added strings, including "Your very first code execution should be `puzzle.keys()`", and 12 protection cases.

## Cell 34: markdown
`# Fast Server Startup`.

## Cell 35: fast n-gram (PLE / engram) load + page-cache drop + util
Constants: `EP_WEIGHT_FILTER = True`, `DROP_TAR_CACHE = True`, `GPU_UTIL_FAST = 0.945`, which becomes `GPU_UTIL`.

**Edits:**
1. `model_executor/model_loader/weight_utils.py`
   - **lazy row-shard view**: class `_FNLazyRows`, inserted before `def safetensors_weights_iterator(`. It wraps a safetensors slice; only `narrow(0, start, n)` reads data.
   - **iterator**: `param = f.get_tensor(name)` becomes `_FNLazyRows(f.get_slice(name))` when `FLASHNEXT_FAST_NGRAM=1` and the name contains `ngram_embedding.shard_`.
2. `models/qwen4_exp/common/ngram_embedding.py`
   - `import os`.
   - The pinned-CPU PLE weight allocation becomes `_flashnext_shm_weight` when `FLASHNEXT_NGRAM_SHM` is set: an `mmap` of the `/dev/shm` table file (global row order, written by step 0, which requires `meta["complete"]` and checks `row_bytes` / `torch_dtype`), `cudaHostRegister(ptr, n, 2)`, and a view. "Nothing is allocated or copied", and `is_pinned()` is asserted.
3. `models/qwen4_exp/common/ple.py`
   - `_flashnext_check_shard` compares 3 rows (first, middle, last of the overlap) byte for byte instead of copying.
   - `copy_ple_embedding_shard_` is skipped when `_flashnext_shm` is set.

It also sets `FLASHNEXT_NGRAM_SHM=NGRAM_SHM` and `FLASHNEXT_FAST_NGRAM=1` (NGRAM_SHM comes from cell05). With `DROP_TAR_CACHE`, `posix_fadvise(DONTNEED)` drops every `UTILITY_PATH/*.tar` from the page cache, with a `meminfo()` print before and after.

## Cell 36: KV sizing
`os.environ.pop("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", None)` and `KV_CACHE_BYTES = None` (so vLLM measures the KV at GPU_UTIL). The print says: "the 104-graph runs had 8.81 GiB per GPU = 4,064,286 tokens".

## Cell 37: wait for the model prefetch inside the vLLM worker
In every `v1/worker/**/*.py` that logs "Loading model from scratch", it inserts code before the logger call. The code waits until the log `FLASHNEXT_WAIT_PREFETCH` contains `PREFETCH DONE`. It raises on a Traceback/Error in the `.err` file, on a dead `FLASHNEXT_WAIT_PREFETCH_PID`, or after **1800 s**. It logs the wait time. This lets the vLLM process start (imports, NCCL, etc.) while the step-0 prefetch is still reading the model.

## Cell 38: EP filter on all tensors
In `weight_utils.py`, every `if should_skip_weight(name, local_expert_ids):` gets `or _fn_skip_nonlocal_expert(...)` (regex `\.experts\.(\d+)\.`). It skips every tensor of a non-local expert when `FLASHNEXT_EP_FILTER_ALL=1` and logs every 25,000 skips. It needs `--enable-ep-weight-filter`. The cell sets `FLASHNEXT_EP_FILTER_ALL=1`.

## Cell 39: markdown
`# Speed Switches and Collapse Fix`.

## Cell 40: speed switches
`SPEED_SWITCHES = True` sets:
- `VLLM_HUMMING_MOE_GEMM_TYPE=indexed`
- `FLASHNEXT_TOPK_EXCHANGE=1`
- `FLASHNEXT_DCP_QUERY_EARLY=1`
- `FLASHNEXT_DCP_BF16_PARTIAL=1`
- `FLASHNEXT_HC_DOWN=1`
- `FLASHNEXT_TRIM_{K4_ROT, GAP_ONCE, TOPK_BF16, K4_FWHT, GDN_AB, HCD_SILU, MAMBA_TILES, QSA_GATE, QSA_SHIFT}=1`

It asserts `FLASHNEXT_BATCH_SHARDED_SAMPLING == '1'`. `False` pops them all.

## Cell 41: miss-deferral collapse fix
Targets: `v1/core/sched/scheduler.py` (the helpers are appended at the end of the file) and `v1/core/block_pool.py`. Marker `# flashnext miss deferral (session 22)`.

Parameters (env, default):

| Parameter | Env var | Default | Note in code |
|---|---|---|---|
| window_s | `FLASHNEXT_DEFER_WINDOW_S` | 60 | |
| max_s | `FLASHNEXT_DEFER_MAX_S` | 1200 | "replay: 180 s re-started the cascade; harness timeout 3600 s" |
| gap_s | `FLASHNEXT_DEFER_GAP_S` | 60 | |
| min_tokens | `FLASHNEXT_DEFER_MIN_TOKENS` | 16384 | |
| frac | `FLASHNEXT_DEFER_FRAC` | 0.5 | |

Logic:
- `_fn_md_pressure()`: an attention-block eviction happened within window_s.
- `_fn_md_defer`: it skips a request. After waiting max_s (aging), it is released, one at a time. The next release waits until the previous released request finished its prefill and gap_s has passed. It counts deferred / released / held_back.
- `_fn_md_report`: every 60 s it prints `flashnext miss defer (total): ...`.

Scheduler edits:
- At `while token_budget > 0:` (where "Requests holding KV blocks are always drained first"), it calls the report.
- After the `_ec_transfer_pending` skip, it skips a waiting request when ALL of these hold: on, not `load_kv_async`, `num_prompt_tokens >= 16384`, `num_computed_tokens < 0.5*prompt`, pressure, (running or other queued), and `_fn_md_defer`.

block_pool edit: in eviction, after `_remove_cached_block_hashes`, unless `_fn_dropping` is set, it counts evictions whose group id is not in `_FN_SEG["state_groups"]` (that is, attention groups, not DeltaNet state groups; `_FN_SEG` is from cell10's coordinator patch) and updates `last_att_evict`.

It asserts `FLASHNEXT_OFFLOAD_EVICT == "1"` (cell16, KV offload) and sets `FLASHNEXT_MISS_DEFER=1`.

**Why it helps.** Under KV pressure, a large uncached prompt (a long conversation whose prefix was evicted) is not admitted. Admitting it would evict more cached blocks, causing a cascade of misses (the "collapse").

## Cell 42: markdown
`# Start the vLLM Server`.

## Cell 43: server launch + retry/classify
Constants:

| Constant | Value |
|---|---|
| `SERVED_NAME` | "Qwen/Qwen3.8-Flash-Next" |
| `BASE_PORT` | 8000 |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | 1 |
| `MAX_LEN` | 1048576 |
| `MIN_LEN` | 16384 |
| `MAX_SEQS` | 128 |
| `MTP_TOKENS` | 0 (MTP off) |
| `USE_IMAGES` | True |
| `MAX_BATCHED` | 4096 |
| `DISABLE_CUSTOM_AR` | True |
| `STARTUP_TIMEOUT_S` | 90 min for flashnext |

**Prefetch hookup.** When `PREFETCH_READERS` is set, it checks that the step-0 prefetch is alive or done, then sets `FLASHNEXT_WAIT_PREFETCH` / `_PID` (used by cell37).

**`build_cmd`.** `vllm.entrypoints.openai.api_server` with:
- `--tensor-parallel-size TP`, `--max-model-len`, `--gpu-memory-utilization`, `--max-num-seqs 128`, `--max-num-batched-tokens 4096`
- `--reasoning-parser qwen3`, `--enable-auto-tool-choice --tool-call-parser qwen3_coder`, `--default-chat-template-kwargs {"preserve_thinking": true}`
- optional `--kv-cache-memory` and `--cudagraph-capture-sizes` (CUDAGRAPH_CAPTURE_SIZES)
- `--disable-custom-all-reduce`, `--enable-prefix-caching`, `--mamba-cache-mode align`, `--kv-cache-dtype KV_DTYPE`
- `--engram-config {"cpu_offload": true}`
- `--enable-expert-parallel` + `--enable-ep-weight-filter`

`strip_flags` drops flags vLLM reports as unknown.

**`start()`.**
- It runs Popen in a new session. Env = `vllm_env` + `CUDA_VISIBLE_DEVICES` + `NCCL_DEBUG=WARN`.
- It polls `/v1/models` every 3 s and scans the log incrementally for a `_FATAL` regex. It sleeps 20 s, then returns `engine_error`. It prints GPU memory every 30 s.

**`start_copy` (up to 8 attempts).**
- An engine_error raises with no retry. A build error or a hardware error also raises.
- badflags: drop the flags. nccl: `NCCL_P2P_DISABLE=1`.
- kv: `max_len = int(est*0.97)//1024*1024` (if >= MIN_LEN).
- mtp_import / load_oom: disable MTP.
- memory: util down to 0.92, then eager, then no MTP.
- flags: the flag variants step through prefix+align, then prefix only, then none.
- other: no MTP, then eager.

**Other.**
- The background import check (cell10 `IMPORT_CHECK`) runs at `nice -n 19 ionice -c 3` while the server starts. On failure the server is stopped.
- It sets `WINDOW`, `N_SEQS`, `PORTS`, `USE_IMAGES_EFFECTIVE`, `MTP_TOKENS_EFFECTIVE` and prints key log lines.
- `stop_servers()`: SIGTERM, 20 s, SIGKILL on the process group, `pkill -9 -f vllm.entrypoints|VLLM::`, then a wait of up to 90 s until GPU used < 1500 MiB.

## Cell 44: verify the switches in the server log
For each set env, it requires its log line:
- `Using indexed gemm for humming moe`
- `flashnext HC-down split-K: ON`
- `flashnext DCP comm: query all-gather early ON`, `... BF16 partials ON`
- `flashnext decode trims: <T> ON` for all 9 trims

A missing line raises. It prints the HC-down first-layer line and `GPU KV cache size:`.

## Cell 45: markdown
`# Run the Tasks and Write the Submission`.

## Cell 46: the run (stage manager + monitor + revive + scoring)
**Notebook constants:**

| Constant | Value |
|---|---|
| `MAX_GEN_TOKENS` | 200_000 (per run) |
| `TEXT_GRIDS` | False (images + sizes + `puzzle`) |
| `CONCURRENCY` | 70 |
| `REASONING_EFFORT` | "xhigh" |
| `SEED` | 0 |
| `TRIM_TARGET` | 16384 |
| `TRIM_MARGIN` | 1024 |
| `TRIM_NOTE` | True |
| `RECOVERY_MAX` | 5 |
| `REPORT_EVERY_S` | 60 |
| `STOP_MARGIN_MIN` | 1 |
| `MAX_REVIVALS` | 3 |
| `THIRD_TRY` | False |
| `TRY3_MIN_LEFT_MIN` | 60 |
| `MIN_LEFT_FOR_REVIVAL_MIN` | 30 |

**Setup.**
- It asserts one server copy (`len(PORTS) == 1`) and that `served_len == WINDOW`.
- `STOP_AT = DEADLINE - 60 s`.
- The task ids are sorted, then shuffled with `random.Random(0)`.
- `SOLUTIONS` comes from `arc-agi_evaluation_solutions.json` next to TEST_FILE, if present (eval mode). Hidden test = no CORRECT counts.

**Stage manager** (`stage_manager.py`, a separate process run with the harness python):
- Settings: `second_try: True`, `round2/3_variants: []`, `total_budget: 0` (no token budget; time decides), `round_limits` 200k for rounds 1/2/3, `only_final_ends: True`, `poll_s 5`, `stop_epoch STOP_AT`, `time_limit_hours 12`.
- Runs: try 1 (`""`) for every task + an independent try 2 (`"<id>_2"`), queued from the start. Try 2 has the same grids, a new conversation, and no note. `INFORMED_TRY2 = False` ("session 22 (owner): no more 'evolve style': try 2 is plain and queued from the start").
- With THIRD_TRY, a plain try 3 would be queued for tasks whose try 1 gave no final answer. It is not started when fewer than 60 min are left. It is off.
- Scheduling: every 5 s, free slots = CONC - running (0 while the PAUSE file exists). Queued round-1 runs go first, then rounds 2/3 sorted by round. A budget check runs only if TOTAL > 0.
- Each run is `puzzle_worker(... num_samples=1)` with `view(task, TRANSFORM[var])`. T/F are self-inverse transforms; for "2" no transform is used.
- Per-run limits are written to `limits.json` (`ARC_RUN_LIMITS`), notes to `notes.json` (`ARC_RERUN_NOTES`).
- A run's `final` = an `attempt_1` grid in its round's submission. `full` = a check() record with ok == n. Tokens come from `V3_GEN (\d+)` in run.log.
- On deadline or SIGTERM, the live tasks are cancelled and finished. `runs.json` and `status.json` are written.
- Legacy round-2/3 notes (NOTE / NOTE_CONFIRM / PREV / STEPS breadth-first text) are present but inactive (empty variants).

**Harness env** (passed to the stage manager):
- `ARC_CTX_WINDOW=WINDOW`, `ARC_TRIM_TARGET=16384`, `ARC_TRIM_MARGIN=1024`, `ARC_TRIM_NOTE=1`
- `ARC_MAX_GEN_TOKENS=200000`, `ARC_IMAGES=1`, `ARC_COMPACT=1`, `ARC_RECOVERY_MAX=5`
- `ARC_DEADLINE_EPOCH=STOP_AT`, `ARC_SAMPLE_SECONDS=0`, `ARC_SALVAGE_AT_DEADLINE=0`
- `ARC_REASONING_FIELD=reasoning`, `ARC_REASONING_EFFORT=xhigh`
- `ARC_CHECK_HELPER=1`, `ARC_GRID_HELPERS=1`, `ARC_CHECK_ATTEMPTS=1`
- `ARC_THINK_CAP=0`, `ARC_BUDGET_REMINDERS=0`, `ARC_FINAL_REVIEW=0`, `ARC_TEXT_GRIDS=0`

**Monitor loop (notebook).** Every 60 s it runs `report()`:
- WORKING counts per try, ANSWERED (via `write_submission(quiet=True)`, from cell03), and live CORRECT per try (`correct_now` maps T/F back).
- Union of correct tasks; tokens decoded.
- **LOAD line** from `/metrics` deltas: decode tok/s, prompt computed tok/s, prefix hit %, RAM (external) hit %, preemptions, KV usage, running/waiting.
- GPU blocks from `KVDIAG pool | num_blocks` and the `FLASHNEXT CENSUS free` log lines, plus the `HITDIAG summary` line.
- RAM available and guard stops; events; the last error line.

**Revive logic.**
- `dead_servers()` flags a server as dead when: the process exited, or /metrics fails 3 reports in a row (`SERVER_DOWN_LIMIT = 3`), or no generated token for 10 min (`NO_PROGRESS_MIN = 10`, "hung").
- `revive()`: write the PAUSE file, print error lines and `free -g` / dmesg OOM, `write_submission()`, back up the log, `stop_servers()`, restart with the same `S` settings, remove PAUSE.
- It is allowed when `revivals < 3` and at least 30 min are left.

**End.**
- At the deadline it waits up to 120 s for the manager, then `terminate` and wait 120 s, then kill.
- Final `report()`, `write_submission()` and "FINAL submission.json ... -> SUBMISSION".
- `_load_summary()` covers the mean of rows from minute 10 on and writes `/kaggle/working/loadtest_rows.json`.
- **Eval score**: per task, the sum over test outputs of `g in (attempt_1, attempt_2)` divided by the number of tests. It prints `SCORE (trial, solutions known): x / N`.
- It copies evaluate_0.log, runs.json and status.json to /kaggle/working, then runs `stop_servers()`.

---

## Dependencies (cross-cell)
- **VROOT, SINK_JUMP, CUDAGRAPH_CAPTURE_SIZES, IMPORT_CHECK, `_FN_SEG`** are defined in cell10.
  - `SINK_JUMP = ("auto", 49152, 65536)`.
  - `CUDAGRAPH_CAPTURE_SIZES = [1, 2, 4, 8, 16, 32, 48, 64, 72, 128, 256, 512]` (cell10 line 7).
  - `_FN_SEG["state_groups"]` comes from the cell10 coordinator patch.
- **WHEELHOUSE, ASITE** are defined in cell07.
- **MODEL_KIND, TP, REPLICAS, KV_DTYPE, EXPERT_PARALLEL, N_GPUS** are defined in cell06; **MTP_PRESENT** in cell08.
- **`FLASHNEXT_BATCH_SHARDED_SAMPLING=1`** is set in cell13. It is required by cells 22 and 40.
- **`FLASHNEXT_OFFLOAD_EVICT=1`** is set in cell16 (KV_OFFLOAD_GIB). It is required by cell41.
- **NGRAM_SHM, UTILITY_PATH, PREFETCH_LOG / PREFETCH_PROC / PREFETCH_READERS** are defined in cell05.
- **DEADLINE, DEADLINE_HIT, write_submission, `_LAST_GOOD` / `_LAST_REC`, TEST_TASKS, TEST_FILE, WORK, SUBMISSION** are defined in cell03.
- **Patch order is load-bearing:**
  - cell25 anchors on lines inserted by 22 (`_fn_topk_all_to_all`, `torch.topk(logits_shard.float()...)`), 23 (`fn_hc_down.py` imports `_fn_hcd_partial, _workspace, plan`) and 24 (`_FN_DCP_QUERY_EARLY` line).
  - cell25 also anchors on lines from earlier cells in qsa_indexer.py (`_fn_sw_gap`, `_fn_compact`, `_FN_SW_CHECK`, `_FN_SW_WINDOW`, `_FN_SW_LIMIT`, `_FN_SW_SINK_TOK`), qsa_k4.py (`k4_rotate`, `_k4_store_kernel`, `k4_views`) and hyperconnection.py (`cute_dsl.hc_down_silu` import). Those come from cell10, cell15 and cell30.
  - The harness cells run in order 29, then (30), then 31 (5v), 32 (5z), 33 (5p). 31 asserts "step 5u was not run". 32 needs 5v's text.

## Memory / capacity facts (from these cells)
- KV: "the 104-graph runs had 8.81 GiB per GPU = 4,064,286 tokens" (cell36). KV is measured by vLLM (`KV_CACHE_BYTES = None`) at `GPU_UTIL = 0.945` (cell35). The fallback on memory errors is 0.92 (cell43).
- Window `MAX_LEN = 1048576`. The auto-shrink on a KV error is to `int(est*0.97)//1024*1024`, with a floor of `MIN_LEN = 16384` (cell43).
- `MAX_SEQS = 128` server-side, but the harness `CONCURRENCY = 70` (cell46). `MAX_BATCHED = 4096`.
- Context trimming in the harness: `TRIM_TARGET 16384`, `TRIM_MARGIN 1024` (cell46). The prompt deliberately omits token numbers (cell33).
- Per-run decode limit: 200,000 tokens. There is no total token budget; time decides (cell46).
- The n-gram/PLE table lives in /dev/shm, mmap'd and `cudaHostRegister`'d. There is no pinned copy per process (cell35). The utility tars are dropped from the page cache to free RAM (cell35).
- Engram config `{"cpu_offload": true}` (cell43).
- The EP weight filter on all expert tensors (cell38) means only local experts are read. `--enable-ep-weight-filter` (cell43).
- HC-down: Humming's copy of the HC-down layer is freed after the switch. The FP8 `[336, 10240]` copy plus scales are kept per HC layer (cell23).
- RAM guard: 8 GiB per tool kernel, a brake below 12 GiB available (cell29).
- Miss deferral parameters: window 60 s, max wait 1200 s, gap 60 s, min 16,384 prompt tokens, under 50 % computed (cell41).
- Prefetch wait limit 1800 s (cell37). Server startup timeout 90 min (cell43).

## Run outputs or measured numbers present in the cells
- cell23 `plan` docstring (L4, N 336, K 10240), split-K vs Humming: "M 60 23.5 vs 28.7 us (weight in L2 16.4 vs 27.6), M 128 25.6 vs 44.0, M 512 ~67 vs 69.6, M 1024 ~125 vs 112.6". The header gives Humming "~33 us at M 60 ... 27 of 58 SMs".
- cell25: HCD_SILU "M 1024: +6.6 us fused" (hence `_FN_HCDS_MAX_M = 128`); "One kernel less per HC mixer (96 per decode step)"; QSA gate replaces "~6 small kernels per QSA layer", with a bit-identity check "over all 65,536 bf16 gate values".
- cell36: "8.81 GiB per GPU = 4,064,286 tokens" (104-graph runs).
- cell41: "replay: 180 s re-started the cascade; harness timeout 3600 s" (the reason max_s is 1200).
- cell46: "Kaggle load test with improvements 1-3 at CONCURRENCY 60 for comparison: decode 651 tok/s (minutes 10-81), prefix hit 94%, preemptions 0".
- cell32: a reference to "the 64K run's broken calls" raising `queue.Empty`.
- cell27: "the sampling kernel will be compiled during the start, ~2 min" when the FlashInfer AOT cache is missing.
- There are no saved cell outputs in these text files (code only).

## Tunable for more speed or a bigger pool (observations with line evidence)
1. **Concurrency headroom.** The server has `MAX_SEQS = 128` (cell43) but the harness runs `CONCURRENCY = 70` (cell46). The only throughput reference is "651 tok/s at CONCURRENCY 60" (cell46 l.561). This is a direct knob, bounded by KV (4.06 M tokens, cell36) and by the miss-deferral collapse behaviour (cell41).
2. **HC-down split-K at large M.** `apply_humming_linear` routes every M to split-K once switched (cell23, edit "routes to the split-K kernel"). The plan docstring shows it slower at M 1024 (125 vs 112.6 us). An M threshold (for example, Humming for M > 512) is not implemented. The `plan()` table is hand-searched; for M > 256 it uses splits 2.
3. **Top-k K.** `FLASHNEXT_TOPK_EXCHANGE_K` defaults to 64 and is not set in cell40. Any request with top_k outside 1..64 falls back for the whole step (cell22 `_fn_topk_reason`). The "full" reasons are printed every 60 s by rank 0, which can be read from the log to confirm the fast path rate.
4. **K4_ROT depends on DCP_QUERY_EARLY** (`"_fn_query_all" in self.__dict__`, cell25). Both are on in cell40.
5. **FWHT kernel launch config.** `BLOCK_M=2, num_warps=1` for the queries; `num_warps=2` for the key store (cell25). These are fixed constants, not autotuned.
6. **HCD_SILU threshold** `_FN_HCDS_MAX_M = 128` (cell25), a fixed constant.
7. **GPU memory share** `GPU_UTIL_FAST = 0.945` (cell35); `KV_CACHE_BYTES = None` (cell36). `--kv-cache-memory` is wired in `build_cmd` (cell43) for an explicit KV size.
8. **CUDA graphs.** The cell36 text says "with the CUDA-graph estimate on (graphs up to 64)", but cell10 sets `CUDAGRAPH_CAPTURE_SIZES = [1, 2, 4, 8, 16, 32, 48, 64, 72, 128, 256, 512]`, and cell36 also mentions "104-graph runs". The cell36 text and the list disagree; the effective capture list should be read from the server log.
9. **Prefill batch** `MAX_BATCHED = 4096` (cell43) is fixed, and the miss-deferral min is 16,384 prompt tokens (cell41). Both are knobs on prefill vs decode balance.
10. **Custom all-reduce** is disabled (`DISABLE_CUSTOM_AR = True`, cell43).
11. **MTP is off** (`MTP_TOKENS = 0`, cell43). The top-k exchange also falls back when draft tokens exist (cell22), so enabling MTP would disable it.
12. **Miss-deferral parameters** are all env-overridable (`FLASHNEXT_DEFER_WINDOW_S/MAX_S/GAP_S/MIN_TOKENS/FRAC`, cell41). Only max_s has a recorded rationale.
13. **Pool / attempt policy.**
    - `MAX_GEN_TOKENS = 200_000` per run.
    - Try 2 is always queued (`second_try: True`). Try 3 (`THIRD_TRY = False`) is implemented, gated by 60 min left.
    - `total_budget: 0` means time-only scheduling.
    - Round-1 runs always go first; then try 2, then try 3 (cell46 `extra_order`).
    - The budget function `committed()` uses `max(r.read_used(), LIMIT)` for running runs. It is inactive here because TOTAL = 0.
14. **Monitoring cost.** `poll_s 5` and `REPORT_EVERY_S 60`. `report()` reads the last 3000 lines of the server log for HITDIAG every minute (cell46). This is minor.
15. **Startup.** The prefetch overlaps with the vLLM start (cells 37 and 43). The FlashInfer AOT cache avoids a ~2 min compile (cell27). The EP filter skips non-local experts (cell38). The /dev/shm n-gram table needs no copy (cell35). Revive restarts reuse all of this, with at most 3 restarts and only when at least 30 min are left (cell46).
