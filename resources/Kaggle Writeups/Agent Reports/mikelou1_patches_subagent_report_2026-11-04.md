## mikelou1/arc-agi2 patches: full read of all 38 files

Every file was read in full, as text only; nothing was executed. Path: `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/mikelou_cells/patches/`

**There are 38 `.py` files in the folder, not 39.** `ls`/`wc` list 38.

**No bench file contains any measured result.** Every bench only computes numbers at runtime and prints one JSON line. So the bench sections below give the method, the gates and the constants. Every real measured number in this folder comes from patch docstrings. Those docstrings name earlier runs: v59, v76, v80, v86, v92, v99, and the "G4 test 2026-10-08".

Line numbers below are within each file.

---

## A. Rung-0 env flags mapped to patches

| Rung-0 env | Patch / file | Notes |
|---|---|---|
| SGLANG_EMBED_HOST=1 | embedhost_patch | embed_tokens in pinned host memory |
| SGLANG_HC_MIX_MAX_ROWS=64 | hcmix32_patch | 17..64 rows go to the fused kernel (clamped to 64, l.78-80). Over 32 rows: split into 32-row ROWS=32 launches (l.185-188) |
| SGLANG_MAMBA_PREFILL_FINAL_ONLY_LAZY=1 | mambafinal_patch | |
| SGLANG_MAMBA_SLOTS_PER_REQ=2 | mambaratio_patch | |
| SGLANG_QSA_MQA_GROUPS_PER_CTA=4 | mqagrid_patch | |
| SGLANG_QSA_PAIR_OWN=1, SGLANG_QSA_PAIR_OWN_GROUP=pair | pairown_patch | hicpairown also reads pairown's `_S["cfg"]` |
| SGLANG_QSA_SM89_DECODE=triton_splitk | apply_patches (qsa_sm89_decode) + qsasplitk_patch | The duck2 selector treats the unknown value as "auto", which gives "triton" on sm_89. splitk_route then switches to split-K. No SPLITK_* tuning env set, so defaults apply: target CTAs = 2 x 58 SMs, BLOCK_N 32, warps 4, stages 2 |
| SGLANG_SM120_ONLINE_MXFP8=1 | densefp8_patch | Stock switch, historical name |
| SGLANG_TOPK_LOGITS_EXCHANGE=32 | topklogits_patch | Turns itself off under speculative decoding (l.97-102) |
| SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION, SGLANG_INIT_NEW_TOKEN_RATIO, SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR, SGLANG_MM_PRECOMPUTE_HASH, SGLANG_VLM_CACHE_SIZE_MB, SGLANG_INPROJ_FP8, SGLANG_FP8_FUSED_DEQUANT | none of the 38 files | grep finds no hits. These are stock SGLang/Pennyroyal knobs, not implemented here |

**Patched but dormant with the rung-0 env as given:**
- marlinclean (needs SGLANG_MARLIN_EP_CLEAN).
- idxshard (needs SGLANG_QSA_INDEXER_SHARD).
- qsadump (needs SGLANG_QSA_DUMP_DIR).
- hisparse (a no-op placeholder).
- ple_fp8_scale (needs `--ple-scale` plus SGLANG_PLE_FP8_SCALE).

**Mandatory patches whose env is not in the rung-0 list** (presumably set elsewhere in the notebook):
- prune needs SGLANG_PRUNE_KEEP.
- think_force needs SGLANG_FORCED_THINK_SUFFIX.

**No env switch at all:** hicpairown, hicqsawait and hicmambafb are always on once applied. They only matter with `--enable-hierarchical-cache`.

**Inference (flagged, not stated in the files):** prune_patch (l.8) and draft_ep_patch show an MTP drafter that keeps 512 experts. If speculative decoding is on in the serve rung, `_tkx_logits_enabled` logs a warning and TOPK_LOGITS_EXCHANGE=32 is ignored.

---

## B. Model shapes visible in the files (Qwen3.8-Flash-Next, TP4)

- **Layers:** 48 total: 12 QSA (full-attention) layers, ids 3, 7, ..., 47 (bench_pairown l.38), and 36 GDN (mamba-type) layers (hicpairown l.6).
- **Attention heads:** Hq 24 and Hkv 2 overall. Per TP4 rank: 6 q heads, 1 KV head, head_dim 256. The KV head is replicated across rank pairs (0,1) and (2,3).
- **QSA indexer:** 4 heads x 128, padded to 8. Compress ratio 4. Block top-k 512, which is 2048 tokens. Top-k row width 2051 = 2048 + 4 - 1 (test_qsa l.8-11). Compressed page 16; KV page 64.
- **Page table:** 2048 pages = 131072 / 4 / 16, i.e. 32768 compressed positions.
- **Hidden / HyperConnection:** hidden 2560, hc_count 4, hc_lowrank 320, so K = 10240.
- **Vocab:** 248320, untied lm_head; 62080 rows per rank.
- **MoE:** 512 experts, top-10, pruned to 384 = 96 per EP4 rank. Bench uses K 2560, N 640, group 128.
- **Per-rank dense shapes:** GDN in-proj 4096x2560, GDN out 2560x1536, QSA qkv 3584x2560, QSA o 2560x1536, shared expert gate/up 320x2560, shared down 2560x160.
- **KV cache:** fp8 e4m3.

---

## C. Per-file reports (alphabetical)

### 1. apply_patches.py
1. **Purpose.** Idempotent patcher for the installed Pennyroyal SGLang `0.5.19+gd00d88efc8d6`. It always installs the sm_89 QSA sparse decode. With `--ple-scale` it also installs the scaled BF16->FP8 PLE cast. Each anchor must match exactly once, otherwise it refuses to patch ("a different SGLang build is refused instead of being half-patched", l.16-17). It deletes the matching `.pyc` files. `--check` only reports status.
2. **Edits.**
   - Copies `qsa_sm89_decode.py` to `sglang/srt/layers/attention/qsa/sm89_decode.py`.
   - In `layers/attention/qwen_sparse_attn_backend.py`, `_forward_paged_attention`: inserts an early return just before `flash_attn_varlen_func = _resolve_flash_attn_varlen_func()`. The return calls `qsa_sparse_decode(sm89_impl, q, k_buffer, v_buffer, req_to_token, row_req_pool_indices or req_pool_indices, topk_indices, sequence_lengths, layer.scaling)`. Marker: `# [duck2 sm89 patch]`.
   - With `--ple-scale`: copies `ple_fp8_scale.py` to `models/duck2_ple_fp8.py` and makes two edits in `models/qwen4_exp.py`. Edit 1: PLE rows are quantised with `quantize_ple_rows` when the target is fp8 and the source is not. Edit 2: `keep_quantised_scale` skips the checkpoint's placeholder `weight_scale`.
3. **Switch.** Runtime `SGLANG_QSA_SM89_DECODE=auto|triton|torch|off`; the PLE part uses `SGLANG_PLE_FP8_SCALE`. The decode path is a new kernel matched to the reference within tolerance, not bit-identical to anything. Stock cannot run here: FA2 is not in the wheelhouse and FA4 is rejected (comment l.31-32).
4. **Numbers.** None.

### 2. bench_densefp8.py (phase 0)
1. **Purpose.** Times the per-rank dense projections at decode. It compares weight-only rowwise FP8 (`sm89_rowwise_fp8_gemv`) against bf16 `F.linear` (cuBLAS) by replaying a CUDA graph over distinct weight copies, so L2 is cold. It sweeps 7 GEMV tile configs per shape and row bucket. It also times the HC fused mix with FP8 vs bf16 weights at 13, 19 and 32 rows, and tests the LinearMethod flow on a fake 2560x1536 layer (decode 19 rows, prefill 300 rows).
2. **Edits.** None. It loads `sm120_online_fp8.py` and `hc_mix_triton.py` by file path and sets `SGLANG_HC_MIX_MAX_ROWS=32` first.
3. **What it measures.** Rows 1, 8, 19, 32, 48 and 64; sweep rows 8, 19, 48 decide buckets {1,8,16}, {24,32}, {48,64}. Headline `baseline_us`/`patched_us` = dense projections x calls per step at 19 rows, plus HC mix x 97. Also reports `saved_ms_per_step` at 19 and 48 rows.
4. **Numbers and gates.**
   - SHAPES with calls/step (l.23-31): gdn_in_qkvz (4096,2560) x36, gdn_out (2560,1536) x36, qsa_qkv (3584,2560) x12, qsa_o (2560,1536) x12, shared_gu (320,2560) x48, shared_down (2560,160) x48, lm_head (62080,2560) x1.
   - Fail gates: kernel rel. error > 0.02 and FP8 quantisation rel. error > 0.06 (l.172-175); HC FP8 vs dequant reference > 0.02 (l.215); method flow > 0.06 (l.244).
   - HC step = per-mix us x 97 (l.214, 221).
   - No measured results.

### 3. bench_hcmix32.py (phase 0)
1. **Purpose.** Correctness and speed test for hcmix32 on one L4. It must run under a hard timeout, because a broken grid barrier can deadlock the persistent kernel. Exit codes: 2 = hang, 3 = watchdog.
2. **Edits.** None. It loads `hc_mix_triton.py` by path. In a fresh subprocess it checks dormancy: with no env, the tuple must read `16, 0, None`.
3. **Method.**
   - Baseline = 97 mixes in one CUDA graph through the stock fallback, i.e. the `torch.compile(_mix_compute)` chain. `ok` requires that the chain compiled ("an eager-chain baseline would overstate the saving", l.9-10, 488-490).
   - Patched = the same 97 mixes through the ROWS=32 kernel.
   - Also: a num_stages sweep over 3, 2, 1; a check that the barrier counters return to 0; an fp64 reference that mirrors the kernel's math.
   - Tolerance TOL_REL 2^-7, TOL_ABS 2^-9, BAD_FRAC 1e-5 (l.46).
   - Rows 0-15 must agree with the stock 16-row kernel, and the 32-row path must be faster than the chain at every row count above 16.
   - It recommends `SGLANG_HC_MIX_MAX_ROWS = max tested rows`, and NUM_STAGES only if more than 2 % faster (l.485-487).
4. **Numbers.** Shapes hc 4, hidden 2560, lowrank 320 (K 10240); 97 mixes per decode step; "1.27 GB total, so every mix streams from DRAM ... L2 is 48 MB" (l.19-21). Defaults: rows 16,17,19,20, prod rows 19, iters 20, reps 5, timeout 540 s, hang timeout 30 s. No measured results.

### 4. bench_hisparse.py (phase 0, 4 GPUs)
1. **Purpose.** Benchmarks the stock HiSparse swap-in kernel (`sglang.kernels.ops.kvcache.hisparse._jit_sparse_module(is_mla=False)`), instantiated for QSA K+V. Items are 4-token blocks; host KV is pinned and read over UVA on a miss. This is a feasibility study for offloading QSA KV to host.
   - Correctness on rank 0: compared with an exact Python LRU oracle. Checks hits, misses, evictions, LRU order, the newest block going to the extra slot, the fast path when nblk <= HOT, -1 padding beyond num_real_reqs, int32 and int64 index types, and the bytes of every selected slot.
   - On an illegal memory access it reruns diagnostic cases one per process.
2. **Edits.** None; it calls the stock kernel.
3. **Method.** Per decode step = 12 layers in a CUDA graph. `io_us` = full kernel with miss copies; `plan_us` = SkipIO (LRU planning only). Grid: rows {19,32,48,64} x churn {3,6,12,30} % of the 512 ids per step, solo vs all four GPUs at once (host-link contention). Headline: rows 32, churn 6 %, `patched_us = io_all4_us`; no gate.
4. **Numbers.** ITEM 1024 B (4 x 256 B fp8) for K and again for V; TOPK 512; HOT 1536 blocks per (layer, request) + 2 extra slots; MAX_BLK 34816; host 2^18 blocks. l.45: "the kernel needs BLOCK_SIZE >= NUM_TOP_K (v80 diag: 128 and 256 threads fault on the slow path)", hence THREADS = 512.

### 5. bench_idxshard.py (phase 0, 4 GPUs, NCCL)
1. **Purpose.** Compares stock selection (full `qsa_mqa_decode` + `fast_topk` 512) with the sharded selection over a real all-gather.
   - Per row the result must be valid-first and -1-padded, unique, inside [0, len), and the same size as stock. Its sorted stock logits must equal stock's sorted logits; equal-value ties may swap.
   - All four ranks must give identical ids.
   - Edge lengths include 0, 1, 15-17, 511-513, 2047-2049 and 32768.
2. **Edits.** None; it loads the patched `mqa.py`.
3. **Method.** Rows {19,32,48} x context {24k, 80k, 130k}, 12 layers. Reports stock_us, shard_compute_us (local stand-in for the gather), nccl_us (eager) and nccl_graph_us (12 gathers in one graph). `patched_us = shard_compute + 12 x min(nccl)`. Headline: rows 32, 80k (l.11-17, 329).
4. **Numbers.** Shapes as in section B; no results.

### 6. bench_marlinclean.py (phase 0, 1 GPU)
1. **Purpose.** Emulates EP4 on one L4: 384 random W4A16 experts (uint4b8, group 128), K 2560, N 640, top-10, four "ranks" of 96 experts. It uses the dispatcher's real local map and -1 ids. "Stock" is always the true unpatched source: the venv file, or `revert_text()` of an already-patched venv, verified by round trip.
2. **Edits.** None; it loads temporary copies of the module.
3. **Checks.** For M in 1, 13, 19, 20, 80 and 2048:
   - clean vs stock per rank;
   - the sum over the four ranks vs one stock call holding all 384 experts;
   - masked rows exactly 0 (token 0 routed only to rank 0, token 1 only to rank 3);
   - with every `torch.empty` NaN-filled;
   - CUDA-graph replay;
   - `moe_align_block_size` with 97 vs 385 buckets gives identical blocks;
   - routed_scaling_factor 2.5;
   - optional fp16;
   - real `marlin_quantize` ground truth (16 experts) vs an fp64 MoE: clean RMS error must be <= 1.10 x stock RMS error.
   - Dormancy: flag off must be bit-exact to stock for M <= 256.
4. **Numbers.** TOL = 2^-5, row-wise relative ("real bugs give >= 0.1", l.48). `est_step_saving_ms_48_layers = 48 x (base - clean)` at M 19 (l.557-558). No results.

### 7. bench_mqagrid.py (phase 0)
1. **Purpose.** Runs the shipped TileLang decode-MQA kernel with groups_per_cta G in {1,4,8,16} at production decode shapes.
   - Logits must be bit-identical to G=1 (`torch.equal`, including the -inf mask) at rows {13,19} x context {6k, 24k}, and on a batch of boundary lengths (0, 1, 15-17, 63-65, 64G-1..64G+1, 32767, 32768).
   - Rows are padded with zero-length rows up to the decode graph batch size (13->16, 19->20), as the server does.
   - Each check is re-run 10 times, and again after CUDA-graph replay over 12 layer caches.
   - G=1 must be deterministic and match `torch_qsa_mqa_decode`.
   - It also checks env parsing, dormancy, and the patched wrapper.
2. **Edits.** None (it also warms the TileLang disk cache).
3. **Method.** Times `fill_(-inf)` + kernel per call from a 12-call graph. It recommends a G>1 variant only if it is at least 3 % faster on average and nowhere more than 2 % slower (l.473-479).
4. **Numbers.** 8 kernel heads (4 real + 4 pad), head_dim 128, page 16, MAX_PAGES 2048, i.e. MAX_MODEL_LEN 32768 compressed positions; NUM_PAGES 12288 per layer (48 MiB bf16, more than L2). No results.

### 8. bench_pairown.py (phase 0, 4 GPUs)
1. **Purpose.** Tests only the pair-exchange plumbing, without the model. It brings up SGLang's TP4 parallel state, loads the patched `qsa/pairown.py`, and checks:
   - gather routing (slot i = pair member i);
   - `pair_attend` with a stand-in attention `q_wide*2+1`, eager and inside the CUDA graph capture context, replayed with changed inputs and compared bit-exactly.
   - Timing: 12-layer graph with exchanges vs a local copy, for "pair" (2-rank group) and "tp" modes; `exchange_us` is the difference.
   - `group_mem_mib` = memory cost of the communicator.
2. **Edits.** None.
3. **Headline.** `patched_us` = best mode's exchange_us at 32 rows.
4. **Numbers.** "q of one rank = 6 heads x 256 bf16 = 3 KB/row" (l.5); 12 layers = 24 pair all-gathers; rows 19,32,48,64; LAYER_IDS 3 + 4i (l.38). No results.

### 9. bench_pcie.py
1. **Purpose.** PCIe microbenchmark for the KV-offload question, one process per GPU, all GPUs starting together so it also measures root-complex contention. It measures bulk H2D and D2H GB/s on 256 MB pinned buffers, and the rate of scattered page-sized pinned->GPU copies (a lower bound for a gather kernel).
2. **Edits.** None.
3. **Method.** 60 repeats of the bulk copy; 512 random pages out of 640.
4. **Numbers.** Page = 390 KB = 64 tokens x 6.1 KB K+V for the 12 QSA layers (l.3-4). No results.

### 10. bench_qsasplitk.py (phase 0)
1. **Purpose.** Correctness and speed for split-K vs the duck2 Triton kernel.
   - Correctness vs the fp32 loop reference: rows {1,13,19,20,48} x {bf16, fp8}, topk 2051, Hq 6 / Hkv 1, D 256, with holes, out-of-range entries, a 3-token row and an empty row (must be exactly 0).
   - "Peaky" cases put the softmax maximum in different splits; topk 17 and 40; GQA 24/2 and 12/1; prefix layout.
   - The NSPLIT=1 (FINAL) path must be bitwise equal to triton.
   - CUDA-graph replay at 19 rows (fp8) must match eager bitwise.
   - Dormancy: `splitk_route` returns the same objects for every env value except `triton_splitk`.
2. **Edits.** None (modules loaded by path, or a temp copy of MODULE_SRC).
3. **Method.** Graph replay of at least 12 calls, enough to touch more than 4 x L2 of K/V. All 2051 entries valid at 8k context, random slots in a 2^20-slot pool. Sweep of block_n {32,64,16}, stages {2,1} and target CTAs {1,2,4,8} x SMs.
4. **Numbers and gates.** Tolerance 2e-2 x max|ref| + 1e-2. `gate_30pct_rows_le20` = minimum speedup over rows <= 20 is at least 1.30 (l.423-424). `step_saving_ms_12_layers = 12 x (triton - splitk)`. A suggested env is emitted only if under 0.97 x the default total (l.437). No results.

### 11. bench_topklogits.py (phase 0)
1. **Purpose.** Tests the exact injected source (PURE_SRC, GLUE_SRC and LP_HELPER are exec'd). With a site dir it also verifies the installed text matches. Single GPU with simulated world 4 (V 248320, W 62080), plus a padded layout whose pad columns hold junk logits (99). Checks:
   - the shard carrier is an exact fp32 cast;
   - greedy is bit-exact with `argmax`, including cross-shard and in-shard ties;
   - the allowed set and probabilities equal the stock rule ("joint" for flashinfer, "sorted" for pytorch); rows within 2e-6 of a cumulative mass are skipped;
   - a reported adversarial tie family;
   - distribution test: total variation under a 2.5x-noise threshold + 0.003, and zero draws outside the allowed set; this also confirms flashinfer implements "joint";
   - glue on every simulated rank: every ineligibility reason falls back with bit-identical stock logits, and prefill is untouched.
   - Multi-GPU with real NCCL: fallback bit-exact to the stock all-gather, ids identical on all ranks, CUDA RNG state identical afterwards (lockstep), plus timing.
2. **Edits.** None.
3. **Method.** GPU time behind a ~4 ms `torch.cuda._sleep`, so CPU launch latency is hidden. Stock = all-gather + movedim + slice + fp32 copy (eager, and in a graph) + stock sampling. Patched = carrier + exchange-sample.
4. **Numbers and gates.** weights error < 5e-5, lse < 1e-4, patched error vs fp64 <= stock + 1e-6 (l.870); defaults K 32,64; B 1,13,19,20; 40000 samples. No results.

### 12. densefp8_patch.py
1. **Purpose.** Opens Pennyroyal's online FP8 dense path to SM89. Weight-only rowwise e4m3 with a per-output-row fp32 scale (absmax/448), bf16 activations, fp32 accumulate, bf16 output. Rationale: decode is HBM-bound.
2. **Edits.**
   - `kernels/ops/gemm/sm120_online_fp8.py`: `capability not in ((12,0),(8,9))`.
   - Same file: `_MAX_KERNEL_ROWS 32 -> 64`.
   - Same file, appended: `_sm89_w8a16_kernel` (Triton; BLOCK_M = next pow2 of rows, at least 16), `sm89_rowwise_fp8_gemv` with tuned table `_SM89_CFG`, keyed (N, K, rows bucket in 1/8/16/24/32/48/64) -> (BLOCK_N, BLOCK_K, stages, warps), default (32,128,4,4). `sm89_rowwise_fp8_linear`: up to 64 rows uses the GEMV; above that it dequantises in chunks and does a bf16 matmul (prefill). `build_sm89_rowwise_method()`: an UnquantizedLinearMethod subclass that quantises in `process_weights_after_loading` and calls `empty_cache` "so KV sizing sees it".
   - `models/qwen4_exp.py` `_maybe_convert_linears_to_mxfp8.build_method`: on (8,9) returns the SM89 rowwise method instead of MXFP8.
   - Eligible linears are the stock set: BF16, >=128x128, K%32==0, not "gate" routers, not FusedMoE, not PLE.
3. **Switch.** `SGLANG_SM120_ONLINE_MXFP8=1`. Numerics-changing: FP8 weights for the HC mix, lm_head, GDN/QSA projections and the shared expert. HC on 17-32 rows needs hcmix32, which accepts the FP8 pair.
4. **Numbers.** l.5-6: "~3.2 GB/step of bf16 dense weights per GPU (HC mix 1.27, GDN proj 1.04, lm_head 0.32, QSA proj 0.30, shared expert 0.28)". l.33: config "tuned on L4 by bench_densefp8.py (v76)". The CFG literal at l.28 covers shapes (320,2560), (2560,160), (2560,1536), (3584,2560), (4096,2560), (62080,2560).

### 13. draft_ep_patch.py (mandatory)
1. **Purpose.** Lets an MTP draft MoE layer with MORE experts (512) than the pruned target (384) load under expert parallelism: `logical_to_all_physical()` returns `[logical_expert_id]` (identity) for ids beyond the target-sized map. The map is used only at weight-load time; there is no EPLB (`ExpertLocationDispatchInfo.init_new` returns None), so routed ids are not remapped at runtime (l.4-5).
2. **Edits.** `eplb/expert_location.py`, after `cpu_map = self.logical_to_all_physical_map_cpu`: `if logical_expert_id >= cpu_map.shape[1]: return [logical_expert_id]`.
3. **Switch.** None. Load-path fix, no numerics effect.
4. **Numbers.** None.

### 14. embedhost_patch.py
1. **Purpose.** Moves the input embedding table into pinned host memory to free GPU memory for KV pool. It reuses the model's own `Qwen4ExpPinnedHostEmbedding` (built for the PLE n-gram table, qwen4_exp.py:801): a meta-device VocabParallelEmbedding template, a page-locked host table loaded by the stock vocab-parallel loader, and a Triton UVA gather that zero-fills ids outside the rank's shard, followed by the same TP all-reduce. The lm_head (untied) stays on the GPU.
2. **Edits.** `models/qwen4_exp.py` `_build_embed_tokens`: if the env is set and `tie_word_embeddings` is false, build the meta template with `params_dtype=bf16`, register a non-persistent `weight_scale = ones(1)` buffer, and return `Qwen4ExpPinnedHostEmbedding(template)`; logs MiB per rank.
3. **Switch.** `SGLANG_EMBED_HOST=1`. Exact: "the gathered bf16 rows are copies and the all-reduce adds one non-zero row to zeros" (l.11-12).
4. **Numbers.**
   - l.3: vocab 248320 x 2560 bf16, 318 MB per TP4 rank.
   - l.6: "v86: 28-34 running at 88% KV use".
   - l.7: "318 MB/GPU is ~98k more pool tokens (+7%) at 3.24 KB/token/GPU (fp8 KV, pair-owned, + indexer keys)".
   - l.12-13: cost "B rows x 5 KB over PCIe per decode step (~200 KB at B=40), ~5 MB per 1024-token prefill chunk".

### 15. hcmix32_patch.py
1. **Purpose.** Extends the persistent fused HyperConnection-mix Triton kernel from 16 to 17-32 rows (and above, in 32-row chunks). Production decode runs 17-19 live rows on the 20-row CUDA graph, so all 97 mixes per step were falling back to the 5-kernel torch.compile chain (down GEMM + splitK reduce, silu, up GEMM, sigmoid-mul-mean). The kernel is already generic in its ROWS constexpr. The grid stays one CTA per SM (58), so the software grid barrier keeps co-residency.
2. **Edits** in `layers/hc_mix_triton.py`, applied by position on the stock text:
   - After `_FUSED_MIX_MAX_ROWS = 16`: env parse `_hcmix32_env`, `_hcmix32_rows_ok` (same-dtype bf16/fp16 pair or an FP8 pair from densefp8), and `_hcmix32_launch`. The launch tries num_stages = pinned value, then 3, 2, 1. If none compiles it logs, computes that one call eagerly, and sets MAX_ROWS = 0 so later mixes use stock.
   - `fused_hc_mix_supported`: OR in `_hcmix32_rows_ok`.
   - `fused_hc_mix`: rows > 32 are concatenated from 32-row recursive calls.
   - `rows_pad = 32` for 16 < rows <= 32; `rows_pad == 32` dispatches to `_hcmix32_launch`.
   - Rows <= 16 use the untouched stock launch.
3. **Switch.** `SGLANG_HC_MIX_MAX_ROWS=N` (clamped to 64), optional `SGLANG_HC_MIX_NUM_STAGES` (1-4). Numerics: the same fp32 atomic path that <= 16 rows already use; the summation order varies per replay. Versus the bf16 torch chain this is "a change of rounding only".
4. **Numbers.** 17-19 rows on the 20-row graph, 97 mixes (l.6-7); 58 SMs (l.10); "3 stages x (32x256 x tile + 32x256 w tile) bf16 = 96 KiB smem is at the sm_89 99 KiB limit" (l.18).

### 16. hic_check.py (HiCache test tool, stdlib only, about 5 min)
1. **Purpose.** Forces host->device reloads early and checks them exactly against a running server.
   - A = cold prefill and B = GPU hit for T random-token target prompts.
   - Then a flood of prompts totalling more than 1.3x the GPU KV pool evicts every target to host.
   - C = each target again: its prefix must come from host (`cached_tokens_details.host`), and its greedy text must equal B's when the reused prefix length is the same.
   - Then W stress workers re-hit targets during decode.
   - ok = no failures, no strict B/C mismatch, and at least one host reload seen.
2. **Edits.** None. Uses `/generate` and `/metrics`.
3. **Switch.** Not applicable.
4. **Numbers.** l.4: "v99 (10-15) crashed on the first host->device load (pair-owned QSA left the layer-done counter at 42 of 48 layers ...)". l.24: L 16384 tokens, N 24 new tokens, T 6 targets, W 8 workers. F = 1.3 x KV / L + 2. Default 6 minutes.

### 17. hicmambafb_patch.py
1. **Purpose.** HiCache x mamba fix (2026-10-16). `match_prefix` stages the mamba state from the deepest match N2. A failed load-back returns the shallower device anchor N1 with N1's KV, so the recurrent state no longer matches the prefix. Result: zero state, or N2's state on N1's prefix, and the 36 GDN layers "silently diverge for the rest of the session". Load-back fails when FULL KV is short after eviction, or when the node-restore mamba allocation fails under slot pressure. Stock H100 TP1 never hits this.
2. **Edits.** `mem_cache/unified_radix_cache.py` `init_load_back`, on the failure return: if `best_match_node_id != last_best_match_device_node_id` and any component is MAMBA, then
   - set `prefix_indices` to empty, `mamba_cow_src_index` to None and `mamba_needs_clear` to True;
   - zero host_hit / swa_host_hit / mamba_host_hit lengths;
   - return the empty device indices and `root_node_handle(extra_key)`, i.e. a full re-prefill.
   - Exceptions keep the stock anchor. Logs `[hicmambafb]` for the first 20, then every 100th.
3. **Switch.** None. Correctness fix; slower but rare.
4. **Numbers.** l.9: "41 running x 2 slots on 96".

### 18. hicpairown_patch.py
1. **Purpose.** Makes the KV + MAMBA (+ QSA compressed) host tier work with pair-owned QSA. The builder set the per-layer transfer count to the number of layer ids stored on this rank: 36 + 6 = 42, while layer ids run 0..47. The H->D loop therefore never reloaded layers 42-47, and the per-layer wait indexed past `load_events`.
2. **Edits.** `mem_cache/hybrid_cache/hybrid_pool_assembler.py`: adds `_hic_layer_span(*mappings)` = `max(len(ids), max(ids)+1)` over stored ids plus every pairown owner id. It replaces both `transfer_layer_num = len(full_layer_mapping | mamba_layer_mapping)` sites in `build_hybrid_mamba_stack`.
3. **Switch.** None (reads pairown cfg). Exact; layers this rank does not store map to None and are skipped.
4. **Numbers.** l.6-10: "36 + 6 = 42 while layer ids run 0..47"; "(v99 smoke, 15:23 UTC 10-15: IndexError in mamba2_layer_index -> wait_until -> load_events[layer_index], scheduler killed, 36 requests lost)"; l.11: "H100 A/B: 48 either way".

### 19. hicqsawait_patch.py
1. **Purpose.** HiCache + QSA ordering fix (2026-10-16). The QSA indexer runs BEFORE attention and reads restored compressed keys through `get_qsa_compressed_k_buffer`, which had no per-layer H2D wait. Top-k could therefore read pages the H2D stream had not written yet: "silent wrong top-k, no error".
2. **Edits.** `mem_cache/qsa_kv_pool.py` `get_qsa_compressed_k_buffer`: calls `self._wait_for_layer(layer_id)` first. It refuses if the pool overrides `_wait_for_layer`. No-op unless a load-back is in flight; host transfers use the buffer pool directly.
3. **Switch.** None. Correctness fix.
4. **Numbers.** None.

### 20. hisparse_patch.py
1. **Purpose.** Placeholder for HiSparse->QSA offload (`research/HISPARSE_QSA_FEASIBILITY.md`). It patches nothing and exits 0, so the plumbing treats the key as applied.
2. **Edits.** None.
3. **Switch.** None.
4. **Numbers.** None.

### 21. idxshard_patch.py
1. **Purpose.** The QSA indexer is a ReplicatedLinear, so every TP4 GPU scans every compressed key, 12 times per step. This patch shards the decode scan by sequence: rank r scans only the pages whose ordinal o satisfies o % W == r.
   - Implementation: a strided view `page_table[:, r::W]` with the local length computed per row; local `fast_topk(512)`; local positions mapped back to global block ids with `((j//16*W + r)*16 + j%16)`.
   - It all-gathers `[rows, 2*512]` (values + int32 ids bit-cast to fp32) and merges with `torch.topk(512)`.
   - Exact set: the global top-512 is a subset of the union of the local top-512s; only tie order at the cut-off can differ. Non-finite entries become -1 (valid first). Pool memory is unchanged.
2. **Edits.**
   - `layers/attention/qsa/mqa.py`: appends `_IDXSHARD_ENV`, `idxshard_group()` (needs TP > 1 and attention TP == TP), `qsa_mqa_topk_sharded`, `qsa_mqa_topk_sharded_tp`.
   - `qsa_indexer.py`: in the decode selection (before `logits = qsa_mqa_decode(`), when `block_topk == 512`, the tensor is on CUDA and the group exists, it returns `expand_qsa_block_indices(sharded ids, ...)`. Decode, verify and draft-extend only; prefill untouched.
3. **Switch.** `SGLANG_QSA_INDEXER_SHARD=1` (not in rung 0). Exact.
4. **Numbers.** l.7-8: "~0.77 KB of keys per context token, 0.416 ms per 100k live KV tokens at ~185 GB/s"; "step-time fit step_ms = 26.97 + 1.092 n + 0.416 KV/100k"; "(10-16 ms at n 32-50 with 80k live ctx)". l.18: all-gather "rows x 4 KB per rank".

### 22. mambafinal_patch.py
1. **Purpose.** Lets the stock "sparse-prefill v2" policy (`mamba_prefill_final_only`: no per-chunk mamba checkpoint, only cold start + prefill-end state) also run under `extra_buffer_lazy`.
   - Under lazy mode every 1024-token chunk called `cache_unfinished_req(chunked=True)`. That allocates a fresh mamba slot and donates the tracked one.
   - Each allocation LRU-evicts an idle session's leaf, dropping its KV with it. That session re-prefills from scratch on its next turn, and the cascade feeds itself.
   - Lazy-safe: during prefill, lazy mode tracks into one write-only slot.
2. **Edits.** `mem_cache/unified_radix_cache.py`: inside `self.mamba_prefill_final_only = (...)`, the `and not params.enable_mamba_extra_buffer_lazy` clause becomes `(not lazy or env == "1")`.
3. **Switch.** `SGLANG_MAMBA_PREFILL_FINAL_ONLY_LAZY=1`. Exact; cache policy only.
4. **Numbers.** l.7-10: "An 85k-token pin re-prefill thus allocates ~83 slots out of a pool of 96 (36 running x 2 = 72 held)"; "(v92 smoke: 19 mid-session misses, 286k re-prefill tokens)".

### 23. mambaratio_patch.py
1. **Purpose.** Overrides the mamba slots budgeted per running request. `_calculate_mamba_ratio()` caps `max_running_requests` at `max_mamba_cache_size // ratio`. The stock ratio under lazy + `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK` is 3. A ratio of 2 frees slots for KV. It is safe because admission is still gated on real free + evictable slots, and a failed transient allocation degrades tracking in place.
2. **Edits.** `mem_cache/kv_cache_configurator.py` `_calculate_mamba_ratio`: `return base + additional_ratio` becomes "return the env value if > 0" (logged).
3. **Switch.** `SGLANG_MAMBA_SLOTS_PER_REQ=2`. Exact.
4. **Numbers.** l.6-10: stock ratio 3 "(base 3 - 1 skip-lock drop + 1 lazy ping-pong), so MAXREQ 40 needs MAMBA 120 = 1.66 GB/GPU of GDN state (13.8 MB per slot on 4x L4). Under the real v86 agent load (serve.log, 1852 decode lines) the pool held 2.08 slots per running request at 32-35 running (max 74 of 144, usage <= 0.51) ... 64-token track boundaries. With ratio 2, MAMBA 96 keeps MAXREQ 40 (96 // 2 = 48) and frees 24 slots = 0.33 GB/GPU = ~100k KV tokens."

### 24. marlin_ep_patch.py (mandatory)
1. **Purpose.** Under `--ep-size > 1` the dispatcher rewrites topk_ids to LOCAL ids with -1 for other ranks' experts. `GPTQMarlinMoEKernel.apply` never passed `expert_map`, so Marlin ran with `is_ep=False` and indexed expert -1: garbage, then NaN. This mirrors what the compressed-tensors WNA16 scheme already does.
2. **Edits.** `hardware_backend/gpu/quantization/gptq_kernels.py`: after the `quant_info` constructor, if `layer.dispatcher.local_expert_mapping` exists, set `quant_info.expert_map` to it and `quant_info.global_num_experts = moe_runner_config.num_experts`.
3. **Switch.** None. Correctness fix.
4. **Numbers.** l.6: "(v59 tensor dump on 4xL4 tp4/ep4: layer 0 experts finite, layer 1 experts all-NaN, every output token = '!')".

### 25. marlinclean_patch.py
1. **Purpose.** An EP-only cleanup of the GPTQ-Marlin MoE glue. It is taken only when `expert_map` is set (from marlin_ep_patch) and the weights are not MXFP4/NVFP4. Changes:
   - (1) `moe_align_block_size` uses E+1 = 97 buckets instead of global+1 = 385.
   - (2) `intermediate_cache13` uses `torch.empty` instead of `torch.zeros`.
   - (3) The `intermediate_cache3.zero_()` before GEMM2 is dropped.
   - (4) GEMM2 has `mul_topk_weights=False`; the final reduce is the masked `moe_fused_mul_sum(is_ep=True, expert_map=None)` in fp32 instead of `moe_sum_reduce` over zeroed rows. "expert_map is deliberately NOT passed ... it is the GLOBAL->local map, topk_ids are local."
   - It provides `apply_text` and `revert_text` (exact inverse) for the bench.
2. **Edits.** `layers/moe/fused_moe_triton/fused_marlin_moe.py`: six edits plus header. Read-only anchors are checked in `kernels/ops/moe/moe_fused_mul_sum.py`.
3. **Switch.** `SGLANG_MARLIN_EP_CLEAN=1` (not in rung 0, so dormant). Numerics: the weight multiply moves from bf16 to fp32, "output differs by bf16 rounding only".
4. **Numbers.** l.9: "E = local experts = w1.shape[0], 96 at 384/EP4 ... instead of global_num_experts + 1 (385)". Line refs inside the file: fused_marlin_moe.py:290, :365-366, :423-427; standard.py:227.

### 26. mqagrid_patch.py
1. **Purpose.** The TileLang decode-MQA kernel already supports `groups_per_cta`, but the wrapper never passed it. Under CUDA graphs the grid is (rows, ceil(max_pages / sub_pages)) with max_pages at the static graph width, so most CTAs launch, copy Q to shared memory and exit. With G > 1 one CTA scores G consecutive 64-key groups, and the grid shrinks by G. Logits are expected bit-identical. A G>1 kernel that fails to COMPILE falls back to the stock grid; runtime faults are not caught. The `torch.full(-inf)` init is kept.
2. **Edits.** `layers/attention/qsa/mqa.py`: inserts `_mqagrid_env_kwargs` / `_MQAGRID_KW` / `_mqagrid_decode_kernel` before `def tilelang_qsa_mqa_decode(`, and routes the kernel factory call (mqa.py:372-374) through the getter.
3. **Switch.** `SGLANG_QSA_MQA_GROUPS_PER_CTA=4`, optional `SGLANG_QSA_MQA_NUM_STAGES` (overrides 3). Expected bit-identical.
4. **Numbers.** l.7-10: "CTX 131072 / ratio 4 / compressed page 16 = 2048 pages, sub_pages 4 -> 512 CTAs per row. At 6k-24k context only 24-96 of those 512 groups hold keys; the other 80-95% of CTAs still launch ... 12 times per decode step."

### 27. pairown_patch.py
1. **Purpose.** At TP4 with 2 KV heads, both ranks of a pair hold byte-identical K/V and byte-identical compressed index keys. This stores each QSA layer once per pair: QSA layer ordinal j is owned by pair rank j % 2.
   - Pool: only owned layer ids go to QSATokenToKVPool (6 of 12 layers), so capacity is about 2x.
   - Layer: every rank still runs the qkv projection. Exchange 1 (pair all-gather) sends each rank's 6 q heads to the owner. The owner runs the indexer + sparse attention for 12 q heads against one KV head, using a shim copy of `attn` with `tp_q_head_num x 2`, and writes the new K/V.
   - Exchange 2 returns the partner's raw output. Each rank applies its own sigmoid gate and o_proj (+ the usual TP all-reduce).
   - The non-owner skips the indexer, the KV write and the attention. The owner may overlap the indexer on an alt stream during capture.
   - The gather uses `pynccl_comm.change_state(enable=True)` under capture.
2. **Edits.**
   - New `layers/attention/qsa/pairown.py` (configure, owned_layer_ids, ensure_group creating the "qsa_pair" group via `init_model_parallel_group`, prepare, Plan, layer_plan, gather, pair_attend).
   - `mem_cache/kv_cache_configurator.py`: `prepare()` before profiling; filters `full_attention_layer_ids` to owned ones (not on the draft worker).
   - `model_executor/pool_configurator.py`: cell size uses only owned layers.
   - `models/qwen4_exp.py`: `_self_attention_impl` dispatches to the new `_self_attention_pairown` when a plan exists.
3. **Switch.** `SGLANG_QSA_PAIR_OWN=1` (needs TP = 2 x KV heads and attention TP == TP), `SGLANG_QSA_PAIR_OWN_GROUP=pair|tp`. "exact in math".
4. **Numbers.** l.6-8: "6.1 KB/token/GPU over the 12 QSA layers ... 0.77 KB/token/GPU: the indexer is a ReplicatedLinear ... (n ~ 20 at ~50k live ctx), so storing every layer once per pair roughly doubles it". l.19: "2 small pair all-gathers per QSA layer (24 per step) ~ 0.05 ms each at decode shapes".

### 28. ple_fp8_scale.py (installed as `models/duck2_ple_fp8.py` with `--ple-scale`)
1. **Purpose.** For BF16-PLE checkpoints (Intel / woochangsim W4A16) served with `ple_embedding_dtype=float8_e4m3fn`, the stock raw `.to(fp8)` cast does not saturate: |w| > 448 becomes NaN and tiny values become coarse subnormals. With the env set, it quantises `clamp(w/s, ±448)` in 8192-row chunks, sets `weight_scale = s` (rounded to bf16), and rejects a checkpoint that already carries a real scale.
2. **Edits.** Applied via apply_patches PLE edits (qwen4_exp.py:2094-2109 and 2160-2172 stock paths; scale applied after the PLE all-reduce at :569-570, 1256-1257).
3. **Switch.** `SGLANG_PLE_FP8_SCALE=<s>`, which must be identical on all ranks. The notebook derives `s = 2*amax(shard_0)/448`. Numerics-changing (better than the raw cast). Reference: the official RadixArk FP8 PLE dataset `sonphamorg/arc3-flashnext-ple-fp8-v1` uses one global per-table scale.
4. **Numbers.** FP8 max 448; 2x margin; chunk 8192 rows.

### 29. prefill_bench.py
1. **Purpose.** Measures prefill cost on an idle running server.
   - Time to first token for N new tokens (max_new_tokens 1, greedy, `/flush_cache` before each request, slices of tokenized ARC text). The slope gives the per-token cost, the intercept the fixed cost per forward.
   - One torch-profiler capture of a multi-chunk prefill, reduced per rank to busy/idle time, prof_summarize categories, top kernels, and the host side (cuda_runtime / cpu_op).
2. **Edits.** None (uses prof_drive and prof_summarize).
3. **Method.** Sizes 64-16384, 2 reps; profile 5120 tokens, 5 steps.
4. **Numbers.** l.2-3: "v86: 1024-token chunks run at ~2.3k tok/s = ~0.44 s each, and every decode request waits for them, ~8% of a harness run's wall time".

### 30. prof_drive.py
1. **Purpose.** Closed-loop decode load generator plus optional profiling. It keeps exactly B long ARC-style requests decoding: ignore_eos, T 1.0 / top_p 0.95 / top_k 20, chat template with `enable_thinking=True, reasoning_effort="xhigh"`. It walks the --Bs list, aborting the newest requests when stepping down and replacing requests that end early.
   - Per B: settle, then an unprofiled window. It parses TP0 "Decode batch" lines from serve.log: ms/step = 1000 x running / gen-throughput, using only clean lines (previous logged event was a decode line) with running == B. It also records #full token, occupancy, retractions and a client-side token-rate cross-check.
   - Optional `/start_profile` per B, then prof_summarize. Requires max_running_requests >= max(Bs).
2. **Edits.** None.
3. **Method.** Bs default 20,19,18,17,16,15,14,13,12,8,1; ctx 8192; 45 s windows; `step_ms_ctx_adj` corrects for KV growth.
4. **Numbers.** l.22-23: "Context grows ~19 tok/s per request during the run". l.310: `--ctx-ms-per-100k` default 0.51, "memo sec. 0: step cost per 100k KV tokens". max_new 30000.

### 31. prof_summarize.py
1. **Purpose.** Reduces per-rank torch-profiler traces to per-decode-step GPU time by category (moe, dense_gemm, nccl, attn_qsa, gdn, hc_mix, sampling, other, idle) with detail buckets, top kernels, NCCL calls per step, and a cross-rank split of NCCL time (pure = min over ranks per collective; wait = excess, i.e. EP imbalance).
   - Steps are split on CUDA-graph replays (kernels sharing a cudaGraphLaunch correlation id) or on AllGather ends.
   - Exclusive attribution: overlapping streams share time evenly.
   - HC heuristic: inductor kernels plus preceding dense GEMMs = the HC fallback chain.
   - `--cg-selftest` checks the assumption on the installed torch/CUPTI.
2. **Edits.** None.
3. **Method.** Steps longer than 1.5x the median are dropped.
4. **Numbers.** Usage example `--baseline-step-ms 53.5` (l.26), which is likely a real earlier step time. Default min graph kernels 50.

### 32. prune_patch.py (mandatory)
1. **Purpose.** Load-time expert pruning of the TARGET model only (the MTP drafter has its own loader). Env `SGLANG_PRUNE_KEEP=/path/keep_N.json` with `{"num_experts": N, "layers": [[sorted kept original expert ids] x 48]}`.
   - The weight iterator is wrapped. Per layer L, kept expert E is renamed to index `keep[L].index(E)` (dict `remap[L] = {e: i}`) via regex `^(model\.(?:language_model\.)?layers\.(\d+)\.mlp\.experts\.)(\d+)(\..+)$`. Pruned experts' tensors are skipped (`continue`).
   - The router `layers.L.mlp.gate.weight [512, H]` is `index_select(0, kept ids)`, so rows are kept in the same order and the router outputs N logits mapping 1:1 to the renamed experts.
   - Logs renamed / skipped / sliced counts.
   - The server must also build the target with `--json-model-override-args '{"text_config":{"num_experts":N}}'`. With MTP on it also needs `--speculative-draft-model-override-args '{}'`, so the drafter keeps its 512 experts. That is why draft_ep_patch exists.
   - **How the keep set is chosen is not in these files.**
2. **Edits.** `models/qwen4_exp.py`: inserts `weights = _prune_expert_weights(weights)` before `num_experts = getattr(self.config, "num_experts", None)` in `load_weights`, and appends the helper.
3. **Switch.** `SGLANG_PRUNE_KEEP`; no-op when unset. Numerics-changing (removes experts; top-k routing renormalises over the remaining ones).
4. **Numbers.** 512-row router, 48 layers. Target N = 384 per the brief and per the marlinclean / bench_marlinclean "384 / EP4 = 96".

### 33. qsa_sm89_decode.py (installed as `qsa/sm89_decode.py`)
1. **Purpose.** QSA sparse decode / verify / draft-extend attention for GPUs without trtllm-gen (SM100/120 only) or a usable FA kernel.
   - The stock fallback packs the top-k K/V into scratch and calls FA2 (not shipped) or FA4-CuTe. FA4-CuTe asserts equal q/k/v dtypes and "FP8 is only supported on SM100", and its hd256 SM80 tile needs 128 KB of shared memory, more than sm_89's 99 KB.
   - QSA layers are always routed to this backend, so no flag avoids the path.
   - **Algorithm (Triton `_qsa_sparse_decode_kernel`):**
     - grid = (query rows, KV heads), static, CUDA-graph safe, no scratch copy, no host sync;
     - each program loads the GROUP q heads (BLOCK_H >= 16) and folds `scale * log2(e)` into q;
     - it walks the row's top-k logical positions in BLOCK_N = 32 chunks. Each position is valid iff `0 <= p < seq_len`; it is mapped through `req_to_token[req, p]` to a physical pool slot, and the K row (256 contiguous elements, fp8 upcast without a scale, bf16 MMA) is loaded;
     - `s = q·kᵀ`; online softmax in exp2 with -inf-safe max; `acc = acc*alpha + p·V`; output `acc / l`;
     - rows with no valid entry output 0.
   - Semantics follow `qsa_sparse_attention_reference`: every valid selected entry counts, even after a -1 hole. The stock packing assumes valid entries form a prefix and would drop entries otherwise.
   - Also contains a vectorised torch fallback (gather + einsum + masked softmax, `nan_to_num`) and a loop reference. `select_sm89_decode_impl`: auto gives triton when compute capability < 9; also accepts triton, torch, off and the older name `SGLANG_QSA_TRITON_DECODE`.
2. **Edits.** Itself installed; call site in apply_patches.
3. **Switch.** `SGLANG_QSA_SM89_DECODE`. Within tolerance of the reference, not bit-identical (bf16 MMA).
4. **Numbers.** l.17-19: "128x64 hd256 tile = 128 KB of shared memory ... exceeds sm_89's 99 KB per block". l.229-230: torch fallback transient memory "~200 MB at 48 rows, topk 2051, D 256, Hkv 1". Source line refs: backend :1869-1961, :53-73, :1926-1948, :76-110; memory_pool.py:2440-2455; attention_registry.py:459-467; kernel.py:491-515; sparse_attn.py:396.

### 34. qsadump_patch.py
1. **Purpose.** Offline instrumentation for the HiSparse study (`research/HISPARSE_QSA_FEASIBILITY.md` §6.5). It dumps per-layer DECODE block top-k (the 512 block ids from `fast_topk`, before `expand_qsa_block_indices` turns them into 2051 token positions) to npz parts. This gives real step-to-step overlap for pricing an LRU hot buffer.
   - Only plain decode (`spec_info is None`) outside CUDA-graph capture/metadata, on the configured ranks. A self-check verifies the parked blocks equal the expanded token ids / 4.
   - Exceptions are caught; it disables itself after 3. Includes `--selftest`.
2. **Edits.**
   - New `qsa/qsa_dump.py`.
   - `qsa_indexer.py`: parks `(block_indices, compressed_lengths)` in `self._qsadump_stash` before `expand_qsa_block_indices`.
   - `models/qwen4_exp.py` `_compute_qsa_topk_indices`: calls the hook before `return topk_indices` (the MTP-reuse early return is not hooked).
3. **Switch.** `SGLANG_QSA_DUMP_DIR`, plus `_ROWS` (default 4), `_FLUSH_STEPS` (64), `_MAX_MB` (400), `_RANKS` ("0"). Requires `--disable-decode-cuda-graph` (it syncs per layer). Bit-identical outputs, but slow.
4. **Numbers.** Defaults only.

### 35. qsasplitk_patch.py
1. **Purpose.** At TP4 there is one KV head per rank, so the duck2 kernel launches one CTA per decode row: 13-20 rows use 13-20 of 58 SMs, and each CTA walks 2051 entries in 65 dependent iterations (latency-bound).
   - Split-K cuts the top-k axis into NSPLIT contiguous chunks of whole blocks: grid (rows, kv heads, NSPLIT). Each split writes a partial state per q head in fp32: m (log2 units), l, and the unnormalised acc.
   - A merge kernel, grid (rows, q heads), combines them: `out = Σ 2^(m_s-M) acc_s / Σ 2^(m_s-M) l_s` with M = max_s m_s.
   - NSPLIT = floor(target_ctas / (rows·kvh)) with target = 2 x SMs, clamped to [1, min(64, n_blocks)] and rebalanced so no split is empty. It depends only on static shapes, so it is CUDA-graph fixed.
   - When NSPLIT = 1 (MTP verify, large batches) the first kernel stores directly: the duck2 computation, with no merge.
   - Only the grouping of the fp32 sums changes; empty rows give exactly 0.
   - `sm89_decode.py` itself is untouched, so re-running apply_patches does not undo it.
2. **Edits.** New `qsa/sm89_decode_splitk.py`. `qwen_sparse_attn_backend.py` call site (requires the duck2 marker) gets `sm89_impl, qsa_sparse_decode = splitk_route(sm89_impl, qsa_sparse_decode)`.
3. **Switch.** `SGLANG_QSA_SM89_DECODE=triton_splitk`; tuning `SGLANG_QSA_SPLITK_CTAS`, `_NSPLIT`, `_BLOCK_N` (16-128 pow2, default 32), `_WARPS` (default 4), `_STAGES` (default 2). Changes rounding order only.
4. **Numbers.** l.6-7 and module l.56-59: 13-20 rows, 58 SMs, 2051 entries, 65 iterations of BLOCK_N = 32; MAX_SPLIT 64; merge chunk 8.

### 36. test_qsa_sm89_decode.py
1. **Purpose.** Unit tests without SGLang.
   - Torch fallback vs reference (bf16 and fp8).
   - The reference equals the stock packed semantics under prefix layout.
   - Triton vs reference for rows/topk (1,2051), (7,2051), (48,2051), (5,17), plus GQA 24/2 and 12/1.
   - CUDA graph replay after input mutation is bit-identical to eager.
   - Timing: Triton vs torch "us per QSA layer call at 48 rows x 2051 (x12 layers -> ms/step)".
   - PLE quantiser checks.
2. **Edits.** None.
3. **Pass criteria.** `max|out-ref| <= 2e-2*max|ref| + 1e-2`; empty rows exactly 0.
4. **Numbers.** l.8-11: TP4 shapes Hq 24/4 = 6, Hkv 1, D 256, topk 2051 = 2048 + 4 - 1 (qsa_indexer.py:559). l.155-176: PLE test with scale 0.004, outlier 5.0 > 448·s = 1.79 must saturate, relative error < 0.04. No timing results quoted.

### 37. think_force_patch.py (mandatory)
1. **Purpose.** Patches Pennyroyal's strict-thinking grammar. When a request's thinking budget runs out (`custom_params.thinking_budget` with `--enable-strict-thinking`; the global `SGLANG_MAX_THINK_TOKENS` is read in the backend), the forced `</think>` is followed by a forced `SGLANG_FORCED_THINK_SUFFIX` (e.g. `"\n\n[["`, which starts the answer grid), token by token, before free generation resumes.
   - Without it, a forced `</think>` mid-thought often led to more reasoning in the answer (cut by max_tokens) or a bare "To solve" + EOS.
   - Natural, unforced think ends are untouched. Empty or unset env = stock.
2. **Edits** in `constrained/reasoner_grammar_backend.py`:
   - module dict `_FORCED_SUFFIX = {"ids": ()}`;
   - `self._forced_end = False` in `__init__`;
   - `_forced_end = True` in the budget-exhausted branch that masks to the think-end token;
   - in `fill_vocab_mask`: if generating and `_forced_end` and `0 <= tokens_after_end < len(sfx)`, allow only `sfx[tokens_after_end]` and return (before any user grammar);
   - copies `_forced_end` in the copy method;
   - in the backend `__init__`, after `self.max_think_tokens = envs.SGLANG_MAX_THINK_TOKENS.get()`: reads the env, `unicode_escape`-decodes it, tokenizes with `add_special_tokens=False`, and logs the ids.
3. **Switch.** `SGLANG_FORCED_THINK_SUFFIX`. Changes output only on budget-forced ends.
4. **Numbers.** l.6: "G4 test 2026-10-08, 3 of 10 forced ARC attempts parsed a grid".

### 38. topklogits_patch.py
1. **Purpose.** At TP > 1 decode, skips the full `[B, V]` logits all-gather and full-vocab sampling.
   - Each rank keeps its `[B, V/TP]` fp32 shard in the shared logits buffer.
   - If the batch only needs top-k sampling (top_k <= K for every row; no logprobs, grammar, penalties, logit_bias, min_p, custom processor, seed, sampling masks or observer; flashinfer or pytorch backend), each rank sends `[top-K values | global ids | logsumexp(logits/T) | world·K Exp(1) draws]` through one small all-gather.
   - Every rank evaluates the stock filter rule exactly on the union of per-rank top-K: flashinfer "joint" is `#{p_j>p_i} < top_k and Σ{p_j>p_i} < top_p`; the pytorch rule is sorted position + exclusive cumsum. It then picks via an exponential race with rank 0's draws (argmax weight/draw).
   - All-greedy batches exchange (max, first argmax): exact including ties.
   - Otherwise it performs the stock all-gather itself, bit-identically.
   - An "inexact" flag counts rows that touch an exact tie at a rank's k-th value.
   - Auto-disabled for TP 1, DP attention / dp lm_head / lm_head all-to-all, softcapping, return_full_logits, RL on-policy, and speculative decoding.
2. **Edits.**
   - `layers/logits_processor.py`: env parse, `_tkx_logits_enabled`, `_tkx_shard_carrier`; decode returns the shard before the all-gather.
   - `layers/sampler.py`: pure math (`tkx_greedy_payload/merge`, `tkx_sample_payload/weights/merge`) and glue (`_tkx_ineligible`, `topk_logits_exchange_sample`, stats logged every 20000 steps).
   - `model_executor/model_runner.py` `sample()`: calls the exchange first; on success updates `ngram_embedding_manager` and returns the ids.
   - `sampling/sampling_batch_info.py`: CPU list `tkx_top_ks_cpu`, maintained through filter and merge.
3. **Switch.** `SGLANG_TOPK_LOGITS_EXCHANGE=K` (1-1024; rung 0 sets 32). Same sampling distribution, but a different RNG stream; greedy exact.
4. **Numbers.** l.5-9: V = 248320; "On 4x L4 (PCIe, multimem off) that is ~1-1.6 ms per step at B 13-20, plus ~0.5 ms of full-vocab sampling passes. The harness only uses top-k sampling (top_k 20, top_p 0.95, T 1.0; salvage is greedy)". Line refs: logits_processor.py:678-695; sampler.py:215-237, 285-292.

---

## D. Short summary of the hierarchical-cache + mamba pieces

- **embedhost:** about 318 MB/GPU of embedding table goes to pinned host memory, giving about 98k more KV tokens (+7 %). Exact.
- **mambaratio:** 2 mamba slots per request instead of 3. 96 slots still support MAXREQ 40 and free 0.33 GB/GPU, about 100k KV tokens. Exact; admission still checks real slots.
- **mambafinal:** under lazy mamba buffering, skips the per-1024-chunk mamba checkpoint donations during chunked prefill. These donations were evicting idle sessions' leaves and causing re-prefill cascades (v92: 19 misses, 286k tokens). Exact.
- **HiCache fixes for HiCache (host-tier KV + mamba + QSA compressed keys) on the L4 rung:**
  - hicpairown: the transfer layer count spans ids 0..47 under pair-own (was 42; v99 crash, 36 requests lost).
  - hicqsawait: the QSA compressed-key getter waits for the per-layer H2D copy (silent wrong top-k before).
  - hicmambafb: a failed load-back with a mamba component falls back to a full re-prefill from root, instead of a mismatched recurrent state that silently corrupts the session.
  - hic_check.py validates host reloads with a greedy text-equality test.
