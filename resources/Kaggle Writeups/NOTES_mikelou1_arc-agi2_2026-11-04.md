# Notes: mikelou1/arc-agi2 ("ARC AGI2", public LB 69.72, last run 2026-11-02)

Source: `resources/Kaggle Writeups/mikelou1_arc-agi2/arc-agi2.ipynb` (16 cells) + `kernel-metadata.json`. Read as text only, nothing executed. Date of notes: 2026-11-04.

Citation scheme:
- `cN:L` = notebook cell N, line L of that cell's source (cells 0-15).
- `h:L` = line L of the harness `arc_agent.py`. It is embedded in cell 12 line 7 as a zlib+base64 literal: 200,670 chars that decode to 546,966 chars, 8,718 lines.
- `<file>:L` = line L of one of the 38 patch/bench files in the `PATCHES` JSON dict at c4:9.
- `[F]` = fact visible in the code. `[I]` = my inference.

Payloads I did not read as code:
- c4:9: a 515 KB JSON dict of 38 `.py` files (patches, benches, profilers). These were read as files, not as base64, and are summarised in section 1.
- c6:277: an 88 KB JSON keep-list of experts. I parsed it: `num_experts` 384, 48 layers, 384 kept ids per layer out of 0..511. Source string: "counts.npz decode + 1.0*counts_code.npz decode + 0.5*counts_code.npz prefill (per-layer normalized)".
- c12:7: the harness payload.
- No cell has saved outputs, so no logs are available. Every number below comes from config or from code comments.

kernel-metadata:
- GPU NvidiaL4, internet off.
- Pinned docker image `gcr.io/kaggle-private-byod/python@sha256:57e612b4...`.
- Models: `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` and `dfranzen/albucino-qwen3-8-flash-next-drafter`.
- Datasets: `sonphamorg/arc3-flashnext-ple-fp8-v1` (FP8 PLE) and `dfranzen/pennyroyal-v253` (wheelhouse).

---

## 1. Model and serving

### 1.1 Model, pruning, quantization
- **Model [F].** Qwen3.8-Flash-Next. Title c0:1: "Qwen3.8-Flash-Next (expert-pruned 384) on 4x L4, program-synthesis agent". Serving line c0:3: "Pennyroyal SGLang (Franzen's ARC-AGI-3 stack, Tufa Labs' Duck), Intel AutoRound W4A16, RadixArk FP8 PLE".
- **Expert pruning 512 -> 384 per layer [F].**
  - `CFG["PRUNE_N"]=384` (c1:5). c6:275-285 writes `keep.json`, sets `--json-model-override-args {"text_config":{"num_experts":384}}` and `SGLANG_PRUNE_KEEP=<keep.json>`.
  - `prune_patch.py` wraps the weight iterator. It renames kept expert E of layer L to index `keep[L].index(E)`, skips pruned experts, and `index_select`s the router `gate.weight [512,H]` rows to the kept ids.
  - Load-time only. It changes numerics, because top-10 routing renormalises over the remaining 384.
  - **Keep set [F]:** profiled expert usage counts from decode (ARC traffic + code traffic) plus 0.5x code prefill, normalised per layer (c6:277 "source"). Keeping 384 of 512 = 96 experts per EP4 rank (marlinclean_patch.py:9).
- **Quantization [F].**
  - Target = Intel AutoRound INT4 (W4A16). c6:156 asserts `quant_method == "auto-round"` and `bits == 4`. Served with `--quantization auto-round` (c7:60). The MoE runs on the GPTQ-Marlin path; marlin_ep_patch fixes EP (see 1.6).
  - Dense projections are additionally converted to weight-only rowwise FP8 e4m3 (per-output-row fp32 scale) by `SGLANG_SM120_ONLINE_MXFP8=1` + `densefp8_patch.py`. This covers the HC mix, lm_head, GDN/QSA projections and shared expert. Bf16 activations, fp32 accumulate. Numerics change.
  - densefp8_patch.py:5-6 quotes "~3.2 GB/step of bf16 dense weights per GPU (HC mix 1.27, GDN proj 1.04, lm_head 0.32, QSA proj 0.30, shared expert 0.28)". Its tile table was "tuned on L4 by bench_densefp8.py (v76)" (densefp8_patch.py:33).
- **PLE (n-gram embedding table) [F].**
  - Option B (`PLE_MODE` auto; `build_fp8ple_view`, c6:88-144): Intel non-PLE shards + RadixArk FP8 PLE shards from `sonphamorg/arc3-flashnext-ple-fp8-v1`, symlinked into `/tmp/flashnext-w4a16-fp8ple`. It sets `ple_embedding_dtype=float8_e4m3fn` and serves with `--ple-offload-embedding` (c7:60), i.e. the table is pinned in host memory.
  - Option A fallback (c6:164-174): Intel BF16 PLE cast to FP8 at load with one global scale `SGLANG_PLE_FP8_SCALE = 2*amax(shard_0)/448` (`ple_fp8_scale.py`, saturating clamp). Comment c6:165: "47.7 GiB pinned instead of 95.4".
  - The PLE files (about 10) are copied to local `/tmp/ple_local` by 32 parallel pread/pwrite readers, starting already in the install cell (c4:39-104, "v111"). The model cell adopts the copy (c6:192-273), and the boot cell waits up to `WAIT_MIN` 150 min (c10:12-22).
  - Reason (c6:194-197): "on 10-19/20 the PLE dataset mount read at 7-32 MB/s uncached (390-725 MB/s on earlier boots) ... every rung re-read it and timed out".
- **Embedding table on host [F].** `SGLANG_EMBED_HOST=1` (rung 0 env, c1:5) + `embedhost_patch.py` puts `embed_tokens` (248320 x 2560 bf16, 318 MB/rank) in pinned host memory with a UVA gather; lm_head stays on GPU.
  - embedhost_patch.py:6-7: "v86: 28-34 running at 88% KV use"; "318 MB/GPU is ~98k more pool tokens (+7%) at 3.24 KB/token/GPU (fp8 KV, pair-owned, + indexer keys)".
  - Cost: "B rows x 5 KB over PCIe per decode step (~200 KB at B=40)" (embedhost_patch.py:12-13).

### 1.2 Drafter / speculative decoding
- **Prepared but OFF in every serving rung [F].**
  - A draft view of `albucino-qwen3-8-flash-next-drafter` is built: INT4 g32 compressed-tensors MTP routed experts, keeping its 512 experts (c6:26-79, c6:177). The draft view is created regardless.
  - All three `RUNGS` have `"SPEC": false` (c1:5). rung_args adds `--speculative-algorithm NEXTN ...` only `if spec` (c7:82-93).
  - The MTP rungs exist only in the unused `LADDER` / `BRINGUP` lists: `fp8kv_mtp`, `tp4_p384_mtp` with `SPEC_STEPS` 3, accept thresholds 1.0 (c1:5, c7:84-89). [F] boot iterates `CFG["RUNGS"]` only (c10:23).
  - `FRSPEC` false (c6:179-185).
- [I] Their final served config runs with **no speculative decoding**. topklogits_patch also disables itself under spec decoding (topklogits_patch.py:97-102). It looks like they traded MTP for more concurrent streams (41) and the top-k logits exchange.

### 1.3 Pennyroyal SGLang build
- [F] apply_patches.py docstring: "Idempotently patch an installed Pennyroyal SGLang (0.5.19+gd00d88efc8d6) for 4x L4 (sm_89)". Wheelhouse = dataset `dfranzen/pennyroyal-v253`, cu130 (c4:218).
- [F] Install (c4:106-143): offline `uv venv` + `uv pip install --no-index -r requirements.lock`, then `--reinstall --no-deps sglang-*.whl` into `/tmp/sgl-intel/venv`. Bundled CUDA 13 nvcc lives in `nvidia/cu13`, with lib symlinks and libcuda discovery (c4:145-173).
- [F] Env (c4:181-202) includes:
  - `TORCH_CUDA_ARCH_LIST=8.9`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  - `SGLANG_MAMBA_CONV_DTYPE=bfloat16`, `SGLANG_MM_PREPROCESS_DEVICE=cpu`
  - `SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1`
  - `SGLANG_QSA_SM89_DECODE=triton`
  - All `SGLANG_SM120*` knobs are removed (c4:203-204), and rungs set `SGLANG_SM120_ONLINE_MXFP8=1` again per rung.
- [F] Driver/PTX smoke on GPU 0 (c4:221-266): bf16 matmul, a Triton kernel, FlashInfer sampling.

### 1.4 Layout: TP / EP / DCP, KV, capacity
- **TP4 + EP4 [F].** `--tp-size 4 --ep-size 4` (c7:46, CFG `TP` 4 / `EP` 4).
  - Options: `--moe-a2a-backend none --moe-runner-backend auto` (c7:59).
  - A `dpattn` layout (TP4 + DP2 attention + dp lm head) exists in code (c7:44-45) but no rung uses it.
  - **No DCP flag anywhere [F].**
  - Pair-owned QSA (below) is their own form of KV de-duplication across the TP pair. [I] It plays the role DCP would.
- **KV dtype [F]:** `--kv-cache-dtype fp8_e4m3` in all rungs, `--page-size 64` (c7:61).
- **Pair-owned QSA [F].** `SGLANG_QSA_PAIR_OWN=1`, `_GROUP=pair`, via `pairown_patch.py`.
  - At TP4 with 2 KV heads, both ranks of a pair hold identical K/V. Each QSA layer is stored once per pair: owner = layer ordinal j % 2. This gives about 2x KV capacity.
  - Two pair all-gathers per QSA layer: about 0.05 ms each, 24 per step (pairown_patch.py:19).
  - pairown_patch.py:6-8: "6.1 KB/token/GPU over the 12 QSA layers".
- **Model shapes (from patch files) [F].**
  - 48 layers: 12 QSA (ids 3, 7, ..., 47) + 36 GDN.
  - Hq 24 / Hkv 2, head_dim 256.
  - QSA indexer top-k 512 blocks = 2048 tokens (+3) = 2051 entries.
  - hidden 2560; vocab 248320; 512 experts, top-10.
- **KV capacity [F].**
  - Rung 0 and rung 1 pass `--max-total-tokens 1850000` (c1:5 SERVE_ARGS).
  - `KV_TOKENS` = min logged `max_total_num_tokens` (c7:26-28). No logs are saved, so the actual value is unknown.
  - Rung 0 adds `--enable-hierarchical-cache --hicache-ratio 2`, a host-RAM KV tier of 2x.
- **Mamba state [F].**
  - `--max-mamba-cache-size` 104 (rung 0) / 96 (rung 1) / 144 (rung 2).
  - `--mamba-radix-cache-strategy extra_buffer_lazy`, `--mamba-track-interval 64`, `--mamba-ssm-dtype bfloat16`, `--mamba-max-states-per-path 1`.
  - `SGLANG_MAMBA_SLOTS_PER_REQ=2` (mambaratio_patch: stock ratio 3 -> 2). mambaratio_patch.py:6-10: "MAXREQ 40 needs MAMBA 120 = 1.66 GB/GPU of GDN state (13.8 MB per slot on 4x L4) ... With ratio 2, MAMBA 96 keeps MAXREQ 40 ... frees 24 slots = 0.33 GB/GPU = ~100k KV tokens".
- **Max running [F]:** `--max-running-requests 41` (rungs 0, 1) and 48 (rung 2).
  - Harness concurrency `INFLIGHT = min(MAXREQ, AGENT.MAX_INFLIGHT 41, KV_TOKENS // CTX_PER_SESSION 44000)` (c12:18-19).
  - [I] With a 1.85M pool, 1.85M // 44k = 42, so 41 sessions.
- **Context length [F].**
  - `--context-length 147456` (CFG `CTX`, c7:62). The harness gets `--max-model-len 147456` (c12:204).
  - Harness ctx-limit is 110000 (AGENT ARGS). Above it, the context is "pinned" (section 2.7).
- **Other server flags [F] (c7:56-79):**
  - `--load-format safetensors --weight-loader-prefetch-checkpoints`, `enable_multithread_load false`
  - `--dtype bfloat16`, `--attention-backend triton`
  - `--chunked-prefill-size 1024 --max-prefill-tokens 1024 --disable-prefill-cuda-graph`. Comment c7:64-66: with prefill graphs on, about 4 GiB/GPU is reserved, "more than the whole L4 KV pool".
  - `--cuda-graph-max-bs-decode` = MAXREQ, with explicit `--cuda-graph-bs-decode` lists per rung. Rung 0: 8,12,14..20,22,...,40,41.
  - `--mamba-backend triton`, `--linear-attn-*-backend triton`
  - `--disable-custom-all-reduce`, `--disable-flashinfer-autotune`
  - `--mm-feature-transport cpu`, `--image-processor-backend pil`
  - `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`
  - `--chat-template chat_template.jinja`, `--default-chat-template-kwargs {"preserve_thinking":true}`
  - `--schedule-policy lpm`, `--watchdog-timeout 1800`, `--enable-cache-report`, `--enable-metrics`
  - `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` (SKIP_LOCK)
  - `--crash-dump-folder` (c10:6-7)
- **Rung 0 env [F] (c1:5):**
  - `SGLANG_CLIP_MAX_NEW_TOKENS_ESTIMATION=8192`, `SGLANG_INIT_NEW_TOKEN_RATIO=1.0`, `SGLANG_MIN_NEW_TOKEN_RATIO_FACTOR=0.8`. These are admission-reserve knobs; they are stock, as no patch file implements them.
  - `SGLANG_HC_MIX_MAX_ROWS=64`, `SGLANG_QSA_MQA_GROUPS_PER_CTA=4`, `SGLANG_QSA_SM89_DECODE=triton_splitk`, `SGLANG_TOPK_LOGITS_EXCHANGE=32`
  - `SGLANG_MAMBA_PREFILL_FINAL_ONLY_LAZY=1`, `SGLANG_MAMBA_SLOTS_PER_REQ=2`
  - `SGLANG_MM_PRECOMPUTE_HASH=1`, `SGLANG_VLM_CACHE_SIZE_MB=24`, `SGLANG_EMBED_HOST=1`
  - `SGLANG_INPROJ_FP8=1`, `SGLANG_FP8_FUSED_DEQUANT=1`

### 1.5 Fallback "rungs" ladder and echo gate
- **The three rungs [F] (c1:5):**
  - **Rung 0 `tp4_p384_p41cap_hic`:** MAXREQ 41, MAMBA 104, MEMFRAC 0.935, HiCache ratio 2, `CRASH_NEXT: true`.
  - **Rung 1 `tp4_p384_p41cap`:** MAXREQ 41, MAMBA 96, MEMFRAC 0.94, no HiCache, no VLM_CACHE env.
  - **Rung 2 `tp4_p384_pair48lm`:** MAXREQ 48, MAMBA 144, MEMFRAC 0.94. No max-total-tokens, no EMBED_HOST, no SLOTS_PER_REQ, no MAMBA_SLOTS env. `LMONLY` false despite the name.
- **Boot [F] (c10:23-29):** tries rungs in order. Per-rung limit `RUNG_MAX_MIN` [45, 30, 30] min. No new rung starts after `SERVE_BUDGET_MIN` 300 min.
- **Failure detection [F] (c7:161-212):**
  - Process exit.
  - Log stall `STALL_MIN` 20 min.
  - A FATAL marker followed by a 45 s grace. Markers: "hit an exception", "CUDA out of memory", "Received sigquit", and others.
  - The failure signature is classified as MEMORY vs deterministic (ImportError, nvcc, ...).
  - The server is stopped, including a wait for GPU memory to fall below 1.5 GiB.
  - If the HiCache patches (hicpairown/hicqsawait/hicmambafb) did not apply, HiCache rungs are dropped (c4:285-289).
- **Echo gate [F] (c7:150-159, 192-196).**
  - When healthy, it sends a greedy request with thinking off: "Copy this grid exactly, one row per line, digits only, nothing else:" with a 5x6 grid. PASS = the grid string appears in the reply.
  - `SANITY_GATE` is **false** (c1:5), so the result is printed only and a FAIL still accepts the rung (`if ok or not CFG["SANITY_GATE"]`, c7:194).
- **Crash restart in the agent stage [F] (c12:235-253):** the same rung is rebooted; a `CRASH_NEXT` rung (HiCache) falls back to the next rung.
  - Limits: `MAX_RESTARTS` 4, only if more than `RESTART_MIN_S` 900 s are left. The harness is relaunched on the remaining ids with `--load-state` (c12:196-215, 302-313).
  - [F] `relaunch_server` (c7:214-222) references `CFG["LADDER"].index(SERVER_RUNG)`. It is never called (dead code).
- **Stall watchdog [F] (c12:76-98):** `STALL_MIN` 10 min without a new Prefill/Decode batch log line while sessions run counts as dead and takes the restart path. `HEALTH_FAILS_MAX` 3.

### 1.6 Every patch applied
Mandatory (c4:269-280; each run then `--check`ed) [F]:
1. **apply_patches.py --ple-scale.**
   - Installs `qsa_sm89_decode.py` as `sglang/srt/layers/attention/qsa/sm89_decode.py` and routes QSA sparse decode to it. Stock needs FA2 (not shipped) or FA4-CuTe, whose "hd256 SM80 tile needs 128 KB of shared memory, more than sm_89's 99 KB" (qsa_sm89_decode.py:17-19).
   - Algorithm: a Triton grid (rows, KV heads) walks the 2051 top-k positions in BLOCK_N 32 chunks through `req_to_token`, with online exp2 softmax. Fp8 K upcast, bf16 MMA. Valid entries after -1 holes still count.
   - Also installs the PLE FP8 scale cast (option A).
   - A GPU unit test `test_qsa_sm89_decode.py` runs (c4:291-302); on FAIL it falls back to the torch gather path.
2. **prune_patch.py:** expert pruning (1.1).
3. **draft_ep_patch.py:** lets a 512-expert MTP drafter load under EP with a 384-expert target (identity map for ids beyond the target map). Load-path only.
4. **marlin_ep_patch.py:** passes `expert_map` / `global_num_experts` to GPTQ-Marlin MoE under EP. Without it Marlin indexes expert -1. marlin_ep_patch.py:6: "(v59 tensor dump on 4xL4 tp4/ep4: layer 0 experts finite, layer 1 experts all-NaN, every output token = '!')".
5. **think_force_patch.py:** after a budget-forced `</think>` (strict thinking), it forces `SGLANG_FORCED_THINK_SUFFIX` (e.g. `"\n\n[["`). It is dormant here: `SOLVE.force_think` is false, so c10:8-11 never sets `--enable-strict-thinking`. think_force_patch.py:6: "G4 test 2026-10-08, 3 of 10 forced ARC attempts parsed a grid".

Perf patches (c4:281-284). They are non-fatal and switch on via env [F]:
- **hcmix32** (`SGLANG_HC_MIX_MAX_ROWS=64`): the fused HyperConnection-mix kernel grows from 16 to 32 rows (and in 32-row chunks beyond). Without it, 17-19 live decode rows sent all "97 mixes per step" to a 5-kernel torch.compile chain (hcmix32_patch.py:6-7).
- **qsasplitk** (`SGLANG_QSA_SM89_DECODE=triton_splitk`): split-K over the 2051 top-k entries, NSPLIT = 2xSMs / (rows·kvh), plus a merge kernel. Only rounding order changes. Reason: 13-20 rows used 13-20 of 58 SMs, each CTA doing 65 dependent iterations (qsasplitk_patch.py:6-7).
- **mqagrid** (`SGLANG_QSA_MQA_GROUPS_PER_CTA=4`): passes groups_per_cta to the TileLang indexer-MQA kernel, so the grid shrinks 4x. Bit-identical. mqagrid_patch.py:7-10: at 6k-24k context, "80-95% of CTAs still launch" while empty.
- **topklogits** (`SGLANG_TOPK_LOGITS_EXCHANGE=32`): skips the [B, V] logits all-gather. Each rank exchanges top-32 values/ids + logsumexp + Exp(1) draws and samples exactly by the stock top-k/top-p rule (greedy exact; same distribution, different RNG stream). topklogits_patch.py:5-9: the stock gather costs "~1-1.6 ms per step at B 13-20, plus ~0.5 ms of full-vocab sampling". Auto-off under spec decoding.
- **marlinclean** (`SGLANG_MARLIN_EP_CLEAN`): not set, so dormant. An EP-only Marlin MoE glue cleanup: 97 instead of 385 align buckets, no zeroing, fp32 masked reduce.
- **densefp8** (`SGLANG_SM120_ONLINE_MXFP8=1`): see 1.1.
- **idxshard** (`SGLANG_QSA_INDEXER_SHARD`): not set, so dormant. It shards the replicated indexer scan across the 4 ranks plus an all-gather merge. Its docstring holds the step-time model below.
- **hisparse**: a no-op placeholder.
- **pairown**: see 1.4.
- **embedhost**: see 1.1.
- **mambaratio**: see 1.4.
- **mambafinal** (`SGLANG_MAMBA_PREFILL_FINAL_ONLY_LAZY=1`): no per-1024-chunk mamba checkpoints under lazy buffering. Those checkpoints evicted idle sessions' leaves and caused re-prefill cascades. mambafinal_patch.py:7-10: "An 85k-token pin re-prefill thus allocates ~83 slots out of a pool of 96 ... (v92 smoke: 19 mid-session misses, 286k re-prefill tokens)".
- **hicpairown / hicqsawait / hicmambafb**: correctness fixes for HiCache with pair-own QSA and mamba.
  - hicpairown: the layer span was 42 instead of 48. hicpairown_patch.py:6-10: "v99 smoke ... IndexError ... scheduler killed, 36 requests lost".
  - hicqsawait: the QSA compressed-key read waits for the H2D copy. Before, it gave "silent wrong top-k".
  - hicmambafb: a failed load-back now re-prefills from root instead of a mismatched recurrent state.
  - `hic_check.py` is an optional reload test, `AGENT.HIC_CHECK`; it is not in CFG, so off.
- **qsadump**: instrumentation only, `CFG.QSADUMP`, off.

Cell 5 `[duck2:inproj]` (c5:1-11, "11-01 TPS build") [F]:
- `patch_inproj.py`, GDN in_proj fix in `qwen3_5.py finalize_fused_in_proj`. Stock fuses in_proj_qkvz + in_proj_ba into one bf16 buffer before the sm89 FP8 pass, which then makes an unused FP8 copy of qkvz.
  - `SGLANG_INPROJ_NODUP=1`: lossless; frees about 10.5 MB/GDN layer/GPU.
  - `SGLANG_INPROJ_FP8=1`: numerics change. It does not fuse, and qkvz runs rowwise FP8. **Rung env sets `INPROJ_FP8=1`.**
- `patch_fp8dequant.py` (`SGLANG_FP8_FUSED_DEQUANT=1`, set): prefill-size rowwise-FP8 linears above `_MAX_KERNEL_ROWS` dequantize in one Triton pass (bit-identical), then one GEMM. c5:7: the old path was "~17% of a 1024-token prefill chunk on 4xL4".

### 1.7 Phase-0 microbenchmarks (gating) [F]
- c8 launches `bench_<k>.py` per GPU and enables env flags only if the bench passes and is faster. Gates: qsasplitk needs a 1.30x minimum speedup at rows <= 20; mqagrid needs >= 3 % faster and nowhere > 2 % slower; others need patched_us < baseline_us.
- **In this build `PHASE0_BENCHES` is `[]` and `PHASE0_TRUSTED` is `{}`** (c1:5), so nothing runs. No rung has a `PERF` key, so `PHASE0["enabled"]` would never be merged anyway (c8:77-79).
- The decode-speed flags are **hard-coded in the rung ENV** instead. [I] They had passed on earlier runs (the c8:71 comment mentions "v66, v67").

### 1.8 Measured speed numbers in comments (no run logs in the notebook)
- idxshard_patch.py:7-8: "step-time fit step_ms = 26.97 + 1.092 n + 0.416 KV/100k" (n = running requests; KV = live KV tokens); "0.416 ms per 100k live KV tokens at ~185 GB/s"; "(10-16 ms at n 32-50 with 80k live ctx)" [indexer cost].
  - [I] Plugging in n = 41 and about 1.6-2.0M live KV gives a step of about 79-81 ms, i.e. about 500-520 tok/s aggregate decode without spec. At n = 32 / 1.6M it is about 69 ms, about 465 tok/s. This is an inference from their fit, not a logged number.
- embedhost_patch.py:6: "v86: 28-34 running at 88% KV use".
- prefill_bench.py:2-3: "v86: 1024-token chunks run at ~2.3k tok/s = ~0.44 s each ... ~8% of a harness run's wall time". Harness h:4396: re-prefill about 0.44 s per 1k tokens with decode stalled.
- h:4230-4232: prefix-cache hits "H100: 73% hits vs 96.8% on Kaggle v86"; "15 of 912 turns missed everything, the rest missed ~33 tokens = page rounding".
- prof_summarize.py:26: usage example `--baseline-step-ms 53.5` (probably an earlier real step time).
- mambaratio_patch.py:6-10: "under the real v86 agent load (serve.log, 1852 decode lines) the pool held 2.08 slots per running request at 32-35 running".
- c12:17 comment: "an evicted prefix is re-prefilled at ~1.5k tok/s, stalling every session".
- Harness `session_rate` clamp 2-80 tok/s per session, prior `est-session-rate` 10 (h:5262-5270). [I] About 10-12 tok/s per session x 41 is about 450-500 tok/s, consistent with the fit above.
- Stale legacy config numbers: `EST_TOK_S` 150, `SOLVE.est_tok_s` 480 (c1:5). Not used by the current agent path.

---

## 2. Harness (cell 12, `arc_agent.py` "v2")

The docstring says "(v1)" (h:2). The run-2 flags are dated "2026-10-31", and "with all of them off the harness behaves exactly as md5 2a0ef9e9" (h:25-34). Stdlib only; the sandbox uses numpy/scipy.

Launch args (c12:202-211 + `AGENT.ARGS` c1:5) [F]:
- `--answer-mode direct`, `--effort xhigh`, `--temperature 0.7 --top-p 0.95 --top-k 20`
- `--turn-max-tokens 32768`, `--act-tokens 8192`, `--session-cap 100000`, `--task-cap 100000`, `--cap-stages 150000,200000`
- `--ctx-limit 110000`, `--context-mode pin --pin-keep 0.8 --pin-keep-reasoning 1500`
- `--images on`, `--geo-kit on --geo-v2 on --geo-fits on`, `--task-kinds on`
- `--two-attempts on --fresh-second on --fresh-min-tokens 80000 --fresh-cap 100000 --fresh-temperature 1.0 --fresh-gate on --cap-action fresh`
- `--force-hypothesis on`, `--continue-cut on --continue-images chat`
- `--salvage on --salvage-mode tool --salvage-tokens 6144 --salvage-turns 2 --salvage-tools session --salvage-fix on`
- `--workspace on`, `--hand-check-rule on`, `--hide-budget on`, `--budget-note on`, `--review-outputs on`, `--propose on`
- `--max-turns 120`, `--max-sessions-per-task 1`, `--exec-timeout 30`, `--max-tool-out 6000`, `--empty-retry 2`, `--wrapup-s 240`
- `--second-opinion off`, `--resume continue`, `--submit-partial on`, `--sandbox-help on`, `--ws-import-fix on`
- `--max-inflight INFLIGHT` (up to 41), `--max-model-len 147456`, `--deadline-epoch DEADLINE`
- Direct mode rejects checks / edge-audit / test-notes / expect / guard / second-opinion / verify and compact context (h:8102-8116).

### 2.1 Prompts
The system prompt is reconstructed from `system_prompt_direct` (h:749-777) with all active edits: SYSTEM_DIRECT h:118-147; workspace h:757-760; HAND_RULE h:489-490; GEO_NOTE_V2 h:597-611 + GEO_FITS_LINE h:614-618 + GEO_CHOICE_LINE h:619-621; IMAGE_NOTE_DIRECT h:160-163 + GEO_IMAGE_NOTE h:507-508; run2_prompt h:649-665 (GEO_V2_ONE/MANY, GEO_V2_SUREST, TWO_ATTEMPTS_EDITS h:575-587); TASK_KINDS_NOTE h:735-746; BUDGET_SECTION_HIDDEN h:670-674. Text for one hidden test, `{N}` = train count:

```
You are a senior software engineer. Work out the exact outputs of the hidden tests of a test suite.

## The task
Each test maps an input grid to an output grid by one rule. A grid is a list of rows of integers 0-9 (colors), 1x1 to 30x30; the background is usually the most common color (often 0, not always). Your deliverable is the exact output grid of each hidden test.

The suite has {N} tests, each an input grid with its exact expected output. There is also 1 hidden test: you see its input but not its output, and it is graded with two attempts: the grid you record for it with `propose` and, if you record one, a second grid recorded with `propose_alt`; it counts as correct when either one matches exactly (every cell). Indices start at 0, as in the Python lists (`tests[0]`, `hidden[0]`).

## Ground rules
- ALL TESTS ARE CORRECT. Every expected output is exactly right: no typos, noise or mislabeled tests. If your output differs in even ONE cell, your rule is incomplete or wrong.
- Find the ONE rule that produces every expected output, then apply it to each hidden input. The hidden inputs can differ from the tests in grid size, colors, and the number, shape and position of objects: apply the rule, do not copy a test's output.
- You do not have to write a general program. Any code that builds the right grid is fine, including code written for one hidden input only (hard-coded positions, colors or cells, e.g. `out = hidden[0].a.copy(); out[2:5, 7] = 3; propose(0, out)`). Nothing is rejected for hard-coding.
- Study the hidden inputs early: they sometimes contain cases the tests only hint at.
- LOOK AT YOUR GRIDS; never assume what your code built. After every `propose`, compare the recorded grid (shown back to you) with what the rule demands and name any difference before you change it.
- Counting characters by eye is unreliable: compute positions, counts and sizes in `python`.
- Never check a rule by hand, cell by cell, in your thinking: write it as a function and test it on every test in one `python` call. One call is cheaper and more reliable than a page of hand-checking.

## Tools
- `python`: runs your code. Preloaded before your code: `tests` (`tests[i].input`, `tests[i].output` are Grids), `hidden` (input Grids), numpy int arrays `tests_in[i]`, `tests_out[i]`, `hidden_in[j]`, 3-D stacks `TESTS_IN`, `TESTS_OUT`, `HIDDEN_IN` ([n, rows, cols], padded with -1 to one common size; sizes in `in_shapes`, `out_shapes`, `hidden_shapes`), `np` (numpy), the Grid helpers, `propose`, and every function, class and import you defined in earlier `python` calls or in your workspace .py files (e.g. `my_rule(tests_in[0])`; later definitions win). Plain variables do not persist between `python` calls, so build a grid and propose it in the same call. scipy and the standard library can be imported (e.g. `from scipy import ndimage` for label / find_objects / binary_fill_holes; itertools, collections). A final bare expression is printed. Output is cut at 6000 characters: print crops, counts and object lists, not whole grids.
- `propose(j, grid)` (in `python`): records `grid` (a list of lists or a numpy array) as your answer for hidden[j]; the latest proposal per hidden input counts, and each one is shown back to you. Build it from the hidden input with numpy edits instead of typing a whole grid out, e.g. `out = hidden_in[0].copy(); out[r0:r1, c0:c1] = 4; propose(0, out)` or `propose(0, my_rule(hidden_in[0]))`. Propose as soon as you can predict a hidden output, and again whenever your prediction improves.
- `propose_alt(j, grid)` (in `python`): records your second attempt for hidden[j] (the latest one counts; it is shown back to you). If two readings of the rule both reproduce every test output but give different hidden outputs, submit both: `propose(j, ...)` for the one you believe more, `propose_alt(j, ...)` for the other. Use it only for a reading you have checked against every test, never for an unchecked guess.
- `submit`: submits your latest proposals (and second attempts) as your final answer and ends your work on this task. Call it once every hidden input has a proposal you have checked against the tests (size, colors, pattern).
- Workspace files (this session only): `write_file(path, content)` creates or replaces a text file; `edit_file(path, old, new)` replaces the one exact occurrence of `old` (copy it from the file, with enough lines to be unique); `read_file(path, start, end)` shows numbered lines. Functions in your .py files are preloaded into `python` too (e.g. keep your rule in rule.py, fix it with `edit_file`, and call it in `python`).

Grid helpers (`g` is a Grid; `Grid(list_or_array)` wraps a grid; `g.a` is the numpy array):
- `g.shape` (rows, cols); `g.crop(r0, c0, r1, c1)` -> Grid of rows r0..r1-1, cols c0..c1-1 (half-open, like numpy slicing); `g.bg()` -> most common color
- `g.show(r0=0, c0=0, r1=None, c1=None)` prints a region with row and column numbers; `print(g)` prints all digits
- `g.colors()` -> {color: count}
- `g.objects(bg='auto', diag=False, multicolor=False)` -> connected components of non-background cells (bg='auto' = most common color; bg=None keeps every cell), each with `.color`, `.size`, `.bbox` (r0, c0, r1, c1 half-open, so `g.crop(*o.bbox)` is the object's box), `.cells` [(r, c)], `.grid` (its box as a Grid; cells not in the object = -1)
- `g.diff(other)` -> list of (r, c, mine, theirs) for differing cells (shapes must match); `stack3(grids)` -> padded 3-D array

## Geometry kit
Python owns the coordinates: never count cells by eye. In the task message each grid is followed by its object table. An object is a 4-connected group of cells of one color other than the grid's background (its most common color), so a shape whose cells touch only at corners, such as a one-cell-thick diagonal line, is split into several objects. Columns: id (O1, O2, ... in the reading order of the bbox top-left corners), col (color), n (cells), bbox `r0,c0,r1,c1` (half-open, as everywhere in `python`: rows r0..r1-1, cols c0..c1-1, so `g.crop(*bbox)` is the object's box), centre (of the box), shape (dot, hline, vline, rect = filled rectangle, box = hollow rectangle, plus, diag = a one-cell-wide 4-connected staircase of 5+ cells, other), touches (the objects 8-adjacent to it; past 8 ids "+N"). Below a table: one `C` line per 8-connected group of several colors that the table splits into several objects (its ids, cells, bbox and the holes it encloses), and the diagonal runs of 3 or more same-color one-cell objects. A grid with many objects lists only its largest (at most 40; the header says so). A grid whose most common color covers less than half of it and is not the inputs' usual background has no background: every color region is listed; an output of another size than its input uses the input's background (none if that color does not occur). Grid names: "tests[i].input", "tests[i].output", "hidden[j]".
Kit functions, preloaded in `python` (`g`: a canvas array, a Grid or a grid name; `cells`: a list of (r, c), or of (r, c, color) which keep their colors, or an object; results are exact; no kit function changes a grid in place):
- `objects_of(g)`: the objects, as in the tables; `obj(g, "O5")`: one of them (.id .color .size .cells .bbox .centre .shape .touches); `canvas(j)`: a numpy copy of hidden[j].
- `axis(row=r)`, `axis(col=c)` (r or c may end in .5), `axis(diag=1 or -1, through=(r, c))`, `axis(point=(r, c))`; `reflect(cells, ax)`: their mirror images.
- `ray(g, start, direction, stop="edge", wall_colors=None, corner="passes", slope=None)`: the cells after start along a direction ("down-right" or (1, 1)) or a staircase (slope=(2, 1)), with `.end` and `.hit`.
- `extend(g, cells, direction=None)`: the new cells that continue a line, dotted line or staircase past its ends (or past the end toward direction).
- `move(cells, dr, dc)`; `slide(g, obj, direction, until="wall", corner="passes")`: the object's cells moved until the next step would hit a wall or leave the grid (`.steps`).
- `recolor(cells, c)`; `paint(g, cells, color=None)`, `fill(g, cells, c)`, `flood(g, seed, c, bounded_by=None)`: new canvases (flood with c=None returns the region's cells).
- `overlay(g, *constructions, ids=False, title="")`: one image of g with up to 3 cell lists (framed, dotted, crossed) and axes drawn on it; `ids=True` labels the objects. It counts toward the images per call.
- `fits(a, b, g=None, rotations=False)`: every placement of b next to a (moved, no overlap, at least one shared edge), best first by shared edges, enclosed holes it creates and how much of the union's box it fills; `fits_wall(piece, wall, g=None, rotations=False)`: placements of a piece against a large object, best first by the notch cells it fills. a, b may be objects, cells or ids ("O3", with g); each placement has `.dr .dc .rot .cells`.
- `check_tests(fn)` (also `check_examples`): runs fn on a copy of every test input and prints which outputs match exactly.
Options of ray, extend and slide: stop="edge" runs through everything, "wall" ends on the first wall cell, "before_wall" just before it; walls are the cells of wall_colors (default: every color but the background); corner="passes" checks only the cell stepped onto, "blocks" also stops a diagonal step when a side cell is a wall. These are settings to try, not hints about the rule.
When the rule involves a choice (which piece pairs with which, where something goes, which orientation, axis or period), list every option in python and print a ranked table instead of deciding by eye; then check that the ranking that explains the test outputs is a natural one.
Your own definitions with these names replace the kit's.

## Two kinds of tasks
Most tasks are one of two kinds. Decide which one this is before you settle on a rule, and state it in one visible line in your first reply (`Kind: 1` or `Kind: 2`, with the evidence).
1. Nothing is removed: every object of the input takes part in the output (kept, moved, recolored, extended, combined). There are no distractors: treat every object of each hidden input, including single cells and objects at the edges, as part of the rule. A rule that leaves an object out or calls it a decoy is wrong for this kind.
2. Removal: some input objects are noise or distractors and are removed, often together with other changes. Find the property that separates the removed objects from the kept ones in every test, and compute it in `python` for every object of each hidden input.
Check the call against the tests: compare which input objects survive into each expected output.

## Workflow
1. Study every test: compare input and output (same-size tests come with a changed-cells map: the output value where it differs, '.' = unchanged; when most cells change only the counts are given). Use `python` to crop, list objects and measure anything you are unsure of. Write small helper functions (they persist) and check each hypothesis on ALL tests in one call, e.g. print per test whether the predicted output size, colors or cells match.
2. Work out the rule precisely and what each hidden input needs. Reproducing every test output is necessary but does not prove the rule: different rules can fit every test and still differ on a hidden input.
3. Build each hidden output in `python` (with your rule function, or by editing `hidden_in[j].copy()` directly) and record it with `propose(j, grid)`.
4. Look at each proposal shown back to you and fix it until it follows the rule, then call `submit`.
Prefer another tool call over long reasoning: running code is cheaper than thinking your way to the same answer. Keep tool output small: it all stays in your context.

## Working style
Many short turns that each run code get further than one long think: as soon as you have a hypothesis, test it with `python`.

## Images
Grids are also shown as images: upscaled cells with thin gray lines, panels side by side on white (white is not a grid color); each panel has row numbers down its left edge and column numbers along its top edge, and a bolder line every 5 cells. Colors: 0 black, 1 blue, 2 red, 3 green, 4 yellow, 5 gray, 6 magenta, 7 orange, 8 light blue, 9 maroon. Use the image for shapes and layout, the text or `python` for exact cells. In changed-cells panels, white = unchanged and a colored cell shows the output value. In `python`, `img(g1, g2, ...)` attaches grids side by side as one image (up to 2 per call), e.g. `img(hidden[0], out)` to look at a grid you built; to look at a whole grid it is much cheaper than printing it.
```

Notes on this prompt:
- With 2 or more hidden tests the grading sentence reads "There are also {n} hidden tests ... each is graded with two attempts ...".
- Every edit is count==1-guarded (h:653-664, 691, 774).
- The program-mode `SYSTEM` prompt (h:80-113, "Implement a Python function that passes a test suite ... Never hard-code outputs ...") is **not** used in this config.
- The budget is hidden from the model: `BUDGET_SECTION` "This session has about {budget_k}k tokens ..." is replaced by "Working style" (h:686-693). Resume/restart notes are stripped of the k-numbers (h:675-683). Force notes say "That is enough thinking for this turn, so" instead of "I have used up this turn's thinking budget, so" (h:677-678, applied at h:5292-5294).

### 2.2 Tools [F] (make_tools_direct h:2154-2186, called at h:6278)
- **`python(code)`.** "Run a Python snippet: analysis, and propose(j, grid) to record your answer for hidden[j] (propose_alt(j, grid): a second attempt). Preloaded: tests ..., hidden ..., numpy arrays tests_in/tests_out/hidden_in and 3-D TESTS_IN/TESTS_OUT/HIDDEN_IN (padded with -1), np, Grid helpers (crop/show/objects/diff/colors/img), propose. Use print or end with an expression; output is truncated. Geometry kit (see the system prompt): objects_of, obj, canvas, axis, reflect, ray, extend, move, slide, recolor, paint, fill, flood, overlay, fits, fits_wall, check_tests."
- **`submit(note?)`.** "Submit your proposals (the latest propose(j, grid) for each hidden input, and any propose_alt second attempts) as your final answer; this ends your work on the task. Every hidden input needs a proposal."
- **`write_file` / `edit_file` / `read_file`.** Workspace with flat names, at most 100,000 chars per file and 20 files per session (h:6161-6209). `.py` definitions are preloaded into python.
- **`run_tests` is not offered.** If called, it returns RUN_TESTS_DIRECT: "Error: run_tests is not available in this task: there is no solve to test ..." (h:882-883). Calling propose as a tool returns PROPOSE_TOOL_DIRECT (h:884-885).
- **Sandbox [F] (run_sandbox h:3642-3725):**
  - Process: a fresh `python -c RUNNER` subprocess per call, in a temp cwd/HOME/TMPDIR, single-thread BLAS, `RLIMIT_AS` 3 GiB (h:2381).
  - Wall limit: exec-timeout 30 s (+10 s kill). Output cap 6000 chars.
  - No seccomp or network restriction in this code.
  - Earlier `def`/`class`/`import` statements are replayed via `pre_codes` (`hist`); plain variables do not persist. With ws-import-fix, a NameError on such a variable gets NAME_HINT (h:2802-2803, 7274-7278).
  - Up to 2 images per python call (img/overlay).
  - propose/propose_alt validate 2-D, 1-30 per side, ints 0-9 (h:2418-2434, 2715-2728).

### 2.3 Grid format: text and images [F]
- **First user message** (`task_content` h:2007-2037):
  - "## Test suite ({N} tests)", then per train pair: `pair_text` + geo object tables (input and output) + a PNG "tests[i] image: input | expected output | changed cells".
  - Then "## Hidden tests ({M})", then per hidden: "### hidden[j]\ninput (HxW):" + ASCII + object table + PNG.
  - Last line: "Work out the output grid of every hidden test, record each with `propose(j, grid)` in `python`, then call `submit`."
- **ASCII** (h:978-1017; `--grid-text digits` h:8065):
  - Digits as-is, so 0 is "0", not '.'. Panels sit side by side with 3-space gaps.
  - Same-size pairs get a third "changed cells" panel: the output value where it changes, '.' where unchanged. If more than half the cells change, only the counts are given.
  - The header holds the change summary "x->y xN" (top 6).
  - Row/column rulers appear when a side is > 10 (`--ruler auto`).
- **Object tables** (geo-v2, h:1689-1905):
  - Per grid: "id col n bbox centre shape touches", with half-open bbox and the background rule.
  - At most 40 objects (shrunk if all tables in a task exceed 12,000 chars).
  - C-lines for multicolor 8-connected groups, holes, diagonal dot runs.
- **Images** (h:958-1281):
  - Palette PNG (ARC colors, white gaps), 10 px/cell (2-32 adaptive), max side 1280, padded to 32-px multiples (Qwen3-VL patching).
  - Ruled panels: row/column numbers and a bold line every 5 cells.
  - Max 16 images per request; feedback images capped at 8 (oldest dropped).
  - A startup probe turns images off if the server rejects them (h:4603-4624).
- **Proposal echo** (h:5406-5479): every propose prints a status line plus the grid text, a changed-cells map vs input, and an image "input | your proposal | changed cells". propose_alt is diffed against attempt 1.

### 2.4 Verification rule / acceptance [F]
- **No train-pass gate.** In direct mode `submit` is refused only if:
  - (a) a hidden input has no proposal: "Error: not submitted: no proposal is recorded for hidden[j]..." (h:7163-7167); or
  - (b) with review-outputs on, a proposal or alt changed in an earlier python call of the same turn. SUBMIT_FRESH_DIRECT: "Error: not submitted. Look at the proposals shown in this turn's python result first; if they look right, call `submit` in your next turn." (h:7168-7170, h:886-887).
- Otherwise the session's latest proposals are accepted as attempt 1 (h:7176-7184).
- **Self-check is model-side only:** prompt + kit `check_tests(fn)` (h:3361-3385). There is no harness check that the model's rule reproduces the train outputs.
- **cap-action fresh** (h:7418-7429): an own session ending with token_cap / max_turns / context_full is closed and its (possibly partial) proposals become attempt 1.

### 2.5 Attempts and token budgets [F]
- **Per turn.** `max_tokens = min(turn-max-tokens 32768, room to max-model-len, session_cap - reserve - session_gen + 2048, task_cap - reserve - task_gen + 2048, session_rate x (wrap - now))` (h:6891-6893).
  - Thinking and answer share max_tokens; there is no separate thinking budget (`answer-reserve 0` gives use_budget False, h:4487, 4634-4635).
- **Per session/task.** 100k tokens, thinking included (session-cap = task-cap = 100000). The session stops at `>= cap - salvage reserve 6144`, i.e. about 93.9k (h:6818-6819).
- **max-turns 120.**
- **cap-stages 150000,200000** (h:8118, 8395-8402): when no unsolved task is under the current cap, task-cap is raised to 150k, then 200k.
  - [I] With cap-action fresh, capped tasks are closed as accepted, so stages mainly apply to tasks whose session ended otherwise (deadline/interrupted).
- **continue-cut** (h:6936-6949, h:5284-5330). A turn cut at 32768 tokens:
  - Cut **inside thinking**: the harness appends "\n\n" + FORCE_NOTE_HYP + "\n</think>\n\n" to the exact token ids and continues for up to `act-tokens` 8192 tokens. Via /generate, or chat `continue_final_message` when images are present ("10-21: every image continuation re-prefilled its whole prompt", h:5289-5291).
  - FORCE_NOTE_HYP (h:893-896), with the hide-budget prefix swap: "That is enough thinking for this turn, so first I will write my current hypothesis (the rule as I now believe it, in 1-3 sentences) as visible text, then make one tool call that tests exactly that hypothesis."
  - Cut **after `</think>`**: the answer is re-sampled from the think end.
- **Salvage** (h:7381-7417, h:5482-5591): after the session loop, when status is in (token_cap, max_turns, context_full, deadline) and a hidden input still has no proposal.
  - Up to 2 turns of 6144 tokens with the session's tools (cached prefix). Turn 1 has thinking on, turn 2 `enable_thinking=False`.
  - Prompt SALVAGE_PROMPT_TOOL_DIRECT (h:178-182): "Time is up for this session: stop exploring. In ONE `python` call, build your best prediction ... Give your single best guess even if you are unsure; do not write grids out as text."
  - Existing proposals are never overwritten.
- **empty-retry 2:** turns whose only calls are empty are re-sampled (h:6982-6991).
- **Two attempts:**
  - (1) `propose_alt` inside the same session (two-attempts on).
  - (2) **fresh-second** (h:8341-8384): an independent second session with the same opener, empty context, no seed, temperature 1.0, budget fresh-cap 100k. It is scheduled at lowest priority, once no attempt-1 work is left, for tasks whose attempt 1 was capped or used >= 80k tokens (capped first, then most tokens).
  - fresh-gate: it starts only if `(deadline - now - wrapup) x observed global rate - tokens committed to running sessions >= 100k`.
  - Its submit accepts nothing; its grids are attempt-2 candidates.

### 2.6 How attempt_1 / attempt_2 are filled [F] (attempt_detail h:7568-7602; choose_attempts delegates at h:7502-7503 because att2 is set, h:8155)
Per hidden j:
- **Candidate pools.** A1 = the accepted (submit or cap-close) proposal, else the latest own proposal. ALT = latest own propose_alt. F = latest fresh-session proposal. FALT = latest fresh propose_alt. OLD = the most recent earlier own proposal differing from attempt_1.
- **attempt_1** = A1, else F, else ALT, else FALT, else the test input.
- **attempt_2** = the first of **F, ALT, FALT, OLD** that is non-None and differs from attempt_1, else a copy of attempt_1. So an independent fresh-session answer outranks the same session's propose_alt.
- **Notebook merge (c1:124-157):** harness out dirs (one per relaunch) are overlaid in launch order. A valid non-placeholder entry overrides; an invalid attempt is replaced by its valid sibling. Placeholder = test input for both attempts (c1:49-52).

### 2.7 Context handling [F]
- Mode pin (h:6848-6864): trigger when the prompt estimate > 110000 or room < 4096.
  - Older turns are stripped to their visible text (2000 chars), tool args (1200), tool results (400). Reasoning keeps its last 1500 chars, "[older reasoning cut to its last part]". The 2 newest turns stay whole. Target 0.8 x 110k.
  - Note PIN_NOTE_KEEP_DIRECT (h:1389-1391).
  - Fallback roll (drop oldest turns, roll-keep 0.6), then context_full.
- Reasoning is stored as `reasoning_content` in each assistant message and re-sent; the server sets `preserve_thinking: true` (c7:77). [I] So full multi-turn thinking stays in the prompt-prefix cache until pinning (they measured 96.8 % prefix hits, h:4230).

### 2.8 Scheduling / concurrency [F]
- `max-inflight` worker threads (up to 41; c12:18-19).
- next_job (h:8386-8456): never-started tasks first, in **sorted id order** (`RUN_IDS = sorted(test_tasks)` in full mode, c1:91). Then unsolved tasks under cap with the fewest tokens. Then cap-stage raises. Then fresh-second jobs.
- No new session if less than 120 s are left, or `left x est-session-rate < 4096`.
- `max-sessions-per-task 1`. A worker also waits ("wait") while sessions run, in case they free budget.
- The KV gate (turn-level admission) is off (`--kv-gate` 0).
- MetricsPoller logs `/metrics` every 30 s.

### 2.9 Sampling and reasoning effort [F]
- Request body (h:4626-4641): `temperature` (0.7; 1.0 in the fresh session, h:6241-6244), `top_p` 0.95, `top_k` 20, `chat_template_kwargs {"enable_thinking": true, "reasoning_effort": "xhigh"}`, `tools`, `return_token_ids`.
- Non-streaming. No stop/seed/penalties. No `thinking_token_budget`.
- `--late-effort` exists but is unset.

### 2.10 Deadline handling [F]
- Harness: `--deadline-epoch DEADLINE`. In full mode `DEADLINE = T0 + 43200 - 1200` = notebook start + 11 h 40 min (c12:11-12).
- Inside the harness: no new turn after `deadline - wrapup 240 s` (- 20 s guard) (h:6820-6821). Per-turn max_tokens is sized by session_rate x time left. Workers are joined until deadline + 300 s.
- Notebook: harness killed at DEADLINE + `KILL_GRACE_S` 420 s (c12:279). The server is stopped without the GPU-release wait in full mode (c13:28-40, "worst-case tail DEADLINE + ~735 s -> ~555 s").
- No restart if less than 900 s are left; the agent is skipped if less than `MIN_SOLVE_S` 600 s are left.

---

## 3. Ops

### 3.1 CFG flags [F] (c1:5, overridable by env `DUCK2_CFG_JSON`, c1:6)
- **`QUICK_VERSION: true`** (c2:1-6). In an interactive/commit run (`RUN_MODE` != full) it writes the placeholder submission.json and appends "quick_version" to FAILED, so every later gated stage (install, model, serve, phase0, boot, agent) skips: "the interactive run of a submission build does no GPU work (~1 min of quota)". Only the competition rerun (`KAGGLE_IS_COMPETITION_RERUN`) does the full run (c1:7-8).
- **`RUN_MODE: "smoke"`.** Interactive mode. On the eval set it takes 40 tasks (`ids[::5]` first, then the rest) into `agent_eval/` (never into submission.json), with `AGENT.SOLVE_MIN` 22 min (c1:82-99, c12:13-14). Solutions are kept in the notebook only and never passed to the harness (c1:94-96). After the run it greps transcripts for "solutions.json" mentions (c12:377-380).
- **`BENCH`:** not in CFG, so off. When set, c11 runs `prof_drive.py`, a closed-loop decode benchmark at several batch sizes with optional torch profiler.
- **`PCIE`:** not in CFG, so off. When set, c9 runs `nvidia-smi topo` + `bench_pcie.py` (bulk H2D/D2H and scattered page copies, solo and all-4) "sizes the KV-offload option".
- **`LBPROBE: false`** (c12:23-26, c15). A diagnostic submission mode: in a competition rerun, the first failure ends the run with no restarts, and the *LB result itself encodes the failure class* (c15:1-15):
  - real score = no failure;
  - 0.00 = server alive but stuck (stalled / stalled_noanswer / stalled_io / health; it writes the placeholder);
  - "Notebook Threw Exception" = the server died / never came up / late / a stage failed (it raises);
  - "Notebook Timeout" = died_leak (it sleeps past the budget);
  - "Submission Scoring Error" = harness crash/hang (it writes an invalid submission.json "lbprobe ...").
- Others:
  - `DRIVER_TEST` true, `QSA_GPU_TEST` true, `PRECACHE_MODEL` true (core shards + drafter, 3 threads, 30 s delay; PLE excluded "would evict them from page cache", c6:186-190).
  - `SANITY_GATE` false, `FRSPEC` false, `PRUNE_N` 384, `PLE_MODE` auto.
  - `MONITOR_MIN` 3 (a system census every 3 min interactive / 10 min rerun, c3:128).
  - `AGENT.LEAK_CHECK_S` 60 (smoke only: abort all, then check that the strict idle memory check finds no leaked KV/mamba slots, c12:100-128).

### 3.2 Time budget [F]
- `FULL_BUDGET_S` 43200 (12 h), `END_MARGIN_S` 1200, so the solve deadline is at +11 h 40 min.
- `SERVE_BUDGET_MIN` 300 for boot attempts; `RUNG_MAX_MIN` 45/30/30; PLE copy wait up to 150 min (counted before rung 0's clock).

### 3.3 Startup steps [F]
1. c1: config, find inputs, write the fallback submission (test inputs as both attempts) "valid (zero-score) file from the first minute on" (c1:52).
2. c3: precache helper + census thread.
3. c4: start the PLE local copy thread, precache the wheelhouse, offline install, CUDA/link setup, env, nvcc check, GPU check (needs >= 4 GPUs, warns if not sm_8.9), driver smoke, write and apply patches (+ `--check`), perf patches, QSA GPU unit test.
4. c5: inproj + fp8dequant patches.
5. c6: model views (FP8-PLE view, draft view, keep-list), precache core shards, adopt the PLE copy.
6. c7: serve helpers. c8: phase0 (empty). c9: pcie (off).
7. c10: wait for the PLE copy, boot rungs.
8. c11: bench (off).
9. c12: agent + monitor loop (status every 3 min, merge into submission.json each tick).
10. c13: final validation. c14: JIT export. c15: lbprobe.
- Each stage is wrapped `if FAILED: skip` / `try ... stage_failed` so a rerun never raises (c1:55, "a raising rerun = 'Notebook Threw Exception'").

### 3.4 JIT cache export [F] (c14)
- Only after an interactive GPU run: never a rerun, never QUICK, never if install failed.
- It tars `/tmp/sgl-intel/cache/{triton,tilelang,torchinductor,flashinfer,cuda,sglang}` (if <= 4 GB) to `/kaggle/working/jit_cache/jit_cache.tar` + manifest.json (sha256, sizes, wheel, patch md5s, versions, driver, rung, GRAPH_BS).
- Comment c14:1-5: "so the user can turn this version's output into a dataset for a later consumer cell. Uploads nothing ... Inert for scoring". No consumer cell exists in this version [F].

### 3.5 Submission validation [F] (c13:3-17)
- Every test task must have a list of the right length.
- Each entry needs valid 2-D int grids (1-30, ints 0-9) for both attempts; otherwise that entry is replaced with the placeholder.
- Keys are restricted to the test tasks; written atomically; on any exception the whole fallback is written.
- `merge_tick` writes the merged submission every status tick during the run (c12:230-233), so a mid-run kill still leaves the progress.

---

## 4. Differences vs our pipeline that could matter

Our side: Merlin (vLLM 0.32 fork) TP2xPP2, klee100 3bpw Flash-Next, 24 streams, MTP or DFlash2, xhigh with 60k thinking floor, caps 104k/120k, Duck/Poetiq/Direct harnesses, E28 K100 eval 107/172 (task-norm 65.0) at 419.5 tok/s, best LB 54+.

**Engine and layout**
1. **Engine / layout [F].** Pennyroyal SGLang 0.5.19 at **TP4 + EP4** with **pair-owned QSA KV** (each QSA layer stored once per TP pair, about 2x KV, c1:5 + pairown_patch) and the **embedding table and PLE in pinned host memory** (c7:60, embedhost). We run Merlin TP2xPP2.
   - [I] Their KV-capacity stack (pair-own, embed-host, mamba ratio 2, mambafinal, FP8 dense weights freeing memory, `--max-total-tokens 1.85M`, HiCache 2x host tier) is what makes 41 concurrent sessions of ~44k live context fit on 4x L4.
2. **Concurrency 41 vs our 24 [F]** (c1:5 MAXREQ 41 / MAX_INFLIGHT 41; c12:18-19).
3. **No speculative decoding in the served rungs [F]** (all RUNGS `SPEC: false`). We use MTP or DFlash2.
   - [I] At 41 streams they rely on batch width plus decode-kernel patches (split-K QSA, MQA grid, HC-mix 32 rows, top-k logits exchange, FP8 dense GEMV) instead of drafting.
   - Their comment-level fit (idxshard_patch.py:7-8) implies about 500 tok/s at n=41 [I]. That is comparable to our 419.5 tok/s, so raw speed is not obviously the gap.
4. **Quantization [F].** Intel AutoRound W4A16 + **expert pruning to 384/512** chosen from profiled ARC/code expert usage (c6:277), FP8 KV, FP8 PLE, FP8 dense projections (densefp8, INPROJ_FP8). We use klee100 3bpw on all 512 experts.
   - [I] Different trade: 4-bit weights with fewer experts vs 3-bit with all experts.

**Harness: request shape and budgets**
5. **Multi-turn tool agent vs long single think [F].**
   - Each turn is capped at 32,768 tokens (h:6892). A turn cut while thinking is force-closed with a "write your hypothesis, then one tool call" note and up to 8,192 tokens of continuation (h:5284-5330, h:893-896).
   - The prompt pushes "Many short turns that each run code get further than one long think" (h:670-674) and "Never check a rule by hand ... write it as a function" (h:489-490).
   - Our 60k thinking floor per request is the opposite design: their per-request thinking is effectively at most 32k, with the 100k spread over up to 120 turns. [I] This matters given our rule "no token cuts below 60k". Their budget is per *session* (100k total), not per request.
6. **Budget per task [F].**
   - 100k tokens per session and per task, thinking included, hidden from the model (hide-budget, h:686-693).
   - cap-stages to 150k/200k when everything is at the cap (h:8395-8402).
   - A fresh independent second session of up to 100k at T=1.0 for tasks that used >= 80k or hit a cap, gated on remaining global token capacity (fresh-gate, h:8354-8376).
   - Our caps are 104k/120k per attempt, and our Exp 28 scheduler runs attempt 2 for train-fails. [I] Similar total per task (~200k), but they spend attempt-2 tokens only on hard/capped tasks and only when spare capacity exists.

**Harness: answers and attempts**
7. **Answer mode "direct" [F].** No `solve` program and no run_tests. The model builds each hidden output with any code, hard-coding explicitly allowed ("Nothing is rejected for hard-coding"), and records it with `propose(j, grid)`. Each proposal is echoed back as text + changed-cells map + image (h:5406-5479).
   - [I] This differs from Poetiq/program-synthesis (ours requires a program passing train) and from our Direct harness (one long think then a grid). It is a hybrid: tool-verified reasoning, but the answer is a grid, not a program.
8. **No train-pass acceptance gate [F]** (h:7159-7187). Submit needs only a proposal for every hidden input, plus "look at the echo first" (SUBMIT_FRESH_DIRECT). Train checking is left to the model via `check_tests(fn)`.
   - Our pipeline uses train-pass / self-submit signals for scheduling (Exp 28 T1/B2). [I] Their attempt-2 trigger is token usage/cap, not train failure.
9. **Attempt filling [F]** (h:7592-7599). attempt_2 = fresh-session answer F if different, else in-session `propose_alt`, else fresh alt, else an older own proposal; never a duplicate if any distinct candidate exists. The prompt tells the model to use propose_alt for "two readings of the rule [that] both reproduce every test output" (h:575-587).
   - [I] Ours puts best-train in attempt_1. Theirs: the main session's submitted answer is attempt_1; an independent T=1.0 re-solve is attempt_2.

**Harness: inputs and prompt**
10. **Geometry kit + object tables [F].**
    - Every grid in the first message is followed by a Python-computed object table (id, color, size, half-open bbox, centre, shape class, touches, multicolor groups, holes, diagonal runs) (h:1689-1905).
    - The sandbox preloads exact construction functions (objects_of, axis, reflect, ray, extend, slide, flood, overlay, fits, fits_wall, check_tests) (h:2858-3639).
    - [I] We give no structured object table in the prompt.
11. **Images on [F].** 10x palette PNGs with row/column rulers for every train pair (input | output | changed cells) and hidden input, proposal-echo images, and img()/overlay() from python (up to 2 per call, 16 per request) (h:958-1281, 2051-2094).
    - Our notes record "grids-as-text in our adapter" for Duck. [I] This is a potentially large difference.
12. **"Two kinds of tasks" prompt [F]** (h:735-746). The model must state `Kind: 1` (nothing removed, no distractors) or `Kind: 2` (removal) in its first reply. A targeted prior we do not use.
13. **Sampling [F].** temperature 0.7 (main) / 1.0 (fresh second), top_p 0.95, top_k 20, effort xhigh, enable_thinking true (h:4626-4631). [I] Check what temperature our harnesses use.

**Harness: context and caching**
14. **Context / cache handling [F].**
    - Full reasoning is kept across turns (`reasoning_content` + server `preserve_thinking: true`, c7:77).
    - Above 110k the context is pinned: old reasoning is cut to its last 1,500 chars and old tool results to 400 chars (h:6848-6864, 1383-1461).
    - `--schedule-policy lpm` (longest-prefix-match), 96.8 % prefix-cache hits measured on Kaggle v86 (h:4230), continuation through exact token ids to keep cache hits (h:5284-5330).
    - Context length 147,456 (c7:62); our max-model-len is 131,072.

**Ordering and scheduling**
15. **Task ordering / coverage [F].** Tasks start in sorted id order, one session each, before any task gets more (h:8389-8392). The fresh second session is lowest priority and gated on spare capacity (h:8411-8416, 8354-8376).
    - [I] Coverage of all 120 hidden tasks is ensured first at 41 streams. This matches our finding that coverage binds; their wider batch plus a 100k cap per task gives about 41 x 100k ~= 4.1M tokens in flight.

**Robustness and tooling**
16. **Robustness ops [F].**
    - Fallback rung ladder, a HiCache rung that falls back on crash (CRASH_NEXT).
    - Server stall watchdog (no batch line for 10 min) and up to 4 restarts with harness relaunch via `--load-state` that continues interrupted sessions (h:7642-7683).
    - Placeholder submission from minute 1, merge every tick.
    - LBPROBE mode that maps the failure class to the LB outcome (c15).
    - QUICK_VERSION commit that does no GPU work (c2).
    - [I] Some of this we have (placeholder, restarts). LBPROBE is a cheap way to diagnose hidden-rerun failures through the LB.
17. **PLE to local disk [F]** (c4:39-104, c6:192-273). The PLE files are copied off the dataset mount to local /tmp with 32 parallel readers before boot, because the mount read at 7-32 MB/s on 10-19/20. This matches our GCSFUSE slowdown observation.
18. **No DCP, no PP [F].** Plain TP4/EP4.

**Items not in the code** (so no comparison is possible): actual measured tok/s of this LB run (no saved outputs), eval score of this exact version, and how the 384-expert keep list was profiled beyond the source string.
