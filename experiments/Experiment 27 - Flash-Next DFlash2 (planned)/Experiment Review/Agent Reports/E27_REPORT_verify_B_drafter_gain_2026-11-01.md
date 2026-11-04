# DFlash2 for Qwen3.8-Flash-Next on Merlin v3.0.0 (4×L4): independent verification of the drafter side and the expected gain

Read-only check. No project files were edited, nothing was pushed, git was not run.

**Sources:**
- Merlin source `_scratch/e23/p2/merlin-src/vllm/` (ad42e37).
- The drafter header bytes (all 81 tensors parsed).
- From HF: `config.json`, README and tree API of incoai/Qwen3.8-Flash-Next-DFlash2.
- The inco.ai DFlash2 blog.
- The Flash-Next tokenizer, `config.json`, index and `quantization_config.json` at `_scratch/e13_rc2_local/tmp/fn_merlin/` (shards there are 128 B stubs).
- The klee100 overlay at `_scratch/e25_kits_out_klee_v2/.../fixkeys_overlay/`.
- The audit zip (extracted to `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\dflash\zip\`).
- A kernel-level re-analysis of our NB4S Q-defer profiler trace: `_scratch/e25_gpu_out/q27-xhigh-e25-nb4s-stall/nb4s/traces/profile_qd/dp0_pp1_tp0_..._rank2...json.gz`.

**Headline:**
1. The drafter matches Merlin's loader exactly.
2. The "silent garbage embedding" claim is **refuted**. Under PP2 the boot fails loudly with a RuntimeError; the embedding fix is still needed.
3. Token 248077 is the first unassigned padding id. That is safe, if the embedding row is byte-exact.
4. Option (b), a repacked drafter with `embed_tokens`, needs no code and costs about the same PP1 memory that removing MTP frees.
5. Our own trace shows MTP K3 drafting costs **9.8-13.4 ms per micro-batch on stage 1**, and 7.5-7.7 ms of that is **three lm_head reads**. DFlash2 needs one, so the gain estimate in the report is too pessimistic.

---

## 1. Drafter vs Merlin's loader

### Tensor list and config: CONFIRMED, exact match
The header has 81 tensors, all BF16, 603,065,600 params. File size 1,206,140,008 B (tree API).

| Item | Checkpoint | Merlin expects | Match |
|---|---|---|---|
| q / k / v / o | `q_proj` [4096,2560], `k_proj` and `v_proj` [1024,2560], `o_proj` [2560,4096] | 32 heads × 128, 8 KV heads (`qwen3_dflash.py` attention; TP2 gives 16 / 4 per rank, asserts pass) | yes |
| q / k norms | [128] | `head_dim` 128 | yes |
| MLP | gate / up [7680,2560], down [2560,7680] | `intermediate_size` 7680 | yes |
| stacked mapper | `q/k/v_proj`, `gate/up_proj` | `.q_proj` etc. stacked into `qkv_proj` / `gate_up_proj` (`qwen3_dflash.py` `hf_to_vllm_mapper`). `kernel_projection` and `hidden_projection` do not hit those substrings. | yes |
| fc | [2560, 12800] | `_get_dflash_fc_input_size` = `target_hidden_size` (None, so `hidden_size` 2560) × len(aux layers) 5 = **12800**. `combine_hidden_states` raises on a mismatch, so it is loud. | yes |
| aux layers | `target_layer_ids` [5,15,25,35,45] | `eagle3_utils.py:108-111` adds +1, giving **(6,16,26,36,46)** | yes |
| grouped conv | `attention_conv` and `mlp_conv`: `base_kernel` [2,2,2560], `kernel_projection` [640,2560] | `DFlashGroupedConv` (`qwen3_dflash2.py:190-202`): `base_kernel` (2, taps=`conv_kernel_size` 2, 2560); `kernel_projection` out = 2·taps·(2560/16) = 640 | yes |
| conv block | `block_size` 8 (as trained) | conv `block_size` = **1 + `num_speculative_tokens`** (`qwen3_dflash2.py:256`). So K=7 is exactly the trained block; K=3 / 4 are truncated blocks, and K=3 is what incoai validated. | ok |
| candidate selector | `predecessor_codebook` and `successor_codebook` [248320,256], `hidden_projection` [256,2560] | `CandidateSelector` (`qwen3_dflash2.py:326-340`): `vocab_size` × `selector_rank` 256; `selector_top_k` 16 (DFlash2 speculator `:195`) | yes |
| norms | `input_layernorm`, `post_attention_layernorm`, `hidden_norm`, `norm`: all [2560] | same | yes |
| attention mode | `is_causal` false; all layers `sliding_attention` with window 2048 | `_dflash_layer_causal` gives non-causal (`qwen3_dflash.py:88-99`); `_resolve_layer_attention` gives SWA 2048; single-type layers do not need the mixed-type check | yes |
| rope | `rope_parameters` theta 10000 | `set_default_rope_theta` only fills a missing theta (`transformers_utils/config.py:554-560`) | yes |
| registry | architecture `DFlash2DraftModel` | `registry.py:645`, mapped to `DFlash2Qwen3ForCausalLM` | yes |

**Model params with no checkpoint tensor:**
- `model.embed_tokens.weight`: mapped to None at `qwen3_dflash.py:918-919` and never loaded.
- `lm_head.weight`: replaced by the target's at `dflash/utils.py:62-73`.
- `model.mask_embedding`: mapped away, because there is no `mask_embedding.pt`.
- `draft_id_to_target_id`: not created, since vocab 248320 equals the target's (`:812-818`).

**Unexpected tensors:** none.

The strict loaded-weights check is off: `DFlashQwen3ForCausalLM.load_weights` returns None (`:886-927`), so `default_loader.py:507-509` skips tracking. A missing tensor is therefore never reported, which is why the embedding matters.

### embed_tokens: loaded from the drafter, or shared from the target?
- It is shared from the target only when the target has one on the same rank (`dflash/utils.py:60`, which calls `eagle/utils.py:42-74`).
- The drafter is built only on the last PP rank (`model_runner.py:381-384`).
- On PP rank 1 the target's `embed_tokens` is a `PPMissingLayer` (`qwen4_exp/nvidia/model.py:604-610`), which becomes None at `eagle/utils.py:49-50`.

### "Silent garbage embedding" claim: REFUTED, it is a loud boot failure
- The early return at `eagle/utils.py:56-59` needs `not hasattr(draft_model, "has_own_embed_tokens")`.
- `DFlash2Qwen3ForCausalLM` inherits from `Qwen3ForCausalLM`, which subclasses SupportsEagle and SupportsEagle3 (`qwen3.py:280-282`).
- The protocol base declares the class attribute `has_own_embed_tokens: bool = False` (`interfaces.py:1512`). `hasattr` is therefore True; I checked Protocol-subclass inheritance in Python.
- So there is no early return. The code goes on to `:61-68`: `target_embed is None`, the draft has `embed_tokens`, and `has_own_embed_tokens` is False. It raises **RuntimeError "DFlash2Qwen3ForCausalLM needs the target input embedding, but it is unavailable on this PP stage"** at boot on PP rank 1.

What still holds from the report: the drafter needs its own embedding under PP2, and the lm_head share on the last rank is correct.

## 2. Tokenizer and mask_token_id 248077
**Evidence** (`fn_merlin/tokenizer.json`, md5 085e1016…, identical to `fn_tok`):
- BPE vocab ids 0-248043.
- 33 added tokens, ids 248044-248076 (248044 `<|endoftext|>`, 248046 `<|im_end|>`, 248068 / 248069 `<think>` / `</think>`, …, 248076 `<|audio_pad|>`).
- **Ids 248077-248319 (243 ids) are unassigned.**

**Verdict:**
- 248077 is not a special token. It is the **first unused padding row** of the 248320-row embedding and lm_head.
- It is fine as an internal mask id: the tokenizer can never emit it, so there is no collision.
- The mask "embedding" is simply row 248077 of the target `embed_tokens`. That row must be **byte-identical** to the one the drafter was trained with. A checkpoint that re-initialised or zeroed padding rows would cause a silent acceptance collapse.

**Vocab consistency:**
- 248320 matches `text_config.vocab_size` 248320 (Merlin W4 and klee100), the drafter `vocab_size` 248320 and `tie_word_embeddings` false on both.
- bos / eos 248044 also match.

**Minor point:** `get_top_k_tokens` masks only org-vocab padding (`logits_processor.py:287-290`, which is 0 here). The 243 unused ids can therefore appear as candidates. That is harmless, because the target rejects them.

## 3. Embedding: options and memory

**W4 checkpoint:**
- `model.language_model.embed_tokens.weight` is in `model-00015-of-00017.safetensors` (index).
- It is **unquantized BF16**:
  - `config.json` `quantization_config.ignore` contains `re:.*embed_tokens.*` and `lm_head`;
  - `quantization_config.json` `extra_config` `.*embed_tokens.*` = `{bits: 16, data_type: float}`;
  - `block_name_to_quantize` covers only `model.language_model.layers`.

**klee100:** `embed_tokens` and `lm_head` are in `model-00002-of-00011`. `block_name_to_quantize` covers only layers and `mtp.layers`, so they are also BF16.

The local shards are stubs, so the header dtype and shape [248320,2560] must be asserted in a kit.

**(a) Loader patch.** The target embed on rank 1 is a `PPMissingLayer`, so read the tensor straight from the target checkpoint on rank 1. Sketch, inside `load_dflash_model` **before** `dflash/utils.py:60`:
```python
pp = get_pp_group()
if pp.world_size > 1 and not pp.is_first_rank and not getattr(dflash_model, "has_own_embed_tokens", False):
    tgt = vllm_config.model_config.model
    wm = json.load(open(f"{tgt}/model.safetensors.index.json"))["weight_map"]
    key = "model.language_model.embed_tokens.weight"
    with safe_open(f"{tgt}/{wm[key]}", "pt", device="cpu") as f:
        full = f.get_tensor(key)                      # [248320, 2560] bf16
    w = draft_inner.embed_tokens.weight
    w.weight_loader(w, full)                          # VocabParallelEmbedding.weight_loader (vocab_parallel_embedding.py:541) narrows to the TP shard
    dflash_model.has_own_embed_tokens = True          # so maybe_share_target_embed keeps it (no raise)
```
- Assert `full.dtype == torch.bfloat16`, then check `norm(row 248077) > 0` and the sha of that row.

**(b) Repacked drafter.**
- Append `embed_tokens.weight` [248320,2560] BF16, byte-copied from the exact checkpoint the notebook serves. Do not prefix it with `model.`.
- The loader prefixes it to `model.embed_tokens.weight` (`qwen3_dflash.py:899-905`). `process_eagle_weight` then sets `has_own_embed_tokens=True` (`models/utils.py:1144-1145`), and `maybe_share_target_embed` keeps the copy on PP rank 1 (`:61-69`).
- On a PP1 boot, `_should_share` runs `torch.equal`, finds the tensors equal and shares (drops the copy).
- No code. Recommended for the first GPU test.

**(c) Optional memory lever.**
- Change `qwen3_dflash.py:461` to `PinnedHostVocabParallelEmbedding` when `should_use_host_embed_table()`. This gives 0 GPU and 0.64 GB pinned host per PP1 rank.
- The same env also moves the target's rank-0 embedding and the MTP embedding to the host, so treat it as a measured arm only.

**Memory per PP1 card** (TP2, BF16, computed from the header):

| Component | Per-card size |
|---|---|
| Drafter weights | 390.0M params = **0.727 GiB** |
| — attention (TP-sharded) | 65.5M |
| — MLP (TP-sharded) | 147.5M |
| — selector codebooks (replicated) | 127.8M |
| — fc (replicated) | 32.8M |
| — conv projections (replicated) | 16.4M |
| Embedding shard (124,160 × 2560) | **0.592 GiB** |
| **(a) or (b) total** | **≈ 1.32 GiB** |
| **(c) total** | ≈ 0.73 GiB GPU |

What dropping MTP frees on each PP1 card:
- MTP's own `embed_tokens`: a GPU `VocabParallelEmbedding` (`mtp.py:248`), 0.59 GiB, because `VLLM_HOST_EMBED_TABLE` is unset in our notebooks.
- MTP routed experts: INT4 g128, 512×3×2560×640, split by EP2, ≈ 0.63 GB.
- MTP attention, shared expert and fc: ≈ 0.05-0.1 GB.
- Total ≈ **1.2-1.3 GiB**.

**Net:**
- (a) / (b): ≈ +0.03 to +0.12 GiB. That is within the W4 margin of 191 MiB, but tight. For the first boot on W4, lower `kv_cache_memory_bytes` by about 256 MiB, or use (c).
- klee100 (1.19 GiB free): fits.

**Drafter KV comes from the pool.** At FP8 with TP2 (4 heads × 128 × K,V) that is 1 KB per token per layer, 5 layers. A window of 2048 tokens in 4096-token blocks needs ≤ 2 blocks, about 40 MB per request per card; 9 requests use about 0.36 GB.
- This is roughly offset by the freed MTP QSA layer, about 0.69 KB per token of full context (from 3.31 GiB / 738k tokens / 7 attention layers).
- It is unmeasured: the KV-spec unification of a 5-layer SWA group with 4096 blocks is the main boot unknown, so gate it in a CPU kit.

## 4. Expected gain

### Published acceptance, confirmed
- **README (Flash-Next, SGLang, GB300 TP1, T=1.0, xhigh, max 4096 new tokens, block 4 so K=3, c1):**

  | Task | MTP | DFlash2 | Ratio |
  |---|---|---|---|
  | GSM8K | 3.08 | 3.26 | ×1.058 |
  | MATH-500 | 2.93 | 3.17 | ×1.082 |
  | HumanEval | 2.52 | 2.94 | ×1.167 |
  | MBPP | 2.65 | 3.06 | ×1.155 |
  | MT-Bench | 2.50 | 2.83 | ×1.132 |

  The mean is ×1.119, which confirms the report's ×1.06-1.17 (mean 1.12).
- README throughput: c8 1.31-1.48× and c32 1.43-1.73× over AR, against MTP's 1.11-1.25× and 1.12-1.24×.
- **Blog:**
  - Qwen3.8-27B at block 8 (vLLM `num_speculative_tokens` 7): MTP 4.28 vs DFlash2 4.80 (×1.12).
  - Conditional acceptance per position is **nearly flat**: Qwen3.5-4B MATH 88.3 % at position 0 and 86.5 % at position 14; MTP falls from 84.6 to 77.9 %.
  - The blog has **nothing on long context**, QSA or the training length. Every published number is at contexts of 4k or less.
  - The drafter attends to a 2048-token window of target features, so it is relative-position only. Long-context behaviour is unmeasured.
- Our Duck MTP (2.83; per-position conditionals 0.795 / 0.753 / 0.733) sits nearest to the README's MATH-500 MTP figure (2.93).

### Cost model, measured from our NB4S Q-defer trace (rank 2 = stage 1, 40 forwards, 26 MoE layers per forward = 23 target + 3 MTP)

| Micro-batch | Target 23 layers (busy) | Target lm_head + sampler | **MTP K3 drafting** | of which 3 × lm_head | MTP MoE |
|---|---|---|---|---|---|
| 7 requests (28 verify tokens) | 29.2 ms | 5.85 ms (+3.1 NCCL) | **10.53 ms + 2.84 NCCL = 13.4 ms** | 7.67 ms | 1.06 ms |
| 1 request (4 tokens) | 14.8 ms | 3.79 ms | **9.19 + 0.63 = 9.8 ms** | 7.49 ms | 0.28 ms |

- Each MTP step reads the full BF16 lm_head shard: 636 MB per card, 2.5-2.8 ms. It also does a full-vocab AllGather of about 0.74 ms at 7 requests.
- MoE per layer is 172 µs at 4 tokens and 645 µs at 28 tokens. Fitted on unique experts touched: about 69 µs plus 2.67 µs per expert.

**DFlash2 drafting estimate, per micro-batch on stage 1** (one pass whatever K is):

| Part | Time |
|---|---|
| lm_head via `compute_candidates` → `get_top_k_tokens`: one 636 MB read and only a top-16 gather, no full-logit AllGather (`logits_processor.py:253-280`) | 2.5-2.8 ms |
| 5 dense layers, 426 MB local weights | ≈ 1.6-2.0 ms |
| fc, 65 MB, plus the fused context-KV projection | ≈ 0.4 ms |
| conv projections | 0.12 ms |
| 10 TP all-reduces at about 64 µs each (measured kernel average) | ≈ 0.65 ms |
| SWA attention on 2048 tokens | ≈ 0.2-0.4 ms |
| selector | ≈ 0.1 ms |
| **Total** | **≈ 5.5-6.6 ms** |

**Corrections to the report:**
- "8-12 ms, net tick −3 to +5 ms" is too pessimistic. The tick changes by **−4 to −7 ms at K=3**.
- "K=7 break-even AL ≈ 3.9" is wrong. With this model the break-even is ≈ 3.1. Even assuming DFlash2 costs the same as MTP, it is ≈ 3.45.
- The extra stage-0 cost is negligible: five combine kernels at about 3.5 µs, plus a 50 % larger PP payload (2 × [T,2560] on top of [T,10240]).

**Tick model:**
- tok/s = N·AL / (2·tick).
- Calibration: 8 × 3.12 / (2 × 41 ms) = 304, against the measured QD 309.
- Today stage 1 is about 33 ms (MTP part about 11.5), stage 0 is 21.6 ms, and the steady tick is about 41 ms.
- DFlash2 K=3 gives stage 1 ≈ 27.5 ms, which is about the "rebalanced ideal" of 27.4, and a tick of ≈ 35.5 ms.
- K=4 adds about 2.3 ms of verify cost (tick ≈ 37.8 ms). The QSA ring stays at 8, so no ring patch is needed.
- K=7 adds about 9 ms on stage 1 (tick ≈ 44.5 ms); stage 0 rises to about 30.5 ms and does not bind. It needs the ring widening, because 4096 % 12 ≠ 0. The roozich port widens it to 16.

**Acceptance assumptions:** DFlash2 conditional acceptance flat at c:
- **c = 0.82** central (calibrated to K=3 AL ≈ 3.04 = 2.83 × ~1.07);
- 0.84 optimistic;
- **0.75** pessimistic, as a long-context haircut.

| K | AL (0.75 / 0.82 / 0.84) | Tick vs 41 ms | **W4 throughput × vs MTP K3** (pess / central / opt) |
|---|---|---|---|
| 3 | 2.73 / 3.04 / 3.14 | ×1.155 | 1.11 / **1.24** / 1.28 |
| 4 | 3.05 / 3.50 / 3.64 | ×1.085 | 1.17 / **1.34** / 1.40 |
| 7 | 3.60 / 4.42 / 4.70 | ×0.92 | 1.17 / **1.44** / 1.53 |

**W4 absolute, 9 streams:**
- Basis: Duck steady 225 tok/s without Q-defer (NB6). Q-defer gave +18-21 % on synthetic, so Duck with Q-defer ≈ 265-270 (unmeasured).
- K=3: ≈ 300 / 335 / 345.
- **K=4: ≈ 315 / 362 / 378.**
- K=7: ≈ 316 / 389 / 413.
- Synthetic 8k@8 (309 today): ≈ 345-470 by the same ratios.

**klee100 (16-24 streams; tick 55.8 / 68.7 ms; MTP AL 3.21):**
- MTP drafting at 8-12 requests per micro-batch is ≈ 13-14 ms (3 lm_head + 3 larger AllGathers). DFlash2 is ≈ 6.5-7.5 ms, so the tick improves ×1.11-1.13.
- K=3: @16 460 → ≈ 500 / 555 / 575; @24 562 → ≈ 600 / 665 / 690.
- **K=4:** @16 ≈ 535 / 615 / 640; @24 ≈ 645 / 740 / 770. Verify tokens per micro-batch go 32→40 and 48→60, and expert touch rises 12-17 %.
- **K=7: not recommended at ≥ 16 streams.** It means 64-96 tokens per micro-batch, so MoE time rises about 40-50 % and the gain falls to ≈ 1.0-1.35×. It also exceeds the 64-token capture maximum.
- **klee100-specific risk:** its token distribution is far from BF16 (rank-1 agreement 0.51-0.65), and the drafter conditions on low-bit tap features. MTP held up (AL 3.21), but DFlash2's acceptance on klee100 is a pure unknown: measure it first.

## 5. Kaggle Model layout
Upload the raw files with no zip or tar.

The drafter needs **no tokenizer**: `speculative.py:1319-1326` uses `target_model_config.tokenizer` unless the vocab is heterogeneous, and it is not. It also needs no `generation_config`.

Do **not** add a `mask_embedding.pt`: `qwen3_dflash.py` `_read_mask_embedding` would substitute it. A missing file in a local directory is safe offline: `repo_utils.py:389-390` returns None.

```
<owner>/qwen38-flash-next-dflash2 / transformers / <variation> / <ver>/
  config.json              # unchanged HF, 1,236 B (architectures DFlash2DraftModel, dflash_config)
  model.safetensors        # variation "bf16": HF original 1,206,140,008 B (sha256 d6723f2a…5e1e99)
                           # variation "bf16-embed": + embed_tokens.weight [248320,2560] BF16 ≈ 2,477,538,xxx B
  E27_DRAFTER_MANIFEST.json  # optional: file sha256s, embed source checkpoint + tensor sha, row-248077 norm
```
- 2-3 files, far under the 50-file cap.
- No names with `[ ] = , & @ +`. The `patches/` folder and the PNG are not needed.
- If the W4 and klee100 embed tensors hash identically, one "bf16-embed" variation serves both notebooks. Otherwise publish one variation per checkpoint.

## Risks and recommendations
1. **Use option (b) for the first test.** Kit asserts:
   - the embed header is BF16 [248320,2560];
   - the sha of row 248077 equals the target checkpoint's row, and its norm is > 0;
   - the W4 vs klee100 embed sha (one variation or two);
   - after load on PP rank 1, `has_own_embed_tokens` is True and the drafter linears use `UnquantizedLinearMethod` (the fused-KV path at `qwen3_dflash.py:533-538` depends on it).
2. **Silent-failure modes to gate with teacher-forced per-position acceptance:**
   - the tap order in the PP relay: remote [6,16] must come before local [26,36,46], or fc gets a permuted input;
   - the tap semantics. The roozich port uses the pure `mlp_hyper_connection.combine` output averaged over hc. incoai captures the materialized state at `normalized_input = norms.output`. This is likely the same thing, but unverified without SGLang source; if acceptance is well below MTP, A/B against the pre-combine state;
   - FP8 drafter KV with no scales. It inherits `fp8_e4m3`; `speculative_config.kv_cache_dtype` can set BF16 (`dflash/utils.py:34-41`).
3. **PP relay:**
   - `reserve_aux_intermediate_tensor_slots` makes [B,2560] slots (`eagle3_utils.py:86-94`), so the relayed taps must already be hc-averaged;
   - rank 1 expects `_aux_slot_base_cached` = bisect_right((6,16,26,36,46), 25) = 2 (`interfaces.py:1597-1600`);
   - the PP boundary at 25/23 is confirmed by the server log ("pipeline layers 0-25 / 25-48").
4. **Q-defer:** it changes only the sampled / draft-token broadcast. `nb4_pack_views` is generic in k. It is independent of the hidden-state and aux send. Re-verify the packed message at K≠3.
5. **Capture sizes:** the V2 runner sizes uniform decode graphs at multiples of K+1. Set `cudagraph_capture_sizes` explicitly to multiples of K+1 up to max_seqs_per_microbatch × (K+1):
   - W4, 9 streams: K3 → 36, K4 → 45, K7 → 72 (above the current maximum of 64);
   - klee100, `max_num_seqs` 32: cap at 24 or extend to 96-120.
6. **Order:**
   - W4: K=4 first (no ring patch, best central gain), then K=3 as the README-validated control, then K=7 with the ring patch.
   - klee100: K=4 first, then K=3; skip K=7 at ≥ 16 streams.
   - MTP K3 stays as the control in both.
7. **Feasibility-report items, confirmed:**
   - the allowlist (`config.py:1049-1058`, only mtp / ngram / ngram_gpu);
   - the +1 tap shift;
   - 2 taps on PP0;
   - the drafter only on the last rank;
   - the lm_head share;
   - QSA ring formula `4·ceil((4+K)/4)`, with an assert for K=5-8 at block 4096 (`qsa_cache.py:960-965`);
   - incoai's numbers and the 27B incompatibility.

   Refuted or corrected:
   - "silent garbage" is a loud RuntimeError;
   - the draft-pass cost and the K=7 break-even are too pessimistic (MTP drafting measured at 9.8-13.4 ms per micro-batch, against ≈ 5.5-6.6 ms for DFlash2).

**Scratch files from this check:**
- `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\dflash\hf\config.json`
- `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\dflash\hf\README.md`
- `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\dflash\hf\tree.json`
- trace scripts: `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\dflash\scripts\k3.py` and `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\dflash\scripts\k4.py`