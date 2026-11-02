## DFlash2 on Merlin v3.0.0 (TP2 × PP2 + EP, V2 runner, FULL graphs, RecoverSSM, FP8 QSA KV): verification report

**Verdict: GO to build the patch set and its Kaggle CPU kit. NO-GO for a GPU run** until the CPU kit has measured the drafter's KV packing (U1) and the memory on the PP1 cards (B9).

The engine path exists end to end in Merlin. But the report's patch list, as written (roozich seam + about 25-line relay + embedding appended on the GPU), would fail or perform badly on our setup:
- the seam does not support PP;
- the drafter KV would be FP8, which on sm_89 means an eager drafter;
- the drafter keeps an orphaned lm_head of 636 MB per card;
- on W4 there is no VRAM for the drafter.

Read-only throughout: I edited nothing, executed nothing, and ran no git. The roozich patch was only read and grep-matched.

Paths below are relative to `D:/ARC AGI 2/_scratch/e23/p2/merlin-src/vllm/` unless stated otherwise.

### Confirmed blockers, each with a fix sketch

**B1. Allowlist**
- `model_executor/models/config.py:1049-1058` accepts only {mtp, ngram, ngram_gpu}.
- Fix: add `"dflash"` only. Roozich also adds `"dspark"`, which we don't need.
- The anchor matches exactly once (I checked the counts with a read-only Python script).

**B2. No seam on the target**
- `Qwen4ExpModel` (`models/qwen4_exp/nvidia/model.py:567`) has no EagleModelMixin, and `Qwen4ExpForCausalLM` / `Qwen4ExpForConditionalGeneration` don't implement SupportsEagle3.
- So `eagle3_utils.py:15-20` raises. It is called at `model_runner.py:524-526`, because `use_aux_hidden_state_outputs` is True for dflash (`:386-393`).
- Fix: the roozich edits (b).
  - All 6 model.py anchors and the tail `]` match once.
  - The files have no CRLF.
  - The architecture is `Qwen4ExpForConditionalGeneration`, so the tail mount must cover both classes, as roozich does.

**B3. The roozich seam breaks under PP2. It is not "PP2 yes after item 3"; it fails three ways.**
- (a) It has no `supports_aux_hidden_states_over_pp = True`, so `eagle3_utils.py:44-50` raises ValueError (via `model_runner.py:528-531`).
- (b) PP0 returns `IntermediateTensors({"hidden_states"})` only (`nvidia/model.py:781-787`). But `reserve_aux_intermediate_tensor_slots` (`eagle3_utils.py:69-98`) adds `aux_hidden_states_0/1` buffers on rank 1. The rank-1 runner then copies every reserved key from the received dict (`model_runner.py:2441-2456`), which gives a KeyError.
- (c) Rank 1 sees only its 3 local taps, so roozich's own guard (`len(aux) != len(layers)`) raises RuntimeError.
- Engine side, nothing is missing:
  - the IntermediateTensors metadata is dynamic per step (`gpu_worker.py:1637-1683`);
  - `PPHandler.relay_aux_hidden_states` (`pp_utils.py:113-127`) only matters for middle stages, and `aux_hidden_state_relay_keys` is empty for PP=2.
- Fix sketch in `Qwen4ExpModel`, following the DeepSeek-V4 template (`models/deepseek_v4/nvidia/model.py:1287-1288, 1462-1527`):

```python
supports_aux_hidden_states_over_pp = True
# forward:
remote_aux = self.collect_remote_aux_hidden_states(intermediate_tensors) if self.aux_hidden_state_layers else []
aux = []
# in the layer loop, after layer(...):
if layer_idx + 1 in self.aux_hidden_state_layers:
    done = layer.mlp_hyper_connection.combine(hidden_states, block_output, injection) if block_output is not None else hidden_states
    tap = done.unflatten(-1, (hc, H)).mean(-2)          # [T, 2560], matches the reserved slot width (eagle3_utils.py:86-91)
    if self.use_hc_sequence_parallel: tap = sp_all_gather(tap)[:full_num_tokens]
    aux.append(tap)
# non-last rank:
return IntermediateTensors({"hidden_states": hidden_states, **self.pack_local_aux_hidden_states(aux)})
# last rank:
if self.aux_hidden_state_layers:
    aux = remote_aux + aux; assert len(aux) == len(self.aux_hidden_state_layers)
    return sample_hidden_states, aux
```

- Layout with 25/23 (`tools/merlin/profiles/flash-next.json:847-850`): taps (6, 16) are on PP0 and (26, 36, 46) on PP1. `bisect_right` gives a slot base of 2 (`interfaces.py:1584-1601`).
- `hc_combine` is pure: it allocates a new tensor (`nvidia/ops/hc.py:235-266`), so the pending tuple is untouched.
- HC-SP is off in our config:
  - `enable_hc_sp` defaults to False (`config/parallel.py:185`);
  - SP-MoE needs DP > 1 (`config/parallel.py:772-788`).
  - Keep the gather anyway for safety.

**B4. The drafter embedding on PP1 is uninitialised: silent garbage (confirmed)**
- The draft builds its own `VocabParallelEmbedding` (`qwen3_dflash.py:461-465`).
- The checkpoint has no `embed_tokens`, so the weight is mapped away (`:903-904, 918-919`).
- `maybe_share_target_embed` returns early under PP when the draft has no `has_own_embed_tokens` attribute (`eagle/utils.py:56-59`). The target embed on rank 1 is a PPMissingLayer anyway (`nvidia/model.py:600-610`).
- Roozich sets `has_own_*` on the TARGET classes. That is a no-op, because the attribute is read off the draft.
- Fix:
  - (a) Get the BF16 target `model.language_model.embed_tokens.weight` into the drafter: either append it to the drafter Kaggle Model, or load it from the target shard. In the W4 checkpoint it is in `model-00015`.
  - (b) For W4, put it in mapped pinned host memory after load (UVA, which graphs can capture), on the pattern of `model_runner.py:210-260` / `PinnedHostVocabParallelEmbedding` (`model_executor/layers/vocab_parallel_embedding.py:636-700`).
  - Do this in `spec_decode/dflash/utils.py`, NOT in `model_runner.py`, which would break nb4_ppack's strict MR_PRE sha (`nb4_ppack.py:74-75, 665-696`).
  - (c) Set `has_own_embed_tokens` on the draft class, so a missing embed raises (`eagle/utils.py:61-67`) instead of producing garbage.
- Note: `mask_token_id` 248077 is an unused padding row. The tokenizer's added tokens end at 248076 (I checked `tokenizer.json`; vocab 248320). Its row must be bit-identical to the original target embed. Gate this per checkpoint, especially klee100.

**B5. The draft-owned lm_head is not freed**
- `DFlashQwen3ForCausalLM` allocates a `ParallelLMHead`: 248320 × 2560 BF16 / TP2 ≈ 636 MB per card (`qwen3_dflash.py:790-799`).
- `load_dflash_model` only does `del dflash_model.lm_head` (`dflash/utils.py:59-70`).
- Merlin's own note (`spec_decode/speculator.py:122-130`) says such a head stays alive until a GC pass, because of a reference cycle.
- Sharing is correct; the memory is the problem: 636 MB is held per PP1 card, which is fatal against W4's 191 MiB free (NB1 R2 B3).
- Fix: after sharing, do what `_release_replaced_draft_head` does (`speculator.py:119-165`): `_parameters.clear()`, `gc.collect()`, `torch.accelerator.empty_cache()`. Or build the head on the meta device.

**B6. The drafter KV is FP8 by default, which makes the drafter eager on sm_89. The report says "FA2 ... already OK", which is wrong for our config.**
- The drafter inherits `--kv-cache-dtype fp8_e4m3`. `dflash/utils.py:35-42` overrides it only if the spec `kv_cache_dtype` is set.
- FA2 rejects FP8 KV off sm90/sm100 (`v1/attention/backends/fa_utils.py:306-311`, `flash_attn.py:433-441`).
- The next backend by priority is FLASHINFER (`platforms/cuda.py:168-176`). Its cudagraph support for non-causal attention on sm_89 is UNIFORM_SINGLE_TOKEN_DECODE (`flashinfer.py:1324-1334`).
- So the DFlash speculator falls back to eager (`dflash/speculator.py:126-143`).
- The FP8 scales would also be uncalibrated (1.0): the drafter has no quant config.
- Fix: `"kv_cache_dtype":"auto","attention_backend":"FLASH_ATTN"` in `--speculative-config`. These fields exist (`config/speculative.py:380-385, 426-430, 1820-1830`). The result is BF16 drafter KV on FA2 with UNIFORM_BATCH full graphs (`flash_attn.py:614-618`).
- Alternative: FP8 + `TRITON_ATTN`, which supports non-causal, FP8 and ALWAYS graphs (`triton_attn.py:92, 302-308, 327-328`), at an acceptance cost still to be measured.

**B7. QSA ring for K ≥ 5**
- Capacity = 4·ceil((4+K)/4): K = 3 → 8, K = 4 → 8, K = 5..8 → 12. 4096 % 12 = 4, so the assert at `qsa_cache.py:960-965` fires. `indexer_compress_ratio` = 4 in our W4 `config.json`.
- Fix: the roozich `qsa_ring_capacity` widening (its 4 anchors match once each). K = 7 becomes 16, and the LCM with 4096 is unchanged.
- The ring is read only from the spec block_size (`qsa_cache.py:784`, `:331`). Nothing else assumes the formula.
- `qsa.py:593` only uses 1+K for tile selection.

**B8. Capture sizes**
- Target decode graphs and DFlash drafter graphs (`decode_query_len` = 1+K, `dflash/speculator.py:145-150`) both round each capture size up to a multiple of 1+K. Sizes above `max_cudagraph_capture_size` are dropped (`cudagraph_utils.py:328-357`).
- The drafter has FULL_DECODE_ONLY and no PIECEWISE fallback (`dflash/speculator.py:139-143`). A step with more requests than the largest graph runs the whole drafter eagerly.
- Largest batch (requests) each list covers:

| | K = 3 | K = 4 | K = 7 |
|---|---|---|---|
| list [1..64] (W4 needs 9 workers) | 16 | 10 | **8 (too few)** |
| list [.., 80, 84] (klee100 needs 21 workers) | 21 | **16 (too few)** | **10 (too few)** |

- Fix: extend the lists to max_num_seqs × (1+K). For klee100 that is 105 (K = 4) or 168 (K = 7); for W4 it is 72 (K = 7).
- This costs extra graph memory on the tight PP1 cards.

**B9. PP1 memory (the report misses it)**
- Drafter per card at TP2 (computed from the safetensors header, all 82 tensors):

| part | how it is stored | size per card |
|---|---|---|
| decoder layers (qkv / o / mlp) | TP-sharded | 459 MB |
| `fc` 2560 × 12800 | replicated | 65.5 MB |
| selector codebooks 2 × 248320 × 256 | replicated | 254 MB |
| **total, without embedding** | | **≈ 0.78 GB** |
| embedding, if kept on the GPU | | +636 MB |
| orphaned lm_head, until B5 is fixed | | +636 MB |
| `draft_logits` fp32 (`dflash2/speculator.py:200-204`) | max_seqs × K × 248320 × 4 B | 63-72 MB at 21-24 seqs, K = 3 |

- Freed in exchange: the MTP module, which I did not measure. My estimate is about 0.6-0.7 GB per card: 512 experts × 3 × 2560 × 640 at W4 over EP2. Also the 21 MB `_mtp_hidden_buffer` (`nvidia/model.py:650-665`).
- W4 has 191 MiB free: it only works with a host embedding plus the B5 fix, or with a lower KV pin.
- klee100 has 1.19 GiB free: it works with both fixes.
- Measure the real numbers in the "Model loading took" line.

### Unknowns, to measure in the CPU kit before any GPU run

**U1. Drafter KV packing in the hybrid packed layout: a likely capacity cost. The report's mamba_hybrid "smaller drafter block" does not exist.**
- The sliding-window drafter block = the primary 4096 (`model_executor/layers/attention/attention.py:619-666`).
- Page per layer per card: 8 MiB in BF16, 4 MiB in FP8.
- Current stride ≈ 15.84 MiB per block (3,553,580,524 B / 214, NB1 R2 B3).
- `_get_packed_kv_cache_groups` (`v1/core/kv_cache_utils.py:1894-2028`) will either:
  - split the drafter into about 5 (BF16) or 3 (FP8) state groups, at about 2 blocks per request each (window 2048 at block 4096), so F goes from about 6.3 to about 12-16; or
  - widen the stride, so the pool falls well below 214.
- Gate: run Merlin's `get_kv_cache_configs` on the per-stage specs (the profile method) and read pool, F and groups.
- Fix lever: an env-gated smaller drafter sliding-window block. 1024 in BF16 gives a 10 MiB group of 5 layers inside one stride, about +4 blocks per request. `prefix_match_unit` 64 and the LCM 4096 stay valid (`kv_cache_utils.py:851-892`).

**U2. K = 7 grows the GDN / RecoverSSM / PLE state:** their shapes use `num_speculative_tokens` (`nvidia/model.py:985-1040`). So the mamba page, stride and pool change. Measure.

**U3. Acceptance on our W4 / klee100 targets** versus incoai's FP8 target.
- The tap semantics match: completed output of layer L, ids +1 (`eagle3_utils.py:115-121`), mean over the 4 HC streams, before any norm. This is the same as incoai's `capture.patch` and the DeepSeek template.
- PLE is at layer id [2] only, so there is no ambiguity at the taps.

**U4. The drafter group is not annotated as an eagle group** (`kv_cache_utils.py:2063-2119`): rule 1 covers MLA only, rule 2 covers mtp / deepseek only. So there is no trailing-block drop on prefix hits. Lossless; it might lower acceptance a little after reuse.

**U5. FA2 build for sm_89 in Merlin's vllm_flash_attn:** confirm with the "Using FLASH_ATTN" log line for the draft layers.

### Answers to your eight questions

**1. Config**
- Method resolution: `config/speculative.py:1357-1373`. Pass `method:"dflash"` explicitly, because the Kaggle path may not contain "dflash".
- The EAGLEConfig wrap keeps `DFlash2DraftModel` (`transformers_utils/configs/eagle.py:65-75`). The registry maps it at `registry.py:645`.
- `parallel_drafting` = True (`:1498-1499`).
- Slots: K (`:1972-1974`). Lookahead: K+1 (`config/vllm.py:661-664`).
- Async scheduling is allowed (`config/vllm.py:1604`).
- The V1 runner is rejected for candidate heads (`config/vllm.py:3025-3026`). We use V2, which is fine.

**2. Speculator**
- Dispatch: `spec_decode/__init__.py:27-43`.
- From the target it needs:
  - 5 aux states concatenated into fc 12800 (`qwen3_dflash.py:864-884`, `dflash/speculator.py:383-389`);
  - the target lm_head, shared (`dflash/utils.py:59-70`);
  - the target embed (B4);
  - `mask_token_id` from `dflash_config` (`spec_decode/utils.py:187-198`);
  - a non-causal, sliding-window-capable backend (B6).
- The checkpoint shapes all match Merlin's DFlash2 modules: conv `base_kernel` [2, 2, 2560], `kernel_projection` 640 = 2·2·160, selector rank 256, top-k 16, `fc` [2560, 12800].
- The conv block is 1+K (`qwen3_dflash2.py:256`). K = 3 equals incoai's SGLang serving setting.

**3. Seam:** see B2 and B3. The drafter expects the mean over streams, of the completed (post-combine) state, before any norm (`capture.patch`).

**4. PP2 and Q-defer: no conflict**
- Q-defer and Q-pack touch only the `pp_broadcast` sibling communicator (`pp_utils.py:97-100`). The aux tensors ride the p2p `isend_tensor_dict` on the device group (`gpu_worker.py:1677-1683`).
- The wire format of the broadcast is unchanged: drafts [n, K] int64, `nb4_pack_views` keyed by K.
- The order (broadcast then broadcast_drafts) is the same as with MTP (`model_runner.py:2677-2820`).
- PP0's output buffers are reused only after the send-handle wait (`gpu_worker.py:1631-1635`).
- Async scheduling with the relay adds no host sync, so it is no longer an unknown.
- Constraint: the DFlash patch must not touch `pp_utils.py`, `model_runner.py` or `gpu_worker.py`, because of nb4_ppack's strict shas.

**5. Drafter KV:** see U1 (block 4096 packing, F = 6.3 at present) and B6 (BF16 versus FP8). Sliding window 2048: FA symmetrises it for non-causal (`flash_attn.py:583-592`), and the AOT schedule is disabled for windowed drafters (`dflash/speculator.py:204-215`).

**6. Graphs:** see B8.
- The draft query pass is always uniform (1+K per request), so it runs in a FULL graph even in mixed or prefill steps.
- Only the context-KV rows beyond num_reqs·(K+1) run eagerly, and that is one fused GEMM (`dflash/speculator.py:477-493`).
- So "draft prefill runs eager" (the report) is imprecise.

**7. RecoverSSM / GDN verify:** DFlash2 produces a CHAIN.
- `_selector_walk_kernel` walks one path over the 16 candidates and writes K tokens per request (`dflash2/speculator.py:15-75`).
- The 16 candidates only shape the fp32 proposal used by the probabilistic rejection (−inf elsewhere).
- RecoverSSM depends only on num_spec > 0 and the architecture (`config/vllm.py:3353-3370`), so the chain-only GDN verify is fine.
- Adaptive verification must stay off: it requires PP = 1 (`config/vllm.py:3053+`).

**8. QSA ring:** K = 3 and K = 4 are fine (capacity 8); K = 5-8 assert without the widening (B7).

### Where the feasibility report is wrong or incomplete
1. "FA2 non-causal on sm_89 already OK": false with our fp8_e4m3 KV (B6).
2. "The lm_head is shared correctly": shared, but the draft-owned head leaks about 636 MB per card (B5).
3. "PP2: yes, after item 3": the roozich patch actively raises under PP. Three fixes are needed (B3).
4. "Hybrid state handles a drafter group with its own smaller block" (mamba_hybrid.py:158-166): that is a comment in `add_request` seeding. The sliding-window drafter block is 4096, and the packing cost is unmeasured and likely large (U1).
5. "Draft prefill runs eager": only the context-KV tail does.
6. The appended-embedding fix costs 636 MB per card on the GPU, which does not fit W4. Use the host (UVA) table.
7. Missing entirely: the PP1 memory budget (B9), the capture-size rounding for K = 4 / 7 (B8), the fp32 `draft_logits` buffer, and the replicated codebooks / fc (254 + 65.5 MB per card).
8. "Unknown: async scheduling with the PP relay": resolved, no issue.
9. Verified as correct: every file:line in Q1, the allowlist, the formula and K threshold of item 5, the item-4 diagnosis, the AL ratio arithmetic (×1.06-1.17, mean 1.12), and that the roozich anchors match Merlin.

### Build order
1. config.py allowlist.
2. model.py: seam with PP pack / collect and the class flag.
3. qsa_cache.py: ring widening.
4. dflash/utils.py: embed load to host UVA, `has_own_embed_tokens` guard, lm_head release.
5. Speculative-config: `{"method":"dflash","model":…,"num_speculative_tokens":3,"draft_sample_method":"probabilistic","kv_cache_dtype":"auto","attention_backend":"FLASH_ATTN"}`.
6. Extended capture lists.

CPU kit gates:
- every patch applies;
- the config resolves with aux layers (6, 16, 26, 36, 46) and fc 12800;
- the reserved slot keys exist on rank 1;
- the embed is loaded and row 248077 equals the target's;
- `get_kv_cache_configs` pool / F / groups are within budget (U1);
- K = 7 ring capacity is 16.
