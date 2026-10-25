# Per-decode-step costs other than weight GEMMs and NCCL all-reduce: Qwen3.8-27B + DFlash2, SGLang 0.5.21, 4x L4 TP4, 12 requests x 8 verify positions

Research date 2026-10-23. Read-only.

Labels:
- **[code]**: read from local SGLang 0.5.21 source.
- **[measured]**: a published or logged measurement.
- **[claimed]**: an author or vendor statement without data.
- **[computed]**: my arithmetic from model dimensions.
- **[inference]**: my reasoning, not verified.

Paths:
- `SGL/` = `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang/`. Line numbers refer to that tree.
- `CFG27` = `D:/ARC AGI 2/_scratch/session_1c6db7aa/e11_03_harness/tok/config.json`, the Qwen3.8-27B INT4 config.
- `CFGD2` = `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/dflash2_config.json`, the DFlash2 HF config mirror.
- `MEAS` = `D:/ARC AGI 2/research_notes/Qwen 27B দ্রুততম সার্ভিং সেটআপ/our_measurements.md`.
- `NCCLN` = `D:/ARC AGI 2/research_notes/৪xL4 ইঞ্জিন বটলনেক যাচাই/nccl_p2p_l4.md`, the parallel note on all-reduce.

Bandwidth assumption for all [computed] times: the L4 is rated 300 GB/s GDDR6 ([GCP GPU machine types](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines), via NCCLN). Times are given at 300 GB/s (best case) and 250 GB/s (assumed achievable). I found no published measured copy bandwidth for the L4 (see Gaps).

Model dimensions used (from CFG27 `text_config`):
- hidden 5120; 64 layers, of which 48 are GatedDeltaNet (GDN) and 16 are full attention (`full_attention_interval` 4).
- Full attention: 24 Q heads, 4 KV heads, head_dim 256.
- GDN: 16 key heads and 48 value heads, both 128-dim, conv kernel 4.
- vocab 248,320; `tie_word_embeddings` false.
- `lm_head` is in the quantization ignore list, so it is BF16.

Per GPU at TP4: 62,080 vocab rows, 1 KV head, 12 GDN value heads.

---

## Q1. Code paths in 0.5.21: target lm_head, logits materialization, sampling per position, TP all-gather

### Takeaway
**All 96 verify positions (12 x 8) go through the target lm_head every step.** Each rank computes its 62,080-column BF16 shard, an NCCL all-gather builds the full **[96 x 248,320] BF16 tensor (47.7 MB) on every rank**, and it is then copied into an **FP32 buffer of 95.4 MB**.

Sampling verification (temperature 1.0, top-k 20, top-p 0.95) then runs on **every position, not only accepted ones**. It is redundant on all 4 ranks. The DFlash2 drafter **reads the target lm_head shard a second time** to get its top-16 candidates.

On a box without NVLink the logits all-gather falls back to NCCL. At our measured SHM bandwidth that alone is plausibly **~7–8 ms per step**.

### Cited Findings
**Verify forward and logits**
- **Verify feeds all 96 tokens through the target.** The DFLASH worker builds `verify_input_ids = draft_tokens.reshape(-1)` with `draft_token_num = block_size` and calls `target_worker.forward_batch_generation(..., is_verify=True)` [code] — `SGL/srt/speculative/dflash_worker_v2.py:2715-2769`.
- **Verify skips the normal sampler.** `tp_worker` returns before sampling ("Skip sampling; spec_v2 worker fires its own publish post-verify"), so verify sampling never goes through `layers/sampler.py` [code] — `SGL/srt/managers/tp_worker.py:656-658`.
- **No pruning before the lm_head.** In `TARGET_VERIFY` mode, `_get_pruned_states` keeps all hidden states (`pruned_states = hidden_states` unless a draft-extend select index exists). `forward()` then calls `_get_logits(pruned_states, ...)` on every row [code] — `SGL/srt/layers/logits_processor.py:622-660`, `:589-602`.
- **The lm_head is a plain matmul against the local vocab shard.** `_compute_lm_head` runs `torch.matmul(hidden_states.to(lm_head.weight.dtype), lm_head.weight.T)` for a dense BF16 head, unless `--enable-fp32-lm-head` is set [code] — `SGL/srt/layers/logits_processor.py:955-1017`.
- **Vocab padding is 64** (`DEFAULT_VOCAB_PADDING_SIZE = 64`) [code] — `SGL/srt/layers/vocab_parallel_embedding.py:47`. 248,320 / 4 = 62,080 rows per rank, so no padding [computed].

**The all-gather**
- **The gather is on whenever TP > 1.** `do_tensor_parallel_all_gather = not skip_all_gather and tp_size > 1`. It goes through `MultimemAllGatherer`, which uses NVLS multicast if available and otherwise falls back to `tensor_model_parallel_all_gather(x, dim=-1)` [code] — `SGL/srt/layers/logits_processor.py:462-488`, `:913-926`; `SGL/srt/distributed/device_communicators/triton_symm_mem_ag.py:449-521`.
- **The multicast path turns itself off when `multicast_ptr == 0`** ("No multicast for this world size / arch… Fall back to NCCL") [code] — `SGL/srt/distributed/device_communicators/triton_symm_mem_ag.py:523-556`. L4s on GCP G2 are PCIe Gen4 with no NVLink [claimed] ([GCP GPU machine types](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines), via NCCLN), so NCCL is used [inference].
- **The NCCL fallback adds a full copy.** `GroupCoordinator.all_gather` allocates `[world*96, 62080]`, runs `all_gather_into_tensor`, then does `reshape → movedim(0, dim) → reshape`. For `dim=-1` that is a full extra copy of 47.7 MB [code] — `SGL/srt/distributed/parallel_state.py:1374-1444`.
- **`--enable-tp-lm-head-all-to-all` does not help pure TP.** Its guard needs `logits.shape[0] == local_rows * tp_size`, a DP-attention layout [code] — `SGL/srt/layers/logits_processor.py:1075-1112`.

**FP32 copy and sampling verification**
- **Logits are always converted to FP32.** `_copy_logits_to_buffer` copies into `next_token_logits_buffer` (asserts `dtype == torch.float`) or calls `logits.float()` [code] — `SGL/srt/layers/logits_processor.py:1139-1158`.
- **DFlash2 sampling goes through the selector accept path.** For a sampling batch with the draft graph replayed, the worker sets `_selector_sample` and `_accept_block` calls `_selector_sampling_accept`. That function scatters the selector's sparse q into a persistent dense buffer of shape `[bs, 7, 248320]` FP32 (83.4 MB at bs 12) and calls DSpark `accept_sampling` [code] — `SGL/srt/speculative/dflash_worker_v2.py:2618-2630`, `:2123-2147`, `:1463-1503`.
- **What `accept_sampling` does at top-k 20 / top-p 0.95** [code] — `SGL/kernels/ops/speculative/dspark/dspark_accept.py:87-157`; `SGL/srt/speculative/dflash_utils.py:1130-1197`:
  - builds full target probabilities with `build_dflash_verify_target_probs`;
  - `scaled_logits = logits / temperature`: a new [96, 248320] FP32 tensor;
  - `torch.topk(scaled_logits, k=20)` over all 96 rows;
  - softmax, then top-p renorm on [96, 20];
  - `target_probs = torch.zeros_like(scaled_logits)`, then scatter: another dense 95 MB tensor;
  - then `chain_speculative_sampling_triton`.
- **The `.item()` host sync is avoided.** `max_top_k` and `uniform_top_k_value` come from host-side request params in `DFlashDraftInputV2` [code] — `SGL/srt/speculative/dflash_info_v2.py:156-179`.
- **The rejection kernel runs one program per request.** Grid = bs = 12. On a rejection it loops twice over the full vocab of p and q (`BLOCK_V` 4096) [code] — `SGL/kernels/ops/speculative/reject_sampling.py:6-170`, `:173-218`.
- **Sampling is replicated on all ranks.** Every rank runs the whole accept path. Rank 0's `accept_len` and `bonus` are then broadcast (`SGLANG_SPEC_TP_SYNC` default `"all"` → `tp_group.broadcast`): two eager NCCL broadcasts per step [code] — `SGL/srt/environ.py:1393`; `SGL/srt/speculative/spec_tp_sync.py:116-119`; `SGL/srt/speculative/dflash_worker_v2.py:2145-2146`.

**Penalties, grammar, logprobs**
- **Penalty and grammar path:** `apply_dflash_verify_logits_adjustments` costs nothing without penalties or bias.
  - With penalties under overlap, it broadcast-adds `acc_linear_penalties` [bs, vocab] into the [96, 248k] FP32 logits.
  - Without `acc_linear_penalties` or with a grammar mask, it allocates a dense `linear_penalty` [bs, vocab] [code] — `SGL/srt/speculative/dflash_utils.py:219-300`.
  - Grammar masks are built per chain position only if `batch.has_grammar` [code] — `SGL/srt/speculative/dflash_worker_v2.py:2773-2792`.
- `compute_spec_logprobs` runs only if `return_logprob` [code] — `SGL/srt/speculative/dflash_worker_v2.py:2839-2845`.

**The drafter's own lm_head use**
- **The drafter reuses the target lm_head.**
  - `DFlash2DraftModel.lm_head` is set to the target's: "The draft has no head of its own".
  - `compute_candidates → candidate_topk` runs `torch.matmul(hidden, weight[:num_org].T)` on the **local target shard** for 12 x 7 = 84 rows.
  - A FlashInfer radix `top_k` (k = 16) per shard follows, then an all-gather of only K values and ids per rank.
  - Code comment: "O(tp*K) instead of O(vocab) gather bandwidth" [code] — `SGL/srt/models/dflash.py:89-152`, `:1199-1237`; `SGL/srt/speculative/dflash_worker_v2.py:830-847`.
- **SGLang's own comments on top-k cost:**
  - "The selector's largest single cost: it reads the whole logits tensor" [code comment] — `SGL/srt/models/dflash.py:64-68`.
  - Without FlashInfer, "the DFlash2 selector falls back to torch.topk, which roughly halves end-to-end throughput on a large vocabulary" [claimed, code comment] — `SGL/srt/models/dflash.py:1092-1098`.

**Our own measured memory**
- Target weights 4.62 GB per GPU, drafter 1.49 GB per GPU [measured, E11-03 D log] — MEAS (section "D memory per GPU").
- Intermediate SSM 1.83–1.97 GB per GPU [measured] — MEAS (caps table).

### Inferences
**Sizes (computed)**
- lm_head shard: 62,080 x 5,120 x 2 B = **635.7 MB per GPU**, read once for verify and again for draft candidates. That is **1.27 GB per step, about 4.2–5.1 ms**, of which about 2.1–2.5 ms is probably not in the "~16 ms of weights" figure (see Q5).
- Per-rank logits [96, 62,080] BF16 = **11.9 MB**. Gathered: **47.7 MB BF16**. After the FP32 copy: **95.4 MB**. The question's "48–95 MB" range is right: 48 MB moves over NCCL and 95 MB is materialized.

**All-gather time (inference)**
- NCCL busbw for all-gather = algbw x (n−1)/n ([nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md), via NCCLN).
- If SHM all-gather busbw ≈ the 4.6–4.8 GB/s implied by our 64 MB all-reduce in 20.9 ms (NCCLN Q3), then t ≈ 0.75 x 47.7 MB / 4.7 GB/s ≈ **7.6 ms per step**.
- Range: about 10 ms at 3.5 GB/s busbw; about 3.6 ms if P2P gives about 10 GB/s.
- This is in addition to the all-reduce budget. One 48 MB all-gather moves as much data as about 35–50 of the 0.98 MB hidden-state all-reduces.

**Post-gather memory passes (computed, at 300 / 250 GB/s)**

| Pass | Traffic | Time |
|---|---|---|
| movedim copy | 95 MB | 0.32–0.38 ms |
| FP32 buffer copy | 143 MB | 0.48–0.57 ms |
| temperature divide | 191 MB | 0.64–0.76 ms |
| `torch.topk` | ≥95 MB, possibly several radix passes | 0.3–1.5 ms |
| dense `zeros_like` | 95 MB | 0.32–0.38 ms |
| rejection kernel (12 CTAs only) | — | 0.2–0.5 ms |
| **Total** | | **≈2.3–4.1 ms** |

Part of this cost is spent repeating identical work on all 4 ranks.

**Fixes (not in 0.5.21)**
- Compute verify sampling vocab-parallel: local top-20 per shard plus an all-gather of 20 values and ids, as `candidate_topk` already does for the drafter.
- That would remove both the 48 MB all-gather and the dense FP32 passes. With top-k 20, exact target probabilities only need each rank's local top-20 and the top-p renorm on the merged 80 [inference].

### Gaps
- No nsys trace of the target-verify CUDA graph, so the logits all-gather time is inferred, not measured. NCCL's SHM all-gather bandwidth on Kaggle's 4xL4 is unmeasured; a single `all_gather_into_tensor` of 11.9 MB per rank would confirm it.
- `torch.topk` time for [96, 248,320] FP32 on sm89 was not found in any source.

---

## Q2. FR-Spec / hot-vocab: Franzen's 64k map, SGLang support for DFLASH, reported speedups

### Takeaway
**SGLang 0.5.21 does not apply `--speculative-token-map` to DFLASH.** The flag is read only by the EAGLE-family workers (`eagle_worker_v2`, `standalone_worker_v2`). The DFLASH worker never references it, so it is silently ignored.

Franzen's 64k FR-Spec map was used with **NEXTN/MTP on Qwen3.8-Flash-Next (single RTX PRO 6000, Pennyroyal fork)**, not with DFlash. FR-Spec reports **1.12x over EAGLE-2** at 32k (1.09x at 64k). It applies only to the drafter's head; verification keeps the full vocab.

For DFlash2 on our setup a hot-vocab could only shrink the **second** lm_head read (the drafter's candidate top-16). At 64k of 248k rows that saves about 470 MB per GPU per step, **≈1.6–1.9 ms (about 1.5% of the step)**.

### Cited Findings
**Franzen's use of FR-Spec**
- **The write-up does not mention FR-Spec** [read fully]. Its serving changes are low-M BF16 GEMM, a spec-state memory budget fix, a Marlin scale-dtype fix, a prefix-cache patch and checkpoint prefetch. Settings: "Speculative steps 3", temperature 0.7, top-p 0.95, top-k 20, 10 streams on a single RTX PRO 6000 — `D:/ARC AGI 2/_scratch/ext/dafr/WRITEUP.md` L35-60.
- **The notebook launcher does use it.** It has `FRSPEC=True` and passes `--speculative-token-map <hot_tokens_64k.pt>` (sha-pinned `becfa41d…`, tokenizer-sha-checked) together with `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4` and an INT4 MTP draft — `D:/ARC AGI 2/_scratch/ext/dfranzen_cells_4_12_16.txt` L161-180, L441-457.
- **The map comes from the Pennyroyal fork** (`configs/pennyroyal/frspec/flash-next-64k.pt`, copied as `hot_tokens_64k.pt`) — `D:/ARC AGI 2/_scratch/ext/dafr/serving/build_bundle_pennyroyal.sh` L72, L1408, L1576-1580.

**SGLang 0.5.21 support (code)**
- `speculative_token_map` is described as "The path of the draft model's small vocab table" [code] — `SGL/srt/arg_groups/fields/spec.py:119-122`.
- **EAGLE family:** it is loaded in `EAGLEWorkerV2.init_token_map`. The draft lm_head is sliced to `head.data[self.hot_token_id]` and draft top-k ids are remapped with `self.hot_token_id[topk_index]` [code] — `SGL/srt/speculative/eagle_worker_v2.py:240`, `:312-364`, `:704-705`, `:834-835`. The standalone worker calls the same `init_token_map` — `SGL/srt/speculative/standalone_worker_v2.py:126`.
- **DFLASH:** a grep for `token_map` or `hot_token` in `dflash_worker_v2.py` and `models/dflash.py` returns nothing. `speculative_hook.py` rejects a token map only for the IQuest MTP draft (`"MTP uses its own full-vocabulary embedding and head."`) [code] — `SGL/srt/arg_groups/speculative_hook.py:934-935`.
- **DFlash2's selector codebooks are full-vocab** (`[vocab_size, state_rank]` = [248,320, 256] each, replicated on every rank) [code] — `SGL/srt/models/dflash.py:1077-1107`.
- **SGLang docs** list FR-Spec / `--speculative-token-map` as an EAGLE-2 option; the EAGLE-3 entry lists no token map [claimed, docs] — [SGLang speculative decoding docs](https://docs.sglang.io/advanced_features/speculative_decoding.html).

**FR-Spec paper (arXiv 2502.14856)**
- **Method:** it restricts the draft search to a frequency-ranked vocabulary subset and cuts LM-head overhead by 75%. Average 1.12x over EAGLE-2 [claimed, abstract] — [arXiv 2502.14856](https://arxiv.org/abs/2502.14856).
- **Profiling:** the LM head is 49% of EAGLE-2 drafting compute on Llama-3-8B (128k vocab); with softmax, 62%; transformer layer 33% [measured] — [arXiv 2502.14856 HTML](https://arxiv.org/html/2502.14856v2).
- **Speed:** Llama-3-8B, T = 0, their C/CUDA framework, 1x A800 [measured] — [arXiv 2502.14856 HTML](https://arxiv.org/html/2502.14856v2):

  | Draft vocab | tok/s | vs EAGLE-2 |
  |---|---|---|
  | full (EAGLE-2) | 180.99 | 1.00x |
  | 64k | 196.86 | 1.09x |
  | 32k | 201.87 | 1.12x |
  | 16k | 196.69 | 1.09x |
  | 8k | 185.98 | 1.03x |

- **Acceptance length** (from 3.89 at full vocab, SlimPajama statistics): 64k 3.80 (−2.3%), 32k 3.63 (−6.7%), 16k 3.40, 8k 3.13 [measured] — same source.
- **Versus other frameworks:** 1.42x over SGLang-EAGLE-2 and 1.82x over HF-EAGLE-2 (part of that comes from their faster base framework: their EAGLE-2 is 1.28x faster than SGLang's) [measured] — same source.
- **Verification is unchanged.** The target verifies with the full vocabulary, so the output distribution is preserved [claimed] — same source.
- **Related work:** DynaSpec and Vocabtrim-style shortlists trade drafter vocab size against acceptance. The cost of a large vocab "burdens the drafter" more than the target [claimed] — [DynaSpec arXiv 2510.13847](https://arxiv.org/pdf/2510.13847); [RDK arXiv 2506.03206](https://arxiv.org/pdf/2506.03206).

### Inferences
- **Why it paid off for Franzen** [inference]: on a single GPU with an MTP head over a 248k vocab and 3 draft steps, the drafter does 3 full lm_head passes per step, the setting FR-Spec targets.
- **Why it pays little here** [inference]: our DFlash2 drafter does **one** candidate pass per step (84 rows). The verify-side lm_head, all-gather and sampling (Q1) still need the full vocab under FR-Spec semantics.
- **Upper bound for us:** at 64k hot rows, 635.7 MB → about 164 MB per GPU, saving about 470 MB = **1.6–1.9 ms** per step.
- **To implement:** slice the target lm_head copy used in `candidate_topk` and remap ids. The selector codebooks are indexed by global id, so they need no change. There is a risk to acceptance length for tokens outside the hot set (FR-Spec measured −2.3% at 64k).

### Gaps
- The Pennyroyal fork's FR-Spec code was not read (only the build script), so I can't say whether it extended the token map beyond EAGLE/NEXTN.
- I found no published FR-Spec result for block-diffusion drafters.

---

## Q3. Published per-kernel timings: lm_head plus sampling with large vocabularies; FlashInfer sampling; issues

### Takeaway
I found **no published per-kernel timing of lm_head plus sampling for 128k–248k vocabularies at batch 64–128 on Ada/L4**. The evidence that exists is partial:
- FR-Spec: LM head is 49% of EAGLE-2 drafting time at 128k.
- SIMPLE (arXiv 2512.00719): sampling is 20–38% of decode time on large-vocab models, rising about 10 points as TP grows from 2 to 8, measured on L40 / H100 / B200.
- Qrita: sort-based top-k/top-p at batch 1024 x 128k = 16.5 ms vs 0.92 ms optimized.
- FlashInfer: its sorting-free kernels cut vLLM sampling time by more than 50% on H100.

SGLang's own code says a torch.topk fallback for the DFlash2 selector "roughly halves end-to-end throughput". For our step, arithmetic (Q1) is more reliable than any of these.

### Cited Findings
**Papers**
- **SIMPLE (arXiv 2512.00719)** [measured] — [arXiv 2512.00719](https://arxiv.org/html/2512.00719v1):
  - sampling takes 20–38% of time on large-vocab models or with constrained decoding;
  - the share rises about 10% from TP2 to TP8;
  - sampling-induced pipeline bubbles are 22–40% (Qwen-2.5-72B, 8xH100);
  - baselines vLLM 0.10.1 and SGLang 0.5.2; GPUs include L40 (PCIe 4.0);
  - cross-rank reconciliation for a global sampling decision adds "several milliseconds" [claimed].
- **Qrita (arXiv 2602.01518)**, k = 50, p = 0.9, vocab 128,256, batch 1024 (GPU not named in that table) [measured] — [arXiv 2602.01518](https://arxiv.org/html/2602.01518):
  - naive sort-based PyTorch: 16.49 ms;
  - sort-based with truncation: 2.10 ms;
  - Qrita: 0.92 ms.
  - The paper says FlashInfer returns one sampled index, not a candidate set, which "limits some uses such as speculative decoding verification" [claimed].
- **FlashInfer sorting-free sampling blog (2025-03-10)** — [FlashInfer blog](https://flashinfer.ai/2025/03/10/sampling.html):
  - "reduce the overall sampling time by more than 50%" in vLLM on 1xH100 [measured, no per-kernel µs given];
  - algorithm: inverse-transform plus (dual-pivot) rejection sampling, worst-case O(log(1/ε)) rounds [claimed].
- **FlashInfer `top_k` docs:** "drop-in replacement for torch.topk with better performance for large tensors (vocab_size > 10000)"; the default is unsorted [claimed] — [flashinfer.top_k docs](https://docs.flashinfer.ai/generated/flashinfer.top_k.html).
- **UNIQUE paper:** its own radix top-k is up to 2.0x faster than FlashInfer and up to 4.8x faster than torch.topk (H100, k = 64) [measured, different shape] — [arXiv 2605.27740](https://arxiv.org/pdf/2605.27740).

**SGLang 0.5.21 itself**
- The verify path (`build_dflash_verify_target_probs`) uses `torch.topk`, not FlashInfer, for the k = 20 target probabilities [code] — `SGL/srt/speculative/dflash_utils.py:1165-1166`.
- The drafter selector uses FlashInfer `top_k(scores, k, sorted=True, deterministic=True)` when importable [code] — `SGL/srt/models/dflash.py:58-68`.
- The selector's warning about the torch.topk fallback halving throughput [claimed, code comment] — `SGL/srt/models/dflash.py:1092-1098`.

**DFlash2's own throughput at T 1.0 / top-p 0.95 / top-k 20, SGLang on 1x H200** [measured] — `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/dflash2_README.md`:

| Concurrency | Speedup vs AR | Example (MT-Bench) |
|---|---|---|
| 1 | 2.67–3.43x | — |
| 8 | 2.27–2.85x | — |
| 32 | 1.01–1.45x | 1,507 → 1,525 tok/s |

The gain shrinks sharply with concurrency, which is consistent with verify-side per-position costs growing with batch [inference].

- **Field report, same model shape** (hidden 5120, 64 layers, 5-layer DFlash2, TP4 on 4x RTX 5060 Ti PCIe Gen3 x8, vLLM 0.30.0): C8 default-temperature decode 370–406 tok/s. No step breakdown [measured] — [HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254).

### Inferences
- [inference] Scaling the Qrita sort-based number: [96 x 248k] is 23.8 M elements vs 131 M, about 5.5x fewer. Even a naive sort pass would be about 3 ms on their GPU, and an L4 has roughly an order of magnitude less bandwidth than an H100-class part.
- [inference] SGLang uses a top-k-first path (topk 20, then softmax on 20) rather than a full sort, so the remaining cost is mostly the dense FP32 passes in Q1 (≈2.3–4.1 ms), not sorting.
- [inference] SIMPLE's "share rises with TP" matches our case: each TP rank repeats full-vocab sampling while weight time per rank shrinks.

### Gaps
- No L4/Ada kernel timings found for lm_head GEMM at M = 96, N = 62k, K = 5120, or for FlashInfer `top_k` / `top_k_renorm_prob` on sm89.
- No SGLang GitHub issue found that quantifies verify-sampling overhead for DFLASH.

---

## Q4. DFlash2 drafter architecture and per-step compute

### Takeaway
The drafter is a **5-layer Qwen3-style block-diffusion model, hidden 5120**:
- Attention: 32 Q heads, 8 KV heads, head_dim 128, all **sliding-window 2048**, non-causal.
- MLP: intermediate 17,408.
- Two-tap grouped dynamic convolutions around each sublayer.
- Fusion: an FC layer from 5 tapped target layers [5, 19, 33, 47, 61] (25,600 → 5,120).
- A low-rank (256) candidate selector over the top-16 tokens per position.

It has **no embedding or lm_head of its own**: it borrows the target's (vocab 248,320). Resident per GPU it is about 1.45 GB by my count (1.49 GB measured).

Per step at 12 x 8 it reads about **1.19 GB of its own weights per GPU** (layers TP-sharded, FC and conv projections replicated) plus the **635.7 MB target lm_head shard**. That is **≈6–7.5 ms of pure memory time** plus 10 drafter all-reduces.

### Cited Findings
**Config and paper**
- **Config:** `architectures ["DFlash2DraftModel"]`, `num_hidden_layers` 5, `hidden_size` 5120, `intermediate_size` 17408, 32 attention heads, 8 KV heads, head_dim 128 [code/config] — CFGD2.
  - `layer_types` all `sliding_attention`, `sliding_window` 2048, `is_causal` false.
  - `dflash_config`: `block_size` 8, `conv_kernel_size` 2, `conv_group_size` 16, `selector_rank` 256, `selector_top_k` 16, `mask_token_id` 248070, `target_layer_ids` [5, 19, 33, 47, 61], `num_target_layers` 64.
  - `vocab_size` 248320, `tie_word_embeddings` false, dtype bfloat16.
- **DFlash paper:** 5 layers by default. The drafter **shares the target's token embedding and LM head (frozen)**. Features from 5 target layers are concatenated, projected, and injected into every draft layer's K/V. A whole block is drafted in one pass [claimed] — [arXiv 2602.06036](https://arxiv.org/html/2602.06036v1).
- **DFlash 2 blog** [claimed] — [inco.ai DFlash 2 blog](https://inco.ai/blog/dflash2/):
  - the selector keeps the top 16 per position;
  - score S_t(a,b) = U_t(b) + ⟨A(a)⊙H(h_t), B(b)⟩ with 256-dim A/B embeddings; "no extra backbone or LM-head pass";
  - two-tap dynamic depthwise conv;
  - Qwen3-4B costs: selector +2.0M params / +0.6% cycle latency; conv +16.5M / +0.7%; together +1.3%;
  - Qwen3.8-27B acceptance length 4.80 vs MTP 4.28.

**Code**
- **Attention is TP-sharded** (`QKVParallelLinear`, `RowParallelLinear`) with per-head Q/K RMSNorm and RoPE. The sliding window passes through `RadixAttention(sliding_window_size=...)`; `is_causal` false maps to `ENCODER_ONLY` [code] — `SGL/srt/models/dflash.py:155-320`.
- **The MLP is TP-sharded** (`MergedColumnParallelLinear`, `RowParallelLinear`) [code] — `SGL/srt/models/dflash.py:426-462`.
- **Replicated modules:**
  - the conv `kernel_projection` is a plain `nn.Linear(hidden, 2*taps*num_groups)`, applied before and after attention and MLP [code] — `SGL/srt/models/dflash.py:480-536`, `:538-611`;
  - `fc` is a plain `nn.Linear(5*5120, 5120)` [code] — `SGL/srt/models/dflash.py:705-709`.
- **Target features are projected per committed token.** `project_target_hidden` (FC plus hidden_norm) feeds the draft KV materialization for all 96 verify tokens each step. A fused KV materialize writes K/V with prefix-valid commit lengths [code] — `SGL/srt/speculative/dflash_worker_v2.py:1758-1943`, `:2863-2877`; `SGL/srt/models/dflash.py:763-781`.
- **The draft forward covers all 96 block positions** (`input_ids = block_ids.flatten()`, `[bs, block_size]`). The noise embedding is a **VocabParallelEmbedding lookup of the target embedding**, which does an all-reduce, eager, before the draft graph [code] — `SGL/srt/speculative/dflash_worker_v2.py:2485-2558`; `SGL/srt/layers/vocab_parallel_embedding.py:586-589`.
- **Draft attention reads only the window.** The FlashInfer backend clamps verify/prefill KV ranges to the sliding window (`update_sliding_window`) [code] — `SGL/srt/layers/attention/flashinfer_backend.py:1614-1636`, `:1925-1990`.
- **Selector:**
  - `CandidateSelector` holds `predecessor_codebook` and `successor_codebook` `[vocab, 256]`, replicated, plus `hidden_projection` 5120 → 256;
  - `_score_edges` gathers codebook rows for the candidates only;
  - the walk is a Triton kernel [code] — `SGL/srt/models/dflash.py:1043-1197`.
- **The selector runs inside the draft CUDA graph** (`_SelectorDraftSampler`, captured for decode batch sizes up to the max captured) [code] — `SGL/srt/speculative/dflash_worker_v2.py:238-299`, `:807-847`.
- **Measured on our setup:**
  - drafter 1.49 GB per GPU; draft KV 0.92 GB (E11-03 D), 2.10 GB in W1 at 881k tokens [measured] — MEAS.
  - draft-verify CUDA graph 0.38 GB [measured] — MEAS.

### Inferences
**Parameter count [computed]**

| Part | Params | Per GPU |
|---|---|---|
| Per layer: qkv 5120 x 6144 + o 4096 x 5120 + MLP 3 x 5120 x 17408 | 319.8 M | /4 → 80 M = 160 MB BF16; 5 layers = **800 MB** |
| Conv projections: 5120 x 1280 x 2 per layer x 5 | 65.5 M | **131 MB** replicated |
| FC: 25,600 x 5,120 | 131 M | **262 MB** replicated |
| Selector codebooks | 2 x 63.6 M | **254 MB** replicated, only gathered rows read |
| **Total resident** | about 2.0 B params | **≈1.45 GB** (matches the measured 1.49 GB) |

**Per-step reads [computed, 300 / 250 GB/s]**

| Component | Traffic | Time |
|---|---|---|
| Layers + conv projections | 931 MB | **3.1–3.7 ms** |
| FC for the 96 committed tokens | 262 MB | **0.87–1.05 ms** |
| Candidate lm_head | 636 MB | **2.1–2.5 ms** |
| Draft attention over the 2048 window: 5 layers x 2 KV heads x 128 x 2 x 1 B (FP8, consistent with 2.10 GiB / 881k ≈ 2.56 KB/token) x 2056 x 12 | 63 MB | 0.2–0.25 ms |
| **Sum** | | **≈6.3–7.5 ms per step**, plus drafter all-reduces (the 10 from 5 layers, plus the eager embedding all-reduce, counted in NCCLN) |

- **Compute** [computed]: draft layers about 2 x 96 x 80 M ≈ 15 GFLOP per GPU; lm_head 84 x 5120 x 62k x 2 ≈ 53 GFLOP. Small next to the memory time at L4 tensor rates (rates not cited, see Gaps).
- [inference] If the sliding window were not honored, the draft attention would read about 1.5 GB at 50k context (5–6 ms). The code does clamp it.

### Gaps
- The DFlash2 blog gives no per-step latency for a 27B target, and the HF config has no FLOP breakdown.
- L4 BF16/FP8 tensor TFLOPS were not fetched from an NVIDIA datasheet.

---

## Q5. Arithmetic budget for one 12 x 8 step vs measured ~115–133 ms; GDN verify, FP8 full attention, host and CUDA-graph overhead

### Takeaway
Items beyond the question's assumed ~16 ms of target weights, ~16 ms of KV at 50k, and the NCCL all-reduce hypothesis add up to about **22–39 ms per step** (central ≈28 ms). The two largest:
1. **GDN verify state traffic, ≈8.3–10 ms.** SGLang writes all 8 per-position FP16 states (1.81 GB per GPU per step) and then copies the accepted one back.
2. **The logits path, ≈6–14 ms.** This is the 48 MB NCCL all-gather (≈3.6–10 ms) plus dense FP32 sampling passes (≈2.3–4.1 ms).

After those come the drafter's own weights (≈4–4.8 ms), the second lm_head read (≈2.1–2.5 ms) and CUDA-graph and eager-host gaps (≈2–7 ms, inferred).

Total: 16 + 16 + ~28, plus about 35–55 ms of all-reduce (NCCLN), is about **95–125 ms**, which brackets most of the measured 115–133 ms. The host CPU (Xeon Platinum 8273CL, 2.9 GHz all-core, 48 vCPUs) is unlikely to dominate, but this is unmeasured.

### Cited Findings
**GDN verify**
- **The verify kernel caches every step's state.** In `TARGET_VERIFY`, `gdn_backend` calls `kernel_dispatcher.target_verify(..., intermediate_states_buffer=intermediate_state_cache, cache_steps=draft_token_num)`. The Triton path is `fused_sigmoid_gating_delta_rule_update(disable_state_update=True, intermediate_states_buffer=..., cache_steps=...)` [code] — `SGL/srt/layers/attention/linear/gdn_backend.py:922-1097`; `SGL/srt/layers/attention/linear/kernels/gdn_triton.py:217-256`.
- **The kernel stores the full state after every token step** (`if CACHE_INTERMEDIATE_STATES: … tl.store(cache_ptr, b_h.to(...))`, `step_offset = step_idx*HV*K*V`). The FP32 register state `b_h` is converted to the buffer dtype [code] — `SGL/kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py:157`, `:330-351`, `:459-541`.
- **The intermediate buffer** is `torch.zeros((num_mamba_layers, spec_state_size+1, speculative_num_draft_tokens, H, K, V), dtype=ssm_dtype)` [code] — `SGL/srt/mem_cache/memory_pool.py:774-787`. Our runs: FP16 SSM, intermediate 1.83–1.97 GB per GPU [measured] — MEAS.
- **After accept**, `update_mamba_state_after_mtp_verify` gathers the last accepted step's state into the request's slot ("fused gather-scatter kernel") [code] — `SGL/srt/speculative/dflash_worker_v2.py:2021-2069`; `SGL/srt/layers/attention/hybrid_linear_attn_backend.py:1420-1450`.
- **An alternative exists: `--enable-linear-replayssm-spec`.** Help text: "a per-slot raw-input window replaces the recurrent verify's per-draft full-state snapshots". The intermediate SSM buffer is then not allocated (`intermediate_ssm_state_cache = None`).
  - It requires a linear chain (topk ≤ 1) and the triton or flashinfer linear-attn decode backend.
  - The STATIC ragged mode imposes no DFLASH restriction [code] — `SGL/srt/arg_groups/fields/exec_.py:460-469`; `SGL/srt/arg_groups/attention_hook.py:350-400`; `SGL/srt/mem_cache/memory_pool.py:655-787`; `SGL/srt/layers/attention/linear/gdn_backend.py:925-1030`.

**Full attention, host, CUDA graphs**
- **Full-attention verify runs on FlashInfer with FP8 E4M3 KV** (`--kv-cache-dtype fp8_e4m3`, attention backend flashinfer, linear-attn kernels triton) [measured config] — MEAS (Common SGLang flags).
- **FlashInfer lists Ada SM 8.9 (L4) as supported**, with "not all features supported across all compute capabilities" [claimed] — [FlashInfer README (fork mirror)](https://github.com/ZKeeer/flashinfer).
- **Host CPU:** G2 uses the **Intel Xeon Platinum 8273CL (Cascade Lake)**: base 2.2 GHz, all-core turbo 2.9 GHz, single-core max turbo 3.7 GHz [claimed, vendor doc] — [GCP CPU platforms](https://docs.cloud.google.com/compute/docs/cpu-platforms). g2-standard-48: 48 vCPUs, 192 GB RAM, 4x L4 [claimed] — [GCP GPU machine types](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines). Spare Cores: 48 vCPU on 24 physical cores, a VM [measured] — [Spare Cores g2-standard-48](https://sparecores.com/server/gcp/g2-standard-48) (via NCCLN).
- **Our CUDA graphs:** target-verify and draft-verify graphs are captured for bs [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]; the prefill graph is disabled in W1 [measured] — MEAS.
- **Eager per-step work outside the graphs** in `forward_batch_generation` [code] — `SGL/srt/speculative/dflash_worker_v2.py:2398-2899`:
  - Triton prepare-block;
  - the VocabParallelEmbedding lookup and its all-reduce;
  - ForwardBatch build;
  - selector param staging;
  - `prepare_for_verify`;
  - logits adjustments;
  - the full accept/sampling chain, 2 NCCL broadcasts, commit scatter;
  - mamba commit;
  - FC plus fused KV materialize.
- **Overlap scheduling:** SGLang's overlap scheduler leaves "no single idle GPU interval" with the Triton backend and "a minor gap" with FlashInfer. It gives 1.1x over v0.3, largest "on small models and large tensor parallelism sizes" [measured/claimed] — [SGLang v0.4 blog](https://lmsys.org/blog/2024-12-04-sglang-v0-4/). DFlashWorkerV2 "Drives both overlap and non-overlap scheduling" [code] — `SGL/srt/speculative/dflash_worker_v2.py:364-369`.
- **CPU starvation** can stall communication and kernel launches in multi-GPU inference even with CUDA graphs [measured, abstract] — [arXiv 2603.22774](https://arxiv.org/abs/2603.22774) (via NCCLN).
- **All-reduce count and size:** about 139 all-reduces per verify step for this model shape, and 64-token / 128-token bf16 x 5120 SHM all-reduce at 267.6 / 498.1 µs [measured on 4x 5060 Ti] — [HyperQwen #254](https://github.com/syv-ai/HyperQwen/issues/254). Our 64 MB all-reduce: 20.9 ms [measured] — MEAS, NCCLN.

### Inferences
**GDN verify [computed]**
- State per request per GPU: 48 layers x 12 heads x 128 x 128 x 2 B (FP16) = **18.87 MB**, which matches the question's 18.9 MB.
- Writes per step: 8 positions x 18.87 MB x 12 requests = **1.81 GB → 6.0–7.25 ms**. Cross-check: the buffer of 12–13 slots x 151 MB = 1.83–1.97 GB matches the measured size.
- Initial-state reads: 226 MB → 0.75–0.9 ms. Commit gather-scatter: 12 x 2 x 18.87 MB = 453 MB → 1.5–1.8 ms.
- **Total ≈8.3–10 ms per step**. Only the state at the accepted position (about 4.3 tokens per step at our ~420 tok/s / 12) is ever used.
- [inference] ReplaySSM would replace the 1.81 GB of snapshots with small per-step records: per layer per request about 12 heads x 8 x 128 x 2 B ≈ 25 KB, versus 3.1 MB of snapshots. That would save roughly **6–7 ms** and free about 1.8–2 GB per GPU for KV. It is untested on our DFLASH + GDN build, so correctness and speed must be validated.

**Full attention [computed]**
- KV per token per GPU: 16 layers x 1 KV head x 256 x 2 (K, V) x 1 B = **8,192 B**.
- At 12 x 50k: 4.92 GB → **16.4–19.7 ms**, matching the question's ~16 ms. At 30k: 9.8–11.8 ms. At 60k: 19.7–23.6 ms.
- [inference] With 8 query positions x 6 Q heads per KV head (48 rows), a FlashInfer prefill-style tile should load each KV page once per request per layer, so verify cost ≈ one KV pass. Q/K/V writes for 96 tokens are negligible.

**Logits path:** Q1, ≈6–14 ms (all-gather 3.6–10 plus dense passes 2.3–4.1).

**Drafter:** Q4. Own weights plus window attention ≈4.2–5.0 ms; candidate lm_head ≈2.1–2.5 ms.

**Target lm_head (verify):** 2.1–2.5 ms.
- [inference] This is probably already inside the "~16 ms of weights". Our measured 4.62 GB/GPU of target weights minus the 0.64 GB embedding shard ≈ 3.98 GB read per step → 13.3–15.9 ms, which matches the question's ~16 ms. The drafter (1.49 GB measured) is **not** inside that number.
- [inference] CFG27's quantization ignore list names `…linear_attn` modules for every GDN layer. The measured 4.62 GB/GPU only fits if the GDN projections are in fact INT4: BF16 GDN projections would give ≈6.5 GB/GPU. Probably only `in_proj_a`, `in_proj_b` and norm are excluded, since a non-Linear parent module name matches nothing.

**CUDA-graph and host gaps [inference]**
- The target-verify graph has roughly 64 layers x ~15–25 kernels plus ~130 NCCL nodes, about 1,100–1,700 nodes; the draft graph about 100–200.
- At about 1.5–3 µs of inter-kernel gap per node, that is **≈2–5 ms of bubbles**.
- The eager section has roughly 60–120 small launches plus 3 eager NCCL calls (embedding all-reduce, 2 broadcasts). On a 2.9 GHz Cascade Lake at about 5–15 µs of Python per launch, that is ≈0.5–2 ms of CPU. With overlap it should hide behind ~120 ms of GPU time.
- The rank-0 broadcasts are cross-rank barriers, so any rank skew becomes exposed wait.
- Total host and graph ≈2–7 ms.

**Other activation traffic [inference]:** MLP intermediates (96 x 17,408/4 x 2 B), norms and residuals over 64 layers are each a few MB, so ≈0.5–1.5 ms.

**Budget table, per GPU per step at 12 x 8, 50k context (computed at 300 / 250 GB/s; NCCL from NCCLN)**

| Component | Traffic per GPU | ms | Inside the question's assumed numbers? |
|---|---|---|---|
| Target weights incl. lm_head (one read) | ≈3.98 GB | 13.3–15.9 | yes (~16) |
| Full-attention FP8 KV @50k | 4.92 GB | 16.4–19.7 | yes (~16) |
| TP all-reduces (~139 x ~0.27–0.4 ms) | — | ≈35–55 | the "NCCL hypothesis" |
| **Logits all-gather (NCCL, 48 MB)** | 47.7 MB over PCIe/SHM | **3.6–10** | no |
| **Logits copies + FP32 + sampling-verify passes** | ≈0.7–1 GB | **2.3–4.1** | no |
| **GDN intermediate-state writes** | 1.81 GB | **6.0–7.25** | no |
| **GDN h0 reads + accept commit** | 0.68 GB | **2.25–2.7** | no |
| **Drafter layers + conv + window attention** | 0.99 GB | **3.3–4.0** | no |
| **Drafter FC (target-hidden projection)** | 262 MB | **0.87–1.05** | no |
| **Drafter candidate lm_head (second read)** | 636 MB | **2.1–2.5** | no |
| Selector, top-16, small all-gathers, 2 broadcasts | small | ≈0.2–0.5 | no |
| CUDA-graph bubbles + exposed host | — | ≈2–7 (inferred) | no |
| Other activations | ≈0.2–0.4 GB | ≈0.5–1.5 | no |
| **Sum** | | **≈87–131 (central ≈108)** | measured 115–133 |

**What could explain the unexplained part [inference]**
- Beyond 16 + 16 ms, the non-all-reduce items sum to about 22–39 ms.
- If all-reduces are about 35–55 ms, everything together lands at about 87–131 ms. The measured step is consistent with this decomposition, and **no single "mystery" component is needed**.
- **Ranking of the non-all-reduce, non-weight, non-KV items:**
  1. GDN snapshot writes and commit, ≈8–10 ms. Lever: ReplaySSM spec flag, untested.
  2. Logits all-gather, ≈4–10 ms. Lever: vocab-parallel sampling verify, which needs a code change; P2P would roughly halve it.
  3. Drafter weights plus second lm_head plus FC, ≈6–7.5 ms. Lever: hot-vocab on the candidate head saves only ≈1.6–1.9 ms.
  4. Dense FP32 sampling passes, ≈2–4 ms. Lever: a fused sparse top-k verify.
  5. Graph and host bubbles, ≈2–7 ms.
- **Sensitivity to context:** at 30k the KV term drops by about 6–8 ms; at 60k it rises by about 3–4 ms. The GDN, logits and drafter terms are context-independent and scale with batch x 8.

### Gaps
- **Everything above except the logged memory sizes and the 64 MB all-reduce is arithmetic.** An `nsys profile --cuda-graph-trace=node` of one decode window, or `torch.profiler` with CUDA-graph node tracing, is needed to confirm the split. Most uncertain: the all-gather, `torch.topk`, and the per-node gaps.
- No measured L4 achievable DRAM bandwidth was found; the 250 GB/s figure is an assumption.
- No FlashInfer FP8-KV prefill/verify benchmark on sm89 was found. The FP8 dequant path could be compute- or latency-bound rather than bandwidth-bound at 48 rows per KV head.
- Whether Kaggle's 4xL4 is exactly a g2-standard-48 (and therefore an 8273CL) is assumed from the question, not verified on the box (`lscpu` would confirm).
- `--enable-linear-replayssm-spec` correctness and speed with DFLASH + GDN + FP16 SSM on sm89 is unverified. It appears permitted by the code paths read.
- Whether our requests send presence/frequency penalties was not checked. If they do, the verify path adds a dense [bs, vocab] penalty broadcast-add, about +0.7 ms (`SGL/srt/speculative/dflash_utils.py:254-300`).
