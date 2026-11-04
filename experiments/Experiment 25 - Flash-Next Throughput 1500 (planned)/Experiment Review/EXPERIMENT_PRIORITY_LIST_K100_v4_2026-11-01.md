# Experiment priority list toward 1500 tok/s, NB-K100 baseline (v4, 2026-11-01): the gap check

**What v4 is:** a second full read of GPT audits #1, #2 and #3, line by line against v3 (`EXPERIMENT_PRIORITY_LIST_K100_v3_2026-11-01.md`).
- v3 stays valid. v4 adds every item v3 dropped or under-weighted, plus new Merlin source checks made today.
- Section E is the merged final ranking.

## A. Items v3 dropped or under-weighted (with the audit and the fix)
| # | Item | Audit | Fix in v4 |
|---|---|---|---|
| A1 | **Duck-side idle / agent overhead** is a third target, besides tick latency and resident concurrency. Window 385 vs 414 with all workers busy means ~7 % lost to idle / tool / turn gaps | #1 §1 | New **P1-7**: per-worker idle timeline (tool exec, turn assembly, admission waits); overlap the next request with tool execution; read back the existing Duck worker timestamps first (0 GPU) |
| A2 | **K100Q prefill sizing:** 24.15 M prompt tokens, 20.81 M prefix hits, **3.34 M fresh-prefill** (13.8 %) | #1 §3 | Added to P1-3 sizing. 58 % of the 3.34 M ≈ 1.94 M generated-tail tokens re-prefilled. Audit #2 notes this breakdown was not independently recomputed: recompute it from the K100Q per-turn records before the port |
| A3 | **PR #52789 real effect:** TTFT −9.4 to −25.5 % (Kimi-K3, 8k in / 1k out), but **total request latency only 0.2-3.8 %** | #1 §3 | P1-3 expectation stays 2-6 %; the TTFT gain is not a throughput gain. Refs #55760, #57111 added |
| A4 | **Audit #2: vLLM RFC #55697 application-directed checkpoints** (5.9 → 12.4 QPS, Qwen3.5 on L40S, a different workload) | #2 §4 | **New port route for P1-3:** Duck knows each turn boundary, so the client can ask the engine to checkpoint at the end of generation. This may be simpler than an automatic generated-tail publish. Inspect the #55697 design |
| A5 | **The four generated-tail port gates:** replayed prefix token IDs == original; checkpoint position == prefix length; attention-KV + recurrent-state numerical equivalence; PP0 / PP1 completion fence before the checkpoint becomes visible | #2 §4, #1 §3 | Made explicit in P1-3 |
| A6 | **Reduced-TP sampling RFC #54333 / PR #54332** (greedy global argmax, finite top-k candidates; full-logits fallback for probabilistic speculative verify) | #2 §2 | Added to P1-1 references; confirms that our probabilistic MTP path keeps full logits |
| A7 | **Hybrid allocator issues vLLM #62752** (avoidable preemption at Attention / Mamba block transitions; CPU reproducer) **and RFC #64508** (hybrid-state internal paging) | #2 §3 | Added to P0-3. Run the #62752 CPU reproducer in a Kaggle CPU kit against Merlin (0 GPU) |
| A8 | **KV ledger gate list:** no page reuse before outstanding PP work completes; no stale Mamba state; greedy token IDs identical; 40k / 90k replay | #2 §3 | Made explicit in P0-3 |
| A9 | **Audit #3 #49793 detail:** an exact local argmax must reduce (max value, global token ID) and keep tie behaviour. Inspect the #49793 diff **before** the sharded-sampling port; if it is small, do the MTP-specific path first | #3 §1B | **Order swapped:** P1-2 (MTP draft-head path) now comes before P1-1 (sharded sampling) |
| A10 | **Silent fallback:** `--enable-batch-sharded-sampling` without `compute_logits_local` falls back silently (documented for GPT-OSS) | #3 §1A | A read-back in P1-1 must prove the sharded path really ran (All-to-All present, AllGather absent) |
| A11 | **#61589 references:** issues **#54360** (hybrid GDN cache regression) and **#53670** (EAGLE block-drop overhead, cache-hit alternation, big recomputation) | #3 §3B | Added to the P0-3 chained-turn test. Per turn: exact input-token identity, cache hit, fresh prefill, SSM checkpoint position, oracle. Fail gate: hit up but oracle changed |
| A12 | **The official Flash-Next recipe** (`recipes/models/Qwen/Qwen3.8-Flash-Next.yaml`) | #3 §11 | **Fetched and compared today.** Section B has the result: one new capacity lever |
| A13 | **DFlash2 / speculative bug list:** SGLang #40001, #41736, #41737; vLLM #54691, #55600, #58894, #54919, #62556 | #1 §7, §9 | **Checked against Merlin today** (section C). Two must become NBDF-K100 gates |
| A14 | **vLLM #62556:** larger mixed-batch capture sizes for Flash-Next helped at concurrency 128 / 256 on 4×B300, at a graph-memory and KV cost. Our decode is 24 × (K3 + 1) ≈ 96 query tokens, so a bigger grid is no obvious win | #1 §8 | P1-4 rule: log num_query_tokens, prefill tokens, capture key, eager fallback, PP0 / PP1 dispatch per iteration. Skip the grid experiment if there is no eager fallback |
| A15 | **The 2.9 ms AllGather split** (communication vs kernel vs synchronization) via torch.profiler before any port | #1 §4 | Added to P0-2 |
| A16 | **The minimal speculative regression matrix:** no-spec baseline vs MTP K3 (or DFlash2) at 8,192 / 40,960 / 90,112 / 116,736 × concurrency 1 / 8 / 24; temperature 0, seed 42; equal token IDs, `pp_state_commit_ok`, `mamba_restore_ok`; plus a separate statistical test for temperature-1 distributions (accepted tokens, rejection sampling, residual correction unchanged) | #1 §4, §9 | This is now the standard correctness gate for every speculative or sampling change (section D) |
| A17 | **SGLang QSA / NEXTN fusion was audit #1's P1, rank 3.** v3 put it in P2 | #1 §5 | Restored as **P1-8**: first a metadata kernel-count baseline inside NB-S1 (free, from the profiler), then one isolated fusion at a time with exact oracle at 8k / 40k / 90k × 24 streams; stop at the first one with no isolated gain |
| A18 | **MTP scale test design:** change only the MTP layer's scale; the 0.124 AL gap may be the whole FP8-target-KV effect, not the MTP default scale. Teacher-forced logits at 1k / 8k / 40k / 90k + greedy oracle + real Duck acceptance | #1 §8 | Made explicit in P2 |
| A19 | **ReplaySSM** (SGLang #35544 / #36970) | #1 §8 | **Merlin check:** `use_replayssm` exists (`config/cache.py`), but it is used only by Mamba2 (`mamba_mixer2.py`, nemotron_h) and Kimi-K3 KDA, not by Qwen4Exp GDN. It cannot be used by flag; GDN would need a port. Stays P2 research |
| A20 | **Humming details:** BF16 vs FP8-activation path (numerical parity), expert routing imbalance, repeated expert dispatch, CTA occupancy, CUDA-graph replay correctness; port = dispatch / heuristic + GEMM specialization tables, klee100 packing unchanged; cancel if a 5 % GEMM gain gives < 0.5 % tick | #1 §8, #2 §6, #3 §5 | Made explicit in P1-4 |
| A21 | **Boot:** vLLM docs provide version-matched precompiled kernels; pin exact Python / CUDA / Torch / FlashInfer / kernel revision; warm vs cold boot | #1 §8 | P2 note |
| A22 | **Extra pass gates:** p90 TTFT, completed tasks per hour, CUDA errors, KV high-watermark | #1 §6 | Added to the NB-S2 / S3 record list (recorded, not run gates) |
| A23 | **DFlash2 references:** z-lab/dflash (implementation reference); vLLM 0.32 DFlash docs state that identical output across precision / batch / sampling is not guaranteed | #2 §5 | NBDF-K100 compares quality on a matched task set, not only token parity |
| A24 | **R2 rule:** only two isolated variants per notebook, never two changes in one arm | #1 §10 | Notebook plan rule |
| A25 | **PP 26 / 22** only as a memory-reserved branch, if the 24-worker profile proves imbalance | #3 §7 | Unchanged from v3 |

## B. New from the official Flash-Next recipe (fetched today): QSA indexer KV dtype
- **Recipe:** Hopper / Blackwell use `--kv-cache-dtype fp8` **and `--attention-config.indexer_kv_dtype fp8`**.
- **Merlin** (`models/qwen4_exp/nvidia/indexer_qsa.py:152-176`): `resolve_indexer_kv_dtype("bf16")`. The default is **bf16**; only bf16 or fp8 are allowed. With fp8, the QSA compressed key cache is plain e4m3 without scales ("Q and the compressed K are RMSNormed before quantization").
- **Our notebooks never set it** (grep of both E28 notebooks and all experiments: no `indexer_kv_dtype`). So every run so far kept the indexer compressed-key cache in **bf16**.
- **sm_89:** the fp8 logits kernel is a plain Triton `tl.dot` fp8 × fp8 (`ops/qsa_indexer.py:309-312`; the comment covers SM90 and SM100). Ada has FP8 MMA and Triton lowers fp8e4nv dots on sm_89. The SM90-only gate at line 641 is the cooperative top-k choice, not the dtype. Compile / run on L4 is UNVERIFIED.
- **Expected:** a capacity gain (pool blocks) plus indexer bandwidth. The size depends on the compressed-key share of a block.
- **Cheapest test, 0 GPU:** add `indexer_kv_dtype fp8` to the CPU-kit R9 `get_kv_cache_configs` calculation on the klee100 worker record → pool blocks with and without.
- **Then GPU:** one arm in NB-S2 with the 8k / 40k / 90k / 116k oracle and the ARC quality check (FP8 QSA KV already matched BF16 quality, 13 / 15 = 13 / 15 in NB5).
- **Rank:** **P0-4** (config only, capacity lever, both 6-h notebook families can use it).

## C. Audit #1 HIGH bug list, checked against Merlin ad42e37 today
| Bug | Merlin status | Action |
|---|---|---|
| PP speculative decode: non-last stage recurrent state not committed → silent wrong outputs (SGLang #40001) | Our MTP K3 PP2 path passes oracles in NB4S / K100Q; DFlash2 PP2 is untested on GPU | NBDF-K100 gate: long oracle on the D arms (already in NBDF); add a cross-stage state parity read-back if available |
| DFlash grouped-conv tap NaN / Inf leaking to other requests | **Isolation present:** `_dflash2_grouped_conv_kernel` masks taps by `position = row % BLOCK_SIZE` (`qwen3_dflash2.py:26-79`), so taps never read across a draft block. This holds only if every request's draft rows are block-aligned | Unit check: NaN-inject one draft block on CPU (torch path `_grouped_conv`) and confirm the neighbours are unchanged; add it to the NBDF-K100 units kit |
| All-NaN score row in the DFlash2 selector → index out of bounds | **No NaN guard found:** `speculator.py:20-75` loads scores with `other=-inf` only for masked lanes; `gumbel_noised_argmax` returns `tl.max(..., return_indices=True)`. With an all-NaN row the index is undefined, and `previous = index` feeds the next step's `score_base`, so an out-of-row read is possible when BLOCK_K > top_k | **NBDF-K100 real-failure gate:** record the non-finite draft-score count per arm (a NaN in draft scores = real failure). Optional seam guard: `scores = where(isnan, -inf)` before the argmax; it changes nothing for finite rows |
| Speculative KV / Mamba unequal block accounting | Merlin has per-group block sizes and the lookahead / reprefill logic (#61589-equivalent) | Covered by the P0-3 KV ledger + the chained-turn test |
| Mixed long prefill starves active decode | Merlin has `long_prefill_token_threshold` (+ adaptive) | P1-6; record p95 decode stall |
| Larger graphs cut KV capacity → admission / OOM | NBDF graph capture at the drafter | Graph memory + KV pool at a realistic 104k occupancy recorded per arm (NBDF already records the pool) |

## D. Standard correctness gate (from A16; applies to every speculative / sampling / cache change)
1. Greedy token-ID parity vs baseline at 8,192 / 40,960 / 90,112 / 116,736 × 1 / 8 / 24 streams.
2. `pp_state_commit_ok`, `mamba_restore_ok`, and no non-finite draft scores.
3. Probabilistic: a separate statistical check of accepted tokens and the rejection / residual distribution.
4. **Gate rule (user rule after NBDF):**
   - Only real failures skip an arm: crash, parity or oracle fail, NaN, real fallback, missing init.
   - GO thresholds (S1 ≥ 3 %, S2 ≥ 5 %, S3 ≥ 8 %) are read after the run and never skip arms.

## E. Final merged ranking (v4)
| # | Experiment | Expected | Cost |
|---|---|---|---|
| **P0-1** | NBDF-K100 DFlash2 (K3 / K4, SW-1024; + the C gates: NaN-score real failure, conv NaN-isolation unit, long oracle). Promotion = 24-worker aggregate completed tok/s at matched quality | −20 to +30 % | High (built, closed; waiting for research agents A / B) |
| **P0-2** | NB-S1 stage-1 profiler at 16 / 24 workers: timeline, the 2.9 ms split, 3 draft heads, graph FULL / PIECEWISE / EAGER per batch, metadata kernel count | sizing | Low |
| **P0-3** | NB-S2 KV ledger + live-block admission + 24 / 28 / 30 / 32; F 6 / 5 / 4; chained-turn + sibling test (#61589 / #54360 / #53670); #62752 CPU reproducer | 0-15 % | Low-medium |
| **P0-4** | **NEW:** `--attention-config.indexer_kv_dtype fp8` (official recipe; Merlin default bf16). CPU-kit pool calc first, then the oracle + quality arm | capacity +?; 0 GPU first | Very low |
| **P1-2** | MTP draft-head path (#49793 diff first; fused trailing all-reduce; LM-head reuse / launches / overlap; greedy local argmax as diagnostic) | 3-9 % | Medium |
| **P1-1** | Qwen4Exp sharded sampling (#61018 + #50465; #54333 notes), with a read-back proving no silent fallback | 0-3 % (max ~4.4 %) | Medium-high |
| **P1-3** | Generated-tail checkpoint: client-directed (#55697) or automatic publish; 4 gates (A5); sizing from 3.34 M fresh-prefill | 2-6 % | High |
| **P1-4** | Humming W2 / W3 bench + dispatch tables (A20 details) | 0-10 % | Medium |
| **P1-5** | Post-target graph capture / fusion (after P0-2) | 2-8 % | High |
| **P1-6** | Long-prefill threshold + adaptive (#57951; #60476 caution) | 1-6 % | Low |
| **P1-7** | **NEW (A1):** Duck idle / agent overhead (385 vs 414 → ~7 %): worker timeline, overlap the next request with tool execution | 0-7 % | Low-medium |
| **P1-8** | **Restored (A17):** SGLang QSA / NEXTN metadata fusion, one isolated fusion at a time | 1-6 % | Medium-high |
| **P2** | MTP-only FP8 scale (A18); boot / JIT cache (A21); GDN ReplaySSM port (A19) | 0.5-5 % each | Mixed |

The do-not-repeat list and the framing (milestone 500-600, then 700+; 1500 needs structural steps; report solved tasks and tokens per solve with every speed winner) are unchanged from v3.
