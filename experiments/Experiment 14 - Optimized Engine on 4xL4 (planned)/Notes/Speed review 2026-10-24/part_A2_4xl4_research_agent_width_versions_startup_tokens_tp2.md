# Part A2: facts from the 4xL4 research notes (agent throughput, DFlash width, engine versions, startup, token-efficient variants, 2xTP2)

Date: 2026-10-24. Extraction only; no experiment design.

**Files read in full** (every line), folder `D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/Research - 4xL4 engine bottleneck validation/`:

| Short name | File | Lines read |
|---|---|---|
| AET | `agent_effective_throughput.md` | 1-280 |
| DFW | `dflash_width_correctness.md` | 1-268 |
| LV | `local_validation.md` | 1-683 |
| NEV | `newer_engine_versions.md` | 1-484 |
| SLE | `startup_load_environment.md` | 1-356 |
| TEV | `token_efficient_variants.md` | 1-415 |
| TP2 | `tp2_replicas.md` | 1-409 |

Section references use each file's own headings (AET §1-§6, DFW Q1-Q6, NEV Q1-Q4, SLE §1-§8, TEV §1-§6; LV and TP2 by heading name).

**Tags.** [M] = measured. [D] = derived by the document's author from measured numbers. [S] = read from source code (SGLang 0.5.21 unless stated). [C] = claimed, not reproduced. [E]/[I] = the author's estimate or inference. "A2 note" = arithmetic or a cross-reference I added; it is not in the source document.

**Hardware and stack behind all "ours" numbers:** Kaggle 4x NVIDIA L4 24 GB, PCIe, no NVLink. SGLang 0.5.21, TP4. Target RedHat Qwen3.8-27B INT4 W4A16 (compressed-tensors, g128). Drafter BF16 Qwen3.8-27B-DFlash2, width 8. FP8 E4M3 KV, unless a row says otherwise.

---

## 1. Measured on OUR Kaggle 4xL4 runs

### 1.0 Run key and production configuration

**Runs referenced:**
- **E11-03 rev3b run1** (foysal, 2026-10-17). Synthetic engine validation, arms:
  - A = no spec;
  - B = MTP;
  - C = DFlash2;
  - D = DFlash2 with fp16 SSM (log `server_D_C_int4_dflash2_ssm_fp16_try0.log`).
- **E11-04 rev2b run1** (esfersami50, 2026-10-18). Capacity validation: W1 lazy fp16, then the W2 and W3 relaunches.
- **E11-05 EVAL30 run1** (esfersami50, 2026-10-18). Production Poetiq-style ARC solver.
- **E11-05 S1COMMIT4 run1** (foysal, 2026-10-18).
- **E11-06 P1-60K EVAL14 run1** (esfersami50, 2026-10-19).
- **E12 Duck runs:** SMOKE1 and EVAL30 run1 (hosen42, 2026-10-19); EVAL6 and EVAL6R5 (10-20); EVAL12R5_LOW.
- **G4B rev 6d** (`EVAL12R6G4B`, 2026-10-21).
- **G4B rev 6g run1** (esfersami50, 2026-10-22). It failed; the log is `[]`.

**Production engine configuration** (LV "Note 01: server configuration"; AET header; identical keys in the E11-05, E11-06, E11-05c, E12 EVAL30 rev 1 and G4B logs):
- **Model and speculation:**
  - `tp_size` 4;
  - `speculative_algorithm` DFLASH, `speculative_num_draft_tokens` 8, `speculative_dflash_block_size` None (runtime `block_size=8`), `speculative_num_steps` 1;
  - drafter `/kaggle/input/models/foysalemonshanto/qwen3-8-27b-dflash2/pytorch/bf16/1`, `speculative_draft_model_quantization` None, type `DFlash2DraftModel`.
- **KV and SSM:**
  - `kv_cache_dtype` fp8_e4m3;
  - `mamba_ssm_dtype` float16.
- **Communication flags:**
  - `disable_custom_all_reduce` False, `enable_mscclpp` False, `enable_torch_symm_mem` False, `enable_symm_mem` False, `enable_nccl_nvls` False, `enable_flashinfer_allreduce_fusion` False, `enable_p2p_check` False;
  - `boundary_reduction` 'rs+rsv', `speculative_boundary_reduction` 'ar'.
- **Capacity and graphs:**
  - `max_running_requests` 13 in E11-06, E12 and G4B; 12 in both E11-05 runs;
  - `mem_fraction_static` 0.83, `context_length` 155,648;
  - `cuda_graph_max_bs_decode` 12, bs list [1..8, 10, 12].
- **Backends:**
  - `attention_backend` flashinfer for target and drafter;
  - `linear_attn_backend` triton;
  - `sglang is using nccl==2.29.7`.
- **Scheduler and cache** (AET header, all five E12/G4B runs):
  - `chunked_prefill_size` 4096, `schedule_policy` lpm, `page_size` 1;
  - `radix_eviction_policy` lru, `mamba_radix_cache_strategy` extra_buffer_lazy, `mamba_track_interval` 256, `max_mamba_cache_size` 56;
  - `enable_mixed_chunk` False, `prefill_decode_interval` 0, `enable_hierarchical_cache` False (`hicache_ratio` resolved to 2.0 but off), `enable_cache_report` True, `decode_log_interval` 40;
  - `startup_weight_load_mode` 'serial' (SLE §3);
  - launch config `no_prefill_graph: true` (AET §2);
  - `speculative_adaptive` False (LV);
  - `preserve_thinking: True` in `default_chat_template_kwargs` (AET §5).
- **Cache tree** (AET §2, G4B log 11:57:30): `UnifiedRadixCache`, components (FULL, MAMBA), `RustUnifiedTreeCore`, `hicache_attached=False`.
- **Client streams** (AET header): EVAL6, EVAL6R5 and EVAL12R5_LOW ran 6 client streams; EVAL30 and G4B ran 12.
- **NCCL environment** (LV "custom all-reduce messages"): SGLang sets only `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0`, and no `NCCL_P2P_*` or `NCCL_SHM_*`. No log contains `NCCL INFO` lines.
- **Unquantized parts of the target** (LV "per-GPU memory"; DFW Q4): the RedHat `quantization_config.ignore` list contains `lm_head`, `re:^mtp.*` and every `language_model.layers.N.linear_attn` (all 48 GDN layers, with `in_proj_a`, `in_proj_b` and norm). These stay BF16.
- **Log-file facts** (LV Note 01): the E11-05 and E11-06 logs have 3,425 and 1,831 newline lines (each also holds 45 `\r`). The ZIP copies are byte-identical (759,380 and 408,964 B).

### 1.1 Decode step time ("inferred step time" = decode-log window wall time / 40 forwards, prefill included; LV source fact 1)

| Quantity | Value | Configuration | Source |
|---|---|---|---|
| Step at 12 running, median / mean | **124.6 / 135.2 ms**; p10-p90 116.2-132.5; p5-p95 115.0-133.3 (n=278) | E11-06, production, B=12 | LV "Note 01/03/04: inferred step time…" |
| Same, prefill-free windows with g>50 | median 124.4, p10-p90 116.1-131.9 (n=273) | E11-06 | same |
| Step at 12 running | **128.4 ms** median, mean 144.9, p10-p90 125.2-143.1, p5-p95 124.2-221.3 (n=114); prefill-free n=92 median 127.7, p10-p90 124.6-131.5 | E11-05 EVAL30, B=12 | same |
| Step at 11 running | E11-05: n=2,370, median 124.2 ms, accept median 3.82. E11-06: n=158, 124.4 ms, accept 4.47 | B=11 | LV "Note 03: claim-by-claim" |
| E11-06 at 12 running, by context per request | 0-10k: n=28, **115.3 ms**, acc 4.880. 10-20k: 55, 118.3, 4.544. 20-30k: 58, 122.5, 4.299. 30-40k: 63, 126.8, 4.167. 40-50k: 58, 130.9, 4.342. 50-60k: 16, **133.3**, 4.049 | E11-06, B=12 | LV step-time section, context-bin table |
| E11-05 at 12 running, by context | 20-30k: 15, 124.3, 3.934. 30-40k: 67, 127.6, 3.806. 40-50k: 32, 131.4, 4.127 | E11-05, B=12 | same |
| E11-05 at 11 running, by context | 0-10k: 36, 113.2, 4.412. 10-20k: 145, 116.8, 3.970. 20-30k: 499, 120.8, 3.883. 30-40k: 1,138, 124.5, 3.837. 40-50k: 553, 127.2, 3.768 | E11-05, B=11 | same |
| E12 EVAL30 rev 1 at 12 running, by context (g>50, no prefill) | 0-10k: 27, **115.4**. 10-20k: 18, 117.4. 30-40k: 10, 128.2. 40-50k: 20, 132.1. 50-60k: 44, 135.7. 60-80k: 73, **140.2** | E12 Duck EVAL30 rev 1 | same |
| E12 EVAL30 rev 1, 12-running median by filter | all 280: 139.4; g>50: 138.4 (n=248); g>100: **137.9** (n=233); prefill-free: 135.7 (n=192). Median context at 12 running 658,066 tokens (~54.8k/request). 87 windows have g ≤ 50 | E12 | same |
| Median total context at 12 running | E11-05: 449,175 (all 114) / 448,853 (g>50). E11-06: 355,966 (all 278) / **357,034** (n=277) | | LV "Note 03" |
| E11-03 D at 12 running | **115.5 ms** raw median (n=67, all 0-10k), p10 114.8 | DFlash2 W8, fp16 SSM, synthetic | LV step-time section |
| E11-03 A (no spec) | 12 running at 0-10k: **37.9 ms** (n=236); 1 running: 21.9 ms; 22 running: 50.0 ms | target only | same |
| E11-03 B (MTP) | 10 running at 0-10k: 79.9 ms; 1 running: 40.4 ms | MTP | same |
| E11-03 D at 1 running | 39.9 ms | DFlash2 | same |
| Prefill-polluted windows | E11-06 B=12: n=5, median **323 ms**. E11-05 B=12: n=22, median 145.8 ms | | same |
| Context slope [D] | E11-06: 115.3 → 133.3 ms over ~50k tokens (+18 ms; KV-read lower bound 16.4 ms at the 300 GB/s spec). A: 37.9 → 44.4 ms from 0-10k to 20-30k; +6.5 ms vs model 12 × 20k × 8,192 B = 1.97 GB / 300 GB/s = 6.6 ms | | LV step-time "Inferences"; "Note 04" row FP8 KV |
| Marginal cost per extra request [D] | target-only (37.9 − 21.9)/11 = **1.45 ms** and (50.0 − 21.9)/21 = 1.34 ms. MTP (79.9 − 40.4)/9 = 4.39 ms = 1.10 ms per verified token. DFlash2 (115.5 − 39.9)/11 = **6.87 ms** = 0.86 ms per verified token. B=1 → B=12 in E11-03 D: +8.4 ms per request | E11-03 | LV "Note 01: remaining checkable claims"; collectives section |
| Step implied by throughput [D] | AR 289.4 tok/s at c12 = 41.5 ms step for 12 tokens. DFlash W8 403.7 tok/s at L = 4.62 ≈ **137 ms per step** | E11-03 c12 | DFW Q2 Inferences |
| TP4 W1 verify step [D] | ≈ 128 ms, i.e. 416 tok/s at 12 running with accept ~4.3-4.5 | from OURS | TP2 "Measured comparisons" Inferences |
| GPT-note throughput model, reproduced [D] | T8 = 12 × 4.3 / 0.1155 = 446.8 tok/s (448.7 with 0.115 s) | | LV "Note 01: remaining…" |
| Compute rate per GPU [D] | 2 × 27B × 96 / (4 × 0.115) ≈ 11.27 TFLOPS/GPU. Measured torch BF16 4096³ GEMM on these L4s: **75.9** (E11-03) / **78.8** TFLOPS (E11-04). The GDN layers and lm_head are BF16 | | LV "Note 04" row TFLOPS; 64 MB NCCL section |

### 1.2 Decode and effective throughput

| Quantity | Value | Configuration | Source |
|---|---|---|---|
| Total tok/s at c12 | AR **289.4**; MTP (3 steps, 4 draft tokens) **263.6** (0.91×), accept 3.11; DFlash2 W8 fp32 SSM **307.3** (1.06×), accept 4.70; DFlash2 W8 fp16 SSM **403.7** (1.39×), accept 4.62 | E11-03 rev3b, 4xL4 TP4 INT4, synthetic | DFW Q6 |
| Same arms at c16 | 0.91× / 0.82× / 1.10× of AR | E11-03 | DFW Q6 |
| Long runs, 12 × 24,576 tokens | AR 282.4; MTP 262.3; DFlash2 fp32 258.3 tok/s | E11-03 | DFW Q6 |
| Production ARC solver | SGLang INT4 + DFlash2 **328.9 tok/s** at 11 running, vs vLLM FP8 + MTP-3 280.9 (16 running) and 254-256 (12-16 running) | E11-05 run1 | DFW Q6; TP2 "Our own TP4 W1 baseline" |
| Decode at 12 running by context | <10k **500.1**; 10-20k 460.3; 20-40k 409.0; 40-60k 389.4 tok/s | "TP4 W1 baseline", E11-06 per OURS | TP2 "Our own TP4 W1 baseline" |
| Whole-run throughput | 379.9 tok/s (E11-06); 328.9 (E11-05); about 35 tok/s per stream at 12 running; DFlash2 vs no-spec at TP4 with 12 running: 404 vs 289 | | same |
| Clean decode rate (no prefill) by running count | **EVAL30** (mostly ≥40k context): 10 running 395, 11 running 422, 12 running **462**. **G4B** (20-40k): 6 → 327, 8 → 357, 10 → 360, 11 → 397, 12 → **436**. 6-stream runs: 6 running 328-421; 1 running 82-191 tok/s | E12 Duck | AET §1 "Decode rate by running count" |
| Decode summaries | EVAL30 419.6 tok/s at max running and ≥60k context; G4B 370.9 at 11 running, 20-40k | `sglang_decode_summary.json` | AET §1 |
| 11 vs 12 running at equal context | 422 vs 462 (EVAL30); 397 vs 436 (G4B), i.e. **≈ −9 %** | | AET §1 "Occupancy" |
| Effective tok/s per run | EVAL30 **167.8** (window 14,046 s; 2,357,584 generated). EVAL6 325.3 (3,744 s; 1,217,904). EVAL6R5 244.2 (3,464 s; 846,080). EVAL12R5_LOW 265.4 (6,570 s; 1,743,889). G4B rev 6d **279.0** (2,050 s; 572,035) | E12 Duck | AET §1 time-accounting table |
| Decode-only rate (generated / decode time) [D] | EVAL30 376; EVAL6 377; EVAL6R5 302; LOW 335; G4B 353 tok/s. Decode time: 6,273 / 3,228 / 2,799 / 5,207 / 1,622 s | | same |
| Earlier EVAL30 headline | ~187 tok/s = 2.36 M / 12.6 ks. The 14,046 s window includes a 679 s deadline overrun | | AET §1 note |

### 1.3 Acceptance length (DFlash2 W8 unless noted)

| Quantity | Value | Configuration | Source |
|---|---|---|---|
| Accept length by concurrency | **4.31 / 4.51 / 4.85 / 4.62 / 4.78** at c = 1 / 4 / 8 / 12 / 16 | E11-03, fp16 SSM | DFW Q2 |
| Long runs and ARC sanity puzzles | 12 × 24,576 tokens: 4.20. ARC sanity puzzles: 3.77-5.50 per puzzle. Overall range 4.2-4.95 (an anchor-shift bug would give ~1.06) | E11-03 | DFW Q1, Q2 |
| Production ARC solver | "Accept length median **4.55**" at 11 running | E11-05 run1 analysis | DFW Q2; TEV §4 ("AL 3.8-4.5, falling from 4.43 to 3.77 across context, median 4.55") |
| Means by running count | E11-06 B=8: n=408, arithmetic **4.4628**, median 4.470, time-weighted 4.4685 / 4.4717, token-weighted 4.5102 / 4.5134. E11-06 B=12: n=278, **4.3708**, median 4.340, 4.3495 / 4.3677, 4.4038 / 4.4048. E11-05 EVAL30 B=12: n=114, **3.9130**, median 3.900, 3.8888 / 3.8930, 3.9417 / 3.9446 | production | LV "Note 01: acceptance means" |
| Other running counts | E11-06 B=9: n=400, mean 4.8127. B=11: n=158, 4.4710. B=10: n=36, 5.2075. Record counts {1:1, 2:6, 3:2, 4:2, 5:1, 7:7, 8:408, 9:400, 10:36, 11:158, 12:278}. E11-05 EVAL30 B=11: n=2,371, mean **3.8473**, median **3.820** | | same |
| Whole-run means | E11-05 **3.8665**; E11-06 **4.5834**; E12 EVAL30 rev 1 **5.0993**. Record counts 2,509 / 1,299 / 1,209 | | same; LV "Note 04" |
| Definition check | max abs(accept rate − (accept len − 1)/7) = 0.0057 over 5,017 records in three logs. First two E11-06 lines: 4.07 ↔ 0.44, 5.90 ↔ 0.70. At 4.3: 3.3 correct drafts = 47.1 % draft acceptance (52.9 % rejection) | | LV "Note 01: acceptance"; "Note 04" |
| Maximum observed | 7.90 (E11-03 C at B=1); width-8 logs never exceed 8 | | LV "what happens when … differs from block_size 8" |

### 1.4 Agent-level prefill, cache and occupancy (E12 Duck runs; AET)

| Quantity | Value | Source |
|---|---|---|
| Burst prefill rate (median `input throughput` of ≥3,500-token prefill lines inside a chunked burst) | EVAL30 **1,474 tok/s** at depth <16k (16-48k: 1,501; 48-80k: 1,375), 2,418 samples. EVAL6 1,548 (126). EVAL6R5 1,534 (157). EVAL12R5_LOW 1,544 (275). G4B rev 6d **1,578** (91) | AET §1 prefill table |
| Prefill stall, method (a) / method (b), and share of window | EVAL30: 7,735 / 7,617 s = **55.1 %**. EVAL6: 499 / 504 s = 13.3 %. EVAL6R5: 645 / 633 s = 18.6 %. LOW: 1,342 / 1,317 s = 20.4 %. G4B: 406 / 425 s = **19.8 %**. The two methods agree within 1-5 % | AET §1 time-accounting table |
| Uncached prompt tokens per generated token | EVAL30 **4.07**; EVAL6 0.64; EVAL6R5 1.15; LOW 1.17; G4B 1.17 | same |
| Logged prefill vs usage uncached | G4B 670,442 vs 669,980 (equal, as in EVAL6/R5/LOW). EVAL30 11.23 M logged vs 9.60 M in usage; the extra ~1.6 M are timed-out or retracted requests. Decode tokens implied by the log match usage within 0.1-0.2 %, except EVAL30 (+13 %) | AET §1 |
| Time below N running (time-weighted) | EVAL30: <12: 67.8 %, <11: 16.3 %, <10: 1.6 %, <6: 0.5 %. G4B: 82.8 / 47.1 / 45.6 / 11.8 %. 6-stream runs <6: 20.5 / 24.6 / 24.4 % | AET §1 occupancy table |
| Queue-length share of time (0 / 1 / 2 / 3) | EVAL30: 34.3 / 54.0 / 11.4 / 0.3 %. G4B: 74.8 / 25.2 %. 6-stream runs: always 0 | same |
| "11 running with queue ≥ 1" share | EVAL30 **50.0 %**; G4B **25.2 %** (= all of its 12-in-flight time before the step-down) | same |
| G4B probe step-down | 27.7 % of the time at exactly 6 running; step-down events at t = 240 / 600 / 780 s at KV 0.13 / 0.26 / 0.31. Rev 6e replaced it with G4-1 (fixed 12 workers, KV gate 0.95) | AET §1 |
| Idle | No interval >60 s with ≤1 running and no prefill. 17-38 s per run before the first or after the last decode line, i.e. ≤ ~1 % of the window | AET §1 "Idle and tails" |
| Tail (last 10 % of window), mean running | EVAL6 1.2; EVAL6R5 1.29; LOW 3.16; G4B 3.2; EVAL30 10.15. In the 6-stream runs, 7-8 % of the window ran at exactly 1 stream (82-191 tok/s) | same |
| Mean cached fraction by request index (idx0 → idx10+) | EVAL30: 0.11, 0.24, 0.76, 0.81, 0.65, 0.70, 0.87, 0.60, 0.69, 0.67, 0.58 (n=185). G4B: 0.32, 0.57, 0.80, 0.67, 0.72, 0.86, 0.61, 0.53, 0.73, 0.78, 0.84. LOW: 0.32, 0.76, 0.77, 0.72, 0.74, 0.72, 0.80, 0.88, 0.89, 0.91, 0.91. EVAL6R5: 0.40, 0.20, 0.84, 0.84, 0.93, 0.91, 0.97, 0.97, 0.77, 0.80, 0.89 | AET §1 cache hit rate |
| Token-weighted cache hit | EVAL30 **59.6 %**; EVAL6 78.8 %; EVAL6R5 84.3 %; LOW **88.7 %**; G4B **75.0 %** | same |
| Hit classes for later requests (full_reply / prefill_end / partial / zero) | EVAL30 (423 requests): 10.4 / **59.1** / 12.1 / 18.4 %. EVAL6 (84): 4.8 / 84.5 / 7.1 / 3.6. EVAL6R5 (115): 4.3 / **91.3** / 2.6 / 1.7. LOW (425): 17.4 / 72.5 / 7.5 / 2.6. G4B (164): 6.7 / 75.6 / 9.1 / 8.5 | AET §1 uncached-sources table |
| Uncached tokens in later <50 %-cached requests / excess re-prefilled previous replies / first requests | EVAL30: **8,394,803** / 1,064,762 / 165,227. EVAL6: 531,830 / 508,217 / 45,417. EVAL6R5: 555,478 / 734,018 / 42,000. LOW: 1,076,913 / 1,035,817 / 86,628. G4B: 386,168 / **245,462** / 34,049. The middle columns overlap and are not additive | same |
| Where the hit stops in the `prefill_end` class | Median **15-32 tokens before** the previous prompt's end (p10 −53 to −60 in EVAL30, G4B and LOW); 58-72 % fall within [−256, 0]. Hits are multiples of 64: G4B 141 of 150 non-zero hits, LOW 413 of 414 (the Mamba checkpoint grid) | AET §1 |
| Reply reuse fraction | median −0.003 to −0.009 in every run; p90 0.76-0.96 | AET §1 |
| `full_reply` hits | end 10-20 tokens short of the reply end and are not 64-aligned. G4B examples (previous completion, hit beyond previous prompt): (247, 234), (2008, 1880), (5585, 5457) | AET §1 |
| Prefix-hit requests with a hit >0 tokens into the reply | 19-35 % | AET §3 Inferences |
| KV and Mamba usage, EVAL30 | Decode-line KV p50/p90/max 0.77 / **0.95** / 1.00; Mamba 0.61 / 0.66 / 0.71. Cold large prefills (2,506 of 2,652 lines): KV p10/p50/p90 0.58 / 0.84 / 0.92, Mamba 0.54 / 0.59 / 0.66. KV at the end of zero-class requests: p50 0.83, p90 0.94 | AET §1 KV/Mamba |
| KV maximum in the other runs | 0.43-0.47 (6-stream runs); **0.44** (G4B). G4B still had 14 zero-hit and 15 partial later requests (317k uncached tokens = 47 % of G4B's uncached) at KV p50 0.28 / 0.10 and Mamba p50 0.61 (max 0.68) | same |
| Mamba slot occupancy | EVAL30 line `mamba num: 36, mamba usage: 0.64` at 12 running, so ≈ **20 of 56** slots remain for cached checkpoints [M/D]. Sampled Mamba usage never exceeded 0.71. KV tokens / chunk = 866,744 / 4,096 = 211.6 vs 56 slots | AET §1, §3 |
| Misses (hit <95 % of previous prompt) vs previous reply length | EVAL6 1/40 (<2k) vs 7/26 (≥10k). LOW 15/260 vs 13/42. G4B 12/114 (<2k) vs 14/39 (2-10k). EVAL30 30/192 vs 55/129. Correlation only (confounded with trims) | AET §1 |
| G4B stage-2 resumes | re-prefilled p50 4.4k / p90 24.7k tokens; at 1,500 tok/s ≈ 3 s / ≈ 16 s each [D] | AET §2 Inferences |
| Admission evidence | EVAL30: 144 of 423 later requests <50 % cached, 8.39 M re-prefill tokens, KV p90 0.95. The rev 6e KV-gate replay on 7 recorded logs holds new work only in EVAL30 (219 samples, 18.2 min) | AET §4 |
| Uncached split for G4B [D] | re-prefilled previous replies ~37 % (245k of 670k); zero/partial hits at low KV ~47 % (317k); first requests ~5 % (34k); unavoidable tool-result deltas ~11 % (~74k) | AET §1 Inferences |
| Cost model [D] | One uncached prompt token ≈ **0.27** generated-token-equivalents of engine time (1,500 tok/s prefill vs ~400 tok/s decode). Ratio ours ~0.27-0.3 vs Franzen's 0.08, so prefix reuse is worth ~3-4× more on 4x L4 | AET §1 Inferences |
| Harness facts | Rev 6 per-stream window 72,228 (G4B `rev6_window`). Planning rule: resident history 10 × 79k ≤ 0.9 × KV. Fake-server prefix check "84 ok, 0 breaks". Token estimate vs server prompt tokens: median +0.4 % (rev 5 B3). Trims: rev 4 EVAL6 ~1 per branch; rev 5 ≤1-2 per branch, the 49k output reserve forced trims at ~82k context; G4B 1 reasoning-only trim. G4-17 minimum chunk 12,288. Mamba slot reservation 4 per request (fix list line 1170) | AET §5, §3 |
| Unmeasured | Rev 6e / 6g on GPU: the rev 6g G4B `kernel_log.txt` contains only `[]`. Only 29 clean 12-running lines exist in G4B | AET §6 Gaps |

### 1.5 Memory per GPU and KV capacity (TP4)

| Quantity | Value | Configuration | Source |
|---|---|---|---|
| Available memory at start | `Load weight begin. avail mem=21.72 GB` | E11-06, all 4 ranks | LV "per-GPU memory at TP4" |
| INT4 target | elapsed 12.79 s, type `Qwen3_5ForConditionalGeneration`, quant compressed-tensors, **mem usage 4.62 GB**, avail 17.10 GB | E11-06 | same |
| DFlash2 drafter | **1.49 GB**, avail 15.58 GB | E11-06 | same |
| Multimodal reserve | 0.10 GB of the KV budget | | same |
| Mamba/GDN state | `max_mamba_cache_size: 56`; conv_state 0.04 GB; ssm_state **1.00 GB**; intermediate_ssm_state_cache **1.97 GB**; intermediate_conv_window 0.03 GB; total 3.04 GB | E11-06 (max_running 13) | same |
| Target KV | #tokens **866,744**; K 3.31 GB + V 3.31 GB = 6.62 GB; avail 5.89 GB | E11-06 | same |
| Drafter KV | 866,744 tokens; K 1.03 + V 1.03 = 2.06 GB; avail 3.82 GB | E11-06 | same |
| Graphs | target-verify: elapsed 372.53 s, **0.68 GB**, avail 2.61 GB. Draft: 9.73 s, **0.35 GB**, avail 2.26 GB | E11-06 | same |
| Sampling pre-warm | peak 293,771,264 B; headroom 389,126,144 B | E11-06 | same |
| Final | `max_total_num_tokens=866744 … max_running_requests=13 … available_gpu_mem=1.88 GB` | E11-06 | same |
| E11-05 differences (max_running 12) | intermediate_ssm 1.83 GB; KV **881,336** tokens with K/V 3.36 GB; drafter K/V 1.05 GB; graph avail 2.57 / 2.22 GB; final 1.84 GB | E11-05 | same |
| TP4 W1 budget | avail 21.72 → 17.10 (target 4.62) → 15.58 (drafter 1.49) GB; KV 6.72 GB target + 2.10 GB draft = 881,336 tokens; graphs 0.68 / 0.35 GB; final 1.84 GB. SGLang's own sizing formula predicts **886,801** (within 0.6 %) | E11-04 W1 | TP2 "Memory per TP2 replica" |
| MTP at TP4 | `Qwen3_5ForCausalLMMTP … mem usage=1.45 GB`; draft KV 0.11 GB K for 474,203 tokens (~256 B/token K) | E11-03 B | same |
| G4B pools | target KV K+V 6.62 GB, draft KV 2.06 GB, Mamba ssm 1.00 GB + intermediate 1.97 GB per GPU; KV pool 866,744 | G4B log | AET header |
| Bytes per token per GPU [D] | Target: 16 full-attention layers × 1 KV head per rank × 256 × 1 B = 4,096 B for K and the same for V → **8,192 B/token/GPU**, 32,768 B across 4 ranks (866,744 × 4,096 B = 3.31 GiB ✔). Drafter: 5 layers × 2 heads per rank × 128 × 1 B = 1,280 B for K → K+V 2,560 B. TP4 total **10,752 B/token/GPU** | | LV memory; TP2 memory table |
| SSM per slot [D] | 48 GDN layers × 12 value heads per rank × 128 × 128 × 2 B = **18.9 MB per slot** (× 56 ≈ 1.00 GB ✔). TP2 note: 1.04 GiB / 57 slots = 18.7 MiB including conv. AET: ≈ 18 MB | | LV; TP2; AET §3 |
| Intermediate (verify) cache [D] | 13 requests × 8 tokens × 18.9 MB = 1.96 GB ✔ (E11-05: 12 × 8 → 1.81 GB vs 1.83 logged). E11-06 at max-running 13: 1.97 GB = 14 × 8 × per_req | | LV; TP2 |
| Cost of extra Mamba slots [D] | KV ≈ 8.68 GB / 866,744 ≈ 10 KB/token/GPU, so +40 Mamba slots ≈ **−72k KV tokens** | | AET §3 Inferences |
| Pools across 4 GPUs [D] | ≈ 34.7 GB KV + 4.2 GB Mamba | | AET §2 |
| Drafter context read [D] | Sliding window 2,048, so ≈ 12 × 2,048 × 2,560 B ≈ **63 MB per GPU per step** | | LV block-size section |
| Drafter parts not TP-sharded [S/D] | `fc` 5 × 5120 × 5120 × 2 B = 262 MB; CandidateSelector codebooks 2 × 248,320 × 256 × 2 B ≈ 254 MB (BF16 assumed); total ≈ 0.52 GB | | LV memory |
| Production KV need | "about 920k KV tokens" (E113A L78). Rev 5 Duck sizing 12 × 58k = 697k at the median, p90 ~890k | | TP2 "What fits" |
| `/get_server_info` | misreports `max_running_requests`; read the "capped" startup line instead | | TP2 implementation Inferences |

### 1.6 Communication (TP4)

| Quantity | Value | Source |
|---|---|---|
| Custom all-reduce | Both `CustomAllReduceV2 is disabled because it's not supported on more than two PCIe-only GPUs` and `CustomAllreduce is disabled …` appear on all 4 ranks in E11-05 EVAL30 (L34-41), E11-05c (L33-40), E11-06 (L33-40), E12 EVAL30 rev 1 (L33-40) and G4B (L34-41). Also `multimem all-gather disabled (no multicast for world_size=4)` | LV "custom all-reduce messages"; TP2 "Custom all-reduce at TP2" |
| Fallback path [S] | pynccl inside CUDA graphs (enabled by `graph_capture`); torch.distributed NCCL in eager mode | LV |
| NCCL 64 MiB all-reduce | E11-03: **20.89 / 20.89 / 20.85 / 20.86 ms** (ranks 0-3), step 19.5 s. E11-04: 20.79 / 20.76 / 20.80 / 20.83 ms, step 11.8 s. One size only: 16 Mi fp32 elements, `torch.distributed` ProcessGroupNCCL, eager, 10 iterations, `time.time()` with synchronize | LV "the 64 MB NCCL all-reduce measurement" |
| Derived bandwidth [D] | algbw 67,108,864 B / 20.89 ms = **3.21 GB/s** (2.99 GiB/s); busbw × 1.5 = 4.82 GB/s. The notes' 3.06-3.1 GB/s uses decimal MB. TP2 note: ≈ 3.06 GB/s algbw ≈ 4.6 GB/s busbw, "matches SHM, not P2P" | LV; TP2 "Custom all-reduce at TP2" |
| Collectives per decode step (12 × W8) [S] | Target verify: **129 all-reduces** (1 embedding + 48 GDN `out_proj` + 16 full-attention `o_proj` + 63 deferred MLP `down_proj` + 1 MLP layer 63 inline) **+ 1 logits all-gather**. Draft step: 11 all-reduces (1 eager embedding + 10 drafter `o_proj`/`down_proj`) + 2 small all-gathers. Accept: 1 broadcast greedy / 2 sampling. Total **140 all-reduces of 983,040 B** (96 × 5120 × 2) per step. Logits AG 96 × 62,080 × 2 = 11,919,360 B in, 47,677,440 B out. Drafter AGs 5,376 B and 10,752 B. Broadcast 768 B (or 48 + 96 B). No reduce-scatter; no AR+RMSNorm fusion on sm_89; `rs+rsv` changes nothing in pure TP | LV "collectives per decode step" |
| Naive comm model [D] | 140 × 983,040 B / 3.06 GB/s = **45.0 ms** (with 128 ARs: 41.1 ms; 125.8 MB) | LV collectives Inferences; "Note 01: remaining" |
| Small-message upper bound [D] | A at 1 running = 21.9 ms per step, holding 130 collectives plus ~4 GB of weight read (~13 ms at 300 GB/s). So NCCL latency at ~10 KB is at most **~70 µs** per collective. This bounds the 10 KB regime only | LV 64 MB section Inferences |
| Step decomposition estimate [E] | TP4 W1 ≈ 128 ms = weights ~22 ms + KV ~17 ms (12 × 40k × 10.5 KiB at 300 GB/s) + all-reduce ~35-55 ms (~130 ARs × 0.3-0.4 ms over SHM) + residual ~35-50 ms | TP2 "Measured comparisons" Inferences |
| P2P check | `enable_p2p_check` False in W1 (`W1LOG` L8); at TP4 the P2P test is never reached | TP2; LV |

### 1.7 Startup and storage

| Quantity | Value | Configuration | Source |
|---|---|---|---|
| server_args → Uvicorn / "fired up" | E11-05 EVAL30: 459 s (7:39) / **480 s** (8:00). E11-06: 450 s (7:30) / **472 s** (7:52). G4B rev 6d: 1,127 s (18:47) / **1,149 s** (19:09). E12 EVAL30 rev 1: 463 s / **485 s** | first launch, fresh session | LV "startup timelines" |
| Weight load (target + drafter) | E11-05: 12.87 + 0.55 = 13.42 s. E11-06: 12.79 + 0.56 = 13.35 s. G4B: **669.26** + 0.56 = 669.82 s. E12: 13.49 + 0.55 = 14.04 s | | same |
| CUDA graphs (target verify + draft) | E11-05: 373.40 + 9.85 = 383.25 s. E11-06: 372.53 + 9.73 = 382.26. G4B: 384.08 + 9.92 = 394.00. E12: 384.53 + 9.96 = 394.49 s | | same |
| "Other" time to "fired up" | 83.3 / 76.4 / 85.2 / 76.5 s. E11-06 breakdown: server_args → Init parallel 11 s; Init parallel 8.9 s (E11-05 and G4B 16.5-16.9 s); KV + Mamba allocation 4.1 s; draft graph end → `DFLASH sampling pre-warm` 22 s; Uvicorn → "fired up" warm-up 22 s, including ~7 × 1.8 s of `chunk_gated_delta_rule_fwd_kkt_solve_kernel` Triton compiles | | same |
| Engine startup timings line | E11-06: `load_weight=13.48, kv_cache_allocation=4.08, scheduler_e2e=438.91, target_verify=372.55, draft_decode=9.74`. G4B: `load_weight=669.96 … scheduler_e2e=1115.52 … target_verify=384.08` | | LV; SLE §1 |
| First capture batch | `bs=12 … 1/10 [06:02<…, 362.28s/it]`; the other 9 sizes take ~10 s (E11-06 L148). EVAL6R5: `1/10 [06:01<54:14, 361.60s/it]`, `Capture target verify CUDA graph end. elapsed=371.83 s`, `load_weight=13.52 … scheduler_e2e=438.49 … target_verify=371.84 … draft_decode=9.78` | | LV; SLE §4 |
| Target-verify capture across first launches (TP0) | 371.27, 373.40, 373.80, 376.71, 378.20, 384.08, 384.53 s | | SLE §4 |
| Warm cache, same session | E11-04 rev2b 10-18: W1 first launch 373.80 s (03:25); W2 relaunch **11.94 s** (03:59) and 13.67 s (04:33). 10-17: 13.17 / 12.43 / 4.58 / 3.52 s | | SLE §4 |
| Ready time, cached relaunches | W2 **71 s**, W3 76 s (target_verify 11.94 s); E11-03 B/C/D 136-166 s. Cold W1 ready "about 520-680 s"; target-verify ~372-385 s; weight load 13-88 s | OURS Startup | TP2 "Other costs" |
| Serving-time Triton compiles | `chunk_gated_delta_rule_fwd_kkt_solve_kernel took 1.77 s` (6 lines, W1) and 1.84 s (EVAL6R5) to compile after serving started; shape-keyed | | TP2; SLE §4 |
| Weight load before the 10-21 slowdown | `Load weight end. elapsed=` 13.16 / 13.37 / 12.84 / 37.48 / 58.70 s (E12 SMOKE1, EVAL30, EVAL6R5, EVAL6, EVAL12R5_LOW on 10-19/10-20) | | SLE §1 |
| Weight load on 10-21 (G4B) | `elapsed=669.26 s`; tqdm `Multi-thread loading shards: 50% … 1/2 [10:57<10:57, 657.15s/it]`; kernel log `ready in 1197s`. The 18.6 GB shard took 657 s ≈ **28 MB/s** while a background `cat` prefetch read the same files | | SLE §1 |
| Page cache on 10-21 | `weight prefetch started: 3 files` at t = 14.0 s (sorted order: draft before target). The DFlash2 draft (3.85 GB) then loaded in **0.56 s**, i.e. from RAM | | SLE §1 |
| Host | RAM total **202.6 GB**, available 198.9-199.3 GB, `/tmp` free **1,122.9 / 1,148.8 GB** (Flash-Next stage-0 checks, early October). The SGLang venv sits under `/tmp/sglang-venv/` | Kaggle 4xL4 | SLE §2 |
| File layout | INT4 = `model.safetensors` 18.6 GB + `model_mtp.safetensors` 0.85 GB; DFlash2 = one 3.85 GB file; total 23.3 GB | | SLE §2 |
| Current prefetch | One background `cat file1 file2 file3 > /dev/null`. At 10.5 MB/s that is ~37 min for 23.3 GB, above the 30-min launch timeout | G4B notebook | SLE §2 |
| Cache env | `SGLANG_CACHE_DIR=/tmp/e11-cache/sglang`, `CUDA_CACHE_PATH=/tmp/e11-cache/cuda`, `TRITON_CACHE_DIR=/tmp/e11-cache/triton`, `TORCH_EXTENSIONS_DIR=/tmp/e11-cache/torch_ext`. `SGLANG_JIT_CACHE_DIR` is unset, so it resolves to `~/.cache/sglang/jit` | builder L345-346 | SLE §4 |
| Launch timeouts | `SGL_LAUNCH_TIMEOUT_SEC = 30*60` (comment: "567 s in E11-04"); `SGL_RELAUNCH_TIMEOUT_SEC = 20*60`; `SGL_GIVE_UP_AT = min(T0 + 75 min, ADMIT_DEADLINE − 5 min)`. 10-21 used 1,197 of 1,800 s (66 %) | | SLE §8 |
| Runner first line | `nbformat MissingIDFieldWarning` at kernel t = 6.44 / 9.34 / 10.64 / 11.02 s; the G4B first cell printed at 13.4 s; install ~1 min (package check at t = 66 s) | | SLE §5, §2 |
| Failed run 10-22 | rev 6g G4B (esfersami50): `kernel_log.txt` = `[]`, no other files, ~114 min, status ERROR. Quota 26.87 → 30.21 / 30 h = 3.34 quota-hours ≈ 100 min of container time at 2×. E9 on the same account started at 28.44 / 30 h and ran 8,879 s (~4.9 quota-hours) | | SLE §6 |
| Quota rate | "4×L4 (confirmed by 2.0× quota burn)". The 1,197 s start ≈ 40 quota-minutes; 6 min of JIT + ~10 min of load saved ≈ 32 quota-minutes per commit run [D] | | SLE §7 |

### 1.8 Correctness and greedy divergence (E11-03 rev3b, 0.5.21, INT4, fp8 KV, W=8)

- **DFlash2 vs target-only (greedy):** 1/4 identical; mean common prefix 155.0 of 512 tokens.
- **MTP vs target-only:** 0/4 identical; prefix 152.
- **fp16 vs fp32 SSM:** 1/4 identical; prefix 166.
- **"contam"** (greedy DFlash2 alone vs inside a 16-batch): **0/4 identical, mean common prefix 15.5 of 512**.
- **No target-only batch-invariance control** was run.
- Sources: DFW Q4; LV "Note 01: remaining checkable claims".

### 1.9 Accuracy and token data from our runs

- **Capped thinking budget:** S49 at a 49k cap scored **LB 32.08**, below the 60k run's **LB 34.58** (TEV §2, §3).
- **W3H, low effort** (EVAL120, 3 h, 2026-10-17):
  - solved 17/172 (9.88 %);
  - all 15 solved tasks are a subset of the xhigh R5 tasks;
  - median 30k tokens per request;
  - 78 % of low requests emitted EOS mid-thinking without `</think>` (TEV §2).
- **xhigh request length:** ~60k tokens per request (cost-derived, TEV §2).

---

## 2. Measured elsewhere (other hardware or other people)

The transferability note in each block is the source document's own statement unless marked "A2 note".

### 2.1 TP2, DP2 and 2xTP2 vs TP4 (TP2 "Measured comparisons")

**club-3090 #773** (4× RTX 3090 24 GB, PCIe Gen3 x16, no NVLink, P2P enabled with the aikitoria patch and verified 12/12, 220 W, vLLM 0.23.0) [M]:
- **Saturation ladder** (max-num-seqs 256, chunked prefill, no spec, short prompts):
  - Qwen3.6-27B INT4 AutoRound: TP4 307 vs 2×TP2 **737** (2.40×);
  - INT8-W8A8: 358 vs 741 (2.07×);
  - gemma-4-12B AWQ, TP4 vs 4×TP1: 1,007 vs 3,425 (3.4×);
  - routing between instances not described.
- **Same sitting, AutoRound INT4:**
  - single stream: TP2 53.80 vs TP4 69.26 wall tok/s (TP4 +29 %);
  - at c=64, a **single** TP2 instance (2 GPUs) 473.1 vs TP4 (4 GPUs) 294.8 (−38 %).
- **Matched MTP n=3, INT4, fp8 KV, vLLM 0.25.1, single stream:**
  - decode TP2 79.26 / 106.50 vs TP4 82.44 / 103.73 (narrative / code), +4.0 % / −2.6 %;
  - prefill at 10k: 1,556 vs 2,407 tok/s (+54.6 %); at 90k: 1,202 vs 1,948 (+62.0 %);
  - TTFT at 90k: 73.1 vs 45.6 s;
  - the TP2 arm had vLLM custom AR on, the TP4 arm had it off.
- **Custom AR forced on at TP4** (gate patched):
  - decode at c1: 80.95 → 92.92 (+14.8 %);
  - output at c16: 144.06 → 164.66 (+14.3 %);
  - TPOT at c16: 82.6 → 66.1 ms; TTFT +9 %;
  - **greedy output diverged on 4 of 5 prompts**.
- **Without spec decoding:** single-stream TP2 → TP4 +38-62 % decode. With MTP the gain disappears.
- **TP2 with custom AR and verified P2P** vs a no-P2P TP2 reference: +15.7 % decode (TP2 "Custom all-reduce at TP2").

**biMEMO/Qwen3.8-27B-int4-AutoRound card** (RTX 3090; interconnect and P2P not stated; vLLM 0.27.1; MTP n=1 with acceptance 89.8 %, mean accepted length 1.90; 512-token responses; max-model-len 262,144; max-num-seqs 32; fp8 KV) [M]:
- one TP2 instance vs TP4, aggregate: c1 90.1 vs 102.4; c2 176.1 vs 187.4; c4 323.3 vs 322.3; c8 **518.6 vs 469.4**;
- [D] 2×TP2 at total c8, even split ≈ 647 vs 469 ≈ 1.38×;
- [C] "TP4 did not scale as well at higher concurrency in our testing".

**NVIDIA forum 381102 "Comprehensive Qwen3.8-27B Study on DGX Sparks"** (2026-08-24; RadixArk/Qwen3.8-27B-NVFP4 on SGLang; DSpark block 7; 128 in / 128 out; nodes on CX-7 100 Gbps, i.e. network, not PCIe; TP2+DP2 behind an SGLang round-robin router) [M]:
- per-user TPS at C = 1/2/4/8/16/32: TP4 77.3 / 61.9 / 48.1 / 27.0 / 18.6 / 11.0; TP2+DP2 54.6 / 52.9 / 46.8 / 36.0 / 24.2 / 16.8;
- aggregate [D]: C4 192 vs 187 (0.97×), C8 216 vs 288 (1.33×), C16 298 vs 387 (1.30×), C32 352 vs 538 (1.53×), C1 0.71×;
- router: "negligible observable overhead" [C].

**vllm-topology-bench** (4× RTX 3090, no NVLink, Qwen3.6-35B-A3B-AWQ MoE, vLLM 0.21.0, 512 in / 256 out, max-model-len 8192, prefix cache off, round-robin) [M]:
- TP4 vs 2×TP2: c1 111 vs 116 (+4 %); c8 375 vs 583 (+56 %); c32 527 vs 1,026 (+95 %); c64 580 vs 1,210 (+109 %); c128 610 vs 1,437 (+136 %);
- TTFT at c16: 3.9 vs 1.4 s;
- [C] "interconnect-bound, not KV-bound".
- Transferability: MoE at short context, the "most favourable regime for DP".

**Openzeka whitepaper** (DGX Spark, Qwen3.6-27B-NVFP4, vLLM + Ray, no spec, 128/128):
- per-user TPS at c16: TP2 14.96, TP4 19.21 [M];
- the 2×TP2 aggregate (~867 vs ~508) is computed from single-instance runs, not measured [C/D];
- "Marginal TP2→TP4 efficiency 1.17× under load" [C].

**Generic guidance** [C]:
- vLLM forum: replicas are "generally better for maximizing tokens/s" when the model fits;
- wz-it: "Model fits on one GPU but KV cache is too small … use TP=2".

**SSM-TP paper** (arXiv 2602.21144): TP gains 1.6-2.1× on 2 GPUs and 2.6-4.0× on 4 GPUs for Mamba/Zamba. No DP baseline or interconnect, so it "does not settle the question".

**Transferability** (TP2 doc):
- Every published DP win is at short context (128-8k tokens), high concurrency, with KV to spare ("44-117× more KV than needed").
- No source tests 12 streams × 60-76k with an 8-token verify and KV binding.
- No L4 data; the sources used 3090s, 5060 Ti or DGX Spark.
- club-3090 and topology-bench do not say how requests were split. biMEMO compares a single TP2 instance.

### 2.2 All-reduce, P2P and PCIe transports

- **HyperQwen #254** (4× RTX 5060 Ti, PCIe Gen3 x8, all PHB, TP4, vLLM 0.30, custom AR off) [M]:
  - NCCL left P2P off by default (`via SHM`) even though the driver reported peer access;
  - `NCCL_P2P_LEVEL=PHB` turned P2P on: decode +6.1 % (c1), +10.1 % (c2), +13.3 % (c4), +2.0 % (c8, greedy); 28k prefill unchanged (TP2 "Measured comparisons", "Alternatives");
  - NEV Q4 quotes the same issue as "2 to 6% faster at C1 and 10 to 16% faster at C4", with no change in prefill.
- **Kaggle 2×T4 VM** (nccl-allreduce-bench) [M]: GPUs `PHB`, `intraNodeP2pSupport 0`, NCCL "via SHM/direct/direct", busBW plateau 3.97 (ring) to 4.58 (tree) GB/s, small-message α ~14-16 µs (TP2 "Custom all-reduce at TP2").
- **GCP L4 P2P** [C, via sibling `nccl_p2p_l4.md`]: a GKE 2×L4 node at PHB with P2P OK; nvbandwidth on g2-standard-48 ~10 GB/s P2P. "So L4 P2P on GCP exists, but not proven inside Kaggle's VM" (TP2).
- **FlashInfer PCIe-IPC all-reduce**, SGLang `pcie_ipc_ar.py` docstring (8× SM120, 2 NUMA nodes, hidden 6144 bf16, FlashInfer vs NCCL) [M by SGLang authors]:
  - 12 KiB: **5.3 vs 205.1 µs (38.8×)**; 96 KiB: 15.3 vs 584.9 µs; 384 KiB: 34.6 vs 1,169.3 µs;
  - at TP8 with 8k context, a decode-sized workspace cut TPOT 21.14 → 13.62 ms and raised output 35.02 → 48.05 tok/s, TTFT level;
  - a prefill-sized workspace "costs 66 % on TTFT";
  - sources: NEV Q1.7, Q4; TP2 "Alternatives".
  - Transferability: unmeasured on Ada/L4 or at TP4; needs CUDA IPC peer access, unknown for Kaggle.
- **vLLM #53576** (opt-in FlashInfer PCIe-IPC backend, `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`): "tested E2E on 4x RTX 6000 Pro (SM120) TP4"; it self-disables if FlashInfer lacks the API (NEV Q2).
- **L4 spec numbers** (LV, unverifiable locally): 300 GB/s memory bandwidth, 121 TFLOPS dense BF16, PCIe Gen4 x16 = 32 GB/s each way (≈31.5 one way / 64 GB/s bidirectional).

### 2.3 DFlash/DFlash2 acceptance, width and throughput (DFW)

**Paper and vendor data:**
- **DFlash paper Table 8**, train/test block-size mismatch (8-layer draft), Math500 / HumanEval / MT-Bench τ [M]:
  - train16-test16: 6.33 / 5.29 / 3.50;
  - train16-test8: 5.09 / 4.44 / 3.18;
  - train8-test16: 5.02 / 4.28 / 3.09;
  - train8-test8: 5.21 / 4.61 / 3.29.
  - "Trained large, run smaller" costs ~2-4 % τ vs training at that size; the reverse costs more (Q1).
- **DFlash paper, compute-bound advice** [C]: "reducing the block size in such cases can therefore yield better overall speedup" (large batches) (Q1).
- **DFlash paper, full-block acceptance:** the block-8 model fully accepts entire blocks 35.7 % of the time on Math500 (a heavy, correlated tail) (Q2).
- **DFlash paper Tables 3 and 5 (B200):** speedup falls with concurrency, e.g. Qwen3-8B Math500 5.1× (c1) → 2.8× (c32); LLaMA-3.1-8B GSM8K 2.4× → 1.6× (Q6).
- **inco.ai DFlash2 blog:**
  - Qwen3-4B DFlash GSM8K recall@1 by position 0-6: 85.4, 80.3, 79.4, 78.3, 77.5, 75.9, 72.9 %;
  - DFlash2 on Qwen3.5-4B MATH-500 holds 85.0-86.5 % from position 2 to 14;
  - Qwen3.8-27B mean acceptance: DFlash2 **4.80**, MTP 4.28, DSpark 3.62 (Q2, Q6).
- **Model card** (H200, b8, xhigh, 4096 max tokens), DFlash2 / MTP / DSpark acceptance:
  - GSM8K 5.46 / 5.02 / 4.36; MATH-500 5.28 / 4.72 / 3.92; HumanEval 4.39 / 3.91 / 3.30; MBPP 4.79 / 3.99 / 3.51; MT-Bench 4.10 / 3.74 / 3.01 (Q2).
- **H200 card, throughput vs AR** (b8 / 7 drafts for all methods), MTP / DSpark / DFlash2:
  - c1: 1.96-2.59× / 2.00-2.69× / 2.67-3.43×;
  - c8: 1.74-2.19× / 1.67-2.23× / 2.27-2.85×;
  - **c32: 0.77-1.04× / 0.74-1.13× / 1.01-1.45×** (Q6).
- **LMSYS DFlash + Spec V2 blog** [C]: the DFlash drafter beats MTP "in all of the settings we tested", c1-32, with block 16 for DFlash. No per-width high-concurrency table (Q6).

**Community and third-party reports:**
- **llama.cpp, 1 stream, RTX 3090, ~176k context** [M]: n_max 4: 56.96 tok/s, acceptance 72.4 %, mean length 3.90; n_max 5: 57.22, 68.38 %, 4.42; n_max 7: 54.94, 64.45 %, 5.51. "n_max=4-5 was the sweet spot" (Q2, Q6).
- **incoai discussion #9** [anecdote]: "Positions 4-7 usually get around 0.1 or less"; "Reducing the number of speculative tokens to 3 makes dflash perform similarly to MTP"; a third user runs 4 and gets 60-75 % with coding. No engine or concurrency stated (Q2).
- **Block-size guidance elsewhere:** z-lab README (MLX only): "For quantized targets or drafts, use `block_size <= 5`". oMLX uses runtime block size 5. The DFlash2 card uses block 8 (Q1).
- **model-serving-minefield #105** [M, search summary]: a draft budget of 2 crashes an SGLang DFlash server at CUDA-graph capture; budgets 4, 6 and 8 work (Q1).

**Verification-width features:**
- **SGLang PR #36136** (DFLASH_CONFIDENCE, Qwen3-4B, H20, TP1, greedy; open, CI failing) [M, author], change vs fixed width as GSM8K / HumanEval / MT-Bench:
  - c1 −3.6 %; c4 −7 to −8 %; c8 −1.7 to −6.3 %;
  - **c12 −0.2 / +4.6 / +10.8 %**; c16 +0.3 / +0.7 / +12.8 %; c32 +4.9 / +9.6 / +20.2 %;
  - accuracy unchanged; the cost table must be re-profiled per hardware (Q3).
- **SGLang PR #43012** (v0.5.22, block verification for the DFlash family; Qwen3.8-27B DFlash2, AIME25, GB300, c8) [M]: accept 4.009 → 4.142 (+3.3 %); tok/s +1.76 % pooled, per seed +2.75 / +2.67 / −0.29 % (Q3).
- **EAGLE adaptive defaults** [S]: candidate steps bs1 [1,3,5,7], bs8 [0,1,3], bs32 [0,1], bs64 [0] (Q3).
- **#23330:** `--speculative-adaptive` with MTP on hybrid GDN Qwen3.5 crashed (`'RadixLinearAttention' object has no attribute 'is_cross_attention'`) (Q3).
- **Transferability** (DFW): the compute-bound advice, SGLang's EAGLE defaults, the #36136 gains from c ≥ 12 and the llama.cpp sweet spots all point toward 4-6. None is DFlash2-27B-on-L4 evidence.

### 2.4 DFlash2 correctness (DFW Q4/Q5; NEV Q1.2)

- **#36548** (gabrielbryk; DGX Spark / GB10, NVFP4 target, fp8 KV, bf16 Mamba, 8 draft tokens, max 8 running, greedy ordering prompt) [M]:
  - DFlash8 serial 0/100 wrong; at c8 7/100 wrong; **111/304 (36.5 %)** on the expanded set. No-spec at c8: **0/304**;
  - each single ablation (fp32 SSM, `--disable-decode-cuda-graph`, **draft depth 4**, the #36014 beta hunk) gave **37/304** (McNemar p ≤ 8.3e-12), with none fixing it;
  - at bs1, width-one TARGET_VERIFY with all drafts rejected diverges from plain decode in 19/32 cases; the first differing value is the selected-step SSM transition output;
  - a dense BF16 lm_head gave 1/304, drift still present.
- **#35150** (DSpark forced-reject; gamma 1): first divergence at token 13; beta alignment → 23; fp32 SSM → 38; BF16 instead of FP8 KV → 39. The drift accumulates.
- **PR #36014:** packed decode rounds `sigmoid(b)` to BF16 while target-verify keeps FP32; with the fix the unit test is bitwise equal. Open; it does not claim to fix all divergence.
- **#38009** (4×3090 TP4, BF16 target and draft, block 8, BF16 KV, CUDA graphs, overlap and radix off):
  - thinking on: first difference at token 30; thinking off: all 64 tokens match; target-only A/A identical;
  - a GB10 NVFP4 FP8-KV commenter: byte-identical within one process, divergent across container restarts, correlated with accept length; block 8 vs 16 changes the output;
  - NEV adds: accept length swings 2.5-3.5 vs 4.4-6.2 per boot; the commenter recommends A/A runs per profile.
- **z-lab/dflash #159** (MLX): greedy DFlash2 diverges from AR at token 30, at block 5 and at block 8.
- **#39087:** W4A16 drafter: accept 1.03 at 38 tok/s; BF16 drafter: 3.71 at 171 tok/s. With `--speculative-draft-model-quantization` unset it defaults to the target's quantization and also gives ~0 % acceptance; `unquant` avoids this.
- **#43514** (v0.5.19/0.5.20): grammar-constrained requests (`tool_choice: required`) plus multi-chunk prefill admissions with `--enable-mixed-chunk`, at `--speculative-num-draft-tokens 6`. The whole batch emits token 0 `"!"`, accept rate 0.00, accept length 1.00, with no error. PR #43518 open.
- **#37475:** fallback `target_layer_ids` `[1,16,31,46,61]` instead of `[5,19,33,47,61]`; dead code when the key is present (ours sets it).
- **#28180:** infinite loop with repetition_penalty; closed 2026-08-17.
- **#40144:** SGLang never validates the draft layout; an anchor-first checkpoint is silently shifted (accept ≈ 1.06). PR #40203 open.
- **First-divergence positions reported:** ours ~152-155 (vs target), 15.5 (in-batch); #38009 token 30; z-lab #159 token 30; #35150 tokens 13-39.
- **Practices** [C]:
  - vLLM: "theoretically lossless up to the precision limits of hardware numerics"; "Changes in batch size may cause variations in logprobs".
  - SGLang `--enable-deterministic-inference` covers FlashInfer, FA3 and Triton, with the test `python3 -m sglang.test.test_deterministic --test-mode single|prefix|radix_cache` ("Unique samples: 1"). It says nothing about spec decoding or hybrids.
- **Transferability** (DFW):
  - Our stack has the #36548 ingredients: Qwen3.8 GDN hybrid, DFlash2 W8, concurrency, fp8 KV, reduced-precision SSM (fp16 here, bf16 there), BF16 beta.
  - Our target's lm_head and GDN are BF16 (the condition that gave 1/304), so the rate is "probably lower on our stack … not proven to be zero".

### 2.5 Hybrid cache, HiCache and agent caching (AET §2-§4)

- **LMSYS HiCache launch post** [C]:
  - up to 6× throughput, up to 80 % TTFT reduction;
  - Novita AI coding agent (Qwen3-Coder-480B, 3FS backend): cache hit 40 % → 80 %, TTFT −56 %, throughput 2×;
  - 8×H20/H800, page size 64, hicache-ratio 2;
  - no hybrid or Mamba statement.
- **HiCache best-practices doc:** `--hicache-ratio 2 --hicache-size 0`; `page_first` with `kernel` io; the host pool must be larger than the GPU pool under write_through.
- **#39830** (open; 0.5.20.dev728, Flash-Next NVFP4, extra_buffer, track interval 64, mamba cache 48): host-tier hits give wrong answers, 0/10 correct vs 10/10 cold or device.
- **#44735** (closed 2026-10-16): v0.5.20 host-tier hits wrong (0/2, 0/3) on Flash-Next. Fixed in nightly `ed8c3139` via #39862 (Oct 4), #43596 (Oct 11) and #40223 (Oct 12). Not tested on non-SM120 GPUs or no_buffer.
- **#45400** (open, 2026-10-20, v0.5.22, Qwen3.8-27B-NVFP4, EAGLE, extra_buffer):
  - re-asks get `cached_tokens: 0` once the evicted working set exceeds ~3 long (89k) prefixes;
  - host Mamba pool 56 slots vs 702 Mamba backups;
  - a proposed "evict pass-through checkpoints first" change restored 89k-token hits (25 s → 0.58-0.92 s); answers stayed correct.
- **#36935** (open, 0.5.17, Qwen3.8-27B-FP8):
  - `--hicache-size 150` lowered the device Mamba pool 165 → 147 slots (EAGLE: 70);
  - interior checkpoints only at chunk boundaries;
  - branch reuse collapses when kv_pool_tokens / chunked_prefill_size > max_mamba_cache_size (cps 2048: 0-token branch hits; bf16 SSM doubling slots 165 → 325 restored 32,768-token hits);
  - `--mamba-track-interval` "is decode-side only" and did not help.
- **#22935** (open): the `input_len − 1` lookup splits a node, leaving the parent without a Mamba value → 0-hit. PR #38625 (Qwen3.8-27B fp8, 4,096-token prompts): cached 0 → 4,032, latency 1.49 → 0.35 s.
- **club-3090 #1288** (4× RTX 3090 at 220 W, 94 GiB RAM, SGLang 0.5.19, Qwen3.8-27B W8A8 + DFlash2 with 8 drafts, TP4, max_running 16, HiCache ratio 1.5):
  - prefill 4,000-4,300 tok/s at 10k, 3,029 at 90k;
  - c16 decode ~1,811 tok/s;
  - 12-turn agentic TTFT 93 ms → 2,446 ms at 35k (attributed to DeltaNet state not prefix-cached);
  - **no HiCache-off baseline and no hit rate**.
- **Daniel Franzen** (ARC-AGI-3 milestone 2, LB 27.89) [C]:
  - 1× RTX PRO 6000, 10 streams, fp8 KV, Mamba cache 60 slots, chunked prefill 8,192, mem 0.96;
  - **93.43 % of prompt tokens reused** over 5,132 requests;
  - long-prompt (>62k) miss rate 0.03 % (1/3,248); prefill time fraction 15.25 %;
  - p90 effective prefill 10,552.6 tok/s; p90 decode 836.6 tok/s (ratio 0.08); harness-side generation 609.0 tok/s (our P1 ARC-3 notes measured 590-617 tok/s for his build);
  - trim: context 128 Ki, 12 Ki output reserve, target ≈ 57 Ki (drain 59,392);
  - patch v2.2: "validated on source/control-flow tests only; GPU integration testing is pending";
  - re-tokenization quote: "the tokenized conversation might not exactly match the generated token sequence from before".
- **Milestone-2 comparison** (Tong Hui Kang) [C]:
  - dfranzen: 10 streams, all 110 games live, KV 1,011,264, trims 118k → ~59k;
  - lordhansolo (vLLM, align-state retention + prompt-tail state caching): 14 streams, KV 1,417,100, trim ~126k → 81.5k;
  - sirikilohit (SGLang, 48 GB hierarchical KV + Mamba host cache): 16 streams, 28 games live, KV 1,004,288, trim 57k → 45k, **LB 22.53, the lowest of the three**.
- **rellik13** [C]: KV sat at 91-97 % and requests queued for cache space; fp8 KV, with the freed memory spent on history (trims 37k→27k became 57k→45k), "took the score from 14.5 to 22.5".
- **Papers** [C]:
  - CachedAttention (ATC'24): TTFT −87 %, prefill throughput up to 7.8×;
  - Pensieve (EuroSys'25): 1.14-3.0× vLLM/TensorRT-LLM (both dense attention);
  - Continuum (ICLR 2026): tool time mean 925 ms (SWE-Bench) and 1,923 ms (BFCL); delay −1.12× to −3.66×; throughput +1.10× to +3.22×;
  - Marconi (MLSys'25): up to 34.4× higher token hit rate.
- **NVIDIA Dynamo, agentic SGLang page** [C]: "Session KV is invisible to eviction"; idle timeout default 300 s, reset on each request; requires SGLang ≥ 0.5.11; metrics `sglang:num_streaming_sessions`, `sglang:streaming_session_held_tokens`; `speculative_prefill` (`max_tokens=1` warm-up) and sticky routing.

### 2.6 Engine versions: per-PR numbers (NEV)

**Release timeline:**
- SGLang v0.5.21: 2026-10-02.
- SGLang v0.5.22: 2026-10-19 (GitHub), 2026-10-16 (PyPI); 950 PRs from 245 contributors; Python ≥ 3.11.
- No v0.5.23 release; nightly `0.5.23.devYYYYMMDD` up to 2026-10-22.
- vLLM v0.30.0: 2026-09-22 (762 commits). v0.31.0: 2026-10-05 (717). v0.32.0: 2026-10-18 (814). Nightly `0.32.1rc1.dev328+g2de0531d8` (2026-10-22).

**SGLang v0.5.22 per-PR numbers** (none for Qwen3.8-27B or Ada):
- Qwen3.8-Flash-Next decode +12 % on B200 (#42175: 242 → 271 tok/s on 4×B200 at c1);
- GLM-5.3-Flash DFlash on 8×B300: 1,091.9 → 1,199.7 tok/s (#42489);
- QSA decode step 28.75 → 26.54 ms at bs64 (#43463);
- Mamba COW warm-up: first-wave cold excess 924 → 28 ms (#36266);
- InstantTensor loader: DeepSeek-R1 on 8×H200 loads in 15.3 s instead of 160 s from 50 GB/s storage (#40453).

**vLLM release items:**
- v0.30: gc frozen during graph capture: capture 12 s → 2 s, engine init 28.9 s → 8.2 s on H200 (#54646).
- **#55533** (MTP cap on a 27B hybrid, RTX 4090 D SM89 GPTQ INT4; repro on RTX 6000 Ada with RedHatAI/Qwen3.8-27B-INT4):
  - at bs 8 only 3 sequences are scheduled; MTP "2-3x slower than no-spec";
  - window = floor((num_gpu_blocks − 1) / (1 + G·(1+k))) with G = 3; the MTP head is 0.88 GiB;
  - a capacity effect.
- **#54360** (DFlash): on the v0.31.0 wheel sibling prompts alternate between a 3.2 s TTFT hit and a ~33 s full re-prefill. ROCm v0.31.0 with MTP at 64 sessions: hit rate 42 % vs 87.5 % without spec.
- **#54690** (4× RTX 4090 SM89, PCIe, TP4, DFlash2): FP8 draft KV crashes startup. DFlash at ~185k context degrades single-stream decode ~70 → ~16 tok/s, because the drafter re-scans the full draft KV.
- **#58894:** DFlash2 acceptance drops to 0 % after the first prefix-cache hit (Orin), or an illegal memory access on a hit (RTX 6000 Ada). Fix #55601 is in v0.32.0; #58894 itself is still open.
- **#59232** (L20, Qwen3.8-27B-FP8, TP1, open): TPOT −0.82 % to −1.15 %; output TPS +0.73 % to +1.15 %.
- **HyperQwen #121** (vLLM 0.28, RTX 4090 SM89): concurrent MTP3 + FP8 Triton returns empty or garbage output; `--no-async-scheduling` avoided it in 9/9 runs, at a decode-speed cost.
- **HyperQwen #328:** 2×3090 PP=2 + fp8 KV + prefix cache crash-loops on 0.31.0 (Xid 31, 75 restarts in ~10.5 h); 0.30 "ran 7 days with zero Xids".
- **thinking_token_budget with MTP:** V2 runner + MTP gives "2000 reasoning tokens and empty content at budgets 64 and 200", while V1 + MTP gives 63 and 199. Fixed by #58231 (main-only).

**Other SGLang items:**
- LMSYS cookbook (v0.5.19, SM12x): DFlash2 on RTX 5090 bf16 "4.92 ms median TPOT at an accept length of 4.29"; `extra_buffer_lazy` lowers state slots "from 5 slots to 4". No Ada, PCIe or INT4 data.
- **#38846** (0.5.20): effective `max_running_requests` is visible in `/get_server_info` as `internal_states[0].effective_max_running_requests_per_dp`. Slots per request: 5 for extra_buffer with overlap, 3 for no_buffer without overlap, 1 for `--disable-radix-cache`.
- **#43484:** `AssertionError: Can not alloc mamba cache` with two concurrent long requests on a 27B GDN model at `--max-mamba-cache-size 10`.
- **#44372:** with spec decoding and no `--max-running-requests`, the default becomes 48.
- **#42209:** the existing compact replay is not bit-exact: "79 of 80 requests diverged from stock in tokens".
- **#42917** (INT4 W4A16 `dbirks/Qwen3.8-27B-W4A16-AutoRound`, RTX 5090, driver 616.92 WSL2):
  - PPL 9.9767 vs vLLM 6.0528 (NEV: 9.98 vs 6.05);
  - emptying only `nv/` fixed it (6.2784 vs vLLM 6.2753 on 5 chunks); emptying `jit/` alone did not;
  - "one bad entry can silently corrupt results".

**Package facts:**
- SGLang 0.5.22 wheel ~34 MB, cp311-cp313 manylinux_2_34.
- sglang-kernel 0.4.9 / 0.4.10: cp310-abi3 manylinux2014, 371-373 MB.
- vLLM wheels: cp38-abi3 manylinux_2_28, 315-327 MB.
- FlashInfer `pcie_ipc_ar.py`: 44,053 B at v0.7.0 and 49,719 B at v0.7.1; SGLang `pcie_ipc_ar.py` 17,248 B in both 0.5.21 and 0.5.22.
- A full wheelhouse is ~220 wheels / ~5 GB.

**Transferability** (NEV Q4): "I found no published like-for-like benchmark of SGLang 0.5.21 vs 0.5.22, or of vLLM 0.30 vs 0.31 vs 0.32, for Qwen3.5/3.8 hybrids on any GPU." "No upgrade can be justified on published throughput for our setup."

### 2.7 Startup, storage and platform (SLE)

**/kaggle/input storage reports:**
- **Jeroen Cottaar** [M]:
  - mount `FUSE.GCSFUSE` (a 10-17 run of the same code was NFS and loaded 10 shards in 88 s; other NFS sessions 88-113 s);
  - on GCSFUSE, a 100.5 GiB 10-shard checkpoint had only 3 of 10 shards after 7 h 22 min (1.8-2.8 h per shard); an earlier offline run took ">9 hours (2 hours per shard)";
  - available RAM at load start was 71 GiB vs 41 GiB in the NFS run (≈ 30 GiB already cached); it is unknown whether this was free or `MemAvailable`.
- **amn** [M], one ~4.6 GiB shard:

  | Commit (UTC) | Time per shard | Throughput |
  |---|---|---|
  | 10-19 10:54 | 8.4 s | ~0.55 GiB/s |
  | 10-20 03:26 | 4.9 s | ~0.95 GiB/s |
  | 10-21 03:21 | 14.4 s | 0.32 GiB/s |
  | 10-21 04:35 | ~115 s | 0.040 GiB/s |
  | 10-21 05:37 | ~177 s | 0.026 GiB/s |

  - 10-21 05:33: "~190 s, about 24 MB/s", 8-13× slower;
  - 10-22: **~10.5 MB/s single-threaded, 107 MB/s with 16 threads**;
  - "part of Kaggle's uploaded files by category comes to NFS, and the other part is GCSFUSE".
- **NguyenThanhNhan:** ~180 GB at "normal NFS speed of ~300MB/s" takes ~10 min; on GCSFUSE "hours"; his save runs took 875-1,300 s to load. A 45-min server timeout gave a placeholder submission (score 0.00 in under 2 h, status Succeeded).
- **mp** (result not posted): copying to local disk once "should cut the startup time by half".

**Vendor and source claims:**
- **Google gcsfuse** [C]:
  - file cache + parallel downloads "up to nine times faster model load time"; RAM disk recommended; `max-parallel-downloads` default 2× cores; parallel downloads on by default from 2.12 when the file cache is on;
  - buffered reads "two to five times" faster, up to 320 MB per handle, incompatible with the file cache;
  - `read_ahead_kb` 128 KB → 1 MB (root);
  - all of these are mount-time or root settings;
  - fio benchmarks with 144-192 jobs reach 13-14.5 GB/s on C4-192; no single-thread figure.
- **Upstream gcsfuse `OpenFile`** [S] sets `KeepPageCache = true` unconditionally.
- **GCP `g2-standard-48`** [C]: 48 vCPUs, 192 GB default memory (192-216 GB custom), 4 L4, max local SSD 1,500 GiB, 50 Gbps. Whether Kaggle uses this shape is not stated.
- **Old Kaggle disk figures** [C]: 20 GB `/kaggle/working`; 16 GB scratch in older guides. **Docker default `/dev/shm`:** 64 MB [C].
- **Kaggle staff (Dustin)** [C]:
  - "one bad cache region which is slow right now" (10-17);
  - image version doesn't affect timers; reruns are always pinned;
  - quota is charged only for container running time; data added after container start is charged;
  - a submission every other day per team is about current capacity;
  - "status code 56" = an attached utility script blocking Jupyter start.
- **Kaggle image size** [C]: ~19.7-22.6 GB compressed; images older than two years retired.

**Other teams' loaders and timeouts:**
- **Franzen:**
  - `precache()` splits all files into 32 MiB chunks; ThreadPoolExecutor with 16 threads for wheels, `threads=3, delay=60` for the model;
  - lookahead patch: disjoint byte ranges, `--weight-loader-prefetch-num-threads` default 4, 16 MiB blocks, one shard ahead;
  - "Smaller shards and a background prefetcher made startup much more reliable";
  - `SERVER_STARTUP_TIMEOUT = 12*60` soft deadline (server not killed); `ARC3_HTTP_RETRY_INITIAL_SECONDS='900'`; older framework `timeout_seconds=900`.
- **khoinguyennguyen:** `preload_weights()` with 16 workers, 64 MiB chunks, per file.

**JIT-cache sources** [C]:
- Red Hat: Triton cache preload ~30 % faster start; OCI-packaged caches cut a demo start "112 seconds" → "just over 62 seconds"; "Cache entries are never shared between platforms".
- FlashInfer offers `flashinfer-cubin` and `flashinfer-jit-cache` (cu129 / cu130 / cu134).

**Platform failures** [C]:
- submissions that waited >15 h were marked "Kaggle Error" (fixed 2026-10-08);
- `ERRORED_MOUNTING_DATASET` after ~13 h;
- CPMP: a ≤10 h notebook timed out once, then ran in 9 h 10 min; "0 in 15 minutes, then 46% the next day";
- CPMP 10-22: "lost one sub with Kaggle error after 12 hours".
- Quota: "L4×4 sessions bill at 2×, so the 30 h weekly quota allows only ~15 h"; CPMP and staff confirmed the 2× accounting fix.

### 2.8 Token-efficient variants and effort levels (TEV)

**ARC Prize (Qwen3.8-27B)** [M]:
- ARC-AGI-2 semi-private: xhigh **42.4 %**, medium 13.2 %, low 22.8 %, none 1.5 %.
- ARC-AGI-1: 87.5 / 68.7 / 69.2 / 34.0 %.
- xhigh $0.4469 per task ($0.425/M in, $2.55/M out on Baseten).
- Cost-derived tokens per task / per request: xhigh ~172k / ~60.1k; medium ~109k / ~38.2k; low ~90k / ~31.6k; none ~20k / ~7.1k. Public-eval xhigh ≈ 63.7k per request.
- Public-eval solves: xhigh 40/120, medium 16, low 19, none 3; xhigh ∪ low = 48.
- Other estimates: Anil Thomas, xhigh median ~64k per generation; Kris Dreemurr, median 70k per task for pass@2 (CV 45 %).
- Medium < low because the template has no medium branch (Baseten, plus our byte check).

**Swift 1.0** (`ukisai/Swift-Qwen3.8-27b`) [C], BF16, base → Swift, with mean / median thinking-token reduction:
- GPQA-D 88.38 → 88.28 (41.0 / 58.3 %); MMLU-Pro 85.47 → 84.95 (46.2 / 28.3); AIME 2026 98.67 → 94.00 (26.7 / 50.2); HMMT Nov 2025 99.33 → 96.00 (31.1 / 45.9);
- LCB v6 76.76 → 81.55 (24.3 / 45.8); Terminal-Bench 2.1 66.74 → 65.84 (26.5 / 38.7); IFBench 73.53 → 71.80 (42.2 / 50.5); ERQA 67.45 → 66.30 (50.6 / 54.6);
- 5 seeds, T 1.0, top_p 0.95, top_k 20, caps 16,384-250,000;
- mean thinking reduction by effort: xhigh 41.0 %, medium 22.7 %, low 25.8 %;
- 18 merged BF16 shards, 55.6 GB [M].

**Swift 1.5** [C] (vLLM 0.27.1, xhigh, 5 seeds), base → Swift 1.5 (mean / median reduction; mean tokens):
- GPQA-D 88.28 → 88.59 (41.9 / 58.5 %; 15,014 → 8,717);
- AIME 2026 98.67 → 96.00 (40.0 / 48.5; 22,014 → 13,203);
- HMMT 99.33 → 97.33 (32.1 / 47.8); LCB 76.76 → 81.71 (24.5 / 46.3);
- Terminal-Bench 69.21 → 72.13 (**16.3 / 0.1 %**; 52,265 → 43,733);
- ERQA 67.45 → 65.40 (53.9 %); IFBench 73.53 → 72.07 (38.5 %);
- GPQA by effort: xhigh −41.9 %, medium 84.14 → 82.22 (−24.8 %), low 84.04 → 84.85 (−28.7 %).

**ThinkingCap-Qwen3.8-27B** [C] (H200, vLLM 0.29.0, MTP k=3; MTP accuracy-neutral: AIME26 98.23 % without vs 98.13 % with):
- 12-benchmark mean thinking 15,735 → 12,144 (−37.2 %); median 7,148 → 4,615 (−42.9 %); matched geometric-mean Δ −41.8 %;
- accuracy 86.65 → 85.79 % (−0.86 pp); answer tokens 1,712 → 1,754; total tokens −36.5 %; truncation 0.51 → 0.34 %; looping 0.06 → 0.05 %;
- per benchmark (accuracy, mean thinking): GPQA-D 89.93 → 88.04, 12,772 → 7,267 (−43.1 %); AIME 2026 98.13 → 94.27, 15,663 → 10,934 (−30.2 %); HMMT Feb 2026 95.83 → 94.70, 23,211 → 18,099 (−22.0 %); LCB 91.14 → 91.21, 28,395 → 22,645 (−20.3 %); **Terminal-Bench 75.84 → 75.28, 72,871 → 65,092 (−10.7 %)**; MMMLU 85.38 → 84.09, 1,656 → 571 (−65.5 %);
- effort table vs base xhigh: base medium −9.16 pp / −52.1 %; ThinkingCap medium −9.90 / −60.2 %; base low −9.71 / −55.4 %; ThinkingCap low −10.79 / −62.3 %; base thinking off −28.57 pp; ThinkingCap off −34.25 pp;
- the 3.6 release: −45.8 % out-of-domain thinking for −0.7 pp; GPQA-D 85.5 → 83.8;
- the Kaitchup: Terminal-Bench thinking −10.7 % becomes total −7.5 %.

**Other tunes and methods:**
- **Qwen Pi** (`bytkim/Qwen3.8-27B-pi`, Apache-2.0) [C]: at medium it matches base xhigh on Terminal-Bench 2.1 with ~41 % fewer output tokens; SciCode at xhigh ~23 % fewer.
- **arXiv 2606.00206** (logit penalty on overthinking markers) [C]: in up to 52 % of quantized failures the right answer appeared in an intermediate step; CoT −12-23 % with accuracy "preserved or improved" (5 models, 1.5B-32B).
- **arXiv 2603.08462:** a surprisal-based penalty gives ~48 % fewer tokens at matched accuracy.

**Independent checks:**
- MindStudio (n=1, Hermes coding task): Swift 1.0 used 60,572 vs 11,744 output tokens, 21 vs 4 tool calls, wall clock 35:52 vs 47:07; Swift stalled mid-task.
- zephel01 (ThinkingCap GGUF Q4_K_M / Q6_K, llama.cpp, T 0.35, max_tokens 49,152): L6 96.7 / 98.3 %; L7 62.5 / 75.0 %.
- kaushikvira (NVFP4, RTX 5090): IFBench mean completion tokens ThinkingCap 3,988 vs Swift 1.5 4,805 (−17 %); GSM8K 95.5 vs 95.0.
- HF #4: at IQ3_XXS, low effort, Swift 1.5 produced 20-44 % more tokens than Swift 1.0. HF #3: "Totaly lobotomized".

**Swift 1.5 INT4 card** (single seed; BF16 / INT4 / base W4A16 reference):
- GPQA-D 91.41 / 90.91 / 86.36; IFBench 72.00 / 70.00 / 72.00; AIME 2026 86.67 / 83.33 / 76.67;
- tokens vs base reference: GPQA mean −45.8 % / median −64.4 %; AIME mean −22.4 % / median −34.0 %;
- "do not establish quality parity".
- Swift 1.0 card quantized rows: W4A16 mixed GPQA 88.38 vs base 88.69; AWQ INT4 AIME 84.00 vs 82.67.

**Acceptance on tuned targets:**
- ThinkingCap card (vLLM 0.29.0, H200, xhigh): MTP k=3 top-token 2.5-3.0, probabilistic 2.8-3.3; DFlash2 (7 drafts) top-token 3.1-4.2, probabilistic 3.3-4.9.
- club-3090 #1746 (Swift 1.5 AutoRound W4A16, vLLM 0.31.0, 24-turn sessions): MTP k=3 2.97 (rate 0.66); DFlash2 5 drafts 3.59 (0.52); 7 drafts 3.85 (0.41); DFlash2 (7) 136.5 tok/s long decode vs 101.9 for MTP; no base run.
- Base vendor reference (z-lab card, block 8, T 1.0, xhigh): DFlash2 GSM8K 5.46, MATH-500 5.28, HumanEval 4.39, MBPP 4.79, MT-Bench 4.10; MTP 5.02 / 4.72 / 3.91 / 3.99 / 3.74.

**Thinking-budget mechanisms:**
- **SGLang #25536** (Qwen3.6-27B-FP8, 0.5.11-0.5.12): `thinking_budget: 200` gave ~1,350-1,430 reasoning tokens. The built-in processor uses IDs 151667/151668 while Qwen3.5/3.6 use **248068/248069**; NEWLINE 198. `--enable-strict-thinking` did not enforce the budget.

**Transferability** (TEV §1 Inferences): savings fall with trace length (Terminal-Bench at 52-73k saves 11-16 %). For ARC xhigh at ~60k the expectation is "nearer −10 to −25 % mean". The accuracy cost lands on the hardest reasoning (AIME −2.7 to −4.7 pp). No HLE or ARC result exists for any variant.

---

## 3. Levers (speed ideas)

Evidence classes:
- **mO** = measured on our runs;
- **mE** = measured elsewhere;
- **est** = estimate or derived model;
- **cl** = claim;
- **src** = established by reading source code (a code fact, not a speed measurement).

### 3.0 Lever table

| # | Lever | Mechanism | Expected gain (as stated) | Evidence | Cost / risk | Prerequisites | Status in docs | Source |
|---|---|---|---|---|---|---|---|---|
| 1 | Admit no more live sessions than KV holds (fixed 12 workers + KV gate at `sglang:full_token_usage` ≥ 0.95, 5 s poll, fail-open) | stops eviction re-prefill (EVAL30: 4.07 uncached tokens per generated token) | EVAL30 168 → ~331 tok/s (+97 %) [D] | mO (EVAL30 vs G4B; KV-gate replay on 7 logs) + cl (rellik13, Franzen) | none (harness) | rev 6 / 6e | **Done** in rev 6/6e; not yet run on GPU (rev 6g log `[]`) | AET §6 rank 1; §4; §5 |
| 2 | Keep occupancy at 11-12 running (remove the probe step-down) | the probe held 6 running for 27.7 % of G4B | G4B 279 → ~303 (+9 %) [D]; "13.3 % loss at the same context bin" | mO | none | rev 6e G4-1 | **Done** in rev 6e | AET §6 rank 2 |
| 3 | Token-identical next-turn prompt (no reply re-prefill) | the template re-renders the assistant turn (`reasoning_content` trim, `'\n</think>\n\n'` join, rebuilt tool-call XML), so the hit stops at the end-of-prefill checkpoint and the reply is prefilled again. Options: (a) offline divergence test; (b) send the assistant turn back raw; (c) token-level continuation via `/generate` `input_ids` deltas or `session_params` | G4B S2 → S3 +10 % (325 → 357); EVAL30 E1 → E2 331 → 368; 37-76 % of uncached tokens in the non-thrash runs; ~76 % of prefill time in EVAL6R5-type runs | mO + est; cl (Franzen re-tokenization) | (c) changes the client transport; must not change what the model sees | `return_input_ids` / `return_output_ids` to find the divergence point | **Not built**; cause to be confirmed offline | AET §1, §5, §6 rank 3 |
| 4 | Protect end-of-prefill Mamba checkpoints | Franzen patch: keep only the final prefill checkpoint + one cold-start branch; refresh the checkpoint to MRU when its final device lock drops (`SGLANG_MAMBA_PREFILL_FINAL_ONLY`, `SGLANG_MAMBA_REFRESH_ON_UNLOCK`; refresh-on-unlock works "including for extra_buffer_lazy"). Or a larger pool (`max_mamba_cache_size` 56 → e.g. 96) or `--enable-int8-mamba-checkpoint` ("~2x cached-prefix capacity") | G4B S3 → S4 up to +14 % (357 → 408) | mO (G4B zero/partial misses at KV ≤ 0.44, correlated with long replies) + cl (Franzen, #36935, #45400, Marconi) | patch written for the Pennyroyal fork (d00d88e), not upstream 0.5.21; names in 0.5.21's Rust tree core unchecked; patch "validated on source/control-flow tests only". +40 slots ≈ −72k KV tokens (~18 MB per slot per GPU). int8 checkpoints are rejected with HiCache | server patch or flag test | **Not built** | AET §3, §6 rank 4 |
| 5 | Fix the "11 running + 1 queued" state | AET: cause unknown (one hypothesis: 4 Mamba slots reserved per request). NEV: PR **#44401** (a chunked-prefill request's held req_to_token row is counted twice, capping the batch at `max_running_requests − 1`; `batch_is_full` latch starves the queue) plus #41872 (empty-batch latch) | +9 % at 12 in flight (11 vs 12 = −9 %); 25-50 % of time in this state; NEV: "12/12 running instead of 11/12, about +9% slots at our cap" | mO (state share; decode rate) + src (0.5.21 has the pre-patch code shape) + cl (PR text) | #44401 is main-only (not in v0.5.22); a small monkey-patch is "low-risk"; current workaround is a cap of 13 for 12 | nightly ≥ 2026-10-16, or a backport of #44401 + #41872 | AET: cause unknown. NEV: fix identified; backport "not executed" | AET §1, §5, §6 rank 5; NEV Q1.1, Q4; TP2 implementation |
| 6 | Streaming sessions (`--enable-streaming-session`) | session KV + Mamba state held in a slot outside the radix tree ("invisible to eviction"); turns appended at the token level | "Could cover levers 3+4 together" | src (supported for Mamba in 0.5.21) + cl (Dynamo) | append-only (a trim must close and reopen the session); 300 s idle reaper; holds KV for every paused session; unclear whether the chat path re-renders the delta | SGLang ≥ 0.5.11; admission ≤ KV | **Untested**; no hybrid result exists | AET §4, §6 rank 6 |
| 7 | `--enable-session-radix-cache` | unreferenced entries evicted first | not stated | src | requires UnifiedRadixCache | — | untested | AET §4 |
| 8 | Radix eviction policy (priority / tlru / slru / lfu) | a policy "only scores; it cannot pin KV" | LRU judged adequate for KV with ≤12 sessions | src/cl | policies act on KV leaves; may not protect Mamba checkpoints (tlru unverified) | tlru needs `next_prompt_estimate` < `threshold` and UnifiedRadixCache | not recommended for KV | AET §4 |
| 9 | Priority scheduling / retraction policy / Dynamo `speculative_prefill` warm-up / sticky routing | scheduler knobs | not stated | src/cl | — | flags exist (default off) | not evaluated | AET §4 |
| 10 | HiCache host tier | host KV + host Mamba pool + draft-KV sidecar | ≈ 0 while KV max ≤ 0.44; pays only if more live sessions than device KV (e.g. 24 paused stage-2 branches) | src + cl (open hybrid bugs) | wrong answers from host hits on Flash-Next (#39830 open; #44735 fixed after 0.5.20); zero host hits past ~3 long prefixes (#45400, v0.5.22); incompatible with int8 checkpoints; `buffer_only` rejects MAMBA in 0.5.21; disables tc_piecewise; no Mamba retraction backup; needs a correctness gate | host RAM ≈ 69 GB KV + ≈ 8 GB Mamba at ratio 2 (fits in 202.6 GB); v0.5.22 adds #36345, #40223, #43596, #40915 | **Not recommended now** | AET §2, §6 rank 7; NEV Q1.8 |
| 11 | Prefill throughput itself (chunk size, prefill CUDA graph) | prefill is ~1.5k tok/s and stalls decode | "Scales all re-prefill cost" | mO (rate) + mE (club-3090 4×3090 W8A8 4.0-4.3k tok/s at 10k) | — | our launch has `no_prefill_graph: true`; tc_piecewise removed in 0.5.22 (#42287) | **Not researched** ("engine scope") | AET §6 rank 8; NEV Q1.9 |
| 12 | Prefill-decode interleaving (`prefill_decode_interval`, `--enable-mixed-chunk`) | today a pending prefill always runs before decode ("Run prefill first if possible"), so 4,096-token chunks back to back stall every stream | not stated | src | #43514: DFlash accept 0 with grammar-constrained requests + multi-chunk admission + mixed chunk (PR #43518 open). v0.5.22 #43461 allows mixed chunk on the Full prefill-graph backend; nightly #45435 keeps checkpoint tracking through mixed chunked prefill | — | **Not evaluated** | AET §1; DFW Q4; NEV Q1.4, Q1.9 |
| 13 | `mamba_track_interval` 256 → 64 | more decode checkpoints | helps only the requests whose divergence lies deep in the reply (19-35 % show a hit >0 into the reply) | mO + src | more transient slots; in lazy mode tracking "degrades in place" silently; #36935: track interval "is decode-side only", did not help | must be ≥ draft tokens and a multiple of page size | low expected value until the divergence is measured | AET §3 |
| 14 | `page_size` 64 | KV pages on the 64 grid | ≤ 63 tokens per hit cost; benefit only HiCache I/O | mO (hits already 64-aligned) | — | — | not proposed | AET §3 |
| 15 | Harness: rare, large trims (hysteresis); keep a branch's slot until it finishes or trims; avoid stage-2 time-slicing | keeps prefixes and checkpoints resident | not quantified | cl (Franzen) + mO (miss vs reply length) | G4-17 minimum chunk 12,288 flagged as "more re-prefill" | — | partly in rev 6 | AET §5 |
| 16 | DFlash2 verify width 4 / 6 | fewer verify rows (96 → 48 at W4), smaller AR payload, smaller KV reserve and intermediate SSM | W4 −10 % to +12 %; W6 −4 % to +7 % [E]; "The sign must be measured" | est + mE (indirect) | acceptance drop unknown (non-causal drafter, not a truncation); new graph capture | `--speculative-num-draft-tokens` 4/6 (warning only) | **Untested on any 27B at batch 8-16** | DFW Q1, Q2, Q6; LV block-size section — detail in 3.B |
| 17 | SSM dtype fp16 vs fp32 | precision vs speed | measured at c12: fp16 1.39× AR, fp32 1.06× | mO; mE (fp32 cut #36548 errors 111 → 37/304) | fp16 has a correctness risk | — | fp16 in production | DFW Q4, Q6 |
| 18 | Dynamic verification (#36127 / PR #36136, DFLASH_CONFIDENCE) | per-request survival-based verify budget | Qwen3-4B H20: c12 −0.2 / +4.6 / +10.8 %; c16 +0.3 / +0.7 / +12.8 % | mE (author) | open, unmerged, CI failing; cost table re-profiled per hardware; no confidence heads for Qwen3.8-27B | — | **Not available** in 0.5.21 / 0.5.22 / nightly | DFW Q3; NEV Q1.3 |
| 19 | `--speculative-adaptive` | adaptive steps | — | src | EAGLE/EAGLE3 only; DFLASH excluded | — | **Rejected by code** | DFW Q3; LV alternatives |
| 20 | DSpark (ragged verify with an SPS table) | cross-request verify budget | — | src + mE (acceptance 3.62 vs DFlash2 4.80) | without an offline SPS table it "degenerates to verify-all" | profiled SPS table | "not an easy win" | DFW Q3, Q6 |
| 21 | Block verification (`--speculative-use-block-verification`, #43012) | a different acceptance rule on the sampled-draft path | accept 4.009 → 4.142 (+3.3 %), +1.76 % tok/s at c8 (GB300) | mE | no effect for greedy drafts; #45162 PP crash (we use no PP) | SGLang 0.5.22 | low-priority experiment | DFW Q3; NEV Q1.3 |
| 22 | MTP/NEXTN instead of DFlash2 | — | 0.91× AR at c12 on ours | mO | — | — | **Rejected** | DFW Q6 |
| 23 | FlashInfer PCIe-IPC all-reduce (`SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1`) | IPC-mapped peer loads/stores instead of NCCL for decode-sized ARs | docstring (8×SM120): 12 KiB 38.8×; TPOT 21.14 → 13.62 ms at TP8 | mE (SM120 only) | needs working CUDA IPC peer access; bf16 only; `tp` group only; world 2/4/8; disabled under deterministic inference; workspace sized max_bs × draft_tokens × hidden (prefill stays on NCCL) | **FlashInfer ≥ 0.7.0** (0.5.21 pins 0.6.18 = dead code; 0.5.22 pins 0.7.1; vLLM 0.31+ pins 0.7.0.post1); 0.7.x jit-cache in the wheelhouse | **Untested** on Ada/TP4 | LV alternatives; NEV Q1.7, Q3; TP2 custom AR, Alternatives |
| 24 | Enable NCCL P2P at TP4 (`NCCL_P2P_LEVEL=PHB`/`SYS`) | NCCL P2P transport instead of SHM | HyperQwen PHB: +6.1 / +10.1 / +13.3 / +2.0 % at c1/2/4/8; doc expects "+5-15 %" if P2P works | mE | no memory cost; P2P on Kaggle unproven | P2P support in the VM | ranked #1 in the TP2 doc; untested (see §6: refuted on 2026-10-23) | TP2 Alternatives |
| 25 | Patched custom AR at TP4 (remove the `world_size > 2 and not full_nvlink` gate) | custom AR on PCIe | club-3090 3090s: +14.8 % c1, +14.3 % c16 | mE | code patch; **greedy divergence on 4/5 prompts**; needs P2P and a correctness gate | P2P | untested | TP2 Alternatives; LV |
| 26 | Torch symm-mem, quick AR, MSCCL++, NCCL symm-mem, NVLS, FlashInfer AR+norm fusion | other AR backends | — | src | torch symm-mem: capability 9/10 only ("Device capability 8 not supported"); quick AR: ROCm only; MSCCL++: world sizes 4/8/16/32/64, package absent (non-NVLS `default_allreduce_packet` ≤ 2 MB, `rsag_zero_copy`); NCCL symm-mem: forces CUMEM/NVLS, PCIe benefit unverified; NVLS needs NVSwitch; fusion auto-enables only on sm90/sm100 | — | **Ruled out or limited** | LV alternatives |
| 27 | 2×TP2 replicas with DFlash2 | two 2-rank servers; 1-peer ARs; custom AR V1 allowed at TP2 | **0.7-1.0×** of TP4 W1 (realistic); 0.9-1.2× if memory did not bind | est (from mE short-context data) | KV ~170k per replica (~340k total vs 881k) → only ~4-8 long streams | see 3.A | **Not recommended** for 12 × 60k | TP2 (all sections) — detail in 3.A |
| 28 | 2×TP2, no spec | fits 12 × 76k (~500k KV per replica) | ~150-250 tok/s total (0.4-0.6×) | est | loses DFlash2's +39 % | — | rejected | TP2 "Measured comparisons" |
| 29 | PP2×TP2 | per-GPU KV like TP4, 2-rank ARs | — | src | "Currently DFLASH speculative decoding only supports pp_size == 1" | — | **Blocked** | TP2 Alternatives |
| 30 | DP-attention | — | — | src/cl | "does not support dp attention on non-NPU devices"; meant for MLA | — | **Blocked** | TP2 Alternatives |
| 31 | Raise `mem_fraction_static` 0.83 → 0.88 at TP2 | more KV | +54k tokens per replica (~224k) | est | leaves ~1.1 GiB after TP2 graphs (estimated 1.3-1.6 GiB); "risky for chunked-prefill activations" | — | estimate only | TP2 memory |
| 32 | SGLang 0.5.22 upgrade | see 3.C | PCIe-IPC usable; robustness fixes; no published throughput delta | src/cl | full wheelhouse rebuild (~220 wheels / ~5 GB); still no #44401; correctness bugs still open | torch 2.14.1, FlashInfer 0.7.1 | not done | NEV — detail in 3.C |
| 33 | SGLang nightly ≥ 2026-10-16 | #44401 + post-release hybrid fixes | +~9 % slots | src/cl | `manylinux_2_39` wheel must be retagged for glibc 2.35 (untested) | FlashInfer 0.7.2, sglang-kernel 0.4.10 | not done | NEV Q1, Q3 |
| 34 | vLLM 0.32 | DFlash2 prefix-hit fix #55601, async DFlash, PCIe-IPC | no benchmark on any GPU | cl | open: #53912, #54360, #55533, #54690 (FP8 draft KV crash on SM89) | torch 2.13.0 base (same as ours) | not done | NEV Q2, Q3 — detail in 3.C |
| 35 | Post-capture KV sizing (`SGLANG_ENABLE_POST_CAPTURE_KV_SIZING`) + #42651 | size the draft KV pool after graph capture, so the target pool is not shrunk | not stated | src/cl | opt-in; the draft-pool fix is only in 0.5.22 | env exists in 0.5.21 (default False) | not evaluated | NEV Q1.4 |
| 36 | Fresh `SGLANG_CACHE_DIR` / `CUDA_CACHE_PATH`; never ship `nv/` | avoid a stale driver cache | correctness (#42917 PPL 9.98 vs 6.05) | mE | — | — | rule | NEV Q1.2; SLE §4 |
| 37 | Range-parallel copy of the weights to `/dev/shm` (fallback `/tmp`, then warming in place) | 16-32 workers × 64 MiB ranges with `os.pread`/`os.pwrite`; `posix_fadvise(DONTNEED)`; header verification; fail-open | ~3.6 min at 107 MB/s vs ~37 min single-stream | mE (amn 16 threads 10×) + est | uses 23.3 GB of RAM (peak ~47 GB); `/dev/shm` size unknown; SGLang must not open partial files | `df`/`statvfs` check at run time | proposed [I] (2026-10-23 measured: see §6) | SLE §2 |
| 38 | Range-parallel page-cache warmer (Franzen `precache` style) | warm the cache in place, 16-32 threads, 32-64 MiB chunks | — | cl | evidence on page cache is mixed (Jeroen: not filled; gcsfuse source and our 10-21 draft: filled) | — | fallback option | SLE §1-§3 |
| 39 | Re-shard the INT4 checkpoint into ~2-4 GB shards | stock per-file parallelism (8 threads per rank) gives ~8 streams | — | cl (Franzen) | new Kaggle model version | — | optional | SLE §3 |
| 40 | `--load-format presharded` / `fastsafetensors` | each rank reads only its ~1/4 slice (4 streams × ~6 GB) | — | src | untested with compressed-tensors INT4 + DFLASH; fastsafetensors package absent, per-file, `enable_gds=True` default | — | untested | SLE §3 |
| 41 | Stock `--weight-loader-prefetch-checkpoints` / `-num-threads` / `-drop-cache-after-load` | per-file prefetch split by rank | ~30 min at 10.5 MB/s for the 18.6 GB file | src + est | one stream per big file; switches the loader to single-thread | — | **does not fix our case** | SLE §3 |
| 42 | `--startup-weight-load-mode=overlap` | load during graph capture | — | src | rejects speculative decoding, TP > 2, quantization and draft workers | — | **Excluded** | SLE §3 |
| 43 | Port Franzen's lookahead prefetch patch | disjoint ranges of one shard, one shard ahead | — | cl | re-base two small hunks (`loader.py`, `startup_weight_load.py`) | — | option | SLE §2, §3 |
| 44 | InstantTensor loader (#40453) | fast loader | DeepSeek-R1 8×H200: 15.3 s vs 160 s | mE | optional extra `sglang[instanttensor]` | SGLang 0.5.22; vLLM 0.32 needs instanttensor ≥ 0.2.2 | not evaluated | NEV Q1.9, Q3 |
| 45 | Ship pre-built JIT/Triton/FlashInfer/Inductor caches as a dataset (not `nv/`) | skip the ~360 s first-bucket compile | ~360 s per cold start (~0.8 % of a 12-h rerun; ~7.5 % of an 80-min commit); Red Hat claims ~30 % | mO (in-session warm capture 12-14 s; relaunch 71-76 s) + cl | needs byte-identical wheels, sm_89 and a compatible driver; a greedy known-answer gate; set `SGLANG_JIT_CACHE_DIR` explicitly; tarball size unknown (hundreds of MB to a few GB) | — | **Untested across sessions** | SLE §4; TP2 Other costs |
| 46 | Progress-tied launch timeout (stall detector) | abort only on no progress (bytes copied, `/proc/<pid>/io` `rchar`, log lines); soft release | risk reduction, not speed | est + cl (Franzen 12-min soft deadline) | — | — | proposed | SLE §8 |
| 47 | v0.5.22 #43172 (clamp decode CUDA-graph max_bs to max_running_requests) | fewer graphs to capture | not stated | cl | — | 0.5.22 | not evaluated. A2 note: LV shows the first bucket is ~362 s and the other 9 sizes ~10 s together | NEV Q1.9; LV startup |
| 48 | v0.5.22 #36266 (warm the COW kernel before serving) | — | first-wave cold excess 924 → 28 ms | mE | — | 0.5.22 | — | NEV Q1.4, Q4 |
| 49 | v0.5.22 #40785 (FP8 KV quantization fused into the prefix-valid commit kernel, DFLASH path) | fused kernel | unmeasured | cl | Triton `fp8e4nv` works only on SM89+ (L4 OK; SM86 crashes, #45488) | 0.5.22 | — | NEV Q1.6 |
| 50 | Pinned image vs latest image | start reliability | unproven | cl (cm391 anecdote) | needs the cp312 wheelhouse to pass on the latest image | — | possible test | SLE §5 |
| 51 | Token-efficient tune (Swift 1.5 INT4 drop-in; ThinkingCap; Qwen Pi) | fewer thinking tokens per request | on ARC "nearer −10 to −25 % mean" → 1.18-1.33× more task-requests | cl + est | licence (Swift and ThinkingCap not OSI); accuracy cost on hard reasoning; DFlash2 AL may drop | user decision | untested on ARC | TEV — detail in 3.D |
| 52 | Lower effort (medium / low) | — | — | mE (42.4 vs 22.8 / 13.2 %) + mO (W3H 17/172) | — | — | **Rejected** | TEV §2 |
| 53 | Thinking-budget truncation below 60k | — | — | mO (49k LB 32.08 < 60k 34.58) | violates the 60k floor rule | — | **Excluded** | TEV §3 |
| 54 | Strict thinking / custom logit processor as a 60k safety cap | force `</think>` at the budget | — | cl | built-in Qwen3 IDs wrong for 3.8 (151667/151668 vs 248068/248069); strict thinking failed in #25536; may conflict with spec decoding | temperature-0 A/B with DFlash2 | untested | TEV §3 |
| 55 | Deterministic inference (`--enable-deterministic-inference`) for correctness gating | batch-invariant ops | — | cl | unverified with DFLASH + Mamba on 0.5.21; disables PCIe-IPC AR | — | not relied on | DFW Q5; NEV Q1.7 |
| 56 | Dense BF16 lm_head | reduces #36548 errors (1/304) | — | mE | — | — | **already true** in our RedHat INT4 (lm_head and GDN unquantized) | DFW Q4 |
| 57 | Keep `--speculative-draft-model-quantization unquant` | avoids ~0 % acceptance (#39087) | — | mE + mO (our AL 4.2-4.95) | — | — | rule | DFW Q4; NEV Q1.2; TEV §4 |
| 58 | ReplaySSM spec-verify for DFlash on GDN | snapshot-free verify | — | src | "requires a KDA" in every version; PRs #36683 and #42209 open; compact replay not bit-exact (79/80 diverged) | — | **Unavailable** | NEV Q1.4 |
| 59 | vLLM DFlash speed features | async scheduling for DFlash (#58065, 0.31) with context K/V precompute in the draft graph (#57632); fused DFlash2 grouped conv (#55960, 0.31); in-graph metadata for DFlash/DSpark drafts (#60692, 0.32); `--max-num-active-seqs`; queue rework (#58947); fix for the throughput cliff at max_num_seqs not a multiple of 8 (#57355) | none stated | cl | async + MTP3 + FP8 Triton gave garbage on SM89 under concurrency (HyperQwen #121, vLLM 0.28) | vLLM 0.31/0.32 | not evaluated | NEV Q2 |

### 3.A 2×TP2 replicas: special detail (TP2; LV)

**Memory per GPU in one TP2 replica** (all [E] or [D]; "needs a startup test"):
- **Target weights:** ~9.24 GiB (2 × 4.62), "if all target parts are sharded (not verified per tensor)".
- **Drafter weights:** two estimates disagree (see §4):
  - LV: ~2.5 GB = 2 × (1.49 − ~0.52 replicated) + 0.52;
  - TP2: **2.19 GiB**, fitted as 0.79 replicated + 2.79 GiB sharded, from 1.49 at TP4 and a ~3.85 GB file.
- **Weights total:** ~11.7 GB (LV) or ~11.4 GiB (TP2), vs 6.11 GB at TP4. The GPT notes' 12.22 GB is "PARTLY TRUE", an upper-bound style estimate. Going TP4 → TP2 adds ~5.3 GiB of weights per GPU.
- **MTP alternative:** ~2.8 GiB (the vocab-parallel embedding and lm_head are ~2.37 GiB at TP2).
- **KV per token per GPU:** target 16,384 B; drafter K+V 5,120 B (5 layers × 4 heads × 128 × 1 B × 2); total **21,504 B** vs 10,752 at TP4. MTP 17,408 B; no-spec 16,384 B.
- **GDN state per request:** 75.5 MB total (fp16) = 37.7 MB per GPU at TP2 (37.4 MiB per slot per GPU including conv).
- **Per running request:** 4 main slots + 8 intermediate slots (DFlash2 W8) ≈ 12 × 37.4 MiB ≈ **449 MiB per GPU**.
- **Sizing rules** [S, `kv_cache_configurator.py`]:
  - KV budget = free after weights − `pre_model_load_memory × (1 − mem_fraction_static)` − multimodal reserve − Mamba state;
  - intermediate scratch = `per_req × (capped_reqs + 1) × speculative_num_draft_tokens`, with `capped_reqs = min(max_running_requests, max_mamba_cache_size // ratio)`;
  - main state = `(max_mamba_cache_size + 1) × per_req`;
  - ratio for extra_buffer_lazy with overlap = 4.
- **CUDA graphs:** TP2 graphs are estimated at 1.3-1.6 GiB (TP4: 1.03 GiB) because activations are not sharded. Unmeasured.

**KV tokens per replica** (TP2 per-replica table; 6 running per replica, Mamba cache 28, fp8 KV):

| Arm | Rest after slack (GiB) | Mamba main | Intermediate | KV GiB/GPU | Bytes/token/GPU | KV tokens per replica | System (×2) |
|---|---|---|---|---|---|---|---|
| DFlash2, R=6, mem 0.83 | 6.50 | 1.06 | 2.04 | 3.40 | 21,504 | **~170k** | ~340k |
| DFlash2, R=6, mem 0.88 | 7.58 | 1.06 | 2.04 | 4.48 | 21,504 | ~224k | ~448k |
| DFlash2, R=4 (K=19), mem 0.83 | 6.50 | 0.73 | 1.46 | 4.31 | 21,504 | ~215k (4 × 54k) | ~430k (8 streams) |
| DFlash2 weight sensitivity (9.0-9.6 + 1.9-3.0 GiB) | — | — | — | 2.2-3.9 | — | 111k-196k | 222k-392k |
| MTP (4 draft tokens), R=6, mem 0.83 | 5.89 | 1.06 | 1.02 | 3.81 | 17,408 | ~235k | ~470k |
| MTP, mem 0.88 | — | — | — | 4.89 | 17,408 | ~302k | ~604k |
| No-spec, R=6, mem 0.83 | 8.69 | 1.06 | 0 | 7.63 | 16,384 | **~500k** | ~1.0M |
| No-spec, mem 0.88 | — | — | — | 8.72 | 16,384 | ~571k | ~1.14M |
| **TP4 W1, measured** | — | 1.04 | 1.83 | 8.82 | 10,752 | — | **881,336** |

- **Need:** 6 × 76k = 456k and 6 × 60k = 360k per replica; system-wide ~700-920k (12 × 58-76k).
- **Fit with DFlash2 (~170k per replica):** 2 streams of 76k or 2-3 of 60k per replica (~4-6 total), or 4 × 54k per replica (8 total). No-spec holds 6 × 76k (456k of 500k) with little prefix-cache headroom.
- **System KV:** falls from 881k to ~340k with DFlash2, about 60 % less [D].

**Concurrency, throughput and the basis for the gain range:**
- **Basis:**
  - DP wins only at short context and high concurrency;
  - total c ≤ 4: TP4 ≥ DP (0.7-1.0×);
  - c = 8-32 at short context: DP 1.3-1.95×;
  - c ≥ 64: DP 2.1-2.4×;
  - with MTP the single-stream TP2 → TP4 decode gain disappears (club-3090), "so spec decoding shifts the break-even toward DP".
- **Why it could help:** 12 × 8 = 96 verify rows → ~0.98 MB per AR, a "high concurrency" token count. TP2 uses 1-peer reductions, and custom AR is allowed at TP2.
- **Why it may not:**
  - each TP2 GPU reads twice the weights per step (~11.4 GiB vs ~6.1 GiB);
  - per-GPU KV bytes per step are unchanged at equal per-stream context;
  - prefill per replica is slower (club-3090: TP2 prefill at 90k is 38 % slower than TP4: 1,202 vs 1,948 tok/s);
  - Duck prefix reuse gets worse (170k vs 881k radix space; E12 EVAL30 already lost prefixes on 144 of 423 requests at ~870k).
- **Step model [E]:**
  - TP4 ≈ 128 ms (weights ~22 + KV ~17 + AR ~35-55 + residual ~35-50);
  - TP2 replica at 6 streams (48 rows) ≈ weights ~41 + KV ~17 + residual ~40-47 + comm ~18 (SHM, 2 ranks, 0.49 MB) ≈ 115-123 ms → ~215-230 tok/s per replica, **~430-460 total, 1.0-1.1×** ("if KV fit");
  - with working P2P custom AR at TP2 (comm ~3-5 ms): ~1.15-1.2×.
- **Realistic DFlash2 case [E]:** ~3-4 streams of 55-60k per replica, ~150-180 tok/s each → **~300-360 total (0.75-0.9×)**, with half the concurrent Duck branches. Range **0.7-1.0×**.
- **No-spec 2×TP2:** ~150-250 tok/s (0.4-0.6×).

**Routing and implementation** (0.5.21):
- **Simplest:** two independent `sglang.launch_server` processes (`CUDA_VISIBLE_DEVICES=0,1` on port 30000 and `2,3` on 30001) with client-side sticky routing (Duck maps `branch_id % 2`, or the replica with the fewest resident tokens at branch creation, to a port).
- **Native `--dp-size 2`:** `routed_dp_rank` in the OpenAI protocol sends a request to a fixed rank (`maybe_external_dp_rank_routing`; `data_parallel_rank` is deprecated). The docs call native DP "highly not recommended": built-in balancing only (round_robin, total_requests, total_tokens), "no cache-aware routing, limited observability, no fault tolerance". `--enable-metrics-for-all-schedulers` is needed for per-replica metrics. Unverified with the hybrid Mamba radix cache.
- **Model Gateway** (`sglang-router` / `smg` 1.13.0, cp38-abi3, manylinux_2_17):
  - **not in our wheelhouse** (225-line SHA256SUMS);
  - policies: cache_aware, round_robin, power_of_two, consistent_hashing, prefix_hash, manual, random, bucket;
  - `cache_aware` defaults: `--cache-threshold 0.3`, `--balance-abs-threshold 64`, `--balance-rel-threshold 1.5`, `--eviction-interval-secs 120`, `--max-tree-size 67108864`;
  - **`--retry-max-retries 5`** on 408/429/500/502/503/504 could silently re-run a 60k generation → `--disable-retries`;
  - `X-SMG-Routing-Key` header for the manual policy (secondary source);
  - Prometheus at `0.0.0.0:29000/metrics`.
- **Ports:** ZMQ channels use fresh temp IPC names; the NCCL port comes from `get_free_port()`, so pin `--nccl-port` per server to avoid a race.
- **Monitoring:**
  - two `/metrics` endpoints (`--enable-metrics` on each);
  - assert real concurrency per replica;
  - read the "capped" line per log (`/get_server_info` misreports);
  - the 11-of-12 bug applies per replica (a replica sized for 6 runs 5 unless the cap is one higher).

**Custom AR at TP2** [S]:
- V2 requires full NVLink. Legacy V1 `CustomAllreduce` is allowed at TP2 with `_MAX_CAR_SIZE` 8 MiB; the ~0.5 MB messages qualify.
- It needs P2P/IPC. By default SGLang skips the real P2P test (`enable_p2p_check` False monkey-patches the check to `True`), so use `--enable-p2p-check` or `--disable-custom-all-reduce`.
- The P2P cache file is keyed by `CUDA_VISIBLE_DEVICES`.
- Expected +10-15 % per replica if P2P works [E from club-3090]. Without P2P: (a) raise and fall back to NCCL, or (b) at worst hang or corrupt [E].

**Cold start for two replicas** [E]:
- Staggered: replica B after A's capture behaves like a cached relaunch (~70-80 s); total ~8-9 min vs ~9.5 min for TP4 W1.
- Parallel start: both pay the ~6 min JIT and contend for CPU.
- TP2 kernel shapes differ, so TP4 JIT caches do not help.
- Weights are shared through the page cache; host RAM holds target (~19.5 GB) + drafter (~3.85 GB).
- vLLM prefetch cut weight load 64 s → 4.8 s and engine start 600 → 532 s [M, ours].

**Ranking in the TP2 doc:** (1) capture topology and NCCL transport; if P2P works, `NCCL_P2P_LEVEL` at TP4 (+5-15 %). (2) PCIe-IPC or patched custom AR at TP4 with a correctness gate. (3) 2×TP2 "only if the stream count drops to 6-8 per run or contexts shrink. Neither is planned: the 60k thinking floor is fixed".

### 3.B DFlash draft width 4 / 6 / 8: special detail (DFW; LV)

**Code path** [S]:
- `--speculative-num-draft-tokens` W ≠ 8 logs `DFLASH block size mismatch …` and calls `set_block_size(W)`. That only stores the integer on the model and the conv layers. No weights change.
- Block layout `[bonus, mask × (W−1)]` at positions prefix_len..prefix_len+W−1; the grouped two-tap conv uses `position % W` and resets taps at block boundaries.
- Drafter attention is **non-causal** (`is_causal: false` → `ENCODER_ONLY` → FlashInfer `causal = False`), so W4 is **not a truncation** of W8: the first positions no longer see the extra masks. The selector walk is causal.
- Verify graph recaptured with `num_tokens_per_req = W`; AR payload 96 → 48 rows at W4.
- Per-request KV reserve per decode step = 2 × W.
- Max accept length = W (W−1 drafts + 1 bonus); "more than 4 output tokens per step at width 4" is FALSE.
- Checkpoint: `block_size 8`, `conv_group_size 16`, `conv_kernel_size 2`, `selector_rank 256`, `selector_top_k 16`, `target_layer_ids [5,19,33,47,61]`, 5 sliding-window layers (2,048), `mask_token_id 248070`; card: "Speculation block size: 8 (7 draft tokens per verification step)".

**Acceptance predictions** [inference], L_W = 1 + Σ s_k, assuming truncation keeps the same predictions:

| Model | L3 | L4 | L6 | L8 |
|---|---|---|---|---|
| GPT geometric α = 0.811 | 2.47 | 3.00 | 3.79 | 4.30 |
| Geometric fit to our measured 4.55 (α = 0.8295) | 2.52 | 3.09 | 3.95 | 4.55 |
| inco per-position recall as conditional (Qwen3-4B) | 2.54 | 3.08 | 3.84 | 4.27 |
| Lower bound, flat survival (all-or-nothing blocks), L8 = 4.55 | 2.01 | 2.52 | 3.54 | 4.55 |
| "Positions 4-7 ≤ 0.1" anecdote | ≈ L8 − ≤0.4 | — | — | ≤ 4.4 (≈ 3.3-3.5 realistic) |

- **Realistic ranges:** L4 ≈ 2.8-3.1; L6 ≈ 3.7-3.95.
- **Cross-check in LV:** α = 0.8109 gives L4 = 3.002, L6 = 3.785, L3 = 2.469.
- **Correlation:** acceptance at successive positions is correlated (DFlash paper: 35.7 % full-block acceptance), which pushes reality toward the flat lower bound.

**Step cost and throughput model** [inference]:
- Calibrated at c12: AR 289.4 tok/s ≈ 41.5 ms; DFlash W8 403.7 tok/s at L = 4.62 ≈ 137 ms.
- step(W) = a + 12·b·W + d·W/8, with drafter cost d ∈ [10, 40] ms.
- Predicted c12 total tok/s for W = 3 / 4 / 6 / 8:
  - geometric: 418-434 / 435-446 / 427-431 / 399;
  - flat lower bound: 334-347 / 355-364 / 382-385 / 399.
- So **W4 ≈ +10 % (geometric, Q6 says up to +12 %) to −10 %; W6 ≈ +7 % to −4 %.** "The linear-in-W cost assumption favours narrow widths. If the verify forward at 48 tokens per step is memory-floor-bound on L4 (W4A16 Marlin), the narrow-width gain shrinks. **The sign must be measured, not assumed.**"
- **GPT-note model, reproduced in LV:** T4 at a 70 ms step = 514.6 tok/s; L4 needed for 550 / 650 tok/s = 3.208 / 3.792; width-4 table 2.5 → 428.6, 3.0 → 514.3, 3.3 → 565.7, 3.5 → 600.0, 3.8 → 651.4, 4.0 → 685.7 tok/s. The 70 ms W4 step is a GPT-note assumption.

**Correctness by width:**
- #36548: draft depth 4 cut wrong answers 111 → 37 of 304 (each of the other single ablations did the same); not removed.
- Width 2 crashes at graph capture (third-party); widths 4, 6 and 8 work.
- Width 3 "should work" (untested) and is predicted no better than W4 (L3 ≈ 2.5).
- #43514 used width 6 (with mixed chunk + grammar).

**External indirect evidence:** see §2.3 (paper compute-bound advice and Table 8; #36136 gains from c ≥ 12; EAGLE adaptive defaults; llama.cpp n_max 4-5; z-lab ≤5 for quantized MLX; oMLX 5).

**Memory side effects:**
- A narrower width shrinks the 2 × W per-request KV reserve and the verify workspace.
- The intermediate SSM cache is `requests × W × slot` (1.97 GB at 14 × 8), so it shrinks linearly ("Fewer tokens would shrink the intermediate scratch linearly", TP2 memory Gaps).

**Gate the doc proposes** (DFW Q5; a fact about the doc, not a design here):
- **Arms:** T (target-only), D8, D4, optional D6, all with the production flags.
- **Step 1, health:** accept means below 3.5 / 2.5 / 3.2 at W8 / W4 / W6 fail; accept rate 0.00 in any 40-step window fails; >50 identical token IDs fails; the mismatch warning must appear.
- **Step 2, short greedy exact match** (thinking off, 20 prompts × 256 tokens, `top_logprobs=2`, A/A across a restart): margin <0.1 nat = near-tie; a flip at >1 nat = bug flag.
- **Step 3, answer-level probe** (#36548 template): 150 trials per arm at c12 plus 50 serial; fail if D wrong >2 % or McNemar p < 0.01.
- **Step 4, workload equivalence:** 32 ARC prompts at ~60k, c12, ~1 GPU-hour per arm, pass within ±2 tasks at 1 sd.
- **Step 5, batch-invariance control:** T alone vs T inside a 16-batch.
- **A/B:** identical recorded ARC prompt mix at a real 12 running (assert from the logs), ~10-15 min steady state per width, tok/s and accept length by context band.

**Gaps:**
- No measurement anywhere of this checkpoint (trained at b8) at b4 or b6.
- No per-position acceptance from any official source; 0.5.21 logs means only (a histogram would need a patch, `spec_trace_dir` or request `meta_info`, unverified).
- No L4 data on how the W4A16 Marlin verify GEMM scales between 48 and 96 tokens.
- The inco blog does not state the training block size (inferred 8 from the config).

### 3.C Newer SGLang / vLLM versions: special detail (NEV; DFW Q3-Q4)

| Version | Adds for L4 / Ada | Hybrid Mamba / GDN | DFlash / MTP | Still open or missing | Install on Kaggle (Python 3.12, glibc 2.35, CUDA 13.0 driver) |
|---|---|---|---|---|---|
| **SGLang 0.5.21** (ours) | PCIe-IPC AR code present but **dead** (FlashInfer 0.6.18 has no `pcie_ipc_ar.py`); `SGLANG_ENABLE_POST_CAPTURE_KV_SIZING` exists (default False) | UnifiedRadixCache FULL+MAMBA; extra_buffer_lazy; HiCache wired; `buffer_only` rejects MAMBA trees; int8 Mamba checkpoints | DFLASH W8; no adaptive or ragged verify for DFLASH; #28180 fix included | #44401 pre-patch shape; #41872 clear absent | current wheelhouse (torch 2.13.0, FlashInfer 0.6.18) |
| **SGLang 0.5.22** (2026-10-19) | FlashInfer 0.7.1 → **PCIe-IPC AR usable** (kernel "No architecture restriction"; optional memory-op protocol needs SM120); #40785 fused FP8 KV quantization in the prefix-valid commit kernel (Triton `fp8e4nv`, SM89+); GDN items #43570, #43227, #43332 (#41486 is SM90 only); no SM89-targeted Marlin/GDN/FlashInfer entry | #43029 ("Hybrid linear models could resume from wrong mamba states when speculative decoding and structured output were used together"); #36266 COW warm 924 → 28 ms; #42354 / #43201 (ChunkCache removed); #43567; #42467; HiCache #36345 (Mamba buffer-only), #40223, #43596, #40915 | #43012 block verification (sampled drafts only; +1.76 % at c8 GB300); NaN isolation #41736 and #41737; #42709 (compact-verify graph wrong tokens); #42002 (Qwen3.5 aux-layer capture); #42651 (draft KV pool after capture); #43037 (packed target verify opt-in); #43172 (graph max_bs clamp); #43177; #43461 (mixed chunk on the Full prefill graph); #40453 InstantTensor; `tc_piecewise` removed (#42287); scheduler fixes #43298, #43290, #43564, #41872 | **No #44401**; #36548, #38009, #43514/#43518, #37475, #39087, #36127, #36889, #38846, #35150, #36014 open; ReplaySSM + DFLASH needs KDA; `MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3` unchanged; new #45400 (HiCache host hits drop to zero) | cp312 `manylinux_2_34` wheel (installs on glibc 2.35); torch 2.14.1 (still the CUDA 13.0.3 toolkit; NCCL 2.30.7; triton ~3.8.0); FlashInfer 0.7.1 + jit-cache 0.7.1+cu130 (the '+' trap); sglang-kernel 0.4.9; transformers 5.19.0; cutlass-dsl 4.8.0; still the sdist-only `cuda-tile==1.6.0rc5` (override with 1.6.0 or 1.7.0 `--no-deps`); nvcc 13.0.88 pin unchanged; ~220 wheels / ~5 GB; one CPU build + one publish run |
| **SGLang nightly ≥ 2026-10-16** (`0.5.23.dev…`) | FlashInfer 0.7.2 | #45435 (checkpoint tracking through mixed chunked prefill); #45472 (plan hybrid Mamba extend batches without host-device syncs) | — | DFlash2 correctness bugs still open | wheel tagged **manylinux_2_39** (pip rejects it on glibc 2.35); native `.so` files reference at most GLIBC_2.34 (heuristic scan); needs a retag or rename (untested on Kaggle); no cuda-tile pin; sglang-kernel 0.4.10; '+' in file names. Lower-risk alternative: 0.5.22 + in-process monkey-patch of #44401 + #41872 |
| **vLLM 0.30** (2026-09-22) | PCIe-IPC AR backend #53576 but FlashInfer 0.6.18.post1 (no kernel); #54646 faster graph capture (H200) | Qwen3.5/Qwen3-Next Triton and GDN RMSNorm warmed at startup (#54797, #54251); Mamba state cached at the EAGLE resume position (#53945) and for padded tails (#55178); FlashInfer ReplaySSM for Mamba2 (#52506); `all` Mamba cache mode deprecated | DFlash2 present since 0.29 (#52816); MR-V2 adaptive verification (#52228) | — | cp38-abi3 manylinux_2_28; torch 2.13.0 |
| **vLLM 0.31** (2026-10-05) | FlashInfer 0.7.0.post1 → PCIe-IPC usable; throughput-cliff fix #57355 | #58368 (prompt-tail hits with MTP); #53864 (CuteDSL BF16 GDN prefill divergence) fixed; `--enable-mamba-shared-prefix-checkpoint` | **async scheduling for DFlash (#58065)** + K/V precompute in the draft graph (#57632); **fused DFlash2 grouped conv (#55960)** | #54360 (DFlash prefix hits lost every other request); crash-loop report on 2×3090 PP=2 (HyperQwen #328) | same shape |
| **vLLM 0.32** (2026-10-18) | NVFP4 KV on SM8x / SM12x (#46963); custom AR under `VLLM_BATCH_INVARIANT=1` (#58623) | **#55601** (seed the hybrid Mamba state index with `mamba_block_size` on prefix hits; cures DFlash2 acceptance → 0 or IMA on a hit, per HyperQwen); #60533, #59759; M-RoPE cache 4× smaller | in-graph metadata for DFlash/DSpark drafts (#60692); ReplaySSM for MTP (#52928, Mamba2) | **#53912** (MTP + prefix cache corruption on hybrids); #54360; #55533 (MTP window; capacity effect); **#54690** (FP8 draft KV crashes on SM89 → plan BF16 draft KV); #58231 thinking-budget fix main-only; #58894 still open | cp38-abi3 manylinux_2_28 (327 MB); torch 2.13.0 (same as our 0.5.21 base); FlashInfer 0.7.0.post1 + jit-cache (excluded from install_requires, #60052; `vllm download-kernels`); cutlass-dsl 4.8.0; instanttensor ≥ 0.2.2 is a hard dependency; transformers ≥ 5.16.1, < 5.20.0; Python ≥ 3.11 — "the simplest of the three" |

- **NEV conclusion:** "No upgrade can be justified on published throughput for our setup." The justification must be specific fixes: #44401 (+~9 % slots), PCIe-IPC AR (unmeasured on L4), and #55601 for vLLM.
- **Validation rule:** every upgrade needs our own paired measurement: same notebook, 12 real running, tok/s by context band, plus greedy and A/A correctness.

### 3.D Token-efficient model variants and licence: special detail (TEV)

| Variant | Licence (OSI?) | INT4 / W4A16 availability | Claimed thinking reduction | Accuracy cost (claimed) | DFlash2 / MTP on it | Status |
|---|---|---|---|---|---|---|
| `ukisai/Swift-Qwen3.8-27b` (1.0) | Swift Open License v1.0: commercial use "not licensed" at ≥ US$1M gross revenue (Sec. 5); not OSI, not in SPDX; HF `license: other`, not gated | card rows for W4A16 mixed and AWQ INT4 | GPQA 41.0 % mean / 58.3 % median; Terminal-Bench 26.5 %; by effort 41.0 % (xhigh) / 22.7 % / 25.8 % | AIME 98.67 → 94.00 | MTP head byte-identical to base; `layers.0` and `layers.63` MLP changed; embed, lm_head and `layers.5.linear_attn.out_proj` identical | untested on ARC |
| `ukisai/Swift-1.5-Qwen3.8-27b` | same Swift licence | **`ukisai/Swift-1.5-Qwen3.8-27b-INT4`**: compressed-tensors pack-quantized, 4-bit g128, `actorder: static`, separate `model_mtp.safetensors`, 19.48 GB vs RedHat 19.47 GB (same layout) — the closest drop-in; also W4A16-AWQ (19.58 GB), W4A16-AutoRound, bjonor GPTQ | GPQA 41.9 % mean; AIME 40.0 %; **Terminal-Bench 16.3 % mean / 0.1 % median** | AIME 98.67 → 96.00; ERQA −2.05 | MTP head byte-identical; embed_tokens and lm_head changed (a plausible drafter mismatch channel); club-3090 #1746: DFlash2 AL 3.85 (7 drafts), 3.59 (5 drafts); MTP 2.97 | untested on ARC; HF user reports of looping and over-generation |
| `bottlecapai/ThinkingCap-Qwen3.8-27B` | PolyForm Small Business 1.0.0 (<100 workers and <$1M revenue) + personal-use grant; `isOsiApproved: false`; gated | vendor ships only FP8, NVFP4, NVFP4A4-AWQ, GGUF, MLX; third-party wasifb AutoRound W4A16 (19.02 GB), halt95 compressed-tensors W4A16 "Merlin" (17.05 GB, has MTP + kv_scales), Tomeamis GPTQ (gated) | mean −37.2 %, median −42.9 %; **Terminal-Bench −10.7 %** (total −7.5 %) | −0.86 pp average; AIME −3.86 | card: original MTP and DFlash2 work "out of the box"; DFlash2 AL 3.1-4.2 top-token, 3.3-4.9 probabilistic; MTP 2.5-3.3 | untested on ARC; could not be byte-checked (gated) |
| `bytkim/Qwen3.8-27B-pi` (Qwen Pi) | **Apache-2.0** (the only OSI-licensed efficiency tune found) | not stated | medium matches base xhigh on Terminal-Bench with ~41 % fewer output tokens; SciCode at xhigh ~23 % fewer | — | ships `z-lab/Qwen3.8-27B-DFlash2` unchanged | "only worthwhile if you run the Pi agent harness" (MindStudio) |
| ThinkingCap-Qwen3.6-27B | Apache-2.0 (gated: auto) | — | 3.6: −45.8 % out-of-domain | −0.7 pp | — | an older model generation, not 3.8 |
| Training-free overthinking-marker logit penalty (arXiv 2606.00206) | base stays Apache-2.0 | n/a | −12-23 % CoT on 1.5B-32B models | "preserved or improved" | n/a | Apache path (a) |
| Own length-aware fine-tune of the Apache base | Apache-2.0 if released | — | — | — | — | Apache path (c) |
| Flash-Next tunes (ThinkingCap and Swift1.5 Flash-Next; Swift-1.5 Flash-Next W4A16-AWQ) | as above | AWQ exists | — | — | — | relevant to E13 |

- **Competition rule** as quoted in topic 750575: Rule 1.a requires an "open-source model and weights as defined by the Open Source AI Definition". OSAID requires "OSI-approved terms" and use "for any purpose without having to ask for permission". So Swift and ThinkingCap-3.8 "appear incompatible with prize eligibility".
- **Host answer:** none yet (746307, 750575).
- **arcprize.org:** own code must be CC0 or MIT-0; third-party code at least Apache-2.0 or GPLv3; silent on model licences.
- **Leaderboard-only use** is "a separate question"; this is not legal advice and is for the user or hosts to decide.
- **Expected effect on ARC:** −10 to −25 % mean tokens → **1.18-1.33×** more task-requests in 12 h at fixed throughput; the headline −40 % would give ~1.67×. This holds only if per-request accuracy holds. Accuracy loss lands on the hardest reasoning. A 10-20 % AL drop would offset part of the saving, so the metric is "tasks finished per hour on L4".
- **Field evidence:** no Kaggle team has reported Swift or ThinkingCap on ARC-AGI-2/3. The only mention is Darshan Makwana (746149), who did not run them. Internally Swift was flagged (E11-04 self-audit "could double coverage if ARC accuracy holds"; E14 gaps note item 7) but never tested.
- **Metrics TEV lists for a paired test:** correct per task; tokens per request (mean, median, cap-hit rate); EOS-without-`</think>` rate; DFlash2 AL per 10k window; tasks per GPU-hour.

### 3.E Agent-level effective throughput: special detail (AET)

**Mechanics** [S]:
- With `prefill_decode_interval` 0 and `enable_mixed_chunk` False, a pending prefill always runs first and **stalls decode for all streams**.
- Prefill runs at ~1,500 tok/s vs ~400 tok/s decode, so an uncached prompt token costs ~0.27 generated-token-equivalents.
- The decode log's `gen throughput` includes prefill stalls; decode lines print every 40 iterations.

**Prefill share of the window:**
- EVAL30 55.1 %, which alone explains "effective ≈ half of decode rate";
- well-behaved runs 13.3-20.4 %;
- G4B 19.8 % (406-425 s of 2,050 s).

**Re-prefill sources:**
- **Reply re-prefill:** in 72-91 % of later requests the hit stops at the previous prompt's end (`prefill_end`), a median 15-32 tokens before it, on the 64-token Mamba grid. The previous reply (reasoning + tool call) is prefilled again; the reply reuse median is ≈ 0.
- **Template cause:** the chat template re-renders the assistant turn as `'<|im_start|>assistant\n<think>\n' + reasoning_content` (trimmed) `+ '\n</think>\n\n' + content`, and rebuilds tool calls from parsed arguments. "Any whitespace trim or non-canonical sampled tokenization makes the re-tokenized history diverge."
- **When reuse does work:** `full_reply` hits show the decode-end Mamba state can be hit when the history matches.
- **G4B split:** replies ~37 %, zero/partial at low KV ~47 %, first requests ~5 %, tool-result deltas ~11 %.
- **EVAL30:** eviction re-prefill 8.39 M tokens (KV thrash with 30 time-sliced branches × 50-100k > the 866,744 pool).

**Cache:**
- Token-weighted hit rates: 59.6 % (EVAL30), 75.0 % (G4B), 78.8-88.7 % (6-stream runs). Franzen reached 93.43 %.
- Only ~20 of 56 Mamba slots are free for cached checkpoints at 12 running. A hybrid hit needs both the KV and a Mamba checkpoint at the exact depth.
- Misses rise after long replies; probable cause is LRU loss of the end-of-prefill checkpoint after a long decode (Franzen's mechanism; not proven).

**Occupancy:**
- "11 running + 1 queued" for 50 % (EVAL30) and 25 % (G4B) of the time, costing ≈ −9 %.
- The G4B probe held 6 running for 27.7 %.
- Idle with nothing in flight ≤ ~1 %.

**Scenario tables** (AET §6, [D]; time = generated / decode rate + uncached / 1,500-1,578 + edge time):

| G4B rev 6d scenario | Decode rate | Uncached | Time (s) | Effective tok/s |
|---|---|---|---|---|
| Measured | 352.7 | 669,980 | 2,050 | 279 |
| S1: probe fixed, steady 11 running | 397 | 669,980 | 1,888 | ~303 |
| S2: 12 running | 436 | 669,980 | 1,759 | ~325 |
| S3: S2 + no reply re-prefill (−245,462) | 436 | ~424,500 | 1,604 | ~357 |
| S4: S3 + no zero/partial misses (−316,865) | 436 | ~107,700 | 1,402 | ~408 |
| Franzen-level 93.4 % reuse on 2.68 M prompt tokens, 12 running | 436 | ~176,900 | 1,446 | ~396 |

G4B inputs: generated 572,035; prefill rate 1,577; edges 22 s.

| EVAL30 scenario | Decode rate | Uncached | Time (s) | Effective tok/s |
|---|---|---|---|---|
| Measured | 375.8 | 9,597,852 | 14,046 | 168 |
| E1: no eviction re-prefill (−8,394,803) | 375.8 | ~1,203,000 | ~7,126 | ~331 |
| E2: E1 + no reply re-prefill (−1,064,762) | 375.8 | ~138,000 | ~6,405 | ~368 |
| E3: E2 at 12 running, ≥60k context | 419.6 | ~138,000 | ~5,750 | ~410 |

EVAL30 inputs: generated 2,357,584; prefill 1,475; edges 38 s.

- **Ceiling:** the decode rate itself, ~400-460 tok/s at 11-12 running = 1.4-1.5× over G4B, ~2.2-2.4× over EVAL30.
- **Hidden-scale target:** 240 × 60k = 14.4 M tokens needs ≈ **370 tok/s** over the 11.46 h window. Going 279 → ~360-400 = +29-43 % more tokens. Levers 2 + 3 (or 2 + 5) reach that range on the G4B profile.
- **Caveats:** the scenarios assume removed prefill time converts fully into decode at the stated rate; they ignore context growth and acceptance changes; the 12-running clean samples are few (29 lines).

---

## 4. Contradictions or corrections between documents

1. **Is the FlashInfer PCIe-IPC all-reduce usable on our 0.5.21 install?**
   - **LV** ("Source: custom all-reduce at TP2…" Takeaway): "The only non-NCCL option at TP4 on PCIe is FlashInfer PCIe-IPC all-reduce, through env `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1`. Its availability in flashinfer 0.6.18 on sm_89 is unverified." Inferences: "The TP4-vs-2×TP2 comparison should include a third arm: TP4 with `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1`. It is a single env var and needs no topology change." Summary fact 4: "SGLang 0.5.21 ships a FlashInfer PCIe-IPC all-reduce…".
   - **NEV** Q1: "It needs FlashInfer ≥ 0.7.0, and 0.5.21 pins 0.6.18, so the backend is dead code in our current install." Q1.7: "`flashinfer/comm/pcie_ipc_ar.py` is **absent at tags v0.6.18 and v0.6.18.post1** … So with our 0.5.21 wheelhouse (flashinfer 0.6.18) the SGLang backend can never activate."
   - **TP2** ("Custom all-reduce at TP2"): "`flashinfer_python-0.6.18` … has **no** `pcie_ipc` code [M, local zip scan]". Its own Gap ("I did not verify which FlashInfer version first ships `PcieIpcAllReduceWorkspace`, or whether it supports sm_89") is answered by NEV: v0.7.0, "No architecture restriction".
   - **Correction:** the env-var-only arm on 0.5.21 cannot activate. It needs SGLang 0.5.22 (FlashInfer 0.7.1) or vLLM ≥ 0.31.

2. **Does a DFlash2 width below 8 work?**
   - **TP2** (memory Gaps): "Whether DFlash2 accepts `--speculative-num-draft-tokens` below its block size of 8 was not checked." Alternatives: "I found no evidence on a smaller DFlash2 verify width."
   - **DFW** Q1: "Setting a width of 4 or 6 is a supported code path. SGLang logs a warning, resizes the verify block…"; a third party reports budgets 4, 6 and 8 work.
   - **LV**: "It is a pure runtime override; no weights are retrained or reshaped."
   - **Correction:** the code path is supported; only acceptance and throughput at W ≠ 8 are unmeasured.

3. **The "median 4.55" acceptance used for the α = 0.83 fit vs the per-line log parse.**
   - **DFW** Q2: "we measured a median accept length of 4.55 at W=8" (source "Production ARC solver, 11 running", `E11-05 run1 analysis`). It uses this to argue the "positions 4-7 ≤ 0.1" anecdote "**cannot** hold for our workload".
   - **TEV** §4: "Our E11-05 on L4 … DFlash2 AL 3.8-4.5, falling from 4.43 to 3.77 across context, median 4.55". The median here lies outside its own stated range.
   - **LV** (acceptance means): "E11-05 EVAL30 is dominated by B=11: n=2,371, mean 3.8473, median 3.820"; whole-run means "E11-05 3.8665, E11-06 4.5834, E12 EVAL30 rev 1 5.0993".
   - The 4.55 is not reproduced by LV's parse of the E11-05 EVAL30 log; it is close to E11-06's whole-run mean. At E11-05's ~3.85-3.9 the anecdote is not excluded. LV: "The 'about 4.3' figure is representative only of E11-06 at 11-12 running."

4. **Step time for the same E11-03 DFlash2 fp16 arm: ~137 ms vs 115.5 ms.**
   - **DFW** Q2 Inferences: "DFlash W8 403.7 tok/s at L = 4.62, i.e. about 137 ms per step" (derived from total tok/s), and the a/b/d step model is calibrated on it.
   - **LV**: "The '115.5 ms' figure … is the raw median of E11-03 run D (INT4 + DFlash2, SSM fp16) at 12 running: n=67, all records in the 0-10k bin."
   - A2 note: 12 × 4.62 / 0.1155 s = 480 tok/s, so the 403.7 tok/s total includes non-decode time. The same applies to AR (DFW 41.5 ms vs LV's A 37.9 ms at 0-10k).

5. **All-reduce count per step: 128 / ~130 / 140.**
   - GPT notes 01/03/04: 64 × 2 = 128.
   - **TP2**: "about 130 all-reduces × 0.3-0.4 ms over SHM".
   - **LV** (source count): "**140 all-reduces of 983,040 B each per step**, plus about 11.9 MB per rank of logits all-gather" (129 + 1 AG per verify; 11 + 2 AG per draft step; 1-2 broadcasts). "The notes' '64 × 2 = 128' is a close underestimate."

6. **TP2 drafter weight per GPU.**
   - **LV**: "Drafter: about 2 × (1.49 − about 0.52 replicated) + 0.52 ≈ 2.5 GB … Total: about 11.7 GB, not 12.22 GB."
   - **TP2**: "DFlash2 2.19 GiB per GPU (fitted: replicated 0.79 + sharded 2.79 GiB …)"; "about 11.4 GiB target + drafter".
   - GPT notes: 12.22 GB (LV: "PARTLY TRUE … upper-bound style").

7. **Drafter KV bytes per token at TP2 (units).**
   - **LV** (memory Inferences): "KV per token doubles to 16,384 B (target) + 2,560 B (drafter)". In the Note 04 table: "8,192 → 16,384 B target + 1,280 → 2,560 B drafter". The 1,280 B figure is the drafter's K only at TP4 ("5 layers × (8 KV heads / 4 = 2) × 128 × 1 B = 1,280 B → 1.03 GiB").
   - **TP2**: TP4 total 10,752 B/token/GPU (= 8,192 + 2,560 drafter K+V); TP2 21,504 (= 16,384 + 5,120).
   - **Correction:** at TP2 the drafter's K+V is 5,120 B; LV's 2,560 B is K only.

8. **DFlash2 drafter KV-head count.**
   - **TP2**: "If that is 8 total heads split 4 ways, TP2 has 4 heads per GPU [E]"; Gap: "DFlash2's real KV-head count are unverified".
   - **LV**: [CFG-D] "5 layers, `num_key_value_heads 8`, `head_dim 128`"; log "`num_kv_heads=2`" at TP4. Confirmed.

9. **Cold-start durations.**
   - **TP2** (Other costs, from OURS): "ready in about 520-680 s; target-verify CUDA graph about 372-385 s; weight load 13-88 s"; "Staggered total ≈ 8-9 min, vs about 9.5 min for TP4 W1".
   - **LV**: server_args → "fired up" = 480 / 472 / 485 s for E11-05 / E11-06 / E12, and 1,149 s for G4B.
   - **SLE**: G4B kernel log "ready in 1197s".
   - The end points differ (kernel start vs server_args; Uvicorn vs "fired up"). LV notes GPT's 7:39 / 7:30 vs 19:09 "use **different end points**". The builder comment says "567 s in E11-04".

10. **Root cause of 11-of-12.**
    - **AET**: "Cause not established" (§1); "if it is caused by per-request Mamba slot reservation (4 slots per request …) rather than KV, one more server slot or a larger Mamba pool may resolve it — unverified" (§5); rank 5 "Cause unknown".
    - **NEV** Q1: PR #44401: "Under chunked prefill this capped the running batch at `max_running_requests - 1`, and when the final chunk's pass set `batch_is_full`, nothing cleared it…"; the local 0.5.21 code "ha[s] exactly the pre-patch shape". **TP2**: "the 11-of-12 bug (a chunked prefill counted twice) applies per replica".
    - **Correction:** NEV identifies the mechanism (scheduler double count), not Mamba slots.

11. **Are the HiCache host-tier fixes in 0.5.21?**
    - **AET** §2 Gap: "Whether 0.5.21 includes the Oct 4-12 host-tier fixes (#39862, #43596, #40223) is not established".
    - **NEV** Q1.8: #40223 and #43596 are "MERGED-in-v0.5.22"; v0.5.21 was released 2026-10-02.
    - A2 note: #39862 (Oct 4) is not listed in NEV, but its date is after the 0.5.21 release.
    - Also AET: "`buffer_only` host mode rejects MAMBA trees" in 0.5.21; NEV: v0.5.22 #36345 "Support Mamba in buffer-only host memory mode". These are consistent (a version change).

12. **#37475 fix status.**
    - **DFW** Q4: "Open, fix PR #37476 open."
    - **NEV** Q1.2: "OPEN, no comments, no linked fix."

13. **#28180 penalties.**
    - **DFW**: "Fixes: PRs #28200 / #28535 … The release that contains them is not identified."
    - **NEV**: "CLOSED on 2026-08-17. This predates 0.5.21, so 0.5.21 already has it. Follow-ups to apply penalties properly under DFLASH are still OPEN: #41738 … #43207."

14. **Message size where PCIe-IPC shines vs our message size.**
    - **NEV** Q1 Inference: "the per-layer decode all-reduces are small (12-400 KiB), which is the regime where the docstring reports 34-39x over NCCL on SM120".
    - **LV**: at 12 × W8 every AR is 983,040 B (96 × 5,120 × 2 B ≈ 960 KiB). The docstring's published sizes stop at 384 KiB.
    - A2 note: our production message is ~2.5× larger than the largest published point.

15. **NCCL bandwidth figure.**
    - Notes and **TP2**: "≈ 3.06 GB/s algbw ≈ 4.6 GB/s busbw".
    - **LV**: "algbw = 67,108,864 B / 20.89 ms = **3.21 GB/s** … busbw … 4.82 GB/s. The notes' '3.06-3.1 GB/s' corresponds to 64e6 B / 20.9 ms (decimal MB). It is a rounding difference, not an error."

16. **HyperQwen #254 numbers quoted differently.**
    - **NEV** Q4: "decode … 2 to 6% faster at C1 and 10 to 16% faster at C4".
    - **TP2**: "+6.1 % at c1, +10.1 % at c2, +13.3 % at c4, +2.0 % at c8 (greedy)". TP2 adds that the gain falls to +2.0 % at c8.

17. **Width-change throughput range quoted twice in DFW.**
    - Q2: "W=4 ranges from about +10 % (geometric) to about −10 % … W=6 from about +7 % to about −4 %".
    - Q6: "The model predicts W4 between −10 % and +12 % and W6 between −4 % and +7 %".

18. **Decode-only rate range in AET.**
    - §6 Takeaway: "Measured decode-only rates are 353-376 tok/s at our running levels".
    - §1 table: decode-only rates 376 / 377 / **302** / **335** / 353 (EVAL6R5 and LOW are 6-stream runs).

19. **Per-step weight-read time.**
    - **TP2**: "weights about 22 ms" (TP4, ~6.1 GiB of target + drafter).
    - **LV** (64 MB section): "a read of about 4 GB of weights per GPU (about 13 ms at 300 GB/s)" for target-only A at 1 running.
    - These are different assumptions (target-only vs target + drafter), not a numeric error.

20. **GPT-note claims corrected by LV** (one side only is in this folder):
    - "115.5 ms may be a context-normalized regression": **FALSE** (a raw short-context median).
    - "At width 4, more than 4 output tokens per step are possible": **FALSE** (max = W).
    - "Our ARC traces ~4.3": PARTLY TRUE.
    - "E12 137.9 ms": filter-dependent.
    - "TP2 restriction does not apply; the optimized path is possible with P2P": TRUE for V1, FALSE for V2.
    - "LayerCommunicator / fused / deferred paths": "This build has no LayerCommunicator in the Qwen3.5 path"; deferred, not fused.
    - The main reason a width change "is not a free optimization" is the non-causal intra-block attention, not the conv.

---

## 5. Open questions the documents say only a GPU run (or a Kaggle session) can answer

**AET:**
1. The exact divergence point of the re-rendered history (template trim or re-tokenization vs a missing decode checkpoint). Needs `return_output_ids` on one request, then the next request's `input_ids` (§1, §5 Gaps).
2. The cause of zero/partial hits at low KV (Mamba checkpoint LRU eviction vs trims vs #22935). Rev 6e G4-15 / H7-13 telemetry is designed to separate them (§1 Gaps).
3. The root cause of "11 running + queue 1" (AET says unknown; NEV attributes it to #44401). A run must confirm 12/12 after the fix.
4. Rev 6e / 6g behaviour on GPU (KV gate, no probe); none has run (§6 Gaps).
5. Whether the clean 12-running rate is sustained (only 29 G4B lines) (§6 Gaps).
6. How often the end-of-prefill checkpoint is evicted (no metric in 0.5.21) (§3 Gaps).
7. Streaming sessions or `--enable-session-radix-cache` on a hybrid GDN model with DFlash (§4 Gaps).
8. Any measured HiCache gain for a Qwen3.8-27B hybrid agent, and PCIe host↔GPU load-back bandwidth on Kaggle L4 (§2 Gaps).

**DFW:**
9. Acceptance (and the per-position survival curve) at W4 / W6 for this b8-trained checkpoint on our ARC prompts (Q1, Q2 Gaps).
10. The throughput sign of W4 / W6 at a real 12 running. It depends on how the W4A16 Marlin verify GEMM cost scales between 48 and 96 tokens on L4 (Q6 Gaps).
11. Whether #36548-type answer-level corruption occurs on our stack, and whether W4 reduces it (Q4 Gaps; Q5 step 3).
12. The target-only batch-invariance control, to judge the 15.5-token in-batch divergence (Q5 step 5).
13. Whether `--enable-deterministic-inference` works with DFLASH + Mamba on 0.5.21 (Q5 Gaps).

**LV:**
14. NCCL transport (P2P vs SHM), the 8 KB-1 MB latency curve, a TP2-pair number and an in-graph pynccl number (NCCL section Gaps).
15. The per-kernel split of the ~115 ms step (GEMM vs GDN vs NCCL vs drafter); needs nsys or the torch profiler (step-time and summary Gaps).
16. P2P availability and FlashInfer PCIe-IPC availability on Kaggle L4 (summary Gaps).
17. Exact TP2 memory: whether the embedding and lm_head shard as assumed, and the TP2 graph and activation sizes (memory Gaps).
18. Why the first capture bucket takes ~362 s (Triton autotune, inductor or GDN kernels), and whether the compile cache carries over to a width-4 or TP2 arm (startup Gaps).
19. PCIe link width and generation (`nvidia-smi -q` never logged) (Note 03 Gaps).

**NEV:**
20. Kaggle 4×L4 CUDA IPC / P2P support (needed by the PCIe-IPC kernel) (Q1 Gaps).
21. The SGLang PCIe-IPC backend on Ada or at TP4 (all published numbers are 8× SM120) (Q1 Gaps).
22. Silent SM89 performance changes 0.5.21 → 0.5.22 in GDN/Marlin (Q1 Gaps).
23. vLLM 0.32 + DFlash2 on SM89 / L4; whether #55601 cures #58894; vLLM PCIe-IPC at TP4 on SM89 (Q2 Gaps).
24. Whether pip on Kaggle accepts a retagged SGLang nightly wheel, and whether its Rust `.so` files load on glibc 2.35 (Q3 Gaps).
25. Any engine-version benchmark on L4 / sm_89 at TP4 for Qwen3.8-27B (Q4 Gaps).

**SLE:**
26. Which `/kaggle/input` subtrees are GCSFUSE vs NFS (detect from `/proc/mounts`), and the gcsfuse version and flags (§1 Gaps).
27. Scaling of read concurrency above 16 threads (§1 Gaps).
28. `df -h /dev/shm`, `/tmp` write MB/s, `nproc`, and whether the hidden-rerun pool has the same disk and `/dev/shm` (§2 Gaps).
29. Whether INT4 + DFLASH2 loads with `--load-format presharded` or `fastsafetensors`, and whether the GCSFUSE page cache is shared across the 4 rank processes (§3 Gaps).
30. Which component takes the 361 s first capture; jit-cache coverage of our flags; tarball size; cache validity in the hidden-rerun pool (§4 Gaps).
31. Whether `/proc/<pid>/io` `rchar` counts FUSE reads, and whether Kaggle cancels idle sessions (§8 Gaps).
32. The latest image's Python version, for a pinned-vs-latest test (§5).

**TP2:**
33. One TP2 replica boot to read "Load weight end" and "KV Cache is allocated" ("would settle both in under 10 minutes"), plus TP2 CUDA-graph memory (memory Gaps).
34. `routed_dp_rank` with the hybrid Mamba radix cache in `--dp-size` mode (implementation Gaps).
35. Kaggle `nvidia-smi topo -m` / `-p2p r` output and `NCCL_DEBUG=INFO` transport lines (custom AR Gaps).
36. Two simultaneous SGLang JIT captures on Kaggle (Other costs Gaps).
37. 2×TP2 vs TP4 at ≥32k context with ≥8 streams and an 8-token verify, on L4 (Measured comparisons Gaps).
38. Published sm_89 results for PCIe-IPC or forced custom AR (Alternatives Gaps).

**TEV:**
39. ARC accuracy and tokens of any token-efficient tune on our harness (paired EVAL vs INT4 base, same 60k floor and drafter) (§1, §6).
40. Same-condition DFlash2 / MTP acceptance (base vs tune) on L4, at TP4, over 60k traces (§4 Gaps).
41. Whether `--enable-strict-thinking` works with DFLASH in 0.5.21, and whether a logit-processor cap conflicts with speculative decoding (temperature-0 A/B) (§3 Gaps).

---

## 6. Check against the 2026-10-23 measurements

The brief provided these measurements; they were taken on Kaggle on 2026-10-23. Each item below is labelled CONFIRMED, REFUTED, PARTLY or STILL OPEN. All arithmetic here is an A2 note.

### 6.1 Topology: all 4 GPU pairs are PHB

- **STILL OPEN → answered:** LV, NEV and TP2 list the topology as never captured ("Kaggle 4×L4 `nvidia-smi topo -m` … never been captured"). It is now known: PHB for every pair.
- **CONFIRMED:** this is the same class as the analogues the docs cited, the Kaggle 2×T4 VM ("`PHB`, `intraNodeP2pSupport 0`, NCCL via SHM") and HyperQwen #254 (4× 5060 Ti "all PHB").
- **A2 note:** since every pair is PHB, no GPU pair is topologically better. Both TP2 pairs cross the host bridge just as TP4 does. LV wanted "both TP2 pairs" measured; topology no longer distinguishes them, but a TP2-pair all-reduce timing is still unmeasured.

### 6.2 NCCL: 983,040-byte all-reduce in a CUDA graph = 345 µs per call; ~56 ms of communication per decode step; P2P/IPC 5.5× slower (1,887 µs)

- **CONFIRMED, message size:** LV's source count gives a 983,040 B all-reduce at 12 × W8 (96 × 5,120 × 2), the size that was measured.
- **CONFIRMED (upper end), communication time:**
  - LV's naive model: 140 × 983,040 B / 3.06 GB/s = **45.0 ms**;
  - TP2's estimate: "All-reduce about 35-55 ms: about 130 all-reduces × 0.3-0.4 ms over SHM";
  - measured: 345 µs per call (inside the 0.3-0.4 ms band); ~56 ms per step, at or just above the top of TP2's range;
  - A2 note: 140 × 345 µs = 48.3 ms, so ~7.7 ms of the 56 ms would be other collectives (the 11.9 MB-in / 47.7 MB-out logits all-gather, the 2 small all-gathers and the broadcasts LV listed). That inference is unverified.
- **CONFIRMED, bandwidth-bound regime:**
  - A2 note: 983,040 B / 345 µs ≈ 2.85 GB/s algbw (≈ 4.27 GB/s busbw), close to the 64 MiB eager measurement (3.21 GB/s algbw, 4.82 busbw);
  - so at ~1 MB the collective is bandwidth-bound; it is not in the latency regime;
  - GPT's warning not to extrapolate from 64 MB was prudent, but the extrapolation would have been roughly right at 1 MB;
  - LV's ≤ ~70 µs bound for 10 KB messages is not contradicted (a different regime).
- **CONFIRMED (partly), share of the step:** 56 ms of a 115-133 ms step is ~42-49 %. This fits the TP2 decomposition (≈ 128 ms = weights ~22 + KV ~17 + AR 35-55 + residual 35-50) at its AR upper end; the implied residual is ~37-38 ms (A2 note).
- **PARTLY, default transport:** LV said the transport was "UNVERIFIABLE"; TP2 said the 64 MB rate "matches SHM, not P2P". The measured default path (345 µs) is much faster than the forced P2P/IPC path (1,887 µs), so the default is not P2P/IPC. The brief does not name the default transport; SHM is the docs' expectation, not confirmed by name.
- **REFUTED, P2P lever at TP4:**
  - the TP2 doc's top-ranked lever was "If P2P works, set `NCCL_P2P_LEVEL=SYS`/`PHB` at TP4 (no memory cost) and expect about +5-15 %", based on HyperQwen #254 (+6.1 / +10.1 / +13.3 / +2.0 %);
  - on Kaggle L4, NCCL P2P/IPC is 1,887/345 ≈ **5.5× slower**;
  - the consumer-card analogue does not transfer.
- **AT RISK (not directly measured), other IPC-based levers.** These depend on the same CUDA IPC peer path:
  - FlashInfer PCIe-IPC all-reduce ("needs P2P as well, because it relies on CUDA IPC peer mapping like custom AR [E]");
  - patched custom AR at TP4 (club-3090 +14-15 %);
  - legacy custom AR V1 at TP2 ("+10-15 % per replica decode [E]" if P2P works).
  - P2P/IPC evidently *functions* (it was timed), so the docs' fear that it might "raise … or at worst hang" is less likely. The bigger risk is now that IPC-based paths are slower than NCCL's default.
  - The FlashInfer kernel uses its own protocol (plain PTX loads/stores), so its speed over this path remains unmeasured.
- **PARTLY, the 2×TP2 "custom AR at TP2" scenario:**
  - TP2's "with working P2P custom AR at TP2 (comm about 3-5 ms), about 1.15-1.2×" assumed fast P2P; the 1,887 µs P2P/IPC result argues against it;
  - the "if KV fit" 1.0-1.1× case used SHM comm (~18 ms for 2 ranks at 0.49 MB), which is still unmeasured on a Kaggle TP2 pair;
  - the decisive memory facts (~170k KV per replica, ~340k total vs 881k; only ~4-8 long streams) are untouched by these measurements and still bind.
- **Inference for the width question** (A2 note, unverified):
  - AR time appears bandwidth-bound at ~1 MB, so W4 (48 rows, 491,520 B per AR) would roughly halve the per-call payload;
  - this supports the communication part of DFW's "linear-in-W cost assumption";
  - the GEMM part (Marlin verify at 48 vs 96 tokens) and the acceptance at W4 remain unmeasured, so "the sign must be measured" still stands.

### 6.3 Decode step ~115-133 ms at 12 running

- **CONFIRMED:** this matches LV's context bins at B=12: E11-06 115.3 ms (0-10k) → 133.3 ms (50-60k), E12 115.4 ms at 0-10k, and the E11-03 D raw median of 115.5 ms.
- **CONFIRMED:** LV's conclusion that "Differences between runs are context (KV read) plus prefill interleaving, not engine regressions" stands. The step on 2026-10-23 is the same engine cost as on 10-17 to 10-21.
- **Bound on DFW's calibration:** DFW's ~137 ms is above the measured range, consistent with §4 item 4 (it includes non-decode time).

### 6.4 Startup: first launch ready 502 s (weights ~65 s from /dev/shm, ~410 s compile/graph capture); relaunch in the same session 71 s

- **CONFIRMED, relaunch:** 71 s matches TP2/OURS exactly ("W2 71 s, W3 76 s"), and SLE §4's finding that a same-session warm cache cuts the capture to 12-14 s.
- **CONFIRMED (approximately), first launch:**
  - 502 s is close to LV's normal-day "fired up" totals (472-485 s) and well below the 10-21 slow-storage start (1,149 s server_args → fired up; 1,197 s in the kernel log);
  - ~410 s compile and graph capture vs the documented target-verify + draft graphs of 382-395 s (first bucket ~362 s) is consistent within ~4-7 %;
  - SLE §8 expected "Phase 2 … ≈ 6.5 min cold"; 410 s ≈ 6.8 min. A2 note: whether the 410 s also includes the ~22 s pre-warm and the warm-up compiles is not stated in the brief.
- **REFUTED, load time from /dev/shm:**
  - SLE §3: "With the copy in `/dev/shm`, the default multithread mmap loader reads from RAM. Our pre-10-21 load times (12.8-13.5 s with a warm cache) are the expected load cost.";
  - measured ~65 s, about **5×** the expected cost;
  - SLE §2 also claimed "SGLang's mmap of a tmpfs file maps the tmpfs pages directly, with no second copy", which does not by itself explain the 65 s;
  - **OPEN:** why tmpfs loading is ~5× slower than the old warm-page-cache loads.
- **STILL OPEN, shipped JIT cache:** a shipped cache across sessions (SLE §4 lever, ~360 s saving) is untested. The 71 s relaunch only proves the in-session effect.

### 6.5 Storage: staging copy 518 MB/s; single-stream /kaggle/input read 34.6 MB/s

- **CONFIRMED, the lever:** range-parallel copying works. A2 note: 518 / 34.6 ≈ **15×** the single-stream rate, more than amn's 16-thread 10× (10.5 → 107 MB/s).
- **REFUTED (in our favour), SLE's time estimates:**
  - SLE §2 estimated "About 3.6 min at 107 MB/s … or about 37 min single-stream" (at 10.5 MB/s);
  - A2 note: at 518 MB/s, 23.3 GB takes ≈ **45 s**; single-stream at 34.6 MB/s ≈ **11.2 min**, not 37;
  - the 10-22 figure of 10.5 MB/s (amn) and our 10-21 ~28 MB/s are both below the 34.6 MB/s measured on 10-23.
- **PARTLY, the 30-min fixed launch timeout risk** (SLE §8, "the single biggest risk"): at 34.6 MB/s a single-stream load (~11 min) would fit inside 30 min. Storage speed varies by day (amn's 10-21 low of 0.026 GiB/s), so the risk is reduced but not removed. With staging, the load is no longer storage-bound.
- **STILL OPEN:** whether the 10-23 session's model mounts were GCSFUSE or NFS (the brief does not say), `/dev/shm` capacity, and hidden-rerun pool parity.

### 6.6 A1-4 greedy parity: serial A/A 10/12 identical (2 ties, margin 0.0); concurrency-12 A/A 2/12 identical; 0 large-margin flips

- **CONFIRMED, DFW Q5's premise:** "Exact token equality … cannot serve as a pass/fail test … AR itself is not batch-invariant" and "The gate should compare against A/A baselines, not against zero". A c12 A/A gives only 2/12 identical sequences, and E11-03's earlier "contam" was 0/4 identical with a 15.5-token prefix.
- **CONFIRMED, the near-tie classification:** under DFW's criteria ("A margin under about 0.1 nat counts as a numeric near-tie (normal). A divergence where T's margin is over 1 nat counts as a bug flag."):
  - the 2 serial divergences are exact ties (margin 0.0), i.e. normal;
  - 0 large-margin flips means no bug flag at the token level.
- **STILL OPEN:**
  - the answer-level #36548 probe (DFW Q5 step 3: 150 trials per arm at c12, McNemar);
  - the workload-equivalence step (step 4);
  - whether DFlash2-specific excess divergence exists. That needs D vs T compared against these A/A rates; A2 note: the brief does not say whether the A/A arm was target-only (which would be the step-5 batch-invariance control) or the DFlash2 production arm (which would be a D A/A baseline).
- **Not yet covered:** #38009's "diverges across container restarts" A/A protocol. The brief does not say whether the serial A/A spanned a restart.

### 6.7 Summary of the 2026-10-23 cross-check

| Doc assumption | Status after 2026-10-23 |
|---|---|
| AR payload 983,040 B × 140 per step; ~45-55 ms of comm (LV, TP2) | **Confirmed** (345 µs per call; ~56 ms per step) |
| Comm is ~1/3-1/2 of the ~115-133 ms step (TP2 decomposition) | **Confirmed** (~42-49 %) |
| Step 115-133 ms at 12 running, set by context (LV) | **Confirmed** |
| Default NCCL transport is SHM, not P2P (TP2) | **Consistent** (the default is faster than forced P2P/IPC); transport not named |
| `NCCL_P2P_LEVEL=PHB`/`SYS` at TP4 gives +5-15 % (TP2 rank 1) | **Refuted** (P2P/IPC 5.5× slower) |
| PCIe-IPC AR / patched custom AR / TP2 custom AR V1 give +10-15 % or more | **At risk** (same IPC peer path is slow; not measured directly) |
| 2×TP2 + DFlash2 is memory-bound to ~4-8 long streams (TP2) | **Unchanged** (not tested; memory facts unaffected) |
| Cold start ~7.5-8 min on a normal day, ~6 min of it graph capture (LV) | **Confirmed** (502 s; ~410 s compile) |
| Warm relaunch ~71-76 s (TP2/OURS) | **Confirmed** (71 s) |
| Loading from `/dev/shm` costs ~13 s (SLE §3) | **Refuted** (~65 s) |
| Range-parallel copy: ~3.6 min at 107 MB/s; single stream ~37 min (SLE §2) | **Lever confirmed; times refuted in our favour** (518 MB/s ≈ 45 s; single stream 34.6 MB/s ≈ 11 min) |
| Exact greedy equality unusable under concurrency; use A/A baselines and margins (DFW Q5) | **Confirmed** (c12 A/A 2/12; ties margin 0.0; 0 large flips) |
| #36548 answer-level risk on our stack | **Still open** |
| W4/W6 acceptance and throughput sign | **Still open**; the bandwidth-bound AR makes the comm part scale with W (inference) |
