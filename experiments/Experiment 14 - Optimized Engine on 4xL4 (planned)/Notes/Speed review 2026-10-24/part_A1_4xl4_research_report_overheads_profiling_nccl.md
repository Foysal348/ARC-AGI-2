# Part A1: fact extraction from the 4xL4 research report, the per-step overheads note, the profiling note and the NCCL/P2P note

Written 2026-10-24. Facts only. This file does not design an experiment.

## Sources read (every line)

All four files are in `D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/Research - 4xL4 engine bottleneck validation/`.

| Code | File | Lines | Language | Labels the file uses |
|---|---|---|---|---|
| **FR** | `00_FINAL_REPORT.md` | 476 | Bengali (translated here) | [M] measured, [D] derived from measured, [E] estimate/model/roofline, [C] vendor or author claim, [S] read in local SGLang 0.5.21 source, [U] GitHub issue or community user report |
| **PSO** | `per_step_overheads.md` | 417 | English | [code], [measured], [claimed], [computed], [inference] |
| **PROF** | `profiling_and_levers.md` | 516 | English | [M], [C], [S], [L] (our own Kaggle logs or notes), [I] (inference) |
| **NCCL** | `nccl_p2p_l4.md` | 307 | English | [measured], [claimed], [inference] |

The FR's reference links still point to the old folder `D:/ARC AGI 2/research_notes/৪xL4 ইঞ্জিন বটলনেক যাচাই/`. The FR also draws on six notes I was not asked to read (LV local_validation, DFW dflash_width_correctness, TP2 tp2_replicas, AGENT agent_effective_throughput, NEV newer_engine_versions, SLE startup_load_environment, TEV token_efficient_variants). Where a number below came to the FR from one of those notes, the FR's own citation is given in brackets.

### FR section codes (the FR's headings are in Bengali)

| Code | Original heading | English meaning |
|---|---|---|
| FR§0 | opening paragraph under "# PCIe যোগাযোগ আর ৯৬-সারির verify গতি আটকাচ্ছে" | Summary: PCIe communication and the 96-row verify are holding speed back |
| FR§R | "পড়ার নিয়ম: স্ট্যাটাস, সূত্র আর ভাষা" | Reading rules |
| FR§1.1 | "মাপা অংশ: step time আসলে কী, আর কনটেক্সট কতটা যোগ করে" | Measured part: what step time is, and what context adds |
| FR§1.2 | "প্রতি step-এ কতগুলো collective, আর কোন পথে" | How many collectives per step, and on which path |
| FR§1.3 | "Step-এর গাণিতিক বাজেট: ১০৮ ms কেন্দ্র…" (includes the bold paragraph "B=1 থেকে B=12-এ marginal খরচের ভাঙন") | Arithmetic step budget, centre 108 ms (and the B=1 to B=12 marginal-cost breakdown) |
| FR§1.4 | "Compute, weight precision আর তিনটি ভুল ধারণা" | Compute, weight precision and three misconceptions |
| FR§2.1 | "সময় কোথায় যায়: পাঁচটি রানের হিসাব" | Where the time goes: five Duck runs |
| FR§2.2 | "সবচেয়ে বড় অবশিষ্ট ক্ষতি: আগের reply আবার prefill হয়" | Largest remaining loss: the previous reply is prefilled again |
| FR§2.3 | "Prefix reuse এখানে বেশি দামি, আর HiCache এখন উত্তর নয়" | Prefix reuse is worth more here; HiCache is not the answer now (plus the paragraph on streaming sessions) |
| FR§3.1 | "সম্মিলিত হিসাব: কোথায় পৌঁছানো সম্ভব" | Combined estimate: what is reachable |
| FR§3.2 | "NCCL transport আর env knob: আগে topology, তারপর P2P" | NCCL transport and env knobs: topology first, then P2P |
| FR§3.3 | "PCIe-IPC all-reduce: 0.5.21-এ code আছে, কিন্তু আমাদের install-এ মৃত" | PCIe-IPC all-reduce: code exists in 0.5.21 but is dead in our install |
| FR§3.4 | "#44401 আর CUDA graph: সবচেয়ে কম ঝুঁকির engine fix" | #44401 and CUDA graphs: the lowest-risk engine fix |
| FR§3.5 | "DFlash2 width 6/4: বৈধ, কিন্তু ফল আগে থেকে বলা যায় না" | DFlash2 width 6/4: valid, but the result cannot be predicted |
| FR§3.6 | "Logits, GDN আর অন্য কম-আলোচিত লিভার" | Logits, GDN and other less-discussed levers |
| FR§3.7 | "2×TP2: প্রায় বাতিল, কারণ মেমরি" | 2×TP2: nearly rejected because of memory |
| FR§4.1 | "GCSFUSE-এ 669 s load, আর stock loader কেন সাহায্য করে না" (plus the "সমাধানের নকশা" design paragraph) | 669 s load on GCSFUSE, and why the stock loader does not help |
| FR§4.2 | "প্রথম target-verify graph-এ 6 মিনিট, আর cache পাঠানোর নিয়ম" | 6 minutes on the first target-verify graph, and the cache-shipping rule |
| FR§4.3 | "Stall-ভিত্তিক timeout…" | Stall-based timeout |
| FR§4.4 | "10-22-এর G4B ERROR…" | The 10-22 G4B ERROR |
| FR§5.1 | "SGLang 0.5.22, nightly আর vLLM 0.32…" | Newer engine versions |
| FR§5.2 | "Token-efficient variant…" | Token-efficient model variants |
| FR§a | "(ক) GPT-এর তিনটি নোটের দাবি-ধরে রায়" | (a) Claim-by-claim verdicts on GPT notes 01, 03, 04 |
| FR§b | "(খ) Claude-এর ভুল আর তাদের সংশোধন" | (b) Claude's mistakes and their corrections |
| FR§c | "(গ) নোটগুলোর মধ্যে বিরোধ: মীমাংসিত আর খোলা" | (c) Contradictions between notes, resolved and open |
| FR§d | "(ঘ) Experiment 14-এর অগ্রাধিকার তালিকা" (+ "রানের ক্রম আর quota বণ্টন") | (d) E14 priority list (+ run order and quota) |
| FR§u | "যা এখনো কেউ জানে না" | What nobody knows yet |
| FR§z | "উপসংহার" | Conclusion |

PSO, PROF and NCCL are cited as `PSO Q1` … `NCCL Q7`, using each file's own "Q" headings:
- PSO Q1 "Code paths in 0.5.21: target lm_head, logits materialization, sampling per position, TP all-gather"; Q2 "FR-Spec / hot-vocab"; Q3 "Published per-kernel timings: lm_head plus sampling…"; Q4 "DFlash2 drafter architecture and per-step compute"; Q5 "Arithmetic budget for one 12 x 8 step…".
- PROF Q1 "Profiling tools in SGLang 0.5.21, Nsight Systems…"; Q2 "A single bounded, fail-closed Kaggle run…"; Q3 "Engine-level levers (other than TP2 and DFlash width), ranked by evidence"; Q4 "Precision options: FP8 W8A8 … vs INT4 Marlin at M ≈ 96"; Q5 "What the 'library code' / 'tiny kernel issue' fixes might be…".
- NCCL Q1 "Topology…"; Q2 "Transport without P2P (SHM)…"; Q3 "Is our measurement (64 MB in 20.9 ms ≈ 3.1 GB/s algbw) consistent with SHM?"; Q4 "NCCL tuning for small/medium messages…"; Q5 "Alternatives to NCCL in SGLang 0.5.x / vLLM on sm_89 PCIe"; Q6 "Overlap during decode … communication share … TP4 vs TP2 vs DP"; Q7 "How to measure this on an offline Kaggle notebook".

Stack named in all four files: Qwen3.8-27B RedHat INT4 W4A16 (compressed-tensors, g128 symmetric) + BF16 DFlash2 drafter (width 8), SGLang 0.5.21, TP4 on 4x L4, fp8_e4m3 KV, fp16 SSM, NCCL 2.29.7, FlashInfer 0.6.18 wheel, torch 2.13. The FR calls this config "W1".

---

## 1. Measured on OUR Kaggle 4xL4 runs

Each number gives its configuration, its source, and the label the document gives it. "[D]" means a number derived by script from our logs.

### 1.1 Server configuration as logged

| Fact | Run / config | Source | Label |
|---|---|---|---|
| `attention_backend=flashinfer`, `linear_attn_backend=triton`, `kv_cache_dtype=fp8_e4m3`, `mamba_ssm_dtype=float16`, `mamba_radix_cache_strategy=extra_buffer_lazy`, `max_mamba_cache_size=56` | E11-06 server args (`sglang_server_try0.log`) | PROF header "Reference facts" | [L] |
| `max_running_requests=13`, `chunked_prefill_size=4096`, `disable_overlap_schedule=False`, `enable_mixed_chunk=False` | E11-06 | PROF header | [L] |
| `stream_interval=1`, `incremental_streaming_output=True`, `sampling_backend=flashinfer`, `enable_torch_compile=False`, `num_continuous_decode_steps=1`; 1 tokenizer and 1 detokenizer worker | E11-06 | PROF header; PROF Q3 "stream interval" | [L] |
| `speculative_algorithm=DFLASH`, `speculative_num_draft_tokens=8`, `speculative_attention_mode=prefill`, `speculative_draft_model_quantization=None` | E11-06 | PROF header | [L] |
| The notebooks do pass `--speculative-draft-model-quantization unquant` (E11-03/05 builder, G4B cell line 791); server_args records it as None | coordinator check | FR§c row 5 | resolved |
| `cuda_graph_config.decode = {backend: full, max_bs: 12, bs: [1,2,3,4,5,6,7,8,10,12]}`; prefill graph disabled; `tp_size=4`; NCCL 2.29.7 | E11-06 (W1) | PROF header; PSO Q5 "Our CUDA graphs" | [L]/[measured] |
| Five E12 Duck runs have the identical server config: TP4, INT4 + DFLASH 8, `max_running_requests` 13, chunk 4096, `schedule_policy lpm`, `page_size 1`, `extra_buffer_lazy`, `mamba_track_interval 256`, mamba cache 56, `enable_mixed_chunk False`, `prefill_decode_interval 0`, **KV pool 866,744 tokens** | E12 EVAL30, EVAL6, EVAL6R5, EVAL12R5_LOW, G4B rev 6d | FR§2.1 | [M] (from [AGENT]) |
| W1 KV pool ~881k tokens; TP2 note's SGLang sizing formula predicted 886,801 vs **measured 881,336** (0.6 %) | W1 | PROF Q4 "Memory"; FR§3.7 | [L]/[E] |
| E11-03/E11-06/E12 server configs are identical (TP4, DFLASH 8, FP8 KV, INT4 target, BF16 drafter) | — | FR§1.1 | [D] |
| Both "CustomAllReduceV2 is disabled because it's not supported on more than two PCIe-only GPUs" and "CustomAllreduce is disabled…" appear on all four ranks | E11-06 log L33-40; E11-05 L34-41; E12 L33-40; G4B L34-41 | FR§1.2; FR§a N04 row 1 | [M] |
| "multimem all-gather disabled (no multicast for world_size=4)" | E11-06 L144-147 (PROF: line 145) | FR§1.2; PROF Q3 "NCCL transport" | [M/S] |
| "sglang is using nccl==2.29.7" | E11-06 L32 | PROF Q3 | [L] |
| GDN dispatcher: `decode=TritonGDNKernel, extend=TritonGDNKernel, verify=TritonGDNKernel packed_decode=True`; `Linear attention kernel backend: decode=triton, prefill=triton, verify=triton` | E11-06 L126, L130 | PROF Q3 "GDN Triton" | [L] |
| No `NCCL INFO` line in any log, so the transport (P2P or SHM) cannot be read from logs | all runs | FR§1.2; NCCL Q3 Gaps | [M] |
| `avail mem=21.72 GB` at start | log | FR§a N04 row 21 | [M] |
| Triton kernel `chunk_gated_delta_rule_fwd_kkt_solve_kernel` "took 1.74–1.94 s to compile after serving started": 24 lines (6 per rank), about 11 s total; a one-time stall after the first long prefill | E11-05 and E11-06 | PROF Q2 "Quota and machine" | [L] |

### 1.2 Decode step time

| Number | Config | Source | Label |
|---|---|---|---|
| Step at 12 running ≈ **115 ms (≤10k context) to 133–140 ms (50–80k)**, the same in three separate runs | W1, 12 × 8 verify | FR§0 | [D] |
| SGLang's `gen throughput` and `accept len` are per 40-forward window (`decode_log_interval` 40), so running × accept len ÷ gen tok/s = window wall time ÷ 40. The "inferred step time" is a window-average wall time that includes prefill, scheduler and idle time, not kernel time | source | FR§1.1 | [S] |
| Median step at 12 running: **E11-06 124.6 ms** (p10–p90 116.2–132.5, n=278); **E11-05 (EVAL30) 128.4 ms** (n=114) | W1 | FR§1.1; FR§a N01 row 16 | [D] |
| By context bin at ≤10k: **E11-03 D 115.5, E11-06 115.3, E12 EVAL30 115.4 ms** | W1 | FR§1.1 | [D] |
| **E11-06 at 50–60k: 133.3 ms**; **E12 at 60–80k: 140.2 ms**; about +18 ms from 0–10k to 50–60k | W1 | FR§1.1 | [D] |
| E11-03 D 115.5 ms is the raw median at 12 running (n=67, all 0–10k); E11-06's 0–10k bin is 115.3 | — | FR§a N03 row 10 | [D] |
| Windows that contain a prefill line infer much higher step times: E11-06 B=12 median 323 ms | W1 | FR§1.1 | [D] |
| E12 EVAL30 at 12 running: median 135.7–139.4 ms depending on filter (all 280 records 139.4; g>50 138.4; g>100 137.9, n=233; prefill-free 135.7); request median context ≈54.8k | E12 EVAL30 rev 1 | FR§1.1; FR§a N04 row 6 | [D] |
| E11-05 B=11: n=2,370, 124.2 ms, accept 3.82; B=12: n=114, 128.4 ms, accept 3.90 (2,371 records, one with g ≤ 50 dropped) | E11-05 | FR§a N03 row 8 | [D] |
| E11-06 B=11: n=158, 124.4 ms, accept 4.47; B=12: n=277, 124.6 ms, accept 4.34 (278 records, one with g = 20.73 dropped) | E11-06 | FR§a N03 row 9 | [D] |
| Median total context at 12 running: E11-05 449,175; E11-06 357,034 (n=277) | — | FR§a N03 row 11 | [D] |
| E11-03 D: step **39.9 ms at 1 running**, **115.5 ms at 12 running (0–10k)** → the 11 extra requests add 75.6 ms = 6.87 ms per request = 0.86 ms per verified token | W1 sweep | FR§1.3 (marginal paragraph); FR§a N01 row 13 | [D] |
| Marginal cost per extra request: no-spec (37.9 − 21.9)/11 = 1.45 ms (range 1.3–1.45); MTP (79.9 − 40.4)/9 = 4.39 ms (1.10 ms/token); DFlash2 (115.5 − 39.9)/11 = 6.87 ms (0.85–0.86 ms/token) | E11-03 sweep | FR§a N01 row 13 | [D] |
| Target-only step at 1 running = **21.9 ms**, so each 10 KB all-reduce is ≤ about 70 µs | E11-03 | FR§1.4 | [D] |
| E11-03 short sweep, W8: client total 403.7 tok/s ≈ 137 ms step (DFW); decode-line median 115.5 (LV); D at 12: server median 477.2 vs 404 client (PREV) | E11-03 | FR§c row 18 | [D] |
| Step at ~40k context, 12 running: ~125–131 ms with accept ~4.3–4.5, giving ~410–420 tok/s decode | W1 | FR§3.1 | [E] from [D] |
| Inputs used elsewhere: step 115–130 ms (NCCL header); step 115–133 ms (PROF header, PSO title) | — | NCCL header; PROF header | [L] |

### 1.3 Decode and whole-run throughput

| Number | Config | Source | Label |
|---|---|---|---|
| E11-06 decode-line tok/s at 12 running: **<10k 500.1; 10–20k 460.3; 20–40k 409.0; 40–60k 389.4** | W1 | FR§1.1 (citing PREV, TP2) | [M] |
| Whole run: **E11-05 328.9 tok/s** (4,253,591 tokens / 12,932.4 s); **E11-06 379.9 tok/s** (2,262,599 / 5,956.2 s) | W1 | FR§1.1 | [M] |
| Whole run 330–380 tok/s overall | W1 | FR§0 | [M] |
| T8 ≈ 447 tok/s (446.8; = 12 × 4.3 ÷ 0.1155 s) | GPT derived | FR§a N01 row 24 | [D] |
| ~420 tok/s ÷ 12 ≈ 4.3 tokens per step per request | — | PSO Q5 "GDN verify" | [computed] |

### 1.4 Speculative acceptance

| Number | Config | Source | Label |
|---|---|---|---|
| `accept len` includes the bonus token; `accept rate` = correct drafts ÷ (7 × Σbs), so accept len = 1 + 7 × accept rate. Over 5,017 records in three logs the maximum deviation is 0.0057 (rounding) | source + logs | FR§1.1 | [S]/[D] |
| Maximum accept len = width (8 at width 8; 4 at width 4) | source | FR§1.1; FR§3.5 | [S] |
| Whole-run mean acceptance: **E11-05 3.87 (3.8665), E11-06 4.58 (4.5834), E12 EVAL30 5.10** | — | FR§1.1; FR§a N04 rows 4–6 | [D] |
| Records: E11-05 2,509; E11-06 1,299; E12 EVAL30 rev 1 1,209 | — | FR§a N04 rows 4–6 | [D] |
| E11-06 at 8 running 4.46 (4.4628, n=408); at 12 running 4.37 (4.3708, n=278); E11-05 at 12 running 3.91 (3.9130, n=114) | — | FR§a N01 row 3 | [D] |
| Request-step-weighted mean equals the simple mean; token-weighted is 0.03–0.05 higher; time-weighted ±0.02 | — | FR§a N01 row 4 | [D] |
| 12 running: 4.34–4.37 | E11-06 | FR§3.5 | [D] |
| E11-05 decode-line median 3.8 (11 running 3.82) vs E11-05 per-request `spec_accept_length` median **4.55** (includes short phase-2 answers) | E11-05 | FR§3.5; FR§c row 13 | [D] |
| Accept len 4.1–4.6 with drafter quantization recorded as None | E11 runs | PROF Q5 "Silent correctness bugs" | [L] |
| Accept ≈ 4.3–4.5 at 8–12 running | — | PROF Q2 "Synthetic vs real prompts" | [L] |
| 4.3 accept = 3.3 correct drafts = ~47 % (47.1 %) of 7 positions; rejection 52.9 % | — | FR§1.4; FR§a N04 row 15 | [D] |

### 1.5 Concurrency and occupancy

| Number | Config | Source | Label |
|---|---|---|---|
| E11-05: max 12, but **2,358 decode lines at 11 running + 1 queued**, only **114 at 12** | E11-05 | FR§1.1; FR§3.4; PROF Q5 | [M] |
| E11-06 decode lines: **278 at 12; 158 at 11 with `#queue-req: 1`; 400 at 9; 408 at 8** (1,299 lines, all `cuda graph: True`) | E11-06 | PROF Q3 "real concurrency" | [L] |
| E11-06 at 11 running: mamba usage **33 of 56** slots, so the mamba pool was not the cap | E11-06 | PROF Q3; FR§3.4 | [L]/[M] |
| 11 vs 12 running at the same context: about −8–9 % (PROF: "about 8 %" in E11-05) | E11-05 | FR§1.1; PROF Q3 | [M] |
| "11 running + 1 queued": **50 % of EVAL30 time, 25 % of G4B time**. 11 vs 12 at the same context: **422 vs 462 tok/s (EVAL30), 397 vs 436 (G4B)**, about −9 % | E12 | FR§2.1 | [M/D] |
| G4B spent **27.7 % of its time at exactly 6 running**: the rev 6d probe stepped 12 → 10 → 8 → 6 at KV of only 0.13–0.31; rev 6e replaced it with 12 fixed workers and a KV gate of 0.95 | G4B rev 6d | FR§2.1 | [M] |
| EVAL30 at 12 running: mamba usage 36/56, leaving about 20 slots for cached checkpoints | E12 EVAL30 | FR§2.2 | [M/D] |
| G4B KV max only 0.44 after the rev 6 admission rules, so device KV was not binding | G4B | FR§2.3 | [M] |
| Engine fully idle (tool time): 17–38 s per run (before first and after last decode line), no long idle gaps | E12 | FR§2.1 | [M] |

### 1.6 Communication (collectives) measured on our box before 2026-10-23

| Number | Config | Source | Label |
|---|---|---|---|
| Preflight all-reduce, one size only: 16 Mi fp32 elements = 64 MiB, eager `torch.distributed`, 10 iterations: **20.89 ms (E11-03), 20.76–20.83 ms (E11-04)** → **algbw 3.21 GB/s, busbw 4.82 GB/s** (3.06 GB/s if 64 MB decimal) | E11-03 `results.json` L88-100; `build_e11_03_gpu_validation.py` `_ar_worker` | FR§1.2; FR§a N01 row 19, N03 rows 2 and 14 | [M/D] |
| Same measurement quoted as "20.9 ms, algbw about 3.1 GB/s" | E11-03 | NCCL header; NCCL Q3; PROF Q2 | [measured]/[L] |
| busbw 4.6–4.8 GB/s for n=4 from that number | — | NCCL Q3 inference; PSO Q1 | [inference] |

### 1.7 Compute and memory

| Number | Config | Source | Label |
|---|---|---|---|
| torch BF16 4096³ GEMM: **75.9 TFLOPS (E11-03), 78.8 TFLOPS (E11-04)** | preflight | FR§1.4; FR§a N01 row 15 | [M] |
| Dense-equivalent target work at 96-row verify ≈ 2 × 27B × 96 ÷ (4 × 0.115 s) ≈ **11.3 TFLOPS per GPU** | — | FR§1.4; FR§a N04 row 18 | [D] |
| Target weights **4.62 GB per GPU**; drafter **1.49 GB per GPU** | E11-03 D log | PSO Q1 "Our own measured memory"; FR§1.3 table | [measured] |
| Total per GPU ≈ 6.11 GB (4.62 + 1.49) | — | FR§a N03 row 21 | TRUE |
| Intermediate SSM buffer **1.83–1.97 GB per GPU** | E11 caps table | PSO Q1; FR§1.3 table | [measured] |
| Draft KV 0.92 GB (E11-03 D); **2.10 GB in W1 at 881k tokens** (≈ 2.56 KB/token) | — | PSO Q4 | [measured] |
| Logged K size 3.31 GB for 866,744 tokens (matches ~32.8 KB/token FP8 total, 8,192 B per GPU) | — | FR§a N03 row 12 | [M] |
| CUDA graph memory, 10 graphs: **target-verify 0.68 GB, draft 0.35 GB** (E11-06 L149-152, L167-170) | E11-06 | FR§3.4; FR§a N04 row 9 | [M] |
| "the current 10 graphs take about 0.88 GB per GPU" | — | PROF Q3 inference 1 | [L] (conflicts, see §4) |
| Draft-verify CUDA graph 0.38 GB | MEAS | PSO Q4 | [measured] (conflicts, see §4) |
| Target-verify graph capture 372.5 s | E11-06 | FR§a N04 row 9 | [M] |
| vLLM FP8 TP4: ~7.15 GiB weights per GPU, KV 1.35M tokens (vs SGLang INT4 + DFlash2 881k; different engines and slot settings) | earlier E-runs via engines_and_config §3 | PROF Q4 | [L] |
| vLLM 0.19 on our L4: all 5 block-FP8 GEMM shapes ran untuned default Triton configs ("Performance might be sub-optimal"); FP8 Marlin disabled on cc ≥ 8.9 unless forced | E12 FP8 kernel tuning README | PROF Q4 | [L] |
| FlashInfer + fp8 KV ran 12 h on L4 under vLLM 0.19 | project experience (delongyang analysis) | PROF Q3 "attention backend" | [L] |

### 1.8 Duck agent: effective throughput and prefill

| Number | Config | Source | Label |
|---|---|---|---|
| Duck effective rate **167–280 tok/s**; engine prefill only **~1,500 tok/s**, and while it runs it stops all decode | E12 | FR§0 | [M/D] |
| Scheduler runs a waiting prefill batch first ("Run prefill first if possible"), so one long 4,096-token-chunk re-prefill stops every other stream's decode | source `managers/scheduler.py` ~L3723-3753 | FR§2.1 | [S] |
| Chunked-burst prefill rate: **EVAL30 1,474 (1,375 at 48–80k), EVAL6 1,548, EVAL6R5 1,534, EVAL12R5_LOW 1,544, G4B 1,578 tok/s** | E12 | FR§2.1 | [M] |
| One uncached prompt token costs about **0.27×** a generated token | — | FR§2.1 | [D] |
| Re-prefill takes 13–20 % of the solve window in good runs, **55 % in EVAL30** | E12 | FR§0 | [M/D] |

Five-run table (FR§2.1, from [AGENT] `wall_time_breakdown_output.txt`, [M/D]):

| Run | Streams | Window (s) | Generated | Effective tok/s | Prefill stall, two methods (s) | Prefill % of window | Decode-only rate | Uncached prompt ÷ generated |
|---|---|---|---|---|---|---|---|---|
| EVAL30 | 12 | 14,046 | 2,357,584 | **167.8** | 7,735 / 7,617 | **55.1 %** | 376 | **4.07** |
| EVAL6 | 6 | 3,744 | 1,217,904 | 325.3 | 499 / 504 | 13.3 % | 377 | 0.64 |
| EVAL6R5 | 6 | 3,464 | 846,080 | 244.2 | 645 / 633 | 18.6 % | 302 | 1.15 |
| EVAL12R5_LOW | 6 | 6,570 | 1,743,889 | 265.4 | 1,342 / 1,317 | 20.4 % | 335 | 1.17 |
| G4B rev 6d | 12 | 2,050 | 572,035 | **279.0** | 406 / 425 | 19.8 % | 353 | 1.17 |

- The two prefill-stall methods (extra decode-interval time; logged prefill tokens ÷ burst rate) agree within 1–5 % (FR§2.1).
- EVAL30's earlier figure was 187 tok/s (2.36M ÷ 12.6 ks). The new window includes 679 s after the deadline, giving 167.8. Both are from the same run with different definitions. G4B was 277 under the old definition (FR§2.1; FR§b B1).
- EVAL30 cause: 30 time-sliced branches × 50–100k context did not fit the 866,744-token pool; decode-line KV p50/p90 0.77/0.95 [M] (FR§2.1).
- E12 EVAL30: 8.39 M re-prefill tokens; effective about 187 vs about 412–431 decode (PROF Q3 "chunked prefill", citing note 05) [L].

### 1.9 Prefix reuse (Duck)

| Number | Config | Source | Label |
|---|---|---|---|
| Cache hit stops exactly at the end of the previous prompt (`prefill_end` class): **EVAL6 84.5 %, EVAL6R5 91.3 %, LOW 72.5 %, G4B 75.6 %, EVAL30 59.1 %** | E12 | FR§2.2 | [M] |
| The hit usually stops a median 15–32 tokens before the previous prompt's end, almost always on a multiple of 64 (G4B 141 of 150 non-zero hits; LOW 413 of 414) → the 64-token-grid end-of-prefill Mamba checkpoint; **the previous reply (reasoning + tool call) is prefilled again** | E12 `reuse_offsets_output.txt` | FR§2.2 | [M] |
| Median reply-reuse fraction in all runs: −0.003 to −0.009 | E12 | FR§2.2 | [M] |
| When the reply is reused (`full_reply`), the hit is not 64-aligned, so the decode-end checkpoint works if the re-rendered history matches the generated tokens exactly | E12 | FR§2.2 | [M] |
| Token ids were not recorded (`return_input_ids`/`return_output_ids` False), so the cause is unproven | — | FR§2.2 | [M] |
| G4B uncached work 670k tokens: re-prefilled reply ~37 % (245k), zero or partial hit at low KV ~47 % (317k), first request ~5 % (34k), unavoidable tool-result delta ~11 % (~74k) | G4B | FR§2.2 | [D] |
| Zero/partial hits happened at KV p50 0.28; more misses after long previous replies (G4B: <2k reply 12/114; 2–10k reply 14/39); correlation only | G4B | FR§2.2 | [M] |
| Scenario chain for G4B: measured 279 → ~303 (probe fix) → ~325 (12/12 running) → ~357 (no reply re-prefill) → ~408 (no zero/partial misses); at Franzen-level 93.4 % reuse ~396 | G4B | FR§2.3 | [D] |
| Scenario chain for EVAL30: 168 → ~331 (no eviction re-prefill) → ~368 (no reply re-prefill) → ~410 (12 running at ≥60k) | EVAL30 | FR§2.3 | [D] |
| Ceiling = decode rate itself, about 400–460 tok/s at 11–12 running; the chain assumes all saved prefill time turns into decode | — | FR§2.3 | [D] |
| Hidden scale (240 tasks, ~11.46 h, 12 workers): 240 × 60k = 14.4M tokens needs ~370 tok/s effective. On G4B's profile, probe fix + reply fix (or probe fix + 12/12) reaches it | — | FR§2.3 | [D] |
| Our prefill/decode ratio ~1,500 vs ~400 (~0.27–0.3) vs Franzen 0.08 → prefix reuse is worth ~3–4× more on 4×L4 | — | FR§2.3 | [D] |

### 1.10 Startup and load

| Number | Config | Source | Label |
|---|---|---|---|
| INT4 load 12.8–58.7 s on 10-19/20; **669.26 s on 10-21 G4B**; the 18.6 GB shard took 657 s (≈28 MB/s); "ready in" **1,197 s** | E11/E12 | FR§4.1 | [M] (from [SLE], [LV]) |
| DFlash2 file loaded in 0.56 s on 10-21, i.e. from RAM, because the sorted-order `cat` prefetch had read it first | G4B | FR§4.1 | [M] |
| "ready in" **507–562 s** on 10-19/10-20; **1,197 s** on 10-21 (GCSFUSE). Weights read: INT4 19.47 GB + DFlash2 3.85 GB = 23.3 GB | — | PROF Q2 "Startup and storage" | [L] |
| First launch 601 s (prefill-graph JIT 447 s); later launches in the same session 136–166 s | E11-03 (via engines_and_config §1) | PROF Q2 | [L] |
| Cold start server_args → "fired up": **E11-05 8:00, E11-06 7:52, E12 8:05, G4B 19:09** (≈7.5–8 min healthy) | — | FR§4.2; FR§a N01 row 33 | [D] |
| GPT note 01's 7:39 (E11-05) and 7:30 (E11-06) use a different endpoint (Uvicorn) | — | FR§4.2; PROF Q2 | [L] |
| Target-verify CUDA graph capture **371–385 s**; almost all of it is the first batch size: tqdm `bs=12 … 1/10 [06:02<…, 362.28s/it]`, the other 9 sizes ~10 s | E11-06 L148 | FR§4.2 | [M] |
| With a warm cache in the same session the whole stage takes **11.94–13.67 s** | E11-04 W2/W3 | FR§4.2 | [M] |
| Current prefetch is a single-stream `cat file1 file2 file3 > /dev/null`; at 10.5 MB/s it takes ~37 min, longer than the 30-min first-launch timeout; on timeout `e11_next()` finds no error in the still-loading log → `E11Fail`, `ENGINE_READY = False`, fallback submission (LB ≈ 0) | — | FR§4.1 | [M/D]/[S] |

### 1.11 Platform and quota

| Number | Source | Label |
|---|---|---|
| 4×L4 bills at 2×; we measured a 2.0× burn ourselves | FR§4.4; FR§a N01 row 34 | [M/C] |
| 10-21 1,197 s start ate ~40 quota-minutes before the first token | FR§4.4 | [D] |
| 10-22 G4B rev 6g commit (esfersami50): pushed ~17:16 UTC, ran ~114 min, ended ERROR; `kernel_log.txt` exactly `[]`; no output files. In earlier runs the runner's `nbformat … MissingIDFieldWarning` appears 6.44–11.02 s after kernel start; our cell 1 writes `submission.json` within seconds | FR§4.4 | [M] |
| Quota 26.87 → 30.21/30 h (3.34 quota-h ≈ 100 min container at 2×); 3.13 h (~94 min) left before the run | FR§4.4 | [D] |
| E9 on 10-13 ran 8,879 s starting from 28.44/30 h | FR§4.4 | [M] |
| Quota left: **foysal 4.71 h, hosen42 8.63 h, esfersami50 used up (30.21/30)**; deadline 2026-11-02; about one submission per two days; 12–16 h turnaround | FR§d; PROF Q2 | [M]/[C] |

### 1.12 Scores

| Number | Source | Label |
|---|---|---|
| LB 49k vs 60k cap: 32.08 vs 34.58 (cutting the cap lowers the score) | FR§5.2 | [M] |
| LB 29.17 vs 34.58 (E14-G is meant to help explain this gap) | FR§d E14-G row | [M] |

---

## 2. Measured elsewhere (other hardware or other people)

The "Transferability" column states what the document itself says about applying the number to us.

### 2.1 Topology and P2P on GCP / Kaggle

| Number | Where | Source | Transferability (per doc) |
|---|---|---|---|
| G2/L4 interconnect "PCIe Gen 4", no NVLink; L4 24 GB GDDR6 300 GBps; 1/2/4/8 GPUs per VM | GCP doc | NCCL Q1 | vendor claim |
| g2-standard-48: 48 vCPU, 192 GB (custom 192–216 GB), 4× L4 (96 GB), 50 Gbps; "only available on the Cascade Lake platform"; nothing on NUMA, PCIe layout or P2P | GCP doc | NCCL Q1 | claim; Kaggle may differ |
| Spare Cores g2-standard-48: 48 vCPU on 24 physical cores, VM, L3 39 MiB, empty topology section; nvbandwidth v0.5 (driver 535.183.01, CUDA 12.6): GPU→host 13.2 GB/s, host→GPU 12.4 GB/s, host-device latency 921 ns; **"P2P supported: yes", P2P 10.1 GB/s, duplex 8.2, gather 9.4 GB/s, P2P latency 921 ns** | GCP g2-standard-48 | NCCL Q1, Q3 | Same machine type, but the P2P latency equals host-device latency exactly ("looks like a copy or labelling error, treat with caution"); Kaggle may use different images or passthrough |
| GKE 1.27 g2-standard-24 (2× L4), driver 535.161.07: `topo -m` GPU0↔GPU1 = PHB, CPU affinity 0-23, NUMA 0; `topo -p2p` r/w/p OK, n and a (atomics) NS | GKE | NCCL Q1; FR§1.2 | different VM size; supports "PHB" |
| Kaggle 2× T4 VM: PHB on a flat virtual PCI bus (00:04, 00:05); NCCL `intraNodeP2pSupport 0` → `via SHM/direct/direct` | Kaggle, T4 | NCCL Q1; FR§1.2 | Kaggle but T4, not L4 |
| NCCL 2.29.7 `ncclTopoCheckP2p` starts from `p2pLevel = PATH_PXB` ("By default don't use P2P across CPU Host Bridges and further apart"); only override PATH_SYS for x86 AMD hosts with ≤2 GPUs → **3+ GPUs on CPU root ports on any x86 host → SHM** | NCCL source via HyperQwen #254 | NCCL Q1 | applies directly to our NCCL version |
| KVM guest, 4 PHB GPUs (4× RTX 5060 Ti): P2P OK by driver, transport SHM by default; `P2P/CUMEM` only with `NCCL_P2P_LEVEL=PHB` or `SYS` (`PXB` stayed SHM) | HyperQwen #254 | NCCL Q1; FR§1.2 | similar virtualized PHB layout |
| NCCL maintainers: 2.23 "avoids P2P in more cases"; NCCL avoids P2P where it would perform poorly even when nominally supported | NVIDIA/nccl #814 | NCCL Q1 | claim |
| NVIDIA: "VMs require ACS to function"; VT-d/IOMMU can redirect "all PCI point-to-point traffic to the CPU root complex", causing slowdowns or hangs; virtual PCI topology through /sys can be suboptimal | NCCL troubleshooting doc | NCCL Q1 | claim; relevant to VMs |
| `topo -p2p w` OK for all 64 pairs, yet TP2 P2P/CUMEM hung under a translated IOMMU (4- and 8-rank SHM runs completed) | vLLM RFC #51513 | NCCL Q1, Q4; FR§3.2 | shows driver "OK" ≠ working |
| ACS redirect on a Gen4 switch: one-way P2P latency 1.79 µs vs 1.58 µs via root complex vs 1.25 µs with ACS cleared | V100 box (squibloads) | NCCL Q1 | other hardware |
| GCP built a software-defined PCIe P2P fabric only for G4 (RTX PRO 6000): "up to 2.2x" on collectives, up to 41 % lower ITL, up to 168 % higher vLLM TP throughput vs "standard non-P2P offerings"; says nothing about G2 | GCP G4 blog | NCCL Q1, Q6; PROF Q3 | vendor, G4 not L4 |

### 2.2 NCCL SHM / P2P latency and bandwidth

| Number | Where | Source | Transferability |
|---|---|---|---|
| SHM plateau ~2.4–4.6 GB/s busbw on PCIe Gen3/Gen4; ~12–25 µs per small all-reduce under CUDA graphs (~40 µs eager); ~200–500 µs for 1 MB | summary | NCCL Q2 takeaway | "No published nccl-tests numbers for L4" |
| Kaggle 2×T4 (Gen3 x16), SHM, NCCL 2.25.1: 8 B–1 KB 14–16 µs SHM vs 60 µs sockets; busBW Ring/Tree/Socket: 16 KB 0.83/0.40/0.13; **1 MB 3.56/3.39/1.57**; 64 MB 3.97/4.58/2.08; 512 MB 3.98/4.53/2.13 GB/s; LL α 12.4 µs β 2.11 GB/s (8–64 KB); Simple α 29.4 µs β 3.97 GB/s (≥512 KB); LL up to 64 KB, Simple from 512 KB; LL128 disabled on PCIe; 21 → 41 µs jump between 8 KB and 16 KB; first run of a session up to ~30 % slower at 64–128 KB | nccl-allreduce-bench | NCCL Q2, Q3, Q7; FR§1.2 | Kaggle 2-GPU T4 |
| 4× RTX 5060 Ti, Gen3 x8, KVM, all PHB, NCCL 2.29.7, CUDA graph, bf16 × 5120, median, default (SHM) vs `NCCL_P2P_LEVEL=SYS`: 1 token 10,240 B 43.88 vs 29.08 µs; 8 tok 81,920 B **64.09** vs 55.51; 16 tok 163,840 B 98.36 vs 85.16; 32 tok 327,680 B 157.34 vs 142.82; 64 tok 655,360 B **267.61** vs 248.50; 128 tok 1,310,720 B **498.09** vs 467.28; 512 tok 5,242,880 B 1,863.94 vs 1,794.32; 2048 tok 20,971,520 B 7,338.91 vs 7,136.45 µs | HyperQwen #254 | NCCL Q2; PSO Q5; FR§1.3 | Different box (Gen3 x8), but our busbw (4.8) is close to theirs (~4.3) (FR§1.3) |
| Interpolated 96-token (983,040 B) SHM all-reduce ≈ **380 µs** (FR: ≈383 µs) | from HyperQwen | NCCL Q3; FR§1.2, §1.3 | estimate |
| 4× 5060 Ti SHM point: 20.97 MB in 7.34 ms = 2.86 GB/s algbw / 4.3 busbw; ours is ~10 % better ("consistent with SHM on a better host") | HyperQwen | NCCL Q3 | inference |
| 6× RTX 4090 `NCCL_P2P_DISABLE=1`: 1 MB 515.2 µs (busbw 3.39); 2 MB 1,233 µs (2.83); 16 MB 10,892 µs (2.57); 64 MB 46,206 µs (algbw 1.45, busbw **2.42**); blamed on AMD CPUs "don't like having lots of flows" (bidirectional can cut >2×). p2pBandwidthLatencyTest: unidirectional P2P-disabled 20–22 GB/s, P2P writes 9.9–10.2 GB/s; latency 11–14 µs disabled vs ~1.2 µs P2P | NVIDIA/nccl #814 | NCCL Q2, Q3; FR§1.2 | AMD host, 6 GPUs |
| 8× A10, NCCL 2.18.1, transport not stated: 32 KB 36.8 µs; 64 KB 38.5; 128 KB 118.1; 256 KB 127.0; 512 KB 153.0; **1 MB 199.6 µs (algbw 5.25, busbw 9.19)** | NVIDIA/nccl #922 | NCCL Q2 | transport unknown |
| Dual RTX 5070 Ti, PCIe 4.0 x16, no P2P, SHM: 20.97 MB in 10,859 µs (1.9 GB/s) "under every NCCL_PROTO"; 22–25 µs per call in a CUDA graph, ~41 µs eager for 4096 bf16 (8 KB) | SGLang PR #39605 | NCCL Q2 | 2 GPUs |
| V100 box without `NCCL_P2P_LEVEL=SYS`: "NCCL staged cross-board traffic through host shared memory at 3.5 GB/s" | squibloads | NCCL Q2, Q3 | other hardware |
| AWS g5.48xlarge (8× A10G), all PHB, NUMA 0-1: 28 MB all-reduce ~2 GB/s on NCCL 2.18, ~1 GB/s on 2.20 (regression); `NCCL_TOPO_FILE` brought it to ~1.6 GB/s | NVIDIA/nccl #1298 | NCCL Q2 | other hardware; version sensitivity |
| 4× RTX 4090 with patched driver forcing P2P: ~25 GB/s busbw at 16 MB | vLLM #36481 (search summary) | NCCL Q2, Q3; FR§1.2 | "working P2P ring on Gen4" reference |
| "PCIe Gen4 vs Gen3 matters little for SHM, because the host-memory staging and CPU-side flows set the limit, not the link" (5070 Ti Gen4 x16 no faster than T4 Gen3) | — | NCCL Q2 inference; FR§b A1 | inference; supports transferring SHM numbers |
| Per-hop protocol latency ~6 µs Simple, 1 µs LL, 2 µs LL128; PCIe needs MB-scale messages before bandwidth dominates | Demystifying NCCL (arXiv 2507.04786) | PROF Q2 | claim |

### 2.3 Effects of NCCL knobs (other setups)

| Knob | Number | Original source | Cited in | Transferability |
|---|---|---|---|---|
| `NCCL_P2P_LEVEL` forcing P2P | 4-GPU PHB KVM (Gen3 x8): AR −33.9 % at 10 KB, −13.3 % at 80 KB, −9.7 % at 328 KB, **−6.1 % at 1.31 MB**; decode +2 % to +16 % (C4 greedy +10–16 %; C8 overlapping, not established); FR lists decode c1 +6.1 %, c2 +10.1 %, c4 +13.3 %, c8 +2.0 %; step saving 0.74 ms at 8 tokens and 3.93 ms at 64 tokens | HyperQwen #254 | NCCL Q4, Q6; FR§3.2 | the small 1.31 MB gain may be due to that box's Gen3 x8 links [E] |
| same | 4× RTX PRO 6000: +3.6 % at c=1 (67.3 → 69.7 tok/s) | rtx6kpro | NCCL Q4; FR§3.2 | other hardware |
| same | 8× A40 PCIe: P2P slightly slower (~23 GB/s busbw at 32 MB) | nccl-tests #74 (search summary) | NCCL Q4 | old issue |
| same | Dual 5090: NCCL P2P toggle +2–5 %, sometimes negative | vLLM forum | PROF Q3 | other hardware |
| `NCCL_SHM_USE_CUDA_MEMCPY=1` | No-P2P dual 5070 Ti: SHM BW 1.9 → **12.5 GB/s**, but 10 KB decode AR 39 → 81 µs | SGLang PR #39605 | NCCL Q4; FR§3.2 | 2 GPUs |
| same | 8× A10 NCCL 2.18.1: 128 KB–1 MB ~2× slower (**1 MB 199.6 → 414.4 µs**); 32–64 KB unchanged; maintainer: "CE engine imposes a higher launch overhead", "designed for medium and large message sizes (e.g. tens of MB to GB)" | NVIDIA/nccl #922 | NCCL Q4; FR§3.2 | other hardware |
| same | maintainer suggested "roughly 2x on G5/G6" (AWS A10G/L4 families), no outcome reported | NVIDIA/nccl #1298 | NCCL Q4 | claim |
| same | hang with PyTorch reported (old) | NVIDIA/nccl #803 (search summary) | NCCL Q4 | risk |
| `NCCL_PROTO` | LL128 only on supported platforms, forcing elsewhere "can corrupt data", users "discouraged from setting it"; on SHM 1.9 GB/s "under every NCCL_PROTO" | NCCL doc; PR #39605 | NCCL Q4 | — |
| `NCCL_ALGO` | SHM T4: Tree +14.5 % at large sizes, slower at 8 KB–1 MB (16 KB 0.40 vs 0.83 GB/s; 1 MB 3.39 vs 3.56), n=2 | nccl-allreduce-bench | NCCL Q4 | Kaggle T4 |
| `NCCL_NTHREADS` | default 512; 512 vs 256 +1.8 % at the T4 SHM plateau | NCCL doc + bench | NCCL Q4 | — |
| `NCCL_MIN_CTAS`/`MAX_CTAS` (replace MIN/MAX_NCHANNELS since 2.17, up to 64) | g5.24xlarge (4× A10G): 1/1 "recovered" to 2.18.3 levels (approximate, from nvtop); maintainer suggested `NCCL_MAX_NCHANNELS=2`; RTX PRO 6000 `NCCL_MIN_NCHANNELS=8` "significant bandwidth improvement" (no numbers); "Excessive CTAs can hurt small-message performance" | #1298; rtx6kpro; NCCL doc | NCCL Q4; FR§3.2 | weak |
| `NCCL_BUFFSIZE` | default 4 MiB per pair; no measured effect found | NCCL doc | NCCL Q4 | — |
| `NCCL_CUMEM_ENABLE` / `NCCL_CUMEM_HOST_ENABLE` | CUMEM auto-on since 2.19; CUMEM_HOST default 1 since 2.24 (driver ≥12.6 / runtime ≥12.2); vLLM advice: try `NCCL_CUMEM_HOST_ENABLE=0` first for hangs, `NCCL_P2P_DISABLE=1` last | NCCL doc; st-hakky blog | NCCL Q4 | SGLang itself sets `NCCL_CUMEM_ENABLE=0` and `NCCL_NVLS_ENABLE=0` (FR§1.2 [S]) |
| `NCCL_SHM_DISABLE=1` | SHM → sockets: α ~4× (14–16 → 60 µs), BW −47 % | nccl-allreduce-bench | NCCL Q4; FR§3.2 | negative |
| `NCCL_TOPO_FILE` / `NCCL_GRAPH_FILE` | g5.48xlarge DDP 7 → 4.5 h (torch 2.0); graph XML +11 % throughput on 2-NUMA Genoa | #1298; rtx6kpro | NCCL Q4 | other setups |
| `NCCL_GRAPH_REGISTER` | matters only with CollNet and full P2P; `NCCL_RUNTIME_CONNECT` default 1 (lazy connect) | NCCL doc | NCCL Q4 | — |
| NCCL version | 2.20.5 halved AR bandwidth vs 2.18.5 on 8× A10G PHB | #1298 | NCCL Q4; FR§3.2 | → benchmark with torch's own NCCL 2.29.7 |
| NCCL docs | forcing env "will prevent NCCL from selecting the best setting automatically", should not stay in production | NCCL env docs | NCCL Q4; PROF Q2; FR§3.2 | — |

### 2.4 Alternative all-reduce kernels

| Number | Where | Source | Transferability |
|---|---|---|---|
| FlashInfer PCIe-IPC docstring (8× SM120, 2 NUMA): 12 KiB 5.3 vs 205.1 µs; 96 KiB 15.3 vs 584.9 µs; 384 KiB 34.6 vs 1,169.3 µs; TP8 at 8k context TPOT 21.14 → 13.62 ms | SGLang module docstring | FR§3.3 | [M, SGLang author]; big multipliers are for 12–384 KiB messages |
| PR #34528: TP8 TPOT −36 % (8k bs1); TP4 DeepSeek-V4-Flash TPOT 20.55 → 18.49 ms (−10 %) bs1, 31.74 → 30.24 ms (−5 %) bs4; TP8 128k bs4 unchanged; tested on SM120 RTX PRO 6000 "switch-free" hosts; gain shrinks with batch | SGLang PR #34528 | NCCL Q5, Q6; FR§3.3 | SM120 only |
| lukealonso PCIe one-shot (4× RTX PRO 6000 TP4, P2P): 8 KB 6.4 vs 13.9 µs NCCL; 128 KB 17.6 vs 24.7; 512 KB 51.0 vs 71.7; **1 MB 95.6 vs 85.2 (NCCL wins)**; decode +5–11 % (Qwen3.5-397B TP4); "Without the config, custom AR runs about 15× slower than NCCL" | rtx6kpro | NCCL Q5; FR§1.2 | P2P required; one-shot gives little at 0.94 MiB |
| V100 PCIe push kernel (1Cat-vLLM): 5 KiB 5.1 vs 16.0 µs; 80 KiB 30.7 vs 33.5 µs; 4-GPU decode 75.9 → 91.5 tok/s; needs `NCCL_P2P_LEVEL=SYS`, `VLLM_CAR_P2P_MESH=1` | squibloads | NCCL Q5 | P2P required |
| 8× 3090 (NVLink pairs + PCIe): forced custom AR "no faster on any model" (87.7 vs 91.75 t/s) | loktar00 | NCCL Q5 | — |
| Forced custom AR at TP4 on 3090 PCIe: **+14.8 % decode**, greedy divergence on 4/5 prompts ("not eval-clean"); TP2 custom AR + verified P2P beat no-P2P by ~+15.7 % | club-3090 #773 | PROF Q3; FR§3.3 | correctness failure |
| 2× 3090 PCIe with P2P: V1 custom AR on vs off +0.6 % / +1.8 % (boot noise) | club-3090 PR #1470 | NCCL Q5 | TP2 |
| Torch symm-mem two-shot PR (closed unmerged 2026-10-07), H200 NVLink: 8 KB two-shot 6.3 µs (ws2) / 7.8–14.6 µs (ws4) vs NCCL 23–27 µs under graph; "all-reduce is about 12% of decode time" there | SGLang PR #30364 | NCCL Q5, Q6 | NVLink |
| Host-staged AR (PR #39605): 20.97 MB 1,494 µs vs NCCL SHM 10,859 µs; 8 KB LL 6.5 µs vs NCCL 41.5 µs eager; world_size 2 only, open/unmerged, CI failing | SGLang PR #39605 | NCCL Q5; FR§3.3 | TP2 only |

### 2.5 Communication share and TP vs DP

| Number | Source | Transferability |
|---|---|---|
| "About 139 TP allreduces per verify step" for a hidden-5120, 64-layer model with 5-layer DFlash2 on TP4 (counted from model code, not traced) | HyperQwen #254 (NCCL Q6; PSO Q5) | same model shape |
| Field report, same shape, TP4 4× 5060 Ti Gen3 x8, vLLM 0.30.0: C8 default-temperature decode 370–406 tok/s, no breakdown | HyperQwen #254 (PSO Q3) | other GPUs, other engine |
| "Tensor parallel inference runs two all-reduces per layer per token" (secondary) | theneuralbase (NCCL Q6) | claim |
| 5090 TP1 beat 2× 3090 TP2 by +46/+51 % (Gemma 4 31B); Qwen3.6-27B on 2× 3090 TP2 peaks ~104 tok/s at N=2, falls to 54 by N=8 (agent shape); single GPU best for decode-dominated chat (22.2 vs TP2 20.5) | club-3090 FAQ; arXiv 2512.01644 (NCCL Q6) | different GPUs |
| Published 2×TP2 wins all at short context (128–8k) and high concurrency with KV to spare: club-3090 #773 c64 2.40×; biMEMO c8 ~1.38× [D]; DGX Spark C8 1.33×; at total concurrency ≤ 4 TP4 equal or ahead | FR§3.7 (from [TP2]) | not our regime |
| CPU starvation can stall communication and kernel launches even with CUDA graphs | arXiv 2603.22774 (NCCL Q6; PSO Q5) | abstract only |

### 2.6 Sampling, lm_head and FR-Spec

| Number | Source | Transferability |
|---|---|---|
| FR-Spec (Llama-3-8B, 128k vocab, T=0, own C/CUDA framework, 1× A800): LM head 49 % of EAGLE-2 drafting compute (62 % with softmax; transformer layers 33 %); tok/s full 180.99 (1.00×), 64k 196.86 (1.09×), 32k 201.87 (1.12×), 16k 196.69 (1.09×), 8k 185.98 (1.03×); acceptance from 3.89: 64k 3.80 (−2.3 %), 32k 3.63 (−6.7 %), 16k 3.40, 8k 3.13; 1.42× over SGLang-EAGLE-2, 1.82× over HF-EAGLE-2 (their EAGLE-2 is 1.28× faster than SGLang's); LM-head overhead cut 75 %; average 1.12× | arXiv 2502.14856 (PSO Q2) | Pays for MTP with several full lm_head passes; our DFlash2 does one candidate pass, so little gain (PSO Q2 inference) |
| Franzen used a 64k FR-Spec map with NEXTN/MTP on Qwen3.8-Flash-Next (single RTX PRO 6000, Pennyroyal fork): `FRSPEC=True`, `--speculative-token-map hot_tokens_64k.pt` (sha `becfa41d…`), `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`, INT4 MTP draft; write-up settings "Speculative steps 3", T 0.7, top-p 0.95, top-k 20, 10 streams; write-up does not mention FR-Spec | dfranzen cells; WRITEUP (PSO Q2) | different algorithm and GPU |
| SIMPLE: sampling 20–38 % of decode time on large-vocab models or with constrained decoding; share rises ~10 % from TP2 to TP8; sampling bubbles 22–40 % (Qwen-2.5-72B, 8× H100); baselines vLLM 0.10.1, SGLang 0.5.2; GPUs include L40 (PCIe 4.0); cross-rank reconciliation adds "several milliseconds" [claimed] | arXiv 2512.00719 (PSO Q3) | "share rises with TP" matches our case (PSO inference) |
| Qrita (k 50, p 0.9, vocab 128,256, batch 1024, GPU not named): naive sort 16.49 ms; sort + truncation 2.10 ms; Qrita 0.92 ms; FlashInfer returns one index, limiting spec-verify use [claimed] | arXiv 2602.01518 (PSO Q3) | scaled: our [96 × 248k] is ~5.5× fewer elements; naive sort ~3 ms on their GPU; L4 has ~10× less bandwidth than H100-class (PSO inference) |
| FlashInfer sorting-free sampling cut vLLM sampling time >50 % on 1× H100 (no per-kernel µs) | FlashInfer blog 2025-03-10 (PSO Q3) | H100 |
| UNIQUE radix top-k up to 2.0× faster than FlashInfer, 4.8× faster than torch.topk (H100, k=64) | arXiv 2605.27740 (PSO Q3) | different shape |
| DFlash2 throughput at T 1.0 / top-p 0.95 / top-k 20, SGLang 1× H200: concurrency 1 2.67–3.43× vs AR; 8 2.27–2.85×; 32 1.01–1.45× (MT-Bench 1,507 → 1,525 tok/s) | DFlash2 README (PSO Q3) | gain shrinks with concurrency, consistent with per-position verify costs (PSO inference) |

### 2.7 DFlash / speculative decoding

| Number | Source | Transferability |
|---|---|---|
| DFlash2 card: Qwen3.8-27B mean acceptance 4.80 (GSM8K 5.46 … MT-Bench 4.10); MTP 4.28; "Speculation block size: 8 (7 draft tokens per verification step)" | DFlash2 blog/card (PSO Q4; FR§a N01 row 11, N03 row 17) | our ARC trace 4.37 at 12 running, 3.87–5.10 per run |
| DFlash 2 blog, Qwen3-4B: selector +2.0M params / +0.6 % cycle latency; conv +16.5M / +0.7 %; together +1.3% | inco.ai blog (PSO Q4) | small model |
| DFlash paper Table 8: a model trained on large blocks run on small blocks loses only ~2–4 % τ vs a model trained on the small block (train16-test8 Math500 5.09 vs train8-test8 5.21); "large blocks can increase verification cost under compute-bound settings (e.g., large batch sizes)" | arXiv 2602.06036 (FR§3.5) | no per-position data |
| "Positions 4–7 usually 0.1 or less" (two users, unspecified workload); another user: 60–75 % at 4 tokens on coding | incoai discussion #9 (FR§3.5; FR§a N04 row 12) | anecdote; incompatible with our L8 in the survival sense (FR) |
| llama.cpp single stream: n_max 4 56.96 tok/s vs n_max 7 54.94 (n_max 4–5 sweet spot) | z-lab discussion #9 (FR§3.5) | single stream |
| #36548: draft depth 4 cut wrong answers ~3× (111 → 37 of 304); target-only control c8 0/304 | SGLang #36548 (FR§3.5; FR§a N01 rows 12 and 32) | [U] |
| Width 2 reported to crash | (FR§3.5, from [DFW]) | [U] |
| PR #36136 (dynamic verification, open): Qwen3-4B c12 −0.2 % to +10.8 %, gain only at c ≥ 12 | (FR§3.5) | [U] |
| PR #43012 block verification (v0.5.22): AIME25 c8 +1.76 % on the sampled path; does not work on greedy DFlash drafts | (FR§3.5; FR§5.1) | [M] other hardware |
| DSpark: vendor AL 3.62 vs DFlash2 4.80 | (FR§b B3) | — |
| Fine-tuned targets with DFlash2: acceptance ~3.6–4.2 (ThinkingCap card 3.1–4.2 top-token; Swift 1.5 AutoRound 3.85 at 7 drafts), no same-condition base comparison | (FR§5.2) | [C/M] |

### 2.8 GEMM, precision and clocks

| Number | Source | Transferability |
|---|---|---|
| L4 dense BF16 121 TFLOPS (sparse 242) | NVIDIA spec (FR§1.4; FR§a N01 row 15) | unverifiable locally |
| Marlin paper: speedups shrink with M and TP; A100×8 BS64 1.04×, BS128 1.07×; BS64–128 1.04–1.23× on multi-GPU A100/A6000; no Ada GPU | arXiv 2408.11743 (PROF Q3, Q4) | no Ada |
| Marlin real efficiency 30–40 % of DRAM bandwidth (A800 user, batch unspecified) | vLLM forum (PROF Q3) | Ampere |
| vLLM: FP8 W8A8 needs cc ≥ 8.9; "up to a 1.6x improvement in throughput" (generic) | vLLM FP8 docs (PROF Q4) | claim |
| "Give Me BF16 or Give Me Death": W4A16 most cost-efficient for synchronous setups, W8A8 dominates in async continuous batching; FP8 W8A8 effectively lossless; A100 roofline break-even W4A16/W8A8 about batch 78 (exact figure not verified) | arXiv 2411.02355 (PROF Q4) | A100; PDF body unreadable |
| RedHat INT4 recovers 98.5–98.7 % on AIME25/GPQA-D; FP8 "nearly identical" to BF16 | weight_quantization.md Q3 (PROF Q4) | same model |
| Franzen low-M BF16 GEMM patch: cuBLAS under graph capture serves decode projections (m ≤ 8) with 1-warp split-K WMMA at ~20–70 % of DRAM BW on sm120 (61.9 µs for m=4, n=4120, k=2560 = 320 GB/s on RTX PRO 6000); lm_head cuBLAS ~94 %; gated to capability (12,0) | build_bundle_pennyroyal.sh L92-102 (PROF Q5) | low prior for our M=96 |
| 5060 Ti power 180 → 150 W cost < 1 % at bs 1 | InventiveHQ (PROF Q3) | bandwidth-bound decode only |
| L4 is a 72 W part | StorageReview (PROF Q3) | — |
| Cohere/LiquidGEMM W4A8 numbers are Hopper | (PROF Q4) | no Ada kernel |

### 2.9 GDN, attention, compile, scheduler

| Number | Source | Transferability |
|---|---|---|
| SGLang keeps per-draft-position GDN snapshots: "for each draft position stores a full recurrent SSM state"; ~15.8× the normal state at D=16 | SGLang #28730 (PROF Q3) | same mechanism |
| Community packed GDN decode kernel: ~+7 % end-to-end on GB10 | HF baladhurgesh97/triton-gdn-decode (PROF Q3) | GB10 |
| 3090 vLLM: fp8-KV (FlashInfer) vs int8-KV (Triton) crossover ~7–10k context; Triton int8 +76.5 % short-context, −71.9 % at 58k | club-3090 #773 (PROF Q3) | backend choice is context-dependent |
| SGLang overlap scheduler 1.1× (v0.4, non-hybrid), largest on small models and large TP; "no single idle GPU interval" with Triton, "a minor gap" with FlashInfer | LMSYS v0.4 blog (PROF Q3; PSO Q5) | already on |
| torch.compile: up to 1.5× lower latency at small batch on small models (v0.3); spec docs: benefit "may be negligible" with CUDA graphs | LMSYS v0.3; SGLang docs (PROF Q3) | weak |
| Single Ada sm_89 (RTX 4080 SUPER 32 GB, AWQ, v0.5.18): decode graph capture hung; `tc_piecewise` "not yet implemented; falling back to 'full'"; eager decode ~6 tok/s vs 64–66 on llama.cpp DFlash2; `gptq_marlin_fp16` JIT stuck in NVCC for minutes | SGLang #36048 (PROF Q3, Q5) | sm_89 hybrid fragility |
| SGLang cookbook: `--chunked-prefill-size 2048` "since 8192-token chunks stall decode about 600ms at a time" | (PROF Q3) | we run 4096 |
| #43514: DFlash + Qwen3.8-27B, grammar requests + `--enable-mixed-chunk` + `extra_buffer` → whole batches emit token 0 with accept 0.00, "No error, no warning, no crash"; seen 5 days on 1 of 3 instances; ~30 % per round on a used instance (v0.5.19/0.5.20, TP4 H20) | SGLang #43514 (PROF Q3, Q5) | we do not use grammar or mixed chunk |

### 2.10 Agent, prefix cache, HiCache

| Number | Source | Transferability |
|---|---|---|
| Franzen (RTX PRO 6000, Flash-Next): p90 prefill 10,553 tok/s (PROF: 10,552) vs decode 836.6 (ratio 0.08); prefill time fraction 15.25 % with 93.43 % prompt-token reuse; miss only 0.03 % on prompts > 62k; KV-resident admission, trim in big blocks | Franzen WRITEUP §2 (FR§2.2, §2.3; PROF Q3) | prefix reuse worth 3–4× more on our hardware |
| Franzen Mamba patch: keep only the last prefill checkpoint + one cold-start branch; "refresh the end-of-prefill checkpoint to MRU when its final device lock drops"; "validated on source/control-flow tests only; GPU integration testing is pending" | (FR§2.2) | [C]; hook names in the 0.5.21 Rust tree unverified |
| Franzen applied 5 patches: low-M BF16 GEMM; spec-state memory-budget correction ("avoiding reservations for intermediate SSM buffers that are not allocated"); Marlin scale-dtype fix (GPTQ MoE); sparse Mamba prefill checkpoints with LRU refresh; one-shard-ahead parallel checkpoint prefetch | dafr README/WRITEUP (PROF Q5) | [C] |
| rellik13: KV 91–97 % full; giving freed fp8-KV space to history raised score 14.5 → 22.5 | (FR§2.2) | [C] |
| "Due to re-tokenization, the tokenized conversation might not exactly match the generated token sequence" | Franzen WRITEUP (FR§2.2) | [C] |
| HiCache: #39830 (open) host-tier hits on Flash-Next give fluent but wrong answers, 0/10 vs 10/10; #44735 wrong in v0.5.20, fixed in nightly; #45400 (open, v0.5.22, Qwen3.8-27B-NVFP4) host hits drop to zero after ~3 long prefixes are evicted, 56-slot host Mamba pool vs 702 Mamba backups | (FR§2.3) | [U] |
| club-3090 #1288: "927 tok/s with HiCache" — narrative 927.5, code 1,811, no HiCache-off baseline or hit rate | (FR§2.3; FR§b B2) | [U], no baseline |

### 2.11 Startup and storage (other people on Kaggle)

| Number | Source | Transferability |
|---|---|---|
| amn's probe: same shard 0.95 GiB/s on 10-20 → 0.026 GiB/s on 10-21 05:37; 10-22 single thread ~10.5 MB/s, 16 threads ~107 MB/s; from 2026-10-21 part of `/kaggle/input` is `FUSE.GCSFUSE` and NFS slowed | topic 751344 (FR§4.1; PROF Q2) | [M, user], same platform |
| Jeroen Cottaar: "reading through the mount did not fill the page cache"; upstream gcsfuse keeps `KeepPageCache = true` | (FR§4.1; PROF Q2) | mixed evidence |
| mp: detect FUSE and copy to local disk once | topic 751344 (PROF Q2) | [C] |
| Franzen resharded and prefetched "in the loader's order with parallel readers"; others reading 100–180 GB got 0.00 | (PROF Q5) | [C] |
| NguyenThanhNhan's two 0.00 scores: 45-min timeout, ~180 GB | (FR§4.1) | [U] |
| Docker default `/dev/shm` is only 64 MB | (FR§4.1) | [C] |
| Staff confirmed a slow "bad cache region"; post-start attach is billed; `ERRORED_MOUNTING_DATASET` exists; Jupyter start deadline (status code 56 precedent); staff: image version affects no timer; cm391 anecdote about the old image | (FR§4.4) | [C] |
| L4×4 bills at 2×; 30 h weekly ≈ 15 h of 4×L4 | topic 689054, Sam-wiz (PROF Q2) | [C] |
| Kaggle tip: do not unpack wheels under `/kaggle/working`; a finished run was marked ERROR at teardown because of 7 GiB of output | topic 732723 (PROF Q2) | [C] |
| FlashInfer sampling would not JIT on Kaggle because "the image has no linkable libcuda.so" | topic 732723 (PROF Q1) | [C] |

### 2.12 Correctness bugs relevant to speed work

| Number | Source |
|---|---|
| #42917: Qwen3.8-27B compressed-tensors W4A16 PPL 9.98 on SGLang 0.5.21 vs 6.05 on vLLM; root cause a stale CUDA driver compute cache under `$SGLANG_CACHE_DIR/nv`; clearing only `nv/` → PPL 6.2784, top-1 93.6 %, bit-identical across boots; clearing `jit/` did nothing | PROF Q5; FR§4.2 |
| #43412: pip-only CUDA on sm_89 (sglang 0.5.21, flashinfer 0.6.18): `top_k_renorm_probs` JIT-fails on the first non-greedy request when nvcc/cicc/ptxas versions mix ("Ninja build failed"), SIGKILLs the server, poisons the flashinfer cache; fix: pin `nvidia-cuda-nvcc`, `-crt`, `nvvm` to 13.0.* | PROF Q3, Q5 |
| #38009: DFlash2 + thinking diverges from target-only at token 30 (4× 3090 TP4); #39087: drafter that inherits target quant gives ~0 % acceptance | PROF Q2, Q5 |
| sgl_kernel `is_fa3_supported()` accepts sm_89 but no sm_89 cubin ships (#38980) | PROF Q3, Q5 |
| #35851 breakable-graph per-layer overhead on hybrid GDN; #35437 DFLASH + prefill graph capture failures | PROF Q5 |

### 2.13 Token-efficient model variants (not engine speed)

| Number | Source |
|---|---|
| Mean thinking-token reduction claims: Swift 1.0 ~41 % (GPQA), Swift 1.5 42 %, ThinkingCap-Qwen3.8 37 % (<1 pt average accuracy loss); all lose 2.7–4.7 pts on AIME 2026 | FR§5.2 [C] |
| Savings shrink with trace length: ThinkingCap MMMLU (1.7k) −65 %, Terminal-Bench (73k) −11 %; Swift 1.5 Terminal-Bench (52k) −16 % → expected −10 to −25 % at ARC xhigh ~60k | FR§5.2 [C]/[E] |
| MindStudio n=1: Swift used 60,572 vs 11,744 tokens and stopped mid-way | FR§5.2 [M] |
| `ukisai/Swift-1.5-Qwen3.8-27b-INT4`: compressed-tensors pack-quantized g128 `actorder: static`, 19.48 vs 19.47 GB; MTP head byte-identical to base (15 tensors hashed); `embed_tokens` and `lm_head` changed (DFlash2 borrows these from the target) | FR§5.2 [M] |
| ARC Prize semi-private Qwen3.8-27B: xhigh 42.4 %, low 22.8 %, medium 13.2 % (medium has no template instruction) | FR§5.2 [M] |
| arXiv 2606.00206 overthinking-marker logit penalty: 12–23 % shorter CoT (claim) | FR§5.2 [C] |

### 2.14 Competitor statements

| Statement | Source |
|---|---|
| TheDivine Aurora: ~19–20 % at 300–500 tok/s; estimates 40 % at 600–800 tok/s | FR§3.1; PROF Q5 [C] |
| Darren Amadeus Martin (rank 12, 2026-10-12): fix "present in the library code itself, which works no matter what GPU you use"; asked others for >500 tok/s aggregate on 4×L4, citing "lower bandwidth and slow gpu communication" | PROF Q5; FR§3.1 [C] |
| mikelou1 (2026-10-16): "a tiny kernel issue I found that made my program score only 40% (instead of the 60% projected by CV)"; scored 66 % (~60 % CV) on 10-18 | PROF Q5 [C] |
| "Domdolus Tolus patched library code for speed" | PROF Q5 [C] |
| NguyenThanhNhan flags (27B FP8 + EAGLE 3/1/4): `--attention-backend flashinfer --linear-attn-backend triton --chunked-prefill-size 2048 --max-prefill-tokens 4096 --mamba-radix-cache-strategy extra_buffer --mamba-full-memory-ratio 0.5 --mamba-ssm-dtype float32` | PROF Q5 [C] |
| Top teams (65–70 LB) published no tok/s | FR§3.1 [C] |

---

## 3. Levers (speed ideas)

Evidence classes: **measured-ours** (our Kaggle logs or runs), **measured-elsewhere**, **estimate** (arithmetic or model), **claim**. "Status" is what the four documents say, before the 2026-10-23 measurements; §6 updates it. FR priority IDs (E14-…) come from FR§d.

### 3.A TP communication (all-reduce, transport)

| # | Lever | Mechanism | Expected gain as stated | Evidence class | Cost / risk | Prerequisites | Status in docs | Source |
|---|---|---|---|---|---|---|---|---|
| A1 | `NCCL_P2P_LEVEL=SYS` or `PHB` (FR E14-N, P2) | Let NCCL use P2P (P2P/CUMEM) across the PHB host bridge instead of SHM; NCCL 2.29.7 default PATH_PXB picks SHM for ≥3 GPUs | FR: "cheapest lever", env only, no memory cost; HyperQwen AR −33.9 % (10 KB) … −6.1 % (1.31 MB), decode +2–16 %; RTX PRO 6000 +3.6 %; NCCL Q3: P2P "about 2x better than what we measured"; PROF Q3: AR ~40 ms → "about 8–12 ms" if P2P works; FR§3.1: AR halved −20–25 ms → ~+20 % (P2P level or PCIe-IPC); FR§d: "TP4 PHB +2–16 %" | measured-elsewhere | Silent hang under translated IOMMU/ACS (vLLM RFC #51513); needs a correctness all-reduce (rank+1 sum = 10) and a timeout; NCCL docs advise against forcing env in production; 8×A40 slightly slower; dual 5090 +2–5 %, sometimes negative | P2P must work on the Kaggle VM (`topo -p2p`, `can_device_access_peer`); bench with the same NCCL 2.29.7 | Untested; in E14-A P1b env matrix; production only if P1b shows ≥20 % faster at 983,040 B with sum check, lower engine step ms and greedy parity | FR§3.2; FR§d E14-N; NCCL Q1, Q4, Q6; PROF Q3 (NCCL transport; inference 2a) |
| A2 | `NCCL_SHM_USE_CUDA_MEMCPY=1` | Copy engines move SHM staging data | 5070 Ti: SHM BW 1.9 → 12.5 GB/s (6.6×) but 10 KB 39 → 81 µs; 8×A10: 1 MB 2× slower (199.6 → 414.4 µs); "roughly 2x on G5/G6" (claim) | measured-elsewhere (conflicting) | Higher launch latency; old PyTorch hang report (#803) | env only | Untested; FR: crossover is near our message size, so no decision without an A/B at exactly 983,040 B inside a CUDA graph; FR§c row 22 open | FR§3.2; FR§c 22; NCCL Q4; NCCL Q6 inference (b) |
| A3 | `NCCL_MIN_CTAS`/`NCCL_MAX_CTAS` (1/1, 4/4, 8/8), `NCCL_MAX_NCHANNELS=2`, `NCCL_MIN_NCHANNELS=8` | Change channel/CTA count | 1/1 "recovered" on g5 (weak); MIN_NCHANNELS=8 "significant" (no numbers); FR: low or negative expected | measured-elsewhere (weak) / claim | "Excessive CTAs can hurt small-message performance" | env only | Untested; in the env matrix | FR§3.2; NCCL Q4, Q7 |
| A4 | `NCCL_ALGO=Tree` / `Ring` | Algorithm choice | T4 SHM: Tree slower at 8 KB–1 MB; FR: low/negative | measured-elsewhere | — | env | Untested; in matrix | FR§3.2; NCCL Q4, Q7 |
| A5 | `NCCL_PROTO=Simple` / `LL` | Protocol choice | LL128 disabled on PCIe; same 1.9 GB/s under every proto on SHM; FR: low; NCCL Q7: "diagnostic only"; PROF: measure LL vs Simple at ~1 MB in a graph | measured-elsewhere | Forcing LL128 "can corrupt data" | env | Untested | FR§3.2; NCCL Q4, Q7; PROF Q3 inference 2b |
| A6 | `NCCL_NTHREADS=256` | Threads per CTA | 512 vs 256 +1.8 % on T4 (512 is default) | measured-elsewhere | — | env | Untested; in matrix | NCCL Q4, Q7; FR§d E14-A |
| A7 | `NCCL_BUFFSIZE` 1 MiB / 8 MiB | Buffer size (default 4 MiB per pair) | No measured effect found | — | — | env | Untested; in matrix | NCCL Q4, Q7 |
| A8 | `NCCL_SHM_DISABLE=1` | Forces sockets | Negative: α ~4×, BW −47 % | measured-elsewhere | — | — | Rejected | FR§3.2; NCCL Q4 |
| A9 | `NCCL_TOPO_FILE` / `NCCL_GRAPH_FILE` | Hand-fed topology or graph XML | g5.48xlarge DDP 7 → 4.5 h; +11 % on 2-NUMA Genoa | measured-elsewhere | — | XML file | Mentioned only; not in the FR priority list | NCCL Q4 |
| A10 | NCCL version pinning | Avoid regressions (2.20.5 halved BW vs 2.18.5) | — | measured-elsewhere | — | nccl-tests must load torch's NCCL 2.29.7 | Hygiene rule | FR§3.2; NCCL Q4, Q7 |
| A11 | `NCCL_CUMEM_HOST_ENABLE=0`, `NCCL_P2P_DISABLE=1` | Hang troubleshooting | not a speed lever | claim | — | — | Mentioned only | NCCL Q4 |
| A12 | FlashInfer PCIe-IPC all-reduce, `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1` (+ `SGLANG_PCIE_IPC_MAX_NUMEL`) (FR E14-I, P3) | Custom AR over CUDA IPC peer mappings for NVLink-less boxes; workspace sized for decode (`cuda_graph max_bs × draft tokens × hidden`), so prefill stays on NCCL; dispatched right after custom AR, created before graph capture; copy-engine path (`COPY_ENGINE_RING`) above ~1 MB | FR§d: unknown; if AR is 40–55 ms and halved → ~+20 %, fully removed → up to +45 %, ~0 if AR is small; published TP4 −10 % TPOT (bs1) / −5 % (bs4) on SM120; gain shrinks with batch | measured-elsewhere (SM120 only) | Needs FlashInfer ≥ 0.7.0 → upgrade (SGLang 0.5.22 pins 0.7.1; vLLM 0.31/0.32 pin 0.7.0.post1); bf16 only; TP group only; off in deterministic mode; kernels "spin" on peer flags with no timeout → hang if P2P is broken; IPC custom AR crashes with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; our `--disable-flashinfer-autotune` skips autotune; tuning cannot run under graph capture; numeric change; tested only on SM120 ("No architecture restriction" per FlashInfer JIT code) | P2P / CUDA IPC working on Kaggle; world size (2, 4, 8) per code | **Dead in our install**: `flashinfer/comm/pcie_ipc_ar.py` is absent in v0.6.18 / v0.6.18.post1 and present in v0.7.0 (44,053 B) and v0.7.1 (49,719 B); local `fi.whl` has no PCIe-IPC module; log would say "this FlashInfer build has no pcie_ipc_comm; falling back to NCCL". Untested; only after E14-U and only if E14-A P0 shows P2P OK; gate: log "FlashInfer PCIe-IPC all-reduce enabled (world=4)", lower ms/step, greedy parity | FR§3.3; FR§d E14-I; FR§c rows 6–8; NCCL Q5, Q6 inference (a) |
| A13 | lukealonso PCIe one-shot AR fork (`--enable-pcie-oneshot-allreduce`, `…-fusion`) | Each GPU writes to all peers over P2P, system-scope barrier; optional AR + RMSNorm fusion; auto crossover | Loses to NCCL at 1 MB (95.6 vs 85.2 µs); decode +5–11 % on Qwen3.5-397B TP4 | measured-elsewhere | P2P required ("about 15× slower than NCCL" without config); fork; sm_89 build unknown | P2P | Not recommended; doc: one-shot gives little at 0.94 MiB | NCCL Q5; FR§1.2 |
| A14 | Forced custom AR at TP4 | Bypass the TP4 gate | +14.8 % decode on 3090 | measured-elsewhere | Greedy divergence on 4/5 prompts, "not eval-clean" | — | **Rejected** ("should not be taken") | FR§3.3; FR§d; PROF Q3 |
| A15 | SGLang V1 / V2 custom AR | — | n/a at TP4 | code | V1 disabled `if world_size > 2 and not full_nvlink`, P2P test (`can_p2p`) never reached; V2 needs `full_nvlink is True`; at TP2 V1 turns on but the P2P check is monkey-patched to True by default → `--enable-p2p-check` required | — | Not available at TP4 | FR§1.2; FR§3.7; FR§a N04 row 2; NCCL Q5 |
| A16 | Torch symm-mem (`--enable-torch-symm-mem`) | One/two-shot AR on symmetric memory | — | code | Needs capability 9/10 table and multicast ("Device capability 8 not supported"); "Only supports CUDA device SM90 and above"; two-shot fallback PR closed unmerged | — | **Blocked in code** | FR§3.3; PROF Q3; NCCL Q5 |
| A17 | Quick AR | — | — | code | ROCm only | — | N/A | FR§3.3; NCCL Q5 |
| A18 | MSCCL++ (`--enable-mscclpp`) | MSCCL++ with NCCL fallback | unknown | code | `mscclpp` package not in our lock; untested on PCIe L4; not researched in depth | package | Untested | FR§3.3; PROF Q3; NCCL Q5 Gaps |
| A19 | NCCL symmetric memory (`--enable-symm-mem`) | — | unknown on PCIe | code | Forces `NCCL_NVLS_ENABLE=1`; NVLS needs NVSwitch | — | Not useful | FR§3.3 |
| A20 | Host-staged / LL all-reduce (SGLang PR #39605: `SGLANG_HOST_STAGED_ALLREDUCE=1`, `SGLANG_LL_ALLREDUCE=1`) | Stage through pinned host memory | 20.97 MB 1,494 vs 10,859 µs; 8 KB 6.5 vs 41.5 µs eager | measured-elsewhere | world_size 2 only, unmerged, CI failing | TP2 | Not usable at TP4 | FR§3.3; NCCL Q5 |
| A21 | FlashInfer AR + RMSNorm fusion / `trtllm_allreduce_fusion`; `boundary_reduction 'rs+rsv'` | Fuse AR with norm | — | code | Auto-enabled only on sm90/sm100; sm_89 undocumented; `rs+rsv` changes nothing in pure TP | — | Not available | FR§1.2; NCCL Q5 inference |
| A22 | Fewer ranks or fewer bytes per all-reduce (TP2 × DP2, DP4) | Halve or remove per-step communication; SHM latency grows almost linearly with bytes | not quantified | estimate | Costs KV capacity; DP4 only "if the INT4 model plus KV fits 24 GB" | — | Inference only (2×TP2 is row F7) | NCCL Q6 inference (c) |

### 3.B Logits, sampling, lm_head

| # | Lever | Mechanism | Expected gain as stated | Evidence class | Cost / risk | Prerequisites | Status in docs | Source |
|---|---|---|---|---|---|---|---|---|
| B1 | Vocab-parallel verify sampling (FR E14-L, P3) | Each rank takes its local top-20, then all-gathers only 20 values and ids (as the drafter's `candidate_topk` already does); removes the 47.7 MB logits all-gather and the dense FP32 passes; with top-k 20, exact target probabilities need only each rank's local top-20 and top-p renorm on the merged 80 | −6–14 ms/step → +5–12 %; PROF: cut the gather ">100×" | estimate (+ source reading) | Code patch (dflash_utils / logits path); exactness test (same sample distribution); GPU ~15 min; risk of changing the sampling distribution | Profile shows AG ≥ 5 ms | Not in 0.5.21; untested; an open SGLang PR #42863 works on speeding the logits all-gather on PCIe-only TP4 | FR§3.6; FR§3.1 table; FR§d E14-L; PSO Q1 "Fixes", Q5 ranking 2; PROF Q3 inference 2c |
| B2 | `--enable-tp-lm-head-all-to-all` | All-to-all instead of all-gather | none in pure TP | code | Guard needs `logits.shape[0] == local_rows * tp_size` (DP-attention layout) | DP attention | **Does not apply** | FR§1.3; PSO Q1 |
| B3 | Fused sparse top-k verify | Replace the dense FP32 [96, 248,320] passes (temperature divide, `torch.topk`, `zeros_like` + scatter, 83.4 MB `[bs,7,248320]` buffer) | dense passes ≈2.3–4.1 ms (PSO ranking "≈2–4 ms") | estimate | code | — | Untested | PSO Q1, Q5 ranking 4 |
| B4 | FR-Spec hot vocab on the drafter's candidate head | Slice the target lm_head copy used in `candidate_topk` to 64k hot rows and remap ids (selector codebooks indexed by global id need no change) | ≤ 1.6–1.9 ms (~1.5 % of step): 635.7 MB → ~164 MB per GPU, saves ~470 MB | estimate | Code; acceptance risk for tokens outside the hot set (FR-Spec −2.3 % at 64k) | — | `--speculative-token-map` silently ignored by DFLASH in 0.5.21 (EAGLE family only); low gain; "outside the list" | PSO Q2; FR§3.6; FR§d (low-gain list) |
| B5 | Avoid presence/frequency penalties | Penalties add a dense [bs, vocab] broadcast-add on the [96, 248k] FP32 logits | ~+0.7 ms if penalties are sent | estimate | — | — | Not checked whether our requests send them | PSO Q1, Q5 Gaps |

### 3.C GDN and attention kernels

| # | Lever | Mechanism | Expected gain as stated | Evidence class | Cost / risk | Prerequisites | Status in docs | Source |
|---|---|---|---|---|---|---|---|---|
| C1 | GDN verify kernel launch-config sweep (FR E14-K, P3) | `fused_sigmoid_gating_delta_rule_update` uses `_select_recurrent_launch_config`, tuned only for AMD gfx95; on CUDA `BV=min(next_pow2(V),32), num_warps=1, num_stages=3`, grid `(NV, N, HV)`, no `triton.autotune`; sweep BV ∈ {16,32,64}, warps ∈ {1,2,4}, stages ∈ {2,3,4} | −5–10 ms (4–8 %) if the profile shows GDN verify ≥ 15 ms; traffic floor ~6–7 ms | estimate (source shows untuned defaults; FLA config sensitivity on other GPUs) | Local monkeypatch + microbench; GPU ~15–20 min; must be bit-exact vs default | Profile GDN ≥ 15 ms | Untested | FR§3.6; FR§3.1 table; FR§d E14-K; PROF Q3 (GDN lever; inference 3) |
| C2 | ReplaySSM (`--enable-linear-replayssm-spec`) | Per-slot raw-input window replaces per-draft full-state snapshots; intermediate SSM buffer not allocated | PSO: saves ~6–7 ms, frees ~1.8–2 GB per GPU for KV | estimate | PSO: correctness and speed untested on DFLASH + GDN + FP16 SSM on sm89 | Needs topk ≤ 1, triton or flashinfer linear-attn backend | **Blocked for DFlash**: 0.5.21 configurator raises "--enable-linear-replayssm-spec with DSPARK/DFLASH requires a KDA model" (`kv_cache_configurator.py` L1129-1136); same in v0.5.22 and nightly; GDN PRs #36683, #42209 open; #42209 says existing compact replay is not bit-exact ("79 of 80 requests diverged"). Applies only to an MTP arm | PSO Q5; FR§3.6; FR§c row 2; FR§d "do not take" |
| C3 | FlashInfer GDN kernels | — | — | claim | Need SM90+ | — | N/A on L4 (stays Triton) | FR§3.6; PROF Q3 |
| C4 | Community packed GDN decode kernel | Packed Triton decode | ~+7 % end-to-end on GB10 | measured-elsewhere | not assessed on L4 | — | Mentioned only | PROF Q3 |
| C5 | Attention backend A/B (flashinfer vs triton, fp8 KV) | With fp8 KV, FlashInfer always uses the tensor-core prefill-template path; spec verify runs `speculative_attention_mode=prefill` (q_len 8) | Gain only if the current kernel is far from bandwidth; floor ~16 ms at 50k | measured-elsewhere (3090 only, context-dependent) | One relaunch | Profile shows attention > 25 ms at 40k | Untested; "outside the list"; FA3 not an option (no sm_89 cubin) | FR§3.6; FR§d; PROF Q3 (attention lever; inference 4) |

### 3.D Scheduler, CUDA graphs, host

| # | Lever | Mechanism | Expected gain as stated | Evidence class | Cost / risk | Prerequisites | Status in docs | Source |
|---|---|---|---|---|---|---|---|---|
| D1 | Backport SGLang #44401 (+ #41872) (FR E14-S, P1) | 0.5.21 counts a continuing chunked-prefill request twice at admission → running capped at `max_running_requests − 1`; the last chunk's pass latches `batch_is_full` until a request finishes. #44401 "Count a request's held req_to_token row once at admission" (merged 2026-10-15; fixes #38118, #35537). #41872 is the empty-batch latch fix | Whole run +3–8 %; 11 → 12 running at the same context ≈ +8–9 % [M/D] | measured-ours (signature: E11-05 2,358 lines, E11-06 158 lines, EVAL30 50 %, G4B 25 %; mamba 33/56) + merged PR + source | Small Python patch (scheduler.py +14/−6) as in-process monkey-patch; low risk (scheduler admission; mamba slots 56 ÷ 4 = 14 ≥ 13) | 0.5.21 has the pre-patch `get_num_allocatable_reqs()` and gate (`scheduler.py` L3784-3800, L3962-3980) and `req.kv.holds_kv`, so the diff should fit almost verbatim. **Not in v0.5.22** (compare "diverged"); only nightly ≥ 2026-10-16; v0.5.22 has only #41872 | Backport not run; gate: running == 12 during chunked prefill even with queue ≥ 1; PROF calls it the most plausible GPU-agnostic "library code" fix; PROF Gap: clean apply unconfirmed (PR mentions `get_num_allocatable_reqs(num_held_rows)` and three gates) | FR§3.4; FR§d E14-S; FR§c row 17; PROF Q3 (lever 1), Q5 |
| D2 | CUDA graph sizes 1..13 (`--cuda-graph-bs-decode 1 2 … 13`) (part of E14-S) | Default spec list `range(1,9) + range(10,33,2) + …` with `max_bs` appended resolved to `[1..8, 10, 12]`, max_bs 12, while `max_running_requests=13`: batch 9 replays the 10-graph, 11 replays the 12-graph (padded rows computed, ~8–10 % extra verify work), 13 runs with no graph | Part of the +3–8 % [E] | measured-ours (graph list) + source | Memory of 3 more verify graphs (current 10: target-verify 0.68 GB + draft 0.35 GB; PROF says ~0.88 GB) and capture time per new shape; no model risk (the 12-graph is unchanged) | — | Untested; FR proposes adding it to the E14-A server (design proposal of the FR) | FR§3.4; FR§d E14-S; PROF Q3 (lever 1) |
| D3 | `--disable-cuda-graph-padding` | Avoid padding to the next graph | — | code | Incompatible with torch.compile | — | Mentioned only | PROF Q3 (exec_.py L511, L546), torch.compile paragraph |
| D4 | Overlap scheduler | Already on (`disable_overlap_schedule=False`); `extra_buffer_lazy` requires it; DFlashWorkerV2 drives both modes | ≈0 (v0.4 gave 1.1×) | measured-elsewhere | — | — | Keep on | PROF Q3 (lever 6); PSO Q5 |
| D5 | `num_continuous_decode_steps` | — | none | code | Dead flag in 0.5.21 (only in `fields/schedule.py:194` and `field_order.py:67`) | — | **Rejected** | PROF Q3; FR§3.6; FR§d |
| D6 | `stream_interval` 1 → 4–8; more tokenizer/detokenizer workers; `batch_notify_size` | Fewer streaming wake-ups | Only if fwd occupancy < 90 % | claim (help text "A larger value makes the throughput higher") | Duck must tolerate chunkier streams; #43304 asks for IPC metrics | `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1` reading | Untested; weak | PROF Q3 (lever 7); FR§3.6 |
| D7 | torch.compile | Compile decode-graph runner (bs ≤ 32) | Weak or negative | measured-elsewhere | Adds cold start; custom ops (Marlin, GDN, FlashInfer); sm_89 hybrid graph variants fragile (#36048) | — | **Rejected** | PROF Q3 (lever 8); FR§3.6; FR§d |
| D8 | `--enable-mixed-chunk` | Mix prefill and decode | Negative | measured-elsewhere | Part of the #43514 trigger (accept 0.00 batch poisoning) | — | **Rejected** | PROF Q3 (lever 9); FR§3.6; FR§d |
| D9 | `--chunked-prefill-size` 4096 → 2048 | Shorter decode stalls during re-prefill | Protects decode latency; no steady-state decode gain | claim (cookbook) | — | — | Untested; "outside the list" | PROF Q3 (lever 5); FR§3.6; FR§d |

### 3.E Weights and precision

| # | Lever | Mechanism | Expected gain as stated | Evidence class | Cost / risk | Prerequisites | Status in docs | Source |
|---|---|---|---|---|---|---|---|---|
| E1 | FP8 W8A8 per-channel (compressed-tensors FP8 dynamic) via SGLang's sm_89 CUTLASS path | FP8 tensor cores; still memory-bound at M ≈ 96 (ridge M ≈ 400) | Wins only if Marlin ≥ 35 ms per step (Marlin below ~55–60 % of its floor); FP8 floor ~26 ms (PROF takeaway says ~32 ms) vs INT4 16 ms | estimate | 2× weight bytes; +2.5 GB weights per GPU removes ~300k of 881k KV tokens (PROF takeaway: ~40 % of KV pool); block-FP8 falls back to Triton on Ada; activation-quant overhead | CUDA ≥ 12.4; `weight_scale.numel() == weight.shape[1]`, N and K multiples of 16; checkpoint compatibility with GDN fused projections unchecked | Untested; profile decides; default stay INT4; "outside the list" | PROF Q4; FR§3.6; FR§d |
| E2 | `--dtype float16` | — | none | code | Marlin's atomic-add reduce is off for bf16 on sm < 9 and only used when n < 2048 and k ≥ 2048; all our per-GPU n ≥ 3584 (qkv+gate 3584, GDN qkvz 4096, gate_up 8704, o/out/down 5120) | — | **Rejected** ("do not spend a run") | PROF Q3; FR§1.4 |
| E3 | W4A8 hybrid | INT4 weights, FP8 activations | — | — | No Ada kernel in SGLang or vLLM (Cohere/LiquidGEMM numbers are Hopper) | — | N/A | PROF Q4 |
| E4 | Machete | Mixed-input GEMM | — | claim | Hopper only | — | N/A | PROF Q3 |
| E5 | Low-M BF16 GEMM patch (Franzen) | Replace 1-warp split-K WMMA for m ≤ 8 | — | measured-elsewhere (sm120) | Gated to capability (12,0); our BF16 GEMMs run at M = 96 / 84–96 where cuBLAS is usually efficient | — | Low prior; profile BF16-GEMM category will show it | PROF Q5 |
| E6 | Speculative-state memory-budget correction (Franzen; "mratsim's 0002"; SGLang #42651 post-capture draft KV sizing, opt-in env in v0.5.22) | Stop reserving memory for intermediate SSM buffers that are not allocated → more KV / mamba slots | not quantified | claim | Python-side patch | — | Untested; PROF ranks it the second most plausible "library code" fix | PROF Q5; FR§5.1 |

### 3.F Speculative decoding

| # | Lever | Mechanism | Expected gain as stated | Evidence class | Cost / risk | Prerequisites | Status in docs | Source |
|---|---|---|---|---|---|---|---|---|
| F1 | DFlash2 width 6 / 4 (`--speculative-num-draft-tokens 6/4`) (FR E14-W, P2) | Supported runtime override: worker warns "DFLASH block size mismatch…" and calls `set_block_size()` (integer + conv `block_size`, no weights). Not the same as cutting the last predictions: drafter `is_causal: false` (bidirectional intra-block attention, `ENCODER_ONLY`); grouped conv and selector walk are causal. Fewer rows cut every row-proportional cost (all-reduce bytes, GDN snapshots (W of them), logits all-gather, sampling) and the KV reserve (`2 × block_size`) | Model: W4 −10 % to +12 %, W6 −4 % to +7 % at c12 (DFW `step(W) = a + 12·b·W + d·W/8`, calibrated AR 289.4 / W8 403.7); geometric α 0.811 (L8 4.3) → L4 3.00, L6 3.79 (3.785); α 0.83 (L8 4.55) → L4 3.09, L6 3.95; flat lower bound L4 2.52, L6 3.54; 35.7 % whole-block acceptance pushes toward the lower bound; α ≈ 0.77–0.81 more realistic for 12-running decode; 550–650 tok/s would need L4 3.21–3.79 at 70 ms (near the max of 4); GPT's arithmetic "W4 ~515 tok/s at 70 ms" is correct as maths (FR§a N01 row 24), but note 00's premise that width 4 gives a ~70 ms step is called baseless (FR§b, 00 §2); #36548 wrong answers ~3× lower at depth 4 | estimate (model) + measured-elsewhere anecdotes | Flag change; relaunch per width 8–15 min (new-shape capture unknown) → Tier C 15–30 min, ~0.5–1 h quota; acceptance can drop; width 2 reported to crash | — | Untested; nobody has measured this checkpoint below width 8; 0.5.21 logs only the mean accept, no per-position histogram; gate: W40k tok/s ≥ W8 + 5 %, accept W6 ≥ 3.2 / W4 ≥ 2.5, answer probe not worse, real running == 12 | FR§3.5; FR§3.1 table; FR§d E14-W; FR§b (00 §2); FR§c rows 13–15 |
| F2 | Adaptive width (`--speculative-adaptive`) | Per-batch draft length | Qwen3-4B c12 −0.2 % to +10.8 % (gain only at c ≥ 12) | measured-elsewhere | EAGLE/EAGLE3 only in code (`adaptive_spec_params.py` L59-63); #36127 / PR #36136 open | — | Not available | FR§3.5; FR§a N01 row 26 |
| F3 | Block verification (#43012, v0.5.22) | — | +1.76 % on the sampled path (AIME25 c8) | measured-elsewhere | Does not work on greedy DFlash drafts; needs upgrade | 0.5.22 | Untested | FR§3.5; FR§5.1 |
| F4 | DSpark | Dynamic-planner drafter | not an easy gain | claim | Vendor AL 3.62 vs DFlash2 4.80; planner needs an offline SPS table, otherwise "verify-all"; 0.5.21 ragged verify is DSpark-only | — | Not tested | FR§b B3; FR§a N01 row 26 |
| F5 | DFlash2 vs NEXTN MTP at 12 real running (previous report's A/B 3) | Different drafter | unknown; marginal cost MTP 4.4 ms/request (1.10/token) vs DFlash2 6.87 (0.86/token); blog AL 4.80 vs 4.28 | measured-ours (marginal) | Relaunch | — | Never measured at 12 running; "outside the list" | FR§d; FR§a N01 row 13 |
| F6 | `--speculative-draft-model-quantization unquant` explicit | Prevent drafter inheriting target quant (#39087) | correctness | — | — | — | Already passed by our notebooks (FR§c 5); PROF recommends setting it | PROF Q5; FR§c row 5 |
| F7 | 2×TP2 replicas (FR E14-T, P5) | Two TP2 servers, half the ranks per all-reduce; simplest form: two `launch_server` (ports 30000/30001) with client-side sticky routing (`branch_id % 2`) | FR: realistic **0.7–1.0×** of TP4 (0.9–1.2× even if KV did not bind); KV per replica with DFlash2 ~170k (mem 0.83; ~224k at 0.88), MTP ~235k, no-spec ~500k; system ~340k vs **700–920k needed** (12 × 58–76k; rev 5 median 697k, p90 ~890k); each GPU gains ~5.3 GiB of weights, taken straight from KV; one replica holds 2 streams of 76k or 2–3 of 60k. Earlier claims (PREV +20–100 %; N00 1.3–1.6×, 0.35–0.4M KV/replica) withdrawn | estimate (SGLang's own sizing formula, matched W1 within 0.6 %) | Duck approval for routing; router's default 5 retries dangerous for 60k-token requests; router wheel not in wheelhouse; V1 custom AR auto-enables at TP2 with the P2P check monkey-patched True (`utils/common.py` L2400-2408) → `--enable-p2p-check` required; TP2 weights ~11.4–11.7 GB/GPU (target 9.24 + 2.19 GiB per TP2 note); with shared page cache 2×TP2 adds no extra weight read | — | **Nearly rejected** because of memory; FR keeps only "boot one replica and read `Load weight end` and `KV Cache is allocated`" (< 10 min, optional at the end of Tier A); TP4 communication levers come first because they do not lose KV | FR§3.7; FR§d E14-T; FR§c rows 11–12; FR§b (00 §3); NCCL Q6 inference (c) |
| F8 | PP2 × TP2, DP-attention | Other parallel layouts | — | code | "Currently DFLASH speculative decoding only supports pp_size == 1" (`arg_groups/speculative_hook.py` L274-283) | — | **Blocked in code** for DFlash | FR§3.7 |

### 3.G Prefix cache, Mamba checkpoints and agent behaviour (effective agent throughput)

| # | Lever | Mechanism | Expected gain as stated | Evidence class | Cost / risk | Prerequisites | Status in docs | Source |
|---|---|---|---|---|---|---|---|---|
| G1 | Duck prompt re-render fix (FR E14-P0 diagnosis P0; E14-P fix P3) | The chat template rebuilds the assistant turn as `'<\|im_start\|>assistant\n<think>\n' + reasoning_content\|trim + '\n</think>\n\n' + content` and rebuilds tool calls from parsed arguments as XML; any trim or non-canonical tokenization breaks the prefix early. Fix: send the assistant turn back exactly as generated, or a token-level session | G4B ~+10 % (S2 → S3, 325 → 357); reply re-prefill is 37–76 % of uncached / prefill time (EVAL6R5-type runs ~76 % of prefill time) [D] | measured-ours (diagnosis) / estimate (gain) | Duck client change needs user approval; must not change what the model sees; streaming session is append-only with a 300 s reaper | First: offline token-diff (generated ids vs next request's re-tokenized history), CPU only, 0 quota | Diagnosis designed; fix untested | FR§2.2; FR§d E14-P0, E14-P |
| G2 | Streaming session (`--enable-streaming-session`) | Session KV + Mamba state kept in one slot "invisible to eviction"; turns appended at token level; the only feature avoiding both eviction and re-tokenization | not quantified | code/claim (Dynamo docs) | Append-only (trim → close and reopen); 300 s idle reaper; no measured result on hybrid + DFlash | — | Untested | FR§2.3 (streaming paragraph) |
| G3 | `--radix-eviction-policy` (values priority, tlru, slru, lfu); `--enable-session-radix-cache` | Agent-oriented eviction policies | not quantified | code | — | — | Exist in 0.5.21; not evaluated | FR§2.3 (streaming paragraph) |
| G4 | Mamba checkpoint protection (FR E14-M, P3): port Franzen's refresh-on-unlock (works in `extra_buffer_lazy` too); `--max-mamba-cache-size` 56 → ~96 (≈ −72k KV); or int8 checkpoints | "mamba usage" counts only running requests' slots, leaving ~20 slots for cached checkpoints; lazy mode takes a temporary slot at track boundaries with "No evict-retry", so tracking silently drops if no slot | G4B S3 → S4 up to +14 % (zero/partial misses 317k tokens) [D]; causation not proven | estimate (from measured-ours correlations) | Patch port (0.5.21 Rust-tree hook names unverified); fewer KV tokens; hybrid cache bugs; Franzen's patch not GPU-tested; int8 checkpoint incompatible with HiCache | — | Untested; gate: share of zero/partial hits at low KV; G4-15 telemetry | FR§2.2; FR§d E14-M |
| G5 | `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` | Lazy mode 4 → 3 slots per request | not quantified | claim | — | — | Untested; "outside the list" | FR§b B4; FR§d |
| G6 | HiCache (FR E14-H, "later") | Host tier for KV + Mamba (wired for FULL + MAMBA hybrid; DFlash draft KV backed up as sidecar) | ~0 at KV max 0.44 [E]; only useful if more live sessions than device KV (e.g. 24 paused stage-2 branches) | estimate | Host RAM ~77 GB; `buffer_only` rejects MAMBA tree; int8 Mamba checkpoint incompatible; no retraction backup for Mamba; #39830 wrong answers on host hits; #45400 host hits drop to zero | — | Low priority, high risk | FR§2.3; FR§d E14-H; FR§c row 20 |
| G7 | Occupancy / probe fix (rev 6e: 12 fixed workers + KV gate 0.95 replacing rev 6d's 12 → 6 step-down) | Keep 12 streams admitted | G4B 279 → ~303; 12/12 → ~325 [D] | measured-ours (27.7 % at 6 running) / estimate | Duck/harness change (already in rev 6e) | — | Already changed in rev 6e (per FR) | FR§2.1; FR§2.3 |

### 3.H Engine version upgrades

| # | Lever | What it gives (as stated) | Evidence class | Cost / risk | Status | Source |
|---|---|---|---|---|---|---|
| H1 | SGLang 0.5.22 (2026-10-19, 950 PRs) + #44401 backport (FR E14-U, P3) | FlashInfer 0.7.1 → PCIe-IPC possible; #43029 (spec + structured output resumed from wrong mamba state); #41736/#41737 (NaN cross-request leak); #42709 (wrong tokens in compact-verify graph); #42651 (post-capture draft KV sizing, opt-in env); #43012 block verification (+1.76 % sampled); #40785 fused fp8 KV commit (SM89 should take the fused path, unmeasured); #43172 (graph max_bs clamp) | measured-elsewhere / claim | Lacks #44401; #36548, #38009, #43514/#43518, #37475, #39087 still open; ReplaySSM + DFlash still KDA-only; `tc_piecewise` removed; offline build: torch 2.14.1 (CUDA 13.0.3, same driver), sglang-kernel 0.4.9, cuda-tile 1.6.0rc5 still sdist only (keep override), cp312 manylinux_2_34 runs on glibc 2.35, ~220 wheels / 5 GB, one CPU build run + publish; nvcc mismatch trap still open upstream (PR #43591), keep our 13.0.88 pin; '+' names; new regressions | "Lowest build-risk path"; no like-for-like Qwen3.5/3.8 hybrid benchmark published for any version; gate: paired measurement by context band + greedy + A/A | FR§5.1; FR§d E14-U |
| H2 | SGLang nightly ≥ 2026-10-16 | Has #44401; #45435/#45472 hybrid fixes; no cuda-tile pin; FlashInfer 0.7.2, sglang-kernel 0.4.10 | claim | Wheel tag manylinux_2_39 that pip rejects on glibc 2.35 (native `.so` max GLIBC_2.34 per heuristic scan) → retag needed, untested | Option only | FR§5.1 |
| H3 | vLLM 0.32 (2026-10-18) | FlashInfer 0.7.0.post1 (PCIe-IPC via `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1`); #55601 (DFlash2 prefix-hit mamba index fix, first release); async DFlash (#58065, since 0.31) | claim | #54360 (spec decode on hybrid GDN silently loses prefix hits, open); #53912 (MTP + prefix cache corruption, open); #54690 (SM89 FP8 draft KV crash → bf16 draft KV); 0.31 2×3090 PP crash-loop; SM89 async MTP garbage (HyperQwen #121); torch 2.13.0 (same as our wheelhouse); cp38-abi3 manylinux_2_28; instanttensor hard dependency; jit-cache must be bundled; engine change means Duck prefix-hit behaviour must be rechecked | Option only | FR§5.1 |

### 3.I Startup and load (time and coverage, not decode rate)

| # | Lever | Mechanism | Expected gain as stated | Evidence class | Cost / risk | Status in docs | Source |
|---|---|---|---|---|---|---|---|
| I1 | Range-parallel copy of weights to `/dev/shm` (FR E14-0, P0) | Log mount type from `/proc/mounts`; `posix_fallocate` destination; copy/symlink small files; 64 MiB ranges, largest file first; 16–32 workers each with own fd, `os.pread` → `os.pwrite`, retry with backoff, `posix_fadvise(DONTNEED)`; MB/s every 30 s; verify size, safetensors header, last-tensor offset = file size − 8 − header length; `--model-path /dev/shm/int4`; fail open to `/kaggle/input`; chain `/dev/shm` (≥24 GB free) → `/tmp` or `/kaggle/tmp` → range-warmer (Franzen `precache()` style) | Avoid 0.00 from a server that never starts; slow day 669 s load → ~13 s from RAM + copy ~3.6 min at 107 MB/s [M/E]; copy can overlap pip/venv setup but launch must wait for copy + verify | measured-ours (the problem) / estimate (the gain) | ~150–300 lines in the load cell; RAM peak ~47 GB (copy + page cache) vs host 202.6 GB; `/dev/shm` and disk caps unknown | Designed; build needs no GPU; validated in E14-A P1a (4–6 min, cap 25) | FR§4.1; FR§d E14-0; PROF Q2 (P1a) |
| I2 | Stall-based timeouts | Phase 0 copy: <64 MiB in 10 min → abandon copy, launch from mount; phase 1 launch → `Load weight end`: stall if neither log nor process-tree `/proc/<pid>/io` `rchar` grows for 15 min on first load (10 otherwise); phase 2 → `Uvicorn running`: 12 min without a new line (first bs=12 capture is ~6 min silent, so > 8 min); keep `SGL_GIVE_UP_AT = min(T0 + 75 min, admit deadline − 5 min)`; never kill a progressing server; heartbeat every 30 s; on a real stall save log tail + `py-spy dump` or `/proc/<pid>/stack`, relaunch once from another source | Avoid false timeouts | estimate | No thinking-budget change | Designed | FR§4.3; FR§d E14-0 |
| I3 | JIT / compile cache tar | At the end of a successful commit, tar `$SGLANG_CACHE_DIR` **without `nv/`**, `TRITON_CACHE_DIR`, an explicitly set `SGLANG_JIT_CACHE_DIR` (default `~/.cache/sglang/jit`, not under `SGLANG_CACHE_DIR`, unset in our env) and `TORCH_EXTENSIONS_DIR`; name includes sglang/torch/triton/flashinfer/sm89 versions + `pip freeze`; untar to `/tmp/e11-cache/` before launch; known-answer greedy gate (64 tokens + first 16 logprobs vs cold-cache reference), else wipe and cold relaunch | ~360 s per cold start ≈ 0.8 % of a 12 h rerun, ≈ 7.5 % of an 80-min commit, ~32 quota-min per commit | measured-ours (capture time) / estimate (gain) | Never tested across sessions; hidden-rerun pool driver may differ; #42917 stale `nv/` corrupted W4A16 (PPL 9.98); width or TP changes create new shapes (~6 min again unless the cache carries them, untested) | Designed | FR§4.2; FR§d E14-0 |
| I4 | Stock loader flags | `--weight-loader-prefetch-checkpoints` splits files across ranks (`sorted_files[local_rank::local_world_size]`), reads each file whole and in order, disables multithread loading (our 2-file target → rank 0 reads 18.6 GB alone); default multithread loader (8 threads per rank) is per file and each TP rank reads every tensor whole then shards; `fastsafetensors` per file and not in wheelhouse; `presharded` untested with INT4 + DFLASH; `--startup-weight-load-mode=overlap` rejects speculative decoding, TP4, quantization and draft worker | — | code | — | **Do not help** | FR§4.1 |
| I5 | First-cell instrumentation | Line flushed before touching `/kaggle/input` (time, hostname, `uname -r`, image digest); `/proc/mounts`, `df`, `free`, 64 MiB timed read per input; competition JSON read in a thread with timeout; heartbeat file in `/kaggle/working` | Diagnoses the 10-22 ERROR class | — | — | Designed | FR§4.4; FR§d E14-0 |
| I6 | Fresh `SGLANG_CACHE_DIR` / `CUDA_CACHE_PATH` per session; keep the nvcc + crt + nvvm trio pinned to 13.0.* | Avoid stale-cache corruption (#42917) and JIT crash (#43412) | correctness/availability, not speed | measured-elsewhere | Conflicts with I3 only if `nv/` is shipped | Recommended | PROF Q5 inference (d), (e) |
| I7 | Warm the GDN Triton JIT with one long prefill before any steady window | 24 × ~1.8 s compile after the first long prefill | benchmark hygiene | measured-ours | — | Designed into E14-A P2/P3 | PROF Q2 |

### 3.J Token efficiency (requests per GPU-hour, not engine speed)

| # | Lever | Expected gain as stated | Evidence class | Cost / risk | Status | Source |
|---|---|---|---|---|---|---|
| J1 | Token-efficient variant (FR E14-V, "later"): Swift-1.5 INT4 drop-in; alternatives ThinkingCap, Apache-2.0 overthinking-marker logit penalty (arXiv 2606.00206), Qwen Pi (Apache-2.0, agent-harness-specific), own length-aware fine-tune | Mean −10–25 % tokens → 1.18–1.33× more requests at fixed throughput, only if ARC accuracy per request holds; AIME −2.7–4.7 pp | claim / estimate | **Licence**: Swift Open License v1.0 Sec. 5 (entities with ≥ US$1M revenue "not licensed"); ThinkingCap PolyForm Small Business 1.0.0 (gated, `isOsiApproved: false`); Rule 1.a requires OSAID-open weights → prize eligibility seems incompatible (user/host decision; host has not answered topics 746307, 750575); DFlash2 AL may drop; SGLang's Qwen3 thinking-budget processor hard-codes old ids 151667/151668 (Qwen3.5+ needs 248068/248069) and spec decoding can run tokens past it (#25536); 60k floor unchanged | Untested; one paired EVAL run (hours) | FR§5.2; FR§d E14-V |

### 3.K Gates the docs attach to speed changes (not speed levers)

| ID | Gate | Facts as stated | Source |
|---|---|---|---|
| E14-G (P1) | DFlash2 vs target-only correctness, Tier B | (1) health: no window with accept rate 0.00, W8 mean ≥ 3.5, no 50+ token repetition; (2) thinking off, greedy, 20 prompts × 256 tokens, serial, D vs T and T vs T across restart (A/A), at the first divergence T's top-1/top-2 margin (> 1 nat = bug); (3) #36548 template: 2–3 single-answer greedy prompts, 150 trials at c12 (batch filled with real ARC requests) + 50 serial; (4) workload equivalence: 32 ARC prompts, ~60k cap, c12; (5) batch-invariance control (T alone vs in a 16-batch). Cost: Tier B ~+10 min (steps 1, 2, 5 and target-only relaunch); step 3 ≈ 13 rounds × ~57 s ≈ 12 min per arm at c12, serial 50 × 2k at 45–107 tok/s ≈ 16–37 min per arm, two arms ~1–1.5 h wall = 2–3 quota-h; step 4 ~1 GPU-hour per arm ≈ 2 quota-h per arm. Pass: D error rate ≤ 2 % and McNemar p ≥ 0.01 vs T; exact match ≈ A/A rate; no large-margin flips; no divergence that grows with concurrency only in D. `--enable-deterministic-inference` with DFLASH + mamba unverified; token equality not expected at 60k | FR§d E14-G |
| E14-A P5/P6 | Greedy token-ID gate | 24 fixed prompts, T=0, thinking on, max_tokens 768, serial then c=12, save text + token ids; P6 target-only relaunch gives token-ID match rate and first divergence | PROF Q2 (P5, P6) |
| E14-A rule 6 | Accept-collapse alarm | Accept len < 1.5 for 3 consecutive decode lines → `spec_collapse`, no profile, go to the greedy gate | PROF Q2 rule 6; FR§d run order |
| — | PROF's weighting | A silent correctness fault, not slowness, is the most plausible "tiny kernel issue costing ~20 points"; the P5/P6 gate and a per-window accept alarm are "worth more than any single speed lever" | PROF Q5 inference |

### 3.L Measurement tooling named in the docs (not speed levers)

- **Free counters**: `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1` adds `fwd occupancy: X%` and `fwd+idle occupancy: Y%` to each decode line (GPU forward ÷ wall); `SGLANG_RECORD_STEP_TIME=1` appends `gap_latency / decode_log_interval` to `step_time_dict[num_running_reqs]`, returned by `/get_server_info` next to `avg_spec_accept_length`. PROF: occupancy ≥ 95 % at 12 running → host/scheduler gaps are not the problem; ≤ 85 % → CPU-side levers move up (PROF Q1).
- **Torch profiler**: `/start_profile` / `/stop_profile` (GET and POST); `ProfileReq` fields `output_dir`, `start_step`, `num_steps`, `activities` (CPU, GPU, MEM, RPD, CUDA_PROFILER), `profile_by_stage`, `with_stack` (default True), `record_shapes` (default False), `profile_id`, `merge_profiles` (asserted off in one path), `profile_prefix`, `profile_stages`, `detailed_annotations`; one `<profile_id>-TP-<rank>.trace.json.gz` per rank; ends with a barrier. Env: `SGLANG_TORCH_PROFILER_DIR` (default `/tmp`), `SGLANG_PROFILE_WITH_STACK` (False avoids the "Python replay stack is empty" crash), `SGLANG_PROFILE_RECORD_SHAPES`, `SGLANG_PROFILE_V2`, `SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS` (capture only full-admission decode steps), `SGLANG_ENABLE_NVTX_SCHEDULER`, `SGLANG_ENABLE_NVTX_OPERATIONS`, `SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE`, `SGLANG_GRAPH_BATCH_CAPTURE`, `SGLANG_MEM_PROFILE_MAX_ENTRIES` (PROF Q1).
- Suggested call: `{"num_steps": 20, "activities": ["CPU","GPU"], "with_stack": false, "record_shapes": false, "profile_by_stage": true, "profile_id": "tp4w8_ctx40k"}` with `SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS=12`; estimated trace ~10–30 MB gz per rank (PROF Q1).
- **Graph kernels in traces** appear on one stream without Python stack; kernel annotations need PyTorch 2.15+ and driver/CUDA-compat ≥ 13.1 (we have torch 2.13); `TEARDOWN_CUPTI=0` with graphs → one profiling window per server process (PROF Q1).
- **Other flags**: `--enable-layerwise-nvtx-marker` (needs CUDA graphs off), `--enable-profile-cuda-graph` (capture only, one trace per bs with `SGLANG_GRAPH_BATCH_CAPTURE`), `--debug-cuda-graph`, `--disable-decode-cuda-graph`; NVTX needs the `nvtx` pip package (PROF Q1).
- **nsys**: no pip wheel; usable only as a pre-uploaded standalone CLI; `nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-graph-trace=graph --trace-fork-before-exec=true --capture-range=cudaProfilerApi --capture-range-end=stop`; `--cuda-graph-trace=node` "may cause significant runtime overhead"; not known whether the Kaggle image ships nsys (PROF Q1).
- **Benchmarks**: `sglang.benchmark.one_batch`, `one_batch_server` (supports `--profile --profile-by-stage`), `serving`, `offline_throughput`, `steady_state`; dataset `random` downloads ShareGPT, `random-ids` is offline-safe but distorts DFlash acceptance; use real ARC prompts with long shared prefixes (PROF Q1, Q2).
- **Trace parsing design**: keep `cat == "kernel"`, group by `cudaGraphLaunch` via `args.correlation` / External id; categories NCCL all-reduce, NCCL all-gather, Marlin, BF16 GEMM (cuBLAS/CUTLASS), GDN Triton, attention (FlashInfer prefill-template), sampling/accept/commit, norm/rope/activation/quant, Memcpy/Memset, other; per collective take the minimum across ranks (max − min = skew), never sum ranks; cross-check Σ categories (TP0) ≈ `step_time_dict[12]` within ~10 %, and accepted tokens × 12 ÷ step = gen throughput (PROF Q1).
- **NCCL measurement**: topology dump (`nvidia-smi topo -m`, `topo -p2p r/w/p/a/n`, PCIe gen/width query, `/proc/cmdline`, `dmesg | grep iommu|dmar`, `lspci -vvv | grep ACSCtl`, `can_device_access_peer` 4×4); `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,ENV,TUNING` with `NCCL_TOPO_DUMP_FILE` / `NCCL_GRAPH_DUMP_FILE` and grep for ` via `; `ar_bench.py` CUDA-graph sweep (sizes 8,192 … 983,040, 1,048,576, 2,097,152, 67,108,864); env matrix baseline, `P2P_LEVEL=SYS`, `PHB`, `SHM_USE_CUDA_MEMCPY=1`, `MIN/MAX_CTAS` 4/4, 8/8, 1/1, `ALGO` Tree/Ring, `PROTO` Simple/LL, `NTHREADS=256`, `BUFFSIZE` 1 MiB/8 MiB, each 3 repeats; nccl-tests built offline against the pip NCCL (`-G` graph mode, `NVCC_GENCODE` sm_89); simpleP2P or nvbandwidth for a real P2P check (NCCL Q7). PROF P1b adds the 983 KB all-reduce × 128 captured in one graph, a 47.7 MB all-gather, and pairs (0,1), (2,3), (0,2), (1,3) for TP2 (PROF Q2).
- **Clocks**: `nvidia-smi dmon -s pucvmet -d 2` throughout; power pinned at 72 W with SM clock below ~1.5 GHz during W40k = throttling confound (PROF Q2 rule 10).

---

## 4. Contradictions or corrections between documents

FR text is translated from Bengali; the quotation marks enclose my translation.

### 4.1 Between the four documents I read

1. **All-reduces per step.**
   - PROF Q3: "128 all-reduces per target pass" and "Per-layer all-reduces: about 0.98 MB each (12 × 8 × 5120 × 2 B), 128 per target pass."
   - NCCL Q6: "about 2 per layer × 64 = 128 for the target model, plus drafter all-reduces (smaller). Use ~128–139."
   - PSO Q5 budget table: "TP all-reduces (~139 x ~0.27–0.4 ms)".
   - FR§1.2: "in total 140 all-reduces per step, each 96 × 5120 × 2 = 983,040 B" [S]: verify = embedding 1 + GDN `out_proj` 48 + full-attention `o_proj` 16 + MLP `down_proj` 63 + last layer inline 1 = 129, plus 1 logits all-gather; draft = 10 (`o_proj` + `down_proj` × 5) + eager embedding 1 = 11, plus 2 small all-gathers (top-16 values 5,376 B, ids 10,752 B); plus 1–2 small broadcasts at accept. FR§c row 9: resolved at 140.

2. **Logits all-gather time.**
   - PROF Q3: "At the measured about 3.1 GB/s … that would be about 10–15 ms [I]."
   - PSO Q1: "t ≈ 0.75 x 47.7 MB / 4.7 GB/s ≈ 7.6 ms per step. Range: about 10 ms at 3.5 GB/s busbw; about 3.6 ms if P2P gives about 10 GB/s."
   - FR§3.6 keeps 3.6–15 ms; FR§c row 10: "open (must be measured)". (The 2026-10-23 measurement of 8.0 ms settles it; see §6.)

3. **How much of the step is unexplained.**
   - PROF Q3: "A rough per-step floor … is about 45–55 ms of a measured 115–133 ms step [I]. The remaining 60–80 ms is unattributed."
   - PSO Q5: "The measured step is consistent with this decomposition, and **no single 'mystery' component is needed**."
   - FR§1.3 / FR§c row 3: "resolved (by definition), but unmeasured": PROF's floor leaves out collectives, logits path and drafter extras; after PSO's central estimate "about 7–25 ms is still unexplained".

4. **PSO's own total.** PSO Q5 takeaway: "Total: 16 + 16 + ~28, plus about 35–55 ms of all-reduce (NCCLN), is about **95–125 ms**". PSO Q5 budget table: "**≈87–131 (central ≈108)**". Same file, two ranges.

5. **ReplaySSM with DFlash.**
   - PSO Q5: "The STATIC ragged mode imposes no DFLASH restriction [code]"; "That would save roughly 6–7 ms and free about 1.8–2 GB per GPU for KV"; Gaps: "It appears permitted by the code paths read."
   - FR§3.6: "ReplaySSM is not available with DFlash: 0.5.21's configurator raises `--enable-linear-replayssm-spec with DSPARK/DFLASH requires a KDA model` for non-KDA models (`kv_cache_configurator.py` L1129-1136, checked by this report)." FR§c row 2: PSO "read attention_hook/exec_, not the configurator". Resolved: not available.

6. **Bytes of target weights read per step.**
   - PROF Q3: "INT4 weights: about 4.9 GB per GPU per pass, about 16 ms at 300 GB/s"; PROF Q4 table "INT4 W4A16 | memory (4.9 GB) | 16 ms".
   - PSO Q5: "Our measured 4.62 GB/GPU of target weights minus the 0.64 GB embedding shard ≈ 3.98 GB read per step → 13.3–15.9 ms". FR§1.3 uses 3.98 GB.

7. **Drafter cost per step.**
   - PROF Q3: "BF16 drafter: 3.85 GB ÷ 4, about 1 GB per GPU, about 3 ms."
   - PSO Q4: drafter resident ≈1.45 GB computed / 1.49 GB measured per GPU (FC 262 MB, conv 131 MB, codebooks 254 MB replicated); reads "about 1.19 GB of its own weights per GPU … plus the 635.7 MB target lm_head shard. That is ≈6–7.5 ms of pure memory time plus 10 drafter all-reduces."

8. **Second lm_head read and a possible second big gather.**
   - PROF Q3: "about 2 ms per use (target verify; possibly again for draft logits)"; "unverified … whether the drafter does a second gather"; PROF Q3/Q5 Gaps: "Whether the DFlash drafter path in 0.5.21 gathers full-vocab logits per draft position … was not traced."
   - PSO Q1/Q4: the drafter has no head of its own, re-reads the target shard for 84 rows, runs FlashInfer radix top-16 per shard, then all-gathers only K values and ids ("O(tp*K) instead of O(vocab)"). Answer: second read yes (2.1–2.5 ms), second large gather no.

9. **GDN verify state traffic.** PROF Q3: "about 1.8 GB per step, about 6–7 ms. Reads add about 0.23 GB." PSO Q5: writes 6.0–7.25 ms + initial-state reads 0.75–0.9 ms + commit gather-scatter 453 MB 1.5–1.8 ms = "≈8.3–10 ms per step". PROF omits the commit.

10. **CUDA graph memory.** PROF Q3 inference 1: "the current 10 graphs take about 0.88 GB per GPU [L]". FR§3.4 and FR§a N04 row 9: "target-verify 0.68 GB and draft 0.35 GB" (= 1.03 GB). PSO Q4: "draft-verify CUDA graph 0.38 GB [measured]". Three different numbers.

11. **Draft-model quantization.** PROF Q5: "Our server args show `speculative_draft_model_quantization=None` … The `unquant` flag is still worth setting explicitly [I]." FR§c row 5: "resolved (coordinator verified): the notebooks pass `--speculative-draft-model-quantization unquant` (E11-03/05 builder, G4B cell line 791), and server_args records it as None."

12. **Is PCIe-IPC usable in our 0.5.21 install?**
    - NCCL Q5: "The only in-tree candidate is `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1` (v0.5.21)"; NCCL Q6 (a): "If P2P works on Kaggle: `NCCL_P2P_LEVEL=SYS` and possibly `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE=1` could remove a large share."
    - FR§3.3: "But in our install it can never turn on … `flashinfer/comm/pcie_ipc_ar.py` is not in the v0.6.18 and v0.6.18.post1 tags … our wheelhouse pins `flashinfer-python==0.6.18`." FR§c row 6: resolved, dead code; needs FlashInfer ≥ 0.7.0.

13. **PCIe-IPC world sizes.** NCCL Q5 (from the PR text): "world sizes 8 and 4 (none for 2)". FR§3.3 (source): "(2, 4, 8)". FR§c row 7: resolved by code; both include 4.

14. **PCIe-IPC copy-engine path at our message size.** NCCL Q5 inference: "Its copy-engine path above 1 MB fits our 0.94 MiB messages." FR§3.3: "the docstring's large multipliers are for small 12–384 KiB messages, and its copy-engine path turns on above about 1 MB [E]" with the gain at 1 MB "unknown". Our 983,040 B is below 1 MB, so the two readings point in opposite directions.

15. **PCIe-IPC on sm_89.** NCCL Q5 Gaps: "Whether FlashInfer `pcie_ipc` kernels compile or run on sm_89 (docs silent; PR tagged SM120)." FR§3.3: FlashInfer JIT says "No architecture restriction: the kernels use only plain PTX loads/stores and CUDA IPC"; FR§c row 8: partly resolved, untested on Ada.

16. **Expected P2P gain magnitude.**
    - NCCL Q3: "If P2P were active, we would expect about 2x better than what we measured."
    - PROF Q3: AR "about 40 ms; if P2P works, about 8–12 ms" (~4×).
    - PSO Q5 ranking: "P2P would roughly halve" the logits all-gather; PSO Q1: "about 3.6 ms if P2P gives about 10 GB/s".
    - HyperQwen as cited in NCCL Q4 and FR§3.2: only −6.1 % at 1.31 MB.
    - FR§3.1: "All-reduce halved (P2P level or PCIe-IPC) −20–25 ms".
    The expectations range from −6 % to −80 %. (All are refuted by the 2026-10-23 bench; see §6.)

17. **Communication share of the step.** NCCL Q6: "roughly **30–50% of our 115–130 ms step**", central "44–53 ms per step ≈ 35–45%". FR§0: "about 35–55 ms, i.e. 30–45 % of the step". PROF Q3: AR ~40 ms plus AG 10–15 ms. Overlapping but not identical.

18. **Per-call SHM cost.** PSO table uses "~0.27–0.4 ms" per all-reduce; NCCL Q3/Q6 and FR use 340–350 µs (loose 340–380 µs).

19. **Host CPU.** PSO Q5: "G2 uses the Intel Xeon Platinum 8273CL (Cascade Lake): base 2.2 GHz, all-core turbo 2.9 GHz, single-core max turbo 3.7 GHz [claimed, vendor doc]". NCCL Q1: GCP "only available on the Cascade Lake platform". PROF Q3: "listed as Cascade Lake by a third-party aggregator … Another lists a Skylake variant (Cloud Mercato); the sources conflict." FR§c row 23: open, `lscpu` in the first cell.

20. **Startup duration.** PROF Q2 (citing GPT note 01): "E11-05 ready in about 7 min 39 s, E11-06 about 7 min 30 s, rev 6d G4B about 19 min 9 s". FR§4.2: same endpoint ("fired up") gives "E11-05 8:00, E11-06 7:52, E12 8:05, G4B 19:09"; FR§c row 19: different endpoints, no real conflict. PROF Q2 also gives "ready in 507–562 s on 10-19/10-20".

21. **Relaunch cost for width arms.** PROF Q2: "later launches 136–166 s"; P7 Tier C "[2×(3+4) min]", Tier C "+12–15 min". FR§c row 21: "LV: a new shape may need ~6 min capture"; open; Tier C must be budgeted at 15–30 min; FR§d E14-W: 8–15 min per width.

22. **FlashInfer in the spec-verify sampling path.** PROF Q3: "The spec verify path calls `flashinfer.sampling.top_k_renorm_probs` (`srt/speculative/eagle_utils.py:931`, per the issue)." PSO Q3: "The verify path (`build_dflash_verify_target_probs`) uses `torch.topk`, not FlashInfer, for the k = 20 target probabilities [code] — `dflash_utils.py:1165-1166`." PROF's line is about the EAGLE utilities; for DFLASH, PSO's code reading says torch.topk.

23. **FP8 W8A8 floor and KV cost (inside PROF).** PROF Q4 takeaway: "It moves 2× the weight bytes of INT4: about 32 ms vs a 16 ms floor per step … doubles weight memory, which costs about 40 % of the KV pool." PROF Q4 table: "FP8 W8A8 | memory (about 7.7 GB) | 26 ms"; inference: "+2.5 GB weights per GPU removes about 300k KV tokens of the 881k" (~34 %). FR§3.6 uses "~26 ms floor" and "~300k KV tokens".

24. **Marlin's position on the roofline (inside PROF).** PROF Q3: "At M=96, Marlin is at the knee" (ridge M ≈ 100). PROF Q4: "INT4 is at about 0.7× of its ridge (near the knee)" (compute floor 10.7 ms vs memory floor 16 ms).

25. **2×TP2 framing.** NCCL Q6 (c): "Configurations with fewer ranks per all-reduce (TP2 × DP2) … remove or halve the per-step communication, at the cost of KV capacity." FR§3.7: "2×TP2 nearly rejected": ~170k KV per replica with DFlash2 (mem 0.83; ~224k at 0.88), ~340k system-wide vs 700–920k needed; realistic throughput 0.7–1.0× of TP4 (0.9–1.2× if KV did not bind).

26. **11 vs 12 running penalty.** PROF Q3: "about 8 %" (E11-05). FR§1.1, §2.1: "−8–9 %" / "about −9 %". Same direction, slightly different size.

### 4.2 Contradictions the FR records with notes I did not read (FR§c, all 24 rows)

| # | Topic | Sides | FR resolution |
|---|---|---|---|
| 1 | GDN layers BF16 or INT4 | LV, DFW: BF16 (config ignore list). PSO: must be INT4 to fit 4.62 GB/GPU | **Resolved INT4**: ignore list has, per GDN layer, exactly `linear_attn` (parent, not a Linear), `norm`, `in_proj_a`, `in_proj_b`; 192 of 304 entries; no `in_proj_qkvz`/`out_proj`; target `["Linear"]`, 4-bit g128 symmetric. BF16 GDN (`in_proj_qkvz` 5120 → 16,384 + `out_proj` 6144 → 5120 = 115.3M params/layer, 5.54B total, 11.07 GB, 2.77 GB/GPU) would give ~6.5–6.7 GB/GPU and would not fit the 18.6 GB body (BF16 GDN 11.07 + BF16 embed + lm_head 5.09 GB leaves only 2.4 GB for 64 MLP layers that need ~8.8 GB even in INT4). BF16 only for `in_proj_a/b` (5120 × 48), norms, lm_head, MTP head, vision |
| 2 | ReplaySSM with DFLASH | PSO yes; NEV, N05, E11 README no | Resolved: no (see 4.1 #5) |
| 3 | Unexplained step portion | PSO vs PROF | Resolved by definition, unmeasured; ~7–25 ms left |
| 4 | Cause of the 10-22 G4B ERROR | N05: pinned image; SLE: attach/mount stall most likely, image weak, quota unlikely | **Open**; ranking (1) attach/mount stall, (2) image, (3) quota (94 vs 100 min coincidence, but E9 precedent), (4) our code ruled out (empty log) |
| 5 | Drafter really unquantized | PROF vs DFW | Resolved: `unquant` passed |
| 6 | PCIe-IPC in FlashInfer 0.6.18 | LV unverified; NCCL needs #4393; TP2 absent in local wheel; NEV absent in v0.6.18 | Resolved: absent |
| 7 | PCIe-IPC world size | NCCL PR text vs source | Resolved (2, 4, 8) |
| 8 | PCIe-IPC on sm_89 | NCCL untested vs NEV "No architecture restriction" | Partly resolved |
| 9 | All-reduce count | 128 / ~139 / 140 | Resolved 140 |
| 10 | Logits all-gather time | PROF 10–15 vs PSO 7.6 (3.6–10) | Open (measure with an 11.9 MB/rank `all_gather_into_tensor`) |
| 11 | TP2 weight and KV | GPT 01/03 12.22 GB; LV ~11.7; TP2 note 9.24 + 2.19 GiB, ~170k KV/replica; N00 9.3 GB, 0.35–0.4M/replica; PREV 0.7–0.8M system | Resolved for TP2 note (formula matches W1 within 0.6 %); confirm by booting one replica (< 10 min) |
| 12 | 2×TP2 gain | PREV +20–100 %; N00 1.3–1.6×; GPT 01 likely best; GPT 04 not yet; TP2 note 0.7–1.0× (0.9–1.2×) | Resolved for TP2 note: nearly rejected |
| 13 | L8 in the width model | DFW 4.55 (α 0.83); GPT 4.3 (α 0.811); E11-05 decode median 3.8; E11-06 12-running 4.34 | Resolved (definition): 4.55 is a per-request median incl. short phase-2 answers; α ≈ 0.77–0.81 for 12-running decode |
| 14 | "Positions 4–7 ≤ 0.1" | N00, GPT 04 quote it; DFW: impossible with our measurements; GPT 03: recall vs survival | Resolved: does not hold for our workload in the survival sense |
| 15 | Max tokens at width 4 | GPT 01: 4; GPT 04: > 4 | Resolved: 4 (source: block = anchor + (w−1) drafts) |
| 16 | Nature of 115.5 ms | GPT 03: regression; LV: raw median | Resolved: raw median |
| 17 | Cause of 11-of-12 | AGENT "cause not established" (mamba reservation possible); PROF, NEV, PREV #44401 | Nearly resolved: #44401 signature matches, mamba 33/56; unproven without a GPU backport test |
| 18 | E11-03 W8 step | DFW ~137 ms (client 403.7 tok/s); LV 115.5 (decode line) | Resolved (definition): client total includes prefill and tail |
| 19 | Startup length | GPT 01 7:39 / 7:30 / 19:09 | Resolved: endpoint differs; 8:00 / 7:52 / 19:09 |
| 20 | HiCache potential | N05: neglected big lever (927 tok/s); AGENT: low priority, high risk | Resolved for AGENT |
| 21 | Relaunch cost per width | PROF 3 + 4 min; LV ~6 min new-shape capture | Open |
| 22 | `NCCL_SHM_USE_CUDA_MEMCPY` | 5070 Ti 6.6× BW; 8× A10 2× slower at 1 MB | Open; A/B at 983,040 B needed |
| 23 | g2-standard-48 CPU | Cascade Lake (GCP doc, 8273CL) vs Skylake (Cloud Mercato) | Open, low importance |
| 24 | Page cache filled through the mount? | Jeroen: no; upstream gcsfuse + our 10-21 drafter: yes | Open; a local copy removes the dependency |

### 4.3 Corrections the FR makes to earlier claims (FR§a FALSE / PARTLY TRUE verdicts and FR§b)

- **FALSE** (FR§a N03 row 10): GPT 03's "115.5 is perhaps a context-normalized regression, not a raw median". It is the raw median (n=67, all 0–10k).
- **FALSE** (FR§a N04 row 16): GPT 04's "more than 4 output tokens possible at width 4 with the bonus-token convention". Block = anchor + (w−1) drafts (`dflash_worker_v2.py` L2466-2467); max accept len = 4.
- **FALSE** (FR§a N04 row 18): LV's added remark "48 GDN layers BF16, so INT4 GEMM is undercounted". The large GDN projections are INT4.
- **PARTLY TRUE** items in FR§a: N01 row 4 (per-step aggregate weighting); row 17 (115.5 vs central); row 21 (128 count); row 22 (64 MB BW cannot extrapolate: true only at 10 KB); row 27 (TP2 weights ~11.4–11.7 GB, not 12.22, because ~0.52 GB of drafter fc/codebook is replicated); row 31 (#37475 and #39087 do not affect us; #28180 closed 2026-08-17 but penalty follow-ups #41738, #43207 open); row 36 (2×TP2 as main lever changed by memory); row 37 (biggest effective-rate lever is on the Duck side); N03 row 13 (no LayerCommunicator on the Qwen3.5 path in this build; `layer_boundary` stage exists); N03 row 18 (4.3 only for E11-06 B=12); N03 row 24 (cap of ~22 min too optimistic); N03 row 25 and N04 row 24 (quota is session wall × 2, not GPU count × time: 30–40 min = 1–1.33 quota-hours); N04 row 6 (137.9 ms depends on filter).
- **UNVERIFIABLE locally** (FR§a): L4 121 TFLOPS; PCIe Gen4 64 GB/s bidirectional (~32 one-way) and 300 GB/s (no `nvidia-smi -q` ever logged); tonyd2wild repo; Learning to Draft (2603.01639); SpecKV (2605.02888); MLSys 2026 "Speculative Decoding: Performance or Illusion?"; "~200 web queries"; "top Kaggle teams made the same optimization".
- FR§b corrections of Claude's notes 00 and 05:
  - 05 A1 "PCIe Gen4 x16 64 GB/s" is bidirectional; one-way ~31.5–32 GB/s; host-staged SHM sets the limit, not the link.
  - 05 A2 "46 % wasted": accept len includes the bonus; 3.3/7 ≈ 47 % draft accept; rejected work is not all waste; width 4 max is 4.
  - 05 A3 "only 10–16 TFLOPS so not compute-bound" not established; Marlin M=96 near ridge; torch GEMM 76–79 TFLOPS.
  - 05 A4 "128 × 64 MB bandwidth = 40–50 ms", later corrected to "small collectives latency-bound": that correction also needs correcting: count 140; 0.98 MB/call is SHM bandwidth-bound, so extrapolation (~340–380 µs/call → 48–53 ms) roughly holds but is unmeasured; logits all-gather (47.7 MB) not counted; latency-bound only applies at 10 KB.
  - 05 A5 PARITY notebook shas stale (f3876b40 / 85da1be3 / bc9e2bc2); current rev 6g PARITYR6 is `82b257a5…`, never run.
  - 05 A6 cm391 "old image slow" is now a candidate, ranked after attach/mount stall; unresolved.
  - 05 B1 effective ≠ decode confirmed (EVAL30 167.8 / 187 old; G4B 279 / 277 old).
  - 05 B2 HiCache checked: low priority, high risk; #1288 has no HiCache-off baseline.
  - 05 B3 DSpark: AL 3.62 vs 4.80; SPS table needed.
  - 05 B4 `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` still untested.
  - 05 B5 "ReplaySSM not with DFlash2 (KDA-only)" was right; PSO wrong.
  - 05 B6 FP8 KV scale: checkpoint has 32 scales, engine says "defaulting to 1.0"; config has `kv_cache_scheme` static per-tensor FP8; runtime use still unverified.
  - 05 B7 Swift/ThinkingCap: −10–25 % expected at ARC length (1.18–1.33×), not headline 37–42 %; licences not OSI.
  - 05 B8 lm_head/sampling/FR-Spec now estimated: logits path 6–14 ms; second lm_head 2.1–2.5 ms; FR-Spec ignored for DFLASH, ≤ 1.6–1.9 ms.
  - 05 B9 new versions checked: #44401 only in nightly; DFlash2 correctness bugs open; PCIe-IPC needs FlashInfer ≥ 0.7.
  - 05 B10 load path: prefetch flag is per file; `SGLANG_JIT_CACHE_DIR` separate; `/dev/shm` and disk size unknown.
  - 00 §1 "0.85–1.3 ms per extra verified token = GEMM": maths right, but more than half of the marginal cost is probably all-reduce growth.
  - 00 §1 115.5 ms is the ≤10k raw median; production 115–140 differs by context and prefill.
  - 00 §2 suspect 1: 128 AR, "~0.42 ms/verified token", 40–50 ms → count 140, 983,040 B/call, 45–53 ms same order; logits AG extra.
  - 00 §2 suspect 2: 8 → 4 gives ~70 ms and 550–650 tok/s → 70 ms baseless; L4 ≈ 2.5–3.1 → model −10 % to +12 %; 650 needs L4 3.79.
  - 00 §3 biMEMO ~1.38×, "1.3–1.6× for us" → rejected (one TP2 vs TP4, short answers; our case ~170k KV/replica → 0.7–1.0×).
  - 00 §3 "INT4 TP2 ~9.3 GB weights; 0.35–0.4M KV per replica; 5–6 long requests" → 9.3 GB is target only; with drafter ~11.4–11.7 GB; KV ~170k (0.83) / 224k (0.88) = 2–3 streams of 60–76k.
  - 00 §3 "reading weights twice on GCSFUSE" → all 4 TP ranks already read the same files; with shared page cache 2×TP2 adds no read; a local copy removes the problem.
  - 00 §3 "LB ~50 at 2.4×" → estimate only (ORDER2 bias caveat); 2.4× from 2×TP2 now unrealistic.
  - 00 header "my validation in note 02" → note 02 was not in the folder; FR§a and LV do that work.
  - 00 recommendation (one measurement run; 2-server only if > 1.3×) → first part right; the TP2 arm should shrink to "boot one replica and read memory".

---

## 5. Open questions the docs say only a GPU run can answer

Items marked **[answered 10-23]** are settled or partly settled by the 2026-10-23 facts in §6.

**Step decomposition**
1. The real split of the 115–133 ms step. Apart from weight and KV sizes and the 64 MiB all-reduce, every budget row is arithmetic; no nsys or torch-profiler trace exists (FR§u; PSO Q5 Gaps; FR§1.3).
2. Kernel timings on sm_89 for: Marlin at M = 96 (efficiency vs its 16 ms floor); FlashInfer FP8-KV prefill/verify at 48 rows per KV head (could be compute- or latency-bound); `torch.topk` on [96, 248,320] FP32; FlashInfer `top_k` / `top_k_renorm_prob`; lm_head GEMM at M=96, N=62k, K=5120 (FR§u; PSO Q1, Q3, Q5 Gaps; PROF Q3 Gaps).
3. GDN Triton verify time and whether a tuned launch config helps on sm_89 (no sm_89 tuning data anywhere) (FR§u; PROF Q3 Gaps).
4. Achievable L4 DRAM bandwidth (250 GB/s is an assumption) (FR§u; PSO Q5 Gaps).
5. fwd occupancy at 12 running (host gaps) and power/clock throttling at the 72 W cap (PROF Q1, Q2 rule 10, Q3 "GPU clocks").
6. Kineto kernel names for our Marlin, FlashInfer fp8-KV and NCCL 2.29.7 builds (regexes need one real trace); whether `profile_by_stage` treats `TARGET_VERIFY` as decode for `SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS`; `/start_profile` export time on Kaggle's CPU (PROF Q1, Q2 Gaps).

**Communication**
7. Whether P2P / CUDA IPC works on Kaggle 4×L4; which transport NCCL picks; latency at 8 KB–1 MB; Kaggle `topo -m` was never published (FR§u; NCCL Q1, Q2, Q7; PROF Q2, Q3 Gaps). **[answered 10-23: PHB on all pairs; baseline 345 µs at 983,040 B; forced P2P works but is 5.5× slower; 8 KB–1 MB curve not in the facts given to me]**
8. Logits all-gather time on our box (11.9 MB per rank `all_gather_into_tensor`) (PSO Q1 Gaps; FR§3.6). **[answered 10-23: 8.0 ms]**
9. Whether `NCCL_SHM_USE_CUDA_MEMCPY=1` helps at 983,040 B under CUDA graphs (FR§c row 22). **[answered 10-23: no]**
10. Effects of `NCCL_BUFFSIZE` and `NCCL_MIN_CTAS` on 4-rank SHM at 1 MB; ALGO/PROTO/NTHREADS at our size (NCCL Q4 Gaps). **[not in the 10-23 facts]**
11. Whether FlashInfer `pcie_ipc` kernels run on sm_89; MSCCL++ on sm_89 PCIe; lukealonso fork build status (NCCL Q5 Gaps).
12. ACS/IOMMU mode of the G2 passthrough; whether the container allows `dmesg`, `lspci`, `nsys`; whether `nvidia-nccl-cu13` ships `include/nccl.h` (NCCL Q1, Q7 Gaps).
13. Whether TP all-reduce overlaps compute in decode (the "no overlap" statement is an inference); TP4 vs TP2 vs DP4 on 4×L4 (no published comparison) (NCCL Q6 Gaps).

**Speculative decoding and correctness**
14. Acceptance and per-position survival of this checkpoint at width 4/6 (0.5.21 logs no per-position histogram) (FR§u; FR§a N03 row 16).
15. Whether DFlash2 + GDN is supposed to be bit-exact with target-only (no maintainer statement), and the #36548 error rate on our stack; whether `--enable-deterministic-inference` works with DFLASH + mamba (FR§u; FR§d E14-G).
16. Relaunch cost of a new width (new-shape graph capture) and whether the same-session JIT cache carries it (FR§c row 21).

**Concurrency, memory, agent**
17. GPU proof that #44401 is the 11 + 1 queued cause; the backport has never been run (FR§u; FR§c row 17).
18. Real memory and graph size of one TP2 replica (boot one replica, read `Load weight end` and `KV Cache is allocated`, `--enable-p2p-check`) (FR§u; FR§d E14-T).
19. Exact cause of reply re-prefill (template trim / re-tokenization vs missing decode checkpoint) and of zero hits at low KV (Mamba LRU, trim, or #22935). The token-diff part needs no GPU (FR§u; FR§2.2).
20. Whether the FP8 KV scale is actually loaded at runtime (FR§b B6; FR§d low-gain list).

**Startup and platform**
21. `/dev/shm` and local disk size; hidden-rerun pool machine; which `/kaggle/input` subtrees are GCSFUSE; whether a shipped JIT cache is valid across sessions; the stage where the 10-22 ERROR happened (FR§u; FR§4.1; PROF Q2 Gaps). **[partly answered 10-23: `/dev/shm` held the 19.47 GB INT4 copy]**
22. Whether nsys is in the Kaggle image (PROF Q1 Gaps).
23. CPU model (`lscpu`: 8273CL Cascade Lake or Skylake) (PSO Q5 Gaps; PROF Q3; FR§c row 23).

**Model**
24. Swift or ThinkingCap on ARC (nobody has run it); host answer on non-OSI licences (FR§u; FR§5.2).
25. No team claiming 600–800 tok/s has published its tok/s or method (FR§u).

**Run-plan facts in the docs (for reference, not a design)**: E14-A as one fail-closed notebook. Tier A = P0 env/topology (cap T+3) + P1a weight copy (cap T+25) + P1b NCCL microbench (cap T+12, concurrent with P1a) + P2 TP4 DFlash8 launch (cap 20 min) + P3 W8k and W40k 90 s windows at real running = 12 (cap 14 min) + P4 profiler (cap 6 min) + P5 greedy gate (cap 8 min) ≈ 35–45 min wall, ~1.2–1.5 h quota. Tier B (P6 target-only relaunch, 24 prompts + 60 s W40k) +10 min. Tier C (P7 widths 6 and 4) +12–15 min (FR: maybe 15–30). Full run ~60–70 min wall (~90 on slow storage) ≈ 2.0–2.4 h quota, worst case ~3 h; global cap ~80 min (stop at cap − 5 min); append-only JSON with fsync after each phase; P1a not done by T+25 → never launch from the mount; real running ≠ 12 for > 60 s → `invalid_concurrency`; one profiler window per process; output tar ≤ ~300 MB; never rebuild a run notebook; user's explicit go required. Suggested account: hosen42 (8.63 h) for the full run; foysal (4.71 h) only Tier A + B (FR§d run order; PROF Q2).

---

## 6. Check against the 2026-10-23 measurements

Arithmetic in this section that is not in the docs is marked **(my arithmetic)**; implications not stated in the docs are marked **(my inference)**.

### 6.1 Fact-by-fact verdicts

| 2026-10-23 fact | What the docs assumed or predicted (source) | Verdict |
|---|---|---|
| `nvidia-smi topo`: all 4 GPU pairs **PHB** | NCCL Q1 inference: "On GCP g2 the L4s most likely appear as PHB pairs … With 4 GPUs at PHB, stock NCCL picks SHM even if P2P works"; FR§1.2: "default transport is almost certainly SHM [E]"; NCCL Q1: probably one NUMA node, SYS penalty probably not in play | **Confirmed** (PHB). No pair is SYS, consistent with the single-host-bridge reading. Since all pairs are equal, the TP2 pairs (0,1), (2,3), (0,2), (1,3) planned in PROF P1b have no topological difference (my inference). |
| Baseline 983,040 B all-reduce in a CUDA graph = **345 µs per call (p90 346)** | FR§1.2 / NCCL Q3: "983,040 B ÷ 3.1 GB/s ≈ 317 µs, plus α of about 20–30 µs ≈ **340–350 µs per all-reduce**"; NCCL Q6 and FR loose range 340–380 µs; HyperQwen interpolation ≈380 µs (FR: ≈383); PSO table 0.27–0.4 ms; NCCL Q2 "1 MB about 200–500 µs" | **Confirmed**, at the centre of the FR/NCCL α–β estimate; ~10 % below the HyperQwen interpolation. Implied algbw 2.85 GB/s, busbw 4.27 GB/s (my arithmetic). The FR's bandwidth-bound reading of a 0.94 MiB SHM call (FR§1.2, FR§b A4) holds. |
| Eager 381 µs | NCCL Q2: SHM ~12–25 µs per small call in graphs vs ~40 µs eager (5070 Ti 22–25 vs ~41 µs) | **Consistent** (+36 µs eager overhead). Relevant to the one eager embedding all-reduce per step (PSO Q5 eager list). |
| 64 MiB busbw **4.9 GB/s** | E11-03/E11-04 eager: 20.76–20.89 ms → busbw 4.82 GB/s (FR§1.2); published SHM plateau 2.4–4.6 GB/s; Kaggle T4 3.97/4.58 | **Confirmed** SHM-class bandwidth, slightly above the top of the published SHM plateau (consistent with NCCL Q3's "about 10 % better … SHM on a better host"). |
| Logits-sized all-gather **8.0 ms** | PSO Q1: "≈7.6 ms per step" (3.6–10); PROF Q3: "about 10–15 ms"; FR§3.6: 3.6–15 ms, FR§c row 10 open; FR§3.1 lever table "logits all-gather 3.6–10" | **Confirms PSO** (within ~5 %); **refutes PROF's 10–15 ms**; closes FR§c row 10. Implied algbw 47.7 MB / 8.0 ms = 5.96 GB/s, busbw × 0.75 = 4.47 GB/s vs PSO's assumed 4.7 (my arithmetic). |
| Estimated comm per decode step **~56 ms** (140 all-reduces + all-gather) | FR§0: all-reduce ~35–55 ms (30–45 % of step), central 48–53 ms (140 × 0.34–0.38); plus AG 4–10 ms; NCCL Q6: 44–53 ms (35–45 %); PROF: ~40 ms AR + 10–15 ms AG; FR§1.3 marginal breakdown: "more than half of the marginal cost is probably all-reduce growth" | **Confirms the "large communication" hypothesis**, at the top of the docs' ranges: 140 × 345 µs = 48.3 ms + 8.0 ms = 56.3 ms ≈ 42 % of a 133 ms step and 49 % of a 115 ms step (my arithmetic). Also confirms the FR's count of 140. Caveat: this is still a standalone microbenchmark summed per step, not an in-engine profiler trace (FR§u: a trace is still needed). The FR's B=1 → B=12 all-reduce growth (~45 ms using HyperQwen's 64 µs at 8 tokens) cannot be checked, because the 81,920 B latency on our box is not in the facts given to me. |
| `NCCL_P2P_LEVEL=SYS` (P2P/IPC transport) = **1,887 µs per call (5.5× slower)**, busbw **0.41 GB/s**, all-gather **81–84 ms** | NCCL Q3: "If P2P were active, we would expect about 2x better"; PROF Q3: AR "if P2P works, about 8–12 ms"; PSO: "P2P would roughly halve" the AG, "about 3.6 ms if P2P gives about 10 GB/s"; Spare Cores g2-standard-48 "P2P 10.1 GB/s"; HyperQwen PHB KVM −6.1 % at 1.31 MB; FR§3.2 "cheapest lever"; FR§3.1 "AR halved −20–25 ms"; FR§d E14-N gate "≥ 20 % faster at 983,040 B" | **Refuted.** P2P is available to NCCL (no hang reported) but far slower than SHM on Kaggle's VM; the E14-N gate fails by a wide margin. Every P2P-based gain number in the docs (HyperQwen, RTX PRO 6000, Spare Cores 10.1 GB/s) does not transfer. The docs' risk warnings fit the observed slowdown branch: NCCL troubleshooting doc ("IO virtualization (VT-d/IOMMU) can redirect all PCI point-to-point traffic to the CPU root complex, causing slowdowns or hangs", NCCL Q1) and NCCL maintainers ("NCCL now avoids P2P where it would perform poorly, even when nominally supported", NCCL Q1). NCCL's default choice of SHM was the right one. Correctness of the forced-P2P result is not stated in the facts given to me. The transport string "P2P/IPC" rather than HyperQwen's "P2P/CUMEM" is consistent with `NCCL_CUMEM_ENABLE=0`, which SGLang sets (FR§1.2) if the bench mirrored it (my inference; unknown). |
| `NCCL_SHM_USE_CUDA_MEMCPY=1`: graph part **timed out at 240 s**; eager **500 µs** | NCCL Q4 / FR§3.2: crossover near our size, needs an A/B at 983,040 B in a graph; 8×A10: 1 MB 2× slower (199.6 → 414.4 µs); 5070 Ti: BW 1.9 → 12.5 GB/s; old hang report with PyTorch (#803) | **Refuted for our case**: eager 500 vs 381 µs baseline eager = +31 % slower (my arithmetic); the CUDA-graph run did not finish in 240 s. Matches the A10 side of FR§c row 22 and the old hang report; closes that open row. |
| Decode step at 12 running ~115–133 ms (earlier logs) | FR§0, §1.1: 115 ms (≤10k) to 133–140 ms (50–80k); PROF header 115–133 ms | **Same numbers** (no new information). |
| First SGLang launch **ready in 502 s** from `/dev/shm`: weights phase ~65 s, then **~410 s of compile/graph capture with no log growth** | PROF Q2: "ready in 507–562 s on 10-19/10-20"; FR§4.2: cold start ~7.5–8 min, target-verify capture 371–385 s, first bs=12 bucket 362.28 s; FR§4.3 phase-2 stall limit 12 min ("must stay > 8 min"); FR§d E14-0: load from RAM ~13 s | **Ready time confirmed** (healthy-storage class). **The ~6-min silent first capture is confirmed** and slightly longer (~410 s ≈ 6.8 min): a 12-min phase-2 limit leaves ~5 min margin; an 8-min limit would leave only ~1.2 min (my arithmetic). **The ~13 s RAM-load estimate is refuted**: ~65 s, about 5× longer, though still ~10× better than the 669 s GCSFUSE load. The cache-tar lever (I3) targets this ~410 s; its claimed ~360 s saving is the right size (my inference). 502 s ≈ 16.7 quota-minutes at 2× (my arithmetic). |
| Relaunch in the same session ready in **71 s** | PROF Q2: later launches 136–166 s (E11-03); FR§4.2: warm-cache capture stage 11.94–13.67 s (E11-04) | **New, better than PROF's 136–166 s.** Same-shape relaunches are cheap (~2.4 quota-minutes). Whether new shapes (width 6/4, other graph bs lists) also relaunch fast is still unknown (FR§c row 21 stays open). |
| Staging copy of **19.47 GB INT4 at 518 MB/s** (16 workers; plateau **786 MB/s**); **single-stream read 34.6 MB/s** | FR§4.1/§d: copy ~3.6 min at 107 MB/s (16 threads, 10-22 user probe); single thread ~10.5 MB/s; current single-stream `cat` ~37 min at 10.5 MB/s; G4B 10-21 ~28 MB/s; PROF P1a "4–6 min at ≥ 100 MB/s"; `/dev/shm` size unknown, Docker default 64 MB | **Confirms the range-parallel copy lever** (I1): 518 / 34.6 ≈ 15× over single stream. Copy took ~37.6 s (19.47 GB ÷ 518 MB/s), not ~3.6 min; a single-stream `cat` at 34.6 MB/s would take ~563 s ≈ 9.4 min (my arithmetic). Storage that day was ~3.3× faster single-stream and ~4.8× faster multi-stream than the 10-22 probe, so storage speed varies by day (consistent with FR§4.1). **Resolves the `/dev/shm` size gap**: it held ≥ 19.47 GB (the Docker 64 MB default does not apply). Whether the 3.85 GB DFlash2 file was also staged is not stated in the facts given to me. |

### 6.2 What the residual budget looks like now (my arithmetic, using the docs' non-communication rows)

- At ≤10k context: measured 115.5 ms − comm 56.3 ms = 59.2 ms left for everything else. The PSO/FR non-comm rows at ≤10k (weights 13.3–15.9; KV ~1.6–2.0 assuming a ~5k average context; GDN 8.25–9.95; logits FP32 passes 2.3–4.1; drafter layers + FC + candidate lm_head 6.27–7.55; selector 0.2–0.5; host 2–7; activations 0.5–1.5) sum to ~34–49 ms, leaving **~11–25 ms unexplained**.
- At 50–60k: measured 133.3 ms − 56.3 = 77.0 ms left; the same rows with KV at 50k (16.4–19.7) sum to ~49–66 ms, leaving **~11–28 ms unexplained**.
- This matches the FR's "about 7–25 ms still unexplained" (FR§1.3, FR§c row 3). The profiler trace is still the only way to close it.

### 6.3 Status changes for the docs' levers (my inference unless a doc is quoted)

| Lever (§3 row) | Status before 10-23 | Status after 10-23 |
|---|---|---|
| A1 `NCCL_P2P_LEVEL=SYS` (E14-N) | untested, "cheapest lever" | **Refuted on Kaggle** (5.5× slower). `PHB` was not measured in the facts given to me; in HyperQwen both PHB and SYS switch NCCL to the P2P transport, so PHB would likely hit the same slow path (inference). |
| A2 `NCCL_SHM_USE_CUDA_MEMCPY=1` | untested, open | **Refuted** (graph timeout, eager +31 %). |
| A12 PCIe-IPC all-reduce (E14-I) | dead in install, P2P-conditional | Uses CUDA IPC peer loads/stores over the same fabric that NCCL P2P/IPC just measured at 0.41 GB/s busbw, so it is very likely slow here too, besides needing the FlashInfer ≥ 0.7 upgrade (inference). The same applies to A13 (one-shot fork) and A21-type P2P kernels. |
| F7 2×TP2 (E14-T) and A15 | nearly rejected (memory); needs `--enable-p2p-check` | V1 custom AR turns on at TP2 with the P2P check monkey-patched to True (FR§3.7), so a TP2 arm would push all-reduces through the slow P2P path unless custom AR is disabled (inference). |
| A3–A7 NCCL knobs (CTAS, ALGO, PROTO, NTHREADS, BUFFSIZE), A9 topo file | untested, low expected | Still untested (not in the facts given to me); they are now the only transport-side knobs left, with small published effects. |
| B1 vocab-parallel sampling (E14-L) | estimate −6–14 ms | The all-gather part is now measured at 8.0 ms on our box; with the 2.3–4.1 ms dense FP32 passes the removable total is ~10.3–12.1 ms ≈ +9–11 % at a ~125 ms step (my arithmetic; the new small gather of 20 values and ids per row is not subtracted). Its gate "profile shows AG ≥ 5 ms" (FR§d) is met by the microbenchmark. |
| F1 width 6/4 (E14-W) | model −10 % to +12 % | FR§1.2 says gains at our message size must come "by raising bandwidth (P2P, copy engine) or by reducing bytes per call (smaller width), not just latency". The two bandwidth routes are now refuted, so cutting bytes per call (width) and removing the logits gather are the remaining communication levers the docs name. With SHM latency near-linear in bytes (NCCL Q6), width 4 (491,520 B per call) would cost roughly 180–200 µs per all-reduce by the same α–β fit, i.e. roughly 20 ms less all-reduce per step plus ~4 ms less all-gather, before any acceptance loss (my estimate, not in the docs). |
| FR§3.1 path to 600–800 | "only credible path: all-reduce truly large and removable, plus logits fix, plus agent re-prefill fix" | "Large" is confirmed (~56 ms); "removable" via P2P or copy engines is refuted. The FR conclusion said the first two numbers (µs at 983,040 B; P2P on Kaggle) decide "whether 600–800 is reachable on TP4, or the ceiling is about 500–550"; by the FR's own framing these results point to the lower ceiling unless bytes per step are cut (inference). |
| I1 copy, I2 stall timeouts, I3 cache tar | designed | I1 confirmed (15× over single stream; ~38 s copy); I2's > 8 min phase-2 rule confirmed necessary (~410 s silent); I3's prize confirmed in size (~410 s compile/capture per cold start; same-session relaunch 71 s). |
| D1/D2 #44401 + graph sizes, C1 GDN sweep, G1–G4 agent levers | untested | Unchanged by these facts. |
