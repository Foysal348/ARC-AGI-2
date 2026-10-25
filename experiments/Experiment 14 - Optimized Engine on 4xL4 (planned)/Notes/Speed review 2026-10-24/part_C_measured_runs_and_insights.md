# Part C: speed facts from our measured runs and the 2026-10-20..23 conversation insights (2026-10-24)

**Scope.** This file only extracts speed facts. It does not design the experiment.

**Files read in full (every line):**
- A. `Notes/Conversation insights 2026-10-20 to 23/insights_00.md` … `insights_11.md` (12 files).
- B. `Experiment 11 …/Experiment Results/`: E11-03, E11-04, E11-05 and E11-06 analyses.
- C. `Experiment 12 …/Experiment Results/`: SMOKE1, EVAL30, EVAL6, agent weakness analysis, rev 5 runs, rev 6d G4B.
- D. `Experiment 13 …/PLAN.md`, `VALIDATION.md`.
- E. `Experiment 14 …/Notes/` 00, 05, 06, 07, 10, 11, 12, 13.

**Source shorthand used in the tables:**

| Short | File |
|---|---|
| E11-03 | `Experiment 11/Experiment Results/E11-03 run1 analysis (2026-10-17).md` |
| E11-04 | `… E11-04 run1 analysis (2026-10-18).md` |
| E11-05 | `… E11-05 run1 analysis (2026-10-18).md` |
| E11-06 | `… E11-06 P1-60K EVAL14 run1 analysis (2026-10-19).md` |
| E12-SM | `Experiment 12/Experiment Results/E12 SGLang SMOKE1 run1 analysis (2026-10-19).md` |
| E12-E30 | `… E12 SGLang EVAL30 run1 analysis (2026-10-19).md` |
| E12-E6 | `… E12 SGLang EVAL6 run1 analysis (2026-10-20).md` |
| E12-W | `… E12 agent weakness analysis (2026-10-20).md` |
| E12-R5 | `… E12 rev 5 runs analysis (2026-10-21).md` |
| E12-G4B | `… E12 rev 6d G4B GPU run analysis (2026-10-21).md` |
| E13-P / E13-V | `Experiment 13/PLAN.md` / `VALIDATION.md` |
| N00 … N13 | `Experiment 14/Notes/00 …` to `13 …` (Claude notes) |
| I00 … I11 | `Experiment 14/Notes/Conversation insights 2026-10-20 to 23/insights_00.md` … `insights_11.md` |

**Engine labels used below:**

| Label | Configuration |
|---|---|
| **A** | SGLang 0.5.21, TP4 on 4×L4, RedHat INT4 W4A16 (g128), fp8_e4m3 KV, no speculation, mamba strategy `extra_buffer` + overlap (5 state slots per request), auto `mem_fraction_static` 0.62 (E11-03) |
| **B** | A + MTP/NEXTN (≈4 tokens verified per request) (E11-03) |
| **C** | A + BF16 DFlash2 drafter, 8 draft tokens (width 8), SSM state fp32 (E11-03) |
| **D** | A + BF16 DFlash2, 8 draft tokens, SSM state fp16 (E11-03) |
| **W1** | SGLang 0.5.21 TP4, RedHat INT4 W4A16, fp8_e4m3 KV, `--speculative-algorithm DFLASH --speculative-draft-model-quantization unquant --speculative-num-draft-tokens 8`, fp16 SSM, `--mamba-radix-cache-strategy extra_buffer_lazy`, `--max-mamba-cache-size 56`, prefill CUDA graph off, `--mem-fraction-static 0.83`; E12 adds `--incremental-streaming-output --stream-interval 1`. G4B log also shows `--chunked-prefill-size 4096`, `--schedule-policy lpm`, `--max-running-requests 13`. Server max-running 12 in E11-04/E11-05; 13 (the "11-of-12 workaround", client 12) in E11-06 and E12. Decode CUDA graphs `[1..8, 10, 12]` (I03 §2, I06 §2, I11 §2) |
| **W2** | W1 with bf16 SSM (E11-04) |
| **V** | vLLM, Qwen3.8-27B FP8 + MTP-3 (the S1-60K / R5 / D66 production line) |
| **FN** | Qwen3.8-Flash-Next (Intel W4A16 AutoRound + BF16 PLE), SGLang 0.5.21 TP4 + EP4 (E13 plan; never run in E13) |

Labels: **[M]** measured; **[D]** derived from logs (not profiled); **[E]** estimate; **[C]** claimed (external or unverified).

---

## 1. Every measured speed number from our runs

### 1.1 E11-03: SGLang engine sweep (kernel `foysalemonshanto/e11-03-sglang-int4-gpu-validation` v1, sha e47d1660, 2026-10-17, 203 min, 0 failed requests)

| # | Engine | Concurrency requested → real | Context | Metric | Value | Source |
|---|---|---|---|---|---|---|
| 1 | A | sweep levels up to 32 → **22** | short | real max running (set by mamba cache) | `max_mamba_cache_size` 113 → 22 running; decode graph bs up to 22 | E11-03 §1 |
| 2 | B | → **10** | short | real max running | mamba cache 50 → 10 | E11-03 §1 |
| 3 | C | 12/16/24/32 → **6** | short | real max running | mamba cache 33 → 6; verify graph bs 1–6 | E11-03 §1 |
| 4 | D | → **13** | short | real max running | mamba cache 68 → 13; verify graph bs 1–13 | E11-03 §1 |
| 5 | all | — | — | `/get_server_info` max_running_requests | 32 (wrong; the cap is visible only in a log warning) | E11-03 §6 |
| 6 | all | — | — | auto `mem_fraction_static`; GPU memory left free after all captures | 0.62; **4.39 GB per GPU (≈19 %) free** | E11-03 §1 |
| 7 | A / B / C / D | — | — | KV pool tokens | 604k / 474k / 395k / 387k (vLLM FP8 for comparison: 1.35M) | E11-03 §1 |
| 8 | D | — | — | per-GPU memory | target weights 4.62, drafter 1.49, mamba 3.26 (ssm 1.21 + intermediate ssm 1.97 + conv 0.08), target KV 2.94, draft KV 0.92, prefill graph 2.60, target verify graph 0.50, draft verify graph 0.38, **free 4.39 GB** | E11-03 §1, §6 |
| 9 | all | — | — | NCCL preflight all-reduce, 64 MB (MiB), fp32, eager torch.distributed, 10 iterations | **20.9 ms** (20.89 / 20.89 / 20.85 / 20.86) = 3.21 GB/s algbw, 4.82 GB/s busbw. Custom all-reduce disabled ("not supported on more than two PCIe-only GPUs") → TP4 uses NCCL | E11-03 §1; I10 §2; I11 §2 |
| 10 | A | real 1 / 4 / 8 / 12 / 16 | short prompts ≈4.3k | total tok/s | 45.6 / 145 / 229 / 289 / 347 | E11-03 §2 |
| 11 | B | real 1 / 4 / 8, then capped at 10 | ≈4.3k | total tok/s | 55.1 / 205 / 297; 12 and 16 ran as 10 + queue (not measured) | E11-03 §2 |
| 12 | C | real 1 / 4, then capped at 6 | ≈4.3k | total tok/s | 93.7 / 250; 8, 12 and 16 ran as 6 + queue | E11-03 §2 |
| 13 | D | real 1 / 4 / 8 / 12 / 16→13 | ≈4.3k | total tok/s | **107.5 / 278 / 356 / 404 / 381 (at 13 real)** | E11-03 §2 |
| 14 | V (earlier runs) | 12 / 16 | — | total tok/s | ~307 / ~373 | E11-03 §2 |
| 15 | D vs A, D vs V | real 12 | short | ratio | D = **1.39× A**, ≈1.3× vLLM; "the only valid 12-slot spec number" | E11-03 §2, §4 |
| 16 | A / B / C | 12 × 24,576 requested (capped by mamba) | long | total tok/s | A 282, B 262, C 258. Unfair: A had 56,742 of 57,122 prompt tokens cached, B and C ≈1k (cold) | E11-03 §2, §6 |
| 17 | C | 4 × 85.5k | 85.5k | server decode median; KV usage | **≈238 tok/s at 4 running**; KV usage 0.87; only ≈4 requests of 85k fit the 395k pool | E11-03 §3 |
| 18 | C | 4 running | 85.5k | cold prefill | ≈1,400 tok/s | E11-03 §3 |
| 19 | C | capped at 6 | — | temperature arm T 0.6 vs T 1.0 | T 0.6 was 0.85× (not faster); inconclusive | E11-03 §3 |
| 20 | C | — | — | contcache: cached ratio, warm vs cold | cached ≥ 0.979; warm 11× faster than cold | E11-03 §3 |
| 21 | all | — | — | launch time | first launch **601 s (prefill-graph JIT 447 s)**; later launches 136–166 s | E11-03 §3 |
| 22 | B vs A, C vs A, D vs C | — | 512 tokens | greedy common prefix | 152 / 155 / 166 tokens; batched vs alone 12–22, 0/4 identical | E11-03 §3 |
| 23 | B | — | — | accept length | MTP ≈3.1 | I01 §2 |
| 24 | C/D | — | — | accept length | DFlash2 (block 8) 4.5–4.95; DFlash agent: median 4.62 at c12 | I01 §2; I10 §2 |
| 25 | A | 1 / 12 / 22 | short | decode step time [D] | 21.9 ms / 37.9 ms / 50.0 ms; marginal ≈1.3 ms per extra request | N00 §1; I09 §2 |
| 26 | B | 1 / 10 | short | decode step time [D] | 40.3 ms / 79.9 ms; marginal ≈4.4 ms | N00 §1 |
| 27 | D | 1 / 12 | short (<10k, n=67) | decode step time [D] | 39.9 ms / **115.5 ms**; +75.6 ms from 1→12 = ≈6.87 ms per request, 0.86 ms per verified token | N00 §1; I10 §2; I11 §2 |
| 28 | D | 12 | 0 → 50k | added step time [D] | ≈+18 ms per step (KV-read arithmetic [E]: 12 × 50k × 8 KB ≈ 4.8 GB ÷ 300 GB/s ≈ 16 ms) | N00 §1 |
| 29 | — | — | — | torch BF16 4096³ GEMM on L4 | 75.9 TFLOPS | I10 §2; I11 §2 |
| 30 | A vs D (re-derived by the research agents) | c12 | short | ratios and step | target-only 289 tok/s; DFlash2 W8 fp16 SSM 404 tok/s ≈ **137 ms per step**; MTP 0.91×, DFlash2 W8 fp32 SSM 1.06×, fp16 SSM 1.39× of target-only | I10 §2 |
| 31 | A | 22 | short | total tok/s | 440 (INT4, no speculation, at 22) | I09 §2 (research table) |

### 1.2 E11-04: memory fix and PROD at real 12 (rev 2b, esfersami50, 2026-10-18, 87 min, 0 errors)

| # | Engine | Concurrency | Context | Metric | Value | Source |
|---|---|---|---|---|---|---|
| 32 | W1 (lazy + mamba 56 + no prefill graph + mem 0.83, max running 12) | 12 → **12 real** | — | real running; KV pool; free memory | 12 real running; **KV 881,336 tokens** (E11-03 D: 387k); 2.2 GB (2.22) free; 98 % of the 900k gate | E11-04 "What the run proved"; I01 §2 |
| 33 | W1 with fp32 SSM | — | — | KV pool | only 599k | I01 §2 |
| 34 | W1 | PROD closed loop, ARC prompts, T 1.0, running share 1.0 at 12 | mean context <10k / 10–20k / 20–40k | tok/s (whole PROD) and log rate by context | **416.5**; 458 / 413 / 388 | E11-04 table |
| 35 | W2 | same | same | tok/s; by context | 412.1; 466 / 402 / 375 | E11-04 table |
| 36 | W1 | PROD shortened to 10 min (KV gate missed by 2 %) | contexts reached only 20–32k | — | ≥40k not proven; trend suggests ≈350–370 at 40–57k [E, "hypothesis"] | E11-04 |
| 37 | W1 | sweep at 12 with short 4,096-token requests and staggered prefill | short | running max | **11** | E11-04 "Sweep at 12" |
| 38 | W1 | — | — | launch time | first 567 s (JIT); later ≈75 s | E11-04 header |
| 39 | W1 | — | — | target-verify graph capture with a warm cache in the same session | 11.94–13.67 s ("relaunch in the same session 11.9–13.7 s") | I10 §2; I11 §2 |
| 40 | W1 | — | short | forced-close continuation | cached ratio ≥ 0.965, 10× faster than cold | E11-04 "Prefix cache" |
| 41 | W1 | — | long prefix | continuation cache | 0 cached (both samples were aborted PROD requests; hypothesis: aborted requests are not inserted) | E11-04 |
| 42 | — | — | — | NCCL 64 MiB all-reduce; BF16 GEMM | 20.76–20.83 ms; 78.8 TFLOPS | I10 §2; I11 §2 |
| 43 | W2 vs W1; W3 vs W1 | — | — | greedy drift (common prefix) | 41.5; 44.0 | E11-04 |
| 44 | W1 drafter flag | — | — | acceptance without `unquant` | "mandatory, else acceptance ~1.0" [C: stated in notes; origin run not in these files] | I01 §2, §6 |

### 1.3 E11-05: Poetiq S49 on W1 (esfersami50 EVAL30 v1, sha a17223f7, 4 h; foysal S1COMMIT4 v1, sha 12013d1a; 2026-10-18)

| # | Engine | Concurrency | Context | Metric | Value | Source |
|---|---|---|---|---|---|---|
| 45 | W1 (server max-running 12) | — | — | engine ready | 532 s / 522 s; server_args → "fired up" 8:00 (7 m 39 s per GPT); weight load 13 s; CUDA graphs 383 s | E11-05 "Runs"; I09 §2; I10 §2; I11 §2 |
| 46 | W1 | — | — | capacity gate | KV 881,336; mamba cache 56; verify graph up to 12; 2.22 GB free | E11-05 |
| 47 | W1 | 12 set → **11 really running** | whole run | tok/s over the whole solver | **328.9** (4,253,591 tokens / 12,932.4 s) | E11-05 table; I11 §2 |
| 48 | W1 | 11 running | <10k / 10–20k / 20–40k / 40–60k | decode tok/s | **429 / 372 / 340 / 323** | E11-05 "Decode by context" |
| 49 | W1 | 11 running | — | accept length | median 4.55 (per request); decode-line median 3.8; per-run mean 3.87; at 12 running 3.9130 (n=114) | E11-05; I09 §2; I10 §2; I11 §2 |
| 50 | W1 | 12 in flight | — | running distribution | 2,358 decode lines at 11 running + 1 queued, only 114 at 12; scan: 2,816 decode lines, 2,615 with queue ≥ 1, mostly at running 11; the 12th waited up to 24 min | E11-05; I01 §2; I06 §2 |
| 51 | W1 | 11 vs 12 | — | throughput cost of 11-of-12 | ≈8 % | E11-05; I01 §2 |
| 52 | W1 | 12 | all | median decode step | 128.4 ms (n=114) | I09 §2; I10 §2; I11 §2 |
| 53 | W1 | — | — | KV usage p50 / p90 / max; mamba max | 0.44 / 0.54 / 0.63 (GPT: peak 0.65); mamba 0.70; 0 lines ≥ 0.95 | I06 §2 |
| 54 | W1 | 12 | — | total resident context at 12 running | 449k | I10 §2 |
| 55 | W1 | — | — | retractions | "No retractions" (E11-05) vs 1 retract line (I06 scan) | E11-05; I06 §2 |
| 56 | W1 | drain | 11–30k each | phase-2 prefix cache after waiting for aborted streams | 52–99.7 % (6 of 8 above 99 %); all 8 answers in 38–83 s | E11-05 "Drain" |
| 57 | W1 | — | — | prefix cache by request type | forced-close phase 2 98.1 % (4.77M prompt tokens); phase 1 iteration ≥ 2 10.6 % (Poetiq shuffles example order); phase 1 iteration 1 18.4 % | E11-05 "Prefix cache" |
| 58 | W1 | — | — | forced closes | 83 of 88 iterations (49,152 cap) | E11-05 |

### 1.4 E11-06: P1-60K EVAL14 (W1 rev 3c, esfersami50, 2026-10-19, 2 h)

| # | Engine | Concurrency | Context | Metric | Value | Source |
|---|---|---|---|---|---|---|
| 59 | W1 (server max-running 13) | max 12 active | whole run | throughput | **379.9 tok/s** (2,262,599 tokens / 5,956.2 s); 87.2 % of prompt tokens cached; 0 errors | E11-06; I11 §2 |
| 60 | W1 | — | — | startup | server_args → "fired up" 7:52 (7 m 30 s per GPT); weight load 13 s; CUDA graphs 382 s | I09 §2; I10 §2 |
| 61 | W1 | 12 | all / <10k / 50–60k | decode step | median **124.6 ms** (p10–p90 116.2–132.5, n=278); 115.3 ms at <10k; 133.3 ms at 50–60k | I10 §2; I11 §2 |
| 62 | W1 | 12 | windows containing a prefill line | step | median **323 ms** | I11 §2 |
| 63 | W1 | 12 | <10k / 10–20k / 20–40k / 40–60k | decode-line tok/s | **500.1 / 460.3 / 409.0 / 389.4** | I11 §2; I09 §2 ("500 / 460 / 409 / 389") |
| 64 | W1 | 8 / 12 | — | accept length mean | 4.4628 (n=408) / 4.3708 (n=278); per-run mean 4.58 | I10 §2; I11 §2 |
| 65 | W1 | — | — | KV p50 / p90 / max; mamba max | 0.40 / 0.50 / 0.69; 0.66; 1,438 decode lines, 128 with queue ≥ 1 (107 at running 11) | I06 §2 |
| 66 | W1 | 11 running | — | mamba slots used | 33 of 56 | I11 §2 |
| 67 | W1 | 12 | — | total resident context at 12 running | 357k | I10 §2 |
| 68 | W1 | — | — | CUDA graphs | resolved decode list `[1..8, 10, 12]`, max_bs 12, while `max_running_requests` = 13; graph memory: target-verify 0.68 GB, draft 0.35 GB | I11 §2 |
| 69 | W1 | — | — | forced closes at the 60k cap | 39/44 iterations (89 %); natural closes median 35k tokens | E11-06 |

### 1.5 W1 memory and startup, cross-run (insights; run not always named)

| # | Engine | Metric | Value | Source |
|---|---|---|---|---|
| 70 | W1 at mrr 13 (KV 866,744) | per-GPU memory | weights: target 4.62, drafter 1.49; mamba: conv 0.04, ssm 1.00, intermediate 1.97 GB (1.83 in E11-05), conv window 0.03; KV: target K/V 3.31 + 3.31, drafter K/V 1.03 + 1.03 GB; free 3.82 GB after pools → 2.61 after target graphs (0.68) → 2.26 after draft graphs (0.35) → **final 1.88 GB** | I10 §2 |
| 71 | W1 | KV formula vs measured | SGLang formula predicts 886,801 vs 881,336 measured | I10 §2 |
| 72 | W1 | KV at server max 13 vs 12; free per GPU | 866,744 (mrr 13) vs 881,336; 1.84–2.2 GB free per GPU | I09 §2 (research) |
| 73 | W1 | cold start | 520–680 s, of which 372–385 s is the verify CUDA-graph JIT; capture 382–394 s with ≈362 s on the first bs=12 bucket ("bs=12 … 1/10 [06:02<…, 362.28s/it]"), silent for ≈6 min; earlier figure for target-verify capture 372.5 s; range 371–385 s | I09 §2; I10 §2, §6; I11 §2 |
| 74 | W1 | server_args → "fired up" | E11-05 8:00, E11-06 7:52, E12 8:05, G4B rev 6d 19:09 | I11 §2 |
| 75 | W1 | per-stream decode | 31–35 tok/s per stream | I09 §2 (research) |
| 76 | W1 | prefill burst speed (chunked prefill) | 1,474–1,578 tok/s across runs (EVAL30 1,474; G4B 1,578); "1,475–1,578"; cold prefill "1,400–1,650 tok/s (E11)", i.e. ≈30–35 s per 50k resume | I10 §2; I11 §2; I02 §2 |
| 77 | W1 | cost of one uncached prompt token | ≈0.27 of a generated token (Franzen's RTX PRO 6000: ≈0.08) | I10 §2; N11 §B |
| 78 | W1 | accept length overall | DFlash2 4.3–5.2; "Poetiq production 3.8 by server-log median vs 4.55 per request"; falls 4.43 → 3.77 as context grows; T 1.0 lowers acceptance (DFlash 4.60 → 4.34); "accepted tokens per step 4.5–5.5" | I09 §2; I01 §6; I00 §2 |
| 79 | W1 | accept identity | accept len = 1 + 7 × accept rate (bonus token included); max deviation 0.0057 over 5,017 records (e.g. 4.07 with rate 0.44) | N05 §A2; I10 §2; I11 §2 |
| 80 | W1 | decode slowdown with context | ≈25 % slower at 40–60k (323 vs 429 tok/s, E11-05) | I01 §6; I02 §2 |
| 81 | W1 (E12 runs) | collectives | "CustomAllReduceV2 is disabled …" and "CustomAllreduce is disabled …" on all 4 ranks in E11-05, E11-06, E12 EVAL30 and G4B; no `NCCL INFO` line in any log, so the transport (P2P or SHM) is unknown | I10 §2; I11 §2 |

### 1.6 E12 SMOKE1 (foysalemonshanto, sha 0fea4d0c, 2026-10-19)

| # | Engine | Concurrency | Context | Metric | Value | Source |
|---|---|---|---|---|---|---|
| 82 | W1 | 2 running | ≤13.6k | engine up; KV | 677 s; KV 866,744 tokens | E12-SM "Infrastructure" |
| 83 | W1 | 2 running | ≤13.6k | decode; DFlash acceptance median | **204 tok/s**; 5.2 | E12-SM |
| 84 | W1 | 2 branches | 4,096 → 13,376 | `cached_tokens` rising | b0 4,096 → 6,867 → 10,515 → 11,776 → 11,968; b1 none → 6,867 → 13,056 → 13,120 → 13,376; overall 80.8 %, up to 99 % per request | E12-SM |
| 85 | W1 | — | — | branch wall time | b0 2 turns, 4,905 generated tokens, 60 s; b1 2 turns, 5,907 tokens, 73 s | E12-SM |

### 1.7 E12 EVAL30 (rev 1, `hosen42/…-eval30` v1, sha b65fb9dc, 2026-10-19; 12 streams, all 30 tasks' branch 0 time-sliced, output cap 12,288)

| # | Engine | Concurrency | Context | Metric | Value | Source |
|---|---|---|---|---|---|---|
| 86 | W1 | 12 streams; "30 time-sliced branches" / "30 active" / "all 60 branches admitted" | prompt p50 47.9k / p75 74,934 / p90 97.7k / max 117.3k (n=454) | prompt-size distribution | as stated | E12-E30 §2; E13-P §3b; I00 §2; I02 §3 |
| 87 | W1 | — | — | requests; prompt tokens; cached | 454 requests, 0 failures; 23.78M prompt, 14.18M cached (59.6 %) | E12-E30 |
| 88 | W1 | — | — | evictions / re-prefill | 144 of 423 non-first requests < 50 % cached (cached = None / 192 / 576); they account for **8.39M of the 9.60M** uncached prefill tokens; "≈35 % of all prompt tokens prefilled again" | E12-E30 §2 |
| 89 | W1 | — | solve window | generation rate | **≈187 tok/s** (2.36M tokens / 12.6 ks) | E12-E30 §2 |
| 90 | W1 | 11–12 running | "even at ≥ 60k" | decode rate | ≈420 tok/s (decode summary); "decode ~412–431" | E12-E30 §2; N05 §B1 |
| 91 | W1 | 12 / 6 / 5 running | — | decode tok/s | **431 / 368 / ≈360** | I00 §2 |
| 92 | W1 | — | window 14,046 s | effective tok/s; prefill share; uncached prompt ÷ generated; decode-only | **167.8** (2,357,584 generated); **55.1 %**; **4.07**; ≈376 | I10 §2; I11 §2 |
| 93 | W1 | — | — | prefill lines | 2,944 lines; 11,232,209 new tokens; 15,348,388 cached; median input throughput for chunks ≥ 2,000 **1,459 tok/s** (p25/p75 1,342/1,551); log window 14,054 s; ≈13,372 prefill-seconds | I00 §2 |
| 94 | W1 | — | — | model time vs real | 9.59M prefill + 2.35M generated = 12,018 s model time vs 12,634 s real; with no thrash (0.39M prefill) 5,706 s (**2.1× faster**) | I00 §2 |
| 95 | W1 | — | — | request timeouts | **40** in 21 of 31 branches (28 full 900 s + 12 deadline-bounded; 31,247 s ≈ 31,000 stream-seconds waited) | E12-W W6; I00 §2 |
| 96 | W1 | — | — | output cap hits | 122 of 454 requests (27 %) ended at 12,288; completion median 2,983; 29/30 first requests hit the cap | E12-E30 §3; I00 §2 |
| 97 | W1 | at 232 min | — | log line | `num_running_reqs` 10, `num_queue_reqs` 1, `gen_throughput` 380.2 | I02 §2 |
| 98 | W1 | at 14,247 s | — | log line (stall) | `num_running_reqs` 11, `num_queue_reqs` 8, **`gen_throughput` 0.0** (unexplained; "likely KV thrash") | I02 §2; I03 §2 |
| 99 | W1 | — | — | KV usage p50 / p90 / max; lines ≥ 0.95; mamba max | 0.84 / 0.93 / 0.99, 167 lines ≥ 0.95, mamba 0.71; 4,153 decode lines, 3,127 with queue ≥ 1 (I06). Other agent: 0.77 / 0.95 / 1.0 (I10) | I06 §2; I10 §2 |
| 100 | W1 | — | — | time below 12 running; "11 running + 1 queued" | below 12 running 67.8 % of the time; 11 + 1 queued 50 % of the time, costing ≈9 % | I10 §2; I11 §2 |
| 101 | W1 | 11 vs 12 running | same context | tok/s | 422 vs 462 (≈−8–9 %) | I11 §2 |
| 102 | W1 | 12 running | — | mamba slots used | 36 of 56 | I10 §2; I11 §2 |
| 103 | W1 | 12 | all / <10k / 60–80k | decode step | 137.9 ms (gen > 100 tok/s filter), 139.4 ms (all records), 138.4 ms (gen > 50); range "135.7–139.4"; 115.4 ms at <10k; 140.2 ms at 60–80k | I10 §2; I11 §2 |
| 104 | W1 | — | — | accept length per-run mean | 5.10 | I10 §2; I11 §2 |
| 105 | W1 | — | — | retract lines | 5 | I06 §2 |
| 106 | W1 | — | — | reply re-prefill share (hit stops at end of previous prompt) | 59.1 % of requests | I11 §2 |
| 107 | W1 | — | — | deadline overrun | solve 733 s → 14,779 s; deadline_in_sec 13,367 is relative to solve start → absolute ≈14,100 s; overrun **679 s** (first reported as 1,412 s); requests with read timeout 419.6 s at 14,436 s and 469.7 s at 14,492 s | E12-E30 "Run"; I02 §2; I03 §2 |
| 108 | W1 | — | — | quota burn | 4.15 h wall → 8.22 h quota (≈2.0×) | E13-P §3 |
| 109 | W1 | per started branch | — | cost and coverage | ≈76k generated tokens per started branch; 30 tasks took the whole 3.5 h; 240 tasks ≈28 h at this rate | E12-E30 §5 |

### 1.8 E12 EVAL6 rev 4 (`hosen42/…-eval6` v1, sha 53a66856, 2026-10-20; 60,000 output per request, 100k soft cap, ≤ 6 live)

| # | Engine | Concurrency | Context | Metric | Value | Source |
|---|---|---|---|---|---|---|
| 110 | W1 | ≤ 6 live; "6 running for most of the run" (max 6) | prompt p50 38,745 / p90 63,699 / max 69,650; context p50 48,515 / p90 69,440 / max 128,399 | — | as stated | E12-E6; I00 §2; I08 §2 |
| 111 | W1 | 6 | — | generated; requests; cache | 1,217,904 tokens in 96 requests; prompt cache 78.8 %; uncached prefill 0.78M | E12-E6; I00 §2 |
| 112 | W1 | 6 | solve 3,744 s (62 min) of a 90-min window; started_sec 811, finished_sec 4,555 | effective tok/s; prefill share; uncached ÷ generated; decode-only | **325** (325.3; 1.22M / 3,744 s); 13.3 %; 0.64; ≈377 | I00 §2; I10 §2; I11 §2 |
| 113 | W1 | — | — | KV p50 / p90 / max; mamba max | 0.24 / 0.39 / 0.46; 0.36 | I08 §2 |
| 114 | W1 | — | — | completions by finish reason | tool_calls n 87: p50 2,273 / p75 12,888 / p90 31,645 / p95 35,608 / max 58,224; length n 7 at 60,000; stop n 2 at 13,109 | I00 §2 |
| 115 | W1 | — | — | GPU time per task | ≈10 GPU-min per task (12 branches in 62 min with 6 live); 240 tasks ≈40 h vs ≈11 h of solving | E12-E6 "Cost"; E12-W W7 |
| 116 | W1 | — | — | previous reply not carried into next prompt | 11 of 84 later requests; **426,781 of 1,217,904 generated tokens (35 %)** thrown out (EVAL30: 12 of 423, 98,222 of 2,357,584 = 4 %) | E12-W W2; I00 §2 |
| 117 | W1 | — | — | reply re-prefill share | 84.5 % of requests | I11 §2 |
| 118 | W1 | — | — | cap overshoot | 100k soft cap overshoots to 159k (max branch 159,464) | E12-W W7; I03 §2 |

### 1.9 E12 rev 5 runs (2026-10-20/21): EVAL6R5 xhigh (`hosen42`, sha f4fed1ac) and EVAL12R5_LOW low (`foysalemonshanto`, sha 26fff41e); both 6 streams, 49,152 output per request, T 1.0

| # | Run | Concurrency | Context | Metric | Value | Source |
|---|---|---|---|---|---|---|
| 119 | EVAL6R5 | 6 | prompt p50 51,917 / p90 71,504 / max 78,158; context p50 58,121 / p90 74,194 / max 95,121 | — | as stated | I00 §2; E12-R5 |
| 120 | EVAL6R5 | 6 | window 3,464 s (solve 58 min; notebook 69 min; started_sec 702, finished_sec 4,166) | generated; requests; effective tok/s | 846,080; 127 requests; **244** (244.2) | E12-R5; I01 §2; I11 §2 |
| 121 | EVAL6R5 | 6 | — | prefill share; uncached ÷ generated; decode-only | 18.6 %; 1.15; ≈302 | I10 §2; I11 §2 |
| 122 | EVAL6R5 | — | — | cache | aggregate **84.26 %** (5,195,178 cached of 6,165,603 prompt); request-level 85.4 % (used by mistake); uncached prefill 0.97M vs generated 0.85M; request-level uncached 888,297 | I08 §2; I06 §2; I01 §2; E12-R5 |
| 123 | EVAL6R5 | — | — | KV p50 / p90 / max | 0.24 / 0.36 / 0.43 | I08 §2 |
| 124 | EVAL6R5 | — | — | per task | 141k tokens, 9.7 min | E12-R5 |
| 125 | EVAL6R5 | — | — | null-cached (= 0-hit) requests | 6 of 127 | I06 §2 |
| 126 | EVAL6R5 | — | — | reply re-prefill share | 91.3 % of requests | I11 §2 |
| 127 | EVAL6R5 | — | — | request size | median completion 1,115; median prompt 51.9k; the 28 of 127 requests ≥ 10k carry 80 % of tokens | I01 §2 |
| 128 | EVAL12R5_LOW | 6 | prompt p50 41,629 / p90 72,552 / max 81,536; context p50 45,708 / p90 75,190 / max 114,787 | — | as stated | I00 §2 |
| 129 | EVAL12R5_LOW | 6 | window 6,570 s (solve 110 min; notebook 123 min, or "COMPLETE after 163 min"; started_sec 799, deadline_in_sec 9,000) | generated; requests; effective tok/s | 1,743,889; 449 requests; **265** (265.4) | E12-R5; I01 §2; I03 §2; I00 §2 |
| 130 | EVAL12R5_LOW | 6 | — | prefill share; uncached ÷ generated; decode-only | 20.4 %; 1.17; ≈335 | I10 §2; I11 §2 |
| 131 | EVAL12R5_LOW | — | — | cache | aggregate **88.70 %** (15,951,513 of 17,983,349); request-level 91.0 %; request-level uncached 1,576,650 | I08 §2; I01 §2 |
| 132 | EVAL12R5_LOW | — | — | KV p50 / p90 / max; mamba max | 0.28 / 0.36 / 0.47 (I06) or 0.28 / 0.35 / 0.47 (I08); 0.36 | I06 §2; I08 §2 |
| 133 | EVAL12R5_LOW | — | — | per task; requests per branch | 145k tokens, 9.1 min; 18.7 requests/branch vs 10.6 (xhigh); 72.7k vs 70.5k tokens per branch | E12-R5; I02 §2 |
| 134 | EVAL12R5_LOW | — | — | null-cached requests | 23 of 449 (≈5 %) | I06 §2 |
| 135 | EVAL12R5_LOW | — | — | reply re-prefill share; 64-multiple stops | 72.5 %; 413 of 414 hits stop on a 64-token multiple | I11 §2 |
| 136 | rev 5 (both) | — | — | small-run tail | 130 decode lines with 1 running | E12-R5 "Coverage arithmetic" |
| 137 | rev 5 vs rev 4 | — | — | effective tok/s | 244 (xhigh) and 265 (low) vs 325 in rev 4; "shorter replies mean more requests and more prefill" | E12-R5 |
| 138 | rev 4 (EVAL30 decode) | 6 → 12 running | — | decode | 368 → 431 tok/s (+17 %; the short E11 sweep showed +30–60 %) | E12-R5 "Next levers"; I02 §2 |
| 139 | rev 5 | — | — | B3 prompt estimate vs server `prompt_tokens` | median +0.4 % / +0.5 % (p10 −3.8 % … p90 +3.5 %) | E12-R5 |
| 140 | EVAL12R5_LOW submission | 6 live, low, 49k, 2 branches | hidden 240 | LB / reach | public **17.50** (ref 57456562); ≈74 of 240 tasks reached (31 %) | I09 §2 |

### 1.10 E12 rev 6d G4B (`foysalemonshanto/…-eval12r6g4b` v1, sha 1bdadd82, 2026-10-21; 12 workers, gate 4b time share)

| # | Engine | Concurrency | Context | Metric | Value | Source |
|---|---|---|---|---|---|---|
| 141 | W1 (mrr 13) | — | — | wall; ready; solve start; window | wall 3,528 s (58.8 min); **"ready in 1197s"** (rev 5 ≈700 s); solve start 1,463 s; window 2,063 s = **171.9 s per task**; finished_sec 3,517 | E12-G4B; I06 §2; I08 §2 |
| 142 | W1 | — | — | startup split | "fired up" 19:09; weight load **670 s** (669.96 / 669.26 s); 18.6 GB shard 657 s (≈28 MB/s); CUDA graphs 394 s; install 208.2 s; DFlash2 file 0.56 s (page cache); profile time_budget 4,800 s, give_up_after 3,780 s; 1,197 s start cost ≈40 quota-min | I10 §2; I11 §2 |
| 143 | W1 | 12 workers | per stream 72,228 window (= 866,744 / 12) | generated; per task; effective | 572,035 tokens = 47.7k per task; **277 tok/s** (572,035 / 2,063 = 277.3); other window definition: 279.0 over 2,050 s | E12-G4B; I08 §2; I10 §2; I11 §2 |
| 144 | W1 | — | — | prefill share; uncached ÷ generated; decode-only | 19.8 %; 1.17; ≈353 | I10 §2; I11 §2 |
| 145 | W1 | — | — | requests; caps | 180 requests / 175 tool calls; caps 12,288 × 20 and 24,576 × 160; 32,768 never used | E12-G4B; I06 §2 |
| 146 | W1 | — | — | cache | aggregate **75.0 %** (2,008,877 of 2,678,857); uncached prompt **669,980 > 572,035 generated**; 20 of 180 requests `cached = None` (= 0) carrying 254,890 prompt tokens (11 %); 15 of 180 later requests < 5 % cached in 9 branches (trims in 2 branches only) | I06 §2; I07 §2 |
| 147 | W1 | — | — | real running max; acceptance | 12; accept 4.2 | E12-G4B "What this run shows" |
| 148 | W1 | 11 running | 20–40k / <10k | tok/s | **370.9** (range 370–406); **406.3** at <10k | E12-G4B F1; I07 §2 |
| 149 | W1 | 6 running | 20–40k | tok/s | **321.4** → step-down cost **13.3 %** (first written 15–20 %; GPT 13–16 %) | E12-G4B F1; I07 §2 |
| 150 | W1 | exactly 12 running | first 13 min | tok/s | 29 decode lines, all 11:59–12:12, alternating 42–250 and 382–497 (visible: 139.74, 167.76, 466.1, 497.19); median 194–258 (claim "12 is slower" withdrawn: admission prefill) | E12-G4B F1; I06 §2 |
| 151 | W1 | capacity probe | — | step-down events | t 240 s → 10 (queue 1, KV 0.13); 600 s → 8 (KV 0.26); 780 s → 6 (KV 0.31); retracted = None | E12-G4B F1; I06 §2 |
| 152 | W1 | 12 in flight | — | queue lines | 188 decode lines with `#queue-req: 1`, 101 of them at running 11; KV 0.17–0.31 and mamba ≤ 0.70 in those lines; scan: 710 decode lines, 194 with queue ≥ 1 | E12-G4B F1; I06 §2 |
| 153 | W1 | — | — | KV p50 / p90 / max; mamba max; retracts | 0.20 / 0.35 / 0.44; 0.70; 1 retract line | I06 §2 |
| 154 | W1 | — | — | time below 11 running; at exactly 6; 11 + 1 queued | 47 %; **27.7 %**; 25 % | I10 §2; I11 §2 |
| 155 | W1 | 11 vs 12 | same context | tok/s | 397 vs 436 (≈−8–9 %) | I11 §2 |
| 156 | W1 | per stream | — | stage-1 rate | stage-1 runs 1,120–1,510 s each for 30–44k tokens ≈ 25–35 tok/s per stream (incl. tool calls and prefill); "11 effective streams × ≈34 tok/s ≈ 370 tok/s" | E12-G4B F3 |
| 157 | W1 | — | — | soft-pause overshoot | past 28k: p50 5.1k (5,107), max 15.8k (15,784) | E12-G4B F3; I06 §2 |
| 158 | W1 | — | — | logged flags; #36935 ratio | `--chunked-prefill-size 4096`, `--max-mamba-cache-size 56`, `extra_buffer_lazy`, `--max-running-requests 13`, `--mem-fraction-static 0.83`, `--schedule-policy lpm`; kv_pool / chunk = 866,744 / 4,096 = **211.6 vs mamba cache 56** | I06 §2 |
| 159 | W1 | — | — | example decode line (startup phase) | `#running-req: 11, #full token: 17257, full token usage: 0.02, mamba num: 35, mamba usage: 0.62, accept len: 4.61, accept rate: 0.52, cuda graph: True, gen throughput (token/s): 21.31, #queue-req: 1` | I08 §2 |
| 160 | W1 | 150 requests with a predecessor | — | cache accounting from server `usage` | reply_lost_at_prefill_end 68 (45 %) / **217,541** tokens; reply_reused 49 (33 %) / 3,446; deeper_miss 18 (12 %) / 66,771; partial_reply 14 (9 %) / 2,442; history_shrank 1 / 23,099; total **313,299** re-prefilled = 79 % of 397,650 uncached (of 2,392,575 prompt tokens) | N11 §B; I10 §2 |
| 161 | W1 | — | — | where lost replies stop | cached − prompt_prev = −63 … +2 (median −34), even for 7,088-token replies; reused example 3,621 → 3,525; reasoning chars median reused 116 (41/49 ≤ 300) vs lost 1,712 (12/68 ≤ 300), deeper_miss 2,046; reply length median 98 tokens (reused) vs 830 (lost) | N11 §B; I10 §2 |
| 162 | W1 | — | — | uncached split (agent) | 670k = ≈37 % re-prefilled replies (245k), 47 % zero/partial hits (317k: 14 zero + 15 partial), 5 % first requests (34k), 11 % tool-result deltas (74k); reply re-prefill share 75.6 %; 141 of 150 hits on a 64-multiple | I11 §2; I10 §2 |
| 163 | W1 | — | — | deadline mechanics | 91 deadline clamps; 0 stragglers; finished 9 s before the deadline; verify median 0.86 s | I06 §2 |
| 164 | W1 | — | — | B3 estimate vs actual prompt tokens | ratio median 0.998 | I06 §2 |
| 165 | W1 | — | — | fair share vs minimum chunk | shares [7,544, 7,266, 7,332, 6,044, 5,510, 3,758] vs minimum chunk 24,576 (inert) | I07 §2 |
| 166 | W1 | branch 1 | — | b1 cost | 123,560 tokens (21.6 %), 0 verified candidates; all 4 b1 first requests cut at 12,288 before any call (first call at 12.4–18.0k) | I06 §2 |

### 1.11 E12 rev 6g G4B (`esfersami50/…-eval12r6g4b` v1, sha 5d2bc39f, 2026-10-22)

| # | Metric | Value | Source |
|---|---|---|---|
| 167 | pre-run estimate | "প্রায় ৬০ মিনিট" (about 60 min): load ≈12–20 min, preflight < 1, solve ≈2,063 s ≈ 34 min, diagnostic ≤ 4, report 1–2 → 55–62 min; profile hard limit 80 min | I09 §2 |
| 168 | actual | **ERROR at ≈114 min**; `kernel_log` `[]` (2 bytes); 0 output files; empty failure message | I09 §2; I11 §2 |
| 169 | quota burn | 26.87/30 before → 29.18/30 at 18:25 (2.31 h in ≈69 min) → **30.21/30 at 18:56 (≈3.34 h in ≈100 min)** = ≈2× per wall hour; ≈3.13 h (≈94 min) were left before the run | I09 §2; I11 §2 |
| 170 | load time | none recorded (the code very likely never started) | I09 §3; N05 §A6 |

### 1.12 Platform, storage and quota (measured by us or by named participants)

| # | Metric | Value | Source |
|---|---|---|---|
| 171 | our INT4 weight load | 12.8–58.7 s on 10-19/10-20; 669.26–669.96 s on 10-21 (18.6 GB shard 657 s, ≈28 MB/s); launch times 507–562 s on 10-19/20 vs 1,197 s on 10-21 | I09 §2; I10 §2; I11 §2 |
| 172 | `/kaggle/input` storage (amn, topic 751344) | NFS → GCSFUSE from ≈10-21; "NFS 0.55 → 0.026 GiB/s" (I09) / "0.95 GiB/s (10-20) → 0.026 GiB/s (10-21 05:37)" (I10, I11); 1 thread ≈10 MB/s (10.5), 16 threads ≈107 MB/s; GCSFUSE reads do not fill the page cache (Jeroen); GCSFUSE 1.8–2.8 h per shard | I09 §2; I10 §2; I11 §2 |
| 173 | load time per storage mode [E from those rates] | 27B INT4 + DFlash2 (23.3 GB): ≈3.6 min (16 threads) / ≈37 min (1 thread); FP8 (30.9 GB) 4.8 / 49 min; Flash-Next 19–29 min / 3–5 h | I09 §2 |
| 174 | prefetch in our notebooks | one single-stream `cat`; 23.3 GB at 10.5 MB/s ≈37 min > 30-min first-launch timeout (`VLLM_LAUNCH_TIMEOUT_SEC = 30 * 60`, builder L352; `SGL_LAUNCH_TIMEOUT_SEC 1800`) | I09 §2; I11 §2; I02 §2 |
| 175 | quota billing | 4×L4 at **2×** wall (EVAL30 4.15 h → 8.22 h; esfer 69 min → 2.31 h, 100 min → 3.34 h; staff fix ≈10-05; "older memory line says 1.05 quota-h per wall-hour"); 125 min wall ≈ 4.2 h quota; queue time is not billed | E13-P §3; I00 §2; I09 §6; I10 §2 |
| 176 | quota balances (latest) | foysal 4.71 h (25.29/30), hosen42 8.63 h (36.37/45), esfersami50 exhausted (30.21/30) | I09 §2; I11 §2 |
| 177 | host | RAM 202.6 GB; host CPU (assumed g2-standard-48) Xeon Platinum 8273CL, 48 vCPU / 24 cores | I10 §2; E13-V #13 |

### 1.13 Earlier pipelines on 4×L4 (vLLM era and others), as quoted in these notes

| # | Run | Metric | Value | Source |
|---|---|---|---|---|
| 178 | E11-05 comparison: R5 EVAL120 (V, 16) | tok/s over the whole solver | 280.9 (281); KV 99.5 % with 9 preemptions | E11-05; I01 §2 |
| 179 | D66 / L66 (V, 16 / 12) | tok/s | 255.8 / 254.1; D66 at 12 slots, context 155,648: KV peak 90 %, 0 preemptions | E11-05; I03 §4; I04 §2 |
| 180 | V at 16 running | decode; whole run; by running count | 352 decode / 333.3 whole run; 44 at 1, 163 at 5, 268 at 10, 293 at 12, 352 at 16; ≈22 per stream | I09 §2 |
| 181 | V acceptance | MTP-3 | 3.0 (T 0.6), 2.67 (T 1.0), 3.45 (low effort); "MTP 3.0 → 2.7 tokens per step" at T 1.0 | I09 §2; I01 §6 |
| 182 | V production | "solver-average speed" | 255–281 tok/s (vs SGLang 329–380); "vLLM FP8 + MTP-3 (production) is about 255–300 on long contexts" | I00 §2; E11-04 |
| 183 | S1-60K (V) | KV peak | 68 % of 1.35M with 12 × 60k | E11-03 §4 |
| 184 | Exp 1 | tok/s | v1.3.5 247.8; v1.4.1 285.3 (prefix hit 0.786); v1.4.4 274.6 (KV 78.7 %, setup 789 s) | I01 §2 |
| 185 | Exp 2 R6 | tok/s; MTP acceptance | 220.7; 0.678 | I01 §2 |
| 186 | Exp 4 run 4 | prefill vs generated | 2.78M vs 0.66M | I01 §2 |
| 187 | Exp 8 | memory | a prompt_logprobs request OOMed all 4 GPUs (790 MiB all-gather) | I01 §2 |
| 188 | Exp 9 (86k) | tok/s; KV | 306.6; 91.4 % | I01 §2 |
| 189 | Exp 10 ORDER2 EVAL120 | tok/s | 333 | I01 §2 |
| 190 | archive (vLLM) | CUDA graphs; custom all-reduce TP4; prefix caching; TP2 | graphs 48 → 129 tok/s at 8 sequences; custom AR at TP4 0.9995× (no effect); prefix caching at 64 sequences killed the engine; TP2 gave 17k KV and failed | I01 §2 |
| 191 | PIV EVAL120 | tok/s | 65.3; 29 % of the time with ≤ 2 requests active | I01 §2 |
| 192 | vLLM startup | init; prefetch | init 291–295 s; prefetch cut weight load from 64 s to ≈5 s | I09 §2 |
| 193 | ARC-3 P1_SPARSE (vLLM NVFP4, RTX PRO 6000) | tok/s; prompt vs generated | ≈241 tok/s; 1,477 tokens per call, 1,340 calls; prefix caching off: 27.6M prompt vs 1.98M generated | I01 §2 |
| 194 | russcore public notebook (LB 10.14) | tok/s | 119.5 at ≈7.5 running; 36,257 tokens and 2,279 s per sample; 20.2 h projected for 240 | I01 §2 |
| 195 | NVARC | time | ≈9.5 h for 240 tasks | I01 §2 |

### 1.14 External or other-GPU reference numbers quoted in these notes (not our runs)

| # | Source of number | Metric | Value | Source |
|---|---|---|---|---|
| 196 | Franzen ARC-3 (1× RTX PRO 6000, TP1, Pennyroyal 0.5.19, FN) | ready; tok/s; RAM/VRAM | ready 562 s; 590–617 tok/s; host RAM 106/177 GB; GPU 91/96 GB | E13-V #2 |
| 197 | Franzen ARC-3 | prefill vs decode; reuse; prefill time share | prefill 10,553 tok/s (p90) vs 836.6 (837) decode; 93.43 % token reuse; 15.25 % prefill time fraction; replies yield every 2,048 tokens, cap 12k | N11 §B; I10 §2 |
| 198 | Franzen ARC-3 serving config | flags; load | mem 0.96, page 64, max running 10, CUDA graph bs {1,2,4,7,8,9,10}, mamba cache 60, lpm, fp8 KV, NEXTN 3 + FR-Spec (64k hot-token map), `--weight-loader-prefetch-checkpoints`, `--weight-loader-drop-cache-after-load`; ≈10.5 min load; 17.5M tokens / 100 trajectories in 8 h 11 min | I01 §2; N05 §B8 |
| 199 | H100 | SGLang vs vLLM | 1,725 vs 1,610 tok/s (needed `--max-mamba-cache-size 256`) | I09 §2 |
| 200 | 3090 | TP4 vs 2×TP2 | 307 vs 737 at 64 concurrent (2.40×); biMEMO Qwen3.8-27B INT4: c8 TP2 518.6 vs TP4 469.4; c4 TP2 323 → 2 replicas ≈646 vs TP4 c8 469 (≈1.38×) | N00 §3; I09 §2 |
| 201 | 3090 TP4 forced custom all-reduce | speed; correctness | +14.8 %, but greedy diverged on 4 of 5 prompts | I09 §2 |
| 202 | club-3090 #1288 (4×3090, HiCache) | tok/s | 927 at c16 | N05 §B2 |
| 203 | vLLM #55533 | MTP on sm89 INT4 | runs only ≈3 sequences (649 vs 211) | I09 §2 |
| 204 | roofline | INT4/FP8 ratio | 1.38× at 8 streams, 1.28× at 16, 1.19× at 32 | I09 §2 |
| 205 | TheDivine Aurora | tok/s vs score | 300–500 tps gave ≈19–20 % | I00 §2; I09 §2 |
| 206 | SSM state size | per slot | 153.9 MB fp32, 78.4 MB bf16; Nemotron: a straight FP16 cast raised verbosity by up to 40 % | I09 §2 |
| 207 | DSpark (RedHat card) | vendor accept length | 3.62 vs DFlash2 4.80; 7 draft tokens recommended; needs block+1 state slots | N05 §B3 |
| 208 | #36548 | DFlash2 errors | 111/304 wrong with DFlash2 vs 0/304 without speculation; width 4, fp32 SSM, no decode graph or the #36014 beta fix each give 37/304; dense BF16 lm_head 1/304 | I10 §2 |

### 1.15 Experiment 13 (Flash-Next on 4×L4) plan numbers (estimates unless marked; never run in E13)

| # | Item | Value | Label / source |
|---|---|---|---|
| 209 | weights on 4×L4 (E3, Intel W4A16, EP4) | 19.06 GB per GPU; load 345 s / 124.8 GB with `--weight-loader-prefetch-checkpoints` | [M] E3, via E13-P R5, §3b; E13-V #17 |
| 210 | free per rank | before weight load 21.62 GB; after weight load **2.55 GB**; after pool + warm-up (bf16 KV 118,080 tokens = 1.36 GB, 10 mamba slots = 0.16 GB, graphs off, NEXTN off) 0.97 GB after the pool, **0.66 GiB** after warm-up | [M] E3 serve.log; E13-P §3b |
| 211 | NEXTN drafter | 4,139,535,872 B = 3.855 GiB → ≈0.96 GiB per rank if sharded perfectly | [M] payload; [E] per rank; E13-P §3b |
| 212 | mamba pool | ≈0.016 GB per slot (10 slots = 0.16 GB); 60 slots ≈0.96 GB | [M] per slot, [E] scaling; E13-P §3b |
| 213 | CUDA graphs (bs ≤ 10) | ≈0.3–0.6 GB | [E]; E13-P §3b |
| 214 | prefill activation peak | 1.5–2.6 GB above post-graph at chunk 4096 and an OOM with 2.4 GB free (RTX PRO 6000, TP1) | [M on another GPU]; E13-P §3b |
| 215 | KV | 6,144 B per token per rank fp8 (12 layers × 1 KV head × 256 × 2): 48k = 0.295 GB, 98k = 0.602 GB, 170k = 1.044 GB; pool estimate ≈170–250k fp8 tokens; target-only scenario ≈80–170k tokens = ≈1–3 Duck contexts | [M arithmetic], [E] pool; E13-P §3, §3b; E13-V #20 |
| 216 | Duck capacity vs 27B | 170–250k KV holds ≈2–4 contexts of p50 48k / p90 98k vs ≈12 for 27B W1 | [E]; E12-E30 "Consequence for E13" |
| 217 | throughput | 420 tok/s (27B W1 decode at 11–12 running) is a comparison threshold; per-stream needed 420 / 210 / 140 at 1 / 2 / 3 streams; 700 tok/s aggregate is a target, not a gate | [E]; E13-P §3b, §6 |
| 218 | session wall | install ≈3 min (E11 181 s) + micro-preflight ≈5 + pinned test ≈2 + weight load ≈8.5 (185 GB at 0.36 GB/s) + JIT/graphs ≈10 + Gate C ≈10 + speed ≈20 → ≈60–70 min; ≤ ≈2 h with B2; hard stop 2.5 h (Launch 1 75 min, Launch 2 60 min); E3 warm-up 7.4 min | [E]; E13-P §3, §3b |
| 219 | host RAM | pinned BF16 PLE ≈95 GB (up to ≈128 GB with rounding); BF16 PLE = 107.6 GB in total (HF sizes); host 202.6 GB | [E]/[M]; E13-P §3, §3c |
| 220 | rev 5 simple path (E13 SMOKE1, built, never run) | NEXTN off, chunk 2048, mem 0.93, `--max-running-requests 4`, context 139,264; Duck streams = min(4, KV // 60k); Flash-Next load ≈25 min; SMOKE1 time 50 → 90 min | E13-P header; I00 §2 |
| 221 | Franzen verbatim items kept | ctx 139,264, 10 running, chunk 8192 (Launch 1 2048), prefill 16384, fp8 KV, page 64, extra_buffer + track 64, mamba cache 60, CUDA graphs ≤ 10, lpm, mem 0.96 | E13-P §2 |

### 1.16 Derived and estimated speed numbers in these notes (not measurements; kept separate on purpose)

| # | Item | Value | Label / source |
|---|---|---|---|
| 222 | step-time formula | step = running × accept len ÷ gen tok/s, from ≈4,000 TP0 "Decode batch" lines (E11-03 sweep + E11-06). The log "step" = logging-window wall ÷ 40 forwards, **prefill included** (`metrics_reporter.py` L898-937, L501) | [D] N00 §1; I10 §2 |
| 223 | marginal cost per verified token | ≈0.85–1.3 ms; "looks like an effective ≈10–16 TFLOPS per GPU vs 121 peak" (**not established**: it also includes communication, launches, dequant, GDN, spec overhead) | [D]/[E] N00 §1; N05 §A3; I09 §2 |
| 224 | all-reduce share, Claude model | 128 all-reduces × ≈10 KB per token → ≈0.42 ms per verified token ≈ 40–50 ms of a 115 ms step (large-message bandwidth applied to small messages; withdrawn as "hypothesis") | [E] N00 §2; N05 §A4 |
| 225 | all-reduce share, source count | 140 all-reduces of 983,040 B per step (129 target verify + 11 draft) + 1 logits all-gather 11.9 MB per rank (47.7 MB total, [96 × 248,320] BF16) + 2 small all-gathers (5,376 B and 10,752 B) + 1–2 broadcasts; naive bandwidth model 45 ms; ≈340–380 µs per call → 48–53 ms; loose range 35–55 ms (30–45 % of a step); **unmeasured** | [D source count] / [E] I10 §2; I11 §2 |
| 226 | step budget | arithmetic 87–131 ms (centre ≈108), ≈7–25 ms unexplained (report); the profiling agent left 60–80 ms unexplained; never reconciled with a trace | [E] I11 §2; I10 §5 |
| 227 | logits all-gather time | 3.6–15 ms (unknown) | [E] I11 §5 |
| 228 | width 8 → 4 | Claude: step ≈70 ms → 550–650 tok/s; GPT: T8 ≈ 12 × 4.3 / 0.1155 ≈ 447 tok/s, constant α ≈ 0.811 → L4 ≈ 3.00, L6 ≈ 3.78 → 4-width at 70 ms ≈515 tok/s; "550–650 tok/s: optimistic hypothesis, expected result নয়" | [E] N00 §2; I09 §2, §4 |
| 229 | width 3 / 2 | width 3 ≈2.5 tokens per step; width 2 crashes at graph capture | [C] DFlash agent, I10 §4 |
| 230 | 2×TP2, first estimate | 1.3–1.6× (biMEMO-derived 1.38×); INT4 TP2 ≈9.3 GB weights per GPU; DFlash2 replica ≈0.35–0.4M KV ≈5–6 long requests | [E] N00 §3; I09 §2 |
| 231 | 2×TP2, revised | ≈170k KV per replica with DFlash2 at mem 0.83 (224k at 0.88; MTP 235k; no-spec 500k) vs 881k for TP4 (≈340k system-wide vs 700–920k needed); expected 0.7–1.0× of TP4; weights per GPU at TP4 INT4 4.62 + DFlash2 1.49 = 6.11 GB → TP2 ≈12.22 GB | [E] I10 §2, §4; I11 §4; I09 §2 |
| 232 | budget by speed (11.3 h, 240 tasks, W1 whole-run ≈355 tok/s) | 1× ≈14.4M tokens ≈60k per task; 1.3× ≈18.8M ≈78k; 1.6× ≈23M ≈96k; 2.4× ≈34.7M ≈144k ("LB near 50 if the quality held"); 460 → ≈78k; 570 → ≈96k; 850 → ≈144k | [E] N00 §3; I09 §2 |
| 233 | coverage requirement | 240 × 60k = 14.4M tokens = 10.81 h at 370 tok/s vs 11.46 h window after a 1,463 s solve start → ≈370 tok/s effective needed; one 60k attempt per task in 12 h ≈333 tok/s, two ≈667; 15.1 h at 265 tok/s, ≈10 h at 400; two branches × 60k = 28.8M = 18.6 h even at 430; 55–72k per task needs 330–430 tok/s | [E] I06 §2; I11 §2; I09 §2; I01 §6; I03 §2 |
| 234 | per-task budget at measured speeds | 11.3 h × 3,600 × 255 / 240 ≈ 43k (43.2k); 350 → 59.3k; 416 → 70.5k; 6 streams for ≈11 h ≈54k vs 63k at 12 | [E] I02 §2; I05 §2; I07 §2 |
| 235 | coverage gap | full coverage ≈2.8 min per task vs ≈9–10 min measured → ≈3.4× short; rev 4 Duck ≈200k per task (≈3× the 55–72k needed); Kris Dreemurr ≈35k per attempt / 70k per task | [E] E12-R5; I00 §2; I09 §2 |
| 236 | stage-1 coverage | 240 × ≈33k ≈ 7.9M tokens ≈5.9 h, i.e. ≥ ≈192 tok/s average; G4B at hidden scale: 11 streams × ≈34 tok/s ≈370 tok/s ≈64k per task at 172 s | [E] E12-G4B F3; I09 §2 |
| 237 | 240-task simulator (`sim240.py`, window 41,257 s, 12 workers) | at 371 tok/s, chunk 12,288: 240/240 exposed, ≈15.0–15.3M tokens, per task p10/p50/p90 49.6k / 62.8–63.2k / 72.1–72.5k, 85–90 tasks under 60k, b1 started 176–181 times, fair share p50 ≈4.0k; chunk 24,576: b1 103, 273–276 stage-2 runs; at 277 tok/s: ≈47.6k per task, 169 tasks under 60k, b0 resumed in stage 2 for 133–151 / 240 (vs 232 at 371); stage 1 finished at 48 % of the window (≈5.5 h) at 371 vs 64 % (≈7.3 h) at 277; KV hold 0; KV max 0.40 | [E sim] I08 §2; I09 §2 |
| 238 | Duck KV fit | 12 × 58k = 697k fits 866,744 at the rev 5 median; 12 × p90 (≈75k) ≈890–900k does not; 10 streams fit p90 (750k); "12 streams need ≤ ≈72k per branch (867k / 12)"; EVAL30: p90 context fits ≈9 concurrent, p50 ≈18 | [E] E12-R5; I00 §2; I01 §2 |
| 239 | per-stream window vs streams | 10 streams: 0.9 × 881k / 10 = 79.3k (79,320); 12 streams: fraction 1.0 → 73,444 (history 48,356 at the 24,576 cap, 60.6k at 12,288), 0.95 → 69,772 / 44,684, 0.9 → 66,100 / 41,012; worst case 12 × 73,444 ≈ 881,328 (no spare KV); real G4B window 72,228 | [E] I02 §2; I03 §2; I04 §2; I06 §2 |
| 240 | Duck context budget | window − max_output − 512: 131,072 / 60,000 → 70,560 (drain 35,280); 49,152 output → 81,408; 12,288 cap → 118,272; server context 155,648 vs Duck window 131,072 (≈24k margin); 10-stream ≤ ≈79k per stream (10 × 118k input budget ≈1.18M > 881k KV) | [E] I00 §2; I01 §5; I02 §2 |
| 241 | KV need for production (E11-03) | ≈920k KV tokens (S1-60K peaked at 68 % of 1.35M with 12 × 60k); S49 ≈12 × (≤10k + 49k + 8k) ≈800k; target + draft KV ≈9.99 KB per token per GPU → ≈9.2 GB (+5.3 GB); sources: free 4.39 (keep ≈1.4) ≈3.0, prefill graph off 2.60, lazy K=56 ≈0.2, ReplaySSM ≈1.8 → 5.8 GB without ReplaySSM, 7.6 GB with | [E] E11-03 §4, §6 |
| 242 | raw-reasoning fix value (G4B scale) | 217.5k + 2.4k re-prefilled reply tokens at ≈1,500 tok/s ≈146 s of server prefill; deeper misses 66.8k ≈45 s; together ≈72 % of uncached prompt tokens; working estimate **≈+10 % effective** (unmeasured); "fixing re-prefill without engine work caps at ≈400–460" | [E] N11 §B; N12 §3; I11 §6 |
| 243 | request durations | 60k reply at 368 tok/s over 6 streams ≈978 s (> 900 s timeout); 49k reply at ≈35–61 tok/s per stream ≈800–1,400 s; 60k request at 31–35 tok/s ≈29 min; 12,288-token first cap ≈6 min; 24,576-token chunk at 25–30 tok/s per stream ≈14–16 min | [E] I00 §3; I01 §7; I09 §7; I04 §4 |
| 244 | rev 4 solve window sizing | 12 × ≈120k = 1.44M tokens ≈65 min decode + ≈14 min prefill → window 60 → 90 min | [E] I00 §4 |
| 245 | load-path estimates | 23.3 GB at 107 MB/s ≈3.6 min, at 50 MB/s ≈7.8 min; weights from /dev/shm ≈13–60 s; RAM peak ≈47 GB of 202.6 GB; JIT cache saving ≈360 s per cold start (also quoted as "−4 to −5 min" and "≈6 min") | [E] N07; N13 §2; I11 §7; I02 §4; N05 §B10 |
| 246 | E14-A plan time and quota (N13) | A1 ≈61 min normal, ≈106 worst → HARD_STOP 95 min; ≈2.0 h quota normal, ≤3.2 h; A2 ≈80 min (93 with A2-7), HARD_STOP 110; 2.7–3.1 h quota, ≤3.7 h; per relaunch graph capture + startup ≈6–8 min; "not ready by T0 + 45 min → stop" (≈1.5 h quota max) | [E] N13 §2 |
| 247 | other phase estimates | GPT caps: topology + NCCL 5–8 min; TP4 DFlash8 startup + profiler 15–25 min; each width restart +10–20 min; report Tiers A–C 60–70 min wall, 2.0–2.4 quota-h, worst ≈3 h; current single-notebook profile 200 min ≈6.7 quota-h with a 75-min give-up; builder steady windows 75 s each (report suggested 90 s), profile window 60 s, latch test 11 decodes + 3 arrivals over 60 s | [E] I09 §2; I11 §2, §7 |
| 248 | token-efficient tunes | Swift / ThinkingCap claim 37–58 % fewer thinking tokens (≈1.6–2.4× coverage if accuracy holds); realistic −10 to −25 % at ≈60k | [C]/[E] N05 §B7; I10 §2; I11 §4 |
| 249 | CUDA graphs 1..13 | "three more verify graphs of memory (today 0.68 + 0.35 GB for 10 graphs)"; each new width or graph shape may cost ≈6 min of recapture | [E] I11 §6, §7 |
| 250 | width-arm health thresholds (proposed) | mean accept below 3.5 (W8) / 3.2 (W6) / 2.5 (W4) fails; accept rate 0.00 in any window fails | [E] I10 §7 |

---

## 2. Speed problems found (with numbers and sources)

| ID | Problem | Numbers | Status / cause | Source |
|---|---|---|---|---|
| S1 | **Mamba state slots silently capped real concurrency** | speculative decoding needs 5 slots per request (extra_buffer + overlap); real max running A 22, B 10, C **6**, D 13 while `/get_server_info` said 32; sweep levels above the cap ran as cap + queue (`log gen None`); C's "12/16/24/32" numbers were really 6 concurrent | Fixed in E11-04 W1 (lazy = 4 slots per request; mamba cache 56 → up to 14 requests). Harness bug: the sweep must assert real running == level | E11-03 §1, §4; I01 §7 |
| S2 | **GPU memory left unused and a small KV pool** | auto mem 0.62; 4.39 GB per GPU (≈19 %) free; D KV only 387k vs ≈920k needed; only ≈4 requests of 85k fit C's 395k pool | Fixed in E11-04: KV 881,336 (866,744 at server mrr 13); 1.88–2.2 GB free per GPU remain | E11-03 §1, §3, §4; E11-04; I10 §2 |
| S3 | **"11 of 12" admission delayer** | E11-05: 2,358 lines at 11 running + 1 queued vs 114 at 12; the 12th waited up to 24 min; cost ≈8 %; E11-04 sweep max 11; EVAL30 11 + 1 queued 50 % of the time (≈9 %); G4B 25 %; same-context cost −8–9 % (EVAL30 422 vs 462; G4B 397 vs 436); queue = 1 is the steady state at 12 in flight (E11-05 2,615/2,816 lines, EVAL30 3,127/4,153, G4B 101 of 188 queue-1 lines at running 11) | Root cause changed four times (see §4 C1). Latest (I11): 0.5.21 DFlash-only min-free-slots delayer (`scheduler.py` L1183-1193, `min_free_slots_delayer.py` L17-22), threshold `min(4, max(2, (mrr+5)//6))` = 2 at mrr 12, **3 at mrr 13**, so the "server 13 / client 12" workaround still delays; not delayed at mrr 14/15. Untested fixes: `--min-free-slots-delay 1` ("nodelay" arm), mrr 14 with client 12, `--max-mamba-cache-size 64`, #44401 backport (which "probably does not fix the symptom") | E11-05; E11-04; I10 §2; I11 §2, §3, §6 |
| S4 | **Capacity probe stepped 12 → 6 on a normal queue = 1** (rev 6d) | events at 240 / 600 / 780 s at KV 0.13 / 0.26 / 0.31; 27.7 % of G4B time at exactly 6 running; 370.9 → 321.4 tok/s at 20–40k = **−13.3 %**; no cooldown (allowed affects only new admissions, so the probe re-triggered) | Root cause: the fake server's "healthy" queue = 0 was an assumption. User replaced it with the KV-only rule (≥ 0.95 → no new admission, 12 workers fixed). Never GPU-tested in its new form (rev 6g G4B ended in ERROR) | E12-G4B F1; I06 §2–§4; I10 §2 |
| S5 | **KV thrash with 30 live Duck contexts** (EVAL30 rev 1) | 30 time-sliced branches × ≈50–100k > ≈870k KV; 144 of 423 later requests lost their prefix; **8.39M of 9.60M** uncached prefill tokens; generation 187 (or 167.8) tok/s vs decode ≈420 (or 376); with no thrash the run would have taken 5,706 s instead of 12,634 s (2.1×); KV p90 0.93–0.95, max 0.99–1.0, 167 lines ≥ 0.95; prefill share 55.1 %; uncached ÷ generated 4.07; branch 1 never started in 29/30 tasks | Cause: scheduler trap `live_branches=None` → `n_live = len(jobs)` (all 480 branches would go live at 240 tasks, `arc2_duck.py:976/984`). Rev 4+ fixed with ≤ 6 live; rev 6 KV-fit admission; KV-only gate replays show no hold in later runs | E12-E30 §2; I00 §2; I01 §3; I02 §6; I06 §2; I10 §2 |
| S6 | **Engine stall with zero throughput** | EVAL30 at 14,247 s: `num_running_reqs` 11, `num_queue_reqs` 8, `gen_throughput` 0.0 | Unexplained; "likely KV thrash"; no runtime stall detector exists (the E14 watchdog covers launch only) | I02 §2, §3; I03 §5; I07 §7 |
| S7 | **Reply re-prefill caused by Duck's reasoning normalization** | hit stops at the end of the previous prompt in 72–91 % of later requests (EVAL6 84.5 %, EVAL6R5 91.3 %, LOW 72.5 %, G4B 75.6 %, EVAL30 59.1 %); G4B server accounting: 68 of 150 requests (45 %) lost the reply, **217,541 tokens** (agent figure 245k); ≈146 s of prefill during which every stream's decode waits | Cause: `tool_agent.py` L3414-3418 `_normalize_message_content` deletes every blank line (and `<think>` tags, L93) before storing reasoning; 0 of 180 `[THINKING]` blocks had a blank line. CPU proof: switch off 24/24 follow-ups break at the first blank line (char 57–58), switch on 24/24 full match. Fix `E12_RAW_REASONING_HISTORY=1` built (arc2_duck 7644aa06 → 758bddb5), default ON in E14, never GPU-tested. Same code in Franzen/Tufa (costs him a few %; on 4×L4 3–4× more). Still possible after the fix: template `reasoning_content\|trim` and tool-call XML re-render, non-canonical sampled tokens (`"\n\n"` = [271] or [198, 198]); 12 lost replies with ≤ 300 reasoning chars unexplained | N06; N11 §B; N12 §2; I10 §2, §6; I11 §2, §6 |
| S8 | **Deeper misses and zero-hit requests at low KV** | G4B: 18 deeper misses = 66,771 tokens (6 at k = 1 inside the system/first message, cached 1,536–1,664 of a 2,186–4,575-token prompt; 10 of 18 within 64 tokens of an older checkpoint); 20/180 requests with `cached = None` = 0 (11 % vs ≈5 % in rev 5); 14 zero + 15 partial hits = 317k tokens (47 % of uncached); cache 75.0 % vs 84.3 / 88.7 % in rev 5 (−9.3 / −13.7 points) | Fake server shows 0 earlier-message edits → attributed to Mamba checkpoint eviction (E14-M; lazy mode has no evict-retry for track slots), #22935 (mamba radix split gives 0-hit) or #36935 (kv_pool / chunk 211.6 > mamba cache 56; downgraded because mamba usage never exceeded 0.70–0.71). Unproven | N11 §B; N12 §2; I06 §2, §3; I07 §3; I08 §5 |
| S9 | **Prefill is slow on 4×L4 and blocks decode** | chunked prefill 1,474–1,578 tok/s vs ≈400 decode; one uncached prompt token ≈0.27 generated token (Franzen ≈0.08); E11-06 windows with a prefill line: median step **323 ms** vs 124.6; G4B 12-running lines alternate 42–250 / 382–497 tok/s; cold resume ≈30–35 s per 50k | Structural; levers are prefix reuse (S7, S8), fewer requests, stable prefixes | I10 §2; I11 §2, §6; I06 §2; I02 §6 |
| S10 | **Effective (agent) throughput far below decode throughput** | effective vs decode-only: EVAL30 167.8 vs 376; EVAL6 325 vs 377; EVAL6R5 244 vs 302; LOW 265 vs 335; G4B 279 vs 353; note 05: "agent-side cache and admission loss (about 35–55 %) is bigger than any engine lever we discussed" | Sum of S3–S9, S11–S14 | I10 §2; I11 §2; N05 §B1 |
| S11 | **Uncached prompt tokens exceed generated tokens** | EVAL6R5 0.97M vs 0.85M (ratio 1.15); LOW ≈2.03M vs 1.74M (1.17); G4B 669,980 vs 572,035 (1.17); rev 4 EVAL6 0.78M vs 1.22M (0.64) | Partly S7/S8; shorter replies (rev 5) mean more requests and more prefill | E12-R5; I00 §6; I06 §2; I10 §2 |
| S12 | **More, shorter requests cost prefill** (rev 5, low effort) | rev 5 effective 244 / 265 vs 325 in rev 4; low 449 requests vs 127 (18.7 vs 10.6 per branch) at about the same tokens per branch (72.7k vs 70.5k) and more uncached prefill (1.58M vs 0.89M request-level) | Observed; not a fix target in itself | E12-R5; I01 §6; I02 §2 |
| S13 | **Previous reply dropped from the next prompt** (rev 4) | EVAL6 11 of 84 later requests, **426,781 of 1,217,904 generated tokens (35 %)** thrown out; EVAL30 12/423 (4 %); 5 cases (≈279k) avoidable: grid-calibrated 1.40–1.72 chars/token estimated 77,269–111,375 vs real 65,203–69,461 | Fixed in rev 5 (B3 real tokenizer: median +0.4 %; G4B ratio 0.998) | E12-W W2; I00 §2, §6; I02 §2 |
| S14 | **Trims and the per-stream window force re-prefill** | trims change the prefix by design (G4B: 1 history_shrank request, 23,099 tokens; 4 opener rewrites after trims, +216 chars ≈50–110 tokens each); 49k reserve forces trims at ≈82k context (low 4c3d4a41_b0: 66,189 + 16,365 + 49,152 > 131,072 → prompt cut to 19,856); more streams = less history (12 streams 72–73k window → 41.5–48.4k history) | Design trade-off; KV fraction 1.0 chosen at 12 streams | N10 §A4; N11 §C; I00 §2; I02 §6; I05 §2 |
| S15 | **Low real concurrency / small-run tail** | rev 4 EVAL6 ran 6 live (368 tok/s at 6 vs 431 at 12); rev 5 130 decode lines with 1 running; EVAL30 below 12 running 67.8 % of the time; G4B below 11 running 47 % | Partly S3 and S4; partly short queues | E12-R5; I00 §2; I10 §2 |
| S16 | **Decode slows and acceptance drops with context** | E11-05 429 → 323 tok/s (<10k → 40–60k, −25 %); E11-06 500.1 → 389.4; step ≈115 ms (≤10k) → 133–140 ms (50–80k); acceptance 4.43 → 3.77; T 1.0 lowers acceptance (DFlash 4.60 → 4.34, MTP 3.0 → 2.7); per-run accept means 3.87 / 4.58 / 5.10 | KV-read bandwidth-shaped (≈16–18 ms per 50k at 12, "not a bug"); short-context benchmarks overstate production speed | N00 §1; I01 §6; I02 §6; I10 §2; I11 §2 |
| S17 | **Heavy DFlash2 width-8 verify step** | 115.5 ms at 12 running vs 37.9 ms without speculation; marginal ≈6.8 ms per request (0.86 ms per verified token) vs ≈1.3 ms; accept len ≈4.3–4.6 of 8 means ≈3.3 of 7 drafts accepted (≈47 %); lm_head 248,320 × 5,120 logits for 96 positions per step plus drafter lm_head; top-k 20 / top-p 0.95 sampling over 248k per position (never quantified) | Width 6/4 arms untested; the checkpoint is trained at block 8, SGLang calls `set_block_size()` with a warning; drafter attention is bidirectional (`is_causal: false`), so width 4 ≠ truncation of width 8 | N00 §1, §2; N05 §A2, §B8; I09 §6; I10 §4, §6 |
| S18 | **TP4 all-reduce over PCIe without custom all-reduce** | custom all-reduce disabled on all 4 ranks in every run ("not supported on more than two PCIe-only GPUs"); 64 MiB fp32 eager all-reduce 20.9 ms (3.21 GB/s algbw, 4.82 busbw); 140 all-reduces of 983,040 B + a 47.7 MB logits all-gather per step; estimated 35–55 ms (30–45 % of a step); transport (P2P vs SHM) unknown (no `NCCL INFO` line); PCIe Gen4 x16 is ≈31.5–32 GB/s per direction (not 64) | **Never measured at the real sizes** (no smaller sizes ever measured); NCCL bench planned in E14-A | E11-03 §1; N05 §A1, §A4; I09 §6; I10 §2; I11 §2 |
| S19 | **CUDA graphs do not cover the running sizes used** | decode graphs `[1..8, 10, 12]` while server mrr is 13: batches of 9 and 11 pad up, **13 runs without a graph** | Arm "graph bs 1..13" planned, untested; costs ≈3 more verify graphs of memory (0.68 + 0.35 GB today) and ≈6 min recapture | I03 §2, §7; I04 §6; I11 §2, §7 |
| S20 | **Slow cold start and expensive relaunches** | cold start 520–680 s, of which 372–385 s is verify CUDA-graph JIT (first bs=12 bucket ≈362 s, silent ≈6 min); E11-03 first launch 601 s (prefill-graph JIT 447 s), later 136–166 s; E11-04 first 567 s, later ≈75 s; E12 SMOKE1 677 s; rev 5 start to solve ≈13 min (733–799 s); G4B rev 6d **1,197 s** (weight load 670 s); install 208 s; 1,197 s cost ≈40 quota-min; each relaunch ≈6–8 min (or 10–20 min per GPT), but warm same-session capture 11.94–13.67 s | JIT cache (save only in first E14 run) and the load path are CPU-built, untested on GPU | E11-03 §3; E11-04; E12-SM; I05 §2; I09 §2; I10 §2; I11 §2; N13 §2 |
| S21 | **GCSFUSE slowdown and a fragile load path** | `/kaggle/input` partly on GCSFUSE since ≈10-21 (0.026 GiB/s; 1 thread ≈10.5 MB/s, 16 threads ≈107 MB/s; reads do not fill the page cache); our 18.6 GB shard at ≈28 MB/s (657 s); single-stream `cat` prefetch ≈37 min for 23.3 GB > fixed 30-min first-launch timeout (`SGL_LAUNCH_TIMEOUT_SEC 1800`, `VLLM_LAUNCH_TIMEOUT_SEC = 30 * 60`) → E11Fail → fallback → LB ≈0; SGLang mmaps safetensors (`weight_utils.py` L1160), so `rchar` never grows and the log stays silent on one 18.6 GB shard (a healthy ≈37 min load would have been killed at 15 min by a log/rchar watchdog) | E14-0 load path rev 2 (ranged parallel copy to /dev/shm, ramp 4 → 24 workers, stall watchdog with page faults, 75-min give-up) — 56 CPU tests, never run on Kaggle | N07; N11 §A; I09 §2, §3; I10 §3; I11 §2 |
| S22 | **Platform ERROR before our code ran** (rev 6g G4B, esfersami50, 10-22) | ≈114 min, empty log `[]`, 0 files; quota 26.87 → 30.21/30 (≈3.34 quota-h); "cancel also saves nothing" | Cause unknown: mount/attach stall > pinned old image > quota | I09 §2, §3; I10 §3; I11 §2 |
| S23 | **Deadline overrun from per-turn timeouts (T8)** | EVAL30 overran its solve deadline by 679 s (first misreported as 1,412 s); requests with read timeout 419.6 s and 469.7 s finished 336 s+ past the deadline; worst case one analyzer timeout (1,800 s); PARITY `urlopen(timeout=900)` × 24 probes = up to 6 h; per-thread join +120 s × 12 = 24 min | Fixed in rev 6 (per-request clamp `min(given, deadline − now)`, verify clamp) and 6d (one join cutoff deadline + 60); G4B: 91 clamps, 0 stragglers, finished 9 s early. PARITY not fixed (NO-GO) | I02 §3; I04 §3; I06 §2; I08 §6 |
| S24 | **Non-streaming Duck request timeout shorter than a long reply** | 900 s default vs a 60k reply at 368 tok/s over 6 streams ≈978 s; EVAL30 40 timeouts in 21 of 31 branches (31,247 s); a ReadTimeout is not retried and the in-flight reply is lost (`tool_agent.py:628`, `:7106`) | Fixed with `E12_DUCK_REQUEST_TIMEOUT=1800` (EVAL6: 0 timeouts). Whether SGLang keeps generating after a client timeout (orphan generation) is unverified | E12-W W6; I00 §2, §3, §7 |
| S25 | **Token waste in repetition loops** (rev 4) | 29.4 % of EVAL6 thinking characters verbatim repeated lines (EVAL30 5.2 %); ≈131k of 368k tokens in three failed branches; Duck's zlib degeneracy threshold ≈0.01 never fired (real 0.037–0.19); presence/frequency/repetition penalties are a no-op under DFlash2 in 0.5.21 (`cumulate_penalty_output_tokens` is EAGLE-only, #28180) | Rev 5 (T 1.0, prompts): 0–3 % (xhigh), 0–8 % (low). A penalty arm would need a 2-line SGLang patch | E12-W W3; E12-R5; I00 §2; I03 §2 |
| S26 | **Soft caps and soft pauses overshoot** | 100k soft cap → 159k (max 159,464); 28k soft pause overshoot p50 5.1k, max 15.8k; a soft 35k cap only cut tokens per task 141k → 91k while accuracy fell 4/6 → 1/6 | By design (cap checked between requests); a hard clamp of `max_tokens` would be needed | E12-W W7; E12-R5; E12-G4B F3 |
| S27 | **Output-cap cuts that cost a request** | EVAL30 27 % of requests at 12,288 (29/30 first requests); G4B: all 4 b1 first requests cut at 12,288 before any call (first call 12.4–18.0k), b1 used 21.6 % of tokens with 0 candidates | G4-2: b1 first cap 24,576 (built in rev 6e+, not GPU-run) | E12-E30 §3; I00 §2; I06 §2 |
| S28 | **Fair share is inert** | shares 3.8–7.5k (G4B) and p50 ≈4.0k (240-task sim) vs minimum chunk 24,576 | G4-17: minimum chunk 12,288 (built, not GPU-run) | I07 §2; I08 §2, §4 |
| S29 | **Tokens spent before the first useful test** | rev 5 first `check()` at median ≈55k (55.1k xhigh, 53.7k low; also 56.6k / 57.5k); first verified candidate ≈56–58k; xhigh first reply p50 33.6k; G4B: 7 of 12 b0 never wrote a transform in 29–59k tokens; 28 of 127 xhigh requests carry 80 % of tokens | Agent-level; G4-6 token-triggered nudge at 24k (built) | E12-R5; I01 §2; I06 §2 |
| S30 | **Poetiq prefix cache broken by example-order shuffle** | phase 1 iteration ≥ 2 10.6 %, iteration 1 18.4 % (E11-05) | Fixed in E11-06 (stable example order): 87.2 % cached, 379.9 vs 328.9 tok/s | E11-05; E11-06; I02 §2 |
| S31 | **Aborted requests lose their cache** | E11-04 long-prefix continuation 0 cached (aborted samples); abort without waiting for the stream gives 0 % vs 52–99.7 % with waiting | Fixed in the E11 rev 2 shim (wait for each stream to end) | E11-04; E11-05; I02 §2 |
| S32 | **Prefix-cache killers in requests** (Franzen measurements and ours) | a request without tool definitions: 99 % → 15 % shared prefix; toggling `enable_thinking` changes the system header: 75 % → 45 % reuse; llama.cpp drops `reasoning` (69 vs 3,670 prompt tokens); Python `hash()` ordering changed an LB between reruns | Rules kept in rev 6 (same tools, same kwargs, no tool-free request) | I01 §6; I02 §2 |
| S33 | **Host state ≠ GPU KV resident** | a paused branch's resume pays ≈30–35 s of cold prefill per 50k | Works against high pause counts | I02 §4, §6 |
| S34 | **Metric traps that produced wrong speed readings** | `/get_server_info` 32 vs real 6; `sglang:token_usage` = max(full, swa, mamba) ("FIXME: misleadingly named", mamba 0.62–0.70 at idle KV) — only `sglang:full_token_usage` is KV; null `cached_tokens` = 0 with `--enable-cache-report` (`usage_processor.py:13-15`); request-level vs aggregate cache (85.4 / 91.0 vs 84.3 / 88.7); 12-running windows contaminated by admission prefill ("12 is slower" withdrawn); mixing context bins gave "15–20 %" instead of 13.3 %; step time includes prefill; vLLM `_E12V_ALIAS` vs SGLang `_E11_SNAP_KEYS` filter mix-up; two KV gates side by side (`SGL_KV_ADMIT_MAX` 0.85 with `SGL_KV_GATE_MIN_ACTIVE` 10 vs `E12_R6_KV_ADMIT` 0.95); `speculative_draft_model_quantization: None` in server_args is how SGLang logs `unquant` | Documented; E14 rules: assert real running from logs or `/metrics`, bin by context, exclude prefill windows | E11-03 §6; I04 §2, §3; I06 §2, §6; I07 §6; I08 §2, §6; I10 §6 |
| S35 | **Quota doubles the cost of every GPU minute** | 4×L4 billed 2× wall; the 200-min E14A profile ≈6.7 quota-h vs hosen42 8.63 h and foysal 4.71 h (esfer exhausted); 11 days left as of 10-22 (deadline 2026-11-02) | Constraint on how many measurement minutes are affordable | I09 §6, §7; I11 §7 |
| S36 | **Startup-phase interference in benchmarks** | E11-04 sweep at 12 with staggered 4,096-token prefill reached only 11 running; PROD was cut to 10 min so ≥40k contexts were never measured; E11-03 long-test comparison was unfair (A 56,742 of 57,122 cached vs ≈1k for B/C) | Lessons recorded; E14 plan uses shared 40k prefix + 1k tails | E11-03 §6; E11-04; N13 §1 |
| S37 | **Flash-Next KV is far too small for Duck contexts** | ≈170–250k fp8 KV (estimate) = ≈2–4 contexts of p50 48k / p90 98k vs ≈12 for 27B W1; target-only ≈80–170k = 1–3 contexts; NEXTN + 10 streams leaves ≈0.2 GB for KV + prefill | E13 never ran; Flash-Next on L4 ruled out in the research ranking | E12-E30; E13-P §3b; I09 §2 |
| S38 | **Thinking-budget enforcement silently absent** | vLLM era Exp 5: 10/12 requests ran to the 57,344 cap (MTP bypasses the logits processor); SGLang Qwen3 thinking-budget processor hard-codes old token ids (151667/151668 vs 248068/248069) and does nothing (#25536) | Not used | I01 §2, §6; I10 §6 |
| S39 | **Speed arms gated by open correctness bugs** | #36548 DFlash2 state contamination under concurrency (111/304 wrong vs 0/304; ≈90 % → ≈50 %); #38009 thinking divergence; #43514 DFlash acceptance → 0 with grammar + mixed-chunk; #39087 quantized drafter ≈0 % acceptance; #42917 stale kernel cache PPL 9.98 vs 6.05 (6.28 after deleting `nv/` only); forced custom all-reduce diverged greedy on 4/5 prompts; greedy parity under batching shares only 152–166 of 512 tokens, an unchanged repeat scored 0.1875 vs a 0.50 threshold | All unverified on our stack; correctness gate redesigned (A/A + target-only + paired CI) | I09 §2; I10 §2; N10 §C; I01 §6 |
| S40 | **Required stream settings under DFlash** | `--stream-interval` must be 1 with DFLASH (rule `len(output_ids) % N == 1`, `output_streamer.py:427-436`; rationale text cut in the transcript) | Kept in every arm | I01 §2; I03 §7 |

---

## 3. Levers proposed or tried in these notes, and the user's rules that constrain speed work

Status key: **TESTED+** tested on GPU, kept; **TESTED−** tested on GPU, negative or rejected after the test; **CPU** built and tested on CPU only; **UNTESTED** proposed, never run; **REJECTED** rejected without a GPU test (reason given); **BLOCKED** cannot work on our stack (source check).

### 3.1 Engine configuration, memory and scheduling

| ID | Lever | Status | Result / reason | Source |
|---|---|---|---|---|
| L1 | INT4 + DFlash2 width 8 with fp16 SSM (config D → W1) instead of INT4 alone, MTP or fp32 SSM | TESTED+ | D 404 tok/s at real 12 = 1.39× A, ≈1.3× vLLM; C (fp32 SSM) capped at 6; adopted as W1 | E11-03 §2, §4 |
| L2 | `extra_buffer_lazy` + `--max-mamba-cache-size 56` + prefill CUDA graph off + `--mem-fraction-static 0.83` | TESTED+ | 12 real running, KV 881,336 (from 387k), 2.2 GB free | E11-04 |
| L3 | fp16 vs bf16 vs fp32 SSM state | TESTED+ (fp16) | W1 fp16 416.5 vs W2 bf16 412.1 tok/s; fp32 gave KV 599k; "copying dfranzen's bf16 SSM" rejected, keep fp16 | E11-04; I01 §2, §4 |
| L4 | `--speculative-draft-model-quantization unquant` for the BF16 drafter | kept (mandatory) | "else acceptance ~1.0" [C]; logged as `None` in server_args, which is correct | I01 §2; I10 §6 |
| L5 | `--max-mamba-cache-size` 60–64 (12 × 5) / 64 "so that 12 really run (+8 %)" | UNTESTED | E11-03 §5 proposal and E11-05 rev 3 candidate; "Needs a short GPU check"; later superseded by the delayer finding | E11-03 §5; E11-05 |
| L6 | Server max-running 13 with client 12 (the "11-of-12 workaround") | TESTED (used since E11-06) | Still 107 of 128 queue lines at running 11 in E11-06 and 11 + 1 queued 50 % of EVAL30 / 25 % of G4B; under the delayer threshold (3 at mrr 13) it does not remove the delay | I06 §2; I10 §2; I11 §3 |
| L7 | `--min-free-slots-delay 1` ("nodelay" arm) | UNTESTED (arm in the E14-A notebook) | Arm order W1 → +#44401 → +nodelay → +graphs; "kept only if it passes the latch test and the speed rule"; whether ≤ 1 disables the delayer is **not confirmed** (source read cut in the transcript) | I11 §4, §5, §7 |
| L8 | mrr 14 with client 12 | UNTESTED | Delayer not active at mrr 14/15 in the code test; needs a memory check first (mrr also sizes the Mamba state) | I11 §3, §5 |
| L9 | #44401 (+#41872) backport (held-row double count during chunked prefill) | CPU | 6 hunks, 36/36 tests on 6 runs; hash-guarded, all-or-nothing; adopt rule D3: 12 real running during chunked prefill, no negative free slots, tok/s ≥ baseline − 2 %; "The backport alone probably does not fix the symptom"; research rank 1a expected ≈+8 %; E14-S +3–8 % "now belongs to nodelay"; not in 0.5.22, only in nightly ≥ dev20261016 (manylinux_2_39, rejected by pip on glibc 2.35) | N13 §0; I09 §2; I11 §2–§4, §6 |
| L10 | CUDA graph decode bs 1..13 (`--cuda-graph-bs-decode`) | UNTESTED | Adopt if capture OK, no OOM, tok/s at 13 ≥ at 12 (D4); rev 6c left the list unchanged ("9 and 11 pad up … avoids capture memory", "numbers show no need") | N13 §0; I03 §4; I05 §4 |
| L11 | DFlash2 width 6 / 4 | UNTESTED | Estimates: ≈515 tok/s at width 4 (GPT) vs 550–650 (Claude); adopt if ≥ +5 % tok/s and the correctness gate passes for that width (D7); width 3 (≈2.5 tokens per step) and width 2 (crashes at graph capture) excluded; drafter bidirectional within a block, checkpoint trained at block 8 | N00 §2; N13 §0; I09 §2; I10 §4 |
| L12 | NCCL env arms: baseline / `NCCL_P2P_LEVEL=SYS` (or PHB, only if `can_device_access_peer` and topology allow) / `NCCL_SHM_USE_CUDA_MEMCPY=1` | UNTESTED | NCCL bench planned at the real sizes (983,040 B × 140 in a CUDA graph, 11.9 MB all-gather, 64 MiB point); adopt if ≥ 20 % faster at 983 KB with the right sum, then ≥ 5 % tok/s (D2); `SHM_USE_CUDA_MEMCPY` helped on one box and halved 1 MB speed on another; P2P can hang silently under IOMMU/ACS; do not touch `NCCL_PROTO` / `NCCL_ALGO=Tree`; never `NCCL_SHM_DISABLE=1` / `NCCL_P2P_DISABLE=1` | N10 §B23; N13 §0, §1; I10 §4, §7; I11 §6 |
| L13 | Custom all-reduce at TP4 | REJECTED | Disabled by SGLang on > 2 PCIe GPUs; forced: +14.8 % on a 3090 box but greedy diverged on 4 of 5 prompts ("unusable without a parity gate"); vLLM era: 0.9995× (no effect) | E11-03 §1; I01 §2; I09 §6; I11 §4 |
| L14 | PCIe-IPC all-reduce (`SGLANG_ENABLE_PCIE_IPC_ALLREDUCE`) | BLOCKED (deferred P3) | Dead code with FlashInfer 0.6.18 (no `pcie_ipc_ar.py`); needs FlashInfer ≥ 0.7 = SGLang 0.5.22 and a wheelhouse rebuild (≈220 wheels / 5 GB), plus P2P/IPC on Kaggle (unknown) | I10 §4, §6; I11 §4 |
| L15 | 2 × TP2 replicas (custom all-reduce allowed at TP2) | UNTESTED → nearly REJECTED (P5) | First offered as an optional E11-04 arm; note 00: build only if the measured gain is > 1.3×; revised: ≈170k KV per replica, 0.7–1.0× of TP4, would worsen Duck prefix thrash; router wheel not in the wheelhouse (retries 5× by default); PP and DP-attention blocked for DFlash2; GPT: only if Stage A shows communication > ≈30 % of the step; breaks the rev 6g "last version" rule | E11-03 §5; N00 §3; I09 §4; I10 §4; I11 §4 |
| L16 | int8 Mamba checkpoints (`--enable-int8-mamba-checkpoint`, `int8_mamba_ckpt_size`, `mamba_track_interval` default 256) | UNTESTED (arm D8) | ≈2× checkpoint capacity against deeper misses; lossy, so gated by resumed-request parity; incompatible with HiCache; keep SSM dtype fp16 | N13 §0, §5; I10 §7; I01 §7 |
| L17 | ReplaySSM spec-verify (`--enable-linear-replayssm-spec`) | BLOCKED | E11-03 proposed it as a candidate arm (removes ≈2 GB scratch, keeps fp32 state); later: it raises for non-KDA models (`kv_cache_configurator.py` L1125-1136) | E11-03 §6; N05 §B5; I10 §3, §4 |
| L18 | `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` (lazy 4 → 3 slots per request) | UNTESTED | — | E11-03 §6; N05 §B4; I11 §5 |
| L19 | DSpark drafter (RedHat-recommended, 7 draft tokens) | UNTESTED | vendor accept length 3.62 vs DFlash2 4.80; needs block + 1 state slots | N05 §B3 |
| L20 | NEXTN/MTP on SGLang INT4 | partly TESTED (E11-03 B, capped at 10) | 55.1 / 205 / 297 tok/s at 1 / 4 / 8; research: "373.7 at 10"; "DFlash2 vs NEXTN MTP at 12 real running (never measured)"; research rank 2 | E11-03 §2; I09 §2; I11 §5 |
| L21 | Load the RedHat calibrated FP8 KV scales (32 tensors, 0.021–0.254) instead of 1.0 | UNTESTED | Both engines log "defaulting to 1.0"; accuracy effect unverified | E11-04; N05 §B6; I11 §5 |
| L22 | SGLang HiCache (KV tier in host RAM, 202.6 GB) | UNTESTED → deprioritised | note 05: "Never considered" (club-3090 4×3090 927 tok/s at c16); later: KV max 0.44 after rev 6 → gain ≈0; open hybrid bugs #39830 (0/10 vs 10/10 answers), #45400 | N05 §B2; I10 §4; I11 §4 |
| L23 | Newer engines (SGLang 0.5.22 / nightly, vLLM 0.31+) | UNTESTED | No one checked changelogs or offline wheel feasibility until E14; 0.5.22 lacks #44401; nightly wheel tag rejected on glibc 2.35; user prefers the newest stack (memory) | N05 §B9; I11 §4 |
| L24 | FR-Spec hot-token map for the drafter | BLOCKED | `--speculative-token-map` is silently ignored by DFLASH | I10 §4 |
| L25 | `--enable-mixed-chunk` | REJECTED | #43514: DFlash acceptance falls to 0 / token-0 batch poisoning | I10 §7; I11 §4 |
| L26 | `num_continuous_decode_steps` | REJECTED | no-op in 0.5.21 | I11 §4 |
| L27 | torch.compile | REJECTED | (listed under "do not use") | I11 §4 |
| L28 | `--dtype float16` | REJECTED | Marlin atomic-add path is off on sm < 9 | I11 §4 |
| L29 | `--enable-deterministic-inference` | not used | unverified with DFLASH + mamba | I11 §7 |
| L30 | logits vocab-parallel; GDN kernel sweep | UNTESTED (P3) | — | I11 §4 |
| L31 | Profiler window (`POST /start_profile {num_steps: 20}`), kernel-category split | UNTESTED (D1) | Split of the 115–140 ms step unknown; one profiler window per process; publish tok/s only from non-profiled windows | N13 §1; I10 §7; I11 §7 |
| L32 | Temperature 0.6 vs 1.0 | TESTED (inconclusive) | E11-03 temp arm capped at 6 (T 0.6 0.85×); T 1.0 lowers acceptance (4.60 → 4.34); user took T 1.0 for Duck (rev 5 B9a) | E11-03 §3; I01 §6; I00 §1 |
| L33 | `--schedule-policy lpm` | in use | Already in the G4B launch flags; a note 05 ARC-3 lever; earlier "used blindly" rejected (#31954 cold-request starvation, claimed) | N05 §B1; I01 §4; I06 §2 |
| L34 | `--page-size 64`, `--mamba-track-interval 64` | UNTESTED | ARC-AGI-3 winners' levers (note 05); harness config `E12_MAMBA_TRACK_64 false` | N05 §B1; I02 §2 |
| L35 | Larger `--max-mamba-cache-size` or different `--chunked-prefill-size` (#36935 regime 211.6 vs 56) | UNTESTED (G4-9: only after the G4-8 measurement) | — | I06 §4; I07 §4 |
| L36 | `--disable-radix-cache`, mamba-strategy switch, kernel patches | REJECTED for now (G4-9 "not now") | speculative | I06 §4; I08 §4 |
| L37 | Franzen patch #4 (keep the end-of-prefill Mamba checkpoint and refresh its LRU) + `--watchdog-timeout 1800`; E14-M Mamba protection | UNTESTED | "only if the prefix hit rate at 10 streams falls below rev 5"; Franzen used it to limit loss to the previous reply, "He did not remove the cause"; P3 in the report | I02 §4; N11 §B; I11 §4 |
| L38 | Prefill CUDA graph off | TESTED+ | frees 2.60 GB for KV | E11-03 §6; E11-04 |
| L39 | Speculation off (target-only) | TESTED (E11-03 A) | 289 tok/s at 12 vs 404 for D; kept only as a parity reference | E11-03 §2; I10 §2 |

### 3.2 Startup and load path

| ID | Lever | Status | Result / reason | Source |
|---|---|---|---|---|
| L40 | Ranged parallel copy of the weights to `/dev/shm` → `/tmp` → in-place warm (E14-0 rev 2: worker ramp 4 → 8 → 16 → cap 24 with a ≥ 15 % gain rule, `.partial` + atomic rename, per-range blake2b + re-read, sha256 per shard, safetensors no-hole check, cgroup + MemAvailable guard, phase record, `E14_LOADPATH=0` off switch, fail open) | CPU (56 tests, 6/6 runs) | never on Kaggle; real GCSFUSE speed with 24 readers, `/dev/shm` size, tmpfs load speed unknown | N07; N10 §B; N11 §A |
| L41 | Stall watchdog instead of a fixed launch timeout (progress = log growth, `rchar`, page faults; 900 s phase-1 / 720 s phase-2 stall limits; give up at `SGL_GIVE_UP_AT`, T0 + 75 min) | CPU | page faults needed because SGLang mmaps safetensors; first bs=12 capture is ≈6 min silent | N07; I10 §7; I11 §7 |
| L42 | First-launch timeout 30 → 75 min | proposed 10-22 17:46, folded into E14 P0 | No user answer recorded; the old `VLLM_LAUNCH_TIMEOUT_SEC = 30 * 60` / `SGL_LAUNCH_TIMEOUT_SEC 1800` remain in the E12 builder | I09 §4, §5, §7 |
| L43 | JIT cache save/restore (`SGLANG_JIT_CACHE_DIR`, exclude `nv/` and `cuda/` (#42917), 13-field manifest, known-answer greedy check) | CPU (save only on the first E14 run) | saving ≈360 s / "−4 to −5 min" / "≈6 min" [E]; earlier rejected as "outside the agent bundle" (rev 6) | N07; N11 §A; I02 §4 |
| L44 | Stock loader flags: `--weight-loader-prefetch-checkpoints`, `overlap` load mode, `fastsafetensors` | REJECTED | prefetch splits files per rank (rank 0 would read 18.6 GB alone); overlap mode rejects spec, TP4, quantization and draft; fastsafetensors not in the wheelhouse | I11 §4 |
| L45 | Page-cache warming with `cat` | REJECTED as a fix | GCSFUSE reads do not fill the page cache (Jeroen) | I09 §6 |

### 3.3 Agent-side (Duck) and Poetiq-side levers that change effective throughput

| ID | Lever | Status | Result / reason | Source |
|---|---|---|---|---|
| A1 | Admit no more live branches than KV holds (rev 4: ≤ 6 live, task-major) | TESTED+ | EVAL6: 325 tok/s effective, cache 78.8 %, 0 timeouts (vs EVAL30 187 / 167.8) | E12-E6; I00 §4 |
| A2 | 12 streams instead of 6 or 10 | TESTED (G4B, with the probe bug) | 11 running 370.9 tok/s at 20–40k vs 321.4 at 6; 6 → 12: 368 → 431 (+17 %) | E12-G4B F1; E12-R5 |
| A3 | 10 qualified streams with KV fraction 0.9 (R6-S5) | REJECTED by the user | user: "১২ stream- নাও" | I03 §1, §4 |
| A4 | KV fraction 1.0 (73,444 per stream) vs 0.95 at 12 streams | TESTED (1.0 in G4B) | hidden-run default (1.0 vs 0.95) left open; G4B KV max 0.44 | I04 §4; I07 §4 |
| A5 | Queue/retraction capacity probe with step-down 12 → 10 → 8 → 6 | TESTED− | −13.3 % at 20–40k, 27.7 % of time at 6 running; removed | E12-G4B F1; I06 §4 |
| A6 | KV-only admission gate (G4-1): no new task/branch while `sglang:full_token_usage` ≥ 0.95, 12 workers fixed, 5 s poll, fail-open, idle override (H6-12) | CPU (rev 6e–6g) | Offline replay on 7 real logs: holds only in E12 EVAL30 rev 1 (219 samples, 18.2 min); never GPU-run (rev 6g G4B ERROR) | I06 §4; I08 §2, §4; I09 §2 |
| A7 | Raw reasoning in history (`E12_RAW_REASONING_HISTORY=1`) | CPU (fake server 24/24; unit tests 15/15) | Expected up to 217.5k fewer re-prefill tokens at G4B scale (≈146 s), working estimate ≈+10 % effective; default ON in E14; verify with `return_output_ids_in_sglext` replay (D5) | N11 §B; N12; N13 §0 |
| A8 | Token-level continuation (SGLang streaming sessions with a 300 s idle reaper, or token ids via the native API) | UNTESTED | "Complete" fix for re-prefill; bigger Duck-side change; needs approval | N06; N11 §B |
| A9 | Exact token-id recording (`--return-input-ids` / `--return-output-ids`, per-request `return_*_ids_in_sglext`, headers `x-sglext-return-*-ids`) | available (0.5.21) | for the E14-A replay client; Duck does not store raw responses | N10 §A1; I10 §7 |
| A10 | Per-request deadline clamp (R6-S6-a), verify clamp, one join cutoff (deadline + 60) | TESTED+ | G4B: 91 clamps, 0 stragglers, finished 9 s early | I06 §2; I07 §2 |
| A11 | Duck request timeout 900 → 1,800 s | TESTED+ | EVAL6: 0 timeouts (EVAL30 had 40) | E12-W W6; I00 §3 |
| A12 | Per-branch soft cap 100k | TESTED | overshoots to 159k | E12-W W7 |
| A13 | Lower soft cap (≈35k per attempt, Kris Dreemurr budget) | REJECTED (simulated) | tokens per task 141k → 91k, accuracy 4/6 → 1/6 (xhigh); still 21–25 h for 240 tasks | E12-R5; I01 §4 |
| A14 | Hard per-attempt budget (clamp `max_tokens` to the remaining budget) | UNTESTED | "would be needed to really bound tokens"; under the 60k floor it needs approval | E12-R5; I01 §4 |
| A15 | Output per request 60,000 → 49,152 (B5) | TESTED+ (rev 5) | approved evidence-based cut; longest productive reply 44,779 | I00 §4 |
| A16 | Request caps 12,288 → 24,576 → 32,768 (R6-Q1); b1 first request 24,576 (G4-2) | TESTED (R6-Q1 in G4B); G4-2 built, not run | 32,768 never used; all 4 b1 first requests were cut at 12,288 | I06 §2, §4 |
| A17 | 28k soft first exposure + fair share (R6-S3); minimum stage-2 chunk 12,288 (G4-17) | TESTED (28k, fair share) / built (G4-17) | fair share inert (shares 3.8–7.5k < 24,576); sim: chunk 12,288 → b1 started 176–181 times, 24,576 → 103 | I07 §2; I08 §2, §4 |
| A18 | Selective / prioritised branch 1 | CPU replay | xhigh 5/8 at 498,253 tokens (−41.1 %), low 9/19 at 1,094,372 (−37.2 %, lost 135a2760); built as "prioritised by evidence class, never skipped" | E12-R5; I01 §2; I04 §3 |
| A19 | Stable prefixes: same tools on every request, same template kwargs, no tool-free request, abort only after the stream ends | kept (R6-D2) | measured killers: tools omitted 99 → 15 %, template conditional 75 → 45 %, abort without wait 0 % vs 52–99.7 % | I02 §2; I03 §7 |
| A20 | Context drain / trims: Duck drain 16,384 tokens; block-level trims, oldest reasoning-only block first (R6-D3); GPT-B drain 11,168 and `ARC2_CONTEXT_DRAIN_FLOOR` rejected (trims every turn kill the prefix cache); Franzen drain hysteresis at 58k | built | — | I00 §4; I02 §4; I03 §4; I01 §6 |
| A21 | B3 real-tokenizer prompt estimate | TESTED+ | median +0.4 %; G4B 0.998; removed the system-only resets | E12-R5; I06 §2 |
| A22 | Loop control: T 1.0 + prompt edits (B9a); LOOPSTOP detector; presence penalty | TESTED+ (T 1.0) / deferred (LOOPSTOP 5/5 caught, 0 false positives) / REJECTED (penalty: no-op under DFlash2, needs a 2-line patch) | loops 29.4 % → 0–3 % | E12-R5; I00 §4; I03 §4 |
| A23 | Early stop of hopeless thinking / REFRAIN / text early exits | REJECTED | AUC 0.51–0.64 (0.63), saves 3–4 %, loses 18 % of train-passes | I00 §2; I01 §4 |
| A24 | Rolling summaries / stripping reasoning from history | REJECTED | Franzen reports a large regression | I02 §4 |
| A25 | Forced or required `tool_choice` | REJECTED | Exp 4 HTTP 500 + 59 empty submits; #43514 poisons DFlash batches | I01 §4; I02 §4 |
| A26 | p90 last-admission cutoff | built, OFF by default | would idle 12 workers for ≈15 min; requests already deadline-clamped | I03 §4; I04 §4 |
| A27 | Code-first facts view (rev 6) + token-triggered nudge at 24k via the tool result (G4-6 / H6-1) | TESTED (facts view in G4B) / built (nudge) | first python call at 46 tokens but 7/12 b0 never wrote a transform; observed first-check median 29,384.5 (vs ≈55k rev 5); nudge reaches 7/12 b0 inside stage 1 in replay | E12-G4B F2; I06 §2; I08 §4 |
| A28 | Lower effort (low) | TESTED− | not cheaper per branch (72.7k vs 70.5k tokens; 449 vs 127 requests); LB 17.50; "Lowering effort: rejected" (ARC Prize medium 13.2 % < low 22.8 % < xhigh 42.4 %) | E12-R5; I09 §2; I11 §4 |
| A29 | Lower thinking cap (S49, 49k) | TESTED− | LB 32.08 vs 60k 34.58; 83/88 forced closes in E11-05 | E11-05; I10 §2 |
| A30 | Stable example order (Poetiq, E11-06) | TESTED+ | cache 11–18 % → 87.2 %; throughput 328.9 → 379.9 tok/s | E11-05; E11-06; I02 §2 |
| A31 | Drain that waits for each aborted stream (E11 rev 2 shim); drain margin 12 → 8 min | TESTED+ / proposed | phase-2 cache 52–99.7 %; answers 38–83 s (83 s worst) | E11-05 |
| A32 | Cache diagnostic after solving, window starts after preflight (H6-3, H6-6) | built | the diagnostic could take up to 240 of 2,063 s (≈12 %) and up to 2.7 % (240 / 9,000 s) | I08 §2, §4 |
| A33 | Pause / resume of branches | context | host state ≠ KV resident: ≈30–35 s cold prefill per 50k on resume | I02 §6 |
| A34 | Token-efficient fine-tunes (Swift-Qwen3.8-27b, Swift-1.5, ThinkingCap) | REJECTED / optional | licences not OSI (prize eligibility doubtful); realistic −10 to −25 % thinking at ≈60k; ARC accuracy and DFlash2 acceptance on a tuned target unknown | N05 §B7; I10 §4; I11 §4 |

### 3.4 User decisions and rules that constrain speed work (verbatim where the notes give it)

| # | Rule / decision | Quote (as recorded) | Effect on speed work | Source |
|---|---|---|---|---|
| U1 | **60k thinking floor; cuts only with evidence and approval** | Memory: "thinking floor 60,000 (user 2026-10-18; S49 49k LB 32.08 < 60k 34.58); applies to THINK_BUDGET + admission guard + rescues; approval needed for any cut; gate-4b time-share profile approved 2026-10-21". 10-20 17:50: "এখন যে পরিবর্তন আমরা ইভিডেন্স পাবো সেটা করবো,এটাও প্রয়োজনে।" (make evidence-backed changes, this one too if needed → B5 49,152). User's own map: "মোট চিন্তা ৬০k বা তার বেশিই থাকবে। আপনার অনুমোদন লাগবে". Report: "60k floor-এর নিয়ম অপরিবর্তিত থাকে". Note 05: "The 60k thinking floor (user rule) is a budget rule. A model that needs fewer tokens is a different lever, but the user decides." | speed must come from throughput or token efficiency, not from cutting thinking below 60k; any per-request or per-branch cut needs a new approval (approvals do not transfer: "12,288 = approved exception to the 60k floor, user 2026-10-19" applied only to rev 1 / EVAL30) | MEMORY.md; I00 §1; I03 §1; I11 §1; N05 §B7; I01 §6 |
| U2 | Gate 4b time share approved | "অনুমতি দিলাম।" (permission granted) and "বাকি টার ও অনুমতি দিলাম" (permission for the rest too: caps, p90 cutoff, target-only parity, engine-setting changes) | a ≈43k-tokens-per-task screening profile is allowed in commit mode only | I03 §1 |
| U3 | **12 streams** | "১২  stream- নাও" (take 12 streams), 10-21 09:26, overriding the 10-stream design | concurrency target is 12 client streams | I03 §1 |
| U4 | **KV-only admission gate, simple rule** | "এই probe stream 12 → 10 → 8 → 6 এটা এত জটিল করার কোনো মানেই ছিলো না।শুধু ৯৫% এর বেশি ক্যাশ থাকলে নতুন কোনো টাস্ক ডুকবে না,এটা সিম্পল একটা শর্ত দিলেই হতো।" (no point making the probe so complex; if the cache is above 95 %, no new task enters) | 12 workers fixed; gate only on `full_token_usage` ≥ 0.95; no step-down | I06 §1; I08 §1 |
| U5 | **One notebook** | "একটাই নোটবুক বানাও।আর সফল হলে তোমাকে আর একটা ফিক্স করে ১২ ঘন্টার ২ থেকে ৩ টা রান দেওয়াবো ক্যাগলে" (build just one notebook; if it succeeds, one more fix, then 2–3 twelve-hour runs) | E14-A = one notebook (measurements + arms + 25-min Duck run), not A1/A2 | I11 §1 |
| U6 | **No base-protection excuses** | "E14-A এটার পুরো প্ল্যান দিও।আর পুরো পাইপ লাইন পরিবর্তন  করা লাগলেও করিও,বেস ভাঙবে এই সেই বলে আমার এক্সপেরিমেন্ট মাটি করিও না আবার।বেস সংরক্ষিত আছেই সমস্যা নেই।" | proven fixes default ON in E14; risky engine levers are measured arms, "not refused in advance" | I10 §1; N13 header |
| U7 | Rev 6g is the last version before GPU | "দুই তিন দিন ধরে এভাবে লুপে পড়লে আমরা হেরে যাবো,এটাই লাস্ট ভার্সন মনে রাখ" | Solver/Duck code changes need approval; "Engine flag আর load cell সেই নিয়মের বাইরে, কিন্তু production notebook বদলাতেও অনুমতি লাগে" (engine flags and the load cell are outside the rule, but changing a production notebook still needs permission) | I09 §1; I11 §1 |
| U8 | The engine-speed goal | "আমাদের ইঞ্জিনের কোন সমস্যা এর গতি ৬০০–৮০০ এ যেতে দিচ্ছে না? সেটা কি বের করা যাবে?" (what keeps our engine from 600–800 tok/s?) | target range 600–800 tok/s | I09 §1 |
| U9 | Create Experiment 14 | "ডিপলি রিসার্স করো।আর একটা এক্সপেরিমেন্ট ফোল্ডার খুলো,আমরা Optimized Experiment এর দিকে যাবো। … সব নোট করে রাখিও,ভুলে যেও না।" | every analysis and GPT paste saved in E14 Notes | I09 §1 |
| U10 | **Full coverage with the same model and Duck** | "একই Qwen3.8-27B + Duck রেখে, reasoning ও tool-use optimize করে প্রতি task-এর token consumption কমানো, 240-task coverage অর্জন করা এবং accuracy বাড়ানো।"; "এই মডেল ও এই ডাক দিয়েই ৮০+ এভালুশনে আর ৭০+ এলবি পাওয়ার প্রমাণ আছে।তারা পুরো কভারেজ দিয়েছে।আমরাও পারবো ইংশা আল্লাহ।" (insult omitted); doc F7: "Duck runs on every task (full coverage)" | never propose running Duck on fewer than all tasks; throughput must deliver 240-task coverage | I01 §1; I00 §1; I02 §1 |
| U11 | Trust the user's observations | "তুই যেখানে দেখতে পারিস না,সেখানে তর্কে যাস কেন,বিশ্বাস করে নে।মেমরিতে সেভ করে রাখ।" + "ওরা সবাই Qwen3.8 + Optimized Duck Use করছে।" | top teams = Qwen3.8 + optimized Duck at full coverage (≈565–729 min runtimes) | I00 §1, §2 |
| U12 | **Bundle changes, no one-change runs** | "আমরা একটা রানে যত সমস্যা পাবো,সবগুলো ফিক্স করে,আবার রান দিবো,ছোট পরিবর্তন করে করে রান দেওয়ার টাইম নেই + জিপিইউ নেই এত বেশিও।"; report: "প্রতিটি submission-এ একাধিক প্রমাণভিত্তিক পরিবর্তন একসাথে যাবে, একটি করে ablation নয়" | engine arms are measured inside one run; submissions bundle levers | I00 §1; I11 §1 |
| U13 | Past negatives are not verdicts | "আগে আলাদা এক্সপেরিমেন্ট ছিলো।সেখানে কাজ করেনি bole ekhane kaj korbe na emon na..." | old negative results (e.g. vLLM custom AR 0.9995×, TP2 17k KV) only shape safeguards | I01 §1 |
| U14 | **Written estimate before every GPU run; explicit go** | "এর আগে বলিও কোনটা কত সময় লাগবে?"; after the rev 6g estimate failed: "তুই না বললি হিসাবে রান লাগার কথা ৫৫–৬২ মিনিট? এবার বেশি কেন লাগতেছে ,…, GPU quata ar koto ache?"; report: "সব GPU রানের জন্য ব্যবহারকারীর স্পষ্ট অনুমতি লাগবে … সাথে লিখিত wall time, quota আর stop point, এবং রানের আগে quota পরীক্ষা।"; "অনুমতি ছাড়া কমিট করবি না" | every run needs worst-case wall time (slow storage), quota at 2× per account before/after, a stop time, the notebook sha, and the user's go | I05 §1; I09 §1; I11 §1 |
| U15 | Deadline pressure | "এই দিকে আর মাত্র ১১ দিন আছে।" (only 11 days left, 10-22); deadline 2026-11-02; ≈1 submission per 2 days, 12–16 h turnaround | GPU measurement minutes are scarce | I09 §1; I11 §2 |
| U16 | Full reports, read everything | "তুই রিপোর্টে অনেক কিছু স্কিপ করিস সর্বদা,এটা আর কখনোই করবি না।পূর্ণ report always" | — | I09 §1; I10 §1 |
| U17 | Temperature 1.0 for Duck | "(a) temperature 0.7 থেকে 1.0 … এটা আমি নিলাম।" | lowers DFlash acceptance (4.60 → 4.34) vs T 0.6 | I00 §1; I01 §6 |
| U18 | Raw-reasoning fix approved behind a switch | "Duck-এর এই সংশোধন করব কি? / "আগের message বদলানো" সমস্যাটা fake server দিয়ে CPU-তে খুঁজব কি? / শুরু করো" | built off by default in rev 6g, default ON in E14 | I10 §1; N12 |
| U19 | Accounts | "আলাদা আলাদা মানুষ।একটা টিম।আর শুরু করো" (three different people, one team) | foysal, hosen42, esfersami50 can each run; quota per account: foysal 4.71 h, hosen42 8.63 h, esfer 0 | I10 §1; I11 §2 |
| U20 | Partial approvals must be respected | "OK ১ থেকে ৫ করে bolio" (rev 6d items 1–5 only; PARITY fixes not taken); "সবগুলোই nio...update" (all G4 items); "Ok update start" (assistant mapped it to four choices) | the PARITYR6 notebook stayed NO-GO; target-only parity never ran | I04 §1; I07 §1; I08 §1 |
| U21 | Notes only when asked | "তোমার আইডিয়া বদলানো লাগলে বদলিও,শুধু জিপিটির পরামর্শ দেখে নোট আপডেট করো।" | — | I10 §1 |
| U22 | Memory rules active in this session (from MEMORY.md, not from the listed files) | "Short GPU smoke over audits — user 2026-10-23: validate with a ~30-min hard-stop GPU commit, not multi-day audits; 1-2 GPT audits max; submit daily"; "Prefer latest stack"; "Verify real concurrency — engine sweeps must assert real running == level + read server caps"; "Pre-run capacity check"; "Use D: drive for scratch"; platform limits "12-16 h submission turnaround …, 1 pending per team, ~1 sub / 2 days" | short GPU smokes preferred over long audits; newest engine versions preferred | MEMORY.md |

---

## 4. Contradictions between notes (both sides quoted)

| ID | Topic | Side 1 (quote, source) | Side 2 (quote, source) | Current reading |
|---|---|---|---|---|
| C1 | **Root cause of "11 of 12 running"** | E11-05: "Likely cause: the PrefillAdder gate (`schedule_policy.py:852-857`, mamba-recoverable slots or `rem_total_tokens`). **Not verified.** … Candidate fix: `--max-mamba-cache-size 64`." I01: "Candidate fixes: `--max-mamba-cache-size 64`, or server 13 / client 12 (already used by E12)." I09: research report "source-verified chunked-prefill double count is better supported than the earlier mamba-gate guess"; ranking "1a. W1 + PR #44401, about +8 %". N13 D3: "#44401 backport (11-of-12 admission)". | I11 §3: "The report (05:50) said the cause was 'প্রায় মীমাংসিত' (nearly settled) as the #44401 double count … The backport agent found that 0.5.21 has a **DFlash-only min-free-slots delayer** … threshold is `min(4, max(2, (mrr+5)//6))`, which is 2 at mrr 12 and **3 at mrr 13** … **The backport alone probably does not fix the symptom.**" | Latest: delayer; "+#44401 alone is expected to show no gain"; the mrr-13 workaround does not avoid the delay; unverified on GPU |
| C2 | EVAL30 deadline overrun | I01 §2: "**finished 1,412 s past the deadline**" | I02 §3: "The assistant subtracted the relative `deadline_in_sec` (13,367) from the absolute `finished_sec` (14,779). The true overrun is 679 s." | 679 s |
| C3 | EVAL30 effective throughput | E12-E30: "generation over the solve window was about 187 tok/s (2.36 M tokens / 12.6 ks)" | I11 table: "EVAL30 \| 12 \| 14,046 \| 2,357,584 \| **167.8**"; "An earlier figure of 187 tok/s for EVAL30 is the same run under a different window definition. The new window includes 679 s after the deadline." | Window definitions differ (14,046 − 12,600 = 1,446 s, which is more than the 679 s given as the reason) |
| C4 | EVAL30 decode rate | E12-E30: "decode itself ran at about 420 tok/s at 11-12 running, even at ≥ 60k context"; I00: "decode 431 tok/s at 12 running"; N05: "decode ~412–431 tok/s" | I10 / I11 table: decode-only "~376" / "376" | Different definitions (decode-line rate at 11–12 running vs decode-only over the whole window); not reconciled |
| C5 | G4B effective throughput | E12-G4B: "572,035 = **47.7k per task**, i.e. 277 tok/s over the window"; I08: "572,035 / 2,063 ≈ 277.3 tok/s" | I11 table: "G4B rev 6d \| 12 \| 2,050 \| 572,035 \| **279.0**" | Window 2,063 s vs 2,050 s |
| C6 | G4B weight-load seconds | I10: "10-21 load: 669.96 s total" | I11: "**669.26 s** in G4B on 10-21"; I09: "The weight load alone was 670 s" | ≈670 s |
| C7 | Rev 5 cache percentages | I02 §2: "**Cached fraction 85.4 % / 91.0 %**"; I01 §2: "Weighted cached fraction 85.4 % / 91.0 %" (same file also lists "cache 84.3 %" and "cache 88.7 %") | I06: "Aggregate `usage_all.cached_pct`: xhigh **84.3 %**, low **88.7 %**. The README had used request-level 85.4 % / 91.0 % (wrong denominator)." I08: 84.26 % (5,195,178 / 6,165,603), 88.70 % (15,951,513 / 17,983,349) | Aggregate 84.3 / 88.7 is the correct comparison base |
| C8 | Rev 5 uncached tokens | E12-R5: "uncached prefill is now 0.97M vs 0.85M generated (xhigh)" (aggregate 6,165,603 − 5,195,178 = 970,425; low 17,983,349 − 15,951,513 = 2,031,836; I00: "Uncached prefill '2.03…'" cut) | I01: "Uncached tokens 888,297 / 1,576,650"; I02: "uncached about 888k / 1.577M"; I01 §6: "more uncached prefill (1.58 M vs 0.89 M)" | Two denominators; the uncached ÷ generated ratios 1.15 / 1.17 (I10) match the aggregate figures |
| C9 | W1 KV pool size | E11-04 / E11-05: "**KV 881,336 tokens**"; rev 6c window "1.0 × 881,336 / 12 = 73,444" | E12-SM: "KV 866,744 tokens"; I06: "**The KV pool on Kaggle was 866,744 tokens, not the assumed W1 881,336.**"; G4B "rev6_window 72,228 ≈ 866.7k / 12"; I10: "The SGLang KV formula predicts 886,801 tokens vs 881,336 measured" | Resolved by I09: "W1 KV 881,336 tokens (866,744 at server max 13)" |
| C10 | W1 decode rate by context | E11-04 (W1, 12 real, PROD): "458 / 413 / 388" (<10k / 10–20k / 20–40k); E11-05 (11 running): "429 / 372 / 340 / 323" | I11: "Decode-line tok/s, E11-06 at 12 running: <10k **500.1**, 10-20k 460.3, 20-40k 409.0, 40-60k 389.4"; I09 research quotes these as W1 "Decode at 12 running: 500 under 10k context, 460 at 10–20k, 409 at 20–40k, 389 at 40–60k" | Different runs (acceptance 3.87 in E11-05 vs 4.58 in E11-06); the often-quoted "−25 % at 40–60k (323 vs 429)" is E11-05 only |
| C11 | E11-03 D step time at 12 | N00: "INT4 + DFlash2 … 39.9 ms \| 115.5 ms (E11-03 D)"; I10: "E11-03 D 115.5 (n=67)" at < 10k | I10 research: "DFlash2 W8 fp16 SSM 404 tok/s, ~137 ms per step" | 115.5 ms is the decode-line median at < 10k; 137 ms comes from the aggregate 404 tok/s |
| C12 | All-reduce count and regime | N00: "128 all-reduces per step × 10 KB per token … about 40–50 ms of a 115 ms step"; N05 A4: "applies large-message bandwidth to small messages. Small collectives are latency-bound, so this is a hypothesis" | I11 §6: "**The 64 MiB all-reduce number is a fair proxy for our 0.98 MB calls.** At that size SHM is bandwidth-bound. Only around 10 KB is the call latency-bound. This corrects note 05 A4's own correction."; I11 §3: "'128 all-reduces': the source count is 140." | 140 calls of 983,040 B; cost 35–55 ms is still unmeasured |
| C13 | Width 8 → 4 gain | N00: "going from 8 to 4 draft tokens would cut the step to about 70 ms and might give 550–650 tok/s" | N00 / I09: "GPT's geometric model gives L4 ≈ 3.0 and about 515 tok/s instead"; GPT: "550–650 tok/s: optimistic hypothesis, expected result নয়" | Both estimates; never measured |
| C14 | 2 × TP2 capacity and gain | N00: "with DFlash2 a replica has about 0.35–0.4M KV, so about 5–6 long requests"; "My estimate for us: 1.3–1.6×"; "INT4 TP2 is about 9.3 GB of weights per GPU" | I11 §3: "The TP2 KV figure of 0.35-0.4M per replica: wrong; it is about 170k."; "Expected gain 0.7-1.0×"; I09 GPT: "INT4 4.62 GB + DFlash2 1.49 GB = 6.11 GB → TP2 ~12.22 GB" | Revised side holds (≈170k per replica) |
| C15 | ReplaySSM with DFlash2 | E11-03 §6: "**Requirements:** GDN or KDA, a linear chain (topk ≤ 1; DFLASH has topk 1) … Candidate arm" | N05: "works with MTP/EAGLE but not with DFlash2 (KDA-only)"; I10: "Per-step agent was wrong that `--enable-linear-replayssm-spec` seems allowed for DFLASH. `kv_cache_configurator.py` L1125-1136 raises for non-KDA models" | Blocked (source) |
| C16 | Mamba slots per request | E11-03 §6 source table: "extra_buffer_lazy \| 4" (`kv_cache_configurator.py:2232-2261`); rev 6c: "12 × 4 = 48 ≤ 56" | I03: "The claim '4 mamba slots per request' is not verified. The gap check said 'mamba slots ≥ the donor ratio (60 per 10 streams)'; at 12 streams that ratio means 72 versus the 56 available."; I09 research: "1 / 3 / 4 / 5, plus 4 for MTP or 8 for DFlash2"; observed "36 of 56 slots" at 12 running (I10) | Source says 4; usage seen ≈3 per request; donor-ratio claim never resolved |
| C17 | HiCache | N05 B2: "SGLang HiCache … evicted Duck prefixes could live there instead of being re-prefilled. Never considered." | I11 §4: "**HiCache: low priority, high risk.** KV max was 0.44, so it would gain about 0. Hybrid host-tier bugs are open: #39830 …, #45400." | Deprioritised |
| C18 | `--schedule-policy lpm` | N05 B1: "The levers are … `--schedule-policy lpm`, `--page-size 64` and `--mamba-track-interval 64`." | I01 §4: "Rejected or deferred: … `lpm` used blindly (#31954, cold-request starvation, claimed)"; I06: G4B log shows "`--schedule-policy lpm`" already on | Already in production; no measured effect either way |
| C19 | `--weight-loader-prefetch-checkpoints` | E13-P R5: "`--weight-loader-prefetch-checkpoints` (345 s / 124.8 GB on 4xL4) \| E3 \| **K**" (worked on Kaggle, E3) | N05 B10: "whether SGLang 0.5.21 has `--weight-loader-prefetch-checkpoints`" (unknown); I11 §4: "splits files per rank, so rank 0 would read 18.6 GB alone" (rejected) | Rejected for the 27B (single 18.6 GB shard); it worked for Flash-Next's many shards in E3 |
| C20 | Copy worker count | N07 rev 1: "64 MiB ranges, 24 daemon reader threads" | N10 B4 (GPT): "Do not default to 16–32 workers; sweep first" → "adaptive ramp 4 → 8 → 16 (cap 24)"; I11 §7 report design: "64 MiB ranges, largest file first, 16-32 workers" | Rev 2 uses the ramp |
| C21 | JIT cache saving | N07: "The JIT cache saving ~360 s per cold start is the research estimate (M/E). Unmeasured." | I02: "JIT cache dataset (−4 to −5 min)"; N05 B10: "would save ~6 min" | All estimates; cold start has 372–385 s of verify-graph JIT |
| C22 | Throughput needed for coverage | I11: "240 × 60k = 14.4M tokens in about 11.46 h needs about **370 tok/s effective**"; I06: "10.81 h at 370 tok/s vs about 11.46 h available" | I09 research: "one 60k attempt for each of 240 tasks in 12 h needs ~333 tok/s"; N00: "W1 whole-run about 355 tok/s today … 1× ~14.4 M … ~60k"; I09: "Coverage needs about 240 × 33k ≈ 7.9M tokens, i.e. ≥ ~192 tok/s average" (stage 1 only) | Different windows (11.3 / 11.46 / 12 h) and targets (stage 1 vs 60k per task) |
| C23 | "W1 today" speed | N00: "W1 whole-run about 355 tok/s today" | I11 Duck table: effective 244.2 / 265.4 / 279.0 (rev 5 and G4B); 167.8 (EVAL30) | 355 matches the Poetiq runs (E11-05 328.9, E11-06 379.9), not Duck |
| C24 | E14-A run shape and stop rules | N13: "Run A1 … **capped by HARD_STOP 95 min**"; "Run A2 … **capped by HARD_STOP 110 min**"; "No ready server: if no server is ready by `T0 + 45` min … the run ends" | User (I11): "একটাই নোটবুক বানাও" (one notebook); I10 §6: "the current notebook has one 200-minute profile with a 75-minute give-up. The E14-A plan here was not final." | One notebook, 200 min, 75-min give-up |
| C25 | Cause of reply re-prefill | N06 (first version): "the chat template is NOT the cause of the reply re-prefill"; "A (text prefix) \| 170 / 170" | N11: "**So every blank line the model wrote in its reasoning is deleted from the history.**"; N12: "**The normalization is the cause of the reply re-prefill, confirmed on CPU end to end.**"; I11 §6: "Raw reasoning may not be enough … `\|trim` and the XML re-render can still diverge from the generated tokens" | Normalization confirmed on CPU; token-level proof pending |
| C26 | Share of G4B requests that lost the reply | N11: "reply_lost_at_prefill_end \| 68 (45 %) \| 217,541" | I11: "the cache hit stops at the end of the previous prompt in **72-91 %** of requests (… G4B 75.6 % …)"; "about 37 % re-prefilled replies (245k)"; N06 header: "In 72–91 % of later Duck requests …" | Different classifiers (N11 uses server usage with a 64-token tolerance and separate partial/deeper classes) |
| C27 | EVAL30 KV profile and gate holds | I06: KV "0.84 / 0.93 / 0.99", "**167**" lines ≥ 0.95; E12-G4B: "291 of 4,153 batch lines" held | I10: "KV p50/p90/max 0.77/0.95/1.0"; I08 / I09: "a hold only in **E12 EVAL30 rev 1** (219 samples, **18.2 min**, KV up to 100 %)" | Different scans and units (lines vs samples) |
| C28 | KV maxima in the other runs | E12-G4B: "it never holds back in G4B, E11-05, E11-06, rev 4 or rev 5 (KV max 0.43–0.73)" | I06 table: E11-05 0.63 (GPT 0.65), E11-06 0.69, rev 5 low 0.47, G4B 0.44; I08: "E11-05/06 (65–73 %)", rev 4 0.46, rev 5 xhigh 0.43; rev 5 low p90 0.36 (I06) vs 0.35 (I08) | All far below 0.95; exact maxima differ by scan |
| C29 | E11-05 retractions | E11-05: "Accept length median 4.55. No retractions." | I06: "Retract lines: G4B 1, E12 EVAL30 5, E11-05 1." | 1 retract line |
| C30 | How many EVAL30 branches were live | E12-E30: "30 time-sliced branches × about 50-100k context exceed the about 870k-token KV"; I00: "EVAL30 had 30 active" | I02: "EVAL30 had all 60 branches admitted and re-prefilled 8.4M"; I02 §6: "With it unset, all 480 branches go live" (at 240 tasks) | Branch 1 never started in 29/30 tasks, so ≈30 ran |
| C31 | EVAL12R5_LOW duration | E12-R5: "COMPLETE \| 123 min (solve 110 min)" | I00: "**EVAL12R5_LOW** (Foysal, low): COMPLETE after 163 min" | Notebook vs session time, not stated |
| C32 | GCSFUSE speed before the drop (amn) | I09: "NFS 0.55 → 0.026 GiB/s" | I10: "amn measured 0.95 → 0.026 GiB/s"; I11: "0.95 GiB/s (10-20) → **0.026 GiB/s** (10-21 05:37)" | 0.026 GiB/s after is agreed |
| C33 | Per-stream decode rate | E12-G4B: "11 effective streams × about 34 tok/s"; "30–44k tokens at about 25–35 tok/s per stream" | I09: "Per stream 31–35 tok/s"; I04: "a 24,576-token chunk at 25-30 tok/s per stream takes 14-16 min"; I01: "a 49k reply at ~35–61 tok/s per stream takes ~800–1,400 s" | Range 25–61 depending on concurrency and context |
| C34 | Gain from 6 → 12 streams | E12-R5: "Rev 4 measured 6 → 12 running: 368 → 431 tok/s." | I02: "6 → 12 streams: +17 % in one measurement, +30-60 % in the short sweep." | Production +17 %; short sweep higher |
| C35 | MTP and fp32-SSM DFlash2 at 12 | E11-03: "B and C at 12 were never measured." (B capped at 10, C at 6) | I10 research: "Our 4×L4 at c12: MTP 0.91×, DFlash2 W8 1.06× (fp32 SSM) and 1.39× (fp16 SSM)." | The c12 ratios for B and C were really at 10 and 6 running |
| C36 | vLLM FP8 + MTP-3 at 12 | E11-03: "~307" | I09 research: "293 at 12" | Different runs |
| C37 | Engine start time reference points | E12-G4B: "the SGLang launch took **1,197 s** ('ready in 1197s')"; E11-05: "ready in 532 s / 522 s" | I10 / I11: "G4B rev 6d \| 19:09", "E11-05 \| 8:00" (server_args → "fired up"); I09 GPT: "E11-05 7 m 39 s, E11-06 7 m 30 s, rev 6d G4B 19 m 09 s" | Different start points (launcher start vs server_args line) |
| C38 | Launch times on 10-19/20 | I09: "Our own launch times: 507–562 s on 10-19/20; **1,197 s on 10-21**" | E12-SM (10-19): "W1 up in 677 s"; E12-G4B: "against about 700 s in the rev 5 runs" (10-20); E11-05 "532 s / 522 s" (10-18) | 507–562 s does not cover SMOKE1's 677 s |
| C39 | Quota rate on 4×L4 | E13-P / I09 / I10: "about 2.0x wall", "~2× quota per wall hour", "the official rule … দ্বিগুণ quota" | I10: "An older memory line says one run used 1.05 quota-h per wall-hour, so verify each time." | 2× measured on 10-19 and 10-22 |
| C40 | Low-quota runs finishing | Memory "Kaggle quota mid-run": "runs started with low quota still finish (user-observed)" | I11 §6: "The G4B run that ERRORed had about 94 min of quota left and burned about 100 min. The memory note … is weakened by this, although E9 is a counter-example (it ran 8,879 s from 28.44/30)." | Open |
| C41 | Cause of the rev 6g G4B ERROR | N05 A6: "A slow start or pull of the pinned old image is now a candidate cause. It is not verified." | I10: "Root cause per the startup agent (20:50): **our code never ran** … Likeliest cause: a dataset/model attach or mount stall … That was then demoted to a weak candidate: staff say the image doesn't affect timers"; I11: "Unresolved conflict: note 05 (old pinned image) vs SLE (attach/mount stall)" | Ranked mount stall > old image > quota; unproven |
| C42 | Rev 6g G4B estimate | I09: "'প্রায় ৬০ মিনিট' ('about 60 min'; 55–62 min including load)" | I09: "At 18:22 it said 'তার ওপর লোডের সময়' ('plus load time'), contradicting itself."; actual ≈114 min (ERROR) | Estimate wrong |
| C43 | DFlash2 "waste" | N00: "DFlash2 verifies 8 tokens and accepts about 4.3"; I09: "~46 % 'wasted'" | N05 A2: "SGLang's 'accept len' includes the bonus token. At width 8 there are 7 draft positions, so 4.3 means 3.3 correct drafts and a draft-accept rate of ~47 %. Not all rejected work is waste either" | Bonus token included (identity verified on 5,017 records) |
| C44 | Max tokens per step at width 4 | GPT note 04: "width 4 can give more than 4 tokens per step" (as reported in I10) | I10: "The maximum equals the width." | Max = width |
| C45 | GDN projection precision | Local-validation agent: "48 GDN layers BF16" | I10 / I11: "only the parent `linear_attn`, `norm`, `in_proj_a` and `in_proj_b` are ignored, so `in_proj_qkvz`/`out_proj` are INT4. BF16 would be ~6.5 GB/GPU, not 4.62." | INT4 |
| C46 | Is 12 running slower than 11? | E12-G4B (first version): "12 running was slower (median 194–258 tok/s vs 370–406 at 11)" | E12-G4B (corrected): "That claim is withdrawn. All 29 lines at exactly 12 running fall in the first 13 minutes … admission prefill interfering with decode" | Withdrawn |
| C47 | Cost of the probe step-down | E12-G4B (first): "15–20 %" ("My first '15–20 %' mixed in the < 10k bin") | E12-G4B (corrected): "370.9 → 321.4 tok/s" (13.3 %); GPT 2: "about 13-16 %" | 13.3 % |
| C48 | #36935 as the cause of G4B cache misses | I06 / I07: "'#36935 ratio 211.6 vs 56 explains the cache misses, stronger than GPT says'" | I06: "Mamba usage never exceeded 0.70, so the slots were never full. The explanation is weak; the cause is unproven until G4-8." | Unproven |
| C49 | KV fraction for 12 streams | Rev 6 design: "Admit by projected KV: Σ(resident + reserved generation) + new prompt + new reserve ≤ 0.9 × usable KV"; window "0.9 × kv / streams" | Rev 6c: "The assistant chose **fraction 1.0 (73,444 per stream)** instead of 0.9 … so the gap check's own rule (Σ ≤ 0.9 × KV) is no longer met"; GPT audit 2 preferred 0.95 for the hidden run | 1.0 in G4B; hidden default open |
| C50 | Which KV gate governs Duck admission | E11 engine cell: "`SGL_KV_ADMIT_MAX` 0.85 (with `SGL_KV_GATE_MIN_ACTIVE` 10)" | Duck rev 6e+: "`E12_R6_KV_ADMIT` 0.95"; user rule "৯৫% এর বেশি" | I08: "Which one governs Duck admission in the notebook was not discussed" |
| C51 | EVAL30 request timeouts | E12-W (original): "~~4 x 900 s request timeouts~~" | E12-W (corrected): "40 timeouts in 21 of 31 branches (28 full 900 s + 12 deadline-bounded, about 31,000 stream-seconds)" | 40 |
| C52 | "12 real running is proven" | E11-04: "PROD itself had a share of **1.0** at 12, so 12 real running is proven." | E11-05: "**11, not 12:** 2,358 decode lines show 11 running + 1 queued … E11-04 never tested this: in its 10-min PROD window no request finished, so nothing new was admitted mid-run." | Steady state is 11 + 1 queued |
| C53 | Planned vs runtime per-stream window | I04: "12 streams at fraction 1.0 → 73,444 per stream" | I06: "Per-stream window 72,228 (GPT 2), which is 866,744 / 12" | Runtime KV must be read from the server |
| C54 | Sampling proposed for loop control | E12-W fix 3: "sampling closer to the Qwen recommendation (temperature 0.6, presence penalty)" | I00: "Claimed 'Qwen recommends 0.6' — Wrong; the HF card says 1.0 for thinking"; "Penalties don't work at all under DFlash in 0.5.21" | T 1.0; penalties no-op |
| C55 | Prefill speed figures | E11-03: "Prefill was about 1,400 tok/s cold"; I00: "Median input throughput for chunks ≥2000 is **1,459 tok/s**" | I02: "Cold prefill about 1,400-1,650 tok/s (E11)"; I10 / I11: "1,474-1,578 tok/s" (EVAL30 1,474; G4B 1,578) | Same order (≈1.4–1.6k tok/s); definitions differ |
| C56 | KV needed for production | E11-03 §4: "Production needs about 920k KV tokens" | E11-04: "That is 98 % of the 900k gate." | 900k gate vs 920k need |
| C57 | E13 Flash-Next concurrency | E13-P §2: Franzen verbatim "10 running (capped by the measured mamba/KV limits)" | E13-P rev 5 header: "4 running … Duck streams = min(4, KV // 60k)" | Rev 5 supersedes |
| C58 | Stage-1 timing at hidden scale | E12-G4B F3: "stage 1 for 240 tasks ≈ 240 × 33k = 7.9M tokens ≈ 5.9 h" (at ≈370 tok/s) | I09 sim: stage 1 finished "48 % of window (~5.5 h)" at 371 tok/s and "64 % (~7.3 h)" at 277 tok/s | Depends on the throughput assumed |
| C59 | EVAL6 rev 4 decode | I00: "Duck uses 6 streams (368 tok/s)" (the EVAL30 decode figure at 6 running) | I10 / I11: EVAL6 "decode-only ~377" | Different runs and definitions |

---

**End of part C.** No file other than this one was created or changed.

