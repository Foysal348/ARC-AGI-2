# Part B: facts from "Research - Qwen 27B fastest serving setup" (extracted 2026-10-24)

## Scope

This file only extracts facts. It does not design an experiment. Six files were read in full, every line:

| Short | File (in `Notes/Research - Qwen 27B fastest serving setup/`) | Lines | Language |
|---|---|---|---|
| FR | 00_FINAL_REPORT.md | 519 | Bengali. I translated the quotes. |
| OM | our_measurements.md | 615 | English |
| SD | speculative_decoding.md | 278 | English |
| EC | engines_and_config.md | 405 | English |
| WQ | weight_quantization.md | 493 | English |
| CE | community_evidence.md | 596 | English |

**Citations.** "FR L62-74" means lines 62-74 of FR. Keys in square brackets inside a quoted source, such as `[E113A L45-57]`, are the documents' own keys for local run files. The key tables are at OM L10-72 and FR L481-519.

**Evidence classes in this file:**
- **measured-ours**: a number from our own Kaggle 4xL4 runs. The docs label these [M] (measured) or [D] (derived on 2026-10-23 from raw logs).
- **measured-elsewhere**: a third party measured it.
- **estimate**: arithmetic, a projection, or a roofline. The docs use [E] or [P].
- **claim**: a vendor or author statement without method, or a search snippet. The docs use [C], [U] or [S].

The docs' own labels are kept where they matter. My own arithmetic is always marked "(my arithmetic)".

**Configuration shorthand used below**

- **W1**, the production SGLang configuration:
  - SGLang 0.5.21, TP4, RedHat INT4 W4A16 (Marlin), `--kv-cache-dtype fp8_e4m3`.
  - DFLASH (DFlash2) with `speculative_num_draft_tokens=8`, `speculative_num_steps=1` and `--speculative-draft-model-quantization unquant`.
  - `--mamba-ssm-dtype float16 --mamba-radix-cache-strategy extra_buffer_lazy --max-mamba-cache-size 56`.
  - `--mem-fraction-static 0.83 --disable-prefill-cuda-graph`.
  - `--chunked-prefill-size 4096` (max_prefill_tokens 16384).
  - Context 155,648 in the logs; the README planned 106,496.
  - Attention backend flashinfer; linear-attention kernels triton for decode, prefill and verify.
  - `--enable-metrics --disable-flashinfer-autotune`.
  - `--max-running-requests 12`, which became 13 from rev 3c, with the client at 12.
  - Target-verify and draft-verify CUDA graphs are captured for bs [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]. The prefill graph is disabled.
  - Sources: OM L80, L92, L318-324; FR L3, L19.
- **W2** = W1 with bf16 SSM state. **W3** = W1 with fp32 SSM state.
- **V**, the production vLLM configuration:
  - vLLM 0.19, our FP8 repack, MTP-3 (`speculative_config {method mtp, num_speculative_tokens 3}`), TP4.
  - fp8_e4m3 KV; FLASHINFER attention with `disable_flashinfer_prefill=True`.
  - `max_num_batched_tokens` 4096, `max_num_seqs` 16, `disable_custom_all_reduce True`.
  - `enable_prefix_caching True` (mamba "align" mode), `language_model_only True`.
  - `max_model_len` 106,496 in R5 (run 7 used 81,920 at gpu_mem 0.9; L66 used 155,648); gpu-memory-utilization 0.90.
  - Sources: OM L154, L365-367.
- **E11-03 arms** (2026-10-17, account foysal):
  - Common: about 4.3k-token prompts, 4,096 new tokens; SGLang 0.5.21 TP4 INT4; fp8_e4m3 KV; auto mem_fraction 0.6187; `extra_buffer`.
  - **A** = no speculation, fp32 SSM.
  - **B** = + NEXTN (3 steps / topk 1 / 4 draft tokens, unquant), fp32 SSM.
  - **C** = + DFlash2 block 8 unquant, fp32 SSM.
  - **D** = C with fp16 SSM.
  - Sources: OM L135-152, L306-311.

---

## 1. Measured on OUR Kaggle 4xL4 runs

### 1.1 SGLang W1: production and closed-loop aggregate throughput

| Run (date, account) | Workload | Real running | Number(s) | Source |
|---|---|---|---|---|
| E11-04 PROD closed loop (2026-10-18, esfersami50) | ARC prompts, T 1.0, up to 60k thinking, 600 s window; contexts only reached 30,177 | 12 really in flight | W1 (fp16 SSM) **416.5 tok/s**; W2 (bf16 SSM) **412.1 tok/s** | OM L94-99; FR L35; EC L108-111 |
| same | decode rate at 12, by mean context | 12 | W1: <10k 458.3, 10-20k 413.3, 20-40k 388.4, ≥40k not reached. W2: 465.5 / 401.9 / 374.9 | OM L96-97; FR L37-42 |
| same | E114A L21 "about 350-370 at 40-57k" | — | the docs call this a hypothesis [P]/[E], not a measurement | OM L99; FR L45 |
| E11-05 S49 Poetiq EVAL30 (2026-10-18, esfersami50) | 49k thinking, T 1.0 Poetiq | 12 set, **11 really running**: 2,358 decode lines show 11 running + 1 queued, only 114 show 12 | **328.9 tok/s** = 4,253,591 tokens / 12,932.4 s solver wall | OM L105-109; FR L35 |
| same | decode at 11 running by context | 11 | <10k 429.2, 10-20k 371.8, 20-40k 340.0, 40-60k 323.4. Median at 12 = 360.8 | OM L108; FR L37-43 |
| E11-06 P1-60K EVAL14 (2026-10-19, esfersami50) | 60k thinking; server `--max-running-requests 13`, client 12 (11-of-12 workaround) | 12 | **379.9 tok/s** = 2,262,599 tokens / 5,956.2 s; 87.2 % cached prompt tokens; 0 errors | OM L110-111; FR L35 |
| same | decode at 12 (278 lines) | 12 | median 416.6. By context: <10k 500.1, 10-20k 460.3, 20-40k 409.0, 40-60k 389.4 | OM L112-113 |
| same | decode at 8 / 9 running (408 / 400 lines) | 8 / 9 | 368.5 / 388.6 | OM L114; FR L45 |

Context decay at 11-12 running, as the EC doc states it: from 429-458 tok/s (<10k) to 323 tok/s (40-60k), "about −25-30 %" (EC L391).

### 1.2 Per-stream rates under production load (derived, gen tok/s divided by running)

| Run | Per-stream rate | Source |
|---|---|---|
| E11-05 | 30.7 tok/s at 11 running | OM L116; FR L45 |
| E11-06 | 34.7 at 12; 43.2 at 9; 46.1 at 8 | OM L117 |
| E11-05, one phase-1 request (rid 6ffbe589~e0~i0\|1) | 49,152 tokens in 1,418.5 s = 34.7 tok/s | OM L118; FR L45 |
| vLLM V at 16 running | about 22 tok/s/stream | FR L45, L108; OM L199 |

FR's estimate: a 60k-thinking request takes about 29 minutes on W1 and about 45 minutes on vLLM (FR L45, [E]).

### 1.3 The same W1 engine inside the Duck REPL agent (E12; server max-running 13, KV 866,744)

| Run | Running | Decode median tok/s [D] | Per stream [D] | Effective / whole-run rate | Source |
|---|---|---|---|---|---|
| E12 EVAL30 (hosen42, 2026-10-19) | 11 (728 lines) / 12 (280 lines, p25 149.7, bimodal) | 412.5 / 431.3 | 37.5 / 35.9 | Effective only about **187 tok/s** over the solve window (2.36 M tokens / 12.6 ks) because of KV thrash. The doc gives decode as about 420 at 11-12 "even at ≥60k context" | OM L124; FR L53 |
| E12 EVAL6 (2026-10-20) | 6 (796 lines) | 416.3 | 69.4 | — | OM L125 |
| EVAL6R5 xhigh | 6 | 322.2 | — | 244 overall | OM L126 |
| EVAL12R5_LOW | 6 | 351.9 | — | 265 overall | OM L127 |
| G4B rev 6d (2026-10-21) | 11 (137 lines) / 6 (151 lines) | 383.3 / 321.4 | — | 277 tok/s over the window | OM L128 |
| E12 SMOKE1 | 2 | 204.0 | — | — | OM L129 |

- **G4B doc values:** 370-406 at 11 running (20-40k). In the same context bin, the 12 → 6 step-down took decode from 370.9 to 321.4, i.e. −13 % (OM L132; FR L60).
- **Rev 4:** "6 → 12 running: 368 → 431 tok/s" (OM L133).
- **CE quoting E12 docs:** "decode at 431 tok/s", 368 at 6 running; "371 tok/s at 12 running (321 at 6)"; "G4B end-to-end was 277" (CE L46, L73-74).
- **EVAL30 KV thrash:**
  - 30 time-sliced branches × 50-100k contexts ran against about 870k KV.
  - 144 of 423 non-first requests lost their prefix. They account for 8.39 M of the 9.60 M uncached prefill tokens.
  - One retraction was logged: "KV cache pool is full. Retract requests. #retracted_reqs: 1".
  - Sources: OM L337-340; FR L60.
- **FR lesson (L60):** the engine's decode rate (about 410-430) and the agent's effective rate (187-277) are different things. KV thrash, prefix eviction and admission errors keep the first from reaching the second.

### 1.4 E11-03 short sweep: spec methods side by side (total tok/s by REAL running, after the cap correction)

| Real running | A INT4 | B INT4 + NEXTN MTP | C DFlash2 fp32 SSM | D DFlash2 fp16 SSM | vLLM FP8 + MTP-3 (earlier runs) |
|---|---|---|---|---|---|
| 1 | 45.6 | 55.1 | 93.7 | 107.5 | — |
| 4 | 145 | 205 | 250 | 278 | — |
| 8 | 229 | 297 | (capped at 6) | 356 | — |
| 12 | 289 | (capped at 10) | (6) | **404** | ~307 |
| 16 | 347 | (10) | (6) | (capped at 13) 381 | ~373 |

Source: OM L135-143 [E113A L45-57]; FR L64-72; EC L98-106; WQ L151-159.

- **Per request, D:** 107.5 / 72.1 / 48.6 / 34.8 tok/s at 1 / 4 / 8 / 12 requests (OM L145).
- **Long runs, 12 × 24,576 (capped):** A 282.4, B 262.3, C 258.3 (OM L146). B was capped at 10 running and C at 6.
- **Server decode-line medians by real running [D]** (OM L147-151; FR L74):
  - A: 45.7 (1), 147.6 (4), 248.5 (8), 297.8 (12), 370.1 (16), 440.0 (22).
  - B: 80.0 (1), 340.9 (8), 373.7 (10).
  - C: 85.3 (1), 329.7 (6).
  - D: 108.0 (1), 299.6 (4), 482.9 (8), 477.2 (12), 489.1 (13).
- **Two different metrics.** The client-side total (D at 12: 404) and the server decode-line median (D at 12: 477.2) are defined differently. The server median excludes prefill and tail (FR L74).
- **Long context on C** (4 × about 85.5k context, 1,024 new tokens):
  - "16.3 tok/s" per request, accept 4.58, KV usage max 0.87 (OM L152).
  - Server decode median about 238 tok/s at 4 running (OM L152; WQ L162).
  - FR (L74) notes the two numbers do not match. It guesses the per-request figure includes the 85k prefill [E], which is unverified.
- **Speed ratios as the docs state them** (FR L222; EC L384-387; OM L278-279):
  - DFlash2 roughly doubles single-stream speed: 107.5 vs 45.6 ("2.4x" in EC).
  - At 8 running: MTP 297 vs no-spec 229 (+30 %); DFlash2 fp16 356 (+55 %).
  - At 12: DFlash2 404 vs 289 (+39 %).

### 1.5 E11-04 short sweep (W1/W2/W3; short 4,096-token requests)

| Config | 1 | 4 | 8 | at "12" (real max 11, FAIL) | Source |
|---|---|---|---|---|---|
| W1 fp16 | 110.6 | 257.5 | 363.8 (log gen 105.6 / 281.4 / 442.6) | 325.9 | OM L100-104; FR L47 |
| W2 bf16 | 114.3 | 272.8 | 357.6 | 350.9 | same |
| W3 fp32 | 100.0 | 273.5 | 337.9 | 321.4 | same |

### 1.6 vLLM 0.19 FP8 + MTP-3 (configuration V)

**ORDER2 86k DIRECT EVAL120** (hosen42, 2026-10-14): whole run **333.3 tok/s** = 10,447,055 gen tokens / 31,343.2 s (OM L156; FR L78). Engine 10 s lines by running [D]:

| Running | 1 | 2 | 5 | 7 | 10 | 11 | 12 | 15 | 16 (2,844 lines) |
|---|---|---|---|---|---|---|---|---|---|
| Aggregate tok/s | 44.3 | 81.3 | 163.1 | 212.2 | 268.5 | 298.0 | 293.4 | 323.3 | **352.0** |
| Per stream | 44.3 | 40.6 | 32.6 | 30.3 | 26.9 | 27.1 | 24.4 | 21.6 | 22.0 |

Source: OM L158-161; FR L80-83.

- **Journal (DIRECT aggregate, per-sequence rate in brackets) [JLB L172-176]:**
  - 3 active: 107 (36); 6: 203 (34); 12: 302-325 (25-27); 16: **351** (22).
  - "We have never run above 16."
  - Context cost: "about −5 % per +10k tokens… 418 tok/s at 0-10k, 330 at 50-60k" [JLB L178].
  - Sources: OM L163-166; FR L85.
- **Poetiq runs at T 1.0** (OM L167-172; FR L86-91):
  - **R5 EVAL120** (hosen42, 2026-10-15): 11,633,447 tokens / 41,332 s = **281.5** [D]. The docs give 281, and E11-05's doc gives 280.9.
    - Median at 16 running: 295.1 per metric interval and 300.5 per 10 s line, i.e. 18.4-18.8 per stream.
    - Heavy-prefill windows: 138 tok/s.
  - **D66** (77.8k, 16 slots): 256.0 [D]; the doc says 255.8. Median at 16: 286.5.
  - **L66** (122.9k, 12 slots): 254.3 [D]; the doc says 254.1. Median at 12: 253.2 (21.1 per stream).
  - **R4 EVAL12** (≤12 of 16 slots): 225.2 [D]; the doc says 224. Medians: 185.1 at 7, 215.2 at 8, 290.8 at 12.
  - **W3H low effort** (esfersami50, 2026-10-17): 404.0 [D]; the doc says 400. Median at 16: 430.0 per 10 s line.
- **Direct v1.x runs (EVAL12)** (OM L173-180; FR L93-101):
  - run 3 v1.3.5: 307 at 12 active, 247.8 overall.
  - run 4 v1.4: **373 tok/s with 16 running**, 248.9 overall. The GPT audit found 215 ten-second windows at 16 active, mean 356.8, median 369.9.
  - run 6 v1.4.1: 285.
  - run 7 v1.4.4: 274.6 = 1,059,550 tokens / 3,859.1 s, max_active 16.
  - E9 86k: 306.6 overall.
  - E6 dual v2: 288.5 at 16 running.
  - E7: 249-285 at 16 running.
- **E8** (from CE L70): "353.5 tok/s, 16 active ... MTP 0.663". The CE summary range for our vLLM runs is "249-353.5 tok/s" (CE L46).
- **Agent / OpenAI-server mode** (Exp 4 run 4, v1.8.3): 256.5 tok/s. Median 289.5 at 12 running, 24.1 per stream [D] (OM L181; FR L102).
- **E12-V vLLM Duck SMOKE1 run 2** (esfersami50, 2026-10-19): 100.2 tok/s at 2 running and 55.2 at 1 [D]. The memory note says "decode at 2 running 91 tok/s vs SGLang 204" (OM L182; FR L103).
- **Early vLLM** (OM L183-185; FR L104):
  - 2026-09-13 Q1 (`enforce_eager=True`, custom all-reduce off, max_num_seqs 16): "≈ 6 tok/s per sequence, ≈ 51 tok/s aggregate".
  - Archive journal: "C0 long@8 ≈ 48.30 tok/s; C1 graph/noAR ≈ 129.17 tok/s; ≈2.67× speedup".

**FR and OM inferences about these numbers** (FR L106-110; OM L193-203):
- At equal real concurrency (11-12) and context, W1 decodes about 15-30 % faster than V: W1 gives 340-420 at 20-60k; V gives 293-352 at 12-16 running.
- vLLM reaches about 350 with 16 streams; W1 reaches 400+ with 11-12, because each stream is faster (about 35 vs 22 tok/s).
- vLLM throughput depends strongly on temperature and effort. At 16 running: DIRECT T 0.6 gave 352, Poetiq T 1.0 gave 295-300, low effort gave 430. The tasks differed, so this is confounded.
- Whole-run tok/s is always 10-40 % below steady decode because of tail slots, prefill and eviction.
- The best-supported whole-run vs whole-run gap is **+17-30 %**. No paired run exists.

### 1.7 Speculative acceptance (ours)

**vLLM MTP-3** (OM L228-249; FR L197-205)
- **Acceptance rate (accepted ÷ draft tokens):**
  - run 7: 0.661 (704,499 of 1,065,303);
  - ORDER2: 0.669; E9: 0.655; E6: 0.664;
  - run 3: 58.8 %; run 4: 65.7 %;
  - Exp 4 agent run 4: 0.736.
- **Poetiq T 1.0, whole-run mean acceptance length (1 + accepted/drafts) [D]:**
  - R5: 2.671, rate 0.557, per position [0.734, 0.537, 0.400];
  - D66: 2.673, per position [0.735, 0.538, 0.401];
  - L66: 2.670; R4: 2.677.
  - Journal: "T = 1.0 lowers MTP acceptance to 0.557 (2.67 tokens per step instead of 3.0)".
- **Low effort (W3H):** 3.446, rate 0.815, per position [0.91, 0.813, 0.723].
- **DIRECT per position:** 0.82 / 0.66 / 0.53, about 3.0 tokens/step.
- **Link to speed:** "3.01 vs 2.67 tokens/step corresponds to 349 vs 298 tok/s".
- **R5 T 1.0 vs ORDER2 T 0.6:** acceptance 0.549 vs 0.664, tokens/step 2.67 vs 3.01, decode 301 vs 351 tok/s, "confounded by task type".
- **Against concurrency (ORDER2 10 s-line medians):** 2.75 at 1, 2.81 at 2, 3.11 at 3, 3.05 at 5, 3.01 at 6, 3.07 at 7, 2.94 at 10, 3.16 at 11, 3.04 at 12, 2.98 at 15, 3.01 at 16. R5: 2.66 at both 15 and 16. Essentially flat.
- **Duck at T 0.7 (E12-V):** 3.39 at 1 and 3.45 at 2 running. Single 10 s windows reach per-position 0.948 / 0.858 / 0.761. The memory note says "MTP accept 3.5 vs DFlash 5.2".

**SGLang NEXTN (B: INT4 target, 3 steps / topk 1 / 4 draft tokens, unquant, fp32 SSM, extra_buffer)** (OM L251-253; FR L207)
- Accept length: 3.17 (1), 3.10 (4), 3.09 (8), 3.11 ("12", really capped at 10), 3.17 (16), 3.12 (24), 3.10 (32).
- Long run: 3.02.

**SGLang DFlash2 (block 8, unquant)** (OM L255-274; FR L209-220)
- **E11-03 C (fp32):** 3.89 (1), 4.37 (4), 4.74 (8), 4.70 (12), 4.95 (16), 4.75 (24), 4.76 (32). Above 6 requests these were really ≤6 running. Long run 4.20.
- **E11-03 D (fp16):** 4.31 (1), 4.51 (4), 4.85 (8), 4.62 (12), 4.78 (16).
- **E11-04 sweep at 1 / 4 / 8 / 12:**
  - W1 4.50 / 4.49 / 4.67 / 4.51;
  - W2 4.61 / 4.74 / 4.55 / 4.79;
  - W3 4.15 / 4.56 / 4.79 / 4.67.
- **Per sanity puzzle:** C 3.77-5.50; W1 3.78-5.13.
- **Temperature (C, capped at 6):** T 1.0 gave 285.8 tok/s with accept 4.34; T 0.6 gave 241.7 with accept 4.60. "T0.6/T1.0 0.846"; [E113A L72] calls this inconclusive.
- **Long context (4 × 85.5k):** accept 4.58.
- **E11-05 production Poetiq: two definitions give two answers.**
  - Server decode-line median **3.8** (3.82 at 11 running) [D].
  - Per-request `spec_accept_length` median **4.554** over 174 requests [D]. The doc's "4.55" is this figure, and it includes short phase-2 answers that accept 5.6+.
- **E11-06:** median 4.5; 4.34 at 12 running, 4.82 at 9 [D].
- **Over context:** "DFlash acceptance only falls 4.43 → 3.77 over context, so no collapse" [JWK L46].
- **Duck runs [D]:** EVAL30 5.05-5.11 at 10-12 running; EVAL6 5.5 at 6; EVAL6R5 4.33 at 6; LOW 4.75 at 6; G4B 4.34 at 11; SMOKE1 5.1 at 2. The docs write "5.2" (SMOKE1, [E12SMA L29]) and "4.2" (G4B, [E12G4BA L101]).
- **Mandatory flag:** `--speculative-draft-model-quantization unquant`. Without it the drafter inherits INT4 and acceptance drops to about 1.0 (issue #39087). Every DFlash2 launch of ours used it (OM L274; FR L267).

**OM inferences** (OM L276-282):
- The production accept length (about 3.8-4.5) is lower than in the sweeps (about 4.5-4.9). That is consistent with T 1.0 Poetiq sampling and long contexts, but the two causes were not separated.
- vLLM MTP acceptance looks independent of concurrency and dependent on sampling.

### 1.8 Memory, KV capacity, running caps (SGLang, from server logs)

| Config | mem_fraction | SSM dtype / strategy | mamba cache (logged) | Running cap | KV tokens | Free after graphs | Source |
|---|---|---|---|---|---|---|---|
| E11-03 A INT4 | 0.6187 (auto) | fp32 / extra_buffer | 113, ssm 4.01 GB | **22** ("capped to 22 … 5 state slots per request") | 604,274 | 5.55 GB | OM L308 |
| E11-03 B + NEXTN | 0.6187 | fp32 / extra_buffer | 50, ssm 1.79 + intermediate 1.55 GB | **10** | 474,203 | 6.14 GB | OM L309 |
| E11-03 C + DFlash2 | 0.6187 | fp32 / extra_buffer | 33, ssm 1.20 + intermediate 1.97 GB | **6** | 394,559 | 4.65 GB | OM L310 |
| E11-03 D + DFlash2 | 0.6187 | **fp16** / extra_buffer | 68, ssm 1.21 + intermediate 1.97 GB | **13** | 386,563 | 3.98 GB (4.39 after draft graph) | OM L311 |
| E11-04 W1 | 0.83 | fp16 / extra_buffer_lazy | 56, ssm 1.00 + intermediate 1.83 GB | 12 (flag) | **881,336** (target K+V 6.72 GB + draft 2.10 GB) | 1.84 GB (2.22 after draft graph) | OM L312 |
| E11-04 W2 | 0.83 | bf16 / lazy | 56 | 12 | 881,336 | 1.84 GB | OM L313 |
| E11-04 W3 | 0.83 | fp32 / lazy | 56, ssm 2.00 + intermediate 3.66 GB | 12 | 598,712 | 1.82 GB | OM L314 |
| E11-05 production | 0.83 | fp16 / lazy | 56 | 12 | 881,336 | 1.84 GB | OM L315 |
| E11-06 and all E12 Duck runs | 0.83 | fp16 / lazy | 56 (intermediate 1.97 GB) | **13** (server) / client 12 | **866,744** | 1.88 GB | OM L316 |

**Other memory facts**
- **D configuration, per GPU** (OM L328; FR L339; EC L188-199):
  - target weights 4.62 GB; drafter 1.49; mamba 3.26; target KV 2.94; draft KV 0.92;
  - prefill graph 2.60; target-verify graph 0.50; draft-verify graph 0.38 (EC: verify graphs 0.88 together);
  - **free 4.39**.
  - WQ L331 splits the 3.26 GB mamba cache as ssm 1.21 + intermediate ssm 1.97 + conv 0.08.
  - Target + draft KV is about 9.99 KB per token per GPU.
  - Auto `mem_fraction_static` 0.62 left about 19 % of memory unused (4.39 GB per GPU).
- **`/get_server_info` is misleading.** It still reports max_running_requests 32 when the real cap is lower. The real cap only appears in a log warning (OM L329).
- **State slots per request, verified in the 0.5.21 source** (`kv_cache_configurator.py:2232-2261`): radix off 1, no_buffer 3, extra_buffer + overlap 5, extra_buffer_lazy 4 (OM L327; FR L323).
- **11-of-12 admission bug (0.5.21)** (OM L330-333; FR L270):
  - Symptom: a chunked-prefill request is counted twice in admission (`scheduler.py:3971-3976`), so `batch_is_full` latches.
  - Upstream fix PR #44401 landed after 0.5.21 and is untested by us.
  - Workaround: server 13 with client 12.
  - Cost: "about 8 %".
- **Queue = 1 is SGLang's normal steady state at 12 in flight.**
  - E12 EVAL30: 3,127 of 4,153 lines showed queue 1.
  - G4B: 188 lines with queue 1, 101 of them at 11 running.
  - The G4B rev 6d probe read queue = 1 as pressure and stepped 12 → 6, costing −13 %.
  - Sources: OM L334, L474; FR L339.
- **Production KV need:** "about 920k KV tokens: S1-60K peaked at 68 % of 1.35M with 12 x 60k" (OM L335).
- **S49 fit:** W1's 881k holds 12 × (≤10k + 49k + 8k) ≈ 800k (OM L336).
- **Capacity gate:** E11-04 missed its 900k target by 2 % (881,336), so PROD was cut to 10 minutes and long context was marked "NOT PROVEN" (OM L471; FR L339).
- **Rev 5 sizing:** 12 streams × 58k = 697k fits 866,744 at the median; at p90 it would be 890k (OM L341).
- **vLLM V memory** (OM L343-367; FR L357; WQ L321-328):
  - "Model loading took 7.15 GiB memory" per GPU.
  - "Available KV cache memory: 10.98 GiB" at gpu-memory-utilization 0.90, equivalent to 0.8693 without CUDA-graph profiling (`VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`).
  - "GPU KV cache size: 337,600 tokens". The real pool is 846 blocks × 1,600 = **1,353,600** tokens, so the log line is 1/4 of the pool (4 KV groups).
  - "Maximum concurrency": 10.32× at 106,496 tokens; 12.63× at 81,920; 8.46× at 135,168.
  - The attention block is 1,600 tokens, so attention pages are at least as large as mamba pages; Mamba cache mode is 'align' under prefix caching.
  - Fixed cost per running request at MTP k=3: 19.2k token-equivalents from the source, 21.2k from the measured fit [E10C L12-13].
- **vLLM KV peak and preemptions** (OM L352-362; FR L357):
  - run 7: 78.7 %, 0;
  - ORDER2: 87.3 %, 0 (p95 79.5 %, mean 68.6 %);
  - E9: 91.4 %, 0;
  - R5: 0.9953, 9 preemptions (prefix cache evicted; prefill 4.18 M vs 1.14 M for ORDER2);
  - **D66: 0.9988, 66**;
  - L66: 0.9008, 0 (12 slots at max_model_len 155,648);
  - Exp 4 agent: 31 %, 0.
- **Projections [P]:** 18 slots ≈ 98 % KV and 20 slots ≈ 109 % (ORDER2); 20 × 57k ≈ 98 %. INT4 on vLLM would free about 2.6-2.8 GB per GPU, about +320k KV [E10W L42] (OM L364).
- **Inferences** (OM L369-373; FR L173):
  - SGLang's KV advantage comes from INT4 weights (4.62 GB vs 7.15 GiB per GPU) plus explicit mamba sizing.
  - vLLM's 1.35M pool is larger than W1's 881k. SGLang reserves memory for DFlash2 draft KV (2.10 GB), intermediate SSM state for speculative verify (1.83-1.97 GB) and drafter weights (1.49 GB). "The drafter eats most of INT4's memory advantage."
  - At 60k per request, both engines are KV-limited to roughly 12-16 streams.

### 1.9 Communication and topology (ours, before 2026-10-23)

- **Custom all-reduce is disabled at TP4 on both engines.**
  - vLLM log: "Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs" [E4LOG L55-58].
  - SGLang log: "CustomAllReduceV2 is disabled because it's not supported on more than two PCIe-only GPUs" [E113LOG-A L34-41].
  - Sources: OM L460, L468; FR L311.
- **Preflight NCCL all-reduce:** 64 MB in **20.9 ms** (OM L468; EC L173; FR L311).
- **Docs' inference (EC L204-206; FR L311) [E]:**
  - The preflight implies about 3.1 GB/s algorithm bandwidth, or about 4.6 GB/s bus bandwidth at n=4, far below PCIe Gen4 x16 (64 GB/s). This "suggests the Kaggle L4s are not using direct P2P (host-staged/SHM path)". It was not verified with `nvidia-smi topo -m` or `NCCL_DEBUG=INFO`.
  - "Decode all-reduces are small and latency-bound: about 2 per layer x 64 layers per step", i.e. about 128 NCCL latencies per step.
- **Our vLLM notebooks** ran with `disable_custom_all_reduce=True` (WQ L76).

### 1.10 Startup and load times (ours)

- **SGLang** (OM L395-408; FR L378):
  - **E11-03 A, first launch:** ready in 601 s; prefill CUDA graph 447.08 s; load_weight 74.48 s; scheduler_e2e 553.29 s.
  - **E11-03 later launches, same session:** B 166 s (load 7.12 s, prefill graph 68.28 s), C 161 s, D 146 s, X/S 136 s.
  - **E11-04:** W1 ready in 567 s (load_weight 87.36 s, target_verify graph 373.80 s, scheduler_e2e 514.11 s). W2 71 s (target_verify 11.94 s) and W3 76 s, both cached. The doc: "first launch took 567 s (JIT); later ones about 75 s".
  - **E11-05:** EVAL30 ready in 532 s, S1COMMIT4 in 522 s; load_weight 13.41 s, target_verify 373.40 s, tokenizer_e2e 458.86 s.
  - **E11-06:** load 13.48 s, target_verify 372.55 s, tokenizer_e2e 450.23 s.
  - **E12:**
    - SMOKE1: "W1 up in 677 s" (load 13.84 s, verify 371.27 s).
    - EVAL30: verify 384.53 s. EVAL6: load 38.17 s. LOW: load 59.38 s.
    - **G4B: load_weight 669.96 s, scheduler_e2e 1,115.52 s, "ready in 1197s"**; the analysis at the time said "a slow cold start on Kaggle, not our code".
    - Rev 5 runs: about 700 s.
  - **Ready times on 10-19/20:** 507, 522, 532, 552, 562 s (engine total `load_sec` 660-758 s) (FR L365; CE L566).
  - **Wheelhouse venv install:** 157 s (E11-02), 150 s (v2), 181 s.
  - **OM inference:** about 70 % of a cold W1 startup is the 8-token target-verify CUDA graph capture/JIT (about 372-385 s). A warm kernel cache (same session) brings it to about 12 s (OM L425).
- **vLLM** (OM L410-421; FR L378):
  - R5: "Loading weights took 62.59 seconds"; "Model loading took 7.15 GiB memory and 64.103997 seconds"; "init engine (profile, create kv cache, warmup model) took 295.10 seconds".
  - E12-V with prefetch: weights 3.11 s, model loading 4.75 s, init engine 291.59 s.
  - Exp 4 run 4: weights 40.00 s, init engine 290.98 s, server 510 s.
  - run 7: engine load_sec 610.8 s, setup_to_scheduler 789.0 s.
  - ORDER2: load 567.1 s, setup 700.0 s. E6: setup 705 s. E9: setup 701 s. Run 1: setup to scheduler 935 s.
  - Weight prefetch: "weight load fell from 64 s to 4.8 s, and engine start from 600 s to 532 s" [JDL L136].
  - Journal: "About 12-13 min of setup".
- **Overall:** about 9-13 minutes of a 12 h run on either engine, a ~1.5 % effect. Disk-speed variance is the bigger risk: G4B 670 s load; E3 0.29 vs 0.65 GB/s (OM L427; FR L378).
- **Launch control in our notebook** (FR L365):
  - First-launch timeout `SGL_LAUNCH_TIMEOUT_SEC` 30 minutes; relaunch 20 minutes. No launch after `SGL_GIVE_UP_AT = min(T0 + 75 min, admit deadline − 5 min)`.
  - On timeout, `e11_next()` finds no error line in the still-loading log and returns None. That leads to `E11Fail`, `ENGINE_READY = False`, CELL 6 does not solve, and the fallback submission remains, i.e. **about 0 on the LB**.
  - The prefetch is a single-threaded background `cat file1 file2 file3 > /dev/null`. At about 10.5 MB/s, 23.3 GB takes about 37 minutes, longer than the 30-minute timeout.
  - On 10-21 the load took 20 of the 30 minutes.

### 1.11 Accuracy and LB (ours)

- **Sanity set** (8 easy puzzles, E10 DIRECT at T 0.6; FP8 expectation 6.74) (OM L506-510; FR L163):
  - C 7/8 (capped at 6 running); W1 **8/8**; W2 7/8. All closed naturally.
  - Task 136b0064 thought 44.7k on INT4 C vs 23.2k on FP8.
  - The doc: "These are 8 easy tier-A puzzles, so this is no proof of eval parity".
- **Greedy drift** (common prefix out of 512 tokens): B vs A 152; C vs A 155; D (fp16) vs C (fp32) 166; W2 vs W1 41.5; W3 vs W1 44.0 (OM L512-514; FR L265).
- **Contamination tests:** common prefix 12-22 and 0/4 identical, read as batch non-determinism rather than cross-request leakage. The W1 marker test passed, with every request returning its own 8/8 codes (OM L516-518).
- **Eval on the same tasks** (OM L520-524; FR L280):
  - E11-05 SGLang INT4 at 49,152 thinking: **11/48, task-norm 23.3 %**.
  - R5 vLLM FP8 at 77,824 on the same 30 tasks: **16/48, 33.3 %**.
  - In E11-05, 83 of 88 iterations were forced closes, and 7 of 15 train-verified programs were wrong on test (R5: 3 of 13).
  - The doc: "Not separable here: the 49k-vs-77.8k cap effect from any INT4-vs-FP8 effect. No vLLM run at 49k exists on eval."
  - E11-06 P1-60K on SGLang: 7/22 queries, 5/14 tasks (35.7 %). S49 (49k, old prompt, SGLang) on the same 14 tasks: 4/14 (28.6 %). "+1 task… sd ≈ 1.7 tasks… not significant."
- **LB by engine** (OM L526-539):

  | Submission | Engine | Thinking cap | LB |
  |---|---|---|---|
  | S1-60K | vLLM FP8 + MTP-3 | 60k | **34.58** |
  | S49 | vLLM | 49k | 32.08 |
  | S1-55K | vLLM | 55k | 31.81 |
  | E11-06 P1-60K (ref 57372299) | SGLang INT4 + DFlash2 W1, new P1 prompts | 60k | **29.17** |
  | v1.4.4 | vLLM | — | 26.94 |
  | E7 | vLLM | — | 27.92 |
  | E9 | vLLM | — | 22.92 |
  | R5 | vLLM | 77.8k | 21.39 |

  - Notes on the table:
    - The memory note says "SGLang engine + P1 prompts did not transfer to LB (cause not separated)".
    - LB noise: "identical notebooks submitted repeatedly scored an sd of 1.0-1.65 LB points".
    - The −5.4 gap is "3-5×" the noise sd (OM L545; FR L3). OM's takeaway (L501) calls it "within about 3-4 sd".
    - S1-60K has no engine metrics because its commit was cancelled; its engine is R5's engine plus weight prefetch (OM L211).
- **Duck engine swap (SMOKE1):** both engines solved 2ba387bc on both branches. vLLM was slower: 301/455 s and 13.5k/22k gen tokens, vs SGLang 60-73 s and 5-6k tokens (OM L541; FR L280).

### 1.12 Failures and bugs seen in our own runs

- **vLLM** (OM L448-462):
  - **run 1 v1.3.2:** "EngineDeadError" after 792 s. The real cause was a FlashInfer JIT link error, `ld: cannot find -lcuda`; it was fixed with a libcuda shim.
  - **run 2 v1.3.4:** the MTP drafter ran out of memory at `max_num_batched_tokens` 32768. Fixed with 4096 + `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`.
  - **Thinking budget ignored:** vLLM 0.19 with MTP ignores `thinking_token_budget` (#39573); 10 of 12 requests thought until max_tokens 57,344. Workaround `VLLM_USE_V2_MODEL_RUNNER=0`; reportedly fixed in 0.21.
  - **Exp 4 agent:** HTTP 400 "Unexpected reasoning effort high"; HTTP 500 `assert content is not None` on forced tool_choice; a readiness bug.
  - **Preemptions:** R5 9; D66 66.
  - **E8:** CUDA OOM at minute 124 lost 15 units; a `prompt_logprobs` OOM.
  - **R5:** 28 prefix-cache recompute stalls cost about 38 minutes.
  - **Kaggle:** "system error" on v1.4.2/3.
  - **Graph mode:** FlashInfer + spec decode forces cudagraph_mode=PIECEWISE.
  - **E12-V run 1:** every request got 404 "model `pending`".
- **SGLang** (OM L464-480):
  - **Mamba-cache caps:** A 22, B 10, C 6, D 13 flattened the E11-03 sweep curve, because the harness did not assert that real running equalled the requested level.
  - **11-of-12 admission bug.**
  - **Benign log noise:** `ModuleNotFoundError: No module named 'wrapt'`.
  - **FP8 KV scale warning:** "Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0".
  - **E11-04 long-prefix contcache:** 0 cached on aborted samples (a test artifact). E11-05 phase 2 hits the cache 52-99.7 % when the shim waits for the end of the stream.
  - **E12 EVAL30 KV thrash** (section 1.3). **G4B probe** stepped 12 → 6 (−13 %).
  - **Toolchain:**
    - `cuda-tile==1.6.0rc5` is sdist-only;
    - `flashinfer-cubin 0.6.18` does not exist;
    - Kaggle drops '+' from wheel file names;
    - the pip CUDA runtime lacks `libcudart.so` / `lib64`, so a symlink shim is needed;
    - nvcc is pinned to 13.0.88;
    - fresh kernel caches are required because of #42917.

### 1.13 Flash-Next and other hardware (ours)

- **FN-PROBE v4.2.1** (Pennyroyal SGLang 0.5.19, Intel W4A16 AutoRound + FP8 PLE overlay, TP4/EP4) (OM L572-581; FR L384-393):
  - load_weight 344.81-344.94 s (19.06 GB per rank; 2.55 GB free after load);
  - KV bf16 "#tokens 118080"; max_running_requests 8; context 16,384;
  - ready at 862.2 s; warm-up about 7.4 minutes; free memory fell to 0.66 GiB;
  - decode at 1 running with `cuda graph: False` was **5.95 tok/s**;
  - the output was 96 "!" characters, with a TileLang "Data race detected: Logits(bx, position)";
  - verdict: "THROUGHPUT: NO VALID COMPLETE MEASUREMENT".
- **Earlier E3 runs:**
  - v3.3: load 257.67-402.57 s, KV 19,968 / 15,424 tokens, FA4 CuTe "MLIRError: unable to compute crd2idx".
  - v3 / v3.2: `--language-model-only` unsupported; MoE runner assert.
  - v4: both launches timed out at 1,080 s still loading; disk 0.29 GB/s vs 0.58-0.65.
- **Experiment 13 estimates [P]:** KV about 170-250k fp8 tokens; target-only 80-170k (1-3 Duck contexts); 700 tok/s "a target, not a gate"; 420 tok/s comparison threshold (OM L592-598).
- **ARC-AGI-3 track** (RTX PRO 6000, vLLM NVFP4, TP1, 32k context, 5 GiB KV, MTP 3; not L4) (OM L600-604):
  - GPU KV 105,202 tokens; "maximum concurrency for 32,768 tokens per request: 3.21x"; ready at 535 s and 455 s.
  - **234.76 and 241.42 tok/s** final.
  - Franzen's TP1 runs: 588.00 / 594.25 tok/s.
  - Our validation of Franzen's notebook: "ready at 562 s; 590-617 tok/s; host RAM 106/177 GB; GPU 91/96 GB" (CE L210).
- **Experiment 12 FP8 kernel tuning:** never run; 0 outputs (OM L588-590).

### 1.14 Sizes and host facts (ours)

- **RedHat INT4:** 19.45 GB = 18.6 GB body + 0.85 GB BF16 MTP head (WQ L319; EC L10; FR L173).
- **Bytes read at launch:** INT4 + DFlash2 = 19.47 + 3.85 = **23.3 GB** (FR L173; CE L346).
- **Our FP8 repack:** 30.89 GB = 16 × 1.524 GB + outside 6.007 GB + mtp 0.477 GB [CLI listing] (CE L121-125; FR L173).
- **Kaggle 4xL4 host RAM:** 202.6 GB (FR L376; CE L587).

### 1.15 Projections written in our docs (not measurements)

- **vLLM step time:** "about 145 ms/step measured vs a ~60 ms bandwidth ceiling" [E10T L67] (OM L188; FR L110). The 145 ms figure is labelled as measured in the source doc, but it is reported second-hand.
- **vLLM at higher concurrency:** 25-30 running projected at about 380-430 tok/s [E10C L27-51].
- **Required throughput:**
  - "about 1,000 tok/s is needed to give 240 tasks the ARC-Prize budget (2 × 85k) in 12 h" [JWK L46].
  - A 240-task first pass needs 16.6 h at 1× [MEM-P].
- **FR coverage arithmetic [E] (FR L27, L477):**
  - 240 × 1 × 60k = 14.4 M tokens, which needs about 333 tok/s in 12 h.
  - 2 attempts = 28.8 M tokens, about 667 tok/s.
  - At 330-380 tok/s, 12 h gives about 14-16 M tokens, about 240-270 requests of 60k. Each +10 % throughput is about +25 requests (EC L400).
- **FR ranking (L13-25):**
  1. W1, conditional on a target-only parity A/B.
  2. SGLang INT4 + NEXTN/MTP (3/1/4), or INT4 without speculation.
  3. vLLM FP8 + MTP-3.
  4. 2×TP2 INT4, "+20-100 % possible" [E].
  - Ruled out: BF16, NVFP4/MXFP4, INT8, TensorRT-LLM, llama.cpp, EAGLE-3, n-gram, and Flash-Next on L4.
- **FR row 1a:** W1 + true 12/12 admission is projected at about 355-410 tok/s on whole runs [E].

---

## 2. Measured elsewhere (number, source, transferability)

The transferability column is my reading. Where a document states its own caveat, it is quoted. "Same model" means Qwen3.8-27B.

### 2.1 Speculative decoding on the same model: acceptance and speed

| Source | Setup | Numbers | Class | Transferability |
|---|---|---|---|---|
| incoai / z-lab Qwen3.8-27B-DFlash2 card (SD L12-14; FR L226; CE L348-360) | Drafter about 2B BF16, block 8 (7 draft tokens per verify), two-tap dynamic convs + path selector, "lossless". 1× H200, SGLang, FA3, T 1.0 / top-p 0.95 / top-k 20, xhigh, **max_new_tokens 4096**. Baselines: built-in seven-token MTP and RadixArk DSpark | AL, MTP / DSpark / DFlash2: GSM8K 5.02 / 4.36 / 5.46; MATH-500 4.72 / 3.92 / 5.28; HumanEval 3.91 / 3.30 / 4.39; MBPP 3.99 / 3.51 / 4.79; MT-Bench 3.74 / 3.01 / 4.10 | vendor (claim) | Same model, but outputs capped at 4k and BF16 target. SD L56: the gap "was measured with outputs capped at 4096 tokens" |
| Inco blog (SD L15-16; FR L226) | Qwen3.8-27B, default sampling, block 8 | Mean AL MTP 4.28, DSpark 3.62, DFlash2 4.80. "Close to 3x" vs AR; 2.7-3.4× at batch 1 in SGLang. Qwen3.5-4B, thinking on, T 1.0: MTP 4.54, DFlash 4.92, DSpark 5.49, DFlash2 5.97 | vendor | Batch 1 only |
| DFlash2 HF discussions /2 and /9 (FR L234; CE L361-364) | — | "works with the FP8 quant target model … acceptance length is very close to the BF16 target"; "Positions 4-7 usually get around 0.1 or less"; "Reducing to 3 tokens makes DFlash perform similarly to MTP" | user report | Relevant to FP8 targets and to block size |
| HackMD thc1006 (SD L17, L166; FR L236) | 1× RTX 3090, llama.cpp, UD-Q4_K_XL, greedy, 400-token window | Baseline 41.55 tok/s. MTP n-max 2: 66.39 (+59.8 %); MTP n-max 3: +52.3 %; DFlash2 n-max 4: +51.9 %; MTP n-max 5: +32.1 %; DFlash2 n-max 7: +22.6 %, negative on 3 of 5 prompt classes. **Best MTP beat best DFlash2.** MTP and DFlash2 outputs diverged from serial greedy in 92-100 % of requests at 1,600 tokens, deterministically; semantic quality not measured | independent | Batch 1, llama.cpp, different engine. The doc says it contradicts the vendor's batch-1 ordering (3.43× DFlash2 vs 2.59× MTP) |
| Jetson AI Lab (SD L18) | AGX Thor, vLLM | MTP depth 3: decode about 13 → about 35 tok/s (about 2.7×) | vendor (NVIDIA) | Batch 1, different hardware |
| kubesimplify (SD L19-28; FR L256) | DGX Spark | c=1 / c=10 aggregate: vLLM NVFP4 11.5 / 84.3; + MTP-3 22.0 / 105.8; SGLang FP8 7.7 / 54.3; + NEXTN 13.4 / 71.0 | independent | Bandwidth-limited single device; c ≤ 10 |
| lcz.me/topic/1712 (FR L236; CE L418-423) | RTX 4090 48 GB, sm89, single card, SGLang 0.5.16 → 0.5.19 | W4A16-AWQ + DFlash2 vs FP8 + MTP: code 245.6 vs 75.7; math 167.4 vs 64.9; Chinese prose 56.3 vs 47.3; 2-way aggregate 349 vs 136. 0.5.19 adds a 1.69 GB `intermediate_ssm_state_cache`; nvcc pinned 13.0.88 | measured (via subagent) | FR: "the only same-architecture (sm89) A/B, but quantization also differs". Concurrency ≤ 2 |
| club-3090 #1746 (FR L236; CE L411-413) | 2× 3090 NVLink, vLLM 0.31.0, TP2, Swift 1.5 | Single-stream DFlash2 136.5 vs MTP k=3 101.9; speculation "stops paying at 4+ clients" | measured | Different model variant (Swift); vLLM |
| hanxiao (FR L236; CE L429) | Single L4, llama.cpp, UD-Q4_K_XL | Plain 14.98; MTP depth 2 + kernel patch 28.05; **DFlash2 (7 drafts) 37.4**; "Effective bandwidth 252.8 of 300 GB/s" | measured | Same GPU, but single GPU, single stream, llama.cpp |
| vramcalculator (FR L236; CE L428) | Single L4, UD-Q4_K_M, no MTP | 13.1 tok/s at 29K context, 6.1 at 236K; prefill 647 → 217 | measured | Single stream |
| Qwen3.8-27B-FP8 discussion #19 (CE L337) | 4090 48 GB, custom NInfer engine | 132.0 tok/s single request with DFlash2 (5 draft tokens, 62.8 % acceptance) | measured | Single stream |
| Qwen3.8-27B-FP8 discussion #13 (FR L256; CE L336) | — | The FP8 MTP head lowers acceptance by about 3-4 % per position (unconfirmed) | user report | Relevant if an FP8 checkpoint's MTP head is used |
| Unsloth MTP guide (SD L236; FR L256) | — | MTP about 1.4-2.2× with no accuracy change; about 160 tok/s for 27B MTP on RTX 6000 | snippet | Weak |
| AzatAI FP8-dynamic card (WQ L139-143; FR L140) | 2× RTX 6000 Ada (sm89), TP2, vLLM 0.19.0, single stream, per-channel FP8 | Decode 38.7 tok/s without MTP, 63.3 with MTP k=3. MTP mean acceptance 2.56 (prose), 3.6 (code). "memory-bandwidth-bound on Ada" at about 18.4 GB of weights per GPU per token | measured | Same architecture family (Ada), same model, single stream |

### 2.2 Thinking mode and long outputs (relevant to 60k thinking)

| Source | Numbers | Class | Transferability |
|---|---|---|---|
| SGLang Flash-Next cookbook (SD L31; FR L240) | MTP, 4 draft tokens: accept 3.5-3.7 of 4 on non-thinking output, "**about 2.5 on thinking output**" (DGX Spark). RTX PRO 6000: about 3.3 (GSM8K/random), 2.9 (ShareGPT) | vendor | Same family, different model. SD L57: plan around 2-2.5 for MTP on thinking output |
| froggeric (SD L32; FR L240) | Qwen3.6-27B MTP, llama.cpp, M2 Max. Thinking on coding: Q8_0 draft acceptance 87 % → 73 %, speedup stays +94 %. By task: code 79-89 %, factual 62-70 %, analysis 48-56 %, creative 39-48 % | independent | Same architecture, batch 1 |
| DFlash paper (SD L33; FR L240) | Qwen3-4B / 8B, thinking on (speedup / τ). T=0 MATH-500: 4.59×/5.74 and 4.64×/5.82. T=1 MATH-500: 3.93×/4.89 and 4.03×/5.06. AIME25: 3.64×/4.68 and 3.70×/4.69. Short max lengths | academic | Short outputs |
| Test-Time Speculation (arXiv 2605.09329) (SD L36; FR L238) | Qwen3-8B, MATH-500, DFlash / EAGLE-3 / PARD: "average acceptance length is 3.7 for the first 10K output tokens, but sharply decreases to 1.5 for the last 10K". EAGLE-3 "reduces to 1.1 … after generating ~20K tokens". Qwen3.6-35B DFlash AIME-2025: "from 15 to 1.7 … within a span of 25K tokens". MTP not tested. Online distillation: AL up to +72 % (+41 % on average) | academic | Directly relevant to 60k thinking. Our measured DFlash2 4.43 → 3.77 contradicts the collapse (section 4) |
| DFlash paper Table 4 (SD L37; FR L238) | Qwen3.5-27B base drafter, trained on 4K. τ by context: hotpotqa 4.91 (1K), 4.46 (8K), 3.61 (16K); gov_report 4.53 (1K), 3.32 (8K), 2.67 (16K), 2.09 (32K). A long-context fine-tuned drafter (1.6K LongAlign samples): gov_report 3.56 at 32K, hotpotqa 6.05 at 16K | academic | Same-size predecessor model |
| z-lab Qwen3.5-27B-DFlash card (SD L38; FR L238) | Drafter trained at **40k** sequence length, with a sliding-window long-context variant | vendor | Conflicts with the paper's 4K (section 4). The Qwen3.8-27B DFlash2 training context is unpublished |
| Attention drift (arXiv 2605.09992) (SD L39; FR L238) | Drift in both EAGLE3 drafters and **MTP heads**; hidden-state magnitude grows with chain depth. A normalization fix gives 1.18× AL on long context | academic | MTP is not immune |
| vLLM #54691 (SD L94; FR L238) | 4× RTX 4090 TP4, vLLM 0.28.0, Qwen3.5-family hybrid + DFlash2, single stream. At **185k context: no spec about 71 tok/s, DFlash DT4 16.0, DT2 12.1** (4.4× slower). At short context: DT4 about 190, DT8 about 218. Draft-side cost grows with context. Open | user report | Closest hardware analog (Ada, TP4); vLLM, not SGLang |
| vLLM PR #61261 (SD L95) | Proposes an opt-in `draft_context_window` to bound DFlash/DSpark draft attention | user report | vLLM only |
| MiaAI #73 (SD L96, L126; FR L238) | GLM-5.3-Flash DFlash2 k=7, 2× GB10. First report: acceptance 70-73 % short → 16-32 % at about 172k. Follow-up ladder: no decay (0.31-0.36 from 4K to 170K), but a slower step at 170K. Also k=4: 47 % and 28.7 tok/s; k=7: 27 % and 20.6 tok/s | user report | Different model; the two reports conflict |

### 2.3 Speculation vs concurrency

| Source | Numbers | Class | Transferability |
|---|---|---|---|
| incoai DFlash2 card, H200 SGLang (SD L70-78; FR L228-234; CE L351-359) | MATH-500 tok/s, Plain / MTP / DFlash2: c1 69.0 / 172.8 (2.51×) / 230.7 (3.34×); c8 480 / 1,023.5 (2.13×) / 1,368.3 (2.85×); c32 1,505.8 / 1,415.6 (**0.94×**) / 1,951.8 (**1.30×**). GSM8K c32: AR 1,329.8, MTP 1,381.1, DFlash2 1,922.5. MT-Bench c32: MTP 0.77×, DFlash2 1.01×. Speedup ranges across benchmarks: c1 MTP 1.96-2.59×, DFlash2 2.67-3.43×; c8 1.74-2.19× vs 2.27-2.85×; c32 **0.77-1.04× vs 1.01-1.45×** | vendor | Same model, compute-rich H200, 4k outputs. FR/SD [E]: on L4 the point where gains vanish should be higher (MagicDec regime) |
| z-lab Qwen3.5-27B-DFlash v1 (SD L79, L123; FR L242) | B200 TP1: c1 block 8 3.48-4.76×, block 16 3.77-6.20×; c32 block 8 2.14-3.01×. "Beats MTP at every matched setting". Block 8 recommended "for higher-concurrency serving", block 16 for c1. Mean AL at c1, block 4 / 8 / 16: GSM8K 3.53 / 5.64 / 7.51, MT-Bench 3.22 / 4.63 / 5.62 | vendor | Same-size predecessor |
| DFlash paper Table 3 (SD L80; FR L242) | B200 SGLang, c 1/4/8/16/32. Qwen3-8B Math500 5.1× / 4.5× / 4.5× / 3.9× / 2.8×; HumanEval 4.2× / 3.6× / 3.6× / 3.0× / 2.4×. Qwen3-Coder-30B-A3B flat at about 3.1-3.5× | academic | Small models |
| DFlash paper Table 5 (SD L81; FR L242) | LLaMA-3.1-8B, B200, c 1 → 32. EAGLE-3 (10) 1.6× → 1.0×; **EAGLE-3 60-token tree 1.9× → 0.6×**; DFlash (10) 2.4× → 1.6× | academic | Trees go negative at high batch |
| EAGLE-3 paper (SD L82; FR L242) | 1.38× throughput at batch 64 in SGLang (H100) | academic | — |
| SGLang Flash-Next cookbook (SD L83-84; FR L242; CE L247-251) | RTX PRO 6000, 1024 in / 256 out. MTP: 148 at 1, 613 at 16. No MTP: 83 at 1, 524 at 16 (**+17 % at 16**), 861 at 64. ShareGPT: 785 at 16 with MTP vs 1,258 at 64 without. DGX Spark: MTP 71.7 at 8 concurrent vs no-MTP 83 at 24; MTP admitted only 8 requests (40 slots) vs 24 | vendor | Different model. Shows the mamba-slot cap limiting MTP concurrency |
| SGLang Qwen3.5 cookbook (SD L85; EC L122; FR L292) | Qwen3.5-397B, H200 TP8, NEXTN 3/1/4: accept 3.12 at c=1 (222.84 tok/s); 3.31 at c=100 (1,767.45 output tok/s). No no-MTP baseline | vendor | MTP acceptance flat with concurrency, matching ours |
| vLLM Qwen3.5 recipe (SD L86; FR L242) | MTP-1 is for latency. "Under high load, it lowers text throughput because speculative tokens consume KV cache capacity and shrink the effective batch size." Throughput configs do not use MTP | vendor | Generic |
| shreyansh26 RESULTS.md (SD L87; FR L242) | B200, c=8, 8K in / 128 out, Qwen3.6-35B-A3B. vLLM MTP +3.5 % (within launch noise of 1,024-1,140 tok/s). SGLang FP8: MTP +23.1 %, EAGLE3 −3.6 %, DFlash v1 block 4 −53.3 %, v2 block 4 −46.3 % (AL 1.2-1.4) | independent | MoE model, prefill-heavy |
| Dre Dyson guide (SD L88) | Qwen3.6-27B llama.cpp: at concurrency 4 the non-MTP setup pulled ahead | snippet | Weak |
| MagicDec (SD L89; FR L242) | At long sequence lengths the bottleneck shifts to KV loading, so speculation still helps at large batch: up to 2.51× for LLaMA-3.1-8B at batch 32-256 | academic | The docs' main argument that L4 at long context stays in a regime where speculation pays |
| SPEED-Bench (SD L91; FR L242) | "At lower batch sizes … longer drafts are preferred"; at higher batch "DL=1 is preferred". Synthetic inputs overestimate throughput by about 23 % on average | academic | Generic |
| vLLM #55533 (FR L242, L274; CE L395-398) | RTX 4090 D (sm89), GPTQ INT4 Marlin, MTP k=1. No-spec vs MTP: bs1 56.4 vs 72.0; bs8 367.1 vs 197.0; **bs20 649.0 vs 210.9**. The scheduler ran "exactly 3 of the 8 sequences" because MTP reserves extra GDN state blocks. Reproduced on RTX 6000 Ada with RedHatAI INT4; no fix | user report | Our GPU architecture and our checkpoint, but vLLM only |
| club-3090 #1092 (FR L242; CE L406-409) | 2× 3090 PCIe, vLLM v0.27.1, TP2, W4A8 AutoRound. MTP n=4 vs spec off: C1 71 vs 58; C8 201 vs 211; **C32 241 vs 308**. "DFlash2 can't do concurrency on 24 GB" (OOM at C8). `MAX_MAMBA_CACHE_SIZE=20` raised the KV pool from 451k to 548k | measured | 24 GB cards like ours, but vLLM TP2 |
| HyperQwen (CE L431-435; FR L222) | RTX 3090, patched vLLM 0.31.0: 127 tok/s single user, about 1,035 tok/s at 64 concurrent. "Speculation wins below ~8 concurrent users, plain batching above". TP=2 and TP=4 work; TP=3 invalid | claim (via subagent) | FR L222 says this "does not hold for this hybrid model on L4" (section 4) |

### 2.4 Settings sweeps (steps / topk / draft tokens / block)

| Source | Numbers | Class |
|---|---|---|
| SGLang Qwen3.5 / Qwen3-Next cookbook (SD L116; FR L254; EC L292) | `--speculative-algorithm EAGLE` (NEXTN is an alias) `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`; tune with `bench_speculative.py`. MTP with the FlashInfer attention backend needs FlashInfer newer than 0.6.15.post1, otherwise use `--attention-backend triton` | vendor |
| vLLM recipes (SD L117; FR L254) | `{"method":"mtp","num_speculative_tokens":N}`, N=1 (latency), 2 (Qwen3.6), 3 (MI300X), 5 (DGX Spark); tune within 1-5 | vendor |
| RedHat INT4 card (SD L118; CE L342) | MTP `num_speculative_tokens` 3, or DSpark with 7 "for optimal performance" | vendor |
| shreyansh26, B200 c=8 (SD L119; FR L254) | SGLang MTP steps 1 / 2 / 3 / 4 / 5 / 7 → 649.9 / 703.1 / 736.2 / 725.0 / 740.5 / 706.6 tok/s. vLLM MTP tokens 1 / 2 / 3 / 5 / 7 → 937.7 / 880.3 / 1,024.0 / 1,033.5 / 1,032.3. EAGLE3 best at steps1_k1_d2 | independent |
| HackMD (SD L120) | 3090 batch 1: MTP n-max 2 > 3 > 5 (+59.8 % / +52.3 % / +32.1 %) | independent |
| froggeric (SD L121; FR L254) | M2 Max: "N=3 is optimal" for all quants except F16 (N=4 slightly better) | independent |
| Jetson (SD L122; FR L254) | Starting points: MTP 3, DFlash 15 (block-16 checkpoint), DSpark 4. Lower DFlash depth can win when acceptance is low | vendor |
| DFlash2 for Qwen3.8-27B (SD L124; FR L254) | Trained at block 8. SGLang `--speculative-num-draft-tokens 8`, vLLM `num_speculative_tokens: 7`, llama.cpp `--spec-draft-n-max 7`. z-lab recommends `block_size <= 5` for quantized targets (MLX quantized-matmul cost) | vendor |
| DFlash paper Table 8 (SD L125; FR L254) | A drafter trained at b16 and run at b8 loses τ (Math500 6.33 → 5.09). Run the trained block size | academic |
| SGLang #23330 (SD L127; FR L254) | `--speculative-adaptive` crashes on hybrid GDN Qwen3.5 (RTX 4090, sm89) because the MTP draft mutates the shared config. Fix PR #23331; the issue was closed by an inactivity bot, so the fix is unconfirmed | user report |

### 2.5 Drafter memory and mamba slots (elsewhere)

| Source | Numbers | Class |
|---|---|---|
| incoai GGUF (SD L182, L219; FR L256) | DFlash2 BF16 3.8 GB (2B params), Q8_0 2.0 GB, Q4_K_M 1.1 GB (llama.cpp only). With a Q4_K_M target: AL 5.28 (BF16 draft), 5.13 (Q8_0), 5.39 (Q4_K_M), vs 5.46 for a BF16 target in SGLang (tiny sample: first 8 GSM8K, 2048 tokens) | vendor |
| Qwen3-Next cookbook / Jetson (SD L183) | MTP layers ship inside the checkpoint; MTP "can need almost no extra model memory" | vendor |
| originalGeek (SD L184; FR L256) | Qwen3.6-27B: 1 MTP layer (15 tensors) sharing embed/lm_head; estimated about 0.4B params, about 0.8 GB BF16 | snippet (estimate) |
| Flash-Next cookbook (SD L185) | Slots per request with MTP (4 draft tokens): **extra_buffer 5, extra_buffer_lazy 4, radix disabled 1, lazy + `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` 3**. `--mamba-ssm-dtype bfloat16` halves the slot (0.109 → 0.055 GiB per slot for Flash-Next). Set `--max-mamba-cache-size` = requests × slots. On RTX PRO 6000 96 GB the default extra_buffer allows about 3 requests with MTP vs 12 without; tuning gives 16 with MTP vs 64 without; ceilings about 20 vs about 70 | vendor |
| SGLang #36889 (SD L151, L186; FR L269; CE L415-416) | DFLASH needs 1 + speculative_num_draft_tokens slots per request. One report saw concurrency capped to 2; fixing it raised c=12 throughput 48.8 → 85.4 tok/s. bf16 SSM state cost about 6 % single-stream decode through shorter accepts. `--max-mamba-cache-size 40 --mamba-ssm-dtype bfloat16` gave +75 % at c=12. On GLM-5.3-Flash, raising the pool from 10 to 40 slots took c8 from 47.3 to 80.3 tok/s (CE describes "5 state slots per request"). club-3090 #1319: 33 mamba slots with DFlash2 vs 55 with MTP | user report |
| jpezzulli #5 (SD L187; FR L323) | 4 slots held per request but the cap computed for 5, leaving about 20 % of the cache unreachable | snippet |
| Qwen3-Next cookbook (SD L188) | `--mamba-full-memory-ratio` (default 0.9) sets the ratio of mamba memory to KV memory. Raising `--max-mamba-cache-size` reduces KV space. In KV-bound cases, V2 overlap must be weighed against lower max concurrency | vendor |
| LMSYS DFlash v2 blog (SD L190) | DFlash's KV-injection design avoids storing target latents, to save KV space | vendor |
| HackMD (SD L191) | On RTX 3090 24 GB, a vLLM MTP comparison for Qwen3.8-27B could not start because of extra BF16 allocations | independent |
| Flash-Next cookbook (SD L230; FR L256) | NVIDIA NVFP4 Flash-Next export ships an FP8 block-scaled MTP; the RadixArk BF16 draft measured about the same: accept 3.33 vs 3.31, TPOT at 16 requests 18.5 vs 19.1 ms | vendor |
| froggeric (SD L220-229; FR L256) | MTP speedup by target quant (Qwen3.6-27B, M2 Max, N=3, 2.5k tokens): code F16 +171 %, Q8_0 +123 %, Q6_K +50 %, Q4_K_M +31 % (creative −9 %); mainline llama.cpp Q4_K_M code +18 % | independent |
| arXiv 2607.04244 (SD L234; FR L256) | A block-diffusion drafter trained in two stages (high-precision target, then the quantized target), Qwen3.5-4B, A10G: 6.978× on average | academic |
| jksdfosd INT4 variants; Kaitchup; froggeric IQ2_M (SD L232-233) | `-mtpq` ships `model_mtp.safetensors` (precision not stated) and keeps lm_head and embed in bf16; a `-both-mtp` variant exists. Kaitchup leaves the MTP head unquantized. llama.cpp IQ2_M keeps MTP tensors at Q4_K | user report / independent |

### 2.6 EAGLE-3, n-gram, suffix

| Source | Numbers | Class |
|---|---|---|
| Ex0bit (SD L42; FR L252) | Qwen3.6-27B EAGLE-3: accept length 2.2 | snippet |
| AQ-MedAI (SD L43) | Qwen3.5-397B EAGLE-3 below MTP AL on every benchmark (MT-Bench 3.03 vs 3.28; CEval 1.77 vs 2.34), though it wins throughput on some tasks | snippet |
| shreyansh26 (SD L44) | Qwen3.6-35B-A3B SGLang B200: native MTP AL 3.715 vs EAGLE3 1.251 | independent |
| DFlash paper Table 1 (SD L45) | Qwen3-8B, thinking off, T=0: EAGLE-3 (16) 1.76× / τ 2.96 vs DFlash (16) 4.86× / τ 6.49 | academic |
| SPEED-Bench (SD L48; FR L252) | Qwen3-Next MTP, BS=32, DL 3, T=0: mean AL 2.81 (coding 3.34, math 3.13, reasoning 2.89, writing 2.46, roleplay 2.09); mean speedup 1.20× (1.18× at T=1). "pre-trained MTP head maintains higher ALs than … EAGLE3" | academic |
| SPEED-Bench n-gram (SD L51; FR L252) | AL 1.41 (0.88×, a slowdown) on Llama 3.3 70B; 1.31 (**0.29×**) on GPT-OSS 120B | academic |
| Snowflake SuffixDecoding (SD L52, L90; FR L252) | 1.96-3.12× on BlazeEdit; 1.0-1.28× on SpecBench; up to 5.3× agentic (AgenticSQL, SWE-Bench); no reasoning numbers; about 10 % overhead at concurrency 64; beats n-gram at every concurrency | vendor |
| MLSys 2026 slides (SD L53) | Up to 2.13× over vLLM n-gram for Qwen3 1.7B/8B/14B reasoning serving and RL rollouts (H100) | snippet |
| Snowflake Arctic blog (SD L163; FR L252) | `"enable_suffix_decoding": true` in Arctic Inference/vLLM; combinable with a model drafter. LSTM+Suffix 209 tok/s on a mixed workload vs 154-155 for either alone (Llama 3.1 70B). MTP + suffix undocumented; no hybrid data | vendor |

### 2.7 Engine comparisons

| Source | Numbers | Class | Transferability |
|---|---|---|---|
| Winder.AI (EC L72-82; FR L292) | H100 SXM, Qwen3.8-27B FP8, SGLang 0.5.20 vs vLLM 0.30.0, about 900 in / 256 out. c1 81 vs 76; c10 627 vs 589; c50 1,725 vs 1,610. SGLang at 50 needed `--max-mamba-cache-size 256 --mem-fraction-static 0.9`; on defaults it "capped itself at 20 running requests" and gave 982. vLLM refused to start until `--max-num-seqs 256`. TTFT at 50: 1.5 s vs 2.1 s | measured | Same model; compute-rich GPU; short outputs |
| HelixML via Winder (EC L83; FR L292) | 8× RTX PRO 6000, 192 concurrent: SGLang 3,833 → 7,883 tok/s after storing the state cache in bfloat16 | claim (second hand) | — |
| CloudRift (EC L84-97; FR L292) | 1× RTX PRO 6000, Qwen3-Coder-Next-FP8. MCR 8/16/24/32/40/48: SGLang 510 / 733 / 808 / 886 / 886 / 864; vLLM 495 / 779 / 846 / 988 / 1,207 / 1,130. SGLang best 898 at MCR 28; vLLM best 1,207 at MCR 40 (+34 %). SGLang needed `--attention-backend triton --disable-radix-cache --kv-cache-dtype bf16 --fp8-gemm-backend triton`, which the author says reduce performance. MCR 4: SGLang 330.52 vs 276.50, TTFT 1,480 vs 5,647 ms | measured | Different model and GPU |
| ComputingForGeeks / Winder (EC L128; FR L292) | Non-hybrid: Llama 3.1 8B on H100 within 2 %; Qwen2.5-7B on RTX 4090 at c64: SGLang 0.5.15 1,718 vs vLLM 0.25.1 1,740 ("a near tie") | measured | Non-hybrid |
| PyTorch blog "Hybrid Models Meet SGLang" (EC L123-127; FR L292) | H200, SGLang v0.5.5, Qwen3-Next-80B-A3B-FP8 TP2, bs=1 MTP: 2-token window 257.20 tok/s (accept 2.709); 3-token 306.94 (3.413); topk 4 / 8 draft tokens 324.57 (4.231) | measured | Batch 1 |
| LMSYS Qwen3.8 day-0 (EC L388) | MTP on the Qwen3.8 dual-PP6 NVFP4 backbone: +10.0 % throughput, 2.33× per-user speed (GB300) | vendor | Not L4 |
| SGLang cookbook Qwen3.8-27B (EC L388; CE L315-317; FR L343) | RTX 5090: DFlash2 4.92 ms median TPOT at accept 4.29. Single-request EAGLE: NVFP4 152.9 (fp32 state) / 144.5 (bf16); FP8 106.3 / 116.1 tok/s/user | vendor | Single GPU, single request |
| LMSYS DFlash v2 blog (SD L147; FR L288) | Qwen3-8B, B200, c=32: about 11.4k → about 15.3k tok/s (+33 %) with Spec V2 overlap (`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`) | vendor | FlashInfer (the Ada default) adds a plan-stream sync point that limits overlap |
| vLLM 25K TPS blog (EC L33, L382; FR L290) | The Blackwell FlashInfer GDN prefill kernel gives 1.02×-5.78× over FLA/Triton (Blackwell only). `--async-scheduling` named a key driver, no isolated number. `--max-num-batched-tokens 16384` about +8 %. A raised `--max-cudagraph-capture-size` gave no isolated gain. Async scheduling had race conditions: "accuracy was zero without the fixes" | vendor | GB200, not Ada |
| TensorRT-LLM docs (EC L37-38; FR L294) | 1.3.0rc15 lists `Qwen3NextForCausalLM` and `Qwen3_5MoeForCausalLM` (AutoDeploy), not dense `Qwen3_5ForConditionalGeneration`; MTP = No, DFlash = No. Qwen3Next footnote: "relatively low accuracy on the SciCode-AA-v2 benchmark". The support matrix lists Ada SM89 FP8/INT8/INT4 "not implemented for every model" | vendor | Not usable for this model |
| llama.cpp (EC L41-43; FR L294) | Qwen3.8-27B GGUF on 1× 3090 (WSL): about 42.9 tok/s single stream with MTP. Build 221f0f6 gave garbage on CUDA GDN layers; fixed at ece963f (about build 10450). 4×3090 tensor-split "did not scale": Q4_K_M 76.4 (2 GPUs) → 78.7 (4 GPUs). Dual 3090: llama.cpp Q4_K_M + MTP about 51/60 TPS vs vLLM dual 89/127 | measured / snippet | No multi-GPU scaling |
| ExLlamaV3 README (EC L46-47; FR L294) | Lists Qwen3-Next, Qwen3.5, Qwen3.5 MoE, "Qwen 3.8-Flash-Next"; TP/EP, continuous dynamic batching, speculative decoding; v1.0.0 "added a graph path for all attn/GDN modules". No throughput numbers | vendor / snippet | Unproven |
| LMDeploy (EC L50; FR L294) | TurboMind: Qwen3.5 (BF16, KV INT8, W4A16), not KV INT4, no vision. PyTorch engine: BF16 only. Qwen3.6/3.8 not listed; no FP8 | vendor | Unproven |
| club-3090 (EC L28; FR L288) | SGLang "currently blocked on Ampere" for Qwen3.6-27B | snippet | Not Ada |
| agens-sglang fork (SD L149; FR L288) | A community fork runs DFLASH on sm89 through FlashInfer | snippet | — |

### 2.8 Parallelism and communication

| Source | Numbers | Class | Transferability |
|---|---|---|---|
| club-3090 #773 (EC L148-158; FR L300) | 4× RTX 3090 PCIe Gen3, no NVLink, P2P patched, Qwen3.6-27B, vLLM 0.23-0.25.1. Matched recipe (MTP n=3), TP2 → TP4: narrative decode 79.26 → 82.44 (+4.0 %); code 106.50 → 103.73 (−2.6 %); prefill at 10k 1,556 → 2,407 (+54.6 %), at 90k 1,202 → 1,948 (+62.0 %); TTFT at 90k 73.1 → 45.6 s. AutoRound INT4 peak aggregate at c=64: one TP2 instance 473.1 vs one TP4 instance 294.8. With `--max-num-seqs 256 --enable-chunked-prefill`: **INT4 TP4 307 vs 2× TP2 737 (2.40×)**; gemma-4-12B AWQ 1,007 (TP4) vs 3,425 (4× TP1, 3.4×). Forced custom AR at TP4 (8 MiB cap): c=1 80.95 → 92.92 (+14.8 %), c=16 144.06 → 164.66 (+14.3 %), but greedy outputs diverged on 4/5 prompts ("not eval-clean"), and prefill-sized tensors still paid registration overhead (about +9 % TTFT at c16). TP2 + custom AR + verified P2P at 220 W beat a no-P2P reference by about +15.7 %. INT8 KV: +76.5 % short context, −71.9 % at about 58k prompt (crossover about 7-10k) | measured | PCIe consumer box like ours, but 3090 (936 GB/s, Gen3) and P2P patched. EC L209: "indicative only" |
| biMEMO card (EC L159-168; FR L300-309) | RTX 3090, Qwen3.8-27B int4-AutoRound, vLLM 0.27.1, MTP 1 token, fp8 KV, 512-token responses, `--gpu-memory-utilization 0.93`, `--max-num-seqs 32`. Aggregate, 1 GPU / TP2 / TP4: c1 55.6 / 90.1 / 102.4; c2 111.4 / 176.1 / 187.4; c4 223.6 / 323.3 / 322.3; c8 — / 518.6 / 469.4. The author calls 2 GPUs the "sweet spot" and says TP4 "did not scale as well at higher concurrency" | measured | Same model, INT4 |
| NVIDIA forum DGX Spark TP/DP study (FR L309; CE L437) | "two TP=2 replicas are the more effective configuration"; at C8, TP2 + DP2 gives 36.0 tok/s per request vs TP4 27.0 | measured | Different interconnect |
| vLLM custom_all_reduce source (EC L169; FR L311) | Returns early when world_size > 2 and the GPUs are not fully NVLink-connected; also disabled if the P2P test fails | vendor | Applies to us |
| vLLM forum 8× L20 (EC L170; FR L311) | SymmMemCommunicator not supported on capability 8.9 | snippet | Applies (sm89) |
| vLLM forum, multi-NUMA PCIe Blackwell (EC L171; FR L311) | Hung at worker init unless `--disable-custom-all-reduce`; `NCCL_P2P_DISABLE=1` also needed | measured | — |
| vLLM forum dual 5090 (EC L172; FR L311) | NCCL P2P toggle gave only about +2-5 % (sometimes negative); "SHM is not always slower than P2P" | measured | Consistent with our 2026-10-23 finding (section 6a) |
| club-3090 #1288 (CE L400-405; FR L416) | 4× RTX 3090 at 220 W, NVLink pairs, SGLang v0.5.19, TP4, W8A8 INT8, bf16 KV, unquantized DFlash2 with 8 draft tokens, HiCache, `max_running_requests=16`, `NCCL_P2P_LEVEL=SYS`. c1: narrative 146.6, code 265.5. Aggregate narrative / code: c4 487.7 / 893.9, c8 730.6 / 1,418.3, **c16 927.5 / 1,811.0**. Prefill peak 5,058 tok/s. "TTFT grows roughly linearly with context because DeltaNet/SSM state is not prefix-cacheable" | measured | CE/FR [E]: scaled by bandwidth (3090 ≈ 3.1× L4), about 300 tok/s on 4×L4, "in line with our 321-431". It assumes no PCIe all-reduce limit; their GPUs have NVLink pairs |
| vLLM DP docs (EC L174; FR L313) | `--data-parallel-size 2 --tensor-parallel-size 2` = 4 GPUs; works with dense and MoE; `--max-num-seqs` per DP rank; each DP engine has its own KV cache | vendor | — |
| SGLang DP guide (EC L175; FR L313) | Native `--dp-size` "highly not recommended for use right now"; use the SGLang Model Gateway (`python -m sglang_router.launch_server ... --dp-size N`). "when there is enough GPU memory, always favor data parallelism" | vendor | — |
| vLLM TP docs mirror (EC L176; FR L313) | PP possibly preferable without NVLink (L40S example); no numbers | snippet | — |
| vLLM recipe 2× RTX 5090 TP2 (EC L178-184; FR L313; CE L325-328) | 262,144 context. Weights per GPU / KV tokens: FP8 14.28 GiB / 377,456; Inferact NVFP4 12.02 GiB / 445,875; unsloth NVFP4 10.64 GiB / 920,517. MTP acceptance 0.771 / 0.897 / 0.788. 1× 5090: `--language-model-only` raised KV from 91,022 to 135,926 | measured | Memory reference for TP2 |

### 2.9 Weight-format speed (elsewhere)

| Source | Numbers | Class | Transferability |
|---|---|---|---|
| Marlin paper (arXiv 2408.11743) (WQ L108-120; FR L126-132) | Kernel near the ideal 3.87× vs FP16 at batch 1, close up to batch 16-32, falls toward about 1.5× at batch 128; memory-bound below about batch 50-64. End-to-end vLLM speedup vs FP16 at batch 1/8/16/32/64/128: Llama-2-7B/A10 2.93/2.90/2.74/2.26/1.78/1.20; Llama-2-70B/4×A6000 2.84/2.84/2.67/2.02/1.74/1.23; Yi-34B/A100 2.90/2.79/2.69/2.36/1.71/1.18 | academic | WQ [E]: on L4 the falloff should start at larger M |
| "Give Me BF16 or Give Me Death" (arXiv 2411.02355) (WQ L121-132; FR L134) | Synchronous: W4A16 best (2-3× lower cost per query, 1.5-2.5× lower latency). Async continuous batching: "W8A8 formats (INT8 and FP8) give the highest throughput", W4A16 "remains competitive and sometimes beats W8A8". Code-completion QPS vs BF16: 70B/4×A6000 INT8 1.91×, INT4 1.92×; 4×A100 1.87× / 1.64×; 4×H100 INT4 1.55×, FP8 1.77×. "Crossover point depends on model size, hardware, and task" | academic | No Ada |
| NVIDIA forum, DGX Spark (WQ L133-138; FR L134) | Qwen3.8-27B, vLLM 0.27.1, 16 concurrent: unsloth NVFP4 (23.4 GB; "4-bit MLP + 8-bit attention + FP8 KV"; Marlin fallback) vs official FP8 (30.9 GB, block 128). Output throughput +29-34 %: decode-heavy 1k → 8k 99.47 → 132.07; balanced 104.44 → 134.41; TPOT 160.19 → 120.60 ms. GB10 thread: vLLM "still falls back to weight-only FP4 via Marlin" | measured | FR: "for L4 the DGX Spark result and our measurements are more relevant" (bandwidth-limited part) |
| bottlecapai ThinkingCap card (WQ L144; FR L141) | Qwen3.6-27B, RTX PRO 6000 Blackwell, vLLM 0.25, batch 16: FP8 decodes about 55 % faster than bf16 (36.4 vs 23.4 tok/s); MTP adds about 1.8-2.0× | measured | Blackwell |
| sakamakismile (WQ L145; FR L142) | Qwen3.8-27B NVFP4 on 4× RTX PRO 2000 Blackwell TP4: 69.0 at c=1, 202.3 at c=4, 377.8 at c=8 (self-reported) | snippet / claim | — |
| Towards AI L40S (WQ L146; FR L143) | FP8 cut chat TPOT 23 → about 15 ms; concurrency-16 throughput 633 (bf16) → 949-971 (page 403, model unknown) | snippet | Weak |
| computingforgeeks (WQ L147-150; FR L144) | A10G, vLLM 0.30.0, Qwen3.5-9B, 1024 in / 128 out: BF16 117, FP8-dynamic 338, GPTQ 363, AWQ 363 tok/s; GSM8K strict FP8 87.5 vs GPTQ 81.8 | snippet | Ampere (FP8 is weight-only Marlin there) |
| vLLM PR #59232 / L40S tuning repo (FR L120; CE L424-426) | Triton block-FP8 tuning for L20: about 1 % in the latest revision (5-31 % in earlier ones). L40S FP8 Triton config tuning: +30.5 % at c=1, +3.8 % at c=128 | user report | Ada (L20 / L40S) |
| Qwen3.8-27B-FP8 discussion #16 (CE L333-335) | RTX 4090 48 GB, vLLM v0.27.1: "Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!" Workaround: tuned configs (massif-01/vllm_benchmark_block_fp8, kryoz/4090-48gb-vllm-fp8) | measured log | Same warning as ours |

### 2.10 Accuracy by weight format (elsewhere)

- **RedHatAI/Qwen3.8-27B-INT4** (WQ L214-225; FR L148-157):
  - Scheme: W4A16 GPTQ with AWQ smoothing (`duo_scaling="both"`), activations unquantized, static per-tensor FP8 KV; 3 seeds (8 for AIME).
  - Kept unquantized: vision, lm_head, embed_tokens, `linear_attn.in_proj_a/b`. The ignore list does not mention MTP layers.
  - Results, BF16 → INT4: AIME 2025 95.42 → 94.17 (98.69 %); GPQA-Diamond 89.23 → 87.88 (98.49 %); MATH-500 83.73 → 83.33 (99.52 %); MMLU-Pro 84.46 → 83.45 (98.81 %); IFEval 92.24 → 91.93 (99.67 %); GSM8K-Platinum 95.75 → 96.77 (101.07 %).
  - 341,574 downloads (CE L344). [vendor, claim]
- **RedHat NVFP4** (WQ L226-229):
  - Scheme: MLP W4A4; attention q/k/v/o, linear-attention `in_proj_qkv`, `in_proj_z`, `out_proj` and the MLP of layers 56-63 in FP8 dynamic; about 24.7 GB.
  - Results: AIME25 96.67 → 95.00 (98.27 %); GPQA 89.56 → 89.22 (99.62 %); MATH-500 100.72 %.
- **RedHat MXFP4:** AIME25 96.67 → 92.50 (**95.7 %**); GPQA 87.71 (97.9 %); MATH-500 101.2 % (WQ L230).
- **Baseline variance:** RedHat's two BF16 baselines for the same model differ (AIME25 95.42 vs 96.67; GPQA 89.23 vs 89.56), so run-to-run noise is about 1 point, the size of the INT4 deltas (WQ L231).
- **nvidia/Qwen3.6-27B NVFP4 vs FP8:** AIME25 93.1 → 92.7; GPQA 86.0 → 85.5; HLE 21.7 → 21.8; MMLU-Pro 86.1 → 86.3; SciCode 44.8 → 44.5 (WQ L232).
- **soyrsoyr Qwen3.6-27B FP8-dynamic** (n=1, 30 problems): AIME-24 83.33 → 73.33; AIME-25 73.33 → 73.33; MATH-500 86.6 → 86.2. Noisy, a minority result (WQ L233).
- **"Quantization Hurts Reasoning?"** (arXiv 2504.04823v2; average over 5 benchmarks) (WQ L234-247; FR L161):
  - W4 AWQ / GPTQ: QwQ-32B −0.8 / −0.9; R1-Distill-Qwen-32B −0.4 / −1.9; Qwen3-8B −1.3 / −1.5.
  - W3: −4.4 to −6.9, and down to −14.5.
  - W8A8KV8: +0.3 / −0.1 / −0.2. KV4: about ±0.3.
  - Harder benchmarks degrade "up to 4×" more than easy ones.
  - Quantized models "do not generally think longer". No FP8 rows.
- **"Quantization Inflates Reasoning"** (arXiv 2606.25519v2; CoT token inflation vs BF16, INT4 g128) (WQ L248-260; FR L161):
  - Qwen3-30B: GPTQ +3.7 %, AWQ +14.4 %, RTN +43.0 %, HQQ +3.9 %.
  - Qwen3-4B: GPTQ +12.0 %, AWQ +8.2 %.
  - INT3: GPTQ +19.0 %, AWQ +32.8 %.
  - BFCLv4: 5351 → 5488 CoT tokens.
  - "1.1x CTIR increase raises p90 TTFT from 1050 s to 1400 s".
- **"Give Me BF16"** (WQ L261-268):
  - W8A8-FP "essentially lossless"; W8A8-INT drops 1-3 % per task.
  - W4A16-INT recovers 99.36 % (V1) and ≥96 % (V2). R1-Distill recovers >99 %, except small INT4 models (Llama-8B 97.2 %, Qwen-1.5B 93.5 %).
  - RULER ≥98 %. GPTQ beats AWQ by 2.9 points on Arena-Hard (8B).
- **Red Hat R1-distill blog:** FP8 / INT8 near-perfect; INT4 97 %+ for ≥7B (WQ L269, snippet).
- **Modules kept in high precision by good recipes** (WQ L270-274):
  - Official FP8 keeps lm_head, the MTP head, vision and GDN `in_proj_a/b` in bf16 ("48-wide output doesn't tile a 128 block").
  - The AzatAI FP8-dynamic recipe ignores all of `linear_attn`.
  - The RedHat Qwen3.5-9B W4A16 reportedly leaves linear attention unquantized (19.3 → 11 GB).

### 2.11 KV-cache dtype (elsewhere)

- **vLLM FP8 KV blog, accuracy** (2026-04-22; uncalibrated scale 1.0, the worst case) (WQ L380-385; FR L183-185):
  - Qwen3.5-27B reasoning "at most 0.7 points; lowest recovery was 99 % (AIME25)".
  - Qwen3.5-27B MRCR up to 1M: "fully matches baseline".
  - Qwen3-30B-A3B-Thinking: at most 1-2 points (GPQA-D 97 %).
  - Before the FA3 two-level accumulation fix, 128k NIAH fell 91 % → 13 %; after the fix, 89 %.
- **vLLM FP8 KV blog, performance (H100)** (WQ L386-392; FR L183):
  - KV storage halved; decode ITL slope 54-80 % of BF16.
  - Break-even about 7k tokens (Llama-3.1-8B); the blog recommends BF16 below about 7k contexts.
  - c=8, about 20k in / 2k out: 450.3 → 517.5 tok/s (+14.9 %).
  - The blog does not cover Ada, Triton or FA2.
- **vLLM quantized-KV docs** (WQ L393-398; FR L187): fp8_e4m3 and e5m2 on CUDA 11.8+; default scales 1.0; calibration via llm-compressor recommended; per-head scales only with the Flash Attention backend; `--kv-cache-dtype-skip-layers` keeps sensitive layers in native dtype.
- **FA2 cannot do FP8 KV** (WQ L399-400; FR L185):
  - vLLM's FA backend raises "FlashAttention does not support fp8 kv-cache on this device".
  - A Qwen3-Next user fixed bad output by switching FlashInfer → Flash_attn, but that backend "doesn't support KV-Cache FP8".
- **HF discussion #182:** FP8 KV at 256k gave MRCR 0.717 vs BF16 KV 0.833 (n=15) (FR L185; CE L439).
- **Papers:** KV4 within about 1 point, including >10k outputs (QwQ-32B −0.3 / +0.3); W8A8KV8 about lossless (WQ L407).
- **club-3090 #773 INT8 KV:** +76.5 % short, −71.9 % at about 58k (FR L189; EC L289).
- **xz (ARC-AGI-3):** "very little NLL penalty to use nvfp4 kv". Not applicable on L4 (FR L189; CE L230).
- **SGLang Qwen3.5 cookbook:** Qwen checkpoints have no calibrated KV scales, so the default 1.0 "may noticeably reduce accuracy on reasoning-heavy tasks" (EC L287; FR L187).
- **Per-token KV for this model:** fp8 32 KiB (32.8 KB), bf16 64 KiB (65.5 KB), i.e. 8 / 16 KiB per GPU at TP4 (SGLang cookbook; FR L183; EC L9).

### 2.12 SSM state dtype (elsewhere)

- **SGLang cookbook Qwen3.8-27B** (EC L252; FR L343):
  - Slot size: fp32 153.9 MB, bf16 78.4 MB.
  - Speed: NVFP4 + EAGLE 152.9 (fp32) vs 144.5 (bf16) tok/s/user; FP8 + EAGLE 106.3 vs 116.1.
  - "Treat bf16 as an accuracy gate"; GSM8K across 202 cells 93.18-95.15 %.
- **AMD recipe:** `SGLANG_MAMBA_SSM_DTYPE=bfloat16` (EC L253).
- **vLLM 25K TPS recipe:** `--mamba-ssm-cache-dtype bfloat16` "significantly" raises KV capacity (no number); GSM8K 88 % (EC L254).
- **vLLM `mamba_ssm_cache_dtype=auto`** resolves to bf16, not the checkpoint's fp32 (EC L255).
- **Nemotron 3 Super:** FP32 → FP16 direct cast "can result in up to 40 % increase in verbosity when combined with W8A8". **Nemotron 3 Ultra:** FP16 with stochastic rounding preserves FP32 accuracy; FP8 E4M3 degrades. Both are different models, and these are snippets (EC L256).
- **Friendli Qwen3.6 card:** outputs "identical with and without" bf16 mamba cache (claim) (EC L257).
- **SGLang #36889:** bf16 state cost about 6 % single-stream through shorter accepts (SD L186).
- **Qwen3-Next cookbook:** `--mamba-ssm-dtype` default float32; bf16 saves memory, fp32 is more accurate (snippet; WQ L275).
- **Franzen (ARC-AGI-3):** bf16 SSM + `SGLANG_MAMBA_CONV_DTYPE=bfloat16` (FR L343; CE L197, L202).

### 2.13 CUDA graphs, overlap, chunked prefill, batching (elsewhere)

- **vLLM hybrids, CUDA graph mode** (EC L278-279, L380; FR L349):
  - Default `FULL_AND_PIECEWISE`, needed because the Triton GDN kernels have CPU launch overhead.
  - H100, v0.10.2, 32k in / 128 out, V1 vs V0: Nemotron-Nano-12B-v2 +2-18 %; granite-4.0-h-tiny up to +91 %. PIECEWISE alone was worse than V0 at low concurrency.
- **vLLM `num_cache_lines >= batch` assert:** fires when the graph capture size exceeds the mamba cache. Fix: lower `--max-cudagraph-capture-size` (default 512; PR #34571). 1× 5090 runs out of memory in graph capture without `--enforce-eager` (EC L279, L336; SD L156).
- **SGLang overlap (zero-overhead) scheduler:** 1.1× (v0.4 blog, 2024, not hybrid-specific). For hybrids, overlap requires extra_buffer or extra_buffer_lazy (EC L381; FR L349).
- **SGLang cookbook chunk tip:** `--chunked-prefill-size 2048`, "since 8192-token chunks stall decode about 600ms at a time". H200 uses 32768; RTX 5090 draft rows use 1024/512 (EC L283; FR L353).
- **vLLM:** `--max-num-batched-tokens 16384` about +8 % total TPS/GPU (GB200 disaggregated) (EC L284, L382).
- **Mem-fraction pins for Qwen3.8-27B:** DGX Spark 0.80 (0.85 caused earlyoom kills); RTX 5090 0.88-0.94 by drafter and SSM dtype; 3090 community 0.93 (EC L273, L275; FR L353).
- **Franzen (RTX PRO 6000):** chunked prefill 8192 / max-prefill 16384; `--cuda-graph-max-bs-decode 10 --cuda-graph-bs-decode 1 2 4 7 8 9 10` (CE L195-196; FR L349, L353).

### 2.14 Prefix caching for hybrids (elsewhere)

- **vLLM V1 unified allocator:** the attention block grows to align with mamba pages (e.g. 672 tokens on Nemotron-Nano-12B-v2); mamba prefix caching is "experimental" (EC L263).
- **vLLM Qwen3.5 recipe:** prefix caching with Mamba is experimental "align" mode. LMCache requires `--mamba-cache-mode align` for GDN ("does not support the all mode"); align snapshots state only at scheduler-step ends (EC L264).
- **vLLM #40696** (0.19.1, Qwen3.5-4B): block 528 tokens, so prompts under 528 get 0 % hits. Measured: 479 tokens about 0 %, 552 tokens 95.4 %, 979 tokens 53.7 %. A maintainer called it a design limitation (EC L265; FR L355).
- **HF discussion #169** (vLLM v0.30.0, RTX PRO 6000): hybrid cache blocks are 1,584 tokens (1,616 with MTP K=4), so prompts under about 5K never hit. With DFlash2 "the lookup granularity becomes 6,592 tokens" (CE L366-369; FR L355).
- **club-3090:** "DeltaNet/SSM state is not prefix-cacheable" for TTFT (CE L404; FR L355).
- **HF discussion #178:** thinking loops of 20k-55k tokens fixed with `thinking_budget`; SGLang's default top_k is −1 vs the card's 20 (CE L440-441; FR L355).
- **Franzen:** prefix-cache reuse 93.43 % of prompt tokens (CE L205).

### 2.15 Kaggle 4xL4 ARC-AGI-2 community (CE Table A L32-46; FR L401-416)

| Who | Engine and weights | Spec / KV / flags | Concurrency, context, budget | Throughput | Score |
|---|---|---|---|---|---|
| NguyenThanhNhan (topic 745787, 10-05) | SGLang, Qwen3.8-27B **FP8** | `--speculative-algorithm EAGLE --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`; `--kv-cache-dtype fp8_e4m3`; `--attention-backend flashinfer`; `--linear-attn-backend triton`; `--chunked-prefill-size 2048`; `--max-prefill-tokens 4096`; `--mamba-radix-cache-strategy extra_buffer`; `--mamba-full-memory-ratio 0.5`; `--mamba-ssm-dtype float32` | xhigh, **1 pass**, 65k cap, forced commit after **49k** thinking tokens; concurrency not disclosed (asked, not answered) | not stated; Darren computed ≥360 tok/s would be needed [ESTIMATE] | 2 submissions, "pure reasoning", about 23 LB [MEASURED, screenshot] |
| NguyenThanhNhan later (topic 751344, 10-21/22) | SGLang, model not stated; reads **~180 GB** from `/kaggle/input`; "mtp draft setup" | — | about 12 h run | not stated | **52.08** (10-18) → **65.97** (10-20); two 0.00s caused by GCSFUSE |
| Adarsh `arc-agent-sub-q38-direct-b98k-c16` | vLLM OpenAI server (wheelhouse v3 + keithtyser overlay), our FP8 repack | Startup ladder: `--kv-cache-dtype fp8 --language-model-only --speculative-config {"method":"qwen3_5_mtp","num_speculative_tokens":2}` + `--reasoning-config`, falling back to plain serving; TP4, prefix caching, 0.90 | max_model_len 131,072; 16 seqs; 2 samples; xhigh; T 1.0 / top_p 0.95 / top_k 20; think 98,304; max_tokens 106,496 | config default `"tok_per_s": 140` (planning assumption) | **21.53**; "re-ran with a small change, score dropped to 13" |
| Adarsh agent | same | same | 81,920; c10; think 40,960; max_tokens 49,152 | — | about 15 |
| russcore | vLLM 0.19.0, FP8 repack | **No MTP, no FP8 KV**; `--dtype bfloat16 --language-model-only --enable-prefix-caching`; `VLLM_USE_FLASHINFER_SAMPLER=0`; `qwen3_coder` tools | 65,536; 24 seqs; 2 samples; T 0.6; xhigh; direct 57,344; turn 40,960; up to 10 tool turns | not published | about 10 (Anil: "try setting it to 64k") |
| khoinguyennguyen | vLLM, mikedan7 FP8 official | fp8 KV + `--attention-backend TRITON_ATTN` + `VLLM_FORCE_ATTN_BACKEND=TRITON_ATTN`; MTP 3 (the text says 6); `--no-async-scheduling`; prefix caching; `preserve_thinking` | 0.96; 262,144; 16 seqs | "Startup after preload: 90-240 s on 4×L4"; 2-5 min with page-cache preload vs 6-10 min without | none; "At 2 agents/task it takes too much time" |
| delongyang | vLLM 0.19, mikedan7 | same + MTP **6**; `--no-async-scheduling`; default effort medium | 131,072; 16; util 0.96 | "5-12 minutes to load the weights and capture CUDA graphs" | none |
| yjsh2003 HyperQwen v7 | patched vLLM | "W4A16/MTP3, TP4/C4, FP8 KV, full 131072 context, 60K/xhigh" | c4 | — | — |
| Souvik | HF transformers on the FP8 repack | — | — | load fails (`TypeError ... NoneType`) | 4B TTT 28.06 |
| Dominic (analog) | vLLM 0.26, Qwen3.5-MoE 35B-A3B (Nex-N2-mini / Aquila-mini), fp8 per-tensor at load, Triton MoE | — | 4×L4 | **438 / 452 tok/s** [MEASURED]; "W4A16 ... 20 % higher throughput" | 0/64 on eval |

**Throughput statements** (CE L76-114; FR L416):
- **Darren** [CLAIM]: "Has anybody got like an aggregate above 500 tok/s?" Later: his team "built every pipeline on top of the 4 L4 ... it fixes some issue that is present in the library code itself".
- **Anil** [ESTIMATE]: "350 tokens/s on a 4x L4 seems plausible for the FP8 quantized model"; `42*12*60*60/(64*1024*330/350) ~ 30`; "median around 64K" at xhigh.
- **TheDivine Aurora:** "at around 300-500 tps i had ~19-20 % score. only attempted half the dataset ... 40 % is possible with 600-800 tps ... it just takes 128k - 256k context window" [MEASURED / ESTIMATE]. Also: the October jump came from "an inference setup with maximum throughput".
- **Kris Dreemurr:** "median 70k tokens for pass@2 (so 35k per slot) gets %45 solves or about 52~ out of 120 ... I cant fit it in the box"; NVARC "run on all 240 tasks in about 4 hours".
- **Adarsh:** 27B at xhigh "is just so slow that it never reaches most of the tasks"; "im milking the life out of the inference speed".
- **Hiểu Vy:** "they've found ways to make their models run faster"; 26.81 LB with 27B; "80 % accuracy" on 120 validation tasks (whether within 12 h is unanswered).
- **mikelou1** (66.39 on 10-17, 69.72 on 10-21) [CLAIM]: "at around 4xL4 speed I got Qwen to 87 %"; "79 % ± 5 % in CV"; "Not a NVARC ensemble"; "a tiny kernel issue I found that made my program score only 40 % (instead of the 60 % projected by CV)". Model not stated.
- **Darren:** "$0.4469/task … roughly 170k tokens per task … maybe 85k-90k ish" per pass [ESTIMATE].
- **Cho Royou:** "we probably wouldn't be able to use a sufficiently long context window anyway".
- **FOYSAL:** "only a small amount of VRAM remains available for the KV cache"; NVARC-vs-Qwen correlation 0.43.
- **ARC Prize official:** Qwen3.8-27B XHigh 42.36 % semi-private, $0.447/task; by effort xhigh 42.4 %, medium 13.2 %, low 22.8 %, none 1.5 %; Baseten at $0.425 / $2.55 per 1M input / output tokens (CE L142-144).

**L4 engine traps in public notebooks** (CE L128-140; FR L189):
- **delongyang:**
  - "FlashInfer's FP8 attention kernels exist only on SM90+ … L4 is SM89", so fp8 KV crashes with "illegal memory access" unless `TRITON_ATTN` is used.
  - "The MTP drafter does not inherit `--attention-backend`" (vLLM PR #53450); only `VLLM_FORCE_ATTN_BACKEND` pins it.
  - "MTP acceptance rate is 0 % — a separate bug" (#44879); "MTP ≥ 2 was the crash-prone region historically".
- **khoinguyennguyen:** "Without this variable the main model runs Triton while the drafter tries FlashInfer and dies with an illegal memory access mid-generation".
- **Dominic:** "`VLLM_USE_FLASHINFER_SAMPLER=0` is necessary — the image has no linkable `libcuda.so`"; "Don't install wheels under `/kaggle/working`" (7 GiB committed as output, then ERROR at teardown).
- **Adarsh:** "Untested-on-L4 options (MTP, --reasoning-config) must never cost the whole run".
- **Shared checkpoints:** our FP8 repack is the de-facto shared 27B checkpoint on Kaggle (Adarsh, russcore, Souvik, and two ARC-AGI-3 notebooks); `mikedan7/qwen3-8-27b-fp8-official` is the second.

**CE/FR inference [E]:** about 250-430 tok/s is what TP4 serving of the 27B achieves on 4×L4 today. That is roughly 30-45 % of a naive bandwidth bound (4 × 300 GB/s over about 5-6 GB of weights per GPU per step ≈ 50-60 steps/s × 16 ≈ 800-950 tok/s) (CE L147-150; FR L416).

### 2.16 ARC-AGI-3 (RTX PRO 6000) (CE §2 L169-279; FR L395)

- **Franzen, Milestone 2 (LB 27.89):**
  - Model path: Qwen3.6-27B FP8 → Qwen3.8-27B FP8 → Flash-Next; "Each switch brought a substantial improvement".
  - Quantization: RadixArk NVFP4, then Intel W4A16 AutoRound ("similar quality and throughput, while leaving more VRAM available for the KV cache").
  - Draft: albucino INT4 MTP drafter. BF16 PLE kept in host RAM. Engine: Pennyroyal v2.5.3 at `d00d88e`.
  - Patches: low-M BF16 GEMM, a spec-state memory budget fix, a Marlin scale-dtype fix, a prefix-cache patch, and a checkpoint-prefetch patch (parallel readers, one shard ahead); the model was resharded.
- **Franzen settings:**
  - 10 streams; harness context 131,072; max output 12,288; server context 139,264; drain 59,392.
  - FP8 E4M3 KV; Mamba cache 60 slots; mem fraction 0.96; chunked prefill 8,192; 3 speculative steps; temp 0.7 / top_p 0.95 / top_k 20.
- **Franzen exact launch:**
  - Core: `--tensor-parallel-size 1 --dtype bfloat16 --quantization auto-round --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.96 --context-length 139264 --page-size 64 --max-running-requests 10 --chunked-prefill-size 8192 --max-prefill-tokens 16384`.
  - CUDA graphs: `--cuda-graph-max-bs-decode 10 --cuda-graph-bs-decode 1 2 4 7 8 9 10`.
  - Mamba and linear attention: `--mamba-ssm-dtype bfloat16 --max-mamba-cache-size 60 --mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64 --mamba-backend flashinfer --linear-attn-decode-backend flashinfer --linear-attn-prefill-backend flashinfer`.
  - Model and parsing: `--moe-runner-backend auto --ple-offload-embedding --reasoning-parser qwen3 --tool-call-parser qwen3_coder --default-chat-template-kwargs '{"preserve_thinking":true}'`.
  - Scheduling and loading: `--schedule-policy lpm --enable-cache-report --enable-metrics --weight-loader-prefetch-checkpoints --weight-loader-drop-cache-after-load`.
  - Speculation: `--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`, draft quantization `compressed-tensors`, plus the FR-Spec 64k hot-token map.
  - Env: `SGLANG_MAMBA_CONV_DTYPE=bfloat16`; `SERVER_STARTUP_TIMEOUT = 12*60`.
- **Franzen metrics** [MEASURED]: prefix-cache reuse 93.43 %; P90 effective prefill 10,552.6 tok/s; **P90 aggregate decode 836.6 tok/s**; harness-side generation 609.0 tok/s; prefill time fraction 15.25 %.
- **Tong Hui Kang's comparison** (dfranzen 27.89 / lordhansolo 23.84 / sirikilohit 22.53):

  | | dfranzen | lordhansolo | sirikilohit |
  |---|---|---|---|
  | Engine | SGLang Pennyroyal v2.5.3 | vLLM 0.29.1rc1 nightly e975732 | SGLang Pennyroyal v2.5.0 |
  | GPU weights | 73.64 GB | 71.94 GB | 73.57 GB |
  | Host offload | 95.37 GB | 96.55 GB | 95.68 GB |
  | KV dtype | fp8_e4m3 | fp8_e4m3 | fp8_e4m3 |
  | KV pool (tokens) | 1,011,264 | 1,417,100 | 1,004,288 |
  | Context (server / harness) | 139,264 / 131,072 | 147,072 / 127,488 | 69,632 / 69,632 |
  | Streams | 10 | 14 | 16 |
  | Start to ready | 531 s | 544 s | 615 s |
  | Peak decode | **946** | **1,135** | **1,159** tok/s |
  | Temperature | 0.7 | 0.6 | 0.6 |

- **Comments on that thread:**
  - rellik13: KV "sat somewhere between 91-97 % full and requests queued for cache space, so the memory was the actual limit"; fp8 KV and more history (trims 37k → 27k became 57k → 45k) "took the score from 14.5 to 22.5".
  - NNbody: "redundancy in the mamba slot allocation for Qwen 3.8 Flash Next".
  - Son Pham: a REAP-pruned Qwen3.8 opened "an entire untapped area".
- **keithtyser Flash-Next NVFP4:** NEXTN 3-token MTP, 32K context, 8K batched-token cap, 8 vLLM sequences, CUDA graphs, prefix caching disabled, 28-way game concurrency.
- **Ya Xu:** Qwen3.8-27B 8-bit gives "a consistent 2x score on the local 25 dataset" vs Qwen3.6 27B.
- **Tufa Labs (Milestone 1):** RTX PRO 6000 96 GB, 9 h, 110 games, Qwen 3.6 27B FP8 vLLM.
- **What transfers to L4, per CE (L254-268):**
  - Transfers: prefix-cache-aware admission, trim hysteresis, end-of-prefill Mamba checkpoint retention, `--schedule-policy lpm`, `--page-size 64`, `--mamba-track-interval 64`, FP8 KV, and parallel prefetch / reshard.
  - Does not transfer: SM120 env vars, NVFP4, FlashInfer GDN, Pennyroyal QSA and TP1.
- **CE inference:** the 946-1,159 peak is about 2.5-3× the best 27B numbers on 4×L4, mostly from hardware (about 1.8 TB/s on one GPU, no TP all-reduce) (CE L271).

### 2.17 Flash-Next elsewhere (CE §4 L461-524; FR L395)

- **4× RTX 3090** (tonyd2wild; vLLM nightly, TP4 + EP, W4A16 AutoRound experts, FP8 n-gram table on NVMe, MTP 3, fp8_e5m2 KV because "FP8 e4m3 KV does not compile on 3090s"):
  - single stream: prose 109.5, code 145.8 tok/s;
  - **6 concurrent agents: 252.6 tok/s aggregate** (317 sustained / 570 peak).
  - Bandwidth-scaled estimate for 4×L4 at 6 streams: about 80-190 tok/s [E].
- **Hiểu Vy:** 26.81 with 27B vs **8.61** with Flash-Next.
- **Adarsh:** "L4 GPUs based on Triton architectures aren't that fast as hopper ones".
- **Darren:** offloading the embedding to RAM might give Flash-Next "a better shot at 4 L4".
- **amn:** an agent claim without numbers; amn's high LB is not shown to be Flash-Next; amn's 0.00 runs read "~110 GiB".
- **Upstream sglang PR #36968:** "QSA sparse decode on NVIDIA SM89 requires classic FlashAttention / FA2", tested on 4× L20 with FP8 (about 173 GB); no tok/s.
- **Sizes:** "125B with 6B activated, plus 51B n-gram embedding and 4B MTP"; FP8 172.78 GiB, BF16 335.28 GiB; "4xH100 with plain TP4 runs out of memory without that offload". Offloading the n-gram embedding costs −0.07 % throughput (LMSYS, H200 TP4).
- **Card claims:** GPQA 91.7 vs 89.2; HLE 35.9 vs 30.8.
- **No official ARC result** (arcprize page 404). The licence question was unanswered by the hosts.
- **Expert-touch arithmetic [E]:** batch 16 touches about 27 % of each layer's experts (1 − (502/512)^16); batch 64 about 72 %.
- **Qwen3.5-MoE 35B-A3B on 4×L4:** 438-452 tok/s.

### 2.18 /kaggle/input storage slowdown (CE §5 L528-596; FR L361-376)

- **Mount change:** from about 2026-10-21 part of `/kaggle/input` is on `FUSE.GCSFUSE` instead of NFS.
- **amn's NFS probes:** 0.55 GiB/s (10-19 10:54), 0.95 (10-20 03:26), 0.32 (10-21 03:21), 0.040 (04:35), **0.026 GiB/s** (05:37). On 10-22: single thread about 10.5 MB/s, 16 threads 107 MB/s.
- **Jeroen Cottaar:** 100.5 GiB in 10 shards: only 3 shards loaded after 7 h 22 min (1.8-2.8 h per shard), vs 88 s on NFS on 10-17. "Reads through the mount don't fill the page cache".
- **Effects on scores:**
  - Jeroen: "usually scoring 0.00, and when it doesn't it scores 5-10 % lower".
  - Kha Vo: "worse than expected by 10 points" plus a Kaggle Error.
  - CPMP: lost one submission to a Kaggle error after 12 h.
  - mp: "~10 points lower".
  - amn: a second 0.00 (about 110 GiB not read).
  - NguyenThanhNhan: about 180 GB; 45-minute server timeout ("normally … at most 15 minutes"); save runs loaded in 875-1300 s; two 0.00s ("server startup never finished in time … placeholder submission jsons were returned").
- **Mitigations:**
  - mp's (untested): copy once to local disk if FUSE.
  - Franzen: reshard, parallel readers, prefetch one shard ahead, a background prefetcher, 12-minute startup timeout.
  - sirikilohit: "parallel PLE row copy on 16 threads", "prefetch once".
  - khoinguyennguyen: page-cache preload with a `ThreadPoolExecutor` in 64 MiB chunks.
- **Load-time arithmetic [E]:**

  | Checkpoint | 16 threads, ~107 MB/s | 1 thread, ~10.5 MB/s | NFS best, ~0.95 GiB/s |
  |---|---|---|---|
  | 27B INT4 + DFlash2 (23.3 GB) | ≈ 3.6 min | ≈ 37 min | ≈ 23 s |
  | 27B FP8 repack (30.9 GB) | ≈ 4.8 min | ≈ 49 min | ≈ 30 s |
  | Flash-Next W4A16 + FP8 PLE + drafter (~121 GB) | ≈ 19 min | ≈ 3.2 h | ≈ 2 min |
  | Flash-Next W4A16 + BF16 PLE + drafter (185 GB) | ≈ 29 min | ≈ 4.9 h | ≈ 3 min |

- At Jeroen's GCSFUSE loader rate (about 1-1.5 MB/s), even 23.3 GB takes hours.
- Host RAM is 202.6 GB, so a 23-31 GB checkpoint fits in `/dev/shm` or the page cache. Local disk size and the `/dev/shm` cap are unknown.

### 2.19 Bugs reported elsewhere that touch our stack

| Bug | What happens | Where seen | Doc's reading for W1 | Source |
|---|---|---|---|---|
| **SGLang #36548** | Qwen3.8-27B + DFlash2 + concurrency: context mixes across requests; accuracy about 50 % vs about 90 % expected. First divergence in the layer-0 SSM transition (possibly a DECODE vs TARGET_VERIFY kernel mismatch). A dense BF16 lm_head cut wrong answers 111/304 → 1/304 but did not remove the drift. The reporter could not reproduce it on 0.5.18 with MTP/EAGLE. No fix PR | issue | Our model, our drafter, our concurrency. RedHat INT4 already keeps lm_head BF16, so the main mitigation is partly in place, but the drift remains. The reporter's quant and engine version are not in the notes. Open | FR L264; SD L150, L235 |
| **SGLang #38009** | DFlash2 + thinking: greedy output diverges from target-only at token 30 (4× RTX 3090 TP4). Another user saw accept length vary across container restarts | issue | The closest TP4 PCIe analog. Our greedy drift is similar for MTP and DFlash2 (152 vs 155), so greedy prefixes cannot separate #38009 from ordinary drift; an accuracy A/B is needed | FR L265; EC L329 |
| **SGLang #43514** | DFlash + Qwen3.8-27B (v0.5.19/0.5.20, 4× H20 TP4): accept 0.00, every request emits token 0 ("!") until max_tokens, no error. Trigger: grammar-constrained requests + multi-chunk prefill admission with `--enable-mixed-chunk` and `extra_buffer`; about 30 % per round on a used instance, 0/10 on a fresh one. Candidate PR #43518 merge status unconfirmed | issue | We do not use `--enable-mixed-chunk`, and our strategy is `extra_buffer_lazy` [E]. Grammar use must be checked; alarm on accept collapsing to about 1.0 | FR L266; EC L322-325 |
| SGLang #39087 | A quantized or inherited-quant DFlash2 drafter gives accept 1.03, about 38 tok/s, vs 3.71 and about 171 with a BF16 drafter (2×3090 TP2) | issue | Mitigated by `unquant` | FR L267; EC L326-328 |
| SGLang #42917 | W4A16 PPL 9.98 vs vLLM 6.05 from a stale CUDA compute cache (`~/.cache/sglang/nv`); 6.2784 vs 6.2753 after clearing (RTX 5090; 0.5.19-0.5.21) | issue | Mitigated by fresh caches; the risk returns if a JIT cache is shipped | FR L268; EC L330 |
| SGLang #36889 / #38846 | Silent mamba cap (see 2.5) | issue | W1's explicit sizing fixed it | FR L269 |
| SGLang #44355 / PR #44372 | Hybrid + spec + `--disable-radix-cache` may fail to start (the default 48 is treated as a user limit) | snippet | We keep radix on | FR L271; SD L152 |
| SGLang spec + structured-output fix | Resuming from wrong mamba states | snippet | — | SD L153 |
| SGLang #19045 | extra_buffer out of memory in disaggregated decode (397B) before a fix | snippet | — | EC L332 |
| SGLang #20774 | Qwen3.5-35B-A3B with CPU offload: Mamba conv_states dtype mismatch | issue | — | EC L333 |
| vLLM #43559 / #53912 | MTP + prefix caching costs about 20 % accuracy on Qwen3.6-35B-A3B; v0.28.0 still corrupts on hybrid GDN (repeated punctuation, empty responses) | issue | Our 0.19 + MTP-3 + prefix caching gave LB 34.58, so no big loss shown on that version; risk on newer vLLM | FR L272; SD L157 |
| vLLM #54360 | Nightly: MTP and DFlash silently drop prefix-cache hits to 0 on hybrid GDN (worked on 0.24.0) | snippet | Caution on newer vLLM | FR L273; SD L158 |
| vLLM #55533 | MTP pins the sm89 INT4 scheduler at about 3 sequences | issue | Assert real running before any vLLM INT4 + MTP arm | FR L274 |
| vLLM #39573 | MTP ignores `thinking_token_budget` in 0.19 | ours | Workaround needed in any vLLM arm | FR L275 |
| vLLM #44879 / PR #53450 | MTP acceptance 0 %; drafter does not inherit the attention backend; "MTP ≥ 2 crash-prone" | notebook | Applies if the vLLM version changes | FR L276 |
| vLLM #54690 | FP8 draft KV crashes on sm89 (4× RTX 4090, TP4, DFlash, vLLM 0.28.0): first `Unrecognized dtype: auto`, then a native worker crash; no workaround | issue | A vLLM + DFlash2 arm must keep draft KV in bf16 [E] | FR L277; SD L162 |
| vLLM async scheduling race | "accuracy was zero without the fixes" | blog | — | EC L337 |
| vLLM MTP num_speculative_tokens 2 error | Errored for one user in March 2026; the maintainer said it should work | snippet | — | EC L338 |
| vLLM MTP × TurboQuant CUDA graph bug #40880 | Fixed by #40914 | snippet | — | EC L339 |
| vLLM Qwen3.5-4B dtype float32 | GDN-kernel warm-up errors | forum | — | EC L340 |
| vLLM DFlash + PP | Unsupported for Qwen3.5 hybrids (v0.30.0; #60765) | snippet | — | SD L161 |
| vLLM DFlash2 | Requires an unmerged PR (`refs/pull/52816/head`). PR #55601 (DFlash2 prefix-hit mamba index) reportedly merged 2026-10-07, after 0.31.0 | vendor / snippet | — | SD L159-160; FR L290 |
| Forced custom AR at TP4 PCIe | +14.8 % decode but greedy divergence on 4/5 prompts | club-3090 #773 | Do not use for eval runs | FR L278 |
| llama.cpp older CUDA builds | Garbage on GDN layers | discussion | — | EC L348 |

---

## 3. Levers

### 3.0 One row per lever

Evidence classes: measured-ours (MO), measured-elsewhere (ME), estimate (E), claim (C).

| # | Lever | Mechanism | Expected gain as the docs state it | Evidence | Cost / risk | Prerequisites | Status | Source |
|---|---|---|---|---|---|---|---|---|
| 1 | DFlash2 block 8 (current W1) | A block-diffusion BF16 drafter proposes 8 tokens and the target verifies them in one pass (num_steps 1) | +39 % over no-spec at 12 running (404 vs 289); 2.4× at 1; production AL 3.8-4.5; vendor H200 c32 1.01-1.45× | MO + ME (vendor) | #36548, #38009, #43514 correctness bugs open; the drafter's 1.49 GB weights, 2.10 GB draft KV and 1.83-1.97 GB intermediate SSM per GPU cut KV to 881k; long-context draft cost (#54691, vLLM); externally trained drafter decay (academic); Blackwell-only recommended backends | `unquant` flag; fresh caches; no grammar or mixed-chunk; explicit mamba sizing | Tested (production) | OM L255-279; FR L209-222, L246, L264-266 |
| 2 | SGLang NEXTN/EAGLE MTP 3/1/4 | Built-in MTP head drafts 3 sequential steps, topk 1 chain, 4 draft tokens verified | 297 vs 229 no-spec at 8 (+30 %); AL ~3.1 flat across concurrency; FR rank 2. SD Q7 [E]: about 1.4-1.7× aggregate at 16 streams if AL is 2-2.5 and concurrency is not capped | MO (short, capped at 10) + ME | Measured only to 10 running (cap); fewer mamba slots than DFlash2; #36548 not reproduced with MTP on 0.5.18 [U]; thinking AL about 2.5 of 4 elsewhere | Explicit `--max-mamba-cache-size`, real-running assert | Partly tested (E11-03 B, short prompts, cap 10) | OM L251-253; FR L21, L207; SD L247-275 |
| 3 | SGLang MTP 2/1/3 | Shorter chain | SD: "then also try 2/1/3"; evidence band 2-3 steps / 3-4 draft tokens | ME + E | — | — | Untested | SD L130, L267; FR L429 |
| 4 | vLLM MTP `num_speculative_tokens` (3 used; 1, 2, 5, 6 seen) | MTP head in vLLM | Our V: AL 2.67 (T 1.0) to 3.0 (T 0.6), 3.45 at low effort; flat with concurrency. vLLM recipe: under high load MTP lowers throughput | MO + ME | #39573 thinking budget ignored (0.19); #55533 caps sm89 INT4 to about 3 seqs; #43559/#53912 corruption in newer versions; delongyang "MTP ≥ 2 crash-prone"; drafter backend must be forced via `VLLM_FORCE_ATTN_BACKEND` | `VLLM_USE_V2_MODEL_RUNNER=0` (0.19) | Tested at k=3 (LB 34.58); k=2 or mem 0.95 or fp16 SSM in vLLM never measured | OM L228-249, L381; FR L23, L272-276 |
| 5 | No speculation (target-only) | Plain decode; no draft state | A: 289 at 12, 347 at 16, 440.0 at 22 (short context); 282.4 on 12 × 24,576. FR [E]: "will hold more streams" because no drafter or verify state | MO | Lower per-stream speed | — | Tested (short sweep); the cleanest parity baseline | OM L139-151; FR L22 |
| 6 | DSpark drafter | RadixArk drafter, block+1 verify slots | Vendor mean AL 3.62 (below MTP 4.28 and DFlash2 4.80); RedHat card suggests DSpark 7 "for optimal performance" | ME (vendor) | — | — | Untested | SD L14-15, L118; EC L234 |
| 7 | DFlash2 other block sizes or drafter quant | Fewer draft positions (e.g. 3-5) | HF disc: positions 4-7 get "about 0.1 or less"; "reducing to 3 tokens makes DFlash perform similarly to MTP"; z-lab `block_size <= 5` for quantized targets (MLX reason); Table 8: running off the trained block loses τ | ME / C | Off-block loses τ; a quantized drafter in SGLang collapses (#39087) | — | Untested ("only block 8 tested") | OM L287; FR L446; SD L124-125; CE L363-364 |
| 8 | EAGLE-3 | External EAGLE-3 head | AL 2.2 on Qwen3.6-27B; below MTP everywhere on Qwen3.x; 60-token tree 1.9× → 0.6× at c32 | ME | Trails native MTP; long-output decay to 1.1 after ~20K | — | Rejected (FR "excluded on evidence") | FR L25, L252; SD L41-45 |
| 9 | n-gram / suffix decoding | Draft from repeated text | n-gram AL 1.31-1.41 (0.29-0.88×); suffix 1.0-1.28× on SpecBench, up to 5.3× agentic; about 10 % overhead at c64; possible gain on grid copying, no evidence | ME / C | vLLM/Arctic only; no hybrid or reasoning data; MTP + suffix undocumented | — | Rejected (FR) / untested | FR L25, L252; SD L50-53, L163, L256 |
| 10 | ReplaySSM (`--enable-linear-replayssm-spec`) | Replays an input window into an fp32 checkpoint instead of per-draft state snapshots (D=0) | Frees about 1.8-2 GB per GPU [E]; keeps fp32 state | E | Requires GDN/KDA, topk ≤ 1, triton/flashinfer linear decode, `SGLANG_RAGGED_VERIFY_MODE=static`; non-fp32 logs a state-drift warning; "DFLASH + ReplaySSM is KDA-only" | MTP/EAGLE path (not DFlash2) | Untested; dropped for DFlash2; FR: a candidate lever for rank 2 (INT4 + MTP) | FR L345; EC L249, L297-298; OM L379 |
| 11 | `--speculative-adaptive` | Adaptive step size | — | C | Crashes on hybrid GDN Qwen3.5 (sm89); fix unconfirmed | — | Rejected / untested | SD L127 |
| 12 | Long-context or online-distilled drafter | Drafter trained on long sequences or adapted online | Online distillation: AL up to +72 % (41 % average); long-context fine-tune holds τ 3.56 at 32K | ME (academic) | Training needed | — | Untested | SD L36-37 |
| 13 | vLLM `draft_context_window` (PR #61261) | Bounds draft attention at long context | Addresses #54691's 4.4× slowdown | C | vLLM only, proposal | — | Untested | SD L95 |
| 14 | INT4 W4A16 Marlin (RedHat GPTQ + AWQ) (current) | Half the weight bytes of FP8 per decode step | Roofline 1.28× FP8 at B=16, 30k (1.19× at B=32, 1.38× at B=8); whole-run +17-30 % vs V (confounded with engine and drafter) | MO (confounded) + E | Accuracy about −1.25 to −1.35 points on AIME/GPQA (card), possibly more on hard tasks; token inflation +3.7 % (GPTQ) to +14.4 % (AWQ) in one paper; Marlin nears its compute ridge (M ≈ 100) at 12 × 8 = 96 | — | Tested (W1); LB 29.17 confounded with P1 prompts | FR L136, L175-177; WQ L171-192, L283-288 |
| 15 | FP8 block-128 (official / our repack) | 8-bit weights; on sm89, Triton `w8a8_block_fp8_matmul` (vLLM) or Triton via `--fp8-gemm-backend auto` (SGLang) | 0.75-0.85× INT4 ceiling; "near-lossless"; FR rank 3 on vLLM | MO (vLLM) + ME | Default untuned tile configs on L4 for 5 shapes; 7.15 GiB/GPU weights | — | Tested on vLLM; not on SGLang L4 | WQ L53-58, L79, L469; FR L118 |
| 16 | L4-tuned block-FP8 Triton configs | Tuned tiles for N=4096,K=5120; 5120×1536; 8704×5120; 5120×4352; 3584×5120 | L20: ~1 % (latest PR #59232 revision; 5-31 % earlier); L40S: +30.5 % c=1, +3.8 % c=128 | ME (user report) + E | Small at high concurrency | E12 FP8 tuning notebook (built, never run) | Untested | FR L120; WQ L80-83; OM L588-590 |
| 17 | Per-channel FP8_DYNAMIC checkpoint (CUTLASS sm89) | `CutlassFP8ScaledMMLinearKernel` (CUDA ≥ 12.4) instead of Triton block | "could narrow the gap to INT4", unmeasured | E | New checkpoint; the AzatAI recipe ignores all `linear_attn` | — | Untested | WQ L48-52, L82, L486 |
| 18 | Forced FP8 Marlin (`VLLM_TEST_FORCE_FP8_MARLIN=1`) | Weight-only FP8 path | None stated | C | Pre-Ada path | — | Untested | OM L590; FR L118 |
| 19 | AWQ-INT4 (cyankiwi) / AutoRound-GPTQ variants | Other INT4 checkpoints | Evidence leans to GPTQ-based schemes (less inflation; Arena-Hard +2.9) | ME (academic) | — | — | Untested | FR L161; WQ L475 |
| 20 | NVFP4 / MXFP4 | W4A4 checkpoints; on L4 they run as weight-only FP4-Marlin | No speed advantage over INT4; NVFP4 24.7 GB (partly FP8); MXFP4 AIME 95.7 % recovery | ME + E | No FP4 tensor cores on L4; whether compressed-tensors W4A4 takes the Marlin path on sm89 is unverified | — | Rejected | FR L122, L177; WQ L424-456 |
| 21 | INT8 W8A8 | CUTLASS INT8 | Same bytes and tensor rate as FP8 (485 TOPS sparse) | ME + E | Needs calibration; 1-3 % per-task drop; no Qwen3.8-27B INT8 checkpoint published | — | Rejected | FR L177; WQ L463-475 |
| 22 | BF16 weights | — | Roofline about 254 tok/s at B=16 | E | About 10 × 60k streams even with FP8 KV; slowest | — | Rejected | FR L177; WQ L465 |
| 23 | Token-reduced fine-tunes (Swift, ThinkingCap) | Fewer thinking tokens per answer | Swift: "58.3 % fewer thinking tokens" (GPQA-D median; mean 41.0 % at xhigh), "1.95× speed-up"; AIME26 98.67 → 94.00 (Swift 1.5: 96.00, 22,014 → 13,203 tokens). ThinkingCap: "cuts reasoning tokens by 37 %" | C | Accuracy loss on AIME; separate licence above US$1M revenue (Swift); gated PolyForm licence (ThinkingCap); no ARC result | — | Untested | CE L371-378 |
| 24 | fp8_e4m3 KV (current) | Half the KV bytes | Doubles capacity; roofline +25-30 % at 16 × 30k (saves about 13 ms of a 55 ms step); vLLM H100 c8 20k/2k +14.9 %; negligible below ~7k | ME + E | Default scale 1.0; Ada kernels untested for >32k reasoning accuracy; MRCR 0.717 vs 0.833 at 256k (n=15) | — | Tested (always on); bf16 KV never A/B'd on L4 | FR L183-189; WQ L370-418 |
| 25 | Calibrated KV scales | Use the INT4 checkpoint's 32 scale tensors (0.021-0.254) | Accuracy, not speed | C | Whether they load at runtime is unverified; both engines log "no scaling factors… 1.0" | — | Untested | OM L377; FR L187 |
| 26 | INT8 / KV4 / nvfp4 KV; `--kv-cache-dtype-skip-layers` | Lower KV precision, or keep sensitive layers native | INT8 KV +76.5 % short / −71.9 % at 58k (3090); KV4 within about 1 point; nvfp4 KV "very little NLL penalty" (not on L4) | ME / C | — | — | Untested | FR L187-189; WQ L397, L407 |
| 27 | SSM state dtype fp16 (current) vs bf16 vs fp32 | Halves the state slot (153.9 → 78.4 MB) | fp16 raised the mamba cap 6 → 13 at the same memory; W1 fp16 416.5 (8/8) vs W2 bf16 412.1 (7/8); fp32 W3 KV only 598,712 vs 881,336 | MO | No public 60k accuracy study; Nemotron FP16 cast +40 % verbosity with W8A8; cookbook "treat bf16 as an accuracy gate" | — | Tested (fp16 production) | FR L341-343; OM L311-316 |
| 28 | Radix strategy extra_buffer_lazy (current) vs extra_buffer vs no_buffer vs radix off | State slots per request S: lazy 4, extra_buffer + overlap 5, no_buffer 3, off 1 | lazy saves 1 slot per request vs extra_buffer; no_buffer saves 2 more but disables overlap (about −10 % [E]) | MO (source-verified slots) + ME | #43514's trigger used extra_buffer; jpezzulli: 4 held but cap computed for 5 | `--page-size 64` + FLA kernels for extra_buffer | lazy tested; others not compared at production | FR L323; EC L232-237, L299 |
| 29 | `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` | Frees 1 slot for extra_buffer strategies (lazy + this = 3) | Capacity | C (vendor) | — | — | Untested | FR L323; SD L185, L268 |
| 30 | Explicit `--max-mamba-cache-size` / `--mamba-full-memory-ratio` | Sizes the state pool; overrides the ratio (default 0.9 "over-provisions the KV pool and clamps concurrency") | W1's 56 gave 12 real running with 881k KV. Elsewhere: 982 → 1,725 at c50 (1.76×); +75 % at c=12; 47.3 → 80.3 | MO + ME | 2.5 "consumes 15.2 GB, starving the KV pool" | Formula `ratio = (S + D) × state_bytes / (L × kv_bytes_per_token)` | Tested (explicit 56); ratio 0.5 untested by us | FR L269, L323, L339; EC L131, L389 |
| 31 | True 12/12 admission (PR #44401 or a newer SGLang) | Fixes the chunked-prefill double count | "about 8 %"; whole run projected 355-410 [E] | MO (cost) + E | Build after 0.5.21 is untested | — | Workaround 13/12 in use; the fix is untested | FR L20, L270, L430; OM L330-333 |
| 32 | `--mem-fraction-static 0.83` (vs auto 0.62) | Reclaims memory left idle by auto | Auto left about 19 % (4.39 GB/GPU) unused | MO | 1.84-1.88 GB free remains | — | Tested | FR L339; EC L274 |
| 33 | `--disable-prefill-cuda-graph` | Drops the 2.60 GB/GPU piecewise prefill graph and its 447 s first-launch JIT | One of the levers to reach 881k KV; decode cost "negligible" [E] | MO + E | Prefill speed on graph-less path | — | Tested | FR L349; EC L280, L300 |
| 34 | Decode / verify CUDA graph batch list (`--cuda-graph-max-bs-decode`) | Graphs captured for bs [1, …, 8, 10, 12] | "already on … remaining gain near zero" [E] | MO (config) + E | Target-verify capture is about 372-385 s of the cold start | — | Tested (on) | FR L349; EC L399; OM L324 |
| 35 | vLLM CUDA graph mode | FULL_AND_PIECEWISE default; spec + FlashInfer forces PIECEWISE | Elsewhere +2-18 % (dense hybrid), up to +91 % (small MoE) vs V0; PIECEWISE alone worse at low concurrency | ME | PIECEWISE forced in our vLLM; capture 7 s, 0.61 GiB | — | Tested (forced PIECEWISE) | FR L349; EC L278-279, L380 |
| 36 | Overlap scheduler / Spec V2 overlap (`SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`) | Overlaps CPU scheduling with GPU | 1.1× (2024, non-hybrid); Spec V2 +33 % at c=32 on B200 | ME | FlashInfer plan stream adds a sync point that limits overlap; Ada defaults to FlashInfer | extra_buffer or lazy | Overlap is on in W1; plan-stream flag untested | FR L288, L349; SD L147-148 |
| 37 | Chunked prefill size / max-prefill-tokens | Caps prefill per scheduler step | Cookbook 2048 ("8192 stalls decode about 600 ms"); small chunks protect decode at TTFT cost [E] | ME (claim) + E | — | — | 4096 / 16384 in use; no Ada measurement | FR L353; EC L283, L300 |
| 38 | vLLM `--max-num-batched-tokens` 16384 | Larger prefill batch | +8 % (GB200 prefill node) | ME | Our vLLM drafter ran out of memory at 32768, so we use 4096 | — | Untested at 16384 | FR L353; EC L284, L382 |
| 39 | vLLM `--async-scheduling` / `--no-async-scheduling` | Async CPU scheduling | "key driver" of 25K TPS, no number | C | Race conditions gave zero accuracy before fixes; public L4 notebooks set `--no-async-scheduling` | — | Untested | EC L337, L382; CE L41-42 |
| 40 | Engine SGLang 0.5.21 vs vLLM 0.19 | Different scheduling, spec verify, all-reduce and GEMM paths; same GDN Triton kernels on Ada [E] | Bundle W1 vs V: +17-30 % whole run; engine alone not isolated | MO (confounded) | — | — | Tested as a bundle only | FR L110, L292; EC L132 |
| 41 | vLLM ≥ 0.28 INT4 + DFlash2 arm | Isolates the engine effect | — | E | DFlash2 needs unmerged PR 52816; FP8 draft KV crashes on sm89 (#54690); #54360 prefix hits 0 on nightly | — | Untested | EC L132, L137; FR L292 |
| 42 | TensorRT-LLM | — | — | ME (docs) | No dense Qwen3.5/3.8 class; MTP = No, DFlash = No | — | Rejected | FR L294; EC L37-38 |
| 43 | llama.cpp | — | Single-L4 DFlash2 37.4 tok/s | ME | Tensor-split does not scale (76.4 → 78.7); no high-concurrency data | — | Rejected | FR L294; EC L41-43 |
| 44 | ExLlamaV3 / LMDeploy | — | No numbers | C | Batching not built for 12-24 × 60k [E] | — | Untested | EC L46-53 |
| 45 | TP4 (current) | Weights sharded 4 ways; 4 KV heads, so TP4 is the largest TP that shards KV without replication | Measured W1 numbers | MO | Custom AR disabled; NCCL on every layer | — | Tested | EC L211; FR L313 |
| 46 | 2× TP2 replicas behind a router (SGLang Model Gateway or vLLM DP) | Halves the all-reduce group; two independent schedulers | "+20-100 % possible" from 3090 analogs [E]; 3090 2.40× at max-num-seqs 256; biMEMO 518.6 vs 469.4 at c8; DGX Spark 36.0 vs 27.0 per request at C8 | ME + E | FP8 TP2 cannot hold 12 × 76k (0.3-0.6M total); INT4 TP2 about 1M total [E], about 0.7-0.8M with DFlash2 (tight); two startups (about 6 minutes of target-verify graph each); weights possibly read twice on GCSFUSE; custom AR only if the P2P test passes | Router; memory fit check | Untested ("the largest unmeasured lever") | FR L24, L298-315, L431; EC L142-221 |
| 47 | 4× TP1 / PP | — | gemma-4-12B 4× TP1 3.4× vs TP4 (3090). PP: no per-layer all-reduce, but bubbles and per-stage GDN state "unattractive" at 12-24 streams | ME / E | The docs give no memory-fit analysis for TP1 of the 27B on a 24 GB L4; DFlash + PP unsupported in vLLM (v0.30.0) | — | Untested (PP: no measurement found) | EC L155, L215; FR L313; SD L161 |
| 48 | Forced custom all-reduce at TP4 | Bypasses NCCL | +14.8 % decode at c=1, +14.3 % at c16 | ME | Greedy divergence 4/5 ("not eval-clean"); prefill registration overhead +9 % TTFT | — | Rejected for eval runs | FR L278; EC L157, L357 |
| 49 | Custom AR at TP2 with verified P2P | — | +15.7 % vs a no-P2P reference (3090) | ME | Requires a P2P test pass on Kaggle (see section 6a) | P2P | Untested | EC L158, L206 |
| 50 | NCCL P2P toggle / `NCCL_P2P_LEVEL=SYS` | Transport choice | +2-5 % (sometimes negative) on dual 5090; SHM not always slower | ME | — | — | Untested before 10-23 (see 6a) | EC L172; CE L400 |
| 51 | Higher concurrency than 12 (W1) / 16 (vLLM) | More streams per step | A no-spec reached 22 → 440 (short); W1 never >12; vLLM never >16; vLLM projection 25-30 → 380-430 [E] | MO (partial) + E | KV is about 881k; vLLM preempts at ≥99.5 % KV (D66: 66) | More KV | Untested | OM L209-210; FR L443 |
| 52 | Temperature / effort (vLLM) | MTP acceptance depends on sampling | 16 running: T 0.6 352 vs T 1.0 295-300 vs low effort 430; AL 3.01 vs 2.67 → 349 vs 298 | MO (confounded) | Changes answers; tasks differed | — | Observed, not A/B'd | FR L108, L203 |
| 53 | KV-resident admission / prefix retention (agent harness) | Keep each branch's prefix resident | Effective 187-277 vs about 410-430 decode in Duck; 8.39 M of 9.60 M uncached prefill from lost prefixes; rellik13 14.5 → 22.5 from more KV/history (ARC-AGI-3) | MO + ME | — | — | Partially addressed in E12 revs | FR L60; CE L229, L263, L273 |
| 54 | `--schedule-policy lpm`, `--page-size 64`, `--mamba-track-interval 64`, `--enable-cache-report` | Cache-aware scheduling and checkpointing (Franzen) | Not stated | C | — | Flags exist in 0.5.21 per our VALIDATION row 10 | Untested on L4 | FR L353; CE L265 |
| 55 | Weight-load path: parallel copy to local disk or `/dev/shm`; raise the first-launch timeout to the 75-minute give-up; `--weight-loader-prefetch-checkpoints --weight-loader-drop-cache-after-load`; reshard; page-cache preload | Avoid slow FUSE reads and the fixed 30-minute timeout | Avoids a 0 LB when the server never starts; 16 threads ≈ 3.6 minutes vs 1 thread ≈ 37 minutes for 23.3 GB [E] | MO (1,197 s G4B) + ME + E | Disk size and the `/dev/shm` cap unknown | — | Not built (two options in 00_SUMMARY; user decision) | FR L363-376, L426 |
| 56 | Persisted Triton/FlashInfer JIT cache across sessions | Skip the ~372-385 s target-verify graph JIT | About 6 minutes per run [E] | E | #42917: a stale cache can wreck accuracy | — | Untested across sessions | FR L378; OM L426, L431 |
| 57 | Fresh kernel caches (`SGLANG_CACHE_DIR` / `CUDA_CACHE_PATH`) | Avoids #42917 | Correctness guard | ME | — | — | In use | FR L268, L282; EC L354 |
| 58 | `--speculative-draft-model-quantization unquant` | Keeps the drafter BF16 | Prevents accept about 1.0 (#39087) | ME + MO | — | — | In use | OM L274 |
| 59 | Avoid `--enable-mixed-chunk` and grammar with DFLASH | Avoids #43514's trigger | Correctness guard | ME | — | — | We do not use mixed-chunk; grammar use to be confirmed | FR L266, L282 |
| 60 | `--language-model-only` (vLLM) | Skips the vision encoder | 1× 5090 KV 91,022 → 135,926 | ME | — | — | In use in V | FR L357; CE L328 |
| 61 | vLLM mem 0.95, MTP k=2, fp16 SSM | More KV, fewer draft slots | Listed only as levers in [E10C] | E | — | — | Never measured | OM L381 |
| 62 | Flash-Next on L4 | 6B active MoE | E13 target 700 tok/s "not a gate"; bandwidth-scaled 80-190 at 6 streams [E] | MO (failed) + ME + E | Garbage output and 5.95 tok/s in E3; 121-185 GB load on GCSFUSE; 118-250k KV | EP4, FA2 for QSA on sm89 | Rejected ("research bet") | FR L382-395 |
| 63 | vLLM `VLLM_USE_V2_MODEL_RUNNER=0` | Workaround for #39573 (thinking budget ignored with MTP) | Correctness | MO | — | — | Needed in any vLLM 0.19 arm | FR L275 |
| 64 | Sampling `top_k` 20 vs SGLang default −1 | Card says 20 | Not stated (affects acceptance and answers) | C | — | — | Not examined | FR L355; CE L441 |
| 65 | FR-Spec 64k hot-token map (Franzen) | Smaller draft vocabulary | Not stated | C | — | — | Untested | CE L201 |
| 66 | HiCache (club-3090 #1288) | Hierarchical KV cache | Not isolated | C | — | — | Untested | CE L400 |
| 67 | vLLM `--max-cudagraph-capture-size` | Avoids the `num_cache_lines >= batch` assert | Stability | ME | — | — | — | EC L279; SD L156 |

### 3.1 Speculative decoding detail: MTP vs DFlash/DFlash2 vs EAGLE-3 vs n-gram/suffix at high batch

**The two document camps**
- **SD Q7 (L249-256)** recommends built-in MTP via SGLang EAGLE/NEXTN (topk 1, 2-3 steps, 3-4 draft tokens) as "the best expected aggregate gain with acceptable risk". Against DFlash2 it lists:
  - the open concurrency-correctness bug #36548;
  - documented acceptance and draft-cost decay at long outputs and contexts (arXiv 2605.09329; vLLM #54691 on 4× 4090 TP4);
  - "about 2x the mamba-slot memory of MTP-4";
  - Blackwell-only recommended attention backends.
- **CE §3 (L446-452)** concludes the opposite for 4×L4: INT4/W4A16 + DFlash2, "provided the mamba-slot and KV budget is checked". Its evidence is lcz.me's sm89 A/B (about 3× faster decode on structured output) and INT4's half bytes.
- **FR's resolution (L244-248):** "Our measurements favor the second."
  - L4 production Poetiq DFlash2 AL is 3.8-4.5, 4.43 → 3.77 across context with no collapse. That is far above vLLM MTP-3's 2.67-3.0 and SGLang NEXTN's about 3.1.
  - At 8 running: DFlash2 356 vs MTP 297.
  - FR's verdict: "Throughput: DFlash2 wins on our evidence. Accuracy risk: MTP is lower." So NEXTN is rank 2.

**Why L4 may differ from H200, per the docs' estimates (FR L246; SD L99-103; CE L451):**
- At 16 streams × about 35k context, one step per GPU is about 35 ms: KV about 15 ms, INT4 weights about 15 ms, GDN about 4 ms. That is an upper bound of about 450 tok/s without speculation.
- Verifying 16 × 4 = 64 tokens needs about 0.86 TFLOP per GPU, about 7-14 ms, which is below the memory time. This is the MagicDec regime.
- L4 has about 121 dense TFLOPS against 300 GB/s; H200 has about 989 against 4.8 TB/s. So the c32 point where gains vanish on H200 should sit higher on L4.
- Overheads that could eat the gain (SD L103):
  - intermediate SSM-state writes per draft position: about 16 × 4 × 37.7 MB ≈ 2.4 GB per step in fp32, about 8 ms; half that in fp16;
  - PCIe all-reduce latency for each sequential draft step;
  - draft-model KV reads at long context.

**MTP-4 vs DFlash2-8 at about 12 streams: what each doc says**

| Aspect | MTP (NEXTN 3/1/4) | DFlash2 block 8 | Source |
|---|---|---|---|
| Our AL | 3.09-3.17 at 1-32 requests (really ≤10 running); long run 3.02 | 4.51-4.85 short (W1/D at 12: 4.51 / 4.62); production 3.8-4.5 (E11-06 4.34 at 12) | OM L251-265 |
| Our throughput | 297 at 8; server decode 340.9 at 8, 373.7 at 10; never measured at 12 (cap 10) | 356 at 8; 404 at 12 (client); server 482.9 at 8, 477.2 at 12; production 416.6 median at 12 | OM L140-151; FR L66-74 |
| Same engine at 12 real running | **Never measured** | — | OM L286; FR L246, L445 |
| Accept-length definition | SGLang's "accept len" may or may not include the bonus token (unchecked); vLLM's does (1 + accepted/drafts) | same | OM L288; FR L246 |
| Thinking-mode AL elsewhere | about 2.5 of 4 (Flash-Next, DGX Spark) | vendor numbers only at 4k outputs | SD L31, L57 |
| Long output elsewhere | MTP not tested in arXiv 2605.09329; attention drift affects MTP heads too | collapses to about 1.5 after 10-25k (DFlash/EAGLE-3/PARD on Qwen3-8B); draft cost grows with context (vLLM #54691) | SD L36-39, L94 |
| Ours over context | — | 4.43 → 3.77, "no collapse"; 4 × 85.5k accept 4.58 | OM L261, L266 |
| Vendor H200 c32 | 0.77-1.04× | 1.01-1.45× | SD L70-78 |
| Vendor H200 c8 | 1.74-2.19× | 2.27-2.85× | SD L75 |
| Mamba slots / verify state | D = 4 (EAGLE/MTP); 16 streams MTP (5 slots) fp32 about 3.0 GB/GPU, fp16 1.5 GB; B intermediate 1.55 GB (fp32, extra_buffer, auto mem) | D = 8; 1 + 8 = 9 slots: fp32 about 5.4 GB, fp16 2.7 GB at 16 streams; intermediate 1.83-1.97 GB; drafter 1.49 GB + draft KV 0.92-2.10 GB per GPU | FR L323, L343; SD L194-203; OM L309-316 |
| Running cap at auto mem 0.62 | 10 (fp32) | 6 (fp32) / 13 (fp16) | OM L309-311 |
| GEMM width M at 12 streams | 12 × 4 = 48 (WQ uses streams × (draft + 1): 16 × 4 = 64 for MTP-3) | 12 × 8 = 96, just under Marlin's ridge (M ≈ 100) | FR L136; WQ L186-190 |
| Correctness | #36548 not reproduced with MTP/EAGLE on 0.5.18 [U]; `--speculative-adaptive` crash (#23330) | #36548, #38009, #43514 open; #39087 mitigated | FR L264-267; SD L127, L150 |
| Greedy drift vs no-spec (ours) | 152 tokens of 512 | 155 tokens of 512 | OM L513 |
| Backends on sm89 | FlashInfer needs > 0.6.15.post1 for MTP, otherwise triton | Recommended `fa4` / `trtllm_mha` are Blackwell-only; FlashInfer on sm89 works in our runs; no upstream doc confirms DFLASH draft attention on sm89 | EC L292; SD L149, L173 |
| Settings evidence | steps 2-3, topk 1, 3-4 draft tokens; SGLang B200 c8 steps3 736.2 ≈ steps5 740.5 | must run its trained block 8; positions 4-7 about 0.1 or less; 3 tokens ≈ MTP | SD L113-131; CE L363-364 |
| Single-stream elsewhere | 3090 llama.cpp: MTP n-max 2 +59.8 % beats DFlash2 n-max 4 +51.9 % | 4090 SGLang W4A16 + DFlash2 245.6 vs FP8 + MTP 75.7 (code); 3090 vLLM 136.5 vs 101.9; L4 llama.cpp 37.4 vs MTP 28.05 | FR L236 |

**EAGLE-3 at high batch:**
- Trails native MTP on Qwen3.x: AL 2.2 (Qwen3.6-27B); 1.251 vs 3.715 (Qwen3.6-35B-A3B).
- Wide trees go negative (1.9× → 0.6× at c32); EAGLE-3 (10) 1.6× → 1.0×.
- Long outputs: 1.1 after about 20K tokens.
- The EAGLE-3 paper reports 1.38× at batch 64.
- FR rejects it (FR L25, L252; SD L41-45, L81).

**n-gram / suffix at high batch:**
- n-gram slows diverse text (0.29-0.88×).
- Suffix has about 10 % overhead at c64 and 1.0-1.28× on SpecBench.
- No hybrid or reasoning data; vLLM/Arctic only; MTP + suffix undocumented.
- "Only interesting as a complement for repetitive grid output", with no evidence for that (SD L50-53, L163, L256; FR L252).

**Topk rule:** use topk = 1 chains, never trees, at high batch (SD L113; FR L254).

### 3.2 Weight formats on Ada (BF16 / FP8 / INT4 W4A16 Marlin / AWQ / GPTQ)

- **Kernel paths on sm89** (WQ Q1 L23-89; FR L118-122):
  - W4A16 (GPTQ / AWQ / compressed-tensors) → Marlin in both engines (`awq_marlin`, `gptq_marlin`). Machete is Hopper-only.
  - Per-channel FP8 W8A8 → `CutlassFP8ScaledMMLinearKernel` (CUDA ≥ 12.4).
  - Block-128 FP8 (official Qwen) → Triton `w8a8_block_fp8_matmul`, because CUTLASS block-scaled needs SM90+ and DeepGEMM SM90/SM100.
  - Our L4 log prints "Selected CutlassFP8ScaledMMLinearKernel", then "Using default W8A8 Block FP8 kernel config. Performance might be sub-optimal!" for 5 shapes (N=4096,K=5120; 5120×1536; 8704×5120; 5120×4352; 3584×5120).
  - SGLang `--fp8-gemm-backend auto` order is DeepGEMM, FlashInfer TRTLLM, CUTLASS (SM120), AITER, then Triton, which is where sm89 lands. `w8a8_fp8` / `w8a8_int8` use sgl-kernel CUTLASS for per-channel + per-token dynamic models.
  - NVFP4 / MXFP4 → FP4-Marlin weight-only (`has_device_capability(80)`; SGLang `--fp4-gemm-backend marlin` on SM80-SM90).
  - INT8 W8A8 → CUTLASS INT8.
  - GDN layers are not GEMM-quantization-bound; vLLM on L4 logs "Using Triton/FLA GDN prefill kernel".
- **L4 hardware:** 24 GB, 300 GB/s, 72 W, PCIe Gen4 x16 64 GB/s, no NVLink. Sparse rates: FP16/BF16 242, FP8 485, INT8 485 TOPS, TF32 120; dense rates are half, about 121 and about 242 (WQ L103-107; EC L5).
- **Roofline per decode step at TP4 [E]** (WQ L171-190; FR L136):
  - Weight bytes per GPU: BF16 about 13.7 GB (46 ms); FP8 7.15 GiB = 7.7 GB (26 ms); INT4 about 4.9 GB (16 ms).
  - FP8 KV at 16 × 30k: 3.9 GB (13 ms). fp32 GDN state read+write: about 1.2 GB (4 ms).
  - Ceilings at B=16: BF16 about 254, FP8 about 376, INT4 about 482 tok/s (1.28×). At B=32: 400 / 537 / 637 (1.19×). At B=8: 147 / 235 / 324 (1.38×).
  - Measured numbers run at about 50 % of these ceilings.
- **Compute ridge [E]:**
  - Marlin on FP16 tensor cores: ridge about 400 FLOP/byte, so it is memory-bound up to M ≈ 100. FP8 W8A8 stays memory-bound up to M ≈ 400.
  - With speculation, M = streams × (draft + 1).
  - "Above 12-16 streams with DFlash2, INT4 Marlin approaches its compute ceiling and FP8's 2× tensor rate begins to matter. Our current 12 × 8 = 96 is just below this limit" (FR L136).
- **Accuracy order [E]:** BF16 ≈ FP8 ≥ INT4 GPTQ+AWQ ≈ NVFP4 > MXFP4. The INT4 gap is about 1-1.5 points: "inside one-seed noise, but consistently negative"; it could exceed 1.3 on hard ARC puzzles at 60k (FR L163; WQ L284-285).
- **AWQ vs GPTQ:** evidence leans to GPTQ-based schemes (less inflation, Arena-Hard +2.9). RedHat's checkpoint is GPTQ + AWQ smoothing, "on the right path" (FR L161).
- **Memory budget at TP4 [E]** (WQ L350-360; FR L167-173). Assumes vLLM's measured non-KV overhead of about 2.1 GiB, with no spec slots counted:

  | Weights | Per-GPU weights | KV + state budget | FP8-KV tokens | BF16-KV tokens | Worst-case 60k streams (FP8 KV / BF16 KV) |
  |---|---|---|---|---|---|
  | BF16 | about 12.8 GiB | about 5.3 GiB | about 0.69M | about 0.35M | about 10-11 / 5 |
  | FP8 | 7.15 GiB (measured) | 10.98 GiB (measured) | about 1.35-1.44M | about 0.7M | about 21-22 / 11 |
  | INT4 | about 4.5 GiB | about 13.6 GiB | about 1.78M | about 0.89M | about 26-28 / 14 |

- **Verdict** (FR L177; WQ L467-488):
  - INT4 W4A16 + fp8 KV is best on throughput and capacity.
  - FP8 + fp8 KV is second: near-lossless, about 0.75-0.85× the INT4 ceiling, "relatively better under heavy speculation".
  - BF16, INT8 and NVFP4/MXFP4 are rejected.
  - The modules kept BF16 in every good recipe (GDN `in_proj_a/b`, lm_head, embed, MTP head, vision) are exactly the ones RedHat INT4 leaves unquantized.

### 3.3 KV-cache dtype

- **Sizes:** KV exists only in the 16 full-attention layers (4 KV heads, head dim 256): fp8 32 KiB per token, bf16 64 KiB; per GPU at TP4, 8 / 16 KiB. A 60k stream needs 0.46 GiB per GPU (fp8) or 0.92 GiB (bf16). Sixteen 70k streams need about 17 GiB/GPU in bf16 (does not fit) or about 8.5 GiB in fp8 (FR L183; WQ L341-343).
- **Capacity:** W1's 881k fp8 KV would be about 440k in bf16 [E], "not even half" of the 920k that 12 × 60k needs, so "bf16 KV has no practical place on L4" (FR L183).
- **Throughput:** the roofline says fp8 KV saves about 13 ms of a 55 ms step at 16 × 30k, about +25-30 % [E]. The vLLM blog measured +14.9 % on H100 c8 at about 20k / 2k, with negligible gain below about 7k (FR L183; WQ L411).
- **Accuracy:** see 2.11. The docs say "validation on L4 kernels is still needed" (WQ L377). On Ada, FP8 KV runs through FlashInfer or Triton attention; FA2 cannot do FP8 KV.
- **vLLM backend conflict on sm89:** see section 4, item 13.

### 3.4 Engines (SGLang 0.5.x vs vLLM 0.19+ vs TensorRT-LLM vs llama.cpp)

- **Shared kernels:** only SGLang and vLLM are realistic. Both run GDN on Triton (FLA-derived) kernels on Ada, because FlashInfer GDN needs SM90+ (`gated_delta_rule_decode` needs SM90; prefill SM90/SM100) (EC L17-25; FR L288).
- **SGLang on Ada:**
  - The Qwen3.8-27B cookbook validates only H200, RTX PRO 6000, RTX 5090, DGX Spark and GB300 (v0.5.19); no Ada.
  - The Ada attention default is flashinfer, else triton; both support FP8 KV and topk = 1 or >1.
  - On our 4×L4 every config launched first try with 0 failed requests, linear decode triton.
  - Offline wheel: torch 2.13.0, flashinfer_python 0.6.18 cu13, transformers 5.12.1, about 220 wheels, about 5 GB (EC L21-27).
- **vLLM:**
  - ≥0.17.0 for Qwen3.5-27B; transformers ≥5.8.0 for Qwen3.8-27B; DFlash2 needs ≥0.28.0 and still the unmerged PR 52816.
  - FLA Triton GDN kernels; full CUDA graph default; the Blackwell FlashInfer GDN prefill does not transfer (EC L30-34; FR L290).
- **Differences on L4 [E]:** "mostly in scheduling, spec-decode verify, all-reduce and FP8/INT4 GEMMs, not in GDN" (EC L54; FR L292). The measured +17-30 % is a bundle (INT4 + DFlash2 + fp16 SSM).
- **Rejected:** TensorRT-LLM (no model class, no MTP). llama.cpp (no multi-GPU scaling). ExLlamaV3 and LMDeploy (no Ada high-concurrency numbers; batching not built for 12-24 × 60k) (EC L53-56; FR L294).
- **Engine bugs:** see 2.19.

### 3.5 Parallelism (TP4 vs 2×TP2 vs others)

- **Community measurements:** TP4 wins single-stream decode slightly and prefill/TTFT clearly (+55-62 %), but 2×TP2 wins aggregate at saturation, by up to 2.4× for a 27B INT4 (EC L145; FR L300).
- **Communication:**
  - Custom AR is disabled at TP4 on >2 PCIe GPUs; allowed at TP2 only if the P2P test passes.
  - SymmMem is not supported on cc 8.9.
  - Our preflight measured 64 MB in 20.9 ms (section 1.9).
  - FR [E] (L311): "2×TP2 will beat TP4 when TP4's per-step all-reduce overhead exceeds the time to read a quarter of the model's weights: about 15 ms for INT4 (4.6 GB / 300 GB/s), about 24 ms for FP8." The 3090 data leans to DP but is "indicative only" (EC L207-209).
- **Memory [E]** (EC L210-214; FR L313):
  - KV per GPU per token: 8.2 KB at TP4, 16.4 KB at TP2 (fp8).
  - **FP8 TP2:** about 14.3 GiB weights per GPU leaves at most about 6 GiB; realistically about 150-300k per replica, 0.3-0.6M total. That "cannot hold 12 × 76k".
  - **INT4 TP2:** about 9.3 GB + 0.4 GB MTP per GPU leaves about 10 GiB; 500k+ per replica after a 6-request fp16 GDN budget, about 1M total, "plausibly fitting 12 x 76k".
  - With DFlash2 (3.85 GB BF16 → about 1.9 GB/GPU at TP2, plus draft KV), capacity falls toward about 0.7-0.8M: "tight".
- **Kaggle costs [E]** (FR L315): two servers mean two startups (each with a target-verify graph of about 6 minutes), and weights may be read twice from GCSFUSE unless they come from one local copy. The 2×TP2 arm was proposed but never run [E113A L89].
- **DP machinery:** vLLM `--data-parallel-size 2 --tensor-parallel-size 2`; SGLang via the Model Gateway (`sglang_router`), with native `--dp-size` "highly not recommended".
- **PP:** no per-layer all-reduce, but bubbles and per-stage GDN state; no measurement; vLLM docs mention PP as possibly preferable without NVLink [S].
- **HyperQwen:** TP=2 and TP=4 work; TP=3 invalid.

### 3.6 Mamba/SSM state dtype and radix cache strategy

- **Slot accounting** (FR L323; EC L232-237):
  - Per request: S state slots (radix off 1, no_buffer 3, extra_buffer + overlap 5, extra_buffer_lazy 4) plus D verify slots (EAGLE/MTP 4, DFLASH2 8, DSPARK block+1, ReplaySSM 0).
  - `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1` frees one slot for extra_buffer strategies (lazy + this = 3); `extra_buffer` frees one more with overlap off.
  - Formula: `ratio = (S + D) × state_bytes / (L × kv_bytes_per_token)`; `--max-mamba-cache-size = target_concurrency × S` overrides it.
  - FR L343: "the source sizes the intermediate verify cache separately". The logs show separate "intermediate" GB for B/C/D/W1-W3.
- **Strategies:**
  - extra_buffer (V2) enables overlap scheduling and branching-point caching; it needs `--page-size 64` (FLA_CHUNK_SIZE 64) and FLA kernels (NVIDIA-only).
  - no_buffer (V1) has no overlap and lower memory.
  - Changelog: spec decoding was added for `extra_buffer_lazy`, and spec-v2 host overhead was cut in hybrid-linear MTP decode (SD L143-145).
  - `extra_buffer_lazy` has "no accuracy cost stated" (EC L232). The Qwen3.8-27B cookbook says lazy cuts slots per request from 5 to 4 (CE L313).
- **Rollback design:** SSM states update in place, so each draft token gets its own cache slot and the last accepted slot is promoted. The elastic memory pool resizes via CUDA VMM; MambaRadixCache copies state on match and forks a checkpoint on insert, with separate LRU lists (SD L146; EC L245-248).
- **State size [E]** (SD L194-197; WQ L344-347):
  - GDN state = 48 layers × 48 heads × 128 × 128 ≈ 151 MB per request in fp32 (37.7 MB per GPU at TP4; 18.9 MB in fp16/bf16). WQ gives about 36 / 18 MiB per GPU.
  - Cross-check: ssm pool 1.21 GB / 68 slots ≈ 17.8 MB per slot in fp16.
  - Conv state is about 0.7 MB per GPU per stream.
  - SGLang's 5 slots per request under speculation come to about 180 MiB fp32 / 90 MiB fp16 per stream per GPU; at 32 streams, about 5.6 / 2.8 GiB.
- **FR [E] (L343):** lazy (S=4) + D=8 at 12 streams needs up to 144 slot-equivalents, which is impossible on 4×24 GB in fp32. That is why fp16 state, the lazy strategy and ReplaySSM are the levers.
- **Our evidence:** fp16 raised the cap 6 → 13 at the same memory; W1 fp16 416.5 (8/8) vs W2 bf16 412.1 (7/8); W3 fp32 KV 598,712. No public 60k accuracy study of fp16 vs fp32 GDN state exists (FR L343; EC L303).
- **WQ guidance (L288):** keep the SSM state in fp32, or fp16 after measurement. Internal evidence says fp16 is safe and bf16 slightly worse.

### 3.7 CUDA graphs

- **SGLang W1** (FR L349; EC L280; OM L425):
  - Target-verify and draft-verify graphs at bs [1, 2, 3, 4, 5, 6, 7, 8, 10, 12].
  - The prefill (piecewise) graph took 2.60 GB per GPU and a 447 s first-launch JIT, so W1 disables it.
  - The target-verify graph capture is about 372-385 s of a cold start, about 12 s warm in the same session.
- **vLLM:** FlashInfer + spec forces PIECEWISE (capture 7 s, 0.61 GiB). The early eager → graph/noAR change gave 48.30 → 129.17 (2.67×) (FR L349).
- **FR/EC [E]:** "Graphs and the overlap scheduler are already on in the measured configs, so their remaining gain is about zero." Switching to `no_buffer`, which turns off overlap, would cost about 10 % (EC L399; FR L349).
- **Franzen:** `--cuda-graph-max-bs-decode 10 --cuda-graph-bs-decode 1 2 4 7 8 9 10` (FR L349).
- **Gap:** CUDA graph, overlap and chunk-size gains have not been measured separately on Ada (EC L403; FR L465).

### 3.8 Chunked prefill size

- **SGLang cookbook:** 2048, because "8192-token chunks stall decode about 600 ms at a time"; H200 32768; RTX 5090 draft rows 1024/512.
- **Values in use:** ours 4096 (max_prefill_tokens 16384); NguyenThanhNhan 2048 / `--max-prefill-tokens 4096`; Franzen 8192 / 16384.
- **vLLM:** 4096 in ours, because 32768 ran the MTP drafter out of memory; 16384 gave +8 % on GB200 prefill nodes.
- **FR/EC [E]:** for this workload (prompt about 8k, output about 60k), prefill is a small share of tokens. Small chunks (1-2k) protect decode at the cost of TTFT. Dropping the prefill graph costs negligible decode.
- **Measured prefill shares:** heavy-prefill windows in R5 ran at 138 tok/s (vs 295-300 steady); Franzen's prefill time fraction was 15.25 %.
- Sources: FR L353; EC L283-284, L300; OM L168; CE L207.

### 3.9 Concurrency vs acceptance trade-off

- **Our acceptance is flat with concurrency.**
  - vLLM MTP-3: 2.75-3.16 from 1 to 16 running.
  - SGLang NEXTN: 3.09-3.17 from 1 to 32 requests.
  - DFlash2: about 4.3-4.95 from 1 to 16; in production 4.34 at 12 and 4.82 at 9.
  - The binding trade-off is memory slots and KV, not acceptance (OM L247, L252, L256-265; OM L282).
- **Per-stream rate falls as concurrency rises.**
  - W1: 46.1 at 8, 43.2 at 9, 34.7 at 12.
  - vLLM: 44.3 at 1 down to 22.0 at 16.
  - D per-request: 107.5 → 34.8 from 1 to 12.
- **Spec state caps concurrency.**
  - Ours: A 22, B 10, C 6, D 13 at auto memory.
  - Elsewhere: Flash-Next 3 with MTP vs 12 without at default, 16 vs 64 tuned; DGX Spark 8 vs 24; vLLM #55533 3 of 8 sequences.
  - SD L274 [E]: "If spec forces fewer concurrent streams, the net gain can vanish", as with Flash-Next (785 tok/s at 16 with MTP vs 1,258 at 64 without).
- **Draft length vs batch:**
  - SPEED-Bench: longer drafts at low batch, DL=1 at high batch.
  - SD Q3 inference: "Fewer draft tokens also mean fewer mamba slots per request, which matters on 24 GB."
- **On L4 [E]:** decode stays memory-bound to a higher batch (MagicDec), so speculation should still pay at 16 streams. The measurement at 12 (+39 % DFlash2) supports this; there is none above 13 (SD L67, L102; FR L222).
- **Concurrency limit by KV:** at 60k both engines are KV-limited to roughly 12-16 streams (OM L373). vLLM at 16 × 77.8k preempted 66 times at 99.88 % KV.

---

## 4. Contradictions or corrections between documents

1. **How much faster is SGLang W1 than vLLM?**
   - OM L88: "SGLang W1 is roughly +17-30 % on whole runs, not 1.5x."
   - EC L69: "+17 % (production) to +39 % (synthetic)"; EC L132: "about +17-29 % on real runs".
   - WQ L191: "+30-60 % for INT4 + DFlash2 at 12 streams"; WQ L483: "+30-60 % measured".
   - FR L110 (translated): "the last figure compares W1's decode against vLLM's long-context production average (255-300), so it is more optimistic. On equal definitions (whole run vs whole run) **+17-30 % is best supported.**"
2. **Cause of the 11-of-12 admission bug.**
   - EC L244 (citing E11-05): "The likely cause is the PrefillAdder mamba-slot gate (unverified), costing about 8 %; candidate fix `--max-mamba-cache-size 64`."
   - OM L331: "Root cause, verified in the sgl 0.5.21 source: a chunked-prefill request is counted twice in admission (`scheduler.py:3971-3976`), so `batch_is_full` latches. Upstream fix PR #44401."
   - FR L270 (translated): "the E11-05 analysis first gave the PrefillAdder mamba-slot gate … later source verification found the double count. **The second is more reliable.**"
3. **Recommended spec method.**
   - SD L250: "The evidence favors built-in MTP via SGLang EAGLE/NEXTN … as the best expected aggregate gain with acceptable risk."
   - CE L446: "On 4×L4, the weights choice that the evidence supports is INT4/W4A16 + DFlash2."
   - FR L246 (translated): "**Our measurements favor the second**", with L248: "DFlash2 wins on throughput in our evidence, but MTP has lower risk on accuracy", so NEXTN is rank 2.
4. **"Speculation only helps below about 8 concurrent."**
   - CE L434 (HyperQwen): "Speculation wins below ~8 concurrent users, plain batching above". CE L288-290: gains "shrink, and can turn negative, at high concurrency on this hybrid GDN model".
   - OM L280 / FR L222: DFlash2 is +39 % at 12, so the claim "does not hold for this hybrid model on L4".
5. **Long-output collapse of external drafters.**
   - SD L8: "DFlash, EAGLE-3 and PARD collapse toward an AL of 1-2 after roughly 10-25k output tokens"; SD L56: externally trained drafters "may lose most of [their] edge after about 10-25k tokens".
   - OM L266 / FR L219, L475: "DFlash acceptance only falls 4.43 → 3.77 over context, so no collapse"; FR: L4 behaviour "falsified both" the concurrency-decay and long-output-collapse arguments.
   - Caveat (FR L246, L448): no source, ours included, has measured AL per 10k window across 60k thinking.
6. **SD's "no L4 measurements" vs OM's L4 data.**
   - SD L67: "there are no L4 measurements"; SD L107: "No measured spec-decode concurrency curve on L4"; SD L278: "No measurement of any method on L4".
   - OM and FR have the E11-03/E11-04/E11-05/E11-06 L4 measurements. SD was written from external sources only. Its gap about 60k thinking at 6-16 streams for MTP vs DFlash2 on the same engine still stands.
7. **What "5 state slots per request" means.**
   - OM L298 and WQ L332: "speculation needs 5 state slots per request, which capped running requests at 6-13".
   - The E11-03 A log (no speculation) also says "capped to 22 … 5 state slots per request" (OM L308). EC/FR give 5 as the extra_buffer + overlap S value, independent of speculation, with D verify slots added for spec (EC L232-235; FR L323).
   - SD L179 / L186: "SGLang reserves 1 + draft tokens SSM slots per request"; "DFLASH needs 1 + speculative_num_draft_tokens slots".
   - CE L415: #36889 "the DFLASH mamba pool silently caps concurrency (5 state slots per request)".
   - Flash-Next cookbook via SD L185: with MTP, "extra_buffer 5, extra_buffer_lazy 4".
   - FR L343 notes the intermediate verify cache is sized separately in the source, which the logs confirm (separate "intermediate" GB). The descriptions are inconsistent; the S + separate-intermediate reading matches our logs.
8. **Default radix strategy and flag name.**
   - The Qwen3.5 cookbook calls `no_buffer` the default; the Qwen3.8-27B page calls `extra_buffer` the default (EC L237; FR L323).
   - The 0.5.21 source also has `auto`, which "now resolves to extra_buffer on CUDA with radix cache enabled" (PR #27220).
   - The flag appears as both `--mamba-scheduler-strategy` and `--mamba-radix-cache-strategy` (#27418).
9. **DFlash drafter training context.** The DFlash paper's Qwen3.5-27B base drafter was "trained on 4K context". The current z-lab Qwen3.5-27B-DFlash card says "trained at **40k** sequence length" (SD L37-38; FR L238). The Qwen3.8-27B DFlash2 training context is unpublished.
10. **FP8 kernels on Ada.**
    - bottlecapai card: "Ada and Ampere GPUs serve the model through weight-only FP8-Marlin kernels".
    - The vLLM matrix and our L4 log show native FP8 (CUTLASS selected, then the Triton block kernel). FR L120 (translated): "**our log is more reliable.** Marlin-FP8 is really the Ampere path" (WQ L59).
    - A further inconsistency: CE L424 summarizes vLLM PR #59232 as "on sm89, block-FP8 falls back to Marlin W8A16", and CE L449 says "On sm89, block-FP8 uses default or Marlin kernels". WQ L53-57 and OM L589-590 say block FP8 runs the Triton `w8a8_block_fp8_matmul` with default configs, and FP8 Marlin is disabled on sm89 unless `VLLM_TEST_FORCE_FP8_MARLIN=1`. FR uses PR #59232 only for tuning gains.
11. **NVFP4 on sm89.** A Dec 2025 Reddit snapshot: "looks like sm100 is required … no fallback". The current vLLM/SGLang docs list FP4-Marlin on Ada (WQ L439; FR L122). FR: the current docs are more reliable, but no L4 run has verified that compressed-tensors NVFP4 W4A4 takes the Marlin path rather than erroring.
12. **FP8 KV accuracy at long context.** The vLLM blog: Qwen3.5-27B ≤0.7 points, MRCR to 1M "fully matches baseline". HF discussion #182: MRCR 0.717 (FP8 KV) vs 0.833 (BF16 KV) at 256k, n=15. FR L185: the blog's evidence is larger and systematic, and our 60-70k contexts are far below 256k, but the blog tested Hopper FA3 / Blackwell FlashInfer, not Ada.
13. **vLLM FP8 KV backend on sm89.**
    - delongyang: "FlashInfer's FP8 attention kernels exist only on SM90+ … fp8 KV crashes with illegal memory access unless TRITON_ATTN". khoinguyennguyen agrees, and CE L266 says FP8 KV works "on vLLM sm89 only with the Triton attention backend".
    - Our R5 production flags used FLASHINFER (`disable_flashinfer_prefill=True`) with fp8 KV, with no crash recorded. E12-V also selected FLASHINFER. Our earlier `harness_e9` asserted `TRITON_ATTN` with fp8.
    - FR L189 (translated): on our stack (vLLM 0.19 + our flags) FlashInfer + fp8 KV worked on long runs, a measured fact; crash reports on other versions may also be true, so re-verify the backend whenever the vLLM version changes.
14. **Token inflation from 4-bit.** arXiv 2504.04823: W4 models "do not generally think longer". arXiv 2606.25519: +3.7 % (GPTQ) to +14.4 % (AWQ) on Qwen3-30B. WQ L260 / FR L161: methods and models differ, so the question is open.
15. **GLM DFlash2 long-context decay.** Original report: 70-73 % → 16-32 % at about 172k. Follow-up ladder: no decay (0.31-0.36 from 4K to 170K) but slower steps (SD L96; FR L238).
16. **Batch-1 MTP vs DFlash2.** HackMD (3090, llama.cpp): the best MTP (+59.8 %) beats the best DFlash2 (+51.9 %), which "contradicts the vendor's batch-1 ordering of 3.43x (DFlash2) vs 2.59x (MTP) on H200 SGLang" (SD L17). lcz.me (4090 sm89), club-3090 #1746 and hanxiao (L4) favour DFlash2 single-stream (FR L236).
17. **W4A16 vs W8A8 at high concurrency.** "Give Me BF16": in async continuous batching, W8A8 gives the highest throughput. DGX Spark: NVFP4 (Marlin) is +29-34 % over FP8 at 16 concurrent. FR L134 (translated): "the conflict is real … for L4 the DGX Spark result and our measurements are more relevant."
18. **E11-05 DFlash2 accept length.** EC L121 cites "DFlash2 accept length median 4.55". OM L262-264 / FR L217: the server decode-line median is 3.8 (3.82 at 11), and 4.554 is the per-request median, which includes short phase-2 answers accepting 5.6+.
19. **E11-03 long-context C.** Per-request "16.3 tok/s" vs server decode median about 238 tok/s at 4 running. FR L74: "The two do not match", guessed (unverified) to include the 85k prefill.
20. **Client totals vs server decode lines.** D at 12: 404 (client total) vs 477.2 (server decode-line median); the definitions differ (FR L74).
21. **Rounded or doc vs derived values.**
    - R5 281.5 [D] vs 281 (doc) vs 280.9 (E11-05 doc, EC L118).
    - D66 256.0 vs 255.8; L66 254.3 vs 254.1; R4 225.2 vs 224; W3H 404.0 vs 400.
    - E12 SMOKE1 accept 5.1 [D] vs doc 5.2; G4B 4.34 [D] vs doc 4.2.
    - E12-V 2-running decode 100.2 [D] vs memory note 91 (OM L168-172, L182, L273).
22. **W1 free memory.** OM gives 1.84 GB (2.22 after the draft graph); EC L200 gives "2.2 GB free". Same run, different capture points.
23. **W1 KV pool.** 881,336 (server cap 12) vs 866,744 (server cap 13). EC L117 gives E11-06 as "max 12"; OM gives server 13 / client 12.
24. **INT4 per-GPU weights.** WQ roofline: "about 4.9 GB (19.45 GB / 4)". WQ memory table: "about 4.5 GiB" [E]. Measured E11-03 D: "target weights 4.62 GB".
25. **vLLM KV figure.** The log's "GPU KV cache size: 337,600 tokens" is per group. The real pool is 1,353,600 (846 × 1,600) (OM L348). WQ L362 [E]: "max concurrency 8.46x at 135k implies about 1.14M usable context tokens", and "This interpretation is inferred from the numbers, not from vLLM documentation."
26. **LB gap vs noise.** FR L3 / OM L545: −5.4 is "3-5×" the noise sd. OM L501: "within about 3-4 sd of LB noise".
27. **Upstream "DFLASH draft attention on sm_89" vs our runs.** SD L173: "No upstream SGLang doc confirms DFLASH draft attention works on sm_89." Our runs did launch and serve DFlash2 on sm89 via FlashInfer with 0 failed requests (EC L26). This is not a doc contradiction, but our runs settle the operational question; correctness (#36548) remains open.
28. **NguyenThanhNhan's 65.97 model.**
    - CE L156-159 / FR L395 [E]: the ~180 GB read "cannot be the 27B alone" and "matches the size of a Flash-Next FP8 checkpoint … his 65.97 run *may* be Flash-Next FP8, or a 27B plus large extra artifacts. This is an inference only."
    - Both docs list the 10-05 flag set (FP8 27B + EAGLE 3/1/4 …) against the about-23-LB "pure reasoning" submissions, not against 65.97.
    - The new evidence in section 6(b) states the 65.97 run used Qwen3.8-27B FP8 with that flag set, which supersedes the docs' Flash-Next inference.

---

## 5. Open questions only a GPU run can answer

(Source gap lists: OM L205-212, L284-288, L375-381, L429-432, L487-490, L550-556, L612-615; FR L438-469; SD gaps; EC gaps; WQ gaps; CE gaps.)

**Speculative decoding**
1. Same engine, same workload, 12 real running, long context (≥60k thinking): DFlash2-8 vs NEXTN MTP 3/1/4 (and 2/1/3) vs no spec. Measure aggregate tok/s, AL per 10k-token output window, mamba slots and KV. Never measured, because B was capped at 10 and C at 6.
2. AL per output-position window (0-10k, 10-30k, 30-60k) for MTP and DFlash2 in xhigh thinking on Qwen3.8-27B. No source has it.
3. Does SGLang's "accept len" include the bonus token? This is a source check, not a GPU run. Without it, SGLang and vLLM AL numbers are not comparable.
4. DFlash2 at other block sizes (e.g. 4-5) and with other drafter quantization on the INT4 target.
5. The DFlash2 draft-side cost at long context in SGLang (the vLLM #54691 effect): does step time grow with context relative to no-spec?
6. Above what concurrency does speculation stop paying on L4? Docs' inference: higher than H200's ~c32. Unmeasured above 13.
7. MTP mamba-slot and KV footprint at 12-16 streams with lazy vs extra_buffer, and fp16 vs fp32 state, at mem 0.83 with explicit sizing.
8. ReplaySSM with MTP/EAGLE on L4: memory freed and correctness, with fp32 state.

**Correctness (gating the speed choice)**
9. Target-only parity: same INT4 checkpoint, same W1 server, 12 concurrent, same tasks and prompts, ≥60k thinking; DFlash2 vs no-spec (or NEXTN). Measure per-task correct, train-pass → test-correct rate, a cross-request marker test at long context, and real running == 12. This addresses #36548 and #38009. The PARITY notebooks are built (f3876b40 / 85da1be3 / bc9e2bc2) with no results.
10. Engine/quant vs prompt: SGLang W1 with the old S1 prompt vs S1-60K vLLM FP8 MTP-3, same tasks, 60k cap. This would explain LB 29.17 vs 34.58.
11. FP8 vs INT4 paired eval at about 60k thinking, and INT4 token inflation at xhigh.
12. fp16 vs fp32 GDN state accuracy at 60k thinking.
13. FP8 KV accuracy on Ada kernels (FlashInfer sm89 / Triton) for >32k reasoning; and whether the INT4 checkpoint's 32 calibrated KV scales load at runtime (a log check).

**Weights and kernels**
14. Same engine and speculator on 4×L4: BF16 / FP8 (block or per-channel) / INT4 at 8 / 16 / 24 / 32 streams with 30-60k contexts.
15. Per-channel FP8_DYNAMIC (CUTLASS) vs block-128 FP8 (Triton) on L4; L4-tuned block-FP8 configs (E12 FP8 tuning, never run).
16. INT4 Marlin GEMM efficiency on sm89 at M = 64-128 (verify width × streams).
17. Whether compressed-tensors NVFP4/MXFP4 W4A4 takes the Marlin path on sm89 or errors.
18. vLLM ≥0.28 INT4 + DFlash2 arm (isolates the engine), and vLLM INT4 + MTP (#55533 real-running check).

**Concurrency, scheduling, memory**
19. W1 above 12 running; vLLM above 16.
20. A build with PR #44401 (true 12/12): measured gain vs the 13/12 workaround.
21. fp8 vs bf16 KV throughput A/B on L4.
22. Separate gains on Ada from CUDA graphs, the overlap scheduler (and `SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1`) and chunk size (2048 vs 4096).
23. vLLM per-request GDN state multiplier with `num_speculative_tokens`; vLLM at mem 0.95, MTP k=2, fp16 SSM.
24. vLLM throughput binned by context (never binned by us).

**Parallelism**
25. 2×TP2 INT4 (MTP or no-spec, router) vs TP4: real per-replica KV, aggregate tok/s, and whether TP2 gets custom AR (log line). Never measured on L4 in any engine.
26. PP on L4: no measurement anywhere.
27. Per-step all-reduce cost on 4 PCIe L4s. Open in the docs; now partly answered by 6(a).

**Startup and storage**
28. SGLang startup with a pre-built JIT cache shipped as a dataset (and its accuracy safety, #42917).
29. vLLM init breakdown (torch.compile vs graph capture).
30. Local disk size, `/dev/shm` cap, GCSFUSE persistence; the measured gain of parallel copy to local disk or RAM before launch.

**Other**
31. Flash-Next on any Ada GPU: no valid tok/s. The docs treat it as a research bet.
32. #43514 / #43518 fix status (an upstream check, not GPU); whether our requests ever use grammar.
33. Whether a 13th running request runs outside the captured graphs (graph bs list ends at 12, server cap 13). The docs do not address this.

---

## 6. Check against new evidence

### 6(a) 2026-10-23 Kaggle measurements vs the docs

| New fact | What the docs said | Agreement / correction |
|---|---|---|
| **All GPU pairs PHB** (PCIe through a host bridge) | "Kaggle L4 P2P capability, and whether TP2 gets custom all-reduce, are unverified" (EC L219). "This suggests the Kaggle L4s are not using direct P2P (host-staged/SHM path). Inferred; not verified with `nvidia-smi topo -m` or `NCCL_DEBUG=INFO`" (EC L204). FR L464: "whether Kaggle L4 has P2P is unknown" | The topology question is now answered. It is consistent with the docs' inference that the L4s do not benefit from direct P2P. |
| **NCCL P2P/IPC transport 5.5× slower** | vLLM forum: "An NCCL P2P toggle on dual consumer GPUs gave only about +2-5 % (sometimes negative); SHM is not always slower than P2P" (EC L172). TP2 custom AR is allowed "only if the P2P test succeeds. If Kaggle L4s lack P2P, TP2 also falls back to NCCL" (EC L206). The 3090 "TP2 + custom AR + verified P2P … +15.7 %" (EC L158) | The docs' P2P-based upside, TP2 custom AR (+15.7 % on 3090), depended on working P2P. With P2P/IPC 5.5× slower on Kaggle, that analog likely does not transfer (my reading; the docs never measured it). The docs' 3090 2×TP2 evidence came from "P2P patched" boxes (EC L148) and NVLink pairs (CE L400), which the docs already flag as "indicative only" (EC L209). |
| **983,040-byte all-reduce inside the graph: 345 µs per call** | "Decode all-reduces are small and latency-bound: about 2 per layer x 64 layers per step" (EC L205). The 64 MB preflight gave 20.9 ms (≈3.1 GB/s algorithm bandwidth) | (my arithmetic) 983,040 B = 12 streams × 8 verify tokens × 5,120 hidden (WQ L8) × 2 B (bf16), i.e. exactly the DFlash2 verify width M = 96 that FR L136 names ("our current 12 × 8 = 96"). 983,040 B / 345 µs ≈ 2.85 GB/s, close to the 64 MB preflight's ≈3.1 GB/s. So at about 1 MB the call runs at about the same effective bandwidth as at 64 MB, which looks bandwidth-bound rather than the "latency-bound" the docs assumed. If that holds, the per-call cost scales with the verify width: MTP-4 at 12 (M = 48) would be 491,520 B and no-spec (M = 12) 122,880 B. This is unmeasured. |
| **~56 ms of communication per decode step of ~115-133 ms at 12 running** | "NCCL allreduce cost per step on 4 PCIe L4s (128 allreduces per decode step for 64 layers) was not quantified in any source" (WQ L197; FR L466). FR L311 [E]: 2×TP2 beats TP4 when TP4's per-step all-reduce overhead exceeds about 15 ms (INT4) or about 24 ms (FP8). vLLM "about 145 ms/step measured vs a ~60 ms bandwidth ceiling" [E10T L67]. SD L101 roofline: about 35 ms per step at 16 × 35k | (my arithmetic) 56 ms / 345 µs ≈ 162 calls per step, above the docs' "about 128" (2 × 64 layers). The docs did not count drafter, verify or logits collectives, so the extra calls are unattributed. Communication is 42-49 % of the step and exceeds the docs' entire per-step memory roofline (about 33-35 ms). That fits the docs' observation that measured numbers run at "about 50 %" of the roofline (WQ L171) and "30-45 % of a naive bandwidth bound" (CE L147). By the docs' own criterion, 56 ms > 15 ms (INT4) and > 24 ms (FP8), so the condition under which 2×TP2 should beat TP4 is met. The criterion ignores TP2's own 2-rank all-reduce cost, which the docs never measured. Consistency check: 12 × AL 4.34 (E11-06 at 12) / 0.133 s ≈ 392 tok/s, and 12 × 4.5 / 0.115 s ≈ 470 tok/s, which brackets the measured 389-500 at 12 running (my arithmetic; it assumes accept length counts all emitted tokens). |
| **First launch 502 s (~410 s compile/graph capture); relaunch 71 s** | W1 cold start "about 520-680 s ready. Most of it, about 372-385 s, is the target-verify CUDA graph capture, plus 13-88 s weight load" (OM L389; FR L378). E11-04 W2 cached relaunch: 71 s (target_verify 11.94 s). FR/OM [E]: a persisted Triton/FlashInfer cache could save about 6 minutes per run, never tested across sessions; #42917 says a stale cache can ruin accuracy | Consistent. 502 s is just below the documented 520-680 s range, and 71 s equals E11-04 W2's cached 71 s exactly. ~410 s of compile/capture is about 25-38 s more than the target-verify capture alone (372-385 s), plausibly the draft-verify graph and other JIT, which the docs do not itemize. The new pair bounds the in-session saving of a warm cache at about 431 s (≈7.2 min), slightly above the docs' "about 6 min" estimate. The cross-session question (a shipped cache) and the #42917 accuracy risk remain open. |

**Docs' items affected by 6(a)**
- **Lever 46 (2×TP2).** FR/EC rank it the largest untested lever, "+20-100 % possible". The new comm share strengthens the docs' own condition for it. The P2P-based sub-lever (TP2 custom AR) now looks unlikely to help, and TP2's own NCCL cost is unmeasured.
- **Lever 50 (NCCL P2P toggle).** It is now measured on Kaggle as negative (5.5× slower), consistent with the forum's "+2-5 %, sometimes negative".
- **Verify width.** The docs tie M to compute (Marlin ridge about 100; FR L136). The new evidence ties it to communication as well (payload ∝ M), which the docs did not consider. This matters for MTP-4 (M = 48) vs DFlash2-8 (M = 96) at 12 streams. It is an inference, unmeasured.
- **SD's MTP overhead.** SD L103 lists "PCIe all-reduce latency for each sequential draft step" as an MTP overhead. MTP 3/1/4 runs 3 sequential draft steps, while DFlash2 here runs 1 (`speculative_num_steps=1`). Their per-step collective count and size on Kaggle are unmeasured. Whether SGLang shards the DFLASH drafter under TP4 is an open gap (SD L208; FR L452).

### 6(b) The 65.97-LB team's SGLang flags vs ours, and what the docs say about each difference

**Attribution note.**
- The docs record exactly this flag set from NguyenThanhNhan's topic 745787 (10-05): "qwen 3.8 27b fp8 + mtp at 65k tokens and xhigh reasoning effort", 1 pass, forced commit after 49k thinking, concurrency not disclosed, about 23 LB from 2 "pure reasoning" submissions (CE L36, L50; FR L405).
- For the later 52.08 → 65.97 run (topic 751344), the docs say the model was not stated. It read about 180 GB, with a 45-minute server timeout ("normally … at most 15 minutes"), 875-1300 s model loads in save runs, "mtp draft setup", and two 0.00s from GCSFUSE (CE L37, L53; FR L406).
- The docs inferred the 180 GB might mean Flash-Next FP8 (CE L156-159; FR L395).
- The new evidence says the 65.97 run used Qwen3.8-27B FP8 with these flags. That supersedes the docs' Flash-Next inference; the docs do not explain the 180 GB read for a 30.9 GB 27B FP8. The 49k forced-commit, 65k cap and 1-pass details are documented only for the 10-05 submissions.
- Darren estimated the 10-05 setup needed ≥360 tok/s (CE L36).

| Setting | 65.97 team | Ours (W1 / E12) | What the docs say about the difference | Evidence class |
|---|---|---|---|---|
| Weights | FP8 (27B) | RedHat INT4 W4A16 (Marlin) | **Accuracy:** official FP8 "nearly identical" to BF16; RedHat INT4 −1.25 (AIME25) / −1.35 (GPQA-D), about 1-1.5 points, "inside one-seed noise but consistently negative", possibly larger on hard tasks ("up to 4×"); possible token inflation +3.7 % (GPTQ) to +14.4 % (AWQ) in one paper, none in another (WQ L204-260, L284-286; FR L148-163). Ours: LB 34.58 (vLLM FP8) vs 29.17 (SGLang INT4), confounded with the P1 prompt; E11-05 7/15 train-verified programs wrong on test vs R5 3/13, small sample. **Speed:** roofline INT4 ≈ 1.28× FP8 at B=16 / 30k (1.19× at B=32, 1.38× at B=8) [E]. In SGLang on sm89, FP8 block-128 GEMMs land on Triton via `--fp8-gemm-backend auto`; vLLM's equivalent runs untuned defaults on L4 (WQ L66-69, L79). **Memory:** FP8 7.15 GiB/GPU vs INT4 4.62 GB measured, about +2.5-3 GB/GPU of weights (my arithmetic from those two), but no DFlash2 drafter (−1.49 GB weights, −0.92-2.10 GB draft KV per GPU). **Under speculation:** FR L136 [E]: INT4 Marlin nears its ridge (M ≈ 100) at 12 × 8 = 96, while FP8 W8A8 is memory-bound to M ≈ 400, and "FP8 does relatively better under heavy speculation" (WQ L469). No SGLang FP8 run on L4 exists in the docs | measured-ours (confounded) + measured-elsewhere + estimate |
| `--kv-cache-dtype fp8_e4m3` | same | same | Docs: mandatory on L4 for capacity (bf16 KV would roughly halve W1's 881k to about 440k [E]); accuracy ≤0.7 points (Qwen3.5-27B, vLLM blog) but untested on Ada kernels at >32k; default scale 1.0 (FR L181-189) | — (no difference) |
| `--attention-backend flashinfer` | same | same | Ada default (flashinfer, else triton). FlashInfer's plan stream adds a sync point that limits Spec V2 overlap. SGLang MTP with FlashInfer needs FlashInfer newer than 0.6.15.post1, otherwise triton; our wheelhouse has flashinfer_python 0.6.18 (EC L23, L27, L292; SD L148) | — |
| `--linear-attn-backend triton` | same | same (triton decode/prefill/verify) | FlashInfer GDN needs SM90+, so triton is the only option on sm89 (EC L24-25) | — |
| `--chunked-prefill-size` | **2048** | **4096** | Cookbook tip: 2048, "since 8192-token chunks stall decode about 600 ms at a time". Small chunks (1-2k) protect decode at TTFT cost; prefill is a small share at ~8k prompt / ~60k output. No Ada measurement (FR L353; EC L283, L300). Both settings are below the 8192 the cookbook warns about | claim + estimate |
| `--max-prefill-tokens` | **4096** | **16384** | Values listed only: NguyenThanhNhan 4096, Franzen 16384, ours 16384. No measured effect on L4 (FR L353) | none |
| `--mamba-radix-cache-strategy` | **extra_buffer** | **extra_buffer_lazy** | extra_buffer: S = 5 slots per request (with overlap), the Qwen3.8-27B page's default, needs `--page-size 64` and FLA kernels. lazy: S = 4, "no accuracy cost stated", spec support added in a changelog. Both run the overlap scheduler (FR L323; EC L232-237; SD L143-145). #43514's silent accept-0 trigger involved `extra_buffer` with `--enable-mixed-chunk` + grammar, and DFlash only (FR L266). jpezzulli: a cap computed for 5 slots while 4 were held left about 20 % of the cache unreachable (SD L187) | measured-ours (slot counts from source) + claim |
| Mamba pool sizing | **`--mamba-full-memory-ratio 0.5`** | **`--max-mamba-cache-size 56`** (explicit; overrides the ratio) + `--mem-fraction-static 0.83` | Default 0.9 "over-provisions the KV pool and clamps concurrency". The ratio sets mamba memory relative to KV memory; `--max-mamba-cache-size = target_concurrency × S` overrides it. 2.5 "consumes 15.2 GB, starving the KV pool" (#160). FR L323 notes NguyenThanhNhan's 0.5. Formula: `ratio = (S + D) × state_bytes / (L × kv_bytes_per_token)`. (My arithmetic, using the doc formula, S = 5 extra_buffer, D = 4 EAGLE, fp32 slot 153.9 MB, fp8 KV 32.8 KB/token, and taking L as the per-request token budget, which the docs do not define: L = 9 × 153.9 MB / (0.5 × 32.8 KB) ≈ 84k tokens, about a 65k cap plus prompt.) Ours: explicit 56 slots / S = 4 = 14 requests of state, KV 881,336 (866,744 at server cap 13). The intermediate verify cache is sized separately in the source (FR L343) | claim (vendor) + measured-ours + my arithmetic |
| `--mamba-ssm-dtype` | **float32** | **float16** | **Memory:** fp32 153.9 MB vs bf16 78.4 MB per slot (whole model). On our L4 at mem 0.83 / lazy / 56 slots with DFlash2: fp32 W3 KV 598,712 vs fp16 W1 881,336 (intermediate 3.66 vs 1.83 GB). At auto memory: DFlash2 cap 6 (fp32) vs 13 (fp16); NEXTN fp32 cap 10 (FR L327-337). FR L343 [E]: fp32 + lazy + D = 8 at 12 streams (144 slot-equivalents) is "impossible on 4×24 GB". With MTP (D = 4) the footprint is smaller: SD estimates 16 streams × 5 slots ≈ 3.0 GB/GPU in fp32 vs 1.5 in fp16. **Speed:** cookbook single-request EAGLE: NVFP4 fp32 152.9 vs bf16 144.5; FP8 fp32 106.3 vs bf16 116.1 tok/s/user. Ours: E11-04 sweep W3 fp32 100.0 / 273.5 / 337.9 vs W1 fp16 110.6 / 257.5 / 363.8 at 1 / 4 / 8. **Accuracy:** fp32 is the checkpoint's declared precision. No public 60k study of fp16 vs fp32; cookbook "treat bf16 as an accuracy gate"; Nemotron FP16 cast "up to 40 % increase in verbosity" with W8A8; Nemotron Ultra FP16 with stochastic rounding matches FP32. Our evidence: W1 fp16 8/8 sanity, W2 bf16 7/8; greedy drift W3 vs W1 44.0 tokens and D vs C 166 (FR L343; WQ L275, L288; EC L252-260). **ReplaySSM** keeps an fp32 checkpoint with D = 0 and works with MTP/EAGLE topk ≤ 1, not DFlash2; untested on L4 (FR L345) | measured-ours + measured-elsewhere + claim |
| Speculative algorithm | **EAGLE (MTP) 3 / topk 1 / 4 draft tokens** | **DFLASH (DFlash2), 8 draft tokens, 1 step, unquant drafter** | See 3.1. Ours on L4: MTP (B, INT4 target, fp32, extra_buffer, auto memory) AL about 3.1 flat, 297 tok/s at 8 vs DFlash2 fp16 356 at 8, capped at 10 (server decode 373.7 at 10); never measured at 12 in the same engine. Long capped runs: B 262.3 (cap 10) vs C 258.3 (cap 6) vs A 282.4 (cap 22). DFlash2 production AL 3.8-4.5 vs vLLM MTP-3 2.67-3.0 on our workloads. Elsewhere: MTP thinking AL about 2.5 of 4; vendor H200 c32 MTP 0.77-1.04× vs DFlash2 1.01-1.45×; external drafters decay on long outputs (MTP untested); 4× 4090 TP4 DFlash draft cost grows with context. **Correctness:** #36548 (DFlash2, not reproduced with MTP/EAGLE on 0.5.18), #38009, #43514 are DFlash-specific; MTP's known SGLang issue is the `--speculative-adaptive` crash (#23330). **Memory:** MTP head inside the checkpoint (0.477 GB in the FP8 repack; 0.85 GB BF16 in RedHat INT4) vs DFlash2 3.85 GB (1.49 GB/GPU) + draft KV. **FR ranking:** DFlash2 wins on throughput in our evidence; MTP has lower correctness risk, so it is rank 2 (FR L246-248) | measured-ours (partial) + measured-elsewhere + claim |
| `--speculative-num-steps 3 / --speculative-eagle-topk 1 / --speculative-num-draft-tokens 4` | same as the SGLang default | n/a (DFlash2: 8 draft tokens, steps 1) | SGLang default; evidence band 2-3 steps / 3-4 draft tokens; topk 1 chains, never trees, at high batch. B200 c8 SGLang MTP steps 3 = 736.2 vs 5 = 740.5 vs 2 = 703.1. SD also suggests trying 2/1/3. GEMM width at 12 streams: 12 × 4 = 48 vs DFlash2 12 × 8 = 96 (FR L136; SD L113-131) | measured-elsewhere + estimate |
| `--max-running-requests` / `--cuda-graph-max-bs-decode` | not given in the evidence (concurrency undisclosed in the 10-05 post) | 13 (server) with client 12 / 12 (graphs captured for bs [1, …, 8, 10, 12]) | 13/12 is the docs' workaround for the 0.5.21 11-of-12 admission bug (double-counted chunked-prefill request), costing about 8 % if not worked around; the fix is PR #44401 (FR L270). The real cap must be asserted from the log, because `/get_server_info` shows 32 (FR L339). W1 has never run above 12. Franzen used `--cuda-graph-max-bs-decode 10` with an explicit list. The docs do not address what happens to a 13th running request beyond the largest captured graph | measured-ours + claim |

**Net, per the docs.** The 65.97 configuration matches the docs' own "rank 2" family and SD's practical recipe in algorithm and settings: SGLang + EAGLE/NEXTN 3/1/4, fp8 KV, flashinfer, triton GDN, small chunks.

It differs from SD's recipe in three ways:
- it uses `extra_buffer`, not `extra_buffer_lazy` or skip-decode-lock;
- it uses fp32 SSM, not half precision with an acceptance check;
- it sizes the pool by ratio rather than an explicit `--max-mamba-cache-size`.

It differs from our W1 in FP8 weights and MTP instead of INT4 and DFlash2. Those are the two axes on which the docs say W1's measured speed advantage rests and on which W1's accuracy is least verified (FR L3, L282, L475).

Every individual difference above lacks a same-engine L4 measurement in the docs. The docs have no SGLang FP8 run, no SGLang MTP run at 12 real running, no SGLang fp32-state run with MTP at mem 0.83, and no chunk-size A/B.
