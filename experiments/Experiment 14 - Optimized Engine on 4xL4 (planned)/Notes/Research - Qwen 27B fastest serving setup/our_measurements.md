# Our measured serving numbers: Qwen3.8-27B (and Flash-Next) on Kaggle 4x L4 (2026-09-13 to 2026-10-22)

These notes use local workspace files only. All paths are relative to `D:/ARC AGI 2/` unless they start with `C:/` or `_scratch/`. A citation is written as `[KEY Lnn]`, and the key table below gives the full path.

Three kinds of numbers appear:
- **[M]** a number written in a run log, report or analysis doc.
- **[D]** a number I derived on 2026-10-23 from a raw log or metrics file, using scratch scripts `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/sgl_extract.py` and `vllm_metrics.py`. Medians of TP0 decode lines, 10 s engine lines, or about 2 min metric snapshots.
- **[P]** a projection or estimate written in the docs. These are not measurements.

**Source keys** (E11 = `experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/`; KR = its `Experiment Results/Kaggle Runs/`):

**Experiment 11 (SGLang INT4 + DFlash2)**

| Key | Path |
|---|---|
| E11R | E11/README.md |
| E113A | E11/Experiment Results/E11-03 run1 analysis (2026-10-17).md |
| E113REP | KR/E11-03 rev3b run1 (foysal, 2026-10-17)/e11_03/report.md |
| E113LOG-A / -B / -C / -D | KR/E11-03 rev3b run1 (foysal, 2026-10-17)/e11_03/server_A_int4_try0.log, server_B_int4_mtp_try0.log, server_C_int4_dflash2_try0.log, server_D_C_int4_dflash2_ssm_fp16_try0.log |
| E114A | E11/Experiment Results/E11-04 run1 analysis (2026-10-18).md |
| E114REP | KR/E11-04 rev2b run1 (esfersami50, 2026-10-18)/e11_04/report.md |
| E114LOG-W1 / -W2 / -W3 | same folder, server_W1_lazy_fp16_try0.log, server_W2_lazy_bf16_try0.log, server_W3_lazy_fp32_try0.log |
| E115A | E11/Experiment Results/E11-05 run1 analysis (2026-10-18).md |
| E115LOG / E115SUM / E115RR / E115REQ | KR/E11-05 EVAL30 run1 (esfersami50, 2026-10-18)/q27x/sglang_server_try0.log, sglang_decode_summary.json, poetiq_run_report.json, sglang_requests.jsonl |
| E115K | KR/E11-05 EVAL30 run1 (esfersami50, 2026-10-18)/q27-xhigh-e11-05-s49-sglang-eval30.log |
| E115S1SUM | KR/E11-05 S1COMMIT4 run1 (foysal, 2026-10-18)/q27x/sglang_decode_summary.json |
| E116A | E11/Experiment Results/E11-06 P1-60K EVAL14 run1 analysis (2026-10-19).md |
| E116LOG / E116SUM / E116RR | KR/E11-06 P1-60K EVAL14 run1 (esfersami50, 2026-10-19)/q27x/sglang_server_try0.log, sglang_decode_summary.json, poetiq_run_report.json |

**Experiment 12 Duck (E12D = `experiments/Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)/Experiment Results/`)**

| Key | Path |
|---|---|
| E12SMA / E12EV30A / E12EV6A / E12R5A / E12G4BA | E12D/E12 SGLang SMOKE1 run1 analysis (2026-10-19).md; E12 SGLang EVAL30 run1 analysis (2026-10-19).md; E12 SGLang EVAL6 run1 analysis (2026-10-20).md; E12 rev 5 runs analysis (2026-10-21).md; E12 rev 6d G4B GPU run analysis (2026-10-21).md |
| E12LOG-SM / -EV30 / -EV6 / -EV6R5 / -LOW | E12D/Kaggle Runs/<run folder>/q27x/sglang_server_try0.log, run folders "E12 SGLang SMOKE1 run1 (foysal, 2026-10-19)", "EVAL30 run1 (hosen42, 2026-10-19)", "EVAL6 run1 (hosen42, 2026-10-20)", "EVAL6R5 run1 (hosen42, 2026-10-20)", "EVAL12R5_LOW run1 (foysal, 2026-10-20)" |
| G4BLOG | _scratch/g4b_bundle_verify/runs/rev6d_EVAL12R6G4B/sglang_server_try0.log (extracted copy of the G4B run outputs) |
| E12VLOG | E12D/Kaggle Runs/E12-V vLLM SMOKE1 run2 (esfersami50, 2026-10-19)/q27x/vllm_openai_server.log |

**vLLM runs (E10K = `experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Results/Kaggle Runs/`)**

| Key | Path |
|---|---|
| O2LOG / O2RR | E10K/GPU EVAL120 ORDER2 86k DIRECT (hosen42 q27-xhigh-e10-order2)/q27-xhigh-e10-order2.log and q27x/run_report.json |
| R5LOG / R5MET | E10K/GPU EVAL120 Poetiq R5 (hosen42 q27-xhigh-e10-poetiq-r5-eval120)/q27-xhigh-e10-poetiq-r5-eval120.log and q27x/poetiq_vllm_metrics.json |
| D66MET / L66MET / W3HMET / R4MET | poetiq_vllm_metrics.json in the D66, L66, W3H low 3h and EVAL12 Poetiq R4 run folders under E10K |
| W3HLOG | E10K/GPU EVAL120 W3H low 3h (esfersami50 q27-xhigh-e10-w3h-low-3h)/q27-xhigh-e10-w3h-low-3h.log |
| R7RR | experiments/Experiment 1 - Q27-XHIGH Direct Reasoning (LB 18.75 and 26.94)/Experiment Results/Kaggle Runs/Q27X_run7_v144/q27x/run_report.json |
| E4LOG | experiments/Experiment 4 - Q27-AGENT REPL Agent and HYBRID (v1 to v1.8.3)/Experiment Results/Kaggle Runs/run4_v1_8_3_foysal_EVAL12_complete/vllm-server.log |

**Docs read and cited by a sub-agent (line numbers as it reported them)**

| Key | Path |
|---|---|
| E1R / E1r3 / E1r4 / E1V / E1aud | Experiment 1 README.md; Kaggle Runs/Q27X_run3/README.md; Q27X_run4/README.md; "v1.4.4 vs all experiments - remaining issues (2026-10-12).md"; Experiment Results/R5 validate output/Validate output/r5_bundle/Q27X_RUN4_DEEP_AUDIT_2026-10-06.md (by GPT) |
| E4R | Experiment 4 README.md |
| E6A | Experiment 6 .../Experiment Results/Analysis/GPU EVAL12 dual v2 analysis (2026-10-12).md |
| E7R | Experiment 7 README.md |
| E9G | Experiment 9 .../Analysis/GPU EVAL12 E9 result (2026-10-13).md |
| E10R / E10O / E10R5 / E10R4 / E10C / E10W / E10T | Experiment 10 README.md and, under Experiment Results/Analysis/: GPU EVAL120 ORDER2 86k analysis (2026-10-14).md; GPU EVAL120 Poetiq R5 analysis (2026-10-15).md; GPU EVAL12 Poetiq R4 deep analysis (2026-10-14).md; Concurrency 16 to 25-30 - math and research (2026-10-15).md; W4A16 checkpoints - MTP and vLLM 0.19 compatibility (2026-10-15).md; Where tokens and time go - Poetiq R5 + web research (2026-10-15).md |
| JLB / JDL / JWK / JJ | resources/Analysis and Journals/: Qwen3.8-27B LB and run deep analysis (2026-10-16).md; D66 and L66 deep runs - findings and decisions (2026-10-17).md; Weakness analysis vs top teams (2026-10-19).md; ARC_AGI2_Qwen38_Project_Journal_through_v2.4G_2026-09-18.md |
| E3R / E3LOG421 / E3REP421 / E3NB421 / E3LOG33 | experiments/Experiment 3 - Flash-Next Feasibility (FN)/README.md; Experiment Results/Kaggle Runs/FN_PROBE_v421_run1/fn_smoke/serve.log; .../fn_smoke/fn_smoke_report.json; .../fn-probe-flashnext-4xl4.log; Kaggle Runs/FN_v33_run/fn_smoke/serve.log |
| E12FP8R | experiments/Experiment 12 - FP8 kernel tuning on L4 (planned)/README.md |
| E13P / E13V | experiments/Experiment 13 - Flash-Next in the Duck harness on 4xL4 (planned)/PLAN.md and VALIDATION.md |

**Memory notes**

| Key | Path |
|---|---|
| MEM-P | C:/Users/hp/.claude/projects/d--ARC-AGI-2/memory/arc-agi2-qwen27b-project.md |
| MEM-SG | same folder, sglang-int4-dflash2-plan.md |
| MEM-T | same folder, no-token-cuts-below-base.md |
| MEM-A3 | same folder, arc-agi3-duck-p1-sparse.md |

---

## Throughput: aggregate and per-stream generation tok/s by running requests

### Takeaway

The fastest configuration we measured on 4xL4 is SGLang 0.5.21 + RedHat INT4 W4A16 + DFlash2 (block 8, unquantized drafter), with fp16 SSM state, `extra_buffer_lazy`, mamba cache 56, no prefill graph and mem 0.83. We call this config **"W1"**.
- **Decode with 11–12 really running:** about 400–460 tok/s at contexts under about 30k, and about 320–390 tok/s at 40–60k.
- **Whole production runs:** 328.9 tok/s (E11-05) and 379.9 tok/s (E11-06).

vLLM 0.19 FP8 + MTP-3 at 16 running decodes about 350 tok/s on DIRECT prompts (T 0.6) and about 295–300 on Poetiq at T 1.0. Whole-run averages:
- DIRECT: 333.3 (ORDER2) and 274.6 (run 7);
- Poetiq: 281.5 (R5), 256.0 (D66) and 254.3 (L66).

So SGLang W1 is roughly +17–30 % on whole runs, not 1.5x. Coverage-level effects (KV thrash, prefill, admission) move the effective rate more than the engine does.

### Cited Findings

**A. SGLang INT4 + DFlash2, production config W1** (TP4, fp8_e4m3 KV, DFLASH 8 draft tokens unquant, SSM fp16, `extra_buffer_lazy`, mamba 56, mem 0.83, `disable_prefill_cuda_graph`, chunk 4096, context 155,648; launch flags at [E114LOG-W1 L8], [E115LOG L8], [E116LOG L8])

- **E11-04 PROD closed loop** (2026-10-18, esfersami50, 12 really in flight, ARC prompts, T 1.0, up to 60k thinking, 600 s window, contexts only up to 30,177): [M]
  - W1 fp16 **416.5 tok/s**; W2 bf16 412.1 tok/s.
  - Decode rate at 12 by mean context, W1: <10k 458.3, 10–20k 413.3, 20–40k 388.4, ≥40k not reached.
  - W2: 465.5 / 401.9 / 374.9.
  - Sources: [E114REP "PROD closed loop" table and "Decode rate at 12" table]; [E114A L12-21].
  - E114A L21 says "about 350–370 at 40–57k" — that is a hypothesis [P].
- **E11-04 sweep** (short 4,096-token requests), total tok/s at 1 / 4 / 8 requests: [M] [E114REP sweep table]
  - W1: 110.6 / 257.5 / 363.8; log gen 105.6 / 281.4 / 442.6.
  - W2: 114.3 / 272.8 / 357.6.
  - W3 (fp32 SSM): 100.0 / 273.5 / 337.9.
  - At 12 requests, real running max was 11 (FAIL): W1 325.9, W2 350.9, W3 321.4.
- **E11-05 S49 Poetiq EVAL30** (2026-10-18, esfersami50, 49k thinking): [M]
  - **328.9 tok/s** = 4,253,591 tokens / 12,932.4 s solver wall [E115RR, `tok_per_sec`].
  - 12 set, **11 really running**: 2,358 decode lines show 11 running + 1 queued, and only 114 show 12 [E115A L19-36].
  - Decode at 11 running by context: <10k 429.2, 10–20k 371.8, 20–40k 340.0, 40–60k 323.4; median at 12 = 360.8 [E115SUM; E115A L25-32].
  - The same doc compares against vLLM R5 EVAL120 280.9 and D66/L66 255.8/254.1 [E115A L19-23].
- **E11-06 P1-60K EVAL14** (2026-10-19, esfersami50, 60k thinking, server `--max-running-requests 13` with client 12 as the 11-of-12 workaround): [M]
  - **379.9 tok/s** = 2,262,599 tokens / 5,956.2 s [E116RR; E116A L18]; 87.2 % cached prompt tokens, 0 errors.
  - Decode at 12 running (278 lines): median 416.6.
  - By context at 12: <10k 500.1, 10–20k 460.3, 20–40k 409.0, 40–60k 389.4.
  - At 8 / 9 running (408 / 400 lines): 368.5 / 388.6 [E116SUM].
- **Per-stream rates under production load** [D] (gen tok/s ÷ running, TP0 decode-line medians):
  - E11-05: 30.7 at 11 running [E115LOG].
  - E11-06: 34.7 at 12, 43.2 at 9, 46.1 at 8 [E116LOG].
  - One E11-05 phase-1 request took 49,152 tokens in 1,418.5 s = 34.7 tok/s per stream [E115REQ, rid 6ffbe589~e0~i0|1].

**A2. Same W1 engine inside the Duck REPL agent (E12; server max-running 13, KV 866,744)**

| Run | Running | Decode, median tok/s [D] | Per stream [D] | Effective / whole-run rate |
|---|---|---|---|---|
| E12 EVAL30 (hosen42, 2026-10-19) | 11 (728 lines) / 12 (280 lines, p25 149.7, bimodal) | 412.5 / 431.3 | 37.5 / 35.9 | Effective only ~187 tok/s over the solve window (2.36 M tokens / 12.6 ks) because of KV thrash; the doc gives decode as ~420 at 11–12 even at ≥60k context [E12EV30A L43-44] |
| E12 EVAL6 (2026-10-20) | 6 (796 lines) | 416.3 | 69.4 | — |
| EVAL6R5 xhigh | 6 | 322.2 | — | 244 overall [E12R5A L60] |
| EVAL12R5_LOW | 6 | 351.9 | — | 265 overall [E12R5A L60] |
| G4B rev 6d (2026-10-21) | 11 (137 lines) / 6 (151 lines) | 383.3 / 321.4 | — | 277 tok/s over the window [E12G4BA L20] |
| E12 SMOKE1 | 2 | 204.0 | — | — |

- Run sources: E12 EVAL30 [E12LOG-EV30]; EVAL6 [E12LOG-EV6]; EVAL6R5 [E12LOG-EV6R5]; EVAL12R5_LOW [E12LOG-LOW]; G4B [G4BLOG]; SMOKE1 [E12LOG-SM; E12SMA L29 "204 tok/s with 2 running"].
- G4B doc values: 370–406 at 11 (20–40k), 370.9 → 321.4 at the same context bin after the 12 → 6 step-down, i.e. −13 % [E12G4BA L44-47].
- Rev 4 "6 → 12 running: 368 → 431 tok/s" [E12R5A L78].

**A3. E11-03 short sweep** (2026-10-17, foysal; prompts ~4.3k, 4,096 new tokens, total tok/s by REAL running, after the cap correction) [E113A L45-57]:

| Real running | A INT4 | B INT4 + MTP (NEXTN) | C INT4 + DFlash2 fp32 SSM | D DFlash2 fp16 SSM | vLLM FP8 + MTP-3 (earlier runs) |
|---|---|---|---|---|---|
| 1 | 45.6 | 55.1 | 93.7 | 107.5 | — |
| 4 | 145 | 205 | 250 | 278 | — |
| 8 | 229 | 297 | (capped at 6) | 356 | — |
| 12 | 289 | (capped at 10) | (6) | **404** | ~307 |
| 16 | 347 | (10) | (6) | (capped at 13) 381 | ~373 |

- Per-request tok/s for D: 107.5 / 72.1 / 48.6 / 34.8 at 1 / 4 / 8 / 12 requests [E113REP D table].
- Long runs, 12 × 24,576 (capped): A 282.4, B 262.3, C 258.3 [E113REP].
- Server decode-line medians by real running [D]:
  - A: 45.7 (1), 147.6 (4), 248.5 (8), 297.8 (12), 370.1 (16), 440.0 (22) [E113LOG-A].
  - B: 80.0 (1), 340.9 (8), 373.7 (10) [E113LOG-B].
  - C: 85.3 (1), 329.7 (6) [E113LOG-C].
  - D: 108.0 (1), 299.6 (4), 482.9 (8), 477.2 (12), 489.1 (13) [E113LOG-D].
- longctx on C (4 × ~85.5k context, 1,024 new tokens): "16.3 tok/s" per request, accept 4.58, KV usage max 0.87 [E113REP]; server decode median about 238 tok/s at 4 running [E113A L69].

**B. vLLM 0.19 FP8 repack + MTP-3** (TP4, fp8_e4m3 KV, FlashInfer attention, `max_num_batched_tokens` 4096, `max_num_seqs` 16, `disable_custom_all_reduce` True, prefix caching on, `language_model_only`; flags at [R5LOG L548])

- **ORDER2 86k DIRECT EVAL120** (hosen42, 2026-10-14): [M] **333.3 tok/s** = 10,447,055 gen tokens / 31,343.2 s [O2RR]. Engine 10 s lines by running [D, O2LOG]:

  | Running | 1 | 2 | 5 | 7 | 10 | 11 | 12 | 15 | 16 (2,844 lines) |
  |---|---|---|---|---|---|---|---|---|---|
  | Aggregate tok/s | 44.3 | 81.3 | 163.1 | 212.2 | 268.5 | 298.0 | 293.4 | 323.3 | **352.0** |
  | Per stream | 44.3 | 40.6 | 32.6 | 30.3 | 26.9 | 27.1 | 24.4 | 21.6 | 22.0 |

- **Journal table, DIRECT aggregate (per sequence)** [M] [JLB L172-176]:
  - 3 active: 107 (36); 6 active: 203 (34); 12 active: 302–325 (25–27); 16 active: **351** (22).
  - "We have never run above 16."
  - Context cost: "about −5 % per +10k tokens… 418 tok/s at 0–10k, 330 at 50–60k" [JLB L178].
- **Poetiq runs (T 1.0):** [M] whole-run rates; per-interval medians [D]
  - **R5 EVAL120** (hosen42, 2026-10-15): 11,633,447 gen tokens over 41,332 s = **281.5 tok/s** [D, R5MET]; doc says 281 [E10R5]. Median at 16 running: 295.1 per metric interval [D, R5MET], 300.5 per 10 s line [D, R5LOG] (18.4–18.8 per stream). Heavy-prefill windows: 138 tok/s [JLB L182-183].
  - **D66** (77.8k, 16 slots): 256.0 tok/s [D, D66MET]; doc 255.8 [JDL L32-33]. Median at 16: 286.5.
  - **L66** (122.9k, 12 slots): 254.3 tok/s [D, L66MET]; doc 254.1. Median at 12: 253.2 (21.1 per stream).
  - **R4 EVAL12** (≤12 of 16 slots): 225.2 tok/s [D, R4MET]; doc 224 [E10R4 L22-26]. Medians: 185.1 at 7, 215.2 at 8, 290.8 at 12.
- **W3H low effort** (esfersami50, 2026-10-17): 404.0 tok/s [D, W3HMET]; doc 400 [E10R L425-429]. Median at 16: 430.0 per 10 s line [D, W3HLOG].
- **Direct v1.x runs (EVAL12):** [M]
  - run 3 v1.3.5: 307 tok/s at 12 active, 247.8 overall [E1r3 L16-31].
  - run 4 v1.4: **373 tok/s with 16 running**, 248.9 overall [E1r4 L28-63]. GPT audit of run 4: 215 ten-second windows at 16 active, mean 356.8, median 369.9 [E1aud L69].
  - run 6 v1.4.1: 285 [E1V L37].
  - run 7 v1.4.4: 274.6 = 1,059,550 tokens / 3,859.1 s, max_active 16 [R7RR].
  - E9 86k: 306.6 overall [E9G L20-25].
  - E6 dual v2: 288.5 at 16 running [E6A L3-7].
  - E7: 249–285 at 16 running [E7R L56-59].
- **Agent / OpenAI-server mode** (Exp 4 run 4, v1.8.3): agent 256.5 tok/s [MEM-P L77]. Median 289.5 at 12 running, 24.1 per stream [D, E4LOG].
- **E12-V vLLM Duck SMOKE1 run 2** (esfersami50, 2026-10-19): 2 running 100.2 tok/s and 1 running 55.2 [D, E12VLOG]. The memory note gives "decode at 2 running 91 tok/s vs SGLang 204" [MEM-P L246].
- **Early vLLM numbers:** [M]
  - 2026-09-13 Q1 (enforce_eager=True, custom all-reduce off, max_num_seqs 16): "≈ 6 tok/s per sequence, ≈ 51 tok/s aggregate" [MEM-P L13].
  - Archive journal: "C0 long@8 ≈ 48.30 tok/s; C1 graph/noAR ≈ 129.17 tok/s; ≈2.67× speedup" [JJ L125].

**C. Analyses and projections** [P]
- "about 145 ms/step measured vs a ~60 ms bandwidth ceiling" [E10T L67, as reported by the sub-agent].
- Concurrency 25–30 projected at about 380–430 tok/s on vLLM [E10C L27-51].
- "about 1,000 tok/s is needed to give 240 tasks the ARC-Prize budget (2 × 85k) in 12 h" [JWK L46].
- 240-task first pass needs 16.6 h at 1× [MEM-P / d66 note].

### Inferences

- At equal real concurrency (11–12) and context, SGLang W1 decodes about 15–30 % faster than vLLM FP8 MTP-3:
  - SGLang W1: 340–420 at 20–60k;
  - vLLM: 293–352 at 12–16 running, with a mean context we did not bin.
- That gap is smaller than the short-prompt sweep suggested (404 vs ~307 at 12).
- vLLM reaches its ~350 with 16 streams. SGLang reaches ~400+ with only 11–12, because each stream is faster: about 35 vs 22 tok/s per stream.
- Throughput is strongly temperature/sampling-dependent on vLLM: at 16 running, DIRECT T 0.6 gave 352, Poetiq T 1.0 gave 295–300, and low effort gave 430.
  - This tracks MTP acceptance (see the next section).
  - The same tasks were not used, so the comparison is confounded.
- Whole-run tok/s is always 10–40 % below steady decode because of tail slots, prefill and evictions. Ranking setups by decode lines alone overstates differences.

### Gaps

- **No paired run:** there is no same-task, same-prompt, same-length vLLM vs SGLang run.
  - The E11-04 report itself says "no paired vLLM FP8 run on this same workload" [E114REP "Limits"].
- **SGLang at >12 running:** never measured for W1. The cap is 12–13 by design. A (no spec) reached 22 running → 440 tok/s at short context [D].
- **vLLM above 16 running:** never run [JLB L176].
- **S1-60K (LB 34.58):** no engine metrics were recorded. The commit was cancelled [JDL L162 per sub-agent]. Its engine is the R5 engine plus weight prefetch [E10R L215, L234-240], so R5/D66 numbers are the closest proxy.
- **vLLM by context:** I did not bin vLLM throughput by context length myself; only the journal statement [JLB L178] exists.

---

## Speculative decoding: MTP and DFlash2 acceptance vs concurrency, temperature and context

### Takeaway

**vLLM MTP-3** gives a mean acceptance length of about 3.0 tokens/step on DIRECT (T 0.6), about 2.67 on Poetiq (T 1.0) and about 3.45 at low effort. It is essentially flat across 5–16 running.

**SGLang DFlash2** (block 8) gives about 4.3–5.2 in short sweeps and in the Duck runs. In Poetiq production, the server-log median falls to about 3.8–4.5, decreasing with context (4.43 → 3.77). It holds at 12 running.

SGLang NEXTN/MTP (3 steps, 4 draft tokens) on INT4 gave about 3.1.

### Cited Findings

**vLLM MTP-3** [M]
- Acceptance rate (accepted ÷ draft tokens):
  - run 7 v1.4.4: 0.661, with 704,499 of 1,065,303 accepted [R7RR];
  - ORDER2: 0.669 [O2RR];
  - E9: 0.655 [E9G];
  - E6: 0.664 [E6A];
  - run 3: 58.8 %; run 4: 65.7 % [E1r3; E1r4];
  - Exp 4 agent run 4: 0.736 [E4R L42].
- **Poetiq T 1.0** [D] (whole-run mean acceptance length = 1 + accepted/drafts):
  - R5: 2.671, rate 0.557, per position [0.734, 0.537, 0.400] [R5MET];
  - D66: 2.673, [0.735, 0.538, 0.401];
  - L66: 2.670;
  - R4: 2.677.
  - Matches the docs: "T = 1.0 lowers MTP acceptance to 0.557 (2.67 tokens per step instead of 3.0)" [JLB L182].
- **Low effort (W3H)** [D]: 3.446, rate 0.815, per position [0.91, 0.813, 0.723] [W3HMET].
- **DIRECT per position** [M]: 0.82 / 0.66 / 0.53, about 3.0 tokens/step [E10T L73].
- **Speed link** [M]: "3.01 vs 2.67 tokens/step corresponds to 349 vs 298 tok/s" (JIM L28 per sub-agent).
- **Same engine, R5 T 1.0 vs ORDER2 T 0.6** [M]: acceptance 0.549 vs 0.664, tokens/step 2.67 vs 3.01, decode 301 vs 351 tok/s, "confounded by task type" [MEM-P L196].
- **Vs concurrency** [D] (ORDER2 10 s-line medians of "Mean acceptance length"):
  - 2.75 at 1, 2.81 at 2, 3.11 at 3, 3.05 at 5, 3.01 at 6, 3.07 at 7, 2.94 at 10, 3.16 at 11, 3.04 at 12, 2.98 at 15, 3.01 at 16 [O2LOG].
  - R5: 2.66 at both 15 and 16 [R5LOG].
- **Duck at T 0.7** [D]: E12-V vLLM SMOKE1 3.39 at 1 and 3.45 at 2 running [E12VLOG]. Per-position up to 0.948 / 0.858 / 0.761 in single 10 s windows [E12VLOG L281]. The memory note says "MTP accept 3.5 vs DFlash 5.2" [MEM-P L246].

**SGLang NEXTN (B, INT4 target, 3 steps / topk 1 / 4 draft tokens, unquant)** [M]
- Accept length: 3.17 (1), 3.10 (4), 3.09 (8), 3.11 (12, capped at 10 running), 3.17 (16), 3.12 (24), 3.10 (32) [E113REP B table].
- Long run 3.02 [E113REP].

**SGLang DFlash2 (block 8, `--speculative-draft-model-quantization unquant`)** [M] unless marked
- **E11-03 C (fp32 SSM):** 3.89 (1), 4.37 (4), 4.74 (8), 4.70 (12), 4.95 (16), 4.75 (24), 4.76 (32). Above 6 requests these are really ≤6 running. Long run 4.20 [E113REP].
- **E11-03 D (fp16 SSM):** 4.31 (1), 4.51 (4), 4.85 (8), 4.62 (12), 4.78 (16) [E113REP].
- **E11-04 sweep:** W1 4.50 / 4.49 / 4.67 / 4.51 at 1 / 4 / 8 / 12; W2 4.61 / 4.74 / 4.55 / 4.79; W3 4.15 / 4.56 / 4.79 / 4.67 [E114REP].
- **Sanity puzzles:** per-puzzle 3.77–5.50 for C [E113REP] and 3.78–5.13 for W1 [E114REP].
- **Temperature** (C, capped at 6 running): T 1.0 285.8 tok/s with accept 4.34; T 0.6 241.7 tok/s with accept 4.60. "T0.6/T1.0 0.846" [E113REP]; called inconclusive in [E113A L72].
- **Long context:** 4 × 85.5k gave accept 4.58 [E113REP].
- **Production Poetiq, E11-05:** two definitions disagree.
  - The server decode-line median is **3.8** [E115SUM]; it is 3.82 at 11 running [D, E115LOG].
  - The per-request median of `spec_accept_length` is 4.554 over 174 requests [D, E115REQ]; the doc's "Accept length median 4.55" [E115A L34] is this per-request figure. It includes short phase-2 answers, which accept 5.6+.
- **Production Poetiq, E11-06:** median 4.5 [E116SUM]; 4.34 at 12 running and 4.82 at 9 [D, E116LOG].
- **Over context:** "DFlash acceptance only falls 4.43 → 3.77 over context, so no collapse" [JWK L46].
- **Duck runs** [D, decode-line medians]:
  - EVAL30: 5.05–5.11 at 10–12 running [E12LOG-EV30];
  - EVAL6: 5.5 at 6 [E12LOG-EV6];
  - EVAL6R5: 4.33 at 6;
  - LOW: 4.75 at 6;
  - G4B: 4.34 at 11 [G4BLOG].
  - SMOKE1: 5.1 at 2 [E12LOG-SM]; doc "DFlash acceptance median 5.2" [E12SMA L29]. G4B doc "acceptance 4.2" [E12G4BA L101].
- **Mandatory flag:** `--speculative-draft-model-quantization unquant`. Otherwise the drafter inherits INT4 and acceptance drops to ~1.0 (issue #39087) [MEM-SG "SGLang flags verified"]. All our DFlash2 launches used it [E11R L118].

### Inferences

- DFlash2 roughly doubles single-stream speed on INT4: 107.5 vs 45.6 tok/s.
- At 12 running it still gives +39 % over no-spec: 404 vs 289 [E113A L55].
- So "speculation only helps below ~8 concurrent" (the HyperQwen claim cited in JWK L129) does not hold for this hybrid model on L4.
- The production accept length (~3.8–4.5) is lower than the sweep (~4.5–4.9). That is consistent with T 1.0 Poetiq sampling and long contexts, but the two were not separated.
- vLLM MTP acceptance looks concurrency-independent and sampling-dependent. The main lever for vLLM throughput is temperature/effort, not concurrency above about 12.

### Gaps

- **DFlash2 vs MTP on the same engine and workload at 12 real running:** never measured. B was capped at 10 and C at 6 in E11-03.
- **No DFlash2 drafter quantization variants or other drafters:** no other block sizes (only 8) were tested.
- **Accept-length definition:** whether SGLang's "accept len" includes the bonus token was not checked in the source. vLLM's "Mean acceptance length" includes it (1 + accepted/drafts). Comparing the two numbers directly assumes the same definition.

---

## Memory: KV capacity, max running requests, mamba/SSM limits and launch flags

### Takeaway

**vLLM FP8 at gpu_mem 0.90** gives a 1,353,600-token KV pool. The log line "GPU KV cache size: 337,600" is per group, about 1/4 of the pool. That holds 16 × 60–86k, but Poetiq at T 1.0 reaches 99.5 % KV with 9 preemptions, and D66 at 16 × 77.8k reached 99.88 % with 66 preemptions.

**SGLang INT4 + DFlash2 under auto memory** (mem_fraction_static 0.62) gave only 387–395k KV. The mamba-state cache silently capped running requests (B 10, C 6, D 13), because speculation needs 5 state slots per request.

**The W1 fix** gives 12 real running and 881,336 KV tokens (866,744 with server max-running 13), with about 1.84–2.2 GB free per GPU. The fix: `extra_buffer_lazy` (4 slots per request), mamba cache 56, fp16 SSM, no prefill CUDA graph, mem 0.83.

### Cited Findings

**SGLang launch caps and memory from the server logs** [M]:

| Config | mem_fraction | SSM dtype / strategy | mamba cache (logged) | Running cap | KV tokens | Avail after graphs | Source |
|---|---|---|---|---|---|---|---|
| E11-03 A INT4 | 0.6187 (auto) | fp32 / extra_buffer | 113, ssm 4.01 GB | **22** ("capped to 22 … 5 state slots per request") | 604,274 | 5.55 GB | [E113LOG-A L91-102, L284] |
| E11-03 B +NEXTN | 0.6187 | fp32 / extra_buffer | 50, ssm 1.79 + intermediate 1.55 GB | **10** | 474,203 | 6.14 GB | [E113LOG-B L113-128, L401] |
| E11-03 C +DFlash2 | 0.6187 | fp32 / extra_buffer | 33, ssm 1.20 + intermediate 1.97 GB | **6** | 394,559 | 4.65 GB | [E113LOG-C L114-133, L314] |
| E11-03 D +DFlash2 | 0.6187 | **fp16** / extra_buffer | 68, ssm 1.21 + intermediate 1.97 GB | **13** | 386,563 | 3.98 GB (4.39 after draft graph) | [E113LOG-D L113-133, L338-343] |
| E11-04 W1 | 0.83 | fp16 / extra_buffer_lazy | 56, ssm 1.00 + intermediate 1.83 GB | 12 (flag) | **881,336** (target K+V 6.72 GB + draft 2.10 GB) | 1.84 GB (2.22 after draft graph) | [E114LOG-W1 L115-128, L215-224; E114REP caps table] |
| E11-04 W2 | 0.83 | bf16 / lazy | 56 | 12 | 881,336 | 1.84 GB | [E114LOG-W2 L113-214] |
| E11-04 W3 | 0.83 | fp32 / lazy | 56, ssm 2.00 + intermediate 3.66 GB | 12 | 598,712 | 1.82 GB | [E114LOG-W3 L116-216] |
| E11-05 production | 0.83 | fp16 / lazy | 56 | 12 | 881,336 | 1.84 GB | [E115LOG L116-218] |
| E11-06 and all E12 Duck runs | 0.83 | fp16 / lazy | 56 (intermediate 1.97 GB) | **13** (server) / client 12 | **866,744** | 1.88 GB | [E116LOG L113-217; E12LOG-EV30 L226; G4BLOG L228] |

**Common SGLang flags** [M] [E11R L123-128; server_args at L8 of each log]:
- `--tp-size 4 --kv-cache-dtype fp8_e4m3 --chunked-prefill-size 4096` (max_prefill_tokens 16384), context 155,648 in the logs (README planned 106,496);
- `--enable-metrics --disable-flashinfer-autotune`, attention backend flashinfer;
- linear-attention kernels triton (decode / prefill / verify) [E113LOG-D];
- DFLASH `speculative_num_draft_tokens=8`, `speculative_num_steps=1`;
- W1 adds `--mamba-radix-cache-strategy extra_buffer_lazy --max-mamba-cache-size 56 --mamba-ssm-dtype float16 --disable-prefill-cuda-graph --mem-fraction-static 0.83 --max-running-requests 12` (13 from rev 3c).
- CUDA graphs: target-verify and draft-verify graphs captured for bs [1, 2, 3, 4, 5, 6, 7, 8, 10, 12] [E114LOG-W1 L149]; the prefill graph is disabled in W1.

**Other memory facts** [M]:
- **State slots per request by strategy:** radix off 1, no_buffer 3, extra_buffer + overlap 5, extra_buffer_lazy 4 [E113A L96-105, from source `kv_cache_configurator.py:2232-2261`].
- **D memory per GPU:** target weights 4.62 GB, drafter 1.49, mamba 3.26, target KV 2.94, draft KV 0.92, prefill graph 2.60, target verify graph 0.50, draft verify graph 0.38, **free 4.39** [E113A L107-119].
- **`/get_server_info` is misleading:** it still reports max_running_requests 32 when the real cap is lower; the cap shows only in the log warning [E113A L121].
- **11-of-12 admission bug:** in E11-05, 2,358 decode lines showed 11 running + 1 queued [E115A L36].
  - Root cause, verified in the sgl 0.5.21 source: a chunked-prefill request is counted twice in admission (`scheduler.py:3971-3976`), so `batch_is_full` latches.
  - Upstream fix PR #44401 (after 0.5.21). Workaround: server `--max-running-requests 13` with client 12 [MEM-P L231].
  - Cost "about 8 %" [E115A L40; JWK L47].
- **Queue = 1 is SGLang's steady state at 12 in flight:** E12 EVAL30 had 3,127 of 4,153 lines with queue 1 [MEM verify-real-concurrency note]; G4B had 188 lines with queue 1, 101 of them at 11 running [E12G4BA L42].
- **Production KV need:** "Production needs about 920k KV tokens: S1-60K peaked at 68 % of 1.35M with 12 x 60k" [E113A L78].
- **S49 fit:** W1's 881k fits S49: 12 × (≤10k + 49k + 8k) ≈ 800k [E114A L8-10].
- **Duck KV thrash:** EVAL30 ran 30 time-sliced branches × 50–100k contexts against ~870k KV [E12EV30A L36-44].
  - 144 of 423 non-first requests lost their prefix.
  - Those requests account for 8.39 M of the 9.60 M uncached prefill tokens.
  - One retraction: "KV cache pool is full. Retract requests. #retracted_reqs: 1" [E12LOG-EV30 L3505].
- **Rev 5 sizing:** 12 streams × 58k = 697k fits 866,744 at the median, and p90 would be 890k [E12R5A L70-71].

**vLLM FP8 MTP-3** [M]:
- **Pool and log lines** (TP0 "Available KV cache memory: 10.98 GiB" at `--gpu-memory-utilization 0.90`, equivalent to 0.8693 without CUDA-graph profiling, `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`):
  - "GPU KV cache size: 337,600 tokens"; "Maximum concurrency for 106,496 tokens per request: 10.32x" [R5LOG L1008-1014];
  - "…81,920 tokens: 12.63x" [E4LOG L224-229];
  - "…135,168 tokens: 8.46x" [E12VLOG L216-222].
  - The run report records `kv_tokens` 1,353,600 [R7RR; O2RR]. The real pool is 846 blocks × 1,600 = 1,353,600 tokens, and the 337,600 log line is 1/4 of it [E10C L9-10; JDL L135 per sub-agent].
- **Weights:** "Model loading took 7.15 GiB memory" per GPU [E4LOG L148; R5LOG L854].
- **CUDA graphs:** "CUDAGraphMode.FULL_AND_PIECEWISE is not supported with spec-decode for attention backend FlashInferBackend … setting cudagraph_mode=PIECEWISE"; graph capture 7 s, 0.61 GiB [E4LOG L204-232; R5LOG L969-1049].
- **Fixed cost per running request at MTP k=3:** 19.2k token-equivalents from source, 21.2k from the measured fit [E10C L12-13].
- **KV peak and preemptions:**

  | Run | KV peak | Preemptions | Notes |
  |---|---|---|---|
  | run 7 | 78.7 % | 0 | [E1V] |
  | ORDER2 | 87.3 % | 0 | p95 79.5 %, mean 68.6 % [E10O L27, L173-181] |
  | E9 | 91.4 % | 0 | [E9G] |
  | R5 | 0.9953 [D] | 9 | [R5MET]; prefix cache evicted, 4.18 M prefill vs 1.14 M for ORDER2 [JLB L183] |
  | D66 | 0.9988 [D] | 66 | [D66MET] |
  | L66 | 0.9008 [D] | 0 | 12 slots at max_model_len 155,648 [L66MET] |
  | Exp 4 agent | 31 % | 0 | [E4R L42] |

- **Projections** [P]: 18 slots ≈ 98 % KV and 20 ≈ 109 % (ORDER2) [E10O L176]; 20 × 57k ≈ 98 % [E7R L62]. INT4 on vLLM would free about 2.6–2.8 GB/GPU, about +320k KV [E10W L42].
- **Flags** [R5LOG L548]: `kv_cache_dtype fp8_e4m3`, `max_model_len 106496`, `tensor_parallel_size 4`, `enable_prefix_caching True`, `max_num_batched_tokens 4096`, `max_num_seqs 16`, `disable_custom_all_reduce True`, `language_model_only True`, `speculative_config {method mtp, num_speculative_tokens 3}`, attention backend FLASHINFER with `disable_flashinfer_prefill=True`.
  - run 7 used max_model_len 81,920 and gpu_mem 0.9 [R7RR config]; L66 used 155,648 [E10R L302-305].
  - Prefix caching uses mamba `align` mode [E1R L24].

### Inferences

- SGLang's KV advantage comes from INT4 weights (4.62 GB vs 7.15 GiB per GPU) plus explicit mamba sizing. Without explicit flags its auto settings waste about 19 % of memory and cap concurrency silently.
- vLLM's 1.35M-token pool is larger than SGLang W1's 881k, because SGLang reserves memory for DFlash2 draft KV (2.10 GB), intermediate SSM state for speculative verify (1.83–1.97 GB) and the drafter weights (1.49 GB).
- At the 60k-per-request workload, both engines are KV-limited to roughly 12–16 streams.

### Gaps

- **Runtime FP8 KV scales:** whether the INT4 checkpoint's 32 calibrated k/v scales (0.021–0.254) are used at runtime is unverified. Both engines log "no scaling factors… 1.0" [E114A L26; E113A L43].
- **Untested arms:**
  - ReplaySSM (`--enable-linear-replayssm-spec`) would free ~1.8 GB. It was dropped because "DFLASH + ReplaySSM is KDA-only" [E11R L208-209].
  - The 2 × TP2 replica arm (custom all-reduce allowed) was proposed but never run [E113A L89].
- **vLLM tuning:** vLLM at mem 0.95, MTP k=2 or fp16 SSM was never measured (all listed only as levers in [E10C]).

---

## Startup: launch and load times

### Takeaway

**SGLang W1 cold start on Kaggle** is about 520–680 s ready. Most of it is the target-verify CUDA graph capture (about 372–385 s), plus 13–88 s weight load. One slow-disk run took 1,197 s because the weight load took 670 s.

**vLLM FP8 MTP-3** cold start is about 500–790 s to scheduler: engine init about 291–295 s plus weight load. Weight prefetch cut the weight load from about 63 s to about 3–5 s.

### Cited Findings

**SGLang** (Engine startup timings line in each log) [M]:
- **E11-03 A (first launch):** ready in 601 s; prefill CUDA graph 447.08 s; load_weight 74.48 s; scheduler_e2e 553.29 s [E113REP; E113LOG-A L251-307].
- **E11-03 later launches** in the same session: B 166 s (load 7.12, prefill graph 68.28), C 161 s, D 146 s, X/S 136 s [E113REP; E113LOG-B L410; E113LOG-D L353]. The doc: "first launch 601 s (prefill-graph JIT 447 s), later launches 136–166 s" [E113A L73].
- **E11-04:**
  - W1 ready in 567 s: load_weight 87.36 s, target_verify graph 373.80 s, scheduler_e2e 514.11 s [E114REP; E114LOG-W1 L230].
  - W2 71 s (target_verify 11.94 s) and W3 76 s, both cached [E114LOG-W2 L225; E114REP].
  - The doc: "first launch took 567 s (JIT); later ones about 75 s" [E114A L3].
- **E11-05:** EVAL30 ready in 532 s and S1COMMIT4 in 522 s [E115A L12; E115K L137]. load_weight 13.41 s, target_verify 373.40 s, tokenizer_e2e 458.86 s [E115LOG L230].
- **E11-06:** load 13.48 s, target_verify 372.55 s, tokenizer_e2e 450.23 s [E116LOG L229].
- **E12 Duck runs:**
  - SMOKE1: "W1 up in 677 s" [E12SMA L27]; log load 13.84 s, verify 371.27 s [E12LOG-SM L229].
  - EVAL30: verify 384.53 s [E12LOG-EV30 L229]; EVAL6: load 38.17 s; LOW: load 59.38 s.
  - G4B: **load_weight 669.96 s**, scheduler_e2e 1,115.52 s, "ready in 1197s" ("a slow cold start on Kaggle, not our code") [G4BLOG L231; E12G4BA L89]. Rev 5 runs: about 700 s [E12G4BA L89].
- **Install:** wheelhouse venv install 157 s (E11-02), 150 s (v2), 181 s (E11 per [E13P L93]) [E11R L62, L95].

**vLLM** [M]:
- **R5:**
  - "Loading weights took 62.59 seconds"; "Model loading took 7.15 GiB memory and 64.103997 seconds" [R5LOG L776, L854];
  - "init engine (profile, create kv cache, warmup model) took 295.10 seconds" [R5LOG L1051].
- **E12-V, with prefetch:** weights 3.11 s, model loading 4.75 s, init engine 291.59 s [E12VLOG L110, L151, L229].
- **Exp 4 run 4:** weights 40.00 s, init engine 290.98 s [E4LOG L114, L236]; server 510 s [E4R L42].
- **run 7 v1.4.4:** engine load_sec 610.8, setup_to_scheduler 789.0 s [R7RR].
- **ORDER2:** load 567.1 s, setup 700.0 s [O2RR].
- **E6:** setup 705 s [E6A]; **E9:** setup 701 s [E9G].
- **Run 1:** setup to scheduler 935 s [E1r1 per sub-agent].
- **Weight prefetch:** "weight load fell from 64 s to 4.8 s, and engine start from 600 s to 532 s" [JDL L136; MEM d66 note].
- **Journal:** "About 12–13 min of setup" [JLB L56 per sub-agent].

### Inferences

- On a cold SGLang W1 launch, about 70 % of startup is the 8-token target-verify CUDA graph capture/JIT (~372–385 s). A warm kernel cache (same session) brings it to ~12 s.
- A persisted Triton/FlashInfer cache could save about 6 min per run. This is an inference; it was never tested across Kaggle sessions.
- Both engines cost about 9–13 min of a 12 h run, so startup is a ~1.5 % effect. Disk speed variance (G4B 670 s load; E3 0.29 vs 0.65 GB/s) is the bigger risk.

### Gaps

- No run measured the SGLang startup with a pre-built JIT cache shipped as a dataset.
- vLLM startup breakdown beyond "init engine ~291–295 s" (torch.compile vs graph capture) was not extracted.

---

## Failures: what broke and why

### Takeaway

**vLLM failures** were mostly early integration problems: the libcuda link for the FlashInfer JIT, an MTP drafter OOM at 32k batched tokens, the thinking budget being ignored with MTP, HTTP 500/400 on tool calls, and preemptions at ≥99.5 % KV.

**SGLang failures** were silent capacity caps (mamba cache), an 11-of-12 admission bug, KV thrash in the Duck agent, and a controller that mistook SGLang's normal queue = 1 for pressure.

Custom all-reduce is disabled on both engines for TP4 over PCIe.

### Cited Findings

**vLLM** [M], all per sub-agent unless a log line is given:
- **run 1 v1.3.2:** "EngineDeadError" after 792 s; the real cause was a FlashInfer JIT `ld: cannot find -lcuda` (Kaggle only has libcuda.so.1). Fixed with a libcuda shim [E1r1 L7-13; E1R L19].
- **run 2 v1.3.4:** OOM in the MTP drafter at `max_num_batched_tokens` 32768. Fixed with 4096 + `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1` [E1R L18, L32; E1r3 L19; MEM-P L43].
- **Thinking budget not enforced:** vLLM 0.19 with MTP ignores `thinking_token_budget` (issue #39573); 10/12 requests thought until max_tokens 57,344. Workaround `VLLM_USE_V2_MODEL_RUNNER=0` [MEM-P L93-98; E10T L78 says fixed in 0.21].
- **Exp 4 agent:**
  - HTTP 400 "Unexpected reasoning effort high" [MEM-P L64];
  - HTTP 500 `assert content is not None` on forced tool_choice (5 attempts) [E4R L50-51];
  - readiness bug (local `urllib` import) [MEM-P L72].
- **Preemptions:** R5 had 9 at ~0.995 KV [E10R L235]; D66 had 66 at 99.88 % [JDL L35-36]. 16 slots at 77.8k gave "nothing for long Poetiq" [MEM d66 note].
- **E8:** CUDA OOM at minute 124 lost 15 units; `prompt_logprobs` OOM [JLB L218, L262].
- **Prefix-cache recompute stalls (R5):** 28 stalls cost about 38 min (JIM L25).
- **Kaggle:** "system error" on v1.4.2/3 [E1V L38].
- **Custom all-reduce:** "Custom allreduce is disabled because it's not supported on more than two PCIe-only GPUs" [E4LOG L55-58].
- **Graph mode:** FlashInfer + spec-decode forces cudagraph_mode=PIECEWISE [E4LOG L204].
- **E12-V vLLM SMOKE1 run 1:** every Duck request got 404 "model `pending`" (served-name alias); fixed in run 2 [MEM-P L244].

**SGLang** [M]:
- **Mamba-cache caps** (A 22, B 10, C 6, D 13) made the E11-03 sweep curve flat. "Every sweep 'level' above the cap ran as cap-many requests plus a queue" [E113A L8-25; logs as in the Memory section].
  - The harness bug was that it did not assert real running == level [E113A L79].
- **11-of-12 admission bug** [E115A L36-40; MEM-P L231]. E11-04's sweep at 12 also ran 11 real ("FAIL") [E114REP sweep table].
- **Custom all-reduce:** "CustomAllReduceV2 is disabled because it's not supported on more than two PCIe-only GPUs"; TP4 uses NCCL at 64 MB in 20.9 ms [E113LOG-A L34-41; E113REP preflight].
- **FP8 KV scale warning:** "Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0" [E113LOG-A L75].
- **Benign log noise:** `ModuleNotFoundError: No module named 'wrapt'` in sitecustomize, repeated [E113LOG-A L1-20].
- **E11-04 capacity gate:** missed 900k by 2 % (881,336), so PROD was cut to 10 min and long context was "NOT PROVEN" [E114A L8-21].
- **E11-04 long-prefix contcache:** 0 cached on aborted samples, a test artifact [E114A L31]. E11-05 shows that phase 2 hits the cache 52–99.7 % when the shim waits for the stream end [E115A L44-46].
- **E12 EVAL30 KV thrash:** 8.39 M re-prefill tokens; effective 187 tok/s vs ~420 decode [E12EV30A L36-44]; one retraction [E12LOG-EV30 L3505].
- **G4B rev 6d:** the probe stepped 12 → 6 on queue = 1, costing −13 % [E12G4BA L42-49].
- **Toolchain issues found while building** [E11R L38-42, L74-91; MEM-SG]:
  - SGLang wheel pins `cuda-tile==1.6.0rc5` (sdist only);
  - `flashinfer-cubin 0.6.18` does not exist;
  - Kaggle drops '+' from wheel file names;
  - the pip CUDA runtime lacks `libcudart.so` / `lib64`, so a symlink shim is needed;
  - fresh kernel caches are required because a stale cache gave PPL 9.98 vs 6.05 (issue #42917).

### Inferences

- Every SGLang throughput claim needs the startup log's capped line and a check that the real running count equals the requested level. Our E11-03 numbers above the caps were invalid until re-read.
- vLLM's binding failure mode at our workload is KV saturation with preemption above ~16 × 77.8k. SGLang's is admission and prefix eviction when the agent holds more contexts than fit in KV.

### Gaps

- No run tested SGLang with a version that includes the PR #44401 admission fix.
- No run measured TP2 × 2 (where custom all-reduce works) on either engine.

---

## Accuracy: FP8 vs INT4, drafters, and LB per engine

### Takeaway

No clean accuracy comparison between FP8 and INT4 exists.
- **Small sanity sets:** INT4 + DFlash2 matched or beat the FP8 expectation: 7/8 (C), 8/8 (W1) and 7/8 (W2), vs 6.74.
- **Eval:** E11-05 (49k) scored below R5 (77.8k) on the same 30 tasks, and the cap was the main suspect.
- **LB:** the SGLang INT4 submission E11-06 P1-60K scored **29.17**, against S1-60K vLLM FP8 **34.58**. That pair is confounded with a prompt change (P1) and is within about 3–4 sd of LB noise.
- **Greedy drift:** spec vs no-spec and fp16 vs fp32 SSM drift similarly, after about 150–166 of 512 tokens.

### Cited Findings

**Sanity solves** (8 easy puzzles, E10 DIRECT at T 0.6; FP8 expectation 6.74) [M]:
- C (fp32 SSM, capped at 6 running): 7/8 [E113REP; E113A L61].
- W1: **8/8**; W2: 7/8 [E114REP; E114A L24].
- All closed naturally. 136b0064 thought 44.7k on INT4 C vs 23.2k on FP8 [E113A L61-63].
- "These are 8 easy tier-A puzzles, so this is no proof of eval parity" [E113A L63].

**Greedy drift** (common prefix of 512 tokens) [M]:
- B vs A 152; C vs A 155; D (fp16) vs C (fp32) 166 [E113A L64].
- W2 vs W1 41.5; W3 vs W1 44.0 [E114A L27].

**Contamination tests** [M]:
- contam: common prefix 12–22, 0/4 identical. This is batch non-determinism, not cross-request leakage [E113A L65-67].
- W1 marker test PASS, 8/8 own codes [E114REP].

**Eval on the same tasks** [M]:
- E11-05 SGLang INT4 at 49,152 thinking: **11/48, task-norm 23.3 %**. R5 vLLM FP8 at 77,824 on the same 30 tasks: **16/48, 33.3 %** [E115A L58-73].
  - 83 of 88 iterations were forced closes; 7 of 15 train-verified programs were wrong on test (R5: 3 of 13).
  - "Not separable here: the 49k-vs-77.8k cap effect from any INT4-vs-FP8 effect. No vLLM run at 49k exists on eval" [E115A L73].
- E11-06 P1-60K on SGLang: 7/22 queries, 5/14 tasks (35.7 %). S49 (49k, old prompt, SGLang) on the same 14 tasks: 4/14 (28.6 %). "+1 task… sd ≈ 1.7 tasks… not significant" [E116A L11-27].

**LB per engine** [M]:

| Submission | Engine | Thinking cap | LB | Source |
|---|---|---|---|---|
| S1-60K (foysal Version 2) | vLLM FP8 + MTP-3 | 60k | **34.58** | [JDL L263-267] |
| S49 | vLLM | 49k | 32.08 | [MEM-P L226; MEM-T L3] |
| S1-55K | vLLM | 55k | 31.81 | [MEM-P L266] |
| **E11-06 P1-60K** (ref 57372299) | SGLang INT4 + DFlash2 W1, new P1 prompts | 60k | **29.17** | "SGLang engine + P1 prompts did not transfer to LB (cause not separated)" [MEM-P L249] |
| R5 | vLLM | 77.8k | 21.39 | [JLB L24] |
| v1.4.4 | vLLM | — | 26.94 | [JLB L15-30] |
| E7 | vLLM | — | 27.92 | [JLB L15-30] |
| E9 | vLLM | — | 22.92 | [JLB L15-30] |

- **LB noise:** "identical notebooks submitted repeatedly scored an sd of 1.0–1.65 LB points" [JLB section 1].
- **RedHat INT4 model card vs BF16:** GPQA Diamond 89.23 → 87.88 (−1.35); AIME 2025 95.42 → 94.17 (−1.25) [E10W L47-48; MEM-SG].
- **Duck engine swap:** vLLM E12-V SMOKE1 run 2 and SGLang SMOKE1 both solved 2ba387bc with both branches. vLLM was slower: 301/455 s, 13.5k/22k gen tokens, vs SGLang 60–73 s, 5–6k [MEM-P L246].

### Inferences

- The SGLang LB result (29.17 vs 34.58, about −5.4) is 3–5× the measured noise sd, so it is probably a real drop. But it bundles two changes: engine/quantization and the P1 prompts.
- E11-06 on EVAL14 was +1 task vs the old prompt on SGLang, so the prompt alone did not show a loss on eval.
- An INT4 or engine effect on hidden-set accuracy is therefore plausible but unproven.
- The throughput gain (+17–30 %) did not translate into LB gain.

### Gaps

- **Missing paired run:** no paired FP8-vLLM vs INT4-SGLang run with the same prompt, cap and tasks (EVAL30 or EVAL120).
  - Stage 3 "paired against FP8 ORDER2/R5 on EVAL120" was planned [E11R L140] but never run.
- **Target-only parity:** PARITY/PARITYR6 notebooks are mentioned in memory as built (shas f3876b40 / 85da1be3 / bc9e2bc2). I found no run results for them.
- **No other drafters:** no comparison of DFlash2 vs NEXTN vs no-spec on eval accuracy (only greedy-drift prefixes).
- **No LB with the old prompt on SGLang:** no LB test of SGLang with the old (S1) prompts exists, so engine vs prompt cannot be separated.

---

## Flash-Next: what we measured on L4 (Experiment 3) and what is only planned (Experiment 13)

### Takeaway

Qwen3.8-Flash-Next never produced a valid output or throughput number on Kaggle 4x L4. Stack: Pennyroyal SGLang 0.5.19, Intel W4A16 AutoRound + FP8 PLE overlay, TP4/EP4.
- The best run (FN-PROBE v4.2.1, 2026-10-10/11) was ready at 862 s with 118,080 bf16 KV tokens.
- It decoded single-stream at 5.95 tok/s and output "!!!!" garbage.
- Experiment 3 is closed as "not viable". Experiment 13 (Flash-Next in Duck) and Experiment 12 FP8-kernel tuning were built but never run on a GPU.
- Our only real Flash-Next serving numbers are on an RTX PRO 6000 in the ARC-AGI-3 track: vLLM NVFP4, about 235–241 tok/s with only about 3 running.

### Cited Findings

**FN-PROBE v4.2.1** (esfersami50 `fn-probe-flashnext-4xl4` v1) [M]:
- load_weight 344.81–344.94 s (19.06 GB per rank, 2.55 GB free after load) [E3LOG421 L101-104].
- KV bf16 "#tokens 118080" [E3LOG421 L121-125]; max_running_requests 8, context 16,384 [E3LOG421 L143].
- Ready = 862.2 s [E3REP421 L113]. Warm-up (Triton/TileLang JIT) about 7.4 min. Free memory fell to 0.66 GiB.
- Decode at 1 running, `cuda graph: False`, gen throughput **5.95 tok/s** [E3LOG421 L577].
- Sanity FAIL: 96 "!" characters [E3REP421 L122-127].
- TileLang "Data race detected: Logits(bx, position)" [E3LOG421 L447-494].
- Verdict "THROUGHPUT: NO VALID COMPLETE MEASUREMENT" [E3NB421 L105].
- README: "not viable on 4×L4 with this stack… Not competitive with the 27B (about 373 tok/s at 16 running)" [E3R L3, L18-25].
- The probe ran eager with a PyTorch FA2 shim, so it is not a ceiling. The garbage cause was never isolated: shim, TileLang race, Marlin+EP4 on sm89, FP8 PLE scales, bf16 SSM, or TP4 over PCIe (`_scratch/e13_zip/reports/E3_flashnext_4xL4_failure_report.md` L88-94 per sub-agent).

**Earlier E3 runs** [M]:
- **v3.3** (2026-10-05): load 257.67–402.57 s, KV 19,968 / 15,424 tokens. Crashed in the warm-up decode with FA4 CuTe "MLIRError: unable to compute crd2idx" (the SM89 FA4 crash) [E3LOG33 L97-553, L1285-1575].
- **v3 / v3.2:** `--language-model-only` unsupported; MoE runner assert (needs `auto`).
- **v4** (2026-10-10): both launches timed out at 1,080 s still loading. The disk read at 0.29 GB/s vs 0.58–0.65 on 2026-10-05.

**Experiment 12 FP8 kernel tuning:** never run, with 0 notebook outputs.
- The only measured fact: the R5 log printed "Using default W8A8 Block FP8 kernel config" for the 5 TP4 shapes 8704×5120, 5120×4352, 5120×1536, 4096×5120 and 3584×5120 [E12FP8R L7-15].
- FP8 Marlin is disabled on sm89 unless `VLLM_TEST_FORCE_FP8_MARLIN=1` [E12FP8R L20].

**Experiment 13:** never run on a GPU.
- Plan: SGLang 0.5.21 + Franzen patch #3, Intel W4A16 AutoRound with BF16 PLE, TP4/EP4, classic FA2 for QSA decode, Triton QSA decode only as a fallback, fp8 KV, mem 0.93, mamba cache = running × 5 [E13P L1-30, L80].
- Estimates [P]:
  - KV about 170–250k fp8 tokens [E13P L97];
  - target-only 80–170k = 1–3 resident Duck contexts [E13P L119];
  - 700 tok/s aggregate is "a target, not a gate" [E13P L39, L240];
  - 420 tok/s (27B W1 at 11–12 running) is the comparison threshold [E13P L120].

**ARC-AGI-3 Flash-Next** (RTX PRO 6000, vLLM NVFP4, TP1, 32k context, 5 GiB KV, MTP 3; not L4) [M]:
- GPU KV 105,202 tokens, "maximum concurrency for 32,768 tokens per request: 3.21x"; ready 535 s and 455 s.
- Final 234.76 and 241.42 generated tok/s (job wall clock).
- Sources: `resources/ARC-AGI-3 related/notebook6e705f697b (1).ipynb` cells 8 and 18 and `notebook6e705f697b1.ipynb` cell 17, per sub-agent; [MEM-A3].
- Franzen's TP1 RTX PRO 6000 runs: 588.00 / 594.25 tok/s final [E13V L12 per sub-agent].

### Inferences

- On L4 the binding problems for Flash-Next were kernels (FA4 on sm89, TileLang race, eager-only shim) and memory: 19.06 GB weights per rank left 2.55 GB.
- Raw bandwidth was not the problem, so the "~4× fewer bytes per token" argument [JWK L48-50] has never been tested on L4.
- Even if it worked, the ~118–250k KV would hold only 2–4 long contexts, against ~12 for 27B W1 [E12EV30A L77].

### Gaps

- **No valid Flash-Next numbers on L4:** no valid aggregate or multi-stream decode number, no CUDA-graph run, and no correct-output run on 4x L4.
- **E12 FP8 kernel tuning:** there are no tuned/default/Marlin FP8 GEMM benchmark numbers for vLLM on L4, because E12 FP8 tuning never ran.
