# SOURCES: where every number in WRITEUP.md and PAPER.md comes from

Compiled 2026-10-24.

**How to read this file**
- Line numbers are 1-based lines of the named file, checked with `grep -n` on 2026-10-24.
- "W §x" = the section of WRITEUP.md; "P §x" = the section of PAPER.md.
- A number that is our own arithmetic on sourced values is marked **(derived)**, with the formula.
- Placeholders are listed at the end.

## Source keys

| Key | File |
|---|---|
| MEM-P | `C:/Users/hp/.claude/projects/d--ARC-AGI-2/memory/arc-agi2-qwen27b-project.md` |
| MEM-K | `…/memory/kaggle-arc2-platform-limits.md` |
| MEM-14 | `…/memory/experiment-14-optimized-engine.md` |
| MEM-H | `…/memory/hidden-similar-to-eval.md` |
| MEM-A | `…/memory/arcprize-qwen38-effort-scores.md` |
| MEM-C | `…/memory/coverage-levers-early-stop-ordering.md` |
| MEM-D | `…/memory/d66-l66-hard-task-findings.md` |
| MEM-S | `…/memory/sglang-int4-dflash2-plan.md` |
| DEEP | `D:/ARC AGI 2/resources/Analysis and Journals/Qwen3.8-27B LB and run deep analysis (2026-10-16).md` |
| WEAK | `D:/ARC AGI 2/resources/Analysis and Journals/Weakness analysis vs top teams (2026-10-19).md` |
| E1 … E9 | `D:/ARC AGI 2/experiments/Experiment N - …/README.md` (Experiment 12 FP8 has its own README; not cited) |
| E10 | `…/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/README.md` |
| R5A | `…/Experiment 10 - …/Experiment Results/Analysis/GPU EVAL120 Poetiq R5 analysis (2026-10-15).md` |
| E11 | `…/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/README.md` |
| E11-03A / -04A / -05A / -06A | `…/Experiment 11 - …/Experiment Results/E11-03 run1 analysis (2026-10-17).md`, `E11-04 run1 analysis (2026-10-18).md`, `E11-05 run1 analysis (2026-10-18).md`, `E11-06 P1-60K EVAL14 run1 analysis (2026-10-19).md` |
| E12P | `…/Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)/PLAN.md` |
| E12-30 / E12-6 / E12-R5 / E12-G4B | `…/Experiment 12 - …/Experiment Results/E12 SGLang EVAL30 run1 analysis (2026-10-19).md`, `E12 SGLang EVAL6 run1 analysis (2026-10-20).md`, `E12 rev 5 runs analysis (2026-10-21).md`, `E12 rev 6d G4B GPU run analysis (2026-10-21).md` |
| N11 … N37 | `…/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/<NN> ….md` (the note with that number) |
| E14S-RR | `…/Experiment 14 - …/Experiment Results/Kaggle Runs/E14S SUBMIT commit1 (hosen42, 2026-10-24)/q27x/e12_run_report.json` |
| E14S-LOG | same folder, `q27x/sglang_server_try0.log` |
| E14S-RES | same folder, `e14_results.json` |
| S55 | `D:/ARC AGI 2/_scratch/s1_55k/make55.py` |
| REV-A / REV-B / REV-C / REV-D | `D:/ARC AGI 2/_scratch/review_4day/A_conversation.md`, `B_notes.md`, `C_notebooks.md`, `D_resources.md` |
| FR | `D:/ARC AGI 2/_scratch/ext/dafr/WRITEUP.md` (Daniel Franzen) |
| LAND | `D:/ARC AGI 2/resources/Research Papers/ARC AGI 2 research paper.pdf` (text extracted with PyMuPDF; `pypdf` is not installed in the local Pythons) |

---

## 1. Leaderboard scores and submission facts (W §0, TL;DR; P §1, §5.1, abstract)

| Number(s) | Used in | Source |
|---|---|---|
| NVARC-derived Qwen3-4B baseline **32.64** | W intro, §0, §9; P §2, §5.1 | MEM-P L11 ("Existing production notebook is an NVARC-derived Qwen3-4B specialist at 32.64 LB"); DEEP L26 |
| PIV v1.3 **1.67** (= EVAL120 1.67 %); 8,192 per task; ref 56785200 | W §0, §6; P §5.1, §6.1 | MEM-P L26; DEEP L19 (method, cap); DEEP L236 (6,144 + 2,048) |
| v1.4 **18.75**, ref 56886033; scored 10-10 | W §0; P §5.1 | MEM-P L46; DEEP L20 |
| v1.4.4 **26.94**, ref 57032990; +8.19 / +8.2 | W TL;DR, §0, §4; P §5.1 | MEM-P L70; E1 L36, L38; DEEP L21, L33 |
| E7 ORDERING_ONLY v2 **27.92** | W §0; P §5.1 | MEM-P L168; DEEP L22 |
| E9 **22.92**, ref 57173293 | W §0, §6; P §5.1 | MEM-P L172; DEEP L23 |
| Poetiq R5 **21.39** | W TL;DR, §0; P abstract, §5.1 | MEM-P L178; DEEP L24 |
| S1-60K **34.58**; +13.19 over R5 | W TL;DR, §0; P abstract, §5.1 | MEM-P L214 |
| S49 **32.08** (49,152 + cells tie-break) | W §0, §4, §6; P §5.1 | MEM-P L226; E10 L445-446 |
| S1-55K **31.81**, 55,000 cap, submitted 10-19 | W §0, §6; P §5.1 | MEM-P L266; S55 L7 (`THINK_BUDGET … 55000`) |
| E11-06 P1-60K **29.17**, ref 57372299 | W §0, §2.2, §6; P §5.1 | MEM-P L237, L249 |
| Duck rev 5 LOW **17.50**, ref 57456562; ~56 % of reached tasks solved; reached ~31 % of 240 | W §0, §3.3; P §5.1, §5.6 | MEM-P L267, L281; N23 L16 |
| E14S submitted 2026-10-24, ref **57574781** | W §0, §2.4, §7; P §5.1, §8 | N31 L98; MEM-14 L132 |
| Public LB = exactly **120** tasks; ~**90** team submissions × 1.2 are whole or simple fractions | W §0, §1; P §2 | DEEP L26 |
| LB noise sd **1.0–1.65**; NVARC ×**10**: **27.36–32.64** | W §0; P §3 | DEEP L28-31; MEM-P L180 |
| Differences: +8.2 real; +1.0 noise; −5.0 / −6.5 ≈ 2.5–3 sd | W §0; P §5.1 | DEEP L33-36 |
| Dates 10-03 / 10-06 / 10-10 / 10-12 / 10-14 / 10-15 | W §0 | DEEP L17-24 |
| Dates 10-17 / 10-18 / 10-19 / 10-20 / 10-21 / 10-24 | W §0 | MEM-P L214, L226, L266, L249, L267 (submitted 10-20 23:58 UTC); N31 L98 |
| One pending submission per team; turnaround **12–16 h** | W §5; P §2 | MEM-K L14-15 |
| Final deadline 2026-11-02 | (context) | N29 L35 |
| Our best is "11 scored Qwen3.8-27B submissions, 1.67 to 34.58" | P abstract | **(derived)**: count of the Qwen rows above (PIV, v1.4, v1.4.4, E7, E9, R5, S1-60K, S49, S1-55K, P1-60K, Duck LOW) |

## 2. Setting, model, effort and coverage model (W §1; P §2, §3)

| Number(s) | Used in | Source |
|---|---|---|
| 4× L4, offline, 12 h | W §1; P §3 | MEM-P L11 |
| 24 GB per L4 | W §1; P §3 | N29 L15 ("our KV is 867k on the same 24 GB cards") |
| PCIe-only, PHB topology, P2P **5.5×** slower | W TL;DR, §1, §2.3; P §3, §5.7 | N22 L51 (topology "all GPU pairs PHB", P2P 1,887 µs, 5.5×); MEM-14 L28 (PCIe-only, custom all-reduce disabled) |
| Hidden test = **240** tasks (fact); public part 120 | W intro, §1; P abstract, §1, §2 | MEM-K L29-31 |
| "120 private" | W §1 | **(derived)**: 240 − 120 |
| 12 h includes container start and mounts | W §1; P §3 | DEEP L40 |
| 48 vCPUs, ~198 GB RAM | P §3 | E14S-RES `env.nproc` = 48, `env.mem_gb.total` = 197.9 |
| Effort xhigh / medium / low only; "high" raises | W §1; P §2 | DEEP L259; E4 L27 |
| xhigh / low / medium template instructions | P §2 | E10 L385-389 |
| ARC Prize semi-private: xhigh **42.4 %**, low **22.8 %**, medium **13.2 %**, none **1.5 %** | W §1, §6; P §1, §2, §5.5 | MEM-A L3, L15-18 |
| ~**60k** output tokens per request (our estimate from ARC Prize cost) | P §1 | MEM-A L15, L20 |
| LB tasks = semi-private set (organiser quote) | P §2 | WEAK L10 |
| Tool calls only after `</think>` | P §2 | DEEP L233; MEM-P L15 |
| Natural close ≤ 49k hits **72 %**; forced closes **8 %** | W §1; P §5.2 | DEEP L138-139, L299 |
| **59.7 %** of tokens to forced closes with no hit (9 runs, 32.5 M tokens) | W §1; P §5.2 | DEEP L142-144 |
| Archive budgets 3k–12k per turn (3,072; 6,144; 8,192; 12,288) | W §1, §6; P §6.1 | DEEP L230-240 |
| Natural close at ~26–45k | P §6.1 | DEEP L242 |
| Hidden ≈ eval difficulty; placeholder premise wrong | W §1; P §3 | MEM-H L13; DEEP L8-11 |
| LB/eval **1.06–1.10 / 0.62 / 0.44**; eval **25.4 % / 36.9 % / 48.6 %** | W TL;DR, §1; P abstract, §5.2 | DEEP L85-89; MEM-H L17-19 |
| 49k eval value reconstructed from the 86k traces, not measured | P §5.2 | DEEP L75, L124 |
| 86k DIRECT **534** min / 120 tasks; Poetiq **691–704** min; iteration 1 by **~540** min | W §1; P §3, §5.3 | DEEP L99; MEM-H L26-27; R5A L19, L31 |
| ~**680** scheduling minutes | W §1; P §3 | DEEP L111 |
| Speed-ups **1.54× / 2.04× / 2.55× / 3.57×**; expected LB ~37 / ~49 / ~52 / ~57; coverage 65 / 49 / 39 / 28 %; eval 36.9 / 48.6 / 51.8 / 56.8 % | W §1; P §1.1, §3 | DEEP L113-120; MEM-H L36-39 |
| Predictions within ~**1–4** points | W §1; P abstract, §3 | DEEP L126 |

## 3. vLLM serving (W §2.1; P §4.1, §5.7)

| Number(s) | Used in | Source |
|---|---|---|
| fp8 KV **1,353,600** tokens; 16 slots; 4,096 batched tokens; gpu mem **0.90**; MTP ×3 | W §2.1; P §4.1, Table 7 | DEEP L54 |
| Larger batched-token setting OOM'd in the MTP drafter (32,768 → 4,096) | W §2.1 | MEM-P L43; E1 L32-33 |
| Prefix caching needs Mamba `align` | W §2.1; P §4.1 | DEEP L264; E1 L24 |
| DIRECT **351** tok/s at 16; **302–325** at 12 | W §2.1; P §4.1, Table 7 | DEEP L172-175 |
| Poetiq R5 ~**16 %** slower; MTP acceptance **0.557**; KV **99.5 %** | W §2.1; P §4.1 | DEEP L181-183 |
| vLLM 0.19 + MTP ignores `thinking_token_budget`, `min_p`, `logit_bias` (vLLM #39573); Exp 5: **10 of 12** tasks hit `max_tokens` | W §2.1; P §6.1 | MEM-P L93, L98; E5 L8-9; E6 L31 |
| `prompt_logprobs` OOM; **15** in-flight units lost | W §2.1, §6; P §6.1 | MEM-P L146; E8 L3 |
| Forced `tool_choice` → empty code / HTTP 500 | W §2.1 | DEEP L261; E4 L48-50 |

## 4. SGLang serving and E11 (W §2.2; P §4.1, §5.7)

| Number(s) | Used in | Source |
|---|---|---|
| INT4 **19.47 GB**, DFlash2 **3.85 GB** | W §2.2; P §9 | E11 L14 |
| **220** wheels; torch 2.13 cu130; flashinfer 0.6.18; transformers 5.12.1 | W §2.2; P §4.1 | E11 L34; MEM-S L14 |
| `unquant` mandatory (#39087), acceptance otherwise ~1.0 | W §2.2 | MEM-S L28 |
| Mamba cache caps: DFlash2 capped at **6** (config C) | W §2.2; P Table 7 | MEM-S L98 |
| DFlash2 + fp16 SSM at a real 12: **404** tok/s, **1.39×** INT4 alone, ~**1.3×** vLLM; KV **387k** | W TL;DR, §2.2; P Table 7 | E11-03A L37, L52, L55; MEM-S L99-100 |
| W1: lazy, Mamba **56**, fp16, no prefill graph, mem **0.83** → 12 real, KV **881k** (881,336), **416.5** tok/s; 458 / 413 / 388 by context band | W §2.2; P §4.1, Table 7, §5.7 | E11-04A L8, L16; MEM-S L108-109 |
| vLLM "about 255–300" tok/s (for the TL;DR range) | W TL;DR | MEM-S L109 |
| E11-05: **328.9** tok/s, **11** really running | (context) | E11-05A L21, L36 |
| E11-06: **379.9** tok/s, **87.2 %** cached | (context) | E11-06A L18 |
| Delayer formula `min(4, max(2, (mrr+5)//6))` = **3** at mrr **13** | W §2.2; P §5.7 | N14 L74; MEM-14 L87 |
| Latch: stuck at **11** in **100 %** of samples; #44401 unchanged | W §2.2; P §5.7 | N30 L34-35 |
| Nodelay crash at **13** running (buffer 12 vs batch 13) | W §2.2, §2.3; P Table 9 | N30 L40 |

## 5. E14 engine measurements (W §2.3, §2.4; P §5.7)

| Number(s) | Used in | Source |
|---|---|---|
| Profiler (20 steps, TP0): all-reduce **50.9** (140 calls), all-gather **8.1**, Marlin **25.3** (512 calls), other GEMM **14.5**, GDN **8.0**, elementwise 2.9, attention 2.7, sampling 2.4, norm 1.1; sum **116.1**; shares 44 / 7 / 22 / 12.5 / 7 % | W TL;DR, §2.3; P abstract, Table 8 | N30 L47-60 |
| Communication **59 of 116 ms (51 %)**; ranks within **0.35 ms** | W TL;DR, §2.3; P §5.7 | N30 L47, L62 |
| "7.8 %" for elementwise + attention + sampling + norm | P Table 8 | **(derived)**: 2.5 + 2.3 + 2 + 1 (N30 L56-59) |
| NCCL bench **345 µs** per 983,040 B; predicted **56.3 ms** (48.3 + 8.0) | W §2.3; P §5.7 | N22 L50; N29 L17; N30 L62 |
| `NCCL_P2P_LEVEL=SYS` **1,887 µs**; SHM memcpy graph timeout | P §5.7 | N22 L51-52 |
| 2×TP2 ~**170k** KV per replica, **0.7–1.0×** | P §5.7 | MEM-14 L48; N25 L18 |
| R2 arms: W1 **505.0** / 114.0 ms / 4.86; #44401 **508.1**; nodelay **490.3**; graphs13 **500.1**; target-only **339.6**; width 6 **531.4** / 96.4 / 4.24; width 4 **526.8** / 74.5 / 3.27; int8 OOM | W §2.3; P Table 7, Table 9 | N30 L19-26 |
| DFlash2-8 = **1.49×** target-only | W §2.3; P abstract, §5.7 | N30 L23 |
| Width 6 = **+5.2 %** vs W1 (531.43 / 505.0) | W TL;DR, §2.3; P Table 9 | N34 L10; MEM-14 L125 |
| Width 4 = +4.3 % vs W1 | P Table 9 | **(derived)**: 526.8 / 505.0 = 1.043 (N30 gives +5.3 % vs graphs13) |
| Long band median **40,877** tokens; widths 6 / 4 never ran it | W §2.3; P §5.7 | N37 L12 |
| **103 of 108** decode lines at 11 running (submitted E14S) | P §5.7 | N37 L19 |
| W1 KV **866,744** | P Table 7 | N22 L67; N30 Duck table |
| E14S KV **917,816** | W §2.4; P §4.1, Table 7 | E14S-LOG L114; E14S-RR `engine.sglang.caps.kv_tokens` |
| E15 MTP predicted KV ~**1,198k** | W §7; P §8 | N35 L74 |
| MTP earlier capped at **10**; **374** tok/s at 10 | W §7; P §5.7 | N35 L17-18 |
| ReplaySSM allowed for EAGLE, blocked for DFlash | P §8 | N35 L56 |
| Explicit draft path needed for MTP | P §8 | N35 L35 |
| Correctness gate: A/A serial **10/12** identical, 2 exact ties | P §4.4 | N22 L77 |
| R2 run **~100** min, quota **3.95 h** | (context for billing) | N30 L9-10 |

## 6. E14S settings table (W §2.4)

| Setting | Source |
|---|---|
| tp 4, kv fp8_e4m3, context **155,648**, mem **0.83**, max-running **13**, chunked prefill **4,096**, lpm, qwen3 / qwen3_coder parsers, stream interval 1, decode graphs 1..**13**, prefill graph disabled, DFLASH, draft tokens **6**, Mamba cache **56**, SSM float16, `extra_buffer_lazy` | E14S-LOG L8 (`server_args`); E14S-RR `engine.sglang.launch_flags` |
| "draft config block_size=8" warning | N31 L94 |
| `unquant` and `preserve_thinking` true | E14S-RR `engine.sglang.launch_flags` |
| Client **12** / server **13** | E14S-RR `configuration.e11.SGL_MAX_RUNNING` / `SGL_SERVER_MAX_RUNNING` |
| Staging to `/dev/shm`, **16** final workers, sha256 | E14S-RES `staging.copies.*.ramp.final_workers`, `.sha256` |
| Duck commit **10882e33**, adapter sha **758bddb5** | E14S-RR `configuration.e12.duck_commit`; E14S-RES `build.arc2_duck_sha` |
| T **1.0**; caps **12,288 / 24,576 / 32,768**; first exposure **28,000**; branches **2**; max gen per branch **100,000**; KV admit **0.95**; nudge **24,000**; workers 12 | E14S-RR `configuration.e12.*` and `configuration.e12.rev6.*` (`E12_DUCK_TEMPERATURE`, `E12_R6_CAP_FIRST`, `E12_R6_CAP_PROGRESS`, `E12_R6_CAP_RESCUE`, `E12_R6_FIRST_EXPOSURE`, `E12_BRANCHES`, `E12_MAX_GEN_PER_BRANCH`, `E12_R6_KV_ADMIT`, `E12_R6_NUDGE_TOKENS`); `e12_run.rev6_workers` |
| Effort = template default (xhigh) | E14S-RR `configuration.e12.E12_REASONING_EFFORT` = ""; E12P L103 (template default is xhigh) |
| Raw reasoning kept | REV-B L46 ("load path, StallMonitor, raw reasoning, payload dataset"); N30 L87 |
| Sandbox **4,096 MB**, Python **3.12.13** | E14S-RR `configuration.e12.E12_SANDBOX_MEM_MB`, `sandbox_selftest.python` |
| Clock **43,020 / 42,720 / T0 + 42,900 s** | N36 L82 |
| Commit: READY **783 s**; **483** tok/s at 12; accept p50 **4.07**; **1/19**; no OOM | N31 L93-97 |
| `ready in 522s` (engine-internal launch time in the same commit) | E14S-RR `engine.sglang.attempts[0].why` |

## 7. Solvers (W §3; P §4.2, §5.3–5.6)

| Number(s) | Used in | Source |
|---|---|---|
| DIRECT recipe from discussion 745787: 49,152 thinking, stop sentence, ≤ 8,192 answer | W §3.1; P §4.2 | E1 L7-9 |
| russcore prompt; T **0.6** / 0.95 / 20; attempt_2 = input or mirror | W §3.1; P §4.2 | E1 L14, L21, L26 |
| Poetiq MIT, repo URL | W §3.2; P refs | E10 L57 |
| Best ≤ **5** earlier solutions; stop at the first train pass | W §3.2; P §4.2 | E10 L66-74 |
| Container blocks `unshare` / `mount`; sandbox = subprocess + rlimits + seccomp + uid drop | W §3.2; P §4.2 | MEM-K L45-47; N36 L19-22 |
| R5 settings: 77,824 + 8,192, 1 expert, ≤ 10 iterations, T 1.0, 16 sequences | W §3.2; P §4.2 | R5A L5-9 |
| R5 **78/172** (45.35 %), **48.61 %**; DIRECT **58/172** (33.72 %), **36.94 %**; union **92/172** (53.5 %), **56.81 %**; 691 min / 11.65 M / 281 tok/s vs 534 min / 10.45 M / 333 tok/s | W TL;DR, §3.2; P abstract, Table 4 | R5A L15-19; E10 L174-175 |
| Overlap 44 / 34 / 14; attempt_2 added **1** | P §5.3 | R5A L21-22 |
| 13 never-solved tasks opened | W §3.2; P §5.3 | R5A L82 |
| **70 %** of solves past 49,152 (39 of 56) | W §3.2 | R5A L44 |
| 77 % of Poetiq solves past 49k | P §5.2 | DEEP L155 |
| 71 verified: **49** right, 7 partial, **15** wrong (69 % / 21 %) | W §3.2; P §5.3 | R5A L53 |
| Multi-test fully solved 14 vs 5 | P §5.3 | R5A L70 |
| Policies 55.14 % vs 56.81 % (97 %) | P §5.3 | R5A L101-108 |
| S1 changes (60,000; ORDER2; F1; F3; A2; F2 / F2b; C1; KV **0.95**; prefetch; clock **11:45 / 11:47 / 11:55**) | W §3.2; P §4.2 | E10 L215, L228-240 |
| R5 repeated attempt_1 on **134 of 172** inputs | W §3.2; P §4.2 | E10 L231 |
| R5 left **16** min unused | W §3.2 | E10 L238 |
| S1-60K sha **ed530071** | W §8; P §9 | E10 L215 |
| Duck vendored from Franzen's fork (Apache-2.0); adapter design | W §3.3; P §4.2 | E12P L7-16, L113; FR L7 |
| rev 1 EVAL30 **7/48** (13.3 %) vs Poetiq **16/48**; branch 1 never started in **29/30**; **8.39 M** re-prefill; 144 / 423 prefix lost; **27 %** at the cap | W §3.3, §6; P Table 6 | E12-30 L17-19, L32, L40-41, L49 |
| rev 4 EVAL6 **3/6** (4/8); ~**10** GPU-min per task; ~**40 h** for 240 | W §3.3; P Table 6 | E12-6 L12, L50-51 |
| rev 5 EVAL6R5 **4/6**, 5/8, **66.7 %**; EVAL12R5_LOW **7/12**, 10/19; 72.7k vs 70.5k per branch | W §3.3; P Table 6 | E12-R5 L19-20 |
| **3 of 5** hits via attempt_2 | W §3.3; P Table 6 | E12-R5 L47 |
| Loop share 0–3 % (rev 4: **29 %**); −30 % tokens | P Table 6 | E12-R5 L19, L54 |
| First `check()` after ~**55k** (median 56.6k / 57.5k); **35 of 36** branches | W §3.3; P §5.6 | E12-R5 L101, L114 |
| 35k soft cap → **1/6** xhigh; **141k → 91k** tokens per task | W §3.3; P §5.6 | E12-R5 L96, L107 |
| rev 6: pause at **28,000**; KV gate **0.95**; 12 workers; nudge at 24k | W §3.3; P §4.3 | N36 L26-29; E14S-RR `configuration.e12.rev6`; MEM-P L277, L279 |
| First Python call median **46** tokens; **7 of 12** b0 runs never defined `transform` in 29–59k | W §3.3; P §5.6 | E12-G4B L24, L26 |
| Gate 4b ≈ **172 s** per task (171.9; formula ≈ 175 s) | W §3.3; P §4.3 | E12-G4B L10; N31 L68 |
| rev 6d **2/19** at **47.7k** tokens/task; step-down 12 → 6, **−13 %** | W §3.3; P Table 6, §5.6 | E12-G4B L17, L20, L47 |
| KV rule replay on 7 logs: holds back only in EVAL30 rev 1 | P §5.6 | E12-G4B L49-53 |
| R2 rev 6g **6/19**, **33.33 %**; pass@2 added 0; **12 of 17** never checked | W §3.3, §6; P Table 6 | N30 L69-70; N34 L20 |
| 240-task simulation: 240/240 exposed | W §4 | MEM-P L278 |

## 8. Scheduling, caps and coverage levers (W §4; P §4.3, §5.2, §5.4, §5.5)

| Number(s) | Used in | Source |
|---|---|---|
| v1.4 admission stop **10:30**; v1.4.4 11:34 / 11:36 / 11:44 | W §4; P §4.3 | DEEP L44-46; E1 L24 |
| Prefix cache saved ~**50k** re-prefill, ~**12 %** of scheduler time | P §5.1 | DEEP L61 |
| ORDER2 **+4–7** points vs random at 50–70 % budget | W §4 | MEM-C L14 |
| ORDER2 AUC **0.57** vs **0.51–0.53**; solve rates 58 / 43 / 20 % | W §4; P §4.3 | E10 L146-147 |
| Cap table 13.75 / **25.42** / **27.92** / 30.42 / **36.94 %** | W §4; P Table 3 | DEEP L77-83 |
| 86k tokens: 10.45 M; **75.4 %** unsolved; **56.6 %** forced; **39/54** natural ≤ 49k | P §5.2 | DEEP L135-139 |
| Early stop AUC **0.63**, saves **3–4 %**; pause / resume worse | W §4, §6; P Table 10 | MEM-C L13, L17 |
| 16 slots @ 77.8k: KV **99.9 %**, **66** preemptions; 12 @ 155,648: 0 preemptions | W §4 | MEM-D L21 |
| D66 **28** / L66 **34** / union **50** of **103**; R5 had 9; 16 / 22 / 12; p = **0.42** | W TL;DR; P abstract, Table 5, §5.4 | E10 L338-354; MEM-D L11-12 |
| 10.80 M / 42,231 s / 256 tok/s; 10.66 M / 41,968 s / 254 tok/s; cut 49/66 vs 9/66; train-pass correct 14/33, 15/43 | P Table 5 | E10 L342-348 |
| **35–42 %** of train-passing programs test-correct | P §5.4 | E10 L359; MEM-D L18 |
| Per slot-hour: D66 0.14 → 0.22; L66 0.27 → 0.25 | P §5.4 | E10 L366-367 |
| **5 of 49** cut answers correct at 77.8k | P §5.4 | E10 L371 |
| W3H **17/172** (9.88 %), 15 tasks inside R5's 63; 0 of 120 at the cap; **78 %** EOS mid-thinking; train pass **13 %** vs **46 %** | W §1, §6; P §5.5, Table 10 | E10 L425-438; MEM-A L33-35 |
| "Advice against caps < 77.8k was wrong" | P §5.4 | MEM-D L45; MEM-P L214 |

## 9. Agent-side prefix reuse (W TL;DR; P §5.8)

| Number(s) | Used in | Source |
|---|---|---|
| `tool_agent.py` L3414–3418 deletes blank lines | P §5.8 | N11 L84 |
| **0 of 180** THINKING blocks contain a blank line | P §5.8 | N11 L93; MEM-14 L77 |
| **68 of 150** (45 %) requests lost the reply; **217.5k** tokens; ~**146 s** at ~**1,500** tok/s | W TL;DR; P abstract, §5.8 | N11 L68, L116; MEM-14 L78 |
| Template token-diff **0/170** | P §5.8 | MEM-14 L55 |
| Fake server: **24/24** diverge (off), **24/24** full prefix (on) | P §5.8 | MEM-14 L82 |
| R2: **115** kept, 0 mismatch; cached **82.0 %** vs G4B **75.0 %** | W TL;DR; P §5.8 | N30 L72-73 |
| Replay 120 follow-ups: 67 / 26 / 27 / 0 | P §5.8 | N30 L77 |
| Same code in Tufa's original harness and Franzen's notebook; RTX PRO 6000 prefill ~**10.5k** tok/s, **3–4×** costlier on L4 | P §5.8 | N11 L138; MEM-14 L84 |

## 10. Platform engineering (W §5; P §6.2)

| Number(s) | Used in | Source |
|---|---|---|
| Commits ran at **329–347 kB** and **407 kB**; **444 kB** ERROR after **114 min** with quota used; **727 kB** no start in **11 min**; **745 kB** `ERRORED_MOUNTING_DATASET` **3 of 3**; limit **407–444 kB**; keep < **400 kB** | W TL;DR, §5; P §6.2 | N21 L66-80; MEM-K L59-65 |
| ~**3.8** quota-hours lost on 10-22 | P §6.2 | N21 L77 |
| Error text only on the web page; API log `[]` | W §5; P §6.2 | N21 L23; MEM-K L66 |
| GCSFUSE from ~10-21; NFS as low as **0.026 GiB/s** | W §5; P §6.2 | MEM-K L50 |
| Launch **507–562 s** → **1,197 s**; 30-min first-launch timeout | W TL;DR, §5; P §6.2 | MEM-K L52; E12-G4B L89 |
| Other teams at 0.00 from the launch timeout | W §5; P §6.2 | MEM-K L51; N24 L20, L22 |
| Staging **518 MB/s** (INT4 19.47 GB); single stream **34.6 MB/s**; plateau at **16** workers, **786 MB/s** | W §5; P §6.2 | N22 L40, L43-44 |
| E14S staging **568.1 MB/s**; single-stream probe **36.9 MB/s** | P §6.2 | E14S-RES `staging.copies.<int4>.mb_s`, `read_probe.mb_s` |
| First launch **502 s**, weights by **65 s**, ~**7 min** compile; relaunch **71 s** | W §5; P §6.2 | N22 L57-62 |
| JIT tar `IndexError` | W §5; P §6.2 | N30 L81 |
| `Thread._stop` crash on Python ≤ 3.12; local harness 3.14; ~**0.8** quota-h lost | W TL;DR, §5; P §6.2 | N22 L23-25; N23 L34 (≈0.8 h) |
| Kaggle CPU kit: Python **3.12.13**, **167/167** tests, harness **32/32** | W §5; P §4.4 | N22 L31; N23 L22 |
| Default image moved to Python **3.13**; pin the old image | W §5; P §6.2 | MEM-P L28 |
| `+` stripped from file names; installer restores them from SHA256SUMS | W §5; P §6.2 | E11 L79-89 |
| Wheelhouse v2: **227** files | W §8; P §9 | E11 L94 |
| Billing **1.05×** (10-13), **1.95×**, **2.37×** | W §5; P §6.2 | MEM-K L25; N23 L58; N31 L75 |
| `E11_ALLOW_VLLM_FALLBACK` has no code; identity ≈ 0 LB | W §5; P §6.2 | N37 L14; REV-C L69-73 |

## 11. Negative-results table (W §6; P §6.1)

| Number(s) | Source |
|---|---|
| Q1 **0/56** at 12,288 (PROXY48) | DEEP L232; MEM-P L13 |
| PIV 0 tool calls at 12,288 | DEEP L233 |
| R6 **2/22**, **92.7k** tok/task, **46/47** forced, 3 of 4 verified wrong | E2 L11-16 |
| Flash-Next ready **862 s**, KV **118k**, garbage output, ~**6** tok/s | E3 L19-21 |
| Q27-AGENT **0/22**; **82** submits forced, **59** empty; **5** HTTP 500; **6,144** per turn | E4 L42, L48-51 |
| v1.9.1 **1/22** | E5 L8 |
| Same-reasoning second answer **+0**; **12/22** ragged or invalid | E6 L5-6 |
| E8 truth in **0/31**; OOM | E8 L3; MEM-P L146-147 |
| Tail resample **0** gain; **41.3 %** of tokens | E9 L3; E10 L18 |
| "0 correct on 8–48-task subsets" (archive) | DEEP L230-238 (Q1 0/56; PIV 0 tool calls; ACT 0 programs; C1 0/16; EVO12 0/12) |
| Poetiq **+11.7** task-norm points over DIRECT | R5A L122 |

## 12. Future work and other teams (W §7; P §7, §8)

| Number(s) | Used in | Source |
|---|---|---|
| E16: nudge after ≥ **2** inspection calls and ≥ **12k** tokens; raw content | W §7; P §4.2, §8 | REV-B L48; MEM-14 L134-136; REV-A L109 |
| EVAL120 G4B commits **234d0919** / **44e11cf6** / **f87c5488**; ~6.5 h each; 120 / 172 | W §7; P Table 6 | MEM-14 L137-139; N31 L67-68 |
| #36548: **111/304** wrong at concurrency 8 (another stack) | W §7; P §7 | N29 L25 |
| E15 arms (EAGLE 3/1/4, 2/1/3, ReplaySSM, 14 / 16 / 20 streams, mrr 14, nodelay at mrr 12, chunked 2048, c12 × 1,024 probe) | W §7; P §8 | N29 L44-51; N37 L22-30 |
| Hybrid: plain reaches **38–57 %** of unverified tasks; 60,000 + 8,192; branch 1 off; regression property | W §7; P §8 | N36 L14-16, L39, L75, L118 |
| "tiny kernel issue": 40 % vs a projected 60 % | W §7 | N24 L19 |
| Other teams at **65–72** (65.56, 65.97, 66–69.72, 72.50) | P §7 | N24 L19-22 |
| No top team's method is public | P §7 | N29 L37 |

## 13. Credits and format sources

| Item | Source |
|---|---|
| Duck harness by Tufa Labs / Jeroen Cottaar; Franzen fork | E12P L7-16; FR L7 |
| Franzen: removing retained thinking is a large regression | FR L69 |
| Franzen checkpoint-prefetch patch and loader flag | FR L41; MEM-P L53 (`--weight-loader-prefetch-checkpoints` in our FN-PROBE) |
| Franzen Mamba-retention patch not ported | REV-B L135 (D8 "OPEN, not in any notebook") |
| NVARC `get_puzzle_priority` copied verbatim | E7 L65-74; MEM-P L91 |
| Land paper: 72.9 % semi-private at $38.99 / task; 76.1 % public at $19.69 / task; arXiv 2606.31543; structure (contributions, negative results §8, limitations §9.1, reproducibility §9.4) | LAND PDF p.1 (abstract and contributions), p.28 (§7), p.31 (§8), p.34 (§9.1, §9.4) |
| Franzen write-up format (TL;DR, numbered sections, settings table) | FR L1-63 |
| AI assistance (Claude and ChatGPT; GPT-authored and Claude-audited versions) | MEM-P L15, L81-88 |
| madarshbb agent and russcore prompt | E4 L7-10; E1 L14 |

---

## Placeholders left in the drafts

| Placeholder | Where | Why |
|---|---|---|
| [PENDING: team name and author list] / [PENDING: author list and affiliations] | W header; P header | Not in our files |
| [PENDING: E14S LB, ref 57574781] | W §0, §7; P §5.1, §8 | Submission still being scored |
| [PENDING: EVAL120 G4B xhigh/E16/low results] | W §4, §7; P Table 6, §8 | Three 6.5 h commits (234d0919, 44e11cf6, f87c5488) running on 2026-10-24 |
| [PENDING: E15 engine arms] | W §7; P §8 | E15 not yet run |
| [PENDING: Hybrid build and smoke] | W §7; P §8 | Design only (N36) |
| [PENDING: source for the 31.94 LB entry; it is not in our files] | W §0; P §5.1 | The caller's list mentions 31.94. No internal file (memory, analyses, Notes, LB dumps) records 31.94 as one of our LB scores. The only hit is another team's "+31.94" delta in `LB arc2 top30 snapshot (2026-10-17).txt` L12 |
| [PENDING: public links to the notebooks and datasets…] / [PENDING: public release…] | W §8; P §9 | Not yet public |
| [PENDING: full bibliographic details … NVARC notebook and discussion threads] | P references | URLs not recorded locally |
