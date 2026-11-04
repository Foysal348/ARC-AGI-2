# FINAL experiment priority list toward 1500 tok/s, NB-K100 baseline (2026-11-01)

## Sources merged
- **Six GPT audits:**
  - #1 "K100 → 1500 Deep Research Audit";
  - #2 "E25 Research Audit (K100 baseline)";
  - #3 "E25 Research Audit: K100 Baseline";
  - #4 "Architecture-Level Research";
  - #5 "4×L4 বড় architectural speedup";
  - #6 "500+ tok/s Structural Breakthrough".
- Research agents A (engine / memory / boot) and B (acceptance / gain / design).
- Our measured data (`E25_1500_PROGRESS_REPORT_2026-11-01.md`, K100Q, K3ONLY v2, NBDF W4).
- A Merlin ad42e37 source check of every item.

This file supersedes lists v1-v7 (kept for the reasoning trail).

## Update 2026-11-01 (0-GPU prep results)
Sources: `Experiment Results/prep_0gpu/duck_timeline/REPORT_duck_idle_and_reprefill_2026-11-01.md` and the K100 real CPU kit (R9_IDX).
- **2-1 (Duck async / ready-queue) drops to ≤ 1 %.**
  - Tools are short: the response → next-request gap is p50 0.29 s against 147-s requests. A ready-queue design recovers only ~1.8 tok/s (0.5 %), at most 3.7 (1 %).
  - The 385 vs 457.5 tok/s gap is **93 % work exhaustion**: 67 of 72 tok/s. From 3,628 s on no attempt was left to start.
- **S-1 (attempts-supply) is promoted to the top of the scheduler track.** Work supply, not the client, decided K100Q's average. Before the queue ran dry the run averaged 452.6 tok/s.
- **2-2 (generated-tail checkpoint) rises to 6.0-8.2 % of the run.**
  - The previous turn's generated tokens are **72.7 %** of server fresh prefill (2.43 M of 3.34 M), not 58 %.
  - A perfect checkpoint saves ~2.43 M prefill tokens. This is a gross figure; the checkpoint's own cost is not counted.
  - The mechanism is confirmed in 677 / 677 later turns: the prefix hit always ends 64-127 tokens before the previous prompt's end.
- **1-4 (indexer KV fp8):** the kit R9 shows the Q pool 480 → **508 blocks (+5.8 %)**. The sm_89 compile and quality are still to be measured on GPU.
- **1-1 (reduced draft vocab) is ready** (`prep_0gpu/draft_vocab/`): 5 tables that load with Merlin's own loader.
  - Held-out coverage 99.36-99.91 %; modelled AL loss −0.74 % (16k) to −0.11 % (64k).
  - Projected ×1.10-1.15 vs I8 (head reads 7.5 ms → ~1 ms).
  - GPU arms: 32k (main), 64k (safe; also used for the + int8 arm), 16k (optional).
- **2-9 (hybrid suffix drafter) → NO-GO** (`prep_0gpu/suffix_replay/`). 87.5 % of generated tokens are reasoning prose, where suffix accepts 0.6 vs MTP's 1.88.
  - Calibrated hybrid: +0.9 % (D 24) for +13.6 % verify tokens.
  - Segment-gated: at most +1.5 %. The oracle upper bound is +8 %.
  - Moved to Tier 3.

## 0. Ground rules for every experiment
1. **Production stays unchanged** (all 6 audits). E28 K100 / W4 keep running. Every lever is tested in a separate notebook, then promoted into the next production build only with evidence.
2. **One change per arm** (audit #1). There is a Q baseline A/A in every notebook.
3. **Bands:** 8k / 40k / 90k / ~116k in every speed test (user rule), plus 20k@24 as the production-level point.
4. **Real-failure gates only** (user rule after the NBDF loss):
   - an arm stops only on a crash, garbage, request errors, zero acceptance, NaN, a real fallback (wrong method / eager / silent path fallback), missing init, or a parity / oracle failure;
   - klee100 is non-deterministic, so parity is judged inside Q's own A/A envelope;
   - a failure stops only its own arm; GO thresholds are read after the run.
5. **Correctness matrix** (audits #1 #3 #6): greedy token-ID parity in Q's envelope; teacher-forced logprobs at 2k / 8k / 40k; a probabilistic distribution check; PP state-commit; Mamba restore position; long oracle 90k / 116k.
6. **Every speed winner also reports** solved tasks, tokens per solve, preemptions, KV residents and memory on all 4 ranks.
7. **Throughput identity** (PP2, two micro-batches): **tok/s = N × AL / (2 × tick)**. Checked: 24 × 3.0 / (2 × 0.0687) = 524 vs 562 measured.
   - 1500 needs N, AL and tick to move together. Example: N 40, AL 3.4, tick 45 ms → 1511.
   - **Capacity (N) is a primary lever**, not only AL.
8. **Two targets** (audit #6):
   - (a) engine capability under a continuously supplied queue;
   - (b) competition whole-run.
   - Work supply bounds (b): K100Q used ~85k tokens per task → ~20 M for 240 tasks ≈ 3.8 h at 1500. The scheduler must turn speed into more attempts (tier S).

## Tier 0: built, next GPU run
| # | Experiment | Expected | Status / gates |
|---|---|---|---|
| **0-1** | **NBDF-K100: DFlash2 on klee100.** Arms Q (E28 production: MTP K3 + Q-defer + I8, seqs 24) → **D3s** (K3, SW-1024) → **D4s** (K4, SW-1024). Windows 8k@16 × 2, 20k@24, 40k@16, 90k@8, 116k@6. Pins from 7,933,526,016; on a D OOM, `DFLASH_DRAFT_EMBED_HOST=1` first, then 256 MiB steps | Central ×1.09-1.13 vs I8 Q (B); a ×1.27 MTP → DFlash2 advantage on a 2-bit-expert model is published (perkel MC); the official card shows the gain rising with concurrency (1.13× → 1.43×) | Build agent finalizing `K100_CFG` from A + B. Gates: A/A-envelope parity, TF logprob D vs Q, zero-accept (own arm only), NaN draft scores, S4 int8-head guard. Recorded: pos0, KV groups / metadata ms, engine block 1024, F (~12), pool (~540), residents vs Q's 29 / 21 / 17 / 13, D/Q at 8k@16 vs 20k@24. CPU kit (A's 10 items) → audit zip → user's go |

## Tier 1: already in Merlin, config only (cheapest, highest value per GPU minute)
| # | Lever | Merlin source | Expected | First step | Vehicle |
|---|---|---|---|---|---|
| **1-1** | **ARC-derived reduced draft vocabulary** (FR-Spec), 32k / 48k / 65k | `VLLM_DRAFT_VOCAB_FILE` (+ `_SHA256`), `models/qwen4_exp/nvidia/draft_head.py`, `mtp.py:525-600`; probabilistic-safe (q = 0 outside the table → target residual); synthetic / adaptive refused | +5-13 % (same-family MiaAI-Lab +9.6 / +13.1 %; FR-Spec head −75 %). Hits stage 1's 3 lm_head reads (7.5 ms) | **0 GPU:** build tables from our saved full responses (train tasks only, specials / tool tokens always in), measure the offline coverage. Then arms: full + I8 (control), 65k, 32k, and 65k + int8 head (`draft_lm_head_quant` wraps a draft-owned head and releases the BF16 copy, `speculator.py:118-135, 585-605`; verify in the kit) | NB-S1 |
| **1-2** | **INT4 draft head** (`draft_lm_head_quant: "int4"`) | `config/speculative.py:610-622` (int4 / int8, g128 Marlin; distribution unchanged) | INT8 gave +3.4 % at 8k and +11.4 % at 40k; INT4 halves the bytes again. Never measured | one arm vs I8 | NB-S1 |
| **1-3** | **GDN / D8 runtime switches:** `VLLM_GDN_SHARED_ALIGN_INDEX`, `VLLM_GDN_DECODE_FASTPATH`, `VLLM_TARGET_SEQ_LENS_ASYNC`, `VLLM_E2_GUARD_MODE=off` | `merlin/envs.py:544-556` (default OFF); = the vLLM #52297 GDN metadata sharing (audit #5) | 0-5 % (CPU / H2D metadata of 36 GDN layers × 2 micro-batches per tick; E2 guard −0.3 ms per step) | one switch per arm, bitwise / greedy parity | NB-S1 |
| **1-4** | **QSA indexer KV fp8** (`--attention-config.indexer_kv_dtype fp8`; official recipe) | `indexer_qsa.py:152-176`, default bf16; plain Triton fp8 dot (sm_89 compile UNVERIFIED) | capacity (+pool blocks) + indexer bandwidth | **0 GPU:** kit R9 pool with / without (already requested in the NBDF-K100 kit), then an arm with oracle + quality | NB-S2 |
| **1-5** | **Stream sweep 24 / 28 / 30 / 32** at 20k / 40k / 90k, with the KV ledger (running-required / reclaimable / free), preemption, recompute | config | 0-15 % (resident bound 43 / 29 / 17 / 13 at 20k / 40k / 90k / 120k) | — | NB-S2 |
| **1-6** | **Dynamic K schedule** `num_speculative_tokens_per_batch_size` (e.g. K4 at ≤ 8 running, K3 above) | `config/speculative.py:493`, scheduler `:348`; FULL graphs kept on V2 (`config/vllm.py:1068-1085`) | gains in the low-concurrency tails (A/B/C re-runs, end of run) | **First measure MTP K4 at 1 / 4 / 8 streams (never measured).** Real-failure read-back: cudagraph mode stays FULL (vLLM #49548) | NB-S2 |
| **1-7** | **Long-prefill threshold + adaptive** (#57951; caution #60476) | `config/scheduler.py:79-90` | 1-6 % | 1024 vs contention-aware 2048; metric: whole-run tok/s + p95 decode stall | NB-S2 |
| **1-8** | `--enable-mamba-shared-prefix-checkpoint` (side arm) | `config/cache.py:198`; writes junctions only inside the prompt | small (branching only; not the 58 % generated-tail re-prefill) | check `num_nextn_predict_layers` = 1 (single-module MTP) | NB-S2 |
| **1-9** | **Stage-1 profiler (measurement, no change)** at 16 / 24 streams: stage tick split, 3 draft heads, the AllGather 2.9 ms split (comm / kernel / sync), eager post-target kernels, FULL / PIECEWISE / EAGER per batch size, metadata kernel count, per-layer expert-ID overlap, unique expert bytes per tick | torch.profiler + existing read-backs | sizes every Tier 2 item | — | NB-S1 (first arm) |

## Tier 2: ports or our own code (medium cost; ordered by value)
| # | Lever | Expected | Notes / gates |
|---|---|---|---|
| **2-1** | **Duck async / ready-queue** (client side): a ready-for-GPU queue separate from waiting-for-tool; prepare the next request during decode | 0-7 % (window 385 vs 414 all-busy) | Our code. First, 0 GPU: the idle timeline from the K100Q / E28 worker timestamps (GPU idle intervals, ready-queue depth, scheduler CPU overhead; audits #1 #4) |
| **2-2** | **Generated-tail SSM checkpoint** (58 % of 3.34 M fresh-prefill tokens in K100Q); client-directed (#55697) or an automatic finish-time publish | 2-6 % (TTFT larger; #52789 total latency only 0.2-3.8 %) | 4 gates: token-ID identity, checkpoint position, KV + recurrent-state equivalence, PP0 / PP1 fence. Recompute the 58 % from per-turn records first |
| **2-3** | **MTP draft path:** #49793 fused trailing all-reduce (applies to probabilistic drafts); greedy local argmax as diagnostic only (exact (max, id) reduce, tie behaviour) | 3-9 % | Inspect the diff before 2-4 (audit #3). Moot if DFlash2 wins |
| **2-4** | **Qwen4Exp sharded sampling:** #61018 + #50465 (+ RFC #54333 notes) | 0-3 % (upstream +1.09 %; AllGather max ~4.4 %) | Read back that the All-to-All really ran (silent fallback risk) |
| **2-5** | **Target-forward fusion:** SGLang #42175 (HC norm / combine, GDN QKV, gated finalize; +12 %, kernels 1432 → 1241 on B200) + #41173 / #41175 / #40972 / #41166-41171 (QSA / NEXTN metadata) | 2-12 % | Benefits MTP and DFlash2 alike. Port the 1-2 largest chains found by 1-9; one isolated fusion at a time; sm_89 numerics, graph capture, PP ordering |
| **2-6** | **Humming W2 g64 / W3 g128 shape bench** T 4-128 on real routes (humming above T64 unmeasured), then dispatch / GEMM tables | 0-10 % | Cancel if a GEMM gain < 0.5 % tick; klee100 packing unchanged |
| **2-7** | **Post-target graph capture** (~4.2 ms eager) | 2-8 % | After 1-9 shows which kernels are outside graphs and why |
| **2-8** | **F reduction / live-block admission port** (F 6 → 4 frees ~48 blocks; #62752 CPU reproducer in a kit; RFC #64508) | 0-10 % | Gates: 0 preemption, no stale state, no page reuse before PP work completes, 40k / 90k replay |
| **2-9** | **Hybrid suffix / n-gram drafter** (Merlin suffix decoding) | unknown | 0 GPU first: replay saved Duck outputs offline, hit rate per segment (code / grid / JSON / prose) |
| **2-10** | **DSpark + adaptive verification** (Merlin: dspark method + `enable_adaptive_verification`, dspark only) | unknown; the author's AL 3.70 < MTP-7 4.24 | Extend `dflash_seam.py` to dspark; after 0-1 |
| **2-11** | **Draft / target KV-bucket sharing** (#58207) | only if 0-1 shows a D pool / resident deficit | the lesson: fewer groups ≠ better |
| **2-12** | **Boot cache** (FlashInfer JIT 108 s; version-matched precompiled kernels) | +0.5-1.2 % whole-run | ABI / arch pinned; warm vs cold |
| **2-13** | **MTP-only FP8 KV scale calibration** (AL 2.833 vs 2.957 BF16) | 0-4 % | Change only the MTP layer's scale; TF logits at 1k / 8k / 40k / 90k |
| **2-14** | **TP1 × PP4 on Merlin + klee100**: the memory-fitting way to remove the per-layer TP collectives (the goal of audit #6's 2 × (TP1 × PP2)) | ≤ ~18 % of the stage tick (AllReduce 3.2 + AllGather 2.9 ms of 33 ms), minus the PP4 cost (4 micro-batches, so dense weights are read twice as often per stepped token) | **Per-GPU weights stay ~11.5-12 GiB** (a quarter of the layers at full width vs half the layers at half width now), so it fits. The last stage gains the full LM head + MTP (+~1.2-1.8 GiB). **The E23 PP4 crash was stock vLLM 0.32 + RedHat INT4 27B, eager, issue #63505, never Merlin + klee100**, so it is context, not a verdict. Steps: CPU memory fit per rank → 2k / 8k / 40k smoke arm with real-failure gates → tok/s at 16 / 24 streams vs TP2 × PP2 |

## Tier S: scheduler (score per hour; after the E28 results)
- **S-1 attempts-supply policy:** extra attempts for train-fail tasks and re-runs of cap-cut attempts while time remains, so a faster engine becomes more useful work (audit #6 §7).
- **S-2 re-analyse the agent budget / early stop** with the E28 per-attempt logs. Past AUC .63 / 3-4 % is context from another setup, not a verdict.

## Tier 3: research, after the competition or conditional
| Item | Condition |
|---|---|
| DFlash2 Domino rollout (SGLang #36899 / #37069) | after 0-1 |
| ARC-specialized drafter (DSpark / DFlash2 fine-tune, speculators; DFlare / AngelSlim) | training; our saved traces are the data (train tasks only) |
| Smaller checkpoint + 2 × (TP1 × PP2) replicas | With klee100: measured 11.45 / 11.97 GiB per GPU now → ~22.9-23.9 GiB per GPU under TP1 × PP2, against ~22 GiB usable on an L4, leaving no KV. Each replica would also get half the KV pool, so long-context residents drop. It becomes viable with a checkpoint of ≤ ~15 GiB per stage (2-bit-class trunk, ARC quality run needed). Its goal (no TP collectives) is pursued by 2-14 instead |
| Expert-locality-aware batching | only if 1-9 shows a large expert overlap |
| Mixed-bit requant / PuzzleMoE / Haberstroh W2 g128 | matched ARC quality run vs 23 / 30 |
| INT4 / NVFP4 PLE tables (primitive-ai) | host RAM / boot, not decode |
| GDN ReplaySSM (Merlin's is Mamba2 / KDA only) | port |
| MegaCapybara / ds4 compact-engine study | paper |

## Do-not list (all audits + our measured negatives)
- TP4 / DP; PP 26 / 22 (only as a memory-reserved branch if the 24-stream profile proves imbalance). PP4 is NOT on this list: see 2-14.
- Live-KV offload, admitting beyond residency, time-slicing.
- Forced NCCL LL128, P2P, IPC all-reduce, NCCL CTA / SHM memcpy tuning.
- BF16 SSM state; prefix caching off with MTP; MTP K2.
- Generic Marlin config sweep; `--enable-hc-sp` with PP; SGLang EP4; NVFP4 KV on QSA.
- DBO / TBO on TP2 × PP2 (DP + EP only); CRISP / lossy expert pruning; GGUF IQ1 / Q2 targets.
- Re-measuring the adopted Q-defer / I8 / FP8 QSA KV.

## Notebook plan (merged from all audits)
| Notebook | Arms (one change each) | GPU min (+ boot) |
|---|---|---|
| **NBDF-K100** (tier 0) | Q, D3s, D4s | ~150 |
| **NB-S1 compute** | Q A/A + profiler (1-9); reduced vocab 65k, 32k, 65k + int8 (1-1); INT4 head (1-2); D8 switches one per arm (1-3) | ~60-90 |
| **NB-S2 capacity** | indexer fp8 (1-4, after the kit R9); stream sweep + ledger (1-5); MTP K4 at 1 / 4 / 8, then the dynamic K schedule (1-6); long-prefill (1-7); shared-prefix flag (1-8); TP1 × PP4 arm (2-14, after the CPU memory fit) | ~75-105 |
| **NB-S3 combined real Duck** | Q vs the stacked winners of tier 0 / S1 / S2 on a matched task set; solved tasks, tokens per solve, tok/s, residents | ~45-75 |
| then | winners → the next production build (E28-style, every proven lever + scheduler S-1) | 6 h |

**Order:** NBDF-K100 and the 0-GPU prep run first and in parallel:
- the vocab tables + offline coverage;
- the R9 indexer fp8 pool;
- the Duck idle timeline;
- the suffix replay;
- the 58 % recompute.

Then NB-S1 and NB-S2 (two accounts in parallel), then NB-S3. Tier 2 ports start from the NB-S1 profile.
