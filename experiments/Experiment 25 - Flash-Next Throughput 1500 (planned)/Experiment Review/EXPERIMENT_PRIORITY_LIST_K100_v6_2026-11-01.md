# Experiment priority list toward 1500 tok/s, NB-K100 (v6, 2026-11-01): GPT audit #5 integrated

**Audit #5:** "4×L4-এ Qwen3.8-Flash-Next: বড় architectural speedup-এর গবেষণা". It is checked against Merlin ad42e37 and our files.

v4 (gap check) and v5 (audit #4 + the PP2 throughput math) stay valid. v6 adds only what is new.

## A. Audit #5 items vs Merlin and our data
| Item | Audit #5 | Check today | Consequence |
|---|---|---|---|
| **Official DFlash2 / MTP table** (GB300 TP1): GSM8K c1 1.13×, MATH-500 c8 1.18×, GSM8K c32 **1.43×**, HumanEval c32 1.31× | promising | Model-card numbers (FP8 target, TP1) | **The gain rises with concurrency there.** That is the opposite of research agent B's PP2 model (×1.21 at 8 streams → ×1.15 at 24), where a fixed draft saving is diluted by a growing tick. NBDF-K100 must measure 8k@16 and 20k@24 and settle it. Do not assume either trend |
| **SGLang #35371** (DFlash2 local conv + candidate selector) | reference | Merlin has both: `qwen3_dflash2.py` grouped-conv Triton kernel ("adapted from LightSeek TokenSpeed PR #1399") + `CandidateSelector` | Nothing to port |
| **DFlash V2 Domino rollout** (SGLang #36899 / #37069, TP > 1) | research | Not in Merlin (no grep hit) | P2 research; needs an SGLang → Merlin port. Only after D3s / D4s are measured |
| **KV-group explosion with DFlash on hybrid GDN** (vLLM #58638: 4 → 70 groups; per-group metadata cost) | the biggest structural lead | NBDF already reads it back (`nbdf_lib.py:1463` `KV_GROUP_EXPLOSION`, recorded). Q has 5 groups (4096, 4096, 4096, 8, 4096); agent A predicts D-SW groups via kit R9 | **Covered.** NBDF-K100 records q_groups vs d_groups plus the per-step metadata time in the 30-s profile. A real explosion shows as a tick regression, and the arm comparison catches it |
| **#58207 draft / target KV-bucket sharing** (Qwen3.5 + DFlash: groups 14 → 5, usable KV +27 %; another config −31 % KV for 1.7 % latency) | layout redesign | Not found by grep in `kv_cache_utils.py` (Merlin has `is_eagle_group` / uniform-page grouping, not bucket sharing) | **New P1-11 (port, after NBDF-K100):** only if the D arm's measured pool or residents lose to Q. Agent A predicts D-SW 540 blocks vs Q 479, but F 12 vs 6, so long residents are 18-20 vs 21 at 64k. The audit's lesson is accepted: fewer groups is not automatically better; optimize pool and metadata together |
| **#58345 sliding-window block-size correction** | | No hit by number; agent A traced the SW-1024 block path in source (`core.py:434-448`, `kv_cache_interface.py:879-886`) | Covered by the kit A3 simulation (block drop + `level_for_band` at 4096) |
| **#52297 GDN metadata sharing** | | **Merlin equivalent exists but is OFF by default:** `VLLM_GDN_SHARED_ALIGN_INDEX` ("the align-mode state-index gather index is computed once per batch and reused by every GDN group") and `VLLM_GDN_DECODE_FASTPATH` ("pure spec-decode FULL-graph batches … stage the GDN metadata with slices and fills instead of CPU-mask advanced indexing and H2D copies") (`merlin/envs.py:544-556`) | **GAP found: these D8 switches were in PLAN_1500 D8 and the progress report, but dropped from priority lists v1-v5 and never measured.** New **P1-12**: config-only A/B on K100 Q (one switch per arm), greedy / bitwise check. They cut CPU / H2D metadata work on 36 GDN layers × 2 micro-batches per tick |
| **SGLang #42175** Flash-Next HC / GDN / gated-finalization fusion (4×B200 BF16 TP4 c1: 242.4 → 271 tok/s, +12 %; kernels per graph 1432 → 1241; correctness comparisons in the PR) | target-forward fusion | Not in Merlin | **Raises v4 P1-5 / P1-8 (target-forward fusion).** It helps MTP and DFlash2 alike. Order: NB-S1 profiler kernel count first, then port the one or two largest chains (HC norm / combine fusion; GDN QKV materialization). Check numerics, graph capture and PP collective ordering on sm_89 with the humming W2 / W3 weights |
| **SGLang #41173 / #41175** | NEXTN metadata | already v4 P1-8 | unchanged |
| **Dynamic speculation** (vLLM docs) **+ vLLM #49548** (a dynamic MTP policy changed the CUDA-graph mode and collapsed throughput) | adaptive K with prebuilt graph tiers | Merlin overrides FULL → PIECEWISE for dynamic SD **only when not on the V2 runner** (`config/vllm.py:1068-1085`). We run V2, so FULL graphs stay | v5 P1-9 gets a hard read-back: **cudagraph_mode must still be FULL_AND_PIECEWISE with dynamic K** (a real fallback = real failure), plus capture lists covering every tier K × (1 + K) × seqs. SGLang's EMA controller is EAGLE-only (accepted) |
| **Context-aware K / DFlash disable at long context** (vLLM #54691 long-context regression) | | NBDF-K100 measures 90k / 116k per arm; agent B adds the "AL at 90k / 116k ≥ 0.9 × 8k" adoption rule | A context-aware policy is a later extension of P1-9; nothing new to build now |
| **Asymmetric TP1 × PP4** with redesigned MTP placement | high-risk structural | Our PP4 crashed at 8k (E23); TP4 / PP4 are on the do-not list | Kept as **paper-only feasibility (P2)**: per-rank W + KV + SSM + graphs + drafter < 24 GiB with measured layer latencies. No GPU run without new quantitative evidence (audit #5 says the same) |
| **incoai DFlash2-GGUF; qtum GGUF (IQ1 / Q2)** | | GGUF loader + HC-mean capture incomplete; big perplexity cost at IQ1 / Q2 | NO (quality risk for exact-match ARC) |
| **Layer / expert-aware mixed-bit retuning; PuzzleMoE expert merging** | P2 | lossy; needs a requant and ARC re-validation | P2, long-term |
| **DFlare / AngelSlim custom drafter** | P2 | needs training | after the competition (same as v5) |
| **Adaptive agent budget / early-stop scheduling** | P2, score-per-hour | Our own data: early stop of hopeless thinking reached AUC .63 and saved only 3-4 % (coverage-levers analysis). That is context from a different setup, not a verdict | The A/B/C scheduler in E28 already re-allocates budget. Re-analyse after the E28 6-h results (they log train-fail / cap-cut per attempt) |

## B. Ranking update (v6 deltas only)
| # | Experiment | Change |
|---|---|---|
| P0-1 | NBDF-K100 (D3s → D4s) | Must settle whether the DFlash2 gain rises (official card) or falls (B's model) with concurrency: compare 8k@16 vs 20k@24 per arm. Record KV groups + metadata time |
| **P1-12 (new, gap)** | `VLLM_GDN_SHARED_ALIGN_INDEX`, `VLLM_GDN_DECODE_FASTPATH` (and the other D8 switches `VLLM_TARGET_SEQ_LENS_ASYNC`, `VLLM_E2_GUARD_MODE`) | config only; one per arm on K100 Q; greedy / bitwise gate; fits NB-S1 |
| P1-5 / P1-8 | Target-forward fusion: SGLang #42175 (HC / GDN / gated finalize), #41173 / #41175 | raised: it benefits both MTP and DFlash2 |
| P1-9 | Dynamic K | add a real-failure read-back: cudagraph mode stays FULL (V2) |
| **P1-11 (new)** | #58207-style draft / target KV-bucket sharing | only if NBDF-K100 shows a D pool / resident deficit |
| P2 | Domino rollout; TP1 × PP4 paper feasibility; mixed-bit retune / PuzzleMoE; DFlare | new P2 rows |
| NO | GGUF IQ1 / Q2 targets | added to the do-not list |
