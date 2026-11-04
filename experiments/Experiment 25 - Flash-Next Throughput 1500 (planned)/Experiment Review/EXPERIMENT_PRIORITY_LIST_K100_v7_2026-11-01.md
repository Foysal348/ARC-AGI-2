# Experiment priority list toward 1500 tok/s, NB-K100 (v7, 2026-11-01): GPT audit #6 "500+ tok/s — Structural Breakthrough Research" integrated

v4 (gap check), v5 (audit #4 + the PP2 math) and v6 (audit #5) stay valid. v7 adds audit #6, checked against Merlin ad42e37 and our GPU logs.

## A. The big finding: FR-Spec reduced draft vocabulary is ALREADY in Merlin for Qwen4Exp MTP
- **Source:** `models/qwen4_exp/nvidia/draft_head.py` ("Reduced draft vocabulary for the Qwen4Exp MTP drafter (default OFF)") and `models/qwen4_exp/nvidia/mtp.py:525-600`.
- **Switch:** `VLLM_DRAFT_VOCAB_FILE=<ids.pt>`, plus optional `VLLM_DRAFT_VOCAB_SHA256`.
- **How it works:** the drafter owns an N-row head holding the listed checkpoint rows and scatters its logits back into full-vocab coordinates with −inf elsewhere. Target verification is untouched.
- **Probabilistic-safe:**
  - "with standard / block rejection (and greedy) the accepted output follows the target's own distribution: a draft vocabulary only moves acceptance". An out-of-table token has q = 0 and is reached through the target residual.
  - Synthetic rejection and adaptive verification are refused (`config/speculative.py:1904-1930`).
  - This answers audit #6's main correctness concern: Merlin already does the probabilistic-correct version, unlike MiaAI-Lab's greedy patch.
- **PP-aware:** the head lives on the last PP stage only, rows aligned to 128 per TP shard. A malformed ids file fails the boot; it never degrades silently. TP fingerprint agreement is checked.
- **Also feeds** `LocalArgmaxMixin.get_top_tokens` (the d2t map).
- **Cost:** an extra N × 2560 × 2 B BF16 head split over TP2 on PP1. 64k rows ≈ 164 MB per rank, 32k ≈ 82 MB. The pool drops by about 6-11 blocks at 14.7-16.5 MB per block.
- **External evidence, same model family:**
  - MiaAI-Lab Flash-Next patch, ~47k draft vocab: +9.6 % (dual Spark) and +13.1 % (single Spark) decode, greedy MTP3.
  - FR-Spec: draft head compute −75 %, ×1.12 overall.
- **New P0-5:** an ARC-derived draft vocabulary.
  - **The ids file is built on CPU from our saved full responses** (every request's thinking + output is saved: K100Q, E28 K100 / W4, NBDF). Take the token frequency, keep the top 32k / 48k / 65k, round to 128, always include special / tool tokens, and use train-split tasks only.
  - **0-GPU offline estimate first:** the coverage of saved outputs by each table, i.e. the fraction of generated tokens inside the table. That is an upper bound on the draft acceptance kept.
  - **GPU arms (one change each, on K100 Q):**
    - A = full vocab + I8 (current);
    - B = 65k table;
    - C = 32k table.
    - Read back AL, per-position acceptance, stage-1 draft ms, 8k / 40k / 90k / 116k tok/s, greedy parity and the distribution check.
  - **Open:** whether the reduced head combines with `draft_lm_head_quant` int8 (I8). Check the source before the build. If they are exclusive, the arms are I8-full vs BF16-reduced.

## B. 2 × (TP1 × PP2) replicas: the memory-fit gate, answered from our own GPU log
- **Measured per-GPU weights on klee100** (K100Q server log, `model_runner.py:544`): **11.45 GiB on PP0 ranks, 11.97 GiB on PP1 ranks** under TP2 × PP2 + EP.
- **Under TP1 × PP2** each GPU holds the whole stage instead of half of it: about **22.9-23.9 GiB of weights per GPU**, against about 22 GiB usable on an L4 after CUDA context and runtime. That leaves no room for KV, graphs or the drafter.
- Removing the MTP head and its duplicated embedding (~1.2 GB) still leaves about 21.7-22.7 GiB.
- **Gate result on klee100: does not fit.** The layout becomes a candidate only with a checkpoint of about ≤ 15 GiB per stage, i.e. roughly 2-bit-class weights for the whole trunk. That is a quality-risk requant (audit #6 agrees W2 needs matched ARC evaluation).
- **Kept as P2:** "smaller-checkpoint + 2 × (TP1 × PP2)" study, paper-only until such a checkpoint passes ARC quality.
- PP4 stays on the do-not list (E23 crash at 8k, pool loss); audit #6 agrees.

## C. Work supply bounds the whole-run average (audit #6 §7, important for the goal definition)
- **Audit #6's cap:** 240 tasks × 2 attempts × 120k = 57.6 M tokens → at most 1,391 tok/s over 41,400 s.
- **Our real supply is smaller.** K100Q generated 2.54 M tokens for 30 tasks (≈ 85k per task, including re-runs) → 240 hidden tasks ≈ 20 M tokens. At 1500 tok/s that is about 3.8 h of work.
- **So at high engine speed the run finishes its queue early.** The whole-run average then falls unless the scheduler turns speed into more useful work: more attempts per task (attempt 3+ for train-fail tasks), larger B2 budgets, and a re-run of every cap-cut attempt.
- **Two targets, tracked separately (accepted from audit #6):**
  - (1) **engine capability:** sustained tok/s under a continuously supplied queue (the 1500 target);
  - (2) **competition whole-run:** tok/s and solved tasks over the full elapsed time.
- **Action:** add an attempts-supply policy to the A/B/C scheduler's next revision (after the E28 results): extra attempts for train-fail tasks while time remains. That is a scheduler change, not an engine one.

## D. Other audit #6 items
| Item | Check | Decision |
|---|---|---|
| Expert-locality-aware micro-batching (group requests by routing histogram; RFC #38256 is offload-oriented) | No Merlin support; the gain depends on L2 reuse | P2. First measure cheaply: per-layer expert-ID overlap within a micro-batch and unique expert bytes per tick (record in the NB-S1 profiler); build only if the overlap is large |
| SGLang Two-Batch Overlap / vLLM DBO | DBO is DP + EP only | NO on TP2 × PP2 (audit agrees a flag will not help) |
| HaberstrohSystems INT2 mixed AutoRound (≈2.57 bpw; 15.5 tok/s single Blackwell + offload) | Merlin humming already routes its int2 path (`merlin-humming-lowbit-path` memory) | P2: a W2 g128 vs g64 sensitivity study needs a matched ARC quality run (23 / 30 reference) |
| PP stage function placement (A: INT4 / reduced head; B: head overlap; C: separate head stream) | A = section A of this file; B / C need the NB-S1 timeline (stage_idle_ms, head_ms, draft_ms, pp_wait_ms, critical_path_ms) | Option A → P0-5. B / C → after the NB-S1 profile (v4 P1-2 / P1-5) |
| SGLang PP roadmap #43885, #39634 (micro-batch pairing, async PP loop, recurrent-state commit fixes) | references | Add to the PP correctness gate list (state commit) |
| **Hybrid drafter:** suffix / n-gram context proposals for code, grids, JSON, tool replies, else MTP / DFlash2 (jitwark qwen38-27b-rtx3090 design) | Merlin has suffix decoding (arctic-inference) | **Raised to P1-13, with a 0-GPU step:** replay our saved Duck outputs through a suffix-tree / n-gram proposer offline and measure the hit rate and accepted length per segment type (code, grid, JSON, prose). Build a hybrid only if code / grid segments show long matches |
| Fused MTP trailing AllReduce (#49793) | already v4 P1-2 | unchanged |
| Dynamic per-request depth (DFlash2) | v5 P1-9 / v6 | unchanged |
| Go / No-Go 5 criteria (correctness, ARC quality, long context, real 4-rank memory, matched throughput) | matches v4 section D | unchanged |

## E. Ranking update (v7 deltas)
| # | Experiment | Why |
|---|---|---|
| **P0-5 (new)** | ARC-derived reduced draft vocabulary via Merlin's built-in `VLLM_DRAFT_VOCAB_FILE` (32k / 48k / 65k). CPU table build + offline coverage, then 3 GPU arms | Already implemented and probabilistic-safe in Merlin; it hits the stage-1 bottleneck (3 lm_head reads, 7.5 ms); same-family external +9.6-13.1 % |
| **P1-13 (new)** | Hybrid suffix / n-gram drafter: offline replay of saved Duck outputs first | 0-GPU first step; ARC grids / code may repeat |
| P2 | Smaller checkpoint + 2 × (TP1 × PP2) | klee100 does not fit (B) |
| P2 | Expert-locality batching | measure the overlap first |
| NO | DBO / TBO on TP2 × PP2 | DP + EP only |
| Scheduler (after E28) | attempts-supply policy so a faster engine turns into more useful work (C) | the whole-run average is bounded by work supply |
