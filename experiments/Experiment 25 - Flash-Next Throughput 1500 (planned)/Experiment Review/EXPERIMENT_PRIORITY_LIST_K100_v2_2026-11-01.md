# Experiment priority list toward 1500 tok/s, NB-K100 baseline (v2, 2026-11-01)

**Inputs:**
- GPT audit #1 ("K100 → 1500 tok/s: Deep Research Audit");
- GPT audit #2 ("E25 — 1500 tok/s Research Audit (K100 baseline)", in Bengali);
- our measured data (`E25_1500_PROGRESS_REPORT_2026-11-01.md`);
- a Merlin source check (ad42e37).

One more GPT audit is pending, and this list will be updated again after it.

v1 (`EXPERIMENT_PRIORITY_LIST_K100_v1_2026-11-01.md`) is kept unchanged for reference.

**Baseline:** NB-K100 (klee100 3bpw-MTP, Q-defer, I8, MTP K3).
- **K100Q** (21 workers, 30 tasks): 385 tok/s window average.
- **E28 K100** (24 workers, 6 h EVAL120): running now. It will become the real 24-worker baseline (audit #2: "24-stream baseline is not in the zip yet").

## A. Merlin source facts that change the ranking (verified today)
1. **`--enable-mamba-shared-prefix-checkpoint` does NOT cover our 58 % re-prefill.**
   - Source: `v1/core/sched/scheduler.py:595-605`, `v1/core/single_type_kv_cache_manager.py:2476-2483`.
   - The junction checkpoint is written only when `junction <= request.num_prompt_tokens` and `num_computed < junction <= num_prompt_tokens`. The source comment says "Past the prompt the manager writes nothing".
   - The junction is where a *sibling* request was observed to resume. That is Marconi-style branching off a shared prompt, not a linear chain.
   - In Duck, turn N+1 re-prefills turn N's *generated* tokens. Their state was produced during decode, past turn N's prompt, so no checkpoint exists there and this flag never publishes one.
   - It can only help where two requests branch inside a prompt (A/B/C re-runs, attempt 2, the same task prompt). These already hit the prompt-tail checkpoint, so the expected gain is small.
   - Activation conditions (`v1/core/kv_cache_manager.py:189-194`): `eagle_group_ids` non-empty (MTP falls back to flagging all groups, `kv_cache_coordinator.py:134-135`, so this is satisfied); partial hash hits on (unit 64 < 4096, ✓); `num_reprefillable_tokens == 0`, i.e. single-module MTP. Our K3 MTP is believed to be single-module; check the startup log line "prefix-cache retention feature".
   - **Verdict:**
     - The flag is a cheap 1-line side arm.
     - The real lever is a **port**: publish an align checkpoint for the generated tail when a request finishes. This must happen after the last decode step's forward, after PP1, after the RecoverSSM flush.
     - Audit #2 section 4 states the same port direction.
2. **Batch-sharded sampling** (`config/parallel.py:192`) needs the model's `compute_logits_local`. Qwen4Exp has none; the DeepSeek-V4 template exists. This is a port.
3. **Local argmax draft reduction** exists only for DeepSeek-V32 / V4 / GLM5 MTP. It is a port, and moot if DFlash2 replaces MTP.
4. **Audit #2 math accepted.**
   - Removing the whole 2.9 ms AllGather at a 68.7 ms tick is at most about 4.4 % tick speed.
   - Reduced-TP sampling (vLLM RFC #54333 / PR #54332) falls back to full logits for probabilistic speculative verification, which is our MTP K3 path. So the gather can only be optimized, not removed, unless we move to greedy drafts.

## B. Priority list (v2)
| # | Experiment | Why | Expected | Cost / risk | Audit support |
|---|---|---|---|---|---|
| **P0-1** | **NBDF-K100: DFlash2 on klee100** (Q / D4s / D3s, SW-1024, 8k / 40k / 90k / 116k bands). Decision metric: accepted tokens / wall-second at the real stream count, not AL | The largest single structural lever. Both audits warn that the pool drops 479 → ~205 blocks, so a per-stream gain may be lost at 24 streams | −20 % to +30 % | High; build in progress (2 research agents + build agent) | #1 rank 5, #2 rank 4 ("decide on GPU evidence") |
| **P0-2** | **NB-S2a KV ledger + F reduction / live-block admission** | F = 6 × 24 = 144 blocks reserved (30 % of 479). F 6 → 4 frees about 48 blocks (+10 % of the pool), and more resident streams follow. Instrument per-group allocated / live / next-boundary / released first (vLLM #62752, RFC #64508) | 0-15 % | Medium-high; gates: 0 preemption, greedy token-ID parity, no stale Mamba state, no page reuse before PP work completes, 40k / 90k replay | #2 rank 2; #1 capacity section |
| **P0-3** | **Stream sweep 24 / 28 / 30 at 20k / 40k / 90k** with KV free blocks, preemption, recompute tokens and accepted tok/s per window | Cheap; it sizes P0-2 and gives the stage-1 tick split under load | 0-12 % | Low; 1 short notebook (folded into NB-S1 / S2) | #1 rank 4, #2 rank 6 |
| **P1-1** | **Stage-1 critical-path instrumentation + graphing post-target work** (LM head, MTP heads 7.5 ms of 3 lm_head reads, AllGather 2.9 ms, eager post-target ~4.2 ms) | Stage 1 binds at 33 ms vs 21.6 ms. Graph capture of the eager post-target kernels (4.2 ms) is larger than the AllGather. GO for the full port only if the mechanism saves ≥ 2 ms (audit #2 gate) | 3-12 % | Medium; the probabilistic path stays untouched, greedy-only control arm | #1 rank 2, #2 rank 1 |
| **P1-2** | **Generated-tail SSM checkpoint port** (section A.1) + the shared-prefix flag as a side arm | 58 % of uncached tokens; TTFT and prefill stalls drop. Est. +4.6 % (audit #2 keeps this unverified: the raw per-turn traces were not in the zip) | 2-6 % | High; gates: replayed token IDs == original, checkpoint position == prefix length, attention-KV and recurrent-state numerical equivalence, PP0 / PP1 completion fence | #1 rank 1 (as a flag), #2 rank 3 (as a port) |
| **P1-3** | **Humming W2 g64 / W3 g128 shape bench** at T 4 / 8 / 16 / 24 / 32 / 48 / 64 on the real expert-token distribution, then tile / CTA tuning. Not Marlin (already measured at ~0 gain) | MoE 172 µs at 4 tokens → 645 µs at 28; humming is worst at T 16; W2 g64 has never been benched. Model-level tick check before GO | 0-10 % | Medium (extend the NB3 bench) | #1 rank 9, #2 rank 5 |
| **P1-4** | **Mixed-prefill graph coverage read-back + prefill chunk 1024 vs 2048.** Metric: combined Duck output tok/s and p95 decode stall | A prefill step is ~233 ms; eager fallback is likely in mixed steps | 0-5 % (chunk); 0-8 % (graph) | Low | #1 rank 6, #2 rank 7 |
| **P2** | Boot / FlashInfer JIT cache (108 s; check ABI / arch / flags before reuse), MTP FP8 KV scale calibration (AL 2.833 vs 2.957), GDN / RecoverSSM metadata, SGLang QSA / NEXTN fusion research, local-argmax draft port | Small or research only | 0.5-5 % each | Low-high | #1 ranks 3 / 7 / 8 / 10, #2 rank 8 |

## C. Notebook plan (merged from both audits)
- **NB-S1 Stage-1 compute (25-45 GPU-min plus boot).**
  - Arms: Q baseline A/A; stage-1 timing instrumentation; post-target graph capture; humming W2 / W3 shape bench.
  - Streams 24 / 28 / 30 at 20k / 40k plus a 90k band (the user rule: 90k and ~120k bands in every speed test).
  - Gates: greedy parity, probabilistic distribution parity, a tick reduction outside A/A noise.
- **NB-S2 KV capacity + prefix replay (30-50 GPU-min plus boot).**
  - Arms: F = 6 baseline; KV ledger instrumentation; F = 4 / 5 with live-block admission; the shared-prefix flag; the generated-tail checkpoint port when ready.
  - Bands: 40k / 90k / 116k.
  - Gates: 0 preemption, no stale state, greedy token-ID parity, re-prefill token reduction measured.
- **NB-S3 Combined real Duck (45-75 GPU-min plus boot).**
  - Q baseline vs the S1 / S2 winners; DFlash2 only if NBDF-K100 passes.
  - Matched task set and time cutoff. Metrics: generated tok/s, solved tasks, tokens per solve, prefix hit, preemptions.
- **Gate rule for all three (user rule after NBDF):**
  - Only real failures gate an arm: crash, oracle / parity fail, real fallback, missing init.
  - Labels and read-backs are recorded only and never skip arms.

## D. Quality note (audit #2)
- In K100Q, K100 used 1.44× the tokens of W4 on the same 15 tasks.
- Every speed winner must also report solved tasks, coverage and tokens per solve.
- The E28 K100 vs W4 6-h runs answer this at EVAL120.

## E. Do-not-repeat list (unchanged)
TP4 / PP4 / DP, live-KV offload, forced LL128, P2P / IPC all-reduce, BF16 SSM, MTP K2, Marlin config tuning, prefix caching off with MTP, partition 26 / 22 at the base pin, `--enable-hc-sp` with PP, SGLang EP4.

## F. Accepted corrections
- K100Q: 23 / 30 solved, 26 / 30 finished (audit #1).
- 385 tok/s is a 21-worker window metric (audits #1 and #2).
- Adding up small % gains does not reach 1500 (audit #2). The intermediate milestone is 500-700 tok/s. 1500 needs structural levers: DFlash2, many more resident streams via KV (F / Mamba padding), stage-1 rebalance.
