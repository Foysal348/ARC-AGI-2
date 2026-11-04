# Experiment priority list toward 1500 tok/s, NB-K100 baseline (v3 FINAL of this round, 2026-11-01)

**Inputs:**
- GPT audit #1 ("K100 → 1500 tok/s: Deep Research Audit");
- GPT audit #2 ("E25 — 1500 tok/s Research Audit (K100 baseline)");
- GPT audit #3 ("E25 — 1500 tok/s Research Audit: K100 Baseline", GO for research);
- our measured data (`E25_1500_PROGRESS_REPORT_2026-11-01.md`);
- a Merlin source check (ad42e37).

v1 and v2 are kept for reference.

**Baseline:** NB-K100 (klee100 3bpw-MTP, Q-defer, I8, MTP K3).
- **K100Q** (21 workers, `solver_tok_s` 385.0, 0 preemptions, max_running 21): 23 / 30 solved, 26 / 30 finished.
- **E28 K100** (24 workers, 6 h EVAL120): running now. It sets the real 24-worker baseline.
- **Production stays unchanged** (all 3 audits). The research runs in separate notebooks.

## A. Merlin source check of every upstream item named by the 3 audits
| Upstream item | Audit | In Merlin ad42e37? | Consequence |
|---|---|---|---|
| `--enable-mamba-shared-prefix-checkpoint` | #1 #2 #3 | **Present** (`config/cache.py:198`) | Junction checkpoints are written only inside the prompt (`scheduler.py:595-605`: "Past the prompt the manager writes nothing"). It does **not** remove the re-prefill of the previous turn's generated tokens. Audit #3 independently says the same. It is a side arm. Eligibility: MTP flags all groups as eagle (`kv_cache_coordinator.py:134`); single-module MTP needed (`use_multi_module_mtp`: `min(num_nextn_predict_layers, K) > 1` → check `num_nextn_predict_layers` in the klee100 config). |
| Generated-tail SSM checkpoint | #2 #3 | **Absent** (no decode-tail publish) | **A port.** This is the lever for the 58 % re-prefill. |
| Batch-sharded sampling, vLLM #50465 | #1 #3 | **Present** (`config/parallel.py:192`) | Qwen4Exp has **no** `compute_logits_local`: present only in deepseek_v4 / v41, minimax_m3 and llama, none in `models/qwen4_exp/`. The flag alone would fall back silently. Upstream gain +1.09 % (audit #3), so the conservative expectation is 0-3 %. |
| Qwen4Exp sharded-sampling integration, vLLM #61018 | #3 | **Absent** (no `compute_logits_local` in qwen4_exp) | The best port starting point: cherry-pick / adapt the #61018 diff onto Merlin's qwen4_exp + MRV2 + PP handoff. |
| MTP local-argmax / fused trailing all-reduce, vLLM #49793 | #1 #3 | **Partly**: `use_local_argmax_reduction` in `config/speculative.py:459` and `llm_base_proposer.py:440`; model paths only in deepseek_v32 / v4 and glm5next MTP | **A port** for qwen4_exp MTP. It is greedy-draft only, and our production draft is probabilistic, so it is a greedy diagnostic arm only. Inspect the #49793 diff for the fused trailing all-reduce part, which can apply to probabilistic drafts. |
| EAGLE / MTP lookahead block hashing, vLLM #61589 | #3 | **Equivalent logic present**: unified `num_prefill_lookahead_tokens` property (`config/vllm.py:674`, "Consumers must read this property … so those components cannot drift apart"), `num_reprefillable_tokens`, `drop_eagle_checkpoint_block` (14 uses). A line-by-line diff vs #61589 is not done | Our measured prefix hit is 85.4 %. Only the chained-turn test (10 turns + siblings, audit #3) can show residual waste. That test goes in NB-S2. |
| Contention-aware lone long prefill, vLLM #57951 | #3 | **Present**: `long_prefill_token_threshold` ("not applied when the request is the only one in the batch") + `long_prefill_token_threshold_adaptive` (`config/scheduler.py:79-90`) | Config-only arms. Audit #3 warns about the #60476 regression; with 24 Duck workers a lone prefill is rare. |
| DFlash2 fixes #60765 / #61180 / #64059 | #3 | Already ported in the E27 seam (taps + PP2 relay, #64059 in-graph metadata) | No new DFlash engine; validate E27 on K100 (NBDF-K100). |
| Humming (vllm-project/humming, blog 2026-10-21) | #1 #2 #3 | In use (klee100 W3 g128 / W2 g64 on layers 12-35) | Missing evidence: K100's own W2 / W3 operator timings. |

## B. Capacity bound (audit #3, checked against our numbers)
Inputs: 479 blocks, F 6, block 4096 → blocks per stream = ceil(ctx / 4096) + 6.

| ctx | blocks / stream | Max resident | Note |
|---|---|---|---|
| 20k | 11 | 43 | |
| 40k | 16 | 29 | Matches our "resident bound 29 at 40k" |
| 90k | 28 | 17 | Below 24 workers |
| 120k | 36 | 13 | Below 24 workers |

Our T1 attempt limit is 104k and B2 is 120k. Under that workload, 24 workers cannot all be resident when the contexts are long. Two consequences:
- **F 6 → 4** gives only 17 → 18 streams at 90k. The larger capacity levers are Mamba page padding (21 %) and the drafter KV size.
- Admission must use live blocks, as audit #3 proposes: `running_required + reclaimable_cached + free`, plus the scheduler token horizon. A KV usage of 99 % is not the same as running-owned.

## C. Final priority list (v3, merged ranking of the 3 audits)
| # | Experiment | Expected | Confidence | Cost | Audits (rank) |
|---|---|---|---|---|---|
| **P0-1** | **NBDF-K100: DFlash2 on klee100** (existing E27 port; Q / D4s / D3s K3 / K4, SW-1024; 8k / 40k / 90k / 116k). Promotion metric: completed output tok/s at the same Duck workload and realistic concurrency vs MTP; reject if single-stream wins but the 24-worker aggregate loses | −20 % to +30 % | Low-medium | High (build in progress) | #1 (5), #2 (4), #3 (2) |
| **P0-2** | **NB-S1 stage-1 profiler at 16 / 24 workers on K100**: timestamped timeline of stage-1 p50 / p95, target head, 3 draft heads, AllGather, eager post-target kernels, graph FULL / PIECEWISE / EAGER per batch size. This is the gating measurement for P1-1 / P1-2 / P1-5 | sizing | High | Low (20-30 min) | #1, #2 (1), #3 (1, 3, 9) |
| **P0-3** | **NB-S2 KV ledger + live-block admission + 24 / 28 / 30 / 32 sweep** at 20k / 40k / 60k / 90k: running, reclaimable, free blocks; preemption; fresh-prefill tokens; 10-turn chain + sibling prefix test (re-prefill, #61589-equivalence); F 6 vs 5 / 4 | 0-15 % | Medium | Low-medium | #1 (4), #2 (2, 6), #3 (5) |
| **P1-1** | **Qwen4Exp sharded sampling port** (#61018 + #50465 on MRV2) | 0-3 % conservative (upstream +1.09 %), up to ~4.4 % max (the whole 2.9 ms AllGather) | Medium | Medium-high | #1 (2), #2 (1), #3 (1) |
| **P1-2** | **MTP draft-head path**: #49793 fused trailing all-reduce; LM-head weight reuse / fewer launches; overlap with target verify; greedy local-argmax as diagnostic only. Moot if DFlash2 wins | 3-9 % | Medium | Medium | #3 (3), #1 (7) |
| **P1-3** | **Generated-tail SSM checkpoint port** + shared-prefix flag side arm. Gates: token-ID identity, checkpoint position, recurrent-state + KV equivalence, PP0 / PP1 fence; hit up but oracle changed = NO-GO | 2-6 % | Medium | High | #1 (1), #2 (3), #3 (4) |
| **P1-4** | **Humming W2 g64 / W3 g128 shape bench** (T 8-64, real captured routes, graph replay), then dispatch / heuristic tables; kernel-only branch first | 0-10 % | Low-medium | Medium | #1 (9), #2 (5), #3 (6) |
| **P1-5** | **Post-target graph capture / fusion** (~4.2 ms eager) once P0-2 shows which kernels are outside graphs and why | 2-8 % | Medium-low | High | #1 (6), #2 (1), #3 (9) |
| **P1-6** | **Long-prefill**: 1024 baseline vs `long_prefill_token_threshold` + adaptive (contention-aware 2048 max). Metric: whole-run tok/s + p95 ITL | 1-6 % | Medium-low | Low (config) | #2 (7), #3 (7) |
| **P2** | MTP FP8 KV scale calibration (AL 2.833 vs 2.957); boot / JIT cache (108 s, ABI-checked); GDN / RecoverSSM metadata; SGLang QSA / NEXTN fusion research | 0.5-4 % each | Mixed | Low-high | all |

## D. Notebook plan (all 3 audits agree on three notebooks; merged)
- **NB-S1 Stage-1 profiler + sampling (20-45 GPU-min + boot).**
  - K100 baseline at 16 / 24 workers; timeline + graph read-back.
  - Later arms: the sharded-sampling port and the MTP head path once built.
  - GO: ≥ 3 % sustained, outside A/A noise, oracle and probabilistic distribution parity, no new preemption.
- **NB-S2 KV + prefix (20-50 GPU-min + boot).**
  - 24 vs 28 / 30 / 32 with live-block admission; F 6 / 5 / 4; chained-turn + sibling test; shared-prefix flag; the generated-tail port when ready.
  - Bands 40k / 60k / 90k / 116k.
  - GO: ≥ 5 % matched workload, 0 sustained preemption, stable quality. Fewer workers can win if the contexts are 90k+.
- **NB-S3 Speculative + kernel decision + combined real Duck (30-75 GPU-min + boot).**
  - MTP K3 vs DFlash2 K3 / K4 (only if NBDF-K100 passes); humming W2 / W3 diagnostic; S1 / S2 winners combined.
  - GO: ≥ 8 % whole-workload throughput, no quality regression (solved tasks, tokens per solve), memory safe.
- **Gate rule for all (user rule after NBDF):**
  - Only real failures gate an arm: crash, oracle / parity fail, real fallback, missing init.
  - GO thresholds are decided after the run from the recorded numbers. They never skip arms during the run.
  - Every speed test includes the 90k and ~120k bands.

## E. Do-not-repeat list (all 3 audits agree)
- Forced LL128, P2P, TP4 / PP4 / DP, BF16 SSM, prefix caching off with MTP, live KV offload, MTP K2, generic Marlin sweep.
- Re-running Q-defer or INT8 draft head; both are adopted.
- PP 26 / 22 only as a controlled memory-reserved branch, and only if the 24-worker stage profile proves imbalance (audit #3).

## F. Accepted framing (all 3 audits)
- Small % levers do not add up to 1500. The audit #3 scenarios are +15 / +40 / +80 % → 443 / 539 / 693 tok/s.
- First milestone: 500-600 sustained real Duck, then 700+.
- 1500 needs a structural step: a much faster MoE path, a different speculation cost model (DFlash2), many more resident streams (capacity), or a pipeline redesign.
- Every speed winner must also report solved tasks and tokens per solve. In K100Q, K100 used 1.44× W4's tokens on the same 15 tasks.
