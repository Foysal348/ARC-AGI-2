# Experiment priority list toward 1500 tok/s, NB-K100 baseline (v1, 2026-11-01)

**Inputs:**
- GPT deep-research audit #1 (the K100 → 1500 audit);
- our measured data (`E25_1500_PROGRESS_REPORT_2026-11-01.md`);
- a Merlin source check (ad42e37) done today.

Two more GPT audits are coming, and this list will be updated after each.

**Baseline:** NB-K100 (klee100 3bpw-MTP, Q-defer, I8, 24 workers, MTP K3) at 385 tok/s window average on real Duck (K100Q). The 6-h EVAL120 run is in progress.

## Merlin source check (verified 2026-11-01)
| GPT lever | In Merlin? | Applies to Qwen4Exp? |
|---|---|---|
| `--enable-mamba-shared-prefix-checkpoint` (vLLM #52789) | **YES**, `config/cache.py:198`; also in `kv_cache_manager.py`, `scheduler.py`, `single_type_kv_cache_manager.py` | Conditions: mamba_cache_mode align (ours ✓), prefix-match-unit 64 < block 4096 (✓), "EAGLE on the Mamba group" (UNVERIFIED for Qwen4Exp MTP; verify A U4 says Qwen4Exp groups are not eagle-annotated). **A flag test.** |
| Batch-sharded sampling (vLLM #50465) | **YES**, `config/parallel.py:192` `enable_batch_sharded_sampling`, `model_runner.py` | Needs the model to implement `compute_logits_local`. **Qwen4Exp does NOT** (DeepSeek-V4 / V41 do, a template exists). **A port.** |
| Local argmax reduction for the draft | **YES**, `config/speculative.py`, `llm_base_proposer.py` | Implemented for DeepSeek-V32 / V4 / GLM5 MTP only, **not Qwen4Exp MTP**; needs greedy draft sampling. **A port.** Moot if DFlash2 replaces MTP. |

## Priority list
| # | Experiment | Why | Expected (whole-run) | Cost | Status |
|---|---|---|---|---|---|
| **P0-1** | **NBDF-K100: DFlash2 on klee100** (Q / D4s / D3s, SW-1024, 8k-116k bands) | The largest single estimate (×1.24-1.34 vs MTP K3); seam, drafter and gates already built and CPU-verified on W4 | −20 % to +34 % (acceptance on the low-bit target is unknown) | Build + 150-min GPU | In build; 2 research agents running (engine / memory / boot, acceptance / design) |
| **P0-2** | **Shared-prefix Mamba checkpoint flag** (`--enable-mamba-shared-prefix-checkpoint`) | 58 % of uncached tokens are previous-turn output re-prefilled; the flag already exists in Merlin | +2-8 % (GPT); TTFT gain larger | One flag plus a correctness gate (greedy oracle / Duck replay cold vs warm, Mamba restore position) | Not started; first check its eagle-group condition on Qwen4Exp |
| **P0-3** | **R1 profiling + capacity sweep on klee100** (no code change): 24 / 28 / 30 streams at 8k / 40k / 90k; PP stage tick split (LM head, AllGather, drafts), eager fallback, KV high-water, preemption | Sizes every stage-1 lever; 24 → 28-30 may add up to +10 % | 0-10 % | 1 short GPU notebook | Not started |
| **P1-1** | **Batch-sharded sampling**: port `compute_logits_local` to Qwen4Exp from the DeepSeek-V4 template | Stage 1 binds after Q-defer; full-vocab AllGather 2.9 ms per tick | +3-10 % (GPT screening range) | Port + oracle | Not started |
| **P1-2** | **Mixed-prefill CUDA graph coverage read-back** (log capture key / eager fallback per step), then targeted capture sizes | Prefill chunks (1024) may run eager with decodes | 0-8 % | Read-back first, low cost | Not started |
| **P1-3** | **SGLang Flash-Next QSA / NEXTN metadata fusions** (PRs 41173 / 41175 / 40972 / 41166-41171) | Model-specific; the gains reported are combined series on B200 | +1-6 % | High (SGLang → vLLM V2 mapping) | Research only |
| **P1-4** | Local argmax draft for Qwen4Exp MTP | Fewer full-vocab gathers in drafting | small; moot if DFlash2 wins | Port + greedy draft | Deferred until P0-1 |
| **P2** | MTP FP8 KV scale calibration (AL 2.833 vs 2.957 BF16); GDN / RecoverSSM metadata; humming W2 g64 tile tuning; boot cache (FlashInfer JIT 108 s) | | 0-5 % each; boot +0.5-1.2 % | Medium | Later |

## Required correctness gates for any of these (audit #1, section 9)
- Greedy token-ID parity vs the baseline at 8k / 40k / 90k / 116k × 1 / 8 / 24 streams.
- PP recurrent-state commit parity (SGLang #40001: PP2 + MTP had normal acceptance but wrong outputs).
- Mamba restore-position checks.
- Graph memory vs KV pool at a realistic 104k occupancy.
- Only real-failure gates (the user rule after NBDF): labels are recorded only.

## Do-not-repeat list
TP4 / PP4, live-KV offload, forced LL128, P2P, BF16 SSM, MTP K2, Marlin tuning, prefix caching off, P26.

## Corrections from audit #1 (accepted)
- K100Q: 23 / 30 solved, 26 / 30 finished. These are different metrics.
- 385 tok/s is the 6,600-s window metric, not a 12-h whole run.
