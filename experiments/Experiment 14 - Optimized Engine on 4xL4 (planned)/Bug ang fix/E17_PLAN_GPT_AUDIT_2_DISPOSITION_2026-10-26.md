# Disposition of GPT audit 2 on PLAN_E17 (2026-10-26)

The audit is saved verbatim in `GPT_E17_PLAN_AUDIT_2_2026-10-26.md`. Audit 1's disposition is in `E17_PLAN_GPT_AUDIT_1_DISPOSITION_2026-10-26.md`.

Audit 2 agrees with audit 1 on the core points:
- port the upstream small-slot repro first;
- natural pressure (env = 0) is primary and the forced-fail flag is secondary;
- full-vocab KL via `token_ids_logprob`;
- lifecycle tracing;
- a separate DFlash correctness control.

Audit 2 got the `token_ids_logprob` point right; audit 1 called it NO-GO. Below, only new claims are listed, with local checks.

## 1. New claims checked

| # | Claim | Check | Disposition |
|---|---|---|---|
| N1 | #43559 (lock release before donation retry) merged 2026-10-18; it depends on #43567; its repro uses the same 128 / 255 / 32 four-slot setup | PR page: "merged ... Oct 18, 2026", commit ba3fc94. Repro: "warm a 128-token prefix, then extend it to 255 tokens and generate 32 tokens ... exactly 4 Mamba slots and 1 running request". | **ACCEPT.** In the small-slot repro, a patched (#43567-only) server may hit `AssertionError: Can not alloc mamba cache`. That is a crash-path outcome, not a correctness fail. Add arm **B2 = #43567 + #43559** only if B asserts; it needs a backport (not yet written). |
| N2 | #43387: misaligned mamba checkpoint after spec truncation (no_buffer + NEXTN) | PR merged 2026-10-17 (86e67f7). Fix: "require the live state length to match `owned_kv_len()` before donation". The 0.5.21 `prepare_for_caching_req` (`mamba.py:539-575`) has **no such check**: `owned_kv_len` is used only in `common.py:313-321` (the KV insert), not in the mamba donation. | **ACCEPT as not on the production path.** Production uses `extra_buffer_lazy`, which donates the tracked checkpoint, not the live state. **But our engine-matrix ReplaySSM arms (`mtp_replay_fp16` / `fp32`, RISKY) use no_buffer + MTP**, so they carry #43387 (and #43749) risk. If ReplaySSM ever wins on speed, it must not be adopted without these fixes. |
| N3 | Wrong DFlash taps (#37475); expected `[5,19,33,47,61]` | The issue is open; its fix PR #37476 is open. 0.5.21 `dflash_utils.py:574-589` falls back to `build_target_layer_ids` when `target_layer_ids` is missing. The HF `z-lab/Qwen3.8-27B-DFlash2` `config.json` **does** carry `dflash_config.target_layer_ids = [5,19,33,47,61]` (also block_size 8, `num_target_layers` 64, vocab 248320). | **ACCEPT as a Phase 0 assertion:** read our Kaggle copy's `config.json` and fail closed if the ids are not exactly these. Note: wrong taps only lower acceptance (speed). The target still verifies every token, so this is not an accuracy bug. Priority P2 is right. |
| N4 | S12 ("same engine and flags, so #43567 cannot create an eval-to-LB gap") is too categorical | Fair. LB = 240 hidden tasks vs eval 120. The admission mix, cache occupancy and request churn differ, so the trigger rate can differ. | **ACCEPT.** S12 is softened to: "#43567 lowers eval and LB alike to first order; second-order differences in trigger rate are possible, and unmeasured." |
| N5 | Six cold launches ≈ 52-56 min of startup | Our own logs disagree. In the RISKY v1 run (`live_foysal_v1.log`), the first launch was 522 s. **The 7 later relaunches in the same session were 71-97 s** (target_only 76, mtp 91, pr45151 71, ntn_int4 76, ntn_fp8 91, replay 76 / 76): weights in `/dev/shm`, JIT warm. In SAFE (`live_hamid_safe.log`), one relaunch was still loading at 540 s when the log was grabbed (cause unknown). | **PARTLY WRONG.** Budget: first launch about 9 min, then about 1.5 min per relaunch, plus a reserve for one slow relaunch. 8 launches ≈ 9 + 7 × 1.5 + 9 ≈ 29 min. The 60-90 min session is feasible for T1a / T1c / T2. T3 is budget-permitting. |
| N6 | Controls: target-only / DFlash × radix on / off; single request, then 16 concurrent | — | **ACCEPT** for T3b (the DFlash correctness control, independent of #43567). |
| N7 | Fixed-token vs fixed-wall-clock paired ARC comparisons are both needed | — | **ACCEPT** for the later ARC stage (not E17). |
| N8 | `/flush_cache` before every scored cold probe; confirm idle, success and `cached_tokens == 0` | — | **ACCEPT.** Cold probes assert `cached_tokens == 0`, and warm probes assert the expected hit length. |
| N9 | Per-TP-rank logs; no per-step JSON; a startup self-test event proves the hook is live; an instrumentation on/off A/A | — | **ACCEPT.** The counters live only in the scheduler process (rank 0 scheduler). Add the self-test and the A/A. |
| N10 | The DFlash max_new budget formula `2 × interval + 2 × draft + 16` | — | **ACCEPT** as the initial budget for T1c. Primary method: choose prompt lengths so the boundary falls 1..2×draft after the prompt end. |
| N11 | FP8 KV scales: the warning is unreliable (#31224); scales may be silently dropped (#39942) | UNVERIFIED by me. | **ACCEPT** as a Phase 0 check: for each full-attention layer, record the checkpoint scale vs the runtime `k_scale_float` / `v_scale_float`. GDN layers are excluded. |
| N12 | #35150 / #36548 (DFlash verify vs decode mismatch without cache); #43514 (DFlash acceptance collapse, repeated output) | UNVERIFIED by me. | **ACCEPT as T3b + run monitors:** acceptance length, repeated-token rate, empty outputs. |
| N13 | T3 split: `LONG_PREFILL_PARITY` vs `LONG_INCREMENTAL_PARITY` | — | **ACCEPT** (same as audit 1's T3a / T3b). The incremental one is the expensive one and runs only if time allows. |
| N14 | fp16 vs fp32 SSM: first measure the target-only cold A/A drift in each dtype | — | **ACCEPT.** |

## 2. Merged plan changes (audits 1 + 2), for PLAN_E17 rev 2

**Phase 0 (provenance and safety asserts):**
- sha of `schedule_batch.py` and patch state;
- the DFlash `target_layer_ids == [5,19,33,47,61]` assert;
- per-layer FP8 KV scales;
- the vocab size from the config;
- the hook self-test event.

**T1a** (small-slot, upstream Class 1 ported, target-only):
- arguments: `--mamba-radix-cache-strategy extra_buffer_lazy --mamba-track-interval 128 --max-mamba-cache-size 4 --max-running-requests 1 --chunked-prefill-size 2048 --cuda-graph-max-bs-decode 1`, plus our numerics (INT4, FP8 KV, fp16 SSM, TP4);
- environment: `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=0`, `SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=0`;
- servers: A unpatched, B with #43567 only;
- sequence: upstream's warm `[44]*128`, warm `[42]*128`, then reference / seed 32 / reuse on `[42]*128+[43]*127`;
- full-vocab KL (threshold `1e-4`) plus the lifecycle trace;
- a variant with interval 256 (the production default).

**T1b:** env = 1 forced fail. A secondary negative-branch check only.

**T1c** (DFlash W6, the same small-slot setup):
- prompts placed so the boundary is 1..2×draft after the prompt end;
- trace `in_window`, `has_pending`, `crossed`, `planned_pos`;
- A vs B;
- plus upstream Class 2 (no_buffer + NEXTN) as a control, if MTP is loadable.

**T2** (production flags, 16 streams, Poetiq-shaped shared task prompts):
- r1-r4 counters and the four-verdict classifier;
- for each shared-prefix node, a warm vs cold (flush, `cached == 0`) full-vocab probe.

**T3b** (DFlash correctness without cache): target-only vs DFlash, radix off, single request then 16, fixed `input_ids`, greedy; record the first divergence.

**T3a** (only if time allows): long-prefill parity, fp16 vs fp32 SSM, each with its own A/A.

**Not in E17:** #43559 backport (only if B asserts), 0.5.22 stock, the paired ARC stage.

**Verdict ceiling:** `ENGINE_CORRECTNESS_VALIDATED`.
