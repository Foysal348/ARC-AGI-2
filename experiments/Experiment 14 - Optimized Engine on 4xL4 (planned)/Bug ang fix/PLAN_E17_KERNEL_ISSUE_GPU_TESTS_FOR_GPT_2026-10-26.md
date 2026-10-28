# E17 plan: GPU tests for the SGLang silent-accuracy suspects (for GPT review, 2026-10-26)

Context: `KERNEL_ISSUE_40_TO_66_RESEARCH_PACK_FOR_GPT_2026-10-26.zip` (you already have it). Four independent AI audits were
read in full and their claims checked locally (CPU only). This README gives:
- (1) what is settled;
- (2) what is open;
- (3) the GPU test plan we intend to run;
- (4) the questions we want you to answer before we build it.

**Please answer the questions in section 5, and correct the plan where it would not catch the bug.**

Stack under test:
- SGLang 0.5.21;
- RedHat Qwen3.8-27B INT4 (W4A16 g128, Marlin) and the BF16 DFlash2 drafter (width 6);
- TP4 on 4x L4 (sm_89, PCIe, NCCL SHM);
- FP8 e4m3 KV, `--mamba-ssm-dtype float16`, `--mamba-radix-cache-strategy extra_buffer_lazy`, `--max-mamba-cache-size 68`, 16৮ running;
- chunked prefill 4096, CUDA graphs 1..17, overlap scheduler on, radix cache on.

The exact flags are in the pack's live log, `engine:` line.

## 1. Settled (verified locally or by our own logs)

| # | Item | Result | Evidence |
|---|---|---|---|
| S1 | PR #43567 (stale lazy Mamba checkpoint) is present in our 0.5.21 | CONFIRMED | `srt/managers/schedule_batch.py:3464-3493` (`mamba_lazy_prealloc_at_boundary`) and `:3495-3533` (`mamba_lazy_spec_prepare`). `alloc(1)` has no evict-retry and no invalidation on failure. The speculative function is the path DFLASH and EAGLE take. |
| S2 | Our backport is exact | CONFIRMED | `e14_sglang_patches.apply(pkg, ("43567",))` on a copy of the real 0.5.21 file adds exactly two `else: req.kv.mamba_last_track_seqlen = None` hunks (:3493, :3529). It compiles, and a second apply says "already applied". |
| S3 | Production applies **no** SGLang patch | CONFIRMED | The E15P notebook has 0 `E14P.apply` calls. `DEFAULT_NAMES = ("44401", "41872")`. |
| S4 | The `None` path after the backport is not new | CONFIRMED (source) | `mamba_last_track_seqlen` starts as `None` (`schedule_batch.py:1990`) and is reset to `None` after every unfinished donation (`unified_cache/components/mamba.py:659`). `prepare_for_caching_req` skips an unfinished request with `None` and clones the keep slot with length 0 for a finished one. Production already takes this path routinely. What a length-0 insert does downstream was not traced to the end. |
| S5 | Logged "mamba usage" cannot show exposure | CONFIRMED | `pool_stats_observer.py:270-276`: `mamba_usage = (size - available - evictable) / size`. The bug needs `available == 0`, which can hold while usage reads 0.65-0.73 (evictable = radix-cached checkpoints). |
| S6 | A deterministic trigger exists in 0.5.21 | CONFIRMED | `envs.SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL`: both functions take the failure branch when it is set. |
| S7 | PR #43559 crash (`AssertionError: Can not alloc mamba cache`) | Present in source; **never observed** | `mamba.py:214` and `:509`. 0 occurrences in all 62 of our real SGLang server logs. It depends on #43567; no backport planned. |
| S8 | SGLang #42917 (stale CUDA driver cache) | NOT applicable | `CUDA_CACHE_PATH` is a fresh `/tmp/e11-cache/cuda` per VM; `_NEVER_SHIP = ("sglang/nv", "cuda")`; production restores no JIT cache. |
| S9 | Rerun-only notebook paths | Checked; no defect found | The `RERUN_ALL` profile has the same per-task time as the eval profile. The `if RERUN` submission write (cell 8 l.232), tested with the notebook's own functions: lengths, keys, JSON and fallback are all correct. The `SGL_GIVE_UP_AT` 240 vs 75 min only matters on repeated launch failure, which was never observed (launches ready in 517-555 s). |
| S10 | The five unfenced programs | 0 full solves | Two audits agree, and so does our C_ours. |
| S11 | The R5 eval-to-LB gap is mostly coverage | Agreed by all audits | 48.61 x about 0.49 = 23.8 vs LB 21.39. R5 ran on **vLLM**, so SGLang bugs cannot explain it. |
| S12 | #43567 cannot explain an eval-to-LB gap by itself | Reasoning | Our eval (commit) and LB (rerun) use the same engine and flags, so the bug would lower both. It could explain SGLang < vLLM: P1-60K SGLang LB 29.17 vs S1-60K vLLM LB 34.58, confounded by the prompt. Duck (multi-turn, a long prefix restored every turn) is the most exposed. |

## 2. Open (needs GPU)

- O1: Does the #43567 stale restore actually corrupt Qwen3.8-27B INT4 on L4? Upstream showed KL 0.649 on Qwen3.5-0.8B / GB300, ordinary decode only, and **not the speculative path**.
- O2: How often does `alloc(1)` fail naturally in our production load: 16 streams, 40-90k contexts, 76-93 % prefix hits?
- O3: Does a stale restore change ARC outcomes (solve rate), not only logprobs?
- O4: DFlash W6 at 16-way concurrency on long contexts vs target-only (#36548 class).
- O5: fp16 vs fp32 SSM at 40-90k positions.
- O6: Are the FP8 KV calibrated scales live at runtime?

## 3. Proposed E17 GPU notebook (one session, about 60-90 min, 4x L4, engine only)

**Phase 0: provenance (minutes).** Log:
- installed `schedule_batch.py` sha256;
- patch state (`E14P.check`);
- effective launch flags;
- `CUDA_CACHE_PATH`;
- per-layer `k_scale_float` / `v_scale_float` on TP0 after load (a sitecustomize hook) → O6.

**Instrumentation (all servers).** A sitecustomize hook in the SGLang venv, active only in the scheduler process:
- wrap `ScheduleBatch.mamba_lazy_prealloc_at_boundary` and `mamba_lazy_spec_prepare` so they count calls, boundary
  attempts and **failed `alloc(1)`** (`new_slot is None`);
- sample `mamba_allocator.available_size()` and `tree_cache.mamba_evictable_size()` every decode interval;
- write the counters to a JSON file every 10 s. → O2.

**T1: forced-fail repro (#43567, deterministic; upstream pattern, scaled to our model).**
Servers, all with production flags; each started fresh, cache flushed:
- A: unpatched, `SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=1`;
- B: patched (`E14P.apply(("43567",))` before start), same env;
- C: unpatched, env off (control).

Each server runs target-only (speculation off) and DFlash W6:
1. Cold: a fixed prompt P (about 2-8k tokens, real ARC text), max_new 1, logprobs at the last position.
2. Seed: a request sharing P's prefix that decodes past several `mamba_track_interval` boundaries, so a checkpoint is donated under failure.
3. Warm: P again (prefix hit), the same logprob read.
4. Repeat for 8 prompts; plus cold-vs-cold A/A.

Metrics: KL over the top-k union (k as large as the API allows), max |Δlogprob|, top-1 change, the prefix-hit length (`#cached-token`), and the hook's failure count.

Expected: A shows KL >> A/A; B and C show KL ≈ A/A.

**T2: natural pressure (production DFlash W6, unpatched, then patched).**
- 16 concurrent streams, Poetiq-shaped: a 30-60k shared context, continued turn by turn with appended feedback so every turn is a prefix hit. Run for about 10-15 min per server.
- Read the hook's failure count and `available_size` trace → O2.
- At the end, for 8 sequences: logprobs of the next 256 tokens of a stored continuation, warm (cached) vs cold (after flush). Compare against the cold-vs-cold A/A → O1 under real load, DFlash path included.

**T3 (if time): DFlash vs target-only long-decode parity at 16-way.** Teacher-forced scoring of stored 40-60k continuations: the NLL difference by 8k bucket → O4. Optionally the same with `--mamba-ssm-dtype float32` → O5.

**Decision rules.**
- T1-A KL > A/A tolerance and T1-B ≈ A/A: the bug is real on our stack.
  - T2 failures > 0 with warm/cold divergence: it fires in production.
  - Then run a paired EVAL subset (the same commit, unpatched vs patched) to measure solve-rate impact (O3), and adopt the patch.
- T1-A ≈ A/A: the path is not reachable as modelled, or our test misses it. Ask why before concluding.
- No LB uplift is promised in advance.

## 4. What the audits disagree on or leave vague (our reading)

- **mikelou1's cause:** all four audits say it is unknown. Its value to us is only "a tiny fix can recover a third of the score" as a pattern.
- **26+ point loss from #43567:** "technically possible" (the 4th audit), but there is no evidence in our data. It is a hypothesis to test, not a projection.
- **Full-vocabulary KL:** two audits ask for it. Our SGLang `/generate` returns top-k logprobs, not the full vocab. **How do we get full logits cheaply in 0.5.21?**
- **`top_logprobs` vs the post-sampler distribution** (3rd audit): our nucleus-audit idea may be invalid. We dropped it from T2/T3.

## 5. Questions for you

1. **Trigger design:** does T1 (seed request + `SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL` + warm repeat) reach the exact stale-donation path of #43567? What exactly must happen between "seed" and "warm" for a stale checkpoint to be donated (overlap scheduler timing, track-interval alignment, request finishing vs unfinished, prefill result processing)? Give the minimal deterministic recipe for SGLang 0.5.21.
2. **Speculative path:** upstream's E2E covered only ordinary decode. How do we make the DFlash / EAGLE `mamba_lazy_spec_prepare` failure produce a stale donation that a later prefix hit restores? Which requests and how many tokens?
3. **Measurement:** what is the best signal in 0.5.21 for "a stale state was restored": logits KL; comparing the restored state tensor with a recomputed one (can we read the mamba state of a radix node?); or something else? Is there an internal assertion or debug flag that checks checkpoint consistency?
4. **Full-vocabulary logits:** the cheapest way in SGLang 0.5.21 (an API field, `return_logprob` + `top_logprobs_num` = vocab, `token_ids_logprob` over all ids, or a server-side hook)? What does each cost at 248k vocab?
5. **Natural frequency (O2):** are our hook points (`mamba_lazy_prealloc_at_boundary` / `mamba_lazy_spec_prepare`, `mamba_allocator.alloc`, `available_size`) the right place to count real failures? Is anything else needed to tell "failed and later donated stale" from "failed but harmless"?
6. **Patch safety:** with the two `else` hunks only (no #43559), is there any new failure mode in 0.5.21, e.g. the length-0 insert for finished requests in `prepare_for_caching_req`? Should we instead test SGLang 0.5.22 as a whole (it contains #43567 and other mamba fixes), or both?
7. **ARC impact (O3):** what is the cheapest experiment that turns "KL > 0" into a solve-rate difference? A paired Duck / Poetiq EVAL subset, with how many tasks to beat our noise (LB sd about 1.5; eval task-normalised noise unknown)?
8. **Anything we miss:** other mechanisms in our exact config that could silently corrupt long multi-turn reasoning and that the same session should test, with the cheapest test for each.

Please keep every URL literal (`https://...`). Mark anything you cannot verify as UNVERIFIED.
