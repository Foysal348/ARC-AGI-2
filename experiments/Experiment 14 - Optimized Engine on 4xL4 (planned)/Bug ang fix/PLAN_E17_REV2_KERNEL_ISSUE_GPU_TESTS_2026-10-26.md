# E17 plan rev 2: GPU tests for the SGLang silent-accuracy suspects (2026-10-26)

Rev 1 is `PLAN_E17_KERNEL_ISSUE_GPU_TESTS_FOR_GPT_2026-10-26.md`. Rev 2 merges:
- three GPT audits: `GPT_E17_PLAN_AUDIT_1/2/3_2026-10-26.md`, with dispositions in `E17_PLAN_GPT_AUDIT_1/2_DISPOSITION_2026-10-26.md` and §8 below;
- the upstream regression test, https://raw.githubusercontent.com/sgl-project/sglang/main/test/registered/radix_cache/test_mamba_cache_kl.py;
- local checks of the 0.5.21 source (`_scratch/session_1c6db7aa/sgl_src/src/sglang`).

All three audits agree that rev 1's T1 was not a decisive reproduction. Rev 2 replaces it with the upstream recipe.

**Verdict ceiling for this session:** `ENGINE_CORRECTNESS_VALIDATED`. ARC solve-rate impact is a separate, later run.

## 1. The mechanism we are testing (0.5.21 source, matches the #43567 PR text)

1. Prefill sets `mamba_last_track_seqlen` to the prefill checkpoint, aligned to the cache chunk grid (`schedule_batch.py:~3110`). For a 255-token prompt that is 192. The lazy keep slot holds that state.
2. The first decode after prefill goes 255 → 256 and crosses a track boundary.
   - `mamba_lazy_prealloc_at_boundary` (`:3464-3493`) calls `alloc(1)` with no evict-retry. It fails when `available == 0`, even if evictable slots exist.
   - The new state is then written **in place into the keep slot**, and the length stays 192.
3. Overlap scheduling: the prefill's result processing runs after that decode was launched. `cache_unfinished_req` → `MambaComponent.prepare_for_caching_req` (`mamba.py:539`) donates the keep slot under length 192.
4. A later request with the same prefix matches 192 cached tokens and restores the wrong state.

Upstream: KL 0.649 unpatched (a 192 hit) vs 0 patched (a 128 hit).

- **Spec path** (DFlash, EAGLE): `mamba_lazy_spec_prepare` (`:3495-3533`). A boundary counts as reachable if it lies within `2 × max_draft_tokens` of `kv_committed_len` (`:2187-2197`). On failure, the verify scatters in place into the keep slot. A real crossing is confirmed later, in `_mamba_lazy_spec_update` (`batch_result_processor.py:~1430-1490`).
- **Production exposure hypothesis (UNVERIFIED; T2 measures it):**
  - Production runs 68 slots with 16 running, and lazy alloc never evicts. So `available` may sit at 0 once the radix cache is full.
  - Poetiq experts and Duck turns share task-prompt prefixes. One stale donation at prompt depth would then be restored by every later request of that task.

## 2. Instrumentation (no source change; a sitecustomize monkeypatch in the SGLang venv, scheduler processes only)

The audits said a wrapper cannot see `new_slot is None`. **It can infer the outcome from the state before and after the call, without touching the source.** This keeps the unpatched arm byte-identical to production.

- **`ScheduleBatch.mamba_lazy_prealloc_at_boundary`.** Before the call, mark each request *eligible* if `seq_lens_cpu[i] % interval == 0` and `buf[1 - next_idx] == -1`. After the call:
  - pending slot still `-1` → `optional_alloc_failed` (covers natural and forced failures);
  - otherwise → success.
  - On failure, set `inplace_pending[rid] = True`.
- **`ScheduleBatch.mamba_lazy_spec_prepare`.** Before the call, record `in_window` and `has_pending`. After it, compare `mamba_lazy_spec_track_positions_cpu[i]` with `next_track_idx`:
  - equal and in window → `spec_inplace_fallback`; set `inplace_pending[rid] = True`;
  - log `kv_committed_len`, the interval and `max_draft_tokens`.
- **`BatchResultProcessor._mamba_prefix_cache_update` / `_mamba_lazy_spec_update`.** Log `mamba_last_track_seqlen` before and after, plus `crossed` and `planned_pos` (spec). When the length changes to a new value, clear `inplace_pending[rid]`.
- **`MambaComponent.prepare_for_caching_req` (entry).** Log rid, `is_finished`, `cache_len = req.kv.mamba_last_track_seqlen`, the keep-slot index and the return value.
  - **`stale_donation_candidate`** = `inplace_pending[rid]` and `cache_len is not None` and the return value is greater than 0.
  - On the patched arm, the length is already `None` at that point, so the candidate count must be 0.
- **Allocator sampling** (every 40 decode steps): `available_size()`, `mamba_evictable_size()`, the running count.
- **Restore:** `meta_info.cached_tokens` per request (external, primary; the 192 vs 128 signature). Optionally, a `match_prefix` wrapper that logs the matched depth.
- **Output and checks:**
  - an in-memory ring buffer, flushed by a daemon thread every 10 s to `/kaggle/working/e17_trace_pid<pid>.jsonl` (one file per scheduler process, so no shared writes);
  - no per-step JSON, no CUDA sync;
  - a startup self-test event proves the patch is live in every scheduler process, and a missing self-test aborts the arm;
  - instrumentation on/off A/A on T1a (one extra launch) checks that the hook does not change timing.
- **Verdicts:**
  - `BUG_EXPOSED` needs the full chain for one rid: `optional_alloc_failed` → `stale_donation_candidate` → a later request with `cached_tokens` at the stale depth → KL above tolerance;
  - an incomplete chain gives `INCONCLUSIVE`, never PASS.

## 3. Measurement

- **Full-vocab next-token distribution:**
  - `/generate` with `return_logprob: true`, `logprob_start_len: -1`, `token_ids_logprob: list(range(V))`, temperature 0, `max_new_tokens` 1, `ignore_eos`;
  - V comes from the model config (248,320);
  - assert the ids are ordered, all values finite and the probabilities sum to 1; renormalise with `logaddexp`.
- **KL(ref ‖ warm)** in float64. Also record the top-1 change and the max |Δlogprob|.
- **The tolerance comes from our own A/A,** not upstream's `1e-4`: INT4 Marlin, FP8 KV, fp16 SSM and TP4 are not bit-stable. A/A means the reference sequence run three times on a fresh flush. The tolerance is `max(5e-3, 10 × max A/A KL)`; upstream's spec test uses `5e-3`.
- **Cold probes:** idle, then `/flush_cache` returns 200, then assert `cached_tokens == 0` (or the expected warm-prefix length). Warm probes assert the expected hit length.
- **Full-vocab scoring** only at a few decisive positions (one per probe), never per token of long runs.

## 4. Session (one Kaggle 4xL4 notebook, all servers TP4 with production numerics)

Production numerics: RedHat INT4 W4A16, FP8 e4m3 KV, `--mamba-ssm-dtype float16`, `--mamba-radix-cache-strategy extra_buffer_lazy`, overlap on, decode CUDA graphs on.

Launch budget, from our RISKY v1 log:
- the first launch was 522 s;
- the 7 later relaunches in the same session took **71-97 s** (weights in `/dev/shm`, JIT warm).

Plan for about 12 launches ≈ 9 + 11 × 1.5 + 9 (reserve) ≈ **35 min**. Hard stop at 150 min.

**Phase 0 (CPU part of the notebook, before any server):**
- the sha256 of the installed `schedule_batch.py`, and `E14P.check(("43567",))` = absent;
- the DFlash2 `config.json` asserts:
  - `dflash_config.target_layer_ids == [5,19,33,47,61]` (otherwise 0.5.21 silently falls back to `build_target_layer_ids`, issue #37475);
  - `num_target_layers == 64`;
  - this was already verified once on our Kaggle drafter: the E12 rev 6d G4B preflight logged `target_layer_ids [5, 19, 33, 47, 61]` and a 2/2 parity control (`Experiment 12/Experiment Results/E12 rev 6d G4B GPU run analysis (2026-10-21).md:29`). The assert stays as a guard;
- V from the INT4 config;
- the presence of the instrumentation module;
- free `/dev/shm` and disk.

The FP8 KV scales are read after the first launch: a hook on TP0 logs `k_scale_float` / `v_scale_float` for each **full-attention** layer, and the checkpoint's stored scales if present. GDN layers are skipped.

**Unpatched block (site-packages original):**
- **L1, T1a-128:** small-slot target-only.
  - Arguments: `--max-mamba-cache-size 4 --max-running-requests 1 --mamba-track-interval 128 --chunked-prefill-size 2048 --cuda-graph-max-bs-decode 1 --max-total-tokens 8192`.
  - Environment: `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=0`, `SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=0`.
  - Sequence (upstream Class 1):
    1. A/A ×3: flush, warm `[44]*128`, warm `[42]*128`, then score `P = [42]*128+[43]*127` (`cached == 128`).
    2. Seed: flush, warm both prefixes, generate 32 from P (`cached == 128`).
    3. Reuse: score P (log `cached`: 192 = stale signature).
    4. Repeat the seed/reuse cycle ×3.
  - If 4 slots do not launch, step up 5 → 6 → 8 and log the actual available and evictable counts.
- **L2, T1a-128 with the hook off:** the instrumentation A/A (the reuse KL distribution must match L1).
- **L3, T1a-256:** as L1 but with `--mamba-track-interval 256` (the production default).
  - P = 255 tokens still crosses at 256.
  - Add `[42]*128+[43]*383` (511 tokens).
- **L4, T1b:** as L1 with `SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=1`. A negative-branch diagnostic only; it never decides anything alone.
- **L5, T1c-DFlash:** small-slot with production DFlash W6 (`--speculative-algorithm DFLASH`, draft-model BF16, `--speculative-num-draft-tokens 6`) and `--mamba-track-interval 64`. The source asserts interval ≥ draft tokens and interval % page = 0.
  - Prompts are aligned so the next boundary lies 1..12 tokens after the prompt end.
  - Seed: generate 128-192 tokens.
  - Trace `in_window`, `spec_inplace_fallback`, `crossed`, `planned_pos`, accepted length. Warm vs ref KL as in T1a.
  - Slots step 4 → 6 → 8 if launch or progress fails (DFlash may need more slots per request).
- **L6, T1c-control (if time allows):** upstream Class 2 unchanged.
  - Arguments: no_buffer, page 1, overlap and CUDA graph off, NEXTN 3/1/4 using our model's MTP head (as in the engine matrix).
  - Probe = prompt + output + `"\n"`, warm vs cold. Tolerance 5e-3.
  - This is a known-bug path (#43387, no guard in 0.5.21); it validates that our measurement harness detects a stale restore.

**Patch** (`E14P.apply(("43567",))`, sha logged, verify "applied"):
- **L7, T1a-128 patched;** L8, T1a-256 patched; L9, T1c-DFlash patched. Same sequences.
- Expect: `stale_donation_candidate == 0`, reuse `cached == 128`, KL ≤ tolerance.
- If the scheduler asserts `Can not alloc mamba cache` (the #43559 path), record `CRASH_PATH_43559` and do not count it as a correctness result.
- **Patch-safety cases** (audits 1-3), on L7 after the T1a sequence, then the same on unpatched L1 as a reference:
  - normal EOS (no `ignore_eos`);
  - a client abort mid-decode;
  - `max_new_tokens` reached;
  - a repeated prefix reuse (5 cycles);
  - a cold request after a flush.
  - After each case: idle, then assert the slot invariant `available + evictable + in_use == max_mamba_cache_size` (read from the allocator through the hook), and no growth of `in_use` across cycles (a slot leak).
  - The next warm probe must hit a valid depth (128 or a cold miss, never 192), and its KL must be within tolerance.
  - Covers the length-0 / `None` path: no dangling radix reference, the unsafe donation skipped, fallback to the shorter checkpoint, and a correct recompute on a miss.

**Decision point A** (in the notebook, automatic):
- `BUG_EXPOSED_43567` = unpatched L1, L3 or L5 has a complete chain with KL > tolerance, **and** the patched counterpart is clean.
- If `BUG_EXPOSED_43567`, run **T2**. Otherwise skip T2 and give its time to T3b.

**T2, natural exposure (conditional; production flags exactly; unpatched L10, then patched L11, 12 min each):**
- 16 streams, Poetiq-shaped.
  - 4 task prompts (30-60k tokens built from real ARC eval prompts), 4 streams each.
  - Each stream runs turns: append about 2k of feedback text, generate up to 4k, repeat. Every turn is a prefix hit on the shared prompt.
  - Temperature 0.
- Counters per 1,000 eligible boundaries:
  - `optional_alloc_failed`;
  - `stale_donation_candidate` per 1,000 prefix-hit opportunities;
  - the `available` / evictable trace.
- At the end, for each task prompt: a warm probe (`cached > 0`) vs a cold probe (flush, `cached == 0`), full-vocab.
- Four-way verdict: `NO_FAILURE_OBSERVED` / `FAILURE_OBSERVED_NO_STALE_RESTORE` / `SILENT_CORRUPTION_CONFIRMED` / `STALE_RESTORE_EXPOSED_NO_OUTPUT_FLIP`.

**T3b, DFlash verify correctness (independent of #43567; unpatched; `--disable-radix-cache`):**
- **L12, target-only:** serial (1 stream), then 16 concurrent.
- **L13, DFlash W6:** serial, then 16 concurrent.
- Inputs: the same 16 frozen ARC prompts, greedy, generating 2k tokens each.
  - Serial runs use 4 of them (a single TP4 stream decodes at roughly 30-40 tok/s, so 16 serial runs would take about 15 min per engine). Concurrent runs use all 16; the 4 shared prompts give the serial-vs-concurrent pairs.
- Compare each stream's token sequence with the serial target-only run.
  - Record the first divergence position.
  - Record the logprob margin at that position (top-2 gap from the top-5 logprobs at that step) and the acceptance length.
  - Separate concurrency effects (16 vs serial within an engine) from verify effects (DFlash vs target, serial).
  - A divergence at a low-margin position (top-2 gap < 0.05) is expected numerics. A divergence at a high-margin position is a bug signal (#36548 / #38009 class).
- Monitors: the repeated-token rate, empty outputs, acceptance collapse (#43514 class).
- Optional, if time allows: a DFlash run with CUDA graphs disabled (L14).

**T3a (if time allows): long-prefill parity.**
- Arms: fp16 vs fp32 SSM (L15, `--mamba-ssm-dtype float32`), target-only, radix off.
- Each dtype first gets its own cold A/A at the same positions.
- Then the final next-token full-vocab KL at 8k / 16k / 32k / 48k prefixes of one stored long continuation.
- If time allows: the same with `--kv-cache-dtype bf16` vs FP8 (L16), for the FP8 KV prefill-vs-decode suspect (#25790, UNVERIFIED). Each dtype gets its own A/A.

## 5. Outputs

- `e17_report.json`:
  - provenance: shas, patch state, flags per launch, ready seconds;
  - per arm: the counters, `cached_tokens` lists, A/A KL, reuse KL, tolerance and verdict;
  - T2 counters and verdict;
  - T3b first-divergence table.
- Trace JSONL files, all generated text, and the server log tail for each launch.
- Full thinking traces are saved (the standing rule), though E17 mostly uses non-thinking probes.

## 6. Decision rules after the session

- `BUG_EXPOSED_43567` and T2 exposure > 0 → the #43567 patch becomes a production candidate. It is adopted only after a paired ARC run:
  - Duck first, then Poetiq S60;
  - 24-32 exposure-rich tasks, the same seeds and budgets, patched vs unpatched;
  - McNemar on discordant pairs;
  - both a fixed-token comparison and a fixed-wall-clock comparison;
  - plus a third arm with the radix cache off (target-only), the causal reference that separates cache corruption from sampling noise (audit 3).
- **The adoption gate** (fail-closed, audit 1): patch hash verified, clean oracle (A/A) verified, trigger path proven, no stale restore on the patched arm, no new crash, no slot leak, and, later, ARC non-regression. E17 alone can reach at most `ENGINE_CORRECTNESS_VALIDATED`.
- `BUG_EXPOSED_43567` but T2 shows no stale candidates in 12 min → adopt the patch anyway. It is upstream-merged, has two hunks and changes no allocation formula. But expect no LB change.
- `INCONCLUSIVE` on T1a → fix the trigger before any conclusion. **No "bug absent" claim.**
- T3b high-margin divergences under DFlash and not under target-only → a separate DFlash correctness item. #43567 does not explain it.
- **No LB uplift is promised in advance.**

## 7. Not in E17

- #43559 backport (only if L7 or L9 asserts);
- SGLang 0.5.22;
- the paired ARC stage;
- the mid-thinking EOS / parser replay: CPU-only, on saved traces, a separate task;
- #43029 (the spec committed-length fix in 0.5.22, audit 1): UNVERIFIED. The T1c trace (`crossed`, `planned_pos`, accepted length) will show a committed-length / boundary mismatch if one occurs. A dedicated arm is only added after a source check;
- #36880 / #36885 (KDA padding sentinel, audit 1): **not applicable** by source.
  - `attention_registry.py:456` selects `GDNAttnBackend` for hybrid GDN models such as Qwen.
  - `KDAAttnBackend` is used only for `kimi_linear` / `glm5_next` configs (`:500-519`).
  - Our real server logs confirm it: the E12 G4B and EVAL12R5 `sglang_server_try0.log` files contain "Using hybrid linear attention backend for hybrid GDN models." 8 times and no KDA line. Phase 0 still greps for that line;
- per-slot generation counters (audit 3): not adopted; CUDA write ordering makes them unreliable. The rid-level `inplace_pending` flag plus `cached_tokens` gives the same chain.

## 8. Audit 3 disposition (new points only)

| Point | Disposition |
|---|---|
| KL threshold from our own A/A, not `1e-4`; upstream's spec test uses `5e-3` (valid reuse 0.000235-0.000942) | **ACCEPT** (§3). |
| DFlash test at `--mamba-track-interval 64`, seed 128-192 tokens, read the real `kv_committed_len` | **ACCEPT** (L5). The 0.5.21 validator requires interval ≥ draft tokens and interval % page = 0 (`arg_groups/mamba_hook.py:123-125`). |
| `SGLANG_TEST_SKIP_CACHE_HIT_ASSERT` exists; it does not replace checking the hit | Exists (`environ.py:427`). **Not used.** We assert `cached_tokens` explicitly. |
| Upstream lazy-MTP test at 68 slots / 16 running could not do a cache-hit KL because entries were evicted | UNVERIFIED (not opened). It is consistent with T2's design: we count prefix-hit opportunities, not only failures. |
| Serial vs concurrent × target vs DFlash controls (#36548 serial 0/100 vs concurrent 111/304; #38009 single-request divergence) | **ACCEPT** (T3b). The issue numbers are UNVERIFIED by me. |
| CUDA graph ON/OFF control | **ACCEPT** as optional L14. |
| Mid-thinking EOS / parser replay | **ACCEPT** as a separate CPU task (§7). |
| Stage C (natural exposure) only if Stage B shows the bug | **ACCEPT** (decision point A). T3b runs regardless. |
| Function wrappers cannot see `new_slot is None` (all three audits) | **REJECTED as stated:** a before/after state diff gives the same outcome without editing the source (§2). |
| `token_ids_logprob` is NO-GO (audit 1 only) | **REJECTED:** upstream's own test uses it (audits 2 and 3 agree). |
| Six launches ≈ 52-56 min (audit 2) / first READY 8-13 min (audit 3) | First READY of about 9 min: agreed. Relaunches measured at 71-97 s, so the total is about 35 min (§4). |
| Global forced-fail stops seeding (audit 1) | **Partly wrong:** in 0.5.21 the flag only skips the optional pending slot. The prefill checkpoint and in-place tracking still run. Kept as the diagnostic L4 only. |
