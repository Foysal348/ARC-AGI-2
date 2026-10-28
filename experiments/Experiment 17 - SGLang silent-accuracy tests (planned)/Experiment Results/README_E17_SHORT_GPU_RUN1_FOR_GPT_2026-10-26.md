# E17 SHORT, GPU run 1 (2026-10-26): all four arms crashed with `Can not alloc mamba cache`. Diagnosis and fix requested

**What we want from you:**
1. Confirm or correct the diagnosis below, against the attached logs, traces and SGLang 0.5.21 source.
2. Give the **minimal working way** to reach the #43567 stale-checkpoint path on SGLang 0.5.21 with our model, so the next run can decide whether the bug exists on our stack (ordinary decode and DFlash W6).
3. Say whether the run tells us anything about production.

Mark anything you cannot verify as UNVERIFIED. Keep URLs literal (`https://...`).

## The run

**Kaggle and stack:**
- Kernel `foysalemonshanto/q27-xhigh-e17-sglang-silent-accuracy` v1, notebook sha 9793b41e, profile E17SHORT, 4x L4.
- SGLang 0.5.21; RedHat Qwen3.8-27B INT4 (W4A16); FP8 KV; fp16 SSM; `extra_buffer_lazy`; overlap on; TP4.

**Arms.** Each is a relaunch of the production command with:
`--max-mamba-cache-size 4 --max-running-requests 1 --mamba-track-interval 128 (64 for DFlash) --chunked-prefill-size 2048 --cuda-graph-bs-decode 1`, plus `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=0` and `SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=0`.
- L1: target-only, unpatched, P = `[42]*128 + [43]*127`, seed 32 tokens.
- L5: DFlash W6, unpatched, P246 / P249 / P252, seed 160.
- L7 = L1 and L9 = L5, after `E14P.apply(("43567",))` (the two-hunk upstream backport only; **#43559 not applied**).

Sequence per arm, upstream `test_mamba_cache_kl.py` Class 1:
1. Reference ×3: flush, warm `[44]*128`, warm `[42]*128`, score P with full-vocab `token_ids_logprob`.
2. Then 3 cycles: flush, warm, seed P + N greedy tokens, score P.

**Instrumentation:** `src/e17_hook.py` (sitecustomize; it wraps the methods in place and does not edit the source).

## What happened (evidence: `run/q27x/sglang_server_e17_*.log`, `run/q27x/e14/e17/trace/*`, `run/q27x/e14/e17_report.md`)

- All four servers launched in 61-77 s. The hook self-test passed on all 4 TP ranks.
- The three reference rounds completed, full-vocab scoring included (HTTP 200).
- The **first seed request** of every arm crashed every scheduler rank, at the same place in all four arms (L1 log lines 188-229):
  ```
  process_batch_result_prefill -> maybe_cache_unfinished_req -> UnifiedRadixCache.cache_unfinished_req (unified_radix_cache.py:1188)
  -> MambaComponent.prepare_for_caching_req (mamba.py:599, extra_buffer branch: new_slot = self._alloc_mamba_slot())
  -> _alloc_mamba_slot (mamba.py:503-510): alloc(1) None -> evict_for_alloc(mamba_num=1) -> alloc(1) None -> AssertionError
  ```
- **Hook trace (identical on all ranks):**
  - counters `optional_alloc_attempts 5`, `optional_alloc_failed 0`, `donations_unfinished 1`, `donations_finished 13`;
  - allocator at the crash: `available 0, size 4, evictable 0`;
  - the lazy pending-slot allocation at the 256 boundary **succeeded** (`alloc_ok` at seq_len 256) on every scoring and seed request.
  - So the #43567 path (optional alloc FAILS → in-place keep write → stale-length donation) was **never reached**. Instead the pending slot took the last free slot, and the overlapped prefill-result donation found no slot and nothing evictable (the consumed `[42]*128` prefix was still locked).
- **The patched arms crashed identically.** This matches the upstream text of PR #43559 (https://github.com/sgl-project/sglang/pull/43559):
  - "the next decode can consume the temporary tracking slot before the previous prefill donates its checkpoint. Donation then needs a replacement slot while the consumed prefix state is still locked, exhausting the pool and asserting";
  - its negative control: #43567 kept and #43559 removed gives the same assertion.
- **The cell recorded each arm as an error.** The partial A/A numbers were lost: the arm's phase raised before returning.
- **Afterwards:** the patch was reverted (sha verified, `not_applied`); no cell error.

## Our reading (please check)

1. On 0.5.21, the upstream 4-slot / 1-running recipe asserts in #43559's path before #43567's path can occur. The upstream KL test therefore needs **both** fixes on 0.5.21 (or main at the time the test was written).
2. **Production:** the #43559 assertion needs available = 0 **and** evictable = 0 at a donation. We run 68 slots / 16 streams, with many evictable cached checkpoints. The assertion appears 0 times in our 62 production SGLang logs. We read this as: not a production crash risk. Do you agree?
3. **FP8 KV scales (side finding):**
   - The log says "Using FP8 KV cache but no scaling factors provided. Defaulting to scaling factors of 1.0".
   - In 0.5.21 that warning comes from `load_model_utils.py:114-135` whenever `--quantization-param-path` is unset.
   - Our checkpoint's `quantization_config.kv_cache_scheme` is static FP8 per-tensor, and the checkpoint carries 32 `k_scale` / `v_scale` keys. So `CompressedTensorsKVCacheMethod` (`kv_cache.py:55-85`) should load the calibrated scales.
   - Our hook did not capture the runtime values. The `ModelRunner.load_model` wrap probably ran after load had started.
   - Is that reading right? What is the cheapest proof at runtime?

## Options we see for reaching the #43567 path on 0.5.21 (please choose or correct, and give the exact recipe)

| Option | Idea | Our concern |
|---|---|---|
| A | `SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=1` (forced fail) with the same 4-slot recipe: the pending slot is never taken, so the prefill-result donation finds the free slot and the in-place write + stale-length donation can happen. Unpatched vs patched; ordinary + DFlash. | Audits 1-3 said forced fail is a secondary, diagnostic-only signal. A positive result still shows the mechanism corrupts OUR model. Does it reach exactly the #43567 chain (in place → donated at old length → restored)? |
| B | Backport #43559 too (release the consumed prefix lock before the donation retry) and run the upstream recipe with natural pressure | The #43559 hunk is not in our patch module; it has to be written and validated against 0.5.21's `mamba.py` / lock code. What exactly does #43559 change? |
| C | 5 slots instead of 4 (room for the donation) | Probably no optional-alloc failure at all, so no trigger. |
| D | Natural production-like pressure (T2: 68 slots, 16 streams, shared prompts) with the hook | It measures the frequency; it does not prove the mechanism. |

Also, what is the cleanest way to keep partial T1 data when the server dies mid-arm (our phase raised and lost the A/A KLs)?

## Contents of the zip

- `run/`: the full Kaggle output, including:
  - the server logs of all four arms;
  - the hook traces (`counters_*.json`, `trace_*.jsonl` per process);
  - `e17_report.json` / `.md`, `e14_results.json` and the kernel log.
- `notebook/`: the notebook, `e17_cell.py`, `e17_report_cell.py`, `build_e17.py`.
- `src/e17_hook.py`.
- `e14_sglang_patches.py`: our #43567 backport (two hunks).
- `sglang_ref/`: the 0.5.21 sources involved:
  - `mamba.py`, `schedule_batch.py`, `batch_result_processor.py`, `unified_radix_cache.py`;
  - `memory_pool.py`, `common.py`, `load_model_utils.py`, `kv_cache.py`, `compressed_tensors.py`, `environ.py`.
- `plan/`:
  - PLAN_E17 rev 2;
  - the audit dispositions;
  - the previous audits.
