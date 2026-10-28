# E17 pre-GPU audit: the built notebook (for GPT, 2026-10-26)

**Answer GO unless you find a critical bug or a blocker.** A critical bug is one that would make the GPU session:
- crash;
- waste its time;
- give a wrong verdict: a false BUG_EXPOSED, a false clean result, or a stale restore that goes undetected;
- leave SGLang patched when it should not be.

Report style issues and nice-to-haves separately, as non-blocking. Mark anything you cannot verify as UNVERIFIED.

## What E17 is

E17 is a one-session Kaggle 4x L4 commit notebook, engine only, that tests the SGLang 0.5.21 silent-accuracy suspects of our production engine:
- RedHat Qwen3.8-27B INT4 and BF16 DFlash2 W6, TP4;
- FP8 e4m3 KV, `--mamba-ssm-dtype float16`, `extra_buffer_lazy`, overlap, decode CUDA graphs.

The plan is `plan/PLAN_E17_REV2_KERNEL_ISSUE_GPU_TESTS_2026-10-26.md`. It merges your three earlier audits (`plan/GPT_E17_PLAN_AUDIT_*`) and our dispositions (`plan/E17_PLAN_GPT_AUDIT_*_DISPOSITION_*`).

## Session flow (`notebook/e17_cell.py`, run section at the end)

1. **Phase 0.**
   - `schedule_batch.py` sha256 and `E14P.check(("43567",))`; it must be `not_applied`, or the cell aborts.
   - DFlash `target_layer_ids == [5,19,33,47,61]` and `num_target_layers == 64`; vocab from the config; the GDN backend line; checkpoint KV-scale keys.
2. **Unpatched block.**
   - **L1** T1a: upstream `test_mamba_cache_kl.py` Class 1 ported.
     - `--max-mamba-cache-size 4 --max-running-requests 1 --mamba-track-interval 128 --chunked-prefill-size 2048 --cuda-graph-bs-decode 1`, target-only, production numerics.
     - Environment: `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=0`, `SGLANG_TEST_MAMBA_LAZY_ALLOC_FAIL=0`.
     - A/A ×3: flush, warm `[44]*128`, warm `[42]*128`, score `P = [42]*128 + [43]*127` with full-vocab `token_ids_logprob`.
     - 3 cycles: flush, warm, seed P + 32 greedy tokens, score P.
     - Plus the patch-safety cases: EOS, abort, max_new, 5× reuse, cold.
   - **L2:** L1 with the hook off.
   - **L3:** interval 256 (P255 + P511).
   - **L4:** the forced-fail env.
   - **L5:** DFlash W6 at interval 64 (P246 / P249 / P252, seed 160 tokens).
   - **L6:** upstream Class 2 control: no_buffer + EAGLE 3/1/4 (our MTP head), overlap and graphs off. This is a known stale path in 0.5.21 (#43387 not present), so it shows the probe can detect a stale restore.
   - **L12 / L13:** T3b, target-only vs DFlash with `--disable-radix-cache`: serial 4 + 16 concurrent, greedy 2,048 tokens. L12 also scores T3a fp16 at 8k / 16k / 32k / 48k prefixes.
3. **Patch block.** `E14P.apply(pkg, ("43567",))` (the exact two-hunk upstream backport), then:
   - L7 = L1 + the safety cases;
   - L8 = L3;
   - L9 = L5.
4. **Decision A.** `BUG_EXPOSED_43567` when an unpatched arm is `BUG_EXPOSED` and its patched twin is `TRIGGER_REACHED_NO_STALE` with 0 confirmed stale donations.
   - Only then: T2 = 16 streams on the production flags, 4 shared ~30k prompts, production sampling, 12 min.
   - Order: patched (L11), then `E14P.revert`, then unpatched (L10).
   - Then warm vs cold full-vocab probes, and the four-way verdict.
5. **L15** (if time is left): T3a fp32 SSM. The patch is always reverted at the end, and also on a cell error.

## Instrumentation (`src/e17_hook.py`)

The hook is written to `/tmp/e17_hook/sitecustomize.py` (sha-checked) and put on PYTHONPATH of the E17 launches only. It is inert without `E17_TRACE_DIR`. A daemon thread waits for the modules and wraps, in place:
- `ScheduleBatch.mamba_lazy_prealloc_at_boundary`
- `ScheduleBatch.mamba_lazy_spec_prepare`
- `SchedulerBatchResultProcessor._mamba_prefix_cache_update` / `_mamba_lazy_spec_update`
- `MambaComponent.prepare_for_caching_req`
- `ModelRunner.load_model` (KV scales on TP0)

**The outcome is inferred from the state before and after each call. No source edit is made.**
- Decode path: eligible = `seq_len % interval == 0` and no pending slot; afterwards the pending slot is still -1 → `optional_alloc_failed`, and the rid is marked.
- Spec path: in window and no pending slot before; afterwards the planned position equals the keep index → `spec_inplace_fallback`, and the rid is marked.
- A new non-None `mamba_last_track_seqlen` clears the mark.
- A donation while marked, with a non-None length and a returned value > 0, is a `stale_donation_candidate`. On the decode path it is confirmed at once. On the spec path it is confirmed only if the next `_mamba_lazy_spec_update` reports `crossed=True`; otherwise it is cleared.
- Output: per-pid `counters_<pid>.json` and `trace_<pid>.jsonl`, flushed every 2 s. `register_at_fork` restarts the thread in forked children.
- With TP4 every rank runs a scheduler. The analysis uses the primary pid (the one with the most decisions), never a sum.

**Verdict per T1 arm** (`e17_verdict`):
- `BUG_EXPOSED` = alloc failure, a confirmed stale donation, a reuse hit deeper than the safe prefix (> 128), and KL > tol, where tol = max(5e-3, 10 × the A/A max).
- Partial chains: `STALE_RESTORE_EXPOSED_NO_OUTPUT_FLIP`, `TRIGGER_REACHED_NO_STALE`, `KL_ABOVE_TOL_CHAIN_INCOMPLETE`, `INCONCLUSIVE`. An incomplete chain is never a PASS.

## Evidence so far

- `kit/`: Kaggle CPU kit `foysalemonshanto/q27-e17-cpu-harness` v1, on the same pinned image as the GPU notebooks (Python 3.12.13). **ALL OK.**
  - Hook unit tests 8/8. They run against reduced copies of the 0.5.21 methods, with the #43567 hunks switchable: the unpatched chain is confirmed, the patched one has no candidate, a success is not a failure, forced fail is counted without an alloc call, the spec candidate is confirmed or cleared, out-of-window is not counted, and the KV scales are logged.
  - Notebook harness 20/20, against the fake SGLang server (`src/harness/make_e17_fake_server.py` models the #43567 / #43387 behaviour).
- Local Windows runs: the same 20/20.
- The fake server is a model of SGLang, not SGLang itself: the GPU session is the first contact with the real scheduler.

## Questions for you

1. **Hook correctness on the real 0.5.21 code** (`sglang_ref/`). Do the before/after inferences match what the real methods do, under overlap scheduling in particular?
   - Is the `seq_lens_cpu[i] % interval` eligibility right for lazy decode? Note that `mamba_track_interval` may be the track grid (lcm with the checkpoint grid) at the call site.
   - Is `mamba_lazy_spec_track_positions_cpu[i] == mamba_next_track_idx` a sound in-place signal?
   - Is "a new non-None length clears the mark" safe? Could the overlap order clear it before the stale donation runs, which would make us miss the bug?
2. **Hook installation.** Does a `sitecustomize` on PYTHONPATH reach the SGLang scheduler processes (spawn vs fork), given that the venv env drops PYTHONPATH and we add it back? Could the hook's polling thread, or the `.item()` reads, change scheduler timing enough to hide the race? The L1 vs L2 (hook off) comparison is meant to show that.
3. **Trigger fidelity.** On our 27B INT4 TP4 L4 server with FP8 KV and graphs `[1]`:
   - Does the upstream recipe (4 slots, 1 running, 255 tokens, seed 32) still produce the overlap race? The prefill result must be processed after the first decode is launched.
   - Is the safe depth 128 and the stale depth 192 right for interval 128? What changes for interval 256, P511, and the DFlash interval-64 prompts?
4. **Measurement.**
   - Full-vocab `token_ids_logprob` at 248,320 ids per probe: are there any 0.5.21 limits, timeouts or JSON costs we missed?
   - Is KL tol = max(5e-3, 10 × A/A) sound for INT4 / FP8 / TP4?
5. **Patch handling.** Apply → revert ordering, and the revert on error. Is there any path that leaves SGLang patched for the unpatched T2, or that runs the "unpatched" arms on a patched build?
6. **T2 / T3b / T3a design.** Any blocker: KV pressure, the 45k context cap, the stream join / abort at the end, server idleness before the flushes, or the `parity_pass` margins?
7. **Budget.** 170 min. The first launch takes ~9 min, then 71-97 s per relaunch (EMS RISKY v1), for about 14 launches. Is anything likely to run out of time or skip silently?
8. Anything else that is a blocker.

## Contents of the zip

- `notebook/`:
  - the built notebook;
  - `build_e17.py`, `e17_cell.py`, `e17_report_cell.py`.
- `src/`:
  - `e17_hook.py` and `test_e17_hook.py`;
  - `harness/`: the fake server and harness makers plus the generated files;
  - the CPU kit builder.
- `e14_modules/`:
  - `e14_sglang_patches.py`, which holds the #43567 backport;
  - `e14_measure.py`, with `parity_pass`, `compare_passes`, `flush_cache` and `post_json`.
- `sglang_ref/`: the 0.5.21 source files the hook wraps or reads (see the list below).
- `base_cells/`: the E15P base cells the notebook reuses (config, E14 facts, install, engine): `e11_flags`, `E11Server`, `e11_caps`.
- `plan/`: plan rev 2, your three audits, our dispositions, the rev 1 plan.
- `kit/`: the Kaggle CPU kit report and logs.
- `ems_logs/`: the EMS RISKY / SAFE live logs, with the relaunch times and the earlier DFlash vs target probes.

The `sglang_ref/` files:
- `schedule_batch.py`
- `batch_result_processor.py`
- `components/mamba.py`
- `unified_radix_cache.py`
- `model_runner.py`
- `environ.py`
- `runtime_context.py`
- `mamba_hook.py`
- `spec_utils.py`
- `pool_stats_observer.py`
- `memory_pool.py`
- `dflash_utils.py`
- `attention_registry.py`

Keep URLs literal (`https://...`).
