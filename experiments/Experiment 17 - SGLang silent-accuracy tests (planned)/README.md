# Experiment 17: SGLang silent-accuracy tests (planned, 2026-10-26)

**Question.** Does SGLang PR #43567 (a stale lazy Mamba checkpoint) silently corrupt our production engine? Does
DFlash W6 diverge from target-only decoding?

**Engine under test.** SGLang 0.5.21, RedHat Qwen3.8-27B INT4, BF16 DFlash2 W6, TP4 on 4x L4, FP8 KV, fp16 SSM,
extra_buffer_lazy.

**Plan.** `../Experiment 14 - Optimized Engine on 4xL4 (planned)/Bug ang fix/PLAN_E17_REV2_KERNEL_ISSUE_GPU_TESTS_2026-10-26.md`
(rev 2: three GPT audits merged; their texts and dispositions are in the same folder).

## Files

| Path | What |
|---|---|
| `src/e17_hook.py` | Instrumentation. It is written as `sitecustomize.py` on PYTHONPATH of the E17 server launches only, and wraps four SGLang methods in place. It never edits the source: the unpatched arm runs byte-identical code. It counts optional-slot failures, in-place fallbacks, stale donation candidates (and confirms them), and logs the KV scales on TP0. |
| `src/test_e17_hook.py` | CPU tests of the hook against reduced copies of the 0.5.21 methods (8 tests). |
| `Experiment Notebooks/Build/build_e17.py` | Builds `Experiment Notebooks/Q27-XHIGH-E17_SGLANG_SILENT_ACCURACY.ipynb` from the E15P rev 2 base cells (production engine, 56a22f7f) + the hook cell + `e17_cell.py` + `e17_report_cell.py`. |
| `src/harness/make_e17_fake_server.py` → `e17_fake_server.py` | The E15P fake SGLang server plus an E17 model: the #43567 chain on small-slot servers, patched vs unpatched, the no_buffer stale path, full-vocabulary logprobs, the hook's output files. |
| `src/harness/make_harness_e17.py` → `harness_e17_nb.py` | The CPU harness. It runs all 10 code cells against the fake server and checks the launch flags, verdicts, patch apply / revert, decision, T2, T3 and the report. |

## Session (one Kaggle 4xL4 commit, 170-min profile `E17`)

1. **Phase 0.** Provenance: the `schedule_batch.py` sha, #43567 absent, DFlash taps `[5,19,33,47,61]`, vocab, GDN backend.
2. **Unpatched.**
   - L1: T1a (upstream Class 1: 4 slots, 1 running, interval 128, prompts `[44]*128`, `[42]*128`, then 255 tokens) plus the patch-safety cases.
   - L2: the same with the hook off.
   - L3: interval 256 (+ a 511-token prompt).
   - L4: the forced-fail env.
   - L5: DFlash W6, interval 64.
   - L6: upstream Class 2 (no_buffer + MTP, a known stale path).
   - L12 / L13: T3b, target-only vs DFlash with the radix cache off (+ T3a fp16 scores).
3. **#43567 applied** (`e14_sglang_patches`): L7 (= L1 + safety), L8 (= L3), L9 (= L5).
4. **Decision A.** If the unpatched arms show the full chain and the patched ones are clean: T2, 16 streams on the production flags, patched (L11), then reverted and unpatched (L10).
5. **L15.** T3a fp32 SSM, if time is left.

**Verdict ceiling:** `ENGINE_CORRECTNESS_VALIDATED`. ARC solve-rate impact is a separate paired run, done only after a positive engine result.

**Status (2026-10-26):** built, notebook sha 418023a5 (188 kB).
- Local: hook tests 8/8, CPU harness 20/20.
- Kaggle CPU kit `foysalemonshanto/q27-e17-cpu-harness` v1 (pinned image, Python 3.12.13): ALL OK, hook 8/8, harness 20/20, T2 0 stream errors.
- GPU run only with the user's go.
