# Q27-XHIGH E8 — Exact Architecture FIXED R2

**Notebook:** `Q27-XHIGH-v1_4_4_E8_EXACT_ARCH_FIXED_R2.ipynb`

## Approved architecture
Stage 1 Frozen E7 Ordering Only → baseline attempts → **validated baseline submission checkpoint** → train-consistent candidate pool → exact Codex 8-view augmentation → Q27 vLLM teacher-forced NLL scorer → protected attempt_2 selector (**apply mode is ON**) → final validation → submission → report/bundle.

The original E7 and Codex donor files are intact. All E7 primary code cells remain identical except **Cell 1 time-deadline configuration**; the original report cell is replaced to run E8 **before** final output/report/bundle.

## Time allocation
`E8_RESERVED_SEC=1800` (30 min) before the original E7 hard stop; `E8_FINALIZE_SEC=120` leaves ~28 min of post-primary scoring time. Values are configurable using `Q27X_E8_RESERVED_SEC` and `Q27X_E8_FINALIZE_SEC` **before notebook execution**. This allocation is experimental and not GPU-calibrated; importantly the E7 primary scheduler ends earlier by design. No additional hidden opt-in is required: E8 applies when it finds valid scored candidates and retains the baseline on failure.

## Audit and tests
Run `python test_e8_r2.py` in an environment with NumPy/nbformat installed. The test script explicitly checks source-parity, F1-F7 regressions, scoring fake engine, submission atomic rollback, and deadline ordering. No Kaggle GPU execution has occurred in this audit.

## Files
- `SOURCE_COMPARISON.md` and `SOURCE_MANIFEST.json` — two-way source comparison.
- `BASE_E7_CELL1_TO_E8_CONFIG.patch`, `BASE_E7_REPORT_TO_E8_FINAL.patch`, `E8_PREVIOUS_TO_FIXED_HELPERS.patch`, `DONOR_LOADER_DIFF.patch` — exact diffs.
- `test_output.txt` — CPU/fake-engine results.
- `patch_e8_r2.py` — reproducible patch builder applied to previous E8 notebook.

**Final extra fix:** Moving up the primary cutoff can legitimately abort outstanding primary units. E8 now continues on completed units rather than skipping the entire scoring stage because `aborted_at_hard_stop > 0`. The same fake-engine scenario is included in tests.
