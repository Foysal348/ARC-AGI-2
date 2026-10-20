# Run plan (nothing has been pushed or run). Every step needs explicit user approval.

## Step 1: upload the overlay (foysalemonshanto, CPU, no GPU quota)
- Notebook: `notebooks/FN-PLEFP8-UPLOAD.ipynb`.
- Settings: Accelerator None, Internet ON.
- Secrets: attach `KAGGLE_USERNAME` and `KAGGLE_KEY`.
- Start from the Kaggle editor (Save Version -> Save & Run All).
- Result: `foysalemonshanto/qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1` (21 files + manifest, 51.24 GB).
- The last cell must print VERIFIED.

## Step 2: FN-SMOKE v3 (foysalemonshanto, 4xL4, internet OFF, pinned Python 3.12 image)
- Notebook: `notebooks/FN-SMOKE-FLASHNEXT-FP8PLE-4XL4.ipynb`.
- Inputs:
  - competition `arc-prize-2026-arc-agi-2`
  - dataset `dfranzen/pennyroyal-v253`
  - model `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1`
  - model `foysalemonshanto/qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1`
- Budget: <= 50 min (~1.7 quota-hours). Load only, plus one 64-token generation.
- Output: `fn_smoke/fn_smoke_report.json`, `FN_SMOKE_BUNDLE.zip`, and a VERDICT line.
