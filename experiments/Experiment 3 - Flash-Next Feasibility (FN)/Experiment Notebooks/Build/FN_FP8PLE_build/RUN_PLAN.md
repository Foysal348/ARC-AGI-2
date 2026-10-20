# Run plan (nothing has been pushed or run). Every step needs explicit user approval.

## Step 1: upload the FP8 PLE overlay (foysalemonshanto, CPU, no GPU quota)
- Notebook: `Research Base Experiment/FN-PLEFP8-UPLOAD.ipynb`, sha256 `9a6fe011…`.
- Settings: Accelerator = None, Internet = ON.
- Secrets: attach `KAGGLE_USERNAME` and `KAGGLE_KEY` of foysalemonshanto.
- Start from the Kaggle editor (Save Version → Save & Run All). A run started through the API cannot read the secrets.
- Result: private model `foysalemonshanto/qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1`, 15 files, 51.22 GB.
- Cell 5 waits until Kaggle lists every file.

## Step 2: FN-SMOKE v3 (foysalemonshanto, 4×L4)
- Notebook: `Research Base Experiment/FN-SMOKE-FLASHNEXT-FP8PLE-4XL4.ipynb`, sha256 `f041533c…`.
- Inputs:
  - competition `arc-prize-2026-arc-agi-2`
  - dataset `dfranzen/pennyroyal-v253`
  - model `dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1`
  - model `foysalemonshanto/qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1`
- Settings: internet OFF, pinned Python 3.12 image.
- Budget: ≤ 45 min (≈ 1.5 quota-hours). Load only, plus one 64-token generation.
- Dual mode: in a competition rerun it only writes a fallback `submission.json`.
- Output: `fn_smoke/fn_smoke_report.json`, `FN_SMOKE_BUNDLE.zip`, and a VERDICT line.
