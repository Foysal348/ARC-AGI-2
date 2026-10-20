# Experiment 3 — Flash-Next Feasibility (FN)

**Status:** closed (2026-10-11): **not viable** on 4×L4 with this stack.

## Idea

Serve Qwen3.8-Flash-Next with SGLang on 4×L4 for much higher throughput than the 27B.
- Model: an Intel W4A16 AutoRound build plus an FP8 PLE overlay (`foysalemonshanto/qwen38-flash-next-fp8ple-overlay`).

## Path

- **FN-SMOKE v3–v3.3:** pin probe passed; failed in the MoE runner (AutoRound needs `auto`).
- **FN-PROBE v4:** two launches timed out loading weights (slow input disk).
- **v4.1–v4.2.1** (five GPT review rounds):
  - `--weight-loader-prefetch-checkpoints` brought weight loading to 345 s;
  - process-group kill, port gate, calibrated decode measurement;
  - local fake-SGLang harness 11/11.
- **v4.2.1 run** (esfersami50, 2026-10-11):
  - server ready at 862 s, KV 118k tokens;
  - first decode did not crash, but the output was **garbage** ("!!!!", token 0; a TileLang data-race warning in the QSA kernel);
  - single-stream decode about **6 tok/s**.

## Conclusion

Not competitive with the 27B (about 373 tok/s at 16 running) → dropped.

## Folders

- **Experiment Notebooks/:** `FN-*.ipynb`; `Build/` (FN_SMOKE, FN_FP8PLE, FN_PROBE_v4 builders and harnesses; paths pre-2026-10-12).
- **Experiment Review/:** GPT review bundles.
- **Experiment Results/:** `Kaggle Runs/` (FN_SMOKE_v3_run, FN_v33_run, FN_PROBE_v4_run1, FN_PROBE_v421_run1), `FN_SMOKE_BUNDLE.zip`.
