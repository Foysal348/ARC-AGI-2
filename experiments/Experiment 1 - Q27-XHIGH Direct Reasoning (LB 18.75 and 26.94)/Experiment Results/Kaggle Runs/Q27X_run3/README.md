# Q27-XHIGH v1.3.5 — last notebook and its EVAL12 run output (2026-10-06)

## Contents
| path | what |
|---|---|
| `notebook/Q27-XHIGH-v1_3_5.ipynb` | Notebook as pushed to Kaggle (foysalemonshanto/q27-xhigh, version 5). sha256 `6dcb8042dc37c0c3b9c00241ac87affcb7dbe84cdf060e724773dda1fde8207e` |
| `run/q27-xhigh_run3_executed.ipynb` | The same notebook after the Kaggle interactive run, with all cell outputs (vLLM log included) |
| `output/q27x/run_report.json` | Run summary: engine, scheduler, score, MTP, live tokens, projection |
| `output/q27x/records.jsonl` | One record per task: mode, tokens, verification, attempt sources, score |
| `output/q27x/answers/<task>.txt` | Per task: thinking tail + the full answer after `</think>` |
| `output/q27x/vllm_metrics.json` | vLLM metric snapshots (spec-decode, KV usage, preemptions) |
| `output/q27x/vllm_frontend.log` | Empty: known bug, the handler was added before `import vllm` and vLLM's logging config replaced it |
| `output/submission.json` | Commit-mode submission (identity fallback for the hidden test tasks; not for upload) |

## Run
- Kaggle 4×L4, interactive. Run id `20261005T202837Z`. Ran 2026-10-06 02:28–03:25 Bangladesh time (UTC+6).
- Setup took 14.3 min; the scheduler ran for 42 min.
- Engine: Qwen3.8-27B-FP8, vLLM 0.19.0, TP4, fp8_e4m3 KV, MTP 3, FlashInfer. KV cache: 10.98 GiB / 337,600 tokens.
- Changes from v1.3.4: `max_num_batched_tokens` 4096 (fixed run 2's OOM) and `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=1`.

## Result (EVAL12: 12 public evaluation tasks, 22 queries)
- Score: **1/22 queries (4.55 %)**; task-normalised 4.17 %. The only hit was 247ef758, query 1, from unverified code (train 2/3).
- All 12/12 tasks hit the 49,152-token thinking cap and were force-closed. None closed thinking naturally.
- Code was written for 12/12 tasks; 1/12 was train-verified (4c3d4a41, which was wrong on test). 2 code errors.
- JSON grids parsed for only 6/12. The other 6 have a JSON block with a missing outer `[` that strict `json.loads` rejects. Parsed leniently, those grids are still 0/22 correct.
- Speed:
  - 307 tok/s mean with 12 active (excluding stalls); 247.8 tok/s overall.
  - MTP acceptance 58.8 %.
  - KV usage peaked at 59.5 %; 0 preemptions.
- Each forced close starts a new request, and prefix caching is off, so ~52k tokens are prefilled again. That stalls generation for about 30 s each time, 13.8 % of the scheduler's wall time.
- Projection for 240 tasks: 14.36 h, so it does not fit in 12 h.

## Reference points
| source | score | setup |
|---|---|---|
| Discussion 745787, NguyenThanhNhan | LB 23.33 / 22.50 | Qwen3.8-27B FP8 + EAGLE (SGLang); xhigh; 65k tokens; forced commit at 49k with a solution grid; pure reasoning; radix cache on |
| russcore/arcagi2-qwen38-freeform-direct (uses our model) | LB 10.14 / 7.78 / 8.61 | Per query; grid-only answer; 2 samples; temperature 0.6; Python tool |
| arcprize.org | ARC-AGI-2 semi-private 42.4 %, public eval 51/120 | Qwen3.8-27B xhigh, ~175k output tokens per task |

How our design differs from these:
- All test inputs of a task share one prompt and one budget. 9 of our 12 sampled tasks had 2–3 test inputs, against ~8 % of tasks in the hidden test.
- The answer must be code plus JSON, not just a grid.
- One sample per task.
- Temperature is 1.0.
