# Q27-AGENT v1.9.0 — synthesis microprobe delivery

## What is implemented

Based on the original v1.8.3 Kaggle notebook and the supplied v1.9 plan.

- No forced submit/tool choice and no xhigh→medium→low effort ladder in the new synthesis execution path.
- One no-tools Qwen3.8-27B chat request per ARC task at exact thinking_token_budget=49,152 and max_tokens=57,344; temperature 0.6, top_p 0.95, top_k 20.
- Last valid fenced Python program with top-level `transform(grid)` parsed, tool-only globals rejected, best-effort unsafe-code detection, existing isolated subprocess used for train/test execution, all-test-output validation.
- Context overflow returns `context_too_large` rather than silently reducing budget.
- Error 400 for unsupported thinking budget fails closed. Transient HTTP 5xx only one retry; bounded no-thinking code commit for absent program if there is time.
- No repair in the initial microprobe. Direct recorded Run 7 candidate remains attempt_1; usable verified agent code may occupy attempt_2.
- Existing Kaggle offline wheelhouse, MTP3, FP8 KV, server ladder, initial fallback and submission validation preserved.
- Hidden-run path remains DIRECT-only unless evidence demonstrates a net AGENT gain.

## Running

Upload `Q27_AGENT_v1_9_0_SYNTH_MICROPROBE.ipynb` as a Kaggle notebook **with the same Kaggle datasets attached as the v1.8.3 notebook**. Choose 4× L4 GPUs and internet OFF. Default commit run is three tasks: `136b0064,247ef758,2b83f449`. The default is NOT the entire EVAL12. To run EVAL12 later, set `Q27A_EVAL_TASKS=eval12` before the launch cell. The engine startup cost is real; this notebook has not run on those GPUs.

For a free CPU pipeline dry run set `Q27A_DRYRUN=1` before launch. That uses a fake vLLM server and cannot establish model correctness or live throughput.

Read `/kaggle/working/agent_server_report.json` for startup and budget contract, and `/kaggle/working/agent_eval/eval_report.json` plus `/kaggle/working/agent_eval/attempts.jsonl` for results. `submission.json` is initialized and finally sanitized in all modes.

## Tests conducted

- 12 Python CPU regression/integration tests (PASS), including fake-server EVAL and DIRECT-only hidden-fallback path.
- The complete 22-cell notebook executed on a local CPU with synthetic Kaggle-style datasets and the fake server (PASS); evaluator, reporting, and final submission validator completed.
- Source-to-notebook embedding, Python code syntax, nbformat validation and deterministic SHA verified.

**Not tested:** actual Qwen3.8 checkpoint, 4×L4 GPUs, real MTP3 throughput, live vLLM tokenizer endpoint semantics and ARC generalization. Consequently, a claim of fully bug-free or guaranteed improved LB cannot be made.

**Security:** model-generated Python is executed by a subprocess guarded with a Python audit hook plus source checks; these are not a complete OS sandbox. For rigorous evaluation integrity, score in a separate process/container not reachable by generated code. This notebook defers solution loading until after inference but Kaggle's mounted solutions might still be readable through unblocked native paths.

## Rebuild

This archive includes the original v1.8.3 reference notebook, `build_notebook.py`, all source modules, and the tests. From the archive root run `python build_notebook.py`.
