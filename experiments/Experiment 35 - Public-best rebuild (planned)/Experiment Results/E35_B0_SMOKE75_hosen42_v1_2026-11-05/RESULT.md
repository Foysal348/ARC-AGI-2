# E35 B0 smoke (hosen42/e35-b0-smoke75 v1, 2026-11-05 10:49-12:05 BDT)

Domdolus #16 notebook unchanged except VALIDATION_MODE = True, 75 min from kernel start, trace saver cell.

- Server: TP4 + EP + DCP2, K4 KV, util 0.945, CUDA graphs, MTP off; up ~5 min after kernel start.
- Weights: 10.04 GiB per GPU (model loading 28 s). KV: 9.09 GiB per GPU = **4,410,970 tokens**.
- SPEED SUMMARY (concurrency 70, minutes 10-69): **decode 899 tok/s**, prompt computed 307 tok/s, prefix hit 97 %, 0 preemptions, KV usage 67 %, running 68. (Their own reference: 651 tok/s at 60.)
- Solve window 68.2 min, round 1 only (try 2 never started): 120 ended, 50 final answers, 12 full check(), 3,284,569 tokens.
- **Score (eval 120, our scorer = notebook's): 78 / 172 queries, task-norm 57.83 / 120 = 48.19 %** in 68 min.
- Files: submission.json, status.json, runs.json, loadtest_rows.json, notebook log, vLLM server log, traces.tar.gz (21.9 MB, all text traces).
