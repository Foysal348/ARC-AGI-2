# Agent final message: Domdolus cells 00-20 reader (verbatim, 2026-11-04)

I read cells 00-09 and 11-20 completely and wrote the report. Cell 15 holds four JSON-encoded source files; I decoded them into the session scratchpad and read all four too: `v4_edits.py` (431 lines), `qsa_v4.py` (105), `k4_edits.py` (280), `qsa_k4.py` (139). Nothing was run on GPU or Kaggle.

Report: `D:/ARC AGI 2/experiments/Experiment 35 - Public-best rebuild (planned)/Experiment Review/Agent Reports/domdolus_cells_00-20_report_2026-11-04.md`

**Cell by cell**
- **00, 01, 04, 09, 18** are headings only.
- **02** prints the session id.
- **03** is the timer and submission writer.
  - Deadline is 11 h 55 min after kernel start (5 h 55 min in validation mode).
  - The writer merges `sub_*/submission.json` with the `check_attempts.json` records. Variant answers are mapped back (T = transpose, F = flip) and check records are ranked by ok/n, then pct.
  - At the deadline a watcher writes the submission, skips all later cells and runs `pkill -9`.
- **05** finds the utility dataset (`UTILITY_MANIFEST.json`) and the model. It starts a background prefetch with 24 readers, reading files in vLLM's load order. It writes the 51 GB n-gram table once into `/dev/shm/flashnext_ngram.bin` (65,536 spare rows).
- **06** sets vLLM 0.31.0, TP4, 1 replica, fp8 KV, `gpu_util=0.92`, expert parallel on. It requires Python 3.12, compute capability 8.9 or higher, and about 70 GB free RAM.
- **07** unpacks `site-vllm.tar` and `site-arc.tar` (no pip), finds `CUDA_HOME` from the bundled nvcc and makes a libcuda stub. It sets the env (offline mode, `VLLM_ENGINE_READY_TIMEOUT_S=3600`). The text mentions torch 2.13 / CUDA 13, humming and cutlass_dsl.
- **08** checks the model files (Qwen4Exp architecture), with `MTP_PRESENT=False`.
- **11** adds `FLASHNEXT KVDIAG` log lines that print the pool size at runtime, and an override of layers per attention KV group (`FLASHNEXT_KV_GROUP_LAYERS=4`).
- **12** sets how DeltaNet states are held (`FLASHNEXT_HOLD_STATES=pinnew`): only the newest state per request stays pinned and older ones become evictable cache. It also adds second-chance and park modes in `block_pool` and the coordinator.
- **13** turns on batch-sharded sampling: a new `compute_logits_local(skip_gather=True)` plus the `enable_batch_sharded_sampling` engine arg.
- **14** moves `embed_tokens` to pinned host RAM, read through a UVA view, so the table leaves each GPU.
- **15** stores the attention KV of the 12 QSA layers in 4 bits:
  - Row bytes per token per KV head: FP8 512, V4 400, K4 272.
  - V4 uses asymmetric per-row 4-bit values with a float32 scale and zero point. K4 also makes keys 4-bit, after a Hadamard rotation; the query is rotated by the same matrix.
  - Both are decoded inside the Triton split-K kernel. Tiles are capped at 64 tokens because the 99 KB shared memory on L4 is too small for 128.
  - Compressed indexer-key blocks are capped near 6,400 tokens. Their evidence: larger blocks dropped decode from 2,200 to 380 tok/s.
  - Prefix match unit goes 800 → 1024 → 1504, and the prefill chunk 1024 → 2048.
- **16** sends the newest DeltaNet state of each running task to RAM and restores it on a cache miss. The restore happens before the Mamba align pre-copy (a fix for their own session 18b bug). It needs pinnew plus the RAM offload from cell 10.
- **17** stores indexer keys in FP8, and two layers share one compressed-key row.
- **19** adds exact resume: when a call stops normally, all its blocks are parked (partial tail, DeltaNet state, ring block) for 600 s (TTL). A follow-up prompt with the same tokens then resumes with no recompute.
- **20** adds token splice: a follow-up prompt reuses the model's own token ids where the bytes are identical, so the prefix cache keeps hitting.

**Memory and capacity**
- Page size on Kaggle is 409,600 B: "attention blocks of 1,024 tokens at the same page size (409,600 B on Kaggle)".
- My own arithmetic: 409,600 / 512 = 800, / 400 = 1,024, / 272 ≈ 1,504. These equal the three prefix-match-unit values, so the unit is the attention block size at a fixed page.
- The pool size is only printed at runtime by the KVDIAG lines; no cell states a number.

**Where more speed or pool could come from** (each with line evidence in the report)
- The K4 rotation runs as unfused float32 torch matmuls, both on keys at store time and on the query every step. This is the clearest fusion target.
- The 64-token tile cap also applies under K4.
- `gpu_util=0.92`, against 0.95 on the 27B path.
- The RAM restore is a Python loop of copies per layer and per block, followed by a full sync.
- Retention settings with their defaults: reserve 64 blocks, TTL 600 s, min-free 64, cap 512, splice 600 keys. Their 60 s report lines give the counters needed to tune them.
- Exact resume and token splice only act on calls that stopped normally; length-capped calls are not reused.

Several of my cells depend on names defined in cell 10: `VROOT`, `fn_trim_sink_of`, `_FN_SEG`, the RAM offload (`KV_OFFLOAD_GIB`), and the starting prefix unit and chunk values (800 and 1024). The report marks each of these.
