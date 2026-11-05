# Agent final message: Domdolus cells 21-46 reader (verbatim, 2026-11-04)

I read cells 21-29 and 31-46 in full (cell46 and cell25 in chunks) and wrote the report. Nothing was run on GPU or Kaggle.

Report: D:/ARC AGI 2/experiments/Experiment 35 - Public-best rebuild (planned)/Experiment Review/Agent Reports/domdolus_cells_21-46_report_2026-11-04.md

It covers each cell as asked (purpose, every patch with target file, anchor, change and env flag, every kernel, harness logic, dependencies, constants), then the three closing sections. Cell30 is not in it.

**Engine patches**
- **cell22, top-k exchange:** in model_runner.py, each tensor-parallel rank sends only its top-64 (value, id) pairs over all_to_all instead of the full logits, and the sampled position is mapped back with a gather. Any request with top_k above 64, penalties, logprobs, grammar or MTP draft tokens sends that whole step down the full path.
- **cell23, HC-down split-K:** a new Triton module `fn_hc_down.py` (K-split partial kernel plus a fixed-order reduce) is patched into Humming `linear.py`. It keeps the FP8 [336, 10240] weight, checks itself against Humming (relative diff below 2e-2), then frees Humming's copy. Their timings: at M 60, 23.5 vs 28.7 us; at M 1024 it is slower, 125 vs 112.6 us. The code still sends every M to split-K.
- **cell24, DCP communication in qsa.py:** the query all-gather starts early on a side stream so it overlaps the indexer, and the partials are cast to BF16 before the DCP combine.
- **cell25, nine decode trims:**
  - K4_ROT: skips the second query rotation when the early all-gather already did it.
  - QSA_GATE: a new `fn_qsa_gate.py` kernel replaces about 6 small kernels; they report it bit-identical.
  - GAP_ONCE: the gap and compact arguments are computed once per forward pass.
  - QSA_SHIFT: the shift-back is one kernel.
  - TOPK_BF16: top-k runs on the BF16 logits.
  - K4_FWHT: a 256-point butterfly replaces the fp32 Hadamard matmul, and is fused into the key store.
  - GDN_AB: removes the b/a copies.
  - HCD_SILU: a new `fn_hcd_silu.py` puts the SiLU into the split-K reduce, only for M up to 128.
  - MAMBA_TILES: 16 tiles become 1.
  - cell25's anchors depend on lines inserted by cells 22-24, so the cell order matters.
- **cell27:** restores the startup caches and copies FlashInfer AOT kernels in so they are not rebuilt (otherwise about 2 min).
- **cells 35, 37, 38, fast startup:**
  - The n-gram table is read lazily and served from /dev/shm with no copy (mmap plus cudaHostRegister, 3-row check per shard).
  - The utility tars are dropped from the page cache, and GPU_UTIL is set to 0.945.
  - The vLLM worker waits inside the loader for the model prefetch (up to 1800 s).
  - Every tensor of a non-local expert is skipped at load.
- **cell36:** vLLM measures the KV itself. The note says 8.81 GiB per GPU = 4,064,286 tokens.
- **cells 40 and 44:** cell40 turns on all switches plus the indexed Humming MoE GEMM. cell44 checks each switch's log line and raises if one is missing.
- **cell41, miss-deferral collapse fix (scheduler and block_pool):** when attention blocks were evicted in the last 60 s, a large uncached prompt (at least 16,384 tokens, under half computed) is held back. After 1200 s it is released, one at a time with a 60 s gap. It needs FLASHNEXT_OFFLOAD_EVICT=1.
- **cell43, server launch:**
  - Window 1,048,576, 128 sequences, 4096 batched tokens, MTP off.
  - Custom all-reduce off, prefix caching on, mamba align, engram CPU offload, expert parallel with weight filter.
  - Up to 8 start attempts, with the error classified to pick the next setting.

**Harness**
- **cell29:** a local IPython-kernel sandbox with a memory guard (8 GiB per tool process, emergency stop below 12 GiB of free RAM).
- **cells 31-33:**
  - Tool output is cut at whole lines, and the grid helpers default to bg='auto'.
  - A rejected final answer now keeps the conversation instead of resetting it.
  - A broken kernel is replaced with the setup code run again, instead of restarting the whole run.
  - The prompt has no token numbers and says "Think carefully"; the `puzzle` protection was loosened.
- **cell46, the run:**
  - Settings: 200k tokens per run, 70 runs at a time, reasoning xhigh, no total token budget (time decides), stop 1 min before the deadline.
  - Try 1 for every task, plus an independent plain try 2 queued from the start. Try 3 exists but is off.
  - Every minute it prints a load line from /metrics.
  - A dead or hung server (no token for 10 min) is restarted up to 3 times if at least 30 min are left.
  - Eval score: per task, (test outputs matched by attempt_1 or attempt_2) divided by the number of test outputs.
- **Throughput reference in cell46:** 651 tok/s decode at 60 concurrent runs, 94 % prefix hit, 0 preemptions.

**Tunables found (line evidence in the report)**
- Concurrency 70 against 128 server sequences.
- No size limit on the split-K route, though it is slower at M 1024.
- The top-k exchange K (64) is never set in cell40.
- The FWHT and HCD_SILU launch settings are fixed constants.
- GPU_UTIL 0.945, or an explicit KV size through the already-wired `--kv-cache-memory`.
- The miss-deferral parameters can all be changed by env var.
- MTP off (turning it on would disable the top-k exchange).
- Try 3 is available but off.

**One inconsistency:** cell36's text says "graphs up to 64" and mentions "104-graph runs", but cell10's capture list goes up to 512. The real capture list should be read from the server log.
