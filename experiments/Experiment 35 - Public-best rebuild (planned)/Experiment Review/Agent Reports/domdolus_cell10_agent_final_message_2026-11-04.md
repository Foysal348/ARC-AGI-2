# Agent final message: Domdolus cell10 reader (verbatim, 2026-11-04)

I read cell10 completely, all 339 lines and 144 KB, including the very long edit-list lines 45-71. The report is written. I ran nothing on GPU or Kaggle.

Report: `D:/ARC AGI 2/experiments/Experiment 35 - Public-best rebuild (planned)/Experiment Review/Agent Reports/domdolus_cell10_report_2026-11-04.md`

**What cell10 is.** It is the "QSA DCP" patch and config cell, not the launch itself.
- It applies 25 anchor-replace edit lists to 23 files of stock vLLM 0.31.0 at `/tmp/site-vllm/vllm`, and adds one new Triton file, `ops/qsa_dcp.py`.
- It is idempotent: it keeps `.qsa_dcp.bak` backups and runs `py_compile` on each patched file. Setting DCP_SIZE=1 restores everything.
- It exports about 30 `FLASHNEXT_*` env vars and builds `MODEL_DIR`.
- It prepares a background import check that step 6 must pass before the server counts as ready.
- The server command is built in cell43 (`build_cmd`), which also reads cell10's `CUDAGRAPH_CAPTURE_SIZES`.

**Key config values:**
- DCP_SIZE=2. The comment says Flash-Next has 2 KV heads at TP 4, so 2 is the only valid size.
- COMPRESSED_BLOCK_FACTOR=8, GPU util 0.945 (the step-1 default is 0.92), indexer logits cap 128 MB, DCP comm backend a2a.
- CUDA graphs: sizes 1 to 512 including 72, max 512.
- MAX_BATCHED_TOKENS=1024. It is applied after CLI parsing, so it overrides cell43's explicit 4096.
- Image cap max_pixels 2097152, video off, expandable_segments on.
- FP8_HC_LMHEAD=True and FP8_DENSE_OVERLAY=False.
- DeltaNet state FP16 with stochastic rounding (the default is FP32). Memory trims on.
- Prefix reuse on: effective match unit 800, dense retention, partial entries inside the generated reply.
- QAR (compressed all-reduce) is None, so it is off.
- SINK_JUMP=("auto", 49152, 65536). TRIM_FREE and QSA_SKIP_GAP on.
- KV offload 30 GiB of host RAM with the evict variant, diagnostics on.
- Defaults the patches read but cell10 never sets: OFFLOAD_AHEAD 4, OFFLOAD_PRESSURE 0.8, SW_CHECK and TRIM_CHECK off.

**Techniques the report documents mechanism by mechanism:**
1. **QSA decode-context parallel.** The KV is sharded over the 2 TP ranks that share a KV head. Each rank filters the selected indices it owns, runs a split-K partial with LSE, and vLLM's `dcp_combine` merges them. The split-K constants are 4 / 32 / 8, "measured on L4". The indexer's compressed keys and the ring buffer stay whole on every rank, and the compressed-key blocks are widened 8x. The cell's own measurement: 16.73x instead of 13.85x max concurrency at 32K.
2. **FP16 stochastic-rounding GDN state.** Patched at 3 store sites in `fused_recurrent.py`.
3. **FP8 lm_head and FP8 hyper-connection mixers.**
4. **Memory trims.** Rotary cache sized to max_model_len, one shared top-k buffer for all QSA layers, Humming padding 64 for the hyper-connection layers.
5. **Prefix hits inside the generated reply.**
6. **QAR int8 a2a/rs compressed all-reduce.** Fully built with fused kernels, but inactive.
7. **Sink-jump selection mask with gap-skipping scoring kernels.** The sink is the prompt before the first `<|im_start|>assistant`; attention is cut back to 49,152 tokens each time it reaches 65,536.
8. **Trim-free.** `FnTrimGapManager` really frees the cut-away KV blocks. Per-request admission is capped at sink + 65,536 + 1,024 tokens, and cache-hit search is gap-aware.
9. **Evict-time RAM offload.** It is forced onto `SimpleCPUOffloadConnector` with lazy_offload, frees segment-ordered, and holds only the needed DeltaNet states.
10. **Diagnostics** (log only).

**Dependencies.**
- The model must be version 2 of `qwen38-flash-next-q2-gsq-vllm` at `.../pytorch/default/2`, which has the FP8 layers merged in.
- Optional overlay datasets: `darrenamadeusmartin/qwen38-flash-next-fp8-dense` and `...-fp8-dense-hc`.
- Host RAM must have MemAvailable of at least 38 GiB (30 + 8), or the cell stops.
- No pip wheels, tar files or kernel caches are referenced in cell10.

**Memory accounting.** Cell10 does not state weights GiB or KV GiB. It tells the user to read them from the server log: the "GPU KV cache size" / "Available KV cache memory" lines, the WEIGHTS DIAG lines, and cell09's FLASHNEXT KVDIAG lines. Context from the neighbour cells: MAX_LEN 1048576, MAX_SEQS 128, KV dtype fp8, mamba-cache-mode align, MTP 0, expert parallel on, engram cpu_offload, harness CONCURRENCY 70.

**Tunable observations** (each with its line evidence in the report):
- QAR is off, although the comments measure prefill all-reduce as bandwidth-bound (~4.9 GB/s through host memory, 3.4 ms at 10.5 MB).
- The prefill chunk is fixed at 1024.
- The block factor is already at the maximum the assert allows.
- GPU util is 0.945, and the assert allows up to just under 0.99.
- CUDA graph sizes 256/512 are larger than MAX_SEQS 128 and CONCURRENCY 70.
- The sink-jump keep/limit values (49,152 / 65,536) can be lowered.
- The 30 GiB RAM tier is limited only by the +8 GiB headroom check.
- The OffloadingConnector edits do nothing while OFFLOAD_EVICT=True.
- The DCP split-K constants are hard-coded, and the KV interleave size is left at vLLM's value.
- Custom all-reduce is explicitly ruled out: "measured 5x slower than NCCL".
