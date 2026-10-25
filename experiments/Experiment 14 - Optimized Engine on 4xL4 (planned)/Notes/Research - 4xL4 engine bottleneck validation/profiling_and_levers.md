# Profiling SGLang 0.5.21 decode of Qwen3.8-27B (INT4 Marlin + BF16 DFlash2, fp8 KV, fp16 SSM, TP4) on Kaggle 4x L4, and the engine levers beyond TP2 / DFlash width

Research date: 2026-10-23. Scope is design and research only; nothing here was run on Kaggle.

Labels used on every claim:
- **[M]** measured by someone and reported with numbers.
- **[C]** claimed or documented, with no run shown.
- **[S]** read directly in the local SGLang 0.5.21 source at `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang/` (path given relative to that root).
- **[L]** from our own Kaggle logs or notes (local path given).
- **[I]** my inference or arithmetic.

**Reference facts used below**
- Model, from a local config.json [L]: hidden 5120, intermediate 17408, 24 Q heads, 4 KV heads, head_dim 256, GDN key heads 16 and value heads 48 (dim 128), 64 layers (16 full attention, 48 GDN), vocab 248,320.
- Production E11-06 server args [L] (`experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/Kaggle Runs/E11-06 P1-60K EVAL14 run1 (esfersami50, 2026-10-19)/q27x/sglang_server_try0.log`):
  - `attention_backend=flashinfer`, `linear_attn_backend=triton`, `kv_cache_dtype=fp8_e4m3`, `mamba_ssm_dtype=float16`, `mamba_radix_cache_strategy=extra_buffer_lazy`, `max_mamba_cache_size=56`
  - `max_running_requests=13`, `chunked_prefill_size=4096`, `disable_overlap_schedule=False`, `enable_mixed_chunk=False`
  - `stream_interval=1`, `incremental_streaming_output=True`, `sampling_backend=flashinfer`, `enable_torch_compile=False`, `num_continuous_decode_steps=1`
  - `speculative_algorithm=DFLASH`, `speculative_num_draft_tokens=8`, `speculative_attention_mode=prefill`, `speculative_draft_model_quantization=None`
  - `cuda_graph_config.decode = {backend: full, max_bs: 12, bs: [1,2,3,4,5,6,7,8,10,12]}`; prefill graph disabled; `tp_size=4`; NCCL 2.29.7.
- Step time at 12 running: about 115–133 ms, inferred from decode lines [L] ([note 00](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/00 Claude - step-time analysis and TP2 gain (2026-10-23).md>); [note 01](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/01 GPT - GPU bottleneck research and validation (2026-10-23).md>)).

---

## Q1. Profiling tools in SGLang 0.5.21, Nsight Systems in a Kaggle container, and turning a trace into cost categories

### Takeaway
SGLang 0.5.21 already has everything needed to profile without new packages:
- `/start_profile` + `/stop_profile`, with `num_steps`, `start_step`, `profile_by_stage` and `CUDA_PROFILER`.
- `SGLANG_TORCH_PROFILER_DIR`.
- `--enable-profile-cuda-graph`, which profiles graph capture only.
- `--enable-layerwise-nvtx-marker`, which needs CUDA graphs off.
- Two zero-overhead counters that are not in the docs:
  - `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1` adds "fwd occupancy" (GPU busy ÷ wall) to every decode log line.
  - `SGLANG_RECORD_STEP_TIME=1` adds a per-batch-size `step_time_dict` to `/get_server_info`.

Kernels inside CUDA-graph replays do appear in torch-profiler (Kineto/CUPTI) traces, but on one stream, with no Python stack. So categories must come from kernel-name rules plus the CPU range that launched each graph.

Nsight Systems has no pip wheel. It is usable offline only if its standalone CLI is uploaded as a dataset. It is optional; the torch profiler is enough for the category split.

### Cited Findings
**Endpoints and request fields**
- [S] `/start_profile` and `/stop_profile` are registered for GET and POST in `srt/entrypoints/http_server.py:1164,1175`. `ProfileReq` (`srt/managers/io_struct.py:2257-2283`) has these fields:
  - `output_dir`, `start_step`, `num_steps` (auto-stop after N steps)
  - `activities` (comment: `["CPU", "GPU", "MEM", "RPD"]`)
  - `profile_by_stage`, `with_stack`, `record_shapes`, `profile_id`, `merge_profiles`, `profile_prefix`, `profile_stages`
  - `detailed_annotations` ("iteration-level annotations (KV / request aggregates) for roofline-style analysis")
- [S] `CUDA_PROFILER` is a real activity: `srt/utils/profile_utils.py:270,406-412` calls `torch.cuda.cudart().cudaProfilerStart()/Stop()`, which is the hook for `nsys --capture-range=cudaProfilerApi`.
- [S] Behaviour of the torch profiler wrapper (`profile_utils.py:316-377`):
  - `with_stack` defaults to True; `record_shapes` defaults to False.
  - One file is written per TP rank: `<profile_id>-TP-<rank>.trace.json.gz`.
  - It ends with a `torch.distributed.barrier`.
  - `merge_profiles` is asserted off in one path ("TODO: migrate `_merge_profile_traces`").
- [S] Environment variables in `srt/environ.py:451-469`:
  - `SGLANG_TORCH_PROFILER_DIR` (default `/tmp`)
  - `SGLANG_PROFILE_WITH_STACK`, `SGLANG_PROFILE_RECORD_SHAPES`, `SGLANG_PROFILE_V2`
  - `SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS`: "do not start the decode-stage capture until a decode batch reaches this many requests … capture steady-state full-admission decode steps instead of the ramp-up"
  - `SGLANG_ENABLE_NVTX_SCHEDULER`, `SGLANG_ENABLE_NVTX_OPERATIONS`
  - `SGLANG_RECORD_STEP_TIME`, `SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE`, `SGLANG_GRAPH_BATCH_CAPTURE`, `SGLANG_MEM_PROFILE_MAX_ENTRIES`
- [S] `SGLANG_RECORD_STEP_TIME=1` appends `gap_latency / decode_log_interval` to `step_time_dict[num_running_reqs]` (`srt/managers/scheduler_components/metrics_reporter.py:908-911`). The dict is returned in server info (`srt/managers/scheduler.py:5283-5284`).
  - It is a wall-clock mean per decode iteration, keyed by real batch size. This is exactly the "step ms at 12 running" number, with no profiler cost.
  - Next to it, `avg_spec_accept_length` is returned (`scheduler.py:5274-5281`).
- [S] `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1` makes the decode log line include `fwd occupancy: X%` and `fwd+idle occupancy: Y%` (`metrics_reporter.py:192-199,758,1024,1256-1290`), computed from GPU forward time over wall time.
  - It measures host or scheduler gaps directly, and is cheap enough to leave on in production.
- [S] NVTX: `srt/utils/nvtx_utils.py:39-72` emits ranges named `scheduler.recv_requests`, `scheduler.get_next_batch_to_run`, `scheduler.run_batch` and `scheduler.process_batch_result`.
  - These are also `torch.profiler.record_function` ranges, so they show in torch traces too.
  - The `nvtx` pip package is needed for real NVTX; otherwise it warns that "NVTX markers are disabled; torch profiler spans still emit".
- [S] `--enable-layerwise-nvtx-marker`, `--enable-profile-cuda-graph`, `--debug-cuda-graph` (breakable graph, eager ops through the capture path) and `--disable-decode-cuda-graph` exist in `srt/arg_groups/fields/exec_.py:546-590`.
- [S] With `--enable-profile-cuda-graph` plus `SGLANG_GRAPH_BATCH_CAPTURE`, one capture trace is produced per batch size (`srt/model_executor/runner_backend/full_cuda_graph_backend.py:121-133`). It profiles capture, not replay.

**Benchmark tools**
- [S] `sglang.benchmark.one_batch`, `one_batch_server` ("Benchmark the latency of running a single batch with a server", supports `--profile --profile-by-stage`), `serving`, `offline_throughput` and `steady_state` all exist in `benchmark/`.
- [S] `serving.py` dataset choices include `random`, `random-ids` and `generated-shared-prefix` (`benchmark/serving.py:2284-2294`).
  - `random` downloads ShareGPT when no file is present (`benchmark/datasets/random.py:84-89`).
  - `random-ids` sets `random_sample=False` (`random.py:39`), so it is the offline-safe synthetic option.
- [S] `serving.py` profile flags: `--profile`, `--profile-activities`, `--profile-start-step`, `--profile-steps`, `--profile-num-steps`, `--profile-by-stage`, `--profile-stages`, `--profile-output-dir` (`serving.py:2531-2567`).
- [C] SGLang docs ([benchmark_and_profiling.mdx](https://github.com/sgl-project/sglang/blob/main/docs/docs/developer_guide/benchmark_and_profiling.mdx)):
  - Set `SGLANG_TORCH_PROFILER_DIR` on server and client.
  - `SGLANG_PROFILE_WITH_STACK=False` is the workaround for the "Python replay stack is empty" crash.
  - `CUDA_PROFILER` pairs with `nsys --capture-range=cudaProfilerApi --capture-range-end=stop`.
  - nsys needs `--trace-fork-before-exec=true --cuda-graph-trace=node`.
  - Layer-wise NVTX "Add `--disable-cuda-graph`. Without it, markers are not emitted for CUDA-graph kernels."
  - "To map kernels back to SGLang Python source, start the server with `--disable-cuda-graph`."
  - `--enable-profile-cuda-graph` profiles "only the decode CUDA-graph runner" capture.
  - Detailed annotations label "speculative decoding verify steps … as `TARGET_VERIFY`".

**Graph replays in traces**
- [C] PyTorch tutorial ([CUDA graph annotations](https://docs.pytorch.org/tutorials/advanced/cuda_graph_annotations_tutorial.html)): without annotations, graphed kernels are shown, but "all kernels appear on the same stream, making it difficult to understand the logical structure".
  - Graphed NCCL kernels "still appear, but without the metadata eager mode records".
  - Kernel annotations need PyTorch 2.15+ and driver/CUDA-compat ≥ 13.1; on older stacks `mark_kernels` is a no-op. Our torch is 2.13, so no annotations.
- [C] PyTorch profiler source sets `TEARDOWN_CUPTI=0` when CUDA graphs are used, because of crashes on repeated CUPTI init ([torch.profiler source 2.5](https://docs.pytorch.org/docs/2.5/_modules/torch/profiler/profiler.html)). One profiling window per server process is the safe pattern.
- [C] NVIDIA: nsys `--cuda-graph-trace=graph` (the default) is low-overhead; `=node` gives per-kernel detail "but may cause significant runtime overhead" ([NVIDIA CUDA Graph performance issues](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/performance-issues.html)).

**Nsight Systems availability**
- [C] There is no PyPI package for nsys in any result. NVIDIA's route for no-root use is the CLI from the standalone package's Target directory, copied anywhere and set up with its script ([NVIDIA forum getting-started](https://forums.developer.nvidia.com/t/getting-started-guide/64426); [conda thread: no nsys package, use .run](https://forums.developer.nvidia.com/t/installing-nsys-with-conda/246439); [install-profilers guide: .deb / .run / `nsight-systems-cli`](https://mcpservers.org/id/agent-skills/nvidia/install-profilers)).
- [C] The Arch package ships `install-dependencies-without-root` helpers ([archlinux file list](https://archlinux.org/packages/extra/x86_64/nsight-systems/files)).
- [L] The Kaggle container blocks unshare/mount; the sandbox is subprocess + rlimits + seccomp + setuid (project memory `kaggle-arc2-platform-limits.md`). No note in our workspace shows nsys present in the Kaggle image: a grep of the experiments found nsys only in prose.
- [C] Kaggle/vLLM note: FlashInfer sampling would not JIT on Kaggle because "the image has no linkable libcuda.so" ([topic 732723 dump](<D:/ARC AGI 2/resources/Analysis and Journals/Kaggle discussions (2026-10-22)/topic_732723.md>)). JIT and tool linking problems are real in this image.

### Inferences
**Recommended tool stack, in order of cost**
1. **Free counters, kept on in every run** [I from S]:
   - `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1` and `SGLANG_RECORD_STEP_TIME=1`.
   - Read `/get_server_info` (step_time_dict, avg_spec_accept_length) at the end of each steady window.
   - If fwd occupancy is ≥ 95 % at 12 running, host or scheduler gaps are not the problem. If it is ≤ 85 %, CPU-side levers (stream interval, tokenizer workers, scheduler) move up the ranking.
2. **One torch-profiler window per server launch** [I]:
   - Call `POST /start_profile` with `{"num_steps": 20, "activities": ["CPU","GPU"], "with_stack": false, "record_shapes": false, "profile_by_stage": true, "profile_id": "tp4w8_ctx40k"}`.
   - Set `SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS=12` so only full-admission decode steps are captured.
   - With stacks off, the CPU overhead per step is small and the trace stays compact. Estimate: about 20 steps × about 2–3k kernels × 4 ranks, roughly 10–30 MB gz per rank.
3. **Optional per-layer eager window, only if graph-level categories are ambiguous** [I]:
   - Relaunch with `--disable-decode-cuda-graph` and `--enable-layerwise-nvtx-marker`.
   - Eager timings inflate launch gaps, so use it only to attribute kernel names to modules, never for step time.
4. **nsys only if a standalone CLI tarball is pre-uploaded** [I]:
   - Command: `nsys profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --cuda-graph-trace=graph --trace-fork-before-exec=true --capture-range=cudaProfilerApi --capture-range-end=stop` around `launch_server`, triggered with `activities:["CUDA_PROFILER"]`.
   - `--sample=none` and `--cpuctxsw=none` avoid needing perf_event privileges. CUPTI GPU tracing needs no root.
   - Its advantage over the torch trace is cleaner per-rank NCCL timing. It is not required.

**Parsing a trace into categories (design)** [I]
- Load each `*-TP-r.trace.json.gz` and keep events with `cat == "kernel"`. Group them into steps by the CPU `cudaGraphLaunch` runtime events, joined to GPU kernels through `args.correlation` / `External id`.
  - One DFlash step has a draft graph launch and a target-verify graph launch. Tag each launch by the enclosing CPU `record_function` range (`scheduler.run_batch`, and draft vs verify forward names) to separate drafter kernels from target kernels.
- Classify kernel names with ordered regexes:

  | Category | Regex |
  |---|---|
  | NCCL all-reduce | `nccl.*AllReduce` |
  | NCCL all-gather (logits) | `nccl.*AllGather` |
  | Marlin INT4 GEMM | `[Mm]arlin` |
  | BF16 GEMM, cuBLAS/CUTLASS (lm_head, drafter, `in_proj_a/b`) | `gemm\|gemv\|cutlass\|xmma\|sm8\d_\|ampere_\|cublas` |
  | GDN Triton | `fused_sigmoid_gating_delta_rule\|fused_recurrent\|gated_delta\|chunk_gated\|causal_conv1d\|conv1d\|l2norm` |
  | Attention (flashinfer prefill-template kernels used for verify; triton `_fwd_kernel` if switched) | `BatchPrefill\|BatchDecode\|PrefillWithKVCache\|flashinfer.*(attention\|prefill\|decode)\|_fwd_kernel\|decode_attention` |
  | Sampling / accept / commit | `sampling\|top_k\|top_p\|renorm\|softmax\|argmax\|verify\|accept\|dflash\|tree` |
  | Norm / rope / activation / quant elementwise | `rms\|norm\|rope\|rotary\|silu\|act_and_mul\|quant` |
  | Copies | `Memcpy\|Memset` |
  | Other | everything else |

- **Per step**, report:
  - the busy union of kernel intervals on the GPU
  - step wall time, from one verify-graph launch to the next
  - host gap = wall − busy union
  - per-category sum and share
- **NCCL needs special care:** an NCCL kernel's duration includes waiting for the slowest rank.
  - Per collective (matched by order within the step on the 4 ranks), take the minimum duration across ranks as the transfer-time estimate. Max − min is rank skew.
  - Do not sum the 4 ranks: GPT note 01 warns about double counting across overlapping execution.
- **Cross-check:**
  - Σ categories (TP0) should match step_time_dict[12] within about 10 %.
  - Accepted tokens per step (from accept len) × 12 ÷ step time should match the log's gen throughput.

### Gaps
- Whether the Kaggle GPU image ships `nsys` (for example under `/usr/local/cuda*/bin` or `/opt/nvidia/nsight-systems`) is unknown. It needs a 1-line probe in the first cell.
- Exact Kineto kernel names for our Marlin, FlashInfer fp8-KV prefill and NCCL 2.29.7 builds are not documented. The regexes need a first pass over a real trace.
- Whether `profile_by_stage` handles the DFlash `TARGET_VERIFY` stage as "decode" for `SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS` was not traced through the code.

---

## Q2. A single bounded, fail-closed Kaggle run (topology, NCCL, TP4 DFlash8 baseline, profiler, greedy check, optional width relaunch): wall time, quota, stop rules

### Takeaway
A three-tier run fits:
- **Tier A** (topology + NCCL + TP4 DFlash8 at 12 streams + one profiler window + DFlash serial-vs-concurrent greedy) needs about 35–45 min wall (about 1.2–1.5 h quota).
- **Tier B** adds a target-only relaunch for the real correctness reference, about +10 min.
- **Tier C** adds widths 6 and 4, about +12–15 min.

The full run is about 60–70 min wall, about 2.0–2.4 h quota at 2×. It fits hosen42 (8.63 h left) and is tight on foysal (4.71 h).

The storage risk dominates. Copy the 23.3 GB of weights to local disk in parallel first, so relaunches do not re-read the GCSFUSE mount. Every phase gets its own cap, and results are flushed after every phase.

### Cited Findings
**Startup and storage**
- [L] Cold start of SGLang "ready in": 507–562 s on 10-19/10-20, but **1,197 s** on 10-21 (GCSFUSE). Weights read are INT4 19.47 GB + DFlash2 3.85 GB = 23.3 GB ([Kaggle discussions 2026-10-22 summary](<D:/ARC AGI 2/resources/Analysis and Journals/Kaggle discussions (2026-10-22)/00_SUMMARY.md>)).
- [L] GPT note 01: E11-05 ready in about 7 min 39 s, E11-06 about 7 min 30 s, rev 6d G4B about 19 min 9 s. CUDA-graph capture is the big part when storage is healthy ([note 01](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/01 GPT - GPU bottleneck research and validation (2026-10-23).md>)).
- [L] Relaunches inside one session are much cheaper: first launch 601 s (prefill-graph JIT 447 s), later launches 136–166 s ([engines_and_config.md §1](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/Research - Qwen 27B fastest serving setup/engines_and_config.md>), citing E11-03).
- [M] Kaggle read speed on 10-22: single thread about 10.5 MB/s, 16 threads about 107 MB/s. GCSFUSE "reading through the mount did not fill the page cache" (Jeroen Cottaar) ([topic 751344](<D:/ARC AGI 2/resources/Analysis and Journals/Kaggle discussions (2026-10-22)/topic_751344.md>)).
- [C] mp's mitigation: detect FUSE and copy to local disk once instead of 4 GPU processes reading the mount ([topic 751344](<D:/ARC AGI 2/resources/Analysis and Journals/Kaggle discussions (2026-10-22)/topic_751344.md>)).

**Quota and machine**
- [C] L4×4 bills at 2×; 30 h weekly ≈ 15 h of 4×L4 ([topic 689054](<D:/ARC AGI 2/resources/Analysis and Journals/Kaggle discussions (2026-10-22)/00_SUMMARY.md>), Sam-wiz). Quota left: esfersami50 30.21/30 h used; foysal 4.71 h; hosen42 8.63 h ([E14 README](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/README.md>)).
- [L] The production log shows `Triton kernel 'chunk_gated_delta_rule_fwd_kkt_solve_kernel' took 1.74–1.94 s to compile after serving started`: 24 lines (6 per rank), about 11 s total, in E11-05 and E11-06.
  - It is a one-time stall after the first long prefill. The benchmark must warm a long prefill before the steady window.
- [L] The E11-03 preflight 64 MB all-reduce took 20.9 ms, about 3.1 GB/s algorithm bandwidth. The TP4 log says custom all-reduce is "not supported on more than two PCIe-only GPUs" ([note 00](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/00 Claude - step-time analysis and TP2 gain (2026-10-23).md>); [E11-06 log](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/Kaggle Runs/E11-06 P1-60K EVAL14 run1 (esfersami50, 2026-10-19)/q27x/sglang_server_try0.log>) lines 33/37).

**NCCL measurement guidance**
- [C] NVIDIA/NCCL: use `nccl-tests` to separate small-message latency from bandwidth ([nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)).
  - `NCCL_DEBUG_SUBSYS` accepts INIT, P2P, SHM, GRAPH, TUNING and others; `NCCL_TOPO_DUMP_FILE` writes the detected topology XML; `NCCL_P2P_LEVEL`/`NCCL_P2P_DISABLE`/`NCCL_SHM_DISABLE` control transports.
  - The NCCL docs warn that forcing such variables "will prevent NCCL from selecting the best setting automatically" and should not stay in production ([NCCL env docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)).
- [C] Per-hop protocol latency is about 6 µs Simple, 1 µs LL, 2 µs LL128. PCIe topologies need MB-scale messages before bandwidth, not startup latency, dominates ([Demystifying NCCL, arXiv 2507.04786](https://arxiv.org/html/2507.04786v2)).
- [C] `nccl-tests` is not in pip wheels; the GPT note suggests an offline PyTorch distributed benchmark instead ([note 01](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/01 GPT - GPU bottleneck research and validation (2026-10-23).md>) §8).

**Correctness gates and stop discipline**
- [C] Known DFlash2 risks give the reason for a greedy gate: #36548 (output inconsistency under concurrency) and #38009 (greedy divergence from target-only with thinking on) ([note 01](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/01 GPT - GPU bottleneck research and validation (2026-10-23).md>) §7; [sglang #38009](https://github.com/sgl-project/sglang/issues/38009)).
- [L] Project rules that bind this design:
  - Assert real running == target and read the server caps ([verify-real-concurrency memory](C:/Users/hp/.claude/projects/d--ARC-AGI-2/memory/MEMORY.md)).
  - Save full outputs of every request (always-save-thinking-traces).
  - Do a pre-run capacity check.
  - Never push or run without the user's go.
- [C] Kaggle tip: do not unpack wheels under `/kaggle/working`; a finished run was marked ERROR at teardown because of 7 GiB of output ([topic 732723](<D:/ARC AGI 2/resources/Analysis and Journals/Kaggle discussions (2026-10-22)/topic_732723.md>)).

### Inferences
**Design: "E14-A" one notebook, fail-closed.** Times are planned caps; nominal values are in brackets. The notebook clock T starts at cell 1.

| Phase | Content | Nominal | Hard cap | Output (flushed at phase end) |
|---|---|---|---|---|
| P0 env + topology | `nvidia-smi -q` (clocks, power limit 72 W, PCIe gen/width, throttle reasons); `nvidia-smi topo -m`; `topo -p2p r/w/n`; `torch.cuda.can_device_access_peer` 4×4; `lscpu`, `nproc`, `free -g`, `df -h /kaggle/working /tmp /dev/shm`; mount type of `/kaggle/input` (`stat -f -c %T`); `which nsys`; NCCL/torch/sglang versions | [1 min] | T+3 | `p0_env.json` |
| P1a weight copy | Parallel range-read copy of the INT4 + DFlash2 dirs to local disk (16–32 threads, 64 MB chunks), verified by size + sha of the index files; `nvidia-smi dmon -s pucvmet -d 2` logger starts here and runs to the end | [4–6 min at ≥ 100 MB/s; much longer on slow GCSFUSE] | T+25 | `p1a_copy.json` (MB/s per file) |
| P1b NCCL microbench (concurrent with P1a; GPU-only, light CPU) | torchrun 4 procs, NCCL backend, `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,GRAPH,P2P,SHM,TUNING NCCL_TOPO_DUMP_FILE=...`; all_reduce bf16 at 8K,16K,…,1M, 2M, 64M (warm 20, iters 100, CUDA events); **plus the same 983 KB (12×8×5120×2 B) all-reduce ×128 captured in one CUDA graph and replayed** (the SGLang-like case); plus an all_gather of 47.7 MB (96×248,320×2 B, the logits gather); repeat 8K–1M on pairs (0,1), (2,3), (0,2), (1,3) for TP2 | [5–7 min] | T+12 | `p1b_nccl.json`, NCCL INFO log, topo XML |
| P2 launch TP4 DFlash8 | Production flags plus `SGLANG_ENABLE_METRICS_DEVICE_TIMER=1`, `SGLANG_RECORD_STEP_TIME=1`, `SGLANG_TORCH_PROFILER_DIR=/kaggle/working/e14a/prof`, `SGLANG_PROFILE_WITH_STACK=0`, `SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS=12`, fresh `SGLANG_CACHE_DIR`/`CUDA_CACHE_PATH`, model path = local copy | [7–8 min] | 20 min after start | `p2_launch.json` (ready_sec, KV tokens, mamba cap, captured bs list) |
| P3 warm + steady windows | 12 concurrent requests from a fixed local prompt pack (real ARC prompts with thinking on, production sampling), long shared-prefix padding to reach about 8k and about 40k context; first a 1-request long prefill to absorb the GDN Triton JIT; then **window W8k** and **window W40k**: 90 s each at real running = 12 (asserted from decode lines), `max_tokens` large, `ignore_eos` | [8–10 min incl. prefill of 12×40k] | 14 min | `p3_steady.json` (step_time_dict, gen tok/s, accept len, fwd occupancy per window) |
| P4 profiler | One `/start_profile` (num_steps 20, CPU+GPU, no stack, by-stage) during W40k; wait for 4 trace files; parse TP0 + TP3 in a subprocess with a 4-min timeout | [3 min] | 6 min | traces + `p4_categories.json` |
| P5 greedy gate (DFlash) | 24 fixed prompts, T=0, thinking on, max_tokens 768: serial (c=1), then concurrent (c=12); save full text + token ids | [5 min] | 8 min | `p5_dflash_serial.jsonl`, `p5_dflash_c12.jsonl` |
| **End of Tier A** | about 35–45 min | | | |
| P6 relaunch target-only (Tier B) | Same flags, no speculative args; same 24 prompts serial + c=12; plus a 60 s W40k window (gives the no-spec step time at 12 running for decomposition) | [3 min launch + 6 min] | 12 min | `p6_target.jsonl`, `p6_steady.json`; computes token-ID match rate and first divergence vs P5 |
| P7 relaunch width 6, then width 4 (Tier C) | `--speculative-num-draft-tokens 6` / `4` (record the `set_block_size()` warning from `dflash_worker_v2.py:484-492`); W40k 90 s + greedy c=1 on 8 prompts each | [2×(3+4) min] | 2×8 min | `p7_w6.json`, `p7_w4.json` |
| P8 teardown | SIGTERM, then SIGKILL the server process group; tar `/kaggle/working/e14a` (logs, JSON, traces ≤ about 300 MB); write `DONE` | [1 min] | global cap | `e14a.tar.gz` |

- **Wall time:**
  - Tier A about 35–45 min, Tier B about 45–55 min, Tier C about 60–70 min (nominal storage).
  - Under the 10-21 GCSFUSE regime, P1a alone could take 25 min (cap), pushing Tier C to about 90 min.
- **Quota (2×):** Tier A about 1.2–1.5 h, full about 2.0–2.4 h, worst case about 3 h. Use hosen42 (8.63 h) for the full run; foysal (4.71 h) only for Tier A+B.
- **Global cap:** set the global wall cap at about 80 min (Tier C), and the notebook-level stop at the cap minus 5 min.

**Fail-closed rules** [I]
1. Results go to `/kaggle/working/e14a/` as append-only JSON lines, with fsync after each phase. A `phase_status.json` records start/end/cap/outcome for every phase. Nothing depends on the end of the notebook to be saved.
2. The global deadline is checked before every phase. If the remaining time is less than the phase cap, skip the phase and mark it `skipped_deadline`.
3. P1a not finished at T+25 → do not launch SGLang from the mount. Keep P0/P1b and stop. A launch on the slow mount is the known 0-result trap (1,197 s cold start).
4. Server not ready by the cap → kill the process group, save the last 3,000 log lines, skip all server phases, and go to P8.
5. Real running ≠ 12 for more than 60 s of a window (from the decode log) → mark the window `invalid_concurrency` and do not use its tok/s. Still keep the profile if it was taken at 12.
6. Accept len < 1.5 for 3 consecutive decode lines (the #43514-style collapse) → mark `spec_collapse`, take no profile, and go to P5 (the correctness gate now matters most).
7. `/start_profile` returns `success=false`, or no trace appears within 120 s after the steps → continue without a profile. Never issue a second start while one is in progress (the source refuses it anyway, `profiler_manager.py`). One profiling window per server process (CUPTI teardown note).
8. The trace parser runs in a subprocess with a timeout. Raw traces are always kept, even if parsing fails.
9. Memory: no wheels or venvs under `/kaggle/working`. The output tar is capped at about 300 MB; drop traces beyond TP0/TP3 if over.
10. GPU health: if `nvidia-smi dmon` shows `pwr` pinned at 72 W with SM clock below about 1.5 GHz during W40k, record it as a throttling confound. Do not treat it as an engine result.
11. The run is research-only. It submits nothing and changes no production notebook. Building it must use a new notebook name (never rebuild a run notebook), and it needs the user's explicit go.

**Synthetic vs real prompts** [I]
- `random-ids` prompts are offline-safe but would distort DFlash acceptance: random tokens produce unrepresentative drafts. Step time per verified token is roughly content-independent, but tok/s is not.
- Use real ARC prompts (stored in the notebook) with long shared prefixes, so accept len stays near production (about 4.3–4.5 at 8–12 running [L]).

### Gaps
- Kaggle 4×L4 local disk free space and `/dev/shm` size are unknown, so P1a's target directory must be chosen at runtime from `df`. The host has 192 GB RAM per the [g2-standard-48 spec](https://cloudprice.net/gcp/compute/instances/g2-standard-48).
- Whether P2P is enabled on Kaggle's g2 VMs is not documented anywhere I found; [Google's G4 blog](https://cloud.google.com/blog/products/compute/g4-vms-p2p-fabric-boosts-multi-gpu-workloads/) only discusses G4. P1b decides it.
- How long `/start_profile` trace export takes per rank on Kaggle's CPU was not measured. The 6-min cap is a guess.

---

## Q3. Engine-level levers (other than TP2 and DFlash width), ranked by evidence

### Takeaway
A rough per-step floor from bandwidth arithmetic at 12 × 8 verify tokens and about 40–50k context is about 45–55 ms of a measured 115–133 ms step [I]. The remaining 60–80 ms is unattributed.

The most likely hidden costs are PCIe collectives:
- 128 all-reduces per target pass,
- plus a 47.7 MB logits all-gather per verify (new finding from the source),
- plus kernel inefficiency of Marlin at M≈96 and the untuned GDN verify kernel.

Evidence-ranked levers:
1. Restore real 12/12 running: the #44401 admission fix, merged after 0.5.21, plus graph sizes covering 9/11/13.
2. NCCL transport/protocol tuning, only if P1b shows a bad transport choice.
3. GDN verify-kernel launch-config tuning: hard-coded `BV=32, num_warps=1` on CUDA.
4. Attention backend A/B (flashinfer vs triton with fp8 KV).
5. Low or zero expected gain:
   - the overlap scheduler (already on)
   - `num_continuous_decode_steps` (a dead flag in 0.5.21)
   - torch.compile
   - stream interval
   - mixed chunk, which is risky with DFlash

Chunked-prefill interference is real but driven by agent re-prefill, which is covered elsewhere.

### Cited Findings
**Step-time floor arithmetic** (inputs [L]/[C]; arithmetic [I])
- **INT4 weights:** about 4.9 GB per GPU per pass, about 16 ms at 300 GB/s ([weight_quantization.md roofline](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/Research - Qwen 27B fastest serving setup/weight_quantization.md>) lines 171-190; L4 300 GB/s [datasheet via engines_and_config](https://bxin.cisco.com/c/dam/en/us/products/collateral/servers-unified-computing/ucs-c-series-rack-servers/nvidia-l4-gpu.pdf)).
- **fp8 KV:** 8 KiB per token per GPU (16 attention layers × 1 KV head × 256 × 2). At 12 × 50k that is about 4.9 GB, about 16 ms. This matches the measured +18 ms per step going from 0 to 50k context ([note 00](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/00 Claude - step-time analysis and TP2 gain (2026-10-23).md>)).
- **GDN verify state traffic:**
  - Per GPU, 12 value heads × 128 × 128 × 2 B (fp16) = 393 KB per request per layer.
  - Writing 8 intermediate states × 12 requests × 48 layers is about 1.8 GB per step, about 6–7 ms. Reads add about 0.23 GB.
  - SGLang keeps per-draft-position snapshots for GDN verify ([sglang #28730](https://github.com/sgl-project/sglang/issues/28730): "for each draft position stores a full recurrent SSM state"; about 15.8× the normal state at D=16).
- **BF16 drafter:** 3.85 GB ÷ 4, about 1 GB per GPU, about 3 ms.
- **lm_head BF16:** 248,320 × 5120 × 2 ÷ 4 = 636 MB per GPU, about 2 ms per use (target verify; possibly again for draft logits).
- **Logits all-gather per verify:** [S] `srt/layers/logits_processor.py:913-927` gathers TP-sharded logits (`_logits_gatherer`) unless DP-attention all-to-all applies.
  - At 96 positions × 248,320 × 2 B that is 47.7 MB gathered per step (each rank receives about 36 MB).
  - At the measured about 3.1 GB/s (E11-03 64 MB all-reduce algbw), that would be about 10–15 ms [I]. It is unverified whether all-gather runs at the same effective rate, and whether the drafter does a second gather.
- **Per-layer all-reduces:** about 0.98 MB each (12 × 8 × 5120 × 2 B), 128 per target pass. At about 3.1 GB/s that is about 40 ms; if P2P works, about 8–12 ms. This is the note 00/01 hypothesis, unmeasured.
- **Sum of floors without collectives:** about 16 + 16–18 + 7 + 3 + 2–4 ≈ 45–50 ms [I].

**Lever: real concurrency and graph-size coverage (strongest evidence)**
- [L] E11-06 decode lines: 278 at 12 running, 158 at 11 running with `#queue-req: 1`, 400 at 9, 408 at 8 (1,299 lines, all `cuda graph: True`). At 11 running the mamba usage was 33 of 56 slots, so the cap was not the mamba pool.
- [C] SGLang PR #44401 (merged 2026-10-15, after 0.5.21): "Under chunked prefill this capped the running batch at `max_running_requests - 1`, and when the final chunk's pass set `batch_is_full`, nothing cleared it until a request finished, so the waiting queue starved meanwhile". It fixes #38118 and #35537 ([sglang PR #44401](https://github.com/sgl-project/sglang/pull/44401)).
- [L] 11 vs 12 running cost about 8 % in E11-05 ([engines_and_config.md §4](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/Research - Qwen 27B fastest serving setup/engines_and_config.md>)).
- [S] The default spec-decode capture list is `range(1,9) + range(10,33,2) + …`, with `max_bs` appended (`srt/arg_groups/cuda_graph_hook.py:640-668`).
  - [L] Our resolved decode graph list was `[1..8, 10, 12]` with `max_bs: 12`, while `max_running_requests=13`.
  - So batch 9 replays the 10-graph, batch 11 replays the 12-graph (padded rows are computed, about 8–10 % wasted verify work [I]), and 13 would run without the decode graph [I].
- [S] `--cuda-graph-bs-decode` (an explicit list) and `--disable-cuda-graph-padding` exist (`exec_.py:511,546`).

**Lever: NCCL transport and protocol**
- [L] The TP4 log says both CustomAllreduce and CustomAllReduceV2 are disabled ("not supported on more than two PCIe-only GPUs"). `multimem all-gather disabled (no multicast for world_size=4)`. `sglang is using nccl==2.29.7` ([E11-06 log](<D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/Kaggle Runs/E11-06 P1-60K EVAL14 run1 (esfersami50, 2026-10-19)/q27x/sglang_server_try0.log>) lines 33, 37, 145, 32).
- [S] The gate is `if world_size > 2 and not full_nvlink` → disabled (`srt/distributed/device_communicators/custom_all_reduce_utils.py:492-495`).
  - `--enable-torch-symm-mem` "Only supports CUDA device SM90 and above" (`exec_.py:611`).
  - `--enable-mscclpp` (MSCCL++ with NCCL fallback) exists (`exec_.py:607`); it is untested on PCIe L4.
- [M] Forcing custom all-reduce at TP4 on 3090 PCIe gave +14.8 % decode but greedy divergence on 4/5 prompts. TP2 with custom AR + verified P2P beat no-P2P by about +15.7 % ([club-3090 #773](https://github.com/noonghunna/club-3090/discussions/773), via [engines_and_config.md §3](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/Research - Qwen 27B fastest serving setup/engines_and_config.md>)).
- [C] Google: P2P gives "up to 2.2x" across collectives and up to 168 % higher vLLM TP throughput vs non-P2P offerings (vendor, G4/RTX PRO 6000, not L4) ([GCP G4 blog](https://cloud.google.com/blog/products/compute/g4-vms-p2p-fabric-boosts-multi-gpu-workloads/)).
- [M] On dual 5090, an NCCL P2P toggle gave only about +2–5 %, sometimes negative ([vLLM forum](https://discuss.vllm.ai/t/dual-rtx-5090-tp-2-shm-vs-patched-bar1-p2p-cumem-single-pass-2-7-mean-point-estimate/2870), via engines_and_config).
- [C] NCCL disables LL128 where 128-byte atomic writes are not guaranteed (PCIe limits). LL/LL128 are chosen for small messages ([arXiv 2507.04786](https://arxiv.org/html/2507.04786v2)).

**Lever: GDN Triton decode/verify kernel config**
- [L] The production dispatcher logs `decode=TritonGDNKernel, extend=TritonGDNKernel, verify=TritonGDNKernel packed_decode=True` and `Linear attention kernel backend: decode=triton, prefill=triton, verify=triton` (E11-06 log lines 126, 130).
- [S] Verify uses `fused_sigmoid_gating_delta_rule_update` (`srt/layers/attention/linear/kernels/gdn_triton.py:217-236`). Its launch config comes from `_select_recurrent_launch_config`, which is tuned only for AMD gfx95. On CUDA it returns `BV=min(next_pow2(V),32), num_warps=1`, with `num_stages=3` and grid `(NV, N, HV)` (`kernels/ops/attention/fla/fused_sigmoid_gating_recurrent.py:14-34,431-455`). There is no `triton.autotune` in this file.
- [C] The FLA maintainers' open RFC "Autotune should consider batch size and number of heads" ([fla #117](https://github.com/fla-org/flash-linear-attention/issues/117)). FLA autotune-related fixes exist for other kernels: re-tuning on new lengths ([#1147](https://github.com/fla-org/flash-linear-attention/issues/1147), [#1499](https://github.com/fla-org/flash-linear-attention/issues/1499)) and wrong results for some autotune configs on Blackwell ([#790](https://github.com/fla-org/flash-linear-attention/issues/790), [#945](https://github.com/fla-org/flash-linear-attention/issues/945)). These show config sensitivity, but none is on Ada.
- [C] FlashInfer GDN decode and prefill kernels need SM90+, so L4 stays on Triton ([FlashInfer GDN decode docs](https://docs.flashinfer.ai/generated/flashinfer.gdn_decode.gated_delta_rule_decode.html), via engines_and_config §1). A community packed GDN decode kernel gave about +7 % end-to-end on GB10 ([HF triton-gdn-decode](https://huggingface.co/baladhurgesh97/triton-gdn-decode)).

**Lever: attention backend with fp8 KV**
- [S] With fp8 KV, FlashInfer decode always uses the tensor-core (prefill-template) path: `should_use_tensor_core` returns True for `float8_e4m3fn` (`srt/layers/attention/flashinfer_backend.py:2492-2537`).
  - Spec verify runs in `speculative_attention_mode=prefill` [L], i.e. the prefill kernel with q_len 8 per request.
  - Both flashinfer and triton support fp8 KV and spec topk=1 ([SGLang attention backend docs](https://docs.sglang.io/docs/advanced_features/attention_backend), via engines_and_config).
- [L] Project experience: FlashInfer + fp8 KV ran 12 h on L4 under vLLM 0.19 ("FlashInfer FP8 KV crashes on L4 is not our experience") ([delongyang analysis](<D:/ARC AGI 2/resources/Analysis and Journals/Public notebook delongyang Qwen3.8-27B within 4 hours - deep analysis (2026-10-17).md>)). The public delongyang notebook forced TRITON_ATTN on vLLM.
- [C] sgl_kernel `is_fa3_supported()` accepts sm_89, but no sm_89 cubin ships ([sglang #38980](https://github.com/sgl-project/sglang/issues/38980)), so FA3 is not an option on L4.
- [C] On 3090 vLLM, the fp8-KV path (FlashInfer) vs int8-KV (Triton) crossover was about 7–10k context: Triton int8 was +76.5 % short-context but −71.9 % at 58k (club-3090 #773, via engines_and_config §4). This shows attention-backend choice is context-dependent.

**Lever: overlap scheduler, CPU overhead, continuous decode steps, stream interval**
- [L]/[S] The overlap scheduler is already on: `disable_overlap_schedule=False`. DFlash uses `DFlashWorkerV2`, which "drives both overlap and non-overlap" (`srt/speculative/spec_info.py:309-316`). `extra_buffer_lazy` requires overlap (`srt/mem_cache/kv_cache_configurator.py:2252`).
- [M] SGLang's zero-overhead overlap scheduler gave about 1.1× (v0.4, non-hybrid) ([LMSYS v0.4 blog](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4), via engines_and_config §6).
- [S] **`num_continuous_decode_steps` is a dead flag in 0.5.21.** A repo-wide grep finds it only in its field definition (`srt/arg_groups/fields/schedule.py:194`) and `field_order.py:67`; no scheduler code reads it.
- [S] `stream_interval` help: "A larger value makes the throughput higher". `batch_notify_size=16` "Reduces asyncio wakeup overhead under high concurrency" (`srt/arg_groups/fields/serving.py:255-267`). We run interval 1 with incremental output and 1 tokenizer / 1 detokenizer worker [L].
- [C] Open SGLang issue #43304 asks for metrics to localize bottlenecks in the multi-tokenizer / detokenizer IPC path, implying it is a known high-concurrency cost area (title only, from gh search).
- [C] Host CPU is g2-standard-48: 48 vCPU, Intel 2.2 GHz base / 3.7 GHz turbo, listed as Cascade Lake by a third-party aggregator ([cloudprice.net](https://cloudprice.net/gcp/compute/instances/g2-standard-48); [gcloud-compute.com](https://gcloud-compute.com/g2-standard-48.html)). Another lists a Skylake variant ([Cloud Mercato](https://pcr.cloud-mercato.com/providers/google/flavors/g2-standard-48.skylake)); the sources conflict.

**Lever: torch.compile**
- [S] Compile applies to the decode-graph runner for batch sizes ≤ `torch_compile_max_bs` (default 32) (`srt/model_executor/runner/base_cuda_graph_runner.py:96-101`; `exec_.py:566-575`). It is incompatible with `--disable-cuda-graph-padding` (`srt/arg_groups/validation_hook.py:107`).
- [M] SGLang v0.3: up to 1.5× lower latency at small batch sizes, on small models ([LMSYS v0.3 blog](https://lmsys.org/blog/2024-09-04-sglang-v0-3)). The docs say it "accelerates small models on small batch sizes"; the speculative-decoding docs warn the benefit "may be negligible" when CUDA graphs are on ([SGLang speculative decoding docs](https://docs.sglang.io/docs/advanced_features/speculative_decoding)).
- [M] Hybrid Qwen on single Ada sm_89 (RTX 4080 SUPER 32 GB, AWQ, v0.5.18): decode CUDA graph capture hung, the `tc_piecewise` decode backend is "not yet implemented; falling back to 'full'", and eager decode ran about 6 tok/s ([sglang #36048](https://github.com/sgl-project/sglang/issues/36048)). Compile and graph variants on sm_89 hybrids are fragile.

**Lever: chunked prefill and mixed chunk**
- [C] SGLang cookbook: `--chunked-prefill-size 2048`, "since 8192-token chunks stall decode about 600ms at a time" (via engines_and_config §4). We run 4096 [L].
- [L] Agent re-prefill is large: E12 EVAL30 had 8.39 M re-prefill tokens, and effective tok/s was about 187 vs about 412–431 decode ([note 05](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/05 Claude - gaps found after full re-read (2026-10-23).md>)).
- [M] The ARC-AGI-3 Duck write-up: prefill time fraction 15.25 % with 93.43 % prompt-token cache reuse; P90 prefill 10,552 tok/s (single RTX PRO 6000, Flash-Next) ([Franzen WRITEUP](<D:/ARC AGI 2/_scratch/ext/dafr/WRITEUP.md>) §2).
- [M] `--enable-mixed-chunk` is part of the trigger for the DFlash "accept rate 0.00, token 0 forever" batch poisoning, together with grammar requests (v0.5.19/0.5.20, TP4 H20, ~30 % per round on a used instance) ([sglang #43514](https://github.com/sgl-project/sglang/issues/43514), open).

**Lever: sampling / accept on GPU vs CPU**
- [S] `sampling_backend=flashinfer` [L]. The spec verify path calls `flashinfer.sampling.top_k_renorm_probs` (`srt/speculative/eagle_utils.py:931`, per the issue).
- [M] On pip-only CUDA bundles on sm_89, that op JIT-compiles at the first non-greedy request. With an nvcc/cicc/ptxas version mix it crashes the server ("Ninja build failed"); greedy works. The fix is to pin the nvcc + crt + nvvm trio to torch's CUDA line (`13.0.*`) ([sglang #43412](https://github.com/sgl-project/sglang/issues/43412)). This matches our "nvcc 13.0.88 pins" memory note. It is a correctness and availability risk, not a speed lever.

### Inferences
**Ranked lever list for after the profile** (expected gain is my estimate; evidence strength in brackets)

1. **Real 12/12 running: backport #44401 (Python scheduler patch) and extend graph sizes to `--cuda-graph-bs-decode 1 2 … 13`** [strong: own logs + merged PR + source].
   - Expected +3–8 % whole-run tok/s: 11→12 running is about +8 %, and padding waste removal applies at 9/11.
   - Costs a little graph memory: 3 more verify graphs; the current 10 graphs take about 0.88 GB per GPU [L].
   - Zero model risk. This is the most "library code fix"-shaped item found.
2. **Collectives (if P1b shows them big): transport and payload** [medium; hypothesis until P1b/P4].
   - (a) If topology shows P2P possible but NCCL chose SHM (or the reverse), set `NCCL_P2P_LEVEL`. This is an env change.
   - (b) Measure LL vs Simple at about 1 MB inside a CUDA graph.
   - (c) The logits all-gather (47.7 MB per step) is a new suspect. A vocab-parallel argmax/top-k before the gather would cut it by >100×, but needs code; SGLang has an all-to-all variant only for DP attention (`logits_processor.py:1075-1100`).
   - The gain range is wide: 0 % if P2P is healthy and collectives are under 10 ms; +20–40 % if they are 40–55 ms of the step.
3. **GDN verify kernel config sweep (BV ∈ {16,32,64}, num_warps ∈ {1,2,4}, num_stages ∈ {2,3,4}) on sm_89** [weak-medium; source shows untuned defaults].
   - Its traffic floor is about 6–7 ms. If the profile shows GDN verify at ≥ 15 ms, a tuned config could save 5–10 ms (4–8 %).
   - It is a local monkeypatch of `_select_recurrent_launch_config`, and must be validated bit-exact against the default (no numeric change expected from tiling, but check).
4. **Attention backend A/B (flashinfer vs triton) at 40k context** [weak; context-dependent evidence on 3090 only].
   - The floor is about 16 ms at 50k. A gain only exists if the current kernel is far from bandwidth.
   - It costs one relaunch; add it to Tier C only if P4 shows attention > 25 ms at W40k.
5. **Chunk size 4096 → 2048** [medium evidence from the cookbook; workload-dependent].
   - It protects decode latency from long re-prefills; it does not raise steady-state decode.
   - Effective-rate levers (cache retention) are covered by the other agent.
6. **Overlap scheduler** [already on]. Gain ≈ 0. Keep it on: `extra_buffer_lazy` requires it.
7. **`stream_interval` 1 → 4–8, more tokenizer/detokenizer workers** [weak].
   - Only if fwd occupancy is < 90 %. The Duck harness must tolerate chunkier streams.
8. **torch.compile** [weak / negative].
   - Graphs already remove launch overhead; Marlin/GDN/FlashInfer are custom ops; compile adds cold-start time; sm_89 hybrid graph variants are fragile (#36048).
   - Not worth Kaggle time before items 1–4.
9. **`--enable-mixed-chunk`** [negative]. Risky with DFlash (#43514). Do not use.
10. **`num_continuous_decode_steps`** [none]. A no-op in 0.5.21.

**Marlin at M = 12 × 8 = 96 on sm_89** [I from S and C]
- **Atomic add is irrelevant here.** SGLang's Marlin disables the atomic-add reduce for bf16 on sm < 9 (`srt/layers/quantization/marlin_utils.py:438-458`), and only uses atomic add when n < 2048 and k ≥ 2048. For this model at TP4 every quantized projection has per-GPU n ≥ 3584:
  - qkv+gate n = 3584, GDN qkvz n = 4096, gate_up n = 8704, o/out/down n = 5120.
  - So the bf16-vs-fp16 activation dtype does not change the Marlin reduce path for us. This is a negative result: do not spend a run on `--dtype float16`.
- **The ridge** for W4A16 on L4 is M ≈ 100 (121 TFLOPS FP16 dense ÷ 300 GB/s ÷ 4 FLOP/byte per row). At M=96, Marlin is at the knee. The Marlin paper shows speedups shrinking with M and with TP sharding: on A100×8, BS64 1.04×, BS128 1.07× ([MARLIN arXiv 2408.11743, Table 2](https://arxiv.org/html/2408.11743v1)).
- **Real efficiency on Ampere** has been reported at 30–40 % of DRAM bandwidth (an A800 user, batch unspecified) ([vLLM forum](https://discuss.vllm.ai/t/why-i-feel-cuda-kernel-marlin-run-not-fast/2197)).
- **Profile check:** if the profile shows Marlin at ≥ 35 ms per step at W40k (vs a 16 ms floor), weight GEMM is a top-2 cost and the precision question (Q4) becomes live.
- **Machete is not an option on L4:** it targets Hopper ([MIT Scale-ML talk abstract](https://csail.mit.edu/event/scale-ml-lucas-wilkinson-machete-cutting-edge-mixed-input-gemm-gpu-kernel-targeting-nvidia); [vLLM forum, mgoin](https://discuss.vllm.ai/t/why-i-feel-cuda-kernel-marlin-run-not-fast/2197)). Neither SGLang nor vLLM offers an alternative W4A16 kernel tuned for Ada in what I found.

**GPU clocks** [I]
- The L4 is a 72 W part ([StorageReview L4 review](https://www.storagereview.com/cs/review/nvidia-l4-gpu-review-low-power-inferencing-wizard)). At M≈96 the GEMMs are at the compute knee, so power-capped clocks could matter. Bandwidth-bound decode is affected less (5060 Ti: 180 → 150 W cost < 1 % at bs 1) ([InventiveHQ](https://inventivehq.com/blog/gpu-power-limit-tokens-per-watt)).
- Without root the cap cannot be changed. `nvidia-smi dmon` in P1a records clocks and power so that a throttling step-time inflation is visible.

### Gaps
- No public measurement of SGLang decode at 12 streams on any Ada GPU for Qwen3.5/3.6/3.8 hybrids with spec decoding. The only numbers are ours.
- No measured NCCL small-message latency on Kaggle L4 (P1b fills this).
- No source quantifies the TP logits all-gather cost on PCIe. Whether the DFlash drafter also gathers full-vocab logits per draft position was not traced in `dflash_worker_v2.py`.
- No L4 numbers for flashinfer vs triton attention with fp8 KV at q_len 8.
- No sm_89 tuning data for the FLA/SGLang GDN recurrent kernel.

---

## Q4. Precision options: FP8 W8A8 on Ada FP8 tensor cores vs INT4 Marlin at M ≈ 96

### Takeaway
SGLang 0.5.21 can run per-channel FP8 W8A8 through CUTLASS on sm_89 (CUDA ≥ 12.4). Block-FP8 checkpoints (the official Qwen FP8) fall back to Triton on Ada.

At M≈96, FP8 W8A8 is still memory-bound on L4 (ridge M≈400). It moves 2× the weight bytes of INT4: about 32 ms vs a 16 ms floor per step. It only beats INT4 if Marlin is running below about 50 % of its bandwidth floor at M=96, and it also doubles weight memory, which costs about 40 % of the KV pool.

The profile decides. Default to staying INT4.

### Cited Findings
- [S] `cutlass_fp8_supported()` returns True for sm 8.9 when CUDA ≥ 12.4 (`srt/layers/quantization/fp8_utils.py:298-307`). Channelwise CUTLASS is used when `weight_scale.numel() == weight.shape[1]` and N, K are multiples of 16, with tuned-Triton override tables where they beat CUTLASS (`fp8_utils.py:2106-2120,2195-2225`; kill switch `SGLANG_ENABLE_FP8_GEMM_CONFIG_TUNE=0`).
- [S] Block-FP8 auto dispatch: DeepGEMM if enabled, FlashInfer TRTLLM on Blackwell, CUTLASS only on SM120, else `triton_w8a8_block_fp8_linear` (`fp8_utils.py:928-958`). `--fp8-gemm-backend=cutlass` raises "deprecated on this hardware" unless SM120 (`fp8_utils.py:897-903`).
- [C] vLLM: FP8 W8A8 needs cc ≥ 8.9 (Ada, Hopper, Blackwell); older GPUs run FP8 as W8A16 via FP8 Marlin. FP8 is "up to a 1.6x improvement in throughput" (generic) ([vLLM FP8 docs](https://docs.vllm.ai/en/latest/features/quantization/llm_compressor/fp8/)).
- [L] On our L4 under vLLM 0.19, all 5 block-FP8 GEMM shapes ran untuned default Triton configs ("Performance might be sub-optimal"), and FP8 Marlin is disabled on cc ≥ 8.9 unless forced ([E12 FP8 kernel tuning README](<D:/ARC AGI 2/experiments/Experiment 12 - FP8 kernel tuning on L4 (planned)/README.md>), via engines_and_config).
- [C] "Give Me BF16 or Give Me Death" (Kurtic et al.): W4A16 is the most cost-efficient for synchronous setups, while W8A8 dominates in asynchronous continuous batching. FP8 W8A8 is effectively lossless ([arXiv 2411.02355](https://arxiv.org/pdf/2411.02355); summary via search, PDF body not readable here).
  - The A100 roofline break-even between W4A16 and W8A8 is about batch 78 (search summary of the same line of work; the exact figure is not verified in the PDF).
- [M] Marlin end-to-end speedup vs FP16 falls to 1.04–1.23× at BS64–128 on multi-GPU A100/A6000 ([MARLIN arXiv 2408.11743](https://arxiv.org/html/2408.11743v1)). No Ada GPU was in that paper.
- [L] Memory: FP8 TP4 is about 7.15 GiB weights per GPU vs about 4.62 GB for INT4. vLLM FP8 TP4 KV was 1.35M tokens vs SGLang INT4 + DFlash2 at 881k (different engines and slot settings) ([engines_and_config §3](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/Research - Qwen 27B fastest serving setup/engines_and_config.md>)).
- [L] Accuracy: RedHat INT4 recovers 98.5–98.7 % on AIME25/GPQA-D; FP8 is "nearly identical" to BF16 ([weight_quantization.md Q3](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/Research - Qwen 27B fastest serving setup/weight_quantization.md>)).

### Inferences
- **Per-step weight-time floors at M=96, TP4, per GPU** [I]:

  | Format | Bound | Floor |
  |---|---|---|
  | INT4 W4A16 | memory (4.9 GB) | 16 ms |
  | INT4 W4A16 | compute (2 × 96 × 27e9/4 FLOP ÷ 121 TFLOPS) | about 10.7 ms |
  | FP8 W8A8 | memory (about 7.7 GB) | 26 ms |
  | FP8 W8A8 | compute at 242 TFLOPS | about 5 ms |

  - INT4 is at about 0.7× of its ridge (near the knee); FP8 is clearly memory-bound.
  - **Break-even:** FP8 wins only if Marlin's measured time at M=96 exceeds about 26 ms + FP8 overheads (per-token activation quant). That is Marlin efficiency below about 55–60 % of its floor.
- **The profile decides:** sum the Marlin kernels per step on TP0 at W40k.
  - If Marlin is ≤ 25 ms, stay INT4.
  - If Marlin is ≥ 35 ms, a per-channel FP8 W8A8 checkpoint (`compressed-tensors` FP8 dynamic, not block-FP8) through SGLang's sm_89 CUTLASS path is worth a microbenchmark (GEMM only, the 5 shapes at M = 96 and 128).
- **The KV cost decides against FP8 for 12 × 60–76k streams** unless 2×TP2 is ruled out too: +2.5 GB weights per GPU removes about 300k KV tokens of the 881k.
- **Hybrid W4A8** (INT4 weights, FP8 activations) would combine both, but I found no Ada kernel for it in SGLang/vLLM. The Cohere/LiquidGEMM W4A8 numbers are Hopper ([Cohere blog via search](https://cohere.com/blog/vllm-integration-and-quality-recovery-techniques-explained); [LiquidGEMM arXiv 2509.01229](https://arxiv.org/pdf/2509.01229)).

### Gaps
- No public Ada batch sweep of FP8 W8A8 vs W4A16 Marlin on a dense 27–32B model.
- The "Give Me BF16" PDF body could not be parsed, so its exact GPU list and Ada numbers are unverified here.
- No per-channel FP8 checkpoint of Qwen3.8-27B was checked for SGLang compressed-tensors compatibility with the GDN fused projections (`in_proj_qkvz`, `in_proj_ba` fusion).

---

## Q5. What the "library code" / "tiny kernel issue" fixes might be: known SGLang/vLLM performance and correctness bugs relevant to sm_89 / L4 / Qwen3.5–3.8 hybrids

### Takeaway
The two hints have no public details:
- Darren: a fix "present in the library code itself, which works no matter what GPU you use".
- mikelou1: "a tiny kernel issue … made my program score only 40 % (instead of the 60 % projected by CV)".

The strongest matching public candidates:
1. **(GPU-agnostic, speed)** the chunked-prefill admission bug that caps running at `max − 1` (#44401, merged after our 0.5.21; matches our 11-of-12 logs).
2. **(score-affecting, kernel-level)** silent correctness failures:
   - a stale CUDA driver compute cache corrupting W4A16 output (#42917)
   - the DFlash batch poisoning (#43514)
   - DFlash greedy divergence (#38009)
   - drafter quant inheritance (#39087)
3. **(speed)** memory-budget and kernel patches used by ARC-AGI-3 winners: speculative-state budget correction, low-M GEMM, prefix-cache retention, prefetch.

None can be confirmed as what those teams did.

### Cited Findings
**The hints themselves**
- [C] Darren Amadeus Martin (rank 12, 2026-10-12): "our optimization also work on other GPUs as it fixes some issue that is present in the library code itself, which works no matter what GPU you use" ([d_748419.md](<D:/ARC AGI 2/_scratch/webresearch/d_748419.md>) line 18; [topteam_comments.md](<D:/ARC AGI 2/_scratch/webresearch/topteam_comments.md>) line 366).
- [C] mikelou1 (2026-10-16): "Not a NVARC ensemble … However there is a tiny kernel issue I found that made my program score only 40% (instead of the 60% projected by CV)". On 2026-10-18 it scored 66 % (~60 % CV) ([topic 748082](<D:/ARC AGI 2/resources/Analysis and Journals/Kaggle discussions (2026-10-22)/00_SUMMARY.md>); [topteam_comments.md](<D:/ARC AGI 2/_scratch/webresearch/topteam_comments.md>) line 354).
- [C] Also reported: "Domdolus Tolus patched library code for speed" ([Weakness analysis 2026-10-19](<D:/ARC AGI 2/resources/Analysis and Journals/Weakness analysis vs top teams (2026-10-19).md>) line 127).
- [C] Darren asked others for an aggregate above 500 tok/s on 4×L4, citing "lower bandwidth and slow gpu communication" ([topic 745787](<D:/ARC AGI 2/resources/Analysis and Journals/Kaggle discussions (2026-10-22)/00_SUMMARY.md>)).
- [C] TheDivine Aurora: about 19–20 % at 300–500 tok/s; estimates 40 % at 600–800 tok/s ([topic 750546](<D:/ARC AGI 2/resources/Analysis and Journals/Kaggle discussions (2026-10-22)/00_SUMMARY.md>)).
- [C] NguyenThanhNhan's SGLang flags (27B FP8 + EAGLE 3/1/4): `--attention-backend flashinfer --linear-attn-backend triton --chunked-prefill-size 2048 --max-prefill-tokens 4096 --mamba-radix-cache-strategy extra_buffer --mamba-full-memory-ratio 0.5 --mamba-ssm-dtype float32` ([topic 745787 dump](<D:/ARC AGI 2/resources/Analysis and Journals/Kaggle discussions (2026-10-22)/topic_745787.md>)).

**GPU-agnostic scheduler bug**
- [C] [sglang PR #44401](https://github.com/sgl-project/sglang/pull/44401) (merged 2026-10-15) says a request holding a `req_to_token` row was counted twice at admission. "Under chunked prefill this capped the running batch at `max_running_requests - 1` … the waiting queue starved". It fixes #38118 and #35537.
- [L] Our E11-05 (11 of 12 for 2,358 lines) and E11-06 (158 lines at 11 with 1 queued) show exactly this signature ([engines_and_config §4](<D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/Research - Qwen 27B fastest serving setup/engines_and_config.md>); E11-06 log).

**Silent correctness bugs**
- [M] [sglang #42917](https://github.com/sgl-project/sglang/issues/42917): Qwen3.8-27B compressed-tensors W4A16 (int4, symmetric, g128, the same scheme as RedHat INT4) gave PPL 9.98 on SGLang 0.5.21 vs 6.05 on vLLM.
  - The root cause was a stale CUDA **driver** compute cache under `SGLANG_CACHE_DIR/nv` (`CUDA_CACHE_PATH`). With an empty `~/.cache/sglang/nv`: PPL 6.2784, top-1 93.6 %, bit-identical across boots.
  - A "kernel issue" that silently cuts accuracy by about a third of top-1 agreement is the right magnitude for "40 % instead of 60 %" [I].
- [M] [sglang #43514](https://github.com/sgl-project/sglang/issues/43514) (open): DFlash + Qwen3.8-27B. Grammar-constrained requests plus `--enable-mixed-chunk` and `extra_buffer` caused whole batches to emit token 0 with accept 0.00. "No error, no warning, no crash". It was seen for 5 days on one of 3 production instances.
- [C] [sglang #38009](https://github.com/sgl-project/sglang/issues/38009): DFlash2 + thinking diverges from target-only at token 30 (4×3090 TP4). [sglang #39087](https://github.com/sgl-project/sglang/issues/39087): a drafter that inherits target quant gives about 0 % acceptance (via engines_and_config §5).
  - [L] Our server args show `speculative_draft_model_quantization=None`, yet accept len was 4.1–4.6, so inheritance did not bite us. The `unquant` flag is still worth setting explicitly [I].
- [M] [sglang #43412](https://github.com/sgl-project/sglang/issues/43412): on pip-only CUDA on sm_89 (sglang 0.5.21, flashinfer 0.6.18), the spec-verify sampling op `top_k_renorm_probs` JIT-fails on the first non-greedy request when nvcc/cicc/ptxas versions mix, and SIGKILLs the server.
  - The fix is to pin `nvidia-cuda-nvcc`, `-crt` and `nvvm` to 13.0.*. The failed build "poisons" the flashinfer cache for every restart.

**sm_89 kernel gaps**
- [M] [sglang #36048](https://github.com/sgl-project/sglang/issues/36048): single Ada sm_89, Qwen3.8-27B AWQ, v0.5.18.
  - The decode CUDA graph hung, and `awq_marlin` JIT of `gptq_marlin_fp16` "repeatedly remains in NVCC for several minutes".
  - Plain AWQ eager ran about 6 tok/s vs 64–66 on llama.cpp DFlash2.
  - Related issues: #35851 (breakable-graph per-layer overhead on hybrid GDN), #35437 (DFLASH + prefill graph capture failures).
- [C] [sglang #38980](https://github.com/sgl-project/sglang/issues/38980): `is_fa3_supported()` accepts sm_89 but no sm_89 cubin ships.

**ARC-AGI-3 winner patches** (local)
- [C] Franzen's SGLang build ([dafr serving/README.md](<D:/ARC AGI 2/_scratch/ext/dafr/serving/README.md>); [WRITEUP §1](<D:/ARC AGI 2/_scratch/ext/dafr/WRITEUP.md>)) applied five patches:
  1. A low-M BF16 GEMM.
  2. A speculative-state memory-budget correction ("avoiding reservations for intermediate SSM buffers that are not allocated").
  3. A Marlin scale-dtype fix for BF16/FP16 mismatch on GPTQ **MoE** weights.
  4. Sparse Mamba prefill checkpoints with LRU refresh.
  5. A one-shard-ahead parallel checkpoint prefetch.
- [M] The low-M patch text: "cuBLAS under CUDA-graph capture serves the qwen3.5/qwen4 decode projections (m <= 8 …) with 1-warp split-K WMMA kernels that reach only ~20-70% of DRAM bandwidth on sm120 (measured 61.9 us for m=4, n=4120, k=2560 = 320 GB/s on an RTX PRO 6000)", while the lm_head cuBLAS streams "~94% of DRAM bandwidth" (embedded in [build_bundle_pennyroyal.sh](<D:/ARC AGI 2/_scratch/ext/dafr/serving/build_bundle_pennyroyal.sh>) lines 92-102). It is gated to capability (12,0) only.
- [C] Startup on GCSFUSE: Franzen resharded the model and prefetched "in the loader's order with parallel readers"; others read 100–180 GB and got 0.00 ([WRITEUP §1](<D:/ARC AGI 2/_scratch/ext/dafr/WRITEUP.md>); [topic 751344](<D:/ARC AGI 2/resources/Analysis and Journals/Kaggle discussions (2026-10-22)/topic_751344.md>)).

### Inferences
- **Most plausible "library code" speed fix for any GPU** [I, medium]: the admission double-count (#44401).
  - It is GPU-agnostic ("works no matter what GPU you use") and costs about 8 % at 12 streams.
  - It worsens with agent churn: every chunked prefill can leave one slot idle until a request finishes.
  - Second is a speculative-state memory-budget correction like mratsim's 0002: more KV / mamba slots on the same GPU.
  - Both are Python-side patches that can be applied to our pinned wheel offline.
- **Most plausible "tiny kernel issue" costing about 20 points** [I, medium]: a silent correctness fault rather than slowness. Speed losses of 20 points would need a ~2× slowdown.
  - Candidates on our exact stack: the stale compute cache (#42917; relevant if we ever ship JIT/CUDA caches to save cold-start time), DFlash divergence (#38009/#36548), and #43514 batch poisoning (needs grammar + mixed chunk, which we do not use).
  - This makes P5/P6 (greedy token-ID gate) and a per-window accept-len alarm worth more than any single speed lever.
- **For our stack specifically** [I]:
  - (a) Backport #44401.
  - (b) Add 9/11/13 to the captured graph sizes.
  - (c) Set `--speculative-draft-model-quantization unquant` explicitly.
  - (d) Keep a fresh `SGLANG_CACHE_DIR`/`CUDA_CACHE_PATH` per session.
  - (e) Keep the nvcc trio pinned (#43412).
  - (f) After the profile, tune the GDN recurrent launch config on sm_89 and check whether the 47.7 MB logits all-gather is a real per-step cost.
- **The low-M BF16 GEMM issue** is documented for m ≤ 8 on sm120. Our BF16 GEMMs run at M = 96 (verify) and 84–96 (drafter), where cuBLAS is usually efficient. It is a low prior for us, but the profile's BF16-GEMM category will show it.

### Gaps
- Neither Darren nor mikelou1 disclosed the fix. All mappings above are inference.
- I could not confirm whether #44401 applies cleanly to the 0.5.21 scheduler without other changes (the PR mentions `get_num_allocatable_reqs(num_held_rows)` and three gates).
- No sm_89-specific SGLang performance bug for GDN Triton kernels was found. Searches of the sglang and FLA trackers for "sm89", "Ada", "L4" and "4090" returned only #36048, #38980 and #43412 as Ada-relevant. A vLLM tracker search returned nothing.
- Whether the DFlash drafter path in 0.5.21 gathers full-vocab logits per draft position, a possible second large all-gather, was not traced.
