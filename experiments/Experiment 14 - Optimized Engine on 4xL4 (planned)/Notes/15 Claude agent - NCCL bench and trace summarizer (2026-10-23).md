# 15 Claude agent: NCCL bench and trace summarizer (2026-10-23)

This note covers two CPU-built components for E14-A. Both were built and tested on this Windows machine, which has no GPU. Nothing was pushed and nothing was run on Kaggle.

| File | Role |
|---|---|
| `src/e14_nccl_bench.py` | 4-rank NCCL bench (decision D2 in Notes/13). Uses stdlib, plus torch imported lazily and only in child processes. |
| `src/e14_trace_summary.py` | Summarizer for SGLang `/start_profile` chrome traces (decision D1). Stdlib only. |
| `src/test_e14_nccl_bench_and_trace.py` | 33 pytest tests. CPU only; they need no torch. |

---

## 1. NCCL bench: what it measures and why

**Facts from the earlier SGLang measurements:**
- Each decode step does **140 all-reduces of 983,040 B**. That is bf16, 96 rows x 5120 hidden.
- Each step also does **one logits all-gather**: 11.9 MB per rank, 47.7 MB in total.
- At TP4 on PCIe, CustomAllReduce is off, so every one of these goes through NCCL.
- A 64 MiB eager all-reduce took 20.9 ms. That is 3.21 GB/s algbw and 4.82 GB/s busbw, which points to the SHM transport.

**The number that matters is the per-call cost of the 983,040 B all-reduce inside a CUDA graph.** The decode graph replays all 140 calls back to back with no CPU launches in between, so the eager latency is the wrong number to use.

### Phases (per rank, in this order; each rank writes its result file after every phase)

| Phase | What it measures | JSON key |
|---|---|---|
| a) correctness | `all_reduce` of a bf16 tensor (491,520 elements = 983,040 B) filled with rank+1. Every element must equal 1+2+3+4 = 10. | `correct`, `correct_detail` |
| b) eager all-reduce | bf16 at 8 KiB, 80 KiB, 256 KiB, 983,040 B, 2 MiB, 64 MiB. Warmup 5, then 20 timed iterations. Each iteration is aligned first (a tiny all-reduce + `cuda.synchronize`), then timed with CUDA events. Reports median and p90 in µs, algbw and busbw. Also a back-to-back loop of 20 calls (`loop_us_per_call`). | `eager.<size>` |
| c) **CUDA-graph replay** | 140 consecutive in-place `all_reduce` calls on a 983,040 B bf16 tensor, captured in one `torch.cuda.CUDAGraph`. 3 warmup replays, then 10 timed replays. **per call = replay time / 140, median over replays.** Also measured: the same 140 calls launched eagerly back to back (`eager_loop_per_call_us`), and a 1-call graph correctness check (input refilled with rank+1 before replay, so the result must be 10). | `graph_983040` |
| d) all-gather | `all_gather_into_tensor` with 11,919,360 B per rank (96 x 62,080 vocab rows x 2 B; output = 4 x that). Eager: warmup 5 + 20 timed. Then the same inside a 1-call CUDA graph, replayed 20 times. If capture fails, `graph.status = capture_failed` and the error text is kept. | `allgather_11_9MB.eager / .graph` |
| e) aggregation | **Headline = max over ranks** of each rank's median (and p90). The slowest rank defines the collective. Also reported: `min_rank_*`, `per_iter_max_median_*` (median over iterations of the slowest rank in that iteration) and `per_rank` stats. | every block |

- **Bandwidth convention** (nccl-tests, decimal GB/s):
  - all-reduce: algbw = bytes / time; busbw = algbw x 2(n-1)/n (x 1.5 at n = 4);
  - all-gather: algbw = TOTAL output bytes / time; busbw = algbw x (n-1)/n.
- **Self-check:** the measured 64 MiB at 20.9 ms gives 3.2110 / 4.8165 GB/s with these formulas, which matches the earlier measurement. This is pinned in `test_bandwidth_math`.
- **Inside the 140-call graph the buffer is all zeros.** The repeated in-place sums therefore stay finite. Timing does not depend on the values.

### Arms

Every arm sets `NCCL_DEBUG=INFO` and `NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV`.
- The subsystem can be overridden with `--nccl-debug-subsys`.
- **Never add `COLL` to it.** COLL logs every single collective call, which would distort the timings.

| Arm | Env | Runs when |
|---|---|---|
| `baseline` | nothing extra | always |
| `p2p_sys` | `NCCL_P2P_LEVEL=SYS` | only when `torch.cuda.can_device_access_peer` is true for **all** ordered pairs. Otherwise `status: skipped`, `reason: "skipped: no P2P"`. `--force-p2p` overrides this. |
| `shm_memcpy` | `NCCL_SHM_USE_CUDA_MEMCPY=1` | always |
| any extra arm | `--arm-env 'name:K=V,K2=V2'` (repeatable), e.g. `ll:NCCL_PROTO=LL` | always |

**SGLang's own NCCL env is mirrored in every arm by default** (turn off with `--no-sglang-env`). Without this, the bench would measure a different transport from the server's (for example P2P/CUMEM instead of P2P/IPC). Verified in the 0.5.21 source:

| Variable | Value | When it is set | Source |
|---|---|---|---|
| `NCCL_CUMEM_ENABLE` | `0` | when not already set and symm-mem is off | `entrypoints/engine.py:1801-1802` |
| `NCCL_NVLS_ENABLE` | `0` | when not already set | `entrypoints/engine.py:1803-1810` |
| `NCCL_GRAPH_REGISTER` | `0` | `setdefault` | `arg_groups/parallel_hook.py:326` |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `8` | forced | `entrypoints/engine.py:1816` |

- An inherited value wins and is recorded as `"<v> (inherited)"`.
- `NCCL_GRAPH_MIXING_SUPPORT` is left at the NCCL default (1). SGLang only sets it to 0 when `dcp_size > 1` (`engine.py:1811-1815`).

**How to confirm that NCCL actually read an arm's variable:** check `arms.<arm>.nccl.env`. NCCL's ENV subsystem prints `NCCL_X set by environment to Y.` for every parameter it reads. If, for example, `NCCL_SHM_USE_CUDA_MEMCPY` does not appear there, NCCL 2.29.7 did not consume it, and that arm equals the baseline.

### Process layout (a hang can never block the notebook)

```
driver (stdlib, no CUDA)            --out result.json; never raises; rewrites the JSON after the probe and every arm
 ├─ nvidia-smi topo -m | topo -p2p r | --query-gpu=index,name,pci.bus_id,pcie.link.gen.current/max,
 │                                   pcie.link.width.current/max,clocks.sm,clocks.max.sm,power.limit,driver_version
 │                                   (falls back to the short field list if a field is rejected)
 ├─ probe child  (--child-probe)     torch/CUDA/NCCL versions (torch.cuda.nccl.version()), driver version,
 │                                   devices, can_device_access_peer matrix; own timeout (120 s)
 └─ per arm: launcher (--child-run)  own session/process group (POSIX start_new_session; Windows
       │                             CREATE_NEW_PROCESS_GROUP); hard timeout --timeout-per-arm (240 s)
       │                             -> os.killpg(SIGKILL) / taskkill /T /F = launcher + all 4 ranks die
       └─ torch.multiprocessing.spawn x4   init_process_group("nccl", tcp://127.0.0.1:<free port>, device_id=cuda:r)
                                           rank_<r>.json rewritten after every phase
```

- **Partial results survive.** After a timeout the driver still reads the rank files, so a hang in, say, the all-gather graph phase keeps the all-reduce results (`status: "timeout"` plus partial data).
- **Arm status rules:**
  - `ok`: every rank reported and the child exited with code 0;
  - `error`: nonzero exit code, a launcher error, or missing ranks;
  - `timeout`: the arm hit its time limit;
  - `skipped`: the P2P gate.
  
  An error inside one phase (for example an all-gather capture failure) leaves the arm `ok`, but lists the error under `phase_errors`.
- After a real timeout the driver waits 3 s, so the killed ranks can release GPU memory before the next arm starts.
- Unless `--world` is given, the launcher requires `torch.cuda.device_count() == 4`.

### Transport parsing

The driver parses the arm's **stdout and stderr together**, because NCCL writes its debug log to stdout by default.

| Line type | Example | Parsed into |
|---|---|---|
| Line prefix | `host:pid:tid [dev] NCCL INFO ...` | the rest of the line; prefix-less `NCCL INFO` lines are accepted too |
| Connection | `Channel 00/0 : 0[0] -> 1[1] via SHM/direct/direct`, `... via P2P/CUMEM/read`, `... via P2P/IPC/read`, `... [send] via NET/Socket/0` | one record per channel. Deduplicated on (channel, sub, src, dst, send/receive, transport). The transport string is kept whole; family = the part before the first `/`. |
| Ring order | `Channel 00/02 : 0 1 2 3` | ring order |
| Version | `NCCL version 2.29.7+cuda13.0` | version |
| Driver | `cudaDriverVersion 13000` | driver version |
| Env | `NCCL_X set by environment to Y.` | env (trailing `.` stripped) |
| Graph search | `Pattern 4, crossNic 0, nChannels 2, bw 6.0/6.0, type SYS/PIX, sameChannels 1` | graph search result |
| Channel counts | `2 coll channels, ... 2 p2p channels` | channel counts |
| P2P type | `Check P2P Type isAllDirectP2p 0 directMode 0` | P2P type |
| Connected | `Connected all rings/trees ...` | connected lines |
| Init | `Init COMPLETE` | count of these lines |
| Init timing | `Init timings ... total 0.61` | total init seconds |
| WARN | any WARN line | warnings |
| P2P notes | `P2P ... disabled / not supported`, `peer access ... not` | P2P notes |

- **Summary output:** `arms.<arm>.transport` has one row per transport string, most frequent first, with count, channels and src->dst pairs. `primary_transport` is the top row.
- **The sample lines in the tests are reconstructed from memory of the NCCL 2.2x format, not copied from a 2.29.7 run** (see section 5).

### Summary block

- `summary.best_arm` is the lowest `graph_983040.per_call_us` **among arms whose correctness passed**.
- `improvement_vs_baseline_pct` compares that arm with the baseline.
- `adopt_candidate` is set only when a non-baseline arm is **≥ 20 % faster** (the D2 rule in Notes/13).
- For each arm, `per_step_comm_ms_est` = 140 x per-call + the all-gather time (graph if available, else eager).

### CUDA-graph capture: details and NCCL pitfalls handled

| Pitfall | How the bench handles it |
|---|---|
| **Warm up before capture.** NCCL communicator init and buffer setup (cudaMalloc, IPC/SHM setup, proxy threads) cannot happen during stream capture. | The comm is initialized eagerly (`device_id=` passed to `init_process_group`, torch ≥ 2.3; older torch falls back to lazy init). Every captured collective runs 3 eager times on a side stream first, and the side stream is joined back before capture (the torch docs pattern). |
| **Capture must be on a non-default stream.** | `torch.cuda.graph(g)` captures on torch's private side stream. ProcessGroupNCCL runs the kernel on its internal NCCL stream; the event record/wait fork and join between the two streams becomes graph edges. Torch skips the watchdog enqueue during capture. |
| **Watchdog vs global capture mode.** ProcessGroupNCCL's watchdog thread polls CUDA events. Under the default `capture_error_mode="global"` that can raise "operation not permitted when stream is capturing". | The bench captures with **`thread_local`** (`--capture-mode` offers global / thread_local / relaxed). The mode used is recorded. Very old torch without the argument falls back to the default. |
| **`NCCL_GRAPH_MIXING_SUPPORT`.** The default 1 allows eager and captured collectives on the same communicator. Setting 0 is slightly faster but is only safe when a comm is used in graphs only. | The bench mixes eager and graph calls on one comm (just as SGLang mixes eager prefill with graph decode), so it must stay at the default. Do not use 0 as an arm without separate comms. |
| **`NCCL_GRAPH_REGISTER`.** User-buffer registration in graphs. | Kept at 0, like SGLang (it can deadlock its LM-head all-to-all, per the docstring of `_disable_nccl_graph_buffer_registration`, `parallel_hook.py:308-325`). |
| **In-place sums grow.** | The 140-call graph works on zeros; the correctness check uses a separate 1-call graph. |
| **Start skew between ranks.** | Each rank replays on its own. The first all-reduce in the graph absorbs the start skew (tens of µs), and the rest run in lockstep. Divided by 140, this adds under 1 µs per call. |
| **The bench calls through a different API than SGLang.** SGLang calls `ncclAllReduce` directly through PyNccl on the capture stream (`distributed/parallel_state.py:1062-1072`, `graph_capture` 626-686). The bench uses `torch.distributed`. | The NCCL kernel and transport are identical. Inside the graph, the PG path only adds event nodes (expected to be a few µs at most; not measurable here). The eager numbers include PG overhead and are therefore an upper bound. |

---

## 2. Trace summarizer

### SGLang 0.5.21 profiler output: verified in the local source

Source: `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang/srt/`, read fully.

**The request** — `POST /start_profile`:
- Defined at `entrypoints/http_server.py:1164-1172`, with the fields of `ProfileReq` (`managers/io_struct.py:2257-2283`):
  - `output_dir`, `start_step`, `num_steps`, `activities`, `profile_by_stage`, `with_stack`, `record_shapes`, `profile_id`, `merge_profiles`, `profile_prefix`, `profile_stages`, `detailed_annotations`.
- Defaults from `managers/tokenizer_control_mixin.py:397-412`:
  - `profile_id = str(time.time())` (line 412), e.g. `1761212345.1234567`;
  - `with_stack` defaults to True and `record_shapes` defaults to True (lines 403-410; env vars `SGLANG_PROFILE_WITH_STACK` / `SGLANG_PROFILE_RECORD_SHAPES`).
- **Recommendation:** send `"with_stack": false, "record_shapes": false`. This shrinks the trace a lot and loses nothing this summary uses.

**Output directory:** `output_dir` or `$SGLANG_TORCH_PROFILER_DIR` or `/tmp`.
- `managers/scheduler_components/profiler_manager.py:126-131`.
- The env default is `/tmp` (`environ.py:466`).

**The file — default path (`SGLANG_PROFILE_V2` unset):**
- Written by `_stop_profile`, `profiler_manager.py:336-360`, via `torch_profiler.export_chrome_trace(...)` at lines 357-359. One file **per TP rank**, followed by a cpu-group barrier (line 360):
  ```
  <output_dir>/[<profile_prefix>-]<profile_id>-TP-<tp_rank>[-DP-<r>][-PP-<r>][-EP-<r>][-<stage>].trace.json.gz
  ```
- The DP/PP/EP parts appear only when that size is > 1 (lines 342-348).
- `<stage>` is `-EXTEND` or `-DECODE`, and only with `profile_by_stage` (lines 334, 421, 426, 435).
- For our TP4 call the result is `<id>-TP-0..3.trace.json.gz`.

**Step counting:**

| Mode | Behaviour | Source |
|---|---|---|
| Non-stage (what `e14_measure.profile_steps` sends: `{output_dir, num_steps, activities:[CPU,GPU]}`) | `_profile` calls `_init_profile` and starts at once (lines 472-486). Target = `forward_ct + num_steps + 1` (line 153). `run_batch` increments `forward_ct` (`managers/scheduler.py:4345`) and then calls the predicate (`scheduler.py:4363`), which stops when target ≤ forward_ct (lines 444-448). **So exactly `num_steps` scheduler forwards are launched inside the window.** | `profiler_manager.py:153, 444-448, 472-486` |
| `profile_by_stage` | Separate EXTEND and DECODE files with `num_steps` each. `SGLANG_PROFILE_BY_STAGE_DECODE_MIN_BS` (`environ.py:457`) delays the decode capture until full admission. | `profiler_manager.py:414-435` |

- **Boundary effects:** with the overlap scheduler, GPU work of the batch launched just before the start, or of the last batch, can straddle the window. Expect about ±1 step at the edges.
- This is why the summary also reports `detected_steps`: the count of CPU `step[...]` spans. They come from `profile_range(build_step_span_name(...))` (`model_executor/model_runner.py:1729-1731`), whose label is `step[<MODE> bs=..]`, `step[VERIFY ..]` or `step[DRAFT ..]` (`utils/profile_utils.py:462-500`).

**V2 path (`SGLANG_PROFILE_V2=1`):**
- `utils/profile_utils.py:351-376`, with the same name pattern and the suffix `-prefill` / `-decode` (line 158).
- It only supports `profile_by_stage=true` (lines 111-113). Manual start raises `NotImplementedError` (lines 137-138).

**Merged trace:**
- With `merge_profiles=true`, rank 0 also writes `merged-<profile_id>.trace.json.gz` (`utils/profile_merger.py:20-21`; discovery at 86-98).
- **Skipped by default**, because it repeats every rank's events (`include_merged=True` reads it).

**Other files in the same directory:**
- `MEM` activity: `-TP-<r>-memory[-stage].pickle` (`profiler_manager.py:375-385`). Ignored, because it has no trace suffix.
- Graph-capture traces: `<dir>/graph_capture_profile/cuda_graph_capture-<runner>-TP-<r>.json.gz` (`utils/profile_utils.py:43-70`). Only with `SGLANG_ENABLE_CUDA_GRAPH_CAPTURE_TRACE`, and in a subdirectory, which is not read unless `recursive=True`.

### Parsing and size handling

- **Streaming parser.** The parser finds `"traceEvents"` (kineto writes `schemaVersion`, `deviceProperties` and `distributedInfo` first), then decodes one event at a time with `json.JSONDecoder.raw_decode` over 4 M-char chunks. It handles:
  - both `.json.gz` and `.json`;
  - a top-level array;
  - a tolerated trailing comma.
- **Bad files are skipped per file with a reason:**
  - a truncated gzip or truncated JSON gives `read failed: ...`;
  - a file over `max_file_mb` (default 2048 MB on disk; configurable, `--max-file-mb`) gives `skipped: ... > max_file_mb`.
- **Rank** comes from `TP-<n>` in the file name. Fallback: `"rank"` in the `distributedInfo` preamble.
- **Group** = the file name with `TP-<n>` replaced by `TP-*`. So EXTEND and DECODE, or two profile ids, form separate groups. `main_group` is the DECODE group if there is one, otherwise the largest group.
- **Scale check, done once on this PC:** a 477 MB plain trace with 1.5 M kernel events.
  - It took 21.4 s (about 22 MB/s), using one core.
  - Python peak heap was 133 MB, measured with tracemalloc in a separate run.
  - So 4 ranks of that size would take about 1.5 min in the notebook. gzip adds decompression time.

### What is counted

- **Kernel events:** `ph == "X"` with `cat` in {`kernel`, `gpu_memcpy`, `gpu_memset`}, using `dur` in µs.
- **Ignored:** CPU ops, cuda_runtime events (for example `cudaGraphLaunch`), python functions and flow events.

**Per rank:**

| Field | Meaning |
|---|---|
| `categories.<c>.ms_per_step` | category total / num_steps |
| `calls_per_step` | calls of that category per step |
| `share_of_kernel_sum` | share of the summed kernel time |
| `share_of_busy` | share of the GPU busy time |
| `kernel_sum_ms_per_step` | sum of kernel durations per step |
| `gpu_busy_ms_per_step` | **union** of kernel intervals, so overlapping streams count once (skipped above 5 M kernels per file) |
| `wall_ms_per_step` | (last kernel end − first kernel start) / steps |
| `step_spans` | per label: count, CPU ms, GPU ms (from `gpu_user_annotation`) |

**Across ranks:**
- `categories.<c>`: min / mean / max ms per step.
- `collectives.<nccl_*>`: `min_over_ranks_ms_per_step`, `max_over_ranks_ms_per_step`, `skew_ms_per_step`.
- `top_kernels` (top N, default 30) by total time across ranks, with `count_total`, `calls_per_step_per_rank`, `ms_per_step_per_rank` and `mean_us`. Names are cut to 240 characters.

**`num_steps`:**
- When given (> 0), it is used as given.
- Otherwise it is detected from the CPU `step[...]` spans: VERIFY spans if present (speculative decoding), else all spans except DRAFT.
- Otherwise totals are reported (`num_steps_source` says which case applied).

### Why the minimum over ranks is the better transfer cost for collectives

- An NCCL kernel starts on each rank as soon as that rank's preceding compute is done. It cannot finish until the **last** rank has arrived and the data has moved.
- A rank that arrives early therefore **spins inside the NCCL kernel** while it waits, and its kernel duration = waiting + transfer.
- The rank that arrives last waits for nobody, so its duration is close to the pure transfer and sync cost.
- Summed over a step:
  - **min over ranks** ≈ the real communication cost;
  - **max − min (`skew_ms_per_step`)** ≈ time lost to load imbalance or launch jitter between ranks, which shows up as "NCCL time" on the faster ranks.
- **What this means for decisions:**
  - Faster NCCL transport only removes the min part.
  - The skew part is removed by balancing the compute, not by a better transport.
  - The per-rank all-reduce sums are also comparable with the bench: `min_over_ranks / 140` vs `graph_983040.min_rank_per_call_us`.

### Category regexes (first match wins, case-insensitive)

Memcpy and memset are taken from the event `cat` first (`gpu_memcpy` / `gpu_memset`, or names starting with `Memcpy` / `Memset`). Then these rules apply in order:

| # | Category | Regex |
|---|---|---|
| 1 | nccl_allreduce | `ncclDevKernel_AllReduce\|ncclKernel_AllReduce\|AllReduce\|all_reduce\|cross_device_reduce` |
| 2 | nccl_allgather | `AllGather\|all_gather` |
| 3 | nccl_other | `nccl` |
| 4 | marlin_gemm | `marlin` |
| 5 | gdn_triton | `fused_sigmoid_gating\|fused_recurrent\|chunk_\|gated_delta\|causal_conv\|mamba\|ssm\|gdn` |
| 6 | sampling | `topk\|top_k\|softmax\|sampling\|argmax\|multinomial\|cumsum\|sort` |
| 7 | norm | `rmsnorm\|layernorm\|norm` |
| 8 | attention | `BatchDecode\|BatchPrefill\|SingleDecode\|SinglePrefill\|attention\|fmha\|flash_fwd\|flash::\|MergeState\|merge_state\|PagedKV\|paged_kv` |
| 9 | other_gemm | `gemm\|cutlass\|sm89_xmma\|sm80_xmma\|ampere_\|cublas\|splitKreduce` |
| 10 | elementwise_other | `elementwise\|vectorized\|unrolled\|act_and_mul\|silu\|gelu\|rotary\|rope\|CatArray\|copy_kernel\|fill_kernel\|index\|scatter\|gather\|reduce_kernel\|triton_` |
| 11 | attention (fallback) | `flashinfer` |
| 12 | other | anything left |

**Why the order differs from the task list:**
- **FlashInfer namespaces.** FlashInfer names its non-attention kernels `flashinfer::sampling::*`, `flashinfer::norm::*` and `flashinfer::activation::*`. A bare `flashinfer` early in the attention rule would put sampling, norm and SiLU-mul kernels into attention. So `flashinfer` is only a late fallback.
- **GDN before sampling.** FLA's `chunk_local_cumsum` (prefill) would otherwise match `cumsum`.
- **Attention before other_gemm.** FA-style kernels are often named `cutlass::device_kernel<flash::...>`.

**Other mappings to know:**
- `l2norm_fwd_kernel` (GDN q/k L2 norm) lands in `norm`.
- `cross_device_reduce_*` (custom all-reduce) lands in `nccl_allreduce`.
- **DFlash2 drafter kernels have the same names as the target kernels.** The drafter's share is visible only through the `step_spans` labels (DRAFT / VERIFY), not through a kernel category.

---

## 3. Tests: exact counts and runs

`src/test_e14_nccl_bench_and_trace.py` has **33 tests**:

| Area | Count | Tests |
|---|---|---|
| Driver / fake | 8 | module import does not load torch; full `--fake` JSON structure (topology, 3 arms, env, transport, correct, 6 eager sizes, graph, all-gather, stderr tail, summary rule); `--fake-p2p no` → `skipped: no P2P`; `--force-p2p`; a failing child (exit 3) plus an unknown arm do not raise; a probe crash plus a KeyboardInterrupt in an arm still write the JSON; `--arm-env` extra arm; SGLang env defaults / inherited / `--no-sglang-env` |
| Timeouts | 4 | sleeper killed at 1.5 s; **grandchild killed** with the tree (Windows `taskkill /T`); ok / spawn-error paths; an arm timeout keeps the partial `rank_0.json` and the parsed transport |
| NCCL log parsing | 5 | SHM (prefix lines, CRLF, deduplication, rings, pattern, channel counts, Init COMPLETE / timings); trailing `.` in env values; P2P/CUMEM/read + P2P/CUMEM/CE; P2P/IPC + NET/Socket `[send]`/`[receive]` + WARN + P2P notes; empty / garbage / prefix-less lines |
| Maths / aggregation | 5 | busbw (incl. the measured 64 MiB → 3.211 / 4.8165 GB/s); percentile; max over ranks / per-rank / per-iteration max; failures / partial ranks / capture_failed; summary 20 % rule with the correctness gate |
| Real child code, stub torch | 3 | `child_run` → `mp.spawn` → `_rank_worker`, all 6 phases with a stand-in torch API; the 4-GPU requirement and the `--world` override; `child_probe` |
| Trace summarizer | 8 | classifier on 33 realistic kernel names + 2 memcpy/memset events; exact ms/step for 13 categories x 4 ranks (2 gz + 2 plain), rank-min / max / skew for AllReduce and AllGather, top-N, busy, wall; DECODE/EXTEND groups + VERIFY/DRAFT step detection; interval union with overlap; size guard + truncated gz / JSON + missing dir; stream parser = `json.load` at chunk sizes 7 / 64 / 1 MiB, top-level array, trailing comma, no traceEvents; rank from `distributedInfo`; CLI `--out` |

**Runs**, with the given command (venv `D:/ARC AGI 2/_scratch/e14_venv`, Python 3.14.6, pytest 9.1.1, torch not installed; a new `--basetemp` under the scratchpad each time):

| Run | Suite | Result |
|---|---|---|
| 1 | before the stub-torch tests were added | 30 passed |
| 2 | final | 33 passed (30.6 s) |
| 3 | final | 33 passed (40.9 s) |
| 4 | final | 33 passed (31.8 s) |
| 5 | final | 33 passed (39.1 s) |
| 6 | final, after the note was written | 33 passed (44.8 s) |

The slowest tests are the timeout tests, about 7 s for the grandchild test.

---

## 4. Expected wall time on 4xL4 (estimate, not measured)

**Per arm: about 20-45 s, dominated by start-up:**
- launcher + 4 spawned ranks importing torch (about 5-10 s each, in parallel);
- CUDA context (about 1-2 s);
- NCCL comm init (about 1-3 s with INFO logging).

**GPU work per arm is under about 5 s:**
- eager sizes: about 1.5 s, mostly the 64 MiB size at about 21 ms x 45 calls;
- 140-call graph: 13 replays x about 40-60 ms, plus the eager loop and the correctness graph, under 1 s;
- all-gather: about 15 ms x 45 eager + 23 graph replays, about 1 s.

**Totals:**
- probe: about 10-20 s; nvidia-smi: under 2 s;
- three arms: **about 1.5-2.5 min in total**, well inside the 240 s per-arm timeout;
- a hung arm costs at most 240 s + 3 s.

---

## 5. What could not be verified on CPU (open risks)

1. **The real torch + NCCL path never ran.** Not checked on CPU:
   - torch's ProcessGroupNCCL capture with `capture_error_mode="thread_local"`;
   - `init_process_group(device_id=...)`;
   - `all_gather_into_tensor` inside a graph.
   
   The stub-torch test proves only names and call shapes. If capture fails on the Kaggle torch, the phase error is recorded and the other phases still report. If it **hangs**, the arm timeout fires and the earlier phases are kept.
2. **NCCL 2.29.7 INFO line formats.** The test samples are reconstructed from 2.2x formats, not copied from a 2.29.7 log. If the connection line format changed, `transport` comes back empty; the raw lines are still in `stdout_tail` and in `<workroot>/<arm>/stdout.log`. First-run check: `nccl.nccl_lines > 0` and `connections_n > 0`.
3. **`NCCL_SHM_USE_CUDA_MEMCPY` and `NCCL_P2P_LEVEL`.** Whether they have any effect in 2.29.7 is shown only by `nccl.env` (the "set by environment" lines) and the timings.
4. **The POSIX kill path (`os.killpg` on a new session) was not exercised here.** The Windows `taskkill /T` path was, including a grandchild.
5. **The torch-PG vs PyNccl difference** (section 1) can make the bench's eager numbers a little higher than SGLang's. In-graph numbers should match within a few µs; this is not verified.
6. **Kineto output for CUDA-graph kernels.** The summarizer assumes CUPTI reports each kernel inside a replayed graph as its own `kernel` event, which is the normal kineto behaviour. First-run check: `collectives.nccl_allreduce.calls_per_step` ≈ 140 (or 140 x the forwards in a step). If graph kernels were missing, the trace would show mostly `cudaGraphLaunch` (cuda_runtime, ignored) and very small kernel sums.
7. **Trace size with the defaults** (`with_stack` and `record_shapes` true). CPU-side events can make each file hundreds of MB. Send `with_stack: false, record_shapes: false`. The 2048 MB guard plus streaming keep the notebook safe either way.
8. **Duplicate ranks in one group** (two files with the same TP rank and profile id) would overwrite each other. This cannot happen with SGLang's naming unless the same `profile_id` is reused.

---

## 6. Side finding (not built, for the plan)

SGLang 0.5.21 contains a **FlashInfer PCIe-IPC all-reduce for hosts without NVLink**, which is exactly our fabric.

**Where it is:**
- `srt/distributed/device_communicators/pcie_ipc_ar.py`, enabled by `SGLANG_ENABLE_PCIE_IPC_ALLREDUCE` (`environ.py:1354`).
- Selected right after the custom all-reduce in `parallel_state.py:969-978`.
- Docstring claim (lines 7-25): **about 34-39x faster than NCCL at 12-384 KiB, measured on an 8x SM120 host**, and 13.6 vs 21.1 ms TPOT at TP8.

**Why it does not run as installed:**
- It needs `flashinfer.comm.PcieIpcAllReduceWorkspace` (line 225).
- The 0.5.21 wheel pins `flashinfer_python[cu13]==0.6.18` (wheel METADATA).
- The local `fi.whl` (FlashInfer 0.6.18, `_scratch/e13_validate/flashinfer_src`) has **no** `PcieIpc` symbol and no pcie module under `flashinfer/comm/`. With 0.6.18 the env var only logs "this FlashInfer build has no pcie_ipc_comm; falling back to NCCL" (lines 226-231, `except ImportError`).

**What it would take:** a newer FlashInfer build, plus working CUDA IPC peer access (the `p2p_matrix` from this bench answers the second part).
