# P5: four-rank host-staged all-reduce (Experiment 23, section 2)

This is a small CUDA torch extension (`p5_host_ar.so`) plus a standalone 4-process benchmark, built for Kaggle's 4x L4
(sm_89). The GPUs are on PCIe with no P2P between them: our P2P/IPC paths measured 0.68x of NCCL. The extension does a
**two-shot all-reduce through pinned, shared host memory**. Each GPU kernel reads and writes the mapped host buffer
directly, and synchronizes through GPU-written flags in that same buffer. A call is one kernel with no CPU work, so it
can be captured in a CUDA graph and replayed.

- The measured baseline is NCCL with its default SHM transport, inside a CUDA graph, for an all-reduce of 983,040 B
  (BF16, our decode step): **343.88 us per call**.
- The target is **<= 250 us per call**.
- In the SGLang profile (W8, 12 streams), all-reduce took about 51 ms of a 116 ms decode step.

## Files

| Path | What |
|---|---|
| `src/host_ar.cu` | Kernel (3 phases, epoch, flags, timeouts, INT8 codec) and launcher. Also the co-residency query. |
| `src/host_ar_binding.cpp` | torch ops (`torch.ops.p5_host_ar.*`): shm create/map, `cudaHostRegister`, handle table, argument checks. |
| `src/host_ar_layout.h` | Partition and layout math shared by device, host and tests (plain C++). |
| `src/host_ar_check.h` | CPU model of the kernel's work partition (`cover_check` op and the local test). |
| `src/build_host_ar.py` | Build with `torch.utils.cpp_extension.load`, `-gencode arch=compute_89,code=sm_89`. Prints `BUILD_JSON`. |
| `bench_host_ar.py` | 4-process benchmark: steps (a) to (d), one JSON, watchdog, exit codes, `--cpu-check`. |
| `tests/test_layout.cpp` | g++ test of the partition and capacity math (22,038 shapes). |
| `tests/sim_protocol.py` | Randomized interleaving model of the flag/epoch protocol, with mutation-tested detectors. |
| `tests/emulate_numerics.py` | CPU emulation of the kernel arithmetic, checked against the bench oracles (bit-exact native, INT8 bound). |
| `build_p5_kernel.py` | Writes `kernel/` (Kaggle CPU notebook and `kernel-metadata.json`). |

### Ops (library `p5_host_ar`)

| Op | What it does |
|---|---|
| `init(rank, world, name, max_bytes, device, timeout_ms) -> handle` | Collective; every rank calls it with the same `name`, a unique `/p5har_<...>`. Rank 0 creates `/dev/shm/<name>` (ftruncate + posix_fallocate) and the others open it. Every rank maps it `MAP_SHARED`, then calls `cudaHostRegister(Portable \| Mapped)` and `cudaHostGetDevicePointer`. A CPU-atomic attach count is reached, and then rank 0 unlinks the name. |
| `all_reduce(handle, inp, out, mode, blocks)` | Stream ordered on the current stream and graph capturable. `mode` 0 is native (FP32 accumulate), 1 is INT8 payload. `blocks` 0 means auto (30 at 983,040 B BF16). Dtypes are f32, bf16 and f16. Tensors must be contiguous and 16-B aligned; in place is allowed, partial overlap is rejected. The size must be <= `max_bytes`. |
| `status(handle)` | Returns `[abort, status_0..status_{W-1}, max_coresident_blocks]`. A status word is `(code << 24) \| (phase << 16) \| block`. Codes: 1 timeout, 2 desync, 3 peer abort, 4 signature mismatch. The first root cause wins: a peer abort never overwrites a recorded error, and later calls on the poisoned handle do not overwrite it. `abort` is sticky. |
| `epoch(handle)` | Completed calls on this rank. Synchronous; do not call it inside capture. |
| `destroy(handle)` | Runs `cudaDeviceSynchronize()` first, then unregisters, unmaps and frees (it releases even when the context is dead, then raises). Drop every CUDA graph that captured the handle **before** `destroy`: a replay afterwards touches unmapped host memory (illegal address). |
| Meta kernel | `all_reduce` also has a Meta (fake-tensor) kernel, a no-op that checks numel and dtype, for torch.compile / FakeTensor tracing. The schema carries the alias annotation `Tensor(a!) out`. For vLLM integration: one handle per TP group per stream, and the op is collective, so tracing must not reorder it against other collectives. |
| `info()` | Build and version string. |
| `cover_check(n, world, dtype, mode, blocks)` | CPU partition model. |

## Design

The tensor of n elements is split into W rank chunks. Each chunk starts at a multiple of 32 elements, the last one may be
shorter, and for tiny n some chunks are empty. Each chunk is split into 8-element units (16 B of BF16, 32 B of FP32).
Relative unit `u` of a chunk is processed by block `(u % (blocks * 512)) / 512`, and this is the same in every phase on
every rank. That is what makes per-block flags sufficient.

The shared file layout is: header (4 KiB), flags `[2][8][64]` u64, a reserved 4 KiB (formerly signatures), W stage
slots, and one reduced area. Each slot is `max_bytes + 4 KiB`, rounded to a page. With 4 MiB this is 21.0 MB.

**Phase 1 (scatter).** Rank r copies chunks j != r of its input into its own slot `stage[r]` (GPU-to-host writes).
Element e is stored at the same position in every slot. Then it publishes `flags[0][r][b] = (epoch << 24) | sig24`.

**Phase 2 (reduce).** Rank r waits until `flags[0][j][b] == epoch` for every j != r. It then reads chunk r from every
`stage[j]` (host-to-GPU reads), takes its own chunk straight from device memory, and sums in FP32 in rank order
`((x0 + x1) + x2) + x3` (the accumulator starts at x0, so -0 sums match torch). It writes the dtype-rounded result to
`out` and to the reduced area, then publishes `flags[1][r][b] = epoch`.

**Phase 3 (gather).** Rank r waits for `flags[1][j][b]` and copies the reduced chunks j != r into `out`. In phases 1 and
3 a thread issues all its W-1 loads before its first dependent store, so the PCIe round trips overlap.

**Epoch.** The epoch is not a kernel argument, because a graph would freeze it. Every block reads the per-handle device
counter `seq` (epoch = seq + 1). The last block to finish (an atomic arrive counter in device memory) stores
`seq = epoch` and resets the counter. All blocks have read `seq` before the last block arrives, so the update cannot
race a reader. Flags only grow, so graph replays never need a reset. A flag can never legitimately be ahead of the
reader's epoch (proof below), so `flag > epoch` is reported as `E_DESYNC`.

**Bounded waits.** Each wait polls with `ld.acquire.sys`. After the first 8 polls it backs off with `__nanosleep`
(100 ns, then 200 ns after 64 polls), because every poll is a non-posted 8-B PCIe read competing with the data reads.
Every 16 polls it checks the shared abort flag and a `%globaltimer` deadline (`timeout_ms`, default 10 s in the bench).

**Errors are loud.** On an error:
- every block that saw it fills **its own units of `out` with NaN** (the unit-to-block map is fixed, so this never
  races another block);
- it writes its status word (first root cause wins), then `fence.sc.sys`, then `abort_flag = 1`;
- every other block and rank sees the abort flag and leaves its wait, filling its units with NaN too.

The abort flag is sticky, so every later call or graph replay on the poisoned handle returns all-NaN instead of stale
data, until a collective `destroy` + `init`. The kernel always terminates. Building with `-DP5_TRAP_ON_ERROR` makes an
erroring block `__trap()` instead, which kills the CUDA context. Blocks that had already finished cleanly before
another rank failed late (for example in a phase-3 wait) keep their valid results; `status` tells that a call failed.

**Call signature, checked in every block.** Each flag word is `(epoch << 24) | sig24`, where sig24 is a 24-bit hash of
(n, dtype, mode, blocks, world). One acquire load gives the epoch and the writer's signature, so every block of every
rank detects a mismatched call (`E_SIG`) before it reads any peer data, at no extra PCIe cost. The epoch keeps 40 bits
(10^12 calls) and flags stay monotonic. Two different calls collide with probability 2^-24. A peer can never read
outside the slot, because every rank's slot is `max_bytes` and `init` checks the header against each rank.

**Co-residency.** Blocks of one rank wait on the peers' blocks of the same index. A grid that cannot be fully resident
could therefore stall until the timeout fires: `sim_protocol.py` deadlocks in 197 of 200 trials with residency below the
grid. The binding caps `blocks` at `cudaOccupancyMaxActiveBlocksPerMultiprocessor x SMs`, minimized over all
instantiations, and `auto` uses at most 48 (L4 has 58 SMs).

**INT8 arm (mode 1).**
- Scaling is symmetric absmax over each group of 32 elements (4 units, one lane quad, `__shfl_xor_sync` with a quad
  mask), with an FP32 scale per group. The payload is 1.125 B per element, against 2 B for BF16.
- Non-owner contributions are quantized once. The owner adds its own chunk unquantized, and the FP32 sum is quantized
  again for the gather.
- The owner writes `dequant(q) * scale` into its own `out` too, so all ranks hold identical bits.
- Every loop that reaches a shuffle is quad-uniform: loop bounds are rounded to 4 units, and padding units feed zeros.
- Non-finite values propagate per group. If any element of a 32-element group is NaN or Inf (input or the owner's
  FP32 sum), the group's scale is NaN, so all 32 dequantized values of that group are NaN: Inf becomes NaN and the
  whole group is poisoned. This is loud, group-granular propagation; it does not match NCCL's per-element semantics.
- Tiny groups: when 0 < amax < 1e-36 the group is quantized as if amax = 1e-36. `127 / amax` then stays finite
  (1.27e38 < FLT_MAX) and the group's absolute error is at most 1e-36/254. Without the floor, amax below about
  3.7e-37 overflowed `inv` and every element snapped to +-amax.

## Memory-ordering argument

1. **One physical buffer, coherent from every GPU.**
   - Every process maps the same `/dev/shm` file `MAP_SHARED` and pins and maps its own view with
     `cudaHostRegister(cudaHostRegisterPortable | cudaHostRegisterMapped)`. This is the exact recipe of NCCL's legacy
     SHM path: `open` + `fallocate` + `mmap(MAP_SHARED)` + `cudaHostRegister(Portable | Mapped)` +
     `cudaHostGetDevicePointer`, in `src/os/linux.cc` `ncclOsShmOpen`, lines 773-870. SGLang PR #39605 does the same
     from Python (flags = 3, "Portable | Mapped").
   - `cudaHostAlloc` + `cudaIpc*` is not an option for host memory, because IPC handles are for device allocations.
     NCCL's newer path exports cuMem host allocations as file descriptors instead (`src/transport/shm.cc` lines
     305-330); that would be a possible later variant.
   - The memory is ordinary write-back memory. It is not write-combined: `cudaHostAllocWriteCombined` is not involved,
     and WC would make the read-back paths very slow. PCIe DMA on x86 is cache-coherent, so a GPU write lands in host
     memory or the LLC, and the next GPU read snoops it. The CPU never writes the data area after `init`.
2. **Writer.** All threads issue weak stores into the slot, then `__syncthreads()`. Thread 0 then runs
   `__threadfence_system()` (`fence.sc.sys`) and `st.relaxed.sys` of the flag.
   - The barrier orders the CTA's stores before thread 0's later operations.
   - The SC fence at system scope is cumulative, so every write ordered before it is visible system-wide before the
     flag store.
   - This is the vLLM custom all-reduce pattern: `__syncthreads(); st.release.sys` in `custom_collective_common.cuh`,
     lines 162-240. Our version uses the stronger `fence.sc.sys`.
3. **Reader.** Thread j polls with `ld.acquire.sys.global.u64`, then `__syncthreads()`, then all threads read with
   `ld.volatile.global.v4` (relaxed.sys).
   - The causality order is transitive: remote fence and store, then the observed acquire, then the barrier, then the
     loads. So every data load observes the published data.
   - Volatile (morally strong, system-scope) loads are not served from a stale L1 line, and the kernel boundary
     invalidates L1 anyway.
4. **Atomicity and ordering on PCIe.**
   - Flags are aligned 8-B single-writer stores and loads, which are single-copy atomic, so a flag value is never torn.
   - No GPU atomics touch host memory. The CUDA guide states that atomics on mapped page-locked memory are not atomic
     with respect to the host or other devices. The attach counter is a CPU `__atomic` op, and the arrive counter lives
     in device memory.
   - PCIe posted-write ordering is not relied on by itself: the GPU must implement `membar.sys` / `fence.sc.sys` by
     making earlier writes globally visible before later ones, which is the same assumption NCCL's SHM transport makes.
5. **Buffer reuse across epochs (no double buffering needed).**
   - For rank A to write any slot or reduced data of epoch e+1, its kernel for epoch e must have finished (stream order).
     That required every rank's `flags[1][*][b] == e`, for every block b. Each rank writes that flag only after its
     phase-2 reads of epoch e. So nobody still reads A's stage data of epoch e when A overwrites it.
   - A's phase-2 write of the reduced area in e+1 waits for every peer's `flags[0] == e+1`. A peer writes that only
     after its whole epoch-e kernel, including its phase-3 reads, has finished.
   - The same argument shows a waiter never sees `flag > epoch`.
   - `sim_protocol.py` checks this over random interleavings: 400/400 trials pass, with W in {2, 4, 8}, up to 12 calls
     and a block count that changes per call. Its detectors were mutation-tested: dropping the phase-3 wait, or waiting
     for only one peer, is caught in 200/200 trials.
6. **In-place calls (inp == out).** Every element is read in phase 1 or 2 and written in phase 2 or 3 by the same thread,
   because the unit-to-thread map is the same in every phase. So there is no cross-thread write-after-read hazard.

## Bandwidth estimate (983,040 B, W = 4)

Bytes per GPU per call:

| Phase | Writes (GPU to host) | Reads (host to GPU) |
|---|---|---|
| 1 | 3S/4 = 737,280 B | none |
| 2 | S/4 = 245,760 B | 3S/4 = 737,280 B |
| 3 | none | 3S/4 = 737,280 B |
| Total | S = 983,040 B | 1.5 S = 1,474,560 B, i.e. 2(N-1)/N S |

Writes total S, not 1.5 S, because host memory works as a broadcast medium: one reduced copy is read by 3 peers.

**Link.** The L4 has PCIe Gen4 x16: 16 GT/s x 16 lanes x 128/130, about 31.5 GB/s raw per direction, or about
25 GB/s of practical DMA. Loads issued by SMs to host memory need many requests in flight. At 2 us latency x 25 GB/s
that is about 50 KB per GPU. The grid of 30 x 512 threads has about 737 KB in flight in phase 2 (3 x 16 B per thread).

**Phases at 25 GB/s with independent links.** The phases are serialized by the flags. Phase 1 takes 29.5 us, phase 2
29.5 us (its 9.8 us of writes overlap on the other direction), and phase 3 29.5 us, for **88.5 us** of transfer. Fixed
costs add about 10 to 20 us:
- graph kernel launch, about 3 to 5 us;
- 2 flag handoffs: `fence.sc.sys`, the flag store and a remote poll round trip, about 3 to 5 us each;
- the abort and `seq` reads at kernel start, about 1.5 us;
- the first-byte read latency of phases 2 and 3.

That gives an estimate of **about 100 to 115 us**.

| Effective per-GPU bandwidth, all 4 concurrent | Per-call estimate |
|---|---|
| 25 GB/s (independent x16 links) | about 100 to 115 us |
| 18 GB/s (SM-issued sysmem reads below CE speed) | about 140 us |
| 12.5 GB/s (GPU pairs share an x16 uplink behind a switch) | about 190 us |
| about 6 GB/s (all 4 behind one x16 uplink) | about 370 us, misses the target |

**Plausibility.** With about 15 us of fixed cost, 250 us needs at least 2,211,840 B / 235 us = **9.4 GB/s** effective
per GPU with all four concurrent. The same model applied to NCCL's 343.88 us gives 6.7 GB/s. NCCL's SHM ring takes 6
hop steps with per-step handshakes and few CTAs, so it is latency-bound rather than link-bound. **So the target is
plausible unless all four GPUs share a single x16 uplink.** The bench's `pcie_probe` (solo vs concurrent H2D/D2H) shows
which row applies. Host memory traffic is 3.9 MB written and 5.9 MB read per call, about 98 GB/s at 100 us. That is
within one socket's DRAM/LLC bandwidth, but NUMA placement matters (see risks).

**INT8 arm.** The payload is 0.5625x, so about 50 us of transfer plus fixed costs, roughly 65 to 80 us at 25 GB/s.

**Next lever if the measured number is phase-bound.** Pipeline the phases with sub-block flags, so phase 1 writes
overlap phase 2 reads (opposite directions) and phase 2 writes overlap phase 3 reads. The floor is then
max(S, 1.5 S) / BW = 59 us at 25 GB/s.

## Benchmark (`bench_host_ar.py`), in the user's test order

The parent never initializes CUDA. Each worker group is 4 subprocesses (one per GPU, NCCL process group over tcp
localhost) plus a ctypes NCCL communicator on the same `libnccl`, so the NCCL baseline is graph-capturable the same way
vLLM's pynccl is. NCCL runs with `NCCL_DEBUG=INFO` to a file, and the transport (`via SHM` / `via P2P`) is parsed into
the JSON.

| Step | Group | What it runs |
|---|---|---|
| (a) | `main_ab` | Numerical oracle. FP32 and BF16 at 64 KB, 245,760, 491,520, 983,040, 1,966,080 and 3,932,160 B, with patterns randn / wide (1e-6..1e4) / cancel / large (near dtype max) / spikes / tiny (denormal) / ramp / nan_inf. Also 12 odd sizes (2 B .. 983,136 B: partial units and groups, empty chunks), in-place, and explicit grids 1 / 7 / max. Strict checks: the result is bit-identical to the ordered FP32-accumulate reference (NaN-aware) and identical on all ranks, with no kernel error. It reports max abs/rel error vs NCCL and vs the exact FP64 sum (with NCCL's own error vs FP64 for context). |
| (b) | `main_ab` | `pcie_probe`; then latency per call (median per rank, max over ranks) for NCCL ctypes eager and graph, torch-NCCL eager, host eager and host graph at the 6 sizes; then a block sweep at 983,040 B. |
| (b) | `nccl_ctas{4,8,16,32}` | NCCL eager and graph again in separate launches with `NCCL_MIN_CTAS = NCCL_MAX_CTAS = x`. NCCL reads these at communicator init, hence the separate processes. |
| (c) | `main_cd` | Graphs of 1 call and of 50 back-to-back calls (distinct buffers). Each is replayed 1000x with new input data copied in before every replay (scale and roll per replay), and every replay is verified bit-exactly against the locally recomputed reference. It also checks an eager call between replays, epoch continuity on all ranks (capture, then replays, then timing), and status every 100 replays, and records the per-call time inside the graph. Last, the error path: a deliberately mismatched call (rank 0 grid 4, the others grid 8) must give all-NaN `out` on every rank with `E_SIG` recorded, the next correct call on the poisoned handle must be all-NaN too with the root cause kept, and after a collective re-init a call must be bit-exact again (check `c_error_path_loud`). |
| (d) | `main_cd` | INT8 at BF16 and FP32, the 6 sizes plus odd sizes, patterns randn / wide / cancel / spikes / ramp / outlier / tiny (1e-39) / tiny37 (amax about 1e-37, below the floor) / nan_inf. It reports error vs FP64 next to the native error. Semantics checked by `int8_eval`: every 32-element group with a non-finite input on any rank must be all-NaN, and every other element must stay within the analytic bound: the sum over non-owners of `max(amax_g, 1e-36)/254`, plus `max(amax(acc)_g, 1e-36)/254`, plus output rounding (including the dtype's subnormal step). Outputs must be identical on all ranks. It also measures INT8 vs native graph latency. |

**Output.** One JSON with `meta`, `env` (nvidia-smi -L / topo -m / PCIe link, numactl, lscpu, /dev/shm), the `groups`
(status, rcs, rank-0 results, other ranks' checks, NCCL log summary, log tails on failure) and a `summary`. The summary
holds every check, the decode-size numbers, the best NCCL, `target_250us_met_native`, `target_250us_met_int8` and the
speedup.

**Never hangs.**
- Kernel waits time out, and the abort flag fans the error out to every block and rank.
- After every case, `status_ok` runs as a collective; on an error all ranks destroy the handle and re-init with a new
  name.
- Each group has a wall-clock budget: 900 s for the main groups, 240 s for the NCCL-only ones.
- A crashed rank gets its group killed after 60 s. Process groups are killed with `SIGKILL`.
- There is a global `--timeout-s` (default 3000, which covers the group budgets 900 + 4 x 240 + 900 = 2760 s), and
  leftover `/dev/shm/p5har_*` files are removed.
- A rank whose step raises writes `fatal` into its part file and leaves with `os._exit` (no collective teardown, which
  would block). The parent kills the group 60 s after the first non-zero exit or `fatal` part file.
- In step (c), if a kernel error forces a re-init during the replay loop, the old graph is dropped and never replayed
  (its handle is gone); timing and the epoch checks for that graph are skipped and recorded as `handle_lost_at_replay`.

**Exit codes.** 0 means everything ran and passed, 1 means a correctness check failed, 2 means a watchdog kill, worker
crash or step error, and 3 means a setup error. `--quick` runs a smoke version. `--cpu-check` checks a CPU-only load:
ops and schemas, the CUDA kernel registration, the partition model, and that `init` fails cleanly without a GPU.

**GPU-notebook usage** (offline, vLLM 0.32 venv; attach this kernel's output):

    /tmp/vllm-venv/bin/python bench_host_ar.py --so /kaggle/input/<p5 output>/p5_host_ar.so --out /kaggle/working/p5_bench.json --timeout-s 3000

(3000 s is also the default; it covers the group budgets of 2760 s.)

The DESIGN estimate for this section is 15 min. Group startup is about 20 to 40 s each, 6 groups.

## Kaggle CPU kernel (`kernel/`, built by `build_p5_kernel.py`)

The kernel is `foysalemonshanto/q27-e23-p5-host-ar`: CPU, internet on, private, pinned image `gcr.io/kaggle-private-byod/python@sha256:37c64f7d...`.
Its dataset is `foysalemonshanto/vllm-0-32-0-cu13-wheelhouse`. The sources are embedded as a sha256-checked zip.

1. Unpack the sources and check the sha256.
2. Install the vLLM 0.32.0 venv with the dataset's `install_wheelhouse.py`, exactly as
   `Experiment 22/.../build_cpu_kit_e22_flags.py` does. This gives torch 2.13.0+cu130 and the pip CUDA 13.0.88 toolkit
   in `site-packages/nvidia/cu13`: `nvidia-cuda-nvcc`, `-crt`, `nvidia-nvvm`, `-runtime` 13.0.96, and `ninja` 1.13.2,
   all pinned in the lock. `CUDA_HOME` comes from the installer.
3. Probe the toolchain: nvcc, g++, ninja, torch versions, key files.
4. Run `build_host_ar.py`: `cpp_extension.load`, `-O3 -gencode arch=compute_89,code=sm_89 -lineinfo -Xptxas -v`, and
   torch's own `-std=c++20`. The `.cu` includes no CCCL/bf16 headers; conversions use inline PTX `cvt.rn`. The ptxas
   register and spill lines are recorded.
5. Check the SASS with `cuobjdump`. The nvcc 13.0.88 wheel ships none (verified: its `bin/` holds nvcc, ptxas,
   fatbinary, nvlink, cudafe++, bin2c) and PyPI's `nvidia-cuda-cuobjdump` starts at 13.2.51, so the kit runs
   `pip download nvidia-cuda-cuobjdump==13.2.51` (the wheel holds `nvidia/cu13/bin/cuobjdump`, 777 kB; a newer
   cuobjdump reads 13.0 cubins), else any 13.x, else the system one. The ELF archs must be exactly `["89"]`, with 0 PTX entries, at least 18 `p5_ar_kernel`
   instantiations, and `MEMBAR*SYS` and `STRONG.SYS` instructions present. It also records an embedded EM_CUDA ELF scan
   and `ldd`.
6. Run `bench_host_ar.py --cpu-check` in the venv. This loads the .so into torch on CPU.
7. Run the host tests: `test_layout.cpp` (g++), `sim_protocol.py`, `emulate_numerics.py`, and `py_compile` under Py3.12.
8. Write the outputs to `/kaggle/working`: `p5_host_ar.so`, `bench_host_ar.py`, `p5_src/`, `p5_sass_head.txt` and
   `p5_report.json`, then print `ALL OK` or `NOT OK`.

## Local results (Windows host, 2026-10-29; no nvcc locally, so the CUDA compile is the kit's job)

- `tests/test_layout.cpp`, MinGW g++ 6.3, `-std=c++14`: **ok**. 22,038 shapes and 726,256,608 element visits, no
  capacity failure, auto grid 30 at 983,040 B BF16, shm total 21,008,384 B for a 4 MiB handle.
- `tests/sim_protocol.py`: **400/400** full-residency trials ok. Partial residency deadlocks in 197/200 trials, which is
  the reason the binding caps the grid. Both mutations are detected in 200/200 trials.
- `tests/emulate_numerics.py`, torch 2.14 CPU (rerun after the audit fixes): native bit-exact in **104/104** cases,
  INT8 passes `int8_eval` in **117/117** (now including tiny, tiny37 and nan_inf; 14,607 non-finite groups, all NaN).
  Negative control: the pre-fix codec (no floor, NaN dropped, Inf -> 0 x inf) **fails** the new checks on both tiny37
  and nan_inf, so the checks bite. The worst error-to-bound ratio is 0.995: the bound is a true worst case and is
  nearly attained at large n. INT8
  rel-L2 vs FP64 is about 0.7 % on randn (BF16 rounding alone is about 0.2 %). On `cancel` inputs INT8 rel-L2 is 46x
  the result's norm (BF16) and up to about 2,700x (FP32), because quantization error scales with the addends, not with
  the tiny sum.
- `bench_host_ar.py` compiles under py3.14 with `py_compile`. A no-GPU run exits 3 with the JSON written.

## Kaggle CPU kit v1 and the audit fixes (rev 2, 2026-10-29)

Kit v1 (before these fixes) ran **ALL OK**: nvcc 13.0.88 build in 19 s, ELF sm_89 only, 18 kernels, `MEMBAR.SC.SYS` x54,
`STRONG.SYS` x1084, 0 spills, all tests passing. An independent read-only audit found no blocker; ordering, epoch,
reuse and the index math were confirmed. Its findings, all fixed in rev 2:

| # | Finding | Fix |
|---|---|---|
| HIGH 1 | Error paths returned partly written `out`; a poisoned handle silently did nothing. | NaN fill of each erroring block's own units; sticky abort, so later calls are all-NaN; optional `-DP5_TRAP_ON_ERROR`. |
| MED 2 | The root-cause status was overwritten by peer-abort and later calls. | First root cause wins. |
| MED 3 | Step (c) replayed a graph of a destroyed handle after a re-init. | The graph is dropped and timing skipped. |
| MED 4 | `destroy` did not synchronize. | `cudaDeviceSynchronize` first; graphs must be dropped first (documented). |
| MED 5 | INT8 Inf/NaN/tiny handling: an Inf group became NaN, NaN was dropped, amax < 3.7e-37 overflowed. | NaN scale for non-finite groups, a 1e-36 floor, new tiny / tiny37 / nan_inf INT8 cases with these semantics. |
| LOW 6 | A failing rank blocked in teardown. | `os._exit` plus a `fatal` part-file watch. |
| LOW 7 | The 1500 s watchdog was below the 2760 s of group budgets. | Default 3000 s. |
| LOW 8 | Polls flooded PCIe. | `__nanosleep` backoff. |
| LOW 9 | Integration notes. | Meta kernel, alias annotation and one-handle-per-stream noted. |

Beyond the audit, the call signature moved from block 0 into every flag word, so every block detects a mismatch and
the error path is fully loud. Step (c) gained the error-path check. The kit must be rerun on rev 2.

## Self-review of the CUDA code (races and edge cases)

| Item | Result |
|---|---|
| Flag reuse across epochs | Flags are monotonic u64 values and the epoch comes from device `seq`. With a block count that changes per call, unused flags stay at older epochs, which are < e, so no wait is satisfied by mistake. `flag > e` is impossible (proof 5), so it is checked as `E_DESYNC`. Modeled with B changing per call. |
| Rank skew | A fast rank cannot overwrite stage or reduced data a slow rank still reads (proof 5). Skew only shows up as wait time, and the timeout is 10 s. |
| Graph replay | No host state per call. Kernel arguments are fixed pointers. The epoch is read at run time, and the arrive counter is reset by the last block. The bench verifies 1000 replays x 50 calls bit-exactly plus epoch arithmetic. |
| `seq` update race | The last arriving block writes `seq` after all blocks did `atomicAdd(arrive)` at their end; each had read `seq` at its start. |
| Barrier divergence | `s_err` is read only after a `__syncthreads`, so every branch around a barrier is block-uniform. Writes to `s_err` happen only inside `wait_peers`, after a barrier every thread has passed. |
| Partial chunks (size not a multiple of 4 x 16 B, or of 4 x 32 elements) | Chunk starts are multiples of 32 elements. Units with c < 8 use element-wise loads and stores that never touch memory past n (explicit `if`, no speculative tail loads). INT8 payload padding is written as zeros inside the slot, which has round_up(n, 32) bytes of payload room. Empty chunks (n < 32 W) run no iterations. `cover_check` checks this for n = 0..1200 x worlds 2/4/8 x 3 dtypes x 2 modes, plus odd bench sizes. The GPU bench adds 12 odd sizes. |
| Alignment | Data pointers must be 16-B aligned (checked); unit offsets are multiples of 16 B (BF16) or 32 B (FP32). Slots are page aligned. INT8 payload is 8-B aligned, and scales are 4-B aligned at a 256-B base. |
| INT8 shuffles | Loops are quad-uniform: bounds are multiples of 4 units, the stride is a multiple of 4, and padding units still run `quantize8`. The shuffle mask is the quad. One lane per group writes the scale: the check confirms each group's scale is staged exactly W-1 times. |
| Identical outputs on all ranks | Only the owner computes a chunk. Native mode stores the rounded value; INT8 writes the dequantized value (a plain multiply, no FMA contraction possible). The bench checks bitwise. |
| Signed zero, NaN, denormals | The accumulator starts at x0 rather than 0 + x0. No fast-math: IEEE division and denormals are kept. NaN results match the reference with a NaN-aware comparison; `cvt.rn.bf16` NaN payload bits may differ from torch's. |
| Error paths | Every wait ends (abort, `%globaltimer`, `__nanosleep` backoff). Every erroring block NaN-fills its own units of `out` (ownership checked by `cover_check` code 8). The sticky abort makes later calls all-NaN. The status word keeps the first root cause. A mismatched call fails with `E_SIG` in every block, because the signature travels in every flag word, before any peer data is read; reads stay inside slots of identical size. `destroy` synchronizes the device first. The bench never replays a graph whose handle was re-created. |
| INT8 codec edge values | Non-finite in a group gives a NaN scale, so the group is all-NaN; this includes the owner's own non-finite values and an overflowing FP32 sum. 0 < amax < 1e-36 is floored, so `inv` stays finite. Checked in the CPU emulator with a negative control, and on the GPU in step (d). |
| Co-residency | The binding enforces the grid cap from the occupancy query (`__launch_bounds__(512, 1)`), so a grid can never exceed what fits at once. |

## Research findings (with URLs)

**SGLang PR #39605**, "Host-staged all-reduce for two GPUs without peer access" (author 05yuki, open, opened 2026-09-15):
https://github.com/sgl-project/sglang/pull/39605
- Only `world_size == 2` is supported, and the path is env-gated. The author says the layout generalizes but "the sum
  does not".
- **Host-staged path (>= 1 MiB):**
  - A shared `/dev/shm` file is mapped with `mmap`, then `cudaHostRegister(..., 3)` ("Portable | Mapped") and
    `cuMemHostGetDevicePointer`.
  - The copies are copy-engine copies in 4 MB pieces, with `cuStreamWriteValue32` / `cuStreamWaitValue32` (GEQ) on
    monotonic counters and double buffering.
  - The path declines to run inside CUDA graph capture.
  - A later race fix makes the add wait for the rank's own D2H copy. Our design has no own staging copy: the own chunk
    is read from device memory.
  - The author reports a 20.97 MB all-reduce at 10,859 -> 1,494 us on 2x RTX 5070 Ti.
- **LL path (BF16 <= 64 KiB):**
  - It uses NCCL's LL format (two bf16 values plus a 32-bit sequence number per 8-B word), with `st/ld.relaxed.sys`
    and no fences.
  - The sequence number comes from a device counter that is read at run time, so the path is graph capturable. Our
    epoch counter follows the same idea, extended to many blocks with an arrive counter.
  - The author reports 6.5 us vs NCCL 41.5 us at 4096 elements.
- **What we took from it:** the same mapping recipe, the run-time device counter for replays, and the LL idea as a
  possible later small-message mode.

**vLLM custom all-reduce (P2P-based): not usable here.**
https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/distributed/device_communicators/custom_all_reduce.py
- Lines 256-292 disable it "on more than two PCIe-only GPUs" and when the P2P check fails (`_can_p2p`). It works
  through CUDA IPC peer buffers.
- Its barrier, which we reuse as the ordering pattern, is in
  https://github.com/vllm-project/vllm/blob/v0.32.0/csrc/custom_collective_common.cuh lines 162-240 (`st.release.sys`
  / `ld.acquire.sys`, `__syncthreads`).
- vLLM 0.32 also has a FlashInfer PCIe IPC all-reduce. It is CUDA-IPC, i.e. peer access, so it is not applicable:
  https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/distributed/device_communicators/flashinfer_pcie_ipc_all_reduce.py

**vLLM quick all-reduce: ROCm MI300 only.**
https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/distributed/device_communicators/quick_all_reduce.py, lines
66-93: "only supported on ROCm MI300 series". It is a two-shot design over IPC buffers with INT8 / INT6 / INT4 / INT3
codecs (`csrc/rocm/quickreduce`). Our INT8 arm borrows the idea: quantize the scatter and the gather, with group
scales.

**NCCL SHM transport internals** (master @ 3e46c089, 2026-10-08):
- Legacy path: `/dev/shm/nccl-XXXXXX` via `mkstemp`, `fallocate`, `mmap(MAP_SHARED)`,
  `cudaHostRegister(cudaHostRegisterPortable | cudaHostRegisterMapped)`, `cudaHostGetDevicePointer`. See
  https://github.com/NVIDIA/nccl/blob/master/src/os/linux.cc, lines 773-870.
- New path: a cuMem host allocation exported as a file descriptor (`ncclCuMemHostAlloc`, `cuMemExportToShareableHandle`).
  See https://github.com/NVIDIA/nccl/blob/master/src/transport/shm.cc, lines 305-330.
- `NCCL_SHM_LOCALITY` defaults to the receiver side (line 56).
- The data is moved by the NCCL kernels' loads and stores to host-mapped buffers; the proxy is used at setup. Why it is
  slow here: the ring all-reduce takes 2(N-1) = 6 dependent steps through host memory, each with a flag handshake, and
  few channels and CTAs mean few outstanding PCIe reads. That is why `NCCL_MIN_CTAS` / `NCCL_MAX_CTAS` are worth
  measuring.

**TensorRT-LLM LOWPRECISION all-reduce (PCIe).**
https://nvidia.github.io/TensorRT-LLM/legacy/advanced/lowprecision-pcie-allreduce.html
- FP8 e4m3 quantization per warp (31 threads x 16 B plus one scalar) for L20/L40 systems.
- It targets "2 NUMA domains, 4 GPUs per NUMA behind a PCIe switch" and "will not accelerate performance in different
  topologies". That implies peer access through the switch, so it is not a no-P2P design. It supports compression as a
  lever.

**Flash Communication** (arXiv 2412.04964): https://arxiv.org/html/2412.04964v2
- Two-step quantized all-reduce: quantize, all-to-all, dequant + sum, requantize, all-gather.
- Fine-grained groups of 128, asymmetric INT8 / INT4 / INT6. INT8 is near-lossless: LLaMA-3-70B average 80.00 at FP16
  vs 80.00 at INT8.
- Up to 3.18x lower latency at 64 MB or more on L40. It uses GPU peer direct access.
- Supports a symmetric INT8 arm with small groups as a reasonable first codec.

**mscclpp.** https://microsoft.github.io/mscclpp/tutorials/04-port-channel.html
- MemoryChannel copies with GPU threads through peer memory, which needs P2P/IPC.
- PortChannel hands transfers to a host `ProxyService` thread (copy engine or RDMA), so the CPU is involved per request.
- Neither gives a pure GPU-driven, graph-only path without P2P.

**CUDA runtime `cudaHostRegister` / `cudaHostGetDevicePointer`.**
https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
- Portable: pinned for all contexts. Mapped: device pointer through `cudaHostGetDevicePointer`.
- With UVA and `cudaDevAttrCanUseHostPointerForRegisteredMem` the device pointer equals the host pointer. The binding
  always uses `cudaHostGetDevicePointer`.
- Requires `cudaDevAttrHostRegisterSupported` (checked in `init`).
- No restriction on file-backed `MAP_SHARED` memory is stated, and NCCL relies on it.

**Memory model references** (standard docs, cited for the argument above, not re-fetched in this session):
- PTX memory consistency model (`fence.sc`, release/acquire, scopes, morally strong, causality order):
  https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#memory-consistency-model
- CUDA C++ guide, memory fence functions and mapped memory (atomics on mapped memory are not atomic for host and other
  devices): https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#memory-fence-functions
- L4 datasheet (PCIe Gen4 x16): https://www.nvidia.com/en-us/data-center/l4/

## Risks

1. **PCIe topology and aggregate bandwidth on the Kaggle VM are unknown.** If all 4 GPUs share one x16 uplink, about
   370 us is expected and the target is missed. The `pcie_probe` and `nvidia-smi topo -m` in the JSON decide. A vIOMMU
   in the VM and 4 KiB tmpfs pages may cut SM-issued sysmem bandwidth (IOTLB misses).
2. **NUMA.** The `/dev/shm` pages are placed by rank 0's `posix_fallocate`, so with 2 NUMA nodes half the GPUs cross
   the socket link. A later variant could place pages per writer (each rank first-touches or `mbind`s its own slot).
3. **Co-residency inside vLLM.** The kernel needs its whole grid resident on every GPU. Kernels on other streams can
   hold SMs and delay it; that is not a deadlock unless they wait on this all-reduce. A spinning rank occupies SMs while
   it waits. The default 10 s timeout must be raised for production, and an abort is fatal for the handle: it needs a
   collective re-init.
4. **One handle, one stream at a time.** Two concurrent calls on the same handle from different streams would corrupt
   the epoch protocol. Integration should use one handle per TP group per stream.
5. **Numerics differ from NCCL.** The result is BF16 RNE of the FP32 sum in rank order; NCCL BF16 rounds at every ring
   hop. Ours is more accurate and deterministic, but not bit-equal to NCCL. INT8 changes model numerics (about 0.7 %
   rel-L2, large error under cancellation, a NaN/Inf poisons its whole 32-element group), so it needs an accuracy run
   before any use.
6. **The build uses the pip CUDA 13.0 toolkit with torch's cpp_extension.**
   - A missing header or the CCCL 13.3 vs nvcc 13.0 mismatch could break the compile. The `.cu` avoids CCCL headers;
     the `.cpp` uses only `ATen`/`c10/cuda`.
   - `cuobjdump` is not in the toolkit; the kit downloads nvidia-cuda-cuobjdump 13.2.51 with internet on (a
     13.2 tool reading 13.0 cubins is expected to work, but is not yet verified).
   - The build is SASS-only sm_89, with no PTX fallback by design.
7. **GPU-side behaviour is unverified until the GPU run.** This covers memory ordering on this platform, actual
   bandwidth, and NCCL graph capture through ctypes. The bit-exact 1000-replay test and the cross-rank checks are the
   guard. NCCL's `torch.distributed` graph capture is not needed, because the ctypes communicator is captured on the
   current stream.
8. **`/dev/shm` size.** The bench needs 21 MB plus NCCL's buffers. `init` fails cleanly with ENOSPC (no SIGBUS) and the
   bench records free space.
