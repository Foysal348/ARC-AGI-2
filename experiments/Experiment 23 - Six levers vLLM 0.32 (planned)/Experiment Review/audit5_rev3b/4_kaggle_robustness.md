# E23 rev 3b: Kaggle runtime robustness audit (agent 4)

Scope: `Experiment Notebooks/Build/build_e23_notebook.py` (CELL1 to CELL6), `src/e23_lab.py`, the P2 Merlin env scripts, the P6 EAGLE install script, the P4 and P5 benches, the CPU kit builder, and the kit v1 outputs (`_scratch/e23/kitout_v1`).
This audit was read-only and edited no project file. Line numbers refer to the builder file `build_e23_notebook.py` (B:) and to `e23_lab.py` (L:).

## Findings

### F1 HIGH: teacher-forced prompt_logprobs at 40k / 90k will very likely OOM-kill the stock server mid-body
- **Where:**
  - B:474-481 (`tforced`), B:638-639 (s6 depth 0), B:722-723 (s5 spec 0);
  - L:484-504 (`teacher_forced`: `prompt_logprobs: 1` over the full 40k / 90k prompt).
- **Scenario:**
  - Stock vLLM 0.32 computes prompt logprobs for every scheduled prefill chunk, and does it all at once. With `--max-num-batched-tokens 4096`, that is 4096 × 248,320-vocab logits:
    - bf16, all-gathered to every TP rank: about 2 GB;
    - then `log_softmax(float32)`: about 4 GB more;
    - so about 6 GB of transient memory per rank.
  - The KV pool already fills `--gpu-memory-utilization 0.90`, which leaves about 2 GiB of headroom on a 22.5 GiB L4.
  - The profile run never budgets these logits: it only computes logits for max_num_reqs rows.
  - The result is a CUDA OOM in EngineCore, and the server dies. It then fails, in order:
    1. the rest of the TF rows;
    2. every steady window after them (`fp8_mtp0` 40k@8-20 / 90k / 120k = the MTP-OFF arm of section 6);
    3. `nvfp4_target_only` 90k / 120k residency (the core of section 5).
  - The step ends `SERVER_DIED`. Only the greedy oracle rows that were saved before the TF survive.
- **Evidence:**
  - The vocab size comes from P4 `bf16_not_fp8_per_rank.lm_head n=62080` per rank × 4 = 248,320.
  - The Merlin fork of the same vLLM 0.32.0 had to add `vllm/merlin/prompt_logprobs.py` for exactly this path: a per-step GPU tensor budget of 384 MiB, with a chunk loop `score_legacy_prompt_chunks` replacing the upstream single `compute_logits(prompt_hidden_states)` at `gpu_model_runner.py:5405`.
  - This path never ran anywhere: the kit uses the fake server.
- **Fix (either option):**
  - (a) Minimal: in s6(0) and s5(0), call `tforced(...)` as the LAST item of `body`, after all `levels(...)`, so that a crash costs only the TF rows. Record `alive` right after it.
  - (b) Better: run the TF rows as their own short step on a dedicated server per KV dtype, with `--max-num-batched-tokens 256 --gpu-memory-utilization 0.85 --max-num-seqs 2`. 256 × 248k × 6 B is about 0.4 GB, and a 90k prompt then takes about 352 steps.

### F2 HIGH: the Merlin path (sections 1a-1d) has never executed through the notebook glue, and a missing payload degrades SMOKE_A silently
- **Where:**
  - B:103 (`MERLIN_PAY ... must=False`);
  - B:190-195 (`if FAKE: MERLIN_ENG = {"fake": True}` short-circuits extraction and install in the kit);
  - B:196-216 (tar extract, install, `import launch`, CUDA_HOME), B:236-261 (`ML.build`, `distributed_ports`, env).
- **Scenario:**
  - In kit v1, MERLIN_PAY was None (wrong variation). Even with the right variation, the kit's FAKE mode never runs `tarfile.extractall`, `install_merlin_offline.sh`, `import launch`, `ML.build("27b", "27b-tp1-sm89-24g", ...)` or `distributed_ports`.
  - Any LaunchError, KeyError or import failure there turns every Merlin step into ERROR in 0 s. P4 and P5 still burn the SMOKE_A GPU session.
  - If the GPU notebook is pushed with the old `.../Other/default/1` handle, SMOKE_A again finishes "rc 0" in about 15 min with no Merlin data. There is no fail-fast.
  - There is no kernel-metadata for the three GPU notebooks in the repo: only `cpu_kernel_e23/kernel-metadata.json` carries the corrected `e23-merlin-env/Other/v1/1`.
- **Evidence:**
  - kitout_v1 `SMOKE_A.steps`: `a_merlin_tp1_gpu0 ERROR "Merlin checkpoint / env not attached or incomplete"` and `d_merlin_tp4 ERROR` (both 0.0 s), notebook rc 0.
  - P2 v7 proved only the install script on the CPU image (`offline_install rc 0`, `INSTALL OK`, 249.5 s). It did not prove the notebook's own glue.
- **Fix:**
  - (1) CELL1: when `MODE == "SMOKE" and PART == "A"`, `raise RuntimeError` if `not MERLIN_OK or MERLIN_PAY is None`, before any GPU time. In FULL, put a loud `R["errors"]` entry instead of failing.
  - (2) Kit v2: add a non-FAKE Merlin block on the CPU box (E23 inputs attached):
    - `merlin_install()` for real (about 5 min of extract plus about 250 s of install);
    - then `merlin_server(...)` for tp1 / rep1 / tp4, with `help_check` skipped (CPU `--help` cannot infer the device);
    - assert that the argv contains `--port 810x --max-model-len 131072 --served-model-name q27`, and that the env has `CUDA_VISIBLE_DEVICES`, `MASTER_PORT`, `VLLM_PORT`, `LD_PRELOAD`, and `CUDA_HOME/bin/nvcc`.
  - (3) Generate `kernel-metadata.json` for the 3 GPU notebooks from the same `model_sources` list as the kit.

### F3 MEDIUM: a P4 timeout leaves an orphan child holding GPU 3, and `join_p4` ignores the GPU-3 release gate
- **Where:**
  - B:534-541 (`subprocess.run(..., timeout=1500|3600, start_new_session=True)`), B:553-558 (`join_p4`: result of `wait_gpu_release(180, gpus="3")` discarded);
  - bench_fp8_gemm.py:563 (per-shape child, `--timeout-per-shape` default 900).
- **Scenario:**
  - SMOKE runs 2 shapes × 900 s = 1800 s, more than the 1500 s notebook timeout. FULL runs 5 × 900 s = 4500 s, more than 3600 s.
  - On `TimeoutExpired`, `subprocess.run` kills only the orchestrator. `start_new_session` means no group signal reaches the running shape child, so the child keeps GPU 3 until its own work ends.
  - In FULL, `c_merlin_tp1_x4` then starts replica 3 on a busy GPU 3, and with KV_BYTES pinned at 7.5 GiB it fails to launch.
  - If the child is still alive at the end of that step, the gate on 0-3 returns False and ABORTs the whole lab.
- **Fix:**
  - Pass `--timeout-per-shape` (SMOKE 600, FULL 650) so the bench self-bounds below the notebook timeout.
  - Use `Popen(start_new_session=True)` + `wait(timeout)` + `os.killpg(p.pid, SIGKILL)` on timeout.
  - In `join_p4`, store the gate result; if it is False, retry once for 300 s, then set `ABORT` or mark the GPU-3 steps `SKIPPED_GPU3_HELD`.

### F4 MEDIUM: the P5 host-staged all-reduce is measured while 16-32 prefetch threads stream 17-64 GB from GCSFUSE
- **Where:** B:793-797 start `PREF` before B:802 (`p5_bench`), in both SMOKE_A (Merlin) and FULL (Merlin + FP8 + INT4).
- **Scenario:**
  - The P5 latency path is host memory plus PCIe (GPU to pinned host to GPU).
  - The concurrent FUSE reads and page-cache fills compete for memory bandwidth and CPU, which inflates `host_eager` / `host_graph` µs and biases the verdict against the lever. NCCL is less affected, so the bias is not symmetric.
- **Fix:** move the `PREF` block to after `step("2_host_allreduce", ...)`. Prefetch then overlaps the Merlin extract and install, which take minutes anyway.

### F5 MEDIUM: Merlin serves with system GCC 11, and the patched FlashInfer tile JIT under that compiler is untested; the fallback retries with the same compiler
- **Where:** B:241-248 (CC / CXX / CUDAHOSTCXX popped, so nvcc and Triton use `/usr/bin/gcc` 11.4), B:574-579 (the fallback changes only seqs / capture).
- **Evidence:**
  - Merlin `release/install-env.sh:90-98`: "The native CUDA ops need GCC/G++ >= 13 and C++20". The wheel products are precompiled.
  - P2 log line: `LIVE TILE PROOF PENDING: serve an MTP request with FLASHINFER_SM8X_FP8_SMALLQ=cta32`, so the patched `batch_prefill.cu` tile is JIT-compiled at first serve. No run has compiled it with GCC 11.
- **Scenario:** a ninja / nvcc host-compile error, then LAUNCH_FAILED, then a fallback with the same compiler, so the TP1 / TP4 rows are lost.
- **Fix:**
  - In `s1_tp1` / `s1_tp4`, if the first log tail matches `ninja: build stopped|error: |nvcc fatal|c\+\+20|-std=`, retry once with `CC=MERLIN_CC CXX=MERLIN_CXX CUDAHOSTCXX=MERLIN_CXX` (the conda GCC 13 that the install already prints).
  - Use a fresh `FLASHINFER_WORKSPACE_BASE` for the retry.

### F6 MEDIUM: GPU-release gate edge cases
- **Where:** L:190-205 and B:451-459.
- **Edge case (a): one 180 s miss aborts the rest of a FULL run.** A single 180 s miss sets `ABORT` for the entire remaining 10 h. Driver teardown of a TP4 server killed mid-capture can exceed 180 s.
  - Fix: on False, list `ps -o pid,stat,cmd --sid <pid>` for every killed server, re-`killpg(SIGKILL)`, wait another 300 s, record `gpu_mem()` in `out["gpu_released_mem"]`, and ABORT only after that.
- **Edge case (b): an nvidia-smi timeout counts as "released".** `gpu_pids` / `gpu_mem` return None on an nvidia-smi exception or timeout, which can happen with a GPU in an Xid state after a crash. `wait_gpu_release` then returns None, and with_servers treats that as released.
  - Fix: once nvidia-smi has worked at CELL1, treat a later None as False.

### F7 LOW-MEDIUM: `/dev/shm` segments of SIGKILLed TP4 servers can accumulate
- **Where:** L:109-134.
- **Scenario:**
  - `killpg(SIGTERM)` hits the API server, EngineCore and all 4 workers at once. The leader then exits, and an immediate group SIGKILL also kills the multiprocessing resource_tracker before it unlinks the shm ring buffers.
  - FULL launches about 25 servers. Leftover `psm_*` segments are RAM. If `/dev/shm` on the GPU box is small, a late TP4 / P5 run fails with SIGBUS / ENOSPC.
- **Fix:**
  - Record `df /dev/shm` in `with_servers` before start and after the gate.
  - After a successful gate with no LIVE server, remove `/dev/shm/psm_*` (and vLLM-named segments) older than the server start.
  - The P5 env probe already records `dev_shm`.

### F8 LOW: a `save()` race with the prefetch thread can raise out of `step()` and abort the cell
- **Where:** B:64-68, B:171-173.
- **Scenario:**
  - `json.dumps(R, indent=1)` uses the pure-Python encoder, which yields between items so the GIL can switch.
  - The prefetch thread inserts `R["setup"]["prefetch_*"]` without the lock. If that happens while the main thread is iterating `R["setup"]`, the main thread gets `RuntimeError: dictionary changed size during iteration` inside `step()`'s `save()`. That `save()` is not in the try, so CELL5 stops and CELL6 never runs.
  - The probability is small, but the effect is fatal.
- **Fix:** in `go()`, do `res = L.prefetch(...)`, then `with _SAVE_LOCK: R["setup"][...] = res`, then `save()` inside a try/except.

### F9 LOW: install timeouts kill only `bash`
- **Where:** B:129-137.
- **Scenario:** on timeout (stock 2400 s, Merlin 5400 s, EAGLE 2400 s), uv / tar / python keep running as orphans and keep using CPU while servers are measured.
- **Fix:** `Popen(["bash", "-c", ...], start_new_session=True)`, then on timeout `os.killpg(SIGKILL)`.

### F10 LOW: fallback servers can start after HARD_STOP
- **Where:** B:573-579, B:618-622; L:239-254 (the watchdog fires once).
- **Scenario:** the watchdog kills the first attempt, which becomes LAUNCH_FAILED, and the fallback then starts a new server that no watchdog will kill. It is bounded only by `timeout = max(60, ...)`, so 1-3 min wasted after the stop.
- **Fix:** at the top of `with_servers`, if `ABORT["why"]`, return `{"status": "SKIPPED_ABORT"}`. Skip the fallback when the first attempt failed under an ABORT.

### F11 LOW: an optional INT4 input that is missing still launches servers on model "None"
- **Where:** B:99-100, B:738, B:754.
- **Scenario:** `s4_int4` / PP4 start with model "None" and fail after a vLLM import. Each also leaves misleading `NO_GO_GARBAGE_OR_ERROR` rows.
- **Fix:** have these step functions return `{"status": "SKIPPED_REQUIRED_ARTIFACT"}` when `INT4 is None`.

### F12 LOW: no accelerator check
- **Scenario:** a commit on the wrong accelerator (2×T4, P100, or GPU off) spends the 2-12 h failing slowly.
- **Fix:** in CELL1 (non-FAKE), require that `nvidia-smi -L` lists exactly 4 lines containing "L4", or raise.

### F13 LOW: no release gate after P5
- **Where:** B:802-804.
- **Fix:** after `s2`, call `L.wait_gpu_release(180, fake=bool(FAKE))` on all 4 GPUs. P5's own watchdog kills the worker groups, so this is defensive.

### F14 LOW: `_patch()` is called outside the try in `s3`
- **Where:** B:684.
- **Scenario:** if `patch` times out (120 s) or raises after `copy2(target, bak)`, the revert in `finally` is never reached. GNU patch writes through a temp file, so the target is very likely intact.
- **Fix:** move `pre = _patch()` inside the try. In finally, restore from `bak` whenever `bak` exists and the target sha differs from it.

### F15 LOW: disk footprint and visibility
- The Merlin install keeps `$DEST/uv-cache`. With `UV_LINK_MODE=copy` this is a second full copy of the about 3.7 GB wheelhouse, unpacked.
- The EAGLE uv cache defaults to `/root/.cache/uv` on the overlay.
- Peak `/tmp` use is under 60 GB (stock venv, Merlin pay + cache + venv, EAGLE pay + venv, caches), so it is fine against 1.1 TB. Note that the overlay is host-shared (P2: `overlay 8.0T 7.0T used`).
- **Fix:** delete `TMP/merlin_off/uv-cache` after INSTALL OK, and log `df -h /tmp /dev/shm /kaggle/working` in `run()`.

### F16 LOW: `CUDA_DEVICE_ORDER` is not set for P4 children and stock servers
- Merlin sets `PCI_BUS_ID`, and the gate uses nvidia-smi (PCI-order) indices.
- P4's `CUDA_VISIBLE_DEVICES=3` uses the default FASTEST_FIRST order. For 4 identical L4s this normally equals PCI order, so the risk is low.
- **Fix:** add `CUDA_DEVICE_ORDER=PCI_BUS_ID` to `STOCK["env"]`.

### F17 LOW: the lab module lives directly in `/tmp`
- **Where:** B:93-94, L:76.
- The lab module and `sys.path` live directly in `/tmp`, and servers run with `cwd=/tmp`.
- `python -m` children, such as inductor compile workers, put cwd on `sys.path`, so any `*.py` that lands in `/tmp` can shadow a module.
- Today the only such file is `e23_lab.py`, so there is no live problem.
- **Fix:** use `TMP / "e23_lab"` as both the module dir and the server cwd.

## Answers to the focus questions

**(a) Input discovery**
- `find()` rglobs from the slug, so it is variation-agnostic and handles the lowercase framework names. `.../e23-merlin-env/other/v1/1/merlin_offline_tar.bin` is found.
- `sorted()` puts `competitions` before `datasets` before `models`, and a top-level `config.json` before any subdir. The kit identity shows top-level dirs for all 4 checkpoints.
- Searches across the whole of INP (KITS, ARC, the RERUN test file) are wide but correct, and walk only a few hundred entries.
- Optional inputs that are missing are handled for EAGLE (SKIPPED), not handled for INT4 (F11), and are silent for Merlin (F2).

**(b) Disk and RAM**
- Fine: under 60 GB on /tmp, and /kaggle/working holds only logs, JSON, the P5 work dir and the P4 parts (MB-scale, under 200 files).
- The profiler dirs are in /tmp and are removed. A trace JSON load is about 1 GB of Python memory.
- RAM against about 202 GB is fine: up to about 64 GB page cache plus 4 Merlin replicas. The shm accumulation is covered in F7.

**(c) Process lifecycle**
- Leader exit, then SIGKILL of the group, is correct (kit `killed_dead` true, port reuse 3.0 s).
- vLLM workers stay in the session.
- Compute-app PIDs are usually not visible inside the Kaggle container, and the gate correctly relies on `memory.used < 1 GiB` per GPU.
- The orphan, gate and thread issues are F3, F6, F9 and F10. The Stream / greedy / steady threads are bounded (stop + join 10 s; `client_leak` recorded).

**(d) GPU concurrency**
- P4 on GPU 3 alongside Merlin TP1 on GPU 0 is safe:
  - the P4 parent never initializes CUDA, and its children get `CUDA_VISIBLE_DEVICES=3`;
  - the gate for TP1 filters to GPU 0;
  - Merlin's `GPUS` sets `CUDA_VISIBLE_DEVICES` (launch.py:498/523) for each replica;
  - each replica has its own port, MASTER_PORT / VLLM_PORT (Merlin's `distributed_ports`) and Triton cache.
- P5 runs before any server, so all 4 GPUs are free. Its contamination by prefetch is F4.

**(e) Patch #60532**
- GNU `patch` exists on the pinned image: Merlin's `apply-flashinfer-patch.sh` requires it, and P2 passed on that image.
- I dry-ran the exact extracted hunk set against the v0.32.0 `vllm/v1/attention/backends/flashinfer.py` (`_scratch/e22six/v032`): all 13 hunks succeeded with offsets only, no fuzz, rc 0.
- The behaviour around the apply is correct:
  - dry-run before apply;
  - restore from the backup copy on a failed apply;
  - revert by `copy2`, whose old mtime invalidates the patched `.pyc`;
  - the revert runs in finally, and a failed revert sets ABORT.
- The one remaining hole is F14.

**(f) RERUN guard**
- Correct: a minimal valid submission, no exception, and cells 2-6 under `if not RERUN:`. Kit v1: rc 0, `submission_format_ok` true.
- A commit never sets the variable. The `[[0]]` submission scores 0, so this notebook must never be submitted.

**(g) Size and offline**
- The 3 notebooks are 116.7 kB each, under 400 kB, and the printed outputs are small.
- Offline settings:
  - `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE` and `VLLM_NO_USAGE_STATS` are set for stock, EAGLE and Merlin (Merlin also sets `DO_NOT_TRACK`);
  - Merlin uv uses `UV_OFFLINE=1`, no-index plus find-links, and `UV_PYTHON_DOWNLOADS=never`;
  - EAGLE uses `--no-index`.
- JIT toolchains:
  - nvcc for stock is the wheelhouse cu13 `CUDA_HOME`, the same pattern as E18 (run on GPU);
  - for Merlin it is `venv/nvidia/cu13/bin/nvcc` (install-env.sh dies without it);
  - the `-lcuda` shim is on `LIBRARY_PATH`.
- The one open toolchain item is F5 (GCC 11 for the Merlin JIT).

## Verified correct (no action)

**Server lifecycle (e23_lab.py)**
- `Server.start`: the connect-based port check with a 90 s wait, and `start_new_session`.
- `kill()`: SIGTERM, wait 40 s, then SIGKILL of the group even after the leader has exited.
- `rss_mb` via `ps --sid`.

**Release gate**
- Per-GPU filtering in `gpu_pids` / `wait_gpu_release`: a bench on another GPU cannot fail a server's gate.

**with_servers**
- Always kills in finally.
- Records `LAUNCH_FAILED` / `SERVER_DIED` / `BODY_ERROR` with log tails.
- Timeout is clamped to `left() - 120`.

**Watchdog**
- A daemon thread; it kills every LIVE server and sets ABORT.
- `step()` skips on ABORT or budget, and `levels()` re-checks ABORT for every level.

**Merlin install**
- Fixed `/tmp/merlin_uvpy` / `/tmp/merlin_gcc13` prefixes.
- `rmtree` of the old `merlin_off` before the install (the script refuses an existing venv).
- The payload is deleted after the install, and the shim is copied out first.
- The failure is cached, so later Merlin steps fail fast.
- `tarfile.extractall(filter="data")` (Python 3.12) is used for both payloads.

**P5 bench**
- Own budgets: main 900 s + 4 × 240 s + cd 900 s = 2760 s, under the 3000 s bench timeout, which is under the 3300 s notebook timeout.
- A watchdog kills the rank groups, and crash leftovers in `/dev/shm/p5har_*` are cleaned.

**Profiler**
- `profile()` is bounded (`delay_iterations` / `max_iterations` with a bare-POST fallback).
- The trace dir is in /tmp and removed after parsing.
