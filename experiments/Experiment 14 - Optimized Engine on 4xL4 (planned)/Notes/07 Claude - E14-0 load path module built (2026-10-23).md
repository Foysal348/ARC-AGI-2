# E14-0 load path: module built and CPU-tested, not yet in a notebook (Claude, 2026-10-23)

**Files:**
- `src/e14_loadpath.py` (stdlib only, ~570 lines).
- `src/test_e14_loadpath.py`: 34 tests.

**How to run the tests:**
```
"D:/ARC AGI 2/_scratch/e14_venv/Scripts/python.exe" -m pytest -q -p no:cacheprovider --noconftest --rootdir=. --confcutdir=. --import-mode=importlib test_e14_loadpath.py
```
- The extra flags are needed because a stray `D:\conftest.py` and `D:\__init__.py` (ARC-AGI-3) would otherwise be imported.
- Result: **34 passed, 5 of 5 full runs.**

**No GPU, no Kaggle push, no notebook built.**

## What it does

| Part | Function | Behaviour |
|---|---|---|
| First-line facts | `env_report(paths)` | mount point and fstype per input path (from `/proc/mounts`, longest prefix), df of /dev/shm, /tmp, /kaggle/working, /kaggle/tmp, MemTotal/MemAvailable, nproc, CPU model. These are the facts the 10-22 ERROR post-mortem lacked |
| Storage probe | `read_probe(dir)` | One stream, 64 MiB of the largest file, MB/s |
| Copy | `parallel_copy(src, dst_root)` | 64 MiB ranges, 24 daemon reader threads, own file handles per range (seek+read), retry 4× with backoff. Files ≤ 256 MiB use plain copy. Destination preallocated (sparse) |
| Verify | `_verify_tree` | Size; **first and last 1 MiB compared byte for byte** (size alone proves nothing after truncate-preallocation); safetensors header (8-byte length, JSON, last tensor end == file size) |
| Destination | `choose_dest` | /dev/shm, then /tmp. Needs free ≥ 1.02 × size + 2 GiB. /dev/shm additionally needs MemAvailable ≥ size + 24 GiB, because tmpfs pages are RAM and cannot be evicted |
| Fail open | `stage_models` | Never raises. Copy stall, deadline, error or verify failure → partial copy deleted → ranged parallel page-cache warm in place → launch from the original paths |
| Background | `Staging` (daemon thread) | Overlaps the copy with the venv install, as the old `cat` prefetch did |
| Launch watchdog | `StallMonitor.wait(is_ready, give_up_at)` | See below |
| JIT cache | `jit_cache_env`, `save_jit_cache`, `restore_jit_cache`, `known_answer_ok` | See below |

**Launch watchdog, in detail:**
- Progress = the server process tree's I/O counter grew, or the log grew.
  - The I/O counter is `rchar` plus 4 KiB × (minflt + majflt), from `/proc/<pid>/io` and `/proc/<pid>/stat`.
  - Phase 1 counts any new log byte. Phase 2 starts at "Load weight end" and counts only new log lines.
- Stall limits: 900 s in phase 1, 720 s in phase 2.
- Gives up only on a stall, a dead process, or `give_up_at`.

**JIT cache, in detail:**
- `SGLANG_JIT_CACHE_DIR` is set explicitly; it is not under `SGLANG_CACHE_DIR`.
- The tar never contains `sglang/nv` or `cuda/` (the CUDA driver compute cache, SGLang #42917).
- It is restored only when the version manifest is an exact match, with a path-traversal check.
- After a restore, a greedy known-answer check must match the cold reference: identical token ids, and the first 16 logprobs within 0.05.

## Problems found and fixed while building (each has a test)

1. **A hung FUSE read would have frozen the stall timeout.**
   - `with ThreadPoolExecutor` joins all workers on exit, and its workers are non-daemon, so interpreter exit also waits on them.
   - Fix: own daemon-thread pool; hung threads are never joined.
   - Tests: `test_hung_read_raises_stall_quickly_and_cleans_up` (gives up in < 5 s) and `test_hung_reader_does_not_block_interpreter_exit` (a subprocess exits while a reader is blocked forever).
2. **Size checks pass on a broken copy.** The destination is preallocated with `truncate`, so its size and last offset are always right. Fix: byte comparison of head and tail (`test_verify_tree_detects_unwritten_tail`).
3. **/dev/shm is RAM.** 23.3 GB of weights there plus 4 TP workers plus the agent could run the machine out of memory. Fix: MemAvailable check (`test_choose_dest_dev_shm_needs_ram`).
4. **Two clocks.** `SGL_GIVE_UP_AT` is `time.perf_counter()`, while my first version used `time.monotonic()` for the copy deadline. Fix: perf_counter everywhere.
5. **rchar alone would have killed a slow but healthy load.**
   - SGLang 0.5.21 reads safetensors through mmap: `safe_open` in `weight_utils.py` L1160, unless `disable_mmap` is set.
   - Page faults on a mapping never show up in rchar.
   - With one 18.6 GB shard, tqdm prints only 0/1 → 1/1, so the log is silent too.
   - This is exactly the fallback case (copy failed, load from GCSFUSE at ~10 MB/s for ~37 min). It would have been called a stall at 15 min.
   - Fix: page faults count as progress. A process hung in a NCCL spin makes almost none (`test_monitor_counts_mmap_faults_as_load_progress`, `test_monitor_flat_counters_still_stall`).
6. **The draft model loads after the target's "Load weight end".** That silent draft load is now covered by the fault counter, not only by new log lines.
   - Source check: the other "Load weight end" (`startup_weight_load.py` L538, after graph capture) applies only to dense Llama/Qwen2/Qwen3 in fp16/bf16, so not to us. Ours is `model_runner.py` L1280.

**Two test flakes were tuned in the tests, not the code:**
- **Monitor timing test:** 1 of 5 full runs failed (0.2 s writes against a 0.5 s limit; Windows scheduler jitter). The margin is now 1.0 s.
- **Speedup test:** 1 of 5 runs failed (3.6× against a 4× threshold). The ideal ratio with 15 jobs on 10 threads plus the 0.2 s poll is about 4–6×. The test now uses 0.3 s per range and a 2.5× threshold.

Production limits are 900/720 s, against counters that change every few seconds.

## Integration plan for a NEW notebook variant (E14-A; never rebuild a Kaggle-run notebook)

The module source is embedded by the builder and written to `/tmp/e14_loadpath.py`, then imported.

| Where (G4B cell / function) | Today | Change |
|---|---|---|
| Cell 3, `# S1` prefetch (`g4b_cells.py` L408-418) | One `cat file1 file2 > /dev/null` | 1. `print(json.dumps(E14L.env_report([SGL_MODEL_PATH, SGL_DRAFT_PATH, SGL_WHEELHOUSE])))`<br>2. `E14L.read_probe(SGL_MODEL_PATH)`<br>3. `E14_STAGE = E14L.Staging([SGL_MODEL_PATH, SGL_DRAFT_PATH], workers=24, stall_sec=600, deadline=NOTEBOOK_T0 + 45*60); E14_STAGE.start()` |
| `e11_bring_up()`, before the first launch (L1210) | — | `E14_STAGE.join(timeout=max(0, SGL_GIVE_UP_AT - 20*60 - time.perf_counter()))`. If `result` is set: `SGL_MODEL_PATH, SGL_DRAFT_PATH = out[...]` and `E11_SGL["e14_stage"] = rep`. Otherwise launch from the originals |
| `E11Server.wait_ready(timeout)` (L887) | Fixed timeout, `/health` every 5 s | `E14L.StallMonitor(self.proc, self.log_path, poll_sec=2 if E11_HARNESS else 5).wait(health, give_up_at=self.t0 + timeout)`; `stats` goes into the attempt record |
| Launch caps (cell 2; L1221, L1530) | `SGL_LAUNCH_TIMEOUT_SEC` 30 min, relaunch 20 min | First launch: `left` (until `SGL_GIVE_UP_AT`, T0 + 75 min). The stall rule is what kills. Relaunch unchanged |
| `e11_install()` env | 4 cache dirs | `env.update(E14L.jit_cache_env())`, which adds `SGLANG_JIT_CACHE_DIR`. First E14-A run: **save only** (`/kaggle/working/e14_jit_cache.tar` + manifest + known-answer reference). Restore only after the user turns that output into a dataset |

**Check before any GPU run:**
- Rerun the local E11 harness (fake server) against the new `wait_ready`.
- Grep that `e11_flags` reads `SGL_MODEL_PATH` / `SGL_DRAFT_PATH` at call time.
- Grep every other use of `SGL_MODEL_PATH` (tokenizer loads) and point them at the same paths. The copy holds identical files, so either path is valid.

## Not tested, and why (honest list)

- **Real Linux `/proc`:** tested only with a fake tree. No WSL or Docker on this machine.
- **Real GCSFUSE speed with 24 readers:** the ~107 MB/s with 16 readers is amn's report (topic 751344), not ours. Our first E14-A log will give `read_probe` (one stream) and the copy MB/s.
  - At 107 MB/s, 23.3 GB ≈ 3.6 min; at 50 MB/s ≈ 7.8 min. Either way it overlaps the install.
- **Size of /dev/shm and RAM on Kaggle 4xL4:** unknown to me. `env_report` logs them, and `choose_dest` falls back to /tmp or in-place by itself.
- **Does SGLang mmap from /dev/shm load fast?** Expected (tmpfs pages are already resident), but unmeasured.
- **The JIT cache saving ~360 s per cold start** is the research estimate (M/E). Unmeasured.
