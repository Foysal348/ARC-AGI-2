# Startup, weight-load path and Kaggle environment risks for Qwen3.8-27B (RedHat INT4 + DFlash2) on SGLang 0.5.21, Kaggle 4×L4, under the 2026-10-21 /kaggle/input slowdown

Research date: 2026-10-23. Read-only. Labels: **[M]** = measured (our logs or a number someone measured and posted), **[C]** = claimed by a person or a vendor, not verified, **[S]** = read directly in source code, **[I]** = my inference.

Local evidence used (each read in full): `D:/ARC AGI 2/resources/Analysis and Journals/Kaggle discussions (2026-10-22)/` (`00_SUMMARY.md`, `topic_751344.md`, `topic_738313.md`, `topic_704408.md`, `topic_689054.md`); ARC dumps in `D:/ARC AGI 2/_scratch/webresearch/` (`d_746520.md`, `d_747515.md`, `d_750234.md`, `d_743927.md`, `topteam_comments.md`); `D:/ARC AGI 2/_scratch/ext/dafr/WRITEUP.md` and `serving/` (Franzen); Franzen's notebook `D:/ARC AGI 2/resources/ARC-AGI-3 related/ARC3_top10_kv/01_dfranzen__arc-agi-3-milestone-2-solution/`; khoinguyennguyen's notebook `D:/ARC AGI 2/_scratch/webresearch/k_arc-agi-2-agentic-multi-turn-reasoning.src`; the SGLang 0.5.21 source `D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang/srt/`; our SGLang server logs and Kaggle kernel logs.

---

## 1. Storage: what is on GCSFUSE vs NFS, measured read rates, page cache, and GCSFUSE read best practices

### Takeaway
Since about 2026-10-21, some `/kaggle/input` content is mounted as `FUSE.GCSFUSE` instead of NFS, and the NFS mounts got slower too. Which content goes where is not documented and has to be detected at run time. Reads are slow per stream (about 10.5 MB/s single-threaded) but scale with parallel streams (about 107 MB/s at 16 threads). Every GCSFUSE tuning knob (file cache, parallel downloads, buffered reads, read-ahead) is a mount-time or root option that a notebook cannot set. The one lever we control is **many concurrent large sequential readers, each on its own file descriptor and its own byte range**.

### Cited Findings
- **[M] The mount type changed.** Jeroen Cottaar (an ARC-AGI-3 run) logged `Filesystem type for checkpoints: FUSE.GCSFUSE`. A 2026-10-17 run of the same code and datasets logged `NFS` and loaded all 10 shards in 88 s; every other session he has logs for was NFS and loaded in 88–113 s — `topic_751344.md` (Jeroen, 2026-10-22 05:52), [thread](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/751344).
- **[M] On GCSFUSE the load barely moved.** vLLM started loading a 100.5 GiB, 10-shard checkpoint 370 s into the session. After 7 h 22 min only 3 of 10 shards were loaded (1.8–2.8 h per shard) — `topic_751344.md` (Jeroen). Earlier he wrote ">9 hours (2 hours per shard)" for an offline run of the same notebook — same thread, 05:48.
- **[C/M] The split is by "category" of upload.** amn: "part of Kaggle's uploaded files by category comes to NFS, and the other part is GCSFUSE reportedly much slower." He does not say which category (dataset vs model vs competition data) goes where — `topic_751344.md` (amn, 2026-10-22 08:12).
- **[M] NFS slowed down as well.** amn's probes, same notebook and files, one shard about 4.6 GiB:

  | Commit (UTC) | Time per shard | Throughput |
  |---|---|---|
  | 10-19 10:54 | 8.4 s | ~0.55 GiB/s |
  | 10-20 03:26 | 4.9 s | ~0.95 GiB/s |
  | 10-21 03:21 | 14.4 s | ~0.32 GiB/s |
  | 10-21 04:35 | ~115 s | ~0.040 GiB/s |
  | 10-21 05:37 | ~177 s | ~0.026 GiB/s |

  On 10-22 he measured about 0.01 GiB/s = 10.5 MB/s single-threaded and 107 MB/s (~0.1 GiB/s) with 16 threads — `topic_751344.md` (amn, 08:12; also 2026-10-21 20:26: "~190 s (05:33 UTC, about 24 MB/s), 8~13 times slower").
- **[M] Other teams' numbers.** NguyenThanhNhan: about 180 GB at "normal NFS speed of ~300MB/s" takes about 10 min, and on GCSFUSE "reading would take hours." His save runs took 875–1,300 s to load the same files — `topic_751344.md` (07:26 and 03:41).
- **[M, ours] Our own load time went up about 50× between 10-20 and 10-21.** The same INT4 target, read by the same SGLang 0.5.21:
  - 10-19 and 10-20 runs: `Load weight end. elapsed=` 13.16 / 13.37 / 12.84 / 37.48 / 58.70 s. Logs: `.../Experiment 12 .../Kaggle Runs/E12 SGLang {SMOKE1, EVAL30, EVAL6R5, EVAL6, EVAL12R5_LOW}/q27x/sglang_server_try0.log`.
  - 10-21 rev 6d G4B run: `Load weight end. elapsed=669.26 s`, and `Engine startup timings (s): load_weight=669.96 ... scheduler_e2e=1115.52`. The tqdm line reads `Multi-thread loading shards: 50% Completed | 1/2 [10:57<10:57, 657.15s/it]`. The kernel log has `ready in 1197s`. Source: `D:/ARC AGI 2/_scratch/e12_rev6_results/G4B_v1/q27x/sglang_server_try0.log`, `kernel_log.txt`.
  - The 18.6 GB shard took 657 s, about 28 MB/s effective, while our background `cat` prefetch was reading the same files. That matches amn's ~24 MB/s for the same day. [I]
- **[M] Reads through the mount did not fill the page cache in Jeroen's session.** He reads shards ahead of the loader to warm the cache. When loading began, available RAM was 71 GiB, against 41 GiB in the NFS run, where about 30 GiB of shards were already cached by then — `topic_751344.md`.
  - **Caveat [I]:** Linux `MemAvailable` counts reclaimable page cache as available. His comparison only holds if he measured free memory rather than `MemAvailable`, and he does not say which.
- **[S] Contradicting evidence (upstream source).** In upstream gcsfuse `OpenFile`, `op.KeepPageCache = true` is set unconditionally. The comment says it is safe to keep the page cache across opens. `UseDirectIO` (page cache bypassed) is set only when `FileSystem.ExperimentalODirect` is on or the open has O_DIRECT — [gcsfuse fs.go](https://raw.githubusercontent.com/GoogleCloudPlatform/gcsfuse/master/internal/fs/fs.go). So stock gcsfuse should fill and keep the page cache. That Jeroen's did not suggests Kaggle runs a different version, `o-direct`, or a different memory accounting (unknown). [I]
- **[M, ours] Page cache was effective on 10-21 for at least one file.** Our `cat` prefetch reads paths in sorted order (the draft directory `qwen3-8-27b-dflash2` sorts before `qwen3-8-27b-int4-redhat`). It started at kernel t = 14.0 s (`weight prefetch started: 3 files`). The DFlash2 draft (3.85 GB) then loaded in `elapsed=0.56 s`, so it came from RAM — `.../G4B_v1/q27x/sglang_server_try0.log`, `kernel_log.txt`. That mount's type was not logged.
- **[C] Google's GCSFUSE read best practices** ([performance page](https://docs.cloud.google.com/storage/docs/cloud-storage-fuse/performance), [file-cache page](https://docs.cloud.google.com/storage/docs/gcsfuse-file-cache)):
  - **File cache + parallel downloads.** `--file-cache-max-size-mb=-1`, `--file-cache-cache-file-for-range-read=true`, `--file-cache-enable-parallel-downloads=true`. The vendor claims "up to nine times faster model load time." A RAM disk is recommended as the cache medium for model weights. Tuning: `parallel-downloads-per-file`, `max-parallel-downloads` (default twice the CPU core count), `download-chunk-size-mb`. Parallel downloads are on by default from gcsfuse 2.12 *when the file cache is enabled*.
  - **Read-ahead.** Raise `read_ahead_kb` from the default 128 KB to 1 MB via `/sys/class/bdi/` (root; after mount).
  - **Buffered reads.** `--enable-buffered-read=true`: "two to five times" faster sequential reads of large files, up to 320 MB of buffers per file handle. Incompatible with the file cache.
  - **Metadata.** Stat/metadata cache TTL `-1` and `--kernel-list-cache-ttl-secs=-1`. These are mount-time; `ls -R` pre-population is a post-mount action.
  - **Which settings we can change.** The page's own table puts the file cache, parallel downloads, buffered reads, metadata and list caches and `enable-kernel-reader` under "Mount-time". `read_ahead_kb` is machine-level (root). It names no client-side thread or read-size settings.
- **[S/C] The default reader detects sequential access.** "Cloud Storage FUSE uses a heuristic to detect when a file is being read sequentially" and then sends "fewer, larger read requests" — [semantics.md](https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/semantics.md).
- **[M, vendor] Throughput needs many concurrent readers.** Google's published gcsfuse fio numbers use 144–192 jobs and reach 13–14.5 GB/s on a C4-192 VM. There is no single-thread figure — [benchmarks.md](https://github.com/GoogleCloudPlatform/gcsfuse/blob/master/docs/benchmarks.md).
- **[C] Kaggle staff, 2026-10-17:** "We're still experiencing one bad cache region which is slow right now. I've been working on a replacement" — `d_750234.md` (Dustin).

### Inferences
- **What we can control.** We cannot remount, enable the file cache or buffered reads, or write `read_ahead_kb` (all need mount options or root). Only reader-side behaviour is ours: the number of concurrent streams, each stream sequential within its own byte range (so the sequential heuristic fires), and large reads (≥ 16–64 MiB per request loop). amn's 10× gain from 16 threads (10.5 → 107 MB/s) is the measured basis. At 107 MB/s, 23.3 GB is about 3.6 min; at 10.5 MB/s it is about 37 min.
- **Our 18.6 GB `model.safetensors` is one file**, so any per-file parallelism (one thread per file, SGLang's rank split, khoinguyennguyen's preload) gives at most about 3 streams for our 3 files. Only **byte-range** parallelism within the big file gives the 16+ streams that matter.
- **Mixed evidence on the page cache** (Jeroen: not filled; upstream gcsfuse and our 10-21 draft: filled). A plan that relies only on warming the page cache is fragile. Copying to a local tmpfs or disk file makes the loader independent of the mount's caching behaviour.

### Gaps
- Which `/kaggle/input` subtrees (competition data, `datasets/…`, `models/…`) are GCSFUSE vs NFS is not documented. Detect it per path from `/proc/mounts` at run time and log it. Our 10-21 run did not log the mount type.
- The gcsfuse version and flags on Kaggle (o-direct, buffered read, file cache) are unknown. `cat /proc/mounts` and `ps aux | grep gcsfuse` in a session would show them if the daemon runs in the container (it probably runs on the host).
- How far concurrency scales above 16 threads on Kaggle (32, 64) has not been measured.

---

## 2. Local copy: disk, `/dev/shm`, RAM, tools, and what other teams do

### Takeaway
A one-time parallel byte-range copy of the 23.3 GB into RAM-backed `/dev/shm` (or local disk if `/dev/shm` is capped), with SGLang launched from the copy, is feasible on RAM grounds: ~23 GB of a ~200 GB host. It removes all dependence on the mount's page-cache behaviour and on the 4 TP ranks' slow mmap faults. The disk and `/dev/shm` sizes on the 4×L4 image are **not documented anywhere I found** and must be read with `df` at run time, with a fallback chain. Python `os.pread`/`os.pwrite` range workers (or parallel `dd` slices) work offline. fpsync and aria2 do not help with a single big file.

### Cited Findings
- **[M, ours] Host RAM is 202.6 GB** (local validation; cited in `D:/ARC AGI 2/research_notes/Qwen 27B দ্রুততম সার্ভিং সেটআপ/community_evidence.md` lines 498 and 587, "local `VALIDATION.md` row 13").
- **[M, ours, found late] `/tmp` free space is about 1.1 TB, and nearly all RAM is free at start.** Our Flash-Next stage-0 checks on Kaggle 4×L4 (early October 2026) logged `"ram_total_gb": 202.6`, `"ram_available_gb": 198.9–199.3` and `"tmp_free_gb": 1122.9` / `1148.8`. Sources: `D:/ARC AGI 2/experiments/Experiment 3 - Flash-Next Feasibility (FN)/Experiment Results/Kaggle Runs/{FN_PROBE_v4_run1, FN_PROBE_v421_run1, FN_SMOKE_v3_run, FN_v33_run}/fn_smoke/fn_smoke_report.json`. This fits the 1,500 GiB local-SSD maximum of `g2-standard-48`. [I] So a 23.3 GB copy to `/tmp` always fits even if `/dev/shm` is capped. `/tmp` write throughput and the `/dev/shm` size were still not measured.
- **[C] The GCP shape that matches 4×L4** is `g2-standard-48`: 48 vCPUs, 192 GB default memory (192–216 GB custom), 4 L4 GPUs, max Local SSD 1,500 GiB, 50 Gbps network — [GCP accelerator-optimized machines](https://docs.cloud.google.com/compute/docs/accelerator-optimized-machines). That Kaggle uses this shape, and how much local SSD it gives a container, is not stated anywhere. [I]
- **[C] Old Kaggle disk figures.** Older sources say 20 GB of saved `/kaggle/working` plus scratch outside it (16 GB in older guides). They predate the L4 pools and are not authoritative — [search summary](https://medium.com/data-science/using-kaggle-for-your-data-science-work-a2e78d692395), [O'Reilly Kaggle Book ch. 3](https://www.oreilly.com/library/view/the-kaggle-book/9781801817479/Text/Chapter_3.xhtml).
- **[C] Container `/dev/shm` defaults.** A plain Docker container gets a 64 MB `/dev/shm` unless `--shm-size` is set — [oneuptime docker shm_size](https://oneuptime.com/blog/post/2026-02-08-how-to-use-docker-compose-shmsize-configuration/markdown). Kaggle's setting is unknown.
- **[M, ours] `/tmp` is writable and already holds our SGLang venv and caches.** The server log shows `/tmp/sglang-venv/...`. The builder sets `SGLANG_CACHE_DIR=/tmp/e11-cache/sglang` etc. — `build_e11_05_s49_sglang.py` lines 345–346.
- **[S, Franzen] A range-parallel page-cache warmer.** His notebook's `precache()` splits **all files into 32 MiB chunks**, sorts them, and reads them with a `ThreadPoolExecutor` (16 threads for wheels; `threads=3, delay=60` for the model, started just before the server). Each chunk opens its own fd, `seek`s, and `readinto`s a reused buffer — `01_dfranzen__arc-agi-3-milestone-2-solution/arc-agi-3-milestone-2-solution.ipynb`, cell 6. The code comment says "128 MB", but the constant is `32 * 1024 * 1024`.
- **[S/C, Franzen] Patched SGLang lookahead plus resharding.** He runs `--weight-loader-prefetch-checkpoints --weight-loader-drop-cache-after-load` plus his lookahead patch. The patch reads **disjoint byte ranges of one shard** with `--weight-loader-prefetch-num-threads` (default 4) in 16 MiB blocks, finishes staging the current shard before loading its tensors, and keeps one shard ahead — `D:/ARC AGI 2/_scratch/ext/dafr/serving/README.md` §Prefetch, `build_bundle_pennyroyal.sh` (patch `05-prefetch-lookahead`). Write-up: "A single very large shard made it difficult to keep prefetching aligned with the loader's access pattern. Smaller shards and a background prefetcher made startup much more reliable" — `WRITEUP.md` §1.
- **[S, khoinguyennguyen] A per-file page-cache preload.** `preload_weights()` walks the model directory and reads **each file whole** with `ThreadPoolExecutor(max_workers=CONFIG["preload_workers"]=16)` and 64 MiB `read()` chunks, then discards the bytes ("the kernel keeps recently-read file pages resident") — `k_arc-agi-2-agentic-multi-turn-reasoning.src`, Block 02. The parallelism is across files only.
- **[C] mp's mitigation (result not posted).** "checking whether `/kaggle/input` is a FUSE mount. If it's NFS, nothing changes. If it's FUSE, try copying it to local disk once at startup, instead of letting all 4 GPU processes read it from the slow GCS Fuse mount. This should cut the startup time by half" — `topic_751344.md` (mp, 06:42).
- **[S, ours] Our current prefetch is a single stream.** One background `cat file1 file2 file3 > /dev/null` — G4B notebook cell 3 (`Q27-XHIGH-E12_DUCK_REPL_SGLANG_EVAL12R6G4B.ipynb`). At 10.5 MB/s that is about 37 min for 23.3 GB, longer than the 30-min launch timeout (`00_SUMMARY.md` §1).
- **[S, ours] File layout.** INT4 = `model.safetensors` 18.6 GB + `model_mtp.safetensors` 0.85 GB; DFlash2 = one 3.85 GB file (project memory, `sglang-int4-dflash2-plan`; `weight prefetch started: 3 files` in the kernel log).

### Inferences
- **Feasibility [I].**
  - **Size.** 23.3 GB in `/dev/shm` costs 23.3 GB of RAM. The source reads through FUSE may add up to 23.3 GB of reclaimable page cache, which can be dropped with `posix_fadvise(DONTNEED)` on the source fd after each range. SGLang's mmap of a tmpfs file maps the tmpfs pages directly, with no second copy.
  - **Peak.** About 47 GB of 202.6 GB before the KV/host buffers. The SGLang process itself needs host RAM, but nothing near 150 GB. HiCache, if adopted later, would need ~77 GB (see `agent_effective_throughput.md` line 139), still within budget.
  - **Time.** About 3.6 min at 107 MB/s (16 streams measured), or about 37 min single-stream.
  - **Fallback chain.**
    1. Use `/dev/shm` if `statvfs` free ≥ 24 GB (+ margin).
    2. Else `/tmp` (or `/kaggle/tmp`) if its free space is ≥ 24 GB.
    3. Else warm the page cache in place with range readers and launch from `/kaggle/input`.
- **A copy design that fits our code [I].**
  1. **Detect.** Log `fstype` for each source dir from `/proc/mounts` (longest-prefix match).
  2. **Pre-create the destination.** Create each destination file with its exact size (`os.ftruncate` or `posix_fallocate`). Symlink or copy the small files (`config.json`, `*.index.json`, tokenizer, templates) so SGLang sees a complete model directory.
  3. **Split.** Cut every file into 64 MiB ranges and put all ranges of all files into one queue, largest file first.
  4. **Copy.** Run N = 16–32 workers. Each opens its own source fd and destination fd and loops `os.pread(src, 16–64 MiB, off)` → `os.pwrite(dst, buf, off)`. It retries on `OSError`/short read (`EINTR`, `EIO`) with backoff, then calls `posix_fadvise(src, off, len, DONTNEED)`.
  5. **Report.** A progress counter (bytes done, MB/s over the last 30 s) feeds the progress watchdog (§8).
  6. **Verify.** Check `os.path.getsize` equality, and parse the safetensors header (the first 8 bytes give the header length; check the JSON and that the last tensor's `data_offsets` end equals the file size − 8 − header length). Optional: xxhash/SHA of a few random 64 MiB ranges against the source. A full hash would double the read.
  7. **Launch.** Start SGLang with `--model-path /dev/shm/int4` and `--speculative-draft-model-path /dev/shm/dflash2`.
  8. **Fail open.** If the copy stalls (no bytes for X min) or verification fails, delete the copy and launch from `/kaggle/input`.
- **Overlap [I].** The copy can run while pip/venv setup runs (our install takes about 1 min; the 10-21 E12 package check was at t = 66 s) and while SGLang starts. But SGLang must not open the files before they are complete, or it will mmap a partial file. Launch only after copy and verification. The long part of startup is the 6-min CUDA-graph/JIT stage, which happens after load. Overlap would need an SGLang patch and is not worth it. Alternatively, launch SGLang from `/kaggle/input` while a range warmer runs (Franzen style), but this depends on the page cache being shared (see §1).
- **Tools [I].**
  - **Offline-safe:** Python `os.pread`/`os.pwrite` + `concurrent.futures` (stdlib, no install); GNU `dd` with `skip=/seek=/count=` slices and `conv=notrunc` in parallel processes (coreutils, in every Debian/Ubuntu image).
  - **Single-stream:** `cp --sparse` (no gain).
  - **Not useful here:** `fpsync`/`fpart` parallelise across files, not within one 18.6 GB file, and are likely not installed. `aria2` segments only over HTTP/FTP/BitTorrent URLs, not local paths (from my knowledge of aria2, not checked here), and is likely not in the image.

### Gaps
- `/tmp` free space (~1.1 TB) and RAM (202.6 GB total, ~199 GB available) are measured; see above. Still missing: `df -h /dev/shm`, `/tmp` write MB/s and `nproc` on the 4×L4 competition image (commit and rerun pools may differ). No source states them. Measure them in the first cell and log them.
- Local-disk write throughput on 4×L4 has not been measured (relevant only if `/dev/shm` is capped).
- Whether the hidden-rerun pool has the same `/dev/shm` and disk as the commit pool. Dustin said submissions run on "their own, much larger pool" (`d_747515.md`), which implies the machines can differ.

---

## 3. SGLang 0.5.21 loader options and how the 4 TP ranks read the shards

### Takeaway
SGLang 0.5.21 has `--weight-loader-prefetch-checkpoints`, `--weight-loader-prefetch-num-threads`, `--weight-loader-drop-cache-after-load`, `--load-format` (including `fastsafetensors` and `presharded`) and `--model-loader-extra-config` (`enable_multithread_load`, `num_threads`). It also has a multithreaded safetensors loader, which is **on by default with 8 threads per rank**. Every parallel path in the stock loader works **per file**, though, and the built-in prefetch splits *files* across ranks. With our one 18.6 GB shard, each of the 4 TP ranks still makes one mmap stream through the whole big file. The stock flags cannot reach 16 parallel range streams. Our own range copier, or Franzen's lookahead patch, is needed. The startup "overlap" mode is excluded for us (quantized + speculative + TP4).

### Cited Findings
- **[S] Prefetch flag text.** `--weight-loader-prefetch-checkpoints` (default False): "Prefetch checkpoint files into OS page cache before loading. Each rank prefetches a fraction of the shards, reducing total network I/O on shared filesystems (NFS/Lustre) from N*checkpoint to 1*checkpoint … When enabled, multi-threaded safetensors loading is disabled by default …; set enable_multithread_load=true in --model-loader-extra-config to keep multi-threaded loading" — `arg_groups/fields/model.py` lines 284–287.
- **[S] Companion flags.** `--weight-loader-prefetch-num-threads` "Number of threads per rank for checkpoint prefetching (default: 4)". `--weight-loader-drop-cache-after-load`: "Call posix_fadvise(DONTNEED) on each safetensors shard after loading it." — same file, lines 288–293.
- **[S] How the prefetch splits work (`_prefetch_all_checkpoints`).** It takes the node-local rank and does `my_files = sorted_files[local_rank::local_world_size]`. Each file is read **whole and sequentially** by `_prefetch_checkpoint_file` in `SGLANG_PREFETCH_BLOCK_SIZE_MB` = 16 MiB blocks (`environ.py` line 319). Up to `num_threads` *files* at once per rank, in a background thread — `model_loader/weight_utils.py` lines 919–1108.
  - For our target (2 files) this is: rank 0 → `model.safetensors` (18.6 GB, one stream), rank 1 → `model_mtp`, ranks 2–3 idle. [I]
- **[S] Default loader (`DefaultModelLoader._get_weights_iterator`).** `use_multithread = extra_config.get("enable_multithread_load", True)` → `buffered_multi_thread_safetensors_weights_iterator(max_workers=num_threads, default DEFAULT_NUM_THREADS = 8)`. That iterator loads up to 8 + 1 *files* concurrently, each via `safetensors.safe_open(...).get_tensor` over mmap. With prefetch active, it falls back to the single-thread iterator unless `enable_multithread_load`/`num_threads` is given explicitly — `model_loader/loader.py` lines 383, 595–712; `weight_utils.py` lines 1240–1300.
- **[M, ours] Our 10-21 log confirms the default path:** `Multi-thread loading shards: … 0/2 … 1/2 [10:57 …, 657.15s/it]` — `G4B_v1/q27x/sglang_server_try0.log`. With 2 files there were at most 2 concurrent file streams per rank. [I]
- **[S] Every TP rank reads every tensor in full.** Each TP rank runs its own loader over **all** files: the iterator is per process, and `get_tensor` materialises the full tensor before `weight_loader` narrows it to the rank's shard (`weight_utils.py` lines 1129–1166; sharding by `tp_rank * shard_size` at lines 1497–1509).
  - Logical reads are therefore 4 × 23.3 GB. Physical reads are 1× only if the 4 ranks share the page cache. The SGLang flag help text states this N×-vs-1× effect. [S/I]
  - Our 4 ranks logged `Load weight begin` and `end` within 0.1 s of each other in every run, which fits lockstep reading through a shared cache. [M/I]
- **[S] `fastsafetensors` load format.** It exists (`configs/load_config.py` line 36). Its iterator maps files to ranks (`rank_file_map = {i: [f] for i, f in enumerate(f_list)}`) and defaults `enable_gds=True` — `weight_utils.py` lines 1169–1235, `loader.py` lines 672–677. It needs the `fastsafetensors` package (not in our wheelhouse as far as I found) and is also per-file. [I]
- **[S] `presharded` load format.** The first run "dumps a per-rank/per-tensor sharded checkpoint … into `<model_path>/presharded/…`". Later runs load directly, and `presharded_path`/`draft_presharded_path` in `--model-loader-extra-config` can point to a writable root — `fields/model.py` lines 52–96; `loader.py` `PreshardedModelLoader` line 1870ff.
  - Shipped as a Kaggle dataset, this would let each rank read only its ~1/4 slice in parallel (4 streams, ~6 GB each). Its support for compressed-tensors INT4 + DFLASH draft is untested. [I]
- **[S] Startup overlap mode excludes our setup.** `--startup-weight-load-mode=overlap` (prefetch during CUDA-graph capture with capture-safe dummy weights) rejects `speculative decoding is not supported`, `only TP1 and TP2 are supported`, `quantization is not supported` and `draft workers are not supported`. Our setup hits all of these — `model_executor/model_runner_components/startup_weight_load.py` lines 327–440. Our server_args show `startup_weight_load_mode: 'serial'`.
- **[S] Franzen's lookahead patch (Pennyroyal fork, not stock 0.5.21).** It adds `_prefetch_checkpoint_range` (unbuffered `open(..., buffering=0)`, disjoint contiguous ranges) and `_lookahead_prefetch_files` (one shard ahead, in loader order, per process, no cross-rank split) — `build_bundle_pennyroyal.sh` patch 05.
  - As written for that fork, it touches `loader.py` and `startup_weight_load.py`. Porting it to stock 0.5.21 means re-basing two small hunks. [I]

### Inferences
- **The stock flags do not fix our case [I].**
  - `--weight-loader-prefetch-checkpoints` alone gives one 16 MiB-block stream on the 18.6 GB file: about 30 min at 10.5 MB/s. It also switches the loader to single-thread.
  - `--weight-loader-prefetch-num-threads` adds threads per *file*, not per range, so it is useless with 1 big file.
  - `--weight-loader-drop-cache-after-load` only frees RAM, which we don't need at 23 GB, and it is incompatible with overlap mode anyway.
- **Best order of fixes [I].**
  1. Our own range-parallel copy to `/dev/shm`, then launch from the copy. No SGLang change, and it is independent of mount caching.
  2. If no local space: our own range-parallel page-cache warmer (Franzen `precache` style, 16–32 threads, 32–64 MiB chunks), started at t ≈ 0 so it runs during pip/venv. Launch SGLang when ≥ 95 % is warmed.
  3. Optionally, re-shard the INT4 checkpoint into ~2–4 GB shards (same tensors; a new Kaggle model version). Even stock per-file parallelism (8 threads/rank) then gives ~8 streams. Franzen calls resharding a "storage-layout change" with the weights unchanged.
- **Reading from tmpfs [I].** With the copy in `/dev/shm`, the default multithread mmap loader reads from RAM. Our pre-10-21 load times (12.8–13.5 s with a warm cache) are the expected load cost.

### Gaps
- Whether the INT4 + DFLASH2 combination loads correctly with `--load-format presharded` or `fastsafetensors` (enable_gds false) is not tested.
- Whether Kaggle's GCSFUSE page cache is shared across the 4 rank processes (it would be if it is cached at all) is not verified on a GCSFUSE-labelled mount.

---

## 4. JIT / CUDA-graph cache: shipping pre-built caches as a Kaggle dataset

### Takeaway
About 360 s of our ~372–385 s "target-verify CUDA graph" stage is first-batch compilation. A warm cache in the same session cut the whole stage to 12–14 s. SGLang 0.5.21 already routes Triton, Inductor, the CUDA driver cache and FlashInfer under `SGLANG_CACHE_DIR`. The **SGLang JIT kernel cache is separate** (`SGLANG_JIT_CACHE_DIR`, default `~/.cache/sglang/jit`), and our env does not set it. Caches can be captured at the end of a commit run, uploaded as a dataset, and copied into a writable `/tmp` path. The #42917 corruption came from the **CUDA driver compute cache (`nv/`)**, so that one directory should not be shipped. A known-answer check must gate any shipped cache.

### Cited Findings
- **[M, ours] The ~6 min is first-batch compile.** 10-20 EVAL6R5: `Capturing batches (bs=12 …): 10%| 1/10 [06:01<54:14, 361.60s/it]`, then the remaining 9 batch sizes in seconds. `Capture target verify CUDA graph end. elapsed=371.83 s`; `Engine startup timings (s): load_weight=13.52, … scheduler_e2e=438.49, cuda_graph={… target_verify=371.84, … draft_decode=9.78 …}` — `E12 SGLang EVAL6R5 run1/q27x/sglang_server_try0.log`.
  - Across runs (TP0): 371.27, 373.40, 373.80, 376.71, 378.20, 384.08, 384.53 s (all first launches in a fresh session).
- **[M, ours] A warm cache in the same session takes 12–14 s.** E11-04 rev2b on 10-18: W1 first launch `373.80 s` (03:25), then relaunch W2 in the same session `11.94 s` (03:59) and `13.67 s` (04:33). 10-17: 13.17 / 12.43 / 4.58 / 3.52 s — `Experiment 11 …/Kaggle Runs/E11-04 rev2b run1 (esfersami50, 2026-10-18)/e11_04/server_W{1,2}_*_try0.log`.
- **[M, ours] Serving-time Triton compiles remain.** After start: `Triton kernel 'chunk_gated_delta_rule_fwd_kkt_solve_kernel' took 1.84 s to compile after serving started …` (several times per rank) — EVAL6R5 log.
- **[S] SGLang 0.5.21 cache routing.**
  - `third_party_cache_defaults()` sets (via `setdefault`) `TRITON_CACHE_DIR=$SGLANG_CACHE_DIR/triton`, `TORCHINDUCTOR_CACHE_DIR=…/inductor`, `CUDA_CACHE_PATH=…/nv`, `TILELANG_CACHE_DIR=…/tilelang` and `FLASHINFER_WORKSPACE_BASE=$SGLANG_CACHE_DIR` ("FlashInfer appends `.cache/flashinfer`"). `redirect_third_party_caches()` "silently does nothing if … FlashInfer was imported" first — `environ.py` lines 1962–1990, called from `sglang/__init__.py` line 14.
  - `SGLANG_CACHE_DIR` defaults to `~/.cache/sglang` (line 1208). DeepGEMM and CuTe AOT also default under it (lines 1198, 1212).
  - **`SGLANG_JIT_CACHE_DIR = EnvStr(None)` — "None = unset, resolving to ~/.cache/sglang/jit; point it at a persistent mount to share builds across CI jobs"** (lines 1220–1222). It does **not** follow `SGLANG_CACHE_DIR` (`kernels/jit/utils/compile/cache.py` lines 301–308).
  - `SGLANG_CRASH_ON_JIT_COMPILE` = "Raise instead of compiling when a module misses the cache, so a deployment that expects a pre-seeded cache fails loudly" (lines 1233–1236).
- **[S, ours] Our env.** `SGLANG_CACHE_DIR=/tmp/e11-cache/sglang`, `CUDA_CACHE_PATH=/tmp/e11-cache/cuda`, `TRITON_CACHE_DIR=/tmp/e11-cache/triton`, `TORCH_EXTENSIONS_DIR=/tmp/e11-cache/torch_ext` — `build_e11_05_s49_sglang.py` lines 345–346 (same in the G4B notebook cell 6).
  - So FlashInfer → `/tmp/e11-cache/sglang/.cache/flashinfer`, Inductor → `/tmp/e11-cache/sglang/inductor`, SGLang JIT → `~/.cache/sglang/jit` (outside `/tmp/e11-cache`). [I]
- **[S, Franzen] He sets every cache path explicitly** under one prefix: `CUDA_CACHE_PATH`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `FLASHINFER_WORKSPACE_BASE`, `SGLANG_CACHE_DIR`, `SGLANG_JIT_CACHE_DIR`, plus `FLASHINFER_NINJA_JOBS=8`, `FLASHINFER_NVCC_THREADS=2`, `MAX_JOBS=8` — Franzen notebook cell 12. (He does not ship them as a dataset in the code I saw.)
- **[M/S] SGLang #42917.** On `dbirks/Qwen3.8-27B-W4A16-AutoRound` (compressed-tensors int4 g128, the same family as ours), SGLang 0.5.21 gave PPL 9.9767 vs vLLM 6.0528. The cause was a **stale CUDA driver compute cache** (`CUDA_CACHE_PATH=$SGLANG_CACHE_DIR/nv`, persisted across 0.5.19–0.5.21 containers via a mounted `/root/.cache`).
  - Emptying only `nv/` fixed it (PPL 6.2784 vs vLLM 6.2753 on 5 chunks). Emptying `jit/` alone did not. A persisted `nv/` was reused correctly on a second boot; "one bad entry can silently corrupt results." RTX 5090 (sm_120), driver 616.92 WSL2 — [sgl-project/sglang#42917](https://github.com/sgl-project/sglang/issues/42917).
- **[C] Triton cache key.** `triton_key()` (Triton version + core compiler file hashes) + source hash + backend hash ("GPU platform and architecture specifics including warp size") + options hash + cache-invalidating env vars, SHA-256 → base32 directory. "Cache entries are never shared between platforms." Preloading the cache improved startup "approximately 30%" in their benchmark — [Red Hat, Understanding Triton Cache](https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/).
  - The article does not mention ptxas or the driver in the key.
- **[C] Shipping caches in images.** Packaging Triton/vLLM caches in OCI images cut a demo start from "112 seconds" to "just over 62 seconds". The compatibility check compares backend/arch/warp size only; the article has no correctness validation of cached kernels — [Red Hat, OCI caching](https://next.redhat.com/2026/01/29/skip-the-jitters-fast-trusted-model-kernels-with-oci-caching/).
- **[C] vLLM guidance:** compile artifacts "can be reused across machines with the same environment … generate the cache directory once and share it among instances" — [vLLM torch.compile blog](https://vllm.ai/blog/2025-08-20-torch-compile). A vLLM forum answer notes that Triton/FlashInfer JIT caches sit outside vLLM's cache root — [discuss.vllm.ai](https://discuss.vllm.ai/t/why-does-changing-vllm-cache-root-not-reproduce-cold-start-latency/2910).
- **[C] FlashInfer offers prebuilt packages.** `flashinfer-cubin` ("Pre-compiled kernel binaries for all supported GPU architectures") and `flashinfer-jit-cache` (CUDA-specific: cu129/cu130/cu134): "For faster initialization and offline usage, install the optional packages to have most kernels pre-compiled" — [FlashInfer installation](https://docs.flashinfer.ai/_sources/installation.rst.txt).

### Inferences
- **What to ship [I].** At the end of a successful commit run (after the server has served traffic, so serving-time Triton kernels are also cached), tar these into `/kaggle/working/jit_cache_<sglang>_<torch>_<triton>_<flashinfer>_sm89.tar`:
  - `$SGLANG_CACHE_DIR` **minus `nv/`** (Triton if routed there, inductor, FlashInfer `.cache/flashinfer`, deep_gemm, cute_aot, tilelang);
  - `TRITON_CACHE_DIR` (`/tmp/e11-cache/triton` in our env);
  - `SGLANG_JIT_CACHE_DIR` (set it explicitly to e.g. `/tmp/e11-cache/sglang/jit`; today it is the default `~/.cache/sglang/jit`);
  - `TORCH_EXTENSIONS_DIR`.

  Upload it as a dataset version. At start, untar into `/tmp/e11-cache/` (writable; `/kaggle/input` is read-only and the JIT loaders take file locks) **before** launching SGLang, with the same env vars.
- **Do not ship `CUDA_CACHE_PATH` / `nv/` [I].** It is the driver's PTX→SASS JIT cache, keyed by driver and the source of the #42917 corruption. Our kernels are mostly prebuilt cubins or Triton/FlashInfer artifacts. If the driver cache is cold, the cost is a little driver JIT.
- **Portability [I].** Triton keys include the Triton build hash and the arch (sm_89); FlashInfer/SGLang JIT `.so` files are built for sm_89 against the venv's torch/CUDA ABI. They should transfer across sessions **only if the venv wheels are byte-identical** (same wheelhouse dataset version), with the same GPU (L4, sm_89) and a compatible driver (≥ the CUDA 13.0 runtime requirement). The Kaggle host driver may differ between machines. That matters for `nv/` (not shipped) and for whether the CUDA 13 runtime loads at all (already handled by our cuda-compat setup), not for cubins.
  - Key the tarball name on these versions, and refuse to use it if `pip freeze` of the venv differs from a manifest stored inside the tar.
- **Correctness gate [I].**
  1. After start with a shipped cache, run a fixed greedy (temperature 0) known-answer request (e.g. 64 output tokens on a fixed ARC prompt, plus `return_logprob` for the first 16 tokens). Compare tokens/logprobs to a reference recorded in the cold-cache commit run that built the cache.
  2. On a mismatch, kill the server, wipe the cache dirs and relaunch cold. This costs about 6 min, versus a silently corrupted run.
  3. Also log whether any JIT compile happened. Leave `SGLANG_CRASH_ON_JIT_COMPILE` off in production so a miss does not abort the run, but use it once in a test to prove the shipped cache is complete.
- **Benefit [I].** About 360 s saved per cold start (12-h rerun: about 0.8 % more solve time; an 80-min commit: about 7.5 %). Under the storage slowdown, startup time is solve time lost directly.

### Gaps
- Which component (FlashInfer JIT, SGLang JIT/tvm-ffi kernels, Triton GDN/verify kernels, Inductor) takes the 361 s is not attributed. Our logs show no JIT log lines during capture. Set `SGLANG_JIT_CACHE_DEBUG=1` and FlashInfer logging, or list the cache dirs with mtimes after a cold run.
- Whether `flashinfer-jit-cache` for cu130 / FlashInfer 0.6.18 covers the kernels our flags use (fp8_e4m3 KV, DFLASH verify), and whether it is in our wheelhouse, was not checked.
- The tarball size is unknown; it is probably hundreds of MB to a few GB.
- Whether caches are still valid in the hidden-rerun pool (possibly different hosts and drivers) can only be learned from a run.

---

## 5. The pinned old Docker image: could it make start slow or hang (ERROR with an empty log)?

### Takeaway
There is no documented evidence that a pinned image causes long starts or ERRORs. One participant's anecdote suggests the old image was slow "because few users use it", and Kaggle staff said image version does not affect the runtime/queue timers. Our own pinned-image runs started normally on 10-19, 10-20 and 10-21: the first runner log line came 6–13 s after kernel start. A pinned image is therefore a **possible but unproven** contributor to a session that never reached the notebook, and less consistent with the ~100 min of billed runtime than a dataset-attach/mount stall (§6). Separately, Kaggle states that submission reruns are always pinned.

### Cited Findings
- **[C] cm391 (2026-10-22):** "Since kaggle provided an updated environment image (02/10/2026) things have changed. I was using this image and it would take ages for anything to work and now it is really fast - think it's just how many users use that image. You guys should just update." CPMP replied that the image change "has absolutely nothing to do with the filesystem issue", and that his team lost a submission when the image *changed* (their code broke) — `topic_751344.md` (06:28, 06:31).
- **[C] Kaggle staff (Dustin, 2026-10-07):** "Image version doesn't affect either timer, use whatever image version works for your code. It will always be 'pinned' when rerun for submissions, even if you set it to 'latest' it will be 'pinned' to whichever one latest was for your save version submission." Also: "Runtime limits only count *running* time, not queued time" — `d_746520.md`.
- **[C] Seth Moudry asked** whether a long image pull or dataset mount counts against the 12 h, and whether to pin for "faster and more reliable starts". Staff answered only as above — `d_746520.md`.
- **[M, ours] The pinned image started normally many times.** The first log entry of every prior run of this notebook family is Kaggle's runner warning `nbformat … MissingIDFieldWarning`, at kernel time 6.44 / 9.34 / 10.64 / 11.02 s:
  - E12-V SMOKE1 run2 and run1 (esfersami50, 10-19);
  - E12 SMOKE1 (foysal, 10-19);
  - E12 EVAL30 (hosen42, 10-19);
  - the 10-21 G4B run's first cell printed at 13.4 s.

  Files: `.../Kaggle Runs/*/q27-xhigh-e12-*.log`, `D:/ARC AGI 2/_scratch/e12_rev6_results/G4B_v1/kernel_log.txt`. The image digest pinned is the caller-given `gcr.io/kaggle-private-byod/python@sha256:37c64f7d…`; Franzen also pins a private-byod digest (`…57e612b4…`) — `01_dfranzen…/kernel-metadata.json`.
- **[C] Image size.** Mirrors of Kaggle's GPU image report about 19.7–22.6 GB compressed. No pull-time measurements are published — [Docker Hub mirror listing](https://hub.docker.com/r/ygqygq2/kaggle-gpu-images.python/tags). Older Kaggle images get retired: "Kaggle's Docker images older than two years are no longer available" — [arXiv 2602.07195 summary](https://www.alphaxiv.org/abs/2602.07195.md).
- **[C] Kaggle has a Jupyter start deadline.** For "Your session ended with an error: unknown error, status code 56" on L4×4, staff said: "You've got a utility script attached that is blocking juptyer from starting up in time, its not an issue with the pool" — `topic_689054.md` (Dustin, 2026-10-02).

### Inferences
- **If an uncached image is the problem [I].** If the old image is rarely used, a fresh node may have to pull ~20 GB before the container starts. That would show as a long "starting" phase. Per Dustin, quota is only charged for container *running* time, so a pre-container pull should not bill the ~100 min that esfersami50 was billed (§6/§7). That argues against the image pull as the main cause of the 10-22 failure. It does not rule out an image-related failure *after* container start (e.g. a slow overlay/snapshot mount of an uncached image delaying Jupyter past its start deadline).
- **Testing the hypothesis [I].** Run the same notebook as a commit once with the pinned image and once with "latest". This only makes sense if our cp312 wheelhouse and the `sys.version_info[:2] != (3, 12)` guard pass on the latest image. That needs checking; the latest image's Python version is not known to me. Under [[prefer-latest-stack]], moving to the latest image with a rebuilt wheelhouse is consistent with user policy, but it is a build change for the user to decide.

### Gaps
- Kaggle publishes no image-pull time, node image cache, or "Pin to original environment" start-latency data. I found no Kaggle doc or product-feedback post that measures it (web search returned nothing relevant).
- Whether the 10-22 esfersami50 session ever got a running container (it was billed, so probably yes) cannot be confirmed without Kaggle's session events.

---

## 6. Failure signatures: ERROR with an empty log and no outputs (commit and submission), and the likely cause of the failed run

### Takeaway
An empty log (`[]`) is strong evidence that **Kaggle's notebook runner never started executing our notebook**, so neither our code nor the SGLang timeout was involved. Every previous run of this notebook family logged the runner's own `nbformat MissingIDFieldWarning` within 6–11 s of kernel start, before any cell ran. Our cell 1 also writes `submission.json` within seconds, and zero output files fits that cell never running (or Kaggle discarding outputs on a system error).

The most consistent explanation is a **platform-side failure between container start and notebook execution**: a dataset/model attach or mount stall (Kaggle staff confirmed a slow "bad cache region", and `ERRORED_MOUNTING_DATASET` happens), or Jupyter missing its start deadline. A pinned-image problem and quota exhaustion are weaker candidates.

### Cited Findings
- **[M, ours] The failed run.** `E12 SGLang EVAL12R6G4B rev6g run1 (esfersami50, 2026-10-22)/kernel_log.txt` contains exactly `[]`; there are no other files. The run was pushed about 17:16 UTC 10-22, ran about 114 min, status ERROR — `Experiment 14 …/README.md` line 45; `Notes/05 … gaps …md` lines 19–21.
- **[S, ours] The notebook's own start sequence.** Cell 1 anchors the clock and reads `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json` *before* its first print. It then writes `submission.json` and prints `MODE: …` — G4B notebook cell 1.
  - A hung read of the competition JSON would block that first print. But the runner's nbformat warning comes even earlier, so a cell-1 hang would still give a non-empty log. [I]
- **[M, ours] The runner's first line.** `/usr/local/lib/python3.12/dist-packages/nbformat/__init__.py:96: MissingIDFieldWarning: Cell is missing an id field …` at t = 6.44–11.02 s in all 4 E12 logs checked. The G4B notebook is nbformat 4.5 without cell ids (checked: `ids [False × 9]`), so this warning is deterministic for it. The E9 log ends with `[NbConvertApp] Writing … __results__.html` — Kaggle runs the notebook with nbconvert.
- **[C] Kaggle "system error" from queue time (fixed).** "Submissions that waited >15h to start were being marked as 'Kaggle Error' even if they eventually did run." This was fixed by "Extended Queue Timeout" (2026-10-08) — `d_747515.md`. Kaggle Errors "don't count towards submission limits" — `d_746520.md`.
- **[C] Dataset mount failure.** mp, 2026-10-06: a notebook came off the queue after ~13 h "and it failed with `ERRORED_MOUNTING_DATASET` from Kaggle's GCS". Staff: "We're experiencing an outage with datasets in Notebooks in a region" — `d_743927.md`.
- **[C] Staff on dataset caching (2026-10-17).** "Datasets are normally pre-downloaded/cached before sessions start … We're still experiencing one bad cache region which is slow right now." "Adding data that happens while the container has already started … *is* charged against quota." — `d_750234.md`.
- **[C] Jupyter start deadline.** "status code 56" was caused by an attached utility script "blocking juptyer from starting up in time" — `topic_689054.md`.
- **[C] Earlier infra-like failures, same code fine on resubmission.**
  - 738313: CPMP's ≤10 h notebook timed out once, then ran in 9 h 10 min. Daniel Hidru got "a Kaggle system error less than 9 hours into a competition submission"; the resubmission succeeded. mmz_001: a Save & Run All "didn't exit and kept running" — `topic_738313.md`.
  - 704408: "a successful submission return a 0.0 public score after finishing in about 15 minutes, with no error message shown". CPMP: "The same submission yields 0 in 15 minutes, then 46% the next day" — `topic_704408.md`.
  - 751344: NguyenThanhNhan, 0.00 in under 2 h with status Succeeded, caused by server start exceeding his 45-min timeout on GCSFUSE. CPMP, 10-22: "we lost one sub with Kaggle error after 12 hours. Same code that ran fine the day before. The day before we already had FUSE." Kha Vo: "Kaggle Error in my submission again" — `topic_751344.md`.
  - Seth Moudry: "Kaggle Error … after about 15 hours"; staff: queued too long. Pranay Manocha: "The available logs only describe the preview, so I cannot tell whether either hidden rerun started executing." — `d_746520.md`.
- **[C] Staff claims about logs** (older third-party course guide, may be outdated): interactive sessions end after 1 h of inactivity "and its log deleted"; batch sessions are exempt — [ND CSE 40657 Kaggle guide](https://www3.nd.edu/~dchiang/teaching/nlp/2023/kaggle.html).
- **[C] Franzen's code comment:** "Kaggle might cancel the run in no action is done in some timeframe. To prevent this, harness will start up even if backend is not yet ready" — Franzen notebook cell 4.
- **[M, ours] Quota numbers around the failed run.** esfersami50 was at 26.87/30 h when the run was queued (project memory, 2026-10-22) and at 30.21/30 h after it (`Experiment 14 …/README.md` line 45). That is 3.34 quota-hours, about 100 min of container time at 2×.
  - Remaining quota before the run was 3.13 h, about 94 min of 4×L4 time, so the session ended about 6 min after the quota would have reached zero. [I]
  - Counter-evidence: E9 on esfersami50 started at 28.44/30 h and ran 8,879 s (2.47 h, i.e. ~4.9 quota-hours) to completion on 10-13 — `Experiment 9 …/GPU EVAL12 E9 (esfer q27-xhigh-e9 v1)/q27-xhigh-e9.log` (last entry t = 8879.5 s); [[kaggle-quota-midrun]].

### Inferences
- **Ranking of causes for the 10-22 ERROR with an empty log [I].**
  1. **Most likely: an attach/mount stall after the container started.** Our inputs are a wheelhouse dataset, two Kaggle Models (~23.3 GB), the E12 package and competition data. Some of them now sit on GCSFUSE or the "bad cache region". Data attached after container start is billed, which fits ~100 min of billed time with no runner output. Kaggle then failed the session (Jupyter not up in time, or a mount error) and kept no log or outputs.
  2. **Possible: the pinned old image** was slow or uncached on that node, delaying Jupyter past its start deadline (cm391's anecdote). It is weaker because most of a pull would happen before billed "running" time.
  3. **Unlikely but not excluded: termination at quota exhaustion.** The ~94-min remaining quota vs ~100 min billed is a suspicious match, but E9 ran ~3 h past a near-empty quota on 10-13. The memory rule says Kaggle does not cancel. If this were the cause, the log of the cells that ran should still exist.
  4. **Ruled out:** our SGLang 30-min launch timeout and our code. Either would have left the runner warning plus `MODE: …` lines in the log and a fallback `submission.json` in the outputs.
- **What would discriminate next time [I].**
  - In cell 1, print one line *before* touching `/kaggle/input` (time, hostname, `uname -r`, image digest from env), flushed.
  - Then log `/proc/mounts` for `/kaggle/input` subtrees, `df` and `free`, and a 64 MiB timed read from each input.
  - Wrap the competition-JSON read in a thread with a timeout so a hung mount still produces log lines.
  - Write an early heartbeat file to `/kaggle/working`.
  - If the log is still `[]` with all this, the failure is certainly before notebook execution.
- **Retry [I].** Since an ERROR before execution is platform-side, the same notebook may simply be resubmitted or re-committed; CPMP, Daniel Hidru and Darragh all succeeded on resubmission. This needs the user's go and an account with quota; esfersami50 is at 30.21/30 h.

### Gaps
- Kaggle's session event history (queued → starting → adding data → running → error) is not visible to us. The exact failure stage cannot be confirmed without it or a staff answer.
- No Kaggle documentation lists the session-error codes or when logs are kept or discarded.

---

## 7. Quota: is 4×L4 charged at 2×, and is quota charged during queue or image start?

### Takeaway
Yes, 4×L4 is billed at 2× the weekly GPU quota: participants say so, staff fixed an accounting bug to enforce it, and our own quota burn measured it. Quota is charged only for container running time. Queue time and data attach *before* container start are not billed. The in-browser counter is an estimate from queue pickup that is corrected when the session ends. Data attached after the container has started is billed.

### Cited Findings
- **[C] Sam-wiz (2026-10-03):** "L4×4 sessions bill at 2×, so the 30 h weekly quota allows only ~15 h of L4×4 time" — `topic_689054.md`.
- **[C] CPMP (2026-10-12):** "Have you also fixed the fact that 4xL4 use is not accounted at 2x rate?" Dustin: "I fixed the accounting error that was allowing more quota a week ago yeah. … a submission every other day per team is about what we have capacity for currently." — `d_747515.md`.
- **[C] A participant:** "4xL4 costs twice the compute of the gpu quota whereas RTX PRO 6000 still uses the same compute as other GPUs" — `topteam_comments.md` line 363.
- **[M, ours] Our measurement.** "4×L4 (confirmed by 2.0× quota burn)" — `Experiment 1 …/sources/our_notebooks/PIV_Q27_V31_S12_RESULT_2026-10-04.md` line 3.
- **[C] Staff on what is billed (2026-10-17).**
  - "Your quota is only actually charged for container 'running' time, and 'adding data' that happens *before* container running does not consume quota. Adding data that happens while the container has already started … *is* charged."
  - The live counter "is based on when the session got picked up (passed the queue) until 'now'. However, once your session 'finishes' it updates the actual usage … which will effectively 'refund' dataset load time." CPMP still reports seeing quota decrease before his kernel started — `d_750234.md`.
- **[C] Staff on the submission timer:** "Runtime limits only count *running* time, not queued time. There is a queue time limit for submissions as well, but its an independent timer." — `d_746520.md`.
- **[C] Competition overview page.** I could not fetch the "upgraded-accelerators" overview page (JavaScript-rendered; WebFetch returned only the title). Its 2× wording is relayed by the caller, not verified by me.

### Inferences
- **Every startup minute costs 2 quota-minutes [I].** Our 1,197-s 10-21 start used about 40 quota-minutes before the first token. A 6-min JIT saving plus a ~10-min load saving is about 32 quota-minutes per commit run.
- **The esfersami50 10-22 bill [I].** The 3.34 h bill implies the container ran ~100 min, assuming the final usage had been posted when 30.21 was read. So the failure was after container start (§6).

### Gaps
- Whether the hidden rerun's 12-h limit counts from container start or from Jupyter start is not stated. Staff only said dataset download does not count in submission runtime (`d_750234.md`).

---

## 8. Timeout design: replacing the fixed 30-min launch timeout with a progress-tied one

### Takeaway
A fixed 30-min first-launch timeout is the single biggest risk under the new storage. It fires while a slow but healthy load is still progressing, and then our code drops the SGLang engine, which leaves the fallback submission (~0 on the LB). Others use either a short *soft* deadline that releases the client while the server keeps loading (Franzen, 12 min) or a long hard timeout (NguyenThanhNhan, 45 min, which failed on GCSFUSE).

The evidence supports a **stall detector**: abort only when no progress has been seen for N minutes, inside a global give-up bound that already exists (min(T0 + 75 min, admit deadline − 5 min)). The progress signals are bytes copied, process read bytes, and new server log lines or stage markers.

### Cited Findings
- **[S, ours] Current values.**
  - `SGL_LAUNCH_TIMEOUT_SEC = 30 * 60    # first launch includes JIT + CUDA graphs: 567 s in E11-04`
  - `SGL_RELAUNCH_TIMEOUT_SEC = 20 * 60`
  - `SGL_GIVE_UP_AT = min(NOTEBOOK_T0 + 75 * 60, ADMIT_DEADLINE - 5 * 60)`
  - The wait is `srv.wait_ready(min(SGL_LAUNCH_TIMEOUT_SEC if i == 0 else SGL_RELAUNCH_TIMEOUT_SEC, left))`.

  Sources: `build_e11_05_s49_sglang.py` lines 107–109, 783–789; G4B cell 2/6. After a timeout, `e11_next()` finds no error line in a log that is still loading and returns None → `E11Fail`, `ENGINE_READY = False`, no solving, fallback submission — `00_SUMMARY.md` §1 table.
- **[M, ours] Margin.** 10-21 used 1,197 s of the 1,800 s (66 %), with load_weight 670 s. At the 10.5 MB/s single-stream rate the load alone would be ~37 min — `00_SUMMARY.md`; G4B_v1 logs.
- **[S, Franzen] Soft deadline.** `SERVER_STARTUP_TIMEOUT = 12 * 60` counted from `NOTEBOOK_START_TIME`. The loop polls `/health` every 3 s, prints elapsed every 30 s, and `break`s on `proc.poll() is not None` (server died). On the deadline it prints `DEADLINE … releasing the benchmark with the server still loading` and does **not** kill the server. The harness starts anyway, with `ARC3_HTTP_RETRY_INITIAL_SECONDS='900'` and one warm-up action game — Franzen notebook cells 4 and 12.
- **[C] NguyenThanhNhan:** a 45-min timeout for server startup (normally ≤ 15 min); on GCSFUSE "the server startup never finished in time, so no inference ran and placeholder submission jsons were returned" — `topic_751344.md`.
- **[C] amn:** "In my intentionally limited 3 hours, the inference engine has never managed to get started, since the reading of ~110 GiB data has not completed" — `topic_751344.md`.
- **[S] Default server wait in Franzen's older framework code:** `wait_for_vllm_server(timeout_seconds: int = 900)` — `dafr/ARC3-Inference/inference/framework/kaggle.py` line 288. The Makefile has `SERVER_START_TIMEOUT ?= 900`.
- **[M, ours] Stage markers SGLang 0.5.21 prints.** A progress watchdog can parse these, all seen in our logs:
  - `Load weight begin` / `Load weight end. elapsed=`
  - tqdm `Multi-thread loading shards: … n/N`
  - `KV Cache is allocated`
  - `Capture target verify CUDA graph begin` / `Capturing batches (bs=…) k/10` / `… end. elapsed=`
  - `Capture draft verify CUDA graph end`
  - `Engine startup timings (s)`
  - `Uvicorn running on`

### Inferences
- **Proposed design [I].** It keeps the existing hard bound, and only the inner rule changes.
  1. **Phase 0, copy/warm** (§2), before launch:
     - progress = bytes copied (from the copier's counter);
     - stall = no ≥ 64 MiB of progress in 10 min → abandon the copy and launch from `/kaggle/input` (fail open);
     - log MB/s every 30 s.
  2. **Phase 1, launch → `Load weight end`:**
     - progress = growth of the server log file **or** growth of `read_bytes`/`rchar` in `/proc/<pid>/io` summed over the SGLang process tree (works even when tqdm is silent during mmap faults);
     - stall = neither grows for 10 min (or 15 min for the first load);
     - no fixed cap other than the global give-up.
  3. **Phase 2, `Load weight end` → `Uvicorn running`:**
     - expected ≈ 6.5 min cold, ≈ 15 s with a shipped cache;
     - stall = no new log line for 12 min (the bs=12 first capture batch prints nothing for ~6 min, so keep this > 8 min);
     - the 361 s gap should be shown as "compiling" in our status line.
  4. **Hard bounds:**
     - keep `SGL_GIVE_UP_AT = min(T0 + 75 min, admit deadline − 5 min)`;
     - after it, do not relaunch;
     - but **do not kill a server that is still progressing**. Keep polling `/health` and let the solve loop start when it comes up (Franzen-style soft release), because the solve window is already computed from real start time (`00_SUMMARY.md`: "A slow but successful load only costs the lost time").
  5. **On a real stall or death:**
     - if `proc.poll()` is not None, use `e11_next()` diagnosis (existing);
     - if it is a stall with the process alive, dump `py-spy dump` or `/proc/<pid>/stack`, if available, plus the log tail, then kill and relaunch once from the other source (local copy ↔ `/kaggle/input`) inside the give-up bound.
- **Values [I].** The 10-min stall windows follow the measured 10-21 rate: at ≥ 24 MB/s, 10 min moves ≥ 14 GB, so any working reader shows large progress. The 75-min outer bound already exists and is approved code. This change does not touch thinking budgets ([[no-token-cuts-below-base]] unaffected).

### Gaps
- Whether `/proc/<pid>/io` `read_bytes` counts FUSE reads served by the gcsfuse daemon. `rchar` (syscall-level) does, `read_bytes` (block layer) may not, so use `rchar`, plus page-fault counts (`/proc/<pid>/stat` majflt) for the mmap loader. This needs a quick check on Kaggle.
- Kaggle's possible idle-session cancellation (Franzen's comment) is not documented. Logging a heartbeat line every 30 s during load and compile, as Franzen does, is a cheap guard.
