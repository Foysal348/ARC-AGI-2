# A6: Kaggle platform, boot, IO, install and memory (past-work mining for Experiment 25)

**Date:** 2026-10-31. **Scope:** read-only mining of every past experiment's boot, load, install, compile, IO, memory and platform-limit evidence, for planning Flash-Next (FN) Merlin cold boots on Kaggle 4×L4.
**Labels:** [M] measured in our runs; [C] claimed by others (Kaggle threads, vendors); [S] read in source; [I] my inference; UNVERIFIED where nothing confirms it.
**Not yet available:** the E13 RC2 GPU run (`foysalemonshanto/q27-xhigh-e13-flashnext-rc2` v1, still running; output will land in `_scratch/e13_rc2_gpu_v1`). It produces the first Merlin Flash-Next boot numbers. All Merlin-FN boot figures below are therefore projections from Merlin 27B, stock vLLM FN and SGLang FN boots.

**Path shorthand used below**

| Tag | Path |
|---|---|
| [E3R] | `experiments/Experiment 3 - Flash-Next Feasibility (FN)/Experiment Results/Kaggle Runs/` |
| [E13R] | `experiments/Experiment 13 - Flash-Next in the Duck harness on 4xL4 (planned)/Experiment Results/` |
| [E13V] | `experiments/Experiment 13 - .../Experiment Review/` |
| [RC1] | `_scratch/e13_rc1_v1/q27x/` (RC1 raw outputs) |
| [NAN] | `_scratch/e13_nanprobe_v1/q27x/` |
| [LP] | `_scratch/e13_loadprobe_v1/q27x/` |
| [E23S] | `experiments/Experiment 23 - Six levers vLLM 0.32 (planned)/Experiment Results/smoke_rev3g/` |
| [E14N] | `experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/Notes/` |
| [E15K] | `experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Results/Kaggle Runs/` |
| [MEM] | `C:/Users/hp/.claude/projects/d--ARC-AGI-2/memory/` |

---

## 1. Time budget table (phase | measured seconds | engine/model | evidence)

### 1a. Flash-Next boots we have actually run on Kaggle 4×L4

| Phase | Measured | Engine / model | Evidence |
|---|---|---|---|
| Kernel start → first notebook line | 6.4-11.0 s (nbformat warning); first own print 13.4-16.9 s | all notebooks | [E14N]`Research - 4xL4 engine bottleneck validation/startup_load_environment.md:209-213`; `_scratch/e23_full_sami/live_stream.txt` (first stdout t = 16.9 s) |
| SGLang 0.5.19 install (wheel) | 24-81 s | E3 FN | [E3R]`*/fn_smoke/fn_smoke_report.json` `stages.2_install.sec` = 30.4 / 24.0 / 80.7 / 58.6 |
| SGLang 0.5.21 venv install | 133.7 s | E13 SMOKE1 | [E13R]`E13_SMOKE1_rev6b_GPU_run1_2026-10-26.md:10` |
| Stock vLLM 0.32.0 venv install (wheelhouse dataset) | 99 s; 175 s | RC1 / NANPROBE | [E13R]`E13_RC1_GPU_run1_2026-10-31.md:13`; `E13_NANPROBE_rev2_GPU_run1_2026-10-30.md:63` |
| Full sha256 of the 124.8 GB FN checkpoint (E3 FP8-PLE merge) | 192.5 s (0.65 GB/s), 201.2 s (0.62), 428.3 s (0.29) | E3 FN | [E3R]`FN_SMOKE_v3_run`, `FN_v33_run`, `FN_PROBE_v4_run1` `stages.1_inputs.hash_sec` / `hash_gb_per_sec` |
| Full sha256 of the 124 GB Merlin FN checkpoint (5 Models + PLE overlay, 6 threads, file-parallel) | 275.3 s (≈ 0.45 GB/s) on a 4-core CPU kernel, concurrent with the Merlin install | RC2 CPU kit | [E13R]`E13_RC2_CPU_kit_v1_2026-10-31.md:11`; `RC2_CPU_kit_v1_2026-10-31/q27-e13-rc2-cpu-kit.log` (R1 sha `"sec": 275.3`) |
| Symlink merge of the checkpoint | 1.9 s | E3 | [E3R]`FN_PROBE_v421_run1/fn_smoke/fn_smoke_report.json` `stages.1_inputs.merge_sec` |
| Merlin offline install (CPU kit, 4 cores) | 394 s | RC2 CPU kit | [E13R]`E13_RC2_CPU_kit_v1_2026-10-31.md:13` |
| SGLang rank init ("Init parallel ends") | staggered 8.6 / 18.7 / 28.8 / 38.9 s (≈ 10 s per rank, serial) | SGLang FN | [LP]`sglang_server_e13_bf16ple.log:36-39`; [E3R]`FN_PROBE_v421_run1/fn_smoke/serve.log:64-67` |
| SGLang weight load, 0.5.19, multi-thread loader, no prefetch | first shard 461 s; both launches timed out at 1,080 s | E3 FN-PROBE v4 (10-10, before the GCSFUSE change) | [E3R]`FN_PROBE_v4_run1/fn_smoke/serve.log:96`; `fn_smoke_report.json` `stages.4_launch.attempts[*].sec` = 1080.1 |
| SGLang weight load, 0.5.19 + one-shard lookahead range prefetch (4 readers per shard) | 345 s | E3 v4.2.1 | `experiments/Experiment 3 .../README.md:15`; [E3R]`FN_PROBE_v421_run1/fn_smoke/serve.log:96-100` |
| SGLang server ready (E3) | 862.2 s | E3 v4.2.1, 0.5.19 | [E3R]`FN_PROBE_v421_run1/fn_smoke/fn_smoke_report.json` `stages.4_launch.attempts[0].sec` |
| SGLang weight load (38 shards, Intel package with BF16 PLE, 181 GB) | 540 s (9:00); 582.1 s; 792.4 s; 613-628 s (second boot, same session); 524.3 s | SMOKE1 / LOADPROBE / NANPROBE arm 1 / NANPROBE arm 2 / RC1 | [E13R]`E13_SMOKE1_...md:15`; [LP]`sglang_server_e13_bf16ple.log:86-92`; [NAN]`e13_loadprobe.json` `arms[0/1].load_weight_end_lines`; [RC1]`e13_loadprobe.json` `arms[0].load_weight_end_lines` |
| SGLang decode CUDA-graph capture (TileLang JIT) | 419.5 s | LOADPROBE | [LP]`sglang_server_e13_bf16ple.log:131-198` |
| SGLang first eager prefill (TileLang JIT, hooks on) | 428 s | NANPROBE | [E13R]`E13_NANPROBE_rev2_...md:47` |
| SGLang FN ready (total) | 1,132.1 s; 1,348.3 s (eager + hooks); 716.1 s (2nd boot, graphs on); 1,072.9 s | LOADPROBE / NANPROBE / NANPROBE / RC1 | [LP]`e13_loadprobe.json` `arms[0].sec`; [NAN]`e13_loadprobe.json` `arms[0].sec`, `arms[1].sec`; [RC1]`e13_loadprobe.json` `arms[0].sec` |
| Stock vLLM FN: "Initializing model took" | 17.5 s | RC1 vllm_fn | [RC1]`vllm_server_e13_vllm_fn.log:47` |
| Stock vLLM FN: weight load 38 shards / 168.75 GiB, direct from the Model mount (auto-prefetch skipped) | 564.3 s (9:24) ≈ 321 MB/s; first two shards 65 s and 57 s each | RC1 vllm_fn | [RC1]`vllm_server_e13_vllm_fn.log:48-49,83-86` |
| Stock vLLM FN: "Model loading took 18.43 GiB and … s" | 610.5 s (rank 1 265 s) | RC1 | [RC1]`vllm_server_e13_vllm_fn.log:64,110-112` |
| Stock vLLM FN: init engine (profile, KV, warmup), eager | 96.0 s | RC1 | [RC1]`vllm_server_e13_vllm_fn.log:143` |
| Stock vLLM FN: process start → "Application startup complete" | 00:15:57 → ≈ 00:29:04 ≈ 787 s; harness ready 825.1 s | RC1 | [RC1]`vllm_server_e13_vllm_fn.log:7,154,193`; `e13_loadprobe.json` `arms[1].sec` |
| FlashInfer sampling JIT at the first sampled request (after "ready") | crashed: `ld: cannot find -lcuda` (stream test 104.3 s, 0 tok/s) | RC1 | [RC1]`vllm_server_e13_vllm_fn.log:271-274`; `e13_loadprobe.json` `arms[1].smoke.streams4_sec` |
| Pinned-memory probe | 64 MiB in 0.03 s; 1,024 MiB in 0.51 s (≈ 2 GB/s) | E3 | [E3R]`FN_PROBE_v421_run1/fn_smoke/fn_smoke_report.json` `stages.3_pinned_memory.output` |

### 1b. Merlin v3.0.0 (27B) and stock vLLM 0.32 boots on Kaggle (the closest Merlin evidence)

| Phase | Measured | Engine / model | Evidence |
|---|---|---|---|
| Merlin payload extract (`merlin_offline_tar.bin`, 4.29 GB, Model mount) | 24.5 s | E23 SMOKE_A | [E23S]`SMOKE_A/e23_results.json` `setup.merlin_extract_sec` |
| Merlin offline install (uv CPython 3.13 + 222-wheel wheelhouse + GCC 13 + glibc fix + rehash + verify), 48-core GPU VM | 175.6 s; 175 s | SMOKE_A; FULL | [E23S]`SMOKE_A/e23_results.json` `setup.install_merlin.sec`; `_scratch/e23_full_sami/live_stream.txt` (`install_merlin rc 0 175s`) |
| Merlin env kit build (CPU, internet) | 503 s | E23 P2 | `experiments/Experiment 23 .../Experiment Results/PREP_KITS_2026-10-29.md:12` |
| Stock vLLM 0.32 install | 175.5 s; 107.5 s; 153 s | SMOKE_A; SMOKE_B; FULL | [E23S]`SMOKE_A|SMOKE_B/e23_results.json` `setup.install_stock.sec`; `live_stream.txt` |
| Prefetch, file-parallel 16 threads (one file per thread), Merlin 27B Model (16 files, 17.05 GB) | 88.1 s = **193.6 MB/s** | SMOKE_A | [E23S]`SMOKE_A/e23_results.json` `setup.prefetch_merlin`; code `experiments/Experiment 23 .../src/e23_lab.py:258-283` |
| Same prefetch, FP8 27B Model (31 files, 30.9 GB) | 53.6 s = **576.4 MB/s** | SMOKE_B | [E23S]`SMOKE_B/e23_results.json` `setup.prefetch_fp8` |
| Same prefetch, INT4 27B Model (12 files, one 18.6 GB file) | 122.3 s = **159.2 MB/s** | SMOKE_B | [E23S]`SMOKE_B/e23_results.json` `setup.prefetch_int4` |
| APIServer "Multi-modal warmup" (no `--language-model-only`) | 22.5-24.9 s | Merlin 27B | [E23S]`SMOKE_A/logs/server_merlin_tp1.log:41`, `server_merlin_tp1_rep0.log:41`, `server_merlin_tp4.log:41` |
| "Initializing model took" | 40.2 s (TP1), 34.0 s (TP4), 37.4 s (warm rep), 33.7 s (stock FP8 TP4) | Merlin / stock 27B | `server_merlin_tp1.log:44`; `server_merlin_tp4.log:68`; `server_merlin_tp1_rep0.log:44`; `SMOKE_B/logs/server_fp8_mtp3.log:65` |
| Weight load from page cache (prefetched) | 0.3-4.9 s | Merlin / stock 27B | `server_merlin_tp1.log:55`; `server_merlin_tp4.log:82`; `server_fp8_mtp3.log:113` |
| torch.compile, cold | 54.6 s (TP1 backbone) + 3.8 s (eagle head); 54.1-54.4 s (TP4); 52.0-63.8 s (stock) | Merlin / stock 27B | `server_merlin_tp1.log:85-96`; `server_merlin_tp4.log:144-159`; `SMOKE_B/logs/server_*.log` `torch.compile took` |
| torch.compile, AOT cache hit (same config, same session) | **0.75-0.92 s** | Merlin 27B TP1 replicas | `server_merlin_tp1_rep0.log:82-83` ("Directly load AOT compilation from path …") and rep1-3 |
| FlashInfer JIT (Merlin excludes `flashinfer-jit-cache`; compiled with GCC 13 on first start): batch_prefill (FP8 KV, head 256) + sampling | ≈ 48 s + ≈ 118 s ≈ **166 s**, first boot only | Merlin 27B TP1 | `server_merlin_tp1.log:113-124`; why: `experiments/Experiment 23 .../Experiment Review/audit5_rev3b/1_launch_commands.md:58-68` |
| FlashInfer JIT on later boots in the same session (other configs too) | 0 builds | Merlin TP4 and 4 replicas | `grep -c 'flashinfer.jit: Building'` = 0 in `server_merlin_tp4.log`, `server_merlin_tp1_rep*.log` |
| CUDA graph capture (vLLM) | 2-12 s per capture; stock first boot 48-74 s when FlashInfer JIT ran inside capture | Merlin / stock | `server_merlin_tp1.log:124`; `server_merlin_tp4.log:269`; `server_fp8_mtp0.log:166-170`; `server_nvfp4_mtp0.log:162-167` |
| Init engine total | cold 258.4 s (TP1, compilation 58.5 s); 96.5 s (TP4 after TP1); warm 37.4-62.7 s; stock 169.9-318.8 s | Merlin / stock | `server_merlin_tp1.log:129`; `server_merlin_tp4.log:285`; `server_merlin_tp1_rep*.log`; `SMOKE_B/logs/*` `init engine` |
| Server ready (harness) | Merlin TP1 cold **384.1 s**; TP4 (after TP1) **204.0 s**; warm replicas **135.0-162.1 s**; stock FP8 MTP0 423.1 s, MTP3 318.0 s, G3 309.0 s, INT4 ref 258.0 s | E23 SMOKE | [E23S]`SMOKE_A/e23_results.json` `ready_sec`; `SMOKE_B/e23_results.json` `servers.*.ready_sec` |
| Merlin 4×3090 reference cold start | 268 s [C] | Merlin bench card | [E13V]`MERLIN_FN_L4_SYNTHESIS_2026-10-31.md:157` |
| Triton kernels JIT-compiled during inference (jit_monitor warnings) | 14-15 per boot, also on warm boots | Merlin 27B | `grep -c 'JIT compilation during inference'` in `SMOKE_A/logs/server_merlin_*.log`; e.g. `server_merlin_tp1_rep0.log:177-180` |

### 1c. Other 27B boots (history, GCSFUSE period, caches)

| Phase | Measured | Engine / model | Evidence |
|---|---|---|---|
| SGLang 27B ready, 10-19/10-20 | 507, 522, 532, 552, 562 s | SGLang 0.5.21 INT4 + DFlash2 | `resources/Analysis and Journals/Kaggle discussions (2026-10-22)/00_SUMMARY.md` (exposure table) |
| SGLang 27B ready, 10-21 (GCSFUSE day) | **1,197 s** (load_weight 669.96 s, 18.6 GB shard at ≈ 28 MB/s) | E12 rev 6d G4B | [E14N]`Research - 4xL4 .../startup_load_environment.md:30-33`; `experiments/Experiment 12 .../Experiment Results/E12 rev 6d G4B GPU run analysis (2026-10-21).md:89` |
| SGLang 27B weights load from warm page cache | 12.8-58.7 s | E12 10-19/20 | `startup_load_environment.md:31` |
| SGLang target-verify graph capture, first launch (compile) | 371-385 s; warm relaunch same session 3.5-14 s | E11/E12 | `startup_load_environment.md:155-157` |
| SGLang first launch vs relaunch | 601 s vs 136-166 s; 567 s vs ≈ 75 s; 532 / 522 s | E11-03 / E11-04 / E11-05 | `experiments/Experiment 11 .../Experiment Results/E11-03 run1 analysis (2026-10-17).md:73`; `E11-04 run1 analysis (2026-10-18).md:3`; `E11-05 run1 analysis (2026-10-18).md:12` |
| SGLang launch from a /dev/shm copy vs /kaggle/input | 502 s vs 507-562 s (weights phase ended at 65 s; ≈ 7 min was compile/JIT/graphs); relaunch 71 s | E14A SLIM run1 | [E14N]`22 Claude - E14A SLIM run1 findings ... (2026-10-23).md:57-64` |
| Range-parallel copy (64 MiB ranges, 4→24 workers) from Model mounts to /dev/shm | 518 / 506 MB/s (E14A); 543 / 480 (R2); 598.5 / 600.6 (R3K1); 678.5 / 549.2 (R3K2); 492 (LOW); ramp 4 → 216-263, 8 → 562-786, 16 → **786-870**, 24 → 786-859 MB/s | E14/E15 load path | [E14N]`22 ...md:40-44`; [E15K]`E15B EVAL120 50K45K R3K1 .../e14_results.json` and `R3K2 .../e14_results.json` `staging.copies.*.mb_s`, `ramp.levels`; [E14N]`EVAL120 analysis 2026-10-24/C_engine.md:47` |
| Single-stream 64 MiB read probe | 34.4, 34.6, 35.8, 109.9 MB/s | E14/E15 | [E15K]`*/e14_results.json` `read_probe.mb_s`; [E14N]`22 ...md:44` |
| Copy + verify re-read total (INT4 19.47 GB) | 93.5-103.2 s, hidden under the 140-178 s venv install | E14/E15 | same; `C_engine.md:47` |
| Startup before the first Duck token (27B SGLang) | ≈ 12-13 min, of which 6.1-6.3 min first verify-graph capture | E14 EVAL120 | [E14N]`EVAL120 analysis 2026-10-24/C_engine.md:47` |
| vLLM 0.32 27B launch → ready | 550 s; solving starts 14.8 min after kernel start; S1 on 0.19 ready at 668-733 s | E18 | `experiments/Experiment 18 .../Experiment Results/E18_SMOKE_GPU_v1_2026-10-27.md:12-15` |
| vLLM 0.19 / 0.32 27B ready (Duck B) | 0.19: INT4 365 s, FP8 250 s, 435 s; 0.32: 535 s, 560 s | E21 smoke | `experiments/Experiment 21 .../Experiment Results/E21_SMOKE_GPU_run1_2026-10-29.md:10` |
| HF → Kaggle CPU kernel download | 339 MB/s (`model.safetensors`, xet) | E23 P1 | `PREP_KITS_2026-10-29.md:11` |
| Streaming uploader HF → Kaggle blob | 32-34 MB/s single file | E11-01 | `experiments/Experiment 11 .../README.md:56`; [E13V]`MERLIN_FN_L4_SYNTHESIS_2026-10-31.md:171` |
| Kaggle Model creation from a 17 GB kernel output | ≈ 6 min on a fresh slug; 30+ min when stacked on an unfinished version | E23 Models | [MEM]`kaggle-model-upload-rules.md:25` |
| Other teams: NFS vs GCSFUSE load | Jeroen: 100.5 GiB in 88-113 s on NFS; 3 of 10 shards in 7 h 22 min on GCSFUSE. NguyenThanhNhan: ≈ 180 GB at ≈ 300 MB/s ≈ 10 min; save runs 875-1,300 s [C] | Kaggle topic 751344 | `startup_load_environment.md:15-29` |

---

## 2. Memory / disk / limits table

| Item | Value | Evidence |
|---|---|---|
| GPUs | 4 × L4, 23,034 MiB (nvidia-smi), sm_89, driver 580.178.04, CUDA 13.0; CUDA total 22.03 GiB per card | [E3R]`FN_PROBE_v421_run1/fn_smoke/fn_smoke_report.json` `stages.0_environment`; [RC1]`vllm_server_e13_vllm_fn.log:136` |
| Free VRAM before load (SGLang view) | 21.62 GB per rank | [LP]`sglang_server_e13_bf16ple.log:59-62` |
| Host RAM (MemTotal) | 197.9 GB (E14 env) / 202.6 GB (E3 psutil); MemAvailable 197.4-199.3 GB at start | [E13V]`MERLIN_FN_L4_SYNTHESIS_2026-10-31.md:139`; [E3R] `ram_total_gb`; [RC1]`e13_loadprobe.json` `arms[0].host_ram` |
| **cgroup memory limit** | **185.8 GB = 177,152 MiB**; usage at start 0.4 GB. Page cache and /dev/shm count against it | [E15K]`*/e14_results.json` `env.cgroup_gb`; [RC1]`vllm_server_e13_vllm_fn.log:67,121` |
| What the cgroup did in RC1 | after the SGLang arm the cgroup sat at 177,152 / 177,152 MiB (page cache); vLLM saw "Available RAM 0.24 GiB" and skipped its auto-prefetch; our guard then skipped arm 3 (cgroup free 8.9 GB, MemAvailable 55 GB) | [RC1]`vllm_server_e13_vllm_fn.log:48-49,67`; [E13R]`E13_RC1_GPU_run1_2026-10-31.md:59` |
| vLLM's cgroup-aware "available RAM" on 27B boots | 92.04 GiB; 99.13 GiB; 78 GiB with 4 replicas | [E13V]`MERLIN_FN_L4_SYNTHESIS_2026-10-31.md:143` |
| /dev/shm | 91.8 GB on the GPU VM (14 GB on the CPU kit VM); tmpfs pages count against the cgroup | [E15K]`*/e14_results.json` `env.df./dev/shm`; [E13R]`RC2_CPU_kit_v1_2026-10-31/q27-e13-rc2-cpu-kit.log` (A0 host) |
| /tmp | 8,758 GB total, 1,122.9-1,148.8 GB free (local, ext4/overlay) | [E15K] `env.df./tmp`; [E3R] `tmp_free_gb` |
| /kaggle/working | 20-21 GB (`/dev/loop2`); everything left there is saved as output | [E15K] `env.df./kaggle/working`; [E3R] `disk_working` |
| CPU | 48 vCPU (GPU VM); 4 cores, 31 GB RAM (CPU kit VM) | [E15K] `env.nproc`; `PREP_KITS_2026-10-29.md:7` |
| memlock (ulimit -l) | 8,388,608 KB soft = hard, not raisable; pinning still worked: SGLang pinned 23.84 GiB per rank (≈ 95 GiB) of BF16 PLE | [E3R] `memlock`; [E13R]`E13_NANPROBE_rev2_...md:36` |
| Mount types | `/kaggle/input/models/...` = **nfs**; `/kaggle/input/datasets/...` (wheelhouse) = **fuse.gcsfuse**; `/kaggle/input` root = ext4. RC1's vLLM logged "Filesystem type for checkpoints: NFS" for the dfranzen Model | [E15K]`*/e14_results.json` `env.paths.*.fstype`; [RC1]`vllm_server_e13_vllm_fn.log:48` |
| Model vs notebook output | a checkpoint attached as a Kaggle Model loads about 15× faster than the same files attached as a notebook output (user observation, fact) | [MEM]`kaggle-model-upload-rules.md:31` |
| Flash-Next checkpoint sizes | Merlin parts p1-p5: 14.61 + 15.54 + 15.52 + 15.44 + 11.45 = 72.56 GB (18/5/5/5/5 files); FP8 PLE overlay 51.24 GB (10 shards); Merlin env 4.29 GB; Intel fallback 181.24 GB (50 files) | [E13V]`RC2_BUILD_2026-10-31.md` (Kaggle inputs table) |
| FN host memory needs | Merlin: one shared pinned FP8 PLE table ≈ 47.7 GiB, 69 GiB measured floor, 96 GB "qualified"; SGLang: 23.84 GiB (BF16) or 11.92 GiB (FP8) per rank | [E13V]`MERLIN_FN_L4_SYNTHESIS_2026-10-31.md:137`; [E13R]`E13_LOADPROBE_GPU_run1_2026-10-27.md:31` |
| FN VRAM after weights | SGLang 19.10 GB used / 2.51 GB free per rank; stock vLLM TP4 18.43 GiB weights, KV 0.57 GiB = 30,208 tokens | [LP]`sglang_server_e13_bf16ple.log:86`; [RC1]`vllm_server_e13_vllm_fn.log:110-128` |
| Merlin KV pin | `KV_BYTES` FP8 2,479,838,700 B (150 blocks), BF16 2,464,415,744 B; RC2 retries −0.25 GiB on boot OOM | [E13V]`RC2_BUILD_2026-10-31.md`; `build_e13_rc2.py:391,595` |
| Notebook size (COMMIT runs) | keep < 400 kB; 407 kB ran, 444 kB ERROR after 114 min with empty log (quota used), 745 kB `ERRORED_MOUNTING_DATASET`; RC1 was 398 kB, RC2 127 kB | [MEM]`kaggle-arc2-platform-limits.md` (size section); [E13V]`RC2_REVIEW_PAST_EXPERIMENTS_2026-10-31.md` D1 |
| Model upload rules | ≤ 50 files; Kaggle unpacks .zip/.tar/.tar.gz and rejects inner names with `[ ] = , & @ +`; wrap archives as `.bin`; no stacked versions while a save runs | [MEM]`kaggle-model-upload-rules.md` |
| Kaggle Model version from a batch kernel | not possible without secrets; secrets only in editor runs | `experiments/Experiment 23 .../prep/P1_merlin_checkpoint/README.md:114-121` |
| Quota | 4×L4 billed at 2×; measured ≈ 1.05 quota-h per wall-h on 10-13 (verify); only container running time is billed, data attached after start is billed; runs with low quota still finish | [MEM]`kaggle-arc2-platform-limits.md`; `startup_load_environment.md:276-295`; [MEM]`kaggle-quota-midrun.md` |
| Submission limits | 12-16 h turnaround; 1 pending per team; about 1 per 2 days; hidden rerun = 240 tasks, pinned image; hidden clock starts at kernel start | [MEM]`kaggle-arc2-platform-limits.md` |
| Run budget | full runs ≈ 11.5 h of the 12 h limit; hard stop ≈ 11:55 | [MEM]`thinking-120k-full-run-12h.md`; `kaggle-arc2-platform-limits.md` (time use) |
| Launch timeouts in our code | SGLang 30 min first, 20 min relaunch, give-up min(T0 + 75 min, admit − 5 min); E21 ready 1,800 s per launch, install 2,400 s; RC2 Merlin 45 min cold, `VLLM_PLE_OFFLOAD_READY_TIMEOUT` 3600 (Merlin default 1800) | `startup_load_environment.md:309-315`; `experiments/Experiment 21 .../Experiment Review/VAL_timing_C1.md:49,54`; `build_e13_rc2.py:427-428,936` |
| Sandbox | no unshare/mount; root, CAP_SETUID; subprocess + rlimits + seccomp | [MEM]`kaggle-arc2-platform-limits.md` |
| Image | pinned Python 3.12 / glibc 2.35 / GCC 11.4 only; system CUDA 12.8 with `lib64/stubs/libcuda.so`; only `libcuda.so.1` at runtime (no linkable `libcuda.so`) | `PREP_KITS_2026-10-29.md:3-6`; `experiments/Experiment 1 .../Experiment Results/Kaggle Runs/Q27X_run1/RUN1_ANALYSIS.md:23` |

---

## 3. GOOD (tricks that saved time or memory)

1. **Big checkpoints as Kaggle Models, not datasets or notebook outputs.** Models mount as NFS, datasets as fuse.gcsfuse ([E15K] `env.paths.*.fstype`). User: about 15× faster than notebook outputs ([MEM]`kaggle-model-upload-rules.md:31`). All five Merlin FN parts, the PLE overlay and the Merlin env are already Models.
2. **Range-parallel reads** (64 MiB ranges, own fd per range, 8-16 workers): 543-870 MB/s from Model mounts versus 34-110 MB/s for one stream and 159-194 MB/s for file-parallel readers dominated by one big file ([E15K] staging ramps; [E23S] prefetch). Throughput plateaus at 16 workers.
3. **Overlap IO with the install.** The E14 staging copy (93-103 s) ran entirely under the 140-178 s venv install ([E14N]`EVAL120 analysis 2026-10-24/C_engine.md:47`). The RC2 sha256 thread runs during the Merlin install (`build_e13_rc2.py:327-375`).
4. **Prefetch before launch removes the load phase.** Prefetched 27B weights loaded in 0.3-4.9 s ([E23S] `Loading weights took`). For SGLang 0.5.19 FN, a one-shard lookahead range prefetch cut the load from > 1,080 s (timeout) to 345 s (E3 README:13-15).
5. **In-session compile reuse.** vLLM AOT compile cache: 54.6 s → 0.75 s on the same config ([E23S]`server_merlin_tp1_rep0.log:82-83`). Merlin's FlashInfer JIT (≈ 166 s) ran once and was reused by every later boot, even with a different TP layout (0 builds in the TP4 and replica logs). Warm Merlin replica boots: 135-162 s versus 384 s cold. SGLang relaunch: 71-75 s versus 500-600 s cold ([E14N]`22 ...md:62`; E11-04).
6. **Shared cache roots per session.** RC2 sets one `VLLM_CACHE_ROOT`, `TRITON_CACHE_DIR` and `FLASHINFER_WORKSPACE_BASE` under /tmp for every boot (`build_e13_rc2.py:427-428`), fixing E23's per-port Triton cache that defeated reuse ([E13V]`MERLIN_FN_L4_SYNTHESIS_2026-10-31.md:164`).
7. **The -lcuda shim** (`libcuda.so` → `/usr/local/nvidia/lib64/libcuda.so.1` on `LIBRARY_PATH`) plus a link probe. It fixed the E1 run-1 crash and is in E11/E18/E23/RC2 (`experiments/Experiment 11 .../Experiment Results/Kaggle Runs/E11-03 rev3b run1 (foysal, 2026-10-17)/e11_03/report.md:3`). Alternative: `VLLM_USE_FLASHINFER_SAMPLER=0` [C] ([E14N]`Speed review 2026-10-24/part_B_fastest_serving_research.md:735`).
8. **Offline Merlin install from one `.bin` Model payload** (glibc 2.35 weak-flag fix + `__isoc23_strto*` shim, GCC 13, uv Python 3.13): 24.5 s extract + 175 s install on the GPU VM, reproducible 3×; no internet ([E23S] `setup.*`; `PREP_KITS_2026-10-29.md:12,31-46`).
9. **gc.collect() hook for the Marlin repack delayed free.** It turned a non-deterministic +0.30 GiB per MoE layer OOM into flat memory ([E13R]`E13_LOADPROBE_GPU_run1_2026-10-27.md:20-41`).
10. **Host-RAM guard that counts page cache as reclaimable** (`memory.max − (memory.current − inactive_file)`), in RC2 after RC1 falsely skipped arm 3 ([E13V]`RC2_REVIEW_PAST_EXPERIMENTS_2026-10-31.md` A4).
11. **`--language-model-only`** removes the vision tower and the APIServer "Multi-modal warmup" (22.5-24.9 s on Merlin 27B boots without it; [E23S]`server_merlin_tp1.log:41`). E18's VMS boot logged a smaller KV pool without it (1,006,517 vs 1,110,415).
12. **Small notebooks, payloads in Models/datasets by sha** (RC2 127 kB; the 400 kB commit limit).
13. **Quick-save pushes** (`kernelExecutionType QUICK_SAVE`) avoid an unwanted commit run and double quota ([MEM]`kaggle-quick-save-push.md`).
14. **Stall watchdog instead of a fixed timeout** (E14 load path: rchar and log-growth progress; E21 rev 2h watchdog never fired falsely) (`startup_load_environment.md:303-351`; `E21_SMOKE_GPU_run1_2026-10-29.md` finding 2).

## 4. BAD (what cost time or failed)

1. **Fixed launch timeouts on slow IO = 0 LB.** E3 FN-PROBE v4: two launches timed out at 1,080 s each while loading weights ([E3R] `stages.4_launch`). SGLang 27B ready went from ≈ 530 s to 1,197 s on 10-21 (GCSFUSE/NFS slowdown). Other teams scored 0.00 when their server missed a 45-min timeout (`00_SUMMARY.md` §1).
2. **Single-stream and per-file loaders.** One `cat` reads ≈ 10.5-35 MB/s; per-file parallelism caps at the biggest file (E23 Merlin 27B prefetch 193.6 MB/s, INT4 159.2 MB/s). SGLang's 4-rank multi-thread loader without prefetch thrashed (461 s for the first FN shard).
3. **Flash-Next loads are 9-13 min even on a good day.** SGLang FN load 524-792 s; stock vLLM FN 564 s for 168.75 GiB read directly from the Model (≈ 321 MB/s). The second boot in the same session was only about 20 % faster (613-628 s vs 792 s), because the checkpoint plus pinned PLE does not fit in the 185.8 GB cgroup as page cache ([NAN] `load_weight_end_lines`).
4. **SGLang FN compile/JIT is huge:** TileLang decode-graph capture 419.5 s, first eager prefill 428 s. SGLang FN ready 1,072-1,348 s. With Merlin/vLLM, graph capture is 2-12 s.
5. **Merlin FlashInfer JIT ≈ 166 s on every fresh session.** Merlin excludes `flashinfer-jit-cache` on purpose and JIT-compiles from its patched headers with GCC 13 (`audit5_rev3b/1_launch_commands.md:58-68`). The sampling JIT alone is ≈ 100-118 s.
6. **JIT on the first sampled request, after "ready".** RC1's first temperature-0.7 request triggered the FlashInfer sampling JIT, which failed on `-lcuda` and killed the engine ([RC1]`vllm_server_e13_vllm_fn.log:271-274`). Even when it links, ≈ 100 s of JIT lands inside the solve window unless the boot warms it.
7. **Page cache fills the cgroup.** After a 181 GB load, the cgroup sat at its 177,152 MiB limit; vLLM's auto-prefetch disabled itself ("Available RAM 0.24 GiB"), and a guard that treated page cache as used skipped a whole arm.
8. **Delayed free OOM in the Marlin MoE repack** (SGLang, 20.86 GiB at layer 7-8); non-deterministic across launches.
9. **RC1's stock vLLM boot hit `expandable_segments … OOM` mapping warnings on all ranks during load** ([RC1]`vllm_server_e13_vllm_fn.log:56-62`); Merlin leaves < 1 GiB slack, so a GPU not fully released by the previous boot (561/515 MiB left after a dead server) can OOM the next boot ([E13V]`RC2_REVIEW_PAST_EXPERIMENTS_2026-10-31.md` A3, A5).
10. **Leaked /dev/shm segments** after SIGKILL (RC1 "1 leaked shared_memory objects"; the PLE transport unlinks its /dev/shm storage only on a graceful teardown), and /dev/shm counts against the cgroup (A6).
11. **The shipped JIT cache was never made to work.** E14's JIT cache tar failed with `IndexError('list index out of range')`, no cache was saved, and cross-session reuse is still untested ([E14N]`30 Claude - E14A SLIM R2 run results (2026-10-24).md:81`; `Speed review 2026-10-24/part_A2_...md:1158`).
12. **Full sha256 as a serial gate.** RC2's first launch waits for the full 124 GB sha256 (275 s on the CPU kit; E3 192-428 s). Merlin never hashes shards itself (`build_e13_rc2.py:381-387`).
13. **Install times vary and add up.** Stock vLLM 0.32 99-175 s from a GCSFUSE-mounted wheelhouse dataset; Merlin 175 s (GPU) / 394 s (CPU kit); SGLang 0.5.21 134-208 s. Every notebook pays this before any GPU work.
14. **Notebook size and mount errors that leave an empty log:** 444 kB commit ERROR after 114 min (quota used), 745 kB `ERRORED_MOUNTING_DATASET` ([MEM]`kaggle-arc2-platform-limits.md`).
15. **Live Triton autotune on sm_89.** Merlin's persisted Triton autotune is sm_86-only (WARM=0), so FN GDN kernels autotune on the first boot and may pick slow configs; 14-15 Triton "JIT compilation during inference" warnings appear per Merlin boot, even warm ones ([E13V]`MERLIN_FN_L4_SYNTHESIS_2026-10-31.md:158,185`).
16. **Stacked Model versions:** a v4 pushed while v3's 17 GB output was saving took 30+ min to save ([MEM]`kaggle-model-upload-rules.md:25`).
17. **Wheels or venvs under /kaggle/working** are committed as output (7 GiB) and errored at teardown for another team [C] (`part_B_fastest_serving_research.md:735`).

---

## 5. Boot-time budget for one Flash-Next Merlin cold boot, and ideas to cut it

### 5a. Budget (projection; RC2 will replace the guesses)

Assumptions: RC2's inputs (5 Merlin Models 72.56 GB + FP8 PLE overlay 51.24 GB = 123.8 GB on NFS; `merlin_offline_tar.bin` Model), profile `fn-quad-24g-sm89` (TP2×PP2+EP, MTP K3, FP8 KV, capture 1..32), cold caches, `--language-model-only`.

| # | Phase | Basis | Cold estimate | Serial? |
|---|---|---|---|---|
| 0 | Kernel start → first cell | measured 6-17 s | 15 s | yes |
| 1 | Merlin extract + install | measured 24.5 + 175.6 s | 200 s | parallel with 2 |
| 2 | Checkpoint read (prefetch or sha256) 123.8 GB | 860 MB/s (best range-parallel) → 145 s; 450 MB/s (RC2 CPU-kit sha) → 275 s; 321 MB/s (vLLM direct) → 385 s; 194 MB/s (E23 file-parallel) → 640 s | 145-640 s, **275 s planning value** | parallel with 1; RC2 gates the boot on it |
| 3 | APIServer + EngineCore start | 27B: 18-53 s (incl. 22 s multimodal warmup when not text-only) | 30 s | yes |
| 4 | Initialize model (allocation, PLE meta) | FN stock 17.5 s; Merlin 27B 34-40 s | 30 s | yes |
| 5 | Weight load to GPU | page-cached: seconds; partly evicted: up to the 564 s RC1 value. The cgroup (185.8 GB) cannot hold 124 GB of cache plus the 47.7 GiB pinned PLE plus worker RSS at once | 60-300 s | yes |
| 6 | PLE pin + offload child ready | pin rate ≈ 2 GB/s (E3 probe) → ≈ 25 s for 51 GB, plus the read if evicted | 30-120 s (UNVERIFIED) | yes |
| 7 | torch.compile (2 PP stages + MTP head) | 27B: 54-64 s per config | 60-120 s | yes |
| 8 | FlashInfer JIT (prefill/sampling with GCC 13) | Merlin 27B ≈ 166 s; FN uses Triton QSA + FA2 for attention, so maybe only the sampler (≈ 100-118 s) | 100-170 s | yes |
| 9 | Triton GDN/QSA kernels + live autotune (WARM=0) | unmeasured on FN | 30-180 s (UNVERIFIED) | yes |
| 10 | KV profile + CUDA graph capture (PIECEWISE under spec decode, 1..32) | 27B 2-12 s per capture | 20-60 s | yes |
| 11 | First sampled request warm-up (sampler JIT if not done in 8) | RC1: the JIT ran here | 0-120 s | yes |
| | **Total cold, kernel start → first useful token** | 15 + max(200, 275) + (30+30+60+30+60+100+30+20) + slack | **≈ 11-16 min** (synthesis guess 15-25 min) | |
| | **Same session, second boot (other config)** | no install; FlashInfer JIT reused; torch.compile re-runs per config hash; weights partly re-read | ≈ 6-10 min (NANPROBE SGLang FN 2nd boot was 716 s) | |
| | **Warm caches shipped (target)** | 15 + 275 (read, overlapped with install) + ≈ 150-200 s (27B warm replica floor 135-162 s + PLE) | **≈ 7-8 min** | |

At 1500 tok/s every minute of boot is ≈ 90,000 tokens, about 1.5 attempts at the 60k floor; each cold boot also costs 2× in quota minutes.

### 5b. Ideas to cut the boot (each needs a measurement; ranked by expected seconds saved)

1. **Compile/JIT cache Model (`.bin`), keyed and gated.** After the first good boot of the production config, tar `VLLM_CACHE_ROOT` (torch_compile_cache incl. `torch_aot_compile`), `FLASHINFER_WORKSPACE_BASE` (Merlin's GCC 13 FlashInfer JIT `.so`) and `TRITON_CACHE_DIR` (incl. autotune results). **Do not ship `CUDA_CACHE_PATH`/`nv/`** (SGLang #42917 stale-cache corruption, `startup_load_environment.md:167-168,183`). Write it to /kaggle/working (< 20 GB, a few files, safe names), wrap as `.bin`, make a Model on a fresh slug. At boot, untar to the same /tmp paths before launch. Key: Merlin commit + payload sha d854bc14 + profile + kv dtype + TP/PP + max-model-len + capture sizes + driver + sm_89. Gate: a greedy known-answer request (token ids + logprobs) vs the reference from the cold boot that built the cache; on mismatch wipe and boot cold. Expected saving 3-6 min per boot (FlashInfer ≈ 166 s + compile 60-120 s + autotune). This replaces Merlin's WARM=1 (sm_86-only) with our own sm_89 autotune results. The E14 attempt failed on an `IndexError` in the tar step, so the tar code needs a CPU-kit test first.
2. **One read, not two, and no serial sha gate.** Make the integrity check the prefetch: range-parallel (64 MiB, 16 workers, E14 `e14_loadpath` design) sha256/blake2b over the Model mounts starting at t = 0, during the install. Boot as soon as the install is done; abort the boot only if a hash mismatches. Or replace the full sha with size + safetensors header checks + a few range hashes (E14 method) after RC2 has verified the bytes once. Saves up to ≈ 75-275 s of serial wait.
3. **Prefetch order matched to the cgroup.** 123.8 GB of cache + 47.7 GiB pinned PLE + worker RSS ≈ the 185.8 GB limit. Read the 72.6 GB non-PLE parts first (in loader order, one shard ahead, as in Franzen's lookahead that cut E3 to 345 s), and the PLE shards last; after the PLE table is pinned, drop their pages (`posix_fadvise(DONTNEED)`). Never stage to /dev/shm (91.8 GB cap, counts against the cgroup; [E13V]`MERLIN_FN_L4_SYNTHESIS_2026-10-31.md:146`).
4. **Pre-built Merlin venv `.bin`** (the installed `/tmp/merlin_off` tree with the glibc fix and rehash already applied, same absolute path) instead of uv + wheelhouse at runtime: 175 s install → maybe 30-60 s extract (UNVERIFIED; venv relocation and `ENVIRONMENT VERIFIED` must still pass).
5. **Move every remaining dataset input into Models.** The vLLM 0.32 wheelhouse and the payload datasets mount as fuse.gcsfuse; Models mount as NFS ([E15K] fstype).
6. **Warm the sampler inside the boot.** One sampled request (T 0.6, top_k 20) before declaring ready, as RC2 A7 does, so the FlashInfer sampling JIT never lands in the solve window, and fails fast if `-lcuda` is missing.
7. **`--language-model-only`** on every FN boot (vision tower not loaded; no 22-25 s multimodal warmup).
8. **One boot per notebook in production;** every arm that needs a second boot costs ≥ 6-10 min. When a second boot is unavoidable: graceful SIGTERM (≥ 75 s), /dev/shm cleanup, GPU-release gate < 150 MiB, keep the same cache roots.
9. **Progress-based ready watchdog** (log growth, `/proc/<pid>/io` rchar, stage markers) inside a global give-up bound, never a fixed 30-min timeout; soft-release the solver if the server is still progressing.
10. **Keep the gc.collect() repack hook and the boot-OOM retry (−0.25 GiB KV_BYTES)** so a boot is not lost to a non-deterministic OOM.
11. **Log every boot phase** (RC2 B3 list: extract, install, prefetch MB/s, hash, "Loading weights took", "Model loading took", "torch.compile took", FlashInfer "Building JIT module", "Graph capturing finished", "init engine … took", PLE ready, "GPU KV cache size"), so the next notebook's budget is measured, not guessed.

---

## 6. Open questions

1. **Merlin FN cold-boot breakdown on Kaggle** (read, load, PLE pin, torch.compile for PP2, FlashInfer JIT, Triton autotune, graph capture): RC2 will give it; nothing measured yet.
2. Does FN on Merlin use FlashInfer JIT beyond the sampler (QSA is Triton + FA2)? How long is the live Triton autotune of the GDN/QSA kernels on sm_89 with WARM=0?
3. Are vLLM AOT / torch.compile, FlashInfer JIT and Triton caches valid **across sessions and machines** (other driver build, hidden-rerun pool)? Untested; E14's tar never worked. Is the FP8-KV config's compile hash different from BF16 (separate caches needed)?
4. Is the 15× "Model vs notebook output" gap explained by NFS vs GCSFUSE? Which mount type do notebook outputs (kernel_sources) get? Do the hidden-rerun machines mount Models as NFS too?
5. Range-parallel read throughput on the five FN Models at once: does it scale past 860 MB/s with more files and workers (all earlier measurements were on 19-31 GB 27B Models)?
6. Can 123.8 GB of page cache + 47.7 GiB pinned PLE + 4-6 workers fit under the 185.8 GB cgroup without eviction during load, or does the loader re-read from NFS? What does Merlin's shared PLE table cost in RSS per worker?
7. Does `memlock` 8 GiB matter for Merlin's PLE (cudaHostAlloc/UVA vs `mlock`)? SGLang pinned ≈ 95 GiB under the same limit, so probably not (UNVERIFIED for Merlin).
8. Is a pre-installed Merlin venv relocatable as a `.bin` Model and does `install-env.sh --verify` pass on it?
9. Do the 14-15 Triton "JIT compilation during inference" warnings per Merlin boot still occur with a warm Triton cache (latency spikes in the first minutes)?
10. Is the hidden-rerun 12 h clock counted from container start or from Jupyter start, and does Kaggle cancel an idle-looking session during a long silent compile (Franzen's heartbeat comment)?
11. What is Kaggle's processing time for a Model of a multi-GB cache tarball, and can several notebooks on other accounts attach it (sharing / visibility)?
12. How large is the FN compile/JIT cache (must fit /kaggle/working 20 GB alongside other outputs)?
