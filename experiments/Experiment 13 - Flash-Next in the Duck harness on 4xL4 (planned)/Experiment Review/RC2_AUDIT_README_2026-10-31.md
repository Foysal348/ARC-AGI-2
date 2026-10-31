# E13 RC2 pre-GPU audit brief (2026-10-31)

**Who this is for:** an independent auditor (GPT) reviewing the RC2 notebook before its first GPU run.
**Goal:** find every defect that would crash the run, waste the GPU time, or produce a wrong verdict.
**How to answer:** give a verdict (GO / NO-GO), then findings ranked HIGH / MEDIUM / LOW. For each finding give the file and line, the failure scenario and the fix. Say "verified" only for what you checked in the attached source. Mark guesses as GUESS.

## What RC2 is

**Model and engine:** Qwen3.8-Flash-Next (qwen4_exp, 125B-A5B MoE, GDN + QSA, PLE) on Kaggle 4×L4 (sm_89, 23,034 MiB per card, PCIe, no NVLink, about 173 GiB host cgroup). It is served by **Merlin v3.0.0**, a vLLM 0.32.0 fork.

**Checkpoint and profile:**
- Checkpoint: `halt95/Qwen3.8-Flash-Next-W4A16-Merlin` rev cec53e9f (INT4 experts, INT8 GDN, FP8 PLE in pinned host RAM).
- Profile: `fn-quad-24g-sm89`: TP2×PP2+EP, MTP K3, prefix caching, mamba align, block 4096, chunk 1024.
- KV pins lowered for the L4: FP8 2,479,838,700 bytes, BF16 2,464,415,744 bytes.

**Purpose:** a correctness-first diagnostic. It is not a solver.

**Run:** it will run as a commit on the **foysalemonshanto** account (push folder `push/kernel-metadata.json`).

**Why Merlin:** RC1 (`E13_RC1_GPU_run1_2026-10-31.md`) showed that stock vLLM 0.32 + Intel W4A16 is correct on 4×L4. SGLang 0.5.21's EP4 GPTQ-Marlin MoE path is wrong from layer 1.

## Files in this bundle

| Path | What |
|---|---|
| `notebook/Q27-XHIGH-E13_FLASHNEXT_RC2.ipynb` | The notebook, sha256 2fb12343…, 127,271 bytes, 7 cells |
| `notebook/rc2_code_export.py` | The same cells as one .py file (easier to read; line numbers for findings) |
| `notebook/build_e13_rc2.py` | The builder: it cuts the E23 pieces verbatim and substitutes the lab module |
| `push/kernel-metadata.json` | Inputs: 5 checkpoint Models, PLE overlay, Merlin env, Intel fallback, wheelhouse, competition |
| `push/model_files_api_2026-10-31.txt` | Every file and size of every Model, listed through the Kaggle API |
| `e23/e23_lab.py` | The embedded lab module (Server, Stream, greedy, compare, teacher_forced, steady, …), Kaggle-proven in E23 |
| `merlin/launch.py`, `merlin/registry.py`, `merlin/flash-next.json` | The Merlin launcher, registry and Flash-Next profiles that `ML.build()` uses |
| `docs/MERLIN_FN_L4_SYNTHESIS_2026-10-31.md` | Design: risks R1-R24, capacity, checkpoint plan |
| `docs/E13_RC1_GPU_run1_2026-10-31.md` | RC1 GPU result |
| `docs/RC2_BUILD_2026-10-31.md` | Arms, budget, which review item each design choice answers |
| `docs/RC2_REVIEW_PAST_EXPERIMENTS_2026-10-31.md` | Our internal review of all past experiments (15 MUST / 14 SHOULD), the input to RC2 |
| `local_run/` | The local run (Windows, E23 fake vLLM server): log, `rc2_results.json`, and a `requests.jsonl` excerpt |

## Inputs at the exact mount paths (no recursive search)

| Mount path under `/kaggle/input/` | What |
|---|---|
| `models/foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p1/transformers/default/1` | 11 small files, shards 00001/00003/00004, `mtp-routed-experts-int4`, `model_extra_tensors`, manifest |
| `…-p2` / `…-p3` / `…-p4` (same pattern) | shards 00005-00007 / 00008-00010 / 00011-00013, plus manifest |
| `models/foysalemonshanto/qwen3-8-flash-next-w4a16-merlin-p5/transformers/default/1` | shards 00014, 00015, 00017, plus manifest |
| `models/foysalemonshanto/qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1` | the 10 `model-plefp8-0000N` shards (sizes equal Merlin's `checkpoint.sha256`) |
| `models/foysalemonshanto/e23-merlin-env/other/v1/1/merlin_offline_tar.bin` | Merlin offline install (E23 P2, includes the glibc 2.35 workaround) |
| `models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1` | Stock fallback only |
| `datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse/install_wheelhouse.py` | Stock fallback only |
| `competitions/arc-prize-2026-arc-agi-2/` | ARC json |

The index needs 27 files: 17 from the parts and 10 PLE. Shards 00002 and 00016 are not referenced by the index.

## What to check (please answer every point)

1. **Merlin build.** Is `merlin_fn_server()` correct against `launch.py` `build()` / `_build_flash_next()` / `host_phase()`?
   - `kv="bf16"|"fp8"` (`KV_TAG`), and `checkpoint="flash-next"`.
   - The env knobs KV_BYTES, P2P=off, VENV, TREE, CHAT_TEMPLATE="" (the checkpoint's own template), WARM=0, CACHE_ROOT, and SCALES (FP8 only).
   - The overrides re-applied after `build()`: LD_PRELOAD shim, CUDA_HOME, LIBRARY_PATH shim, `VLLM_PLE_OFFLOAD_READY_TIMEOUT` 3600, caches, GCC 13, distributed ports.
   - The trailing extra args `--served-model-name q27 --max-model-len 131072 --port N --default-chat-template-kwargs xhigh`. Does the last flag win?
   - Will Merlin accept `--max-model-len 131072` with its own `262144` earlier in argv?
2. **Checkpoint merge.** Symlinks into `/tmp/fn_merlin`, sizes checked at once, full sha256 in a background thread, gated before the first boot.
   - Is the `config.json` `text_config.ple_embedding_dtype` check right?
   - Can a symlinked checkpoint break Merlin or vLLM (realpath, `host_phase` marker, PLE mmap)?
3. **Memory.** The KV pins on a 22.03 GiB CUDA-visible card. The boot-OOM retry (−0.25 GiB once). The boot gate (< 150 MiB per GPU). The graceful kill (SIGTERM, 75 s, then E23's kill). `/dev/shm` cleanup. Host-RAM telemetry (avail = max − (current − inactive_file)). Two Merlin boots in one 12 h container with about 48 GiB of PLE host memory each: will the second boot have the RAM?
4. **Order and gates.**
   - A1 BF16 runs: probes → sampled check → prefill probe → greedy oracle → teacher-forced 40k (reference + A/A) LAST.
   - A3 FP8 runs only if A1 is DONE/TF_OOM and CLEAN. Its order: probes → sampled check → greedy oracle vs BF16 (gate) → ladder 40k@1/2/4/5 + 8k@8 → census 90k@3/4 and 116k@3/4 → long rows (natural xhigh ARC ≤ 120k new, forced 60k) → TF vs BF16 last.
   - The BF16-load arm runs only if the FP8 gate FAILs. The stock fallback runs only if Merlin fails.
   - Is any verdict possible from missing evidence, i.e. a false PASS or a false FAIL?
5. **Capacity.** FP8 149 usable blocks, BF16 about 78; blocks per request = ceil(L/4096) + 7.
   - Are 40k@5, 8k@8, 90k@4 and 116k@3/4 sensible?
   - Does `steady()`'s STEADY rule (running_min == running_max == level, no preemption) work with MTP + PP2?
6. **Time.**
   - BUDGET is 240 min. Step estimates: A0 10, A1 45, A3 110, BF16-load 60, stock 40. Ready timeouts: 45 min (A1), 40 min (A3).
   - Is the 2,700 s cap on the long rows enough at the expected 30-90 tok/s per stream?
   - Does the hard-stop watchdog leave `rc2_results.json` intact?
7. **Saving.** `requests.jsonl` (every request, full text, through a `Stream.run` wrapper), `traces/long_*.txt`, `telemetry.jsonl`, server logs. Any request type not saved?
8. **Rerun / commit.**
   - A competition rerun writes a fallback `submission.json` and skips cells 2-6 without an exception.
   - The commit run also writes one at the end.
   - Notebook size is 127 kB (Kaggle's commit limit is about 400 kB).
9. **Anything else** that would make this GPU run (about 4 h of 4×L4) wasted.

## Already validated locally (Windows, E23 fake vLLM server)

- The full flow runs: A0/A1/A3 DONE, CLEAN.
- The local test also covered:
  - the sampled check;
  - the prefill probe;
  - ladder windows;
  - long rows (120,000 natural / 60,000 forced);
  - TF calls, which are INCONCLUSIVE on the fake (it has no prompt logprobs);
  - 47 saved requests;
  - telemetry;
  - a 120-task rerun submission.
- **Not testable locally:**
  - the real Merlin `build()` (the fake returns before it);
  - the boot-OOM retry;
  - `/dev/shm` (Windows);
  - symlinks (Windows copies instead).
