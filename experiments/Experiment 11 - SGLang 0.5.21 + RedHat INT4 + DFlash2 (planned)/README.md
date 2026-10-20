# Experiment 11: SGLang 0.5.21 + RedHat Qwen3.8-27B INT4 + DFlash2 (planned)

**Goal:** a faster serving stack on 4×L4, so that more tasks are solved per hour, at equal accuracy.
- Test it in a separate validation notebook.
- E10 (FP8 + vLLM 0.19) stays as the fallback.
- Background and verified facts: memory `sglang-int4-dflash2-plan`; Exp10 `Experiment Results/Analysis/W4A16 checkpoints - MTP and vLLM 0.19 compatibility (2026-10-15).md`.

## Stage 1: offline assets (CPU notebooks, internet ON, no GPU quota). Quick-saved 2026-10-15

Both notebooks are on foysalemonshanto. Built by `Experiment Notebooks/Build/build_e11_assets.py`, which is deterministic. They are adapted from the ARC-AGI-3 notebooks that were proven on Kaggle.

| Notebook | Kaggle | What it makes |
|---|---|---|
| `E11_01_model_mirror_upload_on_KAGGLE.ipynb` (sha `40b57adf…`, Kaggle v2: **private** models, user choice) | `foysalemonshanto/e11-01-model-mirror-int4-dflash2` | Streams HF → Kaggle with no disk staging and hash-checks every file. Creates `<user>/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1` (RedHatAI/Qwen3.8-27B-INT4 @ `91bd022d`, 19.47 GB) and `<user>/qwen3-8-27b-dflash2/pytorch/bf16/1` (z-lab/Qwen3.8-27B-DFlash2 @ `50307d4c`, 3.85 GB). Both are Apache-2.0; **private** (`PUBLIC = False`, user choice 2026-10-15) |
| `E11_02_sglang_0_5_21_wheelhouse_build_on_KAGGLE.ipynb` (rev 2, sha `f4d184cd…`, Kaggle v2) | `foysalemonshanto/e11-02-sglang-0-5-21-wheelhouse` | Dataset `<user>/sglang-0-5-21-cu13-wheelhouse` |

**Rev 2 (2026-10-15), after the external audit.** E11-01 code unchanged by rev 2 (later v2 only flips the models to private). E11-02 changes:
- **CUDA compiler pins:** `nvidia-cuda-nvcc`, `-crt`, `nvidia-nvvm` and `nvidia-nvjitlink` are pinned 13.4.92 → **13.0.88**.
  - 13.0.88 is the set torch 2.13's own `cuda-toolkit 13.0.3` defines, with cudart 13.0.96. The Kaggle L4 driver supports CUDA 13.0.
  - 13.4.92 came only from `humming-kernels[cu13]`, which requires nvcc with no version pin.
  - The other 216 pins are unchanged. The uv re-resolve gives exactly these 4 differences.
- **Install test in a venv** (`venv --without-pip` + `pip --python`) instead of `--target` + `PYTHONPATH`.
  - `nvidia-cutlass-dsl` is only a `.pth` file. ARC-AGI-3's vLLM log, with `--target`, showed `cuteDSL (CUTLASS Python) not available`.
  - Console scripts such as ninja now land in `bin/`.
- **CPU toolchain check** of the wheel nvcc: version; sm_89 cubin, object and shared-lib compile/link; header and lib file presence.
  - Reason: in ARC-AGI-3, FlashInfer's JIT never found the wheel nvcc (problem #2 in `Occamy_ARC-AGI-3_Problems_and_Solutions.pdf`).
- **More import probes:** `cutlass`, `sglang.srt.models.qwen3_5`, `sglang.srt.models.dflash`, and the DFlash2 config.
- **The `kaggle datasets create` exit code is now checked.**
- **Kept as is on purpose:**
  - Probes stay non-fatal, because a CPU box cannot load GPU kernels. They are recorded in `BUILD_PROBE.json` and reviewed before Stage 2.
  - E11-01's last cell still only prints. M4 already raises on any hash mismatch before creating the instance.

**About the wheelhouse:**
- 220 wheels, about 5 GB: torch 2.13.0 cu13, flashinfer 0.6.18 + jit-cache 0.6.18+cu130, transformers 5.12.1, sglang-kernel 0.4.7.
- It includes an offline install test, CPU probes (imports; transformers reads the INT4 config; SGLang registry) and a launch-flag check.

**Found while building:**
- **sglang 0.5.19–0.5.21 pin `cuda-tile==1.6.0rc5`, which exists only as an sdist,** so `pip download sglang` (binary-only) fails.
  - The lock was resolved off-Kaggle with `uv pip compile --override cuda-tile==1.6.0` (which has a cp312 wheel) for Linux / CPython 3.12.
  - It is installed with `--no-deps`.
  - Files: `Build/sglang_0_5_21_requirements.lock`, `Build/uv_overrides.txt`.
- `flashinfer-cubin 0.6.18` does not exist (newest is 0.6.13). The precompiled kernels come from `flashinfer-jit-cache 0.6.18+cu130` (about 1 GB, generic). An SM89-only build exists only for 0.7.x.

**How to run each (user):**
1. Open it in the Kaggle editor.
2. Add-ons → Secrets → attach `KAGGLE_USERNAME` and `KAGGLE_KEY`.
3. Save Version → **Save & Run All**. API-started runs cannot read secrets.

### Stage 1 run results (2026-10-15)
Files are in `Experiment Results/Stage 1 runs (2026-10-15)/`.

**E11-01: done (Kaggle version #5).**
- Both instances were created and are **private**. Every file size matches the HF pin.
  - `foysalemonshanto/qwen3-8-27b-int4-redhat/pyTorch/w4a16-g128/1`: 11 files + manifest, 19.47 GB.
  - `foysalemonshanto/qwen3-8-27b-dflash2/pyTorch/bf16/1`: 3 files + manifest, 3.85 GB.
- The first run streamed the files fine (all hashes OK, 32 MB/s) but failed at instance creation on Kaggle's private quota (6.3 GB left).
  - The user made `occamy-1-0-nvfp4` (24 GB) public to free space, then reran.

**E11-02: the build passed on Kaggle; publish failed only because the Secrets were not attached to this notebook.**

What passed:
- 220 wheels, 4.68 GiB, nothing missing (download 297 s). Offline venv install OK in 157 s, 9.7 GiB.
- `.pth` files ran: `cutlass` 4.6.2 imports and `dsl_packages` is on `sys.path`. This confirms the venv fix.
- Versions: torch 2.13.0+cu130, transformers 5.12.1, sglang 0.5.21, flashinfer 0.6.18.
- The SGLang registry has `Qwen3_5ForConditionalGeneration`, `Qwen3_5ForCausalLM` and `DFlash2DraftModel`. Module imports for `qwen3_5` and `dflash` are OK.
- transformers reads the INT4 config (`qwen3_5`, vocab 248077). The chat template ends with `<think>`.
- All 22 launch flags are present.
- nvcc 13.0.88 runs. sm_89 cubin and object compiles work, with and without `-I`. gcc 11.4, ninja 1.13.2.

Expected on CPU:
- `sgl_kernel` cannot load `common_ops`. There is no GPU to pick the arch directory, so this is re-checked on GPU.

**For Stage 2:**
- The pip runtime ships only `lib/libcudart.so.13`. There is no `libcudart.so` and no `lib64`, so `-lcudart` fails.
- With a `libcudart.so` link, nvcc's own link step then needs `libcudadevrt.a`, which is not in the wheels.
- So Stage 2 adds `lib/libcudart.so` → `.so.13` and `lib64` → `lib` under `CUDA_HOME`.
- JIT extensions link with g++ and `-lcudart`, not nvcc's device runtime. Re-check this on GPU with one FlashInfer JIT op.

### Dataset v2 (2026-10-15): Kaggle drops `+` from file names
**What went wrong in run 1.** The wheels were uploaded and the dataset was created: public, 4.99 GB, ready. But:
- The CLI's create call got a non-JSON reply and exited 1, so the notebook raised an error even though the dataset existed.
- Kaggle stored `flashinfer_jit_cache-0.6.18+cu130-…whl` as `…-0.6.18cu130-…whl`. A plain `pip install --find-links` cannot match the lock pin, which would trap any other user or AI.

**E11-02 v2** (sha `4fe26019`, Kaggle notebook version 7) fixes this:
- **`install_wheelhouse.py`** is now in the dataset. It restores the original names from SHA256SUMS and checks the sha256 of each restored wheel, creates the venv, installs with `--no-deps`, and adds the `libcudart.so` and `lib64` links. One command: `python <dataset>/install_wheelhouse.py /tmp/sglang-venv`.
- **W3 tests the installer before anything is published.** It runs on a copy laid out the way Kaggle stores the files (`+` removed).
- **W7 publishes v2 as a new version of the same dataset** (`kaggle datasets version`). It waits until `install_wheelhouse.py` is listed instead of trusting the CLI exit code.

**Checked locally on toy wheels:** the name is restored, pip installs `0.6.18+cu130`, and a corrupted file stops the installer with "sha256 mismatch".

E11-03 has the same name restore built in, so it works with both v1 and v2.

**E11-02 v2 run: COMPLETE (2026-10-15). Stage 1 is done.**
- The dataset is now on **version 2**: 227 files = 220 wheels + 7 others (`install_wheelhouse.py`, README, SHA256SUMS, lock, BUILD_*).
- On the Kaggle-style layout, the installer restored `flashinfer_jit_cache-0.6.18+cu130` and the install took 150 s.
- The `datasets version` call exited 0, and the new version was listed after 33 s.
- CPU toolchain: every file is present (the installer made `libcudart.so` and `lib64`). Every nvcc compile/link test passes, including the shared-lib link.
- The `cutlass` `.pth` import works, the registry is OK, and every flag is present.
- Only `sgl_kernel` cannot be imported on CPU. That is expected, and the GPU preflight re-checks it.
- Files are in `Experiment Results/Stage 1 runs (2026-10-15)/E11-02 v2 run (dataset v2 published)/`.

## Stage 2: GPU validation notebook (built 2026-10-15, not pushed yet)

The notebook is `Experiment Notebooks/E11_03_sglang_int4_gpu_validation_on_KAGGLE.ipynb`, built by `Build/build_e11_03_gpu_validation.py`.

**Settings:**
- 4×L4, Internet OFF, run on foysalemonshanto (the models there are private).
- Attach: the wheelhouse dataset, both models, and the competition data.
- It is **not a submission**.
- It saves `results.json` after every stage. `report.md` and every server log go to `/kaggle/working/e11_03/`.

| Cell | What it does |
|---|---|
| 1 | Builds the venv from the wheelhouse, as in the E11-02 test. Shims under `CUDA_HOME`: `lib/libcudart.so` and `lib64` (from the E11-02 probe), plus a `libcuda.so` shim on `LIBRARY_PATH` (the E10 lesson). Uses fresh SGLang, CUDA, Triton and torch-extension caches. |
| 2 | Preflight inside the venv. **Fatal checks:** bf16 GEMM on 4 GPUs, `sgl_kernel` import, NCCL TP4 all-reduce. **Recorded only:** Triton kernel, FlashInfer decode at head_dim 256 (bf16 and fp8 KV, our model's shape at TP4), nvcc JIT torch extension, CuTe DSL import. If a fatal check fails, every server stage is skipped. |
| 3 | Builds E10's exact prompt (chat template byte-identical to FP8, xhigh) and tokenizes it in the venv with transformers 5.12.1. Uses 32 sweep puzzles (seeded) and the 8 sanity puzzles. |
| 4 | Server helpers: launch, health, `/generate` (`input_ids`), `/get_server_info`, decode-log parsing, E10's grid parser (verbatim) and E10's DIRECT protocol (think cap 77,824 → natural or forced close → answer 8,192). |
| 5 | Three configs, each with: smoke test, greedy 4 × 512 tokens, sweep at 1/4/8/16/24/32 requests × 4,096 tokens, and a long run of 16 × 24,576 tokens. **A** = INT4 alone. **B** = INT4 + MTP (`NEXTN`; draft path = INT4; `unquant`; 3 steps, topk 1, 4 draft tokens). **C** = INT4 + DFlash2 (`DFLASH`, `unquant`, block size 8). If a launch fails, it retries with `--attention-backend triton`. |
| 6 | **D** = whichever of B and C is faster at 16 requests, rerun with `--mamba-ssm-dtype float16`: greedy drift vs float32, plus the sweep. |
| 7 | Sanity solve on the faster of B and C: 8 cheap puzzles (`2ba387bc bf45cf4b 58f5dbd5 136b0064 31f7f899 1818057f e8686506 7491f3cf`), E10 DIRECT at temperature 0.6. FP8's expected score is ~6.6 of 8. |
| 8 | `report.md`. |

**Common flags:**
- `--tp-size 4 --kv-cache-dtype fp8_e4m3 --context-length 106496`
- `--max-running-requests 32 --chunked-prefill-size 4096 --mamba-ssm-dtype float32`
- `--enable-metrics --disable-flashinfer-autotune`
- Optional flags, added only when `--help` lists them: `--cuda-graph-max-bs-decode 32`, `--watchdog-timeout 1800`, `--decode-log-interval 40`.
- Mamba radix strategy is left on `auto`. SGLang resolves it to `extra_buffer` for Qwen3.5 when the overlap scheduler is on.

**Verified in the SGLang 0.5.21 source:**
- MTP: `Qwen3_5ForConditionalGeneration` is not in the auto-draft-path list. The draft config is rewritten to `Qwen3_5ForCausalLMMTP`.
- DFLASH: `max_running_requests` defaults to 48, so we set it explicitly. The draft attention falls back to `flashinfer`.
- `/generate` `meta_info` returns `spec_accept_length` and `spec_verify_ct`.

**Expected run time:** ~2.5–3 h. A time guard stops new stages after 5 h.

**Local CPU harness:** `scratchpad/e11_03_harness` runs every cell against a fake SGLang server.
- Tokenization uses the real INT4 tokenizer with transformers 5.12.1.

Then EVAL accuracy (Stage 3) runs with the production harness ported to the winning config. It is paired against FP8 ORDER2/R5 on EVAL120.

### Stage 2 rev 3 (2026-10-17): extra tests from D66/L66 and GPT (notebook sha `4a1538f9…`)

**Builder changes:**
- The sweep now includes **12 requests** (D66/L66: 12 slots suit long-thinking Poetiq; 16 gave 66 preemptions at 77.8k).
- **New cell 5b** runs four tests on the best speculative config (`X_<best>_extra`; falls back to A):

| Test | What | Pass / what to read |
|---|---|---|
| contcache | 4 prompts think 6,144 tokens. The forced-close continuation (prompt + thinking + stop text) is sent again; `cached_tokens` and wall time are compared with a cold request of the same length (2nd token changed) | `cached_tokens / prompt_tokens ≥ 0.9`. If it fails, every forced close re-prefills, a large hidden cost on a hybrid model (mamba radix cache) |
| longctx | Up to 12 requests (as many as KV allows) at about 90k-token contexts, 1,024 new tokens each | tok/s, accept length, token usage at the context lengths E10 actually reaches (78k–123k thinking) |
| contam | The same 4 greedy prompts alone, then inside a 16-request batch | identical / common prefix; an early divergence into other content means cross-request contamination |
| temp | 12 × 4,096 tokens at T 1.0 vs T 0.6 | tok/s and accept length. vLLM data hinted T 0.6 decodes about 17 % faster; DFlash2 may differ |

`run_config` gained an `EXTRA_STAGES` hook; each stage is isolated by try/except and time-guarded. The report gained an "Extra tests (rev 3)" section.

**Harness** (`scratchpad/e11_03_harness`):
- The fake server now has a radix-style prefix cache.
- 6 new checks were added.
- main 39/39, ladder 40/40 and bfail 33/33 PASS.
- The hvenv was recreated with `python -m venv --upgrade`; its site-packages were unchanged.

**Expected run time:** about 3–3.5 h (rev 2 was 2.5–3 h); the 5 h guard is unchanged.

**Blocker to run it:** the INT4 and DFlash2 models are private on foysalemonshanto, whose GPU quota is exhausted (35.45 of 30 h). Options:
- share the models with another account (hosen42);
- make them public;
- wait for the weekly reset.

## Stage 3: E11-04 capacity + real-concurrency validation (built 2026-10-17, not pushed)

**Files and pedigree:**
- Notebook `Experiment Notebooks/E11_04_sglang_capacity_validation_on_KAGGLE.ipynb`, built by `Build/build_e11_04_capacity_validation.py`.
- The builder reuses the E11-03 cells that ran clean (install, preflight, tokenizer, server helpers) verbatim and patches only the E11-03 run-1 lessons.
- Validation of the GPT audits: `Experiment Results/E11-03 run1 analysis (2026-10-17).md` §6.

**Arms.** All use INT4 + DFlash2 (8 draft tokens, unquant drafter), TP4, `--max-running-requests 12`, `--disable-prefill-cuda-graph` and `--mem-fraction-static` 0.83 (ladder 0.78).

| arm | SSM state handling | slots per request | SSM dtype | role |
|---|---|---|---|---|
| W1 | `extra_buffer_lazy`, mamba cache 56 | 4 | fp16 | E11-03's D, uncapped |
| W2 | `no_buffer`, mamba cache 48, `--enable-linear-replayssm-spec` | 3 | fp32 | no per-draft SSM snapshots; accuracy reference |
| W3 | as W1 | 4 | bf16 | the dtype SGLang's own warning suggests |

- W2 uses no_buffer, not lazy, because the ReplaySSM ring is not reset on the extra_buffer donation path (source).

**New in E11-04:**
- **Startup caps read from the server log**, not `/get_server_info`: effective running, capped_to, KV tokens, target and draft KV GB, mamba cache, verify graph bs, free memory, FP8 scale warning.
- **Capacity gate:** 12 real, KV ≥ 900k, verify graph at bs 12.
- **Real-concurrency check at every sweep level**, fail-closed.
- **PROD closed loop:** 12 always in flight, ARC prompts, up to 60,000 thinking at T 1.0, refill on finish.
  - tok/s comes from the `sglang:generation_tokens_total` delta; `abort_all` ends the window.
  - Windows: W1 35 min, W2 25 min, W3 15 min; 10 min if the gate fails.
  - Target ≥ 332 tok/s.
- **Marker contamination test:** 8 companions each write their own codename; a leak is a blocker.
- **contcache** on W1 and W2.
- **Sanity 8 puzzles** on W1, and on W2 if time allows.
- **Checks** whether the checkpoint carries k/v scale tensors.
- **GO / NO-GO table** per arm.
- **Expected time:** about 3 h; the guard is 4.5 h.

**CPU harness:** `_scratch/session_1c6db7aa/e11_04_harness/` (`run_harness_04.py`, `fake_sglang_server_04.py`).
- The fake now models the mamba-cache cap: queueing beyond the cap, the cap warning, verify graph bs, KV sizing, abort, the generation counter and periodic Decode lines.
- Scenarios: main, capfail, replayfail, contam.

### E11-04 rev 2 → rev 2b (2026-10-18, after two GPT audits + Claude self-audit)

- **rev 2 (GPT findings, each verified in the 0.5.21 source):**
  - ReplaySSM arm removed (DFLASH + ReplaySSM is KDA-only).
  - Arms: W1 fp16 / W2 bf16 (full) / W3 fp32 (short).
  - PROD tok/s comes from all-response completion tokens; the finished-only counter is a lower bound.
  - Fail-closed capacity gate.
  - Frozen PROD log.
  - Long-context proof from `#full token`.
  - Fail-closed marker and contcache tests.
  - No inherited results.
  - Separate capacity winner and production candidate.
  - Stage order: sweep → PROD → marker → cache → sanity.
  - Checkpoint FP8 scale values are read.
- **rev 2b (self-audit):**
  - A diagnosed launch fallback: OOM → 0.78, lazy rejected → extra_buffer + mamba 70, prefill-graph complaint → keep the graph. Max 3 launches.
  - Long-prefix samples prefer the most recent responses.
  - W2 PROD 35 min.
  - Marker companions get 1,024 tokens.
  - Slots per request come from the launched flags.
- **Notebook sha** `48324697…`. Harness: main, shortctx, oomfail, capfail, contam and lazyfail all PASS.
- **Write-up:** `Experiment Review/E11-04 rev2b self-audit + goal check (2026-10-18).md`.
- **2026-10-18: committed** to `esfersami50/e11-04-sglang-capacity-validation` v1 (SAVE_AND_RUN_ALL, sha `48324697`). Status RUNNING at 08:50 UTC. The user shared the two private Foysal models with esfersami50 first.
