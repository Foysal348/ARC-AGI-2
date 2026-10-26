# EMS rev 2: GPT pre-GPU serving audit, disposition for the SGLang engine-matrix notebook

Date: 2026-10-25. Scope: only `Q27-XHIGH-EMS_ENGINE_MATRIX_SGLANG.ipynb` (builder `Experiment Notebooks/Build/build_ems.py`).
Sources read in full: `README_QWEN38_SERVING_PRE_GPU_AUDIT_2026-10-25.md` and `ARC_AGI2_LLM_Serving_PreGPU_Audit_README.md`.
The PDF (`ARC_AGI2_LLM_Serving_PreGPU_Audit_BN.pdf`) is the Bengali rendering of the same audit.
Every claim was checked against the local SGLang 0.5.21 source (`_scratch/session_1c6db7aa/sgl_src/src/sglang`). The two
main upstream issues were also read: #42917 and #36548.

## Changed in rev 2

| # | Audit item | Source check | Change |
|---|---|---|---|
| 1 | ACC-2 / ACC-3: DFlash2 greedy output differs from plain decoding, and is worse under concurrency (#38009, #36548) | #36548: DFlash8 concurrent at 8 gave 111/304 wrong; no speculation gave 0/304. fp32 SSM, graphs off and width 4 all still failed. Root cause is open, with no fix PR. Not shown on L4 | target-only on the same INT4 checkpoint is now the identity oracle. Its 4-pass A/A is the gate reference for every later arm. The DFlash reference is judged against it from rows it already has (no GPU time). The 1,024-token probe is compared with both the reference and target-only. If target-only gives no complete A/A, the gate falls back to the DFlash reference and the report says so |
| 2 | Decision rule: wrong output = NO-GO regardless of speed | — | A gate FAIL is recorded and never adopted (`_gated` needs PASS), but the speed is still measured. This lets the report separate a true failure from a near-tie false alarm. Only a dirty server skips the windows |
| 3 | S8: FlashInfer / CuTe DSL GDN backend on SM 8.9 | `kernels/gdn_flashinfer.py:135-136` sets available only when capability >= 9; `FlashInferGDNKernel()` then raises at startup (:184-187). CuTe DSL prefill is SM100+ only and falls back to Triton (`gdn_backend.py:304-324`). GDN verify stays Triton unless the backend is FlashInfer (:347-360) | Both `linattn_*` arms are dropped, and their reasons are recorded in `EMSR["dropped"]`. Under DFlash / MTP every target step is a verify, so CuTe DSL would run the same kernels as the reference |
| 4 | Gate C / #36889: the mamba cap silently lowers real concurrency | `kv_cache_configurator.py:2306-2322` caps mrr at max_mamba_cache_size // ratio and logs it | A `capped` flag is set per arm when the measured level is below the requested level. The report has a CAPPED column |
| 5 | S9 / section 7: 2 x TP2 isolation and pairing | Without DP attention, SGLang uses IPC temp files plus `--nccl-port` only (`server_args.py:708-719`), so the distinct nccl ports already suffice. Custom all-reduce turns on for world size 2 when P2P works (`custom_all_reduce_utils.py:492-511`) | GPU pairs come from `nvidia-smi topo -m`: the pairing whose weaker pair has the best link wins, and ties keep 0-1 / 2-3. Each replica gets its own cache dirs, seeded from the warm caches but never from the CUDA driver cache (#42917 / E14 `_NEVER_SHIP`). The custom all-reduce state is recorded per replica |

## Checked and left unchanged

| Audit item | Finding |
|---|---|
| S5 / S6: 68 mamba slots cannot hold 20 / 24 streams | This was already handled: `_streams(n)` sets mrr n+1, graphs 1..n+1 and mamba 4 x (n+1), so 20 streams get 84 slots and 24 get 100 |
| Gate C claim: "DFlash needs 1 + draft = 7 slots per request" | False for 0.5.21. The ratio is 3 + 1 (lazy) = 4, or 3 + 2 (extra_buffer with overlap) = 5 (`kv_cache_configurator.py:179-184, 2232-2261`). Draft intermediates are budgeted separately (:2462-2482). The other GPT file says the same ("do NOT add D again") |
| ACC-1 / section 5.2: SGLang overrides `CUDA_CACHE_PATH` | False. `environ.py:1983` uses `os.environ.setdefault`, so our explicit `/tmp/e11-cache/cuda` stands. #42917 was triggered by a Docker volume persisted on WSL2 (RTX 5090), and was not reproduced on native Linux. A Kaggle container starts empty, and the JIT cache we ship already excludes the driver cache |
| ACC-4: keep `--disable-flashinfer-autotune` | This is in every arm's base flags, and no arm drops it (NTN arms included) |
| S3: ReplaySSM compatibility | `--enable-linear-replayssm-spec` is allowed with extra_buffer_lazy; the no_buffer requirement applies only to `--enable-linear-replayssm` (`attention_hook.py`). fp16 logs a warning. The fp16 and fp32 arms already exist |
| S4: NTN flags (extra_buffer, ratio 0.5, fp32 SSM) | These stay exact, as the arm is meant to test them. The auto-sized pool and the logged cap are recorded, and CAPPED shows any shortfall |
| S2 / #44145: duplicate BF16 MTP weights shrink the KV pool | No backport, since no patch was verified against the installed source. KV tokens are recorded per arm, so an MTP KV loss is visible against the reference |
| #43029 (speculative mamba committed-length fix, after 0.5.21) | Not backported blindly. The target-only gate is the detector |
| ACC-7: thinking-budget processor under speculation | Not used in EMS (fixed max tokens) |

## Run time note

At about 10-15 min per arm (launch + gate + two 75 s windows + probe), the 200-min budget covers the reference, target-only, roughly 7-8 arms and the reserved 2 x TP2 slot. Arms later in the plan are skipped and recorded, not squeezed. The plan order is the priority order.
