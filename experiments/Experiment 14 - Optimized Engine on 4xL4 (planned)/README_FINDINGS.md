# Experiment 14: findings (consolidated 2026-10-26)

This file collects every finding of Experiment 14 so far: GPU measurements, facts checked in the engine source, upstream issues and whether they apply, bugs our checks caught, and the levers we rejected.

Each line names its evidence. Measured means on the Kaggle GPU. Source means the SGLang 0.5.21 or vLLM v0.32.0 code. The details are in `Notes/` (numbered), `Bug ang fix/` (the 2026-10-25/26 audits) and `Experiment Results/Kaggle Runs/`.

Purpose and work log: `README_PURPOSE_AND_WORK.md`. History: `README.md`.

## 1. Measured on the GPU (4x L4, Qwen3.8-27B)

### 1.1 Engine arms (E14A SLIM R2, hosen42, 2026-10-23, Notes/30)

The short band is ~5.9k context per request at 12 streams. Long-band prompts had a median of 40,877 tokens (Notes/37 corrects the "8.7k" in Notes/30).

| Arm | tok/s | step ms | accept | Decision |
|---|---|---|---|---|
| SGLang INT4 + DFlash2 width 8 (W1, reference) | 505.0 | 114.0 | 4.86 | reference |
| **DFlash2 width 6** | **531.4 (+6.3 %)** | 96.4 | 4.24 | adopted (gate PASS) |
| DFlash2 width 4 | 526.8 (+5.3 %) | 74.5 | 3.27 | gate PASS, not adopted |
| target-only (no speculation) | 339.6 | 35.3 | — | DFlash2-8 = 1.49x target-only |
| #44401 backport | 508.1 | 114.3 | 4.84 | not adopted (no occupancy gain) |
| delayer off (`--min-free-slots-delay 1`) with graph max 12 | 490.3 | 114.4 | 4.63 | **crashed at 13 running** (see 4) |
| CUDA graphs 1..13 | 500.1 | 114.7 | 4.79 | neutral, adopted for safety |
| int8 Mamba checkpoint | — | — | — | **OOM** at mem 0.83 |

### 1.2 Where a decode step goes (profiler, 20 steps, TP rank 0)

Kernel sum is 116 ms per step:

| Category | ms/step | Share |
|---|---|---|
| NCCL all-reduce, 140 calls | 50.9 | 44 % |
| NCCL all-gather | 8.1 | 7 % |
| Marlin INT4 GEMM | 25.3 | 22 % |
| Other GEMM (drafter, lm_head) | 14.5 | 12.5 % |
| GDN (Triton) | 8.0 | 7 % |
| Attention | 2.7 | 2.3 % |

- **Communication is about 51 % of the step.** The NCCL bench predicted it: 345 µs x 140 + 8 ms.
- TP4 custom all-reduce is disabled on 4 PCIe GPUs.
- The P2P/IPC path is **5.5x slower** than the NCCL baseline: 1,880 vs 344 µs per call.

### 1.3 Production-engine throughput in real runs

| Run | Engine | Streams | tok/s |
|---|---|---|---|
| E15B 45K30K / 40K35K commits (EVAL20) | INT4 + DFlash W6, delayer off, mrr 17, graphs 1..17 | 16 | 451 / 469 at 16 running |
| E15P Poetiq EVAL120 6 h (2026-10-25) | same engine | 15 | **389** whole run; 0 retractions, 0 restarts |
| Poetiq R5 EVAL120 12 h (vLLM FP8 + MTP 3) | vLLM | 16 | 281 |

**What this means for the score:**
- In the 6-h E15P run, 99 of 120 first answers hit the 60k cap. Only 18 tasks reached Poetiq iteration 2.
- Earlier full Poetiq runs got 13 of 63 solved tasks from iteration ≥ 2 (R5), and 9-12 on HARD66.
- **Engine speed is the lever for both Duck and Poetiq.**
- Scores in the same window:
  - E15P: 64/172, task-norm 39.31 %.
  - Duck R3K2: 65/172, 39.03 %.
- LB references:
  - Poetiq S1-60K: 34.58.
  - Duck E15B 40K35K: 22.92.

### 1.3b SGLang radix-cache hits in real runs (log evidence, 2026-10-26)

Summed from the TP0 `Prefill batch, #new-token: X, #cached-token: Y` lines of our real Kaggle server logs:

| Runs | Radix-cache hit (cached / (new + cached)) |
|---|---|
| E15B EVAL120 R3K1 / R3K2 and the E15B SUBMIT commits | 91-93 % |
| EVAL120 G4B (LOW E16, XHIGH E16, XHIGH noE16) | 76-82 % |
| E14A SLIM R2 arms (the int8-mamba arm excluded: 1.1 % on 9 lines) | 59-78 % |

So the "hybrid prefix cache silently hits 0 %" concern (vLLM #45238 / #51250) has not been seen on our SGLang stack. EMS rev 2e-d still measures it per arm and band.

### 1.4 Concurrency

- **The 11-of-12 loss under churn is DFlash's min-free-slots delayer, not #44401.** Under the latch test, W1 was stuck at 11 running in 100 % of samples (Notes/30).
- Production fix (E15B #36): delayer off, with `--cuda-graph-bs-decode 1..17` and mrr 17. This removes the 13-running crash (the buffers are sized by the graph list). E15P ran at 15 running in 3,743 of 3,790 decode-log samples.

## 2. Facts checked in the engine source

### 2.1 SGLang 0.5.21

**Mamba slots per request** (`kv_cache_configurator.py:179-184, 2232-2261`):
- 4 with `extra_buffer_lazy`, 5 with `extra_buffer` (overlap on), 3 with `no_buffer`.
- Speculative draft intermediates are budgeted separately (`:2462-2484`). GPT's "1 + draft = 7 slots" claim is wrong for 0.5.21.
- An auto-sized pool caps mrr (`:2292-2330`) and logs "max_running_requests is capped to N".

**Linear attention (GDN) on the L4 (SM 8.9):**
- FlashInfer GDN raises at startup: it requires SM90+ (`kernels/gdn_flashinfer.py:135-136, 184-187`).
- CuTe DSL prefill is SM100+ only and falls back to Triton.
- GDN verify is always Triton unless FlashInfer.
- So no alternative linear-attention kernel exists for a speculative run on the L4 (`gdn_backend.py:304-360`).

**ReplaySSM:**
- `--enable-linear-replayssm-spec` works with `extra_buffer_lazy`. The no_buffer rule applies only to the decode variant.
- fp16 only warns (`attention_hook.py`).
- ReplaySSM + DFLASH is blocked in code.

**Caches** (`environ.py:1963-1990`):
- `CUDA_CACHE_PATH`, `FLASHINFER_WORKSPACE_BASE` and `TORCHINDUCTOR_CACHE_DIR` are set with `setdefault` under `SGLANG_CACHE_DIR`. They never override an explicit value.
- Per-server cache dirs therefore isolate two TP2 replicas.

**Ports:** without DP attention, a server uses IPC temp files plus `--nccl-port` only (`server_args.py:708-719`). Two TP2 servers need distinct `--port` and `--nccl-port`, nothing more.

**Custom all-reduce:**
- The legacy custom all-reduce refuses world size > 2 without NVLink.
- At world size 2 it turns on when the P2P test passes (`custom_all_reduce_utils.py:442-512`).
- On this VM, P2P measured 5.5x slower, so the TP2 arms use `--disable-custom-all-reduce`.

**EAGLE MTP on Qwen3.8 (agent 2):**
- The draft path must be the target dir.
- The draft is rewritten to `Qwen3_5ForCausalLMMTP`.
- Draft quantization inherits the target's: INT4 MTP layers fall back to unquantized, because RedHat ignores `re:^mtp.*`.
- 3/1/4 and 2/1/3 satisfy `draft = steps + 1`.

**Competitor (NTN) flags:**
- `--attention-backend flashinfer` and `--linear-attn-backend triton` equal our resolved defaults.
- `--max-prefill-tokens 4096` is moot under chunk 2048.
- With `extra_buffer` + fp32 + ratio 0.5, the auto mamba pool caps at ~11 (INT4) / ~9 (FP8) running. Our arms therefore set mamba to 85 (5 x 17).

### 2.2 vLLM v0.32.0

**FP8 GEMM on SM 8.9:**
- For FP8 block-128 the automatic choice is `MarlinFP8ScaledMMLinearKernel`, i.e. **weight-only W8A16**. CUTLASS block FP8 needs SM 90+ (`kernels/linear/__init__.py:478-489`, `cutlass_scaled_mm_entry.cu:161-174`).
- Our S1-60K reference therefore never ran W8A8 FP8.
- `--linear-backend triton` (`arg_utils.py:1808-1810`, `config/kernel.py:244-266`) is the only real W8A8 arm.
- `VLLM_TEST_FORCE_FP8_MARLIN` no longer exists in `envs.py`.

**INT4 + MTP:**
- The MTP draft is built under prefix `mtp` with the target's quant config (`qwen3_5_mtp.py:93-105, 233-235`).
- vLLM matches ignore entries by exact name or `re.match`.
- RedHat's `re:^mtp.*` exempts `mtp.fc` and every fused MTP linear. **#58807 does not apply to this checkpoint.**

**Data parallel:**
- `--max-num-seqs` is per DP rank.
- DP defaults to one API server per rank (`cli/serve.py:107-130`), so the TP2 arms use `--api-server-count 1`.
- `/metrics` reports each engine under its own label. KV usage must be the per-engine max, not the sum.

**Scheduling and caching:**
- A request counts as running once its first prefill chunk is scheduled. Windows must wait for the prefill to settle.
- Prefix caching sets `mamba_cache_mode=align`.
- Model Runner V2 is the default.

## 3. Upstream issues and whether they apply to us

| Issue | What | Applies? | Action |
|---|---|---|---|
| SGLang #36548 / #38009 | DFlash2 greedy output differs from plain decoding, worse when concurrent (other GPUs) | not reproduced on L4 | target-only identity oracle in EMS; every arm gets a verdict vs it and vs the DFlash reference. rev 2e adds three guards: fail closed without the oracle, a cross-request sentinel canary, and an ~85k-context XL parity |
| SGLang #44145 / PR #45151 | the EAGLE/MTP draft keeps a duplicate BF16 embed + lm_head (~4.74 GiB, ~1.19 GiB/GPU at TP4) while the target KV pool is sized | **yes**: proven in 0.5.21 `scheduler.py:1049-1075`, `eagle_worker_v2.py:226-241`; PR open | env-gated patch + the arm `mtp_3_1_4_pr45151` (EMS rev 2c) |
| SGLang #43029 | mamba track boundary under grammar truncation | no (we use no grammar) | none |
| SGLang #42917 | stale CUDA driver cache → wrong W4A16 output | low (Kaggle starts clean; we never ship `nv/`) | E14 `_NEVER_SHIP` |
| SGLang #39597 | FlashInfer autotune non-determinism | guarded | `--disable-flashinfer-autotune` in every arm |
| vLLM #53912 | hybrid MTP + prefix cache → empty / punctuation output | not proven on 0.32.0 | VMS oracle with prefix caching off; a grid-aware degenerate count over every started stream, with empty outputs counted separately (rev 2e / 2e-c) |
| vLLM #58807 | W4A16 + dense MTP load crash | no (see 2.2) | static check before any INT4 MTP launch |
| vLLM #52182 | `VLLM_TEST_FORCE_FP8_MARLIN` removed | yes | arm dropped |

## 4. Bugs our own checks caught before they cost GPU time

| When | Bug | Caught by |
|---|---|---|
| 10-23 | `Sampler._stop` shadowed `Thread._stop()`, crashing on Py3.12 (E14A SLIM run1, ~0.8 quota-h lost) | GPU run; led to the Kaggle CPU-kit rule |
| 10-26 | **the same trap in VMS `VGen._stop`**: every greedy pass of the reference and oracle raised `TypeError` | Kaggle CPU kit run 1 (local Py3.14 passed) |
| 10-23 | staging collision: both Kaggle model dirs end in `/1` | GPT audit 17-19 |
| 10-23 | commit notebook-size limit 407-444 kB: larger commits never start | E14A start failures |
| 10-23 | the delayer was hiding a crash at 13 running with DFlash: `tensor a (12) must match b (13)` in `dflash_worker_v2.py:273` | server-log signs (the parity gate missed it) |
| 10-23 | JIT cache tar `IndexError` | R2 run |
| 10-26 | VMS `lb_marlin` arm = the reference (auto is Marlin on SM 8.9) | agent 4 |
| 10-26 | NTN arms capped at ~9-11 streams by the auto mamba pool | agent 2 |
| 10-26 | VMS whole-text gate would FAIL every MTP arm; an arm could eat the 2 x TP2 reserve; a dead server cost 900 s per window | agents 3 and 4 |
| 10-26 | VMS pre-read INT4 + DFlash2 instead of the FP8 checkpoint it loads | agent 3 |
| 10-26 | EMS: missing probe evidence counted as a pass; dirty-server flags lost; a dead server still "ok"; TP2 judged on replica 0 only | agent 1 |
| 10-26 | my first `vms_degenerate` used a distinct-character ratio that flags every long normal text | local harness |
| 10-26 | rev 2b report lines and the try/except around the EMS body never applied (a silent `str.replace` miss) | rev 2d build; now asserted replacements |
| 10-26 | rev 2e dropped every window whose prefill did not settle in 120 s: the no-prefix arm (16 x 40k prefill) would lose valid data | self-audit (rev 2e-b): 300 s budget, kept as `mixed_load` |
| 10-26 | the rev 2e tail / degenerate rules flagged a normal ARC grid ("0 0 0 ...") as degenerate, which would FAIL correct arms | self-audit (rev 2e-c): grid-aware shared rule |
| 10-26 | the reports said BOUNDARY is "never auto-certified" while `adoptable` accepts it; the VMS no-prefix note said tok/s includes the prefix recompute (it is decode after settle) | self-audit (rev 2e-d): text fixed, `certified_exact` column |
| 10-26 | VMS `vms_window` waited up to 900 s for the level after its streams had died or finished; the EMS reference never got `probe_complete_vs_target`, so it could never be adoptable | rev 2e-d harness run (rev 2e-e fix) |
| 10-26 | a first rev 2e-d try put the prefix-hit count in `src/e14_measure.py`, but the notebooks load the E14 modules from the Kaggle payload dataset shared with E15P | caught before any push; reverted, done inside the EMS cell |

## 5. Levers rejected or closed

- **#44401 backport:** no occupancy gain; the 11-of-12 cause is the delayer.
- **int8 Mamba checkpoint:** OOM.
- **NCCL env knobs:** the baseline is best.
- **GDN kernel tuning:** GDN is only 8 ms of the step.
- **FlashInfer / CuTe DSL linear-attention backends:** unavailable or a no-op on SM 8.9.
- **`VLLM_TEST_FORCE_FP8_MARLIN`:** gone. `--linear-backend marlin` = the default on SM 8.9.
- **2 x TP2 on SGLang:** estimated ~115-170k KV per replica, against the ~800k that 16 x 50k needs. It stays a measured diagnostic arm, flagged `kv_ok_for_production`.

## 6. Open (to be answered by the EMS / VMS GPU runs)

Status of the matrices: rev 2g (2026-10-26), three notebooks (rev 2f, user 2026-10-26: a crash in one must not damage the others): `Q27-XHIGH-EMS_SAFE_ENGINE_MATRIX_SGLANG.ipynb` `9ed2b035`, `Q27-XHIGH-EMS_RISKY_ENGINE_MATRIX_SGLANG.ipynb` `5b8b91f2`, `Q27-XHIGH-VMS_ENGINE_MATRIX_VLLM.ipynb` `6c2a39c5`. Both independent audits of rev 2c were checked item by item and applied; see `Bug ang fix/EMS_VMS_rev2e_independent_audits_disposition_2026-10-26.md` and section 7.

1. **DFlash vs target-only greedy identity on our stack**, at 12 concurrent with thinking on.
2. **MTP (EAGLE 3/1/4, 2/1/3) vs DFlash W6** in tok/s and accept, at 16 and 20 streams. Also the KV gain from #45151.
3. DFlash W6 at 20 / 24 streams.
4. The NTN bundle on FP8 and INT4.
5. ReplaySSM fp16 / fp32.
6. **vLLM:** FP8 MTP 2/3/4, 20/24 seqs, SSM fp16, prefix on/off, **Triton W8A8 vs Marlin W8A16**, INT4 + MTP, and DP2 x TP2.
7. Width 6 at the 40k band (never measured in R2).
8. **Not in the matrices** (next round): target-only at 12/20/24, chunked prefill 2048 vs 4096, radix strategy (lazy / extra / none), vLLM `--language-model-only`, `--kv-cache-dtype auto`. PCIe-IPC all-reduce has been in the matrices since rev 2d (`dflash_w6_pcie`, `mtp_3_1_4_pcie`, `fp8_mtp3_pcie`).
9. Speed at production context: the XL band (~85k block, positions to ~93k) per arm (rev 2d).
10. The real KV gain of the PR #45151 patch (KV tokens of `mtp_3_1_4_pr45151` vs `mtp_3_1_4`).

## 7. Before any production switch (cannot be closed on CPU)

1. The EMS and VMS GPU runs (user go; ~260 min each, ~9 h quota at 2x). The EMS PCIe arms and all of VMS need the dataset `foysalemonshanto/vllm-0-32-0-cu13-wheelhouse` attached.
2. Long generation (60k+ output tokens, with churn and rejected drafts) on the winner. The matrix covers only 256-token greedy gates, a 1,024-token probe and 75-s windows (the XL windows decode up to 8k tokens).
3. A real ARC task run (Poetiq / Duck) at production sampling (temperature 1, top_p 0.95, top_k 20) on the winner.
4. A BOUNDARY winner (near-tie divergences only: adoptable, but not bit-exact) must also be confirmed on ARC task accuracy. `certified_exact` marks the PASS_EXACT arms.
