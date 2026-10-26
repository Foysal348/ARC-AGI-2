# Experiment 14: purpose and work (updated 2026-10-26)

Findings: `README_FINDINGS.md`. Full history and the 2026-10-23 plan: `README.md`.

## Purpose

**Goal:** make Qwen3.8-27B serve **600-800 output tok/s** on Kaggle's 4x NVIDIA L4 24 GB, without losing any correctness.

| Setting | Value |
|---|---|
| GPU | Ada SM 8.9, PCIe only, no NVLink |
| Run | offline, 12 h |
| Hidden test | 240 tasks |
| Thinking per answer | ≥ 60k |

**Why:**
- The competition is bound by coverage. In 12 h, each of the 240 hidden tasks gets about `12 h x tok/s / 240` tokens.
- At ~389-470 tok/s, the first answer alone eats the window:
  - in the 6-h EVAL120 Poetiq run, 83 % of first answers hit the 60k cap;
  - only 18 of 120 tasks reached a second Poetiq iteration.
- Earlier 12-h runs show the second and later iterations solve a large share: 13 of 63 solved tasks in R5.
- Top teams use the same model with an optimized engine and full coverage.
- **A ~1.5-2x faster engine is the most direct way to more LB score, for both Duck and Poetiq.**

**Rules this experiment follows:**
- **Correctness first:**
  - every engine arm passes a greedy identity gate against a same-checkpoint target-only oracle;
  - a FAIL is recorded and never adopted.
- **No GPU run without the user's go.** Every GPU notebook first passes the Kaggle CPU kit: the same image, Python 3.12.
- **One notebook per engine,** holding every config of that engine (user, 2026-10-25).

## The work, in order

### 2026-10-23: diagnosis and the first engine run
- **Step-time analysis:** two research rounds (5 + 10 agents) and five GPT notes, all validated (Notes/00-20).
- **E14-0 load path** (`src/e14_loadpath.py`):
  - ranged parallel copy of the model to `/dev/shm`;
  - stall watchdog;
  - phase record;
  - JIT cache that never ships the driver cache.
- **E14-P0 / E14-P:**
  - The Duck "reply re-prefill" comes from blank lines being deleted from the reasoning.
  - Fixed as `E12_RAW_REASONING_HISTORY`.
- **E14A SLIM R2** on the GPU (hosen42, Notes/30): engine arms, profiler, NCCL bench, latch test and a Duck run.
  - Result: DFlash width 6 = +6.3 %.
  - Communication is 51 % of a step.
  - The delayer causes the 11-of-12 loss.
  - Delayer off crashes at 13 running.

### 2026-10-24: submission build and E15 design
- **E14S** (Notes/31): the submission engine, W6 + graphs + raw reasoning.
- **E15 engine design** (Notes/25, 35): MTP preflight, stream counts 14 / 16 / 20, fixes for the 11-of-12 loss.
- The forensic reviews of the EVAL120 runs are in Notes/42-46.

### 2026-10-25: matrices and the vLLM wheelhouse
- **The engine-config matrix, one notebook per engine:**
  - **EMS** (SGLang 0.5.21): `Experiment Notebooks/Q27-XHIGH-EMS_ENGINE_MATRIX_SGLANG.ipynb`, from `Build/build_ems.py`.
  - **VMS** (vLLM 0.32.0): `Experiment Notebooks/Q27-XHIGH-VMS_ENGINE_MATRIX_VLLM.ipynb`, from `Build/build_vms.py`.
  - Each has a 200-min budget. Each arm is its own launch: identity gate, then 8k and 40k steady windows at the level the server really runs, then (EMS) a 1,024-token probe.
  - 2 x TP2 runs last, from a reserved time slot.
- **vLLM 0.32.0 wheelhouse:**
  - Dataset `foysalemonshanto/vllm-0-32-0-cu13-wheelhouse`: 199 wheels, 3.65 GiB, torch 2.13 cu130.
  - Built and install-tested on Kaggle (`Build/build_vllm_wheelhouse.py`, `vllm_0_32_0_requirements.lock`).
- **GPT pre-GPU serving audit** (`Bug ang fix/`) was applied. EMS rev 2 and VMS rev 2 are each checked claim by claim in the source; see the disposition notes in that folder.

### 2026-10-26: second audit round, CPU kits and fixes
- **4-agent implementation audit:** 2 for EMS, 2 for VMS. Every finding was verified and fixed in **rev 2b**:
  - token / logprob-margin gates;
  - one deadline per arm;
  - dead-server stops;
  - complete-evidence picks;
  - NTN mamba 85;
  - TP2 gated on both replicas with custom all-reduce off;
  - the `lb_marlin` no-op dropped;
  - the INT4 static check made exact.
  - Disposition: `Bug ang fix/EMS_VMS_rev2b_agent_audit_disposition_2026-10-26.md`.
- **Kaggle CPU kit for both notebooks:** `Build/build_cpu_kit_engine.py`, dataset `foysalemonshanto/q27-engine-cpu-kit`, kernel `q27-engine-cpu-harness`.
  - Run 1 caught a Py3.12-only crash (`VGen._stop` shadowing `Thread._stop`).
  - Run 2 passed everything: EMS 25/25, VMS 19/19 normal + 19/19 bad-INT4.
- **Deep re-read of the audit (rev 2c):**
  - SGLang #44145 is proven to affect 0.5.21. PR #45151 is now an env-gated patch with its own MTP arm.
  - #43029 is not applicable.
  - Topology, P2P matrix and NCCL INIT logging were added.
  - See `Bug ang fix/Deep_reread_rev2c_2026-10-26.md`.
- **GPT audit packs:** `Experiment Review/ENGINE_MATRIX_EMS_VMS_REV2B_AUDIT_FOR_GPT_2026-10-26.zip`, then `..._REV2C_...zip`, then `..._REV2E_...zip` (current).
- **rev 2d (user: "take 1 and 2"):** see `Bug ang fix/Rev2d_long_context_and_pcie_ipc_2026-10-26.md`.
  - An XL band: a shared ~85k-token block, streams = KV // 100k, a third 75-s window on every TP4 arm.
  - FlashInfer PCIe-IPC all-reduce arms:
    - EMS `dflash_w6_pcie` and `mtp_3_1_4_pcie`, through a FlashInfer 0.7.0.post1 overlay from the vLLM wheelhouse;
    - VMS `fp8_mtp3_pcie`, through the native `VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC`.
  - Budget 200 → 260 min.
  - rev 2d-b: the rev 2b report / try-except edits had silently not applied. They are now applied with asserts.
- **rev 2e: the two independent audits of rev 2c**, `Bug ang fix/README_EMS_VMS_REV2C_INDEPENDENT_AUDIT_2026-10-26.md` and `Bug ang fix/ENGINE_MATRIX_REV2C_INDEPENDENT_AUDIT_README_BN.md`.
  - Every item was checked in the code and the sources before the change. Disposition, item by item: `Bug ang fix/EMS_VMS_rev2e_independent_audits_disposition_2026-10-26.md`.
  - Fail closed: nothing is adoptable without the target-only oracle (EMS `_gated`, VMS `oracle_ready`).
  - VMS compares integer token ids (`return_token_ids`). Every greedy row needs `[DONE]` + `finish_reason`, else it is `INCOMPLETE_SSE`.
  - `gate_class` = INCOMPLETE / FAIL / BOUNDARY / PASS_EXACT, plus a post-divergence tail check.
  - Only healthy windows count: no stream error, empty or degenerate output, and an EMS running share ≥ 0.80 (our proposed bar).
  - VMS joins the streams, then requires an idle server between windows (otherwise `DIRTY_SERVER_AFTER_WINDOW`).
  - VMS records both decode tok/s and end-to-end tok/s. EMS tok/s is labelled decode only.
  - Prefix-cache counters: VMS reads the vLLM metrics. EMS (rev 2e-d) reads the SGLang prefill log lines.
  - Also added: a cross-request sentinel canary, an XL parity gate, and separate `measurement_ok` / `capacity_ok` / `correctness_ok` / `adoptable` fields.
  - The #45151 patch:
    - is written atomically, with both snippets verified;
    - is backed up and its sha recorded;
    - its arm is skipped unless the patch applied.
  - `ntn_fp8` moved after the 20/24-stream arms.
- **Self-audit (user: "research before changing, remove nothing without evidence"):**
  - **rev 2e-b:** an unsettled VMS window is kept as `mixed_load` (settle budget 300 s), not dropped. The next band is skipped only after a dirty or dead server.
  - **rev 2e-c:** a grid-aware degenerate / tail rule. The old rules flagged normal ARC grids.
  - **rev 2e-d:**
    - The report text now matches the code: BOUNDARY is adoptable under the documented near-tie rule, but it is not bit-exact. A `certified_exact` column was added.
    - The stale VMS no-prefix note was fixed.
    - EMS prefix-cache hits are measured inside the EMS cell, because the shared `e14_measure` comes from the Kaggle payload dataset.
  - **rev 2e-e** (found by the rev 2e-d harness run; `_scratch/vms/patch_rev2e_e.py`):
    - VMS `vms_window` waited up to 900 s for `level` running requests even after the streams had died or finished. It now stops once fewer than `level` streams are alive, with the same error plus the stream errors. On the GPU this saves an arm's time cap.
    - The EMS reference record never got `probe_complete_vs_target`, so `_gated` could never mark the production reference adoptable. Fixed.
    - Harness only: both fake servers listen with a backlog of 256 (Windows dropped some of 16-24 simultaneous connects), and the negative tests use longer strings (Qwen BPE merges a run of '!' into few ids).

## The two matrices (current arms)

**EMS (SGLang 0.5.21, INT4 RedHat + DFlash2):**

| # | Arm(s) |
|---|---|
| 1 | Reference: DFlash W6, 16 streams, delayer off, graphs 1..17, mamba 68 |
| 2 | target-only (the identity oracle) |
| 3 | MTP EAGLE 3/1/4 |
| 4 | MTP 3/1/4 with PR #45151 |
| 5 | NTN bundle on FP8 and on INT4 (mamba 85) |
| 6 | DFlash W6 at 20 and 24 streams |
| 7 | the best gated MTP at 20 streams |
| 8 | MTP 2/1/3 |
| 9 | mem 0.86 |
| 10 | ReplaySSM fp16 / fp32 |
| 11 | DFlash W4, W8 |
| 12 | Last: 2 x TP2 (DFlash W6, then the best MTP) |
| + | rev 2d: `dflash_w6_pcie` and `mtp_3_1_4_pcie` (PCIe-IPC all-reduce); an XL band on every TP4 arm. rev 2e: `ntn_fp8` runs after the 20/24-stream arms |

**VMS (vLLM 0.32.0):**

| # | Arm(s) |
|---|---|
| 1 | Reference = the S1-60K engine: FP8 block-128, fp8 KV, MTP 3, FLASHINFER, prefix cache, 16 seqs |
| 2 | FP8 target-only with prefix caching off (the oracle) |
| 3 | MTP 2, 4 |
| 4 | INT4 + MTP 3 at 16 and 20 (after a static MTP check) |
| 5 | FP8 MTP 3 at 20 and 24 seqs |
| 6 | SSM fp16 |
| 7 | prefix caching off |
| 8 | `--linear-backend triton` (W8A8) |
| 9 | Last: DP2 x TP2, FP8 and INT4 |
| + | rev 2d: `fp8_mtp3_pcie` first after the oracle; an XL band. The order since rev 2e: pcie, noprefix, mtp2, mtp4, s20, s24, lb_triton, ssm_fp16, int4_s16, int4_s20 |

Time: about 10-15 min per arm, plus the XL band. The budget is 260 min (rev 2d). Arms that do not fit are recorded as `NOT_TESTED_BUDGET: <reason>`; they are never squeezed and never reported as tested.

## How to run (the gate order)

1. **Build:**

   ```
   python "Experiment Notebooks/Build/build_ems.py"
   python "Experiment Notebooks/Build/build_vms.py"
   ```

   The `PUSHED` sets protect notebooks that were already run.
2. **Local harness:** `src/harness/make_harness_{ems,vms}.py`, then `harness_{ems,vms}_nb.py <notebook>`. VMS also runs with `VMS_BAD_INT4=1`. Use Python 3.12 if possible: `_scratch/py312venv`.
3. **Kaggle CPU kit:** `python "Experiment Notebooks/Build/build_cpu_kit_engine.py"`, then `kaggle datasets version -p cpu_kit_engine` and `kaggle kernels push -p cpu_kernel_engine`. Require `all_ok`.
4. **GPU:** only on the user's go. The account needs ≥ 2 x 270 min of quota (~9 h quota each at 2x). Attach the vLLM wheelhouse dataset to both notebooks. Push with QUICK_SAVE and commit from the editor.

## Status (2026-10-26)

- **Current: rev 2g** (rev 2f split + the rev 2f audit fixes, `Bug ang fix/EMS_VMS_rev2g_disposition_rev2f_audits_2026-10-26.md`), three notebooks (rev 2f, user 2026-10-26: a crash in one must not damage the others): `Q27-XHIGH-EMS_SAFE_ENGINE_MATRIX_SGLANG.ipynb` `9ed2b035`, `Q27-XHIGH-EMS_RISKY_ENGINE_MATRIX_SGLANG.ipynb` `5b8b91f2`, `Q27-XHIGH-VMS_ENGINE_MATRIX_VLLM.ipynb` `6c2a39c5`.
  - EMS SAFE: oracle + reference + MTP 3/1/4, DFlash 20 / 24 (+ the best MTP at 20), MTP 2/1/3, mem 0.86, DFlash width 4 / 8. No site-packages patch, no FlashInfer overlay, no 2 x TP2.
  - EMS RISKY: oracle + reference + MTP 3/1/4 (the direct A/B partner), #45151, PCIe-IPC (DFlash / MTP), NTN INT4 / FP8, ReplaySSM fp16 / fp32, then 2 x TP2.
  - VMS: unchanged layout (its risky INT4 / TP2 arms are already last).
  - Every arm call is in its own try/except in all three, so one arm's crash never ends the remaining arms.
  - Each part reruns install + oracle + reference (~25-35 min overhead). Total GPU about 9.2 h (about 18.5 h quota).
  - Local Python 3.12 harnesses: see `_scratch/ems/run13_py312.log` and `_scratch/vms/run13_py312.log`.
  - The Kaggle CPU kit v4 on these shas is pending.
  - GPT pack: `Experiment Review/ENGINE_MATRIX_EMS_VMS_REV2E_AUDIT_FOR_GPT_2026-10-26.zip`.
  - Open before production: `README_FINDINGS.md` section 7.
- Earlier (superseded):

- **EMS rev 2c** (`c062255e`) and **VMS rev 2c** (`675db353`) are built:
  - local harnesses pass (EMS 26/26, VMS 19/19);
  - **Kaggle CPU kit run 3 ALL OK** on these shas (EMS 26/26, VMS 19/19 normal + 19/19 bad-INT4, Python 3.12.13);
  - GPT audit pack: `Experiment Review/ENGINE_MATRIX_EMS_VMS_REV2C_AUDIT_FOR_GPT_2026-10-26.zip`.
- Not yet run on the GPU (needs the user's go and an account with ≥ 2 x 210 min quota).
- E15P (Poetiq on the SGLang engine) was submitted on 2026-10-26 as ref 57644341 and is PENDING.
