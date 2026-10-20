# E10 Poetiq R4: audit (Claude, 2026-10-14)

**Bundle:** `Q27_E10_POETIQ_R4_CONFIRMED_FIXES_ALL_IN_ONE.zip`, extracted to `R4 bundle (extracted, untrusted)/`.
- R4 notebook sha `596bf694…`, which matches the bundle README.
- Only cells 5 and 7 changed from R3; the base cells are unchanged.

**Method:**
- Read the R3→R4 diff of cells 5 and 7, including the new runner.
- Ran the real R4 notebook cells in our fake-vLLM harness (`harness_r4_probe.py`). The harness model writes a *float* identity program (`grid.astype(float)`), which R3 used to reject.
- Windows has no setuid or seccomp, so for the scheduler run the harness swaps in a plain subprocess runner (scenario B). The real runner can only be tested on Linux, and that means Kaggle.

## The four R3 findings

| R3 finding | R4 | Verified |
|---|---|---|
| **B1** `LOG_EVERY_SEC` undefined | defined in cell 7 | ✅ Harness, without injecting anything: 120 requests, 120 iterations, solver complete, no `NameError` |
| **B2** sandbox needs `unshare` | new runner: root→UID 65534, `PR_SET_NO_NEW_PRIVS`, libseccomp, rlimits, process-group timeout; plus a startup self-test | ⚠️ The code reads correctly. **The Kaggle positive path is not tested yet** (probe built, see below) |
| **H1** float grids rejected | `poetiq_normalize_grid` accepts integral floats and NumPy integers; rejects fractions, NaN and bools | ✅ Float identity programs now get scores of 0.73–0.95 (R3 gave 0) |
| **H2** thinking below 49,152 | drops the lowest-scored feedback solutions until thinking ≥ 49,152; skips the request if even no feedback fits | ✅ The harness run showed 12 trims and 0 skips; the logic matches the user's rule |

## New findings

| # | Severity | Finding | Evidence | Fix |
|---|---|---|---|---|
| N1 | **Medium (policy)** | **A failed self-test raises an exception in cell 5.** In a hidden rerun, a notebook exception makes the submission fail. The base's rule is "do not raise; the identity submission stays valid". In practice both outcomes score about 0, but the exception also breaks the "a valid submission is always produced" rule. | Harness scenario A on a machine without setuid: `AttributeError` in cell 5, and the notebook stops. On Kaggle a seccomp or setuid failure would raise `RuntimeError` the same way. | Catch the failure, log it loudly, put it in the report, and keep the submission path alive. |
| N2 | **Medium (unverified)** | **Threads are probably broken in the child.** R4 makes `clone3` return EPERM. glibc 2.34+ (Kaggle has 2.35) tries `clone3` first in `pthread_create` and falls back to `clone` **only on ENOSYS**, so every new thread fails. This affects OpenCV's parallel ops and Python `threading`; NumPy/OpenBLAS with 1 thread is fine. | Code reading plus glibc behaviour. The R4 probe has `python thread` and `cv2 resize+blur` cases to confirm. | Return **ENOSYS** for `clone3`, so glibc falls back to `clone`, which R4 already filters to allow `CLONE_THREAD` only. |
| N3 | Low | The self-test checks a file inside a root-only 0700 temp directory, not `/kaggle/working`. Whether UID 65534 can create or delete files in `/kaggle/working` depends on that directory's permissions, which are unknown. | Self-test code. | The probe covers it: create, overwrite and delete cases in `/kaggle/working`. |
| N4 | Low (test trap) | The thinking floor is hard-coded at 49152. With any `THINK_BUDGET` below 49,152 (for example the harness default of 3,000) **every request is skipped**. Production uses 77,824, so this does not matter there. | Harness scenario C: 12 tasks skipped, 0 requests. | Use `min(49152, THINK_BUDGET)`. |
| N5 | Info | The bundle README says the donor prompts are "abbreviated". They are not: cell 5 ends by replacing them with the verbatim Poetiq prompts. | Cell 5. | No action needed. |

**Unchanged (user decisions or GPU observation):** Poetiq-only with no DIRECT attempt; temperature 1.0; H3 KV concurrency is left to be observed on the GPU.

## Kaggle gate

- **Probe:** `Experiment Notebooks/Build/sandbox_probe/E10_SANDBOX_PROBE_R4_CPU.ipynb`, built by `build_sandbox_probe_r4.py`.
- **What it runs:** R4's cell 5 verbatim, except that the self-test is wrapped so the probe can report every case.
- **Cases:** self-test, NumPy/SciPy/cv2, threads, float grids, writes/deletes in `/kaggle/working`, subprocess, fork, socket, a 6 GB allocation, a timeout, per-call timing, and a real task.
- **Cost:** CPU only, about 1 minute, no GPU quota.
- **Not run yet;** it needs the user's go.

## Kaggle CPU probe result (2026-10-14): **PASS**

**Run:** `foysalemonshanto/q27-e10-sandbox-probe-r4` v1. CPU, the pinned Q27 image, about 15 s.
- Outputs are in `Experiment Results/Kaggle Runs/CPU sandbox probe R4 (foysal q27-e10-sandbox-probe-r4 v1)/`.

**R4's own self-test PASSES on Kaggle:** NumPy transform works; a protected-file write, a spawn and a network call are all blocked.

| Case | Result |
|---|---|
| identity, float `np.zeros`, `scipy.ndimage`, `cv2.flip`, cv2 resize + blur on 300×300 | ✅ work |
| read `/kaggle/input` | readable, by design |
| new file, overwrite or delete in `/kaggle/working` (root-owned, mode 755) | ✅ blocked (Errno 13). So **N3 is resolved**: `submission.json` cannot be touched by generated code |
| subprocess, `os.fork`, socket | ✅ blocked (EPERM) |
| 6 GB allocation | ✅ `MemoryError` |
| infinite loop | ✅ killed at 5.0 s |
| **Python thread** | ❌ `RuntimeError: can't start new thread`. **N2 is confirmed** (clone3 returns EPERM, so glibc 2.35 does not fall back). cv2 ops still worked in this test |
| timing | **0.27 s per call**; a real task (all train and test inputs) takes 1.34 s; float output is accepted (`test_usable` True) |

**Remaining items for GPT:**
- **N1:** a failed self-test should not raise.
- **N2:** `clone3` should return ENOSYS, so threads work. This is optional, because the threading need is rare.
- **N4:** the floor should be `min(49152, THINK_BUDGET)`, which is a test-only trap.
