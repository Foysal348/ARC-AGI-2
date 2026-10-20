# E10 Poetiq R3: deep audit (Claude, 2026-10-14)

**Bundle:** `Q27_E10_POETIQ_R3_APPROVED_FIXES_ALL_IN_ONE.zip`, extracted to `R3 bundle (extracted, untrusted)/`.
- Notebook: `Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R3.ipynb`, sha `4418ebbf…` (matches the bundle README).
- Base sha `530cb119…` verified. Cells 1, 3 and 6 are unchanged from the base; cells 2, 4, 5, 7 and 8 were replaced or extended.

**Method:**
- Read all of `poetiq_core.py` and `poetiq_scheduler.py`, and the build script.
- AST check of names that are used before any cell defines them.
- Ran the real notebook cells in our fake-vLLM harness (`harness_e10.py`, patched so that the fake model returns a `transform` code block).
- Measured prompt tokens with the real Qwen tokenizer and chat template, on all 120 evaluation tasks.

## Blockers

| # | Problem | Evidence | Fix |
|---|---|---|---|
| **B1** | **`LOG_EVERY_SEC` is undefined.** The base defined it in cell 7 (`LOG_EVERY_SEC = 120.0`), and R3 replaced that cell. The solver dies with a `NameError` right after the first `engine.step()`. The `except` turns this into an identity submission, so the **score is 0**. | Our harness run of the shipped notebook gave `NameError: name 'LOG_EVERY_SEC' is not defined`: 12 requests, 1 step, 0 iterations recorded. **The bundle's own tests inject `'LOG_EVERY_SEC': 5 / 50` into the namespace** (`test_e10_cpu.py:81`, `test_e10_r3_approved_fixes.py:178`), which is why they passed. | Add one line in cell 7: `LOG_EVERY_SEC = 120.0`. Make the tests run the real cells without injecting names. |
| **B2** | **The sandbox fails closed and is unverified on Kaggle.** Every program needs `unshare --user --mount --net --pid` plus `mount --bind` and libseccomp. Container runtimes often block mount-namespace operations; the bundle itself says Kaggle is not tested. If any part fails, every verification fails. Because R3 is **Poetiq-only** (no DIRECT), every task then falls back to the identity grid, so the **score is about 0**. | In our harness on a machine without `unshare`: 120 iterations, **0 train passes, eval 0/22**, and no warning appears anywhere in the run. | **Before any GPU run:** (a) a short Kaggle **CPU** probe notebook that runs `poetiq_sandbox_run` on a trivial program; (b) a startup probe inside the notebook. If the probe fails, there must be a fallback runner (Poetiq's original subprocess with timeout and rlimits; Kaggle submissions have no internet anyway), or at least a loud abort. Both are the user's decision. |

## High

| # | Problem | Evidence | Fix |
|---|---|---|---|
| H1 | **Float grids are rejected.** Both train and test require `type(v) == int`. `np.zeros((h, w))` without `dtype=int` is very common in LLM code: its `.tolist()` gives floats, so a correct program fails as "Invalid or unavailable grid". Poetiq's original converts with `astype(int)`. R2 removed that to prevent lossy coercion. | `poetiq_valid_grid`; `poetiq_verify_code` lines 141–150; the R2 regression "lossy_float". | Accept integral floats (`v == int(v)`) and NumPy integers. Reject only non-integral values and values outside 0–9. |
| H2 | **The thinking floor is broken in feedback iterations.** The feedback holds up to 5 solutions, each with code and full `pred/correct` grids for every training example, so the prompts grow. `phase1_cap` then shrinks thinking; `MIN_THINK_BUDGET = 4096` skips only below 4k. This violates the user's rule against going below 49,152. | Real tokenizer, 120 eval tasks, 5 solutions of ~1.2k-token code: see the table below. | Before admitting, drop the lowest-scored solutions until the prompt is ≤ 48,896 tokens (thinking ≥ 49,152), or ≤ 20,224 (full 77,824). Which limit to use is the user's decision. |
| H3 | **KV and concurrency.** `EXPECTED_LIVE_TOKENS_PER_SEQ = 43,000` was calibrated on DIRECT prompts of 2–7k tokens. Feedback iterations have median prompts of ~21k, plus up to 78k of thinking. With 16 sequences, the KV cache will be oversubscribed, causing preemption and recomputation. | E9 already peaked at 91.4 % KV with short prompts. | Admit by live KV tokens (prompt + generated), or lower the concurrency for feedback iterations. Measure on the GPU. |

**Prompt tokens with feedback** (iteration 1 has no feedback):

| Scenario | Median | Max | Thinking < 77,824 | Thinking < 49,152 | Skipped |
|---|---|---|---|---|---|
| Iteration 1 | 6,279 | 17,617 | 0 / 120 | 0 | 0 |
| 5 solutions, 20 % wrong cells | 20,875 | 67,970 | **63 / 120** | **4** | 0 |
| 5 solutions, all cells wrong | 26,527 | 96,778 | **82 / 120** | **10** | 1 |

## Design points (user decisions, not bugs)

- **D1. R3 is Poetiq-only.** There is no DIRECT attempt_1. The E10 README plan had attempt_1 = DIRECT (never changed) and Poetiq in attempt_2.
  - The bundle says the user approved replacing DIRECT.
  - **Our own evidence:** code-only runs scored R6 2/22 and v1.9.1 1/22, against DIRECT's 2–4/22.
- **D2. Temperature 1.0** (the Poetiq default) applies to all generation. It was 0.6 in every Q27 run so far, and 1.0 is untested with Qwen xhigh.
- **D3. Hidden-run time.** The first iteration over 240 tasks costs about the same as a full DIRECT run at 86k. 86k has never run on the hidden set.

## Checked and fine

- **Base parity:** cells 1, 3 and 6 are identical to the base.
- **Queue:** round-robin, so iteration 1 runs for every task before any task gets iteration 2.
- **Stale outputs:** request ids carry the iteration number, so stale outputs are ignored.
- **Lifecycle:** the drain, the hard stop and the final harvest are handled.
- **Submission:** written after every iteration and in `finally`; identity in eval mode, Poetiq grids in the rerun.
- **No answer leaks:** test feedback carries only execution and format errors, never test answers.
- **Voting:** follows Poetiq's `solve_parallel_coding`. History candidates only extend attempt_2.
- **Sandbox design:** fine where namespaces are available.

## Minor (ignored per the user)

- A `transform` returning a list of NumPy scalars fails JSON output.
- The code regex accepts only ```` ```python ````.
- Any `add_request` exception aborts the whole solver.
- The 5 s sandbox timeout includes namespace setup and the NumPy/SciPy import, so it may cause false timeouts under CPU load. Measure it on Kaggle.

## B2 confirmed on Kaggle: CPU sandbox probe (2026-10-14)

**Run:** `foysalemonshanto/q27-e10-sandbox-probe` v1. CPU, the pinned Q27 docker image, no internet. It embeds R3's core cell verbatim.
- Builder: `Experiment Notebooks/Build/sandbox_probe/build_sandbox_probe.py`.
- Outputs: `Experiment Results/Kaggle Runs/CPU sandbox probe (foysal q27-e10-sandbox-probe v1)/`.

**Verdict: `FAIL_SANDBOX_UNAVAILABLE`.**
- **Every** `unshare` fails with `Operation not permitted`, even the user namespace alone.
- Every R3 sandbox call fails in 0.003 s, so every program, including a correct identity program, scores 0.
- A real task through `poetiq_verify_code` gives score 0 and `test_usable` False.

**Why:**
- The container runs with AppArmor `docker-default (enforce)` and a seccomp filter (`Seccomp: 2`).
- It has the default Docker capabilities (`CapEff a80425fb`): CAP_SYS_CHROOT, CAP_SETUID and CAP_SETGID are present; **CAP_SYS_ADMIN is not**, so there are no namespaces and no mounts.

**What does work there:**
- Python 3.12.13 (`/usr/bin/python3`), running as root (uid 0); 4 CPUs.
- `libseccomp.so.2` is present; numpy 2.0.2, scipy 1.16.3 and cv2 4.13.0 are installed.
- A plain subprocess that imports numpy and scipy takes **0.26 s** on average (max 0.31 s).

**A fallback that can work on this container (user decision):** a plain subprocess, as in Poetiq's original sandbox, hardened with:
- the parent's wall-clock timeout plus a process-group kill;
- rlimits (CPU, address space, file size, open files);
- `PR_SET_NO_NEW_PRIVS` plus a libseccomp filter that blocks execve, fork, clone and socket (both work without CAP_SYS_ADMIN);
- dropping to uid/gid 65534 (nobody), so that `/kaggle/working` (owned by root) cannot be written;
- a private temporary working directory.

**Caveat:** the GPU container and the hidden-rerun container were not probed. They use the same Kaggle docker image and Docker defaults, so the same result is expected.
