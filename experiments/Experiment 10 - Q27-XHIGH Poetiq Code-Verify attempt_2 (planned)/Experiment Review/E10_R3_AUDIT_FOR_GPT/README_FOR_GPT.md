# E10 Poetiq R3: audit results and requested fixes (2026-10-14)

**To:** GPT, author of `Q27_E10_POETIQ_R3_APPROVED_FIXES_ALL_IN_ONE.zip`.
**From:** Claude's audit of R3 (notebook sha256 `4418ebbf…`).

**Verdict: NO-GO for GPU.**
- **Two blockers.** Each one alone makes the run score about 0.
- **Three high-severity issues.** They cost accuracy or break the user's thinking-budget rule.

Every finding below comes with evidence you can reproduce. Please verify each one yourself before fixing it.

## Blockers

### B1. `LOG_EVERY_SEC` is undefined, so the solver dies at the first step

- **Cause:** the base defined `LOG_EVERY_SEC = 120.0` in cell 7. R3 replaced that cell, and the scheduler still reads the name (`poetiq_scheduler.py:259`).
- **What happens:**
  - the `NameError` is raised right after the first `engine.step()`;
  - the `except` block aborts every request;
  - `finally` writes the identity submission.
- **Evidence:**
  - `evidence/r3_probe_shipped.txt` shows the real R3 notebook cells run in our fake-vLLM harness (`evidence/harness_r3_probe.py`): 12 requests, 1 step, 0 iterations, `NameError: name 'LOG_EVERY_SEC' is not defined`.
  - `evidence/r3_probe_injected.txt` is the same run with the name injected: 120 iterations, and the solver completes.
- **Why your tests passed:** they inject the name into the namespace (`test_e10_cpu.py:81` `'LOG_EVERY_SEC': 5`, `test_e10_r3_approved_fixes.py:178`).
- **Requested:**
  1. Define it in cell 7.
  2. Make the CPU tests execute the real notebook cells **without injecting any global** that the notebook does not define.
  3. Add a static check: no name may be used before some cell defines it.

### B2. The sandbox cannot run on Kaggle: confirmed by a Kaggle probe

- **The run:** `foysalemonshanto/q27-e10-sandbox-probe` v1, on CPU, with the **same pinned docker image** as the Q27 GPU runs. It embeds R3's cell 5 verbatim (`sandbox_probe/`; the Kaggle output is in `sandbox_probe/kaggle_output/`).
- **Results:**
  - `unshare --user --map-root-user id` fails with `Operation not permitted`. So does every larger flag set R3 uses.
  - Every `poetiq_sandbox_run` call fails in 0.003 s.
  - `poetiq_verify_code` on a real evaluation task returns score 0 and `test_usable` False.
  - The verdict is `FAIL_SANDBOX_UNAVAILABLE`.
- **What the container allows:**
  - Blocked: AppArmor `docker-default (enforce)` and seccomp mode 2 are active.
  - Capabilities (`CapEff 00000000a80425fb`): CAP_SETUID, CAP_SETGID and CAP_SYS_CHROOT are **present**; **CAP_SYS_ADMIN is absent**.
  - Available: uid 0, Python 3.12.13 at `/usr/bin/python3`, 4 CPUs, `libseccomp.so.2`, numpy 2.0.2, scipy 1.16.3, cv2 4.13.0.
  - A plain subprocess that imports numpy and scipy takes 0.26 s on average.
- **Consequence:** R3 is Poetiq-only, so every task falls back to the identity grid and the score is about 0.
- **Requested (needs the user's approval of the design):** a runner that works without namespaces or mounts, in the spirit of Poetiq's original `sandbox.py`. Proposed hardening, all possible without CAP_SYS_ADMIN:
  - a subprocess in a private temporary directory, as its own process group, with a parent wall-clock timeout that kills the group;
  - rlimits: CPU, address space, file size, open files, no core dumps;
  - `PR_SET_NO_NEW_PRIVS` plus a libseccomp filter blocking execve, execveat, fork, vfork, clone or clone3 for new processes (threads must stay allowed, or NumPy/OpenCV break), socket and connect, ptrace, mount and unshare;
  - `setgid` / `setuid` to 65534 (nobody) before running untrusted code, so root-owned `/kaggle/working` (where `submission.json` lives) is not writable;
  - **a startup self-test in the notebook** (identity program, a blocked write, a timeout). If it fails, the notebook must say so loudly in the log and the report. Do not fail silently into identity submissions.

## High

### H1. Integral float grids are rejected

- `poetiq_valid_grid` requires `type(v) == int`.
- A common LLM pattern like `out = np.zeros((h, w))` returns floats from `.tolist()`, so a **correct** program fails every training example as "Invalid or unavailable grid".
- Poetiq's original does `np.array(obj).astype(int)`.
- R2 removed the coercion to avoid lossy casts.
- **Requested:** accept values that are integral and within 0–9 (floats where `v == int(v)`, NumPy integers, and bools only if you decide so). Reject non-integral values and values outside 0–9. Apply the same rule to train and test.

### H2. Feedback prompts push thinking below the user's floor

- The user's rule is **never less than 49,152 thinking tokens per request** without approval.
- `phase1_cap = min(77824, 106496 - prompt - 8192 - 256)`, and `MIN_THINK_BUDGET = 4096` only skips requests below 4k.
- **Measured** with the real Qwen tokenizer and chat template on all 120 evaluation tasks (`evidence/measure_feedback_prompt_tokens.py`), with 5 feedback solutions of ~1.2k-token code in R3's own diff format:

| Scenario | Median prompt | Max | Thinking < 77,824 | Thinking < 49,152 | Skipped |
|---|---|---|---|---|---|
| Iteration 1 (no feedback) | 6,279 | 17,617 | 0/120 | 0 | 0 |
| 5 solutions, 20 % wrong cells | 20,875 | 67,970 | 63/120 | **4** | 0 |
| 5 solutions, all cells wrong | 26,527 | 96,778 | 82/120 | **10** | 1 |

- **Requested:** before admitting a request, drop the lowest-scored feedback solutions (keeping Poetiq's improving order) until the prompt is ≤ 48,896 tokens (thinking ≥ 49,152). Never admit a request with less than 49,152 thinking. If even the best single solution does not fit, send the request without feedback.
  - The user decides whether the target is ≥ 49,152 or the full 77,824.

### H3. KV oversubscription

- `EXPECTED_LIVE_TOKENS_PER_SEQ = 43,000` (giving concurrency 16) was calibrated on DIRECT prompts of 2–7k tokens. E9 already reached 91.4 % KV with those.
- Feedback prompts have a median of ~21k tokens, plus up to 77.8k of thinking, so 16 sequences will cause preemption and recomputation.
- **Requested:** admit by an estimate of live KV tokens (prompt + cap) against the engine's `kv_tokens`, or report and justify the concurrency for feedback iterations.

## Design points (for the user, not for you to change)

- **D1. R3 is Poetiq-only, with no DIRECT attempt.** Earlier code-only runs scored R6 2/22 and v1.9.1 1/22, against DIRECT's 2–4/22 on EVAL12.
- **D2. Temperature 1.0** (the Poetiq default). Every Q27 run so far used 0.6.

## Fixed constraints (user rules)

- Base: `Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb`, sha `530cb119…`. Cells 1, 3 and 6 must stay identical.
- Thinking 77,824 + answer 8,192; never below 49,152 thinking without approval.
- A valid `submission.json` must always be written.
- Change only what is listed; keep the code simple.

## Files

| Path | What |
|---|---|
| `R3 deep audit (Claude, 2026-10-14).md` | the full audit |
| `evidence/harness_r3_probe.py` | the fake-vLLM harness used for B1 (a patched copy of our `harness_e10.py`) |
| `evidence/r3_probe_shipped.txt`, `r3_probe_injected.txt` | B1: R3 as shipped, and with the name injected |
| `evidence/measure_feedback_prompt_tokens.py` | H2 measurement script |
| `sandbox_probe/build_sandbox_probe.py`, `E10_SANDBOX_PROBE_CPU.ipynb` | B2: the probe notebook (R3 cell 5 verbatim) |
| `sandbox_probe/kaggle_output/sandbox_probe.json`, `q27-e10-sandbox-probe.log` | B2: the Kaggle results |
