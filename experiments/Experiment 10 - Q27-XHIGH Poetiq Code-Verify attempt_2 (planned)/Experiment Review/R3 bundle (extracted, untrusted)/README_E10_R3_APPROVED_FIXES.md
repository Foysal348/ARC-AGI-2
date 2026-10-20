# E10 Poetiq R3 — User-approved three-fix implementation

**Deliverable:** `Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R3.ipynb`

**User authorization:** Three specifically approved modifications to E10 Poetiq R2:
1. Continue refinement when *training passes* but one or more test outputs are missing, malformed, fail to execute, or are invalid ARC grids. Include test execution/format diagnostics in next-iteration feedback. Never use hidden test answers.
2. Allow a single expert's earlier usable programs to provide an alternative `attempt_2` when they yield a distinct grid. Keep Poetiq's expert-best voting results first, in unchanged order. Additional historical candidates are appended and ranked by the existing Poetiq vote/soft-score policy.
3. Implement effective OS-level generated-Python isolation, with filesystem/process/resource/network protections, rather than only a temporary directory and environment-variable filtering.

## Source / architecture freeze

- User E10 86K base `Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb` SHA-256 `530cb119e302899426a2e3f24c5e100279e7167aef77525b6cb93b90a7fc10c0` verified.
- R2 notebook SHA-256 `b0f82d96703779450263b2bd6f0c8e2c944ffbb87edfd04e1ea4da2ed2aecdc4` verified.
- **Only physical code cells 5 (Poetiq core) and 7 (Poetiq scheduler) changed in R3.** Other seven cells including all 86K model/config/load/settings/deadlines/Poetiq prompts/source task selection are source-identical to R2.
- 77,824 thinking, 8,192 answer, context 106,496, Qwen3.8-27B FP8+MTP3, original task ordering, no E9 tail resample, default 1 expert, 10 iterations, three Poetiq prompt variants, voting and feedback mechanisms unchanged.
- User has **not** approved altering the separate inherited minimum-thinking-budget boundary behavior (`MIN_THINK_BUDGET=4096`); R3 does not change it. Long prompts can still receive less than 49,152 thinking. This remains a known user-policy question, not silently patched.
- No automatic extra expert, no hidden model requests solely for attempt_2, no new generation/retry policy or runtime admission cap.

## Exact patches

### Training-pass / test-unusable refinement

`poetiq_verify_code` now reports `passed` (train-only), `test_usable` and `verified_usable`. Unusable test predictions get a short error/format-only feedback section. `e10_commit_iteration` early-stops only when both train passes and all test grids are usable. A nonusable train-perfect program remains in its history for the next refinement, but does not displace usable best programs.

### One-expert historical attempt_2

`E10_STATES[key]['candidate_history']` retains previous verified/evaluated program results. `e10_rank_task` first runs the original Poetiq voting on the best program from each expert, then adds usable history results ranked using the same donor voting. The original best candidate remains `attempt_1`; a different earlier grid may become `attempt_2` without another sampling request. If there is no second distinct usable program, the Kaggle two-attempt schema still duplicates the first grid, as before.

### Actual OS sandbox

Parent uses util-linux `unshare --user --map-root-user --mount --net --pid --fork` to make private user, mount, network and PID namespaces for **every generated program**. The trusted namespace bootstrap chroots into an ephemeral root with only the Python interpreter/runtime libraries mounted **read-only**; no Kaggle input/model/checkpoint/notebook/user-storage path is mapped into the chroot. The runner imports NumPy/SciPy, installs a `libseccomp` syscall filter prohibiting process creation, exec, sockets, mount/unshare, ptrace and related operations, and enforces CPU, memory/address-space, open-file, output-file, and child-process limits. Parent enforces wall-clock timeout and kills the entire process group. Stdout/stderr are bounded and kept in ephemeral scratch. No Kaggle API, GitHub or Internet access is used at runtime. General Python NumPy/SciPy/OpenCV transformations remain available when supported by the installed environment.

**Compatibility caveat:** This sandbox requires Linux user/mount/network/PID namespaces, util-linux `unshare` + `mount`, and `libseccomp.so.2`. They work in the current Linux CPU test container; **Kaggle's final 4×L4 container has not been tested**. If it disallows user namespaces or mount namespaces, sandbox execution returns an explicit error; **it does not secretly run unrestricted code**. The requested experiment's Kaggle readiness remains conditional on verifying these OS capabilities. No process sandbox is an absolute security guarantee against kernel vulnerabilities; these tests establish intended access boundaries for common untrusted-code behaviors.

## How to reproduce CPU checks

Extract ZIP, then (using Python with NumPy installed):

```bash
python build_e10.py
python test_e10_cpu.py
python test_e10_r2_regressions.py
python test_e10_r3_approved_fixes.py
```

The R3 suite verifies training-perfect test exception and >30-row outputs, continued refinement from execution/format error to valid program in a full mock-engine loop, true one-expert second-attempt diversity without changing the first, and adversarial attempts at reading/writing outside the sandbox, writing to runtime libraries, spawning subprocesses, creating sockets and infinite loops. NumPy/SciPy/OpenCV and generated debug prints were also checked locally. All syntax/schema and SHA parity checks passed.

## What was NOT tested

- No actual 4×L4 GPU model inference or Kaggle submission.
- Kaggle namespace/seccomp availability remains unverified.
- No evidence of real ARC accuracy improvement or runtime feasibility from these CPU tests.

## Approved scope only

The R3 notebook is an E9-86K-derived *Poetiq-only solver*, not the original direct-grid primary with a sidecar. The user explicitly approved replacing that architecture earlier. Neither direct-grid fallback nor Codex Ordering/Adaptive KV has been added.

**R3 notebook SHA-256:** `4418ebbf5ccc1665f9dff9ee1e6c19a649e4bd23e913fa635bbfc4fb0a898853`
