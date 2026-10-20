# Q27 E10 Poetiq R4 — Confirmed Bug Fix Bundle

## User scope
Only verified defects from the independent R3 audit were repaired. This is **Poetiq-only** E9-derived 86K inference (not a direct-grid baseline sidecar). No unapproved change to the model, 77,824 thinking configured cap, 8,192 answer allowance, 106,496 context, original task order, temperature, 1/2/8 expert support, 10 iterations, voting, prompt variants, or base installation/engine was made. Kaggle submission/GPU never executed by the author.

## Source hashes
- E10 base: `530cb119e302899426a2e3f24c5e100279e7167aef77525b6cb93b90a7fc10c0`
- R3 prior: `4418ebbf5ccc1665f9dff9ee1e6c19a649e4bd23e913fa635bbfc4fb0a898853`
- **R4 corrected:** `596bf694ce65a4d2360051b5c065d996df4c9851e5d09ebb0bcad03c3d597cce`
- Only physical code cells **5 and 7** differ from R3.
- Physical base cells 1, 3, 6 are byte-for-byte source-identical.

## Confirmed fixes
1. **BLOCKER — LOG_EVERY_SEC missing:** restored `LOG_EVERY_SEC=120.0` in Cell 7. A regression test executes the fake-engine notebook scheduler **without injecting LOG_EVERY_SEC**.
2. **BLOCKER — Kaggle `unshare` unavailable:** replaced namespace/mount-dependent runner with a Kaggle-oriented Poetiq subprocess using root→UID 65534 irreversible drop, `PR_SET_NO_NEW_PRIVS`, `libseccomp` restrictions (network sockets, exec/spawn, ptrace, mount), CPU/AS/FSIZE/NOFILE/NPROC rlimits, bounded stdout/stderr, a new process group and parent-enforced timeout/kill. Added an **automatic pre-engine startup positive/negative self-test** in Cell 5; it fails loudly rather than secretly bypassing restrictions. Does not call `unshare`, `mount` or require CAP_SYS_ADMIN.
3. **HIGH — valid NumPy integral floats rejected:** lossless normalization accepts integer-valued float and np integer, rejects fractional, NaN, bool, strings and values outside ARC palette 0–9. Both train and test verification use this conversion. Prior R2's lossy 1.99→1 false PASS remains blocked.
4. **HIGH — long feedback violates 49,152 minimum thinking allowance:** keep the configured 77,824 maximum; if the actual tokenized prompt would push the effective thinking cap below 49,152, progressively remove the **lowest scored** prior feedback solutions until the prompt fits. Keep the expert's full candidate history intact. If even a prompt without feedback is too large, explicitly skip that request rather than secretly reducing thinking below the floor. No global concurrency or expert/iteration changes.

## Explicit security limitation
**This is not a mount-namespace/chroot/landlock filesystem sandbox.** In the Kaggle probe namespaces are prohibited; the new runner uses **UID privilege separation** so it cannot modify root-owned notebooks or checkpoints and seccomp prevents sockets and child processes. **World-readable host files may still be readable**, including `/kaggle/input`. The user explicitly requested an experimental, Kaggle-compatible runner. Do not describe this as fully filesystem-isolated or claim it protects world-readable secrets. The untrusted program inherits no parent secrets/environment variables, sees no GPU via CUDA_VISIBLE_DEVICES, and has no network socket syscall permission. Kaggle itself has NOT yet run this R4 startup self-test; that remains the operational compatibility gate.

## CPU tests
- `test_e10_cpu.py`: PASS, fake vLLM program/refinement, original architectural tests.
- `test_e10_r2_regressions.py`: PASS, invalid output, debug printing, no environment inheritance.
- `test_e10_r4_no_injected_logger.py`: PASS, original fake-engine runner **without injected LOG_EVERY_SEC**.
- `test_e10_r4_targeted.py`: PASS, decimal-normalization adversarial tests; accepted integral float NumPy program, denied subprocess/socket/fork/root-owned file write, hard timeout, source parity and thinking-floor/pruning synthetic tests.
- Legacy `test_e10_r3_approved_fixes.py` is preserved only for provenance: it assumes full namespace filesystem-read isolation and therefore **is not** the expected passing security suite for R4. Its replacement is `test_e10_r4_targeted.py`.

## To reproduce
```
python build_e10.py
python test_e10_cpu.py
python test_e10_r2_regressions.py
python test_e10_r4_no_injected_logger.py
python test_e10_r4_targeted.py
```

## What is not established
- **No real Kaggle CPU R4 or 4×L4 GPU run:** UID/capabilities, syscall filter, numpy/scipy/cv2 and performance must still be verified inside the exact competition image.
- Real 86K+iterated Poetiq throughput, accuracy, cache pressure, candidate diversity and completing 12h constraints are unknown.
- The R3 donor prompts were abbreviated relative to Poetiq upstream. **Not modified in R4** because this is not one of the user-approved targeted fixes and would change experiments.
- `EXPECTED_LIVE_TOKENS_PER_SEQ` and `MAX_NUM_SEQS` were NOT silently changed; KV oversubscription is a GPU performance risk to observe, not a reproduced source blocker.

## Verdict
**R4 local CPU GO; Kaggle environment validation pending.** No generated artifact is claimed GPU validated.
