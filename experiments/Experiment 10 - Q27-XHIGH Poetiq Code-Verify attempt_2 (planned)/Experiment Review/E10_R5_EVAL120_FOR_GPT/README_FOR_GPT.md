# E10 Poetiq R5 + EVAL120: pre-GPU audit before a 12-hour full evaluation run (2026-10-14)

**What we ask:**
- **Blockers and real bugs only.** That means anything that would crash the 12 h run, waste GPU time, lose answers, score wrongly, write an invalid `submission.json`, or break the hidden rerun.
- Please verify against the files.
- Do not propose new features or design changes; the design is the user's decision and is fixed.
- If you find nothing, say "GO".

## The notebook

| | |
|---|---|
| **To audit** | `notebook/Q27-XHIGH-v1_4_4_E10_86K_POETIQ_R5_EVAL120.ipynb`, sha256 `498109b72382d3ab…` |
| Source | `notebook/E10_POETIQ_R4T_SAFE_SANDBOX_R5.ipynb`, sha256 `68b0d28f…`. This is your R5: R4 + full-thinking logging + a non-raising sandbox self-test + clone3 → ENOSYS |
| Builder | `build/build_e10_r5_eval120.py` (deterministic) |
| **Only change vs R5** | cells 0 and 2 (`R5_TO_R5_EVAL120.diff`): a new profile **EVAL120**, made the commit-run default |
| EVAL120 | all 120 evaluation tasks / 172 test inputs, scored. The hidden-run clock: budget `12 h − 10 min`, `admit_frac` 0.90, `drain_frac` 0.94, `hard_margin` 6 min, `verify_margin` 3 min, `drain_margin` 8 min, `admit_margin` 2 min. Those are the RERUN_ALL values, giving admit until 694 min, drain at 696 and hard stop at 704 |
| Kept as R5 | 77,824 thinking + 8,192 answer, `MAX_MODEL_LEN` 106,496, 1 expert, up to 10 iterations, temperature 1.0, the Poetiq prompts and voting, the full-thinking log (`q27x/poetiq_thinking/`), the UID-drop + seccomp sandbox, the 16-sequence engine |

**Expected behaviour:**
- Round 1 is about 120 requests of roughly 60 min each over 16 slots, so about 7.5 h.
- The remaining time goes to feedback iterations on tasks whose programs did not pass training.
- The notebook runs until the clock.
- About 23 quota-hours (4×L4 is charged 2×). It runs on a teammate account.

## Questions (please check)

1. Does EVAL120 work end to end for 120 tasks / 172 inputs? Check the report and scoring for multi-input tasks, the iteration records, and the thinking files.
2. Is anything unsafe near the end of the clock? Check the drain of active Poetiq requests, the hard stop, the verification harvest window, the final `e10_update_task_submission`, the report writing and the zip bundle. The run must stay inside Kaggle's 12 h session.
3. **Disk and memory over 12 h.** Full thinking per iteration is about 0.3 MB. With about 120 + ~60 iterations, check the thinking files, `poetiq_programs/`, the in-memory `candidate_history` and the `poetiq_iterations.jsonl` growth.
4. **The hidden rerun is unchanged** (`RERUN_ALL`): the default profile only affects the commit run.
5. Anything in R5 itself that would break at this scale. R5 was only GPU-tested as R4 on EVAL12.

## Evidence already gathered

- **Kaggle CPU probe of R5's sandbox:** PASS (`cpu_validation/kaggle_sandbox_probe_r5.json`).
  - Threads work. subprocess, fork and socket give EPERM.
  - Writes and deletes in `/kaggle/working` are blocked. 6 GB allocation gives `MemoryError`. The infinite loop is killed at 5 s.
  - 0.27 s per call.
- **R4 on GPU, EVAL12** (same code as R5 apart from the two R5 fixes):
  - 4/22; 0 errors; 0 preemptions; KV max 79 %.
  - 27 iterations; sandbox self-test PASS on the GPU container.
  - Details: `context/GPU EVAL12 Poetiq R4 deep analysis (2026-10-14).md`.
- **Local fake-vLLM harness** (`build/harness_r5_probe.py`; the real notebook cells, the real Qwen tokenizer and the real competition files; on Windows the Linux-only runner is replaced by `build/r4_sandbox_stub.py` after cell 5 runs):

| Scenario | Log | Result |
|---|---|---|
| A. EVAL120, runner stubbed | `cpu_validation/A_mixed_eval120_stub.txt` | profile EVAL120, `120 tasks, 172 queries`; 1,180 iterations; solver complete; report `eval … / 172`; no traceback. The 2 `skipped_context` and 272 `feedback_trims` are harness artifacts: the harness reserves a 40,000-token answer, so prompts hit the 49,152 floor earlier |
| B. Real R5 runner, which fails on Windows (no `geteuid`) | `cpu_validation/B_selftest_fail_path.txt` | self-test `{'ok': False}`, `ENGINE INVALID - keeping the identity submission: sandbox_selftest_failed`, 0 requests, report written, **no exception** |
| C. Hidden rerun (240 tasks), runner stubbed | — | still running at zip time; will be sent separately |

## Rules (fixed)

- Never below 49,152 thinking tokens per request.
- A valid `submission.json` always.
- Simple changes, one at a time.
- Do not run, push or submit anything.
