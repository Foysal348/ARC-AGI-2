# E12 step 3.5: the two GPT audits, validated (2026-10-19)

**Method.**
- **No project code was changed.** Every claim was checked in the source.
- **Tests:** the testable claims were run with `_scratch/session_1c6db7aa/e12_validate/validate_gpt_findings.py`, using the real adapter, the real Duck sandbox and scripted test code.

| # | Claim (source) | Verdict | Evidence |
|---|---|---|---|
| 1 | Concurrent `submission.json` writes race on one `.tmp` (both) | **VALID, reproduced** | 12 writers × 40 writes: on Windows 478 writes failed with PermissionError and a reader saw a broken file 6 times; GPT saw FileNotFoundError on Linux. The final write (after `solve_all`, single thread) is valid, so the risk applies only if the notebook dies before that final write. |
| 2 | A wrong program can be accepted by monkeypatching `print` (GPT 1) | **VALID, reproduced; practical risk ~0** | Case 4: a wrong function that patches `__builtins__['print']` and emits the hidden marker plus the true train outputs was **accepted**. An accidental override (4b), a forged marker printed by the snippet (3) and a plain wrong program (2) are all rejected. The attack needs the marker string (never shown to the model) and the outputs. GPT 1's patch sets the test outputs to None when train fails, which would remove the plan's "best partial" fallback (Rev 1 #6). |
| 3 | 900 s first-request HTTP grace ignores the deadline (both) | **VALID, reproduced** | Dead server with a 5 s deadline: the branch was **still running after 90 s**. The grace covers only connection errors / 5xx / 429 on each agent's first request; a ReadTimeout is not retried (`tool_agent.py:629-631`). If SGLang dies late in a hidden run, the run can overrun by up to 15 min and lose the final report. |
| 4 | The gate wait is unbounded (GPT 1) | **Partly** | At the deadline active branches exit and release; each queued branch is then admitted and exits at once (`should_stop` comes first). It hangs only when an active branch is stuck, i.e. #3. |
| 5 | 480 worker threads exceed limits / need a bounded pool (GPT 2) | **NOT reproduced** | 240 tasks × 2: 480 branches, peak 495 threads, all accepted in 102 s, ≤ 12 in flight, all first branches before any second. The held-slot release has no try/finally (minor: a branch that fails during setup delays the release by ≤ 120 s). |
| 6 | No sandbox memory limit (both) | **VALID (known)** | Pending the user's decision; proposal: `resource.prlimit` on the child after Popen, before the code is sent (no `preexec_fn`). |
| 7 | Duck request logs are off (GPT 1) | **VALID, diagnostic gap** | `transcript.txt` holds the thinking but not the exact messages sent, so the preserved-thinking replay and trimming cannot be checked from it. |
| 8 | Different branch seeds not implemented (GPT 1) | **VALID wording; no effect** | SGLang 0.5.21 ignores the per-request seed without `--enable-deterministic-inference` (E11 CELL 5 note). Diversity comes from T 0.7. Fix the PLAN wording only. |
| 9 | Mirror fallback in `select_attempts` (GPT 2) | **VALID deviation** | My addition: when no distinct second grid exists, attempt_2 = the test input mirrored horizontally. Not in the approved plan text. |
| 10 | Whole-notebook harness not self-contained (GPT 2) | **VALID** | It needs the E11-05 fixtures (`beh_e11_05.py`, tokenizers) and uses absolute Windows paths. |

GPT 2 rated host verification GO. GPT 1's spoof (#2) contradicts that, and it reproduces.
