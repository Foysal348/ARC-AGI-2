# E12 step 3.5, rev 2: re-audit request (2026-10-19)

**For:** the two GPT reviewers of the rev-1 zip.

**Ask.** Please check only that the rev-1 findings are really fixed, and that the fixes introduce no new **blocker or bug**. The plan, Duck itself, 2 branches, the milestone settings and the W1 engine stay unchanged. Please cite the file and line for every finding.

## What changed since rev 1

Validation of every rev-1 claim: `E12 step 3.5 - GPT audits validated (2026-10-19).md` (in this zip). Fix table: `PLAN.md` §2e. **Vendored Duck: still 20 files byte-identical to da-fr @ 10882e33.**

| Rev-1 finding | Where the fix is |
|---|---|
| Concurrent `submission.json` writes | builder CELL 6: `_E12_WRITE_LOCK`, `_e12_write_json_atomic` (unique `mkstemp` + `os.replace`) |
| Verification forgeable through `print` | `arc2_duck.py`: `_grid_call_code` + `_verify_submission` (one sandbox per grid, structured `result`, host validates/compares; test grids always run) |
| 900 s first-request HTTP grace | `arc2_duck.py` `Arc2ToolAgent.__init__`: `_http_initial_grace_used = True` |
| Gate slots / setup failure | `arc2_duck.py` `solve_all`: try/finally release, setup failures counted |
| No sandbox memory limit | `arc2_duck.py` `install_sandbox_memory_limit` (prlimit RLIMIT_AS after Popen, before the code is sent; the vendored module's `subprocess` swapped for a proxy); CELL 3 self-test |
| Request snapshots off | CELL 1 `E12_REQUEST_LOGS` (SMOKE1 on), CELL 6 `save_request_logs=`, CELL 7 bundles `duck/*/requests.jsonl` |
| Mirror fallback | `select_attempts`: removed |
| Seeds | PLAN wording only (SGLang 0.5.21 ignores per-request seeds without deterministic mode) |
| Harness not self-contained | `harness/fixtures/` + relative paths |
| 480 threads (GPT 2 B2) | not changed: reproduced as fine (`harness/validate_gpt_findings.py scale`) |

## Tests in this zip (all run on Windows; Linux-only parts noted)

**`python harness/smoke_adapter.py` (39 checks).** Covers:
- the 6 forgery attacks (all rejected);
- verification code with no task data;
- a slow function bounded;
- a tuple-returning correct function accepted;
- grace disabled;
- the fallback without the mirror;
- the memory limit: on Linux, read-back, 5 GiB rejected and 128 MiB ok; on Windows, reported unavailable.

**`python harness/harness_e12.py` (12 scenarios).** Two new scenarios:
- `deadserver`: returns near its deadline;
- `setupfail`: no 120 s wait; the healthy task still solves.

**`python harness/harness_e12_nb.py "Experiment Notebooks/Q27-XHIGH-E12_DUCK_REPL_SGLANG_SMOKE1.ipynb" smoke|rerun|launchfail` (18 / 17 / 6).** Covers:
- 12 concurrent writers × 20 (plus a concurrent reader on Linux);
- request snapshots bundled with the replayed `reasoning_content`;
- the memory-limit evidence.

**`python harness/validate_gpt_findings.py race|spoof|grace|scale`.** The rev-1 reproductions:
- spoof case 4 is now rejected;
- grace: returns in 13 s.

## Questions

1. Can a sandbox child run any model code before `prlimit` is applied? The child reads its code from stdin, and that code is sent only after `Popen.__init__` returns.
2. Is the per-grid verification budget (2 × 30 s, stop after a timeout) safe for legitimate slow programs, compared with rev 1's single 30 s call?
3. With per-grid verification, is there still any path to accept a program that is not exact on every train pair?
4. Any remaining path where `submission.json` is missing or invalid in a hidden rerun?
