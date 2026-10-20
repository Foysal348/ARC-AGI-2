# E12 step 3.5 rev 2: the two GPT re-audits, validated (2026-10-19)

**Method.**
- **No project code was changed.**
- **Tests:** `_scratch/session_1c6db7aa/e12_validate/validate_rev2_findings.py`, run with the real adapter, the real Duck sandbox and scripted test code.

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| H1 (GPT 1) | If `prlimit` fails, model code still runs without the limit (fail-open) | **VALID, reproduced** | A fake `resource` whose `prlimit` raises PermissionError → `MODEL_CODE_EXECUTED` printed; `SANDBOX_LIMIT` failed=1. Fix: kill the child and raise `OSError` in `_LimitedPopen`. Duck already turns an `OSError` at Popen into "Sandbox process could not start." (`python_tool_sandbox.py:818-842`). Optionally also gate CELL 3 on Linux. |
| H2 (GPT 1) | A verification started just before the deadline (≤ 60 s) plus finalisation can pass HARD_STOP | **VALID, low risk** | RERUN_ALL: TOTAL_BUDGET = 12 h − 3 min and hard margin 2 min, so HARD_STOP = 11:55 and the E12 deadline = 11:54. Worst case, the verification ends ≈ 11:55 and CELL 7 then needs seconds (the server log is < 1 MB per 4 h; transcripts are ~100s of MB). Kaggle's limit is 12:00, so ~4-5 min remain and nothing in E12 kills at HARD_STOP. Raising `E12_STOP_BEFORE_HARD_SEC` 60 → 180 costs ~2 min of solving and removes the doubt. |
| BLOCKER (GPT 2) | A wrong function patches `json.dumps` so the sandbox's `final` payload carries the true outputs → accepted | **VALID, reproduced; no new capability** | `JSONSPOOF: accepted=True`. An honest lookup-table function (`LOOKUP`) is accepted the same way, because the train outputs are in the prompt. Train-exact therefore never proved generalisation, and forging gains nothing over hardcoding; the test outputs still come from the function itself. An accidental `json` patch breaks the result, it does not forge it. GPT 2's guard covers `json.dumps` only; `json.encoder` / `JSONEncoder` internals stay reachable in the same process. |
| MEDIUM (GPT 2) | The 60 s budget can reject a legitimately slow program | **VALID (trade-off, not a regression)** | Rev 1 had 30 s for all grids together. Proposal: record `verify_sec` / `verify_timeouts` per submission for EVAL30. |
| extra test (GPT 1) | Mixed case: 2 exact + 1 timeout → must be rejected | sensible | Add to smoke. |
| 480 threads | — | both accept (no blocker) | — |
