# EVO24 — response to review round 3 (NO-GO: 1 HIGH engine-failure bug; 2 MEDIUM; 2 LOW)

Every finding was verified before fixing (code read, arithmetic, file bytes). All confirmed. Two further gaps of the
same kind were found during that check and fixed in the same patch (marked **own**). Notebook after the fixes:
`PIV-Q27-EVO24-EVOLUTION.ipynb` sha256 `5ea755174524282518efa4ce8326e20baf7ccf4b6f087ac1e3e014cb45e0ab37` (LF only).

| # | finding | verified | fix | test |
|---|---|---|---|---|
| HIGH | `arc_llm.generate()` exception or short output list uncaught → cell 8 crashes, no report/bundle | no try/except in cell 8; `evo_generate` called bare | `_evo_safe_generate()` wraps both phases: exception or `len(out) != len(prompts)` / empty `outputs` → recorded in `EVO_ENGINE_FAILURES`, `evo_generate` returns False, the round is logged as an engine failure, `run_evo` stops with `engine_error`, verdict **INVALID / ENGINE FAILURE**, failures listed in the report | harness `HARNESS_EVO_ENGINE=p1`, `=p2`, `=short`: each run completes, cell 9 runs, bundle exists, VERDICT INVALID / ENGINE FAILURE (`h4_p1/p2/short.txt`); report shows phase, type, error |
| own | an exception inside `_evo_verify` (thread pool) would also crash cell 8 | no try in `_evo_verify` | `_evo_verify` wraps `_evo_verify_inner`; an exception becomes a recorded infrastructure failure → INVALID | self-test: verify with `task=None` raises inside, recorded as infra failure |
| own | any other unexpected exception in `run_evo`, or in cell 9 scoring, would lose the evidence | no catch-all | cell 8: catch-all → `stopped: evo_exception`, **INVALID / EVO EXCEPTION** with traceback; cell 9: scoring wrapped, the bundle is always written | self-test of the verdict; code |
| MEDIUM | train-exact candidate that fails on a test input was excluded from repair (only an alternative was asked) | `exclude = {all exact}` | exact candidates split; if any exact candidate fails on a test input, slot 1 repairs it with the verifier report, which contains `TEST INPUT FAILURE: …` | self-test: repair prompt for such a candidate contains TEST INPUT FAILURE; harness pos-5 scenario (train-exact lookup raising on test): 6 such repair prompts observed, `exact_partial_repairs: 6` |
| MEDIUM | 4+2+2 calls not guaranteed under the 32,768 cap (8 × 5,120 = 40,960) | arithmetic | wording: "4 initial candidates, then up to two 2-slot follow-up generations subject to the 32,768-token hard cap" (notebook header, reviewer prompt); report: `tasks_with_8_calls`, `tasks_with_7_calls`, `tasks_with_6_or_fewer_calls`, `budget_skipped_calls` | `h4_normal.txt` (scripted answers: 24 / 0 / 0, 0 skipped) |
| LOW | notebook SHA depends on OS line endings | 4,304 CRLF in the file | builder writes bytes with LF only | rebuilt notebook: 0 CRLF |
| LOW | reviewer prompt still described the old early stop | text | updated to the per-query completion rule | `PROMPT_FOR_REVIEWER.md` |

Quota planning: reserve ~4 wall-hours (~8 quota-hours) for this run as suggested; expected 1.3–2.5 h.

Harness on the fixed notebook (six runs, `harness_logs/h4_*.txt`):

| run | verdict | bundle |
|---|---|---|
| normal (all scripted scenarios) | PROMISING (by construction) | yes |
| deadline | INCOMPLETE / NO DECISION (stop before round 0) | yes |
| verifier infrastructure injection | INVALID / VERIFIER INFRASTRUCTURE FAILURE | yes |
| engine failure phase 1 | INVALID / ENGINE FAILURE | yes |
| engine failure phase 2 | INVALID / ENGINE FAILURE | yes |
| short output list | INVALID / ENGINE FAILURE | yes |
