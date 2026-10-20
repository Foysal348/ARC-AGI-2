# EVO24 — response to review round 1 (NO-GO, 4 HIGH + 3 MEDIUM)

Every finding was first checked against the code and the harness logs; all were confirmed. Notebook after the fixes:
`PIV-Q27-EVO24-EVOLUTION.ipynb` sha256 `d2c7650c134e7dc79286284139bb8d4dbf974d0ff1faca0a5e4f39c7e6351bbb`
(previous `23100b40…`). Model, tasks, verifier, helper library and the evolution architecture are unchanged.

| # | finding | confirmed by | fix | evidence (harness) |
|---|---|---|---|---|
| HIGH-1 | first train-exact candidate stops the task → pass@2 is effectively pass@1 | `st["solved"]` stop; old log: 271d71e2 / b5ca7ac4 / eee78d87 exact but not strict, calls=5 | a task is finished only with **two distinct test outputs**; with one, slot 0 is an **alternative-hypothesis** prompt ("find a genuinely different transformation that also fits every training example"), slot 1 repairs the best other candidate | new scenario: train-exact-but-wrong first, correct alternative later → 271d71e2, 4c3d4a41, b5ca7ac4: attempts=2, strict=True (`h2_normal.txt`) |
| HIGH-2 | parent ranking uses raw wrong-cell fraction, not relative to the identity program (the S12 lesson) | `_evo_select_parents` sorted by `_v3_train_quality` | `evo_identity_wrong_frac(task)`; per candidate `wrong_frac`, `identity_wrong_frac`, `gain_vs_identity`; rank key = (exact&test, exact, exact pairs, right-shaped pairs, **gain_vs_identity**, test runs, no-exception pairs, −wrong, shorter code); used for parents and pass@2; report shows all three columns | self-test: candidate worse than identity has negative gain and ranks below an identity-level one; report rows show identity / wrong / gain |
| HIGH-3 | deadline stop reported as WEAK | old `h_deadline.txt`: STOP after round 0, VERDICT WEAK | `evo_verdict()`: deadline → **INCOMPLETE / NO DECISION**; no model → INVALID; interface failure → WEAK (interface) | `h2_deadline.txt`: VERDICT INCOMPLETE / NO DECISION |
| HIGH-4 | verifier infrastructure failures absorbed as model failures | `_evo_verify` ignored `infrastructure_failure` | recorded per candidate; the run **stops after the round** (fail-closed) and the verdict is **INVALID / VERIFIER INFRASTRUCTURE FAILURE** | `HARNESS_EVO_INFRA=1` injection: run stops after round 0, verdict INVALID (`h2_infra.txt`) |
| MED-1 | 32k per-task cap not hard (two calls planned against the same used total) | `evo_run_round` | per-round **reservation**: each admitted call reserves its maximum (5,120) | code (`reserved[tid]`) |
| MED-2 | `finish_reason=="stop"` treated as a natural `</think>` close; EOS inside thinking would be mis-closed | `evo_generate` | `_evo_close_phase1(ids, text, stop_reason)`: natural close only if the `</think>` id is in the ids, is the `stop_reason`, or is in the text; EOS/length → trailing EOS stripped + forced close | self-tests for all four cases; harness scenarios: `stop_reason=</think>` without echo (pos 5) counted closed, EOS in thinking (pos 7) forced (round-0 think_closed 72/96 = 24 forced as scripted) |
| MED-3 | a task whose 4 initial answers had no code is abandoned | `_evo_select_parents` returns [] → no calls | slots without a usable parent are **backfilled with fresh attempts** (rotated lens + "try a substantially different hypothesis") | round kinds: R1 `fresh: 30`, R2 `fresh: 21`; no-code tasks now get all 8 calls |
| concern | PROMISING allowed 3 solves with 0 NVARC-miss recoveries | — | tightened (pre-registered, before any run): PROMISING = evo_only ≥ 2 **or** (solved ≥ 3 **and** evo_only ≥ 1) | self-test of `evo_verdict` |
| minor | NVARC partial task not a separate category | — | `NVARC_PARTIAL_tasks`, `EVO_SOLVES_NVARC_PARTIAL` | report |
| LOW | builder used absolute Windows paths | — | bundle-relative paths (`notebooks/`, `posthoc/`) with project-layout fallback | `build/make_evo24.py` |

Trade-off to note: HIGH-4 is fail-closed, so a single transient verifier failure ends the run after that round
(the bundle and report are still written). I kept it, as requested, because a silent infrastructure failure would
produce a false WEAK verdict.

Runtime: I accept the reviewer's planning range of ~1.3–2.5 h wall. Backfill and alternative calls keep round 1 and 2 at
48 calls each instead of ≤48 (they no longer shrink when tasks have no code), so the upper end is more likely than before.
The per-task cap (32,768, now hard) bounds the worst case at 24 × 32,768 ≈ 786k generated tokens.

Harness runs on the fixed notebook (fake vLLM, real tokenizer/template/data/verifier): `h2_normal.txt` (all scenarios,
verdict PROMISING by construction of the scripted answers), `h2_deadline.txt` (INCOMPLETE), `h2_infra.txt` (INVALID).
