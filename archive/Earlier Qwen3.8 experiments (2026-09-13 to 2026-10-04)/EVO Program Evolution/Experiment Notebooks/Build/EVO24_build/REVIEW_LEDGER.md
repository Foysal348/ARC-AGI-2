# Review ledger — EVO24 review rounds 1–3: what was right, what was wrong, on both sides

Purpose: before you review round 4, see exactly how each of your earlier claims was checked, which were correct, which
were partly correct or had low effect, which errors were mine, and what neither of us caught until later. Every claim
was verified against the code, the data or the logs **before** anything was changed. Labels:

* **CONFIRMED** — claim true as stated; fixed.
* **CONFIRMED, LOW EFFECT** — claim true, but the proposed remedy changes little; kept or adjusted.
* **INCORRECT** — claim false (none so far; see section 4).

## 1. Reviewer claims, round by round

### Round 1 (verdict NO-GO, notebook `23100b40…`)

| claim | verdict | how it was checked |
|---|---|---|
| HIGH-1: first train-exact candidate stops the task → pass@2 is effectively pass@1 | **CONFIRMED** | `st["solved"]` stop at cell_evo.py:238/275; harness log showed 271d71e2, b5ca7ac4, eee78d87 with an exact candidate, strict False, 5 calls — exactly the lines cited |
| HIGH-2: parent ranking ignores the identity baseline (the S12 lesson) | **CONFIRMED** as a reporting/selection gap; **LOW EFFECT** as a ranking change — see 3.1 | rank key was `_v3_train_quality`; S12 report table (97d7923e: candidate 4.5 %, identity 4.0 %) matches |
| HIGH-3: deadline-stopped run reported WEAK | **CONFIRMED** | old deadline log: STOP after round 0, VERDICT WEAK |
| HIGH-4: verifier infrastructure failure absorbed as model failure | **CONFIRMED** | `_evo_verify` ignored `infrastructure_failure`; base verifier sets it |
| MED-1: 32,768 cap not hard (two calls planned against the same used total) | **CONFIRMED** | code path; arithmetic in the claim correct |
| MED-2: `finish_reason == "stop"` treated as `</think>`; EOS inside thinking mis-closed | **CONFIRMED** | code; vLLM `stop_reason` semantics as cited |
| MED-3: tasks with no code in round 0 abandoned | **CONFIRMED** | `_evo_select_parents` returns [] → no calls |
| concern: PROMISING allowed 3 solves with 0 NVARC-miss recoveries | **CONFIRMED** | rule text; tightened before any run |
| minor: NVARC partial task not categorised | **CONFIRMED** | 4a21e3da NVARC per-query `[False, True]` |
| LOW: absolute paths in builder | **CONFIRMED** | builder source |
| runtime: 1.2–1.7 h is not a safe upper bound; baseline 65 tok/s exploration, 39 tok/s finalizer; 192 × 5,120 ≈ 983k | **CONFIRMED** | canonical baseline doc: 65.3 / 38.8 tok/s; with the then-soft cap 24 × 8 × 5,120 = 983,040 |
| suggested fail-closed by `raise RuntimeError` | **adjusted** | a raise would also lose the report/bundle; used the reviewer's alternative (record + INVALID verdict) instead |

### Round 2 (verdict NO-GO, notebook `d2c7650c…`)

| claim | verdict | how it was checked |
|---|---|---|
| HIGH: pass@2 took two whole programs, not two outputs per query; canonical definition is per query | **CONFIRMED** | `_evo_attempts` + scoring code; canonical doc: "every test query … correct with attempt_1 or attempt_2"; 10 of the 24 frozen tasks have 2 queries |
| same root: partial `[grid, None]` outputs discarded | **CONFIRMED** | `solved_train` requires all test outputs; verifier keeps partial outputs (self-test `partial_multiquery_test_outputs_preserved`) |
| same root: search stopped once two global programs differed | **CONFIRMED** | `n_att >= 2` stop |
| MEDIUM: no deadline check before round 0 | **CONFIRMED** | `run_evo` |
| MEDIUM: notebook header still showed the old PROMISING rule | **CONFIRMED** | builder header text |
| concern: within a task, `gain_vs_identity` ranks the same as `-wrong_frac`; identity-level candidates still get repair slots | **CONFIRMED** (mathematically: identity is constant per task, so the two orders are identical) | see 3.1; metric `repair_parents_nonpositive_gain` added |
| runtime: 716,922 tokens at 65.3 tok/s ≈ 3.05 h generation in the baseline | **CONFIRMED** | 716,922 / 65.3 / 3600 = 3.05 |

### Round 3 (verdict NO-GO, notebook `30fb30db…`)

| claim | verdict | how it was checked |
|---|---|---|
| HIGH: `arc_llm.generate()` exception or short output list crashes cell 8, report and bundle lost | **CONFIRMED** | no try/except anywhere in cell 8; older PIV notebooks have `_terminalize_engine_exception` |
| MEDIUM: a train-exact program that fails on a test input is never repaired directly | **CONFIRMED** | `exclude = {all exact}` in planning |
| MEDIUM: 4+2+2 calls not guaranteed under the cap (8 × 5,120 = 40,960 > 32,768; two gen-2 slots need ≤ 22,528 used) | **CONFIRMED** | arithmetic |
| LOW: notebook SHA depends on line endings | **CONFIRMED** | 4,304 CRLF in the file |
| LOW: reviewer prompt described the old early stop | **CONFIRMED** | prompt text |
| runtime: 786,432-token ceiling ≈ 3.35 h at 65.3 tok/s; plan ~4 wall-h / ~8 quota-h | **CONFIRMED** | 786,432 / 65.3 / 3600 = 3.35 |

## 2. My errors (author side)

| # | error | caught by | status |
|---|---|---|---|
| A1 | compared the perf_counter `deadline` with `time.time()` — every real run would have stopped after round 0 | my harness, before round 1 | fixed |
| A2 | pass@1 stop, deadline → WEAK, infra not handled, soft cap, `finish_reason` close, no-code abandonment | reviewer R1 | fixed |
| A3 | wrote in my own S12 report that near-miss metrics must be relative to identity, then did not apply it in EVO | reviewer R1 | reported per candidate; see 3.1 |
| A4 | in the R1 response I called pass@2 "fixed" while it was still global-program pass@2, not per query | reviewer R2 | fixed in R2 |
| A5 | dropped the engine-exception handling that the older PIV notebooks had | reviewer R3 | fixed in R3 |
| A6 | runtime estimate 1.2–1.7 h presented as if it were an upper bound | reviewers R1–R3 | planning now ~1.3–2.5 h, buffer 4 wall-h |
| A7 | (found by me while verifying R3) an exception inside `_evo_verify`, any other exception in `run_evo`, or a failure of the scoring cell would also have lost the evidence | own check | fixed in R3 (catch-all, verify wrapper, protected bundle write) |

Earlier, outside EVO but relevant to how far to trust my numbers: the "36 near misses, 25 on NVARC misses" in the
canonical baseline used wrong-cell fractions without an identity reference; three reviewers accepted it and the S12 GPU
run showed those near misses were at identity level. Both sides missed it until real data arrived.

## 3. Reviewer suggestions that were weaker than stated

### 3.1 R1 HIGH-2 — identity-relative ranking

The diagnosis (S12 lesson not applied) was right. The proposed remedy — put `gain_vs_identity` into the parent rank key
— adds almost nothing: inside one task the identity wrong fraction is a constant, so ordering by
`identity − wrong` is the same as ordering by `−wrong` (checked numerically). The only effect is that the term now sits
before "test runs" and "pairs without exception" in the tuple. R2 pointed this out itself. What does carry information:
reporting `identity_wrong_frac`, `best_wrong_frac`, `gain_vs_identity` per task, and the new count of repair parents with
zero exact pairs and gain ≤ 0. Filtering such parents out (giving their slots to fresh attempts) is the substantive fix;
it is deferred to the next revision so the first run measures the architecture as designed.

### 3.2 R1 HIGH-4 — `raise RuntimeError`

Raising would have reproduced the R3 problem (no report, no bundle). The non-raising variant the reviewer also offered
was used.

## 4. Reviewer claims found incorrect

None so far. Every factual claim (line numbers, task IDs, counts, arithmetic, baseline throughput, vLLM `stop_reason`
behaviour) checked out. The disagreements are limited to the effect size of one remedy (3.1) and the choice between two
offered fail-closed variants (3.2).

## 5. Not caught in any review round (found by the harness or by me)

* A1 (deadline clock) — harness.
* A7 (verify / run_evo / scoring exceptions) — own check during R3 verification.
* Scripted lookup answers exceeding the 2,048-token answer cap made the first harness under-test the solved path —
  a harness defect, fixed by compressing the fake answers.
