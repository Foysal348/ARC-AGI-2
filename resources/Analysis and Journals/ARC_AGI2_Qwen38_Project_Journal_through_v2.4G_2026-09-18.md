# ARC-AGI-2 / Qwen3.8-27B — Canonical Project Journal
## Through the real v2.4G C1 Adaptive SMOKE run — 2026-09-18

> **Purpose.** This is the persistent experiment/decision log for the project. It records the important findings, failures, corrections, lessons, discussion-derived insights, decisions that proved useful, decisions later revised, current evidence state, and the exact next experiment.  
> **Evidence labels:** **VERIFIED** = directly supported by executed artifacts/bundles; **INFERENCE** = synthesis from multiple verified facts; **RETIRED** = invalidated/obsolete assumption; **UNKNOWN** = requires a new experiment.  
> **Do not use this journal as a substitute for raw evidence.** Raw bundles, traces, verifier outputs and hashes remain the authority.

---

## 0. Current executive snapshot

**Current state: `mechanism works; semantic program quality is now the primary blocker`.**

The v2.4G real 4×L4 run is the first run in this lineage where the complete agent pipeline reached a standalone program on **every real trajectory**:

- **16/16 final programs produced**
- **16/16 syntax-valid**
- **16/16 fresh verifier executions**
- **16/16 terminal JSON artifacts parsed**
- **0 infrastructure failures**
- **0 budget deaths**
- **0/16 all-train-exact**

This moves the bottleneck from **artifact production/finalization** to **program correctness**.

### Canonical v2.4G real-run identity

| Item | Value |
|---|---|
| Run ID | `20260918T104631Z` |
| Profile | `ADAPTIVE_SMOKE` |
| GPU | 4× NVIDIA L4 |
| Canonical source SHA256 | `50b9acb8b62b3c1860914f562c5154dd58e8906b3342cd8ad6d8875fdc882769` |
| Executed notebook SHA256 | `44934b207807f2b4a870304205b579f78daea3f23b64720aa6fec73455ffb66f` |
| Evidence bundle SHA256 | `387d4bb7383c9e7917c6deff5d2999e0ab83ff9b105d1487aa7dcd7a8cb56837` |
| Bundle manifest | 81/81 entries verified; 0 SHA/size mismatches |
| Code source match | Executed code cells are source-exact to canonical v2.4G |
| Total elapsed | 4691.31 sec ≈ 78.19 min |
| Model load | 8.25 min |
| Warmup | 0.17 min |

### Real arm results

| Metric | ACT-LOW | ACT-BUDGET3072 |
|---|---:|---:|
| Tool engagement | 75.0% | 100.0% |
| Final program | 100.0% | 100.0% |
| Syntax valid | 100.0% | 100.0% |
| Verifier executed | 100.0% | 100.0% |
| All-train-exact | 0.0% | 0.0% |
| Budget death | 0.0% | 0.0% |
| Infrastructure failure | 0.0% | 0.0% |
| Mean generated tokens/task | 9180 | 9371 |
| sec/output-token | 0.0464 | 0.0513 |
| Proxy48 projected +20% | 4.75 h | 3.64 h |

**Most important near-quality signal:** `ACT-LOW / 291dc1e1` matched **2/4 training examples exactly**. This is the strongest current seed for verifier-feedback repair. BUDGET3072 had zero exact train pairs across all eight tasks; `136b0064` timed out in the train verifier.

---

# 1. Experiment chronology — finding → failure → fix → lasting lesson

## A. Direct/Q1 baseline

### Direct OFF vs XHIGH
- **VERIFIED:** direct generation did not solve tasks; OFF produced some valid grids but no exact task; XHIGH frequently spent the whole generation budget before useful action.
- **Lesson:** raw model generation is insufficient; code/tool interaction is needed.
- **Persistent rule:** use fresh program verification, not prose/grid plausibility, as correctness authority.

---

## B. PIV harness lineage v1.0 → v1.7

### v1.0 — first stateful REPL + fresh verifier
- Introduced the right core direction: persistent Python, puzzle preloaded/protected, final `solution(grid)`, fresh verification.
- Early implementation still had source/runtime mismatches.

### v1.2 — separate train-perfect from later test execution
- **Bug fixed:** a test-stage failure could erase valid train-perfect evidence.
- **Persistent rule:** train correctness and test execution are separate states.

### v1.3 — parser/retry/grid/worker/sandbox hardening
- Added bounded retry, think-aware parsing, worker restart, tolerant grid normalization, partial-output preservation.
- **Lesson:** compile-clean code does not imply runtime-safe agent behavior.

### v1.4 — same-history retry vs outer restart
- **Bug fixed:** local retry and outer source-style restart had been conflated.
- **Persistent rule:** preserve REPL/history on step retry; use outer restart only after bounded retry exhaustion.

### v1.5 — viability gate correction
- Replaced weak first-turn-length heuristic with terminal budget-death + empirical wall-clock feasibility.
- **Lesson:** measure the terminal failure mode, not a proxy that can be misleading.

### v1.6 — deadline/SMOKE8/accounting hardening
- Added projected-deadline protection, representative stratified SMOKE8, dynamic guard accounting, sanitized tool feedback, terminal coverage and real-scheduler PREFLIGHT.
- **Lesson:** harness invariants must be actively exercised, not merely present.

### v1.7 — dynamic multi-call budget + canonical XHIGH failure
- **VERIFIED real GPU:** 8/8 tasks consumed 12,288 tokens in first-turn thinking; tool engagement 0/8; final programs 0/8; all-train-exact 0/8; infrastructure clean.
- **Root cause:** template/reasoning-budget interaction, not REPL/parser/verifier failure.
- **Persistent rule:** XHIGH natural reasoning is a retired path under this budget.

---

## C. Reasoning activation v1.8 → v1.9d

### LOW / MEDIUM / BUDGET3072 probe
- **VERIFIED real GPU:** LOW 7/8 tool activation; BUDGET3072 7/8; MEDIUM 1/8; XHIGH frozen reference 0/8.
- LOW tool execution was materially cleaner and cheaper than BUDGET3072 in the cheap canary.
- **Decision:** retain LOW and BUDGET3072 independently; retire MEDIUM from near-term solver runs.
- **Correction:** `reasoning_effort="high"` is invalid for this model family; only `xhigh/medium/low`.
- **Persistent rule:** activation is not solving. Probe zeros do not establish solver-quality failure.

### Forced-close lessons
- Forced close only exits thinking; it does **not** guarantee a tool call or final code.
- Token-prefix continuation must remain exact; no silent text re-render.
- Guard rejection must not consume executed tool budget.
- Phase-2 seed remains separate and deterministic.

---

## D. Throughput v1.2.4

### C1 graph/noAR
- **VERIFIED real 4×L4:** C0 long@8 ≈ 48.30 tok/s; C1 graph/noAR ≈ 129.17 tok/s; ≈2.67× speedup.
- Custom all-reduce added no meaningful gain.
- C1 became the prime serving candidate.

### Retired numerics gate
- The old `greedy-prefix agreement >= 0.50` safety gate was invalidated because the unchanged C0 repeat itself scored ≈0.1875.
- **RETIRED:** token fingerprint/prefix agreement as solver-correctness authority.
- **Persistent rule:** serving safety must be judged by actual solver behavior: final program, fresh all-train-exact, test execution, official exact.

### Prefix64 / TP2 OOM
- **Lesson:** stressed OOM rows are capacity failures, not proof that a feature is intrinsically unsupported.
- Parent-side fail-fast was insufficient and wasted large wall time.
- **Persistent rule:** terminal worker failure should be reclaimed quickly; do not wait full timeout if failure is already known.

---

## E. P1-SPARSE side branch
- Aggregate outcomes were interesting but game-level variance and teardown/runtime-health issues remained.
- **Decision:** `PROMISING/HOLD`; keep its metric logic separate from the Qwen ARC-AGI-2 main solver.
- **Persistent rule:** side-branch scores cannot drive main-line promotion.

---

# 2. C1 solver-quality lineage v2.0 → v2.3

### v2.0 → v2.2
Correctness/failure-path audits found and fixed:
- train-exact promotion without usable test output;
- strict task metric mislabeling;
- verifier state overstatement;
- test timeout erasing train evidence;
- generate exception handling;
- unsafe phase-2 draft salvage;
- infrastructure failure being confused with model-quality failure.

### v2.3 PREFLIGHT
- Harness cleared the major P0/P1 audit issues.
- Important rule locked: **PREFLIGHT is harness evidence, not model success.**

### v2.3 real C1 quality smoke
**VERIFIED real GPU:**
- C1 serving worked.
- LOW tool engagement 7/8; BUDGET3072 8/8.
- Persistent REPL was genuinely used.
- But **0 final programs / 16 trajectories**, therefore verifier was never reached.
- LOW spent ~92% of generation in thinking; BUDGET forced close frequently led to another tool call/malformed action rather than a final program.
- **Finding:** bottleneck moved from “can the model act?” to “can it commit a standalone program?”
- **Decision:** do not run Proxy48/Public. Build a protected terminal commit mechanism.

---

# 3. Terminal-commit lineage v2.4 → v2.4G

## v2.4 — protected terminal budget
Design:
- total 12,288 tokens/task unchanged;
- exploration ≤8,192;
- protected final reserve 4,096;
- non-thinking, no-tools structured finalizer;
- fresh verifier remains authority.

### Strategic correction
The first roadmap was too serial: `PREFLIGHT → upload/review → 2-task pilot → upload/review → SMOKE8`.
We changed to **result-driven one-session execution**: embedded safety checks → positive control → 2-task interface diagnostic → automatic SMOKE8 if structurally clean.

**Lesson:** after basic infrastructure is stable, endless review loops can become the bottleneck themselves.

## v2.4A
Found deterministic blockers:
- terminal wave `"phase":"terminal_finalizer"` crashed downstream `int(phase)`;
- adaptive gate could false-pass a broken finalizer and false-fail a natural early solution;
- runtime accumulation was overwritten;
- context-reserve guarantee was incomplete.

## v2.4B
Major fixes:
- hardened structured JSON parsing and special-token handling;
- safe phase coercion;
- artifact-based adaptive interface gate;
- cumulative stage runtime;
- dedented tool payloads;
- verifier return contract consistency;
- engine output-count guards.

## v2.4C
Fixed:
- verifier infrastructure propagation;
- terminal context reserve against tool-feedback growth;
- full-SMOKE-only recommendation scope.

## v2.4D
Fixed:
- bounded reasoning handoff to finalizer;
- retry exhaustion now uses protected terminal slot;
- REPL startup/runtime infrastructure propagation;
- unclosed-thinking draft not treated as committed final;
- terminal JSON parser no longer mutates control-token-looking literals inside program strings;
- positive control includes fresh verifier;
- Proxy readiness respects compute/promotion gate.

## v2.4E
Fixed two runtime blockers:
- parser self-test had stale expectations around unclosed thinking;
- PREFLIGHT recommendation variable initialization.

## v2.4F
Fixed:
- PREFLIGHT terminal-policy assertion;
- shared terminal-trigger matrix;
- engine output cap guard;
- retry-accounting noise at terminal boundary;
- TERMINAL_PILOT report labels.

## v2.4G
Fixed defensive cap-guard blast radius:
- a single per-request `max_tokens` overrun no longer kills the whole wave/arm;
- output-count mismatch remains wave-fatal because mapping becomes untrustworthy.

**Hard-stop decision before run:** no more pre-run micro-patching unless a deterministic default-path blocker is reproduced. The next information had to come from real GPU execution.

---

# 4. v2.4G real GPU run — what it actually proved

## 4.1 Integrity
- **VERIFIED:** executed code cells are source-exact to canonical v2.4G.
- **VERIFIED:** evidence bundle manifest 81/81 entries passed SHA/size verification.
- **VERIFIED:** engine bootstrap complete on 4×L4; C1 engine fingerprint unchanged.
- **VERIFIED:** real-Qwen terminal positive control passed structured output → parser → fresh verifier.
- **VERIFIED:** adaptive 2-task gate passed; remaining SMOKE6 ran automatically.

## 4.2 The central progress result

### Before v2.4G (v2.3)
`reasoning → tools → no final artifact`

### Now
`reasoning → tools → terminal commit → standalone code → fresh verifier → wrong program`

This is **substantial progress**, because harness/interface failure is no longer the dominant blocker.

## 4.3 ACT-LOW
- Tool engagement: 75%.
- 17/17 executed tool calls succeeded; no exceptions/timeouts/guard rejects.
- 8/8 final programs; 8/8 syntax valid; 8/8 verifier executed.
- 0/8 all-train-exact.
- Train pair evidence: `291dc1e1` achieved 2/4 exact training pairs; all other LOW tasks 0 exact pairs.
- Finalizer artifact parse failures: 0.
- Mean generated tokens/task: ~9,180.
- Median unused token budget after candidate #1: ~2646; worst-case unused budget across LOW tasks: ~1229.
- Runtime: pilot 2 ≈ 9.58 min; remaining 6 ≈ 28.46 min; total LOW loop ≈ 38.04 min.

## 4.4 ACT-BUDGET3072
- Tool engagement: 100%.
- 22 executed calls: 19 OK, 3 Python exceptions; no tool timeout/guard rejection.
- 8/8 final programs; 8/8 syntax valid; 8/8 verifier executed.
- 0/8 all-train-exact.
- 7 `train_rejected`; `136b0064` train verifier timeout.
- 17 forced-close turns; 5 action-cap hits; BUDGET still operates in roughly two full forced-close turns + partial third under 8,192 exploration.
- Median unused token budget after candidate #1: ~2758; worst-case unused budget across BUDGET tasks: ~2175.
- Runtime: pilot 2 ≈ 11.13 min; remaining 6 ≈ 17.67 min; total BUDGET loop ≈ 28.80 min.

## 4.5 What we must NOT conclude
- Do **not** declare LOW or BUDGET a quality winner: correctness is 0/8 for both.
- Do **not** run Proxy48 merely because compute gate passed.
- Do **not** return to terminal-reserve tuning: terminal artifact production succeeded 16/16.
- Do **not** infer that C1 hurts quality from this run; there is no matched C0-quality comparator here and the current blocker is semantic synthesis.
- Do **not** call BUDGET “bad” solely because it has fewer effective exploration turns; that is a known property of the chosen terminal-policy package.

---

# 5. Decisions that proved useful

1. **Keeping LOW and BUDGET3072 independent** after activation. LOW failing would not have been a valid reason to suppress BUDGET, and vice versa.
2. **Retiring MEDIUM** from near-term quality experiments after weak activation.
3. **Using only one serving intervention:** `enforce_eager=True → False`; custom AR and prefix cache stayed off.
4. **Rejecting the old greedy-prefix numerics gate.**
5. **Blocking Proxy48/Public until real verified single-lineage quality exists.**
6. **Preserving fresh all-train-exact as the correctness authority.**
7. **Separating infrastructure validity from model-quality failure.**
8. **Insisting on source hashes, executed notebook, bundle manifest and trace evidence.**
9. **Switching from review-driven to result-driven execution** once the harness became stable.
10. **Freezing v2.4G and actually running it** instead of continuing pre-run patch loops.

---

# 6. Decisions/assumptions later corrected

1. **“Compile/nbformat pass means runtime-safe.” — RETIRED.** Sequential execution and targeted counterexamples are necessary.
2. **“PREFLIGHT pass means model will activate/solve.” — RETIRED.** Fake-engine success proves harness behavior only.
3. **“Forced close guarantees a tool/final answer.” — RETIRED.**
4. **“Tool engagement implies solving progress.” — RETIRED as a promotion rule.** It is diagnostic only.
5. **“Greedy-prefix ≥0.50 proves serving safety.” — RETIRED.**
6. **“OOM means prefix/TP feature unsupported.” — RETIRED.** It may be capacity/concurrency stress.
7. **“Every notebook revision deserves a separate upload/review milestone.” — REVISED.** This created an avoidable serial loop.
8. **Unsupported attribution that Tufa Labs used Qwen3.8-27B on ARC-AGI-2 — CORRECTED.** Public evidence supported a Qwen-27B-class ARC-AGI-3 harness, not that specific ARC-AGI-2 model claim.
9. **Early tendency to keep polishing the harness after v2.3 — REVISED.** The project imposed a hard stop and required a real GPU result.
10. **Runtime estimate 2.5–3.5 h for v2.4G — OVERESTIMATED.** Actual full run was ~78.2 min.

---

# 7. Regression firewall — rules that must persist

- No `reasoning_effort="high"`.
- Preserve exact forced-close/token-prefix continuation semantics; never silently text-rerender phase 2.
- Guard rejection does not consume executed tool budget.
- Same-history retry and outer restart remain distinct.
- Train-perfect evidence cannot be erased by later test failure.
- Missing/duplicate terminal task IDs fail coverage.
- Infrastructure failure is separate from candidate correctness.
- No old greedy-prefix/raw-cell similarity quality gate.
- TP2/prefix OOM is not feature-infeasibility proof.
- No prompt search/PTQ/NVARC/reviewer/evolution/seed diversification mixed into a single-mechanism experiment.
- No Proxy48/Public until at least one reproducible fresh-verifier all-train-exact program with usable test execution.
- One major mechanism per experiment.
- New P2/P3 issues are logged/deferred; only reproduced P0/P1 blockers should interrupt the run cadence.

---

# 8. Current progress state

## VERIFIED
- Stateful REPL/tool transport works.
- LOW and BUDGET3072 activate tools on real Qwen.
- C1 graph/noAR serving is fast enough for SMOKE and projected Proxy48 compute.
- Terminal reserve/finalizer solves the previous artifact-production bottleneck.
- Structured terminal output, parser, context reserve, fresh verifier and adaptive gate all worked in the real run.
- Program syntax/execution validity is no longer the immediate blocker.

## UNKNOWN / unresolved
- Can verifier feedback repair a partially correct candidate into all-train-exact?
- Which arm is better after repair, not merely before repair?
- Can single-lineage quality become reproducibly non-zero?
- Once non-zero quality exists, how much does multi-sample/pass@K help?
- Does a matched C0-LOW quality control materially differ from C1? Only needed if serving behavior becomes suspect.
- Prefix caching/TP2 remain later throughput questions, not current solver questions.

## RETIRED / do not reopen now
- XHIGH natural full-budget path.
- MEDIUM near-term solver arm.
- Old numerics/prefix correctness gate.
- Terminal reserve tuning as the primary next step.
- Broad pre-run notebook review loops without an observed blocker.

---

# 9. Exact next experiment — v2.5 Candidate-First Verifier-Feedback Repair

**Primary question:** after candidate #1 fails fresh train verification, can one compact verifier-feedback repair turn produce an all-train-exact program?

### Frozen from v2.4G
- same model/runtime/C1 engine;
- same frozen SMOKE8 tasks/order;
- same two arms and seed policy;
- same REPL, terminal commit, JSON parser and fresh verifier;
- same 12,288 total token ceiling;
- same test embargo: no test execution until all training pairs are exact.

### Single new mechanism
For a failed candidate:
1. run fresh train verifier;
2. construct compact feedback containing only training-side evidence: failed training indices, execution error if any, and bounded candidate-prediction vs expected-grid diagnostics;
3. verify that the feedback is actually present in the repair prompt using hash + exact-presence instrumentation;
4. make **one** non-thinking/no-tool structured repair call;
5. fresh-verify candidate #2;
6. only if all-train-exact, execute tests.

### Why one repair round first
- v2.4G leaves median unused budget of about **2646 LOW tokens** and **2758 BUDGET tokens** after candidate #1.
- Every real task retained at least roughly **1229 tokens** of unused total budget.
- Therefore repair can be tested inside the same 12,288-token cap without introducing a larger-budget confound.

### Diagnostic anchor
`291dc1e1 / ACT-LOW` is the strongest current seed because candidate #1 was already **2/4 train-exact**.

### Run discipline
Do **not** return to a 2-task → upload → review loop. Use one notebook, full frozen SMOKE8, same two arms, one repair round, then inspect real quality evidence.

### Promotion gate after v2.5
- `all_train_exact > 0` + usable test execution → reproduce once, then Proxy48.
- final programs remain 100% but repaired train-exact remains 0 → inspect semantic failure taxonomy / candidate-search strategy next.
- infrastructure invalid → fix only the observed blocker.
- do not add multi-sample, reviewer, DSL or evolution until single-lineage repair quality is non-zero.

---

# 10. Canonical artifact map

| Artifact | Role |
|---|---|
| `ARC_AGI2_Qwen38_Deep_Audit_Reference_2026-09-17.pdf` | pre-v2.4 canonical historical audit |
| `PIV-Q27-C1-v2_4G-ONE-SESSION-ADAPTIVE-SMOKE-FINAL.ipynb` | canonical source for decisive terminal-commit run |
| `experiment-1-arc-agi-2-with-qwen-3-8 (12)(5).ipynb` | executed v2.4G notebook |
| `PIV_Q27_C1_V24G_ADAPTIVE_SMOKE_BUNDLE.zip` | canonical v2.4G evidence |
| This annotated notebook | living project journal + executed evidence notebook; **markdown-only annotation, solver code unchanged** |

---

## Final status line

**Progress is real:** the project has moved from **no action → tools → no final artifact → 100% executable final artifacts**.  
**The next bottleneck is now narrowly defined:** **semantic program repair from fresh verifier feedback**.  
The next experiment should produce new solver-quality evidence, not another harness-hardening loop.
