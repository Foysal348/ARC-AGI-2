# 01 — Prioritized fix list for config B (2 × 35k independent attempts)

**Date:** 2026-10-24.

**The design was fixed by the user:**
- 2 × 35k independent attempts per task;
- attempt 1 runs on every task first, then attempt 2.

**Sources** (all re-read in full on 2026-10-24):
- Exp 14 `Notes/EVAL120 analysis 2026-10-24/`:
  - `A_score_coverage.md` (score, coverage, Poetiq complementarity, projections);
  - `B_behaviour.md` (agent behaviour);
  - `C_engine.md` (engine).
- Exp 14 Notes/44, 45, 46: three GPT deep-research reports, verbatim.
- Exp 14 Notes/43: merged list (ours + GPT), with my verifications (sections 0, 7, 8).
- Exp 14 Notes/42: the first fix list from the EVAL120 runs.

**Base:**
- notebook E14S (`build_e14.py`, variant E14S): XH 41/172 = 25.83 %, LB 22.78;
- Duck adapter rev 6g (`Experiment 12 .../duck_arc2/arc2_duck.py`).

**Engine:** unchanged.
- SGLang 0.5.21 TP4, RedHat INT4 W4A16, DFlash2 width 6;
- graphs bs 1..13, mrr 13, mamba cache 56;
- FP8 KV, fp16 SSM, context 155,648.

Legend: **[A]** / **[B]** / **[C]** = agent reports; **[G1]** / **[G2]** / **[G3]** = GPT Notes/44 / 45 / 46; **[V]** = my verification in Notes/43; **N43 #n** = item number in Notes/43.

---

## P0-A — core of config B

| # | Change | Where / how | Evidence |
|---|---|---|---|
| 1 | **Two independent attempts per task.** Attempt 2 is a fresh Duck run of the same task (the facts view, as branch 0, with a different sampling seed). It is **not** a resume of attempt 1 and **not** the old indexed-view branch 1. | `solve_all_r6(branches=2, b1_view="facts")` and a per-attempt seed. **Check** that a b1 started with the facts view gets its own agent and history. | [A] §2: b1 (indexed) gave 0 correct in 244 runs. [B] §1.6: two xhigh runs overlap on 26 of 61 queries. [G1-G3] |
| 2 | **Attempt 1 on every task before any attempt 2.** Order: ORDER2 + cells, as today. | Stage 1 = attempt 1 on all tasks; stage 2 = attempt 2 only. Keep "stage 1 before any stage-2 work". | [A] §4: coverage of the first look already 120/120 |
| 3 | **35,000 generated tokens per attempt, then the attempt ends.** No 28k pause, no stage-2 chunks, no resumes. | `max_generated_tokens=35000`, `first_exposure_tokens=None`; remove the stage-2 resume classes for B. | [C] §3.4: 58-66 % of prefill was re-prefill of resumed branches; throughput −18-30 % after hour 2. [A] §4: 156-170 branches paused and never resumed |
| 4 | **No 12,288 / 24,576 / 32,768 request caps.** Each request may use the attempt's remaining budget. | `_arc2_apply_cap`: first = progress = rescue = remaining attempt budget. | [B] §3.1, [C] §6: 84-98 length cuts with no tool call. The caps came from ARC-AGI-3 (memory: ARC-AGI-2 needs long thinking) |
| 5 | **No context trim inside an attempt.** Today the trim budget = window 76,484 − reply reserve − 512. With a 35k reserve the trim would start at ~41k, *earlier* than today's 51,396.<br>Fix:<br>• reply reserve = the attempt's remaining budget;<br>• raise the Duck window to ~100k (the server context is 155,648).<br>The trimmer stays only as a logged last resort. | `_arc2_window`, `_reply_reserve_tokens`, `_context_budget_tokens` (`arc2_duck.py:1530-1542`) | [B] §5.1b: trimmed branches correct 16 % vs 34 % (XH+E16), 5 % vs 30 % (LOW). [G1]: 37 of the 55 LOW no-progress tasks were trimmed. **[V]** branches with a trimmed request: 34 / 41 / 64 |
| 6 | **Admission guard per attempt:** start an attempt only if the remaining time ≥ the expected time for 35k (≈ 35,000 / 30 tok/s per stream ≈ 1,170 s, plus a margin). Otherwise leave the worker idle or give the time to a running attempt's harvest. | `min_admit_sec` ≈ 1,300 s for B (today 240 s) | [A] §4: per-stream rate p50 30.6 tok/s; an attempt is ~19 min of wall time |
| 7 | **attempt_1 / attempt_2 assembly:**<br>• a1 = attempt 1's verified program;<br>• a2 = attempt 2's verified program when its test output differs;<br>• if only one verified program exists, a2 = the other attempt's best partial program run on the test input;<br>• the test input only last.<br>Keep `select_attempts_r5()` (it works: 0 correct candidates lost). | `select_attempts_r5` input = candidates of both attempts | [A] §2, [B] §3.2: pass2 = 0; 58 of 60 a2 slots were the test input. **[V]** 46 = 46 (no loss) |
| 8 | **End-of-attempt harvest:** when an attempt ends at 35k without an accept, run `check()` on any retained one-grid transform and keep a verified one. | Proven fix ON. | [A] §2: harvested programs gave 12 / 4 / 5 of the solved tasks |
| 9 | **Effort xhigh only.** | `E12_REASONING_EFFORT` default | [A] §3, [B] §2: LOW −10 to −15 queries (p 0.014-0.031) |

## P0-B — bugs and robustness (must be fixed in the same build)

| # | Fix | Where | Evidence |
|---|---|---|---|
| 10 | **Duck turn rollback:** when the per-turn yield (`yield_tokens` 2,048) or a stop fires after the first of two parallel tool calls, keep the turn and drop only the unexecuted call. | `vendor/inference/agent/tool_agent.py` L7094-7101, L7183-7186 | [B] §5.1a: 56 / 47 / 6 events; 0.52 / 0.46 / 0.06 M context tokens lost; 1ae2feb7 re-analysed train 0 four times |
| 11 | **`submit` called as a tool:** route it to the python `submit` helper instead of failing. | Duck tool dispatch | [C] §4: `Tool 'submit' is not defined` 22 / 28 / 23. **[V]** 22 / 27 / 23 branches |
| 12 | **Tool contract in the prompt:**<br>• exactly one tool, `python`;<br>• `check()` and `submit()` are functions inside it;<br>• never emit `<function=check>` / `<function=submit>`.<br>Plus a unit test: the tool names == {"python"} and the prompt contains the contract. | `arc2_r6_system_prompt()` | [G3] P0-3. A second layer under #11, because prompt rules are often ignored |
| 13 | **Autocheck names:** add `candidate` and `trans` to `_R6_AUTOCHECK_RE`.<br>Tests:<br>• `trans(grid)` autochecks;<br>• `trans(x, y)` does not;<br>• non-grid output is rejected;<br>• wrong-train is not retained. | `arc2_duck.py:1177`, `src/e16/arc2_duck_e16.py:1177` | [G1] Cause B (cbebaa4b). **[V]** the regex lacks them |
| 14 | **Sandbox imports:**<br>• a safe `sys` shim (`setrecursionlimit`, `maxsize` only; no `sys.modules`);<br>• allow `time`;<br>• a clear error when the model imports a prelude helper as a module ("`inspect_pair` is already defined; call it").<br>`numpy` stays blocked, and the error says so. | `vendor/.../python_tool_sandbox.py:281`, `ARC2_R6_ALLOWED_IMPORTS` (`arc2_duck.py:1018`) | **[V]** `sys` blocked 21 / 21 / 13 times per run; helper imports `inspect_pair`, `arc_overview`, `comp_sets` |
| 15 | **Server watchdog:** restart SGLang if it dies mid-run; more connection retries. | notebook engine cell | code audit H1 (Exp 14 Notes/37) |
| 16 | **Launch fallback:** if SGLang fails to launch with the E14S flags, retry with the W1 flags (today a failed launch means LB ≈ 0). | notebook engine cell | code audit H2 |
| 17 | Sandbox self-test and tokenizer probe on a fixed task in the notebook, not the first hidden task. | notebook | code audit M4 |
| 18 | Scheduler worker loop fully in try/except, so a worker cannot die silently. | `solve_all_r6` | code audit M5 |

## P1 — same build, smaller

| # | Fix | Where / notes | Evidence |
|---|---|---|---|
| 19 | **Attempt-2 order among unverified tasks:**<br>• tasks with partial progress (a transform with k < n train pairs) first;<br>• then tasks with no progress;<br>• tasks accepted in attempt 1 last (if D2 = yes). | scheduler rank for stage 2 (gold-free) | [G2] P1-2, [V] §7.2: progress is a priority signal, not a gate |
| 20 | **CPU rule-library fallback** instead of the test input when no program exists. | attempt assembly | 4-day review D. [G1] §3.3 |
| 21 | **Retained-state inventory:** each tool result lists the functions now defined in the sandbox. | tool result formatter | [G3] P2: NameError in 19 / 22 / 23 branches |
| 22 | **Bounded search guidance:** a node counter for DFS / packing; run the candidate on one train input before `check()` (shape + runtime). | prompt + helper | [G1] Cause C, e8686506. Only 2-3 timeouts a run, so small |
| 23 | Do not use `overfit_flags` for selection. | — | [B] §4: precision 42 / 30 / 63 % |
| 24 | pip / venv subprocess timeouts. | notebook | code audit M6 |
| 25 | Keep the two payload `.bin` files in the dataset until the hidden run ends. | process | code audit M6 |
| 26 | Write `submission.json` once at the end (not per branch). | notebook | code audit L2 |
| 27 | Abort the staging copy on timeout. | notebook | code audit L7 |
| 28 | Fix the stale labels (CONFIG sha, "w1", variant names). | notebook | code audit L5 / L7 |

## P1-R — reporting and measurement (needed to read the run)

| # | Item | Why |
|---|---|---|
| 29 | **Per-attempt record:**<br>• attempt no., seed, tokens used, end reason (accept / 35k / deadline);<br>• first transform and first check tokens;<br>• best train k/n;<br>• harvested or not. | To score B; [G1] diagnostic fields |
| 30 | **The B-risk metric:** attempts that reached 35k **with** a transform or partial progress, and the late-solve rate of attempt 2 on those tasks. | [V] §0: under rev 6, 21/46 (XH+E16) and 17/41 (XH) correct programs first appeared after 35k branch tokens. The run must show what B keeps or loses there |
| 31 | **Candidate fields:** `candidate_source_written`, `candidate_retained`, `check_attempted`, `check_executed`, `check_timeout`, `train_pairs_exact`, `host_verified`; AST class for written-but-never-executed code and for code only in the reasoning text; blocked-import counts. | [G1] §4 |
| 32 | Count deadline-cut tokens (server-side). | [A] §4, [C] §3.2: ~1 % |
| 33 | Fix the retract counter (it matches `retraction_policy`). | [A] §9 |
| 34 | Report accepted and harvested separately. | [A] |
| 35 | Record the effective request effort (the LOW report's top level said `xhigh`). | [G3] Fix 4, **[V]** |

---

## Open decisions (the user's)

| # | Decision | Data | My recommendation |
|---|---|---|---|
| D2 | Attempt 2 also on tasks whose attempt 1 was accepted, after all unverified tasks? | Wrong accepts 5 / 8 / 9; train-exact-test-wrong tasks 10 / 14 / 16. Fits only if the rate is ≈ 380 tok/s (see the capacity check). | Yes, last in the queue |
| D3 | E16 (early nudge + raw content) off? | Paired p ≈ 0.44-0.57; zero effect on the 89 nudged tasks; raw content kept 0 on xhigh. All three GPT reports: "experimental only". | Off (keep the flag) |
| D4 | Isolation INCONCLUSIVE → 6 workers (your approved safeguard)? | code audit M1 | Keep |
| D6 | The S1-60K notebook (LB 34.58) stays the fallback submission. | GPT P0 in all reports | Yes (no build needed) |
| D7 | Validation: EVAL120 G4B (as for E14S) or GPT's EVAL12 / EVAL24 matched set? | Bundle rule vs GPT's cheaper set; 12 tasks are very noisy | EVAL120 G4B after the CPU kit |

**Resolved by the user:**
- D1 = B (2 × 35k).
- D8 = independent Duck attempt. The Poetiq-style attempt 2 is a later arm.

---

## Held — not in this build

These stay listed and are not dropped.

| Item | Source | Why held |
|---|---|---|
| E15 engine experiment (fixes N43 #35-41, arms #42-53) | Exp 14 Notes/38-40 | User: off for now |
| #43567 backport, 14-16 streams, 11-of-12 fix, MTP, JIT / compile cache | N43 #18, #42-44, #64; [C] §8 | Engine unchanged in this experiment ([G3] rule; the user's E15 off) |
| E17 accuracy tests (KL, fp32 SSM, score pack, nucleus, cache parity, EOS, KV scales, INT4 vs FP8, target-only vs DFlash2) | N43 #54-63, #77, #90 | Separate experiment |
| Poetiq-style attempt 2 (one long think → transform → verify → iteration 2) | [G1-G3], N43 D8(b) | Later arm. Poetiq per solver-minute 0.113 = Duck 0.117 ([G3] §3) |
| Candidate-first / program-commit prompt, ≤ 2 inspections then a provisional transform | N43 #30, #74, #84 | Measured arm; E16's similar nudge had no effect |
| Generalization review / soft sanity score for train-exact programs | N43 #31, #86 | Arm. 78332cb0 transposed, b0039139 width, 247ef758 one cell |
| Facts vs compact-grid / object views | N43 #32, #78 | Arm |
| Hybrid Poetiq + Duck; S1-60K EVAL120; SGLang vs vLLM S1 | N43 #33, #62, #69-70 | Separate experiment |
| Stratified EVAL12 / 24 paired sets | N43 #34, #73, #89 | See D7 |
| Raw-prefix / Mamba cache diagnostic; DSL / mutation / ensembles | N43 #79, #80 | Later |
| Platform: wheelhouse as a Kaggle Model; 12 h margin; cancel API; **10-26 rules acceptance (user)**; **final deadline 11-02 (check)**; write-up `[PENDING]` | N43 #65-68, #71-72 | Process |

---

## Capacity check for B (estimate; it must be re-done with the built notebook before any GPU run)

**Hidden run:**
- 240 tasks, solve window ≈ 42,000 s.
- Effective rate: 341 tok/s measured (E14S with resumes); ≈ 380 tok/s expected without the resume re-prefill ([C] §3.4 model).

**Attempt 1:**
- About 23 % of attempts accept by 35k, at ≈ 20k tokens; the rest use 35k.
- 240 × (0.23 × 20k + 0.77 × 35k) ≈ **7.6 M tokens** → 22.2 ks at 341 / 19.9 ks at 380.
- **All 240 attempt 1s finish by ~47-53 % of the window.**

**Attempt 2 on the ≈ 185 unverified tasks:**
- 185 × ≈ 33.5k ≈ **6.2 M tokens** → 18.2 ks at 341 / 16.3 ks at 380.

**Total:**
- 13.8 M tokens → 40.4 ks at 341 (fits; ~1.6 ks left) / 36.2 ks at 380 (~5.8 ks left ≈ 2.2 M tokens).
- **D2** (attempt 2 on the ~55 accepted tasks, ~1.9 M tokens) fits only at ≈ 380 tok/s.

**Memory and context:**
- KV: 12 streams × ≤ ~55k context ≈ 660k in use of 917,816 (72 %, vs 47 % max today). B needs no cache for paused branches.
- Duck window ~100k ≤ the server context 155,648.

**EVAL120 G4B** (120 tasks at the hidden per-task share, ≈ 21,000 s window): the same shares hold.

## Validation path (each GPU step needs the user's go)

1. Build an E15B variant in a **new builder copy**. `build_e14.py` must not overwrite PUSHED notebooks (memory: never rebuild run notebooks).
2. Unit tests:
   - turn rollback;
   - `submit` routing;
   - regex;
   - sandbox shim;
   - assembly;
   - scheduler order;
   - no trim within 35k (CPU replay with a fake server).
3. Harness (commit / rerun / eval120) and the Kaggle CPU kit on the exact sha.
4. Written capacity check (above, recomputed).
5. EVAL120 G4B GPU commit (D7).
6. Submit only with the user's go.

---

**Superseded by Notes/02 (2026-10-24):** Notes/02 adds the delayer fix and real 14-16 streams (items 36-44), the GPU smoke step, D9 and the capacity check at 16 streams. The "Held" row above for "14-16 streams, 11-of-12 fix" no longer applies.
