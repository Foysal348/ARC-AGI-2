# E33: Duck + Poetiq + Direct attempt chain on the E29 Flash-Next engine (design, user 2026-11-02)

**Status:** fixed spec. The user decided the scheduler and the two open points below. Builders follow this text exactly.

## 1. Base
- **Engine and notebook:** E29_DFLASH_D3S, the GPU-smoked engine.
  - Qwen3.8-Flash-Next klee100 3bpw on Merlin (vLLM-0.32 fork), TP2×PP2.
  - DFlash2 D3s (dflash_seam 341d859b with GRD), DFLASH_DRAFT_EMBED_HOST=1, QSA 128/64.
  - 24 seqs. D ladder from 7,665,090,560 B; MTP fallback at 7,933,526,016 B / 24 seqs / I8.
  - Builder: `experiments/Experiment 29 - Final production K100 (planned)/Build/build_e29_final.py`. Smoke result: `Experiment Results/E29_SMOKE_hosen42_v1_2026-11-02.md`.
- **Unchanged and byte-identical to E29:**
  - CELL 0 (identity submission), the writer, the rerun path (RERUN_PROD = 12 h - 3 min, E18 R2), the clock;
  - the boot ladder, the F2 probe, the watchdog, LIVE (extended for the new phases), the submission check.
- **E29's DIRECT scheduler diff** (`e29d_sched_patch.py`, C1 scope / early DIRECT) is NOT used. E33 has its own diff on the E29 post `arc2_duck.py` (sha d8603109…).

## 2. Methods
| Stage | Method | Budget | Train pass means |
|---|---|---|---|
| A | Duck (arc2_duck agent), attempt 1 | T1 = 104,000 | Duck kind TRAIN_PASS (all train examples exact) |
| B | Duck re-run (E28 rule: B1 = 104,000 same T1 for B1 / FAIL kinds; B2 = 120,000 + the B2 window for a cap-cut attempt) | as E28 | TRAIN_PASS |
| C | Poetiq iteration 1 | 60,000 think + 8,192 answer (S1-60K, E18 R2) | the `transform` program is exact on every train example (Poetiq score 1.0) |
| D | Poetiq iteration 2 = feedback iteration | 60,000 + 8,192 | as C |
| E | Direct (e29_direct.py = the E31 solver) | 104,000 think + 8,192 answer, temperature 1.0 | no train check: E is the last stage |

**Poetiq (E18 R2 = S1-60K, LB-proven).** Notebook `experiments/Experiment 18 - S1-60K on vLLM 0.32 (planned)/Experiment Notebooks/Q27-XHIGH-E18_S1_60K_VLLM032_S20_R2.ipynb`, scored version https://www.kaggle.com/code/foysalemonshanto/q27-xhigh-e18-s1-60k-vllm032-s20?scriptVersionId=363633621.
- Copy its Poetiq core verbatim: prompt build, problem format, diagrams, code parse, sandbox (subprocess + rlimits), verify, per-example feedback (correct / shape mismatch / wrong-cells diff / error), solution history with scores (max_solutions 5, improving order), voting, two attempts.
- Settings: temperature 1.0, prompt variant 1, shuffle on, 1 expert, per-request seed, two phases. Phase 1 = think to 60,000, then forced close with STOP_THINKING_TEXT; phase 2 = answer 8,192.
- Iteration 2 (stage D) receives iteration 1's solution(s) and feedback in Poetiq's own format (`poetiq_create_examples`).
- Only the transport changes, to the E31 / E29 DIRECT client: streaming token-id completions on the booted Merlin server, the Flash-Next template (enable_thinking True, reasoning_effort xhigh), FN_STOP_IDS, EOS from generation_config, fail-closed stop ids, resends keep streamed ids, request errors are never a finish.

## 3. The scheduler (attempt based)
- **Attempt 1 chain, for every task:** A → B → C → D → E.
  - STRICTLY SEQUENTIAL per task: the next stage is queued only when the previous stage ended without a train pass.
  - The chain stops at the first train pass. A TIME end (solve window over) queues nothing.
  - B is queued only for the E28 B kinds (`e28_b_phase`): B1 for B1 / FAIL, B2 for B2. Any other non-pass A kind goes straight to C.
- **Attempt 2 chain:** the same stages A′ → B′ → C′ → D′ → E′ on separate branches.
  - Queued for a task as soon as its attempt-1 A ended.
  - Order:
    - group 1: tasks whose attempt-1 A was NOT a train pass, in task order;
    - group 2: the rest, ascending attempt-1 A tokens (the E28 C2 key).
  - Stops at the first train pass of attempt 2.
- **Priority:** every attempt-1 item comes before any attempt-2 item. Inside an attempt:
  - waiting B2 first (E28 rule);
  - then A, B, C, D, E (earlier stage first, like Poetiq's "iteration 1 before iteration 2");
  - inside a stage: group, then task order.
  - Attempt-2 items fill slots that attempt 1 cannot use. Supply (S1 / S2) runs only when both attempt queues are empty, as now.
- **Admission:**
  - the E28 KV reservation hook (`E28_TRY_RESERVE`) and force-when-idle stay as they are;
  - Poetiq / Direct items start only when their thinking can reach the 60,000 floor before the drain point (user rule). Otherwise that stage is skipped for that task, recorded, and the chain ends.
  - Each solver's own drain (forced close, then answer) and hard stop stay as E31 / E18.
- **Branches:** every (task, stage, attempt) has its own branch id, its own reservation key and its own record. A task never runs two stages of the same attempt at once. Attempt 1 and attempt 2 of a task may run at the same time.

## 4. Answers (user accepted 2026-11-02)
- **attempt_1** = from the attempt-1 chain, in this order:
  1. the first train-pass answer (Duck accepted, or the train-exact Poetiq program run on the test inputs);
  2. else the best train match among the attempt-1 Duck / Poetiq records (E28 best-train rule, with Poetiq's train score as the match);
  3. else the attempt-1 Direct grid;
  4. else the E28 fallbacks;
  5. else the test input.
- **attempt_2** = from the attempt-2 chain by the same rule, when it differs from attempt_1. Else the next distinct candidate from attempt 1 (e.g. its Direct grid, or the second best). Else the E28 fallbacks. Else the test input.
- **Every query:** multi-test grid K → test K. Valid 0-9 grids only.

## 5. Profiles and outputs
- **Profiles:**
  - commit E28EVAL120 (6 h, 120 eval tasks);
  - a SMOKE copy (one profile line differs);
  - a SUBMIT copy (= the smoke bytes, a separate slug);
  - rerun RERUN_PROD (all hidden tasks, 12 h - 3 min).
- **Switches** (plain CELL 1 literals): each method on / off (Poetiq C+D, Direct E, attempt 2), the Poetiq budgets, the Direct config. A bad value turns only that item off.
- **Outputs:**
  - LIVE: one line per finished stage (task, attempt, stage, train result, exact vs gold in eval, totals).
  - Report: per stage counts / tokens / time, stage reached per task, skipped-for-time counts.
  - Save every thinking trace (user rule).

## 6. Validation
- Local unit tests for:
  - the Poetiq client (fake token server);
  - the scheduler (the E29 scripted-Duck harness `Build/e29_sched_tests.py` with stub Poetiq / Direct hooks);
  - the answer rule;
  - edge cases.
- Kaggle CPU kits (the kit mirrors the notebook launch); an audit zip; a ~70-min GPU smoke; then the 6-h EVAL120 and the submission (user go for every GPU commit).
