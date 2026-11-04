# E28 scheduler agent: rev 3 (classification fix + T1 104k / B2 120k), verbatim

## E28 SCHED rev 3: the classification fix and your T1 104k / B2 120k decision are both in. 33 tests, all pass on Python 3.12 (7 full runs).

**The two notebook agents must re-embed the module.** The N1-N4 edit texts did not change.

### Final shas (all LF, no CR), in `D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Build/`

| What | sha256 |
|---|---|
| `e28_sched_patch.py` (69,195 B, 1,221 lines) | `108bb75cab0e3a4499621c0e615299b48ccbc783b1eb59cd630f6ed1f8d16730` |
| `arc2_duck.py` pre | `b9f67010237becf9a0c2631e14f48acd97f5883b4ef357c93becf4fa16543128` |
| `arc2_duck.py` post | `35b8d8577ab05be60e4e66f80400e57ccec3005dfad9b5290d9fa6f53adee57f` |
| patch diff (same 8 hunks, +744 / −3) | `8089ee65a67c589afb22ac51035c2c181e6a8d6ff954612404374e8c5af9dae1` |
| applier (unchanged) | `b871f0e409c5f44ad22a44cb61051138efc220657010b3a85ba4b53a72340e88` |
| `e28_sched_tests.py` (1,110 lines, 33 tests) | `0439d632c6b3995bccb75babb11515d02952f0bb6432378d7c09126518a616a0` |
| `E28_SCHED_API.md` | `6b7faa8b0f2595d1bda88acc42065fb1e2b6706ded42d32deae7c942cbb837fa` |

The API file is updated: constants, `server_env`, the new classification order, the N1 / N2 / N4 block numbers, the timeout note, the dropped 262k items, and the status section with the shas above.

### 1. Classification (`e28_classify`): first match wins
1. TRAIN_PASS: a host-verified full train match.
2. B2: a cap verdict. `E28_CAP_VERDICTS` is now CONTEXT_EXHAUSTED, **INCONCLUSIVE_CONTEXT_LENGTH**, BUDGET_LIMITED.
3. TIME: an admission-window note.
4. B2: explicit budget evidence (`b_events.tiny_request_skipped`, or `commit_at_tokens` set), even when a note exists.
5. TIME: `status == "paused"`, at the deadline, or the run ending within 90 s of the window end.
6. FAIL: any other verdict or note; re-run as B1.
7. B2: the near-budget heuristic, tokens ≥ budget − 1,024.
8. FAIL if there are no tokens; otherwise B1.

### 2. Your budget decision
- `T1 = 104000`, `B2_TOKENS = 120000`, `MAX_MODEL_LEN = 131072`, `B2_WINDOW = 130560`.
- `server_env()` = `{"E28_SCHED": "1", "E28_T1": "104000", "E28_B2_TOKENS": "120000", "E28_B2_WINDOW": "130560"}`. The `e28_config` defaults match.
- **Peaks (N1 / N2 / N4 use the same formula, so the edit texts are unchanged):**
  - B2 peak = min(131,072, 136,384) = 131,072, which is 32 + F blocks.
  - T1 peak = 120,384, which is 30 + F blocks.
- **Timeouts:** the default B2 request timeout is base × 120k / 104k. When `E28_B2_REQUEST_TIMEOUT` is set, as the notebooks do with 9,000 s, that value is used. The first B2 request cap is 115,904 tokens.

### 3. New and updated tests
All of these pass:
- budget cut + note → B2 (`commit_done_tiny_rest`);
- forced commit + note → B2;
- INCONCLUSIVE_CONTEXT_LENGTH → B2, then `e28_b_phase` = B2;
- paused near the budget → TIME;
- at the deadline near the budget, and recorded window end near the budget → TIME;
- near the budget with no time cut → B2 (`budget_spent`);
- an HTTP 500 note, and a SERVER_ERROR verdict → FAIL;
- accepted, even with a cap verdict and a note → TRAIN_PASS;
- a normal non-accepted end → B1.

**INCONCLUSIVE B2 plan test (scheduler run):**
- the B2 plan is 120,000, window 130,560, session cap 120,000;
- with the env timeout set, the B2 request timeout is 9,000; the agent budgets are 120,000 (B2) and 104,000 (A);
- without the env, the timeout is 7,200 × 120k / 104k.

**Real agent test:**
- B2 gets 120,000, window 130,560, "about 120,000" in its prompt, first cap 115,904;
- C gets 104,000, window 130,560, "about 104,000" in its prompt.

**Other updates:**
- The backfill and N4 tests were re-sized to the new block counts (capacity 73: two T1 attempts fit, a B2 fits only alone).
- The backfill test is now deterministic: one long attempt 1 keeps blocks busy while the C items backfill. This removed the earlier 1-in-6 flake.
- Slots-mode no-idle test, 120-task and reservation simulations, env-off identity, patch apply / revert, and the N1-N4 export edits all pass with the new numbers.

### 4. Real records (all 90 attempts), unchanged
- **B2:**
  - klee100 16b78196 a2 (CONTEXT_EXHAUSTED, 95,825);
  - klee100 20270e3b a1 (`commit_done_tiny_rest` through its explicit `tiny_request_skipped`, 103,427);
  - W4 2b83f449 a1 (`commit_done_tiny_rest`, 103,761).
- **TIME:** the 5 window cuts that had no train pass.
- B1 and FAIL are empty, and end_reason matches the recorded report on every attempt.
- About 20270e3b a1: its run ended 978 s **before** the window end, not near it. It comes out B2 from the explicit evidence and is not affected by the time-cut rule. 2b83f449 a1 ended 330 s before the window end and is also B2 from explicit evidence.

### Open
- Still to do: the Kaggle CPU kit run.
- The rev 1 risks still stand: no fixed `LOCAL_ANALYZER_SEED`, and CELL 7 counts branch 2 as attempt 3+.
