# 10 — 2X35K GPU commit (stopped at ~41 min) findings, the 3f / 3g runs, and rev 3h

**Date:** 2026-10-25.

## GPU runs of 2026-10-25 (20 eval tasks, both attempts; commit mode)

| Run | Account | Notebook sha | Status |
|---|---|---|---|
| 2X35K (rev 3) | foysalemonshanto | `0f0d6f67` | **stopped by the user at ~41 min**; output in `Experiment Results/Kaggle Runs/E15B SUBMIT commit1 (foysal, …)` |
| 40K35K (rev 3g) | foysalemonshanto | `560ed8b8` | running; **submitted to the competition by the user** |
| 45K30K (rev 3g) | esfersami50 | `5d8069b9` | running |

## 2X35K findings

The run was stopped at ~41 min, so its output is partial: 18 attempt-1 results and 4 attempt-2 results; 10-13 attempt 2s were running.

### Engine
- READY in 527 s.
- KV 852,152 tokens; isolation 16/16.
- 16 running and queue 0 at every sample: **the delayer fix works.**
- Gen throughput:
  - 257-311 tok/s in the first samples;
  - then 430-442 tok/s. That is about the 12-stream decode of E14S (415-434), so 16 streams give no aggregate decode gain.
- Cache hit 0.97-0.99.
- KV peak about 0.59 at 32 min.

### Attempt 1 (35k)

| Result | Count | Tasks |
|---|---|---|
| Accepted and correct | 7 | 16de56c4, 1818057f, 1ae2feb7, 269e22fb, 2ba387bc, 45a5af55, bf45cf4b |
| Accepted but test-wrong | 5 | 221dfab4, 247ef758, 4c3d4a41, 62593bfd, 9aaea919 (9aaea919 was 3/3 in the EVAL120 runs) |
| At budget, no accept | 6 | 135a2760, 136b0064, 65b59efc, 898e7135, 9385bd28, b5ca7ac4 |

- 135a2760: its partial program (train 0/2) gives the correct test output, and the partial tier would select it.

### Commit turn
- 10 attempts reached it; 8 of them made a tool call.
- **269e22fb was solved in the commit turn** (commit at 30,904, accepted at 32,331).

### Problems found → fixes

| # | Problem | Fix |
|---|---|---|
| 1 | A tool call cut at the request cap loses its code. 898e7135: code lost. 269e22fb: rescued by the commit turn. | **rev 3h:** the cut call's code is kept; its complete leading definitions go to the end-of-attempt harvest (sandbox + host verifier). 6 unit tests. |
| 2 | 65b59efc and 9385bd28 never wrote a `transform` in 35k, and their commit turn made no tool call. | **rev 3h:** the nudge is scaled to the attempt budget (min(24,000, 0.69 × budget); a 30k attempt: 20,700). |
| 3 | 6 of 18 attempts ended at budget with no accept | design: 40K35K / 45K30K measure it |
| 4 | 5 of 12 accepts are test-wrong | known rate; attempt 2's second answer is the remedy |
| 5 | Log lines printed twice; stale config prints | cosmetic, not changed |

## Other fixes in rev 3h

**Auditor 4:**
- **#3:** 40K35K commit window 92 min / budget 115 min, as for 45K30K.
- **#4:** `setup_failed` in `b_summary`, plus `cut_code_saved` / `cut_code_harvested`.
- **#6:** the fair-share pause uses the session's own cap.
- **#7:** request timeout 3,600 s (the 45k first request asks for 40,904 tokens).
- **#9:** comments.

**Process:**
- The Kaggle CPU kit takes the Duck bin that the notebooks load. Kit v2 took the old `19192e8006e0`, so its 6 unit "failures" were the old code.
- Harness kill2 uses a slower fake server. Kit v3 on Kaggle showed a 34-s solve that ended before the second kill.

**Kaggle CPU kit v3** (push-time shas, the right package): unit 86/86; 6 of 7 harness modes passed; kill2 was the timing issue above and passes 28/28 after the harness change.

## Rev 3h builds (new names; the pushed rev 3g notebooks stay as run)

| Notebook | sha |
|---|---|
| `Q27-XHIGH-E15B_SUBMIT_45K30K_S16_R3H_DUCK_SGLANG.ipynb` | `9ee0aa4e` |
| `Q27-XHIGH-E15B_SUBMIT_40K35K_S16_R3H_DUCK_SGLANG.ipynb` | `a7837609` |

- **Package:** `q27x_e15b_duck_85a805b5d8bb.bin`.
- **Unit tests:** 93/93 PASS.
- **`verify_items.py`:** ALL CODE ITEMS PRESENT (both).
- **Harness:** see `Experiment Review/`.

**Not pushed.** Next: the 40K35K and 45K30K run results. More problems found there go into the next revision.
