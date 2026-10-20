# GPT forensic audit of Poetiq R5 EVAL120: validation (Claude, 2026-10-15)

Script: session scratchpad `order2/verify_gpt_r5.py`. Every GPT number was recomputed from the files.

## Claims checked

| GPT claim | Our check | Verdict |
|---|---|---|
| 78/172 (48.61 %); DIRECT 58 (36.94 %); union 92 (56.81 %); both 44 / Poetiq only 34 / DIRECT only 14 | identical | ✅ |
| attempt_2 rescued only `71e489b6#0`; 38 distinct attempt pairs; 9 cases with attempt_1 right and attempt_2 wrong | identical | ✅ |
| **A.** Two iteration-2 caps fell below 77,824: `9aaea919` 76,758 (prompt 21,290); `d8e07eb2` 74,361 (prompt 23,687) | identical; minimum cap 74,361, above 49,152 | ✅ minor: 2 of 172, at most −3.5k tokens |
| **C.** KV peak 99.5 %, mean 76.4 %, 9 preemptions | 99.5 / 76.4 / 9; 11 of 345 samples above 95 % | ✅ **important: Poetiq at 16 sequences is at the KV limit** |
| **D.** Gate policy 89/172 = 55.14 % | identical | ✅ offline arithmetic; not schedule-proven |
| Misses by status: V1 17, V2 11, U1 17, U2 49 = 94; wrong shape 15; right shape with 1–10 cells 21 | identical | ✅ |
| Warning signals on verified tasks: new colour 7/15 caught vs 20/49 false alarms; larger test 11/15 vs 24/49 | identical (new colour vs train inputs; 6/15 vs 20/49 if train outputs count too) | ✅ weak signals |
| 13 iteration-1 forced closes passed train → 11 correct inputs; `20270e3b`, `271d71e2`, `8698868d` fail | identical | ✅ |
| Forced phase-2 median 2,107 tokens; "def transform" in 113 of 172 thinking traces | identical | ✅ |
| No loop degeneration in the Poetiq traces | 0 tails with zlib ratio < 0.05. **Also: our LOOPSTOP detector gives 0 flags on all 172 full traces (11.25 M tokens)** | ✅ |
| Iteration 2: train score improved 32 / unchanged 8 / worsened 12; 16 newly verified → 12 inputs; 6 verified-2 tasks with 0 hits; the 8 example progressions | identical | ✅ |
| Tokens: iteration 1 8.408 M, iteration 2 3.241 M; DIRECT on never-verified tasks 5.360 M → +11 = 2.05 per M | identical | ✅ |
| First-score bins of the iteration-1 failures → DIRECT-only 5 / 1 / 5 / 2 | identical | ✅ |
| Prompts 7,944 vs 4,864 on average; total 1.366 M vs 0.837 M; p2 tokens 272,114 vs 71,862 | identical | ✅ |

## Where we add to or differ from GPT

1. **Iteration 2 vs DIRECT per token.**
   - Iteration 2 yields at most 12 inputs from 3.24 M tokens: **≤ 3.70 per M**.
   - DIRECT on never-verified tasks yields 2.05 per M.
   - So iteration 2 is *not* less efficient than DIRECT. The two are complementary; DIRECT still adds 10 inputs on the iteration-2 tasks.
   - GPT ranks "DIRECT routing" first. We agree on testing it, but as **cost-aware routing between iteration 2 and DIRECT**, not DIRECT replacing iteration 2.
   - From the bins, DIRECT helps most when the first train score is exactly 0 (5 inputs over 16 tasks).
2. **ORDER2 for Poetiq.** GPT doubts that an ordering learned on DIRECT transfers. It does: Poetiq task score by order_score 2 / 1 / 0 = **59.6 / 42.1 / 33.3 %**, monotone like DIRECT (48.1 / 30.8 / 20.0). Adding ORDER2 to Poetiq is supported.
3. **Risk scoring of verified programs (GPT rank 2) has low payoff:**
   - on tasks verified at iteration 1, DIRECT adds 1 input;
   - on the 15 verified-wrong tasks, 2;
   - the cheap warning signals are weak (7/15 caught vs 20/49 false alarms).
   - **Park.**
4. **The KV limit (finding C) is the most important new fact for the next notebook.** Keep 16 sequences, and do not add concurrency to Poetiq.

## Order we now recommend (one change at a time, user's go)

1. **R5 + ORDER2.** A small change, and the LB candidate once E9's hidden result is known.
2. **Hybrid:** after round 1, cost-aware routing of the train-failed tasks between Poetiq iteration 2 and DIRECT as attempt_2. Measure extra inputs per M tokens.
3. **Feedback trimming** to keep the full 77,824 cap (2 of 172 affected): low priority.
4. **Parked:** verified-program risk scoring, prompt "commit earlier", concurrency changes.
