# One expert with more iterations vs two experts (2026-10-26)

Script: `experts_vs_iters.py`. Output: `experts_vs_iters_out.txt`. No GPU was used.

**Units.**
- Yield is counted in test queries hit (172 queries = EVAL120).
- ARC scoring counts a query if either of its two attempts is right, so a second expert's answer works as attempt_2.
- Cost is output tokens (p1 + p2) from `poetiq_iterations.jsonl`.
- E15P has no iterations file. Its tokens are estimated as 389 tok/s x 6 h = 8.4M.

## Yield per million output tokens

| Option | Data | Queries per M tokens |
|---|---|---|
| Iteration 1 (the first attempt) | R5 6.66, D66 2.23, L66 2.41, E11-05 2.65 | 2.2-6.7 |
| **Iteration 2 (tasks that failed iteration 1)** | R5 3.70, D66 2.54, L66 3.30, E11-05 5.26 | **2.5-5.3** |
| Iteration 3+ | D66 0.52, L66 0.00, E11-05 0.00 (its 3-5) | **0-0.5** |
| **A second independent expert** | D66 to L66 2.06, L66 to D66 1.48, R5 to E15P 1.90, E15P to R5 2.58 | **1.5-2.6** |

**Unions.**
- D66 28 + L66 34 = union 50 of 103 queries.
- R5 78 + E15P 64 = union 94 of 172 queries (+16 over the better run).

**Caveat on option B.** These pairs differ in configuration: D66 used 77.8k and L66 122.9k; R5 ran on vLLM for 12 h and E15P on SGLang for 6 h. Two experts with the same configuration are likely more correlated, so 1.5-2.6 is an upper estimate for a same-config second expert.

## Order of spending the token budget

1. Iteration 1 on every task (coverage first).
2. Iteration 2 on every task that failed iteration 1: about 1.5-2x better per token than a second expert.
3. Only then a second expert on still-unsolved tasks, submitted as attempt_2.
4. Never iteration 3+: 0-0.5 queries per M tokens.

## Budget at the engine speeds

The hidden test is 240 tasks.
- Iteration 1 costs about 8.4M tokens per 120 tasks (R5), so about 16.8M for 240.
- Iteration 2 costs about 62k tokens per failed task.

| Engine | Tokens in 12 h | What fits for 240 tasks |
|---|---|---|
| 389 tok/s (now) | 16.8M | Iteration 1 only, just barely (coverage binds) |
| 650 tok/s (matrix target) | 28M | Iteration 1 (16.8M) + iteration 2 on about 60 % failed (about 144 x 62k = 8.9M) = 25.7M |
| about 950-1,000 tok/s | 41-43M | All of the above + a second expert on unsolved tasks (about 15M) |

## Conclusion

A faster engine should first buy full iteration-2 coverage. A second expert pays only beyond about 950 tok/s, or with a lower per-attempt cap. Lowering the cap needs the user's approval, because the 60k floor rule applies.

## Does a larger thinking cap solve more? (same data, train-pass rate by thinking length)

| Run | Cap | Attempts that hit the cap | Train-passes after more than 60k tokens | Train-passes after more than 78k tokens |
|---|---|---|---|---|
| R5 EVAL120 | 77.8k | 93 of 172 (54 %); only 20 % of those pass | 37 of 71 | 0 (cap) |
| D66 HARD66 | 77.8k | 100 of 151 (66 %); 12 % pass | 20 of 33 | 0 (cap) |
| L66 HARD66 | 122.9k | 11 of 123 (9 %) | 34 of 43 | **22 of 43** |

**Per task: yes.**
- With 122.9k, only 9 % of the attempts hit the cap. Half of L66's train-passes came after 78k tokens; a 77.8k cap would have forced those attempts closed.
- On the same 66 hard tasks, L66 got 34 queries vs D66's 28 for the same total tokens (10.7M vs 10.8M). L66 ran fewer iterations.
- Per million tokens: L66 3.19 vs D66 2.59 queries.

**Over 12 h on the LB: only with enough speed.**
- E9 86k scored LB 22.92, below 49k's 26.94: coverage bound.
- S1-60K scored LB 34.58, the best so far, above S49's 32.08.
- A larger cap costs coverage of the 240 hidden tasks at the current speed.

**Order.**
1. A faster engine first.
2. Then iteration 2 on all failed tasks.
3. Then a larger cap for hard / cap-hit tasks, or a second expert.

Pause-resume continuation was negative before (`coverage-levers` memory), so a longer second attempt should be a fresh attempt with a higher cap, not a continuation.
