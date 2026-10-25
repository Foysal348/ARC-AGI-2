# E12 rev 5 runs (2026-10-20/21): EVAL6R5 (xhigh) and EVAL12R5_LOW (low)

**Runs:**

| Run | Kernel | Notebook sha | Status | Notebook time |
|---|---|---|---|---|
| EVAL6R5 (xhigh) | `hosen42/q27-xhigh-e12-duck-repl-sglang-eval6r5` v1 | `f4fed1ac` | COMPLETE | 69 min (solve 58 min) |
| EVAL12R5_LOW (low) | `foysalemonshanto/q27-xhigh-e12-duck-repl-sglang-eval12r5-low` v1 | `26fff41e` | COMPLETE | 123 min (solve 110 min) |

**Sources:**
- Outputs: `Kaggle Runs/E12 SGLang EVAL6R5 run1 (hosen42, 2026-10-20)/`, `Kaggle Runs/E12 SGLang EVAL12R5_LOW run1 (foysal, 2026-10-20)/`.
- Script: `Analysis scripts/rev5_analysis.py` (+ `_output.txt`).

## Results

| Run | Inputs | Fully solved | Task-norm | Accepted branches | Generated tokens | Per branch | Solve time |
|---|---|---|---|---|---|---|---|
| rev 4 EVAL6 (xhigh, 60k, T 0.7) | 4/8 | 3/6 | 50.0 % | 4/12 | 1,217,904 | 101k | 62 min |
| **EVAL6R5 (xhigh, rev 5)** | **5/8** | **4/6** | **66.7 %** | **9/12** | **846,080 (−30 %)** | 70.5k | 58 min |
| **EVAL12R5_LOW (low, rev 5)** | **10/19** | **7/12** | **58.3 %** | 17/24 | 1,743,889 | 72.7k | 110 min |

### Per task (which attempt hit, and which candidates matched)

| Task | rev 4 xhigh | R5 xhigh | R5 low |
|---|---|---|---|
| 136b0064 | ✓ a1 | ✓ a1 | ✓ a1 |
| 898e7135 | – | – | **✓ a1 (first Duck solve ever)** |
| 135a2760 | ✓ | ✓ | ✓ |
| b5ca7ac4 | – | **✓ via attempt_2 (b0)** | – (both attempts wrong) |
| 4c3d4a41 | ✓ 2/2 | **– overfit**: both branches accepted wrong programs | – overfit |
| 62593bfd | – | **✓ 2/2 via attempt_2 (b0)**; b1's accepted program was wrong | – |
| 247ef758 | n/a | n/a | ✓ 2/2 |
| bf45cf4b | n/a | n/a | ✓ (14-19k tokens per branch) |
| f931b4a8 | n/a | n/a | ✓ 2/2 |
| 65b59efc | n/a | n/a | ✓ 2/2 |
| 9385bd28 | n/a | n/a | – |
| cbebaa4b | n/a | n/a | – |

**Totals:**
- On the 6 shared tasks: xhigh 4/6, low 3/6; union **5/6** (only 4c3d4a41 missing).
- On the 4 tasks Duck rev 1 solved: low kept all 4. The 2 mid tasks: 0/2.

## What the rev 5 fixes did (measured)

| Fix | Evidence |
|---|---|
| B2 two attempts | **3 of the 5 xhigh hits came from attempt_2** (pass1 2, pass2 3). Before rev 5, attempt_2 hit 0 of 56 times |
| B1 harvest | Every branch that found an exact function also kept it (kept/verified per branch in the script output). This time every hit was also submitted, so harvest added no unique hit; it worked as a safety net |
| B3 estimate | **Prompt estimate vs the server's `prompt_tokens`: median +0.4 % / +0.5 %** (p10 −3.8 % … p90 +3.5 %). Only 1 + 21 requests over by more than 5 % |
| B3/B4 trims | ≤ 1-2 per branch; no system-only resets |
| B6 | 0 cut tool calls (no reply reached the cap with a call) |
| Timeout | 0 |
| Tokenizer | loaded in both runs |
| B5/B9 loops | **Repeated-line share 0-3 %** in xhigh branches (rev 4: 29 %); low 0-8 %. Replies hitting 49,152: 2 (xhigh), 2 (low) |

## Coverage arithmetic (the open problem)

- **Per task:** xhigh 141k tokens, 9.7 min; low 145k tokens, 9.1 min.
  - Low is **not** cheaper per branch: it makes many more, shorter requests (449 vs 127 requests).
- **Throughput:** overall 244 tok/s (xhigh) and 265 (low) vs 325 in rev 4.
  - Shorter replies mean more requests and more prefill: uncached prefill is now 0.97M vs 0.85M generated (xhigh).
  - Small-run tail: 130 decode lines with 1 running.
- **Context per request (prompt + reply):**

  | Run | p50 | p90 | max |
  |---|---|---|---|
  | R5 xhigh | 58k | 74k | 95k |
  | R5 low | 46k | 75k | 115k |

  - 12 streams x 58k = 697k fits the KV of 866,744 at the median; at p90 it would be 890k (slight evictions).
  - **10 streams fit up to p90** (750k).
- **Full coverage target** (240 tasks in about 11.3 h): about 2.8 min per task. We are at about 9-10 min, so about **3.4x** short.
  - Kris Dreemurr (discussion): about 35k tokens per attempt.
  - Ours: about 70k per branch.

## Next levers (to be decided by the user)

1. **More concurrency (10-12 streams)** now that contexts are smaller. Rev 4 measured 6 → 12 running: 368 → 431 tok/s.
2. **Per-branch cap:**
   - Failed branches still burn about 105-115k.
   - Accepted branches finished at 14-98k.
   - A lower cap needs care: some accepts came at 88-98k.
3. **Prefill share:** keep prefixes stable across requests.
4. **Overfit** (4c3d4a41 in both runs, b5ca7ac4 / 62593bfd in low):
   - attempt_2 diversity helps;
   - the selection order could be revisited (in xhigh the correct answer was attempt_2 twice).

## What if each attempt had about 35k tokens (Kris Dreemurr's budget)? (`Analysis scripts/budget_sim.py`)

**Method:** every host-verified candidate records the branch's generated tokens when it was found. A candidate counts if the branch had generated fewer than C tokens before the request that produced it (Duck's cap is soft). The notebook's own `select_attempts_r5` then picks the attempts, scored with the truth. Time is scaled by the token ratio.

| Cap per branch | EVAL6R5 xhigh: tasks / inputs | tokens per task | 240-task need | EVAL12R5 low: tasks / inputs | tokens per task | 240-task need |
|---|---|---|---|---|---|---|
| 15k | 0/6, 0/8 | 70.9k | 19.3 h | 1/12, 1/19 | 49.8k | 12.5 h |
| 25k | 0/6, 0/8 | 73.4k | 20.0 h | 1/12, 1/19 | 67.1k | 16.9 h |
| **35k** | **1/6, 1/8** | 90.9k | 24.8 h | **2/12, 3/19** | 83.3k | 20.9 h |
| 50k | 2/6, 2/8 | 105.8k | 28.9 h | 4/12, 6/19 | 98.9k | 24.8 h |
| 70k | 4/6, 5/8 | 125.3k | 34.2 h | 5/12, 8/19 | 123.6k | 31.0 h |
| none (actual) | 4/6, 5/8 | 141.0k | 38.5 h | 7/12, 10/19 | 145.3k | 36.5 h |

**Why it collapses:** our agent reaches its **first `check()` only after about 55k tokens** (both runs: median 56.6k xhigh, 57.5k low), and about 100 % of those tokens are reasoning.
- **xhigh:** the first reply alone is p50 33.6k (p75 47.9k) before any tool call.
- **low:** the first reply is about 200 tokens, but it takes a median 8.5 requests (each with thinking) to reach a `check()`.
- **First verified candidate per branch:** median 58k (xhigh) and 56k (low).
- **First correct candidate:** 32k-62k (xhigh) and 14k-98k (low).

**A soft 35k cap does not save much:** the request that crosses the cap still finishes, often a 30-49k xhigh reply. Tokens per task fall only from 141k to 91k, while accuracy drops from 4/6 to 1/6.

**Conclusion:** to reach about 35k per attempt at useful accuracy, the agent must reach a tested `transform` much earlier: about 15-25k instead of about 55k. A cap alone gives neither the coverage nor the accuracy. A hard per-attempt budget (clamping `max_tokens` to the remaining budget) would be needed to really bound tokens.

## Where the ~55k pre-check tokens go (2026-10-21; `Analysis scripts/precheck/`)

**Measured over all 36 branches:**
- In 35/36 branches the model notices a misread of its own hand counting before the first check.
- 37 % of xhigh pre-check tokens (20 % in low) are generated after the model has already said it will use code in that same response.
- The 49k output reserve forces history trims at about 82k of context, which caused one overfit accept (low 4c3d4a41_b0).

**Fast branches** make transform + check their first real call after a think.

**Candidate changes:** R6-1..R6-6 in `Experiment Review/E12 next fix list (running, 2026-10-20).md`.
