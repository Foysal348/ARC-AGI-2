# Transcript reader B: EVAL6 branches 62593bfd_b0/b1, b5ca7ac4_b0, 4c3d4a41_b0

Folder `runs/EVAL6_q27x/duck/`. Token counts are from `branch_result.json` usage; line numbers are in `transcript.txt`.

## 62593bfd_b0 (failed, 159,464 tokens)

1. **Request 1** (L210): `length`, 126k reasoning characters (60,000 tokens), no tool call. All by hand; it never found the rule (shapes with a congruent twin go up, the rest down).
2. **Request 2:**
   - The prompt was 9,395 tokens, which is the system prompt alone: the 60k reply was dropped. The model restarted ("Let me analyze", L3923).
   - 39,464 tokens, then one tool call (L4878) with the dict-unpack bug `for i,(inp,out) in enumerate(train)`. Garbage result (L4920).
3. **Request 3:**
   - The prompt was 9,440 tokens; `prompts/prompt.log` shows message_count 2. The reasoning, tool call and result were all gone.
   - The resume still said "Your tool results above are still valid".
   - It restarted again: 60,000 tokens, `length`.
4. **Overshoot:** the total was 99,464 after request 2. Because the cap is checked only between requests, one more full 60k request was allowed.

## 62593bfd_b1 (exact, test-correct function, never submitted; 102,547 tokens)

1. **Request 1:** `length`, 138k characters (60,000 tokens), no tool call, rule not found. It was then dropped (request 2's prompt was 9,395).
2. **Request 2:** restarted. It found the congruent-twin rule in its first ~21k characters (L5191-5199), then spent ~38k more characters on test inputs by hand. 24,173 tokens.
3. **Calls 2-5** found that shapes are 8-connected and that `min()` over frozensets is nondeterministic (L7128); fixed (L7170).
4. **Call 6** (L7204): `transform` + `check` gave 2/2 exact (L7228). Cumulative about 97.1k.
5. **After the exact check:**
   - **Request 8** (663 tokens): `stop`, no tool call: "Let me verify the test-input groupings … before submitting".
   - **Nudge:** the ARC-2 nudge was sent (it says submit once every pair is exact).
   - **Call 7:** test-grouping/collision check instead of submit.
   - **Call 8** (4,245 tokens): re-checked the test shapes, said "I can submit with confidence… confirm and submit", then called `check(transform); print("final transform ready")`, not `submit` (L7674).
   - **Stop:** 102,547 tokens, `stop_requested` (L7688).

## b5ca7ac4_b0 (exact at call 5, cut before submitting; 100,222 tokens)

| Request | Tokens | What happened |
|---|---|---|
| 1 | 35,608 | 72.9k characters before the first code, 23 % of them copied grid rows. Hand-listed box coordinates were wrong ("I'm probably making mistakes", L940s) |
| 2 | 34,962 | In-head theorising about lane positions, then a fill-block detector that worked |
| 3 | 18,987 | Derived the correct conflict-graph 2-colouring by hand (L2876-2956), drafted the code in its thinking (L3036-3116), then called a data dump instead of `check` |
| 4 | 484 | A simplified greedy version: 0/3 |
| 5 | 10,181 | Re-derived the rule, wrote it twice, 3/3 exact (L3903). The total reached 100,222, so `stop_requested` came immediately and no submit was possible |

`check` and `submit` in the same call would have saved it.

## 4c3d4a41_b0 (failed; sibling b1 succeeded)

- **Requests 1-3** (23k + 35k + 20k tokens):
  - hand analysis mixing pair-1 and pair-2 columns;
  - dumps of data it already had;
  - a wrong model formed in its thinking ("digits shift up by 1 …"), then "Let me implement and test", followed by another dump.
- **Request 4:**
  - the prompt fell from 64,662 to 23,608 tokens: turns 1-2 evicted;
  - 28,404 tokens debating test 2 before running anything.
- **First and only `check`** at about 106k tokens: 0/2. Stopped.
- **Why b1 won:** b1 reasoned in digit counts per column, stacked from the bottom. It reached `check` 2/2 early and submitted at 83.4k.

## Harness findings

- **History eviction:** whole earlier turns are dropped, yet the resume says "Your tool results above are still valid". Each time the model restarted.
  - The reader's inference: Duck's chars/token calibration (1.40) comes from the digit-heavy grid prompt; prose thinking is about 2.26, so 126k characters are counted as ~90k tokens, above the 70,560 budget.
  - *Main-session check (`analysis/` + usage lists):* in EVAL6, 11 of 84 later requests did not carry the previous reply. Five were full 60k replies, about 487k tokens in all (40 % of all EVAL6 output). In EVAL30, 12 of 423 (98k tokens).
- **Cap overshoot:** the cap is checked between requests, and each request may use 60k (b0 reached 159k).

## Top causes, ranked by wasted tokens

1. **Long in-head hypothesis search before any code** (~300k tokens across the 4 branches).
2. **History eviction plus a misleading resume, forcing restarts from zero** (~180k tokens).
3. **Checking data or test inputs instead of running `check` / `submit` when a rule was in hand.**
4. **No auto-submit when `check` is exact, and no "submit now" warning near the cap.** Two correct functions were lost.
5. **Hand-transcription errors** (misread cells, box coordinates, mixed-up pair columns).
