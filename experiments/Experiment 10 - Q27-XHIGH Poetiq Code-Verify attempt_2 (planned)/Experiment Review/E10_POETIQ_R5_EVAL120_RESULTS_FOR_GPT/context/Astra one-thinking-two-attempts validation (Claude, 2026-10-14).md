# Astra's report on "one thinking, two attempts": validation (Claude, 2026-10-14)

**Method:**
- Astra's `audit_e10.py` was read in full. It only parses files and never executes generated programs.
- It was then run on our bundle `E10_ONE_THINKING_TWO_ATTEMPTS_FOR_ASTRA/`.
- Copy of the script and output: session scratchpad `astra/` (`audit_summary.json`, `trace_candidates.json`).

## Claims checked

| Claim | Our run of the script / our data | Verdict |
|---|---|---|
| Poetiq 4/22, DIRECT 4/22, union 5/22. DIRECT only: `b6f77b65#1`; Poetiq only: `4c3d4a41#1` | identical | ✅ |
| 13 distinct Poetiq second guesses gave 0 gain | 13 inputs had attempt_2 ≠ attempt_1, and none were correct | ✅ |
| In round 3, 6 of 7 thinking requests ended below 49,152 because of the drain | 35.3k, 41.8k, 42.6k, 33.0k, 28.1k and 35.2k were below; 59.4k was above | ✅ |
| Grid answer tokens per task: median 972, p90 2,648, max 3,608; 2 of 24 code + grid overflows | identical to our measurement | ✅ |
| Two 49k paths = 98,304 tokens, +26.3 % vs one 77.8k; with answers +33.3 % | arithmetic correct | ✅ |
| **41 DIRECT full traces: the final answer grid appears explicitly in 14/41; the true grid in 7/41; 0 "rescues"** (no trace contains a correct grid that the final answer missed) | `baseline_grid_found` 14, `truth_grid_found` 7, `oracle_rescues` [] | ✅ the parser is conservative, but these are real counts |
| **DIRECT writes its answer grid late:** median at 89 % of the trace; 0 traces by 50 %, 1 by 75 %, 8 by 90 % | identical | ✅ the grid is produced at the end of the reasoning |
| Template: `enable_thinking = False` also drops the xhigh system instruction (lines 46–54), so re-rendering changes the prefix and the cache is lost | confirmed in `chat_template.jinja` | ✅ |
| Research figures: BARC ensemble 56.75 % on ARC-1, Imbue numbers, ARChitects, NVARC, vLLM APC | consistent with sources we already checked; BARC and ARChitects not re-fetched | plausible |
| "The Qwen card documents thinking-mode temperature 1.0" | not verified by us; our DIRECT runs use 0.6 | open |
| "Could not verify 60+ with Qwen3.8-27B" | the user has their own evidence | unchanged |

## What it means

1. **One trace yields two good answers only if the trace already contains the second answer.**
   - DIRECT computes its grid only at the very end, at about 89 % of the trace.
   - A code-focused Poetiq trace probably never writes the test grid, so a grid branch would be written without reasoning.
   - This confirms the earlier doubt. The 5/22 union came from **two independent traces**.
2. **There is nothing to harvest from existing traces:** they contain no correct grid that the final answer missed.
3. **Astra's rank 1 is a separate answer branch after the same reasoning, with the original answer kept.**
   - Astra suggests doing it first on the **DIRECT side: DIRECT primary + a code sibling.**
   - This is the more promising direction. DIRECT's trace holds the rule and usually the grid, so a short code branch can turn the rule into an executable program, check it on the training pairs, and fix transcription or shape errors.
   - E9 had 4 shape errors that are of this kind.
   - Cost: about 2.5k code tokens plus a prefix-cached branch. Accuracy gain unmeasured.
4. **Rank 2 (a fresh, alternate-modality reasoning on selected tasks) is the honest route to independent attempts,** at a cost of at least 49k thinking each.
