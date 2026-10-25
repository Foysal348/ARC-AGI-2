# Reader (xhigh, EVAL6R5): where the pre-check() tokens go — 2026-10-21

Source: the subagent report, condensed. Lines refer to each branch's `transcript.txt`. Spot-checked: L289/L1247 (62593bfd_b0), L1241 (4c3d4a41_b0), L2511 (135a2760_b1) all match.

**Label correction:** 62593bfd_b0 checked 2/2 exact at about 58k and accepted (L1903-1942). The unsolved one is 135a2760_b1 (first check about 54k, L948, failed). Failing checks are not in `checked`.

## Slow branches

**62593bfd_b0** (first check 58k)
- **Rule:** correct (shared cell count → top, unique → bottom) at L1247-1250, about 37k.
- **Miscounts:** response 1 (29k) was built on hand miscounts (shape 4 counted as 6, really 5; shape 1 as 6, really 7). That gave a wrong "size-chunk" rule at about 9k.
- **Delayed code:** "recompute with code rather than guessing" at L289 (about 3k); the same intent recurs many times before acting.
- **Tool calls:**
  - call 1 (L723) refuted the chunk rule;
  - call 2 re-checked moves it already knew;
  - call 3 (L1820) showed the test hand count was wrong ("My manual count was wrong"), and the next call was transform + check.
- **Waste:** about 3k re-reading rows, about 3k hand-counting test shapes, about 18k on a false test "collision" built on a miscount (L1255-1517), and about 2k drafting the transform inside thinking.

**4c3d4a41_b0** (first check 63.8k)
- **Rule:** at L1612-1632, about 55k.
- **Response 1:** hit 49,152 with no tool call; about 37 % of its text is row / coordinate reading.
- **Misreads:** L377 (staircase 16 vs 14 cells) and L945 ("Not 4! I made an error").
- **Response 2:** L1241 "stop hand-simulating and actually use Python". 3 small count calls (about 1.1k) found the 16 vs 14 difference, and the rule followed in about 5.5k. Then a geometry call and a frame-diff call came instead of transform.
- **First check:** 0/2, fixed in one round.
- **Waste:** about 45k circling hypotheses on misread data; about 5.7k hand-simulating test outputs before any check.

**135a2760_b1** (first check 54k, never solved)
- **Rule idea:** "repair each box to its periodic tile" at L153, about 2k.
- **Hand transcription:** of the 29x29 test grid (L182-273, about 6k). It was wrong: L2511 "The data is completely different from my visual reading".
- **Hand tables:** about 25k of hand mismatch tables and tie-break debate. The transform drafted in thinking was cut by the limit.
- **After the first check:**
  - the check gave an IndexError;
  - 6 debug calls followed, two of them on disallowed `traceback` / `sys` imports;
  - the T=8 rule from the hand tables was refuted by one check (L1878);
  - about 45k more went on re-deriving selection rules.

**898e7135_b0** (check only at about 107k, crashed)
- **No rule found.** L1037 says "let me just do it in code. Let me stop guessing", then about 25k more characters came before the call.
- **15 analysis calls,** 6 of which failed (sys import, unpack, IndexError, syntax).
- **No partial transform was ever checked.** Reasoning blocks ran 23-66k characters between calls.

## Fast branches

- **136b0064_b1:**
  - the rule was complete at L1020-1034 (about 24k);
  - about 6k went on test simulation, then 1 fact call, then transform + check (L1464);
  - check's KeyError exposed a misread shape, which was fixed from code-extracted shapes. Exact at 32.5k.
- **136b0064_b0:**
  - the core rule came at about 21k, and about 6.5k went on an offset question;
  - transform + check anyway, then 3 fix rounds. Exact at 35k.
- **Difference:** small grids; the first real call is transform + check; check settles open details.

## Proposals (by the reader)

1. **Facts block** computed by the harness (dims, background, colour counts, components with count / bbox / shape, per-pair diff count) + "use only numbers printed by Python or the facts block".
2. **Code-intent stop:** after 3k tokens, a code-intent phrase ends the stream, and the resume says "Your reasoning is kept. Emit the python call you just planned". Alternative: cap only the first request at about 12k. Needs approval (60k floor; the earlier pause-resume result was negative).
3. **Replace "compare their test outputs":** "When one rule explains every training pair, call check(transform) in that same reply; compare test outputs only after check is exact, in Python".
4. **"If two or more variants fit, write each as a function and check them all in one call"** + `check({'A': fA, 'B': fB})`. check errors show the failing line inside transform.
