# "Qwen3.8 or something else?" (Anil Thomas, about 2026-10-19) — pasted by the user 2026-10-21

**User:** "They all use Qwen3.8 + optimized Duck." Taken as fact.

## Anil Thomas (40th), first post
- "the sudden flood of 55+ scores makes me think that there is an easy recipe out there."
- He went from 35 to 45 by ensembling his NVARC-based solution with a Qwen3.8-27B run.
- "could it be a better harness? a harness that includes a python REPL can help Qwen3.8-27B quite a bit. but squeezing it into a 12 hour window on 4x L4 is hard."

## Kris Dreemurr (90th)
- His Qwen harness at xhigh, at full policy, uses a **median 70k tokens for pass@2 (35k per slot)** and gets **about 45 %, i.e. about 52 of 120**, averaged across runs.
- "The issue is I cant fit it in the box."
- He runs NVARC on all 240 tasks in about 4 h.

## Darren Amadeus Martin (9th)
- "getting 52/120 from 70k tokens per task is impressive."

## mikelou1 (13th)
- "With some tricks and stuff I currently get around **79 % ± 5 % in CV** runs."
- Later: "With time constraints at around 4xL4 speed I got Qwen to **87 %** (and from reasoning traces it got the rule to almost every problem)."
- "The 87 % will likely not transfer … I do hope this run gets me to 75 %-ish."
- "Not too worried about overfit."

## Adarsh (271st)
- Qwen3.8-27B at xhigh "is just so slow that it never reaches most of the tasks".
- Flash-Next on L4 (Triton) "hasn't worked out that well".
- "an inference-optimized qwen 3.8 27b with a proper iterative feedback harness could take you quite far … python repl should do almost all of the thinking for qwen and reduce the token usage".

## TheDivine Aurora (1707th)
- "at around 300-500 tps i had ~19-20 % score. only attempted half the dataset. so practically 40 % is possible with 600-800 tps, 1 attempt per task."
- "it just takes 128k - 256k context window."
- "ion think anyone is ensembling."

## Takeaways for E12

- **Per-attempt budget:** about **35k tokens per attempt** (Kris: 45 % CV at 70k per task for pass@2). Our rev 4 Duck used about 100k per branch.
- **Ceiling:** mikelou1's 87 % at 4xL4 speed shows Qwen3.8 has the ability. The work is token efficiency and throughput inside the harness.
