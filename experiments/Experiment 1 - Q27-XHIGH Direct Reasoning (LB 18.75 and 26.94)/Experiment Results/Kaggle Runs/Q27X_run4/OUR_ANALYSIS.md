# Our analysis of run 4 (read after forming your own view)

## Evidence
The evidence comes from:
- `records.jsonl`;
- the thinking tails in `answers/`;
- the vLLM log in `run4_executed.ipynb`;
- a comparison of each model grid against copying the test input (identity).

## Weaknesses, most important first
1. **Rule search runs out of budget.** In ~12 of the 18 force-closed queries the model was still testing rule hypotheses at 49k tokens ("Maybe…", "Not.", "Another angle…"). It checks every hypothesis by hand against the examples. That is 87 % of all tokens for 0 hits.
2. **Rule found, but the output is computed by hand.** In ~4 force-closed queries the model was building the output grid when it was cut:
   - 2b83f449 counted cells one by one ("token1 c0=3, token2 c1=8 …") and ended 6/323 cells wrong;
   - 8b7bacbf#1 was re-indexing rows;
   - 6e4f6532#0 was transforming coordinates by hand;
   - 247ef758#0 wrote "final answer ready. But let's not rush…" and kept exploring.
3. **Transcription errors.** Two of the four natural finishes are 1 cell wrong; their rules were right. Two grids had ragged rows.
4. **pass@2 is unused.** attempt_2 is the test input, which has almost no value.
5. **Throughput and coverage:**
   - 12 % of the time is lost to re-prefill after a force-close (prefix caching off);
   - only 16 queries run at once although KV usage peaked at 74 %;
   - the queue runs in task-id order, so whichever queries come last are the ones cut in the hidden run.
6. **The projection is pessimistic.** The 22-query run has a long tail with few active queries. With 259 queries the slots stay full, giving about 373 × 0.88 ≈ 330 tok/s and 259 × 47k tokens ≈ 10.5 h. This is our estimate, not a measurement.
7. **The sample is small.** 2/22 has a wide uncertainty range (~1–29 %).

## Fixes we are considering
| priority | fix | addresses | cost |
|---|---|---|---|
| 1 | Python-tool harness: the model tests hypotheses in code on all training pairs and produces the output by running code | 1, 2, 3 | large |
| 2 | Shortest prompts first, 20 queries at once, prefix caching (after checking that vLLM 0.19 supports it for the hybrid model with MTP) | 5 | small |
| 3 | A second sample or vote for attempt_2; repair ragged rows | 3, 4 | small to medium |
| 4 | Submit v1.4 now to get a real LB number for comparison | 6, 7 | one submission |

## Open questions
- Why does NguyenThanhNhan's pure-reasoning setup score 23 when russcore's similar free-form setup scores 10 and our EVAL shows ~9 %? Possible reasons: his prompt, the answer format, 2× sampling, or the speed and coverage of the radix cache.
- Is xhigh the right effort, or would medium or low with a tool loop give more correct answers per token?
