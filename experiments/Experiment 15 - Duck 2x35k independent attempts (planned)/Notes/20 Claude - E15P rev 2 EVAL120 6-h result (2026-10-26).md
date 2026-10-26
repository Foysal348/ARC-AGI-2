# E15P rev 2 EVAL120 6-h result (Sami, esfersami50/q27-xhigh-e15p-eval120-s60-s15-poetiq-sglang)

Notebook sha 56a22f7f. Status COMPLETE. Outputs are in `_scratch/e15b/e15p_pull/` (poetiq_run_report.json, e14_report.md, log).

## Score

- 64 / 172 queries (37.21 %), all pass1 (one expert, so pass2 is 0).
- Task-normalized: 47.17 / 120 = 39.31 %. Fully solved tasks: 43.
- Same window on Duck config B: R3K2 65/172, task-norm 39.03 %; R3K1 50/172, 33.47 %. Poetiq S60 on the E15B engine is at the best Duck run's level.

## Coverage and engine

- Iteration 1 started and completed for all 120 tasks; nothing was left unstarted.
- 99 / 120 iteration-1 requests were force-closed at the 60k thinking cap; 21 finished naturally.
- Only 18 tasks reached iteration 2 (138 iterations in total). Phase-2 tokens: 0.22 M of 7.83 M.
- Engine: 389 tok/s over 20,127 s of solver wall time, at 15 running for 3,743 of 3,790 decode-log samples.
  - Peaks: token_usage 0.83, mamba 0.72.
  - Queue 0, 0 retractions, 0 KV-gate blocks, 0 restarts or relaunches; accept length median 3.4.

## Train verification precision

- 52 tasks train-verified: 42 have a hit (81 %), 39 have every query right.
- 68 not verified (the best partial program is submitted): 10 have a hit, 4 fully.

## Reading

Iteration 1 uses the whole 6 h, because 83 % of the first answers hit the 60k cap. The Poetiq feedback loop (iterations 2-10) almost never runs in a 6-h window at 389 tok/s. That loop is the part that gave Poetiq its 88 % oracle, so engine speed is the lever.
