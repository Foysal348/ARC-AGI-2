# EVO24 — dual mode (author change after round 3, at the team's request)

Like the public top notebooks and the team's own NVARC notebooks, EVO24 now branches on `KAGGLE_IS_COMPETITION_RERUN`.
Notebook sha256 `ed7fd1abdb1857589cfc415eceb7a62e477b711d3c11b7ddd73323a457b11736`.

| | interactive / commit run (unset) | scoring rerun after Submit (set) |
|---|---|---|
| tasks | the 24 frozen evaluation tasks (answers known) | every task in `arc-agi_test_challenges.json` (no answers) |
| how `eval_challenges` is used | public evaluation data | rebound to the hidden test tasks at the top of cell 4 (`EVAL_CHALLENGES_PUBLIC` keeps the public set); `eval_solutions = {}`; the EVAL120 answer check is skipped |
| cell 8 | unchanged evolution; rounds now run in chunks of 128 calls with a deadline projection before each chunk (measured throughput ×1.25; before any measurement: 5,120 tokens per call at 60 tok/s) | same; the chunking is what keeps a 120–240-task hidden set inside 12 h |
| submission.json | written for the 24 eval tasks | written for **every** hidden task |
| fallback | a valid `submission.json` (test inputs as answers) is written before any generation and refreshed after every chunk; the final write in cell 9 never removes the last valid file | same |
| cell 9 | strict per-query pass@2 scoring, NVARC complementarity, verdict | no scoring (no answers): summary of tasks run / attempted, rounds, failures |

Grid validity (`_evo_valid_grid`: rectangular, 1–30 × 1–30, ints 0–9, no bools) is checked for every attempt; an
invalid attempt is replaced by the fallback.

Harness (`harness_logs/h5_*.txt`), all with bundle and a valid submission.json:

| run | mode | result |
|---|---|---|
| eval | evaluation | PROMISING (scripted), submission 24 tasks |
| rerun | `KAGGLE_IS_COMPETITION_RERUN=1`, placeholder test set | 240 tasks / 259 queries, 15 chunks over 3 rounds, submission 240 tasks |
| deadline | evaluation | INCOMPLETE / NO DECISION (stop before round 0 chunk 1), fallback submission |
| infra | evaluation | INVALID / VERIFIER INFRASTRUCTURE FAILURE |
| p1 / p2 / short | evaluation | INVALID / ENGINE FAILURE |

Expected cost: the commit run (evaluation) ~1.3–2.5 h wall (~2.5–5 quota-h, buffer 8). The scoring rerun runs on
Kaggle's side; on a 120-task hidden set the evolution needs roughly 5× the evaluation run and the chunk guard stops
before the 12 h limit, leaving fallback answers for unfinished tasks.
