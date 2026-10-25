# R3K1 / R3K2 EVAL120 6-h runs (2026-10-25)

## R3K1 (after3, Sami)

- branch results read: 200 of 200 done (COMPLETE)
- score: 50/172 queries, task-norm 33.47 %, pass1 49, pass2 1
- scheduler: deferred attempt 2 41 (started 0), attempts >= 3 started 1, weak accepts 28, weak-only tasks continued 8, queue left 114

| attempt | n | tokens mean | min mean | any accept | strong accept | strong correct | weak-only | weak correct | any candidate correct |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 120 | 42.5k | 0.0 | 49/120 (41 %) | 41/120 (34 %) | 27/41 (66 %) | 8/120 (7 %) | 4/8 (50 %) | 37/120 (31 %) |
| 2 | 79 | 40.5k | 0.0 | 10/79 (13 %) | 6/79 (8 %) | 5/6 (83 %) | 4/79 (5 %) | 1/4 (25 %) | 8/79 (10 %) |
| 3 | 1 | 0.1k | 0.0 | 0/1 (0 %) | 0/1 (0 %) | 0/0 | 0/1 (0 %) | 0/0 | 0/1 (0 %) |

- tasks with a hit (any query): 44; tasks where some candidate of some attempt was right: 44; right somewhere but no hit: []
- attempt 2 right where attempt 1 was not: 7 ['409aa875', '981571dc', 'b10624e5', 'b6f77b65', 'b99e7126', 'c4d067a0', 'dd6b8c4b']
- attempt 1 weak-only (W1 continued): 8; its weak answer right: 4; attempt 2 ran: 8; attempt 2 strong-correct: 0
- generated tokens (read attempts): 8.30 M; mean per attempt 41.5k; attempts at >= 98 % of budget: 128 of 200
- usage_all: completion 8297791, prompt 54215140, cached 92.8 %, requests 2167; finished at 21301 s

## R3K2 (before3, Hamid)

- branch results read: 198 of 198 done (COMPLETE)
- score: 65/172 queries, task-norm 39.03 %, pass1 64, pass2 1
- scheduler: deferred attempt 2 37 (started 0), attempts >= 3 started 0, weak accepts 43, weak-only tasks continued 16, queue left 108

| attempt | n | tokens mean | min mean | any accept | strong accept | strong correct | weak-only | weak correct | any candidate correct |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 120 | 42.5k | 0.0 | 53/120 (44 %) | 37/120 (31 %) | 28/37 (76 %) | 16/120 (13 %) | 4/16 (25 %) | 35/120 (29 %) |
| 2 | 78 | 40.5k | 0.0 | 18/78 (23 %) | 12/78 (15 %) | 10/12 (83 %) | 6/78 (8 %) | 1/6 (17 %) | 17/78 (22 %) |

- tasks with a hit (any query): 49; tasks where some candidate of some attempt was right: 49; right somewhere but no hit: []
- attempt 2 right where attempt 1 was not: 14 ['135a2760', '136b0064', '142ca369', '36a08778', '581f7754', '7b3084d4', '800d221b', '89565ca0', '8b9c3697', 'a251c730', 'b6f77b65', 'cb2d8a2c', 'db0c5428', 'f931b4a8']
- attempt 1 weak-only (W1 continued): 16; its weak answer right: 4; attempt 2 ran: 16; attempt 2 strong-correct: 3
- generated tokens (read attempts): 8.26 M; mean per attempt 41.7k; attempts at >= 98 % of budget: 125 of 198
- usage_all: completion 8259733, prompt 59424138, cached 93.4 %, requests 2303; finished at 21301 s

## Overlap

- queries hit: R3K1 (after3, Sami) 50, R3K2 (before3, Hamid) 65, both 35, union 80, only R3K1 15, only R3K2 30
