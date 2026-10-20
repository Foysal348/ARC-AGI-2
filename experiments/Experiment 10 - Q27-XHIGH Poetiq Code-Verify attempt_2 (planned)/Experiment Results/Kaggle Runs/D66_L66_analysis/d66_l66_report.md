# D66 / L66 deep-run analysis

Hard tasks: 66 with 103 inputs. R5 had 9 of them right and 94 wrong.
15 zero-score tasks were train-verified in R5 (train pass, test wrong).

All rates are conditioned on R5's failure, so they are NOT ARC-wide rates. A D66/L66 'solve' counts as such only for this configuration (S1 fixes + fresh seeds), not as a pure seed effect.

## D66

- Requests 151; tokens 10.80 M; slot-hours 187.6; finish {'stop': 34, 'length': 102, 'drained': 15}
- Iterations per task: {1: 14, 3: 33, 2: 19}
- Deployable two-attempt result at the end: 28/103 inputs, task-normalised 18.83/66 tasks
- Oracle (any program output correct, any iteration): 28/103 inputs; of the 94 R5-wrong inputs: 22

| Iteration | Requests | Slot-hours | Mean thinking | Forced at cap | Train pass | New correct inputs (oracle) | New per slot-hour | Of R5-wrong |
|---|---|---|---|---|---|---|---|---|
| 1 | 66 | 83.9 | 73,020 | 49 | 14 | 12 | 0.14 | 9 |
| 2 | 52 | 66.7 | 73,497 | 39 | 11 | 15 | 0.22 | 12 |
| 3 | 33 | 36.9 | 56,845 | 14 | 8 | 1 | 0.03 | 1 |

Deployable score against run time (attempt curve):

| Hour | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Inputs right | 0 | 3 | 5 | 8 | 11 | 12 | 15 | 20 | 25 | 27 | 28 | 28 |

R5 train-pass/test-wrong tasks (15): {'train fail + test right': 1, 'train pass + test right': 5, 'train pass + test wrong (overfit again)': 6, 'train fail + test wrong': 3}

## L66

- Requests 123; tokens 10.66 M; slot-hours 139.6; finish {'stop': 100, 'length': 11, 'drained': 12}
- Iterations per task: {1: 25, 2: 25, 3: 16}
- Deployable two-attempt result at the end: 34/103 inputs, task-normalised 21.50/66 tasks
- Oracle (any program output correct, any iteration): 34/103 inputs; of the 94 R5-wrong inputs: 29

| Iteration | Requests | Slot-hours | Mean thinking | Forced at cap | Train pass | New correct inputs (oracle) | New per slot-hour | Of R5-wrong |
|---|---|---|---|---|---|---|---|---|
| 1 | 66 | 80.9 | 93,952 | 9 | 25 | 22 | 0.27 | 18 |
| 2 | 41 | 48.0 | 88,577 | 2 | 18 | 12 | 0.25 | 11 |
| 3 | 16 | 10.7 | 47,425 | 0 | 0 | 0 | 0.00 | 0 |

Deployable score against run time (attempt curve):

| Hour | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Inputs right | 2 | 5 | 7 | 12 | 17 | 20 | 22 | 22 | 26 | 27 | 30 | 34 |

R5 train-pass/test-wrong tasks (15): {'train fail + test right': 1, 'train pass + test right': 4, 'train pass + test wrong (overfit again)': 9, 'train fail + test wrong': 1}

## D66 (77,824) vs L66 (122,880): paired on the same tasks

Iteration 1, tasks reached by both: 66. Fully correct: D66 7, L66 11. Only D66 5, only L66 9, both 2.
- D66 iteration 1: mean thinking 73,020; forced at cap 49/66; mean slot-minutes 76.3
- L66 iteration 1: mean thinking 93,952; forced at cap 9/66; mean slot-minutes 73.6
- L66 iteration-1 natural closes between 77,824 and 122,880 tokens (would be cut at D66's cap): 39; of them fully correct: 3; D66 got the same task right at iteration 1: 5
- D66 whole run: 28/103 deployable, 28 oracle, 187.6 slot-hours, 10.8 M tokens -> 2.59 deployable inputs per M tokens
- L66 whole run: 34/103 deployable, 34 oracle, 139.6 slot-hours, 10.66 M tokens -> 3.19 deployable inputs per M tokens
