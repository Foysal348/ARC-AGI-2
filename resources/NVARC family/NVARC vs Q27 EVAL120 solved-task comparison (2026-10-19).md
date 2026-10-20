# NVARC (`arc-2026d-with-codex`) vs our Qwen3.8-27B full EVAL120 runs (2026-10-19)

All three runs cover all 120 public-eval tasks / 172 test inputs (units). Scored against `arc-agi_evaluation_solutions.json`,
pass@2 per unit, task-normalised. Scripts and unit lists: `_comparison/` (`compare.py`, `slot.py`, `units.json`).

| run | units | task-norm |
|---|---|---|
| NVARC `arc-2026d-with-codex` (4.75 h, all 120 trained) | 45 / 172 | 33.50 = **27.92 %** |
| Q27 E10 ORDER2 86k DIRECT (`hosen42/q27-xhigh-e10-order2`) | 58 / 172 | 44.33 = **36.94 %** |
| Q27 E10 Poetiq R5 (`hosen42/q27-xhigh-e10-poetiq-r5-eval120`) | 78 / 172 | 58.33 = **48.61 %** |
| Q27 union (DIRECT or Poetiq) | 92 / 172 | 68.17 = 56.81 % |

## Overlap (units)

| pair | both | only NVARC | only Q27 | union task-norm |
|---|---|---|---|---|
| NVARC vs Poetiq R5 | 32 | 13 (8.83) | 46 (33.67) | 55.97 % |
| NVARC vs DIRECT | 23 | 22 (15.33) | 35 (26.17) | 49.72 % |
| NVARC vs Q27 union | 36 | 9 (6.00) | 56 (40.67) | **61.81 %** |

NVARC is mostly a subset of what Qwen can solve: 36 of its 45 units are also solved by some Q27 run.
Tasks only NVARC solved (vs the Q27 union, 7 tasks): `2d0172a1`, `3a25b0d8`, `71e489b6`, `78332cb0`, `7b80bb43`, `de809cff`, `eee78d87`
(mostly tier 1-2, `rare`, `multi_query`; ARC Prize xhigh solved only `2d0172a1`).

## Realistic two-slot ensemble (one submission, 2 attempts per unit)

- Poetiq R5 `attempt_2` adds only 1 unit over `attempt_1` (77 -> 78): the second slot is almost free.
- NVARC `attempt_1` alone = 36 units; its `attempt_2` adds 9.
- **Poetiq attempt_1 + NVARC attempt_1 as attempt_2 = 89 units, task-norm 65.67 = 54.72 %** (vs Poetiq alone 48.61 %: **+6.1 points**).

## Caveat

These are eval runs without the hidden-run coverage limit. On the LB, Poetiq R5 scored 21.39 (LB/eval 0.44, coverage binds) and NVARC
about 29 (LB/eval about 1.0: it finishes). So on hidden, NVARC's value is larger than +6: it delivers its solves on tasks Qwen does not reach
in 12 h. Any ensemble needs both to fit in one 12 h run (NVARC about 4.75 h per 120 tasks on 4xL4) - user decision, not started.
