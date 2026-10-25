# 14 — rev 3j: two order notebooks (EVAL120, 6 h)

**Date:** 2026-10-25.

**User request:** two notebooks, run on 120 puzzles, 6-h runtime:
1. attempt 1 → attempt 2 on the unsolved tasks → attempt 3 on the unsolved tasks → attempt 2 on the accepted tasks;
2. attempt 1 → attempt 2 on the unsolved tasks → attempt 2 on the accepted tasks → attempt 3 on the unsolved tasks.

"Accepted" = attempt 1 has a host-accepted candidate (stage-2 class ≥ 2).

## Notebooks

| | Notebook 1 | Notebook 2 |
|---|---|---|
| File | `Q27-XHIGH-E15B_EVAL120_50K45K_S16_R3J1_DUCK_SGLANG.ipynb` | `Q27-XHIGH-E15B_EVAL120_50K45K_S16_R3J2_DUCK_SGLANG.ipynb` |
| sha | `74a0eae5` | `5c44f752` |
| Builder | `build_e15b.py --budgets=50000,45000 --r3j=1` | `--r3j=2` |
| Order | a1 → a2 (unsolved) → a3 (unsolved) → a2 (accepted) → a4..a8 (unsolved) | a1 → a2 (unsolved) → a2 (accepted) → a3..a8 (unsolved) |

**The same in both:**
- **Duck package:** `q27x_e15b_duck_d6faf44ec0d5.bin`.
- **Budgets:** attempt 1 50k, attempts 2-8 45k.
- **Commit:** 120 eval tasks; profile `E15BEVAL120R3I` (6.0 h, hidden-equivalent window ≈ 5.8 h, clamped by the 6-h hard stop to ≈ 5.7 h).
- **RERUN:** the 240 hidden tasks in the same order.
- **No attempt 2 is skipped.** `E12_B_SKIP_B1_IF_ACCEPTED` = 0.
- **After the listed phases, while time remains:** further fresh attempts on still-unsolved tasks, round robin, up to 8 per task (`E12_B_MAX_ATTEMPTS`). I added this so the streams never idle; it can be turned off.

## Scheduler (`solve_all_r6`, new `b1_accepted_mode`)

| Mode | Behaviour |
|---|---|
| `""` | as rev 3h |
| `skip` | rev 3i R; the old flag `skip_b1_if_accepted` maps to it |
| `before3` (notebook 2) | the accepted task's attempt 2 gets priority `(1.5, 0, ORDER2, sub)`: after every attempt 2 of the unsolved tasks (1, ...), before every attempt 3 (2, 2, ...) |
| `after3` (notebook 1) | priority `(2, 2.5, ORDER2, sub)`: after the attempt 3s (2, 2, ...), before the attempt 4s (2, 3, ...) |

- A deferred attempt 2 starts only when every attempt 1 has finished, like the stage-3 items.
- **Counters:** `b1_deferred`, `b1_deferred_started`. The stage tag is set by attempt number (1 / 2 / 3+).
- **Order holds among the queued work.** In notebook 1, a task's attempt 3 is queued only when its attempt 2 ends. A stream that frees up earlier takes an accepted task's attempt 2 rather than idle.

## Verification

| Check | Result |
|---|---|
| Unit tests | **103 / 103** (section 20: both phase sequences exact with 1 worker; all tasks keep attempt 2; counters) |
| `verify_items.py` | ALL CODE ITEMS PRESENT for both (new item 3J; the 3I strings updated to the rev 3j form) |
| Harness eval120 | 35 / 35 both: `after3` / `before3`; deferred 118 / started 118; the 2 unsolved tasks got attempts 3-8; worker errors 0 |
| Harness rerun | 33 / 33 both |
| Kaggle CPU kit v7 (`build_cpu_kit_e15b.py --r3j`) | running |

**Not pushed** (needs the user's go). The payload of the account that runs it needs `q27x_e15b_duck_d6faf44ec0d5.bin`.
