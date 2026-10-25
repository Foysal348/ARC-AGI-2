# 13 — rev 3i: R + stage 3, EVAL120 in a 6-h notebook

**Date:** 2026-10-25.

**User request:** apply "R + attempt 3" to the last committed notebook (50K45K) and run it for exactly 6 hours on 120 tasks. The evidence is Notes/12, sections C-D.

## What changed

### Scheduler (`src/duck_b/arc2_duck.py`, `solve_all_r6`)

New parameters, off by default; with them off the behaviour is rev 3h:
- **`skip_b1_if_accepted` (R):** a task whose attempt 1 has a host-accepted candidate (stage-2 class ≥ 2) gets no attempt 2. Counted in `b1_skipped`.
- **`extra_attempts` (stage 3), with `max_attempts = 8`:**
  - A task with no host-accepted candidate after all its attempts, none of them running or queued, gets a fresh attempt with the next branch number.
  - That attempt uses the attempt-2 budget (45k) and the facts view.
  - Priority `(2, attempt number, ORDER2 index)`: round robin, every attempt 3 before any attempt 4.
  - Stage-3 items start only when every attempt 1 has finished. By then every attempt-2 item is queued, and their priority is higher, so they always start first.
  - Stage 3 runs until the solve deadline (min-admit 300 s; requests are deadline-clamped).
  - **Counters:** `extra_started`, `extra_by_attempt`, `extra_waits`. The flags are also recorded in `SCHED_LAST`.

### Builder `--r3i` (`build_e15b.py --budgets=50000,45000 --r3i`)

- Name `Q27-XHIGH-E15B_EVAL120_50K45K_S16_R3I_DUCK_SGLANG.ipynb`; default profile `E15BEVAL120R3I`.
- **Commit:** all 120 eval tasks, `time_budget_sec` = **6.0 h**, `hidden_equiv`.
  - The window is the hidden run's per-task share, 120 × (hidden solve end − start) / 240 ≈ 5.8 h, clamped by this run's hard stop − 180 s ≈ 5.7 h, i.e. ~8 min (~4 %) under the hidden share.
- **RERUN** (the submission) = all 240 hidden tasks with R + stage 3.
- The E15B block gains `E12_B_SKIP_B1_IF_ACCEPTED` 1, `E12_B_EXTRA_ATTEMPTS` 1 and `E12_B_MAX_ATTEMPTS` 8, and they are passed to `solve_all_r6`.
- The per-attempt budget lambda covers branch ≥ 1 (attempt 3+ = the attempt-2 budget).
- **Report additions:**
  - `b_summary`: `b1_skipped`, `extra_started`, `extra_by_attempt`, `extra_accepted`;
  - the eval block: `tasks_with_attempt3plus`, `attempt3plus_acc_tasks`, `hit_only_from_attempt3plus`.
- Without `--r3i`, the builder text output is as before. The Duck package is new, so a rebuilt rev 3h notebook would reference the new package.

### Build

`Q27-XHIGH-E15B_EVAL120_50K45K_S16_R3I_DUCK_SGLANG.ipynb`
- sha `2c8eb52e`;
- 247 kB;
- Duck package `q27x_e15b_duck_6b861740bb09.bin`.

## Verification

| Check | Result |
|---|---|
| Unit tests | **100 / 100** (93 + 7 for rev 3i) |
| `verify_items.py` | ALL CODE ITEMS PRESENT (item 3I) |
| Harness, eval120 mode | 35 / 35 (4 rev-3i checks) |
| Harness, rerun mode | 33 / 33 |
| Kaggle CPU kit v6 (R3I: eval120 + rerun) | running |

**What the rev-3i unit tests check:**
- R skips attempt 2 after an accept (6 / 6), with no stage 3 for those tasks.
- An unsolved task gets attempts 3 and 4 up to the maximum; a task that accepts at attempt 3 stops.
- No attempt 3 starts before the last attempt 1 has finished.
- Round robin holds.
- The counters are right.
- With the defaults, the behaviour equals rev 3h.

**Harness eval120 (fake server):**
- 118 / 120 tasks accepted at attempt 1, so their attempt 2 was skipped (`b1_skipped` 118).
- The 2 unsolved tasks got attempt 2, then attempts 3-8 (2 each), and the run stopped at max 8.
- Worker errors 0; heap empty at the end.

## Expected effect

Notes/12 D, a time-sliced replay at hidden scale with the 45K30K caps:
- R alone: −2 tasks;
- R + stage 3: +4 to +12 tasks of 240 vs now (attempt-3 rate 0.043-0.10 at a 30k cap).

At 50K45K the attempts are longer (45k for stage 3, the rate is higher) and the spare time is shorter. The EVAL120 6-h commit measures it.

**Not pushed.** It needs the user's go. The payload for Sami's account needs the new package `6b861740bb09`.
