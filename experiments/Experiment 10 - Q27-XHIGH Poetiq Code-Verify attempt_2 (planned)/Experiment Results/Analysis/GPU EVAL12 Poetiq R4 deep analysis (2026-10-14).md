# E10 Poetiq R4: GPU EVAL12 deep analysis (Claude, 2026-10-14)

**Run:** `foysalemonshanto/q27-xhigh-e10-poetiq-r4` v1 (scriptVersionId 358208943).
- Notebook R4, sha `596bf694…`. EVAL12: 12 tasks / 22 inputs, 3 h. 1 expert, up to 10 iterations, 77,824 thinking + 8,192 answer, temperature 1.0.
- Outputs are in `../Kaggle Runs/GPU EVAL12 Poetiq R4 (foysal q27-xhigh-e10-poetiq-r4 v1)/`.

## Result: 4/22 (18.18 %), the same score as E9 DIRECT 86k, with a different set of hits

| Input | R4 Poetiq | E9 DIRECT 86k | All our earlier runs |
|---|---|---|---|
| `136b0064#0` | ✅ | ✅ | 8/10 |
| `247ef758#0` | ✅ | ✅ | 5/9 |
| `4c3d4a41#0` | ✅ (iteration 2) | ✅ | 3/9 |
| **`4c3d4a41#1`** | **✅ new** | ❌ (2 cells) | **0/9** |
| `b6f77b65#1` | ❌ (36 cells) | ✅ | 1/9 |

- **Union of E9 and R4: 5/22.** Taking attempt_1 from DIRECT and attempt_2 from Poetiq on the same samples would have scored 5/22 here.
- **attempt_2 added nothing** (pass@2 hits = pass@1 hits = 4). With 1 expert, most tasks have only one program, so attempt_2 duplicates attempt_1.

## Engine and runtime: clean

- No errors, 0 preemptions, KV max 79 %.
- MTP acceptance 0.56. Prefix-cache hits 79 % (the forced-close continuations).
- Sandbox self-test PASS on the GPU container; 27 programs verified, 0 sandbox failures.
- Throughput 224 tok/s over the solver wall time of 8,249 s. EVAL12 used at most **12 of 16 slots**, and fewer as tasks finished.
- **Tokens:** 1.85 M generated for 27 iterations, about 68k per iteration. E9 used 1.47 M (primary) + 1.03 M (resample).

## Iterations

| Round | Natural closes | Forced at 77,824 | Train-verified |
|---|---|---|---|
| 1 (12 tasks) | 2 (`136b0064` 40k, `247ef758` 71k) | **10** | 4: `136b0064`, `247ef758`, `898e7135`, `2b83f449` |
| 2 (8 tasks) | 1 (`4c3d4a41`, 67k) | 7 | 1: `4c3d4a41` (0.906 → 1.0) |
| 3 (7 tasks, cut at the 147 min drain after 28–59k thinking) | 0 | 7 (drain) | 1: `6e4f6532` (0.92 → 0.95 → 1.0) |

**The Poetiq prompt makes Qwen think longer than DIRECT.** For example, `4c3d4a41` closed naturally at 25–32k under DIRECT but was forced at 77.8k in round 1.

**Feedback works on the train score:**
- `4c3d4a41` 0.906 → 1.0 (both test inputs then correct);
- `6e4f6532` 0.92 → 0.95 → 1.0;
- `8b7bacbf` 0.90 → 0.992 → 0.995;
- `88bcf3b4` 0 → 0.9993;
- `b6f77b65` 0.58 → 0.67;
- `20a9e565` 0 → 0.22 → 0.33;
- `21897d95` 0.05 → 0 → 0.34.

## The main weakness: train-verified programs are often wrong on test

| Task | Train examples | Train | Test |
|---|---|---|---|
| `136b0064` | 3 | 1.0 | ✅ |
| `4c3d4a41` | **2** | 1.0 | ✅ ✅ |
| `247ef758` | 3 | 1.0 | `#0` ✅ / **`#1` 1 cell wrong**: (6,12) 4→6, the **same cell as DIRECT E9**. Train does not settle the overlap order |
| `2b83f449` | **2** | 1.0 | ❌ 4/323 wrong: 8 written where 3 belongs, the same marker error as E9's resample |
| `898e7135` | **2** | 1.0 | ❌ 18/900 wrong: one hole filled with colour 2 instead of 3 |
| `6e4f6532` | **2** | 1.0 | ❌ `#0` 98/840, `#1` 14/480 |

- Of the 6 train-verified tasks, **only 2 are fully right on test**, and 1 is half right.
- Of the 4 two-example tasks, 3 failed on test. With 2 examples the rule is under-determined, so the program can fit both and still be wrong. This matches the capability profile (`low_train`) and R6 (3 of 4 verified programs were wrong).
- **Test near misses (within 20 cells):** `247ef758#1` (1), `2b83f449` (4), `88bcf3b4` (12 and 11), `6e4f6532#1` (14), `8b7bacbf#1` (16), `898e7135` (18). That is 7 inputs, all from programs with train ≥ 0.99.

## What this means

1. **Poetiq alone does not beat DIRECT on EVAL12** (4 = 4), but it solves *different* inputs. It is the first method to solve `4c3d4a41#1`, which shows that one shared program can recover a failed sibling.
2. **Most of the time is spent on round 1:** 10 of 12 tasks were forced at 77.8k, and round 3 was cut by the drain. More iterations clearly raise the train score.
3. **The weakest point is train verification as the selector,** especially for tasks with 2 training examples.

## Candidate next steps (user decision; one at a time)

- **A. Hybrid:** attempt_1 = DIRECT, attempt_2 = Poetiq (the original E10 plan). On this sample the union is 5/22. It costs about twice the compute.
- **B. Thinking cap of 49,152 from iteration 2** (within the user's floor), so tasks get about 3–4 iterations instead of 2–3.
- **C. Use idle slots for a 2nd expert** on EVAL-sized sets. This gives diverse programs, so attempt_2 stops being a duplicate and votes become meaningful (Poetiq's 2-expert config).
- **Hidden-run caution:** iteration 1 alone costs about as much as a DIRECT run at 86k, so few feedback iterations would fit in 12 h for 240 tasks.

**Ledger:** added as the run "E10 Poetiq R4 EVAL12" (`build_ledger.py`, kind `poetiq_report`).
