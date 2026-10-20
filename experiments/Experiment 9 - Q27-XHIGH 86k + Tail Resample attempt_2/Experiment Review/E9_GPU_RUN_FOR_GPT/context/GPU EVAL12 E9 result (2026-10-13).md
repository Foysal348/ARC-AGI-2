# E9 GPU run (EVAL12): first result (Claude, 2026-10-13)

**Run:** `esfersami50/q27-xhigh-e9` v1, Save & Run All, EVAL12 (12 tasks / 22 inputs), 3 h. Notebook sha `845ff050`. Outputs are in `../Kaggle Runs/GPU EVAL12 E9 (esfer q27-xhigh-e9 v1)/`.

## Score

| | Value |
|---|---|
| **Eval pass@2** | **4 / 22 (18.18 %)**, task-normalised 19.44 %: the best of all our EVAL12 runs (earlier at 49k: run4 2, run6 2, run7 3, dual 1) |
| Primary-only | 4 / 22; **resample gain 0** |
| Hits | `136b0064#0`, `247ef758#0`, `4c3d4a41#0`, **`b6f77b65#1` (new: never solved before by us or by ARC Prize at any level)** |
| Errors | none (no OOM, no scheduler error), 22/22 finished |

## 86k budget

- **Modes:** 9 natural, 13 forced. At 49k, run7 had 19/22 forced.
- **4 natural closes landed beyond the old 49,152 cap:** 58.4k (`b6f77b65#1` ✓ **hit**), 59.6k, 69.5k, 72.2k.
- **So the 86k budget produced one hit that 49k would have cut**, plus three more natural closes.
- All 13 forced closes (77.8k) missed.
- **Engine:**
  - KV max **91.4 %**;
  - **0 preemptions**;
  - MTP acceptance 0.655;
  - 306.6 tok/s overall (lower than at 49k: longer contexts, and fewer active sequences late in the run);
  - setup 701 s.
- **Timing:** primary units finished by 118 min (two waves). The last primary unit was admitted at 53.9 min.

## Tail resample

| | Value |
|---|---|
| Resamples admitted / finished | 14 / 14 (13 bases were forced closes, 1 was a long natural, `2b83f449`) |
| Resample modes | 13 forced again, 1 natural (`2b83f449`, 75.3k) |
| attempt_2 changed | 12 |
| **New hits** | **0** |
| Cut by the drain at 147 min | 4 (`b6f77b65#0`, `b6f77b65#2`, `8b7bacbf#0`, `8b7bacbf#1`) |

- On these EVAL12 forced-close units (tasks nobody solves, except `20a9e565` and `898e7135` by ARC Prize), a second 77.8k sample also failed to close.
- The mechanism worked as designed: it started only after the queue was empty, before the admission deadline, and attempt_1 never changed. But on this very hard set it added nothing.
- This does **not** decide the hidden case (easier puzzles, more natural closes). It does show that resampling units that are forced at 77.8k rarely flips them on evaluation-hard tasks.

## Notes

- Sample size is tiny (22 inputs); run-to-run variance on EVAL12 is 1–3 hits at 49k.
- Added to `resources/Puzzle Capability Ledger` (primary units).
