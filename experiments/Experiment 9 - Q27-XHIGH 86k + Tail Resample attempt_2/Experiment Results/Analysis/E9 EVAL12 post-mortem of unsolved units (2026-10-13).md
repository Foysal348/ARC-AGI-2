# E9 EVAL12: post-mortem of the 18 unsolved units (Claude, 2026-10-13)

**Method:** each unit's attempt_1 (and its resample's grid, where one ran) was compared with the evaluation solution; "copy acc" is how close a plain copy of the test input would be. The last 1,500 thinking tokens of each unit were read.

## A. Rule found, but the grid was written wrong: **5 units, the cheapest to fix**

| Unit | Close | Error | Note |
|---|---|---|---|
| `247ef758#1` | natural 42k | **1 / 216 cells** wrong | sibling `#0` was solved with the same rule |
| `4c3d4a41#1` | natural 25k | **2 / 160 cells** wrong | sibling `#0` was solved |
| `2b83f449#0` | natural 72k | **18×17 instead of 19×17 (one row dropped)** | train pairs are all same-size, so the error is detectable at runtime. The resample got 3/323 wrong |
| `6e4f6532#1` | natural 69k | **28×16 instead of 30×16 (2 rows dropped)** | same-size task, detectable |
| `88bcf3b4#1` | forced 77k | **26×27 instead of 27×27 (one row dropped)** | same-size task, detectable; the resample made the same mistake |

**What these show:**
- The model is confident in its tails ("Good. Now final…"), yet the output transcription of large grids loses rows or flips cells.
- **3 of the 5 are shape errors on tasks where every train pair keeps the input size.** A host-side shape guard can detect them (answer shape ≠ test-input shape while all train pairs are same-size). It could trigger a cheap answer-only retry: continue from the existing thinking (phase 2), so no new thinking is needed.

## B. Natural close, but the wrong rule: 1 unit

- `4c7dc4dd#0` (tier 3, natural 59k): XOR hypothesis, 12/25 cells wrong.

## C. Forced close with partial understanding (same shape; better than a copy, or close to it)

| Unit | Model acc | Copy acc | Task kind (from tails) |
|---|---|---|---|
| `88bcf3b4#0` | 0.965 | 0.950 | path / outline drawing |
| `8b7bacbf#1` | 0.965 | 0.932 | fill closed loops (enclosure test) |
| `8b7bacbf#0` | 0.949 | 0.947 | same task, about no better than a copy |
| `b6f77b65#0` | 0.854 | 0.778 | gravity / falling blocks |
| `b6f77b65#2` | 0.840 | 0.660 | same (sibling `#1` solved at 58k) |

The tails are still testing hypotheses ("Maybe…", "Not.", "Hard.") at 77.8k: the rule was not settled.

## D. Forced close, rule not found: 6 units

| Unit | Result |
|---|---|
| `20a9e565#0` | wrong shape (3×21 vs 2×21) |
| `20a9e565#1` | 0.60 |
| `21897d95#0` | 0.016 |
| `21897d95#1` | 0.25, worse than a copy |
| `4c7dc4dd#1` | wrong shape (5×5 vs 6×6) |
| `898e7135#0` | 0.58 |
| `6e4f6532#0` | no grid |

These are mostly dynamic / extreme-ratio or multi-object tasks. ARC Prize solved `20a9e565` and `898e7135` at XHigh only.

## E. The tail resamples on these units

- 14 resamples ran, 0 correct.
- Several got closer: `2b83f449` 3 cells, `88bcf3b4#0` 15 cells.
- Others repeated or worsened the error; for example `88bcf3b4#1` gave the same 26×27.

## Lessons

1. **5 of 18 failures (28 %) are transcription or execution errors after a correct rule.** That is the largest cheap opportunity, especially multi-input siblings and large grids.
2. **3 shape errors were detectable at runtime** (same-size tasks). Candidate fix, for the user's decision:
   - a deterministic shape guard;
   - plus an answer-only retry from the saved thinking (cheap, no new thinking);
   - plus marking the unit high-priority for any second attempt.
3. **The forced closes are genuine reasoning failures** on hard EVAL tasks. More thinking (beyond 77.8k) or new seeds rarely fixes them here.
4. **Multi-input tasks:** siblings disagree even when one is solved (`247ef758`, `4c3d4a41`, `b6f77b65`). Reusing the solved sibling's reasoning, or asking for all test outputs in one answer, could recover these.

## Corrections after the GPT audit (validated 2026-10-13; see `GPT post-mortem audits validation (2026-10-13).md`)

- **Group A is not "transcription only."**
  - After the best row insertion, `2b83f449#0` still has 13 wrong cells and `88bcf3b4#1` 17. Only `6e4f6532#1` (2) is a near-pure row omission.
  - `247ef758#1` (layer order) and `4c3d4a41#1` (shift condition) are small **rule** errors.
- `6e4f6532#0` is not "no grid": it is a 31×28 grid that the parser rejects.
- **The shape guard flags 4 primary units:** `2b83f449#0`, `6e4f6532#0`, `6e4f6532#1`, `88bcf3b4#1`. It flags no hits.
- Group D has 7 units, not 6.
- **Resample:**
  - 9 hit the full cap, 1 closed naturally, 4 were cut by the drain.
  - 41.3 % of the generated tokens, 0 gain.
  - `6e4f6532#1` was skipped (69,486 < 70,041.6).
