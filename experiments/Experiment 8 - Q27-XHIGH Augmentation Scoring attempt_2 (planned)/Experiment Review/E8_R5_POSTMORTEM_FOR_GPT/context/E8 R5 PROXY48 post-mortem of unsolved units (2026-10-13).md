# E8 R5 (PROXY48, 49k budget): post-mortem of the unsolved units (Claude, 2026-10-13)

**Run:** `foysalemonshanto/q27-xhigh-e8` v1.
- 41 of 56 units finished, 10 hits, so **31 finished units are unsolved**.
- 15 more were never finished (the engine died at minute 124; see `GPU PROXY48 E8 R5 analysis`).

**Method:** attempt_1 was compared with the evaluation solution. "copy" = the accuracy of copying the test input. "same-size" = every train pair keeps the input size. The classification is automatic, with the thresholds shown.

| Class | Units | Rule used |
|---|---|---|
| **A1. Grid written with the wrong shape on a same-size task (transcription)** | **3**: `35ab12c3#0` (20×21 vs 21×21), `53fb4810#0` (24×18 vs 25×18), `7b0280bc#0` (29×30 vs 30×30) | answer shape ≠ input shape = solution shape. **Detectable at runtime** |
| **A2. No valid grid** | **2**: `271d71e2#0` (answer looped to the 8,192 cap, 158 rows), `4a21e3da#0` (21×21 with one 20-wide row; 18/21 rows correct) | **detectable at runtime** |
| **A3. Almost right (≤ 3 cells wrong, better than a copy)** | **3**: `dd6b8c4b#0` (1/121), `dd6b8c4b#1` (2/121), `28a6681f#0` (2/100) | **all forced at 49k**: the rule was nearly found, but thinking ran out |
| B. Natural close, wrong answer | 6 | `3dc255db` 4/156, `8f3a5a89` 10/144, `db695cfb` 35/525, `97d7923e`, `dbff022c`, `cb2d8a2c#1` (the last three no better than a copy) |
| C. Forced, partial rule (≥ 3 points better than a copy) | 8 | `269e22fb` ×2 (dynamic), `2b83f449`, `332f06d7`, `4a21e3da#1`, `6e453dd6` (9 cells), `71e489b6#0`, `7666fa5d` |
| D. Forced, rule not found (about a copy, or worse) | 9 | `135a2760`, `36a08778#1` (its sibling `#0` was solved), `71e489b6#1`, `7b80bb43` (4/493 wrong, but a copy is 0.972), `800d221b`, `c7f57c3e`, `cb2d8a2c#0`, `da515329`, `de809cff` |

## Findings

1. **5 of 31 failures (16 %) are detectable without the solution:** a wrong shape on a same-size task (3), or no valid grid (2). Together with E9 (3 of 18), that is **8 of 49 failures across both runs.** An answer-only retry from the saved thinking, with the required H×W stated, targets exactly these.
2. **3 forced units were within 1–2 cells at 49k** (`dd6b8c4b` ×2, `28a6681f`). These are **budget cases**: the rule was nearly found when thinking was cut. They are the kind the 86k budget (E9) or the "pause at 49k, resume later" idea addresses.
3. **Natural-but-wrong (6):** two are near misses (`3dc255db` 4 cells, `8f3a5a89` 10 cells); the rest are wrong rules. They cannot be detected at runtime by shape.
4. **Forced with no rule (9) and partial (8):** genuine reasoning failures at 49k. Most are tier 1, so "NVARC-easy" is not Qwen-easy.
5. **Siblings:** `36a08778#0` was solved, but `#1` was forced and wrong (0.667). The same pattern appears in E9 (`247ef758`, `4c3d4a41`, `b6f77b65`).

## Combined with E9 (EVAL12, 86k)

| Failure class | E8 (49k, 31) | E9 (86k, 18) |
|---|---|---|
| Detectable: shape or invalid grid | 5 | 3 |
| Near miss, ≤ 3 cells | 3 (all forced) | 2 (natural; siblings solved) |
| Natural, wrong rule | 6 | 2 (`4c7dc4dd#0` + `6e4f6532#1`, which was a shape error, counted above) |
| Forced, partial / no rule | 17 | 11 |
