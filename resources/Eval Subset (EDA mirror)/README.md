# EVAL20: a 20-task evaluation subset built with the EDA notebook's method (2026-10-12)

**Source:** `ARC2_Advanced_EDA_v8.ipynb` in this folder (the user's EDA, originally `D:/arc-2-advanced-eda (8).ipynb`; sha256 `1a9384d4…`).
- `puzzle_features()` comes from cell 35. Its tier and fine scores are the same as NVARC's `get_puzzle_priority`.
- The tags and the weighted **mirror gap** come from cell 49: the distance, in percentage points, between a subset's category mix and `TEST_TARGET_PCT`. Lower is better.

**Script:** `select_eval20.py <evaluation_challenges.json>` uses greedy filling plus a swap search, with a fixed seed.

## Result

| Set | Tasks | Test outputs | Mirror gap |
|---|---|---|---|
| EVAL12 (used so far) | 12 | 22 | **25.37 pp** (much harder than the target: 17 % tier 3, only 8 % easy_static, 75 % multi-query) |
| All 120 evaluation tasks | 120 | 172 | 19.78 pp |
| **A: EVAL12 + 8** | 20 | 30 | 11.47 pp |
| **B: fresh 20** | 20 | 22 | **1.32 pp** |

**Set A adds:**
- 7 easy_static tasks: 1818057f, 271d71e2, 332f06d7, 409aa875, 53fb4810, 7666fa5d, da515329;
- 1 dynamic tier-2 task: edb79dae.

**Set B:**
```
["136b0064","16b78196","1818057f","20270e3b","271d71e2","2b83f449","2ba387bc","332f06d7","409aa875","4a21e3da",
 "53fb4810","6e453dd6","7666fa5d","8698868d","89565ca0","898e7135","8f3a5a89","97d7923e","da515329","db695cfb"]
```
- Mix: tier 1 / 2 / 3 = 75 / 20 / 5 %; easy_static 60 %; rare 55 %; multi-query 10 %; dynamic 30 %.

## Caveats

1. **Recorded DIRECT answers exist only for EVAL12** (run 7).
   - Set B shares 3 tasks with EVAL12 (136b0064, 2b83f449, 898e7135); Set A shares all 12.
   - Any task without a recorded answer needs one DIRECT reference run before agent-vs-DIRECT comparisons, or a live DIRECT stage.
2. **`TEST_TARGET_PCT` was measured on the visible `arc-agi_test_challenges.json`.** That file is training-like (its average cell count of 182 is close to training's 171, against 353 in evaluation).
   - The hidden test set may be harder than this target. EVAL20-B is a proxy for that distribution, not a guarantee.
