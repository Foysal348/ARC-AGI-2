# CPU validation: ORDER2 + LOOPSTOP (Claude, 2026-10-15)

**Notebook:** `Experiment Notebooks/Q27-XHIGH-v1_4_4_E10_86K_ORDER2_LOOPSTOP.ipynb`, sha256 `8f8c732d765eb1fd…`.
- Builder: `Build/build_e10_order2_loopstop.py`. It is deterministic; two builds give the same sha.
- Source: ORDER2, sha `b62ae567…`.
- Diff: `ORDER2_TO_LOOPSTOP.diff` (cells 1 and 6 only, plus the markdown).

## The one change

A thinking trace that has degenerated into an exact short repetition is force-closed early.
- It uses the same path as a trace at the cap: stop text + `</think>` + the answer phase.
- The slot is freed for the next unit.

**When the check runs:**
- only after **49,152** thinking tokens (the user's floor);
- only while `</think>` has not been written;
- every 256 tokens.

**What counts as a loop:**
- In the last 1,024 tokens, there is a period p ≤ 128 tokens with ≥ 97 % of tokens equal to the token p back.
- The repeating unit must contain a letter. Grid rows of digits are never flagged.
- The loop must be seen on 2 consecutive checks.

**Switch and reporting:**
- `Q27X_LOOP_STOP=0` turns the feature off.
- Records carry `loop_stop_at`; the report carries `scheduler.loop_stopped`.

## Offline check on real traces (session scratchpad `order2/loop_detector_offline*.py`)

| Set | Result |
|---|---|
| 333 saved thinking tails (ORDER2 172, E9 36, E8 41, run4/6/7 66, dual v2 22) | **5/5 real loops found**: 4 in ORDER2, plus `20a9e565#1` in dual v2 ("row? S? …"). **0 false positives** |
| 41 full E8 traces (1.86 M tokens), checked from token 1,024 rather than 49,152 | **0 flags** |
| Three 30×30 uniform or striped grids written back to back | not flagged |
| The notebook's own pure-Python `loop_period` | identical results to the numpy prototype |
| CPU cost | 0.42 ms per check; about 5 s over a whole EVAL120 run |

## Harness (`Build/harness/harness_loopstop.py` = `harness_e10` + scenarios loop / loopoff)

| Scenario | Log | Result |
|---|---|---|
| loop | `loop.txt` | **PASS.** 43 looping traces stopped at 3.2–3.6k of a 12k harness cap (`LOOP_MIN_THINK` lowered to 2,048 in the harness only; the notebook value of 49,152 is asserted first). 43 grid-row traces and 43 varied forced traces ran to the cap, untouched. All 172 hit. Phase-2 prompts end with the stop text. `loop_stopped` = 43 |
| loopoff | `loopoff.txt` | **PASS.** 0 stops; looping traces run to the cap |
| mixed | `mixed.txt` | **PASS**, as for ORDER2 (172/172) |
| drain | `drain.txt` | **PASS**: drain and hard stop |
| rerun | `rerun.txt` | **PASS**: 259/259 hidden-test units; model grids written to `submission.json` |

## Expected effect

- In the ORDER2 EVAL120 run, 4 of 172 units ended in such a loop.
- Each stop saves the thinking between the loop's start (≥ 49,152) and 77,824: at most 28,672 tokens per unit, and the slot frees earlier.
- How much is really saved depends on when the loops start. Only a GPU run can show this.

**Added the same day: no false positives on the Poetiq R5 traces.**
- The 172 full Poetiq R5 thinking traces (11.25 M tokens, heavy in code and grids) were checked from token 1,024: **0 flags**.
- Negative evidence in total: 13.1 M tokens of real thinking with 0 false positives.
