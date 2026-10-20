# Validation of the GPT post-mortem audits for E9 and E8 (Claude, 2026-10-13)

**Inputs:**
- the two GPT audits the user pasted (E9 post-mortem, E8 full-thinking post-mortem);
- `E9_ANALYSIS_AND_EVIDENCE.zip`, extracted to `GPT E9 analysis zip (extracted)/`.

**Method:**
- `verify_gpt_audits.py` (session scratchpad) recomputed every claim from **our own copies** of the run outputs, not from GPT's JSON.
- Grids were parsed with the E9 notebook's own parser (cell 5).
- Row alignment is a minimum-edit match: a missing or extra row costs the row width, and a matched row costs its mismatched cells. It uses the solution, so it is a **diagnostic only**, never a runtime repair.
- `diagnostic_prototypes.py` was read in full first (pure Python, it only reads the zip), then run with `python -I`.

## E9 (EVAL12, 86k): GPT's claims

| Claim | Our recomputation | Verdict |
|---|---|---|
| Tokens: primary 1,468,025, resample 1,033,046 (**41.3 %**), 0 gain | same | ✅ |
| 14 resamples = 9 full-cap + 1 natural + **4 drained** (248,254 tokens) | same; `b6f77b65#2~r1` stopped at 45,333 thinking | ✅ ("13 forced again" in our note mixed the cap with the drain) |
| `think_tokens` 77,847 on forced closes = 77,824 generated + 23 host stop tokens | all 13 forced: (77,847, 77,824) | ✅ (counting only) |
| `6e4f6532#1` (69,486) missed resample: threshold 0.9 × 77,824 = 70,041.6 | same | ✅ (the priority ignores a visible shape error) |
| `2b83f449#0` 18×17: **13 cells** still wrong after inserting the best row | 13 | ✅ not a pure row drop |
| `6e4f6532#1` 28×16: **2 cells** wrong after the 2 rows | 2 | ✅ the only near-pure row omission |
| `88bcf3b4#1` 26×27: **17** after the row; resample 6 | 17 / 6 | ✅ |
| `6e4f6532#0` = **31×28** grid rejected by the parser (not "no grid"); after alignment 56 (resample 96) | same | ✅ |
| Same-size shape guard flags **4** primary units (`2b83f449#0`, `6e4f6532#0`, `6e4f6532#1`, `88bcf3b4#1`); no hit is flagged; 8 including resamples | same | ✅ (our note said 3) |
| `247ef758#1` (6,12): model 4, truth 6 → a layer-order **rule** error | diff confirmed. A rule that puts the upper source on top passes 3/3 train + 2/2 test; the reverse order fails train 0 | ✅ |
| `4c3d4a41#1` (1,17) 7→5, (3,17) 0→7 → wrong condition for which column shifts | diff confirmed. The prototype rule passes 2/2 train + 2/2 test | ✅ |
| `2b83f449#0~r1` 3 wrong: (4,15), (10,1), (18,6), each 8 written where 3 belongs | same | ✅ |
| Our D group heading says 6 but lists 7 | yes | ✅ |
| Context fallback: `phase1_cap` drops below 77,824 when the prompt > 20,224, and `MIN_THINK_BUDGET = 4096` allows it below 49,152 | code confirmed. EVAL12 max prompt 7,464, so nothing was reduced. Going below 49k needs a prompt > 48,896, which is practically impossible for ARC | ✅ real, but low risk |

**Caveat (GPT states it too):** the two prototype rules were written after seeing the solutions. They prove that the model's **rule** was wrong. They are not a solver and promise no score gain.

## E8 (PROXY48, 49k): GPT's claims

| Claim | Ours | Verdict |
|---|---|---|
| After inserting the missing row, `35ab12c3#0` / `53fb4810#0` / `7b0280bc#0` still have **16 / 3 / 17** wrong | 16 / 3 / 17 | ✅ only `53fb4810` is close to a pure row drop |
| `4a21e3da#0`: besides the 20-wide row 18, **9 more cells** are wrong | 9 | ✅ |
| `7b80bb43#0` is a **near miss**: 4 wrong, while a copy is 14 wrong (model 16 edits) | 4 vs 14: (14,10–12) 9 written where 8 belongs, (22,14) 8 written where 9 belongs | ✅ our "D: rule not found" was **wrong** |
| `6e453dd6#0`: 9 cells, all 2→6, in 3 rows × 3 columns | 9, all 2→6, rows 5/6/9, cols 9–11 | ✅ a partial rule (copy is 68 wrong) |

## What changes in our post-mortems

1. **"Rule found, only transcription"** is overstated in both runs.
   - Most shape errors also hide rule or geometry errors.
   - Near-pure row drops: only `6e4f6532#1` (E9, 2 cells) and `53fb4810#0` (E8, 3 cells).
   - `247ef758#1` and `4c3d4a41#1` are small **rule** errors, not digit slips.
2. **The shape guard covers 4 E9 primary units** (not 3), plus 3 in E8 (and 2 invalid grids).
   - It is a reliable detector.
   - **Recoveries are not proven:** realistically 0–2, of which `6e4f6532#1` and `53fb4810#0` are the most likely.
3. **E8 `7b80bb43#0` moves to near miss** (forced at 49k), so E8 has 4 forced near misses.
4. **E9 tail resample:** 41 % of the generated tokens for 0 gain.
   - 4 of the 14 were cut by the drain, one below 49k thinking.
   - The priority ignores visible shape errors (`6e4f6532#1` was skipped).
5. **No plan change is forced.** GPT's own E10 suggestion is ours too: shape guard + one answer-only repair as attempt_2, with nothing else changed. Implementation only with the user's go.

## Not accepted / open

- GPT's "train-verified program + feedback" direction (its E12) is a large change. Our earlier code-execution runs scored 0–2/22. It stays a research idea and is out of scope.
- The vLLM structured-output (grammar) suggestion for the repair is untested with our `add_request` + MTP + fp8 setup, so it is not a first step.
