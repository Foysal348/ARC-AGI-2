# E8 R5 GPU run (PROXY48, 49k budget): deep failure analysis and solution proposals (2026-10-13)

**Request:**
1. Analyse every unsolved finished unit in depth: what the model understood, and whether the error is reasoning, execution, transcription or budget.
2. Find the underlying problems and propose concrete, testable solutions.

Verify everything, including Claude's post-mortem. The **PROXY48 solutions are included** (`data/`), so you can score independently.

**User rules:**
- base v1.4.4 DIRECT (LB 26.94);
- never less than 49,152 thinking + 8,192 answer;
- a valid `submission.json` always;
- simple changes, one at a time;
- no run or push without approval.

## The run

| | |
|---|---|
| Kaggle | `foysalemonshanto/q27-xhigh-e8` v1 |
| Notebook | `notebook/…E8_PER_UNIT_R5_FINAL_CPU.ipynb`: v1.4.4 + Codex ordering + E8 candidate mining / scoring. The E8 part is irrelevant to this analysis; the primary solver is v1.4.4 |
| Set | PROXY48: 48 tasks / 56 inputs, 3 h, 16 concurrent, 49,152 + 8,192, temperature 0.6, xhigh, MTP×3, fp8 KV |
| Result | 41/56 finished (the engine died at minute 124 from an E8 scorer OOM, so 15 inputs never finished); **10/56 hits**; 15 natural (9 hits) and 26 forced (1 hit) |

## Claude's post-mortem of the 31 unsolved finished units (`context/E8 R5 PROXY48 post-mortem…md`)

| Class | n | Units |
|---|---|---|
| A1. Wrong shape on a same-size task (transcription; detectable) | 3 | `35ab12c3#0`, `53fb4810#0`, `7b0280bc#0` (each missing one row) |
| A2. No valid grid (detectable) | 2 | `271d71e2#0` (answer loop), `4a21e3da#0` (one jagged row) |
| A3. ≤ 3 cells wrong (all forced at 49k: budget) | 3 | `dd6b8c4b` ×2, `28a6681f#0` |
| B. Natural close, wrong | 6 | `3dc255db` (4 cells), `8f3a5a89` (10), `db695cfb`, `97d7923e`, `dbff022c`, `cb2d8a2c#1` |
| C. Forced, partial rule | 8 | `269e22fb` ×2, `2b83f449`, `332f06d7`, `4a21e3da#1`, `6e453dd6`, `71e489b6#0`, `7666fa5d` |
| D. Forced, rule not found | 9 | `135a2760`, `36a08778#1` (sibling solved), `71e489b6#1`, `7b80bb43`, `800d221b`, `c7f57c3e`, `cb2d8a2c#0`, `da515329`, `de809cff` |

**Combined with E9** (EVAL12, 86k; a separate bundle): **8 of 49 failures are detectable at runtime** (shape or invalid grid).

**Claude's proposed fixes (not implemented):**
1. A shape guard plus an answer-only retry from the saved thinking (state H×W).
2. Reuse a solved sibling's reasoning for the other test input.
3. More thinking for the forced near misses: 86k (now tested in E9), or "pause at 49k, resume later".

## Questions

1. **Per unit:** the error type with evidence from `outputs/q27x/answers/*.txt` (the last 1,500 thinking tokens plus the answer) and `outputs/q27x/e8_full_thinking/*.txt` (the **full** thinking for all 41 finished units, which is rare data).
2. **Using the full thinking traces:** where does the reasoning go wrong for the C/D units? Wrong object segmentation, a wrong hypothesis kept, a correct rule mis-applied, looping? Is there a common, fixable pattern, for example in the prompt (input format) or the answer format?
3. **Which classes are cheaply fixable** without lowering the budget, and how many of the 31 would each fix recover?
4. **Evaluate Claude's fixes 1–3:** risks (could they turn a correct answer wrong?), cost, and CPU/GPU tests.

## Files

| Path | What |
|---|---|
| `notebook/` | the notebook that ran |
| `outputs/q27x/records.jsonl` | 41 finished units (mode, think tokens, score, tier, admit/finish minutes) |
| `outputs/q27x/answers/*.txt` | thinking tail and answer per unit |
| `outputs/q27x/e8_full_thinking/*.txt` | the full thinking text per finished unit |
| `outputs/q27x/e8_primary_baseline.jsonl` | the attempt_1 / attempt_2 grids per unit |
| `outputs/q27x/run_report.json`, `vllm_*`, `outputs/q27-xhigh-e8.log`, `*_stdout.txt` | reports and logs |
| `data/proxy48_challenges.json`, `data/proxy48_solutions.json` | the 48 tasks with solutions |
| `context/` | Claude's run analysis and post-mortem, the E8 final report, the capability ledger (ARC Prize per-task results, category profile, oracle) |
