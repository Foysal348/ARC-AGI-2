# E9 GPU run (EVAL12): deep failure analysis and solution proposals (2026-10-13)

**Request:**
1. Give your own reading of this run.
2. **Analyse every unsolved unit in depth:** what the model understood, where it went wrong, and whether the error is reasoning, execution or transcription.
3. Find the underlying problems and **propose concrete, testable solutions.**

Verify every claim below against the files, including Claude's post-mortem. The evaluation solutions for the 12 tasks are included (`data/`), so you can score independently.

**User rules (fixed):**
- the base is v1.4.4 DIRECT (LB 26.94);
- never less than 49,152 thinking + 8,192 answer per test input (this run used 77,824 + 8,192);
- a valid `submission.json` must always be produced;
- keep changes simple, one change at a time;
- nothing is run or pushed without the user's approval.

## The run

| | |
|---|---|
| Kaggle | `esfersami50/q27-xhigh-e9` v1, Save & Run All |
| Notebook | `notebook/Q27-XHIGH-v1_4_4_E9_86K_RESAMPLE.ipynb` (sha256 `845ff050…`) |
| Change versus v1.4.4 | thinking 77,824 (was 49,152), `MAX_MODEL_LEN` 106,496, tail resample for attempt_2, original order |
| Set | EVAL12: 12 tasks / 22 test inputs, 3 h |
| Engine | Qwen3.8-27B-FP8, vLLM 0.19.0 TP4 on 4×L4, MTP×3, fp8 KV, prefix cache, 16 concurrent |
| Sampling | temperature 0.6 / top_p 0.95 / top_k 20; xhigh; host-forced close (stop text + `</think>`) at the thinking cap |

**Result** (from `outputs/q27x/run_report.json`, `records.jsonl`):
- **4/22 (18.18 %):** `136b0064#0`, `247ef758#0`, `4c3d4a41#0`, `b6f77b65#1` (`b6f77b65#1` closed naturally at 58.4k).
- Earlier v1.4.4 runs at 49k on the same set scored 2, 2, 3 and 1.
- Modes: 9 natural (4 hits), 13 forced (0 hits).
- **Tail resample:** 14 ran, 13 forced again, 12 changed attempt_2, 0 gains.
- **Engine:** KV max 91.4 %, 0 preemptions, 306.6 tok/s.

## Claude's post-mortem (`context/E9 EVAL12 post-mortem…md`); please verify or refute

| Group | Units | Claim |
|---|---|---|
| **A. Rule found, grid written wrong** | `247ef758#1` (1/216 cells), `4c3d4a41#1` (2/160), `2b83f449#0` (18×17 vs 19×17), `6e4f6532#1` (28×16 vs 30×16), `88bcf3b4#1` (26×27 vs 27×27) | 5/18 failures are transcription or execution, not reasoning. The 3 shape errors are on tasks whose every train pair keeps the input size, so they are detectable at runtime |
| B. Natural close, wrong rule | `4c7dc4dd#0` | XOR hypothesis, 12/25 cells wrong |
| C. Forced, partial rule | `88bcf3b4#0`, `8b7bacbf#0`, `8b7bacbf#1`, `b6f77b65#0`, `b6f77b65#2` | still testing hypotheses at 77.8k |
| D. Forced, rule not found | `20a9e565` ×2, `21897d95` ×2, `4c7dc4dd#1`, `898e7135`, `6e4f6532#0` | dynamic / extreme-ratio / multi-object |

**Claude's proposed fixes (not implemented; user approval needed):**
1. **Shape guard + answer-only retry:**
   - infer the expected output shape from the train pairs (same as input, or a constant shape);
   - on a mismatch, a jagged grid or no grid, re-run only the answer phase from the saved thinking, with "the output grid must have H rows and W columns".
   - This is cheap: no new thinking.
2. **Multi-input siblings:** reuse the reasoning of a naturally solved sibling input when answering the other input (or answer all test inputs together).
3. **Larger options:** diff-style answers for same-size tasks; code-executed answers.

## Questions

1. **For each of the 18 unsolved units:** the error type (reasoning / execution / transcription / budget), with evidence from `outputs/q27x/answers/*.txt` (the last 1,500 thinking tokens plus the full answer) and the solutions.
2. **Which failure classes are fixable cheaply** without lowering the thinking budget? Estimate how many of the 18 each fix would recover, based on these files.
3. **Evaluate Claude's fixes 1 and 2:** correctness, risks (could they make a correct answer wrong?), cost, and how to test them on CPU and on GPU.
4. **Anything else** you find in the logs or the notebook (bugs, wasted time, better use of the tail).

## Files

| Path | What |
|---|---|
| `notebook/` | the notebook that ran |
| `outputs/q27x/records.jsonl` | 22 primary records and 14 resample records (mode, think tokens, score, admit/finish minutes) |
| `outputs/q27x/answers/*.txt` | per unit: the thinking tail (last 1,500 tokens) and the full answer; `*_r1.txt` = the resample |
| `outputs/q27x/run_report.json`, `vllm_metrics.json`, `vllm_frontend.log` | report and engine metrics |
| `outputs/q27-xhigh-e9.log`, `q27-xhigh-e9_stdout.txt` | the Kaggle log (raw and flattened) |
| `outputs/submission.json` | the eval-mode identity test submission |
| `data/eval12_challenges.json`, `data/eval12_solutions.json` | the 12 tasks with solutions, for independent scoring |
| `context/` | Claude's result note and post-mortem, the Experiment 9 README, the capability ledger (ARC Prize per-task results, category solve rates, oracle) |
