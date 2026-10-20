# E11-06 P1: prompt redesign. Two GPT answers, verified, and the decision (2026-10-18)

## Verified against the notebook (S49 / E11-05 cell 4)

| Claim | Check |
|---|---|
| The donor prompts override the short ones | **True.** The end of cell 4 sets `POETIQ_SOLVER_PROMPTS = {1:SOLVER_PROMPT_1,...}` and `POETIQ_FEEDBACK_PROMPT = FEEDBACK_PROMPT`. The active prompt is the long donor text, with "Test and Refine", debugging, docstrings, an explanation and worked examples. |
| Example order is reshuffled every iteration | **True.** `poetiq_format_problem(task, shuffle, seed+iteration)`. |
| Feedback = up to 5 programs, each with full code and a full-grid diff | **True.** `poetiq_create_examples(max_examples=5)`, `poetiq_array_diff`. |
| A train-perfect program ends the expert's refinement | **True.** Kept: it is unchanged in P1. |

## Measured, so we do not guess

- **Solver prompt template:** 1,248 tokens (donor) → **429** (P1), with the real tokenizer.
- **An iteration-2 message with 3 earlier programs:** 9,204 → 6,392 tokens.
- **Feedback in EVAL30:** a median of +4.3k prompt tokens per later iteration; 35 % of phase-1 prompt tokens.
- **Prompt tokens are prefill:** 0.89M prefill against 4.25M decode in EVAL30, a few minutes of a 3.6 h run. Compact feedback is about the model's focus, not about speed. **The real lever is the generated thinking:**
  - test-grid simulation + code dry-runs = 10–11 % of the thinking (sim_tokens.py);
  - 94 % forced closes.

## Decision: one bundled variant, P1

1. **New solver prompt** (merged from both answers). It covers:
   - "the sandbox checks your code";
   - no hand execution, no writing out grids, no computing the Challenge output;
   - rule-level hypothesis testing;
   - a general, input-derived rule (constants only when the examples show them);
   - "write the code and stop" once one rule fits;
   - output = one python block, no explanation or docstring.
2. **New feedback prompt:**
   - "results are exact";
   - bug vs wrong rule;
   - no cell patches or special cases;
   - "train pass ≠ proof";
   - write one new program and stop.
3. **Feedback history: 5 → 2.**
   - The best program keeps its **full diff** (my choice; both GPTs wanted compact for all). The spatial error map is the most informative part, and it costs little: prefill is cheap.
   - The second, different program gets a compact summary: wrong-cell count, rows, columns, 12 sample cells.
4. **Stable example order** (no shuffle), which is cache-friendly.
5. **Drain back to 8 min.** In EVAL30 every drained answer finished in ≤ 83 s.

## Unchanged

- xhigh, the 49,152 cap (for a clean A/B against the E11-05 EVAL30 control), and the sampling (T 1.0 / 0.95 / 20).
- The grid encoding, the sandbox, the voting, the attempts, the engine and the forced-close path.

## Not in P1

| Idea | Why not |
|---|---|
| Grid re-encoding | Quality risk unmeasured |
| Repetition / logit penalties | DFLASH path not verified |
| Mid-stream wrap-up hint | No live-append API; a restart costs prefill |
| Two-stage rule → code requests | Extra requests |
| Removing the Challenge input | Helps generalization |
| Stop at the first fitting rule without a generalization check | Overfit risk |

## A/B plan

E11-06 P1 EVAL30 against the E11-05 EVAL30 control: the same 30 tasks, 4 h, engine and cap.

Metrics:
- hits, task-norm and the paired task table;
- natural-close %;
- thinking tokens per iteration;
- iterations per hour;
- train-pass and the overfit rate;
- the A+B share (sim_tokens.py);
- phase-1 cache %.

The P1 notebook is `Q27-XHIGH-v1_4_4_E11_06_P1_SGLANG_INT4_DFLASH2_EVAL30.ipynb`, built by `build_e11_05_s49_sglang.py --eval30 --p1` (+ `e11_06_p1_cell4_patch.py`).
