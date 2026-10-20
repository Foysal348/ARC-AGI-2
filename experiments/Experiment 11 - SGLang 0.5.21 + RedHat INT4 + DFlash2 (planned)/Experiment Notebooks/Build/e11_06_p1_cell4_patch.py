
# ==================================================================== E11-06 P1: execution-aware prompts (inserted after the donor block)
# Why (EVAL30 2026-10-18 + two GPT audits, verified against this cell): the donor prompt asks the model to "Test and
# Refine: Test your code on all examples" and to debug, document and explain. The model cannot run code while thinking,
# so it hand-simulates: test-grid simulation + code dry-runs = 10-11 % of all thinking (sim_tokens.py), while 94 % of
# iterations hit the 49,152 cap and 7/15 train-verified programs were wrong on test. P1 says the sandbox checks the code,
# asks for a general input-derived rule, and asks for code only. xhigh, the thinking cap, the sampling, the sandbox,
# the voting and the Poetiq loop are unchanged.
SOLVER_PROMPT_1 = '''You are an expert at solving ARC (Abstraction and Reasoning Corpus) tasks by writing Python code. The training pairs below define a grid transformation. Find the single rule that maps every training input to its output, and implement it as a Python function that also works on the Challenge inputs.

HOW YOUR CODE IS CHECKED
An external sandbox will run your `transform` on every training input, compare each result with the expected grid cell by cell, and return the exact results (pass or fail, wrong cells, shapes, errors) in the next round. You cannot run code while thinking, so do not try to: do not execute your code line by line in your head, do not write out predicted output grids cell by cell, and do not hand-compute the Challenge output. The sandbox does all of that exactly.

HOW TO SPEND YOUR THINKING
1. Study the training pairs: objects, colors and their roles, shapes, sizes, positions, counts, symmetry, connectivity and grid dimensions. Compare each input with its output.
2. Form a few genuinely different hypotheses and test them against the training pairs at the level of the rule (does it explain what changes and what stays the same?), not by redrawing grids.
3. Prefer the simplest rule that explains all pairs, and make it general: derive positions, sizes, colors and counts from the input. Use a fixed coordinate, color or size only when the examples show that it is part of the rule. Check that the rule also fits the structure of the Challenge inputs.
4. When one rule explains all training pairs and you know how to implement it, write the code and stop. Do not restart the same analysis or re-verify it by hand.

OUTPUT
Return exactly one complete markdown ```python code block that contains the imports and `def transform(grid: np.ndarray) -> np.ndarray`. NumPy and the Python standard library are available. No explanation, no docstring, no tests, no __main__ block and no other code blocks.

PROBLEM
Below are the training examples and the Challenge input(s).

$$problem$$
'''
FEEDBACK_PROMPT = '''
PREVIOUS PROGRAMS AND SANDBOX RESULTS
The sandbox ran each program below on every training input; these results are exact. Wrong cells are shown as prediction/correct. The Challenge outputs are unknown: for Challenge inputs only crashes and invalid grids are reported.

$$feedback$$

NEXT PROGRAM
1. Find why the best program fails: an implementation bug or a wrong rule. Use the wrong cells, shapes and errors above; do not re-run the old code in your head or redraw whole grids.
2. If it is a bug, fix it. If the rule is wrong, change the rule; do not patch single cells, coordinates or one example with special cases.
3. Passing every training pair does not prove the rule is right: keep it general and derived from the input.
4. Then write one new complete program and stop; the sandbox will check it.

Return exactly one complete ```python code block with the imports and `def transform(grid: np.ndarray) -> np.ndarray`. No explanation.
'''
POETIQ_SOLVER_PROMPTS = {1: SOLVER_PROMPT_1, 2: SOLVER_PROMPT_2, 3: SOLVER_PROMPT_3}
POETIQ_FEEDBACK_PROMPT = FEEDBACK_PROMPT
E11_PROMPT_VARIANT = "P1"


def _e11_compact_feedback(fb):
    """P1: a wrong-cell diff grid -> count, affected rows/columns and up to 12 sample cells spread over the grid.
    Used for the second (older) program only; the best program keeps its full diff grid."""
    def _summ(m):
        rows = [l.split() for l in m.group(1).strip().splitlines()]
        wrong = [(r, c, t) for r, row in enumerate(rows) for c, t in enumerate(row) if "/" in t]
        n = sum(len(row) for row in rows)
        rs = sorted({r for r, _, _ in wrong}); cs = sorted({c for _, c, _ in wrong})
        sample = wrong[::max(1, len(wrong) // 12)][:12]
        cells = ", ".join(f"({r},{c}: {t.split('/')[0]}->{t.split('/')[1]})" for r, c, t in sample)
        return (f"{len(wrong)} of {n} cells wrong; rows {rs}; columns {cs}; "
                f"sample (row,col: prediction->correct): {cells}")
    return re.sub(r"Wrong cells appear as prediction/correct:\n```\n(.*?)\n```", _summ, fb, flags=re.S)


def poetiq_create_examples(solutions, max_examples=2, improving_order=True):
    """P1: the best program (full feedback) + one different program (compact feedback); was the top 5, all in full.
    Every program stays in the history and in the voting."""
    if not solutions:
        return ''
    ranked = sorted(solutions, key=lambda z: z['score'], reverse=True)
    chosen = [ranked[0]]
    for s in ranked[1:]:
        if len(chosen) >= max(1, max_examples):
            break
        if s['code'].strip() != chosen[0]['code'].strip() and s['feedback'] != chosen[0]['feedback']:
            chosen.append(s)
    items = [(s, k == 0) for k, s in enumerate(chosen)]
    if improving_order:
        items.reverse()                          # the best program last, closest to the new attempt (as before)
    return '\n\n'.join(
        f'<solution_{k}>\n<solution_code>\n```python\n{s["code"]}\n```\n</solution_code>\n<solution_evaluation>\n'
        f'{s["feedback"] if best else _e11_compact_feedback(s["feedback"])}\n</solution_evaluation>\n<solution_score>\n'
        f'{s["score"]:.2f}\n</solution_score>\n</solution_{k}>' for k, (s, best) in enumerate(items, 1))
