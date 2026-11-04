# E12 Duck REPL agent (Qwen3.8-27B, ARC-AGI-2): weakness analysis (2026-10-20)

## Ask

Find and rank the **weaknesses of our agent** from the evidence in this bundle, and say what would most raise the solve rate per GPU-hour.

- Separate *measured* from *inferred*.
- Check our findings below; tell us where they are wrong or incomplete.

**Setting:** Kaggle ARC Prize 2026 (ARC-AGI-2), 4xL4, 12 h, about 240 hidden tasks. Our best LB is 34.58 (a non-agent Poetiq pipeline).

## The agent

- **Harness:** Duck REPL harness (da-fr/arc-agi-3-solution @ 10882e33, built for ARC-AGI-**3** games), adapted to ARC-AGI-2 by `adapter/arc2_duck.py`.
- **Model:** Qwen3.8-27B INT4 on SGLang 0.5.21, thinking on, temperature 0.7 / top_p 0.95 / top_k 20 (Duck milestone settings).
- **Tools and verification:**
  - one `python` tool, with globals `train`, `test_inputs`, `check(transform)` and `submit(transform)`;
  - `submit` re-verifies on the host, and an accept ends the branch.
  - 2 independent branches per task.
  - Duck yields after every request (2,048-token yield), and the solver resumes with "Nothing has been submitted since the message above … Your tool results above are still valid: continue from them".
- **Context:** window 131,072, so the input budget is window − max_output − 512. History is trimmed in whole blocks.
- **Note on transcripts:** a `[USER PROMPT]` with ARC-AGI-3 words (current_frame, action(), world model) is the text Duck **logs before** the adapter rewrites it (`tool_agent.py:6983-7044` builds and marks it as a nudge; `arc2_duck.py:344` replaces the game tail). The model received the ARC-2 version ("…write or fix `transform(grid)`, run `check(transform)` …, `submit(transform)` once every pair is exact").

## The runs

| Run | Output per request | Per-branch cap | Branches | Result |
|---|---|---|---|---|
| **EVAL30** (2026-10-19) | 12,288 | none; 30 live, KV thrash | branch 1 ran in only 1 task | 7/48 inputs, 4/30 tasks; Poetiq ~12/30 |
| **EVAL6** (2026-10-20) | 60,000 | 100k soft | 6 live, both branches | 3/6 tasks fully solved, on 6 tasks EVAL30 Duck solved 0 of (selected because other solvers solved them) |

## Findings (ranked by our estimate of lost solves)

### W1. Correct functions found but never submitted, and nothing harvests them (measured)

- **EVAL6:** replaying the tool calls (`analysis/replay_all.py`) shows **two branches had a train-exact function that is also test-correct**, and neither submitted:
  - 62593bfd_b1: check 2/2, both test inputs correct;
  - b5ca7ac4_b0: check 3/3, test correct.
- **Why they did not submit:**
  - 62593bfd_b1 "verified before submitting", then called `check()` where it meant `submit()`; the soft cap stopped it.
  - b5ca7ac4_b0 reached 3/3 on the request that crossed 100k and was stopped.
- **The gap:** only `submit()` verdicts reach selection (`arc2_duck.py` `select_attempts`).
- **Effect:** with check-exact functions harvested, **EVAL6 = 7/8 inputs, 5/6 tasks** instead of 4/8 and 3/6 (reader D, `analysis/replay_all.jsonl`).

### W2. Fresh long replies dropped from context right after generation (measured; mechanism inferred)

- **EVAL6:** in 11 of 84 later requests, the next prompt did **not** carry the previous reply. Example: 62593bfd b0/b1, a 60,000-token reply, then a next prompt of 9,395 tokens = the system prompt alone.
  - That is **426,781 of 1,217,904 generated tokens (35 %) thrown out immediately**.
  - The model then restarts from zero, while the resume says "Your tool results above are still valid".
- **EVAL30:** 12 of 423 requests (98k tokens, 4 %).
- **Inferred mechanism (not verified in code):**
  - Duck's chars/token calibration is measured on the digit-heavy grid prompt (1.40-1.72 in the dropped cases, 2.10 in a kept one), so prose reasoning is overestimated against the 70,560 budget.
  - A tool-less reply then does not survive the trim.
- **Earlier claim was wrong:** in our pre-run validation we wrote "the newest block always stays". That is not true for these replies.

### W3. Degenerate repetition loops in long thinking (measured)

- **EVAL6:** **29.4 %** of thinking characters are verbatim repeated long lines; 11 of 96 thinking blocks are more than 20 % repeats. EVAL30 (12k cap): 5.2 %.
- **Reader A:** about 131k of 368k tokens in three failed branches were loops ("So that's not it either." x N).
- **Settings:** temperature 0.7, no presence/repetition penalty, no loop detection, a single request may use 60k.

### W4. Thinking in the head instead of testing in code (measured + transcript reading)

- **Failed branches:**
  - the median share of tokens spent before the first `check()` is **100 %**;
  - 22/24 failed EVAL30 branches and 5/8 failed EVAL6 branches never called `check()`;
  - ~~no failed branch ever produced a function that got even one train pair exact~~ **WRONG (GPT 1, verified):** 62593bfd_b1 (2/2) and b5ca7ac4_b0 (3/3) are unaccepted branches with exact checks (W1), and EVAL30 135a2760_b0 reached 1/2. Correct: most failed branches never ran `check()`.
- **Solved branches:** they start coding early (median 85-87 % of tokens before the first check, but small totals), read the grids through code, and accept lookup tables (136b0064_b1 `patmap`).
- **Hand perception errors** sent branches down wrong paths (readers A, B, C):
  - misread cells and sizes;
  - colour confused with length;
  - a phantom glyph;
  - wrong hand-scaling that killed a correct hypothesis (898e7135_b0 L2188).
- **"Waiting for a general explanation":** 136b0064 found a rule that fit all pairs and would not code a lookup. The prompt line "derive positions, sizes, colors … from the input" may push this.

### W5. Overfit accepts; attempt_2 is wasted (measured)

- **Wrong accepts:** 4 of 11 accepted programs were wrong on test (36 %).
  - Each used an arbitrary tie-break the model itself called "arbitrary" or "odd".
  - The test exercises exactly that case.
  - The model's test check is self-consistency (prediction = its own rule).
- **attempt_2:** it was the test input in **56 of 56** query slots and never correct. An accept ends the branch, so no alternative is ever kept (reader D).

### W6. Tool and API friction (measured)

- **`for i,(a,b) in enumerate(train)`** (train items are dicts) silently yields the keys 'input' / 'output' in 3 of 4 EVAL30 tasks read, plus 62593bfd_b0.
- **Retention rules** cause NameErrors ("Functions that read `train` … are not retained").
- **A tool call cut by the token cap** was executed as a 3-line stub.
- **EVAL30:** ~~4 x 900 s request timeouts~~ **corrected 2026-10-21: 40 timeouts in 21 of 31 branches (28 full 900 s + 12 deadline-bounded, about 31,000 stream-seconds)**; fixed by the 1,800 s timeout (EVAL6: 0).

### W7. Cost and coverage (measured)

- **Cost:** about 10 GPU-minutes per task (EVAL6: 12 branches in 62 min, 6 live).
- **Hidden set:** 240 tasks would need about 40 h against about 11 h of solving.
- **Cap overshoot:** the 100k soft cap overshoots to 159k because the cap is checked between requests.

### W8. Branches are fully independent (observation)

- b0 and b1 re-derive the same facts.
- In EVAL6, b0 accepted 0/6 and b1 4/6 with identical prompts (n = 6, possibly chance).

## Candidate fixes (for discussion; nothing applied)

1. **Harvest every check-exact function** (and auto-submit it) → W1. Use distinct check-exact functions from both branches as attempt_1/attempt_2 → W5.
2. **Never drop the newest reply:** fix the token estimate (separate calibration for reasoning text, or the server's real `prompt_tokens`), or stub dead replies. Make the resume message truthful ("your previous reply was cut off / dropped; call python now") → W2.
3. **Loop control:** a repetition detector to stop and resume, sampling closer to the Qwen recommendation (temperature 0.6, presence penalty), or a shorter per-request ceiling with real continuation → W3.
4. **Prompt:**
   - code-first ("read every grid in code; never transcribe grids by hand");
   - "a lookup table that is exact is acceptable";
   - "list rules that fit training and compare them on the test inputs before submitting" → W4/W5.
5. **Provide `train` as a list of (input, output) pairs as well** → W6.
6. **Coverage:** whatever the above, 240 tasks at 10 min do not fit. Order and admission are needed (our ORDER2 work) → W7.

## Files

| Path | What |
|---|---|
| `runs/EVAL6_q27x/` | Full EVAL6 outputs: `duck/<task>_b<k>/transcript.txt` (full thinking), `branch_result.json` (per-request usage), `e12_run_report.json`, SGLang logs, `submission.json` |
| `runs/EVAL30_q27x/` | EVAL30 transcripts and reports |
| `adapter/arc2_duck.py`, `adapter/vendor_inference_agent/` | ARC-2 adapter and the vendored Duck agent (`tool_agent.py`, prompts) |
| `analysis/agent_weakness_parse.py` + `_output.json` | Per-branch behaviour metrics |
| `analysis/replay_exact.py`, `replay_all.py`, `replay_all.jsonl` | Replays a branch's tool calls, finds every check-exact function and scores it against the truth |
| `analysis/eval6_analysis.py` + output | Run-level EVAL6 metrics |
| `agent_reports/A-D` | Four transcript readings (EVAL6 failed A/B, EVAL30 failed C, wrong accepts + selection D) |
| `data/eval_challenges_subset.json`, `eval_solutions_subset.json` | The 30 tasks with truth (contains the EVAL6 six) |
| `docs/` | Earlier run analyses (EVAL6, EVAL30) |
