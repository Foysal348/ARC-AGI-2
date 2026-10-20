# One thinking, two attempts (a Poetiq program + a DIRECT grid): a research brief (2026-10-14)

## The question

Can **one** reasoning pass (one long "thinking" generation) of Qwen3.8-27B produce **two good, different ARC answers**?
- **attempt_1:** the output of a Python `transform()` program that has been verified on the training pairs (Poetiq style).
- **attempt_2:** a grid the model writes directly (DIRECT style).

If it can, we get the complementarity of the two methods (see the evidence below) at roughly the cost of one method. If it cannot, what is the cheapest way to get two **genuinely independent** attempts?

**Please:**
1. Analyse our data in this bundle (details below).
2. **Search widely** on GitHub, arXiv, the web, Hugging Face and Kaggle (ARC Prize 2024/2025/2026 notebooks, write-ups and discussions) for methods that get program + grid, or two diverse answers, from one model under a tight compute budget. Examples to start from:
   - induction + transduction ensembles (BARC: "Combining Induction and Transduction for Abstract Reasoning");
   - Poetiq's and Imbue's two-output selection;
   - ARChitects / NVARC candidate scoring;
   - self-consistency between a program's output and a directly predicted grid;
   - "answer extraction from a reasoning trace" / multi-answer decoding from one trace;
   - prefix-cached answer branching in vLLM.
3. Give a ranked list of solutions. For each: the source link, the evidence (numbers), the extra cost in tokens and GPU time for our setup, the risks, and how it plugs into our notebook. Mark anything unverified as a hypothesis.

## Our setup (fixed constraints)

- **Competition:** Kaggle ARC Prize 2026 (ARC-AGI-2), **offline**. The hidden test is about 240 tasks / 259 test inputs, with a **12 h** limit; scoring is pass@2 per test input.
- **Model and serving:** **Qwen3.8-27B-FP8** on vLLM 0.19.0, TP4 on 4×L4.
  - fp8 KV cache, MTP×3 speculative decoding, prefix caching.
  - 16 concurrent requests, about 300–365 tok/s aggregate (about 22–27 tok/s per sequence).
  - KV capacity 1.35 M tokens.
- **Reasoning:** `reasoning_effort = xhigh` (the template accepts only xhigh / medium / low).
  - When thinking hits the cap, we inject a stop text plus `</think>` and the model writes the answer (`prompts/forced_close_stop_text.txt`).
- **Budgets:**
  - Never less than **49,152** thinking tokens per request (user rule).
  - Current runs use 77,824 thinking + 8,192 answer, with `MAX_MODEL_LEN` 106,496.
- **Our best LB:** 27.92 (DIRECT, 49k, with ordering).
- **Target:** about 60. Other teams report 60+ with this model.

## The evidence we have

| | DIRECT 86k (E9, `runs/direct_e9_86k_eval12/`) | Poetiq R4 (`runs/poetiq_r4_eval12/`) |
|---|---|---|
| Method | thinking → the grid written directly; one request per test input | thinking → `transform()` code → sandbox run on the training pairs → diff feedback → refinement (up to 10 iterations); one request per task, covering all its test inputs |
| EVAL12 (12 tasks / 22 inputs) | **4/22** | **4/22** |
| Solved | `136b0064#0`, `247ef758#0`, `4c3d4a41#0`, **`b6f77b65#1`** | `136b0064#0`, `247ef758#0`, `4c3d4a41#0`, **`4c3d4a41#1`** |
| **Union (offline oracle)** | **5/22** | |

**Facts:**
- Poetiq's attempt_2 added nothing: with 1 expert there is usually a single program, so attempt_2 duplicated attempt_1.
- Only 2–3 of Poetiq's 6 train-verified programs were right on test. Three of the failures are tasks with only 2 training examples.
- **Poetiq makes Qwen think longer:** 10 of 12 round-1 requests hit the 77.8k cap, against 13 of 22 for DIRECT on the same tasks.
- **Why `b6f77b65#1` was lost under Poetiq:**
  - DIRECT's reasoning used a deletion / marker rule.
  - The Poetiq program assumed gravity / flip, and its BFS merged neighbouring cells of different colours (see `runs/poetiq_r4_eval12/poetiq_programs/b6f77b65_e0_i3.py`).
- **The full DIRECT thinking traces of 41 units** (`runs/direct_e8_49k_proxy48/full_thinking/`, PROXY48, 49k) show how DIRECT computes grids inside its thinking.
  - The E9 answers hold the last 1,500 thinking tokens plus the answer.
  - R4 did **not** save its thinking. R5 (`notebooks/E10_POETIQ_R4T_SAFE_SANDBOX_R5.ipynb`) saves it from now on.
- **Thinking length and success** (all our runs): about 50 attempts thought for more than 50k tokens; 3 of them solved. Every solve above 50k was a natural close; none came from a forced close at 77.8k. Forced-close grids are right only about 2 % of the time.

## Claude's analysis of the "one thinking, two answers" idea (verify or refute)

`analysis/measure_grid_answer_tokens.py` (results in its docstring) measures the cost of writing the test grids after the code:
- About 2 tokens per cell. Per task: median 972 tokens, p90 2,648, max 3,608.
- With the code answer added (median 2,581, max 6,562 tokens), **2 of 24** answers would exceed the 8,192 answer cap. So the grid must come after the code, or in a separate answer-only branch from the same prefix (prefix-cached).

**The real doubt is quality, not tokens.** Two cases:
1. The model computes the test grid inside its thinking. Then the grid **competes for the same capped thinking budget**, which is already exhausted in 10 of 12 round-1 requests.
2. The model does not. Then the grid is written without reasoning, and we measured such grids to be right only about 2 % of the time.

Also, one trace yields one rule. For `b6f77b65#1`, the Poetiq trace followed the wrong gravity rule, so a grid from the same trace would probably be wrong too. **The 5/22 union came from two independent traces.** It is unclear whether one trace can reproduce that complementarity. We want evidence either way.

**The data question that can be checked now:** in the DIRECT full-thinking files, does the model write out the full test-output grid inside its thinking? How often, and how early? (Poetiq thinking is not available yet.)

## Files

| Path | What |
|---|---|
| `notebooks/E10_POETIQ_R4T_SAFE_SANDBOX_R5.ipynb` | the current Poetiq notebook (R4 + full-thinking logging + safe sandbox). Cell 5: prompts, sandbox, verification, voting. Cell 7: the scheduler |
| `notebooks/Q27-XHIGH-v1_4_4_E10_BASE_86K.ipynb` | the DIRECT notebook (LB-26.94 code with an 86k budget). Cell 4: prompt. Cell 7: scheduler and answer parsing |
| `prompts/` | the Poetiq solver prompts 1–3 and feedback prompt (verbatim from poetiq-ai), the DIRECT prompt builder, and the forced-close stop text |
| `runs/poetiq_r4_eval12/` | iteration records (prompt / thinking / answer tokens, train score), every generated program, the final grids, the run report |
| `runs/direct_e9_86k_eval12/` | DIRECT 86k records and answers (thinking tail + answer) for the same 22 inputs |
| `runs/direct_e8_49k_proxy48/` | DIRECT 49k on PROXY48: records, attempts, and **41 full thinking traces** |
| `data/` | EVAL12 and PROXY48 challenges and solutions, and the full evaluation set (120 tasks) with solutions |
| `tokenizer/` | Qwen3.8-27B tokenizer and chat template, for exact token counts |
| `analysis/` | the R4 deep analysis, the validation of GPT's R4 audit, and the grid-token measurement script |
| `ledger/` | per-puzzle history of Qwen3.8-27B solves (our runs + ARC Prize's verified run at xhigh / medium / low / none) and the capability profile |

## Rules

- At least 49,152 thinking tokens per request.
- A valid `submission.json` always.
- Simple, one-change-at-a-time proposals.
- Do not run, push or submit anything.
- Do not project hidden-run time from evaluation runs (the hidden set is easier).
