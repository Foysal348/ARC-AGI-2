# E10 Poetiq R4 on GPU (EVAL12): analysis, fixes and solution research (2026-10-14)

**Goal:** Kaggle ARC Prize 2026 (ARC-AGI-2), offline. The model is Qwen3.8-27B-FP8 on vLLM 0.19.0 with 4×L4 and 12 h per hidden run. We target LB ~60; others have reached 60+ with this same model. Our best LB so far is 26.94 (v1.4.4 DIRECT, 49k thinking).

## What we ask (three parts)

### Part 1: problems in the code or the run, and fixes

- Find **real bugs or design defects** in R4 (`notebook/`, `source/`) that this run exposed: anything that wasted GPU time, lost a correct answer, or picked the wrong attempt.
- Propose concrete, minimal fixes, each with the evidence from the files that shows the defect.
- Verify every claim against the outputs; do not assume.

### Part 2: for every unsolved input, why it failed

For each of the 18 unsolved inputs:
- What rule the program implements, and where it is wrong. Use `outputs/q27x/poetiq_programs/*.py`, the generated grids and `data/eval12_solutions.json`.
- Whether the failure is:
  - wrong rule;
  - right rule, wrong execution;
  - a rule that fits the training pairs but is under-determined by them;
  - out of iterations or time;
  - an invalid test output.
- Compare with the DIRECT 86k answers on the same inputs (`context/E9_direct_86k_same_tasks/`).
- Explain especially the **6 train-verified programs, of which only 2–3 are right on test**:
  - `247ef758#1`: 1 cell wrong, the same cell as DIRECT;
  - `2b83f449`: 4 cells;
  - `898e7135`: 18 cells;
  - `6e4f6532`: 98 and 14 cells.
- Also explain why `b6f77b65#1` (solved by DIRECT) was lost.

### Part 3: research solutions (GitHub, the web, Hugging Face, Kaggle; many searches)

Search widely for methods that fix our failure classes **with a ~27B open model, offline, on 4×L4 within 12 h for 240 tasks**. For each method, give the source link, what it does, the evidence (its numbers), and how it would plug into this notebook. Topics:

- **Selecting among train-consistent programs** when the training pairs under-determine the rule (2 training examples), for example:
  - simplicity or MDL priors;
  - augmentation consistency (rotations, flips, colour permutations);
  - test-input sanity checks;
  - transfer and generalisation judges;
  - voting across several programs.
- Program-synthesis loops on **small or open models** (Poetiq's GPT-OSS configs, Imbue's code evolution with Kimi, BARC, ARChitects, NVARC, the ARC Prize 2025 winners) and their sample, iteration and token budgets.
- **Making the thinking shorter.** Under this prompt Qwen thinks longer than in DIRECT mode: 10 of 12 tasks hit the 77.8k cap in round 1. Cover prompting, budget forcing, early-exit and "think less, verify more" approaches.
- **Combining DIRECT grids and programs** (hybrid attempt_1 / attempt_2; agreement as a signal), and test-time training or fine-tuning that is feasible on Kaggle.
- Top Kaggle notebooks and discussions for ARC Prize 2026 that use Qwen3.8-27B.

Rank the solutions by expected gain per GPU hour for **our** setup. Mark anything unverified as a hypothesis.

## Where to apply fixes

- **Apply all code fixes on top of R4T:** `notebook/Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R4T.ipynb`, sha `1f8fd03e…`, built by `source/build_e10_r4t.py`.
- **R4T = R4 + logging of the full thinking per iteration** to `q27x/poetiq_thinking/`. It does not change generation or scoring.
- **Keep that logging in every new version.**
- Deliver the new notebook with its sha256 and a patch limited to the cells you change. Give each fix a one-line reason that points to this run's evidence.

## Run facts

| | |
|---|---|
| Kaggle | `foysalemonshanto/q27-xhigh-e10-poetiq-r4` v1 |
| Notebook | `notebook/…POETIQ_PROGRAM_SOLVER_R4.ipynb`, sha `596bf694…`. Base: `…E10_BASE_86K.ipynb` (v1.4.4 DIRECT + 86k) |
| Method | Poetiq (poetiq-ai/poetiq-arc-agi-solver): Qwen writes `transform()`; a sandbox runs it on the training pairs; diff feedback; refinement with up to 10 iterations; 1 expert; temperature 1.0; 77,824 thinking + 8,192 answer |
| Set | EVAL12: 12 tasks / 22 test inputs; 3 h; at most 12 of 16 slots used |
| **Score** | **4/22 (18.18 %)**: `136b0064#0`, `247ef758#0`, `4c3d4a41#0`, `4c3d4a41#1` |
| DIRECT 86k on the same set (E9) | 4/22: the same first three plus `b6f77b65#1`. **Union 5/22** |
| Iterations | 27 total. Round 1: 10 of 12 forced at 77.8k. Round 3 was cut by the 147-min drain |
| Engine | 0 errors, 0 preemptions, KV max 79 %, 224 tok/s, 1.85 M tokens |

**Limitation:** R4 saves the **programs only, not the thinking text.** The DIRECT answers in `context/E9_direct_86k_same_tasks/answers/` hold the last 1,500 thinking tokens plus the answer for the same inputs.

## Files

| Path | What |
|---|---|
| `outputs/q27x/poetiq_run_report.json` | score, per-unit hits, expert states, scheduler stats |
| `outputs/q27x/poetiq_iterations.jsonl` | per iteration: mode, prompt / think / answer tokens, seconds, train score |
| `outputs/q27x/poetiq_programs/*.py` | every generated program (task, expert, iteration) |
| `outputs/q27x/poetiq_eval_generated_submission.json` | the final attempt_1 / attempt_2 grids per eval input |
| `outputs/q27x/poetiq_vllm_metrics.json`, `outputs/*.log` | engine metrics and the Kaggle log |
| `data/eval12_challenges.json`, `data/eval12_solutions.json` | tasks and solutions, for scoring |
| `context/GPU EVAL12 Poetiq R4 deep analysis (2026-10-14).md` | Claude's analysis of this run; verify or refute it |
| `context/R4 audit …md`, `context/README.md` | the pre-run audit, the Kaggle sandbox probe and the experiment README |
| `context/E9_direct_86k_same_tasks/` | the DIRECT 86k run on the same 22 inputs (records, answers, post-mortem) |
| `context/ledger/` | per-puzzle history of Qwen3.8-27B solves (our runs + ARC Prize's verified run) and the capability profile |

## User rules (fixed)

- Never budget below 49,152 thinking tokens per request without approval.
- A valid `submission.json` must always be produced.
- Simple changes, one at a time.
- Nothing is run or pushed by you.
- The hidden test set is easier than evaluation, so do not project hidden-run time from EVAL12.
