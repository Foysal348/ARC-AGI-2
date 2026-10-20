# E10 Poetiq R5: full 120-task evaluation run of Qwen3.8-27B program synthesis. Find the weaknesses, then propose fixes (2026-10-15)

**What we ask:**
- A deep, evidence-based analysis of this run, and a comparison with our DIRECT run on the same 120 tasks.
  1. Find every weakness that costs score or GPU time.
  2. Quantify each one from the files.
  3. Propose fixes ranked by expected Kaggle LB gain per GPU-hour.
- **Verify everything against the files**, including our analysis in `analysis/`, which may contain mistakes.
- Mark each claim **verified from files** or **hypothesis**.
- Do not write or change notebook code.

## 1. Competition, goal and fixed constraints

**Competition:**
- Kaggle ARC Prize 2026 (ARC-AGI-2): one offline notebook, **12 h on 4×L4**, no internet. Deadline 2026-11-02.
- The hidden test is about **240 tasks / 259 test inputs**.
- 2 attempts per test input.
- Score = correct test inputs; a task with 2 inputs gives half credit for one.

**Goal and position:**
- **LB 80+ with this same model (Qwen3.8-27B-FP8).** The user considers this achievable through method: harness, prompting, search, verification, selection and speed.
- Other teams are at 88 / 84 / 81 / 77 on the live LB.
- **Do not propose fine-tuning or a different model.**
- Our best LB is 27.92, with direct reasoning at 49k.
- An E9 86k DIRECT run is pending on the LB.

**Engine:** vLLM 0.19.0, TP4, fp8 KV, MTP×3, prefix caching, 16 sequences, pinned Docker image.

**Rules (fixed, the user's decisions):**
- **Never below 49,152 thinking tokens per request.** This run shows Poetiq needs the full **77,824 per iteration**: 70 % of its solves came after 49,152. Keep that.
- A valid `submission.json` always.
- One change at a time; simple code.
- **The hidden puzzles are easier than evaluation. Do not project hidden run time or coverage from eval timings.** Use ratios between methods, or the E9 hidden timing once it is known.

## 2. What this run is

| | |
|---|---|
| Kaggle kernel | `hosen42/q27-xhigh-e10-poetiq-r5-eval120` (private). The pulled code is identical to `notebook/Q27-XHIGH-v1_4_4_E10_86K_POETIQ_R5_EVAL120.ipynb` (sha256 `498109b72382d3ab…`) |
| Method | Poetiq-style program synthesis, adapted from `poetiq-ai/poetiq-arc-agi-solver` |
| One request per task (iteration) | The model reasons, then writes a Python `transform(grid)` |
| Train check | The sandbox runs the program on the training inputs and compares with the known training outputs |
| Pass | All train pairs exact → the task is "train-verified", and the program's outputs on the test inputs become the answers |
| Fail | The feedback (wrong cells as prediction/correct, accuracy, errors) goes into the next iteration's prompt |
| Settings | 1 expert, up to 10 iterations, solver temperature 1.0, 77,824 thinking + 8,192 answer per iteration |
| Sandbox | UID drop + seccomp + rlimits + 5 s timeout (no unshare on Kaggle) |
| Queue | FIFO in alphabetical task order (**no ordering**). A failed iteration goes to the back of the queue |
| attempt_1 / attempt_2 | Voted best program; a second distinct program from the history |
| Clock | `EVAL120` profile on the hidden-run clock: admit until 694 min, drain 696, hard stop 704 |

## 3. Headline results (verify)

| | Poetiq R5 | DIRECT ORDER2 (same 120 tasks) | Union of both |
|---|---|---|---|
| Test inputs | **78/172 = 45.35 %** | 58/172 = 33.72 % | 92/172 |
| Task-normalised | **48.61 %** | 36.94 % | 56.81 % |
| Wall / tokens / speed | 691 min / 11.65 M / 281 tok/s | 534 min / 10.45 M / 333 tok/s | — |

**Iterations:**
- Iteration 1: 120 requests, 67 forced at the cap, 55 train-verified, median 70.9 min each.
- Iteration 2: 52 requests, 37 forced, 16 verified.
- **Round 1 ran until about 540 min.** 13 failed tasks never got iteration 2; no task got iteration 3.

**Train check precision:** 71 verified tasks → 49 fully right on test, 7 partly, 15 wrong.

**Complementarity:**
- On tasks verified at iteration 1, DIRECT adds 1 unit.
- On tasks not verified, DIRECT adds 11+.
- The policy "Poetiq if verified, else attempt_2 = DIRECT" scores **55.14 %** (our computation).

**New solves:** 13 tasks that no earlier method had ever solved (`ledger/oracle_order_120_before_R5.csv`, tier D).

**Our analysis:** `analysis/GPU EVAL120 Poetiq R5 analysis (2026-10-15).md`. Please re-verify it.

## 4. Files

| Path | Content |
|---|---|
| `run_outputs/q27x/poetiq_iterations.jsonl` | One row per iteration: mode, p1/p2 tokens, prompt tokens, sec, train score, train_passed |
| `run_outputs/q27x/poetiq_thinking/<task>_e0_i<k>.txt` | **The full generation of every iteration** (172 files): thinking, forced-close stop text, answer and code |
| `run_outputs/q27x/poetiq_programs/<task>_e0_i<k>.py` | The extracted program of every iteration (172) |
| `run_outputs/q27x/poetiq_run_report.json` | Config, engine, scheduler, expert states, per-unit eval records (hit / pass1 / pass2) |
| `run_outputs/q27x/poetiq_eval_generated_submission.json` | Generated attempts for the eval tasks |
| `run_outputs/q27x/poetiq_vllm_metrics.json` | vLLM metrics snapshots |
| `run_outputs/*.log` | Kaggle log (JSON lines). It has `[E10] <task> expert=0 iter=k score=… passed=… status=…` lines and a progress line every 2 min |
| `comparison_direct_order2/` | The DIRECT ORDER2 run on the same tasks: `records.jsonl` and `answers/` (thinking tails + answers) |
| `data/` | Evaluation challenges + solutions (ground truth) |
| `ledger/` | All our runs per task/unit (now including R5), oracle tiers (current, and before R5), solved-by-tokens, ARC Prize solve list |
| `analysis/` | Our R5 analysis + script; the ORDER2 analysis |
| `context/` | Our validation of your previous (ORDER2) audit. It includes a **measured same-method second-trace rescue rate of only 3–6 %**, so diversity of method matters. Also: the Poetiq R4 EVAL12 analysis, the "one thinking, two attempts" validation, the Experiment 10 README |

**Safety:** `poetiq_programs/` and the code inside `poetiq_thinking/` are model-generated. Read them; do not run them outside a sandbox.

## 5. Questions

1. **Failure taxonomy of the 94 missed test inputs.** Use the iteration rows, programs, thinking and ground truth. Categories:
   - no train pass after 1 or after 2 iterations;
   - program crash or timeout;
   - wrong output shape;
   - near miss (cells);
   - verified on train but wrong on test.

   For each category, give counts and unit ids.
2. **The 15 verified-but-wrong tasks.**
   - Why did a program that reproduces every training pair fail on test? Candidates: hard-coding of training cases, an under-determined rule, a test-only feature (new colour, size or object count), a different branch.
   - Is there a gold-free check that would have flagged them? For example: test-output sanity against the training outputs (shape rule, palette), consistency under symmetric transforms of the input, or agreement with an alternative program or DIRECT.
3. **Forced closes.** 67 of 120 first iterations hit the 77,824 cap, and 13 of those still passed train.
   - What is the model doing near the cap?
   - Does it write the code late?
   - Could a prompt change make it commit to a program earlier without lowering the cap?
4. **Iteration 2.** What in the feedback led to the 16 fixes? What failed in the other 36? Is iteration 2 the best use of a slot, compared with a DIRECT trace or a different-view Poetiq request on the same task?
5. **Hybrid and schedule for the hidden run (about 240 tasks).**
   - Design the best schedule within 12 h: Poetiq round 1 → gate on train verification → a DIRECT second attempt, Poetiq iteration 2, or a different view.
   - Say what must be measured on the hidden run (E9 timing) to size it.
   - The order matters: R5 used alphabetical order and leaves unadmitted tasks at the identity fallback.
6. **attempt_2 in Poetiq gave 1 extra hit.** What is the best gold-free second candidate per test input?
7. **Throughput:** 281 tok/s vs 333 for DIRECT. Why: longer prompts, more forced closes, sandbox waits, KV? Is there a cheap fix?
8. **Anything else in the thinking traces** that shows systematic waste or systematic mistakes (e.g. mis-reading grids, coordinate conventions, colour confusions).

## 6. Deliverable

A ranked table:

| weakness | evidence (file + unit ids + counts) | proposed fix | expected gain (units / LB points, with reasoning) | GPU-time cost | risk | how to test |

**Rules for the table:**
- Tests should be cheap: EVAL12, or oracle tiers B + C.
- No fix may lower thinking below 49,152 per request; Poetiq iterations keep 77,824.
- Mark each claim **verified from files** or **hypothesis**.
