# E10 ORDER2: a full 120-task evaluation run of Qwen3.8-27B. Find the weaknesses, then propose fixes (2026-10-14)

**What we ask:**
- A deep, evidence-based analysis of this run.
  1. Find the weaknesses that cost score or GPU time.
  2. Quantify each one from the files.
  3. Propose fixes ranked by expected Kaggle LB gain per GPU-hour.
- **Verify everything against the files.** That includes our own analysis in `analysis/`, which may contain mistakes.
- Do not trust a claim (ours or yours) that the data cannot show.

## 1. Competition and constraints (fixed; the user's decisions)

**Competition:**
- Kaggle ARC Prize 2026 (ARC-AGI-2).
- One offline notebook, **12 h on 4×L4**, no internet. Deadline 2026-11-02.
- The hidden test is about 240 tasks / 259 test inputs.
- Each test input gets 2 attempts (`attempt_1`, `attempt_2`).
- **Our best LB is 27.92**, and our team rank is 109.
- **Goal: LB 80+ with this same model.** The user considers this achievable through method: harness, prompting, search, verification, selection and speed. Do not propose fine-tuning or a different model as the answer.

**Model and engine:**
- Qwen3.8-27B-FP8 only.
- vLLM 0.19.0, TP4, fp8 KV, MTP×3, prefix caching, FlashInfer.
- Docker image pinned.

**Rules:**
- **Never below 49,152 thinking tokens per request.**
- A valid `submission.json` always.
- One change at a time; simple code.

**Platform facts:**
- About 33 h submission queue, 1 pending submission per team.
- The hidden puzzles are easier than evaluation. **Do not project hidden run time or coverage from evaluation numbers.**
- An E9 86k run (direct 77.8k + 8k, file order, plus resample in leftover time) is now pending on the LB. It is the direct hidden-run test of the 86k budget.

## 2. What this run is

| | |
|---|---|
| Kaggle kernel | `hosen42/q27-xhigh-e10-order2` (private). The pulled code cells are identical to `notebook/Q27-XHIGH-v1_4_4_E10_86K_ORDER2.ipynb` (sha256 `b62ae567aa4d8af2…`) |
| Builders | `notebook/build/` (`build_e10_base.py` → `build_e10_order2.py`; sources not included) |
| Method | Direct reasoning (no code): one prompt per test input |
| Prompt | russcore's free-form prompt (cell 3): train pairs + final input; the answer is a fenced `grid` block |
| Thinking budget | 77,824 thinking + 8,192 answer (`MAX_MODEL_LEN` 106,496) |
| Sampling | temperature 0.6, top_p 0.95, top_k 20 |
| Concurrency | 16 sequences; continuous admission |
| Close | Host-forced close at the cap (stop text + `</think>`, then the answer phase) |
| `attempt_2` | **Just the test input, or its mirror** (`make_attempts`, cell 4) — in practice no real second attempt |
| ORDER2 order | `order_score = [n_train ≥ 3] + [n_test == 1]`, sorted 2 → 1 → 0 with file order inside a score |
| Profile | `EVAL120`: all 120 public-evaluation tasks / 172 test inputs, on the hidden-run clock (admit until 694 min, drain 696, hard stop 704) |

## 3. Headline results (verify)

- **Score:**
  - 58/172 test inputs = **33.72 %** (`run_report.json` `eval_score_pct`);
  - task-normalised **36.94 %** (44.33/120);
  - 37 tasks fully solved, 15 partly.
- **Finish:** all 172 finished by 534 min. 0 preemptions; KV max 87.3 %; throughput 333 tok/s; MTP acceptance 0.669.
- **How thinking ended:**
  - natural close 97 units → 52 hits;
  - forced close at the cap 74 → 5 hits;
  - continue 1 → 1 hit.
- **Late solves:** 19 of the 58 hits needed more than 49,152 thinking tokens.
- **Comparison:** ARC Prize's own Qwen3.8-27B xhigh run solved 40/120 tasks (`ledger/arcprize_qwen38_27b_public_eval.json`). Overlap with us: 26 both, 26 only us, 14 only them.
- **Oracle tiers** (`ledger/oracle_order_120.csv`, all our 12 runs + ARC Prize at all levels):
  - **72/120 tasks have been solved by the Qwen3.8-27B methods tried so far.** 48 have not been solved yet.
  - This is a limit of our methods, not of the model. On the live leaderboard (2026-10-15), other teams score 88.06 / 83.89 / 80.56 / 77.08, and the user's position is that this model can reach 80+. So treat tier D as the largest headroom, not as unsolvable.
  - Tiers B + C (46 tasks) are the cheapest subset for detecting a gain.

## 4. Files

| Path | Content |
|---|---|
| `run_outputs/q27x/records.jsonl` | One row per test input: `mode`, `end_reason`, `prompt_tokens`, `p1_tokens` (thinking phase), `p2_tokens` (answer phase after a forced close), `think_tokens`, `sec`, `has_grid`, `src`, `score{hit,pass1}`, `t_admit_min`, `t_done_min`, `order_score`, `admit_rank` |
| `run_outputs/q27x/answers/<task>_q<i>.txt` | **The last ~3–4k characters of the thinking** (`### THINKING TAIL`) and the full answer (`### ANSWER`). **Full traces were not saved.** |
| `run_outputs/q27x/run_report.json` | Config, engine, scheduler, modes, projections, MTP and prefix-cache stats |
| `run_outputs/q27x/vllm_metrics.json` | vLLM Prometheus snapshot every 2 min (KV usage, running/waiting, preemptions, spec-decode counters) |
| `run_outputs/q27-xhigh-e10-order2.log` | Kaggle log (JSON stream lines) |
| `run_outputs/submission.json` | The written submission |
| `data/` | Evaluation challenges + solutions (ground truth) |
| `ledger/` | Per-task/unit history over all our runs (`ledger_units.csv`, `ledger_all120.csv`), the oracle order, solved units sorted by token cost, the ARC Prize solve list, the capability profile, the README and scripts. The scripts point to local paths; their outputs are included |
| `analysis/` | Our analysis (`GPU EVAL120 ORDER2 86k analysis (2026-10-14).md`), the per-task admission/outcome CSV, and scripts `an.py` and `ordering.py` |
| `context/` | Poetiq R4 EVAL12 analysis (code-synthesis variant, 4/22); the "one thinking, two attempts" validation (direct traces write their grid at ~89 % of the trace; 0 rescues in 41 traces); the Experiment 10 README |

## 5. Weakness candidates we already see (confirm, refute or add)

1. **`attempt_2` is wasted.** It is the test input or its mirror, which never hits. Every Kaggle test input allows 2 attempts.
2. **Forced closes: 74/172 units (43 %) hit the 77.8k cap, and only 5 of them hit.** They also take the most GPU time (median 61 min vs 34 min for natural closes).
3. **Confident wrong early closes.** Example: 6 tasks that ARC Prize solved closed naturally at 26–50k with a wrong answer: `6e453dd6`, `edb79dae`, `dbff022c`, `7c66cb00`, `db695cfb`, `d35bdbdc`. Of all natural closes at or below 49k, 15/54 were wrong.
4. **Repetition loops:** 4 forced closes end in a pure loop (`0934a4d8#0`, `6ffbe589#0`, `800d221b#0`, `3a25b0d8#0`). We cannot see how many tokens the loop wasted.
5. **No parseable grid after a forced close:** 7/74.
6. **Multi-test tasks are weak:** 101 units; hit 26 % vs 45 % for single-test.
   - 15 tasks were half solved: one input right, the other wrong, in independent traces.
   - Each test input re-derives the rule from scratch.
7. **Same-rule failures on the training data are never checked.** The model's answer is never verified against the training pairs. This is direct reasoning, with no program.
8. **Scheduling:** a 44-min tail with 7 of 16 slots active. Mean KV is 68.6 % (max 87.3 %), so there is headroom that a KV-aware admission might use.
9. **Run-to-run variance is large** (our run vs ARC Prize vs our earlier runs), which suggests independent second traces could add hits. But E9's resample (second traces on leftover time) gave 0 gain on EVAL12.

## 6. Questions

1. **Taxonomy of failures.** Classify every one of the 114 missed units from the answers, the thinking tails and the ground truth, with counts and unit ids. Categories:
   - wrong rule;
   - right rule with an execution or transcription error;
   - wrong output size;
   - partial/near miss (cell accuracy);
   - loop;
   - no grid;
   - ran out of budget.
2. **Near misses.** How many wrong answers are within a few cells of the truth, or have the right shape? Those are where a verifier or a second attempt helps most.
3. **The best use of `attempt_2` within 12 h on 4×L4, with ≥ 49,152 thinking per request.** Use the evidence here and in `context/`. Options include:
   - a second independent trace on selected units;
   - a code sibling after the direct trace (program + check on the training pairs);
   - a self-check pass;
   - an augmentation or transform view;
   - something else.

   Estimate the gain from the data, not from intuition.
4. **Multi-test tasks.** Is there evidence that one trace per task (all test inputs together) would do better or cost less?
5. **Forced closes.** What do the thinking tails show the model doing at the cap: still exploring, verifying, looping, or writing the grid? Is there a cheap signal to stop a doomed trace early and reuse the slot?
6. **Hidden-run time.** Hidden puzzles are easier, so do not project from eval. Instead, say which measurement from the E9 LB result would settle the questions above.

## 7. Deliverable format

A ranked table with these columns:

| weakness | evidence (file + unit ids + counts) | proposed fix | expected gain (units / LB points, with the reasoning) | GPU-time cost | risk | how to test |

Rules for the table:
- Tests should be cheap: oracle tiers B + C (46 tasks) or EVAL12.
- Mark each claim **verified from files** or **hypothesis**.
- No fix may lower thinking below 49,152 tokens per request.
