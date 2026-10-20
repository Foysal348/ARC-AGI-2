# Puzzle Capability Ledger: what Qwen3.8-27B solves on ARC-AGI-2 evaluation (started 2026-10-13)

**Purpose:** record, for every one of the 120 public evaluation puzzles, which runs of Qwen3.8-27B solved it (ours and others'), together with the user's EDA category. The goal is to order puzzles **by this model's real capability** near the end of the competition, instead of NVARC's size and colour heuristic.

## Files

| File | What |
|---|---|
| `build_ledger.py` | Rebuilds everything. To add a run, append one line to `RUNS` |
| `arcprize_qwen38_27b_public_eval.json` | ARC Prize's official Qwen3.8-27B run: pass/fail per task at XHigh / Medium / Low / None (source below) |
| `ledger_all120.csv` | All 120 tasks: EDA tier, fine and tags; in PROXY48 or not; ARC Prize result per level; our attempts and hits |
| `ledger_units.csv` | Every test input we ran: runs attempted and solved, think tokens, mode (forced_close / natural) |
| `ledger_tasks.csv` | The same, aggregated per task |
| `summary.json` | Per-run hits, and the ARC Prize XHigh solve rate per EDA category |

**EDA categories:** `puzzle_features()`, copied verbatim from `resources/Eval Subset (EDA mirror)/ARC2_Advanced_EDA_v8.ipynb` cell 35. `tier` and `fine` are NVARC's `get_puzzle_priority`.

## Sources

1. **ARC Prize verified run** (others): https://arcprize.org/results/alibaba-qwen3-8-27b, retrieved 2026-10-13.
   - The table "ARC-AGI-2 Public Eval, 120 tasks" gives one run per reasoning level, served via Baseten.
   - Token cap: not stated. From the cost ($0.4469 per task, $2.55 per M output) the output is about 173k tokens per task, which is about **60k tokens per generation on average** (2 attempts × 1.43 test inputs). That is a mean, not a cap. The earlier "about 85k" estimate was wrong.
   - Overall semi-private scores: XHigh 42.4 %, Low 22.8 %, Medium 13.2 %, None 1.5 %.
   - Public eval, task level: **XHigh 40/120**, Medium 16, Low 19, None 3.
   - **Solved at any level: 51/120.**
2. **Our GPU evaluation runs:** EVAL12, 12 tasks / 22 test inputs.

| Run | Method | Hits /22 |
|---|---|---|
| run3 v1.3 | direct + code check | 1 |
| run4 v1.4 | direct, xhigh 49k + 8k | 2 |
| run6 v1.4.1 | direct, xhigh 49k + 8k | 2 |
| run7 v1.4.4 (LB 26.94 code) | direct, xhigh 49k + 8k | 3 |
| dual v2 | direct, attempt_2 = second answer | 1 |
| run5 R6 | program-synthesis loop | 2 |
| agent v1.8.3 | REPL agent only | 0 |
| agent v1.9.1 | long-think program only | 1 |

3. **Pending:** E8 R5 PROXY48 (`q27-xhigh-e8` v1, 48 tasks / 56 inputs). Add it to `RUNS` when its outputs are downloaded.
   - Partial log, 13 Oct: `36a08778#0`, `1818057f#0`, `e376de54#0`, `31f7f899#0` hit.
   - `36a08778` and `1818057f` were **not** solved by ARC Prize at XHigh.
4. **Other Kaggle teams:** no per-puzzle results are public. Discussion 748082 reports only LB totals, not puzzle IDs.

## Key findings (so far)

### 1. NVARC's easy-first order does not match Qwen3.8's capability

ARC Prize XHigh, 120 tasks:

| EDA category | Tasks | Solved | % |
|---|---|---|---|
| all | 120 | 40 | 33.3 |
| tier1 | 65 | 21 | 32.3 |
| tier2 | 51 | 17 | 33.3 |
| tier3 | 4 | 2 | 50.0 |
| **easy_static** (NVARC's easiest) | 16 | 4 | **25.0** |
| easy (tier 1, fine ≤ 0.22) | 26 | 7 | 26.9 |
| hard_tier1 | 10 | 4 | 40.0 |
| **dynamic** (output size changes) | 39 | 16 | **41.0** |
| high_ratio (size ratio > 5) | 20 | 9 | 45.0 |
| many_colors (8 or more) | 59 | 23 | 39.0 |
| large (input of 500 cells or more) | 30 | 11 | 36.7 |
| rare (colour 6 or 9) | 96 | 34 | 35.4 |
| sparse_bg | 21 | 8 | 38.1 |
| dense_non_bg | 80 | 26 | 32.5 |
| **multi_query** (more than one test input) | 49 | 13 | **26.5** |
| **low_train** (2 or fewer examples) | 34 | 7 | **20.6** |

- **fine score versus solved:** AUC = **0.435**, below 0.5. NVARC's "fine" score does not predict what Qwen3.8 solves; if anything it is slightly reversed.
- **Fine quartiles** (easy → hard) solve 9, 10, 9 and 12 of 30.
- **Weakest for Qwen:** few training examples (low_train 20.6 %) and multi-test tasks (26.5 %). By number of test inputs: 1 input 27/71, 2 inputs 11/46, 3 inputs 2/3.
- **Stronger for Qwen:** dynamic and high-ratio tasks, the ones NVARC puts last.

### 2. Run-to-run variance is large; one run per level is a weak label

- 11 tasks failed at XHigh but passed at Medium or Low: `1818057f 291dc1e1 38007db0 4c3d4a41 53fb4810 58490d8a 6e453dd6 8f3a5a89 a395ee82 b10624e5 db695cfb`.
- Our E8 run solved `36a08778` and `1818057f` at XHigh, although ARC Prize failed them at XHigh.
- Our EVAL12 runs of identical code scored 2, 2, 3 and 1.
- **So the capability label should combine several runs:** solved at any level (51/120) plus our hits. Do not rely on a single XHigh pass/fail.

### 3. Our 49k + 8k budget versus ARC Prize's budget (mean about 60k per generation, cap unknown) (EVAL12)

| Task | ARC Prize XHigh | Ours | Note |
|---|---|---|---|
| `136b0064` | ✓ | 6/8 | |
| `20a9e565` | ✓ | **0/16** | all forced_close at 49k |
| `898e7135` | ✓ | **0/8** | forced_close |
| `247ef758` | ✗ | 4/16 | |
| `4c3d4a41` | ✗ | 2/16 | |

- Tasks ARC Prize solved but we never did were all cut at 49k thinking. This is a hint that the token cap costs some solves on hard tasks.
- Our EVAL12 runs (88 units): **natural closes 7/16 hit (44 %)**, **forced_close at 49k 1/72 hit (1.4 %)**. Natural think lengths: p50 38.7k, p90 44.8k; 5 of 16 closed between 40k and 49k.
- This is not a recommendation to change the budget (user rule: no change without approval). The hidden set is easier, and more tokens means fewer puzzles fit in 12 h.

## ARC Prize XHigh-solved tasks (40)

`0934a4d8` `136b0064` `16de56c4` `1ae2feb7` `20270e3b` `20a9e565` `2ba387bc` `2d0172a1` `31f7f899` `35ab12c3` `3dc255db` `409aa875` `4a21e3da` `581f7754` `58f5dbd5` `65b59efc` `67e490f4` `7491f3cf` `7666fa5d` `7b3084d4` `7c66cb00` `80a900e0` `898e7135` `8e5c0c38` `8f215267` `981571dc` `a6f40cea` `b5ca7ac4` `b99e7126` `bf45cf4b` `cb2d8a2c` `d35bdbdc` `d59b0160` `d8e07eb2` `db0c5428` `dbff022c` `e376de54` `e8686506` `edb79dae` `f931b4a8`

**PROXY48:**
- 16 of its 48 tasks are XHigh-solved, and 23 are solved at any level. That is a rough expectation for the E8 run.
- Its run is capped at 49k + 8k with a 3-hour deadline, so expect fewer.

## Idea for the end-game ordering (not implemented)

1. **Capability score per hidden task,** predicted from features, because the hidden tasks themselves have no labels:
   - fit "solved by Qwen (any run)" on the 120 evaluation tasks plus our runs, against the EDA features (low_train, multi_query, dynamic, n_colors, size, n_train);
   - training-set tasks (1,000) are much easier and could add labels if we ever run them.
2. **Order hidden tasks by predicted P(solve) / expected tokens.** Likely-solvable and short tasks go first; low_train and multi_query tasks go later.
3. **Validate on PROXY48:** compare NVARC order against capability order on the same tasks and deadline.

**Caveat:** 120 labelled tasks with one or a few runs each is a small, noisy sample. Several category differences above are within a few tasks, so treat them as hypotheses and update them as more runs are added.
