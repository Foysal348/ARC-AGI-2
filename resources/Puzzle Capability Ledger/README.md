# Puzzle Capability Ledger: what Qwen3.8-27B solves on ARC-AGI-2 evaluation (started 2026-10-13; updated 2026-10-14 with the E10 ORDER2 EVAL120 run)

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
2. **Our GPU evaluation runs:** 9 runs, 45 tasks / 61 test inputs seen.

| Run | Set | Method | Hits |
|---|---|---|---|
| run3 v1.3 | EVAL12 | direct + code check | 1/22 |
| run4 v1.4 | EVAL12 | direct, xhigh 49k + 8k | 2/22 |
| run6 v1.4.1 | EVAL12 | direct, xhigh 49k + 8k | 2/22 |
| run7 v1.4.4 (LB 26.94 code) | EVAL12 | direct, xhigh 49k + 8k | 3/22 |
| dual v2 | EVAL12 | direct, attempt_2 = second answer | 1/22 |
| run5 R6 | EVAL12 | program-synthesis loop | 2/22 |
| agent v1.8.3 | EVAL12 | REPL agent only | 0/22 |
| agent v1.9.1 | EVAL12 | long-think program only | 1/22 |
| **E8 R5** (`q27-xhigh-e8` v1) | **PROXY48** | direct, xhigh 49k + 8k, Codex order; the engine died at minute 124 | **10/41 finished** (10/56 planned) |
| E9 EVAL12 (primary units) | EVAL12 | direct, xhigh 77.8k + 8k (+ resample, 0 gain) | 4/22 |
| E10 Poetiq R4 | EVAL12 | Poetiq program synthesis | 4/22 |
| **E10 ORDER2** (`hosen42/q27-xhigh-e10-order2`, 2026-10-14) | **EVAL120 (all)** | direct, xhigh 77.8k + 8k, ORDER2 order | **58/172; task-normalised 36.94 %** |

**Coverage since the E10 ORDER2 run: all 120 tasks / 172 test inputs have at least one attempt** (`summary.json`). Analysis: `experiments/Experiment 10 …/Experiment Results/Analysis/GPU EVAL120 ORDER2 86k analysis (2026-10-14).md`.

3. **E8 R5 PROXY48 (added):**
   - 41 of 56 units finished; 15 were lost when the engine died. These PROXY48 tasks therefore have **no completed attempt yet**: `0934a4d8`, `20a9e565`×2, `4c7dc4dd`×2, `89565ca0`, `291dc1e1`, `16b78196`, `2c181942`, `4c416de3`, `64efde09`, `7c66cb00`, `58490d8a`, `80a900e0`, `8b9c3697`. `20a9e565` and `4c7dc4dd` were attempted in EVAL12 runs.
   - Hits: `1818057f`, `36a08778#0`, `e376de54`, `31f7f899`, `409aa875`, `7491f3cf`, `b99e7126`, `5961cc34`, `c4d067a0`, `136b0064`.
4. **Other Kaggle teams:** no per-puzzle results are public. Discussion 748082 reports only LB totals, not puzzle IDs.

## Capability profile: what Qwen3.8-27B is good and bad at (2026-10-13)

Sources:
- **ARC Prize:** one run per effort level, all 120 tasks; the cleanest data.
- **Oracle:** ARC Prize at any level, or ours.
- **Our runs:** 151 unit attempts over 45 tasks. They are biased: the EVAL12 tasks were run many times.

| Puzzle type | Tasks | ARC Prize XHigh | ARC Prize any level | Oracle | Our hit % (units) | Our natural-close % |
|---|---|---|---|---|---|---|
| **Output smaller than input (crop / extract / summarise)** | 31 | **41.9** | 51.6 | 51.6 | 19.4 (31) | 19.4 |
| **dynamic** (output size changes) | 39 | **41.0** | 48.7 | 48.7 | 14.0 (43) | 14.0 |
| **high_ratio** (size ratio > 5) | 20 | **45.0** | 50.0 | 50.0 | 0 (20) | 5.0 |
| **hard_tier1** (NVARC's "hard") | 10 | 40.0 | 50.0 | 50.0 | 37.5 (16) | 31.2 |
| **3 train examples** | 62 | 38.7 | 46.8 | 50.0 | **30.6** (49) | **46.9** |
| **1 test input** | 71 | 38.0 | 50.7 | 53.5 | **31.8** (44) | **43.2** |
| medium input (150–400 cells) | 48 | 33.3 | 45.8 | 50.0 | **25.8** (66) | **43.9** |
| output same size as input | 81 | 29.6 | 39.5 | 45.7 | 14.8 (108) | 31.5 |
| same size, few cells change (< 10 %) | 40 | 25.0 | 37.5 | 40.0 | 17.5 (40) | 27.5 |
| **easy_static** (NVARC's easiest) | 16 | **25.0** | 56.2 | 62.5 | 11.5 (26) | 26.9 |
| **2+ test inputs (multi_query)** | 49 | **26.5** | 30.6 | 36.7 | **7.5** (107) | 19.6 |
| **≤ 2 train examples (low_train)** | 34 | **20.6** | 32.4 | 32.4 | **6.4** (47) | 23.4 |
| large input (≥ 400 cells) | 53 | 32.1 | 39.6 | 43.4 | 7.5 (40) | 12.5 |
| same size, many cells change (≥ 30 %) | 9 | 11.1 | 22.2 | 22.2 | 0 (3) | 33.3 |
| output larger (tile / expand) | 3 | 0 | 0 | 0 | 0 (2) | 0 |
| **all** | 120 | 33.3 | 42.5 | 46.7 | 14.6 (151) | 26.5 |

**Strong:**
- extraction or crop tasks where the output is smaller;
- size-changing (dynamic, high-ratio) tasks;
- tasks with 3 examples, one test input and medium-sized grids.

**Weak:**
- few examples (≤ 2);
- several test inputs, which is mostly an all-inputs-must-be-right effect;
- large grids (our runs mostly forced-close there);
- same-size outputs where many cells change;
- tiling or expansion (n = 3).

**NVARC's easy class is not Qwen-easy:** easy_static is 25 % at XHigh. It is solvable sometimes, though (56 % at any level), so it is high-variance.

**Caveats:** groups overlap; several groups are small (n < 20); ARC Prize labels are single runs; our unit counts are dominated by repeated EVAL12 tasks. Treat these as **ordering hints, not facts.** The natural-close % is the most useful runtime signal: hits come almost only from natural closes.

## Oracle (union of every Qwen3.8-27B result we know), 2026-10-13

| Set | Tasks | ARC Prize XHigh | ARC Prize any level | Ours, any run | **Oracle (union)** |
|---|---|---|---|---|---|
| All 120 evaluation | 120 | 40 (33.3 %) | 51 (42.5 %) | 10.5 (8.8 %; only 45 tasks run) | **54.0 (45.0 %)** |
| PROXY48 | 48 | 16 (33.3 %) | 23 (47.9 %) | 9.5 (19.8 %; 37 tasks finished) | **25.5 (53.1 %)** |
| EVAL12 | 12 | 3 (25.0 %) | 4 (33.3 %) | 2.0 (16.7 %) | **4.5 (37.5 %)** |
| PROXY56 (E9 set) | 56 | 17 (30.4 %) | 25 (44.6 %) | 10.5 (18.8 %) | **28.0 (50.0 %)** |

- **How the scores are counted:**
  - our task score = the fraction of its test inputs solved at least once across our runs;
  - ARC Prize ✓ counts as a solved task;
  - oracle = solved by anyone.
- The oracle is an **upper bound**: the score you would get if one could always pick the right answer from all runs and effort levels. It is not reachable by any single run.
- **Tasks only we solved** (not ARC Prize at any level): `247ef758`, `36a08778`, `5961cc34`, `c4d067a0`.
- On the 61 test inputs we ran: ours 12, and 28 together with ARC Prize any-level tasks.
- **Single-run Qwen3.8 versus the oracle:** ARC Prize XHigh one run is 33.3 %, the union of 4 effort levels is 42.5 %, and with ours added it is 45.0 %. Independent samples add about 10 points on the evaluation set. This is the case for the E9 tail resample.

## Per-task record: every task we have run (45 tasks, 61 inputs)

Sorted by our hit rate. "Our hits / attempts" counts test-input runs across all 9 runs; (units) = test inputs. ARC Prize columns are XHigh / Medium / Low (✓ solved, · not).

| Task | Tier | fine | Key tags | Our hits / attempts (units) | ARC Prize X / M / L | PROXY48 |
|---|---|---|---|---|---|---|
| `1818057f` | 1 | 0.118 | easy_static | **1/1** (1) | · / · / ✓ | yes |
| `31f7f899` | 1 | 0.209 | many_colors | **1/1** (1) | ✓ / ✓ / ✓ | yes |
| `409aa875` | 1 | 0.185 | easy_static | **1/1** (1) | ✓ / · / · | yes |
| `5961cc34` | 2 | 0.248 | — | **1/1** (1) | · / · / · | yes |
| `7491f3cf` | 1 | 0.258 | many_colors | **1/1** (1) | ✓ / · / · | yes |
| `b99e7126` | 2 | 0.380 | large | **1/1** (1) | ✓ / · / · | yes |
| `c4d067a0` | 2 | 0.276 | large | **1/1** (1) | · / · / · | yes |
| `e376de54` | 1 | 0.225 | — | **1/1** (1) | ✓ / · / · | yes |
| `136b0064` | 1 | 0.474 | hard_tier1,dynamic | **7/9** (1) | ✓ / · / ✓ | yes |
| `36a08778` | 1 | 0.079 | easy_static,multi_query | **1/2** (2) | · / · / · | yes |
| `247ef758` | 1 | 0.236 | multi_query,many_colors | **4/16** (2) | · / · / · |  |
| `4c3d4a41` | 1 | 0.213 | multi_query,many_colors,low_train | **2/16** (2) | · / ✓ / · |  |
| `135a2760` | 1 | 0.208 | low_train | **0/1** (1) | · / · / · | yes |
| `20a9e565` | 3 | 1.512 | dynamic,multi_query,high_ratio,many_colors,large | **0/16** (2) | ✓ / · / · | yes |
| `21897d95` | 1 | 0.493 | hard_tier1,dynamic,multi_query,many_colors | **0/16** (2) | · / · / · |  |
| `269e22fb` | 1 | 0.440 | dynamic,multi_query | **0/2** (2) | · / · / · | yes |
| `271d71e2` | 1 | 0.155 | easy_static | **0/1** (1) | · / · / · | yes |
| `28a6681f` | 1 | 0.173 | — | **0/1** (1) | · / · / · | yes |
| `2b83f449` | 1 | 0.171 | easy_static,low_train | **0/9** (1) | · / · / · | yes |
| `332f06d7` | 1 | 0.138 | easy_static | **0/1** (1) | · / · / · | yes |
| `35ab12c3` | 1 | 0.265 | many_colors | **0/1** (1) | ✓ / · / · | yes |
| `3dc255db` | 1 | 0.163 | — | **0/1** (1) | ✓ / · / · | yes |
| `4a21e3da` | 1 | 0.168 | easy_static,multi_query | **0/2** (2) | ✓ / ✓ / ✓ | yes |
| `4c7dc4dd` | 3 | 1.818 | dynamic,multi_query,high_ratio,large,low_train | **0/16** (2) | · / · / · | yes |
| `53fb4810` | 1 | 0.206 | easy_static,low_train | **0/1** (1) | · / ✓ / · | yes |
| `6e453dd6` | 1 | 0.148 | easy_static | **0/1** (1) | · / · / ✓ | yes |
| `6e4f6532` | 2 | 0.329 | multi_query,many_colors,large,low_train | **0/16** (2) | · / · / · |  |
| `71e489b6` | 1 | 0.153 | easy_static,multi_query | **0/2** (2) | · / · / · | yes |
| `7666fa5d` | 1 | 0.165 | easy_static,low_train | **0/1** (1) | ✓ / · / · | yes |
| `7b0280bc` | 1 | 0.305 | many_colors | **0/1** (1) | · / · / · | yes |
| `7b80bb43` | 2 | 0.272 | large,low_train | **0/1** (1) | · / · / · | yes |
| `800d221b` | 1 | 0.284 | many_colors | **0/1** (1) | · / · / · | yes |
| `88bcf3b4` | 1 | 0.244 | multi_query,many_colors | **0/16** (2) | · / · / · |  |
| `898e7135` | 2 | 0.589 | dynamic,many_colors,low_train | **0/8** (1) | ✓ / · / · |  |
| `8b7bacbf` | 1 | 0.228 | multi_query | **0/16** (2) | · / · / · |  |
| `8f3a5a89` | 1 | 0.164 | easy_static | **0/1** (1) | · / ✓ / ✓ | yes |
| `97d7923e` | 1 | 0.164 | — | **0/1** (1) | · / · / · | yes |
| `b6f77b65` | 1 | 0.248 | multi_query,many_colors | **0/24** (3) | · / · / · |  |
| `c7f57c3e` | 1 | 0.228 | low_train | **0/1** (1) | · / · / · | yes |
| `cb2d8a2c` | 1 | 0.177 | easy_static,multi_query | **0/2** (2) | ✓ / · / · | yes |
| `da515329` | 1 | 0.138 | easy_static | **0/1** (1) | · / · / · | yes |
| `db695cfb` | 1 | 0.146 | easy_static | **0/1** (1) | · / ✓ / · | yes |
| `dbff022c` | 1 | 0.234 | many_colors | **0/1** (1) | ✓ / · / · | yes |
| `dd6b8c4b` | 1 | 0.140 | easy_static,multi_query | **0/2** (2) | · / · / · | yes |
| `de809cff` | 1 | 0.253 | low_train | **0/1** (1) | · / · / · | yes |

**Summary of our results versus ARC Prize XHigh:**
- 12 of 61 units have been solved by us at least once.
- Solved by both us and ARC Prize XHigh (6): `136b0064` `31f7f899` `409aa875` `7491f3cf` `b99e7126` `e376de54`.
- **Solved by us but not by ARC Prize XHigh (6):** `1818057f` `247ef758` `36a08778` `4c3d4a41` `5961cc34` `c4d067a0`.
- **Solved by ARC Prize XHigh but never by us (8):** `20a9e565` `35ab12c3` `3dc255db` `4a21e3da` `7666fa5d` `898e7135` `cb2d8a2c` `dbff022c`. Most of ours were forced_close at 49k.
- Never solved by anyone so far: 21 of the 45 tasks.

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

### 3. Codex tiers in our own PROXY48 run (E8 R5)

| Codex tier | Finished | Hits |
|---|---|---|
| tier 1 | 37 | 7 (easy_static 3/21) |
| tier 2 | 4 | **3** |
| tier 3 | 0 (all lost in the crash) | — |

- Easy-first spent the deadline on tier 1 tasks that Qwen mostly fails.
- There is no matched control, and the run was truncated, so this is a hint, not proof.

### 4. Near misses (same shape, clearly better than copying the input; candidates for an independent second sample)

`dd6b8c4b#0` 1 cell wrong, `dd6b8c4b#1` 2, `28a6681f` 2, `3dc255db` 4, `6e453dd6` 9, `8f3a5a89` 10. Most were forced_close.

### 5. Our 49k + 8k budget versus ARC Prize's budget (mean about 60k per generation, cap unknown) (EVAL12)

| Task | ARC Prize XHigh | Ours | Note |
|---|---|---|---|
| `136b0064` | ✓ | 6/8 | |
| `20a9e565` | ✓ | **0/16** | all forced_close at 49k |
| `898e7135` | ✓ | **0/8** | forced_close |
| `247ef758` | ✗ | 4/16 | |
| `4c3d4a41` | ✗ | 2/16 | |

- Tasks ARC Prize solved but we never did were all cut at 49k thinking. This is a hint that the token cap costs some solves on hard tasks.
- Our EVAL12 runs (88 units): **natural closes 7/16 hit (44 %)**, **forced_close at 49k 1/72 hit (1.4 %)**. Natural think lengths: p50 38.7k, p90 44.8k; 5 of 16 closed between 40k and 49k.
- E8 R5 PROXY48 (41 units): natural **9/15 (60 %)**, forced_close **1/26 (3.8 %)**; 6 of 15 natural closes came after 44k. **Over all runs: natural 16/31 (52 %), forced 2/98 (2 %).** This is observational: natural closes may be easier tasks.
- This is not a recommendation to change the budget (user rule: no change without approval). The hidden set is easier, and more tokens means fewer puzzles fit in 12 h.

## ARC Prize XHigh-solved tasks (40)

`0934a4d8` `136b0064` `16de56c4` `1ae2feb7` `20270e3b` `20a9e565` `2ba387bc` `2d0172a1` `31f7f899` `35ab12c3` `3dc255db` `409aa875` `4a21e3da` `581f7754` `58f5dbd5` `65b59efc` `67e490f4` `7491f3cf` `7666fa5d` `7b3084d4` `7c66cb00` `80a900e0` `898e7135` `8e5c0c38` `8f215267` `981571dc` `a6f40cea` `b5ca7ac4` `b99e7126` `bf45cf4b` `cb2d8a2c` `d35bdbdc` `d59b0160` `d8e07eb2` `db0c5428` `dbff022c` `e376de54` `e8686506` `edb79dae` `f931b4a8`

**PROXY48:** 16 of its 48 tasks are XHigh-solved, and 23 are solved at any level. Our E8 R5 run reached 10/56 test inputs, with 15 inputs lost to the crash.

## Idea for the end-game ordering (not implemented)

1. **Capability score per hidden task,** predicted from features, because the hidden tasks themselves have no labels:
   - fit "solved by Qwen (any run)" on the 120 evaluation tasks plus our runs, against the EDA features (low_train, multi_query, dynamic, n_colors, size, n_train);
   - training-set tasks (1,000) are much easier and could add labels if we ever run them.
2. **Order hidden tasks by predicted P(solve) / expected tokens.** Likely-solvable and short tasks go first; low_train and multi_query tasks go later.
3. **Validate on PROXY48:** compare NVARC order against capability order on the same tasks and deadline.

**Caveat:** 120 labelled tasks with one or a few runs each is a small, noisy sample. Several category differences above are within a few tasks, so treat them as hypotheses and update them as more runs are added.

## Capability profile by transformation type (2026-10-14, `capability_profile.py`)

**Data:**
- **Label:** a task counts as "solved by anyone" if any of these solved it: ARC Prize's verified Qwen3.8-27B run (xhigh, medium, low or none), any of our ledger GPU runs, or the archived PIV EVAL120 run (`1818057f`, `2ba387bc`). That gives **56 of 120 (47 %)**.
- **Features:** computed from the training pairs only.
- **Excluded:** the archived Q1 PROXY48 run (0/56; the xhigh arm was always truncated at 12k) and EVO12 (0/12). They carry no capability signal.
- **Significance:** p-values are from Fisher's exact test.

| Feature | Better for Qwen | Worse for Qwen | p |
|---|---|---|---|
| Training examples | 3 or more: 45/86 (52 %); 4+: 58 % | 2: 11/34 (32 %) | 0.07 |
| Test inputs | 1: 38/71 (54 %) | 2: 15/46 (33 %) | 0.04 |
| Same-size: share of cells changed | 15–30 %: 14/21 (67 %) | < 5 %: 33 %; ≥ 30 %: 22 % | 0.04 (15–30 % vs the rest) |
| Output size (ARC Prize xhigh) | output smaller: 42 % | same size: 30 %; output larger: 0/3 | 0.26 (weak) |
| New colour in the output | none: 48 % | new colour: 38 % | 0.59 (no evidence) |
| Same-size edit kind | only draws on background: 54 % | recolours: 40 %; mixed: 44 % | 0.56 (no evidence) |
| Grid size, number of objects | no trend | | |

**Reading:**
- Qwen does best when the rule is well specified by 3 or more examples and the edit is moderate: 15–30 % of cells change.
- It does worse on subtle edits (< 5 % of cells: a few cells decide the rule) and on full rewrites.
- **The "test inputs" effect is partly a scoring artifact:** a task passes in ARC Prize's table only if every input is right.
- **Runtime signal (our runs):** natural close before 40k thinking gives ~67 % hits; forced close gives ~2 %. Qwen solves what it can settle quickly.

## Ordering rules: derived and validated (2026-10-14, `derive_ordering_rules.py`)

| Rule set | In-sample AUC | **Honest AUC (nested 10-fold CV, search redone inside each fold, 20 repeats)** |
|---|---|---|
| Greedy search over ~80 training-pair features and thresholds: picks (15–30 % changed or output smaller) + n_train ≥ 3 + 1 test input | 0.681 | **0.521** (0.44–0.57): overfit, do not use |
| EDA tags only (defined before any Qwen result), 2 rules: not `low_train` + not `multi_query` | 0.622 | **0.570** (0.51–0.62) |
| EDA tags only, 3 rules (adds `easy_static`, unstable) | 0.639 | 0.588 (0.55–0.62) |
| NVARC fine / tier (the Codex ordering) | 0.528 / 0.510 | no search, so in-sample ≈ honest |

**Recommended if an ordering is used:**
- **score = [≥ 3 training examples] + [1 test input].**
- These were the two rules chosen in more than 85 % of the fold searches.
- Solve rate by score: 2 → 30/52 (58 %); 1 → 23/53 (43 %); 0 → 3/15 (20 %).

**Label check:**
- Rules searched on ARC Prize labels alone also rank **our own runs'** results (45 tasks) at AUC 0.66.
- So the signal is not ARC-Prize-run noise.
- But it is the same puzzles, so this is not evidence about new puzzles.

## Oracle order of the 120 puzzles (2026-10-14): `oracle_order.py` → `oracle_order_120.csv`

**Sources:**
- All 12 of our GPU runs in this ledger, including the ORDER2 EVAL120 86k run that covers every task.
- The ARC Prize Qwen3.8-27B solve list at all four levels.

**Estimate:**
- `p_hat = (S + X + B + 0.35) / (N + 1 + B + 1)`.
  - S is the sum over our runs of the task fraction solved; N is the number of our runs.
  - X is ARC Prize xhigh.
  - B is the number of ARC Prize lower levels solved. These count only when solved, as an easiness bonus.
- **Order:**
  1. tasks never solved by anyone go last;
  2. then `p_hat`, descending;
  3. then the measured ORDER2 minutes per unit, ascending.

| Tier | Rule | Tasks | ORDER2 86k task score | ARC Prize xhigh |
|---|---|---|---|---|
| A | p_hat ≥ 0.6 | 26 | 24.0 (92 %) | 24 |
| B | 0.35–0.6 | 21 | 11.0 (52 %) | 10 |
| C | solved somewhere, p_hat < 0.35 | 25 | 9.33 (37 %) | 6 |
| D | **never solved by anyone** | **48** | 0 | 0 |

- **72 of the 120 tasks (60 %) have been solved at least once by the Qwen3.8-27B methods tried so far** (ours or ARC Prize's plain xhigh run). The other 48 have not been solved by these methods yet.
  - This measures our methods, not the model. Other teams score 80+ on the leaderboard (2026-10-15), so tier D is the largest headroom for a better method.
- **Fair check:** the oracle was built without the ORDER2 run, then replayed on the ORDER2 run (16 slots, measured durations). Task-norm % finished by each minute:

  | Order | 120 min | 240 min | 360 min | 480 min |
  |---|---|---|---|---|
  | Oracle without ORDER2 | 22.9 | 30.7 | 32.8 | 36.9 |
  | ORDER2 rules | 19.2 | 24.2 | 30.8 | 36.5 |

- **Use:**
  - Only for these 120 evaluation tasks. The hidden tasks are different, so the oracle cannot order the hidden run.
  - It serves as a label set for finding better ordering features, and for choosing test subsets. Tiers B + C (46 tasks) are the cheapest subset for detecting a method gain. Tier D (48 tasks) is the largest headroom, since our current methods never solve it.

## Solved units by token cost (2026-10-14): `solved_by_tokens.py` → `solved_by_tokens.csv`

- Every solved test input is sorted by the thinking tokens of its cheapest solve. Only direct runs record tokens.
- **60 units / 54 tasks solved by us with token data:**

  | Thinking at cheapest solve | Units |
  |---|---|
  | ≤ 20k | 8 |
  | 20–30k | 11 |
  | 30–40k | 11 |
  | 40–49k | 11 |
  | 49–60k | 6 |
  | 60–70k | 4 |
  | 70k – cap | 9 (5 of them forced closes) |

  **41 units (68 %) were solved at ≤ 49,152 at least once; 19 (32 %) only above it.**
- Solved without token data: `4c3d4a41#1` (Poetiq R4).
- **18 tasks were solved only by ARC Prize.** In ORDER2 86k:
  - **6 of them closed early and wrong:** `6e453dd6` 26k, `edb79dae` 27k, `dbff022c` 32k, `7c66cb00` 37k, `db695cfb` 38k, and `d35bdbdc` (all 3 inputs at ≤ 50k). This is confident failure; more tokens would not help.
  - Most of the rest ran to the 77.8k cap.

**Update 2026-10-15 (Poetiq R5 EVAL120 added; ledger, oracle and token list rebuilt):**
- Tiers are now A 30 / B 20 / C 35 / **D 35**. Before R5 they were 26 / 21 / 25 / 48.
- **85/120 tasks have now been solved by at least one method.**
- Poetiq solved 13 tasks that had never been solved before (tier D before R5): `142ca369`, `28a6681f`, `2c181942`, `45a5af55`, `62593bfd`, `6e4f6532`, `88bcf3b4`, `8b7bacbf`, `8b9c3697`, `a251c730`, `aa4ec2a5`, `c7f57c3e`, `fc7cae8d`.
- **A better method moved the "never solved" set. It is not a model ceiling.**
- The pre-R5 oracle is kept in the session scratchpad (`oracle_order_120_before_R5.csv`).

## Oracle by method family (2026-10-25; `oracle_by_family.py`)

**Oracle** = a test query counts as solved if any run of the group solved it. The task-norm is out of 120.

**Outputs:**
- `oracle_by_family.json`: per group, the runs, counts, and the Duck-only / ARC-Prize-only lists.
- `oracle_units_by_family.csv`: a 0/1 table, one row per query, one column per group.

| Group | Runs | **Oracle queries (of 172)** | **Oracle task-norm** | % |
|---|---|---|---|---|
| Duck REPL agent (ours) | 14 | 71 | 52.00 | 43.3 |
| Poetiq / code + train verify | 8 | 122 | 88.33 | 73.6 |
| Direct reasoning | 8 | 60 | 45.33 | 37.8 |
| Other code / agent (R6 loop, REPL v1.8-1.9) | 3 | 2 (of 22) | 1.50 | 1.2 |
| ARC Prize verified run (xhigh / medium / low / none, task level) | 4 | 68 | 51.00 | 42.5 |
| Our non-Duck runs | 19 | 125 | 90.33 | 75.3 |
| **All our runs** | 33 | **128** | **92.33** | **76.9** |
| **Total oracle (ours + ARC Prize)** | 37 | **132** | **94.17** | **78.5** |

**Overlaps:**
- Duck-only queries: 3 (269e22fb#0, 269e22fb#1, de809cff#0).
- Duck ∩ our other methods: 68.
- Our other methods only: 34 (22.67 task-norm).
- ARC Prize only: 4 (20a9e565#0/#1, 581f7754#0, d35bdbdc#0).
- Solved by nobody: 40 queries.

The Poetiq oracle includes the deep HARD66 runs (D66 77.8k, L66 122.9k) and R5 EVAL120 (78 / 172). The Duck oracle includes the 3 EVAL120 G4B runs and the E15B commits.
