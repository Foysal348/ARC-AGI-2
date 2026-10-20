# Canonical baseline — PIV on the 120 public evaluation tasks (frozen 2026-10-03)

Source run: Kaggle `esfersami50/piv-q27-eval120-measurement` v1 (notebook `PIV-Q27-EVAL120-MEASUREMENT.ipynb`),
budget 8,192 tokens/task (6,144 exploration + 2,048 finalizer), arm ACT-LOW, concurrency 12.
Cross-checked independently by three reviewers; every later PIV run is compared against these numbers using these
definitions.

## Outcome metrics

| metric | definition | value |
|---|---|---|
| **Train-Verified** | final program reproduces every training pair exactly in the fresh verifier | **3/120** (`1818057f`, `2ba387bc`, `7b5033c1`) |
| **Strict Solved** | every test query of the task correct with attempt_1 or attempt_2 | **2/120** (`1818057f`, `2ba387bc`) |
| ARC score (task-normalised) | mean over tasks of fraction of queries correct | **1.67 %** |
| query exact | correct queries / all queries | **2/172 = 1.16 %** |
| submission-policy rescore | same programs, unverified ones also executed on test, attempt_1/attempt_2 ranking, identity fallback | **1.67 %**, unverified programs contributed 0 hits |

`7b5033c1` is Train-Verified but not Strict Solved (fits all training pairs, wrong on test).

## Process metrics (the levers)

| metric | value |
|---|---|
| usable program produced | 82/120 |
| finalizer truncated (exactly 2,048 tokens, `length`, JSON parse error) | 38/120; successful finalizers median 929.5, max 1,824 tokens |
| finalizer comment share | failed ≈65 % of code characters in comment-only lines (≈27–28/38 exceed 50 %); successful ≈23–24 %. Exact value depends on the counting rule |
| zero-tool tasks | 49/120; all 49 first turns = 6,144 tokens, `length`, with row-by-row grid transcription |
| tool calls that define `solution()` AND run it on training pairs | 3 of 173 (one per Train-Verified task) |
| `puzzle.keys()` | present in 65 of 173 tool calls; 11 calls are nothing but that probe |
| reasoning handoff truncated to last 1,024 tokens | 102/120 (source median 6,093.5 tokens) |
| exact training pairs per program | candidate-1: 20/82 = 0.244; post-repair selected: 21/82 = 0.256 |
| repair | 79 attempted; 52 parse failures (51 truncation, 1 syntax error `38007db0`); 27 parsed; 0 new Train-Verified; cap median 1,103, min 224 |
| near misses (partially-wrong group, 46 tasks) | 36 match 0 training pairs, 8 match 1, 2 match 2; on the 39 tasks whose every pair is shape-comparable, median wrong-cell fraction over failed same-shape pairs = 12.85 % (≤5 %: 9 tasks) |

## Compute metrics

| metric | value |
|---|---|
| wall clock | 16,404.66 s = **4.56 h** (pipeline complete 16,392.70 s; agent loop 15,664.19 s) |
| quota cost | ≈9.1 quota-hours (4×L4 burns 2 quota-hours per wall-hour) |
| exploration | 60 waves, 10,983 s, 716,922 tokens, 65.3 tok/s, mean active 4.75, 16 waves with active=1 |
| finalizer | 49 waves, 3,914 s, 151,920 tokens, 38.8 tok/s, median active 2 |
| low-occupancy waste | 58 of 109 exploration+finalizer waves had ≤2 active requests: 4,336 s = 29.1 % |
| prompt tokens | eval median 6,511 / mean 6,863; max rendered 26,748 of 29,696; 0 context deaths |

## Comparison with the NVARC pipeline (the team's 32.64 LB entry)

Reconstructed from the NVARC experiment notebook's own per-query output (`Document/31_experiment Notebook_arc-2026d-with-codex (22).ipynb`, cell 15, Probmul rank 1–2 = dual selection). Note that NVARC's printed "33.50" is a task count, not a percentage.

| | eval-120 |
|---|---|
| NVARC | 33.5 task-equivalents = **27.92 %**; 29 tasks fully solved, 81 tasks with no query solved |
| PIV | 1.67 % |
| **PIV-only solves** | **0** — both PIV solves are also NVARC solves |
| PIV near misses (≤20 % wrong cells on training, not yet verified) | 36, of which **25 are tasks NVARC misses entirely** |
| `7b5033c1` (PIV Train-Verified, test wrong) | NVARC misses it too |

**Success criterion for PIV from now on:** additional Strict Solved tasks among NVARC's 81 misses, not PIV's standalone score.
