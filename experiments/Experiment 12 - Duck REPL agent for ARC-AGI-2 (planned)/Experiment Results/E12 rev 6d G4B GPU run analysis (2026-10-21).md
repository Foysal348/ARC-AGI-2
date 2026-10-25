# E12 rev 6d: EVAL12R6G4B GPU run, analysis (2026-10-21)

**Run**
- Kernel `foysalemonshanto/q27-xhigh-e12-duck-repl-sglang-eval12r6g4b`, v1, commit run; notebook sha `1bdadd82…`.
- Status COMPLETE; wall time 3,528 s (58.8 min).
- All 71 output files are in `_scratch/e12_rev6_results/G4B_v1/`.

**Gate 4b** (hidden-equivalent time share; a screening test on 12 selected tasks, not an LB projection):
- solve start 1,463 s;
- 171.9 s per task;
- window 2,063 s (34.4 min).

## Result

| Measure | Value |
|---|---|
| Eval | **2 / 19 inputs, task-norm 16.67 %**. Solved: bf45cf4b (b0, 19.4k tokens), b5ca7ac4 (b0, 27.4k). Both by `submit` with a host-verified candidate |
| Tasks exposed | **12 / 12**; stage-1 starts left **0**; stragglers 0 |
| Branches started | 16 (12 × b0 + 4 × b1). 14 paused at the deadline, 2 done |
| Generated tokens | 572,035 = **47.7k per task**, i.e. 277 tok/s over the window |
| Requests / tool calls | 180 / 175. Caps: 12,288 × 20, 24,576 × 160 |
| Cache | per-request cached 75.0 %; uncached prompt 670k tokens (more than the 572k generated) |
| Facts | p50 627 / p90 763 / max 798 tokens; 0 over the target |
| First python call | median **46 tokens** (b0 facts view). b1 grids view: 12.4k–18.0k |
| First check | 5 of 12 b0 branches within the budget (15.4k–37.0k); **10 of 16 branches never checked** |
| Transforms | **7 of 12 b0 branches never defined a `transform`** in 29k–59k tokens |
| Grid printing | 432 grid rows suppressed. Most b0 branches tried `print(row)` dumps first, then `show()` crops (150-cell limit) to rebuild whole grids |
| Overfit signs | `fixed_offset` × 2 (the 2 accepted candidates) |
| Preflight | isolation **PASS** 12/12; drafter `target_layer_ids` [5, 19, 33, 47, 61]; parity control 2/2 identical |

## Finding 1: false capacity step-down (12 → 10 → 8 → 6), a real bug in my rev 6d probe

- **Events:**

  | t (s) | allowed | queue | KV usage |
  |---|---|---|---|
  | 240 | 10 | 1 | 0.13 |
  | 600 | 8 | 1 | 0.26 |
  | 780 | 6 | 1 | 0.31 |

  The retraction counter was not present (`None`).
- **The server log shows queue = 1 is the steady state at 12 requests in flight:** 188 decode lines have `#queue-req: 1`, 101 of them at `#running-req: 11`. In those lines KV usage is 0.17–0.31 and mamba usage ≤ 0.70.
- **Neither pool is under pressure.** The server keeps one request waiting while 11 run (the known "11 of 12" behaviour).
- **Throughput by running level:**
  - at 11 running: 370–406 tok/s (20–40k context);
  - at 6 running: about 321 tok/s;
  - so the step-down cost about **13 %** of throughput in the second half. Corrected after GPT audit 2: at the same 20–40k context bin it is 370.9 → 321.4 tok/s. My first "15–20 %" mixed in the < 10k bin.
- **A second flaw:** the allowed count drops for new admissions only. The probe kept seeing queue = 1 from runs already in flight and stepped down again before the earlier step could take effect.
- **User decision (2026-10-21): replace the whole probe with one rule: no new task or branch is admitted while KV usage ≥ 0.95.** No step-down, queue, retraction or streak logic. Workers stay at 12.
- **Replay on 7 server logs** (`analysis/kv_queue_scan.py`):
  - the rule holds new work back only in E12 EVAL30 rev 1 (291 of 4,153 batch lines; the real KV-thrash run);
  - it never holds back in G4B, E11-05, E11-06, rev 4 or rev 5 (KV max 0.43–0.73);
  - mamba usage never exceeds 0.71.
- **Why the bug got through:** queue = 1 at 11 running was already in our own logs (E11-05 EVAL30: 2,615 of 2,816 lines). The "healthy" test fixture used queue = 0, which was my assumption, not a measured value.
- **Throughput detail, corrected after GPT audit 2:** I first wrote that 12 running was slower (median 194–258 tok/s vs 370–406 at 11). That claim is withdrawn.
  - All 29 lines at exactly 12 running fall in the first 13 minutes, before the first step-down.
  - They alternate between about 42–250 and 382–497 tok/s, which fits admission prefill interfering with decode.
  - So the low median is not a 12-stream ceiling.

## Finding 2: the facts view gives an early first call, but not an early transform

**Corrected 2026-10-21** after a full count (`analysis/g4b_fix_evidence.py` in the G4B review bundle). My first version said "many `show()` crops" and "a median of 3 viewing calls before the first transform". The count shows the `show()` crop loop in only 2 branches, and the median was not verified. The measured picture:

- **The first Python call comes at once** (median 46 tokens), almost always `arc_overview()`, which repeats the facts already in the system prompt (14 of 16 branches).
- **The model then prints the grids:**
  - 432 grid rows hidden by the guard (38 notes, 11 branches);
  - **guard gap:** 173 rows printed as `print(r, row)` (`0 [0, 0, 5, …]`) reached the model in 3 branches (4c3d4a41_b0 48, f931b4a8_b0 97, 247ef758_b1 28). `_GRID_ROW_RE` does not match a leading row index followed by a list;
  - `show()` crops (150-cell limit): 138 rows, mostly 136b0064_b0 (6 calls) and bf45cf4b_b0 (9 calls, solved).
- **The reasoning still sits in one or two long thinks per branch:** the median request has 874 reasoning characters, but the p90 is 25.6k and the max 64.9k, with no transform written. **7 of 12 b0 branches never defined a `transform`** in 29k–59k tokens.
- **The helpers are rarely used:**
  - `inspect_pair` in 7 branches; `components` in 3; `color_counts` in 2; `object_diff` and `diff_summary` in 1;
  - `crop`, `bbox`, `objects`, `grid_diff` and `check_many` in 0.
  - The 2 accepted transforms call no helper, so T1 was not exercised.
- **Grids view (b1, the hedge):** the model thinks 12–18k tokens before the first call, which is the rev 5 behaviour.
- **First check vs rev 5 xhigh on the same tasks:** mixed.
  - Earlier: 898e7135 107k → 15.4k; b5ca7ac4 46.6k → 26.2k.
  - Not reached within the budget: 136b0064 (rev 5: 34.5k), 4c3d4a41 (62.8k), 62593bfd (58.1k).
  - The median target of ≤ 25k was not met.
- **Token-matched comparison with rev 5** (`analysis/compare_rev5_vs_g4b.py`):
  - would rev 5's branch 0 have had a correct candidate within the tokens G4B's branch 0 used? xhigh 1 of 6, low 3 of 12; G4B solved 2;
  - per task: b5ca7ac4 better (26k vs 54k xhigh); 136b0064 (xhigh 35k), 247ef758 (low 29k) and f931b4a8 (low 47k) solved earlier in rev 5 than G4B's branch-0 budget;
  - one sample each, so the per-task differences can be sampling noise.
- **Options (need the user's go):**
  - close the guard gap (row-index prefix);
  - let `show(g)` print one whole grid up to 30 × 30 with row and column indices.

## Finding 3: time budget

- **Load:** the SGLang launch took **1,197 s** ("ready in 1197s"), against about 700 s in the rev 5 runs. This is a slow cold start on Kaggle, not our code.
- **Stage 1:** first-exposure runs took 1,120–1,510 s each, i.e. 30–44k tokens at about 25–35 tok/s per stream. That includes tool calls and prefill.
- **Overshoot past 28k:** p50 5.1k, max 15.8k (24,576 cap).
- **Stage 2:** 4 branch-1 starts and 2 branch-0 resumes in the remaining ~10 min. The heap had 22 items left at the deadline.
- **At hidden scale:**
  - with the step-down fixed, 11 effective streams × about 34 tok/s ≈ 370 tok/s;
  - that is ≈ 64k tokens per task at 172 s per task;
  - stage 1 for 240 tasks ≈ 240 × 33k = 7.9M tokens ≈ 5.9 h, within the 11.3 h solve window.

## What this run does and does not show

- **It shows:**
  - the engine is stable at 11–12 running: no OOM, real running max 12, acceptance 4.2, isolation PASS;
  - full first exposure works and the deadline holds (finished on time, 0 stragglers);
  - at about 48k tokens per task Duck solves 2 of these 12 selected (hard) tasks.
- **It does not show** the LB. The tasks are hand-selected, the queue is 12 tasks instead of 240, and the throughput was reduced by Finding 1.

## Proposed next step (the user decides)

1. Fix Finding 1 (probe), the decisive one for any further run.
2. Optionally Finding 2 (`show` whole grid).
3. Rebuild, run the CPU tests, then EVAL12R6 (unconstrained, 150 min) to compare accuracy with rev 5 at full budget.
