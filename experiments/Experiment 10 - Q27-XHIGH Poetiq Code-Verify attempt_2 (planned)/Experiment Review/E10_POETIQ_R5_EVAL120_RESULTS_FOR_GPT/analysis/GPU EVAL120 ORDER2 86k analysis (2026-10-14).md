# GPU EVAL120 ORDER2 86k DIRECT: deep analysis (Claude, 2026-10-14)

**Run:** `hosen42/q27-xhigh-e10-order2`. Notebook `Q27-XHIGH-v1_4_4_E10_86K_ORDER2.ipynb`, sha `b62ae567…`, profile **EVAL120**.

**Setup:**
- All 120 evaluation tasks, 172 test inputs.
- Direct reasoning only: 77,824 thinking + 8,192 answer tokens.
- 16 concurrent requests, ORDER2 admission order.
- Hidden-run clock: admit until 694 min.

**Outputs:** `Kaggle Runs/GPU EVAL120 ORDER2 86k DIRECT (hosen42 q27-xhigh-e10-order2)/`

**Scripts:**
- Session scratchpad `order2/an.py`.
- The ledger was rebuilt with this run (`resources/Puzzle Capability Ledger`).

## Headline

| | |
|---|---|
| Score | **58/172 test inputs = 33.72 %**; task-normalised **36.94 %** (44.33/120) |
| Tasks | 37 fully solved, 15 partly solved, 52 with at least one hit |
| Finish | all 172 units; no drain, no unadmitted units, no error, 0 preemptions |
| Time | first unit done at 15.9 min; last admitted 489.6 min; last done **534.0 min** |
| How thinking ended | natural close 97 (**52 hits, 54 %**), forced close at the cap 74 (**5 hits, 6.8 %**), continue 1 (1 hit) |
| Throughput | 333 tok/s overall; MTP acceptance 0.669; prefix-cache hit 0.834 |
| KV | max **87.3 %** (at 318 min), p95 79.5 %, mean 68.6 % with 16 running |

## 1. Token allocation: what the extra 49k → 78k thinking bought

Hits lost if thinking were capped at X. This assumes a hit that needed more than X would be lost. That is close to true: forced closes hit only 6.8 % here.

| Cap X | Hits kept (of 58) | Lost |
|---|---|---|
| 40,000 | 28 | 30 |
| **49,152** | **39** | **19 (33 %)** |
| 60,000 | 44 | 14 |
| 70,000 | 48 | 10 |
| 75,000 | 52 | 6 |

**Thinking-token yield per band.** The cost of a band is the tokens all live units spent inside it; hits are attributed to the band where thinking ended.

| Band | Units still thinking | Tokens | Hits | Hits per M tokens |
|---|---|---|---|---|
| 0–20k | 172 | 3.42 M | 6 | 1.8 |
| 20–30k | 166 | 1.59 M | 12 | 7.5 |
| 30–40k | 149 | 1.41 M | 10 | 7.1 |
| 40–49k | 132 | 1.14 M | 11 | 9.7 |
| 49–60k | 118 | 1.19 M | 5 | 4.2 |
| 60–70k | 104 | 0.93 M | 4 | 4.3 |
| 70–77.8k, plus forced closes at the cap | 84 | 0.61 M | 5 + 5 | ≈16 |
| **Total 49–77.8k** | | **2.74 M** | **19** | **6.9** |
| **Total 0–49k** | | 7.56 M | 39 | 5.2 |

**P(solve | still thinking past X):**

| Past | Units | Solved |
|---|---|---|
| 20k | 166 | 31.3 % |
| 30k | 149 | 26.8 % |
| 40k | 132 | 22.7 % |
| 49,152 | 118 | 16.1 % |
| 60k | 104 | 13.5 % |
| 70k | 84 | 11.9 % |

**Natural-close accuracy by length:**

| Natural close at | Hits / closes |
|---|---|
| ≤ 49k | 39/54 (72 %) |
| 49–60k | 5/14 |
| 60–70k | 4/20 |
| 70–78k | 4/9 |

**Most of the 19 solves after 49k are genuinely new:**
- 14 of the 19 had never been attempted before.
- The other 5 had been attempted before, mostly at 49k:
  - `898e7135`: 0/10 before; solved here at 71.6k.
  - `b6f77b65`: 1/30 before; `#1` solved here at 71.4k.
  - `135a2760` (0/1) solved at 72.0k, and `7b0280bc` (0/1) at 66.1k.
  - `71e489b6` (0/2) was solved by a forced close.

**Verdict on direct reasoning:** the 77,824-token cap is justified.
- One third of all solves need more than 49,152.
- The marginal yield after 49k (6.9 per M) is **not lower** than the yield of the first 49k (5.2 per M).
- The extra band costs about 2.74 M tokens, roughly 2.3 GPU-hours at 333 tok/s, for +19 hits (+49 %).

**For Poetiq iterations:**
- **Iteration 1:** use 77,824. Reasons:
  - this data;
  - the R4 Poetiq solves came at 67–71k;
  - 10 of 12 R4 round-1 requests reached the cap.
- **Iteration 2 and later:** this run has **no evidence**. A feedback iteration is not a fresh reasoning.
  - Decide from the R5 EVAL120 run, which saves the full thinking per iteration in `q27x/poetiq_thinking/`.
  - Needed from it: think tokens per iteration number, and at which length the iteration ≥ 2 solves happened.

## 2. ORDER2 ordering: validated, but weak

| order_score | Tasks (units) | Task score | Unit hits | Forced | Mean think | Median min/unit | ARC Prize xhigh |
|---|---|---|---|---|---|---|---|
| 2 (≥3 train + 1 test) | 52 (52) | **25.00 = 48.1 %** | 25/52 | 33 % | 53.3k | 41.4 | 21/52 |
| 1 | 53 (90) | 16.33 = 30.8 % | 27/90 | 46 % | 62.0k | 55.0 | 18/53 |
| 0 | 15 (30) | 3.00 = 20.0 % | 6/30 | 53 % | 64.9k | 49.6 | 1/15 |

- **Monotone, as predicted.** AUC of order_score against task solved: 0.551; against task fraction ≥ 0.5: 0.557. Nested CV had predicted 0.57.
- **Front-loading works:**

  | Admit quartile | Admitted at | Hits |
  |---|---|---|
  | 1 | 12–94 min | **53 %** |
  | 2 | 100–226 min | 26 % |
  | 3 | 227–358 min | 30 % |
  | 4 | 361–490 min | 26 % |

  | By | Units done | Hits |
  |---|---|---|
  | 2 h | 37 | **20** |
  | 4 h | 78 | 33 |
  | 6 h | 113 | 45 |
  | 8 h | 154 | 57 |

- **It mattered nothing for this eval score:** every unit finished, at 534 min.
  - Its value is only for a run that cannot finish all tasks before the admit deadline.
  - Score-2 tasks also run shorter (41 vs 55 min median), so they free slots faster.
- **Previously run vs never run:**
  - 45 tasks previously run: 38.5 % task score.
  - 75 never-run tasks: 36.0 %.

## 3. Comparison with ARC Prize's own Qwen3.8-27B xhigh run (public eval)

| | ARC Prize xhigh | Ours (ORDER2) |
|---|---|---|
| Tasks fully solved | 40/120 (33.3 %) | 37/120; task-normalised 44.33/120 (36.9 %) |
| Tasks with any hit | — | 52 |

- **Overlap:**
  - **26 tasks solved by both.**
  - **26 only by us:** `135a2760`, `1818057f`, `247ef758`, `332f06d7`, `38007db0`, `3e6067c3`, `446ef5d2`, `58490d8a`, `5961cc34`, `5dbc8537`, `71e489b6`, `7b0280bc`, `7b5033c1`, `7ed72f31`, `88e364bc`, `8f3a5a89`, `9385bd28`, `9aaea919`, `a395ee82`, `a47bf94d`, `b0039139`, `b10624e5`, `b6f77b65`, `c4d067a0`, `cbebaa4b`, `dd6b8c4b`.
  - **14 only by them:** `0934a4d8`, `20270e3b`, `20a9e565`, `2d0172a1`, `35ab12c3`, `3dc255db`, `67e490f4`, `7b3084d4`, `7c66cb00`, `8e5c0c38`, `cb2d8a2c`, `d35bdbdc`, `dbff022c`, `edb79dae`.
- **Our per-task ability with 86k is at least as good as ARC Prize's xhigh measurement**, whose semi-private score was 42.4 %.
  - The gap to our LB (27.92 with 49k) is therefore mainly budget and coverage on the hidden run, not reasoning ability.
  - The E9 86k LB result (ref 57173293, pending) is the direct test of this.
- **The union is 66 tasks with a hit.** Run-to-run variance is large.
  - Over the 61 units seen before: 11 hit both times, 9 new hits, and 3 lost: `36a08778#0` (1/1 before), and `4c3d4a41#0` / `#1` (4/10 and 1/10).
  - This is the room an independent second trace could fill.

## 4. Which tasks think long (input for easy/hard mixing)

Per unit:

| Feature | Units | Hit | Forced | Median think |
|---|---|---|---|---|
| single test | 71 | **45 %** | 34 % | 61.8k |
| multi test | 101 | 26 % | **50 %** | 75.5k |
| same size | 117 | 36 % | 35 % | 64.6k |
| dynamic size | 55 | 29 % | **60 %** | 77.8k |
| n_train ≥ 3 | 123 | 37 % | 41 % | 66.1k |
| n_train ≤ 2 | 49 | 27 % | 47 % | 72.0k |
| change < 5 % | 15 | 53 % | 33 % | 53.5k |
| change 5–30 % | 90 | 34–35 % | 30–32 % | 63–66k |
| change ≥ 30 % | 12 | 25 % | **67 %** | 77.8k |
| tier 3 | 6 | 0 % | **100 %** | 77.8k |

- **Long thinking is only weakly predictable before the run:**
  - AUC of prompt tokens → forced close: 0.574.
  - AUC of order_score → natural close: 0.578.
  - Rank correlation of prompt tokens with think tokens: 0.16.
- The usable signals are dynamic size, multi test, change ≥ 30 % and tier 3.
- Mixing easy and hard tasks by these signals can only smooth KV a little.

## 5. KV and concurrency

- With 16 running: mean **68.6 %**, max **87.3 %**, 0 waiting, 0 preemptions.
- The peak came in the middle of the run (318 min), not at the synchronised start.
- **Fixed concurrency:**
  - Scaling the peak linearly, 18 slots would reach about 98 % and 20 about 109 %, which means preemptions.
  - So a fixed 18–20 is **not safe**.
- **KV-aware admission:** admit an extra request only when KV usage is below about 65 %, with a hard cap of 20.
  - This would average about 18 running, given the 31 % mean headroom.
  - The throughput gain on 4×L4 is unmeasured; batching gains are sub-linear. It needs a test run.
- Slot utilisation over the whole run: **95.3 %** (mean 15.25 active).
  - The only idle period is the 44-min tail after the last admit, with 7.2 active on average.

## 6. Time

- **Per unit:** median 51 min. Natural closes 33.7 min, forced 61.4 min. About 46 s per 1k thinking tokens per slot.
- **The notebook's own hidden projection (13.36 h for 259 queries) is not a valid estimate:**
  - the hidden puzzles are easier (shorter thinking) than eval;
  - per the standing rule, hidden time is not projected from eval.
  - The E9 86k LB run shows the real hidden behaviour.

## 7. Small observations (not blockers)

- **7 forced closes produced no parseable grid:** `aa4ec2a5#0`, `13e47133#1`, `36a08778#1`, `45a5af55#0`, `de809cff#0`, `4e34c42c#0`, `6e4f6532#0`.
  - The submission falls back to the test input for these, as `first_attempt_source` test_input 7 shows.
  - Forced closes rarely hit anyway.
- **4 forced closes ended in a pure repetition loop.** Their saved thinking tails (zlib ratio < 0.02) are:
  - `0934a4d8#0` and `6ffbe589#0`: "row? row? …";
  - `800d221b#0`: " c? c? …";
  - `3a25b0d8#0`: " r10 c? c? …".
  - Only the last ~3–4k characters of thinking are saved, so the loop's start (the wasted tokens) is unknown.
- **Multi-test tasks are 101 of 172 units (59 %) and the longest-thinking group.**
  - 15 tasks were half solved: one test input hit, the other missed, in independent traces.

## Decisions this run supports

1. **Direct reasoning keeps 77,824 thinking.** One third of solves come after 49k, at an equal-or-better token yield.
2. **Poetiq iteration 1 gets 77,824.** Iteration 2+ is decided from the R5 EVAL120 thinking logs.
3. **ORDER2 is kept** as a free front-loading for the hidden run, but it is weak (AUC 0.55). It adds nothing when all units finish.
4. **Concurrency stays at 16 unless a KV-aware admission controller is built and measured.** A fixed 18–20 risks preemptions.

## 8. Did the ordering work? Puzzle by puzzle (added the same day)

**Files:**
- Per task: `ORDER2 EVAL120 per-task admission vs outcome (2026-10-14).csv`. It gives admission position, order_score, hits, ARC Prize xhigh, file position, admit/done minutes, think tokens and NVARC fine.
- Script: `ordering.py`.

**Subtypes:**

| Subtype | Tasks | Ours (task score) | ARC Prize xhigh |
|---|---|---|---|
| 2: ≥3 train + 1 test | 52 | **48.1 %** | 40.4 % |
| 1b: ≤2 train + 1 test | 19 | 36.8 % | 31.6 % |
| 1a: ≥3 train + multi test | 34 | 27.5 % | 35.3 % |
| 0: ≤2 train + multi test | 15 | **20.0 %** | 6.7 % |

**First-k tasks:**

| Order | First 20 | First 40 | First 60 | First 80 |
|---|---|---|---|---|
| ORDER2 (solved-any) | 9 | **20** | 28 | 36 |
| File order (solved-any) | — | 14 | 25 | 34 |

- Median admission position: solved tasks 55, unsolved 63.

**Misranked tasks:**
- 27 of the 52 score-2 tasks failed (the sort has only 3 levels).
- 5 of the 15 score-0 tasks had a hit: `38007db0` (2/2), and `446ef5d2`, `5dbc8537`, `7ed72f31`, `cbebaa4b` (1/2 each).

**Replay.** Each unit keeps its measured duration; 16 slots; the score counts units finished by minute X.

| Order | 120 min | 240 min | 360 min | 480 min | All done |
|---|---|---|---|---|---|
| **ORDER2 (as run)** | **19.2 %** (23 hits) | **24.2 %** | **30.8 %** | 36.5 % | 522 min |
| File order (no ordering, as in E9) | 7.1 % (12 hits) | 15.0 % | 25.0 % | 36.5 % | 528 min |
| Random order (mean of 1,000) | 8.5 % | 17.3 % | 26.1 % | 34.9 % | — |
| NVARC fine, easy first | 9.6 % | 16.1 % | 26.5 % | 35.3 % | 518 min |
| Reverse ORDER2 | 3.8 % | 9.0 % | 15.3 % | 33.6 % | 527 min |
| ARC Prize xhigh solved first (external oracle) | 17.1 % | 28.3 % | 32.5 % | 36.5 % | 520 min |
| Our own outcome first (perfect oracle) | 32.9 % | 36.9 % | 36.9 % | 36.9 % | 513 min |

ORDER2 beats this share of the 1,000 random orders:

| At | Share beaten |
|---|---|
| 120 min | 100 % |
| 240 min | 99.4 % |
| 360 min | 98.3 % |
| 480 min | 90 % |

**Verdict:**
- **The ordering works as designed.** Solvable puzzles came clearly earlier than in file order, random order or NVARC order.
- **In this run it changed nothing in the final score,** because every unit finished (534 min). It only pays off if the hidden run cannot finish all units.
- **Caveats:**
  - The two rules were picked using these same 120 tasks (ARC Prize + our runs), so the effect on new tasks is probably smaller. Nested CV predicted AUC 0.57; here it was 0.55.
  - The replay assumes unit durations do not depend on order.
