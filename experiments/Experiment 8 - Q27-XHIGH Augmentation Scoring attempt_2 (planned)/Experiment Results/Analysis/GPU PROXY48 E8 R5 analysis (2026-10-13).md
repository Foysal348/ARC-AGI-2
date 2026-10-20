# GPU PROXY48 E8 R5 run: analysis (Claude, 2026-10-13)

**Run:** `foysalemonshanto/q27-xhigh-e8` v1, scriptVersionId 357913772. Outputs are in `../Kaggle Runs/GPU PROXY48 E8 R5 (foysal q27-xhigh-e8 v1)/`; the GPT bundle is `Experiment Review/E8_R5_GPU_RUN_FOR_GPT.zip`.

## 1. Outcome

| | Value |
|---|---|
| Units planned / admitted / finished | 56 / 56 / **41** (15 lost) |
| Eval hits (pass@2, per test input) | **10 / 56 = 17.86 %** (task-normalised 19.79 %) |
| E8 net gain | **0** (final = baseline) |
| Modes | natural 15 (**9 hits, 60 %**), forced_close 26 (**1 hit, 4 %**) |
| Throughput / preemptions / KV / MTP | 353.5 tok/s; 0 preemptions; KV about 67 %; MTP acceptance 0.663 |
| Mean think tokens | 45,389 |

## 2. The crash: predicted OOM, confirmed

**Timeline:**
- The queue was empty at about 108 min of scheduler time.
- At notebook minute 124.0 `7b80bb43#0` finished; the primary-first rule then let the **first and only** scorer request start.

**The failure:**
- That request: 1,665 prompt tokens, `prompt_logprobs=1`.
- All 4 TP workers raised `torch.OutOfMemoryError` in `LogitsProcessor._gather_logits` → `all_gather`: "Tried to allocate 790.00 MiB … 349–417 MiB free".
- 790 MiB = 1,665 × 248,077 × 2 B, i.e. the bf16 full-vocabulary logits for every prompt position, all-gathered to every GPU.
- Free memory during the run was only about 0.4 GB, plus about 0.5 GB reserved but unallocated. A 1,500-token guard would not have helped (about 710 MiB). A 200-token prompt (about 95 MiB bf16 + 190 MiB fp32) might fit, but the safe limit is unmeasured. (Corrected 2026-10-13: the earlier "even 200 tokens is unsafe" was overstated.)

**Consequence:**
- `EngineDeadError`: the 15 in-flight units were lost about 23 minutes before the drain.
- The scheduler handler kept the 41 finished units, and the submission and reports were still written.
- That is the "scorer OOM kills primaries" risk from `Experiment Review/E8_R5_QUICK_CHECK (2026-10-13).md`.

**Lost units (15):**
- tier 3: `0934a4d8`, `20a9e565`×2, `4c7dc4dd`×2, `89565ca0`;
- tier 2: `16b78196`, `2c181942`, `4c416de3`, `64efde09`, `7c66cb00`, `58490d8a`, `80a900e0`, `8b9c3697`;
- tier 1: `291dc1e1`.
- **5 of them were solved by ARC Prize at XHigh** (`0934a4d8`, `20a9e565`×2, `7c66cb00`, `80a900e0`).

## 3. Gate B (candidate mining): negative

- Full thinking was saved for all 41 finished units; the baseline was wrong on 31.
- **The correct grid appears nowhere in the full thinking text: 0 / 31**, with no context filter.
  - Grids per thinking: median 5, max 25.
  - Only 1 unit has any same-shape grid within 5 % of the solution.
- Together with the earlier 0 / 88 (thinking tails, 4 EVAL12 runs), **thinking-derived candidates do not contain new correct answers.**
- **The 8-view scorer therefore has nothing to rank.** Gate B fails; E8 as designed cannot improve attempt_2.
- The 9 one-novel direct replacements: none correct. 3 of the 9 were on units already solved by attempt_1 (`7491f3cf`, `5961cc34`, `c4d067a0`). No harm, because the old fallback never hits.

## 4. Primary stage and ordering

| Codex tier | Units | Finished | Hits | Natural |
|---|---|---|---|---|
| tier 1 | 38 | 37 | 7 | 12 |
| tier 2 | 12 | 4 | **3** | 3 |
| tier 3 | 6 | 0 | 0 | 0 |

- **Codex easy-first** ran almost all tier 1 first:
  - tier 1 solved **7 / 37 (19 %)**, `easy_static` 3 / 21;
  - the few tier 2 units that finished solved **3 / 4**;
  - this matches the ledger finding that NVARC's "easy" is not Qwen's easy (fine-score AUC 0.435; ARC Prize: easy_static 25 %, dynamic 41 %).
- In this 3 h, deadline-bound run (plus the crash), the ordering spent the time on tier 1 tasks that Qwen mostly fails.
  - In the hidden 12 h run, ordering matters only if the deadline binds.
- **Agreement with ARC Prize XHigh on finished units:**

  | | Ours ✓ | Ours ✗ |
  |---|---|---|
  | ARC Prize ✓ | 6 | 8 |
  | ARC Prize ✗ | 4 | 23 |

  Run-to-run variance is large, so labels need several runs.
- **New tasks solved here** that ARC Prize missed: `1818057f`, `36a08778#0`, `5961cc34`, `c4d067a0`.

## 5. Conclusions

1. **Stop E8 as designed.**
   - Mining yields no correct candidates (0 / 31 here, 0 / 88 before).
   - In-engine `prompt_logprobs` scoring is unsafe on this engine at any useful length; it killed 15 primaries.
2. **The baseline primary result here (10 / 56) cannot be compared with E7.** The crash removed 15 units, and no PROXY48 run without E8 exists.
3. **attempt_2 needs new candidates, not ranking:** an independent second sample (Experiment 6 showed diversity), or a different method. That is a GPU-time trade-off against primary coverage.
4. **Ordering:** evidence keeps pointing to capability-based ordering rather than NVARC tiers; see `resources/Puzzle Capability Ledger/`, where this run has been added.

---

# Deeper analysis (second pass, 2026-10-13)

## 6. Where the hits come from: natural closes only

| Mode | Units | Hits | Mean unit time | Mean generated tokens | Total GPU slot-minutes |
|---|---|---|---|---|---|
| natural | 15 | **9 (60 %)** | 29.4 min | 39.6k | 441 |
| forced_close (cut at 49,152) | 26 | **1 (4 %)** | 38.3 min | 50.4k | **996** |

- Across all our runs, forced-close answers hit **2 / 98** (this run 1/26, EVAL12 runs 1/72). The answer written after the host-forced close is almost worthless.
- **69 % of the slot time went to units that yield about 4 %.**
- **Natural close lengths:** 17.6k, 18.7k, 31.0k, 35.3k, 37.0k, 39.5k, 39.7k, 41.8k, 41.8k, 44.5k, **46.1k ✓, 46.6k ✓, 46.7k ✓, 48.0k, 48.3k**.
  - **6 of 15 natural closes came after 44k, just under the 49,152 cap**, and 3 of those 6 hit.
  - The close-length distribution is still dense at the cap, so a part of the 26 forced units would probably have closed naturally a little later.
  - This is the strongest evidence so far for the planned **86k budget** (77,824 think). The fraction that would close between 49k and 77.8k is unmeasured; the 86k run will measure it.

## 7. How close the wrong answers were

Of the 31 wrong units:

| Kind of wrong answer | Units |
|---|---|
| no grid | 2 |
| wrong shape | 3 |
| right shape | 26 |

Near misses that are clearly better than copying the input (model cell accuracy versus identity accuracy):

| Unit | Mode | Model | Identity | Wrong cells |
|---|---|---|---|---|
| `dd6b8c4b#0` | forced | 0.992 | 0.901 | **1** |
| `dd6b8c4b#1` | forced | 0.983 | 0.851 | 2 |
| `28a6681f#0` | forced | 0.980 | 0.900 | 2 |
| `3dc255db#0` | natural | 0.974 | 0.897 | 4 |
| `6e453dd6#0` | forced | 0.970 | 0.773 | 9 |
| `8f3a5a89#0` | natural | 0.931 | 0.674 | 10 |
| `7b80bb43#0` | forced | 0.992 | 0.972 | 4 |

- These are real "almost solved" cases, mostly forced closes: the rule was largely found, but the run ran out of thinking.
- They are also the cases where an **independent second sample** (attempt_2) has the best chance.
- Mined candidates did not help here: no thinking grid was within 5 %, apart from 1 unit.
- **No-grid units:**
  - `4a21e3da#0`: 21×21 answer with one 20-wide row, so the jagged grid is rejected (18/21 rows were correct).
  - `271d71e2#0`: answer looped to the 8,192 cap (158 rows); the first 26 rows are only 65 % correct.
  - A parser salvage would not have produced a hit.

## 8. Multi-input tasks: sibling inputs are solved independently

| Task | q0 | q1 |
|---|---|---|
| `36a08778` | natural ✓ | forced ✗ |
| `cb2d8a2c` | forced ✗ | natural ✗ |
| `dd6b8c4b` | forced 99.2 % | forced 98.3 % |
| `71e489b6`, `4a21e3da`, `269e22fb` | forced ✗ | forced ✗ |

- Each test input re-derives the same rule from scratch, and the sibling results disagree.
- **Idea, not implemented:** once one input of a task closes naturally, its reasoning or rule could seed the sibling input.
- Multi-input tasks are about 7 % of the hidden set (adam99), so the gain is limited there. It matters for evaluation-style sets.

## 9. Throughput, KV and memory (for the 86k plan)

**Steady state:**

| Notebook minutes | Gen tok/s | Running | KV max |
|---|---|---|---|
| 0–20 | 373 | 15.3 | 38 % |
| 20–40 | 368 | 16 | 71 % |
| 40–60 | 326 | 16 | **73.6 %** |
| 60–80 | 359 | 16 | 69 % |
| 80–100 | 358 | 16 | 69 % |
| 100–120 | 354 | 16 | 70 % |
| 120–140 | 368 | 16 | 68 % |

- No waiting after the start, 0 preemptions.
- MTP mean acceptance length 2.99 (range 2.65–3.37).
- Setup before the scheduler started: 13.4 min.

**KV accounting:** the engine's KV usage is about **1.25 × the live tokens plus about 11k per sequence** (fitted at t = 40 and 80 min: 960k KV-equivalent for 628k live tokens, and 880k for 564k). Block size 1,600, GDN/MTP state and fragmentation explain the overhead.

**At 86k cap:**

| Case | KV need | Pool 1.354M |
|---|---|---|
| Worst case: all 16 near the cap | 16 × 91k × 1.25 + 175k ≈ **2.0M** | over |
| Typical, if mean generation rises to about 65k | about 0.9M (≈ 67 %) | fits |
| Peaks, when several long units coincide | ≈ 95–105 % (est.) | around the limit |

So expect **occasional preemptions at 16 sequences**. Either run 14 sequences, or keep 16 and watch `num_preemptions`.

**Free GPU memory beside the KV pool was only about 0.4 GB.** Any extra activation (prompt logprobs, larger batches) is unsafe.

## 10. Hidden-run time: sensitivity, not a projection

The hidden set is easier and its forced share is unknown, so EVAL numbers must not be projected (see memory `hidden-test-easier-than-eval`). Below is only a sensitivity table, using the measured 353 tok/s and the per-mode token means. The generation window is about 11.3 h (admission to 11:34, minus 13 min setup), i.e. about 14.4M tokens.

| Forced share f (hidden, unknown) | Current cap (57k): tokens / hours | 86k cap (forced units about 80k): tokens / hours |
|---|---|---|
| 0.2 | 10.8M / 8.5 h | 12.4M / 9.7 h |
| 0.4 | 11.4M / 9.0 h | 14.4M / **11.3 h (at the limit)** |

**Implications** (user's premise: evaluation is about 2× harder than the hidden test, so the hidden forced share is well below the evaluation 63 %; the evaluation row was removed on purpose):
- At a hidden forced share of about 0.3 or less, 86k fits (about 10–10.5 h of the 11.3 h window).
- A time-aware cap stays optional, as a safety margin.

## 11. Summary of what this run established

1. **In-engine prompt-logprob scoring is unsafe** (OOM at 1,665 tokens with about 0.4 GB free), and it kills the primaries.
2. **Thinking-mined candidates contain no new correct answers** (0/31). E8 mining plus scoring cannot work.
3. **Hits come from natural closes** (60 %); forced closes almost never hit (about 2 %); many natural closes land right under the cap. This is consistent with raising the thinking budget, but it is observational: natural closes may simply be easier tasks.
4. **Several wrong answers are 1–10 cells from correct.** An independent second sample is the plausible attempt_2 route.
5. **Codex easy-first spent the deadline on tier 1 tasks** that Qwen mostly fails (7/37), while tier 2 hit 3/4. This supports capability-based ordering.
6. **KV and memory headroom figures** for the 86k plan: expect occasional preemption at 16 sequences, and almost no free memory beside the KV pool.
