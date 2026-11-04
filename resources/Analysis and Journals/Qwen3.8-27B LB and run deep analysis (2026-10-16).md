# Qwen3.8-27B: every LB submission and every GPU run, analysed together (2026-10-16)

**Sources**
- The six Qwen3.8-27B LB submissions. The exact submitted source was pulled from Kaggle; all six are byte-identical (code cells) to our local copies.
- 13 GPU commit runs with data (19 run folders), 655 requests, the Puzzle Capability Ledger, the full team submission history (90 submissions), and the archive phase.
- Working files are in `scratchpad/agent_lb_code/` (rerun-path code trace, line-cited) and `scratchpad/agent_runs/` (`runs.csv`, `units.csv`, `timeline.csv`, scripts).

**Premise (user correction, 2026-10-16):** the hidden test is about as hard as the public evaluation set.
- The earlier claim that hidden is easier came from the placeholder file and was wrong.
- So LB ÷ eval at the same cap reads directly as the share of hidden work finished inside 12 h.
- Eval-measured times scale to the about 240 hidden tasks.

---

## 1. The LB history and how much of it is noise

| Date | Notebook (account) | Method | Thinking cap | LB | LB × 1.2 = tasks solved (of 120) |
|---|---|---|---|---|---|
| 10-03 | PIV-Q27-SUB v1.3 (esfer) | REPL program synthesis, low effort | 8,192 per task | 1.67 | 2.0 |
| 10-06 | Q27-XHIGH v1.4 (foysal) | direct grid, one request per test input | 49,152 | 18.75 | 22.5 |
| 10-10 | Q27-XHIGH v1.4.4 (foysal) | = v1.4 + prefix cache + later clock | 49,152 | 26.94 | 32.33 |
| 10-12 | E7 ordering v2 (foysal) | = v1.4.4 + easy-first queue | 49,152 | **27.92** | **33.5** |
| 10-14 | E9 (esfer) | direct + tail resample, ordering dropped | 77,824 | 22.92 | 27.5 |
| 10-15 | E10 Poetiq R5 (hosen42) | Poetiq program synthesis, ≤10 sequential iterations | 77,824 per iteration | 21.39 | 25.67 |

- **The public LB is exactly 120 tasks.** Every score × 1.2 is a whole number or a ½, ⅓ or ⅙ (multi-test credit). That holds for all 90 team submissions, e.g. 32.64 × 1.2 = 39.17.
- The hidden rerun scores public and private together, so the rerun sees **about 240 tasks**. Our EVAL120 runs had 120.
- **Noise, measured:** identical notebooks submitted repeatedly scored an sd of 1.0–1.65 LB points.
  - The NVARC notebook v331102752 was submitted 10 times and scored 27.36–32.64, sd 1.65.
  - Five other notebooks submitted 3–6 times had sd 0.7–1.5.
  - So one submission is ± about 1.5; a difference of 2 submissions has sd about 2.1.
- **What that means for each step:**
  - v1.4 → v1.4.4 is +8.2: **real**.
  - v1.4.4 → E7 is +1.0: **inside the noise**. Ordering is not proven by the LB, though its direction is right.
  - E7 → E9 is −5.0 and E7 → R5 is −6.5: about 2.5–3 sd, **very likely real**.
  - E9 → R5 is −1.5: noise.

## 2. What each version changed in the rerun path (code-verified)

Clock times are measured from cell 0. Kaggle's 12 h also counts container start and mounts before that.

| Field | v1.4 | v1.4.4 | E7 | E9 | R5 |
|---|---|---|---|---|---|
| Admission stops | **10:30** | 11:34 | 11:34 | 11:34 | 11:34 |
| Drain (forced close) | 11:06 | 11:36 | 11:36 | 11:36 | 11:36 |
| Hard stop | 11:35 | 11:44 | 11:44 | 11:44 | 11:44 |
| Prefix cache | **off** | on | on | on | on |
| Queue order | sorted ids | sorted | **easy tier first** | sorted | sorted; a failed task goes to the back |
| Thinking cap | 49,152 | 49,152 | 49,152 | **77,824** | **77,824 per iteration** |
| Sampling T | 0.6 | 0.6 | 0.6 | 0.6 | **1.0** |
| attempt_2 | test input / mirror | same | same | resample grid if one finished | 2nd distinct program grid, else **a copy of attempt_1** |

**Common to every Q27 version:**
- 16 slots, MTP-3, fp8 KV (1,353,600 tokens), 4,096 batched tokens, gpu_mem 0.90.
- The forced close is token-exact: strip EOS, append the stop sentence and `</think>`, then answer with ≤ 8,192 tokens.
- About 12–13 min of setup before scheduling starts.

**The LB moves, read against the code:**
- **v1.4 → v1.4.4 (+8.2).** The only rerun changes are more usable time:
  - admission runs to 11:34 instead of 10:30;
  - prefix caching means a forced close no longer re-prefills about 50k tokens (about 12 % of scheduler time in v1.4.1).
  - Prompt, budget, sampling, parser and attempts are identical.
- **E7 → E9 (−5.0).** Three changes were bundled:
  - the cap went to 77,824 (+38 % mean tokens per input on EVAL12);
  - ordering was dropped;
  - a tail resample was added that never fired (0 hits).
- **E9 → R5.** Poetiq at T = 1.0 with sequential iterations. On EVAL120 it finished the 120 first iterations at about minute 540 and 52 second iterations by minute 703.

## 3. The central finding: the LB is limited by how many hidden inputs get a good answer inside 12 h

**Evidence 1: more usable time gave the biggest LB gain we have ever had.** v1.4 → v1.4.4 changed only the clock and the prefix cache, and gained +8.2.

**Evidence 2: a bigger per-request budget raised eval and lowered the LB.**

EVAL120, FP8 DIRECT (ORDER2 run), with the score a 49k cap would have kept (hits that needed more thinking count as lost):

| Cap | Eval task-normalised |
|---|---|
| 32,768 | 13.75 % |
| **49,152** | **25.42 %** (30.5 / 120) |
| 60,000 | 27.92 % |
| 70,000 | 30.42 % |
| **77,824** | **36.94 %** (44.33 / 120) |

| Run | Eval, same cap | LB | LB ÷ eval |
|---|---|---|---|
| 49k DIRECT (v1.4.4 / E7) | 25.4 % | 26.94 / 27.92 | **1.06–1.10** |
| 86k DIRECT (E9) | 36.9 % | 22.92 | **0.62** |
| Poetiq 86k (R5) | 48.6 % | 21.39 | **0.44** |

**Hidden ≈ eval difficulty (user, 2026-10-16), so LB ÷ eval is the share of the hidden run that got finished:**
- the 49k runs finished essentially everything;
- the 86k DIRECT run finished about **62 %**;
- Poetiq finished about **44 %**.

The rest is score the method reaches on puzzles of this difficulty but that the 12-hour run never delivered.

**Evidence 3: the work per pass doubled, the clock did not.**
- Both EVAL120 runs needed most of the 12 h for 120 tasks: ORDER2 86k DIRECT finished at minute 534, and R5 at minute 704.
- The hidden rerun has about 240 tasks in the same clock.
- Full-cap demand per pass (an upper bound, not a forecast):

| Config | Full-cap demand per pass |
|---|---|
| 49k direct | 259 × 57,344 ≈ 14.9 M tokens |
| 86k direct | ≈ 22.3 M tokens |
| R5, first iteration only | ≈ 20.6 M tokens |

- Engine capacity up to the 11:34 admission stop is about 11.5–12.5 M tokens. R5 actually produced 11.65 M.

**Time model.** Each EVAL120 run's own pace is scaled to 240 tasks. Scheduling time available is about 680 min: admission to 11:34, minus about 13 min of setup.

| Method (EVAL120-measured) | Eval task-normalised | Min per 120 tasks | Min needed for 240 | Coverage at today's speed → expected LB | Speed-up needed for full coverage → expected LB |
|---|---|---|---|---|---|
| 49k DIRECT (score at a 49k cap from the ORDER2 traces) | 25.4 % | ~386 | ~773 | 88 % → ~22 (actual 26.9 / 27.9) | 1.14× → ~25 |
| 86k DIRECT (ORDER2) | 36.9 % | 522 | 1,044 | 65 % → ~24 (**actual E9 22.9**) | **1.54× → ~37** |
| Poetiq iteration 1 only | 39.4 % | ~528 | 1,056 | 64 % → ~25 (**actual R5 21.4**) | 1.55× → ~39 |
| Poetiq R5 (iterations 1 + 2) | 48.6 % | ~692 | 1,384 | 49 % → ~24 | **2.04× → ~49** |
| Poetiq iteration 1, plus DIRECT on the 65 tasks iteration 1 did not train-verify | 51.8 % | ~868 | 1,736 | 39 % → ~20 | **2.55× → ~52** |
| Both methods on every task (ORDER2 ∪ R5 as attempt_1 / attempt_2) | **56.8 %** | ~1,214 | 2,428 | 28 % → ~16 | **3.57× → ~57** |

**Limits of this model:**
- It is linear and rough.
- The 49k eval score is inferred from ORDER2 traces, not measured. Easy-first ordering also moves the coverage loss onto hard tasks, which is why the 49k runs beat their line.
- Each eval score is a single run, ± a few points.
- The 86k and Poetiq predictions land within about 1–4 points of the real LB, so the model is usable for planning.

**Conclusion:** what moves the LB is score delivered per hour of the 12-hour clock.
- A method's eval score only counts for the inputs it reaches.
- The same Qwen3.8-27B already reaches 36.9 % (DIRECT 86k) and 48.6 % (Poetiq) task-normalised on the hard eval set.
- So making those methods fit 240 tasks in 12 h is the direct route to a much higher LB: 1.5× speed gives about 37, 2× gives about 49, and 3.6× gives about 57 (the two-method union).

## 4. Where the tokens go (measured)

**ORDER2, EVAL120 86k DIRECT, 10.45 M tokens:**
- 75.4 % of tokens went to units that were not solved.
- Forced or continued closes: 75 units, 6 hits (**8 %**), 56.6 % of all tokens. The unsolved ones alone took 52.1 %.
- Natural close at ≤ 49k: **39 / 54 hits (72 %)**.
- Natural close at 49–77.8k: 13 / 43 hits (30 %).
- About 180k tokens per solved unit.

**Pooled over 9 runs with per-unit data (32.5 M tokens):**
- 76.8 % of tokens produced no hit.
- **59.7 % went to forced closes that produced no hit.**

**The 49k EVAL12 runs** spent 89–98 % of tokens on unsolved units, 81–87 % on unsolved forced closes alone. EVAL12 is our hardest subset.

**Thinking needed by solved units:**

| Run | ≤ 32k | ≤ 49,152 | ≤ 65,536 | ≤ 77,824 |
|---|---|---|---|---|
| ORDER2 DIRECT | 34.5 % | **67.2 %** | 79.3 % | 100 % |
| Poetiq R5 | 3.8 % | **23.1 %** | 61.5 % | 100 % |

- Poetiq needs long thinking: 77 % of its solves came from iterations that thought past 49k.
- 28 % of Poetiq solves came from iterations that were force-closed and still wrote a passing program. A forced Poetiq iteration is worth more than a forced DIRECT answer: 19 % vs 8 %.

**Poetiq R5, iteration 1 vs iteration 2:**

| | Iteration 1 | Iteration 2 |
|---|---|---|
| Requests | 120 | 52 |
| Solved units | 63 | 15 |
| Share of tokens | 72 % | 28 % |
| **Model minutes per solved unit** | **7.8** | **13.3** |
| Wall clock | 12–540 min | 462–703 min |

13 tasks failed iteration 1 and never got a retry.

## 5. Throughput facts (all runs: vLLM 0.19, 4×L4, FP8, MTP-3, ≤ 16 sequences)

| Active sequences | 3 | 6 | 12 | 16 |
|---|---|---|---|---|
| DIRECT aggregate tok/s | 107 | 203 | 302–325 | **351** |
| Per sequence | 36 | 34 | 25–27 | **22** |

- Going from 6 to 16 sequences gave 1.73×. **We have never run above 16.**
- Live context costs about −5 % per +10k tokens per sequence: 418 tok/s at 0–10k, 330 at 50–60k.
- MTP tokens per step stay at about 3.0 at any length.

**Poetiq R5 ran about 16 % slower:**
- T = 1.0 lowers MTP acceptance to 0.557 (2.67 tokens per step instead of 3.0), which costs about 11 %.
- KV hit 99.5 %, which evicted the prefix cache. R5 prefilled 4.18 M prompt tokens against 1.14 M for ORDER2, and caused 9 preemptions. Heavy-prefill windows decoded at 138 tok/s.

**Slot utilisation:**
- 72–73 % in the 49k EVAL12 runs (a long tail below 16 active);
- 95.6 % in ORDER2;
- 99.9 % in R5.

## 6. Run-to-run variance is large, and it is an asset

- Seeds are fixed, yet traces differ every rerun: each rerun behaves as an independent sample.
- **EVAL120: ORDER2 solved 58 and R5 solved 78, but their union is 92.** Only 44 units were solved by both.
- **EVAL12:** six DIRECT runs averaged 2.67 hits, with a union of 5.
- **A natural close predicts the hit.** On units DIRECT ever solved, natural-close runs hit 39 / 44 (89 %) and force-closed runs 3 / 21 (14 %).
- 9 of the 13 flipping units hit only when thinking closed naturally.

## 7. The second attempt is currently wasted

- **Direct versions (v1.4 → E9):** attempt_2 is the test input or its mirror, which is almost never correct on ARC-AGI-2. Only E9's resamples changed it, and they gave 0 hits.
- **R5:** when only one usable program exists, attempt_2 = attempt_1. That is verified in code (`poetiq_two_attempts`).
- **Measured value of a real second answer:**
  - the 92-vs-78 union and the 5-vs-2.67 union above;
  - in R5, on tasks whose iteration 1 failed train, one DIRECT trace matched Poetiq's iteration 2 (21 vs 20 units) with little overlap (union 30).

## 8. Bugs and risks in the rerun paths (code-verified items marked ✔)

1. ✔ **R5 gives no grid when its code fails.** A program that doesn't parse, or crashes on any test input, leaves the task on the test-input fallback. Multi-test tasks lose every query.
2. ✔ **R5's queue starves refinement.** A failed task is re-queued at the back, so every iteration 2 waits behind all first iterations.
3. ✔ **R5 drops pending verifications.** Those still pending at the 11:44 hard stop are never committed, because the wait deadline is already past.
4. **R5 sandbox self-test failure empties the engine ladder,** which would score 0. It passed this time.
5. **R5 runs the first ```python block,** so a short illustrative snippet written before the real solution would be the one executed.
6. **Late admissions (11:00–11:34) get only minutes before the 11:36 drain.** With sorted order (v1.4.4, E9, R5) some of them are easy inputs.
7. **All deadlines are measured from cell 0,** not from container start.
8. **86k KV headroom:**
   - R5 hit 99.5 % KV and 9 preemptions;
   - E9 at full length would need about 1,136 blocks vs 846;
   - E8 crashed with a CUDA OOM at minute 124 and lost 15 units (445k tokens).
9. **PIV v1.3** (LB 1.67) had three design problems:
   - lock-step 12-task waves;
   - no MTP, bf16 KV;
   - an 8k low-effort budget, and a 2-task gate that could block the whole run.

## 9. Archive phase (09-13 → 10-04), Experiments 2–5, and public reference notebooks

The full digest, with file paths for every claim, is `scratchpad/agent_archive/archive_qwen_digest.md`.

**Every code / agent / evolution / repair rejection in the archive ran at a budget far below Qwen's natural-close length.**

| Attempt | Budget | Result |
|---|---|---|
| Q1 direct v1.2, PROXY48 | 12,288 | 0/56; thinking never closed |
| PIV v1.7 | 12,288 | 0 tool calls; Qwen calls tools only after `</think>` |
| ACT v1.9 | 3,072 per turn | 0 programs |
| C1 v2.4G | 8,192 exploration | 16/16 programs, 0/16 matched train |
| EVAL120 / SUB v1.3 | 6,144 + 2,048 | 1.67 % = LB 1.67; 38/120 finaliser outputs truncated at 2,048 |
| EVO12 | medium effort, 3,072 thinking | 0/12; 0/80 closed on their own |
| E2 R6 | repair turns capped at 12,288 | 2/22; 46/47 turns forced |
| E4 REPL agent | 6,144 per turn | 0/22; all 82 submits forced, 59 empty |
| E5 v1.9.1 | `thinking_token_budget` | 1/22; vLLM 0.19 + MTP silently ignores the budget (vLLM #39573) |

- Qwen3.8 hits come almost only from thinking that closes on its own, at about 26–45k tokens. Those negatives say the budget was too small, not that code-verify fails.
- E10 Poetiq at 77,824 per iteration proved it: 78/172 vs 58/172 for DIRECT on EVAL120.
- **The training-copy placeholder test file is a poor proxy.** SUB v1.3 scored 43.6 % on it and 1.67 on the LB.

**Public notebooks documented in the workspace (all below our 27.92):**
- **NguyenThanhNhan, 23.33 / 22.50** (screenshot only):
  - pure reasoning, about 65k tokens, forced commit after 49k;
  - SGLang + EAGLE (3 steps, 4 draft tokens), fp8 KV, mamba radix cache extra_buffer, chunked prefill 2048.
- **madarshbb, 21.5, then 13 on a rerun:**
  - one prompt per task, T 1.0;
  - attempt_2 = an independent second sample, run after every task has its first sample, cheapest first;
  - grids written with explicit (R×C) size.
- **russcore, 10.14:** our current free-form prompt, T 0.6, bf16, no MTP, 24 concurrent, attempt_2 = mirror.
- **ARC Prize official run:** xhigh 42.4 % semi-private, 40/120 public eval, about 173k tokens per task.
- **Not in the workspace:** the recipe of the team at about 60. Finding it would be valuable.

**Technical facts confirmed across phases:**
- `reasoning_effort` accepts only xhigh, medium or low; "high" raises an error, and medium or low do not keep thinking short.
- vLLM 0.19 + MTP ignores `thinking_token_budget`, `min_p` and `logit_bias`.
- Forced `tool_choice` gives empty code or HTTP 500.
- `prompt_logprobs` ran out of memory (E8).
- The MTP drafter needs `max_num_batched_tokens` 4096.
- Prefix caching needs mamba `align` mode.
- Fixed waves wasted 29 % of wall time.

**Second answers tried so far:**
- same-reasoning second answer (E6): +0;
- grids mined from thinking with NLL scoring (E8): truth in 0/31;
- tail resample (E9): 0 hits.

Section 6 shows where the value of a second answer lies: in an **independent** sample or a **different method**.

**Never run yet, or dropped for infrastructure or sequencing reasons only** (all keep ≥ 49k thinking):
- Poetiq round 1 plus a DIRECT grid for tasks that are not train-verified (55.1 % computed on EVAL120);
- repair with its own full budget;
- K candidates with a verified pick;
- temperature A/B, 0.6 vs 1.0;
- (R×C) or CSV grid encoding;
- prompts that stop hand-simulating grids (22.7 % of thinking is digit tokens);
- concurrency above 16;
- SGLang + speculative decoding + mamba radix cache (Experiment 11);
- the NVARC ∪ Qwen union or routing;
- LOOPSTOP (built and harness-tested).

## 10. What this means for reaching a high LB with Qwen3.8-27B (how, with the data)

1. **Throughput is the master lever. Experiment 11 is the right next step.** Section 3 gives the targets: 1.54× for 86k DIRECT (~37), 2.04× for Poetiq R5 (~49), 2.55× for Poetiq + DIRECT fallback (~52), 3.57× for both methods (~57).
   - Every point of LB lost at 86k is answered-input coverage, and both 86k methods are far stronger per input than 49k on eval.
   - Levers with measured or verified headroom:
     - **more than 16 concurrent sequences** (never tried; 6 → 16 already gave 1.73×);
     - **INT4 weights,** which free KV and avoid R5's 99.5 % KV and the E8 OOM;
     - **DFlash2,** with a block of 8;
     - **MTP at the temperature in use:** T = 1.0 cost about 11 % of MTP speed;
     - **keeping the prefix cache from being evicted.**
   - The E11-03 GPU validation measures exactly these.
2. **Tokens on forced closes are the largest pool to recover.**
   - 59.7 % of all generated tokens went to forced closes that produced no hit.
   - A forced DIRECT answer hits 8 %; a natural close at ≤ 49k hits 72 %.
   - Ways to use this, for the user to choose:
     - detect non-converging or looping thinking early (LOOPSTOP is built and harness-tested);
     - for Poetiq, keep the cap: forced Poetiq iterations still pass 19 %.
3. **Make attempt_2 a real answer.** It is wasted in every submitted version, and run-to-run union data shows a second independent answer is worth a lot. How to source it without new time is a design choice (see the R5 analysis): for example a DIRECT grid when Poetiq fails train, or the second-ranked distinct program.
4. **For Poetiq, fix the rerun-path losses before any new method:**
   - a direct-grid fallback when the code fails;
   - attempt_2 ≠ attempt_1;
   - re-queue order;
   - commit pending verifications at the hard stop.
   - Iteration 1 costs 7.8 model-minutes per solve, the cheapest solve source we have measured.
5. **Keep easy-first ordering.** It is not proven by the LB (+1.0 is inside the noise), but it is free.
6. **Each LB result is ± about 1.5.** Judge single-submission changes under about 3 points with that in mind.
