# D66 and L66 deep runs: findings and decisions (2026-10-17)

Two 12 h GPU runs of Qwen3.8-27B FP8 Poetiq on the **66 eval tasks R5 did not fully solve** (103 test inputs; R5 had 9 of them right).

Sources:
- outputs: `experiments/Experiment 10 …/Experiment Results/Kaggle Runs/GPU HARD66 D66 77.8k …` and `… L66 122.9k …`;
- our analysis: `…/Kaggle Runs/D66_L66_analysis/`;
- GPT deep audit (2026-10-17): every number below was re-checked by Claude against the files unless marked *hypothesis*;
- GPT bundle: `Experiment Review/E10_D66_L66_RESULTS_FOR_GPT_2026-10-17.zip`.

## 1. Setup

| | D66 | L66 |
|---|---|---|
| Thinking cap | 77,824 (as R5) | 122,880 (`MAX_MODEL_LEN` 155,648) |
| Slots | 16 | 12 |
| Seeds (`SEED` / `POETIQ_SEED`) | 20261016 / 1 | 20261017 / 2 |
| Everything else | S1 code (ORDER2, F1–F3, A2, F2b, C1, KV gate 0.95, prefetch, 11:55 clock), T 1.0, 1 expert, ≤ 10 iterations, per-iteration test logging | same |
| Kaggle | foysalemonshanto/q27-xhigh-e10-d66-hard-deep v1 | …-l66-hard-120k v1 |

**The comparison is confounded:** thinking cap, seed and slot count all differ. It is a paired observation, not a one-variable ablation.

## 2. Verified results

| | D66 | L66 |
|---|---|---|
| **Inputs right, pass@2 (of 103)** | **28** | **34** |
| Fully solved tasks (of 66) | 16 | 18 |
| R5-wrong inputs now right (of 94) | 22 | 29 |
| R5-right inputs kept (of 9) | 6 | 5 |
| Requests (iterations 1 / 2 / 3) | 151 (66 / 52 / 33) | 123 (66 / 41 / 16) |
| Generated tokens / decode speed | 10.80 M / 255.8 tok/s | 10.66 M / 254.1 tok/s |
| Wall (solver) | 42,231 s | 41,968 s |
| Iteration 1 cut at the cap | 49 / 66 | 9 / 66 |
| Peak KV usage / samples > 0.95 | **99.88 % / 20 of 351** | 90.08 % / 0 of 349 |
| vLLM preemptions | **66** | 0 |
| Errors, unit errors, unharvested verifications | 0 | 0 |

**Per input:**

| | Inputs |
|---|---|
| Both right | 12 |
| Only D66 | 16 |
| Only L66 | 22 |
| Both wrong | 53 |

**Statistical test:** McNemar exact p = 0.42, so **L66 > D66 is not significant**. GPT's task-bootstrap 95 % interval for L66 − D66 is about −10 to +18 points.

## 3. Findings

### 3.1 Independent attempts are highly complementary (the biggest finding)

| | Inputs right (of 103) |
|---|---|
| D66 alone | 28 |
| L66 alone | 34 |
| D66 ∪ L66 (oracle) | **50** |
| R5 ∪ D66 ∪ L66 | 51 |
| attempt_1 of D66 and attempt_1 of L66 as a two-attempt pair | **48** |

- Two separate 12 h runs together solve far more hard inputs than either alone.
- Much of hard-task failure is **which trajectory the sample takes**, not a hard model limit.
- *Hypothesis:* diverse independent attempts are more coverage-efficient than one very long attempt.
- **Agreement as a signal:** the two first answers agreed on 18 inputs, and 12 of those were right (67 %). Agreement is a useful but imperfect selection signal.
- **Caveat:** pass@3+ cannot be estimated from two confounded runs; the union is retrospective (ground truth picks the right one).

### 3.2 attempt_2 is almost unused

| | D66 | L66 |
|---|---|---|
| attempt_1 right | 27 | 33 |
| Only attempt_2 right | **1** | **1** |
| attempt_2 = test input (A2 fallback) | 46 / 103 | 57 / 103 |

- The cause: 1 expert plus stopping at the first train pass, so most tasks never produce a second distinct program.
- The A2 fallbacks (test input or mirror) never score on these tasks: no hard solution equals its input or its mirror.
- **attempt_2 should come from a genuinely independent attempt.**

### 3.3 Train pass is a weak correctness signal on hard tasks

| | D66 | L66 |
|---|---|---|
| Train-perfect program executions | 33 | 43 |
| All test inputs right | 14 (42 %) | 15 (35 %) |
| Train-perfect but at least one test wrong | 19 | 28 |

- Of the 15 R5 train-pass/test-wrong tasks, 6 (D66) and 9 (L66) overfit again: 247ef758, 2b83f449, 3dc255db and others.
- **Long natural reasoning does not protect.** 88bcf3b4 (L66) passed train after more than 121k thinking and was wrong on both test inputs.
- **Train-failing programs are sometimes right:**

  | Correct inputs that came from… | D66 | L66 |
  |---|---|---|
  | only train-failing programs | 6 | 7 |
  | train-failing and train-passing programs | 0 | 2 |

  Examples: 8b7bacbf (train 0.993), b6f77b65 (train 0.999) and 20270e3b.
- **Consequence:** train pass should be a strong ranking signal, not a hard gate. Near-perfect candidates must stay in the pool, as S1's F3 history change partly does.
- **Code length:** train-perfect, test-wrong programs are longer in L66 (median 11,096 vs 6,545 characters for correct ones) and slightly longer in D66 (8,075 vs 7,439). A possible secondary signal, not validated.

### 3.4 Iterations

| Iteration | D66 tokens | D66 new right inputs | D66 per slot-hour | L66 tokens | L66 new right | L66 per slot-hour |
|---|---|---|---|---|---|---|
| 1 | 4.93 M | 12 | 0.14 | 6.23 M | 22 | 0.27 |
| 2 | 3.94 M | **15** | **0.22** | 3.64 M | 12 | 0.25 |
| 3 | 1.93 M | 1 | 0.03 | 0.80 M | 0 | 0.00 |

- **Iteration 2 (with feedback) is clearly worth it:** about as productive as iteration 1 per slot-hour, and in D66 more so.
- Iteration 3 gave almost nothing, but many iteration-3 requests were cut by the drain at the end. Its value is not settled; give it low priority after unseen tasks and iteration 2.

### 3.5 Thinking length

**Answers cut at the cap rarely succeed:**
- D66: 5 of the 49 iteration-1 requests cut at 77.8k were right.
- L66: 1 of the 9 cut at 122.9k was right.

**L66 iteration 1, by how the request ended:**

| How it ended | n | Train pass | Fully right |
|---|---|---|---|
| ≤ 49k natural | 6 | 5 | 3 |
| 49–78k natural | 12 | 7 | 4 |
| 78–123k natural | 39 | 12 | **3** |
| Cut at 122.9k | 9 | 1 | 1 |

- **Long natural closes are mostly wrong.** Most correct programs come from requests that close before about 78k.
- L66's advantage is not explained by the 78–123k closes alone. Seed and slot differences may account for part of it.
- **For the threshold decision:** a cap below 77.8k (such as S1-60K) would cut more of the requests that close at 49–78k, which are the productive ones. The evidence leans **against** going below 77.8k. A cap above 77.8k is not shown to be better; it is at least not worse per GPU-hour on hard tasks.

### 3.6 Engine and memory

- **12 slots at 155,648 `max_model_len` was safe:** KV peak 90 %, 0 preemptions, decode speed equal to 16 slots (254 vs 256 tok/s).
- **16 slots at 77.8k hit the KV ceiling:** 99.9 %, 66 preemptions (R5: 9). Throughput was no higher. For long-thinking Poetiq, 16 slots buy nothing over 12.
- **vLLM's "GPU KV cache size" line is one quarter of the real capacity** (hybrid model, 4 cache groups): R5 showed 337,600 against an engine JSON of 1,353,600. Use the "Maximum concurrency" line instead: about 1.1–1.18 M tokens.
- **Weight prefetch works:** weight load fell from 64 s (R5) to 4.8 s, and engine start from 600 s to 532 s.

### 3.7 Failure taxonomy of the 53 inputs both runs missed (GPT, from traces; categories are inference)

| L66's best train score on the input | Inputs | Examples |
|---|---|---|
| Train-perfect but test wrong | 24 | 247ef758, 2b83f449, 3dc255db, 88bcf3b4 |
| ≥ 0.98, not perfect | 9 | 7b80bb43, a25697e4 |
| 0.90–0.98 | 5 | 8b7bacbf, de809cff |
| Lower | the rest | — |

## 4. What is NOT established

- That 122k thinking by itself is better than 77.8k (confounded, p = 0.42).
- pass@3 or more, or any LB 60+ projection, from these two runs.
- That a ground-truth-free selector can recover the 48–50 union.
- How these hard-task results transfer to the hidden 240-task mix, where coverage is the limit. The 66 tasks are R5's hard subset.

## 5. Decisions and next steps, ranked

| Rank | Decision / next step | Evidence | Status |
|---|---|---|---|
| 1 | **Coverage first:** every task gets iteration 1 before any iteration 2 (ORDER2 + FIFO already does this); iteration 2 kept; iteration 3 only after unseen tasks and iteration 2 | §3.4 | In S1; iteration-3 deprioritisation is a candidate change |
| 2 | **Real second candidate:** attempt_2 from an independent attempt (fresh seed or a second expert), not the test-input fallback, using agreement or train fit to order attempt_1 and attempt_2 | §3.1, §3.2: union 48–50 vs 28–34; attempt_2 gives +1 now | Needs compute, so it pairs with E11 speed. Design after the E11-03 results |
| 3 | **Do not trust a single train pass on hard tasks:** keep near-perfect and passing candidates in the pool; when time allows, a second independent attempt even after a train pass | §3.3 | Candidate change |
| 4 | **Slots:** 12 is enough for long-thinking Poetiq; 16 gives no speed and causes preemptions | §3.6 | Use 12 (or test 13–14) in E11 sizing |
| 5 | **Thinking cap:** do not go below 77.8k. For the S1 submission, prefer 77.8k + S1 fixes over 60k (user's call; S1-60K's commit was cancelled) | §3.5 | Pending user decision |
| 6 | **Controlled ablations if needed:** 12 vs 16 slots at the same cap and seed, then 77.8k vs 122.9k at 12 slots, on a small paired subset | §2 confound | Optional |
| 7 | **E11 engine speed** (1.5–3×) is what makes ranks 1–3 affordable on 240 hidden tasks | coverage | E11-03 next |
| 8 | **Prompt / feedback work on overfit and near-miss failures** (24 train-perfect-wrong, 14 at 0.90–0.99) | §3.7 | Later, on a controlled subset |

## 6. Corrections to earlier notes

- An earlier partial-log reading said L66 KV peaked at 84.7 %. The **final peak is 90.08 %**, still with 0 preemptions.
- The D66 preemptions (66) were missing from our first analysis report. GPT found them; verified.
- The D66/L66 `submission.json` files are the 240-task placeholder (commit run). The 66-task scores come from `poetiq_eval_generated_submission.json` / `poetiq_final_generated.json`.

## 7. Feedback (iteration 2) vs a fresh attempt: when is which better? (Claude, verified from files)

**Method:**
- For every task that failed iteration 1 in run A, compare two outcomes on the same task:
  - **feedback:** A's iteration 2;
  - **fresh:** the other run B's independent iteration 1.
- Both directions were pooled (D66 → L66 and L66 → D66), then split by A's iteration-1 train result.

| A's iteration-1 result (task still wrong) | n | Feedback (iteration 2) fully right | Fresh attempt (other run, iteration 1) fully right |
|---|---|---|---|
| Train score < 0.80 | 45 | 6 / 45 (13 %) | 4 / 45 (9 %) |
| Train score 0.80–0.95 | 16 | 1 / 16 (6 %) | 0 / 16 (0 %) |
| **Train score ≥ 0.95, not passed (near miss)** | 30 | **8 / 30 (27 %)** | 5 / 30 (17 %) |
| **Train pass, test wrong (overfit)** | 23 | **never runs** (Poetiq stops at a train pass) | **5 / 23 (22 %)**, 9 with at least one input right |

**Reading:**
- **When train fails, feedback is at least as good as a fresh attempt.** It is clearly better for near misses (27 % vs 17 %); low-score tasks are similar (13 % vs 9 %). The costs per request are similar.
- **When train passes, feedback never happens, and a fresh attempt is the only route to the overfit 22 %.** It also gives an agreement signal for choosing attempt_1 and attempt_2: when two first answers agreed, 12 of 18 were right.
- **Policy implied:**
  - failed train → iteration 2 with feedback, near misses first;
  - train passed → one fresh independent attempt for attempt_2 when time allows.
- **Caveats:**
  - n is small;
  - "fresh" is the other configuration (cap, seed and slots differ);
  - some iteration-2 requests near the end were cut by the drain.

## 8. Complementary evidence from R5 (all 120 tasks), GPT 6-question analysis, re-verified by Claude

GPT's bundle held only R5 data, not the D66/L66 runs. Every number below was recomputed from `R5_units.csv` / `poetiq_iterations.jsonl`.

**Train-pass overfit depends on difficulty.**

| | Train-passing programs | All test inputs right | Partial | None right | At least one wrong |
|---|---|---|---|---|---|
| R5, all 120 tasks | 71 | 49 | 6 | 16 | **31 %** |
| D66 / L66, 66 hard tasks | 33 / 43 | 14 / 15 | — | — | **58–65 %** |

Overfit concentrates in hard tasks.

**A natural close is a reliability signal, not a rule.**

| R5 train-passing program | All test inputs right |
|---|---|
| Natural close | 36 / 48 (75 %) |
| Forced / continue | 13 / 23 (57 %) |

**Productivity by thinking length** (R5 iteration 1, correct inputs per M generated tokens):

| How the request ended | n | Correct inputs | Per M tokens |
|---|---|---|---|
| Natural ≤ 49k | 16 | 15 | 22.9 |
| Natural 49–65.5k | 24 | 24 | 16.7 |
| Natural 65.5–77.8k | 10 | 8 | 11.3 |
| Forced or continued at the cap | 70 | 16 | **2.9** |

- Short natural closes are the most token-efficient, but 48 of the 63 iteration-1 hits came from requests above 49k.
- This is observed productivity, not the effect of a lower cap. **Keep 77.8k** (consistent with §3.5).

**Feedback (iteration 2) by iteration-1 train score, R5: a U-shape** (small n).

| R5 iteration-1 train score | Iteration-2 requests | Tasks solved |
|---|---|---|
| 0 | 12 | 4 |
| 0–0.75 | 11 | **0** |
| 0.75–1 (not passed) | 29 | 8 |

- R5 iteration 1: 63 hits from 8.41 M tokens (7.5 per M).
- R5 iteration 2: 15 hits from 3.24 M tokens (4.6 per M), on harder tasks by selection.
- In D66/L66 (§7), near misses gained most from feedback, matching the 0.75–1 row.

**Coverage arithmetic for E11** (R5 iteration-1 tokens ×2 for 240 tasks, R5's 281 tok/s as 1×, 15 min reserve; uniform-speed assumption):

| E11 speed-up | First pass over 240 tasks | Time left | Rough extra iterations |
|---|---|---|---|
| 1.0× | 16.6 h (does not fit) | — | 0 |
| 1.5× | 11.1 h | 0.7 h | about 16 |
| 2.0× | 8.3 h | 3.4 h | about 110 |
| 2.5× | 6.7 h | 5.1 h | about 200 |
| 3.0× | 5.5 h | 6.2 h | about 300 |

- Even a single full pass needs **≥ 1.5×**.
- Iterations, a second independent attempt and the union lever (§3.1) become affordable only from about 2×.
- This makes the E11 speed-up the gate for everything else.

**GPT's priority (agreed):** first-pass coverage → recover train-verified / usable outputs → a distinct second candidate → feedback iterations of measured value.

Before relying on the 48–50 union, measure a label-blind selector: agreement, natural close, train fit, code size.

## 9. LB result that overrides §3.5 / §5 rank 5 (2026-10-17)

**S1-60K = LB 34.58** (foysalemonshanto, Version 2), against R5 21.39 (77.8k) and NVARC 32.64.

| | R5 | S1-60K |
|---|---|---|
| LB | 21.39 | **34.58** (+13.2) |
| Thinking cap | 77.8k | 60,000 |
| Other changes | — | ORDER2, the S1 fixes, KV gate, prefetch, +11 min clock |

- The two runs differ in more than the cap, so this is confounded.
- On the hidden 240-task run, **coverage dominates**: fewer tokens per task means more tasks reached.
- §3.5's "do not go below 77.8k" is valid only for hard tasks with time to spare. **It is wrong as an LB policy.**
- The next threshold question is 49,152 (the floor) with the S1 fixes, decided on the LB.
