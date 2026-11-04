# Coverage-Bound Test-Time Compute: Running Qwen3.8-27B on ARC-AGI-2 within 4× L4 and 12 Hours

*[PENDING: author list and affiliations]*

*Draft, 2026-10-24. Not peer-reviewed. Every number is taken from our run logs and analyses; `SOURCES.md` maps each one to its internal source. Results still running are marked [PENDING].*

## Abstract

**Setting.** The ARC Prize 2026 Kaggle track for ARC-AGI-2 gives each submission 4× NVIDIA L4 GPUs (24 GB, PCIe), no internet and 12 hours to solve 240 hidden tasks. We study how far a single open-weight reasoning model, Qwen3.8-27B, can go under this budget.

**Central finding: the score is bounded by coverage.** We compared each method's leaderboard (LB) score with its own score on the 120-task public evaluation set (EVAL120) at the same cap:

| Method | LB ÷ eval |
|---|---|
| 49k-token direct reasoning | 1.06–1.10 |
| 86k direct reasoning | 0.62 |
| Poetiq-style program synthesis | 0.44 |

So better methods can score lower on the LB. Poetiq reached 48.61 % task-normalised on EVAL120 but only 21.39 on the LB. A simple linear time model built from these runs predicts LB scores within about 1–4 points.

**What we report:**
- an LB history of 11 scored Qwen3.8-27B submissions, from 1.67 to 34.58;
- a controlled comparison of three solver families on the same model:
  - direct long thinking;
  - program synthesis with train verification;
  - a REPL tool agent adapted from the Duck harness;
- evidence that independent attempts are highly complementary: 28 and 34 inputs alone vs a union of 50, on 103 hard inputs;
- serving measurements on 4× L4. About half of each decode step (59 of 116 ms) is NCCL communication, and speculative decoding with DFlash2 gives 1.49× over target-only;
- an agent-side prefix-cache bug (blank lines removed from stored reasoning) that caused re-prefill in 45 % of requests;
- a catalogue of platform failure modes and negative results.

Our best Qwen3.8-27B submission scores 34.58 on the public LB. Our current submission is pending.

## 1. Introduction

ARC-AGI-2 asks a solver to infer a grid transformation from 2–5 demonstration pairs and apply it to new inputs, with two attempts per test input. Large reasoning models solve a sizeable share of these tasks with long chains of thought. ARC Prize's verified run of Qwen3.8-27B at `reasoning_effort` xhigh scored 42.4 % on the semi-private set. From ARC Prize's cost data we estimate about 60k output tokens per request.

The Kaggle competition setting changes the question. A submission is a notebook that must, offline and inside 12 hours on 4× L4:
- install its own serving stack;
- load the model;
- solve 240 hidden tasks: the 120 public-LB tasks plus the private ones.

Under this budget, *throughput and allocation* decide how much of the model's ability is realized.

This paper documents our team's path through that problem over about six weeks (2026-09-13 → 2026-10-24). It is written as an engineering record. We report what we measured, what did not work, and where our own earlier conclusions were wrong.

### 1.1 Contributions

1. **A coverage model for compute-bounded ARC evaluation.**
   - LB ÷ eval ratios for three methods, under the user-verified premise that hidden difficulty ≈ public evaluation difficulty.
   - A linear time model that turns each run's measured pace into required speed-ups: 1.54× / 2.04× / 2.55× / 3.57× for four method mixes (§3).
2. **A same-model comparison of three solver families** on EVAL120 and fixed subsets (§5.3–5.6):
   - direct reasoning with a host-forced commit;
   - a Poetiq port;
   - an ARC-2 adapter for the Duck REPL agent.

   It includes per-run unions showing large sampling complementarity.
3. **Serving measurements on a PCIe-only 4× L4 node** (§5.7):
   - a per-category profiler split of the decode step;
   - NCCL transport benches;
   - speculative-decoding width arms;
   - root causes of two concurrency losses: a DFlash admission delayer, and a probe of ours.
4. **An agent-side prefix-cache root cause** that also exists in the original Duck harness, and its fix (§5.8).
5. **Platform findings** that cost more GPU time than any model bug, with their fixes (§6.2).
6. **Negative results**, with the conditions under which they were measured (§6).

## 2. Background and related work

**ARC-AGI-2 and the ARC Prize Kaggle track.**
- The public LB scores a fixed set of 120 tasks. Every one of our team's ~90 LB scores × 1.2 is a whole number or a simple fraction.
- Per the organisers' statements quoted in discussion, the LB tasks are the semi-private set used for ARC Prize's verified leaderboard.
- The hidden rerun covers 240 tasks.
- One submission may be pending per team, and turnaround is 12–16 h.

**Qwen3.8-27B.**
- The chat template accepts `reasoning_effort` ∈ {xhigh, medium, low}. xhigh adds an instruction to think carefully and consider alternatives, low adds "keep your thinking brief", and medium adds no instruction.
- ARC Prize's verified semi-private scores: xhigh 42.4 %, low 22.8 %, medium 13.2 %, none 1.5 %.
- The model emits tool calls only after `</think>`. This shaped every agent design below.

**Program synthesis with verification.** Poetiq's open solver has the model write a `transform` function, executes it on the training pairs, and iterates with cell-level feedback. Its configuration for an open-weight model is cheap enough to port to a local engine.

**REPL agents.** Tufa Labs' Duck harness (Jeroen Cottaar) won the first ARC-AGI-3 milestone. A model plays through a Python tool, with preserved thinking, many short requests, function retention and a priority scheduler. Daniel Franzen's milestone-2 fork extended it, and his write-up reports that removing retained thinking is a large regression.

**Test-time training baselines.** NVARC's Qwen3-4B pipeline was the strongest public notebook family at the time (32.64 for our copy). It contributed the `get_puzzle_priority` task ordering we reuse.

**Candidate selection.** Land (arXiv 2606.31543) generates candidates across text, image and code channels and selects them with a holistic judge. He reports that prescriptive templates and iterative refinement reduce hypothesis diversity. His results use frontier API models at $19.69–$38.99 per task. Our budget is a fixed local GPU slice; the diversity argument transfers, the cost model does not.

**Speculative decoding and serving:**
- vLLM 0.19 with the model's MTP head;
- SGLang 0.5.21 with z-lab's DFlash2 block drafter on RedHatAI's W4A16 INT4 checkpoint.

The model is a hybrid with GDN (linear-attention) layers. Prefix caching therefore depends on Mamba-state checkpoints, not only on KV blocks.

## 3. Problem setting: compute-bounded evaluation

**Hardware:**
- 4× L4 (24 GB each), PCIe; every GPU pair reports PHB, and P2P was measured 5.5× slower than the default NCCL path;
- 48 vCPUs and ~198 GB RAM in our runs.

**Run budget.** The 12 h limit includes container start and input mounts. From the clock fixes onward, our notebooks measure deadlines from kernel start.

**Premise.** We treat the hidden set as about as hard as the public evaluation set. An earlier belief that hidden tasks are easier came from the placeholder test file (240 training tasks) and was wrong.

**Coverage.** Under that premise, LB ÷ eval at the same cap estimates the fraction of hidden work finished (Table 2). The pace of the long methods on EVAL120:
- 86k direct finished 120 tasks in 534 min;
- Poetiq R5 ran its first iteration on all 120 tasks by ~540 min and stopped at ~704 min.

The rerun must do 240 tasks in about 680 scheduling minutes.

**Time model.** Scale each EVAL120 run's measured pace to 240 tasks and compare with the available minutes:

| Method (EVAL120) | Eval task-norm | Coverage at our speed | Speed-up for full coverage | Expected LB at full coverage |
|---|---|---|---|---|
| 86k direct | 36.9 % | 65 % (actual LB 22.9) | 1.54× | ~37 |
| Poetiq R5 (iterations 1 + 2) | 48.6 % | 49 % (actual LB 21.4) | 2.04× | ~49 |
| Poetiq iteration 1 + direct on unverified tasks | 51.8 % | 39 % | 2.55× | ~52 |
| Both methods on every task | 56.8 % | 28 % | 3.57× | ~57 |

The model is linear and rough, and each eval score is one run. Still, its 86k and Poetiq predictions landed within ~1–4 points of the real LB. We used it to rank levers.

**Noise.** Identical notebooks resubmitted scored with an sd of 1.0–1.65 LB points. Our team's NVARC notebook was submitted 10 times and scored 27.36–32.64. We treat single-submission differences under ~3 points as unresolved.

## 4. Methods

### 4.1 Serving stacks

**S-vLLM (Experiments 1, 6–10):**
- vLLM 0.19 offline engine, TP 4;
- FP8 weights (our Kaggle repack of Qwen3.8-27B-FP8), fp8 KV (pool 1,353,600 tokens);
- MTP ×3, 16 sequences;
- `max_num_batched_tokens` 4,096;
- prefix caching with Mamba `align` mode.

Speed: 351 tok/s aggregate at 16 sequences on direct reasoning; ~16 % lower at T = 1.0, from lower MTP acceptance (0.557) and prefix-cache eviction.

**S-SGLang W1 (Experiments 11–14):**
- SGLang 0.5.21, TP 4;
- RedHatAI/Qwen3.8-27B-INT4 (W4A16 g128) + z-lab DFlash2 (BF16 drafter, `unquant`, block 8);
- fp8 KV; Mamba `extra_buffer_lazy` with 56 cache slots; SSM fp16;
- `mem-fraction-static` 0.83; no prefill CUDA graph;
- server max-running 13, client 12.

Throughput: 416.5 tok/s in a 12-stream production loop; KV 881k tokens.

**S-SGLang E14S (current submission):** W1 with DFlash2 width 6 (KV 917,816), plus decode CUDA graphs for batch sizes 1..13.

**Offline installation.** A 220-wheel wheelhouse (torch 2.13 cu130, flashinfer 0.6.18, transformers 5.12.1) is installed into a venv in `/tmp` by our own installer, which restores file names and checks hashes (§6.2).

### 4.2 Solvers

**DIRECT (Experiment 1).** One request per test input:
- xhigh thinking up to a cap (49,152 or 77,824);
- if the cap is hit, the host strips EOS, appends Qwen's stop sentence and `</think>` token-exactly, and asks for ≤ 8,192 answer tokens;
- prompt: free-form grids, "give ONLY the final output grid";
- T 0.6, top_p 0.95, top_k 20;
- attempt_2: the test input or its mirror (a placeholder).

**Poetiq (Experiment 10).**
- The model writes `transform(grid)` (numpy allowed).
- A sandbox runs it on every training pair with a timeout. It is a subprocess with rlimits, seccomp and a uid drop, because the container blocks `unshare` and `mount`.
- Feedback per example (wrong cells as pred/correct, shapes, traces, soft score) and the best ≤ 5 earlier solutions seed the next iteration.
- The loop stops at the first train pass.
- R5 setting: 1 expert, ≤ 10 iterations, T 1.0, 77,824 thinking + 8,192 answer per iteration.

S1 changes the rerun path and allocation:

| Change | What |
|---|---|
| Thinking | 60,000 |
| ORDER2 queue | score = [≥ 3 train pairs] + [1 test input], higher first; all iteration-1 requests before any iteration 2 |
| F1 | execute the last ```python block defining `transform` |
| F3 | per-input attempts |
| A2 | attempt_2 ≠ attempt_1. R5 had repeated attempt_1 on 134 of 172 inputs |
| F2 / F2b | harvest pending verifications at the hard stop; cancel queued ones |
| C1 | per-unit error isolation |
| KV gate | no new request at ≥ 0.95 KV usage with ≥ 12 running |
| Weights | prefetch |
| Clock | from kernel start: admit 11:45, drain 11:47, hard stop 11:55 |

**Duck ARC-2 adapter (Experiment 12).**
- **Vendored unchanged:** the Duck `inference/` package from Franzen's fork (commit 10882e33, Apache-2.0).
- **Our adapter:**
  - an ARC-2 system prompt;
  - a trusted prelude with the task grids and `check(transform)`, which returns per-pair pass/fail and cell diffs;
  - `submit` through Duck's JSON action channel, re-verified by the host in a clean sandbox;
  - an ARC-2 nudge text;
  - two branches per task, selected by host-verified train pass, then agreement, then distinct second grids.
- **Sandbox:** Duck's own (no numpy; pure-Python lists).
- **Rev 5 added:**
  - a check()-exact harvest hook;
  - tokenizer-based context estimates;
  - a truthful resume message;
  - invalidation of length-cut tool calls;
  - T 1.0;
  - 49,152 output per request.
- **Rev 6 → 6g added:**
  - a token-bounded "facts" view instead of raw grids in the prompt;
  - auto-check of new grid functions;
  - a grid-print guard;
  - a provisional nudge at 24k tokens;
  - a coverage-first scheduler (§4.3).
- **E14-P:** an optional switch that stores the server's raw `reasoning_content` in history (§5.8).
- **E16** [PENDING results]: an early nudge (after ≥ 2 inspection calls and ≥ 12k tokens) and raw assistant content.

### 4.3 Scheduling

- **Continuous admission (DIRECT, Poetiq).** v1.4 stopped admitting at 10:30. v1.4.4 admitted until 11:34, drained at 11:36 and stopped hard at 11:44. S1 moved these to 11:45 / 11:47 / 11:55 from kernel start.
- **Ordering.**
  - E7 used NVARC's `get_puzzle_priority` verbatim.
  - ORDER2 is our Qwen-specific rule. On the capability ledger it had nested-CV AUC 0.57 vs 0.51–0.53 for NVARC's tiers, with solve rates by score of 2 → 58 %, 1 → 43 %, 0 → 20 %.
  - S49 added a "smaller total cells first" tie-break.
- **Duck coverage-first (rev 6).**
  - Stage 1 runs branch 0 of every task until 28,000 generated tokens.
  - Stage 2 resumes branches or starts branch 1, by progress class.
  - Admission is held only while SGLang's `full_token_usage` ≥ 0.95; workers are fixed at 12.
  - Per-request caps: 12,288 for the first request, 24,576 for progress, 32,768 for rescue; ≤ 100,000 generated tokens per branch.
- **Gate 4b.** For eval runs we hold each task to its hidden time share: (deadline − start) / 240, about 172 s per task. Eval-subset results are then comparable to what the hidden run can afford.

### 4.4 Validation methodology

- **Deterministic builders.** Every notebook is produced by a builder and identified by sha256. Kaggle-run notebooks are never rebuilt in place.
- **Local CPU harnesses.** They run the real notebook cells against a fake engine (vLLM or SGLang) with the real tokenizer.
- **Kaggle CPU harness kit.** From 2026-10-23, every GPU notebook first runs this kit on the pinned image (Python 3.12.13; 167/167 unit tests, harness 32/32).
- **Correctness gate on every engine arm:**
  - exact greedy parity against an A/A repeat;
  - "large-margin flips" decide, because DFlash2 greedy output differs across restarts (#38009);
  - in the first E14A run, serial A/A was 10/12 identical; the 2 divergences were exact ties.

## 5. Experiments

**Protocol.**
- Eval sets: EVAL120 (all public eval tasks, 172 test inputs), EVAL12 (a hard 12-task / 22-input subset), EVAL30, EVAL6, EVAL14, HARD66 (the 66 tasks R5 did not fully solve; 103 inputs) and G4B (12 selected tasks / 19 inputs).
- Metrics: inputs correct under pass@2, and task-normalised score (each task's share of correct inputs, averaged).
- Seeds: SGLang ignores per-request seeds without deterministic mode, so reruns are independent samples.

### 5.1 Leaderboard history

**Table 1.** Scored submissions, public LB (120 tasks).

| Submission | Solver | Engine | Cap | LB |
|---|---|---|---|---|
| NVARC Qwen3-4B (baseline) | TTT | — | — | 32.64 |
| PIV v1.3 | REPL program synthesis, low effort | vLLM, no MTP, bf16 KV | 8,192 / task | 1.67 |
| v1.4 | DIRECT | vLLM FP8 MTP | 49,152 | 18.75 |
| v1.4.4 | DIRECT + prefix cache + late admission | vLLM FP8 MTP | 49,152 | 26.94 |
| E7 | v1.4.4 + NVARC ordering | vLLM FP8 MTP | 49,152 | 27.92 |
| E9 | DIRECT + tail resample | vLLM FP8 MTP | 77,824 | 22.92 |
| R5 | Poetiq | vLLM FP8 MTP | 77,824 / iteration | 21.39 |
| **S1-60K** | **Poetiq + ORDER2 + fixes** | **vLLM FP8 MTP** | **60,000** | **34.58** |
| S49 | S1 + tie-break | vLLM FP8 MTP | 49,152 | 32.08 |
| S1-55K | S1 | vLLM FP8 MTP | 55,000 | 31.81 |
| P1-60K | Poetiq, execution-aware prompts | SGLang INT4 DFlash2 | 60,000 | 29.17 |
| Duck rev 5 LOW | Duck agent, effort low | SGLang INT4 DFlash2 | 49,152 / request | 17.50 |
| E14S | Duck rev 6g | SGLang E14S (width 6) | Duck caps | [PENDING: E14S LB, ref 57574781] |
| (one more entry from the user's list) | — | — | — | [PENDING: source for the 31.94 LB entry; it is not in our files] |

**What the table shows:**
- **Usable time.** v1.4 → v1.4.4 (+8.2) is the cleanest causal result in the table. Only usable time changed: prefix caching avoids re-prefilling ~50k tokens after each forced close (about 12 % of scheduler time before), and admission ran until the end.
- **Bundled changes.** R5 → S1-60K (+13.19) is bundled, so it cannot be attributed to one change.
- **The S1 cap series** is 60k 34.58 / 55k 31.81 / 49k 32.08. The 49k and 55k points are 1–2 sd below 60k.

### 5.2 Coverage

**Table 2.** LB vs EVAL120 at the same cap.

| Run | EVAL120 task-norm, same cap | LB | LB ÷ eval |
|---|---|---|---|
| 49k DIRECT (v1.4.4 / E7) | 25.4 % | 26.94 / 27.92 | 1.06–1.10 |
| 86k DIRECT (E9) | 36.9 % | 22.92 | 0.62 |
| Poetiq 77.8k (R5) | 48.6 % | 21.39 | 0.44 |

The 49k value is reconstructed from the 86k DIRECT traces (hits that needed > 49,152 thinking tokens count as lost); it was not measured directly.

**Table 3.** DIRECT thinking cap vs EVAL120 task-normalised score (86k DIRECT traces re-capped).

| Cap | 32,768 | 49,152 | 60,000 | 70,000 | 77,824 |
|---|---|---|---|---|---|
| Task-norm | 13.75 % | 25.42 % | 27.92 % | 30.42 % | 36.94 % |

**Where the tokens go:**
- **86k DIRECT on EVAL120 (10.45 M tokens):**
  - 75.4 % went to units that were never solved;
  - forced or continued closes took 56.6 % of tokens for 6 hits (8 %);
  - natural closes at ≤ 49k were correct on 39 of 54 units (72 %).
- **Pooled over 9 runs:** 59.7 % of all generated tokens went to forced closes with no hit.
- **Poetiq** needs long thinking: 77 % of its solves came from iterations that thought past 49k.

The per-task optimum (long caps) and the per-run optimum (enough caps to reach every task) therefore pull in opposite directions. On the LB the second one won.

### 5.3 Program synthesis vs direct reasoning on EVAL120

**Table 4.** EVAL120, same engine (vLLM FP8 MTP).

| | Poetiq R5 (77.8k / iteration, T 1.0) | DIRECT ORDER2 (86k, T 0.6) | Union |
|---|---|---|---|
| Inputs correct / 172 | 78 (45.35 %) | 58 (33.72 %) | 92 (53.5 %) |
| Task-normalised | 48.61 % | 36.94 % | 56.81 % |
| Wall / tokens / speed | 691 min / 11.65 M / 281 tok/s | 534 min / 10.45 M / 333 tok/s | — |

**Findings:**
- **Overlap:** both runs 44; Poetiq only 34; DIRECT only 14. Poetiq's own attempt_2 added 1.
- **Multi-test tasks:** Poetiq solved 14 fully, against 5 for DIRECT. One program per task applies one rule to every test input.
- **New tasks:** Poetiq solved 13 tasks that no method of ours had solved before.
- **Train verification** was a useful but imperfect gate. Of 71 train-verified tasks, 49 were fully right on test, 7 partly and 15 wrong.
- **Allocation policies, computed from both runs:**

  | Policy | Task-norm |
  |---|---|
  | "Poetiq if train-verified, otherwise attempt_2 = DIRECT" | 55.14 % |
  | Full union | 56.81 % |

  The first keeps 97 % of the union while running DIRECT only on unverified tasks.

### 5.4 Independent attempts on hard tasks (HARD66)

Two 12 h runs on the 66 tasks R5 did not fully solve (103 inputs; R5 had 9 right), both with fresh seeds:
- **D66:** 77,824 thinking, 16 slots;
- **L66:** 122,880 thinking, 12 slots.

**Table 5.**

| | D66 | L66 |
|---|---|---|
| Inputs right | 28 | 34 |
| Tokens / wall / speed | 10.80 M / 42,231 s / 256 tok/s | 10.66 M / 41,968 s / 254 tok/s |
| Iteration-1 requests cut at the cap | 49 / 66 | 9 / 66 |
| Train-passing programs that were test-correct | 14 / 33 | 15 / 43 |

**Findings:**
- **Complementarity.** Only-D66 16, only-L66 22, both 12; McNemar p = 0.42. The **union is 50/103**, against 28 and 34 alone. On hard tasks, failure is largely sampling. So a genuinely independent second attempt is the largest attempt_2 lever we measured. R5's attempt_2 had added 1.
- **Train pass on hard tasks:** only 35–42 % of train-passing programs were test-correct, and Poetiq stops at the first pass. Train pass should rank candidates, not end the search.
- **Iteration 2 (with feedback)** was about as productive per slot-hour as iteration 1 (D66 0.14 → 0.22; L66 0.27 → 0.25).
- **Cap-forced answers** rarely succeeded: 5 of 49 at 77.8k.
- **Our wrong conclusion.** From these runs we advised against caps below 77.8k. The LB then showed the opposite for the 240-task run (S1-60K 34.58 vs R5 21.39). The D66/L66 conclusion holds per GPU-hour on hard tasks; it does not hold for coverage.

### 5.5 Effort level

W3H ran S1-60K at low effort on EVAL120 with a 3 h clock:
- 17/172 inputs (9.88 %), on 15 tasks, all inside R5's 63 xhigh tasks;
- 0 of 120 requests hit the 60k cap;
- but 78 % emitted EOS mid-thinking without `</think>`, so the harness force-closed unfinished thinking;
- train pass at iteration 1: 13 % vs 46 % for xhigh.

We did not run medium, given ARC Prize's 13.2 % for medium against 22.8 % for low.

### 5.6 The Duck REPL agent

**Table 6.** Duck adapter runs (SGLang W1 unless noted).

| Run | Tasks / inputs | Result | Key measurement |
|---|---|---|---|
| rev 1 EVAL30 (12,288 out / request) | 30 / 48 | 7/48 (13.3 %) | Branch 1 never started in 29/30 tasks; 144/423 later requests lost their prefix (8.39 M re-prefill tokens); 27 % hit the output cap |
| rev 4 EVAL6 (60k out, 100k / branch) | 6 / 8 | 3/6 tasks (4/8) | Tasks rev 1 solved 0 of; ~10 GPU-min per task |
| rev 5 EVAL6R5 (xhigh) | 6 / 8 | 4/6 tasks, 5/8 (66.7 %) | 3 of 5 hits via attempt_2; −30 % tokens; loop share 0–3 % (rev 4: 29 %) |
| rev 5 EVAL12R5_LOW | 12 / 19 | 7/12 tasks, 10/19 | Low is not cheaper per branch (72.7k vs 70.5k) |
| rev 5 LOW, LB | 240 hidden | **17.50** | Solved ~56 % of reached tasks; reached ~31 % of 240 |
| rev 6d G4B (gate 4b) | 12 / 19 | 2/19 | 47.7k tokens/task; false capacity step-down 12 → 6 (−13 % tok/s) |
| rev 6g, E14 engine (R2, 25-min window) | 12 / 19 | 6/19 (33.33 %) | Cached prompt 82.0 %; pass@2 added 0; 12 of 17 branches never called `check()` |
| E14S commit smoke (8-min window) | 12 / 19 | 1/19 | READY 783 s; 483 tok/s at 12 running |
| EVAL120 at the hidden share: xhigh, xhigh + E16, low + E16 | 120 / 172 | [PENDING: EVAL120 G4B xhigh/E16/low results] | — |

**Why Duck lagged in coverage:**
- At rev 5 the first `check()` came after a median of ~55k generated tokens, almost all of it reasoning. In 35 of 36 branches the model hand-read grids and then noticed its own misreads before writing code.
- A soft per-attempt cap of 35k would have kept 1 of 6 xhigh solves, with tokens per task falling only from 141k to 91k, because the crossing request still finishes.
- So the lever is an earlier tested transform, not a cap.
- Rev 6's facts view brought the first Python call to a median of 46 tokens. But the model then printed grids, and 7 of 12 branch-0 runs never defined a `transform` in 29–59k tokens.

**A bug in our own capacity probe.** In rev 6d the probe treated SGLang's steady-state queue = 1 at 11 running as overload. It stepped 12 → 10 → 8 → 6 streams and cost ~13 % throughput. That queue = 1 state was already in our own logs, but our test fixture assumed queue = 0. We replaced the probe with a single KV ≥ 0.95 rule. Replayed on 7 server logs, the rule would have held work back only in the one real KV-thrash run.

### 5.7 Serving measurements on 4× L4

**Table 7.** Concurrency caps and throughput across engine configurations.

| Config | Real running | KV tokens | tok/s |
|---|---|---|---|
| vLLM FP8 MTP3, DIRECT | 16 | 1,353,600 | 351 |
| SGLang INT4 + DFlash2, auto memory (E11-03 C) | capped at 6 by the Mamba cache | — | — |
| SGLang INT4 + DFlash2, fp16 SSM (E11-03 D) | 12 | 387k | 404 |
| SGLang W1 (E11-04) | 12 | 881k | 416.5 (production loop) |
| SGLang W1, E14 R2 short window | 12 | 866,744 | 505.0 |
| SGLang W1 + width 6 (E14S) | 12 | 917,816 | 531.4 (R2 short window) |

**Notes on the configs:**
- **Context costs speed.** W1's production rate fell with context: 458 tok/s below 10k, 413 at 10–20k, 388 at 20–40k.
- **Speculation helps.** DFlash2-8 is 1.49× target-only (505.0 vs 339.6).
- **MTP was never compared fairly.** E11-03's MTP arm used fp32 SSM state and was capped at 10 running. It gave 374 tok/s at 10, against DFlash2 at 12–13.

**Table 8.** Profiler, 20 decode steps at 12 running, TP rank 0 (other ranks within 0.35 ms).

| Category | ms / step | Share |
|---|---|---|
| NCCL all-reduce (140 × 983,040 B) | 50.9 | 44 % |
| NCCL all-gather (3) | 8.1 | 7 % |
| INT4 Marlin GEMM (512 calls) | 25.3 | 22 % |
| Other GEMM (drafter, lm_head) | 14.5 | 12.5 % |
| GDN (Triton) | 8.0 | 7 % |
| Elementwise / attention / sampling / norm | 2.9 / 2.7 / 2.4 / 1.1 | 7.8 % |
| **Sum** | **116.1** | |

An independent NCCL bench predicted 345 µs × 140 + 8.0 ms = 56.3 ms. The profiler's 59.0 ms confirms it.

**Transport arms:**
- `NCCL_P2P_LEVEL=SYS`: 1,887 µs per call, 5.5× slower;
- `NCCL_SHM_USE_CUDA_MEMCPY=1`: its graph test timed out.

Custom or IPC all-reduce is therefore not a lever on this topology. 2×TP2 would avoid the four-way all-reduce but holds only ~170k KV per replica, which we estimated at 0.7–1.0× of TP4.

**Table 9.** Engine arms (R2, short ~6k-context window, one launch each, correctness gate first).

| Arm | tok/s | Step ms | Accept length | Outcome |
|---|---|---|---|---|
| W1 (width 8) | 505.0 | 114.0 | 4.86 | reference |
| + #44401 backport | 508.1 | 114.3 | 4.84 | still 11 running under churn; not adopted |
| `--min-free-slots-delay 1` | 490.3 | 114.4 | 4.63 | crashed at 13 running (sampling buffer 12 vs batch 13) |
| CUDA graphs 1..13 | 500.1 | 114.7 | 4.79 | neutral; adopted |
| Width 6 | 531.4 | 96.4 | 4.24 | +5.2 % vs W1; adopted |
| Width 4 | 526.8 | 74.5 | 3.27 | +4.3 %; not chosen |
| int8 Mamba checkpoints | — | — | — | OOM |

**Width.** Accept length falls with width while the step shortens, so the net gain is small. Neither width 6 nor width 4 was measured on the long-context band (median prompt 40,877 tokens).

**The 11-of-12 loss.**
- With DFlash and no explicit setting, SGLang 0.5.21 resolves the min-free-slots threshold to `min(4, max(2, (mrr+5)//6))` = 3 at mrr 13. A returning stream therefore waits while 11 run.
- In a churn test the server sat at 11 running in 100 % of samples, and the #44401 backport did not change that.
- Under Duck churn, 103 of 108 decode lines in the submitted configuration showed 11 running.
- Disabling the delayer exposed a crash at 13 running. Two untested alternatives are in E15.

### 5.8 Agent-side prefix reuse

Duck stores each reply's reasoning after `_normalize_message_content`, which deletes blank lines (`tool_agent.py` L3414–3418):
- 0 of 180 stored THINKING blocks in our G4B transcripts contain a blank line.
- The re-rendered prompt therefore diverges from the generated tokens at the first paragraph break.
- In the G4B run, 68 of 150 requests (45 %) lost the previous reply, i.e. 217.5k tokens re-prefilled, about 146 s at our measured ~1,500 tok/s prefill. Prefill blocks every running stream's decode.

**Is it the template?** An offline token-diff of the chat template showed 0/170 reconstructed divergences, so the template is not the cause.

**The fix and its effect:**
- A fake server with the real template reproduced the loss with the switch off (24/24 replies diverge at the first blank line) and fixed it with the switch on (24/24 full prefix).
- On GPU, the first run with raw reasoning (R2) kept 115 raw blocks with 0 mismatches. Its cached prompt share was 82.0 %, against 75.0 % in the earlier G4B run without it. The two runs also differ in engine flags and code revision, so this is not a clean A/B.
- A replay of R2's 120 follow-ups: reply_reused 67, lost_at_prefill_end 26, partial 27, deeper_miss 0.

**Same code upstream.** The normalization is the same in Tufa's original harness and in Franzen's notebook. On an RTX PRO 6000 (prefill ~10.5k tok/s) it costs little; on 4× L4 it costs 3–4× more.

## 6. Negative results and failure modes

### 6.1 Methods

**Table 10.** Negative or non-transferring results, with their conditions.

| Idea | Condition | Result |
|---|---|---|
| Direct, short budget (Q1) | 12,288 cap, PROXY48 | 0/56; xhigh never closed |
| REPL agent PIV | 12,288 / turn | 0 tool calls; tools only after `</think>` |
| PIV submission v1.3 | 6,144 + 2,048 | LB 1.67 = EVAL120 1.67 % |
| R6 code loop | 12k repair caps | 2/22 at 92.7k tok/task; 46/47 turns forced; 3 of 4 train-verified programs wrong |
| Flash-Next on 4× L4 | SGLang fork, W4A16 + FP8 PLE | ready at 862 s, KV 118k; garbage tokens; ~6 tok/s single-stream |
| Q27-AGENT REPL | vLLM 0.19, 6,144 / turn, forced `tool_choice` | 0/22; 82 submits all forced, 59 empty; 5 HTTP 500 |
| Long-think synthesis v1.9.1 | server `thinking_token_budget` + MTP | budget ignored (vLLM #39573); 1/22 |
| Same-reasoning second answer | v1.4.4 | +0 hits; 12/22 ragged or invalid |
| NLL scoring of thinking-mined grids | `prompt_logprobs` | truth in 0/31 traces; OOM killed 15 in-flight units |
| Tail resample attempt_2 | 86k, EVAL12 | 0 gain for 41.3 % of tokens |
| 86k / 77.8k caps on the LB | E9 / R5 | 22.92 / 21.39, below 49k |
| Early stop of hopeless thinking | offline, text signals | AUC ~0.63; saves 3–4 % of tokens |
| Pause-and-resume at 30–60k | offline | worse at every budget |
| Low effort | W3H | 17/172; no complementary solves |
| SGLang + new prompts | P1-60K | LB 29.17 vs 34.58 (confounded) |
| Duck at 12,288 / request | rev 1 | 7/48; KV thrash |
| Duck second branch | R2 | +0 |
| Delayer off, int8 Mamba, P2P, NCCL env, #44401 | E14 arms | crash / OOM / 5.5× slower / no gain / no gain |

**These negatives are context, not verdicts.** Most early "code / agent / repair does not work" conclusions came from budgets of 3k–12k tokens per turn. That is far below the 26–45k at which Qwen3.8 closes naturally on solvable tasks. Poetiq at 77,824 per iteration later beat DIRECT by 11.7 task-norm points on EVAL120.

### 6.2 Platform failure modes (Kaggle)

1. **Commit notebook size.** Commits started for notebooks of 329–407 kB:
   - 444 kB never started its code (ERROR after 114 min, quota used);
   - a 727 kB control padded only with comments did not start in 11 min;
   - 745 kB failed 3/3 with `ERRORED_MOUNTING_DATASET`.

   The same 727 kB notebook ran interactively. The limit lies between 407 and 444 kB. This also explains the loss of a 10-22 run worth ~3.8 quota-hours.
2. **Error visibility.** Mount errors appear only on the web page. The API log is `[]` and the failure message is empty.
3. **Storage.** From ~10-21, parts of `/kaggle/input` mounted via GCSFUSE, and NFS slowed to as low as 0.026 GiB/s.
   - Our server launch went from 507–562 s to 1,197 s, against a 30-min first-launch timeout. Other teams reported 0.00 scores from exactly this timeout.
   - Our staging copy (64 MiB ranges, adaptive 4 → 8 → 16 workers, hashes, `.partial` + rename) moved weights at 518–568 MB/s, against 34.6–36.9 MB/s for a single-stream read.
4. **Python version drift.** A `Thread` subclass storing an Event as `self._stop` crashes `join()` on Python ≤ 3.12. Our local harness ran on 3.14, so it missed the crash; one GPU run (~0.8 quota-h) was lost. Hence the Kaggle CPU kit (§4.4).
5. **Image drift.** Kaggle's default image moved to Python 3.13, and our cp312 wheelhouse broke. Every kernel now pins the original image.
6. **File names.** Kaggle drops `+` from dataset file names, which breaks pip's match against a lock. Our installer restores names from SHA256SUMS and checks each sha256.
7. **Billing.** Quota was billed at 1.05× wall time in one run (10-13), and at 1.95× and 2.37× (including container time) in later runs.
8. **No-fallback paths.** In the Duck notebooks, an engine bring-up failure produces the identity submission (≈ 0 LB). A vLLM-fallback flag exists but has no code behind it.
9. **First launch is compile-bound.** Of a 502 s first launch, weights finished at 65 s and ~7 min went to compile, JIT and graph capture. A relaunch takes 71 s. Restoring the compile cache across sessions is unproven (our tar step failed).

## 7. Discussion and limitations

**Limits of the coverage reading:**
- It depends on the premise that hidden ≈ eval difficulty. This is consistent with three LB/eval ratios but not directly observable: hidden logs are not visible.
- Each eval number is a single run. LB noise is ±1.5.
- Several LB steps bundle changes (R5 → S1-60K; S49's cap plus tie-break; P1-60K's engine plus prompts).

**Engine correctness is not established at length and concurrency:**
- Our gate uses 256-token probes.
- Public DFlash2 reports (#36548: wrong outputs in 111/304 cases at concurrency 8 on another stack; #38009: greedy drift with thinking) are not excluded by it.
- The unexplained 29.17 vs 34.58 gap could be engine, quantization or prompt.

**Duck evidence is thin:**
- Duck results come from 6–30-task subsets and one LB run.
- The gate-4b runs approximate, but do not reproduce, the hidden queue of 240 tasks.

**Comparison with other teams.** Public discussion reports Qwen3.8-27B teams at 65–72 on the same hardware. We did not reproduce any of them, and no top team's method is public. We read the gap as engineering and allocation, not a model ceiling. That is a hypothesis, not a result.

## 8. Future work (in progress)

- **E14S** (Duck rev 6g on DFlash2 width 6): [PENDING: E14S LB, ref 57574781].
- **EVAL120 at the hidden time share**, three commits: xhigh, xhigh + E16, low + E16. They measure Duck accuracy and coverage on all eval tasks: [PENDING: EVAL120 G4B xhigh/E16/low results].
- **E15**, all with the correctness gate plus a c12 × 1,024-token reasoning-on probe against target-only:
  - MTP from the checkpoint's own BF16 head instead of DFlash2. Predicted KV ~1,198k vs 917,816; this needs an explicit draft path, since without it 0.5.21 sizes KV without the draft layer;
  - MTP 2/1/3;
  - MTP + ReplaySSM (allowed for EAGLE, blocked for DFlash in 0.5.21);
  - 14 / 16 / 20 real streams;
  - two delayer-safe settings (mrr 14 with graphs 1..14; delayer off at mrr 12);
  - chunked prefill 2048.

  [PENDING: E15 engine arms]
- **Hybrid:** Duck stage 1 on every task, then one S1-60K Poetiq request (60k thinking + 8,192) on Duck-unverified tasks, with Duck branch 1 off.
  - The merge keeps Duck's selection when no Poetiq rows exist.
  - The time arithmetic says stage 2 reaches 38–57 % of unverified tasks.

  [PENDING: Hybrid build and smoke]
- **Not started:**
  - a paired SGLang-INT4 vs vLLM-FP8 accuracy test;
  - an engine watchdog with mid-run relaunch;
  - a CPU rule library to fill identity slots;
  - an NVARC union.

## 9. Reproducibility

**Notebooks:**
- Each is built by a deterministic builder in its experiment folder and named by sha256. For example, S1-60K is `ed530071…` (LB 34.58), and the E14S submission is `3dde5cbd…`, built by `build_e14.py`.
- Each runs in dual mode: a commit run scores an eval subset, and the competition rerun solves the hidden set.

**Inputs:**
- Kaggle Models: our Qwen3.8-27B-FP8 repack, RedHat INT4 (`w4a16-g128`, 19.47 GB), DFlash2 (`bf16`, 3.85 GB);
- the public dataset `sglang-0-5-21-cu13-wheelhouse` (227 files, with installer);
- the private payload dataset with the Duck package and E14 modules (sha-named `.bin`).

**Run data.** Every GPU run saves the full thinking of every request, server logs, per-request records and run reports. These are kept per experiment under `Experiment Results/Kaggle Runs/`.

**To publish:** [PENDING: public release of notebooks, datasets and run bundles].

## 10. Acknowledgements

We build directly on:
- **Tufa Labs and Jeroen Cottaar**: the Duck harness;
- **Daniel Franzen**: the milestone-2 fork we vendor, his write-up, the checkpoint-prefetch idea and the Mamba-retention patch on our list;
- **Poetiq**: the open ARC-AGI solver (MIT);
- **NVARC**: the baseline notebook and `get_puzzle_priority`;
- **Johan Land**: whose paper structure this draft follows;
- **Qwen** (Qwen3.8-27B), **RedHatAI** (the INT4 checkpoint), **z-lab** (DFlash2), and the **SGLang** and **vLLM** projects;
- **ARC Prize**, for per-effort verified results and the competition;
- **Kaggle community members**, for public notebooks and discussion: the 745787 recipe thread, russcore's prompt, madarshbb's agent notebook.

**AI-assisted development.** Most notebooks, harnesses and analyses were produced with Claude and ChatGPT. Several notebook versions were authored by one assistant and audited by the other. The team directed the experiments, verified claims against code and logs, and made every run and submission decision.

## References

- ARC Prize. Qwen3.8-27B verified results (arcprize.org/results/alibaba-qwen3-8-27b).
- Cottaar, J. / Tufa Labs. Duck harness (github.com/Tufalabs/duck-harness).
- Franzen, D. ARC-AGI-3 Milestone 2: Building on the Duck Harness (github.com/da-fr/arc-agi-3-solution).
- Land, J. Modality-Driven Search with Holistic Trace Judging for ARC-AGI-2. arXiv:2606.31543, 2026.
- Poetiq. poetiq-arc-agi-solver (github.com/poetiq-ai/poetiq-arc-agi-solver).
- RedHatAI. Qwen3.8-27B-INT4. z-lab. Qwen3.8-27B-DFlash2.
- SGLang v0.5.21; vLLM v0.19.0. Issues referenced: vLLM #39573; SGLang #36548, #38009, #39087, #44401.
- [PENDING: full bibliographic details and URLs for the NVARC notebook and Kaggle discussion threads 745787, 747515, 751344]
