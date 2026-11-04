# ARC Prize 2026 (ARC-AGI-2): Qwen3.8-27B on 4× L4, where coverage is the binding constraint

*[PENDING: team name and author list]*

**DRAFT, 2026-10-24.** Every number below comes from our own run logs, analyses and notes. `SOURCES.md` (next to this file) lists the internal source of each one. Items still running are marked **[PENDING: …]**.

We entered ARC-AGI-2 with one open-weight model, Qwen3.8-27B, on the competition machine: 4× NVIDIA L4, offline, a 12-hour run, 240 hidden tasks. Our starting point was an NVARC-derived Qwen3-4B notebook at 32.64. Our best Qwen3.8-27B submission so far is **34.58** (Poetiq-style program synthesis at a 60k thinking cap with an easy-first queue). Our current submission (E14S, a Duck REPL agent on an optimized SGLang engine) is still being scored.

This write-up is mostly about one finding. On this hardware, **how many of the 240 tasks get a good answer inside 12 hours decides the leaderboard score.** Per-task accuracy matters less. Most of what we learned are measured consequences of that. Many results are negative, and we report them as such.

**TL;DR**

- **Coverage binds.** The public LB is 120 tasks, and the hidden rerun solves 240 tasks in 12 h. We compared each method's LB with its own EVAL120 score at the same cap:
  - 49k DIRECT: LB/eval ≈ **1.06–1.10** (almost everything finished);
  - 86k DIRECT: **0.62**;
  - Poetiq at 77.8k: **0.44**.

  Our best eval method (Poetiq, 48.61 % task-normalised on EVAL120) gave one of our lower LB scores (21.39).
- **Shorter caps plus a better queue won on the LB.** Poetiq at 60,000 thinking tokens, with an easy-first queue (ORDER2) and six rerun-path fixes, scored **34.58**, against R5's 21.39 at 77,824. At 55k and 49k it scored 31.81 and 32.08, so 60k is the best cap we measured.
- **The clock matters as much as the model.** v1.4 → v1.4.4 changed only prefix caching and admission timing, and gained +8.19 LB (18.75 → 26.94).
- **The second attempt was wasted in every early version.** Independent samples, though, are highly complementary:
  - on the 66 hardest eval tasks, two fresh runs solved 28 and 34 of 103 inputs, with a union of **50**;
  - on EVAL120, DIRECT ∪ Poetiq = 92 of 172 inputs, against 78 for Poetiq alone.
- **Engine: about half of a decode step on 4× L4 is GPU-to-GPU communication.** The profiler measured 59 of 116 ms per step. P2P is 5.5× slower than the default NCCL path on this PCIe (PHB) topology. Results:
  - SGLang 0.5.21 + RedHat INT4 + DFlash2 ran at 404–417 tok/s at 12 really running, against ~255–351 tok/s for vLLM FP8 + MTP;
  - narrowing the DFlash2 width to 6 added +5.2 %;
  - other flag levers were neutral, or crashed.
- **Agent-side cache loss was a harness bug, not the engine.** Duck's history normalization deletes blank lines from stored reasoning, so the previous reply was re-prefilled in 45 % of requests. Storing the raw reasoning raised the cached prompt share from 75.0 % to 82.0 % (different runs, so this is not a clean A/B).
- **Kaggle platform failures cost us more GPU time than any model bug:**
  - a commit-notebook size limit between 407 and 444 kB;
  - mount errors that are visible only on the web page;
  - a storage slowdown that pushed our server launch to 1,197 s against a 30-min timeout;
  - a Python 3.12-only crash that our Python 3.14 harness missed.

## 0. Submissions and leaderboard

The public LB has 120 tasks. Identical notebooks resubmitted scored with an sd of **1.0–1.65** LB points (NVARC ×10: 27.36–32.64). So differences under ~3 points between single submissions are inside the noise.

| Date (2026) | Submission | Method, in one line | Thinking cap | Public LB |
|---|---|---|---|---|
| before 10-03 | NVARC-derived Qwen3-4B (team baseline) | NVARC test-time-training specialist | — | **32.64** |
| 10-03 | PIV-Q27-SUB v1.3 | REPL program synthesis, low effort | 8,192 per task | 1.67 |
| 10-06 (scored 10-10) | Q27-XHIGH v1.4 | DIRECT: one xhigh request per test input, forced close | 49,152 | 18.75 |
| 10-10 | Q27-XHIGH v1.4.4 | v1.4 + prefix cache + admission until 11:34 | 49,152 | 26.94 |
| 10-12 | E7 ORDERING_ONLY v2 | v1.4.4 + NVARC easy-first queue | 49,152 | 27.92 |
| 10-14 | E9 | DIRECT 86k + tail resample, file order | 77,824 | 22.92 |
| 10-15 | E10 Poetiq R5 | Poetiq program synthesis, ≤ 10 iterations, T 1.0 | 77,824 / iteration | 21.39 |
| 10-17 | **S1-60K** | R5 + ORDER2 queue + 6 fixes + KV gate + new clock | 60,000 | **34.58** (best) |
| 10-18 | S49 | S1-60K + 49k + "smaller grids first" tie-break | 49,152 | 32.08 |
| 10-19 | S1-55K | S1-60K with a 55k cap | 55,000 | 31.81 |
| 10-20 | E11-06 P1-60K | Poetiq on SGLang INT4 + DFlash2, execution-aware prompts | 60,000 | 29.17 |
| 10-21 | E12 Duck rev 5 LOW | Duck REPL agent, reasoning effort low | 49,152 output / request | 17.50 |
| 10-24 | **E14S** | Duck rev 6g on the E14 engine (DFlash2 width 6) | Duck caps (§3.3) | **[PENDING: E14S LB, ref 57574781]** |
| — | (one more entry from the user's list) | — | — | **[PENDING: source for the 31.94 LB entry; it is not in our files]** |

**Reading the history:**
- v1.4 → v1.4.4 (+8.2) is real. Nothing changed but usable time.
- v1.4.4 → E7 (+1.0) is inside the noise.
- E7 → E9 (−5.0) and E7 → R5 (−6.5) are about 2.5–3 sd, so very likely real.
- R5 → S1-60K (+13.19) is the largest gain we have. It bundles several changes, so it cannot be split.

## 1. The setting and why coverage binds

**Hardware and rules:**
- 4× NVIDIA L4 (24 GB each, PCIe; all GPU pairs report PHB topology, no NVLink);
- no internet;
- a 12 h limit that includes container start, input mounts and engine launch;
- 240 hidden tasks: the 120 public-LB tasks plus 120 private.

**Model:** Qwen3.8-27B. Its chat template accepts only `reasoning_effort` xhigh, medium or low; "high" raises an error.

**Effort.** ARC Prize's own verified run of this model on the semi-private set scored:

| Effort | Semi-private score |
|---|---|
| xhigh | 42.4 % |
| low | 22.8 % |
| medium | 13.2 % |
| none | 1.5 % |

Our own low-effort probe on EVAL120 solved only a subset of what xhigh solves (17/172 inputs). So we kept xhigh except in labelled experiments.

**Thinking length.** Qwen3.8 hits come almost only from thinking that closes on its own:
- In our DIRECT runs, natural closes at ≤ 49k tokens were correct **72 %** of the time, and forced closes **8 %**.
- **59.7 %** of all generated tokens, pooled over 9 runs, went to forced closes that produced no hit.
- Every archive-phase rejection of code, agents or repair (Sept–early Oct) ran at budgets of 3k–12k tokens per turn, far below the model's natural close length. Those were budget failures, not method failures.

**The coverage model.** The hidden set is about as hard as the public evaluation set (user-confirmed; the old "hidden is easier" premise came from the placeholder test file). So LB ÷ eval at the same cap estimates the share of hidden work finished in 12 h:

| Run | EVAL120 at the same cap | LB | LB ÷ eval |
|---|---|---|---|
| 49k DIRECT (v1.4.4 / E7) | 25.4 % | 26.94 / 27.92 | 1.06–1.10 |
| 86k DIRECT (E9) | 36.9 % | 22.92 | 0.62 |
| Poetiq 77.8k (R5) | 48.6 % | 21.39 | 0.44 |

- **Pace of the long methods.** On EVAL120, 86k DIRECT needed 534 min for 120 tasks and Poetiq R5 needed 691–704 min. The hidden rerun has 240 tasks in the same clock.
- **A linear time model:** scale each run's own pace to 240 tasks, against ~680 scheduling minutes. It predicted the 86k and Poetiq LB results within ~1–4 points.
- **Speed-up needed for full coverage:**

  | Method | Speed-up | Expected LB |
  |---|---|---|
  | 86k DIRECT | 1.54× | ~37 |
  | Poetiq R5 | 2.04× | ~49 |
  | Poetiq + DIRECT on unverified tasks | 2.55× | ~52 |
  | Both methods on every task | 3.57× | ~57 |

That model set our agenda: make good methods fit 240 tasks in 12 h, through the engine, the queue, the clock and the token budget.

## 2. Model and serving

### 2.1 vLLM 0.19 + FP8 + MTP (Experiments 1, 6–10)

**Setup (v1.4.4 / S1-60K):**
- offline vLLM engine, TP 4, FP8 weights, fp8 KV (1,353,600 tokens);
- MTP speculative decoding ×3;
- `max_num_batched_tokens` 4,096 (larger values OOM'd in the MTP drafter);
- 16 sequences, gpu memory 0.90;
- prefix caching with Mamba `align` mode.

**Speed:**
- DIRECT aggregate: 351 tok/s at 16 sequences, 302–325 at 12.
- Poetiq R5 ran ~16 % slower. T = 1.0 lowered MTP acceptance to 0.557, and KV at 99.5 % evicted the prefix cache.

**Pitfalls we hit:**
- With MTP on, vLLM 0.19 silently ignores the server `thinking_token_budget`, `min_p` and `logit_bias` (vLLM #39573). Our Experiment 5 lost a whole run to this: 10 of 12 tasks thought until `max_tokens`.
- `prompt_logprobs` scoring OOM'd in the logits all-gather next to 16 long requests (Experiment 8).
- Forced named `tool_choice` produced empty code or HTTP 500 (Experiment 4).

### 2.2 SGLang 0.5.21 + RedHat INT4 + z-lab DFlash2 (Experiment 11)

**Stack:**
- target RedHatAI/Qwen3.8-27B-INT4 (W4A16 g128, 19.47 GB);
- BF16 drafter z-lab/Qwen3.8-27B-DFlash2 (3.85 GB);
- SGLang 0.5.21, from a 220-wheel offline wheelhouse (torch 2.13 cu130, flashinfer 0.6.18).

**What we found:**
- **`--speculative-draft-model-quantization unquant` is mandatory.** Without it the drafter inherits INT4 and acceptance collapses (issue #39087).
- **The first flat throughput curve was a cap, not a speed limit.** SGLang caps running requests by the Mamba state cache; DFlash2 was capped at 6 running in the first config.
- **The config that held 12 real requests (W1):**
  - `extra_buffer_lazy`, Mamba cache 56, fp16 SSM state;
  - no prefill CUDA graph;
  - `mem-fraction-static` 0.83.

  It gave 12 real running, a KV pool of 881k tokens, and **416.5 tok/s** in a 12-stream production loop.
- **DFlash2 vs no speculation.** In the earlier capacity-limited run, DFlash2 with an fp16 SSM state reached 404 tok/s at a real 12 running: 1.39× INT4 alone and about 1.3× vLLM.
- **"11 of 12" root cause.** The DFlash-only min-free-slots delayer in 0.5.21 uses threshold `min(4, max(2, (mrr+5)//6))` = 3 at max-running 13, so a returning stream waits while 11 run. Upstream #44401 does not fix it; we confirmed this on the GPU. Disabling the delayer (`--min-free-slots-delay 1`) crashed the server at 13 running.

**LB transfer.** The first LB test of this engine (P1-60K) scored **29.17**, against S1-60K's 34.58 on vLLM. It changed the engine and the prompts at once, so the cause is not separated. It is our largest open correctness question (§7).

### 2.3 Experiment 14: the optimized engine

**Profiler at 12 running, TP rank 0, 20 decode steps:**

| Category | ms per step |
|---|---|
| NCCL all-reduce (140 calls of 983,040 B) | 50.9 |
| NCCL all-gather | 8.1 |
| INT4 Marlin GEMM | 25.3 |
| other GEMM | 14.5 |
| GDN | 8.0 |
| **Kernel sum** | **116.1** |

**Communication is 59 of 116 ms (51 %).**

**NCCL bench on the same box:**
- baseline 345 µs per 983 KB all-reduce inside a graph;
- `NCCL_P2P_LEVEL=SYS` is 5.5× slower;
- so custom/IPC all-reduce paths are dead here.

**Arms (R2 run; short ~6k-context window; each arm its own launch with the correctness gate first):**

| Arm | tok/s | Decision |
|---|---|---|
| W1 baseline (DFlash2 width 8) | 505.0 | reference |
| #44401 backport | 508.1 | not adopted (still 11 running) |
| `--min-free-slots-delay 1` | 490.3 | **crashed at 13 running** (`dflash_worker_v2.py`, buffer 12 vs batch 13) |
| CUDA graphs 1..13 | 500.1 | adopted (neutral) |
| target-only (no speculation) | 339.6 | DFlash2-8 = 1.49× target-only |
| **DFlash2 width 6** | **531.4** | **adopted: +5.2 % vs W1** |
| DFlash2 width 4 | 526.8 | gate PASS, not chosen |
| int8 Mamba checkpoints | — | **OOM** |

Width 6 has been measured only at short context. The long-band prompts (median 40,877 tokens) never ran for widths 6 and 4, so width 6 at long context is still open.

### 2.4 Settings of the current submission (E14S, ref 57574781)

| Setting | Value |
|---|---|
| GPUs | 4× L4, TP 4 |
| Engine | SGLang 0.5.21 (offline wheelhouse, Python 3.12 venv in `/tmp`) |
| Target | RedHatAI/Qwen3.8-27B-INT4 (W4A16 g128, compressed-tensors) |
| Drafter | z-lab/Qwen3.8-27B-DFlash2 (BF16), `--speculative-draft-model-quantization unquant` |
| Speculation | DFLASH, `--speculative-num-draft-tokens 6` (draft config block size 8) |
| KV cache | fp8_e4m3; pool 917,816 tokens |
| Context length | 155,648 |
| Running requests | server 13 / client 12 |
| Chunked prefill | 4,096; prefill CUDA graph off |
| Decode CUDA graphs | batch sizes 1..13 |
| Mamba | `extra_buffer_lazy`, cache 56, SSM state fp16 |
| Memory fraction | 0.83 |
| Parsers | `--reasoning-parser qwen3`, `--tool-call-parser qwen3_coder`, `preserve_thinking` true |
| Schedule policy | lpm |
| Weights | staged to `/dev/shm` with a parallel ranged copy (16 workers, sha-checked) |
| Agent | Duck rev 6g adapter (vendored Duck @ 10882e33, adapter sha 758bddb5) |
| Agent settings | template-default effort (xhigh), T 1.0; per-request caps 12,288 first / 24,576 progress / 32,768 rescue; first-exposure pause at 28,000 generated tokens; 2 branches; ≤ 100,000 generated per branch; 12 workers; KV admit gate 0.95; nudge at 24,000; raw reasoning kept in history |
| Sandbox | subprocess with a 4,096 MB memory limit, Python 3.12.13 |
| Clock (rerun) | budget 43,020 s; Duck deadline 42,720 s; hard stop at T0 + 42,900 s |

**Commit smoke** (12 tasks, 8-min Duck window):
- READY at 783 s;
- 483 tok/s median generation at 12 running, accept length p50 4.07;
- no OOM or exceptions;
- 1/19 eval inputs. That is a smoke result, not a score.

## 3. Solvers

### 3.1 DIRECT long-thinking (Experiment 1)

**Method** (the recipe from public discussion 745787):
- one prompt per test input;
- xhigh thinking up to 49,152 tokens;
- then Qwen's stop sentence + `</think>` appended token-exactly;
- then ≤ 8,192 answer tokens with a grid-only answer;
- free-form prompt adapted from the public russcore notebook; T 0.6.

**Results:** 18.75 and 26.94. The second attempt was the test input or its mirror, so it scored almost nothing.

### 3.2 Poetiq program synthesis (Experiment 10, S1)

We ported Poetiq's solver (MIT; poetiq-ai/poetiq-arc-agi-solver) to the offline engine:
- the model writes `transform(grid)`;
- a sandbox runs it on every training pair;
- feedback (wrong cells, shapes, errors, a score per example) seeds the next iteration, which gets the best ≤ 5 earlier solutions;
- a task stops at the first train pass;
- the sandbox is a subprocess with rlimits, seccomp and a uid drop, because the container blocks `unshare` and `mount`.

**R5 on EVAL120** (77,824 thinking + 8,192 answer per iteration, 1 expert, ≤ 10 iterations, T 1.0, 16 sequences):

| | Poetiq R5 | DIRECT 86k |
|---|---|---|
| Inputs hit | **78/172** (task-norm **48.61 %**) | 58/172 (36.94 %) |
| Union | 92/172 (56.81 %) | |

- Poetiq opened 13 tasks no method of ours had ever solved.
- 70 % of its solves thought past 49,152 tokens.
- Train verification was 69 % precise: 49 of 71 verified tasks were fully right, 15 wrong.

**S1-60K: the R5 code plus these changes:**
- thinking 60,000;
- the ORDER2 queue: score = [≥ 3 training examples] + [1 test input], higher first. Every task gets iteration 1 before any iteration 2;
- F1: run the *last* python block that defines `transform`;
- F3: per-input attempts;
- A2: attempt_2 is never a copy of attempt_1. R5 had copied it on 134 of 172 inputs;
- F2/F2b: harvest verifications pending at the hard stop, and cancel queued ones;
- C1: a bad output closes only its own unit;
- a KV admission gate at 0.95;
- weight prefetch;
- a clock measured from kernel start: admit 11:45, drain 11:47, hard stop 11:55. R5 had left about 16 minutes of the 12 h unused.

Result: **LB 34.58**.

### 3.3 The Duck REPL agent for ARC-AGI-2 (Experiment 12)

**What we vendored.** Tufa Labs' Duck harness (Jeroen Cottaar), through Daniel Franzen's milestone-2 fork `da-fr/arc-agi-3-solution`, unchanged (Apache-2.0). It provides the chat loop, tool-call recovery, the Python sandbox, function retention, context trimming and the priority gate.

**Our own code** is a thin ARC-2 adapter:
- a system prompt;
- task grids in a trusted prelude;
- `check(transform)` with cell diffs;
- `submit` through the JSON action channel, re-verified by the host in a clean sandbox;
- two branches per task as attempt_1 / attempt_2.

**Revisions and results:**

| Revision | Run | Result |
|---|---|---|
| rev 1 | EVAL30 | 7/48 (13.3 %), below Poetiq's 16/48 on the same tasks. Branch 1 never started in 29/30 tasks; KV thrash re-prefilled 8.4 M tokens |
| rev 4 | EVAL6 (60k output / request) | 3/6 tasks that rev 1 had solved 0 of. About 10 GPU-min per task, i.e. ~40 h for 240 tasks |
| rev 5 | EVAL6R5 | 4/6 tasks (66.7 %); **3 of 5 hits came through attempt_2** |
| rev 5 | EVAL12R5_LOW | 7/12 tasks |
| rev 5 LOW | **LB** | **17.50**: solved ~56 % of the tasks it reached, but reached only ~31 % of 240 |

rev 5 fixes: harvest of `check()`-exact functions, tokenizer-based context estimates, a truthful resume, two attempts, and T 1.0.

**The coverage problem in Duck:**
- The first `check()` came only after ~55k tokens (median 56.6k xhigh, 57.5k low). In 35 of 36 branches the model hand-read grids and then caught its own misreads.
- A simulated soft 35k cap per attempt kept only 1 of 6 xhigh solves.

**rev 6 → 6g targeted coverage:**
- **Stage 1:** branch 0 for every task, paused at 28,000 generated tokens.
- **Stage 2:** resumes and branch 1, by progress class.
- A KV-only admission gate (0.95); 12 workers.
- A token-bounded facts view, so the first Python call comes after a median of 46 tokens.
- Auto-check, a grid-print guard, and a provisional nudge at 24k.

**Gate-4b runs** (12 hard tasks, each at the hidden per-task time share of ~172 s):

| Run | Result |
|---|---|
| rev 6d | 2/19 inputs at 47.7k tokens/task. Our capacity probe wrongly stepped 12 → 6 streams on SGLang's normal queue = 1, costing 13 % throughput |
| rev 6g on the E14 engine (R2, 25-min window) | **6/19** (task-norm 33.33 %) |

## 4. Scheduling and coverage

- **Admission until the end.** v1.4 stopped admitting at 10:30. v1.4.4 admitted until 11:34 and drained at 11:36. That clock plus prefix caching was worth +8.2 LB.
- **Easy-first queues.** Offline analysis put ORDER2 at +4–7 points over random order under a 50–70 % budget; its nested-CV AUC was 0.57 vs 0.51–0.53 for NVARC's tier order. On the LB the NVARC ordering gave +1.0, which is inside the noise.
- **Caps.** On EVAL120 DIRECT traces, re-capping at 49k keeps 25.42 % task-norm, 60k keeps 27.92 % and 77.8k keeps 36.94 %. On the LB, 60k beat both 77.8k and 49k for Poetiq (34.58 vs 21.39 / 32.08). The short cap wins only because it buys coverage.
- **What did not work for coverage:**
  - early stop of "hopeless" thinking: best text signal AUC 0.63, saving only 3–4 % of tokens;
  - pause-and-resume: worse at every budget.
- **Concurrency must be sized by KV, not by a number.** 16 slots at 77.8k on vLLM gave KV 99.9 % and 66 preemptions at no speed gain; 12 slots at 155,648 gave 0 preemptions.
- **Duck needs its own form of coverage.** Rev 6's stage-1-first scheduler exposes every task before any deep work. A 240-task simulation of the real scheduler exposed 240/240. That is not yet proven on a real 240-task run: **[PENDING: EVAL120 G4B xhigh/E16/low results]**.

## 5. Platform engineering (what Kaggle taught us)

1. **Commit notebook size limit.** Commit runs started for notebooks of 329–407 kB, but never for 444 kB and up:
   - a 444 kB notebook: ERROR after 114 min, with quota used;
   - a 727 kB comment-padded control: no start in 11 min;
   - a 745 kB notebook: `ERRORED_MOUNTING_DATASET` on 3 of 3 attempts.

   The same padded notebook ran fine interactively. **Fix:** keep notebooks < 400 kB, and move payloads into a private dataset as sha-named `.bin` files.
2. **Mount errors show only on the web page.** The API's log is `[]` and the failure message is empty, so we had to ask a human for the page text.
3. **Storage slowdown (from ~10-21):** parts of `/kaggle/input` moved to GCSFUSE, and NFS slowed to as low as 0.026 GiB/s.
   - Our SGLang launch went from 507–562 s to **1,197 s**, against a 30-min first-launch timeout. A timeout means the fallback submission, about 0 LB. Other teams reported 0.00 scores from exactly this.
   - **Our load path:** a 64 MiB ranged parallel copy to `/dev/shm`, with an adaptive worker ramp, `.partial` + rename and per-range hashes. It moved the 19.47 GB INT4 file at **518 MB/s**, against **34.6 MB/s** for a single-stream read; the rate plateaus at 16 workers (786 MB/s).
   - **The first launch is mostly compile:** of the 502 s, the weights were loaded after 65 s, and about 7 min was compile, JIT and graph capture. A relaunch in the same session takes 71 s. Restoring the compile cache across sessions is open (our tar step hit an `IndexError`).
4. **Validate on a Kaggle CPU kernel with the same pinned image.** Our local harness ran Python 3.14. On Kaggle's Python 3.12, a `threading.Thread` subclass that stored an Event as `self._stop` crashed every measurement window, because `join()` calls `Thread._stop()` on ≤ 3.12. Since then every GPU notebook passes a Kaggle CPU harness kit first (Python 3.12.13; 167/167 tests).
5. **Pin the docker image.** Kaggle's default image moved to Python 3.13, and our cp312 wheelhouse could not install vLLM. Every kernel pins the old image.
6. **Kaggle strips `+` from dataset file names.** `flashinfer_jit_cache-0.6.18+cu130` became `0.6.18cu130`, which pip cannot match to the lock. The wheelhouse now ships an installer that restores names from SHA256SUMS and checks each sha256.
7. **Quota billing.** One measured run used about 1.05 quota-hours per wall-hour (10-13). Later runs were billed **1.95×** and **2.37×** of wall time, the latter including container time. We now budget runs at ≥ 2×.
8. **No silent fallbacks.** The Duck notebooks carry an `E11_ALLOW_VLLM_FALLBACK` flag with no code behind it. If SGLang fails to come up, the submission is the identity file. We keep the loud failure but now know the cost.
9. **Submission logistics.** One pending submission per team, 12–16 h turnaround. Every idle slot is lost evidence.

## 6. What did not work (or did not transfer)

Each line is a measured result. Several were negative because of how we set them up, so they are context, not verdicts.

| Idea | Result |
|---|---|
| Short-budget code / REPL / evolution loops (archive phase) | 0 correct on 8–48-task subsets at 3k–12k tokens per turn; PIV v1.3 LB 1.67. Budget far below Qwen's natural close |
| R6 code loop (Exp 2) | 2/22 at 92.7k tokens/task; 46/47 repair turns force-closed |
| Flash-Next on 4× L4 (Exp 3) | Server up at 862 s, KV 118k, but garbage output ("!!!!") and ~6 tok/s single-stream |
| Q27-AGENT REPL (Exp 4) | 0/22. The model never submitted on its own; 59 forced submits had empty code; 5 HTTP 500s; 6,144 thinking per turn |
| Server thinking budget (Exp 5) | Ignored under MTP (vLLM #39573): 1/22 |
| Second answer from the same reasoning (Exp 6) | +0 hits; 12/22 ragged or invalid |
| NLL-scored candidates mined from thinking (Exp 8) | Truth in 0/31 traces; scorer OOM killed 15 in-flight units |
| Tail resample for attempt_2 (Exp 9) | 0 gain; 41.3 % of tokens |
| 86k DIRECT on the LB | 22.92 (< 27.92 at 49k) despite 36.9 % eval |
| Poetiq 77.8k on the LB | 21.39 despite 48.6 % eval |
| Low effort (W3H) | 17/172; 78 % of requests emitted EOS mid-thinking; no complementary solves |
| Medium effort | Not run: ARC Prize medium 13.2 % < low 22.8 % |
| 49k and 55k caps for S1 | 32.08 and 31.81, both below 60k's 34.58 |
| SGLang engine + P1 prompts (E11-06) | LB 29.17 < 34.58 (cause not separated) |
| Duck at 12,288 output / request (rev 1) | 7/48; branch 1 never ran; KV thrash |
| Duck second branch (R2) | pass@2 added 0; 12 of 17 branches never called `check()` |
| `--min-free-slots-delay 1`, int8 Mamba, NCCL env knobs, P2P | crash / OOM / no gain / 5.5× slower |
| #44401 backport | no occupancy change (still 11 of 12) |
| Early-stop of hopeless thinking, pause/resume | AUC 0.63, 3–4 % tokens saved; pause/resume worse at every budget |

## 7. What is next

1. **E14S LB.** The first hidden run of Duck on the width-6 engine: **[PENDING: E14S LB, ref 57574781]**.
2. **EVAL120 at the hidden time share.** Three 6.5 h commits run 120 eval tasks with each task held to its hidden-run time share:
   - xhigh without E16;
   - xhigh with **E16**, which nudges early (after ≥ 2 inspection calls and ≥ 12k tokens) and keeps raw content;
   - E16 at low effort.

   They measure Duck coverage and accuracy as the hidden run would see them: **[PENDING: EVAL120 G4B xhigh/E16/low results]**.
3. **E15 engine arms.**
   - MTP (EAGLE 3/1/4 using the INT4 checkpoint's own BF16 MTP head) instead of DFlash2. The memory model predicts a KV pool of ~1.2M tokens vs 917,816. MTP was capped at 10 running in its only earlier test, so it has never been compared fairly.
   - MTP + ReplaySSM.
   - 14 / 16 / 20 real streams.
   - Two safe fixes for 11-of-12: max-running 14, or the delayer off at max-running 12.
   - Chunked prefill 2048.
   - A longer c12 × 1,024-token correctness probe against target-only. The DFlash2 issue #36548 reported wrong outputs at concurrency 8 on another stack.

   **[PENDING: E15 engine arms]**
4. **Hybrid.** Duck stage 1 on every task, then one S1-60K Poetiq request (60k thinking) on Duck-unverified tasks, with Duck branch 1 off. Its arithmetic says plain requests reach only 38–57 % of unverified tasks, so it is bounded by coverage too. **[PENDING: Hybrid build and smoke]**
5. **Explain 29.17 vs 34.58.** Run a paired SGLang-INT4 vs vLLM-FP8 test with the S1 prompt. One team reported in public discussion that a "tiny kernel issue" held its LB at 40 % against a projected 60 %.

## 8. Reproduction

- **Notebooks and builders:** every notebook is built by a deterministic builder and identified by sha256.
  - S1-60K: `Q27-XHIGH-v1_4_4_E10_S1_60K_POETIQ_FIX_ORDER2.ipynb`, sha `ed530071…`.
  - E14S: `Q27-XHIGH-E14S_SUBMIT_W6G13_DUCK_SGLANG.ipynb`, sha `3dde5cbd…`, built by `build_e14.py`.
- **Dual mode:** a commit run scores an eval subset; the competition rerun solves all hidden tasks.
- **Inputs:**
  - our Kaggle Models: Qwen3.8-27B-FP8 repack, RedHat INT4, DFlash2;
  - the public SGLang 0.5.21 cu13 wheelhouse (227 files, with installer);
  - the private payload dataset for the Duck package and E14 modules.
- **Thinking traces.** Full traces of every request are saved in every GPU run, and the run bundles are kept per experiment.
- **[PENDING: public links to the notebooks and datasets once they are made public]**

## 9. Acknowledgements and credits

- **Tufa Labs and Jeroen Cottaar** for the Duck harness, which our agent vendors unchanged.
- **Daniel Franzen**, whose milestone-2 fork we vendored. His write-up gave us:
  - the milestone settings;
  - the "keep the thinking in history" finding;
  - the checkpoint-prefetch idea we used in our Flash-Next probe;
  - the Mamba checkpoint-retention patch on our list (not yet ported).

  This write-up also follows the format of his write-up.
- **Poetiq** for the open program-synthesis solver (MIT) that our best submission adapts.
- **NVARC**, for:
  - the baseline notebook (32.64);
  - `get_puzzle_priority`, the easy-first ordering we copied verbatim.
- **Johan Land** for "Modality-Driven Search with Holistic Trace Judging for ARC-AGI-2" (arXiv 2606.31543). Our paper draft follows its structure.
- **Qwen** for Qwen3.8-27B.
- **RedHatAI** for the INT4 checkpoint, and **z-lab** for DFlash2.
- **The SGLang and vLLM teams.**
- **ARC Prize** for the per-effort verified results and the competition.
- **Kaggle community members** whose public notebooks and discussion posts shaped our recipe:
  - the 745787 discussion;
  - russcore's free-form prompt;
  - madarshbb's agent notebook.
- **AI assistance.** Most code and analyses were written with Claude and ChatGPT. We directed the experiments, reviewed every change and made every submission decision.
