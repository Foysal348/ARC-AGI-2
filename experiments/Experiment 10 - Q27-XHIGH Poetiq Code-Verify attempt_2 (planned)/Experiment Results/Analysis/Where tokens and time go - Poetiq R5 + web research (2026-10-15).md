# Where our tokens and time go, and how to solve faster: Poetiq R5 base + web research (Claude, 2026-10-15)

**Base:** Poetiq R5 EVAL120. 78/172, task-normalised 48.61 %; round 1 of 120 tasks took about 9 h. On the LB, E9 (86k DIRECT) scored 22.92, because time binds on the hidden run.

**Sources:**
- Our files: session scratchpad `order2/r5_token_sinks.py`, `r5_analysis.py`, `r5_order2_replay.py`.
- 5 research agents with about 290 web searches and fetches; their reports are summarised below with URLs.
- The PDF given earlier (`resources/Research Papers/ARC_AGI_2_High_Score_Systems_SOURCE_FIRST_2026 (1).pdf`). No new PDF was attached.

## Part A. Our own data: where R5's 11.65 M tokens went

| Sink | Measured | Meaning |
|---|---|---|
| **Iterations that hit the 77,824 cap** | 104 of 172 (60 %) use **66 %** of all tokens and give 21 of the 71 train passes | The cap is not the waste: forced traces still yield 30 % of the passes |
| **Forced traces that never wrote এcode** | **58 of 104**: no `def transform` anywhere in the thinking | The model was still searching for the rule by hand when it ran out |
| **Digit tokens (cell values, coordinates)** | **22.7 %** of all thinking; 24–25 % in failed or forced traces vs 19.6 % in passed ones; 22.8 coordinate mentions per 1k tokens | The model reads and scans grids cell by cell in prose ("r4 c13 = 9 … r? c?") |
| **Thinking after the first code draft** | **35 % of the thinking of solved traces** comes after their first `def transform`. Median 5 drafts per solved trace; first draft at 60 % of the trace | The model re-checks its code in its head, which a sandbox does in 0.3 s |
| Grid rows copied line by line | 1.7 % | Small |
| Code inside thinking | 9.2 % | Moderate |
| Reflection words (Wait/Hmm/Actually/re-check) | 1.4–1.7 per 1k tokens (about 100 per trace), slightly higher in failed traces | Present but not dominant |
| Explicit hypothesis switches | about 0.1 per 1k tokens | Rare |
| **Loops** | **0 of 172 Poetiq traces** (11.25 M tokens; two independent detectors) | Unlike DIRECT (4/172) |
| Tokens by outcome | solving iterations 29 %; earlier failed iterations of solved tasks 13 %; verified-but-wrong tasks 12 %; **never-verified tasks 52 %** | Half of all compute goes to tasks that never verify |
| Tokens by pre-R5 tier | tier D (never solved before) **47 %** of tokens → 11.5 task points; tier A 16 % → 22.0 | Hard tasks are expensive, but they gave 13 new solves |
| Value per request | iteration 1: 0.40 task points per request; iteration 2: 0.17 | A first try on a new task is worth 2.3× a retry, so coverage comes first |

## Part B. The user's three ideas, checked against the data

1. **"One request per puzzle with all test inputs"**
   - **Poetiq already does this.** One request per task, and one program for every test input; that is why Poetiq solves 14 multi-test tasks fully vs DIRECT's 5.
   - The 29 % saving was measured for DIRECT.
   - **Correction (same day):** the earlier claim "hidden has only 7.1 % multi-test tasks" is wrong.
     - These figures (240 tasks, 259 inputs, 17 multi-test = 7.1 %, median max area 184.5; Kaggle discussion 745846) are exactly those of the public **placeholder** `arc-agi_test_challenges.json`.
     - All 240 of its task ids are **training** tasks.
     - The real hidden set is unknown; it is plausibly eval-like (inference).
2. **"Loops: 30 public / 45 private, 20k+ tokens each"**
   - **Not supported:**
     - Poetiq R5 has 0 loops in 172 full traces.
     - DIRECT ORDER2 has 4/172, and the dual v2 run has 1/22.
   - LOOPSTOP stays a cheap safety net, but it saves almost nothing for Poetiq.
3. **"Send selected puzzles to iteration 2 first"**
   - **Supported, with a correction:** a new task's first iteration is worth 2.3× a retry. So coverage of every task comes first (R5's FIFO already does this), and *then* retries are routed by signal.
   - Signal for that routing: the first train score. At score exactly 0, DIRECT rescued 5 inputs in 16 tasks.

## Part C. What the research says

### C1. The top teams (agent report 1)

- **ARC Prize 2025 winners all used small models (≤ 8B) with per-task test-time training and direct grid decoding, with no chain-of-thought.**
  - NVARC: Qwen3-4B, full fine-tune on 103k synthetic puzzles, LoRA TTT, DFS, about $0.20/task.
  - NVIDIA wrote that the Kaggle limit "ruled out chain-of-thought, tool use and RL-style agents".
  - Sources: https://github.com/1ytic/NVARC , https://developer.nvidia.com/blog/nvidia-kaggle-grandmasters-win-artificial-general-intelligence-competition/
- **2026 top teams have published nothing yet.**
  - nvbanana (CPMP / JFPuget) reached 49 → 71 before Qwen3.8 existed, with runs of 7–10 h.
  - CPMP: "We got 70+ before Qwen 3.8, and this model reshuffled everything" (discussion 745787).
  - Tufa Labs: 32.5 → 88.06 between Aug 22 and Oct 7.
  - Teams ranked #5–#11 jumped from about 30 to 55–67 on Oct 2–5, roughly 7 weeks after Qwen3.8; whether this came from Qwen3.8 harnesses is unconfirmed.
- **Public Qwen3.8 notebooks are all at or below about 21.5 LB** (madarshbb agent: 21.5, then 13 on a rerun).
  - mikelou1 reports 36 → 38.75, closed source.
  - **Our 27.92 is above every public Qwen3.8 notebook.**
- **Unverified, and probably not about the hidden set:** discussion 745846 says the "hidden set" has median max input area 184.5 vs 551 and 7.1 % vs 40.8 % multi-test.
  - Our check: these are exactly the statistics of the public placeholder test file, which is made of 240 **training** tasks.
  - The real hidden set is only visible during the rerun.

### C2. Engine speed on 4×L4, vLLM 0.19 (agent report 3)

**Arithmetic:** decode is limited by memory reads. The ceiling is about 60 ms per step (KV ≈ 35 ms + weights ≈ 25 ms). We measure about 145 ms per step (333 tok/s ÷ 16 sequences ÷ about 3 tokens per step). **Roughly 85 ms per step of overhead may be recoverable** (inference).

| Option | Evidence | Note |
|---|---|---|
| `attention_backend = TRITON_ATTN` (A/B vs FLASHINFER) | FlashInfer on Ada drops spec-decode to PIECEWISE CUDA graphs; Triton allows full graphs. A public Qwen3.8 notebook pins TRITON_ATTN + `VLLM_FORCE_ATTN_BACKEND` for the MTP drafter too | Easiest A/B test. Guess: +5–15 % |
| `gpu_memory_utilization` 0.90 → 0.93–0.94 | about +120k KV tokens | Fewer preemptions (Poetiq R5 had 9 at 99.5 % KV) |
| MTP k=3 vs k=2 | on the hybrid model, each extra draft token holds an extra state block per request (about 148 MiB): k=3 ≈ 9 GiB for 16 sequences. **Our measured per-position acceptance: DIRECT 0.82 / 0.66 / 0.53 (about 3.0 tokens per step); Poetiq R5 0.73 / 0.54 / 0.40 (about 2.7 tokens per step, at temperature 1.0)** | For Poetiq the 3rd draft position is accepted only 40 % of the time, so k=2 (freeing KV) is worth testing |
| `rejection_sample_method="probabilistic"` | +2–9 % in one fork at temperature 1 | Risk of bugs |
| Concurrency 16 → 24 | russcore: 158 → 255 tok/s going from C=8 to C=24 | Only where KV allows. DIRECT peaked at 87 % KV, Poetiq at 99.5 % |
| TP4 vs 2× TP2 | Sources disagree: +30–50 % on DGX Spark and +56 % on 3090 vs "keep TP4" (the KV pool halves; custom all-reduce is off at TP4 on PCIe, so NCCL is used) | **Benchmark it, don't assume** |
| Per-channel FP8 or W4A16 weights | Marlin INT4 is about 3.9× faster at batch ≤ 16 (A10); W4A16 Qwen3.8 checkpoints exist | Same model, different quantization. Accuracy risk. **Needs the user's approval** |
| **0.19 limitation** | **With MTP on, logit_bias, `thinking_token_budget` and custom logits processors are disabled.** Only bad_words, penalties and stop strings work (fixed in 0.21) | Most "think less" methods can't run with MTP on 0.19 |

### C3. Thinking less without cutting the 49k floor (agent report 4)

- **Avoid NoWait / DEER / ES-CoT:** Qwen3 loses 5–23 points on hard tasks. And our reflection-word rate is low.
- **DeepConf** (stop low-confidence traces): −16 to −59 % tokens at near-equal accuracy. It needs per-token logprobs and a calibration pass; past 49k only.
- **TIP thought-switch penalty:** this needs a custom logits processor, so it is blocked with MTP on 0.19, and our switch rate is low anyway.
- **Qwen3.8 card:** thinking mode uses temperature 1.0 (Poetiq uses 1.0; DIRECT 0.6). `reasoning_effort` medium used more tokens and scored lower than low (ARC Prize). Keep xhigh.

### C4. Program synthesis done cheaper (agent report 5)

- **Run code during the reasoning** instead of one long monologue: stop at each closed python block, run it on the train pairs, append the result, resume, and exit when train passes.
  - Pivotools: GPT-OSS-120B 6 % → 26 % on ARC-AGI-2.
  - Confluence Labs and Imbue use the same loop.
  - **This directly targets our three biggest sinks:** 22.7 % of thinking is hand-reading of grids; 58 forced traces never wrote code; 35 % of solved-trace thinking is mental re-checking of code.
  - Caution: our own earlier REPL agents (Experiments 4 and 5: 0/22, 1/22) were a different, agent-only design.
- **Pre-computed facts in the prompt** (objects per colour, bounding boxes, changed cells for same-size tasks, shape rule): +5–10 points in KAAR, Xu and Greenblatt. Cheap, and it targets the 22.7 % of digit-reading.
- **Refine only tasks with signal; give "no signal" tasks a fresh sample** (Explorer-Definer, Greenblatt). Stop as soon as there are 2 distinct train-passing programs.
- **Gold-free safety checks after a train pass:** the program contains train outputs literally; the test output equals the test input; the test shape or colours break a rule that every train pair follows.
- **attempt_2:** in Poetiq's own code it is empty with 1 expert; our R5 fills it from the program history, which gave +1 hit.

## Part D. Ranked plan (one change at a time, each needs the user's go)

| # | Change | What it removes | Expected effect (inference) | GPU-free check first | Risk |
|---|---|---|---|---|---|
| 1 | **Engine A/B on GPU:** TRITON_ATTN vs FLASHINFER; mem 0.94; MTP k=2 vs 3; TP4 vs 2× TP2 | Engine overhead (we run at about 40 % of the bandwidth ceiling) | +10–50 % tok/s with **unchanged answers** | Read the per-position MTP acceptance and preemptions in our metrics | Low (fallback ladder exists) |
| 2 | **Code execution during reasoning** in Poetiq (stop at each python block → sandbox on train → resume; exit on pass) | Hand-scanning of grids, mental code re-checks, forced traces with no code | Largest upside: fewer tokens per solve and more passes | CPU harness with a fake engine | High (engineering; our earlier REPL agents failed) |
| 3 | **Fact sheet in the Poetiq prompt** (objects, diffs, shape rule) | Part of the 22.7 % digit-reading | Literature: +5–10 points | Prompt size check | Medium (prompt sensitivity) |
| 4 | **Scheduling for the hidden run:** iteration 1 for all tasks first (as R5) + ORDER2 + cheaper-first inside a score band; iteration 2 routed by first train score | Late coverage | +3 points under time pressure (replay) | Clock replay (done) | Low |
| 5 | **attempt_2:** second program or DIRECT answer | — | +1–11 inputs on eval | Offline | Zero GPU |
| 6 | W4A16 or FP8-dynamic weights | Weight bandwidth | Up to about 2× decode (literature) | — | **Needs approval:** accuracy unvalidated |

**Not recommended:**
- NoWait or answer-probe exits (accuracy losses on Qwen3);
- suffix or n-gram drafting instead of MTP (it turns off async scheduling);
- LOOPSTOP as a time source for Poetiq (0 loops).

## Key source URLs (from the agents)

- NVARC: https://github.com/1ytic/NVARC
- ARChitects 2025: https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/
- MindsAI/Tufa 2025: https://github.com/jcole75/arc_2025_mindsai
- Kaggle discussions 745787, 745846, 748082, 748419, 729509 (arc-prize-2026-arc-agi-2)
- madarshbb notebook: https://www.kaggle.com/code/madarshbb/arc-agent-sub-q38-direct-b98k-c16-792ef9
- delongyang notebook: https://www.kaggle.com/code/delongyang/arc-agi-2-inference-qwen3-8-27b-fp8-within-4-hours
- russcore notebook: https://www.kaggle.com/code/russcore/arcagi2-qwen38-freeform-direct
- vLLM 0.19 sources: speculative config, logits_processor/__init__.py, attention backends docs; issue 39573 (thinking budget ignored with MTP)
- HyperQwen: https://github.com/syv-ai/HyperQwen
- Poetiq: https://github.com/poetiq-ai/poetiq-arc-agi-solver ; Imbue: https://imbue.com/blog/2026-02-27-arc-agi-2-evolution ; Pivotools: https://pivotools.github.io/pivotools-quarto-blog/posts/agentic_coding_arc_agi/
- Greenblatt: https://blog.redwoodresearch.org/p/getting-50-sota-on-arc-agi-with-gpt
- KAAR: https://arxiv.org/html/2505.17482v2 ; DeepConf: arXiv 2508.15260 ; NoWait: arXiv 2506.08343 ; Qwen3.8 card: https://unsloth.ai/docs/models/qwen3.8-27b
