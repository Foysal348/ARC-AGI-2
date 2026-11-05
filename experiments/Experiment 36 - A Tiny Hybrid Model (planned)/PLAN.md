# Experiment 36: A Tiny Hybrid Model

Started 2026-11-05 (BDT). User goal: one hybrid system, built from **our own solving methods** (Duck, Direct, Poetiq, and the Domdolus REPL agent of Experiment 35), that tries to solve **every** ARC-AGI-2 evaluation puzzle inside the Kaggle environment (4x L4, 12 h, no internet). Candidate for the ARC Prize paper track. (A TRM tiny-network variant was proposed first and dropped by the user the same day.)

## Why a hybrid: the methods solve different puzzles

From `resources/Puzzle Capability Ledger/method_overlap.md` (Qwen3.8-27B runs, 120 eval tasks, 172 test inputs):

| | queries | task-norm |
|---|---|---|
| best single run (Poetiq R5) | 78 | 58.33 |
| one run each, Duck + Poetiq + Direct, perfect selection | 99 | 72.67 |
| all our runs, perfect selection (oracle) | 128 | 92.33 (76.9 %) |

- One run each: only Duck 6 tasks, only Poetiq 20, only Direct 11.
- Strengths: Poetiq strongest overall and on "many colours" / "3+ train pairs"; Direct adds same-size-output tasks; Duck adds larger outputs.
- Flash-Next with the Domdolus REPL agent alone: LB 72.36-83.47 over 5 submissions (mean ~77.5).

The hybrid target is to turn the oracle (union of methods) into a single run's score: run every method where it adds coverage, then pick the 2 answers well.

## Why it was not possible before, and is now

E33 (Duck + Poetiq + Direct chain on our Merlin engine, 24 streams, ~420-490 tok/s) was simulated: the chain rarely got past Duck, Poetiq / Direct ran on 0-16 tasks (`Experiment 33/Experiment Results/E33_SPEED_COVERAGE_ESTIMATE_2026-11-03.md`). "Speed is the lever that opens E33's coverage." The Experiment 35 base engine has a ~4.06M-token pool and runs 70 agents at once (128 allowed), so all methods fit in one run.

## Design

**Engine:** Experiment 35 base (Domdolus vLLM 0.31 stack, Q2 experts, K4 KV, DCP2), plus bundle S when ready.

**Model (user, 2026-11-05): a very small 2-bit Flash-Next, about 21 GB.**
- No public model of that size exists on Kaggle (checked 2026-11-05: only Domdolus Q2 = 34.01 GB experts + 6.85 GB dense + 51.2 GB n-gram table in host RAM; the rest are 4-bit / NVFP4 / FP8).
- 21 GB = the Domdolus GSQ Q2 experts cut from 512 to **320 experts** (34.01 × 320 / 512 ≈ 21.3 GB). Build by slicing, no re-quantization: keep the Q2 tensors of the kept experts and the matching router rows, update `num_experts` in config.json. Dense 6.85 GB and the n-gram table stay as they are.
- Keep-list from routing counts collected on our own Flash-Next ARC traces **including tool-call turns** (notnitsuj: pruning that ignores tool-call routing dropped verified sessions 0.78 → 0.34). Public REAP lists (`yokoinaba/qwen38-flash-next-reap*`, `boristown/...-reap-448e`) as a cross-check.
- Expected effect: about 3.2 GB less weight per GPU → KV pool 8.81 → about 12 GiB per GPU (+36 %, ~5.5M tokens), more agents at once. Accuracy cost must be measured: B0 (512 experts) vs the 320-expert model on the same smoke.
- **Evidence (discussion 755801, Yi-Chia Chen, public LB 2nd, saved in `resources/Kaggle Writeups/discussion_755801_YiChiaChen_public2nd_2026-11-04.md`):**
  - 512 → 320 experts with **gEAN** importance = router weight × expert output norm, summed over tokens of ARC-AGI-2/3 **agent trajectories**; 58 % less log-prob deviation than REAP at 320.
  - 320 experts + INT3 W3A8 (GPTQ g128) + block-FP8 attention/GDN: local ARC-AGI-2 eval pass@1 **81.6 before any distillation**, 85.0 after QPAD (reverse-KL distillation from the 512-expert FP8 teacher, top-32 logits, ~1.15B tokens; needs big training GPUs).
  - Laureano Arcanio (same thread): pruning without fine-tuning did not work for him; quantization only took his harness 35 → 61.
  - So use gEAN, not plain routing counts. The extra step from 3-bit to 2-bit is unmeasured.
- The pruned model is internal only (never handed to other teams' or the Gemma teammate's work).

**Methods (all on the same server, different prompts / loops):**

| id | method | source to reuse | proof |
|---|---|---|---|
| R | Domdolus REPL agent (multi-turn Python tool, `check()`, verified final answer) | Domdolus cells 29-33, 46 | LB 72-83 on Flash-Next |
| P | Poetiq: program + per-example feedback iterations, voting over solutions | E18 R2 S1-60K core, `Experiment 33/Build/e33_poetiq.py` | LB-proven on 27B; 88.33 task-norm oracle over all runs |
| X | Direct: long thinking, output grid only | E31 / `e29_direct.py` | 11 Direct-only tasks |
| D | Duck: code-first agent | `arc2_duck` (E28/E29) | 6 Duck-only tasks |

**Router (per task, one chain per attempt, from the E33 design, now with room to run):**
1. R on every task first (strongest single method on Flash-Next).
2. Task not verified by R → P (iteration 1, then the feedback iteration) and X run **in parallel** on that task.
3. Still nothing verified → D, then a second R with a different seed / try 2.
4. Verified tasks still get an independent second attempt late in the run (E28 found all its attempt_2 hits there).

**Selector (2 answers per test input):**
1. Programs verified on all train pairs, grouped by identical test output; most votes (across methods) first.
2. If none: answers that two different methods agree on.
3. Then: best train-match record (R `check()` / P score), then Direct grid.
4. attempt_2 = the next distinct candidate by the same order.

**Same idea measured by Yi-Chia Chen (755801):** two independent solvers per task gave +6-7 points (letting solver 2 see solver 1 did not help); a refiner that checks both candidates on the train pairs and improves or combines them gave +4.6 [+1.7, +8.3]. Solver budget 200k + 32k finalization, refiner ~100k. Add a refiner stage (R or P prompt that receives the candidates) after the independent solvers.

**Outputs:** per method per task per test input saved (feeds the ledger and the paper's complementarity table), LIVE line per finished stage, full thinking traces saved.

## Steps

1. B0 smoke (`hosen42/e35-b0-smoke75`, running) → R's per-task results on Flash-Next.
2. Port P and X clients to the vLLM 0.31 server (OpenAI API, Flash-Next template); local unit tests with a fake server (reuse the E33 tests).
3. Router + selector as one module beside the Domdolus harness; local scheduler tests.
4. Kaggle CPU kit, GPU smoke (~75 min), then EVAL120 run; add every run to the ledger.

GPU steps need the user's go (account per run).
