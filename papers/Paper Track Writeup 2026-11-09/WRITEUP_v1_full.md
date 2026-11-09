# A Verifying Tool Agent on Qwen3.8-Flash-Next with Four L4 GPUs

**Subtitle:** ARC-AGI-2 public leaderboard 54.17 (submission 57851995): a sparse MoE served offline at 4-bit, with every answer produced by a Python program that the agent checks against the training pairs.

Team Hack2Publish. Notebook: https://www.kaggle.com/code/foysalemonshanto/q27-xhigh-e25-nb6-submit (version 1).

## Abstract
Our ARC-AGI-2 submission runs Qwen3.8-Flash-Next fully offline on Kaggle's 4×L4 machine, with 4-bit (W4A16) routed experts, an FP8 KV cache and 3-token MTP speculation. Nine agent sessions run at once. Each one reasons, writes a `transform(grid)` function in a stateful Python sandbox, scores it on every training pair with `check()`, and submits the program's test output. The submission scored **54.17** on the public leaderboard. Our earlier best was 34.58 (a 27B dense model); our test-time-training reproductions stayed at or below 32.64. Our measurements point to two reasons for the gain. First, a stronger sparse model fits in the same time budget. Second, the train check is a precise filter: of 26 submitted attempts that passed every training pair, 25 were correct. A faster 3-bit version of the same model had train-checked answers right only 76 % of the time and scored lower.

## 1. Introduction
ARC-AGI-2 asks for the exact output grid of an unseen task, with two attempts per test input. The Kaggle version gives 240 hidden tasks, 12 hours and four 24 GB L4 GPUs, with no internet. Reasoning models do well on ARC when they can think for a long time: our 27B runs needed about 60k thinking tokens per task to reach their best. On four L4s, this makes the problem a **budget problem**. The score is the number of tasks reached in time multiplied by the chance of solving a task once reached. We changed the model, the engine and the solver to raise both.

## 2. Prior work
- **Test-time training (TTT).** The ARChitects (2024) and NVARC (2025) fine-tune a small model on each task and select candidates by augmentation scoring. We reproduced the NVARC line many times. Our public scores ranged from 23.75 to 32.64 (grey points in Figure 2).
- **Direct reasoning and refinement.** Long-thinking LLMs and Poetiq-style refine loops. Our best with Qwen3.8-27B was 34.58.
- **Tool agents.** Our solver adapts the "Duck" agent harness from the ARC-AGI-3 track (Franzen) to ARC-AGI-2 grids. That harness gives a model a persistent Python tool, and we added a train-pair checker and a `submit` call.

Our change from this work is the combination. We use a large sparse MoE instead of a small fine-tuned model. We serve it at low precision on L4s, a setting where we found no public reports. And we put a program-level verifier inside every attempt, instead of voting over sampled grids.

## 3. Approach
Figure 1 shows the pipeline.

**Model and engine.** Qwen3.8-Flash-Next is a 48-layer MoE with 512 routed experts per layer, top-10 routing, and a large n-gram embedding table. We serve:
- routed experts as W4A16 (five checkpoint shards);
- the per-layer embedding table in FP8;
- an FP8 E4M3 KV cache;
- MTP speculative decoding with 3 draft tokens.

The engine is Merlin 3.0.0, a vLLM 0.32 based server, with a pinned-memory ladder, prefix caching, and a shared draft workspace. The KV pool holds 738,142 tokens. Nine sessions run at once with an attempt budget of 104k tokens, and we measured 225 tok/s (p50) of total throughput.

**Agent.** Each attempt is one tool-using session:
1. The model sees the task.
2. It thinks, then writes Python in a stateful sandbox with grid helpers.
3. It calls `check()`, which runs `transform` on every training pair and returns per-pair accuracy and cell diffs.
4. It iterates on the program until the pairs pass, or until the budget runs out.
5. It calls `submit()`, and the program's output on the test input becomes the answer.

Because the answer comes from a program and not a typed grid, transcription errors on 30×30 grids disappear. The sandbox fails closed, so a crash or timeout yields no answer rather than a broken grid.

**Scheduling.**
- Attempt 1 runs on every task first, in a fixed priority order.
- Attempt 2 is an **independent** fresh session, not a continuation of attempt 1.
- `attempt_1` and `attempt_2` hold the two sessions' answers. Every test input gets a fallback answer, so the file is always complete.

**Run safety.** The budget is 12 h minus 3 min. A failed engine boot moves to the next memory step (at most three tries), a stalled server is relaunched, and a complete `submission.json` is written early and again at the end.

The commit run checks the same code on three evaluation tasks; it scored 3/3 with 0 errors.

## 4. Results
**Leaderboard (public, 50 % of hidden tasks).** Figure 2 shows all our scored submissions:

| Family | Best public LB |
|---|---|
| NVARC-style TTT reproductions | 32.64 |
| Qwen3.8-27B, direct / refine | 34.58 |
| Qwen3.8-27B, tool agent | 22.92 |
| Flash-Next 3-bit (klee100), tool agent | 49.58 |
| **Flash-Next W4A16, tool agent (this submission)** | **54.17** |

**What the measurements show.**
- **FP8 KV costs nothing measurable.** On 15 evaluation tasks, FP8 and BF16 KV each solved 13 of 15 tasks (18/20 outputs), with 2 discordant tasks (McNemar p = 1.0; the 15 tasks were picked as 7 easy, 5 medium, 3 hard). FP8 halves KV memory: the same boot fits 9 concurrent sessions with FP8 against 5 with BF16.
- **The train check is precise at 4 bits (Figure 3).** Of 30 W4A16 attempts, 26 passed every training pair and were submitted. 25 of those were correct on the test, and the other 4 never passed (3 were cut by the time window, 1 by the budget). The 3-bit build ran at 385 tok/s against our 225 (it also carried extra speed levers) and used 1.44× the tokens on the same tasks. On those same 15 tasks it had 8 train-passing wrong answers against W4A16's 1. Overall, 13 of its 54 train-passing answers were wrong.
- **Attempt 1 does almost all the work.** In 90 measured attempts, attempt 2 rescued 1 task out of 30. It still guards against an unlucky first session.
- **Coverage is still the limit.** In a 6-hour, 120-task evaluation run of a W4A16 build, 63 tasks were fully solved: 44 of the first 60 in queue order but only 19 of the last 60. The faster 3-bit run was balanced (39 and 35). The 12-hour submission reaches further into the queue.

## 5. Why it works
1. **Program-level verification turns thinking into checked answers.** The training pairs act as a free unit test. A model that writes a short program which reproduces every training pair has very likely found the rule. A model that only writes a grid has no such check. This explains why the 27B agent did not help (22.92): its first `check()` came at a median of about 55k tokens, close to its 35–45k attempt budgets, so few sessions reached a passing program. The stronger Flash-Next model reaches one often enough for the filter to matter.
2. **Precision of the filter depends on model quality, not just speed.** The 3-bit model reached more passing programs per hour. But those programs were more often fitted to the training pairs without capturing the rule, so its leaderboard score was lower (49.58 vs 54.17) despite a better 6-hour evaluation run. For a verifying agent, quantization error shows up as false verification, which a throughput benchmark does not show.
3. **Memory is the real currency on L4s.** 4-bit experts and the FP8 KV cache turn a model that does not fit into one that serves nine 100k-token sessions. MTP speculation (median acceptance length 2.88) then raises tokens per second without changing the output distribution.

## 6. Universality and limits
The recipe has three parts: a sparse model at low precision, a stateful code tool, and a cheap verifier built from the task's own examples. It transfers to any task family where candidate solutions can be executed and checked against given examples, such as program synthesis, data transformation, or spreadsheet formulas.

Limits:
- The leaderboard score comes from one run on the public half of the hidden set. Our repeated runs of one notebook vary by about ±1.5 points.
- Evaluation figures come from subsets (15 and 120 tasks) and from builds that are close to, but not identical with, the submitted one.
- We did not fine-tune anything.

## 7. Conclusion
A large sparse model, served at 4 bits with FP8 cache and speculation on four L4 GPUs, inside an agent that only submits train-checked program output, scored 54.17 on the ARC-AGI-2 public leaderboard. That is 20 points above our best dense-model result. The main lesson for future ARC systems on fixed hardware is to judge quantization and speed choices by **verified-answer precision**, not tokens per second.

## Artifacts
- Notebook (public): `foysalemonshanto/q27-xhigh-e25-nb6-submit`, version 1, submission 57851995.
- Model inputs: Merlin environment, W4A16 shards p1–p5, the FP8 PLE overlay, and the agent payload dataset (`q27-e15b-payload`).

## References
- ARC Prize 2025 Technical Report, arXiv:2601.10904.
- The ARChitects, ARC Prize 2024 solution.
- NVARC, ARC Prize 2025 solution.
- Poetiq, ARC-AGI refinement solver.
- Franzen, Duck agent harness for ARC-AGI-3.
- vLLM; Qwen3.8-Flash-Next model card.
