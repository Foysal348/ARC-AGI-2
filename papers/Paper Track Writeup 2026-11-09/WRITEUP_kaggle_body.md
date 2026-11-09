Team Hack2Publish: Md. Hamid Hosen, MD. Foysal Emon Shanto, Md Esfer Abdus Sami.

**Main result:** a Qwen3.8-Flash-Next tool agent on 4×L4 scored **54.17** on the ARC-AGI-2 public leaderboard (submission 57851995). A faster 3-bit build of the same agent verified as many answers but was right less often (61/79 vs 50/79 on the tasks both builds reached), so we propose verified-answer precision as the acceptance test for any speed or compression change.

**Main paper (PDF, full detail):** attached to this Writeup under *Files* (Hack2Publish_Verified_Answer_Precision_ARC_AGI_2.pdf, 16 pages).

**Companion paper:** *Decomposing the Score of an NVARC-Derived Pipeline on ARC-AGI-2: Generation, Selection, and Run-to-Run Noise* (same authors, 2026), attached under *Files* (Hack2Publish_Companion_NVARC_Decomposition_ARC_AGI_2.pdf, 18 pages).

**Submission notebook:** https://www.kaggle.com/code/foysalemonshanto/q27-xhigh-e25-nb6-submit (version 1). The "q27" prefix in its name is historical: the notebook serves Qwen3.8-Flash-Next, not the 27B model.

This Writeup is a summary. The PDF is the main document, with all tables, confidence intervals, the full agent and engine specification, and the data sources.

## The idea in one paragraph
On fixed hardware you can make an ARC-AGI-2 system faster in two ways: compress the model, or run more sessions at once. Each change raises coverage, the share of hidden tasks that get a full attempt. A tool agent grades its own answers, though: a program that reproduces every demonstration pair counts as verified. A weaker model passes that test with the wrong rule more often. **We judge a speed or compression change by the correct verified answers it produces at the deployment's time per task and task order, and we require that verification precision does not fall on the tasks both configurations reach.** Tokens per second alone is not enough.

## The submission
- **Model and engine:** Qwen3.8-Flash-Next (48 layers, 512 routed experts, top-10 routing). It is served offline with Merlin v3.0.0, a vLLM fork, on Kaggle's 4×L4.
  - 4-bit routed experts (W4A16) and an FP8 n-gram embedding table
  - FP8 KV cache (738k tokens, measured at the same memory setting)
  - MTP speculation with 3 draft tokens
  - up to 9 concurrent sessions, 104k tokens per attempt
- **Agent:** one Python tool. The model writes `transform(grid)` and calls `check()`, which runs it on every training pair and shows the wrong cells. It then revises the program and calls `submit()`, which the host re-verifies.
- **Attempts:** attempt 1 runs for every task, in priority order. Attempt 2 is an independent session.
- **Run safety:** a complete `submission.json` is written at the start and rewritten after every attempt.

*Figure 1 (media gallery): the submission pipeline.*

## What we measured
We compare the submitted W4A16 build with a faster "3-bit" build of the same agent (klee100's mixed W3/W2 AutoRound experts, about 3 bits per weight).

1. **Verified programs are less precise at 3 bits.** We use the first attempts of 6-hour runs on the 92 public evaluation tasks both builds started.
   - Both builds verified 79 programs.
   - W4A16: 61 were right on the test (77.2 %).
   - 3-bit: 50 were right (63.3 %).
   - Fisher p = 0.08 (p = 0.043 counting host-accepted submissions only).
   - On 15 matched tasks: 24/26 against 22/28.

   *Figure 3 (media gallery): share of train-verified programs that are right on the test.*
2. **Most false verifications are underdetermined rules.** We read by hand all 15 train-verified wrong programs from two smaller quality runs (W4A16 on 15 tasks: 2; 3-bit on 30 tasks: 13):
   - 11 fit a rule that the demonstrations do not determine;
   - 2 hard-code positions;
   - 1 meets a test unlike the demonstrations;
   - 1 gets the output size wrong.

   Both W4A16 errors produced the same wrong grids as the 3-bit build, so on those tasks the cause is the task, not quantization.
3. **Coverage against precision.**
   - The 3-bit build generated 2.3× the tokens, started all 120 tasks (a 12-session W4A16 variant started 92), and won on evaluation: 65.0 against 53.75.
   - On the 92 tasks both builds reached, it solved fewer: 57 against 63.
   - On the leaderboard it scored 49.58, against our 54.17. The two submitted builds also differ in task order, session count and scheduler, so this is consistent with the precision gap, not proof of it.
4. **FP8 KV costs no measured accuracy.** In a paired 15-task test, FP8 and BF16 each solved 13 tasks. FP8 fits 9 sessions instead of 5.
5. **Evaluation is a weak leaderboard proxy.** We found three rank inversions:
   - our 3-bit vs W4A16 builds;
   - our 27B solver: 77.8k thinking cap gave 48.61 % on eval but 21.39 on the LB, while a 60k cap gave 34.58 on the LB;
   - two public Flash-Next notebooks we ran under equal windows: the one with the higher eval score has the lower published LB.

*Figure 2 (media gallery): our leaderboard history across four method families.*

## Why it works
- **The demonstrations are a unit test of varying strength.** With two or three pairs, or with a test grid larger than all of them, several programs fit. In our runs the 3-bit build more often settled on a rule that fits the demonstrations but not the test; we hypothesise that quantization weakens the preference for the simplest rule.
- **The trade-off depends on reach.** A faster build wins only on tasks the slower one never reaches.
  - Our 12-hour budget gives about 30–40k tokens per hidden task.
  - Correct first attempts needed a median of 6.3k tokens on usually-solved tasks and 23.6k on sometimes-solved ones.
  - The submission's priority order runs tasks with many demonstrations first, so the tasks it leaves unreached are likely the hard ones. In the 6-hour runs the order was alphabetical.
- **Memory sets concurrency on L4.** Too many sessions for the KV pool cause preemption: a 12-session W4A16 run on a pool sized for 9 had 2,028 preemptions in 6 hours.

## Negative results and a path forward
**What did not help:**
- early stopping of hopeless attempts (AUC 0.61–0.64; out of sample it lost 18 % of passing attempts);
- pause and resume;
- thinking caps above or below 60k for the 27B model (separate notebooks, not a controlled test);
- difficulty-predicted task ordering;
- a dense 27B tool agent;
- the faster 3-bit build.

**Headroom:**
- Our best single evaluation run answered 107 of 172 outputs.
- The union of all our runs answered 143.
- Adding two public notebooks gives 161.
- Different systems fail on different tasks.

**Next steps:**
1. Use the precision acceptance test for every speed change.
2. Add checks that target underdetermined rules: invariants, symmetry, and rejecting hard-coded structure.
3. Cut duplicate second attempts, but only where an independent check agrees. In the 6-hour 3-bit run, later attempts rescued 9 tasks.
4. Route tasks with no verified program to a different solver.

## Universality and limits
The rule applies to any execute-and-check agent on fixed hardware: code with unit tests, programming by example, and data scripts. It cannot catch errors the verifier cannot see; in ARC these are the underdetermined rules.

Limits:
- Leaderboard comparisons rest on one run per build; the noise is about ±1.5 points.
- The precision gap is at the edge of significance (p = 0.04–0.08).
- The 6-hour W4A16 run differs from the submitted W4A16 build; the 6-hour 3-bit run is the submitted 3-bit build.

## Prior work and credit
Our agent is Franzen's fork of the Tufa Labs Duck harness, with an ARC-AGI-2 adapter. The engine is halt95's Merlin. Other 2026 teams on the same hardware scored 75.7–87.2 at about 680–1,240 tok/s (one of them with a different model). We sit at the low-throughput point, with no pruning and no training. notnitsuj reported first that compression shows up as less precise verification, and our matched-task measurement replicates it. The full references are in the PDF.

**Licence:** our submission notebook and our ARC-AGI-2 adapter (`arc2_duck.py`, in the public dataset q27-e15b-payload) are public under Apache 2.0, as are Franzen's Duck fork and our W4A16 model parts (p1–p5). Upstream model weights keep their own licences.
