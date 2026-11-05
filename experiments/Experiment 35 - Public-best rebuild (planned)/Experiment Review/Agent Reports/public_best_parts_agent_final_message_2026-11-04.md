# Agent final message: public best parts reader (verbatim, 2026-11-04)

I read all seven sources in full (the andy2709 launcher script is 1,471 lines) and wrote the report. To fill the "present in Domdolus?" column I searched the Domdolus cell extracts in Build/domdolus_cells/ for each item; those are grep checks, not full reads. Nothing was run or changed apart from writing the report.

Report: D:/ARC AGI 2/experiments/Experiment 35 - Public-best rebuild (planned)/Experiment Review/Agent Reports/public_best_parts_report_2026-11-04.md

It has the four sections you asked for: facts per source with quotes, a table of 27 engine and 21 harness techniques, harness ideas ranked by evidence, and the Domdolus ablations and negatives. I added a section 5 with a short list of what looks addable.

**Things Domdolus does not have, with the measured effect each source states:**
- **Speculative decoding (MTP) is off in Domdolus** (`MTP_TOKENS = 0`). Their server cell drops MTP when memory runs out ("retrying without MTP"), so memory is the likely reason (my inference). notnitsuj runs MTP K3 at 48 sequences and 1,087–1,120 tok/s. A draft fine-tuned for accepted length gave +6.1 % tok/s, and an int8 draft head over the 32k most-used tokens gave +19 %. Folding accepted tokens into one BF16 DeltaNet checkpoint allowed 49 % more concurrent sequences. andy2709 refit its draft to the pruned model: accept length 2.807 → 2.992.
- **notnitsuj harness pieces:**
  - Voting over verified programs from about 3 sessions per task, with new sessions going to the task with the fewest so far. Verified sessions were right 75 % of the time.
  - 11 content invariants plus the `ak` helper library: +2.6 [+0.2, +5.2], precision 0.82 vs 0.72. Domdolus checks train-pass only.
  - Tie-break on kept symmetry and no test-only code: +0.54. Tasks with 2+ votes and a lead go to the back of the queue: +0.8.
  - Warm start from the best unverified program: +0.21 right answers per session, 17 % fewer tokens.
  - A drain near the deadline: +1–3 % tokens. Domdolus stops all tries 1 minute before the limit.
  - Two sandbox fixes: the import alias (+2.25) and correct traceback lines (+0.88). Both are paired A/Bs with confidence intervals.
- **andy2709 harness pieces:**
  - A 12,288-token cap per request with a "write solution.py now" nudge: 6.9 % fewer tokens, +1.1 to +2.2 points. Domdolus sends no max_tokens. This caps a single reply, but it touches our 60k thinking-floor rule, so it needs your approval.
  - A −1 logit bias on "Hmm"/"Alternatively": 4.5 % fewer tokens.
  - Second attempts go to unverified tasks first: 4.6 vs 2.6 solves per million tokens.
- **andy2709 SGLang kernels, each about −4 to −6 % decode step on 4×L4:** a fused MoE-down + all-reduce, the hyper-connection up projection kept locally in INT8 (also −18 % prefill), a CUDA hyper-connection mix kernel, and a prefill CUDA graph (−6.5 %). Porting them to vLLM 0.31 is kernel work.
- **Host-memory collectives:** +8.6 % at notnitsuj, neutral at andy2709's 10 sequences, and Domdolus found NCCL faster than vLLM's custom all-reduce. The evidence is mixed.
- **Model work:** RL on the quantization scales gave +7.0 pass@2 [+3.8, +10.6]. Expert pruning must count tool-call routing, or verified sessions fall from 0.78 to 0.34. Both are training projects.

**Domdolus' own ablations:**
- **Clear positives:** 3 rounds (as is / transposed / mirrored) took 51.67 to 65.14. Images-only grids beat text by about 8 points (66.81 vs 58.75). Harness fixes gave +9. The 48–64k memory window plus prompt fixes reached 78.47, and 2 tries at 150k tokens reached 82.08.
- **The big negative:** RAM offload off + token-id reuse + informed try 2 scored 59.72. That row changed three things at once, so it does not prove any one of them bad. The informed try 2 is off in the final notebook ("no more 'evolve style'").
- **Dropped:** the NVARC port, the vision-network pipeline, DeltaNet-state test-time tuning (no effect), the 4-bit model with 8-bit KV in RAM (did not run in time), the 27B, 35B-A3B and Swift variants (fell short), DeepSeek V4.1 Flash (too slow on L4), the transposed/mirrored rounds and the third try.
- **Noise:** the same notebook #16 scored 72.36–83.47 across five submissions, so 83.47 is the best draw and the mean is about 77.5. Most single-row changes of 5 points or less are inside that spread.

**Where the sources disagree:** on temperature and reasoning effort, andy2709 uses 0.7 / medium and retired both low and xhigh effort. notnitsuj says 1.0 is best for Flash-Next, and Domdolus uses 1.0 / xhigh. The report makes no recommendation here.
