# Experiment 8: final report (augmentation scoring for attempt_2), 2026-10-13

**Status: CLOSED (NO-GO).** One GPU run: `foysalemonshanto/q27-xhigh-e8` v1, scriptVersionId 357913772, PROXY48 commit run.

Every number below was re-checked against the run files: `Experiment Results/Kaggle Runs/GPU PROXY48 E8 R5 (foysal q27-xhigh-e8 v1)/`.

## 1. What was tried

- **Goal:** improve attempt_2 (in v1.4.4 it is the test input or its mirror, which never hits) without touching attempt_1. The idea was NVARC's augmentation scoring (`calc_scores`).
- **Base:** E7 ORDERING_ONLY, i.e. v1.4.4 (LB 26.94) + exact Codex ordering.
- **Final design (GPT R5, approved):**
  - for each test input, as soon as its primary answer is finished:
    1. save the baseline;
    2. save the full thinking text;
    3. mine candidate grids from the thinking and answer (context-filtered);
  - 0 novel candidates: keep the fallback;
  - 1 novel candidate: use it as attempt_2 directly;
  - 2 or more: rank them with Codex 8-view augmentation and Q27 teacher-forced NLL (`prompt_logprobs`) in the same vLLM engine, primary-first.
- **History:**
  - R2 had a 30-min post-primary reserve and a train-consistent candidate pool;
  - Claude's audit showed that pool cannot add a correct answer (0/172 eval inputs), plus the OOM risk;
  - so the design moved to per-unit scoring and thinking-mined candidates (R5).

## 2. Results (verified)

| | Value |
|---|---|
| Units admitted / finished | 56 / **41** (15 lost; 35 tasks fully finished) |
| Eval pass@2 | **10 / 56 (17.86 %)**; E8 final = baseline, **net gain 0** |
| Modes | natural 15 → **9 hits (60 %)**; forced_close 26 → **1 hit (3.8 %)** |
| Primary engine | 353.5 tok/s, 16 active, 0 preemptions, KV max 73.6 %, MTP 0.663, prefix-cache hit 0.721, engine load 616 s, scheduler start 803 s |
| E8 counters | novel candidates 0 / 1 / 2+ = 22 / 9 / 10 units; 52 accepted and 385 rejected grids; 9 direct replacements; 1 scorer request started; 0 views scored; 0 ranked |

**The crash (09:43:01 UTC, notebook minute 124):**
- Primary-first admission held the scorer back until the queue was empty and a slot freed (`e8_primary_waiting_steps` = 32,726).
- The first scoring request (1,665 prompt tokens, `prompt_logprobs=1`) was in a step of 3,858 scheduled tokens, together with a 2,137-token phase-2 prefill and decodes; KV usage 63.9 %.
- **All 4 TP workers OOMed** in `LogitsProcessor._gather_logits` → all_gather: "Tried to allocate 790.00 MiB", with 349–417 MiB free and 484–552 MiB reserved but unallocated.
- 790 MiB = 1,665 × 248,832 (padded vocabulary) × 2 bytes: the bf16 full-vocabulary logits for every prompt position, gathered to every GPU.
- → `EngineDeadError`; 15 in-flight primaries were lost about 23 min before the drain. 5 of them are ARC Prize XHigh-solved tasks (`0934a4d8`, `20a9e565`×2, `7c66cb00`, `80a900e0`).
- The finished units, `submission.json` (valid identity test file, 240 / 259) and the reports were preserved. attempt_1 changed 0/41; attempt_2 changed 9/41.

**Candidate quality (Gate B):** on the 31 finished units where the baseline was wrong, a search of the full thinking text **in any format** (spaces, commas, no separators, lists) found:
- the complete solution grid in **0 / 31**;
- the solution as a contiguous block in **0 / 31**;
- at least half of the solution rows that differ from the input in only 3 / 31.

So **no parser or context filter can recover a correct candidate from these traces.** The upper bound of thinking-mining on these 31 units is 0. The 9 direct replacements had the wrong kind of shape (5×5, 3×4, 1×30 … against 20×20 or 27×27 answers): intermediate fragments. They gained 0 and lost 0.

## 3. Validation of the three audits (GPT audit 1, GPT audit 2, Claude)

| Claim | Who | Verdict |
|---|---|---|
| OOM root cause: full-vocabulary logits all-gather from `prompt_logprobs`; 790 MiB = 1,665 × 248,832 × 2 B | all; the padded-vocabulary arithmetic is from audit 2 | **Confirmed** (log, traceback, arithmetic) |
| Step had 3,858 scheduled tokens, a 2,137-token phase-2 request, KV 63.9 % | audit 2 | **Confirmed** (scheduler dump in the log) |
| 41/56 finished, 35 tasks complete, 353.5 tok/s, 0 preemptions, MTP 0.663, prefix hit 0.721, load 616 s | audit 2 | **Confirmed** |
| Novel distribution 22/9/10, 52 accepted / 385 rejected, 9 replacements (3 on already-correct units), net 0 | audits 1 and 2 | **Confirmed** |
| `scorer_disabled_reason = "primary_hard_stop"` is wrong (the real cause is engine death); `not_admitted = 0` / `aborted_at_hard_stop = 0` hide the 15 unfinished; report `version` says R4 | audits 1 and 2 | **Confirmed**: reporting bugs, not submission bugs |
| "R5 deviates from the approved architecture (interleaved scorer was not approved)" | audit 1 | **Rejected.** The user approved per-unit scoring (2026-10-13, "ওকে যুক্ত করে দিও", i.e. "OK, add it"), and GPT's R5 README states the scheduler change. The run-time isolation failure is real; the "unapproved deviation" is not |
| "Improve the Stage 2A parser / provenance using the 41 traces, then re-evaluate" | audits 1 and 2 (top priority) | **Refuted for these units.** The truth is absent from the text in any format (0/31), so no parser can help. Only the 15 unfinished units were not checked |
| "Natural vs forced is observational; do not change the budget on this basis" | audit 2 | **Valid caveat.** Natural closes may simply be easier tasks. Claude's "strongest evidence for 86k" was overstated; it is consistent with the plan, not proof of it. The user's 86k decision is a raise, which the rules allow; its effect is to be measured |
| "Even a 200-token scoring prompt is unsafe" | Claude | **Overstated, corrected.** 200 tokens ≈ 95 MiB bf16 + ≈ 190 MiB fp32 log-softmax, against 349–417 MiB free (+ about 0.5 GB reserved). It might fit; the safe limit is unmeasured |
| "0/88 in the earlier EVAL12 tails" not verifiable from the zip | audit 1 | Fair. Claude computed it from the local run folders (Experiment 1/2/6 answers); it was not in the bundle |
| "One-novel direct could lower the score on hidden if the fallback is correct" | audit 2 | Valid but negligible: identity/mirror is correct for 1 of 1,076 training and 0 of 172 eval inputs, and a train-supported fallback is already protected |
| "R5 rerun as is: NO-GO" | all | **Agreed** |

## 4. Lessons

1. **`prompt_logprobs` in the shared 4×L4 engine is unsafe at useful lengths.** Free memory beside the KV pool was only about 0.4 GB, and vLLM computes and all-gathers full-vocabulary logits for every prompt position. KV % says nothing about physical free memory. Any future likelihood scoring needs a measured, isolated memory budget.
2. **A failure in an auxiliary stage must never share the primary engine.** Interleaving turned an E8 error into a primary loss (15 units). CPU and fake-engine tests cannot catch GPU-memory faults; only a real-engine smoke test can.
3. **Q27's thinking does not contain the right answer when the final answer is wrong** (0/31 here, 0/88 tails before). **Ranking or mining cannot create correct attempt_2s; only new generation can.**
4. **Hits come from natural closes:**
   - natural 9/15, forced 1/26; over all our runs, forced closes hit 2/98;
   - 6/15 natural closes landed in 44–49k, right under the cap;
   - this is consistent with a larger thinking budget (the 86k plan); causality is unproven.
5. **Near misses exist:** `dd6b8c4b#0` 1 cell wrong, `dd6b8c4b#1` and `28a6681f` 2, `3dc255db` 4, `6e453dd6` 9; all better than copying the input. These are the targets for an independent second sample.
6. **Codex easy-first versus Qwen capability:**
   - tier 1: 7/37 hits; tier 2: 3/4; tier 3 never ran;
   - ARC Prize 120-task data: the NVARC fine score does not predict Qwen solves (AUC 0.435);
   - not proven to hurt here (no matched control, and the run was truncated), but it argues for capability-based ordering later (`resources/Puzzle Capability Ledger/`).
7. **Rough KV accounting at 16 sequences:** KV ≈ 1.25 × live tokens + about 11k per sequence (two-point fit). At an 86k cap, expect occasional preemption; watch `num_preemptions`.
8. **Process:** pre-run CPU audits predicted both the OOM and the empty candidate pool. A GPU microprobe of the risky component, before a full 3 h run, would have saved most of the run.

## 5. Next work (decided or proposed)

| # | Item | Status |
|---|---|---|
| 1 | **Next base:** ORDERING_ONLY + **86k budget** (think 77,824 + answer 8,192; MAX_MODEL_LEN about 106,496; EXPECTED_LIVE about 43k; asserts updated), **no E8 code** | user decision; build after approval |
| 2 | Measure in that run: natural closes between 49k and 77.8k and their hit rate; KV max / preemptions at 16 seqs; mean tokens per unit | with item 1 |
| 3 | attempt_2 from **new generation** (an independent second sample), not mining or ranking; target the near-miss and forced-close units | proposal, needs approval |
| 4 | Capability-based ordering using the ledger (all runs added) | later, near the end |
| 5 | If likelihood scoring is ever revisited: an isolated microprobe first (separate engine or post-primary, prompt cap), plus the reporting fixes (fatal-cause propagation, `unfinished_inflight`, version string) | parked |

## 6. Files

| Path | What |
|---|---|
| `Experiment Notebooks/Q27-XHIGH-v1_4_4_E8_PER_UNIT_R5_FINAL_CPU.ipynb` | the notebook that ran (sha `16c6c688…`) |
| `Experiment Results/Kaggle Runs/GPU PROXY48 E8 R5 (foysal q27-xhigh-e8 v1)/` | the raw log, stdout, `q27x/` outputs, submission |
| `Experiment Results/Analysis/GPU PROXY48 E8 R5 analysis (2026-10-13).md` | Claude's detailed analysis (sections 1–11) |
| `Experiment Results/GPT Outputs/` | GPT R2 and R5 deliveries |
| `Experiment Review/` | Claude's R2 audit, the R5 quick check, the GPU-run bundle for GPT (`E8_R5_GPU_RUN_FOR_GPT.zip`) |
| `README.md` | the original plan and status |
