# E12 Duck REPL agent, SGLang EVAL30 run 1: analysis (2026-10-19)

## Run

- **Kernel:** `hosen42/q27-xhigh-e12-duck-repl-sglang-eval30` v1 (commit on 2026-10-19, L4x4, notebook sha b65fb9dc).
- **Timing:** started about 10:47 UTC, COMPLETE at 14:56 UTC. Solving ran from 733 s to 14,779 s; the deadline was 13,367 s.
- **Outputs:** `Kaggle Runs/E12 SGLang EVAL30 run1 (hosen42, 2026-10-19)/` (190 files, 18.4 MB, full transcripts).
- **Health:**
  - engine W1 ready, 12 streams;
  - 454 requests, 0 failures;
  - cache 59.6 % of prompt tokens.

## Score: below every earlier run on the same 30 tasks / 48 inputs

| Run | Hits | Task-norm |
|---|---|---|
| **E12 Duck (this run)** | **7/48** | **13.3 %** |
| E11-05 S49 Poetiq (SGLang) | 11/48 | 23.3 % |
| E10 R5 Poetiq (vLLM) | 16/48 | 33.3 % |
| E10 ORDER2 DIRECT 86k | 12/48 | 27.8 % |
| NVARC `arc-2026d-with-codex` | 10/48 | 22.8 % |
| ARC Prize xhigh | 8/30 tasks | — |

- **Solved by Duck (4 tasks, 7 inputs):** bf45cf4b, f931b4a8, 247ef758 (2/2; the only task no other run fully solved), 65b59efc.
- **Solved by S49 or R5 but not by Duck:** 135a2760, 136b0064, 4c3d4a41, 62593bfd, 6e4f6532, 78332cb0, 88bcf3b4, 898e7135, 8b7bacbf, b5ca7ac4.
- Per-task table: `_scratch/e13_validate/eval30_cmp.py` output, reproduced from `e12_run_report.json`.

## Causes, all measured

### 1. Only one branch per task ran

- 29 of 30 branch-1 runs never started (0 turns). The exception is eee78d87 b1, with 2 turns.
- With no per-branch cap (`E12_MAX_GEN=None`, as in the milestone), each unsolved branch 0 ran until the deadline: 12-20 turns and 61k-124k generated tokens, with **0 submissions**.
- The planned 2 independent attempts (D66/L66: independent attempts are complementary) therefore never happened.

### 2. KV thrash: about 35 % of all prompt tokens were prefilled again

- **Volume:** 454 requests; prompt tokens 23.78 M, of which 14.18 M were cached.
- **Context size:** prompt length p50 47.9k, p90 97.7k, max 117.3k tokens.
- **Evictions:** 144 of the 423 non-first requests had < 50 % cached (cached = None / 192 / 576 tokens), i.e. the branch's prefix had been evicted. Those requests alone account for **8.39 M of the 9.60 M uncached prefill tokens**.
- **Why:** 30 time-sliced branches × about 50-100k context exceed the about 870k-token KV, so the radix cache evicts a paused branch's prefix (a paused branch is waiting on a tool, a yield or the gate).
- **Effect on speed:**
  - generation over the solve window was about 187 tok/s (2.36 M tokens / 12.6 ks);
  - decode itself ran at about 420 tok/s at 11-12 running, even at ≥ 60k context (decode summary).
  - So roughly half the time went to re-prefill and idle.

### 3. Output cap

- 122 of 454 requests (27 %) ended at the 12,288-token output cap.
- The completion median is 2,983.

### 4. Precision

- 7 branches were accepted (train-exact). Only 4 were correct on the test inputs.
- 3dc255db, b5ca7ac4 and 4c3d4a41 passed train but failed test: an overfit rule.

### 5. Coverage

- At about 76k generated tokens per started branch, 30 tasks took the whole 3.5 h.
- 240 hidden tasks would need about 28 h at this rate.

## What this means

- The Duck harness itself worked: no errors, structured tool calls, submit path and selection all fine.
- But in this configuration, on 4xL4 with 27B, it is **slower per solve** than Poetiq.

The biggest measured levers:
1. A per-branch token cap and a fair start for branch 1 (stop hopeless branches).
2. Keep the number of **resident contexts within KV**, i.e. admit no more branches than KV holds, so no prefix is evicted.
3. A lower context drain.

These are proposals only. No code was changed, pending the user's decision.

## Consequence for E13 (Flash-Next)

- Measured Duck contexts on ARC-2 are p50 48k / p90 98k.
- Flash-Next's estimated fp8 KV of about 170-250k tokens would hold only about **2-4 such contexts**, against about 12 for 27B W1.
- Unless contexts are kept much smaller, KV thrash would be far worse than here.
- This is now the top E13 risk; PLAN rev 3 risk 1 is updated with these numbers.
