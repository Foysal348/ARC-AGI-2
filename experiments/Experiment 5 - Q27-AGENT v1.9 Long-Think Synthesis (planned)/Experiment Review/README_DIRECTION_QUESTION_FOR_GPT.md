# Direction question after the v1.9.1 deep audit (2026-10-12)

**Request: a deep, unrestricted analysis.** Everything below is evidence plus Claude's observations, offered as suggestions. Nothing here limits what you may conclude or propose.

**Goal:** raise the Kaggle ARC Prize 2026 (ARC-AGI-2) leaderboard score toward about 60 with **Qwen3.8-27B-FP8 on 4× L4**.
- Constraints: 12 h, no internet, deadline 2026-11-02.
- Best so far: LB **26.94** (Q27-XHIGH v1.4.4 DIRECT).
- Other people have reached about 60 with this same model, so the target is reachable. The question is **how**.

**Standing user rules:**
- never use less than 49,152 continuous thinking tokens per request without explicit approval;
- xhigh effort;
- v1.4.4 is the base.

## 1. New fact found after your audit: your recommended fix already ran once

Your first choice for v1.9.2 was: offline `vllm.LLM`, host-forced close at 49,152 (stop sentence + `</think>`), then a program, then verification. **Q27-XHIGH v1.3.5 did exactly this.** It ran on GPU as Exp1 **run 3** on the same EVAL12 (12 tasks, 22 test outputs).

**Run 3 setup:**
- temperature **1.0** (top_p 0.95, top_k 20);
- MTP ×3, prefix cache off;
- 49,152 thinking, then a forced close, then a 16,384-token answer;
- the prompt asked for `def solution(grid)` **plus** a JSON `{"test_outputs": [...]}`.
- **Attempt order:** verified code → direct JSON grid → unverified code → test input.

**Run 3 result:**
- **1/22** (4.55 %).
- The forced close worked: **12/12** tasks were force-closed at 49,175 tokens, and every task returned code.
- Only **one** program passed training: 4c3d4a41, train 2/2, but **wrong on the test input**.
- The single hit came from an unverified program (247ef758 #0).

**Per task (`records.jsonl`):** train correct / total = 4c7dc4dd 0/2, 20a9e565 0/3, 898e7135 0/2, 6e4f6532 0/2, 4c3d4a41 **2/2**, 136b0064 0/3, 21897d95 0/4, 8b7bacbf 1/4, 247ef758 2/3, 88bcf3b4 0/5, b6f77b65 1/5, 2b83f449 0/2.

## 2. What the DIRECT runs show (grid answer, one prompt per test input)

| Run | Version | Temperature | Hits / 22 | Natural thinking closes (unit, thinking tokens, hit) |
|---|---|---|---|---|
| 4 | v1.4 | 0.6 | 2 | 136b0064#0 26,523 ✔; 4c3d4a41#1 29,173 ✘; 4c3d4a41#0 34,958 ✔; 247ef758#1 41,799 ✘ |
| 6 | v1.4.1 | 0.6 | 2 | 136b0064#0 23,818 ✔; 4c3d4a41#0 38,987 ✘; 247ef758#0 39,905 ✔; 247ef758#1 42,733 ✘; 4c3d4a41#1 44,764 ✘ |
| 7 | v1.4.4 | 0.6 | 3 | 136b0064#0 25,934 ✔; 247ef758#0 36,770 ✔; 247ef758#1 46,470 ✘ — plus 4c3d4a41#0 hit **after a forced close** |

**Pattern:**
- Across runs 4, 6 and 7, **6 of 7 hits came from thinking that closed naturally** before 49,152.
- Natural closes happen only on the 3 easiest EVAL12 tasks: 136b0064, 247ef758, 4c3d4a41.
- Forced-close answers were right 1 time in about 55.
- EVAL12 is much harder than the hidden set. Its EDA mirror gap is 25.4 pp, and the LB has been about 2× the EVAL12 percentage.

## 3. The agent attempts so far

| Attempt | Mechanism | EVAL12 agent result |
|---|---|---|
| Exp1 run 3 (v1.3.5) | 49k forced close → `solution()` + JSON | 1/22; 1 train-verified program, wrong on test |
| Exp4 run 4 (v1.8.3) | REPL tool agent, short turns, forced submit; the 6,144 budget was silently NOT enforced (MTP bug) | 0/22 |
| Exp5 v1.9.1 | no tools, one long request; the 49,152 budget was silently NOT enforced (MTP bug) → 10/12 tasks had no content | 1/22 (136b0064, natural close at 49,982, also solved by DIRECT) |

**What never happened:** no agent run has produced a correct answer on a task that DIRECT missed.

## 4. Questions (answer them, or replace them with better ones)

1. With run 3 in view, is a host-forced-close v1.9.2 (temperature 0.6, `transform`-only prompt, prefix cache) still worth a GPU run? What exactly would you expect to change compared with run 3?
2. **Natural thinking closes look like the key to hits.** What raises the share of requests whose thinking closes naturally, or the quality of what comes after a forced close, without cutting thinking below 49,152?
   - prompt changes, temperature, a different stop sentence, task ordering, more samples per task?
3. For the LB (hidden set, easier than EVAL12), is the bigger lever **coverage**?
   - More finished requests in 12 h: puzzle ordering (NVARC `get_puzzle_priority`), a dynamic KV controller, `max_num_seqs` above 16. Measured KV: 1,353,600 tokens, 68 % peak at 12 × 60k, 0 preemptions.
   - Or a second DIRECT sample per test input, i.e. 2 attempts from 2 samples instead of the test-input fallback?
4. Teams reaching about 60 with this model: what is the most likely architecture, given everything in this zip? Suggest a concrete next experiment that can be measured within one GPU run.
5. Anything else you see.

## 5. Files

| Path | What |
|---|---|
| `run3_v1_3_5/` | the v1.3.5 notebook that ran, `records.jsonl`, `run_report.json`, `vllm_metrics.json`, the full output log |
| `direct_runs/run4_v1_4/`, `direct_runs/run6_v1_4_1/`, `direct_runs/run7_v1_4_4/` | `records.jsonl` and `run_report.json` for each DIRECT run; run 7 also has every answer file (thinking tail + answer) |
| `notebooks/Q27-XHIGH-v1_4_4.ipynb` | the LB-26.94 notebook |
| `v1_9_1_gpu/` | `eval_report.json` and `attempts.jsonl` from the v1.9.1 GPU run |
| `prior_review/README_v1_9_1_GPU_BUG_REVIEW.md` | the previous review README (your deep audit answered it) |
| `resources/` | `puzzle_order.py` (NVARC ordering, verbatim), `select_eval20.py` (EDA mirror subsets) and their READMEs |
