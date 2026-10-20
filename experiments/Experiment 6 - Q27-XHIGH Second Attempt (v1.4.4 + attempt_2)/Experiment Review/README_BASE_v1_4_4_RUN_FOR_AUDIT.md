# Base notebook Q27-XHIGH v1.4.4 (LB 26.94): notebook + its Kaggle run output, for audit

## What this is

| Item | Value |
|---|---|
| Notebook | `notebook/BASE_Q27-XHIGH-v1_4_4_LB26.94.ipynb` (sha256 `95995980…`) |
| Kaggle kernel | `foysalemonshanto/q27-xhigh`. Its latest version is cell-for-cell identical to this file (checked 2026-10-12). |
| Leaderboard | public LB **26.94**: submission 57032990 (competition rerun on the hidden test), scored 2026-10-12 |
| Run in this zip | the kernel's **commit run** = EVAL12 evaluation (`run_id 20261010T025027Z`). The same `run_id` comes back from the kernel's current output, so this is the run of this exact notebook version. |
| Not in this zip | the hidden-test rerun's logs. Kaggle does not return them; only the LB score is known. |

**Setup:**
- Qwen3.8-27B-FP8 on an offline vLLM 0.19.0 engine: TP 4, fp8 KV, MTP ×3, prefix cache (mamba "align");
- `max_model_len` 81,920, 16 concurrent;
- xhigh thinking up to 49,152 tokens, then a host-forced close (Qwen stop sentence + `</think>`, token-exact), then a grid answer of ≤ 8,192 tokens;
- temperature 0.6 / top_p 0.95 / top_k 20;
- russcore prompt, one prompt per test input.
- **attempt_1** = the model's grid; **attempt_2** = the test input, or its mirror if the model returned the input.

## EVAL12 result (12 evaluation tasks, 22 test outputs)

- **3/22** (13.64 %): 136b0064#0 natural close; 247ef758#0 natural close; 4c3d4a41#0 forced close.
- **Thinking:** 3 natural closes, 19 forced closes at 49,175 tokens.
- **Answers:** every unit parsed a grid.
- **Server:** 274.6 tok/s; MTP acceptance 0.661; 0 preemptions; KV pool 1,353,600 tokens, peak use 78.7 %; prefix-cache hit rate 0.572.
- **Time:** setup 789 s; scheduler 3,859 s.
- **Projection:** the 12.89 h projection in `run_report.json` comes from the hard EVAL12 tasks. The private hidden puzzles are much easier, so it does **not** apply to the hidden run.

## Files

| Path | What |
|---|---|
| `notebook/BASE_Q27-XHIGH-v1_4_4_LB26.94.ipynb` | the notebook (all code; dual mode: commit = EVAL12, rerun = hidden test) |
| `run_output/q27-xhigh.log` | the full Kaggle notebook log (stdout and stderr with timestamps) |
| `run_output/q27x/run_report.json` | the run report: config, engine, scheduler, scores, metrics, projection, deadlines |
| `run_output/q27x/records.jsonl` | one record per test input: mode (natural or forced_close), thinking tokens, phase tokens, time, grid parsed, hit |
| `run_output/q27x/answers/*.txt` | per test input: the end of the thinking (last 1,500 tokens) and the full answer text |
| `run_output/q27x/vllm_metrics.json` | vLLM Prometheus snapshots every 2 min (KV usage, spec decode, preemptions, prefix cache) |
| `run_output/q27x/vllm_frontend.log` | the vLLM engine log |
| `run_output/submission.json` | the commit-run `submission.json` (identity fallback for the 240 visible placeholder test tasks; in eval mode the model's answers go only to the records) |
| `run_output/Q27X_BUNDLE.zip` | the notebook's own bundle of the above |

## Updated notebook: dual answer (`updated_dual_answer/`)

| Path | What |
|---|---|
| `updated_dual_answer/Q27-XHIGH-v1_4_4_DUAL_ANSWER_EVAL.ipynb` | v1.4.4 plus a second-answer stage after the primary scheduler. attempt_2 = a re-sampled answer from the same reasoning prefix. On in EVAL12, off in the hidden rerun. **Not yet run on GPU.** |
| `updated_dual_answer/GPT_README_Q27_DUAL_ANSWER.md`, `GPT_test_q27_dual_cpu.py` | GPT's design notes and CPU tests |
| `updated_dual_answer/CLAUDE_VALIDATION.md` | Claude's validation: diff, tests, the expected-benefit caveat |
| `updated_dual_answer/diff_vs_v1_4_4/` | unified diffs of the 3 changed cells against the base |
| `updated_dual_answer/validation_outputs/` | Experiment 1 harness outputs (mixed / drain / rerun) and the phase-3-capable harness with its output |

## Context: current work (Experiment 6)

- **Only scope:** make **attempt_2** useful without changing the attempt_1 path.
- **Known evidence:**
  - Earlier independent runs on the same 22 outputs (v1.4 run 4 ∪ v1.4.1 run 6) would give 3/22 from two attempts, where each alone gives 2/22.
  - In this run every answer contains exactly 1 grid.
- **Constraint:** with MTP on, vLLM 0.19 ignores `min_p`, `logit_bias` and the server thinking budget (vLLM issue #39573). v1.4.4 is unaffected because it closes thinking itself.
