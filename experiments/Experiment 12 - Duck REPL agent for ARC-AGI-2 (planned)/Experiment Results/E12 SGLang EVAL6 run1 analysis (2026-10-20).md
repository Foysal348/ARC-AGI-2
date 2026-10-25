# E12 rev 4 EVAL6 run 1 (hosen42, 2026-10-20): analysis

**Run:**
- Kernel `hosen42/q27-xhigh-e12-duck-repl-sglang-eval6` v1, notebook sha `53a66856…`;
- COMPLETE in about 110 min of wall time;
- outputs in `Kaggle Runs/E12 SGLang EVAL6 run1 (hosen42, 2026-10-20)/`.

Script: `Analysis scripts/eval6_analysis.py` (output in `eval6_analysis_output.txt`). The plan is in `Experiment Review/E12 rev 4 EVAL6 - GPT audits validated (2026-10-20).md`.

## Result

- **Fully solved: 3 / 6 tasks** (136b0064, 135a2760, 4c3d4a41); 4 / 8 test inputs; task-norm 50.0 %.
- **Baseline:** E12 EVAL30 Duck (12,288 output per request) solved **0** of these 6. That is how they were selected: S49 / R5 / DIRECT solved them.
- **Decision rule:** ≥ 2-3 of 6 → full rev 4 EVAL30.
- **Wrong accept: 1.** b5ca7ac4 b1 passed train 3/3, test wrong (overfit).

## Exposure and confounds

| Check | Result |
|---|---|
| Branches run | 12/12 (all finished: 4 accepted, 8 stopped by the 100k soft cap `stop_requested`; none by the deadline) |
| Solve time | 3,744 s (62 min) of the 90 min window. The window did not bind; the per-branch cap did |
| Request timeouts | 0 |
| Context trims | 10 (about one per branch, after the first long reply); overflow recoveries 0 |
| Serializer tampering in model code | none |
| Tokens | 1,217,904 generated in 96 requests; prompt cache 78.8 %; 6 running for most of the run |

## Pattern (per-request completions)

- **Accepted branches** (4):
  - first reply 21-45k;
  - then a long **check → fix loop** of short requests (8-22 requests, mostly < 5k).
- **Failed branches** (8):
  - 2-7 requests, mostly long replies of 20-60k (10 requests hit the 60,000 ceiling);
  - they reached the 100k cap after few tool iterations.
- **No accepted branch had a 60k reply.**

## Branch 0 vs branch 1

| Branch | Accepted |
|---|---|
| Branch 0 | 0 / 6 |
| Branch 1 | 4 / 6 |

- The prompts are identical. Only the queue position and sampling differ.
- With n = 6 this can be chance (Fisher p ≈ 0.06). It is not a confirmed effect.

## Cost (for the next capacity check)

- **Per task:** about 10 min of GPU time (12 branches in 62 min with 6 live).
- **Hidden set:** 240 tasks would need about 40 h against about 11 h of solving. Coverage, not accuracy, is the open question for a submission.
