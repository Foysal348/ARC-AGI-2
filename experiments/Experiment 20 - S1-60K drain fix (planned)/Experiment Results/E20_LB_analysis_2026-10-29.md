# E20 (S1-60K + drain 15 min) LB analysis (2026-10-29)

E20 scored **LB 31.53** (submission 57710684, notebook `q27-xhigh-e20-s1-60k-drain15` v2, submitted 2026-10-27 18:48).

E20 = S1-60K with one change: in the hidden rerun only, `drain_margin_sec` goes from 8 to 15 min. The clock moves as follows:

| Clock point | S1 | E20 |
|---|---|---|
| admit | 11:45 | 11:38 |
| drain | 11:47 | 11:40 |
| hard stop | 11:55 | 11:55 (unchanged) |

## 1. The S1 family on the LB

| Submission | Date | Thinking cap | Change | LB |
|---|---|---|---|---|
| S1-60K v2 | 10-16 | 60k | — | **34.58** |
| S1-49K v2 | 10-17 | 49k | budget | 32.08 |
| S1-55K v2 | 10-19 | 55k | budget | 31.81 |
| S1-60K v3 | 10-22 | 60k | none (the same notebook again) | 31.94 |
| **E20 drain15 v2** | 10-27 | 60k | drain 8 → 15 min (rerun only) | **31.53** |

- Mean of all five: 32.39; median 31.94.
- Four of the five lie within 0.55 points (31.53-32.08).
- **The same notebook gave 34.58 and 31.94: a 2.64-point gap with no change.**
- The four 55-60k runs have sd 1.42, consistent with the noise sd of about 1.5 we already assumed.

**Conclusions:**
1. 34.58 was a high draw. S1's expected LB is about 32 (31.9-32.4), not 34.6. Every comparison with "our best" should use about 32 ± 1.4.
2. E20 against S1-60K v3 (identical except the drain) is -0.41, about 0.3 sd. Against the mean of the two S1-60K runs (33.26) it is -1.73, about 1.2 sd. **Neither is a measurable effect.** The drain change neither helped nor hurt detectably.
3. One LB point is about 1.2 of the 120 LB tasks (task-normalised, 2 attempts). One sd (1.4) is about ±1.7 tasks. A change has to move about 3 LB points (2 sd, about 3.5 tasks) to show up in one submission.

## 2. What the drain change could buy (expected value)

Evidence comes from the D66 GPU run (same S1 scheduler, same RERUN margins; `Experiment 17/Experiment Review/AGENT_S1_rerun_path_audit_2026-10-27.md` F1):
- At the 8-min drain, 15 units were active.
- Re-prefill of prompt + thinking (+609k tokens) ran at about 1.7k tok/s and was still running at the hard stop. Generation was +19.6k tokens in 8 min (normal: about 120k).
- 14 of 15 drained units produced no answer.

| Term | Estimate |
|---|---|
| Drained units per hidden run | about 14-16 (the tail tasks of ORDER2) |
| With 15 min: re-prefill (~700k tokens / 1.7k tok/s) + answer (1.5-3k tokens at ~15 tok/s) | 7 + 2-3 min: fits |
| Solve rate of a partial-thinking forced close | 10-20 % (audit estimate, not measured) |
| Gain | 1.5-3 hidden tasks of 240 → about 0.75-1.5 LB tasks → **+0.6 to +1.3 LB** |
| Cost: admit 7 min earlier (about 1 % of the ~11.3 h of decode) | about -0.2 to -0.4 LB |
| **Net expected** | **about +0.3 to +1.0 LB**, under 1 sd: undetectable in one submission |

The commit profile (S1COMMIT4, 4 tasks, 4/4) never runs the drain. **No run has measured the drained units' answer rate under 15 min.** The rerun log is not visible, so it is unknown whether drained units actually answered in the hidden run.

## 3. E20's commit run: engine health (no regression)

| | E20 v2 commit |
|---|---|
| Engine | vLLM 0.19, FP8, fp8 KV, MTP-3, FlashInfer, prefix caching (align), 16 streams |
| Load | 552.6 s (earlier vLLM loads: 530-582 s) |
| KV tokens | 1,353,600 |
| Commit score | 4/4 (pass-1) |
| Errors / aborts | 0 / 0 |

The commit-run engine was normal. Nothing points to a broken run; 31.53 is in S1's normal range.

## 4. What this means

1. **Baseline:** S1's level is about 32. 34.58 should not be the bar any new method must beat to count as "better". Judge a new method against about 32, and only an LB above about 35 (≥ 2 sd) is a clear win from a single submission.
2. **Drain15:** keep it (the mechanism is sound and it shows no harm), but it is not a lever. The real fix, an in-request forced close without re-prefill, is worth the same +0.5-1.5 at most.
3. **Final submission choice:** S1-60K and E20 are statistically the same notebook. Choosing on a single LB value (34.58) picks noise. The private LB will regress toward about 32 for either.
4. **Where +3 or more can come from:** only from coverage and quality, not from the clock.
   - In 12 h, about 205-225 iteration-1 requests serve 240 hidden tasks; about 20-35 tasks get identity, and no task reaches iteration 2.
   - The candidates are a faster engine at the same accuracy (E21 vLLM FP8 SSM16, 18-19 streams) and a stronger method per request.
5. **The measurement gap:** the drain effect can only be measured with an eval run on the RERUN clock (EVAL120, RERUN_ALL margins), counting drained units, answers and correct answers.
