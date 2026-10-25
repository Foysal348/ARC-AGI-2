# 11 — 45K30K and 40K35K GPU commit results (rev 3g)

**Update (same day):** the 40K35K v1 commit did complete. The CLI downloads only the latest version; v1 was fetched with `version_label="v1"` into `Experiment Results/Kaggle Runs/E15B SUBMIT 40K35K commit1 v1 (foysal, 2026-10-25, 560ed8b8, sv361956451)/`. F9 and the "Variant" section are superseded by the 40K35K v1 section at the end.

**Date:** 2026-10-25.
**Commit run:** 20 eval tasks (32 test queries), both attempts.

| Run | Account | Notebook sha | Result |
|---|---|---|---|
| 45K30K rev 3g | esfersami50 | `5d8069b9` | **COMPLETE: 19 / 32 queries, 13 / 20 tasks, task-norm 65.0 %** |
| 40K35K rev 3g | foysalemonshanto | `560ed8b8` | version v1 (scriptVersionId 361956451) **COMPLETE: 11 / 32 queries, 7 / 20 tasks, task-norm 37.5 %**. Version v2 (= submission 57609813, PENDING) is the commit that was cancelled at ~2 min. |

**Outputs:**
- `Experiment Results/Kaggle Runs/E15B SUBMIT 45K30K commit1 (esfersami50, 2026-10-25, 5d8069b9)/`. `kernel_log.txt` is the flattened log.
- `Experiment Results/Kaggle Runs/E15B SUBMIT 40K35K commit1 (foysal, 2026-10-25, 560ed8b8)/` holds only the PREPARE / COPY state.

## 45K30K engine

| Item | Value |
|---|---|
| READY | 542 s; solve started at 804 s |
| Solve time | 2,998 s (50 min) for all 40 attempts; window 92 min |
| KV | 852,152 tokens |
| Running | 16 for 337 / 575 decode lines; queue 0 throughout |
| Retract / OOM / traceback / relaunch | 0 / 0 / 0 / 0 |
| Gen tok/s p50 by running | 16: 451; 15: 470; 14: 433; 12: 383; 8: 312; 2: 144; 1: 78 |
| Accept length p50 | 3.8 |
| Cache hit | 94.0 % of prompt tokens |
| KV full_token_usage peak | 0.71 (first wave, 38 min) |
| Generated | 1,155,991 tokens (b0 659,633, mean 33.0k; b1 496,358, mean 24.8k) |
| Ideal makespan | 2,645 s (sum of attempt durations 42,323 s / 16) |

`E15B WARNING: mean server running 14.17` is the end tail (4-1 attempts left). While the client had all 16 streams busy, the mean was 15.6.

## Score

| | Count |
|---|---|
| Tasks hit | 13 / 20 |
| Attempt 1 (45k) accepted and correct | 11 |
| Attempt 2 (30k) correct | 8 |
| Both attempts correct | 6 |
| Attempt 2 only (the remedy works) | 2: 136b0064 (end-of-attempt reasoning harvest at 30,000), 1ae2feb7 (b0 accepted a wrong program) |
| Queries hit | 19 / 32: pass1 17, pass2 2 (247ef758 q0, 9aaea919 q0) |
| Selection loss (a correct program existed but no hit) | 0 |
| Accepted but test-wrong | 6 of 24 accepts: 1ae2feb7 b0, 221dfab4 b0 and b1, 247ef758 b1, 4c3d4a41 b0, 9aaea919 b1 |

**Attempt 1 needed more than 35k on 4 tasks**:

| Task | First transform | Generated | Note |
|---|---|---|---|
| 135a2760 | 41,897 | 43,065 | accepted in the commit turn |
| 9aaea919 | 41,611 | 44,688 | commit turn |
| 247ef758 | 30,597 | 38,557 | accepted at 38.5k |
| f931b4a8 | 36,283 | 36,884 | |

Under 40K35K (commit zone from 35,904), 135a2760 and 9aaea919 would not have had a transform in attempt 1.

## Per attempt

Columns: A = accepted; ok = test-correct (eval); 1stTf = the first `transform` (tokens); nudge = the provisional nudge seen (tokens); commit = commit-turn resumes; cut = cut tool calls dropped.

| task | b | A | ok | gen | 1stTf | nudge | commit | cut |
|---|---|---|---|---|---|---|---|---|
| 135a2760 | 0 | A | ok | 43065 | 41897 | 24608 | 1 | 0 |
| 135a2760 | 1 | - | x | 30000 | 26947 | - | 1 | 0 |
| 136b0064 | 0 | - | x | 45000 | - | 36753 | 1 | 0 |
| 136b0064 | 1 | (harvest) | ok | 30000 | 30000 | 24190 | 0 | 0 |
| 16de56c4 | 0 | A | ok | 18242 | 17492 | - | 0 | 0 |
| 16de56c4 | 1 | A | ok | 25134 | 23848 | - | 0 | 0 |
| 1818057f | 0 | A | ok | 7006 | 6557 | - | 0 | 0 |
| 1818057f | 1 | A | ok | 4272 | 3882 | - | 0 | 0 |
| 1ae2feb7 | 0 | A | x | 29841 | 24942 | - | 0 | 0 |
| 1ae2feb7 | 1 | A | ok | 27017 | 23131 | - | 1 | 1 |
| 221dfab4 | 0 | A | x | 40344 | 40281 | 25563 | 1 | 0 |
| 221dfab4 | 1 | A | x | 26484 | 26424 | - | 1 | 0 |
| 247ef758 | 0 | A | ok | 38557 | 30597 | 29349 | 0 | 0 |
| 247ef758 | 1 | A | x | 29847 | 29790 | - | 2 | 0 |
| 269e22fb | 0 | A | ok | 38947 | 10528 | - | 0 | 0 |
| 269e22fb | 1 | - | x | 30000 | 27904 | - | 4 | 1 |
| 2ba387bc | 0 | A | ok | 4514 | 4155 | - | 0 | 0 |
| 2ba387bc | 1 | A | ok | 6107 | 4705 | - | 0 | 0 |
| 45a5af55 | 0 | A | ok | 6421 | 5115 | - | 0 | 0 |
| 45a5af55 | 1 | A | ok | 9647 | 9378 | - | 0 | 0 |
| 4c3d4a41 | 0 | A | x | 41612 | 36500 | 31123 | 1 | 0 |
| 4c3d4a41 | 1 | - | x | 30000 | - | - | 1 | 0 |
| 62593bfd | 0 | - | x | 45000 | - | 31333 | 0 | 0 |
| 62593bfd | 1 | - | x | 30000 | - | - | 1 | 0 |
| 65b59efc | 0 | - | x | 45000 | 42457 | 34144 | 1 | 1 |
| 65b59efc | 1 | - | x | 30000 | - | - | 1 | 0 |
| 898e7135 | 0 | - | x | 45000 | 44401 | 32101 | 2 | 1 |
| 898e7135 | 1 | - | x | 30000 | 27863 | - | 1 | 1 |
| 9385bd28 | 0 | - | x | 45042 | 44324 | 28773 | 2 | 0 |
| 9385bd28 | 1 | - | x | 30030 | 27080 | - | 3 | 0 |
| 9aaea919 | 0 | A | ok | 44688 | 41611 | 24474 | 2 | 0 |
| 9aaea919 | 1 | A | x | 25942 | 19113 | - | 0 | 0 |
| b5ca7ac4 | 0 | A | ok | 29881 | 18912 | - | 0 | 0 |
| b5ca7ac4 | 1 | A | ok | 26572 | 20074 | - | 1 | 0 |
| bf45cf4b | 0 | A | ok | 9589 | 9104 | - | 0 | 0 |
| bf45cf4b | 1 | A | ok | 15306 | 11557 | - | 0 | 0 |
| cbebaa4b | 0 | - | x | 45000 | - | 28846 | 1 | 1 |
| cbebaa4b | 1 | - | x | 30000 | - | - | 1 | 0 |
| f931b4a8 | 0 | A | ok | 36884 | 36283 | 33322 | 0 | 0 |
| f931b4a8 | 1 | - | x | 30000 | - | - | 1 | 0 |

## Findings

| # | Finding | Evidence | Status |
|---|---|---|---|
| F1 | Tool calls cut at the request cap lose their code. | 6 attempts: 65b59efc b0, 898e7135 b0 and b1, cbebaa4b b0, 269e22fb b1, 1ae2feb7 b1 (logged twice each). rev 3g did not keep the raw arguments, so a replay is not possible. | **Fixed in rev 3h** (salvage + harvest). |
| F2 | The commit turn often makes no tool call. | 18 / 24 made one. In the 6 that did not, the model thinks for the whole 4,096 reserve (reasoning 8-10k chars, `finish_reason: length`). Examples: 136b0064 b0, f931b4a8 b1, cbebaa4b b0. The commit turn is still valuable: 135a2760 b0 and 9aaea919 b0 were solved in it. | **Open, not changed.** A no-thinking commit request (`enable_thinking: false`) changes the system header (the reasoning-effort line), so the prefix cache misses and ~45-55k tokens are prefilled again for every commit turn (~290 in a hidden run). SGLang's thinking-budget logit processor is untested with DFlash. The expected gain is small: most of the 6 had no candidate at all. |
| F3 | The provisional nudge is delivered only at a request boundary (the next tool result or opener). | With per-request cap = remaining − 4,096, a reply ends only at a tool call. The 45k attempts saw the 24,000 nudge at 24.5-36.8k (median ~30k). rev 3h's 30k-attempt nudge (20,700) needs a boundary in [20,700, 24,880) before the 25,904 commit zone. Of the 5 attempt-2s that never wrote a transform (4c3d4a41, 62593bfd, 65b59efc, cbebaa4b, f931b4a8), only 65b59efc has one (22,753). | Limitation of the rev 3h fix, reported. An on-time nudge needs a request split = a pause in thinking. Not added (user rule: ARC-AGI-2 long uninterrupted thinking; pause-resume was negative). |
| F4 | 8 small requests (cap 256-700) after the commit turn. | **Not waste (corrected):** such requests carry the final `submit` after a passing check. 9aaea919 b0 was accepted, **correct**, with 408 tokens left; 247ef758 b1 with 210; 40K35K 247ef758 b0 with 470. Never cut attempts at < 1,024 left. |
| F5 | The branch record's `accepted` and `branches_accepted` (24) do not count a harvest-accepted attempt. | 136b0064 b1 (harvested, correct). | Reporting only; the submission uses it (hit). |
| F6 | Branch `sec` is ~2,895 for every attempt 1. | The real durations are in `rev6_scheduler.durations` and `runs[].sec`. | Reporting only. |
| F7 | `E15B WARNING: mean server running 14.17 < streams - 0.5` | The end tail; `server_running_mean_client_full` = 15.6. | Reporting only. |
| F8 | No KV pressure. | full_token_usage peak 0.71. Worst case 16 × ~55k = 880k > 852k; that needs all 16 near 45k at once (first wave). SGLang would retract, not fail. | Watch in the hidden run. |
| F9 | The 40K35K commit was cancelled at ~2 min. | `e14_state.json`: phase COPY; status CANCEL_ACKNOWLEDGED. Its lastRunTime 23:58:27 equals the submission time 23:58:28. | No 40K35K eval data. The LB score will come from submission 57609813. |

None of F2-F9 is a correctness bug in the scoring path.

## Hidden-run projection (45K30K)

- **Measured:** 2,116 stream-seconds per task.
  - 240 tasks / 16 streams → 31,740 s = **8.8 h**.
  - The solve window in RERUN ≈ 12 h − 3 min − ~14 min startup − margins ≈ 11.5 h → ~2.7 h spare.
- **Worst case** (every attempt at budget, 75k tokens per task, ~440 tok/s): 240 × 75k / 440 = 40,900 s = 11.4 h ≈ the window.
  - Attempt 1 of every task runs first, so only the last attempt 2s are at risk.

## Variant

45K30K is backed by data:
- 2 of the 11 attempt-1 solves came after 40k (commit turn);
- attempt 2 at 30k still added 2 tasks.

40K35K has no commit data; its LB score is pending.

**Next submission (after 57609813 finishes; 1 pending per team): `Q27-XHIGH-E15B_SUBMIT_45K30K_S16_R3H_DUCK_SGLANG.ipynb` (sha `9ee0aa4e`)**
- Contains rev 3h: F1 fixed, plus Notes/10.
- CPU kit v4 on Kaggle: ALL OK (unit 93/93, 7/7 harness modes, Py 3.12.13, pinned image).
- Not pushed; it needs the user's go.

## 40K35K v1 (added later)

| | 45K30K | 40K35K |
|---|---|---|
| queries / tasks / task-norm | 19/32, 13/20, 65.0 % | 11/32, 7/20, 37.5 % |
| attempt 1 correct | 11 | 6 |
| attempt 2 correct | 8 | 7 |
| attempt 2 only | 2 | 1 (1ae2feb7 is b0 and b1; b1-only = 16de56c4) |
| at budget | 16 | 21 |
| commit turns (with tool call) | 24 (18) | 26 (20) |
| cut tool calls | 6 | 5 |
| never wrote a candidate | 8 | 5 |
| solve | 50.0 min | 50.9 min |
| tok/s at 16 running | 451 | 469 |
| selection loss | 0 | 0 |

- **Attempt 1 accepted-correct at or below 35.9k:** 6 in both runs.
  - 45K30K: 16de56c4, 1818057f, 2ba387bc, 45a5af55, b5ca7ac4, bf45cf4b.
  - 40K35K: 1818057f, 1ae2feb7, 2ba387bc, 45a5af55, 9aaea919, bf45cf4b.
- **The whole gap is the 36-45k zone of attempt 1** (45K30K: 135a2760, 9aaea919, 247ef758, 269e22fb, f931b4a8).
  - In 40K35K the same tasks reached 36.8-39.7k with a transform but no accept: 135a2760, 247ef758, 269e22fb, f931b4a8.
- Attempt 2 at 35k did not make up for it (7 vs 8).
- **Conclusion (n = 20, noisy):** 45K30K > 40K35K.
- The pending LB submission is 40K35K. The next submission should be R3H 45K30K.

## Agent loop scan (all 80 transcripts, both runs)

- No stuck loop, and no repeated reasoning text (no line of 40+ chars repeated 3+ times, except printed grid rows).
- 2 python tool timeouts (30 s), recovered.
- **One bounded pattern:** after a rejected submit in the commit zone, the model re-submits the same unchanged `transform`.
  - 40K35K 65b59efc b1: 4 in a row.
  - 45K30K 9385bd28 b1: 3.
  - 40K35K 269e22fb b0: 3.
  - It stays inside the 4,096 reserve. The score impact is 0, and the time is ~1-2 min of wall clock on 240 tasks. Not changed.

## 50K45K build (user 2026-10-25)

`Q27-XHIGH-E15B_SUBMIT_50K45K_S16_R3H_DUCK_SGLANG.ipynb`, sha `91b219bc`.
- Same rev 3h code: Duck package `85a805b5d8bb`.
- Attempt 1 = 50,000, attempt 2 = 45,000.
- Commit window 105 min, budget 128 min.
  - Worst path ≈ a 50k attempt 1 at budget (~2,000 s) + the last attempt 1s + a 45k attempt 2 (~1,800 s) ≈ 97 min.
  - Expected ≈ 60 min.
- KV_PER_REQUEST stays 50,000 (it is a time-weighted mean), so the concurrency stays 16.
- **Hidden projection:** ~10.8 h for both attempts on all 240 tasks (attempt 1 ~5.5 h + attempt 2 ~5.3 h) vs ~11.6 h of solve time.
  - Break-even: a mean of ~72k generated tokens per task.
  - If hidden is harder, the last attempt 2s are cut, never an attempt 1.
- The rebuild left the shas of the 45K30K and 40K35K R3H notebooks unchanged.

## 50K45K R3H commit (esfersami50, sha 91b219bc; added 2026-10-25)

| | 45K30K (rev 3g) | 40K35K (rev 3g) | **50K45K (R3H)** |
|---|---|---|---|
| Queries | 19/32 | 11/32 | **20/32** (pass1 14, pass2 6) |
| Task-norm | 65.0 % | 37.5 % | 62.5 % (12 full + 62593bfd 1 of 2) |
| Attempt 1 / attempt 2 correct | 11 / 8 | 6 / 7 | 10 / 8 |
| Attempt 2 only | 2 | 1 | 2 (135a2760, 1ae2feb7) |
| Solve | 50.0 min | 50.9 min | 61.6 min |
| Generated per task | 57.8k | 59.2k | 69.8k |
| Hidden projection (240 tasks, 16 streams) | 8.8 h | 9.0 h | **10.9 h** (of ~11.6 h) |
| tok/s at 16 running | 451 | 469 | 452 |
| KV peak | 0.71 | — | 0.81 (no retract) |
| Commit turns with a tool call | 18/24 | 20/26 | 18/22 |
| Cut tool calls | 6 (lost) | 5 (lost) | 3 (saved + harvested; 0 accepted) |

**Hits vs 45K30K:**
- **New:** 221dfab4 (attempt 1 at 35-40k), 4c3d4a41 (attempt 1 at 46.7k: needed > 45k), 62593bfd (1 of 2 queries).
- **Lost:** 136b0064, 269e22fb, b5ca7ac4 (in this run attempt 1 accepted a wrong answer).
- n = 20 at T 1.0, so this is noise-level. 50K45K ≈ 45K30K on these 20 tasks; both are far above 40K35K.
