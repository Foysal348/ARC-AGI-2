# E25 NB5 QUALITY (foysalemonshanto, sha 9c71ba89, 163 min) — FP8 vs BF16 KV, Flash-Next W4 + Duck config B, 2026-11-01

- **Output:** `_scratch/e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x/` (`nb5_report.json`, two `phase_*` reports and bundles). Errors: [].
- **Setup:** 15 eval tasks (7 solid, 5 sometimes, 3 never), 2 attempts of 104k each, the same seed, task order and Duck config B (variant A). The solve windows were equal, 4,050 s each. Both boots used pin +1.0 GiB and #60490.

## Scores
| | FP8 KV (pool 214, 9 workers) | BF16 KV (pool 112, 5 workers) |
|---|---|---|
| tasks solved | **13 / 15** | **13 / 15** |
| test queries | 18 / 20 (pass1 17, pass2 1) | 18 / 20 (pass1 18) |
| task-norm | 86.67 % | 90.00 % |
| tool-call rate | 0.996 | 1.000 |
| generated tokens | 670,902 counted, 696,911 server | 599,876 counted, 640,099 server |
| window tok/s | 172.0 | 158.0 |
| steady p50 (all lines) | 225.0 | 169.2 |
| share of lines at full workers | 0.347 | 0.822 |
| duty (time) | 0.999 | 0.9996 |

**Paired (McNemar):**
- Both configs solved 12 tasks.
- **FP8-only:** 247ef758 (sometimes; BF16 got 1 of its 2 queries).
- **BF16-only:** 16b78196 (never).
- **Neither:** 2b83f449 (never).
- Discordant 2, p = 1.0, reading NOT_SIGNIFICANT_NOT_PARITY_UNDERPOWERED.
- Over the tasks finished in both configs (12): FP8 12, BF16 11.

**By group:**
- solid: 7 / 7 in both.
- sometimes: FP8 5 / 5, BF16 4 / 5.
- never: FP8 1 / 3 (64efde09), BF16 2 / 3 (64efde09, 16b78196).

**Reading:** on 15 real ARC tasks, FP8 QSA KV shows no measurable accuracy loss against BF16, despite NB0's teacher-forced NO_GO (FP8 label UNVALIDATED_FP8_UNDER_TEST). The sample is far too small to prove parity, but it is the first end-to-end ARC evidence. FP8 gives 1.9× the pool and 9 vs 5 workers.

## Tokens per task (generated, attempt 1 / attempt 2)
| group | FP8 attempt-1 median | FP8 per-task mean | BF16 attempt-1 median | BF16 per-task mean |
|---|---|---|---|---|
| solid | 6,344 | 14,327 | 7,200 | 14,638 |
| sometimes | 32,679 | 50,089 | 21,358 | 52,656 |
| never | 90,477 (max 103,761) | 106,724 | 69,218 | 78,042 |

- In the never group, attempt 2 had little time left and ended at 3-31k tokens.
- **Attempt 2 added 1 query in FP8 and 0 in BF16.** Skipping attempt 2 after an accepted attempt 1 would save ~25-35 % of the tokens with no loss on this set.
- **Accepted ≠ correct, twice:** 16b78196 attempt 1 was accepted but wrong in FP8, and 2b83f449 attempt 1 was accepted but wrong in BF16.

## Coverage projection (12 h, hidden 240 tasks, FP8 9 workers ≈ 215 tok/s × ~41,000 s ≈ 8.8M tokens)
| hidden mix (solid / sometimes / never-like) | tokens per task | coverage |
|---|---|---|
| this sample, 7 / 5 / 3 | ~45k | ~80 % |
| 30 / 30 / 40 % | ~62k | ~59 % |
| 20 / 30 / 50 %, never-like tasks at full 2 × 100k | ~100k | ~37 % |

The coverage binds. Higher tok/s plus skipping attempt 2 after an accepted attempt 1 are the direct levers.
