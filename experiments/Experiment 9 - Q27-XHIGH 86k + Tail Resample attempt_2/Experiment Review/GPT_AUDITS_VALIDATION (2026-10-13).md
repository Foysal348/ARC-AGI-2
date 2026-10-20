# Validation of the two GPT pre-GPU audits of E9 (Claude, 2026-10-13)

**User instruction:** the audits do not change the plan; **do not modify the notebook now.** The notebook is unchanged (sha256 `806a2088…`).

| # | Finding (audit A / B) | Real? | Changes the plan? | Notes |
|---|---|---|---|---|
| 1 | 86k × 16 sequences GPU risk; requires a GPU microprobe (A-F1, B-F4 "blocker") | real **risk**, not a bug | **No** | E9 adds no memory-hungry operation (no `prompt_logprobs`). Long sequences only raise KV pressure, which vLLM handles by preemption (recompute). Measured at 49k: KV max 73.6 %, 0 preemptions. The PROXY56 commit run *is* the measurement (KV peak, `num_preemptions`, tok/s). A separate microprobe would cost GPU time for the same answer |
| 2 | `admit_resample` sets `STATE` / `RESAMPLED` before `engine.add_request` (A-F2) | true | No | The same order as v1.4.4's own `admit()`. If `add_request` raises, the outer scheduler handler ends the run anyway (keeping finished units), so a phantom entry has no effect. A one-line reorder is optional, later |
| 3 | `phase1_cap` can go below 49,152 for prompts over 48,896 tokens (A-F3, B-F2) | true code path | No | Inherited from v1.4.4, where the threshold was even lower (24,320). Real prompts: PROXY56 max 11,962, hidden placeholder max 12,788 (worst-case ARC prompt about 38k). The run logs `reduced thinking budget: N`. E9 is stricter than the base, not looser |
| 4 | A resample can overwrite a *correct* attempt_2 (A-F4, B-F3) | true by spec | No | The old attempt_2 (input or mirror) is correct for 0/172 eval and 1/1,076 training inputs, so the risk is negligible. Both audits keep the policy. A useful report addition later: `resample_lost_hits` beside the gain |
| 5 | `eval_task_normalized_pct` still uses primary-only rows (A-F5A, B-F1) | **true, a reporting bug** | No | A diagnostic field only. `eval_query_hits` / `eval_score_pct` (the Kaggle metric) are correct (final pairs). Fix later in the report cell only |
| 6 | The hidden-time projection mixes primary-only tokens with mixed throughput; `projection_valid` stays true with resamples (A-F5B, B-F6) | true, reporting | No | A diagnostic only. By the user's rule, hidden time is not projected from eval anyway. Later: set `projection_valid = False` when resamples ran |
| 7 | "259 × 66.5k / 353 tok/s = 13.55 h, so the hidden run will not even finish the primary" (B-F5) | **rejected** | No | It applies the **evaluation** mean (66.5k per unit) to the hidden set, against the user's premise (hidden about 2× easier; memory `hidden-test-easier-than-eval`). Whether the hidden run leaves tail time is unknown and is not derivable from eval |
| 8 | Stale `now` in the tail-admission loop: admission after the deadline (B-F7) | technically true | No | The primary admission loop of v1.4.4 has the same pattern. `add_request` takes milliseconds; the audit's demo used an artificial 0.8 s per registration. The drain follows 2 min (hidden) or 36 min (PROXY56) later anyway. Optional `time.perf_counter()` check later |
| 9 | A resample aborted at hard stop may promote a partial grid to attempt_2 (B §6) | true | No | Inherited `finish()` behaviour; the replaced fallback is worth about 0. Records keep `end_reason`, so it is visible |
| 10 | Harness makes every resample correct, so the +16 is mechanics only, and no negative case is tested (B §7) | true | No | Stated in our README as harness behaviour. A negative test needs item 4's `lost_hits` field first |
| — | Invariants (attempt_1 immutable, no resample before the queue is empty or after the admission deadline or drain, one per unit, new seed, original order, hidden deadlines, unchanged primary prompt/sampling/parser, hidden schema) | **both audits: PASS** | — | Agrees with Claude's checks |

## Conclusion

- **No blocker.**
- The confirmed items are reporting issues (5, 6, and the optional `lost_hits` field) plus two harmless inherited patterns (2, 8).
- One claim is rejected because it projects eval token use onto the hidden set (7).
- The GPU questions (1) are answered by the planned PROXY56 run itself.
- If the user later wants the small fixes, they touch only the report cell and two lines in the scheduler, with no behaviour change for primary units.
