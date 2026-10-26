# EMS / VMS rev 2e: disposition of the two independent audits of rev 2c (2026-10-26)

Inputs (this folder):
- `README_EMS_VMS_REV2C_INDEPENDENT_AUDIT_2026-10-26.md` (audit 1: HIGH 1-10 + a 10-point acceptance checklist).
- `ENGINE_MATRIX_REV2C_INDEPENDENT_AUDIT_README_BN.md` / `.pdf` (audit 2: E1-E12).

Builds after all changes (rev 2f): EMS SAFE **c0ade787** (199 kB), EMS RISKY **b39016d6** (199 kB), VMS **b01992c5**
(122 kB). All are under the 400 kB commit-size limit. rev 2f split EMS into SAFE and RISKY notebooks (user 2026-10-26: a crash
in one must not damage the others; VMS is the third part) and put every arm call in its own try/except
(`_scratch/ems/patch_ems_split_rev2f.py`, `_scratch/vms/patch_vms_rev2f.py`).

Rule followed (user, 2026-10-26): research before changing, and remove nothing valid without evidence. Every claim was
checked in the builder source, the SGLang 0.5.21 / vLLM 0.32.0 sources, or our own real Kaggle logs before any change.
Patch scripts (assert-guarded, every replacement must match exactly once) are in `_scratch/ems/` and `_scratch/vms/`:
`patch_ems_rev2e*.py`, `patch_vms_rev2e*.py`, `patch_grid_aware_rev2e_c.py`, `patch_rev2e_d_labels.py`,
`patch_rev2e_d_prefix_inline.py` (`patch_rev2e_d_prefix.py` was the first try, on the shared module, reverted).

Status words used below:
- **FIXED**: changed in code, with a harness check.
- **PARTLY**: part of the request is done; the rest is listed, with the reason.
- **NOT CHANGED**: deliberately not changed, with the evidence or reason.
- **OPEN**: needs a real GPU run; it cannot be closed on CPU.

## Audit 1 (README_EMS_VMS_REV2C_INDEPENDENT_AUDIT)

| ID | Claim | Verified? | Status | What the code does now |
|---|---|---|---|---|
| HIGH 1 | Without the target-only oracle, EMS falls back to the DFlash reference; VMS keeps the MTP reference | Yes: EMS `oracle='dflash_ref'` path; VMS `gate_reference` | FIXED | EMS `_gated` returns False unless `oracle == 'target_only'`, `EMS_TGT` exists and the target A/A probe is complete. VMS `adoptable` needs `VMS['oracle_ready'] is True`. Speed diagnostics still run; nothing becomes adoptable. |
| HIGH 2 | VMS incomplete SSE counts as a valid identical row | Yes: VGen did not keep `finish_reason` / `[DONE]` | FIXED | VGen records `done_seen` and `finish_reason`. A greedy row lacking both is `INCOMPLETE_SSE` (error). `vms_diverge` counts it as an error, never as identical. vLLM 0.32.0 sends `[DONE]` (serving.py:503). |
| HIGH 3 | Partial stream failures do not invalidate throughput arms | Yes, in both builders | FIXED | EMS records `window_errors_*` and `running_share_*`. `_gated` rejects any window error and a running share < 0.80. VMS `measurement_ok` = clean windows only (`vms_window_healthy`). 0.80 is our proposed quality bar, not an upstream invariant (stated in the report line). |
| HIGH 4 | VMS degenerate check drops empty outputs (`if g.text`) | Yes | FIXED | All started streams are checked. `empty_outputs`, `degenerate` and `no_first_token` are counted separately (E5 nuance: an unstarted/aborted stream is not counted as model corruption). |
| HIGH 5 | VMS decode-only tok/s is shown as if it included the 40k prefix recompute | Yes | FIXED (VMS); NOT CHANGED (EMS, labelled) | VMS records `tok_s` (decode after the prefill settled) plus `end_to_end_tok_s` (output tokens from submission to window end / wall), `settle_sec`, `prefill_settled` and `running_mean`. rev 2e-d corrected a stale report note that still said the no-prefix arm's tok/s "includes the recompute". EMS tok/s is the SGLang decode-line rate; the report says "decode only, not end-to-end". EMS e2e is not measured (SGLang decode lines carry no per-request submission time). The production comparison is decode at equal occupancy. |
| MED/HIGH 6 | PR #45151 patch: atomic write, exact marker, memory proof | Yes | FIXED (code); OPEN (memory proof) | The patch checks both snippets (not just the marker), compiles, writes a temp file, `os.replace`s it, keeps a `.ems_orig` backup and records the source sha256. A partial marker or a missing anchor means not applied. The `_pr45151` arm is skipped unless the patch applied. The real KV-token / free-HBM difference can only come from the GPU run: the arm records the KV tokens next to `mtp_3_1_4`. The patch is called "upstream PR #45151 (open)", never a merged fix. |
| MED 7 | 40k is not a > 64k long-context test | Yes | PARTLY | Added an XL band: ~85k-token prompts (`E14_XL_TOKENS` = 85000), streams = KV // 100k, a greedy 256-token XL parity against the oracle's XL A/A (`xl_gate`), and adoptable needs `xl_gate` PASS. The XL speed windows decode up to 8k tokens on the 85k block, so positions reach about 93k. **Not done:** a >= 100k position. With the 8k window output, VMS max-model-len 106,496 does not fit it (100k + 8k = 108k). A separate 100k prompt with 256-token parity would fit, but it was not added: it would mean one more ~100k prefill on every arm. `E14_XL_TOKENS` can raise the block on EMS (context 155,648). **Not done:** 60k+ *generated* tokens with churn. That is a release requirement for the production run (see "Open before production"), not something the 75-s matrix can cover. |
| MED 8 | Timing: arms will not fit 200 min; reserve does not guarantee both TP2 | Yes | FIXED (order + labels) | Budget 260 min (user-approved increase). EMS order: oracle, then reference, PCIe, MTP 3/1/4, the #45151 A/B, MTP+PCIe, NTN INT4, DFlash 20, DFlash 24, and NTN FP8 (moved after 20/24: E12). The rest follow. The TP2 reserve is 30 min. Every arm that does not fit is recorded `NOT_TESTED_BUDGET: <reason>` and never reported as tested. VMS has the same labels. |
| MED 9 | Cross-request contamination (#36548) is not caught by a punctuation heuristic | Yes (the heuristic only sees degeneration) | FIXED (canary) | Canary: `level` concurrent greedy requests, each repeats its own random sentinel. A sentinel found in another request's output is a leak. Any leak (or canary error) makes the arm non-adoptable. The oracle gets the canary too. Own-sentinel hits are recorded, not gated: a model that paraphrases is not a leak. The long-generation (> 8k) canary is not done (same reason as MED 7). |
| HIGH 10 | A margin PASS is not bit-exact greedy identity | Yes (reproduced by audit 2) | FIXED, with a documented rule | `gate_class`: INCOMPLETE / FAIL / BOUNDARY / PASS_EXACT, plus a post-divergence tail check (E2). The audit allows "a distinct rule if the team intentionally accepts BOUNDARY". **We accept BOUNDARY as adoptable under the near-tie tolerance.** Reason: a different batch size or kernel path changes bf16 rounding, so requiring exact identity would reject every arm with no evidence of a wrong answer. rev 2e-d adds `certified_exact` (adoptable and PASS_EXACT) as its own column. The report text now says this, instead of the earlier wrong "never auto-certified" (fixed in rev 2e-d: the text and the code disagreed). |
| Checklist 1-10 | Production acceptance list | - | Mapped | 1 = HIGH 1; 2 = HIGH 2; 3 = HIGH 3/4; 4 = HIGH 5 (VMS both rates, EMS decode only); 5 = HIGH 10; 6: the KV pool, mamba slots, `max_concurrency_at_max_len` and preemptions are recorded, and the slots-aware stream count means no silent concurrency cut; 7 = OPEN (production long-run); 8 = harness plus the new negative tests (below); 9: flags, `pip freeze`, patch sha, cache dirs, prompts and output ids are saved in the bundle; 10: missing evidence = INCOMPLETE / None, never PASS. |

## Audit 2 (ENGINE_MATRIX_REV2C_INDEPENDENT_AUDIT, E1-E12)

| ID | Claim | Verified? | Status | What the code does now |
|---|---|---|---|---|
| E1 | VMS compares decoded strings, not integer token ids | Yes (`logprobs.tokens` are strings) | FIXED | `return_token_ids: True`. `choices[].token_ids` must be a list of ints (`MISSING_STREAM_TOKEN_IDS` / `INVALID_STREAM_TOKEN_IDS` otherwise). The comparison is on ids. Harness test: same text with different ids is a divergence. |
| E2 | A first-divergence margin can accept a corrupted continuation | Yes (the audit reproduced PASS) | FIXED | After the first divergence, the tail is decoded and judged by the shared degenerate rule. If the candidate has more degenerate tails than the oracle's own A/A, it FAILS. Near-tie-only divergences are BOUNDARY (see HIGH 10). Harness: near-tie followed by "!" x 120 gives FAIL. |
| E3 | Fail closed without the target-only oracle | Yes | FIXED | = HIGH 1 (EMS `_gated`, VMS `oracle_ready`). |
| E4 | The #45151 arm runs even if the patch failed | Yes | FIXED | = MED/HIGH 6: skipped with `PR #45151 patch not applied: <why>` unless `applied`. The harness (no SGLang installed) still exercises the arm logic. |
| E5 | Empty VMS outputs are excluded from the degenerate count | Yes (3 empty gave 0) | FIXED | = HIGH 4, with the nuance kept: `no_first_token` (never started), `empty_outputs` (started, empty), `degenerate` and `stream_errors` are separate fields. |
| E6 | VMS does not join streams or check the server is idle between windows | Yes | FIXED | `finally`: stop, join (5 s each), `threads_not_joined`, `idle_after` (0 running / 0 waiting, 60 s). Otherwise `DIRTY_SERVER_AFTER_WINDOW`, and the next band is skipped only in that case (rev 2e-b). |
| E7 | The 120-s settle loop measures even if the prefill never settled | Yes | FIXED, then corrected | rev 2e made a timeout `PREFILL_NOT_SETTLED` and dropped the window. **Self-audit (rev 2e-b): that threw valid data away.** The no-prefix arm prefills 16 x 40k = 640k tokens, which can take more than 120 s at ~1-3k tok/s prefill. Now the settle budget is 300 s. A window that still did not settle is measured and labelled `mixed_load=True`. It is excluded from `measurement_ok` (needs `prefill_settled is True`) but kept in the record. |
| E8 | VMS `ok` means only "two speeds exist" | Yes | FIXED | `measurement_ok`, `capacity_ok`, `correctness_ok` and `adoptable` are separate (`vms_classify`, re-run for every arm at the end, once the oracle exists). `ok` = measurement_ok. EMS has an explicit `adoptable` column (+ `certified_exact`, rev 2e-d). DP/TP2 rows are diagnostic (E10). |
| E9 | Prefix caching may silently be a no-op (vLLM hybrid align mode) | Yes for vLLM (#45238, #51250) | FIXED (VMS + EMS) | VMS records `prefix_cache_hits`, `prefix_cache_queries` and `prompt_tokens_cached` deltas per window (vLLM 0.32.0 metric names verified in source). Harness: 0 hits on the no-prefix arm, > 0 on the reference. **EMS (new in rev 2e-d):** SGLang 0.5.21 prints `Prefill batch, #new-token: X, #cached-token: Y` (verified in our E14A log). The EMS cell takes the server-log offset before each band, sums the prefill lines after it and records `prefix_hit_rate`, shown as a report column. The count includes the band's warm request, so it slightly understates the streams' own rate; a silent no-op still shows as ~0. This is done in the notebook, not in `e14_measure`: those modules come from the Kaggle payload dataset shared with E15P, so a change there would not reach Kaggle and would touch other notebooks. Historical evidence from our real SGLang logs: E15B 91-93 %, G4B 76-82 %, E14A 59-78 % radix hits. So the 0 % concern is not seen on our SGLang stack, but it is measured per arm now. |
| E10 | TP2 reports KV for 40k at reduced streams; that does not prove production fit | Yes | FIXED (label) | TP2 rows are diagnostic, with `kv_ok_for_production` = every replica's KV >= per-replica streams x `SGL_KV_PER_REQUEST`. They are never adoptable (family is not mtp/dflash). |
| E11 | No 65k-155k or 60k+ generated-token corruption test | Yes | PARTLY | = MED 7: an XL ~85k-context parity plus canary per arm. 60k+ generation with churn is OPEN for production. |
| E12 | The FP8 NTN cold launch comes before DFlash 20/24 | Yes | FIXED | `ntn_fp8` moved after `dflash_w6_s24`. Harness asserts the order. |
| "Production realism" | 75-s windows with `ignore_eos` measure capacity, not task success | Agreed | NOT CHANGED (by design) | The matrix is an engine-speed / correctness-isolation tool. A winner must still pass a real ARC (Poetiq/Duck) run at temperature 1, top_p 0.95, top_k 20 before production. That is listed as open. |

## Self-audit corrections (no audit asked for these; found while checking rev 2e)

1. **rev 2e-b (VMS), settle timeout:** see E7. Dropping unsettled windows lost real data; they are kept as `mixed_load` now. The next band is skipped only after a dirty/dead server, not after any window error.
2. **rev 2e-c (both), grid false positives:** the rev 2e tail rule (16 identical ids, or > 50 % repeated 8-grams) and the degenerate rule (< 8 distinct characters over 200) both flag a normal ARC grid ("0 0 0 0 ...") as degenerate. That would have failed correct arms. New shared rule: text whose non-space characters are >= 30 % digits is grid-like and never degenerate. Otherwise these are degenerate: empty, a 32-character run of one character, < 8 distinct characters over 200, or < 10 % letters/digits (the #53912 shapes). The tail check decodes the ids and applies the same rule. Tests: an ARC grid, a JSON grid and normal prose are not flagged; "!!!", ", , ," and empty are.
3. **rev 2e-d (both), labels:** the BOUNDARY text now matches the code, and `certified_exact` was added (HIGH 10). The stale VMS no-prefix note was fixed (HIGH 5). The EMS prefix-cache hit rate is measured (E9).

4. **rev 2e-e (found by the local Python 3.12 harness run of rev 2e-d; script `_scratch/vms/patch_rev2e_e.py`):**
   - **VMS, real bug:** `vms_window` waited up to 900 s for `level` running requests even after the streams had died or finished. In the harness, 4 of 16 streams died at connect and 12 finished; the loop then sat 15 min at running 0. It now stops as soon as fewer than `level` streams are alive, records `streams_ended_before_level`, and ends with the same error as before. On GPU this saves up to 15 min of an arm's cap.
   - **EMS, real bug:** the reference arm's record never got `probe_complete_vs_target`. So with the target-only oracle, `_gated` could never mark the production reference (DFlash W6, 16 streams) adoptable, even when it passed.
   - **Harness only:** both fake servers get a listen backlog of 256. Python's default of 5 dropped some of 16-24 simultaneous connects on Windows; that caused the timeouts and resets in the logs. The negative tests now use long strings: Qwen BPE merges a run of "!" into a few ids, and fewer than 16 ids are not judged.

## Harness negative tests added (Python 3.12, same as the Kaggle image)

- VMS: an incomplete SSE row is an error; different ids are a divergence; near-tie followed by a degenerate tail ("!" x 120, tokenizer-encoded) FAILS; an ARC grid tail is not flagged; identical rows are PASS_EXACT; an empty plus "!" x 100 give 2 degenerate, while prose and a grid give 0.
- VMS: the oracle is ready; FP8 MTP arms are PASS_EXACT, XL PASS, 0 leaks and adoptable; INT4 is never adoptable; every window is clean, with e2e tok/s and prefix-cache counters recorded; certified_exact = adoptable and PASS_EXACT.
- EMS: the oracle canary plus XL A/A; INT4 arms are PASS_EXACT, XL PASS, 0 leaks and adoptable; ntn_fp8 runs after the 20/24 arms; certified_exact is consistent; prefix hit rates are recorded.
- EMS: the tail rule flags "!" x 120 and ", " x 120, and passes an ARC grid and normal prose.

## Open before production (cannot be closed on CPU)

1. A real GPU run of EMS and VMS (user go needed; ~260 min each, ~9 h quota at 2x; Hamid's account has quota). EMS PCIe arms and all of VMS need access to the dataset `foysalemonshanto/vllm-0-32-0-cu13-wheelhouse`.
2. The real PR #45151 memory gain (KV tokens vs `mtp_3_1_4`) is unknown until that run.
3. Long generation (60k+ output tokens, with churn and rejected drafts) on the winner, plus a real ARC task run at production sampling before any production switch.
4. BOUNDARY winners: confirm on ARC tasks (task-level accuracy), since they are not bit-exact.
