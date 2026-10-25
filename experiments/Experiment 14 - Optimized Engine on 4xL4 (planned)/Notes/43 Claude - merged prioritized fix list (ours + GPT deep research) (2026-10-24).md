# 43 — Merged, prioritized fix list: our sources + GPT's EVAL120 deep research

**Date:** 2026-10-24.

**Sources merged (nothing dropped):**
- Our 63-item list:
  - the EVAL120 analyses A / B / C (`Notes/EVAL120 analysis 2026-10-24/`);
  - the GPT E15 audits (Notes/38-40);
  - the notebook code audit (4-day review, agent C);
  - the 4-day review (Notes/37);
  - the "kernel issue" research.
- Notes/42: the fixes from the EVAL120 runs plus the config-B design.
- Notes/44: GPT's EVAL120 forensic deep research, including the LOW+E16 55-task follow-up (verbatim).
- Notes/45: GPT's second EVAL120 deep-research report (verbatim). Its verification and merged items are in section 7 at the end; it changes the D1 recommendation.

**Tags:**
- `[ours]` — from our sources;
- `[GPT]` — from Notes/44;
- `[GPT2]` — from Notes/45;
- `[both]` — both sources;
- `[new-verified]` — found while verifying GPT's claims on 2026-10-24.

**Old #n** gives the item's number in the 63-item chat list.

---

## 0. Verification of GPT's claims against our data (2026-10-24)

| GPT claim | Check | Result |
|---|---|---|
| No test-correct host-verified candidate is lost in the final attempts | The count of test-correct train-exact queries in `checked[]` against the score | **Confirmed.** 46 = 46, 41 = 41, 31 = 31. |
| `_R6_AUTOCHECK_RE` does not match `trans` or `candidate` | `arc2_duck.py:1177`, `src/e16/arc2_duck_e16.py:1177` | **Confirmed.** The pattern is `^(transform\|solve\|solution\|predict)\w*$\|^\w+_(alt\w*\|v\d+)$`. |
| e8686506: `sys` import blocked, then `check()` timed out after 30 s | LOW+E16 transcript | **Confirmed.** Also `Module 'time' is not allowed`. |
| Stage-2 b0 refinement is productive | Tokens at the first test-correct program (table below) | **Confirmed, and stronger than GPT states.** |
| E16 paired: 20 vs 15 discordant, p ≈ 0.50 | Our B analysis gave p ≈ 0.44-0.50 | Agrees. |
| SGLang #36548, #38009, #35723 and Qwen #216 | Not opened by me | **GPT-cited, not verified.** Treat them as leads. |

**New finding while verifying (not in either list):**
- Blocked sandbox imports, as `Module 'X' is not allowed` in transcripts:
  - XH+E16: 38 in 26 branches. `sys` ×21 (mostly `sys.setrecursionlimit` for flood-fill / DFS); `traceback` ×3; `time`; `inspect`; `shutil`; `numpy`.
  - The model also tries to *import the prelude helpers as modules* (`inspect_pair`, `arc_overview`, `comp_sets`).
  - XH: `sys` ×21, `time`, `numpy`. LOW+E16: `sys` ×13, `inspect` ×2, `time` ×2, `base64`, `numpy`, `cmath`.
- `Tool timed out after 30s`: only 2-3 per run.

### Tokens per branch at which each test-correct program first appeared

The script is `_scratch/eval120_analysis/scripts/tok_at_correct.py`. Its input is `checked[].tokens` (the branch's cumulative generated tokens) for train-exact checks whose test output equals gold.

| Branch tokens | XH+E16 queries | XH queries | LOW+E16 queries |
|---|---|---|---|
| 0-20k | 9 | 11 | 11 |
| 20-28k | 5 | 7 | 7 |
| 28-35k | 11 | 6 | 7 |
| **35-40k** | **8** | **3** | **0** |
| **40-50k** | **7** | **10** | **5** |
| **50-70k** | **6** | **4** | **1** |
| > 70k | 0 | 0 | 0 |
| **after 35k** | **21 / 46 (16 tasks)** | **17 / 41 (13 tasks)** | **6 / 31 (6 tasks)** |

- All the correct programs came from b0; b1 gave 0 in every band.
- Accepted branches: median 30.7k / 32.0k / 26.4k; 18 / 19 / 10 of them were accepted after 35k.
- **Caveat:** these counts are under the rev 6 structure (12k / 24k request caps, the 28k pause, resumes, and the context trim at 51,396). An uninterrupted attempt may reach a program in fewer tokens. That is unmeasured.

---

## P0-DECISION — needed before building config B

| # | Decision | Evidence | Recommendation |
|---|---|---|---|
| D1 | **Attempt budget.** Option B (approved): 2 × 35k independent attempts. Option B′: attempt 1 runs continuously up to ~70k on every task first; attempt 2 uses the time left, unverified tasks first. | `[new-verified]` 37-45 % of today's XHIGH solves (21/46, 17/41) appeared at 35-70k branch tokens. A 35k stop would cut them unless an independent 35k attempt 2 re-finds them, and so far no second branch has found a new correct program (b1: 0). `[GPT]` Stage-2 b0 refinement is productive. | **Revised after Notes/45 (section 7): B″** — attempt 1 runs continuously up to **60k** on every task first, the same as the 60k floor and S1-60K.<br>Why 60k and not 70k:<br>• correct programs above 60k: 0 / 0 / 1 (the one is LOW c4d067a0 at 60,744);<br>• expected spend 240 × (0.35 × 31k + 0.65 × 60k) ≈ 12.0 M tokens ≈ 35 ks at 341 tok/s (≈ 31 ks at 384) against the ~42 ks window;<br>• this leaves about 2-3 M tokens for attempt 2 on unverified tasks.<br>B′ (70k) needs ≈ 40 ks at 341 tok/s and fails at GPT's 280 tok/s.<br>**The user's call:** B (2 × 35k) was approved, and the top competitor's 35k was for their harness. |
| D2 | Attempt 2 on tasks where attempt 1 was accepted, run last. | `[ours]` Wrong accepts: 5 / 8 / 9. `[GPT P4]` 10 / 14 / 16 train-exact-but-test-wrong tasks per run; a second independent hypothesis is GPT's P4. | Yes, after the unverified tasks. Old #14. |
| D3 | E16 off. | `[both]` Mechanically delivered; gain inconclusive (paired p ≈ 0.5); raw content had 0 effect on xhigh. GPT: "experimental only; no more nudge-threshold tweaks." | Off by default; keep the flag. Old #15. |
| D4 | Isolation INCONCLUSIVE → fall back to 6 workers. | `[ours]` code audit M1. This is a safeguard the user approved; changing it needs approval. | Keep unless the user says otherwise. Old #27. |
| D5 | fp32 SSM arm. | `[ours]` kernel research. `[GPT]` cites #36548 (FP32 SSM reduced, but did not remove, concurrent errors). | Run as an E17 arm. Old #48. |
| D6 | **The Poetiq 34.58 baseline stays the fallback submission** until a Duck config beats it under the same conditions. | `[GPT P0, mandatory]` Hidden LB: S1-60K 34.58 (31.94 resubmit) vs Duck rev 6g 22.78. | Freeze the S1-60K notebook sha, model paths and 240-task schema. **New item.** |
| D7 | Validation design: GPT's stratified 24-task paired experiment (arms A-D) or our full EVAL120 G4B. | `[GPT]` Cheaper, and the arms can be read separately. `[ours]` With 24 tasks a 2-3 task difference is noise (eval sd ≈ 1.5 points at 120 tasks). The user's rule is to bundle changes and make big jumps. | Our choice: EVAL120 G4B for the bundled config B / B′. The 24-task paired set for the solver-design arms (P2). |

---

## P0 — must go into the next build (config B / B′)

| # | Fix | Source | Notes |
|---|---|---|---|
| 1 | Remove branch 1 (the indexed view); its time goes to attempt 2. | `[both]` 244 b1 runs (1.46-1.85 M tokens a run) → 0 correct programs. GPT P1. | GPT: keep b1 as a flag or arm, not deleted forever → **kept behind `branches=`**. Old #1. |
| 2 | No 28k stage-1 pause and no stage-2 chunks; each attempt runs continuously. | `[ours]` 58-66 % of prefill was resumed branches re-prefilled; throughput −18-30 % after hour 2; 156-170 branches paused and never resumed. `[GPT]` "no-pause arm unresolved", but stage-2 b0 is productive. | Old #2. |
| 3 | One long request per think phase; drop the 12,288 / 24,576 / 32,768 caps. | `[both]` 98 / 84 tool-less length cuts at 24,576 in XHIGH. GPT: "investigate request-cap / behaviour interaction". | The caps came from ARC-AGI-3. Old #3. |
| 4 | **No context trim inside an attempt:** raise the server context length so the prompt + attempt budget + tool outputs fit (the trim fires at 51,396 today). | `[both]` Trimmed branches were correct 16 % vs 34 % of the time (XH+E16) and 5 % vs 30 % (LOW). GPT: 37 of the 55 LOW no-progress tasks were trimmed. | A 70k attempt (D1-B′) needs a context of ~95-100k. KV p90 was 0.40 at 12 streams, so check the KV fit in the capacity check. Old #8, extended. |
| 5 | **Bug:** Duck turn rollback (`tool_agent.py` L7094-7101, L7183-7186). Keep the turn; drop only the unexecuted call. | `[ours]` 56 / 47 events; 0.52 / 0.46 M context tokens lost. | Old #4. |
| 6 | **Bug:** `submit` called as a tool fails. Route it to the python helper. | `[ours]` 22 / 28 / 23 times per run. | Old #5. |
| 7 | attempt_1 / attempt_2 assembly:<br>• a1 = verified program 1;<br>• a2 = a distinct verified program;<br>• else the best partial program run on test;<br>• the test input last. | `[ours]` pass2 = 0; 58 of 60 a2 slots were the test input. `[GPT]` Select two *distinct* valid outputs; merge independent solver candidates into one schema; keep `select_attempts_r5()`'s shape- and source-aware logic. | Old #6 + GPT §3.3. |
| 8 | At the end of an attempt without an accept: harvest — run `check()` on an existing one-grid transform. **Plus** a short candidate-construction turn after a length cut (not a repeat of `R6_TEXT["length"]`). | `[ours]` Harvested programs gave 12 / 4 / 5 of the hits. `[GPT]` Cause D: db0c5428 had `def transform` only inside the reasoning text, never executed. | Old #7 + GPT. |
| 9 | Autocheck regex: add `candidate` and `trans` names, as `^(transform\|solve\|solution\|predict\|candidate\|trans)\w*$\|^\w+_(alt\w*\|v\d+)$`. | `[GPT, verified]` Cause B (cbebaa4b: helpers, but no `transform` name). | Keep the `_r6_single_grid_arg()` guard and the 30 s budget.<br>Tests:<br>• `trans(grid)` autochecks;<br>• `trans(x, y)` does not;<br>• non-grid output is rejected;<br>• wrong-train is not retained. **New.** |
| 10 | **Sandbox imports:**<br>• a safe `sys` shim (only `setrecursionlimit`, `maxsize`);<br>• allow `time`;<br>• a clear error for helper names ("`inspect_pair` is already defined; call it, do not import it");<br>• keep `numpy` blocked, but say so in the error. | `[new-verified]` 21 / 21 / 13 `sys` blocks per run; the model imports helpers as modules. | Do not expose `sys.modules` (escape risk). **New.** |
| 11 | Bounded search: guidance plus a node counter for DFS / packing; run the candidate on one train input before `check()` (shape + runtime). | `[GPT, verified]` Cause C, e8686506 (`check()` timed out twice). Only 2-3 timeouts a run, so the impact is small. | **New.** |
| 12 | Effort xhigh only. | `[both]` LOW −10 to −15 queries; GPT P0 "XHIGH primary". | Old #13. |
| 13 | **Server watchdog:** restart SGLang if it dies mid-run; more connection retries. | `[ours]` code audit H1. | Old #9. |
| 14 | **Launch fallback:** retry with the W1 flags if SGLang fails to launch (today that means LB ≈ 0). | `[ours]` code audit H2. | Old #10. |
| 15 | Sandbox self-test and tokenizer probe on a fixed task in the notebook, not the first hidden task. | `[ours]` code audit M4. | Old #11. |
| 16 | Wrap the scheduler worker loop in try/except so a worker cannot die silently. | `[ours]` code audit M5. | Old #12. |

---

## P1 — next build, small items

| # | Fix | Source | Notes |
|---|---|---|---|
| 17 | Do not select with `overfit_flags`. | `[ours]` Precision 42 / 30 / 63 %. | Old #16. |
| 18 | #43567 backport (stale lazy mamba checkpoint). The patch is built and 171/171 tests pass. | `[ours]` web research. | Old #17. |
| 19 | CPU rule-library fallback instead of identity. | `[ours]` 4-day review D. `[GPT]` NVARC / Poetiq / deterministic candidates for the second slot. | Old #18 + GPT §5.4. |
| 20 | Count the tokens cut at the deadline in the report (server-side count). | `[both]` About 1 % missing. | Old #19. |
| 21 | Fix the retract regex (it matches `retraction_policy`). | `[ours]` | Old #20. |
| 22 | Report accepted and harvested separately. | `[ours]` | Old #21. |
| 23 | **Diagnostic fields per branch:**<br>• `candidate_source_written`, `candidate_retained`;<br>• `check_attempted`, `check_executed`, `check_timeout`;<br>• `train_pairs_exact`, `host_verified`;<br>• AST classification of written-but-never-executed code, and of code only in the reasoning text;<br>• plus the count of blocked imports. | `[GPT]` Without these, e8686506 reads as "model never coded". | Tests: valid candidate / unsupported import / timed-out check / reasoning-only code / host-verified each land in their own category. **New.** |
| 24 | Timeouts on the pip and venv subprocesses. | `[ours]` code audit M6. | Old #22. |
| 25 | Keep the two payload `.bin` files in the dataset until the hidden run ends. | `[ours]` code audit M6. | Old #23. |
| 26 | Write `submission.json` once at the end, not per branch. | `[ours]` code audit L2. | Old #24. |
| 27 | Abort the staging copy on timeout. | `[ours]` code audit L7. | Old #25. |
| 28 | Fix the stale labels (the CONFIG sha, "w1", variant names). | `[ours]` code audit L5 / L7. | Old #26. |

---

## P2 — solver-design experiments (measured arms, not defaults)

| # | Experiment | Source | Metric / notes |
|---|---|---|---|
| 29 | **Same-engine A/B:** direct Poetiq-style program synthesis vs Duck on the same SGLang INT4 + DFlash2 server, the same tasks and the same time per task. | `[GPT P2]` Poetiq R5 got 78/172 vs Duck 46/172 on the same 120 tasks, but with a different engine (vLLM FP8 + MTP). Linked to old #55 (SGLang vs vLLM S1, 29.17 vs 34.58) and old #62 (S1-60K EVAL120). | Candidate rate, pass@2 accuracy and effective tok/s, all three. **New.** |
| 30 | **Candidate-first protocol:** minimal hypothesis → `transform` → `check()` milestone, then failure-driven fixes. Includes composing the helpers into one grid→grid `transform`. | `[GPT P3, Causes A and B]` 64-67 no-progress tasks. LOW: 53/55 defined functions and 1,287 Python calls with no recorded check. | First valid candidate, first check, no-transform count, correct queries. Roll back if generalization drops. **New.** |
| 31 | **Train-exact generalization review / second independent hypothesis** on accepted tasks. | `[GPT P4]` 10 / 14 / 16 train-exact-test-wrong tasks. History cases: 0934a4d8, 78332cb0, 1ae2feb7. | Must not wrongly reject correct fixed-coordinate rules; never discard the accepted candidate. Pairs with D2. **New.** |
| 32 | **Facts view vs compact actual-grid view.** | `[GPT]` Facts and indexed views "operationally worked, insufficient candidate yield". | Candidate rate and accuracy. **New.** |
| 33 | **Hybrid:** Poetiq primary + targeted Duck on uncertain or incomplete tasks; host verify + diversity for a1 / a2. | `[both]` Old #61. GPT §5.4. | GPT's warning: Duck stage-1 added only +5 queries over Poetiq for 3.86 M tokens. The oracle union is 88/172, but routing may use only train-derived signals, never gold. |
| 34 | **Stratified 24-task paired set** from the 55 no-progress tasks (XHIGH-solved, Poetiq-only, none-solved, timeout / length-cut).<br>Arms:<br>• A — current XHIGH Duck;<br>• B — `branches=1`;<br>• C — hypothesis-first;<br>• D — direct Poetiq-style. | `[GPT]` | Success = **correct queries per GPU-hour**, not more checks. See D7 for the noise caveat. **New.** |

---

## P3 — engine experiments (E15)

**GPT E15 audit fixes (mandatory before the GPU run):**

| # | Fix | Notes |
|---|---|---|
| 35 | An arm counts as PASS when its probe did not run or failed → fail closed (`_probe_valid`). | Old #28 |
| 36 | The latch fix has no long-context speed condition. | Old #29 |
| 37 | The 14 / 16 / 20-stream latch tests actually measure at 12 streams. | Old #30 |
| 38 | The Duck launch is not verified: check the real arm, the capacity, and a smoke run. | Old #31 |
| 39 | The probes for N > 12 streams do not compare every row. | Old #32 |
| 40 | An arm shows "ok" despite in-window errors. | Old #33 |
| 41 | Verify the real MTP KV size (#44145). | Old #34 |

**Remaining E15 arms and work:**

| # | Item | Notes |
|---|---|---|
| 42 | Measure the 11-of-12 fix (+2.2 %). | Old #35 |
| 43 | 14-16 streams (+8-12 % predicted). | Old #36. With D1-B′ / item 4, re-check the KV fit at the longer context. |
| 44 | MTP. | Old #37 |
| 45 | Width 4. | Old #38 |
| 46 | Chunked prefill 2048. | Old #39 |
| 47 | NCCL protocol. | Old #40 |
| 48 | FP8 drafter. | Old #41 |
| 49 | Triton attention. | Old #42 |
| 50 | `SKIP_DECODE_LOCK`. | Old #43 |
| 51 | 2×TP2 boot test. | Old #44 |
| 52 | Re-measure the reference at the end. | Old #45 |
| 53 | The 65k window. | Old #46. **Becomes ~100k if B′ is chosen.** |

---

## P4 — accuracy and the "kernel issue" (E17)

| # | Test | Source |
|---|---|---|
| 54 | #43567 KL test, with and without the patch. | `[ours]` Old #47 |
| 55 | fp32 vs fp16 SSM (needs approval; = D5). | `[ours]` Old #48 |
| 56 | Score pack (teacher forcing). | `[ours]` Old #49 |
| 57 | Nucleus audit. | `[ours]` Old #50 |
| 58 | Cache-restore parity. | `[ours]` Old #51 |
| 59 | p(EOS) at mid-thinking EOS sites. | `[ours]` Old #52. GPT: Qwen #216 (empty-content EOS) is not the primary culprit; XH had 0 `finish_reason=stop` of ~1,569 requests. |
| 60 | Log the FP8 KV scales. | `[ours]` Old #53 |
| 61 | INT4 vs FP8. | `[ours]` Old #54 |
| 62 | SGLang vs vLLM on S1 (the 29.17 vs 34.58 puzzle). | `[ours]` Old #55. Links to item 29. |
| 63 | **Target-only vs DFlash2 W6 quality isolation:**<br>• same INT4 target, prompt tokens, seed, FP8 KV and FP16 SSM;<br>• short, 30k and 60k contexts;<br>• concurrency 1 and 12;<br>• **greedy parity** (numerical divergence) and **sampled pass@2** (task quality) kept separate;<br>• check tool-call integrity. | `[GPT P5]` + our E15 `target_only` probe. GPT-cited, unverified: SGLang #36548 (DFlash2 errors at concurrency 8), #38009 (thinking divergence at token 30), #35723 (repetition loops / no tool call). Disable DFlash2 only after this result. **New (merges with 54 / 58).** |

---

## P5 — platform and process

| # | Item | Source |
|---|---|---|
| 64 | JIT tar bug fix and compile cache (−6 min). | Old #56 |
| 65 | Wheelhouse as a Kaggle Model. | Old #57 |
| 66 | Measure the margin from container start to 12 h (~300 s). | Old #58 |
| 67 | Test cancel via the API. | Old #59 |
| 68 | **10-26 deadline:** every team member must accept the rules (user action). | Old #60 |
| 69 | Hybrid notebook (= item 33). | Old #61 |
| 70 | S1-60K EVAL120: never run; it is the baseline for item 29. | Old #62 |
| 71 | Fill the `[PENDING]` parts of the write-up and the paper. | Old #63 |
| 72 | GPT mentions a **2 November** final deadline: confirm it on the competition page and plan the remaining submissions. | `[GPT]` **New.** |

---

## Recorded findings (not fixes; they constrain decisions)

1. **Tool calls do not buy accuracy.** `[GPT]` LOW made 2,506 tool calls and about 60 % more requests than XHIGH, with better cache hits (82 %), yet had the worst accuracy.
2. **Most tokens go to no-progress tasks.** `[GPT]` 63-65 % of generated tokens (4.5-4.6 M) went to the 64-67 tasks with no recorded transform or progress, so engine speed alone cannot fix the score.
3. **Root-cause ranking (GPT).**

   | Rank | Cause |
   |---|---|
   | 1 | Hypothesis generation and tool-use behaviour |
   | 2 | Compute allocation (b1) |
   | 3 | Generalization (train-exact, test-wrong) |
   | 4 | Engine numerics (unmeasured) |
   | 5 | Parser, EOS and infrastructure (minor) |

4. **What worked and stays.** `[GPT]`
   - stage-1 full coverage, 120/120;
   - KV-only admission at 12 workers;
   - the rev 5 host verification;
   - the pass@2 selector (correct, but short of candidates);
   - cache tracking as a diagnostic.
5. **Public-eval oracle unions are not LB forecasts.** Poetiq R5 went 48.61 % eval → 21.39 LB. `[GPT]`
6. **Public task IDs and labels** may be used for analysis and balancing, never for routing on hidden tasks. `[GPT]`
7. **Prompt-only code-first instructions failed** as a behavioural lever. `[both]` GPT: redesign how hypotheses are proposed (→ item 30).

---

## 7. GPT's second report (Notes/45): verification and merged items

The script is `_scratch/eval120_analysis/scripts/verify_gpt2.py`.

### 7.1 Verification

| GPT2 claim | Our check | Result |
|---|---|---|
| Scores 46 / 41 / 31 re-scored from `e12_generated_attempts.json` | Same re-score | **Confirmed.** |
| Poetiq R5 78/172 = 48.61 % | `ledger_units.csv` | **Confirmed.** |
| "Duck train-exact → else Poetiq" retrospective = 79/172 (49.86 %) | Same rule, gold-free switch | **Confirmed** for XH+E16.<br>Other runs: XH 78/172 (48.75 %); LOW 67/172 (41.81 %).<br>It ignores the time that Poetiq would need on about 73 tasks. |
| `E12_BRANCHES = int(os.environ.get("E12_BRANCHES", 2))` exists, so b1 can be turned off without a scheduler rewrite | `build_e14.py:145` | **Confirmed.** |
| `finish_reason=stop`: 1 / 0 / 8 (LOW: 7 at 58490d8a) | `requests[].finish` | **Confirmed** (1 / 0 / 8). |
| Branches with a reasoning trim: 66 / 78 / 86 | Branches with any `requests[].trimmed`: **34 / 41 / 64** | **Definition mismatch.** GPT's count probably includes the rev5 `resume_rewritten_after_trim` events. Not confirmed as stated. |
| 33 of the 56 first checks came after 28k | Consistent with section 0 (most late solves have their first check after 28-35k) | Agrees. |
| SGLang #43956 (batch-shape-dependent numerics, 0.5.21, FP8 / BF16) | Not opened by me | **GPT-cited, not verified.** A new lead → item 63. |

### 7.2 Key test: GPT2's "progress-aware continuation instead of a longer budget"

GPT2 recommends extending only branches that show code progress, not giving every branch more tokens.

We tested this rule at 35k: extend a branch only if it had a recorded check or transform by 35k.

| | XH+E16 | XH | LOW+E16 |
|---|---|---|---|
| Branches still unaccepted at 35k that kept running | 87 | 84 | 87 |
| … of which had a check or transform by 35k (would be extended) | 8 | 7 | 9 |
| Branches whose first correct program came after 35k | 16 | 13 | 6 |
| … of which had progress by 35k (**kept by the gate**) | **6** | **4** | **3** |
| Late solves **lost by the gate** | **10** | **9** | **3** |

**Conclusion:**
- Most late solves were still thinking without code at 35k: their first check came after 35k, at 35-61k. 135a2760, 36a08778, 409aa875, a395ee82 and eee78d87 checked for the first time at the very token that solved them.
- So a code-progress gate at 35k removes most of the late gains. GPT2's own regression examples (135a2760, 898e7135) show this: in XH+E16 the gate keeps 898e7135 but drops 135a2760.
- **Decision input:** a continuous budget to 60k (D1-B″) keeps every late solve but one. The progress gate is cheaper but loses 10 / 9 / 3.
- The progress-aware *priority* idea (which unverified tasks get attempt 2 first) stays useful for **attempt-2 ordering** → item 75.

### 7.3 New or extended items from GPT2

| # | Item | Priority | Notes |
|---|---|---|---|
| 73 | **12-task equal-wall comparisons** (EVAL12):<br>• b1 off + Poetiq-style fallback;<br>• Poetiq first pass vs Duck stage 1 on the same engine;<br>• DFlash2 vs target-only / MTP. | P2 (= items 29, 34, 63) | GPT2 prefers 12 tasks; GPT1 preferred 24. Our D7 caveat applies: 12 tasks are very noisy. |
| 74 | **Bounded program-commit path:** after long analysis the model writes an executable candidate of its current best hypothesis, even a partial one, and gets check feedback, instead of another inspection cycle. | P2 (extends item 30) | GPT2: "more nudges are not the highest priority." LOW shows that early code without a good hypothesis does not pay. |
| 75 | **Progress-aware ordering for attempt 2:**<br>• tasks with partial or train-near programs first;<br>• then verified-but-ambiguous tasks;<br>• then tasks with no progress. | P1 for config B″ | Gold-free. Regression set: 135a2760, 898e7135 must not be cut early, and cbebaa4b-type low-yield branches stay bounded. |
| 76 | **Distinct second-method candidate as attempt_2** (Duck + Poetiq or DIRECT distinct verified predictions; reject an invalid shape or a non-grid output; dedupe the same prediction). | P1 (extends items 7 and 19) | GPT2: the selector works and loses 0, so do not rewrite it; feed it candidates. |
| 77 | **Engine correctness gate before any engine switch:** same input and context; greedy mismatch, numerical margin, tool-call validity and ARC code correctness, measured separately; an MTP arm next to target-only. | P4 (extends item 63) | GPT2: "no direct DFlash2 fingerprint in our 604 transcripts". |
| 78 | **Object-centric or grid views on the identified hard failures** (A/B). | P2 (extends item 32) | Transcripts show spatial misreadings, e.g. 13e47133. |
| 79 | **Raw token-prefix / Mamba cache diagnostic** (CPU token diff, then a bounded GPU check). | P3 | XH+E16 `e16_raw_content_kept` = 0, so the cache gain is unproven. A serving lever, not accuracy. |
| 80 | **Later research branch:**<br>• DSL search;<br>• bounded program mutation;<br>• third-party ensembles. | Later | Higher engineering risk. |
| 81 | **Do not cut the Poetiq iteration budget to 24k** in any Poetiq arm; stay at 60k / 77.8k. | Rule | Matches the user's 60k floor. |
| 82 | GPT2's second NO-GO: "only engine TPS tuning will not reach 60+". | Recorded | Consistent with recorded finding 2. Engine work (P3) stays, but is not the main lever. |

### 7.4 Additional GPT2 findings (recorded)

- **Missed queries by class (XH+E16):**

  | Class | Missed queries | Share |
  |---|---|---|
  | No usable transform | 95 | 75.4 % |
  | Checked, not train-exact | 15 | 11.9 % |
  | Train-exact, test-wrong | 16 | 12.7 % |

  No-transform misses in the other runs: XH 102 / 131; LOW 84 / 141.
- **Length-capped requests:** the 98 capped requests in XH+E16 hold ≈ 2.41 M tokens (34 % of generation). They are not all waste, but they are why the caps go (items 2-3).
- **Repetition:** branches with ≥ 1k repeated characters: 13 / 18 / 23. The repeated share is about 1 % of reasoning characters (detector-defined).
- **b1:** no tool call in 60 / 55 / 0 branches; a check in 1 / 3 / 9; host-verified 1 / 0 / 0; new correct 0 / 0 / 0.
- **E16 flips:**
  - E16 gained: 136b0064, 5dbc8537, 221dfab4.
  - E16 lost: 1ae2feb7, b5ca7ac4, 269e22fb.
  - Bootstrap +4.17 points, 95 % CI about −4.17 to +12.08. Supports D3.
- **Poetiq R5 context:** 11.65 M tokens; ~11.5 h; 77,824 thinking + 8,192 answer tokens per iteration; vLLM FP8 + MTP. It is not a one-variable comparison with Duck.
- **GPT2's final recommendation:** keep the 34.58 submission intact (= D6). Run E12_BRANCHES=1 + a Poetiq-style fallback same-wall test first; no notebook rewrite yet.
  - Our position: the user's rules ("bundle changes, big jumps"; "no base-protection excuses") favour building B″ with the proven fixes ON and testing it on EVAL120 G4B.
  - The P2 solver arms run as separate measured arms.
  - The user decides.

---

## 8. GPT's third report (Notes/46): verification and merged items

**Tag:** `[GPT3]`.

**Script:** `_scratch/eval120_analysis/scripts/verify_gpt3.py`.

### 8.1 Verification

| GPT3 claim | Our value | Result |
|---|---|---|
| XH first transform in 53 tasks: 7 below 12k, 20 at 12-28k, 26 above 28k | 53: 7 / 20 / 26 | **Confirmed.**<br>Other runs: XH+E16 56: 7 / 16 / 33; LOW 63: 13 / 24 / 26. |
| b1 tool-free tokens 1.39 / 1.28 / 0 M | 1.39 / 1.28 / 0.00 | **Confirmed.** |
| Branches with an `Unknown tool: submit` transcript: 22 / 27 / 23 | 22 / 27 / 23 | **Confirmed.** |
| Branches with a NameError: 19 / 21 / 23 | 19 / 22 / 23 | Confirmed (±1). |
| Branches with a Traceback: 53 / 59 / 69 | 53 / 59 / 69 | **Confirmed.** |
| XH train-exact tasks: 46 → 30 full, 2 partial, 14 wrong | 47 → 30 / 2 / 15 | Confirmed (±1). Ours also counts non-accepted train-exact checks. |
| XH queries with ≥ 2 distinct train-exact predictions: 6 | 7 (XH+E16 3, LOW 2) | Confirmed (±1). |
| Poetiq got 30 correct queries on XH's 67 no-transform tasks | 30 | **Confirmed.** Other runs: XH+E16 26 on 64; LOW 28 on 57. |
| E16 task-level: 16 wins, 12 losses, sign-test p ≈ 0.57 | 16 / 12 | **Confirmed.** |
| 78332cb0: transposed output shape (23×5 vs 5×23, both queries) | Same in XH and XH+E16 | **Confirmed.** |
| 247ef758 q1: one cell wrong | 1 cell (XH, LOW) | **Confirmed.** |
| b0039139 q1: width 24 vs 29 | 5×24 vs 5×29 | **Confirmed.** |
| Poetiq R5: 691 min, 11.65 M tokens | R5 analysis doc: 691 min / 11.65 M / **281 tok/s** | **Confirmed.** |
| LOW report: `configuration.REASONING_EFFORT = xhigh` at top level, effective `e12.E12_REASONING_EFFORT = low` | Same | **Confirmed (reporting bug).** |
| SGLang #40959, Qwen #236 | Not opened by me | **GPT-cited, not verified** — leads. |

### 8.2 The most useful new fact: efficiency, not total score

| | Duck XH (submitted) | Poetiq R5 |
|---|---|---|
| Correct queries | 41 | 78 |
| Generated tokens | 7.09 M | 11.65 M |
| Solver minutes | ≈ 350 | 691 |
| Correct queries per M tokens | 5.78 | 6.70 |
| Correct queries per solver minute | **0.117** | **0.113** |

- Poetiq's lead on EVAL120 comes from about twice the time. Per minute the two methods are equal.
- R5 ran at 281 tok/s on vLLM. On our SGLang engine (341-384 tok/s) the same Poetiq loop would get ~20-35 % more tokens per minute.
  - That is the strongest argument for item 29 (Poetiq on our engine).
- It also explains R5's LB of 21.39: 691 min for 120 tasks cannot cover 240 hidden tasks in 12 h.
  - S1-60K (the shorter Poetiq) got 34.58.

### 8.3 New decision

| # | Decision | Evidence | Recommendation |
|---|---|---|---|
| D8 | **What runs as attempt 2 in B″?**<br>(a) another independent Duck attempt;<br>(b) a Poetiq-style program-synthesis request: one long think at 60k-77.8k → `transform` → train verify → bounded iteration 2, on our engine. | `[GPT, GPT2, GPT3]` All three reports: the second branch must be a *different method*, since 79-83 second Duck branches gave 0 train-exact programs. Poetiq solved 26-30 queries on Duck's no-transform tasks. Equal per-minute efficiency (8.2). | **(b)** for unverified tasks.<br>Duck-verified tasks: Poetiq-style as the independent second hypothesis (D2), if time remains.<br>Needs the Poetiq prompt and verifier ported into the Duck runner (they exist in E10 / S1). |

### 8.4 New items

| # | Item | Priority | Notes |
|---|---|---|---|
| 83 | **Tool contract in the prompt:**<br>• exactly one tool, `python`;<br>• `check(transform)` and `submit(transform)` are functions inside it, not tools;<br>• never emit `<function=check>` / `<function=submit>`.<br>Plus a regression test: the tool names == {"python"} and the prompt contains the contract. | P0 (with item 6) | Item 6 (routing `submit` to the helper) is the main fix, because prompt rules are often ignored (memory: code-first lessons). This is the second layer. |
| 84 | Prompt arm: after at most 2 inspection-only calls, write a provisional `transform` and `check()` it; keep reasoning deeply after the feedback. | P2 arm (= items 30 / 74) | E16's nudge after 2 inspection calls had no proven gain, and LOW shows early code is not enough. Measure; no default. |
| 85 | **Retained-state inventory:** each tool result lists the functions currently defined in the sandbox. A "fresh call" contract: every call that uses a function must define it or name it from the inventory. | P1 | NameError in 19 / 22 / 23 branches (e.g. 0934a4d8_b1 `f1`). Possibly linked to the turn rollback (item 5): dropped turns leave history and sandbox state out of sync. |
| 86 | **Soft sanity score for train-exact candidates** (output-shape rule consistency, palette relation, object counts, agreement of an independent second program). A tie-break and ordering only, **never** a hard reject. | P2 (extends item 31) | 78332cb0 transposed shape, b0039139 wrong width, 247ef758 one cell. The test set is the 14-16 train-exact-test-wrong tasks (offline retrospective). |
| 87 | **Report the effective effort:** record the request-level effort, its source, and default vs override. The LOW report's top-level config said `xhigh`. | P1 (with item 28) | Reporting bug, confirmed. |
| 88 | **First solver experiment keeps the engine unchanged** (INT4, DFlash2 W6, KV, SSM), so any change is attributable to the agent design. | Rule for P2 | GPT3. Matches keeping P3 and P4 separate. |
| 89 | **Matched EVAL12 / EVAL24 suite:**<br>• fixed seed 20261005 and the same task order;<br>• strata: no-transform, transform-not-exact, train-exact-test-wrong, and already-solved controls (a regression check on solved tasks). | P2 (= items 34 / 73) | GPT3 does not recommend another full 120 run now → D7. |
| 90 | **New leads, unverified:**<br>• SGLang #40959: Qwen3.8 reasoning-history rendering problem (Anthropic endpoint);<br>• Qwen #236: long-prompt non-thinking tool-call failure. | P4 (with item 63) | The history rendering is relevant to item 5 and Duck's multi-turn prompts. |

### 8.5 Recorded (GPT3)

- **XH missed queries (131), by class:**

  | Class | Missed queries |
  |---|---|
  | No transform | 102 |
  | Transform, not exact | 8 |
  | Train-exact, test-wrong | 19 |
  | Partial-task remainder | 2 |

- **Generalization once train-exact:**
  - Duck: 30 full / 2 partial / 14 wrong of 46;
  - Poetiq: 49 / 7 / 15 of 71.
  - The two are similar, so the first need is **more candidates**, not better selection.
- **Duck stage-1 + Poetiq (oracle):** 79 / 83 / 80 (XH / XH+E16 / LOW) — same as GPT2. Running Duck stage 1 before Poetiq adds little.
- **GPT3 verdicts:**
  - engine: keep the baseline;
  - b1: NO-GO unchanged;
  - E16: inconclusive;
  - Poetiq-style program-first: GO for offline development.
