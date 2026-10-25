# 15 — GPT deep audit of 45K30K vs 40K35K (pasted by the user, 2026-10-25) + Claude's check

**Source:** the user pasted a GPT analysis of `Experiment Review/E15B_DEEP_ANALYSIS_45K30K_vs_40K35K_2026-10-25.zip`. The original Bengali text is in the session transcript. Below is an English summary of its claims (section numbers as in the paste), followed by Claude's check against our code and data.

## GPT's claims (summary)

1. **45K30K beats 40K35K, but not only because of the budget.**
   - 19/32 vs 11/32; 13 vs 7 fully solved tasks.
   - Examples:
     - 16de56c4: the 45K30K attempt 1 solved it at 18.2k, while the 40K35K attempt 1 failed after 40k.
     - b5ca7ac4: in 40K35K, attempt 1 accepted a wrong program at 25.4k.
   - Sampling and hypothesis variation are large, so "the whole gap is the 36-45k zone" is too strong.
2. **Paired comparison:** both solved 7; only 45K30K 6; only 40K35K 0; neither 7. +27.5 points, bootstrap 95 % CI about +10 to +47.5. Not a causal proof of the budget effect.
3. **Where compute goes (45K30K / 40K35K):**
   - no usable transform: 285k / 185k tokens;
   - train-checked with 0 pairs exact: about 210k / 525k;
   - output-limit no-tool events: 30 / 37;
   - cut calls: 6 / 5;
   - effective rate: 385.6 / 387.6 tok/s.
   - **Problems:**
     - A: the hypothesis never becomes a program;
     - B: the first candidate comes late (41-44k);
     - C: 6 / 24 accepts are test-wrong;
     - D: 6 / 24 commit turns have no tool call (`enable_thinking=False` risks a cache miss).
4. **What worked and what did not:**
   - **Worked:** 16 streams + delayer off; the 100k window; the independent attempt 2; the host verifier; the reasoning harvest (136b0064 b1); pass@2 assembly.
   - **Partial:** the commit reserve; the nudge.
   - **Not GPU-proven:** rev 3h cut-code recovery and nudge scaling.
   - **Not reliable for selection:** overfit flags.
   - The fresh attempt 2 is unlike the old indexed-view b1, so "switch branch 1 off" no longer applies.
5. **Rev 3h bugs:**
   - **F1 (P1):** in `b_harvest_source`, a source over 24,000 characters is cut in the middle of the raw text, so a valid candidate can be lost (reproduced with a 26k-character function: names `['transform']`, source length 0). Fix: AST-node-bounded selection; return only the names present in the source.
   - **F2:** cut-call recovery. `b_cut_call_code` can return "" for an odd truncation, and `_cc[-24000:]` can cut the start of a long source. Fix: keep `b_complete_prefix(code)`, count `cut_code_unrecoverable`, and use AST-bounded harvesting.
6. **Hidden budget:**
   - At the effective 385.6 tok/s: 45K30K ≈ 10.0 h (margin ≈ 1.5 h); the worst case 18M tokens ≈ 13.0 h.
   - 50K45K ≈ 11.55 h (conditional), so it is not approved as the production budget.
7. **Outside methods:**
   - Reported ARC Prize scores of frontier models.
   - Poetiq solver (API-based, program-commit interface).
   - Johan Land (multi-modal independent hypotheses + holistic judging).
   - HF datasets (Trelis, Nemotron-SFT-ARC-AGI, ritwika96).
   - Tufa Duck harness.
   - **Advice:** no fine-tune now; analyse datasets offline.
8. **Next steps:**
   1. the rev 3h harvest fix (F1 / F2) + tests;
   2. per-attempt `ended_by` metrics;
   3. freeze 45K30K R3H (sha 9ee0aa4e…);
   4. a hidden-run budget gate that never cuts the first exposure;
   5. a commit-turn experiment (no template change in production);
   6. overfit-aware comparison without auto-reject.
9. **GPU order:** CPU rev 3h regression → EVAL120 G4B 45K30K → a matched budget probe with repeated seeds → a Poetiq-style arm → long-context correctness → 50K45K only if the time allows.
10. **Verdict:**
    - GO: 45K30K as the next evaluation candidate, after the harvest fix.
    - NO-GO: 50K45K or frontier replacement for hidden directly.

## Claude's check (2026-10-25)

| GPT claim | Check against our code / data | Verdict |
|---|---|---|
| 1-2: the gap is not only the budget | Agreed. Notes/11 already says n = 20 and noise-level; 16de56c4 is a sampling difference. Since then, 50K45K measured 20/32 (62.5 %), ≈ 45K30K | Agree |
| 3: compute categories / problems A-D | Consistent with Notes/11 F2-F3 and the loop scan | Agree |
| 4: what worked | Consistent; overfit flags are useless for selection (Notes/12 A4: flagged class 2 gain 0) | Agree |
| 5 F1: harvest source truncation | **Confirmed in code** (`b_harvest_source`: a raw slice at 24,000 characters, then cut to the first `\ndef `; can drop a whole function or the imports / helpers, and the names returned may not be in the source). **Never triggered in our runs:** the largest harvested source was 5,229 characters (50K45K 898e7135 b1); most harvests found no solution function at all. Agent 1 found the same risk (helpers cut). | Real edge bug, cheap fix → fix |
| 5 F2: cut-code `_cc[-24000:]` | Confirmed in code: the suffix keeps the end of a long call, while the complete leading definitions are at the start. In 50K45K the 3 saved cut codes were harvested (0 accepted) | Real edge bug → fix (keep `b_complete_prefix` of the full code) |
| 6: 50K45K ≈ 11.55 h, not approved | **The rate used is biased for 240 tasks.** 385.6 tok/s = generated / wall in a 20-task run, where the streams were busy only 88 % of the time (the tail of the last attempts). Busy-stream rate ×16 = 437 (45K30K) / 427 (50K45K). Measured 50K45K: 69.8k tokens per task → 240 × 69.8k / 427 ≈ 10.9 h of 11.6 h (Notes/11). Notes/12 E: even when the attempts need more than 11.6 h, the cut part is the lowest-value tail (the attempt 2s of accepted tasks run last) | Partly disagree: 50K45K fits (≈ 10.9 h, tight) |
| 7: outside methods | Information only; frontier scores are not verifiable here and not deployable offline. Fine-tune: not now (also the user's rule) | Noted |
| 8 step 2: `ended_by` metrics | Partly present (`b_events`, `commit_at_tokens`, `first_transform_tokens`); `ended_by` is not explicit | Cheap; optional |
| 8 step 4: budget gate | Present as the stage order (all attempt 1s first) + min-admit 300 s + deadline-clamped requests; rev 3j adds the deferred attempt 2 / stage 3 order | Already covered |
| 8 step 5: commit-turn experiment | Same as Notes/11 F2 (open) | Agree, open |
| 8 step 6: no overfit auto-reject | Selection does not use the flags; R3J dropped "flagged first" (Notes/12) | Agree |
| 9: EVAL120 before a hidden 50K45K | **The user's plan already does this:** R3J1 / R3J2 EVAL120 6-h runs | Agree |

**New fixes to bundle with the validation agents' findings** (before pushing R3J):
- F1: AST-bounded harvest source.
- F2: keep the complete prefix of a cut call (not the last 24k characters).
- Agent 1: the partial test must run the best-progress source, not the current definition with that name.

---

# Second GPT audit (pasted with the first one again, 2026-10-25)

## Its claims (summary)

1. 45K30K leads: 11 queries both, 8 only 45K30K, 0 only 40K35K, 13 neither; one stochastic run each, so it is not proof.
2. **The worst-case budget uses the agent-effective rate, 386 tok/s.**
   - 45K30K all at the cap: 18M / 386 ≈ 12.95 h, which does not fit.
   - 50K45K: no-go as is.
   - An attempt-2 admission gate is needed.
3. **Tokens on solved / unsolved tasks:** 642k / 514k (45K30K) and 243k / 942k (40K35K).
4. **Agent problems:**
   - **A:** no candidate in 8 attempts, 285k tokens (e.g. 62593bfd b1 nearly wrote the rule). Fix A1: a measurable "commit instruction" arm.
   - **B:** 6/24 and 6/26 commit turns with no tool call. Qwen docs: forced `tool_choice` is unsupported in thinking mode; `enable_thinking=false` lowers tool-call reliability on long prompts. Keep the thinking-on prefix; test forced choice or a prefilled `</think>` only as an arm.
   - **C:** 8 tiny post-commit requests. Fix: finalize when commit has been sent, remaining < 768, and no pending call (P2).
   - **D:** 6/24 accepts test-wrong. A gold-free generalization ranking (features) as a soft ranking, not auto-reject.
5. **Rev 3h:**
   - Cut salvage OK, with limits: an incomplete escape; the last 24k characters.
   - Nudge scaling is partial.
   - The scheduler cap and the slot-leak fix are present.
   - No fatal regression found; CPU-validated, GPU-unverified.
6. **Outside:**
   - ARC Prize frontier scores; Poetiq (candidate → feedback → refine → self-audit);
   - NVIDIA Nemotron-SFT-ARC-AGI (68 % of the successful traces are reasoning without tools) and the Nemotron RL ARC (transductive vs python_inductive);
   - a Bayesian program-induction repo (0 % on ARC-AGI-2: negative evidence).
7. **Proposal:** a direct-grid fallback for tasks with no verified program. ARC Prize Qwen3.8 xhigh direct solved 136b0064, 65b59efc, 898e7135, which our 45K30K missed. Prompt + grid parser sketch.
8. **Priorities:**
   - P0: rev 3h negative tests; a 240-task budget-safe scheduler.
   - P1: a matched holdout comparison; the direct-grid fallback; a Poetiq-style revise loop.
   - P2: generalization ranking; removing the tiny calls.
   - P3: 50K45K; engine changes.
9. **Verdict:**
   - GO: rev 3h fixes; 45K30K as the baseline.
   - Conditional: the 240-task submission.
   - NO-GO: 50K45K as is.

## Claude's check

| Claim | Check | Verdict |
|---|---|---|
| 2: the 386 tok/s worst case | Same error as in audit 1. In the 20-task runs the streams were busy only 88 % of the wall time (the tail); the busy-stream rate × 16 = 437 / 427 tok/s. 240 tasks have a near-zero tail share. Measured 50K45K: 69.8k tokens per task → ≈ 10.9 h. A full-cap worst case does exceed the window, but the scheduler already runs every attempt 1 first and the accepted tasks' attempt 2 last (R3J), so the cut falls on the lowest-value work (Notes/12 E) | Disagree with the no-go; the admission order already exists |
| 4C: the tiny post-commit requests | **It corrects Claude's own Notes/11 F4 correction.** 9aaea919 b0 (45K30K): the accepted check came at 44,592 tokens, inside the commit-turn request; the 408-token request only added a redundant accepted submission (44,688). Same for 247ef758 b1 (check 29,790, then submission 29,847). The selection already uses accepted checks, so finishing when < ~768 remain with no pending call would not have lost them. The earlier claim "never cut at < 1,024: 9aaea919 would be lost" was wrong | Agree: a safe, small time saving (P2) |
| 4B: no forced tool choice / thinking off | Consistent with Notes/11 F2 (open) | Agree |
| 4D: generalization ranking | Overfit flags are not usable (Notes/12); a feature ranking would need data (only 2-slot conflicts matter; selection loss is 0 in 3 runs) | Agree, low priority |
| 7: direct-grid fallback | **Measured on our data** (`Experiment Results/Analysis/scheduling rules (2026-10-25)/direct_vs_duck.py`), on the 74 tasks where a G4B XHIGH attempt 1 had no accepted answer. Credit per attempt: our E10 direct 86k run **0.255**; ARC Prize xhigh direct 0.297; a fresh Duck xhigh attempt 0.122 / 0.142. The direct run uses ≈ 101k tokens per task vs ≈ 49k for a Duck attempt, so per 100k tokens: direct 0.252 vs Duck 0.267, about equal. It is a different method, so its value need not decay with repeated attempts the way repeated Duck attempts do (k = 4-5: 0.26-0.30 per 100k). | A real alternative for stage 3, about equal per token. Not built (a new prompt, parser and request path); candidate arm |
| 5: rev 3h | Consistent with our tests | Agree |
| 6: outside | Information | Noted |

---

# Third GPT audit (pasted 2026-10-25)

## New claims (beyond audits 1-2)

- **90-91 % of generated tokens come before the first executable transform** (1.05M / 1.07M).
  - The tokens in attempts with no accepted program: 49 % / 66 %.
- **Transcript counts (45K30K / 40K35K):** Traceback 11 / 8 attempts; NameError 4 / 4; invalid grid 2 / 1; unknown tool 0 / 0 (the submit-dispatch fix works).
- **Weakness B:** after a failed `check()`, the correction loops run out of budget (65b59efc b0). Proposal: a compact, structured check result (the failing pair, expected vs actual shape, the best previous program).
- **Cut code (line ~1976):** `_cc[-24000:]` keeps the suffix, so a long call loses its `def transform` header. Reproduced with a 75k-character call. Minimal patch: keep `_cc[:24000]` and report `cut_code_truncated`.
- **SGLang issues #26330 / #25536:**
  - the thinking-budget logit processor can be bypassed under speculative decoding;
  - Qwen3.5 / 3.6 thinking-token-ID mismatches.
  - So a thinking-budget processor on 0.5.21 + DFlash2 is not a verified commit-turn fix.
- **Outside:**
  - frontier ARC Prize scores (closed / API);
  - NVARC / ARChitects (test-time LoRA);
  - LLM + genetic search (Qwen3.5-4B 2/60 → 6/60 with genetic search);
  - Daniel Franzen's Duck modifications (prefix reuse, faithful tool feedback, progress-weighted scheduling);
  - a manual DSL solver;
  - a residual portfolio (2/120);
  - ThinkingCap-Qwen3.8 (unverified).
- **Paired sign test:** 5 discordant tasks, all for 45K30K, p = 0.0625.
- **Idea:** progress-dependent extra tokens. A task with train progress but no exact program at the end of attempt 2 gets a few thousand extra tokens.
- **Priorities:** (1) the rev 3h salvage hardening + SHA-matched validation; (2) compact check feedback; (3) failure-aware compute routing; (4) a Poetiq-style refinement arm; (5) NVARC complementarity; (6) DSL / genetic mutation of partial programs; (7) prefix-cache efficiency; (8) engine rewrite later.
- **Verdict:** 45K30K is GO after SHA-matched validation; a blind 50K45K hidden rerun is NO-GO.

## Claude's check

| Claim | Check | Verdict |
|---|---|---|
| Cut code `_cc[-24000:]` → keep the prefix | Same as audit 1 F2; confirmed in code. A prefix of the full code (then `b_complete_prefix`) keeps the leading complete definitions | Fix (in the bundle) |
| SGLang thinking-budget processor issues | Consistent with Notes/11 F2: not adopted | Agree |
| Compact check feedback | The current `check()` result already gives the exact-pair count, identity detection, wrong-cell counts and a sample of (row, col, got, expected) per pair (seen in 9aaea919 b0). Expected-vs-actual shape and the best previous program are not shown. An extra could be tested as an arm; no data on the gain | Partly exists; optional arm |
| 386 tok/s worst case; 50K45K no-go | Answered in audits 1-2 (busy-stream rate ≈ 430; measured 50K45K ≈ 10.9 h; R3J order makes a cut fall on the lowest-value work) | Disagree with the no-go |
| Progress-dependent extra tokens | A continuation of the same attempt, not a pause; untested. Our data cannot measure it (no attempt continued past its cap). Notes/12 E: per-attempt gains plateau at about 55-60k | Hypothesis; not now |
| 90 % of tokens before the first transform | Consistent with the late first transforms (Notes/11); it includes the attempts that never wrote one | Agree (descriptive) |
| Traceback / NameError counts | Text-pattern counts; recovered tool errors are normal in a REPL loop. No loop or crash found (Notes/11 loop scan) | Noted |
| Outside methods | Information; none deployable now | Noted |
