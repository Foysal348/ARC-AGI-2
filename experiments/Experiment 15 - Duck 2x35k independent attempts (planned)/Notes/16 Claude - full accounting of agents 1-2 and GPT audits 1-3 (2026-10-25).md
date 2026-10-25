# 16 — Every item from validation agents 1-2 and GPT audits 1-3, and what happened to it

**Date:** 2026-10-25.

**User request:** re-read both agents and the three GPT audits; nothing important may be left out.

**Sources:**
- Agent 1 (selection) and agent 2 (R3J scheduler): reports in the session.
- GPT audits 1-3: Notes/15.
- Agent 3 (weaknesses in the past data) is still running; its items will be added here.

**Code state:** rev 3k = rev 3j + the fixes below, in `src/duck_b/arc2_duck.py`. Unit tests **112 / 112**. Notebooks not rebuilt yet.

## Fixed in rev 3k

| Id | Item | Source | Test |
|---|---|---|---|
| S1 | Deferred attempt 2s and attempts ≥ 3 wait only until every attempt 1 has *started*. The old rule ("finished") idled streams behind a long attempt 1, and one leaked or stuck attempt 1 blocked all later work for the whole run | agent 2 B6, risk 1 | section 21 (no idle behind a long attempt 1; still no item before the last attempt-1 start) |
| S2 | The stage tag is as in rev 3h for stage 1 / 2 items (a resumed branch 0 = 2 again); deferred attempt 2 = 2, attempts ≥ 3 = 3 | agent 2 B8 | section 21 |
| S3 | A `stage2_items` error is counted (`stage2_errors`) and logged | agent 2 risk 2 | — |
| H1 | The harvest source is bounded by whole definitions, newest first. A raw 24k slice could cut a function or its imports / helpers. Only names whose definition is kept are returned | GPT 1 F1, agent 1 #8 | section 21 |
| H2 | A cut tool call keeps the start of its code (complete leading definitions), not the last 24k characters | GPT 1 F2, GPT 2 §5, GPT 3 F1 | section 21 |
| P1 | The partial test runs the best-progress source (a snapshot taken at the new best), not a later redefinition with the same name: seen in 4 of 23 partial tests | agent 1 A4 / #1 | section 21 |
| SEL1 | In a vote tie, attempt 1 ranks before a later attempt; then the latest time. Attempt 1 was right in 8 of 10 conflicts that had a right grid. pass@2 is unchanged with 2 grids; with 3+ grids attempt 1's best is no longer dropped first | agent 1 A8, #2 | section 22 |
| ORD1 | End of an attempt: the harvest runs before the partial test. The harvest gave the only end-of-attempt correct answer in the three runs; near the deadline the partial test could use up the grace. The old F1 concern is covered by P1 + the kept0 restore | agent 1 #7 | section 13 (order) + the F1 functional test |
| TINY1 | After the commit turn, with < 768 tokens left, no tiny final request. In 9aaea919 b0 and 247ef758 b1 the accepted checks already existed | GPT 2 §4C | section 22 |
| T1 | Test gaps: section 19 now checks "started"; the section 20 check `>= [0, 1]` (which passed for `[1]`) is now `{0, 1} <= set` | agent 2 risk 7 | — |

## Not changed, with reason

| Item | Source | Reason |
|---|---|---|
| **Notebook 1 (after3): accepted tasks' attempt 2 rarely runs in 6 h / 12 h** (simulation: 0 / 38, 0 / 72) | agent 2 B2 | The order the user specified; the consequence is reported. **User decision.** |
| B4: "all attempt 3s before any attempt 4" holds only among queued items | agent 2 | Inherent in a work queue; ordering among queued work holds |
| Deferred attempt 2s ordered by ORDER2 only (rev 3h had class 2 → 3, flagged first) | agent 2 B9 | Notes/12: flagged-first was not supported (gain 0); class 3 vs 2 has n = 10. Matters only when part of them fit (notebook 2: 32 / 38) |
| A setup failure of attempt 2 (or of a deferred / extra attempt) ends that task's chain | agent 2 B4 / risk 3 | Rare: the constructor only assigns fields. Not fixed |
| Attempts admitted up to 300 s before the deadline cannot finish | agent 2 risk 4 | The streams have nothing else to do; an early accept or harvest still pays (26 % of solves come at ≤ 20k tokens) |
| Env pitfalls (`E12_B_B1_ACC_MODE` not 1 / 2 → KeyError; skip flag ignored with a mode) | agent 2 risk 5 | Not used by the notebooks |
| `prio[0] == 1 and stage1_left > 0` check is dead | agent 2 risk 6 | Harmless |
| The origin key treats "auto" / "partial_test" accepted checks as submissions | agent 1 A1 | Never changed an order in 4 runs |
| Agreement counts only each attempt's best | agent 1 #3 | Low impact; not seen |
| Train-exact rule outputs rank below 0-pair partials | agent 1 #4 | No rule fired in 4 runs |
| A selection exception replaces the task with the input | agent 1 #9 | Never seen; there is a task-level try |
| `ended_by` per-attempt metric | GPT 1 step 2 | Reporting only; most fields exist |
| Commit turn without a tool call (6 / 24) | GPT 1-3 | Open. Thinking off → cache miss and lower tool reliability (GPT 2 cites Qwen docs). The thinking-budget processor is broken under speculative decoding (SGLang #26330 / #25536, GPT 3). Arm only |
| A generalization ranking for accepted-but-wrong programs | GPT 1 step 6, GPT 2 D | Selection loss 0 in 3 runs; overfit flags useless (Notes/12); no data for a ranker |
| **Direct-grid fallback** | GPT 2 §7 | **Measured** (Notes/15): 0.255 credit per attempt vs Duck 0.12-0.14 on Duck-unsolved tasks, about equal per token. Candidate arm; needs a prompt, parser and request path |
| Poetiq-style revise loop; compact check feedback; progress-dependent extra tokens; NVARC / genetic search / DSL | GPT 1-3 | Research arms; not in this build |
| "50K45K no-go" (386 tok/s) | GPT 1-3 | Disagree: busy-stream rate ≈ 430, measured 50K45K ≈ 10.9 h (Notes/15) |
| A matched holdout / repeated seeds before choosing a variant | GPT 1-3 | The user's R3J1 / R3J2 EVAL120 6-h runs are the next measurement |
| Keep the S1-60K 34.58 submission as fallback | GPT 2 | Informational (the user's submission choice) |

## Claude's own claims that the checks corrected

- 50K45K 135a2760: "the hit came from a partial". **False.** It came from an attempt-2 accepted check in the commit turn (agent 1 A9).
- Notes/11 F4 correction "never cut at < 1,024 left (9aaea919 would be lost)". **False.** The accepted check existed before the tiny request (GPT 2 §4C, verified).
- R3J "streams never idle", "defaults = rev 3h". **False** (agent 2); fixed by S1 / S2.

## Agent 3 (weaknesses on all past runs: 13 runs, 863 attempts) — every item

| # | Finding | Impact (tasks / 240) | Action |
|---|---|---|---|
| S1 | Wrong-above-right costs no hit within a run (15 queries rescued by attempt_2). Pooled loss appears only with many accepted attempts: 1-3 tasks per pool, 16.27 → 15.28 at k = 8 | ~0 in R3J (≤ 2 accepted attempts per task) | none; "vote all versions" only if attempts continue after an accept |
| S2 | Within one attempt, "latest accepted = best" is right 12 / 14 (earlier versions right in 2); "earliest first" variants are worse | ≤ 1 | keep (SEL1 changes only the cross-attempt tie; ~0 impact in R3J) |
| S3 | Votes are unreliable: wrong accepts agree in 31 / 60 pairs; 3 / 15 tasks have a wrong majority | — | do not stop on "two agree" |
| S4 | Partial ranking: only 1 correct partial in 448 no-accept queries; all rankings tested put it top 2 | ~0 | none |
| S5 | Partials are not near misses: median 30 % wrong cells; about 1 / 5 are near-identity | — | none |
| S6 | The test-input fallback is worth 0 and fills most slots; grids in the thinking are never test outputs. **Direct reasoning on no-accept tasks: 0.226 per attempt (13.33 / 59)**, 14 tasks never solved by any Duck attempt | **+1 to +3** (a method change, untested in Duck) | candidate arm (as Notes/15) |
| **S7 / W1** | **Accepts after the commit text are 67 % wrong (14 / 21) vs 21 % organic (53 / 67); Fisher p = 0.0002**; the same direction in all 4 E15 runs. Overfit flags carry no signal (31 % vs 30 %) | **+1 to +2** | **fixed: rev 3k W1** |
| S8 | The rule library fires on 0 / 120 eval tasks | 0 | none |
| R1 | The stage-3 tasks are mostly hopeless: of the 59 tasks with no accept in 2 G4B attempts, 27 were never solved by anything. **The E15 20 tasks are easier than average (G4B correct-accept 0.50 vs 0.22), so Notes/12's attempt-3 rate 0.156 is biased upward** | — | **Notes/12 A5 corrected:** attempt 3 ≈ 0.057-0.094 per attempt |
| R2 | Per attempt: attempt 2 on unsolved 0.124; deferred attempt 2 on accepted 0.059 (8 / 135); attempt 3 0.057-0.094; attempts 4-8 0.083 → 0.064 (model). The two competing items overlap per token | — | informs the order |
| **R3** | **At hidden scale, time binds right after the attempt-2 phase.** after3 (notebook 1) runs 0 deferred attempt 2s and 0 attempts 4+, so it behaves exactly like "skip" (R3I). before3 (notebook 2) runs 0 attempts 3, so it behaves like plain rev 3h. The same holds in the 6-h commit. `max_attempts` never binds | R3J order vs rev 3h: **−1 to +0.6** | report to the user |
| **R4** | after3 vs before3: \|difference\| ≤ about 1 task / 240; the sign depends on the attempt-3 value. Two EVAL120 runs: single-run sd ≈ 4.1 / 120, **P(notebook 1 run beats notebook 2 run) = 0.46**. The real G4B E16 vs noE16 runs differed by 5 tasks on identical settings | ≤ 1 | **do not choose the order from the two commit scores**; compare the per-item yields inside each run |
| R5 | Speed decides stage 3: at 39-40 s per 1k there are almost no attempt 3s; at 44 s even attempt 2 is unfinished. The measured E15 per-attempt speed is 37.0-38.5 s per 1k; decode at 40-60k contexts is 426 vs 480 tok/s | ±1 | a planning risk |
| R6 | Time waste is small: idle 0.07 %, 0 from the attempt-1 gate (130 queued attempt 2s); about 12-13 attempts cut at the deadline (1.1-1.3 % of compute), ≤ 0.4 correct | ≤ 0.4 | none (optional: a time-aware commit turn ≈ +0.5) |
| R7 | KV: max 0.81 (50K45K), 0 holds / retractions; 16 no-accept peaks would reach 945k > 852k, but staggered ≈ 0.6 | low | none |
| W2 | A second opinion after a later first accept (about 40 % wrong) is about neutral | ~0 | not built |
| W3 | Stage-3 targeting: no better gold-free signal than ORDER2 | — | none |
| — | Wrong-accept rates by run: E15 0.24 / 0.32 / 0.35 / 0.39; G4B XHIGH 0.26 / 0.35; LOW 0.51 (not 26-31 %) | — | the quote corrected |

## Rev 3k build (with W1)

| | Notebook 1 | Notebook 2 |
|---|---|---|
| File | `Q27-XHIGH-E15B_EVAL120_50K45K_S16_R3K1_DUCK_SGLANG.ipynb` | `Q27-XHIGH-E15B_EVAL120_50K45K_S16_R3K2_DUCK_SGLANG.ipynb` |
| sha | `6baa9361` | `c058d58b` |
| Order | after3 | before3 |

- **Duck package:** `q27x_e15b_duck_9e4b89285136.bin`.
- **Flag:** `E12_B_WEAK_COMMIT` = 1.
- **Unit tests:** 116 / 116.
- **`verify_items`:** ALL CODE ITEMS PRESENT for both (new item 3K). The older items' strings are updated where rev 3k changed the code on purpose: 52, 1F1, 3I, 3J.
