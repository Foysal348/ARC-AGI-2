# 42 — Fixes and changes from the three EVAL120 G4B runs; design input for config B (2 × 35k independent attempts)

**Sources** (re-read in full on 2026-10-24):
- `Notes/EVAL120 analysis 2026-10-24/A_score_coverage.md`, `B_behaviour.md`, `C_engine.md`;
- the run folders `Experiment Results/Kaggle Runs/EVAL120G4B *`.

**Runs:** XH+E16 46/172 (30.0 %), XH 41/172 (25.83 %, = E14S, LB 22.78), LOW+E16 31/172 (18.89 %).

**User decision (2026-10-24):** "B: 2 × 35k independent attempts per task; first attempt on every task first, then the second" ("এটা করবো" — "let's do this").
- This is an explicit approval below the 60k-per-request floor, for this design.
- The rule "ARC-AGI-2 needs long thinking; no ARC-AGI-3 short caps" applies.

## A. Agent / solver (Duck adapter + scheduler)

| # | Change | Evidence (EVAL120) | Status for B |
|---|---|---|---|
| A1 | **Remove branch 1 (indexed view).** Its time goes to independent attempt 2. | 244 b1 runs → 0 correct programs; 26 % of stream-seconds; first request runs to the 24,576 cap with no tool call (B §3.1) | core of B |
| A2 | **No 28k pause and no stage-2 chunks.** Each attempt runs continuously to its 35k budget or an accept. | 58-66 % of prefill is resumed branches re-prefilled after cache eviction; throughput −18-30 % after hour 2 (C §3.4-3.5); 156-170 branches paused and never resumed (A §4) | core of B |
| A3 | **One long request per think phase:** cap = the attempt's remaining budget (≤ 35k). No 12,288 / 24,576 / 32,768 caps. | 98 / 84 length cuts with no tool call at 24,576 (C §6); caps came from ARC-AGI-3 (Franzen), not ARC-AGI-2 | core of B |
| A4 | **Fix the Duck turn rollback.** When the per-turn yield (`yield_tokens` 2,048) or a stop fires after the first of two parallel tool calls, keep the turn and drop only the unexecuted call. Today `preserve_history=False` discards the whole turn (`vendor/inference/agent/tool_agent.py` L7094-7101, L7183-7186). | 56 events / 35 branches / 0.52 M context tokens lost (XH+E16); 47 / 29 / 0.46 M (XH) (B §5.1a) | must |
| A5 | **Accept `submit` called as a tool**: route it to the python helper. | `Tool 'submit' is not defined` 22 / 28 / 23 per run (C §4; B §5.1) | must |
| A6 | **attempt_1 / attempt_2 assembly:** a1 = attempt 1's verified program; a2 = attempt 2's verified program when its output differs; then the best partial program executed on test; the test input last. | pass2 = 0: 58 of 60 accepted a1 had the test input as a2; partial programs never run on test (B §3.2) | must |
| A7 | **Attempt 2 also runs on tasks where attempt 1 was accepted** (after the unverified tasks, if time allows). Train-exact accepts are wrong 21-30 % of the time. | wrong accepts 5 / 8 / 9; another run solved 5-12 of the accepted-but-wrong queries (B §3.2, §4) | proposed ordering: unverified first, then accepted |
| A8 | **No context trimming inside an attempt.** 35k generated plus the prompt fits the window, so the trim at 51,396 must not fire. | trimmed branches correct 16 % vs 34 % (XH+E16), 5 % vs 30 % (LOW); mixed in XH (B §5.1b) | check the budget arithmetic |
| A9 | **Effort xhigh only.** | LOW −10 to −15 queries (p 0.014-0.031); 2× the wrong accepts (B §2) | fixed |
| A10 | **E16:** the early nudge is neutral and raw content is a no-op on xhigh. Turn E16 off in B (fewer moving parts). | the 89 nudged tasks score 23.50 vs 23.50; raw_content kept 0 (B §1.4, §1.2) | off (decision recorded; reversible) |
| A11 | **Do not use `overfit_flags`** for selection. | precision 42 / 30 / 63 % (B §4) | — |
| A12 | **End of an attempt at 35k without an accept:** if a one-grid transform exists, run check() and harvest it. Harvested check() programs gave 12 / 4 / 5 of the hits. | A §2 source table | must |

## B. Engine

| # | Change | Evidence | Status |
|---|---|---|---|
| E1 | Prefill re-reads disappear with A2 (continuous attempts keep the cache warm): about +12-15 % effective throughput | C §3.4 model, 341 → ~384 tok/s | comes with B |
| E2 | 11-of-12 delayer fix (mrr 14 + graphs 1..14, or nodelay + mrr 12): +2.2 % | C §2.3; occupancy 11.21 | measure in E15 first |
| E3 | 14-16 real streams: +8-12 % end-to-end predicted; KV p90 0.40 at 12 streams | C §8.2b | measure in E15 first |
| E4 | Compile/JIT cache: the first verify-graph capture is 363-376 s, so a cache saves about 6 min | C §1 | separate fix (the tar IndexError) |
| E5 | Width 6 is numerically clean up to about 52k tokens per request (accept never below 3.05). B's requests are ≤ 35k + prompt. | C §5-6 | keep |
| E6 | The server uses about 3 mamba slots per running request (the client assumes 4) | C §4 | information for the stream sizing |

## C. Reporting

- **R1** — the report misses about 1 % of tokens (requests cut at the deadline). Add the server-side count.
- **R2** — `retract_lines = 1` is a false positive: it matches `retraction_policy`. Fix the regex.
- **R3** — "accepted" means two different things. Report accepted and harvested separately.

## D. Budget check for B

- **Solve window:** about 42,000 s.
- **Tokens needed:** 240 × 2 × 35k = 16.8 M, i.e. about 400 tok/s.
- **What we have:** about 341 tok/s now. About 384 tok/s once A2 removes the resume re-prefill (C model). Tasks that are solved stop early (solved p50 40k across both branches today).
- **Expectation:** attempt 1 on all 240 tasks fits easily (8.4 M tokens ≈ 6.1 h at 384 tok/s). Attempt 2 reaches most tasks: unverified first, accepted last.
- **Measurement:** EVAL120 G4B at the hidden share against XH 25.83 % / XH+E16 30.0 %.
