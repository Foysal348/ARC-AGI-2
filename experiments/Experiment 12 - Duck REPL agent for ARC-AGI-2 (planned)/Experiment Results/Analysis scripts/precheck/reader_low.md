# Reader (low effort, EVAL12R5_LOW): where the pre-check() tokens go — 2026-10-21

Source: the subagent report, kept close to verbatim. Line numbers refer to each branch's `transcript.txt`; token positions are cumulative completions from `branch_result.json`.

**Requests of ≥ 2k tokens carry 73-99 % of the pre-check tokens:**

| Branch | Big requests / pre-check total |
|---|---|
| cbebaa4b | 48k / 65k |
| 65b59efc | 44k / 54k |
| 4c3d4a41 | 76k / 77k |
| 62593bfd | 101k / 103k |

## Slow branches

**cbebaa4b_b0** (first check at R40, 61.5k)
- **R1-R13** (0-23k): object / bbox dumps, full-grid prints, delta tables.
- **R14-R30**: 13 calls compare partial pipelines with the model's own code. The fitting rule first matches both pairs at **R27 (L6218, 51.8k)**; no `transform` exists, so nothing is banked.
- **R31-R39** (~7.4k): test-0 ambiguity resolved before any check.
- **R40:** transform + check, 2/2.
- **Errors:** 5 calls errored. Code is about 20 % of pre-check characters because helpers keep being redefined.

**65b59efc_b1** (check at R27, 53.6k)
- **R13** (one 14.4k-token think): the full rule is stated at **L2362-2372 (~27k)**. One pair looked anomalous because the model **misread** the output block `3 0 3/3 3 3/0 3 0` as a plus (L2261, L2401).
- **R14-R17** (~18.4k): test parsing, then the phantom anomaly. At L3586: "let me first implement the pure as-is rule and see the check results", but it does not.
- **R18-R26:** transform with its own `print("train exact:", …)` 4 times, 2 hand diffs, a special case.
- **R26:** dropping the special case gives 3/3. About **26k tokens** between stating the rule and checking it.

**4c3d4a41_b0** (check at R6, 76.8k)
- **R4:** 49,152 tokens, no tool call (cap hit). The rule appears at **L2101 (~31.5k)**.
- **R5:** 16.4k tokens, stop with no tool call, although it wrote "Let me code this up".
- **R6:** the prompt shrank from 66k to 19.9k (a real overflow; earlier tool results and R4's thinking gone). It wrote a transform with hard-coded coordinates, giving an **overfit accept**.

**62593bfd_b0** (never checked, 103k)
- **R1:** 49,152 tokens, no tool call. It says it will use code 6 times (L1236 ~26k … L1968 ~49k) but never does.
- **R6:** 46k tokens of hand-testing direction rules over 10 objects. No `transform` is ever written.

## Waste (approximate)

| Category | Tokens |
|---|---|
| No-tool reasoning that announces code | ~65k (4c3d) + ~25k (6259) |
| Rule known, check delayed | ~26k (65b5) + ~10k (cbeb) |
| Own comparison instead of check() | ~9k (65b5) + ~24k (cbeb) |
| Hand-read grids → phantom anomaly | ~13k (65b5) |
| Context loss after a no-tool turn | 1 overfit accept (4c3d) |

## Fast branches

- **bf45cf4b (14k):** one 12.5k think, then transform + check (3/3).
- **247ef758 (24k):** a 23k think, then a transform that fails one pair (2/3), fixed in two rounds from check's wrong-cell list.
- **b5ca7ac4 (20k):** a box-finder helper is built, kept and reused in `transform`; 2/3, fixed in one round.
- **Common to all three:** every big think ends in transform + check, even when imperfect. check() is the debugger; there is no own-compare; test inputs are looked at only after a check.

## Proposed changes (by the reader; not yet validated)

1. **Stream watchdog:** abort at about 10k reasoning tokens since the last tool call, or on "in code / write code / code this up", and re-prompt "Run that code now". This changes the per-request length, so it needs the user's approval under the floor rule. Also keep tool results in history after a no-tool turn.
2. **Prompt:** "As soon as your reasoning states a rule that explains most pairs, your next tool call must define `transform` and run `check(transform)`, even if a pair looks inconsistent …"
3. **Harness:** auto-run check() when a call defines `transform`. **Prompt:** "Never compare against `p['output']` yourself; check() prints the wrong cells. To test a partial idea, write a transform that does only that part and check it."
4. **Harness:** a precomputed object table + per-pair diff in the first message. **Prompt:** "If you are about to verify a hypothesis on more than 2 objects or cells, compute it in Python."
