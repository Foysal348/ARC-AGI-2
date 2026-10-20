# GPT forensic audit of ORDER2 EVAL120: validation (Claude, 2026-10-15)

**Method:**
- Script: session scratchpad `order2/verify_gpt_audit.py`.
- It uses the notebook's own `parse_grid` (cell 4) to reconstruct the 172 primary grids from `answers/`.
- The reconstruction matches `records.jsonl` (has_grid + pass1) on **172/172**.

## Claims checked

| GPT claim | Our check | Verdict |
|---|---|---|
| Miss taxonomy: no grid 7, wrong shape 29, 1–3 cells 11, 4–10 cells 20, 11–30 cells 14, >30 cells 33 (= 114) | identical | ✅ |
| 1–3-cell list: `247ef758#1`, `3a25b0d8#1`, `8e5c0c38#1`, `d35bdbdc#2`, `62593bfd#1`, `9385bd28#0`, `a251c730#0`, `c7f57c3e#0`, `2b83f449#0`, `4c3d4a41#0`, `dd6b8c4b#0` | identical | ✅ |
| 11 wrong shapes differ by a single row or column | 11 | ✅ |
| Forced-close misses: no grid 7, shape 21, 1–3 2, 4–10 7, 11–30 7, >30 25; plus 5 hits | identical | ✅ |
| `21897d95#1` is exactly right after a rotation; no other miss is recovered by any of the 7 symmetries | `np.rot90` k=1 (counter-clockwise 90° = clockwise 270°); the only rescue | ✅ one hindsight case, not a policy |
| `submission.json` shares no task id with eval | 0 of 240 overlap. Expected: the commit run writes the placeholder test file, and eval is scored separately | ✅ not a bug |
| Multi-test: 49 tasks, 101 requests, about 2.97 M (28.8 %) of thinking removable with one trace per task | 2.97 M of 10.30 M = 28.8 % | ✅ idealised |
| 15/54 natural closes at or below 49,152 were wrong | 15/54 | ✅ |
| B + C: 46 tasks / 68 inputs / 41 missed | identical | ✅ |
| `MIN_THINK_BUDGET` = 4096: a prompt over ~48,896 tokens gets less than 49,152 thinking instead of being skipped | `phase1_cap` = min(77,824, 106,496 − prompt − 8,192 − safety); skip only below 4,096 | ✅ real but conditional. Observed impact is 0: max prompt 16,400, all 172 at 77,824. GPT's fix (skip the unit) would turn such a unit into a certain miss. **The user decides** |
| "The 19 late hits would surely be lost with a 49k cap" is a hypothesis | correct; no 49k counterfactual exists | ✅ |

## Additional finding (ours)

**Multi-test tasks are mostly half solved:**
- Of the 49 multi-test tasks, **5 are fully solved, 15 partly and 29 not at all**.
- In 15 tasks the rule was found for one test input but not applied correctly to its sibling, because the separate traces are independent.
- This strengthens GPT rank 3 (one reasoning trace per task, all test outputs together). It could turn partial tasks into full ones **and** save up to ~29 % of thinking. Both need a GPU measurement.

## Where we differ from GPT's ranking

**GPT rank 2 ("one request, two useful outputs": a final-stage second grid or self-check):**
- The bundled Astra validation (`context/`) shows that direct traces write their grid at ~89 % of the trace.
- It also shows 0 rescues in 41 full traces.
- So a second grid without new reasoning is doubtful. We rank it below rank 3 and rank 4.

**GPT rank 4 (selective independent second trace):**
- This is the largest complementarity measured so far: ORDER2 ∪ ARC Prize xhigh reaches 50.3 % eval, vs 36.9 % alone.
- Its cost could be paid by the multi-test savings.
- We rank it second, after the multi-test experiment.

**GPT's "27.92 → 60 not established by these files"** is true of these files only. The user's position is that the model can reach 80+ ([[no-impossible-claims-lb60]]). The files show where our method loses, not a model ceiling.

## Suggested order (first draft; superseded by the revised order at the end: row 4 data shows a same-method second trace rescues only ~3–6 % of misses)

1. **One trace per multi-test task.**
   - Test on B + C multi-test tasks and EVAL12.
   - Measure: tokens, per-task full-solve rate, answer truncation.
2. **Selective second independent trace (≥ 49,152), funded by the savings from step 1.**
   - Measure: extra hits per M tokens.
3. **Cheap CPU items:** loop detection after 49k, and malformed-grid repair as an attempt_2 candidate.
4. **Parked:** rotation-as-attempt_2 (one hindsight case), KV-aware admission (needs a benchmark), and the floor-enforcement change (user decision).

## Row-by-row validation of GPT's ranked table (added the same day; script `order2/verify_gpt_table.py`)

| Row | GPT claim | Our data | Verdict |
|---|---|---|---|
| 0 | Floor gap (`MIN_THINK_BUDGET` 4096) | Thinking drops below 49,152 only for a prompt over **48,896** tokens. The largest prompt over all 120 eval tasks **and all training tasks** is **16,400**. Template overhead is exactly 52 tokens | ✅ real, but 3× away from ever triggering. Fixing it gains nothing; skipping would only turn such a unit into a certain miss. **Leave it** |
| 1 | attempt_2 gives 0 hits; rotation +1/172 | 0 attempt_2 hits; `21897d95#1` is the only symmetry rescue | ✅. A gold-picked single case, worth at most +0.58 pt. **Low priority** |
| 2 | 31 misses ≤ 10 cells; same-request verification | 31 ✅. 9 of the 11 one-to-three-cell misses are natural closes, i.e. confident decisions (`247ef758#1` overlap colour, `2b83f449` marker rule repeated 3×). Astra: grid written at ~89 % of the trace, 0 rescues in 41 traces | ✅ numbers. Mostly rule-decision errors, not typos, so a same-trace check has little upside. **Low–medium** |
| 3 | Multi-test: 49 tasks, 101 requests, 2.97 M thinking removable | ✅. Also 5 full / 15 partial / 29 none. **Answer cap is not a risk:** the gold grids of ALL test outputs of a task need at most 3,608 tokens (median 1,608 for multi-test tasks) vs the 8,192 cap. Answers today: p50 805, p90 1,793 tokens | ✅ **Rank 1** |
| 4 | Selective second trace; p unmeasured | **Measured now on all direct runs: a miss in one run is hit by another run of the same method only 6.1 % of the time (54/890).** Same-budget 49k → 49k: 20/620 = 3.2 %. Into an 86k run: 34/270 = 12.6 %, mostly the larger budget. E9 resamples: 0/14. ORDER2 misses seen before: 2/41 hit before | ⚠️ **Correction to our earlier ranking:** a second trace of the *same* method is poor value (≈ 3–6 % per miss, ~1 hit per M tokens vs 6.9 per M for the 49–78k band). The second attempt must be a **different method or view** (Poetiq program, another prompt or representation, an augmented view, an ARC-Prize-like configuration). R5 EVAL120 measures the Poetiq case |
| 5 | 6 malformed + 1 degenerate grid | ✅ shapes and widths as stated; `4e34c42c#0` is a single row of 4,095 "3"s. Gold-free repairs (pad/truncate to the majority width; drop off-width rows) recover **0/7** | ✅. **Drop** as an attempt_2 source |
| 6 | 4 loops; ≤ 114,688 tokens avoidable | Exactly 4 tails with zlib ratio < 0.05, all forced closes; arithmetic correct | ✅. Small; a cheap CPU-side safety item |
| 7 | 15/54 natural closes ≤ 49k wrong | ✅. `6e453dd6#0` 26.6k, `edb79dae#0` 27.7k, `dbff022c#0` 32.9k, all natural misses (ARC Prize solved them) | ✅. Needs a different method or view, not more tokens |
| 8 | ORDER2 AUC ≈ 0.55; 27 of 52 score-2 tasks without a hit | ✅ | ✅. Keep ORDER2 |
| 9 | KV max 87.3 %, 0 preemptions, 333 tok/s, 44-min tail | ✅ | ✅. Hypothesis until benchmarked |
| 10 | Poetiq 4/22, DIRECT 4/22, union 5/22; real selector gain 0 | ✅. 13 distinct Poetiq second guesses, 0 correct (Astra) | ✅. The R5 EVAL120 run decides |

**Revised order:**
1. One trace per multi-test task.
2. A second attempt from a *different* method, chosen by the R5 EVAL120 data (Poetiq vs DIRECT union on 120 tasks).
3. Loop stop (CPU-side).
4. Parked: rows 0, 1, 2, 5, 9.
