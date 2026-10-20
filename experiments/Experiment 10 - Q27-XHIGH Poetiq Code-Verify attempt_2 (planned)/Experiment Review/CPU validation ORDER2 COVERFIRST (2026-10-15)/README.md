# CPU validation: ORDER2 + COVER-FIRST (Claude, 2026-10-15)

**Notebook:** `Experiment Notebooks/Q27-XHIGH-v1_4_4_E10_ORDER2_COVERFIRST.ipynb`, sha256 `de9b98e82ea15e95…`.
- Builder: `Build/build_e10_coverfirst.py`. It is deterministic; two builds give the same sha.
- Source: ORDER2, sha `b62ae567…` (GPU EVAL120 58/172).
- Diff: `ORDER2_TO_COVERFIRST.diff` (cells 1, 6 and 7).

## Why

| Run | Thinking per unit | Ordering | LB |
|---|---|---|---|
| v1.4.4 | 49k | — | 26.94 |
| E7 | 49k | yes | 27.92 |
| E9 | 86k | — | **22.92** |

- At full coverage, 86k beats 49k on EVAL120 (36.9 % vs about 25 %).
- So the hidden run most likely could not finish every unit at 86k.
- **Coverage must come first; depth only with the time that is left.**

## The one change

**1. Pass 1.** Every unit thinks up to **`PASS1_THINK` = 49,152** (the user's floor), in ORDER2 order. Natural closes, continue and forced closes work as before.

**2. Depth pass.**
- A unit becomes a candidate when its pass-1 trace reached 49,152 without closing and at least `DEEPEN_MIN_TOKENS` = 2,048 of thinking budget remain.
- **Only after every unit has been admitted**, candidates are resumed from their own thinking. The new request's prompt is the unit's prompt + its pass-1 thinking ids, token-exact; it continues for the rest of `THINK_BUDGET` = 77,824.
- Candidates go in ORDER2 rank order, while slots and the admit deadline allow.
- A resumed trace closes naturally, continues, or gets the usual forced close. The drain and hard stop apply as before.

**3. KV guard.**
- A resumed trace is admitted only if every running request can still reach its maximum length without preemption: the sum of (prompt + full generation budget + answer) ≤ KV tokens. One request always fits.
- This is needed because Poetiq R5 hit 99.5 % KV with 9 preemptions on long requests.
- On 4×L4 (1,353,600 KV tokens) this allows about 14 resumed traces at once.

**4. Answers.**
- The deeper answer becomes attempt_1; the pass-1 answer moves to attempt_2.
- **A unit never loses its pass-1 answer.** Kaggle counts a hit if either attempt is right.

**5. Logs and switch.**
- `deep_records.jsonl` logs every depth request.
- At the end, `records.jsonl` is rewritten with each unit's final attempts and score.
- The report has a `cover_first` block.
- `Q27X_COVER_FIRST=0` gives exactly ORDER2's behaviour.

## Harness (`Build/harness/harness_coverfirst.py` = `harness_e10` + 4 cover scenarios)

In the harness only, `PASS1_THINK` is set to 1,500 and `DEEPEN_MIN_TOKENS` to 256, so that a 3,000-token budget stands in for 77,824. The notebook values (49,152 and 2,048) are asserted first.

| Scenario | Log | Result |
|---|---|---|
| **cover** (EVAL120, 172 units) | `cover.txt` | **PASS.** 138 candidates; all admitted and finished; 104 changed attempt_1. Depth modes: natural 97 / continue 6 / forced 35. Depth requests start only after the last pass-1 admission, and each is checked to be **token-exact** (prompt + its own pass-1 thinking; max tokens = cap − 1,500). Scripted outcomes all as designed: a wrong pass-1 answer fixed by depth, a no-grid pass-1 answer fixed by a forced depth answer, a right pass-1 answer kept as attempt_2 when depth goes wrong. **No hit was ever lost.** `records.jsonl` rewritten with `deepened` |
| coveroff | `coveroff.txt` | **PASS.** `Q27X_COVER_FIRST=0`: no depth requests (ORDER2 behaviour) |
| coverdrain (EVAL12) | `coverdrain.txt` | **PASS.** Drain fired while depth requests ran: 11 candidates, 3 admitted and finished (forced at the drain or natural), 8 not started. Hits went 1 → 3; no crash; report and submission written |
| coverrerun (hidden rerun, 259 units) | `coverrerun.txt` | **PASS.** 207 depth requests; the deep answers reach `submission.json` (attempt_1 = FINAL for every unit) |
| mixed / drain / rerun | `mixed.txt` / `drain.txt` / `rerun.txt` | **PASS** (regression; with 49,152 > the 3,000-token harness budget no unit is a candidate, so ORDER2 behaviour) |

The `deepen_fits` unit test passes: one request always fits, and a reservation larger than KV blocks.

## What a GPU run will show (not a projection)

- Pass 1 is DIRECT at 49k: the setting of our best LB (27.92 with ordering), now with ORDER2.
- The depth pass then uses whatever time is left to bring forced traces to 77,824.
- On EVAL120 at full coverage, the result should match ORDER2 86k (58/172) plus any extra hits from the 49k answer in attempt_2.
- **On the hidden run, it adapts to the real time instead of assuming it.**
