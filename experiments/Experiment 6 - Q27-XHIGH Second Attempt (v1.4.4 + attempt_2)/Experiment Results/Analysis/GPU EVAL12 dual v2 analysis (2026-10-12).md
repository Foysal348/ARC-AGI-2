# GPU EVAL12: dual-answer v2 (`foysalemonshanto/q27-xhigh-dual` v1, 2026-10-12)

**Run:** 18:47 → 20:14 (87 min).
- Setup: 705 s.
- Primary scheduler: 3,608 s at 288.5 tok/s, 0 preemptions, MTP acceptance 0.664.
- Second stage: 880 s.
- Prefix-cache hit rate: 0.844.
- `primary_unchanged_all` = true.

## Result

| | Hits / 22 |
|---|---|
| Primary (attempt_1) | **1** (136b0064#0) |
| Dual (attempt_1 + new attempt_2) | 1 |
| Marginal from the second answer | **0**; no fallback hit lost |
| Counterfactual policies (keep_fallback / replace_any / protect_supported) | 1 / 1 / 1 |

**Second-answer stage:**
- 22 submitted, 22 completed.
- **Only 10 parsed as a valid grid; 12 were invalid**: ragged rows, i.e. row lengths differ inside the fenced grid.
- 7 were valid and novel; they replaced attempt_2 and were all wrong.
- 20a9e565#0 used all 8,192 tokens (`length`), which alone made up most of the 880 s.

## Findings

1. **Re-sampling the answer from the same reasoning adds nothing here.**
   - On natural closes the second grid duplicates the first (3 of 4).
   - On forced closes it is often ragged or invalid (11 of 18) or a different wrong grid.
2. **Run-to-run variance is large with identical code and seeds** (GPU batching and MTP nondeterminism). Compared with run 7 (v1.4.4, 3/22):

| Unit | Run 7 | This run |
|---|---|---|
| 247ef758#0 | natural close at 36.8k, **hit** | forced close, ragged grid |
| 4c3d4a41#0 | forced close, **hit** | natural close at 26.7k, wrong |
| Primary-answer ragged grids (fallback to the test input) | 0/22 | **5/22** |

   - EVAL12 primary results by run: run 4 = 2, run 6 = 2, run 7 = 3, this run = 1.
   - **A single EVAL12 run cannot resolve differences of ±1–2 hits.**
3. **Independent samples remain the only evidence-backed source of second-attempt hits.** Across runs 4, 6, 7 and this run, the union of correct units is {136b0064#0, 4c3d4a41#0, 247ef758#0}, while each run alone gets 1–3.
4. **Ragged grids are a real failure mode:** 5 primary answers and 12 second answers.
   - These are large grids written after a forced close; one row is too short or too long.
   - The parser correctly rejects them. A "repair" (padding or truncating rows) would be a guess, so it is not proposed as a fix. It is listed only as a measurement item.

## Suggestions (user decision; nothing changed)

- Do not enable the same-reasoning second answer in the hidden run; it showed no value here.
- **Second-attempt candidates with evidence:** a second **independent** full sample (new seed, full 49,152 thinking).
  - In the hidden run, only using spare time, low priority and KV-guarded (see the Experiment 7 plan, "Part C").
  - attempt_2 = the second sample's grid when it is valid and differs from attempt_1.
- Treat EVAL12 differences of 1–2 hits as noise. Use a larger proxy set (PROXY-48) and/or repeat runs for decisions.
