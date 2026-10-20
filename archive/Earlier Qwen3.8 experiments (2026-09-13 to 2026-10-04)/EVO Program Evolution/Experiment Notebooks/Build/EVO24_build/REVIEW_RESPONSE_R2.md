# EVO24 — response to review round 2 (NO-GO: 1 HIGH + 2 MEDIUM + 1 efficiency concern)

All findings checked against the code and data first; all confirmed. 10 of the 24 frozen tasks have two test queries,
so the HIGH finding affects the measurement directly. Notebook after the fixes:
`PIV-Q27-EVO24-EVOLUTION.ipynb` sha256 `30fb30dbfc9874175313d12da57a49b3d187301a00545b39732b8bbc1f8560a9`
(R2 reviewed `d2c7650c…`).

| finding | fix | evidence |
|---|---|---|
| **HIGH** pass@2 chose two whole programs, not two outputs per query; partially valid programs (`[grid, None]`) were discarded | `evo_attempts_by_query(cands, n_queries)`: train-exact candidates in train-only rank order; per query up to two distinct non-None outputs. Programs that fail on one query still contribute their valid outputs on the others. Ground truth is used only in cell 9 scoring. | self-tests: the reviewer's A/X, B/X, B/Y case gives attempts `[[A0,B0],[X,Y]]` and is strict for truth (B0,Y); `[A0,None]+[None,Y]` gives `[[A0],[Y]]`. Harness: multi-query 36a08778 scored strict with per-query attempts |
| **HIGH** (same root) search stopped once two global programs differed | `evo_task_complete`: a task stops only when **every query** has two distinct attempts; any train-exact candidate (even partially valid on test) triggers the alternative-hypothesis slot | self-test: `_abc[:2]` incomplete (query 1 has only X), `_abc` complete |
| **HIGH** (scoring) | cell 9 scores `attempts_by_query[i]` directly; rows show `attempts_per_query` | `h3_normal.txt` |
| **MEDIUM** no deadline check before round 0 | worst-case round-0 estimate = 96 calls × 5,120 tokens / 60 tok/s (conservative vs the 65 tok/s baseline) ≈ 8,192 s; if it would cross the deadline the run stops with INCOMPLETE | `h3_deadline.txt`: "STOP before round 0", VERDICT INCOMPLETE / NO DECISION |
| **MEDIUM** notebook header still showed the old PROMISING rule | header now states exactly what `evo_verdict` does, incl. INCOMPLETE / INVALID and per-query pass@2 | notebook cell 0 |
| efficiency: identity-level candidates can still take repair slots | not changed for the first run (architecture frozen, as suggested); new metric `repair_parents_nonpositive_gain` per task and total (parents with 0 exact pairs and gain ≤ 0) | report field; harness total 35 (scripted answers) |

Runtime: planning range ~1.3–2.5 h wall, quota buffer for up to ~3 h wall (~6 quota-hours) as the reviewer suggested.

Harness runs on the fixed notebook: `h3_normal.txt` (all scenarios; per-query scoring; verdict PROMISING by construction),
`h3_deadline.txt` (stop before round 0 → INCOMPLETE), `h3_infra.txt` (injected verifier infrastructure failure → INVALID).
