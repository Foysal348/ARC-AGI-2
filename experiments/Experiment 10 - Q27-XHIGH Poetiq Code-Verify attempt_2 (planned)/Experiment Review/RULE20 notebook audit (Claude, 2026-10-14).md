# RULE20 notebook audit (Claude, 2026-10-14)

**Notebook:** `Q27-XHIGH-v1_4_4_E10_BASE_86K_RULE20.ipynb`, sha `2168cfb2…`. It is built by `build_e10_base.py --rule20`.

## 1. Difference from the E10 base (`530cb119…`)

The two notebooks were compared cell by cell. Only two cells differ:
- **Cell 0:** the banner.
- **Cell 2:** a new profile row, `RULE20`, and the default commit profile changed from `EVAL12` to `RULE20`.

Everything else is identical: the other cells, the metadata, and the absence of stored outputs.

## 2. The rules

- **The selection checks out.** All 20 tasks are in the evaluation file, and each has at least 3 training examples and exactly 1 test input.
- **Reproducible.** Re-sampling with `random.Random(20261014)` from the 52 tasks that meet both rules gives the same list.
- **The rules choose the commit-run tasks only.** They do **not** order or filter the hidden run. In the rerun, the code does `CHALLENGES = dict(test_challenges); TASK_IDS = sorted(CHALLENGES)`, the same original order as v1.4.4 and E10 base.

## 3. Hidden run (240 tasks, 259 inputs)

| Check | Result |
|---|---|
| Profile in the rerun | `RERUN_ALL`. The new default does not apply (`PROFILE = "RERUN_ALL" if RERUN`) |
| Deadlines (minutes from notebook start; budget 11 h 50 min) | admit at **694** (11:34), drain at 696 (11:36), hard stop at 704 (11:44), verify at 707. These match v1.4.4's LB-26.94 run |
| Budget | thinking 77,824 + answer 8,192; `MAX_MODEL_LEN` 106,496 |
| Prompts (real tokenizer, placeholder test file) | median 2,064, max 12,788 tokens. No thinking budget reduced, none skipped |
| Prompts on evaluation tasks, for reference | RULE20: median 5,000, max 16,400. The full 77,824 holds up to a 20,224-token prompt |
| CPU rerun scenario (fake engine) | 259/259 inputs finished; model grids written to `submission.json`; PASS |

## Open risk (unchanged from E10 base)

86k has never run on the hidden set, so its time there is unknown.
- **The submission is protected by design:** units are not admitted after 11:34, the drain is at 11:36, and `submission.json` is always written.
- **If time runs short:** units still unadmitted keep the identity grid; units cut at the drain are force-closed.
- **Measuring it:** the only way to measure the hidden time is an LB submission.
