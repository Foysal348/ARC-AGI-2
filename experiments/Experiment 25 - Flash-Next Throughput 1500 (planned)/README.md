# Experiment 25: Flash-Next throughput 1500+ on Kaggle 4×L4 (planned)

**Goal:** sustain ≥ 1500 tok/s aggregate decode with Qwen3.8-Flash-Next on 4×L4 during a 12 h ARC-AGI-2 run. Correctness is held against BF16, and thinking stays at ≥ 60k, target 120k.

**Scope:** Flash-Next only. The 27B track is off (user 2026-10-31).

## Layout

| Path | What |
|---|---|
| `PLAN_1500.md` | The living research plan and action list. It is updated after every GPT research report and every GPU result |
| `Experiment Review/GPT research inputs/` | The research prompt (`00_…`) and each GPT report as received (`01_…`, `02_…`, `03_…`), verbatim |
| `Experiment Review/` | Source checks of report claims; fix lists |
| `Experiment Notebooks/` (+ `Build/`) | FN-S1 capacity, FN-S2 decode, FN-S3 architecture, FN-Duck EVAL builders and notebooks |
| `Experiment Results/` | GPU and CPU-kit results |
| `Resources/` | Downloaded docs, bench cards, configs (each with its URL and date) |

## Depends on
- E13 RC2: `foysalemonshanto/q27-xhigh-e13-flashnext-rc2` v1, notebook 12eaa96c, running since 2026-10-31. It gives the first correctness and speed numbers for Merlin Flash-Next on 4×L4.
- E13: the checkpoint Models, the RC2 code (`build_e13_rc2.py`), `rc2_unit_tests.py` and the CPU kit.
