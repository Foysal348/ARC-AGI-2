# Archive — earlier Qwen3.8 experiments (2026-09-13 → 2026-10-04)

These experiments came before the current phase: before Kaggle discussion 745787 showed that Qwen3.8 needs long xhigh thinking. Each family folder keeps the same sub-structure: `Experiment Notebooks/`, `Experiment Review/`, `Experiment Results/` (`Kaggle Runs/`).

| Family | Dates | What | Outcome |
|---|---|---|---|
| Q1 Direct Baseline (PROXY48) | 09-13 | Direct answer, thinking OFF vs XHIGH on 48 tasks | **0/56**; XHIGH hit the 12,288 cap 48/48 with no `</think>` |
| PIV-P0 Stateful REPL Agent | 09-14 | Stateful REPL + fresh train verifier (v1.0–v1.7) | XHIGH + tools: 0 tool calls; thinking never closed within 12k |
| ACT Reasoning-Activation Ablation | 09-15 | effort low / medium / forced-close arms (v1.8–v1.9) | low and forced close activate tools (0.875); medium does not |
| TP Throughput Probe | 09-16 | Engine and serving ladder | C1 engine about 129 tok/s vs 48 (eager) |
| C1 Terminal-Commit and Repair Solver | 09-17 → 09-18 | Exploration cap + structured finaliser + repair (v2.0–v2.5B) | ADAPTIVE_SMOKE **0/16** all-train-exact |
| PIV-SUB Hidden-Test Submission | 10-02 → 10-03 | First submission notebooks (v1.0–v1.3) | **LB 1.67** (matched EVAL120 1.67%) |
| V3 EVAL120 Baseline and S12 Subset | 10-03 → 10-04 | V3.1 all-fix, S12 subset | S12 strict 1/12, no new solves |
| EVO Program Evolution | 10-04 | Evolution over verified programs (EVO12 / EVO24) | built and reviewed; no GPU result |

The full history is in the project memory and in `resources/Analysis and Journals/`. Builders in these folders still point at the pre-2026-10-12 paths; see `../reorganization_manifest_2026-10-12.csv`.
