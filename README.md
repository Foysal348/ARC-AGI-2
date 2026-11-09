# ARC Prize 2026: ARC-AGI-2 and Paper Track (Team Hack2Publish)

Workspace of our team for the Kaggle competition **ARC Prize 2026: ARC-AGI-2** (4x NVIDIA L4, no internet, 12-hour hidden rerun) and the **ARC Prize 2026 Paper Track**. Licence: Apache-2.0 (see `LICENSE`).

**Best submission:** 57851995, public leaderboard **54.17**. It is a Qwen3.8-Flash-Next tool agent served offline with the Merlin engine (vLLM 0.32 fork) on 4x L4, with W4A16 routed experts, an FP8 KV cache and MTP speculation. The submission notebook is public: https://www.kaggle.com/code/foysalemonshanto/q27-xhigh-e25-nb6-submit (version 1).

## Papers (ARC Prize 2026 Paper Track)

| Paper | PDF | LaTeX source |
|---|---|---|
| Verified-Answer Precision as the Acceptance Test for Speed and Quantization: A Qwen3.8-Flash-Next Tool Agent for ARC-AGI-2 on Four L4 GPUs | [PDF](papers/Paper%20Track%20Writeup%202026-11-09/pdf/Hack2Publish_Verified_Answer_Precision_ARC_AGI_2.pdf) | [paper/](papers/Paper%20Track%20Writeup%202026-11-09/paper/) |
| Decomposing the Score of an NVARC-Derived Pipeline on ARC-AGI-2: Generation, Selection, and Run-to-Run Noise (companion) | [PDF](papers/Paper%20Track%20Writeup%202026-11-09/pdf/Hack2Publish_Companion_NVARC_Decomposition_ARC_AGI_2.pdf) | [paper/](papers/1%20-%20NVARC%20Solution%202025%20Limitation/paper/) |

Authors: Md. Hamid Hosen, MD. Foysal Emon Shanto, Md Esfer Abdus Sami. The analysis scripts behind the main paper's numbers are in `papers/Paper Track Writeup 2026-11-09/analysis/`; those of the companion paper are in `papers/1 - NVARC Solution 2025 Limitation/scripts/`.

## Leaderboard history by method family (best public LB)

| Method family | Best public LB |
|---|---|
| NVARC-lineage test-time training (public-notebook derivatives) | 32.64 |
| Qwen3.8-27B, code-verifying solver (60k thinking) | 34.58 |
| Qwen3.8-27B, tool agent | 22.92 |
| Qwen3.8-Flash-Next 3-bit (klee100), tool agent (57896498) | 49.58 |
| **Qwen3.8-Flash-Next W4A16, tool agent (57851995)** | **54.17** |

The experiment index below was written during the competition; earlier entries describe the 27B phase.

## Layout

```
ARC AGI 2/
├── README.md                     ← this file (project overview + experiment index)
├── experiments/                  ← current phase (from 2026-10-05), one folder per experiment
│   └── Experiment N - <name>/
│       ├── README.md             ← goal, method, versions, results, lessons, status
│       ├── Experiment Notebooks/ ← our notebooks (+ Build/ = builders, tests, harnesses)
│       ├── Experiment Review/    ← zips and documents sent to GPT for audit
│       └── Experiment Results/   ← GPT outputs (GPT Outputs/), Kaggle run outputs (Kaggle Runs/), analyses
├── archive/                      ← earlier Qwen3.8 experiments (2026-09-13 to 2026-10-04), grouped by family
├── resources/                    ← papers, journals, plans/prompts, public reference notebooks, ARC-AGI-3 material
├── reorganization_manifest_2026-10-12.csv   ← every file moved on 2026-10-12: old path -> new path
└── reorganize_workspace.py       ← the script that did it (rules; dry-run by default)
```

Competition data is kept outside the workspace, at `D:\arc-prize-2026-arc-agi-2 (1)\`.

## Experiment index (current phase)

| # | Experiment | Dates | Status | Key result |
|---|---|---|---|---|
| 1 | [Q27-XHIGH Direct Reasoning](experiments/Experiment%201%20-%20Q27-XHIGH%20Direct%20Reasoning%20(LB%2018.75%20and%2026.94)/README.md) | 10-05 → 10-12 | **baseline in use** | LB 18.75 (v1.4), **26.94** (v1.4.4) |
| 2 | [R6 Code Loop](experiments/Experiment%202%20-%20R6%20Code%20Loop/README.md) | 10-06 | closed | EVAL12 2/22, 92.7k tokens/task |
| 3 | [Flash-Next Feasibility](experiments/Experiment%203%20-%20Flash-Next%20Feasibility%20(FN)/README.md) | 10-05 → 10-11 | closed (not viable) | garbage output, ~6 tok/s single stream |
| 4 | [Q27-AGENT REPL Agent and HYBRID](experiments/Experiment%204%20-%20Q27-AGENT%20REPL%20Agent%20and%20HYBRID%20(v1%20to%20v1.8.3)/README.md) | 10-11 → 10-12 | closed (redesign in #5) | EVAL12 agent 0/22 (interface failures) |
| 5 | [Q27-AGENT v1.9 Long-Think Synthesis](experiments/Experiment%205%20-%20Q27-AGENT%20v1.9%20Long-Think%20Synthesis%20(planned)/README.md) | 10-12 | deferred (GPU stopped) | v1.9.1 EVAL12 agent 1/22; server thinking budget ignored with MTP (vLLM #39573) |
| 6 | [Q27-XHIGH Second Attempt](experiments/Experiment%206%20-%20Q27-XHIGH%20Second%20Attempt%20(v1.4.4%20%2B%20attempt_2)/README.md) | 10-12 → | **active** (only scope) | — |
| 7 | [Q27-XHIGH Puzzle Ordering + KV Controller](experiments/Experiment%207%20-%20Q27-XHIGH%20Puzzle%20Ordering%20%2B%20KV%20Controller%20(planned)/README.md) | 10-12 → | **planned** (plan under GPT review) | — |

The earlier phase is summarised in [archive/README.md](archive/README.md).

## Working rules (agreed with the user)

**Approval and accounts**
- Nothing is pushed, run or submitted on Kaggle without explicit approval of that specific action.
- Accounts: `foysalemonshanto` (default) and `esfersami50`.
- Tokens are read in-process only, never printed.

**Notebook requirements**
- **Dual mode** for every notebook. A commit run = evaluation subset (EVAL12) and a report; a competition rerun = all hidden test tasks, full-schema `submission.json`.
- **Docker image pinned:** `gcr.io/kaggle-private-byod/python@sha256:37c64f7d…` (Python 3.12; the default Kaggle image moved to 3.13).
- **12-hour budget:** stop admitting work at about 11:34–11:35; cut the process at 11:50; the validator runs last.

**Token budget**
- **No thinking budget below the LB base** (49,152 continuous xhigh thinking per test input) without the user's explicit approval.

**Before every push**
- Static gate (`check_static.py`).
- Real-GPU-path rehearsal (`test_gpu_path_local.py`).
- Kaggle CPU DRYRUN (free).

**Audits**
- External (GPT) audit claims are verified in code and data before anything is changed.

**Submissions**
- One pending submission at a time (Kaggle, 2026-10-08).
- Scoring currently takes about 1.5–2 days.
- EVAL12 is our hardest subset; LB has been about 2× the EVAL12 percentage.
