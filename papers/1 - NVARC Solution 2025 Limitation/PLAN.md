# PLAN: Paper 1, NVARC Solution 2025 Limitation

Keep this file current. Newest status first in section 1. Dates are BDT (UTC+6).

## 1. Status

- **2026-11-04 (later)**
  - All 58 saved versions fetched and scored. The union is unchanged at 64 units.
  - 6 full runs exist. The union-growth curve is ready (findings 4c).
  - Step 2 is done. Among the new logs:
    - EXPERIMENT cells exist only in arc-2026d-with-codex v1 / v2;
    - experiment-with-arc-agi-2 v10 adds a 3-selector benchmark (33.67 / 33.67 / 34.67);
    - arc-2026d-32-22lb-updated-v2 v1 has the old v3-style CHECK diagnostics (subset run, 35 puzzles).
- **2026-11-04**
  - Folder built. Findings written (`v25_v2_ceiling_findings_2026-11-04.md`, `v3_misleading_claims_audit_2026-11-04.md`).
  - `scripts/score_all.py` reproduces all scores.
  - The arc-2026d-with-codex v1 run was found (second candidate pool).
  - `scripts/fetch_versions.py` is downloading every saved version of all 34 NVARC notebooks into `kaggle versions/`. Each new version must be scored and searched for new insight (step 2).

## 2. Steps

1. [done] Collect notebooks; score every saved submission; candidate-pool analysis of v1 and v2; audits of v3 and v25.
2. [done 2026-11-04] Every version of every NVARC notebook:
   - fetch (`scripts/fetch_versions.py`);
   - re-run `scripts/score_all.py`;
   - read each new log for pool / experiment data;
   - update the findings file and README section 5.
3. Decide the open item: run the diversity / more-samples test, or state it as a limitation.
   - A GPU run needs the user's explicit go.
   - Candidate: v2 Experiment 34 (diversity penalty in `turbo_dfs`), and a run with more augmentations or samples, measuring pool growth.
4. Paper outline (sections below), then the LaTeX draft. Every number must come from `scripts/score_all.py` output or a cited log line.
5. Figures:
   - top-k coverage curve (v1, v2);
   - union growth vs number of runs;
   - where the answer is lost (generation classes: close / partial / far / no grid; selection);
   - unique-candidate histogram.
6. Endorsement for arXiv cs.LG / cs.AI (first-time submitters need an endorser). LaTeX source + figures + code / data link.
7. Kaggle discussion post: a short version of the findings, after the user approves the text.

## 3. Paper outline (draft)

- **Title (chosen by the user 2026-11-04).** "Where the NVARC Pipeline Saturates on ARC-AGI-2: Candidate-Pool Ceiling, Selection Headroom and Generation Collapse".
- **Sections:**
  1. Introduction: NVARC 2025 as the base of many 2026 entries; question: where does it stop?
  2. Pipeline: model, test-time training, `turbo_dfs`, augmentation scoring, dual selection.
  3. Data: our runs (dates, versions, wall time, tasks trained), scoring method (section 3 of README).
  4. Results:
     - selection headroom (oracle vs dual; top-k);
     - generation wall (68 % no correct candidate; similarity classes);
     - diversity collapse (unique candidates, pairwise similarity, no diversity term);
     - repeated runs (union growth);
     - LB noise.
  5. Pitfalls in the notebooks' own diagnostics (why numbers must be recomputed).
  6. Limitations: one model and one search procedure; more-samples and diversity variants not measured (unless step 3 runs).
  7. Conclusion.

## 4. Decisions and rules

- **Docs and chat.** Docs are in English; chat with the user is in Bengali.
- **Claims.** Only claims the data supports (README section 1).
- **No Kaggle push / run / submit without the user's go.**
