# PLAN: Paper 1, NVARC Solution 2025 Limitation

Keep this file current. Newest status first in section 1. Dates are BDT (UTC+6).

## 1. Status

- **2026-11-04 (equations and pseudocode)**
  - Equations added (numbered, referenced from the text):
    - search bound (at most 4 outputs per view, at most 64 per test input);
    - probmul / kgmon labels;
    - pass@2 indicator and task-normalised score S;
    - oracle O, headroom H = O - S and top-k coverage C(k);
    - mean cell agreement a_u (as in notebook cell 30);
    - duplicate ratio d_u;
    - union growth U_m.
  - Pseudocode (algorithm / algpseudocode), written from the code:
    - Algorithm 1: the solver for one task, R2 code (TTT, 16 views in batches of 4, the 1600 s check, decode, inverse
      augmentation, rescoring, placeholder, dual selection = probmul top 2);
    - Algorithm 2: `turbo_dfs` as a recursive thresholded search (EOS and the K cap as in the code, 540 s per call);
    - Algorithm 3: our pool decomposition (`paper_stats.py`): classes, S / O / H / C(k), the subset assert, union growth.
  - No number changed. The paper is now 14 pages, 5 figures, 2 tables, 3 algorithms.
  - arXiv package rebuilt (`--draft`): clean build, abstract unchanged. Not submitted.
- **2026-11-04 (architecture figure)**
  - New Figure 1 (`fig:arch`, TikZ, at the start of Section 2):
    - (a) the solver pipeline: generation boxes in blue, scoring and selection boxes in orange;
    - red callouts give each measurement at the stage where it applies (generation, pool and diversity, selection, repeated runs), as R1 / R2 ranges taken from numbers already in the paper;
    - (b) how the numbers are produced: notebooks -> saved outputs -> `score_all.py` / `paper_stats.py` -> `NUMBERS.md` -> paper.
  - The paper is now 12 pages with 5 figures; METADATA comments are updated to match.
  - `make_arxiv.py --draft`: clean build (0 errors, 0 overfull / underfull), abstract unchanged (1,900 characters). Not submitted.
- **2026-11-04 (checked against the competition overview)**
  - The Kaggle citation is now the official one from the competition page: Chollet, Knoop, Kamradt, Reade, Cruz, Howard,
    "ARC Prize 2026 - ARC-AGI-2", Kaggle 2026.
  - Scoring:
    - The overview words the score as an average over test outputs.
    - All 18 distinct LB values in our records are multiples of 100/720, which matches per-task averaging over 120 tasks
      with 1-3 test inputs. No plain average over N outputs (N = 100-399) fits them. The check is in `paper_stats.py` `[lb]`.
    - The paper now says this, and states that LB scores are percentages (32.64 %, 5.28 percentage points).
  - The rest of the overview matches the paper: 2 attempts, offline, 12 h, 4xL4, pre-trained models allowed.
  - Timeline from the overview:
    - the final submission deadline was 2026-11-02;
    - Grand Prize write-ups are due within 7 days of it, i.e. by 2026-11-09;
    - the paper-track deadline is not on this page.
  - The arXiv package was rebuilt with `make_arxiv.py --draft`: 11 pages, clean build, abstract 1,900 characters. Not submitted.
- **2026-11-04 (step 6, arXiv conversion; draft package, NOT submitted)**
  - Re-checked before any change:
    - `score_all.py` gives the expected output (41; 64/172, 46.50; pools 55 / 56; pool union 61, 44.50; all evidence 70, 50.83);
    - `paper_stats.py` leaves `data/` and `paper/figures/data/` unchanged;
    - the draft builds (11 pages).
  - No number or claim changed. Layout fixes only:
    - title broken into three balanced lines;
    - Figure 2 legend left-aligned;
    - "ARC Prize" / "Kaggle" capitalised in the bibliography;
    - `\pdfoutput=1` on line 1.
    - The build now has no overfull and no underfull boxes.
  - Abstract shortened for the arXiv metadata limit: 1,900 characters (limit 1,920).
    - Points (i)-(iii) and their numbers are kept. No number is added; only "37 correct in all five runs" is dropped (still in Section 4.4).
    - `main.tex` has the same text.
  - NVARC citation filled: Sorokin & Puget, "NVARC solution to ARC-AGI-2 2025", https://github.com/1ytic/NVARC.
    - Source: the write-up PDF in that repo, which ARC Prize 2025 Technical Report (arXiv:2601.10904) cites as ref. [25].
    - Hamid confirmed the URL choice.
  - Code / data link: the team will make the repo public, so the URL stays and its TODO is removed.
    - On 2026-11-04 it returned 404 without login: check before submitting.
  - New `scripts/make_arxiv.py` builds `paper/arxiv/`:
    - `main.tex` without comments, `main.bbl` from this exact `refs.bib`, `refs.bib`, `figures/data/*.dat`;
    - `arxiv_source.tar.gz`;
    - a clean pdflatex-only test build from the extracted tarball: 11 pages, 0 errors, 0 undefined, 0 overfull / underfull;
    - checks against `METADATA.md`.
    - The clean build's PDF text is identical to that of `paper/main.pdf`.
  - pgfplots: arXiv uses TeX Live 2025 (default) or 2023, both with pgfplots 1.18.1. `compat=1.18` works, so there is no externalisation.
  - `paper/arxiv/METADATA.md` holds the title, abstract, comments line, cs.LG + cs.AI, the suggested CC BY 4.0 license, the endorsement note and the open items.
  - **Open:**
    - authors: Hamid said leave open; the `TODO(authors)` line stays, so the package is built with `--draft`;
    - make the repo public;
    - endorser / submitter: not decided;
    - Foysal's final review.
- **2026-11-04 (Hamid's AI, draft v1)**
  - Numbers reproduced: `scripts/score_all.py` gives exactly the expected output (41 runs; 64/172, 46.50; pools 55 and 56;
    pool union 61, 44.50; all evidence 70, 50.83).
    - On macOS it rewrites `data/run_scores.csv` and `data/pool_units.json` with LF endings and one row in another order
      (same content). Those changes were reverted (byte-exact rule).
  - New `scripts/paper_stats.py` computes every other paper number from the run logs and submissions. It writes
    `data/paper_numbers.json`, `data/lb_by_notebook.csv` and `paper/figures/data/*.dat`.
    - It reproduces the findings: top-k, the 48/33/22/14 classes, union growth 51 -> 62, core 37, pairwise 41-45.
    - New numbers:
      - R1 classes 47/32/22/15;
      - the submitted right set equals probmul rank <= 2 in both runs;
      - the placeholders fall in 11 tasks per run, all of which finished normally (no time-out, no exception);
      - 5961cc34_0 has 1 distinct grid of 16 (95.9 %);
      - union band over all 120 run orders;
      - LB table extracted to csv.
  - Correction: Experiment 18 `avg_sim` is the MEAN agreement over all candidates (not the closest candidate). Fixed in the
    findings file, section 3.
  - Code fact used in the paper: both rankers order the same distinct grids, so kgmon never changes the submitted pair. The
    "dual selection" is probmul top-2. The search keeps outputs with p > 0.2, so a view returns at most 4 and a test input
    at most 64 (observed max 27).
  - Step 4 and step 5 done:
    - `paper/main.tex` and `paper/refs.bib`, 10 pages; `paper/NUMBERS.md` maps every number to its source;
    - 4 figures drawn with pgfplots from the `.dat` tables;
    - compiles with `latexmk -pdf main.tex`.
  - Independent audit of every number and claim (agent, read-only):
    - All major numbers were confirmed.
    - Fixed:
      - the logs hold per-unit summaries, not every candidate;
      - 41 = 39 files from 20 notebooks + 2 printed sets (B is not one of the 34 notebooks);
      - kgmon top-2 recomputed: R2 +1.33, R1 -1.67;
      - Exp 10 is not a re-weighting of the shipped ranker (it scores each occurrence alone), so it moved to the pitfalls;
      - 142ca369 holds for R2 only;
      - "Pruned by max_score" comes from the dummy score 999;
      - the beam cap is 10 in R2 and 5 in R1 / A / D, and never binds;
      - the differences from the reproduction notebook (baseline 3, kgmon top 2, 1200 s, `hash()` seeds) are now stated;
      - the abstract claim is limited to the two logged runs.
    - New: R1 and R2 are the same program with identical seeds (240 lines) but differ (49 vs 45 units, pool union 61).
      This is run-to-run noise, and it shows that reruns add some correct candidates (stated in the limitations).
    - Search calls took at most 246.9 s, below the 540 s cap.
    - Draft v1b: 11 pages.
  - Step 3 is written as a limitation for now (no GPU run). If Hamid gives a GPU go, the result would replace the limitation.
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
3. [open, written as a limitation in draft v1] Decide: run the diversity / more-samples test, or state it as a limitation.
   - A GPU run needs the user's explicit go.
   - Candidate: v2 Experiment 34 (diversity penalty in `turbo_dfs`), and a run with more augmentations or samples, measuring pool growth.
4. [done 2026-11-04, draft v1] Paper outline, then the LaTeX draft (`paper/main.tex`). Every number must come from
   `scripts/score_all.py`, `scripts/paper_stats.py`, a data csv or a cited log line (`paper/NUMBERS.md`).
5. [done 2026-11-04] Figures (pgfplots inside `main.tex`, data in `paper/figures/data/`):
   - top-k coverage curve (v1, v2);
   - union growth vs number of runs;
   - where the answer is lost (generation classes: close / partial / far / no grid; selection);
   - unique-candidate histogram.
6. [in progress 2026-11-04: package built as a draft] arXiv package `paper/arxiv/` (`scripts/make_arxiv.py`, `METADATA.md`).
   - Done:
     - format;
     - short abstract (1,900 characters);
     - NVARC citation;
     - code-link TODO;
     - package + clean test build;
     - metadata sheet.
   - Open before submission (see `paper/arxiv/METADATA.md`, last section):
     - author list, affiliations and contact (`TODO(authors)` in `main.tex`);
     - make https://github.com/Foysal348/ARC-AGI-2 public and check the link;
     - decide who submits and requests the cs.LG / cs.AI endorsement (a human action);
     - rerun `python -B scripts/make_arxiv.py` without `--draft` and commit;
     - Foysal's final review and approval. Only then submit (never by the AI).
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
