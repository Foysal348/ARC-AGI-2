# Experiment 7 v2: Codex ordering + guarded KV admission, the final pre-GPU audit (2026-10-12)

**Request: a deep, unrestricted audit.** Everything below is Claude's work and claims; verify or refute any of it. Nothing has run on GPU; nothing was pushed or submitted.

**Standing user rules:**
- never less than 49,152 continuous xhigh thinking + 8,192 answer per test input;
- v1.4.4 (LB 26.94) is the base;
- dual mode: the commit run evaluates and scores, the competition rerun runs every hidden task;
- **a valid full-schema `submission.json` must always be produced**;
- no push, run or submission without the user's approval.

## 1. What this is

**Two independent GPT implementations of Experiment 7 were delivered.** Claude audited both.

| | Implementation A (`Q27_XHIGH_v1_4_4_E7_CODEX_ORDERING_KV.ipynb`) | **Implementation B** (`…E7_CODEX_ORDER_KV_CONTROLLER/COMBINED_PILOT.ipynb`) |
|---|---|---|
| Adaptive mode on a real GPU | **never activates (bug)**: it requires `attention_backend_resolved` ∈ {"FLASHINFER", "NONE"}, but vLLM 0.19 reports `"AttentionBackendEnum.FLASHINFER"` (run 7 `run_report.json`), so adaptive silently becomes baseline | OK: substring match; every "preferred recipe" condition is satisfied by the real run 7 engine values |
| Defaults | all off (= v1.4.4) | ordering on + shadow; the hidden `submission.json` is still identical to v1.4.4 (verified) |
| Chosen | — | **yes** |

**Gaps that Claude found in B, now fixed in v2:**
1. There was no deadline-constrained evaluation workload (only EVAL12, where all 22 finish, so ordering can never show).
2. With engine cap 16, shadow mode never reaches its decision code (`decide()` returns `engine_cap` first).
3. Kaggle batch runs cannot set environment variables, so each arm needs its own notebook.

## 2. v2 changes (deterministic builder `build/build_e7b.py`; source = implementation B CONTROLLER, sha256 `d10475e4…`)

Every replacement is asserted to hit exactly once.
1. **`PROXY48` evaluation profile:**
   - the EDA MAIN_DEBUG_48 list: 48 tasks, 56 test outputs, mirror gap 9.41 pp;
   - the same 3 h budget and admit/drain fractions as EVAL12 (admit until 140 min, drain 147, hard stop 155);
   - the workload is **deadline-constrained** (not every unit is expected to finish), so ordering effects become visible;
   - every ID is asserted to exist in the evaluation file.
2. **Three notebooks with per-variant defaults** (all with Codex ordering on):

| Notebook | Eval default | Engine cap | KV mode | Purpose |
|---|---|---|---|---|
| `Q27-XHIGH-v1_4_4_E7_CONTROLLER_v2.ipynb` | EVAL12 | 16 | shadow | safe development version |
| **`Q27-XHIGH-v1_4_4_E7_B1_PROBE.ipynb`** | **PROXY48** | **20** | **shadow** | **next GPU run (proposed)**: the KV pool at cap 20 and real-metric "would admit" decisions; at most 16 active |
| `Q27-XHIGH-v1_4_4_E7_PILOT_ADAPT18.ipynb` | PROXY48 | 20 | adaptive (≤ 18) | only after B1 passes |

3. **Per-unit timing in `records.jsonl`:** `t_admit_min` and `t_done_min`, plus B's `e7_admit_rank` and `e7_priority_tier`. These support the "which unit finished when, by tier" ordering analysis.
4. **Final submission guarantee** (appended after the report in the last cell):
   - re-read `submission.json` and validate it against the test challenges (every task, every test input, two valid grids);
   - anything missing or invalid is replaced by the test-input fallback;
   - print `SUBMISSION OK | tasks | outputs | answered | replaced`.

**Unchanged:** prompt, sampling, seeds, MTP ×3, fp8 KV, prefix cache, the 49,152 + 8,192 allowances, the host-forced close, the parser, and the rerun deadlines (admit 11:34, drain 11:36, hard stop 11:44, end 11:50). B already replaced the latent `phase1_cap` budget reduction with an explicit `oversized_prompt_full_budget_required` skip.

## 3. Claude's verification (CPU only)

| Check | Result |
|---|---|
| Builder determinism | two builds give identical bytes (sha256: CONTROLLER_v2 `cad72c01…`, B1_PROBE `9316dcd5…`, PILOT_ADAPT18 `c56afa2a…`) |
| GPT B's own CPU suite (`test_q27_e7_cpu.py`: 31 real vLLM snapshots, controller gates, 20→16 fallback, transactional admit, …) against each v2 notebook | **3 × ALL PASS** |
| pyflakes over all joined code cells | only the base's unused imports |
| Experiment 1 harness (real Qwen tokenizer + chat template, fake vLLM step engine): `mixed`, `drain`, `rerun` × 3 notebooks | **9/9 SCENARIO PASS**, 0 tracebacks; `SUBMISSION OK 240 tasks / 259 outputs` in every run |
| **Hidden-mode equivalence:** `rerun` `submission.json` against the base v1.4.4 on the same harness | **identical for all three notebooks** |
| B1_PROBE with its PROXY48 default (harness `mixed`) | 48 tasks / 56 queries; the report shows ordering on, kv_mode shadow, engine cap 20, baseline 16; shadow decisions are reached (the fake metrics carry no labels, so they are `missing_telemetry`; GPT B's suite parses the real labelled snapshots); records carry rank, tier, `t_admit_min` and `t_done_min` |
| Codex ordering = verbatim NVARC `get_puzzle_priority` | identical (also confirmed earlier against `resources/Puzzle Ordering (NVARC 32.22 ref)/puzzle_order.py`) |

## 4. Proposed GPU sequence (each run needs the user's approval)

1. **B1_PROBE on PROXY48**, about 2.6–2.7 h. It reports:
   - the KV pool at engine cap 20 compared with 1,353,600 at 16;
   - startup time;
   - real-metric shadow decisions (`shadow_would_admit` / `shadow_reject_*`);
   - the ordering completion profile (`t_done_min` by tier);
   - the primary score.
   - **NO-GO for the pilot if:** the pool shrinks by more than about 10 %, the engine falls back to 16, or there are preemptions.
2. **PILOT_ADAPT18 on PROXY48**, only if B1 passes. It is compared with B1 (the same workload and deadline) on units finished, correct outputs, peak KV and preemptions.
3. A hidden submission only after both runs, with the user's explicit approval.

## 5. Known limits and open questions for the auditor

- The thresholds (extra admission below 0.62, projected ceiling 0.82, emergency 0.90) are uncalibrated. In "guarded fixed" mode extra admission still needs KV < 0.62, so it is not a true fixed-18 comparator.
- Same-code run-to-run variance is large (EVAL12: 2, 2, 3, 1 hits), so primary scores are sanity checks, not decision metrics.
- PROXY48's mirror target comes from the placeholder visible test file (training-like); the hidden set is easier than evaluation but its exact profile is unknown.
- Is a 3 h PROXY48 budget the right deadline, or should B1 run without a deadline to get the full KV curve?

## 6. Files in this zip

| Path | What |
|---|---|
| `notebooks/` | the three v2 notebooks |
| `build/build_e7b.py`, `build/source/…CONTROLLER.ipynb` | the deterministic builder and its source (implementation B) |
| `gpt_impl_B/` | B's README, the source patch, `test_q27_e7_cpu.py`, the donor function |
| `gpt_impl_A_audit.md` | Claude's audit of implementation A (the backend bug) |
| `validation/` | harness outputs (9 scenarios + PROXY48), B's suite outputs for each v2 notebook |
| `context/` | the Experiment 7 plan README and the Experiment 6 GPT Outputs summary (both reviewers' decisions) |
| `base/Q27-XHIGH-v1_4_4.ipynb` | the frozen LB-26.94 base |
