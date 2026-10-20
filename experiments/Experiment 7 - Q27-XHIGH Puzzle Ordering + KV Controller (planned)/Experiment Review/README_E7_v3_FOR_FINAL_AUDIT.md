# Experiment 7 v3: response to the two final audits (2026-10-12)

**Request: a deep, unrestricted re-audit.** Claude's claims below are to be verified, not trusted. Nothing has run on GPU; nothing was pushed or submitted.

**User rules:**
- at least 49,152 continuous xhigh thinking + 8,192 answer per test input;
- v1.4.4 (LB 26.94) is the base;
- dual mode;
- **a valid full-schema `submission.json` must always be produced**;
- no push, run or submission without approval.

v3 is built deterministically by `build/build_e7b.py` from GPT implementation B (`build/source/…CONTROLLER.ipynb`, sha256 `d10475e4…`). Three successive builds give identical bytes.

## 1. Audit findings, and what v3 did

| Finding (audit) | Verified by Claude? | v3 change | Test |
|---|---|---|---|
| Shadow decisions counted on every scheduler loop instead of once per telemetry snapshot (audit 1 F2 / audit 2 F1) | **yes**: `decide()` runs after every engine step | count at most once per `last_poll_at`; nothing counted before the first poll; an invalid snapshot's rejection is counted once | T4 (1,000 calls give 1; a new poll gives 2; invalid gives 1). T9 integrated: 5 shadow decisions ≤ 11 polls ≪ 934 loop steps |
| `start_phase2()` mutates host state and counters before `engine.add_request()` (audit 1 F3) | **yes**: inherited from v1.4.4 | transactional: build the prefix locally; mutate `st` / `SCHED` only after the engine accepts; abort the request on failure; the no-room path behaves exactly as v1.4.4 | T5: success path identical to v1.4.4 (calls, prefix, seed, cap, state, counters) for forced-close + EOS and for closed + cut; an injected failure leaves `st` and the counters unchanged and aborts `u#0\|2` |
| Final validator discards a valid attempt when the other attempt is invalid, and runs after the ZIP (audit 2 F2 / audit 1 F4) | **yes** | `ensure_final_submission()` is attempt-wise, per test input, and **runs first in cell 8** (before `run_report.json` and `Q27X_BUNDLE.zip`); `report["submission_audit"]` holds the counts; an idempotent re-check stays at the end | T6 (valid attempt_1 kept, valid attempt_2 kept, missing task and garbage file handled, unknown keys dropped, idempotent, ordering of calls asserted). Harness: **ZIP `submission.json` == disk file in all 15 runs** |
| No matched PROXY48 controls (audit 1 F5 / audit 2 F3) | agreed | new **A0_PROXY48_BASE16** (original order, cap 16, KV off) and **A1_PROXY48_ORDER16** (Codex order, cap 16, KV off) | T1: A0 and A1 cells are identical except the `Q27X_ORDERING` default (and the banner) |
| Adaptive 17–18 never exercised end to end (audit 1 F1 / audit 2 F5) | **yes**: on the harness the engine is not the preferred recipe, so the mode falls back to shadow | new integrated tests that run the **real cell 7 code** with the preferred engine (`attention_backend_resolved` = `"AttentionBackendEnum.FLASHINFER"`, cap 20), labelled metrics with `running` = the engine's live count, and a timed fake engine | **T7:** adaptive mode active; max_active = engine max = **18**; 2 controller-authorised extras; 26/26 finished; no duplicate request IDs. **T8:** a preemption after the first extra freezes expansion (max 17). **T9:** B1 shadow never exceeds 16 |
| The adaptive pilot stays on in a hidden rerun (audit 1 F6) | agreed | **hidden-rerun guard:** in `RERUN`, unless `E7_HIDDEN_APPROVED` (env-only, so effectively False on Kaggle), every E7 feature is OFF (original order, KV off, cap 16 = v1.4.4 scheduling), applied before `CONFIG` and the engine | harness `rerun` × 5 notebooks: `hidden_guard_applied` true, ordering off, KV off, cap 16, and **`submission.json` identical to v1.4.4**. With `Q27X_E7_HIDDEN_APPROVED=1`, B1 rerun runs with ordering on, shadow, cap 20 and still gives the identical submission |
| `not_admitted` wrong after a scheduler exception (audit 1 F7) | **yes** | `finally`: `not_admitted = planned − admitted` when `SCHED_ERROR` | T10 (engine failure at step 40: not_admitted = 14 = 30 − 16) |
| CPU tests not runnable from the zip (audit 2 F4) | **yes** | a new portable suite `tests/test_e7_v3.py` with bundled `tests/fixtures/` (base notebook, Codex donor notebook, real vLLM metrics, PROXY48 manifest) | `python tests/test_e7_v3.py` → **ALL 10 PASS** |
| PROXY48 manifest unverifiable (audit 2) | — | `tests/fixtures/proxy48_manifest.json`, generated from the real evaluation file: 48 tasks, **56 test outputs**, all present; the source EDA cell is named | T1 asserts the notebook list equals the manifest |
| "Guarded fixed" is not a true fixed-18 | acknowledged | unchanged; it stays labelled as guarded (it needs KV < 0.62). A true fixed-18 arm is out of scope until GPU safety data exists | — |
| Fake harness uses THINK 3,000 / ANSWER 40,000 (audit 2) | acknowledged | the harness overrides budgets only for speed. **The notebooks themselves keep 49,152 / 8,192** (T1 asserts the default lines) | T1 |
| Deadline drain can force-close before 49,152 (audit 2 §11) | acknowledged | inherited v1.4.4 behaviour (part of the LB base); the records keep `mode` and `think_tokens`, so drained units are distinguishable | — |

## 2. The five notebooks (Codex ordering + KV controller code is identical; only the defaults differ)

| Notebook | sha256 | Eval default | Order | Engine cap | KV | Purpose |
|---|---|---|---|---|---|---|
| `Q27-XHIGH-v1_4_4_E7_A0_PROXY48_BASE16.ipynb` | `56edc264…` | PROXY48 | original | 16 | off | G0 matched baseline |
| `Q27-XHIGH-v1_4_4_E7_A1_PROXY48_ORDER16.ipynb` | `1ef54ec5…` | PROXY48 | Codex | 16 | off | G1 ordering only |
| `Q27-XHIGH-v1_4_4_E7_B1_PROBE.ipynb` | `0224ccef…` | PROXY48 | Codex | 20 | shadow | G2 engine-cap and shadow measurement |
| `Q27-XHIGH-v1_4_4_E7_PILOT_ADAPT18.ipynb` | `9b6af834…` | PROXY48 | Codex | 20 | adaptive ≤ 18 | G3, only after G2 |
| `Q27-XHIGH-v1_4_4_E7_CONTROLLER_v3.ipynb` | `bbcf8564…` | EVAL12 | Codex | 16 | shadow | development |

In a hidden rerun, all five behave like v1.4.4 (the guard) unless approval is set explicitly.

**Proposed GPU order** (each step needs the user's approval; about 2.6–2.7 h each):
- G0 = A0 and G1 = A1 together give the ordering effect.
- G2 = B1 (with A1 as its matched comparison) gives the engine-cap overhead and the shadow opportunities.
- G3 = PILOT_ADAPT18 only if G2 passes:
  - the preferred recipe runs at cap 20;
  - the KV pool is no more than about 10 % below 1,353,600;
  - no preemptions.

The GPU budget is limited (about 12.7 h on foysal until 10-17). If needed, G0 can be skipped by comparing A1 with A0 later, or B1 can run first as a diagnostic (the auditors allowed this once the shadow counter was fixed).

## 3. Test evidence (CPU only)

| Suite | Result |
|---|---|
| `tests/test_e7_v3.py` (portable, bundled fixtures) | **10/10 PASS** (`validation/test_e7_v3_output.txt`) |
| GPT B's `test_q27_e7_cpu.py` against each v3 notebook | **5 × ALL PASS** |
| Experiment 1 harness (real Qwen tokenizer + chat template, fake vLLM step engine) `mixed` / `drain` / `rerun` × 5 notebooks | **15/15 SCENARIO PASS**, 0 tracebacks, `repaired_attempts` 0, **ZIP == disk submission in all 15**, rerun == v1.4.4 in all 5 |
| PROXY48 default runs, A0 / A1 / B1 | 48 tasks / 56 queries each. A0 admits alphabetically (`0934a4d8…`); A1 and B1 admit identically in Codex order (`36a08778#0, 36a08778#1, 1818057f…`; tier 1 first, tier 3 last). Records carry `t_admit_min` and `t_done_min`. B1 shadow counted once per poll |
| pyflakes over joined cells | only the base's unused imports |

## 4. Not provable on CPU (GPU gates)

- vLLM 0.19 startup with `max_num_seqs` 20 under fp8 KV + MTP ×3 + prefix cache;
- the real KV pool;
- preemption behaviour at 17–18;
- real throughput;
- accuracy and run-to-run variance (EVAL12 identical-code runs gave 2, 2, 3 and 1 hits).

## 5. Files

| Path | What |
|---|---|
| `notebooks/` | the five v3 notebooks |
| `build/build_e7b.py`, `build/source/` | the deterministic builder and its source |
| `build/tests/test_e7_v3.py`, `build/tests/fixtures/` | the portable suite and fixtures (base, Codex donor, real metrics, PROXY48 manifest) |
| `validation/` | test output, harness logs (15 + PROXY48 + approved-rerun), GPT B's suite outputs |
| `gpt_impl_B/` | B's original README, patch and test |
| `previous_audit/` | the v2 audit README (for diffing claims) |
