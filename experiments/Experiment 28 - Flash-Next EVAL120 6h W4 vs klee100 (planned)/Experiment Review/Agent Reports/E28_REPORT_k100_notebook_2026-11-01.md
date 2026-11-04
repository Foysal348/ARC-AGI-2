# E28 K100 notebook agent: final report, verbatim

## E28 K100 notebook (klee100 EVAL120, 6 h): report

The notebook, its builder, the lib, the unit tests, the CPU kit builder, 4 kit folders, the GPU push folder and the audit zip are all done. Nothing was pushed and nothing ran on a GPU; there are no git operations.

The local unit run passes 35/35. The units run from inside the built units kit passes 34/34, with E4 skipped because it needs the real klee100 config and Merlin source, which the real kit checks instead. The Kaggle kits have not been run yet.

### What you need to act on
- **The scheduler module moved after it was declared final.** The coordinator gave `e28_sched_patch.py` as final at `1ff9943a…`. The file on disk changed at 13:24 to `cbf9ebc5…`. That version adds edit N4 ("no worker may idle while something fits": a non-blocking fit check so smaller attempts can backfill) and a new post sha `95e85004…`. Its API file still says `1ff9943a`.
  - My notebook embeds the current file, `cbf9ebc5`, and integrates N4.
  - The scheduler agent's own tests failed locally on this version: **29/30** (`test_backfill_b2_waits_c_starts`).
  - When the module is final, rerun `python build_e28_k100.py` then `python build_e28_k100_cpu_kit.py`. The builders apply any of N1–N4 by label.
- **Admission differs from the W4 notebook.** As instructed, I use the projected-peak gate (waits, never preempts) with 21 workers. At pool 479 / F 6 it admits about **12 attempts at 120k at once**, or 11 / 9 / 7 with 1 / 2 / 3 B2 re-runs running. K100Q ran "slots" (mean 16.5 running, 0 preemptions at 104k), and the W4 builder keeps "slots". This is one profile key (`admit_mode`) if you want them the same.
- **The prerequisite is already met.** The notebook reads the same tf-ref dataset version K100Q ran on, so there is nothing new to push. Next steps are the kits on esfersami50, then the GPU run (user go only).

### Notebook
- **File:** `D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Q27-XHIGH-E28_K100_EVAL120_6H.ipynb`
- **sha256** `7b57ab6ae5b662312e8df57eecaff92a92d4e61ae73ea360d09492ce802f0cd2`, **394,425 B** (< 400,000), compact JSON, 9 cells, LF.
- **Base check:** the build reproduces K100Q `4506a71f…` byte for byte, also checked against the K100Q notebook on disk. The import of the K100Q builder rewrites `k100q_ref27b.json` with identical bytes (checked).
- **GPU push folder:** `D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e28-k100-eval120-6h__esfersami50/` (id `esfersami50/q27-xhigh-e28-k100-eval120-6h`, the same models, datasets and docker image as K100Q, L4).

### Files (Build = `D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Build/`)

| File | sha256 | What it is |
|---|---|---|
| `build_e28_k100.py` | `3268e99cec7a4ead2f2b9363d522f6950d69c933e9f7585749f9e8c1764e8960` | notebook builder |
| `build_e28_k100_cpu_kit.py` | `178a41e4c0b317c1d9898669532a1aa611115addf89c38715e6d2cacf223c0d0` | kit builder |
| `e28k100_lib.py` | `68e5b0f9e5dc5f2b726c45ece162ca979911b8d4a03c10f33685a7887e781ab2` | new code, embedded packed |
| `e28k100_unit_tests.py` | `03cc99720dbb22c4cc87766c153b971a5aab0199f9c0b8cac99bae0fb74ba855` | unit tests |
| `e28k100_unit_tests_local_2026-11-01.log` | `434c0c25dc02208e7281eda54b750004fe1279602c1fd5660676fe708aed2929` | local unit run (35/35) |
| `e28k100_build_facts.json` | `e02b1f0880bfd2d82fcc5c8f31d456fae31c69e8e29bfe703687ea9edddde38f` | build facts |
| `e28k100_kit_facts.json` | `cc49238454e86b902659cbe50c1a21da998a4a10674e46fa5dda3c89160b5bc7` | kit facts |
| `e28k100_code_export.py` | `a00afd7c463fc4a8cc63e161da4c1e8d3a1a4fada2fec27c297e09c8a3f5f51d` | readable code; e21_cell7 unpacked |
| `e28k100_review_diff_vs_k100q.diff` | `8cc48f9c9edbf7b31174e331e94e618094b2a2518d7a227d98b27046164a9755` | +241 / −98 lines vs K100Q |
| `e28_sched_patch.py` (read only, not mine) | `cbf9ebc50ed519c58c23e22cbe45a2eea733c46e886a2316639c45c41a5a18e0` | the version embedded |

### Changes
Line numbers are in `build_e28_k100.py` and, for the notebook text, in `e28k100_code_export.py` (marked `# E28 K100` / `# E28 SCHED`).

- **Deadman (K1):** 24,900 s = HARD_STOP + 5 min (export :54). Comment updated (:50).
- **Commit default (K1):** `E28EVAL120` (:119).
- **Profile guard (K2):** `K100_PROFILES = ("E28EVAL120", "RERUN_PROD")` (:180).
- **E28 constants and the B2-only assert (K2, export :187–205):** T1 120,000, B2 200,000, server max-model-len 262,144, T1 context ceiling 131,072, 6-h window, I8 env. The B2-only assert: 200,000 − 4,096 ≤ 261,632. The original T1 asserts (≤ 120,000) are kept.
- **Shared levers (K2):** `E28_LEVERS` = gate 1.0, `nb5_attempt` 120k / 120k, the NB6 / E15B SUBMIT env (no fixed seed), projected admission, the workers rule, measured F, the pin ladder, #60490, `MERLIN_SHARE_DRAFT_WORKSPACE=1`, i8, plain order, the scheduler, the klee100 boot levers.
- **Profiles (K2, builder :207, export :251–257):**
  - `E28EVAL120`: eval, all 120 tasks, 415-min budget, margin 5 → HARD_STOP at 410 min. One phase `Q_E28_K100_EVAL120` with the fixed 21,600-s window; it is shortened, never skipped, if the boot is late.
  - `K100QUALITY` and its 30-task list are removed.
  - `RERUN_PROD` = NB6's (every test task, 12 h − 3 min, E18 ladder) plus the same levers.
- **Request timeout (K3b, builder :231–237, export :411):** `E12_DUCK_REQUEST_TIMEOUT` = 7,200 s for T1. B2 uses the module's scaling, 12,000 s.
- **`E21_VLLM_MAX_MODEL_LEN` = 262,144 (K3, export :452).**
- **No W4 / 27B reference tables (K4, :527).**
- **Packed modules (K5, :528–570):** one sha-pinned lzma + base85 literal holding `e28k100_lib.py` and `e28_sched_patch.py`, plus the E28 config record.
- **Scheduler apply (K6, builder :271, export :617–).** In CELL 3, after the manifest check and before `import arc2_duck`:
  - steps: apply → verify → `server_env()` into the notebook env;
  - harness-only override: T1 = `NB5_ATTEMPT_B0`, B2 = `E28H_B2_TOKENS`;
  - commit: any failure raises `NO_GO_E28_SCHED`;
  - rerun: any failure reverts the file and the original scheduler runs.
- **Projected-peak edits (K7, builder :309–329, export :770, :946):** the module's N1 / N2 / N4, plus one ceiling: an attempt's peak is capped at its Duck window + 512. That gives T1 131,072 / 38 blocks (= K100Q) and B2 216,384 / 59 blocks, the same in N1, N2 and N4.
- **`MAX_LEN` = 262,144 (K8, builder :361, export :1012).** The server and fake argv get `--max-model-len 262144`. The capacity plan uses the T1 ceiling 131,072 at all 3 calls (export :2771, :2795, :3203).
- **Workers (K9, builder :374, export :2777, :2792):** 21 by the R2 rule under projected admission, plus a capacity note in the records.
- **I8 (K10, builder :360–410, export :3559–3584, :3452, :1925):**
  - `DRAFT_HEAD=int8` goes into `build()` while I8 is on; tries are named `_i8`.
  - Drop rule: a try that fails with the INT8 head (launch / OOM / C-11 config mismatch before any phase) retries the **same pin** with the BF16 head, recorded as `I8_UNSUPPORTED_ON_K100` with the kind: `DRAFT_LM_HEAD_QUANT_REFUSED`, `OOM_WITH_INT8_HEAD` or `BOOT_FAILED_WITH_INT8_HEAD`.
  - NB4T's read-back runs on every try (recorded).
  - `e28_i8_static()` runs before the first boot; an UNSUPPORTED_STATIC verdict drops I8 before booting.
- **No ORDER2 (K11, export :3648):** N3 plus `plain_order`.
- **Task list for the per-task table (K12, export :3126):** EVAL120 uses the phase's `TASK_IDS`.
- **Runner (K13, export :4821–4833):** the I8 setup and static check, and the 6-h window label.
- **Report (K14, builder :474–529, export :4942–5023):**
  - `e28_k100_report.json`: score (run report vs recomputed, must agree), tok/s, preemptions, prefix hit, the per-attempt table with phase A / B1 / B2 / C, the B2 rows, I8 / Q-defer / boot facts, and `SCHED_LAST["e28"]` (phase counts, classes, slots, rows).
  - It is in the bundle, and the K100Q quality report is removed.
  - The existing E15B summary counts B2 as "attempt 3+" with the b1 budget; the new report uses each record's own budget.
- **Size:** e21_cell7 (E15B CELL 7) is packed as a sha-pinned literal using the E27 NBDF technique (builder :421–436).
- **Header (K15):** new markdown header (`MD0`).
- **Kept unchanged:** CELL 0 = E18 R2, Q-defer, gates, the pin ladder, relaunch, watchdog, deadman thread and full saved responses (B re-runs get their own `b2` directories).

### Tests
- **Local units (Python 3.14): 35/35 PASS.**
  - A, provenance: facts = notebook, CELL 0 = E18 R2, packed cell7 / modules shas, every changed hunk vs K100Q is marked.
  - B, profiles: both profile rows, the time plan (HARD_STOP 410, deadman 415), the request timeouts.
  - C, edits: all edits present.
  - D, numbers: N1 / N2 / N4 give 131,072 / 38 and 216,384 / 59; the N4 fit check works on a Reservation-shaped object.
  - E, lib:
    - the I8 read-back, failure reasons and static check;
    - on the real klee100 config plus the real Merlin source: SUPPORTED_STATIC (head not tied, shard 124160 × 2560 Marlin-aligned);
    - the report on K100Q's real GPU output: VALID, 31/38 queries, 23/30 tasks solved;
    - the report reads the `SCHED_LAST["e28"]` section.
  - F, scheduler: apply / revert / pre-sha refusal on a copy of the payload's real `arc2_duck.py`.
- **From the units kit's own files: 34/34**, only E4 skipped.

### CPU kits (esfersami50, private, CPU, image pinned; not pushed)
Folders are under `D:/ARC AGI 2/_scratch/push_e25/`. The kits wrap the K100Q kits (that builder's source is exec'd with exact-once edits; it is not written).

| Kit | sha256 | Size |
|---|---|---|
| `q27-e28-k100-kit-units__esfersami50` | `fc1809a0…` | 276,802 B |
| `q27-e28-k100-kit-e2e-rerun__esfersami50` | `f8cffeef…` | 210,507 B |
| `q27-e28-k100-kit-e2e-commit__esfersami50` | `429940d8…` | 210,498 B |
| `q27-e28-k100-kit-real__esfersami50` | `54867d90…` | 269,025 B |

- **units:** the unit tests on Py3.12, the dataset version check, the payload and pin checks, and the real `arc2_duck.py` from the mounted payload.
- **e2e-rerun:**
  - rerun mode: the fake 6-task hidden set, the scheduler applied, plain order, I8 on.
  - retry mode, I8 path: try 1 (rung 1, INT8 head) OOMs → `I8_UNSUPPORTED_ON_K100` (`OOM_WITH_INT8_HEAD`); try 2 (same rung, BF16 head) OOMs; try 3 (rung 2) boots.
- **e2e-commit:** all 120 tasks on a **fake 3,000-s deadline**, with the profile still baked at 415 min. Checks: the window shortened, scoring, the report VALID with scheduler rows, the deadman armed and not fired. The kit's copy of the fake server is edited to accept `--max-model-len 262144` (the original refused anything but 131072).
- **real:** K100Q's real checks plus:
  - 262,144 in every argv and in vLLM's argparse;
  - rung 1 of both profiles built through Merlin's real `build()` with `"draft_lm_head_quant":"int8"`;
  - the scheduler applied in both namespaces, reverted to the exact manifest bytes, then re-applied;
  - `e28_i8_static` on the real merged dir and the installed tree;
  - Merlin's INT8 quantizer (FakeL4 platform) on 8,192 real rows of klee100's lm_head.
- **Kit lessons applied:** `dataset-metadata.json` is not expected on the mount; no probe is named after a stdlib module; JSON is compared by parsing; vLLM is imported on CPU through FakeL4.

### Audit zip
`D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Review/Audit Zips/E28_K100_audit_FLAT_2026-11-01.zip` (26 files, flat, 1,130,081 B, sha256 `d504e545…`). It includes `00_GPT_PROMPT.md`, which asks for GO / NO-GO plus blockers.

### Open risks
1. **I8 on klee100 has never run on a GPU.** It passes the static check and the CPU quantizer probe only. On W4 the INT8 head added about 310 MiB per PP1 card and OOMed at the base pin. At the 7.93 GB pin an OOM is plausible; it then costs one extra boot of about 13 min before the BF16 try.
2. **200k context has never run on Flash-Next** (RC2 validated 120k). B2 attempts are recorded per attempt.
3. **The scheduler module is not settled** (see the first section).
4. **Fewer attempts run at once than in K100Q** under the projected gate (see the first section), and phase A of 120 tasks alone is about 4 h at 385 tok/s.
5. **T1 attempts end at 129,024 tokens** with CONTEXT_EXHAUSTED: 131,072 minus the guard's 2,048 margin, the same rule as in K100Q.
6. **The notebook has 5.5 KB of room left** under the 400 kB limit.
