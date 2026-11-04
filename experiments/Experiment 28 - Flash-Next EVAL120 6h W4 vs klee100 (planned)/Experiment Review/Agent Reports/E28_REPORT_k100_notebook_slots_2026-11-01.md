# E28 K100 notebook agent: slots change report, verbatim

## E28 K100: the slots change is in; waiting for the final scheduler sha

Admission is now "slots" and the full build passes, but it is an interim build. It embeds the scheduler file as it is on disk now (`47ab9874…`), which is still being changed. When you send the final sha, I rebuild with one command. Nothing was pushed and no GPU was used.

### Interim results
- **Notebook:** `D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Q27-XHIGH-E28_K100_EVAL120_6H.ipynb`
  - sha256 `c839ef384efe876b193881b520e517afbb5bd20948b23045a23722f8fab3a3be`, **395,820 B**.
  - It embeds `e28_sched_patch.py` `47ab9874a9f3a83f2c9a0fbf9b674bd260e4d3b34f3076b0b9601b5bdb5382ef`. The scheduler file changed twice since you called it final: `1ff9943a` → `cbf9ebc5` → `47ab9874`.
  - The GPU push folder `…/_scratch/push_e25/q27-xhigh-e28-k100-eval120-6h__esfersami50/` holds the same notebook.
- **Units:** 37/37 PASS locally. From the units kit's own embedded files: 36/36 PASS (E4 is skipped there; the real kit covers it).
- **Kit folders** (`D:/ARC AGI 2/_scratch/push_e25/`, esfersami50, CPU, private, not pushed):

| Kit | sha256 | Size |
|---|---|---|
| units | `4bdb117f…` | 280,802 B |
| e2e-rerun | `c9937756…` | 211,657 B |
| e2e-commit | `fefd08c9…` | 211,648 B |
| real | `9452e295…` | 270,155 B |

- **Audit zip:** `…/Experiment Review/Audit Zips/E28_K100_audit_FLAT_2026-11-01.zip`, 27 files, flat, 1,143,720 B, sha256 `c8156887…`. `00_GPT_PROMPT.md` is updated for slots.

### What changed
- **Admission is "slots"** in both `E28EVAL120` and `RERUN_PROD`, with 21 workers by the R2 rule (export :203). Builder comments, docstring, header and the facts' `decisions.admission` now say so.
- **The scheduler edits stay in, because they are needed or harmless in slots:**
  - **N1** sets the per-request token clamp from each attempt's own peak (C-9). B2 needs this to reach 216,384 tokens; without it B2 is capped at 131k.
  - **N2** now only feeds the "need above the whole pool" check (C-7); each attempt still takes one slot.
  - **N4** takes one slot per attempt without blocking, consistent with the normal acquire. New unit check D5 proves it: one slot for a T1 or a B2 attempt, refused only when all 21 are held.
  - The 131,072 / 262,144 ceiling stays on all three edits.
  - The projected-capacity note is recorded only.
- **Preemptions** (`e28k100_lib.py`; the notebook's CELL 9 passes `phase_<name>/nb5_sampler.jsonl`):
  - Run level: the server counter and `preemption_intervals` / `preemptions_sampler` from the 15-s sampler.
  - Per attempt: `preempt_while_running` = preemptions counted while that attempt ran. vLLM does not name the preempted request, so this is an upper bound for that attempt.
  - Also `attempts_with_preemption`, the same field in the B2 rows, the markdown summary line, and the note "a recompute, not a failure".
  - The report also reads the scheduler's new no-idle fields when present (`backfill_starts`, `forced_admits`, `fit_checks`, `fit_fails`, `fit_errors`, `admission_checks`, `idle`).
- **Tests:**
  - B1 / B2 now expect slots.
  - D5 is new (slots).
  - E9 is new: the preemption flag, ignoring a counter reset after a relaunch, and None without sampler data.
  - E5 on K100Q's real GPU output now uses its real sampler: 0 preemptions, every attempt flagged 0, still VALID with 31/38 queries.
  - The units-kit fixture now carries the sampler rows.
- **New helper `Build/e28k100_refresh.py`.** One command runs, in order:
  1. rebuild the notebook;
  2. local units, writing the log;
  3. rebuild the kits;
  4. units from the units kit's files;
  5. rewrite the review diff vs K100Q;
  6. rewrite the audit zip.

  The prompt source is `Build/e28k100_00_GPT_PROMPT.md`.

### Current file shas (Build)

| File | sha256 |
|---|---|
| `build_e28_k100.py` | `d0141021df94fd14d7e92d85bc345aefe086be63102b1d80a7ba47a26dec5830` |
| `build_e28_k100_cpu_kit.py` | `178a41e4c0b317c1d9898669532a1aa611115addf89c38715e6d2cacf223c0d0` |
| `e28k100_lib.py` | `b82c8a79552ae0afac7ee88f46f9071e872fb701e186390ebb4f48f35b872bc6` |
| `e28k100_unit_tests.py` | `b25abc59f5bb43304f052c7cb623fcb184cdeba157b1473480b09b0f95195d3e` |
| `e28k100_refresh.py` | `0bddb4aba1a6457721a678566bc571413d3c0850a9a4bfa19654554791f771e2` |
| `e28k100_00_GPT_PROMPT.md` | `5d7c8815d9fc95c484ae919d33583dfbce432fe49a5a6da981ba66f266d970eb` |
| `e28k100_build_facts.json` | `1da96442c255179d830747c5f5421190b5e0e21f28718418642db04bdd07fc7e` |
| `e28k100_kit_facts.json` | `9ff25784fe093682cd7d9d6c01e9777948d2c9853c805d02c4239f44e11bf1d0` |
| `e28k100_code_export.py` | `62bdda3a8ab345aef004ec3165e00d6dac93f08ca09d576572e8244cfd4d9cec` |
| `e28k100_review_diff_vs_k100q.diff` | `7feb0f450b246ce0d1496ca115ebaa49422ba27a905552dd115d267b2488abbc` |
| `e28k100_unit_tests_local_2026-11-01.log` | `bbb7c43784130232e91b2d34f63bcdda703a2c0a2aba40d72c95fa560be78402` |

### When the final scheduler sha arrives
Resume me, or run `python e28k100_refresh.py` in Build. It prints the new notebook sha, the scheduler sha it embedded, both unit-test totals, the four kit shas and the zip sha, and exits 1 on any failure.

The builder applies any of the scheduler's N1–N4 edits by label and keeps the ceiling on N1 / N2 / N4. The kit builder refuses to build if the Build files differ from what the notebook embeds.

### Open risks
1. **Size margin is 4,180 B** under the 400 kB limit. If the final scheduler module is much larger, the build stops with a size error. The fix ready for that case is deleting the unused NB5-era profile entries in CELL 1 (about 4 KB). It is not applied.
2. **KV pressure under slots:** with 120k attempts and 200k B2 re-runs, the engine may preempt (recompute). These are counted and flagged per attempt; K100Q had 0 at 104k with KV max 99.4 %.
3. **I8 on klee100 and 200k-context correctness are both still untested on GPU.**
