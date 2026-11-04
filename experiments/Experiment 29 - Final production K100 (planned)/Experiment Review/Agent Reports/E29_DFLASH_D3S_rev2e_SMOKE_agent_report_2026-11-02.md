## E29_DFLASH_D3S rev 2e: the GPU smoke is built (nothing pushed)

The smoke notebook is the production notebook with exactly one line changed. Production (new sha) and smoke push folders are written, the 3 kits are rebuilt, and every test passes: adversarial `--strict` 58/58 plus all local tests.

DEFAULT, MTP_V64 and ALL_OFF were rebuilt and are byte-identical to the audited builds (asserted against the on-disk notebook and push copies): `5c74abf8…`, `13419f69…`, `d7c39120…`.

**The payload dataset is unchanged since rev 2d**, so nothing new needs uploading. The rev 2d version must be the one mounted.

### Notebooks
**Production**
- `D:\ARC AGI 2\experiments\Experiment 29 - Final production K100 (planned)\Experiment Notebooks\Q27-XHIGH-E29_DFLASH_D3S.ipynb`, also in `D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e29-dflash-d3s__foysalemonshanto/`
- sha `d3aab2e74f63172f6af79183762363bd78c5279654293b889bb805a99999ec34`, 396,385 B (3.6 kB under 400 kB). It supersedes 2d `80d4aeff`.
- The default profile is still E28EVAL120. The E29SMOKE row is present but not selected.

**Smoke**
- `...\Experiment Notebooks\Q27-XHIGH-E29_DFLASH_D3S_SMOKE.ipynb`, also in `D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e29-dflash-d3s-smoke__foysalemonshanto/` (kernel `foysalemonshanto/q27-xhigh-e29-dflash-d3s-smoke`, same sources, models, image and NvidiaL4).
- sha `4b6bfd631716f1a360eabe61a19782a4928b5a6e40ec7117511a0ec9436b06f1`, 396,383 B.
- The builder asserts that exactly one line differs:
  - production: `E29_PROFILE = "E28EVAL120"`
  - smoke: `E29_PROFILE = "E29SMOKE"`
- That line is CELL 1 line 3, a plain editable literal.

**How the profile line drives the run** (needed because the deadman is armed before the existing profile line):
- `NB5_DEFAULT_PROFILE` follows `E29_PROFILE`.
- The deadman is 70 min for the smoke; production stays at 24,900 s.
- `K100_PROFILES` now includes E29SMOKE.
- `E29_SMOKE = {tasks, window_min 40, budget_min 70}` sits on CELL 1 line 4, identical in both notebooks.

### The 12 tasks (plain sorted)
| Group | Tasks |
|---|---|
| E28 K100 solved, single test, short first attempt (3.9k–9.3k tokens) | 1818057f, 2ba387bc, 45a5af55, bf45cf4b |
| K100 failed | 446ef5d2 (2 tests), b6f77b65 (3 tests), 7b5033c1, 800d221b (both passed train, wrong on test) |
| Mixed | 31f7f899, d59b0160 (solved), 409aa875 (failed), a47bf94d (context exhausted, so it gets a B2 re-run) |

The short solved tasks give phase-C items early, so those go to DIRECT. Later C items take the Duck C route, so both routes run.

### Smoke profile and timeline
- The E29SMOKE row equals the production row except for four values: budget 70 min, the 12-task list, the fixed 40-min window, and n_tasks. Everything else is production: DFlash2 D3S, seam + GRD, EMBED_HOST from the first try, QSA 128/64 with the 4-rank gate, the 130k first-prefill probe and worker-free gate, the OOM / XID ladder, the MTP top-rung fallback, 24 workers, DIRECT (104,000 / 8,192 / T 1.0 / drain 26), LIVE, effective_algorithm, submission_check, the ledger.
- Timeline:
  - boot about 15 min (each extra D rung about 4–5 min warm);
  - the fixed 40-min window;
  - about 10 min of reports;
  - hard stop at 65 min, deadman at 70 min.
- A boot that ends by minute 21 still gets the full 40-min window (asserted at build time and at runtime); a later boot shortens the window rather than skipping it.
- DIRECT is only admitted for C items starting in the first ~13 min of the window. The thinking cap stays 104,000; the 26-min drain is a time cut, so early DIRECT answers will mostly be forced closes.

### `[SMOKE SUMMARY]` (production prints the same block as `[E29 RUN SUMMARY]`), plus JSON
Record only, no new gate. It shows:
- effective_algorithm;
- DFlash engaged (booted try, read-back, GRD), EMBED_HOST, the QSA label, the seam state;
- the F2 probe per try with per-worker and minimum free MiB;
- every try and status, drops, relaunch plans, GPU device faults;
- tok/s per 15 min and over the window (plus solver tok/s), AL, preemptions, prefix hit %;
- attempts started / done per phase;
- DIRECT started / records / no_record / errors, Duck-C late routes and why, DIRECT modes (forced_close / natural …), attempt_2 taken from DIRECT;
- submission_check ok;
- eval hits / task-norm on the 12;
- host cgroup peak / max (the ledger now keeps the peak).

### Tests
- Adversarial `python e29_rev2d_adversarial_tests.py --strict`: 58/58 PASS.
- Retargeted E28 units: 44/44. I relaxed B3 to also accept E29SMOKE in the profile list, and the CELL 1 test exec now reads the `E29_SMOKE` line.
- e29 kit-lib: 44/44. New:
  - R29: the smoke row differs from production only in those 4 keys; HARD_STOP 65 / 410 and deadman 70 / 415 min per profile; the smoke sha equals the production bytes with the one line replaced.
  - R30: run_summary / summary_text on a fake run; the report-cell wiring.
  - R28 now also checks the host peak.
- SCHED's DIRECT tests: 20/20.
- From the kits' own files in an empty folder, Py 3.12 and 3.14: units 44/44 + 44/44 with `python -I`, direct 20/20.

### Kits (rebuilt for the new production sha, not pushed)
| Kit | Folder (`_scratch/push_e25/`) | sha | Size |
|---|---|---|---|
| units | `q27-e29-kit-units-dflash-d3s__foysalemonshanto` | d429b84c | 377,275 B |
| real | `q27-e29-kit-real-dflash-d3s__foysalemonshanto` | ad71f8af | 363,433 B |
| direct | `q27-e29-kit-direct__foysalemonshanto` | f51640fb | 84,724 B |

### Files changed
All in `D:\ARC AGI 2\experiments\Experiment 29 - Final production K100 (planned)\Build\` unless noted; all LF.

| File | sha | Note |
|---|---|---|
| `build_e29_final.py` | d765c56f | |
| `e29_lib.py` | 48f3b9f8 | adds `run_summary` and `summary_text` |
| `build_e29_cpu_kit.py` | 4271df02 | |
| `e29_kit_lib_tests.py` | f4042756 | |
| `e29_dflash_d3s_code_export.py` | 89f688d7 | |
| `README.md` (experiment folder) | — | rev 2e entry |

The build facts files are updated too and now include the smoke record (sha, diff, tasks, window, budget).

### Open risks
- **DIRECT in the smoke:** with drain 26 inside a 40-min window, DIRECT answers will mostly be forced closes after about 10–13 min of thinking. The smoke proves the route and the mechanics, not DIRECT quality.
- **Two-test task coverage:** 446ef5d2 and b6f77b65 will likely still be in their first attempt or a B2 re-run when the 40-min window ends, so the 2- and 3-test DIRECT mapping may not be exercised.
