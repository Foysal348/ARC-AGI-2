NB-S2 "capacity" is rebuilt on the current NBDF-K100. Closed reproduction = e2198285 and gated reproduction = c20eba7f, both byte-identical and asserted. It has the CELL 1 switches and both CPU kits are built. Nothing was pushed or uploaded. No Exp 27 file changed: the builder's before/after sha check gives `e27_files_changed_by_this_build: []`. Tests: local 185/185 PASS; kit-layout simulation 180/180 PASS plus `@@KIT units ALL OK@@`. The 5-check difference is the notebook-only checks, which the units kit skips by design. All files are LF only (0 CR).

**Before any Kaggle run: upload a new version of the payload dataset `foysalemonshanto/q27-e25-nbs2-payload`.** The copy on Kaggle (mounted in kits_wave2) has the old body 3f124167 and old lib f9f23786. The new notebook's `_nbs2_ds` checks the sha and would raise `NBS2_PAYLOAD missing or sha mismatch`. The real kit would fall back to its embedded copy and report `nbs2_payload_dataset_mounted = False`.

## 1. Base retarget (`build_e25_nbs2.py`, sha e6d9375c)
- `NBDF_K100_CLOSED_SHA` = e21982855d6430616f277784f162cad27cac94f7bcd8488fed97f123645ba730.
- `K100_REPORT` = "D:/ARC AGI 2/_scratch/kits_wave3/q27-e27-nbdf-k100-kit-real/nbdf_kit_real_report.json". The report's `notebook_sha256` is e2198285 and it is ALL_OK.
- `NBDF_K100_SHA` = c20eba7f00b90b8d8e08308c48d57f09b404214039d6c4fa46c059fea92e2004. I got it by running the NBDF builder (k100, `--test-out`) closed and then gated with that report, in scratch; it is deterministic.
- New asserts:
  - Exp 27's `nbdf_k100_build_facts.json` sha256 == e2198285.
  - Seam = 341d859b (the B1 seam fix).
  - nbdf_lib = b0147751 (the B2 hard-only gates).
- The embedded gate design is kept: the notebook carries NBDF-K100's passed KV kit gate (`passed` True, D3s `per_k_success` True). There is no separate NB-S2 gate.
- Docstring and markdown text updated to c20eba7f / e2198285. New facts keys: `switch`, `k100_report`.

## 2. Exact-once edits vs the new NBDF body
- All existing NB-S2 anchors still matched; none needed adapting.
- Old vs new NBDF-K100 body differ only by the B2 hard-only block and the KV gate JSON. `nbdf_main` and the other cells are identical apart from the lib / seam / body blobs.

## 3. Gate audit: four NB-S2 items neutralized to record-only
1. `s2_window_ledger` ran inside `run_window`'s try. An exception there would have turned a measured window into status ERROR, losing the data. It now has its own try; an error gives `kv_ledger = {"error": ...}` and drops `_s2_samples`.
2. `s2_records` at the end of the arm body: an exception would have failed the arm after all its measurements. Now caught into `s2_records = {"error": ...}`.
3. `s2_readback_failure`: a code exception would have crashed the arm. Now caught into `o["s2_readback_error"]` and is not a stop.
4. B's smoke relabel: under B2, `nbdf_lib.smoke_gate` now emits `INCONCLUSIVE_SMOKE_CONCURRENCY_CONTAMINATION` (never FAIL), so the old relabel was dead code. It now matches both labels → `BASE_LADDER_NONDET_RECORDED` (record-only; it never stopped anything).

Stops I kept as real failures:
- **IDX_NOT_ENGAGED:** fires only when the worker read-back shows the indexer `compressed_key_cache` spec is not fp8, i.e. IDX's one change did not apply. A missing read-back is None and only recorded. I checked Merlin's `indexer_qsa.py` and the `nb1_worker_ext` spec dtype string.
- **DYN_GRAPH_FALLBACK:** cannot fire, because DYN is not among the arms.
- **`nbs2_main` checks:** these mirror `nbdf_main` — merge, #60490, `k100_setup`, sha gate, drafter, KV kit gate (embedded True), seam, ring, budget, and each arm's own tree check.

B not usable is recorded only. With B switched off, the levers' references are missing; the B2 quality and long-oracle gates then return INCONCLUSIVE, not FAIL, so the levers really run.

## 4. Switches
- **CELL 1:** the first lines are a 4-line comment block plus `NBS2_ON = {'B': True, 'IDX': True, 'S28': True, 'LP': True, 'S32': True, 'SPC': True}`. It is written with repr, is a plain line in the notebook, and is not in the payload; the builder asserts this.
- **`nbs2_lib.py`** (1.2, sha d99dc466) adds `SWITCH_KEYS`, `SWITCH_DEFAULT`, `switch_map` and `switch_record`:
  - absent → every arm on;
  - a missing key → that arm off;
  - a value that is not a dict → every arm on, recorded as invalid.
- **Body setup (`nbs2_body_add.py`, sha 684e9627):** reads `globals().get("NBS2_ON")` into `NBS2_SWITCH`, records `NBS2["switch"]`, prints the map, and the planned walk uses only the arms that are on.
- **`nbs2_main`:**
  - Arms switched off are marked `SKIPPED_SWITCH_OFF` first, before any other check, and are never booted.
  - Only the arms that are on (`RUN_KEYS`, recorded) get the later setup-failure skips and the loop.
  - With B off, the levers run; the ppack and seam are still applied.
  - With every arm off: `ALL_ARMS_SWITCHED_OFF`, no setup, finish and finalize only.

## 5. Unit tests (`nbs2_unit_tests.py`, sha d837d868)
- Base checks: closed / gated / seam / nbdf_lib shas and the report path. The new `--export` argument is added.
- The facts checks moved to a new `t_facts` that needs no notebook. `t_notebook` keeps only the notebook checks.
- B2 body checks added.
- `t_switch_lib`: absent, default, missing key, empty and invalid inputs, plus the record.
- `t_switch_main`:
  - all on → all launch;
  - B off → the levers launch and seam/ppack are still applied;
  - each lever off → not launched, `SKIPPED_SWITCH_OFF`;
  - missing keys → off;
  - a switched-off arm keeps `SKIPPED_SWITCH_OFF` when the drafter or merge fails;
  - switching B off frees budget;
  - all off → no setup.
- Switch line in the export, read from `nbs2_code_export.py` via the facts' `cell_bytes`: present once, at the top of CELL 1, the literal equals the default, it runs, and it is not in the body or lib.
- Results:
  - Local: **185/185 PASS**.
  - Negative control against the old build: 144/181, with 37 FAILs, every one a new check — so the tests do tell old from new.
  - Kit-layout simulation (empty folder `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\q27-e25-nbs2-kit-units`, kit code run with `python -I`, no repo paths): **180/180 PASS**, only `SKIP notebook`. It ran on local Python 3.14, so the kit's Py3.12 check was waived through `KIT_LOCAL_SIM=1`; Py3.12 is only confirmed on Kaggle.

## 6. Kits
- **`build_e25_nbs2_units_kit.py`: NEW** (sha 88ef703d) — this file did not exist before.
  - Flat layout following NB-S1's pattern, 10 shipped files: nbdf_lib, nb4t_lib, nb4_ppack, nb4_timeline, nbs2_lib, tests, body, export, facts, Merlin `dynamic/utils.py`.
  - Has the `--sim DIR` option.
  - Output `D:/ARC AGI 2/_scratch/push_e25/q27-e25-nbs2-kit-units__foysalemonshanto/`: 367,082 bytes, sha 3e24bd6c.
- **`build_e25_nbs2_cpu_kit.py`** (sha 5931def4) now builds the real kit only; its old units kit was removed so two builders don't write the same folder.
  - S2-4 fix, kit only: the argparse probe now does what `vllm serve` itself does (`cli/serve.py` `ServeSubcommand.cmd`): `ns.model = ns.model_tag` before `EngineArgs.from_cli_args`, with `HF_HUB_OFFLINE=1` already set by `venv_py`. So `get_model_path` resolves the local model path instead of looking up the default HF repo id offline. It also records `model` and `model_path_exists`.
  - New check `nbs2_switch_default_all_on`, and the switch is recorded.
  - The notebook's own pre-launch sequence (from the NBDF-K100 kit template) is unchanged.
  - Output `q27-e25-nbs2-kit-real__foysalemonshanto/`: 384,481 bytes, sha 85139e40.

## 7. New outputs and shas
| File | sha256 | Note |
|---|---|---|
| `Q27-XHIGH-E25_NBS2_CAPACITY.ipynb` | 50654cc6c278a2b82bd245f1da6ee8e3ccc74f970fd1068c0957706b46178fc0 | 346,843 B (< 400 kB), 11 cells; was a0faf93c |
| `nbs2_cell7_body.py` | c7129537d78d0cb489b561f47ec0ccd8a4dad57c864aa2f6b821908f49cb00e1 | |
| `nbs2_lib.py` | d99dc466fff6bccfc2040518c18e0f613bd96a23f2e4a3a539a3717b1d6f2750 | |
| `nbs2_code_export.py` | b079ddca03fad183e7d019e2a3af06e9a5fe016367beee693bd457ba669e3b0f | |
| `nbs2_build_facts.json` | 037b87e1 | |

- **Payload** `D:/ARC AGI 2/_scratch/push_e25/ds_e25_nbs2_payload/`: body c7129537, lib d99dc466, `SHA256SUMS.json` fdadc274, `dataset-metadata.json` 5bee030e. Not uploaded.
- **Push folders** `q27-xhigh-e25-nbs2-capacity__esfersami50` and `__foysalemonshanto` both hold notebook 50654cc6.
- **Unchanged:** nbdf_lib b0147751, seam 341d859b, ppack 8dadf076.

## 8. Not done / open
- Nothing was pushed or uploaded.
- The real kit's S2-4 fix could not be run here (it needs the Merlin venv on Kaggle). It is confirmed only against Merlin's own source (`serve.py`, `arg_utils` `__post_init__`, `repo_utils.get_model_path`).
- **Budget, from the facts' planned walk (design unchanged):** at the default 175 min only B, IDX and S28 fit; LP, S32 and SPC would be `SKIPPED_BUDGET`. The switches can be used to pick which arms run.
- Backups of the old NB-S2 files are in `D:\DevCache\Temp\claude\d--ARC-AGI-2\1c6db7aa-c78b-4b3a-b97a-5c67187fe52f\scratchpad\nbs2_before\`.