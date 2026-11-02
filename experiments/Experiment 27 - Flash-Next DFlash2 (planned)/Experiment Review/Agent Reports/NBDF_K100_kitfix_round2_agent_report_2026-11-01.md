## K100 real kit fixed and rebuilt (not yet re-run)

Both failures were kit-side. The notebook logic and both notebook shas are unchanged: K100 6b83cbbb…, W4 2abac1af…. Nothing was pushed and git was not run.

All units pass locally: W4 281/281, K100 43/43, NaN 173/173, gate test 11/11. The new K100 real kit is built but has not run yet. The W4 gated rebuild was dry-run into a test folder and passes; I did not write it to the Build or push folders.

### 1. Host phase, launch check, spawn env, argv fields and K100_CAPTURES (one root cause)
- **Cause:** the kit started the R4 host-phase / launch dry-run without first writing the QSA scales sidecar into `fn_klee100`. Merlin's host phase refuses FP8 KV without that file. That failure left argv as None, which produced the K100_CAPTURES `argv_bad` list.
- **Fix:** before `plans = {}`, the kit now calls the notebook's own CELL 2b `k100_setup()` (overlay merge + K6, then `k100_scales` copies the sidecar). This is the same function the notebook runs, not a copy. The result is recorded as `K100_SETUP` with `sidecar_exists`.
- **New verdict key:** `k100_setup_scales_sidecar_ok`.
- **The notebook has no such gap:** `nbdf_main` runs merge → `merlin_install` → #60490 → `step("A0", "k100_setup", …)` → sha gate → ppack/seam → Q. So the sidecar is written before the first launch, in the same order as K100Q. If `k100_setup` fails, no arm boots. Not a blocker.

### 2. k100_r9: the 540 vs 480 question — answer (a), a kit modelling bug
- **Mechanism:**
  - The probe's `layer_names` (template line 546) picks attention layers with `LT[i] == "full_attention"`.
  - klee100's config names its QSA layers `"qwen_sparse_attention"` (`fixkeys_overlay/config.json`, `text_config.layer_types`).
  - So no attention names matched, and the names fell back to synthetic `layers.{1000+…}`.
  - `drop_mtp` (`worker_specs`, which matches names starting with `mtp.`) never fired. The MTP layer-48 QSA/MLA/indexer specs stayed in the D-SW simulation.
  - The kit reported PP1 groups as 7 + 7 layers at 16,515,072 B, which is the Q layout.
- **W4 does not have this bug:** the W4 config uses `"full_attention"`. The W4 real kit's R9 shows D-SW at 12 layers / 14,155,776 B / 205 blocks, with the MTP layer dropped.
- **Merlin builds no MTP module on a dflash arm (source lines):**
  - `vllm/v1/worker/gpu/spec_decode/__init__.py:27-44`: `method == "dflash"` + `DFlash2DraftModel` → `DFlash2Speculator`. `MTPSpeculator` exists only for `method == "mtp"` (`:63-66`). So `Qwen4ExpMTP` (`models/qwen4_exp/nvidia/mtp.py:487`), which owns the MTP attention/QSA layer and therefore its KV spec, is never built.
  - The target drops `mtp.*` weights: `models/qwen4_exp/nvidia/model.py:1139`, `:1358`.
  - The MTP hidden buffer exists only when the method is mtp: `model.py:651-668`.
  - So no MTP KV group can exist on PP1 for D3s/D4s; option (b) does not apply. The GPU KV-layout read-back records the real layout either way.
- **Fix (K100 template only, via `_k100_sub`, so the W4 kit is unchanged):**
  - Attention layers are now `LT[i] != "linear_attention"`.
  - The ~540-block and PP1-14,155,776 values are now record-only under `predicted`.
  - Still gated: Q at 480 ± 2, and D-SW blocks > 0, no error, drafter page 10,485,760, PP0 page 14,680,064.
  - Verdict key renamed to `k100_r9_q480_dsw_kv_configs_ok`.

### 3. Recorded
`r9_idx_blocks`: Q 480 → 508 with the indexer KV in fp8 (+5.8 %). It stays record-only in the K100 real kit (`K100_R9_IDX`).

### Gated rebuild: auto-checks and the per-K success field
- **New K100 gate hook:**
  - `build_e27_nbdf.py`: after `kv_gate_from_report`, it calls `KX.k100_gate_extra(KV_GATE, _rep)` in K100 mode only.
  - `build_e27_nbdf_k100.py`: `k100_gate_extra` can only close the gate, never open it. It requires the kit's ALL_OK and all 10 `k100_*` checks.
- **Explicit per-K success (your second-audit addition):**
  - It adds `per_k_success[D3s|D4s].success`.
  - That is `kv_configs_ok` (nbdf_lib's `per_k` boolean from the real `get_kv_cache_configs` runs `K3_swblock1024` / `K4_swblock1024`: num_blocks > 0, no error, an int group count) AND `k100_r9_run_ok` (the kit's per-run real KV-config ok).
  - It is never taken from the 540 prediction.
- **Tested against the old failing K100 report** (`--test-out`, scratch only): passed = False, and `per_k_success` = False for both arms.
- **New audit script `nbdf_gate_verify.py`.** It checks the build facts against the CELL 7 body's embedded `KV_KIT_GATE`:
  - `passed`, `gpu_ready`, embedded == facts;
  - `per_k[arm] is True` for each arm;
  - for K100 also `per_k_success[arm].success is True`.
  - It exits 1 on any miss.
- **W4 gated dry run** from `q27-e27-nbdf-kit-real/nbdf_kit_real_report.json` (W4 report notebook sha = 2abac1af = the current W4 closed build): gated sha 11d4a6a387e1883f627b638eb8d3d2185a3758b9ce2f5a73f804a8b56e23f6c2, 365,647 B. passed True, gpu_ready True, every per_k True (D4s / D3s / D4 / D4t / D7 / D4b), embedded == facts. `nbdf_gate_verify.py` returns ok.

### Commands (from `Experiment 27/Experiment Notebooks/Build/`)

**W4 (real report ALL OK; the run is up to you):**
```
python build_e27_nbdf.py --kv-gate-report "D:/ARC AGI 2/_scratch/e27_k100_kits_out/q27-e27-nbdf-kit-real/nbdf_kit_real_report.json"
python nbdf_gate_verify.py nbdf_build_facts.json nbdf_cell7_body.py D3s D4s D4 D4t D7 D4b
```

**K100 (only after the NEW real kit prints ALL OK):**
```
python build_e27_nbdf_k100.py --kv-gate-report "<new K100 real report>/nbdf_kit_real_report.json"
python nbdf_gate_verify.py nbdf_k100_build_facts.json nbdf_k100_cell7_body.py D3s D4s
```

### Files and shas (all LF, in `Experiment 27/Experiment Notebooks/Build/`)

| File | sha256 | Change |
|---|---|---|
| `build_e27_nbdf_cpu_kit.py` | 231e455e44d50a3897812795cdffeadbb1d117d048b4a1bccd306e1de95d02e8 | K100-only edits: `k100_setup` before R4, the attention-type fix, r9 prediction record-only, setup verdict key |
| `build_e27_nbdf.py` | 81efd33be81d8327e20b7d137004b7926e0a78e7f5dd8574da352321e486baf4 | the 2-line K100 gate hook; W4 output unchanged, 2abac1af |
| `build_e27_nbdf_k100.py` | d55314ba6b999a9659b25fff5568ef37ca9f12636a149bce6773e71e29f33691 | `k100_gate_extra`; notebook unchanged, 6b83cbbb |
| `nbdf_gate_verify.py` (new) | 05989f9586fef08fefd13594361a4a37f7794e8f940fbdffd146e6b84a92addd | audit check after a gated rebuild |

`nbdf_lib.py` is unchanged (42314240…).

### Kits

| Kit | Bytes | sha256 | Status |
|---|---|---|---|
| K100 real (`push_e25/q27-e27-nbdf-k100-kit-real__foysalemonshanto`) | 334,650 | **22cad829a6a03ae3d555a67adf75d7daccd2ffe994a3ae78b44f58d14a90ea4f** | rebuilt, ready to push |
| K100 units | 174,034 | 5ac937ca… | unchanged, already ALL OK |
| W4 units | — | 6b5a17b6… | unchanged |
| W4 real | — | 1a98edab… | unchanged |

K100 audit zip refreshed: `Experiment Review/Audit Zips/E27_NBDF_K100_audit_FLAT_2026-11-01.zip`, sha 7af8b5fb6b6f93330f9b4fa94dd8dda6f47f37d5c6e4efa6191eaabded0c98a9.

### Watch on the next K100 real run
The old report also listed identity fields (tp, pp, partition) as missing. That followed from the failed host phase and should clear now. The gate hook will keep the gate closed if it does not.
