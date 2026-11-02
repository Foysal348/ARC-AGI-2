## NBDF round 4: all four items done; both notebooks and all four kits rebuilt

The `--help` failure was kit-only. The notebook's gate is the same as in K100Q and E28, so I did not change it. Because of the TF-verdict fix in the shared lib, both notebooks have new shas (you accepted this).

Nothing was pushed and git was not run. All edited files are LF. Units pass locally. Both units kits (run locally once) print ALL OK. The new real kits have **not** run yet.

### 1. Why `--help` gave nothing usable in the kit (kit-only)
- **Cause:** the notebook's own `L.help_check` (the pinned lab `k100load_lab.py`, sha ef393fbb, `help_check` at lines 224-237) runs `vllm serve --help=all` with the launch env. On a GPU-less kernel vLLM cannot infer a device and dies with a long traceback.
  - The traceback is at least 2000 characters, so the lab marks the help as `ran: True`, and no flag text appears in it. Every flag therefore shows as "missing", which matches all 36 in the report.
  - This is the same failure klee100 LOAD kit v1 hit ("every --help and argparse died in DeviceConfig").
- **Kit fix (K100 kit only):**
  - `VLLM_TARGET_DEVICE=cpu` is set in `os.environ` just for the R4 loop. It reaches the help subprocess through the notebook's launcher (base env = `os.environ` → Merlin `build()`), and it is popped afterwards. The K100Q and klee100 kits use this same fix and got ALL OK.
  - The kit now records the help text (head 3000 chars and length).
  - New check `k100_help_flag_gate_real_ok`. It needs real help text (contains `--max-num-seqs`, `--speculative-config`, `--kv-cache-memory-bytes`, with no "Failed to infer device" and no Traceback), and the notebook's own `cli_flags` must be ok True for Q, D3s and D4s. The notebook's gate itself is unchanged.
- **The notebook calls `--help` exactly as K100Q and E28 do:**
  - `nbdf_k100_code_export.py:575` `HELP[name] = L.help_check("merlin", argv[:argv.index("serve") + 1], env, argv)` is the same line as `e26_k100q_code_export.py:1375` and `e28k100_code_export.py:1460`: same lab function, the launch env, the default 300 s timeout.
  - The gate is `nbdf_k100_code_export.py:848-852`: `KL.cli_flags_check(HELP[name])`, and ok False raises `FAIL_CLI_FLAGS`. That matches `e28k100:1881-1884` and `e26_k100q:1796-1799`.
  - The only difference: NBDF skips the call when `HELP[name]` is None, while E28 passes None in. Both give ok None and no raise.

### 2. Step coverage: notebook to kit (K100 real kit)
While mapping the steps I found two more kit-only gaps and fixed them:
- **Q's I8 head:** the kit never set Q's draft head or `DRAFT_HEAD=int8`, so it was not testing the argv the notebook launches.
- **The two patch steps:** the kit called `_ppack_apply` directly, and it never ran the per-D-arm `SSEQ.check` / `PSEQ.before_arm`. The notebook runs both.

| # | Step | Notebook function | Kit equivalent | Covered |
|---|---|---|---|---|
| 1 | merge | CELL 3 merge (`R["sections"]["A0"]["merge"]`) | R1 merge (the notebook's) | yes |
| 2 | install | `merlin_install()` | R2 `merlin_install()` | yes |
| 3 | #60490 + worker extension | `apply_60490` | R3 `apply_60490()` | yes |
| 4 | k100_setup / scales sidecar | `k100_setup()` | `k100_setup()` + sidecar exists | yes |
| 5 | sha gate | `sha_gate(...)` | R1 `sha_gate(4*3600)` | yes |
| 6 | ppack | `pseq.before_arm(ARMS["Q"])` | **now** `PSEQ.before_arm(ARMS["Q"])` (was `_ppack_apply` directly) | yes (fixed) |
| 7 | seam apply | `sseq.apply()` + `ring_status` | R3S `SSEQ.apply()` + ring + combined state | yes |
| 8 | per-D-arm checks | `sseq.check(k)`, `pseq.before_arm(a)` | **now** in `plan_one` for D3s / D4s | yes (fixed) |
| 9 | tree check | `arm_boot` → `tree_check(arm)` | `tree_check(a)` | yes |
| 10 | Q I8 | `arm_boot`: Q_I8 → `draft_head` / `expect_cfg`, `NB4T_BUILD_ENV` `DRAFT_HEAD=int8` | **now** the same lines in `plan_one` | yes (fixed) |
| 11 | per-launch knobs | `NB1_EXTRA` / `NB1_ENV` / `NB1_KV` at `K100_CFG q_pin`, `ARM_OF` | **now** at `q_pin` (was `a["kv_bytes"]`, which has the same value) | yes |
| 12 | K6 before launch | K100 wrapper `KL.k6_check` | through `merlin_fn_server` (recorded per plan) | yes |
| 13 | Merlin plan + host_phase | CELL 2 `build()` + `host_phase()` | through `merlin_fn_server` | yes |
| 14 | `--help` flag gate | CELL 2 `L.help_check` + K100 `cli_flags_check` | the notebook path, `VLLM_TARGET_DEVICE=cpu`, help text must be real | yes (fixed) |
| 15 | NBDF launch check | `_nbdf_launch_check` → `N.launch_check` | the same wrapper, plus kit `launch_check` at `kvb` | yes |
| 16 | VLLM_QSA_KV_SCALES env | K100 wrapper (fp8) | `s.env` recorded and checked to end in `qsa_kv_scales_262k.json` | yes |
| 17 | env reaches the spawned child | worker spawn | R4 `spawn_env` | yes |
| 18 | spec / captures / seqs / KV argv | `launch_check` `want_sp`, captures, seqs 24 | R5 (Q spec now expects `draft_lm_head_quant` int8, as `launch_check` does), `K100_CAPTURES` | yes (fixed) |
| 19 | boot_gate, LoadMonitor, `with_servers`, readiness | GPU only | none (no GPU) | no: cannot run on CPU |

New check `k100_notebook_launch_steps_all_ok` gates on steps 1-18; the per-step results are in `checks.K100_STEPS`.

### 3. TF verdict fix (shared `nbdf_lib`, HIGH)
- **The fix:** in `arm_verdict`, after the FAIL check, when `TF_REQUIRED` is set and the TF status is not PASS, it sets `out["tf_verification_pending"] = True` and the flag `TF_NOT_VERIFIED`. Before the final update, a WIN or SHORT_ONLY_WIN becomes `INCONCLUSIVE_TF`, with the old verdict kept in `out["speed_verdict"]`. Speed and long data are untouched, and no arm is skipped or stopped.
- **Why a knob:** W4 never computes `tf_gate`, so applying the rule unconditionally would have turned every W4 WIN into INCONCLUSIVE_TF. I added `TF_REQUIRED` (default False), and K100 sets it through `N.configure(..., TF_REQUIRED=True)`. W4 verdicts are unchanged.
- **Tests (K5):**
  - TF PASS + speed PASS → WIN, with no flag.
  - TF FAIL → CORRECTNESS_FAIL_TF (existing test).
  - TF absent → INCONCLUSIVE_TF, speed_verdict WIN, speed data intact.
  - TF INCONCLUSIVE → INCONCLUSIVE_TF, the same way.
  - W4 mode (knob off) with TF absent → still WIN.

### 4. `nbdf_gate_verify.py` fix (HIGH: it failed open)
- **K100 rule:** when the variant is `k100`, `per_k_success` must be a dict. D3s and D4s must each have `success`, `kv_configs_ok` and `k100_r9_run_ok` all `is True` (the field names `k100_gate_extra` emits).
- **Notebook sha:** a new third argument takes the .ipynb path; its bytes must hash to the facts' `sha256`.
- **Fails closed:** every missing field fails. The verifier lists each reason in `why`.
- **New test file `nbdf_gate_verify_test.py`, 10/10:**
  - The valid K100 gate passes.
  - These all fail: `per_k_success` missing, `per_k_success` None, D4s entry missing, `kv_configs_ok` False, `k100_r9_run_ok` False, notebook sha mismatch, notebook file missing.
  - A W4 gate with no `per_k_success` passes. A W4 gate with one per_k None fails.

### New notebooks (closed builds, gpu_ready false)

| Notebook | sha256 | Bytes |
|---|---|---|
| K100 `Q27-XHIGH-E27_NBDF_K100.ipynb` | **456c47e6ef8df4bae5a4db581b5d2e189156dc8111056145b47a95e6f2b6aa12** | 375,740 |
| W4 `Q27-XHIGH-E27_NBDF_DFLASH2.ipynb` | **ba8ae0514b46612c7235b80a3855c8a2fd674b67d2b2a991a666853e1fbc4ad5** | 364,997 |

**Both real kits must run again** before either gated rebuild. The earlier W4 real ALL OK was for 0572996a, and the K100 report was for d0d3ff0d; both are now stale.

### Kits (`D:/ARC AGI 2/_scratch/push_e25/`)

| Kit | Bytes | sha256 |
|---|---|---|
| `q27-e27-nbdf-k100-kit-real__foysalemonshanto` | 339,783 | **870c8f2a602f904e73c7d7b79c740f581e8200989343383362302a2318a0d222** |
| `q27-e27-nbdf-k100-kit-units__foysalemonshanto` | 176,509 | 248dc93db96369bc9f5c4d86bed571df9493808b8c90639917761f4a103f4ee7 |
| `q27-e27-nbdf-kit-real__foysalemonshanto` (W4) | 295,479 | ea33192987a7400e8ebbe056ae935123bcdeb69a703fa9f8dbd7e395101e6672 |
| `q27-e27-nbdf-kit-units__foysalemonshanto` (W4) | 371,647 | bac763b7c9cfb126a7598f198118c1e3d8f077304099bbc7c4396fefc713b9cc |

The W4 real kit only picked up the new notebook sha; none of the K100 kit changes are in it.

### Tests

| Run | Result |
|---|---|
| `nbdf_unit_tests.py` (local) | 281/281 |
| `nbdf_k100_unit_tests.py` (local) | 63/63 |
| `nbdf_k100_nan_tests.py` (local) | 173/173 |
| `nbdf_build_gate_test.py` | 11/11 |
| `nbdf_gate_verify_test.py` | 10/10 |
| W4 units kit (local run) | 279/279, ALL OK |
| K100 units kit (local run) | k100 57/57, NaN 173/173, ALL OK |

### Files changed (`Experiment 27/Experiment Notebooks/Build/`)

| File | sha256 (prefix) | Change |
|---|---|---|
| `nbdf_lib.py` | cd2c7281 | `TF_REQUIRED` knob + INCONCLUSIVE_TF |
| `build_e27_nbdf_k100.py` | d59da5ef | `configure(TF_REQUIRED=True)` |
| `nbdf_k100_unit_tests.py` | 3bd6550b | the TF tests |
| `build_e27_nbdf_cpu_kit.py` | 79ceb8f4 | K100 subs: the launch path through the notebook's own functions, the help fix, the R5 Q spec, `K100_STEPS` and the two new checks |
| `nbdf_gate_verify.py` | 9fc37b25 | rewritten |
| `nbdf_gate_verify_test.py` | 37a14ba8 | new |
| `nbdf_cell7_body.py` (generated) | 8507aaf2 | W4 body, unchanged |
| `nbdf_k100_cell7_body.py` (generated) | f62f47ef | K100 body |

### Audit zips (in `Experiment Review/Audit Zips/`)
The verifier and its test file are now in both zips.
- K100: `E27_NBDF_K100_audit_FLAT_2026-11-01.zip`, sha 2e1218fdd6c101b22729bab3b6d1ef434d6d2d0f89c25f68c8558db52edfc876
- W4: `E27_NBDF_audit_FLAT_2026-11-01.zip`, sha 116b9352ec2f2259073df0779ef0c79e9a61ee167bdd8e12415bb9e94183484f

### Gated rebuild commands
Run from `Build/`, and only after the new real kit for that notebook prints ALL OK. The verifier now takes the notebook path.
```
python build_e27_nbdf_k100.py --kv-gate-report "<K100 real report>/nbdf_kit_real_report.json"
python nbdf_gate_verify.py nbdf_k100_build_facts.json nbdf_k100_cell7_body.py "D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e27-nbdf-k100__foysalemonshanto/q27-xhigh-e27-nbdf-k100.ipynb" D3s D4s
python build_e27_nbdf.py --kv-gate-report "<W4 real report>/nbdf_kit_real_report.json"
python nbdf_gate_verify.py nbdf_build_facts.json nbdf_cell7_body.py "D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e27-nbdf__foysalemonshanto/q27-xhigh-e27-nbdf.ipynb" D3s D4s D4 D4t D7 D4b
```
No README referenced the verifier; its usage is in the docstring.
