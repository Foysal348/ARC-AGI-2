## Blocker batch done: B1 (e2-guard seam edit) and B2 (hard-only gates) are in; all three notebooks rebuilt (closed)

NBDF-K100, W4 and NB-S1 are rebuilt as closed builds (`gpu_ready: false`), waiting for your kit run and gated rebuilds. The six NBDF / NB-S1 kits are rebuilt on foysal. All local units pass, and both NBDF units kits print ALL OK when run locally. The real kits only run on Kaggle.

Nothing was pushed and git was not run. Frozen-hash asserts: `build_e25_nbs1.py` and `build_e25_nbs1_cpu_kit.py` hash the NBDF files at start and compare at the end, so they follow the new shas automatically; both passed. All edited files are LF.

### New notebooks (closed)

| Notebook | sha256 | Bytes |
|---|---|---|
| NBDF-K100 `Q27-XHIGH-E27_NBDF_K100.ipynb` | **e21982855d6430616f277784f162cad27cac94f7bcd8488fed97f123645ba730** | 378,900 |
| W4 `Q27-XHIGH-E27_NBDF_DFLASH2.ipynb` (build only) | **e5723dfa33bf1c1e95cd629a4cd0c8768b4107d175ff3aad6987469e8b16b653** | 368,127 |
| NB-S1 `Q27-XHIGH-E25_NBS1.ipynb` | **d5e8423aa2c8c4969bfc0413de3e6b9c5a4a16d1cf2593e4b58e7374194e03e1** | 385,104 |

### Kits (`D:/ARC AGI 2/_scratch/push_e25/`, all `__foysalemonshanto`)

| Kit | sha256 | Bytes |
|---|---|---|
| `q27-e27-nbdf-k100-kit-real` | **d8780585bd89d8ef341b3dbcd5ea99204c178c4f03066ca95aa375da63ed1127** | 345,090 |
| `q27-e27-nbdf-k100-kit-units` | 3da7b3b61a4e13c928c1c029a31fd1579af3197992b3d655e011d9790bb0fc9c | 187,979 |
| `q27-e25-nbs1-kit-real` | **dc0dd6afac5e095758f3bba9bd1ee588fc5e28b5783cf3615c9d433633307344** | 358,120 |
| `q27-e25-nbs1-kit-units` | a57862e9e4055fdc58ca0bd919b9d66c661c384975c92be24afcd46ce40f4ec3 | 385,698 |
| `q27-e27-nbdf-kit-real` (W4, build only) | c0819097f418f5109520195647c51d10bcc3dd6e65ea061f977ef95da063a4ee | 300,581 |
| `q27-e27-nbdf-kit-units` (W4) | ccb0395096f1f8e022a39419a58ab0a1ebf399a353f3830f4bd4bfc1eb068eb5 | 379,957 |

### B1: the e2-guard edit in `dflash_seam.py`
- **The edit:** a new exact-once patch, GRD, on `vllm/v1/e2_guards.py`:
  - it adds `"draft.query_cudagraph_manager": "FULL_DECODE_ONLY"` to `EXPECTED_RESOLVED_MODES`;
  - it adds `"query_cudagraph_manager"` to `_DRAFT_MANAGER_ATTRS`.
- **File shas:** pre `aa25244647523c74b43d6fe93d61de965bbbe810c92a20f7d5bd4c2dd516e5c3` (Merlin ad42e37), post `d50d754e8479cf77b577f283ef6456a8a8db550b179ca9aa6a4c26d3126bde11`.
- **Seam integration:** GRD is in `PATCHES` and `CORE_RELS`, so it goes through the same apply / verify / revert and three-state check as every other seam file.
- **The guard stays strict:** it is still unconditional. An eager drafter (NONE) or a downgraded target still raises, and MTP speculators are judged exactly as before.
- **Seam sha:** `341d859be5b2cc619247ff7b645304b9473d09f07995358ead2ee02be116c60b`. `dflash_seam_tests.py` passes (ALL PASS).
- **Unit test K8** (`nbdf_k100_unit_tests.py`). It runs the real Merlin text, patched by the seam, with a fake DFlash2Speculator:
  - the ORIGINAL guard raises `<unrecognised speculator DFlash2Speculator>`, which reproduces the GPU failure;
  - the patched guard does not raise with FULL_DECODE_ONLY;
  - it still raises with NONE, and still raises for a downgraded target;
  - the patch applies and reverses exactly.
  - `e2_guards.py` is now bundled in the K100 units kit so K8 runs there too.
- **Real kit, new R13** (W4, K100 and NB-S1 real kits). With the seam applied to the installed tree, it imports the real `vllm.v1.e2_guards` in the Merlin venv and calls `assert_resolved_cudagraph_modes` with a fake DFlash2Speculator:
  - it must pass with FULL_DECODE_ONLY and raise with NONE;
  - the result is the new verdict key `e2_guard_dflash_ok`, added to `nbdf_lib.REAL_KIT_CHECKS`, so the gated rebuild requires it.

### B2: only hard breakage stops an arm
- **Garbage check:**
  - The body rebinds the lab's `L.degenerate` to `N.hard_only(...)`: only EMPTY / REPEATED_CHAR / BANG_GARBAGE count. That covers every gate that uses it: lab steady, smoke, oracle, long oracle, sampled, windows.
  - `nbdf_lib.garbage_review` still uses the full check, so LOW_VARIETY & co. are recorded in the window's `labels`, and the window keeps STEADY (its tok/s counts).
  - A complete row with no tokens, no text and no error is EMPTY, which is a stop.
- **Smoke:** EMPTY, a request error, hard garbage, no drafts, and zero accept (< 2 % after ≥ 200 drafts) still stop. A short stream becomes INCONCLUSIVE_SMOKE_SHORT, and concurrency contamination becomes INCONCLUSIVE_SMOKE_CONCURRENCY_CONTAMINATION. Both are recorded only.
- **The cells' gates, redefined in the body** (they call the new `nbdf_lib` helpers `short_gate_hard`, `quality_gate_hard`, `oracle_gate_hard`, `sampled_hard_fail`):
  - `short_gate_failed`, `quality_gate`, `oracle_gate` and `evidence_gate` stop or NO_GO only on: a probe request error, a sampled request error / dead server / no token / hard garbage, an oracle request error or hard oracle garbage, or a window with hard garbage.
  - FAIL_HIGH_MARGIN, wrong probe answers and a short sampled request are now INCONCLUSIVE_*, recorded only.
- **Long oracle:** a high margin vs Q becomes INCONCLUSIVE_LONG_ORACLE_HIGH_MARGIN. A request error or hard garbage still stops.
- **`arm_verdict`:**
  - CORRECTNESS_FAIL now only for hard correctness or a hard quality FAIL. CORRECTNESS_FAIL_LONG only for a long-oracle request error or garbage.
  - FAIL_HIGH_MARGIN, NO_GO_QUALITY and long-oracle labels, and contamination, are recorded as flags; the measured verdict stands.
  - CORRECTNESS_FAIL_TF stays the verdict-level correctness gate.
- **Tests:**
  - K9 covers:
    - ARC grid text reads LOW_VARIETY, yet the window is not garbage, keeps STEADY with `labels`, and counts for speed;
    - EMPTY, REPEATED_CHAR and BANG_GARBAGE still stop;
    - a recorded label never makes a stop condition true (smoke, short, quality, oracle, evidence and long-oracle gates);
    - the body's redefined gates on Q's exact GPU case: no stop, no NO_GO;
    - a probe request error or an EMPTY sampled request still stop.
  - Two old tests that encoded the old rule were updated: T6 (contamination) and K5 (ladder on a clean Q).
  - The T15 high-margin test now expects WIN plus the flag.
  - The Q-failure matrix (K6) still passes.

### Local test results

| Suite | Result |
|---|---|
| `nbdf_unit_tests.py` | 281/281 |
| `nbdf_k100_unit_tests.py` | 91/91 |
| `nbdf_k100_nan_tests.py` | 173/173 |
| `nbdf_build_gate_test.py` | 11/11 |
| `nbdf_gate_verify_test.py` | 10/10 |
| `dflash_seam_tests.py` | ALL PASS |
| `nbs1_unit_tests.py` | 56/56 |
| W4 units kit (local run) | 279/279, ALL OK |
| K100 units kit (local run) | k100 85/85, NaN 173/173, ALL OK |

### Changed files (NBDF `Build/`)

| File | sha256 (prefix) |
|---|---|
| `dflash_seam.py` | 341d859b |
| `nbdf_lib.py` | b0147751 (full: b014775162fb55401826fa8c498f7009b5d47ad15cc5cb365ab23e5ed37b1929) |
| `build_e27_nbdf.py` | 585c5ddc |
| `build_e27_nbdf_cpu_kit.py` | 47f5b123 |
| `nbdf_k100_unit_tests.py` | b6231537 |
| `nbdf_unit_tests.py` | 6fc16f73 |
| `nbdf_cell7_body.py` (generated) | a18e9d66 |
| `nbdf_k100_cell7_body.py` (generated) | 1e6049ce |

NB-S1 sources are unchanged (`nbs1_lib` 9668adca, builder 0b66edf2, your kit builder 54b851b8); NB-S1 inherits B1 and B2 through the shared files.

### Step coverage
- **NB-S1 real kit:** `checks.NBS1.steps` is unchanged in structure: merge → install → #60490 → k100_setup + sidecar → sha gate → PSEQ (Q-defer) → seam apply → B base gate → arm plan → per arm (D-loop checks, tree_check, arm_boot knobs, K6, build + host_phase, `--help` gate, launch_check at the pin, QSA scales env, spawned-child env, the arm's own env change, argv / captures, KV probe) → revert. All are covered; the GPU-only items are listed as not covered.
- **New in all three real kits:** R13 covers the capture-time e2-guard, which is what killed D3s and D4s on GPU. It is gated through `e2_guard_dflash_ok` in the base verdict, not as a row in the NBS1 steps table.

### Audit zips (refreshed)

| Zip | sha256 |
|---|---|
| `E27_NBDF_K100_audit_FLAT_2026-11-01.zip` | e476a72c96c1aa24db58cc47a034e3c143461dcf19d54506bc5e70538cac91ba |
| `E27_NBDF_audit_FLAT_2026-11-01.zip` | 0d052d34db640caf1253974fb87b9b8201eaea427acc2d6c1b39bc0fd6ab7c4d |
| `E25_NBS1_audit_FLAT_2026-11-01.zip` | ff125fa156c6d74d447c587d4f0df567922b7f259a20e23c2c12a6e82214bcc9 |

### Next
Push the K100 and NB-S1 kits. After each real kit prints ALL OK, run the gated rebuilds:
```
python build_e27_nbdf_k100.py --kv-gate-report <report>        # from Exp 27 Build/
python build_e25_nbs1.py --kv-gate-report <report>             # from Exp 25 Build/
```
Then check each with `nbdf_gate_verify.py` (facts, body, notebook path).
