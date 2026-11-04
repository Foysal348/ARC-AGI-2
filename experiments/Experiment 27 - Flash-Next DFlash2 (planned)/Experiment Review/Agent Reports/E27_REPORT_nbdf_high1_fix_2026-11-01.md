# E27 NBDF agent: HIGH-1 (F validation) fix report, verbatim

## HIGH-1 fixed; notebook rebuilt with the gate closed, kits and audit zip refreshed for kit v6 (nothing pushed)

**Kit v6 must run on this notebook:** `532d6dd2e31012c9973d8e18fd43dd874797b48a3a9e2c52625db94608373c07`. The closed-build sha is the same value. It is 393,917 B, `gpu_ready: false`, and the NB4T / E25 files are unchanged.

**Tests (local):**
- `nbdf_unit_tests.py`: **216/216 PASS** (15 new HIGH-1 checks).
- `nbdf_build_gate_test.py`: **11/11 PASS**.
- Units kit, run once: **`@@KIT units ALL OK@@`** (214/214; the 2 export-layout checks need the notebook file, which the kit does not bundle).

The seam is unchanged (`7356d473`). The v5 report is now stale for the gate, as expected. After kit v6: `python build_e27_nbdf.py --kv-gate-report <v6 nbdf_kit_real_report.json>`.

### The fix (`nbdf_lib.py`)
1. **`f_blocks` (`:527`) now returns None** unless pool > 0, mc is a finite number > 0, block is a positive int, and 0 ≤ F ≤ pool. `(205, 4.857, 1024)` now gives None instead of −85.
2. **`level_for_band` (`:558`)** uses only this boot's own numbers, in order:
   - **a.** the boot's max-concurrency line together with its `num_gpu_blocks`, at its own block size;
   - **b.** else a conservative F from the KV layout read-back (`f_from_layout`, `:542`): 2 blocks per request for each state group, plus ceil(2048 / block) + 1 for a sliding-window group;
   - **c.** else the conservative level `CONSERVATIVE_LEVEL` (`:539`): 40k at 4, 90k and 116k at 2, labelled `CAPACITY_UNVERIFIED`. The optimistic 8 / 6 / 5 fallback is deleted.
   - The source used is recorded in `long_levels[band]`.
   - A valid F with no room returns level 0 with `skip: SKIPPED_CAPACITY`.
3. **`capacity_window_status` (`:577`):** a window run at a `CAPACITY_UNVERIFIED` level counts only if it is STEADY with running == level and 0 preemptions. Otherwise it becomes `INCONCLUSIVE_CAPACITY` and gives no speed number. The existing KV-bound step-down in `run_window` is kept.

### In the CELL 7 body (`nbdf_cell7_body.py`)
- `band_level` (`:608`) calls `level_for_band` with the boot's pool, its max concurrency at 131k, its engine block and its KV layout.
- **Level 0:** `run_window` records `<band>@0` = `SKIPPED_CAPACITY` (`:629`). There is no prefix-cache reset and no 116k profile at level 0. The verdict then has a missing long band, so the arm is SHORT_ONLY_WIN at best.
- **Unverified levels:** the window-validity rule above is applied at `:637`.

### Regression tests
| Case | Result |
|---|---|
| `(205, 4.857, 1024)` | F None; the 90k level is the conservative 2 (`CAPACITY_UNVERIFIED`), not an optimistic fallback |
| A consistent 1024-unit pool / mc (820, 4.857) | F = 41 ≥ 0 and a real level |
| Missing pool, missing mc, NaN mc, a bool | the conservative level, labelled; 40k gives 4 |
| Valid F with no room | level 0 with `SKIPPED_CAPACITY` |
| Valid level 1 | a one-stream window |
| Layout estimate when the log line is missing | used, recorded, F = 7 |
| Unverified KV-bound window (preemptions, running ≠ level) | `INCONCLUSIVE_CAPACITY` |
| Unverified but valid window | counts |
| A long band that is `INCONCLUSIVE_CAPACITY` or skipped | SHORT_ONLY_WIN, never a false WIN |

### New shas (all LF)
| File | sha256 |
|---|---|
| `build_e27_nbdf.py` | `0a14a5c60f878e4a19f6ba93f5f9dcb2bbe697b381ea8e2680615b2646a28e6a` |
| `nbdf_lib.py` | `2532eb4497288564af6c57ee8815c044f681af0041b9a6c3664d72583ae30641` |
| `nbdf_cell7_body.py` | `3807b78992da7b19d8ed927d0fb8e0d67f938e0881fcaa2330f72fce6a707fc2` |
| `nbdf_code_export.py` | `d2532d086eb111a4c84f2ea4390b7ecc0321ffd941141136f47c5c4ef2686b28` |
| `nbdf_unit_tests.py` | `a2c18b09cdf98932f32d3be87926e2f15f85cacf6f74c99017623f8d27c787c4` |

`build_e27_nbdf_cpu_kit.py` is unchanged.

**Kit folders**, under `D:/ARC AGI 2/_scratch/push_e25/`, both on notebook `532d6dd2`:
| Folder | Size | sha256 |
|---|---|---|
| `q27-e27-nbdf-kit-units__foysalemonshanto` | 356,674 B | `817f3651…` |
| `q27-e27-nbdf-kit-real__foysalemonshanto` | 282,249 B | `a557c120…` |
| `q27-xhigh-e27-nbdf__foysalemonshanto` (GPU) | — | this notebook |

The audit zip `Experiment Review/Audit Zips/E27_NBDF_audit_FLAT_2026-11-01.zip` is refreshed: 754,310 B, `bb23fb91…`.
