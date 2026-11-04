## NBDF-K100 round 3: all three fixes in, plus the kit fixes; both notebooks have new shas

Nothing was pushed and git was not run. All new/edited files are LF. Units pass locally and both units kits (run locally once) print `@@KIT units ALL OK@@`. The real kits are rebuilt but have not been run.

**The W4 notebook changed, so the old W4 real report no longer matches.** Fix 1 changes the shared body and fix 2 changes `nbdf_lib`. The old W4 real report (notebook 2abac1af) therefore cannot open the W4 gate; the W4 real kit has to run again before any W4 gated rebuild.

### New notebooks (closed builds)

| Notebook | sha256 | Bytes | GPU push folder |
|---|---|---|---|
| K100 `Q27-XHIGH-E27_NBDF_K100.ipynb` | **d0d3ff0d1c1e3f7f9256d0f64d877bd656c53f0c687d6a6aa46d459ac0e11559** | 375,540 | `push_e25/q27-xhigh-e27-nbdf-k100__foysalemonshanto` |
| W4 `Q27-XHIGH-E27_NBDF_DFLASH2.ipynb` | 0572996ae2ae4dea7920b0d06e85266ca2bd81f09e80ae29977d54382be1c21b | 364,807 | `q27-xhigh-e27-nbdf__foysalemonshanto` |

### 1. The last cross-arm skip is removed
- **Fix** (in the shared body template, so it applies to both W4 and K100):
  - A failed `pseq.before_arm(ARMS["Q"])` now marks only Q as NOT_INTEGRATED. The "no base: Q" skip of every D arm and the `return` are gone; `q_tree_ok` is recorded.
  - Q is not launched in that case, and the run continues.
  - Each D arm then runs its own `sseq.check` / `pseq.before_arm` tree check, which decides for that arm.
  - The string "no base: Q" no longer appears in either body.
- **What happens next differs by notebook:**
  - In K100, Q being unusable is recorded only, so D3s and D4s run.
  - In W4, the W4 base gate (`base_usable`, real failures) then still skips the D arms, with the reason "Q not a usable base". That is the W4 rule from the earlier round; I did not change it.
- **Test K6:** if Q's ppack `before_arm` fails, Q ends NOT_INTEGRATED and is not launched, while D3s and D4s pass their own checks and the runner is called for both. A second check confirms no "no base: Q" text is left in the body.

### 2. draft_load_check: the seam's emit line now parses cleanly
- **Emit side** (`dflash_seam.py`, `DFL_PATCH`): `_e27_check_loaded` → `_e27_log("draft_load_check", expected=, loaded=, dropped=)`, which does `print("E27_DFLASH " + json.dumps(rec, sort_keys=True), flush=True)`. This runs once per drafter worker, i.e. on the 2 PP1 TP ranks.
- **Stream:** worker `print()` output does reach the server log the notebook reads (`s.log()`). It arrives **without** vLLM's `(Worker_PP1_TP0_EP2 pid=…)` prefix. The proof is the W4 GPU log `server_nbdf_q.log`: nb4_ppack's worker-printed `NB4_PPQ {...}` lines appear there unprefixed, while logger lines carry the tag.
  - The parser regex `EVENT_RX` (`nbdf_lib.py:142`) needs no prefix, and an event without a tag defaults to PP1. So the format matched already.
- **Parser hardening in `nbdf_lib.dflash_readback`**, so a correctly loaded drafter cannot fail on log format. The old rule `len(events) == tp` would fail if a line were printed twice. The new rule needs all of:
  - at least `tp` distinct pids;
  - no skipped entry;
  - `loaded == expected` wherever counts are present;
  - no entry tagged PP0.
- **Test K7:**
  - It executes the seam's own `_e27_log` and `_e27_check_loaded`, taken verbatim from `DS.DFL_PATCH`, on a 67-parameter model for pids 2410 and 2411.
  - It parses to PASS three ways: no prefix (the real GPU format), with a vLLM prefix, and with one line re-printed.
  - It stays a real failure three ways: only one worker, DFLASH_LOAD_CHECK=0 (skipped), or PP0-tagged lines.

### 3. Q failure matrix (K6, in the K100 units)
D3s and D4s are launched (runner called) in every case below:
- Q boot crash, as LAUNCH_FAILED and as SERVER_DIED;
- Q I8 retry failed (I8_READBACK_UNVERIFIED);
- Q smoke fail (NO_GO_SMOKE);
- Q oracle NO_GO (NO_GO_QUALITY);
- Q not a usable base (garbage);
- Q-defer real failure (NOT_INTEGRATED, init lines 2 of 4);
- Q's ppack `before_arm` fails.

### Kit fixes from the earlier failing report (kept)
- **Scales sidecar:** the kit now runs the notebook's own `k100_setup()` before the R4 host phase. Check `k100_setup_scales_sidecar_ok`. The notebook itself already writes the sidecar before the first launch (merge → install → #60490 → `k100_setup` → Q), so there was no notebook blocker.
- **The 540-block question is (a), a kit modelling bug.**
  - The probe picked attention layers with `LT[i] == "full_attention"`, but klee100's config names its QSA layers `"qwen_sparse_attention"`. So `drop_mtp` never removed MTP layer 48, and the D-SW simulation kept its KV group.
  - Merlin builds no MTP module on a dflash arm: `v1/worker/gpu/spec_decode/__init__.py:27-44` builds `DFlash2Speculator` for dflash, while `MTPSpeculator` exists only for `method == "mtp"` (`:63-66`). The target also drops `mtp.*` weights (`nvidia/model.py:1139`, `:1358`).
  - The W4 config uses `"full_attention"`, so the W4 kit dropped MTP correctly (12 layers, 14,155,776 B, 205 blocks).
  - The probe now treats every non-linear layer as attention, in the K100 template only.
  - The ~540-block and PP1 14,155,776 B values are record-only. The real KV-config checks (blocks > 0, no error, drafter page, PP0 page, Q 480 ± 2) stay gates; the check key is `k100_r9_q480_dsw_kv_configs_ok`.
- **per_k success field (your second-audit addition):** `k100_gate_extra` adds `per_k_success[D3s|D4s].success`. It is `per_k` (the real `get_kv_cache_configs` boolean) AND the kit's K100_R9 per-run ok. It can only close the gate: it also requires the kit's ALL_OK and all 10 `k100_*` checks.
- **Record:** `r9_idx_blocks` shows Q 480 → 508 with the indexer KV in fp8 (+5.8 %).

### Tests

| Run | Result |
|---|---|
| `nbdf_unit_tests.py` (local) | 281/281 |
| `nbdf_k100_unit_tests.py` (local) | 59/59 |
| `nbdf_k100_nan_tests.py` (local) | 173/173 |
| `nbdf_build_gate_test.py` | 11/11 |
| W4 units kit (local run) | 279/279, ALL OK |
| K100 units kit (local run) | k100 53/53, NaN 173/173, ALL OK |

The kit shows fewer K100 checks (53 vs 59) because the 6 checks that read the notebook skip there.

### Kits (`D:/ARC AGI 2/_scratch/push_e25/`)

| Kit | Bytes | sha256 |
|---|---|---|
| `q27-e27-nbdf-k100-kit-real__foysalemonshanto` | 334,765 | **8f8ba2d3d37b880bbbbe359cb058538e0994da70fbdb0f5d051f3c58e8ee4b5d** |
| `q27-e27-nbdf-k100-kit-units__foysalemonshanto` | 175,984 | 71b9728c0006f7933225a96b47d84595dd32a0b3fac85ec7d04c09aa7cb8534f |
| `q27-e27-nbdf-kit-real__foysalemonshanto` (W4) | 295,279 | 6f286a08e4ab430c201b72f4d23adf56f8979d27e466816018789f4061ec1bd0 |
| `q27-e27-nbdf-kit-units__foysalemonshanto` (W4) | 371,457 | fd1edc036004532b2457a985eaffb24be9f325de1ba622a238c9587a5828eb68 |

### Files changed (`Experiment 27/Experiment Notebooks/Build/`)

| File | sha256 | Change |
|---|---|---|
| `build_e27_nbdf.py` | b3ad1ea8… | Q-only ppack skip; the K100 gate hook |
| `nbdf_lib.py` | 7a13f93f… | `draft_load_check` parser |
| `nbdf_k100_unit_tests.py` | 6b096a99… | K6 matrix, K7 emit-to-parse |
| `build_e27_nbdf_cpu_kit.py` | 231e455e… | sidecar step, attention-type fix, 540 record-only, setup check key |
| `build_e27_nbdf_k100.py` | d55314ba… | `k100_gate_extra` (unchanged this round) |
| `nbdf_gate_verify.py` | 05989f95… | new audit check after a gated rebuild |
| `nbdf_cell7_body.py` (generated) | 8507aaf2… | W4 body |
| `nbdf_k100_cell7_body.py` (generated) | b5dc2237… | K100 body |

### Audit zips (refreshed, in `Experiment Review/Audit Zips/`)
- K100: `E27_NBDF_K100_audit_FLAT_2026-11-01.zip`, sha d9be8427593768466c01ac79ab9d4940e4790aacf0b941401fb221c2fa9953d3
- W4: `E27_NBDF_audit_FLAT_2026-11-01.zip`, sha 54b902a9…

### Gated rebuild commands
Run from `Build/`, and only after the new real kit for that notebook prints ALL OK.
```
python build_e27_nbdf_k100.py --kv-gate-report "<K100 real report>/nbdf_kit_real_report.json"
python nbdf_gate_verify.py nbdf_k100_build_facts.json nbdf_k100_cell7_body.py D3s D4s
python build_e27_nbdf.py --kv-gate-report "<new W4 real report>/nbdf_kit_real_report.json"
python nbdf_gate_verify.py nbdf_build_facts.json nbdf_cell7_body.py D3s D4s D4 D4t D7 D4b
```
`nbdf_gate_verify.py` asserts:
- `passed` and `gpu_ready`;
- the embedded CELL 7 `KV_KIT_GATE` equals the build facts;
- `per_k[arm] is True` for each arm;
- for K100 also `per_k_success[arm].success is True`.

It never edits anything.
