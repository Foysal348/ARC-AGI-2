# E27 NBDF agent (acfe40bd525f7510b): fix 2 report (3 GPT audits, kit v2, M2-M4, M11), verbatim

## Every audit item is in; the notebook is rebuilt (gate still closed) and both kits are refreshed (nothing pushed)

Local results:
- **`nbdf_unit_tests.py`: 178/178 PASS.**
- **New `nbdf_build_gate_test.py`: 11/11 PASS.** It runs the real builder into a temp dir with synthetic kit reports.
- **Units kit, run once locally: `@@KIT units ALL OK@@`.** That covers nbdf 176/176 (the 2 export-layout checks only run where the notebook file exists) and the seam tests 555/555.

The new seam is embedded: `dflash_seam.py` `2df1da39495416b574a98317c9aef9da477354c50e42c18b2550c73807c9eed6`, tests `a46ea2db…e64e`.

**Size is now tight:** the notebook is 397,812 B, only about 2.2 kB under 400 kB. Any further addition needs a trim first; the markdown cell or the FLAGS text are the easiest places.

### B1: a closed gate skips every D arm (expected)
The gate stays closed until a passing real-kit report is fed to the builder; this build is `gpu_ready: false`.

To make the binding checkable, `build_e27_nbdf.py:1535` `assemble(kv_gate)` always builds the closed-gate notebook first and records its sha (`closed_build_sha256`, `a40be02b…`). That sha is what the kit runs.

The regression test the audit asked for is in `nbdf_build_gate_test.py`:
- no report → closed;
- a passing report → `KV_KIT_GATE["passed"]` is True inside the built CELL 7 body, `gpu_ready` is true, D4 / D3 / D7 / D4b are all in ARMS, provenance is `closed_rebuild`, and the gated notebook differs from the closed one only by the gate;
- `drafter_loaded_weights_ok` False → closed;
- a stale notebook sha → closed (`STALE_NOTEBOOK_SHA`).

### B2: the AL early exit is gone; an 8k throughput rule replaces it
- **AL is a diagnostic only:** `nbdf_lib.py:1173` `al_vs_q`, recorded at `nbdf_cell7_body.py:760`. The `EARLY_EXIT` constant is removed. The early-exit block and its `stop_d` assignment are deleted. `stop_d` is now set only for a broken tree (body `:1170` and `:1176`).
- **Throughput rule:** `nbdf_lib.py:1183` `short_exit` is applied at body `:805`, for D4 only (`SHORT_EXIT_ARMS`).
  - It fires when both 8k@8 windows are valid and each is below Q's 8k mean by more than max(Q A/A spread, 2 %).
  - It skips only D4's own 40k / 90k / 116k windows (and its 116k profile), recorded as `SKIPPED_SHORT_EXIT`.
  - D3 always runs if the budget allows, then D7. `SKIPPED_BUDGET` is unchanged.
- **Tests:**
  - D4 AL 2.6 < Q 2.8, with a synthetic D3 faster than Q → D3 is WIN and the best arm.
  - D4 below Q at 8k → D4 is REGRESSION, with `short_exit` recorded.
  - A body check finds no AL stop and exactly 2 `stop_d` assignments, both for the tree.

### B3: the gate requires all 17 real-kit checks plus bound provenance
`nbdf_lib.py:1340` holds the `REAL_KIT_CHECKS` list; `:1359` `kv_gate_from_report` is called from `build_e27_nbdf.py:1564`.

**All 17 checks must be True:**
- `cell0_identity_submission`, `inputs_found`, `merge_ok`, `sha256_all_ok`;
- `drafter_pins_ok`;
- `merlin_install_ok`, `merlin_flag_sources_equal`, `patch_60490_ok`;
- `seam_apply_verify_revert_ok`;
- `host_phase_all_5_ok`, `launch_check_all_5_ok`, `env_in_spawned_child_all_5`, `tree_check_all_5`, `argv_fields_ok`;
- `spec_config_preflight_ok`;
- `kv_cache_configs_ok`;
- `drafter_loaded_weights_ok`.

**One exception, CPU_LIMIT:** accepted only for an R12 device / CUDA / platform error that is not a "got no tensor" raise. It is recorded in the gate. The GPU `draft_load_check` read-back on both PP1 workers is then a hard pass condition. I made that read-back a hard condition for every D arm regardless: body `:727` stops the arm with `NO_GO_DRAFT_LOAD_CHECK` before any window if it is missing.

**Provenance is bound, never inferred:**
- The report's notebook sha must equal the closed rebuild or the pre-rebuild file; which one matched is recorded.
- Seam sha in the report = the embedded seam sha.
- nb4_ppack sha = `8dadf076…`.
- Drafter: the file shas, the embed tensor sha and the mask-row sha must match the pins.
- Merlin source shas must equal the audited tree, recomputed by the builder.
- Identity from the report's own evidence: tp 2 and pp 2 from Q's argv, partition 25,23 from Q's env, Merlin commit ad42e37. The kit now records the commit from "every seam file at its ad42e37 pre-sha before apply" (kit `:629`).
- A per-K KV summary must exist:
  - K3 and K4 are required;
  - a missing K7 summary leaves D7 as `SKIPPED_NO_KV_K7`, and a missing K4_fp8 summary leaves D4b as `SKIPPED_NO_KV_K4FP8` (body `:1155`); neither fails the build.

**Tests (T12):** a passing report opens the gate. Each of these keeps it closed:
- `drafter_loaded_weights_ok` False with kv + seam OK;
- a "got no tensor" raise (never CPU_LIMIT);
- a stale notebook sha;
- a different seam sha;
- missing tp, partition or commit;
- different Merlin source shas;
- a different drafter sha;
- a missing K4 run;
- any single check False;
- a wrong ppack sha.

A missing K7 run leaves the gate open with D7 not runnable.

### HIGH-3: the memory gate is fail-closed
- `nbdf_lib.py:1252` `memory_gate`:
  - needs exactly 4 finite numeric readings (no bool, NaN or inf), else `(None, "INCOMPLETE_FOUR_RANK_MEMORY: …")`;
  - any rank below 512 MiB gives `(False, "LOW_MEMORY_RANKS:[i…]")`.
- **Runtime:** body `:662` `_mem_gate` retries a None reading once after 5 s. False steps the pin down. None is recorded and does not step down.
- **Verdict** (`nbdf_lib.py:1655`): a D arm needs both gates True (after ready, after the first decode) for WIN or SHORT_ONLY_WIN; otherwise it is `INCONCLUSIVE_MEMORY_GATE`, with the ranked verdict kept. For Q the gate is recorded in `base.memory_gate`, and Q is never skipped for it.
- **Tests:** None, [700, 700], [700, 700, 700, nan], [700, 700, 700, 0], bool and inf are not ok; [700, 700, 700, 700] passes. Arms with a missing or failed gate cannot be WIN.

### Prefix test widened to 2 prompts × 3 turns
`nbdf_cell7_body.py:548` `prefix_test`; gate at `nbdf_lib.py:1202`.

Per prompt (variants 40 and 41, 6,144 tokens):
- **Warm:** the cache reset, then:
  - turn 1: greedy 2,048 tokens, so block 2 fills with KV written by speculative steps;
  - turn 2: a filler sized so that its 512 tokens complete the next block;
  - turn 3: greedy 256 tokens, which reuses that block.
- **Cold:** the cache reset, then turn 2 alone; the cache reset, then turn 3 alone.
- The block size comes from this boot's read-back, not an assumed 4096.

Recorded per turn: acceptance, per-position rates, drafted / accepted, prefix-hit rate, and token ids cold vs warm. The gate is COLLAPSE when any warm turn's AL is below 0.8 × its cold AL, and PASS needs every (prompt, turn) pair measured with a warm hit.

### Kit v2's two failures (R9, R12)
**R9 KV probe** (kit `:430`):
- Calls the engine core's own `resolve_kv_cache_layout(vc, supported, specs)`, as `v1/engine/core.py:388-392` does, at kit `:556`. The supported layouts come from `get_supported_kv_cache_layouts` over the target's backends (Qwen4ExpQSAFlashAttention, QSAState) plus the drafter's (FLASH_ATTN, or TRITON_ATTN for D4b). Nothing is hard-set.
- Uses a real `ModelConfig` of the merged W4 checkpoint.
- Q must calibrate to 213-217 blocks at 3,553,580,524 before any D run is attempted.
- D runs K3 / K4 / K7 / K4_fp8 / K4 with a 1024 drafter block, at the D pin, with:
  - the K-dependent state rows scaled from source: GDN conv K+3, RecoverSSM replay K+1, PLE conv K+9;
  - the MTP layer removed on PP1;
  - the 5 drafter SWA layers (`model.layers.48-52`) added.
- Records per run: `kv_group_count`; for each group the spec, inner types, layers, block size, page size, `is_eagle_group` and drafter layers; `drafter_group_count`; `attn_metadata_builders`; wasted bytes per block where Merlin exposes them; max concurrency; and whether the draft-group warning and the fine-grained-disabled warning were logged.
- **This K-scaling is my reading of the source, not measured. Treat the D pool / F numbers as an estimate until the GPU read-back confirms them.**

**R12 drafter probe** (kit `:375`):
- Builds the draft config the way the engine does: a target `ModelConfig`, then `SpeculativeConfig(method="dflash", model=<dir>, K 4, probabilistic, kv auto, FLASH_ATTN, target_model_config, target_parallel_config)` (kit `:397`).
- The draft vllm_config is derived as `load_dflash_model` does it: attention_config non-causal plus backend, and the cache dtype.
- Then builds `DFlash2Qwen3ForCausalLM` on meta under the gloo group and runs the real patched `load_weights` with `DFLASH_SEAM=1`.
- Pass = a `draft_load_check` line and no raise. A "got no tensor" raise is flagged `false_raise_suspect`.

### Third audit and the seam update
- **Seam re-embedded** (2df1da39). The provenance seam sha is now the embedded one, not a hard-coded value.
- **ingraph_meta read-back:**
  - the seam's `readback_label` is called with `no_ingraph_meta=` (body, `dflash_rb`), and `nbdf_lib.py:462` `ingraph_meta` records the lines;
  - DFLASH_ZERO_ACCEPT with `kernel_map_refresh` true gives ONE relaunch with `DFLASH_NO_INGRAPH_META=1` (`nometa_arm`, `nbdf_lib.py:456`; body `:1177`), labelled `INGRAPH_META_FALLBACK` in the verdict and budget-gated;
  - the zero-accept canary stays, and that case does not also trigger the Q-defer isolation rerun.
- **vLLM #53786:** the "Disabling fine-grained prefix-cache hits" warning, the prefix-hit rate per window and per prefix turn are recorded. `PREFIX_REUSE_DROP` is flagged when D is more than 10 points below Q (recorded only).
- **Identity binding** and the **memory-telemetry retry**: covered under B3 and HIGH-3 above.

### Coordinator's missed items M2-M4 and M11 (all recorded flags; the tok/s windows decide)
- **M2, Q-defer at K ≠ 3:**
  - T13 unit test: nb4_ppack's real `nb4_pack_views`, extracted from its patch, round-trips byte-exactly with draft shapes [n, 3], [n, 4] and [n, 7] (n = 1, 3, 8), with the right sizes and offsets;
  - GPU: `QDEFER_K_FALLBACK` when a D arm's maximum Q-defer fallback share is above Q's (`nbdf_lib.py:502`).
  - **Gap:** the deferred-receive path's buffers are not exercised; the test covers the pack views only.
- **M3:** the "no KV cache group could be identified as the draft model's" warning is read back per arm and captured in R9 per run.
- **M4:**
  - per-window preemption counts, plus the SpecDecoding acceptance series of any window with preemptions (`nbdf_lib.py:512`);
  - `PREEMPT_ACCEPT_DROP` when the after value is below 0.8 × before;
  - **this is an approximation, labelled as such:** Merlin logs no preemption time, so "before / after" means the window's first vs last SpecDecoding interval.
- **M11:** target graphs per stage and mode plus the "Graph capturing finished" lines (`nbdf_lib.py:481`). `PP0_GRAPH_DEGRADED` is flagged when a D arm has fewer PP0 FULL graphs than Q, or none (`:493`).
- **#58638:** `KV_GROUP_EXPLOSION` is flagged when the D arm's group count on the drafter rank exceeds Q's plus the kit's drafter group count (`:1276`).

### New shas
| File | Lines | sha256 |
|---|---|---|
| `../Q27-XHIGH-E27_NBDF_DFLASH2.ipynb` | — | `a40be02b78b4e18369415ae4e2f5772d39fb81d4f3b46fe2729d519627eef9b6` (397,812 B, = `closed_build_sha256`; 9 NB4T files unchanged) |
| `build_e27_nbdf.py` | 1644 | `b0b17242efab7a7b39761f7a342638357ac3e5f9fd42326f245e3d136d98ed82` |
| `nbdf_lib.py` | 1765 | `f71e8b6ad7df0748c7e74b5d174cf2029b113b4fc26910c601f554baee65338d` |
| `nbdf_cell7_body.py` | 1230 | `1d01e2a6d9a19b5a2b7471e20bfc64ffd89a91bfaea4d680406e9bf0b3d9a450` |
| `nbdf_code_export.py` | — | `b31eb2f876c5c21c376cb61200b36d021f9800202c777e8e062b3c56c0cf5beb` |
| `nbdf_unit_tests.py` | 617 | `8a35249fe81f3250178e4d5e0eb637e24a4c87e4b5e62ea9e4171108a48e3cc0` |
| `nbdf_build_gate_test.py` (new) | 85 | `ae4eeca75c1b8c3a9e16d1301ebf2843d0a0412717ca325e9d3527e318b00225` |
| `build_e27_nbdf_cpu_kit.py` | 740 | `b7a2facf132f965701e645c7dbe673301994e56ad7240cece7c4cc6fb8c21a98` |

All files are LF.

**Kit folders**, under `D:/ARC AGI 2/_scratch/push_e25/`:
| Folder | Size | sha256 |
|---|---|---|
| `q27-e27-nbdf-kit-units__foysalemonshanto` | 344,929 B | `6b067f35…` (now also bundles the new seam's `dflash/speculator.py` and `v1/worker/utils.py`) |
| `q27-e27-nbdf-kit-real__foysalemonshanto` | 271,074 B | `5bf3fcb9…` |
| `q27-xhigh-e27-nbdf__foysalemonshanto` (GPU) | — | updated with the new notebook |

The audit zip `Experiment Review/Audit Zips/E27_NBDF_audit_FLAT_2026-11-01.zip` is refreshed with 28 files, including the gate test and the PR #64059 seam report: 725,918 B, sha `043e5010…`.

### Open items
1. **Kit v3 R9 / R12 are untested locally** (no vllm here). If `ModelConfig` of the merged checkpoint or the draft `SpeculativeConfig` still fails on the CPU box, R12 records it; a device / CUDA error is accepted as CPU_LIMIT.
2. **Final gated rebuild:** `python build_e27_nbdf.py --kv-gate-report <kit v3 nbdf_kit_real_report.json>`. Kit v3 must run on this exact notebook, `a40be02b`; any code change before then makes the report stale and keeps the gate closed.
3. **Budget estimates not revised:** each arm now has a longer prefix test (about +2 min) and the extra read-backs, but the per-arm `est_min` values (Q 47, D 37-39) were not raised. At the 140-min hard stop, D3 may fall to SKIPPED_BUDGET if Q and D4 run long. That is recorded, not a failure.
