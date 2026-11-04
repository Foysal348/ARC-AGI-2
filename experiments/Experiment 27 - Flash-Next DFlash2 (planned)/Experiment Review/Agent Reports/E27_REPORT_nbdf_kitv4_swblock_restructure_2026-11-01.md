# E27 NBDF agent (acfe40bd525f7510b): kit v4 reading + SW-1024 restructure, verbatim

## D4s / D3s on the 1024 drafter block are now the primary DFlash arms; notebook rebuilt with the gate closed, kits refreshed for v5 (nothing pushed)

**Kit v5 must run on this notebook:** `9353c0d62cb3b5fb90caf630c070b2f45c8af51f24d36985264dfa2de70c409a`. The closed-build sha is the same value. It is 392,712 B, `gpu_ready: false`, and the NB4T / E25 files are unchanged.

**Tests (local):**
- `nbdf_unit_tests.py`: **200/200 PASS**.
- `nbdf_build_gate_test.py`: **11/11 PASS**.
- Units kit, run once: **`@@KIT units ALL OK@@`** (198/198; the 2 export-layout checks need the notebook file, which the kit does not bundle).

The seam is still `7356d473`. The kit v4 report is now stale for the gate, as expected; after kit v5 run `python build_e27_nbdf.py --kv-gate-report <v5 nbdf_kit_real_report.json>`.

### Kit v4 findings (read in full)
**18 / 18 checks True**, run time 8.6 min:
- `cell0_identity_submission`, `inputs_found`, `merge_ok`, `sha256_all_ok`, `drafter_pins_ok`, `arms_q_d4_d3_d7_d4b`;
- `merlin_install_ok`, `merlin_flag_sources_equal`, `patch_60490_ok`, `seam_apply_verify_revert_ok`;
- `host_phase_all_5_ok`, `launch_check_all_5_ok`, `env_in_spawned_child_all_5`, `tree_check_all_5`, `argv_fields_ok`;
- `spec_config_preflight_ok`, `kv_cache_configs_ok`, `drafter_loaded_weights_ok`.

Merlin commit ad42e37 confirmed.

**R9 KV probe:** Q calibrated with **no page mismatches**. Group stride is per block; numbers are from PP1 rank 2.

| Run | Pin (B) | Blocks | Groups | Max conc. at 131k | Drafter group |
|---|---|---|---|---|---|
| Q | 3,553,580,524 | **215** | 5 | 6.11 | none |
| K3 / K4 / K7 BF16 | 3,016,709,612 | **71** | 6 | 1.84 | SlidingWindowSpec, 5 layers, block 4096, **41.9 MB** page |
| K4 FP8 | 3,016,709,612 | **143** | 6 | 3.74 | 21.0 MB page |
| K4, SW block 1024 | 3,016,709,612 | **205** | 6 | 4.86 | 10.5 MB page (below the target's 14.2 MB, so the stride is unchanged) |

- Q's 5 groups match the GPU's rank-2 layout exactly: Mamba 5 / 6 / 6, CircularBuffer 7 (block 8), FullAttention + MLA 14. Wasted 6.19 MB per block; 6 metadata builders.
- On D the target groups are Mamba 5 / 6 / 6, CircularBuffer 6, FullAttention + MLA 12 (the MTP layer is gone), plus 1 drafter group. Wasted bytes per block: 6.19 MB (K3), 5.60 MB (K4), 3.82 MB (K7).
- No draft-group warning and no fine-grained-disabled warning in any run.

**R12 drafter loaded-weights check: PASS.**
- `DFlash2Qwen3ForCausalLM` built on meta in bf16, with 69 named parameters.
- The real patched `load_weights` emitted `draft_load_check` with expected 67 = loaded 67, dropping `draft_id_to_target_id` and `mask_embedding`.
- `false_raise_suspect` is False, so `DFLASH_LOAD_CHECK=0` is not needed.

### The restructure
**Arm order (`nbdf_lib.py:42`):** Q → D4s → D3s → D4 → D4t → D7 → D4b, each budget-gated.

| Arm | What it is |
|---|---|
| D4s / D3s (primary) | D4 / D3 + `DFLASH_DRAFT_SW_BLOCK=1024`; full item set: smoke + ladder, oracle, long oracle, prefix 2 × 3, all windows, profiles |
| D4 | the plain BF16 control at block 4096 (71 blocks); a short arm: smoke + oracle + 8k × 2; verdict `SHORT_CONTROL_<8k vs Q>`, never a winner |
| D4t | moved onto the SW base: D4s + `DFLASH_CONV_IMPL=torch`; short; compared with D4s (`CONV_BASE`) |
| D7 / D4b | unchanged |

The 8k throughput rule (`SHORT_EXIT_ARMS`) now applies to D4s.

**SW-block safeguards:**
- `SW_BLOCK_ARMS` (`nbdf_lib.py:45`) is D4s / D3s / D4t, and the value must be exactly 1024. The plan check refuses anything else, and the launch check refuses the env on every other arm (the variants inherit their parent's rule).
- The primaries still have to pass the token-ID oracle vs Q, the 90k / 116k long oracle, the concurrency ladder and the prefix cold / warm test.
- **Read-backs:**
  - the engine block from "Setting attention block size to N tokens", else the FullAttention spec block (`engine_block`, `nbdf_lib.py:533`);
  - the KV layout;
  - the seam's `draft_sw_block` lines, which must show `draft_block` 1024 on SW arms (`draft_sw_block_applied`, `:965`); such a line on a non-SW arm is a contradiction.
- `SW_BLOCK_LOWERED_ENGINE_BLOCK` is recorded when the engine block drops below 4096 (`:1702`).

**Block-size-aware residency:**
- The NB4T library computes F assuming 4096-token blocks.
- `f_blocks(pool, mc, block)` (`:526`) recomputes F at the boot's real block, used in `band_level` (body `:613`).
- The prefix test sizes its turns on the engine block.

**Gate per-K summaries (`KV_RUN_OF`, `:1413`):**
- D4s and D4t use `K4_swblock1024`; D3s uses the new `K3_swblock1024` run, added to R9.
- D4 uses K4, D7 uses K7, D4b uses K4_fp8.
- Required: D4s and D3s (R9 `ok` now means Q calibrated plus both SW runs). The others are skipped with `SKIPPED_NO_KV_*` if their summary is missing.

**Pins:**
- SW arms start at Q's final pin (3,553,580,524) and step down on a launch OOM or the 512 MiB gate, up to 3 rungs (3,285,145,068 → 3,016,709,612 → 2,748,274,156) (`SW_OOM_RETRIES`; body `:917-921`).
- A later SW arm starts at the pin that worked (`SW_PIN`).
- The non-SW D arms keep the `pin_plan` D pin.
- **Expected cost:** the PP1 memory plan predicts Q's pin is about 0.7 GiB short of the 512 MiB margin. The first SW boot will most likely step down 2-3 rungs, about 5-7 min per failed boot, and that comes out of the 150-min budget. You chose this ladder; I'm flagging the likely cost.

**Budget (T10):** at 150 min, Q 50 + D4s 41 + D3s 40 are planned. D4 and D4t run only when the earlier arms finish faster than their minimums.

### New shas (all LF)
| File | sha256 |
|---|---|
| `build_e27_nbdf.py` | `6be503f89b1c41cd816b334ad40ea6abd6eec75f29b75e2f50f43644d5440a90` |
| `nbdf_lib.py` | `5a1937b4c3a3b173906b0e4f4e8bbe8cf00d96d7aa23f0a07764cc46efbc46dc` |
| `nbdf_cell7_body.py` | `f799778ff39bf62622d7ccfe7f50f522eee1ea73f4b45ac2ca58325ca08df567` |
| `nbdf_code_export.py` | `c1427a82394c6ffc881f77da9ffacf4fb9ece973ba80a1adecb9da636008fdec` |
| `nbdf_unit_tests.py` | `adb062667ca0373d163c68c5e32f290d9580df0e0ba5d2ed5c9f190f1befdd7a` |
| `nbdf_build_gate_test.py` | `6d5953acbddb2e89706e3e0993d9ba22b206c83d313858b6a50eb65d020c0e86` |
| `build_e27_nbdf_cpu_kit.py` | `a47daaff823e665c7bfafd67177aebf043fb7c14209b49294ccec73b3e7c1a90` |

**Kit folders**, under `D:/ARC AGI 2/_scratch/push_e25/`, on notebook `9353c0d6`:
| Folder | Size | sha256 |
|---|---|---|
| `q27-e27-nbdf-kit-units__foysalemonshanto` | 354,729 B | `05fc5637…` |
| `q27-e27-nbdf-kit-real__foysalemonshanto` | 281,014 B | `b2b33842…` (R4 covers all 7 arms; R9 adds `K3_swblock1024`) |
| `q27-xhigh-e27-nbdf__foysalemonshanto` (GPU) | — | this notebook |

The audit zip `Experiment Review/Audit Zips/E27_NBDF_audit_FLAT_2026-11-01.zip` is refreshed: 750,028 B, `5af0ea71…`.

### Open risk
R9 does not model the engine-wide block drop (`v1/engine/core.py:436-448`). On the GPU the target attention groups will also move to 1024-token blocks, so the real D4s / D3s pool, F and groups will differ from the kit's 205 blocks. The verdict compares capacity on the measured numbers, and residency uses this boot's real block size.
