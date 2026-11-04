# E27 NBDF agent (acfe40bd525f7510b): kit v3 reading + R9/R12 probe fixes, verbatim

## Kit v3 read in full; R9 and R12 probe fixes are in the kit builder; the notebook did not change

**Kit v4 runs on this notebook:** `487a4dc3ea982164e7be7e2666bcfd4b6a0c4b9f71cb578c01a936656e2934f9`. The closed-build sha is the same value. It is 391,297 B, `gpu_ready: false` (gate closed as designed), and the NB4T / E25 files are unchanged.

Both fixes are inside the kit's probe scripts only, so the notebook bytes did not move: the rebuild is deterministic and reproduced the same sha. Local units after the change: 190/190 PASS. Nothing was pushed.

### Kit v3 results (it ran the old notebook `a40be02b` with seam `2df1da39`)
**Units:** ALL OK. nbdf units 176/176 and the seam tests all pass.

**Real: 16 of 18 verdict checks passed; the 2 failures are `kv_cache_configs_ok` and `drafter_loaded_weights_ok` (both fixed below).**

| Check | Result |
|---|---|
| `cell0_identity_submission` | True |
| `inputs_found` | True |
| `merge_ok` | True |
| `sha256_all_ok` | True |
| `drafter_pins_ok` | True |
| `arms_q_d4_d3_d7_d4b` | True |
| `merlin_install_ok` | True |
| `merlin_flag_sources_equal` | True |
| `patch_60490_ok` | True |
| `seam_apply_verify_revert_ok` | True |
| `host_phase_all_5_ok` | True |
| `launch_check_all_5_ok` | True |
| `env_in_spawned_child_all_5` | True |
| `tree_check_all_5` | True |
| `argv_fields_ok` | True |
| `spec_config_preflight_ok` | True |
| **`kv_cache_configs_ok`** | **False** |
| **`drafter_loaded_weights_ok`** | **False** |

The run took 11.9 min.

**R10, spec preflight: OK.**
- Every spec key is a `SpeculativeConfig` field.
- method, `draft_sample_method`, KV dtype and backend are all valid; an unset draft TP falls back to the target TP.
- Merlin's own argparse parsed all 4 D-arm specs on the FakeL4 sm_89 platform.

**R11, patched-model import: OK.** The MRO is `Qwen4ExpModel → Module → EagleModelMixin → TorchCompileWithNoGuardsWrapper`. `aux_over_pp` is True, the compile wrapper is present, and the served class has `SupportsEagle3`.

**R9, KV probe: NOT calibrated.**
- The layout now resolved through the engine core's own call: supported layouts `[BLNHC, BLHNC]` → `BLNHC`.
- A real `ModelConfig` of the merged W4 checkpoint was built.
- **Q calibration: 161 blocks (target 213-217; the GPU measured 215).**
  - 4 groups: Mamba 9 layers at 18,874,368 B, Mamba 8 layers at 16,777,216 B, CircularBuffer 7 layers (block 8), FullAttention + MLA 14 layers at 22,020,096 B.
  - The GPU had 5 groups.
  - Wasted bytes per block 6,190,720; max concurrency at 131k 4.71 (GPU: 5.63); 5 metadata builders.
  - The draft-group warning and the fine-grained-disabled warning were both absent.
- Because Q did not calibrate, **no D run executed**, so there are no D pool / F / group numbers from v3.

**R12, drafter loaded-weights probe: failed before the check.**
- The draft config built fine: 82 checkpoint tensors with the embedding, draft architecture `DFlash2DraftModel`.
- `DFlash2Qwen3ForCausalLM` construction then failed: `ValueError: Selected backend AttentionBackendEnum.FLASH_ATTN is not valid for this configuration. Reason: ['dtype not supported']`.

### The two fixes (`build_e27_nbdf_cpu_kit.py`, sha `e89d2cc6f40ffded11f869fdbf528b655e67993aee14bf9fffd1527108eb0e95`)
**R12 cause:** the probe built the drafter's layers under torch's default float32. FA2 rejects float32.
- **Fix:** build under `set_default_torch_dtype(draft_model_config.dtype)` (bf16), which is exactly what Merlin's `get_model` does. The draft dtype is now recorded.

**R9 cause (a real bug in my probe):** the synthesized layer names were `layers.1000+i` on both PP stages, so PP0's and PP1's layers overwrote each other in the merged spec set. That alone gives fewer layers, wrong groups and the 161-block count.
- **Fix 1, real per-layer names per stage from the checkpoint's `layer_types`:**
  - GDN on the `linear_attention` layers (19 on PP0, 17 on PP1);
  - the QSA `attn`, `raw_key_cache` and `compressed_key_cache` on the `full_attention` layers (6 + 6), plus `mtp.layers.48.*` on PP1;
  - the PLE at `layers.1.ple`.
  - I checked this naming locally against the NB4T fixture; every rank's counts match.
- **Fix 2, spec fields the NB1 record does not carry:**
  - `tokens_per_state`, inferred from the recorded page size (the compressed indexer cache, ratio 4);
  - `mamba_type` as the real `MambaAttentionBackendEnum` member;
  - `mamba_cache_mode="align"` (Merlin's flag);
  - `tp_replicated=True` for the PLE layer (the GPU log says "folded replicated Mamba group (ple)").
- **New diagnostics:** `page_mismatch` lists every spec whose computed page differs from the recorded page, and `q_target` records the GPU's real group layout for ranks 0 and 2, to compare against.

### Kit folders (refreshed, not pushed)
Under `D:/ARC AGI 2/_scratch/push_e25/`, all on notebook `487a4dc3`:
- `q27-e27-nbdf-kit-units__foysalemonshanto`: 353,119 B, `4a563a56…` (unchanged).
- `q27-e27-nbdf-kit-real__foysalemonshanto`: 279,420 B, `4c257fb0…`.

The audit zip `Experiment Review/Audit Zips/E27_NBDF_audit_FLAT_2026-11-01.zip` is refreshed: 744,976 B, `de8f1345…`.

### Remaining risk for kit v4
- **R9 may still not calibrate to 215.** The NB1 record has no `kv_quant_mode`, `num_speculative_blocks` or MLA cache-role fields, and I could not run the probe locally (no vllm here).
  - If it misses again, `page_mismatch` and `q_target` will show which spec or group differs, and I'll fix from that.
  - The fallback is a meta-device build of the real target layers.
- **R12 may hit a later CPU-only limit** inside the drafter build. A device / CUDA error is recorded as CPU_LIMIT, which the gate accepts, and the GPU `draft_load_check` stays a hard condition for every D arm.

After kit v4 passes, run the gated rebuild: `python build_e27_nbdf.py --kv-gate-report <v4 nbdf_kit_real_report.json>`.
