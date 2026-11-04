# E27 seam agent (aefbfc6004f4dc34f): final report, verbatim

## Result: `dflash_seam.py` + `dflash_seam_tests.py` are built, and all 520 local CPU checks pass (13.5 s; two full runs, both ALL PASS)

**Files** (LF only, ASCII only, English):
- `D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build/dflash_seam.py`
  - 1516 lines, sha256 `c4ea1ecc287036bcadbc98cd33e40309d17f2c58b471e5c876ea7acddba51f78`.
- `D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build/dflash_seam_tests.py`
  - 1733 lines, sha256 `1fdde1e647be1f79a8c2c3b6a0d3ef2c4b213b39c54dc21780c30767d27c7676`.
- The generator that produced the diffs lives in the scratchpad and is not part of the deliverable. It holds the exact old→new edits, and difflib turns them into strict diffs.
  - `D:/DevCache/Temp/claude/d--ARC-AGI-2/1c6db7aa-c78b-4b3a-b97a-5c67187fe52f/scratchpad/e27/gen_dflash_seam.py`
  - plus `gen_edits.py`, `gen_edits_dfu_att.py` and `template_dflash_seam.py` in the same folder.
- `build_e27_nbdf.py` and `nbdf_lib.py`, in the same Build folder, are not mine. I did not touch them.

**What I did not do:**
- No git, no Kaggle, no downloaded code executed. The roozich and incoai patches were only read.
- The Merlin source copy is unchanged (`git status` is clean).

**Structure** is the nb4_ppack structure:
- `PATCHES` = {rel: (pre_sha, post_sha, marker, diff)};
- the same strict no-fuzz `apply_unified` / `reverse_unified`;
- atomic writes (`.e27tmp`);
- `apply_dflash_patch` / `verify_dflash_patch` / `revert_dflash_patch(tree, py, ring=True, draft_block=True)`, with `py_compile` and pyc drop;
- refusal on a foreign or mixed tree, with nothing written.

New helpers in the module:
- `tree_state(tree)` → ORIGINAL / FULLY_PATCHED / INVALID_MIXED;
- `combined_tree_state(tree, nb4)`, which reports the nb4_ppack state alongside;
- `server_env(...)`;
- `embed_row_sha(path, row=248077)`, which reads one row of the target checkpoint via `get_slice`;
- `parse_log` / `readback_label` for the `E27_DFLASH {json}` lines.

## Files, shas and every hunk
Merlin ad42e37, tree 07c16a79. Line numbers are pre-patch.

| File | pre sha256 | post sha256 |
|---|---|---|
| `vllm/model_executor/models/config.py` | 1a16138305f395f10e87b1ec9de40d501eba2a3c0f1379a6c787bb5c2d2e61ac | 0f6f63d355a269743cf8b6c0c2ef2b43aad066f5c1d202e534b02d3f44ff2711 |
| `vllm/models/qwen4_exp/nvidia/model.py` | 068adee6c0e062342765d10e1e22da5967c428059197067138064b137f973397 | fee6564366f4b26b66d664f032e1b7d3339a96c507f929489ea4d4543d597438 |
| `vllm/model_executor/models/qwen3_dflash.py` | 5375543288e72219abc1fc777219a640f0d173b3e8ac531718d666483a46ea7a | 774201928124445e72052ead3df482ce95a414e5467b3a8db3caca838fd14e24 |
| `vllm/v1/worker/gpu/spec_decode/dflash/utils.py` | f0584417149c0a6fa67e700b2a725e40a41f4546a806b26e6cc64b64c59cfd66 | 5b3cd014ce1272d3a8f4bbea9b7333850378030765c638292aea8deea92dcb5d |
| `vllm/models/qwen4_exp/common/qsa_cache.py` (ring, optional) | 43f323bb347e2152e47acb36a99d0682eaebc5cdafe9b5ba168b5c4b4e9acf13 | 07b0b61f9947d36ab2b3f87958830aaf591e899a93ab61f67c4debc50c442bde |
| `vllm/model_executor/layers/attention/attention.py` (drafter SW block, optional) | bde77677fd0e95ea09c6a580dfaf09b57741e8183e00cb106e2d1cc64d8e03f2 | 5817d764b870409ef0ef5b0c9e619f532a6410cece82833bd464dcea69181a5c |

`pp_utils.py`, `model_runner.py` and `gpu_worker.py` are not touched, so nb4's strict PRE shas stay valid.

### `config.py`
- **@@-1047,14:** the allowlist admits `"dflash"` only when `DFLASH_SEAM=1`.
  - When the env is unset, the set and the error message are the original ones.
  - When it is set, refused methods keep the original message plus a suffix.

### `nvidia/model.py`
- **@@-40,11:** imports `EagleModelMixin` and `SupportsEagle3`.
- **@@-99,6:** imports `grouped_gemma_rmsnorm` from `.ops.hc` (used only by the `norm_input` tap mode).
- **@@-564,8:**
  - `class Qwen4ExpModel(nn.Module, EagleModelMixin)`;
  - `supports_aux_hidden_states_over_pp = True`;
  - `_e27_local_aux_ids` and `_e27_tap_mode` defaults.
- **@@-669,6 (+152 lines):**
  - `_set_aux_hidden_state_layers` override:
    - raises unless `DFLASH_SEAM=1`;
    - requires ids within 0..48 and non-empty;
    - reads `DFLASH_TAP_MODE` once;
    - calls EagleModelMixin's setter (the PP layout cache);
    - records the local ids;
    - logs one `aux_layers` line per rank with local ids, slot_base and upstream_total.
  - `_e27_next_hc`, `_e27_dflash_tap` and `_e27_check_aux`. The check raises `DFLASH_AUX_COUNT_MISMATCH` on a wrong tap count and `DFLASH_AUX_WIDTH_MISMATCH` on a width other than 2560.
- **@@-724,6 (forward start):** when taps are set, `remote = collect_remote_aux_hidden_states(it)`; id 0 is the embedding.
- **@@-776,6 (layer loop):**
  - for tap id `layer_idx + 1`: `layer.mlp_hyper_connection.combine(...)`, then `.unflatten(-1, (4, 2560)).mean(-2)`;
  - placed after the deepstack addition;
  - with HC-SP on, `sp_all_gather(tap)[:full_num_tokens]`.
- **@@-783,6 (PP hand-off):** non-last ranks return `IntermediateTensors({"hidden_states", **pack_local_aux_hidden_states(local)})`.
- **@@-817,6 (last rank):** returns `(hidden, remote + local)`, remote first.
- **@@-873,6:** `Qwen4ExpForCausalLM` gets `SupportsEagle3`.
- **@@-1163,6:** `Qwen4ExpForConditionalGeneration` gets an explicit `SupportsEagle3`. This is the served class; it also inherits the protocol via Qwen3VL.
- The protocol's default setter and the default-layers method are used unchanged (Merlin's name is `get_eagle3_default_aux_hidden_state_layers`).

### `qwen3_dflash.py`
- **@@-768,8:**
  - sidecar helpers: `_e27_log`, `_e27_mark_embed_loaded`, `_e27_check_loaded`, `_e27_with_embed_sidecar`, `_e27_embed_sidecar_iter`, `_e27_read_embed_sidecar`;
  - an explicit `has_own_embed_tokens = False` on `DFlashQwen3ForCausalLM`. This is the same value it already inherits, so behaviour is unchanged, and it means Merlin's silent early return under PP can never be taken.
- **@@-884,6:** `load_weights` wraps the weights with the sidecar fallback.
- **@@-917,13:**
  - records `_e27_embed_loaded`;
  - keeps the loader's returned set and checks that every expected drafter parameter got a tensor. `DFLASH_LOAD_CHECK=0` turns this off.

### `dflash/utils.py`
- **@@-11,6:** `_e27_release_module` and `_e27_finish_draft`.
- **@@-56,6:** remembers the draft's own embedding before the share.
- **@@-72,4:** calls `_e27_finish_draft` after the lm_head share. It does the following:
  - **Verify embedding:** it must be the target's module, or own *and* loaded. The weight must be 2-D and not meta, and a 4096-row sample must be finite and not all zero.
  - **Verify lm_head:** shared or own.
  - **Linears:** every drafter linear must use `UnquantizedLinearMethod`.
  - **Mask row:** the sha256 and norm of row `mask_token_id` are logged on the TP rank that holds it; with `DFLASH_MASK_ROW_SHA` set, a mismatch raises.
  - **Release:** it frees the replaced draft lm_head (B5, about 636 MiB per card) and, after a PP1 share, the replaced draft embed_tokens. The method is the `_release_replaced_draft_head` one: clear the dicts, `gc.collect`, `empty_cache`.
  - **Host embedding:** `DFLASH_DRAFT_EMBED_HOST=1` puts the drafter-only embedding in pinned host memory with a UVA view (the `_host_embed_tables` pattern). It does not use `VLLM_HOST_EMBED_TABLE` and has no effect on the target or MTP tables.
  - **Log:** one `embed` line.

### `qsa_cache.py`
- **@@-886,6:** an upstream-style `qsa_ring_capacity(ratio, K, block)`, own text with PR #54912 semantics:
  - the minimal ring when it divides the block;
  - otherwise the next multiple of the ratio that divides it, up to 2× the minimal ring;
  - otherwise an informative ValueError.
- **@@-959,6:** used only with `DFLASH_QSA_RING_WIDEN=1`. With the env unset, the original code and assert run.
- At block 4096: K3/K4 → 8 (identical to the original), K5..K8 → 16.

### `attention.py`
- **@@-130,6 and @@-634,6:** `DFLASH_DRAFT_SW_BLOCK=<n>` changes the KV block of drafter sliding-window layers (those with `is_draft_layer`).
  - n must divide 4096, so the LCM is unchanged; otherwise it raises.
  - With the env unset, the original block choice is kept.

## Embedding decision
- **Primary is option (a), the repacked drafter with `embed_tokens.weight`.** It needs no loader code: Merlin's own path loads it and sets `has_own_embed_tokens=True`.
- **Option (b), the sidecar, is an env-gated fallback only.**
- **"Silent garbage" is refuted (agreeing with report B).** In Merlin, `maybe_share_target_embed` raises "needs the target input embedding" under PP2 when the embedding is missing. Test T7 runs the real function.
- Our explicit class attribute plus the load check make the outcome loud in every case. That includes the no-attribute early-return path, which the original passes silently; the test shows that.
- Merlin's native `spec_decode_needs_target_embed` pattern is noted in the module but not used.

## Composition with nb4_ppack (Q-defer)
- The file sets are disjoint; an overlap is refused at import.
- Both apply orders give the same 9 file shas, each patch reverts on its own, and the runner's aux relay and setter lines survive nb4's patch (T10).

## Test counts, 520 total, all pass

| Section | Checks | What it covers |
|---|---|---|
| T1 | 69 | apply / verify / revert, exact shas, LF, idempotent, refusals, no fuzz, subsets, every hunk has E27, ast.parse at 3.12, `tree_state` |
| T2 | 36 | AST interfaces on both served classes; an MRO mimic of the real class graph builds for the original and patched trees; Merlin's `_hc_combine` is pure; the tap has no in-place ops |
| T3 | 90 | Meta forward on the real `Qwen4ExpModel` methods + real EagleModelMixin / SupportsEagle3 / eagle3_utils; tiny hc 4 × H 8, 48 layers (see below) |
| T4 | 17 | config allowlist |
| T5 | 209 | ring: env off identical over 195 cases; widening vs brute force; K3/4/5/7 at 4096; informative error |
| T6 | 29 | drafter load: repacked embed, loaded-weights check, sidecar matrix, `embed_row_sha` |
| T7 | 26 | `load_dflash_model` with the real `maybe_share_target_embed`; release; mask-row sha; quantized linear; host guards; env-off identity |
| T8 | 12 | drafter sliding-window block |
| T9 | 13 | readback |
| T10 | 19 | nb4 composition |

T3 in detail:
- With taps unset: bitwise identical outputs and op traces for PP1, PP2 stage 0/1, HC-SP and deepstack.
- Real dflash config → ids (6, 16, 26, 36, 46). Each tap is bitwise equal to the mean over hc of an independently computed completed layer output.
- PP2 at 25/23:
  - stage 0 sends keys `aux_hidden_states_0/1`, and the real reserve gives the same keys as [B, H] slots;
  - slot base is 2 and the upstream total is 2;
  - stage 1 returns remote first, [6, 16, 26, 36, 46], bitwise equal to PP1; a permutation is detected.
- HC-SP gather, boundary ids 0/25/26/48, the tap modes, and every fail-loud path.
- Real shapes (hc 4, H 2560) against a per-stream reference, inputs unmutated, and the SP gather.

## Open risks and what is not covered
1. **Merlin's `get_kv_cache_configs` (pool / F / groups) was not run.** It needs an installed Merlin and the real per-stage specs; there is no vllm locally. This is the kit gate, per K: the GDN / RecoverSSM / PLE state grows with K (`nvidia/model.py:985-1040`). It should also cover each `DFLASH_DRAFT_SW_BLOCK` value, because the effect of the drafter-block option on packing is unmeasured.
2. **Compiled forward with the PP aux relay is new in Merlin.** None of DeepSeek-V4, Kimi-K3 or MiniMax-M3 is `@support_torch_compile`, and Qwen4ExpModel is. The FULL / PIECEWISE capture of PP0 with the extra IntermediateTensors keys must be confirmed by a kit import plus a GPU boot.
3. **The tap definition matches incoai's capture.patch as read, but there is no SGLang source.** `DFLASH_TAP_MODE=pre_combine|norm_input` is there for a GPU A/B if acceptance falls short of MTP.
4. **The loaded-weights check may raise falsely** if Merlin's AutoWeightsLoader returns names that differ from `named_parameters`. It is tested only with a fake loader; `DFLASH_LOAD_CHECK=0` is the escape hatch. The kit should confirm it on the real drafter.
5. **`DFLASH_DRAFT_EMBED_HOST`:** the UVA path itself needs CUDA. Only its guards are tested on CPU.
6. **PP1 with the repacked drafter and a host-pinned target embedding:** Merlin's `_should_share` would `torch.equal` a GPU tensor against a CPU tensor. That is a Merlin edge case and not ours to fix here; our runs are PP2.
7. **Launch settings that need no patch** (documented in the module):
   - `--speculative-config` with `"method":"dflash","kv_cache_dtype":"auto","attention_backend":"FLASH_ATTN"`;
   - capture sizes up to max_seqs × (1+K);
   - the expected sha for `DFLASH_MASK_ROW_SHA`, computed in the notebook from the served checkpoint with `embed_row_sha`.
8. **Two reports were wrong on two points:**
   - A said the missing embedding is "silent"; Merlin actually raises (B was right).
   - The feasibility report said the drafter has its own smaller block; the drafter block is the primary 4096.

