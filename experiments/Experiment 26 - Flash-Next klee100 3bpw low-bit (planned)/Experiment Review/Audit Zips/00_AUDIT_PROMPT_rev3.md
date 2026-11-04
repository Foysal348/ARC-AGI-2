# Audit prompt: klee100 GPU load notebook, rev 3 (2026-11-01)

## Context
The K3_ONLY GPU run v1 (Kaggle 4×L4, Merlin v3.0.0) loaded klee100 cleanly: ready in 468 s on the first rung (5,270,142,976 B per GPU), 1,096,865 KV tokens, chat probes 3/3, MTP acceptance 4.00. It still ended `K3_FP8_CONFIG_MISMATCH`, because of the read-back probe:
- every layer read `["HummingMoEMethod", null, null]`;
- the MTP drafter was not found (MTP map `{}`);
- the raw per-rank rows were not saved.

Files: `E26_K3ONLY_GPU_v1_2026-11-01.md`, `gpu_v1_k100_results.json`, `gpu_v1_k100_verdict.json`, `gpu_v1_server_K3.log`. The notebook that ran is `Q27-XHIGH-E25_KLEE100_LOAD_K3ONLY_RAN_v1_40c9bde3.ipynb`.

## What rev 3 changes
**1. Bits / group (worker extension, `K100_EXT_ADD` in `build_e25_klee100_load.py`).**

v1 asked `HummingMoEMethod.quant_config.config_parser`. That object is a `HummingLayerQuantizationConfig`, which has no `config_parser` (`merlin_humming.py:343-360`), so bits came back null.

Rev 3 reads every source that exists, records all of them in `sources`, and flags any disagreement as `bits_conflict`:
- `method.weight_schema`: `merlin_humming.py:646`. This is the humming `GPTQWeightSchema(bits, group_size)` (`humming_schema_gptq.py:12-17`), built by `merlin_inc_wna16_scheme.py:299-306`.
- `method.quant_config`: the `AutoGPTQConfig` of the MTP W4 experts, `weight_bits` / `group_size` (`merlin_auto_gptq.py:143/147`). It is read only when that config has no `config_parser`, so the INCConfig's global default (2 / 64) is never taken as a layer's bits.
- `inc.config_parser.resolve`: the module's own INCConfig (`merlin_routed_experts.py:112/114`), called as `resolve(module, module.layer_name)` (`merlin_inc_config_parser.py:64`). This is the same call that `INCConfig.get_quant_method` makes (`merlin_inc.py:336`).
- `layer.weight_schemas.{w13,w2}`: the converted `HummingWeightSchema` (`merlin_humming_moe.py:620`), read as `b_dtype.num_bits` / `weight_scale_group_size` (`humming_schema_humming.py:16-20`).

**2. MTP drafter.** Merlin uses the V2 model runner ("Using V2 Model Runner" in the log). The drafter is `get_draft_model()` / `speculator.model` (`merlin_v2_model_runner.py:378-384, 690-697`; `merlin_spec_decode_speculator.py:343`), and it exists only on the last PP rank. `model_runner.drafter.model` (the V1 runner path) is kept as a last fallback. The path that was used is recorded per rank.

**3. Raw rows.** `k100_facts` stores the RPC result as `method_map_raw` in the boot record, and so in `k100_results.json`.

**4. Verdicts (`method_map_check` in `k100load_lib.py`).**
- `GPU_METHOD_MAP_PASS`: everything matches, and bits were read.
- `METHOD_CLASS_ONLY`: bits / group are unreadable on every row of a family (routed or MTP), and every class matches. Then `gate_ok` is True and `ok` is False, it is not `CONFIG_MISMATCH`, and the label gets ` + METHOD_CLASS_ONLY`.
- `GPU_METHOD_MAP_FAIL`: a class mismatch, a read bit that is wrong, `BITS_SOURCES_DISAGREE`, `BITS_PARTIALLY_READ`, missing MTP, or any of the earlier rank / PP checks failing.

**5. KV scales probe (record only).** `k100_kv_scales` now matches the QSA `layer_name` (`merlin_qsa.py:531/662`). In v1 it found 0 modules.

**6. K3 pin ladder raised (user request).** `kv_cache_memory_bytes` is per worker GPU (`merlin_config_cache.py:260-267`).
- New rungs (`k3_measured_rungs`) = v1 pin + (min free − leave) MiB, rounded to 2 MiB. The v1 free memory was 4583 / 4579 / 3257 / 3257 MiB. With leave = 0.7 / 1.7 GiB this gives 7,933,526,016 and 6,859,784,192.
- The ladder (`k3_ladder`) is 7,933,526,016 → 6,859,784,192 → 5,270,142,976 → 3,553,580,524 → 2,479,838,700. Every rung is tried, from index 0.
- The K3_ONLY hard stop is raised from 100 to 115 min.

**7. CPU kits.**
- The real kit has a new MM check. It runs the notebook's own probe on real Merlin objects in the Merlin venv, and checks the source anchors.
- The units kit carries the v1 method-map record as T25's fixture.

## Please check
1. Does every attribute path match the Merlin source? Can `bits_conflict` fire falsely on a correct model, e.g. through a group representation that differs after humming conversion?
2. Is `METHOD_CLASS_ONLY` scoped correctly (per family, every row)? Does any real mismatch now pass?
3. Is the drafter lookup on PP0 / PP1 right?
4. Is 0.7 GiB left on the PP1 cards at 7.39 GiB enough for TF prompt-logprob transients? The fallback is a single body-OOM retry at 6.39 GiB.
5. Are the replayed time budgets (T22) right with two extra launch-OOM rungs?
