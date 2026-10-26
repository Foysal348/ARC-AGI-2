# VMS rev 2: GPT pre-GPU serving audit, disposition for the vLLM 0.32.0 engine-matrix notebook

Date: 2026-10-26. Scope: only `Q27-XHIGH-VMS_ENGINE_MATRIX_VLLM.ipynb` (builder `Experiment Notebooks/Build/build_vms.py`,
patch `_scratch/vms/patch_vms_rev2.py`). Audit sources: the two READMEs in this folder (sections V1-V6).

Every claim was checked against the vLLM v0.32.0 tag source in `_scratch/vllm_whl/src032/`:
- `vllm/envs.py`
- `vllm/config/kernel.py`
- `vllm/engine/arg_utils.py`
- `vllm/model_executor/models/qwen3_5_mtp.py`

Issue #58807 was also read, and the RedHat checkpoint's real config and safetensors index were inspected.

## Changed in rev 2

| # | Audit item | Source check | Change |
|---|---|---|---|
| 1 | V1 / #53912: hybrid MTP + prefix caching can return empty or punctuation output with a normal finish | Not reproduced on our stack. The rev 1 reference (MTP 3 + prefix caching) was also the gate reference, so a cache-hit fault in it would have been invisible | New oracle arm `fp8_target_only_noprefix` (FP8, no speculation, `--no-enable-prefix-caching`). Its A/A becomes the gate reference for every later arm. The production reference is judged against it from rows it already has (`VMS["ref_vs_oracle"]`). If the oracle A/A has empty or degenerate rows, the gate stays on the reference and the report says so. The rev 1 `fp8_target_only` arm (prefix on) is replaced by the oracle |
| 2 | V1 detection: repeated punctuation, empty output | — | `vms_degenerate()` counts outputs that are empty, contain a run of 32+ identical characters, or (at 200+ characters) have < 8 distinct characters or < 10 % alphanumerics. The first harness run caught that a distinct-character ratio (< 4 %) flagged every long normal text, so it is an absolute count. It is applied to every gate (an arm fails if it has more than the oracle's A/A) and to every window's sampled text. The 40k band shares a 36k system block, so it is a prefix-cache-hit test under MTP |
| 3 | Decision rule | — | A gate FAIL is recorded and never adopted, but the speed is still measured. Rev 1 skipped the windows on a FAIL |
| 4 | V6: `VLLM_TEST_FORCE_FP8_MARLIN` removed (PR #52182) | Confirmed: the name is not in v0.32.0 `envs.py` (only `VLLM_MARLIN_USE_ATOMIC_ADD` and `VLLM_MARLIN_INPUT_DTYPE`). `--linear-backend` exists (`arg_utils.py:1808-1810`; choices `config/kernel.py:244-266`, including marlin and triton). A layer type without the requested backend falls back to auto | `fp8_mtp3_marlin` is dropped (recorded in `VMS["dropped"]`) and replaced by `fp8_mtp3_lb_marlin` and `fp8_mtp3_lb_triton`. Every launch records the "Selected / Using ... kernel" lines from the server log (`vms_kernels`), and the report lists them per arm, so a silent fallback is visible |
| 5 | V3 / #58807: W4A16 + dense BF16 MTP load crash | #58807 (v0.30.0, a different W4A16 checkpoint) failed because MTP was missing from `quantization_config.ignore`. v0.32.0 builds the predictor under prefix `mtp` with the target's quant config (`qwen3_5_mtp.py:93-105, 233-235`). RedHat's config has `re:^mtp.*` in `ignore`, and its index has 15 `mtp.*` keys, all dense `.weight` with 0 packed | Static check `vms_int4_mtp_static()` runs on CPU before any INT4 MTP launch: config ignore list plus index, every dense MTP linear must be exempted. On failure, `int4_mtp3_s16`, `int4_mtp3_s20` and `tp2x2_int4_mtp3` are skipped with the reason. On RedHat it passes, so a GPU boot is still the final proof |

## Checked and left unchanged

| Audit item | Finding |
|---|---|
| V5: `--max-num-seqs` is per DP rank | Already set: the 2 x TP2 arms use `--max-num-seqs 8` with DP 2, measured at 16 in total |
| V2: MTP 2 / 3 / 4 | Separate arms already exist; each is now gated against the oracle |
| V4: 16 / 20 / 24 seqs | The arms exist; preemptions, KV max and the measured level are recorded per window |
| `--kv-cache-dtype auto` diagnostic | Not added: it would cost one more launch, and the oracle plus the prefix-off arm already isolate the prefix-cache risk |
| Mamba / prefix fixes #56531, #59759, #60797 | Reported merged before v0.32.0. Not re-verified line by line; the oracle gate is the runtime check |

## CPU validation

Local harness, two modes: normal, and `VMS_BAD_INT4=1`. The second rewrites the ignore list without the mtp entry, so every INT4 arm must be skipped. The Kaggle CPU kit runs the same harnesses: `Build/build_cpu_kit_engine.py` builds `foysalemonshanto/q27-engine-cpu-kit` and the kernel `q27-engine-cpu-harness`.
