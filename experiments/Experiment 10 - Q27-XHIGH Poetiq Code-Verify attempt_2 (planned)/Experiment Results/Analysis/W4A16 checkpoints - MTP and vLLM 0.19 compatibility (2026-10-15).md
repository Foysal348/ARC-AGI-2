# W4A16 for Qwen3.8-27B on our Kaggle stack: MTP head and vLLM 0.19 compatibility (Claude, 2026-10-15)

**Method:**
- Read the Hugging Face API and files of every checkpoint: file list, sizes, `model.safetensors.index.json` and `config.json`.
- Read the vLLM v0.19.0 source (`qwen3_5_mtp.py`).
- Checked each config's schema against the **compressed-tensors 0.14.0.1** that our wheelhouse installs.
- Nothing was run on a GPU.

## 1. Checkpoints

| Repo | Size | MTP head | Quantization | Ignored layers | Config readable by our compressed-tensors 0.14 |
|---|---|---|---|---|---|
| Qwen/Qwen3.8-27B-FP8 (official) | 30.87 GB | FP8 (22 keys, in `mtp.safetensors`) | FP8 block-128 | — | — |
| **RedHatAI/Qwen3.8-27B-INT4** | **19.45 GB** (`model.safetensors` 18.6 GB as ONE file + `model_mtp.safetensors` 0.85 GB) | **BF16, 15 keys, ignored via `re:^mtp.*`** | W4A16, group 128, **symmetric** (GPTQ + AWQ smoothing) | GDN `in_proj_a/b`, norms, lm_head | ✅ no unknown keys |
| philbert440/Qwen3.8-27B-W4A16-AWQ | 19.55 GB | BF16 (separate file), ignored by name | W4A16 g128 **asymmetric** | GDN parts, lm_head | ✅ |
| Avesed/Qwen3.8-27B-INT4-W4A16 | 20.59 GB | BF16, `re:.*mtp.*` | W4A16 **g32** sym | GDN parts, lm_head | ✅ |
| pearsonkyle/Qwen3.8-27B-GPTQ-W4A16 | 19.42 GB | BF16 (`model-mtp-bf16.safetensors`) | W4A16 g128 sym | whole linear_attn modules, lm_head | ✅ |
| cyankiwi/Qwen3.8-27B-AWQ-INT4 | 21.02 GB | BF16, ignored by name | W4A16 g32 asym | GDN parts, lm_head | ✅ |
| avyukth/Qwen3.8-27B-AWQ-INT4 | 18.57 GB | **none found** (no index, no mtp file) | W4A16 g128 sym | — | ✅, but **no MTP → no speculative decoding** |
| Frozenlock / muxiln int4-AutoRound | 19.00 GB | **quantized** (qweight / qzeros) | auto-round, auto_gptq packing | in_proj_a/b | n/a (AutoRound path; not compressed-tensors) |
| Ar4ikov HyperQwen AWQ asym | 16.84 GB | **quantized + custom `draft_lm_head`** | W4A16 g128 asym | GDN parts | ✅, but a custom drafter, likely not standard vLLM MTP |
| lued INT8-W8A16-MTP | 31.62 GB | BF16 | W8A16 | — | ✅, but no memory gain |

## 2. vLLM 0.19 MTP loading (from source)

- The MTP predictor is built under the prefix **`mtp`** (`mtp.fc`, `mtp.layers.0.*`), with the **same quant_config as the target**. So the checkpoint's ignore rule must match `mtp.*`.
  - RedHat's `re:^mtp.*` matches.
  - philbert and cyankiwi list `mtp.*` names explicitly.
  - **[inference] With these, the BF16 MTP head should load as unquantized layers.**
- **A silent failure mode:** the MTP `load_weights` **skips unmatched tensors with a warning** ("Parameter … not found in params_dict, skip loading") instead of failing. A broken MTP load may therefore look like a normal start with very low acceptance.
  - **GPU check:** grep the log for "skip loading" and confirm the MTP per-position acceptance is about 0.7 / 0.5 / 0.4.

## 3. Other compatibility checks

- **Marlin shapes at TP4, group 128:** every per-rank dimension is divisible by 128 and by 64.
  - down_proj input: 17408/4 = 4352.
  - o_proj input: 24×256/4 = 1536.
  - qkv input: 5120.
- **Weights per GPU:**
  - INT4 ≈ 19.45/4 ≈ 4.9 GB.
  - Official FP8 ≈ 7.7 GB (minus the vision tower, which `language_model_only` skips).
  - **About 2.6–2.8 GB per GPU freed** → about +200 KV blocks → **about +320k tokens (+24 %)** [inference].
- **Accuracy (RedHat card, vs BF16):**

  | Benchmark | BF16 | INT4 | Δ |
  |---|---|---|---|
  | GPQA Diamond | 89.23 | 87.88 | −1.35 |
  | AIME 2025 | 95.42 | 94.17 | −1.25 |
  | MMLU-Pro | | | −1.0 |
  | MATH-500 | | | −0.4 |

  Not measured on ARC, and reasoning length is not reported.

## 4. Recommendation

1. **Candidate: RedHatAI/Qwen3.8-27B-INT4.**
   - Symmetric g128 gives the best Marlin path.
   - BF16 MTP head included and correctly ignored.
   - Config schema compatible with our wheelhouse.
   - Publisher accuracy table available.
2. **Kaggle upload:** about 19.5 GB, including one 18.6 GB file. Options:
   - (a) download locally and upload as a Kaggle model or dataset;
   - (b) a Kaggle notebook with internet → `snapshot_download` → save the output as a dataset. /kaggle/working is about 20 GB, so it is tight.
   - Verify the file hashes after upload.
3. **First GPU check** (when the GPU quota resets):
   - load with `quantization="compressed-tensors"`;
   - confirm the Marlin kernel in the log;
   - confirm no "skip loading" for MTP;
   - confirm MTP acceptance about the same as FP8;
   - confirm the memory freed;
   - then measure tok/s at C=16 and C=24, and thinking length + accuracy on EVAL12 vs FP8.
4. **Not recommended:**
   - checkpoints without MTP (avyukth);
   - quantized or custom drafters (AutoRound, HyperQwen) on 0.19;
   - W8A16 (no memory gain).
