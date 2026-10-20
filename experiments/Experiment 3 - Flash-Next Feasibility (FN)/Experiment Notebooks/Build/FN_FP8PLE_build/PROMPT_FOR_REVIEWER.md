# Review request: FN-PLEFP8-UPLOAD + FN-SMOKE v3 (Qwen3.8-Flash-Next W4A16 + FP8 PLE on Kaggle 4×L4)

You are reviewing two Kaggle notebooks before they are run. Find real defects only, and cite the cell and line for each.
For every claim, mark it as verified against the code here or inferred.

## Goal
Load Qwen3.8-Flash-Next (Intel W4A16 AutoRound) with Pennyroyal SGLang on Kaggle 4×L4 (sm89, 24 GB each).

FN-SMOKE v2, already reviewed in round 1 (see `REVIEW_RESPONSE_R1.md` in the previous bundle), used Intel's BF16 PLE:
- ~102 GB on disk;
- ~95 GB of pinned host RAM, which was the biggest risk.

v3 swaps that for albucino's FP8 PLE: the official RadixArk FP8 table plus scale, ~51 GB.

## Facts already verified (evidence in this bundle)
| fact | evidence |
|---|---|
| albucino's 15 main shards are byte-identical to `Intel/Qwen3.8-Flash-Next-W4A16-AutoRound@4c67bf6` | HF LFS sha256 compared, 15/15 equal (`manifest.json` holds the albucino values) |
| albucino config = Intel config + `text_config.ple_embedding_dtype: "float8_e4m3fn"` (no other difference) | recursive JSON diff of both config.json files |
| albucino's index maps every non-PLE tensor to the same file as Intel's index; MTP tensors (`model_extra_tensors.safetensors`, 1565 keys) are not referenced | index comparison |
| Pennyroyal builds the PLE parameter as FP8 when `ple_embedding_dtype == "float8_e4m3fn"`; with `--ple-offload-embedding` it is built on meta, so the pinned host table is the only allocation | `sglang/srt/models/qwen4_exp.py` lines ~490-517 (wheel `sglang-0.5.19+gd00d88efc8d6`) |
| embeddings are multiplied by `ngram_embedding.weight_scale` | `qwen4_exp.py:570`, `:1257`; albucino index contains `...ngram_embedding.weight_scale` |
| dfranzen's launcher comment: "MODEL_DIR: Intel AutoRound INT4 target (BF16 PLE or albucino FP8 PLE)" | `launcher_cell12.py:4` (previous bundle) |

## Notebook 1: `FN-PLEFP8-UPLOAD.ipynb` (CPU, internet ON, run once from the Kaggle editor)
- Built from the ARC-AGI-3 team's streaming uploader, which uploaded 132.7 GB and served it on 2026-09-26/27.
  `adapt_streaming_uploader.py` changes only:
  - the config (repo, pinned revision, slugs, `EXPECTED_FILES`);
  - an account check (secrets must belong to foysalemonshanto before any transfer);
  - descriptive text.
- The stream / resume / publish / verify code is unchanged (cell 3 differs only in the manifest notes; cell 5 is identical).
- Uploads 15 files, 51.22 GB, with no disk staging. The instance is created only if every hash matches.
- Local checks:
  - cell 1 asserts pass;
  - the real HF reader returned correct sha256 for the 4 small files;
  - a Range resume on a 4.4 GB shard works (`val_upload_reader_output.txt`).
- Not testable locally: the Kaggle blob upload and `model_instance_create`. Both are the unchanged, previously proven code.

## Notebook 2: `FN-SMOKE-FLASHNEXT-FP8PLE-4XL4.ipynb` (4×L4, ≤45 min, load-only)
- v2 plus changes produced by `patch_v3.py` and `stage1_v3.txt`.
- Stage 1 builds `/tmp/fn_model` from symlinks:
  - the overlay's config, index and 10 FP8 PLE shards;
  - from dfranzen's Intel mount, everything else non-safetensors (tokenizer and similar), except config, index, ple manifest and README;
  - only the weight files named in the overlay index (so Intel's BF16 PLE and MTP files are excluded).
- Every shard is checked for size and safetensors-header SHA256 against the pinned HF commit. The overlay's config and index are fully SHA256-checked, and the config must say FP8 PLE.
- The RAM gate drops from 110 to 70 GB. The launch profile is unchanged from v2 (TP4+EP4, Triton GDN, BF16 KV, 8K context, 1 request).
- Local tests are in `validate_v3_output.txt`:
  - 8 merge cases;
  - hardware gates (75 GB passes, 60 GB fails, 2×L4, driver 570, 30 s deadline);
  - the embedded expected values equal the HF manifest;
  - every overlay file needed is in the uploader's list.

## Questions for you
1. Any way stage 1 could produce a folder that SGLang loads differently from albucino's own repository layout?
   For example, extra non-weight files from the Intel mount such as `quantization_config.json`, or symlink resolution.
2. Is 70 GB of available RAM a sound gate for an FP8 PLE pinned table of 51.2 GB total, TP4-sharded at ~12.8 GB per rank, plus 4 worker processes?
3. Anything in the uploader adaptation that breaks the proven code path?
4. Any remaining v2 risk that the FP8 PLE does not address (the 160 % 128 Marlin/TP issue is avoided by EP4)?
