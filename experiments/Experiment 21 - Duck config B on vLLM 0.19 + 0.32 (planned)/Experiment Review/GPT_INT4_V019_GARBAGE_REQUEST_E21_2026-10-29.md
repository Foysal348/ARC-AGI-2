# E21: RedHat INT4 on vLLM 0.19.0 generates garbage. Find the root cause and give a fix (2026-10-29)

## What we need from you

1. **Root cause.** Give it with the exact evidence: log lines or config fields in this zip, plus what you know of the vLLM 0.19.0 source. Rank the causes if several are possible. Say which you can prove from the files and which are inference.
2. **A concrete fix** that makes NB1 half A (vLLM 0.19.0, RedHat INT4 W4A16, MTP-3, 20 streams) produce correct output. Acceptable forms:
   - server flags;
   - environment variables;
   - a config.json overlay staged next to the weights in `/dev/shm` (the weights must not change);
   - a small monkeypatch of vLLM 0.19 applied before the server starts.

   Give exact code that drops into `Build/e21_engine_new.py` / `Build/build_e21.py`.
3. **A cheap GPU check** that proves the fix before the 12-h run. A content smoke plus something stronger is better, for example greedy parity of the first N tokens against vLLM 0.32 INT4 or against FP8.
4. **If 0.19 + this checkpoint cannot be fixed safely,** recommend what NB1 should run instead. Constraints:
   - The hidden rerun runs half A only, so half A must be the config we trust most.
   - NB1 and NB2 are pushed together.

Report only BLOCKER / HIGH items about this INT4 problem and the fix. No style notes, and no re-audit of the rest of E21 (rev 2h is closed).

## Setup

- **Hardware and image:** Kaggle 4x L4 (SM 8.9, 24 GB, PCIe, no NVLink). The pinned image has Python 3.12 and its own vLLM 0.19.0. vLLM 0.32.0 is installed into a venv from an offline wheelhouse. The run is offline.
- **Model:** Qwen3.8-27B, `Qwen3_5ForConditionalGeneration`: hybrid Gated DeltaNet (linear_attention) + full attention every 4th layer, with an MTP head (`mtp_num_hidden_layers` 1).
- **NB1 weights:** RedHat INT4, `foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1`. Its quantization config is in `checkpoint/int4_config.json`:
  - compressed-tensors `pack-quantized`, W4A16, group 128, symmetric, `actorder: static`;
  - compressed-tensors version `0.18.1.a20260824`; transformers 5.16.1;
  - `kv_cache_scheme`: static per-tensor 8-bit float, so the checkpoint carries calibrated KV scales (the FP8 checkpoint uses 1.0);
  - ignore list (304 entries): vision tower, `lm_head`, `re:^mtp.*`, and in each GDN layer `linear_attn`, `linear_attn.norm`, `in_proj_a`, `in_proj_b`. GDN `in_proj_qkv`, `in_proj_z` and `out_proj` are INT4. The MTP head is BF16.
  - The checkpoint is one ~18.6 GB safetensors file. It is staged to `/dev/shm/e14-weights/...` by `code/e14_loadpath.py`: a byte copy verified with per-range blake2b plus a safetensors layout check. The files are not modified.
- **NB2 / fallback weights:** FP8 repacked (`qwen3-8-27b-fp8-repacked-v1`). These work on both 0.19 and 0.32.

## Evidence

| run | vLLM | weights | result |
|---|---|---|---|
| E21 NB1 smoke half A primary (2026-10-28 19:38) | 0.19.0 | INT4, MTP-3, FLASHINFER, fp8_e4m3 KV, `--language-model-only`, TP4 | **garbage** |
| same notebook, half A fallback | 0.19.0 | FP8, same flags, 16 seqs | "2 + 2 equals 4.", Duck ran normally |
| E21 NB2 smoke half A | 0.19.0 | FP8 + `--mamba-ssm-cache-dtype float16` | correct, 3/22 hits in 20 min |
| Exp 14 VMS (2026-10-26) `int4_mtp3_s16` / `s20` | 0.32.0 | INT4, MTP-3, fp8_e4m3 KV, FLASHINFER, TP4 (same checkpoint, read from /kaggle/input) | **coherent** ARC reasoning (`evidence/vms_032_int4/vms_greedy_rows_int4.jsonl`), MTP accept 2.9-3.0 |

The 0.19 INT4 smoke output: `"ogan_plural_METADATAipesipp捧iperovitortsaderilarityal遂懿 Tidgej_pluralxic閒_pluralidding��orse�Intl徒�越邀ixelipes�LO (正�愈uban-linux Anonymous延Intlloiunalzensitud..."`. The request was greedy (temperature 0), thinking off, max_tokens 96. The prompt was 26 tokens: "Answer in one short sentence: what is 2 + 2?". All 96 tokens were garbage. Since speculative decoding verifies every token with the target model, garbage from the first token means the **target** forward is wrong, not only the drafter.

Differences between the two INT4 server logs (0.19 `evidence/e21_smoke_nb1/vllm_server_halfA_primary.log`; 0.32 `evidence/vms_032_int4/vllm_server_int4_mtp3_s20.log`):

| | 0.19 | 0.32 |
|---|---|---|
| kernel lines | `Using MarlinLinearKernel for CompressedTensorsWNA16` only | also `Using MarlinLinearKernel for mixed-precision linear` and `GDN spec decode kernel: cuda` |
| `Model loading took` per rank | **4.49 GiB** (3.7 s from /dev/shm) | **4.69 GiB** (40 s) |
| model runner | V1 | "Using V2 Model Runner" |
| max model len | 135,168 | 106,496 |
| CUDA graphs | PIECEWISE (FULL not supported with spec decode on FlashInfer) | PIECEWISE |
| warnings | `Mamba cache mode is set to 'align'`; `Add 3 padding layers` | same family |

There is no load error, no missing-key warning and no NaN warning in the 0.19 log.

## Our hypotheses (unverified)

- **H1:** vLLM 0.19's compressed-tensors path mis-handles this newer checkpoint format: version 0.18.1, `actorder: static`, or a scheme 0.32 routes through "mixed-precision linear". Wrong dequant gives garbage silently.
- **H2:** the checkpoint's calibrated KV scales (`kv_cache_scheme`, `k_scale` / `v_scale`) are loaded or applied wrongly in 0.19 with `--kv-cache-dtype fp8_e4m3` on the hybrid model, for example scales loaded for the wrong layers.
- **H3:** weight-name or shape mapping. The INT4 GDN `in_proj_qkv` / `in_proj_z` are separate tensors, which 0.19 may fuse or shard differently under TP4. The 0.2 GiB per-rank gap in "Model loading took" could be parameters left uninitialised. The same applies to `language_model_only` / `--language-model-only` prefix mapping (`model.language_model.*`).
- **H4:** INT4 Marlin on 0.19 with group 128 under TP4 for a dimension that does not divide cleanly per rank.

We have no vLLM 0.19 source locally, so we could not check these.

## Files in this zip

- `README_FIRST.md`: this file.
- `evidence/e21_smoke_nb1/`:
  - the INT4 0.19 server log (primary), the FP8 0.19 fallback log and the half B 0.32 FP8 log;
  - `e21_report.json`, the half A `e12_run_report.json`;
  - the kernel log (notebook stdout).
- `evidence/e21_smoke_nb2/`: the healthy FP8 + SSM16 0.19 server log, `e21_report.json`.
- `evidence/vms_032_int4/`:
  - the vLLM 0.32 INT4 server logs (s16, s20) and the FP8 reference log;
  - `vms_report.md` (the INT4 MTP static check is on line 8);
  - the INT4 greedy rows (text).
- `checkpoint/`: the INT4 `config.json`, `generation_config.json` and `tokenizer_config.json`, and the FP8 `config.json`.
- `code/`:
  - `e21_engine_new.py` (launch, smoke, relaunch);
  - `build_e21.py`, `e21_cell3_patch.py`;
  - `e14_loadpath.py` (staging);
  - the four built notebooks (NB1, NB2, and their SMOKE variants), rev 2h.
- `docs/`: E21 README, provenance (R1 = this risk, flagged before the run), smoke results, findings tracker.
