# E13 plan self-validation (2026-10-19)

Every claim behind PLAN.md rev 3, with its evidence. **V** = verified in files or source; **M** = must still be measured on GPU; **F** = a rev-2 claim found false.

Paths:
- 0.5.21 = `_scratch/session_1c6db7aa/sgl_src/src/sglang` (unpacked from the official 0.5.21 wheel; same version as our wheelhouse).
- Penny = `_scratch/e13_validate/penny_src/sglang`, unpacked from `dfranzen/pennyroyal-v253/wheels/sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl`, downloaded 2026-10-19.

| # | Claim | Evidence | Verdict |
|---|---|---|---|
| 1 | Franzen's 3 versions share the same code; only cell 0 (text) and cell 16 (n_passes 1→4, demo exclusions, per-game time) differ | `_scratch/ext/kaggle_dfranzen_versions/` v1/v2/v3 + a cell diff; v3 equals the latest pull | V |
| 2 | Franzen: TP1 on 1x RTX PRO 6000 (cap 12.0), sglang `0.5.19+gd00d88efc8d6`, torch 2.13 cu130; ready at 562 s; 590-617 tok/s; host RAM 106/177 GB; GPU 91/96 GB; v1 mean 36.56 (10 games), v3 46.49 (25 games x 4 passes) | v3 / v1 cell outputs 12, 18, 20 | V |
| 3 | Franzen's model = Intel W4A16 AutoRound **with BF16 PLE** (22 `model-ple-*` shards, `ple_reshard_manifest.json`) + albucino drafter (`mtp-routed-experts-int4`, `mtp-dense`) | Kaggle model file listings (`_scratch/e13_validate/inputs_listing.json`); his precache read 185.4 GB (model + drafter) | V |
| 4 | **F (rev 2):** "Pennyroyal + one-condition P1" | Penny `qwen_sparse_attn_backend.py` `_forward_paged_attention` (from line 1869): no `is_hip()` branch, no `sparse_gqa_packed_decode_triton` anywhere in the wheel; scratch allocated with `k_buffer.dtype` (fp8 with fp8 KV) and passed to `flash_attn_varlen_func` | **F**: Pennyroyal dropped for L4 |
| 5 | 0.5.21 decode on sm89: the TRT-LLM path only for SM100/SM120 → FA2 `flash_attn` → FA4 CuTe | 0.5.21 `qwen_sparse_attn_backend.py:48-96` | V |
| 6 | Our wheelhouse has `flash_attn_4-4.0.0b35` and `tilelang-0.1.12`, and no classic FA2 | `e11_02_run2/sglang-0.5.21-wheelhouse/SHA256SUMS` | V |
| 7 | 0.5.21 has the Triton packed decode under `if is_hip():` (line 1601); it is pure Triton, converts fp8 K/V to Q's dtype (`sparse_attn.py`, chunk-prefill kernel comment "Unsupported rhs dtype fp8e4nv"), does no host sync, and its scratch uses `q.dtype` | 0.5.21 `qsa/sparse_attn.py:245-330`, backend 1521-1632 | V (numerics on sm89: **M**) |
| 8 | 0.5.21 indexer MQA = TileLang if importable, else the torch reference | 0.5.21 `qsa/mqa.py:396-410` | V |
| 9 | 0.5.21 GPTQ-Marlin MoE scales are still `dtype=torch.half` (2 places); Franzen patch #3 changes exactly these to `params_dtype`; the Penny wheel's `gptq_moe.py` sha starts `47e65140`, his AFTER_SHA | 0.5.21 `gptq/schemes/gptq_moe.py:239,247`; `dafr/serving/build_bundle_pennyroyal.sh:250-290` | V |
| 10 | Every Franzen launch flag exists in Penny; in 0.5.21 all exist except `--gdn-mtp-cache-mode` | script over both sources (flag list `_scratch/e13_validate/flags.txt`) | V |
| 11 | 0.5.21 has the `qwen4_exp` model, `auto_round`, `--ple-offload-embedding` (default on for BF16 Qwen4-Exp), `QwenSparseMultiStepDraftBackend` (NEXTN, topk 1) | 0.5.21 `srt/models/qwen4_exp*.py`, `arg_groups/fields/exec_.py:942` | V |
| 12 | PLE offload = `VocabParallelEmbedding` (sharded over TP ranks) wrapped by `Qwen4ExpPinnedHostEmbedding`; backend `pinned` or `file` | 0.5.21 `qwen4_exp.py:505-540` | V (host RAM use: **M**) |
| 13 | Host RAM on Kaggle 4xL4 is 202.6 GB | E3 `FN_PROBE_v421_run1/fn_smoke/fn_smoke_report.json` stage 0 | V |
| 14 | FlashInfer GDN only for the validated SM90/SM100 domains, otherwise Triton | 0.5.21 `linear/gdn_backend.py:189` | V |
| 15 | SM120 env vars (`SGLANG_ENABLE_SM120_LOWM_BF16_GEMM`, `SGLANG_SM120_*`) are absent from the 0.5.21 environ; `SGLANG_MAMBA_CONV_DTYPE` and `SGLANG_NUMA_BIND_V2` exist (kept verbatim) | 0.5.21 `srt/environ.py` | V |
| 16 | EP4 needed: TP4 MoE split 640/4 = 160 is not a multiple of group 128; AutoRound MoE needs runner `auto` | E3 `patch_v3_3.py` header; v3.2 crash `gptq_kernels.py:360` | V (on Penny; on 0.5.21: **M**) |
| 17 | W4A16 loads on 4xL4 at 19.06 GB/GPU; 345 s with prefetch | E3 v4.2.1 serve.log | V (with FP8 PLE; BF16 PLE adds host RAM only) |
| 18 | E3 garbage was **not** caused by the Marlin scale bug (Penny already had the fix) | claim 9 + E3 used the same Penny wheel | V |
| 19 | Our old shim's decode maths is correct (all keys, bottom-right causal) | E3 `make_fn_probe_v421_generated.py:393` | V; still not reused (X1) |
| 20 | KV per token per rank (fp8) = 12 layers x 1 KV head x 256 x 2 x 1 B = 6 KiB | config (24 Q / 2 KV heads, d 256; TP4 gives 1 KV head per rank); E3 bf16 0.68 GB K per rank for 118k tokens | V (arithmetic); usable tokens **M** |
| 21 | Duck contexts on ARC-2: SMOKE1 easy task 6.9k → 13.6k prompt tokens | E12 SMOKE1 `duck/2ba387bc_b*/requests.jsonl` | V (hard tasks: EVAL30 **M**) |
| 22 | Our ARC-3 Flash-Next runs were vLLM NVFP4 on RTX PRO 6000 (not reusable on L4) | ARC-3 report, `P1_SPARSE_RUN_EVIDENCE.zip` | V |
| 23 | amn's 70.28 is **not** shown to be Flash-Next ("might reach a better score ... by letting it work as an agent") | `_scratch/webresearch/topteam_comments.md:325-326`, `lb.json` | V (corrected 2026-10-19) |

Unknowns that only the GPU probe answers: numerics of P1 + TileLang on sm89, EP4 + Marlin on 0.5.21, KV tokens with the drafter, tok/s with CUDA graphs, pinned PLE RAM.
