# FN-SMOKE — response to review round 1, with a true / false ledger

Every claim was checked before changing anything. External claims were checked against the **actual installed
code**: the Pennyroyal wheel `sglang-0.5.19+gd00d88efc8d6` and `sglang_kernel-0.4.6.post1+cu130` were downloaded from
`dfranzen/pennyroyal-v253` and inspected (excerpts in `evidence/`). Notebook v2 sha256 is in `SHA256SUMS.txt`.

## Ledger

| # | reviewer claim | verdict | evidence | action |
|---|---|---|---|---|
| 1 | P0: PLE is first materialised on GPU (~23.84 GiB/rank) before offload; Pennyroyal issue still open | **INCORRECT for this wheel** | `evidence/pennyroyal_ple_offload.txt`: with `--ple-offload-embedding` the embedding is built under `torch.device("meta")` ("the pinned host table becomes its first and only allocation instead of a transient per-rank device shard (sgl-project/sglang#39841)"), and the host table is pinned at exact size via `mmap` + `cudaHostRegister` (#40626). An open issue on the repo does not mean this build lacks the fix. | none needed; kept host-RAM gate (TP4 shards the PLE: ~23.8 GB pinned per rank, ~95 GB total) |
| 2 | FlashInfer GDN needs SM90+, L4 is SM89 | **CONFIRMED** | `evidence/flashinfer_gdn_sm90.txt` ("Requires SM90+ and FlashInfer with GDN kernel support") | FlashInfer GDN removed; Triton only |
| 3 | FP8 KV + QSA dtype bug at startup | **NOT VERIFIED** in this code (no such assert found in `layers/attention/qsa/`); QSA uses a BF16 reference indexer + Triton kernels + `sgl_kernel` top-k | cheap to avoid | `--kv-cache-dtype auto` (BF16) for the first boot |
| 4 | DP-attention not supported for `qwen4_exp` | **PARTLY INCORRECT** — the model code handles DP-attention (`is_dp_attention_enabled`, `gather_dp_tokens`). **But** a stronger reason exists: with DP-attention the PLE table is no longer TP-sharded (`use_attn_tp_ngram`), so every rank would pin the full ~95 GB table (~381 GB host RAM) | `evidence/dp_attention_and_kernels.txt` | DP-attention removed |
| 5 | plain TP4 fallback conflicts with 640/128 | **CONFIRMED** | `evidence/moe_partitioning.txt`: Marlin requires `input_size_per_partition % group_size == 0` (160 % 128 ≠ 0) | TP4-only attempt removed |
| 6 | "hard 45 min" not hard | **CONFIRMED** (my bug) | install timeouts 1800 s ×3, late-starting 18-min attempt, 600 s request | absolute `DEADLINE`; every blocking call `clamp()`ed; no attempt starts with < 10 min + safety left |
| 7 | expiry leaves `stopped_at=None` → "all stages passed" | **CONFIRMED** (my bug) | `go()` returned False silently | `go()` records `global_deadline`; final cell sets `incomplete_unknown` if any expected stage is missing; verdict line never claims success otherwise |
| 8 | stage 0 does not gate 4×L4/sm89, driver, available RAM | **CONFIRMED** | — | gates: 4 GPUs, all "L4", all CC 8.9, driver ≥ 580, driver CUDA ≥ 13, MemAvailable ≥ 110 GB, /tmp free ≥ 25 GB; `ulimit -l` logged |
| 9 | stage 3 matmul gives false confidence | **CONFIRMED** | — | stage `3_preflight`: imports torch/sglang/sgl_kernel/flashinfer/triton, 4 GPUs all (8,9), `sglang serve --help` must list every flag used — all before the 181 GB load |
| 10 | first boot too aggressive | **AGREED** | — | 8K context, 1 request, chunk 2048 / prefill 4096, mem 0.80, BF16 KV, `--language-model-only`, no CUDA graphs, no speculative decoding |
| 11 | broad `pkill -f 'sglang serve'` | **CONFIRMED** | — | only the tracked process group is killed |
| 12 | /tmp space not a gate | **CONFIRMED** | — | gate added (see 8) |

## Things the reviewer did not raise (found while verifying)

* `sgl_kernel` compatibility on L4: `sm90/common_ops.abi3.so` and `sm100/common_ops.abi3.so` both contain `sm_89`
  code; `flash_ops` has sm_80/86 (forward-compatible with sm_89). So the core kernels are built for this GPU.
* The shared expert (640) equals `moe_intermediate_size` (640), so `can_fuse_shared_expert` fuses it as an extra
  expert (`mlp.experts.512`) — it is not TP-split, which avoids a second 160/128 problem.
* `--max-mamba-cache-size` exists (generated from the `max_mamba_cache_size` field) even though the literal string is
  not in the source; `--language-model-only` exists.

## Launch ladder now

1. `tp4_ep4_triton_moe_auto`  — TP4 + EP4, Triton GDN, `--moe-runner-backend auto`
2. `tp4_ep4_triton_moe_triton` — same with `--moe-runner-backend triton` (only if 1 fails and ≥10 min + safety remain)

## Validation (local, `validate_v2_output.txt`, real notebook code, injected hardware facts)

| test | expected | result |
|---|---|---|
| 30 s budget | stopped_at = global_deadline, no success claim | PASS |
| good 4×L4 / driver 580 / CUDA 13 / 170 GB available / 100 GB tmp | stage 0 PASS (then inputs missing locally) | PASS |
| 2×L4 | stage 0 FAIL (`four_gpus`) | PASS |
| 4×T4 | stage 0 FAIL (`all_l4`, `all_sm89`) | PASS |
| driver 570 / CUDA 12.8 | stage 0 FAIL (driver, CUDA) | PASS |
| 80 GB available RAM | stage 0 FAIL (`ram_available_enough`) | PASS |
| 10 GB /tmp | stage 0 FAIL (`tmp_free_enough`) | PASS |
| stages skipped without stopped_at | `incomplete_unknown`, verdict "NOT SHOWN TO WORK" | PASS |

Not testable locally (no Linux GPU): process-group kill, the real install, the real load. The tracked-PGID kill is
`os.killpg(SERVER.pid, 15)` then 9, only for the process started by this notebook.
