# What the F = 6 extra blocks per request are (E28 K100 server log + Merlin source, 2026-11-03)

## Evidence
- **Server log** `server_merlin_fp8_mtp_pin508_i8.log` (E28 K100):
  - `kv cache group sizes (4096, 4096, 4096, 8, 4096)`. That is 5 groups: 1 QSA attention group, 3 Mamba (GDN) groups and the QSA ring group (block 8, which "packs 1024 requests per pool block").
  - `folded replicated Mamba group 3 (...layers.1.ple) into Mamba group 1 (now 13 layers)`.
  - `Padding mamba page size by 21.01% to ensure that mamba page size and attention page size are exactly equal.`
  - `GPU KV cache size: 1,652,197 tokens, Maximum concurrency for 131,072 tokens per request: 12.61x`. With pool 479: 479 / 12.61 = 38 = 32 attention blocks + **6**.
  - `Mamba align: each request holds its partial-tail CoW checkpoint until it is freed`.
  - GDN RecoverSSM commit contexts of 7 / 6 / 6 layers (PP0) and 5 / 6 / 6 (PP1): 36 GDN layers in 3 groups per PP stage.
- **Merlin source** `v1/kv_cache_interface.py:1077-1083`, MambaSpec.max_memory_usage_bytes:
  - in `align` mode = page × (2 + num_speculative_blocks + num_prefill_checkpoint_blocks);
  - num_speculative_blocks = 0 under ReplaySSM / RecoverSSM (`mamba/abstract.py:77-83`).
- **Model config:** `mamba_ssm_dtype float32`.
  - GDN, per layer per TP2 rank: 24 value heads × 128 × 128.
  - SSM state fp32 = 1.50 MiB.
  - Conv state bf16 (5,120 × 3) = 30 KB.
  - RecoverSSM replay record fp32 [24, 4, 257] = 99 KB.
  - Total ≈ 1.70 MB a layer. 7 layers ≈ 11.9 MB, which matches the Mamba page = 14 MiB / 1.21 = 12.1 MB.

## So
- **What F is:** F = 3 Mamba groups × 2 pages (the running state + the align CoW prefix checkpoint). Each page is padded to the attention page of a 4,096-token block.
- **What it costs:** 6 × 4,096 = 24,576 token-equivalents a request. At 24 streams that is 144 of 479 pool blocks (30 %).
- **The fp32 SSM state** is about 88 % of each Mamba page.

## Levers
1. **SSM state bf16** (`--mamba-ssm-cache-dtype bfloat16`).
   - andy2709 runs production with `--mamba-ssm-dtype bfloat16`. RecoverSSM accepts bf16 / fp32 (`config/vllm.py`).
   - Effect: the Mamba page ≈ halves, so block_size 2048 fits it, so F costs 6 × 2,048 = 12,288 token-equivalents.
   - Estimated resident gain: +23 % at 40k, +17 % at 60k, +11 % at 100k. Prefix granularity also gets finer (2,048).
   - Needs a teacher-forced KL / accuracy check and a GPU memory check.
2. **Padding (21 %).** Pick a block size whose attention page is closer to the Mamba page (FA2 needs multiples of 256). For example, 3,584 cuts the padding to ≈ 6 %.
3. `mamba_cache_mode none` would give F = 3, but it loses the Mamba prefix checkpoint. Every Duck turn would then re-prefill. Not worth it.
