# Fix list: DFlash2 PP1 first-request OOM (3 GPT research reports, 2026-11-02)

**Source verification (local Merlin).**
- `tools/merlin/launch.py:419-420` sets `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=256` and `VLLM_E2_QSA_LOGITS_GRANULE_MB=256`.
- `ops/qsa_indexer.py:497 _prefill_workspace_bytes` = min(cap, max_num_batched_tokens x width x 4 B). Our config gives 1024 x 32768 x 4 = 128 MiB.
- `_prefill_backing_numel` rounds the need up to the granule, inside the cap, so it asks for 256 MiB. This matches the GPU OOM ("Tried to allocate 256.00 MiB").
- Fixed backing is used only when granule == 0 (`qsa_indexer.py:789`). Otherwise the backing is allocated per prefill call; the caching allocator may reuse the block.

| # | Item | R1 | R2 | R3 | Verified | Proposed |
|---|---|---|---|---|---|---|
| O-1 | `DFLASH_DRAFT_EMBED_HOST=1` from the first try | yes | yes (GPU-verified) | yes | GPU: PP1 free at ready 631 -> 1047 MiB; every window 8k-116k ran, 0 preemption | Already in E29 rev 2b |
| O-2 | Cap / shrink the QSA prefill workspace | cap 128 MiB (64 if the probe fails), production first choice | granule 256 -> 64, or cap 64, as an A/B only, not production yet | cap 64 + granule 64 (8k-64k in one 64-MiB chunk; 90k-131k chunked) | Source matches; GPU UNVERIFIED | Plain switch line in CELL 1, e.g. `E29_QSA = {'max_logits_mb': 128, 'granule_mb': 64}`, set into the server env (all 4 workers, read back). Test it in the DFlash smoke, then decide the production value |
| O-3 | Memory gate after a REAL first prefill, not only at ready | yes | yes (512 MiB at ready is not enough) | yes; use the worker-level free, not nvidia-smi (577 vs 1047 MiB) | Agreed | At boot run a forced max-size prefill probe (a 116k / 131k prompt at the batched-tokens chunk) and gate on the worker-view minimum free after it. If OOM -> the existing ladder; if below the margin -> one ladder step. A real failure, not a label |
| O-4 | KV pin reduction | fallback | fallback, 7.39 -> 7.14 GiB is about 54k tokens | not needed first | — | Keep as the existing ladder only |
| O-5 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | secondary | unverified | — | 350.91 MiB reserved but unallocated at the OOM | Optional A/B arm, not production |
| O-6 | Host RAM: the cgroup is near 177,152 MiB; EMBED_HOST adds host memory | watch | — | — | Logs show the warning | Record the host cgroup current / max in the LIVE 15-min line plus a boot read-back; no gate |
| O-7 | #60490 does not change an explicit `kv_cache_memory_bytes` pin | noted | — | noted | `gpu_worker.py:904-950` | No change; O-3 covers it |
| O-8 | Per-PP-rank asymmetric KV, or repartition 24/24 | not now | — | — | — | Not doing |

## Decision needed (user)
1. The production value of O-2:
   - (a) Merlin default 256/256 + EMBED_HOST: the only GPU-proven config;
   - (b) cap 128 + granule 64 (R1);
   - (c) cap 64 + granule 64 (R3).

   In all cases the value is a CELL 1 switch, measured in the DFlash smoke first.
2. Whether to add O-3 (first-prefill probe gate) and O-6 (host RAM in LIVE) to E29 rev 2c. This needs a rebuild and the kits again.
