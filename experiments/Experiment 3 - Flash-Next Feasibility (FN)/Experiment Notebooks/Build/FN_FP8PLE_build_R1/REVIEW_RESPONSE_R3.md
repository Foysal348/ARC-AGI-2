# Response to the R2 and R3 reviews: true / false ledger and v3.2 changes

Every claim was checked against the Pennyroyal source (wheel `sglang-0.5.19+gd00d88efc8d6`) or live Kaggle state.
New source excerpts are in `evidence/pennyroyal_mamba_slots.txt`.

## Ledger

| # | claim | verdict | evidence | action |
|---|---|---|---|---|
| 1 | P0: `--max-mamba-cache-size 4` gives 0 requests with `extra_buffer` (5 slots/request) | **CONFIRMED in the source** | `kv_cache_configurator.py` 1919-1947 (`_calculate_mamba_ratio`), constants 156-161: ratio = 1 if the radix cache is disabled; otherwise 3, plus 2 for `extra_buffer` with overlap schedule (the default). `resolve_max_num_reqs`: `max_mamba_cache_size // ratio`, and `<= 0` raises RuntimeError after the model load. | Cache is set per attempt: no-radix 2 (needs 1), extra_buffer 6 (needs 5). The validator computes slots with the same rule and rejects the v3.1 combination (extra_buffer + 4) as a regression test. |
| 2 | P0: 8 MiB hard memlock may block cudaHostRegister of 12.8 GB/rank | **OPEN, now tested on the machine instead of argued** | `run1_2026-10-05/fn_smoke_report.json` shows memlock soft = hard = 8 MiB. Whether the NVIDIA driver charges cudaHostRegister to RLIMIT_MEMLOCK is not settled by the source. I withdraw my "dfranzen proved it" claim: the bundle has no dfranzen log showing his memlock. | Stage `3_pinned_memory` (after install, before the load) registers 64 MiB and then 1 GiB of mmap'd anonymous memory with Pennyroyal's exact call and flags (PORTABLE\|MAPPED), copies it to the GPU, and unregisters it. Failure stops the run before the 125 GB load. Tested: probe failure → no server start; success → launch proceeds. |
| 3 | P1: attempt order should be simplest first, MoE auto later | **AGREED** | Public Marlin TP4 failure reports; the sm100 override in the source selects the `triton` attention backend for this family (`overrides.py` 1570-1600), so triton attention is the path dfranzen's hardware used. | A1: radix cache off + triton attention + triton MoE. A2: extra_buffer (interval 64) + triton attention + triton MoE. MoE `auto` is deferred to a later run. |
| 4 | P1: mem-fraction-static 0.80 is too tight | **AGREED (inference)** | 15 main shards are 73.58 GB, about 17.1 GiB per GPU with EP4, against a static budget of 0.80 × 22.49 = 18.0 GiB | 0.90. Context 4096, chunked prefill 1024, max prefill 2048 for this load-only probe. |
| 5 | P2: the validator said PASS without checking mamba-slot viability or memlock | **CONFIRMED** | — | Semantic tests added: slots per attempt, the v3.1 regression, and pin-probe gating. |
| 6 | (R2) uploader creates the parent model after the 51 GB transfer; existence check reads only page 1 | **True, but moot** | The upload already finished: 22 files and 51.244 GB verified through the Kaggle API on 2026-10-05 | No change. Noted for any future uploader reuse. |
| 7 | (R2) license label | Reviewer withdrew the R1 claim | — | Kept. |
| 8 | (R3) hash-timeout threads keep running after a timeout | **True, minor** | Stage 1 then fails and no server starts | No change. |

## Run 1 recap (why v3.1/v3.2 exist)
- Stages 0-3 passed on 4×L4. The full SHA256 of all 25 shards (124.8 GB) matched in 192.5 s.
- The launch stopped in argument resolution: `--language-model-only` is not supported for Qwen4Exp.
- v3.1 removed it and added the radix settings that `page_size = 64` requires. v3.2 fixes the mamba slots, adds the
  pin probe, reorders the attempts and raises the GPU memory budget.
