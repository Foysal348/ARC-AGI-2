# Review request R4: FN-SMOKE v3.2 (go / no-go for the second 4×L4 run)

Read `REVIEW_RESPONSE_R3.md` first. It holds my verdict on each of your R2/R3 claims, with source evidence. Then check
v3.2 (`build/patch_v3_2.py`, `notebooks/FN-SMOKE-FLASHNEXT-FP8PLE-4XL4.ipynb`). Report real defects only, citing the
file and line. For each claim, say whether it was verified against a quoted source or inferred.

## v3.2 in short
- Base profile:
  - TP4 + EP4, auto-round, FP8 PLE offload
  - BF16 KV, context 4096, 1 request, chunked prefill 1024 / max prefill 2048
  - mem-fraction-static 0.90, page 64 (forced by QSA)
  - triton GDN, no CUDA graphs
- Attempt 1: `--disable-radix-cache --attention-backend triton --moe-runner-backend triton --max-mamba-cache-size 2`
- Attempt 2: `--mamba-radix-cache-strategy extra_buffer --mamba-track-interval 64 --attention-backend triton
  --moe-runner-backend triton --max-mamba-cache-size 6`
- New stage `3_pinned_memory` (before the model load): cudaHostRegister of 64 MiB and then 1 GiB with Pennyroyal's
  exact call and flags, a copy to the GPU, then unregister. Failure stops the run.
- Unchanged and already proven on Kaggle in run 1: stages 0-3 (gates, merge plus full SHA256 of 124.8 GB, install,
  preflight).

## Questions
1. Do you accept the R3 ledger?
2. With the quoted rules (`evidence/`), will either attempt fail in argument resolution or in `resolve_max_num_reqs`?
3. Is the pin probe a valid test of the memlock question? For example: is 1 GiB enough, and is a single process enough
   when TP4 pins from 4 processes?
4. GO or NO-GO for the second 4×L4 run (≤ 50 min, ~1.7 quota-hours)?

## Reproduce
`python build/validate_v3.py`
