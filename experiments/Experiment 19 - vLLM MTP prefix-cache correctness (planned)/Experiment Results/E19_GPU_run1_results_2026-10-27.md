# E19 GPU run 1 (2026-10-27): S1's engine config shows no detectable corruption; MTP + prefix caching OFF does

Three kernels, all COMPLETE:
- P1 `foysalemonshanto/q27-xhigh-e19-mtppc-p1-d-c` (24e40238), arms D and C;
- P2 `foysalemonshanto/q27-xhigh-e19-mtppc-p2-e-a` (e057849b), arms E and A;
- P3 `hosen42/q27-xhigh-e19-mtppc-p3-b-k` v2 (50ee5178), arms B and K.

Outputs are in `GPU_run1_2026-10-27/` (p1, p2, p3, and compare/ = `src/e19_compare.py` over all three). Engine: vLLM 0.19.0, S1's flags, 4x L4.

## Verdicts (all 6 arms completed every planned row)

| Arm | Config | Final | Rows | Serial A/A | Cold vs warm | Churn vs serial | Cont 8k | Cont 30k | Corrupt | MTP accept min / median | KV tokens |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | MTP off, prefix off (reference) | PASS | 60/60 | 6/6 (0) | - | 0/24 (0) | 5/6 (0) | 6/6 (0) | 0 | - | 376,320 |
| B | MTP 3, prefix OFF | **FAIL** | 60/60 | 6/6 (0) | - | 0/24 (0) | 4/6 (**2**) | 4/6 (**2**) | 0 | 0.58 / 0.79 | 337,600 |
| C | MTP off, prefix on | PASS | 66/66 | 6/6 (0) | 6/6 (0) | 0/24 (0) | 6/6 (0) | 6/6 (0) | 0 | - | 376,320 |
| **D** | **S1 exactly** | **PASS** | 66/66 | 6/6 (0) | 6/6 (0) | 0/24 (0) | 6/6 (0) | 6/6 (0) | 0 | 0.58 / 0.80 | 337,600 |
| E | S1 + --no-async-scheduling | PASS | 66/66 | 6/6 (0) | 6/6 (0) | 0/24 (0) | 6/6 (0) | 6/6 (0) | 0 | 0.57 / 0.76 | 337,600 |
| K | S1 with bf16 KV | PASS | 66/66 | 6/6 (0) | 6/6 (0) | 0/24 (0) | 6/6 (0) | 6/6 (0) | 0 | 0.59 / 0.82 | 176,800 |

Cells are identical / compared, with large-margin first-divergence flips (> 1 nat) in brackets.

## Arm B (MTP on, prefix caching off): high-confidence state divergences

| Probe | Where it diverges | Margins at the divergence | Which side is the odd one |
|---|---|---|---|
| long8k idx 10 | continuation diverges from the natural tail at token 60 | natural 10.94 nat, continuation 0.63 | the continuation (recomputed prompt + filler + 512 generated tokens) |
| long8k idx 4 | token 40 | 10.75 vs 2.25 | the continuation |
| long30k idx 6 | B's natural diverges from A's natural at token 965 | both sides confident, 10.6 / 11.3 | B's natural decode |
| long30k idx 8 | token 1005 | 11.7 / 4.0 | B's natural decode |

In arm D (same MTP, prefix caching ON / mamba align) every long natural is identical to arm A's, and every continuation reproduces its natural tail.

These are not low-margin numeric ties. They are flips of 4 to 12 nats, which is a model-state divergence. They come from vLLM 0.19's MTP path with mamba_cache_mode 'none'.

## Conclusions

1. **S1's production engine config (D) shows no detectable corruption** in 66 rows. That covers serial A/A, cache hits, 24 churn requests at up to 14 concurrent with staggered lengths, and 8k / 30k continuations. MTP acceptance never collapsed (median 0.80).
   - Sensitivity limit: about 60 requests per arm. A rare (< ~2 %) corruption cannot be excluded.
   - The trailing Mamba block over generated tokens was NOT_EXERCISED on the MTP arms.
2. **The async-scheduling race (#53919) did not show** (D = E). `--no-async-scheduling` brings no measured benefit.
3. **fp8 KV vs bf16 KV:** no correctness difference in these probes (K = D). bf16 halves the KV pool (176,800 vs 337,600 logged).
4. **Do NOT disable prefix caching while MTP is on.**
   - The "safe flag" proposed from the upstream issues (prefix caching off) is the only configuration that diverged here.
   - Mechanism UNVERIFIED (MTP + mamba cache mode 'none' in 0.19).
5. Together with the 679-trace scan (0 visible corruption), the engine is not the explanation for S1's LB. The confirmed losses are elsewhere: drain, re-prefill, coverage.
