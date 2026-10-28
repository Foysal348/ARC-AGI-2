# Engine matrix rev 2g GPU results: EMS SAFE, EMS RISKY v1, VMS (2026-10-26)

**Sources:**
- EMS SAFE: `hosen42/q27-xhigh-ems-safe-engine-matrix-sglang`, COMPLETE. Output in `_scratch/ems_out/safe/`.
- VMS: `hosen42/q27-xhigh-vms-engine-matrix-vllm`, COMPLETE. Output in `_scratch/ems_out/vms/`.
- EMS RISKY v1: `foysalemonshanto/q27-xhigh-ems-risky-engine-matrix-sglang`, v1 complete after 8,045 s. The latest version of the kernel shows CANCEL_ACKNOWLEDGED, so `kernels output` returned nothing; the report table was rebuilt from the v1 session log, `_scratch/ems_out/risky/v1_stdout.txt`.

**What the columns mean:**
- tok/s is the SGLang decode-line rate in a steady window.
- The 8k band is about 3.9k-token prompts, the 40k band about 40.9k, the XL band about 90.6k.
- Streams are the real running count.
- The gate is greedy parity against the target-only oracle (same INT4 checkpoint).

## SGLang 0.5.21 (INT4)

| arm | run | streams 8k / 40k | 8k tok/s | 40k tok/s | XL tok/s (streams) | accept 8k / 40k | gate | note |
|---|---|---|---|---|---|---|---|---|
| ref DFlash W6, 16 streams | SAFE | 16 / 16 | 583 | 521 | 370 (8) | 4.21 / 4.11 | PASS (BOUNDARY), XL PASS, canary 0 | production |
| ref DFlash W6, 16 streams | RISKY | 16 / 16 | 572 | 544 | 403 (8) | 4.14 / 4.33 | PASS (BOUNDARY), XL PASS, canary 0 | production (second measurement) |
| DFlash W6, 20 streams | SAFE | 20 / 20 | 597 | 536 | 364 (7) | 4.17 / 4.21 | PASS | +2-3 % vs ref (within run-to-run spread: the ref itself is 572-583 / 521-544) |
| DFlash W4 | SAFE | 16 / 16 | 592 | 526 | 363 (9) | 3.30 / 3.31 | PASS | = ref |
| DFlash W8 | SAFE | 16 / 16 | 573 | 477 | 363 (7) | 5.00 / 4.47 | PASS | worse at 40k |
| DFlash W6, 24 streams | SAFE | — | — | — | — | — | INCOMPLETE | out of memory + traceback, CAPPED |
| DFlash W6 + PCIe-IPC AR | SAFE | 16 / 16 | 230 | 202 | 233 (8) | 4.32 / 3.91 | PASS | 0.39x: FlashInfer 0.7 PCIe-IPC is much slower than NCCL here |
| MTP 3/1/4 | SAFE / RISKY | 16 / 16 | 478 / 492 | 402 / 396 | 296 / 292 | 2.9-3.0 | PASS | 0.82-0.86x |
| MTP 3/1/4 + #45151 | RISKY | 16 / 16 | 492 | 394 | 295 (12) | 2.98 / 2.90 | PASS | same speed; KV 1.29M vs 1.14M tokens |
| MTP 3/1/4 + PCIe-IPC | SAFE | 16 / 16 | 248 | 205 | 192 | 3.08 / 2.80 | PASS | 0.42x |
| MTP 2/1/3 | SAFE | 16 / 16 | 509 | 403 | 269 | 2.47 / 2.45 | PASS | 0.87x |
| MTP 2/1/3, 20 streams | SAFE | 20 / 20 | 542 | 413 | 263 | 2.48 / 2.45 | PASS | best MTP in SAFE |
| MTP 3/1/4, mem 0.86 | SAFE | 16 / 16 | 483 | 408 | 296 | 2.98 / 3.00 | PASS | — |
| MTP ReplaySSM fp16 | RISKY | 16 / 16 | 497 | 410 | 313 | 2.91 / 2.90 | PASS | no_buffer: carries the #43387 / #43749 risk (not adoptable without those fixes) |
| MTP ReplaySSM fp32 | RISKY | 16 / 16 | 501 | 415 | 288 | 2.91 / 2.98 | PASS | same |
| MTP ReplaySSM fp32, 20 streams | RISKY | 20 / 20 | 543 | 428 | 275 | 3.01 / 2.95 | PASS | same |
| NTN flags on INT4 | RISKY | 16 / 16 | 444 | 363 | 230 (7) | 2.95 / 2.88 | PASS | 0.78x |
| NTN flags on FP8 | RISKY | 16 / 16 | 400 | 343 | 151 (4) | 2.97 / 2.97 | diagnostic | 0.70x; KV only 419k tokens |
| target-only | SAFE / RISKY | 16 / 16 | 405 / 400 | 354 / 348 | 275 / 274 (15) | — | oracle | — |
| 2 x TP2 DFlash W6 | RISKY | 8 / 3 per replica | 658 (sum) | 308 (sum) | — | — | PASS | 1.15x at 8k, but 152k KV per replica: only 3 streams at 40k (0.57x) -> unusable for our long contexts |
| 2 x TP2 MTP ReplaySSM fp32 | RISKY | 8 / 4 | 511 | 279 | — | — | PASS | same KV limit |
| DFlash W6 / MTP + PCIe-IPC | RISKY | — | — | — | — | — | — | not run: the overlay failed (no pip in the venv; fixed in SAFE rev 2g-d) |

## vLLM 0.32.0

| arm | streams 8k / 40k | 8k tok/s | 40k tok/s | XL tok/s (streams) | gate |
|---|---|---|---|---|---|
| ref FP8 MTP3, 16 streams | 16 / 16 | 434 | 399 | 281 (10) | PASS (BOUNDARY) |
| FP8 MTP3, 20 streams | 20 / 20 | 452 | 423 | 281 | PASS |
| FP8 MTP3, 24 streams | 24 / 24 | 472 | 424 | 256 | PASS |
| FP8 MTP3, SSM fp16 | 16 / 16 | 450 | 414 | 312 (11) | PASS |
| FP8 MTP2 / MTP4 | 16 / 16 | 448 / 399 | 397 / 381 | 260 / 267 | PASS |
| FP8 MTP3, Triton linear | 16 / 16 | 438 | 398 | 283 | PASS |
| FP8 MTP3 + PCIe-IPC | 16 / 16 | 297 | 278 | 206 | PASS (0.68x) |
| FP8 MTP3, prefix caching off | 16 / 2 | 435 | 3 | 0 | 40k recomputes the prompt |
| INT4 MTP3, 16 streams | 16 / 16 | 495 | 434 | 321 (12) | diagnostic (no INT4 oracle) |
| INT4 MTP3, 20 streams | 20 / 20 | 518 | 439 | 332 (12) | diagnostic |
| 2 x TP2 FP8 / INT4 MTP3 | 14-16 / 2 | 191 / 483 | 22 / 92 | — | KV limit, unusable |

The FP8 oracle (target-only, prefix caching off) gave a gate reference but no speed row.

## Findings

1. **No configuration reaches 800-900 tok/s on this stack.**
   - Best single server: SGLang DFlash W6 at 20 streams, 597 / 536 / 364 (8k / 40k / XL).
   - The production reference measured 572-583 / 521-544 / 370-403 across the two runs.
   - The 20-stream gain (+2-3 %) is inside that spread.
2. DFlash W4 and W6 are equal. W8 loses at 40k.
3. **PCIe-IPC all-reduce (FlashInfer 0.7) is 0.39-0.42x on SGLang and 0.68x on vLLM: drop it.**
4. The MTP family is 0.82-0.95x of DFlash on SGLang. #45151 frees KV (+13 %), but gives no speed.
5. 2 x TP2 is faster only at 8k. The KV per replica (152k) cannot hold the 40k-90k contexts.
6. The vLLM 0.32 best (INT4 MTP3 at 20 streams, 518 / 439) stays below SGLang DFlash.
7. **Correctness.**
   - Every gated arm: PASS (BOUNDARY), XL parity PASS, canary leaks 0.
   - DFlash reference vs the target-only oracle: PASS, with 0 large-margin flips in both runs.
   - This does not cover the #43567 cache path, which E17 tests.
8. The XL band (90k prompts) runs at only 7-8 streams on the production KV; there, decode is 370-403 tok/s.
