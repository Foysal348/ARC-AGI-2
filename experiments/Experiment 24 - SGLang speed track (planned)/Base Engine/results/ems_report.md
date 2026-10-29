# EMS SAFE: SGLang engine-config matrix

profile EMS, rerun False, engine sglang-0.5.21-int4-dflash2-w6-s16
prompts {'short_p50': 3852.0, 'long_p50': 40877.0, 'xl_p50': 90563.0}
reference A/A probe flips: 0
gate reference: target_only (same INT4 checkpoint, no speculation) | oracle flag: target_only | reference: ref_dflash_w6_s16 {'strategy': 'extra_buffer_lazy', 'mamba_cache': 68, 'no_prefill_graph': True, 'mem': '0.83'}
EMS cell error: none
tok/s = SGLang decode-line rate in a steady window (decode only, not end-to-end); adoptable = gate PASS vs the target-only oracle (PASS_EXACT, or BOUNDARY = only near-tie divergences, accepted under the documented tolerance but not exact: confirm on ARC tasks before production), complete probe, no stream error, real running share >= 0.80, no canary leak, XL parity PASS; certified_exact = adoptable and PASS_EXACT
oracle canary: {'level': 16, 'own_hits': 16, 'foreign_leaks': 0, 'errors': 0, 'server_not_idle': False, 'informative': True, 'classification': 'PASS_NO_OBSERVED_LEAK'}
notes: {"ntn": "NTN flag bundle + --max-mamba-cache-size 85 (5 x 17) so 16 streams fit; attention flashinfer / linear triton = the reference's values; max-prefill 4096 moot under chunk 2048"}
FlashInfer 0.7 overlay (PCIe-IPC arms): {"ok": true, "dir": "/tmp/ems-fi07", "pip_rc": 0, "pip_tail": "Processing /tmp/ems-fi07-wheels/flashinfer_python-0.7.0.post1-py3-none-any.whl\nProcessing /tmp/ems-fi07-wheels/flashinfer_jit_cache-0.7.0.post1+cu130-cp39-abi3-manylinux_2_28_x86_64.whl\nProcessing /tmp/ems-fi07-wheels/nccl_extensions-0.1.0-cp312-cp312-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl\nInstalling collected packages: ncc
PR #45151 patch: {'applied': False, 'why': 'SAFE part: not needed'}
DFlash reference vs target-only (same INT4, greedy 256 tok): PASS; 1,024-tok probe: identical 1 of 12, large-margin flips 0 (target-only probe A/A flips 0)
dropped arms: {"linattn_flashinfer": "FlashInfer GDN needs SM90+: kernels/gdn_flashinfer.py:135-136 sets available = capability >= 9 and FlashInferGDNKernel() raises 'FlashInfer GDN kernels are not available' (:184-187), so the server dies at startup on the L4", "linattn_cutedsl": "CuTe DSL GDN prefill is SM100+ only and falls back to Triton (gdn_backend.py:304-324); GDN verify stays Triton unless FlashInfer (gdn_backend.py:347-360); under DFlash / MTP every target step is a verify, so the arm would run the reference's kernels"}
2 x TP2 GPU pairs: [[0, 1], [2, 3]] links {'0-1': 'PHB', '0-2': 'PHB', '0-3': 'PHB', '1-2': 'PHB', '1-3': 'PHB', '2-3': 'PHB'}

| arm | ok | gate | streams (8k / 40k) | capped | 8k tok/s | x ref | 40k tok/s | x ref | accept 8k / 40k | step ms 8k / 40k | KV tokens | log errors | probe flips vs ref / vs target | gate vs DFlash ref / skip / error | XL tok/s (streams, ctx/req) | x ref | accept / step ms XL | PCIe-IPC workspace built | adoptable / gate class | XL gate | canary leaks | prefix-cache hit 8k / 40k (incl. warm) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ref_dflash_w6_s16 | True | PASS | 16 / 16 |  | 583 | 1.00 | 521 | 1.00 | 4.21 / 4.11 | 115.7 / 127.3 | 852152 |  | — / 0 |  | 370 (8, 16774) | 1.00 | 4.01 / 86.8 |  | True BOUNDARY | PASS | 0 | 0.99 / 0.86 |
| target_only | True | reference | 16 / 16 |  | 405 | 0.69 | 354 | 0.68 | — / — | 39.5 / 45.1 | 1576170 |  | 0 / — |  | 275 (15, 9974) | 0.74 | — / 54.6 |  | False  |  |  | 0.23 / 0.87 |
| dflash_w6_pcie | True | PASS | 16 / 16 |  | 230 | 0.39 | 202 | 0.39 | 4.32 / 3.91 | 299.6 / 309.9 | 852152 |  | 0 / 0 | PASS | 233 (8, 16114) | 0.63 | 4.53 / 155.5 | True | False BOUNDARY | PASS | 0 | 0.20 / 0.87 |
| mtp_3_1_4 | True | PASS | 16 / 16 |  | 478 | 0.82 | 402 | 0.77 | 2.88 / 2.94 | 96.4 / 117.3 | 1142688 |  | 0 / 0 | PASS | 296 (11, 13332) | 0.80 | 3.05 / 113.5 |  | True BOUNDARY | PASS | 0 | 0.22 / 0.87 |
| mtp_3_1_4_pr45151 | False | None | — / — |  | — | — | — | — | — / — | — / — | None |  | — / — | OTHER_PART: runs in the EMS RISKY notebook | — (—, —) | — | — / — |  | False  |  |  | — / — |
| mtp_3_1_4_pcie | True | PASS | 16 / 16 |  | 248 | 0.42 | 205 | 0.39 | 3.08 / 2.80 | 198.7 / 217.8 | 1142688 |  | 0 / 0 | PASS | 192 (11, 12952) | 0.52 | 2.95 / 169.3 | True | True BOUNDARY | PASS | 0 | 0.19 / 0.87 |
| ntn_int4 | False | None | — / — |  | — | — | — | — | — / — | — / — | None |  | — / — | OTHER_PART: runs in the EMS RISKY notebook | — (—, —) | — | — / — |  | False  |  |  | — / — |
| dflash_w6_s20 | True | PASS | 20 / 20 |  | 597 | 1.02 | 536 | 1.03 | 4.17 / 4.21 | 140.3 / 157.1 | 779192 |  | 0 / 0 | PASS | 364 (7, 18615) | 0.99 | 4.22 / 81.3 |  | True BOUNDARY | PASS | 0 | 0.29 / 0.89 |
| dflash_w6_s24 | True | INCOMPLETE | — / — | CAPPED | — | — | — | — | — / — | — / — | 706232 | {'out of memory': 1, 'Traceback': 1} | — / — | INCOMPLETE | — (—, —) | — | — / — |  | False INCOMPLETE | INCOMPLETE | 0 | — / — |
| ntn_fp8 | False | None | — / — |  | — | — | — | — | — / — | — / — | None |  | — / — | OTHER_PART: runs in the EMS RISKY notebook | — (—, —) | — | — / — |  | False  |  |  | — / — |
| mtp_2_1_3 | True | PASS | 16 / 16 |  | 509 | 0.87 | 403 | 0.77 | 2.47 / 2.45 | 77.7 / 97.6 | 1183245 |  | 0 / 0 | PASS | 269 (11, 13215) | 0.73 | 2.45 / 100.2 |  | True BOUNDARY | PASS | 0 | 0.14 / 0.87 |
| mtp_3_1_4_mem86 | True | PASS | 16 / 16 |  | 483 | 0.83 | 408 | 0.78 | 2.98 / 3.00 | 96.9 / 117.3 | 1223053 |  | 0 / 0 | PASS | 296 (12, 12393) | 0.80 | 2.93 / 118.9 |  | True BOUNDARY | PASS | 0 | 0.19 / 0.87 |
| mtp_replay_fp16 | False | None | — / — |  | — | — | — | — | — / — | — / — | None |  | — / — | OTHER_PART: runs in the EMS RISKY notebook | — (—, —) | — | — / — |  | False  |  |  | — / — |
| mtp_replay_fp32 | False | None | — / — |  | — | — | — | — | — / — | — / — | None |  | — / — | OTHER_PART: runs in the EMS RISKY notebook | — (—, —) | — | — / — |  | False  |  |  | — / — |
| dflash_w4 | True | PASS | 16 / 16 |  | 592 | 1.01 | 526 | 1.01 | 3.30 / 3.31 | 89.2 / 100.9 | 917816 |  | 0 / 0 | PASS | 363 (9, 15580) | 0.98 | 3.41 / 83.5 |  | True BOUNDARY | PASS | 0 | 0.16 / 0.87 |
| dflash_w8 | True | PASS | 16 / 16 |  | 573 | 0.98 | 477 | 0.92 | 5.00 / 4.47 | 138.8 / 149.9 | 786488 |  | 0 / 0 | PASS | 363 (7, 18556) | 0.98 | 4.94 / 94.3 |  | True BOUNDARY | PASS | 0 | 0.18 / 0.87 |
| mtp_2_1_3_s20 | True | PASS | 20 / 20 |  | 542 | 0.93 | 413 | 0.79 | 2.48 / 2.45 | 92.0 / 118.2 | 1120156 |  | 0 / 0 | PASS | 263 (11, 13199) | 0.71 | 2.38 / 100.5 |  | True BOUNDARY | PASS | 0 | 0.30 / 0.88 |
| tp2x2_dflash_w6 | False | None | — / — |  | — | — | — | — | — / — | — / — | None |  | — / — | OTHER_PART: 2 x TP2 runs in the EMS RISKY notebook | — (—, —) | — | — / — |  | False  |  |  | — / — |
| tp2x2_mtp_best | False | None | — / — |  | — | — | — | — | — / — | — / — | None |  | — / — | OTHER_PART: 2 x TP2 runs in the EMS RISKY notebook | — (—, —) | — | — / — |  | False  |  |  | — / — |
