# E11-03 GPU validation · run 20261017T050445Z

Driver 580.178.04 (CUDA 13.0) · shims {"renamed_wheels": ["flashinfer_jit_cache-0.6.18cu130-cp39-abi3-manylinux_2_28_x86_64.whl -> flashinfer_jit_cache-0.6.18+cu130-cp39-abi3-manylinux_2_28_x86_64.whl"], "libcudart.so": "created", "lib64": "created", "libcuda.so": "shim -> /usr/local/nvidia/lib64/libcuda.so.1"}

## Preflight

| step | ok | detail |
|---|---|---|
| versions | OK | {"torch": "2.13.0", "sglang": "0.5.21", "sglang-kernel": "0.4.7", "flashinfer-python": "0.6.18", "flashinfer-jit-cache": "0.6.18+cu130", "transformers": "5.12.1", "triton": "3.7.1", "nvidia-cuda-nvcc": "13.0.88", "nvidia |
| torch_cuda_gemm (fatal) | OK | {"torch": "2.13.0+cu130", "cuda": "13.0", "n_gpu": 4, "names": ["NVIDIA L4", "NVIDIA L4", "NVIDIA L4", "NVIDIA L4"], "capability": [[8, 9], [8, 9], [8, 9], [8, 9]], "bf16_gemm_tflops": 75.9, "rel_err": 0.0027702548541128 |
| sgl_kernel_import (fatal) | OK | {"file": "/tmp/sglang-venv/lib/python3.12/site-packages/sgl_kernel/__init__.py", "ops": 1} |
| nccl_allreduce_tp4 (fatal) | OK | [{"rank": 0, "ok": true, "ms_per_64MB": 20.89}, {"rank": 1, "ok": true, "ms_per_64MB": 20.89}, {"rank": 2, "ok": true, "ms_per_64MB": 20.85}, {"rank": 3, "ok": true, "ms_per_64MB": 20.86}] |
| triton_kernel | OK | {"triton": "3.7.1", "max_err": 0.0} |
| flashinfer_decode_hd256 | OK | {"flashinfer": "0.6.18", "bf16_max_abs_err": 0.00030272454023361206, "fp8_kv_max_abs_err": 0.00036012381315231323} |
| nvcc_jit_torch_extension | OK | {"sum": 2000.0} |
| cutlass_dsl_import | OK | {"cutlass": "4.6.2"} |

## Engines

### A_int4 · OK
- launch: [(0, [], 'ready in 601s')]
- KV tokens 604274 · max running 32 · attention flashinfer · mamba extra_buffer · SSM float32 · draft attention None
- smoke: {'tokens': 16, 'text': '\n\n1 2 3 4 5 6 7 8'}

| requests | total tok/s | log gen tok/s | per-request tok/s | accept len | wall s | ok |
|---|---|---|---|---|---|---|
| 1 | 45.6 | 45.6 | 45.6 | - | 89.9 | 1/1 |
| 4 | 145.4 | 147.6 | 36.4 | - | 112.7 | 4/4 |
| 8 | 229.3 | 248.5 | 28.7 | - | 142.9 | 8/8 |
| 12 | 289.4 | 317.2 | 24.1 | - | 169.9 | 12/12 |
| 16 | 347.1 | 370.1 | 21.7 | - | 188.8 | 16/16 |
| 24 | 293.2 | - | 18.7 | - | 335.3 | 24/24 |
| 32 | 349.3 | - | 20.0 | - | 375.3 | 32/32 |
- long 12 x 24576: 282.4 tok/s total, accept -, wall 1028.9 s, usage max 0.53

### B_int4_mtp · OK
- launch: [(0, [], 'ready in 166s')]
- KV tokens 474203 · max running 32 · attention flashinfer · mamba extra_buffer · SSM float32 · draft attention None
- smoke: {'tokens': 16, 'text': '\n\n1 2 3 4 5 6 7 8'}
- greedy_vs_A: 0/4 identical, mean common prefix 152.2 of 512 tokens

| requests | total tok/s | log gen tok/s | per-request tok/s | accept len | wall s | ok |
|---|---|---|---|---|---|---|
| 1 | 55.1 | 81.8 | 55.1 | 3.17 | 74.3 | 1/1 |
| 4 | 205.1 | 218.0 | 53.4 | 3.10 | 79.9 | 4/4 |
| 8 | 297.3 | 343.0 | 38.1 | 3.09 | 110.2 | 8/8 |
| 12 | 263.6 | - | 33.7 | 3.11 | 186.4 | 12/12 |
| 16 | 314.5 | - | 30.9 | 3.17 | 208.4 | 16/16 |
| 24 | 275.5 | - | 15.6 | 3.12 | 356.8 | 24/24 |
| 32 | 281.8 | - | 14.3 | 3.10 | 465.1 | 32/32 |
- long 12 x 24576: 262.3 tok/s total, accept 3.02, wall 1109.4 s, usage max 0.8

### C_int4_dflash2 · OK
- launch: [(0, [], 'ready in 161s')]
- KV tokens 394559 · max running 32 · attention flashinfer · mamba extra_buffer · SSM float32 · draft attention flashinfer
- smoke: {'tokens': 16, 'text': '\n\n1 2 3 4 5 6 7 8'}
- greedy_vs_A: 1/4 identical, mean common prefix 155.0 of 512 tokens

| requests | total tok/s | log gen tok/s | per-request tok/s | accept len | wall s | ok |
|---|---|---|---|---|---|---|
| 1 | 93.7 | 90.0 | 93.7 | 3.89 | 43.7 | 1/1 |
| 4 | 250.1 | 276.3 | 68.5 | 4.37 | 65.5 | 4/4 |
| 8 | 267.9 | - | 49.5 | 4.74 | 122.3 | 8/8 |
| 12 | 307.3 | - | 36.4 | 4.70 | 159.9 | 12/12 |
| 16 | 282.9 | - | 24.4 | 4.95 | 231.7 | 16/16 |
| 24 | 273.4 | - | 17.3 | 4.75 | 359.6 | 24/24 |
| 32 | 284.1 | - | 15.0 | 4.76 | 461.3 | 32/32 |
- long 12 x 24576: 258.3 tok/s total, accept 4.20, wall 1141.7 s, usage max 0.73

### X_C_int4_dflash2_extra · OK
- launch: [(0, [], 'ready in 136s')]
- KV tokens 394559 · max running 32 · attention flashinfer · mamba extra_buffer · SSM float32 · draft attention flashinfer
- smoke: {'tokens': 16, 'text': '\n\n1 2 3 4 5 6 7 8'}

### D_C_int4_dflash2_ssm_fp16 · OK
- launch: [(0, [], 'ready in 146s')]
- KV tokens 386563 · max running 32 · attention flashinfer · mamba extra_buffer · SSM float16 · draft attention flashinfer
- smoke: {'tokens': 16, 'text': '\n\n1 2 3 4 5 6 7 8'}
- greedy_vs_fp32: 1/4 identical, mean common prefix 165.8 of 512 tokens

| requests | total tok/s | log gen tok/s | per-request tok/s | accept len | wall s | ok |
|---|---|---|---|---|---|---|
| 1 | 107.5 | 109.8 | 107.5 | 4.31 | 38.1 | 1/1 |
| 4 | 278.2 | 301.2 | 72.1 | 4.51 | 58.9 | 4/4 |
| 8 | 355.5 | 482.7 | 48.6 | 4.85 | 92.2 | 8/8 |
| 12 | 403.7 | 492.5 | 34.8 | 4.62 | 121.8 | 12/12 |
| 16 | 380.6 | - | 33.6 | 4.78 | 172.2 | 16/16 |
| 24 | 390.4 | - | 26.8 | 4.64 | 251.8 | 24/24 |
| 32 | 349.1 | - | 15.0 | 4.68 | 375.4 | 32/32 |

### S_C_int4_dflash2_sanity · OK
- launch: [(0, [], 'ready in 136s')]
- KV tokens 394559 · max running 32 · attention flashinfer · mamba extra_buffer · SSM float32 · draft attention flashinfer
- smoke: {'tokens': 16, 'text': '\n\n1 2 3 4 5 6 7 8'}

## Speed-up vs A (total tok/s)

| requests | B_int4_mtp | C_int4_dflash2 | X_C_int4_dflash2_extra | D_C_int4_dflash2_ssm_fp16 | S_C_int4_dflash2_sanity |
|---|---|---|---|---|---|
| 1 | 1.21x | 2.05x | - | 2.36x | - |
| 4 | 1.41x | 1.72x | - | 1.91x | - |
| 8 | 1.30x | 1.17x | - | 1.55x | - |
| 12 | 0.91x | 1.06x | - | 1.39x | - |
| 16 | 0.91x | 0.82x | - | 1.10x | - |
| 24 | 0.94x | 0.93x | - | 1.33x | - |
| 32 | 0.81x | 0.81x | - | 1.00x | - |

## Sanity solve (S_C_int4_dflash2_sanity): 7/8 solved · FP8 expectation 6.74

| puzzle | hit | mode | think tokens | FP8 think | FP8 p | accept | sec |
|---|---|---|---|---|---|---|---|
| 2ba387bc#0 | 1 | natural | 7216 | 5355 | 0.907 | 4.99 | 354.8 |
| bf45cf4b#0 | 1 | natural | 15136 | 17485 | 0.907 | 5.26 | 367.7 |
| 58f5dbd5#0 | 1 | natural | 21954 | 22674 | 0.838 | 4.90 | 373.0 |
| 136b0064#0 | 1 | natural | 44718 | 23164 | 0.834 | 3.86 | 810.1 |
| 31f7f899#0 | 1 | natural | 31393 | 25425 | 0.764 | 3.77 | 632.7 |
| 1818057f#0 | 0 | natural | 31721 | 26891 | 0.725 | 5.50 | 562.3 |
| e8686506#0 | 1 | natural | 31692 | 27730 | 0.892 | 3.88 | 708.1 |
| 7491f3cf#0 | 1 | natural | 32618 | 31874 | 0.87 | 3.91 | 795.6 |
## Extra tests (rev 3) on C_int4_dflash2

- **contcache** (forced-close continuation hits the prefix cache): pass True · min cached ratio 0.979 · warm vs cold speed-up 11.06x · 
- **longctx** (4 x ~85458 ctx, 1024 new): 16.3 tok/s · accept 4.58 · usage max 0.87 · ok 4/4 · 
- **contam** (greedy alone vs in a 16-batch): 0/4 identical · mean common prefix 15.5 of 512 · 
- **temp** (12 x 4096): T1.0 285.8 tok/s accept 4.34 · T0.6 241.7 tok/s accept 4.60 · T0.6/T1.0 0.846


Total time 203 min · best config (12 requests, complete sweeps): C_int4_dflash2