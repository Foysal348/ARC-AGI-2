# E23 P4: SM89 block-FP8 GEMM (vLLM PR #61224) as a standalone sm_89 torch extension

The goal is to build only the PR #61224 CUTLASS block-FP8 W8A8 kernel, not a full vLLM rebuild, into `p4_sm89_fp8.so`
for sm_89. The build runs on a Kaggle CPU kernel against the vLLM 0.32.0 venv's torch. A GPU microbenchmark then runs in
the E23 GPU notebook (section 4a). It compares this kernel with vLLM 0.32's Marlin FP8 path on the same FP8 weights and
scales, using our per-rank TP4 GEMM shapes. Numerical match comes first, then latency.

Status (2026-10-29): built and checked locally. Nothing has been pushed or run on Kaggle; the coordinator audits and pushes.

## Findings (with evidence)

1. **PR #61224**: https://github.com/vllm-project/vllm/pull/61224 is "[Kernel] CUTLASS blockwise FP8 GEMM for SM89 (Ada)
   with per-token activation scales". It is open and not merged; the author is amankarki151, it was opened 2026-10-12,
   and the head is `dbe22a894e66c11e03f47a70867c93b0c26bb410`. The diff has 6 files and +822/-0 lines.
   `patches/pr61224.diff` is byte-identical to `gh pr diff 61224` (sha256 `083962b3...63da87`).
   - The PR author reports these numbers on an L4 with Qwen3-8B-FP8: 1929 -> 2739 tok/s throughput, batch 32 at
     20.16 -> 15.11 s, batch 1 at 4.77 -> 4.94 s (3.6 % slower), and gsm8k unchanged. That validation used torch 2.13.0
     +cu129 and CUDA 12.9; ours is cu130 / nvcc 13.0.88. In the PR's own GEMM table, the M <= 16 results are mixed:
     gate_up at M=1 is 432 us against Marlin's 399, and o_proj at M=1 is 19.2 against 16.0. down_proj wins at every M.
     The large wins are at M = 32..128.
   - Per the PR text, the new kernel is selected over Marlin because it flips `cutlass_scaled_mm_supports_block_fp8(89)`.
     In stock v0.32.0 that function returns false for sm_89
     ([scaled_mm_entry.cu#L161](https://github.com/vllm-project/vllm/blob/v0.32.0/csrc/quantization/w8a8/cutlass/scaled_mm_entry.cu#L161)).
     As a result, `CutlassFp8BlockScaledMMKernel` is skipped and `MarlinFP8ScaledMMLinearKernel` is chosen from
     `_POSSIBLE_FP8_BLOCK_KERNELS`
     ([kernels/linear/__init__.py#L478](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/kernels/linear/__init__.py#L478)).
2. **The patch applies to v0.32.0.** I checked against tag `v0.32.0` = `5df63a7203cdb9328d824a27b682e77c7b9928be`, using
   a sparse checkout of `csrc/`, `CMakeLists.txt` and `cmake/`. `git apply --check -v` is clean for all 6 files.
3. **CUTLASS pin.** vLLM v0.32.0 `CMakeLists.txt` L472 has `set(CUTLASS_REVISION "v4.7.1")`, used by FetchContent
   `GIT_TAG ${CUTLASS_REVISION}` with `GIT_SHALLOW TRUE`. CUTLASS `v4.7.1` = `cb4247394dd82148787aed73e5dc7cef33cbf862`.
   The PR headers say "Adapted from CUTLASS v4.7.1". `MmaMultistage` in v4.7.1 has every member the PR uses:
   `PublicPipeState`, `Detail`, `copy_tiles_and_advance`, `prologue`, `gmem_wait`, `advance_smem_*_stage` and
   `wind_down`.
4. **What the new .cu really needs.** It does not include `scaled_mm_c2x.cuh` or any vLLM epilogue header. Its includes
   are `torch/csrc/stable/tensor.h`, `torch/headeronly/core/ScalarType.h`, vLLM `csrc/torch_utils.h`, five CUTLASS
   headers, and the PR's own `cutlass_extensions/gemm/kernel/gemm_blockwise_pertoken.h`. That last header includes
   `../threadblock/mma_multistage_blockwise_pertoken.h`. The vendored tree is therefore 3 PR files (unchanged; their git
   blob ids match the diff's `index` lines) plus vLLM v0.32.0 `csrc/torch_utils.h` (unchanged, blob `f0234673`). The
   PR's edits to `scaled_mm_c2x.cu`, `scaled_mm_entry.cu` and `CMakeLists.txt` only wire the kernel into vLLM's dispatch.
   A standalone .so does not need them.
5. **Stable ABI.** In v0.32.0 the CUTLASS scaled_mm kernels live in `_C_stable_libtorch`, and the PR .cu uses
   `torch::stable::Tensor`. The standalone build mirrors that target:
   - flags: C++20, `-DUSE_CUDA`, `TORCH_TARGET_VERSION=0x020B000000000000ULL`, `CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1`
     and `-DENABLE_FP8`;
   - torch's `COMMON_NVCC_FLAGS`, with the four `__CUDA_NO_HALF*/BFLOAT16` defines undone, as vLLM does for CUDA >= 12;
   - `-gencode=arch=compute_89,code=sm_89` only.

   The binding (`src/p4_binding.cpp`) registers one op the way vLLM does (`STABLE_TORCH_LIBRARY` + `_IMPL(CUDA)` +
   `TORCH_BOX`), under its own namespace: `torch.ops.p4_sm89.cutlass_scaled_mm_blockwise_sm89_fp8`. torch 2.13.0 has the
   `stableivalue_conversions.h` string leak that vLLM patches for 2.13. It only affects string arguments, and our op has
   none.
6. **Torch / CUDA ABI.** The vLLM 0.32.0 lock (Exp 14 `vllm_0_32_0_requirements.lock`, 199 pins) is the wheelhouse
   `foysalemonshanto/vllm-0-32-0-cu13-wheelhouse`, installed by its `install_wheelhouse.py`. It pins:
   - torch 2.13.0 (cu130), cuda-toolkit 13.0.3.0;
   - nvidia-cuda-nvcc / crt / nvvm / nvjitlink 13.0.88, cuda-runtime 13.0.96, cublas 13.1.1.3;
   - ninja 1.13.2;
   - **nvidia-cuda-cccl 13.3.4.3.1 (CCCL 3.4.3)**, while cuda-toolkit 13.0.3 itself pins `nvidia-cuda-cccl==13.0.85`
     (CCCL 3.0.1).

   The kernel builds first with the venv's CCCL. If that does not compile, it retries with the toolkit-matched CCCL
   13.0.85 (sha256-pinned PyPI wheel) first on the include path. Other facts:
   - The pip nvcc wheel has no `cuobjdump`, and PyPI's `nvidia-cuda-cuobjdump` starts at 13.2.51. The kernel downloads
     13.2.51 (sha256-pinned), and `p4_build.py` also parses the `.nv_fatbin` section itself. I tested that parser on
     vLLM 0.32.0's own `_C_stable_libtorch.abi3.so`: it found 489 entries, with ELF archs 75/80/86/89/90/100/120 and
     PTX 80/89/90.
   - `nvcc.profile` adds `-isystem $(TOP)/$(_TARGET_DIR_)/include/cccl`, so the build also passes the CUDA include dir and
     the CCCL dir explicitly.
7. **Shapes.** I read them from our checkpoint config (`foysalemonshanto/qwen3-8-27b-fp8-repacked-v1`, the E22 model
   path), the local copy in the Exp 1 build sources, and the vLLM 0.32 Qwen3.5 / Qwen3-Next / GDN layer code.
   - Config: `quant_method fp8`, `fmt e4m3`, `activation_scheme dynamic`, **`weight_block_size [128, 128]`** (confirmed).
   - Model: hidden 5120, intermediate 17408, 64 layers (48 `linear_attention` GDN + 16 `full_attention`, interval 4).
     Full attention has 24 heads x head_dim 256 with `attn_output_gate` (q_proj = 2 x 6144) and 4 KV heads. GDN has 16 key
     heads x 128 (key_dim 2048) and 48 value heads x 128 (value_dim 6144). There is 1 MTP layer (full attention + MLP).
   - BF16, not FP8 (`modules_to_not_convert`): `in_proj_a/b` (`in_proj_ba`), `conv1d`, norms, `A_log`, `dt_bias`,
     `mlp.gate`, `shared_expert_gate`, `embed_tokens`, `lm_head`, `mtp.fc`.
8. **The exact vLLM paths the bench calls.**
   - Marlin (`MarlinFP8ScaledMMLinearKernel`,
     [scaled_mm/marlin.py#L29](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/kernels/linear/scaled_mm/marlin.py#L29)):
     at load it runs `process_fp8_weight_block_strategy` and then
     `prepare_fp8_layer_for_marlin(layer, size_k_first=False)`. That repacks with `gptq_marlin_repack(num_bits=8)`,
     expands the block scales to group-128 scales in bf16, and fuses the exponent bias. Per call it runs
     `apply_fp8_marlin_linear(...)` ->
     `torch.ops.vllm.marlin_gemm` ([marlin_utils.py#L500](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L500))
     -> `torch.ops._C.marlin_gemm` with `b_q_type float8_e4m3fn` and `use_fp32_reduce=True`. vLLM keeps a persistent
     workspace, so the bench creates one with `marlin_make_workspace_new(dev, 4)`; passing `None` would allocate one per
     call.
   - CUTLASS on SM89 with the PR (`CutlassFp8BlockScaledMMKernel`,
     [scaled_mm/cutlass.py#L277](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py#L277)):
     the activations go through `QuantFP8(group (1,128), column_major_scales=True, use_ue8m0=False)`, which is
     `per_token_group_quant_fp8(x, 128, column_major_scales=True)` ->
     `torch.ops._C.per_token_group_fp8_quant`. The GEMM is then `ops.cutlass_scaled_mm(A, B.T, scale_a=As, scale_b=Bs.T)`.
     The op layout is:

     | Operand | Shape | Layout |
     |---|---|---|
     | `out` | [M, N] | bf16, row-major |
     | `a` | [M, K] | fp8, row-major |
     | `b` | [K, N] | fp8, column-major, = `weight[N, K].t()` (ld = K) |
     | `a_scales` | [M, K/128] | fp32, strides (1, M) |
     | `b_scales` | [K/128, N/128] | fp32, = `weight_scale_inv.t()`, strides (1, K/128) |

     The kernel requires K % 16 == 0 and N % 16 == 0, does not support bias, and does not support split-K. M picks the
     tile: <= 16 -> 16x128x128 s5; <= 32 -> 32x64x128 s4; <= 64 -> 64x128x128 s3; else 128x128x128 s3 swz4.

## Per-rank TP4 GEMM shapes (Qwen3.8-27B-FP8)

| linear | FP8? | parallel | N (per rank) | K (per rank) | count / forward | bench row |
|---|---|---|---|---|---|---|
| full-attn qkv_proj (q+gate 3072, k 256, v 256) | FP8 | column (QKV) | 3584 | 5120 | 16 (+1 MTP) | qkv_proj |
| full-attn o_proj | FP8 | row | 5120 | 1536 | 16 (+1 MTP) | o_proj+gdn_out_proj |
| GDN in_proj_qkvz (q 512, k 512, v 1536, z 1536) | FP8 | column (merged) | 4096 | 5120 | 48 | gdn_in_proj_qkvz |
| GDN out_proj | FP8 | row | 5120 | 1536 | 48 | o_proj+gdn_out_proj |
| MLP gate_up_proj (2 x 4352) | FP8 | column (merged) | 8704 | 5120 | 64 (+1 MTP) | gate_up_proj |
| MLP down_proj | FP8 | row | 5120 | 4352 | 64 (+1 MTP) | down_proj |
| GDN in_proj_ba (b 12, a 12) | BF16 | column (merged; replicated only for AWQ/GPTQ/INC) | 24 | 5120 | 48 | not FP8 |
| mtp.fc | BF16 | column | 1280 | 10240 | 1 (MTP) | not FP8 |
| lm_head | BF16 | vocab | 62080 | 5120 | 1 | not FP8 |

Every FP8 N and K is a multiple of 128, so the block scales align with the TP shards (scale rows/cols = N/128, K/128).
The per-rank FP8 weight streamed per decode forward is about 6.1 GB, which is about 20 ms at L4 DRAM bandwidth. Real
weights (`--weights real/auto`) are the rank-0 shards cut from layer 3 (attention) and layer 0 (GDN + MLP), exactly as
vLLM's loaders shard them.

## Files

- `src/csrc/quantization/w8a8/cutlass/scaled_mm_blockwise_sm89_fp8.cu`: PR file, unchanged.
- `src/csrc/cutlass_extensions/gemm/kernel/gemm_blockwise_pertoken.h`: PR file, unchanged.
- `src/csrc/cutlass_extensions/gemm/threadblock/mma_multistage_blockwise_pertoken.h`: PR file, unchanged.
- `src/csrc/torch_utils.h`: vLLM v0.32.0, unchanged.
- `src/p4_binding.cpp`: our one-op stable-ABI binding.
- `src/p4_build.py`: the build plus checks, run with the venv python. It builds with `torch.utils.cpp_extension.load(...,
  build_directory, is_python_module=False)` through ninja, with `TORCH_CUDA_ARCH_LIST=8.9`. Then it checks:
  - SASS with `cuobjdump --list-elf/--list-ptx/-sass` (counts QMMA/HMMA) and with its own fatbin parse;
  - `readelf -d` NEEDED and `nm -D` undefined CUDA-driver symbols (gate: none);
  - `ptxas -v` registers and spills;
  - a CPU load in a fresh interpreter: `torch.ops.load_library`, the schema, and that a call with CPU tensors must fail
    with a CPU-dispatch error.
- `src/SOURCES.json`: provenance, sha256 and git blob of every vendored file. `src/LICENSE.vllm` (Apache-2.0) and
  `src/LICENSE.cutlass.txt` (BSD-3).
- `build_p4_kernel.py`: writes `kernel/q27-e23-p4-sm89-fp8.ipynb` (about 168 kB, under 400 kB) and
  `kernel/kernel-metadata.json`. The metadata is `foysalemonshanto/q27-e23-p4-sm89-fp8`: private, CPU, internet on,
  pinned image `sha256:37c64f7d...`, dataset_sources = the wheelhouse. The sources are embedded as a sha256-checked zip.
- `bench_fp8_gemm.py`: the GPU microbenchmark. It also has a CPU `--check-imports` mode, which the kernel runs against
  the real vLLM 0.32.
- `tools/check_includes.py`: a local include-graph check. Its results are in `tools/local_checks/`.

## Kernel steps (`q27-e23-p4-sm89-fp8`, one code cell; every step records `ok`, and failures never stop later steps)

1. **kit**: decode the embedded zip and check its sha256 and the sha256 of all 11 files.
2. **venv**: run `install_wheelhouse.py /tmp/vllm-venv` (same call as the E22 flags kit). Gate: torch 2.13.0,
   `torch.version.cuda == 13.0`, vllm 0.32.0, and `nvcc` present under `CUDA_HOME`. It records the nvcc, cccl, crt and
   ninja versions and the host c++.
3. **cutlass**: `git clone --depth 1 --branch v4.7.1 --filter=blob:none` with a sparse `include/` +
   `tools/util/include/`. Gate: `HEAD == cb4247394dd8...`. The fallback is the codeload tarball, whose pax commit must
   match.
4. **cuobjdump** (info only): `pip download nvidia-cuda-cuobjdump==13.2.51`, check sha256 `95c69d3d...`, extract
   `bin/cuobjdump`.
5. **build** (gate): `p4_build.py` with the venv CCCL. If nothing compiled, it retries with CCCL 13.0.85. The .so goes to
   `/kaggle/working/p4_sm89_fp8.so`. Gate: build, sm_89 SASS, no driver symbols, CPU load and schema.
6. **include_graph_info** (info only): the include closure against the installed headers.
7. **bench_check_imports** (gate): `bench_fp8_gemm.py --check-imports`, which checks:
   - the `.so` loads;
   - the signatures of `per_token_group_quant_fp8`, `process_fp8_weight_block_strategy`, `apply_fp8_marlin_linear`,
     `prepare_fp8_layer_for_marlin`, `marlin_make_workspace_new` and `replace_parameter`;
   - that `torch.ops.vllm.marlin_gemm` exists;
   - that the sources of `MarlinFP8ScaledMMLinearKernel` and `CutlassFp8BlockScaledMMKernel` still make the same calls
     as the bench.
8. **handoff**: copy `bench_fp8_gemm.py` and `p4_src.zip`, then print `ALL OK` or `NOT OK [failed steps]`.

The outputs in `/kaggle/working` are:
- `p4_sm89_fp8.so`
- `p4_report.json`
- `p4_build_report.json`
- `p4_build.log` and `p4_build_<attempt>.log` (with ptxas -v)
- `p4_build.ninja`
- `p4_build_driver_<attempt>.log`
- `p4_bench_check_imports.json`
- `include_graph_kaggle.json`
- `bench_fp8_gemm.py`
- `p4_src.zip`

## GPU notebook usage (section 4a, with no vLLM server holding that GPU)

```
VPY=/tmp/vllm-venv/bin/python   # same wheelhouse venv
P4=<this kernel's output under /kaggle/input>   # attach q27-e23-p4-sm89-fp8 via kernel_sources
$VPY $P4/bench_fp8_gemm.py --so $P4/p4_sm89_fp8.so --out /kaggle/working/p4_bench.json --gpu 0 \
     [--weights auto|real|synthetic] [--model-dir /kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1]
```

The defaults are M = 1,4,8,16,32,48,64,80,128, 20 warmup, 100 eager iterations (per-iteration CUDA events), and a CUDA
graph replay (GPU time, the decision number). The main process never touches CUDA, and each shape runs in a child with
a 900 s timeout. Exit codes are 0 = all gates passed, 1 = a gate or a shape failed, 2 = fatal setup, 3 = check-imports
failed. The JSON is always written.

**Per shape:**
- Correctness for each M, first:
  - Marlin vs `x @ Wdq^T` in fp32. Gate: rel Frobenius <= 1e-2.
  - CUTLASS vs `dequant(xq) @ Wdq^T`. Gate: <= 1e-2.
  - CUTLASS with row-major `a_scales` must reproduce the column-major result. Gate: <= 1e-6, plus a bitwise flag.
  - Info only: CUTLASS end to end vs the W8A16 reference (flagged above 8e-2), and CUTLASS vs Marlin.
  - The activations have a 2^U(-3,3) per-token magnitude spread, and synthetic weights have a 2^U(-2,2) per-block
    spread, so wrong per-row or per-block scale indexing would show.
- Latency for `marlin_w8a16`, `cutlass_w8a8_gemm` (pre-quantized), `cutlass_w8a8_quant_gemm` (quant + alloc + GEMM, the
  real layer cost), `quant_only` and `bf16_cublas`. It reports median/mean/p10/p90 in us and the weight GB/s.
- The weights rotate over 2..16 copies, so each kernel streams at least 144 MB (3 x L2), which rules out warm-L2 numbers.

`summary_per_m` gives the main-model sum (count x graph median) per kernel, plus `speedup_marlin_over_cutlass_quant_gemm`.
Our decode M with MTP-3 at 12..20 streams is 48..80.

## Local checks (Windows host, no GPU, no nvcc) and results

- `compile()` on every Python file and on the generated notebook code: OK. `kernel-metadata.json` parses: OK.
- Include graph (`tools/check_includes.py`) for the .cu + binding against vLLM-vendored + CUTLASS v4.7.1 + torch 2.13.0
  headers (from the cp312 manylinux wheel) + CUDA 13.0 headers (runtime 13.0.96, crt 13.0.88, cublas 13.1.1.3, read
  from the PyPI wheels):

  | CCCL | files reached | include edges | unexplained unresolved |
  |---|---|---|---|
  | 3.4.3 (venv) | 777 | 4762 | 0 |
  | 3.0.1 (fallback) | 693 | 3750 | 0 |

  Every unresolved header is a C/C++/POSIX system header or sits in a branch that is inactive here; each is listed with
  its reason in the tool (for example SYCL/HIP/ARM/MSVC/libc++/NVRTC, and CCCL `_LIBCUDACXX_HAS_SPACESHIP_OPERATOR() 0`).
- The binding compiled with clang 21 (zig 0.16, `-target x86_64-linux-gnu -std=c++20 -DUSE_CUDA
  -DTORCH_TARGET_VERSION=0x020B...`) against the real torch 2.13.0 headers. The object has the stable-library inits and
  references `vllm::cutlass_scaled_mm_blockwise_sm89_fp8(Tensor&, Tensor const&, x4)`, the same mangling as the PR's
  declaration.
- The fatbin parser was validated on vLLM 0.32.0's real `_C_stable_libtorch.abi3.so`.
- Bench (CPU, torch 2.14 cpu venv):
  - The real-checkpoint shard loader was run on a fake checkpoint with the real tensor names and full shapes. All 5
    shards match independent slicing.
  - Synthetic quantization, column-major scale views, metrics, `summarize()` and the tile map all pass.
  - The failure paths give the documented results: missing .so -> exit 2 + JSON; no CUDA -> per-shape errors, exit 1;
    unknown shape recorded; check-imports without vLLM -> exit 3.
- Notebook control flow was simulated on Windows with remapped paths. Every step records its result, and the expected
  failures (no Linux venv/nvcc) give `NOT OK [...]` with no crash. In that run the CUTLASS clone verified `cb4247394dd8`
  and the kit verified 11/11 files.

## Risks / open items

- The first real compile happens on Kaggle. There was no nvcc locally, so template errors with nvcc 13.0 + CCCL
  3.4.3/3.0.1 + C++20 can only show there. The PR author built with CUDA 12.9. The CCCL fallback covers one known
  mismatch.
- `QMMA` SASS counts are information, not a gate. The gate is sm_89 ELF present and no driver symbols.
- The bench is a microbenchmark. vLLM's torch.compile may fuse the activation quant into RMSNorm, so the real per-layer
  cost lies between `cutlass_w8a8_gemm` and `cutlass_w8a8_quant_gemm`.
- Wiring the kernel into serving is a separate step. It would need these monkey-patches:
  - `CUTLASS_BLOCK_FP8_SUPPORTED` / `CutlassFp8BlockScaledMMKernel.is_supported` -> True on sm_89;
  - `apply_block_scaled_mm` -> our op;
  - a `torch.library.register_fake` for `p4_sm89::cutlass_scaled_mm_blockwise_sm89_fp8` (it mutates `out`, returns ()),
    for torch.compile.
- L4 clocks/thermals: the bench logs `nvidia-smi` before and after. Compare kernels within one run, not across runs.
- The microbench needs the GPU free of a vLLM server. Setup uses under 1.5 GiB per shape.
