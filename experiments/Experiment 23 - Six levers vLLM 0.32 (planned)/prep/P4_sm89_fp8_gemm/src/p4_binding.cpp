// E23 P4: minimal torch stable-ABI binding for the vLLM PR #61224 SM89 block-FP8 GEMM.
//
// Exposes exactly one op, torch.ops.p4_sm89.cutlass_scaled_mm_blockwise_sm89_fp8, with the
// argument order and layout of vllm::cutlass_scaled_mm_blockwise_sm89_fp8 (PR file
// csrc/quantization/w8a8/cutlass/scaled_mm_blockwise_sm89_fp8.cu, compiled unchanged):
//
//   out      [M, N]                bf16 or fp16, row-major (written in place)
//   a        [M, K]                float8_e4m3fn, row-major (stride(1) == 1)
//   b        [K, N]                float8_e4m3fn, column-major (= weight[N, K].t())
//   a_scales [M, ceil(K/128)]      fp32, any strides (vLLM: column-major, stride (1, M))
//   b_scales [ceil(K/128), ceil(N/128)] fp32, any strides (vLLM: weight_scale_inv.t())
//
// Registered the same way vLLM 0.32 registers its _C_stable_libtorch ops
// (STABLE_TORCH_LIBRARY + STABLE_TORCH_LIBRARY_IMPL + TORCH_BOX), under our own namespace
// so it cannot collide with vLLM's torch.ops._C. Load with torch.ops.load_library(path).

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

namespace vllm {
void cutlass_scaled_mm_blockwise_sm89_fp8(
    torch::stable::Tensor& out, torch::stable::Tensor const& a,
    torch::stable::Tensor const& b, torch::stable::Tensor const& a_scales,
    torch::stable::Tensor const& b_scales);
}  // namespace vllm

STABLE_TORCH_LIBRARY(p4_sm89, m) {
  m.def(
      "cutlass_scaled_mm_blockwise_sm89_fp8(Tensor! out, Tensor a, Tensor b, "
      "Tensor a_scales, Tensor b_scales) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(p4_sm89, CUDA, m) {
  m.impl("cutlass_scaled_mm_blockwise_sm89_fp8",
         TORCH_BOX(&vllm::cutlass_scaled_mm_blockwise_sm89_fp8));
}
