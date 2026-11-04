/*
 * Flash-Next local HC branch mix (SGLANG_FLASH_NEXT_HC_MIX_CUDA=1), decode batches of 1-64 tokens.
 *
 * The same math as hyperconnection.py's _flash_next_hc_local_mix_kernel (Triton): per token i, branch c, channel n,
 *   logit = sum over 10 groups of (bf16 lowrank . int8 weight over the group's 32 columns, fp32) * fp16 scale
 *   out[i, n] = bf16(0.25 * sum over c of sigmoid(logit / 1024) * x[i, c * 2560 + n])
 * for hc = 4, hidden = 2560, lowrank = 320 only. The device code is ForgeKernel's hc_local_mix (run
 * hc_local_mix_261030-112210-4ed5, iteration 2, commit 594674a, sha256 28061c52; unchanged here): 80 CTAs x 128
 * threads, each owns 32 channels x 4 branches x all tokens; a cp.async ring of lowrank + weight groups, mma.sync
 * m16n8k16 bf16, the x slab loaded after the last weight stage. Built with --use_fast_math it is bit-identical to the
 * Triton kernel on an L4 for every t from 1 to 64 (fast math sets the sigmoid's rounding). Cold-L2 timing on a vast.ai
 * L4: 18.5 / 19.8 / 20.7 / 23.6 us at t = 16 / 32 / 40 / 64 against the Triton kernel's 23.8 / 45.2 / 46.0 / 48.4.
 */
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>

namespace sglang {
namespace device::fn_hc_local_mix {

constexpr int kHC = 4;
constexpr int kHidden = 2560;
constexpr int kRows = kHC * kHidden;     // 10240 weight rows
constexpr int kLowrank = 320;
constexpr int kGroups = 10;
constexpr int kBN = 32;                  // channels per CTA
constexpr int kThreads = 128;
constexpr int kWSlot = kHC * kBN * 32;   // 4096 B of int8 weight per stage
constexpr int kScaleBytes = kHC * kBN * kGroups * 2;  // 2560
constexpr int kXPitch = 272;             // 256 B + 16 B pad

template <int MT>
struct Cfg {
    // D1.1 lever: S = 4 for Mt <= 3; S = 3 at Mt = 4 (S = 4 would exceed 48 KB there)
    static constexpr int kStages = MT <= 3 ? 4 : 3;
    static constexpr int kTpad = 16 * MT;
    static constexpr int kASlot = kTpad * 64;
    static constexpr int kSlot = kASlot + kWSlot;
    static constexpr int kScaleOff = kStages * kSlot;
    static constexpr int kXOff = kScaleOff + kScaleBytes;
    static constexpr int kSmem = kXOff + kTpad * kXPitch;
};

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16;\n" :: "r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async16_zfill(uint32_t dst, const void* src, int src_bytes) {
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;\n" :: "r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// two int8 (bytes lo, hi of v at byte position p, p+1) -> packed bf16x2 (lo in low half). Exact.
__device__ __forceinline__ uint32_t i8x2_to_bf16x2(uint32_t v, int p) {
    float lo = (float)(int8_t)(v >> (8 * p));
    float hi = (float)(int8_t)(v >> (8 * (p + 1)));
    __nv_bfloat162 r = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&r);
}

// Issue the copies of ring stage g into its slot.
template <int MT>
__device__ __forceinline__ void issue_stage(uint32_t slot, const char* lowrank, const char* weight,
                                            int t, int n0, int g, int tid) {
    using C = Cfg<MT>;
    // lowrank rows [0, Tpad) x group g: 4 chunks of 16 B per row; rows >= t zero-filled
    #pragma unroll
    for (int idx = tid; idx < C::kTpad * 4; idx += kThreads) {
        int r = idx >> 2, ch = idx & 3;
        uint32_t dst = slot + r * 64 + ch * 16;
        if (r < t) cp_async16(dst, lowrank + (size_t)r * (kLowrank * 2) + g * 64 + ch * 16);
        else cp_async16_zfill(dst, lowrank, 0);
    }
    // weight rows c*2560 + n0 + col, group g: 2 chunks of 16 B per row, packed [c][col][32 B]
    #pragma unroll
    for (int idx = tid; idx < kHC * kBN * 2; idx += kThreads) {
        int rr = idx >> 1, h = idx & 1;
        int c = rr >> 5, col = rr & 31;
        cp_async16(slot + C::kASlot + rr * 32 + h * 16,
                   weight + (size_t)(c * kHidden + n0 + col) * kLowrank + g * 32 + h * 16);
    }
}

template <int MT>
__global__ void __launch_bounds__(kThreads, 2)
hc_local_mix_kernel(const __nv_bfloat16* __restrict__ lowrank, const int8_t* __restrict__ weight,
                    const __half* __restrict__ scale, const __nv_bfloat16* __restrict__ x,
                    __nv_bfloat16* __restrict__ out, int t) {
    using C = Cfg<MT>;
    constexpr int kStages = C::kStages;
    extern __shared__ __align__(16) char smem[];
    const uint32_t sbase = (uint32_t)__cvta_generic_to_shared(smem);

    const int tid = threadIdx.x;
    const int warp = tid >> 5, lane = tid & 31;
    const int gq = lane >> 2, q = lane & 3;
    const int n0 = blockIdx.x * kBN;

    const char* lr = reinterpret_cast<const char*>(lowrank);
    const char* wt = reinterpret_cast<const char*>(weight);

    // ---- prologue: stage 0 + scale slab (group 0), stages 1 .. S-2 ----
    issue_stage<MT>(sbase, lr, wt, t, n0, 0, tid);
    {
        const char* sc = reinterpret_cast<const char*>(scale);
        for (int idx = tid; idx < kScaleBytes / 16; idx += kThreads) {
            int c = idx / 40, k = idx % 40;
            cp_async16(sbase + C::kScaleOff + c * 640 + k * 16,
                       sc + (size_t)(c * kHidden + n0) * (kGroups * 2) + k * 16);
        }
    }
    cp_async_commit();
    #pragma unroll
    for (int s = 1; s < kStages - 1; ++s) {
        issue_stage<MT>(sbase + s * C::kSlot, lr, wt, t, n0, s, tid);
        cp_async_commit();
    }

    float logit[kHC][MT][4];
    #pragma unroll
    for (int c = 0; c < kHC; ++c)
        #pragma unroll
        for (int m = 0; m < MT; ++m)
            #pragma unroll
            for (int e = 0; e < 4; ++e) logit[c][m][e] = 0.f;

    const __half* scale_s = reinterpret_cast<const __half*>(smem + C::kScaleOff);
    const int colw = warp * 8 + 2 * q;   // accumulator column within the CTA's 32 channels

    #pragma unroll 1
    for (int g = 0; g < kGroups; ++g) {
        cp_async_wait<kStages - 2>();
        __syncthreads();
        {
            const int gn = g + kStages - 1;
            if (gn < kGroups) {
                issue_stage<MT>(sbase + (gn % kStages) * C::kSlot, lr, wt, t, n0, gn, tid);
            } else if (gn == kGroups) {
                // x slab: rows < t, 4 branches x 64 B, row pitch 272 B
                const char* xb = reinterpret_cast<const char*>(x);
                for (int idx = tid; idx < t * 16; idx += kThreads) {
                    int i = idx >> 4, c = (idx >> 2) & 3, ch = idx & 3;
                    cp_async16(sbase + C::kXOff + i * kXPitch + c * 64 + ch * 16,
                               xb + ((size_t)i * kRows + c * kHidden + n0) * 2 + ch * 16);
                }
            }
            cp_async_commit();
        }

        const char* slot = smem + (g % kStages) * C::kSlot;
        // lowrank fragments of both k-steps for all m-tiles: rows gq, gq+8 of each m16 tile
        uint4 alo[MT], ahi[MT];
        #pragma unroll
        for (int m = 0; m < MT; ++m) {
            alo[m] = *reinterpret_cast<const uint4*>(slot + (16 * m + gq) * 64 + q * 16);
            ahi[m] = *reinterpret_cast<const uint4*>(slot + (16 * m + gq + 8) * 64 + q * 16);
        }
        const char* wslot = slot + C::kASlot;
        #pragma unroll
        for (int c = 0; c < kHC; ++c) {
            const uint2 wv = *reinterpret_cast<const uint2*>(wslot + (c * kBN + warp * 8 + gq) * 32 + q * 8);
            const uint32_t b00 = i8x2_to_bf16x2(wv.x, 0), b01 = i8x2_to_bf16x2(wv.x, 2);
            const uint32_t b10 = i8x2_to_bf16x2(wv.y, 0), b11 = i8x2_to_bf16x2(wv.y, 2);
            const float s0 = __half2float(scale_s[(c * kBN + colw) * kGroups + g]);
            const float s1 = __half2float(scale_s[(c * kBN + colw + 1) * kGroups + g]);
            #pragma unroll
            for (int m = 0; m < MT; ++m) {
                float d[4] = {0.f, 0.f, 0.f, 0.f};
                const uint32_t a0[4] = {alo[m].x, ahi[m].x, alo[m].y, ahi[m].y};
                mma_bf16(d, a0, b00, b01);
                const uint32_t a1[4] = {alo[m].z, ahi[m].z, alo[m].w, ahi[m].w};
                mma_bf16(d, a1, b10, b11);
                logit[c][m][0] = fmaf(d[0], s0, logit[c][m][0]);
                logit[c][m][1] = fmaf(d[1], s1, logit[c][m][1]);
                logit[c][m][2] = fmaf(d[2], s0, logit[c][m][2]);
                logit[c][m][3] = fmaf(d[3], s1, logit[c][m][3]);
            }
        }
    }

    // ---- epilogue: gates, then x slab, mix, store ----
    const float kexp = -1.4426950408889634f / 1024.f;
    #pragma unroll
    for (int c = 0; c < kHC; ++c)
        #pragma unroll
        for (int m = 0; m < MT; ++m)
            #pragma unroll
            for (int e = 0; e < 4; ++e)
                logit[c][m][e] = 1.f / (1.f + exp2f(logit[c][m][e] * kexp));

    cp_async_wait<0>();
    __syncthreads();

    const char* xs = smem + C::kXOff;
    #pragma unroll
    for (int m = 0; m < MT; ++m) {
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = 16 * m + gq + 8 * h;
            float o0 = 0.f, o1 = 0.f;
            #pragma unroll
            for (int c = 0; c < kHC; ++c) {
                const __nv_bfloat162 xv =
                    *reinterpret_cast<const __nv_bfloat162*>(xs + row * kXPitch + c * 64 + colw * 2);
                const float2 xf = __bfloat1622float2(xv);
                o0 = fmaf(logit[c][m][2 * h], xf.x, o0);
                o1 = fmaf(logit[c][m][2 * h + 1], xf.y, o1);
            }
            if (row < t) {
                *reinterpret_cast<__nv_bfloat162*>(out + (size_t)row * kHidden + n0 + colw) =
                    __floats2bfloat162_rn(o0 * 0.25f, o1 * 0.25f);
            }
        }
    }
}

}  // namespace device::fn_hc_local_mix

// out[t, 2560] = the branch mix of lowrank[t, 320], weight[10240, 320] int8, scale[10240, 10] fp16 and x[t, 10240].
inline void hc_local_mix(tvm::ffi::TensorView lowrank, tvm::ffi::TensorView weight, tvm::ffi::TensorView scale,
                         tvm::ffi::TensorView x, tvm::ffi::TensorView out) {
  using namespace host;
  namespace fn = device::fn_hc_local_mix;
  auto T = SymbolicSize{"tokens"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({T, fn::kLowrank}).with_dtype<__nv_bfloat16>().with_device(device).verify(lowrank);
  TensorMatcher({fn::kRows, fn::kLowrank}).with_dtype<int8_t>().with_device(device).verify(weight);
  TensorMatcher({fn::kRows, fn::kGroups}).with_dtype<__half>().with_device(device).verify(scale);
  TensorMatcher({T, fn::kRows}).with_dtype<__nv_bfloat16>().with_device(device).verify(x);
  TensorMatcher({T, fn::kHidden}).with_dtype<__nv_bfloat16>().with_device(device).verify(out);
  const int t = static_cast<int>(T.unwrap());
  RuntimeCheck(t >= 1 && t <= 64, "hc_local_mix: 1 to 64 tokens");
  for (const auto& v : {lowrank, weight, scale, x}) {
    RuntimeCheck(reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "hc_local_mix: inputs must be 16-byte aligned");
  }
  const auto* lr = static_cast<const __nv_bfloat16*>(lowrank.data_ptr());
  const auto* wt = static_cast<const int8_t*>(weight.data_ptr());
  const auto* sc = static_cast<const __half*>(scale.data_ptr());
  const auto* xs = static_cast<const __nv_bfloat16*>(x.data_ptr());
  auto* o = static_cast<__nv_bfloat16*>(out.data_ptr());
  const dim3 grid(fn::kHidden / fn::kBN), block(fn::kThreads);
  switch ((t + 15) / 16) {
    case 1: LaunchKernel(grid, block, device.unwrap(), fn::Cfg<1>::kSmem)(fn::hc_local_mix_kernel<1>, lr, wt, sc, xs, o, t); break;
    case 2: LaunchKernel(grid, block, device.unwrap(), fn::Cfg<2>::kSmem)(fn::hc_local_mix_kernel<2>, lr, wt, sc, xs, o, t); break;
    case 3: LaunchKernel(grid, block, device.unwrap(), fn::Cfg<3>::kSmem)(fn::hc_local_mix_kernel<3>, lr, wt, sc, xs, o, t); break;
    default: LaunchKernel(grid, block, device.unwrap(), fn::Cfg<4>::kSmem)(fn::hc_local_mix_kernel<4>, lr, wt, sc, xs, o, t); break;
  }
}

}  // namespace sglang
