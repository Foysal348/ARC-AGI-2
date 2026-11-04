/*
 * Flash-Next QSA indexer decode scoring (SGLANG_FLASH_NEXT_QSA_DECODE=1), speculative verify and draft-extend graphs.
 *
 * The same math as SGLang's TileLang paged decode MQA kernel (qsa/mqa.py tilelang_qsa_mqa_decode), score scale
 * 1/sqrt(128):
 *   logits[b, n] = sum over 4 heads of relu(q[b, h] . K[b, n]) / sqrt(128)   for n < context_lens[b], else -inf
 *   K[b, n] = k_cache[page_table[b, n / 16], n % 16, 0]
 * for 4 indexer heads of 128 dims and 16 compressed keys per page only. Rows 4i .. 4i + 3 must be one request's four
 * verify rows (request-major, identical page-table rows): the kernel reads page-table row 4i for all four, and each
 * row's own length masks its scores. q may carry zero padding heads (the TileLang kernel's 8); only heads 0-3 are read.
 *
 * The device code is ForgeKernel's qsa_index_decode (run qsa_index_decode_261031-034012-b28e, iteration 0, commit
 * 5f05466, sha256 957f418c) with two changes: it takes the padded q row stride, and it divides by sqrt(128) passed at
 * run time, as TileLang does (the export multiplied by its rounded reciprocal). One CTA scores one request's 4 rows x 4
 * heads (16 queries, one m16n8k16 tile) against a 64-key tile, so each key is read once; tiles past the request's
 * longest row write -inf without loading keys. Grid (rows / 4, max_model_len / 64) from shapes only, so it replays in
 * CUDA graphs.
 *
 * Built with --use_fast_math like TileLang's kernel (TL_ENABLE_FAST_MATH), so the division, ReLU and adds round and
 * flush as there; the MMA k order and the head sum ((h0 + h1) + (h2 + h3), TileLang's quad butterfly over its 8 padded
 * heads) match too. Bit-identical to TileLang on an L4 in 64 serving-shaped cases, eager and replayed in a CUDA graph
 * (check_flash_next_qsa_index_decode.py). Cold-L2 timing on a vast.ai L4 at 40 rows (ForgeKernel, kernel spans):
 * 49.9 / 82.8 / 122.8 / 239.4 us at 16k / 32k / 48k / 96k mean context against TileLang's 234 / 326 / 435 / 742.
 */
#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>
#include <sgl_kernel/utils.cuh>

#include <tvm/ffi/container/tensor.h>

#include <cuda_bf16.h>
#include <cstdint>

namespace sglang {
namespace device::fn_qsa_index_decode {

constexpr int kThreads   = 128;
constexpr int kTileKeys  = 64;                 // keys per CTA
constexpr int kPageKeys  = 16;
constexpr int kHeads     = 4;
constexpr int kHeadDim   = 128;
constexpr int kKPlane    = kTileKeys * 128;    // one dim-half plane of the K tile: 64 rows x 128 B
constexpr int kQOff      = 2 * kKPlane;        // q tile after the two K planes (16384 B)
constexpr int kQPlane    = 16 * 128;           // one dim-half plane of the q tile: 16 rows x 128 B
constexpr int kSmemBytes = kQOff + 2 * kQPlane; // 20480 B
constexpr float kSqrtDim = 11.313708498984761f;   // fp32 sqrt(128), TileLang's Scale; a kernel argument, never folded

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// Byte offset of 16-B chunk `chunk16` (0..15 over the 256-B row) of row `row` in a two-plane
// tile whose planes are `plane_bytes` apart: 128-B rows, chunk index XOR (row & 7).
__device__ __forceinline__ uint32_t swz(int row, int chunk16, int plane_bytes) {
    return (chunk16 >> 3) * plane_bytes + row * 128 + ((((chunk16 & 7) ^ (row & 7))) << 4);
}

__device__ __forceinline__ void sts128(uint32_t addr, int4 v) {
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};\n"
                 :: "r"(addr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w) : "memory");
}

__device__ __forceinline__ void ldsm_x4(uint32_t addr, uint32_t& r0, uint32_t& r1,
                                        uint32_t& r2, uint32_t& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2,
                                         uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ void __launch_bounds__(kThreads, 4)
qsa_index_decode_kernel(const __nv_bfloat16* __restrict__ q,
                        const __nv_bfloat16* __restrict__ k_cache,
                        const int* __restrict__ page_table,
                        const int* __restrict__ context_lens,
                        float* __restrict__ logits,
                        int maxp, int mml, int q_row_stride, float scale) {
    __shared__ __align__(128) unsigned char smem[kSmemBytes];

    const int tid  = threadIdx.x;
    const int req  = blockIdx.x;
    const int tile = blockIdx.y;
    const int key0 = tile * kTileKeys;
    const int page0 = tile * (kTileKeys / kPageKeys);

    // ---- round 1: the request's 4 context lengths + the tile's 4 page ids, issued together.
    const int4 ctx = __ldg(reinterpret_cast<const int4*>(context_lens + 4 * req));
    const int* pt_row = page_table + static_cast<int64_t>(4 * req) * maxp;
    int4 pid;
    if ((maxp & 3) == 0) {
        pid = __ldg(reinterpret_cast<const int4*>(pt_row + page0));
    } else {
        pid.x = page0 + 0 < maxp ? __ldg(pt_row + page0 + 0) : 0;
        pid.y = page0 + 1 < maxp ? __ldg(pt_row + page0 + 1) : 0;
        pid.z = page0 + 2 < maxp ? __ldg(pt_row + page0 + 2) : 0;
        pid.w = page0 + 3 < maxp ? __ldg(pt_row + page0 + 3) : 0;
    }
    const int maxctx = max(max(ctx.x, ctx.y), max(ctx.z, ctx.w));

    float* out_base = logits + static_cast<int64_t>(4 * req) * mml;

    // ---- dead tile: the 4 x 64 -inf block with 16-B stores, no K / q loads.
    if (key0 >= maxctx) {
        if (tid < 64) {
            const int r = tid >> 4;
            const int col = key0 + (tid & 15) * 4;
            if (col < mml) {
                const float ninf = -INFINITY;
                *reinterpret_cast<float4*>(out_base + static_cast<int64_t>(r) * mml + col) =
                    make_float4(ninf, ninf, ninf, ninf);
            }
        }
        return;
    }

    // ---- round 2: all K loads (8 x 16 B per thread) + q loads (2 x 16 B), back to back.
    const int pids[4] = {pid.x, pid.y, pid.z, pid.w};
    const int chunk16 = tid & 15;
    int4 kv[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int key = j * 8 + (tid >> 4);          // key within the tile
        const int p = j >> 1;                         // page within the tile (CTA-uniform)
        const bool load = (key0 + p * kPageKeys < maxctx) && (page0 + p < maxp);
        if (load) {
            const __nv_bfloat16* src = k_cache
                + (static_cast<int64_t>(pids[p]) * kPageKeys + (key & 15)) * kHeadDim
                + chunk16 * 8;
            kv[j] = __ldg(reinterpret_cast<const int4*>(src));
        } else {
            kv[j] = make_int4(0, 0, 0, 0);
        }
    }
    // q chunk c (0..255) of the request: row c / 64, then heads 0-3 x 128 dims in 16-B chunks.
    int4 qv[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int c = tid + j * kThreads;
        qv[j] = __ldg(reinterpret_cast<const int4*>(
            q + static_cast<int64_t>(4 * req + (c >> 6)) * q_row_stride + (c & 63) * 8));
    }

    // ---- STS with the 128-B XOR swizzle, then the one barrier.
    const uint32_t sbase = smem_u32(smem);
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int key = j * 8 + (tid >> 4);
        sts128(sbase + swz(key, chunk16, kKPlane), kv[j]);
    }
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int c = tid + j * kThreads;
        sts128(sbase + kQOff + swz(c >> 4, c & 15, kQPlane), qv[j]);
    }
    __syncthreads();

    // ---- MMA: warp w scores keys 16w .. 16w+15 against the 16 query vectors.
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int a_row = warp * 16 + (lane & 15);
    const int a_hi  = lane >> 4;
    const int b_row = (lane & 7) + ((lane >> 4) << 3);
    const int b_hi  = (lane >> 3) & 1;

    float acc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        uint32_t a0, a1, a2, a3, b0, b1, b2, b3;
        ldsm_x4(sbase + swz(a_row, s * 2 + a_hi, kKPlane), a0, a1, a2, a3);
        ldsm_x4(sbase + kQOff + swz(b_row, s * 2 + b_hi, kQPlane), b0, b1, b2, b3);
        mma_bf16(acc[0], a0, a1, a2, a3, b0, b1);
        mma_bf16(acc[1], a0, a1, a2, a3, b2, b3);
    }

    // ---- epilogue: ReLU per head, (h0+h1)+(h2+h3) through one quad shuffle (TileLang's reduce_sum order over its
    // 8 padded heads), division by the runtime scale, mask, store.
    const int quad = lane & 3;
    const int kr = key0 + warp * 16 + (lane >> 2);
    const int ctxs[4] = {ctx.x, ctx.y, ctx.z, ctx.w};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        float lo = fmaxf(acc[j][0], 0.f) + fmaxf(acc[j][1], 0.f);   // key kr
        float hi = fmaxf(acc[j][2], 0.f) + fmaxf(acc[j][3], 0.f);   // key kr + 8
        const float lo_p = __shfl_xor_sync(0xffffffffu, lo, 1);
        const float hi_p = __shfl_xor_sync(0xffffffffu, hi, 1);
        if ((quad & 1) == 0) {
            const int r = 2 * j + (quad >> 1);
            const int c = ctxs[r];
            const float vlo = (lo + lo_p) / scale;
            const float vhi = (hi + hi_p) / scale;
            float* orow = out_base + static_cast<int64_t>(r) * mml;
            if (kr < mml)     orow[kr]     = kr < c ? vlo : -INFINITY;
            if (kr + 8 < mml) orow[kr + 8] = kr + 8 < c ? vhi : -INFINITY;
        }
    }
}

}  // namespace device::fn_qsa_index_decode

// logits[rows, 16 * maxp] fp32 = the scores of q[rows, heads >= 4, 128] (heads 0-3 read) over k_cache[pool, 16, 1, 128]
// through page_table[rows, maxp] int32, masked by context_lens[rows] int32; rows % 4 == 0, request-major.
inline void qsa_index_decode(tvm::ffi::TensorView q, tvm::ffi::TensorView k_cache, tvm::ffi::TensorView page_table,
                             tvm::ffi::TensorView context_lens, tvm::ffi::TensorView logits) {
  using namespace host;
  namespace fn = device::fn_qsa_index_decode;
  auto R = SymbolicSize{"rows"};
  auto H = SymbolicSize{"q_heads"};
  auto P = SymbolicSize{"pages"};
  auto M = SymbolicSize{"max_pages"};
  auto L = SymbolicSize{"max_model_len"};
  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({R, H, fn::kHeadDim}).with_dtype<__nv_bfloat16>().with_device(device).verify(q);
  TensorMatcher({P, fn::kPageKeys, 1, fn::kHeadDim}).with_dtype<__nv_bfloat16>().with_device(device).verify(k_cache);
  TensorMatcher({R, M}).with_dtype<int32_t>().with_device(device).verify(page_table);
  TensorMatcher({R}).with_dtype<int32_t>().with_device(device).verify(context_lens);
  TensorMatcher({R, L}).with_dtype<float>().with_device(device).verify(logits);
  const int rows = static_cast<int>(R.unwrap());
  const int heads = static_cast<int>(H.unwrap());
  const int maxp = static_cast<int>(M.unwrap());
  const int mml = static_cast<int>(L.unwrap());
  RuntimeCheck(rows % 4 == 0, "qsa_index_decode: rows must be a multiple of 4 (one request's verify rows)");
  RuntimeCheck(heads >= fn::kHeads, "qsa_index_decode: q needs at least 4 heads");
  RuntimeCheck(mml == fn::kPageKeys * maxp, "qsa_index_decode: max_model_len must be 16 x max_pages");
  for (const auto& v : {q, k_cache, page_table, context_lens, logits}) {
    RuntimeCheck(reinterpret_cast<uintptr_t>(v.data_ptr()) % 16 == 0, "qsa_index_decode: tensors must be 16-byte aligned");
  }
  if (rows == 0 || mml == 0) return;
  const dim3 grid(rows / 4, (mml + fn::kTileKeys - 1) / fn::kTileKeys), block(fn::kThreads);
  LaunchKernel(grid, block, device.unwrap(), 0)(
      fn::qsa_index_decode_kernel, static_cast<const __nv_bfloat16*>(q.data_ptr()),
      static_cast<const __nv_bfloat16*>(k_cache.data_ptr()), static_cast<const int*>(page_table.data_ptr()),
      static_cast<const int*>(context_lens.data_ptr()), static_cast<float*>(logits.data_ptr()), maxp, mml,
      heads * fn::kHeadDim, fn::kSqrtDim);
}

}  // namespace sglang
