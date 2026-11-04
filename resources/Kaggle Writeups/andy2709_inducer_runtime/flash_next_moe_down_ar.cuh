/*
 * Flash-Next fused MoE down projection + host-memory all-reduce (prototype, 2026-10-30).
 *
 * MoeDownAR's body is SGLang 0.5.20's Marlin MoE kernel (sglang/kernels/jit/csrc/gemm/marlin_moe/marlin_template.h,
 * sha256 6b43bc72d64a591bc86f62df2db14bdea5adce7b70b00af1a3e601a3c0d24fdb; Apache-2.0, IST-DASLab and Neural Magic),
 * copied with anchored edits:
 *   1. Scheduling. A tile is one 8-row MoE block times 128 output columns, and one CTA computes all of its K: nothing
 *      is reduced across CTAs. Tiles are numbered column-chunk-major and CTA b takes tiles b, b + G, b + 2G, ..., so
 *      the output's column chunks complete one after another. (Stock Marlin stripes K across CTAs in block-major
 *      order, so its fp32 sums can differ in the last bit and a chunk is only complete at the end.)
 *   2. After writing a tile the CTA adds it to its chunk's counter.
 *   3. In fused mode the last `comm_ctas` CTAs wait for each chunk and turn its rows into this rank's partial exactly
 *      as moe_sum_reduce_kernel_general and fused_gate_sigmoid_mul_add would (fp32 sum over the top-k slots in slot
 *      order, times the routed scale, one bf16 rounding; then gate * shared + routed with one fma and one rounding).
 *      They all-reduce it across the four ranks through pinned host memory like flash_next_host_ar.py's two_shot:
 *      every element is the fp32 sum in rank order 0..3, rounded once to bf16. So the fused output is bit-identical
 *      to running the same tiles unfused, then the stock sum, gate and host all-reduce.
 *
 *      The comm CTAs form groups that take chunks round-robin, so several chunks' host round trips are in flight at
 *      once (v1, one chunk at a time, paid 22-36 us of latency per chunk on Kaggle's 4xL4). They stage the top-k rows
 *      through shared memory with cp.async (v5): while the GEMM saturates DRAM, each dependent round of loads costs
 *      ~7-8 us on L4, so the rounds have to be few and wide.
 *
 * Host buffer: two halves selected by epoch parity; chunk c of a half holds 16 scatter quarters
 * (writer w, quarter q) and 4 result quarters, each `prob_m * chunk_cols / 32` int4. Flags live in their own pages:
 * [phase 2][writer 4][chunk][kCommSlots] epochs written with st.release.sys.
 */
#pragma once

#include <sgl_kernel/tensor.h>

#include <sgl_kernel/scalar_type.hpp>

#include "gemm/marlin_moe/moe_wna16_marlin.cuh"

namespace sglang {

namespace device::fn_moe_down_ar {
using namespace device::marlin;
using namespace device::marlin_moe;

constexpr int kWorld = 4;
constexpr int kCommSlots = 32;      // comm CTAs at most
constexpr int kMaxValidBlocks = 480;  // MoE blocks of local experts at most (8 rows each)
constexpr int kValidBytes = 2048;   // shared memory tail: kMaxValidBlocks block ids + 32 warp counts
constexpr int kMaxTopK = 16;
// Optional globaltimer trace (ns, this GPU's clock; Tail::trace, zero-filled with start = ~0 before the call):
// [0] first CTA start, [1] last GEMM CTA end, [2 + c] last tile of chunk c written,
// [kTraceComm + (c * kCommSlots + slot) * 8 + event] per comm CTA and chunk: 0 tiles counted, 1 partial built and
// fenced, 2 peers' partials arrived, 3 quarter reduced and fenced, 4 peers' sums arrived, 5 chunk written,
// 6 partial built (before the fence), 7 quarter reduced (before the fence). LL mode: 1 = 6 (no fence), 3 = 7 = quarter
// reduced including the wait for peers' partials; 2 and 4 are not written.
constexpr int kMaxChunks = 20;
constexpr int kPublishFence = 0, kPublishRelease = 1, kPublishLL = 2;  // Tail::publish
constexpr int kTraceComm = 2 + kMaxChunks;
constexpr int kTraceLength = kTraceComm + kMaxChunks * kCommSlots * 8;

struct HostComm {
  char* data;          // two halves of `half` bytes
  unsigned* flags;     // [phase 2][writer 4][chunk][kCommSlots]
  unsigned* state;     // this rank's device memory: epoch, comm CTAs finished in the current call
  long long half;
  long long timeout;   // clock64 cycles before a wait traps
  int rank;
  int pad;
};

struct Tail {
  const int4* shared;  // [prob_m, prob_n] bf16: the shared expert's output on this rank
  const float* gate;   // [prob_m]: sigmoid(hidden . gate_weight), as fused_gate_sigmoid_mul_add computes it
  int4* out;           // [prob_m, prob_n] bf16: the all-reduced MoE output
  int* done;           // [chunks]: tiles written per chunk; zero between calls
  float scale;         // routed scaling factor
  int chunk_tiles;     // column tiles per chunk
  int gemm_ctas;       // CTAs that compute tiles
  int comm_ctas;       // CTAs after them that reduce (fused mode)
  int comm_groups;     // comm CTA groups working on different chunks at once (divides comm_ctas)
  int fused;
  int m;               // tokens
  int top_k;           // slots per token (the GEMM itself sees m * top_k rows and top_k = 1)
  unsigned long long* trace;  // kTraceLength globaltimer stamps, or nullptr
  int publish;         // kPublishFence, kPublishRelease or kPublishLL
};

__device__ __forceinline__ unsigned ld_acquire_sys(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_release_sys(unsigned* p, unsigned v) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ int ld_acquire_gpu(const int* p) {
  int v;
  asm volatile("ld.acquire.gpu.global.s32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ unsigned long long now_ns() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
__device__ __forceinline__ unsigned ld_volatile(const unsigned* p) {
  unsigned v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ int4 ld_host(const int4* p) {
  int4 v;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
               : "l"(p)
               : "memory");
  return v;
}

// bf16 <-> fp32 by bit manipulation, rounding to nearest even like torch (NaN -> 0x7fc0).
__device__ __forceinline__ void set8(float* a, int4 v) {
  const unsigned w[4] = {(unsigned)v.x, (unsigned)v.y, (unsigned)v.z, (unsigned)v.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    a[2 * i] = __uint_as_float(w[i] << 16);
    a[2 * i + 1] = __uint_as_float(w[i] & 0xffff0000u);
  }
}
__device__ __forceinline__ void add8(float* a, int4 v) {
  const unsigned w[4] = {(unsigned)v.x, (unsigned)v.y, (unsigned)v.z, (unsigned)v.w};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    a[2 * i] += __uint_as_float(w[i] << 16);
    a[2 * i + 1] += __uint_as_float(w[i] & 0xffff0000u);
  }
}
__device__ __forceinline__ unsigned rne(float f) {
  const unsigned u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0u;
  return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}
__device__ __forceinline__ int4 pack8(const float* a) {
  int4 r;
  r.x = rne(a[0]) | (rne(a[1]) << 16);
  r.y = rne(a[2]) | (rne(a[3]) << 16);
  r.z = rne(a[4]) | (rne(a[5]) << 16);
  r.w = rne(a[6]) | (rne(a[7]) << 16);
  return r;
}

// LL lines (NCCL's LL protocol): a 16-byte store {data, epoch, data, epoch} lands as whole 8-byte halves, so a
// reader that sees the epoch in both flag words of both lines of an element has all of its 16 data bytes.
__device__ __forceinline__ void st_ll(int4* p, int4 v, unsigned e) {
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p), "r"(v.x), "r"(e), "r"(v.y), "r"(e) : "memory");
  asm volatile("st.volatile.global.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(p + 1), "r"(v.z), "r"(e), "r"(v.w), "r"(e)
               : "memory");
}
// Fill v[p] for every p != r from the LL element at line(p), polling (with backoff) until all of them carry epoch e.
template <typename Line>
__device__ __forceinline__ void wait_ll(int4* v, Line line, int r, unsigned e, long long timeout) {
  int4 a[kWorld], b[kWorld];
  long long t0 = 0;
  for (bool first = true;; first = false) {
    bool ready = true;
#pragma unroll
    for (int p = 0; p < kWorld; p++)
      if (p != r) {
        a[p] = ld_host(line(p));
        b[p] = ld_host(line(p) + 1);
      }
#pragma unroll
    for (int p = 0; p < kWorld; p++)
      if (p != r)
        ready &= (unsigned)a[p].y == e && (unsigned)a[p].w == e && (unsigned)b[p].y == e && (unsigned)b[p].w == e;
    if (ready) break;
    if (first) t0 = clock64();
    __nanosleep(128);
    if (clock64() - t0 > timeout) asm volatile("trap;");
  }
#pragma unroll
  for (int p = 0; p < kWorld; p++)
    if (p != r) v[p] = make_int4(a[p].x, a[p].z, b[p].x, b[p].z);
}

// Polls back off so waiting CTAs do not crowd the shared host path with flag reads.
__device__ __forceinline__ void wait_sys(const unsigned* f, unsigned e, long long timeout) {
  if ((int)(ld_acquire_sys(f) - e) < 0) {
    const long long t0 = clock64();
    while ((int)(ld_acquire_sys(f) - e) < 0) {
      __nanosleep(256);
      if (clock64() - t0 > timeout) asm volatile("trap;");
    }
  }
}

// The valid (local-expert) MoE blocks in order: their ids go to sh_valid; returns how many.
template <int threads, bool kIsEP>
__device__ int compact_valid_blocks(const int32_t* expert_ids, int num_blocks, int* sh_valid, int* sh_warp) {
  constexpr int warps = threads / 32;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  int total = 0;
  for (int base = 0; base < num_blocks; base += threads) {
    const int i = base + threadIdx.x;
    const bool valid = i < num_blocks && (!kIsEP || expert_ids[i] != -1);
    const unsigned ballot = __ballot_sync(0xffffffffu, valid);
    if (lane == 0) sh_warp[warp] = __popc(ballot);
    __syncthreads();
    int before = total, step = 0;
#pragma unroll
    for (int w = 0; w < warps; w++) {
      before += w < warp ? sh_warp[w] : 0;
      step += sh_warp[w];
    }
    const int pos = before + __popc(ballot & ((1u << lane) - 1u));
    if (valid && pos < kMaxValidBlocks) sh_valid[pos] = i;
    total += step;
    __syncthreads();
  }
  if (total > kMaxValidBlocks) asm volatile("trap;");
  return total;
}

// Comm CTA `cta` of `ctas`, in group cta % groups (slot cta / groups of its group). Group g owns chunks g, g + groups,
// ...: for each it waits for the chunk's tiles, builds this rank's partial for the slot's share of every quarter and
// two-shots it through host memory. Groups run concurrently, so the chunks' host round trips overlap each other.
template <int threads, int moe_block_size, int tile_n, bool kIsEP>
__device__ void comm_role(
    const int4* __restrict__ C,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_past_padded_ptr,
    int prob_n,
    int max_shared_mem,
    const Tail& tail,
    const HostComm& hc,
    int cta,
    int ctas) {
  __shared__ unsigned s_epoch;
  const int t = threadIdx.x, r = hc.rank, prob_m = tail.m, top_k = tail.top_k;
  const int groups = tail.comm_groups, slots = ctas / groups, group = cta % groups, slot = cta / groups;
  if (t == 0) s_epoch = ld_volatile(hc.state) + 1;
  int num_blocks = num_tokens_past_padded_ptr[0] / moe_block_size;
  int parallel = 0;
  for (int base = 0; base < num_blocks; base += threads) {
    const int i = base + t;
    parallel += __syncthreads_count(i < num_blocks && (!kIsEP || expert_ids[i] != -1));
  }
  __syncthreads();
  const unsigned e = s_epoch;
  const int tiles_per_chunk = parallel * tail.chunk_tiles;
  const int chunk_cols = tail.chunk_tiles * tile_n;
  const int chunks = prob_n / chunk_cols;
  const int row8 = prob_n / 8;  // int4 per output row
  const int cc8 = chunk_cols / 8;  // int4 per row of a chunk
  const long long qn = (long long)prob_m * cc8 / kWorld;  // int4 per quarter of a chunk
  const long long per = (qn + slots - 1) / slots;
  const long long lo = slot * per, hi = lo + per < qn ? lo + per : qn;
  int4* half = reinterpret_cast<int4*>(hc.data + (e & 1) * hc.half);
  auto flag = [&](int phase, int writer, int c) {
    return hc.flags + ((phase * kWorld + writer) * chunks + c) * kCommSlots + slot;
  };
  auto stamp = [&](int c, int event) {
    if (tail.trace && t == 0) tail.trace[kTraceComm + (c * kCommSlots + slot) * 8 + event] = now_ns();
  };
  // Publish this CTA's writes for (phase, chunk); one thread waits for the same slot of every peer.
  // kPublishFence: every thread fences its host writes at system scope before the flag (v1-v3).
  // kPublishRelease: only thread 0's st.release.sys after bar.sync; release is cumulative over the CTA's writes.
  auto exchange = [&](int phase, int c) {
    if (tail.trace) {
      __syncthreads();
      stamp(c, 6 + phase);  // every thread done with the phase, before any fence
    }
    if (tail.publish == kPublishFence) __threadfence_system();
    __syncthreads();
    if (t == 0) {
      stamp(c, 1 + 2 * phase);
      st_release_sys(flag(phase, r, c), e);
#pragma unroll
      for (int p = 0; p < kWorld; p++)
        if (p != r) wait_sys(flag(phase, p, c), e, hc.timeout);
      stamp(c, 2 + 2 * phase);
    }
    __syncthreads();
  };
  auto at = [&](int c, long long u) { return (u / cc8) * row8 + (long long)c * cc8 + u % cc8; };

  // Phase A of chunk c: this rank's partial of every element (q, lo + j) of the CTA's share, handed to sink(q, i, v).
  // The partial is moe_sum_reduce (fp32 over the top-k rows in slot order, * scale, rne), then
  // fma(gate, shared, routed), rne. Rows are staged in the CTA's dynamic shared memory with cp.async (comm CTAs do not use
  // Marlin's layout): a round has every top-k row and the shared-expert row of `wave` elements in flight, because under
  // the GEMM's DRAM load each dependent round of loads costs several microseconds.
  extern __shared__ int4 stage[];
  const int rows = top_k + 1;
  const int wave = (max_shared_mem / 16) / rows;
  const long long span = hi > lo ? hi - lo : 0;
  auto build = [&](int c, auto&& sink) {
    const long long n = kWorld * span;
    for (long long w0 = 0; w0 < n; w0 += wave) {
      const int count = (int)(n - w0 < wave ? n - w0 : wave);
      for (int li = t; li < count * rows; li += threads) {
        const long long el = w0 + li / rows;
        const int k = li % rows;
        const long long u = (el / span) * qn + lo + el % span;
        const long long tok = u / cc8, col8 = (long long)c * cc8 + u % cc8;
        cp_async4(&stage[li], k < top_k ? C + (tok * top_k + k) * row8 + col8 : tail.shared + tok * row8 + col8);
      }
      cp_async_fence();
      cp_async_wait<0>();
      __syncthreads();
      for (int j = t; j < count; j += threads) {
        const long long el = w0 + j;
        const int q = (int)(el / span);
        const long long i = lo + el % span;
        const long long tok = (q * qn + i) / cc8;
        float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
        for (int k = 0; k < top_k; k++) add8(acc, stage[j * rows + k]);
        float routed[8], shared[8], res[8];
#pragma unroll
        for (int x = 0; x < 8; x++) acc[x] *= tail.scale;
        set8(routed, pack8(acc));
        set8(shared, stage[j * rows + top_k]);
        const float g = tail.gate[tok];
#pragma unroll
        for (int x = 0; x < 8; x++) res[x] = __fmaf_rn(g, shared[x], routed[x]);
        sink(q, i, pack8(res));
      }
      __syncthreads();
    }
  };
  auto sum4 = [&](const int4* v) {
    float a[8];
    set8(a, v[0]);
    add8(a, v[1]);
    add8(a, v[2]);
    add8(a, v[3]);
    return pack8(a);
  };

  for (int c = group; c < chunks; c += groups) {
    if (t == 0 && ld_acquire_gpu(tail.done + c) < tiles_per_chunk) {
      const long long t0 = clock64();
      while (ld_acquire_gpu(tail.done + c) < tiles_per_chunk) {
        __nanosleep(64);
        if (clock64() - t0 > hc.timeout) asm volatile("trap;");
      }
    }
    __syncthreads();
    stamp(c, 0);
    if (tail.publish == kPublishLL) {
      // No flags and no fences: every 8 bytes of a line carry 4 data bytes and the epoch, and readers poll their own
      // lines. Chunk c: 16 scatter quarters, then 4 result quarters, each qn elements of 2 lines.
      int4* scatter = half + (long long)c * 40 * qn;
      int4* result = scatter + 32 * qn;
      build(c, [&](int q, long long i, int4 v) {
        if (q == r)
          tail.out[at(c, q * qn + i)] = v;
        else
          st_ll(scatter + ((r * kWorld + q) * qn + i) * 2, v, e);
      });
      if (tail.trace) {
        __syncthreads();
        stamp(c, 6);
        stamp(c, 1);
      }
      for (long long i = lo + t; i < hi; i += threads) {
        int4 v[kWorld];
        const int4 own = tail.out[at(c, r * qn + i)];
#pragma unroll
        for (int p = 0; p < kWorld; p++)
          if (p == r) v[p] = own;  // compile-time index: v stays in registers
        wait_ll(v, [&](int p) { return scatter + ((p * kWorld + r) * qn + i) * 2; }, r, e, hc.timeout);
        const int4 y = sum4(v);
        tail.out[at(c, r * qn + i)] = y;
        st_ll(result + (r * qn + i) * 2, y, e);
      }
      if (tail.trace) {
        __syncthreads();
        stamp(c, 7);
        stamp(c, 3);
      }
      for (long long i = lo + t; i < hi; i += threads) {
        int4 v[kWorld];
        wait_ll(v, [&](int q) { return result + (q * qn + i) * 2; }, r, e, hc.timeout);
#pragma unroll
        for (int q = 0; q < kWorld; q++)
          if (q != r) tail.out[at(c, q * qn + i)] = v[q];
      }
    } else {
      int4* scatter = half + (long long)c * 20 * qn;  // [writer][quarter][qn]
      int4* result = scatter + 16 * qn;               // [quarter][qn]
      build(c, [&](int q, long long i, int4 v) {
        if (q == r)
          tail.out[at(c, q * qn + i)] = v;
        else
          scatter[(r * kWorld + q) * qn + i] = v;
      });
      exchange(0, c);
      // Reduce quarter r in rank order; keep it and publish it. Two elements a thread per round trip.
      for (long long i = lo + t; i < hi; i += 2 * threads) {
        const bool second = i + threads < hi;
        int4 v[2][kWorld];
#pragma unroll
        for (int h = 0; h < 2; h++) {
          if (h == 1 && !second) break;
#pragma unroll
          for (int p = 0; p < kWorld; p++)
            v[h][p] = p == r ? tail.out[at(c, r * qn + i + h * threads)]
                             : ld_host(scatter + (p * kWorld + r) * qn + i + h * threads);
        }
#pragma unroll
        for (int h = 0; h < 2; h++) {
          if (h == 1 && !second) break;
          const int4 y = sum4(v[h]);
          tail.out[at(c, r * qn + i + h * threads)] = y;
          result[r * qn + i + h * threads] = y;
        }
      }
      exchange(1, c);
      // The other quarters' sums: every load of an element pair issued before its stores.
      for (long long i = lo + t; i < hi; i += 2 * threads) {
        const bool second = i + threads < hi;
        int4 v[2][kWorld];
#pragma unroll
        for (int h = 0; h < 2; h++) {
          if (h == 1 && !second) break;
#pragma unroll
          for (int q = 0; q < kWorld; q++)
            if (q != r) v[h][q] = ld_host(result + q * qn + i + h * threads);
        }
#pragma unroll
        for (int h = 0; h < 2; h++) {
          if (h == 1 && !second) break;
#pragma unroll
          for (int q = 0; q < kWorld; q++)
            if (q != r) tail.out[at(c, q * qn + i + h * threads)] = v[h][q];
        }
      }
    }
    if (tail.trace) {
      __syncthreads();
      stamp(c, 5);
    }
  }
  // The last comm CTA resets the chunk counters (every tile of the call is counted) and advances the epoch.
  __syncthreads();
  if (t == 0) {
    __threadfence();
    if (atomicAdd(hc.state + 1, 1u) == (unsigned)ctas - 1) {
      for (int c = 0; c < chunks; c++) tail.done[c] = 0;
      hc.state[1] = 0;
      __threadfence();
      hc.state[0] = e;
    }
  }
}

template <
    typename scalar_t,                   // compute dtype, half or nv_float16
    const host::ScalarTypeId w_type_id,  // weight ScalarType id
    const host::ScalarTypeId s_type_id,  // weight scale ScalarType id
    const int threads,                   // number of threads in a threadblock
    const int thread_m_blocks,           // number of 16x16 blocks in the m
                                         // dimension (batchsize) of the
                                         // threadblock
    const int thread_n_blocks,           // same for n dimension (output)
    const int thread_k_blocks,           // same for k dimension (reduction)
    const bool m_block_size_8,           // whether m_block_size == 8
                                         // only works when thread_m_blocks == 1
    const int stages,                    // number of stages for the async global->shared
                                         // fetch pipeline
    const int group_blocks,              // number of consecutive 16x16 blocks
                                         // with a separate quantization scale
    const bool is_zp_float,              // is zero point of float16 type?
    const bool kIsEP,                    // expert parallelism
    const bool kHasBias                  // has per-expert bias
    >
__global__ void __launch_bounds__(threads, 3) MoeDownAR(
    const int4* __restrict__ A,  // fp16 input matrix of shape mxk
    const int4* __restrict__ B,  // 4bit quantized weight matrix of shape kxn
    int4* __restrict__ C,        // fp16 output buffer of shape mxn
    int4* __restrict__ C_tmp,    // fp32 tmp output buffer (for reduce)
    const int4* __restrict__ b_bias_ptr,
    const int4* __restrict__ scales_ptr,                     // fp16 quantization scales of shape
                                                             // (k/groupsize)xn
    const uint16_t* __restrict__ scale2_ptr,                 // fp16 global scale (for nvfp4
                                                             // only)
    const int4* __restrict__ zp_ptr,                         // 4bit packed zero-points of shape
                                                             // (k/groupsize)x(n/pack_factor)
    const int* __restrict__ g_idx,                           // int32 group indices of shape k
    const int32_t* __restrict__ sorted_token_ids_ptr,        // moe sorted_ids
    const int32_t* __restrict__ expert_ids_ptr,              // moe expert ids
    const int32_t* __restrict__ num_tokens_past_padded_ptr,  // moe num tokens
    const float* __restrict__ topk_weights_ptr,              // moe top weights
    int top_k,                                               // num of experts per token
    bool mul_topk_weights,                                   // mul topk weights or not
    bool is_ep,                                              // expert parallelism
    int num_groups,                                          // number of scale groups per output channel
    int prob_m,                                              // batch dimension m
    int prob_n,                                              // output dimension n
    int prob_k,                                              // reduction dimension k
    int* locks,                                              // extra global storage for barrier synchronization
    bool has_bias,
    bool use_atomic_add,   // whether to use atomic add to reduce
    bool use_fp32_reduce,  // whether to use fp32 global reduce
    int max_shared_mem,
    HostComm hc,
    Tail tail) {
  if (tail.trace && threadIdx.x == 0) atomicMin(tail.trace, now_ns());
  // Fused mode: the last CTAs reduce and all-reduce the output chunk by chunk (comm_role above).
  if (tail.fused && blockIdx.x >= tail.gemm_ctas) {
    comm_role<threads, (m_block_size_8 ? 8 : 16 * thread_m_blocks), 16 * thread_n_blocks, kIsEP>(
        C, expert_ids_ptr, num_tokens_past_padded_ptr, prob_n, max_shared_mem, tail, hc,
        blockIdx.x - tail.gemm_ctas, gridDim.x - tail.gemm_ctas);
    return;
  }
  // Each threadblock processes one "stripe" of the B matrix with (roughly) the
  // same size, which might involve multiple column "slices" (of width 16 *
  // `thread_n_blocks`). Stripes are defined as shown in the 3x3 matrix 5 SM
  // example:
  //   0 1 3
  //   0 2 3
  //   1 2 4
  // While this kind of partitioning makes things somewhat more complicated, it
  // ensures good utilization of all SMs for many kinds of shape and GPU
  // configurations, while requiring as few slow global cross-threadblock
  // reductions as possible.
  using Dtype = ScalarType<scalar_t>;
  using scalar_t2 = typename ScalarType<scalar_t>::scalar_t2;
  using FragA = typename ScalarType<scalar_t>::FragA;
  using FragB = typename ScalarType<scalar_t>::FragB;
  using FragC = typename ScalarType<scalar_t>::FragC;
  using FragS = typename ScalarType<scalar_t>::FragS;
  using FragZP = typename ScalarType<scalar_t>::FragZP;

  extern __shared__ int4 sh[];
  static constexpr auto w_type = host::ScalarType::from_id(w_type_id);
  static constexpr auto s_type = host::ScalarType::from_id(s_type_id);
  if constexpr (w_type == host::kFE2M1f) {
    static_assert(s_type == host::kFE4M3fn && group_blocks == 1 || s_type == host::kFE8M0fnu && group_blocks == 2);
  } else if constexpr (std::is_same<scalar_t, nv_bfloat16>::value) {
    static_assert(s_type == host::kBFloat16);
  } else if constexpr (std::is_same<scalar_t, half>::value) {
    static_assert(s_type == host::kFloat16);
  }

  constexpr bool has_zp = w_type == host::kU4 || w_type == host::kU8;
  constexpr bool is_int_type =
      w_type == host::kU4 || w_type == host::kU8 || w_type == host::kU4B8 || w_type == host::kU8B128;
  constexpr bool is_8bit_scale = s_type.size_bits() == 8;
  // see comments of dequant.h for more details
  constexpr bool dequant_skip_flop = w_type == host::kFE4M3fn || w_type == host::kFE2M1f && s_type == host::kFE4M3fn ||
                                     has_zp && !is_zp_float && !std::is_same<scalar_t, nv_bfloat16>::value ||
                                     has_zp && !is_zp_float && !(w_type == host::kU8);

  scalar_t2 global_scale;

  constexpr bool has_act_order = group_blocks == 0;

  constexpr int pack_factor = 32 / w_type.size_bits();
  static_assert(thread_m_blocks == 1 || !m_block_size_8);
  constexpr int moe_block_size = m_block_size_8 ? 8 : (16 * thread_m_blocks);
  const int group_size = (!has_act_order && group_blocks == -1) ? prob_k : prob_k / num_groups;
  const int scales_expert_stride = prob_n * prob_k / group_size / (is_8bit_scale ? 16 : 8);
  const int zp_expert_stride =
      is_zp_float ? prob_n * prob_k / group_size / 8 : prob_n * prob_k / group_size / (pack_factor * 4);
  const int b_bias_expert_stride = prob_n / 8;

  int num_tokens_past_padded = num_tokens_past_padded_ptr[0];
  // 16-byte aligned: max_shared_mem is not a multiple of 16 when three CTAs share an H200 SM.
  int* sh_valid = reinterpret_cast<int*>(reinterpret_cast<char*>(sh) + ((max_shared_mem - kValidBytes) & ~15));
  int parallel = compact_valid_blocks<threads, kIsEP>(
      expert_ids_ptr, num_tokens_past_padded / moe_block_size, sh_valid, sh_valid + kMaxValidBlocks);

  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  int iters = div_ceil(k_tiles * n_tiles * parallel, gridDim.x);

  if constexpr (!has_act_order && group_blocks != -1) {
    if (group_blocks >= thread_k_blocks) {
      // Ensure that the number of tiles in each stripe is a multiple of the
      // groupsize; this avoids an annoying special case where a stripe starts
      // in the middle of group.
      iters = (group_blocks / thread_k_blocks) * div_ceil(iters, (group_blocks / thread_k_blocks));
    }
  }

  const int chunk_tiles = tail.chunk_tiles;  // column tiles per output chunk
  const int tiles_per_chunk = chunk_tiles * parallel;
  const int total_tiles = n_tiles * parallel;
  int tile = blockIdx.x;
  int cur_chunk = 0;
  int slice_row = 0;
  int slice_col_par = 0;
  int slice_col = 0;
  int slice_iters;      // number of threadblock tiles in the current slice
  int slice_count = 0;  // total number of active threadblocks in the current slice
  int slice_idx;        // index of threadblock in current slice; numbered bottom to
                        // top

  int par_id = -1;  // no MoE block loaded yet
  int block_id = -1;
  int64_t expert_id = 0;  // use int64 to avoid computation result overflow
  int old_expert_id = 0;
  int64_t B_expert_off = 0;

  int4* sh_block_sorted_ids_int4 = sh;
  int4* sh_rd_block_sorted_ids_int4 = sh_block_sorted_ids_int4 + moe_block_size / 4;
  int4* sh_block_topk_weights_int4 = sh_rd_block_sorted_ids_int4 + moe_block_size / 4;
  // sh_block_topk_weights_int4 only need (moe_block_size / 4);
  // but we pad to align to 256 bytes
  int4* sh_new = sh_block_topk_weights_int4 + moe_block_size / 2 + moe_block_size;
  int32_t* sh_block_sorted_ids = reinterpret_cast<int*>(sh_block_sorted_ids_int4);
  int32_t* sh_rd_block_sorted_ids = reinterpret_cast<int*>(sh_rd_block_sorted_ids_int4);
  scalar_t2* sh_block_topk_weights = reinterpret_cast<scalar_t2*>(sh_block_topk_weights_int4);

  int32_t block_num_valid_tokens = 0;
  int32_t locks_off = 0;


  int prob_m_top_k = prob_m * top_k;
  // read moe block data given block_id
  // block_sorted_ids / block_num_valid_tokens / block_topk_weights
  auto read_moe_block_data = [&](int block_id) {
    block_num_valid_tokens = moe_block_size;

    cp_async4_pred(
        sh_block_sorted_ids_int4 + threadIdx.x,
        reinterpret_cast<const int4*>(sorted_token_ids_ptr) + (block_id * moe_block_size / 4 + threadIdx.x),
        threadIdx.x < moe_block_size / 4);

    cp_async_fence();
    cp_async_wait<0>();

    __syncthreads();

    if (threadIdx.x >= threads - 32) {
      constexpr int size_per_thread = div_ceil(moe_block_size, 32);
      int lane_id = threadIdx.x - (threads - 32);

      int local_count = 0;
#pragma unroll
      for (int i = 0; i < size_per_thread; i++) {
        int j = lane_id * size_per_thread + i;
        if (j < moe_block_size) {
          int idx = sh_block_sorted_ids[j];
          if (idx < prob_m_top_k) local_count++;
        }
      }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 750
      if constexpr (moe_block_size >= 16) local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 16);
      if constexpr (moe_block_size >= 8) local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 8);
      if constexpr (moe_block_size >= 4) local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 4);
      if constexpr (moe_block_size >= 2) local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 2);

      local_count += __shfl_down_sync(0xFFFFFFFF, local_count, 1);
      block_num_valid_tokens = local_count;
#else
      block_num_valid_tokens = __reduce_add_sync(0xffffffff, local_count);
#endif

      if (lane_id == 0) reinterpret_cast<int*>(sh_new)[0] = block_num_valid_tokens;
    }

    if (threadIdx.x < moe_block_size) {
      int idx = sh_block_sorted_ids[threadIdx.x];
      sh_rd_block_sorted_ids[threadIdx.x] = idx / top_k;

      if (mul_topk_weights) {
        idx = idx < prob_m_top_k ? idx : 0;
        scalar_t topk_weight_tmp = Dtype::float2num(topk_weights_ptr[idx]);
        if constexpr (w_type == host::kFE2M1f && s_type == host::kFE4M3fn) {
          sh_block_topk_weights[threadIdx.x] = __hmul2(global_scale, Dtype::num2num2(topk_weight_tmp));
        } else {
          sh_block_topk_weights[threadIdx.x] = Dtype::num2num2(topk_weight_tmp);
        }
      }
    }

    __syncthreads();

    block_num_valid_tokens = reinterpret_cast<int*>(sh_new)[0];
    __syncthreads();
  };

  // when move to next moe block, find the next block_id and expert_id
  // and then read moe block data
  auto update_next_moe_block_data = [&]() {
    if (par_id >= parallel) return;

    old_expert_id = expert_id;
    block_id = sh_valid[par_id];
    expert_id = expert_ids_ptr[block_id];

    if constexpr (w_type == host::kFE2M1f && s_type == host::kFE4M3fn) {
      uint16_t val = scale2_ptr[expert_id];
      global_scale = Dtype::num2num2(*reinterpret_cast<scalar_t*>(&val));
    }

    B_expert_off = expert_id * prob_n * prob_k / (pack_factor * 4);
    scales_ptr += (expert_id - old_expert_id) * scales_expert_stride;
    if constexpr (has_zp) {
      zp_ptr += (expert_id - old_expert_id) * zp_expert_stride;
    }
    if constexpr (has_act_order) {
      g_idx += (expert_id - old_expert_id) * prob_k;
    }
    if constexpr (kHasBias) {
      b_bias_ptr += (expert_id - old_expert_id) * b_bias_expert_stride;
    }

    read_moe_block_data(block_id);
  };

  // Tile f is MoE block p, output chunk c, column tile j of that chunk (f = (c * parallel + p) * chunk_tiles + j).
  // One CTA computes all of its K: slice_count is always 1, so nothing is reduced across CTAs.
  auto init_slice = [&](bool first_init = false) {
    slice_iters = 0;
    if (tile >= total_tiles) return;
    cur_chunk = tile / tiles_per_chunk;
    int rem = tile - cur_chunk * tiles_per_chunk;
    int p = rem / chunk_tiles;
    slice_col = cur_chunk * chunk_tiles + rem % chunk_tiles;
    slice_col_par = p * n_tiles + slice_col;
    slice_row = 0;
    slice_iters = k_tiles;
    slice_count = 1;
    slice_idx = 0;
    if (p != par_id) {
      par_id = p;
      update_next_moe_block_data();
    }
  };

  init_slice(true);

  // A sizes/strides

  // stride of the A matrix in global memory
  int a_gl_stride = prob_k / 8;
  // stride of an A matrix tile in shared memory
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
  // delta between subsequent A tiles in global memory
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
  // between subsequent accesses within a tile
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  // between shared memory writes
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  // between shared memory tile reads
  constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));
  // within a shared memory tile
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  // overall size of a tile
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
  // number of shared write iterations for a tile
  constexpr int a_sh_wr_iters = div_ceil(a_sh_stage, a_sh_wr_delta);

  // B sizes/strides
  int b_gl_stride = 16 * prob_n / (pack_factor * 4);
  constexpr int b_sh_stride = ((thread_n_blocks * 16) * 16 / pack_factor) / 4;
  constexpr int b_thread_vecs = w_type.size_bits() == 4 ? 1 : 2;
  constexpr int b_sh_stride_threads = b_sh_stride / b_thread_vecs;

  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride_threads);
  constexpr int b_sh_wr_delta = threads * b_thread_vecs;
  constexpr int b_sh_rd_delta = threads * b_thread_vecs;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  // Scale sizes/strides without act_order
  int s_gl_stride = prob_n / (is_8bit_scale ? 16 : 8);
  constexpr int s_sh_stride = 16 * thread_n_blocks / (is_8bit_scale ? 16 : 8);
  constexpr int s_tb_groups =
      !has_act_order && group_blocks != -1 && group_blocks < thread_k_blocks ? thread_k_blocks / group_blocks : 1;
  constexpr int s_sh_stage = s_tb_groups * s_sh_stride;
  int s_gl_rd_delta = s_gl_stride;

  // Scale size/strides with act_order
  constexpr int tb_k = 16 * thread_k_blocks;
  constexpr int g_idx_stage = has_act_order ? (tb_k * sizeof(int)) / 16 : 0;
  // constexpr int act_s_row_stride      = 1;
  // int           act_s_col_stride      = act_s_row_stride * num_groups;
  constexpr int act_s_max_num_groups = 32;
  int act_s_col_stride = 1;
  int act_s_col_warp_stride = act_s_col_stride * 8;
  int tb_n_warps = thread_n_blocks / 4;
  int act_s_col_tb_stride = act_s_col_warp_stride * tb_n_warps;

  // Zero-points sizes/strides
  int zp_gl_stride = is_zp_float ? prob_n / 8 : (prob_n / pack_factor) / 4;
  constexpr int zp_sh_stride = is_zp_float ? 16 * thread_n_blocks / 8 : ((16 * thread_n_blocks) / pack_factor) / 4;
  constexpr int zp_tb_groups = s_tb_groups;
  constexpr int zp_sh_stage = has_zp ? zp_tb_groups * zp_sh_stride : 0;
  int zp_gl_rd_delta = zp_gl_stride;

  // Global A read index of current thread.
  int a_gl_rd_row = threadIdx.x / a_gl_rd_delta_o;
  int a_gl_rd_col = a_gl_rd_delta_o * slice_row + threadIdx.x % a_gl_rd_delta_o;

  // Shared write index of current thread.
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  // Shared read index.
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % (16 / (m_block_size_8 ? 2 : 1))) +
                (threadIdx.x % 32) / (16 / (m_block_size_8 ? 2 : 1));
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride_threads) + (threadIdx.x % b_sh_stride_threads) * b_thread_vecs;
  const int b_gl_rd_base = b_gl_rd;
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  auto b_sh_wr = threadIdx.x * b_thread_vecs;
  auto b_sh_rd = threadIdx.x * b_thread_vecs;

  // For act_order
  constexpr int k_iter_size = tb_k / b_sh_wr_iters;
  int slice_k_start = tb_k * slice_row;
  int slice_k_finish = slice_k_start + tb_k * slice_iters;
  int slice_k_start_shared_fetch = slice_k_start;
  int slice_n_offset = act_s_col_tb_stride * slice_col;

  // No act_order
  int s_gl_rd;
  if constexpr (!has_act_order) {
    if constexpr (group_blocks == -1) {
      s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
    } else if constexpr (group_blocks >= thread_k_blocks) {
      s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + s_sh_stride * slice_col + threadIdx.x;
    } else {
      s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks + threadIdx.x / s_sh_stride) +
                s_sh_stride * slice_col + threadIdx.x % s_sh_stride;
    }
  }
  auto s_sh_wr = threadIdx.x;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stage;

  // Zero-points
  int zp_gl_rd;
  if constexpr (has_zp) {
    if constexpr (group_blocks == -1) {
      zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
    } else {
      zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + zp_sh_stride * slice_col + threadIdx.x;
    }
  }
  auto zp_sh_wr = threadIdx.x;
  bool zp_sh_wr_pred = threadIdx.x < zp_sh_stride;

  // We use a different scale layout for grouped and column-wise quantization as
  // we scale a `half2` tile in column-major layout in the former and in
  // row-major in the latter case.
  int s_sh_rd;
  if constexpr (group_blocks != -1)
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;
  else if constexpr (group_blocks == -1 && (m_block_size_8 || (has_zp && !dequant_skip_flop)))
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 8;
  else
    s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) % 4;

  int bias_sh_rd;
  if constexpr (m_block_size_8) {
    bias_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 8;
  } else {
    bias_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) % 4;
  }

  int bias_sh_wr = threadIdx.x;
  int bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;

  // Zero-points have the same read layout as the scales
  // (without column-wise case)
  constexpr int num_col_threads = 8;
  constexpr int num_row_threads = 4;
  constexpr int num_ints_per_thread = 8 / pack_factor;
  int zp_sh_rd;
  if constexpr (has_zp) {
    if constexpr (is_zp_float) {
      if constexpr (group_blocks != -1) {
        zp_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;
      }
    } else {
      zp_sh_rd = num_ints_per_thread * num_col_threads * ((threadIdx.x / 32) % (thread_n_blocks / 4)) +
                 num_ints_per_thread * ((threadIdx.x % 32) / num_row_threads);
    }
  }

  // To ensure that writing and reading A tiles to/from shared memory, the
  // latter in fragment format, is fully bank conflict free, we need to use a
  // rather fancy XOR-based layout. The key here is that neither reads nor
  // writes of the 16-byte `int4` blocks of 8 consecutive threads involve the
  // same shared memory banks. Further, it seems (based on NSight-Compute) that
  // each warp must also write a consecutive memory segment?
  auto transform_a = [&](int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ (row % 8);
  };
  // Since the computation of this remapping is non-trivial and, due to our main
  // loop unrolls, all shared memory accesses are static, we simply precompute
  // both transformed reads and writes.
  int a_sh_wr_trans[a_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
#pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  // Since B-accesses have non-constant stride they have to be computed at
  // runtime; we break dependencies between subsequent accesses with a tile by
  // maintining multiple pointers (we have enough registers), a tiny
  // optimization.
  const int4* B_ptr[b_sh_wr_iters];
#pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++)
    B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

  // Shared memory storage for global fetch pipelines.
  constexpr int sh_red_size = (2 * thread_n_blocks + 1) * 16 * thread_m_blocks;
  constexpr int sh_b_size = stages * b_sh_stage;
  int4* sh_b = sh_new;
  int4* sh_red = sh_new;

  constexpr int sh_size_b_red_min = (sh_red_size < sh_b_size ? sh_red_size : sh_b_size);
  constexpr int sh_size_b_red_max = (sh_red_size > sh_b_size ? sh_red_size : sh_b_size);
  constexpr int sh_bias_size = (thread_n_blocks * 16 / 8);
  constexpr int sh_b_red_bias_size =
      sh_size_b_red_max > (sh_size_b_red_min + sh_bias_size) ? sh_size_b_red_max : (sh_size_b_red_min + sh_bias_size);

  int4* sh_bias = sh_new + sh_size_b_red_min;
  int4* sh_g_idx = sh_new + sh_b_red_bias_size;
  int4* sh_zp = sh_g_idx + (stages * g_idx_stage);
  constexpr int sh_s_size = has_act_order ? (act_s_max_num_groups * s_sh_stride) : (stages * s_sh_stage);
  int4* sh_s = sh_zp + (stages * zp_sh_stage);
  // shared memory reused by reduction should be smaller than
  // shared memory used by weight.
  static_assert(thread_m_blocks * 16 * thread_n_blocks * 16 / 8 <= stages * b_sh_stage);
  int4* sh_a = sh_s + sh_s_size;
  constexpr int shm_size_used = moe_block_size + stages * (g_idx_stage + zp_sh_stage) + sh_s_size + sh_b_red_bias_size;

  // all remaining shared memory is used to cache A (input)
  // sh_a_max_row is at least ` stages * 16 * thread_m_blocks `
  int sh_a_max_row = ((max_shared_mem - kValidBytes - 1024) / 16 - shm_size_used) / (thread_k_blocks * 2);

  // Register storage for double buffer of shared memory reads.
  FragA frag_a[2][thread_m_blocks];
  I4 frag_b_quant[2][b_thread_vecs];
  FragC frag_c[thread_m_blocks][4][2];
  FragS frag_s[2][4];  // No act-order
  FragS frag_bias[2][4];
  FragS act_frag_s[2][4][4];             // For act-order
  int frag_qzp[2][num_ints_per_thread];  // Zero-points
  FragZP frag_zp;                        // Zero-points in fp16
  FragZP frag_zpf[2];                    // Zero-points in fp16 in HQQ

  // Zero accumulators.
  auto zero_accums = [&]() {
#pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(frag_c)[i] = 0;
  };

  int sh_first_group_id = -1;
  int sh_num_groups = -1;

  auto fetch_act_order_scales_to_shared = [&](bool is_async, int first_group_id, int last_group_id) {
    sh_first_group_id = first_group_id;
    sh_num_groups = last_group_id - first_group_id + 1;

    if (sh_num_groups > act_s_max_num_groups) {
      sh_num_groups = act_s_max_num_groups;
    }

    if (sh_first_group_id + sh_num_groups > num_groups) {
      sh_num_groups = num_groups - sh_first_group_id;
    }

    int row_offset = first_group_id * s_gl_stride;

    if (is_async) {
      for (int i = 0; i < sh_num_groups; i++) {
        if (threadIdx.x < s_sh_stride) {
          cp_async4_pred(
              &sh_s[(i * s_sh_stride) + threadIdx.x],
              &scales_ptr[row_offset + (i * s_gl_stride) + slice_n_offset + threadIdx.x]);
        }
      }
    } else {
      for (int i = 0; i < sh_num_groups; i++) {
        if (threadIdx.x < s_sh_stride) {
          sh_s[(i * s_sh_stride) + threadIdx.x] =
              scales_ptr[row_offset + (i * s_gl_stride) + slice_n_offset + threadIdx.x];
        }
      }
    }
  };

  // Asynchronously fetch the next A, B and s tile from global to the next
  // shared memory pipeline location.
  bool should_load_a = true;
  int max_num_stage_groups = ((sh_a_max_row - moe_block_size) / moe_block_size + 1) / stages;
  max_num_stage_groups = max(max_num_stage_groups, 1);
  auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true, int pipe_a = 0) {
    if (pred) {
      if (should_load_a) {
        int4* sh_a_stage = sh_a + moe_block_size * a_sh_stride * pipe_a;
#pragma unroll
        for (int i = 0; i < a_sh_wr_iters; i++) {
          int row = a_gl_rd_delta_i / a_gl_stride * i + a_gl_rd_row;
          int64_t sorted_row = 0;
          if (!m_block_size_8 || row < 8) sorted_row = sh_rd_block_sorted_ids[row];
          int64_t true_idx = sorted_row * a_gl_stride + a_gl_rd_col + a_gl_rd_delta_o * a_off;
          cp_async4_pred(&sh_a_stage[a_sh_wr_trans[i]], &A[true_idx], row < block_num_valid_tokens);
        }
      }

      int4* sh_b_stage = sh_b + b_sh_stage * pipe;
#pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
#pragma unroll
        for (int j = 0; j < b_thread_vecs; j++) {
          cp_async4(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr + j], B_ptr[i] + j + B_expert_off);
        }

        B_ptr[i] += b_gl_rd_delta_o;
      }

      if constexpr (has_act_order) {
        // Fetch g_idx thread-block portion
        int full_pipe = a_off;
        int cur_k = slice_k_start_shared_fetch + tb_k * full_pipe;
        if (cur_k < prob_k && cur_k < slice_k_finish) {
          int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;

          int4 const* cur_g_idx_stage_ptr = reinterpret_cast<int4 const*>(&g_idx[cur_k]);

          if (threadIdx.x < g_idx_stage) {
            cp_async4_pred(&sh_g_idx_stage[threadIdx.x], &cur_g_idx_stage_ptr[threadIdx.x]);
          }
        }
      } else {
        if constexpr (group_blocks != -1) {
          int4* sh_s_stage = sh_s + s_sh_stage * pipe;
          if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0) {
            if (s_sh_wr_pred) {
              cp_async4(&sh_s_stage[s_sh_wr], &scales_ptr[s_gl_rd]);
            }
            s_gl_rd += s_gl_rd_delta * s_tb_groups;
          }
        }

        if constexpr (has_zp && group_blocks != -1) {
          int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;
          if (pipe % div_ceil(group_blocks, thread_k_blocks) == 0) {
            if (zp_sh_wr_pred) {
              cp_async4(&sh_zp_stage[zp_sh_wr], &zp_ptr[zp_gl_rd]);
            }
            zp_gl_rd += zp_gl_rd_delta * zp_tb_groups;
          }
        }
      }
    }
    // Insert a fence even when we are winding down the pipeline to ensure that
    // waiting is also correct at this point.
    cp_async_fence();
  };

  auto fetch_col_zp_to_shared = [&]() {
    if (zp_sh_wr_pred) {
      cp_async4(&sh_zp[zp_sh_wr], &zp_ptr[zp_gl_rd]);
    }
  };

  auto fetch_col_scale_to_shared = [&]() {
    if (s_sh_wr_pred) {
      cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
    }
  };

  // Wait until the next thread tile has been loaded to shared memory.
  auto wait_for_stage = [&]() {
    // We only have `stages - 2` active fetches since we are double buffering
    // and can only issue the next fetch when it is guaranteed that the previous
    // shared memory load is fully complete (as it may otherwise be
    // overwritten).
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  // Load the next sub-tile from the current location in the shared memory pipe
  // into the current register buffer.
  auto fetch_to_registers = [&](int k, int pipe, int pipe_a = 0) {
    int4* sh_a_stage = sh_a + moe_block_size * a_sh_stride * pipe_a;
#pragma unroll
    for (int i = 0; i < thread_m_blocks; i++)
      ldsm<m_block_size_8 ? 2 : 4, scalar_t>(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    int4* sh_b_stage = sh_b + b_sh_stage * pipe;

#pragma unroll
    for (int i = 0; i < b_thread_vecs; i++) {
      frag_b_quant[k % 2][i] = *reinterpret_cast<I4*>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd + i]);
    }
  };

  bool is_same_group[stages];
  int same_group_id[stages];

  auto init_same_group = [&](int pipe) {
    if constexpr (!has_act_order) {
      return;
    }

    int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;
    int* sh_g_idx_int_ptr = reinterpret_cast<int*>(sh_g_idx_stage);

    int group_id_1 = sh_g_idx_int_ptr[0];
    int group_id_2 = sh_g_idx_int_ptr[tb_k - 1];

    is_same_group[pipe] = group_id_1 == group_id_2;
    same_group_id[pipe] = group_id_1;
  };

  auto fetch_scales_to_registers = [&](int k, int full_pipe) {
    int pipe = full_pipe % stages;

    if constexpr (!has_act_order) {
      // No act-order case
      if constexpr (group_blocks == -1) {
        // load only when starting a new slice
        if (k == 0 && full_pipe == 0) {
          reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd];
          reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
        }
      } else if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          constexpr int g = group_blocks / thread_k_blocks;
          if (pipe % g == 0) {
            if (k % b_sh_wr_iters == 0) {
              int4* sh_s_stage = sh_s + s_sh_stage * (g * (pipe / g));
              reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd];
            } else {
              reinterpret_cast<int4*>(&frag_s[1])[0] = reinterpret_cast<int4*>(&frag_s[0])[0];
            }
          }
        } else {
          auto warp_id = threadIdx.x / 32;
          int n_warps = thread_n_blocks / 4;

          int warp_row = warp_id / n_warps;
          int cur_k = warp_row * 16;
          cur_k += k_iter_size * (k % b_sh_wr_iters);
          int k_blocks = cur_k / 16;
          int cur_group_id = k_blocks / group_blocks;

          int4* sh_s_stage = sh_s + s_sh_stage * pipe;

          if constexpr (!is_8bit_scale) {
            reinterpret_cast<int4*>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd + cur_group_id * s_sh_stride];
          } else {
            reinterpret_cast<int2*>(&frag_s[k % 2])[0] =
                reinterpret_cast<int2*>(sh_s_stage)[s_sh_rd + cur_group_id * (2 * s_sh_stride)];
          }
        }
      }

      return;
    }

    // Act-order case

    // Determine K of the "current" thread-block
    int cur_k = slice_k_start + tb_k * full_pipe;
    if (cur_k >= prob_k || cur_k >= slice_k_finish) {
      return;
    }

    // Reset (to current thread-block) since we read g_idx portion from the
    // shared memory
    cur_k = 0;

    // Progress to current iteration
    cur_k += k_iter_size * (k % b_sh_wr_iters);

    // Determine "position" inside the thread-block (based on warp and
    // thread-id)
    auto warp_id = threadIdx.x / 32;
    int n_warps = thread_n_blocks / 4;  // Each warp processes 4 16-size tiles over N

    int warp_row = warp_id / n_warps;
    int warp_col = warp_id % n_warps;

    cur_k += warp_row * 16;

    auto th_id = threadIdx.x % 32;
    cur_k += (th_id % 4) * 2;  // Due to tensor-core layout for fp16 B matrix

    int s_col_shift =
        /*slice_n_offset +*/ (act_s_col_warp_stride * warp_col) + (th_id / 4) * act_s_col_stride;

    if (is_same_group[pipe]) {
      if (k % 2 == 0) {
        *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0]))) =
            sh_s[(same_group_id[pipe] - sh_first_group_id) * s_sh_stride + s_col_shift];
      } else {
        *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0]))) =
            *(reinterpret_cast<int4*>(&(act_frag_s[(k - 1) % 2][0][0])));
      }

      for (int i = 1; i < 4; i++) {
        *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][i][0]))) = *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][0][0])));
      }
      return;
    }

    int4* sh_g_idx_stage = sh_g_idx + g_idx_stage * pipe;
    int* sh_g_idx_int_ptr = reinterpret_cast<int*>(sh_g_idx_stage);

    constexpr int k_frag_offsets[4] = {0, 1, 8, 9};  // Tensor core offsets per thread

#pragma unroll
    for (int i = 0; i < 4; i++) {
      int actual_k = cur_k + k_frag_offsets[i];

      int group_id = sh_g_idx_int_ptr[actual_k];
      int rel_group_id = group_id - sh_first_group_id;

      *(reinterpret_cast<int4*>(&(act_frag_s[k % 2][i][0]))) = sh_s[rel_group_id * s_sh_stride + s_col_shift];
    }
  };

  auto fetch_zp_to_registers = [&](int k, int full_pipe) {
    // This code does not handle group_blocks == 0,
    // which signifies act_order.
    // has_zp implies AWQ, which doesn't have act_order,
    static_assert(!has_zp || group_blocks != 0);

    if constexpr (has_zp && !is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks == -1) {
        // load only when starting a new slice
        if (k == 0 && full_pipe == 0) {
#pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp))[zp_sh_rd + i];
          }
        }

      } else if constexpr (group_blocks >= thread_k_blocks) {
        if (k % b_sh_wr_iters == 0) {
          int4* sh_zp_stage =
              sh_zp + zp_sh_stage * ((group_blocks / thread_k_blocks) * (pipe / (group_blocks / thread_k_blocks)));
#pragma unroll
          for (int i = 0; i < num_ints_per_thread; i++) {
            frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
          }
        }
      } else {
        auto warp_id = threadIdx.x / 32;
        int n_warps = thread_n_blocks / 4;

        int warp_row = warp_id / n_warps;

        int cur_k = warp_row * 16;
        cur_k += k_iter_size * (k % b_sh_wr_iters);

        int k_blocks = cur_k / 16;
        int cur_group_id = 0;

        // Suppress bogus and persistent divide-by-zero warning
#pragma nv_diagnostic push
#pragma nv_diag_suppress divide_by_zero
        cur_group_id = k_blocks / group_blocks;
#pragma nv_diagnostic pop

        int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

        sh_zp_stage += cur_group_id * zp_sh_stride;

#pragma unroll
        for (int i = 0; i < num_ints_per_thread; i++) {
          frag_qzp[k % 2][i] = (reinterpret_cast<int*>(sh_zp_stage))[zp_sh_rd + i];
        }
      }
    }

    else if constexpr (has_zp && is_zp_float) {
      int pipe = full_pipe % stages;

      if constexpr (group_blocks != -1) {
        if constexpr (group_blocks >= thread_k_blocks) {
          if (k % b_sh_wr_iters == 0) {
            int4* sh_zp_stage =
                sh_zp + zp_sh_stage * ((group_blocks / thread_k_blocks) * (pipe / (group_blocks / thread_k_blocks)));
            reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] = sh_zp_stage[zp_sh_rd];
          }
        } else {
          auto warp_id = threadIdx.x / 32;
          int n_warps = thread_n_blocks / 4;

          int warp_row = warp_id / n_warps;

          int cur_k = warp_row * 16;
          cur_k += k_iter_size * (k % b_sh_wr_iters);

          int k_blocks = cur_k / 16;
          // Suppress bogus and persistent divide-by-zero warning
#pragma nv_diagnostic push
#pragma nv_diag_suppress divide_by_zero
          int cur_group_id = k_blocks / group_blocks;
#pragma nv_diagnostic pop

          int4* sh_zp_stage = sh_zp + zp_sh_stage * pipe;

          reinterpret_cast<int4*>(&frag_zpf[k % 2])[0] = sh_zp_stage[zp_sh_rd + cur_group_id * zp_sh_stride];
        }
      }
    }
  };

  auto dequant_data = [&](int q, scalar_t2* frag_b_ptr) {
    dequant<scalar_t2, w_type_id, dequant_skip_flop>(q, frag_b_ptr);
  };

  // Execute the actual tensor core matmul of a sub-tile.
  bool is_first_matmul_in_slice = true;
  auto matmul = [&](int k) {
    int k2 = k % 2;
    const bool is_new_zp = ((group_blocks != -1) && (group_blocks < thread_k_blocks || k == 0)) ||
                           (group_blocks == -1 && is_first_matmul_in_slice);
    if constexpr (has_zp && !is_zp_float) {
      if (is_new_zp) {
        if constexpr (group_blocks == -1) is_first_matmul_in_slice = false;
        int zp_quant_0, zp_quant_1;

        if constexpr (w_type.size_bits() == 4) {
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = zp_quant_0 >> 8;
        } else {
          static_assert(w_type.size_bits() == 8);
          zp_quant_0 = frag_qzp[k2][0];
          zp_quant_1 = frag_qzp[k2][1];
        }

        dequant_data(zp_quant_0, reinterpret_cast<scalar_t2*>(&frag_zp));
        dequant_data(zp_quant_1, reinterpret_cast<scalar_t2*>(&frag_zp) + 2);
      }
    }
    if constexpr (!dequant_skip_flop && has_zp && is_zp_float) {
      if (is_new_zp) {
        reinterpret_cast<int4*>(&frag_zp)[0] = reinterpret_cast<int4*>(&frag_zpf[k2])[0];
      }
    }

    // FP4/FP8 scale dequantization (E4M3 for NVFP4 and E8M0 for MXFP4).
    if constexpr (
        (s_type == host::kFE4M3fn || s_type == host::kFE8M0fnu) &&
        !(std::is_same<scalar_t2, half2>::value && s_type == host::kFE8M0fnu)) {
      int s_quant_0 = reinterpret_cast<int*>(frag_s[k2])[0];
      int s_quant_1 = reinterpret_cast<int*>(frag_s[k2])[1];

      dequant_fp8_scales<scalar_t2, s_type_id>(s_quant_0, reinterpret_cast<scalar_t2*>(&frag_s[k2]));
      dequant_fp8_scales<scalar_t2, s_type_id>(s_quant_1, reinterpret_cast<scalar_t2*>(&frag_s[k2]) + 2);
    }

// We have the m dimension as the inner loop in order to encourage overlapping
// dequantization and matmul operations.
#pragma unroll
    for (int j = 0; j < 4; j++) {
      FragB frag_b0;
      FragB frag_b1;
      int b_quant_0, b_quant_1;

      if constexpr (w_type_id == host::kFE2M1f.id()) {
        b_quant_1 = frag_b_quant[k2][0][j];
        b_quant_0 = b_quant_1 << 8;
      } else if constexpr (w_type.size_bits() == 4) {
        b_quant_0 = frag_b_quant[k2][0][j];
        b_quant_1 = b_quant_0 >> 8;
      } else {
        static_assert(w_type.size_bits() == 8);
        int* frag_b_quant_ptr = reinterpret_cast<int*>(frag_b_quant[k2]);
        b_quant_0 = frag_b_quant_ptr[j * 2 + 0];
        b_quant_1 = frag_b_quant_ptr[j * 2 + 1];
      }

      dequant_data(b_quant_0, reinterpret_cast<scalar_t2*>(&frag_b0));
      dequant_data(b_quant_1, reinterpret_cast<scalar_t2*>(&frag_b1));

      if constexpr (dequant_skip_flop && has_zp && !is_zp_float) {
        sub_zp<scalar_t>(frag_b0, frag_zp[j], 0);
        sub_zp<scalar_t>(frag_b1, frag_zp[j], 1);
      }

      // Apply scale to frag_b0
      if constexpr (has_act_order) {
        static_assert(group_blocks != -1);
        scale4<scalar_t>(
            frag_b0, act_frag_s[k2][0][j], act_frag_s[k2][1][j], act_frag_s[k2][2][j], act_frag_s[k2][3][j], 0);
        scale4<scalar_t>(
            frag_b1, act_frag_s[k2][0][j], act_frag_s[k2][1][j], act_frag_s[k2][2][j], act_frag_s[k2][3][j], 1);
      } else if constexpr (!dequant_skip_flop && has_zp && !is_zp_float && group_blocks == -1) {
        int idx = (threadIdx.x / 4) % 2;
        scalar_t2 s2 = Dtype::nums2num2(
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 0])[idx],
            reinterpret_cast<scalar_t*>(&frag_s[j / 2][j % 2 * 2 + 1])[idx]);
        if (is_new_zp) frag_zp[j] = __hmul2(frag_zp[j], s2);
        scale_and_sub<scalar_t>(frag_b0, s2.x, frag_zp[j].x);
        scale_and_sub<scalar_t>(frag_b1, s2.y, frag_zp[j].y);
      } else if constexpr (!dequant_skip_flop && has_zp && group_blocks != -1) {
        if (is_new_zp) frag_zp[j] = __hmul2(frag_zp[j], *reinterpret_cast<scalar_t2*>(&frag_s[k2][j]));
        scale_and_sub<scalar_t>(frag_b0, frag_s[k2][j][0].x, frag_zp[j].x);
        scale_and_sub<scalar_t>(frag_b1, frag_s[k2][j][0].y, frag_zp[j].y);
      } else if constexpr (group_blocks != -1) {
        scale<scalar_t>(frag_b0, frag_s[k2][j], 0);
        scale<scalar_t>(frag_b1, frag_s[k2][j], 1);
      }

#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        if constexpr (m_block_size_8) {
          mma_trans<scalar_t>(frag_a[k2][i], frag_b0, frag_b1, frag_c[i][j][0]);
        } else {
          mma<scalar_t>(frag_a[k2][i], frag_b0, frag_c[i][j][0]);
          mma<scalar_t>(frag_a[k2][i], frag_b1, frag_c[i][j][1]);
        }
      }
    }
  };

  // Since we slice across the k dimension of a tile in order to increase the
  // number of warps while keeping the n dimension of a tile reasonable, we have
  // multiple warps that accumulate their partial sums of the same output
  // location; which we have to reduce over in the end. We do in shared memory.
  auto thread_block_reduce = [&]() {
    constexpr int red_off = threads / b_sh_stride_threads / 2;
    if (red_off >= 1) {
      auto red_idx = threadIdx.x / b_sh_stride_threads;
      constexpr int red_sh_stride = b_sh_stride_threads * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride_threads;
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride_threads) + (threadIdx.x % b_sh_stride_threads);

      // Parallel logarithmic shared memory reduction. We make sure to avoid any
      // unnecessary read or write iterations, e.g., for two warps we write only
      // once by warp 1 and read only once by warp 0.

#pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
#pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
#pragma unroll
            for (int j = 0; j < 4 * 2; j += (m_block_size_8 ? 2 : 1)) {
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(&sh_red[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&sh_red[red_sh_wr]);
#pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];
              }
              sh_red[red_sh_wr] = reinterpret_cast<int4*>(&frag_c)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
#pragma unroll
          for (int i = 0; i < 4 * 2; i += (m_block_size_8 ? 2 : 1)) {
            float* c_rd = reinterpret_cast<float*>(&sh_red[red_sh_delta * i + red_sh_rd]);
#pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  // Since multiple threadblocks may process parts of the same column slice, we
  // finally have to globally reduce over the results. As the striped
  // partitioning minimizes the number of such reductions and our outputs are
  // usually rather small, we perform this reduction serially in L2 cache.
  auto global_reduce_fp16 = [&](bool first = false, bool last = false) {
    // We are very careful here to reduce directly in the output buffer to
    // maximize L2 cache utilization in this step. To do this, we write out
    // results in FP16 (but still reduce with FP32 compute).
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    bool is_th_active = threadIdx.x < active_threads;
    if (!is_th_active) {
      return;
    }

    int c_gl_stride = prob_n / 8;
    int c_gl_wr_delta_o = 8 * c_gl_stride;
    int c_gl_wr_delta_i = 4 * (active_threads / 32);
    int c_gl_wr;
    if constexpr (m_block_size_8) {
      c_gl_wr = c_gl_stride * ((threadIdx.x % 4) * 2) + 4 * (threadIdx.x / 32) + (threadIdx.x % 32) / 8;
      c_gl_wr += (2 * thread_n_blocks) * slice_col;
    } else {
      c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) + 4 * (threadIdx.x / 32) + threadIdx.x % 4;
      c_gl_wr += (2 * thread_n_blocks) * slice_col;
    }
    constexpr int c_sh_wr_delta = active_threads;
    int c_sh_wr = threadIdx.x;

    if (!first) {

#pragma unroll
      for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
        int c_idx;
        if constexpr (m_block_size_8)
          c_idx = c_gl_wr + i * c_gl_stride + (threadIdx.x % 8) / 4 * c_gl_wr_delta_i;
        else
          c_idx = c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2);
        if (c_idx / c_gl_stride < block_num_valid_tokens) {
          int64_t sorted_row = sh_block_sorted_ids[c_idx / c_gl_stride];
          int64_t true_idx = sorted_row * c_gl_stride + c_idx % c_gl_stride;
          sh_red[c_sh_wr + c_sh_wr_delta * i] = C[true_idx];
        }
      }
    }

#pragma unroll
    for (int i = 0; i < (m_block_size_8 ? 2 : thread_m_blocks * 4); i++) {
      if (!first) {
        int4 c_red = sh_red[c_sh_wr + i * c_sh_wr_delta];
#pragma unroll
        for (int j = 0; j < 2 * 4; j++) {
          int delta = 0;
          if constexpr (m_block_size_8) {
            delta = j % 2 == 1 ? -2 : 0;
          }
          reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4) + delta] +=
              Dtype::num2float(reinterpret_cast<scalar_t*>(&c_red)[j]);
        }
      }
      if (!last) {
        int4 c;
#pragma unroll
        for (int j = 0; j < 2 * 4; j++) {
          int delta = 0;
          if constexpr (m_block_size_8) {
            delta = j % 2 == 1 ? -2 : 0;
          }
          reinterpret_cast<scalar_t*>(&c)[j] =
              Dtype::float2num(reinterpret_cast<float*>(&frag_c)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4) + delta]);
        }

        int c_idx;
        if constexpr (m_block_size_8)
          c_idx = c_gl_wr + i * c_gl_stride + (threadIdx.x % 8) / 4 * c_gl_wr_delta_i;
        else
          c_idx = c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2);
        if (c_idx / c_gl_stride < block_num_valid_tokens) {
          int64_t sorted_row = sh_block_sorted_ids[c_idx / c_gl_stride];
          int64_t true_idx = sorted_row * c_gl_stride + c_idx % c_gl_stride;
          C[true_idx] = c;
        }
      }
    }
  };

  // Globally reduce over threadblocks that compute the same column block.
  // We use a tmp C buffer to reduce in full fp32 precision.
  auto global_reduce_fp32 = [&](bool first = false, bool last = false) {
    constexpr int tb_m = thread_m_blocks * 16;
    constexpr int tb_n = thread_n_blocks * 16;

    constexpr int c_size = tb_m * tb_n * sizeof(float) / 16;

    constexpr int active_threads = 32 * thread_n_blocks / 4;
    bool is_th_active = threadIdx.x < active_threads;

    constexpr int num_floats = thread_m_blocks * 4 * 2 * 4;
    constexpr int th_size = num_floats * sizeof(float) / 16;

    int c_cur_offset = locks_off * c_size;

    if (!is_th_active) {
      return;
    }

    if (!first) {
      float* frag_c_ptr = reinterpret_cast<float*>(&frag_c);
#pragma unroll
      for (int k = 0; k < th_size; k++) {
        if constexpr (m_block_size_8) {
          if (k % 2) continue;
        } else {
          if (k / 8 * 16 + (threadIdx.x % 32) / 4 >= block_num_valid_tokens) continue;
        }

        sh_red[threadIdx.x] = C_tmp[c_cur_offset + active_threads * k + threadIdx.x];

        float* sh_c_ptr = reinterpret_cast<float*>(&sh_red[threadIdx.x]);
#pragma unroll
        for (int f = 0; f < 4; f++) {
          frag_c_ptr[k * 4 + f] += sh_c_ptr[f];
        }
      }
    }

    if (!last) {
      int4* frag_c_ptr = reinterpret_cast<int4*>(&frag_c);
#pragma unroll
      for (int k = 0; k < th_size; k++) {
        if constexpr (m_block_size_8) {
          if (k % 2) continue;
        } else {
          if (k / 8 * 16 + (threadIdx.x % 32) / 4 >= block_num_valid_tokens) continue;
        }

        C_tmp[c_cur_offset + active_threads * k + threadIdx.x] = frag_c_ptr[k];
      }
    }
  };

  // Write out the reduce final result in the correct layout. We only actually
  // reshuffle matrix fragments in this step, the reduction above is performed
  // in fragment layout.
  auto write_result = [&](bool last) {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr;
    if constexpr (m_block_size_8) {
      c_sh_wr = (8 * c_sh_stride) * ((threadIdx.x % 32) % 4 * 2) + (threadIdx.x % 32) / 4;
      c_sh_wr += 64 * (threadIdx.x / 32);
    } else {
      c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
      c_sh_wr += 32 * (threadIdx.x / 32);
    }

    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));

    // We first reorder in shared memory to guarantee the most efficient final
    // global write patterns
    auto write = [&](int idx, float c0, float c1, FragS& s, FragS& b_bias) {
      scalar_t2 res = Dtype::nums2num2(Dtype::float2num(c0), Dtype::float2num(c1));

      // For per-column quantization we finally apply the scale here (only for
      // 4-bit)
      if constexpr (
          !has_act_order && group_blocks == -1 && w_type.size_bits() == 4 && (has_zp && dequant_skip_flop || !has_zp)) {
        scalar_t2 tmp_scale = s[0];
        if constexpr (m_block_size_8) {
          tmp_scale = Dtype::num2num2(reinterpret_cast<scalar_t*>(&s[0])[(threadIdx.x % 8) / 4]);
        }
        res = __hmul2(res, tmp_scale);
      }

      if constexpr (w_type == host::kFE2M1f && s_type == host::kFE4M3fn) {
        if (!mul_topk_weights) {
          res = __hmul2(res, global_scale);
        }
      }
      if constexpr (kHasBias) {
        if (last) {
          scalar_t2 tmp_bias = b_bias[0];
          if constexpr (m_block_size_8) {
            tmp_bias = Dtype::num2num2(reinterpret_cast<scalar_t*>(&b_bias[0])[(threadIdx.x % 8) / 4]);
          }
          res = __hadd2(res, tmp_bias);
        }
      }

      if constexpr (m_block_size_8) {
        ((scalar_t*)sh_red)[idx] = res.x;
        ((scalar_t*)sh_red)[idx + 8 * c_sh_stride] = res.y;
      } else {
        ((scalar_t2*)sh_red)[idx] = res;
      }
    };

    if (threadIdx.x / 32 < thread_n_blocks / 4) {
#pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          if constexpr (m_block_size_8) {
            int wr = c_sh_wr + 16 * j;
            write(
                wr,
                frag_c[i][j][0][0],
                frag_c[i][j][0][1],
                frag_s[j / 2][2 * (j % 2) + 0],
                frag_bias[j / 2][2 * (j % 2) + 0]);
            write(
                wr + 8,
                frag_c[i][j][0][2],
                frag_c[i][j][0][3],
                frag_s[j / 2][2 * (j % 2) + 1],
                frag_bias[j / 2][2 * (j % 2) + 1]);
          } else {
            int wr = c_sh_wr + 8 * j;
            write(
                wr + (4 * c_sh_stride) * 0 + 0,
                frag_c[i][j][0][0],
                frag_c[i][j][0][1],
                frag_s[j / 2][2 * (j % 2) + 0],
                frag_bias[j / 2][2 * (j % 2) + 0]);
            write(
                wr + (4 * c_sh_stride) * 8 + 0,
                frag_c[i][j][0][2],
                frag_c[i][j][0][3],
                frag_s[j / 2][2 * (j % 2) + 0],
                frag_bias[j / 2][2 * (j % 2) + 0]);
            write(
                wr + (4 * c_sh_stride) * 0 + 4,
                frag_c[i][j][1][0],
                frag_c[i][j][1][1],
                frag_s[j / 2][2 * (j % 2) + 1],
                frag_bias[j / 2][2 * (j % 2) + 1]);
            write(
                wr + (4 * c_sh_stride) * 8 + 4,
                frag_c[i][j][1][2],
                frag_c[i][j][1][3],
                frag_s[j / 2][2 * (j % 2) + 1],
                frag_bias[j / 2][2 * (j % 2) + 1]);
          }
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < div_ceil(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {
      int row = c_gl_wr / c_gl_stride;
      if (row < block_num_valid_tokens) {
        int64_t sorted_row = sh_block_sorted_ids[row];
        int64_t true_idx = sorted_row * c_gl_stride + c_gl_wr % c_gl_stride;
        scalar_t2 topk_weight_score;
        if (mul_topk_weights) topk_weight_score = sh_block_topk_weights[row];
        if (use_atomic_add && slice_count > 1 || mul_topk_weights) {
          scalar_t2* C_half2 = reinterpret_cast<scalar_t2*>(&C[true_idx]);
          scalar_t2* sh_red_half2 = reinterpret_cast<scalar_t2*>(&sh_red[c_sh_rd]);
#pragma unroll
          for (int a = 0; a < 4; a++) {
            scalar_t2 res = sh_red_half2[a];
            if (mul_topk_weights) {
              res = __hmul2(res, topk_weight_score);
            }

            if (use_atomic_add && slice_count > 1) {
              atomicAdd(&C_half2[a], res);
            } else {
              C_half2[a] = res;
            };
          }
        } else {
          C[true_idx] = sh_red[c_sh_rd];
        }
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
    __syncthreads();
  };

  // Start global fetch and register load pipelines.
  auto start_pipes = [&]() {

#pragma unroll
    for (int i = 0; i < stages - 1; i++) {
      if (has_act_order && i == 0) {
        int last_g_idx = slice_k_start + stages * tb_k * 2;
        if (last_g_idx >= prob_k) {
          last_g_idx = prob_k - 1;
        }
        fetch_act_order_scales_to_shared(true, g_idx[slice_k_start], g_idx[last_g_idx]);
      }

      if constexpr (has_zp && !is_zp_float && group_blocks == -1) {
        if (i == 0) {
          fetch_col_zp_to_shared();
          if constexpr (!dequant_skip_flop) {
            fetch_col_scale_to_shared();
          }
        }
      }
      fetch_to_shared(i, i, i < slice_iters, i);
    }

    zero_accums();
    wait_for_stage();
    init_same_group(0);
    fetch_to_registers(0, 0);
    fetch_scales_to_registers(0, 0);
    fetch_zp_to_registers(0, 0);
    a_gl_rd_col += a_gl_rd_delta_o * (stages - 1);
    if constexpr (has_act_order) {
      slice_k_start_shared_fetch += tb_k * (stages - 1);
    }
  };
  if (slice_iters) {
    start_pipes();
  }

  // Main loop.
  while (slice_iters) {
    // We unroll over both the global fetch and the register load pipeline to
    // ensure all shared memory accesses are static. Note that both pipelines
    // have even length meaning that the next iteration will always start at
    // index 0.

    for (int stage_group_id = 0; stage_group_id < max_num_stage_groups; stage_group_id++) {
#pragma unroll
      for (int pipe = 0; pipe < stages;) {
#pragma unroll
        for (int k = 0; k < b_sh_wr_iters; k++) {
          int idx = (pipe >= stages && stage_group_id == max_num_stage_groups - 1) ? (pipe - stages)
                                                                                   : (pipe + stage_group_id * stages);
          fetch_to_registers(k + 1, pipe % stages, idx);
          fetch_scales_to_registers(k + 1, pipe);
          fetch_zp_to_registers(k + 1, pipe);
          if (k == b_sh_wr_iters - 2) {
            int idx = (pipe >= 1 && stage_group_id == max_num_stage_groups - 1)
                          ? (pipe - 1)
                          : (pipe + (stage_group_id + 1) * stages - 1);
            fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages, idx);
            pipe++;
            wait_for_stage();
            init_same_group(pipe % stages);
          }
          matmul(k);
        }
        slice_iters--;
        if (slice_iters == 0) {
          break;
        }
      }

      a_gl_rd_col += a_gl_rd_delta_o * stages;

      if constexpr (has_act_order) {
        slice_k_start += tb_k * stages;

        if (slice_k_start < prob_k) {
          slice_k_start_shared_fetch += tb_k * stages;
          int first_group_id = g_idx[slice_k_start];
          int last_g_idx = slice_k_start + stages * tb_k * 2;
          if (last_g_idx >= prob_k) {
            last_g_idx = prob_k - 1;
          }
          int last_group_id = g_idx[last_g_idx];
          if (last_group_id >= sh_first_group_id + sh_num_groups) {
            fetch_act_order_scales_to_shared(false, first_group_id, last_group_id);
            __syncthreads();
          }
        }
      }
      if (slice_iters == 0) {
        break;
      }
    }

    // Process results and, if necessary, proceed to the next column slice.
    // While this pattern may not be the most readable, other ways of writing
    // the loop seemed to noticeably worse performance after compilation.
    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      // For per-column scales, we only fetch them here in the final step before
      // write-out
      if constexpr (!has_act_order && group_blocks == -1 && (has_zp && dequant_skip_flop || !has_zp)) {
        if (w_type.size_bits() == 8 || (last || use_atomic_add)) {
          if (s_sh_wr_pred) {
            cp_async4(&sh_s[s_sh_wr], &scales_ptr[s_gl_rd]);
          }
          cp_async_fence();
        }
      }

      thread_block_reduce();

      if constexpr (kHasBias) {
        if (last) {
          __syncthreads();
          cp_async4_pred(&sh_bias[bias_sh_wr], &b_bias_ptr[bias_gl_rd], threadIdx.x < 16 * thread_n_blocks / 8);
          cp_async_fence();
        }
      }

      if constexpr (!has_act_order && group_blocks == -1 && (has_zp && dequant_skip_flop || !has_zp)) {
        if (w_type.size_bits() == 8 || (last || use_atomic_add)) {
          cp_async_wait<0>();
          __syncthreads();
          if (threadIdx.x / 32 < thread_n_blocks / 4) {
            reinterpret_cast<int4*>(&frag_s)[0] = sh_s[s_sh_rd + 0];
            reinterpret_cast<int4*>(&frag_s)[1] = sh_s[s_sh_rd + 4];
            if constexpr (m_block_size_8) {
              int idx = (threadIdx.x / 4) % 2;
              scalar_t2* frag_s_half2 = reinterpret_cast<scalar_t2*>(frag_s);
#pragma unroll
              for (int i = 0; i < 8; i++) {
                frag_s_half2[i] = Dtype::num2num2(reinterpret_cast<scalar_t*>(&frag_s_half2[i])[idx]);
              }
            }
          }
        }
      }

      // For 8-bit channelwise, we apply the scale before the global reduction
      // that converts the fp32 results to fp16 (so that we avoid possible
      // overflow in fp16)
      if constexpr (
          !has_act_order && group_blocks == -1 && w_type.size_bits() == 8 && (has_zp && dequant_skip_flop || !has_zp)) {
        if (threadIdx.x / 32 < thread_n_blocks / 4) {
#pragma unroll
          for (int i = 0; i < thread_m_blocks; i++) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
              scale_float<scalar_t>(reinterpret_cast<float*>(&frag_c[i][j][0][0]), frag_s[j / 2][2 * (j % 2) + 0]);
              scale_float<scalar_t>(
                  reinterpret_cast<float*>(&frag_c[i][j][0][2]), frag_s[j / 2][2 * (j % 2) + (m_block_size_8 ? 1 : 0)]);

              if constexpr (!m_block_size_8) {
                scale_float<scalar_t>(reinterpret_cast<float*>(&frag_c[i][j][1][0]), frag_s[j / 2][2 * (j % 2) + 1]);
                scale_float<scalar_t>(reinterpret_cast<float*>(&frag_c[i][j][1][2]), frag_s[j / 2][2 * (j % 2) + 1]);
              }
            }
          }
        }
      }

      if (slice_count > 1 && !use_atomic_add) {
        // only globally reduce if there is more than one block in a slice
        barrier_acquire(&locks[locks_off], slice_idx);
        if (use_fp32_reduce) {
          global_reduce_fp32(slice_idx == 0, last);
        } else {
          global_reduce_fp16(slice_idx == 0, last);
        }
        barrier_release(&locks[locks_off], last);
      }

      if constexpr (kHasBias) {
        if (last) {
          cp_async_wait<0>();
          __syncthreads();
          reinterpret_cast<int4*>(&frag_bias)[0] = sh_bias[bias_sh_rd];
          reinterpret_cast<int4*>(&frag_bias)[1] = sh_bias[bias_sh_rd + 4];
          __syncthreads();
        }
      }

      if (use_atomic_add && slice_count > 1 && slice_idx != 0) wait_negative_and_add(&locks[locks_off]);
      if (last || use_atomic_add)
        // only the last block in a slice actually writes the result
        write_result(last);
      // write_result ends with __syncthreads(): every row of the tile is stored. Publish it to the comm CTAs.
      if (tail.fused && threadIdx.x == 0) {
        __threadfence();
        atomicAdd(tail.done + cur_chunk, 1);
      }
      if (tail.trace && threadIdx.x == 0) atomicMax(tail.trace + 2 + cur_chunk, now_ns());
      tile += tail.gemm_ctas;
      is_first_matmul_in_slice = true;
      init_slice();
      // A CTA's consecutive tiles are other blocks or chunks: always reload A.
      should_load_a = true;

      if (slice_iters) {
        a_gl_rd_col = (threadIdx.x % a_gl_rd_delta_o);
#pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd_base + b_sh_stride * slice_col;

        bias_gl_rd = (thread_n_blocks * 16 / 8) * slice_col + threadIdx.x;
        // Update slice k/n for scales loading
        if constexpr (has_act_order) {
          slice_k_start = tb_k * slice_row;
          slice_k_finish = slice_k_start + tb_k * slice_iters;
          slice_k_start_shared_fetch = slice_k_start;
          slice_n_offset = act_s_col_tb_stride * slice_col;
        } else {
          if constexpr (group_blocks == -1) {
            s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
            zp_gl_rd = zp_sh_stride * slice_col + threadIdx.x;
          } else if constexpr (group_blocks >= thread_k_blocks) {
            s_gl_rd =
                s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + s_sh_stride * slice_col + threadIdx.x;
            zp_gl_rd =
                zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks) + zp_sh_stride * slice_col + threadIdx.x;
          } else {
            s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / group_blocks + threadIdx.x / s_sh_stride) +
                      s_sh_stride * slice_col + threadIdx.x % s_sh_stride;
            zp_gl_rd = zp_gl_stride * ((thread_k_blocks * slice_row) / group_blocks + threadIdx.x / zp_sh_stride) +
                       zp_sh_stride * slice_col + threadIdx.x % zp_sh_stride;
          }
        }
        start_pipes();
      }
    }
  }
  if (tail.trace && threadIdx.x == 0) atomicMax(tail.trace + 1, now_ns());
}


}  // namespace device::fn_moe_down_ar

// The routed experts' down projection of Flash-Next's EP target: bf16 activations, uint4b8 weights with bf16 group-128
// scales in Marlin's layout, 8-row MoE blocks, Marlin's own decode config (128 threads, 128 x 64 tiles, 4 stages).
// fused = false writes `c` only (the reference: stock sum, gate and all-reduce follow); fused = true also writes `out`,
// the all-reduced MoE output. `c` must be zero where the slot's expert is not local, as fused_marlin_moe leaves it.
template <typename scalar_t>
void moe_down_ar(
    tvm::ffi::TensorView a,
    tvm::ffi::TensorView c,
    tvm::ffi::TensorView b_q_weight,
    tvm::ffi::TensorView b_scales,
    tvm::ffi::TensorView sorted_token_ids,
    tvm::ffi::TensorView expert_ids,
    tvm::ffi::TensorView num_tokens_post_padded,
    tvm::ffi::TensorView topk_weights,
    tvm::ffi::TensorView shared,
    tvm::ffi::TensorView gate,
    tvm::ffi::TensorView out,
    tvm::ffi::TensorView done,
    tvm::ffi::TensorView state,
    tvm::ffi::TensorView trace,
    int64_t comm_data,
    int64_t comm_flags,
    int64_t comm_half,
    int64_t comm_timeout,
    int64_t rank,
    int64_t top_k,
    int64_t size_m,
    int64_t size_n,
    int64_t size_k,
    int64_t chunk_tiles,
    int64_t comm_ctas,
    int64_t comm_groups,
    int64_t publish,
    bool fused,
    double scale) {
  using namespace host;
  using namespace device::marlin_moe;
  constexpr int kThreads = 128, kThreadN = 128, kThreadK = 64, kGroupBlocks = 8;

  auto device = SymbolicDevice{};
  device.set_options<kDLCUDA>();
  TensorMatcher({-1, -1}).with_dtype<scalar_t>().with_device(device).verify(a);
  for (auto t : {c, b_q_weight, b_scales, sorted_token_ids, expert_ids, num_tokens_post_padded, topk_weights, shared,
                 gate, out, done, state}) {
    device.verify(t.device());
    RuntimeCheck(t.is_contiguous(), "moe_down_ar: every tensor must be contiguous");
  }
  RuntimeCheck(a.size(0) == size_m * top_k && a.size(1) == size_k, "moe_down_ar: a must be [m * top_k, k]");
  RuntimeCheck(c.size(0) == size_m * top_k && c.size(1) == size_n, "moe_down_ar: c must be [m * top_k, n]");
  RuntimeCheck(b_scales.dim() == 3 && b_scales.size(1) * 128 == size_k && b_scales.size(2) == size_n,
               "moe_down_ar: group-128 scales [experts, k / 128, n] only");
  RuntimeCheck(size_n % (kThreadN * chunk_tiles) == 0, "moe_down_ar: n must split into chunks of chunk_tiles tiles");
  RuntimeCheck(size_k % kThreadK == 0, "moe_down_ar: k must be a multiple of 64");
  RuntimeCheck(!fused || (comm_groups >= 1 && comm_ctas % comm_groups == 0 &&
                          comm_ctas / comm_groups <= device::fn_moe_down_ar::kCommSlots),
               "moe_down_ar: comm CTAs in equal groups of at most 32");
  RuntimeCheck(top_k <= device::fn_moe_down_ar::kMaxTopK, "moe_down_ar: top_k above 16");
  RuntimeCheck(publish >= 0 && publish <= 2, "moe_down_ar: publish is 0 (fence), 1 (release) or 2 (LL)");
  RuntimeCheck(size_n / (kThreadN * chunk_tiles) <= device::fn_moe_down_ar::kMaxChunks, "moe_down_ar: over 20 chunks");
  RuntimeCheck(trace.size(0) == 0 || trace.size(0) == device::fn_moe_down_ar::kTraceLength,
               "moe_down_ar: trace must be empty or kTraceLength long");
  if (size_m == 0) return;

  DLDevice dl_device = device.unwrap();
  int dev = dl_device.device_id;
  cudaStream_t stream = LaunchKernel::resolve_device(dl_device);
  int sms = 0, max_shared_mem = 0;
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  RuntimeDeviceCheck(cudaDeviceGetAttribute(&max_shared_mem, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));

  auto kernel = device::fn_moe_down_ar::MoeDownAR<
      scalar_t, kU4B8.id(), kBFloat16.id(), kThreads, 1, kThreadN / 16, kThreadK / 16, true, pipe_stages,
      kGroupBlocks, false, true, false>;
  // Marlin's occupancy rule (determine_exec_config) for this config, so CTAs and shared memory match stock.
  thread_config_t config{kThreadK, kThreadN, kThreads};
  int cache_size = get_kernel_cache_size(config, true, 1, size_m, size_n, size_k, 4, 128, false, true, false, false);
  cudaFuncAttributes attr;
  RuntimeDeviceCheck(cudaFuncGetAttributes(&attr, kernel));
  int reg_size = std::max(attr.numRegs, 1) * kThreads * 4;
  int per_sm = std::min(
      255 * 1024 / reg_size, max_shared_mem / (cache_size + kSharedMemoryValidityMargin + kSharedMemoryLaunchReserve));
  per_sm = std::max(std::min(per_sm, 4), 1);
  if (per_sm > 1) max_shared_mem = max_shared_mem / per_sm - kSharedMemoryLaunchReserve;
  int blocks = sms * per_sm;
  RuntimeCheck(!fused || comm_ctas < blocks, "moe_down_ar: too many comm CTAs");

  device::fn_moe_down_ar::HostComm hc{
      reinterpret_cast<char*>(comm_data), reinterpret_cast<unsigned*>(comm_flags),
      static_cast<unsigned*>(state.data_ptr()), comm_half, comm_timeout, static_cast<int>(rank), 0};
  device::fn_moe_down_ar::Tail tail{
      static_cast<const int4*>(shared.data_ptr()), static_cast<const float*>(gate.data_ptr()),
      static_cast<int4*>(out.data_ptr()), static_cast<int*>(done.data_ptr()), static_cast<float>(scale),
      static_cast<int>(chunk_tiles), fused ? blocks - static_cast<int>(comm_ctas) : blocks,
      fused ? static_cast<int>(comm_ctas) : 0, static_cast<int>(comm_groups), fused ? 1 : 0, static_cast<int>(size_m), static_cast<int>(top_k),
      trace.size(0) > 0 ? static_cast<unsigned long long*>(trace.data_ptr()) : nullptr, static_cast<int>(publish)};

  RuntimeDeviceCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, max_shared_mem));
  // clang-format off
  kernel<<<blocks, kThreads, max_shared_mem, stream>>>(
      static_cast<const int4*>(a.data_ptr()), static_cast<const int4*>(b_q_weight.data_ptr()),
      static_cast<int4*>(c.data_ptr()), nullptr, nullptr, static_cast<const int4*>(b_scales.data_ptr()), nullptr,
      nullptr, nullptr, static_cast<const int32_t*>(sorted_token_ids.data_ptr()),
      static_cast<const int32_t*>(expert_ids.data_ptr()),
      static_cast<const int32_t*>(num_tokens_post_padded.data_ptr()),
      static_cast<const float*>(topk_weights.data_ptr()), 1, true, true,
      static_cast<int>(b_scales.size(1)), static_cast<int>(size_m * top_k), static_cast<int>(size_n),
      static_cast<int>(size_k), nullptr, false, false, true, max_shared_mem, hc, tail);
  // clang-format on
  RuntimeDeviceCheck(cudaGetLastError());
}

}  // namespace sglang
