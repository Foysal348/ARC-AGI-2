// host_ar.cu - P5 four-rank HOST-STAGED two-shot all-reduce for GPUs without peer access (4x L4, PCIe, no P2P).
//
// One kernel per call, stream ordered, CUDA-graph capturable, no CPU work per call:
//   phase 1 (scatter)  rank r copies chunk j (j != r) of its input into its own host slot  stage[r]  (GPU -> host writes)
//                      then publishes flags[0][r][block] = epoch
//   phase 2 (reduce)   rank r waits flags[0][j][block] == epoch for all j != r, reads chunk r from every stage[j]
//                      (host -> GPU reads), sums in FP32 in rank order 0..W-1 with its own chunk from device memory,
//                      writes the result to out (device) and to the reduced area (host), publishes flags[1][r][block]
//   phase 3 (gather)   rank r waits flags[1][j][block] == epoch, copies chunk j of the reduced area into out
// Bytes per GPU per call (size S): writes S (3S/4 + S/4), reads 3S/2 (2 x 3S/4) = the two-shot 2(N-1)/N S each way.
//
// The epoch is NOT a kernel argument (a graph would freeze it): every block reads a per-handle device counter `seq` at
// start (epoch = seq + 1), and the last block to finish stores seq = epoch. Flags are monotonic, so replays never need
// a reset. Every wait has a %globaltimer timeout and also exits on the shared abort flag; errors are reported in the
// host-mapped status words (read with the `status` op), never by hanging. A failed call is LOUD: every block that saw
// an error fills its own units of `out` with NaN, and the abort flag is sticky, so every later call (or graph replay)
// on the poisoned handle also returns all-NaN instead of stale data. Build with -DP5_TRAP_ON_ERROR to __trap() instead
// (kills the CUDA context). See README.md for the memory-ordering argument and the race review.
#include <cuda_runtime.h>
#include <stdint.h>

#include "host_ar_layout.h"

using namespace p5;

struct P5Params {
  const void* in;          // device memory, n elements of the dtype, 16-B aligned
  void* out;               // device memory (may equal in)
  char* stage;             // device pointer of the mapped host memory: W slots of slot_bytes
  char* reduced;           // device pointer of the mapped host memory: 1 slot
  uint64_t* flags;         // mapped host memory: [2][kMaxWorld][kMaxBlocks]
  uint32_t* abort_flag;    // mapped host memory
  uint32_t* status;        // mapped host memory: [kMaxWorld]
  unsigned long long* seq; // device memory: completed epochs of this handle on this rank
  unsigned int* arrive;    // device memory: blocks finished in the current call
  int64_t n;
  int64_t slot_bytes;
  uint64_t timeout_ns;
  uint32_t sig24;          // call signature (n, dtype, mode, grid, world), carried in the low 24 bits of every flag
  int rank;
};

// ------------------------------------------------------------------------------------------------ memory primitives
__device__ __forceinline__ uint64_t globaltimer_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}
// Flag poll: acquire at system scope (pairs with the writer's fence.sc.sys + relaxed.sys store).
__device__ __forceinline__ uint64_t ld_acquire_sys_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_relaxed_sys_u64(uint64_t* p, uint64_t v) {
  asm volatile("st.relaxed.sys.global.u64 [%0], %1;" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ uint64_t ld_volatile_u64(const uint64_t* p) {
  uint64_t v;
  asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ uint32_t ld_volatile_u32(const uint32_t* p) {
  uint32_t v;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}
__device__ __forceinline__ void st_volatile_u32(uint32_t* p, uint32_t v) {
  asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(p), "r"(v) : "memory");
}
// Reads of peer-written host memory: volatile (= relaxed.sys in the PTX model), never served from a stale L1 line.
__device__ __forceinline__ uint4 ld_sys_v4(const void* p) {
  uint4 r;
  asm volatile("ld.volatile.global.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p) : "memory");
  return r;
}
__device__ __forceinline__ uint2 ld_sys_v2(const void* p) {
  uint2 r;
  asm volatile("ld.volatile.global.v2.u32 {%0,%1}, [%2];" : "=r"(r.x), "=r"(r.y) : "l"(p) : "memory");
  return r;
}
__device__ __forceinline__ uint16_t ld_sys_u16(const void* p) {
  uint16_t r;
  asm volatile("ld.volatile.global.u16 %0, [%1];" : "=h"(r) : "l"(p) : "memory");
  return r;
}
__device__ __forceinline__ uint32_t ld_sys_u32(const void* p) {
  uint32_t r;
  asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(r) : "l"(p) : "memory");
  return r;
}

// ------------------------------------------------------------------------------------------------ dtype conversions
// Inline PTX instead of cuda_bf16.h / cuda_fp16.h (keeps the .cu free of CCCL headers; cvt.rn = round to nearest even,
// the same rounding torch uses for float -> bfloat16 / half).
__device__ __forceinline__ float bf16_to_f(uint16_t h) { return __uint_as_float(((uint32_t)h) << 16); }
__device__ __forceinline__ uint16_t f_to_bf16(float f) {
  uint16_t r;
  asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
  return r;
}
__device__ __forceinline__ float f16_to_f(uint16_t h) {
  float f;
  asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
  return f;
}
__device__ __forceinline__ uint16_t f_to_f16(float f) {
  uint16_t r;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(r) : "f"(f));
  return r;
}

template <int DT> struct Elem { using S = uint16_t; };
template <> struct Elem<F32> { using S = float; };

template <int DT> __device__ __forceinline__ float to_f(typename Elem<DT>::S x);
template <> __device__ __forceinline__ float to_f<F32>(float x) { return x; }
template <> __device__ __forceinline__ float to_f<BF16>(uint16_t x) { return bf16_to_f(x); }
template <> __device__ __forceinline__ float to_f<F16>(uint16_t x) { return f16_to_f(x); }
template <int DT> __device__ __forceinline__ typename Elem<DT>::S from_f(float x);
template <> __device__ __forceinline__ float from_f<F32>(float x) { return x; }
template <> __device__ __forceinline__ uint16_t from_f<BF16>(float x) { return f_to_bf16(x); }
template <> __device__ __forceinline__ uint16_t from_f<F16>(float x) { return f_to_f16(x); }

// A unit = kUnit (8) elements = 16 B (2-B dtypes) or 32 B (fp32) = 1 or 2 uint4.
template <int DT> struct UnitV { static constexpr int kV = (DT == F32) ? 2 : 1; uint4 v[kV]; };

template <int DT> __device__ __forceinline__ void unpack(const UnitV<DT>& u, float f[kUnit]) {
  if constexpr (DT == F32) {
    const float* p = reinterpret_cast<const float*>(u.v);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) f[i] = p[i];
  } else {
    const uint16_t* p = reinterpret_cast<const uint16_t*>(u.v);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) f[i] = to_f<DT>(p[i]);
  }
}
template <int DT> __device__ __forceinline__ void pack(const float f[kUnit], UnitV<DT>& u) {
  if constexpr (DT == F32) {
    float* p = reinterpret_cast<float*>(u.v);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) p[i] = f[i];
  } else {
    uint16_t* p = reinterpret_cast<uint16_t*>(u.v);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) p[i] = from_f<DT>(f[i]);
  }
}

// Raw (bit-exact) unit I/O. base points at element 0; e0 is a multiple of kUnit; c = valid elements (1..8).
// Partial units (c < 8, only at the very end of the tensor) go element by element and never touch memory past n.
template <int DT> __device__ __forceinline__ void load_unit_dev(const void* base, int64_t e0, int c, UnitV<DT>& u) {
  using S = typename Elem<DT>::S;
  const S* p = reinterpret_cast<const S*>(base) + e0;
  if (c == kUnit) {
#pragma unroll
    for (int k = 0; k < UnitV<DT>::kV; ++k) u.v[k] = reinterpret_cast<const uint4*>(p)[k];
  } else {
    S* q = reinterpret_cast<S*>(u.v);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      if (i < c) q[i] = p[i];  // never read past element n-1
      else q[i] = S(0);
    }
  }
}
template <int DT> __device__ __forceinline__ void load_unit_sys(const char* slot, int64_t e0, int c, UnitV<DT>& u) {
  using S = typename Elem<DT>::S;
  const char* p = slot + native_off(e0, DT);
  if (c == kUnit) {
#pragma unroll
    for (int k = 0; k < UnitV<DT>::kV; ++k) u.v[k] = ld_sys_v4(p + 16 * k);
  } else {
    S* q = reinterpret_cast<S*>(u.v);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) {
      if constexpr (DT == F32) {
        q[i] = (i < c) ? __uint_as_float(ld_sys_u32(p + 4 * i)) : 0.f;
      } else {
        q[i] = (i < c) ? ld_sys_u16(p + 2 * i) : (uint16_t)0;
      }
    }
  }
}
// Store a unit to device memory (dev) or to mapped host memory (plain weak stores; published by the block's
// __syncthreads + fence.sc.sys + flag store).
template <int DT> __device__ __forceinline__ void store_unit(void* base_or_slot, bool is_slot, int64_t e0, int c,
                                                            const UnitV<DT>& u) {
  using S = typename Elem<DT>::S;
  S* p = is_slot ? reinterpret_cast<S*>(reinterpret_cast<char*>(base_or_slot) + native_off(e0, DT))
                 : reinterpret_cast<S*>(base_or_slot) + e0;
  if (c == kUnit) {
#pragma unroll
    for (int k = 0; k < UnitV<DT>::kV; ++k) reinterpret_cast<uint4*>(p)[k] = u.v[k];
  } else {
    const S* q = reinterpret_cast<const S*>(u.v);
    for (int i = 0; i < c; ++i) p[i] = q[i];
  }
}

// ------------------------------------------------------------------------------------------------ INT8 codec
// Symmetric absmax per group of 32 elements (= 4 units = 4 consecutive lanes). All 4 lanes of a quad must call these
// (loops are quad-uniform, see chunk_units4); padding units pass zeros.
// Semantics:
//   * a group with any NaN/Inf element gets scale = NaN (payload 0): its 32 dequantized values are NaN, so a
//     non-finite input poisons its whole group of the result (loud, group-granular; Inf becomes NaN);
//   * 0 < amax < kQuantFloor is quantized with amax = kQuantFloor (inv = 127 / amax stays finite): the absolute error
//     of such a group is <= kQuantFloor / 254 instead of every element snapping to +-amax.
constexpr float kQuantFloor = 1e-36f;  // largest-resolution floor with 127 / floor = 1.27e38 < FLT_MAX
__device__ __forceinline__ float quad_absmax(float a) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned mask = 0xFu << (lane & 28u);
  a = fmaxf(a, __shfl_xor_sync(mask, a, 1));
  a = fmaxf(a, __shfl_xor_sync(mask, a, 2));
  return a;
}
__device__ __forceinline__ int quad_or(int v) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned mask = 0xFu << (lane & 28u);
  v |= __shfl_xor_sync(mask, v, 1);
  v |= __shfl_xor_sync(mask, v, 2);
  return v;
}
// Quantize 8 floats with the quad's scale. Returns the packed int8 payload; *scale_out = amax / 127, or NaN when the
// group holds a non-finite value.
__device__ __forceinline__ uint2 quantize8(const float f[kUnit], float* scale_out) {
  float a = 0.f;
  int nonfinite = 0;
#pragma unroll
  for (int i = 0; i < kUnit; ++i) {
    nonfinite |= ((__float_as_uint(f[i]) & 0x7f800000u) == 0x7f800000u) ? 1 : 0;  // exponent all ones: Inf or NaN
    a = fmaxf(a, fabsf(f[i]));
  }
  a = quad_absmax(a);
  nonfinite = quad_or(nonfinite);  // quad-uniform like the absmax shuffle
  uint32_t w[2] = {0u, 0u};
  if (nonfinite) {
    *scale_out = __int_as_float(0x7fc00000);  // NaN scale: 0 * NaN = NaN for every element of the group
    return make_uint2(0u, 0u);
  }
  if (a > 0.f) a = fmaxf(a, kQuantFloor);
  const float scale = a / 127.f;
  const float inv = (a > 0.f) ? (127.f / a) : 0.f;
#pragma unroll
  for (int i = 0; i < kUnit; ++i) {
    int q = __float2int_rn(f[i] * inv);
    q = q > 127 ? 127 : (q < -127 ? -127 : q);
    w[i >> 2] |= ((uint32_t)(q & 0xFF)) << (8 * (i & 3));
  }
  *scale_out = scale;
  return make_uint2(w[0], w[1]);
}
__device__ __forceinline__ void dequant8(uint2 q, float scale, float f[kUnit]) {
  const uint32_t w[2] = {q.x, q.y};
#pragma unroll
  for (int i = 0; i < kUnit; ++i) {
    const int v = (int)(int8_t)((w[i >> 2] >> (8 * (i & 3))) & 0xFF);
    f[i] = __int2float_rn(v) * scale;
  }
}

// ------------------------------------------------------------------------------------------------ synchronization
// Wait until flag_epoch(*f) == e. A flag word is (epoch << 24) | sig24 (host_ar_layout.h flag_value): one acquire load
// gives both the epoch and the writer's call signature, so EVERY block (not only block 0) detects a mismatched call.
// Returns an Err code. Bounded by timeout_ns (%globaltimer) and by the shared abort flag.
__device__ __noinline__ int wait_flag(const uint64_t* f, uint64_t e, uint32_t sig24, const uint32_t* abortp,
                                      uint64_t timeout_ns) {
  const uint64_t t0 = globaltimer_ns();
  for (uint32_t it = 0;; ++it) {
    const uint64_t v = ld_acquire_sys_u64(f);
    const uint64_t ve = flag_epoch(v);
    if (ve >= e) {
      if (ve != e) return E_DESYNC;  // a peer can never be ahead by design: ahead = protocol broken
      return flag_sig(v) == sig24 ? E_OK : E_SIG;
    }
    // back off after the first polls: every poll is a non-posted 8-B PCIe read competing with the data reads
    if (it >= 8u) __nanosleep(it < 64u ? 100u : 200u);
    if ((it & 15u) == 15u) {
      if (ld_volatile_u32(abortp) != 0u) return E_PEER_ABORT;
      if (globaltimer_ns() - t0 > timeout_ns) return E_TIMEOUT;
    }
  }
}

// Publish: all threads' prior stores to host memory (barrier) -> system fence -> flag store (thread 0 only).
__device__ __forceinline__ void publish(uint64_t* flag, uint64_t fv) {
  __syncthreads();
  if (threadIdx.x == 0) {
    __threadfence_system();  // fence.sc.sys: cumulative over the block's writes ordered before it by bar.sync
    st_relaxed_sys_u64(flag, fv);
  }
}

// Threads 0..W-1 (except the own rank) each wait for one peer's flag; any error (including a call-signature mismatch:
// different n / dtype / mode / grid on some rank) lands in s_err.
template <int W>
__device__ __forceinline__ void wait_peers(const P5Params& p, int phase, uint64_t e, volatile int* s_err) {
  const int tid = threadIdx.x;
  if (tid < W && tid != p.rank) {
    const int code = wait_flag(p.flags + flag_index(phase, tid, blockIdx.x), e, p.sig24, p.abort_flag, p.timeout_ns);
    if (code != E_OK) *s_err = code;
  }
  __syncthreads();
}

// NaN of the dtype for the error fill.
template <int DT> __device__ __forceinline__ typename Elem<DT>::S nan_of();
template <> __device__ __forceinline__ float nan_of<F32>() { return __int_as_float(0x7fc00000); }
template <> __device__ __forceinline__ uint16_t nan_of<BF16>() { return (uint16_t)0x7fc0u; }
template <> __device__ __forceinline__ uint16_t nan_of<F16>() { return (uint16_t)0x7e00u; }

// ------------------------------------------------------------------------------------------------ the kernel
template <int W, int DT, int MODE>
__global__ void __launch_bounds__(kThreads, 1) p5_ar_kernel(P5Params p) {
  __shared__ unsigned long long s_epoch;
  __shared__ int s_err;
  __shared__ int s_phase;
  const int tid = threadIdx.x;
  const int r = p.rank;
  const int64_t n = p.n;
  const int64_t gt = (int64_t)blockIdx.x * blockDim.x + tid;
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;

  if (tid == 0) {
    s_epoch = *reinterpret_cast<volatile unsigned long long*>(p.seq) + 1ull;
    s_err = (ld_volatile_u32(p.abort_flag) != 0u) ? E_PEER_ABORT : E_OK;
    s_phase = 0;
  }
  __syncthreads();
  const uint64_t e = s_epoch;
  char* my_slot = p.stage + (int64_t)r * p.slot_bytes;

  // Phases 1 and 3 walk relative unit u of all W-1 foreign chunks together (outer u loop, inner jj unrolled) so every
  // load of a thread is issued before the first dependent store: W-1 PCIe round trips overlap instead of serializing.
  // The bound UM is the unit count of chunk 0 (the longest); units past a shorter chunk's end have c <= 0 and only
  // feed zeros to the INT8 quad shuffles. unit u -> block mapping is the same as in phase 2 (u % stride).
  const int64_t UM = chunk_units4(chunk_end(n, W, 0));

  // ---------------- phase 1: stage chunks j != r into my slot
  if (s_err == E_OK) {
#pragma unroll 1
    for (int64_t u = gt; u < UM; u += stride) {
      UnitV<DT> v[W];
      int64_t e0s[W];
      int cs[W];
#pragma unroll
      for (int jj = 1; jj < W; ++jj) {
        const int j = (r + jj) % W;
        e0s[jj] = chunk_begin(n, W, j) + u * kUnit;
        cs[jj] = unit_count(e0s[jj], chunk_end(n, W, j));
        if (cs[jj] > 0) load_unit_dev<DT>(p.in, e0s[jj], cs[jj], v[jj]);
      }
#pragma unroll
      for (int jj = 1; jj < W; ++jj) {
        if constexpr (MODE == NATIVE) {
          if (cs[jj] > 0) store_unit<DT>(my_slot, true, e0s[jj], cs[jj], v[jj]);
        } else {
          float f[kUnit];
          if (cs[jj] > 0) {
            unpack<DT>(v[jj], f);
          } else {
#pragma unroll
            for (int i = 0; i < kUnit; ++i) f[i] = 0.f;
          }
          float sc;
          const uint2 q = quantize8(f, &sc);  // quad-uniform: every lane of the quad reaches this for every jj
          if (cs[jj] > 0) {
            *reinterpret_cast<uint2*>(my_slot + q_payload_off(e0s[jj])) = q;
            if ((tid & 3) == 0) *reinterpret_cast<float*>(my_slot + q_scale_off(n, e0s[jj])) = sc;
          }
        }
      }
    }
    publish(p.flags + flag_index(0, r, blockIdx.x), flag_value(e, p.sig24));
    wait_peers<W>(p, 0, e, &s_err);
    if (tid == 0) s_phase = 1;
  }

  // ---------------- phase 2: reduce my chunk r (FP32, rank order 0..W-1), write out + reduced area
  __syncthreads();
  if (s_err == E_OK) {
    const int64_t cb = chunk_begin(n, W, r), ce = chunk_end(n, W, r);
    const int64_t U = chunk_units4(ce - cb);
#pragma unroll 1
    for (int64_t u = gt; u < U; u += stride) {
      const int64_t e0 = cb + u * kUnit;
      const int c = unit_count(e0, ce);
      if constexpr (MODE == NATIVE) {
        if (c <= 0) continue;
        UnitV<DT> v[W];
#pragma unroll
        for (int j = 0; j < W; ++j) {  // issue every load first (memory-level parallelism over PCIe)
          if (j == r) load_unit_dev<DT>(p.in, e0, c, v[j]);
          else load_unit_sys<DT>(p.stage + (int64_t)j * p.slot_bytes, e0, c, v[j]);
        }
        float acc[kUnit], f[kUnit];
        unpack<DT>(v[0], acc);  // acc = x0 (not 0 + x0: keeps -0 sums identical to torch's x0 + x1 + ...)
#pragma unroll
        for (int j = 1; j < W; ++j) {
          unpack<DT>(v[j], f);
#pragma unroll
          for (int i = 0; i < kUnit; ++i) acc[i] += f[i];
        }
        UnitV<DT> o;
        pack<DT>(acc, o);
        store_unit<DT>(p.out, false, e0, c, o);
        store_unit<DT>(p.reduced, true, e0, c, o);
      } else {
        float acc[kUnit], f[kUnit];
#pragma unroll
        for (int i = 0; i < kUnit; ++i) acc[i] = 0.f;
        if (c > 0) {
          uint2 q[W];
          float s[W];
          UnitV<DT> own;
#pragma unroll
          for (int j = 0; j < W; ++j) {
            if (j == r) {
              load_unit_dev<DT>(p.in, e0, c, own);
            } else {
              const char* sl = p.stage + (int64_t)j * p.slot_bytes;
              q[j] = ld_sys_v2(sl + q_payload_off(e0));
              s[j] = __uint_as_float(ld_sys_u32(sl + q_scale_off(n, e0)));
            }
          }
#pragma unroll
          for (int j = 0; j < W; ++j) {
            if (j == r) unpack<DT>(own, f);
            else dequant8(q[j], s[j], f);
            if (j == 0) {
#pragma unroll
              for (int i = 0; i < kUnit; ++i) acc[i] = f[i];
            } else {
#pragma unroll
              for (int i = 0; i < kUnit; ++i) acc[i] += f[i];
            }
          }
          for (int i = c; i < kUnit; ++i) acc[i] = 0.f;  // padding elements never enter the group scale
        }
        float s_red;
        const uint2 qr = quantize8(acc, &s_red);  // quad-uniform
        if (c > 0) {
          // The owner's own output is the dequantized value too, so every rank ends with identical bits.
          float d[kUnit];
          dequant8(qr, s_red, d);
          UnitV<DT> o;
          pack<DT>(d, o);
          store_unit<DT>(p.out, false, e0, c, o);
          *reinterpret_cast<uint2*>(p.reduced + q_payload_off(e0)) = qr;
          if ((tid & 3) == 0) *reinterpret_cast<float*>(p.reduced + q_scale_off(n, e0)) = s_red;
        }
      }
    }
    publish(p.flags + flag_index(1, r, blockIdx.x), flag_value(e, p.sig24));
    wait_peers<W>(p, 1, e, &s_err);
    if (tid == 0) s_phase = 2;
  }

  // ---------------- phase 3: gather the reduced chunks j != r into out
  __syncthreads();
  if (s_err == E_OK) {
#pragma unroll 1
    for (int64_t u = gt; u < UM; u += stride) {
      UnitV<DT> o[W];
      uint2 q[W];
      float sc[W];
      int64_t e0s[W];
      int cs[W];
#pragma unroll
      for (int jj = 1; jj < W; ++jj) {  // all loads first
        const int j = (r + jj) % W;
        e0s[jj] = chunk_begin(n, W, j) + u * kUnit;
        cs[jj] = unit_count(e0s[jj], chunk_end(n, W, j));
        if (cs[jj] > 0) {
          if constexpr (MODE == NATIVE) {
            load_unit_sys<DT>(p.reduced, e0s[jj], cs[jj], o[jj]);
          } else {
            q[jj] = ld_sys_v2(p.reduced + q_payload_off(e0s[jj]));
            sc[jj] = __uint_as_float(ld_sys_u32(p.reduced + q_scale_off(n, e0s[jj])));
          }
        }
      }
#pragma unroll
      for (int jj = 1; jj < W; ++jj) {  // no shuffles in phase 3: per-thread skips are safe
        if (cs[jj] <= 0) continue;
        if constexpr (MODE == INT8) {
          float d[kUnit];
          dequant8(q[jj], sc[jj], d);
          pack<DT>(d, o[jj]);
        }
        store_unit<DT>(p.out, false, e0s[jj], cs[jj], o[jj]);
      }
    }
    if (tid == 0) s_phase = 3;
  }

  // ---------------- finish: on error fill this block's units of out with NaN, report, count blocks, advance epoch
  __syncthreads();
  if (s_err != E_OK) {  // block-uniform (read after a barrier)
    // The block owns relative units u = gt + k*stride of EVERY chunk (phase 2 for chunk r, phases 1/3 for the others,
    // same map), so these stores touch only its own elements and never race another block.
    using S = typename Elem<DT>::S;
    UnitV<DT> nanu;
    S* qn = reinterpret_cast<S*>(nanu.v);
#pragma unroll
    for (int i = 0; i < kUnit; ++i) qn[i] = nan_of<DT>();
#pragma unroll 1
    for (int64_t u = gt; u < UM; u += stride) {
#pragma unroll
      for (int j = 0; j < W; ++j) {
        const int64_t e0 = chunk_begin(n, W, j) + u * kUnit;
        const int c = unit_count(e0, chunk_end(n, W, j));
        if (c > 0) store_unit<DT>(p.out, false, e0, c, nanu);
      }
    }
  }
  if (tid == 0) {
    const int err = s_err;
    if (err != E_OK) {
      // First root cause wins: a peer-abort never overwrites a recorded error; a root cause may replace a recorded
      // peer-abort (another block of this rank can report its own timeout after one block saw the abort first).
      const uint32_t cur = ld_volatile_u32(p.status + r);
      if (cur == 0u || (err != E_PEER_ABORT && (cur >> 24) == (uint32_t)E_PEER_ABORT)) {
        st_volatile_u32(p.status + r, ((uint32_t)err << 24) | ((uint32_t)s_phase << 16) | (uint32_t)blockIdx.x);
      }
      __threadfence_system();
      st_volatile_u32(p.abort_flag, 1u);  // sticky: every other block / rank leaves its waits, later calls fill NaN
#ifdef P5_TRAP_ON_ERROR
      __trap();
#endif
    }
    __threadfence();
    const unsigned prev = atomicAdd(p.arrive, 1u);
    if (prev == gridDim.x - 1) {  // every block has read seq (it arrived), so the update cannot race a reader
      *reinterpret_cast<volatile unsigned int*>(p.arrive) = 0u;
      *reinterpret_cast<volatile unsigned long long*>(p.seq) = e;
      __threadfence();
    }
  }
}

// ------------------------------------------------------------------------------------------------ launcher
template <int W, int DT>
static cudaError_t launch_dm(const P5Params& p, int mode, int blocks, cudaStream_t s) {
  if (mode == NATIVE) p5_ar_kernel<W, DT, NATIVE><<<blocks, kThreads, 0, s>>>(p);
  else p5_ar_kernel<W, DT, INT8><<<blocks, kThreads, 0, s>>>(p);
  return cudaGetLastError();
}
template <int W>
static cudaError_t launch_w(const P5Params& p, int dtype, int mode, int blocks, cudaStream_t s) {
  switch (dtype) {
    case F32: return launch_dm<W, F32>(p, mode, blocks, s);
    case BF16: return launch_dm<W, BF16>(p, mode, blocks, s);
    case F16: return launch_dm<W, F16>(p, mode, blocks, s);
    default: return cudaErrorInvalidValue;
  }
}

// Largest grid whose blocks are all co-resident on this device (blocks of one rank wait on peers' blocks of the same
// index, so a grid that cannot be fully resident could stall until the timeout fires).
template <int W, int DT, int MODE>
static int coresident(int sms) {
  int per_sm = 0;
  if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, p5_ar_kernel<W, DT, MODE>, kThreads, 0) != cudaSuccess)
    return -1;
  return per_sm * sms;
}
int p5_max_coresident_blocks(int world, int device) {
  int sms = 0;
  if (cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device) != cudaSuccess) return -1;
  int best = 1 << 30;
  // the minimum over every instantiation of this world size (one cap per handle, whatever dtype/mode a call uses)
  int v[6];
  switch (world) {
    case 2: v[0] = coresident<2, F32, NATIVE>(sms); v[1] = coresident<2, BF16, NATIVE>(sms);
            v[2] = coresident<2, F16, NATIVE>(sms); v[3] = coresident<2, F32, INT8>(sms);
            v[4] = coresident<2, BF16, INT8>(sms); v[5] = coresident<2, F16, INT8>(sms); break;
    case 4: v[0] = coresident<4, F32, NATIVE>(sms); v[1] = coresident<4, BF16, NATIVE>(sms);
            v[2] = coresident<4, F16, NATIVE>(sms); v[3] = coresident<4, F32, INT8>(sms);
            v[4] = coresident<4, BF16, INT8>(sms); v[5] = coresident<4, F16, INT8>(sms); break;
    case 8: v[0] = coresident<8, F32, NATIVE>(sms); v[1] = coresident<8, BF16, NATIVE>(sms);
            v[2] = coresident<8, F16, NATIVE>(sms); v[3] = coresident<8, F32, INT8>(sms);
            v[4] = coresident<8, BF16, INT8>(sms); v[5] = coresident<8, F16, INT8>(sms); break;
    default: return -1;
  }
  for (int i = 0; i < 6; ++i) {
    if (v[i] < 0) return -1;
    best = v[i] < best ? v[i] : best;
  }
  return best;
}

// Called by host_ar_binding.cpp (argument checks are done there).
cudaError_t p5_launch(const void* in, void* out, char* dbase, int64_t slot_bytes, unsigned long long* seq,
                      unsigned int* arrive, int64_t n, int world, int rank, int dtype, int mode, int blocks,
                      uint64_t timeout_ns, uint64_t sig_value, cudaStream_t stream) {
  // (kSigOff area of the shm layout is unused since the signature moved into the flag words)
  P5Params p;
  p.in = in;
  p.out = out;
  p.stage = dbase + kDataOff;
  p.reduced = dbase + kDataOff + (int64_t)world * slot_bytes;
  p.flags = reinterpret_cast<uint64_t*>(dbase + kFlagsOff);
  ShmHeader* h = reinterpret_cast<ShmHeader*>(dbase + kHeaderOff);
  p.abort_flag = &h->abort_flag;
  p.status = h->status;
  p.seq = seq;
  p.arrive = arrive;
  p.n = n;
  p.slot_bytes = slot_bytes;
  p.timeout_ns = timeout_ns;
  p.sig24 = sig24_of(sig_value);
  p.rank = rank;
  switch (world) {
    case 2: return launch_w<2>(p, dtype, mode, blocks, stream);
    case 4: return launch_w<4>(p, dtype, mode, blocks, stream);
    case 8: return launch_w<8>(p, dtype, mode, blocks, stream);
    default: return cudaErrorInvalidValue;
  }
}
