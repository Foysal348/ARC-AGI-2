// host_ar_layout.h - P5 host-staged all-reduce: layout and partition helpers shared by the CUDA kernel (host_ar.cu),
// the torch binding (host_ar_binding.cpp), the CPU coverage check (host_ar_check.h) and the local g++ test.
// Plain C++ (no CUDA or torch headers) so the same index math is compiled for device, host and the local test.
#pragma once
#include <stdint.h>

#if defined(__CUDACC__)
#define P5_HD __host__ __device__ __forceinline__
#else
#define P5_HD inline
#endif

namespace p5 {

constexpr int kVersion = 1;
constexpr int kMaxWorld = 8;     // flag/status arrays are sized for 8 ranks; kernels are instantiated for 2, 4, 8
constexpr int kMaxBlocks = 64;   // per-block flags; grid must be <= this (and co-resident, see README)
constexpr int kAutoMaxBlocks = 48;  // auto grid cap: L4 has 58 SMs, 1 block/SM, leave room for other work
constexpr int kThreads = 512;    // threads per block (multiple of 4: INT8 quads)
constexpr int kUnit = 8;         // elements per thread work unit (16 B of bf16/fp16, 32 B of fp32)
constexpr int kGroup = 32;       // elements per INT8 scale group = 4 consecutive units = one lane quad
constexpr int64_t kPage = 4096;

// Shared-memory file layout (byte offsets from the mapping base).
constexpr int64_t kHeaderOff = 0;       // ShmHeader (4 KiB)
constexpr int64_t kFlagsOff = 4096;     // uint64 flags[2][kMaxWorld][kMaxBlocks] = 8 KiB
constexpr int64_t kSigOff = 12288;      // reserved (the call signature now travels in the flag words)
constexpr int64_t kDataOff = 16384;     // stage slots (world x slot_bytes), then the reduced area (slot_bytes)

enum DType : int { F32 = 0, BF16 = 1, F16 = 2 };
enum Mode : int { NATIVE = 0, INT8 = 1 };
// Error codes written by the kernel into ShmHeader::status[rank] as (code << 24) | (phase << 16) | block.
enum Err : int { E_OK = 0, E_TIMEOUT = 1, E_DESYNC = 2, E_PEER_ABORT = 3, E_SIG = 4 };

constexpr uint64_t kMagic = 0x5035484152763031ull;  // "P5HARv01"

struct ShmHeader {
  uint64_t magic;        // set last by rank 0 (release) once the header is valid
  int32_t world;
  int32_t version;
  int64_t max_bytes;
  int64_t slot_bytes;
  int64_t total_bytes;
  int32_t attached;      // CPU atomic counter: ranks that finished cudaHostRegister
  int32_t pad0;
  uint32_t abort_flag;   // GPU-written: any rank hit an error; every wait loop exits when it sees != 0
  uint32_t pad1;
  uint32_t status[kMaxWorld];  // GPU-written per-rank error word (0 = ok)
};

P5_HD int dtype_size(int dt) { return dt == F32 ? 4 : 2; }
P5_HD int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
P5_HD int64_t round_up(int64_t a, int64_t b) { return ceil_div(a, b) * b; }
P5_HD int64_t min64(int64_t a, int64_t b) { return a < b ? a : b; }
P5_HD int64_t max64(int64_t a, int64_t b) { return a > b ? a : b; }

// Rank chunk partition of n elements: every chunk starts at a multiple of kGroup elements (so INT8 groups never straddle
// two owners and every unit start is 16-B aligned for all dtypes); the last non-empty chunk may be shorter; for tiny n
// trailing chunks may be empty (begin == end == n).
P5_HD int64_t chunk_len(int64_t n, int world) { return round_up(ceil_div(n, world), kGroup); }
P5_HD int64_t chunk_begin(int64_t n, int world, int j) { return min64(n, chunk_len(n, world) * j); }
P5_HD int64_t chunk_end(int64_t n, int world, int j) { return chunk_begin(n, world, j + 1); }
// Unit loop bound for a chunk of len elements, rounded up to a multiple of 4 units so the 4 lanes of an INT8 quad run
// the same number of iterations (grid stride is a multiple of 4, lanes 4q..4q+3 hold consecutive unit indices).
P5_HD int64_t chunk_units4(int64_t len) { return round_up(ceil_div(len, kUnit), 4); }
// Valid elements of the unit starting at element e0 inside a chunk ending at ce (<= 0: unit is padding only).
P5_HD int unit_count(int64_t e0, int64_t ce) { return (int)max64(-1, min64(kUnit, ce - e0)); }
// Block that owns relative unit u of any chunk (kernel loop: u = blockIdx*threads + tid + k*stride).
P5_HD int unit_block(int64_t u, int blocks, int threads) { return (int)((u % ((int64_t)blocks * threads)) / threads); }

// Staging layouts inside one slot (stage slot of a writer rank, or the reduced area). Element e of the full tensor is
// always stored at the same position, whichever chunk it belongs to.
P5_HD int64_t native_off(int64_t e, int dt) { return e * dtype_size(dt); }
P5_HD int64_t q_payload_off(int64_t e) { return e; }  // int8 payload, 1 B per element
P5_HD int64_t q_scale_base(int64_t n) { return round_up(round_up(n, kGroup), 256); }
P5_HD int64_t q_scale_off(int64_t n, int64_t e) { return q_scale_base(n) + (e / kGroup) * 4; }  // fp32 scale per group
// Bytes of a slot touched by a call of n elements (INT8 payload writes are whole 8-B units, so up to round_up(n, 32)).
P5_HD int64_t slot_need(int64_t n, int dt, int mode) {
  return mode == INT8 ? q_scale_base(n) + ceil_div(n, kGroup) * 4 : round_up(n, kGroup) * dtype_size(dt);
}
// Slot size for a handle created with max_bytes (largest native payload): native fits by construction, INT8 needs
// 1.125 n + 256 <= 0.5625 max_bytes + 256 for 2-B dtypes and 0.28 max_bytes + 256 for fp32.
P5_HD int64_t slot_bytes_for(int64_t max_bytes) { return round_up(max_bytes + 4096, kPage); }
P5_HD int64_t total_bytes_for(int world, int64_t max_bytes) {
  return kDataOff + (int64_t)(world + 1) * slot_bytes_for(max_bytes);
}
P5_HD int64_t flag_index(int phase, int rank, int block) {
  return ((int64_t)phase * kMaxWorld + rank) * kMaxBlocks + block;
}

// Auto grid: one block per 512 units of a rank chunk, 1..kAutoMaxBlocks.
P5_HD int auto_blocks(int64_t n, int world) {
  int64_t u = chunk_units4(chunk_len(n, world));
  int64_t b = ceil_div(u, kThreads);
  return (int)min64(kAutoMaxBlocks, max64(1, b));
}

// Call signature checked across ranks every call (all ranks must pass identical n / dtype / mode / grid / world).
P5_HD uint64_t call_sig(int64_t n, int dt, int mode, int blocks, int world) {
  uint64_t h = 1469598103934665603ull;
  uint64_t v[5] = {(uint64_t)n, (uint64_t)dt, (uint64_t)mode, (uint64_t)blocks, (uint64_t)world};
  for (int i = 0; i < 5; ++i) { h ^= v[i] + 0x9e3779b97f4a7c15ull; h *= 1099511628211ull; }
  return h | 1ull;  // never 0 (0 = never written)
}

// Flag word = (epoch << 24) | sig24. Epochs keep 40 bits (10^12 calls), flags stay monotonic (epoch in the high bits).
constexpr int kSigBits = 24;
constexpr uint64_t kSigMask = (1ull << kSigBits) - 1ull;
P5_HD uint32_t sig24_of(uint64_t sig) { return (uint32_t)((sig ^ (sig >> 24) ^ (sig >> 48)) & kSigMask); }
P5_HD uint64_t flag_value(uint64_t epoch, uint32_t sig24) { return (epoch << kSigBits) | (uint64_t)(sig24 & kSigMask); }
P5_HD uint64_t flag_epoch(uint64_t v) { return v >> kSigBits; }
P5_HD uint32_t flag_sig(uint64_t v) { return (uint32_t)(v & kSigMask); }

}  // namespace p5
