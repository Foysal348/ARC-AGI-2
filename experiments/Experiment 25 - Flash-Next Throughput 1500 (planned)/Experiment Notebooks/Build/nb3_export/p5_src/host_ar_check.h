// host_ar_check.h - CPU model of the kernel's work partition (same index helpers as host_ar.cu).
// For one call shape (n, world, dtype, mode, blocks) it replays every (rank, phase, chunk, unit) the kernel visits and
// checks:
//   1. coverage: every output element of every rank is written exactly once (phase 2 for the own chunk, phase 3 for
//      the others) and nothing outside [0, n) is written to device memory;
//   2. the per-block flag invariant: an element read in phase 2 (rank r, block b, from slot j) was staged in phase 1 by
//      rank j with the SAME block index b, and an element read in phase 3 from the reduced area was written in phase 2
//      by its owner with the same block b (so waiting on flags[phase][j][b] covers exactly the data the block reads);
//   3. bounds: every staging byte written or read lies inside slot_need(n) <= slot_bytes_for(n * esz);
//   4. INT8 quads: the 4 lanes of a quad visit the same number of units, the units of a quad belong to one scale group,
//      and the scale is written by exactly one lane per group;
//   5. the error NaN fill (kernel finish) writes every element exactly once, from the block that owns it.
// Returns {ok, fail_code, fail_detail, elements_checked}. fail codes: 1 coverage, 2 phase-1->2 block mismatch,
// 3 phase-2->3 block mismatch, 4 bounds, 5 quad, 6 scale writer, 7 bad args, 8 NaN-fill coverage/ownership.
#pragma once
#include <stdint.h>

#include <vector>

#include "host_ar_layout.h"

namespace p5 {

inline std::vector<int64_t> cover_check(int64_t n, int world, int dt, int mode, int blocks, int threads = kThreads) {
  auto fail = [&](int64_t code, int64_t detail, int64_t checked) {
    return std::vector<int64_t>{0, code, detail, checked};
  };
  if (n < 0 || world < 2 || world > kMaxWorld || blocks < 1 || blocks > kMaxBlocks || threads % 4 != 0 ||
      (dt != F32 && dt != BF16 && dt != F16) || (mode != NATIVE && mode != INT8))
    return fail(7, 0, 0);
  const int64_t need = slot_need(n, dt, mode);
  const int64_t slot = slot_bytes_for(n * dtype_size(dt));
  if (need > slot) return fail(4, need, 0);
  // stage_block[j][e]: block of rank j that staged element e (-1 none); red_block[e]: block that wrote reduced e.
  std::vector<std::vector<int16_t>> stage_block(world, std::vector<int16_t>(n > 0 ? n : 1, -1));
  std::vector<int16_t> red_block(n > 0 ? n : 1, -1);
  std::vector<int8_t> scale_writes((size_t)ceil_div(n, kGroup) + 1, 0);
  int64_t checked = 0;
  const int64_t UM = chunk_units4(chunk_end(n, world, 0));  // longest chunk (chunk 0)
  if (UM % 4 != 0) return fail(5, UM, 0);
  for (int j = 1; j < world; ++j)
    if (chunk_end(n, world, j) - chunk_begin(n, world, j) > chunk_end(n, world, 0)) return fail(5, j, 0);

  auto quad_ok = [&](int64_t cb, int64_t ce) -> bool {
    if (ce == cb) return true;  // empty chunk (tiny n): the kernel runs no iteration, begin may be unaligned (= n)
    const int64_t U = chunk_units4(ce - cb);
    if (U % 4 != 0) return false;
    // units 4q..4q+3 are consecutive lanes of one quad (stride % 4 == 0, threads % 4 == 0) and start one group
    for (int64_t u = 0; u < U; u += 4) {
      const int64_t e0 = cb + u * kUnit;
      if (e0 % kGroup != 0) return false;
      if (unit_block(u, blocks, threads) != unit_block(u + 3, blocks, threads)) return false;
    }
    return cb % kGroup == 0;
  };

  // phase 1 on every rank
  for (int r = 0; r < world; ++r) {
    for (int jj = 1; jj < world; ++jj) {
      const int j = (r + jj) % world;
      const int64_t cb = chunk_begin(n, world, j), ce = chunk_end(n, world, j);
      if (mode == INT8 && !quad_ok(cb, ce)) return fail(5, j, checked);
      const int64_t U = UM;  // kernel phase 1 walks u < UM for every foreign chunk (c <= 0 past a chunk's end)
      for (int64_t u = 0; u < U; ++u) {
        const int64_t e0 = cb + u * kUnit;
        const int c = unit_count(e0, ce);
        if (c <= 0) continue;
        const int b = unit_block(u, blocks, threads);
        for (int i = 0; i < c; ++i) {
          const int64_t e = e0 + i;
          if (e < 0 || e >= n) return fail(1, e, checked);
          if (stage_block[r][e] != -1) return fail(1, e, checked);
          stage_block[r][e] = (int16_t)b;
        }
        if (mode == INT8) {
          if (q_payload_off(e0) + 8 > q_scale_base(n)) return fail(4, e0, checked);
          if (q_scale_off(n, e0) + 4 > need) return fail(4, e0, checked);
          if (u % 4 == 0) {
            if (scale_writes[e0 / kGroup] >= world) return fail(6, e0, checked);
            ++scale_writes[e0 / kGroup];
          }
        } else if (native_off(e0 + c, dt) > need) {
          return fail(4, e0, checked);
        }
      }
    }
  }
  // INT8: each group's scale is staged by each of the (world - 1) non-owner ranks exactly once
  if (mode == INT8) {
    for (int64_t g = 0; g * kGroup < n; ++g) {
      if (scale_writes[g] != world - 1) return fail(6, g, checked);
    }
  }
  // phase 2 (owner reduce) on every rank, then phase 3 (gather)
  std::vector<int8_t> out_cnt(n > 0 ? n : 1, 0);
  for (int r = 0; r < world; ++r) {
    for (int64_t e = 0; e < n; ++e) out_cnt[e] = 0;
    const int64_t cb = chunk_begin(n, world, r), ce = chunk_end(n, world, r);
    if (mode == INT8 && !quad_ok(cb, ce)) return fail(5, r, checked);
    const int64_t U = chunk_units4(ce - cb);
    for (int64_t u = 0; u < U; ++u) {
      const int64_t e0 = cb + u * kUnit;
      const int c = unit_count(e0, ce);
      if (c <= 0) continue;
      const int b = unit_block(u, blocks, threads);
      for (int i = 0; i < c; ++i) {
        const int64_t e = e0 + i;
        for (int j = 0; j < world; ++j) {
          if (j == r) continue;
          if (stage_block[j][e] != b) return fail(2, e, checked);
        }
        if (red_block[e] != -1 && red_block[e] != b) return fail(3, e, checked);
        red_block[e] = (int16_t)b;
        ++out_cnt[e];
      }
    }
    for (int jj = 1; jj < world; ++jj) {
      const int j = (r + jj) % world;
      const int64_t cbj = chunk_begin(n, world, j), cej = chunk_end(n, world, j);
      const int64_t Uj = UM;  // kernel phase 3 walks u < UM as well
      for (int64_t u = 0; u < Uj; ++u) {
        const int64_t e0 = cbj + u * kUnit;
        const int c = unit_count(e0, cej);
        if (c <= 0) continue;
        const int b = unit_block(u, blocks, threads);
        for (int i = 0; i < c; ++i) {
          const int64_t e = e0 + i;
          // the reduced element must already have been written by owner j with the same block
          // (owners are processed in rank order here, so for j > r check it lazily below)
          if (j < r && red_block[e] != b) return fail(3, e, checked);
          ++out_cnt[e];
        }
      }
    }
    for (int64_t e = 0; e < n; ++e) {
      if (out_cnt[e] != 1) return fail(1, e, checked);
      ++checked;
    }
  }
  // lazy part of check 3: every reduced element was written by its owner; phase 3 of ANY rank reads element e with
  // block unit_block(u_rel(e)), which is the same function the owner used, so equality of red_block with that function
  for (int j = 0; j < world; ++j) {
    const int64_t cb = chunk_begin(n, world, j), ce = chunk_end(n, world, j);
    for (int64_t e = cb; e < ce; ++e) {
      const int64_t u = (e - cb) / kUnit;
      if (red_block[e] != unit_block(u, blocks, threads)) return fail(3, e, checked);
    }
  }
  // 5. error NaN fill (kernel finish): block b walks u < UM over every chunk; each element of out must be filled by
  //    exactly one block, and that block must be the one that owns the element in phase 2 / 3 (no cross-block race).
  std::vector<int8_t> fill_cnt(n > 0 ? n : 1, 0);
  for (int j = 0; j < world; ++j) {
    const int64_t cb = chunk_begin(n, world, j), ce = chunk_end(n, world, j);
    for (int64_t u = 0; u < UM; ++u) {
      const int64_t e0 = cb + u * kUnit;
      const int c = unit_count(e0, ce);
      if (c <= 0) continue;
      const int b = unit_block(u, blocks, threads);
      for (int i = 0; i < c; ++i) {
        const int64_t e = e0 + i;
        if (e < 0 || e >= n || fill_cnt[e] != 0) return fail(8, e, checked);
        if (red_block[e] != b) return fail(8, e, checked);
        fill_cnt[e] = 1;
      }
    }
  }
  for (int64_t e = 0; e < n; ++e)
    if (fill_cnt[e] != 1) return fail(8, e, checked);
  return std::vector<int64_t>{1, 0, 0, checked};
}

}  // namespace p5
