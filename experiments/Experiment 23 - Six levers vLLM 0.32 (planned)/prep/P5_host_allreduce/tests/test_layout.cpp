// test_layout.cpp - host-only test of the P5 partition/layout (host_ar_layout.h + host_ar_check.h).
//   g++ -O2 -std=c++17 -I../src test_layout.cpp -o test_layout && ./test_layout
// Prints one line "LAYOUT_JSON {...}" and exits 0 when every case passes.
#include <stdio.h>

#include <string>
#include <vector>

#include "host_ar_check.h"
#include "host_ar_layout.h"

using namespace p5;

int main() {
  int64_t cases = 0, fails = 0, elems = 0;
  std::string first_fail;
  auto run = [&](int64_t n, int world, int dt, int mode, int blocks) {
    std::vector<int64_t> r = cover_check(n, world, dt, mode, blocks);
    ++cases;
    elems += r[3];
    if (r[0] != 1) {
      ++fails;
      if (first_fail.empty()) {
        char buf[200];
        snprintf(buf, sizeof buf, "n=%lld world=%d dt=%d mode=%d blocks=%d code=%lld detail=%lld", (long long)n,
                 world, dt, mode, blocks, (long long)r[1], (long long)r[2]);
        first_fail = buf;
      }
    }
  };
  const int worlds[3] = {2, 4, 8};
  const int blist[6] = {1, 2, 3, 7, 30, 64};
  // every small n (partial units, partial groups, empty chunks for n < world * 32)
  for (int64_t n = 0; n <= 1200; ++n)
    for (int w : worlds)
      for (int dt = 0; dt < 3; ++dt)
        for (int mode = 0; mode < 2; ++mode) run(n, w, dt, mode, (int)(n % 3) + 1);
  // bench sizes (bytes) and odd sizes, all block counts, world 4
  const int64_t sizes[] = {65536, 245760, 491520, 983040, 1966080, 3932160, 983042, 983040 + 96, 65538, 4098, 1000};
  for (int64_t bytes : sizes)
    for (int dt = 0; dt < 3; ++dt) {
      const int64_t esz = dtype_size(dt);
      if (bytes % esz) continue;
      const int64_t n = bytes / esz;
      for (int mode = 0; mode < 2; ++mode) {
        for (int b : blist) run(n, 4, dt, mode, b);
        run(n, 4, dt, mode, auto_blocks(n, 4));
      }
    }
  // capacity: a 4 MiB handle holds every bench size in every dtype and mode
  int cap_fail = 0;
  const int64_t max_bytes = 4 * 1024 * 1024;
  for (int64_t bytes : sizes)
    for (int dt = 0; dt < 3; ++dt)
      for (int mode = 0; mode < 2; ++mode)
        if (bytes % dtype_size(dt) == 0 && slot_need(bytes / dtype_size(dt), dt, mode) > slot_bytes_for(max_bytes))
          ++cap_fail;
  // auto grid for the decode size (bf16 983,040 B): 491,520 elements, 122,880 per rank chunk, 15,360 units -> 30
  const int ab = auto_blocks(983040 / 2, 4);
  const bool ok = fails == 0 && cap_fail == 0 && ab == 30 && total_bytes_for(4, max_bytes) < 32 * 1024 * 1024;
  printf("LAYOUT_JSON {\"ok\": %s, \"cases\": %lld, \"fails\": %lld, \"elements_checked\": %lld, "
         "\"capacity_fail\": %d, \"auto_blocks_983040_bf16\": %d, \"shm_total_bytes_4MiB\": %lld, "
         "\"first_fail\": \"%s\"}\n",
         ok ? "true" : "false", (long long)cases, (long long)fails, (long long)elems, cap_fail, ab,
         (long long)total_bytes_for(4, max_bytes), first_fail.c_str());
  return ok ? 0 : 1;
}
