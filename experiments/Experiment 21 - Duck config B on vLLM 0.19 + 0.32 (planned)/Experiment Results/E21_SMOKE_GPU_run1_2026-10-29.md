# E21 GPU smoke run 1 (2026-10-29, foysal v1, rev 2h)

Profile E21SMOKE: 12 eval tasks (22 queries), 20-minute solve per half. Both kernels COMPLETE, no errors.
Raw outputs are in `SMOKE_GPU_run1_2026-10-29/nb1` and `/nb2`.

| | NB1 A | NB1 B | NB2 A | NB2 B |
|---|---|---|---|---|
| engine | vLLM 0.19 | vLLM 0.32 | vLLM 0.19 | vLLM 0.32 |
| config used | **fallback FP8 s16** (INT4 smoke failed) | FP8 s16 (B follows A) | primary FP8 SSM fp16 s16 | primary FP8 SSM fp16 s16 |
| ready (s) | INT4 365 -> garbage; FP8 250 | 535 | 435 | 560 |
| KV blocks x block size | 855 x 1600 | 865 x 1600 | 1627 x 832 | 1663 x 832 |
| streams fit at 57k / 65k / 79k | 16 / 15 / 13 | 16 / 15 / 13 | 19 / 17 / 14 | 19 / 17 / 15 |
| real running mean | 11.68 | 11.65 | 11.38 | 11.70 |
| gen tok/s p50 / mean | 301.5 / 269.7 | 305.5 / 259.3 | 303.9 / 291.8 | 328.8 / 283.2 |
| MTP accepted per draft (of 3) | 1.89 | 1.92 | 1.94 | 1.94 |
| prefix hit (server counters) | 69.0 % | 54.2 % | 82.2 % | 55.1 % |
| server gen tokens | 329,738 | 316,407 | 357,791 | 345,483 |
| preemptions | 0 | 0 | 0 | 0 |
| contract probe / cache probe | OK / equal | OK / equal | OK / equal | OK / equal |
| stalls / relaunches / length_restored | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| solve overrun (s) | 0.1 | 1.1 | 0.1 | 0.1 |
| query hits / 22 (task-norm) | 0 (0 %) | 0 (0 %) | **3 (12.5 %)**: 247ef758 q0, 4c3d4a41 q0+q1 | 0 (0 %) |
| config_status / comparison_valid | FALLBACK_BOTH / True | | PRIMARY_BOTH / True | |

## Findings

1. **RedHat INT4 (w4a16-g128, Marlin) on vLLM 0.19 + MTP3 generates garbage.**
   - The output was multilingual token soup ("ogan_plural_METADATAipes...").
   - The rev 2g content smoke caught it, and the FP8 s16 fallback ran.
   - Because of E21_B_FOLLOWS_A, half B also ran FP8 s16, so INT4 on vLLM 0.32 was never run inside Duck.
   - Exp 14 VMS ran INT4 on 0.32 only as a diagnostic, with no oracle, so its correctness there is unknown.
2. **The engines are healthy.**
   - No stall, relaunch, preemption or deadline violation.
   - The contract and cache probes passed on both versions.
   - The rev 2h stall watchdog never fired. The maximum no-progress interval was 10 s.
3. **Speed is equal.** 0.32 is at most about +8 % p50 (NB2), and both versions run 300-330 tok/s at about 11.5 running requests.
4. **vLLM 0.32 has a lower prefix hit rate than 0.19:** 54-55 % against 69-82 %, on the same Duck prompts.
5. **SSM fp16 roughly halves the block size** (832 against 1600): 19 streams fit at 57k instead of 16, and the prefix hit is higher (82 % against 69 % on 0.19).
6. **The scores carry no signal.**
   - At 50K45K, a 20-minute window ends most attempt-1 turns before they finish: 2 of 12 were cut, and attempt 2 started 0 times.
   - Only 12 streams of 16 were used, because there are 12 tasks.
   - NB2 A's 3 hits (2 tasks) against 0 elsewhere is noise.
