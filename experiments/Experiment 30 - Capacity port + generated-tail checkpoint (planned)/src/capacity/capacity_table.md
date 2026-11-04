| Arm | groups | stride PP0 / PP1 (B) | pool (usable) | F adm | F steady | F no-retain | SW adm | mc@131k s24 |
|---|---|---|---|---|---|---|---|---|
| Q (MTP K3) baseline | 5 | 14,680,064 / 16,515,072 | 480 (479) | 6 | 6 | 3 | - | 12.61 |
| Q + P1 mamba pad->state | 5 | 14,155,776 / 16,515,072 | 480 (479) | 6 | 6 | 3 | - | 12.61 |
| Q + P1 + P2 mamba 2 groups | 4 | 17,329,920 / 16,515,072 | 457 (456) | 4 | 4 | 2 | - | 12.47 |
| Q + idxfp8 (known lever, record) | 5 | 14,680,064 / 15,597,568 | 508 (507) | 6 | 6 | 3 | - | 13.34 |
| Q + idxfp8 + P1 | 5 | 13,369,344 / 15,597,568 | 508 (507) | 6 | 6 | 3 | - | 13.34 |
| D4s baseline (SW1024 bf16) | 6 | 14,680,064 / 14,155,776 | 540 (539) | 12 | 10 | 7 | 6 | 12.25 |
| D3s baseline (SW1024 bf16) | 6 | 14,680,064 / 14,155,776 | 540 (539) | 12 | 10 | 7 | 6 | 12.25 |
| D4s + P1 mamba pad->state | 6 | 14,155,776 / 14,155,776 | 560 (559) | 12 | 10 | 7 | 6 | 12.70 |
| D4s + P1 + P2 mamba 2 groups | 5 | 17,679,040 / 15,911,136 | 448 (447) | 11 | 8 | 6 | 6 | 10.42 |
| D4s + P1 + P2 mamba 4 groups | 7 | 14,155,776 / 14,155,776 | 560 (559) | 14 | 12 | 8 | 6 | 12.15 |
| D4s P3 SW2048 fp8 (TRITON) | 6 | 14,680,064 / 14,155,776 | 540 (539) | 10 | 9 | 6 | 4 | 12.83 |
| D4s P3 SW2048 fp8 + P1 | 6 | 14,155,776 / 14,155,776 | 560 (559) | 10 | 9 | 6 | 4 | 13.31 |
| D3 SW1024 fp8 | 6 | 14,680,064 / 14,155,776 | 540 (539) | 12 | 10 | 7 | 6 | 12.25 |
| D3 SW2048 bf16 | 6 | 14,680,064 / 20,971,520 | 378 (377) | 10 | 9 | 6 | 4 | 8.98 |
| D3 SW2048 fp8 | 6 | 14,680,064 / 14,155,776 | 540 (539) | 10 | 9 | 6 | 4 | 12.83 |
| D3 SW2048 fp8 + P1 | 6 | 14,155,776 / 14,155,776 | 560 (559) | 10 | 9 | 6 | 4 | 13.31 |
| D3 SW1024 bf16 + P1 | 6 | 14,155,776 / 14,155,776 | 560 (559) | 12 | 10 | 7 | 6 | 12.70 |
| Q layer-outer LBHNC (uniform-page planner, #58207 path) | 76 | 2,097,152 / 2,097,152 | 3783 (3782) | 87 | 74 | 37 | - | 6.82 |
| D3s layer-outer LBHNC (uniform-page planner, #58207 path) | 78 | 2,097,152 / 2,097,152 | 3783 (3782) | 116 | 78 | 41 | 6 | 6.90 |
| D4s SW2048 bf16 (widens, record) | 6 | 14,680,064 / 20,971,520 | 378 (377) | 10 | 9 | 6 | 4 | 8.98 |
| D4s SW512 bf16 + P1 | 6 | 14,155,776 / 14,155,776 | 560 (559) | 17 | 12 | 9 | 11 | 11.41 |
| D4s + P1 + idxfp8 | 6 | 13,369,344 / 13,369,344 | 593 (592) | 12 | 10 | 7 | 6 | 13.45 |
| D4s + P1 + P3 fp8 2048 + idxfp8 | 6 | 13,369,344 / 13,369,344 | 593 (592) | 10 | 9 | 6 | 4 | 14.10 |

Residents (F_admission) = min(seqs, (usable - ring) // (ceil(L / 4096) + F)); columns 40k / 64k / 90k / 116k

| Arm | seqs 24 | seqs 28 | seqs 32 |
|---|---|---|---|
| Q (MTP K3) baseline | 24 / 21 / 17 / 13 | 28 / 21 / 17 / 13 | 29 / 21 / 17 / 13 |
| Q + P1 mamba pad->state | 24 / 21 / 17 / 13 | 28 / 21 / 17 / 13 | 29 / 21 / 17 / 13 |
| Q + P1 + P2 mamba 2 groups | 24 / 22 / 17 / 13 | 28 / 22 / 17 / 13 | 31 / 22 / 17 / 13 |
| Q + idxfp8 (known lever, record) | 24 / 23 / 18 / 14 | 28 / 23 / 18 / 14 | 31 / 23 / 18 / 14 |
| Q + idxfp8 + P1 | 24 / 23 / 18 / 14 | 28 / 23 / 18 / 14 | 31 / 23 / 18 / 14 |
| D4s baseline (SW1024 bf16) | 24 / 19 / 15 / 13 | 24 / 19 / 15 / 13 | 24 / 19 / 15 / 13 |
| D3s baseline (SW1024 bf16) | 24 / 19 / 15 / 13 | 24 / 19 / 15 / 13 | 24 / 19 / 15 / 13 |
| D4s + P1 mamba pad->state | 24 / 19 / 16 / 13 | 25 / 19 / 16 / 13 | 25 / 19 / 16 / 13 |
| D4s + P1 + P2 mamba 2 groups | 21 / 16 / 13 / 11 | 21 / 16 / 13 / 11 | 21 / 16 / 13 / 11 |
| D4s + P1 + P2 mamba 4 groups | 23 / 18 / 15 / 12 | 23 / 18 / 15 / 12 | 23 / 18 / 15 / 12 |
| D4s P3 SW2048 fp8 (TRITON) | 24 / 20 / 16 / 13 | 26 / 20 / 16 / 13 | 26 / 20 / 16 / 13 |
| D4s P3 SW2048 fp8 + P1 | 24 / 21 / 17 / 14 | 27 / 21 / 17 / 14 | 27 / 21 / 17 / 14 |
| D3 SW1024 fp8 | 24 / 19 / 15 / 13 | 24 / 19 / 15 / 13 | 24 / 19 / 15 / 13 |
| D3 SW2048 bf16 | 18 / 14 / 11 / 9 | 18 / 14 / 11 / 9 | 18 / 14 / 11 / 9 |
| D3 SW2048 fp8 | 24 / 20 / 16 / 13 | 26 / 20 / 16 / 13 | 26 / 20 / 16 / 13 |
| D3 SW2048 fp8 + P1 | 24 / 21 / 17 / 14 | 27 / 21 / 17 / 14 | 27 / 21 / 17 / 14 |
| D3 SW1024 bf16 + P1 | 24 / 19 / 16 / 13 | 25 / 19 / 16 / 13 | 25 / 19 / 16 / 13 |
| Q layer-outer LBHNC (uniform-page planner, #58207 path) | 24 / 24 / 24 / 24 | 28 / 28 / 28 / 28 | 32 / 32 / 32 / 32 |
| D3s layer-outer LBHNC (uniform-page planner, #58207 path) | 24 / 24 / 24 / 24 | 28 / 28 / 27 / 26 | 30 / 28 / 27 / 26 |
| D4s SW2048 bf16 (widens, record) | 18 / 14 / 11 / 9 | 18 / 14 / 11 / 9 | 18 / 14 / 11 / 9 |
| D4s SW512 bf16 + P1 | 20 / 16 / 14 / 12 | 20 / 16 / 14 / 12 | 20 / 16 / 14 / 12 |
| D4s + P1 + idxfp8 | 24 / 21 / 17 / 14 | 26 / 21 / 17 / 14 | 26 / 21 / 17 / 14 |
| D4s + P1 + P3 fp8 2048 + idxfp8 | 24 / 22 / 18 / 15 | 28 / 22 / 18 / 15 | 29 / 22 / 18 / 15 |

Residents (F_steady) = min(seqs, (usable - ring) // (ceil(L / 4096) + F)); columns 40k / 64k / 90k / 116k

| Arm | seqs 24 | seqs 28 | seqs 32 |
|---|---|---|---|
| Q (MTP K3) baseline | 24 / 21 / 17 / 13 | 28 / 21 / 17 / 13 | 29 / 21 / 17 / 13 |
| Q + P1 mamba pad->state | 24 / 21 / 17 / 13 | 28 / 21 / 17 / 13 | 29 / 21 / 17 / 13 |
| Q + P1 + P2 mamba 2 groups | 24 / 22 / 17 / 13 | 28 / 22 / 17 / 13 | 31 / 22 / 17 / 13 |
| Q + idxfp8 (known lever, record) | 24 / 23 / 18 / 14 | 28 / 23 / 18 / 14 | 31 / 23 / 18 / 14 |
| Q + idxfp8 + P1 | 24 / 23 / 18 / 14 | 28 / 23 / 18 / 14 | 31 / 23 / 18 / 14 |
| D4s baseline (SW1024 bf16) | 24 / 20 / 16 / 13 | 26 / 20 / 16 / 13 | 26 / 20 / 16 / 13 |
| D3s baseline (SW1024 bf16) | 24 / 20 / 16 / 13 | 26 / 20 / 16 / 13 | 26 / 20 / 16 / 13 |
| D4s + P1 mamba pad->state | 24 / 21 / 17 / 14 | 27 / 21 / 17 / 14 | 27 / 21 / 17 / 14 |
| D4s + P1 + P2 mamba 2 groups | 24 / 18 / 14 / 12 | 24 / 18 / 14 / 12 | 24 / 18 / 14 / 12 |
| D4s + P1 + P2 mamba 4 groups | 24 / 19 / 16 / 13 | 25 / 19 / 16 / 13 | 25 / 19 / 16 / 13 |
| D4s P3 SW2048 fp8 (TRITON) | 24 / 21 / 17 / 14 | 28 / 21 / 17 / 14 | 28 / 21 / 17 / 14 |
| D4s P3 SW2048 fp8 + P1 | 24 / 22 / 18 / 14 | 28 / 22 / 18 / 14 | 29 / 22 / 18 / 14 |
| D3 SW1024 fp8 | 24 / 20 / 16 / 13 | 26 / 20 / 16 / 13 | 26 / 20 / 16 / 13 |
| D3 SW2048 bf16 | 19 / 15 / 12 / 9 | 19 / 15 / 12 / 9 | 19 / 15 / 12 / 9 |
| D3 SW2048 fp8 | 24 / 21 / 17 / 14 | 28 / 21 / 17 / 14 | 28 / 21 / 17 / 14 |
| D3 SW2048 fp8 + P1 | 24 / 22 / 18 / 14 | 28 / 22 / 18 / 14 | 29 / 22 / 18 / 14 |
| D3 SW1024 bf16 + P1 | 24 / 21 / 17 / 14 | 27 / 21 / 17 / 14 | 27 / 21 / 17 / 14 |
| Q layer-outer LBHNC (uniform-page planner, #58207 path) | 24 / 24 / 24 / 24 | 28 / 28 / 28 / 28 | 32 / 32 / 32 / 32 |
| D3s layer-outer LBHNC (uniform-page planner, #58207 path) | 24 / 24 / 24 / 24 | 28 / 28 / 28 / 28 | 32 / 32 / 32 / 32 |
| D4s SW2048 bf16 (widens, record) | 19 / 15 / 12 / 9 | 19 / 15 / 12 / 9 | 19 / 15 / 12 / 9 |
| D4s SW512 bf16 + P1 | 24 / 19 / 16 / 13 | 25 / 19 / 16 / 13 | 25 / 19 / 16 / 13 |
| D4s + P1 + idxfp8 | 24 / 22 / 18 / 15 | 28 / 22 / 18 / 15 | 29 / 22 / 18 / 15 |
| D4s + P1 + P3 fp8 2048 + idxfp8 | 24 / 23 / 19 / 15 | 28 / 23 / 19 / 15 | 31 / 23 / 19 / 15 |

Residents (F_steady_P4_no_retain) = min(seqs, (usable - ring) // (ceil(L / 4096) + F)); columns 40k / 64k / 90k / 116k

| Arm | seqs 24 | seqs 28 | seqs 32 |
|---|---|---|---|
| Q (MTP K3) baseline | 24 / 24 / 19 / 14 | 28 / 25 / 19 / 14 | 32 / 25 / 19 / 14 |
| Q + P1 mamba pad->state | 24 / 24 / 19 / 14 | 28 / 25 / 19 / 14 | 32 / 25 / 19 / 14 |
| Q + P1 + P2 mamba 2 groups | 24 / 24 / 18 / 14 | 28 / 24 / 18 / 14 | 32 / 24 / 18 / 14 |
| Q + idxfp8 (known lever, record) | 24 / 24 / 20 / 15 | 28 / 26 / 20 / 15 | 32 / 26 / 20 / 15 |
| Q + idxfp8 + P1 | 24 / 24 / 20 / 15 | 28 / 26 / 20 / 15 | 32 / 26 / 20 / 15 |
| D4s baseline (SW1024 bf16) | 24 / 23 / 18 / 14 | 28 / 23 / 18 / 14 | 31 / 23 / 18 / 14 |
| D3s baseline (SW1024 bf16) | 24 / 23 / 18 / 14 | 28 / 23 / 18 / 14 | 31 / 23 / 18 / 14 |
| D4s + P1 mamba pad->state | 24 / 24 / 19 / 15 | 28 / 24 / 19 / 15 | 32 / 24 / 19 / 15 |
| D4s + P1 + P2 mamba 2 groups | 24 / 20 / 15 / 12 | 27 / 20 / 15 / 12 | 27 / 20 / 15 / 12 |
| D4s + P1 + P2 mamba 4 groups | 24 / 23 / 18 / 15 | 28 / 23 / 18 / 15 | 31 / 23 / 18 / 15 |
| D4s P3 SW2048 fp8 (TRITON) | 24 / 24 / 19 / 15 | 28 / 24 / 19 / 15 | 32 / 24 / 19 / 15 |
| D4s P3 SW2048 fp8 + P1 | 24 / 24 / 19 / 15 | 28 / 25 / 19 / 15 | 32 / 25 / 19 / 15 |
| D3 SW1024 fp8 | 24 / 23 / 18 / 14 | 28 / 23 / 18 / 14 | 31 / 23 / 18 / 14 |
| D3 SW2048 bf16 | 23 / 17 / 13 / 10 | 23 / 17 / 13 / 10 | 23 / 17 / 13 / 10 |
| D3 SW2048 fp8 | 24 / 24 / 19 / 15 | 28 / 24 / 19 / 15 | 32 / 24 / 19 / 15 |
| D3 SW2048 fp8 + P1 | 24 / 24 / 19 / 15 | 28 / 25 / 19 / 15 | 32 / 25 / 19 / 15 |
| D3 SW1024 bf16 + P1 | 24 / 24 / 19 / 15 | 28 / 24 / 19 / 15 | 32 / 24 / 19 / 15 |
| Q layer-outer LBHNC (uniform-page planner, #58207 path) | 24 / 24 / 24 / 24 | 28 / 28 / 28 / 28 | 32 / 32 / 32 / 32 |
| D3s layer-outer LBHNC (uniform-page planner, #58207 path) | 24 / 24 / 24 / 24 | 28 / 28 / 28 / 28 | 32 / 32 / 32 / 32 |
| D4s SW2048 bf16 (widens, record) | 23 / 17 / 13 / 10 | 23 / 17 / 13 / 10 | 23 / 17 / 13 / 10 |
| D4s SW512 bf16 + P1 | 24 / 22 / 18 / 14 | 28 / 22 / 18 / 14 | 29 / 22 / 18 / 14 |
| D4s + P1 + idxfp8 | 24 / 24 / 20 / 16 | 28 / 25 / 20 / 16 | 32 / 25 / 20 / 16 |
| D4s + P1 + P3 fp8 2048 + idxfp8 | 24 / 24 / 21 / 16 | 28 / 26 / 21 / 16 | 32 / 26 / 21 / 16 |

Real KVCacheManager holding (sim): fixed blocks per request after 200 decode steps; pool used per request

| Arm | P=40k | P=64k | P=90k |
|---|---|---|---|
| Q (MTP K3) baseline | fixed 6, ctx 10, used/req 16.25 | fixed 6, ctx 16, used/req 22.25 | fixed 6, ctx 23, used/req 29.50 |
| Q + P1 mamba pad->state | fixed 6, ctx 10, used/req 16.25 | fixed 6, ctx 16, used/req 22.25 | fixed 6, ctx 23, used/req 29.50 |
| Q + P1 + P2 mamba 2 groups | fixed 4, ctx 10, used/req 14.50 | fixed 4, ctx 16, used/req 20.50 | fixed 4, ctx 23, used/req 27.50 |
| Q + idxfp8 (known lever, record) | fixed 6, ctx 10, used/req 16.25 | fixed 6, ctx 16, used/req 22.25 | fixed 6, ctx 23, used/req 29.50 |
| Q + idxfp8 + P1 | fixed 6, ctx 10, used/req 16.25 | fixed 6, ctx 16, used/req 22.25 | fixed 6, ctx 23, used/req 29.50 |
| D4s baseline (SW1024 bf16) | fixed 9, ctx 10, used/req 19.25 | fixed 9, ctx 16, used/req 25.25 | fixed 9, ctx 23, used/req 32.50 |
| D3s baseline (SW1024 bf16) | fixed 9, ctx 10, used/req 19.25 | fixed 9, ctx 16, used/req 25.25 | fixed 9, ctx 23, used/req 32.50 |
| D4s + P1 mamba pad->state | fixed 9, ctx 10, used/req 19.25 | fixed 9, ctx 16, used/req 25.25 | fixed 9, ctx 23, used/req 32.50 |
| D4s + P1 + P2 mamba 2 groups | fixed 7, ctx 10, used/req 18.00 | fixed 7, ctx 16, used/req 24.00 | fixed 7, ctx 23, used/req 31.00 |
| D4s + P1 + P2 mamba 4 groups | fixed 11, ctx 10, used/req 21.25 | fixed 11, ctx 16, used/req 27.25 | fixed 11, ctx 23, used/req 34.50 |
| D4s P3 SW2048 fp8 (TRITON) | fixed 8, ctx 10, used/req 18.25 | fixed 8, ctx 16, used/req 24.25 | fixed 8, ctx 23, used/req 31.50 |
| D4s P3 SW2048 fp8 + P1 | fixed 8, ctx 10, used/req 18.25 | fixed 8, ctx 16, used/req 24.25 | fixed 8, ctx 23, used/req 31.50 |
| D3 SW1024 fp8 | fixed 9, ctx 10, used/req 19.25 | fixed 9, ctx 16, used/req 25.25 | fixed 9, ctx 23, used/req 32.50 |
| D3 SW2048 bf16 | fixed 8, ctx 10, used/req 18.25 | fixed 8, ctx 16, used/req 24.25 | fixed 8, ctx 23, used/req 31.50 |
| D3 SW2048 fp8 | fixed 8, ctx 10, used/req 18.25 | fixed 8, ctx 16, used/req 24.25 | fixed 8, ctx 23, used/req 31.50 |
| D3 SW2048 fp8 + P1 | fixed 8, ctx 10, used/req 18.25 | fixed 8, ctx 16, used/req 24.25 | fixed 8, ctx 23, used/req 31.50 |
| D3 SW1024 bf16 + P1 | fixed 9, ctx 10, used/req 19.25 | fixed 9, ctx 16, used/req 25.25 | fixed 9, ctx 23, used/req 32.50 |
| Q layer-outer LBHNC (uniform-page planner, #58207 path) | fixed 74, ctx 10, used/req 243.00 | fixed 74, ctx 16, used/req 321.00 | fixed 74, ctx 23, used/req 425.00 |
| D3s layer-outer LBHNC (uniform-page planner, #58207 path) | fixed 89, ctx 10, used/req 245.00 | fixed 89, ctx 16, used/req 317.00 | fixed 89, ctx 23, used/req 413.00 |
| D4s SW2048 bf16 (widens, record) | fixed 8, ctx 10, used/req 18.25 | fixed 8, ctx 16, used/req 24.25 | fixed 8, ctx 23, used/req 31.50 |
| D4s SW512 bf16 + P1 | fixed 11, ctx 10, used/req 21.25 | fixed 11, ctx 16, used/req 27.25 | fixed 11, ctx 23, used/req 34.50 |
| D4s + P1 + idxfp8 | fixed 9, ctx 10, used/req 19.25 | fixed 9, ctx 16, used/req 25.25 | fixed 9, ctx 23, used/req 32.50 |
| D4s + P1 + P3 fp8 2048 + idxfp8 | fixed 8, ctx 10, used/req 18.25 | fixed 8, ctx 16, used/req 24.25 | fixed 8, ctx 23, used/req 31.50 |

Pressure (#62752 port) D4s baseline (SW1024 bf16): {"fcfs": {"preemptions": 0, "recompute_tokens": 0, "admit_fail": 0, "min_free": 11, "burst_done_step": 323, "stall_steps": 0, "decode_tokens": 143700, "running_end": 24, "waiting_end": 0, "pool": 540, "mamba_groups": 3}, "headroom": {"preemptions": 0, "recompute_tokens": 0, "admit_fail": 0, "min_free": 11, "burst_done_step": 323, "stall_steps": 0, "decode_tokens": 143700, "running_end": 24, "waiting_end": 0, "pool": 540, "mamba_groups": 3}}

SW mask test (D4s baseline (SW1024 bf16)): {"group": 5, "block": 1024, "reach": 2047, "steps": 624, "max_stale": 12, "unmasked_stale": 0, "masked_live_in_window": 0, "stale_ids_stream1": 7, "stale_ids_now_owned_by_stream2": 0, "pass": true}

Pressure (#62752 port) D4s + P1 mamba pad->state: {"fcfs": {"preemptions": 0, "recompute_tokens": 0, "admit_fail": 0, "min_free": 31, "burst_done_step": 323, "stall_steps": 0, "decode_tokens": 143700, "running_end": 24, "waiting_end": 0, "pool": 560, "mamba_groups": 3}, "headroom": {"preemptions": 0, "recompute_tokens": 0, "admit_fail": 0, "min_free": 31, "burst_done_step": 323, "stall_steps": 0, "decode_tokens": 143700, "running_end": 24, "waiting_end": 0, "pool": 560, "mamba_groups": 3}}

Pressure (#62752 port) D4s P3 SW2048 fp8 + P1: {"fcfs": {"preemptions": 0, "recompute_tokens": 0, "admit_fail": 0, "min_free": 55, "burst_done_step": 323, "stall_steps": 0, "decode_tokens": 143700, "running_end": 24, "waiting_end": 0, "pool": 560, "mamba_groups": 3}, "headroom": {"preemptions": 0, "recompute_tokens": 0, "admit_fail": 0, "min_free": 55, "burst_done_step": 323, "stall_steps": 0, "decode_tokens": 143700, "running_end": 24, "waiting_end": 0, "pool": 560, "mamba_groups": 3}}

