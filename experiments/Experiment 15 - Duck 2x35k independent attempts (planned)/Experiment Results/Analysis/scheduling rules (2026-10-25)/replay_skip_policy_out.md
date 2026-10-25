| replayed run | tasks | P0 credit (now) | attempt 2s skipped | tokens saved | credit lost by skipping | attempt 3s bought | **P1 credit** (vs P0) | P2 = P0 + same tokens for attempt 3 (vs P0) |
|---|---|---|---|---|---|---|---|---|
| E15B_45K30K (b0, b1 = fresh) | 20 | 13.00 | 14 | 316,328 | 0.33 (1ae2feb7) | 7.9 | **13.34** (+0.34) | 13.71 (+0.71) |
| E15B_40K35K (b0, b1 = fresh) | 20 | 7.50 | 10 | 226,884 | 0.00 (-) | 5.9 | **9.60** (+2.10) | 9.53 (+2.03) |
| EVAL120: a1 = G4B_XHIGH_E16 b0, a2 = G4B_XHIGH_noE16 b0 | 120 | 46.50 | 46 | 1,705,261 | 1.50 (6e453dd6, 7ed72f31) | 22.6 | **46.00** (-0.50) | 47.50 (+1.00) |
| EVAL120: a1 = G4B_XHIGH_noE16 b0, a2 = G4B_XHIGH_E16 b0 | 120 | 46.50 | 46 | 1,704,173 | 5.00 (221dfab4, 31f7f899, 58490d8a, 5961cc34, 80a900e0) | 26.5 | **42.50** (-4.00) | 47.50 (+1.00) |

Attempt 3 = an unused independent xhigh attempt on the same task (G4B XHIGH b0 of either run, E15B b0/b1 of either run). For EVAL120 only the 20 E15B tasks have such attempts left, so P1/P2 there can buy attempt 3 only on those 20 tasks (a lower bound for the other 100).
