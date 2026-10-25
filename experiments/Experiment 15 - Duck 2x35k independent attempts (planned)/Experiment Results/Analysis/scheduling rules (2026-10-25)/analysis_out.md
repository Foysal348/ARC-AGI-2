
### A1. Cross-run: attempt 1 = b0 of one XHIGH EVAL120 run, fresh attempt = b0 of the other (120 tasks x 2 directions)
| class of attempt 1 | pairs | attempt 1 credit | fresh attempt credit | **gain** | fresh tokens | gain / 100k tok |
|---|---|---|---|---|---|---|
| 0 partial progress, no accept | 13 | 0.000 | 0.077 | **0.077** | 45,385 | 0.169 |
| 1 nothing | 135 | 0.000 | 0.137 | **0.137** | 49,322 | 0.278 |
| 2 one accepted answer | 84 | 0.726 | 0.524 | **0.065** | 37,272 | 0.176 |
| 3 two+ distinct accepted | 8 | 0.750 | 0.438 | **0.125** | 34,826 | 0.359 |
| &nbsp;&nbsp;2a flagged | 13 | 0.692 | 0.462 | 0.000 | 39,329 | |
| &nbsp;&nbsp;2b not flagged | 71 | 0.732 | 0.535 | 0.077 | 36,895 | |
| &nbsp;&nbsp;2/3 accepted answer wrong | 23 | 0.000 | 0.261 | 0.261 | 39,076 | |
| &nbsp;&nbsp;2/3 accepted answer right | 65 | 1.000 | 0.623 | 0.000 | 35,504 | |

### A2. Within run, E15B (b0 -> fresh b1, facts view; 45K30K + 40K35K, 20 tasks x 2)
| class of attempt 1 | pairs | attempt 1 credit | fresh attempt credit | **gain** | fresh tokens | gain / 100k tok |
|---|---|---|---|---|---|---|
| 0 partial progress, no accept | 6 | 0.000 | 0.083 | **0.083** | 33,019 | 0.252 |
| 1 nothing | 10 | 0.000 | 0.200 | **0.200** | 31,608 | 0.633 |
| 2 one accepted answer | 22 | 0.758 | 0.568 | **0.015** | 21,789 | 0.070 |
| 3 two+ distinct accepted | 2 | 0.500 | 0.500 | **0.000** | 31,930 | 0.000 |
| &nbsp;&nbsp;2a flagged | 3 | 0.667 | 0.667 | 0.000 | 27,196 | |
| &nbsp;&nbsp;2b not flagged | 19 | 0.772 | 0.553 | 0.018 | 20,935 | |
| &nbsp;&nbsp;2/3 accepted answer wrong | 6 | 0.000 | 0.000 | 0.000 | 31,658 | |
| &nbsp;&nbsp;2/3 accepted answer right | 17 | 1.000 | 0.735 | 0.000 | 19,191 | |

### A3. Within run, G4B XHIGH (b0 -> stage-2 b1, indexed view; b1 ran on 161 of 240 task-runs)
| class of attempt 1 | pairs | attempt 1 credit | fresh attempt credit | **gain** | fresh tokens | gain / 100k tok |
|---|---|---|---|---|---|---|
| 0 partial progress, no accept | 13 | 0.000 | 0.000 | **0.000** | 23,506 | 0.000 |
| 1 nothing | 114 | 0.000 | 0.000 | **0.000** | 22,257 | 0.000 |
| 2 one accepted answer | 33 | 0.788 | 0.000 | **0.000** | 21,544 | 0.000 |
| 3 two+ distinct accepted | 1 | 1.000 | 0.000 | **0.000** | 23,644 | 0.000 |
| &nbsp;&nbsp;2a flagged | 7 | 0.714 | 0.000 | 0.000 | 22,442 | |
| &nbsp;&nbsp;2b not flagged | 26 | 0.808 | 0.000 | 0.000 | 21,302 | |
| &nbsp;&nbsp;2/3 accepted answer wrong | 6 | 0.000 | 0.000 | 0.000 | 23,158 | |
| &nbsp;&nbsp;2/3 accepted answer right | 26 | 1.000 | 0.000 | 0.000 | 21,018 | |

### A4. A1 + A2 pooled
| class of attempt 1 | pairs | attempt 1 credit | fresh attempt credit | **gain** | fresh tokens | gain / 100k tok |
|---|---|---|---|---|---|---|
| 0 partial progress, no accept | 19 | 0.000 | 0.079 | **0.079** | 41,480 | 0.190 |
| 1 nothing | 145 | 0.000 | 0.141 | **0.141** | 48,101 | 0.294 |
| 2 one accepted answer | 106 | 0.733 | 0.533 | **0.055** | 34,058 | 0.162 |
| 3 two+ distinct accepted | 10 | 0.700 | 0.450 | **0.100** | 34,247 | 0.292 |
| &nbsp;&nbsp;2a flagged | 16 | 0.688 | 0.500 | 0.000 | 37,054 | |
| &nbsp;&nbsp;2b not flagged | 90 | 0.741 | 0.539 | 0.065 | 33,526 | |
| &nbsp;&nbsp;2/3 accepted answer wrong | 29 | 0.000 | 0.207 | 0.207 | 37,541 | |
| &nbsp;&nbsp;2/3 accepted answer right | 82 | 1.000 | 0.646 | 0.000 | 32,122 | |

### A5. k-th independent xhigh attempt on tasks where all earlier attempts had no correct accepted answer
Pool per task: G4B XHIGH E16 b0, G4B XHIGH noE16 b0, E15B 45K30K b0+b1, E15B 40K35K b0+b1; 300 random orders per task.
| k | cases still at 0 | P(k-th adds credit) | mean tokens | gain / 100k tok | tasks with >= k attempts |
|---|---|---|---|---|---|
| 1 | 36000 | 0.270 | 43,615 | 0.618 | 120 |
| 2 | 25975 | 0.134 | 46,946 | 0.286 | 120 |
| 3 | 2501 | 0.156 | 39,105 | 0.400 | 20 |
| 4 | 2105 | 0.119 | 39,071 | 0.305 | 20 |
| 5 | 1854 | 0.103 | 39,214 | 0.263 | 20 |
| 6 | 1663 | 0.098 | 39,678 | 0.247 | 20 |

### B1. AUC of each order key for 'attempt 1 got credit > 0' (0.5 = random)
| key | G4B_XHIGH_E16 | G4B_XHIGH_noE16 | G4B_LOW_E16 | mean |
|---|---|---|---|---|
| ORDER2 + cells (current) | 0.573 | 0.566 | 0.547 | 0.562 |
| ORDER2 only | 0.575 | 0.583 | 0.564 | 0.574 |
| fewer total cells | 0.498 | 0.473 | 0.472 | 0.481 |
| more train pairs | 0.515 | 0.530 | 0.533 | 0.526 |
| 1 test input | 0.580 | 0.565 | 0.505 | 0.550 |
| output smaller | 0.509 | 0.452 | 0.438 | 0.466 |
| same size | 0.481 | 0.530 | 0.528 | 0.513 |
| fewer colours | 0.645 | 0.594 | 0.676 | 0.638 |
| smaller mean input | 0.495 | 0.486 | 0.503 | 0.495 |
| (positives per run) | 37 | 32 | 25 | |
| (AUC of tokens used: fewer tokens -> solved?) | 0.788 | 0.805 | 0.850 | |

### B2. Attempt-1 task credit collected when only 30 % / 60 % of the run's attempt-1 tokens fit, in the given order
| key | G4B_XHIGH_E16 @30% | G4B_XHIGH_E16 @60% | G4B_XHIGH_noE16 @30% | G4B_XHIGH_noE16 @60% | G4B_LOW_E16 @30% | G4B_LOW_E16 @60% |
|---|---|---|---|---|---|---|
| ORDER2 + cells (current) | 13.0 | 24.5 | 8.0 | 22.0 | 7.0 | 16.2 |
| ORDER2 only | 16.0 | 26.5 | 12.0 | 24.5 | 10.0 | 19.2 |
| fewer total cells | 10.5 | 20.0 | 7.0 | 17.0 | 7.2 | 11.2 |
| more train pairs | 11.5 | 22.5 | 9.0 | 19.0 | 5.2 | 15.2 |
| 1 test input | 13.0 | 26.0 | 9.0 | 22.0 | 7.0 | 15.0 |
| output smaller | 12.5 | 21.5 | 8.5 | 16.0 | 4.0 | 12.2 |
| same size | 11.0 | 21.5 | 8.5 | 18.5 | 6.2 | 15.2 |
| fewer colours | 20.0 | 25.0 | 15.0 | 19.0 | 13.0 | 17.0 |
| smaller mean input | 11.5 | 19.0 | 9.0 | 15.0 | 6.7 | 11.7 |
| random (mean of 500) | 10.8 | 21.6 | 9.3 | 18.5 | 6.7 | 13.5 |
| (total credit at 100 %) | 36.0 | 36.0 | 31.0 | 31.0 | 22.7 | 22.7 |

### B3. Two-key orders chosen on 2 runs (mean AUC), tested on the held-out run
| held-out G4B_XHIGH_E16 | fewer colours, then ORDER2 only (train 0.638) | test 0.652 | current 0.573 |
| held-out G4B_XHIGH_noE16 | fewer colours, then ORDER2 only (train 0.665) | test 0.599 | current 0.566 |
| held-out G4B_LOW_E16 | fewer colours, then 1 test input (train 0.627) | test 0.672 | current 0.547 |

### B4. The same keys on labels from other methods (not used to pick anything)
| key | ARC Prize xhigh | ARC Prize low | E10 direct 86k |
|---|---|---|---|
| ORDER2 + cells (current) | 0.627 | 0.654 | 0.572 |
| ORDER2 only | 0.603 | 0.627 | 0.551 |
| fewer total cells | 0.572 | 0.568 | 0.567 |
| more train pairs | 0.576 | 0.563 | 0.532 |
| 1 test input | 0.562 | 0.586 | 0.521 |
| output smaller | 0.550 | 0.534 | 0.459 |
| same size | 0.444 | 0.443 | 0.549 |
| fewer colours | 0.425 | 0.560 | 0.494 |
| smaller mean input | 0.506 | 0.524 | 0.541 |
| fewer colours, then ORDER2 (proposed) | 0.438 | 0.576 | 0.499 |
| ORDER2, then fewer colours | 0.571 | 0.650 | 0.549 |
| (positives) | 40 | 19 | 52 |

### B5. B2 for the combined keys
| ORDER2 + cells (current) | 13.0 | 24.5 | 8.0 | 22.0 | 7.0 | 16.2 |
| ORDER2 only | 16.0 | 26.5 | 12.0 | 24.5 | 10.0 | 19.2 |
| fewer colours, then ORDER2 (proposed) | 20.5 | 25.0 | 14.0 | 19.5 | 12.0 | 17.0 |
| ORDER2, then fewer colours | 17.0 | 27.5 | 13.0 | 22.0 | 12.0 | 18.0 |

### A6. Stage-2 queue simulation on the A4 pairs: credit gained when only part of the attempt-2 tokens fit
| order | @25 % | @50 % | @75 % | @100 % |
|---|---|---|---|---|
| current (0, 1, 2 flagged first, 3) | 7.9 | 16.3 | 22.9 | 28.8 |
| no-accept first (0 = 1), then 3, then 2 | 8.2 | 16.6 | 24.1 | 28.8 |
| 1, 0, 3, 2 | 8.6 | 17.1 | 24.1 | 28.8 |
| no-accept first, then accepted (2 = 3) | 8.2 | 16.7 | 23.8 | 28.8 |
(pairs 280; credit units = tasks; 100 % = every pair's fresh attempt)
