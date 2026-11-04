D:\ARC AGI 2\_scratch\order_rules\budget_grid.py:12: SyntaxWarning: "\ " is an invalid escape sequence. Such sequences will not work in the future. Did you mean "\\ "? A raw string is also an option.
  print("| attempt 1 \ attempt 2 | " + " | ".join(f"{b // 1000}k" for b in B2S) + " |")
D:\ARC AGI 2\_scratch\order_rules\budget_grid.py:23: SyntaxWarning: "\ " is an invalid escape sequence. Such sequences will not work in the future. Did you mean "\\ "? A raw string is also an option.
  print("| attempt 1 \ attempt 2 | " + " | ".join(f"{b // 1000}k" for b in B2S) + " |")
### P0 (now: attempt 2 on every task) - task credit at 11.6 h (of 240) | hours both attempts need
| attempt 1 \ attempt 2 | 30k | 35k | 45k | 50k | 60k |
|---|---|---|---|---|---|
| 35k | 47.0 (9.2 h) | 53.0 (9.8 h) | - | - | - |
| 40k | 55.5 (9.8 h) | 61.5 (10.4 h) | - | - | - |
| 45k | 64.5 (10.2 h) | 70.5 (10.8 h) | 79.5 (11.8 h) | - | - |
| 50k | 67.0 (10.6 h) | 73.0 (11.2 h) | 81.0 (12.2 h) | 83.5 (12.5 h) | - |
| 55k | 71.0 (10.8 h) | 77.0 (11.4 h) | 84.0 (12.4 h) | 86.5 (12.8 h) | - |
| 60k | 72.0 (11.0 h) | 78.0 (11.6 h) | 85.0 (12.6 h) | 86.5 (12.9 h) | 87.5 (13.3 h) |
| 70k | 72.0 (11.1 h) | 77.0 (11.7 h) | 85.0 (12.7 h) | 86.5 (13.0 h) | 86.5 (13.4 h) |

### R + stage 3 (estimate; attempt-3 p = 0.10 uncapped x share of solves within the attempt-2 budget)
| attempt 1 \ attempt 2 | 30k | 35k | 45k | 50k | 60k |
|---|---|---|---|---|---|
| 35k | 52.6 | 58.3 | - | - | - |
| 40k | 60.4 | 66.0 | - | - | - |
| 45k | 68.7 | 74.6 | 81.9 | - | - |
| 50k | 70.6 | 76.3 | 83.5 | 84.0 | - |
| 55k | 73.3 | 79.2 | 85.2 | 86.0 | - |
| 60k | 74.2 | 79.9 | 86.0 | 86.9 | 88.2 |
| 70k | 73.9 | 79.6 | 85.8 | 86.6 | 87.7 |

share of correct solves within the budget (G4B XHIGH b0, 69 solves): {'30k': 0.435, '35k': 0.58, '40k': 0.725, '45k': 0.87, '50k': 0.913, '55k': 0.986, '60k': 1.0}
