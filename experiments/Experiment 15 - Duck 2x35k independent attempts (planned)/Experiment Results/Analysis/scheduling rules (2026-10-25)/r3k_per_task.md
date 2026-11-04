# R3K1 vs R3K2 per task (EVAL120, 6 h, 2026-10-25)

Cell = outcome of the attempt. `OK @Xk` = an accepted answer that is correct, X = generated tokens when it was accepted; `OK-weak` = correct but accepted only after the commit text; `WRONG` / `WRONG-weak` = accepted, wrong (tokens used in brackets); `partial-right` = no accept, but a partial candidate had a right query; `no-accept`; — = the attempt did not run. hit = queries right in the submission.

Tasks with a hit in both runs: 30; only R3K1: 14 ['0934a4d8', '269e22fb', '291dc1e1', '409aa875', '6e453dd6', '7666fa5d', '7b0280bc', '80a900e0', 'b99e7126', 'c4d067a0', 'd35bdbdc', 'dfadab01', 'e3721c99', 'e8686506']; only R3K2: 19 ['142ca369', '20270e3b', '221dfab4', '4a21e3da', '581f7754', '5961cc34', '62593bfd', '71e489b6', '7b3084d4', '800d221b', '89565ca0', '8b9c3697', 'b5ca7ac4', 'cb2d8a2c', 'd8e07eb2', 'db0c5428', 'eee78d87', 'f931b4a8', 'fc7cae8d']; in neither: 57.
Correct accepted answers: 80; tokens at the first correct accept: p25 18.0k, median 29.5k, p75 39.0k, max 49.9k; <= 20k: 24, 20-35k: 28, > 35k: 28.

| # | task | queries | R3K1 attempt 1 | R3K1 attempt 2 | R3K1 hit | R3K2 attempt 1 | R3K2 attempt 2 | R3K2 hit |
|---|---|---|---|---|---|---|---|---|
| 1 | 135a2760 | 1 | OK-weak @46.5k | no-accept (45.0k) | 1/1 | no-accept (50.0k) | OK-weak @44.8k | 1/1 |
| 2 | 136b0064 | 1 | OK-weak @46.7k | OK-weak @43.3k | 1/1 | no-accept (50.0k) | OK @35.6k | 1/1 |
| 3 | 16de56c4 | 2 | OK @43.9k | — | 2/2 | OK @31.3k | — | 2/2 |
| 4 | 1818057f | 1 | OK @3.3k | — | 1/1 | OK @6.2k | — | 1/1 |
| 5 | 1ae2feb7 | 3 | OK @23.0k | — | 3/3 | OK @15.6k | — | 3/3 |
| 6 | 2ba387bc | 1 | OK @4.3k | — | 1/1 | OK @4.5k | — | 1/1 |
| 7 | 31f7f899 | 1 | OK @16.1k | — | 1/1 | OK @12.6k | — | 1/1 |
| 8 | 36a08778 | 2 | OK @39.9k | — | 2/2 | no-accept (50.0k) | OK @32.3k | 2/2 |
| 9 | 38007db0 | 2 | OK @12.9k | — | 2/2 | OK @17.2k | — | 2/2 |
| 10 | 45a5af55 | 1 | OK @13.0k | — | 1/1 | OK @8.4k | — | 1/1 |
| 11 | 53fb4810 | 1 | OK @36.6k | — | 1/1 | OK-weak @46.7k | OK @29.5k | 1/1 |
| 12 | 58490d8a | 1 | OK @22.9k | — | 1/1 | OK @10.9k | — | 1/1 |
| 13 | 58f5dbd5 | 1 | OK @19.4k | — | 1/1 | OK @15.3k | — | 1/1 |
| 14 | 67e490f4 | 1 | OK @18.0k | — | 1/1 | OK @34.0k | — | 1/1 |
| 15 | 7c66cb00 | 1 | OK @23.7k | — | 1/1 | OK @28.7k | — | 1/1 |
| 16 | 8f215267 | 1 | OK @31.1k | — | 1/1 | OK @22.7k | — | 1/1 |
| 17 | 8f3a5a89 | 1 | OK-weak @48.7k | no-accept (44.5k) | 1/1 | OK @26.3k | — | 1/1 |
| 18 | 981571dc | 1 | no-accept (50.0k) | OK @39.0k | 1/1 | OK @44.2k | — | 1/1 |
| 19 | 9aaea919 | 1 | OK @14.3k | — | 1/1 | OK @17.2k | — | 1/1 |
| 20 | a251c730 | 1 | OK @44.2k | — | 1/1 | no-accept (50.0k) | OK @35.6k | 1/1 |
| 21 | a395ee82 | 1 | OK @21.2k | — | 1/1 | OK @33.9k | — | 1/1 |
| 22 | aa4ec2a5 | 1 | OK @17.8k | — | 1/1 | OK @13.1k | — | 1/1 |
| 23 | b10624e5 | 1 | no-accept (49.6k) | OK @35.6k | 1/1 | OK @36.8k | — | 1/1 |
| 24 | bf45cf4b | 1 | OK @11.9k | — | 1/1 | OK @8.9k | — | 1/1 |
| 25 | db695cfb | 1 | OK @17.6k | — | 1/1 | OK @18.3k | — | 1/1 |
| 26 | 3e6067c3 | 2 | WRONG (23.4k) | — | 1/2 | OK @28.7k | — | 2/2 |
| 27 | 7ed72f31 | 2 | WRONG (28.2k) | — | 1/2 | OK @34.7k | — | 2/2 |
| 28 | dd6b8c4b | 2 | no-accept (50.0k) | partial-right (44.8k) | 1/2 | OK @22.1k | — | 2/2 |
| 29 | 0934a4d8 | 1 | OK @39.1k | — | 1/1 | WRONG-weak (47.1k) | WRONG (45.0k) | 0/1 |
| 30 | 142ca369 | 2 | no-accept (50.0k) | no-accept (45.0k) | 0/2 | no-accept (50.0k) | partial-right (45.0k) | 2/2 |
| 31 | 221dfab4 | 2 | WRONG (33.5k) | — | 0/2 | OK @32.9k | — | 2/2 |
| 32 | 269e22fb | 2 | OK @26.6k | — | 2/2 | no-accept (49.3k) | no-accept (45.0k) | 0/2 |
| 33 | 291dc1e1 | 1 | OK @43.2k | — | 1/1 | WRONG-weak (48.5k) | no-accept (45.0k) | 0/1 |
| 34 | 409aa875 | 1 | no-accept (50.0k) | OK @28.9k | 1/1 | WRONG-weak (47.0k) | no-accept (45.0k) | 0/1 |
| 35 | 4a21e3da | 2 | no-accept (49.6k) | no-accept (45.0k) | 0/2 | OK-weak @47.7k | OK @27.8k | 2/2 |
| 36 | 5961cc34 | 1 | WRONG (38.4k) | — | 0/1 | OK @21.8k | — | 1/1 |
| 37 | 62593bfd | 2 | no-accept (50.0k) | no-accept (45.0k) | 0/2 | OK-weak @48.3k | no-accept (45.0k) | 2/2 |
| 38 | 6e453dd6 | 1 | OK @36.7k | — | 1/1 | WRONG (35.9k) | — | 0/1 |
| 39 | 71e489b6 | 2 | WRONG (42.5k) | — | 0/2 | OK @45.0k | — | 2/2 |
| 40 | 7666fa5d | 1 | partial-right (50.0k) | no-accept (45.0k) | 1/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 41 | 7b0280bc | 1 | OK @45.7k | — | 1/1 | WRONG-weak (47.1k) | no-accept (45.0k) | 0/1 |
| 42 | 7b3084d4 | 1 | no-accept (50.0k) | no-accept (44.8k) | 0/1 | no-accept (50.0k) | OK @28.6k | 1/1 |
| 43 | 800d221b | 1 | no-accept (49.7k) | WRONG-weak (44.0k) | 0/1 | no-accept (50.0k) | OK @27.2k | 1/1 |
| 44 | 80a900e0 | 1 | OK @33.6k | — | 1/1 | WRONG-weak (47.8k) | no-accept (45.0k) | 0/1 |
| 45 | 89565ca0 | 1 | no-accept (49.7k) | no-accept (45.0k) | 0/1 | no-accept (49.4k) | partial-right (44.5k) | 1/1 |
| 46 | 8b9c3697 | 1 | WRONG (46.5k) | — | 0/1 | WRONG-weak (48.0k) | OK @36.4k | 1/1 |
| 47 | b5ca7ac4 | 1 | WRONG (33.2k) | — | 0/1 | OK @27.1k | — | 1/1 |
| 48 | b99e7126 | 1 | no-accept (50.0k) | OK @34.0k | 1/1 | WRONG-weak (49.2k) | no-accept (45.0k) | 0/1 |
| 49 | c4d067a0 | 1 | no-accept (49.3k) | OK @18.4k | 1/1 | WRONG (44.5k) | — | 0/1 |
| 50 | cb2d8a2c | 2 | no-accept (50.0k) | no-accept (45.0k) | 0/2 | no-accept (50.0k) | OK @37.9k | 2/2 |
| 51 | d8e07eb2 | 2 | no-accept (50.0k) | no-accept (17.9k) | 0/2 | partial-right (50.0k) | partial-right (45.0k) | 2/2 |
| 52 | db0c5428 | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | no-accept (50.0k) | partial-right (45.0k) | 1/1 |
| 53 | dfadab01 | 1 | OK @33.8k | — | 1/1 | WRONG-weak (47.3k) | WRONG-weak (44.8k) | 0/1 |
| 54 | e8686506 | 1 | OK-weak @49.8k | no-accept (45.0k) | 1/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 55 | eee78d87 | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | OK @42.3k | — | 1/1 |
| 56 | f931b4a8 | 2 | no-accept (50.0k) | no-accept (44.3k) | 0/2 | no-accept (50.0k) | OK @32.8k | 2/2 |
| 57 | fc7cae8d | 1 | WRONG-weak (48.9k) | WRONG-weak (41.4k) | 0/1 | OK-weak @49.9k | WRONG-weak (43.3k) | 1/1 |
| 58 | 247ef758 | 2 | WRONG (22.1k) | — | 1/2 | WRONG (46.6k) | — | 1/2 |
| 59 | b6f77b65 | 3 | no-accept (50.0k) | partial-right (44.6k) | 1/3 | no-accept (50.0k) | partial-right (44.9k) | 1/3 |
| 60 | 20270e3b | 2 | no-accept (50.0k) | no-accept (45.0k) | 0/2 | WRONG-weak (47.6k) | no-accept (45.0k) | 1/2 |
| 61 | 581f7754 | 2 | no-accept (50.0k) | no-accept (45.0k) | 0/2 | no-accept (50.0k) | WRONG-weak (44.2k) | 1/2 |
| 62 | e3721c99 | 2 | WRONG (29.0k) | — | 1/2 | WRONG (25.3k) | — | 0/2 |
| 63 | d35bdbdc | 3 | WRONG (38.7k) | — | 1/3 | WRONG (47.9k) | — | 0/3 |
| 64 | 13e47133 | 2 | no-accept (50.0k) | no-accept (45.0k) | 0/2 | no-accept (50.0k) | no-accept (31.3k) | 0/2 |
| 65 | 16b78196 | 1 | no-accept (50.0k) | no-accept (40.9k) | 0/1 | no-accept (50.0k) | no-accept (17.1k) | 0/1 |
| 66 | 195c6913 | 2 | no-accept (50.0k) | no-accept (36.0k) | 0/2 | no-accept (50.0k) | no-accept (11.6k) | 0/2 |
| 67 | 20a9e565 | 2 | no-accept (49.6k) | no-accept (45.0k) | 0/2 | no-accept (50.0k) | no-accept (44.4k) | 0/2 |
| 68 | 21897d95 | 2 | no-accept (50.0k) | no-accept (44.8k) | 0/2 | no-accept (49.3k) | no-accept (45.0k) | 0/2 |
| 69 | 271d71e2 | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 70 | 28a6681f | 1 | no-accept (49.3k) | no-accept (45.0k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 71 | 2b83f449 | 1 | no-accept (50.0k) | no-accept (44.5k) | 0/1 | no-accept (49.3k) | no-accept (45.0k) | 0/1 |
| 72 | 2c181942 | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 73 | 2d0172a1 | 2 | no-accept (49.6k) | no-accept (44.9k) | 0/2 | no-accept (50.0k) | no-accept (44.5k) | 0/2 |
| 74 | 332f06d7 | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 75 | 35ab12c3 | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 76 | 3a25b0d8 | 2 | no-accept (50.0k) | no-accept (7.0k) | 0/2 | no-accept (49.3k) | no-accept (44.8k) | 0/2 |
| 77 | 3dc255db | 1 | no-accept (50.0k) | WRONG-weak (43.0k) | 0/1 | no-accept (50.0k) | WRONG-weak (44.8k) | 0/1 |
| 78 | 446ef5d2 | 2 | no-accept (50.0k) | no-accept (20.5k) | 0/2 | no-accept (50.0k) | no-accept (8.7k) | 0/2 |
| 79 | 4c3d4a41 | 2 | no-accept (50.0k) | WRONG (29.0k) | 0/2 | WRONG (36.5k) | — | 0/2 |
| 80 | 4c416de3 | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 81 | 4c7dc4dd | 2 | no-accept (49.4k) | no-accept (45.0k) | 0/2 | no-accept (50.0k) | no-accept (45.0k) | 0/2 |
| 82 | 4e34c42c | 2 | no-accept (50.0k) | no-accept (11.1k) | 0/2 | no-accept (50.0k) | — | 0/2 |
| 83 | 5545f144 | 1 | no-accept (49.7k) | no-accept (45.0k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 84 | 5dbc8537 | 2 | no-accept (50.0k) | no-accept (25.6k) | 0/2 | no-accept (50.0k) | no-accept (1.1k) | 0/2 |
| 85 | 64efde09 | 1 | no-accept (50.0k) | no-accept (34.5k) | 0/1 | no-accept (50.0k) | no-accept (32.5k) | 0/1 |
| 86 | 65b59efc | 2 | no-accept (49.4k) | no-accept (45.0k) | 0/2 | no-accept (50.0k) | no-accept (45.0k) | 0/2 |
| 87 | 6e4f6532 | 2 | no-accept (50.0k) | no-accept (19.3k) | 0/2 | no-accept (50.0k) | — | 0/2 |
| 88 | 6ffbe589 | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | no-accept (49.4k) | no-accept (45.0k) | 0/1 |
| 89 | 7491f3cf | 1 | WRONG (38.4k) | — | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 90 | 78332cb0 | 2 | WRONG (39.6k) | — | 0/2 | no-accept (50.0k) | no-accept (44.5k) | 0/2 |
| 91 | 7b5033c1 | 1 | WRONG (3.0k) | — | 0/1 | WRONG (3.8k) | — | 0/1 |
| 92 | 7b80bb43 | 1 | WRONG-weak (46.7k) | no-accept (44.6k) | 0/1 | WRONG-weak (47.0k) | no-accept (45.0k) | 0/1 |
| 93 | 8698868d | 1 | no-accept (50.0k) | no-accept (44.6k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 94 | 88bcf3b4 | 2 | no-accept (50.0k) | no-accept (45.0k) | 0/2 | no-accept (50.0k) | no-accept (45.0k) | 0/2 |
| 95 | 88e364bc | 2 | no-accept (50.0k) | no-accept (40.3k) | 0/2 | no-accept (50.0k) | no-accept (34.0k) | 0/2 |
| 96 | 898e7135 | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 97 | 8b7bacbf | 2 | no-accept (50.0k) | no-accept (44.3k) | 0/2 | no-accept (49.3k) | no-accept (45.0k) | 0/2 |
| 98 | 8e5c0c38 | 2 | no-accept (50.0k) | no-accept (22.5k) | 0/2 | no-accept (50.0k) | — | 0/2 |
| 99 | 9385bd28 | 2 | no-accept (50.0k) | no-accept (44.3k) | 0/2 | no-accept (49.4k) | no-accept (45.0k) | 0/2 |
| 100 | 97d7923e | 1 | WRONG-weak (46.9k) | no-accept (45.0k) | 0/1 | WRONG-weak (46.3k) | WRONG-weak (41.9k) | 0/1 |
| 101 | 9bbf930d | 1 | no-accept (50.0k) | no-accept (44.7k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 102 | a25697e4 | 2 | no-accept (50.0k) | no-accept (44.4k) | 0/2 | no-accept (50.0k) | no-accept (21.9k) | 0/2 |
| 103 | a32d8b75 | 2 | no-accept (49.7k) | no-accept (45.0k) | 0/2 | no-accept (49.2k) | no-accept (44.9k) | 0/2 |
| 104 | a47bf94d | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 105 | a6f40cea | 1 | no-accept (50.0k) | no-accept (44.3k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 106 | abc82100 | 2 | no-accept (50.0k) | no-accept (45.0k) | 0/2 | no-accept (49.5k) | no-accept (45.0k) | 0/2 |
| 107 | b0039139 | 2 | WRONG-weak (46.8k) | no-accept (45.0k) | 0/2 | no-accept (50.0k) | WRONG (32.7k) | 0/2 |
| 108 | b9e38dc0 | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | no-accept (49.5k) | no-accept (45.0k) | 0/1 |
| 109 | c7f57c3e | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | WRONG (40.8k) | — | 0/1 |
| 110 | cbebaa4b | 2 | no-accept (49.7k) | no-accept (23.5k) | 0/2 | no-accept (50.0k) | — | 0/2 |
| 111 | d59b0160 | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 112 | da515329 | 1 | no-accept (49.9k) | no-accept (44.8k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 113 | dbff022c | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | WRONG-weak (47.6k) | no-accept (45.0k) | 0/1 |
| 114 | de809cff | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | no-accept (50.0k) | no-accept (44.8k) | 0/1 |
| 115 | e12f9a14 | 2 | no-accept (50.0k) | no-accept (45.0k) | 0/2 | no-accept (50.0k) | no-accept (33.6k) | 0/2 |
| 116 | e376de54 | 1 | WRONG (33.4k) | — | 0/1 | WRONG (19.6k) | — | 0/1 |
| 117 | e87109e9 | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 118 | edb79dae | 1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 | no-accept (50.0k) | no-accept (45.0k) | 0/1 |
| 119 | f560132c | 2 | no-accept (50.0k) | no-accept (21.5k) | 0/2 | no-accept (50.0k) | — | 0/2 |
| 120 | faa9f03d | 2 | no-accept (50.0k) | no-accept (45.0k) | 0/2 | no-accept (50.0k) | no-accept (45.0k) | 0/2 |
