# Facts v2 relations on the 120 evaluation tasks (offline)

- tasks with >= 1 relation: 91/120; relations block tokens: median 67, max 148
- tasks whose relation determines the answer outright: 0; correct on every test input: 0 []; Duck R3K solved them in 0/0 runs

| relation | tasks | holds on the test pairs (precision) | of those, Duck solved in >= 1 R3K run |
|---|---|---|---|
| change_scope | 61 | 24/61 (39 %) | 33/61 |
| paint_only | 18 | 18/18 (100 %) | 10/18 |
| out_symmetry | 8 | 5/8 (62 %) | 5/8 |
| out_shape_const | 5 | 2/5 (40 %) | 3/5 |
| objects_move | 4 | 2/4 (50 %) | 2/4 |
| objects_recolor | 2 | 2/2 (100 %) | 2/2 |
| erase_only | 1 | 1/1 (100 %) | 0/1 |

- Duck correct accepts (R3K, both runs): tasks WITH a relation: 62, median tokens 29.2k; WITHOUT: 18, median 30.7k
- tasks Duck never solved (0/2 runs) that have only precise relations (all hold on the test): 19
- tasks solved in < 2 runs that have only precise relations: 33
- tasks where some relation does NOT hold on the test (would mislead): 45 [('135a2760', ['change_scope']), ('16b78196', ['change_scope']), ('16de56c4', ['change_scope']), ('195c6913', ['change_scope']), ('221dfab4', ['change_scope']), ('247ef758', ['change_scope']), ('2c181942', ['change_scope']), ('31f7f899', ['change_scope']), ('38007db0', ['out_shape_const']), ('3a25b0d8', ['out_symmetry']), ('3dc255db', ['change_scope']), ('409aa875', ['change_scope']), ('446ef5d2', ['change_scope']), ('4c3d4a41', ['change_scope']), ('4c416de3', ['change_scope']), ('4e34c42c', ['out_symmetry']), ('53fb4810', ['change_scope']), ('581f7754', ['objects_move']), ('62593bfd', ['change_scope']), ('67e490f4', ['out_symmetry']), ('6e4f6532', ['change_scope']), ('7491f3cf', ['change_scope']), ('7b0280bc', ['change_scope']), ('7b80bb43', ['change_scope']), ('7c66cb00', ['change_scope']), ('800d221b', ['change_scope']), ('80a900e0', ['change_scope']), ('88bcf3b4', ['change_scope']), ('88e364bc', ['objects_move']), ('8f215267', ['change_scope']), ('9385bd28', ['change_scope']), ('97d7923e', ['change_scope']), ('9aaea919', ['change_scope']), ('a25697e4', ['change_scope']), ('a32d8b75', ['out_shape_const']), ('abc82100', ['change_scope']), ('b5ca7ac4', ['change_scope']), ('b99e7126', ['change_scope']), ('cbebaa4b', ['change_scope']), ('d35bdbdc', ['change_scope']), ('de809cff', ['change_scope']), ('dfadab01', ['change_scope']), ('e3721c99', ['change_scope']), ('e376de54', ['change_scope']), ('e87109e9', ['out_shape_const'])]

## Per task (tasks with relations)

| task | Duck runs solved | Duck correct-accept tokens | relations (test ok?) |
|---|---|---|---|
| 135a2760 | 2/2 | [46.5, 44.8] | change_scope (TEST FAILS) |
| 13e47133 | 0/2 | [] | paint_only |
| 142ca369 | 1/2 | [] | paint_only |
| 16b78196 | 0/2 | [] | change_scope (TEST FAILS); objects_move |
| 16de56c4 | 2/2 | [43.9, 31.3] | change_scope (TEST FAILS) |
| 1818057f | 2/2 | [3.3, 6.2] | change_scope |
| 195c6913 | 0/2 | [] | change_scope (TEST FAILS) |
| 1ae2feb7 | 2/2 | [23.0, 15.6] | paint_only |
| 221dfab4 | 1/2 | [32.9] | change_scope (TEST FAILS) |
| 247ef758 | 2/2 | [] | change_scope (TEST FAILS) |
| 269e22fb | 1/2 | [26.6] | out_shape_const |
| 271d71e2 | 0/2 | [] | change_scope |
| 28a6681f | 0/2 | [] | change_scope |
| 2b83f449 | 0/2 | [] | change_scope |
| 2c181942 | 0/2 | [] | change_scope (TEST FAILS) |
| 31f7f899 | 2/2 | [16.1, 12.6] | change_scope (TEST FAILS) |
| 332f06d7 | 0/2 | [] | change_scope |
| 35ab12c3 | 0/2 | [] | paint_only |
| 36a08778 | 2/2 | [39.9, 32.3] | paint_only |
| 38007db0 | 2/2 | [12.9, 17.2] | out_shape_const (TEST FAILS) |
| 3a25b0d8 | 0/2 | [] | out_symmetry (TEST FAILS) |
| 3dc255db | 0/2 | [] | change_scope (TEST FAILS) |
| 3e6067c3 | 2/2 | [28.7] | paint_only |
| 409aa875 | 1/2 | [28.9] | change_scope (TEST FAILS) |
| 446ef5d2 | 0/2 | [] | change_scope (TEST FAILS) |
| 45a5af55 | 2/2 | [13.0, 8.4] | out_symmetry |
| 4a21e3da | 1/2 | [47.7, 27.8] | change_scope |
| 4c3d4a41 | 0/2 | [] | change_scope (TEST FAILS) |
| 4c416de3 | 0/2 | [] | change_scope (TEST FAILS) |
| 4e34c42c | 0/2 | [] | out_symmetry (TEST FAILS) |
| 53fb4810 | 2/2 | [36.6, 46.7, 29.5] | change_scope (TEST FAILS) |
| 581f7754 | 1/2 | [] | change_scope; objects_move (TEST FAILS) |
| 5961cc34 | 1/2 | [21.8] | change_scope |
| 62593bfd | 1/2 | [48.3] | change_scope (TEST FAILS); objects_move |
| 64efde09 | 0/2 | [] | paint_only |
| 67e490f4 | 2/2 | [18.0, 34.0] | out_symmetry (TEST FAILS) |
| 6e453dd6 | 1/2 | [36.7] | change_scope |
| 6e4f6532 | 0/2 | [] | change_scope (TEST FAILS) |
| 71e489b6 | 1/2 | [45.0] | change_scope |
| 7491f3cf | 0/2 | [] | change_scope (TEST FAILS) |
| 7666fa5d | 1/2 | [] | paint_only |
| 7b0280bc | 1/2 | [45.7] | change_scope (TEST FAILS); objects_recolor |
| 7b5033c1 | 0/2 | [] | out_symmetry |
| 7b80bb43 | 0/2 | [] | change_scope (TEST FAILS) |
| 7c66cb00 | 2/2 | [23.7, 28.7] | change_scope (TEST FAILS) |
| 7ed72f31 | 2/2 | [34.7] | paint_only |
| 800d221b | 1/2 | [27.2] | change_scope (TEST FAILS) |
| 80a900e0 | 1/2 | [33.6] | change_scope (TEST FAILS); objects_recolor |
| 88bcf3b4 | 0/2 | [] | change_scope (TEST FAILS) |
| 88e364bc | 0/2 | [] | change_scope; objects_move (TEST FAILS) |
| 8b7bacbf | 0/2 | [] | paint_only |
| 8b9c3697 | 1/2 | [36.4] | change_scope |
| 8e5c0c38 | 0/2 | [] | erase_only |
| 8f215267 | 2/2 | [31.1, 22.7] | change_scope (TEST FAILS) |
| 8f3a5a89 | 2/2 | [48.7, 26.3] | change_scope |
| 9385bd28 | 0/2 | [] | change_scope (TEST FAILS) |
| 97d7923e | 0/2 | [] | change_scope (TEST FAILS) |
| 981571dc | 2/2 | [44.2, 39.0] | change_scope; out_symmetry |
| 9aaea919 | 2/2 | [14.3, 17.2] | change_scope (TEST FAILS) |
| 9bbf930d | 0/2 | [] | change_scope |
| a25697e4 | 0/2 | [] | change_scope (TEST FAILS) |
| a32d8b75 | 0/2 | [] | out_shape_const (TEST FAILS) |
| a395ee82 | 2/2 | [21.2, 33.9] | change_scope |
| a47bf94d | 0/2 | [] | change_scope |
| aa4ec2a5 | 2/2 | [17.8, 13.1] | change_scope |
| abc82100 | 0/2 | [] | change_scope (TEST FAILS) |
| b10624e5 | 2/2 | [36.8, 35.6] | paint_only |
| b5ca7ac4 | 1/2 | [27.1] | change_scope (TEST FAILS) |
| b99e7126 | 1/2 | [34.0] | change_scope (TEST FAILS) |
| b9e38dc0 | 0/2 | [] | paint_only |
| c4d067a0 | 1/2 | [18.4] | paint_only |
| c7f57c3e | 0/2 | [] | change_scope |
| cb2d8a2c | 1/2 | [37.9] | change_scope |
| cbebaa4b | 0/2 | [] | change_scope (TEST FAILS) |
| d35bdbdc | 1/2 | [] | change_scope (TEST FAILS) |
| d59b0160 | 0/2 | [] | change_scope |
| d8e07eb2 | 1/2 | [] | change_scope |
| da515329 | 0/2 | [] | paint_only |
| db0c5428 | 1/2 | [] | paint_only |
| db695cfb | 2/2 | [17.6, 18.3] | paint_only |
| dbff022c | 0/2 | [] | paint_only |
| dd6b8c4b | 2/2 | [22.1] | change_scope |
| de809cff | 0/2 | [] | change_scope (TEST FAILS) |
| dfadab01 | 1/2 | [33.8] | change_scope (TEST FAILS) |
| e12f9a14 | 0/2 | [] | paint_only |
| e3721c99 | 1/2 | [] | change_scope (TEST FAILS) |
| e376de54 | 0/2 | [] | change_scope (TEST FAILS) |
| e8686506 | 1/2 | [49.8] | out_symmetry |
| e87109e9 | 0/2 | [] | out_shape_const (TEST FAILS) |
| eee78d87 | 1/2 | [42.3] | out_symmetry; out_shape_const |
| faa9f03d | 0/2 | [] | change_scope |
