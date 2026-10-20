# E11-04 (rev 2) capacity + real-concurrency validation · run 20261018T031243Z

Driver 580.178.04 (CUDA 13.0) · preflight PASS

- FP8 KV: checkpoint k/v scale tensors 32 · values {"n_tensors": 32, "sample_names": ["model.language_model.layers.11.self_attn.k_scale", "model.language_model.layers.11.self_attn.v_scale", "model.language_model.layers.15.self_attn.k_scale", "model.language_model.layers.15.self_attn.v_scale"], "n_values": 32, "min": 0.0213623046875, "max": 0.2539062 · kv_cache_scheme {"actorder": null, "block_structure": null, "dynamic": false, "group_size": null, "num_bits": 8, "observer": "static_minmax", "observer_kwargs": {}, "scale_dtyp

## Startup caps (server log; fail-closed gate)

| arm | launches | effective running (mamba cache / slots) | logged cap | KV tokens | KV GB target+draft | verify graph bs | prefill graph | free GB | FP8 scale warning | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| W1_lazy_fp16 | [('0.83', 'ready in 567s', '')] | 12 (56 / 4) | None | 881336 | 6.72+2.1 | [1, 2, 3, 4, 5, 6, 7, 8, 10, 12] | False | 2.22 | True | FAIL {"running_12": true, "kv_min": false, "verify_graph_12": true, "no_prefill_graph": true, "headroom": true, "pass": false} |
| W2_lazy_bf16 | [('0.83', 'ready in 71s', '')] | 12 (56 / 4) | None | 881336 | 6.72+2.1 | [1, 2, 3, 4, 5, 6, 7, 8, 10, 12] | False | 2.22 | True | FAIL {"running_12": true, "kv_min": false, "verify_graph_12": true, "no_prefill_graph": true, "headroom": true, "pass": false} |
| W3_lazy_fp32 | [('0.83', 'ready in 76s', '')] | 12 (56 / 4) | None | 598712 | 4.56+1.42 | [1, 2, 3, 4, 5, 6, 7, 8, 10, 12] | False | 2.21 | True | FAIL {"running_12": true, "kv_min": false, "verify_graph_12": true, "no_prefill_graph": true, "headroom": true, "pass": false} |

## Sweep (short prompts, 4,096 tokens) with real concurrency

| arm | requests | total tok/s | log gen tok/s | accept | real running max | real level? |
|---|---|---|---|---|---|---|
| W1_lazy_fp16 | 1 | 110.6 | 105.6 | 4.50 | 1 | PASS |
| W1_lazy_fp16 | 4 | 257.5 | 281.4 | 4.49 | 4 | PASS |
| W1_lazy_fp16 | 8 | 363.8 | 442.6 | 4.67 | 8 | PASS |
| W1_lazy_fp16 | 12 | 325.9 | - | 4.51 | 11 | **FAIL** |
| W2_lazy_bf16 | 1 | 114.3 | 115.6 | 4.61 | 1 | PASS |
| W2_lazy_bf16 | 4 | 272.8 | 315.5 | 4.74 | 4 | PASS |
| W2_lazy_bf16 | 8 | 357.6 | 444.7 | 4.55 | 8 | PASS |
| W2_lazy_bf16 | 12 | 350.9 | - | 4.79 | 11 | **FAIL** |
| W3_lazy_fp32 | 1 | 100.0 | 97.7 | 4.15 | 1 | PASS |
| W3_lazy_fp32 | 4 | 273.5 | 287.2 | 4.56 | 4 | PASS |
| W3_lazy_fp32 | 8 | 337.9 | 419.8 | 4.79 | 8 | PASS |
| W3_lazy_fp32 | 12 | 321.4 | - | 4.67 | 11 | **FAIL** |

## PROD closed loop (12 in flight, ARC prompts, up to 60000 thinking, T 1.0) · target >= 332 tok/s

| arm | window s | **tok/s (all responses)** | counter lower bound | log rate at 12 | real 12 share | long-ctx lines (>=40k at 12) | context max | retract | drained | errors | meets target |
|---|---|---|---|---|---|---|---|---|---|---|---|
| W1_lazy_fp16 | 600.0 | **416.5** | 0.0 | 418.4 | 1.0 | 0 (NOT PROVEN) | 30177 | 0 | True | 0 | True |
| W2_lazy_bf16 | 600.1 | **412.1** | 0.0 | 402.6 | 1.0 | 0 (NOT PROVEN) | 31926 | 0 | True | 0 | True |

Decode rate at 12 running by mean context per request (server log):

| arm | <10k | 10-20k | 20-40k | >=40k |
|---|---|---|---|---|
| W1_lazy_fp16 | 458.3 (31 lines) | 413.3 (61 lines) | 388.4 (35 lines) | - (0 lines) |
| W2_lazy_bf16 | 465.5 (29 lines) | 401.9 (63 lines) | 374.9 (34 lines) | - (0 lines) |

## Correctness (each arm's own results only)

- **W1_lazy_fp16 marker:** pass True (valid True, contamination False) · own codes 8/8 · leaks [] · cross [] · solo repeat identical 4/4 · solo vs batch mean prefix 27.5
- **W1_lazy_fp16 contcache:** pass False · short True (min ratio 0.965, warm vs cold 10.04x) · long-prefix answer False [{'unit': '135a2760#0', 'prefix_tokens': 30202, 'cached': 0, 'cached_ratio': 0.0, 'wall_s': 24.53, 'answer_tokens': 1024, 'answer_nonempty': True, 'grid_parsed': False}, {'unit': '3a25b0d8#0', 'prefix_tokens': 27839, 'cached': 0, 'cached_ratio': 0.0, 'wall_s': 19.65, 'answer_tokens': 347, 'answer_nonempty': True, 'grid_parsed': True}]
- **W2_lazy_bf16 marker:** pass False (valid False, contamination False) · own codes 7/8 · leaks [] · cross [] · solo repeat identical 4/4 · solo vs batch mean prefix 97.8
- **W2_lazy_bf16 contcache:** pass False · short True (min ratio 0.979, warm vs cold 9.61x) · long-prefix answer False [{'unit': '135a2760#0', 'prefix_tokens': 30319, 'cached': 0, 'cached_ratio': 0.0, 'wall_s': 24.53, 'answer_tokens': 1024, 'answer_nonempty': True, 'grid_parsed': False}, {'unit': '8f215267#0', 'prefix_tokens': 31951, 'cached': 0, 'cached_ratio': 0.0, 'wall_s': 27.38, 'answer_tokens': 1024, 'answer_nonempty': True, 'grid_parsed': False}]
- **W2_lazy_bf16 greedy vs W1:** 0/4 identical, mean prefix 41.5
- **W3_lazy_fp32 greedy vs W1:** 0/4 identical, mean prefix 44.0

### Sanity W1_lazy_fp16: 8/8 solved · FP8 expectation 6.74

| puzzle | hit | mode | think tokens | FP8 think | accept | sec |
|---|---|---|---|---|---|---|
| 2ba387bc#0 | 1 | natural | 4871 | 5355 | 5.13 | 99.3 |
| bf45cf4b#0 | 1 | natural | 27621 | 17485 | 4.61 | 519.3 |
| 58f5dbd5#0 | 1 | natural | 33797 | 22674 | 5.05 | 561.7 |
| 136b0064#0 | 1 | natural | 19975 | 23164 | 4.28 | 409.8 |
| 31f7f899#0 | 1 | natural | 27698 | 25425 | 3.79 | 602.4 |
| 1818057f#0 | 1 | natural | 44384 | 26891 | 5.01 | 681.0 |
| e8686506#0 | 1 | natural | 28102 | 27730 | 3.78 | 607.2 |
| 7491f3cf#0 | 1 | natural | 33467 | 31874 | 3.97 | 659.7 |

### Sanity W2_lazy_bf16: 7/8 solved · FP8 expectation 6.74

| puzzle | hit | mode | think tokens | FP8 think | accept | sec |
|---|---|---|---|---|---|---|
| 2ba387bc#0 | 1 | natural | 23770 | 5355 | 3.74 | 550.4 |
| bf45cf4b#0 | 1 | natural | 21006 | 17485 | 4.90 | 406.0 |
| 58f5dbd5#0 | 0 | natural | 27911 | 22674 | 4.97 | 505.6 |
| 136b0064#0 | 1 | natural | 21832 | 23164 | 4.37 | 456.4 |
| 31f7f899#0 | 1 | natural | 35114 | 25425 | 3.77 | 724.1 |
| 1818057f#0 | 1 | natural | 30751 | 26891 | 5.67 | 496.1 |
| e8686506#0 | 1 | natural | 36725 | 27730 | 3.79 | 739.2 |
| 7491f3cf#0 | 1 | natural | 36379 | 31874 | 3.88 | 727.2 |

## GO / NO-GO (production candidate = every column PASS; NOT TESTED never counts as PASS)

| arm | capacity gate | real 12 in PROD | long context proven | PROD >= target | no retraction | tail drained | no contamination | prefix cache | sanity >= 6/8 | capacity winner? | production candidate? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| W1_lazy_fp16 | **FAIL** | PASS | **FAIL** | PASS | PASS | PASS | PASS | **FAIL** | PASS | - | - |
| W2_lazy_bf16 | **FAIL** | PASS | **FAIL** | PASS | PASS | PASS | **FAIL** | **FAIL** | PASS | - | - |
| W3_lazy_fp32 | **FAIL** | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | - | - |

Total time 87 min · capacity winner: None · production candidate: None · notes: []

Limits: FP8 KV scale runtime use is unverified; no paired vLLM FP8 run on this same workload (planned E11-05).