# E19 vLLM MTP / prefix-cache correctness: profile E19P3, run 20261027T173653Z

| arm | what | final | in-arm | churn (mode) | rows ok / planned | serial A/A ident/cmp (flips) | cache serial | churn vs serial | cont8k | cont30k | corrupt rows | MTP accept min / median | cache warm hits | launch s | KV tokens | block |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B | MTP 3, prefix caching OFF | **FAIL** | FAIL | PASS (uncalibrated) | 60/60 | 6/6 (0) | - | 0/24 (0) | 4/6 (2) | 4/6 (2) | 0 | 0.582 / 0.786 | 0/18 | 475.0 | 337600 | 1600 |
| K | S1 with bf16 KV (--kv-cache-dtype auto) | **PASS** | PASS | PASS (uncalibrated) | 66/66 | 6/6 (0) | 6/6 (0) | 0/24 (0) | 6/6 (0) | 6/6 (0) | 0 | 0.593 / 0.818 | 18/18 | 325.0 | 176800 | 800 |

## Reasons

- **B** (FAIL): cont_long8k: 2 large-margin flips > the arm's serial A/A 0; cont_long30k: 2 large-margin flips > the arm's serial A/A 0
- **K** (PASS): trailing Mamba block over generated tokens NOT_EXERCISED (no continuation hit reached into the server's generated tokens): no safety claim for that path

## Secondary: token comparison against arm A (informational; MTP on vs off may differ at low margins)


Cells: identical / compared (large-margin flips). A large-margin flip = the first token divergence where a known top-1 minus top-2 logprob margin is > 1 nat. In-arm FAIL: corrupt rows with clean partners, cache-hit or continuation flips above the arm's serial A/A, MTP acceptance collapse. Churn FAIL: flips vs serial above the calibrators' (A / B) churn flips, or above the arm's own serial A/A when no calibrator ran ('uncalibrated'). Rows not run in the budget make an arm NOT_TESTED_BUDGET, never PASS.

Calibrators in this run: B

## Stages

- B: smoke {'tokens': 16, 'error': None, 'margins_present': True, 'usage_seen': True, 'text': '\n\n1 2 3 4 5 6 7 8'}; stages {"serial": {"sec": 249.6, "metric_delta": {"gen": 12288.0, "prompt": 61712.0, "pc_queries": 0.0, "pc_hits": 0.0, "preempt": 0.0, "drafts": 3736.0, "draft_tokens": 11208.0, "accepted": 8566.0, "running": 0.0}}, "churn": {"level": 14, "sec": 154.4, "metric_delta": {"gen": 27648.0, "prompt": 123424.0, "pc_queries": 0.0, "pc_hits": 0.0, "preempt": 0.0, "drafts": 8266.0, "draft_tokens": 24798.0, "accepted": 19394.0, "running": 0.0}}, "long8k": {"level": 6, "sec": 134.5, "metric_delta": {"gen": 9216.0, "prompt": 169600.0, "pc_queries": 0.0, "pc_hits": 0.0, "preempt": 0.0, "drafts": 2307.0, "draft_tokens": 6921.0, "accepted": 6910.0, "running": 0.0}}, "long30k": {"level": 6, "sec": 309.1, "metric_delta": {"gen": 9216.0, "prompt": 432000.0, "pc_queries": 0.0, "pc_hits": 0.0, "preempt": 0.0, "drafts": 2305.0, "draft_tokens": 6915.0, "accepted": 6912.0, "running": 0.0}}}
- K: smoke {'tokens': 16, 'error': None, 'margins_present': True, 'usage_seen': True, 'text': '\n\n1 2 3 4 5 6 7 8'}; stages {"serial": {"sec": 355.8, "metric_delta": {"gen": 18432.0, "prompt": 92568.0, "pc_queries": 92568.0, "pc_hits": 24000.0, "preempt": 0.0, "drafts": 5397.0, "draft_tokens": 16191.0, "accepted": 13044.0, "running": 0.0}}, "churn": {"level": 14, "sec": 114.4, "metric_delta": {"gen": 27648.0, "prompt": 123424.0, "pc_queries": 152989.0, "pc_hits": 72000.0, "preempt": 0.0, "drafts": 8321.0, "draft_tokens": 24963.0, "accepted": 19335.0, "running": 0.0}}, "long8k": {"level": 6, "sec": 90.3, "metric_delta": {"gen": 9216.0, "prompt": 169600.0, "pc_queries": 380160.0, "pc_hits": 75200.0, "preempt": 0.0, "drafts": 2309.0, "draft_tokens": 6927.0, "accepted": 6911.0, "running": 0.0}}, "long30k": {"level": 6, "sec": 177.4, "metric_delta": {"gen": 9216.0, "prompt": 432000.0, "pc_queries": 2015936.0, "pc_hits": 206400.0, "preempt": 0.0, "drafts": 2304.0, "draft_tokens": 6912.0, "accepted": 6912.0, "running": 0.0}}}
