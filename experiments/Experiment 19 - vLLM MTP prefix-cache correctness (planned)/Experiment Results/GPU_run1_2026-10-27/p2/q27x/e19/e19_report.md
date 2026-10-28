# E19 vLLM MTP / prefix-cache correctness: profile E19P2, run 20261027T173154Z

| arm | what | final | in-arm | churn (mode) | rows ok / planned | serial A/A ident/cmp (flips) | cache serial | churn vs serial | cont8k | cont30k | corrupt rows | MTP accept min / median | cache warm hits | launch s | KV tokens | block |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E | S1 + --no-async-scheduling | **PASS** | PASS | PASS (calibrated) | 66/66 | 6/6 (0) | 6/6 (0) | 0/24 (0) | 6/6 (0) | 6/6 (0) | 0 | 0.57 / 0.763 | 18/18 | 500.1 | 337600 | 1600 |
| A | reference: MTP OFF, prefix caching OFF | **PASS** | PASS | PASS (uncalibrated) | 60/60 | 6/6 (0) | - | 0/24 (0) | 5/6 (0) | 6/6 (0) | 0 | None / None | 0/18 | 285.0 | 376320 | 1568 |

## Reasons

- **E** (PASS): trailing Mamba block over generated tokens NOT_EXERCISED (no continuation hit reached into the server's generated tokens): no safety claim for that path
- **A** (PASS): none

## Secondary: token comparison against arm A (informational; MTP on vs off may differ at low margins)

- E: churn/churn 0/24 (0); long30k/cont 6/6 (0); long30k/nat 6/6 (0); long8k/cont 5/6 (1); long8k/nat 6/6 (0); serial/cold 0/6 (0); serial/warm 0/6 (0)

Cells: identical / compared (large-margin flips). A large-margin flip = the first token divergence where a known top-1 minus top-2 logprob margin is > 1 nat. In-arm FAIL: corrupt rows with clean partners, cache-hit or continuation flips above the arm's serial A/A, MTP acceptance collapse. Churn FAIL: flips vs serial above the calibrators' (A / B) churn flips, or above the arm's own serial A/A when no calibrator ran ('uncalibrated'). Rows not run in the budget make an arm NOT_TESTED_BUDGET, never PASS.

Calibrators in this run: A

## Stages

- E: smoke {'tokens': 16, 'error': None, 'margins_present': True, 'usage_seen': True, 'text': '\n\n1 2 3 4 5 6 7 8'}; stages {"serial": {"sec": 390.7, "metric_delta": {"gen": 18432.0, "prompt": 92568.0, "pc_queries": 92568.0, "pc_hits": 17600.0, "preempt": 0.0, "drafts": 5598.0, "draft_tokens": 16794.0, "accepted": 12849.0, "running": 0.0}}, "churn": {"level": 14, "sec": 131.1, "metric_delta": {"gen": 27648.0, "prompt": 123424.0, "pc_queries": 161443.0, "pc_hits": 52800.0, "preempt": 0.0, "drafts": 8661.0, "draft_tokens": 25983.0, "accepted": 18987.0, "running": 0.0}}, "long8k": {"level": 6, "sec": 96.7, "metric_delta": {"gen": 9216.0, "prompt": 169600.0, "pc_queries": 464768.0, "pc_hits": 65600.0, "preempt": 0.0, "drafts": 2309.0, "draft_tokens": 6927.0, "accepted": 6909.0, "running": 0.0}}, "long30k": {"level": 6, "sec": 187.5, "metric_delta": {"gen": 9216.0, "prompt": 432000.0, "pc_queries": 2349376.0, "pc_hits": 196800.0, "preempt": 0.0, "drafts": 2304.0, "draft_tokens": 6912.0, "accepted": 6910.0, "running": 0.0}}}
- A: smoke {'tokens': 16, 'error': None, 'margins_present': True, 'usage_seen': True, 'text': '\n\n1 2 3 4 5 6 7 8'}; stages {"serial": {"sec": 499.5, "metric_delta": {"gen": 12288.0, "prompt": 61712.0, "pc_queries": 0.0, "pc_hits": 0.0, "preempt": 0.0, "drafts": null, "draft_tokens": null, "accepted": null, "running": 0.0}}, "churn": {"level": 14, "sec": 214.3, "metric_delta": {"gen": 27648.0, "prompt": 123424.0, "pc_queries": 0.0, "pc_hits": 0.0, "preempt": 0.0, "drafts": null, "draft_tokens": null, "accepted": null, "running": 0.0}}, "long8k": {"level": 6, "sec": 173.3, "metric_delta": {"gen": 9216.0, "prompt": 169600.0, "pc_queries": 0.0, "pc_hits": 0.0, "preempt": 0.0, "drafts": null, "draft_tokens": null, "accepted": null, "running": 0.0}}, "long30k": {"level": 6, "sec": 346.2, "metric_delta": {"gen": 9216.0, "prompt": 432000.0, "pc_queries": 0.0, "pc_hits": 0.0, "preempt": 0.0, "drafts": null, "draft_tokens": null, "accepted": null, "running": 0.0}}}
