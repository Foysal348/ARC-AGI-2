# E19 vLLM MTP / prefix-cache correctness: profile E19P1, run 20261027T173641Z

| arm | what | final | in-arm | churn (mode) | rows ok / planned | serial A/A ident/cmp (flips) | cache serial | churn vs serial | cont8k | cont30k | corrupt rows | MTP accept min / median | cache warm hits | launch s | KV tokens | block |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D | S1 exactly: MTP 3 + prefix caching ON, fp8 KV | **PASS** | PASS | PASS (uncalibrated) | 66/66 | 6/6 (0) | 6/6 (0) | 0/24 (0) | 6/6 (0) | 6/6 (0) | 0 | 0.575 / 0.795 | 18/18 | 495.1 | 337600 | 1600 |
| C | MTP OFF, prefix caching ON | **PASS** | PASS | PASS (uncalibrated) | 66/66 | 6/6 (0) | 6/6 (0) | 0/24 (0) | 6/6 (0) | 6/6 (0) | 0 | None / None | 18/18 | 290.0 | 376320 | 1568 |

## Reasons

- **D** (PASS): trailing Mamba block over generated tokens NOT_EXERCISED (no continuation hit reached into the server's generated tokens): no safety claim for that path
- **C** (PASS): none

## Secondary: token comparison against arm A (informational; MTP on vs off may differ at low margins)


Cells: identical / compared (large-margin flips). A large-margin flip = the first token divergence where a known top-1 minus top-2 logprob margin is > 1 nat. In-arm FAIL: corrupt rows with clean partners, cache-hit or continuation flips above the arm's serial A/A, MTP acceptance collapse. Churn FAIL: flips vs serial above the calibrators' (A / B) churn flips, or above the arm's own serial A/A when no calibrator ran ('uncalibrated'). Rows not run in the budget make an arm NOT_TESTED_BUDGET, never PASS.

Calibrators in this run: none (run src/e19_compare.py over the three accounts' bundles)

## Stages

- D: smoke {'tokens': 16, 'error': None, 'margins_present': True, 'usage_seen': True, 'text': '\n\n1 2 3 4 5 6 7 8'}; stages {"serial": {"sec": 368.9, "metric_delta": {"gen": 18432.0, "prompt": 92568.0, "pc_queries": 92568.0, "pc_hits": 17600.0, "preempt": 0.0, "drafts": 5550.0, "draft_tokens": 16650.0, "accepted": 12873.0, "running": 0.0}}, "churn": {"level": 14, "sec": 126.7, "metric_delta": {"gen": 27648.0, "prompt": 123424.0, "pc_queries": 152989.0, "pc_hits": 52800.0, "preempt": 0.0, "drafts": 8425.0, "draft_tokens": 25275.0, "accepted": 19216.0, "running": 0.0}}, "long8k": {"level": 6, "sec": 95.4, "metric_delta": {"gen": 9216.0, "prompt": 169600.0, "pc_queries": 411136.0, "pc_hits": 65600.0, "preempt": 0.0, "drafts": 2304.0, "draft_tokens": 6912.0, "accepted": 6906.0, "running": 0.0}}, "long30k": {"level": 6, "sec": 185.3, "metric_delta": {"gen": 9216.0, "prompt": 432000.0, "pc_queries": 2385920.0, "pc_hits": 196800.0, "preempt": 0.0, "drafts": 2304.0, "draft_tokens": 6912.0, "accepted": 6911.0, "running": 0.0}}}
- C: smoke {'tokens': 16, 'error': None, 'margins_present': True, 'usage_seen': True, 'text': '\n\n1 2 3 4 5 6 7 8'}; stages {"serial": {"sec": 727.0, "metric_delta": {"gen": 18432.0, "prompt": 92568.0, "pc_queries": 92568.0, "pc_hits": 28224.0, "preempt": 0.0, "drafts": null, "draft_tokens": null, "accepted": null, "running": 0.0}}, "churn": {"level": 14, "sec": 163.8, "metric_delta": {"gen": 27648.0, "prompt": 123424.0, "pc_queries": 159748.0, "pc_hits": 84672.0, "preempt": 0.0, "drafts": null, "draft_tokens": null, "accepted": null, "running": 0.0}}, "long8k": {"level": 6, "sec": 123.8, "metric_delta": {"gen": 9216.0, "prompt": 169600.0, "pc_queries": 446336.0, "pc_hits": 83104.0, "preempt": 0.0, "drafts": null, "draft_tokens": null, "accepted": null, "running": 0.0}}, "long30k": {"level": 6, "sec": 215.9, "metric_delta": {"gen": 9216.0, "prompt": 432000.0, "pc_queries": 2368576.0, "pc_hits": 211680.0, "preempt": 0.0, "drafts": null, "draft_tokens": null, "accepted": null, "running": 0.0}}}
