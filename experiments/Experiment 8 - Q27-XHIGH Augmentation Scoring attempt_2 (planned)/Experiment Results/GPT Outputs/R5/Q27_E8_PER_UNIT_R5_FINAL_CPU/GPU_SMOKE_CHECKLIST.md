# Q27 E8 R5 — GPU preflight gates (no GPU verification has occurred)

**Do not equate CPU/fake-engine PASS with GPU/hidden GO.**

1. First GPU run must be a **small EVAL smoke test**, not the hidden 259-unit full run. Save complete stdout/stderr, `run_report.json`, `e8_per_unit_report.json`, `vllm_metrics.json`, and corresponding model token generation traces. The E7 fallback `submission.json` is produced per unit.
2. Confirm model, 4×L4 configuration, MTP3, FP8-KV and pinned vLLM 0.19.0 initialize without changing E7 primary inference parameters. Record available GPU memory, peak allocated/reserved per rank; score requests may allocate full-vocabulary logit/log-softmax buffers that normal startup profiling misses.
3. Confirm `SamplingParams(prompt_logprobs=1)` returns **chosen-token** logprobs for each of the eight donor views, with no missing positions. Inspect `boundary_mismatch_views`, `scoring_skipped_unverified_alignment`, `score_request_prompt_tokens`, and `scorer_disabled_reason`. Do not loosen strict token alignment without measured evidence.
4. Record peak memory and scheduling latency with **1 in-flight** scorer request. Existing `E8_MAX_SCORE_REQUEST_TOKENS = MAX_MODEL_LEN-8` checks context length, **NOT memory safety**. An OOM inside `engine.step()` may prevent Python cleanup; do not start with a full hidden run.
5. Compare primary-only E7 against R5 using the SAME evaluation units: completed primary units, pass@1, primary pass@2, mean tokens/sec, preemptions, contention steps, wall time; unchanged deadlines do **not** mean identical runtime/throughput.
6. Measure *unfiltered and context-filtered* reasoning candidate oracle coverage where full thinking traces are available. Check gained and lost pass@2 hits, units with 0/1/2+ novel candidates, scorer coverage, false positives from scratch grids, and exact final selected candidates.
7. Only after memory, likelihood and paired performance evidence, decide whether to deploy hidden. Do not assume 30-minute scorer throughput, 4×L4 VRAM safety or positive pass@2 gain.

No optional LoRA, extra model, donor DFS or undocumented primary time reservation is introduced in R5.
