# GPT audit 2 of the E17 bug-hunt restart bundle (2026-10-27), pasted by the user (verbatim text in the conversation transcript)

Saved summary of its claims (the full Bengali text is in the session transcript of 2026-10-27):
1. vLLM 0.19.0 MambaManager.find_longest_cache_hit() takes use_eagle but does not use it when picking the Mamba prefix-hit block; later PRs #43650, #48375 work on the trailing speculative block. Source: https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/v1/core/single_type_kv_cache_manager.py
2. vLLM issue #43559: ~20 pp accuracy drop with MTP + prefix caching (a later vLLM); #53912: empty / repeated punctuation.
3. Risks: MTP+hybrid prefix cache (HIGH), forced-close re-prefill (HIGH), FP8 KV scale 1.0 (HIGH, UNVERIFIED), hidden-rerun unfinished output (HIGH), E17 partial-evidence loss (HIGH).
4. Raw E17 trace_*.jsonl hold kv_scales events for 16 full-attention layers in each arm (the report's zero capture was a reporting gap).
5. GPU plan (60 min): manifest 0-5; arm A MTP3+cache ON 5-20; arm B MTP3+cache OFF 20-35; arm C target-only+cache ON 35-50; analysis 50-60. Then --no-async-scheduling as a discriminator if A vs B differ; FP8 vs BF16 KV after that.
6. Per-request log schema (prompt_tokens, cached_tokens, prefill_seconds, queue_seconds, generated_tokens, finish_reason); workload-aware drain deadline (required = pending_prompt_tokens / low prefill tps * 1.25 + 90 s).
7. LB-ish CV: same code path, coverage x conditional accuracy decomposition, paired comparisons, bootstrap CIs.
8. Ranking: 1 MTP3 + prefix cache; 2 re-prefill / drain coverage; 3 FP8 KV scales; 4 async scheduling race; 5 rerun-only notebook issue; 6 SGLang #43567/#43559.
