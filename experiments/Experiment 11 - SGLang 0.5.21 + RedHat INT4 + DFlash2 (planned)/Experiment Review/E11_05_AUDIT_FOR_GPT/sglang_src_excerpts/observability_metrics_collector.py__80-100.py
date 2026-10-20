# sglang 0.5.21 observability/metrics_collector.py lines 80-100
    num_grammar_queue_reqs: int = 0
    gen_throughput: float = 0.0
    cache_hit_rate: float = 0.0
    decode_sum_seq_lens: int = 0

    # Memory pool usage ratios (0.0–1.0).
    # Each pool tracks: used = total - available - evictable, usage = used / total.
    #
    # token_usage:      max(full, swa, mamba) — the bottleneck across all pools.
    #                   FIXME: misleadingly named "token_usage"; rename requires API deprecation.
    # full_token_usage: full-attention KV cache pool usage (always active).
    # swa_token_usage:  sliding-window attention KV cache pool usage (hybrid SWA models only, e.g. Gemma2).
    # mamba_usage:      Mamba SSM state pool usage (hybrid SSM models only, e.g. Jamba).
    token_usage: float = 0.0
    full_token_usage: float = 0.0
    swa_token_usage: float = 0.0
    mamba_usage: float = 0.0

    # Absolute token counts for the full-attention KV cache pool.
    # Invariant: kv_available_tokens + kv_evictable_tokens + kv_used_tokens <= max_total_num_tokens
    # (the gap accounts for protected/session-held tokens not exposed here).
