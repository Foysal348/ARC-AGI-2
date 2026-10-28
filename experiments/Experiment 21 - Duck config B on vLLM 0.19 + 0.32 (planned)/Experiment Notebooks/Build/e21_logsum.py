def _e15b_server_log_summary(text):
    # E15B #42 / #53 on vLLM (E21): vLLM's 10-s engine stats lines (rev 2: e21_stat_rows, every optional field incl.
    # Preemptions / Deferred) instead of SGLang decode lines: running-level histogram (lines with work), queue > 0 share,
    # generation throughput, prefix-cache hit rate (rolling), KV usage, MTP acceptance, preemptions, the startup KV line
    import re as _re
    rows = e21_stat_rows(text)
    work = [r for r in rows if r["running"] > 0 or r["waiting"] > 0]
    _hist, _hq, _tpl = {}, {}, {}
    for r in work:
        k = r["running"]
        _hist[k] = _hist.get(k, 0) + 1
        if r["waiting"] > 0:
            _hq[k] = _hq.get(k, 0) + 1
        _tpl.setdefault(k, []).append(r["gen_tps"])
    _tp = sorted(r["gen_tps"] for r in work)
    _acc = sorted(float(a) for a in _re.findall(r"Mean acceptance length: ([\d.]+)", text))
    _kv = _re.findall(r"GPU KV cache size: ([\d,]+) tokens", text)
    _pre = [r["prefix_pct"] for r in rows if r["prefix_pct"] is not None]
    _n = len(work)
    return {"decode_lines": _n, "running_hist": dict(sorted(_hist.items())),
            "running_hist_queue_gt0": dict(sorted(_hq.items())),
            "startup": {"kv_tokens_logged": int(_kv[0].replace(",", "")) if _kv else None},
            "accept_len_p50": _acc[len(_acc) // 2] if _acc else None,
            "gen_tok_s_p50_by_running": {k: sorted(v)[len(v) // 2] for k, v in sorted(_tpl.items())},
            "running_mean_lines": round(sum(k * v for k, v in _hist.items()) / _n, 2) if _n else None,
            "queue_gt0_line_share": round(sum(_hq.values()) / _n, 3) if _n else None,
            "gen_tok_s_p50": _tp[len(_tp) // 2] if _tp else None,
            "gen_tok_s_mean": round(sum(_tp) / len(_tp), 1) if _tp else None,
            "prefix_cache_hit_pct_last_rolling": _pre[-1] if _pre else None,
            "kv_usage_pct_max": max((r["kv_pct"] for r in rows), default=None),
            "preemptions_in_stats_lines": sum(r["preempt"] for r in rows),
            "preempt_lines": sum(1 for r in rows if r["preempt"]),
            "oom_lines": sum(1 for _l in text.splitlines() if "out of memory" in _l.lower()),
            "traceback_lines": sum(1 for _l in text.splitlines() if "Traceback" in _l)}
