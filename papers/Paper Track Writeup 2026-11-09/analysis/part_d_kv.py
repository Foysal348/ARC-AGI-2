"""Part D: did 9 sessions x 104k fit the 738,142-token KV pool in NB5 FP8 (and K100Q for contrast)?
Sources: phase_*/duck/*/branch_result.json 'requests' (per-request server snapshot: kv, running, queue at send time;
prompt/completion tokens, start t, duration sec) and the server log Running/Waiting lines.
Live context of a request = prompt + completion, held from start t to t + sec.
"""
import glob, json, os, re, collections

ROOT = r"D:\ARC AGI 2\_scratch"
RUNS = {
    "W4_NB5_FP8": (os.path.join(ROOT, r"e25_gpu_out\q27-xhigh-e25-nb5-quality\q27x\phase_Q_FP8_K3\duck"),
                   os.path.join(ROOT, r"e25_gpu_out\q27-xhigh-e25-nb5-quality\q27x\engine\logs\server_merlin_fp8_mtp_pin100.log"), 9, 738142),
    "K100Q": (os.path.join(ROOT, r"e26_gpu_out\q27-xhigh-e26-k100-quality\q27x\phase_Q_K100_FP8_K3\duck"),
              os.path.join(ROOT, r"e26_gpu_out\q27-xhigh-e26-k100-quality\q27x\engine\logs\server_merlin_fp8_mtp_pin508.log"), 21, None),
}
HERE = os.path.dirname(os.path.abspath(__file__))
res = {}
for name, (duck, log, workers, kvtok) in RUNS.items():
    reqs = []
    for p in glob.glob(os.path.join(duck, "*", "branch_result.json")):
        b = json.load(open(p))
        for r in b.get("requests", []):
            if r.get("prompt") is None:
                continue
            reqs.append(r)
    q = [r for r in reqs if (r.get("queue") or 0) > 0]
    # live-context sweep
    ev = []
    for r in reqs:
        ctx = (r["prompt"] or 0) + (r.get("completion") or 0)
        ev.append((r["t"], ctx, 1))
        ev.append((r["t"] + (r.get("sec") or 0), -ctx, -1))
    ev.sort()
    cur = n = peak = peak_n = 0
    for t, d, k in ev:
        cur += d; n += k
        if cur > peak:
            peak, peak_n = cur, n
    maxctx = max((r["prompt"] or 0) + (r.get("completion") or 0) for r in reqs)
    logrows = []
    if os.path.exists(log) and os.path.getsize(log) > 0:
        for l in open(log, encoding="utf-8", errors="replace"):
            m = re.search(r"Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%", l)
            if m:
                logrows.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    kvcap = None
    if os.path.exists(log):
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", open(log, encoding="utf-8", errors="replace").read())
        kvcap = int(m.group(1).replace(",", "")) if m else None
    res[name] = {
        "requests_with_snapshot": len(reqs),
        "requests_sent_while_server_queue_gt0": len(q),
        "queue_values": collections.Counter(r["queue"] for r in q),
        "running_at_send_dist": dict(sorted(collections.Counter(r.get("running") for r in reqs).items(), key=lambda x: str(x[0]))),
        "kv_at_send_max": max((r.get("kv") or 0) for r in reqs),
        "max_single_request_context": maxctx,
        "peak_sum_live_context_tokens": peak, "requests_live_at_peak": peak_n,
        "kv_pool_tokens_log": kvcap, "workers": workers,
        "log_lines": len(logrows),
        "log_lines_waiting_gt0": sum(1 for r in logrows if r[1] > 0),
        "log_waiting_rows": [r for r in logrows if r[1] > 0][:20],
        "log_max_running": max((r[0] for r in logrows), default=None),
        "log_kv_usage_max_pct": max((r[2] for r in logrows), default=None),
    }
json.dump(res, open(os.path.join(HERE, "part_d_kv.json"), "w"), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
