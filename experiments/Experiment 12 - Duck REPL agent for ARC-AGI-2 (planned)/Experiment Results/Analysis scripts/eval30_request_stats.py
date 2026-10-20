"""E12 SGLang EVAL30 run 1: per-request prompt / completion / cache distribution from every duck/*/branch_result.json
(usage list = one entry per /v1/chat/completions request, as recorded by the notebook). Run: python -I this.py <q27x dir>"""
import json, glob, os, sys, statistics as st
d = sys.argv[1]
R = []
for f in sorted(glob.glob(os.path.join(d, "duck", "*", "branch_result.json"))):
    b = json.load(open(f, encoding="utf-8"))
    for i, x in enumerate(b["usage"]):
        R.append(dict(x, task=b["task_id"], br=b["branch"], i=i))
def pct(v, q): v = sorted(v); return v[min(len(v) - 1, int(q * len(v)))]
pr = [r["prompt"] for r in R]; co = [r["completion"] for r in R]
cf = [(r["cached"] or 0) / r["prompt"] for r in R if r["prompt"]]
out = {
    "n_branch_files": len(glob.glob(os.path.join(d, "duck", "*", "branch_result.json"))),
    "n_requests": len(R),
    "prompt_tokens": {"p10": pct(pr, .1), "p25": pct(pr, .25), "p50": st.median(pr), "p75": pct(pr, .75), "p90": pct(pr, .9), "p95": pct(pr, .95), "max": max(pr), "sum": sum(pr)},
    "completion_tokens": {"p50": st.median(co), "p90": pct(co, .9), "max": max(co), "at_cap_12288": sum(c >= 12288 for c in co), "sum": sum(co)},
    "cached_fraction": {"p10": pct(cf, .1), "p25": pct(cf, .25), "p50": st.median(cf), "p75": pct(cf, .75), "share_below_0.5": sum(c < .5 for c in cf) / len(cf)},
    "uncached_prefill_tokens": sum(r["prompt"] - (r["cached"] or 0) for r in R),
    "later_requests_below_50pct_cache": sum(1 for r in R if r["i"] > 0 and (r["cached"] or 0) < .5 * r["prompt"]),
    "later_requests": sum(1 for r in R if r["i"] > 0),
}
print(json.dumps(out, indent=1))
