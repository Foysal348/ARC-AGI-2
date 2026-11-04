"""Tool-gap and tool-result -> server-arrival estimates (numeric only). Run after timeline.py.
Usage: python -I tool_to_server.py <out1_dir>
The harness records no tool start/end timestamps, so the gap t1(k) -> t0(k+1) holds tool execution + client turn
assembly. Context-proportional assembly is estimated from the gap vs serialized-chars fit (p90-trimmed slope);
the guard /tokenize and HTTP/API-layer times are Little's-law means from the 15 s sampler (per request).
"""
import csv, json, math, statistics as st, sys
from pathlib import Path

O = Path(sys.argv[1])
A = json.load(open(O / "analysis1_summary.json"))
G = list(csv.DictReader(open(O / "gaps.csv")))
g = A["gaps"]
slope = g["gap_vs_context_fit_p90_trimmed"]["sec_per_MB_chars"]
guard = g["per_request_guard_tokenize_mean_sec_est"]
http = g["per_request_http_api_mean_sec_est"]


def pct(xs, p):
    xs = sorted(xs); k = (len(xs) - 1) * p / 100.0; lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def d(xs):
    return {"n": len(xs), "mean": round(st.mean(xs), 3), "p50": round(pct(xs, 50), 3), "p90": round(pct(xs, 90), 3),
            "p99": round(pct(xs, 99), 3), "max": round(max(xs), 3)}


rows = [r for r in G if r["next_chars"] not in ("", "None")]
gap = [float(r["gap_sec"]) for r in rows]
asm = [slope * float(r["next_chars"]) / 1e6 for r in rows]
tool_est = [max(0.0, a - b) for a, b in zip(gap, asm)]
central = [a + guard + http / 2 for a in asm]            # tool result -> request reaches the API server (central)
upper = [x + guard + http for x in gap]                  # previous response -> engine-side arrival if the tool took 0 s
out = {"gap_tool_plus_assembly": d(gap), "assembly_est": d(asm), "tool_exec_est": d(tool_est),
       "tool_result_to_server_arrival_central": d(central), "prev_response_to_server_upper": d(upper),
       "lower_bound_guard_only_mean": guard, "inputs": {"slope_sec_per_MB": slope, "guard_mean": guard, "http_api_mean": http}}
json.dump(out, open(O / "tool_to_server.json", "w"), indent=1)
print(json.dumps(out, indent=1))
