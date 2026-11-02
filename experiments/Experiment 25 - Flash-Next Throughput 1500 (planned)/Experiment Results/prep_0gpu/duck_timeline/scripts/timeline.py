"""Analysis 1: Duck occupancy / idle timeline for K100Q (numeric data only; run extract.py first).

Usage: python -I timeline.py <data_dir> <out_dir>
Writes: timeline_10s.csv, worker_budget.json, gaps.csv, analysis1_summary.json, plots (if matplotlib is present).
"""
import csv, json, math, statistics as st, sys, collections, bisect
from pathlib import Path

DATA = Path(sys.argv[1]); OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)
W0, W1 = 1793505683.4254444, 1793512283.8254445          # window start/end epoch (k100_quality_report inputs)
T = W1 - W0
WORKERS = 21
DT = 0.25                                                 # integration grid (s)


def f(x):
    if x in ("", "None", None):
        return None
    if x in ("True", "False"):
        return x == "True"
    try:
        return float(x)
    except ValueError:
        return x


def load(name):
    return [{k: f(v) for k, v in r.items()} for r in csv.DictReader(open(DATA / name, encoding="utf-8"))]


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def dist(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return {"n": 0}
    return {"n": len(xs), "mean": round(st.mean(xs), 3), "p50": round(pct(xs, 50), 3), "p90": round(pct(xs, 90), 3),
            "p99": round(pct(xs, 99), 3), "min": round(min(xs), 3), "max": round(max(xs), 3), "sum": round(sum(xs), 1)}


R = load("requests.csv")
B = load("branches.csv")
S = [r for r in load("srvstats.csv") if W0 < r["t"] <= W1 + 1]
P = [r for r in load("sampler.csv") if r.get("running") is not None]
sched = json.load(open(DATA / "sched_meta.json"))

# ---------------------------------------------------------------- per-branch state intervals (relative seconds)
rel = lambda t: t - W0
byb = collections.defaultdict(list)
for r in R:
    byb[(r["task"], int(r["branch"]))].append(r)
intervals = {"setup": [], "request": [], "gap": [], "tail": []}
branch_rows, gap_rows = [], []
for b in B:
    key = (b["task"], int(b["branch"]))
    rs = sorted([r for r in byb[key] if r["t1"] is not None], key=lambda r: r["i"])
    noreq = [r for r in byb[key] if r["t1"] is None]
    start = rel(b["t0"])
    end = min(rel(b["t0"] + b["run_sec"]), T)
    if not rs:
        continue
    intervals["setup"].append((start, rel(rs[0]["t0"]), key))
    for k, r in enumerate(rs):
        intervals["request"].append((rel(r["t0"]), min(rel(r["t1"]), T), key))
        if k + 1 < len(rs):
            g0, g1 = rel(r["t1"]), rel(rs[k + 1]["t0"])
            intervals["gap"].append((g0, g1, key))
            nxt = rs[k + 1]
            gap_rows.append({"task": key[0], "branch": key[1], "i": r["i"], "t_rel": round(g0, 2), "gap_sec": round(g1 - g0, 3),
                             "after_finish": r["finish"], "prev_completion": r["completion"], "next_prompt": nxt["prompt"],
                             "next_chars": nxt["m_chars"], "next_kind": nxt["kind"], "m_idle": nxt["m_idle"],
                             "next_req_sec": round(nxt["t1"] - nxt["t0"], 3), "next_completion": nxt["completion"]})
    last = min(rel(rs[-1]["t1"]), T)
    intervals["tail"].append((last, max(last, end), key))
    branch_rows.append({"task": key[0], "branch": key[1], "start": round(start, 2), "end": round(end, 2),
                        "setup": round(rel(rs[0]["t0"]) - start, 3), "tail": round(max(0, end - last), 3),
                        "n_req": len(rs), "noreq": len(noreq),
                        "req_sec": round(sum(min(rel(r["t1"]), T) - rel(r["t0"]) for r in rs), 2),
                        "gap_sec": round(sum(rel(rs[k + 1]["t0"]) - rel(rs[k]["t1"]) for k in range(len(rs) - 1)), 2),
                        "verify_sec_sum": b["verify_sec_sum"], "generated": b["generated_tokens"]})

# ---------------------------------------------------------------- fine grid counts
N = int(math.ceil(T / DT)) + 1
grid = {k: [0] * N for k in intervals}
for k, iv in intervals.items():
    for a, c, _ in iv:
        a, c = max(0.0, a), min(T, c)
        if c <= a:
            continue
        for j in range(int(a / DT), min(N, int(math.ceil(c / DT)))):
            # fraction of the cell covered
            lo, hi = j * DT, (j + 1) * DT
            cov = max(0.0, min(hi, c) - max(lo, a)) / DT
            grid[k][j] += cov
active = [sum(grid[k][j] for k in grid) for j in range(N)]
last_start = max(br["start"] for br in branch_rows)


def avg_over(k, a, c):
    j0, j1 = max(0, int(a / DT)), min(N, int(math.ceil(c / DT)))
    arr = grid[k] if k != "active" else active
    return sum(arr[j0:j1]) / max(1, j1 - j0)


def at(k, t):
    j = min(N - 1, max(0, int(t / DT)))
    return (grid[k] if k != "active" else active)[j]


# ---------------------------------------------------------------- exact heap (startable work) reconstruction
# scheduler semantics (arc2_duck.run_scheduler): all 30 attempt-1 items are queued at t=0 and popped at their start;
# a task's attempt-2 item is queued when its attempt 1 ends (stage2_items) and popped at its start.
brk = {(b_["task"], b_["branch"]): b_ for b_ in branch_rows}
push_pop = []
for (tk, bb), b_ in brk.items():
    if bb == 0:
        push_pop.append((0.0, b_["start"]))
    else:
        b0 = brk.get((tk, 0))
        push_pop.append((b0["end"] if b0 else 0.0, b_["start"]))
heap_cnt = [0] * N
for a, c in push_pop:
    for j in range(max(0, int(a / DT)), min(N, int(c / DT))):
        heap_cnt[j] += 1
first_empty = next((j * DT for j in range(N) if heap_cnt[j] == 0 and j * DT > 1.0), None)

# ---------------------------------------------------------------- worker-second budget
ws = {k: sum(grid[k]) * DT for k in grid}
ws["idle_no_branch"] = WORKERS * T - sum(active) * DT
# idle split: before the last branch start (work-dependency: b1 waits for its b0) vs after (end drain)
jL = int(last_start / DT)
idle_before = sum(WORKERS - active[j] for j in range(0, jL)) * DT
idle_after = sum(WORKERS - active[j] for j in range(jL, N)) * DT
idle_heap = sum(WORKERS - active[j] for j in range(N) if heap_cnt[j] > 0) * DT
idle_noheap_before = sum(WORKERS - active[j] for j in range(0, jL) if heap_cnt[j] == 0) * DT
budget = {"idle_while_work_queued_sec": round(idle_heap, 1),
          "idle_heap_empty_before_last_start_sec": round(idle_noheap_before, 1),
          "first_heap_empty_rel_sec": first_empty,
          "window_sec": round(T, 1), "workers": WORKERS, "worker_seconds": round(WORKERS * T, 1),
          "states_sec": {k: round(v, 1) for k, v in ws.items()},
          "states_share": {k: round(v / (WORKERS * T), 4) for k, v in ws.items()},
          "idle_before_last_start_sec": round(idle_before, 1), "idle_after_last_start_sec": round(idle_after, 1),
          "last_branch_start_rel_sec": round(last_start, 1), "max_active": round(max(active), 2),
          "kv_hold_sec_scheduler": sched.get("kv_hold_sec"), "kv_holds": sched.get("kv_holds")}

# ---------------------------------------------------------------- in-request split via the 15 s sampler (Little's law)
samp = []
for p in P:
    t = rel(p["t"])
    if not (0 < t < T):
        continue
    nreq = at("request", t)
    samp.append({"t": t, "nreq": nreq, "inflight": p["inflight"], "running": p["running"], "waiting": p["waiting"],
                 "active": at("active", t), "gap": at("gap", t), "setup": at("setup", t), "tail": at("tail", t)})
m = lambda k: st.mean(s[k] for s in samp)
guard_avg = st.mean(max(0.0, s["nreq"] - s["inflight"]) for s in samp)
http_avg = st.mean(max(0.0, s["inflight"] - s["running"] - s["waiting"]) for s in samp)
inreq_split = {"samples": len(samp), "mean_client_requests": round(m("nreq"), 3), "mean_inflight_after_guard": round(m("inflight"), 3),
               "mean_server_running": round(m("running"), 3), "mean_server_waiting": round(m("waiting"), 3),
               "mean_in_guard_tokenize": round(guard_avg, 3), "mean_in_http_api_layer": round(http_avg, 3),
               "guard_worker_sec_est": round(guard_avg * T, 1), "http_worker_sec_est": round(http_avg * T, 1),
               "samples_nreq_minus_inflight_negative": sum(1 for s in samp if s["nreq"] - s["inflight"] < -0.5),
               "samples_inflight_gt_running_plus_waiting": sum(1 for s in samp if s["inflight"] > s["running"] + s["waiting"])}
# per-request means for comparison with Prometheus means (e2e 142.46 s, ttft 2.01 s, queue 0.17 s, prefill 1.68 s)
rdur = [min(rel(r["t1"]), T) - rel(r["t0"]) for r in R if r["t1"] is not None]
inreq_split["client_request_mean_sec"] = round(st.mean(rdur), 3)
inreq_split["prometheus_e2e_mean_sec"] = 142.46
inreq_split["client_minus_server_e2e_mean_sec"] = round(st.mean(rdur) - 142.46, 3)
ok = [r for r in R if r["t1"] is not None and not r["error"]]
inreq_split["client_sum_sec_successful"] = round(sum(r["t1"] - r["t0"] for r in ok), 1)
inreq_split["successful_requests"] = len(ok)
inreq_split["prometheus_e2e_sum_sec"] = round(762 * 142.46, 1)
inreq_split["prometheus"] = {"ttft_mean": 2.01, "queue_mean": 0.17, "prefill_mean": 1.68, "decode_mean": 140.48, "count": 762}

# ---------------------------------------------------------------- 10 s interval table + R(n)
rows10 = []
for i, s in enumerate(S):
    t = rel(s["t"])
    a = max(0.0, t - 10.0)
    prev_run = S[i - 1]["running"] if i > 0 else s["running"]
    rows10.append({"t": round(t, 1), "gen_tps": s["gen_tps"], "prompt_tps": s["prompt_tps"], "running": s["running"],
                   "running_prev": prev_run, "running_avg": (s["running"] + prev_run) / 2, "waiting": s["waiting"],
                   "kv_pct": s["kv_pct"], "al": s.get("al"),
                   "active": round(avg_over("active", a, t), 3), "req": round(avg_over("request", a, t), 3),
                   "gap": round(avg_over("gap", a, t), 3), "setup": round(avg_over("setup", a, t), 3),
                   "tail": round(avg_over("tail", a, t), 3)})
stable = collections.defaultdict(list)
for r in rows10:
    if r["running"] == r["running_prev"]:
        stable[int(r["running"])].append(r)
Rn = {n: st.mean(x["gen_tps"] for x in v) for n, v in stable.items() if len(v) >= 3}
Rn_pure = {n: st.mean(x["gen_tps"] for x in v if x["prompt_tps"] == 0) for n, v in stable.items()
           if sum(1 for x in v if x["prompt_tps"] == 0) >= 2}
# least-squares fit gen = a + b n + c n^2 on stable intervals (smooth R(n))
X = [(x["running"], x["gen_tps"]) for v in stable.values() for x in v]
import itertools


def lsq(pts, deg=2):
    # normal equations
    n = deg + 1
    A = [[sum(p[0] ** (i + j) for p in pts) for j in range(n)] for i in range(n)]
    y = [sum(p[1] * p[0] ** i for p in pts) for i in range(n)]
    # gaussian elimination
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(A[r][c]))
        A[c], A[piv] = A[piv], A[c]; y[c], y[piv] = y[piv], y[c]
        for r in range(n):
            if r != c:
                fac = A[r][c] / A[c][c]
                A[r] = [A[r][k] - fac * A[c][k] for k in range(n)]
                y[r] -= fac * y[c]
    return [y[i] / A[i][i] for i in range(n)]


coef = lsq(X, 2)
Rfit = lambda n: coef[0] + coef[1] * n + coef[2] * n * n
R21 = Rn.get(21)
R21_all = st.mean(r["gen_tps"] for r in rows10 if r["running"] == 21)
R21_med = st.median(r["gen_tps"] for r in rows10 if r["running"] == 21)

# prefill stall: gen vs prompt tps at stable 21
s21 = stable[21]
pts = [(x["prompt_tps"], x["gen_tps"]) for x in s21]
lin = lsq(pts, 1)
prefill_cost_gen_per_prompt_tok = -lin[1]           # generated tokens lost per fresh-prefill token at 21 running
# also across all stable intervals with a per-n intercept: gen = R_n + b * prompt
num = den = 0.0
for n, v in stable.items():
    if len(v) < 3:
        continue
    mg = st.mean(x["gen_tps"] for x in v); mp = st.mean(x["prompt_tps"] for x in v)
    for x in v:
        num += (x["prompt_tps"] - mp) * (x["gen_tps"] - mg); den += (x["prompt_tps"] - mp) ** 2
b_within = num / den if den else 0.0

# ---------------------------------------------------------------- loss decomposition vs the all-21 level
total_gen = sum(r["gen_tps"] for r in rows10) * 10.0
mean_rate = total_gen / T
ref = None
causes = ["idle_work_queued", "idle_dependency", "idle_drain", "setup", "gap", "tail", "client_in_request", "server_waiting",
          "unattributed"]
loss = {c: 0.0 for c in causes}
loss["prefill_and_other_residual"] = 0.0
occ_slots = {c: 0.0 for c in causes}
ref = Rfit(21)
for r in rows10:
    nbar = min(21.0, r["running_avg"])
    occ = max(0.0, Rfit(21) - Rfit(nbar))
    t = r["t"]
    a = max(0.0, t - 10.0)
    j0, j1 = int(a / DT), min(N, int(math.ceil(t / DT)))
    idle_q = sum(WORKERS - active[j] for j in range(j0, j1) if heap_cnt[j] > 0) / max(1, j1 - j0)
    idle_e = sum(WORKERS - active[j] for j in range(j0, j1) if heap_cnt[j] == 0) / max(1, j1 - j0)
    c = {"idle_work_queued": idle_q, "idle_dependency": idle_e if t <= last_start else 0.0,
         "idle_drain": idle_e if t > last_start else 0.0,
         "setup": r["setup"], "gap": r["gap"], "tail": r["tail"],
         "client_in_request": max(0.0, r["req"] - nbar - r["waiting"]), "server_waiting": r["waiting"]}
    tot = sum(c.values())
    missing = max(0.0, 21.0 - nbar)
    c["unattributed"] = max(0.0, missing - tot)
    tot = sum(c.values())
    for k in causes:
        occ_slots[k] += c[k] * 10.0
        if tot > 0:
            loss[k] += occ * c[k] / tot * 10.0
    loss["prefill_and_other_residual"] += (Rfit(nbar) - r["gen_tps"]) * 10.0
loss_total = ref * T - total_gen
decomp = {"window_sec": round(T, 1), "generated_tokens_log": round(total_gen), "window_mean_tok_s": round(mean_rate, 1),
          "reference_all21_stable_mean_tok_s": round(ref, 1), "all21_lines_mean_tok_s": round(R21_all, 1),
          "all21_lines_median_tok_s": round(R21_med, 1),
          "median_all_lines_tok_s": round(st.median(r["gen_tps"] for r in rows10), 1),
          "Rfit_coef": [round(c_, 4) for c_ in coef], "Rfit_21": round(Rfit(21), 1),
          "R_n_stable_mean": {str(n): round(v, 1) for n, v in sorted(Rn.items())},
          "R_n_stable_pure_decode": {str(n): round(v, 1) for n, v in sorted(Rn_pure.items())},
          "lost_tokens_vs_reference": round(loss_total),
          "lost_tokens_by_cause": {k: round(v) for k, v in loss.items()},
          "lost_tok_s_by_cause": {k: round(v / T, 1) for k, v in loss.items()},
          "missing_slot_seconds_by_cause": {k: round(v, 1) for k, v in occ_slots.items()},
          "prefill_regression_at21": {"intercept_pure_decode_tok_s": round(lin[0], 1), "slope_gen_per_prompt_tps": round(lin[1], 4),
                                       "n": len(pts)},
          "prefill_regression_within_n_slope": round(b_within, 4)}
decomp["check_sum_causes_minus_total"] = round(sum(loss.values()) - loss_total)

# prefill stall cost over the run: gen tokens lost = |slope| * fresh prefill tokens
fresh = sum(r["prompt_tps"] for r in rows10) * 10.0
decomp["fresh_prefill_tokens_log"] = round(fresh)
decomp["prefill_stall_tokens_est"] = round(prefill_cost_gen_per_prompt_tok * fresh)
decomp["prefill_stall_tok_s_est"] = round(prefill_cost_gen_per_prompt_tok * fresh / T, 1)
decomp["prefill_stall_tokens_est_within_n"] = round(-b_within * fresh)

# ---------------------------------------------------------------- running < workers, zero running, segments
seg, cur = [], None
for r in rows10:
    low = r["running"] < WORKERS
    if low and cur is None:
        cur = [r["t"] - 10, r["t"], r["running"]]
    elif low:
        cur[1] = r["t"]; cur[2] = min(cur[2], r["running"])
    elif cur is not None:
        seg.append(cur); cur = None
if cur is not None:
    seg.append(cur)
occupancy = {"lines": len(rows10), "frac_running_lt_workers": round(sum(1 for r in rows10 if r["running"] < WORKERS) / len(rows10), 4),
             "frac_running_eq_workers": round(sum(1 for r in rows10 if r["running"] == WORKERS) / len(rows10), 4),
             "zero_running_lines": sum(1 for r in rows10 if r["running"] == 0),
             "min_running": min(r["running"] for r in rows10),
             "sampler_zero_running": sum(1 for p in samp if p["running"] == 0),
             "sampler_min_running": min(p["running"] for p in samp),
             "mean_running_log": round(st.mean(r["running"] for r in rows10), 2),
             "running_hist": dict(sorted(collections.Counter(int(r["running"]) for r in rows10).items())),
             "segments_below_workers": [[round(a), round(c), int(mn)] for a, c, mn in seg],
             "frac_below_before_last_start": round(sum(1 for r in rows10 if r["running"] < WORKERS and r["t"] <= last_start)
                                                   / max(1, sum(1 for r in rows10 if r["t"] <= last_start)), 4),
             "mean_running_before_last_start": round(st.mean(r["running"] for r in rows10 if r["t"] <= last_start), 2),
             "mean_running_after_last_start": round(st.mean(r["running"] for r in rows10 if r["t"] > last_start), 2),
             "mean_active_before_last_start": round(st.mean(r["active"] for r in rows10 if r["t"] <= last_start), 2),
             "mean_active_after_last_start": round(st.mean(r["active"] for r in rows10 if r["t"] > last_start), 2),
             "tok_s_before_last_start": round(st.mean(r["gen_tps"] for r in rows10 if r["t"] <= last_start), 1),
             "tok_s_after_last_start": round(st.mean(r["gen_tps"] for r in rows10 if r["t"] > last_start), 1)}
# the first time 21 is reached (ramp)
first21 = next((r["t"] for r in rows10 if r["running"] == WORKERS), None)
occupancy["first_line_at_21_sec"] = first21

# ---------------------------------------------------------------- gap (tool + client) distributions
gaps = [g["gap_sec"] for g in gap_rows]
# client assembly vs context size: linear fit gap = a + b * chars (MB)
gp = [(g["next_chars"] / 1e6, g["gap_sec"]) for g in gap_rows if g["next_chars"]]
gfit = lsq(gp, 1)
gp_low = [(x, y) for x, y in gp if y <= pct([y for _, y in gp], 90)]
gfit_low = lsq(gp_low, 1)
# lower envelope: per chars decile, the 10th percentile gap
dec = sorted(gp)
env = []
for q in range(10):
    part = dec[q * len(dec) // 10:(q + 1) * len(dec) // 10]
    env.append([round(st.mean(x for x, _ in part), 3), round(pct([y for _, y in part], 10), 3), round(pct([y for _, y in part], 50), 3)])
first_gap = [g["gap_sec"] for g in gap_rows if g["i"] == 0]
# request durations (client) and the time from response to the next request start / next server admission
setup = [b_["setup"] for b_ in branch_rows]
tail = [b_["tail"] for b_ in branch_rows]
gapstats = {"tool_plus_client_gap": dist(gaps), "gap_first_after_request0": dist(first_gap),
            "gap_vs_context_fit": {"intercept_sec": round(gfit[0], 3), "sec_per_MB_chars": round(gfit[1], 3)},
            "gap_vs_context_fit_p90_trimmed": {"intercept_sec": round(gfit_low[0], 3), "sec_per_MB_chars": round(gfit_low[1], 3)},
            "gap_lower_envelope_by_chars_decile": env,
            "setup_before_first_request": dist(setup), "tail_after_last_request": dist(tail),
            "client_request_duration": dist(rdur),
            "next_chars_MB": dist([g["next_chars"] / 1e6 for g in gap_rows if g["next_chars"]]),
            "idle_sec_harness_field": dist([g["m_idle"] for g in gap_rows if g["m_idle"] is not None])}
# time from tool result to next request's arrival at the server = gap's post-tool part + guard tokenize + HTTP.
# guard + HTTP per request from Little's law: (guard + http worker-sec) / number of requests
nreq = len(rdur)
per_req_guard = guard_avg * T / nreq
per_req_http = http_avg * T / nreq
gapstats["per_request_guard_tokenize_mean_sec_est"] = round(per_req_guard, 3)
gapstats["per_request_http_api_mean_sec_est"] = round(per_req_http, 3)
gapstats["per_request_server_queue_mean_sec_prometheus"] = 0.17

# greedy per-worker assignment (threads are interchangeable; any feasible assignment gives the same totals)
free_at = [0.0] * WORKERS
assign = collections.defaultdict(list)
for b_ in sorted(branch_rows, key=lambda x: x["start"]):
    cand = [k for k in range(WORKERS) if free_at[k] <= b_["start"] + 1e-6]
    w_ = min(cand, key=lambda k: free_at[k]) if cand else min(range(WORKERS), key=lambda k: free_at[k])
    assign[w_].append(b_); free_at[w_] = b_["end"]
per_req_overhead = (guard_avg + http_avg) * T / max(1, len(rdur))
worker_rows = []
for w_ in range(WORKERS):
    bl = assign[w_]
    req = sum(x["req_sec"] for x in bl); nq = sum(x["n_req"] for x in bl)
    cli = nq * per_req_overhead
    srv_wait = nq * 0.17
    worker_rows.append({"worker": w_, "attempts": len(bl), "requests": nq,
                        "gpu_in_server_sec": round(req - cli - srv_wait, 1),
                        "client_in_request_sec_est": round(cli, 1), "server_queue_sec_est": round(srv_wait, 1),
                        "tool_plus_assembly_gap_sec": round(sum(x["gap_sec"] for x in bl), 1),
                        "setup_tail_sec": round(sum(x["setup"] + x["tail"] for x in bl), 1),
                        "idle_sec": round(T - sum(x["end"] - x["start"] for x in bl), 1),
                        "last_end_sec": round(max(x["end"] for x in bl), 1) if bl else 0})
with open(OUT / "per_worker.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(worker_rows[0].keys())); wr.writeheader(); wr.writerows(worker_rows)
with open(OUT / "heap_and_active_1s.csv", "w", newline="") as fh:
    wr = csv.writer(fh); wr.writerow(["t", "active_attempts", "in_request", "in_gap", "heap_items"])
    step = int(1.0 / DT)
    for j in range(0, N, step):
        wr.writerow([round(j * DT, 1), round(active[j], 2), round(grid["request"][j], 2), round(grid["gap"][j], 2), heap_cnt[j]])

summary = {"per_worker": worker_rows, "budget": budget, "in_request_split": inreq_split, "occupancy": occupancy, "decomposition": decomp,
           "gaps": gapstats, "n_requests": nreq, "n_gaps": len(gaps), "n_branches": len(branch_rows)}

# ---------------------------------------------------------------- ready-queue estimate
# (1) prepare-next-during-decode: removes client assembly + guard + HTTP from the critical path. Upper bound = those
#     slot-seconds converted at the marginal rate; the post-tool assembly share of the gap is bounded by the fit.
# (2) ready-for-GPU queue separate from waiting-for-tool (oversubscription: more branches than GPU slots).
marg = (Rfit(21) - Rfit(20))
avg_per_stream_21 = Rfit(21) / 21
ready = {"marginal_tok_s_per_stream_at_20_21": round(marg, 2), "avg_tok_s_per_stream_at_21": round(avg_per_stream_21, 2)}
for k in causes:
    ready[f"lost_tok_s_{k}"] = round(loss[k] / T, 1)
ready["note"] = "see report for interpretation"
summary["ready_queue"] = ready

json.dump(summary, open(OUT / "analysis1_summary.json", "w"), indent=1)
json.dump(budget, open(OUT / "worker_budget.json", "w"), indent=1)
with open(OUT / "timeline_10s.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows10[0].keys())); wr.writeheader(); wr.writerows(rows10)
with open(OUT / "gaps.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(gap_rows[0].keys())); wr.writeheader(); wr.writerows(gap_rows)
with open(OUT / "branches_timeline.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(branch_rows[0].keys())); wr.writeheader(); wr.writerows(branch_rows)
with open(OUT / "sampler_split.csv", "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(samp[0].keys())); wr.writeheader(); wr.writerows(samp)
print(json.dumps(summary, indent=1))
