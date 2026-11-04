"""Compute every paper number that score_all.py does not print, and write the figure data for the LaTeX draft.

usage: python -B scripts/paper_stats.py            (run from the paper folder, after scripts/score_all.py)
reads only; writes data/paper_numbers.json, data/lb_by_notebook.csv and paper/figures/data/*.dat (pgfplots tables),
prints a summary.

Sources:
  - the two runs with pool data (arc-2026d-with-codex v1 and v2): their run logs' stdout
      EXPERIMENT 1 "GT Present cases" table  -> rank of the answer among the unique candidates (probmul / kgmon)
      EXPERIMENT 18 "Parse Audit Summary" table -> per test input: answer in pool, fallback, candidates, unique, avg_sim
      "FALLBACK_SAVE <bk>: <reason>" lines -> why a placeholder grid was written
    and their submission.json (the pass@2 result actually submitted);
  - the six full 120-puzzle runs: submission.json, or for the two provided notebooks the printed run_dual_selection set
    (same rule as score_all.py).
avg_sim in EXPERIMENT 18 is the mean cell agreement with the answer over ALL candidates of a test input; candidates
whose shape differs from the answer count as 0 (notebook cell 30). Classes use the notebook's own cut points
(> 90 close, > 50 partial, else far); the fallback test inputs are counted separately.
"""
import itertools, json, os, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOL = json.load(open(os.path.join(ROOT, "data", "arc-agi_evaluation_solutions.json")))
UNITS = [(t, i) for t in sorted(SOL) for i in range(len(SOL[t]))]
FIG = os.path.join(ROOT, "paper", "figures", "data")

POOL_RUNS = {  # stdout extract, submission, Kaggle JSON event log (timestamps)
    "v2": ("source notebook arc-2026d-with-codex/output/arc-2026d-with-codex_stdout.txt",
           "source notebook arc-2026d-with-codex/output/submission.json",
           "source notebook arc-2026d-with-codex/output/arc-2026d-with-codex.log"),
    "v1": ("kaggle versions arc-2026d-with-codex/v1/output/arc-2026d-with-codex_v1_stdout.txt",
           "kaggle versions arc-2026d-with-codex/v1/output/submission.json",
           "kaggle versions arc-2026d-with-codex/v1/output/arc-2026d-with-codex_v1.log"),
}
FULL_RUNS = [  # (label, source); listed in the order used in the findings (section 4c)
    ("arc-2026d v4", "kaggle versions/arc-2026d/v4/output/submission.json"),
    ("arc-2026d-with-codex v1", "kaggle versions arc-2026d-with-codex/v1/output/submission.json"),
    ("arc-2026d-with-codex v2", "source notebook arc-2026d-with-codex/output/submission.json"),
    ("fork-of-arc-2026d-32-22lb-updated-v3", "provided notebooks/fork-of-arc-2026d-32-22lb-updated-v3.ipynb"),
    ("arc-2026d-with-codex v25", "provided notebooks/arc-2026d-with-codex_v25.ipynb"),
    ("experiment-with-arc-agi-2 v10", "kaggle versions/experiment-with-arc-agi-2/v10/output/full_eval_submission.json"),
]
KS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 100]


def tn(units):
    return sum(1 / len(SOL[t]) for t, _ in units)


def read(rel):
    return open(os.path.join(ROOT, rel), encoding="utf-8", newline="").read().replace("\r\n", "\n").replace("\r", "\n")


def score_submission(rel):
    sub = json.load(open(os.path.join(ROOT, rel)))
    return {(t, i) for t, outs in SOL.items() if t in sub for i, g in enumerate(outs)
            if i < len(sub[t]) and g in (sub[t][i].get("attempt_1"), sub[t][i].get("attempt_2"))}


def notebook_dual_set(rel):  # same rule as score_all.py
    nb = json.load(open(os.path.join(ROOT, rel), encoding="utf-8"))
    out = "".join("".join(x.get("text", "")) for x in nb["cells"][8].get("outputs", []))
    sets = [set(re.findall(r"'([0-9a-f]{8})_(\d+)'", m.group(1))) for m in re.finditer(r"\{('[0-9a-f]{8}_\d+'[^}]*)\}", out)]
    return {(t, int(i)) for t, i in sets[2]}


def right_units(rel):
    return notebook_dual_set(rel) if rel.endswith(".ipynb") else score_submission(rel)


def bk(s):
    t, i = s.rsplit("_", 1)
    return t, int(i)


def section(text, start, end):
    s = text.index(start)
    return text[s:text.index(end, s)]


def exp1_ranks(text):
    seg = section(text, "EXPERIMENT 1:", "EXPERIMENT 2:")
    seg = seg[seg.index("GT Present cases"):]
    return {bk(m.group(1)): (int(float(m.group(2))), int(float(m.group(3))))
            for m in re.finditer(r"^\s*([0-9a-f]{8}_\d+)\s+([\d.]+)\s+([\d.]+)\s", seg, re.M)}


def exp18_rows(text):
    seg = section(text, "Parse Audit Summary:", "Classification of Missing GT cases")
    rows = {}
    for m in re.finditer(r"^\s*([0-9a-f]{8}_\d+)\s+(True|False)\s+(True|False)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\S+)\s*$", seg, re.M):
        rows[bk(m.group(1))] = {"gt_present": m.group(2) == "True", "fallback": m.group(3) == "True",
                                "cands": int(m.group(4)), "unique": int(m.group(5)), "avg_sim": float(m.group(6))}
    return rows


def max_decode_gap(rel):
    """Longest time from a worker's 'decoding' line to its next line (one search call plus scoring), from the JSON log."""
    events = json.load(open(os.path.join(ROOT, rel), encoding="utf-8"))
    last, gap = {}, 0.0
    for e in events:
        if e.get("stream_name") != "stdout":
            continue
        for ln in e["data"].splitlines():
            m = re.match(r"\[Rank (\d)\] (.*)", ln)
            if not m:
                continue
            if m.group(1) in last:
                gap = max(gap, e["time"] - last.pop(m.group(1)))
            if m.group(2).startswith("decoding"):
                last[m.group(1)] = e["time"]
    return round(gap, 1)


def pearson(xs, ys):
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5


def one_line(text, pattern):
    m = re.search(pattern, text)
    return m.group(0).strip() if m else None


def write_dat(name, header, rows):
    os.makedirs(FIG, exist_ok=True)
    with open(os.path.join(FIG, name), "w", newline="\n") as fh:
        fh.write(" ".join(header) + "\n")
        for r in rows:
            fh.write(" ".join(str(x) for x in r) + "\n")


out = {"pool_runs": {}, "full_runs": {}}

# 1. the two runs with pool data
for name, (log_rel, sub_rel, json_rel) in POOL_RUNS.items():
    text = read(log_rel)
    ranks, rows = exp1_ranks(text), exp18_rows(text)
    sub = score_submission(sub_rel)
    pool = set(ranks)
    assert len(rows) == 172 and set(rows) == set(UNITS), (name, len(rows))
    assert pool == {u for u, r in rows.items() if r["gt_present"]}, f"{name}: EXPERIMENT 1 and 18 pools differ"
    assert sub <= pool, f"{name}: a submitted right unit is not in the pool"
    topk = {k: sum(1 for r, _ in ranks.values() if r <= k) for k in KS}
    topk_tn = {k: round(tn({u for u, (r, _) in ranks.items() if r <= k}), 2) for k in KS}
    absent = {u: r for u, r in rows.items() if not r["gt_present"]}
    cls = {"fallback": [u for u, r in absent.items() if r["fallback"]]}
    rest = {u: r for u, r in absent.items() if not r["fallback"]}
    cls["close"] = [u for u, r in rest.items() if r["avg_sim"] > 90]
    cls["partial"] = [u for u, r in rest.items() if 50 < r["avg_sim"] <= 90]
    cls["far"] = [u for u, r in rest.items() if r["avg_sim"] <= 50]
    lost_sel = pool - sub
    fb_reasons = {}
    for m in re.finditer(r"FALLBACK_SAVE ([0-9a-f]{8}_\d+): ([A-Z_]+)", text):
        fb_reasons[m.group(2)] = fb_reasons.get(m.group(2), 0) + 1
    assert all(rows[u]["cands"] == 1 == rows[u]["unique"] for u in cls["fallback"]), f"{name}: a fallback unit has real grids"
    kg2 = {u for u, (_, b) in ranks.items() if b <= 2}
    fb_tasks = sorted({u[0] for u in cls["fallback"]})
    finished = {m.group(1): float(m.group(2)) for m in re.finditer(r"finished ([0-9a-f]{8}) in ([\d.]+)s", text)}
    fb_secs = [finished[t] for t in fb_tasks if t in finished]
    real = [r for r in rows.values() if not r["fallback"]]
    uniq_hist = {}
    for r in real:
        uniq_hist[r["unique"]] = uniq_hist.get(r["unique"], 0) + 1
    same_rank = sum(1 for a, b in ranks.values() if a == b)
    out["pool_runs"][name] = {
        "log": log_rel, "submission": sub_rel,
        "submitted_right_units": len(sub), "submitted_task_norm": round(tn(sub), 2),
        "pool_units": len(pool), "pool_task_norm": round(tn(pool), 2),
        "selection_headroom_task_norm": round(tn(pool) - tn(sub), 2),
        "in_pool_not_submitted_units": len(lost_sel),
        "no_correct_candidate_units": len(absent), "no_correct_candidate_pct": round(100 * len(absent) / 172, 1),
        "topk_units": topk, "topk_task_norm": topk_tn,
        "submitted_equals_top2": sub == {u for u, (r, _) in ranks.items() if r <= 2},
        "absent_classes": {k: len(v) for k, v in cls.items()},
        "fallback_reasons": fb_reasons,
        "fallback_tasks": len(fb_tasks), "fallback_tasks_finished": len(fb_secs),
        "fallback_task_seconds": [min(fb_secs), max(fb_secs)] if fb_secs else None,
        "tasks_finished": len(finished), "timeouts": text.count("timeout after"),
        "puzzle_exceptions": text.count("PUZZLE_EXCEPTION"),
        "max_search_call_plus_scoring_s": max_decode_gap(json_rel),
        "kgmon_top2_units": len(kg2), "kgmon_top2_task_norm": round(tn(kg2), 2),
        "kgmon_minus_submitted_task_norm": round(tn(kg2) - tn(sub), 2),
        "rank_pearson": round(pearson([a for a, _ in ranks.values()], [b for _, b in ranks.values()]), 3),
        "duplicate_ratio": round(sum(1 - r["unique"] / r["cands"] for r in rows.values()) / 172, 3),
        "train_seed_lines": re.findall(r"(?:train|eval)_seed=\d+ for puzzle [0-9a-f]{8}", text),
        "candidates_mean_all": round(sum(r["cands"] for r in rows.values()) / 172, 2),
        "unique_mean_all": round(sum(r["unique"] for r in rows.values()) / 172, 2),
        "candidates_mean_nonfallback": round(sum(r["cands"] for r in real) / len(real), 2),
        "unique_mean_nonfallback": round(sum(r["unique"] for r in real) / len(real), 2),
        "unique_le3_nonfallback": sum(1 for r in real if r["unique"] <= 3),
        "unique_le3_absent_nonfallback": sum(1 for r in rest.values() if r["unique"] <= 3),
        "candidates_max": max(r["cands"] for r in rows.values()),
        "unique_le3_absent_examples": [f"{t}_{i} {r['unique']}/{r['cands']} sim {r['avg_sim']:.1f}" for (t, i), r in
                                       sorted(rest.items(), key=lambda x: (x[1]["unique"], -x[1]["cands"]))[:4]],
        "nonfallback_units": len(real),
        "unique_hist_nonfallback": dict(sorted(uniq_hist.items())),
        "probmul_kgmon_same_rank": f"{same_rank}/{len(ranks)}",
        "log_lines": {
            "exp2_cands": one_line(text, r"Average candidates per subkey: [\d.]+"),
            "exp2_unique": one_line(text, r"Average unique candidates: [\d.]+"),
            "exp2_dup": one_line(text, r"Average duplicate ratio: [\d.]+"),
            "exp4_same": one_line(text, r"Same rank in both scorers: .*"),
            "exp4_corr": one_line(text, r"Rank correlation: [\d.]+"),
            "exp13_dual": one_line(text, r"Current Dual Score: +[\d.]+.*"),
            "exp13_oracle": one_line(text, r"Oracle Score: +[\d.]+.*"),
        },
    }
    out["pool_runs"][name]["_units"] = {"pool": sorted(pool), "submitted": sorted(sub),
                                       **{f"absent_{k}": sorted(v) for k, v in cls.items()}}

# 1b. R1 vs R2: same program (arc_loader / arc_decoder identical; arc_solver differs only in MAX_BEAM_CANDIDATES 5 vs 10,
# which cannot bind: a child token survives only with conditional probability > 0.2, so at most 4 do), same seeds
seeds = {n: sorted(out["pool_runs"][n].pop("train_seed_lines")) for n in ("v1", "v2")}
U = {n: {k: set(map(tuple, v)) for k, v in out["pool_runs"][n]["_units"].items()} for n in ("v1", "v2")}
out["r1_vs_r2"] = {
    "seed_lines_each": len(seeds["v1"]), "seed_lines_identical": seeds["v1"] == seeds["v2"],
    "submitted_shared": len(U["v1"]["submitted"] & U["v2"]["submitted"]),
    "submitted_union": len(U["v1"]["submitted"] | U["v2"]["submitted"]),
    "pool_shared": len(U["v1"]["pool"] & U["v2"]["pool"]), "pool_union": len(U["v1"]["pool"] | U["v2"]["pool"]),
    "pool_union_task_norm": round(tn(U["v1"]["pool"] | U["v2"]["pool"]), 2),
}

# 2. the six full runs
sets = {lab: right_units(rel) for lab, rel in FULL_RUNS}
wall = {}  # hours: data/version_log_summary.csv for fetched versions, papermill metadata for the provided notebooks
for line in read("data/version_log_summary.csv").splitlines()[1:]:
    c = line.split(",")
    wall[f"{c[0]} {c[1]}"] = float(c[2])
for lab, rel in FULL_RUNS:
    if rel.endswith(".ipynb"):
        pm = json.load(open(os.path.join(ROOT, rel), encoding="utf-8"))["metadata"]["papermill"]
        wall[lab] = round(pm["duration"] / 3600, 2)
    out["full_runs"][lab] = {"source": rel, "units": len(sets[lab]), "task_norm": round(tn(sets[lab]), 2), "wall_h": wall[lab]}
labs = [lab for lab, _ in FULL_RUNS]
five = labs[:5]
growth, acc = [], set()
for lab in five:
    new = sets[lab] - acc
    acc |= sets[lab]
    growth.append({"run": lab, "new_units": len(new), "union_units": len(acc), "union_task_norm": round(tn(acc), 2)})
perm_curves = []
for order in itertools.permutations(five):
    acc, c = set(), []
    for lab in order:
        acc |= sets[lab]
        c.append(len(acc))
    perm_curves.append(c)
core5 = set.intersection(*(sets[l] for l in five))
union6 = set.union(*sets.values())
pairs = [(a, b, len(sets[a] & sets[b]), len(sets[a] | sets[b])) for a, b in itertools.combinations(five, 2)]
counts = {}
for u in set.union(*(sets[l] for l in five)):
    n = sum(u in sets[l] for l in five)
    counts[n] = counts.get(n, 0) + 1
out["union_growth_5"] = growth
out["union_growth_all_orders"] = {"min": [min(c[i] for c in perm_curves) for i in range(5)],
                                  "max": [max(c[i] for c in perm_curves) for i in range(5)]}
out["core_5"] = {"units": len(core5), "task_norm": round(tn(core5), 2)}
out["union_6"] = {"units": len(union6), "task_norm": round(tn(union6), 2),
                  "v10_new_over_5": len(sets[labs[5]] - set.union(*(sets[l] for l in five)))}
out["pairwise_5"] = {"shared_min": min(p[2] for p in pairs), "shared_max": max(p[2] for p in pairs),
                     "union_min": min(p[3] for p in pairs), "union_max": max(p[3] for p in pairs)}
out["units_by_runs_right_5"] = dict(sorted(counts.items()))

# 2b. public LB per notebook: the table in the NVARC family README (team submissions list, matched by notebook slug)
lb = []
for line in read("NVARC family (34 notebooks, downloaded 2026-10-19)/README.md").splitlines():
    m = re.match(r"\| `([^`]+)` \| [^|]+ \| ([\d-]+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| ([\d,]*) \|", line)
    if m:
        lb.append([m.group(1), m.group(2), int(m.group(3)), float(m.group(4)), float(m.group(5)), float(m.group(6)),
                   m.group(7).replace(",", " ")])
with open(os.path.join(ROOT, "data", "lb_by_notebook.csv"), "w", newline="\n") as fh:
    fh.write("slug,last_run,submissions,lb_min,lb_mean,lb_max,scored_versions\n")
    fh.writelines(",".join(str(x) for x in r) + "\n" for r in lb)
best = max(lb, key=lambda r: r[5])
out["lb"] = {"notebooks_with_scored_submissions": len(lb), "submissions": sum(r[2] for r in lb),
             "best_max": best[5], "best_slug": best[0], "best_slug_subs": best[2], "best_slug_min": best[3],
             "best_slug_mean": best[4], "best_slug_versions": best[6]}
# Which scoring rule do the LB values fit? Per-task averaging over 120 tasks with 1-3 test inputs (as in the public set)
# gives multiples of 100/720; a plain average over N test outputs gives multiples of 100/N. Values are rounded to 0.01.
lbv = sorted({v for r in lb for v in (r[3], r[5]) if v > 0})
fit = lambda step: all(abs(v - round(v / step) * step) <= 0.005 + 1e-9 for v in lbv)
out["lb"]["distinct_min_max_values"] = len(lbv)
out["lb"]["fit_task_average_100_720"] = fit(100 / 720)
out["lb"]["unit_average_N_fitting_100_399"] = [n for n in range(100, 400) if fit(100 / n)]

json.dump(out, open(os.path.join(ROOT, "data", "paper_numbers.json"), "w", newline="\n"), indent=1, default=list)

# 3. figure tables (pgfplots)
P = out["pool_runs"]
write_dat("topk.dat", ["k", "v1", "v2"], [[k, P["v1"]["topk_units"][k], P["v2"]["topk_units"][k]] for k in KS if k <= 20])
g = out["union_growth_all_orders"]
write_dat("union_growth.dat", ["n", "union", "lo", "hi", "single"],
          [[i + 1, growth[i]["union_units"], g["min"][i], g["max"][i], out["full_runs"][five[i]]["units"]] for i in range(5)])
write_dat("loss.dat", ["run", "submitted", "selection", "close", "partial", "far", "fallback"],
          [[{"v1": "R1", "v2": "R2"}[r], P[r]["submitted_right_units"], P[r]["in_pool_not_submitted_units"],
            *(P[r]["absent_classes"][c] for c in ("close", "partial", "far", "fallback"))] for r in ("v1", "v2")])
hmax = max(max(P[r]["unique_hist_nonfallback"]) for r in ("v1", "v2"))
write_dat("unique_hist.dat", ["unique", "v1", "v2"],
          [[u, P["v1"]["unique_hist_nonfallback"].get(u, 0), P["v2"]["unique_hist_nonfallback"].get(u, 0)] for u in range(1, hmax + 1)])

# 4. summary
for name in ("v2", "v1"):
    p = P[name]
    print(f"[{name}] submitted {p['submitted_right_units']} units ({p['submitted_task_norm']}), pool {p['pool_units']} "
          f"({p['pool_task_norm']}), headroom +{p['selection_headroom_task_norm']}, submitted == top-2: {p['submitted_equals_top2']}")
    print(f"     top-k {p['topk_units']}")
    print(f"     no correct candidate {p['no_correct_candidate_units']} ({p['no_correct_candidate_pct']} %): {p['absent_classes']}, "
          f"fallback reasons {p['fallback_reasons']} in {p['fallback_tasks']} tasks ({p['fallback_tasks_finished']} finished, "
          f"{p['fallback_task_seconds']} s); tasks finished {p['tasks_finished']}, timeouts {p['timeouts']}, "
          f"puzzle exceptions {p['puzzle_exceptions']}")
    print(f"     candidates / unique: all {p['candidates_mean_all']} / {p['unique_mean_all']}, "
          f"non-fallback {p['candidates_mean_nonfallback']} / {p['unique_mean_nonfallback']} (n={p['nonfallback_units']}), "
          f"<= 3 unique: {p['unique_le3_nonfallback']} (answer absent: {p['unique_le3_absent_nonfallback']}, e.g. "
          f"{p['unique_le3_absent_examples']}), max candidates {p['candidates_max']}; "
          f"probmul = kgmon rank {p['probmul_kgmon_same_rank']}")
    print(f"     kgmon top-2 {p['kgmon_top2_units']} units ({p['kgmon_top2_task_norm']}, {p['kgmon_minus_submitted_task_norm']:+.2f}); "
          f"rank Pearson {p['rank_pearson']}; duplicate ratio {p['duplicate_ratio']}; "
          f"longest search call + scoring {p['max_search_call_plus_scoring_s']} s")
    print(f"     log: {p['log_lines']}")
print(f"[R1 vs R2] {out['r1_vs_r2']}")
for lab, r in out["full_runs"].items():
    print(f"[full] {lab}: {r['units']} units, task-norm {r['task_norm']}, wall {r['wall_h']} h")
for x in growth:
    print(f"[growth] {x['run']}: +{x['new_units']} -> {x['union_units']} units ({x['union_task_norm']})")
print(f"[growth] all 120 orders: min {g['min']}, max {g['max']}")
print(f"[core] right in all 5: {out['core_5']}; pairwise shared {out['pairwise_5']}; units by #runs right {out['units_by_runs_right_5']}")
print(f"[union6] {out['union_6']}")
print(f"[lb] {out['lb']}")
