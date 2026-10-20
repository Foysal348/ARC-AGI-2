"""Puzzle capability ledger: which ARC-AGI-2 evaluation puzzles Qwen3.8-27B solved in our GPU runs, with the
user's EDA categories (ARC2_Advanced_EDA_v8.ipynb cell 35 puzzle_features, copied verbatim below).

usage: python build_ledger.py            (writes ledger_units.csv, ledger_tasks.csv, summary.json next to this file)
Add a new run: append one entry to RUNS (kind = unit_records | task_records | task_frac) and re-run.
"""
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EXP = ROOT / "experiments"
FX = (EXP / "Experiment 1 - Q27-XHIGH Direct Reasoning (LB 18.75 and 26.94)" / "Experiment Notebooks" / "Build"
      / "Q27X_build" / "fixtures" / "fake_input" / "competitions" / "arc-prize-2026-arc-agi-2")
E1 = EXP / "Experiment 1 - Q27-XHIGH Direct Reasoning (LB 18.75 and 26.94)" / "Experiment Results" / "Kaggle Runs"

# name, method, profile, kind, path (relative to experiments/)
RUNS = [
    ("run3 v1.3 direct+code", "direct + code check", "EVAL12", "task_records", E1 / "Q27X_run3/bundle/q27x/records.jsonl"),
    ("run4 v1.4 direct", "direct (xhigh 49k+8k)", "EVAL12", "unit_records", E1 / "Q27X_run4/bundle/q27x/records.jsonl"),
    ("run6 v1.4.1 direct", "direct (xhigh 49k+8k)", "EVAL12", "unit_records", E1 / "Q27X_run6_v141/q27x/records.jsonl"),
    ("run7 v1.4.4 direct (LB 26.94 code)", "direct (xhigh 49k+8k)", "EVAL12", "unit_records",
     E1 / "Q27X_run7_v144/q27x/records.jsonl"),
    ("run5 R6 code loop", "program synthesis loop", "EVAL12", "task_records",
     EXP / "Experiment 2 - R6 Code Loop/Experiment Results/Kaggle Runs/Q27X_run5_R6/bundle/q27x/records.jsonl"),
    ("agent v1.8.3 (agent only)", "REPL agent", "EVAL12", "task_frac",
     (EXP / "Experiment 4 - Q27-AGENT REPL Agent and HYBRID (v1 to v1.8.3)/Experiment Results/Kaggle Runs"
            "/run4_v1_8_3_foysal_EVAL12_complete/agent_eval/eval_report.json", "per_task_agent_only")),
    ("agent v1.9.1 (agent only)", "long-think program", "EVAL12", "task_frac",
     (EXP / "Experiment 5 - Q27-AGENT v1.9 Long-Think Synthesis (planned)/Experiment Results/Kaggle Runs"
            "/GPU EVAL12 v1.9.1 (foysal v3)/agent_eval/eval_report.json", "per_task.agent_correct_frac")),
    ("dual v2 direct", "direct (xhigh 49k+8k), attempt_2 = 2nd answer", "EVAL12", "unit_records",
     EXP / "Experiment 6 - Q27-XHIGH Second Attempt (v1.4.4 + attempt_2)/Experiment Results/Kaggle Runs"
           "/GPU EVAL12 dual v2 (foysal q27-xhigh-dual v1)/q27x/records.jsonl"),
    ("E9 EVAL12 direct 86k (primary units)", "direct (xhigh 77.8k+8k), original order", "EVAL12", "unit_records_primary",
     EXP / "Experiment 9 - Q27-XHIGH 86k + Tail Resample attempt_2/Experiment Results/Kaggle Runs"
           "/GPU EVAL12 E9 (esfer q27-xhigh-e9 v1)/q27x/records.jsonl"),
    ("E10 Poetiq R4 EVAL12 (86k, code + train verify, 1 expert)", "Poetiq program synthesis (77.8k think per iteration)", "EVAL12",
     "poetiq_report", EXP / "Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Results/Kaggle Runs"
                           "/GPU EVAL12 Poetiq R4 (foysal q27-xhigh-e10-poetiq-r4 v1)/q27x/poetiq_run_report.json"),
    ("E8 R5 PROXY48 direct (engine died at 124 min, 41/56 finished)", "direct (xhigh 49k+8k) + E8 attempt_2", "PROXY48",
     "unit_records", EXP / "Experiment 8 - Q27-XHIGH Augmentation Scoring attempt_2 (planned)/Experiment Results/Kaggle Runs"
                          "/GPU PROXY48 E8 R5 (foysal q27-xhigh-e8 v1)/q27x/records.jsonl"),
    ("E10 Poetiq R5 EVAL120 (86k per iteration, 1 expert, <=2 iterations reached)", "Poetiq program synthesis + train verify", "EVAL120",
     "poetiq_report", EXP / "Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Results/Kaggle Runs"
                           "/GPU EVAL120 Poetiq R5 (hosen42 q27-xhigh-e10-poetiq-r5-eval120)/q27x/poetiq_run_report.json"),
    ("E10 ORDER2 EVAL120 direct 86k (all 120 tasks)", "direct (xhigh 77.8k+8k), ORDER2 order", "EVAL120", "unit_records",
     EXP / "Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Results/Kaggle Runs"
           "/GPU EVAL120 ORDER2 86k DIRECT (hosen42 q27-xhigh-e10-order2)/q27x/records.jsonl"),
]


def cells(g):
    return int(np.prod(np.shape(g)))


def puzzle_features(key, data):          # verbatim logic of ARC2_Advanced_EDA_v8.ipynb cell 35
    p = data[key]
    train = p["train"]
    input_sizes = [cells(x["input"]) for x in train]
    output_sizes = [cells(x["output"]) for x in train]
    ratios = [max(i, o) / max(min(i, o), 1) for i, o in zip(input_sizes, output_sizes)]
    colors = set()
    px = []
    for ex in train:
        for name in ["input", "output"]:
            arr = np.array(ex[name])
            colors.update(np.unique(arr).tolist())
            px.extend(arr.reshape(-1).tolist())
    avg_input_size = float(np.mean(input_sizes))
    avg_ratio = float(np.mean(ratios))
    max_ratio = float(max(ratios))
    n_colors = len(colors)
    n_train = len(train)
    n_queries = len(p["test"])
    dynamic = any(np.shape(x["input"]) != np.shape(x["output"]) for x in train)
    rare = bool(6 in colors or 9 in colors)
    bg_density = px.count(0) / max(len(px), 1)
    if dynamic and max_ratio > 30:
        tier = 3
    elif dynamic and max_ratio > 5:
        tier = 2
    elif avg_input_size > 400:
        tier = 2
    else:
        tier = 1
    fine = (avg_input_size / 900 * 0.30 + (avg_ratio - 1) / 10 * 0.25 + n_colors / 10 * 0.20
            + (0.25 if dynamic else 0.0))
    if n_train >= 6 and tier < 3:
        fine *= 0.7
    tags = [f"tier{tier}"]
    if tier == 1 and (not dynamic) and max_ratio == 1 and n_colors <= 5:
        tags.append("easy_static")
    if tier == 1 and fine <= 0.22:
        tags.append("easy")
    if tier == 1 and fine > 0.45:
        tags.append("hard_tier1")
    if dynamic:
        tags.append("dynamic")
    if rare:
        tags.append("rare")
    if n_queries > 1:
        tags.append("multi_query")
    if max_ratio > 5:
        tags.append("high_ratio")
    if max_ratio > 30:
        tags.append("extreme_ratio")
    if n_colors >= 8:
        tags.append("many_colors")
    if avg_input_size >= 500:
        tags.append("large")
    if bg_density >= 0.70:
        tags.append("sparse_bg")
    if bg_density <= 0.30:
        tags.append("dense_non_bg")
    if n_train <= 2:
        tags.append("low_train")
    return dict(tier=tier, fine=round(fine, 4), avg_input_size=round(avg_input_size, 1), max_ratio=round(max_ratio, 2),
                n_colors=n_colors, dynamic=dynamic, n_train=n_train, n_queries=n_queries, tags=tags)


def load_run(kind, path, ch):
    """-> {(task, q): {"hit": 0/1, "pass1": 0/1|None, "mode": str|None, "think": int|None}}"""
    out = {}
    if kind in ("unit_records", "unit_records_primary"):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if r.get("resample_of"):
                continue
            sc = r.get("score")
            if not isinstance(sc, dict) or "hit" not in sc:
                continue
            out[(r["task_id"], int(r["query"]))] = {"hit": int(sc["hit"]), "pass1": int(sc.get("pass1", 0)),
                                                   "mode": r.get("mode"), "think": r.get("think_tokens")}
    elif kind == "task_records":
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            sc = r.get("score") or {}
            qh = sc.get("query_hits")
            if qh is None:
                continue
            p1 = sc.get("pass1") or [None] * len(qh)
            for q, h in enumerate(qh):
                out[(r["task_id"], q)] = {"hit": int(h), "pass1": p1[q] if q < len(p1) else None,
                                          "mode": r.get("mode") or r.get("status"), "think": r.get("think_tokens")}
    elif kind == "poetiq_report":           # E10 Poetiq: per-unit hits in poetiq_run_report.json (task-level programs)
        rep = json.load(open(path, encoding="utf-8"))
        for r in rep["eval_unit_records"]:
            out[(r["task"], int(r["query"]))] = {"hit": int(r["hit"]), "pass1": int(r["pass1"]), "mode": "poetiq", "think": None}
    elif kind == "task_frac":
        file, key = path
        rep = json.load(open(file, encoding="utf-8"))
        k1, _, k2 = key.partition(".")
        for tid, v in rep[k1].items():
            frac = float(v[k2] if k2 else v)
            nq = len(ch[tid]["test"])
            hits = int(round(frac * nq))
            for q in range(nq):            # per-query detail is not recorded: the first `hits` queries are marked
                out[(tid, q)] = {"hit": int(q < hits), "pass1": None, "mode": "agent", "think": None,
                                 "approx": nq > 1 and 0 < hits < nq}
    return out


def main():
    ch = json.load(open(FX / "arc-agi_evaluation_challenges.json"))
    feats = {t: puzzle_features(t, ch) for t in ch}
    runs = []
    for name, method, profile, kind, path in RUNS:
        res = load_run(kind, path, ch)
        runs.append((name, method, profile, res))
    units = [(t, q) for t in sorted(ch) for q in range(len(ch[t]["test"]))]
    rows = []
    for t, q in units:
        f = feats[t]
        att = [(n, r[(t, q)]) for n, _, _, r in runs if (t, q) in r]
        if not att:
            continue
        solved = [n for n, x in att if x["hit"]]
        direct = [x for n, x in att if "direct" in n]
        rows.append({"task": t, "query": q, "tier": f["tier"], "fine": f["fine"], "tags": "|".join(f["tags"]),
                     "dynamic": f["dynamic"], "n_colors": f["n_colors"], "avg_input_size": f["avg_input_size"],
                     "n_train": f["n_train"], "n_queries": f["n_queries"],
                     "runs_attempted": len(att), "runs_solved": len(solved),
                     "solve_rate": round(len(solved) / len(att), 3),
                     "direct_think_tokens": ";".join(str(x["think"]) for x in direct if x.get("think") is not None),
                     "direct_modes": ";".join(str(x["mode"]) for x in direct),
                     "solved_by": ";".join(solved), "failed_by": ";".join(n for n, x in att if not x["hit"])})
    with open(HERE / "ledger_units.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    by_task = defaultdict(list)
    for r in rows:
        by_task[r["task"]].append(r)
    trows = []
    for t, rs in sorted(by_task.items()):
        f = feats[t]
        trows.append({"task": t, "tier": f["tier"], "fine": f["fine"], "tags": "|".join(f["tags"]),
                      "units_seen": len(rs), "attempts": sum(r["runs_attempted"] for r in rs),
                      "hits": sum(r["runs_solved"] for r in rs),
                      "solve_rate": round(sum(r["runs_solved"] for r in rs) / max(1, sum(r["runs_attempted"] for r in rs)), 3),
                      "ever_solved": any(r["runs_solved"] for r in rs)})
    with open(HERE / "ledger_tasks.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(trows[0]))
        w.writeheader()
        w.writerows(trows)
    # ---- all 120 evaluation tasks: EDA categories + ARC Prize official Qwen3.8-27B result + our runs
    ap = json.load(open(HERE / "arcprize_qwen38_27b_public_eval.json", encoding="utf-8"))["tasks"]
    proxy48 = set(json.load(open(ROOT / "resources" / "Eval Subset (EDA mirror)" / "proxy48.json"))) \
        if (ROOT / "resources" / "Eval Subset (EDA mirror)" / "proxy48.json").exists() else set()
    allrows = []
    for t in sorted(ch):
        f = feats[t]
        a = ap.get(t, {})
        tr = next((r for r in trows if r["task"] == t), None)
        allrows.append({"task": t, "tier": f["tier"], "fine": f["fine"], "tags": "|".join(f["tags"]),
                        "n_queries": f["n_queries"], "in_proxy48": t in proxy48,
                        "arcprize_xhigh": int(bool(a.get("xhigh"))), "arcprize_medium": int(bool(a.get("medium"))),
                        "arcprize_low": int(bool(a.get("low"))), "arcprize_none": int(bool(a.get("none"))),
                        "our_attempts": tr["attempts"] if tr else 0, "our_hits": tr["hits"] if tr else 0})
    with open(HERE / "ledger_all120.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(allrows[0]))
        w.writeheader()
        w.writerows(allrows)
    cats = ["tier1", "tier2", "tier3", "easy_static", "easy", "hard_tier1", "dynamic", "rare", "multi_query",
            "high_ratio", "extreme_ratio", "many_colors", "large", "sparse_bg", "dense_non_bg", "low_train"]
    by_cat = {}
    for c in cats:
        sel = [r for r in allrows if c in r["tags"].split("|")]
        if sel:
            by_cat[c] = {"tasks": len(sel), "arcprize_xhigh_solved": sum(r["arcprize_xhigh"] for r in sel),
                         "arcprize_xhigh_pct": round(100 * sum(r["arcprize_xhigh"] for r in sel) / len(sel), 1)}
    summary = {"runs": [{"name": n, "method": m, "profile": p, "units": len(r), "hits": sum(x["hit"] for x in r.values())}
                        for n, m, p, r in runs],
               "tasks_seen": len(by_task), "units_seen": len(rows), "evaluation_tasks": len(ch),
               "arcprize_xhigh_solved": sum(r["arcprize_xhigh"] for r in allrows), "by_category": by_cat}
    json.dump(summary, open(HERE / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
