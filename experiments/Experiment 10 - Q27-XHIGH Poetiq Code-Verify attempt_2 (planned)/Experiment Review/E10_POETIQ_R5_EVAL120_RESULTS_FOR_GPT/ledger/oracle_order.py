"""Oracle order of the 120 ARC-AGI-2 evaluation puzzles from ALL our GPU runs + the ARC Prize Qwen3.8-27B solve list
(Claude, 2026-10-14, user request). Needs ledger_units.csv / ledger_all120.csv (run build_ledger.py first).

Solve estimate per task (smoothed rate, prior 0.35 with weight 1):
  p_hat = (S + X + B + 0.35) / (N + 1 + B + 1)
  S = sum over our runs of the task fraction solved in that run (hits / units that run attempted), N = number of our runs
  X = ARC Prize xhigh solved (one trial), B = ARC Prize lower levels solved (medium/low/none; counted only when solved)
Order: tasks never solved by anyone go last; then p_hat descending; then the measured ORDER2 86k minutes
  per unit ascending (cheaper first).
Tiers: A p_hat >= 0.6 | B 0.35-0.6 | C solved somewhere but p_hat < 0.35 | D never solved by anyone.

Check: the same order built WITHOUT the ORDER2 run is replayed on the ORDER2 run (16 slots, measured durations).
usage: python oracle_order.py  -> oracle_order_120.csv
"""
import csv
import heapq
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent.parent / "experiments"
ORDER2_RUN = "E10 ORDER2 EVAL120 direct 86k (all 120 tasks)"
ORDER2_REC = (EXP / "Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)" / "Experiment Results" / "Kaggle Runs"
              / "GPU EVAL120 ORDER2 86k DIRECT (hosen42 q27-xhigh-e10-order2)" / "q27x" / "records.jsonl")
PRIOR, PRIOR_W = 0.35, 1.0

units = list(csv.DictReader(open(HERE / "ledger_units.csv", encoding="utf-8")))
a120 = {r["task"]: r for r in csv.DictReader(open(HERE / "ledger_all120.csv", encoding="utf-8"))}
rec = [json.loads(l) for l in open(ORDER2_REC, encoding="utf-8")]
hit = lambda x: int(bool(x["score"] and x["score"]["hit"]))
R = defaultdict(list)
for x in rec:
    R[x["task_id"]].append(x)

# per (task, run): [hits, units]
per = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for u in units:
    for name in filter(None, u["solved_by"].split(";")):
        per[u["task"]][name][0] += 1; per[u["task"]][name][1] += 1
    for name in filter(None, u["failed_by"].split(";")):
        per[u["task"]][name][1] += 1


def p_hat(t, exclude=()):
    runs = {n: v for n, v in per[t].items() if n not in exclude}
    S = sum(h / n for h, n in runs.values()); N = len(runs)
    a = a120[t]; X = int(a["arcprize_xhigh"]); B = sum(int(a[k]) for k in ("arcprize_medium", "arcprize_low", "arcprize_none"))
    return (S + X + B + PRIOR * PRIOR_W) / (N + 1 + B + PRIOR_W), S, N, X, B


minutes = {t: st.mean(x["sec"] for x in R[t]) / 60 for t in R}


def never_solved(t, exclude=()):
    _, S, _, X, B = p_hat(t, exclude)
    return S + X + B == 0


def order(exclude=()):
    return sorted(a120, key=lambda t: (never_solved(t, exclude), -round(p_hat(t, exclude)[0], 6), minutes[t]))


def tier(t):
    p, S, N, X, B = p_hat(t)
    if S + X + B == 0:
        return "D"
    return "A" if p >= 0.6 else "B" if p >= 0.35 else "C"


rows = []
for i, t in enumerate(order(), 1):
    p, S, N, X, B = p_hat(t); a = a120[t]; xs = sorted(R[t], key=lambda x: x["query"])
    rows.append({"oracle_rank": i, "task": t, "tier": tier(t), "p_hat": round(p, 3),
                 "our_runs": N, "our_runs_with_hit": sum(1 for h, n in per[t].values() if h),
                 "our_task_score_sum": round(S, 2),
                 "arcprize_xhigh": a["arcprize_xhigh"], "arcprize_medium": a["arcprize_medium"],
                 "arcprize_low": a["arcprize_low"], "arcprize_none": a["arcprize_none"],
                 "order2_86k_hits": f"{sum(map(hit, xs))}/{len(xs)}", "order2_score": xs[0]["order_score"],
                 "n_train": a["tags"].count("low_train") and "<=2" or ">=3", "n_test": len(xs),
                 "order2_min_per_unit": round(minutes[t], 1), "order2_think": ";".join(str(x["think_tokens"]) for x in xs),
                 "solved_by": " | ".join(n for n, (h, _) in per[t].items() if h)})
with open(HERE / "oracle_order_120.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# ---- replay on the ORDER2 run
def replay(task_order, marks=(120, 240, 360, 480)):
    q = [u for t in task_order for u in sorted(R[t], key=lambda x: x["query"])]
    slots = [0.0] * 16; heapq.heapify(slots); done = []
    for u in q:
        t0 = heapq.heappop(slots); t1 = t0 + u["sec"] / 60; done.append((t1, u)); heapq.heappush(slots, t1)
    return {m: round(100 * sum(hit(u) / len(R[u["task_id"]]) for t1, u in done if t1 <= m) / 120, 1) for m in marks}


order2 = sorted(R, key=lambda t: min(x["admit_rank"] for x in R[t]))
print("tiers:", {k: sum(r["tier"] == k for r in rows) for k in "ABCD"})
for k in "ABCD":
    sel = [r for r in rows if r["tier"] == k]
    sc = sum(sum(hit(x) for x in R[r["task"]]) / len(R[r["task"]]) for r in sel)
    print(f"  tier {k}: {len(sel):3d} tasks | ORDER2 86k task score {sc:5.2f} ({100*sc/max(1,len(sel)):.0f} %) | ARC Prize xhigh {sum(int(r['arcprize_xhigh']) for r in sel)}")
print("\nreplay on the ORDER2 run (task-norm % finished by minute 120/240/360/480):")
print("  ORDER2 rules (as run)                    ", replay(order2))
print("  oracle WITHOUT the ORDER2 run (fair test) ", replay(order(exclude=(ORDER2_RUN,))))
print("  oracle with all runs (circular, reference)", replay(order()))
