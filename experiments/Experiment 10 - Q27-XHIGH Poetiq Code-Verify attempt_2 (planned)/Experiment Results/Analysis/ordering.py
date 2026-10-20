"""Did ORDER2 work? Puzzle-by-puzzle view + 16-slot replay with the measured unit durations under other orders."""
import csv, json, random, sys, heapq, statistics as st
from collections import defaultdict
from pathlib import Path

RUN = Path(sys.argv[1]); LED = Path(sys.argv[2]); OUT = Path(sys.argv[3])
r = [json.loads(l) for l in open(RUN / "q27x/records.jsonl", encoding="utf-8")]
hit = lambda x: int(bool(x["score"] and x["score"]["hit"]))
ap = json.load(open(LED / "arcprize_qwen38_27b_public_eval.json", encoding="utf-8"))["tasks"]
all120 = {row["task"]: row for row in csv.DictReader(open(LED / "ledger_all120.csv", encoding="utf-8"))}
sys.path.insert(0, str(LED))
import importlib.util
spec = importlib.util.spec_from_file_location("bl", LED / "build_ledger.py"); bl = importlib.util.module_from_spec(spec); spec.loader.exec_module(bl)
ch = json.load(open(bl.FX / "arc-agi_evaluation_challenges.json"))
FILE_ORDER = list(ch)                       # notebook TASK_IDS = challenge-file order (tie-break of ORDER2)

T = defaultdict(list)
for x in r:
    T[x["task_id"]].append(x)
tasks = sorted(T, key=lambda t: min(x["admit_rank"] for x in T[t]))
frac = {t: sum(map(hit, T[t])) / len(T[t]) for t in T}

# ---- 1. per-task table in admission order
rows = []
for i, t in enumerate(tasks, 1):
    xs = sorted(T[t], key=lambda x: x["query"])
    rows.append({"order_pos": i, "task": t, "order_score": xs[0]["order_score"], "n_train": len(ch[t]["train"]), "n_test": len(ch[t]["test"]),
                 "hits": f"{sum(map(hit, xs))}/{len(xs)}", "task_score": round(frac[t], 2), "arcprize_xhigh": int(bool(ap[t].get("xhigh"))),
                 "file_pos": FILE_ORDER.index(t) + 1, "admit_min": min(x["t_admit_min"] for x in xs), "done_min": max(x["t_done_min"] for x in xs),
                 "think": ";".join(str(x["think_tokens"]) for x in xs), "modes": ";".join(x["mode"] for x in xs),
                 "fine_nvarc": all120[t]["fine"]})
with open(OUT / "order2_per_task.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

# ---- 2. subtypes of score 1
def sub(t):
    a, b = len(ch[t]["train"]) >= 3, len(ch[t]["test"]) == 1
    return {(True, True): "2: >=3 train + 1 test", (True, False): "1a: >=3 train, multi test",
            (False, True): "1b: <=2 train, 1 test", (False, False): "0: <=2 train, multi test"}[(a, b)]
print("subtypes (tasks, task score, ARC Prize xhigh):")
for s in sorted({sub(t) for t in T}, reverse=True):
    ts = [t for t in T if sub(t) == s]
    print(f"  {s:28s} {len(ts):3d} tasks | ours {sum(frac[t] for t in ts):5.2f} = {100*sum(frac[t] for t in ts)/len(ts):4.1f} % | ARC Prize {sum(bool(ap[t].get('xhigh')) for t in ts)}/{len(ts)} = {100*sum(bool(ap[t].get('xhigh')) for t in ts)/len(ts):4.1f} %")

# ---- 3. misordered puzzles
print("\nscore-2 tasks that FAILED:", [t for t in tasks if T[t][0]["order_score"] == 2 and frac[t] == 0])
print("score-0 tasks that were SOLVED (any):", [(t, round(frac[t], 2)) for t in tasks if T[t][0]["order_score"] == 0 and frac[t] > 0])

# ---- 4. replay: 16 slots, measured durations, completion-based deadline
units_by_task = {t: sorted(T[t], key=lambda x: x["query"]) for t in T}
def replay(task_order, deadlines=(120, 240, 360, 480, 600)):
    q = [u for t in task_order for u in units_by_task[t]]
    slots = [(0.0, None)] * 16; heapq.heapify(slots); done = []
    for u in q:
        t0, _ = heapq.heappop(slots); t1 = t0 + u["sec"] / 60
        done.append((t1, u)); heapq.heappush(slots, (t1, u["unit"]))
    out = {}
    for d in deadlines:
        fin = [u for t1, u in done if t1 <= d]
        out[d] = (sum(map(hit, fin)), round(100 * sum(hit(u) / len(T[u["task_id"]]) for u in fin) / len(T), 1), len(fin))
    out["end"] = round(max(t1 for t1, _ in done))
    return out

orders = {
    "ORDER2 (as run)": tasks,
    "file order (no ordering = E9)": FILE_ORDER,
    "reverse ORDER2": list(reversed(tasks)),
    "NVARC fine, easy first": sorted(T, key=lambda t: float(all120[t]["fine"])),
    "ARC Prize xhigh solved first (external oracle)": sorted(T, key=lambda t: (not ap[t].get("xhigh"), tasks.index(t))),
    "our own outcome first (perfect oracle)": sorted(T, key=lambda t: (-frac[t], tasks.index(t))),
}
print("\nreplay with measured durations (16 slots): hits / task-norm % / units finished, by minute")
print(f"  {'order':48s} " + "  ".join(f"{d:>14d}" for d in (120, 240, 360, 480, 600)) + "   all done at")
res = {}
for name, o in orders.items():
    res[name] = replay(o)
    print(f"  {name:48s} " + "  ".join(f"{res[name][d][0]:3d}/{res[name][d][1]:4.1f}%/{res[name][d][2]:3d}" for d in (120, 240, 360, 480, 600)) + f"   {res[name]['end']} min")
random.seed(1); rnd = defaultdict(list)
for _ in range(1000):
    o = list(T); random.shuffle(o); rr = replay(o)
    for d in (120, 240, 360, 480, 600):
        rnd[d].append(rr[d][1])
print(f"  {'random order (mean of 1000)':48s} " + "  ".join(f"{'':4s}{st.mean(rnd[d]):4.1f}% (p10 {sorted(rnd[d])[100]:4.1f})"[:14].rjust(14) for d in (120, 240, 360, 480, 600)))
for d in (120, 240, 360, 480):
    better = sum(v < res["ORDER2 (as run)"][d][1] for v in rnd[d]) / 10
    print(f"  at {d} min: ORDER2 {res['ORDER2 (as run)'][d][1]} % beats {better:.1f} % of random orders (random mean {st.mean(rnd[d]):.1f} %)")

# ---- 5. rank quality
pos = {t: i for i, t in enumerate(tasks)}
solved = [t for t in T if frac[t] > 0]; unsolved = [t for t in T if frac[t] == 0]
print(f"\nmedian admission position: solved tasks {st.median(pos[t] for t in solved)+1:.0f}, unsolved {st.median(pos[t] for t in unsolved)+1:.0f} (of 120)")
for k in (20, 40, 60, 80, 100, 120):
    top = tasks[:k]
    print(f"  first {k:3d} tasks: solved-any {sum(frac[t] > 0 for t in top):2d}, task score {sum(frac[t] for t in top):5.2f}, ARC Prize xhigh {sum(bool(ap[t].get('xhigh')) for t in top)}")
fo = FILE_ORDER
for k in (40, 60, 80):
    top = fo[:k]
    print(f"  file order first {k}: solved-any {sum(frac[t] > 0 for t in top)}, task score {sum(frac[t] for t in top):.2f}")
