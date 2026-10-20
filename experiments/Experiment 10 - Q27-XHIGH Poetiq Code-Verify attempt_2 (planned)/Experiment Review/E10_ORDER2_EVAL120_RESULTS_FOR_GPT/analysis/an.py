import json, csv, sys, statistics as st
from collections import defaultdict
from pathlib import Path
import importlib.util
import numpy as np

RUN = Path(sys.argv[1]); LED = Path(sys.argv[2])
r = [json.loads(l) for l in open(RUN / "q27x/records.jsonl", encoding="utf-8")]
hit = lambda x: int(bool(x["score"] and x["score"]["hit"]))
ap = json.load(open(LED / "arcprize_qwen38_27b_public_eval.json", encoding="utf-8"))["tasks"]
all120 = {row["task"]: row for row in csv.DictReader(open(LED / "ledger_all120.csv", encoding="utf-8"))}

# ---- task level
T = defaultdict(list)
for x in r:
    T[x["task_id"]].append(x)
tfrac = {t: sum(map(hit, xs)) / len(xs) for t, xs in T.items()}
full = {t for t, f in tfrac.items() if f == 1.0}
partial = {t for t, f in tfrac.items() if 0 < f < 1}
print("tasks", len(T), "| task score", round(sum(tfrac.values()), 2), "| fully solved", len(full), "| partial", len(partial),
      sorted((t, round(tfrac[t], 2)) for t in partial))
axh = {t for t, v in ap.items() if v.get("xhigh")}
anyhit = full | partial
print(f"\nARC Prize xhigh solved {len(axh)} tasks | ours (any query hit) {len(anyhit)} | full {len(full)}")
print("both:", len(anyhit & axh), "| ours only:", len(anyhit - axh), sorted(anyhit - axh), "| theirs only:", len(axh - anyhit), sorted(axh - anyhit))
print("ours full vs theirs: both", len(full & axh), "| ours-full only", len(full - axh), "| theirs only", len(axh - full))

# ---- ordering
print("\nORDER2 validation:")
for s in (2, 1, 0):
    ts = [t for t in T if T[t][0]["order_score"] == s]
    us = [x for x in r if x["order_score"] == s]
    print(f"  score {s}: {len(ts):3d} tasks ({len(us)} units) task-score {sum(tfrac[t] for t in ts):5.2f} = {100*sum(tfrac[t] for t in ts)/len(ts):4.1f} %"
          f" | unit hits {sum(map(hit, us))}/{len(us)} | forced {sum(x['mode']=='forced_close' for x in us)}/{len(us)}"
          f" | mean think {st.mean(x['think_tokens'] for x in us):.0f} | median min {st.median(x['sec'] for x in us)/60:.1f}"
          f" | ARC Prize xhigh {sum(t in axh for t in ts)}/{len(ts)}")
prev = {t for t, row in all120.items() if int(row["our_attempts"]) > 0}
for name, sel in (("previously run (ledger)", [t for t in T if t in prev]), ("never run before", [t for t in T if t not in prev])):
    print(f"  {name}: {len(sel)} tasks, task-score {sum(tfrac[t] for t in sel):.2f} = {100*sum(tfrac[t] for t in sel)/len(sel):.1f} %"
          f" | ARC Prize xhigh on same {sum(t in axh for t in sel)}")


def auc(score, label):
    pos = [s for s, l in zip(score, label) if l]; neg = [s for s, l in zip(score, label) if not l]
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return w / (len(pos) * len(neg))


tl = list(T)
print(f"\nAUC order_score -> our task any-hit: {auc([T[t][0]['order_score'] for t in tl], [t in anyhit for t in tl]):.3f}"
      f" | -> task frac>=0.5: {auc([T[t][0]['order_score'] for t in tl], [tfrac[t] >= .5 for t in tl]):.3f}"
      f" | -> ARC Prize xhigh: {auc([T[t][0]['order_score'] for t in tl], [t in axh for t in tl]):.3f}")

# ---- features
spec = importlib.util.spec_from_file_location("bl", LED / "build_ledger.py")
bl = importlib.util.module_from_spec(spec); spec.loader.exec_module(bl)
ch = json.load(open(bl.FX / "arc-agi_evaluation_challenges.json"))
F = {t: bl.puzzle_features(t, ch) for t in T}
CH = {}
for t in T:
    fr = []
    for ex in ch[t]["train"]:
        a, b = np.array(ex["input"]), np.array(ex["output"])
        fr.append(float((a != b).mean()) if a.shape == b.shape else None)
    CH[t] = None if any(v is None for v in fr) else float(np.mean(fr))
groups = {
    "n_train>=3": lambda x: F[x["task_id"]]["n_train"] >= 3,
    "n_train<=2": lambda x: F[x["task_id"]]["n_train"] <= 2,
    "single test": lambda x: F[x["task_id"]]["n_queries"] == 1,
    "multi test": lambda x: F[x["task_id"]]["n_queries"] > 1,
    "same size": lambda x: not F[x["task_id"]]["dynamic"],
    "dynamic size": lambda x: F[x["task_id"]]["dynamic"],
    "tier1": lambda x: F[x["task_id"]]["tier"] == 1,
    "tier2": lambda x: F[x["task_id"]]["tier"] == 2,
    "tier3": lambda x: F[x["task_id"]]["tier"] == 3,
    "avg_in<=200 cells": lambda x: F[x["task_id"]]["avg_input_size"] <= 200,
    "avg_in 200-400": lambda x: 200 < F[x["task_id"]]["avg_input_size"] <= 400,
    "avg_in>400": lambda x: F[x["task_id"]]["avg_input_size"] > 400,
    "change<5%": lambda x: CH[x["task_id"]] is not None and CH[x["task_id"]] < .05,
    "change 5-15%": lambda x: CH[x["task_id"]] is not None and .05 <= CH[x["task_id"]] < .15,
    "change 15-30%": lambda x: CH[x["task_id"]] is not None and .15 <= CH[x["task_id"]] < .30,
    "change>=30%": lambda x: CH[x["task_id"]] is not None and CH[x["task_id"]] >= .30,
    "prompt<=6k tok": lambda x: x["prompt_tokens"] <= 6000,
    "prompt 6-12k": lambda x: 6000 < x["prompt_tokens"] <= 12000,
    "prompt>12k": lambda x: x["prompt_tokens"] > 12000,
}
print("\nfeature -> per-unit hit rate, P(forced close), think, minutes:")
for g, f in groups.items():
    s = [x for x in r if f(x)]
    if s:
        print(f"  {g:18s} n={len(s):3d} hit {100*sum(map(hit, s))/len(s):4.0f} % | forced {100*sum(x['mode']=='forced_close' for x in s)/len(s):4.0f} %"
              f" | median think {st.median(x['think_tokens'] for x in s):6.0f} | median min {st.median(x['sec'] for x in s)/60:5.1f}")
pt = np.array([x["prompt_tokens"] for x in r]); th = np.array([x["think_tokens"] for x in r])
rk = lambda v: np.argsort(np.argsort(v))
print("\nrank corr prompt_tokens vs think:", round(float(np.corrcoef(rk(pt), rk(th))[0, 1]), 3))
fc = [x["mode"] == "forced_close" for x in r]
print("AUC prompt_tokens -> forced:", round(auc(list(pt), fc), 3), "| AUC order_score -> NOT forced:", round(auc([x["order_score"] for x in r], [not v for v in fc]), 3))

# ---- schedule / slots
first_admit = min(x["t_admit_min"] for x in r); last_admit = max(x["t_admit_min"] for x in r); last_done = max(x["t_done_min"] for x in r)
grid = np.arange(first_admit, last_done, 0.5)
A = np.array([sum(1 for x in r if x["t_admit_min"] <= g < x["t_done_min"]) for g in grid])
print(f"\nschedule: first admit {first_admit:.1f} min, last admit {last_admit:.1f}, last done {last_done:.1f} | mean active {A.mean():.2f}"
      f" | slot-util {100*A.mean()/16:.1f} % | tail after last admit {last_done-last_admit:.1f} min, mean active {A[grid >= last_admit].mean():.1f}")
for h in (2, 4, 6, 8):
    print(f"  by {h} h: units done {sum(x['t_done_min'] <= 60*h for x in r):3d}, hits {sum(hit(x) for x in r if x['t_done_min'] <= 60*h):2d}")
# hits by admission order quartile
rs = sorted(r, key=lambda x: x["admit_rank"])
q = len(rs) // 4
for i in range(4):
    s = rs[i*q:(i+1)*q if i < 3 else None]
    print(f"  admit quartile {i+1}: units {len(s)}, hits {sum(map(hit, s))} ({100*sum(map(hit, s))/len(s):.0f} %), admitted {s[0]['t_admit_min']:.0f}-{s[-1]['t_admit_min']:.0f} min")

# ---- KV
m = json.load(open(RUN / "q27x/vllm_metrics.json", encoding="utf-8"))
def g(s, name):
    return next((mm["value"] for mm in s["metrics"] if mm["name"] == name), None)
kv = [(s.get("t_min"), s.get("active"), g(s, "vllm:kv_cache_usage_perc"), g(s, "vllm:num_requests_running")) for s in m]
kvv = [k for _, _, k, _ in kv if k is not None]
print(f"\nKV samples {len(kv)} | max {100*max(kvv):.1f} % | p95 {100*np.percentile(kvv, 95):.1f} % | p50 {100*np.median(kvv):.1f} %"
      f" | waiting max {max(g(s, 'vllm:num_requests_waiting') or 0 for s in m)} | preemptions {g(m[-1], 'vllm:num_preemptions')}")
print("KV peak:", max(kv, key=lambda z: z[2] or 0))
full16 = [k for t, a, k, run in kv if run == 16 and k is not None]
print(f"KV while 16 running: n={len(full16)} mean {100*np.mean(full16):.1f} % max {100*max(full16):.1f} %")

# ---- token economics (thinking)
CAP = 77824
b1 = sum(min(x["think_tokens"], 49152) for x in r); b2 = sum(max(0, min(x["think_tokens"], CAP) - 49152) for x in r)
h1 = sum(hit(x) for x in r if x["think_tokens"] <= 49152); h2 = sum(hit(x) for x in r if x["think_tokens"] > 49152)
print(f"\nthinking 0-49k: {b1/1e6:.2f} M tok -> {h1} hits ({h1/(b1/1e6):.2f}/M) | 49k-78k: {b2/1e6:.2f} M tok -> {h2} hits ({h2/(b2/1e6):.2f}/M)")
print("\n49k+ hits:")
for x in sorted([x for x in r if hit(x) and x["think_tokens"] > 49152], key=lambda x: x["think_tokens"]):
    t = x["task_id"]
    print(f"  {x['unit']:12s} think {x['think_tokens']:6d} {x['mode']:12s} os={x['order_score']} ntrain={F[t]['n_train']} nq={F[t]['n_queries']}"
          f" tier={F[t]['tier']} arcprize_xhigh={int(t in axh)} prev_attempts={all120.get(t, {}).get('our_attempts')} prev_hits={all120.get(t, {}).get('our_hits')}")
json.dump({"tfrac": tfrac}, open(Path(sys.argv[3]) / "tfrac.json", "w"))
