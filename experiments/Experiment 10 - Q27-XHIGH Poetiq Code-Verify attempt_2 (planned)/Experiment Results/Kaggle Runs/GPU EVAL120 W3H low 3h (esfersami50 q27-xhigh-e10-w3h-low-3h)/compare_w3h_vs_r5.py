import json, sys, statistics as st, collections
sys.stdout.reconfigure(encoding="utf-8")
E10 = r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)"
SOL = json.load(open(E10 + "/Experiment Review/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_solutions.json", encoding="utf-8"))
AP = json.load(open(r"D:/ARC AGI 2/_scratch/arcprize/qwen38_effort_per_task_v2_public.json"))
def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
W = load(r"D:/ARC AGI 2/_scratch/session_1c6db7aa/w3h_run1/out/q27x/poetiq_iterations.jsonl")
R = load(E10 + "/Experiment Results/Kaggle Runs/GPU EVAL120 Poetiq R5 (hosen42 q27-xhigh-e10-poetiq-r5-eval120)/q27x/poetiq_iterations.jsonl")
def hits(rec):
    g = rec.get("test_grids") or []
    s = SOL[rec["task_id"]]
    return sum(1 for i, t in enumerate(s) if i < len(g) and g[i] == t), len(s)
def it1(rows):
    d = {}
    for r in rows:
        if r["iteration"] == 1 and r["task_id"] not in d: d[r["task_id"]] = r
    return d
w1, r1 = it1(W), it1(R)
def summ(name, d):
    toks = [r["p1_tokens"] for r in d.values()]
    nat = sum(r["finish"] == "natural" for r in d.values())
    tp = sum(bool(r.get("train_passed")) for r in d.values())
    q = sum(hits(r)[0] for r in d.values()); qt = sum(hits(r)[1] for r in d.values())
    tasks_full = sum(hits(r)[0] == hits(r)[1] for r in d.values())
    secs = [r["sec"] for r in d.values()]
    print(f"{name}: tasks {len(d)} | think tokens median {st.median(toks):.0f} mean {st.mean(toks):.0f} p90 {sorted(toks)[int(.9*len(toks))-1]} | natural close {nat} ({nat/len(d):.0%}) | capped {len(d)-nat} | train-pass {tp} | test hits {q}/{qt} | tasks fully solved {tasks_full} | sec median {st.median(secs):.0f}")
    return toks
summ("W3H low  it1", w1); summ("R5 xhigh it1", r1)
common = sorted(set(w1) & set(r1))
print("common tasks", len(common))
wl = {t for t in common if hits(w1[t])[0] > 0}; xr = {t for t in common if hits(r1[t])[0] > 0}
print("it1 any-hit: low", len(wl), "xhigh", len(xr), "both", len(wl & xr), "low-only", sorted(wl - xr), "xhigh-only", len(xr - wl))
# tokens on tasks both natural
pairs = [(w1[t]["p1_tokens"], r1[t]["p1_tokens"]) for t in common]
print("token ratio low/xhigh median", round(st.median(a / b for a, b in pairs if b), 3))
# accuracy when train-passed
for name, d in (("low", w1), ("xhigh", r1)):
    tpq = [hits(r) for r in d.values() if r.get("train_passed")]
    print(name, "train-passed tasks", len(tpq), "-> test hit", sum(h > 0 for h, n in tpq))
# ARC Prize per-task low vs ours
apl = {t for t, v in AP.items() if v["low"]}; apx = {t for t, v in AP.items() if v["xhigh"]}
print("ARC Prize low solved", len(apl), "| ours low it1 any-hit", len(wl), "| overlap", len(apl & wl))
# extra iterations in W3H
it2 = [r for r in W if r["iteration"] > 1]
print("W3H iteration>1 records", len(it2), [(r["task_id"], r["iteration"], hits(r)) for r in it2])
# finish mode breakdown and natural-close accuracy
for name, d in (("low", w1), ("xhigh", r1)):
    nat = [r for r in d.values() if r["finish"] == "natural"]; cap = [r for r in d.values() if r["finish"] != "natural"]
    print(name, "natural:", len(nat), "hit", sum(hits(r)[0] > 0 for r in nat), "| capped:", len(cap), "hit", sum(hits(r)[0] > 0 for r in cap))
print("=== R5 final vs W3H")
R5DIR = E10 + "/Experiment Results/Kaggle Runs/GPU EVAL120 Poetiq R5 (hosen42 q27-xhigh-e10-poetiq-r5-eval120)/q27x/"
def final_hits(path):
    sub = json.load(open(path, encoding="utf-8")); out = {}
    for t, s in SOL.items():
        atts = sub.get(t) or []
        h = 0
        for i, truth in enumerate(s):
            a = atts[i] if i < len(atts) else {}
            if isinstance(a, dict) and (a.get("attempt_1") == truth or a.get("attempt_2") == truth): h += 1
        out[t] = (h, len(s))
    return out
r5f = final_hits(R5DIR + "poetiq_eval_generated_submission.json")
w3f = final_hits(r"D:/ARC AGI 2/_scratch/session_1c6db7aa/w3h_run1/out/q27x/poetiq_eval_generated_submission.json")
print("R5 final query hits", sum(h for h, n in r5f.values()), "| W3H final", sum(h for h, n in w3f.values()))
r5t = {t for t, (h, n) in r5f.items() if h > 0}; w3t = {t for t, (h, n) in w3f.items() if h > 0}
print("tasks with any hit: R5", len(r5t), "W3H", len(w3t), "overlap", len(r5t & w3t), "W3H-only", sorted(w3t - r5t))
print("W3H-only in ARC Prize xhigh?", [(t, AP.get(t, {}).get("xhigh")) for t in sorted(w3t - r5t)])
print("R5 p1 at cap (77824):", sum(r["p1_tokens"] == 77824 for r in r1.values()), "of", len(r1))
print("R5 it1 natural-stop tokens median", st.median(r["p1_tokens"] for r in r1.values() if r["p1_tokens"] < 77824))
print("W3H p1 at cap (60000):", sum(r["p1_tokens"] >= 60000 for r in w1.values()))
print("=== easy-task cost: tasks W3H train-verified at it1")
tv = [t for t, r in w1.items() if r.get("train_passed")]
lt = sum(w1[t]["p1_tokens"] + w1[t]["p2_tokens"] for t in tv); xt = sum(r1[t]["p1_tokens"] + int(r1[t]["p2_tokens"]) for t in tv)
print(len(tv), "tasks | low tokens", lt, "| R5 xhigh it1 tokens", xt, "| R5 it1 train-pass on them", sum(bool(r1[t].get("train_passed")) for t in tv))
print("R5 it1 tokens on all 120:", sum(r["p1_tokens"] + int(r["p2_tokens"]) for r in r1.values()), "| W3H it1:", sum(r["p1_tokens"] + r["p2_tokens"] for r in w1.values()))
