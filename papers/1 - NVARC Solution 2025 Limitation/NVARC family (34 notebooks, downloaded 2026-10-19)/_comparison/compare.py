import json, sys, csv
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
E10 = "D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/"
K = E10 + "Experiment Results/Kaggle Runs/"
SOL = json.load(open(E10 + "Experiment Review/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_solutions.json"))
def units_from_sub(p):
    sub = json.load(open(p, encoding="utf-8")); u = set()
    for t, sols in SOL.items():
        for i, s in enumerate(sols):
            pr = (sub.get(t) or [{}] * 9)[i] if i < len(sub.get(t) or []) else {}
            if isinstance(pr, dict) and any(pr.get(k) == s for k in ("attempt_1", "attempt_2")): u.add((t, i))
    return u
R = {
    "NVARC": units_from_sub("D:/ARC AGI 2/resources/NVARC family/arc-2026d-with-codex/output/submission.json"),
    "DIRECT": {(r["task_id"], r["query"]) for r in map(json.loads, open(K + "GPU EVAL120 ORDER2 86k DIRECT (hosen42 q27-xhigh-e10-order2)/q27x/records.jsonl")) if (r.get("score") or {}).get("hit")},
    "POETIQ": units_from_sub(K + "GPU EVAL120 Poetiq R5 (hosen42 q27-xhigh-e10-poetiq-r5-eval120)/q27x/poetiq_eval_generated_submission.json"),
}
n = {t: len(s) for t, s in SOL.items()}
def tn(u): return sum(1 / n[t] for t, _ in u)
for k, u in R.items(): print(f"{k:7s} units {len(u):3d}/172  task-norm {tn(u):6.2f}/120 = {100*tn(u)/120:5.2f} %")
Q = R["DIRECT"] | R["POETIQ"]
print(f"Q27 union (DIRECT|POETIQ) units {len(Q)} task-norm {tn(Q):.2f} = {100*tn(Q)/120:.2f} %")
for a, b in [("NVARC", "POETIQ"), ("NVARC", "DIRECT")]:
    A, B = R[a], R[b]
    print(f"\n{a} vs {b}: both {len(A & B)} ({tn(A & B):.2f}) | only {a} {len(A - B)} ({tn(A - B):.2f}) | only {b} {len(B - A)} ({tn(B - A):.2f}) | union {len(A | B)} units, task-norm {tn(A | B):.2f} = {100*tn(A | B)/120:.2f} %")
A, Qa = R["NVARC"], Q
print(f"\nNVARC vs Q27 union: both {len(A & Qa)} | only NVARC {len(A - Qa)} ({tn(A - Qa):.2f}) | only Q27 {len(Qa - A)} ({tn(Qa - A):.2f}) | ALL union {len(A | Qa)} units, task-norm {tn(A | Qa):.2f} = {100*tn(A | Qa)/120:.2f} %")
# ledger categories
L = {r["task"]: r for r in csv.DictReader(open("D:/ARC AGI 2/resources/Puzzle Capability Ledger/ledger_all120.csv", encoding="utf-8"))}
only_n = sorted({t for t, _ in A - Qa}); only_q = sorted({t for t, _ in Qa - A})
print("\nonly-NVARC tasks:", len(only_n))
for t in only_n: print("  ", t, "tier", L[t]["tier"], "| arcprize xhigh", L[t]["arcprize_xhigh"], "|", L[t]["tags"])
print("\nonly-Q27 tasks:", len(only_q))
tagc_n, tagc_q = Counter(), Counter()
for t in only_n: tagc_n.update(L[t]["tags"].split("|"))
for t in only_q: tagc_q.update(L[t]["tags"].split("|"))
print("tags only-NVARC:", tagc_n.most_common(10)); print("tags only-Q27:", tagc_q.most_common(10))
tierc = lambda S: Counter(L[t]["tier"] for t in S)
print("tier of NVARC-solved tasks", tierc({t for t, _ in A}), "| Q27-solved", tierc({t for t, _ in Qa}), "| all 120", tierc(L))
json.dump({k: sorted(map(list, v)) for k, v in R.items()}, open("D:/ARC AGI 2/_scratch/nvarc_scan/cmp/units.json", "w"))
