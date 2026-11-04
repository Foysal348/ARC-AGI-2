import json, sys
E10 = "D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/"
K = E10 + "Experiment Results/Kaggle Runs/"
SOL = json.load(open(E10 + "Experiment Review/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_solutions.json"))
N = json.load(open("D:/ARC AGI 2/resources/NVARC family/arc-2026d-with-codex/output/submission.json"))
P = json.load(open(K + "GPU EVAL120 Poetiq R5 (hosen42 q27-xhigh-e10-poetiq-r5-eval120)/q27x/poetiq_eval_generated_submission.json"))
n = {t: len(s) for t, s in SOL.items()}
def get(sub, t, i, k):
    l = sub.get(t) or []
    return l[i].get(k) if i < len(l) and isinstance(l[i], dict) else None
res = {"N_a1": 0, "N_a2only": 0, "P_a1": 0, "P_a2only": 0, "P1+N1": 0, "P_both(a1,a2)": 0, "P1+N1 tn": 0.0, "P2 tn": 0.0}
for t, sols in SOL.items():
    for i, s in enumerate(sols):
        n1, n2, p1, p2 = (get(N, t, i, "attempt_1") == s, get(N, t, i, "attempt_2") == s, get(P, t, i, "attempt_1") == s, get(P, t, i, "attempt_2") == s)
        res["N_a1"] += n1; res["N_a2only"] += (n2 and not n1); res["P_a1"] += p1; res["P_a2only"] += (p2 and not p1)
        if p1 or n1: res["P1+N1"] += 1; res["P1+N1 tn"] += 1 / n[t]
        if p1 or p2: res["P_both(a1,a2)"] += 1; res["P2 tn"] += 1 / n[t]
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in res.items()})
print("P1+N1 task-norm %", round(100 * res["P1+N1 tn"] / 120, 2), "| Poetiq own pass@2 %", round(100 * res["P2 tn"] / 120, 2))
