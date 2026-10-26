import json, collections, sys
from pathlib import Path
E = Path(r"D:/ARC AGI 2/experiments")
X10 = E / "Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Results/Kaggle Runs"
X11 = E / "Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/Kaggle Runs"
RUNS = {"R5 EVAL120 (vLLM, 77.8k)": X10 / "GPU EVAL120 Poetiq R5 (hosen42 q27-xhigh-e10-poetiq-r5-eval120)/q27x",
        "D66 HARD66 (77.8k)": X10 / "GPU HARD66 D66 77.8k (foysal q27-xhigh-e10-d66-hard-deep)/q27x",
        "L66 HARD66 (122.9k)": X10 / "GPU HARD66 L66 122.9k (foysal q27-xhigh-e10-l66-hard-120k)/q27x",
        "W3H LOW 3h": X10 / "GPU EVAL120 W3H low 3h (esfersami50 q27-xhigh-e10-w3h-low-3h)/q27x",
        "E11-05 EVAL30 (SGLang S49)": X11 / "E11-05 EVAL30 run1 (esfersami50, 2026-10-18)/q27x",
        "E11-06 P1-60K EVAL14 (SGLang)": X11 / "E11-06 P1-60K EVAL14 run1 (esfersami50, 2026-10-19)/q27x"}
for name, d in RUNS.items():
    rep = json.load(open(d / "poetiq_run_report.json", encoding="utf-8"))
    hit = collections.defaultdict(list)
    for u in rep["eval_unit_records"]:
        hit[u["task"]].append(bool(u.get("pass1") or u.get("pass2")))
    its = collections.defaultdict(list)
    p = d / "poetiq_iterations.jsonl"
    for ln in open(p, encoding="utf-8"):
        try:
            j = json.loads(ln)
        except Exception:
            continue
        its[j["task_id"]].append(j)
    def best_it(rows):
        sc = [(r.get("score") if r.get("score") is not None else -1, -r["iteration"]) for r in rows]
        m = max(sc)
        return -m[1], (m[0] if m[0] >= 0 else None)
    maxit = collections.Counter(max(r["iteration"] for r in v) for v in its.values())
    reach2 = [t for t, v in its.items() if max(r["iteration"] for r in v) >= 2]
    first_pass = {t: min((r["iteration"] for r in v if r.get("train_passed")), default=None) for t, v in its.items()}
    pass_at2p = [t for t in reach2 if first_pass[t] and first_pass[t] >= 2]
    best_from2 = [t for t in reach2 if best_it(its[t])[0] >= 2]
    solved = lambda t: bool(hit[t]) and all(hit[t])
    anyhit = lambda t: any(hit[t])
    print(f"== {name}: tasks {len(hit)}, queries {rep['eval_query_hits']}/{rep['eval_query_total']}, iterations {sum(len(v) for v in its.values())}")
    print("   max iteration reached per task:", dict(sorted(maxit.items())))
    print(f"   tasks reaching iteration >= 2: {len(reach2)}; first train-pass at iteration >= 2: {len(pass_at2p)} "
          f"-> test hit {sum(anyhit(t) for t in pass_at2p)}, fully {sum(solved(t) for t in pass_at2p)}")
    nb = [t for t in best_from2 if t not in pass_at2p]
    print(f"   not train-passed, best program from iteration >= 2: {len(nb)} -> test hit {sum(anyhit(t) for t in nb)}")
    print(f"   iteration >= 2 credited test-hit tasks: {sum(anyhit(t) for t in set(pass_at2p) | set(nb))}; "
          f"all hit tasks {sum(anyhit(t) for t in hit)}")
    det = [(t, first_pass[t], sum(hit[t]), len(hit[t])) for t in pass_at2p]
    print("   iter>=2 train-pass detail (task, first pass iter, hits/queries):", det)
