"""Option A (1 expert, more iterations) vs option B (2 experts = 2 independent attempts, scored as attempt_1 / attempt_2)
from our real Poetiq runs. Units: test queries hit (ARC scoring: a query counts if either attempt is right); cost = output
tokens (p1 + p2) from poetiq_iterations.jsonl."""
import json, collections
from pathlib import Path
E = Path(r"D:/ARC AGI 2/experiments")
X10 = E / "Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Results/Kaggle Runs"
X11 = E / "Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/Kaggle Runs"
X15 = E / "Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Results/Kaggle Runs"
R = {"R5": X10 / "GPU EVAL120 Poetiq R5 (hosen42 q27-xhigh-e10-poetiq-r5-eval120)/q27x",
     "E15P": X15 / "E15P EVAL120 S60 S15 rev2 commit1 (esfersami50, 2026-10-25, 56a22f7f)/q27x",
     "D66": X10 / "GPU HARD66 D66 77.8k (foysal q27-xhigh-e10-d66-hard-deep)/q27x",
     "L66": X10 / "GPU HARD66 L66 122.9k (foysal q27-xhigh-e10-l66-hard-120k)/q27x",
     "E11-05": X11 / "E11-05 EVAL30 run1 (esfersami50, 2026-10-18)/q27x"}


def load(d):
    rep = json.load(open(d / "poetiq_run_report.json", encoding="utf-8"))
    hits = {(u["task"], u["query"]): bool(u.get("hit") or u.get("pass1") or u.get("pass2")) for u in rep["eval_unit_records"]}
    its = collections.defaultdict(list)
    p = d / "poetiq_iterations.jsonl"
    if p.exists():
        for ln in open(p, encoding="utf-8"):
            try:
                j = json.loads(ln)
            except Exception:
                continue
            its[j["task_id"]].append(j)
    return hits, its


D = {k: load(v) for k, v in R.items()}
tok = lambda r: (r.get("p1_tokens") or 0) + (r.get("p2_tokens") or 0)
print("== A. one expert, more iterations: what each iteration level cost and brought")
for k in ("R5", "D66", "L66", "E11-05"):
    hits, its = D[k]
    first_hit_it = {}
    for t, rows in its.items():
        tp = sorted((r["iteration"] for r in rows if r.get("train_passed")))
        first_hit_it[t] = tp[0] if tp else None
    hit_tasks = {t for (t, q), h in hits.items() if h}
    by_it = collections.defaultdict(lambda: [0, 0, 0])      # tasks reaching, tokens, newly train-passed tasks that hit the test
    for t, rows in its.items():
        for r in rows:
            by_it[r["iteration"]][0] += 1
            by_it[r["iteration"]][1] += tok(r)
        fi = first_hit_it[t]
        if fi and t in hit_tasks:
            by_it[fi][2] += 1
    tot = sum(v[1] for v in by_it.values())
    print(f"  {k}: total output {tot/1e6:.2f}M tok, tasks {len(its)}, hit tasks {len(hit_tasks)}")
    for i in sorted(by_it):
        n, tk, new = by_it[i]
        print(f"     iteration {i}: {n:3d} tasks ran it, {tk/1e6:5.2f}M tok, {new:2d} new solved tasks (first train-pass here + test hit)"
              f" -> {new / (tk/1e6) if tk else 0:.2f} tasks per M tok")
print()
print("== B. two independent attempts (2 experts): union over the same queries, scored as attempt_1 / attempt_2")
for a, b in (("D66", "L66"), ("R5", "E15P")):
    ha, hb = D[a][0], D[b][0]
    qs = sorted(set(ha) & set(hb))
    A_ = sum(ha[q] for q in qs); B_ = sum(hb[q] for q in qs); U = sum(ha[q] or hb[q] for q in qs)
    ta = {t for (t, _), h in ha.items() if h}; tb = {t for (t, _), h in hb.items() if h}
    toka = sum(tok(r) for rows in D[a][1].values() for r in rows); tokb = sum(tok(r) for rows in D[b][1].values() for r in rows)
    print(f"  {a} vs {b}: {len(qs)} common queries: {a} {A_}, {b} {B_}, union {U} (+{U - max(A_, B_)} over the better one); "
          f"tasks with a hit {len(ta)} / {len(tb)} / union {len(ta | tb)}; output tok {toka/1e6:.2f}M / {tokb/1e6:.2f}M")
    if toka and tokb:
        print(f"     2nd attempt yield: +{U - A_} queries over {a} for {tokb/1e6:.2f}M more tok = {(U - A_)/(tokb/1e6):.2f} queries per M tok; "
              f"+{U - B_} over {b} for {toka/1e6:.2f}M = {(U - B_)/(toka/1e6):.2f}")

print()
print("== A in queries (comparable with B): test queries hit by tasks whose first train-pass came at iteration k")
for k in ("R5", "D66", "L66", "E11-05"):
    hits, its = D[k]
    qh = collections.Counter(t for (t, q), h in hits.items() if h)
    agg = collections.defaultdict(lambda: [0, 0])
    for t, rows in its.items():
        for r in rows:
            agg[r["iteration"]][1] += tok(r)
        tp = sorted(r["iteration"] for r in rows if r.get("train_passed"))
        if tp:
            agg[tp[0]][0] += qh.get(t, 0)
    print("  " + k + ": " + "; ".join(f"it{i} +{q} q / {tk/1e6:.2f}M = {q/(tk/1e6) if tk else 0:.2f} q/M" for i, (q, tk) in sorted(agg.items())))
e15p_tok = 389 * 6 * 3600
ha, hb = D["R5"][0], D["E15P"][0]
qs = sorted(set(ha) & set(hb)); U = sum(ha[q] or hb[q] for q in qs)
print(f"  B with E15P tokens estimated at 389 tok/s x 6 h = {e15p_tok/1e6:.1f}M: +{U - sum(ha[q] for q in qs)} over R5 = "
      f"{(U - sum(ha[q] for q in qs))/(e15p_tok/1e6):.2f} q/M; +{U - sum(hb[q] for q in qs)} over E15P for R5's 11.65M = "
      f"{(U - sum(hb[q] for q in qs))/11.65:.2f} q/M")
