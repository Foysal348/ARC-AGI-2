"""Lever #1 (user 2026-10-25): would seeding the Duck with Poetiq's candidates help? From ALL past runs (no GPU):
per eval task, Poetiq's best train score and train passes over every Poetiq run (poetiq_iterations.jsonl), Poetiq and
Duck hits over every run (oracle_by_family), and Duck's tokens to the first correct accept (R3K + G4B attempts).
Writes seed_potential.md next to this file."""
import collections, contextlib, glob, io, json, statistics as st, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
with contextlib.redirect_stdout(io.StringIO()):
    import oracle_by_family as O
sys.path.insert(0, r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Results/Analysis/scheduling rules (2026-10-25)")
from load_attempts import SOL, cands, qhits  # noqa: E402

EXP = Path(r"D:/ARC AGI 2/experiments")
CH = O.CH
NQ = {t: len(CH[t]["test"]) for t in CH}
fam_hits = collections.defaultdict(lambda: collections.defaultdict(set))   # family -> task -> {queries}
for n, f, r in O.runs:
    for (t, q), h in r.items():
        if h:
            fam_hits[f][t].add(q)
PO, DU = "Poetiq / code + train verify", "Duck REPL agent (ours)"
# Poetiq train scores / passes / tokens per task over every Poetiq run
best = collections.defaultdict(float)
passed = collections.defaultdict(int)
units = collections.defaultdict(int)
tok_to_pass = collections.defaultdict(list)
for p in glob.glob(str(EXP / "*" / "Experiment Results" / "Kaggle Runs" / "*" / "q27x" / "poetiq_iterations.jsonl")):
    recs = [json.loads(l) for l in open(p, encoding="utf-8")]
    acc = collections.defaultdict(int)
    for r in recs:
        t = r["task_id"]
        if t not in CH:
            continue
        units[t] += 1
        acc[t] += int(r.get("p1_tokens") or 0) + int(r.get("p2_tokens") or 0)
        if r.get("score") is not None:
            best[t] = max(best[t], float(r["score"]))
        if r.get("train_passed"):
            if not passed[t] or True:
                tok_to_pass[t].append(acc[t])
            passed[t] += 1
# Duck: tokens to the first correct accept over the R3K and G4B attempts
K14 = EXP / "Experiment 14 - Optimized Engine on 4xL4 (planned)/Experiment Results/Kaggle Runs"
K15 = EXP / "Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Results/Kaggle Runs"
duck_tok = collections.defaultdict(list)
duck_att = collections.defaultdict(int)
for root in [K14 / "EVAL120G4B XHIGH E16 (hosen42, 2026-10-24)", K14 / "EVAL120G4B XHIGH noE16 (hosen42, 2026-10-24)"] + \
        [Path(x) for x in glob.glob(str(K15 / "E15B EVAL120 50K45K R3K*"))]:
    for p in glob.glob(str(root / "q27x" / "duck" / "*_b*" / "branch_result.json")):
        r = json.load(open(p, encoding="utf-8"))
        t = r["task_id"]
        duck_att[t] += 1
        good = [v for v in cands(r) if v.get("accepted") and all(qhits(v.get("test_outputs"), t)[q] for q in range(NQ[t]))]
        if good:
            duck_tok[t].append(min(int(v.get("tokens") or 0) for v in good))

rows = []
for t in sorted(CH):
    ph, dh = fam_hits[PO].get(t, set()), fam_hits[DU].get(t, set())
    rows.append(dict(task=t, nq=NQ[t], p_hit=len(ph) / NQ[t], d_hit=len(dh) / NQ[t], p_best=round(best.get(t, 0.0), 3),
                     p_pass=passed.get(t, 0), p_units=units.get(t, 0), d_att=duck_att.get(t, 0),
                     d_tok=min(duck_tok[t]) if duck_tok.get(t) else None, d_solves=len(duck_tok.get(t, []))))


def cat(r):
    if r["p_hit"] and not r["d_hit"]:
        return "A: Poetiq solved, Duck never"
    if r["p_pass"] and not r["p_hit"] and not r["d_hit"]:
        return "B: Poetiq train-passed but wrong on test, Duck never"
    if not r["p_pass"] and r["p_best"] >= 0.9 and not r["d_hit"]:
        return "C: Poetiq near miss (train score >= 0.9), Duck never"
    if not r["p_pass"] and 0 < r["p_best"] < 0.9 and not r["d_hit"]:
        return "D: Poetiq partial (0 < score < 0.9), Duck never"
    if r["p_hit"] and r["d_hit"]:
        return "E: both solved"
    if r["d_hit"] and not r["p_hit"]:
        return "F: Duck solved, Poetiq never"
    return "G: neither (Poetiq score 0 or no unit)"


out = ["# Seeding the Duck with Poetiq candidates: potential from all past runs (2026-10-25)\n",
       f"Poetiq runs read: every poetiq_iterations.jsonl under experiments/ ({sum(units.values())} units); Duck tokens from the "
       f"G4B XHIGH x2 and R3K x2 EVAL120 runs ({sum(duck_att.values())} attempts). Hits = any run of the family (oracle).\n",
       "| category | tasks | task-norm (Poetiq hit) | example tasks |", "|---|---|---|---|"]
by = collections.defaultdict(list)
for r in rows:
    by[cat(r)].append(r)
for k in sorted(by):
    v = by[k]
    out.append(f"| {k} | {len(v)} | {sum(r['p_hit'] for r in v):.1f} | {', '.join(r['task'] for r in v[:10])} |")
E = by["E: both solved"]
pairs = [(r["d_tok"], r) for r in E if r["d_tok"]]
out.append(f"\n- both solved: {len(E)}; Duck tokens to the first correct accept (best of its attempts): median "
           f"{st.median([p for p, _ in pairs]) / 1e3 if pairs else 0:.1f}k; Duck attempts per task {st.mean([r['d_att'] for r in E]):.1f}, "
           f"correct in {sum(r['d_solves'] for r in E)}/{sum(r['d_att'] for r in E)} attempts")
A = by["A: Poetiq solved, Duck never"]
out.append(f"- A (seed = Poetiq's verified program, the Duck only checks / submits): {len(A)} tasks, task-norm "
           f"{sum(r['p_hit'] for r in A):.1f}; Duck attempts spent on them without a correct answer: {sum(r['d_att'] for r in A)}")
C = by["C: Poetiq near miss (train score >= 0.9), Duck never"]
out.append(f"- C (seed = near-miss program to repair): {len(C)} tasks: {[(r['task'], r['p_best']) for r in C]}")
B = by["B: Poetiq train-passed but wrong on test, Duck never"]
out.append(f"- B (a train-passing but wrong program: a seed could mislead): {len(B)} tasks: {[r['task'] for r in B]}")
out.append("\n## Per task\n\n| task | queries | Poetiq hit | Poetiq best train score | Poetiq train passes / units | Duck hit | "
           "Duck correct / attempts | Duck best tokens | category |")
out.append("|---|---|---|---|---|---|---|---|---|")
for r in sorted(rows, key=lambda r: (cat(r), r["task"])):
    out.append(f"| {r['task']} | {r['nq']} | {r['p_hit']:.2f} | {r['p_best']} | {r['p_pass']}/{r['p_units']} | {r['d_hit']:.2f} | "
               f"{r['d_solves']}/{r['d_att']} | {'' if r['d_tok'] is None else round(r['d_tok'] / 1e3, 1)} | {cat(r)[0]} |")
(HERE / "seed_potential.md").write_text("\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out[:20]))
