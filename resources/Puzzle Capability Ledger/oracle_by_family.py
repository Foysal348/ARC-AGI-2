"""Oracle solves by method family (2026-10-25, user: Duck oracle + other methods' oracles + total oracle).

Oracle = a test query (unit) counts as solved if ANY run of that family solved it. Task-norm = per task, solved queries /
queries, summed over the 120 evaluation tasks. Sources: every run of build_ledger.RUNS (to 2026-10-15), the later Poetiq /
SGLang runs, every Duck run with eval_unit_records, and ARC Prize's verified run (task level only: a passed task counts as
all of its queries solved).
usage: python oracle_by_family.py   (writes oracle_by_family.json and oracle_units_by_family.csv next to this file)"""
import csv, json
from collections import defaultdict
from pathlib import Path
import build_ledger as BL

HERE = Path(__file__).resolve().parent
EXP = BL.EXP
CH = json.load(open(BL.FX / "arc-agi_evaluation_challenges.json"))
UNITS = [(t, q) for t in sorted(CH) for q in range(len(CH[t]["test"]))]
E10K = EXP / "Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Results/Kaggle Runs"
E11K = EXP / "Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Results/Kaggle Runs"
LATER = [   # non-Duck runs after the ledger's last update (2026-10-15)
    ("E10 W3H low 3h EVAL120 (Poetiq, effort low)", "poetiq", E10K / "GPU EVAL120 W3H low 3h (esfersami50 q27-xhigh-e10-w3h-low-3h)/q27x/poetiq_run_report.json"),
    ("E10 D66 HARD66 77.8k (Poetiq)", "poetiq", E10K / "GPU HARD66 D66 77.8k (foysal q27-xhigh-e10-d66-hard-deep)/q27x/poetiq_run_report.json"),
    ("E10 L66 HARD66 122.9k (Poetiq)", "poetiq", E10K / "GPU HARD66 L66 122.9k (foysal q27-xhigh-e10-l66-hard-120k)/q27x/poetiq_run_report.json"),
    ("E11-05 EVAL30 (SGLang Poetiq S1)", "poetiq", E11K / "E11-05 EVAL30 run1 (esfersami50, 2026-10-18)/q27x/poetiq_run_report.json"),
    ("E11-05 S1COMMIT4 (SGLang Poetiq S1)", "poetiq", E11K / "E11-05 S1COMMIT4 run1 (foysal, 2026-10-18)/q27x/poetiq_run_report.json"),
    ("E11-06 P1-60K EVAL14 (SGLang Poetiq P1)", "poetiq", E11K / "E11-06 P1-60K EVAL14 run1 (esfersami50, 2026-10-19)/q27x/poetiq_run_report.json"),
]


def family(name, method):
    m = (name + " " + method).lower()
    if "poetiq" in m:
        return "Poetiq / code + train verify"
    if "direct" in m:
        return "Direct reasoning"
    return "Other code / agent (R6 loop, REPL agent v1.8-1.9)"


runs = []          # (name, family, {(t, q): hit})
for name, method, profile, kind, path in BL.RUNS:
    res = BL.load_run(kind, path, CH)
    runs.append((name, family(name, method), {k: v["hit"] for k, v in res.items()}))
for name, fam, path in LATER:
    res = BL.load_run("poetiq_report", path, CH)
    runs.append((name, "Poetiq / code + train verify", {k: v["hit"] for k, v in res.items()}))
seen = set()
for p in sorted(EXP.glob("*/Experiment Results/Kaggle Runs/*/q27x/e12_run_report.json")):
    rep = json.load(open(p, encoding="utf-8"))
    recs = rep.get("eval_unit_records") or []
    key = (rep.get("run_id"), rep.get("profile"))
    if not recs or key in seen:
        continue
    seen.add(key)
    runs.append(("Duck " + p.parent.parent.name, "Duck REPL agent (ours)",
                 {(r["task"], int(r["query"])): int(bool(r["hit"])) for r in recs}))
AP = json.load(open(HERE / "arcprize_qwen38_27b_public_eval.json"))
for lv in ("xhigh", "medium", "low", "none"):
    runs.append((f"ARC Prize {lv}", "ARC Prize verified run (others, task level)",
                 {(t, q): int(bool(AP["tasks"][t][lv])) for t, q in UNITS if t in AP["tasks"]}))

fams = defaultdict(list)
for n, f, r in runs:
    fams[f].append((n, r))


def oracle(run_list):
    att, sol = set(), set()
    for _, r in run_list:
        for k, h in r.items():
            att.add(k)
            if h:
                sol.add(k)
    return att, sol


def tasknorm(sol):
    return sum(sum((t, q) in sol for q in range(len(CH[t]["test"]))) / len(CH[t]["test"]) for t in CH)


groups = {f: v for f, v in sorted(fams.items())}
ours = [x for f, v in fams.items() if not f.startswith("ARC Prize") for x in v]
groups["ALL OUR RUNS (every method)"] = ours
groups["Our non-Duck runs"] = [x for f, v in fams.items() if not f.startswith(("ARC Prize", "Duck")) for x in v]
groups["TOTAL ORACLE (ours + ARC Prize)"] = [x for v in fams.values() for x in v]
out = {}
print("| group | runs | queries attempted (of 172) | tasks touched (of 120) | **oracle queries** | **oracle task-norm** | % of 120 |")
print("|---|---|---|---|---|---|---|")
S = {}
for g, rl in groups.items():
    att, sol = oracle(rl)
    S[g] = sol
    tn = tasknorm(sol)
    out[g] = {"runs": [n for n, _ in rl], "attempted_units": len(att), "solved_units": len(sol), "task_norm": round(tn, 2),
              "tasks_touched": len({t for t, _ in att})}
    print(f"| {g} | {len(rl)} | {len(att)} | {len({t for t, _ in att})} | **{len(sol)}** | **{tn:.2f}** | {100 * tn / 120:.1f} % |")
D = S["Duck REPL agent (ours)"]
O = S["Our non-Duck runs"]
A = S["ARC Prize verified run (others, task level)"]
T = S["TOTAL ORACLE (ours + ARC Prize)"]
print("\n| overlap (queries) | count | task-norm |")
print("|---|---|---|")
for name, s in (("Duck only (no other method of ours, no ARC Prize)", D - O - A), ("Duck and our other methods", D & O),
                ("our other methods only (not Duck, not ARC Prize)", O - D - A), ("ARC Prize only (none of ours)", A - D - O),
                ("solved by none of ours but by ARC Prize", A - (D | O)), ("ours (any) but not ARC Prize", (D | O) - A)):
    print(f"| {name} | {len(s)} | {tasknorm(s):.2f} |")
print("\nDuck runs counted:")
for n, r in fams["Duck REPL agent (ours)"]:
    print(f"  {n}: {sum(r.values())}/{len(r)} queries")
out["overlap"] = {"duck_only": sorted(f"{t}#{q}" for t, q in D - O - A), "arcprize_only": sorted(f"{t}#{q}" for t, q in A - D - O)}
json.dump(out, open(HERE / "oracle_by_family.json", "w"), indent=1)
with open(HERE / "oracle_units_by_family.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["task", "query"] + list(groups))
    for t, q in UNITS:
        w.writerow([t, q] + [int((t, q) in S[g]) for g in groups])
