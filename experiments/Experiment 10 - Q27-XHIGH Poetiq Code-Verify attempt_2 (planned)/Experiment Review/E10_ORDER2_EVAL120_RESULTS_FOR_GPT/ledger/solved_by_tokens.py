"""Every solved test input (unit) of the 120 evaluation puzzles, sorted by the thinking tokens of its cheapest solve
(Claude, 2026-10-14, user request). Uses the runs in build_ledger.RUNS that record think tokens (direct runs);
solves from runs without token data (Poetiq, code loop, agent) and ARC Prize-only tasks are listed separately.
usage: python solved_by_tokens.py  -> solved_by_tokens.csv
"""
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bl", HERE / "build_ledger.py")
bl = importlib.util.module_from_spec(spec); spec.loader.exec_module(bl)
ch = json.load(open(bl.FX / "arc-agi_evaluation_challenges.json"))
a120 = {r["task"]: r for r in csv.DictReader(open(HERE / "ledger_all120.csv", encoding="utf-8"))}
oracle = {r["task"]: r for r in csv.DictReader(open(HERE / "oracle_order_120.csv", encoding="utf-8"))}
ORDER2 = "E10 ORDER2 EVAL120 direct 86k (all 120 tasks)"

solves = defaultdict(list)          # unit -> [(think, run, mode)]
notok = defaultdict(list)           # unit -> [run] (hit without token data)
attempts = defaultdict(int)
o2 = {}
for name, method, profile, kind, path in bl.RUNS:
    for (t, q), x in bl.load_run(kind, path, ch).items():
        u = f"{t}#{q}"; attempts[u] += 1
        if name == ORDER2:
            o2[u] = x
        if x["hit"]:
            if isinstance(x.get("think"), int):
                solves[u].append((x["think"], name, x.get("mode")))
            else:
                notok[u].append(name)

rows = []
for u, s in solves.items():
    s.sort(); t = u.split("#")[0]
    rows.append({"unit": u, "min_solve_think": s[0][0], "min_solve_mode": s[0][2], "min_solve_run": s[0][1],
                 "solves_with_tokens": len(s), "all_solve_thinks": ";".join(str(v[0]) for v in s),
                 "attempts_all_runs": attempts[u], "order2_think": o2[u]["think"] if u in o2 else "",
                 "order2_hit": o2[u]["hit"] if u in o2 else "", "arcprize_xhigh": a120[t]["arcprize_xhigh"],
                 "oracle_tier": oracle[t]["tier"], "other_solvers": ";".join(notok.get(u, []))})
rows.sort(key=lambda r: r["min_solve_think"])
with open(HERE / "solved_by_tokens.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

bands = [(0, 20000), (20000, 30000), (30000, 40000), (40000, 49152), (49152, 60000), (60000, 70000), (70000, 10**6)]
print(f"units solved with token data: {len(rows)} (tasks {len({r['unit'].split('#')[0] for r in rows})})")
for lo, hi in bands:
    b = [r for r in rows if lo < r["min_solve_think"] <= hi]
    print(f"\n({lo//1000}k, {min(hi, 77847)//1000 if hi < 10**6 else 'cap'}k]: {len(b)} units")
    print("  " + ", ".join(f"{r['unit']} {r['min_solve_think']/1000:.1f}k{'F' if r['min_solve_mode'] == 'forced_close' else ''}"
                           f"{'' if r['solves_with_tokens'] == 1 else ' x' + str(r['solves_with_tokens'])}" for r in b))
only = {u: v for u, v in notok.items() if u not in solves}
print("\nsolved only by runs without token data:", only)
solved_tasks = {r["unit"].split("#")[0] for r in rows} | {u.split("#")[0] for u in only}
ap_only = sorted(t for t in a120 if t not in solved_tasks and any(a120[t][k] == "1" for k in ("arcprize_xhigh", "arcprize_medium", "arcprize_low", "arcprize_none")))
print(f"\nARC Prize-only tasks ({len(ap_only)}), with ORDER2 86k thinking spent (failed):")
print("  " + ", ".join(f"{t} ({';'.join(str(o2[f'{t}#{q}']['think']//1000) + 'k' for q in range(len(ch[t]['test'])))})" for t in ap_only))
