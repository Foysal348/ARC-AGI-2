import json, sys, csv
sys.stdout.reconfigure(encoding="utf-8")
E10 = "D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/"
SOL = json.load(open(E10 + "Experiment Review/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_solutions.json"))
U = {k: {tuple(x) for x in v} for k, v in json.load(open("D:/ARC AGI 2/resources/NVARC family/_comparison/units.json")).items()}
L = {r["task"]: r for r in csv.DictReader(open("D:/ARC AGI 2/resources/Puzzle Capability Ledger/ledger_all120.csv", encoding="utf-8"))}
def cell(k, t):
    n = len(SOL[t]); h = sum((t, i) in U[k] for i in range(n))
    return "✅" if h == n else ("❌" if h == 0 else f"🟡 {h}/{n}")
def solved(k, t): return any((t, i) in U[k] for i in range(len(SOL[t])))
groups = [("NVARC ও Q27 দুজনেই", lambda t: solved("NVARC", t) and (solved("DIRECT", t) or solved("POETIQ", t))),
          ("শুধু NVARC", lambda t: solved("NVARC", t) and not (solved("DIRECT", t) or solved("POETIQ", t))),
          ("শুধু Q27", lambda t: not solved("NVARC", t) and (solved("DIRECT", t) or solved("POETIQ", t))),
          ("কেউ পারেনি", lambda t: not solved("NVARC", t) and not (solved("DIRECT", t) or solved("POETIQ", t)))]
out = []
for g, f in groups:
    ts = sorted(t for t in SOL if f(t))
    out.append(f"\n### {g} ({len(ts)} টা task)\n\n| # | task | test | tier | NVARC | Q27 DIRECT | Q27 Poetiq | ARC Prize xhigh |\n|---|---|---|---|---|---|---|---|")
    for i, t in enumerate(ts, 1):
        ap = "✅" if L[t]["arcprize_xhigh"] == "1" else "❌"
        out.append(f"| {i} | `{t}` | {len(SOL[t])} | {L[t]['tier']} | {cell('NVARC', t)} | {cell('DIRECT', t)} | {cell('POETIQ', t)} | {ap} |")
md = "\n".join(out)
open("D:/ARC AGI 2/resources/NVARC family/NVARC vs Q27 EVAL120 tick table (2026-10-19).md", "w", encoding="utf-8").write(
    "# NVARC vs Q27 EVAL120: task-by-task (✅ all test inputs, 🟡 some, ❌ none)\n\nNVARC = `arc-2026d-with-codex`; Q27 DIRECT = E10 ORDER2 86k; Q27 Poetiq = E10 Poetiq R5; ARC Prize xhigh = official Qwen3.8-27B run (task level).\n" + md + "\n")
print(md)
