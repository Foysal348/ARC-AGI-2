"""Per eval task: how many generated tokens each method needed for a CORRECT answer, over every run with token data
(user 2026-10-25). Writes tokens_to_solve.md + tokens_to_solve.csv next to this file.
- Duck: every attempt (branch_result.json) of every Duck run on eval tasks; tokens at the first accepted candidate that is
  correct on every test input of the task (the attempt's own generated-token counter).
- Poetiq: every run (poetiq_iterations.jsonl + poetiq_run_report.json); tokens = all units of the task in that run up to
  and including the first train-passing unit, counted only when the run's submission hit the task (any query).
- Direct 86k (E10 ORDER2 EVAL120): the one thinking of the task when its submission hit."""
import collections, csv, glob, json, statistics as st
from pathlib import Path
HERE = Path(__file__).resolve().parent
EXP = Path(r"D:/ARC AGI 2/experiments")
DATA = EXP / "Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data"
CH = json.load(open(DATA / "arc-agi_evaluation_challenges.json", encoding="utf-8"))
SOL = json.load(open(DATA / "arc-agi_evaluation_solutions.json", encoding="utf-8"))
NQ = {t: len(SOL[t]) for t in SOL}


def ok(outs, t):
    return all(q < len(outs or []) and outs[q] == g for q, g in enumerate(SOL[t]))


# ---- Duck
duck = collections.defaultdict(lambda: {"att": 0, "tok": []})
duck_runs = set()
for p in glob.glob(str(EXP / "*" / "Experiment Results" / "Kaggle Runs" / "*" / "q27x" / "duck" / "*_b*" / "branch_result.json")):
    r = json.load(open(p, encoding="utf-8"))
    t = r.get("task_id")
    if t not in SOL:
        continue
    duck_runs.add(Path(p).parents[3].name)
    duck[t]["att"] += 1
    cs = [v for v in (r.get("checked") or []) + (r.get("submissions") or []) if isinstance(v, dict)]
    good = [int(v.get("tokens") or 0) for v in cs if v.get("accepted") and ok(v.get("test_outputs"), t)]
    if good:
        duck[t]["tok"].append(min(good))

# ---- Poetiq
poe = collections.defaultdict(lambda: {"runs": 0, "tok": []})
poe_runs = []
for p in glob.glob(str(EXP / "*" / "Experiment Results" / "Kaggle Runs" / "*" / "q27x" / "poetiq_iterations.jsonl")):
    root = Path(p).parent
    rp = root / "poetiq_run_report.json"
    if not rp.exists():
        continue
    rep = json.load(open(rp, encoding="utf-8"))
    hit = collections.defaultdict(bool)
    for u in rep.get("eval_unit_records") or []:
        hit[u["task"]] |= bool(u["hit"])
    if not hit:
        continue
    poe_runs.append(root.parent.name)
    acc, first = collections.defaultdict(int), {}
    for x in (json.loads(l) for l in open(p, encoding="utf-8")):
        t = x["task_id"]
        acc[t] += int(x.get("p1_tokens") or 0) + int(x.get("p2_tokens") or 0)
        if x.get("train_passed") and t not in first:
            first[t] = acc[t]
    for t in set(acc) & set(SOL):
        poe[t]["runs"] += 1
        if hit[t]:
            poe[t]["tok"].append(first.get(t, acc[t]))   # a hit without a train pass: all tokens spent on the task

# ---- Direct 86k
direct = {}
for r in csv.DictReader(open(EXP / "Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/"
                              "E10_ORDER2_EVAL120_RESULTS_FOR_GPT/analysis/ORDER2 EVAL120 per-task admission vs outcome (2026-10-14).csv",
                              encoding="utf-8")):
    h = int(r["hits"].split("/")[0])
    direct[r["task"]] = (sum(int(x) for x in (r["think"] or "0").split(";") if x), h > 0)   # one thinking per test input

k = lambda v: f"{v / 1e3:.0f}k"
rows = []
for t in sorted(SOL):
    d, p = duck.get(t, {"att": 0, "tok": []}), poe.get(t, {"runs": 0, "tok": []})
    dt, pt = sorted(d["tok"]), sorted(p["tok"])
    di = direct.get(t)
    alls = dt + pt + ([di[0]] if di and di[1] else [])
    rows.append(dict(task=t, queries=NQ[t], duck_correct=len(dt), duck_attempts=d["att"],
                     duck_tokens=" ".join(k(x) for x in dt), duck_min=min(dt) if dt else "",
                     poetiq_solved=len(pt), poetiq_runs=p["runs"], poetiq_tokens=" ".join(k(x) for x in pt),
                     poetiq_min=min(pt) if pt else "", direct_hit=(di[1] if di else ""), direct_tokens=(di[0] if di else ""),
                     min_any=min(alls) if alls else ""))
with open(HERE / "tokens_to_solve.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

solved = [r for r in rows if r["min_any"] != ""]
mins = sorted(r["min_any"] for r in solved)
band = lambda lo, hi: sum(lo < m <= hi for m in mins)
md = ["# Tokens to a correct answer, per eval task, every run with token data (2026-10-25)\n",
      f"Duck runs: {len(duck_runs)} ({sum(r['duck_attempts'] for r in rows)} attempts); Poetiq runs: {len(poe_runs)}; "
      "Direct: E10 ORDER2 86k (one thinking per task).",
      "Duck cell = tokens of each correct attempt (first correct accept), sorted; Poetiq cell = tokens per solving run "
      "(all units of the task up to the first train pass); Direct = the task's thinking when it hit.\n",
      f"- tasks solved by any method in any run: {len(solved)}/120; never: {120 - len(solved)}",
      f"- cheapest correct answer per task: <= 10k: {band(0, 10e3)}, 10-20k: {band(10e3, 20e3)}, 20-30k: {band(20e3, 30e3)}, "
      f"30-45k: {band(30e3, 45e3)}, 45-60k: {band(45e3, 60e3)}, 60-90k: {band(60e3, 90e3)}, > 90k: {band(90e3, 1e9)}",
      f"- Duck correct attempts: {sum(r['duck_correct'] for r in rows)} of {sum(r['duck_attempts'] for r in rows)}; "
      f"tokens median {st.median([x for r in rows for x in duck[r['task']]['tok']]) / 1e3:.1f}k",
      f"- Poetiq solving runs: {sum(r['poetiq_solved'] for r in rows)}; tokens median "
      f"{st.median([x for r in rows for x in poe[r['task']]['tok']]) / 1e3:.1f}k\n",
      "| # | task | q | Duck correct / attempts | Duck tokens (each correct attempt) | Poetiq solved / runs | Poetiq tokens (each solving run) | Direct 86k | cheapest |",
      "|---|---|---|---|---|---|---|---|---|"]
order = sorted(rows, key=lambda r: (r["min_any"] == "", r["min_any"] if r["min_any"] != "" else 0, r["task"]))
for i, r in enumerate(order, 1):
    dr = ("—" if r["direct_hit"] == "" else (k(r["direct_tokens"]) if r["direct_hit"] else f"wrong ({k(r['direct_tokens'])})"))
    md.append(f"| {i} | {r['task']} | {r['queries']} | {r['duck_correct']}/{r['duck_attempts']} | {r['duck_tokens'] or '—'} | "
              f"{r['poetiq_solved']}/{r['poetiq_runs']} | {r['poetiq_tokens'] or '—'} | {dr} | "
              f"{k(r['min_any']) if r['min_any'] != '' else 'never'} |")
(HERE / "tokens_to_solve.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))
