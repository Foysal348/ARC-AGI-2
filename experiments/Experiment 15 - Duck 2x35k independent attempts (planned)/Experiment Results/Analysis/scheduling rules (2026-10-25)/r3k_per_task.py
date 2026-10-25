"""R3K1 vs R3K2 per task, side by side (user 2026-10-25): every attempt's tokens and outcome, and the token count at the
first correct accepted candidate. Writes r3k_per_task.md and r3k_per_task.csv next to this file."""
import csv, glob, json
from pathlib import Path
from load_attempts import SOL, cands, qhits

HERE = Path(__file__).resolve().parent
K15 = Path(r"D:\ARC AGI 2\experiments\Experiment 15 - Duck 2x35k independent attempts (planned)\Experiment Results\Kaggle Runs")
RUNS = {"R3K1": K15 / "E15B EVAL120 50K45K R3K1 commit1 (esfersami50, 2026-10-25, 6baa9361)",
        "R3K2": K15 / "E15B EVAL120 50K45K R3K2 commit1 (hosen42, 2026-10-25, c058d58b)"}


def att(r, tid):
    """(tokens used, code, tokens at the first correct accepted candidate)"""
    tok = int(r.get("generated_tokens") or 0)
    acc = [v for v in cands(r) if v.get("accepted")]
    nq = len(SOL[tid])
    ok = lambda v: all(qhits(v.get("test_outputs"), tid)[q] for q in range(nq))
    good = sorted([v for v in acc if ok(v)], key=lambda v: int(v.get("tokens") or 10 ** 9))
    if good:
        g = good[0]
        return tok, ("OK-weak" if g.get("weak") else "OK"), int(g.get("tokens") or 0)
    if acc:
        return tok, ("WRONG-weak" if all(v.get("weak") for v in acc) else "WRONG"), None
    part = any(any(qhits(v.get("test_outputs"), tid)) for v in cands(r))
    return tok, ("partial-right" if part else "no-accept"), None


data = {}
for run, root in RUNS.items():
    rep = json.load(open(root / "q27x" / "e12_run_report.json", encoding="utf-8")) if (root / "q27x" / "e12_run_report.json").exists() else None
    if rep is None:
        import zipfile
        z = zipfile.ZipFile(root / "Q27X_E12_DUCK_BUNDLE.zip")
        rep = json.loads(z.read([n for n in z.namelist() if n.endswith("e12_run_report.json")][0]))
    hits = {}
    for u in rep["eval_unit_records"]:
        hits.setdefault(u["task"], [0, 0])
        hits[u["task"]][0] += bool(u["hit"])
        hits[u["task"]][1] += 1
    order = {t: i + 1 for i, t in enumerate(rep.get("order") or [])}
    for p in glob.glob(str(root / "q27x" / "duck" / "*_b*" / "branch_result.json")):
        r = json.load(open(p, encoding="utf-8"))
        data.setdefault(r["task_id"], {}).setdefault(run, {})[r["branch"]] = att(r, r["task_id"])
    for t, h in hits.items():
        data.setdefault(t, {}).setdefault(run, {})["hit"] = h
        data[t][run]["order"] = order.get(t)


def cell(d, b):
    if b not in d:
        return "—"
    tok, code, at = d[b]
    return f"{code} @{at / 1e3:.1f}k" if at is not None else f"{code} ({tok / 1e3:.1f}k)"


rows, md = [], ["| # | task | queries | R3K1 attempt 1 | R3K1 attempt 2 | R3K1 hit | R3K2 attempt 1 | R3K2 attempt 2 | R3K2 hit |",
                "|---|---|---|---|---|---|---|---|---|"]
def key(t):
    a, b = data[t].get("R3K1", {}), data[t].get("R3K2", {})
    sa = a.get("hit", [0, 1])[0] / a.get("hit", [0, 1])[1]
    sb = b.get("hit", [0, 1])[0] / b.get("hit", [0, 1])[1]
    return (-(sa + sb), -max(sa, sb), t)
for i, t in enumerate(sorted(data, key=key), 1):
    a, b = data[t].get("R3K1", {}), data[t].get("R3K2", {})
    ha, hb = a.get("hit", [0, 0]), b.get("hit", [0, 0])
    md.append(f"| {i} | {t} | {ha[1]} | {cell(a, 0)} | {cell(a, 1)} | {ha[0]}/{ha[1]} | {cell(b, 0)} | {cell(b, 1)} | {hb[0]}/{hb[1]} |")
    row = {"task": t, "queries": ha[1]}
    for run, d, h in (("R3K1", a, ha), ("R3K2", b, hb)):
        for br in (0, 1, 2):
            v = d.get(br)
            row[f"{run}_a{br + 1}_tokens"] = v[0] if v else ""
            row[f"{run}_a{br + 1}_outcome"] = v[1] if v else ""
            row[f"{run}_a{br + 1}_correct_at_tokens"] = v[2] if v and v[2] is not None else ""
        row[f"{run}_hit"] = f"{h[0]}/{h[1]}"
    rows.append(row)
both = sum(1 for t in data if data[t].get("R3K1", {}).get("hit", [0])[0] and data[t].get("R3K2", {}).get("hit", [0])[0])
only1 = sorted(t for t in data if data[t].get("R3K1", {}).get("hit", [0])[0] and not data[t].get("R3K2", {}).get("hit", [0])[0])
only2 = sorted(t for t in data if data[t].get("R3K2", {}).get("hit", [0])[0] and not data[t].get("R3K1", {}).get("hit", [0])[0])
none = sum(1 for t in data if not data[t].get("R3K1", {}).get("hit", [0])[0] and not data[t].get("R3K2", {}).get("hit", [0])[0])
ats = [d[b][2] for t in data for run in ("R3K1", "R3K2") for d in [data[t].get(run, {})] for b in (0, 1) if b in d and d[b][2] is not None]
ats.sort()
summ = [f"Tasks with a hit in both runs: {both}; only R3K1: {len(only1)} {only1}; only R3K2: {len(only2)} {only2}; in neither: {none}.",
        f"Correct accepted answers: {len(ats)}; tokens at the first correct accept: p25 {ats[len(ats) // 4] / 1e3:.1f}k, "
        f"median {ats[len(ats) // 2] / 1e3:.1f}k, p75 {ats[3 * len(ats) // 4] / 1e3:.1f}k, max {ats[-1] / 1e3:.1f}k; "
        f"<= 20k: {sum(a <= 20000 for a in ats)}, 20-35k: {sum(20000 < a <= 35000 for a in ats)}, > 35k: {sum(a > 35000 for a in ats)}."]
legend = ("Cell = outcome of the attempt. `OK @Xk` = an accepted answer that is correct, X = generated tokens when it was "
          "accepted; `OK-weak` = correct but accepted only after the commit text; `WRONG` / `WRONG-weak` = accepted, wrong "
          "(tokens used in brackets); `partial-right` = no accept, but a partial candidate had a right query; `no-accept`; "
          "— = the attempt did not run. hit = queries right in the submission.")
(HERE / "r3k_per_task.md").write_text("# R3K1 vs R3K2 per task (EVAL120, 6 h, 2026-10-25)\n\n" + legend + "\n\n" +
                                       "\n".join(summ) + "\n\n" + "\n".join(md) + "\n", encoding="utf-8")
with open(HERE / "r3k_per_task.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print("\n".join(summ))
print("\n".join(md))
