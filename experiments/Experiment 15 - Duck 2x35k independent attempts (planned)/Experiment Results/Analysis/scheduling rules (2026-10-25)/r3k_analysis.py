"""R3K1 / R3K2 EVAL120 6-h runs (2026-10-25): per-attempt yields, W1 weak accepts, selection, tokens and time.
Reads branch_result.json of every attempt and the run report. Writes r3k_analysis_out.md next to this file."""
import collections, glob, json, statistics as st, zipfile
from pathlib import Path
from load_attempts import SOL, cands, qhits

HERE = Path(__file__).resolve().parent
K15 = Path(r"D:\ARC AGI 2\experiments\Experiment 15 - Duck 2x35k independent attempts (planned)\Experiment Results\Kaggle Runs")
RUNS = {"R3K1 (after3, Sami)": K15 / "E15B EVAL120 50K45K R3K1 commit1 (esfersami50, 2026-10-25, 6baa9361)",
        "R3K2 (before3, Hamid)": K15 / "E15B EVAL120 50K45K R3K2 commit1 (hosen42, 2026-10-25, c058d58b)"}
out = []
P = out.append


def report(root):
    p = root / "q27x" / "e12_run_report.json"
    if p.exists():
        return json.load(open(p, encoding="utf-8"))
    z = zipfile.ZipFile(root / "Q27X_E12_DUCK_BUNDLE.zip")
    return json.loads(z.read([n for n in z.namelist() if n.endswith("e12_run_report.json")][0]))


def pct(a, b):
    return f"{a}/{b} ({100 * a / b:.0f} %)" if b else "0/0"


ALL = {}
for name, root in RUNS.items():
    rep = report(root)
    rows = []
    for p in sorted(glob.glob(str(root / "q27x" / "duck" / "*_b*" / "branch_result.json"))):
        r = json.load(open(p, encoding="utf-8"))
        tid, b = r["task_id"], r["branch"]
        cs = cands(r)
        acc = [v for v in cs if v.get("accepted")]
        strong = [v for v in acc if not v.get("weak")]
        weak = [v for v in acc if v.get("weak")]
        nq = len(SOL[tid])
        corr = lambda vs: [any(qhits(v.get("test_outputs"), tid)[q] for v in vs) for q in range(nq)]
        rows.append(dict(task=tid, b=b, tokens=int(r.get("generated_tokens") or 0), sec=float(r.get("sec") or 0),
                         acc=bool(acc), strong=bool(strong), weak=bool(weak) and not strong,
                         strong_ok=all(corr(strong)) if strong else False, weak_ok=all(corr(weak)) if weak else False,
                         acc_ok=all(corr(acc)) if acc else False, any_ok=any(corr(cs)), nq=nq))
    ALL[name] = (rep, rows)
    units = rep.get("eval_unit_records") or []
    P(f"\n## {name}\n")
    P(f"- branch results read: {len(rows)} of {rep['e12_run'].get('branches_done')} done "
      f"({'COMPLETE' if len(rows) == rep['e12_run'].get('branches_done') else 'PARTIAL download'})")
    P(f"- score: {rep['eval_query_hits']}/{rep['eval_query_total']} queries, task-norm {rep['eval_task_norm_pct']} %, "
      f"pass1 {rep['eval_pass1_hits']}, pass2 {rep['eval_pass2_hits']}")
    s = rep["e12_run"].get("rev6_scheduler") or {}
    P(f"- scheduler: deferred attempt 2 {s.get('b1_deferred')} (started {s.get('b1_deferred_started')}), attempts >= 3 "
      f"started {s.get('extra_started')}, weak accepts {s.get('weak_accepts')}, weak-only tasks continued "
      f"{s.get('weak_only_tasks_continued')}, queue left {s.get('heap_left')}")
    P("\n| attempt | n | tokens mean | min mean | any accept | strong accept | strong correct | weak-only | weak correct | any candidate correct |")
    P("|---|---|---|---|---|---|---|---|---|---|")
    for b in sorted({r["b"] for r in rows}):
        R = [r for r in rows if r["b"] == b]
        sa = [r for r in R if r["strong"]]
        wo = [r for r in R if r["weak"]]
        P(f"| {b + 1} | {len(R)} | {st.mean(r['tokens'] for r in R) / 1e3:.1f}k | {st.mean(r['sec'] for r in R) / 60:.1f} | "
          f"{pct(sum(r['acc'] for r in R), len(R))} | {pct(len(sa), len(R))} | {pct(sum(r['strong_ok'] for r in sa), len(sa))} | "
          f"{pct(len(wo), len(R))} | {pct(sum(r['weak_ok'] for r in wo), len(wo))} | {pct(sum(r['any_ok'] for r in R), len(R))} |")
    # tasks: solved by which attempt, selection loss
    bytask = collections.defaultdict(dict)
    for r in rows:
        bytask[r["task"]][r["b"]] = r
    hit_t = {u["task"] for u in units if u.get("hit")}
    any_ok_t = {t for t, d in bytask.items() if any(x["any_ok"] for x in d.values())}
    P(f"\n- tasks with a hit (any query): {len(hit_t)}; tasks where some candidate of some attempt was right: {len(any_ok_t)}; "
      f"right somewhere but no hit: {sorted(any_ok_t - hit_t)}")
    a2_rescue = [t for t, d in bytask.items() if 1 in d and d[1]["any_ok"] and not (0 in d and d[0]["any_ok"])]
    P(f"- attempt 2 right where attempt 1 was not: {len(a2_rescue)} {sorted(a2_rescue)}")
    w_cont = [t for t, d in bytask.items() if 0 in d and d[0]["weak"]]
    P(f"- attempt 1 weak-only (W1 continued): {len(w_cont)}; its weak answer right: "
      f"{sum(bytask[t][0]['weak_ok'] for t in w_cont)}; attempt 2 ran: {sum(1 in bytask[t] for t in w_cont)}; "
      f"attempt 2 strong-correct: {sum(1 in bytask[t] and bytask[t][1]['strong_ok'] for t in w_cont)}")
    tot = sum(r["tokens"] for r in rows)
    P(f"- generated tokens (read attempts): {tot / 1e6:.2f} M; mean per attempt {tot / max(1, len(rows)) / 1e3:.1f}k; "
      f"attempts at >= 98 % of budget: {sum(r['tokens'] >= (49000 if r['b'] == 0 else 44100) for r in rows)} of {len(rows)}")
    us = rep.get("usage_all") or {}
    P(f"- usage_all: completion {us.get('completion_tokens')}, prompt {us.get('prompt_tokens')}, cached {us.get('cached_pct')} %, "
      f"requests {us.get('requests')}; finished at {rep['e12_run'].get('finished_sec')} s")
# overlap of the two runs (identical work: a1 everywhere + a2 on non-accepted)
if len(ALL) == 2:
    (n1, (r1, _)), (n2, (r2, _)) = ALL.items()
    h1 = {(u["task"], u["query"]) for u in r1["eval_unit_records"] if u["hit"]}
    h2 = {(u["task"], u["query"]) for u in r2["eval_unit_records"] if u["hit"]}
    P(f"\n## Overlap\n\n- queries hit: {n1} {len(h1)}, {n2} {len(h2)}, both {len(h1 & h2)}, union {len(h1 | h2)}, "
      f"only R3K1 {len(h1 - h2)}, only R3K2 {len(h2 - h1)}")
(HERE / "r3k_analysis_out.md").write_text("# R3K1 / R3K2 EVAL120 6-h runs (2026-10-25)\n" + "\n".join(out) + "\n", encoding="utf-8")
print("\n".join(out))
