"""Analysis of the D66 (77,824 thinking) and L66 (122,880 thinking) deep runs on the 66 eval tasks R5 did not fully solve.

Reads each run's q27x/ outputs: poetiq_iterations.jsonl (per-iteration test_correct etc.), poetiq_attempt_curve.jsonl,
poetiq_run_report.json. R5 is the baseline (its final attempts define the 94 R5-wrong inputs and the 15
train-passing but test-wrong tasks). Never executes model code. Selection is conditioned on R5 failure on purpose:
every rate below is P(... | R5 failed), not an ARC-wide rate.

usage: python -I analyze_d66_l66.py --r5 R5_q27x_dir --d66 D66_q27x_dir [--l66 L66_q27x_dir] --solutions eval_solutions.json
                                   --out report_dir
"""
import argparse, collections, json, os, statistics as st

ap = argparse.ArgumentParser()
ap.add_argument("--r5", required=True)
ap.add_argument("--d66", required=True)
ap.add_argument("--l66")
ap.add_argument("--solutions", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
SOL = json.load(open(a.solutions, encoding="utf-8"))
SLOTS = {"D66": 16, "L66": 12}


def jl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()] if os.path.exists(path) else []


# ---------------------------------------------------------------- R5 baseline on the hard tasks
r5_final = json.load(open(os.path.join(a.r5, "poetiq_final_generated.json"), encoding="utf-8"))
r5_rep = json.load(open(os.path.join(a.r5, "poetiq_run_report.json"), encoding="utf-8"))
r5_hit = {t: [SOL[t][q] in (r["attempt_1"], r["attempt_2"]) for q, r in enumerate(rows)] for t, rows in r5_final.items()}
HARD = sorted(t for t, h in r5_hit.items() if not all(h))
R5_WRONG = [(t, q) for t in HARD for q, h in enumerate(r5_hit[t]) if not h]
R5_RIGHT = [(t, q) for t in HARD for q, h in enumerate(r5_hit[t]) if h]
r5_status = {k.split("~")[0]: v.get("status") for k, v in r5_rep["expert_states"].items()}
TPTW = sorted(t for t in HARD if r5_status.get(t) == "train_verified" and not any(r5_hit[t]))   # train pass, test wrong
N_IN = sum(len(SOL[t]) for t in HARD)
L = []
P = lambda *x: L.extend(str(v) for v in x)      # one argument = one line
P("# D66 / L66 deep-run analysis", "")
P(f"Hard tasks: {len(HARD)} with {N_IN} inputs. R5 had {len(R5_RIGHT)} of them right and {len(R5_WRONG)} wrong.",
  f"{len(TPTW)} zero-score tasks were train-verified in R5 (train pass, test wrong).", "")
P("All rates are conditioned on R5's failure, so they are NOT ARC-wide rates. A D66/L66 'solve' counts as such only "
  "for this configuration (S1 fixes + fresh seeds), not as a pure seed effect.", "")


def load(name, d):
    if not d or not os.path.exists(os.path.join(d, "poetiq_iterations.jsonl")):
        return None
    it = [r for r in jl(os.path.join(d, "poetiq_iterations.jsonl")) if r["task_id"] in SOL]
    cur = jl(os.path.join(d, "poetiq_attempt_curve.jsonl"))
    rep = json.load(open(os.path.join(d, "poetiq_run_report.json"), encoding="utf-8")) if os.path.exists(
        os.path.join(d, "poetiq_run_report.json")) else {}
    return {"name": name, "it": it, "curve": cur, "rep": rep, "slots": SLOTS.get(name, 16)}


RUNS = [r for r in (load("D66", a.d66), load("L66", a.l66)) if r]
SUMMARY = {}


def first_correct(run):
    """input -> (iteration, record) of the first iteration whose program output is correct (oracle)"""
    fc = {}
    for r in sorted(run["it"], key=lambda r: (r["task_id"], r["iteration"])):
        for q, ok in enumerate(r.get("test_correct") or []):
            if ok and (r["task_id"], q) not in fc:
                fc[(r["task_id"], q)] = r["iteration"]
    return fc


for run in RUNS:
    n, it = run["name"], run["it"]
    by_it = collections.defaultdict(list)
    for r in it:
        by_it[r["iteration"]].append(r)
    toks = sum(r["p1_tokens"] + r["p2_tokens"] for r in it)
    slot_h = sum(r["sec"] for r in it) / 3600
    fin = collections.Counter(r.get("p1_finish") or r["finish"] for r in it)
    fc = first_correct(run)
    # deployable: last attempt-curve entry per task (what the two attempts would score)
    last = {}
    for c in run["curve"]:
        last[c["task_id"]] = c
    dep_hits = sum(sum(c["hit"]) for c in last.values())
    dep_task = sum(sum(c["hit"]) / len(c["hit"]) for c in last.values() if c["hit"])
    P(f"## {n}", "")
    P(f"- Requests {len(it)}; tokens {toks / 1e6:.2f} M; slot-hours {slot_h:.1f}; finish {dict(fin)}")
    P(f"- Iterations per task: {dict(collections.Counter(collections.Counter(r['task_id'] for r in it).values()))}")
    P(f"- Deployable two-attempt result at the end: {dep_hits}/{N_IN} inputs, task-normalised {dep_task:.2f}/{len(HARD)} tasks")
    P(f"- Oracle (any program output correct, any iteration): {len(fc)}/{N_IN} inputs; "
      f"of the {len(R5_WRONG)} R5-wrong inputs: {sum(1 for x in R5_WRONG if x in fc)}")
    P("", "| Iteration | Requests | Slot-hours | Mean thinking | Forced at cap | Train pass | New correct inputs (oracle) | New per slot-hour | Of R5-wrong |",
      "|---|---|---|---|---|---|---|---|---|")
    rows = []
    for k in sorted(by_it):
        g = by_it[k]
        sh = sum(r["sec"] for r in g) / 3600
        new = [x for x, kk in fc.items() if kk == k]
        rows.append({"iteration": k, "requests": len(g), "slot_hours": round(sh, 2), "new": len(new),
                     "new_r5_wrong": sum(1 for x in new if x in R5_WRONG)})
        P(f"| {k} | {len(g)} | {sh:.1f} | {st.mean(r['p1_tokens'] for r in g):,.0f} | "
          f"{sum(r.get('p1_finish') == 'length' for r in g)} | "
          f"{sum(bool(r.get('train_passed')) for r in g)} | {len(new)} | {len(new) / sh if sh else 0:.2f} | "
          f"{rows[-1]['new_r5_wrong']} |")
    # compute -> score: deployable score against wall time (from the curve's t)
    P("", "Deployable score against run time (attempt curve):", "")
    pts = []
    for hour in range(1, 13):
        lt = {}
        for c in run["curve"]:
            if c["t"] <= hour * 3600:
                lt[c["task_id"]] = c
        pts.append((hour, sum(sum(c["hit"]) for c in lt.values())))
    P("| Hour | " + " | ".join(str(h) for h, _ in pts) + " |", "|---|" + "---|" * len(pts),
      "| Inputs right | " + " | ".join(str(v) for _, v in pts) + " |", "")
    # train pass but test wrong in R5 (Q3)
    cat = collections.Counter()
    per = {}
    for t in TPTW:
        g = [r for r in it if r["task_id"] == t]
        if not g:
            per[t] = "not reached"; cat["not reached"] += 1; continue
        if not any(r.get("has_python") for r in g):
            c = "no code"
        else:
            tp = [r for r in g if r.get("train_passed")]
            tc = [r for r in g if any(r.get("test_correct") or [])]
            if tp and any(any(r.get("test_correct") or []) for r in tp):
                c = "train pass + test right"
            elif tp:
                c = "train pass + test wrong (overfit again)"
            elif tc:
                c = "train fail + test right"
            else:
                c = "train fail + test wrong"
        per[t] = c; cat[c] += 1
    P(f"R5 train-pass/test-wrong tasks ({len(TPTW)}): {dict(cat)}", "")
    SUMMARY[n] = {"requests": len(it), "tokens_M": round(toks / 1e6, 2), "slot_hours": round(slot_h, 1), "finish": dict(fin),
                  "deployable_inputs": dep_hits, "deployable_task_norm": round(dep_task, 2), "oracle_inputs": len(fc),
                  "oracle_r5_wrong": sum(1 for x in R5_WRONG if x in fc), "per_iteration": rows, "time_curve": pts,
                  "tptw": per}

# ---------------------------------------------------------------- D66 vs L66 (paired, same tasks, iteration 1 and overall)
if len(RUNS) == 2:
    D, Lr = RUNS
    def it1(run):
        return {r["task_id"]: r for r in run["it"] if r["iteration"] == 1}
    d1, l1 = it1(D), it1(Lr)
    both = sorted(set(d1) & set(l1))
    ok = lambda r: bool(r.get("test_correct")) and all(r["test_correct"])
    tab = collections.Counter((ok(d1[t]), ok(l1[t])) for t in both)
    P("## D66 (77,824) vs L66 (122,880): paired on the same tasks", "")
    P(f"Iteration 1, tasks reached by both: {len(both)}. Fully correct: D66 {sum(ok(d1[t]) for t in both)}, "
      f"L66 {sum(ok(l1[t]) for t in both)}. Only D66 {tab[(True, False)]}, only L66 {tab[(False, True)]}, both {tab[(True, True)]}.")
    for nm, g in (("D66", [d1[t] for t in both]), ("L66", [l1[t] for t in both])):
        P(f"- {nm} iteration 1: mean thinking {st.mean(r['p1_tokens'] for r in g):,.0f}; forced at cap "
          f"{sum(r.get('p1_finish') == 'length' for r in g)}/{len(g)}; mean slot-minutes "
          f"{st.mean(r['sec'] for r in g) / 60:.1f}")
    # L66 natural closes between 77,824 and 122,880: they would have been cut in D66
    mid = [l1[t] for t in both if l1[t]["finish"] == "natural" and 77824 < l1[t]["p1_tokens"] <= 122880]
    P(f"- L66 iteration-1 natural closes between 77,824 and 122,880 tokens (would be cut at D66's cap): {len(mid)}; "
      f"of them fully correct: {sum(ok(r) for r in mid)}; D66 got the same task right at iteration 1: "
      f"{sum(ok(d1[r['task_id']]) for r in mid)}")
    for run in RUNS:
        s = SUMMARY[run["name"]]
        P(f"- {run['name']} whole run: {s['deployable_inputs']}/{N_IN} deployable, {s['oracle_inputs']} oracle, "
          f"{s['slot_hours']} slot-hours, {s['tokens_M']} M tokens -> "
          f"{s['deployable_inputs'] / max(1e-9, s['tokens_M']):.2f} deployable inputs per M tokens")
    SUMMARY["paired_iteration1"] = {"tasks": len(both), "table": {str(k): v for k, v in tab.items()}, "l66_mid_natural": len(mid),
                                    "l66_mid_correct": sum(ok(r) for r in mid)}

open(os.path.join(a.out, "d66_l66_report.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
json.dump(SUMMARY, open(os.path.join(a.out, "d66_l66_summary.json"), "w", encoding="utf-8"), indent=1, default=str)
print("\n".join(L))
