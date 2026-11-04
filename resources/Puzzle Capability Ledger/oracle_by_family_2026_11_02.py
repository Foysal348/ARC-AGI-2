"""Oracle solves by method family, updated 2026-11-02 (user: per architecture how many solved: Direct, Poetiq, the new ones,
and the oracle of all).

= oracle_by_family.py (every run to 2026-10-25, imported) + every eval run since then:
  - 27B: E15P Poetiq EVAL120, the E18 / E20 S1-60K Poetiq commits, the E21 Duck config-B halves on vLLM;
  - Qwen3.8-Flash-Next (a different model: klee100 3bpw / W4 on Merlin): the E25 NB5 / NB6 Duck runs, E26 K100 quality,
    E28 W4 + E28 K100 EVAL120 (Duck), E31 Direct (30 tasks), the E29 hybrid GPU smoke (Duck + DIRECT, 12 tasks).
A query counts as solved by a run when attempt_1 or attempt_2 of that run's generated attempts equals the gold output
(e12_generated_attempts.json) or the run's own unit record says hit (Poetiq reports). Attempted = the run's own unit
records when present, else the queries of the tasks the run planned (generated attempts carry identity rows for the rest).
usage: python oracle_by_family_2026_11_02.py  -> oracle_by_family_2026_11_02.json + oracle_units_by_family_2026_11_02.csv
"""
import csv
import io
import json
import contextlib
from pathlib import Path

with contextlib.redirect_stdout(io.StringIO()):
    import oracle_by_family as OF          # recomputes the 2026-10-25 families (its own files, unchanged rules)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SCR = ROOT / "_scratch"
EXP = ROOT / "experiments"
CH, UNITS, BL = OF.CH, OF.UNITS, OF.BL
GOLD = json.load(open(BL.FX / "arc-agi_evaluation_solutions.json"))

F27D, F27P, F27K = "27B Direct reasoning", "27B Poetiq / code + train verify", "27B Duck REPL agent"
FND, FNX, FNH = "Flash-Next Duck (klee100 / W4)", "Flash-Next Direct (E31)", "Flash-Next hybrid Duck + DIRECT (E29 smoke)"
REMAP = {"Direct reasoning": F27D, "Poetiq / code + train verify": F27P, "Duck REPL agent (ours)": F27K,
         "Other code / agent (R6 loop, REPL agent v1.8-1.9)": "27B other code / agent (R6, REPL v1.8-1.9)",
         "ARC Prize verified run (others, task level)": "ARC Prize verified run (27B, others, task level)"}


def from_generated(gen_path, report_path=None, planned=None):
    g = json.load(open(gen_path, encoding="utf-8"))
    rep = json.load(open(report_path, encoding="utf-8")) if report_path and Path(report_path).is_file() else {}
    recs = rep.get("eval_unit_records") or []
    att = {(r["task"], int(r["query"])) for r in recs} if recs else None
    out = {}
    for t, rows in g.items():
        if t not in GOLD:
            continue
        for q, d in enumerate(rows):
            k = (t, q)
            if att is not None and k not in att:
                continue
            if att is None and planned is not None and t not in planned:
                continue
            gd = GOLD[t][q] if q < len(GOLD[t]) else None
            out[k] = int(gd is not None and (d.get("attempt_1") == gd or d.get("attempt_2") == gd))
    return out


def from_poetiq(path):
    return {k: v["hit"] for k, v in BL.load_run("poetiq_report", path, CH).items()}


def planned_of(e12_report):
    p = Path(e12_report)
    if not p.is_file():
        return None
    r = json.load(open(p, encoding="utf-8"))
    for key in ("task_ids", "tasks", "order"):
        v = r.get(key)
        if isinstance(v, list) and v and isinstance(v[0], str):
            return set(v)
    return None


E15K = EXP / "Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Results/Kaggle Runs"
NEW = [  # (name, family, kind, path, report / planned helper)
    ("E15P Poetiq EVAL120 S60 rev2 (27B, 2026-10-25)", F27P, "poetiq",
     E15K / "E15P EVAL120 S60 S15 rev2 commit1 (esfersami50, 2026-10-25, 56a22f7f)/q27x/poetiq_run_report.json", None),
    ("E18 S1-60K vLLM 0.32 commit v1 (27B Poetiq)", F27P, "poetiq", SCR / "e18_gpu_v1/q27x/poetiq_run_report.json", None),
    ("E18 S1-60K vLLM 0.32 commit v2 (27B Poetiq)", F27P, "poetiq", SCR / "e18_gpu_v2/q27x/poetiq_run_report.json", None),
    ("E18 S1-60K vLLM 0.32 commit v3 (27B Poetiq)", F27P, "poetiq", SCR / "e18_gpu_v3/q27x/poetiq_run_report.json", None),
    ("E20 S1 drain commit (27B Poetiq)", F27P, "poetiq", SCR / "e20_out/q27x/poetiq_run_report.json", None),
    ("E21 Duck B vLLM NB1 INT4 s20 half A (27B)", F27K, "gen",
     SCR / "e21_sami_logs/out_q27-xhigh-e21-duckb-vllm-nb1-int4-s20/q27x/halfA/e12_generated_attempts.json",
     SCR / "e21_sami_logs/out_q27-xhigh-e21-duckb-vllm-nb1-int4-s20/q27x/halfA/e12_run_report.json"),
    ("E21 Duck B vLLM NB1 INT4 s20 half B (27B)", F27K, "gen",
     SCR / "e21_sami_logs/out_q27-xhigh-e21-duckb-vllm-nb1-int4-s20/q27x/halfB/e12_generated_attempts.json",
     SCR / "e21_sami_logs/out_q27-xhigh-e21-duckb-vllm-nb1-int4-s20/q27x/halfB/e12_run_report.json"),
    ("E25 NB5 quality BF16 K3 (Flash-Next)", FND, "gen", SCR / "e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x/phase_Q_BF16_K3/e12_generated_attempts.json",
     SCR / "e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x/phase_Q_BF16_K3/e12_run_report.json"),
    ("E25 NB5 quality FP8 K3 (Flash-Next)", FND, "gen", SCR / "e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x/phase_Q_FP8_K3/e12_generated_attempts.json",
     SCR / "e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x/phase_Q_FP8_K3/e12_run_report.json"),
    ("E25 NB6 SUBMIT commit (Flash-Next)", FND, "gen", SCR / "e25_gpu_out/q27-xhigh-e25-nb6-submit/q27x/phase_EVAL/e12_generated_attempts.json",
     SCR / "e25_gpu_out/q27-xhigh-e25-nb6-submit/q27x/phase_EVAL/e12_run_report.json"),
    ("E26 K100 quality FP8 K3 (Flash-Next klee100)", FND, "gen",
     SCR / "e26_gpu_out/q27-xhigh-e26-k100-quality/q27x/phase_Q_K100_FP8_K3/e12_generated_attempts.json",
     SCR / "e26_gpu_out/q27-xhigh-e26-k100-quality/q27x/phase_Q_K100_FP8_K3/e12_run_report.json"),
    ("E28 W4 EVAL120 6h (Flash-Next W4)", FND, "gen", SCR / "e28_w4_out/q27-xhigh-e28-w4-eval120-6h/q27x/phase_EVAL/e12_generated_attempts.json",
     SCR / "e28_w4_out/q27-xhigh-e28-w4-eval120-6h/q27x/phase_EVAL/e12_run_report.json"),
    ("E28 K100 EVAL120 6h (Flash-Next klee100)", FND, "gen", SCR / "e28_k100_out/q27-xhigh-e28-k100-eval120-6h/_bundle/e12_generated_attempts.json",
     None),
    ("E31 Flash-Next Direct 30 tasks", FNX, "gen", SCR / "e31_hamid_out/q27x/phase_Q_E31_FN_DIRECT30/e12_generated_attempts.json",
     SCR / "e31_hamid_out/q27x/phase_Q_E31_FN_DIRECT30/e12_run_report.json"),
    ("E29 hybrid GPU smoke 12 tasks (Duck + DIRECT, rev 2e)", FNH, "gen",
     SCR / "e29_smoke_out/q27x/phase_Q_E28_K100_EVAL120/e12_generated_attempts.json",
     SCR / "e29_smoke_out/q27x/phase_Q_E28_K100_EVAL120/e12_run_report.json"),
]

runs = [(n, REMAP.get(f, f), r) for n, f, r in OF.runs]
missing = []
for name, fam, kind, path, rp in NEW:
    if not Path(path).is_file():
        missing.append(name)
        continue
    r = from_poetiq(path) if kind == "poetiq" else from_generated(path, rp, planned_of(rp) if rp else None)
    if r:
        runs.append((name, fam, r))
    else:
        missing.append(name + " (no eval units)")

fams = {}
for n, f, r in runs:
    fams.setdefault(f, []).append((n, r))
groups = dict(sorted(fams.items()))
ours = [x for f, v in fams.items() if not f.startswith("ARC Prize") for x in v]
groups["ALL 27B RUNS OF OURS"] = [x for f, v in fams.items() if f.startswith("27B") for x in v]
groups["ALL FLASH-NEXT RUNS"] = [x for f, v in fams.items() if f.startswith("Flash-Next") for x in v]
groups["ALL OUR RUNS (both models, every method)"] = ours
groups["TOTAL ORACLE (ours + ARC Prize)"] = [x for v in fams.values() for x in v]

out, S = {}, {}
print("| group | runs | queries attempted (of 172) | tasks touched | oracle queries | oracle task-norm | % of 120 | best single run |")
print("|---|---|---|---|---|---|---|---|")
for g, rl in groups.items():
    att, sol = OF.oracle(rl)
    S[g] = sol
    tn = OF.tasknorm(sol)
    best = max(rl, key=lambda x: (sum(x[1].values()), -len(x[1])))
    out[g] = {"runs": [n for n, _ in rl], "attempted_units": len(att), "solved_units": len(sol), "task_norm": round(tn, 2),
              "tasks_touched": len({t for t, _ in att}), "best_run": best[0], "best_run_hits": sum(best[1].values()),
              "best_run_attempted": len(best[1])}
    print(f"| {g} | {len(rl)} | {len(att)} | {len({t for t, _ in att})} | {len(sol)} | {tn:.2f} | {100 * tn / 120:.1f} % | "
          f"{best[0]}: {sum(best[1].values())}/{len(best[1])} |")
print("\nnew runs added:")
for name, fam, r in [(n, f, r) for n, f, r in runs if any(n == x[0] for x in NEW)]:
    print(f"  [{fam}] {name}: {sum(r.values())}/{len(r)} queries")
print("missing / skipped:", missing)
A27 = S["ALL 27B RUNS OF OURS"]
AFN = S["ALL FLASH-NEXT RUNS"]
AP = S["ARC Prize verified run (27B, others, task level)"]
ex = {"Flash-Next only (no 27B run of ours)": AFN - A27, "27B only (no Flash-Next run)": A27 - AFN,
      "Flash-Next Direct only (no other run of ours, no ARC Prize)": S[FNX] - OF.oracle([x for f, v in fams.items() if f != FNX for x in v])[1],
      "Flash-Next Direct vs the E28 K100 run only": S[FNX] - {k for n, f, r in runs if n.startswith("E28 K100") for k, h in r.items() if h},
      "unsolved by every run (ours + ARC Prize)": set(UNITS) - S["TOTAL ORACLE (ours + ARC Prize)"]}
print()
for k, s in ex.items():
    print(f"{k}: {len(s)} queries, task-norm {OF.tasknorm(s):.2f} | {sorted(f'{t}#{q}' for t, q in s)[:40]}")
out["exclusive"] = {k: sorted(f"{t}#{q}" for t, q in s) for k, s in ex.items()}
out["missing_runs"] = missing
json.dump(out, open(HERE / "oracle_by_family_2026_11_02.json", "w"), indent=1)
with open(HERE / "oracle_units_by_family_2026_11_02.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["task", "query"] + list(groups))
    for t, q in UNITS:
        w.writerow([t, q] + [int((t, q) in S[g]) for g in groups])
