"""Load every Duck attempt (branch_result.json) of our runs with its class, correctness, tokens (2026-10-25)."""
import json, glob
from pathlib import Path
A = Path(r"D:\ARC AGI 2\experiments")
SOL = json.load(open(A / "Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_solutions.json"))
CH = json.load(open(A / "Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_challenges.json"))
K14 = A / "Experiment 14 - Optimized Engine on 4xL4 (planned)/Experiment Results/Kaggle Runs"
K15 = A / "Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Results/Kaggle Runs"
K12 = A / "Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)/Experiment Results/Kaggle Runs"
RUNS = {
    "G4B_XHIGH_E16": (K14 / "EVAL120G4B XHIGH E16 (hosen42, 2026-10-24)", "xhigh"),
    "G4B_XHIGH_noE16": (K14 / "EVAL120G4B XHIGH noE16 (hosen42, 2026-10-24)", "xhigh"),
    "G4B_LOW_E16": (K14 / "EVAL120G4B LOW E16 (foysal, 2026-10-24)", "low"),
    "E15B_45K30K": (K15 / "E15B SUBMIT 45K30K commit1 (esfersami50, 2026-10-25, 5d8069b9)", "xhigh"),
    "E15B_40K35K": (K15 / "E15B SUBMIT 40K35K commit1 v1 (foysal, 2026-10-25, 560ed8b8, sv361956451)", "xhigh"),
    "E12_EVAL12R5_LOW": (K12 / "E12 SGLang EVAL12R5_LOW run1 (foysal, 2026-10-20)", "low"),
}

def cands(r):
    return [v for v in (r.get("checked") or []) + (r.get("submissions") or []) if isinstance(v, dict)]

def qhits(outs, tid):
    g = SOL[tid]
    return [qi < len(outs or []) and outs[qi] == gold for qi, gold in enumerate(g)]

def load():
    rows = []
    for run, (root, eff) in RUNS.items():
        for p in sorted(glob.glob(str(root / "q27x" / "duck" / "*_b*" / "branch_result.json"))):
            r = json.load(open(p, encoding="utf-8"))
            tid, b = r["task_id"], r["branch"]
            if tid not in SOL:
                continue
            acc = [v for v in cands(r) if v.get("accepted")]
            sigs = {json.dumps(v.get("test_outputs")) for v in acc}
            best = ((r.get("progress") or {}).get("best")) or {}
            prog = bool(best) and (best.get("k", 0) > 0 or best.get("acc", 0) > 0)
            cls = 3 if len(sigs) >= 2 else 2 if len(sigs) == 1 else (0 if prog else 1)
            nq = len(SOL[tid])
            # per query: correct among accepted candidates (any of them) / among all candidates incl. partial
            acc_q = [any(qhits(v.get("test_outputs"), tid)[q] for v in acc) for q in range(nq)]
            any_q = [any(qhits(v.get("test_outputs"), tid)[q] for v in cands(r)) for q in range(nq)]
            acc_tok = min([int(v.get("tokens") or 0) for v in acc] or [None]) if acc else None
            rows.append(dict(run=run, eff=eff, task=tid, b=b, cls=cls, n_sigs=len(sigs), prog=prog,
                             flagged=any(v.get("overfit_flags") for v in acc), acc_q=acc_q, any_q=any_q,
                             acc_task=all(acc_q), acc_any=any(acc_q), tokens=int(r.get("generated_tokens") or 0),
                             acc_tok=acc_tok, sigs=sigs, acc_outs=[v.get("test_outputs") for v in acc],
                             view=r.get("view"), nruns=len(r.get("runs") or [])))
    return rows

if __name__ == "__main__":
    import collections
    rows = load()
    c = collections.Counter((r["run"], r["b"]) for r in rows)
    for k in sorted(c): print(k, c[k])
    print(collections.Counter((r["run"], r["view"]) for r in rows))
