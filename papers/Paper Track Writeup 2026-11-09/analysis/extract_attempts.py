"""Extract per-attempt records for NB5 QUALITY (W4 FP8 phase) and K100Q (klee100).

Sources (read-only):
  W4 : _scratch/e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x/nb5_report.json  phases.Q_FP8_K3.qab
       _scratch/e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x/phase_Q_FP8_K3/duck/<task>_b<k>/branch_result.json
  K100: _scratch/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x/nb5_report.json  phases.Q_K100_FP8_K3.qab
       _scratch/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x/phase_Q_K100_FP8_K3/duck/<task>_b<k>/branch_result.json
  gold: experiments/Experiment 10 .../E10_ONE_THINKING_TWO_ATTEMPTS_FOR_ASTRA/data/arc-agi_evaluation_{challenges,solutions}.json
Writes attempts.json next to this script.
"""
import json, os, sys

ROOT = r"D:\ARC AGI 2"
DATA = os.path.join(ROOT, r"experiments\Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)\Experiment Review\E10_ONE_THINKING_TWO_ATTEMPTS_FOR_ASTRA\data")
RUNS = {
    "W4": (os.path.join(ROOT, r"_scratch\e25_gpu_out\q27-xhigh-e25-nb5-quality\q27x"), "Q_FP8_K3"),
    "K100": (os.path.join(ROOT, r"_scratch\e26_gpu_out\q27-xhigh-e26-k100-quality\q27x"), "Q_K100_FP8_K3"),
}
HERE = os.path.dirname(os.path.abspath(__file__))

chal = json.load(open(os.path.join(DATA, "arc-agi_evaluation_challenges.json")))
sol = json.load(open(os.path.join(DATA, "arc-agi_evaluation_solutions.json")))


def shape(g):
    if not isinstance(g, list) or not g or not isinstance(g[0], list):
        return None
    return [len(g), len(g[0])]


def colours(g):
    return sorted({v for r in g for v in r})


out = []
for build, (base, phase) in RUNS.items():
    rep = json.load(open(os.path.join(base, "nb5_report.json")))
    ph = rep["phases"][phase]
    groups = {t: g for g, ts in rep["config"]["quality_ab"]["groups"].items() for t in ts}
    order = ph["order"]
    for tid, trec in ph["qab"]["tasks"].items():
        c = chal[tid]
        gold = sol[tid]
        for a in trec["attempts"]:
            k = a["attempt"] - 1
            bdir = os.path.join(base, f"phase_{phase}", "duck", f"{tid}_b{k}")
            br = None
            p = os.path.join(bdir, "branch_result.json")
            if os.path.exists(p):
                br = json.load(open(p))
            sub = None
            if br:
                acc = [s for s in br.get("submissions", []) if s.get("accepted")]
                sub = acc[-1] if acc else None
            test_ok = None
            if sub is not None:
                outs = sub.get("test_outputs") or []
                test_ok = [i < len(outs) and outs[i] == gold[i] for i in range(len(gold))]
            # train-pass seen anywhere (checked list) even if not submitted
            train_pass_seen = False
            first_train_pass_tokens = None
            if br:
                for s in (br.get("checked") or []) + (br.get("submissions") or []):
                    te = s.get("train_exact") or []
                    if te and all(te):
                        train_pass_seen = True
                        tk = s.get("tokens")
                        if tk is not None and (first_train_pass_tokens is None or tk < first_train_pass_tokens):
                            first_train_pass_tokens = tk
            out.append({
                "build": build, "task": tid, "group": groups.get(tid), "queue_pos": order.index(tid) + 1,
                "attempt": a["attempt"], "outcome": a["outcome"], "end_reason": a.get("end_reason"),
                "finished": a.get("finished"), "tokens": a.get("tokens"), "budget_limited": a.get("budget_limited"),
                "accepted": bool(a.get("accepted")), "exact_accepted": a.get("exact_accepted"),
                "exact_any_candidate": a.get("exact_any_candidate"),
                "branch_dir": os.path.relpath(bdir, ROOT) if br else None,
                "sub_name": sub.get("name") if sub else None, "sub_tokens": sub.get("tokens") if sub else None,
                "sub_src_key": sub.get("src_key") if sub else None,
                "test_ok_vs_gold": test_ok,
                "train_pass_seen": train_pass_seen, "first_train_pass_tokens": first_train_pass_tokens,
                "n_train": len(c["train"]), "n_test": len(c["test"]),
                "train_in_shapes": [shape(p["input"]) for p in c["train"]],
                "train_out_shapes": [shape(p["output"]) for p in c["train"]],
                "test_in_shapes": [shape(p["input"]) for p in c["test"]],
                "gold_shapes": [shape(g) for g in gold],
                "pred_shapes": [shape(o) for o in (sub.get("test_outputs") or [])] if sub else None,
            })

json.dump(out, open(os.path.join(HERE, "attempts.json"), "w"), indent=1)
print("attempts", len(out))
