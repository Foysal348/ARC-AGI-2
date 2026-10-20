"""Synthetic D66/L66 q27x outputs (same record format as the D66/L66 notebooks) for testing analyze_d66_l66.py.
Known answers: every value it should report can be recomputed from the fixture. usage: python -I make_fake_runs.py OUTDIR"""
import json, os, random, sys

OUT = sys.argv[1]
E = r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)"
SOL = json.load(open(E + "/Experiment Review/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_solutions.json"))
R5 = E + "/Experiment Results/Kaggle Runs/GPU EVAL120 Poetiq R5 (hosen42 q27-xhigh-e10-poetiq-r5-eval120)/q27x/"
fin = json.load(open(R5 + "poetiq_final_generated.json"))
HARD = sorted(t for t, rows in fin.items() if not all(SOL[t][q] in (r["attempt_1"], r["attempt_2"]) for q, r in enumerate(rows)))
assert len(HARD) == 66
for name, cap, p_ok, seed in (("D66", 77824, 0.15, 1), ("L66", 122880, 0.25, 2)):
    rnd = random.Random(seed); d = os.path.join(OUT, name); os.makedirs(d, exist_ok=True)
    it, curve, t = [], [], 0.0
    for tid in HARD:
        nq = len(SOL[tid]); hits = [False] * nq
        for k in range(1, rnd.choice([1, 2, 3]) + 1):
            p1 = rnd.choice([cap, cap, int(cap * 0.7), int(cap * 0.95)])
            pf = "length" if p1 == cap else "stop"
            correct = [rnd.random() < p_ok for _ in range(nq)]
            grids = [SOL[tid][q] if correct[q] else [[0]] for q in range(nq)]
            passed = all(correct) or rnd.random() < 0.05
            t += rnd.uniform(200, 900)
            it.append({"task_id": tid, "expert": 0, "iteration": k, "mode": "forced_close" if pf == "length" else "natural",
                       "finish": "forced_close" if pf == "length" else "natural", "p1_tokens": p1, "p2_tokens": 1500 if pf == "length" else 0,
                       "prompt_tokens": 5000, "p1_cap": cap, "sec": rnd.uniform(1800, 5400), "has_python": True, "score": 0.9,
                       "train_passed": passed, "test_usable": True, "test_grids": grids, "test_errors": [None] * nq,
                       "train_soft_scores": [1.0, 0.8], "test_correct": correct, "t_done_sec": round(t, 1),
                       "t_admit": round(t - 1000, 1), "p1_finish": pf})
            hits = [h or c for h, c in zip(hits, correct)]
            curve.append({"task_id": tid, "iterations_done": k, "t": round(t, 1), "pass1": hits, "hit": hits})
            if passed:
                break
    with open(os.path.join(d, "poetiq_iterations.jsonl"), "w") as f:
        for r in it: f.write(json.dumps(r) + "\n")
    with open(os.path.join(d, "poetiq_attempt_curve.jsonl"), "w") as f:
        for c in curve: f.write(json.dumps(c) + "\n")
    json.dump({"eval_query_hits": None}, open(os.path.join(d, "poetiq_run_report.json"), "w"))
    oracle = {(r["task_id"], q) for r in it for q, c in enumerate(r["test_correct"]) if c}
    last = {}
    for c in curve: last[c["task_id"]] = c
    exp = {"requests": len(it), "oracle_inputs": len(oracle), "deployable_inputs": sum(sum(c["hit"]) for c in last.values())}
    json.dump(exp, open(os.path.join(d, "EXPECTED.json"), "w"))
    print(name, exp)
