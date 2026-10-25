"""E12 rev 4 offline simulator: replays the REAL E12 SGLang EVAL30 branches (every /v1/chat/completions request: prompt,
completion, cached tokens) under candidate settings, with the throughput measured in that run.

Measured inputs (E12 EVAL30 run 1, hosen42, 2026-10-19):
  prefill P = 1,459 tok/s (median TP0 'input throughput' of chunks >= 2000 tokens); decode D = 431 tok/s aggregate at 12
  running (decode summary median); KV 866,744 tokens; 30 branch-0 runs with outcomes (accepted correct / accepted wrong /
  unsolved), branch 1 never ran.
Model (stated, simple): GPU time of a branch = its prefill tokens / P + its generated tokens / (D / 12) shared by 12 streams,
i.e. total time = sum(prefill)/P + sum(gen)/D. 'No thrash' = every later request prefills only its new tokens
(prompt_i - prompt_{i-1} - completion_{i-1}, >= 0); 'measured' = the uncached tokens the run really prefilled.
A per-branch cap C truncates a branch's requests once its generated tokens reach C (a branch that solved above C is lost).
Usage: python -I rev4_simulator.py <q27x dir> <eval30 report json>
"""
import json, glob, os, sys

P, D, KV = 1459.0, 431.0, 866744
d = sys.argv[1]
rep = json.load(open(sys.argv[2], encoding="utf-8"))
hits = {(u["task"]) for u in rep["eval_unit_records"] if u["hit"]}
BR = []
for f in sorted(glob.glob(os.path.join(d, "duck", "*_b0", "branch_result.json"))):
    b = json.load(open(f, encoding="utf-8"))
    reqs = [(x["prompt"], x["completion"], x["cached"] or 0) for x in b["usage"]]
    if not reqs:
        continue
    outcome = "correct" if (b["accepted"] and b["task_id"] in hits) else ("wrong" if b["accepted"] else "unsolved")
    BR.append({"task": b["task_id"], "reqs": reqs, "outcome": outcome, "gen": sum(r[1] for r in reqs)})


def branch_cost(reqs, cap, thrash):
    gen = pre = 0
    prev = None
    for p, c, cached in reqs:
        if cap is not None and gen >= cap:
            break
        new = (p - cached) if thrash else (p if prev is None else max(0, p - prev[0] - prev[1]))
        pre += new
        use = c if cap is None else min(c, cap - gen)
        gen += use
        prev = (p, c)
    return pre, gen


print(f"branches {len(BR)}: correct {sum(b['outcome']=='correct' for b in BR)}, wrong {sum(b['outcome']=='wrong' for b in BR)}, "
      f"unsolved {sum(b['outcome']=='unsolved' for b in BR)}")
print("generated tokens at acceptance: correct", sorted(b["gen"] for b in BR if b["outcome"] == "correct"),
      "| wrong", sorted(b["gen"] for b in BR if b["outcome"] == "wrong"))
pre_m = sum(branch_cost(b["reqs"], None, True)[0] for b in BR); gen_all = sum(b["gen"] for b in BR)
pre_n = sum(branch_cost(b["reqs"], None, False)[0] for b in BR)
print(f"\nmeasured run : prefill {pre_m/1e6:.2f} M + gen {gen_all/1e6:.2f} M -> model time {pre_m/P + gen_all/D:,.0f} s "
      f"(the real solve window was 12,634 s: model check)")
print(f"no thrash    : prefill {pre_n/1e6:.2f} M + gen {gen_all/1e6:.2f} M -> {pre_n/P + gen_all/D:,.0f} s")

HIDDEN_SEC = 11.5 * 3600 - 20 * 60          # 12 h minus about 50 min (load, drain, margins)
print(f"\nHidden projection: 240 tasks x 2 branches in {HIDDEN_SEC:,.0f} s, no thrash, branch 1 assumed = branch 0 cost")
print(f"{'cap C':>8} | {'correct kept':>12} | {'wrong kept':>10} | {'s / branch':>10} | {'branches fit':>12} | {'tasks covered (2 br)':>20}")
for cap in (None, 100000, 80000, 70000, 60000, 50000, 40000, 30000):
    cost = [branch_cost(b["reqs"], cap, False) for b in BR]
    sec = sum(p / P + g / D for p, g in cost) / len(BR)
    keep_c = sum(1 for b in BR if b["outcome"] == "correct" and (cap is None or b["gen"] <= cap))
    keep_w = sum(1 for b in BR if b["outcome"] == "wrong" and (cap is None or b["gen"] <= cap))
    fit = HIDDEN_SEC / sec
    print(f"{str(cap):>8} | {keep_c:>12} | {keep_w:>10} | {sec:>10,.0f} | {fit:>12,.0f} | {min(240, fit / 2):>20,.0f}")
print("\nResident-context check (KV 866,744): prompts p50 47.9k / p90 97.7k -> about 9 contexts at p90, 18 at p50."
      " Live branches above that evict prefixes (the EVAL30 thrash: 30 live b0 contexts).")
