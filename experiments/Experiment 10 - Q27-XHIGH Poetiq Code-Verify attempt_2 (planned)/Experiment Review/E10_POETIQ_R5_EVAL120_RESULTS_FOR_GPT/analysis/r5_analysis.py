"""Deep analysis of the Poetiq R5 EVAL120 run vs DIRECT ORDER2 EVAL120 (Claude, 2026-10-15)."""
import json, re, csv, sys, statistics as st
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

EXP = Path("D:/ARC AGI 2/experiments"); E10 = EXP / "Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)"
R5 = E10 / "Experiment Results/Kaggle Runs/GPU EVAL120 Poetiq R5 (hosen42 q27-xhigh-e10-poetiq-r5-eval120)"
O2 = E10 / "Experiment Results/Kaggle Runs/GPU EVAL120 ORDER2 86k DIRECT (hosen42 q27-xhigh-e10-order2)"
LED = Path("D:/ARC AGI 2/resources/Puzzle Capability Ledger")
FX = EXP / "Experiment 1 - Q27-XHIGH Direct Reasoning (LB 18.75 and 26.94)/Experiment Notebooks/Build/Q27X_build/fixtures/fake_input"
tok = Tokenizer.from_file(str(FX / "models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1/tokenizer.json"))
ch = json.load(open(FX / "competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"))
sol = json.load(open(FX / "competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json"))
rep = json.load(open(R5 / "q27x/poetiq_run_report.json"))
its = [json.loads(l) for l in open(R5 / "q27x/poetiq_iterations.jsonl", encoding="utf-8")]
o2 = {x["unit"]: x for x in map(json.loads, open(O2 / "q27x/records.jsonl", encoding="utf-8"))}
orc = {r["task"]: r for r in csv.DictReader(open(LED / "oracle_order_120.csv", encoding="utf-8"))}
a120 = {r["task"]: r for r in csv.DictReader(open(LED / "ledger_all120.csv", encoding="utf-8"))}
P = {f"{r['task']}#{r['query']}": r for r in rep["eval_unit_records"]}
hitP = lambda u: bool(P[u]["hit"]); hitD = lambda u: bool(o2[u]["score"]["hit"])
U = sorted(P); T = sorted(ch); nq = {t: len(ch[t]["test"]) for t in T}
tn = lambda f: round(100 * sum(sum(f(f"{t}#{q}") for q in range(nq[t])) / nq[t] for t in T) / 120, 2)

print("== 1. headline")
print(f"Poetiq: {sum(map(hitP, U))}/172 = {100*sum(map(hitP, U))/172:.2f} % | task-norm {tn(hitP)} % | pass1 {rep['eval_pass1_hits']} pass2 {rep['eval_pass2_hits']}"
      f" | attempt_2-only hits {sum(1 for u in U if P[u]['hit'] and not P[u]['pass1'])}")
print(f"DIRECT ORDER2: {sum(map(hitD, U))}/172 | task-norm {tn(hitD)} %")
both = [u for u in U if hitP(u) and hitD(u)]; po = [u for u in U if hitP(u) and not hitD(u)]; do = [u for u in U if hitD(u) and not hitP(u)]
print(f"both {len(both)} | Poetiq only {len(po)} | DIRECT only {len(do)} | union {len(both)+len(po)+len(do)}/172 = {100*(len(both)+len(po)+len(do))/172:.1f} %"
      f" | union task-norm {tn(lambda u: hitP(u) or hitD(u))} %")
print("DIRECT only:", do)
sch = rep["scheduler"]; print(f"wall {sch['wall_sec']/60:.0f} min, tokens {sch['tokens']/1e6:.2f} M, {sch['tok_per_sec']} tok/s | ORDER2: 534 min, 10.45 M, 333 tok/s")

print("\n== 2. iterations")
by_t = defaultdict(list)
for x in its: by_t[x["task_id"]].append(x)
print("iterations per task:", dict(sorted(Counter(len(v) for v in by_t.values()).items())))
print("status:", dict(Counter(s["status"] for s in rep["expert_states"].values())))
for k in sorted({x["iteration"] for x in its}):
    s = [x for x in its if x["iteration"] == k]
    print(f"  iter {k}: {len(s):3d} requests | forced {sum(x['mode']=='forced_close' for x in s):3d} | train passed {sum(x['train_passed'] for x in s):2d}"
          f" | p1 tokens median {int(st.median(x['p1_tokens'] for x in s))} | min/req median {st.median(x['sec'] for x in s)/60:.1f}")
ver_it = {x["task_id"]: x["iteration"] for x in its if x["train_passed"]}

print("\n== 3. train-verified -> test correct (precision of the sandbox check)")
def task_hits(t): return sum(hitP(f"{t}#{q}") for q in range(nq[t]))
V = sorted(ver_it); full = [t for t in V if task_hits(t) == nq[t]]; part = [t for t in V if 0 < task_hits(t) < nq[t]]; zero = [t for t in V if task_hits(t) == 0]
print(f"train-verified tasks {len(V)}: fully right on test {len(full)}, partly {len(part)}, wrong {len(zero)}")
for lab, sel in (("n_train<=2", lambda t: len(ch[t]["train"]) <= 2), ("n_train>=3", lambda t: len(ch[t]["train"]) >= 3)):
    vv = [t for t in V if sel(t)]
    print(f"  {lab}: verified {len(vv)}, fully right {sum(task_hits(t)==nq[t] for t in vv)}, wrong {sum(task_hits(t)==0 for t in vv)}")
print("  verified but WRONG on test:", zero)
nv = [t for t in T if t not in ver_it]
print(f"not verified tasks {len(nv)}: units hit anyway {sum(task_hits(t) for t in nv)} ({[t for t in nv if task_hits(t)]})")
print("solves by verifying iteration:", dict(Counter(ver_it[t] for t in V if task_hits(t))), "| units:", dict(Counter(ver_it[t] for t in V for q in range(nq[t]) if hitP(f'{t}#{q}'))))

print("\n== 4. thinking tokens of solving iterations (from saved thinking files)")
def think_tokens(t, k):
    f = R5 / f"q27x/poetiq_thinking/{t}_e0_i{k}.txt"
    s = f.read_text(encoding="utf-8").split("\n", 1)[1]
    s = s.split("</think>", 1)[0]
    return len(tok.encode(s, add_special_tokens=False).ids)
solve_think = sorted((think_tokens(t, ver_it[t]), t, ver_it[t]) for t in V if task_hits(t))
th = [x[0] for x in solve_think]
print(f"solved tasks via verified program: {len(th)} | think p25/median/p75/max: {int(np.percentile(th,25))}/{int(np.median(th))}/{int(np.percentile(th,75))}/{max(th)}")
for X in (49152, 60000, 70000):
    print(f"  solves with thinking > {X}: {sum(v > X for v in th)}")
print("  later-iteration solves:", [(t, k, v) for v, t, k in solve_think if k > 1])
it1 = [x for x in its if x["iteration"] == 1]
print("iter-1 forced:", sum(x['mode']=='forced_close' for x in it1), "of", len(it1), "| forced iter-1 that passed train:", sum(x['train_passed'] for x in it1 if x['mode']=='forced_close'))

print("\n== 5. tiers (oracle, built before this run) and new solves")
for k in "ABCD":
    ts = [t for t in T if orc[t]["tier"] == k]
    sP = sum(task_hits(t) / nq[t] for t in ts); sD = sum(sum(hitD(f"{t}#{q}") for q in range(nq[t])) / nq[t] for t in ts)
    print(f"  tier {k}: {len(ts):3d} tasks | Poetiq {sP:5.2f} | DIRECT {sD:5.2f}")
newD = [t for t in T if orc[t]["tier"] == "D" and task_hits(t)]
print("  tier-D (never solved by any earlier method) now solved by Poetiq:", newD)
print("  ARC-Prize-only tasks solved by Poetiq:", [t for t in T if a120[t]["arcprize_xhigh"] == "1" and orc[t]["our_runs_with_hit"] == "0" and task_hits(t)])

print("\n== 6. multi-test / low_train / order_score")
for lab, sel in (("single test", lambda t: nq[t] == 1), ("multi test", lambda t: nq[t] > 1), ("n_train<=2", lambda t: len(ch[t]["train"]) <= 2),
                 ("n_train>=3", lambda t: len(ch[t]["train"]) >= 3)):
    us = [u for u in U if sel(u.split("#")[0])]
    print(f"  {lab:11s}: units {len(us):3d} | Poetiq {sum(map(hitP, us)):2d} ({100*sum(map(hitP, us))/len(us):.0f} %) | DIRECT {sum(map(hitD, us)):2d} ({100*sum(map(hitD, us))/len(us):.0f} %)")
mt = [t for t in T if nq[t] > 1]
print(f"  multi-test tasks: Poetiq full {sum(task_hits(t)==nq[t] for t in mt)}, partial {sum(0<task_hits(t)<nq[t] for t in mt)} | DIRECT full 5, partial 15")
json.dump({"poetiq_only": po, "direct_only": do, "both": both, "verified_wrong": zero}, open(Path(sys.argv[1]) / "r5_sets.json", "w"))
