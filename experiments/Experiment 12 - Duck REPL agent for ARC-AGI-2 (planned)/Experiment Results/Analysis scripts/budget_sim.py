"""What if each branch (attempt) had a token budget C? Uses the recorded rev 5 runs: every host-verified candidate
(submission or harvested check) carries the branch's generated tokens when it was found. Duck's cap is soft (checked
between requests), so a candidate counts if the branch had generated < C before the request that produced it.
Selection = the notebook's own select_attempts_r5 on the surviving candidates; scored with the truth.
Tokens a branch would use = completions up to and including the first request that starts at >= C... (soft cap:
the request that crosses C still finishes). Time is scaled from the measured solve window by the token ratio.
Usage: python -I budget_sim.py <adapter dir> <solutions.json> <q27x dir> [...]"""
import json, os, sys, glob
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, sys.argv[1]); os.environ["E12_ARC2_REV5"] = "1"
import arc2_duck as A
sol = json.load(open(sys.argv[2]))
CAPS = [15000, 25000, 35000, 50000, 70000, None]
for d in sys.argv[3:]:
    rep = json.load(open(os.path.join(d, "e12_run_report.json"), encoding="utf-8"))
    ch = None
    tasks = rep["order"]
    solve_sec = rep["e12_run"]["finished_sec"] - rep["e12_run"]["started_sec"]
    gen_total = rep["usage_all"]["completion_tokens"]
    print("=" * 110); print(d.split("Kaggle Runs")[-1], "| tasks", len(tasks), "| solve", round(solve_sec / 60), "min | generated", gen_total)
    print(f"{'cap/branch':>11} | {'inputs':>7} | {'tasks':>6} | {'tok/task':>8} | {'min/task':>8} | {'240-task need (h)':>17} | lost vs uncapped / gained")
    base = None
    for C in CAPS:
        hits_in = full = 0; tok = 0; per_task = {}
        for t in tasks:
            recs = []
            for b in (0, 1):
                p = os.path.join(d, "duck", f"{t}_b{b}", "branch_result.json")
                if not os.path.exists(p):
                    continue
                br = json.load(open(p, encoding="utf-8"))
                comp = [u["completion"] for u in br.get("usage") or []]
                cum, before = [], []
                s = 0
                for c in comp:
                    before.append(s); s += c; cum.append(s)
                # tokens this branch would use under C (soft cap)
                if C is None:
                    used = s
                else:
                    used = 0
                    for bf, c in zip(before, comp):
                        if bf >= C:
                            break
                        used += c
                # also stop where the real branch stopped (accepted earlier)
                tok += used
                def keep(v):
                    if C is None:
                        return True
                    gt = v.get("tokens") or 0          # generated tokens at verification (after that request)
                    # gen before that request = largest cum < gt ... find the request index
                    idx = next((i for i, x in enumerate(cum) if x >= gt), len(cum) - 1)
                    return before[idx] < C if cum else True
                recs.append({"branch": b, "submissions": [v for v in br.get("submissions") or [] if keep(v)],
                             "checked": [v for v in br.get("checked") or [] if keep(v)]})
            att = A.select_attempts_r5({"test": [{"input": [[0]]}] * len(sol[t])}, recs)
            h = [a["attempt_1"] == g or a["attempt_2"] == g for a, g in zip(att, sol[t])]
            hits_in += sum(h); full += all(h); per_task[t] = all(h)
        n_in = sum(len(sol[t]) for t in tasks)
        min_task = solve_sec / 60 * (tok / gen_total) / len(tasks)
        need_h = 240 * min_task / 60
        if C is None:
            base = per_task
        lost = gained = ""
        print(f"{str(C):>11} | {hits_in:>3}/{n_in:<3} | {full:>2}/{len(tasks):<3} | {tok/len(tasks):>8.0f} | {min_task:>8.1f} | {need_h:>17.1f} | ", end="")
        print(per_task)
    if base:
        pass
