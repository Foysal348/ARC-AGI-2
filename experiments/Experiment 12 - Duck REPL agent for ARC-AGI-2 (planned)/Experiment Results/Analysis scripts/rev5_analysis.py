"""E12 rev 5 run analysis (EVAL6R5 xhigh, EVAL12R5_LOW). Usage: python -I rev5_analysis.py <q27x dir> [<q27x dir> ...]"""
import glob, json, os, re, sys, statistics as st
from collections import defaultdict, Counter
sys.stdout.reconfigure(encoding="utf-8")
HDR = re.compile(r"^\[([A-Z][A-Za-z _:]*)\]\s*$")
def rep_lines(text):
    cur, buf = None, []
    for line in text.splitlines() + ["[END]"]:
        m = HDR.match(line)
        if m:
            if cur: yield cur, buf
            cur, buf = m.group(1), []
        else:
            buf.append(line)
for d in sys.argv[1:]:
    rep = json.load(open(os.path.join(d, "e12_run_report.json"), encoding="utf-8"))
    print("=" * 100); print(d.split("Kaggle Runs")[-1]); print("profile", rep["profile"], "| run", {k: rep["e12_run"][k] for k in ("started_sec", "finished_sec", "branches_done", "branches_accepted")})
    by = defaultdict(list)
    for u in rep["eval_unit_records"]:
        by[u["task"]].append(u)
    full = [t for t, us in by.items() if all(u["hit"] for u in us)]
    print(f"inputs {rep['eval_query_hits']}/{rep['eval_query_total']} (pass1 {rep['eval_pass1_hits']}, pass2 {rep['eval_pass2_hits']}) | fully solved {len(full)}/{len(by)} | task-norm {rep['eval_task_norm_pct']}")
    tot_gen = 0; allpe = []
    print(f"{'task':9} br|acc|sub|kept/ver|turns|req|gen tok |max cmp|fin len|sec  |cut|trim|tmo|loop%| hit")
    for t in rep["order"]:
        for b in (0, 1):
            p = os.path.join(d, "duck", f"{t}_b{b}")
            if not os.path.exists(os.path.join(p, "branch_result.json")):
                continue
            br = json.load(open(os.path.join(p, "branch_result.json"), encoding="utf-8"))
            us = br.get("usage") or []; comp = [x["completion"] for x in us]; tot_gen += br["generated_tokens"]
            ev = br.get("rev5_events") or {}
            allpe += ev.get("prompt_estimates") or []
            tx = open(os.path.join(p, "transcript.txt"), encoding="utf-8", errors="replace").read()
            fin = Counter(re.findall(r"finish_reason: (\S+)", tx))
            trims = sum(1 for i in range(1, len(us)) if us[i]["prompt"] < us[i - 1]["prompt"])
            tmo = tx.count("request_timeout (yielding")
            tot = rep_ = 0
            for name, buf in rep_lines(tx):
                if name == "THINKING":
                    L = [l.strip() for l in buf if len(l.strip()) >= 40]
                    c = Counter(L); tot += sum(len(l) for l in L); rep_ += sum(len(l) * (k - 1) for l, k in c.items() if k > 1)
            ck = br.get("checked") or []
            hits = [u["hit"] for u in by[t]]
            print(f"{t:9} {b} | {'Y' if br['accepted'] else '-'} | {len(br.get('submissions') or [])} | {sum(1 for v in ck if v['accepted'])}/{len(ck)} | {br['turns']:4} |{len(us):3} | {br['generated_tokens']:7} |{max(comp) if comp else 0:7}|{fin.get('length',0):5}  |{br['t1']-br['t0']:5.0f}|{ev.get('cut_tool_calls_dropped',0):3}|{trims:4}|{tmo:3}|{100*rep_/max(1,tot):4.0f} | {sum(hits)}/{len(hits)}")
    print("generated total", tot_gen, "| per branch mean", round(tot_gen / max(1, rep["e12_run"]["branches_done"])))
    if allpe:
        err = [(e - a) / a for e, a in allpe if a > 0]
        over = [(e, a) for e, a in allpe if e > a * 1.05]
        print(f"prompt estimate vs actual: n={len(err)} median {100*st.median(err):+.1f}% p10 {100*sorted(err)[len(err)//10]:+.1f}% p90 {100*sorted(err)[9*len(err)//10]:+.1f}% | over by >5%: {len(over)} | max actual {max(a for _, a in allpe)}")
    ds = rep.get("e11", {}).get("decode_summary") or {}
    print("decode running hist:", ds.get("running_hist"), "| accept len", ds.get("accept_len_median"))
