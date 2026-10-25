"""E12 rev 4 EVAL6 run 1 analysis (plan in 'E12 rev 4 EVAL6 - GPT audits validated (2026-10-20).md').
Usage: python -I eval6_analysis.py <q27x dir>"""
import glob, json, os, re, sys
from collections import defaultdict
sys.stdout.reconfigure(encoding="utf-8")
d = sys.argv[1]
rep = json.load(open(os.path.join(d, "e12_run_report.json"), encoding="utf-8"))
print("run:", rep["e12_run"])
# 1 fully solved per task
by = defaultdict(list)
for u in rep["eval_unit_records"]:
    by[u["task"]].append(bool(u["hit"]))
full = [t for t, h in by.items() if h and all(h)]
print(f"\n1. inputs {rep['eval_query_hits']}/{rep['eval_query_total']}, task-norm {rep['eval_task_norm_pct']} %, "
      f"fully solved tasks {len(full)}/{len(by)}: {sorted(full)}")
for t in rep["order"]:
    print(f"   {t}: inputs {sum(by[t])}/{len(by[t])}")
# 2-5 per branch
print("\n2-5. per branch (order = queue order)")
print(f"{'task':9} br | acc | turns | req | gen tok | max compl | at 60k | sec  | timeouts | trims | overflow | submits(train exact)")
tot_to = tot_tr = 0
for t in rep["order"]:
    for b in (0, 1):
        p = os.path.join(d, "duck", f"{t}_b{b}")
        br = json.load(open(os.path.join(p, "branch_result.json"), encoding="utf-8"))
        tx = open(os.path.join(p, "transcript.txt"), encoding="utf-8", errors="replace").read()
        us = br["usage"]
        comp = [x["completion"] for x in us]
        trims = sum(1 for i in range(1, len(us)) if us[i]["prompt"] < us[i - 1]["prompt"])
        to = tx.count("request_timeout (yielding"); ov = tx.count("context_overflow_recovered")
        tot_to += to; tot_tr += trims
        subs = [s.get("train_exact") for s in br.get("submissions", [])]
        sx = [f"{sum(map(bool, e))}/{len(e)}" for e in subs if e is not None]
        print(f"{t:9} {b}  | {'Y' if br['accepted'] else '-':3} | {br['turns']:5} | {len(us):3} | {br['generated_tokens']:7} | "
              f"{max(comp) if comp else 0:9} | {sum(c >= 60000 for c in comp):6} | {br['t1'] - br['t0']:5.0f}| {to:8} | {trims:5} | {ov:8} | {sx}")
print(f"   total timeouts {tot_to}, trims {tot_tr}")
# 7 serializer tampering in model code
pat = re.compile(r"json\.dumps\s*=|json\.encoder|JSONEncoder|import builtins|builtins\.|sys\.modules|HOST_STDOUT|_send\s*\(")
hits = []
for f in glob.glob(os.path.join(d, "duck", "*", "transcript.txt")):
    for m in pat.finditer(open(f, encoding="utf-8", errors="replace").read()):
        hits.append((os.path.basename(os.path.dirname(f)), m.group(0)))
print("\n7. serializer tampering patterns in transcripts:", hits or "none")
print("\nusage_all:", rep["usage_all"])
ds = rep.get("e11", {}).get("decode_summary", {})
print("decode running hist:", ds.get("running_hist"), "| retract lines:", ds.get("retract_lines"))
