import json, sys
from pathlib import Path
R = Path(sys.argv[1])
d = json.load(open(R / "q27x/e12_run_report.json"))
rows = {r["task"]: r for r in d["eval_b_attempts"]["rows"]}
print("task     b acc  gen    sec   1stTool 1stChk  1stTf  nudge  commit cutDrop lcut req tools note")
for b in sorted(d["branches"], key=lambda x: (x["task"], x["branch"])):
    e6 = b.get("rev6_events") or {}; e5 = b.get("rev5_events") or {}
    r = rows.get(b["task"], {})
    acc_ok = r.get("b%d_acc" % b["branch"])
    print("%-8s %d %-4s %-5s %6d %6.0f %6s %6s %6s %6s %3s %3s %3s %3s %3s %s" % (
        b["task"], b["branch"], "A" if b["accepted"] else "-", {True: "ok", False: "x", None: "?"}[acc_ok],
        b["generated_tokens"], b["sec"], e6.get("first_tool_tokens"), e6.get("first_check_tokens"),
        e6.get("first_transform_tokens"), e6.get("provisional_nudge_tokens"), (e6.get("resumes") or {}).get("commit", 0),
        e5.get("cut_tool_calls_dropped"), e6.get("length_cut_no_call"), e6.get("requests_started"), e6.get("tool_calls"),
        (b.get("note") or "")[:60]))
