"""Parse Duck transcripts (E12 EVAL6 + EVAL30) into per-branch behaviour metrics for the weakness analysis.
Usage: python -I agent_weakness_parse.py <q27x dir> [<q27x dir> ...] > out.json"""
import glob, json, os, re, sys
from collections import Counter
sys.stdout.reconfigure(encoding="utf-8")
HDR = re.compile(r"^\[([A-Z][A-Za-z _:]*)\]\s*$")
CHK = re.compile(r"check:\s*(\d+)\s*/\s*(\d+)\s*train pairs correct")
def blocks(text):
    cur, buf = None, []
    for line in text.splitlines():
        m = HDR.match(line)
        if m:
            if cur: yield cur, "\n".join(buf)
            cur, buf = m.group(1), []
        else:
            buf.append(line)
    if cur: yield cur, "\n".join(buf)
out = []
for d in sys.argv[1:]:
    rep = json.load(open(os.path.join(d, "e12_run_report.json"), encoding="utf-8"))
    hit = {}
    for u in rep["eval_unit_records"]:
        hit.setdefault(u["task"], []).append(bool(u["hit"]))
    for bd in sorted(glob.glob(os.path.join(d, "duck", "*_b*"))):
        br = json.load(open(os.path.join(bd, "branch_result.json"), encoding="utf-8"))
        if not br.get("usage"):
            continue
        tx = open(os.path.join(bd, "transcript.txt"), encoding="utf-8", errors="replace").read()
        fin, rchars, ccalls, code_lens = Counter(), [], 0, []
        checks, errs, trunc, subs, users, yields = [], 0, 0, [], [], Counter()
        first_tool_reason = None; n_resp = 0; tool_calls = 0; no_tool_resp = 0
        exact_seen_at = None; tools_since = 0
        for name, body in blocks(tx):
            if name == "MODEL RESPONSE META":
                n_resp += 1
                f = re.search(r"finish_reason:\s*(\S+)", body); fin[f.group(1) if f else "?"] += 1
                rc = re.search(r"reasoning_chars:\s*(\d+)", body); rc = int(rc.group(1)) if rc else 0
                rchars.append(rc)
                tc = re.search(r"tool_call_count:\s*(\d+)", body); tc = int(tc.group(1)) if tc else 0
                if tc == 0: no_tool_resp += 1
                if tc and first_tool_reason is None: first_tool_reason = sum(rchars)
            elif name.startswith("TOOL CALL"):
                tool_calls += 1; code_lens.append(len(body))
                if "check(" in body: ccalls += 1
            elif name.startswith("TOOL RESULT"):
                for m in CHK.finditer(body):
                    k, n = int(m.group(1)), int(m.group(2)); checks.append((k, n))
                    if k == n and exact_seen_at is None: exact_seen_at = tool_calls
                if "Traceback" in body or '"error"' in body or "Error:" in body: errs += 1
                if "cut off" in body or "truncated" in body.lower(): trunc += 1
                s = re.search(r'"submission":\s*"([^"]{0,160})', body) or re.search(r"submission[\"']?:\s*[\"']?([^\n]{0,160})", body)
                if s: subs.append(s.group(1))
            elif name == "USER PROMPT":
                users.append(body.strip().splitlines()[0][:90] if body.strip() else "")
            elif name == "ANALYZER STATUS":
                y = re.search(r"Yielded control to solver:\s*([^\n.]+)", body)
                if y: yields[y.group(1)] += 1
        task = br["task_id"]
        out.append({"run": os.path.basename(os.path.dirname(d.rstrip("/\\"))), "task": task, "branch": br["branch"],
                    "accepted": br["accepted"], "task_hit_all": all(hit.get(task, [False])),
                    "gen_tokens": br["generated_tokens"], "requests": len(br["usage"]), "responses": n_resp,
                    "finish": dict(fin), "reasoning_chars": rchars, "reasoning_before_first_tool": first_tool_reason,
                    "no_tool_responses": no_tool_resp, "tool_calls": tool_calls, "check_calls": ccalls,
                    "checks": checks, "best_check": max(checks, key=lambda x: x[0] / max(1, x[1])) if checks else None,
                    "exact_check_at_tool": exact_seen_at, "tool_errors": errs, "tool_truncated": trunc,
                    "submissions": subs, "user_prompts": Counter(users).most_common(6), "yields": dict(yields),
                    "code_chars_median": sorted(code_lens)[len(code_lens) // 2] if code_lens else 0})
print(json.dumps(out, indent=0))
