import json, glob, os, re, sys, collections
root = sys.argv[1]
BLOCK = re.compile(r"^\[(SYSTEM PROMPT|USER PROMPT|THINKING|TOOL CALL: python|TOOL RESULT: python|ASSISTANT|"
                   r"MODEL RESPONSE META|ANALYZER STATUS)\]\s*$", re.M)
rows = {(r["br"], r["k"]): r for r in json.load(open("rows.json"))}
out = []
for d in sorted(glob.glob(os.path.join(root, "duck", "*"))):
    br = os.path.basename(d)
    b = json.load(open(os.path.join(d, "branch_result.json")))
    u = b.get("usage") or []
    text = open(os.path.join(d, "transcript.txt"), encoding="utf-8", errors="replace").read()
    text = re.sub(r"^--- analysis_step=.*---\s*$", "", text, flags=re.M)
    parts = BLOCK.split(text)
    blocks = [(parts[i], parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    metas = []
    cur = None
    for kind, body in blocks:
        if kind == "MODEL RESPONSE META":
            m = re.search(r"reasoning_chars:\s*(\d+)", body)
            cur = {"rc": int(m.group(1)) if m else None, "think": None, "meta": body}
            metas.append(cur)
        elif kind == "THINKING" and cur is not None and cur["think"] is None:
            cur["think"] = body
    print(br, "usage", len(u), "meta blocks", len(metas))
    for i, m in enumerate(metas):
        t = m["think"]
        if t is None or m["rc"] is None:
            lead = trail = None
        else:
            body = t[1:] if t.startswith("\n") else t      # the line break after the [THINKING] header
            r = body[:m["rc"]]
            lead = len(r) - len(r.lstrip())
            trail = len(r) - len(r.rstrip())
        nxt = rows.get((br, i + 1))
        out.append(dict(br=br, i=i, rc=m["rc"], lead=lead, trail=trail, cls_next=nxt["cls"] if nxt else None))
json.dump(out, open("ws.json", "w"))
tab = collections.Counter((o["lead"], o["trail"], o["cls_next"]) for o in out if o["cls_next"])
for k, v in sorted(tab.items(), key=lambda x: -x[1]):
    print("lead_ws", k[0], "trail_ws", k[1], "-> next request", k[2], ":", v)
