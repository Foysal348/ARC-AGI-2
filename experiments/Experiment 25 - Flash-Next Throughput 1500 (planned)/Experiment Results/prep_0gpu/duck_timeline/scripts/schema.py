# Print structure only: keys, types, string lengths. Never prints string contents
# except short strings (<=24 chars) that look like identifiers/numbers when SAFE=1.
import json, sys, re
SAFE = True
idre = re.compile(r'^[A-Za-z0-9_.:\-/ %]{0,24}$')
def desc(v, depth, maxd, ind):
    p = '  '*ind
    if isinstance(v, dict):
        if depth >= maxd: return f"dict({len(v)})"
        out = [f"dict({len(v)})"]
        for i,(k,x) in enumerate(v.items()):
            if i >= 60: out.append(p+'  ...'); break
            kk = k if idre.match(str(k)) else f"<key len {len(str(k))}>"
            out.append(f"{p}  {kk}: {desc(x, depth+1, maxd, ind+1)}")
        return '\n'.join(out)
    if isinstance(v, list):
        if not v: return "list(0)"
        return f"list({len(v)}) of " + desc(v[0], depth+1, maxd, ind+1)
    if isinstance(v, str):
        if SAFE and idre.match(v): return f"str={v!r}"
        return f"str(len {len(v)})"
    return f"{type(v).__name__}={v!r}"[:60]
path = sys.argv[1]; maxd = int(sys.argv[2]) if len(sys.argv)>2 else 4
if path.endswith('.jsonl'):
    lines = open(path, encoding='utf-8', errors='replace').read().splitlines()
    print("lines", len(lines))
    keys = {}
    for i, l in enumerate(lines):
        try: d = json.loads(l)
        except Exception: print("bad line", i); continue
        if isinstance(d, dict):
            ks = tuple(sorted(d.keys()))
            keys.setdefault(ks, []).append(i)
    for ks, idx in keys.items():
        print(f"--- keyset n={len(idx)} first={idx[0]}")
        print(desc(json.loads(lines[idx[0]]), 0, maxd, 0))
else:
    d = json.load(open(path, encoding='utf-8', errors='replace'))
    print(desc(d, 0, maxd, 0))
