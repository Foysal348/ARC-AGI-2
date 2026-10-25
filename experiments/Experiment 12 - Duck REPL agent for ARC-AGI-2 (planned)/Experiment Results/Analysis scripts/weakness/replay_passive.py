"""Approximate replay (host exec, NOT the Duck sandbox): after every tool call, run every function named
transform*/solve*/solver* on train; record the exact ones and whether that same source was ever passed to check() exact."""
import json, re, sys, inspect
tx_path, task_id, ch, so = sys.argv[1:5]
tx = open(tx_path, encoding="utf-8", errors="replace").read()
task = json.load(open(ch))[task_id]; truth = json.load(open(so))[task_id]
codes = re.findall(r"\[TOOL CALL: python\]\n<tool_call>\n<function=python>\n<parameter=code>\n(.*?)\n</parameter>", tx, re.S)
NAME = re.compile(r"(?:transform|solve|solver)\w*$")
ns, seen, exact_checked, out = {}, set(), set(), []
def check(fn, verbose=True):
    try:
        ok = [fn([list(r) for r in p["input"]]) == p["output"] for p in task["train"]]
    except Exception:
        ok = [False]
    if all(ok):
        exact_checked.add(getattr(fn, "_src", None) or id(fn))
    return ok
for i, code in enumerate(codes):
    g = dict(ns); g.update({"train": task["train"], "test_inputs": [t["input"] for t in task["test"]], "check": check,
                            "submit": lambda f: None, "print": lambda *a, **k: None})
    for m in re.finditer(r"^def (\w+)\(", code, re.M):
        pass
    try:
        exec(code, g)
    except Exception:
        pass
    for k, v in g.items():
        if callable(v) and not k.startswith("__") and k not in ("train", "test_inputs", "check", "submit", "print"):
            ns[k] = v
    for k, v in list(ns.items()):
        if not NAME.match(k) or not callable(v):
            continue
        key = (k, getattr(v, "__code__", None) and v.__code__.co_code)
        if key in seen:
            continue
        seen.add(key)
        try:
            ok = all(v([list(r) for r in p["input"]]) == p["output"] for p in task["train"])
        except Exception:
            ok = False
        if ok:
            try:
                tc = all(v([list(r) for r in t["input"]]) == s for t, s in zip(task["test"], truth))
            except Exception:
                tc = None
            out.append({"call": i + 1, "name": k, "test_correct": tc})
print(json.dumps({"task": task_id, "passive_exact": out, "checked_exact_count": len(exact_checked)}))
