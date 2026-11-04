"""Replay a branch's tool calls up to its first exact check(); apply that transform to the test inputs; compare with truth."""
import json, re, sys
tx_path, task_id, ch, so = sys.argv[1:5]
tx = open(tx_path, encoding="utf-8", errors="replace").read()
task = json.load(open(ch))[task_id]; truth = json.load(open(so))[task_id]
codes = re.findall(r"\[TOOL CALL: python\]\n<tool_call>\n<function=python>\n<parameter=code>\n(.*?)\n</parameter>", tx, re.S)
results = re.findall(r"\[TOOL RESULT: python\]\n(.*?)(?=\n\[[A-Z])", tx, re.S)
ns = {}
import collections, itertools, math, copy, functools
picked = []
found = []
best = [0.0]
def check(fn, verbose=True):
    ok = [fn([list(r) for r in p["input"]]) == p["output"] for p in task["train"]]
    picked.append((fn, sum(ok), len(ok)))
    return ok
for i, code in enumerate(codes):
    g = dict(ns); g.update({"train": task["train"], "test_inputs": [t["input"] for t in task["test"]], "check": check, "submit": lambda f: None,
                            "print": lambda *a, **k: None})
    try:
        exec(code, g)
    except Exception as e:
        pass
    for k, v in g.items():
        if callable(v) and not k.startswith("__") and k not in ("train", "test_inputs", "check", "submit", "print"):
            ns[k] = v
        elif k in ("Counter", "deque", "defaultdict"):
            ns[k] = v
    while picked:
        fn, k, n = picked.pop(0)
        best[0] = max(best[0], k / max(1, n))
        if k == n:
            try:
                outs = [fn([list(r) for r in t["input"]]) for t in task["test"]]
                found.append((i + 1, all(o == t for o, t in zip(outs, truth))))
            except Exception:
                found.append((i + 1, None))
print(json.dumps({"task": task_id, "tool_calls": len(codes), "exact": found, "best_frac": round(best[0], 2)}))
