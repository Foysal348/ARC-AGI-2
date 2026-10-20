"""Q27-AGENT v1.4: sandbox + python tool from 16 threads without preexec_fn (hundreds of subprocesses), an infinite
loop still times out, the child memory-limit preamble never breaks a run (on Windows `resource` is absent -> skipped).
usage: python test_v14_concurrency.py <notebook>"""
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

NB = sys.argv[1]
d = Path(tempfile.mkdtemp(prefix="q27a14_")) / "agent"
d.mkdir(parents=True)
for c in json.load(open(NB, encoding="utf-8"))["cells"]:
    s = "".join(c["source"])
    if s.startswith("%%writefile /kaggle/working/agent/"):
        (d / s.splitlines()[0].split("/")[-1]).write_text("\n".join(s.splitlines()[1:]) + "\n", encoding="utf-8")
sys.path.insert(0, str(d))
import repl, sandbox   # noqa: E402

assert "preexec_fn=" not in (d / "sandbox.py").read_text() and "preexec_fn=" not in (d / "repl.py").read_text()
task = {"train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}], "test": [{"input": [[5, 6], [7, 8]]}]}
prog = "import numpy as np\ndef transform(grid):\n    return np.rot90(np.array(grid), 2).tolist()\n"


def job(i):
    if i % 2:
        r = sandbox.run_one(prog, [task["train"][0]["input"], task["test"][0]["input"]], timeout=30)
        return r[0].get("output") == [[4, 3], [2, 1]] and r[1].get("output") == [[8, 7], [6, 5]]
    o = repl.run_snippet("print(train_inputs[0].shape, int(train_outputs[0].sum()))", task, timeout=30)
    return "(2, 2) 10" in o


t0 = time.time()
with ThreadPoolExecutor(max_workers=16) as ex:
    res = list(ex.map(job, range(320)))
took = time.time() - t0
assert all(res), res.count(False)
t1 = time.time()
r = sandbox.run_one("def transform(grid):\n    while True:\n        pass\n", [[[1]]], timeout=3)
assert "timed out" in r[0].get("error", "") and time.time() - t1 < 10, r
o = repl.run_snippet("while True:\n    pass", task, timeout=3)
assert "timed out" in o, o
print(f"320 subprocesses from 16 threads OK in {took:.1f}s; infinite loops time out (sandbox + python tool)")
print("V1.4 CONCURRENCY TEST PASS")
