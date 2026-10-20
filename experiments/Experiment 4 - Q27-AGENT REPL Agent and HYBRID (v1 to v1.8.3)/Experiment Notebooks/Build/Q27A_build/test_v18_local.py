"""Q27-AGENT v1.8.3 local tests (CPU, no server), one group per finding of the GPT final pre-GPU audit plus the v1.6
features: guard (fail-closed, canary, native routes), scope-aware tool-only names incl. `train`, two-tier caps,
merge, run_eval DEPLOYED vs all-rescued scores + verified-but-wrong from the agent + stage throughput + loss
diagnostics + rescue checkpoint, run_submit rerun caps + rescue order, breaker (first-N and sliding window),
agent end-to-end probe, launcher config.
usage: python test_v18_local.py <notebook> <dir with arc-agi_evaluation_{challenges,solutions}.json>"""
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

NB, COMP = sys.argv[1], sys.argv[2]
work = Path(tempfile.mkdtemp(prefix="q27a18_"))
agent_dir = work / "agent"
agent_dir.mkdir()
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"]]
for s in cells:
    if s.startswith("%%writefile /kaggle/working/agent/"):
        (agent_dir / s.splitlines()[0].split("/")[-1]).write_text("\n".join(s.splitlines()[1:]) + "\n",
                                                                  encoding="utf-8")
sys.path.insert(0, str(agent_dir))
results = {}

# ---------------------------------------------------------------- 1. guard (sandbox + python tool)
probe = work / "probe"
probe.mkdir()
sentinel = probe / "solution_sentinel.json"                 # "solution" in the name -> denied
sentinel.write_text("[[9]]")
fake_in = probe / "fakeinput"                               # stands in for /kaggle/input
fake_in.mkdir()
(fake_in / "data.json").write_text("[[8]]")
allowed = probe / "allowed.json"
allowed.write_text("[[7]]")
os.environ["Q27_DENY_PATHS"] = "/kaggle/input|solution|fakeinput"
import sandbox, repl          # noqa: E402

P = lambda p: repr(str(p))    # noqa: E731
RET = "\n    return grid\n"
LIBC = ("import ctypes, ctypes.util\n    lib = ctypes.CDLL(" + ("'msvcrt'" if os.name == "nt" else
                                                          "ctypes.util.find_library('c')") + ")\n    ")
OPEN_SYM = "_open" if os.name == "nt" else "open"
cases = {
    "open": f"def transform(grid):\n    open({P(sentinel)}).read(){RET}",
    "builtins_open": f"def transform(grid):\n    __import__('builtins').open({P(sentinel)}).read(){RET}",
    "io_open": f"import io\ndef transform(grid):\n    io.open({P(sentinel)}).read(){RET}",
    "os_open": f"import os\ndef transform(grid):\n    os.close(os.open({P(sentinel)}, os.O_RDONLY)){RET}",
    "pathlib": f"import pathlib\ndef transform(grid):\n    pathlib.Path({P(sentinel)}).read_text(){RET}",
    "listdir": f"import os\ndef transform(grid):\n    os.listdir({P(fake_in)}){RET}",
    "scandir": f"import os\ndef transform(grid):\n    list(os.scandir({P(fake_in)})){RET}",
    "relative_after_chdir": f"import os\ndef transform(grid):\n    os.chdir({P(probe)})\n    open('fakeinput/data.json').read(){RET}",
    "symlink": f"import os\ndef transform(grid):\n    os.symlink({P(sentinel)}, {P(probe / 'lnk')}){RET}",
    "subprocess": f"import subprocess, sys\ndef transform(grid):\n    subprocess.run([sys.executable, '-c', 'print(1)']){RET}",
    "os_system": f"import os\ndef transform(grid):\n    os.system('echo hi'){RET}",
    # v1.7: native routes (GPT reproduced a native-library read of the sentinel in v1.6)
    "ctypes_fopen": f"def transform(grid):\n    {LIBC}f = lib.fopen{RET}",
    "ctypes_getitem_open": f"def transform(grid):\n    {LIBC}f = lib[{OPEN_SYM!r}]{RET}",
    "ctypes_read": f"def transform(grid):\n    {LIBC}f = lib.fread{RET}",
    "cffi": f"def transform(grid):\n    import cffi{RET}",
    "sqlite_path": f"import sqlite3\ndef transform(grid):\n    sqlite3.connect({P(fake_in / 'x.db')}){RET}",
}
if sys.platform != "win32":
    cases["fork_exec"] = f"import _posixsubprocess\ndef transform(grid):\n    _posixsubprocess.fork_exec(){RET}"
for name, code in cases.items():
    r = sandbox.run_one(code, [[[1]]], timeout=30)
    txt = json.dumps(r)
    assert ("access denied" in txt or "not allowed" in txt) and r[0].get("output") is None, (name, txt[:300])
ok_code = (f"import numpy as np, json\ndef transform(grid):\n    v = json.load(open({P(allowed)}))[0][0]\n"
           f"    return (np.rot90(np.array(grid), 2) * 0 + v).tolist()\n")
r = sandbox.run_one(ok_code, [[[1, 2], [3, 4]]], timeout=30)
assert r[0].get("output") == [[7, 7], [7, 7]], r          # numpy + an allowed file still work
r = sandbox.run_one("import numpy as np, ctypes\ndef transform(grid):\n    return np.array(grid).tolist()\n",
                    [[[1, 2]]], timeout=30)
assert r[0].get("output") == [[1, 2]], r                   # importing ctypes itself (numpy does) still works
try:                                                       # scipy imports subprocess -> must keep working
    import scipy.ndimage  # noqa: F401
    r = sandbox.run_one("import numpy as np\nfrom scipy import ndimage\ndef transform(grid):\n"
                        "    lab, n = ndimage.label(np.array(grid) > 0)\n    return lab.tolist()\n",
                        [[[1, 0], [0, 1]]], timeout=60)
    assert r[0].get("output") == [[1, 0], [0, 2]], r
    scipy_note = "scipy OK"
except ImportError:
    scipy_note = "scipy not installed"
task = {"train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}], "test": [{"input": [[5, 6], [7, 8]]}]}
o = repl.run_snippet(f"print(open({P(sentinel)}).read())", task, timeout=30)
assert "access denied" in o and "[[9]]" not in o, o
o = repl.run_snippet(f"import os\nprint(os.listdir({P(fake_in)}))", task, timeout=30)
assert "access denied" in o and "data.json" not in o, o
o = repl.run_snippet(f"{LIBC.strip()}\nprint(lib.fopen)".replace("\n    ", "\n"), task, timeout=30)
assert "not allowed" in o, o
o = repl.run_snippet("print(train_inputs[0].shape, int(train_outputs[0].sum()))", task, timeout=30)
assert "(2, 2) 10" in o, o
# fail-closed: a guard that cannot prove itself (empty deny list -> canary not denied) refuses to run any code
os.environ["Q27_DENY_PATHS"] = ""
r = sandbox.run_one("def transform(grid):\n    return grid\n", [[[1]]], timeout=30)
assert r[0].get("output") is None, r
o = repl.run_snippet("print('should not run')", task, timeout=30)
assert "guard failed" in o and "should not run" not in o.replace("print('should not run')", ""), o
os.environ["Q27_DENY_PATHS"] = "/kaggle/input|solution|fakeinput"
r = sandbox.run_one("def transform(grid):\n    return grid\n", [[[1]]], timeout=30)
assert r[0].get("output") == [[1]], r
results["guard"] = (f"PASS ({len(cases)} routes denied in the sandbox incl. ctypes/cffi/sqlite, 3 in the python tool; "
                    f"numpy + ctypes import + allowed file OK; {scipy_note}; inactive guard -> child refuses)")

# ---------------------------------------------------------------- 2. tool-only names, scope-aware (incl. train)
import repl_agent             # noqa: E402
ident = {"train": [{"input": [[1, 2]], "output": [[1, 2]]}], "test": [{"input": [[3, 4]]}]}
rej = {"train_global": "def transform(grid):\n    return train[0][1].tolist()\n",
       "train_outputs": "def transform(grid):\n    return train_outputs[0].tolist()\n",
       "module_level": "x = test_inputs[0]\ndef transform(grid):\n    return grid\n",
       "global_decl": "def transform(grid):\n    global train\n    return train\n"}
acc = {"local_train": "def transform(grid):\n    train = [grid]\n    return train[0]\n",
       "module_train": "train = 5\ndef transform(grid):\n    return grid if train == 5 else None\n",
       "param_train": "def f(train):\n    return train\ndef transform(grid):\n    return f(grid)\n",
       "local_train_inputs": "def transform(grid):\n    train_inputs = 3\n    return grid\n",
       "attribute": "def transform(grid):\n    class A:\n        train = 1\n    return grid if A.train else None\n"}
for k, code in rej.items():
    n_ok, tests, fb = repl_agent.verify(code, ident)
    assert n_ok == 0 and tests == [None] and "only inside the python tool" in fb, (k, n_ok, fb)
for k, code in acc.items():
    n_ok, tests, fb = repl_agent.verify(code, ident)
    assert n_ok == 1 and tests == [[[3, 4]]], (k, n_ok, fb)
assert repl.tool_only_globals(repl.HELPERS) == set()
st = repl.new_store()
o = repl.run_snippet("import numpy as np\ndef helper(g):\n    return np.array(g)[::-1].tolist()\n"
                     "def transform(grid):\n    return train[0][1].tolist()\nprint('ok')", task, timeout=30, store=st)
assert "helper" in st["defs"] and "transform" not in st["defs"] and "not kept: transform" in o, (o, st["defs"].keys())
o = repl.run_snippet("def transform(grid):\n    train = helper(grid)\n    return train\nprint('ok2')", task, timeout=30,
                     store=st)
assert "transform" in st["defs"], o                       # a local `train` is portable -> kept
o = repl.run_snippet("def transform(grid):\n    return test_inputs[0].tolist()\nprint('ok3')", task, timeout=30, store=st)
assert "transform" not in st["defs"] and "helper" in st["defs"], o   # nonportable redefinition drops the old copy
full, added = repl.with_kept("def solve(g):\n    return helper(g)\ndef transform(grid):\n    return solve(grid)\n", st)
n_ok, _, fb = repl_agent.verify(full, {"train": [{"input": [[1], [2]], "output": [[2], [1]]}], "test": [{"input": [[3], [4]]}]})
assert n_ok == 1 and added == ["helper"], (n_ok, added, fb)
results["tool_only_names"] = f"PASS ({len(rej)} rejected incl. global train, {len(acc)} accepted incl. local/param/module train; store)"

# ---------------------------------------------------------------- 3. attempt_caps
seen = []
_real_run_attempt = repl_agent.run_attempt


def fake_run_attempt(att, task, client, shared, deadline, stop_after=1, seed=0, **over):
    seen.append((att.task_id, att.idx, over.get("max_gen_tokens")))
    att.status, att.note = "failed", "fake"


repl_agent.run_attempt = fake_run_attempt
repl_agent.solve_repl({"t1": task, "t2": task}, None, attempts=2, concurrency=2, deadline=time.time() + 60,
                      max_gen_tokens=60000, attempt_caps=[24000, 36000], log=lambda m: None)
assert sorted(seen) == [("t1", 0, 24000), ("t1", 1, 36000), ("t2", 0, 24000), ("t2", 1, 36000)], seen
seen.clear()
repl_agent.solve_repl({"t1": task}, None, attempts=1, concurrency=1, deadline=time.time() + 60,
                      max_gen_tokens=50000, log=lambda m: None)
assert seen == [("t1", 0, 50000)], seen
repl_agent.run_attempt = _real_run_attempt
results["attempt_caps"] = "PASS"

# ---------------------------------------------------------------- 4. merge (unit) + run_rescue
import rescue                 # noqa: E402
V, V2, U, D, I = [[1]], [[2]], [[3]], [[4]], [[0]]
M = lambda ranked, d: (lambda m: [m["attempt_1"], m["attempt_2"]])(rescue.merge(ranked, d, I))   # noqa: E731
assert M([(V, "verified x2"), (U, "unverified 2/3")], D) == [V, D]
assert M([(U, "unverified 2/3")], D) == [D, U]
assert M([], D) == [D, I]
assert M([], None) == [I, I]
assert M([(V, "verified x1"), (V2, "verified x1")], D) == [V, D]      # v1.8: one verified slot above direct
assert M([(V, "verified x1"), (V2, "verified x1")], None) == [V, V2]
assert M([(U, "unverified 1/3")], U) == [U, I]
assert M([(U, "unverified 1/3"), (V2, "unverified 0/3")], None) == [U, V2]
calls = []


class DummyClient:
    max_tokens = 36864
    budget_dropped = False


rescue.direct_client = lambda cfg: DummyClient()


def fake_answer(client, task, j, seed, deadline, commit_tokens=8192, think_budget=None):
    calls.append(task["_id"])
    if task["_id"] == "boom":
        raise RuntimeError("model went away")
    return {"status": "answered", "grid": task["_direct"][j], "gen_tokens": 100}


rescue.direct_answer = fake_answer
tk = {"a": dict(task, _id="a", _direct=[[[5]]]), "boom": dict(task, _id="boom", _direct=None)}
got_cb = []
recs, _ = rescue.run_rescue(["a", "boom"], tk, {"concurrency": 2}, time.time() + 3600, log=lambda m: None,
                            on_result=lambda t, r: got_cb.append(t))
assert recs["a"]["grids"] == [[[5]]] and recs["boom"]["status"].startswith("crashed") and sorted(got_cb) == ["a", "boom"]
calls.clear()
recs, _ = rescue.run_rescue(["a"], tk, {"rescue_min_start_s": 300}, time.time() + 10, log=lambda m: None)
assert recs["a"]["status"].startswith("skipped") and calls == []
rescue = importlib.reload(rescue)
c = rescue.direct_client({"thinking_token_budget": 6144})          # v1.7 crash: a cfg WITHOUT base_url / model
assert c.url == "http://127.0.0.1:1234/v1/chat/completions" and c.max_tokens == 49152 + 8192
assert c.thinking_token_budget == 49152 and c.template_kwargs == {"reasoning_effort": "xhigh"}
assert c.sampling == {"temperature": 0.6, "top_p": 0.95, "top_k": 20}
c2 = rescue.direct_client({"thinking": False, "thinking_token_budget": None})   # the agent's thinking fallback
assert c2.thinking is True and c2.thinking_token_budget == 49152, (c2.thinking, c2.thinking_token_budget)
assert "must NOT read train_inputs" in repl.RETAIN_DOC and "the preloaded names" not in repl.RETAIN_DOC
# the LB-18.75 prompt and parser, verbatim behaviour
assert rescue.render_grid([[1, 0], [2, 3]]) == "1 0\n2 3"
m0 = rescue.build_user_message(task["train"], task["test"][0]["input"])
assert "xhigh" not in m0 and "FINAL INPUT:" in m0 and m0.endswith("Output nothing after the block.")
G = [[1, 0, 2], [3, 4, 5]]
for txt in ("Done.\n```grid\n1 0 2\n3 4 5\n```", "```\n1 0 2\n3 4 5\n```", "```text\n102\n345\n```",
            "```grid\n1,0,2\n3,4,5\n```", "```json\n[[1, 0, 2], [3, 4, 5]]\n```", "1 0 2\n3 4 5",
            "```json\n{\"test_outputs\": [[[1, 0, 2], [3, 4, 5]]]}\n```"):
    assert rescue.parse_grid(txt) == G, txt
assert rescue.parse_grid("no grid here") is None


class ScriptClient:
    def __init__(self, replies):
        self.replies, self.reqs, self.max_tokens = list(replies), [], 57344

    def chat(self, messages, seed=None, max_tokens=None, timeout=None, template_kwargs=None, **k):
        self.reqs.append({"messages": messages, "max_tokens": max_tokens, "template_kwargs": template_kwargs})
        return self.replies.pop(0)


ok_r = {"content": "```grid\n4 3\n2 1\n```", "reasoning": "r", "finish_reason": "stop", "completion_tokens": 30000}
sc = ScriptClient([ok_r])
r = rescue.direct_answer(sc, task, 0, 7, time.time() + 3600)
assert r["status"] == "answered" and r["grid"] == [[4, 3], [2, 1]] and r["gen_tokens"] == 30000
assert sc.reqs[0]["messages"] == [{"role": "user", "content": m0}] and sc.reqs[0]["max_tokens"] == 57344
cut = {"content": "", "reasoning": "thinking " * 50, "finish_reason": "length", "completion_tokens": 57344}
sc = ScriptClient([cut, {"content": "```grid\n4 3\n2 1\n```", "reasoning": "", "finish_reason": "stop",
                         "completion_tokens": 40}])
r = rescue.direct_answer(sc, task, 0, 7, time.time() + 3600)
assert r["status"] == "committed" and r["gen_tokens"] == 57384
assert sc.reqs[1]["template_kwargs"] == {"enable_thinking": False, "reasoning_effort": "low"}
assert "Stop analysing now" in sc.reqs[1]["messages"][0]["content"] and sc.reqs[1]["max_tokens"] == 8192
over = {"error": "HTTP 400: maximum context length", "context_overflow": True}
sc = ScriptClient([over, ok_r])
r = rescue.direct_answer(sc, task, 0, 7, time.time() + 3600)
assert r["status"] == "answered" and sc.reqs[1]["max_tokens"] == int(57344 * 0.8)
now = time.time()
# T4: 10:50-like (53 min to the finish), fast stream -> capped at 49,152
assert rescue.time_budget(49152, now + 53 * 60, 30.0) == 49152
b = rescue.time_budget(49152, now + 53 * 60, 17.2)                       # GPT's example: ~45k, not the full 49k
assert abs(b - (int((53 * 60 - 120) * 17.2) - 4096)) <= 20 and b < 49152, b
# T3: 11:34-like (9 min left), slow stream -> no new long request
assert rescue.time_budget(49152, now + 9 * 60, 5.0) is None
assert rescue.time_budget(49152, now + 60, 17.0) is None
real = rescue.direct_client({})
cap = {}


class CapClient:
    def __init__(self, base):
        self.max_tokens, self.thinking_token_budget = base.max_tokens, base.thinking_token_budget

    def chat(self, messages, seed=None, max_tokens=None, timeout=None, template_kwargs=None, **k):
        cap.update(max_tokens=max_tokens, budget=self.thinking_token_budget, timeout=timeout)
        return ok_r


base = CapClient(real)
r = rescue.direct_answer(base, task, 0, 7, time.time() + 1200, think_budget=15000)     # T1: per-request budget
assert cap["budget"] == 15000 and cap["max_tokens"] == 15000 + 8192 and r["think_budget"] == 15000, cap
assert base.thinking_token_budget == 49152 and base.max_tokens == 57344            # the shared client is untouched
assert 1150 < cap["timeout"] <= 1260, cap["timeout"]
# T2: the server rejects the budget -> the client drops it, but the request stays BOUNDED by max_tokens (no
# unlimited thinking): the retry carries max_tokens = budget + answer and no thinking_token_budget
import threading as _th
from http.server import BaseHTTPRequestHandler as _BH, ThreadingHTTPServer as _TS
seen2 = []


class _H(_BH):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        seen2.append({"budget": body.get("thinking_token_budget"), "max_tokens": body.get("max_tokens")})
        if body.get("thinking_token_budget"):
            data = json.dumps({"error": "thinking_token_budget is not supported"}).encode()
            self.send_response(400)
        else:
            data = json.dumps({"choices": [{"message": {"content": "```grid\n4 3\n2 1\n```"}, "finish_reason": "stop"}],
                               "usage": {"completion_tokens": 9}}).encode()
            self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


_srv = _TS(("127.0.0.1", 0), _H)
_th.Thread(target=_srv.serve_forever, daemon=True).start()
cl2 = rescue.direct_client({"base_url": f"http://127.0.0.1:{_srv.server_address[1]}/v1"})
r = rescue.direct_answer(cl2, task, 0, 7, time.time() + 600, think_budget=15000)
_srv.shutdown()
assert r["status"] == "answered" and seen2[0] == {"budget": 15000, "max_tokens": 23192}, seen2
assert seen2[1] == {"budget": None, "max_tokens": 23192}, seen2                     # bounded retry
assert cl2.thinking_token_budget == 49152                                           # shared client unchanged
# planning inside run_rescue: finish = deadline + 480 s, conservative 15 tok/s at the start
plans = []
rescue.direct_client = lambda cfg: DummyClient()


def plan_answer(client, task, j, seed, deadline, commit_tokens=8192, think_budget=None):
    plans.append((round(deadline - time.time()), think_budget))
    time.sleep(0.2)
    return {"status": "answered", "grid": [[1]], "gen_tokens": 2}                 # 2 tokens / 0.2 s = 10 tok/s


rescue.direct_answer = plan_answer
rescue.run_rescue(["a"], {"a": dict(task, _id="a", _direct=[[[5]]])}, {}, time.time() + 1800, log=lambda m: None)
fin, bud = plans[-1]
assert 1800 + 480 - 5 <= fin <= 1800 + 480 and abs(bud - (int((1800 + 480 - 120) * 15) - 4096)) <= 40, plans
# T5: the measured speed drops -> later budgets shrink (25th percentile of finished requests after 4)
plans.clear()
six = {"t%d" % i: dict(task, _id="t%d" % i, _direct=[[[5]]]) for i in range(6)}
rescue.run_rescue(list(six), six, {"concurrency": 1}, time.time() + 1800, log=lambda m: None)
assert plans[4][1] < plans[0][1] and plans[5][1] < plans[3][1], plans                # 15 tok/s -> ~10 tok/s
# T3 inside run_rescue: 90 s before the deadline nothing useful fits -> skipped, no request
plans.clear()
recs, _ = rescue.run_rescue(["a"], {"a": dict(task, _id="a", _direct=[[[5]]])}, {}, time.time() + 90, log=lambda m: None)
assert recs["a"]["status"] == "skipped" and not plans, (recs, plans)
# T6: KV usage >= 95 % -> no new request until it drops


class _KV:
    def __init__(self):
        self.t0 = time.time()

    def latest_kv(self):
        return 0.97 if time.time() - self.t0 < 0.5 else 0.5


rescue.SAMPLER = _KV()
plans.clear()
t_kv = time.time()
rescue.run_rescue(["a"], {"a": dict(task, _id="a", _direct=[[[5]]])}, {"rescue_throttle_sleep_s": 0.1},
                  time.time() + 1800, log=lambda m: None)
assert plans and time.time() - t_kv >= 0.5, (plans, time.time() - t_kv)
rescue.SAMPLER = None
# T8: a crashed unit does not remove an answer already written
done8 = {}


def crash_second(client, task, j, seed, deadline, commit_tokens=8192, think_budget=None):
    if task["_id"] == "t1":
        raise RuntimeError("worker died")
    return {"status": "answered", "grid": [[7]], "gen_tokens": 5}


rescue.direct_answer = crash_second
recs8, _ = rescue.run_rescue(["t0", "t1"], six, {"concurrency": 1}, time.time() + 1800, log=lambda m: None,
                             on_result=lambda t, r: done8.__setitem__(t, list(r["grids"])))
assert done8["t0"] == [[[7]]] and recs8["t1"]["status"].startswith("crashed"), (done8, recs8["t1"])
rescue.direct_answer = fake_answer
# metrics sampler: peak not final, counter deltas, restart-safe
seq = iter([{"kv_cache_usage_perc": 0.2, "num_preemptions_total": 2.0}, {"kv_cache_usage_perc": 0.8, "num_preemptions_total": 3.0},
            {"kv_cache_usage_perc": 0.4, "num_preemptions_total": 5.0}, {"kv_cache_usage_perc": 0.0, "num_preemptions_total": 1.0},
            {"error": "down"}])
_real_sm = rescue.server_metrics
rescue.server_metrics = lambda url="": next(seq)
smp = rescue.MetricsSampler()
for _ in range(5):
    smp.sample()
summ = smp.summary()
rescue.server_metrics = _real_sm
assert summ["kv_usage_peak"] == 0.8 and summ["preemptions"] == 3.0 + 1.0 and summ["failures"] == 1, summ
assert rescue.MetricsSampler("http://127.0.0.1:9/v1").stop()["metrics_unavailable"] is True
METRICS = ("# HELP x\n" 'vllm:cache_config_info{block_size="1600",num_gpu_blocks="846"} 1.0\n'
           'vllm:num_preemptions_total{model_name="qwen"} 0.0\n'
           'vllm:num_requests_running{model_name="qwen"} 12.0\n'
           'vllm:spec_decode_num_accepted_tokens_total{model_name="qwen"} 661.0\n'
           'vllm:spec_decode_num_draft_tokens_total{model_name="qwen"} 1000.0\n')


class _R:
    def read(self):
        return METRICS.encode()


_real_urlopen = rescue.urllib.request.urlopen
rescue.urllib.request.urlopen = lambda url, timeout=10: _R()
mx = rescue.server_metrics()
rescue.urllib.request.urlopen = _real_urlopen
assert mx["kv_tokens"] == 846 * 1600 and mx["num_preemptions_total"] == 0.0 and mx["mtp_acceptance"] == 0.661, mx
assert mx["num_requests_running"] == 12.0 and "error" in rescue.server_metrics("http://127.0.0.1:9/v1")
results["v183_timing"] = ("PASS (T1 per-request budget, T2 bounded retry on budget reject, T3 late reject, T4 cap, "
                          "T5 p25 speed adapts, T6 KV throttle, T8 crash keeps answers; sampler peak/delta/restart)")
results["merge_and_rescue"] = "PASS (merge v1.8; per test input; LB-18.75 prompt/parser; commit call; overflow shrink; no-base_url cfg)"

# ---------------------------------------------------------------- 5. run_eval: deployed vs all-rescued (GPT blocker)
ch = json.load(open(Path(COMP) / "arc-agi_evaluation_challenges.json"))
sol = json.load(open(Path(COMP) / "arc-agi_evaluation_solutions.json"))
A, B, C, Dt = "136b0064", "247ef758", "4c3d4a41", "2b83f449"
wrong = lambda t, j: [[(sol[t][j][0][0] + 1) % 10] + sol[t][j][0][1:]] + sol[t][j][1:]   # noqa: E731
wrong2 = lambda t, j: [[(sol[t][j][0][0] + 2) % 10] + sol[t][j][0][1:]] + sol[t][j][1:]  # noqa: E731
import run_eval               # noqa: E402

nA, nB, nC, nD = (len(sol[t]) for t in (A, B, C, Dt))
q = nA + nB + nC + nD
captured = {}


def att_rec(tid, status, idx=0, gen=1000):
    return {"task_id": tid, "idx": idx, "status": status, "note": "", "turns": 3, "python_calls": 1, "submits": 1,
            "cuts": 0, "forced": 0, "gen_tokens": gen, "prompt_tokens": 500, "wall": 1.0, "best_ok": 1,
            "n_train": len(ch[tid]["train"]), "effort_steps": 0, "evictions": 0, "transcript": [], "programs": []}


def fake_solve_eval(tasks, client, deadline=None, on_done=None, **kw):
    captured["deadline"] = deadline
    answers = {A: [[(wrong(A, j), "verified x1")] for j in range(nA)],          # train-verified but WRONG on test
               B: [[(wrong(B, j), "unverified 1/3")] for j in range(nB)],
               C: [[] for _ in range(nC)],
               Dt: [[(wrong(Dt, j), "unverified 2/3"), (sol[Dt][j], "unverified 1/3")] for j in range(nD)]}
    atts = [att_rec(A, "verified", gen=12000), att_rec(B, "failed"), att_rec(C, "failed"), att_rec(Dt, "failed")]
    atts[1]["transcript"] = [{"role": "tool", "content": "PermissionError: [sandbox] access denied: /x/solution"}]
    for r in atts:
        on_done(r)
    return answers, {"attempts": atts, "wall_s": 10.0}


direct_by = {A: [sol[A][j] for j in range(nA)], B: [sol[B][j] for j in range(nB)], C: None,
             Dt: [wrong2(Dt, j) for j in range(nD)]}


def fake_answer2(client, task, j, seed, deadline, commit_tokens=8192, think_budget=None):
    tid = next(t for t in (A, B, C, Dt) if task["test"] == ch[t]["test"])
    calls.append(tid)
    g = direct_by[tid][j] if direct_by[tid] else None
    return {"status": "answered" if g else "no grid", "grid": g, "gen_tokens": 2000}


rescue.direct_answer = fake_answer2
rescue.direct_client = lambda cfg: DummyClient()
run_eval.solve_repl = fake_solve_eval
for rescue_all in (True, False):
    calls.clear()
    out = work / f"eval_{rescue_all}"
    t_start = time.time()
    D_ = t_start + 1000
    run_eval.main({"base_url": "http://127.0.0.1:9/v1", "model": "qwen", "data_dir": COMP, "eval_out": str(out),
                   "eval_task_ids": [A, B, C, Dt], "deadline_epoch": D_, "rescue": True, "rescue_all": rescue_all,
                   "agent_frac": 0.6, "concurrency": 4, "attempt_caps": [24000, 36000],
                   "rerun_attempt_caps": [16000, 32000], "rescue_smoke_ok": True, "eval_direct": "live"})
    assert abs(captured["deadline"] - (t_start + 0.6 * (D_ - t_start))) < 3, (captured["deadline"], t_start)
    assert sorted(set(calls)) == (sorted([A, B, C, Dt]) if rescue_all else sorted([B, C, Dt])), calls
    assert len(calls) == (q if rescue_all else nB + nC + nD), calls            # one request per test input
    rep = json.loads((out / "eval_report.json").read_text())
    cm = rep["complementarity"]
    assert cm["queries"] == q and cm["agent_correct"] == nD and cm["direct_correct"] == (nA + nB if rescue_all else nB), cm
    assert cm["hybrid_correct"] == nB, cm                                           # deployed: A not rescued
    assert abs(rep["official_score_pct"] - 100.0 * nB / q) < 0.01, rep["official_score_pct"]
    if rescue_all:
        assert cm["hybrid_all_rescued_correct"] == nA + nB and abs(rep["hybrid_all_rescued_pct"] - 100.0 * (nA + nB) / q) < 0.01
        assert cm["hybrid_lost_vs_direct"] == nA and cm["all_rescued_lost_vs_direct"] == 0, cm
        assert cm["oracle_union_hits"] == nA + nB + nD and cm["three_way_complete"] is True, cm
    else:
        assert cm["hybrid_all_rescued_correct"] == nB and cm["three_way_complete"] is False, cm
    assert cm["hybrid_lost_vs_agent"] == nD and cm["all_rescued_lost_vs_agent"] == nD, cm   # merge evicted D's answer
    assert rep["verified_but_wrong_tasks"] == [A], rep["verified_but_wrong_tasks"]          # from the AGENT submission
    assert rep["verified_but_wrong_queries"] == [f"{A}#{j}" for j in range(nA)]
    assert rep["isolation_denied_tasks"] == [B] and rep["isolation_guard_failed_tasks"] == []
    assert rep["agent_generated_tokens"] == 15000 and abs(rep["agent_gen_tok_per_s"] - 1500.0) < 0.01, rep["agent_gen_tok_per_s"]
    assert rep["rescue_generated_tokens"] == 2000 * len(calls) and rep["rescue_wall_s"] >= 0
    assert rep["gen_tokens_total"] == 15000 + 2000 * len(calls)
    assert rep["verified_first_attempt_within_rerun_scout_cap"] == {"cap": 16000, "tasks": [A]}
    lines = (out / "rescue_attempts.jsonl").read_text().splitlines()
    assert sorted(json.loads(x)["task_id"] for x in lines) == sorted(calls)
    assert (out / "eval_submission.partial.json").exists()
    es = json.loads((out / "eval_submission.json").read_text())
    ea = json.loads((out / "eval_all_rescued_submission.json").read_text())
    assert es[A][0]["attempt_1"] == wrong(A, 0) and es[A][0]["attempt_2"] == ch[A]["test"][0]["input"]
    assert ea[A][0]["attempt_2"] == (sol[A][0] if rescue_all else ch[A]["test"][0]["input"])
    assert es[Dt][0] == {"attempt_1": wrong2(Dt, 0), "attempt_2": wrong(Dt, 0)}
results["run_eval_deployed_vs_all"] = (f"PASS (deployed {nB}/{q} vs all-rescued {nA + nB}/{q}; verified-but-wrong "
                                       f"[A] from agent; lost_vs_agent {nD}, lost_vs_direct {nA}; stage throughput; "
                                       f"rescue jsonl + partial snapshot; denials listed)")
calls.clear()
t_start = time.time()
run_eval.main({"base_url": "http://127.0.0.1:9/v1", "model": "qwen", "data_dir": COMP, "eval_out": str(work / "e0"),
               "eval_task_ids": [A, B, C, Dt], "deadline_epoch": t_start + 1000})
assert abs(captured["deadline"] - (t_start + 1000)) < 1e-6 and calls == []
rep = json.loads((work / "e0" / "eval_report.json").read_text())
assert rep["complementarity"]["three_way_complete"] is False
results["run_eval_no_rescue"] = "PASS"
# v1.8 default: DIRECT from the LB-18.75 run records (no GPU), agent keeps the whole time
REF = json.loads((agent_dir / "direct_ref_eval12.json").read_text())
calls.clear()
t_start = time.time()
run_eval.main({"base_url": "http://127.0.0.1:9/v1", "model": "qwen", "data_dir": COMP, "eval_out": str(work / "eref"),
               "eval_task_ids": [A, B, C, Dt], "deadline_epoch": t_start + 1000, "rescue": True, "rescue_all": True,
               "agent_frac": 0.6})
assert abs(captured["deadline"] - (t_start + 1000)) < 1e-6 and calls == []
rep = json.loads((work / "eref" / "eval_report.json").read_text())
cm = rep["complementarity"]
exp7 = sum(q_["correct"] for t in (A, B, C, Dt) for q_ in REF["run7"][t])
exp4 = sum(q_["correct"] for t in (A, B, C, Dt) for q_ in REF["run4"][t])
assert cm["direct_correct"] == exp7 and cm["reference_run4"]["direct_correct"] == exp4, (cm, exp7, exp4)
assert cm["three_way_complete"] is True and "26.94" in cm["direct_source"] and rep["rescue_generated_tokens"] == 0
assert cm["net_gain"]["direct_tokens_per_query"] and cm["net_gain"]["net_after_displacement"] is not None, cm["net_gain"]
assert not (work / "eref" / "rescue_attempts.jsonl").exists()
results["run_eval_ref"] = f"PASS (DIRECT from run7 records: {exp7} correct on these 4 tasks, run4 {exp4}; no live calls)"

# ---------------------------------------------------------------- 6. run_submit: rerun caps, rescue order, merge
import run_submit             # noqa: E402

sdir = work / "submit"
sdir.mkdir()
(sdir / "arc-agi_test_challenges.json").write_text(json.dumps({t: ch[t] for t in (A, B, C)}))
os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "1"


def fake_solve_submit(tasks, client, deadline=None, on_done=None, **kw):
    captured["deadline"], captured["caps"] = deadline, kw.get("attempt_caps")
    for tid, n_ok_fn, outs in ((A, lambda t: len(tasks[t]["train"]), [sol[A][j] for j in range(nA)]),
                               (B, lambda t: 1, [wrong(B, j) for j in range(nB)])):
        r = att_rec(tid, "verified" if tid == A else "failed")
        r["programs"] = [{"n_ok": n_ok_fn(tid), "test": outs}]
        on_done(r)
    return {}, {"attempts": [dict(att_rec(A, "verified")), dict(att_rec(B, "failed"))]}


direct_by = {A: [wrong(A, j) for j in range(nA)], B: [sol[B][j] for j in range(nB)], C: [sol[C][j] for j in range(nC)]}


def fake_answer3(client, task, j, seed, deadline, commit_tokens=8192, think_budget=None):
    tid = next(t for t in (A, B, C) if task["test"] == ch[t]["test"])
    calls.append(tid)
    return {"status": "answered", "grid": direct_by[tid][j], "gen_tokens": 10}


rescue.direct_answer = fake_answer3
run_submit.solve_repl = fake_solve_submit
calls.clear()
t_start = time.time()
sub_path = str(sdir / "submission.json")
run_submit.main({"base_url": "http://127.0.0.1:9/v1", "model": "qwen", "data_dir": str(sdir), "sub_path": sub_path,
                 "deadline_epoch": t_start + 2000, "mode": "repl", "rescue": True, "agent_frac": 0.5,
                 "concurrency": 1, "attempt_caps": [24000, 36000], "rerun_attempt_caps": [16000, 32000]})
assert abs(captured["deadline"] - (t_start + 0.5 * 2000)) < 3, captured["deadline"]
assert captured["caps"] == [16000, 32000], captured["caps"]
assert list(dict.fromkeys(calls)) == [C, B] and len(calls) == nC + nB, calls   # C (never attempted) first; A never
sub = json.load(open(sub_path))
assert sub[A][0]["attempt_1"] == sol[A][0]
assert sub[B][0] == {"attempt_1": sol[B][0], "attempt_2": wrong(B, 0)}, sub[B][0]
assert sub[C][0] == {"attempt_1": sol[C][0], "attempt_2": ch[C]["test"][0]["input"]}, sub[C][0]
results["run_submit_hybrid"] = "PASS (rerun caps 16k/32k, rescue order C then B, merged submission.json)"

# ---------------------------------------------------------------- 7. circuit breaker
ra = importlib.reload(repl_agent)
http = {"status": "failed", "gen_tokens": 0, "note": "HTTP 400: Unexpected reasoning effort high"}
refused = {"status": "failed", "gen_tokens": 0, "note": "<urlopen error [Errno 111] Connection refused>"}
healthy = {"status": "failed", "gen_tokens": 900, "note": "turn budget"}
for _ in range(5):
    ra.breaker(http)
ra.breaker(healthy)
assert ra.ABORT["flag"] is False
ra = importlib.reload(repl_agent)
for _ in range(6):
    ra.breaker(http)
assert ra.ABORT["flag"] is True and "first 6" in ra.ABORT["why"]
ra = importlib.reload(repl_agent)
for _ in range(6):
    ra.breaker(healthy)
for _ in range(11):
    ra.breaker(refused)
assert ra.ABORT["flag"] is False                          # 11 of the last 12 failed -> keep going
ra.breaker(refused)
assert ra.ABORT["flag"] is True and "last 12" in ra.ABORT["why"], ra.ABORT
ra = importlib.reload(repl_agent)
for _ in range(20):
    ra.breaker({"status": "failed", "gen_tokens": 0, "note": "token budget"})
    ra.breaker({"status": "stopped", "gen_tokens": 0, "note": "deadline"})
assert ra.ABORT["flag"] is False                          # budget / deadline endings are not request errors
ra = importlib.reload(repl_agent)
for _ in range(6):
    ra.breaker(http)


class NoCallClient:
    max_tokens = 100

    def chat(self, *a, **k):
        raise AssertionError("the client must not be called after the breaker tripped")


_, rep = ra.solve_repl({"t1": task}, NoCallClient(), attempts=2, concurrency=1, deadline=time.time() + 60,
                       max_gen_tokens=1000, max_turns=5, log=lambda m: None)
assert all((a["status"], a["note"][:8]) == ("stopped", "aborted:") for a in rep["attempts"])
results["circuit_breaker"] = "PASS (first-6 trip, sliding last-12 trip, 11/12 no trip, budget/deadline ignored)"

# ---------------------------------------------------------------- 8. agent end-to-end probe (GPT Patch B)
import client as client_mod   # noqa: E402
import submit_main            # noqa: E402

log_req = []


def fake_chat(self, messages, tools=None, max_tokens=None, seed=None, timeout=None, tool_choice=None,
              template_kwargs=None, _deadline=None):
    eff = self.template_kwargs.get("reasoning_effort")
    log_req.append({"effort": eff, "max_tokens": self.max_tokens, "budget": self.thinking_token_budget,
                    "tools": [t["function"]["name"] for t in tools or []], "n_msgs": len(messages),
                    "roles": [m["role"] for m in messages]})
    if eff == "xhigh":                                     # the configured effort never calls a tool
        return {"content": "thinking...", "reasoning": "x", "tool_calls": [], "finish_reason": "length",
                "completion_tokens": 8192}
    if len(messages) == 2:
        return {"content": "", "tool_calls": [{"id": "c1", "name": "python",
                                               "arguments": {"code": "print(train_inputs[0].shape)"}}],
                "finish_reason": "tool_calls", "completion_tokens": 300}
    return {"content": "", "tool_calls": [{"id": "c2", "name": "submit",
                                           "arguments": {"code": "def transform(grid):\n    return grid\n"}}],
            "finish_reason": "tool_calls", "completion_tokens": 200}


client_mod.ChatClient.chat = fake_chat
ecfg = {"template_kwargs": {"reasoning_effort": "xhigh"}, "effort_ladder": ["medium", "low"], "temperature": 0.6,
        "turn_max_tokens": 8192, "thinking": True, "thinking_token_budget": 6144, "retain_functions": True}
e2e = submit_main.agent_e2e(ecfg, time.time() + 600)
assert e2e["ok"] is True and e2e["effort"] == "medium" and len(e2e["tries"]) == 2, e2e
assert log_req[0] == {"effort": "xhigh", "max_tokens": 8192, "budget": 6144, "tools": log_req[0]["tools"],
                      "n_msgs": 2, "roles": ["system", "user"]} and "submit" in log_req[0]["tools"]
assert log_req[-1]["roles"] == ["system", "user", "assistant", "tool", "user"], log_req[-1]
log_req.clear()
e2e = submit_main.agent_e2e(dict(ecfg, effort_ladder=[]), time.time() + 600)
assert e2e["ok"] is False and len(e2e["tries"]) == 1
# v1.8.3 soft limit: python worked but little time is left -> bounded extension lets the submit finish
mcfg = dict(ecfg, template_kwargs={"reasoning_effort": "medium"}, effort_ladder=[])
e2e = submit_main.agent_e2e(mcfg, time.time() + 70, extend_to=time.time() + 400)
assert e2e["ok"] is True and e2e["tries"][0].get("extended") is True, e2e      # < 120 s left after python -> extended
e2e = submit_main.agent_e2e(mcfg, time.time() + 70)                 # no extension offered -> none applied
assert e2e["ok"] is True and not e2e["tries"][0].get("extended"), e2e
e2e = submit_main.agent_e2e(mcfg, time.time() + 900, extend_to=time.time() + 1200)   # plenty of time -> not needed
assert e2e["ok"] is True and not e2e["tries"][0].get("extended"), e2e
sm_src = (agent_dir / "submit_main.py").read_text()
assert 'cfg["template_kwargs"] = dict(cfg.get("template_kwargs") or {}, reasoning_effort=e2e["effort"])' in sm_src
assert 'cfg["rescue"] = False' in sm_src and "KERNEL_ISO" in sm_src
results["agent_e2e"] = "PASS (real request shape 8192 + 6144 + tools; xhigh fails -> medium chosen; roundtrip roles)"

# ---------------------------------------------------------------- 9. launcher config
c12 = next(s for s in cells if s.startswith("# ==================================================================== launch"))
AG = json.loads(c12.split('AGENT = json.loads(r"""', 1)[1].split('""")', 1)[0])
assert AG["attempt_caps"] == [24000, 36000] and AG["rerun_attempt_caps"] == [16000, 32000] and AG["rescue"] is True
assert AG["agent_frac"] == 0.5 and AG["rescue_think"] == 49152 and AG["rescue_answer"] == 8192
assert AG["template_kwargs"] == {"reasoning_effort": "xhigh"} and AG["turn_max_tokens"] == 10240
assert AG["eval_direct"] == "ref" and AG["rescue_commit_tokens"] == 8192
SV = json.loads(c12.split('SERVER = json.loads(r"""', 1)[1].split('""")', 1)[0])
assert '"num_speculative_tokens": 3' in SV["server_ladder"][0][-1] and '"method": "mtp"' in SV["server_ladder"][0][-1]
assert '"num_speculative_tokens": 2' in SV["server_ladder"][1][-1] and SV["server_ladder"][-1] == []
assert 'cfg["e2e_max_s"] = 300' in c12 and 'cfg["e2e_extend_s"] = 600' in c12
# T7: the process cut is still deadline + 15 min with deadline = START + 11 h 35 min (11:50), untouched by the rescue
assert 'cfg["deadline_epoch"] = START + 12 * 3600 - 25 * 60' in c12
assert 'timeout=max(60, cfg["deadline_epoch"] + 15 * 60 - time.time())' in c12
assert AG["rescue_finish_extra_s"] == 480 and AG["rescue_min_think"] == 4096 and AG["rescue_kv_throttle"] == 0.95
assert AG["e2e_max_s"] == 600
assert "DRYRUN = (not RERUN)" in c12 and any(x.startswith("# v1.8 DRYRUN check") for x in cells)
assert any(x.startswith("%%writefile /kaggle/working/agent/fake_vllm.py") for x in cells)
assert 'os.environ.get("Q27A_EVAL_HOURS", 1.5)' in c12 and "rescue_all=True" in c12 and '"Q27A_AGENT_FRAC", 0.6' in c12
results["launcher_config"] = "PASS"

print(json.dumps(results, indent=1))
print("V1.8.3 LOCAL TESTS PASS")
