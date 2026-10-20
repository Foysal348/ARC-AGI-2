"""Q27-AGENT v1.2 local tests: every audit R1 + R2 fix that can run on a CPU (no GPU, no real vLLM).

usage: python test_v12_local.py <notebook> <dir with arc-agi_evaluation_{challenges,solutions}.json>
"""
import json
import os
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

NB, COMP = sys.argv[1], sys.argv[2]
work = Path(tempfile.mkdtemp(prefix="q27a12_"))
agent_dir = work / "agent"
agent_dir.mkdir()
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"]]
for s in cells:
    if s.startswith("%%writefile /kaggle/working/agent/"):
        (agent_dir / s.splitlines()[0].split("/")[-1]).write_text("\n".join(s.splitlines()[1:]) + "\n",
                                                                   encoding="utf-8")
sys.path.insert(0, str(agent_dir))
import agent, client, run_eval, submit_main   # noqa: E402

ch = json.load(open(Path(COMP) / "arc-agi_evaluation_challenges.json"))
sol = json.load(open(Path(COMP) / "arc-agi_evaluation_solutions.json"))
A, B, C, D, E = "136b0064", "247ef758", "4c3d4a41", "2b83f449", "898e7135"
results = {}

# ---- 1. markup recovery + argument normalisation (5 cases)
closed = "<tool_call>\n<function=submit>\n<parameter=code>\ndef transform(g):\n    return g\n</parameter>\n</function>\n</tool_call>"
unclosed = "<tool_call>\n<function=submit>\n<parameter=code>\ndef transform(g):\n    return g\n</function>\n</tool_call>"
empty = "<tool_call>\n<function=submit>\n</function>\n</tool_call>"
assert "def transform" in client.recover_markup(closed)[0]["arguments"]["code"]
assert "def transform" in client.recover_markup(unclosed)[0]["arguments"]["code"]
assert client.recover_markup(empty)[0]["arguments"] == {}
mk = lambda a: {"choices": [{"message": {"content": "", "tool_calls": [{"id": "x", "function": {"name": "submit",
                                                                                                 "arguments": a}}]},
                              "finish_reason": "stop"}]}
assert client.normalise(mk("{not json"))["tool_calls"][0]["arguments"] == {"_raw": "{not json"}
assert "_invalid_arguments" in client.normalise(mk("[1, 2]"))["tool_calls"][0]["arguments"]
results["parser_5_cases"] = "PASS"


# ---- 2. agent with a scripted client
def lookup_program(tid, with_test=True):
    pairs = [(p["input"], p["output"]) for p in ch[tid]["train"]]
    if with_test:
        pairs += [(t["input"], s) for t, s in zip(ch[tid]["test"], sol[tid])]
    return "PAIRS = " + json.dumps(pairs) + "\ndef transform(grid):\n    for i, o in PAIRS:\n        if i == grid:\n" \
           "            return o\n    return grid\n"


class Mock:
    def __init__(self, *a, **k):
        self.max_tokens = k.get("max_tokens", 8192)
        self.log = []
        self.lock = threading.Lock()

    def chat(self, messages, tools=None, max_tokens=None, seed=None, timeout=None, tool_choice=None,
             template_kwargs=None):
        prompt = messages[1]["content"]
        tid = next(t for t in (A, B, C, D, E) if "".join(map(str, ch[t]["train"][0]["input"][0])) in prompt)
        turn = sum(m["role"] == "assistant" for m in messages)
        with self.lock:
            self.log.append((tid, turn, max_tokens))
        base = {"reasoning": "r", "prompt_tokens": 900, "completion_tokens": 7000, "finish_reason": "stop",
                "content": ""}
        call = lambda name, args: dict(base, tool_calls=[{"id": f"c{turn}", "name": name, "arguments": args}])
        if tid == A:
            return call("python", {"code": "print(1)"}) if turn == 0 else call("submit", {"code": lookup_program(A)})
        if tid == B:
            if turn == 0:
                return call("submit", {"wrong_key": "x"})            # no usable code -> targeted feedback
            return call("submit", {"code": "def transform(grid):\n    return grid\n"})
        if tid == D:                                                 # touches the filesystem -> flagged
            return call("python", {"code": "import os\nprint(os.listdir('/kaggle/input'))"}) if turn == 0 else \
                call("submit", {"code": "def transform(grid):\n    return grid\n"})
        if tid == E:                                                 # fits train, wrong on test
            return call("submit", {"code": lookup_program(E, with_test=False)})
        return dict(base, tool_calls=[])                             # C: plain text


mock = Mock(max_tokens=8192)
run_eval.ChatClient = lambda *a, **k: mock
out = work / "agent_eval"
cfg = {"base_url": "http://mock", "model": "mock", "data_dir": COMP, "eval_task_ids": [A, B, C, D, E],
       "eval_out": str(out), "deadline_epoch": time.time() + 600, "attempts": 2, "concurrency": 6, "max_turns": 30,
       "eval_cap": 20000, "context_budget": 28000, "nudge_every": 4, "turn_max_tokens": 8192, "in_python": True,
       "force_submit_after": 5, "effort_ladder": ["medium", "low"], "retain_functions": True, "keep_recent": 6,
       "trim_drain": 6000, "degenerate_guard": True, "timeout_as_cut": True, "snippet_timeout": 30.0}
run_eval.main(cfg)
rep = json.loads((out / "eval_report.json").read_text())
recs = [json.loads(l) for l in open(out / "attempts.jsonl", encoding="utf-8")]
caps_b = [m for t, turn, m in mock.log if t == B and turn == 2]
assert caps_b and max(caps_b) <= 20000 - 2 * 7000, caps_b
assert all(m <= 8192 for _, _, m in mock.log)
results["request_cap"] = f"PASS (B turn-3 caps {sorted(set(caps_b))})"
by_task = {}
for r in recs:
    by_task.setdefault(r["task_id"], []).append(r["idx"])
assert sorted(by_task[A]) == [0] and sorted(by_task[B]) == [0, 1] and sorted(by_task[C]) == [0, 1], by_task
results["phased_attempts"] = f"PASS {by_task}"
b0 = next(r for r in recs if r["task_id"] == B and r["idx"] == 0)
assert b0["status"] != "crashed" and any("no valid 'code' string" in (x.get("output") or "") for x in b0["transcript"])
results["invalid_code_feedback"] = "PASS"
nq = sum(len(sol[t]) for t in (A, B, C, D, E))
assert rep["queries"] == nq and abs(rep["official_score_pct"] - 100.0 * rep["query_hits"] / nq) < 0.01   # report is rounded to 2 decimals
results["official_metric"] = f"PASS official {rep['official_score_pct']}% task-macro {rep['task_macro_score_pct']}%"
assert rep["file_access_suspicious_tasks"] == [D], rep["file_access_suspicious_tasks"]
assert rep["verified_but_wrong_tasks"] == [E], rep["verified_but_wrong_tasks"]
results["contamination_flag_and_verified_but_wrong"] = "PASS"
pj = json.loads((out / "progress.json").read_text())
assert pj["attempts_done"] == len(recs) and not (out / "progress.json.tmp").exists()
assert "os.replace(out / \"progress.json.tmp\"" in (agent_dir / "run_eval.py").read_text()
results["progress_atomic"] = "PASS"

# ---- 3. last cell (rewritten validator) on 8 kinds of input
val_src = cells[14]
tdir = work / "val"
(tdir / "input").mkdir(parents=True)
test_ch = {"t1": {"train": [], "test": [{"input": [[1, 2]]}]}, "t2": {"train": [], "test": [{"input": [[3]]}, {"input": [[4]]}]}}
(tdir / "input" / "arc-agi_test_challenges.json").write_text(json.dumps(test_ch))
sub_path = tdir / "submission.json"
src = val_src.replace('"/kaggle/input/**/arc-agi_test_challenges.json"', repr(str(tdir / "input" / "**" / "arc-agi_test_challenges.json"))) \
             .replace('PATH = "/kaggle/working/submission.json"', "PATH = " + repr(str(sub_path)))
good = {"attempt_1": [[5]], "attempt_2": [[6]]}
cases = {
    "valid": ({"t1": [good], "t2": [good, good]}, {"t1": [good], "t2": [good, good]}),
    "missing_task": ({"t1": [good]}, None),
    "null_rows": ({"t1": None, "t2": [good, good]}, None),
    "string_row": ({"t1": ["invalid row"], "t2": [good, good]}, None),
    "ragged_grid": ({"t1": [{"attempt_1": [[1], [1, 2]], "attempt_2": [[1]]}], "t2": [good, good]}, None),
    "unknown_task": ({"t1": [good], "t2": [good, good], "zzz": [good]}, {"t1": [good], "t2": [good, good]}),
    "bool_value": ({"t1": [{"attempt_1": [[True]], "attempt_2": [[1]]}], "t2": [good, good]}, None),
    "truncated_json": ("{\"t1\": [", None),
}
for name, (content, expect) in cases.items():
    sub_path.write_text(content if isinstance(content, str) else json.dumps(content))
    exec(compile(src, "<cell14>", "exec"), {"__name__": "__main__"})
    got = json.loads(sub_path.read_text())
    assert set(got) == {"t1", "t2"} and len(got["t1"]) == 1 and len(got["t2"]) == 2, (name, got)
    for t, task in test_ch.items():
        for r, x in zip(got[t], task["test"]):
            assert r["attempt_1"] in ([[5]], x["input"]) and isinstance(r, dict), (name, r)
    if expect is not None:
        assert got == expect, (name, got)
    if name == "string_row":
        assert got["t1"][0]["attempt_1"] == [[1, 2]] and got["t2"] == [good, good]   # only the bad row replaced
results["validator_8_cases"] = "PASS"


# ---- 4. tool smoke: semantic checks, budget drop, one global deadline
def fake_server(mode, delay=0.0):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if delay:
                time.sleep(delay)
            if mode == "budget_rejected" and body.get("thinking_token_budget"):
                data = b'{"error": "thinking_token_budget is not supported without reasoning config"}'
                self.send_response(400)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            choice = body.get("tool_choice")
            name = choice["function"]["name"] if isinstance(choice, dict) else "python"
            if mode == "invalid_code":
                code = "definitely NOT executable python code !!!"
            elif name == "submit":
                code = "def transform(grid):\n    return grid"
            else:
                code = "print(train_inputs[0].shape)"
            msg = {"content": "", "tool_calls": [{"id": "t1", "type": "function",
                                                  "function": {"name": name, "arguments": json.dumps({"code": code})}}]}
            data = json.dumps({"choices": [{"message": msg, "finish_reason": "tool_calls"}],
                               "usage": {"prompt_tokens": 10, "completion_tokens": 20}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
    srv = ThreadingHTTPServer(("127.0.0.1", 1234), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


with socket.socket() as s_:
    assert s_.connect_ex(("127.0.0.1", 1234)) != 0, "port 1234 busy"
scfg = {"retain_functions": True, "template_kwargs": {"reasoning_effort": "high"}}
for mode, expect in (("ok", dict(ok=True, forced_ok=True, auto_ok=True, thinking_ok=True)),
                     ("invalid_code", dict(ok=False, forced_ok=False, auto_ok=False, thinking_ok=False)),
                     ("budget_rejected", dict(ok=True, forced_ok=True, auto_ok=True, thinking_ok=False,
                                              thinking_budget_retained=False))):
    srv = fake_server(mode)
    try:
        sm = submit_main.smoke(scfg, time.time() + 120)
    finally:
        srv.shutdown()
        srv.server_close()
    assert all(sm[k] == v for k, v in expect.items()), (mode, {k: sm[k] for k in expect})
srv = fake_server("ok", delay=20.0)            # hanging server: the whole smoke must stop at its deadline
try:
    t0 = time.time()
    sm = submit_main.smoke(scfg, time.time() + 6)
    took = time.time() - t0
finally:
    srv.shutdown()
    srv.server_close()
assert took < 15 and not sm["ok"] and sm["deadline_hit"], (took, sm["deadline_hit"], sm["ok"])
results["tool_smoke"] = f"PASS (ok / invalid code FAIL / budget dropped -> thinking off / hang stopped in {took:.1f}s)"

# ---- 5. server ladder: smoke-gated, explicit no-thinking fallback, reserve respected
calls = {"start": [], "kill": 0, "smoke_deadlines": []}


class FakeProc:
    pid = 999999


submit_main.install = lambda cfg: None
submit_main.ensure_libcuda = lambda: {"probe": "FAIL", "probe_out": "ld: cannot find -lcuda"}
submit_main.kill = lambda proc: calls.__setitem__("kill", calls["kill"] + 1)
smokes = iter([{"ok": False, "forced_ok": False, "thinking_ok": False},
               {"ok": True, "forced_ok": False, "thinking_ok": False, "thinking_budget_retained": False}])


def fake_smoke(cfg_, deadline):
    calls["smoke_deadlines"].append(deadline - time.time())
    return next(smokes)


submit_main.smoke = fake_smoke
seen_cfg = {}
stub = type(sys)("run_eval")
stub.main = lambda c: seen_cfg.update(c)
sys.modules["run_eval"] = stub
submit_main.REPORT_PATH = work / "agent_server_report.json"
submit_main.SITE = work / "site"


def fake_start(cfg_, extra, timeout=1500):
    calls["start"].append((tuple(extra), round(timeout)))
    return FakeProc()


submit_main.start_server = fake_start
submit_main.time.sleep = lambda s: None
base = {"server_ladder": [["--a"], ["--b"], []], "eval": True, "force_submit_after": 5, "min_agent_s": 1800,
        "thinking_token_budget": 6144}
submit_main.main(dict(base, deadline_epoch=time.time() + 4 * 3600))
srep = json.loads((work / "agent_server_report.json").read_text())
assert len(calls["start"]) == 2 and srep["chosen"]["index"] == 1, (calls, srep)
assert seen_cfg.get("force_submit_after") == 0 and seen_cfg.get("thinking") is False
assert seen_cfg.get("thinking_token_budget") is None
assert all(d <= 180 for d in calls["smoke_deadlines"]), calls["smoke_deadlines"]
assert "install_s" in srep and "remaining_agent_s" in srep and all("smoke_s" in r for r in srep["rungs"])
calls["start"].clear()
seen_cfg.clear()
submit_main.main(dict(base, deadline_epoch=time.time() + 1800 + 200))     # < 300 s beyond the agent reserve
assert calls["start"] == [] and not seen_cfg, calls
results["server_ladder"] = "PASS (probe FAIL logged, not fatal; smoke deadline <= 180 s; thinking off + budget None)"

print(json.dumps(results, indent=1))
print("V1.2 LOCAL TESTS PASS")
