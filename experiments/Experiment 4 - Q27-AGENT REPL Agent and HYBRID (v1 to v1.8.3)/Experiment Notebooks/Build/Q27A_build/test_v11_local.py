"""Q27-AGENT v1.1 local tests for every audit-R1 fix that can run on a CPU (no GPU, no real vLLM).

usage: python test_v11_local.py <notebook> <dir with arc-agi_evaluation_{challenges,solutions}.json>
"""
import json
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

NB, COMP = sys.argv[1], sys.argv[2]
work = Path(tempfile.mkdtemp(prefix="q27a11_"))
agent_dir = work / "agent"
agent_dir.mkdir()
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"]]
for s in cells:
    if s.startswith("%%writefile /kaggle/working/agent/"):
        (agent_dir / s.splitlines()[0].split("/")[-1]).write_text("\n".join(s.splitlines()[1:]) + "\n",
                                                                   encoding="utf-8")
sys.path.insert(0, str(agent_dir))
import agent, client, repl_agent, run_eval, submit_main   # noqa: E402

ch = json.load(open(Path(COMP) / "arc-agi_evaluation_challenges.json"))
sol = json.load(open(Path(COMP) / "arc-agi_evaluation_solutions.json"))
A, B, C = "136b0064", "247ef758", "4c3d4a41"
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


# ---- 2-4. agent with a scripted client: request cap, phased attempts, invalid code feedback, official metric
def lookup_program(tid):
    pairs = [(p["input"], p["output"]) for p in ch[tid]["train"]] + \
            [(t["input"], s) for t, s in zip(ch[tid]["test"], sol[tid])]
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
        tid = next(t for t in (A, B, C) if "".join(map(str, ch[t]["train"][0]["input"][0])) in prompt)
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
        return dict(base, tool_calls=[])                             # plain text


mock = Mock(max_tokens=8192)
run_eval.ChatClient = lambda *a, **k: mock
out = work / "agent_eval"
cfg = {"base_url": "http://mock", "model": "mock", "data_dir": COMP, "eval_task_ids": [A, B, C], "eval_out": str(out),
       "deadline_epoch": time.time() + 600, "attempts": 2, "concurrency": 6, "max_turns": 30, "eval_cap": 20000,
       "context_budget": 28000, "nudge_every": 4, "turn_max_tokens": 8192, "in_python": True,
       "force_submit_after": 5, "effort_ladder": ["medium", "low"], "retain_functions": True, "keep_recent": 6,
       "trim_drain": 6000, "degenerate_guard": True, "timeout_as_cut": True, "snippet_timeout": 30.0}
run_eval.main(cfg)
rep = json.loads((out / "eval_report.json").read_text())
recs = [json.loads(l) for l in open(out / "attempts.jsonl", encoding="utf-8")]
# request cap: with 7000 tokens per turn and a 20000 cap, the 3rd request may ask for at most 6000
caps_b = [m for t, turn, m in mock.log if t == B and turn == 2]
assert caps_b and max(caps_b) <= 20000 - 2 * 7000, caps_b
assert all(m <= 8192 for _, _, m in mock.log)
results["request_cap"] = f"PASS (B turn-3 caps {sorted(set(caps_b))})"
# phased attempts: A verified on attempt 0 -> no attempt 1 for A; B and C get attempt 1 after attempt 0 ended
by_task = {}
for r in recs:
    by_task.setdefault(r["task_id"], []).append(r["idx"])
assert sorted(by_task[A]) == [0] and sorted(by_task[B]) == [0, 1] and sorted(by_task[C]) == [0, 1], by_task
results["phased_attempts"] = f"PASS {by_task}"
# invalid-code feedback reached the model and the attempt did not crash
b0 = next(r for r in recs if r["task_id"] == B and r["idx"] == 0)
assert b0["status"] != "crashed" and any("no valid 'code' string" in (x.get("output") or "") for x in b0["transcript"])
results["invalid_code_feedback"] = "PASS"
# official metric = test outputs; A has 1 output, B 1, C 1 -> 1/3
nq = sum(len(sol[t]) for t in (A, B, C))
assert rep["queries"] == nq and abs(rep["official_score_pct"] - 100.0 * rep["query_hits"] / nq) < 1e-6
assert (out / "progress.json").exists() and json.loads((out / "progress.json").read_text())["attempts_done"] == len(recs)
results["official_metric_progress"] = f"PASS official {rep['official_score_pct']}% task-macro {rep['task_macro_score_pct']}%"
s_sub = {"t1": [{"attempt_1": [[1]], "attempt_2": [[0]]}], "t2": [{"attempt_1": [[0]], "attempt_2": [[0]]}] * 4}
assert agent.score(s_sub, {"t1": [[[1]]], "t2": [[[2]], [[3]], [[4]], [[5]]]})[1] == 50.0   # macro (kept, labelled)

# ---- 5. last-cell validator
c14 = cells[14]
src = c14[c14.index("    def ok_grid"):c14.index("    bad = [")]
ns = {}
exec("\n".join(l[4:] for l in src.splitlines()), ns)
ok = ns["ok_grid"]
assert ok([[0]]) and ok([[9] * 30] * 30)
assert not ok([]) and not ok([[1], [1, 2]]) and not ok([[1]] * 31) and not ok([[True]]) and not ok([[1.0]])
assert "os.replace(" in c14
results["validator"] = "PASS"


# ---- 6. tool smoke test against a fake OpenAI server on 127.0.0.1:1234
def fake_server(mode):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            choice = body.get("tool_choice")
            name = choice["function"]["name"] if isinstance(choice, dict) else "python"
            msg = {"content": "", "tool_calls": []}
            if mode == "ok" or (mode == "no_forced" and not isinstance(choice, dict)):
                code = "def transform(grid):\n    return grid" if name == "submit" else "print(train_inputs[0].shape)"
                msg["tool_calls"] = [{"id": "t1", "type": "function",
                                      "function": {"name": name, "arguments": json.dumps({"code": code})}}]
            elif mode == "no_forced":
                msg["content"] = "sure"
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
for mode, expect in (("ok", (True, True, True)), ("no_forced", (True, False, True)), ("none", (False, False, False))):
    srv = fake_server(mode)
    try:
        sm = submit_main.smoke({"retain_functions": True, "template_kwargs": {"reasoning_effort": "high"}})
    finally:
        srv.shutdown()
        srv.server_close()
    assert (sm["ok"], sm["forced_ok"], sm["auto_ok"]) == expect, (mode, sm)
results["tool_smoke"] = "PASS (ok / forced-broken / none)"

# ---- 7. server ladder: deadline-bounded, smoke-gated, forced submits off when the forced test fails
calls = {"start": [], "kill": 0}


class FakeProc:
    pid = 999999


submit_main.install = lambda cfg: None
submit_main.ensure_libcuda = lambda: {"probe": "SKIP"}
submit_main.kill = lambda proc: calls.__setitem__("kill", calls["kill"] + 1)
smokes = iter([{"ok": False, "forced_ok": False, "thinking_ok": False},
               {"ok": True, "forced_ok": False, "thinking_ok": True}])
submit_main.smoke = lambda cfg: next(smokes)
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
base = {"server_ladder": [["--a"], ["--b"], []], "eval": True, "force_submit_after": 5, "min_agent_s": 1800}
submit_main.main(dict(base, deadline_epoch=time.time() + 4 * 3600))
srep = json.loads((work / "agent_server_report.json").read_text())
assert len(calls["start"]) == 2 and calls["kill"] >= 1 and srep["chosen"]["index"] == 1, (calls, srep)
assert seen_cfg.get("force_submit_after") == 0 and seen_cfg.get("thinking", True) is True
assert all(t <= 1500 for _, t in calls["start"])
calls["start"].clear()
seen_cfg.clear()
submit_main.main(dict(base, deadline_epoch=time.time() + 1800 + 200))     # < 300 s beyond the agent reserve
assert calls["start"] == [] and not seen_cfg, calls
results["server_ladder"] = "PASS (smoke-gated rung 1 chosen, forced off; no rung when the reserve is reached)"

print(json.dumps(results, indent=1))
print("V1.1 LOCAL TESTS PASS")
