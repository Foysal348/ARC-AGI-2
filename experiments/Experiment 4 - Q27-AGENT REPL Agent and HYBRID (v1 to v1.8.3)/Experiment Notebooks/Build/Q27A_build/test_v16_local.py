"""Q27-AGENT v1.6 local tests (CPU, no server): audit-hook isolation, nonportable submit names, two-tier caps,
circuit breaker, DIRECT rescue merge, run_eval HYBRID report, run_submit HYBRID path, launcher config.
usage: python test_v16_local.py <notebook> <dir with arc-agi_evaluation_{challenges,solutions}.json>"""
import importlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

NB, COMP = sys.argv[1], sys.argv[2]
work = Path(tempfile.mkdtemp(prefix="q27a16_"))
agent_dir = work / "agent"
agent_dir.mkdir()
cells = ["".join(c["source"]) for c in json.load(open(NB, encoding="utf-8"))["cells"]]
for s in cells:
    if s.startswith("%%writefile /kaggle/working/agent/"):
        (agent_dir / s.splitlines()[0].split("/")[-1]).write_text("\n".join(s.splitlines()[1:]) + "\n",
                                                                  encoding="utf-8")
assert (agent_dir / "rescue.py").exists()
sys.path.insert(0, str(agent_dir))
results = {}

# ---------------------------------------------------------------- 1. audit-hook isolation (sandbox + python tool)
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
}
for name, code in cases.items():
    r = sandbox.run_one(code, [[[1]]], timeout=30)
    txt = json.dumps(r)
    assert ("access denied" in txt or "not allowed" in txt) and r[0].get("output") is None, (name, txt[:300])
ok_code = (f"import numpy as np, json\ndef transform(grid):\n    v = json.load(open({P(allowed)}))[0][0]\n"
           f"    return (np.rot90(np.array(grid), 2) * 0 + v).tolist()\n")
r = sandbox.run_one(ok_code, [[[1, 2], [3, 4]]], timeout=30)
assert r[0].get("output") == [[7, 7], [7, 7]], r          # numpy + an allowed file still work
task = {"train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}], "test": [{"input": [[5, 6], [7, 8]]}]}
o = repl.run_snippet(f"print(open({P(sentinel)}).read())", task, timeout=30)
assert "access denied" in o and "[[9]]" not in o, o
o = repl.run_snippet(f"import os\nprint(os.listdir({P(fake_in)}))", task, timeout=30)
assert "access denied" in o and "data.json" not in o, o
o = repl.run_snippet("print(train_inputs[0].shape, int(train_outputs[0].sum()))", task, timeout=30)
assert "(2, 2) 10" in o, o
results["audit_hook"] = f"PASS ({len(cases)} escape routes denied in the sandbox, 2 in the python tool; numpy + allowed file OK)"

# ---------------------------------------------------------------- 2. nonportable names rejected with feedback
import repl_agent             # noqa: E402
n_ok, tests, fb = repl_agent.verify("def transform(grid):\n    return train_outputs[0].tolist()\n", task)
assert n_ok == 0 and tests == [None] and "train_outputs" in fb, (n_ok, tests, fb)
n_ok, tests, fb = repl_agent.verify("import numpy as np\ndef transform(grid):\n    return np.rot90(np.array(grid), 2).tolist()\n", task)
assert n_ok == 1 and tests == [[[8, 7], [6, 5]]], (n_ok, tests, fb)
n_ok, _, _ = repl_agent.verify("def transform(grid):\n    train_inputs = 3\n    return grid\n", task)
assert n_ok == 0                 # conservative: any use of the name, even a local assignment, is rejected
st = repl.new_store()
o = repl.run_snippet("import numpy as np\ndef helper(g):\n    return np.array(g)[::-1].tolist()\n"
                     "def transform(grid):\n    return train_outputs[0].tolist()\nprint('ok')", task, timeout=30, store=st)
assert "helper" in st["defs"] and "transform" not in st["defs"] and "not kept: transform" in o, (o, st["defs"].keys())
o = repl.run_snippet("def transform(grid):\n    return helper(grid)\nprint('ok2')", task, timeout=30, store=st)
assert "transform" in st["defs"], o                       # a portable redefinition is kept
o = repl.run_snippet("def transform(grid):\n    return test_inputs[0].tolist()\nprint('ok3')", task, timeout=30, store=st)
assert "transform" not in st["defs"] and "helper" in st["defs"], o   # nonportable redefinition drops the old copy
full, added = repl.with_kept("def solve(g):\n    return helper(g)\ndef transform(grid):\n    return solve(grid)\n", st)
n_ok, _, fb = repl_agent.verify(full, {"train": [{"input": [[1], [2]], "output": [[2], [1]]}], "test": [{"input": [[3], [4]]}]})
assert n_ok == 1 and added == ["helper"], (n_ok, added, fb)   # kept helper still works in submit
results["nonportable_submit"] = "PASS (verify rejects; tool-only functions never kept; portable helpers still kept + usable)"

# ---------------------------------------------------------------- 3. attempt_caps: scout 24k, deep 36k
seen = []
_real_run_attempt = repl_agent.run_attempt


def fake_run_attempt(att, task, client, shared, deadline, stop_after=1, seed=0, **over):
    seen.append((att.task_id, att.idx, over.get("max_gen_tokens")))
    att.status, att.note = "failed", "fake"


repl_agent.run_attempt = fake_run_attempt
repl_agent.solve_repl({"t1": task, "t2": task}, None, attempts=2, concurrency=2, deadline=time.time() + 60,
                      max_gen_tokens=60000, attempt_caps=[24000, 36000], log=lambda m: None)
caps = sorted(seen)
assert caps == [("t1", 0, 24000), ("t1", 1, 36000), ("t2", 0, 24000), ("t2", 1, 36000)], caps
seen.clear()
repl_agent.solve_repl({"t1": task}, None, attempts=1, concurrency=1, deadline=time.time() + 60,
                      max_gen_tokens=50000, log=lambda m: None)
assert seen == [("t1", 0, 50000)], seen                    # no attempt_caps -> the supplied cap
repl_agent.run_attempt = _real_run_attempt
results["attempt_caps"] = "PASS [24000, 36000] by attempt index; absent -> max_gen_tokens"

# ---------------------------------------------------------------- 4. DIRECT rescue merge (unit)
import rescue                 # noqa: E402
V, V2, U, D, I = [[1]], [[2]], [[3]], [[4]], [[0]]
M = lambda ranked, d: (lambda m: [m["attempt_1"], m["attempt_2"]])(rescue.merge(ranked, d, I))   # noqa: E731
assert M([(V, "verified x2"), (U, "unverified 2/3")], D) == [V, D]
assert M([(U, "unverified 2/3")], D) == [D, U]
assert M([], D) == [D, I]
assert M([], None) == [I, I]
assert M([(V, "verified x1"), (V2, "verified x1")], D) == [V, V2]
assert M([(U, "unverified 1/3")], U) == [U, I]
assert M([(U, "unverified 1/3"), (V2, "unverified 0/3")], None) == [U, V2]
results["merge_unit"] = "PASS 7 cases (verified > direct > unverified > input, no duplicates)"

# run_rescue: deadline skip, crash isolation, on_result callback
calls = []


class DummyClient:
    max_tokens = 36864
    budget_dropped = False


rescue.direct_client = lambda cfg: DummyClient()


def fake_answer(client, task, seed, deadline, max_tokens=None, rescue_tokens=None):
    calls.append(task["_id"])
    if task["_id"] == "boom":
        raise RuntimeError("model went away")
    return {"status": "answered", "grids": task["_direct"], "gen_tokens": 100}


rescue.answer = fake_answer
tk = {"a": dict(task, _id="a", _direct=[[[5]]]), "boom": dict(task, _id="boom", _direct=None)}
got_cb = []
recs, _ = rescue.run_rescue(["a", "boom"], tk, {"concurrency": 2}, time.time() + 3600, log=lambda m: None,
                            on_result=lambda t, r: got_cb.append(t))
assert recs["a"]["grids"] == [[[5]]] and recs["boom"]["status"].startswith("crashed") and sorted(got_cb) == ["a", "boom"]
calls.clear()
recs, _ = rescue.run_rescue(["a"], tk, {"rescue_min_start_s": 300}, time.time() + 10, log=lambda m: None)
assert recs["a"]["status"] == "skipped: deadline" and calls == []
c = importlib.reload(rescue).direct_client({"base_url": "http://x/v1", "model": "qwen", "rescue_think": 32768,
                                            "rescue_answer": 4096, "thinking_token_budget": 6144,
                                            "rescue_effort": "xhigh"})
assert c.max_tokens == 36864 and c.thinking_token_budget == 32768 and c.template_kwargs == {"reasoning_effort": "xhigh"}
rescue.direct_client = lambda cfg: DummyClient()
rescue.answer = fake_answer
results["run_rescue"] = "PASS (deadline skip, crash isolated, callback, client = xhigh 32768+4096)"

# ---------------------------------------------------------------- 5. run_eval HYBRID report (agent stage faked)
ch = json.load(open(Path(COMP) / "arc-agi_evaluation_challenges.json"))
sol = json.load(open(Path(COMP) / "arc-agi_evaluation_solutions.json"))
A, B, C = "136b0064", "247ef758", "4c3d4a41"
for t in (A, B, C):
    assert t in ch, t
wrong = lambda t, j: [[(sol[t][j][0][0] + 1) % 10] + sol[t][j][0][1:]] + sol[t][j][1:]   # noqa: E731
import run_eval               # noqa: E402

captured = {}


def att_rec(tid, status):
    return {"task_id": tid, "idx": 0, "status": status, "note": "", "turns": 3, "python_calls": 1, "submits": 1,
            "cuts": 0, "forced": 0, "gen_tokens": 1000, "prompt_tokens": 500, "wall": 1.0, "best_ok": 1,
            "n_train": len(ch[tid]["train"]), "effort_steps": 0, "evictions": 0, "transcript": [], "programs": []}


def fake_solve_eval(tasks, client, deadline=None, on_done=None, **kw):
    captured["deadline"] = deadline
    answers = {A: [[(sol[A][j], "verified x1")] for j in range(len(sol[A]))],
               B: [[(wrong(B, j), "unverified 1/3")] for j in range(len(sol[B]))],
               C: [[] for _ in sol[C]]}
    report = {"attempts": [att_rec(A, "verified"), att_rec(B, "failed"), att_rec(C, "failed")]}
    for r in report["attempts"]:
        on_done(r)
    return answers, report


direct_by = {A: [wrong(A, j) for j in range(len(sol[A]))], B: [sol[B][j] for j in range(len(sol[B]))], C: None}


def fake_answer2(client, task, seed, deadline, max_tokens=None, rescue_tokens=None):
    tid = next(t for t in (A, B, C) if task["test"] == ch[t]["test"])
    calls.append(tid)
    return {"status": "answered" if direct_by[tid] else "no grid", "grids": direct_by[tid], "gen_tokens": 10}


rescue.answer = fake_answer2
run_eval.solve_repl = fake_solve_eval
nA, nB, nC = len(sol[A]), len(sol[B]), len(sol[C])
for rescue_all in (True, False):
    calls.clear()
    out = work / f"eval_{rescue_all}"
    t_start = time.time()
    D_ = t_start + 1000
    run_eval.main({"base_url": "http://127.0.0.1:9/v1", "model": "qwen", "data_dir": COMP, "eval_out": str(out),
                   "eval_task_ids": [A, B, C], "deadline_epoch": D_, "rescue": True, "rescue_all": rescue_all,
                   "agent_frac": 0.6, "concurrency": 3})
    assert abs(captured["deadline"] - (t_start + 0.6 * (D_ - t_start))) < 3, (captured["deadline"], t_start)
    assert sorted(calls) == (sorted([A, B, C]) if rescue_all else sorted([B, C])), calls
    rep = json.loads((out / "eval_report.json").read_text())
    cm = rep["complementarity"]
    q = nA + nB + nC
    assert cm["queries"] == q and cm["agent_correct"] == nA and cm["hybrid_correct"] == nA + nB, cm
    assert cm["direct_correct"] == nB and cm["agent_only"] == nA and cm["direct_only"] == nB and cm["both"] == 0, cm
    assert cm["rescued_queries"] == (q if rescue_all else nB + nC), cm
    assert abs(rep["official_score_pct"] - 100.0 * (nA + nB) / q) < 0.01, rep["official_score_pct"]
    assert abs(rep["agent_only_official_pct"] - 100.0 * nA / q) < 0.01
    assert abs(rep["direct_only_official_pct"] - 100.0 * nB / q) < 0.01
    assert rep["circuit_breaker"]["flag"] is False
    es = json.loads((out / "eval_submission.json").read_text())
    assert es[A][0]["attempt_1"] == sol[A][0]                                     # verified beats direct
    assert es[B][0]["attempt_1"] == sol[B][0] and es[B][0]["attempt_2"] == wrong(B, 0)   # direct beats unverified
    assert es[C][0]["attempt_1"] == ch[C]["test"][0]["input"]                     # nothing -> test input
results["run_eval_hybrid"] = (f"PASS (agent deadline = 60 % of the time; agent {nA}/{q}, direct {nB}/{q}, "
                              f"hybrid {nA + nB}/{q}; rescue_all on/off)")

# no rescue: the v1.5 behaviour (agent until the final deadline, no direct calls)
calls.clear()
t_start = time.time()
run_eval.main({"base_url": "http://127.0.0.1:9/v1", "model": "qwen", "data_dir": COMP, "eval_out": str(work / "e0"),
               "eval_task_ids": [A, B, C], "deadline_epoch": t_start + 1000})
assert abs(captured["deadline"] - (t_start + 1000)) < 1e-6 and calls == []
results["run_eval_no_rescue"] = "PASS (agent keeps the whole time, no direct calls)"

# ---------------------------------------------------------------- 6. run_submit HYBRID (agent stage faked)
import run_submit             # noqa: E402

sdir = work / "submit"
sdir.mkdir()
(sdir / "arc-agi_test_challenges.json").write_text(json.dumps({t: ch[t] for t in (A, B, C)}))
os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "1"


def fake_solve_submit(tasks, client, deadline=None, on_done=None, **kw):
    captured["deadline"] = deadline
    for tid, n_ok_fn, outs in ((A, lambda t: len(tasks[t]["train"]), [sol[A][j] for j in range(nA)]),
                               (B, lambda t: 1, [wrong(B, j) for j in range(nB)])):
        r = att_rec(tid, "verified" if tid == A else "failed")
        r["programs"] = [{"n_ok": n_ok_fn(tid), "test": outs}]
        on_done(r)
    return {}, {"attempts": [dict(att_rec(A, "verified")), dict(att_rec(B, "failed"))]}


direct_by = {A: [wrong(A, j) for j in range(nA)], B: [sol[B][j] for j in range(nB)],
             C: [sol[C][j] for j in range(nC)]}


def fake_answer3(client, task, seed, deadline, max_tokens=None, rescue_tokens=None):
    tid = next(t for t in (A, B, C) if task["test"] == ch[t]["test"])
    calls.append(tid)
    return {"status": "answered", "grids": direct_by[tid], "gen_tokens": 10}


rescue.answer = fake_answer3
run_submit.solve_repl = fake_solve_submit
calls.clear()
t_start = time.time()
sub_path = str(sdir / "submission.json")
run_submit.main({"base_url": "http://127.0.0.1:9/v1", "model": "qwen", "data_dir": str(sdir), "sub_path": sub_path,
                 "deadline_epoch": t_start + 2000, "mode": "repl", "rescue": True, "agent_frac": 0.5,
                 "concurrency": 3})
assert abs(captured["deadline"] - (t_start + 0.5 * 2000)) < 3, captured["deadline"]
assert sorted(calls) == sorted([B, C]), calls                 # the verified task is never re-asked
sub = json.load(open(sub_path))
assert sub[A][0]["attempt_1"] == sol[A][0]
assert sub[B][0] == {"attempt_1": sol[B][0], "attempt_2": wrong(B, 0)}, sub[B][0]
assert sub[C][0] == {"attempt_1": sol[C][0], "attempt_2": ch[C]["test"][0]["input"]}, sub[C][0]
results["run_submit_hybrid"] = "PASS (agent until 50 %, rescue only unverified tasks, merged submission.json)"

# ---------------------------------------------------------------- 7. circuit breaker (fresh module state)
ra = importlib.reload(repl_agent)
http = {"gen_tokens": 0, "note": "HTTP 400: Unexpected reasoning effort high"}
for _ in range(5):
    ra.breaker(http)
ra.breaker({"gen_tokens": 900, "note": ""})
ra.breaker(http)
assert ra.ABORT["flag"] is False                           # one healthy attempt among the first 6 -> keep going
ra = importlib.reload(repl_agent)
for _ in range(6):
    ra.breaker(http)
assert ra.ABORT["flag"] is True and "HTTP 400" in ra.ABORT["why"]


class NoCallClient:
    max_tokens = 100

    def chat(self, *a, **k):
        raise AssertionError("the client must not be called after the breaker tripped")


_, rep = ra.solve_repl({"t1": task}, NoCallClient(), attempts=2, concurrency=1, deadline=time.time() + 60,
                       max_gen_tokens=1000, max_turns=5, log=lambda m: None)
st = [(a["status"], a["note"][:8]) for a in rep["attempts"]]
assert all(s == ("stopped", "aborted:") for s in st), st
results["circuit_breaker"] = "PASS (5 HTTP + 1 healthy -> no trip; 6 HTTP -> trip; no requests afterwards)"

# ---------------------------------------------------------------- 8. launcher config + smoke wiring
c12 = next(s for s in cells if s.startswith("# ==================================================================== launch"))
AG = json.loads(c12.split('AGENT = json.loads(r"""', 1)[1].split('""")', 1)[0])
assert AG["attempt_caps"] == [24000, 36000] and AG["rescue"] is True and AG["agent_frac"] == 0.5
assert AG["rescue_effort"] == "xhigh" and AG["rescue_think"] == 32768 and AG["template_kwargs"] == {"reasoning_effort": "xhigh"}
assert 'os.environ.get("Q27A_EVAL_HOURS", 2.0)' in c12 and "rescue_all=True" in c12 and '"Q27A_AGENT_FRAC", 0.6' in c12
sm = (agent_dir / "submit_main.py").read_text()
assert "rescue_params_ok" in sm and "from rescue import direct_client" in sm
assert "RESCUE" not in [s.splitlines()[0] for s in cells]
results["launcher_config"] = "PASS"

print(json.dumps(results, indent=1))
print("V1.6 LOCAL TESTS PASS")
