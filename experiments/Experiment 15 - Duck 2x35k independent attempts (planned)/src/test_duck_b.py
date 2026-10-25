"""Unit tests for the E15B Duck package (Experiment 15 Notes/02). Run: python test_duck_b.py
The package is DUCK_B_PKG (an extracted package dir) or, locally, assembled from the Exp 12 vendor tree + src/duck_b."""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
E12_DUCK = Path(r"D:/ARC AGI 2/experiments/Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)/duck_arc2")
TASKS = Path(os.environ.get("E15B_TASKS", r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/"
                            r"E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_challenges.json"))   # the CPU kit sets it


def _package() -> Path:
    if os.environ.get("DUCK_B_PKG"):
        return Path(os.environ["DUCK_B_PKG"])
    d = Path(tempfile.mkdtemp(prefix="duck_b_"))
    shutil.copytree(E12_DUCK / "vendor", d / "vendor", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy(HERE / "duck_b" / "arc2_duck.py", d / "arc2_duck.py")
    for n in ("tool_agent.py", "python_tool_sandbox.py"):
        shutil.copy(HERE / "duck_b" / n, d / "vendor" / "inference" / "agent" / n)
    return d


PKG = _package()
sys.path[:0] = [str(PKG), str(PKG / "vendor")]
os.environ.update({"E12_ARC2_REV6": "1", "E12_ARC2_REV5": "1", "E12_B_MODE": "1", "E12_B_ATTEMPT_TOKENS": "35000",
                   "E12_B_RESERVE": "4096", "E12_DUCK_CONTEXT_WINDOW": "100000"})
import arc2_duck as A                                    # noqa: E402
os.environ.update(A.duck_env("http://127.0.0.1:9/v1", "fake-model", 16))
TA, DP, RUN, WRS = A._import_duck()
FAILS, SKIPS = [], []
if os.name == "nt":
    # the E14/E15 harness's Windows shim: Windows cannot pass the ~65 KB sandbox bootstrap as `-c`, so it is written to a
    # file and run from there (no other change); killpg does not exist on Windows
    import subprocess as _sp
    _real = _sp.Popen
    _bdir = Path(tempfile.gettempdir()) / "duck_b_bootstrap"
    _bdir.mkdir(parents=True, exist_ok=True)

    class _Shim(_real):
        def __init__(self, args, *a, **kw):
            if isinstance(args, list) and len(args) >= 5 and args[1:4] == ["-I", "-S", "-c"]:
                f = _bdir / "sandbox_bootstrap.py"
                if not f.exists() or f.read_text(encoding="utf-8") != args[4]:
                    f.write_text(args[4], encoding="utf-8")
                args = [args[0], "-I", "-S", str(f)] + args[5:]
            super().__init__(args, *a, **kw)
    _sp.Popen = _Shim
    if not hasattr(os, "killpg"):
        import signal as _sig
        os.killpg = lambda pid, sig: os.kill(pid, 9)
        if not hasattr(_sig, "SIGKILL"):
            _sig.SIGKILL = 9
# the sandbox runs here if a probe call works (Linux, or Windows with the shim); otherwise its checks are SKIPPED and
# must pass on the Kaggle CPU kit
_probe = RUN(code="print(7)", timeout_seconds=20, initial_state={"current_frame": None, "history": [], "valid_actions": [],
                                                                 "last_action_call_result": {}},
             action_handler=lambda a, stale_after=None: {"action_result": {}, "state": {}}, kept_functions=[], retain_imports=True)
LINUX = (_probe.get("stdout") or "").strip() == "7"
print("sandbox available:", LINUX, "" if LINUX else _probe, flush=True)


def check(name, cond, detail="", sandbox=False):
    if sandbox and not LINUX:
        print("SKIP " + name + "  [needs the Linux sandbox: Kaggle CPU kit]", flush=True)
        SKIPS.append(name)
        return
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  [{detail}]"), flush=True)
    if not cond:
        FAILS.append(name)


def _task():
    if TASKS.is_file():
        d = json.loads(TASKS.read_text())
        return d[sorted(d)[0]]
    return {"train": [{"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]}] * 2, "test": [{"input": [[1, 1], [0, 0]]}]}


TASK = _task()
STATE0 = {"current_frame": None, "history": [], "valid_actions": [], "last_action_call_result": {}}


def _sandbox(code):
    return RUN(code=code, timeout_seconds=20, initial_state=STATE0,
               action_handler=lambda a, stale_after=None: {"action_result": {}, "state": {}}, kept_functions=[],
               retain_imports=True)


# ---------------------------------------------------------------- 1. sandbox (#14)
r = _sandbox("import sys\nsys.setrecursionlimit(5000)\nprint(sys.getrecursionlimit(), sys.maxsize > 0)")
check("sandbox: sys shim setrecursionlimit", (r.get("stdout") or "").strip() == "5000 True", r, sandbox=True)
r = _sandbox("import sys\nsys.setrecursionlimit(10**9)\nprint(sys.getrecursionlimit())")
check("sandbox: setrecursionlimit is capped at 20000", (r.get("stdout") or "").strip() == "20000", r, sandbox=True)
r = _sandbox("import sys\nprint(hasattr(sys, 'modules'), hasattr(sys, 'exit'))")
check("sandbox: sys shim exposes no modules / exit", (r.get("stdout") or "").strip() == "False False", r, sandbox=True)
r = _sandbox("import time\nt = time.time()\nprint(t > 0)")
check("sandbox: time allowed", (r.get("stdout") or "").strip() == "True", r, sandbox=True)
r = _sandbox("import numpy")
check("sandbox: numpy message", "pure Python lists" in (r.get("error") or ""), r, sandbox=True)
r = _sandbox("def inspect_pair(i):\n    return i\nimport inspect_pair")
check("sandbox: helper-import message", "already defined" in (r.get("error") or ""), r, sandbox=True)
r = _sandbox("import os")
check("sandbox: os still blocked", "not allowed" in (r.get("error") or ""), r, sandbox=True)
os.environ["ARC2_SANDBOX_SYS_SHIM"] = "0"
r = _sandbox("import sys")
check("sandbox: flag off -> sys blocked as before", "not allowed" in (r.get("error") or ""), r, sandbox=True)
os.environ["ARC2_SANDBOX_SYS_SHIM"] = "1"

# ---------------------------------------------------------------- 2. regex (#13)
R = A._autocheck_re()
for n, want in (("trans", True), ("trans2", True), ("trans_v", True), ("candidate_rule", True), ("transform", True),
                ("solve_task", True), ("rule_alt", True), ("transpose", False), ("translate", False), ("helper", False)):
    check(f"regex: {n} -> {want}", bool(R.match(n)) == want)
check("regex: trans(grid) is one-grid", A._r6_single_grid_arg("def trans(grid):\n    return grid"))
check("regex: trans(x, y) is not one-grid", not A._r6_single_grid_arg("def trans(x, y):\n    return x"))

# ---------------------------------------------------------------- 3. reasoning-code harvest (#50)
txt = ("I think the rule is a flip.\n```python\ndef helper(g):\n    return g[::-1]\n```\nThen:\n"
       "def transform(grid):\n    return helper(grid)\n\nThat should work.\nprint(transform(x))\n")
src, names = A.b_harvest_source([txt])
check("harvest: fenced + unfenced blocks", "def helper" in src and "def transform" in src, src)
check("harvest: names = [transform]", names == ["transform"], names)
check("harvest: no top-level calls kept", "print(" not in src, src)
src2, names2 = A.b_harvest_source(["def transform(a, b):\n    return a\n"])
check("harvest: two-arg transform not a candidate", names2 == [], names2)
src3, names3 = A.b_harvest_source(["no code here"])
check("harvest: no code", src3 == "" and names3 == [])

# ---------------------------------------------------------------- 4. prompt edits (#12, #13, #14, #22, #48)
p = A.arc2_system_prompt_r6(TASK, DP, 3072, view="facts")
check("prompt: tool contract", "Python functions inside it, not tools" in p)
check("prompt: bounded search", "Bound any search" in p)
check("prompt: budget 35,000", "about 35,000 generated tokens" in p)
check("prompt: autocheck names", "`candidate...`, `trans`" in p)
check("prompt: imports incl. time and sys subset", "time, and `sys` (only setrecursionlimit" in p)
os.environ["E12_B_MODE"] = "0"
p0 = A.arc2_system_prompt_r6(TASK, DP, 3072, view="facts")
check("prompt: B off -> unchanged rev 6 prompt", "Python functions inside it, not tools" not in p0)
os.environ["E12_B_MODE"] = "1"
for old, _new in A.B_PROMPT_EDITS:
    check("prompt edit matches once: " + old[:40], p0.count(old) == 1, p0.count(old))

# ---------------------------------------------------------------- 5. caps / budget / commit (#4, #5, #48)
Agent = A.make_agent_class()
ag = Agent(task=TASK, task_id="t0", branch=0, view="facts", model="fake-model", base_url="http://127.0.0.1:9/v1",
           provider="vllm")
check("agent: B mode on", ag._arc2_b and ag._arc2_b_budget == 35000 and ag._arc2_b_reserve == 4096)
for gen, want_cap in ((0, 30904), (20000, 10904), (30000, 5000), (30905, 4095), (34900, 256)):
    ag._session_generated_tokens = gen
    try:
        cur = ag.generated_tokens
    except Exception:
        cur = None
    if cur != gen:                                   # generated_tokens is derived; set what it reads
        for attr in ("_generated_tokens", "_session_generated_tokens"):
            setattr(ag, attr, gen)
    cap = ag._arc2_apply_cap()
    check(f"cap at {gen}: {want_cap}", cap == want_cap and ag._reply_reserve_tokens == cap
          and ag._context_budget_tokens == 100000 - cap - ag._request_safety_margin_tokens,
          (cap, ag._context_budget_tokens, ag.generated_tokens))
for attr in ("_generated_tokens", "_session_generated_tokens"):
    setattr(ag, attr, 31000)
check("commit zone at 31,000 (left 4,000 <= 5,120)", ag._arc2_b_in_commit())
ag._resume_after_yield, ag._resume_reason = True, "yield_bare"
op = ag._arc2_r6_opener()
check("commit opener text", "almost spent" in op and "4,000" in op, op)
check("commit event recorded", ag.arc2_b_events["commit_at_tokens"] == 31000 and ag.arc2_b_events["commit_via"] == "opener")
ag2 = Agent(task=TASK, task_id="t0", branch=1, view="facts", model="fake-model", base_url="http://127.0.0.1:9/v1",
            provider="vllm")
ag2._resume_after_yield = False
check("first opener carries the budget", "35,000 generated tokens" in ag2._arc2_r6_opener())

# ---------------------------------------------------------------- 6. submit as a tool (#11)
d = Path(tempfile.mkdtemp(prefix="duck_b_state_"))
sp = d / TA.RUNTIME_STATE_FILENAME
WRS(sp, current_frame=None, history=[])
ag3 = Agent(task=TASK, task_id="t1", branch=0, view="facts", model="fake-model", base_url="http://127.0.0.1:9/v1",
            provider="vllm")
res = ag3._dispatch_tool(sp, "check", {"transform": "transform"})
body = json.loads(res.content)
check("check-as-tool routed to python", "not a tool" in body.get("harness_note_tool", "") and ag3.arc2_b_events["submit_tool_routed"] == 1, body)
res = ag3._dispatch_tool(sp, "python", {"code": "def transform(grid):\n    return [row[:] for row in grid]\nprint('ok')"})
check("python tool still works", "ok" in json.loads(res.content).get("stdout", ""), res.content[:300], sandbox=True)
check("defined_functions inventory", "transform" in json.loads(res.content).get("defined_functions", []), res.content[:400], sandbox=True)
res = ag3._dispatch_tool(sp, "submit", {"code": "submit(transform)"})
check("submit-as-tool routed (code arg)", ag3.arc2_b_events["submit_tool_routed"] == 2 and "submission" in json.loads(res.content),
      res.content[:300], sandbox=True)
res = ag3._dispatch_tool(sp, "fly", {})
check("other unknown tools unchanged", "Unknown tool: fly" in res.content)

# ---------------------------------------------------------------- 7. turn rollback fix (#10): a 2-call reply past the yield
def _fake_reply(n_calls=2, completion=2600):
    calls = [{"id": f"c{i}", "type": "function",
              "function": {"name": "python", "arguments": json.dumps({"code": f"v{i} = {i}\nprint('call {i}')"})}}
             for i in range(n_calls)]
    return TA._ChatCompletionResult(message={"role": "assistant", "content": "", "reasoning_content": "think " * 50,
                                             "tool_calls": calls}, finish_reason="tool_calls",
                                    usage={"prompt_tokens": 5000, "completion_tokens": completion,
                                           "total_tokens": 5000 + completion})


def _run_turn(flag: str):
    os.environ["ARC3_YIELD_AFTER_BATCH"] = flag
    dd = Path(tempfile.mkdtemp(prefix="duck_b_turn_"))
    st = dd / TA.RUNTIME_STATE_FILENAME
    WRS(st, current_frame=None, history=[])
    a = Agent(task=TASK, task_id="t2", branch=0, view="facts", model="fake-model", base_url="http://127.0.0.1:9/v1",
              provider="vllm")
    a._arc2_deadline = time.monotonic() + 120
    replies = iter([_fake_reply()])
    orig = TA.ToolAgent._chat_completion
    TA.ToolAgent._chat_completion = lambda self, messages, **kw: next(replies)
    try:
        a.analyze(st, 0, valid_actions=[], step_env=None, transcript_path=dd / "transcript.txt", analysis_step=1,
                  request_timeout_seconds=60, should_stop=lambda: False)
    finally:
        TA.ToolAgent._chat_completion = orig
    hist = a._history_messages or []
    tools = [m for m in hist if isinstance(m, dict) and m.get("role") == "tool"]
    return len(tools), (dd / "transcript.txt").read_text(encoding="utf-8", errors="replace") if (dd / "transcript.txt").exists() else ""


n_on, tx_on = _run_turn("1")
n_off, tx_off = _run_turn("0")
check("rollback fix ON: both tool results kept in history", n_on == 2, (n_on, tx_on[-600:]))
check("rollback fix OFF: upstream rollback (history discarded)", n_off == 0, (n_off, tx_off[-600:]))
check("rollback fix ON: second call executed", "call 1" in tx_on, tx_on[-400:])

# ---------------------------------------------------------------- 8. rule library (#20)
tk = {"train": [{"input": [[1, 2], [3, 4]], "output": [[3, 1], [4, 2]]}, {"input": [[5, 6, 7]], "output": [[5], [6], [7]]}],
      "test": [{"input": [[1, 0], [0, 0]]}]}
outs = A.rule_library_outputs(tk)
check("rules: rot90 found", [[[0, 1], [0, 0]]] in outs, outs)
tk2 = {"train": [{"input": [[1, 2]], "output": [[2, 3]]}, {"input": [[2, 1]], "output": [[3, 2]]}], "test": [{"input": [[1, 1]]}]}
check("rules: color map found", [[[2, 2]]] in A.rule_library_outputs(tk2), A.rule_library_outputs(tk2))
tk3 = {"train": [{"input": [[1, 2]], "output": [[9, 9, 9]]}], "test": [{"input": [[1]]}]}
check("rules: none when nothing fits", A.rule_library_outputs(tk3) == [])
sel = A.select_attempts_r5(tk, [])
check("selection: a rule fills attempt_1 when there is no candidate", sel[0]["attempt_1"] == [[0, 1], [0, 0]], sel)
cand = {"accepted": True, "test_outputs": [[[7]]], "t": 1, "origin": "check"}
sel = A.select_attempts_r5(tk, [{"checked": [cand], "submissions": []}])
check("selection: program first, rule second", sel[0]["attempt_1"] == [[7]] and sel[0]["attempt_2"] == [[0, 1], [0, 0]], sel)
part = {"accepted": False, "test_outputs": [[[8]]], "train_exact": [True, False], "t": 2, "origin": "partial_test"}
sel = A.select_attempts_r5(tk, [{"checked": [cand, part], "submissions": []}])
check("selection: partial program before rules", sel[0]["attempt_2"] == [[8]], sel)
os.environ["E12_B_MODE"] = "0"
sel = A.select_attempts_r5(tk, [])
check("selection: B off -> no rules (test input)", sel[0]["attempt_1"] == [[1, 0], [0, 0]], sel)
os.environ["E12_B_MODE"] = "1"

# ---------------------------------------------------------------- 9. candidate fields (#31) and partial test (#7)
ag5 = Agent(task=TASK, task_id="t5", branch=0, view="facts", model="fake-model", base_url="http://127.0.0.1:9/v1",
            provider="vllm")
cf = ag5.arc2_b_candidate_fields()
check("candidate fields: keys", set(cf) >= {"candidate_source_written", "candidate_retained", "check_attempted",
                                            "check_executed", "check_timeout", "train_pairs_exact", "host_verified"}, cf)
check("partial test: skipped without progress", ag5.arc2_b_partial_test().get("skipped") == "no partial program")

# ---------------------------------------------------------------- 10. server-down wait (#15)
import threading as _th
check("SERVER_DOWN event exists and is clear", isinstance(A.SERVER_DOWN, type(_th.Event())) and not A.SERVER_DOWN.is_set())

# ---------------------------------------------------------------- 11. deadline grace (#54)
ag6 = Agent(task=TASK, task_id="t6", branch=0, view="facts", model="fake-model", base_url="http://127.0.0.1:9/v1",
            provider="vllm")
ag6._arc2_deadline = time.monotonic() - 1.0
check("harvest skipped past the deadline without grace", "left" in str(ag6.arc2_b_reasoning_harvest(sp).get("skipped")))
ag6._arc2_verify_grace = 30.0
r6 = ag6.arc2_b_reasoning_harvest(sp)
check("harvest proceeds within the grace", "left" not in str(r6.get("skipped")), r6)

# ---------------------------------------------------------------- 12. Notes/07 A6: a rule-library error keeps the programs
_task6 = {"train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}], "test": [{"input": [[5, 6], [7, 8]]}]}
_rec6 = [{"branch": 0, "submissions": [{"accepted": True, "t": 1, "origin": "submit", "test_outputs": [[[9, 9], [9, 9]]]}],
          "checked": []}]
_orig_cm = A._rl_color_map
A._rl_color_map = lambda task: (_ for _ in ()).throw(ValueError("injected"))
try:
    _sel6 = A.select_attempts_r5(_task6, _rec6)
finally:
    A._rl_color_map = _orig_cm
check("A6: rule-library error -> the program candidate still selected", _sel6[0]["attempt_1"] == [[9, 9], [9, 9]], _sel6)
_sel6b = A.select_attempts_r5(_task6, [])
check("A6: rules still used without a program (rotate 180)", _sel6b[0]["attempt_1"] == [[8, 7], [6, 5]], _sel6b)
_dsrc = Path(A.__file__).read_text(encoding="utf-8")
check("A4: kv_stale_checks counted and reported", 'state["kv_stale_checks"] += 1' in _dsrc
      and 'kv_stale_checks=state["kv_stale_checks"]' in _dsrc)

# ---------------------------------------------------------------- 13. Auditor 1 (2026-10-25) regressions
def _new_agent13(task, tid):
    d = Path(tempfile.mkdtemp(prefix="duck_b_a1_"))
    sp_ = d / TA.RUNTIME_STATE_FILENAME
    WRS(sp_, current_frame=None, history=[])
    a = Agent(task=task, task_id=tid, branch=0, view="facts", model="fake-model", base_url="http://127.0.0.1:9/v1",
              provider="vllm")
    a._arc2_deadline = time.monotonic() + 300
    return a, sp_


# F1: the harvest must not replace the checked partial program (the partial test then runs the draft)
_t13 = {"train": [{"input": [[1, 2, 3]], "output": [[3, 2, 1]]}, {"input": [[4, 5], [6, 7]], "output": [[5, 4], [6, 7]]}],
        "test": [{"input": [[8, 9]]}]}
a13, sp13 = _new_agent13(_t13, "a1f1")
a13._dispatch_tool(sp13, "python", {"code": "def transform(grid):\n    return [r[::-1] for r in grid]\ncheck(transform)"})
a13._history_messages = [{"role": "assistant", "reasoning_content": "First idea:\ndef transform(grid):\n    return [[0]]\n",
                          "content": None}]
a13._arc2_verify_grace = 30.0
h13 = a13.arc2_b_reasoning_harvest(sp13)
p13 = a13.arc2_b_partial_test()
a13._arc2_verify_grace = 0.0
check("Auditor1 F1: harvest restores the checked program; partial test runs it",
      "[r[::-1] for r in grid]" in (a13._kept_functions.get("transform") or "")
      and a13.arc2_checked and a13.arc2_checked[-1]["test_outputs"] == [[[9, 8]]], (h13, p13, a13._kept_functions), sandbox=True)
_dsrc13 = Path(A.__file__).read_text(encoding="utf-8")
_i_pt = _dsrc13.find('agent.arc2_b_events["partial_test"] = agent.arc2_b_partial_test()')
_i_hv = _dsrc13.find('agent.arc2_b_events["reasoning_harvest"] = agent.arc2_b_reasoning_harvest(self.state_path)')
check("rev3k ORD1: the run tail runs the harvest before the partial test (F1 safety kept by P1 + the kept0 restore)",
      0 < _i_hv < _i_pt, (_i_pt, _i_hv))
# F2: a rejected submit inside the commit zone -> the commit text in the tool result and in the next opener
a14, sp14 = _new_agent13({"train": [{"input": [[1, 2]], "output": [[2, 1]]}], "test": [{"input": [[3, 4]]}]}, "a1f2")
a14._ensure_session(sp14)
a14._session_generated_tokens = 31000
r14 = a14._dispatch_tool(sp14, "python", {"code": "def transform(grid):\n    return grid\nsubmit(transform)"})
a14._resume_after_yield = False
op14 = a14._arc2_r6_opener()
check("Auditor1 F2: commit text after a rejected submit (tool result + next opener)",
      a14._arc2_b_in_commit() and "almost spent" in r14.content and "almost spent" in op14 and "not accepted" in op14,
      (r14.content[:300], op14[:300]), sandbox=True)
# the harvest's own tool call never records a commit turn
a15, sp15 = _new_agent13({"train": [{"input": [[1, 2]], "output": [[2, 1]]}], "test": [{"input": [[3, 4]]}]}, "a1f2b")
a15._ensure_session(sp15)
a15._session_generated_tokens = 34990
a15._history_messages = [{"role": "assistant", "reasoning_content": "def transform(grid):\n    return grid\n", "content": None}]
a15._arc2_verify_grace = 30.0
h15 = a15.arc2_b_reasoning_harvest(sp15)
check("Auditor1 F2: the harvest does not mark a commit turn", h15.get("ran") and a15.arc2_b_events.get("commit_at_tokens") is None,
      (h15, a15.arc2_b_events.get("commit_at_tokens")), sandbox=True)
# F3: a tool call after the deadline runs at most ~1 s (deadline + grace clamp)
check("Auditor1 F3: _run_python_tool clamps the sandbox timeout to the deadline + grace",
      "_tmo = max(1, min(self._python_timeout, int(_left) - 1))" in _dsrc13)

# ---------------------------------------------------------------- 14. auditor 3 R9 / L41, auditor 2 F1 (rule cache)
src16, names16 = A.b_harvest_source(["```python\nimport numpy as np\ndef transform(grid):\n    return [r[::-1] for r in grid]\n```"])
check("Auditor3 R9: harvest imports wrapped in try", "try:\n    import numpy as np\nexcept Exception:\n    pass" in src16
      and names16 == ["transform"], src16)
a16, sp16 = _new_agent13({"train": [{"input": [[1, 2]], "output": [[2, 1]]}], "test": [{"input": [[3, 4]]}]}, "a3r9")
a16._history_messages = [{"role": "assistant", "reasoning_content": "```python\nimport numpy as np\ndef transform(grid):\n"
                          "    return [r[::-1] for r in grid]\n```", "content": None}]
a16._arc2_verify_grace = 30.0
h16 = a16.arc2_b_reasoning_harvest(sp16)
check("Auditor3 R9: a blocked import no longer aborts the harvest (accepted)", h16.get("new_accepted") == 1, h16, sandbox=True)
a17, sp17 = _new_agent13({"train": [{"input": [[1, 2]], "output": [[2, 1]]}], "test": [{"input": [[3, 4]]}]}, "a3l41")
a17._ensure_session(sp17)
r17 = a17._dispatch_tool(sp17, "function_calls", {"code": "def transform(grid):\n    return [r[::-1] for r in grid]\ncheck(transform)"})
check("Auditor3 L41: an unknown tool with a code argument runs as python",
      "not a tool" in r17.content and any(v.get("accepted") for v in a17.arc2_checked), r17.content[:300], sandbox=True)
r17b = a17._dispatch_tool(sp17, "browser", {"url": "x"})
check("Auditor3 L41: an unknown tool without code stays an error", "Unknown tool" in r17b.content, r17b.content[:200])
_task18 = {"train": [{"input": [[1, 2], [3, 4]], "output": [[4, 3], [2, 1]]}], "test": [{"input": [[5, 6], [7, 8]]}]}
_s18a = A.select_attempts_r5(_task18, [])
_c18 = A._RULE_CACHE.get(id(_task18))
_s18b = A.select_attempts_r5(_task18, [])
check("Auditor2 F1: rule outputs cached per task (same result)", _c18 is not None and _c18[0] is _task18 and _s18a == _s18b,
      (_c18, _s18a, _s18b))

# ---------------------------------------------------------------- 15. GPT final audit: session-setup failures never leak
_BAD0 = {"t00", "t03", "t07"}            # attempt 1 setup fails (repeatedly: 3 tasks)
_BAD1 = {"t05"}                          # attempt 2 setup fails
_Real = A.BranchSession


class _FailBS(_Real):
    def __init__(self, agent_cls, **kw):
        if (kw["branch"] == 0 and kw["task_id"] in _BAD0) or (kw["branch"] == 1 and kw["task_id"] in _BAD1):
            raise OSError("injected session init failure")
        super().__init__(agent_cls, **kw)

    def run(self, pause_at_tokens=None, tag=None):        # no server: an attempt that ends at once
        time.sleep(0.05)
        self.status = "done"
        self.record.setdefault("runs", []).append({"status": "done", "tokens": 0, "pause_at": pause_at_tokens,
                                                   "stage": (tag or {}).get("stage")})


_tasks15 = {f"t{i:02d}": {"train": [{"input": [[i % 10]], "output": [[i % 10]]}], "test": [{"input": [[1]]}]} for i in range(20)}
A.BranchSession = _FailBS
_t15 = time.monotonic()
try:
    _rec15 = A.solve_all_r6(_tasks15, sorted(_tasks15), out_dir=Path(tempfile.mkdtemp(prefix="e15b_sched_")),
                            deadline=time.monotonic() + 120, base_url="http://127.0.0.1:9/v1", model_id="fake", branches=2,
                            workers=16, first_exposure_tokens=None, b1_view="facts", min_admit_sec=5, fair_share=False,
                            max_generated_tokens=35000)
finally:
    A.BranchSession = _Real
_dt15, _S15 = time.monotonic() - _t15, dict(A.SCHED_LAST)
_notes15 = {(r["task_id"], r["branch"]): r.get("note", "") for v in _rec15.values() for r in v}
check("GPT blocker: setup failures counted (4), no worker error, no straggler, stage 1 complete, heap empty",
      _S15.get("setup_failed") == 4 and not _S15.get("worker_errors") and not _S15.get("stragglers")
      and _S15.get("stage1_left") == 0 and _S15.get("heap_left") == 0, {k: _S15.get(k) for k in
      ("setup_failed", "worker_errors", "stragglers", "stage1_left", "heap_left")})
check("GPT blocker: every other attempt ran (36 sessions done) and the run ended at once, not at the deadline",
      sum(1 for v in _S15.get("sessions", {}).values() if v == "done") == 36 and _dt15 < 60, (_dt15, _S15.get("sessions")))
check("GPT blocker: failed attempts recorded once; attempt 2 still ran for the tasks whose attempt 1 failed",
      all("session setup" in _notes15.get((t, 0), "") for t in _BAD0) and "session setup" in _notes15.get(("t05", 1), "")
      and all(_S15["sessions"].get(f"{t}_b1") == "done" for t in _BAD0)
      and sum(1 for k in _notes15 if "session setup" in _notes15[k]) == 4, _notes15)
_sub15 = A.build_submission(_tasks15, _rec15)
check("GPT blocker: the submission is complete and valid", set(_sub15) == set(_tasks15)
      and all(len(_sub15[t]) == 1 and {"attempt_1", "attempt_2"} <= set(_sub15[t][0]) for t in _tasks15), list(_sub15)[:3])

# ---------------------------------------------------------------- 16. rev 3f: per-attempt budgets (45k + 30k)
os.environ.update(E12_B_ATTEMPT_TOKENS="45000", E12_B_ATTEMPT_TOKENS_B1="30000")
try:
    _g0 = Agent(task=TASK, task_id="r3f0", branch=0, view="facts", model="fake-model", base_url="http://127.0.0.1:9/v1",
                provider="vllm")
    _g1 = Agent(task=TASK, task_id="r3f1", branch=1, view="facts", model="fake-model", base_url="http://127.0.0.1:9/v1",
                provider="vllm")
    check("rev3f: attempt budgets 45,000 / 30,000 (agent)", _g0._arc2_b_budget == 45000 and _g1._arc2_b_budget == 30000,
          (_g0._arc2_b_budget, _g1._arc2_b_budget))
    check("rev3f: the system prompt states each attempt's own budget",
          "about 45,000 generated tokens" in _g0._system_prompt and "about 30,000 generated tokens" in _g1._system_prompt)
    check("rev3f: attempt 2's first cap = 30,000 - 4,096", _g1._arc2_apply_cap() == 30000 - 4096, _g1._arc2_apply_cap())
finally:
    os.environ.update(E12_B_ATTEMPT_TOKENS="35000")
    os.environ.pop("E12_B_ATTEMPT_TOKENS_B1", None)
check("rev3f: unset attempt-2 budget = attempt 1's", A.b_attempt_budget(1) == A.b_attempt_budget(0) == 35000)
_caps16 = []
A.BranchSession = type("_CapBS", (_FailBS,), {"__init__": lambda self, agent_cls, **kw: (_caps16.append(
    (kw["branch"], kw.get("max_generated_tokens"))), _Real.__init__(self, agent_cls, **kw))[-1]})
try:
    A.solve_all_r6({"u0": _tasks15["t01"]}, ["u0"], out_dir=Path(tempfile.mkdtemp(prefix="e15b_cap_")),
                   deadline=time.monotonic() + 60, base_url="http://127.0.0.1:9/v1", model_id="fake", branches=2, workers=2,
                   first_exposure_tokens=None, b1_view="facts", min_admit_sec=5, fair_share=False,
                   max_generated_tokens=45000, max_generated_tokens_b1=30000)
finally:
    A.BranchSession = _Real
check("rev3f: session caps 45,000 (attempt 1) / 30,000 (attempt 2)", sorted(_caps16) == [(0, 45000), (1, 30000)], _caps16)

# ---------------------------------------------------------------- 17. rev 3h: code of a tool call cut at the output limit
_cut_raw = '{"code": "def transform(grid):\\n    return [r[::-1] for r in grid]\\n\\ndef helper(x):\\n    return [x\\n\\nprint(check(transf'
check("rev3h: cut-call code decoded from a JSON prefix", A.b_cut_call_code(_cut_raw).startswith("def transform(grid):\n    return [r[::-1]"),
      A.b_cut_call_code(_cut_raw))
check("rev3h: cut-call code from a dict / a full JSON string / nothing",
      A.b_cut_call_code({"code": "x = 1"}) == "x = 1" and A.b_cut_call_code('{"code": "y = 2"}') == "y = 2"
      and A.b_cut_call_code("") == "" and A.b_cut_call_code('{"other": 1') == "")
check("rev3h: a cut escape at the end is dropped", A.b_cut_call_code('{"code": "a = 1\\nb = 2\\') == "a = 1\nb = 2")
_pref = A.b_complete_prefix(A.b_cut_call_code(_cut_raw))
check("rev3h: complete leading definitions only (the cut helper and call dropped)",
      "def transform" in _pref and "def helper" not in _pref and "check(" not in _pref, _pref)
_tcut = {"train": [{"input": [[1, 2, 3]], "output": [[3, 2, 1]]}, {"input": [[4, 5]], "output": [[5, 4]]}],
         "test": [{"input": [[6, 7]]}]}
a18, sp18 = _new_agent13(_tcut, "r3h")
a18._ensure_session(sp18)
_cut_reply = TA._ChatCompletionResult(message={"role": "assistant", "content": "", "reasoning_content": "think " * 20,
                                               "tool_calls": [{"id": "k1", "type": "function", "function": {
                                                   "name": "python", "arguments": _cut_raw}}]},
                                      finish_reason="length", usage={"prompt_tokens": 900, "completion_tokens": 30000,
                                                                     "total_tokens": 30900})
_orig18 = TA.ToolAgent._chat_completion
TA.ToolAgent._chat_completion = lambda self, messages, **kw: _cut_reply
try:
    _r18 = a18._chat_completion([{"role": "user", "content": "x"}], max_tokens=30904, request_timeout_seconds=60)
finally:
    TA.ToolAgent._chat_completion = _orig18
check("rev3h: the cut call is still neutralized (never run) and its code kept",
      "<cut off" in _r18.message["tool_calls"][0]["function"]["arguments"] and len(a18._arc2_b_cut_code) == 1
      and a18.arc2_b_events.get("cut_code_saved") == 1, (a18._arc2_b_cut_code, _r18.message["tool_calls"]))
a18._arc2_verify_grace = 30.0
h18 = a18.arc2_b_reasoning_harvest(sp18)
check("rev3h: the end-of-attempt harvest verifies the cut code (host-accepted)",
      h18.get("cut_code_blocks") == 1 and h18.get("new_accepted") == 1 and h18.get("names") == ["transform"], h18,
      sandbox=True)

# ---------------------------------------------------------------- 18. rev 3h: the nudge follows the attempt budget
os.environ.update(E12_B_ATTEMPT_TOKENS="45000", E12_B_ATTEMPT_TOKENS_B1="30000")
try:
    _n0 = Agent(task=TASK, task_id="n0", branch=0, view="facts", model="fake-model", base_url="http://127.0.0.1:9/v1",
                provider="vllm")
    _n1 = Agent(task=TASK, task_id="n1", branch=1, view="facts", model="fake-model", base_url="http://127.0.0.1:9/v1",
                provider="vllm")
    check("rev3h: nudge 24,000 for a 45k attempt, 20,700 for a 30k attempt (before its 25,904 commit zone)",
          _n0._arc2_nudge_tokens == 24000 and _n1._arc2_nudge_tokens == 20700, (_n0._arc2_nudge_tokens, _n1._arc2_nudge_tokens))
finally:
    os.environ.update(E12_B_ATTEMPT_TOKENS="35000")
    os.environ.pop("E12_B_ATTEMPT_TOKENS_B1", None)

# ---------------------------------------------------------------- 19. rev 3i: R + stage 3 (user 2026-10-25, Notes/12 C-D)
_ACC0 = {f"u{i:02d}" for i in range(0, 12, 2)}          # attempt 1 accepts on even tasks
_ACC2 = {"u03", "u07"}                                  # these two accept only at attempt 3 (branch 2)
_LOG19 = []


class _BS19(_Real):
    def run(self, pause_at_tokens=None, tag=None):
        _LOG19.append((self.task_id, self.branch, time.monotonic(), (tag or {}).get("stage")))
        time.sleep(0.15 if self.branch == 0 else 0.05)
        self.status = "done"
        self.record.setdefault("runs", []).append({"status": "done", "tokens": 0, "stage": (tag or {}).get("stage")})

    def verified_signatures(self):
        ok = (self.branch == 0 and self.task_id in _ACC0) or (self.branch == 2 and self.task_id in _ACC2)
        return {f"sig-{self.task_id}"} if (ok and self.status == "done") else set()

    def has_progress(self):
        return False


_tasks19 = {f"u{i:02d}": {"train": [{"input": [[i % 10]], "output": [[i % 10]]}], "test": [{"input": [[1]]}]} for i in range(12)}
A.BranchSession = _BS19
try:
    _rec19 = A.solve_all_r6(_tasks19, sorted(_tasks19), out_dir=Path(tempfile.mkdtemp(prefix="e15b_r3i_")),
                            deadline=time.monotonic() + 60, base_url="http://127.0.0.1:9/v1", model_id="fake", branches=2,
                            workers=4, first_exposure_tokens=None, b1_view="facts", min_admit_sec=1, fair_share=False,
                            max_generated_tokens=50000, max_generated_tokens_b1=45000,
                            skip_b1_if_accepted=True, extra_attempts=True, max_attempts=4)
finally:
    A.BranchSession = _Real
_S19 = dict(A.SCHED_LAST)
_br19 = {}
for t_, b_, _, _ in _LOG19:
    _br19.setdefault(t_, []).append(b_)
check("rev3i R: a task whose attempt 1 accepted gets no attempt 2 (6 skipped), and no stage-3 attempt",
      _S19.get("b1_skipped") == 6 and all(_br19[t] == [0] for t in _ACC0), (_S19.get("b1_skipped"), _br19))
check("rev3i stage 3: an unsolved task gets attempts 3, 4 up to max_attempts; a task that accepts at attempt 3 stops",
      all(sorted(_br19[t]) == [0, 1, 2] for t in _ACC2)
      and all(sorted(_br19[t]) == [0, 1, 2, 3] for t in _tasks19 if t not in _ACC0 and t not in _ACC2), _br19)
_t_last_b0_end = max(t0 for t_, b_, t0, _ in _LOG19 if b_ == 0)       # rev 3k S1: the last attempt-1 START
_t_first_b2 = min(t0 for t_, b_, t0, _ in _LOG19 if b_ >= 2)
check("rev3i/3k stage 3: no attempt 3 starts before every attempt 1 has started; stage tag 3",
      _t_first_b2 >= _t_last_b0_end - 0.02 and all(st_ == 3 for t_, b_, _, st_ in _LOG19 if b_ >= 2),
      (_t_first_b2, _t_last_b0_end))
_ord19 = [(b_, t_) for t_, b_, _, _ in _LOG19 if b_ >= 2]
check("rev3i stage 3: round robin (every attempt 3 before any attempt 4)",
      max(i for i, (b_, _) in enumerate(_ord19) if b_ == 2) < min(i for i, (b_, _) in enumerate(_ord19) if b_ == 3), _ord19)
check("rev3i: counters (extra_started 10 = 6 attempt 3s + 4 attempt 4s; by attempt), no worker error, heap empty",
      _S19.get("extra_started") == 10 and _S19.get("extra_by_attempt") == {"3": 6, "4": 4} and not _S19.get("worker_errors")
      and _S19.get("heap_left") == 0, {k: _S19.get(k) for k in ("extra_started", "extra_by_attempt", "worker_errors", "heap_left")})
_sub19 = A.build_submission(_tasks19, _rec19)
check("rev3i: records for attempts 3+ exist and the submission is complete",
      any(r.get("branch") == 3 for v in _rec19.values() for r in v) and set(_sub19) == set(_tasks19))
# the defaults keep rev 3h behaviour: both attempts, no stage 3
_LOG19.clear()
A.BranchSession = _BS19
try:
    A.solve_all_r6(_tasks19, sorted(_tasks19), out_dir=Path(tempfile.mkdtemp(prefix="e15b_r3h_")),
                   deadline=time.monotonic() + 60, base_url="http://127.0.0.1:9/v1", model_id="fake", branches=2,
                   workers=4, first_exposure_tokens=None, b1_view="facts", min_admit_sec=1, fair_share=False,
                   max_generated_tokens=50000, max_generated_tokens_b1=45000)
finally:
    A.BranchSession = _Real
check("rev3i defaults = rev 3h: every task gets attempts 1 and 2 only",
      sorted((t_, b_) for t_, b_, _, _ in _LOG19) == sorted((t, b) for t in _tasks19 for b in (0, 1))
      and A.SCHED_LAST.get("b1_skipped") == 0 and A.SCHED_LAST.get("extra_started") == 0)

# ---------------------------------------------------------------- 20. rev 3j: order of the accepted tasks' attempt 2
def _run20(mode):
    _LOG19.clear()
    A.BranchSession = _BS19
    try:
        A.solve_all_r6(_tasks19, sorted(_tasks19), out_dir=Path(tempfile.mkdtemp(prefix="e15b_r3j_")),
                       deadline=time.monotonic() + 60, base_url="http://127.0.0.1:9/v1", model_id="fake", branches=2,
                       workers=1, first_exposure_tokens=None, b1_view="facts", min_admit_sec=1, fair_share=False,
                       max_generated_tokens=50000, max_generated_tokens_b1=45000,
                       extra_attempts=True, max_attempts=4, b1_accepted_mode=mode)
    finally:
        A.BranchSession = _Real
    return [(t_, b_) for t_, b_, _, _ in _LOG19], dict(A.SCHED_LAST)


def _phase20(seq):
    out = []
    for t_, b_ in seq:
        ph = ("a1" if b_ == 0 else ("a2acc" if (b_ == 1 and t_ in _ACC0) else "a2" if b_ == 1 else f"a{b_ + 1}"))
        if not out or out[-1] != ph:
            out.append(ph)
    return out


_q1, _s1 = _run20("after3")
_q2, _s2 = _run20("before3")
check("rev3j notebook 1 (after3): attempt 1 -> attempt 2 (unsolved) -> attempt 3 (unsolved) -> attempt 2 (accepted) -> attempt 4",
      _phase20(_q1) == ["a1", "a2", "a3", "a2acc", "a4"], _phase20(_q1))
check("rev3j notebook 2 (before3): attempt 1 -> attempt 2 (unsolved) -> attempt 2 (accepted) -> attempt 3 -> attempt 4",
      _phase20(_q2) == ["a1", "a2", "a2acc", "a3", "a4"], _phase20(_q2))
check("rev3j: every task gets its attempt 2 in both orders; counters (deferred 6, extra 10)",
      all({0, 1} <= {b_ for t_, b_ in q if t_ == t} for q in (_q1, _q2) for t in _tasks19)
      and all(x.get("b1_deferred") == 6 and x.get("b1_deferred_started") == 6 and x.get("extra_started") == 10
              and x.get("b1_skipped") == 0 for x in (_s1, _s2)),
      [{k: x.get(k) for k in ("b1_deferred", "b1_deferred_started", "extra_started", "b1_skipped")} for x in (_s1, _s2)])

# ---------------------------------------------------------------- 21. rev 3k fixes (validation agents + GPT audits)
_LOG21 = []
_SLOW21 = "u05"                       # one long attempt 1 (an unsolved task)


class _BS21(_Real):
    def run(self, pause_at_tokens=None, tag=None):
        t0 = time.monotonic()
        time.sleep(1.2 if (self.branch == 0 and self.task_id == _SLOW21) else 0.05)
        _LOG21.append((self.task_id, self.branch, t0, time.monotonic(), (tag or {}).get("stage")))
        self.status = "done"
        self.record.setdefault("runs", []).append({"status": "done", "tokens": 0, "stage": (tag or {}).get("stage")})

    def verified_signatures(self):
        return {f"sig-{self.task_id}"} if (self.branch == 0 and self.task_id in _ACC0 and self.status == "done") else set()

    def has_progress(self):
        return False


A.BranchSession = _BS21
try:
    A.solve_all_r6(_tasks19, sorted(_tasks19), out_dir=Path(tempfile.mkdtemp(prefix="e15b_r3k_")),
                   deadline=time.monotonic() + 60, base_url="http://127.0.0.1:9/v1", model_id="fake", branches=2,
                   workers=4, first_exposure_tokens=None, b1_view="facts", min_admit_sec=1, fair_share=False,
                   max_generated_tokens=50000, max_generated_tokens_b1=45000,
                   extra_attempts=True, max_attempts=3, b1_accepted_mode="before3")
finally:
    A.BranchSession = _Real
_slow_end = max(e for t_, b_, s0, e, _ in _LOG21 if t_ == _SLOW21 and b_ == 0)
_def_starts = [s0 for t_, b_, s0, e, _ in _LOG21 if b_ == 1 and t_ in _ACC0]
check("rev3k S1: deferred attempt 2s start while a long attempt 1 still runs (no idle behind it)",
      _def_starts and min(_def_starts) < _slow_end - 0.3, (min(_def_starts) if _def_starts else None, _slow_end))
check("rev3k S1: still no deferred / stage-3 item before every attempt 1 has started",
      min(s0 for t_, b_, s0, e, _ in _LOG21 if (b_ == 1 and t_ in _ACC0) or b_ >= 2)
      >= max(s0 for t_, b_, s0, e, _ in _LOG21 if b_ == 0) - 0.02)
check("rev3k S2: stage tags = attempt 1 -> 1, attempt 2 (normal or deferred) -> 2, attempts >= 3 -> 3",
      all(st_ == (1 if b_ == 0 else 2 if b_ == 1 else 3) for t_, b_, _, _, st_ in _LOG21),
      sorted({(b_, st_) for t_, b_, _, _, st_ in _LOG21}))
_big = "```python\ndef transform(grid):\n    x = '" + "1" * 26000 + "'\n    return grid\n```"
_small = "```python\ndef helper(g):\n    return g\n\ndef transform(g):\n    return helper(g)\n```"
_src_b, _nm_b = A.b_harvest_source([_big])
_src_s, _nm_s = A.b_harvest_source([_big, _small])
check("rev3k H1: a single definition over the bound is skipped (never cut); no name without its definition",
      _src_b == "" and _nm_b == [] and "def transform(g)" in _src_s and "def helper" in _src_s and _nm_s == ["transform"]
      and "1" * 100 not in _src_s, (len(_src_b), _nm_b, _nm_s, len(_src_s)))
_long_tail = "\\n".join(f"y{i} = {i}" for i in range(4000))
_cut_raw2 = '{"code": "def transform(grid):\\n    return [r[::-1] for r in grid]\\n\\n' + _long_tail + '\\nprint(check(trans'
a21, sp21 = _new_agent13(_tcut, "r3k")
a21._ensure_session(sp21)
_cut_reply2 = TA._ChatCompletionResult(message={"role": "assistant", "content": "", "reasoning_content": "think",
                                                "tool_calls": [{"id": "k2", "type": "function", "function": {
                                                    "name": "python", "arguments": _cut_raw2}}]},
                                       finish_reason="length", usage={"prompt_tokens": 900, "completion_tokens": 30000,
                                                                      "total_tokens": 30900})
_orig21 = TA.ToolAgent._chat_completion
TA.ToolAgent._chat_completion = lambda self, messages, **kw: _cut_reply2
try:
    a21._chat_completion([{"role": "user", "content": "x"}], max_tokens=30904, request_timeout_seconds=60)
finally:
    TA.ToolAgent._chat_completion = _orig21
check("rev3k H2: a long cut call keeps the START of its code (def transform), not the last 24k characters",
      a21._arc2_b_cut_code and a21._arc2_b_cut_code[0].startswith("def transform(grid):")
      and "def transform" in A.b_complete_prefix(a21._arc2_b_cut_code[0]), (a21._arc2_b_cut_code or [""])[0][:60])
a22, sp22 = _new_agent13(_tcut, "r3k2")
a22._ensure_session(sp22)
_BEST_SRC = "def transform(grid):\n    return [r[::-1] for r in grid]\n"
a22._kept_functions = {"transform": "def transform(grid):\n    return grid\n"}
a22.arc2_progress["best"] = {"name": "transform", "k": 1, "n": 2, "acc": 0.5, "src": "BESTKEY"}
a22._arc2_best_kept = {"transform": _BEST_SRC}
_seen22 = {}


def _fake_verify22(name):
    _seen22["src"] = a22._kept_functions.get(name)
    return {"name": name, "accepted": False, "train_exact": [True, False], "test_outputs": [[[7]]]}


a22._verify_submission = _fake_verify22
a22._arc2_closure_key = lambda n: "LATERKEY________________"
_r22 = a22.arc2_b_partial_test()
check("rev3k P1: the partial test runs the best-progress source (not a later redefinition); retained functions restored",
      _seen22.get("src") == _BEST_SRC and a22.arc2_checked[-1].get("best_source_restored") is True
      and a22._kept_functions == {"transform": "def transform(grid):\n    return grid\n"}, (_seen22, _r22))

# ---------------------------------------------------------------- 22. rev 3k part 2 (SEL1, TINY1)
_task22 = {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[0]]}]}
_r22a = {"branch": 0, "submissions": [{"accepted": True, "test_outputs": [[[1]]], "t": 100.0, "branch": 0,
                                       "train_exact": [True]}], "checked": []}
_r22b = {"branch": 1, "submissions": [{"accepted": True, "test_outputs": [[[2]]], "t": 200.0, "branch": 1,
                                       "train_exact": [True]}], "checked": []}
_sel22 = A.select_attempts_r5(_task22, [_r22a, _r22b])
_sel22r = A.select_attempts_r5(_task22, [_r22b, _r22a])
check("rev3k SEL1: a 1-1 vote tie between attempts puts attempt 1's grid first (both grids still in the two slots)",
      _sel22[0]["attempt_1"] == [[1]] and _sel22[0]["attempt_2"] == [[2]] and _sel22r == _sel22, (_sel22, _sel22r))
_r22c = {"branch": 2, "submissions": [{"accepted": True, "test_outputs": [[[2]]], "t": 300.0, "branch": 2,
                                       "train_exact": [True]}], "checked": []}
_sel22v = A.select_attempts_r5(_task22, [_r22a, _r22b, _r22c])
check("rev3k SEL1: votes still decide first (2 attempts agree on [[2]] -> attempt_1)", _sel22v[0]["attempt_1"] == [[2]]
      and _sel22v[0]["attempt_2"] == [[1]], _sel22v)
_dsrc22 = Path(A.__file__).read_text(encoding="utf-8")
check("rev3k TINY1: after the commit turn, < 768 tokens left ends the attempt (no tiny final request)",
      'getattr(agent, "_arc2_b_commit_sent", False)' in _dsrc22 and "agent._arc2_b_left() < 768" in _dsrc22
      and 'agent.arc2_b_events["tiny_request_skipped"] = 1' in _dsrc22)

# ---------------------------------------------------------------- 23. rev 3k W1: weak (commit-zone) accepts
_WEAK23 = {"u00", "u02"}                 # attempt 1 accepts only in the commit zone (weak) on these two
_LOG23 = []


class _BS23(_Real):
    def run(self, pause_at_tokens=None, tag=None):
        _LOG23.append((self.task_id, self.branch, time.monotonic(), (tag or {}).get("stage")))
        time.sleep(0.05)
        self.status = "done"
        self.record.setdefault("runs", []).append({"status": "done", "tokens": 0, "stage": (tag or {}).get("stage")})

    def verified_signatures(self, strong_only=False):
        if self.status != "done" or not (self.branch == 0 and self.task_id in _ACC0):
            return set()
        return set() if (strong_only and self.task_id in _WEAK23) else {f"sig-{self.task_id}"}

    def has_progress(self):
        return False


def _run23(weak):
    _LOG23.clear()
    A.BranchSession = _BS23
    try:
        A.solve_all_r6(_tasks19, sorted(_tasks19), out_dir=Path(tempfile.mkdtemp(prefix="e15b_w1_")),
                       deadline=time.monotonic() + 60, base_url="http://127.0.0.1:9/v1", model_id="fake", branches=2,
                       workers=1, first_exposure_tokens=None, b1_view="facts", min_admit_sec=1, fair_share=False,
                       max_generated_tokens=50000, max_generated_tokens_b1=45000,
                       extra_attempts=True, max_attempts=3, b1_accepted_mode="after3", weak_commit=weak)
    finally:
        A.BranchSession = _Real
    return [(t_, b_, st_) for t_, b_, _, st_ in _LOG23], dict(A.SCHED_LAST)


_q23on, _s23on = _run23(True)
_q23off, _s23off = _run23(False)
_pos = lambda q, t, b: next(i for i, (t_, b_, _) in enumerate(q) if t_ == t and b_ == b)
_first_a3_on = min(i for i, (t_, b_, _) in enumerate(_q23on) if b_ == 2)
check("rev3k W1 on: a weak-only accepted task's attempt 2 is a normal stage-2 item (before any attempt 3) and it gets stage 3",
      all(_pos(_q23on, t, 1) < _first_a3_on for t in _WEAK23) and all((t, 2) in {(t_, b_) for t_, b_, _ in _q23on} for t in _WEAK23)
      and _s23on.get("b1_deferred") == 4 and _s23on.get("weak_commit") is True, (_q23on, _s23on.get("b1_deferred")))
_first_a3_off = min(i for i, (t_, b_, _) in enumerate(_q23off) if b_ == 2)
check("rev3k W1 off: the same tasks count as solved (attempt 2 deferred after the attempt 3s, no stage 3)",
      all(_pos(_q23off, t, 1) > _first_a3_off for t in _WEAK23) and not any(b_ == 2 and t_ in _WEAK23 for t_, b_, _ in _q23off)
      and _s23off.get("b1_deferred") == 6, (_q23off, _s23off.get("b1_deferred")))
_task23 = {"train": [{"input": [[1]], "output": [[1]]}], "test": [{"input": [[0]]}]}
_rw0 = {"branch": 0, "submissions": [{"accepted": True, "weak": True, "test_outputs": [[[5]]], "t": 300.0, "branch": 0,
                                      "train_exact": [True]}], "checked": []}
_rw1 = {"branch": 1, "submissions": [{"accepted": True, "test_outputs": [[[6]]], "t": 100.0, "branch": 1,
                                      "train_exact": [True]}], "checked": []}
os.environ["E12_B_WEAK_COMMIT"] = "1"
try:
    _selw = A.select_attempts_r5(_task23, [_rw0, _rw1])
finally:
    os.environ.pop("E12_B_WEAK_COMMIT", None)
_selw0 = A.select_attempts_r5(_task23, [_rw0, _rw1])
check("rev3k W1 selection: with the flag an organic accept ranks before a weak one; without it, unchanged (attempt 1 first)",
      _selw[0]["attempt_1"] == [[6]] and _selw[0]["attempt_2"] == [[5]]
      and _selw0[0]["attempt_1"] == [[5]] and _selw0[0]["attempt_2"] == [[6]], (_selw, _selw0))
_dsrc23 = Path(A.__file__).read_text(encoding="utf-8")
check("rev3k W1: an accept after the commit text is marked weak (verifier)",
      'if getattr(self, "_arc2_b", False) and getattr(self, "_arc2_b_commit_sent", False):\n                    v["weak"] = True'
      in _dsrc23)

print(f"\n{len(FAILS)} FAIL, {len(SKIPS)} SKIP" if FAILS else f"\nALL PASS ({len(SKIPS)} SKIP)", flush=True)
sys.exit(1 if FAILS else 0)
