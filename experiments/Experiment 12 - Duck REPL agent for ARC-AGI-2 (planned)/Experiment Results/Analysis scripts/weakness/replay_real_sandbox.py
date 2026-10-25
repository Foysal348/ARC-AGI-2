"""Faithful replay: feed a branch's recorded tool-call code, in order, through the REAL adapter + Duck sandbox
(Arc2ToolAgent._run_python_tool: same prelude, retention rules, sandbox). After every call, if the model's own
check() printed all pairs exact, run the adapter's clean host verifier _verify_submission(name) and compare its
test outputs with the truth. No model involved; ground truth only for scoring the analysis.
Usage: python replay_real_sandbox.py <transcript> <task_id> <challenges.json> <solutions.json>"""
import json, os, re, sys, tempfile
from pathlib import Path
sys.argv, ARGS = sys.argv[:1], sys.argv[1:]
exec(open(Path(__file__).with_name("smoke_adapter.py"), encoding="utf-8").read().split("flip = lambda g")[0])
tx_path, task_id, ch, so = ARGS
task = json.load(open(ch))[task_id]; truth = json.load(open(so))[task_id]
tx = open(tx_path, encoding="utf-8", errors="replace").read()
codes = re.findall(r"\[TOOL CALL: python\]\n<tool_call>\n<function=python>\n<parameter=code>\n(.*?)\n</parameter>", tx, re.S)
tmp = Path(tempfile.mkdtemp(prefix="replay_", dir=str(Path(__file__).with_name("runs"))))
state = tmp / TA.RUNTIME_STATE_FILENAME
write_runtime_state(state, current_frame=None, history=[])
ag = Agent(task=task, task_id=task_id, branch=0, model="qwen", base_url="http://127.0.0.1:9/v1", provider="vllm")
found = []
CHK = re.compile(r"check:\s*(\d+)\s*/\s*(\d+)\s*train pairs correct")
for i, code in enumerate(codes):
    code = code.replace("submit(", "check(")          # never let a replayed submit end the replay
    r = ag._run_python_tool(state, {"code": code})
    out = r.content
    m = [(int(a), int(b)) for a, b in CHK.findall(out)]
    if any(a == b for a, b in m):
        names = re.findall(r"check\(\s*([A-Za-z_]\w*)", code)
        for name in dict.fromkeys(names):
            v = ag._verify_submission(name)
            tc = [o == t for o, t in zip(v.get("test_outputs") or [], truth)] if v.get("accepted") else None
            found.append({"call": i + 1, "name": name, "host_accepted": v["accepted"], "train_exact": v.get("train_exact"), "test_correct": tc})
print(json.dumps({"task": task_id, "tool_calls": len(codes), "verified": found}))
