"""Local CPU test of Q27-AGENT v1's agent package + run_eval with a scripted mock client (no GPU, no server).

Extracts every %%writefile cell into a temp agent/ folder, then runs run_eval.main on 3 real evaluation tasks:
  task A: python call, then a submit that reproduces every example (lookup program) -> verified, correct
  task B: wrong submits (identity) and one cut turn -> never verified, scored from the best unverified program
  task C: the model never calls a tool (plain text) -> hint path, turn budget
usage: python test_agent_local.py <notebook> <competition_dir>
"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

NB, COMP = sys.argv[1], sys.argv[2]
work = Path(tempfile.mkdtemp(prefix="q27a_"))
agent_dir = work / "agent"
agent_dir.mkdir()
for c in json.load(open(NB, encoding="utf-8"))["cells"]:
    s = "".join(c["source"])
    if s.startswith("%%writefile /kaggle/working/agent/"):
        name = s.splitlines()[0].split("/")[-1]
        (agent_dir / name).write_text("\n".join(s.splitlines()[1:]) + "\n", encoding="utf-8")
print("agent files:", sorted(p.name for p in agent_dir.iterdir()))
sys.path.insert(0, str(agent_dir))
import run_eval  # noqa: E402

ch = json.load(open(Path(COMP) / "arc-agi_evaluation_challenges.json"))
sol = json.load(open(Path(COMP) / "arc-agi_evaluation_solutions.json"))
A, B, C = "136b0064", "247ef758", "4c3d4a41"


def lookup_program(tid):
    pairs = [(p["input"], p["output"]) for p in ch[tid]["train"]] + \
            [(t["input"], s) for t, s in zip(ch[tid]["test"], sol[tid])]
    return ("PAIRS = " + json.dumps(pairs) + "\n"
            "def transform(grid):\n"
            "    for i, o in PAIRS:\n"
            "        if i == grid:\n"
            "            return o\n"
            "    return grid\n")


class MockClient:
    """Same interface as client.ChatClient.chat; decides by task (found in the user prompt) and turn."""
    def __init__(self, *a, **k):
        self.calls = 0

    def chat(self, messages, tools=None, max_tokens=None, seed=None, timeout=None, tool_choice=None,
             template_kwargs=None):
        self.calls += 1
        prompt = messages[1]["content"]
        turn = sum(m["role"] == "assistant" for m in messages)
        tid = None                     # identify the task by its first train input row text
        for t in (A, B, C):
            row = "".join(str(v) for v in ch[t]["train"][0]["input"][0])
            if row in prompt and len(ch[t]["train"][0]["input"][0]) == len(row):
                tid = t
        base = {"reasoning": "thinking...", "prompt_tokens": 1000, "completion_tokens": 500, "finish_reason": "stop"}
        call = lambda name, code: dict(base, content="", tool_calls=[{"id": f"c{self.calls}", "name": name,
                                                                       "arguments": {"code": code}}])
        if tid == A:
            if turn == 0:
                return call("python", "print([g.shape for g in train_inputs])")
            return call("submit", lookup_program(A))
        if tid == B:
            if turn == 1:
                return dict(base, content="", tool_calls=[], finish_reason="length", completion_tokens=12288)
            return call("submit", "def transform(grid):\n    return grid\n")
        return dict(base, content="I think the rule is a reflection.", tool_calls=[])


run_eval.ChatClient = MockClient
out = work / "agent_eval"
cfg = {"base_url": "http://mock", "model": "mock", "data_dir": COMP, "eval_task_ids": [A, B, C, "nonexistent"],
       "eval_out": str(out), "deadline_epoch": time.time() + 600, "attempts": 2, "concurrency": 6,
       "max_turns": 8, "eval_cap": 60000, "context_budget": 28000, "nudge_every": 6, "turn_max_tokens": 12288,
       "in_python": True, "force_submit_after": 10, "effort_ladder": ["medium", "low"], "retain_functions": True,
       "keep_recent": 6, "trim_drain": 6000, "degenerate_guard": True, "timeout_as_cut": True, "snippet_timeout": 30.0}
run_eval.main(cfg)
rep = json.loads((out / "eval_report.json").read_text())
print(json.dumps({k: rep[k] for k in ("score_tasks", "query_hits", "queries", "tasks_with_verified_program",
                                      "status_counts", "notes", "cuts_total", "submits_total")}, indent=1))
assert rep["n_tasks"] == 3, rep["n_tasks"]                      # unknown id dropped
assert rep["per_task"][A] == 1.0 and rep["per_task"][B] == 0.0 and rep["per_task"][C] == 0.0, rep["per_task"]
assert rep["tasks_with_verified_program"] == 1
assert rep["cuts_total"] >= 1 and rep["submits_total"] >= 2
assert (out / "attempts.jsonl").exists() and sum(1 for _ in open(out / "attempts.jsonl")) == 5   # v1.1: A is verified on attempt 0, so no attempt 1
print("LOCAL AGENT TEST PASS")
