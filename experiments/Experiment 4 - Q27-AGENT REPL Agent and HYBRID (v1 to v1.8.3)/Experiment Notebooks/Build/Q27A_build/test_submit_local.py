"""Rerun path (run_submit, mode repl) with the mock client: submission.json must be valid and use the verified answer."""
import json, os, sys, tempfile, time
from pathlib import Path
sys.argv = [sys.argv[0]] + sys.argv[1:]
NB, COMP = sys.argv[1], sys.argv[2]
import importlib.util
spec = importlib.util.spec_from_file_location("t", str(Path(__file__).with_name("test_agent_local.py")))
# reuse the extraction + MockClient from the eval test without running its main part
src = Path(__file__).with_name("test_agent_local.py").read_text(encoding="utf-8").split("run_eval.ChatClient = MockClient")[0]
g = {"__name__": "x", "__file__": str(Path(__file__).with_name("test_agent_local.py"))}
exec(src, g)
import run_submit
run_submit.ChatClient = g["MockClient"]
ch, sol, A, B, C = g["ch"], g["sol"], g["A"], g["B"], g["C"]
data = Path(tempfile.mkdtemp(prefix="q27a_data_"))
(data / "arc-agi_test_challenges.json").write_text(json.dumps({t: ch[t] for t in (A, B, C)}))
sub_path = str(data / "submission.json")
os.environ["KAGGLE_IS_COMPETITION_RERUN"] = "1"
cfg = {"base_url": "http://mock", "model": "mock", "data_dir": str(data), "sub_path": sub_path, "mode": "repl",
       "deadline_epoch": time.time() + 600, "attempts": 2, "concurrency": 6, "max_turns": 8,
       "tok_per_s": 300, "min_cap": 15000, "max_cap": 60000, "context_budget": 28000, "nudge_every": 6,
       "turn_max_tokens": 12288, "in_python": True, "force_submit_after": 10, "effort_ladder": ["medium", "low"],
       "retain_functions": True, "keep_recent": 6, "trim_drain": 6000, "degenerate_guard": True,
       "timeout_as_cut": True, "snippet_timeout": 30.0}
run_submit.main(cfg)
sub = json.load(open(sub_path))
assert set(sub) == {A, B, C}
assert all(len(sub[t]) == len(ch[t]["test"]) for t in sub)
assert sub[A][0]["attempt_1"] == sol[A][0], "verified answer must be attempt_1"
assert sub[C][0]["attempt_1"] == ch[C]["test"][0]["input"], "no program -> fallback test input"
print("RERUN PATH TEST PASS")
