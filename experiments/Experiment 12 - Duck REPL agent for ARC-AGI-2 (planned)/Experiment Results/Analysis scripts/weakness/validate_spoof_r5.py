"""GPT rev 5 audit 2 (P0): reproduce the json.dumps spoof through the rev 5 CHECKED path and the submit path, then test
GPT's proposed _grid_call_code guard (monkeypatched, prototype). Real Duck sandbox, scripted code only."""
import json, os, sys
from pathlib import Path
os.environ["E12_ARC2_REV5"] = "1"
sys.argv = sys.argv[:1]
exec(open(Path(__file__).with_name("smoke_adapter.py"), encoding="utf-8").read().split("# 1 system prompt carries the task")[0])
SPOOF = ("def evil(g):\n    import json\n    old = json.dumps\n    want = [list(reversed(r)) for r in g]\n"
         "    def forged(payload, *a, **k):\n"
         "        if isinstance(payload, dict) and payload.get('type') == 'final':\n"
         "            payload = dict(payload); payload['result'] = want\n"
         "        return old(payload, *a, **k)\n"
         "    json.dumps = forged\n    return g\n")
def fresh():
    a = Agent(task=task, task_id="flip", branch=0, model="qwen", base_url="http://127.0.0.1:9/v1", provider="vllm")
    return a
def run_on(a, code):
    return a._run_python_tool(state, {"code": code})
def trial(label):
    a = fresh(); run_on(a, SPOOF + "action({'action': '" + A.CHECKED_PREFIX_ALIAS + "evil'})")
    c = a.arc2_checked[-1] if a.arc2_checked else None
    b = fresh(); run_on(b, SPOOF + "submit(evil)")
    s = b.arc2_submissions[-1]
    g = fresh(); run_on(g, "def good(g):\n    import json\n    k = json.dumps(g)\n    return [list(reversed(r)) for r in g]\ncheck(good)")
    gc = g.arc2_checked[-1] if g.arc2_checked else None
    w = fresh(); run_on(w, "def ident(g):\n    return g\nsubmit(ident)")
    print(f"{label}: forged CHECKED accepted={c and c['accepted']} | spoof submit accepted={s['accepted']} | "
          f"correct (uses json.dumps) kept={gc and gc['accepted']} | wrong identity accepted={w.arc2_submissions[-1]['accepted']}")
A.CHECKED_PREFIX_ALIAS = A.CHECK_PREFIX
trial("current rev 5")
_orig = A._grid_call_code
def guarded(name, grid):
    encoded = json.dumps(grid, separators=(",", ":"))
    return ("import json as _arc2_ser\n_arc2_original_dumps = _arc2_ser.dumps\n"
            f"_arc2_in = {encoded}\ntry:\n    result = {name}([list(r) for r in _arc2_in])\nfinally:\n"
            "    _arc2_dumps_changed = (_arc2_ser.dumps is not _arc2_original_dumps)\n"
            "    _arc2_ser.dumps = _arc2_original_dumps\n"
            "if _arc2_dumps_changed:\n    raise RuntimeError('Verification program modified sandbox serializer')\n")
A._grid_call_code = guarded
trial("with GPT guard")
