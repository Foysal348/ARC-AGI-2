"""GPT rev 5 audit 3: B6 bypass via Duck's raw-markup recovery. Mock a length-cut reply whose tool call is only in the
text (tool_calls == []), run the rev 5 _chat_completion, then Duck's own recovery logic (tool_agent.py:6842-6849)."""
import json, os, sys
from pathlib import Path
os.environ["E12_ARC2_REV5"] = "1"
sys.argv = sys.argv[:1]
exec(open(Path(__file__).with_name("smoke_adapter.py"), encoding="utf-8").read().split("# 1 system prompt carries the task")[0])
MARKUP = "<tool_call>\n<function=python>\n<parameter=code>\nprint(123)\n</parameter>\n</function>\n</tool_call>"
class R:
    def __init__(self, fr, content="", reasoning="", calls=None):
        self.finish_reason, self.usage, self.served_by = fr, None, ""
        self.message = {"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": calls or []}
def duck_view(res):   # the exact sequence of tool_agent.py after _chat_completion
    m = res.message
    raw_reasoning = TA._extract_reasoning_text(m); raw_content = TA._normalize_message_content(m.get("content", ""))
    calls = json.loads(json.dumps(m.get("tool_calls") or []))
    markup = TA._contains_tool_call_markup(raw_reasoning, raw_content)
    if not calls and markup:
        calls = TA._recover_tool_calls_from_markup(raw_reasoning, raw_content)
    bad = []
    for tc in calls:
        a = (tc.get("function") or {}).get("arguments", "{}")
        if isinstance(a, str):
            try: json.loads(a)
            except json.JSONDecodeError: bad.append(1)
    return "DROPPED (malformed)" if bad else (f"EXECUTES {[(c['function']['name'], c['function']['arguments']) for c in calls]}" if calls else "no call")
ag2 = Agent(task=task, task_id="flip", branch=1, model="qwen", base_url="http://127.0.0.1:9/v1", provider="vllm")
_orig = TA.ToolAgent._chat_completion
for label, mk in [("length + markup in content", lambda: R("length", content=MARKUP)),
                  ("length + markup in reasoning", lambda: R("length", reasoning="thinking... " + MARKUP + " more thinking")),
                  ("length + parsed call", lambda: R("length", calls=[{"id": "c", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": "print(1)"})}}])),
                  ("stop + markup in content (normal recovery)", lambda: R("stop", content=MARKUP))]:
    TA.ToolAgent._chat_completion = lambda self, messages, _mk=mk, **kw: _mk()
    print(f"{label:45} -> {duck_view(ag2._chat_completion([], tools=None))}")
TA.ToolAgent._chat_completion = _orig
