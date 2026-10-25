"""E14-P unit checks for arc2_duck's E12_RAW_REASONING_HISTORY switch (CPU, no server). Our own scripted data only.
usage: python test_raw_reasoning.py"""
import os
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
E12 = Path(r"D:/ARC AGI 2/experiments/Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)/duck_arc2")
FIX = Path(r"D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures")
sys.path[:0] = [str(E12), str(E12 / "vendor")]
os.environ.update({"E12_ARC2_REV5": "1", "E12_ARC2_REV6": "1", "E12_TOKENIZER_JSON": str(FIX / "tok_int4" / "tokenizer.json")})
import arc2_duck as A  # noqa: E402

os.environ.update(A.duck_env("http://127.0.0.1:9/v1", "qwen", 4))
TA, DP, _, _ = A._import_duck()
Agent = A.make_agent_class()
task = {"train": [{"input": [[1, 2]], "output": [[2, 1]]}], "test": [{"input": [[3, 4]]}]}
ok = True


def expect(name, cond, info=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"  -> {info}"))
    ok &= bool(cond)


def agent(on):
    os.environ["E12_RAW_REASONING_HISTORY"] = "1" if on else "0"
    return Agent(task=task, task_id="t", branch=0, model="qwen", base_url="http://127.0.0.1:9/v1", provider="vllm")


RAW = "First idea.\n\nSecond paragraph.\n  \nThird.\n"
NORM = TA._normalize_message_content(RAW)
KEY = TA._reasoning_history_keys()[0]
expect("setup: normalization deletes the blank lines", NORM == "First idea.\nSecond paragraph.\nThird.", repr(NORM))
expect("setup: the history key is reasoning_content", KEY == "reasoning_content", KEY)


def am(text=NORM, **extra):
    return {"role": "assistant", KEY: text, **extra}


a = agent(False)
a._arc2_last_raw_reasoning = RAW
m = a._arc2_raw_reasoning_message(am())
expect("off (default): message unchanged", m[KEY] == NORM and a.arc2_raw_reasoning == {"kept": 0, "same": 0, "mismatch": 0, "markup": 0})
expect("off (default): env unset means off", (os.environ.pop("E12_RAW_REASONING_HISTORY", None) or True) and not A._raw_reasoning_history())

a = agent(True)
a._arc2_last_raw_reasoning = RAW
src = am(content="", tool_calls=[{"id": "c1"}])
m = a._arc2_raw_reasoning_message(src)
expect("on: the stored reasoning is the raw text", m[KEY] == RAW and a.arc2_raw_reasoning["kept"] == 1, m)
expect("on: content and tool calls untouched; the original dict is not mutated", m["tool_calls"] == [{"id": "c1"}]
       and m["content"] == "" and src[KEY] == NORM)
expect("on: the capture is consumed once", a._arc2_last_raw_reasoning is None and a._arc2_raw_reasoning_message(am())[KEY] == NORM)

a._arc2_last_raw_reasoning = RAW
u = a._arc2_raw_reasoning_message({"role": "user", "content": "nudge"})
expect("on: a user message does not consume the capture", u == {"role": "user", "content": "nudge"}
       and a._arc2_last_raw_reasoning == RAW)
m = a._arc2_raw_reasoning_message(am())
expect("on: ... the following assistant message gets it", m[KEY] == RAW)

a._arc2_last_raw_reasoning = RAW
m = a._arc2_raw_reasoning_message(am("Some other reply."))
expect("on: a different reply is never overwritten (mismatch counted)", m[KEY] == "Some other reply."
       and a.arc2_raw_reasoning["mismatch"] == 1)

a._arc2_last_raw_reasoning = "Plan.\n\n<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n</tool_call>"
m = a._arc2_raw_reasoning_message(am("Plan."))
expect("on: reasoning with tool-call markup keeps Duck's stripped text (markup counted)", m[KEY] == "Plan."
       and a.arc2_raw_reasoning["markup"] == 1)

a._arc2_last_raw_reasoning = "One line only."
m = a._arc2_raw_reasoning_message(am("One line only."))
expect("on: identical text -> nothing to do (same counted)", m[KEY] == "One line only." and a.arc2_raw_reasoning["same"] == 1)

a._arc2_last_raw_reasoning = RAW
m = a._arc2_raw_reasoning_message({"role": "assistant", "content": "answer"})
expect("on: an assistant message without reasoning stays as it is", m == {"role": "assistant", "content": "answer"})

# the real append path: arc2's _append_context_message runs the replacement before Duck appends
a._arc2_last_raw_reasoning = RAW
msgs = []
a._append_context_message(msgs, am())
expect("on: _append_context_message appends the raw-reasoning message", len(msgs) == 1 and msgs[0][KEY] == RAW, msgs)

# the template still renders what the model generated: '<think>\n' + raw|trim + '\n</think>\n\n'
expect("render: raw|trim keeps the inner blank lines", RAW.strip() == "First idea.\n\nSecond paragraph.\n  \nThird.")

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
