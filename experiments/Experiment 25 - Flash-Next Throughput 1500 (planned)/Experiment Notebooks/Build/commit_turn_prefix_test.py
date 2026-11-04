"""E25 CPU test: does a no-thinking Duck commit turn keep the prefix cache? (Flash-Next tokenizer + chat template)

Variants for the commit request after a multi-turn tool conversation:
  A  current harness: thinking on (generation prompt ends '<think>\n')
  B  enable_thinking=False (template drops the reasoning-effort line from the SYSTEM block and appends '<think>\n\n</think>\n\n')
  C  thinking-on render, then the trailing '<think>\n' replaced by '<think>\n\n</think>\n\n' (same header; sent as token ids)
Measured against the previous request's prompt + its generated reply (what the prefix cache holds): common prefix tokens,
first mismatch, reusable fraction, and the reusable fraction at the Merlin Flash-Next block granularity (4096).

Usage: python commit_turn_prefix_test.py <dir with tokenizer.json, tokenizer_config.json, chat_template.jinja> [out.json]
"""
import hashlib
import json
import random
import sys
from pathlib import Path

from tokenizers import Tokenizer
from transformers import AutoTokenizer

D = Path(sys.argv[1])
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else None
TOK = Tokenizer.from_file(str(D / "tokenizer.json"))
HF = AutoTokenizer.from_pretrained(str(D))
TPL = (D / "chat_template.jinja").read_text(encoding="utf-8")
BLOCK = 4096

rng = random.Random(7)
words = ["grid", "row", "column", "colour", "object", "shift", "mirror", "pattern", "cell", "rule", "train", "test", "count",
         "border", "fill", "symmetry", "rotate", "copy", "region", "shape", "check", "hypothesis", "verify", "output"]


def think(n_words):
    return " ".join(rng.choice(words) for _ in range(n_words))


def grid(h, w):
    return "\n".join("".join(str(rng.randrange(10)) for _ in range(w)) for _ in range(h))


TOOLS = [{"type": "function", "function": {"name": "run_python", "description": "Run Python code in the sandbox and return stdout.",
                                           "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                                                          "required": ["code"]}}},
         {"type": "function", "function": {"name": "submit", "description": "Submit the transform function for the test input.",
                                           "parameters": {"type": "object", "properties": {"code": {"type": "string"}},
                                                          "required": ["code"]}}}]
system = "You solve ARC tasks. Write a Python transform(grid) function, test it on the training pairs, then submit."
task = "Task 1a2b3c4d\n\n" + "\n\n".join(f"Example {i} input:\n{grid(12, 14)}\nExample {i} output:\n{grid(12, 14)}" for i in range(3))
msgs = [{"role": "system", "content": system}, {"role": "user", "content": task}]
for t in range(4):                                         # 4 earlier agent turns with long thinking + a tool call each
    msgs.append({"role": "assistant", "content": "", "reasoning_content": think(2500),
                 "tool_calls": [{"type": "function", "function": {"name": "run_python",
                                                                  "arguments": {"code": f"print(transform_{t}(train[0]))"}}}]})
    msgs.append({"role": "tool", "content": grid(12, 14)})


def render(messages, **kw):
    return HF.apply_chat_template(messages, tools=TOOLS, tokenize=False, add_generation_prompt=True, chat_template=TPL,
                                  reasoning_effort="xhigh", **kw)


def ids(text):
    return TOK.encode(text, add_special_tokens=False).ids


# what the prefix cache holds: the previous request's prompt + its generated reply (thinking + tool call)
prev_msgs, last_reply = msgs[:-2], msgs[-2]
prev_prompt = render(prev_msgs)
reply_text = last_reply["reasoning_content"] + "\n</think>\n\n<tool_call>\n<function=run_python>\n<parameter=code>\n" + \
             last_reply["tool_calls"][0]["function"]["arguments"]["code"] + "\n</parameter>\n</function>\n</tool_call>"
cached = ids(prev_prompt + reply_text)

commit_note = {"role": "user", "content": "Time is almost up. Call submit now with your best transform."}
full = msgs + [commit_note]
A = ids(render(full))
B = ids(render(full, enable_thinking=False))
A_text = render(full)
assert A_text.endswith("<think>\n"), A_text[-40:]
C = ids(A_text[:-len("<think>\n")] + "<think>\n\n</think>\n\n")


def common(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def report(name, req):
    c = common(cached, req)
    blocks = (c // BLOCK) * BLOCK
    return {"variant": name, "request_tokens": len(req), "common_prefix_with_cache": c, "first_mismatch": c,
            "reusable_frac": round(c / len(req), 4), "reusable_frac_at_block_4096": round(blocks / len(req), 4),
            "to_prefill": len(req) - c, "header_sha16": hashlib.sha256(json.dumps(req[:64]).encode()).hexdigest()[:16]}


res = {"cached_tokens": len(cached), "A_vs_C_common": common(A, C),
       "rows": [report("A thinking on (current)", A), report("B enable_thinking=False", B),
                report("C same render + closed think", C)]}
res["B_header_differs"] = res["rows"][1]["header_sha16"] != res["rows"][0]["header_sha16"]
res["C_header_same"] = res["rows"][2]["header_sha16"] == res["rows"][0]["header_sha16"]
print(json.dumps(res, indent=1))
if OUT:
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
