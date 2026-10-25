"""E14 CPU probe (2026-10-23): an OpenAI-compatible fake SGLang server that renders every request with the REAL INT4 chat
template (transformers apply_chat_template, preserve_thinking=True as the server default, the request's
chat_template_kwargs, tool-call arguments parsed to dicts as SGLang does) and checks, character by character, whether the
rendered prompt of request n+1 starts with: rendered prompt n + the text the "model" generated for reply n.

The generated text of a reply is what SGLang's qwen3 reasoning parser and qwen3_coder tool parser would have parsed:
    <reasoning R with paragraph breaks>\n</think>\n\n<tool_call>\n<function=python>\n<parameter=code>\n<code>\n</parameter>\n</function>\n</tool_call><|im_end|>
and the response carries reasoning_content = R + "\n" (text before </think>, unstripped), content "", and the call.

Per request the server records where the first difference is:
- full          : the whole previous prompt + reply is a prefix (the cache could reuse everything)
- in_reply      : the difference is inside the previous reply (offset into the reply, and whether inside the reasoning)
- before_reply  : the difference is inside the previous PROMPT: the client edited earlier content (the deeper-miss
                  question). The message index and a short context are recorded.
- cut_reply     : the previous reply was cut at the output limit (expected: the template closes </think>)
GET /probe_log returns everything. Scripted replies only (our own text and code)."""
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from transformers import AutoTokenizer

TOKDIR = sys.argv[1]
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 30999
tok = AutoTokenizer.from_pretrained(TOKDIR)
TEMPLATE = open(os.path.join(TOKDIR, "chat_template.jinja"), encoding="utf-8").read()
DEFAULT_KW = {"preserve_thinking": True}          # = --default-chat-template-kwargs in our launch flags

LOCK = threading.Lock()
STATE = {}          # conv -> {"step": i, "prev_prompt": str, "prev_gen": str, "prev_cut": bool, "prev_msgs": int}
LOG = []
CONV_RE = re.compile(r"CONV=(\w+)")
COUNT = [0]

INSPECT = "print(grid_shape(train[0]['input']))\nprint(len(train), len(test_inputs))"
IDENT = "def transform(g):\n    return [list(r) for r in g]\ncheck(transform)"
PROGRAM = ["call:" + INSPECT, "cut", "call:" + IDENT, "call:" + INSPECT, "plain", "call:" + INSPECT,
           "call:" + IDENT, "call:" + INSPECT]


def reasoning_text(conv, i):
    # paragraph breaks on purpose: models write them, Duck's normalization deletes them
    return (f"CONV={conv} step {i}. Let me look at the training pairs first.\n\n"
            f"The first grid has a clear structure; the output seems to keep the input.\n\n"
            f"  \n"
            f"Plan: run a small check, then decide.")


def render(body):
    msgs = []
    for m in body.get("messages") or []:
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for c in m["tool_calls"]:
                c = json.loads(json.dumps(c))
                f = c.get("function") or {}
                a = f.get("arguments")
                if isinstance(a, str):
                    try:
                        f["arguments"] = json.loads(a)
                    except Exception:
                        f["arguments"] = {}
                tcs.append(c)
            m["tool_calls"] = tcs
        msgs.append(m)
    kw = dict(DEFAULT_KW)
    kw.update(body.get("chat_template_kwargs") or {})
    return tok.apply_chat_template(msgs, tools=body.get("tools"), tokenize=False, add_generation_prompt=True,
                                   chat_template=TEMPLATE, **kw)


def conv_of(body):
    for m in body.get("messages") or []:
        if m.get("role") == "assistant":
            f = CONV_RE.search(str(m.get("reasoning_content") or m.get("reasoning") or ""))
            if f:
                return f.group(1)
    return None


def msg_index_at(body, prompt, pos):
    """Which message of the request renders the character at pos (approximate: by message content search)."""
    for i, m in enumerate(body.get("messages") or []):
        for key in ("content", "reasoning_content"):
            v = m.get(key)
            if isinstance(v, str) and len(v) > 20:
                at = prompt.find(v[:40])
                if at >= 0 and at <= pos <= at + len(v) + 40:
                    return i, m.get("role"), key
    return None, None, None


def reply(body):
    prompt = render(body)
    conv = conv_of(body)
    with LOCK:
        COUNT[0] += 1
        if conv is None:
            conv = f"c{COUNT[0]}"
            STATE[conv] = {"step": 0}
        st = STATE[conv]
        i = st["step"]
        st["step"] = i + 1
        rec = {"conv": conv, "i": i, "prompt_chars": len(prompt), "n_msgs": len(body.get("messages") or [])}
        if "prev_prompt" in st:
            exp = st["prev_prompt"] + st["prev_gen"]
            lcp = len(os.path.commonprefix([exp, prompt]))
            pp = len(st["prev_prompt"])
            if st["prev_cut"]:
                rec["cls"] = "cut_reply"
                rec["lcp_minus_prev_prompt"] = lcp - pp
            elif lcp >= len(exp):
                rec["cls"] = "full"
            elif lcp >= pp:
                rec["cls"] = "in_reply"
                rec["offset_in_reply"] = lcp - pp
                rec["in_reasoning"] = (lcp - pp) < st["prev_reason_len"]
                rec["expected"] = exp[lcp:lcp + 40]
                rec["got"] = prompt[lcp:lcp + 40]
            else:
                rec["cls"] = "before_reply"
                rec["chars_before_prev_end"] = pp - lcp
                rec["at_message"] = msg_index_at(body, prompt, lcp)
                rec["expected"] = exp[max(0, lcp - 30):lcp + 50]
                rec["got"] = prompt[max(0, lcp - 30):lcp + 50]
        step = PROGRAM[min(i, len(PROGRAM) - 1)]
        R = reasoning_text(conv, i)
        if step.startswith("call:"):
            code = step[5:]
            gen = (R + "\n</think>\n\n<tool_call>\n<function=python>\n<parameter=code>\n" + code +
                   "\n</parameter>\n</function>\n</tool_call><|im_end|>")
            msg = {"role": "assistant", "content": "", "reasoning_content": R + "\n",
                   "tool_calls": [{"id": f"call_{conv}_{i}", "type": "function",
                                   "function": {"name": "python", "arguments": json.dumps({"code": code})}}]}
            fin, cut = "tool_calls", False
        elif step == "cut":                       # output limit inside the thinking: no </think>, no call
            gen = R + " and then I keep thinking about the colours"
            msg = {"role": "assistant", "content": "", "reasoning_content": gen}
            fin, cut = "length", True
        else:                                     # a plain answer without a tool call
            text = "I believe the output equals the input."
            gen = R + "\n</think>\n\n" + text + "<|im_end|>"
            msg = {"role": "assistant", "content": text, "reasoning_content": R + "\n"}
            fin, cut = "stop", False
        st.update(prev_prompt=prompt, prev_gen=gen, prev_cut=cut, prev_reason_len=len(R))
        rec["reply"] = step.split(":")[0]
        LOG.append(rec)
    n_prompt = max(1, len(prompt) // 3)
    return {"id": f"chatcmpl-{conv}-{i}", "object": "chat.completion", "model": body.get("model"),
            "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": 900, "total_tokens": n_prompt + 900,
                      "prompt_tokens_details": {"cached_tokens": 0}}}


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, body, code=200):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._send("{}")
        if self.path == "/v1/models":
            return self._send(json.dumps({"object": "list", "data": [{"id": "fake-int4", "object": "model"}]}))
        if self.path == "/probe_log":
            with LOCK:
                return self._send(json.dumps(LOG))
        return self._send("{}", 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path != "/v1/chat/completions":
            return self._send("{}", 404)
        try:
            return self._send(json.dumps(reply(body)))
        except Exception as e:
            with LOCK:
                LOG.append({"error": repr(e)[:300]})
            return self._send(json.dumps({"error": {"message": repr(e)[:300]}}), 500)


srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
srv.daemon_threads = True
print("fake prefix server ready", PORT, flush=True)
srv.serve_forever()
