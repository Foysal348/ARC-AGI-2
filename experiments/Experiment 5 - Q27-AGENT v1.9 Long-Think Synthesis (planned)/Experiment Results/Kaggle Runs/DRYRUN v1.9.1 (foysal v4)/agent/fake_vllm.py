"""v1.8 DRYRUN: a stand-in for the vLLM OpenAI server on the same port and API, for a CPU-only Kaggle run (no GPU
quota) that drives the REAL notebook path end to end -- launcher cfg, smoke, agent e2e probe, agent loop, DIRECT
rescue, eval report, submission writing, guard -- on Kaggle's exact image. It checks plumbing, not model quality.

Behaviour: rejects unknown reasoning efforts like Qwen3.8's chat template; with tools it answers a named tool_choice
with that tool, otherwise python first, then submit (identity transform); without tools it answers the LB-18.75 prompt
with the FINAL INPUT grid in a ```grid block, except every third prompt (no grid -> exercises the commit call).
    python fake_vllm.py [port]
"""
import json
import sys
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

STATS = {"requests": 0, "tool": 0, "direct": 0, "commit": 0, "rejected": 0}
LOCK = threading.Lock()


def final_input_grid(text):
    if "FINAL INPUT:" not in text:
        return None
    rows = []
    for line in text.split("FINAL INPUT:", 1)[1].splitlines()[1:]:
        if not line.strip() or not line.strip()[0].isdigit():
            break
        rows.append(line.strip())
    return "\n".join(rows) or None


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            return self._send(200, {"object": "list", "data": [{"id": "qwen", "object": "model"}]})
        if self.path.startswith("/stats"):
            return self._send(200, STATS)
        if self.path.startswith("/metrics"):            # v1.8.3: the vLLM counters the reports read
            text = ("# HELP fake\n"
                    'vllm:cache_config_info{block_size="1600",num_gpu_blocks="846"} 1.0\n'
                    'vllm:num_preemptions_total{model_name="qwen"} 0.0\n'
                    'vllm:num_requests_running{model_name="qwen"} 3.0\n'
                    'vllm:num_requests_waiting{model_name="qwen"} 0.0\n'
                    'vllm:kv_cache_usage_perc{model_name="qwen"} 0.12\n'
                    'vllm:spec_decode_num_accepted_tokens_total{model_name="qwen"} 66.0\n'
                    'vllm:spec_decode_num_draft_tokens_total{model_name="qwen"} 100.0\n').encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(text)))
            self.end_headers()
            return self.wfile.write(text)
        self._send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        with LOCK:
            STATS["requests"] += 1
        tk = body.get("chat_template_kwargs") or {}
        eff = tk.get("reasoning_effort")
        if eff is not None and eff not in ("xhigh", "medium", "low"):
            with LOCK:
                STATS["rejected"] += 1
            return self._send(400, {"error": {"message": f"Unexpected reasoning effort {eff}. Supported types are "
                                                         "xhigh (default), medium, and low.", "code": 400}})
        if self.path.startswith('/tokenize'):
            msgs = body.get('messages') or []
            s = ''.join(str(x.get('content', '')) for x in msgs)
            return self._send(200, {'count': max(1, len(s)//3 + 48)})
        msgs = body.get("messages") or []
        last_user = next((m.get("content") or "" for m in reversed(msgs) if m.get("role") == "user"), "")
        max_tokens = int(body.get("max_tokens") or 1024)
        if body.get("tools"):
            with LOCK:
                STATS["tool"] += 1
            choice = body.get("tool_choice")
            prior_calls = sum(1 for m in msgs if m.get("role") == "assistant" and m.get("tool_calls"))
            if isinstance(choice, dict):
                name = choice["function"]["name"]
            elif "submit" in last_user or prior_calls:
                name = "submit"
            else:
                name = "python"
            code = ("def transform(grid):\n    return grid\n" if name == "submit"
                    else "print(train_inputs[0].shape)")
            msg = {"role": "assistant", "content": "", "reasoning_content": "fake reasoning",
                   "tool_calls": [{"id": f"call{STATS['requests']}", "type": "function",
                                   "function": {"name": name, "arguments": json.dumps({"code": code})}}]}
            return self._send(200, {"choices": [{"message": msg, "finish_reason": "tool_calls"}],
                                    "usage": {"prompt_tokens": 500, "completion_tokens": min(max_tokens, 700)}})
        if 'Return your final complete standalone Python program' in last_user:
            content = '```python\ndef transform(grid):\n    return [row[:] for row in grid]\n```'
        elif 'Output format: ```python' in last_user:
            content = '```python\ndef transform(grid):\n    return [row[:] for row in grid]\n```'
        elif "Reply with the digit 7" in last_user:
            content = "7"
        elif "Stop analysing now" in last_user:
            with LOCK:
                STATS["commit"] += 1
            g = final_input_grid(last_user)
            content = f"```grid\n{g}\n```" if g else "no grid"
        else:
            with LOCK:
                STATS["direct"] += 1
            g = final_input_grid(last_user)
            no_grid = zlib.crc32(last_user.encode()) % 3 == 0
            content = "I could not finish." if (no_grid or not g) else f"```grid\n{g}\n```"
        msg = {"role": "assistant", "content": content, "reasoning_content": "fake reasoning about the rule"}
        return self._send(200, {"choices": [{"message": msg, "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 800, "completion_tokens": min(max_tokens, 1200)}})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 1234
    srv = ThreadingHTTPServer(("127.0.0.1", port), H)
    print(f"fake vLLM server on {port}", flush=True)
    srv.serve_forever()
