"""Fake vLLM 0.32.0 OpenAI server for the E18 CPU harness (no model, no GPU). Launched by the REAL E18 engine cell with the
REAL `vllm serve` flags (argv = [model, --flag, value, ...]); scripted Poetiq replies per task (harness_s1.py behaviours).
- the startup lines the cell parses ("version 0.32.0", "GPU KV cache size: N tokens"); GET /health; GET /metrics
  (vllm:kv_cache_usage_perc, vllm:num_requests_running, vllm:generation_tokens_total, vllm:num_preemptions_total)
- POST /v1/completions stream=true with prompt token ids and return_token_ids: SSE chunks with integer token_ids, the last
  one with finish_reason, then [DONE]; a client disconnect aborts the request (counted)
- the task is found from the prompt (digits of its grids), the iteration by counting its phase-1 requests, phase 2 =
  the prompt holds </think>
Config: env E18_FAKE_CFG = JSON file {tokenizer, challenges, tasks, behaviour, close_id, scenario, log}."""
import hashlib
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


CFG = json.load(open(os.environ["E18_FAKE_CFG"], encoding="utf-8"))
SCEN = CFG["scenario"]
PORT = int(arg("--port", 8011))
SEQS = int(arg("--max-num-seqs", 16))
print(f"fake vllm serve {' '.join(argv)}", flush=True)
print("(APIServer pid=1) INFO [api_utils.py:347]  version 0.32.0", flush=True)
if SCEN == "lmofail" and "--language-model-only" in argv:
    print("ValueError: simulated: --language-model-only rejected", flush=True); sys.exit(1)
if SCEN == "allfail":
    print("RuntimeError: simulated: engine core failed to start", flush=True); sys.exit(1)
for k, v in {"VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1", "HF_HUB_OFFLINE": "1"}.items():
    if os.environ.get(k) != v:
        print(f"ValueError: E18 harness: env {k}={os.environ.get(k)!r}", flush=True); sys.exit(1)
time.sleep(0.3)
print("(EngineCore pid=2) INFO [kv_cache_utils.py:2327] GPU KV cache size: 1,006,517 tokens, Maximum concurrency for "
      "106,496 tokens per request: 9.45x", flush=True)

import tokenizers  # noqa: E402
TOK = tokenizers.Tokenizer.from_file(CFG["tokenizer"])
CH = json.load(open(CFG["challenges"], encoding="utf-8"))
TASKS = CFG["tasks"]
BEH = CFG["behaviour"]
CLOSE = int(CFG["close_id"])
FILL = TOK.encode(" thinking", add_special_tokens=False).ids


def _digits(g):
    return "".join(str(v) for row in g for v in row)


FP = {t: [_digits(x) for ex in CH[t]["train"] for x in (ex["input"], ex["output"])] + [_digits(q["input"]) for q in CH[t]["test"]]
      for t in TASKS}


def task_of(ids):
    d = re.sub(r"\D", "", TOK.decode(ids, skip_special_tokens=False))
    best = max(TASKS, key=lambda t: (sum(1 for g in FP[t] if g in d), sum(len(g) for g in FP[t] if g in d)))
    return best


def answer_code(tid, it):
    b = BEH.get(tid, "fail_same")
    if b in ("pass", "never"):
        return "```python\ndef transform(grid):\n    return grid  # MARK:PASS_ALL\n```"
    if b == "partial_then_fail":
        return ("```python\ndef transform(grid):\n    return grid  # MARK:PASS_PARTIAL\n```" if it == 0 else
                f"```python\ndef transform(grid):\n    return grid  # MARK:FAIL_ITER {it}\n```")
    if b == "forced_then_pass":
        return (f"```python\ndef transform(grid):\n    return grid  # MARK:FAIL_ITER {it}\n```" if it == 0 else
                "```python\ndef transform(grid):\n    return grid  # MARK:PASS_ALL\n```")
    if b == "fail_distinct":
        return f"```python\ndef transform(grid):\n    return grid  # MARK:FAIL_ITER {it}\n```"
    if b == "illustrative_first":
        return ("For example:\n```python\nx = helper(grid)  # MARK:ZERO no transform here\n```\n"
                "First idea:\n```python\ndef transform(grid):\n    return grid  # MARK:ZERO\n```\n"
                "Final:\n```python\ndef transform(grid):\n    return grid  # MARK:PASS_ALL\n```")
    return "```python\ndef transform(grid):\n    return grid  # MARK:ZERO\n```"


LOCK = threading.Lock()
P1 = {}
DEATH = [0]
STATE = {"phase1": {}, "running": 0, "gen": 0, "disconnects": 0, "requests": [], "max_running": 0}


def log_request(rec):
    with LOCK:
        STATE["requests"].append(rec)
        with open(CFG["log"], "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in STATE.items()}, f)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, body, code=200, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._send("{}")
        if self.path == "/metrics":
            with LOCK:
                run = STATE["running"]
                # kvgate: full as soon as one solve request arrived (the solver caches the value 5 s: one read)
                kv = 0.97 if (SCEN == "kvgate" and STATE["phase1"]) else 0.30
                txt = (f"vllm:kv_cache_usage_perc{{engine=\"0\"}} {kv}\n"
                       f"vllm:num_requests_running{{engine=\"0\"}} {run}\n"
                       f"vllm:generation_tokens_total{{engine=\"0\"}} {STATE['gen']}\n"
                       f"vllm:num_preemptions_total{{engine=\"0\"}} 0\n"
                       "vllm:spec_decode_num_accepted_tokens_total{engine=\"0\"} 10\n"
                       "vllm:spec_decode_num_drafts_total{engine=\"0\"} 5\n"
                       "vllm:e2e_request_latency_seconds_bucket{le=\"1.0\"} 3\n")
            return self._send(txt, ctype="text/plain")
        self._send('{"error": "not found"}', 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path != "/v1/completions":
            return self._send('{"error": "not found"}', 404)
        ids = body["prompt"]
        cap = int(body["max_tokens"])
        greedy = float(body.get("temperature", 1.0)) == 0.0
        rec = {"len": len(ids), "max_tokens": cap, "temperature": body.get("temperature"), "top_p": body.get("top_p"),
               "top_k": body.get("top_k"), "seed": body.get("seed"), "stream": body.get("stream"),
               "return_token_ids": body.get("return_token_ids"), "skip_special_tokens": body.get("skip_special_tokens"),
               "model": body.get("model"), "prompt_sha": hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:12], "phase": None, "task": None, "it": None, "outcome": None}
        if greedy:                                     # the engine smoke
            stream, reason = TOK.encode(" 1 2 3 4 5 6 7 8", add_special_tokens=False).ids[:cap], "stop"
            rec["phase"] = "smoke"
        else:
            tid = task_of(ids)
            phase = 2 if CLOSE in ids else 1
            with LOCK:
                if phase == 1:
                    STATE["phase1"][tid] = STATE["phase1"].get(tid, -1) + 1
                    P1[tid] = list(ids)
                    DEATH[0] += 1
                    if SCEN == "death" and DEATH[0] >= 4 and not os.path.exists(CFG["marker"]):
                        open(CFG["marker"], "w").write("died")      # the first launch dies mid-run; the relaunch lives
                        threading.Timer(0.3, lambda: os._exit(1)).start()
                it = STATE["phase1"].get(tid, 0)
                p1 = P1.get(tid)
            b = BEH.get(tid, "fail_same")
            rec.update(phase=phase, task=tid, it=it)
            if phase == 2:
                # phase 2 = phase-1 prompt + its tokens (None: its phase 1 ran on a server process that died: death scenario)
                rec["prefix_ok"] = None if not p1 else ids[:len(p1)] == p1
                stream, reason = TOK.encode(answer_code(tid, it), add_special_tokens=False).ids[:cap], "stop"
            elif b == "never":                         # streams until the client disconnects (drain / hard stop)
                stream, reason = None, None
            elif b == "forced_then_pass" and it == 0:
                stream, reason = (FILL * (cap // len(FILL) + 1))[:cap], "length"
            else:
                stream = FILL * 30 + [CLOSE] + TOK.encode("\n\n" + answer_code(tid, it), add_special_tokens=False).ids
                stream, reason = stream[:cap], ("stop" if len(stream) <= cap else "length")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        with LOCK:
            STATE["running"] += 1
            STATE["max_running"] = max(STATE["max_running"], STATE["running"])
        out = 0
        try:
            def emit(d):
                self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
                self.wfile.flush()
            if stream is None:
                while True:
                    emit({"choices": [{"index": 0, "text": "", "token_ids": FILL, "finish_reason": None}]})
                    out += len(FILL)
                    time.sleep(0.05)
            for i in range(0, max(1, len(stream)), 512):
                chunk = stream[i:i + 512]
                last = i + 512 >= len(stream)
                emit({"choices": [{"index": 0, "text": "", "token_ids": chunk, "finish_reason": reason if last else None}]})
                out += len(chunk)
                time.sleep(0.003)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            rec["outcome"] = reason
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            with LOCK:
                STATE["disconnects"] += 1
            rec["outcome"] = "disconnect"
        finally:
            with LOCK:
                STATE["running"] -= 1
                STATE["gen"] += out
            rec["out"] = out
            log_request(rec)
        self.close_connection = True


class S(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


S(("127.0.0.1", PORT), H).serve_forever()
