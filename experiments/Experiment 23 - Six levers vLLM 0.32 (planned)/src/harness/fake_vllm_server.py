"""E23 copy of the Experiment 14 harness fake server (only text_for differs).
Fake vLLM 0.32.0 OpenAI server for the VMS CPU harness (no model, no GPU). Launched by the REAL VMS cell with the REAL
`vllm serve` flags (argv after the script = [model, --flag, value, ...]).
- startup log lines the notebook parses: "GPU KV cache size: N tokens", "Maximum concurrency for M tokens per request: Xx"
- GET /health, GET /metrics (Prometheus: vllm:num_requests_running, vllm:generation_tokens_total,
  vllm:spec_decode_num_accepted_tokens_total, vllm:spec_decode_num_drafts_total, vllm:kv_cache_usage_perc,
  vllm:num_preemptions_total)
- POST /v1/completions stream=true: SSE chunks; temperature 0 = deterministic text from the prompt; ignore_eos keeps
  streaming until max_tokens or the client disconnects (= vLLM's abort on disconnect)
- running is capped at max_num_seqs x data_parallel_size (the rest wait)
FAKE_SPEED = tokens per second per request."""
import json, os, sys, threading, time, hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


MODEL = argv[0] if argv and not argv[0].startswith("--") else "?"
PORT = int(arg("--port", 8011))
SEQS = int(arg("--max-num-seqs", 16))
DP = int(arg("--data-parallel-size", 1))
TP = int(arg("--tensor-parallel-size", 4))
CAP = SEQS * DP
SPEC = json.loads(arg("--speculative-config", "null") or "null")
SPEED = float(os.environ.get("FAKE_SPEED", 400))
KV = 900000 if "int4" in MODEL.lower() or "w4a16" in MODEL.lower() else 600000
print(f"fake vllm serve {' '.join(argv)}", flush=True)
assert TP * DP <= 4, "more GPUs than the box has"
time.sleep(0.5)
_lb = arg("--linear-backend", "auto")
# vLLM 0.32.0 on SM 8.9: auto (and marlin) -> MarlinFP8ScaledMMLinearKernel; triton -> TritonFp8BlockScaledMMKernel
print(f"INFO [__init__.py] Selected {'TritonFp8BlockScaledMMKernel' if _lb == 'triton' else 'MarlinFP8ScaledMMLinearKernel'} for Fp8LinearMethod", flush=True)
if os.environ.get("VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC") == "1":   # vLLM 0.32.0 flashinfer_pcie_ipc_all_reduce.py log line
    print("INFO Initialized FlashInfer PCIe IPC all-reduce for TP4, hidden_dim=5120, dtype=torch.bfloat16, max_tokens=64.", flush=True)
print(f"INFO GPU KV cache size: {KV:,} tokens", flush=True)
print(f"INFO Maximum concurrency for {arg('--max-model-len', '106496')} tokens per request: {KV / int(arg('--max-model-len', 106496)):.2f}x", flush=True)
LOCK = threading.Lock()
RUN, WAIT = set(), []
CNT = {"gen": 0.0, "acc": 0.0, "drf": 0.0}
CTX = {}
T_READY = time.time() + 0.5


_SYL = ["ka", "lo", "mi", "ne", "ru", "sa", "ti", "vo", "ze", "pa", "qu", "ri", "bel", "dor", "fen", "gul"]
_VOCAB = [a + b + c for a in _SYL for b in _SYL for c in _SYL]       # 4,096 alphabetic words
_TOKID = {w + " ": 1000 + i for i, w in enumerate(_VOCAB)}          # one token id per word


def text_for(prompt, n, greedy):
    """E23 copy: varied words (one token = one word), so the lab's garbage check sees healthy text; deterministic for
    greedy, the prompt tail + time otherwise (the E14 original emitted a repeating 64-char hex string)."""
    import random
    h = hashlib.sha256((json.dumps(prompt[-64:]) + ("" if greedy else str(time.time()))).encode()).hexdigest()
    rng = random.Random(h)
    return [_VOCAB[rng.randrange(len(_VOCAB))] + " " for _ in range(n)]


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, body, code=200, ctype="application/json"):
        b = body.encode() if isinstance(body, str) else body
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._send("", 200 if time.time() >= T_READY else 503)
        if self.path == "/metrics":
            with LOCK:
                n, ctx = len(RUN), sum(CTX.values())
                lines = [f'vllm:num_requests_running{{engine="0"}} {n}', f'vllm:num_requests_waiting{{engine="0"}} {len(WAIT)}',
                         f'vllm:generation_tokens_total{{engine="0"}} {CNT["gen"]}',
                         f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {CNT["acc"]}',
                         f'vllm:spec_decode_num_drafts_total{{engine="0"}} {CNT["drf"]}',
                         f'vllm:kv_cache_usage_perc{{engine="0"}} {min(0.99, ctx / KV):.4f}',
                         'vllm:num_preemptions_total{engine="0"} 0',
                         f'vllm:prompt_tokens_total{{engine="0"}} {CNT.get("prompt", 0.0)}',
                         f'vllm:prefix_cache_queries_total{{engine="0"}} {CNT.get("pq", 0.0)}',
                         f'vllm:prefix_cache_hits_total{{engine="0"}} {CNT.get("ph", 0.0)}',
                         f'vllm:prompt_tokens_cached_total{{engine="0"}} {CNT.get("ph", 0.0)}']
            return self._send("\n".join(lines) + "\n", ctype="text/plain")
        return self._send("{}", 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path != "/v1/completions":
            return self._send("{}", 404)
        rid = f"r{time.time_ns()}"
        prompt, max_t = req.get("prompt") or [], int(req.get("max_tokens", 16))
        greedy = float(req.get("temperature", 1.0)) == 0.0
        nlp = int(req.get("logprobs") or 0)
        with LOCK:
            WAIT.append(rid)
        while True:
            with LOCK:
                if WAIT and WAIT[0] == rid and len(RUN) < CAP:
                    WAIT.pop(0)
                    RUN.add(rid)
                    CTX[rid] = len(prompt)
                    break
            time.sleep(0.01)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        toks = text_for(prompt, max_t, greedy)
        k = int((SPEC or {}).get("num_speculative_tokens", 0))
        step = (1 + 0.6 * k) if k else 1.0
        try:
            sent = 0
            while sent < max_t:
                chunk = toks[sent:sent + 8]
                sent += len(chunk)
                d = {"choices": [{"text": "".join(chunk), "index": 0,
                                  "finish_reason": "length" if sent >= max_t else None}]}
                if req.get("return_token_ids"):      # vLLM 0.32.0: token_ids = the delta ids of this chunk (ints)
                    d["choices"][0]["token_ids"] = [_TOKID.get(c, 0) for c in chunk]
                if nlp:     # vLLM completions logprobs shape: tokens / token_logprobs / top_logprobs (dict per token)
                    d["choices"][0]["logprobs"] = {"tokens": list(chunk), "token_logprobs": [-0.05] * len(chunk),
                                                   "top_logprobs": [{c: -0.05, "␀": -2.5} for c in chunk]}
                b = b"data: " + json.dumps(d).encode() + b"\n\n"
                self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
                self.wfile.flush()
                with LOCK:
                    CNT["gen"] += len(chunk)
                    if sent == len(chunk):           # first chunk: the prompt counted once (prefill done)
                        CNT["prompt"] = CNT.get("prompt", 0.0) + len(prompt)
                        CNT["pq"] = CNT.get("pq", 0.0) + len(prompt)
                        CNT["ph"] = CNT.get("ph", 0.0) + (len(prompt) * 0.9 if "--no-enable-prefix-caching" not in argv else 0)
                    if k:
                        CNT["drf"] += len(chunk) / step
                        CNT["acc"] += len(chunk) - len(chunk) / step
                    CTX[rid] = CTX.get(rid, 0) + len(chunk)
                time.sleep(len(chunk) / SPEED if not greedy else 0.0)
            b = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n" + b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            with LOCK:
                RUN.discard(rid)
                CTX.pop(rid, None)


ThreadingHTTPServer.request_queue_size = 256      # rev 2e-e: default 5 drops simultaneous connects (Windows)
srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
srv.daemon_threads = True
print(f"INFO Started server process; Uvicorn running on http://127.0.0.1:{PORT}", flush=True)
srv.serve_forever()
