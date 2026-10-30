"""Fake vLLM 0.32 OpenAI server for the E13 RC1 harness (arm vllm_fn): checks the Flash-Next serve flags, prints the log
lines the arm parses (PLE / INC / KV / concurrency), serves /health, /v1/completions, /v1/chat/completions."""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]
print("INFO fake vllm serve args:", " ".join(argv), flush=True)
REQ = [["--tensor-parallel-size", "4"], ["--enable-expert-parallel"], ["--enforce-eager"], ["--max-model-len", "8192"],
       ["--max-num-seqs", "4"], ["--gpu-memory-utilization", "0.93"], ["--language-model-only"],
       ["--engram-config", json.dumps({"cpu_offload": True})], ["--served-model-name", "flashnext"],
       ["--no-enable-prefix-caching"], ["--dtype", "bfloat16"]]
for r in REQ:
    i = argv.index(r[0]) if r[0] in argv else -1
    if i < 0 or (len(r) == 2 and argv[i + 1] != r[1]):
        print(f"ValueError: E13 RC1 harness: missing / wrong vLLM flag {r}", flush=True)
        sys.exit(1)
for bad in ("--speculative-config", "--trust-remote-code", "--quantization"):
    if bad in argv:
        print(f"ValueError: E13 RC1 harness: unexpected vLLM flag {bad}", flush=True)
        sys.exit(1)
if not argv or argv[0].startswith("--"):
    print("ValueError: E13 RC1 harness: the model path must be the first argument", flush=True)
    sys.exit(1)
PORT = int(argv[argv.index("--port") + 1])
import os
_hp, _td = os.environ.get("PYTHONPATH", ""), os.environ.get("E13_VLLM_TRACE_DIR")
if not (_hp and os.path.isfile(os.path.join(_hp, "sitecustomize.py")) and _td) or "E13_MEMTRACE_DIR" in os.environ:
    print(f"ValueError: E13 RC1 harness: vLLM gc hook missing (PYTHONPATH={_hp!r}, E13_VLLM_TRACE_DIR={_td!r})", flush=True)
    sys.exit(1)
print("(EngineCore pid=1) INFO Using INC quantization (auto-round W4A16): AutoGPTQMoEMethod with Marlin", flush=True)
print("(Worker_TP0_EP0 pid=2) INFO Initialized PLE embedding model.layers.1.ple.ple_embedding.ngram_embedding: "
      "quantization_method=Qwen4ExpPLEUnquantizedEmbeddingMethod, weight_dtype=torch.bfloat16, weight_device=cpu, pinned=True", flush=True)
print("(EngineCore pid=1) INFO GPU KV cache size: 61,440 tokens", flush=True)
print("(EngineCore pid=1) INFO Maximum concurrency for 8,192 tokens per request: 7.50x", flush=True)
T_READY = time.time() + 2


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, code=200):
        b = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._send("", 200 if time.time() >= T_READY else 503)
        return self._send("{}", 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if req.get("model") != "flashnext":
            return self._send(json.dumps({"error": "model"}), 404)
        if self.path == "/v1/completions":
            if str(req.get("prompt", "")).startswith("1, 2, 3"):
                t = " 9, 10, 11, 12, 13, 14, 15"
                return self._send(json.dumps({"choices": [{"text": t, "logprobs": {"token_logprobs": [-0.01] * 8}}],
                                              "usage": {"completion_tokens": 20}}))
            k = int(req.get("max_tokens", 16))
            time.sleep(0.2)
            return self._send(json.dumps({"choices": [{"text": "x " * k}], "usage": {"completion_tokens": k}}))
        if self.path == "/v1/chat/completions":
            return self._send(json.dumps({"choices": [{"message": {"content": "<think>2 plus 2</think>\n\n4"}, "finish_reason": "stop"}],
                                          "usage": {"completion_tokens": 9}}))
        return self._send("{}", 404)


print(f"INFO Starting vLLM API server on http://127.0.0.1:{PORT}", flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
