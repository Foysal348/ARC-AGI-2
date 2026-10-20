"""Fake SGLang 0.5.21 server for the E11-05 harness (CPU, no model).
- launch log in the real format (KV / Mamba cache lines, verify graph bs), /health, /metrics (Prometheus text)
- POST /generate with stream=true: SSE 'data: {...}' chunks with DISJOINT output_ids (as --incremental-streaming-output),
  meta_info.finish_reason {"type": stop|length|abort}, then 'data: [DONE]'. Natural stops end with EOS (no_stop_trim).
- POST /abort_request {"rid"}: the stream ends with finish type 'abort' (FAKE_SCEN=abortbroken ignores it)
- GET /fake_log: per rid input ids, sent ids, abort flag, finish (harness checks)
- decode ticker: '[... TP0] Decode batch, #running-req: N, #full token: X, ...' lines while requests run
FAKE_SCEN: lazyfail (extra_buffer_lazy launch fails) | launchfail (every launch fails) | abortbroken | serverdeath | kvgate"""
import json, os, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import beh_e11_05 as B

argv = sys.argv[1:]


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


SCEN = os.environ.get("FAKE_SCEN", "")
PORT = int(arg("--port", 30000))
STRAT = arg("--mamba-radix-cache-strategy", "extra_buffer")
K = int(arg("--max-mamba-cache-size", 56))
MAXRUN = int(arg("--max-running-requests", 12))
INCREMENTAL = "--incremental-streaming-output" in argv
print("[2026-10-18 10:00:00] fake sglang server args:", " ".join(argv), flush=True)

if SCEN == "launchfail":
    print("ValueError: simulated permanent configuration error", flush=True)
    sys.exit(1)
if SCEN == "lazyfail" and STRAT == "extra_buffer_lazy":
    print("[2026-10-18 10:00:01 TP0] Mamba Cache is allocated. max_mamba_cache_size: 56 (normal startup line)", flush=True)
    print("AssertionError: extra_buffer_lazy unsupported here (simulated)", flush=True)
    sys.exit(1)
if not INCREMENTAL:
    print("ValueError: the E11-05 shim needs --incremental-streaming-output", flush=True)
    sys.exit(1)

for tp in range(4):
    print(f"[2026-10-18 10:00:02 TP{tp}] Load weight end. elapsed=2.4 s, type=Qwen3_5ForConditionalGeneration, quant=compressed-tensors, avail mem=17.10 GB, mem usage=4.62 GB.", flush=True)
    if SCEN != "nocaplog":                       # nocaplog: the capacity evidence is missing -> the gate must fail
        print(f"[2026-10-18 10:00:03 TP{tp}] Mamba Cache is allocated. max_mamba_cache_size: {40 if SCEN == 'lowcap' else K}, conv_state size: 0.04GB, ssm_state size: 1.00GB intermediate_ssm_state_cache size: 1.83GB", flush=True)
    print(f"[2026-10-18 10:00:03 TP{tp}] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 881336, K size: 3.36 GB, V size: 3.36 GB", flush=True)
    print(f"[2026-10-18 10:00:03 TP{tp}] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 881336, K size: 1.05 GB, V size: 1.05 GB", flush=True)
print("[2026-10-18 10:00:04 TP0] Capture target verify CUDA graph begin. backend=full, num_tokens_per_req=8, bs=[1, 2, 3, 4, 5, 6, 7, 8, 10, 12], avail mem=5.27 GB", flush=True)
print("[2026-10-18 10:00:05 TP0] Capture draft verify CUDA graph end. elapsed=1.0 s, mem usage=0.38 GB, avail mem=2.22 GB.", flush=True)

from transformers import AutoTokenizer          # noqa: E402  (after the launch-failure exits, like the real server)
TOK = AutoTokenizer.from_pretrained(str(B.TOKDIR_INT4))
FILL = TOK.encode(" thinking", add_special_tokens=False)
CLOSE = TOK.encode("</think>", add_special_tokens=False)[0]
EOS = 248046
_ch, _sol, MULTI, SINGLE = B.load()
BEH = B.beh_map(MULTI, SINGLE, SCEN)

LOCK = threading.Lock()
LOG = {}            # rid -> {"input": [...], "sent": [...], "aborted": bool, "finish": str}
ACTIVE = {}         # rid -> current context length
ABORT = set()
N_GEN = [0]
SEQS = []           # finished input+output sequences (prefix-cache emulation)
T_READY = time.time() + 0.5


def lcp(a, b):
    n = min(len(a), len(b)); i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def plan(rid, max_new):
    """(token stream, finish type, seconds per chunk, chunk sizes)"""
    if rid == "e11-smoke-a":
        return (FILL * 20)[:min(16, max_new)], "length", 0.0, [3, 5]
    if rid == "e11-smoke-b":
        return (FILL * 5000)[:max_new], "length", 0.01, [8]
    tid, it, phase = B.parse_rid(rid)
    b = BEH.get(tid, "fail_same")
    code = TOK.encode("\n\n" + B.answer_code(b, it), add_special_tokens=False)
    if phase == 1 and (b == "forced_then_pass" and it == 0):
        return (FILL * (max_new // len(FILL) + 1))[:max_new], "length", 0.0, [1, 7, 40, 256, 2048]
    if phase == 1 and b == "slow_pass":
        return (FILL * (max_new // len(FILL) + 1))[:max_new], "length", 0.01, [32]
    if phase == 1:
        s = FILL * 30 + [CLOSE] + code + [EOS]
    else:
        s = code + [EOS]
    s = s[:max_new]
    return s, ("stop" if s and s[-1] == EOS else "length"), 0.0, [1, 7, 40, 256]


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"         # E11-05: chunked SSE like the real server (uvicorn)

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
            return self._send("" if time.time() >= T_READY else "loading", 200 if time.time() >= T_READY else 503)
        if self.path == "/metrics":
            with LOCK:
                n = len(ACTIVE); ctx = sum(ACTIVE.values())
            usage = 0.97 if SCEN == "kvgate" else min(0.9, ctx / 881336)
            lines = ["# HELP sglang:num_running_reqs x", "# TYPE sglang:num_running_reqs gauge",
                     f'sglang:num_running_reqs{{model_name="int4",engine_type="unified",tp_rank="0"}} {n}',
                     f'sglang:num_queue_reqs{{model_name="int4"}} 0',
                     f'sglang:token_usage{{model_name="int4"}} {max(usage, n * 4 / K):.4f}',
                     f'sglang:full_token_usage{{model_name="int4"}} {usage:.4f}',
                     f'sglang:gen_throughput{{model_name="int4"}} 400.0',
                     f'sglang:spec_accept_length{{model_name="int4"}} 3.5',
                     f'sglang:generation_tokens_total{{model_name="int4"}} 1234.0',
                     'sglang:e2e_request_latency_seconds_bucket{le="1.0",model_name="int4"} 3.0']
            return self._send("\n".join(lines) + "\n", ctype="text/plain")
        if self.path == "/fake_log":
            with LOCK:
                return self._send(json.dumps(LOG))
        return self._send("{}", 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/abort_request":
            if SCEN == "drain":                   # like SGLang: 200 at once, the request ends a little later
                threading.Timer(0.5, lambda r=req.get("rid"): ABORT.add(r)).start()
            elif SCEN != "abortbroken":
                with LOCK:
                    ABORT.add(req.get("rid"))
            return self._send("", 200)
        if self.path != "/generate":
            return self._send("{}", 404)
        with LOCK:
            N_GEN[0] += 1
            n_gen = N_GEN[0]
        if SCEN == "serverdeath" and n_gen > 14:
            print("[2026-10-18 10:30:00 TP0] Scheduler hit an exception: simulated crash", flush=True)
            os._exit(3)
        rid, ids, sp = req["rid"], list(req["input_ids"]), req["sampling_params"]
        assert req.get("stream") is True and sp.get("no_stop_trim") is True and sp.get("skip_special_tokens") is False
        stream, fin, delay, sizes = plan(rid, int(sp["max_new_tokens"]))
        with LOCK:
            cached = max((lcp(ids, s) for s in SEQS), default=0)
            LOG[rid] = {"input": ids, "sent": [], "aborted": False, "finish": None, "max_new": sp["max_new_tokens"],
                        "t_start": time.time(), "t_end": None,
                        "temperature": sp.get("temperature"), "top_k": sp.get("top_k")}
            ACTIVE[rid] = len(ids)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")         # like uvicorn's StreamingResponse
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def emit(b):
            self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
            self.wfile.flush()
        pos, k, finish = 0, 0, None
        try:
            while finish is None:
                with LOCK:
                    ab = rid in ABORT
                if ab:
                    finish = {"type": "abort", "message": "Abort requested"}
                    chunk = []
                else:
                    sz = sizes[k % len(sizes)]; k += 1
                    chunk = stream[pos:pos + sz]; pos += len(chunk)
                    if pos >= len(stream):
                        finish = {"type": "stop", "matched": EOS} if fin == "stop" else {"type": "length", "length": len(stream)}
                with LOCK:
                    LOG[rid]["sent"] += chunk
                    if finish is not None:
                        LOG[rid]["t_finish"] = time.time()          # before the finish chunk leaves the server
                    ACTIVE[rid] = len(ids) + len(LOG[rid]["sent"])
                meta = {"id": rid, "finish_reason": finish, "prompt_tokens": len(ids), "completion_tokens": len(LOG[rid]["sent"]),
                        "cached_tokens": cached, "spec_accept_length": 3.5}
                emit(b"data: " + json.dumps({"output_ids": chunk, "meta_info": meta}).encode() + b"\n\n")
                if delay:
                    time.sleep(delay)
            emit(b"data: [DONE]\n\n"); self.wfile.write(b"0\r\n\r\n"); self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            finish = {"type": "abort", "message": "client disconnected"}
        with LOCK:
            LOG[rid]["finish"] = finish["type"]; LOG[rid]["aborted"] = finish["type"] == "abort"; LOG[rid]["t_end"] = time.time()
            ACTIVE.pop(rid, None)
            SEQS.append(ids + LOG[rid]["sent"])


def ticker():
    while True:
        time.sleep(0.05)
        with LOCK:
            n = len(ACTIVE); ft = sum(ACTIVE.values())
        if n:
            print(f"[2026-10-18 10:01:00 TP0] Decode batch, #running-req: {n}, #full token: {ft}, full token usage: 0.10, "
                  f"mamba num: {n * 4}, mamba usage: 0.50, accept len: 3.50, accept rate: 0.40, cuda graph: True, "
                  f"gen throughput (token/s): {350 + n:.2f}, #queue-req: 0", flush=True)


threading.Thread(target=ticker, daemon=True).start()
srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
srv.daemon_threads = True
print(f"[2026-10-18 10:00:06] The server is fired up and ready to roll! port {PORT}", flush=True)
srv.serve_forever()
