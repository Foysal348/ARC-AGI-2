"""E14 fake SGLang 0.5.21 server for the CPU notebook harness (no model, no GPU). Launched by the REAL engine cell with the
REAL flags. Built from the E12 fake (fake_sgl_duck_server.py, rev 6 scripted Duck program, copied below) plus what the E14
measurement cells need:
- /generate non-streaming with rid, abort, greedy determinism, logprobs (+ top-2), prompt_tokens / cached_tokens;
- decode-batch log lines while requests run (running, #full token, accept len (not in target-only), gen throughput,
  mamba usage), prefill-batch lines when a request starts;
- /metrics: real in-flight counts (the server cap from --max-running-requests), KV and mamba usage;
- /v1/chat/completions: the Duck program for solver traffic; requests with the X-E14-Probe header get a probe reply
  (reasoning WITH a blank line, one python call) and, with return_token_ids, prompt/response token ids from a byte-level
  fake tokenizer and a byte-exact chat render, so the E14-P replay logic and the cache classes are testable;
- cached tokens = the longest prefix shared with any earlier prompt + output, rounded down to 64 (a radix-cache model);
- /start_profile writes per-rank chrome traces (FAKE_TRACE_NAME pattern) into output_dir after num_steps.
Scripted replies only (our own text and code). FAKE_SPEED = generated tokens per second per request."""
import gzip
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FIX = os.environ.get("E14_FIXTURES", r"D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures")
sys.path.insert(0, FIX)
import e12_fixtures as B  # noqa: E402

argv = sys.argv[1:]


def arg(name, default=None, n=1):
    if name not in argv:
        return default
    i = argv.index(name)
    return argv[i + 1] if n == 1 else argv[i + 1:i + 1 + n]


PORT = int(arg("--port", 30000))
K = int(arg("--max-mamba-cache-size", 56))
CAP = int(arg("--max-running-requests", 13))
SPEC = "--speculative-algorithm" in argv
WIDTH = int(arg("--speculative-num-draft-tokens", 8)) if SPEC else 0
INT8 = "--enable-int8-mamba-checkpoint" in argv
SPEED = float(os.environ.get("FAKE_SPEED", 2000))
# a faster-but-wrong arm for the correctness gate test: when this flag is on the command line, greedy outputs change
# at token 5 with a confident (2.4 nat) margin and everything runs 10 % faster (audit 19: the gate must reject it)
CORRUPT = bool(os.environ.get("FAKE_CORRUPT_IF_FLAG")) and os.environ["FAKE_CORRUPT_IF_FLAG"] in argv
if CORRUPT:
    SPEED *= 1.1


def _delay_threshold():
    """SGLang 0.5.21 managers/min_free_slots_delayer.py resolve_min_free_slots: an explicit value wins (capped by the
    server cap, <= 1 disables); unset + DFlash + cap >= 8 -> min(4, max(2, (cap + 5) // 6)) = 3 at cap 13."""
    v = arg("--min-free-slots-delay")
    if v is not None:
        t = min(int(v), CAP)
        return t if t > 1 else None
    if SPEC and CAP >= 8:
        return min(4, max(2, (CAP + 5) // 6))
    return None
if "--cuda-graph-bs-decode" in argv:
    i = argv.index("--cuda-graph-bs-decode") + 1
    BS = []
    while i < len(argv) and not argv[i].startswith("--"):
        BS.append(int(argv[i]))
        i += 1
else:
    mx = int(arg("--cuda-graph-max-bs-decode", 12))
    BS = [b for b in (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16) if b <= mx]
LOCK = threading.Lock()


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def out(line):
    print(f"[{ts()}{line}", flush=True)


print(f"[{ts()}] fake sglang server args:", " ".join(argv), flush=True)
REQUIRED = [["--reasoning-parser", "qwen3"], ["--tool-call-parser", "qwen3_coder"], ["--schedule-policy", "lpm"],
            ["--enable-cache-report"], ["--default-chat-template-kwargs", json.dumps({"preserve_thinking": True})]]
for req in REQUIRED:
    i = argv.index(req[0]) if req[0] in argv else -1
    if i < 0 or (len(req) == 2 and argv[i + 1] != req[1]):
        print(f"ValueError: E14 harness: missing Duck serving flag {req}", flush=True)
        sys.exit(1)
time.sleep(0.3)
for tp in range(4):
    out(f" TP{tp}] Load weight begin. avail mem=21.72 GB")
time.sleep(0.3)
for tp in range(4):
    out(f" TP{tp}] Load weight end. elapsed=2.4 s, type=Qwen3_5ForConditionalGeneration, quant=compressed-tensors, avail mem=17.10 GB, mem usage=4.62 GB.")
    out(f" TP{tp}] Mamba Cache is allocated. max_mamba_cache_size: {K}, conv_state size: 0.04GB, ssm_state size: 1.00GB intermediate_ssm_state_cache size: 1.83GB")
    out(f" TP{tp}] KV Cache is allocated. dtype: torch.float8_e4m3fn, #tokens: 881336, K size: 3.36 GB, V size: 3.36 GB")
if SPEC:
    out(f" TP0] Capture target verify CUDA graph begin. backend=full, num_tokens_per_req={WIDTH}, bs={BS}, avail mem=5.27 GB")
    out(" TP0] Capture draft verify CUDA graph end. elapsed=1.0 s, mem usage=0.38 GB, avail mem=2.22 GB.")
else:
    out(f" TP0] Capture cuda graph begin. This can take up to several minutes. avail mem=5.27 GB bs={BS}")

# ------------------------------------------------------------------ the E12 rev 6 scripted Duck program (copied)
CH, SOL, _, _ = B.load()
LOG = {"chat": [], "generate": [], "probe_chat": 0, "profiles": 0}
CONV_STEP, CONV_META, TASK_ROLES, NEW_CONVS = {}, {}, {}, []
INFLIGHT = [0, 0]
ABORT = set()
T_READY = time.time() + 0.5
CONV_RE = re.compile(r"CONV=(\w+)")


def rows(g):
    return "\n".join(" ".join(str(v) for v in r) for r in g)


def _counts(g):
    d = {}
    for r in g:
        for v in r:
            d[v] = d.get(v, 0) + 1
    return dict(sorted(d.items()))


def _r6_sig(t):
    out_ = []
    for p in t["train"][:2]:
        for tag, g in (("in ", p["input"]), ("out", p["output"])):
            out_.append(f"  {tag} {len(g)}x{len(g[0])} colors {_counts(g)}")
    return out_


def task_of(system_text):
    for marker in ("Training pair 1 input:\n", "Training pair 0 input:\n"):
        if marker in system_text:
            head = system_text.split(marker, 1)[-1].split("\nTraining pair " + marker[14] + " output:", 1)[0]
            for tid, t in CH.items():
                if rows(t["train"][0]["input"]) == head:
                    return tid
    if "ARC TASK FACTS" in system_text:
        hits = [tid for tid, t in CH.items() if all(s in system_text for s in _r6_sig(t))]
        if len(hits) == 1:
            return hits[0]
        hits = [tid for tid, t in CH.items() if all(
            f"TRAIN {i}: in {len(p['input'])}x{len(p['input'][0])} {_counts(p['input'])} -> out {len(p['output'])}x"
            f"{len(p['output'][0])} {_counts(p['output'])}" in system_text for i, p in enumerate(t["train"][:2]))]
        if len(hits) == 1:
            return hits[0]
    return None


CUT_CODE = "print('CUT_CODE_RAN')"
NOCALL = "@@NOCALL@@"
INSPECT_R6 = "print(grid_shape(train[0]['input']), len(components(train[0]['input'])))\ninspect_pair(0)"


def lookup_r6(tid, name="transform"):
    t = CH[tid]
    table = {json.dumps(p["input"]): p["output"] for p in t["train"]}
    for qi, q in enumerate(t["test"]):
        table[json.dumps(q["input"])] = SOL[tid][qi]
    return (f"def {name}(g):\n    import json\n    table = " + json.dumps(table) + "\n"
            "    return table.get(json.dumps(g), flip_h(flip_h(g)))")


def lookup_train_only_r6(tid):
    t = CH[tid]
    table = {json.dumps(p["input"]): p["output"] for p in t["train"]}
    return ("def transform_alt(g):\n    import json\n    table = " + json.dumps(table) + "\n"
            "    return table.get(json.dumps(g), [list(r) for r in g])")


def program(tid, role):
    return ([NOCALL, INSPECT_R6, lookup_r6(tid), INSPECT_R6] if role == 0
            else [CUT_CODE, lookup_train_only_r6(tid), INSPECT_R6])


# ------------------------------------------------------------------ fake tokenizer + render + radix-cache model
def tok(text):
    return list(text.encode("utf-8"))


def call_markup(calls):
    s = ""
    for j, c in enumerate(calls or []):
        f = c.get("function") or {}
        a = f.get("arguments")
        a = json.loads(a) if isinstance(a, str) and a.strip().startswith("{") else (a or {})
        s += ("" if j == 0 else "\n") + "<tool_call>\n<function=" + str(f.get("name")) + ">\n"
        for k, v in (a or {}).items():
            s += f"<parameter={k}>\n{v}\n</parameter>\n"
        s += "</function>\n</tool_call>"
    return s


def render(msgs):
    """Byte-exact stand-in for the Qwen template: assistant = <think>\\n + reasoning|trim + \\n</think>\\n\\n + content + calls."""
    s = ""
    for m in msgs:
        r = m.get("role")
        if r == "assistant":
            s += "<|im_start|>assistant\n<think>\n" + str(m.get("reasoning_content") or "").strip() + "\n</think>\n\n" + \
                 str(m.get("content") or "") + call_markup(m.get("tool_calls")) + "<|im_end|>\n"
        else:
            s += f"<|im_start|>{r}\n" + str(m.get("content") or "") + "<|im_end|>\n"
    return s + "<|im_start|>assistant\n<think>\n"


SEQS = []          # every prompt + output seen (token lists)


def cached_for(ids):
    best = 0
    with LOCK:
        for s in SEQS[-400:]:
            n = 0
            for a, b in zip(s, ids):
                if a != b:
                    break
                n += 1
            best = max(best, n)
    return (min(best, len(ids) - 1) // 64) * 64 if ids else 0


def remember(ids):
    with LOCK:
        SEQS.append(list(ids))


# ------------------------------------------------------------------ decode-line printer (while anything runs)
RUN = {}           # rid -> {"ctx": tokens, "t0": start}   (admitted = running)
WAIT = []          # rids waiting for admission (FIFO), the real #queue-req
DELAY = _delay_threshold()   # the min-free-slots delayer threshold (None = off)


def printer():
    last = 0.0
    while True:
        time.sleep(0.15)
        with LOCK:
            n = len(RUN)
            nq = len(WAIT)
            ft = sum(v["ctx"] for v in RUN.values())
        if n == 0:
            continue
        acc = 1 + 0.47 * (WIDTH - 1) if SPEC else 1.0
        step = 0.115 + 0.002 * n
        gen = n * acc / step * (1.1 if CORRUPT else 1.0)
        mamba = round(min(1.0, (3 * n + (8 if INT8 else 20)) / K), 2)
        line = (f" TP0] Decode batch, #running-req: {n}, #full token: {ft}, full token usage: {min(0.99, ft / 881336):.2f}, "
                f"mamba usage: {mamba:.2f}, ")
        line += (f"accept len: {acc:.2f}, " if SPEC else "") + f"cuda graph: True, gen throughput (token/s): {gen:.2f}, "
        line += f"#queue-req: {nq}"
        if time.time() - last > 0.0:
            out(line)
            last = time.time()


threading.Thread(target=printer, daemon=True).start()


def decode_wait(rid, n_tokens, ctx):
    """Hold the request as running for n_tokens / FAKE_SPEED seconds (abortable). Returns 'stop'/'length'/'abort'."""
    with LOCK:
        WAIT.append(rid)
    while True:                      # admission: the server cap and the DFlash min-free-slots delayer
        with LOCK:
            free = CAP - len(RUN)
            delayed = DELAY is not None and len(RUN) > 0 and free < DELAY
            if WAIT and WAIT[0] == rid and free > 0 and not delayed:
                WAIT.pop(0)
                RUN[rid] = {"ctx": ctx, "t0": time.time()}
                INFLIGHT[0] = len(RUN)
                INFLIGHT[1] = max(INFLIGHT[1], INFLIGHT[0])
                break
            if rid in ABORT:
                WAIT.remove(rid)
                return "abort"
        time.sleep(0.01)
    out(f" TP0] Prefill batch, #new-seq: 1, #new-token: {ctx}, #cached-token: 0, token usage: 0.01, #running-req: {len(RUN)}")
    t_end = time.time() + n_tokens / SPEED
    fin = "length"
    while time.time() < t_end:
        if rid in ABORT:
            fin = "abort"
            break
        time.sleep(0.02)
    with LOCK:
        RUN.pop(rid, None)
        INFLIGHT[0] = len(RUN)
    return fin


def fake_ids(seed_ids, n, greedy=True, rid=""):
    import hashlib
    h = hashlib.sha256((json.dumps(seed_ids[-64:]) + ("" if greedy else rid + str(time.time()))).encode()).digest()
    o = [32 + (h[i % 32] + i) % 90 for i in range(n)]
    if CORRUPT and greedy and len(o) > 5:
        o[5] = 31                      # a different confident token: the arm is fast but wrong
    return o


# ------------------------------------------------------------------ chat
def probe_reply(body, conv_hint):
    msgs = body.get("messages") or []
    n_turn = sum(1 for m in msgs if m.get("role") == "assistant")
    R = (f"Probe step {n_turn}. Let me look at the grids first.\n\nThe second paragraph checks the colours {conv_hint}.")
    code = "print(grid_shape(train[0]['input']))"
    calls = [{"id": f"call_probe_{n_turn}_{int(time.time() * 1e6) % 10 ** 9}", "type": "function",
              "function": {"name": "python", "arguments": json.dumps({"code": code})}}]
    msg = {"role": "assistant", "content": "", "reasoning_content": R + "\n", "tool_calls": calls}
    gen_text = R + "\n</think>\n\n" + call_markup(calls) + "<|im_end|>"
    return msg, gen_text, "tool_calls"


def chat_reply(body, probe):
    msgs = body.get("messages") or []
    last = str(msgs[-1].get("content", "")) if msgs else ""
    prompt_text = render(msgs)
    p_ids = tok(prompt_text)
    cached = cached_for(p_ids)
    mt = int(body.get("max_tokens") or 16)
    if not body.get("tools") and "Reply with exactly this code and nothing else: " in last:   # rev 6 isolation preflight
        mark = last.split("Reply with exactly this code and nothing else: ", 1)[1].strip()
        with LOCK:
            LOG.setdefault("preflight", []).append(mark)
        return {"id": f"chatcmpl-pre-{mark}", "object": "chat.completion", "model": body.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": mark}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28,
                          "prompt_tokens_details": {"cached_tokens": 0}}}
    if probe:
        with LOCK:
            LOG["probe_chat"] += 1
        msg, gen_text, fin = probe_reply(body, "")
        o_ids = tok(gen_text)
        if mt <= 1:
            o_ids, fin, msg = o_ids[:1], "length", {"role": "assistant", "content": ""}
        rid = f"probe-{time.time()}"
        decode_wait(rid, min(len(o_ids), mt), len(p_ids))
        remember(p_ids + o_ids)
        ch = {"index": 0, "message": msg, "finish_reason": fin}
        if body.get("return_token_ids"):
            ch.update(prompt_token_ids=p_ids, response_token_ids=o_ids)
        return {"id": rid, "object": "chat.completion", "model": body.get("model"), "choices": [ch],
                "usage": {"prompt_tokens": len(p_ids), "completion_tokens": len(o_ids), "total_tokens": len(p_ids) + len(o_ids),
                          "prompt_tokens_details": {"cached_tokens": cached}}}
    # ---- the Duck program (E12 fake, rev 6)
    tid = task_of(str(msgs[0].get("content", ""))) if msgs else None
    conv = None
    for m in msgs:
        if m.get("role") == "assistant":
            f = CONV_RE.search(str(m.get("reasoning_content") or ""))
            if f:
                conv = f.group(1)
    with LOCK:
        if conv is None:
            role = TASK_ROLES.get(tid, 0)
            TASK_ROLES[tid] = role + 1
            conv = f"{tid}r{role}"
            CONV_META[conv] = (tid, role)
            CONV_STEP[conv] = 0
            NEW_CONVS.append([tid, role, time.time()])
        tid, role = CONV_META[conv]
        i = CONV_STEP[conv]
        CONV_STEP[conv] = i + 1
    steps = program(tid, role) if tid else [INSPECT_R6]
    code = steps[min(i, len(steps) - 1)]
    R = f"CONV={conv} THINK step {i} for {tid} role {role}.\n\nA second paragraph (a blank line above, as models write)."
    calls = [{"id": f"call_{conv}_{i}", "type": "function", "function": {"name": "python", "arguments": json.dumps({"code": code})}}]
    msg = {"role": "assistant", "content": "", "reasoning_content": R + "\n", "tool_calls": calls}
    fin = "length" if code == CUT_CODE else "tool_calls"
    gen_text = R + "\n</think>\n\n" + call_markup(calls) + "<|im_end|>"
    if code == NOCALL:
        msg = {"role": "assistant", "content": "", "reasoning_content": R + " (cut at the output limit, no tool call)."}
        fin = "length"
        gen_text = msg["reasoning_content"]
    o_ids = tok(gen_text)
    decode_wait(f"duck-{conv}-{i}", min(2100, 200), len(p_ids))
    remember(p_ids + o_ids)
    return {"id": f"chatcmpl-{conv}-{i}", "object": "chat.completion", "model": body.get("model"),
            "choices": [{"index": 0, "message": msg, "finish_reason": fin}],
            "usage": {"prompt_tokens": len(p_ids), "completion_tokens": 2100, "total_tokens": len(p_ids) + 2100,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


# ------------------------------------------------------------------ profiler traces
def write_traces(d, steps):
    os.makedirs(d, exist_ok=True)
    pat = os.environ.get("FAKE_TRACE_NAME", "{t}-TP-{r}.trace.json.gz")
    stamp = int(time.time())
    for r in range(4):
        ev, t = [], 0
        for s in range(steps):
            for name, dur, n in (("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 340 + 20 * r, 140),
                                 ("Marlin<256, 1, 8, 8, 4, 8>", 60, 200), ("BatchDecodeWithPagedKVCacheKernel", 400, 16),
                                 ("fused_sigmoid_gating_delta_rule_update_kernel", 150, 48),
                                 ("ncclDevKernel_AllGather_RING_LL", 8000, 1), ("topk_kernel", 900, 2)):
                for _ in range(n):
                    ev.append({"ph": "X", "cat": "kernel", "name": name, "ts": t, "dur": dur, "pid": r, "tid": 7})
                    t += dur
        p = os.path.join(d, pat.format(t=stamp, r=r))
        data = json.dumps({"traceEvents": ev}).encode()
        with (gzip.open(p, "wb") if p.endswith(".gz") else open(p, "wb")) as f:
            f.write(data)


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
            return self._send("" if time.time() >= T_READY else "loading", 200 if time.time() >= T_READY else 503)
        if self.path == "/v1/models":
            return self._send(json.dumps({"object": "list", "data": [{"id": "fake-int4", "object": "model"}]}))
        if self.path == "/metrics":
            with LOCK:
                n = len(RUN)
                ft = sum(v["ctx"] for v in RUN.values())
                nq = len(WAIT)
            lines = [f'sglang:num_running_reqs{{model_name="int4",tp_rank="{r}"}} {n}' for r in range(4)]
            lines += [f'sglang:num_queue_reqs{{model_name="int4"}} {nq}',
                      'sglang:num_retracted_requests_total{model_name="int4"} 0',
                      f'sglang:full_token_usage{{model_name="int4"}} {min(0.99, ft / 881336):.4f}',
                      f'sglang:mamba_usage{{model_name="int4"}} {min(1.0, (3 * min(n, CAP)) / K):.4f}',
                      'sglang:token_usage{model_name="int4"} 0.1', 'sglang:gen_throughput{model_name="int4"} 300.0',
                      'sglang:cache_hit_rate{model_name="int4"} 0.9']
            return self._send("\n".join(lines) + "\n", ctype="text/plain")
        if self.path == "/fake_log":
            with LOCK:
                return self._send(json.dumps(dict(LOG, new_convs=NEW_CONVS, max_inflight=INFLIGHT[1],
                                                  argv=argv, spec=SPEC, width=WIDTH, int8=INT8, graph_bs=BS,
                                                  delay_threshold=DELAY, corrupt=CORRUPT)))
        return self._send("{}", 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        probe = self.headers.get("X-E14-Probe") == "1"
        if self.path == "/abort_request":
            with LOCK:
                ABORT.add(req.get("rid"))
            return self._send("", 200)
        if self.path == "/start_profile":
            d, steps = req.get("output_dir") or "/tmp/fake_profile", int(req.get("num_steps") or 5)
            with LOCK:
                LOG["profiles"] += 1
            threading.Timer(1.0, write_traces, args=(d, steps)).start()
            return self._send("Start profiling.\n", 200, "text/plain")
        if self.path.split("?")[0] == "/flush_cache":     # 0.5.21 http_server.py:987: plain text, refused while busy
            with LOCK:
                busy = bool(RUN) or bool(WAIT)
                if not busy:
                    SEQS.clear()
            # the real server answers 400 on a refused flush (http_server.py:999-1001)
            return self._send("Flush cache failed.\n" if busy else "Cache flushed.\nPlease check backend logs for more "
                              "details. (When there are running or waiting requests, the operation will not be "
                              "performed.)\n", 400 if busy else 200, "text/plain")
        if self.path == "/v1/chat/completions":
            out_ = chat_reply(req, probe)
            with LOCK:
                LOG["chat"].append({k: req.get(k) for k in ("model", "max_tokens", "temperature", "top_p", "top_k", "stream",
                                                            "chat_template_kwargs", "tool_choice")}
                                   | {"n_messages": len(req.get("messages") or []), "id": out_["id"], "probe": probe}
                                   | ({} if req.get("tools") else {"tools": False}))
            return self._send(json.dumps(out_))
        if self.path != "/generate":
            return self._send("{}", 404)
        rid, sp = req.get("rid") or f"g{time.time()}", req.get("sampling_params") or {}
        ids = list(req.get("input_ids") or [])
        with LOCK:
            LOG["generate"].append(rid)
        total = 16 if rid == "e11-smoke-a" else int(sp.get("max_new_tokens", 16))
        if req.get("stream"):                                   # the E11 shim's streaming path (smoke)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def emit(b):
                self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
                self.wfile.flush()
            sent, fin = 0, None
            while fin is None:
                with LOCK:
                    ab = rid in ABORT
                if ab:
                    fin, chunk = {"type": "abort"}, []
                else:
                    chunk = [11] * min(8, total - sent)
                    sent += len(chunk)
                    if sent >= total:
                        fin = {"type": "length", "length": total}
                meta = {"id": rid, "finish_reason": fin, "prompt_tokens": len(ids), "completion_tokens": sent,
                        "cached_tokens": 0}
                emit(b"data: " + json.dumps({"output_ids": chunk, "meta_info": meta}).encode() + b"\n\n")
                if rid == "e11-smoke-b":
                    time.sleep(0.01)
            emit(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
            return
        greedy = float(sp.get("temperature", 1.0)) == 0.0
        cached = cached_for(ids)
        fin = decode_wait(rid, total, len(ids))
        done = total if fin != "abort" else max(1, total // 3)
        o = fake_ids(ids, done, greedy=greedy, rid=rid)
        remember(ids + o)
        meta = {"id": rid, "finish_reason": {"type": fin}, "prompt_tokens": len(ids), "completion_tokens": len(o),
                "cached_tokens": cached, "spec_accept_length": (1 + 0.47 * (WIDTH - 1)) if SPEC else None}
        if req.get("return_logprob"):
            meta["output_token_logprobs"] = [(-0.1, x, None) for x in o]
            meta["output_top_logprobs"] = [[(-0.1, x, None), (-2.5, x + 1, None)] for x in o]
        return self._send(json.dumps({"text": "", "output_ids": o, "meta_info": meta}))


ThreadingHTTPServer.request_queue_size = 256      # rev 2e-e: default 5 drops simultaneous connects (Windows)
srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
srv.daemon_threads = True
out(f"] The server is fired up and ready to roll! port {PORT}")
srv.serve_forever()
