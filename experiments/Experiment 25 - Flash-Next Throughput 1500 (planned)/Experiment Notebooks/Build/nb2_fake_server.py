"""NB2 local-harness fake of the Merlin vLLM 0.32 server (no model, no GPU), extended from the E23 / E14 fake.

Takes the REAL `vllm serve` flags (from Merlin's launch.build() in the local harness) and reads every flag with argparse's
last-occurrence rule, so trailing overrides behave as on the real engine. Prints the startup lines NB2 parses (formats
copied from the RC2 v1 FP8 server log), serves /health, /metrics (incl. the labelled per-position acceptance counter and a
block-based kv_cache_usage with retained finished turns), streaming /v1/completions, /start_profile + /stop_profile
(writes 4 per-rank torch-trace files), /reset_prefix_cache (only with VLLM_SERVER_DEV_MODE=1).

Env: FAKE_SPEED tok/s per request; NB2_FAKE_F fixed blocks per request (default 4 + K); NB2_FAKE_BLOCKS pool (149);
NB2_FAKE_FAIL_SEQS16=1 exits at startup when max-num-seqs > 8; NB2_FAKE_EOS_AFTER n = natural requests stop after n tokens;
NB2_FAKE_LOOP_SEEDS=s1,s2 = requests with one of these seeds loop on "row? " after NB2_FAKE_LOOP_AFTER (64) tokens.
Logprobs: with return_tokens_as_token_ids the top_logprobs keys are "token_id:N" (vLLM's form) with `logprobs` entries
(sampled -0.05, the rest -2.5, -3.0, ...); without it the old text-keyed two-entry form.
"""
import gzip
import json
import math
import os
import sys
import threading
import time
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]


def arg(name, default=None):
    """argparse semantics: the LAST occurrence wins."""
    idx = [i for i, a in enumerate(argv) if a == name]
    return argv[idx[-1] + 1] if idx and idx[-1] + 1 < len(argv) else default


PORT = int(arg("--port", 8011))
SEQS = int(arg("--max-num-seqs", 8))
MAXLEN = int(arg("--max-model-len", 131072))
SPEC = json.loads(arg("--speculative-config", "null") or "null")
K = int((SPEC or {}).get("num_speculative_tokens", 0))
COMP = json.loads(arg("--compilation-config", "null") or "null") or {}
CAPS = sorted(COMP.get("cudagraph_capture_sizes") or [1, 2, 4, 8])
PROF = json.loads(arg("--profiler-config", "null") or "null")
SPEED = float(os.environ.get("FAKE_SPEED", 400))
F = int(os.environ.get("NB2_FAKE_F") or (4 + K))
BLOCKS = int(os.environ.get("NB2_FAKE_BLOCKS") or 149)
EOS_AFTER = int(os.environ.get("NB2_FAKE_EOS_AFTER") or 0)
LOOP_SEEDS = {int(x) for x in (os.environ.get("NB2_FAKE_LOOP_SEEDS") or "").split(",") if x.strip()}
LOOP_AFTER = int(os.environ.get("NB2_FAKE_LOOP_AFTER") or 64)
RATES = [0.85, 0.68, 0.52, 0.40][:K]
print(f"fake vllm serve {' '.join(argv)}", flush=True)
if os.environ.get("NB2_FAKE_FAIL_SEQS16") == "1" and SEQS > 8:
    print("RuntimeError: fake refusal of max_num_seqs > 8", flush=True)
    sys.exit(3)
dups = sorted({a for a in argv if a.startswith("--") and argv.count(a) > 1})
if dups:
    print("(APIServer pid=1) WARNING 10-31 05:51:22 [argparse_utils.py:458] Found duplicate keys " + ", ".join(dups), flush=True)
nd = {"max_model_len": MAXLEN, "max_num_seqs": SEQS, "max_num_batched_tokens": int(arg("--max-num-batched-tokens", 1024))}
if SPEC:
    nd["speculative_config"] = SPEC
nd["compilation_config"] = {"cudagraph_mode": "<CUDAGraphMode.FULL_AND_PIECEWISE: (2, 1)>", "cudagraph_capture_sizes": CAPS}
print("(APIServer pid=1) INFO 10-31 05:51:22 [api_utils.py:286] non-default args: " + repr(nd).replace("'<", "<").replace(">'", ">"),
      flush=True)
spec_txt = f"SpeculativeConfig(method='mtp', model='/tmp/fn_merlin', num_spec_tokens={K})" if SPEC else "None"
print(f"(EngineCore pid=2) INFO 10-31 05:51:49 [core.py:225] Initializing a V1 LLM engine (v0.32.0) with config: model='/tmp/fn_merlin', "
      f"speculative_config={spec_txt}, tokenizer='/tmp/fn_merlin'", flush=True)
if SPEC:
    print("(APIServer pid=1) WARNING 10-31 05:51:25 [vllm.py:2311] max_num_scheduled_tokens is set to 1024 based on the speculative "
          "decoding settings. This may lead to suboptimal performance.", flush=True)
    print(f"(Worker_PP0_TP0_EP0 pid=10) INFO 10-31 05:52:07 [qwen_gdn_linear_attn.py:756] GDN RecoverSSM speculative verify active "
          f"(spec_query_len {K + 1})", flush=True)
else:
    print("(EngineCore pid=2) WARNING 10-31 05:51:49 [vllm.py:3400] --use-replayssm on Qwen4Exp selects GDN RecoverSSM, which needs "
          "speculative decoding; num_speculative_tokens=0, so the stock GDN path is used (as without --use-replayssm)", flush=True)
for w in ("PP0_TP0_EP0", "PP0_TP1_EP1", "PP1_TP0_EP0", "PP1_TP1_EP1"):
    print(f"(Worker_{w} pid=10) INFO 10-31 05:54:09 [model_runner.py:544] Model loading took 16.87 GiB memory and 124.66 seconds", flush=True)
    print(f"(Worker_{w} pid=10) INFO 10-31 06:06:58 [gpu_worker.py:942] Initial free memory 21.67 GiB, reserved 2.31 GiB memory for KV Cache",
          flush=True)
    print(f"(Worker_{w} pid=10) INFO 10-31 05:54:09 [interface.py:1121] Padding mamba page size by {21.01 - 2 * (3 - K):.2f}% to ensure "
          f"that mamba page size and attention page size are exactly equal.", flush=True)
if SPEC and SPEC.get("draft_lm_head_quant"):
    q = SPEC["draft_lm_head_quant"].upper()
    for w in ("PP1_TP0_EP0", "PP1_TP1_EP1"):
        print(f"(Worker_{w} pid=11) INFO 10-31 06:06:40 [draft_lm_head_quant.py:121] Draft lm_head: {q} weight-only copy of the "
              f"124160x2560 shard ({303.1 if q == 'INT8' else 151.6} MiB; the target keeps its torch.bfloat16 head)", flush=True)
tokens = BLOCKS * 4096 - 64 * F * 1000
print("(EngineCore pid=2) INFO 10-31 06:07:06 [kv_cache_utils.py:2826] Side-cache layout: on; per-worker block bytes unchanged "
      "[14680064, 16515072]; QSA ring group 3 packs 1024 requests per pool block", flush=True)
print(f"(EngineCore pid=2) INFO 10-31 06:07:06 [kv_cache_utils.py:2361] GPU KV cache size: {tokens:,} tokens, Maximum concurrency for "
      f"{MAXLEN:,} tokens per request: {tokens / MAXLEN:.2f}x", flush=True)
print(f"(EngineCore pid=2) INFO 10-31 06:07:06 [kv_cache_utils.py:854] kv cache group sizes (4096, 4096, 4096, {8 if K else 4}, 4096)",
      flush=True)
q = K + 1
full = sorted({-(-c // q) * q for c in CAPS if -(-c // q) * q <= min(SEQS * q, max(CAPS))})
print("(Worker_PP1_TP0_EP0 pid=11) INFO 10-31 06:07:28 [e2_guards.py:290] e2-guard[runtime]: resolved cudagraph modes before capture: "
      "target=FULL_AND_PIECEWISE" + (" draft.prefill_cudagraph_manager=FULL_AND_PIECEWISE draft.decode_cudagraph_manager=FULL_DECODE_ONLY."
                                     if SPEC else "."), flush=True)
print(f"Capturing CUDA graphs (PIECEWISE): 100%|##########| {len(CAPS)}/{len(CAPS)} [00:01<00:00]", flush=True)
print(f"Capturing CUDA graphs (FULL): 100%|##########| {len(full)}/{len(full)} [00:01<00:00]", flush=True)
if SPEC:
    print("(Worker_PP1_TP0_EP0 pid=11) INFO 10-31 06:07:32 [speculator.py:151] Capturing model for speculator...", flush=True)
    print("(Worker_PP1_TP0_EP0 pid=11) INFO 10-31 06:07:06 [e2_identity.py:262] e2-identity[PP1_TP0]: cudagraph_mode="
          "prefill:FULL_AND_PIECEWISE,decode:FULL_DECODE_ONLY role=draft", flush=True)
for w in ("PP0_TP0_EP0", "PP1_TP0_EP0"):
    print(f"(Worker_{w} pid=10) INFO 10-31 06:07:33 [model_runner.py:1341] Graph capturing finished in 4 secs, took 0.33 GiB", flush=True)
time.sleep(0.3)

LOCK = threading.Lock()
RUN, WAIT, RETAINED = {}, [], []          # RUN: rid -> context tokens; RETAINED: blocks of finished turns (<= 16)
CNT = {"gen": 0.0, "acc": 0.0, "drf": 0.0, "dtok": 0.0, "prompt": 0.0, "it": 0.0, "pos": [0.0] * K}
T_READY = time.time() + 0.5
_SYL = ["ka", "lo", "mi", "ne", "ru", "sa", "ti", "vo", "ze", "pa", "qu", "ri", "bel", "dor", "fen", "gul"]
_VOCAB = [a + b + c for a in _SYL for b in _SYL for c in _SYL]
_TOKID = {w + " ": 1000 + i for i, w in enumerate(_VOCAB)}


def text_for(prompt, n, greedy):
    import random
    h = hashlib.sha256((json.dumps(prompt[-64:]) + ("" if greedy else str(time.time()))).encode()).hexdigest()
    rng = random.Random(h)
    return [_VOCAB[rng.randrange(len(_VOCAB))] + " " for _ in range(n)]


def used_blocks():
    return sum(math.ceil(c / 4096) + F for c in RUN.values()) + sum(RETAINED)


def _spec_logger():
    last = (0.0, 0.0)
    while True:
        time.sleep(2)
        with LOCK:
            d, a = CNT["dtok"], CNT["acc"]
            pos = list(CNT["pos"])
        if K and d > last[0]:
            dd = (d - last[0]) / K
            rates = ", ".join(f"{r:.3f}" for r in RATES)
            print(f"(APIServer pid=1) INFO 10-31 06:07:57 [metrics.py:120] SpecDecoding metrics: Mean acceptance length: "
                  f"{1 + (a - last[1]) / max(dd, 1e-9):.2f}, Accepted throughput: 1.00 tokens/s, Drafted throughput: 1.00 tokens/s, "
                  f"Accepted: {int(a - last[1])} tokens, Drafted: {int(d - last[0])} tokens, Per-position acceptance rate: {rates}, "
                  f"Avg Draft acceptance rate: 60.0%", flush=True)
            last = (d, a)


threading.Thread(target=_spec_logger, daemon=True).start()


def write_traces():
    d = (PROF or {}).get("torch_profiler_dir")
    if not d:
        return
    os.makedirs(d, exist_ok=True)
    g = max(1, min(SEQS, 8) // 2)
    for pp in (0, 1):
        for tp in (0, 1):
            base = 1_700_000_000_000_000_000 + (pp * 7 + tp) * 1_000_000      # different per process, same absolute clock
            off = (pp * 7 + tp) * 1000.0                                          # us
            ev, c = [], 1
            for i in range(30):
                t = i * 40000.0 - off
                ev.append({"ph": "X", "cat": "user_annotation", "name": f"execute_context_0(0)_generation_{g}({g * q})",
                           "ts": t, "dur": 30000, "tid": 1, "pid": 1})
                n_graph = 1 + (K if (pp == 1 and K) else 0)
                for j in range(n_graph):
                    ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": t + 100 + j, "dur": 5, "tid": 1,
                               "args": {"correlation": c}})
                    k0 = t + 15000 * pp + 1000 * j
                    ev.append({"ph": "X", "cat": "kernel", "name": "void marlin_moe_wna16::Marlin<256>(int)" if j == 0 else
                               "void Marlin<128>(int)", "ts": k0, "dur": 20000 if j == 0 else 500, "args": {"correlation": c}})
                    c += 1
                if K:
                    ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": t + 200, "dur": 3, "tid": 1,
                               "args": {"correlation": c}})
                    ev.append({"ph": "X", "cat": "kernel", "name": "_recoverssm_commit_fold_kernel", "ts": t + 15000 * pp + 20500,
                               "dur": 100, "args": {"correlation": c}})
                    c += 1
            path = os.path.join(d, f"dp0_pp{pp}_tp{tp}_dcp0_ep{tp}_rank{2 * pp + tp}.{int(time.time())}.pt.trace.json.gz")
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump({"baseTimeNanoseconds": base, "traceEvents": ev}, fh)


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
                e = '{engine="0",model_name="q27"}'
                lines = [f"vllm:num_requests_running{e} {len(RUN)}", f"vllm:num_requests_waiting{e} {len(WAIT)}",
                         f"vllm:generation_tokens_total{e} {CNT['gen']}", f"vllm:prompt_tokens_total{e} {CNT['prompt']}",
                         f"vllm:kv_cache_usage_perc{e} {min(1.0, used_blocks() / BLOCKS):.6f}",
                         f"vllm:num_preemptions_total{e} 0", f"vllm:iteration_tokens_total_count{e} {CNT['it']}"]
                if K:
                    lines += [f"vllm:spec_decode_num_accepted_tokens_total{e} {CNT['acc']}",
                              f"vllm:spec_decode_num_drafts_total{e} {CNT['drf']}",
                              f"vllm:spec_decode_num_draft_tokens_total{e} {CNT['dtok']}"]
                    lines += [f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",model_name="q27",position="{i}"}} {v}'
                              for i, v in enumerate(CNT["pos"])]
            return self._send("\n".join(lines) + "\n", ctype="text/plain")
        return self._send("{}", 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) or b"{}"
        if self.path in ("/start_profile", "/stop_profile"):
            if not PROF:
                return self._send(json.dumps({"error": "Profiling is not enabled"}), 500)
            if self.path == "/stop_profile":
                write_traces()
            return self._send("{}")
        if self.path == "/reset_prefix_cache":
            if os.environ.get("VLLM_SERVER_DEV_MODE") != "1":
                return self._send("{}", 404)
            with LOCK:
                busy = bool(RUN)
                if not busy:
                    RETAINED.clear()
            return self._send(json.dumps({"success": not busy}))     # vLLM: HTTP 200 + {"success": bool}
        if self.path != "/v1/completions":
            return self._send("{}", 404)
        req = json.loads(raw)
        with LOCK:
            CNT["rid"] = CNT.get("rid", 0) + 1
            rid = f"r{CNT['rid']}"          # unique (time_ns() collides on Windows clocks)
        prompt, max_t = req.get("prompt") or [], int(req.get("max_tokens", 16))
        greedy = float(req.get("temperature", 1.0)) == 0.0
        nlp = int(req.get("logprobs") or 0)
        natural = not req.get("ignore_eos", False) and EOS_AFTER
        with LOCK:
            WAIT.append(rid)
        while True:
            with LOCK:
                if WAIT and WAIT[0] == rid and len(RUN) < SEQS:
                    need = math.ceil((len(prompt) + 1) / 4096) + F
                    while RETAINED and used_blocks() + need > BLOCKS:
                        RETAINED.pop(0)
                    if used_blocks() + need <= BLOCKS:
                        WAIT.pop(0)
                        RUN[rid] = len(prompt) + 1
                        break
            time.sleep(0.01)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        limit = min(max_t, EOS_AFTER) if natural else max_t
        toks = text_for(prompt, limit, greedy)
        if req.get("seed") in LOOP_SEEDS:          # DIAG harness: this request falls into a "row? row?" loop (token id 0)
            toks = toks[:LOOP_AFTER] + ["row? "] * max(0, limit - LOOP_AFTER)
        sent = 0
        try:
            while sent < limit:
                chunk = toks[sent:sent + 8]
                sent += len(chunk)
                fin = None if sent < limit else ("stop" if natural and limit < max_t else "length")
                d = {"choices": [{"text": "".join(chunk), "index": 0, "finish_reason": fin}]}
                if req.get("return_token_ids"):
                    d["choices"][0]["token_ids"] = [_TOKID.get(c, 0) for c in chunk]
                if nlp and req.get("return_tokens_as_token_ids"):    # vLLM: keys "token_id:N"; top-nlp entries (+ sampled)
                    tops = []
                    for c in chunk:
                        top = {f"token_id:{_TOKID.get(c, 0)}": -0.05}
                        for j in range(1, max(2, nlp)):
                            top[f"token_id:{900000 + j}"] = -2.5 - 0.5 * (j - 1)
                        tops.append(top)
                    d["choices"][0]["logprobs"] = {"tokens": [f"token_id:{_TOKID.get(c, 0)}" for c in chunk],
                                                   "token_logprobs": [-0.05] * len(chunk), "top_logprobs": tops}
                elif nlp:
                    d["choices"][0]["logprobs"] = {"tokens": list(chunk), "token_logprobs": [-0.05] * len(chunk),
                                                   "top_logprobs": [{c: -0.05, "x": -2.5} for c in chunk]}
                b = b"data: " + json.dumps(d).encode() + b"\n\n"
                self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
                self.wfile.flush()
                with LOCK:
                    CNT["gen"] += len(chunk)
                    if sent == len(chunk):
                        CNT["prompt"] += len(prompt)
                    steps = len(chunk) / (1 + sum(RATES))
                    CNT["it"] += steps / 2
                    if K:
                        CNT["drf"] += steps
                        CNT["dtok"] += steps * K
                        for i, r in enumerate(RATES):
                            CNT["pos"][i] += steps * r
                        CNT["acc"] += steps * sum(RATES)
                    RUN[rid] = RUN.get(rid, 0) + len(chunk)
                time.sleep(len(chunk) / SPEED if not greedy else 0.0)
            b = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n" + b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            with LOCK:
                ctx = RUN.pop(rid, None)
                if ctx:
                    RETAINED.append(math.ceil(ctx / 4096))
                    del RETAINED[:-16]


ThreadingHTTPServer.request_queue_size = 256
srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
srv.daemon_threads = True
print(f"INFO Started server process; Uvicorn running on http://127.0.0.1:{PORT}", flush=True)
srv.serve_forever()
