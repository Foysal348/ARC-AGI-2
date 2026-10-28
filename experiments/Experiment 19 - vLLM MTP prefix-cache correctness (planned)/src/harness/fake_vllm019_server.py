"""Fake vLLM 0.19.0 OpenAI server for the E19 CPU harness (no model, no GPU). Launched by the REAL E19 core cell with the
REAL api_server flags (argv = --model X --flag value ...). It imitates what E19 measures:
- startup lines the cell parses (API server version, attention block size, mamba cache mode, GPU KV cache size,
  maximum concurrency); GET /health; GET /metrics (prefix-cache queries / hits, generation tokens, preemptions,
  spec-decode drafts / accepted, running); POST /reset_prefix_cache (only with VLLM_SERVER_DEV_MODE=1)
- POST /v1/completions stream=true with prompt token ids: SSE chunks with integer token_ids and logprobs
  (top_logprobs keyed 'token_id:N' when return_tokens_as_token_ids), a finish_reason, a usage chunk when
  stream_options.include_usage (prompt_tokens_details.cached_tokens only when > 0 and --enable-prompt-tokens-details,
  as vLLM 0.19), then [DONE]
- a deterministic 'model': the next token is a hash of the last 8 context tokens, so a continuation request over the
  server's own tokens reproduces the natural continuation; near-tie positions (margin 0.03) flip benignly when MTP is
  on, when other requests run concurrently (random) or when the prompt came from the cache
- a block-hash prefix cache (block 1,600 for fp8 KV, 800 for bf16); generated blocks are cached only without MTP
  (align mode + MTP rarely lands on a block boundary, E18 finding)
Scenario (env E19_FAKE_SCEN): normal | launchfail (the MTP + prefix-off arm exits at start) | death (the S1 arm dies
mid-stream after 14 completions) | corrupt (the S1 arm emits '!' x N with a 9-nat margin on every cache hit) |
race (every MTP arm WITH async scheduling: when the running batch changed since the request's previous token, 3 % of
positions emit a confidently wrong token, 6-nat margin; serial requests never see it) | collapse (the S1 arm reports
0 accepted MTP tokens) | kvretry (bf16 KV at max_model_len 106,496 fails like vLLM's KV check; 65,536 loads) |
slow (2 ms per token)."""
import hashlib
import json
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]


def arg(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


SCEN = os.environ.get("E19_FAKE_SCEN", "normal")
LOG = os.environ.get("E19_FAKE_LOG")
PORT = int(arg("--port", 8019))
SPEC = "--speculative-config" in argv
PREFIX = "--enable-prefix-caching" in argv
KV = arg("--kv-cache-dtype", "auto")
MAXLEN = int(arg("--max-model-len", 106496))
ASYNC = "--no-async-scheduling" not in argv
S1_ARM = SPEC and PREFIX and KV == "fp8_e4m3" and ASYNC
BS = 1600 if KV.startswith("fp8") else 800
print(f"fake api_server {' '.join(argv)}", flush=True)
print("(APIServer pid=1) INFO [api_server.py:1] vLLM API server version 0.19.0", flush=True)
for k, v in {"VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1", "VLLM_SERVER_DEV_MODE": "1"}.items():
    if os.environ.get(k) != v:
        print(f"ValueError: E19 harness: env {k}={os.environ.get(k)!r}", flush=True)
        sys.exit(1)
if SCEN == "launchfail" and SPEC and not PREFIX:
    print("RuntimeError: simulated: engine core failed to start", flush=True)
    sys.exit(1)
print(f"INFO [config.py:281] Setting attention block size to {BS} tokens to ensure that attention page size is >= mamba page size.", flush=True)
print("INFO [scheduler.py:1] Asynchronous scheduling is %s." % ("enabled" if ASYNC else "disabled"), flush=True)
if PREFIX:
    print("WARNING [config.py:441] Mamba cache mode is set to 'align' for Qwen3_5ForCausalLM by default when prefix caching is enabled", flush=True)
if SCEN == "kvretry" and KV == "auto" and MAXLEN > 65536:
    print("ValueError: To serve at least one request with the models's max seq len (106496), (12.5 GiB KV cache is needed, "
          "which is larger than the available KV cache memory (9.1 GiB).", flush=True)
    sys.exit(1)
KV_TOKENS = 312000 if KV.startswith("fp8") else 156000
time.sleep(0.3)
print(f"(EngineCore pid=2) INFO [kv_cache_utils.py:1] GPU KV cache size: {KV_TOKENS:,} tokens", flush=True)
print(f"(EngineCore pid=2) INFO [kv_cache_utils.py:2] Maximum concurrency for {MAXLEN:,} tokens per request: "
      f"{(KV_TOKENS * 4) / MAXLEN:.2f}x", flush=True)

import tokenizers  # noqa: E402
TOK = tokenizers.Tokenizer.from_file(os.path.join(arg("--model"), "tokenizer.json"))
_par = ("the grid shows a pattern of colored cells where each object moves toward the nearest border and keeps its "
        "shape while the background stays black so the rule must map every input row to an output row by checking "
        "neighbors counting components comparing sizes rotating mirrored copies and filling enclosed regions with "
        "the color of the marker cell then verify against all training pairs before writing the final code answer")
WORDS = sorted(set(TOK.encode(" " + _par, add_special_tokens=False).ids))
BANG = TOK.encode("!", add_special_tokens=False).ids[0]
LOCK = threading.Lock()
CACHE = set()
ST = {"running": 0, "gen": 0, "prompt": 0, "pc_q": 0, "pc_h": 0, "drafts": 0, "dtok": 0, "acc": 0, "joins": 0, "completions": 0, "resets": 0,
      "requests": [], "max_running": 0}


def h64(*xs):
    return int.from_bytes(hashlib.blake2b(repr(xs).encode(), digest_size=8).digest(), "little")


def block_hashes(ids):
    out, prev = [], 0
    for k in range(len(ids) // BS):
        prev = h64(prev, tuple(ids[k * BS:(k + 1) * BS]))
        out.append(prev)
    return out


def logrec(rec):
    with LOCK:
        ST["requests"].append(rec)
        if LOG:
            with open(LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")


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
            return self._send("")
        if self.path == "/metrics":
            with LOCK:
                txt = (f'vllm:num_requests_running{{engine="0"}} {ST["running"]}\n'
                       f'vllm:generation_tokens_total{{engine="0"}} {ST["gen"]}\n'
                       f'vllm:generation_tokens_created{{engine="0"}} 1.7e9\n'
                       f'vllm:prompt_tokens_total{{engine="0"}} {ST["prompt"]}\n'
                       f'vllm:prefix_cache_queries_total{{engine="0"}} {ST["pc_q"]}\n'
                       f'vllm:prefix_cache_hits_total{{engine="0"}} {ST["pc_h"]}\n'
                       f'vllm:num_preemptions_total{{engine="0"}} 0\n'
                       f'vllm:spec_decode_num_drafts_total{{engine="0"}} {ST["drafts"]}\n'
                       f'vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {ST["dtok"]}\n'
                       f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {ST["acc"]}\n')
            return self._send(txt, ctype="text/plain")
        self._send('{"error": "not found"}', 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) or b"{}"
        if self.path.startswith("/reset_prefix_cache"):
            with LOCK:
                CACHE.clear()
                ST["resets"] += 1
            logrec({"reset": True})
            return self._send("")
        if self.path != "/v1/completions":
            return self._send('{"error": "not found"}', 404)
        body = json.loads(raw)
        ids = list(body["prompt"])
        cap = int(body["max_tokens"])
        rec = {k: body.get(k) for k in ("temperature", "top_p", "logprobs", "return_token_ids", "return_tokens_as_token_ids",
                                         "stream", "stream_options", "skip_special_tokens", "ignore_eos", "model", "top_k", "seed")}
        rec.update(len=len(ids), max_tokens=cap, sha=hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16])
        if len(ids) + cap > MAXLEN:
            return self._send(json.dumps({"error": {"message": "context too long"}}), 400)
        greedy = float(body.get("temperature") or 0.0) == 0.0
        as_id = bool(body.get("return_tokens_as_token_ids"))
        nlp = int(body.get("logprobs") or 0)
        bh = block_hashes(ids) if PREFIX else []
        with LOCK:
            hit = 0
            if PREFIX:
                for k, hh in enumerate(bh):
                    if (k + 1) * BS > len(ids) - 1 or hh not in CACHE:
                        break
                    hit = (k + 1) * BS
                CACHE.update(bh)
            ST["pc_q"] += len(ids) if PREFIX else 0
            ST["pc_h"] += hit
            ST["prompt"] += len(ids) - hit
            ST["completions"] += 1
            my_no = ST["completions"]
            ST["running"] += 1
            ST["joins"] += 1
            ST["max_running"] = max(ST["max_running"], ST["running"])
        rec.update(cached=hit, no=my_no)
        die_now = SCEN == "death" and S1_ARM and my_no == 14
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        ctx, out, reason = list(ids), [], "length"
        rng = random.Random()
        sleep = 0.002 if SCEN == "slow" else 0.0005

        def emit(d):
            self.wfile.write(b"data: " + json.dumps(d).encode() + b"\n\n")
            self.wfile.flush()
        with LOCK:
            seen = (ST["joins"], ST["running"])
        try:
            buf_ids, buf_lp = [], []
            for j in range(cap):
                hh = h64(tuple(ctx[-8:]))
                p, alt = WORDS[hh % len(WORDS)], WORDS[(hh // 7919) % len(WORDS)]
                if alt == p:
                    alt = WORDS[(hh // 7919 + 1) % len(WORDS)]
                near = (hh // 104729) % 23 == 0
                margin = 0.03 if near else 1.5 + ((hh // 13) % 700) / 100.0
                tok = p
                if near:
                    with LOCK:
                        conc = ST["running"] > 1
                    if (SPEC and (hh // 31) % 3 == 0) or (conc and rng.random() < 0.3) or (hit and (hh // 37) % 5 == 0):
                        tok = alt
                top = {tok: -0.05, (alt if tok == p else p): -0.05 - margin}
                if SCEN == "race" and SPEC and ASYNC:
                    with LOCK:
                        now = (ST["joins"], ST["running"])
                    if now != seen and rng.random() < 0.03:
                        tok, top = alt, {alt: -0.01, p: -6.0}      # a row moved: a confidently wrong token
                    seen = now
                if SCEN == "corrupt" and S1_ARM and hit and j >= 24:
                    tok, top = BANG, {BANG: -0.0001, p: -9.0}
                ctx.append(tok)
                out.append(tok)
                buf_ids.append(tok)
                buf_lp.append({(f"token_id:{k}" if as_id else TOK.decode([k])): v for k, v in top.items()})
                if PREFIX and not SPEC and len(ctx) % BS == 0:
                    with LOCK:
                        CACHE.add(block_hashes(ctx)[-1])
                last = j == cap - 1
                if len(buf_ids) == 8 or last:
                    ch = {"index": 0, "text": TOK.decode(buf_ids), "finish_reason": reason if last else None}
                    if body.get("return_token_ids"):
                        ch["token_ids"] = buf_ids
                    if nlp:
                        ch["logprobs"] = {"tokens": [f"token_id:{k}" for k in buf_ids], "token_logprobs": [-0.05] * len(buf_ids),
                                          "top_logprobs": buf_lp, "text_offset": [0] * len(buf_ids)}
                    emit({"id": "cmpl-x", "object": "text_completion", "choices": [ch]})
                    buf_ids, buf_lp = [], []
                    time.sleep(sleep * 8)
                if die_now and j == 200:
                    with open(LOG + ".died", "w") as f:
                        f.write("died")
                    os._exit(1)
            if (body.get("stream_options") or {}).get("include_usage"):
                u = {"prompt_tokens": len(ids), "completion_tokens": len(out), "total_tokens": len(ids) + len(out)}
                if hit and "--enable-prompt-tokens-details" in argv:
                    u["prompt_tokens_details"] = {"cached_tokens": hit}
                emit({"id": "cmpl-x", "object": "text_completion", "choices": [], "usage": u})
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            rec["outcome"] = reason
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            rec["outcome"] = "disconnect"
        finally:
            with LOCK:
                ST["running"] -= 1
                ST["gen"] += len(out)
                if SPEC:
                    ST["drafts"] += len(out) // 3
                    ST["dtok"] += len(out)
                    ST["acc"] += 0 if (SCEN == "collapse" and S1_ARM) else int(len(out) * 0.55)
            rec["out"] = len(out)
            rec["greedy"] = greedy
            logrec(rec)
        self.close_connection = True


class S(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 256


S(("127.0.0.1", PORT), H).serve_forever()
