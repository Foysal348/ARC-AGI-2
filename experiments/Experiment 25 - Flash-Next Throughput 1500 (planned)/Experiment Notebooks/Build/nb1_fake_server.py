"""E25 NB1 local fake of the Merlin vLLM 0.32 server (no model, no GPU), for the CPU end-to-end harness only.

Extends the E23 fake (Experiment 23/src/harness/fake_vllm_server.py) with what NB1's new code reads:
- block-accurate KV accounting: pool P blocks of 4096 tokens (149 at the RC2 FP8 pin, 78 at the BF16 pin, scaled by
  --kv-cache-memory-bytes), F fixed blocks per request, and Merlin-style pinned finished turns (they show in
  vllm:kv_cache_usage_perc and are yielded under pressure), so the census can recover P and F;
- startup lines in the real Merlin wording (RC2 server_merlin_fp8.log): non-default args, Initial free memory, Model
  loading took, Padding mamba page size, GPU KV cache size + Maximum concurrency, graph capture, e2-guard modes,
  the --mamba-ssm-cache-dtype warning, the MERLIN_SHARE_DRAFT_WORKSPACE line;
- non-streaming /v1/completions with usage.prompt_tokens_details.cached_tokens (prefix reuse at prompt positions,
  block floor, as Merlin's "Mamba prefix checkpoints are published for prompt positions only" under PP), token_ids,
  prompt_logprobs (deterministic; a small FP8-dependent offset so FP8-vs-BF16 teacher-forced rows differ);
- /v1/chat/completions arithmetic probes; /start_profile + /stop_profile writing four per-rank torch traces
  (baseTimeNanoseconds, distributedInfo, execute_context scopes, graph launches) into --profiler-config's dir.
Arguments: the LAST occurrence of a flag wins (as argparse in vllm serve). FAKE_SPEED = tokens/s per request.
"""
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]


def arg(name, default=None):
    idx = [i for i, a in enumerate(argv) if a == name]
    return argv[idx[-1] + 1] if idx and idx[-1] + 1 < len(argv) else default


def first(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


PORT = int(arg("--port", 8011))
SEQS = int(arg("--max-num-seqs", 8))
BF16 = first("--max-num-seqs", "8") == "3"                     # RC2's fake branch: seqs 3 = the BF16 boot
KVB = int(arg("--kv-cache-memory-bytes", "0") or 0) or (2464415744 if BF16 else 2479838700)
BLOCK = int(arg("--block-size", 4096))
MAX_LEN = int(arg("--max-model-len", 131072))
SSM = arg("--mamba-ssm-cache-dtype", "auto")
LMO = "--language-model-only" in argv
F = 6
P = max(8, round((78 if BF16 else 149) * KVB / (2464415744 if BF16 else 2479838700)))
SPEED = float(os.environ.get("FAKE_SPEED", 300))
K = 3
PROF = json.loads(arg("--profiler-config", "null") or "null")
CAPTURE = (json.loads(arg("--compilation-config", "null") or "null") or {}).get("cudagraph_capture_sizes") or [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]


def say(s):
    print(s, flush=True)


say(f"fake merlin vllm serve {' '.join(argv)[:400]}")
nd = {"model_tag": "/tmp/fn_merlin", "max_model_len": MAX_LEN, "block_size": BLOCK, "max_num_seqs": SEQS,
      "max_num_batched_tokens": int(arg("--max-num-batched-tokens", 1024)), "kv_cache_memory_bytes": KVB,
      "kv_cache_dtype": "bfloat16" if BF16 else "fp8_e4m3", "mamba_cache_mode": "align",
      "speculative_config": {"method": "mtp", "num_speculative_tokens": K}}
if SSM != "auto":
    nd["mamba_ssm_cache_dtype"] = SSM
if LMO:
    nd["language_model_only"] = True
nd["compilation_config"] = {"cudagraph_capture_sizes": CAPTURE}
say(f"(APIServer pid={os.getpid()}) INFO [api_utils.py:286] non-default args: {nd!r}")
if SSM != "auto":
    say("(APIServer) WARNING [config.py:930] Qwen3.5 model specifies mamba_ssm_dtype='float32' in its config, but "
        f"--mamba-ssm-cache-dtype='{SSM}' was passed. Using the user-specified value.")
if not LMO:
    say("(APIServer) INFO [base.py:280] Multi-modal warmup completed in 0.5s")
for r in range(4):
    say(f"(Worker_PP{r // 2}_TP{r % 2}_EP{r % 2}) INFO [model_runner.py:544] Model loading took "
        f"{17.33 - (0.4 if LMO and r < 2 else 0) if r < 2 else 16.87:.2f} GiB memory and 1.5 seconds")
pad = 21.01 if SSM in ("auto", "float32") else 142.02
say(f"(Worker_PP0_TP0_EP0) INFO [interface.py:1121] Padding mamba page size by {pad:.2f}% to ensure that mamba page size "
    "and attention page size are exactly equal.")
for r in range(4):
    say(f"(Worker_PP{r // 2}_TP{r % 2}_EP{r % 2}) INFO [gpu_worker.py:942] Initial free memory 21.67 GiB, reserved "
        f"{KVB / 2 ** 30:.2f} GiB memory for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling.")
conc = P / (math.ceil(MAX_LEN / BLOCK) + F)
say(f"(EngineCore) INFO [kv_cache_utils.py:2361] GPU KV cache size: {int(conc * MAX_LEN):,} tokens, Maximum concurrency for "
    f"{MAX_LEN:,} tokens per request: {conc:.2f}x")
say("(EngineCore) INFO [kv_cache_utils.py:854] kv cache group sizes (4096, 4096, 4096, 8, 4096)")
if os.environ.get("MERLIN_SHARE_DRAFT_WORKSPACE") == "1":
    say("(Worker_PP1_TP0_EP0) INFO [draft_workspace.py:88] MERLIN_SHARE_DRAFT_WORKSPACE: drafter does NOT share the target "
        "FlashInfer workspace (0 B, target 0x0, draft 0x0)")
say("(Worker_PP1_TP0_EP0) INFO [e2_guards.py:290] e2-guard[runtime]: resolved cudagraph modes before capture: "
    "target=FULL_AND_PIECEWISE draft.prefill_cudagraph_manager=FULL_AND_PIECEWISE draft.decode_cudagraph_manager=FULL_DECODE_ONLY.")
nfull = len([c for c in CAPTURE if c % (K + 1) == 0])
say(f"Capturing CUDA graphs (PIECEWISE): 100%|##########| {len(CAPTURE)}/{len(CAPTURE)} [00:01<00:00]")
say(f"Capturing CUDA graphs (FULL): 100%|##########| {nfull}/{nfull} [00:01<00:00]")
say("(Worker_PP0_TP0_EP0) INFO [model_runner.py:1341] Graph capturing finished in 1 secs, took 0.33 GiB")
say("(Worker_PP1_TP0_EP0) INFO [model_runner.py:1341] Graph capturing finished in 1 secs, took 0.43 GiB")
say("(Worker_PP0_TP0_EP0) INFO [gdn_recoverssm.py:344] GDN RecoverSSM path taken: 8 spec rows, align=True")
if "--worker-extension-cls" in argv:
    say(f"(Worker pid=1) INFO [worker_base.py:330] Injected <class '{arg('--worker-extension-cls')}'> into Worker for extended collective_rpc calls ['nb1_cache_info']")
if os.environ.get("NCCL_DEBUG") == "INFO":
    for r in range(4):
        say(f"node:{r}:{r} [{r}] NCCL INFO comm 0x55aa{r} rank {r % 2} nranks 2 cudaDev {r} busId {r}000 - Init COMPLETE")
        say(f"node:{r}:{r} [{r}] NCCL INFO Channel 00/0 : {r % 2}[{r}] -> {(r + 1) % 2}[{r ^ 1}] via SHM/direct/direct")
    say("node:0:0 [0] NCCL INFO Using network Socket")
if not BF16:
    say("(Worker_PP0_TP0_EP0 pid=1) INFO [model.py:307] Qwen4Exp QSA: applied static K/V scales from /tmp/fn_merlin/qsa_kv_scales_262k.json to 6 of 12 layer(s) (pipeline layers 0-25): language_model.model.layers.3.self_attn.attn ...")
    say("(Worker_PP1_TP0_EP0 pid=3) INFO [qsa.py:506] Qwen4Exp QSA E4M3 KV active on mtp.layers.48.self_attn.attn: k_scale=1 v_scale=1  (DEFAULT 1.0 -- no static calibration loaded)")
COMMIT = {"mode": os.environ.get("VLLM_RECOVERSSM_COMMIT_GRAPH") or "eager", "eager_steps": 0, "staged_steps": 0, "replays": 0,
          "captures": 6 if os.environ.get("VLLM_RECOVERSSM_COMMIT_GRAPH") in ("graph", "staged") else 0, "fallbacks": {}}
JIT_DONE = set()

LOCK = threading.Lock()
RUN, WAIT = {}, []              # rid -> context tokens
PINNED = []                     # [(rid, blocks)] finished turns kept referenced (Merlin: up to 16)
CACHE = []                      # [(prompt tokens tuple, prompt_len)] prefix entries
CNT = {"gen": 0.0, "acc": 0.0, "drf": 0.0, "prompt": 0.0, "pq": 0.0, "ph": 0.0, "iter": 0.0, "p0": 0.0, "p1": 0.0, "p2": 0.0}
PROFILING = {"on": False, "t0": None}
T_READY = time.time() + 0.5


def blocks(ctx):
    return math.ceil(max(1, ctx) / BLOCK) + F


def used_blocks():
    return sum(blocks(c) for c in RUN.values()) + sum(b for _, b in PINNED)


_SYL = ["ka", "lo", "mi", "ne", "ru", "sa", "ti", "vo", "ze", "pa", "qu", "ri", "bel", "dor", "fen", "gul"]
_VOCAB = [a + b + c for a in _SYL for b in _SYL for c in _SYL]


SALT = {"n": 0}


def gen_ids(prompt, n, greedy, seed):
    salt = ""
    if greedy and os.environ.get("FAKE_GREEDY_SALT"):          # fault injection: greedy rows differ on every call
        SALT["n"] += 1
        salt = f"{os.environ['FAKE_GREEDY_SALT']}{SALT['n']}"
    h = hashlib.sha256((json.dumps(prompt[-64:]) + ("" if greedy else f"{seed}{time.time()}") + salt).encode()).hexdigest()
    rng = random.Random(h)
    return [1000 + rng.randrange(len(_VOCAB)) for _ in range(n)]


def text_of(ids):
    return "".join(_VOCAB[(i - 1000) % len(_VOCAB)] + " " for i in ids)


def cached_for(prompt):
    best = 0
    for toks, plen in CACHE:
        n = min(len(toks), len(prompt))
        lo = 0
        while lo < n and toks[lo:lo + 512] == tuple(prompt[lo:lo + 512]):
            lo += 512
        while lo < n and toks[lo] == prompt[lo]:
            lo += 1
        best = max(best, min(lo, plen) // BLOCK * BLOCK)      # prompt positions only, block floor
    return min(best, max(0, len(prompt) - 1))


def admit(rid, prompt):
    need = blocks(len(prompt))
    with LOCK:
        WAIT.append(rid)
    while True:
        with LOCK:
            if WAIT and WAIT[0] == rid and len(RUN) < SEQS:
                while PINNED and used_blocks() + need > P:      # need-sized yields of pinned finished turns
                    PINNED.pop(0)
                if used_blocks() + need <= P:
                    WAIT.pop(0)
                    RUN[rid] = len(prompt)
                    return
        time.sleep(0.01)


def finish(rid, toks, plen):
    with LOCK:
        ctx = RUN.pop(rid, None)
        if ctx is not None:
            PINNED.append((rid, blocks(ctx)))
            del PINNED[:-4]
        CACHE.append((tuple(toks), plen))
        del CACHE[:-48]


def prompt_logprobs(prompt):
    out = [None]
    for pos in range(1, len(prompt)):
        t = prompt[pos]
        lp = -(((t * 131 + pos * 7) % 50) / 25.0) - (0.01 * ((t + pos) % 3) if not BF16 else 0.0)
        rank = 1 if (t + pos) % 4 else 2
        d = {f"token_id:{t}": {"logprob": round(lp, 5), "rank": rank}}
        if rank != 1:
            d[f"token_id:{t + 1}"] = {"logprob": round(lp + 0.2, 5), "rank": 1}
        out.append(d)
    return out


def write_traces():
    d = (PROF or {}).get("torch_profiler_dir")
    if not d:
        return
    os.makedirs(d, exist_ok=True)
    base = time.time_ns()
    for r in range(4):
        ev, corr, t = [], 1, 0.0
        stage = r // 2
        for it in range(20):
            start = t + (10000.0 if stage else 0.0)
            ev.append({"ph": "X", "cat": "user_annotation", "name": "execute_context_0(0)_generation_4(16)", "ts": start,
                       "dur": 30000.0, "tid": 1, "pid": r})
            for j, kn in enumerate(("void (anonymous namespace)::gdn_decode_post_conv_mtp_kernel<float, 3>(...)",
                                    "fused_marlin_moe_kernel", "qsa_sparse_attn_split_k", "ncclDevKernel_AllReduce_Sum_bf16",
                                    "persistent_topk_kernel")):
                ts = start + 100 + j * 5000
                ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": ts, "dur": 5.0, "tid": 1,
                           "pid": r, "args": {"correlation": corr}})
                ev.append({"ph": "X", "cat": "kernel", "name": kn, "ts": ts + 10, "dur": 4000.0, "tid": 7, "pid": r,
                           "args": {"correlation": corr}})
                corr += 1
            t += 40000.0
        tr = {"traceEvents": ev, "baseTimeNanoseconds": base, "distributedInfo": {"rank": r, "world_size": 4}}
        with gzip.open(os.path.join(d, f"rank{r}.{base}.pt.trace.json.gz"), "wt") as fh:
            json.dump(tr, fh)


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
                lines = [f'vllm:num_requests_running{{engine="0"}} {len(RUN)}',
                         f'vllm:num_requests_waiting{{engine="0"}} {len(WAIT)}',
                         f'vllm:generation_tokens_total{{engine="0"}} {CNT["gen"]}',
                         f'vllm:spec_decode_num_accepted_tokens_total{{engine="0"}} {CNT["acc"]}',
                         f'vllm:spec_decode_num_drafts_total{{engine="0"}} {CNT["drf"]}',
                         f'vllm:spec_decode_num_draft_tokens_total{{engine="0"}} {CNT["drf"] * K}',
                         f'vllm:kv_cache_usage_perc{{engine="0"}} {min(1.0, used_blocks() / P)}',
                         'vllm:num_preemptions_total{engine="0"} 0',
                         f'vllm:prompt_tokens_total{{engine="0"}} {CNT["prompt"]}',
                         f'vllm:prefix_cache_queries_total{{engine="0"}} {CNT["pq"]}',
                         f'vllm:prefix_cache_hits_total{{engine="0"}} {CNT["ph"]}',
                         f'vllm:iteration_tokens_total_count{{engine="0"}} {CNT["iter"]}'] + [
                         f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{engine="0",position="{i}"}} {CNT["p%d" % i]}' for i in range(3)]
            return self._send("\n".join(lines) + "\n", ctype="text/plain")
        return self._send("{}", 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) or b"{}"
        try:
            req = json.loads(raw)
        except Exception:
            req = {}
        if self.path.startswith("/reset_prefix_cache"):
            with LOCK:
                ok = not RUN and not WAIT
                if ok:
                    CACHE.clear()
                    PINNED.clear()
            return self._send(json.dumps({"success": ok}), 200)
        if self.path == "/collective_rpc":
            meth, args = req.get("method"), req.get("args") or []
            if meth == "nb1_cache_info":
                ssm = "bfloat16" if SSM == "bfloat16" else "float32"
                res = [{"rank": r, "num_gpu_blocks": P + 1, "mamba_ssm_cache_dtype": SSM, "cache_block_size": BLOCK,
                        "cache_tensors": [{"module": "QwenGDNLinearAttention", "layers": 9 if r < 2 else 8, "example": "gdn",
                                           "tensors": [{"dtype": "bfloat16", "shape": [P + 1, 16, 3], "bytes": 1},
                                                       {"dtype": ssm, "shape": [P + 1, 16, 128, 128], "bytes": 1}]},
                                          {"module": "Qwen4ExpQSAAttention", "layers": 6, "example": "qsa",
                                           "tensors": [{"dtype": "uint8" if not BF16 else "bfloat16", "shape": [P + 1, 4096, 256], "bytes": 1}]}],
                        "kv_cache_specs": [{"type": "MambaSpec", "block_size": BLOCK, "page_size_bytes": 16515072, "layers": 12,
                                            "state_dtypes": ["bfloat16", ssm], "unpadded_page_bytes": int(16515072 / (1 + pad / 100))}],
                        "mem": {"free_mib": 1600 - 10 * r, "total_mib": 23034}} for r in range(4)]
                return self._send(json.dumps({"results": res}))
            if meth == "nb1_routing":
                act = args[0] if args else ""
                res = []
                for r in range(4):
                    d = {"action": act, "rank": r, "passes": 0 if act == "start" else 4}
                    if act == "stop" and len(args) > 1:
                        try:                       # the same part-file layout the worker extension writes
                            import numpy as np
                            rows, offs, o_ = [], [], 0
                            for ps in range(4):
                                for lay in ((3, 7) if r < 2 else (27, 31)):
                                    rows.append((lay, ps, 16, (np.arange(160, dtype=np.int16).reshape(16, 10) * (lay + 1)) % 512))
                            for x in rows:
                                offs.append(o_)
                                o_ += x[2]
                            path = f"{args[1]}_rank{r}.npz"
                            np.savez_compressed(path, ids=np.concatenate([x[3] for x in rows]), layer=np.array([x[0] for x in rows]),
                                                pass_index=np.array([x[1] for x in rows]), mb_size=np.array([x[2] for x in rows]),
                                                offset=np.array(offs), rank=np.array([r]))
                            d["path"] = path
                        except Exception as e:
                            d["error"] = repr(e)[:120]
                    res.append(d)
                return self._send(json.dumps({"results": res}))
            if meth in ("set_recoverssm_commit_mode", "get_recoverssm_commit_stats"):
                with LOCK:
                    snap = {k: v for k, v in COMMIT.items() if k != "mode"}
                    if meth.startswith("set") and args:
                        COMMIT["mode"] = args[0]
                return self._send(json.dumps({"results": [dict(snap, mode=COMMIT["mode"]) for _ in range(4)]}))
            return self._send(json.dumps({"detail": "unknown method"}), 400)
        if self.path == "/start_profile":
            PROFILING.update(on=True, t0=time.time())
            return self._send("", 200)
        if self.path == "/stop_profile":
            PROFILING["on"] = False
            write_traces()
            return self._send("", 200)
        if self.path == "/v1/chat/completions":
            q = (req.get("messages") or [{}])[-1].get("content", "")
            m = re.search(r"(\d+)\s*([*+-])\s*(\d+)", q)
            ans = str(eval(m.group(0))) if m else "?"
            return self._send(json.dumps({"choices": [{"message": {"content": ans, "reasoning": ""}, "finish_reason": "stop"}]}))
        if self.path != "/v1/completions":
            return self._send("{}", 404)
        prompt, max_t = list(req.get("prompt") or []), int(req.get("max_tokens", 16))
        greedy = float(req.get("temperature", 1.0)) == 0.0
        nlp = int(req.get("logprobs") or 0)
        rid = f"r{time.time_ns()}"
        plp = req.get("prompt_logprobs") is not None
        with LOCK:
            hit = 0 if plp else cached_for(prompt)
        admit(rid, prompt)
        with LOCK:
            CNT["prompt"] += len(prompt) - hit
            CNT["pq"] += len(prompt)
            CNT["ph"] += hit
        ids = gen_ids(prompt, max_t, greedy, req.get("seed"))
        with LOCK:
            k_ = "eager_steps" if COMMIT["mode"] == "eager" else "staged_steps"
            COMMIT[k_] += max(1, max_t // 4)
            COMMIT["fallbacks"]["pp_deferred"] = COMMIT["fallbacks"].get("pp_deferred", 0) + (max(1, max_t // 4) if COMMIT["mode"] == "graph" else 0)
            if len(RUN) > 1 and "_fill_num_accepted_kernel" not in JIT_DONE:
                JIT_DONE.add("_fill_num_accepted_kernel")
                say("(Worker_PP0_TP0_EP0 pid=1) WARNING [jit_monitor.py:140] Triton kernel JIT compilation during inference: "
                    "_fill_num_accepted_kernel. This causes a latency spike; consider extending warmup to cover this shape/config.")
        if not req.get("stream"):
            try:
                if not greedy:
                    time.sleep(min(30.0, max_t / SPEED))
                with LOCK:
                    CNT["gen"] += max_t
                    CNT["drf"] += max_t / (1 + 2.2)
                    CNT["acc"] += max_t - max_t / (1 + 2.2)
                    CNT["iter"] += max(1, max_t // 4)
                ch = {"index": 0, "text": text_of(ids), "token_ids": ids if req.get("return_token_ids") else None,
                      "finish_reason": "length"}
                if nlp:
                    ch["logprobs"] = {"tokens": [f"token_id:{c}" for c in ids], "token_logprobs": [-0.05 - (c % 7) / 100 for c in ids],
                                      "top_logprobs": [{f"token_id:{c}": -0.05 - (c % 7) / 100, "token_id:1": -2.5} for c in ids]}
                if plp:
                    ch["prompt_logprobs"] = prompt_logprobs(prompt)
                body = {"choices": [ch], "usage": {"prompt_tokens": len(prompt), "completion_tokens": max_t,
                                                   "prompt_tokens_details": {"cached_tokens": hit}}}
                return self._send(json.dumps(body))
            finally:
                finish(rid, prompt + ids, len(prompt))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        sent = 0
        try:
            while sent < max_t:
                chunk = ids[sent:sent + 8]
                sent += len(chunk)
                d = {"choices": [{"text": text_of(chunk), "index": 0, "finish_reason": "length" if sent >= max_t else None}]}
                if req.get("return_token_ids"):
                    d["choices"][0]["token_ids"] = chunk
                if nlp:
                    d["choices"][0]["logprobs"] = {"tokens": [text_of([c]) for c in chunk], "token_logprobs": [-0.05] * len(chunk),
                                                   "top_logprobs": [{f"token_id:{c}": -0.05, "token_id:1": -2.5} for c in chunk]}
                b = b"data: " + json.dumps(d).encode() + b"\n\n"
                self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
                self.wfile.flush()
                with LOCK:
                    CNT["gen"] += len(chunk)
                    CNT["drf"] += len(chunk) / (1 + 2.2)
                    CNT["acc"] += len(chunk) - len(chunk) / (1 + 2.2)
                    for i, f_ in enumerate((0.85, 0.7, 0.55)):
                        CNT["p%d" % i] += f_ * len(chunk) / (1 + 2.2)
                    CNT["iter"] += 1
                    if rid in RUN:
                        RUN[rid] += len(chunk)
                time.sleep(len(chunk) / SPEED if not greedy else 0.0)
            b = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n" + b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            finish(rid, prompt + ids[:sent], len(prompt))


ThreadingHTTPServer.request_queue_size = 256
srv = ThreadingHTTPServer(("127.0.0.1", PORT), H)
srv.daemon_threads = True
say(f"INFO Started server process; Uvicorn running on http://127.0.0.1:{PORT}")
srv.serve_forever()
