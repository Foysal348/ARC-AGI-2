"""E25 NB0 local fake of the Merlin vLLM 0.32 server (no model, no GPU), for the CPU end-to-end harness only.

What NB0 reads from a server, in the real Merlin wording (RC2 server_merlin_{bf16,fp8}.log):
- startup: `non-default args: {...}` (kv_cache_dtype, max_num_seqs, speculative_config, cudagraph_capture_sizes), the
  `Initializing a V1 LLM engine ... speculative_config=...` line, the MTP-off warning `--use-replayssm on Qwen4Exp selects
  GDN RecoverSSM, which needs speculative decoding; num_speculative_tokens=0, ...`, `GDN RecoverSSM path taken` (MTP on),
  the static-scale lines (`applied static K/V scales ... to 6 of 12 layer(s) (pipeline layers a-b)` per stage, `QSA E4M3
  KV active on <layer>: k_scale=.. v_scale=..`, the MTP layer's DEFAULT 1.0 line), `GPU KV cache size`, per-worker
  `Model loading took`, and the NB0 boot-E marker when NB0_QSA_FORCE_BYTE_DECODER=1;
- 4 worker child processes (they exit when this process exits: stdin EOF) so the session scan has real pids on Linux;
- /health, /metrics, /v1/chat/completions (arithmetic), streaming and non-streaming /v1/completions with token_ids,
  logprobs (top-k), prompt_logprobs (top-k + actual token, rank), usage.prompt_tokens_details.cached_tokens (block-4096
  prefix cache; prompt_logprobs requests never READ the cache, as sampling_params.py:622-629, but write it).
Deterministic token model: the top-1 token at a position depends only on the previous token, so a greedy decode and a
teacher-forced pass over prompt + decode agree exactly unless a fault mode says otherwise.
FAKE_TF_MODE: clean | kv | kv_long | mtp | native | path | shift | repeat (see perturb()).
Arguments: the LAST occurrence of a flag wins (argparse); `--speculative-config None` or no flag = MTP off.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

argv = sys.argv[1:]


def arg(name, default=None):
    idx = [i for i, a in enumerate(argv) if a == name]
    return argv[idx[-1] + 1] if idx and idx[-1] + 1 < len(argv) else default


PORT = int(arg("--port", 8010))
KV = arg("--kv-cache-dtype", "fp8_e4m3")
FP8 = KV.startswith("fp8")
SEQS = int(arg("--max-num-seqs", 8))
_spec = arg("--speculative-config", None)
SPEC = json.loads(_spec) if _spec not in (None, "", "None") else None
MTP = bool(SPEC and int(SPEC.get("num_speculative_tokens", 0)) > 0)
CAPTURE = (json.loads(arg("--compilation-config", "null") or "null") or {}).get("cudagraph_capture_sizes") or []
BYTE = os.environ.get("NB0_QSA_FORCE_BYTE_DECODER") == "1"
MODE = os.environ.get("FAKE_TF_MODE", "clean")
BOOT_SEC = float(os.environ.get("FAKE_BOOT_SEC", "1.0"))
VOCAB = 248000
BLOCK = 4096
LOCK = threading.Lock()
CACHE = set()                    # block-chain hashes of cached prompt prefixes
SEEN = {}                        # prompt sha -> teacher-forced count (repeat mode)


def say(s):
    print(s, flush=True)


def ts():
    return time.strftime("%m-%d %H:%M:%S")


# ------------------------------------------------------------------ startup lines
nd = {"kv_cache_dtype": KV, "max_num_seqs": SEQS, "enable_prefix_caching": "--enable-prefix-caching" in argv,
      "port": PORT, "compilation_config": {"cudagraph_capture_sizes": CAPTURE}}
if SPEC:
    nd["speculative_config"] = SPEC
say(f"(APIServer pid={os.getpid()}) INFO {ts()} [api_utils.py:286] non-default args: {nd!r}")
if not MTP and "--use-replayssm" in argv:
    say(f"(APIServer pid={os.getpid()}) WARNING {ts()} [vllm.py:3389] --use-replayssm on Qwen4Exp selects GDN RecoverSSM, "
        "which needs speculative decoding; num_speculative_tokens=0, so the stock GDN path is used (as without --use-replayssm)")
spec_s = (f"SpeculativeConfig(method='mtp', model='/tmp/fn_merlin', num_spec_tokens={SPEC['num_speculative_tokens']})"
          if MTP else "None")
say(f"(EngineCore pid={os.getpid()}) INFO {ts()} [core.py:225] Initializing a V1 LLM engine (v0.32.0) with config: "
    f"model='/tmp/fn_merlin', speculative_config={spec_s}, tokenizer='/tmp/fn_merlin', kv_cache_dtype={KV}, "
    f"enable_prefix_caching=True")
WORKERS = []
NAMES = ["Worker_PP0_TP0_EP0", "Worker_PP0_TP1_EP1", "Worker_PP1_TP0_EP0", "Worker_PP1_TP1_EP1"]
for n in NAMES:
    try:
        w = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)   # never share the log handle
    except Exception:
        w = None
    WORKERS.append(w)
SCALES = os.environ.get("VLLM_QSA_KV_SCALES", "")
sc = json.load(open(SCALES)) if (FP8 and SCALES and os.path.isfile(SCALES)) else {}
for name, w in zip(NAMES, WORKERS):
    pid = w.pid if w else 0
    if BYTE:
        say(f"({name} pid={pid}) NB0 boot E: QSA E4M3 byte decoder forced (qsa_e4m3_uses_native_cast -> False) pid={pid}")
    if sc:
        pp1 = "PP1" in name
        lay = sorted((k for k in sc if (int(re.search(r"layers\.(\d+)\.", k).group(1)) >= 25) == pp1),
                     key=lambda k: int(re.search(r"layers\.(\d+)\.", k).group(1)))
        say(f"({name} pid={pid}) INFO {ts()} [model.py:307] Qwen4Exp QSA: applied static K/V scales from {SCALES} to "
            f"{len(lay)} of {len(sc)} layer(s) (pipeline layers {'25-48' if pp1 else '0-25'}): {', '.join(lay[:4])} ...")
    say(f"({name} pid={pid}) INFO {ts()} [model_runner.py:544] Model loading took "
        f"{'16.87' if 'PP1' in name else '17.33'} GiB memory and {BOOT_SEC:.6f} seconds")
pool = 513940 if FP8 else 269042
say(f"(EngineCore pid={os.getpid()}) INFO {ts()} [kv_cache_utils.py:1300] GPU KV cache size: {pool:,} tokens")
say(f"(EngineCore pid={os.getpid()}) INFO {ts()} [kv_cache_utils.py:1305] Maximum concurrency for 131,072 tokens per "
    f"request: {pool / 131072 / 1.0:.2f}x")
for name, w in zip(NAMES, WORKERS):
    pid = w.pid if w else 0
    if sc:
        for k in sorted(sc, key=lambda k: int(re.search(r"layers\.(\d+)\.", k).group(1))):
            if (int(re.search(r"layers\.(\d+)\.", k).group(1)) >= 25) == ("PP1" in name):
                say(f"({name} pid={pid}) INFO {ts()} [qsa.py:506] Qwen4Exp QSA E4M3 KV active on {k}: "
                    f"k_scale={sc[k]['k_scale']:.6g} v_scale={sc[k]['v_scale']:.6g}")
        if MTP and "PP1" in name:
            say(f"({name} pid={pid}) INFO {ts()} [qsa.py:506] Qwen4Exp QSA E4M3 KV active on mtp.layers.48.self_attn.attn: "
                "k_scale=1 v_scale=1  (DEFAULT 1.0 -- no static calibration loaded)")
    if MTP:
        say(f"({name} pid={pid}) INFO {ts()} [gdn_recoverssm.py:344] GDN RecoverSSM path taken: {SEQS} spec rows, align=True")


# ------------------------------------------------------------------ the token model
def top1(prev):
    return (prev * 2654435761 + 97) % 200000 + 1000


def h(*xs):
    return int(hashlib.md5(repr(xs).encode()).hexdigest()[:8], 16)


def diverges(p, tf):
    """True where this server's numerics differ from the BF16 reference at position p (fault modes)."""
    if MODE == "kv":
        on = FP8
    elif MODE == "kv_long":
        on = FP8 and p >= 30000
    elif MODE == "mtp":
        on = FP8 and MTP
    elif MODE == "native":
        on = FP8 and not BYTE
    elif MODE == "path":
        on = FP8 and tf
    else:
        on = False
    return on and h(p) % 10 < 6


def dist(prev, p, k, tf=False, rep=0):
    """[(token, logprob)] sorted best first: the top k + 1 candidates of position p after token `prev`."""
    a = top1(prev)
    c = [(a, -0.05 - (prev % 17) * 0.01)] + [((a + j * 7919) % VOCAB, -1.5 - 0.4 * j - (prev % 5) * 0.01)
                                            for j in range(1, k + 1)]
    if diverges(p, tf):                                  # swap the top two and move the mass: a real divergence
        c[0], c[1] = (c[1][0], c[0][1]), (c[0][0], c[1][1] - 1.2)
    elif FP8 and MODE in ("clean", "kv_long", "mtp", "native", "path", "shift", "repeat"):
        c = [(t, lp - 0.002 * ((p * 31 + t) % 5) / 4) for t, lp in c]        # FP8 rounding-size noise
    if rep and MODE == "repeat" and FP8:
        c = [(t, lp - 0.4) for t, lp in c]
    return sorted(c, key=lambda x: -x[1])


def score(prev, p, tok, k, tf=False, rep=0):
    """(logprob, rank) of `tok` and the top-k list at position p."""
    d = dist(prev, p, k, tf, rep)
    for r, (t, lp) in enumerate(d, 1):
        if t == tok:
            return lp, r, d[:k]
    off = 0.4 if (rep and MODE == "repeat" and FP8) else 0.0       # the repeat fault moves every token
    return -9.0 - (tok % 101) * 0.01 - off, 50 + tok % 9000, d[:k]


def word(t):
    syl = ["ka", "ru", "mi", "pa", "ze", "lo", "ti", "ne", "vo", "sa", "gul", "fen", "bel", "dor", "qu", "ri"]
    return syl[t % 16] + syl[(t // 16) % 16] + syl[(t // 256) % 16] + " "


def block_hashes(ids):
    out, hsh = [], hashlib.sha256()
    for i in range(0, len(ids) - len(ids) % BLOCK, BLOCK):
        hsh.update(json.dumps(ids[i:i + BLOCK]).encode())
        out.append(hsh.hexdigest())
    return out


def cache_rw(ids, read):
    bh = block_hashes(ids)
    with LOCK:
        hit = 0
        if read:
            for i, x in enumerate(bh):
                if x not in CACHE:
                    break
                hit = (i + 1) * BLOCK
            hit = min(hit, max(0, (len(ids) - 1) // BLOCK * BLOCK))
        CACHE.update(bh)
    return hit


def generate(prompt, n, k):
    """Greedy continuation: tokens, per-token logprob of the chosen token, top-k dicts."""
    toks, lps, tops = [], [], []
    prev = prompt[-1] if prompt else 0
    for i in range(n):
        p = len(prompt) + i
        d = dist(prev, p, max(k, 2), tf=False)
        t = d[0][0]
        toks.append(t)
        lps.append(round(d[0][1], 6))
        tops.append({f"token_id:{x}": round(lp, 6) for x, lp in d[:max(k, 2)]})
        prev = t
    return toks, lps, tops


def prompt_logprobs(ids, k, rep):
    out = [None]
    for p in range(1, len(ids)):
        prev = ids[p] if MODE == "shift" and FP8 else ids[p - 1]
        lp, rank, top = score(prev, p, ids[p], k, tf=True, rep=rep)
        d = {f"token_id:{t}": {"logprob": round(l, 6), "rank": r, "decoded_token": f"token_id:{t}"}
             for r, (t, l) in enumerate(top, 1)}
        d[f"token_id:{ids[p]}"] = {"logprob": round(lp, 6), "rank": rank, "decoded_token": f"token_id:{ids[p]}"}
        out.append(d)
    return out


def arith(q):
    m = re.search(r"(\d+)\s*([*+-])\s*(\d+)", q)
    if not m:
        return "?"
    a, o, b = int(m.group(1)), m.group(2), int(m.group(3))
    return str(a * b if o == "*" else a + b if o == "+" else a - b)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj, ctype="application/json"):
        data = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/health"):
            return self._send(200, b"", "text/plain")
        if self.path.startswith("/metrics"):
            return self._send(200, b"vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0.0\n",
                              "text/plain")
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.path.startswith("/v1/chat/completions"):
            q = (req.get("messages") or [{}])[-1].get("content", "")
            return self._send(200, {"choices": [{"message": {"content": arith(q), "reasoning": None}, "finish_reason": "stop"}]})
        if not self.path.startswith("/v1/completions"):
            return self._send(404, {"error": "not found"})
        prompt = list(req.get("prompt") or [])
        mx = int(req.get("max_tokens") or 16)
        k = int(req.get("logprobs") or 0)
        plp = req.get("prompt_logprobs")
        sha = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
        hit = cache_rw(prompt, read=plp is None)
        if plp is not None:
            if int(plp) > 20:
                return self._send(400, {"error": {"message": "prompt_logprobs per-step prompt-scoring tensor bound exceeded"}})
            with LOCK:
                rep = SEEN.get(sha, 0)
                SEEN[sha] = rep + 1
        toks, lps, tops = generate(prompt, mx, k)
        text = "".join(word(t) for t in toks)
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Connection", "close")
            self.end_headers()
            for i in range(0, len(toks), 16):
                ch = {"text": "".join(word(t) for t in toks[i:i + 16]), "token_ids": toks[i:i + 16],
                      "finish_reason": "length" if i + 16 >= len(toks) else None}
                if k:
                    ch["logprobs"] = {"tokens": [f"token_id:{t}" for t in toks[i:i + 16]], "token_logprobs": lps[i:i + 16],
                                      "top_logprobs": tops[i:i + 16]}
                self.wfile.write(b"data: " + json.dumps({"choices": [ch]}).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
            return
        ch = {"text": text, "token_ids": toks, "finish_reason": "length"}
        if k:
            ch["logprobs"] = {"tokens": [f"token_id:{t}" for t in toks], "token_logprobs": lps, "top_logprobs": tops}
        if plp is not None:
            ch["prompt_logprobs"] = prompt_logprobs(prompt, int(plp), rep)
        self._send(200, {"choices": [ch], "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(toks),
                                                    "prompt_tokens_details": {"cached_tokens": hit}}})


time.sleep(BOOT_SEC)
say(f"(APIServer pid={os.getpid()}) INFO {ts()} [launcher.py:47] Route: /v1/completions; fake mode {MODE}, MTP {MTP}, byte {BYTE}")
ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
