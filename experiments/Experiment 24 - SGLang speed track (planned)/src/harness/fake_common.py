"""E24 fake servers, shared code (WP2, rev 2, 2026-10-30; stdlib only). Imported by fake_sglang_server.py and
fake_vllm_server.py from the same directory; never imports torch / sglang / vllm.

What lives here:
- env knobs and the time scale (FAKE_TIME_SCALE, falling back to E24_TIME_SCALE; 1.0 = real-time-like);
- the deterministic toy LM shared by BOTH fakes, so a same-weights cross-engine pair (vLLM INT4 vs SGLang INT4) agrees
  except at near ties, exactly like real engines (DESIGN rev 2 §3.2):
    * next-token logits are a pure function of (last 3 tokens, absolute position, weights family, engine noise key, path);
    * 8 candidate ids per position + a flat background at -20 nat over the rest of the 248,320-row lm_head, so a
      full-vocab request (token_ids_logprob over 248k ids, the section-6 KL method) has a well-defined value for every id;
    * logits are quantized to multiples of 1/8 nat after all perturbations (bf16-like): top-1/top-2 margins are
      multiples of 1/8 and about 1 % of positions are exact ties (lowest id wins, as torch.argmax);
    * perturbations before quantization: weights family (int4 / fp8 / bf16: +-0.25 nat, a cross-weights pair FLIPs below
      about 0.5 nat), engine noise key (+-FAKE_SG_NOISE, default 0.004: only near ties move), path noise (decode vs
      prefill / teacher forcing, +-0.004 by default), SSM dtype (folded into the engine key by the SGLang fake);
    * FAKE_SG_DIVERGE_AT=n: decode-path logits at generation index >= n put a wrong token on top with a 6-nat margin
      (a STATE_FLIP in greedy rows, and STATE_DRIFT for decode-vs-prefill because teacher forcing stays clean);
- a byte-free toy detokenizer (id -> short word) so degenerate() sees normal text unless a fault asks otherwise;
- the threaded HTTP server base (chunked SSE, uvicorn-style access lines), prometheus value formatting, a histogram;
- process helpers: Linux comm title (prctl, what setproctitle also does), parent-death watch, JIT cache file writes.
"""
import ctypes
import gzip
import hashlib
import json
import math
import os
import random
import socket
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ------------------------------------------------------------------------------------------------ env + time scale
def env_str(name, default=None):
    v = os.environ.get(name)
    return default if v is None or v == "" else v


def env_float(name, default):
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def env_int(name, default):
    try:
        return int(float(os.environ[name]))
    except (KeyError, ValueError):
        return default


def env_bool(name, default=False):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def time_scale():
    """Wall seconds per modeled second (0.02 = 50x faster). FAKE_TIME_SCALE wins over E24_TIME_SCALE."""
    for k in ("FAKE_TIME_SCALE", "E24_TIME_SCALE"):
        try:
            v = float(os.environ[k])
            if v > 0:
                return v
        except (KeyError, ValueError):
            pass
    return 1.0


def time_scaled_env():
    return any(os.environ.get(k) for k in ("FAKE_TIME_SCALE", "E24_TIME_SCALE"))


# ------------------------------------------------------------------------------------------------ hashing
MASK64 = (1 << 64) - 1


def mix64(x):
    """splitmix64 finalizer (fast, well mixed, deterministic across processes and Python runs)."""
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK64
    return x ^ (x >> 31)


def str_key(s):
    return int.from_bytes(hashlib.blake2b(str(s).encode(), digest_size=8).digest(), "little")


def u01(h):
    return (h >> 11) * (1.0 / (1 << 53))


# ------------------------------------------------------------------------------------------------ toy tokenizer text
LM_VOCAB = 248320            # lm_head rows (config.json text_config.vocab_size)
TOK_VOCAB = 248077           # tokenizer: 248,044 BPE ids + 33 added tokens (248044..248076)
EOS_IDS = (248046, 248044)   # <|im_end|>, <|endoftext|> (generation_config eos)
SPECIAL = {248044: "<|endoftext|>", 248045: "<|im_start|>", 248046: "<|im_end|>", 248058: "<tool_call>",
           248059: "</tool_call>", 248068: "<think>", 248069: "</think>", 248070: "<|MASK|>"}
_WORDS = ("the grid row column color cell input output pattern rule shape object left right top bottom each first "
          "last next same different copy move fill line border block red blue green yellow black gray pink orange "
          "purple teal count size width height larger smaller symmetric mirror rotate flip pixel region area "
          "background foreground corner edge center path step check test case train example result answer so then "
          "if not and or but every two three four five six seven eight nine zero one where which while when this that "
          "these those becomes remains stays changes repeats extends connects touches surrounds inside outside above "
          "below between around along across diagonal vertical horizontal stripe square rectangle cross dot frame "
          "hole gap segment cluster group pair single multiple most least only also note wait hmm let me verify "
          "so the transformation maps").split()
_DIGITS = [str(d) for d in range(10)]


def tok_text(tid):
    tid = int(tid)
    if tid in SPECIAL:
        return SPECIAL[tid]
    if tid >= TOK_VOCAB:
        return ""
    h = mix64(tid * 2654435761 + 17)
    r = h % 100
    if r < 22:
        return " " + _DIGITS[(h >> 8) % 10]
    if r < 26:
        return "\n"
    if r < 30:
        return ","
    if r < 32:
        return "."
    return " " + _WORDS[(h >> 16) % len(_WORDS)]


def detok(ids, skip_special=True):
    out = []
    for t in ids:
        if skip_special and int(t) in SPECIAL:
            continue
        out.append(tok_text(t))
    return "".join(out)


# ------------------------------------------------------------------------------------------------ toy LM
NCAND = 8
QUANT = 0.125                # bf16-like logit grid (1/8 nat)
BG_LOGIT = -20.0             # every non-candidate id
P_EOS = 0.0006               # EOS becomes top-1 at ~1 in 1,700 positions (natural EOS for non-ignore_eos requests)
WEIGHTS_NOISE = {"int4": 0.25, "fp8": 0.25, "bf16": 0.0}
SALT_C, SALT_M, SALT_G, SALT_E, SALT_N = 0xC0FFEE, 0xBEEF01, 0x6A7B0F, 0xE05E05, 0x5EED42


def weights_family(model_path):
    """INT4 (RedHat w4a16) / FP8 repack / bf16 from the path or its config.json (quantization_config)."""
    p = str(model_path or "").lower()
    if "fp8" in p:
        return "fp8"
    if "int4" in p or "w4a16" in p or "awq" in p or "gptq" in p:
        return "int4"
    try:
        cfg = json.load(open(os.path.join(model_path, "config.json")))
        q = json.dumps(cfg.get("quantization_config") or cfg.get("text_config", {}).get("quantization_config") or {}).lower()
        if "fp8" in q or "float8" in q:
            return "fp8"
        if q not in ("{}", "null"):
            return "int4"
    except Exception:
        pass
    return "int4"


class Dist:
    """One next-token distribution: candidate ids sorted by (-logit, id), quantized logits, log Z."""
    __slots__ = ("ids", "vals", "logz", "bg_lp", "_rank")

    def __init__(self, ids, vals):
        order = sorted(range(len(ids)), key=lambda i: (-vals[i], ids[i]))
        self.ids = [ids[i] for i in order]
        self.vals = [vals[i] for i in order]
        top = self.vals[0]
        z = sum(math.exp(v - top) for v in self.vals) + (LM_VOCAB - len(self.ids)) * math.exp(BG_LOGIT - top)
        self.logz = top + math.log(z)
        self.bg_lp = BG_LOGIT - self.logz
        self._rank = None

    def lp(self, tid):
        tid = int(tid)
        for i, c in enumerate(self.ids):
            if c == tid:
                return self.vals[i] - self.logz
        return self.bg_lp

    def argmax(self):
        return self.ids[0]

    def top(self, k):
        """[(logprob, id)] for the k most likely ids; ties (equal logits) by ascending id, background ids after."""
        out = [(v - self.logz, c) for v, c in zip(self.vals, self.ids)][:k]
        if k > len(out):
            have = set(self.ids)
            t = 0
            while len(out) < k:
                if t not in have:
                    out.append((self.bg_lp, t))
                t += 1
        return out

    def rank(self, tid):
        """vLLM-style rank: 1 + number of ids with a strictly higher logprob (background ids all tie)."""
        tid = int(tid)
        for i, c in enumerate(self.ids):
            if c == tid:
                return 1 + sum(1 for v in self.vals[:i] if v > self.vals[i])
        return len(self.ids) + 1

    def lp_table(self):
        """{candidate id: logprob}; every other id has self.bg_lp."""
        return {c: v - self.logz for c, v in zip(self.ids, self.vals)}

    def sample(self, rng, temperature=1.0, top_p=1.0, top_k=-1):
        if temperature is None or temperature < 1e-6 or top_k == 1:
            return self.ids[0]
        t = float(temperature)
        k = len(self.ids) if (top_k is None or top_k <= 0 or top_k > len(self.ids)) else int(top_k)
        ws = [math.exp((v - self.vals[0]) / t) for v in self.vals[:k]]
        bg_mass = 0.0
        if top_k is None or top_k <= 0 or top_k > len(self.ids):       # the flat background is in the support
            bg_mass = (LM_VOCAB - len(self.ids)) * math.exp((BG_LOGIT - self.vals[0]) / t)
        tot = sum(ws) + bg_mass
        if top_p is not None and 0 < top_p < 1.0:                       # nucleus over the sorted candidates
            acc, keep = 0.0, 0
            for w in ws:
                keep += 1
                acc += w / tot
                if acc >= top_p:
                    break
            ws, bg_mass = ws[:keep], (bg_mass if acc < top_p else 0.0)
            tot = sum(ws) + bg_mass
        r = rng.random() * tot
        for w, c in zip(ws, self.ids):
            r -= w
            if r <= 0:
                return c
        bg = rng.randrange(1000, 247000)
        while bg in self.ids:
            bg += 1
        return bg


class ToyLM:
    """Deterministic next-token model. `engine_key` identifies the engine build/arm numerics (noise seed); `noise` is the
    pre-quantization perturbation amplitude for that key; `path_noise` separates decode/verify from prefill."""

    def __init__(self, weights="int4", engine_key="", noise=0.004, path_noise=0.004, diverge_at=None,
                 diverge_margin=6.0):
        self.weights = weights
        self.wkey = str_key("w:" + weights)
        self.wamp = WEIGHTS_NOISE.get(weights, 0.25)
        self.ekey = str_key("e:" + engine_key)
        self.eamp = float(noise)
        self.pamp = float(path_noise)
        self.diverge_at = diverge_at
        self.diverge_margin = float(diverge_margin)

    @staticmethod
    def ctx_hash(t1, t2, t3, pos):
        return mix64((t1 * 0x100000001B3) ^ (t2 * 0xC2B2AE3D27D4EB4F) ^ (t3 * 0x165667B19E3779F9) ^ (pos * 0x27D4EB2F165667C5))

    def dist(self, ids, pos, path="decode", gen_idx=None):
        """Distribution of the token at absolute position `pos` given ids[:pos] (only the last 3 tokens matter)."""
        t1 = ids[pos - 1] if pos >= 1 else 0
        t2 = ids[pos - 2] if pos >= 2 else 0
        t3 = ids[pos - 3] if pos >= 3 else 0
        return self.dist_h(self.ctx_hash(t1, t2, t3, pos), path, gen_idx)

    def dist_h(self, h, path="decode", gen_idx=None):
        cands, seen = [], set()
        for i in range(NCAND):
            c = 1000 + mix64(h ^ (SALT_C + i * 0x9E37)) % 246000
            while c in seen:
                c += 1
            seen.add(c)
            cands.append(c)
        if u01(mix64(h ^ SALT_E)) < P_EOS:
            cands[0] = EOS_IDS[0]
        r = u01(mix64(h ^ SALT_M))
        u = u01(mix64(h ^ SALT_M ^ 0xFF))
        if r < 0.012:
            m = 0.05 * u                     # near tie: usually an exact tie after quantization
        elif r < 0.16:
            m = 0.06 + 1.0 * u
        elif r < 0.40:
            m = 1.0 + 2.5 * u
        else:
            m = 3.0 + 6.0 * u
        base = 8.0 + 4.0 * u01(mix64(h ^ SALT_G))
        vals = [base, base - m]
        for i in range(2, NCAND):
            vals.append(vals[-1] - (0.3 + 2.0 * u01(mix64(h ^ (SALT_G + i)))))
        pk = 1 if path == "decode" else 2
        out = []
        for i, v in enumerate(vals):
            if self.wamp:
                v += self.wamp * (2.0 * u01(mix64(h ^ self.wkey ^ (i * 0x51))) - 1.0)
            if self.eamp:
                v += self.eamp * (2.0 * u01(mix64(h ^ self.ekey ^ (i * 0x77 + 3))) - 1.0)
            if self.pamp:
                v += self.pamp * (2.0 * u01(mix64(h ^ (SALT_N * pk) ^ (i * 0x35 + 9))) - 1.0)
            out.append(round(v / QUANT) * QUANT)
        if self.diverge_at is not None and path == "decode" and gen_idx is not None and gen_idx >= self.diverge_at:
            j = 1 + mix64(h ^ 0xD1CE) % (NCAND - 1)          # a wrong candidate jumps 6 nat above the old top-1
            out[j] = max(out) + round(self.diverge_margin / QUANT) * QUANT
        return Dist(cands, out)


# ------------------------------------------------------------------------------------------------ speculative accept
ACCEPT_MEAN = {1: 1.0, 2: 1.8, 3: 2.55, 4: 3.3, 5: 3.8, 6: 4.2, 7: 4.6, 8: 5.0, 16: 6.0}


def accept_p(width, mean=None):
    """Per-draft acceptance probability p such that E[1 + leading accepted drafts of width-1] = mean (stop at first miss)."""
    w = int(width)
    if w <= 1:
        return 0.0
    target = (mean if mean is not None else ACCEPT_MEAN.get(w) or min(6.0, 1 + 0.85 * (w - 1))) - 1.0
    lo, hi = 0.0, 0.999
    for _ in range(60):
        p = (lo + hi) / 2
        e = sum(p ** k for k in range(1, w))
        lo, hi = (p, hi) if e < target else (lo, p)
    return (lo + hi) / 2


def draw_accept(width, p, h):
    """1 (bonus) + leading accepted drafts, deterministic in h."""
    a = 1
    for k in range(1, int(width)):
        if u01(mix64(h ^ (k * 0x1F1F))) < p:
            a += 1
        else:
            break
    return a


# ------------------------------------------------------------------------------------------------ prometheus helpers
def pnum(v):
    """prometheus_client float formatting: repr of the float, +Inf / NaN spelled out."""
    v = float(v)
    if v != v:
        return "NaN"
    if v == math.inf:
        return "+Inf"
    if v == -math.inf:
        return "-Inf"
    return repr(v)


def plabels(d):
    if not d:
        return ""
    return "{" + ",".join('%s="%s"' % (k, str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"'))
                          for k, v in sorted(d.items())) + "}"


class Hist:
    """Cumulative prometheus histogram (one label set)."""

    def __init__(self, buckets):
        self.b = list(buckets)
        self.c = [0] * len(self.b)
        self.n = 0
        self.s = 0.0

    def observe(self, v, times=1):
        for i, ub in enumerate(self.b):
            if v <= ub:
                self.c[i] += times
        self.n += times
        self.s += v * times

    def lines(self, name, labels):
        out = ["%s_sum%s %s" % (name, plabels(labels), pnum(self.s))]
        for ub, c in zip(self.b, self.c):
            out.append("%s_bucket%s %s" % (name, plabels(dict(labels, le=pnum(ub) if ub != math.inf else "+Inf")), pnum(c)))
        out.append("%s_bucket%s %s" % (name, plabels(dict(labels, le="+Inf")), pnum(self.n)))
        out.append("%s_count%s %s" % (name, plabels(labels), pnum(self.n)))
        return out


# ------------------------------------------------------------------------------------------------ HTTP plumbing
class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 512
    allow_reuse_address = True

    def server_bind(self):
        """HTTPServer.server_bind calls socket.getfqdn() (a reverse-DNS lookup that can block ~30 s on macOS)."""
        import socketserver
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = host, port


class BaseHandler(BaseHTTPRequestHandler):
    """HTTP/1.1, JSON helpers, chunked SSE, uvicorn-style access lines through the owner's access_log()."""
    protocol_version = "HTTP/1.1"
    server_version = "uvicorn"
    sys_version = ""
    owner = None            # set by the fake: object with access_log(client, method, path, code)

    def log_message(self, *a):
        pass

    def read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n:
            return self.rfile.read(n)
        if (self.headers.get("Transfer-Encoding") or "").lower() == "chunked":
            data = b""
            while True:
                line = self.rfile.readline().strip()
                size = int(line.split(b";")[0] or b"0", 16)
                if size == 0:
                    self.rfile.readline()
                    break
                data += self.rfile.read(size)
                self.rfile.readline()
            return data
        return b""

    def _access(self, code):
        try:
            if self.owner is not None:
                self.owner.access_log(self.client_address, self.command, self.path, code)
        except Exception:
            pass

    def send_bytes(self, body, code=200, ctype="application/json", extra=None):
        b = body if isinstance(body, bytes) else str(body).encode()
        self.send_response(code)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(b)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(b)
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        self._access(code)

    def send_json(self, obj, code=200):
        self.send_bytes(json.dumps(obj, separators=(",", ":"), allow_nan=False).encode(), code, "application/json")

    def start_sse(self, code=200):
        self.send_response(code)
        self.send_header("content-type", "text/event-stream; charset=utf-8")
        self.send_header("transfer-encoding", "chunked")
        self.send_header("connection", "close")
        self.end_headers()
        self.close_connection = True
        self._access(code)

    def sse(self, payload):
        """One SSE event `data: <payload>\\n\\n` as one HTTP chunk; raises on a closed client."""
        b = b"data: " + (payload if isinstance(payload, bytes) else payload.encode()) + b"\n\n"
        self.wfile.write(b"%x\r\n%s\r\n" % (len(b), b))
        self.wfile.flush()

    def end_chunked(self):
        try:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass


def dumps(obj):
    """Compact JSON like orjson (no spaces); NaN/inf are not expected in payloads."""
    return json.dumps(obj, separators=(",", ":"))


# ------------------------------------------------------------------------------------------------ process helpers
def set_comm(title):
    """Linux: what setproctitle also does for /proc/<pid>/comm (15 bytes). No-op elsewhere."""
    try:
        if sys.platform.startswith("linux"):
            libc = ctypes.CDLL(None)
            libc.prctl(15, ctypes.c_char_p(title.encode()[:15]), 0, 0, 0)        # PR_SET_NAME
    except Exception:
        pass


def set_proctitle(title):
    """What SGLang/vLLM children do with the `setproctitle` package: use it when installed, else register a minimal
    stand-in module in sys.modules (getproctitle() answers the title, as the hook reads it) + the Linux comm name."""
    set_comm(title)
    try:
        import setproctitle as spt
        spt.setproctitle(title)
        return
    except Exception:
        pass
    import types
    m = sys.modules.get("setproctitle")
    if m is None or not getattr(m, "__fake__", False):
        m = types.ModuleType("setproctitle")
        m.__fake__ = True
        m.__file__ = "<fake-setproctitle>"
        sys.modules["setproctitle"] = m
    m._title = title
    m.setproctitle = lambda t: setattr(m, "_title", str(t))
    m.getproctitle = lambda: m._title


def watch_parent(on_dead, poll=0.25):
    """Call on_dead() when the original parent disappears (what SGLang/vLLM children do via PDEATHSIG / pipes)."""
    ppid = os.getppid()

    def run():
        while True:
            time.sleep(poll)
            if os.getppid() != ppid:
                try:
                    on_dead()
                finally:
                    os._exit(0)
    t = threading.Thread(target=run, daemon=True, name="parent-watch")
    t.start()
    return t


def write_cache_files(root, names, payload=b"", rank=0):
    """Simulated JIT artefacts: create `names` (relative paths) under root if missing; returns files written."""
    n = 0
    if not root:
        return 0
    for rel in names:
        p = os.path.join(root, rel)
        try:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if not os.path.exists(p):
                with open(p, "wb") as f:
                    f.write(payload or (b"fake-jit rank %d %s\n" % (rank, rel.encode())))
                n += 1
        except OSError:
            pass
    return n


def dir_has_files(root):
    try:
        for _dp, _dn, fn in os.walk(root):
            if fn:
                return True
    except OSError:
        pass
    return False


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def gz_json(path, obj):
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt", encoding="utf-8") as f:
        json.dump(obj, f, separators=(",", ":"))
    os.replace(tmp, path)


def new_rid():
    return hashlib.blake2b(("%d-%d-%f-%d" % (os.getpid(), threading.get_ident(), time.time(), random.getrandbits(32))
                            ).encode(), digest_size=16).hexdigest()
