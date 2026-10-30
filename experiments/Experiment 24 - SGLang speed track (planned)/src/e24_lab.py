"""E24 lab, rev 1 (2026-10-30; BUILD_SPEC rev 2, WP1): engine-aware server lifecycle and measurement for the SGLang 0.5.21
speed track and its vLLM 0.32 oracle arms (stdlib only, Python 3.12).

Base: e23_lab rev 3b (connect port check, always-SIGKILL kill, GPU release gate, watchdog, Stream with token ids and
[DONE] + finish_reason, greedy oracle, garbage checks, steady control flow, the vLLM paths section 1 needs). SGLang facts,
read in the 0.5.21 wheel (paths under sglang/) and checked on the 7 EMS server logs:
- /metrics is label-aware, never summed across ranks. realtime_tokens_total{mode=decode|prefill_compute|prefill_cache}
  moves every batch on TP0 (metrics_reporter.py:857-871); generation/prompt_tokens_total move only at request FINISH
  (metrics_collector.py:1780-1799). cuda_graph_passes_total{mode=decode_*} is +1 per verify step on EVERY rank
  (batch_result_processor.py:962-965): tp_rank 0, a missing mode counts 0. Gauges are TP0, refresh every
  --decode-log-interval steps and stay flat during a chunked prefill (that request is neither running nor queued):
  settle / plateau rules use the prefill counters + TP0 log lines. num_retracted_requests_total exists only after the
  first retraction (metrics_collector.py:474-478): absent = 0 while the scheduler series are live, no TP0
  "Retract requests" line and gauge num_retracted_reqs not > 0.
- Logs: Decode / Prefill / Retract regexes validated with 0 misses on the EMS logs (metrics_reporter.py:709-739,
  919-999; scheduler.py:4249-4261); startup args = the server_args={...} dict line (engine.py:1123); the error scan
  stops at our own SIGTERM (the SIGQUIT / "crashed with exit code -15" lines after it are kill noise).
- Native /generate: greedy rows keep top-2 ids (output_top_logprobs [[lp, id, null], ...]). Teacher forcing uses
  logprob_start_len = len(prompt)-1 and drops entry 0: a row at p scores ids[p+1] (schedule_batch.py:2894-2908), the
  reply is [None]+rows[:-1] with idx = ids[start:] (logprob_result_processor.py:37-82), so entry k = lp(cont[k] |
  prompt+cont[:k]); every reply is asserted (length, ids) else INCONCLUSIVE_TF_ALIGN. Servers that score input
  logprobs need SGLANG_LOGPROB_CHUNK_SIZE=128 (2048 materialises 3-4 GB fp32 logits per rank, logprob_processor.py:
  504-546) and one scoring request at a time. /v1/completions ignores prompt_logprobs / return_tokens_as_token_ids
  (protocol.py:334-421): only the load streams use it (return_token_ids deltas, rid, include_usage). No endpoint
  applies generation_config (serving_completions.py:147-173): T>0 sends temperature 1.0, top_p 0.95, top_k 20.
  prompt + max_new > context length is an HTTP 400 (tokenizer_manager.py:1306-1331).
- GET /ready = readiness (http_server.py:692-696; /health runs a 1-token generate). /start_profile {output_dir, num_steps,
  activities, with_stack, record_shapes} (io_struct.py:2257-2283; vLLM keys are ignored = unbounded) writes
  <id>-TP-<r>.trace.json.gz per rank (profiler_manager.py:327-360); per-step wall = consecutive TP0 `step[VERIFY` (else
  `step[DECODE`) span starts (profile_utils.py:462-498): DFlash launches two graphs per step.
- Flag check: exact ServerArgs.add_cli_args option strings (argparse allow_abbrev accepts prefixes, --cuda-graph-bs is
  ambiguous: server_args.py:628-664) + prepare_server_args; resolve_once() (:227-253) only on the GPU box.
- Caches: import sglang only setdefaults TRITON / INDUCTOR / CUDA_CACHE_PATH / TILELANG / FLASHINFER under
  SGLANG_CACHE_DIR (environ.py:1962-1990); SGLANG_JIT_CACHE_DIR is separate; vLLM rewrites Inductor / Triton under
  VLLM_CACHE_ROOT: the isolation verdict checks containment, not equality.
Every number is measured; missing data -> None / INCONCLUSIVE, never 0. TS (env E24_TIME_SCALE) scales only the lab's
internal floors and idle polls; callers scale their own time arguments.
"""
import ast
import gzip
import hashlib
import json
import math
import os
import re
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.request as U
import uuid
from collections import Counter
from pathlib import Path

FAIL_MARGIN = 0.5          # floor of T_pair: a first divergence with m_ref + m_cand below it is a NEAR_TIE
STATE_FLIP_MARGIN = 4.0    # m_ref + m_cand >= 4 nat = STATE_FLIP (E19: real state bugs gave 4-12 nat flips)
LIVE = set()               # every running Server, for the hard-stop watchdog
TS = float(os.environ.get("E24_TIME_SCALE") or 1.0)
TF_BINS = ((0, 2000), (2000, 8000), (8000, 16000), (16000, 10 ** 9))
CACHE_VARS = ("SGLANG_CACHE_DIR", "SGLANG_JIT_CACHE_DIR", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR",
              "FLASHINFER_WORKSPACE_BASE", "CUDA_CACHE_PATH", "TILELANG_CACHE_DIR", "TORCH_EXTENSIONS_DIR")
_CACHE_SUB = dict(zip(CACHE_VARS + ("VLLM_CACHE_ROOT",), ("sglang", "sglang_jit", "triton", "inductor", "flashinfer", "nv",
                                                         "tilelang", "torch_extensions", "vllm")))


# ------------------------------------------------------------------ small helpers
def _body(e):
    try:
        return e.read()[:300].decode("utf-8", "replace")
    except Exception:
        return ""


def _http(port, path, body=None, timeout=60.0):
    """GET / POST JSON on 127.0.0.1; JSON else {"text"}; HTTP errors raise with the body."""
    req = U.Request(f"http://127.0.0.1:{int(port)}{path}", data=None if body is None else json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"})
    try:
        with U.urlopen(req, timeout=timeout) as r:
            raw = r.read()
    except U.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code} {path}: {_body(e)}") from None
    try:
        return json.loads(raw)
    except ValueError:
        return {"text": raw.decode("utf-8", "replace")[:2000]}


_MODELS = {}


def _model(port, model=None):
    """Served model id (GET /v1/models, cached per port) or "default"."""
    if model:
        return model
    if port not in _MODELS:
        try:
            _MODELS[port] = (_http(port, "/v1/models", timeout=10).get("data") or [{}])[0].get("id") or "default"
        except Exception:
            return "default"
    return _MODELS[port]


def _rid(tag="r"):
    return f"e24-{tag}-{uuid.uuid4().hex[:16]}"


def _q(v, f):
    """Nearest-rank quantile; None for no data."""
    v = sorted(x for x in v if x is not None)
    return v[min(len(v) - 1, max(0, math.ceil(f * len(v)) - 1))] if v else None


def _stats(v, nd=2):
    v = [x for x in v if x is not None]
    r = lambda x: None if x is None else round(x, nd)
    return {"n": len(v), "p10": r(_q(v, .1)), "p50": r(statistics.median(v)) if v else None, "p90": r(_q(v, .9)),
            "p95": r(_q(v, .95)), "mean": r(statistics.mean(v)) if v else None, "max": r(max(v)) if v else None}


def _mean(v, nd=3):
    v = [x for x in v if x is not None]
    return round(statistics.mean(v), nd) if v else None


def _delta(a, b, absent0=True):
    """Counter delta; absent0: absent at start -> the end value; absent at end -> None."""
    if b is None or (a is None and not absent0):
        return None
    return b if a is None else b - a


def _raw(raw_dir, name, obj):
    """Atomic raw artifact under raw_dir; the path or None."""
    if not raw_dir:
        return None
    try:
        p = Path(raw_dir) / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.with_suffix(".tmp").write_text(json.dumps(obj, default=str), encoding="utf-8")
        os.replace(p.with_suffix(".tmp"), p)
        return str(p)
    except Exception:
        return None


def _tag(tag):
    return str(tag) if tag else str(int(time.time() * 1000))


def _key(row, i):
    return row.get("key") if isinstance(row, dict) and row.get("key") is not None else f"#{i}"


def prompt_key(ids):
    """Prompt identity for row alignment across arms."""
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()[:12]


def _num(x):
    for f in (int, float):
        try:
            return f(x)
        except (TypeError, ValueError):
            pass
    return None


def _f(x):
    return None if x is None else float(x)


def _i(x):
    return None if x is None else int(x)


# ------------------------------------------------------------------ server lifecycle
def port_listening(port):
    """Something accepts connections on 127.0.0.1:port (TIME_WAIT does not)."""
    s = socket.socket()
    s.settimeout(1.0)
    try:
        s.connect(("127.0.0.1", int(port)))
        return True
    except OSError:
        return False
    finally:
        s.close()


class Server:
    def __init__(self, name, argv, env, port, log_dir, gpus="0,1,2,3", engine="sglang"):
        self.name, self.argv, self.env, self.port = name, [str(a) for a in argv], dict(env), int(port)
        self.gpus, self.engine, self.model_name = str(gpus), engine, None
        self.log_path = Path(log_dir) / f"server_{name}.log"
        self.proc, self.t0, self.ready_sec, self._fh = None, None, None, None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        end = time.time() + 90
        while port_listening(self.port):
            if time.time() > end:
                raise RuntimeError(f"PORT_BUSY {self.port} (a listener is still there)")
            time.sleep(3)
        _MODELS.pop(self.port, None)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.log_path, "w", encoding="utf-8")
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.argv, env=self.env, stdout=self._fh, stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd="/tmp" if os.path.isdir("/tmp") else None)
        LIVE.add(self)
        return self

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def log(self):
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def rss_mb(self):
        """Host RAM of the server session, MB; None off Linux."""
        try:
            r = subprocess.run(["ps", "-o", "rss=", "--sid", str(self.proc.pid)], capture_output=True, text=True, timeout=20)
            return round(sum(int(x) for x in r.stdout.split() if x.isdigit()) / 1024, 1)
        except Exception:
            return None

    def wait_ready(self, timeout):
        """SGLang GET /ready (side-effect free) / vLLM /health, plus the engine ready line in OUR log; then the served
        model id; (False, why) on exit or timeout."""
        path = "/ready" if self.engine == "sglang" else "/health"
        # a 200 counts only once OUR log shows the engine's ready line: another listener on the port must not pass
        mark = "The server is fired up and ready to roll!" if self.engine == "sglang" else "Application startup complete"
        while time.perf_counter() - self.t0 < timeout:
            if not self.alive():
                return False, f"exited rc={self.proc.returncode}"
            try:
                U.urlopen(self.url + path, timeout=10).read()
                if mark in self.log():
                    self.ready_sec = round(time.perf_counter() - self.t0, 1)
                    self.model_name = _model(self.port)
                    return True, f"ready in {self.ready_sec}s"
            except U.HTTPError as e:
                path = "/health" if e.code == 404 else path      # only a server without /ready (not 0.5.21)
            except Exception:
                pass
            time.sleep(min(3.0, max(0.25, TS)))      # 1 s on the GPU box: the 600 s startup gate is sharp
        return False, "startup timeout"

    def kill(self):
        if self.proc is None:
            return
        for sig, wait in ((signal.SIGTERM, 40), (signal.SIGKILL, 30)):
            # e23 rev 3: the leader may exit while schedulers / workers of the session still hold GPUs: always SIGKILL
            try:
                os.killpg(self.proc.pid, sig) if hasattr(os, "killpg") else self.proc.kill()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=wait)
            except Exception:
                pass
        try:
            self._fh.close()
        except Exception:
            pass
        LIVE.discard(self)

    def server_info(self):
        """SGLang GET /server_info; None for vLLM."""
        if self.engine != "sglang":
            return None
        try:
            return _http(self.port, "/server_info", timeout=60)
        except Exception as e:
            return {"error": repr(e)[:300]}

    def facts(self):
        return dict(_facts_sglang(self.log()) if self.engine == "sglang" else _facts_vllm(self.log()), ready_sec=self.ready_sec)


def kill_orphans(fake=False):
    """SIGKILL leftover engine processes (sglang::*, vLLM EngineCore / workers) only when no Server is live."""
    if LIVE:
        return []
    out, me = [], {os.getpid(), os.getppid()}
    for pat in (["fake_sglang_server.py", "fake_vllm_server.py"] if fake else
                ["sglang::", "sglang.launch_server", "VLLM::", "vllm.entrypoints", "vllm serve"]):
        try:
            pids = subprocess.run(["pgrep", "-f", pat], capture_output=True, text=True, timeout=20).stdout.split()
        except Exception:
            continue
        for x in pids:
            if x.isdigit() and int(x) not in me:
                try:
                    os.kill(int(x), signal.SIGKILL)
                    out.append([pat, int(x)])
                except Exception:
                    pass
    return out


def gpu_pids(gpus=None):
    """Compute PIDs (on `gpus`); None without nvidia-smi."""
    try:
        a = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        g = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if a.returncode or g.returncode:
        return None
    idx = {u.strip(): int(i) for i, u in (l.split(",", 1) for l in g.stdout.strip().splitlines() if "," in l)}
    want = None if gpus is None else {int(x) for x in str(gpus).split(",") if str(x).strip() != ""}
    pairs = [[x.strip() for x in l.split(",", 1)] for l in a.stdout.strip().splitlines() if "," in l]
    return sorted({int(p) for p, u in pairs if p.isdigit() and (want is None or idx.get(u) in want)})


def gpu_mem():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return [[int(v) for v in l.split(",")] for l in r.stdout.strip().splitlines() if l.strip()]
    except Exception:
        return None


def wait_gpu_release(timeout=180, fake=False, gpus=None, free_mib=1024):
    """True when the GPUs hold no compute PID and < free_mib MiB each (memory.used is the proven gate); None without
    nvidia-smi."""
    if fake:
        return True
    want = None if gpus is None else {int(x) for x in str(gpus).split(",") if str(x).strip() != ""}
    end = time.time() + timeout
    while time.time() < end:
        p, mem = gpu_pids(gpus), gpu_mem()
        if p is None or mem is None:
            return None
        if not p and all(m[1] < free_mib for m in mem if want is None or m[0] in want):
            return True
        time.sleep(3)
    return False


def identity(model_dir):
    """Checkpoint identity (sha256 of config / tokenizer / index files)."""
    root = Path(model_dir)
    files = [p for n in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
             for p in [root / n] if p.is_file()] + sorted(root.glob("*.safetensors.index.json"))
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return {"dir": str(root), "files": [p.name for p in files], "sha256": h.hexdigest()[:16]}


def start_watchdog(hard_stop_epoch, on_fire):
    """Hard stop: at hard_stop_epoch kill every live Server and call on_fire()."""
    def run():
        while time.time() < hard_stop_epoch:
            time.sleep(5)
        try:
            on_fire()
        finally:
            for s in list(LIVE):
                try:
                    s.kill()
                except Exception:
                    pass
    t = threading.Thread(target=run, daemon=True, name="e24-watchdog")
    t.start()
    return t


def prefetch(paths, threads=16, chunk=16 << 20):
    """Page-cache warm-up of a cold mount with many readers."""
    files = sorted((p for root in paths for p in Path(root).rglob("*") if p.is_file()), key=lambda p: -p.stat().st_size)
    t0, done, lock = time.time(), [0], threading.Lock()

    def worker(fs):
        for p in fs:
            try:
                with open(p, "rb") as f:
                    while True:
                        b = f.read(chunk)
                        if not b:
                            break
                        with lock:
                            done[0] += len(b)
            except Exception:
                pass
    ths = [threading.Thread(target=worker, args=(x,), daemon=True) for x in (files[i::threads] for i in range(threads)) if x]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    dt = time.time() - t0
    return {"files": len(files), "bytes": done[0], "sec": round(dt, 1), "MBps": round(done[0] / 1e6 / max(dt, 1e-3), 1)}


# ------------------------------------------------------------------ flag check (enforced by the notebook)
_FLAG_SCRIPT = r'''
import argparse, json, sys
o = {"ran": False, "ok": False, "unknown": [], "ambiguous": [], "prefix": [], "parse_error": None, "resolve_error": None,
     "method": "syntax_only"}
try:
    argv, resolve = json.loads(sys.argv[1]), sys.argv[2] == "1"
    from sglang.srt.server_args import ServerArgs, prepare_server_args
    p = argparse.ArgumentParser(prog="sglang serve")
    ServerArgs.add_cli_args(p)
    opts = set(p._option_string_actions)
    o["ran"], o["n_options"] = True, len(opts)
    for f in (a.split("=", 1)[0] for a in argv if isinstance(a, str) and a.startswith("--")):
        hits = [x for x in opts if x.startswith(f)]
        if f not in opts:
            (o["ambiguous"] if len(hits) > 1 else o["prefix"] if hits else o["unknown"]).append(f)
    try:
        sa = prepare_server_args(argv)
        if resolve:
            o["method"] = "resolved"
            try:
                sa.resolve_once()
            except BaseException as e:
                o["resolve_error"] = (type(e).__name__ + ": " + str(e))[:400]
    except SystemExit as e:
        o["parse_error"] = "argparse exit %s" % (e.code,)
    except BaseException as e:
        o["parse_error"] = (type(e).__name__ + ": " + str(e))[:400]
except BaseException as e:
    o["error"] = (type(e).__name__ + ": " + str(e))[:400]
print("E24_FLAG_CHECK " + json.dumps(o), flush=True)
'''
_FLAGS, _HELP = {}, {}


def _flags_only(engine, argv):
    """SGLang: from the first "--" token; vLLM: after "serve"."""
    a = [str(x) for x in argv]
    if engine == "vllm":
        return a[a.index("serve") + 1:] if "serve" in a else a
    return a[next((k for k, x in enumerate(a) if x.startswith("--")), len(a)):]


def _parse_flag_out(txt):
    """Last flag-check JSON line of a checker's stdout."""
    for line in reversed((txt or "").splitlines()):
        s = line.strip()
        s = s[15:] if s.startswith("E24_FLAG_CHECK ") else s
        try:
            d = json.loads(s) if s.startswith("{") else None
        except ValueError:
            d = None
        if isinstance(d, dict) and "ok" in d:
            return d
    return None


def _vllm_help_missing(help_txt, flags):
    return [f for f in flags if not re.search(r"(?<![\w-])" + re.escape(f) + r"(?![\w-])", help_txt)]


def flag_check(engine, py, argv, env=None, timeout=300, fake_server=None, resolve=False, model_path=None):
    """Enforced, cached by argv hash. SGLang (`py -I -c`): every "--x" an EXACT 0.5.21 option string (prefixes refused),
    prepare_server_args(argv), and with resolve=True (GPU box) resolve_once() (else method syntax_only); vLLM: `serve
    --help=all` exact tokens; fake_server: its --e24-flag-check. status GO | NO_GO_FLAG_UNSUPPORTED |
    NO_GO_FLAG_RESOLUTION | NOT_RUN (ok False)."""
    flags = _flags_only(engine, argv)
    if model_path:
        flags = [str(model_path) if i and flags[i - 1] in ("--model-path", "--model") else x for i, x in enumerate(flags)]
    key = hashlib.sha256(json.dumps([engine, str(py), flags, bool(resolve), str(fake_server or "")]).encode()).hexdigest()
    if key in _FLAGS:
        return dict(_FLAGS[key], cached=True)
    t0 = time.time()
    out = {"ran": False, "ok": False, "unknown": [], "ambiguous": [], "prefix": [], "parse_error": None,
           "resolve_error": None, "method": None, "engine": engine, "n_flags": sum(1 for x in flags if x.startswith("--"))}
    try:
        if fake_server or engine == "sglang":
            r = subprocess.run([str(py), str(fake_server), "--e24-flag-check", json.dumps(flags)] if fake_server else
                               [str(py), "-I", "-c", _FLAG_SCRIPT, json.dumps(flags), "1" if resolve else "0"],
                               capture_output=True, text=True, timeout=timeout, env=env)
            d = _parse_flag_out(r.stdout)
            if d is None:
                out["error"] = f"no flag-check JSON (rc {r.returncode}): {(r.stdout + r.stderr)[-400:]}"
            else:
                out.update({k: v for k, v in d.items() if k not in ("engine", "n_flags", "ok")})
                if fake_server:
                    out["ran"], out["method"] = bool(d.get("ran", True)), "fake:" + str(d.get("method") or "exact")
        else:
            if str(py) not in _HELP:
                r = subprocess.run([str(py), "-m", "vllm.entrypoints.cli.main", "serve", "--help=all"], capture_output=True,
                                   text=True, timeout=timeout, env=env)
                _HELP[str(py)] = r.stdout + r.stderr
            txt, out["method"] = _HELP[str(py)], "help_all"
            if len(txt) < 2000:
                out["error"] = "help output too short: " + txt[-300:]
            else:
                out["ran"] = True
                out["unknown"] = _vllm_help_missing(txt, sorted({f.split("=", 1)[0] for f in flags if f.startswith("--")}))
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:300]
    syn = out["unknown"] or out["ambiguous"] or out["prefix"] or out["parse_error"]
    out["ok"] = bool(out["ran"] and not (syn or out["resolve_error"]))
    out["status"] = "GO" if out["ok"] else "NOT_RUN" if not out["ran"] else \
        "NO_GO_FLAG_UNSUPPORTED" if syn else "NO_GO_FLAG_RESOLUTION"
    out["sec"] = round(time.time() - t0, 1)
    if out["ran"]:
        _FLAGS[key] = dict(out)
    return dict(out, cached=False)


# ------------------------------------------------------------------ caches, hook records, isolation verdict
def cache_env(engine, root):
    """Per-arm cache env (DESIGN §3.3) under `root` (dirs created); + VLLM_CACHE_ROOT for vLLM."""
    env = {"CUDA_CACHE_DISABLE": "0"}
    for k in CACHE_VARS + (("VLLM_CACHE_ROOT",) if engine == "vllm" else ()):
        env[k] = str(Path(root) / _CACHE_SUB[k])
        Path(env[k]).mkdir(parents=True, exist_ok=True)
    return env


def default_cache_dirs():
    """DESIGN §3.3 default cache locations (must get no new files during an isolated arm)."""
    return [os.path.expanduser(p) for p in ("~/.cache/sglang", "~/.cache/flashinfer", "~/.triton", "~/.nv/ComputeCache",
                                            "~/.cache/vllm")] + sorted(str(p) for p in Path("/tmp").glob("torchinductor_*"))


def dir_stats(paths):
    """{path: files, bytes, newest} (missing: 0, 0, None)."""
    out = {}
    for p in paths:
        st = []
        for dp, _, fns in os.walk(str(p)):
            for fn in fns:
                try:
                    st.append(os.stat(os.path.join(dp, fn)))
                except OSError:
                    pass
        out[str(p)] = {"files": len(st), "bytes": sum(s.st_size for s in st), "newest": max((s.st_mtime for s in st), default=None)}
    return out


def _read_json_file(p):
    """A JSON object, else the last parseable JSON line; None."""
    txt = ""
    try:
        txt = Path(p).read_text(encoding="utf-8", errors="replace")
        return json.loads(txt)
    except Exception:
        for line in reversed(txt.splitlines()):
            try:
                return json.loads(line)
            except ValueError:
                pass
    return None


def _hook_files(hook_dir, pat):
    d = Path(hook_dir) if hook_dir else None
    return [(p, _read_json_file(p)) for p in sorted(d.glob(pat))] if d and d.is_dir() else []


def hook_records(hook_dir):
    """e24_hook env_<pid>.json records (+ _file)."""
    return [dict(d, _file=p.name) for p, d in _hook_files(hook_dir, "env_*.json") if isinstance(d, dict)]


def _rank_of(d):
    """(role_tag, tp_rank) of a counters record: hook rev 2b top level, else the nested role dict (older records)."""
    r = d.get("role") if isinstance(d.get("role"), dict) else {}
    tag = d.get("role_tag") or {"sglang_scheduler": "scheduler"}.get(r.get("kind"), r.get("kind"))
    tp = d.get("tp_rank") if d.get("tp_rank") is not None else r.get("tp_rank")
    return tag, tp if isinstance(tp, int) and not isinstance(tp, bool) else None


def hook_counters(hook_dir, tp=4, role="scheduler"):
    """e24_hook counters_<pid>.json aggregated PER RANK over the `role` files with an int tp_rank in 0..tp-1 (rev 2b:
    top-level role_tag / tp_rank, the nested role only as a fallback): totals / min / max / equal per counter (equal False
    = INJECTION_DESYNC), self-test (ok on every rank), ranks found. status OK only with exactly ranks 0..tp-1, one file
    each; else INVALID_RANKS (fail closed: self_test_ok False). per_pid keeps every file."""
    per = {str(d.get("pid") or p.stem.split("_", 1)[-1]): d for p, d in _hook_files(hook_dir, "counters_*.json")
           if isinstance(d, dict)}
    keep = {pid: d for pid, d in per.items() if _rank_of(d)[0] == role and _rank_of(d)[1] in range(int(tp))}
    ranks = [_rank_of(d)[1] for d in keep.values()]
    ok_ranks = sorted(ranks) == list(range(int(tp)))
    cnt = [d.get("counters") or {} for d in keep.values()]
    keys = sorted({k for c in cnt for k, v in c.items() if isinstance(v, (int, float))})
    mn, mx = ({k: f(c.get(k, 0) for c in cnt) for k in keys} for f in (min, max))
    st = [d.get("self_test") if isinstance(d.get("self_test"), dict) else {} for d in keep.values()]
    return {"files": len(per), "rank_files": len(keep), "ranks": sorted(set(ranks)), "n_ranks": len(set(ranks)),
            "status": "OK" if ok_ranks else "INVALID_RANKS", "per_pid": per, "min": mn, "max": mx,
            "totals": {k: sum(c.get(k, 0) for c in cnt) for k in keys}, "equal": {k: mn[k] == mx[k] for k in keys},
            "self_test_ok": (all(x.get("ok") for x in st) and ok_ranks) if keep else (False if per else None),
            "self_test_missing": sorted({t for x in st for t in x.get("missing") or ()}),
            "patched": {pid: d.get("patched") or d.get("targets") for pid, d in keep.items()}}


def _rec_resolved(rec):
    r = rec.get("resolved")
    return {k: v for k, v in r.items() if isinstance(v, str)} if isinstance(r, dict) else {}


def _rec_role(rec):
    """e24_hook rev 2 role_tag (main | scheduler | detokenizer | engine_core | worker | inductor_compile | tracker | forked |
    other); never argv (the SGLang launcher's sys.argv lacks sglang.launch_server)."""
    tag = rec.get("role_tag") or (rec.get("role") if isinstance(rec.get("role"), str) else None)
    return str(tag).lower() if tag else "unknown"


def _roles(recs, engine):
    """Expected processes by role_tag (SGLang main + 4 schedulers + detokenizer; vLLM main + engine_core + 4 workers),
    else by the pid tree (trackers / inductor compile workers not counted)."""
    roles = {r.get("pid"): _rec_role(r) for r in recs}
    cnt, kids, skip = Counter(roles.values()), {}, ("tracker", "inductor_compile")
    for r in recs:
        if roles[r.get("pid")] not in skip:
            kids.setdefault(r.get("ppid"), []).append(r.get("pid"))
    roots = [r.get("pid") for r in recs if r.get("ppid") not in roles and roles[r.get("pid")] not in skip]
    need = {"main": 1, "scheduler": 4, "detokenizer": 1} if engine == "sglang" else {"main": 1, "engine_core": 1, "worker": 4}
    tree = any(len(kids.get(p, [])) >= 5 for p in roots) if engine == "sglang" else \
        any(len(kids.get(c, [])) >= 4 for p in roots for c in kids.get(p, []))
    tagged = all(cnt.get(k, 0) >= v for k, v in need.items())
    return {"counts": dict(cnt), "by_tags": tagged, "by_tree": tree, "processes": len(recs),
            "missing": [] if tagged or tree else [f"{k}x{v}" for k, v in need.items() if cnt.get(k, 0) < v]}


def _under(path, root):
    try:
        r = os.path.realpath(str(root))
        return os.path.commonpath([os.path.realpath(path), r]) == r
    except Exception:
        return False


def cache_verdict(engine, root, hook_dir, before, after, dflash=False, expect=None):
    """Section-1 isolation proof (DESIGN §3.3): expected processes logged (role_tag, else the pid tree), every logged /
    first / changed / resolved cache path UNDER the root, Triton (+ Inductor if dflash; vLLM:
    VLLM_CACHE_ROOT/torch_compile_cache + Triton) non-empty, default dirs unchanged (before / after =
    dir_stats(default_cache_dirs())). INVALID_NOT_LOGGED > INVALID_PATH_MISMATCH > INVALID_DEFAULT_DIR_WRITTEN >
    INVALID_EMPTY_CACHE > ISOLATED."""
    recs, before, after = hook_records(hook_dir), before or {}, after or {}
    out = {"engine": engine, "root": str(root), "records": len(recs), "reasons": [], "roles": _roles(recs, engine)}
    if not recs or out["roles"]["missing"]:
        out["reasons"].append("INVALID_NOT_LOGGED: " + (f"missing roles {out['roles']['missing']}" if recs else "no records"))
    bad, seen = [], {}
    names = CACHE_VARS + (("VLLM_CACHE_ROOT",) if engine == "vllm" else ())
    for r in recs:
        for k in names:                       # every copy (top level, env, cache) of every cache var, last snapshot
            vals = {v for v in (r.get(k), (r.get("env") or {}).get(k), (r.get("cache") or {}).get(k)) if isinstance(v, str) and v}
            seen.setdefault(k, set()).update(vals)
            bad += [{"pid": r.get("pid"), "var": k, "value": v} for v in sorted(vals) if not _under(v, root)] or \
                ([] if vals else [{"pid": r.get("pid"), "var": k, "value": None}])
        other = [(k, v) for k, v in (r.get("env_first") or {}).items() if k in names and v] + \
            [(c.get("var"), c.get("new")) for c in r.get("env_changes") or () if c.get("var") in names and c.get("new")] + \
            [("resolved:" + k, v) for k, v in _rec_resolved(r).items() if v.startswith("/")]
        bad += [{"pid": r.get("pid"), "var": k, "value": v} for k, v in other if not _under(v, root)]
    out["path_mismatch"] = bad[:20]
    if bad:
        out["reasons"].append(f"INVALID_PATH_MISMATCH: {len(bad)} logged paths outside the arm root")
    old = lambda p, k: (before.get(p) or {}).get(k)
    out["default_dirs_written"] = [p for p, a in after.items() if a.get("files", 0) > (old(p, "files") or 0)
                                   or (a.get("newest") or 0) > (old(p, "newest") or a.get("newest") or 0)]
    if out["default_dirs_written"]:
        out["reasons"].append(f"INVALID_DEFAULT_DIR_WRITTEN: {out['default_dirs_written']}")
    base, out["expected_nonempty"] = cache_env(engine, root), {}
    for e in expect or (["TRITON_CACHE_DIR"] + (["TORCHINDUCTOR_CACHE_DIR"] if dflash else []) if engine == "sglang" else
                        ["VLLM_CACHE_ROOT/torch_compile_cache", "TRITON_CACHE_DIR"]):
        var, _, sub = e.partition("/")
        tag = {"TRITON_CACHE_DIR": "triton", "TORCHINDUCTOR_CACHE_DIR": "inductor"}.get(var, "\0")
        cands = {os.path.join(base.get(var, os.path.join(str(root), var)), sub)} if sub else \
            {base.get(var)} | {v for v in seen.get(var, ()) if v and _under(v, root)} | \
            {v for r in recs for k, v in _rec_resolved(r).items() if tag in k.lower() and _under(v, root)}
        st = dir_stats(sorted(c for c in cands if c))
        out["expected_nonempty"][e] = {"files": sum(x["files"] for x in st.values()), "dirs": sorted(st)}
    empty = [e for e, v in out["expected_nonempty"].items() if not v["files"]]
    if empty:
        out["reasons"].append(f"INVALID_EMPTY_CACHE: {empty}")
    out["informational"] = {k: dir_stats([base[k]])[base[k]] for k in ("SGLANG_JIT_CACHE_DIR", "CUDA_CACHE_PATH",
                                                                       "FLASHINFER_WORKSPACE_BASE")}
    out["status"] = next((s for s in ("INVALID_NOT_LOGGED", "INVALID_PATH_MISMATCH", "INVALID_DEFAULT_DIR_WRITTEN",
                                      "INVALID_EMPTY_CACHE") if any(x.startswith(s) for x in out["reasons"])), "ISOLATED")
    return out


# ------------------------------------------------------------------ metrics (label-aware)
_PNAME = re.compile(r"[A-Za-z_:][A-Za-z0-9_:]*")
_PLAB = re.compile(r'\s*,?\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"')


def parse_prom(text):
    """Prometheus text -> [(name, {labels}, value)]; quoted label values may hold commas / escapes; NaN skipped."""
    out = []
    for line in (text or "").splitlines():
        line = line.strip()
        m = _PNAME.match(line) if line and line[0] != "#" else None
        if not m:
            continue
        name, i, lab = m.group(0), m.end(), {}
        if line[i:i + 1] == "{":
            i += 1
            lm = _PLAB.match(line, i)
            while lm:
                lab[lm.group(1)] = re.sub(r"\\(.)", lambda x: "\n" if x.group(1) == "n" else x.group(1), lm.group(2))
                i, lm = lm.end(), _PLAB.match(line, lm.end())
            while line[i:i + 1] in (" ", ","):
                i += 1
            if line[i:i + 1] != "}":
                continue
            i += 1
        try:
            v = float(line[i:].split()[0])
        except (ValueError, IndexError):
            continue
        if v == v:
            out.append((name, lab, v))
    return out


def prom(port):
    return parse_prom(U.urlopen(f"http://127.0.0.1:{int(port)}/metrics", timeout=15).read().decode("utf-8", "replace"))


def _lab_ok(lab, want):
    return all((lab.get(k) in v) if isinstance(v, (tuple, list, set, frozenset)) else
               bool(lab.get(k) is not None and v.fullmatch(lab[k])) if hasattr(v, "fullmatch") else lab.get(k) == v
               for k, v in (want or {}).items())


def pval(snap, name, labels=None, agg="max"):
    """One value of `name` from a prom() snapshot; labels match by value, tuple or regex; agg max | sum | tp0 (tp_rank
    "0" or none, summed) | first; None when absent."""
    vals = [(lab, v) for n, lab, v in (snap or ()) if n == name and _lab_ok(lab, labels)
            and (agg != "tp0" or lab.get("tp_rank", "0") == "0")]
    if not vals:
        return None
    return math.fsum(v for _, v in vals) if agg in ("sum", "tp0") else vals[0][1] if agg == "first" else max(v for _, v in vals)


SG = "sglang:"
VLLM = {"gen": "vllm:generation_tokens_total", "prompt": "vllm:prompt_tokens_total", "run": "vllm:num_requests_running",
        "wait": "vllm:num_requests_waiting", "preempt": "vllm:num_preemptions_total", "kvu": "vllm:kv_cache_usage_perc",
        "iter": "vllm:iteration_tokens_total_count", "acc": "vllm:spec_decode_num_accepted_tokens_total",
        "drf": "vllm:spec_decode_num_drafts_total", "dtok": "vllm:spec_decode_num_draft_tokens_total"}
_CENSUS = ("kv_available_tokens", "kv_evictable_tokens", "kv_used_tokens", "mamba_available_tokens", "mamba_evictable_tokens",
           "mamba_used_tokens", "mamba_usage", "full_token_usage", "token_usage", "max_total_num_tokens", "num_retracted_reqs")


class Metrics:
    """SGLang: gen / prefill = realtime_tokens_total{decode | prefill_compute} TP0, steps =
    cuda_graph_passes_total{decode_*} TP0 (missing mode 0), gauges = max; vLLM: the e23 counters (summed)."""

    def __init__(self, engine):
        self.engine, self.sg = engine, engine == "sglang"

    def _rt(self, s, mode):
        return pval(s, SG + "realtime_tokens_total", {"mode": mode}, "tp0")

    def gen(self, s):
        return self._rt(s, "decode") if self.sg else pval(s, VLLM["gen"], None, "sum")

    def prefill(self, s):
        return self._rt(s, "prefill_compute") if self.sg else pval(s, VLLM["prompt"], None, "sum")

    def prefill_cache(self, s):
        return self._rt(s, "prefill_cache") if self.sg else None

    def prefill_all(self, s):
        """(prefill_compute, prefill_cache)."""
        return self.prefill(s), self.prefill_cache(s)

    def running(self, s):
        return pval(s, SG + "num_running_reqs") if self.sg else pval(s, VLLM["run"], None, "sum")

    def waiting(self, s):
        return pval(s, SG + "num_queue_reqs") if self.sg else pval(s, VLLM["wait"], None, "sum")

    def kv_usage(self, s):
        return pval(s, SG + "full_token_usage") if self.sg else pval(s, VLLM["kvu"])

    def has_core(self, s):
        return (self.running(s) is not None and self.waiting(s) is not None) if self.sg else \
            pval(s, VLLM["gen"]) is not None and pval(s, VLLM["prompt"]) is not None

    def preempt(self, s, log_slice=None):
        """Retractions so far (vLLM: preemptions). SGLang counter (TP0); ABSENT = 0 while the scheduler series are live,
        gauge num_retracted_reqs not > 0 and no TP0 retraction line in log_slice; else None."""
        if not self.sg:
            return pval(s, VLLM["preempt"], None, "sum")
        c = pval(s, SG + "num_retracted_requests_total", None, "tp0")
        if c is not None:
            return c
        bad = self.running(s) is None or (pval(s, SG + "num_retracted_reqs") or 0) > 0
        return None if bad or (log_slice and retract_lines(log_slice)) else 0.0

    def steps(self, s):
        if not self.sg:
            return pval(s, VLLM["iter"], None, "sum")
        v = [pval(s, SG + "cuda_graph_passes_total", {"mode": m}, "tp0") for m in ("decode_cuda_graph", "decode_none")]
        return (0.0 if self.has_core(s) else None) if v == [None, None] else math.fsum(x for x in v if x is not None)

    def spec(self, s):
        return {"accept_length": pval(s, SG + "spec_accept_length"), "accept_rate": pval(s, SG + "spec_accept_rate")} \
            if self.sg else {k: pval(s, VLLM[k], None, "sum") for k in ("acc", "drf", "dtok")}

    def census(self, s):
        """Capacity gauges (the mamba_*_tokens gauges count SLOTS, pool_stats_observer.py:245-288) + running / queue."""
        out = {n: pval(s, SG + n) for n in _CENSUS} if self.sg else {"kv_usage": self.kv_usage(s)}
        return dict(out, running=self.running(s), waiting=self.waiting(s))


def snapshot(engine, port):
    """Gauges + counters between loads; {"error"} when /metrics fails."""
    try:
        s, M = prom(port), Metrics(engine)
        return dict(M.census(s), gen=M.gen(s), prefill=M.prefill(s), preempt=M.preempt(s), t=time.time())
    except Exception as e:
        return {"error": repr(e)[:200]}


def flush_cache(engine, port, wait=30.0, server=None):
    """Idle server, then SGLang /flush_cache ("Cache flushed"; 400 while busy) / vLLM /reset_prefix_cache."""
    idle = wait_idle(engine, port, wait, server)
    try:
        r = _http(port, "/flush_cache" if engine == "sglang" else "/reset_prefix_cache", {}, timeout=60)
        txt = str((r.get("text") if isinstance(r, dict) else r) or "")
        ok = txt.strip().startswith("Cache flushed") or engine != "sglang"
    except Exception as e:
        txt, ok = "error: " + repr(e)[:200], False
    return {"ok": ok, "idle": idle, "reply": txt.strip()[:200]}


def _last_prefill(txt):
    """The last TP0 Prefill batch line of a log (searched from the end)."""
    i = txt.rfind("Prefill batch")
    while i >= 0:
        lo = txt.rfind("\n", 0, i) + 1
        p = prefill_lines(txt[lo:txt.find("\n", i) if txt.find("\n", i) >= 0 else len(txt)])
        if p:
            return p[0]
        i = txt.rfind("Prefill batch", 0, lo)
    return None


def wait_idle(engine, port, wait=90.0, server=None):
    """Idle = two consecutive polls 1 s apart with running == 0, waiting == 0, the prefill counters unchanged and (server
    given) the TP0 Prefill line count unchanged and no chunked prefill in flight (last Prefill line #pending-token > 0
    within 15 s x TS, min 2 s): a lone chunked prefill is neither running nor queued (0.5.21). Absent gauges: not idle."""
    M, end, prev = Metrics(engine), time.perf_counter() + wait, None
    while time.perf_counter() < end:
        cur = None
        try:
            m = prom(port)
            if M.running(m) == 0 and M.waiting(m) == 0:
                txt = server.log() if server is not None else ""
                lp = _last_prefill(txt) if txt else None
                busy = lp is not None and lp["pending"] > 0 and time.time() - (lp["t"] or 0) < max(2.0, 15.0 * TS)
                cur = None if busy else (M.prefill_all(m), txt.count("Prefill batch"))
        except Exception:
            pass
        if cur is not None and cur == prev:
            return True
        prev = cur
        time.sleep(1.0)
    return False


# ------------------------------------------------------------------ server log (TP0 lines; regexes validated on the EMS logs)
_PFX = r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{3})?)(?: DP\d+)?(?: TP(?P<tp>\d+))?\] "
DECODE_RE = re.compile(_PFX + (
    r"Decode batch(?: \[(?P<iter>\d+)\])?, #running-req: (?P<running>\d+), "
    r"(?:#full token: (?P<full_tok>-?\d+), full token usage: (?P<full_use>-?[\d.]+), mamba num: (?P<mamba_num>-?\d+), "
    r"mamba usage: (?P<mamba_use>-?[\d.]+), |#token: (?P<tok>-?\d+), token usage: (?P<tok_use>-?[\d.]+), )"
    r"(?:accept len: (?P<acc_len>[\d.]+), accept rate: (?P<acc_rate>[\d.]+), )?(?:cap len: [\d.]+, )?"
    r"(?:block accept len: [\d.]+, )?.*?(?:cuda|cpu|npu|musa) graph: (?P<graph>True|False), "
    r"gen throughput \(token/s\): (?P<gen_tps>[\d.]+), #queue-req: (?P<queue>\d+)"))
PREFILL_RE = re.compile(_PFX + (
    r"Prefill batch(?: \[(?P<iter>\d+)\])?, #new-seq: (?P<new_seq>\d+), #new-token: (?P<new_tok>\d+), "
    r"#cached-token: (?P<cached>\d+), (?:full token usage: (?P<full_use>-?[\d.]+), )?(?:swa token usage: [\d.]+, )?"
    r"(?:mamba usage: (?P<mamba_use>-?[\d.]+), )?(?:token usage: (?P<tok_use>-?[\d.]+), )?#running-req: (?P<running>\d+), "
    r"#queue-req: (?P<queue>\d+), #pending-token: (?P<pending>\d+), (?:#replay-token: (?P<replay>\d+), )?.*?"
    r"(?:cuda|cpu|npu|musa) graph: (?P<graph>True|False), input throughput \(token/s\): (?P<in_tps>[\d.]+)"))
RETRACT_RE = re.compile(r"(?:KV cache pool is full\. Retract requests\.|Testing retraction\.) #retracted_reqs: (?P<n>\d+), "
                        r"#new_tokens_gained: (?P<tok>-?\d+)(?:, #mamba_num_gained: (?P<mamba>-?\d+))?"
                        r"(?:, #new_token_ratio: (?P<r0>[\d.]+) -> (?P<r1>[\d.]+))?")
_TAG = re.compile(r"^\[[^\]]*? TP(\d+)\]")


def _ts(s):
    try:
        return time.mktime(time.strptime(s[:19], "%Y-%m-%d %H:%M:%S")) + (float("0" + s[19:]) if len(s) > 19 else 0.0)
    except Exception:
        return None


def _tp0(rx, text, start, word):
    for n, line in enumerate((text or "")[start:].splitlines()):
        m = rx.match(line) if word in line else None
        if m and m.group("tp") in (None, "0"):
            yield n, m.groupdict()


def decode_lines(text, start=0):
    """TP0 "Decode batch" lines from char offset `start` (line_no in the slice; after_prefill = a Prefill line since the
    previous one)."""
    rows, pf = [], {n for n, _ in _tp0(PREFILL_RE, text, start, "Prefill batch")}
    for n, g in _tp0(DECODE_RE, text, start, "Decode batch"):
        rows.append({"t": _ts(g["ts"]), "running": int(g["running"]),
                     "tokens": _i(g["full_tok"] if g["full_tok"] is not None else g["tok"]),
                     "token_usage": _f(g["full_use"] if g["full_use"] is not None else g["tok_use"]),
                     "full_usage": _f(g["full_use"]), "mamba_num": _i(g["mamba_num"]), "mamba_usage": _f(g["mamba_use"]),
                     "accept_len": _f(g["acc_len"]), "accept_rate": _f(g["acc_rate"]), "graph": g["graph"] == "True",
                     "gen_tps": float(g["gen_tps"]), "queue": int(g["queue"]), "line_no": n,
                     "after_prefill": any(p in pf for p in range((rows[-1]["line_no"] if rows else -1) + 1, n))})
    return rows


def prefill_lines(text, start=0):
    """TP0 "Prefill batch" lines from char offset `start`."""
    return [{"t": _ts(g["ts"]), "new_seq": int(g["new_seq"]), "new_token": int(g["new_tok"]), "cached_token": int(g["cached"]),
             "running": int(g["running"]), "queue": int(g["queue"]), "pending": int(g["pending"]), "replay": _i(g["replay"]),
             "full_usage": _f(g["full_use"]), "mamba_usage": _f(g["mamba_use"]), "in_tps": float(g["in_tps"]), "line_no": n}
            for n, g in _tp0(PREFILL_RE, text, start, "Prefill batch")]


def retract_lines(text, start=0):
    """TP0 (or untagged) retraction warnings (scheduler.py:4249-4261)."""
    rows = []
    for n, line in enumerate((text or "")[start:].splitlines()):
        m = RETRACT_RE.search(line) if "etract" in line else None
        t = _TAG.match(line) if m else None
        if m and not (t and t.group(1) != "0"):
            rows.append({"n": int(m.group("n")), "tokens_gained": int(m.group("tok")), "mamba_gained": _i(m.group("mamba")),
                         "ratio": [_f(m.group("r0")), _f(m.group("r1"))], "line_no": n})
    return rows


def log_offset(server):
    """Current length (chars) of the server log, to slice "the lines of a window"."""
    return len(server.log()) if server is not None else 0


_SG_ERR = re.compile(r"Traceback|CUDA error|illegal memory|out of memory|OutOfMemory|\bOOM\b|CUDACachingAllocator|"
                     r"Can ?not alloc mamba|AssertionError|RuntimeError|no kernel image|invalid device function|"
                     r"Scheduler hit an exception|falling back to greedy argmax verification|release hit lock_ref=0|"
                     r"lock receipt anchored on node|alloc_req_slots runs out of memory|memory leak detected|"
                     r"SIGQUIT received|Subprocess \S+(?: \(pid=\d+\))? crashed|Health check failed|must match the size of tensor")
# the PyTorch allocator's C++ warning after a failed cudaMalloc it recovered from by freeing cached blocks: the EMS run of
# the BASE S20 config logged it 3 s after /flush_cache and kept serving (sglang_server_ems_dflash_w6_s20.log:1062); a
# real OOM raises torch.OutOfMemoryError ("CUDA out of memory"), which still matches (review: real platform #1)
_ALLOC_WARN = re.compile(r"CUDACachingAllocator\.cpp:\d+\] memory allocation failed with OOM")
_SG_BENIGN = re.compile(r"avoid CUDA OOM|Error in sitecustomize|No module named 'wrapt'|" + _ALLOC_WARN.pattern)
_OOM = re.compile(r"out of memory|OutOfMemory|\bOOM\b|CUDACachingAllocator|runs out of memory")
_BADK = re.compile(r"no kernel image|invalid device function")


def log_errors(text):
    """Error-sign lines up to our own SIGTERM (later SIGQUIT / exit -15 lines are kill noise); benign Triton pre-load
    OOM warnings excluded."""
    body, _, tail = (text or "").partition("SIGTERM received")
    err = [l.strip()[:300] for l in body.splitlines() if _SG_ERR.search(l) and not _SG_BENIGN.search(l)]
    return {"error_lines": err[:12], "n_error_lines": len(err), "oom_lines": [l for l in err if _OOM.search(l)][:12],
            "shutdown_seen": bool(tail), "mamba_alloc_fail": sum(1 for l in err if re.search(r"Can ?not alloc mamba", l)),
            "cuda_invalid_kernel": any(_BADK.search(l) for l in err),
            "greedy_fallback": sum(1 for l in err if "falling back to greedy argmax verification" in l),
            "allocator_oom_warnings": sum(1 for l in body.splitlines() if _ALLOC_WARN.search(l))}   # memory pressure, recovered


def _facts_sglang(t):
    """Startup facts of an SGLang 0.5.21 log (TP0 / untagged lines; EMS log layout)."""
    f, tp0 = {"engine": "sglang", "server_args": None}, []
    for l in t.splitlines():
        m = _TAG.match(l)
        if not m or m.group(1) == "0":
            tp0.append(l)
        m2 = re.search(r"server_args=(\{.*\})\s*$", l) if f["server_args"] is None and "server_args={" in l else None
        try:
            f["server_args"] = ast.literal_eval(m2.group(1)) if m2 else f["server_args"]
        except Exception as e:
            f["server_args_error"] = repr(e)[:200]
    z = "\n".join(tp0)
    one = lambda pat, conv=_num: (lambda m: conv(m.group(1)) if m else None)(re.search(pat, z))
    cap = re.search(r"max_total_num_tokens=(\d+), chunked_prefill_size=(-?\d+), max_prefill_tokens=(\d+), "
                    r"max_running_requests=(\d+), context_len=(\d+), available_gpu_mem=([\d.]+) GB", z)
    for i, k in enumerate(("kv_tokens", "chunked_prefill_size", "max_prefill_tokens", "max_running", "context_len",
                           "available_gpu_mem_gb")):
        f[k] = _num(cap.group(i + 1)) if cap else None
    ms = re.search(r"conv_state size: ([\d.]+)GB, ssm_state size: ([\d.]+)GB(?: intermediate_ssm_state_cache size: "
                   r"([\d.]+)GB)?", z)
    ends = [(m.group(1), float(m.group(2))) for m in re.finditer(r"Capture (.+?) CUDA graph end\. elapsed=([\d.]+) s", z)]
    f.update(max_running_capped=one(r"max_running_requests is capped to (\d+) by the mamba state cache"),
             mamba_cache_size=one(r"Mamba Cache is allocated\. max_mamba_cache_size: (\d+)"),
             mamba_sizes_gb=[_f(x) for x in ms.groups()] if ms else None,
             kv_pools=[[m.group(1), int(m.group(2))]
                       for m in re.finditer(r"KV Cache is allocated\. dtype: ([\w.]+), #tokens: (\d+)", z)],
             graphs=[{"kind": m.group(1), "backend": m.group(2), "num_tokens_per_req": int(m.group(3)),
                      "bs": [int(x) for x in m.group(4).replace(" ", "").split(",") if x],
                      "elapsed": next((e for k, e in ends if k == m.group(1)), None)} for m in re.finditer(
                 r"Capture (.+?) CUDA graph begin\. backend=(\w+), num_tokens_per_req=(\d+), bs=\[([\d, ]+)\]", z)],
             capture_sec=round(sum(e for _, e in ends), 2) if ends else None,
             load_weight=[{"elapsed": float(m.group(1)), "type": m.group(2), "mem_gb": float(m.group(3))} for m in re.finditer(
                 r"Load weight end\. elapsed=([\d.]+) s, type=(\w+).*?mem usage=([\d.]+) GB", z)],
             nccl_version=one(r"sglang is using nccl==(\S+)", str), custom_ar_disabled="CustomAllreduce is disabled" in t,
             prefill_graph_disabled="Disable prefill CUDA graph" in z,
             dflash_lines=[l.strip()[:200] for l in tp0 if "DFLASH" in l or "GDN kernel dispatcher" in l][:6],
             fp8_scale_lines=sum(1 for l in tp0 if "Using FP8 KV cache but no scaling factors" in l),
             ready_line="The server is fired up and ready to roll!" in t, retract_lines=len(retract_lines(t)))
    f["cuda_graph_bs"] = f["graphs"][0]["bs"] if f["graphs"] else []
    st = re.search(r"Engine startup timings \(s\): (.*)$", z, re.M)
    f["startup_timings"] = {k: float(v) for k, v in re.findall(r"(\w+)=([\d.]+)", st.group(1))} if st else None
    t0 = re.search(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)", t, re.M)
    rl = re.search(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d(?:\.\d+)?)[^\]]*\] The server is fired up", t, re.M)
    f["ready_log_sec"] = round(_ts(rl.group(1)) - _ts(t0.group(1)), 1) if rl and t0 else None
    return dict(f, nccl=nccl_info(t), **log_errors(t))


def _facts_vllm(t):
    """What vLLM 0.32 logged at startup (e23 rev 3 regexes)."""
    kv = [int(x.replace(",", "")) for x in re.findall(r"GPU KV cache size: ([\d,]+) tokens", t)]
    mc = re.findall(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", t)
    pick = lambda pat, n=8: [l.strip()[:240] for l in t.splitlines() if re.search(pat, l, re.I)][:n]
    errs = pick(r"Traceback|CUDA error|illegal memory|out of memory|RuntimeError|AssertionError|no kernel image|"
                r"invalid device function", 12)
    return {"engine": "vllm", "kv_tokens": min(kv) if kv else None, "kv_tokens_all": kv,
            "max_concurrency": float(mc[-1]) if mc else None, "kernels": sorted(set(re.findall(r"Selected (\w+Kernel) for", t))),
            "error_lines": errs, "n_error_lines": len(errs), "oom_lines": [l for l in errs if "out of memory" in l],
            "cuda_invalid_kernel": any(_BADK.search(l) for l in errs),
            "graph_capture_modes": sorted(set(re.findall(r"Capturing CUDA graphs \(([^)]+)\)", t))),
            "kv_dtype_lines": pick(r"kv_cache_dtype|kv cache dtype|scaling factor"), "nccl": nccl_info(t)}


# ------------------------------------------------------------------ NCCL (NCCL_DEBUG=INFO lines in the server log)
_NC = [(re.compile(r"NCCL version\s+(\S+)"), "version"), (re.compile(r"(NCCL_[A-Z0-9_]+)\s+set by environment to\s+(\S+)"), "env"),
       (re.compile(r"Channel\s+\d+(?:/\d+)?\s*:\s*\d+\[\d+\]\s*->\s*\d+\[\d+\].*?via\s+(\S+)"), "conn"),
       (re.compile(r"Channel\s+\d+/(\d+)\s*:\s*\d+(?:\s+\d+)+\s*$"), "ring"), (re.compile(r"(\d+)\s+coll channels"), "coll")]
_NC_ERR = re.compile(r"NCCL error|ncclInternalError|ncclSystemError|ncclUnhandledCudaError|unhandled (?:cuda|system) error",
                     re.I)


def nccl_info(text):
    """NCCL version, "<VAR> set by environment" (env_applied), channels from "Channel xx/NN" ring lines (else coll
    channels), transports, WARN / error lines."""
    m = re.search(r"sglang is using nccl==(\S+)", text or "")
    out = {"version": None, "sglang_nccl": m.group(1) if m else None, "env_applied": {}, "transport": {}, "warnings": [],
           "errors": [], "lines": 0}
    rings, coll, tr = set(), set(), Counter()
    for line in (text or "").splitlines():
        if _NC_ERR.search(line) and len(out["errors"]) < 20:
            out["errors"].append(line.strip()[:300])
        i, j = line.find("NCCL INFO"), line.find("NCCL WARN")
        if i < 0 and j < 0:
            continue
        out["lines"] += 1
        msg = line[(i if i >= 0 else j) + 9:].strip()
        if i < 0:
            out["warnings"] = (out["warnings"] + [msg[:300]])[:30]
            continue
        for rx, kind in _NC:
            m = rx.search(msg)
            if m and kind == "version":
                out["version"] = m.group(1)
            elif m and kind == "env":
                out["env_applied"][m.group(1)] = m.group(2).rstrip(".")
            elif m:
                {"conn": lambda: tr.update([m.group(1)]), "ring": lambda: rings.add(int(m.group(1))),
                 "coll": lambda: coll.add(int(m.group(1)))}[kind]()
    return dict(out, channels=max(rings) if rings else (max(coll) if coll else None), coll_channels=sorted(coll) or None,
                ring_channel_counts=sorted(rings), transport=dict(tr.most_common()))


def nccl_env_proof(env, nccl, trace=None):
    """DESIGN §4.3 env proof: each NCCL_* knob read by NCCL ("set by environment"; CTAs: or the channel count) AND, for
    ALGO / PROTO / CTAs, the TP0 trace's all-reduce kernels agree (algo / proto in the kernel name, grid.x = CTAs).
    APPLIED | INVALID_ENV_NOT_APPLIED (a contradiction) | INCONCLUSIVE_NO_TRACE_EVIDENCE (the trace lacks the field) |
    INCONCLUSIVE_NO_NCCL_LINES."""
    want = {k: str(v) for k, v in (env or {}).items() if k.startswith("NCCL_") and not k.startswith("NCCL_DEBUG")}
    if not (nccl or {}).get("lines"):
        return {"status": "INCONCLUSIVE_NO_NCCL_LINES", "checks": [], "want": want}
    tn = (trace or {}).get("nccl") or {}
    seen = {"algo": set(tn.get("allreduce_algo") or ()), "proto": set(tn.get("allreduce_proto") or ()),
            "grid": set(tn.get("allreduce_grid_x") or ())}
    checks = []
    for k, v in sorted(want.items()):
        read, ch = nccl.get("env_applied", {}).get(k), nccl.get("channels")
        # PROTO: log line only. Since NCCL 2.19 a kernel is named after its specialised variant (_RING_LL / _TREE_LL) and
        # picks the protocol inside: the real E14A trace names the 11.9 MB all-gather ncclDevKernel_AllGather_RING_LL,
        # a size NCCL never runs as LL (review: real platform #2). The algorithm and the grid stay trace-checked.
        kind, exp = ("grid", {int(v)}) if re.match(r"NCCL_(MIN|MAX)_(CTAS|NCHANNELS)$", k) and v.isdigit() else \
            ("algo", {v.split(":")[-1].split(",")[0].upper()}) if k == "NCCL_ALGO" else (None, None)
        c = {"var": k, "want": v, "read": read, "log_ok": read == v or (kind == "grid" and str(ch) == v), "channels": ch}
        if kind:
            c.update(trace=sorted(seen[kind]), trace_ok=(seen[kind] == exp) if seen[kind] else None)
        c["ok"] = c["log_ok"] and c.get("trace_ok") is not False
        checks.append(c)
    st = "INVALID_ENV_NOT_APPLIED" if not all(c["ok"] for c in checks) else \
        "INCONCLUSIVE_NO_TRACE_EVIDENCE" if any("trace" in c and c["trace_ok"] is None for c in checks) else "APPLIED"
    return {"status": st, "checks": checks, "trace_used": bool(tn)}


# ------------------------------------------------------------------ one streaming /v1/completions request (both engines)
def _tok_id(k):
    k = str(k)[9:] if str(k).startswith("token_id:") else str(k)
    return int(k) if k.lstrip("-").isdigit() else None


class Stream(threading.Thread):
    """Streaming /v1/completions (both engines, e23 body): per-chunk token_ids, [DONE] + finish_reason required,
    chunk_times [(t, n_new)], the final usage chunk (cached tokens); SGLang gets an abortable rid and no vLLM-only
    keys."""

    def __init__(self, port, ids, max_new, temperature=1.0, seed=None, logprobs=0, ignore_eos=True, model="default",
                 engine="sglang", rid=None, top_p=None, top_k=None, timeout=7200):
        super().__init__(daemon=True)
        self.port, self.engine, self.timeout, self.n_prompt = int(port), engine, timeout, len(ids)
        self.rid = rid or (_rid("s") if engine == "sglang" else None)
        b = {"model": model or "default", "prompt": list(ids), "max_tokens": int(max_new), "temperature": temperature,
             "stream": True, "ignore_eos": bool(ignore_eos), "return_token_ids": True, "stream_options": {"include_usage": True},
             "top_p": (0.95 if temperature else 1.0) if top_p is None else top_p}
        if temperature or top_k is not None:
            b["top_k"] = 20 if top_k is None else top_k
        if seed is not None:
            b["seed"] = int(seed)
        if logprobs:
            b["logprobs"] = int(logprobs)
            if engine == "vllm":
                b["return_tokens_as_token_ids"] = True          # vLLM: logprob keys by id
        if engine == "sglang":
            b["rid"] = self.rid
        self.body = b
        self.text, self.tokens, self.margins, self.top2, self.lps, self.chunk_times = [], [], [], [], [], []
        self.err, self.done_seen, self.finish_reason, self.usage = None, False, None, None
        self.t_start = self.t_first = self.t_end = None
        self._resp, self._halt = None, False

    def _choice(self, c):
        self.text.append(c.get("text") or "")
        ids_, lp = c.get("token_ids"), c.get("logprobs") or {}
        if ids_ is None:
            if c.get("text") or lp.get("tokens"):
                self.err = "MISSING_STREAM_TOKEN_IDS"
        elif any(not isinstance(v, int) or isinstance(v, bool) for v in ids_):
            self.err = "INVALID_STREAM_TOKEN_IDS"
        elif ids_:
            now = time.perf_counter()
            self.t_first = self.t_first or now
            self.tokens.extend(ids_)
            self.chunk_times.append((now, len(ids_)))
        self.lps.extend(lp.get("token_logprobs") or [])
        for top in lp.get("top_logprobs") or []:
            it = sorted(((v, _tok_id(k)) for k, v in (top or {}).items() if isinstance(v, (int, float))), key=lambda z: -z[0])
            self.top2.append([it[0][1], it[0][0]] + ([it[1][1], it[1][0]] if len(it) > 1 else [None, None]) if it else None)
            self.margins.append(round(it[0][0] - it[1][0], 5) if len(it) > 1 else None)
        if c.get("finish_reason") is not None:
            self.finish_reason = c["finish_reason"]

    def run(self):
        self.t_start = time.perf_counter()
        try:
            req = U.Request(f"http://127.0.0.1:{self.port}/v1/completions", data=json.dumps(self.body).encode(),
                            headers={"Content-Type": "application/json"})
            self._resp = U.urlopen(req, timeout=self.timeout)
            for raw in self._resp:
                line = raw.strip()
                if self._halt:
                    break
                if not line.startswith(b"data:"):
                    continue
                if line[5:].strip() == b"[DONE]":
                    self.done_seen = True
                    break
                d = json.loads(line[5:].strip())
                if "error" in d or d.get("object") == "error":
                    self.err = ("stream error: " + json.dumps(d.get("error") or d.get("message") or d))[:300]
                    break
                self.usage = d.get("usage") or self.usage
                for c in d.get("choices") or []:
                    self._choice(c)
                if self.err:
                    break
        except U.HTTPError as e:            # e.g. 400: prompt + max_tokens > context length (tokenizer_manager.py:1306)
            self.err = f"HTTP {e.code}: {_body(e)[:240]}"
        except Exception as e:
            if not self._halt:
                self.err = repr(e)[:300]
        finally:                         # e23 rev 3: a stop() before the response existed still closes the socket here
            try:
                if self._resp is not None:
                    self._resp.close()
            except Exception:
                pass
        self.t_end = time.perf_counter()
        if not self._halt and not self.err and not (self.done_seen and self.finish_reason):
            self.err = f"INCOMPLETE_SSE done={self.done_seen} finish={self.finish_reason}"

    def stop(self):
        self._halt = True
        for f in (lambda: self._resp.fp.raw._sock.shutdown(socket.SHUT_RDWR), lambda: self._resp.close()):
            try:
                f()
            except Exception:
                pass

    def cached_tokens(self):
        d = (self.usage or {}).get("prompt_tokens_details")
        return d.get("cached_tokens") if isinstance(d, dict) else None

    def row(self):
        return {"text": "".join(self.text), "tokens": list(self.tokens), "margins": list(self.margins),
                "top2": list(self.top2), "lps": list(self.lps), "error": self.err, "done": self.done_seen,
                "finish_reason": self.finish_reason, "cached_tokens": self.cached_tokens(), "n_prompt": self.n_prompt}


def _spawn(engine, port, prompts, n, max_new, temperature, seed, model):
    ss = [Stream(port, prompts[i % len(prompts)], max_new, temperature, seed + i, model=model, engine=engine) for i in range(n)]
    for g in ss:
        g.start()
    return ss


def _abort_rid(port, rid):
    try:
        _http(port, "/abort_request", {"rid": rid}, timeout=5)
    except Exception:
        pass


def stop_all(engine, port, streams, budget=15.0):
    """Stop streams, SGLang /abort_request per rid, joins in one budget; returns the client threads still alive."""
    for g in streams:
        g.stop()
    ths = [threading.Thread(target=_abort_rid, args=(port, g.rid), daemon=True) for g in streams if g.rid and engine == "sglang"]
    for t in ths:
        t.start()
    end = time.monotonic() + budget
    for t in ths + [g for g in streams if g.ident is not None]:
        t.join(max(0.0, end - time.monotonic()))
    return sum(1 for g in streams if g.is_alive())


# ------------------------------------------------------------------ correctness: greedy rows, logged generation, teacher forcing
def _sg_sampling(max_new, temperature, ignore_eos, top_p=None, top_k=None, seed=None):
    sp = {"max_new_tokens": int(max_new), "temperature": float(temperature or 0.0), "ignore_eos": bool(ignore_eos),
          "top_p": (0.95 if temperature else 1.0) if top_p is None else top_p,
          "top_k": (20 if temperature else 1) if top_k is None else top_k}
    if seed is not None:
        sp["sampling_seed"] = int(seed)         # used only under --enable-deterministic-inference (harmless otherwise)
    return sp


def _sg_post(port, body, timeout):
    r = _http(port, "/generate", dict(body, rid=_rid("g"), return_text_in_logprobs=False), timeout)
    r = r[0] if isinstance(r, list) and r else r
    if not isinstance(r, dict) or r.get("error"):
        raise RuntimeError("generate error: " + json.dumps(r.get("error") if isinstance(r, dict) else r)[:300])
    return r


def _sg_out(r):
    """tokens / lps / top-k ([[id, lp]] best first) / meta of a native /generate reply."""
    mi = r.get("meta_info") or {}
    otl = mi.get("output_token_logprobs") or []
    toks = list(r.get("output_ids") or [x[1] for x in otl])
    if otl and [x[1] for x in otl] != toks:
        raise RuntimeError("LOGPROB_ID_MISMATCH output_token_logprobs ids != output_ids")
    fr = mi.get("finish_reason")
    return {"tokens": toks, "lps": [x[0] for x in otl], "text": r.get("text") or "",
            "top": [[[i, lp] for lp, i in sorted(((x[0], x[1]) for x in (t or []) if x and x[0] is not None), key=lambda z: -z[0])]
                    for t in mi.get("output_top_logprobs") or []],
            "finish_reason": fr.get("type") if isinstance(fr, dict) else fr, "cached_tokens": mi.get("cached_tokens"),
            "accept_len": mi.get("spec_accept_length"), "completion_tokens": mi.get("completion_tokens")}


def _run_all(fns, concurrent):
    out = [None] * len(fns)

    def one(i):
        try:
            out[i] = fns[i]()
        except Exception as e:
            out[i] = {"error": f"{type(e).__name__}: {e}"[:300]}
    th = [threading.Thread(target=one, args=(i,), daemon=True) for i in range(len(fns))]
    for t in th:
        t.start()
        if not concurrent:
            t.join()
    for t in th:
        t.join()
    return out


def _prompts_keyed(prompts, keys=None):
    if isinstance(prompts, dict):
        return [(str(k), list(v)) for k, v in prompts.items()]
    return [(p[0], list(p[1])) if isinstance(p, (tuple, list)) and len(p) == 2 and isinstance(p[0], str) else
            (str(keys[i]) if keys else prompt_key(p), list(p)) for i, p in enumerate(prompts)]


def _stream_once(engine, port, ids, max_new, timeout, **kw):
    g = Stream(port, ids, max_new, engine=engine, timeout=timeout, **kw)
    g.start()
    g.join(timeout)
    late = g.is_alive()
    if late:
        g.stop()
        g.join(10)
    return dict(g.row(), error=g.err or ("TIMEOUT" if late else None))


def greedy(engine, port, prompts, n_tok=256, top=2, timeout=900, model=None, concurrent=False, keys=None, raw_dir=None,
           tag=None):
    """Greedy oracle rows (concurrent=True: the c16 rows); prompts = id lists (key = prompt hash), {key: ids} or (key,
    ids). Rows {key, tokens, margins, top2 [[id1, lp1, id2, lp2]], lps, text, error, finish_reason, elapsed,
    cached_tokens}. SGLang native /generate; vLLM the e23 Stream path."""
    items, model = _prompts_keyed(prompts, keys), _model(port, model)

    def make(k, ids):
        def run():
            t0 = time.perf_counter()
            try:
                if engine == "sglang":
                    o = _sg_out(_sg_post(port, {"input_ids": ids, "sampling_params": _sg_sampling(n_tok, 0.0, True),
                                                "return_logprob": True, "top_logprobs_num": int(top), "logprob_start_len": -1},
                                         timeout))
                    row = dict(o, error=None, top2=[t[0] + (t[1] if len(t) > 1 else [None, None]) if t else None for t in o["top"]],
                               margins=[round(t[0][1] - t[1][1], 5) if len(t) > 1 else None for t in o["top"]])
                    row.pop("top")
                else:
                    row = _stream_once(engine, port, ids, n_tok, timeout, temperature=0.0, logprobs=top, model=model)
            except Exception as e:
                row = {"tokens": [], "margins": [], "top2": [], "lps": [], "text": "", "finish_reason": None,
                       "cached_tokens": None, "error": f"{type(e).__name__}: {e}"[:300]}
            row.update(key=k, n_prompt=len(ids), elapsed=round(time.perf_counter() - t0, 2))
            if not row["error"] and row["finish_reason"] == "abort":
                row["error"] = "ABORTED"
            elif not row["error"] and row["finish_reason"] == "length" and len(row["tokens"]) != n_tok:
                row["error"] = f"LENGTH_MISMATCH {len(row['tokens'])} != {n_tok}"
            return row
        return run
    rows = _run_all([make(k, ids) for k, ids in items], concurrent)
    _raw(raw_dir, f"greedy_{_tag(tag)}.json", rows)
    return rows


def generate_logged(engine, port, ids, max_new, temperature=0.0, top=1, timeout=7200, model=None, ignore_eos=True,
                    top_p=None, top_k=None, eos_ids=(), seed=None, key=None, raw_dir=None, tag=None):
    """Long generation with decode-time logprobs (DFlash: target verify logprobs); T>0 -> top_p 0.95 / top_k 20 unless
    given; natural_eos_pos = first eos_ids token (ignore_eos keeps going)."""
    t0, out = time.perf_counter(), {"key": key or prompt_key(ids), "n_prompt": len(ids), "max_new": max_new,
                                    "temperature": temperature}
    try:
        if engine == "sglang":
            o = _sg_out(_sg_post(port, {"input_ids": list(ids), "return_logprob": True, "logprob_start_len": -1,
                                        "top_logprobs_num": int(top),
                                        "sampling_params": _sg_sampling(max_new, temperature, ignore_eos, top_p, top_k, seed)},
                                 timeout))
        else:
            r = _stream_once(engine, port, ids, max_new, timeout, temperature=temperature, seed=seed, logprobs=max(1, int(top)),
                             ignore_eos=ignore_eos, model=_model(port, model), top_p=top_p, top_k=top_k)
            if r["error"]:
                raise RuntimeError(r["error"])
            o = dict(r, top=[[t[0:2], t[2:4]] if t else [] for t in r["top2"]], accept_len=None)
        eos = set(eos_ids or ())
        out.update({k: o.get(k) for k in ("tokens", "lps", "top", "text", "finish_reason", "cached_tokens", "accept_len")},
                   n=len(o["tokens"]), natural_eos_pos=next((k for k, t in enumerate(o["tokens"]) if t in eos), None),
                   status="MEASURED" if len(o["lps"]) == len(o["tokens"]) else "INCONCLUSIVE_LOGPROBS_MISSING")
    except Exception as e:
        out.update(status="ERROR_REQUEST", error=f"{type(e).__name__}: {e}"[:300], tokens=[], lps=[], n=0)
    out["elapsed"] = round(time.perf_counter() - t0, 1)
    _raw(raw_dir, f"gen_{_tag(tag)}_{out['key']}.json", out)
    return out


def generate_logged_many(engine, port, prompts, max_new, temperature=1.0, top=1, seeds=None, timeout=7200, model=None,
                         ignore_eos=True, top_p=None, top_k=None, eos_ids=(), keys=None, raw_dir=None, tag=None):
    """Concurrent generate_logged (the c16 production-sampled M4 traces) with per-stream seeds (default 1000 + i)."""
    items, model = _prompts_keyed(prompts, keys), _model(port, model)
    seeds = list(seeds) if seeds is not None else [1000 + i for i in range(len(items))]
    return _run_all([(lambda k=k, ids=ids, s=s: generate_logged(engine, port, ids, max_new, temperature, top, timeout, model,
                                                                 ignore_eos, top_p, top_k, eos_ids, s, k, raw_dir, tag))
                     for (k, ids), s in zip(items, seeds)], True)


def _top_row(lp, ents, top):
    """[lp_actual, top1_id, top1_lp, top2_id, top2_lp, ...]."""
    s = sorted((e for e in ents if e[0] is not None), key=lambda z: -z[0])
    return [lp] + [x for j in range(top) for x in ([s[j][1], s[j][0]] if j < len(s) else [None, None])]


def teacher_forced(engine, port, prompt_ids, cont_ids, top=2, timeout=1800, model=None, raw_dir=None, tag=None):
    """positions[k] = [lp(cont[k] | prompt+cont[:k]), top1_id, top1_lp, top2_id, top2_lp]. SGLang: logprob_start_len =
    len(prompt)-1, entry 0 dropped, length and ids asserted else INCONCLUSIVE_TF_ALIGN
    (SGLANG_LOGPROB_CHUNK_SIZE=128, one at a time); vLLM: prompt_logprobs[p] = lp(ids[p] | ids[:p])."""
    t0, prompt, cont = time.perf_counter(), list(prompt_ids), list(cont_ids)
    out = {"key": prompt_key(prompt + cont), "n_prompt": len(prompt), "n": len(cont), "top": top, "positions": None}
    align = lambda msg: dict(out, status="INCONCLUSIVE_TF_ALIGN", error=msg[:300])
    if not prompt or not cont:
        return align("empty prompt or continuation")
    try:
        if engine == "sglang":
            mi = _sg_post(port, {"input_ids": prompt + cont, "sampling_params": _sg_sampling(1, 0.0, True), "return_logprob": True,
                                 "logprob_start_len": len(prompt) - 1, "top_logprobs_num": int(top)}, timeout)
            mi = mi.get("meta_info") or {}
            itl, itop = mi.get("input_token_logprobs") or [], mi.get("input_top_logprobs") or []
            if len(itl) != len(cont) + 1 or (top and len(itop) != len(cont) + 1):
                return align(f"input logprob entries {len(itl)} / {len(itop)} != len(cont)+1 = {len(cont) + 1}")
            if itl[0][1] != prompt[-1] or [e[1] for e in itl[1:]] != cont:
                return align("entry ids != [prompt[-1]] + continuation")
            pos = [_top_row(e[0], [(x[0], x[1]) for x in (itop[k + 1] or []) if x] if top else [], top)
                   for k, e in enumerate(itl[1:])]
        else:
            ids = prompt + cont
            r = _http(port, "/v1/completions", {"model": _model(port, model), "prompt": ids, "max_tokens": 1, "temperature": 0,
                                                "prompt_logprobs": max(1, int(top)), "return_tokens_as_token_ids": True}, timeout)
            pl = ((r.get("choices") or [{}])[0] or {}).get("prompt_logprobs") or []
            if len(pl) != len(ids):
                return align(f"prompt_logprobs {len(pl)} entries != {len(ids)}")
            pos = []
            for p in range(len(prompt), len(ids)):
                ent = [((v.get("logprob") if isinstance(v, dict) else v), _tok_id(k)) for k, v in (pl[p] or {}).items()]
                act = next((e for e in ent if e[1] == ids[p]), None)
                if act is None:
                    return align(f"prompt_logprobs[{p}] lacks the actual token {ids[p]}")
                pos.append(_top_row(act[0], ent, top))
        out.update(status="MEASURED", positions=pos)
    except Exception as e:
        out.update(status="ERROR_REQUEST", error=f"{type(e).__name__}: {e}"[:300])
    out["elapsed"] = round(time.perf_counter() - t0, 1)
    _raw(raw_dir, f"tf_{_tag(tag)}_{out['key']}.json", out)
    return out


def next_token_dist(port, ids, vocab_n, timeout=600):
    """SGLang full-vocab next-token logprobs (E17: token_ids_logprob = range(V)), renormalised (float64)."""
    try:
        mi = _sg_post(port, {"input_ids": list(ids), "sampling_params": _sg_sampling(1, 0.0, True), "return_logprob": True,
                             "logprob_start_len": -1, "token_ids_logprob": list(range(int(vocab_n)))}, timeout)
        mi = mi.get("meta_info") or {}
        pairs = (mi.get("output_token_ids_logprobs") or [None])[0] or []
        if len(pairs) != vocab_n or any(int(x[1]) != i for i, x in enumerate(pairs)):
            return {"error": f"no / wrong output_token_ids_logprobs ({len(pairs)} entries for {vocab_n})"}
        lp = [float(x[0]) for x in pairs]
        if not all(math.isfinite(x) for x in lp):
            return {"error": "non-finite logprobs"}
        s = _lse(lp)
        return {"lp": [x - s for x in lp], "sum_raw": round(math.exp(s), 6), "cached": mi.get("cached_tokens"), "n": len(lp)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:300]}


def _lse(v):
    m = max(v)
    return m + math.log(math.fsum(math.exp(x - m) for x in v))


def kl(p_lp, q_lp):
    """KL(p || q), float64, both renormalised; None on bad input."""
    if not p_lp or not q_lp or len(p_lp) != len(q_lp):
        return None
    p, q = [float(x) for x in p_lp], [float(x) for x in q_lp]
    if any(x != x for x in p + q):
        return None
    sp, sq = _lse(p), _lse(q)
    return math.fsum(math.exp(a - sp) * ((a - sp) - (b - sq)) for a, b in zip(p, q) if a - sp > -745.0)


def degenerate(text):
    """Garbage: empty, a 200-char run, '!!!!', very low variety."""
    t = text or ""
    if not t.strip():
        return "EMPTY"
    if re.search(r"(\S)\1{200,}", t, re.S):      # e23 rev 3: grid rows / separators are not garbage
        return "REPEATED_CHAR"
    if t.count("!") > 0.5 * max(1, len(t)):
        return "BANG_GARBAGE"
    words = [w for w in re.findall(r"\w+", t) if not w.isdigit()]   # space-separated grid digits are ARC content, not
    return "LOW_VARIETY" if len(words) > 40 and len(set(words)) < 0.08 * len(words) else None   # garbage (review: real platform #3)


def _at(v, k):
    return v[k] if isinstance(v, (list, tuple)) and k < len(v) else None


def compare(ref, cand, kind="same_weights", t_pair=FAIL_MARGIN, aa_high=None):
    """Aligned oracle pair (DESIGN §3.2): rows by "key" (else index), truncated to the common length; no divergence =
    EXACT_TO_N. First divergence: m_sum = m_ref + m_cand -> STATE_FLIP (>= 4, whatever T_pair: a calibrated T_pair
    above 4 never hides one) / NEAR_TIE (< T_pair, floor 0.5) / FLIP;
    swap = cand token is the ref top-2. same_weights: FAIL_HIGH_MARGIN iff FLIP+STATE_FLIP > aa_high + 1, else any is
    BOUNDARY; cross_weights: FAIL_STATE_FLIP or MEASURED; garbage (ref clean) -> FAIL_HIGH_MARGIN (fail_reason
    GARBAGE); missing rows / margins -> INCONCLUSIVE."""
    tp = max(FAIL_MARGIN, float(t_pair or 0.0))
    R, C = ({_key(r, i): r for i, r in enumerate(x or [])} for x in (ref, cand))
    keys = [k for k in R if k in C]
    res = {"kind": kind, "t_pair": tp, "aa_high": aa_high, "n_pairs": len(keys), "missing_keys": sorted(set(R) ^ set(C))[:20],
           "rows": [], "counts": {}, "high": 0, "positions_examined": 0, "swaps": 0, "garbage": 0}
    if not keys:
        return dict(res, status="INCONCLUSIVE_PAIRS_MISSING")
    cnt = Counter()
    for k in keys:
        a, b = R[k], C[k]
        row = {"key": k, "n": None, "first_diff": None, "m_ref": None, "m_cand": None, "m_sum": None, "swap": False}
        if a.get("error") or b.get("error") or not a.get("tokens") or not b.get("tokens"):
            cls, row["why"] = "INCONCLUSIVE", (a.get("error") or b.get("error") or "empty row")[:120]
        else:
            x, y = a["tokens"], b["tokens"]
            row["n"] = n = min(len(x), len(y))
            d = next((i for i in range(n) if x[i] != y[i]), None)
            res["positions_examined"] += n if d is None else d
            cls = "EXACT_TO_N"
            if d is not None:
                mr, mc, t2 = _at(a.get("margins"), d), _at(b.get("margins"), d), _at(a.get("top2"), d)
                row.update(first_diff=d, m_ref=mr, m_cand=mc, ref_tok=x[d], cand_tok=y[d],
                           swap=bool(t2 and len(t2) >= 3 and t2[2] is not None and t2[2] == y[d]))
                if mr is None or mc is None:
                    cls, row["why"] = "INCONCLUSIVE", "missing margin at the first divergence"
                else:
                    row["m_sum"] = s = round(mr + mc, 5)
                    cls = "STATE_FLIP" if s >= STATE_FLIP_MARGIN else ("NEAR_TIE" if s < tp else "FLIP")
        g = degenerate(b.get("text")) if b.get("tokens") else None
        row.update(garbage=g if (g and a.get("tokens") and degenerate(a.get("text")) is None) else None, **{"class": cls})
        cnt[cls] += 1
        res["swaps"] += bool(row["swap"])
        res["garbage"] += bool(row["garbage"])
        res["rows"].append(row)
    high, incon = cnt["FLIP"] + cnt["STATE_FLIP"], cnt["INCONCLUSIVE"]
    res.update(counts=dict(cnt), high=high, state_flips=cnt["STATE_FLIP"])
    if res["garbage"]:
        st, res["fail_reason"] = "FAIL_HIGH_MARGIN", "GARBAGE"
    elif kind == "cross_weights":
        st = "FAIL_STATE_FLIP" if cnt["STATE_FLIP"] else ("INCONCLUSIVE" if incon == len(keys) else "MEASURED")
    elif aa_high is not None and high > aa_high + 1:
        st = "FAIL_HIGH_MARGIN"
    else:
        st = "BOUNDARY" if high else "INCONCLUSIVE" if incon else "NEAR_TIE" if cnt["NEAR_TIE"] else "EXACT_TO_N"
    if incon and st in ("BOUNDARY", "MEASURED"):
        res["inconclusive_rows"] = incon
    return dict(res, status=st)


def aa_high_of(cmps):
    """A/A allowance: max FLIP+STATE_FLIP over usable A/A comparisons; None if none."""
    v = [c.get("high") for c in cmps or () if isinstance(c, dict) and isinstance(c.get("high"), int)
         and not str(c.get("status", "")).startswith("INCONCLUSIVE")]
    return max(v) if v else None


def _tf_pairs(ref, cand):
    """[(k, pos_ref, pos_cand)] over aligned TF results; None when unusable."""
    ra, ca = (ref if isinstance(ref, list) else [ref]), (cand if isinstance(cand, list) else [cand])
    if not ra or len(ra) != len(ca):
        return None
    out = []
    for a, b in zip(ra, ca):
        pa, pb = (a or {}).get("positions") or [], (b or {}).get("positions") or []
        if "MEASURED" != (a or {}).get("status", "MEASURED") or "MEASURED" != (b or {}).get("status", "MEASURED") \
                or not pa or len(pa) != len(pb):
            return None
        out += [(k, x, y) for k, (x, y) in enumerate(zip(pa, pb))]
    return out


def _dmargin(a, b):
    if a and b and len(a) >= 5 and len(b) >= 5 and None not in (a[2], a[4], b[2], b[4]):
        return abs((a[2] - a[4]) - (b[2] - b[4]))
    return None


def tpair(tf_ref, tf_cand):
    """T_pair = max(0.5, q99.9 of |d(top1 - top2)| over the pair's teacher-forced positions)."""
    q = _q([x for x in (_dmargin(a, b) for _, a, b in (_tf_pairs(tf_ref, tf_cand) or ())) if x is not None], 0.999)
    return round(max(FAIL_MARGIN, q), 5) if q is not None else FAIL_MARGIN


def compare_tf(ref, cand, bins=TF_BINS, offset=0):
    """Teacher-forced pair on the same ids: mean / max |dlp|, rank-1 agreement, dNLL (lp_ref - lp_cand), |d(top1 -
    top2)| quantiles (feed T_pair), overall and per bin of offset + index."""
    use = [(k + offset, a, b) for k, a, b in _tf_pairs(ref, cand) or () if a and b and a[0] is not None and b[0] is not None]
    if not use:
        return {"status": "INCONCLUSIVE_TF_MISSING"}

    def summ(it):
        d = [abs(a[0] - b[0]) for _, a, b in it]
        r1 = [a[1] == b[1] for _, a, b in it if len(a) > 1 and len(b) > 1 and None not in (a[1], b[1])]
        dm = [x for x in (_dmargin(a, b) for _, a, b in it) if x is not None]
        return {"n": len(d), "mean_abs_dlogprob": _mean(d, 5), "max_abs_dlogprob": round(max(d), 4) if d else None,
                "rank1_agreement": round(sum(r1) / len(r1), 5) if r1 else None,
                "dnll": _mean([a[0] - b[0] for _, a, b in it], 5), "dmargin_p50": _q(dm, .5), "dmargin_p99": _q(dm, .99),
                "dmargin_q999": _q(dm, .999)}
    return dict(summ(use), status="MEASURED", offset=offset,
                bins=[dict(summ([x for x in use if lo <= x[0] < hi]), lo=lo, hi=hi) for lo, hi in bins])


def decode_vs_prefill(gen, tf, base=None, bins=TF_BINS):
    """M4 drift: decode-time lps (generate_logged) vs the same server's prefill re-score (teacher_forced); one or
    aligned lists. Per bin: mean |d|, SE, counts >= 1/2/4 nat, NLL. STATE_DRIFT: >= 3 positions >= 4 nat, or (base =
    this function on the matched baseline arm) a bin mean > base + 3 SE (combined) with excess > 0.02 nat where the
    base bin is clean."""
    gl, tl = (gen if isinstance(gen, list) else [gen]), (tf if isinstance(tf, list) else [tf])
    if not gl or len(gl) != len(tl):
        return {"status": "INCONCLUSIVE_TRACES_MISSING", "bins": []}
    items, bad = [], 0
    for g, t in zip(gl, tl):
        lps, pos = (g or {}).get("lps") or [], (t or {}).get("positions") or []
        if not lps or (g or {}).get("error") or (t or {}).get("status") != "MEASURED" or len(lps) != len(pos):
            bad += 1
            continue
        items += [(k, abs(ld - p[0]), -ld, -p[0]) for k, (ld, p) in enumerate(zip(lps, pos))
                  if ld is not None and p and p[0] is not None]
    out = {"traces": len(gl), "traces_bad": bad, "positions": len(items), "bins": []}
    for lo, hi in bins:
        d = [x for x in items if lo <= x[0] < hi]
        v = [x[1] for x in d]
        out["bins"].append({"lo": lo, "hi": hi, "n": len(v), "mean": _mean(v, 6),
                            "se": round(statistics.stdev(v) / math.sqrt(len(v)), 6) if len(v) > 1 else None,
                            "ge1": sum(x >= 1 for x in v), "ge2": sum(x >= 2 for x in v), "ge4": sum(x >= 4 for x in v),
                            "nll_decode": _mean([x[2] for x in d], 5), "nll_prefill": _mean([x[3] for x in d], 5)})
    out["gross_ge4"] = gross = sum(1 for x in items if x[1] >= 4)
    reasons = [f"gross: {gross} positions with |d| >= 4 nat"] if gross >= 3 else []
    bb = {(b["lo"], b["hi"]): b for b in (base or {}).get("bins") or []}
    for b in out["bins"]:
        z = bb.get((b["lo"], b["hi"]))
        if not z or not b["n"] or b["se"] is None or z.get("se") is None or z.get("ge4"):
            continue
        se = math.sqrt(b["se"] ** 2 + z["se"] ** 2)
        b["base_mean"], b["excess"] = z["mean"], round(b["mean"] - z["mean"], 6)
        if b["mean"] > z["mean"] + 3 * se and b["excess"] > 0.02:
            reasons.append(f"excess in bin {b['lo']}-{b['hi']}: {b['mean']:.4f} vs base {z['mean']:.4f} (3SE {3 * se:.4f})")
    return dict(out, reasons=reasons, status="STATE_DRIFT" if reasons else
                "INCONCLUSIVE_TRACES_MISSING" if bad or not items else "MEASURED")


def chat_probes(engine, port, model=None, cases=(("Reply with exactly one number: 7 * 6 =", "42"),
                                                ("Reply with exactly one number: 2 + 2 =", "4"),
                                                ("Reply with exactly one number: 9 - 3 =", "6")), timeout=300):
    """Arithmetic via /v1/chat/completions, thinking off (answer in content; thinking in reasoning_content / vLLM
    reasoning)."""
    out, model = [], _model(port, model)
    for q, want in cases:
        try:
            msg = _http(port, "/v1/chat/completions", {"model": model, "messages": [{"role": "user", "content": q}],
                                                       "temperature": 0, "max_tokens": 32,
                                                       "chat_template_kwargs": {"enable_thinking": False}}, timeout)
            msg = msg["choices"][0]["message"]
            got = (msg.get("content") or "").strip()
            out.append({"q": q, "want": want, "content": got[:80], "ok": re.search(r"(?<!\d)" + want + r"(?!\d)", got) is not None,
                        "reasoning_chars": len(msg.get("reasoning_content" if engine == "sglang" else "reasoning") or "")})
        except Exception as e:
            out.append({"q": q, "want": want, "error": repr(e)[:200], "ok": False})
    return {"ok": all(x["ok"] for x in out), "cases": out}


def tool_probe(engine, port, model=None, timeout=300):
    """M6 tool call via /v1/chat/completions (thinking off): ok = get_weather with parseable arguments."""
    body = {"model": _model(port, model), "temperature": 0, "max_tokens": 256, "tool_choice": "auto",
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Current weather of a city",
                       "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]}
    try:
        msg = _http(port, "/v1/chat/completions", body, timeout)["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        fn = (calls[0].get("function") or {}) if calls else {}
        args = json.loads(fn.get("arguments") or "null") if calls else None
        return {"ok": fn.get("name") == "get_weather" and isinstance(args, dict) and isinstance(args.get("city"), str),
                "name": fn.get("name"), "arguments": args, "content": (msg.get("content") or "")[:120], "n_calls": len(calls)}
    except Exception as e:
        return {"ok": False, "error": repr(e)[:200]}


# ------------------------------------------------------------------ load: settle, steady window, census, interference
def _settle(M, port, ss, level, admit, sample, server=None, depth=0, need=3):
    """e23 rev 3 settle + SGLang: prefill counters flat, no new Prefill line, >= 2 Decode lines (gauge refreshes) since
    the last prefill (no log: 2 extra calm samples), running streams gaining; KV-bound plateau held 30 s; then median
    depth >= `depth`."""
    t_adm, need = time.perf_counter(), need + (2 if (M.sg and server is None) else 0)
    end, sample = t_adm + admit, min(sample, max(0.2, admit / 20.0))
    best = ok = quiet = 0
    prev = flat_since = depth_end = None
    samples, last = [], [0] * len(ss)
    off = log_offset(server) if server is not None else None
    info = lambda: {"sec": round(time.perf_counter() - t_adm, 1), "best_running": best, "tail": samples[-6:]}
    while time.perf_counter() < max(end, depth_end or 0):
        time.sleep(sample)
        if all(not g.is_alive() for g in ss):
            return {"status": "STREAMS_ENDED_IN_ADMISSION", "info": dict(info(), errors=sorted({g.err for g in ss if g.err})[:3])}
        try:
            m = prom(port)
        except Exception:
            continue
        run, wait, pf, pre = M.running(m), M.waiting(m), M.prefill_all(m), M.preempt(m)
        now = [len(g.tokens) for g in ss]
        first, gaining = sum(g.t_first is not None for g in ss), sum(1 for a, b in zip(last, now) if b > a)
        last, pl, dl = now, 0, 0
        if off is not None:
            txt = server.log()
            new, off = txt[off:], len(txt)
            pl, dl = len(prefill_lines(new)), len(decode_lines(new))
        quiet = 0 if (pl or prev is None or pf != prev[0]) else quiet + dl
        best = max(best, run or 0)
        samples.append([round(time.perf_counter() - t_adm, 1), run, wait, first, gaining, pl, dl])
        calm = (run is not None and run > 0 and prev is not None and pf == prev[0] and pre == prev[1] and pre is not None
                and pl == 0 and first >= run and gaining >= run and (off is None or not M.sg or quiet >= 2))
        if calm and run >= level and not wait:
            ok += 1
        elif calm and run < level and wait:          # KV-bound: the rest wait; accept a plateau held 30 s
            flat_since = flat_since or time.perf_counter()
            ok = ok + 1 if time.perf_counter() - flat_since > 30 * TS else 0
        else:
            ok, flat_since = 0, None
        prev = (pf, pre)
        if ok >= need:
            started = [n for n, g in zip(now, ss) if g.t_first is not None]
            dep = statistics.median(started) if started else 0
            if dep >= depth:
                return {"status": "SETTLED", "depth": dep, "info": dict(info(), depth=dep)}
            depth_end = depth_end or (time.perf_counter() + TS * max(180.0, depth / 5.0))
    return {"status": "DEPTH_NOT_REACHED" if ok >= need else "NO_SETTLE", "info": info()}


def _client_steps(ss, t0, t1):
    """Client inter-chunk intervals (ms) inside [t0, t1] (one chunk per verify step per stream)."""
    iv = []
    for g in ss:
        ts = [t for t, n in list(g.chunk_times) if t0 <= t <= t1]
        iv += [1000.0 * (b - a) for a, b in zip(ts, ts[1:])]
    return _stats(iv)


def _line_stats(dls):
    """Decode-line stats (EMS step_ms = 1000 x running x accept / gen_tps, after-prefill lines excluded)."""
    use = [d for d in dls if not d["after_prefill"] and d["gen_tps"]]
    return {"lines": len(dls), "running": _stats([d["running"] for d in dls], 1),
            "accept_len_mean": _mean([d["accept_len"] for d in dls]), "mamba_usage_mean": _mean([d["mamba_usage"] for d in dls]),
            "gen_tps_p50": statistics.median([d["gen_tps"] for d in use]) if use else None,
            "line_step_ms": _stats([1000.0 * d["running"] * (d["accept_len"] or 1.0) / d["gen_tps"] for d in use]),
            "running_values": sorted({d["running"] for d in dls})}


def _window(M, port, secs, sample, server, every=None):
    """Measurement window: metric samples, m0 / m1, TP0 log slice (decode / prefill / retract lines), decode and
    retraction deltas."""
    off, m0, t0 = log_offset(server) if server is not None else None, prom(port), time.perf_counter()
    w = {"m0": m0, "t0": t0, "mids": [], "run": [], "wait": [], "kv": [], "cg": [], "errs": 0}
    while time.perf_counter() - t0 < secs:
        time.sleep(min(sample, max(0.05, secs - (time.perf_counter() - t0))))
        try:
            m = prom(port)
        except Exception:
            w["errs"] += 1
            continue
        w["mids"].append(m)
        for k, v in (("run", M.running(m)), ("wait", M.waiting(m)), ("kv", M.kv_usage(m))):
            w[k] += [v] if v is not None else []
        w["cg"].append(M.census(m))
        if every and every(w):
            break
    try:
        w["m1"], w["t1"] = prom(port), time.perf_counter()
    except Exception as e:                    # a server that died inside the window: the last good snapshot, flagged
        w["m1"], w["t1"], w["m1_error"] = (w["mids"] or [m0])[-1], time.perf_counter(), repr(e)[:200]
    txt = server.log() if server is not None else None
    w["log"], w["win"] = txt, (txt[off:] if txt is not None else "")
    w["dls"], w["pls"], w["rls"] = decode_lines(w["win"]), prefill_lines(w["win"]), retract_lines(w["win"])
    w["gen"] = _delta(M.gen(m0), M.gen(w["m1"]))
    w["pre"] = _delta(M.preempt(m0, txt[:off] if txt is not None else None), M.preempt(w["m1"], txt), absent0=False)
    return w


def _wrap_up(r, engine, port, streams, raw_dir, name, server=None):
    """End of a load: error snapshot, stop + abort, leak / idle checks, raw file."""
    r["stream_errors"] = sorted({g.err for g in streams if g.err})[:3]
    r["client_leak"] = stop_all(engine, port, streams) > 0
    r["idle_after"] = wait_idle(engine, port, 90.0, server)
    _raw(raw_dir, name, r)


def steady(engine, port, prompts, level, secs=60.0, admit=None, max_new=8192, temperature=1.0, seed=1234, model=None,
           sample=3.0, prefill_tok_s=1300.0, server=None, barrier=None, gpu_sample=True, min_depth=1000, raw_dir=None,
           tag=None):
    """`level` streams; the window opens after settle AND median tokens per stream >= min_depth (aligned depth). tok/s =
    decode-counter delta, steps = decode-pass delta, step_ms, tokens_per_step, running (gauges + TP0 lines), accept,
    client_step_ms, retractions. e23 rev 3 statuses; a retraction -> INCONCLUSIVE_RETRACTION (numbers kept). prompt +
    max_new must fit the context length."""
    M, plen = Metrics(engine), len(prompts[0]) if prompts else 0
    admit = admit or TS * max(420.0, 120.0 + level * plen / prefill_tok_s)
    r = {"engine": engine, "level": level, "status": "INCONCLUSIVE_NOT_STARTED", "port": port, "prompt_tokens": plen,
         "max_new": max_new, "admit_limit_sec": round(admit, 1), "min_depth": min_depth, "log_checks": server is not None}

    def fail(st, **kw):
        r.update(status=st, **kw)
        if barrier:
            barrier.abort()
        return r
    if not wait_idle(engine, port, 60, server):
        return fail("INCONCLUSIVE_NOT_IDLE_AT_START")
    try:
        p0 = prom(port)
    except Exception as e:
        return fail("INCONCLUSIVE_METRICS_UNAVAILABLE", error=repr(e)[:200])
    if not M.has_core(p0):
        return fail("INCONCLUSIVE_COUNTERS_MISSING")
    ss = _spawn(engine, port, prompts, level, max_new, temperature, seed, _model(port, model))
    try:
        st = _settle(M, port, ss, level, admit, sample, server, depth=min_depth)
        r["admission"] = dict(st["info"], prefill_tok_s=round(((M.prefill(prom(port)) or 0) - (M.prefill(p0) or 0)) /
                                                               max(1e-3, st["info"]["sec"]), 1))
        if st["status"] != "SETTLED":
            return fail("INCONCLUSIVE_" + st["status"])
        if barrier:
            try:
                barrier.wait(timeout=max(60.0, admit))
            except threading.BrokenBarrierError:
                r["status"] = "INCONCLUSIVE_REPLICA_BARRIER"
                return r
        tok0, stalls, gmem, last = [len(g.tokens) for g in ss], [0] * len(ss), [], [None]

        def watch(w):                      # stalled streams (no tokens over 3 samples) end the window; GPU memory samples
            now = [len(g.tokens) for g in ss]
            stalls[:] = [(s + 1 if b <= a else 0) if g.t_first is not None and g.is_alive() else s
                         for s, a, b, g in zip(stalls, last[0] or tok0, now, ss)]
            last[0] = now
            if gpu_sample and len(w["mids"]) % 5 == 1:
                gmem.extend(x for x in [gpu_mem()] if x)
            return max(stalls) >= 3
        r["depth_at_start"] = st["depth"]
        w = _window(M, port, secs, min(sample, max(0.1, secs / 10.0)), server, watch)
        dt, runs, gen, pre, steps = w["t1"] - w["t0"], w["run"], w["gen"], w["pre"], _delta(M.steps(w["m0"]), M.steps(w["m1"]))
        gen = 0.0 if gen is None and M.has_core(w["m1"]) else gen      # the decode series never appeared: no decode at all
        rm, gs = (statistics.mean(runs) if runs else None), steps and gen is not None
        r.update(tok_s=round(gen / dt, 1) if gen is not None else None, gen_tokens=gen, window_sec=round(dt, 1), steps=steps,
                 step_ms=round(dt * 1000 / steps, 2) if steps else None, tokens_per_step=round(gen / steps, 3) if gs else None,
                 accept_per_req=round(gen / steps / rm, 3) if gs and rm else None,
                 client_tok_s=round((sum(len(g.tokens) for g in ss) - sum(tok0)) / dt, 1),
                 client_step_ms=_client_steps(ss, w["t0"], w["t1"]), running_mean=round(rm, 2) if rm is not None else None,
                 running_min=min(runs) if runs else None, running_max=max(runs) if runs else None,
                 waiting_max=max(w["wait"]) if w["wait"] else None, kv_usage_mean=_mean(w["kv"]),
                 kv_usage_max=max(w["kv"]) if w["kv"] else None, retractions=pre, metric_errors=w["errs"],
                 retract_log_lines=len(w["rls"]) if w["log"] is not None else None, decode_lines=_line_stats(w["dls"]),
                 prefill_lines_in_window=len(w["pls"]) if w["log"] is not None else None, t0=w["t0"], t1=w["t1"],
                 gpu_mem_peak_mib=[max(row[i][1] for row in gmem if len(row) > i) for i in range(max(map(len, gmem)))]
                 if gmem else None)
        sp0, sp1 = M.spec(w["m0"]), M.spec(w["m1"])
        acc, drf, dtk = (None, None, None) if M.sg else (_delta(sp0[k], sp1[k]) for k in ("acc", "drf", "dtok"))
        r["spec"] = sp1 if M.sg else {"mean_acceptance_length": round(1 + acc / drf, 3) if acc is not None and drf else None,
                                      "accept_rate": round(acc / dtk, 3) if acc is not None and dtk else None}
        ra = runs + [d["running"] for d in w["dls"]]
        ended = sum(g.is_alive() for g in ss) < min(level, st["info"]["best_running"] or level)
        pf_in = bool(w["pls"]) or any(M.prefill_all(x) != M.prefill_all(w["m0"]) for x in w["mids"] + [w["m1"]])
        r["status"] = (
            "INCONCLUSIVE_STALLED_STREAM" if max(stalls) >= 3 else
            "INCONCLUSIVE_STREAMS_ENDED_IN_WINDOW" if ended else "INCONCLUSIVE_PREFILL_IN_WINDOW" if pf_in else
            "INCONCLUSIVE_METRICS_UNAVAILABLE" if w.get("m1_error") else
            ("INCONCLUSIVE_RETRACTION" if M.sg else "INCONCLUSIVE_PREEMPTION") if (pre or 0) > 0 or w["rls"] else
            "INCONCLUSIVE_RETRACTION_UNKNOWN" if pre is None else
            "INCONCLUSIVE_COUNTERS_MISSING" if gen is None or steps is None else
            "FAIL_NO_DECODE" if gen <= 0 else "INCONCLUSIVE_METRICS_MISSING" if not runs else
            "INCONCLUSIVE_RUNNING_SET_CHANGED" if min(ra) < level and (len(set(ra)) > 1 or len(set(w["wait"])) > 1) else
            "INCONCLUSIVE_RUNNING_UNSTABLE" if min(ra) < 0.8 * max(ra) else
            "STEADY" if min(ra) == max(ra) == level else "CAPACITY_LIMITED" if max(ra) < level else
            "INCONCLUSIVE_RUNNING_ABOVE_LEVEL")          # RUNNING_SET_CHANGED: a KV-bound window must hold one fixed set
        return r
    finally:
        gb = [degenerate("".join(g.text)[-4000:]) if g.tokens else None for g in ss]
        r["garbage"] = {"streams": len(ss), "with_output": sum(1 for g in ss if g.tokens), "garbage": [x for x in gb if x],
                        "sample_tail": "".join(ss[0].text)[-200:] if ss and ss[0].tokens else ""}
        if r["garbage"]["garbage"] and r.get("status") in ("STEADY", "CAPACITY_LIMITED"):
            r["status"] = "FAIL_GARBAGE_UNDER_LOAD"
        _wrap_up(r, engine, port, ss, raw_dir, f"steady_{_tag(tag)}.json", server)


def equal_n(pool, band, growth, reserve=2900, margin_frac=0.05):
    """floor(pool x (1 - margin_frac) / (band + growth + reserve)); growth = tokens generated in settle + window + 60 s;
    None without a pool."""
    if not pool or band is None or growth is None:
        return None
    per = float(band) + float(growth) + float(reserve)
    return max(0, int(float(pool) * (1.0 - float(margin_frac)) // per)) if per > 0 else None


def steady_multi(jobs, secs=60.0, **kw):
    """jobs = [(engine, port, prompts, level)] in one shared window (barrier); aggregate only if all are STEADY /
    CAPACITY_LIMITED and overlap >= 80 %."""
    out, bar = [None] * len(jobs), threading.Barrier(len(jobs))

    def one(i, j):
        try:
            out[i] = steady(j[0], j[1], j[2], j[3], secs=secs, barrier=bar, **kw)
        except Exception as e:
            out[i] = {"status": "ERROR_REPLICA", "error": f"{type(e).__name__}: {e}"[:300], "port": j[1]}
            bar.abort()
    th = [threading.Thread(target=one, args=(i, j), daemon=True) for i, j in enumerate(jobs)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    good = all(o and o.get("status") in ("STEADY", "CAPACITY_LIMITED") for o in out)
    ov = (min(o["t1"] for o in out) - max(o["t0"] for o in out)) if good else 0
    aligned = good and ov > 0.8 * min(o["window_sec"] for o in out)
    return {"replicas": out, "aligned": aligned, "overlap_sec": round(ov, 1),
            "tok_s": round(sum(o["tok_s"] for o in out), 1) if aligned else None,
            "running_mean": round(sum(o["running_mean"] for o in out), 2) if aligned else None,
            "status": "AGGREGATE_STEADY" if aligned else "INCONCLUSIVE_REPLICA_WINDOWS"}


def block_reason(cg, queue, running, next_len, mamba_need=1, max_running=None, reserve=0.7 * 4096):
    """ALL_ADMITTED (queue 0) | KV_BOUND (kv available + evictable < next + 0.7 x 4096) | KV_BOUND_RESERVED (< next +
    0.7 x 4096 x (running + 1): running reservations, schedule_policy.py:788-795) | MAMBA_BOUND (available +
    evictable slots < mamba_need, :713-717) | MAX_RUNNING; None = no recorded reason."""
    if queue is None or queue == 0:
        return None if queue is None else "ALL_ADMITTED"
    why, ka, ma = [], cg.get("kv_available_tokens"), cg.get("mamba_available_tokens")
    free = None if ka is None else ka + (cg.get("kv_evictable_tokens") or 0)
    if free is not None and next_len and free < next_len + reserve * ((running or 0) + 1):
        why.append("KV_BOUND" if free < next_len + reserve else "KV_BOUND_RESERVED")
    if ma is not None and ma + (cg.get("mamba_evictable_tokens") or 0) < mamba_need:
        why.append("MAMBA_BOUND")
    if max_running and running is not None and running >= max_running:
        why.append("MAX_RUNNING")
    return "+".join(why) or None


def max_running_of(port, server=None):
    """The resolved running cap and its source: /server_info internal_states[0].effective_max_running_requests_per_dp
    (after the Mamba cap), its max_running_requests, else the log ("capped to N by the mamba state cache", then the
    capacity line); (None, None) when unknown."""
    try:
        si = _http(port, "/server_info", timeout=30)
        st = (si.get("internal_states") or [{}])[0] or {}
        for src, v in (("server_info.effective_max_running_requests_per_dp", st.get("effective_max_running_requests_per_dp")),
                       ("server_info.max_running_requests", si.get("max_running_requests"))):
            if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0:
                return int(v), src
    except Exception:
        pass
    for src, pat in (("log.capped", r"max_running_requests is capped to (\d+) by the mamba"),
                     ("log.capacity", r"max_total_num_tokens=\d+, chunked_prefill_size=-?\d+, max_prefill_tokens=\d+, "
                                      r"max_running_requests=(\d+)")):
        m = re.search(pat, server.log()) if server is not None else None
        if m:
            return int(m.group(1)), src
    return None, None


def _fatal(txt):
    e = log_errors(txt)
    return ("FAIL_MAMBA_ALLOC" if e["mamba_alloc_fail"] else "FAIL_OOM" if e["oom_lines"] else
            "FAIL_SERVER_ERROR" if any(re.search(r"Traceback|Scheduler hit an exception|SIGQUIT", l) for l in e["error_lines"])
            else None)


def census(engine, port, prompts, offered, secs=60.0, plateau_sec=30.0, max_wait=None, server=None, model=None, band=None,
           max_new=32768, sample=3.0, mamba_need=1, max_running=None, min_hold=60.0, temperature=1.0, seed=777, raw_dir=None,
           tag=None):
    """Offer `offered` streams (never client-capped); plateau = prefill counters flat + no new Prefill line + queue
    unchanged + a block_reason, held max(plateau_sec, TS x max(min_hold, 1.5 x band / 1300)) within max_wait = TS x
    (offered x band / 1300 + 120); then a `secs` window. max_running None -> max_running_of() (source recorded).
    ADMITTED_ALL | CAPACITY_LIMITED | INCONCLUSIVE_* | FAIL_MAMBA_ALLOC / FAIL_OOM / FAIL_SERVER_ERROR / FAIL_SERVER_DIED."""
    M = Metrics(engine)
    band = int(band or (statistics.median([len(p) for p in prompts]) if prompts else 0))
    hold = max(float(plateau_sec), TS * max(float(min_hold), 1.5 * band / 1300.0))
    max_wait = max_wait or TS * (offered * band / 1300.0 + 120.0)
    smp = min(sample, max(0.2, hold / 10.0))
    r = {"engine": engine, "offered": offered, "band": band, "plateau_hold_sec": round(hold, 1),
         "max_wait_sec": round(max_wait, 1), "status": "INCONCLUSIVE_NOT_STARTED", "log_checks": server is not None}
    if not wait_idle(engine, port, 60, server):
        return dict(r, status="INCONCLUSIVE_NOT_IDLE_AT_START")
    off0 = off = log_offset(server) if server is not None else None
    r["baseline"] = snapshot(engine, port)
    r["max_running"], r["max_running_source"] = (max_running, "argument") if max_running else \
        (max_running_of(port, server) if engine == "sglang" else (None, None))
    max_running = r["max_running"]
    ss = _spawn(engine, port, prompts, offered, max_new, temperature, seed, _model(port, model))
    try:
        t_s, prev, since, reason, samples, plateau = time.perf_counter(), None, None, None, [], False
        while time.perf_counter() - t_s < max_wait and not plateau:
            time.sleep(smp)
            pl = 0
            if server is not None:
                txt = server.log()
                r["status"] = "FAIL_SERVER_DIED" if not server.alive() else _fatal(txt[off0:]) or r["status"]
                if r["status"].startswith("FAIL"):
                    r["errors"] = log_errors(txt[off0:])
                    return r
                pl, off = len(prefill_lines(txt[off:])), len(txt)
            if all(not g.is_alive() for g in ss):
                r["status"] = "INCONCLUSIVE_STREAMS_ENDED"
                return r
            try:
                m = prom(port)
            except Exception:
                continue
            pf, q, run = M.prefill_all(m), M.waiting(m), M.running(m)
            reason = block_reason(M.census(m), q, run, band, mamba_need, max_running)
            flat = prev is not None and pf == prev[0] and q == prev[1] and pl == 0 and q is not None
            since = (since or time.perf_counter()) if (flat and reason) else None
            samples.append([round(time.perf_counter() - t_s, 1), run, q, pl, reason])
            prev, plateau = (pf, q), bool(since and time.perf_counter() - since >= hold)
        r.update(admission={"sec": round(time.perf_counter() - t_s, 1), "tail": samples[-8:]}, blocking_reason=reason,
                 admitted_client=sum(1 for g in ss if g.t_first is not None))
        if not plateau:
            r.update(status="INCONCLUSIVE_NO_PLATEAU", last_gauges=snapshot(engine, port))
            return r
        w = _window(M, port, secs, smp, server)
        runs, txt = w["run"], w["log"]
        r.update(window_sec=round(w["t1"] - w["t0"], 1),
                 tok_s=round(w["gen"] / (w["t1"] - w["t0"]), 1) if w["gen"] is not None else None,
                 running_gauge=_stats(runs, 1), running_lines=_stats([d["running"] for d in w["dls"]], 1),
                 queue=_stats(w["wait"], 1), decode_lines=_line_stats(w["dls"]), retractions=w["pre"],
                 retract_log_lines=len(w["rls"]) if txt is not None else None,
                 prefill_lines_in_window=len(w["pls"]) if txt is not None else None,
                 gauges_mean={k: _mean([c.get(k) for c in w["cg"]], 4) for k in _CENSUS[:-2]}, gauges_end=M.census(w["m1"]),
                 errors=log_errors(txt[off0:]) if txt is not None else None)
        fat = _fatal(txt[off0:]) if txt is not None else None
        if not fat and w.get("m1_error"):
            fat = "FAIL_SERVER_DIED" if server is not None and not server.alive() else "INCONCLUSIVE_METRICS_UNAVAILABLE"
        r["status"] = fat or ("ADMITTED_ALL" if reason == "ALL_ADMITTED" and runs and max(runs) >= offered else "CAPACITY_LIMITED")
        return r
    finally:
        _wrap_up(r, engine, port, ss, raw_dir, f"census_{_tag(tag)}.json", server)


def _warm_once(engine, port, ids, model, timeout=1800):
    """1-token request (D1 warm-up: a Mamba checkpoint within 64 tokens of the prompt end)."""
    t0 = time.perf_counter()
    try:
        if engine == "sglang":
            c = _sg_out(_sg_post(port, {"input_ids": list(ids), "sampling_params": _sg_sampling(1, 0.0, True)}, timeout))
            c = c["cached_tokens"]
        else:
            u = _http(port, "/v1/completions", {"model": model, "prompt": list(ids), "max_tokens": 1, "temperature": 0}, timeout)
            c = ((u.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        return {"ok": True, "sec": round(time.perf_counter() - t0, 1), "cached_tokens": c}
    except Exception as e:
        return {"ok": False, "error": repr(e)[:200], "sec": round(time.perf_counter() - t0, 1)}


def interference(engine, port, residents, inject, n_resident, warm_prompt=None, secs_after=30.0, server=None, model=None,
                 window_sec=None, d0_median_step=None, max_new=8192, inject_max_new=None, sample=1.0, admit=None,
                 prefill_tok_s=1300.0, gap_sec=5.0, stall_frac=0.2, temperature=1.0, seed=4242, raw_dir=None, tag=None):
    """Section 5 load: residents settle, `inject` arrives at t_inj; ONE fixed window [t_inj, t_inj + window_sec] (None:
    the last injected first token + secs_after). starvation = worst resident max inter-chunk gap > gap_sec or stall_s
    (gaps > 5 x the D0 median step, d0_median_step ms) > stall_frac x window; also prefill-phase resident tokens,
    decode-flat sec, TPOT, window tok/s (Y11 mixed), prefill_sec per injected. D1 (warm_prompt): cached >= 0.99 x
    prompt (usage, else the Prefill line #cached-token) else INVALID_WARM_MISS."""
    M = Metrics(engine)
    inject = list(inject or ([warm_prompt] if warm_prompt is not None else []))
    admit = admit or TS * max(420.0, 120.0 + n_resident * (len(residents[0]) if residents else 0) / prefill_tok_s)
    r = {"engine": engine, "n_resident": n_resident, "n_inject": len(inject), "status": "INCONCLUSIVE_NOT_STARTED",
         "window_sec_preset": window_sec, "log_checks": server is not None}
    if not wait_idle(engine, port, 60, server):
        return dict(r, status="INCONCLUSIVE_NOT_IDLE_AT_START")
    model = _model(port, model)
    if warm_prompt is not None:
        r["warmup"] = _warm_once(engine, port, warm_prompt, model)
        wait_idle(engine, port, 60, server)
    rs, inj = _spawn(engine, port, residents, n_resident, max_new, temperature, seed, model), []
    try:
        t_set = time.perf_counter()
        st = _settle(M, port, rs, n_resident, admit, min(sample * 3, 3.0), server)
        r["admission"] = st["info"]
        if st["status"] != "SETTLED":
            r["status"] = "INCONCLUSIVE_" + st["status"]
            return r
        t_inj = time.perf_counter()
        r["pre_injection_step_ms"] = _client_steps(rs, t_inj - max(10.0 * TS, (t_inj - t_set) / 3), t_inj)
        d0 = d0_median_step if d0_median_step is not None else r["pre_injection_step_ms"]["p50"]
        inj = _spawn(engine, port, inject, len(inject), inject_max_new or max_new, temperature, seed + 1000, model)
        flat, last = [0.0], [None, time.perf_counter()]

        def until(w):                      # decode-counter flat seconds; the window end (fixed, or first tokens + secs_after)
            g_, t = M.gen(w["mids"][-1]), time.perf_counter()
            flat[0] += (t - last[1]) if g_ is not None and g_ == (last[0] if last[0] is not None else M.gen(w["m0"])) else 0.0
            last[:] = [g_, t]
            if server is not None and not server.alive():
                r["status"] = "FAIL_SERVER_DIED"
                return True
            f_ = [g.t_first for g in inj]
            return window_sec is None and all(f_) and t >= max(f_ + [t_inj]) + secs_after
        w = _window(M, port, window_sec if window_sec is not None else admit + secs_after, sample, server, until)
        if r["status"] == "FAIL_SERVER_DIED":
            return r
        t_end, W, firsts = w["t1"], w["t1"] - t_inj, [g.t_first for g in inj]
        t_pf = max(firsts) if inj and all(firsts) else t_end
        thr, per, tp = (5.0 * d0 / 1000.0 if d0 else None), [], []
        for g in rs:
            ct = [(t, n) for t, n in list(g.chunk_times) if t_inj <= t <= t_end]
            pts = [t_inj] + [t for t, _ in ct] + [t_end]
            gaps = [b - a for a, b in zip(pts, pts[1:])]
            tp += [1000.0 * (b[0] - a[0]) / b[1] for a, b in zip(ct, ct[1:]) if b[1]]
            per.append({"max_gap_s": round(max(gaps), 3), "stall_s": round(sum(x for x in gaps if x > thr), 3) if thr else None,
                        "tokens_prefill_phase": sum(n for t, n in ct if t <= t_pf), "tokens_window": sum(n for _, n in ct)})
        max_gap = max(p["max_gap_s"] for p in per) if per else None
        stall = max(p["stall_s"] for p in per) if per and thr else None
        r.update(window_sec=round(W, 1), d0_median_step_ms=d0, resident=per, max_gap_s=max_gap, stall_s_max=stall,
                 stall_s_mean=_mean([p["stall_s"] for p in per]) if stall is not None else None,
                 resident_tokens_prefill_phase=sum(p["tokens_prefill_phase"] for p in per),
                 prefill_phase_sec=round(t_pf - t_inj, 2), decode_flat_sec=round(flat[0], 2), tpot_ms=_stats(tp),
                 tok_s_window=round(w["gen"] / W, 1) if w["gen"] is not None and W else None,
                 resident_tok_s=round(sum(p["tokens_window"] for p in per) / W, 1) if W else None,
                 prefill_sec=[round(g.t_first - t_inj, 2) if g.t_first else None for g in inj],
                 waiting_max=max(w["wait"]) if w["wait"] else None, running_min=min(w["run"]) if w["run"] else None,
                 retractions=w["pre"], retract_log_lines=len(w["rls"]) if server is not None else None,
                 prefill_lines=len(w["pls"]) if server is not None else None,
                 starvation=None if max_gap is None else True if max_gap > gap_sec or (stall or 0) > stall_frac * W
                 else None if stall is None else False)
        if warm_prompt is not None:
            c_use = [x for x in (g.cached_tokens() for g in inj) if x is not None]
            c_log = [p["cached_token"] for p in w["pls"] if p["new_seq"]]
            cached = max(c_use) if c_use else (max(c_log) if c_log else None)
            r["warm_check"] = {"cached_tokens": cached, "need": round(0.99 * len(warm_prompt)),
                               "ok": cached is not None and cached >= 0.99 * len(warm_prompt),
                               "source": "usage" if c_use else ("prefill_line" if c_log else None)}
        r["status"] = ("INVALID_WARM_MISS" if warm_prompt is not None and not r["warm_check"]["ok"] else
                       "INCONCLUSIVE_INJECT_NO_FIRST_TOKEN" if inj and not all(firsts) else
                       "INCONCLUSIVE_STREAMS_ENDED_IN_WINDOW" if sum(g.is_alive() for g in rs) < n_resident else
                       "INCONCLUSIVE_NO_D0_STEP" if r["starvation"] is None else "MEASURED")
        return r
    finally:
        _wrap_up(r, engine, port, rs + inj, raw_dir, f"interference_{_tag(tag)}.json", server)


# ------------------------------------------------------------------ profiler
# e23 rev 3: matched on the kernel BASE name, attention before GDN ("fla" is a substring of "flashinfer")
CATS = [("nccl", ("nccl", "allreduce", "all_reduce", "allgather", "all_gather")),
        ("attention", ("flashinfer", "flash_fwd", "unified_attention", "paged", "attn", "attention", "batchdecode",
                       "batchprefill")),
        ("gdn_linear_attn", ("gdn", "delta_rule", "chunk_", "recurrent", "causal_conv", "l2norm", "solve_tril", "merge_16x16",
                             "recompute_w_u", "kkt", "cumsum", "post_conv", "layer_norm_gated", "gated_delta")),
        ("gemm_marlin", ("marlin",)), ("gemm_other", ("gemm", "cutlass", "xmma", "cublas", "sm89_", "ampere_", "sm80_")),
        ("sampling", ("sampl", "topk", "top_k", "argmax", "softmax", "rejection"))]
_NC_KERN = re.compile(r"_(RING|TREE|COLLNET_DIRECT|COLLNET_CHAIN|NVLS_TREE|NVLS|PAT)_(LL128|LL|SIMPLE)", re.I)


def _cat(name):
    base = str(name).split("<")[0].split("(")[0].lower()
    return next((k for k, pats in CATS if any(p in base for p in pats)), "other")


def _iter_events(path):
    """Trace events: e14_trace_summary's streaming reader when importable, else one json.load."""
    try:
        from e14_trace_summary import iter_trace_events
        return iter_trace_events(str(path))
    except ImportError:
        with (gzip.open if str(path).endswith(".gz") else open)(path, "rt", encoding="utf-8", errors="replace") as f:
            d = json.load(f)
        return iter(d.get("traceEvents", []) if isinstance(d, dict) else d)


def _scan_trace(path):
    """One rank: step span starts per label, kernel us by category / name, NCCL kernels (count, us, grid.x, algo,
    proto)."""
    steps, cats, names, nccl, n_k = {}, Counter(), Counter(), {}, 0
    for e in _iter_events(path):
        if not isinstance(e, dict) or e.get("ph") != "X":
            continue
        cat, name = str(e.get("cat") or "").lower(), str(e.get("name") or "")
        if cat == "user_annotation" and name.startswith("step["):
            steps.setdefault(name[5:].split(" ")[0].rstrip("]"), []).append(float(e.get("ts") or 0))
        elif cat in ("kernel", "gpu_kernel"):
            d, n_k, base = float(e.get("dur") or 0), n_k + 1, name.split("(")[0][:160]
            cats[_cat(name)] += d
            names[base] += d
            if "nccl" in name.lower():
                k = nccl.setdefault(base, {"count": 0, "us": 0.0, "grid_x": set(), "algo": None, "proto": None})
                g, m = (e.get("args") or {}).get("grid"), _NC_KERN.search(name)
                k.update(count=k["count"] + 1, us=k["us"] + d,
                         **({"algo": m.group(1).upper(), "proto": m.group(2).upper()} if m else {}))
                k["grid_x"] |= {int(g[0])} if isinstance(g, list) and g else set()
    return {"steps": steps, "cats_us": dict(cats), "nccl": nccl, "kernels": n_k,
            "top_kernels_us": {k: round(v) for k, v in names.most_common(15)}}


def _pct(cats):
    tot = sum(cats.values())
    return {k: round(100 * v / tot, 1) for k, v in cats.items()} if tot else {}


def trace_summary(pdir):
    """vLLM *.pt.trace.json* (rank-0 file): category %, top kernels, NCCL kernels, step spans (no E24 section profiles
    vLLM)."""
    files = sorted(Path(pdir).rglob("*.pt.trace.json*"), key=lambda p: p.stat().st_mtime)
    if not files:
        return {"error": "NO_TRACE_FILE", "listing": [str(p) for p in Path(pdir).rglob("*")][:20]}
    f = next((p for p in files if re.search(r"rank0\.|rank[-_]?0[._]|tp0_", p.name)), files[0])
    try:
        s = _scan_trace(f)
    except Exception as e:
        return {"error": f"TRACE_PARSE {type(e).__name__}: {e}"[:300], "file": f.name}
    return {"file": f.name, "files": len(files), "kernels": s["kernels"], "kernel_us_sum": round(sum(s["cats_us"].values())),
            "by_category_pct": _pct(s["cats_us"]), "top_kernels_us": s["top_kernels_us"],
            "step_spans": {k: len(v) for k, v in s["steps"].items()},
            "nccl_kernels": {k: dict(v, grid_x=sorted(v["grid_x"])) for k, v in s["nccl"].items()}}


def trace_summary_sglang(pdir):
    """<id>-TP-<r> traces: e14 summarize (when importable), TP0 per-step wall (consecutive step[VERIFY, else step[DECODE
    starts), NCCL kernels / algo / proto / grid / ms per step, rank skew. Never raises."""
    out = {"dir": str(pdir)}
    try:
        files = sorted(Path(pdir).glob("*-TP-*.trace.json*"))
        out["files"] = [p.name for p in files]
        if not files:
            return dict(out, error="NO_TRACE_FILE")
        try:
            import e14_trace_summary as E14T
            out["e14"], out["e14_source"] = E14T.summarize(str(pdir), 0), "e14_trace_summary"
        except ImportError:
            out["e14"], out["e14_source"] = None, "fallback"
        except Exception as e:
            out["e14"], out["e14_source"] = {"error": repr(e)[:300]}, "e14_trace_summary"
        ranks, per = {}, {}
        for p in files:
            rk = int(re.search(r"-TP-(\d+)", p.name).group(1))
            try:
                ranks[rk] = _scan_trace(p)
            except Exception as e:
                ranks[rk] = per[str(rk)] = {"error": repr(e)[:200]}
        for rk, s in sorted(ranks.items()):
            if "error" not in s:
                label = next((x for x in ("VERIFY", "DECODE") if s["steps"].get(x)), None)
                n, nc = len(s["steps"].get(label) or []), s["nccl"].values()
                per[str(rk)] = {"step_label": label, "steps": n, "kernels": s["kernels"], "by_category_pct": _pct(s["cats_us"]),
                                "nccl_ms_per_step": round(sum(v["us"] for v in nc) / 1000 / n, 3) if n else None,
                                "nccl_calls_per_step": round(sum(v["count"] for v in nc) / n, 2) if n else None,
                                "kernel_ms_per_step": round(sum(s["cats_us"].values()) / 1000 / n, 3) if n else None,
                                "top_kernels_us": s["top_kernels_us"]}
        out["ranks"], t0 = per, ranks.get(0) or {}
        if "steps" in t0:
            label = per["0"]["step_label"]
            ts = sorted(t0["steps"].get(label) or []) if label else []
            out["step_wall_ms"] = dict(_stats([(b - a) / 1000.0 for a, b in zip(ts, ts[1:])], 3), label=label)
            ar = [v for k, v in t0["nccl"].items() if re.search(r"allreduce|all_reduce", k, re.I)]
            out["nccl"] = {"kernels": {k: dict(v, grid_x=sorted(v["grid_x"])) for k, v in t0["nccl"].items()},
                           "allreduce_algo": sorted({v["algo"] for v in ar if v["algo"]}),
                           "allreduce_proto": sorted({v["proto"] for v in ar if v["proto"]}),
                           "allreduce_grid_x": sorted({g for v in ar for g in v["grid_x"]}),
                           "ms_per_step": per["0"]["nccl_ms_per_step"], "calls_per_step": per["0"]["nccl_calls_per_step"]}
        v = [x["nccl_ms_per_step"] for x in per.values() if x.get("nccl_ms_per_step") is not None]
        out["nccl_rank_skew_ms_per_step"] = round(max(v) - min(v), 3) if len(v) > 1 else None
    except Exception as e:
        out["error"] = f"TRACE_PARSE {type(e).__name__}: {e}"[:300]
    return out


def _profile_vllm(port, pdir, secs=6.0, delay_iters=20, max_iters=40):
    """e23: bounded /start_profile (bare POST fallback), sleep, /stop_profile, summary."""
    rec, bare = {}, lambda p: U.urlopen(U.Request(f"http://127.0.0.1:{port}{p}", data=b"", method="POST"), timeout=900).read()
    try:
        _http(port, "/start_profile", {"delay_iterations": delay_iters, "max_iterations": max_iters}, timeout=900)
        rec["start_profile"] = "ok_bounded"
    except Exception as e:
        rec["start_profile_bounded_error"] = repr(e)[:200]
        try:
            bare("/start_profile")
            rec["start_profile"] = "ok"
        except Exception as e2:
            rec["start_profile"] = repr(e2)[:200]
    time.sleep(secs)
    try:
        bare("/stop_profile")
        rec["stop_profile"] = "ok"
    except Exception as e:
        rec["stop_profile"] = repr(e)[:200]
    time.sleep(20 * TS)
    return dict(rec, trace=trace_summary(pdir))


def profile(engine, port, out_dir, num_steps=30, timeout=900, tp=4):
    """SGLang: /start_profile {output_dir, num_steps, CPU+GPU, no stack / shapes} (stops itself, traffic must run),
    waits for `tp` stable per-rank files, summary = trace_summary_sglang. vLLM: the e23 bounded profile. Never
    overlap a window."""
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    if engine != "sglang":
        return dict(_profile_vllm(port, str(d), max_iters=num_steps), dir=str(d))
    rec, t0 = {"dir": str(d), "num_steps": num_steps}, time.perf_counter()
    try:
        rec["start_reply"] = str(_http(port, "/start_profile", {"output_dir": str(d), "num_steps": int(num_steps),
                                                                 "activities": ["CPU", "GPU"], "with_stack": False,
                                                                 "record_shapes": False}, timeout=120))[:200]
    except Exception as e:
        return dict(rec, error="START_PROFILE " + repr(e)[:200])
    prev, stable, files, ranks = None, 0, [], set()
    while time.perf_counter() - t0 < timeout and stable < 2:
        time.sleep(min(3.0, max(0.2, 3.0 * TS)))
        files = sorted(p for p in d.glob("*-TP-*.trace.json*") if not p.name.endswith(".tmp"))
        sizes = [(p.name, p.stat().st_size) for p in files]
        ranks = {int(m.group(1)) for p in files for m in [re.search(r"-TP-(\d+)", p.name)] if m}
        stable, prev = (stable + 1 if sizes == prev and len(ranks) >= tp else 0), sizes
    rec.update(files=[p.name for p in files], bytes=sum(p.stat().st_size for p in files), sec=round(time.perf_counter() - t0, 1),
               ranks=sorted(ranks), trace=trace_summary_sglang(str(d)))
    if len(ranks) < tp:
        rec["error"] = f"INCOMPLETE_TRACES {len(ranks)}/{tp} ranks"
    return rec


def profiled_load(engine, port, prompts, level, out_dir, num_steps=30, max_new=8192, model=None, prefill_tok_s=1300.0,
                  server=None, sample=3.0, tp=4):
    """Own load (never a speed number): settle as steady(), then profile()."""
    M = Metrics(engine)
    settle = TS * max(300.0, 120.0 + level * (len(prompts[0]) if prompts else 0) / prefill_tok_s)
    ss = _spawn(engine, port, prompts, level, max_new, 1.0, 4321, _model(port, model))
    rec = {"level": level, "diagnostic_only": "profiler window: never a speed reference"}
    try:
        st = _settle(M, port, ss, level, settle, sample, server)
        rec.update(settled=st["status"] == "SETTLED", admission=st["info"], running_at_start=M.running(prom(port)))
        rec.update(profile(engine, port, out_dir, num_steps=num_steps, tp=tp))
        if not rec["settled"] and isinstance(rec.get("trace"), dict):
            rec["trace"]["note2"] = "prefill had not settled: the trace may mix prefill and decode"
        return rec
    finally:
        stop_all(engine, port, ss)
        rec["idle_after"] = wait_idle(engine, port, 90.0, server)
