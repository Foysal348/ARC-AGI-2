"""E23 lab, rev 3: port-parameterized server lifecycle and measurement for the six-lever notebook (stdlib only).

Base: the Exp 14 VMS harness (VServer / VGen: token ids from return_token_ids, [DONE] + finish_reason required, socket
shutdown on stop, top-1/top-2 margins, garbage checks on every window). Adopted from the consolidated v3 notebook:
same-weight greedy oracle classified by the oracle's logprob margin at the first differing token, a steady decode window
that starts only after prefill settled, a GPU-release gate, fail-closed statuses, checkpoint identity, a runtime
--help flag check, KV + GPU memory samples.
rev 3 (validation agents 2026-10-29, measurement + robustness):
  - port check by connect (a TIME_WAIT socket of the previous server is not "busy"), with a 90 s wait;
  - kill(): SIGKILL the whole session after SIGTERM, even when the leader already exited;
  - steady(): idle gate at entry, admission cap from the prefill to do (700 tok/s assumed), settle needs no preemption
    and every running stream gaining tokens; in the window: stalled streams, a changing running/waiting set (KV-bound),
    preemptions and prefills make it INCONCLUSIVE (numbers kept); garbage under load fails it; steps from the
    vllm:iteration_tokens_total histogram count; replicas measured in one shared window (barrier);
  - profiled_load(): settle as steady(), bounded profile (delay / max iterations), with_stack off, frontend ignored;
  - trace_summary(): categories on the kernel base name (attention before GDN: "fla" is inside "flashinfer"), per-forward
    graph / kernel launch counts inside the gpu_model_runner forward ranges (VLLM_CUSTOM_SCOPES_FOR_PROFILING=1) and the
    CUPTI correlation fallback -> FULL / PIECEWISE / EAGER / UNKNOWN; startup cudagraph lines parsed explicitly;
  - teacher_forced(): prompt logprobs of the last N positions of a long prompt (NVFP4 vs FP8 KV at 40k / 90k);
  - logprobs keyed by token id (return_tokens_as_token_ids); stop() race closed; compare() flags garbage only when the
    oracle itself is clean.
"""
import gzip
import hashlib
import json
import os
import re
import signal
import socket
import statistics
import subprocess
import threading
import time
import urllib.request as U
from pathlib import Path

FAIL_MARGIN = 0.5          # first differing token: oracle top-1 minus top-2 logprob at or above this = real divergence
LIVE = set()               # every running Server, for the hard-stop watchdog


# ------------------------------------------------------------------ server lifecycle
def port_listening(port):
    """True when something accepts connections on 127.0.0.1:port (TIME_WAIT leftovers do not)."""
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
    def __init__(self, name, argv, env, port, log_dir, gpus="0,1,2,3"):
        self.name, self.argv, self.env, self.port = name, [str(a) for a in argv], dict(env), int(port)
        self.gpus = str(gpus)
        self.log_path = Path(log_dir) / f"server_{name}.log"
        self.proc, self.t0, self.ready_sec = None, None, None

    @property
    def url(self):
        return f"http://127.0.0.1:{self.port}"

    def start(self):
        end = time.time() + 90
        while port_listening(self.port):
            if time.time() > end:
                raise RuntimeError(f"PORT_BUSY {self.port} (a listener is still there)")
            time.sleep(3)
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
        """Host RAM of the server's whole session (engine + workers), MB; None off Linux."""
        try:
            r = subprocess.run(["ps", "-o", "rss=", "--sid", str(self.proc.pid)], capture_output=True, text=True, timeout=20)
            return round(sum(int(x) for x in r.stdout.split() if x.isdigit()) / 1024, 1)
        except Exception:
            return None

    def wait_ready(self, timeout):
        while time.perf_counter() - self.t0 < timeout:
            if not self.alive():
                return False, f"exited rc={self.proc.returncode}"
            try:
                U.urlopen(self.url + "/health", timeout=10).read()
                self.ready_sec = round(time.perf_counter() - self.t0, 1)
                return True, f"ready in {self.ready_sec}s"
            except Exception:
                time.sleep(3)
        return False, "startup timeout"

    def kill(self):
        if self.proc is None:
            return
        pg = self.proc.pid
        try:
            os.killpg(pg, signal.SIGTERM) if hasattr(os, "killpg") else self.proc.terminate()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=40)
        except Exception:
            pass
        # rev 3: the leader may exit while EngineCore / workers of the session still hold GPUs: always SIGKILL the group
        try:
            os.killpg(pg, signal.SIGKILL) if hasattr(os, "killpg") else self.proc.kill()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=30)
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        LIVE.discard(self)

    def facts(self):
        """What vLLM logged at startup: KV pool, max concurrency, selected kernels, graph mode, backend, KV dtype."""
        t = self.log()
        kv = [int(x.replace(",", "")) for x in re.findall(r"GPU KV cache size: ([\d,]+) tokens", t)]
        mc = re.findall(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", t)
        kern = sorted(set(re.findall(r"Selected (\w+Kernel) for", t)))
        errs = [l[:300] for l in t.splitlines() if re.search(r"Traceback|CUDA error|illegal memory|out of memory|"
                                                               r"RuntimeError|AssertionError", l)][:12]
        pick = lambda pat, n=8: [l.strip()[:240] for l in t.splitlines() if re.search(pat, l, re.I)][:n]
        # rev 3 (measurement H5): the decisive startup evidence for the graph mode actually used with spec decode
        downgrade = pick(r"not supported with spec-decode.*setting cudagraph_mode|setting cudagraph_mode\s*=", 4)
        capture = sorted(set(re.findall(r"Capturing CUDA graphs \(([^)]+)\)", t)))
        configured = sorted(set(re.findall(r"CUDAGraphMode\.(\w+)", t)))
        return {"kv_tokens": min(kv) if kv else None, "kv_tokens_all": kv, "max_concurrency": float(mc[-1]) if mc else None,
                "kernels": kern, "error_lines": errs, "ready_sec": self.ready_sec,
                "graph_capture_modes": capture, "graph_downgrade_lines": downgrade, "cudagraph_modes_logged": configured,
                "backend_lines": pick(r"attention backend|Using \w+ backend|backend=|AttentionBackend"),
                "kv_dtype_lines": pick(r"kv_cache_dtype|kv cache dtype|nvfp4"),
                "host_embed_lines": pick(r"host[- _]?(resident|embed)|HOST_EMBED|embed.*(cpu|host)", 6)}


def gpu_pids(gpus=None):
    """Compute PIDs on all GPUs, or only on the GPU indices in `gpus` (a bench on another GPU must not fail the release
    gate of a server that used GPU 0 only). None when nvidia-smi is unavailable."""
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
    out = set()
    for l in a.stdout.strip().splitlines():
        if "," not in l:
            continue
        pid, uuid = (x.strip() for x in l.split(",", 1))
        if pid.isdigit() and (want is None or idx.get(uuid) in want):
            out.add(int(pid))
    return sorted(out)


def gpu_mem():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return [[int(v) for v in l.split(",")] for l in r.stdout.strip().splitlines() if l.strip()]
    except Exception:
        return None


def wait_gpu_release(timeout=180, fake=False, gpus=None, free_mib=1024):
    """True when the GPUs in `gpus` (all when None) hold no compute PID AND use < free_mib MiB each (rev 3: compute-app
    PIDs are often invisible inside Kaggle containers; memory.used is the proven E18 gate); None without nvidia-smi."""
    if fake:
        return True
    want = None if gpus is None else {int(x) for x in str(gpus).split(",") if str(x).strip() != ""}
    end = time.time() + timeout
    while time.time() < end:
        p, mem = gpu_pids(gpus), gpu_mem()
        if p is None or mem is None:
            return None
        used = [m[1] for m in mem if want is None or m[0] in want]
        if not p and all(u < free_mib for u in used):
            return True
        time.sleep(3)
    return False


def identity(model_dir):
    """Checkpoint identity: sha256 of config / tokenizer / index files (an oracle is only compared with the same weights)."""
    root = Path(model_dir)
    files = [p for n in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json")
             for p in [root / n] if p.is_file()] + sorted(root.glob("*.safetensors.index.json"))
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return {"dir": str(root), "files": [p.name for p in files], "sha256": h.hexdigest()[:16]}


_HELP = {}


def help_check(key, argv_prefix, env, flags, timeout=300):
    """The engine's own `serve --help=all` (run once per engine `key`, cached) must list every flag of THIS launch
    (rev 3: every launch is checked, not only the first); returns the missing ones."""
    if key not in _HELP:
        try:
            r = subprocess.run(list(argv_prefix) + ["--help=all"], capture_output=True, text=True, timeout=timeout, env=env)
            _HELP[key] = r.stdout + r.stderr
        except Exception as e:
            _HELP[key] = "ERROR " + repr(e)[:200]
    txt = _HELP[key]
    if len(txt) < 2000:
        return {"ran": False, "error": "help output too short or failed", "tail": txt[-400:]}
    want = sorted({f for f in flags if isinstance(f, str) and f.startswith("--")})
    return {"ran": True, "missing": [f for f in want if f not in txt]}


def start_watchdog(hard_stop_epoch, on_fire):
    """A real hard stop: at `hard_stop_epoch` every live Server is killed and on_fire() is called."""
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
    t = threading.Thread(target=run, daemon=True, name="e23-watchdog")
    t.start()
    return t


def prefetch(paths, threads=16, chunk=16 << 20):
    """Read files once with many readers (page cache warm-up of a cold /kaggle/input mount); returns bytes and seconds."""
    files = [p for root in paths for p in sorted(Path(root).rglob("*")) if p.is_file()]
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
    big = sorted(files, key=lambda p: -p.stat().st_size)
    parts = [big[i::threads] for i in range(threads)]
    ths = [threading.Thread(target=worker, args=(x,), daemon=True) for x in parts if x]
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    dt = time.time() - t0
    return {"files": len(files), "bytes": done[0], "sec": round(dt, 1), "MBps": round(done[0] / 1e6 / max(dt, 1e-3), 1)}


# ------------------------------------------------------------------ metrics
def prom(port):
    txt = U.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=15).read().decode("utf-8", "replace")
    out = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)", line)
        if not m:
            continue
        try:
            v = float(m.group(3))
        except ValueError:
            continue
        if v != v:
            continue
        out[m.group(1)] = out.get(m.group(1), 0.0) + v
        out[m.group(1) + "@max"] = max(out.get(m.group(1) + "@max", v), v)
    return out


GEN, PROMPT, RUN, WAIT = ("vllm:generation_tokens_total", "vllm:prompt_tokens_total", "vllm:num_requests_running",
                          "vllm:num_requests_waiting")
PREEMPT, KVU, ITER = "vllm:num_preemptions_total", "vllm:kv_cache_usage_perc", "vllm:iteration_tokens_total_count"
ACC, DRF, DTOK = ("vllm:spec_decode_num_accepted_tokens_total", "vllm:spec_decode_num_drafts_total",
                  "vllm:spec_decode_num_draft_tokens_total")


# ------------------------------------------------------------------ one streaming request (Exp 14 VGen, port-aware)
class Stream(threading.Thread):
    def __init__(self, port, ids, max_new, temperature=1.0, seed=None, logprobs=0, ignore_eos=True, model="q27"):
        super().__init__(daemon=True)
        self.port = int(port)
        self.body = {"model": model, "prompt": list(ids), "max_tokens": int(max_new), "temperature": temperature,
                     "top_p": 0.95 if temperature else 1.0, "stream": True, "ignore_eos": ignore_eos, "return_token_ids": True}
        if temperature:
            self.body["top_k"] = 20
        if seed is not None:
            self.body["seed"] = int(seed)
        if logprobs:
            self.body["logprobs"] = int(logprobs)
            self.body["return_tokens_as_token_ids"] = True      # rev 3: logprob keys by id (no string collisions)
        self.text, self.tokens, self.margins = [], [], []
        self.err, self.done_seen, self.finish_reason = None, False, None
        self.t_start = self.t_first = self.t_end = None
        self._resp, self._halt = None, False

    def run(self):
        self.t_start = time.perf_counter()
        try:
            req = U.Request(f"http://127.0.0.1:{self.port}/v1/completions", data=json.dumps(self.body).encode(),
                            method="POST", headers={"Content-Type": "application/json"})
            self._resp = U.urlopen(req, timeout=7200)
            for raw in self._resp:
                if self._halt:
                    break
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    self.done_seen = True
                    break
                d = json.loads(payload)
                if "error" in d or d.get("object") == "error":
                    self.err = ("stream error: " + json.dumps(d.get("error") or d.get("message") or d))[:300]
                    break
                for c in d.get("choices") or []:
                    self.text.append(c.get("text") or "")
                    ids_ = c.get("token_ids")
                    lp = c.get("logprobs") or {}
                    if ids_ is None:
                        if c.get("text") or lp.get("tokens"):
                            self.err = "MISSING_STREAM_TOKEN_IDS"
                    elif any(not isinstance(v, int) or isinstance(v, bool) for v in ids_):
                        self.err = "INVALID_STREAM_TOKEN_IDS"
                    else:
                        if ids_ and self.t_first is None:
                            self.t_first = time.perf_counter()
                        self.tokens.extend(ids_)
                    for top in lp.get("top_logprobs") or []:
                        vals = sorted((v for v in (top or {}).values() if isinstance(v, (int, float))), reverse=True)
                        self.margins.append(round(vals[0] - vals[1], 4) if len(vals) >= 2 else None)
                    if c.get("finish_reason") is not None:
                        self.finish_reason = c["finish_reason"]
                if self.err:
                    break
        except Exception as e:
            if not self._halt:
                self.err = repr(e)[:300]
        finally:                         # rev 3: a stop() before the response existed still closes the socket here
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
        try:
            self._resp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            if self._resp is not None:
                self._resp.close()
        except Exception:
            pass

    def row(self):
        return {"text": "".join(self.text), "tokens": list(self.tokens), "margins": list(self.margins), "error": self.err,
                "done": self.done_seen, "finish_reason": self.finish_reason}


# ------------------------------------------------------------------ correctness
def greedy(port, prompts, n_tok=128, logprobs=2, timeout=900, model="q27"):
    """Greedy rows for the same-weight oracle (one request at a time: batch-size effects stay out of the comparison)."""
    rows = []
    for ids in prompts:
        g = Stream(port, ids, n_tok, temperature=0.0, logprobs=logprobs, ignore_eos=True, model=model)
        g.start()
        g.join(timeout)
        if g.is_alive():
            g.stop()
            g.join(10)
            r = g.row()
            r["error"] = r["error"] or "GREEDY_TIMEOUT"
        else:
            r = g.row()
        rows.append(r)
    return rows


def degenerate(text):
    """Objective garbage checks: empty, a run of one char, '!!!!' (the E3 / E13 garbage), very low variety."""
    t = text or ""
    if not t.strip():
        return "EMPTY"
    if re.search(r"(\S)\1{200,}", t, re.S):      # rev 3: grid rows / separators are not garbage
        return "REPEATED_CHAR"
    if t.count("!") > 0.5 * max(1, len(t)):
        return "BANG_GARBAGE"
    words = re.findall(r"\w+", t)
    if len(words) > 40 and len(set(words)) < 0.08 * len(words):
        return "LOW_VARIETY"
    return None


def compare(ref, cand):
    """Same-weight oracle: at the first differing token the ORACLE's top-1 - top-2 logprob margin decides:
    >= FAIL_MARGIN -> FAIL_HIGH_MARGIN, below -> NEAR_TIE; identical -> EXACT; missing data -> INCONCLUSIVE."""
    if not ref or not cand or len(ref) != len(cand):
        return {"status": "INCONCLUSIVE_PAIRS_MISSING"}
    per, worst = [], "EXACT"
    rank = {"EXACT": 0, "NEAR_TIE": 1, "INCONCLUSIVE": 2, "FAIL_HIGH_MARGIN": 3}
    for a, b in zip(ref, cand):
        if a.get("error") or b.get("error") or not a.get("tokens") or not b.get("tokens"):
            st, d, m = "INCONCLUSIVE", None, None
        else:
            x, y = a["tokens"], b["tokens"]
            d = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)
            if d is None:
                st, m = ("EXACT" if len(x) == len(y) else "INCONCLUSIVE"), None
            else:
                mm = a.get("margins") or []
                m = mm[d] if d < len(mm) else None
                st = "INCONCLUSIVE" if m is None else ("FAIL_HIGH_MARGIN" if m >= FAIL_MARGIN else "NEAR_TIE")
        g = degenerate(b.get("text"))
        if g and degenerate(a.get("text")) is None:           # rev 3: only when the oracle itself is clean
            st = "FAIL_HIGH_MARGIN"
        per.append({"first_diff": d, "oracle_margin": m, "status": st, "garbage": g})
        worst = st if rank[st] > rank[worst] else worst
    return {"status": worst, "rows": per}


def chat_probes(port, model="q27", cases=(("Reply with exactly one number: 7 * 6 =", "42"),
                                          ("Reply with exactly one number: 2 + 2 =", "4"),
                                          ("Reply with exactly one number: 9 - 3 =", "6")), timeout=300):
    """Arithmetic through /v1/chat/completions with thinking off. vLLM 0.32 returns the answer in message.content and
    the thinking in message.reasoning (NOT reasoning_content: the consolidated notebook's blocker)."""
    out = []
    for q, want in cases:
        body = {"model": model, "messages": [{"role": "user", "content": q}], "temperature": 0, "max_tokens": 32,
                "chat_template_kwargs": {"enable_thinking": False}}
        try:
            r = json.loads(U.urlopen(U.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                               headers={"Content-Type": "application/json"}), timeout=timeout).read())
            msg = r["choices"][0]["message"]
            got = (msg.get("content") or "").strip()
            out.append({"q": q, "want": want, "content": got[:80], "reasoning_chars": len(msg.get("reasoning") or ""),
                        "ok": re.search(r"(?<!\d)" + want + r"(?!\d)", got) is not None})
        except Exception as e:
            out.append({"q": q, "want": want, "error": repr(e)[:200], "ok": False})
    return {"ok": all(x["ok"] for x in out), "cases": out}


def teacher_forced(port, ids, last=256, model="q27", timeout=1800):
    """rev 3 (measurement M3): prompt logprobs of the last `last` positions of a long prompt (one prefill, 1 token out),
    for a lossy-KV comparison (NVFP4 vs FP8 KV) at the context lengths that matter. Returns per-position
    (logprob of the actual token, rank) or an error."""
    body = {"model": model, "prompt": list(ids), "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1,
            "return_tokens_as_token_ids": True}
    try:
        r = json.loads(U.urlopen(U.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                           headers={"Content-Type": "application/json"}), timeout=timeout).read())
        pl = r["choices"][0].get("prompt_logprobs") or []
    except Exception as e:
        return {"error": repr(e)[:300]}
    out = []
    for pos in range(max(1, len(ids) - last), len(ids)):
        d = pl[pos] if pos < len(pl) else None
        if not isinstance(d, dict):
            out.append(None)
            continue
        e = d.get(str(ids[pos])) or d.get(f"token_id:{ids[pos]}") or next(iter(d.values()), None)
        out.append([e.get("logprob"), e.get("rank")] if isinstance(e, dict) else None)
    return {"positions": out, "n": len(out)}


def compare_tf(ref, cand):
    """Teacher-forced comparison: mean / max |delta logprob| of the actual tokens and rank-1 agreement."""
    if not ref or not cand or ref.get("error") or cand.get("error"):
        return {"status": "INCONCLUSIVE"}
    pairs = [(a, b) for a, b in zip(ref["positions"], cand["positions"]) if a and b and a[0] is not None and b[0] is not None]
    if not pairs:
        return {"status": "INCONCLUSIVE"}
    d = [abs(a[0] - b[0]) for a, b in pairs]
    r1 = sum(1 for a, b in pairs if (a[1] == 1) == (b[1] == 1)) / len(pairs)
    return {"status": "MEASURED", "n": len(pairs), "mean_abs_dlogprob": round(statistics.mean(d), 4),
            "max_abs_dlogprob": round(max(d), 3), "rank1_agreement": round(r1, 4)}


# ------------------------------------------------------------------ steady decode
def steady(port, prompts, level, secs=60.0, admit=None, max_new=8192, temperature=1.0, seed=1234, model="q27",
           sample=2.0, prefill_tok_s=700.0, gpu_sample=True, barrier=None):
    """`level` distinct prompts, then a decode window that starts only once prefill has settled. Status:
    STEADY / CAPACITY_LIMITED (fewer running than asked: KV) / INCONCLUSIVE_* / FAIL_*; numbers are kept on an
    INCONCLUSIVE window so a KV census survives; never a made-up number. `barrier` (steady_multi) starts every replica's
    window at the same moment."""
    plen = len(prompts[0]) if prompts else 0
    admit = admit or max(420.0, 120.0 + level * plen / prefill_tok_s)
    r = {"level": level, "status": "INCONCLUSIVE_NOT_STARTED", "port": port, "prompt_tokens": plen, "max_new": max_new,
         "admit_limit_sec": round(admit)}
    if not wait_idle(port, 60):
        r["status"] = "INCONCLUSIVE_NOT_IDLE_AT_START"
        if barrier:
            barrier.abort()
        return r
    try:
        p0 = prom(port)
    except Exception as e:
        r.update(status="INCONCLUSIVE_METRICS_UNAVAILABLE", error=repr(e)[:200])
        if barrier:
            barrier.abort()
        return r
    if GEN not in p0 or PROMPT not in p0:
        r["status"] = "INCONCLUSIVE_COUNTERS_MISSING"
        if barrier:
            barrier.abort()
        return r
    ss = [Stream(port, prompts[i % len(prompts)], max_new, temperature, seed + i, model=model) for i in range(level)]
    for g in ss:
        g.start()
    try:
        t_adm, best, settled, prev, samples, flat_since = time.perf_counter(), 0, 0, None, [], None
        last_tok = [0] * len(ss)
        while time.perf_counter() - t_adm < admit:
            time.sleep(sample)
            if all(not g.is_alive() for g in ss):          # rev 3: checked before the metrics read (a dead server)
                r.update(status="INCONCLUSIVE_STREAMS_ENDED_IN_ADMISSION", errors=[g.err for g in ss if g.err][:3])
                return r
            try:
                m = prom(port)
            except Exception:
                continue
            run, wait, pt, pre = int(m.get(RUN, 0)), int(m.get(WAIT, 0)), m.get(PROMPT), m.get(PREEMPT, 0)
            now_tok = [len(g.tokens) for g in ss]
            first = sum(g.t_first is not None for g in ss)
            gaining = sum(1 for a, b in zip(last_tok, now_tok) if b > a)
            last_tok = now_tok
            best = max(best, run)
            samples.append([round(time.perf_counter() - t_adm, 1), run, wait, first, gaining])
            # rev 3: no preemption in the sample, every running request emitted and is gaining tokens
            calm = run > 0 and prev is not None and pt == prev[0] and pre == prev[1] and first >= run and gaining >= run
            if calm and run >= level and wait == 0:
                settled += 1
            elif calm and run < level and wait > 0:       # KV-bound: the rest wait; accept a plateau held 30 s
                flat_since = flat_since or time.perf_counter()
                settled = settled + 1 if time.perf_counter() - flat_since > 30 else 0
            else:
                settled, flat_since = 0, None
            prev = (pt, pre)
            if settled >= 3:
                break
        r["admission"] = {"sec": round(time.perf_counter() - t_adm, 1), "best_running": best, "tail": samples[-6:],
                          "prefill_tok_s": round((prom(port).get(PROMPT, 0) - p0.get(PROMPT, 0)) /
                                                 max(1e-3, time.perf_counter() - t_adm), 1)}
        if settled < 3:
            r["status"] = "INCONCLUSIVE_NO_SETTLE"
            if barrier:
                barrier.abort()
            return r
        if barrier:
            try:
                barrier.wait(timeout=max(60.0, admit))
            except threading.BrokenBarrierError:
                r["status"] = "INCONCLUSIVE_REPLICA_BARRIER"
                return r
        m0, t0 = prom(port), time.perf_counter()
        runs, waits, kvs, mids, gmem, stalls = [], [], [], [], [], [0] * len(ss)
        last_tok = [len(g.tokens) for g in ss]
        while time.perf_counter() - t0 < secs:
            time.sleep(min(sample, max(0.1, secs - (time.perf_counter() - t0))))
            m = prom(port)
            mids.append(m)
            runs.append(m.get(RUN, 0))
            waits.append(m.get(WAIT, 0))
            if KVU in m:
                kvs.append(m[KVU])
            now_tok = [len(g.tokens) for g in ss]
            for i, g in enumerate(ss):
                if g.t_first is not None and g.is_alive():
                    stalls[i] = stalls[i] + 1 if now_tok[i] <= last_tok[i] else 0
            last_tok = now_tok
            if any(x >= 3 for x in stalls):
                break
            if gpu_sample and len(mids) % 5 == 1:
                g_ = gpu_mem()
                if g_:
                    gmem.append(g_)
        m1, t1 = prom(port), time.perf_counter()
        gen = m1.get(GEN, 0) - m0.get(GEN, 0)
        dt = t1 - t0
        acc, drf, dtk, it = (m1.get(k, 0) - m0.get(k, 0) for k in (ACC, DRF, DTOK, ITER))
        rm = statistics.mean(runs) if runs else 0
        steps = it if it else ((drf / rm) if (drf and rm) else ((gen / rm) if rm else 0))
        r.update(tok_s=round(gen / dt, 1) if dt else None, gen_tokens=int(gen), window_sec=round(dt, 1),
                 running_mean=round(rm, 2), running_min=min(runs or [0]), running_max=max(runs or [0]),
                 waiting_min=min(waits or [0]), waiting_max=max(waits or [0]),
                 tok_s_per_stream=round(gen / dt / rm, 2) if rm and dt else None,
                 accept_per_draft=round(acc / drf, 3) if drf else None,
                 mean_acceptance_length=round(1 + acc / drf, 3) if drf else None,
                 accept_rate=round(acc / dtk, 3) if dtk else None,
                 steps=int(steps), steps_source="iteration_histogram" if it else "estimate",
                 step_ms=round(dt * 1000 / steps, 1) if steps else None,
                 tokens_per_step=round(gen / steps, 2) if steps else None,
                 kv_usage_mean=round(statistics.mean(kvs), 3) if kvs else None, kv_usage_max=max(kvs) if kvs else None,
                 gpu_mem_peak_mib=[max(row[i][1] for row in gmem if len(row) > i) for i in range(max(map(len, gmem)))] if gmem else None,
                 preemptions=m1.get(PREEMPT, 0) - m0.get(PREEMPT, 0), t0=t0, t1=t1)
        if any(x >= 3 for x in stalls):
            r["status"] = "INCONCLUSIVE_STALLED_STREAM"
        elif sum(g.is_alive() for g in ss) < min(level, best):
            r["status"] = "INCONCLUSIVE_STREAMS_ENDED_IN_WINDOW"
        elif any(x.get(PROMPT) != m0.get(PROMPT) for x in mids + [m1]):
            r["status"] = "INCONCLUSIVE_PREFILL_IN_WINDOW"
        elif r["preemptions"] > 0:
            r["status"] = "INCONCLUSIVE_PREEMPTION"
        elif gen <= 0:
            r["status"] = "FAIL_NO_DECODE"
        elif min(runs or [0]) < level and (len(set(runs)) > 1 or len(set(waits)) > 1):
            r["status"] = "INCONCLUSIVE_RUNNING_SET_CHANGED"      # KV-bound window must hold one fixed set
        elif min(runs or [0]) < 0.8 * max(runs or [1]):
            r["status"] = "INCONCLUSIVE_RUNNING_UNSTABLE"
        else:
            r["status"] = ("STEADY" if min(runs or [0]) == max(runs or [0]) == level else
                           "CAPACITY_LIMITED" if max(runs or [0]) < level else "INCONCLUSIVE_RUNNING_ABOVE_LEVEL")
        return r
    finally:
        errs = sorted({g.err for g in ss if g.err})[:3]           # rev 3: snapshot before stop() sets _halt
        for g in ss:
            g.stop()
        for g in ss:
            g.join(10)
        r["client_leak"] = any(g.is_alive() for g in ss)
        r["stream_errors"] = errs
        gb = [degenerate("".join(g.text)[-4000:]) if g.tokens else None for g in ss]
        r["garbage"] = {"streams": len(ss), "with_output": sum(1 for g in ss if g.tokens),
                        "garbage": [x for x in gb if x], "sample_tail": "".join(ss[0].text)[-200:] if ss and ss[0].tokens else ""}
        if r["garbage"]["garbage"] and r.get("status") in ("STEADY", "CAPACITY_LIMITED"):
            r["status"] = "FAIL_GARBAGE_UNDER_LOAD"
        r["idle_after"] = wait_idle(port)


def steady_multi(jobs, secs=60.0, **kw):
    """Several servers at once (TP1 replicas): jobs = [(port, prompts, level)]; one shared window (barrier). The aggregate
    counts only when every window is STEADY / CAPACITY_LIMITED and the windows overlap >= 80 % of the shortest."""
    out = [None] * len(jobs)
    bar = threading.Barrier(len(jobs))

    def one(i, j):
        try:
            out[i] = steady(j[0], j[1], j[2], secs=secs, barrier=bar, **kw)
        except Exception as e:          # rev 3: a dead replica is recorded, not printed to stderr only
            out[i] = {"status": "ERROR_REPLICA", "error": f"{type(e).__name__}: {e}"[:300], "port": j[0]}
            try:
                bar.abort()
            except Exception:
                pass
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


def wait_idle(port, wait=90.0):
    end, hits = time.perf_counter() + wait, 0
    while time.perf_counter() < end:
        try:
            m = prom(port)
            hits = hits + 1 if m.get(RUN, 0) == 0 and m.get(WAIT, 0) == 0 else 0
        except Exception:
            hits = 0
        if hits >= 3:
            return True
        time.sleep(2)
    return False


# ------------------------------------------------------------------ profiler (FULL-graph proof + per-category time)
# rev 3: matched on the kernel BASE name, attention before GDN ("fla" is a substring of "flashinfer"), real 0.32 GDN names
CATS = [("nccl", ("nccl", "allreduce", "all_reduce", "allgather", "all_gather")),
        ("attention", ("flashinfer", "flash_fwd", "unified_attention", "paged", "attn", "attention", "batchdecode", "batchprefill")),
        ("gdn_linear_attn", ("gdn", "delta_rule", "chunk_", "recurrent", "causal_conv", "l2norm", "solve_tril", "merge_16x16",
                             "recompute_w_u", "kkt", "cumsum", "post_conv", "layer_norm_gated", "gated_delta")),
        ("gemm_marlin", ("marlin",)),
        ("gemm_other", ("gemm", "cutlass", "xmma", "cublas", "sm89_", "ampere_", "sm80_")),
        ("sampling", ("sampl", "topk", "top_k", "argmax", "softmax", "rejection"))]


def _cat(name):
    base = str(name).split("<")[0].split("(")[0].lower()
    return next((k for k, pats in CATS if any(p in base for p in pats)), "other")


def trace_summary(pdir):
    files = sorted(Path(pdir).rglob("*.pt.trace.json*"), key=lambda p: p.stat().st_mtime)
    if not files:
        return {"error": "NO_TRACE_FILE", "listing": [str(p) for p in Path(pdir).rglob("*")][:20]}
    f = next((p for p in files if re.search(r"rank0\.|rank[-_]?0[._]|tp0_", p.name)), files[0])
    with (gzip.open if f.name.endswith(".gz") else open)(f, "rt", encoding="utf-8") as fh:
        tr = json.load(fh)
    ev = tr.get("traceEvents", [])
    kern = [e for e in ev if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("kernel", "gpu_kernel")]
    rt = [e for e in ev if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("cuda_runtime", "cuda_driver")]
    is_graph = lambda e: "GraphLaunch" in str(e.get("name", ""))
    is_kl = lambda e: re.search(r"LaunchKernel|cuLaunchKernel", str(e.get("name", ""))) is not None
    # per-forward counts inside gpu_model_runner forward annotations (needs VLLM_CUSTOM_SCOPES_FOR_PROFILING=1)
    fw = [e for e in ev if e.get("ph") == "X" and "forward" in str(e.get("name", "")).lower()
          and "gpu_model_runner" in str(e.get("name", "")).lower()]
    per_fw = []
    for a in fw[:400]:
        lo, hi, tid = a.get("ts", 0), a.get("ts", 0) + a.get("dur", 0), a.get("tid")
        inside = [e for e in rt if e.get("tid") == tid and lo <= e.get("ts", -1) <= hi]
        per_fw.append([sum(1 for e in inside if is_graph(e)), sum(1 for e in inside if is_kl(e))])
    corr = {(e.get("args") or {}).get("correlation"): str(e.get("name", "")) for e in rt
            if (e.get("args") or {}).get("correlation") is not None}
    per, tot, names = {k: 0.0 for k, _ in CATS + [("other", ())]}, 0.0, {}
    src = {k: {"graph": 0, "eager": 0, "unknown": 0} for k in per}
    for e in kern:
        d = float(e.get("dur", 0.0))
        tot += d
        names[e.get("name", "")] = names.get(e.get("name", ""), 0.0) + d
        k = _cat(e.get("name", ""))
        per[k] += d
        launcher = corr.get((e.get("args") or {}).get("correlation"))
        src[k]["graph" if launcher and "GraphLaunch" in launcher else ("eager" if launcher else "unknown")] += 1
    tot_k = sum(v["graph"] + v["eager"] for v in src.values())
    eager_share = round(sum(v["eager"] for v in src.values()) / tot_k, 3) if tot_k else None
    fw_g = statistics.median([x[0] for x in per_fw]) if per_fw else None
    fw_k = statistics.median([x[1] for x in per_fw]) if per_fw else None
    if per_fw:
        verdict = "FULL" if fw_g == 1 and fw_k == 0 else ("PIECEWISE" if fw_g and fw_g > 1 else ("EAGER" if not fw_g else "MIXED"))
        basis = "per_forward_annotation"
    elif tot_k:
        att = src["attention"]
        verdict = ("FULL" if att["eager"] == 0 and (eager_share or 0) < 0.05 else
                   "PIECEWISE" if att["eager"] > 0 and any(v["graph"] for v in src.values()) else
                   "EAGER" if not any(v["graph"] for v in src.values()) else "MIXED")
        basis = "cupti_correlation"
    else:
        verdict, basis = "UNKNOWN", "none"
    return {"file": f.name, "files": len(files), "kernels": len(kern), "kernel_us_sum": round(tot),
            "graph_launches": sum(1 for e in rt if is_graph(e)), "kernel_launches": sum(1 for e in rt if is_kl(e)),
            "forwards": len(per_fw), "per_forward_median_graph_launches": fw_g, "per_forward_median_kernel_launches": fw_k,
            "launch_source": src, "eager_kernel_share": eager_share, "graph_verdict": verdict, "verdict_basis": basis,
            "by_category_pct": {k: round(100 * v / tot, 1) for k, v in per.items()} if tot else {},
            "top_kernels_us": dict(sorted(((k, round(v)) for k, v in names.items()), key=lambda x: -x[1])[:25]),
            "note": "kernel time sums overlapping streams; NCCL time includes rank-skew waiting"}


def profile(port, pdir, secs=6.0, delay_iters=20, max_iters=40):
    """POST /start_profile (bounded by iterations; falls back to a bare POST), decode, POST /stop_profile, summary."""
    rec = {}
    try:
        U.urlopen(U.Request(f"http://127.0.0.1:{port}/start_profile", method="POST",
                            data=json.dumps({"delay_iterations": delay_iters, "max_iterations": max_iters}).encode(),
                            headers={"Content-Type": "application/json"}), timeout=900).read()
        rec["start_profile"] = "ok_bounded"
    except Exception as e:
        rec["start_profile_bounded_error"] = repr(e)[:200]
        try:
            U.urlopen(U.Request(f"http://127.0.0.1:{port}/start_profile", data=b"", method="POST"), timeout=900).read()
            rec["start_profile"] = "ok"
        except Exception as e2:
            rec["start_profile"] = repr(e2)[:200]
    time.sleep(secs)
    try:
        U.urlopen(U.Request(f"http://127.0.0.1:{port}/stop_profile", data=b"", method="POST"), timeout=900).read()
        rec["stop_profile"] = "ok"
    except Exception as e:
        rec["stop_profile"] = repr(e)[:200]
    time.sleep(20)
    try:
        rec["trace"] = trace_summary(pdir)
    except Exception as e:
        rec["trace"] = {"error": f"TRACE_PARSE {type(e).__name__}: {e}"[:300]}
    return rec


def profiled_load(port, prompts, level, pdir, secs=6.0, max_new=8192, model="q27", prefill_tok_s=700.0):
    """`level` decoding streams (own window: the profiler slows decode), profile() only after prefill settled exactly
    as in steady() (prompt counter + preemptions flat, every running request emitted and gaining tokens)."""
    plen = len(prompts[0]) if prompts else 0
    settle = max(300.0, 120.0 + level * plen / prefill_tok_s)
    ss = [Stream(port, prompts[i % len(prompts)], max_new, 1.0, 4321 + i, model=model) for i in range(level)]
    for g in ss:
        g.start()
    rec, t0, flat, ok, prev, last_tok = {"level": level}, time.perf_counter(), None, 0, None, [0] * level
    try:
        while time.perf_counter() - t0 < settle:
            time.sleep(2)
            try:
                m = prom(port)
            except Exception:
                continue
            run, wait, pt, pre = int(m.get(RUN, 0)), int(m.get(WAIT, 0)), m.get(PROMPT), m.get(PREEMPT, 0)
            now_tok = [len(g.tokens) for g in ss]
            gaining = sum(1 for a, b in zip(last_tok, now_tok) if b > a)
            last_tok = now_tok
            first = sum(g.t_first is not None for g in ss)
            calm = run > 0 and prev is not None and (pt, pre) == prev and first >= run and gaining >= run
            if calm and run >= level and wait == 0:
                ok += 1
            elif calm and wait > 0:
                flat = flat or time.perf_counter()
                ok = ok + 1 if time.perf_counter() - flat > 30 else 0
            else:
                ok, flat = 0, None
            prev = (pt, pre)
            if ok >= 3:
                break
        rec["settled"] = ok >= 3
        try:
            rec["running_at_start"] = prom(port).get(RUN)
        except Exception:
            pass
        rec.update(profile(port, pdir, secs))
        if not rec["settled"] and isinstance(rec.get("trace"), dict):
            rec["trace"]["note2"] = "prefill had not settled: the trace may mix prefill and decode"
        return rec
    finally:
        for g in ss:
            g.stop()
        for g in ss:
            g.join(10)
        rec["idle_after"] = wait_idle(port)
