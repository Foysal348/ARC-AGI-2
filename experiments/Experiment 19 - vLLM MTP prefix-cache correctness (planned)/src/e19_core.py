# ==================================================================== CELL E19-1 — vLLM 0.19 OpenAI server per arm, streaming greedy client
# E19 (2026-10-27): the S1-60K engine (vLLM 0.19.0, FP8 repack, TP4, fp8_e4m3 KV, MTP 3, FLASHINFER if -lcuda links,
# prefix caching, 4,096 batched tokens, mem 0.90, custom all-reduce off, language-model-only, max_model_len 106,496,
# seed 20261005, 16 seqs) is launched as an OpenAI server from the kernel's own Python (the vLLM 0.19 that CELL 2
# installed), one server per arm, killed between arms. Only the arm's own switch changes (MTP on/off, prefix caching
# on/off, KV dtype). Two frontend-only additions on EVERY arm: --enable-prompt-tokens-details (cached tokens in the
# usage chunk) and VLLM_SERVER_DEV_MODE=1 (POST /reset_prefix_cache); neither changes the engine's computation.
# Arm E adds --no-async-scheduling (suspect: the async-scheduling race in the V1 model runner, vLLM #53919).
# Requests: streaming /v1/completions with the prompt token ids, temperature 0 (greedy), logprobs 2 with
# return_tokens_as_token_ids (top-1 / top-2 margins keyed by token id), return_token_ids (integer ids per chunk) and
# stream_options.include_usage (prompt_tokens_details.cached_tokens).
import shlex, socket, signal as _signal, urllib.request as _urlreq, statistics
import shutil as _shutil

E19_FAKE = os.environ.get("E19_FAKE_SERVER")          # local CPU harness only: a fake vLLM 0.19 OpenAI server script
E19_PORT = int(os.environ.get("E19_PORT", 8019))
E19_URL = f"http://127.0.0.1:{E19_PORT}"
E19_DIR = OUT / "e19"
E19_DIR.mkdir(parents=True, exist_ok=True)
E19_ROWS_PATH = E19_DIR / "e19_rows.jsonl"
E19_STATE_PATH = E19_DIR / "e19_state.json"
E19_SILENCE_SEC = float(os.environ.get("E19_SILENCE_SEC", 600))
E19_READY_MAX_SEC = float(os.environ.get("E19_READY_MAX_SEC", 25 * 60))
E19_LOCK = threading.Lock()
E19_SAVE_LOCK = threading.Lock()
E19_STATE = {"run_id": RUN_ID, "profile": PROFILE, "arms": {}, "order": [], "events": []}
E19_SERVER = None

# @@S1_LIBCUDA@@


def e19_event(msg):
    line = f"[{time.perf_counter() - NOTEBOOK_T0:7.0f}s] {msg}"
    print(line, flush=True)
    with E19_LOCK:
        E19_STATE["events"].append(line[:600])


def e19_save_state():
    """Atomic (tmp + fsync + replace): a crash mid-write leaves the previous state file. One writer at a time: the
    churn / long-stage worker threads save concurrently."""
    with E19_SAVE_LOCK:
        with E19_LOCK:
            snap = json.dumps(E19_STATE, default=str, indent=1)
        tmp = Path(str(E19_STATE_PATH) + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(snap)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, E19_STATE_PATH)


def e19_write_row(row):
    """Append one raw row (flush + fsync) as soon as its request ends: a crash keeps every finished probe."""
    line = json.dumps(row, separators=(",", ":"), default=str)
    with E19_LOCK:
        with open(E19_ROWS_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        a = E19_STATE["arms"].get(row.get("arm"))
        if a is not None:
            a.setdefault("row_status", {})
            a["row_status"][row["status"]] = a["row_status"].get(row["status"], 0) + 1
    e19_save_state()


def e19_flags(arm, backend, max_len=None):
    """S1's ENGINE_CONFIG (CELL 5) as api_server flags; the arm sets MTP, prefix caching, the KV dtype and (arm E)
    async scheduling. Async scheduling is left at vLLM's default on every other arm, as in S1."""
    cfg = E19_ARMS[arm]
    f = ["--model", str(MODEL_PATH), "--tensor-parallel-size", "4", "--distributed-executor-backend", "mp",
         "--dtype", "auto", "--load-format", "safetensors", "--language-model-only",
         "--max-model-len", str(max_len or MAX_MODEL_LEN), "--max-num-seqs", str(MAX_NUM_SEQS),
         "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS), "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
         "--disable-custom-all-reduce", "--seed", str(SEED), "--attention-backend", backend,
         "--kv-cache-dtype", cfg["kv"]]
    f += ["--enable-prefix-caching"] if cfg["prefix"] else ["--no-enable-prefix-caching"]
    if cfg["spec"]:
        f += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": int(cfg["spec"])})]
    if cfg.get("async") is False:
        f += ["--no-async-scheduling"]           # arm E: the cheapest workaround for the async-scheduling race
    f += ["--host", "127.0.0.1", "--port", str(E19_PORT), "--served-model-name", "q27", "--enable-prompt-tokens-details"]
    return f


class E19Server:
    def __init__(self, name, flags):
        self.name, self.flags = name, list(flags)
        self.cmd = ([sys.executable, E19_FAKE] if E19_FAKE else
                    [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]) + self.flags
        self.env = dict(os.environ, VLLM_SERVER_DEV_MODE="1", PYTHONUNBUFFERED="1")
        self.log_path = E19_DIR / f"vllm_server_{name}.log"
        self.proc = None

    def start(self):
        self.log_f = open(self.log_path, "w", encoding="utf-8")
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.cmd, env=self.env, stdout=self.log_f, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True, cwd=None if E19_FAKE else "/tmp")

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def log_text(self):
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def wait_ready(self, timeout, silence=E19_SILENCE_SEC):
        """ready = /health answers AND this server's own 'GPU KV cache size' line is in its log (E18 rev 2); a log that
        stops growing for `silence` seconds is a stall."""
        last, last_t = -1, time.perf_counter()
        while time.perf_counter() - self.t0 < timeout:
            if not self.alive():
                return False, f"exited rc={self.proc.returncode}"
            try:
                _urlreq.urlopen(E19_URL + "/health", timeout=10).read()
                if "GPU KV cache size" in self.log_text():
                    return True, f"ready in {time.perf_counter() - self.t0:.0f}s"
            except Exception:
                pass
            sz = self.log_path.stat().st_size if self.log_path.exists() else 0
            if sz != last:
                last, last_t = sz, time.perf_counter()
            elif time.perf_counter() - last_t > silence:
                return False, f"stall ({silence:.0f} s of log silence)"
            time.sleep(0.5 if E19_FAKE else 5)
        return False, "timeout"

    def kill(self):
        """SIGTERM to the process group, a grace, then SIGKILL to the group ALWAYS (engine core / TP workers stuck in
        CUDA or NCCL can outlive the API server), then wait."""
        if self.proc is None:
            return
        grp = hasattr(os, "killpg")
        try:
            os.killpg(self.proc.pid, _signal.SIGTERM) if grp else self.proc.terminate()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=20)
        except Exception:
            pass
        try:
            os.killpg(self.proc.pid, getattr(_signal, "SIGKILL", _signal.SIGTERM)) if grp else self.proc.kill()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=30)
        except Exception:
            pass
        try:
            self.log_f.close()
        except Exception:
            pass


def e19_kill_gpu_pids():
    """Kill every process still holding a GPU compute context (never this kernel)."""
    if E19_FAKE:
        return []
    killed = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20).stdout
        for tok in out.split():
            if tok.isdigit() and int(tok) != os.getpid():
                try:
                    os.kill(int(tok), getattr(_signal, "SIGKILL", _signal.SIGTERM))
                    killed.append(int(tok))
                except Exception:
                    pass
    except Exception:
        pass
    return killed


def e19_wait_gpu_free(max_sec=float(os.environ.get("Q27X_GPU_WAIT_MAX", 180))):
    """S1's rule: every GPU < 1 GB used (nvidia-smi)."""
    if E19_FAKE:
        return {"released": True, "sec": 0.0, "fake": True}
    w0, used = time.perf_counter(), []
    while time.perf_counter() - w0 < max_sec:
        try:
            used = [int(x) for x in subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                                   capture_output=True, text=True, timeout=20).stdout.split()]
        except Exception:
            used = []
        if used and max(used) < 1024:
            return {"released": True, "sec": round(time.perf_counter() - w0, 1), "last_used_mb": used}
        time.sleep(5)
    return {"released": False, "sec": round(time.perf_counter() - w0, 1), "last_used_mb": used}


def e19_stop_server():
    global E19_SERVER
    rec = {}
    if E19_SERVER is not None:
        E19_SERVER.kill()
        E19_SERVER = None
    rec["killed_gpu_pids"] = e19_kill_gpu_pids()
    rec["gpu"] = e19_wait_gpu_free()
    return rec


def e19_prom():
    txt = _urlreq.urlopen(E19_URL + "/metrics", timeout=15).read().decode("utf-8", "replace")
    out = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)", line)
        if m and not m.group(1).endswith("_created"):
            try:
                out[m.group(1)] = out.get(m.group(1), 0.0) + float(m.group(3))
            except ValueError:
                pass
    return out


E19_METRICS = {"gen": "vllm:generation_tokens_total", "prompt": "vllm:prompt_tokens_total",
               "pc_queries": "vllm:prefix_cache_queries_total", "pc_hits": "vllm:prefix_cache_hits_total",
               "preempt": "vllm:num_preemptions_total", "drafts": "vllm:spec_decode_num_drafts_total",
               "draft_tokens": "vllm:spec_decode_num_draft_tokens_total",
               "accepted": "vllm:spec_decode_num_accepted_tokens_total", "running": "vllm:num_requests_running"}


def e19_metrics():
    try:
        p = e19_prom()
        return {k: p.get(v) for k, v in E19_METRICS.items()}
    except Exception as e:
        return {"error": repr(e)[:200]}


def e19_delta(a, b):
    return {k: (round(b[k] - a[k], 1) if isinstance(a.get(k), (int, float)) and isinstance(b.get(k), (int, float)) else None)
            for k in E19_METRICS}


def e19_reset_cache():
    """POST /reset_prefix_cache (VLLM_SERVER_DEV_MODE=1). Called on every arm (a no-op with prefix caching off).
    vLLM does not report whether it worked; the cold rows' cached_tokens show it."""
    try:
        req = _urlreq.Request(E19_URL + "/reset_prefix_cache", data=b"", method="POST")
        return _urlreq.urlopen(req, timeout=60).status
    except Exception as e:
        return repr(e)[:120]


class E19Req(threading.Thread):
    """One greedy streaming /v1/completions request with prompt token ids. Records integer token ids, top-1 / top-2
    margins, positions where the emitted token is not the top-1 of its own logprobs, finish_reason, [DONE], the usage
    chunk's cached tokens and TTFT. stop() closes the stream (vLLM aborts the request on disconnect).
    The halt flag is _halt, NOT _stop (Thread._stop is a method join() calls on Python 3.12)."""
    def __init__(self, ids, max_new):
        super().__init__(daemon=True)
        self.body = {"model": "q27", "prompt": list(ids), "max_tokens": int(max_new), "temperature": 0.0, "top_p": 1.0,
                     "stream": True, "stream_options": {"include_usage": True}, "return_token_ids": True,
                     "logprobs": 2, "return_tokens_as_token_ids": True, "skip_special_tokens": False,
                     "ignore_eos": False}
        self.tokens, self.margins, self.mismatch = [], [], []
        self.err, self._resp, self._halt = None, None, False
        self.done_seen, self.finish_reason, self.usage = False, None, None
        self.t_start, self.t_first, self.t_end = None, None, None

    def run(self):
        self.t_start = time.perf_counter()
        try:
            req = _urlreq.Request(E19_URL + "/v1/completions", data=json.dumps(self.body).encode(), method="POST",
                                  headers={"Content-Type": "application/json"})
            self._resp = _urlreq.urlopen(req, timeout=900)
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
                if d.get("usage") is not None and not d.get("choices"):
                    self.usage = d["usage"]
                for c in d.get("choices") or []:
                    ids_ = c.get("token_ids")
                    lp = c.get("logprobs") or {}
                    if ids_ is None:
                        if c.get("text"):
                            self.err = "MISSING_STREAM_TOKEN_IDS"
                        ids_ = []
                    if any(not isinstance(v, int) or isinstance(v, bool) for v in ids_):
                        self.err = "INVALID_STREAM_TOKEN_IDS"
                        break
                    tops = lp.get("top_logprobs") or []
                    for j, t in enumerate(ids_):
                        top = tops[j] if j < len(tops) else None
                        m, mm = None, False
                        if isinstance(top, dict) and top:
                            items = []
                            for k_, v_ in top.items():
                                try:
                                    items.append((int(str(k_).split(":")[-1]), float(v_)))
                                except ValueError:
                                    pass
                            items.sort(key=lambda x: -x[1])
                            if len(items) >= 2:
                                m = round(items[0][1] - items[1][1], 4)
                            own = next((v_ for k_, v_ in items if k_ == t), None)
                            mm = bool(items) and (own is None or items[0][1] - own > 1e-4)
                        if mm:
                            self.mismatch.append(len(self.tokens) + j)
                        self.margins.append(m)
                    if ids_ and self.t_first is None:
                        self.t_first = time.perf_counter()
                    self.tokens.extend(ids_)
                    if c.get("finish_reason") is not None:
                        self.finish_reason = c["finish_reason"]
                if self.err:
                    break
        except Exception as e:
            if not self._halt:
                self.err = repr(e)[:300]
        self.t_end = time.perf_counter()
        if not self._halt and not self.err and not (self.done_seen and self.finish_reason):
            self.err = "INCOMPLETE_SSE: [DONE] seen %s, finish_reason %s" % (self.done_seen, self.finish_reason)

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


def e19_run_req(ids, max_new, deadline):
    """Run one request to completion or until `deadline` (then it is stopped: status not_tested_budget)."""
    g = E19Req(ids, max_new)
    g.start()
    while g.is_alive():
        g.join(min(2.0, max(0.05, deadline - time.perf_counter())))
        if time.perf_counter() >= deadline and g.is_alive():
            g.stop()
            g.join(10)
            g.err = g.err or "stopped: arm deadline"
            g.budget_stop = True
            break
        if E19_SERVER is not None and not E19_SERVER.alive() and g.is_alive():
            g.join(5)
            if g.is_alive():
                g.stop()
                g.join(10)
            g.err = g.err or "server process died"
            break
    return g


def e19_decode(ids):
    try:
        return tokenizer.decode(list(ids), skip_special_tokens=False)
    except Exception as e:
        return f"<decode error {e!r}>"[:200]


def e19_row(arm, stage, ps, idx, ids, g, conc, extra=None):
    usage = g.usage or {}
    det = usage.get("prompt_tokens_details") or {}
    cached = det.get("cached_tokens") if isinstance(det, dict) else None
    if g.usage is not None and cached is None:
        cached = 0           # vLLM 0.19 omits prompt_tokens_details when 0 tokens were cached (exclude_none)
    if not g.err:
        status = "ok"
    elif getattr(g, "budget_stop", False):
        status = "not_tested_budget"
    elif E19_SERVER is not None and not E19_SERVER.alive():
        status = "server_dead"
    else:
        status = "error"
    r = {"arm": arm, "stage": stage, "pass": ps, "idx": idx, "task": E19_PROMPTS[idx]["task"], "status": status,
         "error": g.err, "prompt_len": len(ids), "prompt_sha": hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()[:16],
         "max_tokens": g.body["max_tokens"], "n_tokens": len(g.tokens), "tokens": g.tokens, "margins": g.margins,
         "top1_mismatch": g.mismatch, "finish_reason": g.finish_reason, "done_seen": g.done_seen,
         "usage_seen": g.usage is not None, "prompt_tokens": usage.get("prompt_tokens"), "cached_tokens": cached,
         "usage": usage,
         "ttft": round(g.t_first - g.t_start, 3) if g.t_first and g.t_start else None,
         "sec": round((g.t_end or time.perf_counter()) - (g.t_start or time.perf_counter()), 2),
         "t_start": round((g.t_start or 0) - NOTEBOOK_T0, 1), "conc": conc, "text": e19_decode(g.tokens)}
    r.update(extra or {})
    return r


def e19_skip_row(arm, stage, ps, idx, status, why, extra=None):
    r = {"arm": arm, "stage": stage, "pass": ps, "idx": idx, "task": E19_PROMPTS[idx]["task"], "status": status,
         "error": why, "tokens": [], "margins": [], "top1_mismatch": [], "n_tokens": 0}
    r.update(extra or {})
    return r
