# ==================================================================== CELL 5 — engine: vLLM 0.32.0 server + in-process shim (E18)
# E18 (user 2026-10-27): S1-60K (LB 34.58) with ONE change, the engine: vLLM 0.32.0 instead of 0.19.0, 20 streams instead
# of 16. Everything the solver sees is unchanged:
#   - the solver (next cell) is byte-identical to S1-60K and calls engine.add_request / step / abort_request,
#     llm.get_metrics(), SamplingParams(...) and RequestOutputKind.DELTA;
#   - this cell provides those names over vLLM 0.32's OpenAI server (streaming /v1/completions with prompt token ids and
#     return_token_ids), so the kernel keeps S1's tokenizer stack (transformers 4.57.6) and the prompts are the same ids.
# Engine flags = S1's ENGINE_CONFIG (FP8 block-128, fp8 KV, MTP 3, prefix caching, FlashInfer if -lcuda links, 4,096
# batched, mem 0.90, custom all-reduce off, language-model-only, generation config 'auto' as LLM()) with max-num-seqs 20.
# vLLM 0.32 server = the VMS launch (Experiment 14, Kaggle 2026-10-26: fp8_mtp3_s20 PASS, 452 / 423 / 281 tok/s at
# 8k / 40k / 90k). Ladder: server (language-model-only) -> server (VMS-exact) -> server (Triton attention) -> S1's own
# in-process vLLM 0.19 engine cell, verbatim (cell 2 still installs vLLM 0.19, as in S1).
import shlex, socket, queue as _queue, signal as _signal, urllib.request as _urlreq, types as _types
import shutil as _shutil

E18_FAKE = os.environ.get("E18_FAKE_SERVER")          # local CPU harness only: a fake vLLM OpenAI server script
E18_PORT = int(os.environ.get("E18_PORT", 8011))
E18_URL = f"http://127.0.0.1:{E18_PORT}"
E18_STREAM_IDLE_SEC = None      # rev 2 (audits A-M1 / B-L1): no idle timeout; a preempted or waiting request sends no
                                # bytes, as in-process vLLM would simply wait; a dead server is caught by proc.poll()
E18_SERVER = None
E18_SILENCE_SEC = float(os.environ.get("E18_SILENCE_SEC", 600))   # rev 2 (B-M3): smoke v1's longest startup log gap 60 s


def ensure_libcuda_linkable(candidates=None, workdir="/tmp/q27x_libcuda_shim"):
    """S1's cell 5 function, verbatim: make -lcuda linkable for FlashInfer's JIT and prove it with a tiny link."""
    info = {"needed": None, "real": None, "shim": None, "probe": "FAIL"}
    try:
        work = Path(workdir); work.mkdir(parents=True, exist_ok=True)
        if candidates is None:
            existing = [Path(d, "libcuda.so") for d in ("/usr/local/cuda/lib64", "/usr/local/cuda/lib64/stubs",
                                                        "/usr/lib/x86_64-linux-gnu", "/usr/lib64")]
            existing = [p for p in existing if p.exists()]
        else:
            existing = []
        if existing:
            info.update(needed=False, real=str(existing[0]))
        else:
            cands = list(candidates) if candidates is not None else []
            if candidates is None:
                for d in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64",
                          "/usr/local/cuda/compat"):
                    try:
                        cands += sorted(str(p) for p in Path(d).glob("libcuda.so.*") if p.is_file())
                    except Exception:
                        pass
                try:
                    for line in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=30).stdout.splitlines():
                        if "libcuda.so.1" in line and "=>" in line:
                            cands.append(line.split("=>")[-1].strip())
                except Exception:
                    pass
            real = next((Path(c) for c in cands if Path(c).is_file()), None)
            if real is None:
                info["reason"] = "libcuda.so.1 not found"
                return info
            link = work / "libcuda.so"
            if link.exists() or link.is_symlink():
                link.unlink()
            try:
                link.symlink_to(real.resolve())
            except OSError:
                _shutil.copyfile(real, link)
            old = os.environ.get("LIBRARY_PATH", "")
            os.environ["LIBRARY_PATH"] = f"{work}:{old}" if old else str(work)
            info.update(needed=True, real=str(real), shim=str(link), library_path=os.environ["LIBRARY_PATH"])
        cxx = _shutil.which("c++") or _shutil.which("g++")
        if not cxx:
            info["reason"] = "no C++ compiler for the link probe"
            return info
        cp = subprocess.run([cxx, "-shared", "-fPIC", "-x", "c++", "-", "-lcuda", "-o", str(work / "probe.so")],
                            input='extern "C" int q27x_probe() { return 0; }\n', text=True, capture_output=True,
                            timeout=120, env=os.environ.copy())
        info["probe"] = "PASS" if cp.returncode == 0 else "FAIL"
        if cp.returncode != 0:
            info["probe_stderr"] = (cp.stderr or "")[-1500:]
    except Exception as e:
        info["reason"] = f"{type(e).__name__}: {e}"[:500]
    return info


def e18_install():
    """The VMS installer: the wheelhouse's install_wheelhouse.py builds /tmp/vllm-venv (--no-deps, '+' restored)."""
    if E18_FAKE:
        return {"fake": True, "launch": [sys.executable, E18_FAKE], "env": dict(os.environ)}
    inst = next(iter(sorted(E18_VLLM_WHEELHOUSE.rglob("install_wheelhouse.py"))), None)
    if inst is None:
        raise FileNotFoundError(f"install_wheelhouse.py not under {E18_VLLM_WHEELHOUSE}: attach the vLLM 0.32.0 wheelhouse")
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, str(inst), "/tmp/vllm-venv"], capture_output=True, text=True, timeout=2400)
    if r.returncode != 0:
        raise RuntimeError("vLLM 0.32 wheelhouse install failed: " + (r.stdout + r.stderr)[-3000:])
    kv = dict(l.split("=", 1) for l in r.stdout.splitlines() if l.startswith(("VENV_PYTHON=", "CUDA_HOME=")))
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_PATH")}
    env.update(CUDA_HOME=kv["CUDA_HOME"], PATH=f"/tmp/vllm-venv/bin:{kv['CUDA_HOME']}/bin:{env.get('PATH', '')}",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1", VLLM_LOGGING_LEVEL="INFO",
               VLLM_CACHE_ROOT="/tmp/vllm-cache", VLLM_ALLOW_LONG_MAX_MODEL_LEN="1")
    return {"sec": round(time.perf_counter() - t0, 1), "venv_python": kv["VENV_PYTHON"],
            "launch": ["/tmp/vllm-venv/bin/vllm", "serve"], "env": env}


def e18_flags(backend, mode):
    """mode 's1': S1's ENGINE_CONFIG as vllm serve flags (incl. language_model_only, trust_remote_code False) with the
    20 seqs of cell 1. mode 'vms': the exact flag set of the Kaggle-run VMS arm fp8_mtp3_s20 (2026-10-26)."""
    f = [str(MODEL_PATH), "--tensor-parallel-size", "4", "--kv-cache-dtype", KV_CACHE_DTYPE,
         "--max-model-len", str(MAX_MODEL_LEN), "--max-num-seqs", str(MAX_NUM_SEQS),
         "--max-num-batched-tokens", str(MAX_NUM_BATCHED_TOKENS), "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
         "--disable-custom-all-reduce", "--seed", str(SEED), "--host", "127.0.0.1", "--port", str(E18_PORT),
         "--served-model-name", "q27", "--attention-backend", backend]
    if mode == "s1":
        f += ["--distributed-executor-backend", "mp", "--dtype", "auto", "--load-format", "safetensors",
              "--language-model-only"]
    else:
        f += ["--trust-remote-code"]
    f += ["--enable-prefix-caching"] if ENABLE_PREFIX_CACHING else ["--no-enable-prefix-caching"]
    if SPECULATIVE_CONFIG is not None:
        f += ["--speculative-config", json.dumps(SPECULATIVE_CONFIG)]
    return f


class E18Server:
    def __init__(self, name, cmd, env):
        self.name, self.cmd, self.env, self.proc = name, list(cmd), dict(env), None
        self.log_path = OUT / f"vllm_server_{name}.log"

    def start(self):
        self.log_f = open(self.log_path, "w", encoding="utf-8")
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.cmd, env=self.env, stdout=self.log_f, stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd=None if E18_FAKE else "/tmp")

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def log_text(self):
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def wait_ready(self, timeout, silence=E18_SILENCE_SEC):
        """rev 2: ready = /health answers AND this server's own "GPU KV cache size" line is in its log (B-L8); a launch
        whose log stops growing for `silence` seconds is a stall (B-M3)."""
        last, last_t = -1, time.perf_counter()
        while time.perf_counter() - self.t0 < timeout:
            if not self.alive():
                return False, f"exited rc={self.proc.returncode}"
            try:
                _urlreq.urlopen(E18_URL + "/health", timeout=10).read()
                if "GPU KV cache size" in self.log_text():
                    return True, f"ready in {time.perf_counter() - self.t0:.0f}s"
            except Exception:
                pass
            sz = self.log_path.stat().st_size if self.log_path.exists() else 0
            if sz != last:
                last, last_t = sz, time.perf_counter()
            elif time.perf_counter() - last_t > silence:
                return False, f"stall ({silence:.0f} s of log silence)"
            time.sleep(1 if E18_FAKE else 5)
        return False, "timeout"

    def kill(self, fast=False):
        """rev 2 (B-M2): SIGTERM to the process group, a grace for the API server, then SIGKILL to the group ALWAYS
        (engine core / workers stuck in CUDA or NCCL can outlive the API server)."""
        if self.proc is None:
            return
        grp = hasattr(os, "killpg")
        try:
            os.killpg(self.proc.pid, _signal.SIGTERM) if grp else self.proc.terminate()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=0.1 if fast else 30)
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


def _e18_kill_gpu_pids():
    """rev 2 (B-M2): kill every process still holding a GPU compute context (never this kernel)."""
    if E18_FAKE:
        return []
    killed = []
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=20).stdout
        for tok in out.split():
            if tok.isdigit() and int(tok) != os.getpid():
                try:
                    os.kill(int(tok), getattr(_signal, "SIGKILL", _signal.SIGTERM)); killed.append(int(tok))
                except Exception:
                    pass
    except Exception:
        pass
    return killed


# ---- the shim: S1's vLLM engine calls on top of the vLLM 0.32 OpenAI server
class SamplingParams:
    """Stand-in for vllm.SamplingParams: keeps the fields the solver sets (max_tokens, seed, temperature, top_p, top_k,
    detokenize, skip_special_tokens, output_kind)."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


RequestOutputKind = _types.SimpleNamespace(DELTA="delta", CUMULATIVE="cumulative", FINAL_ONLY="final_only")


class _E18Completion:
    __slots__ = ("token_ids", "finish_reason")

    def __init__(self, ids, reason):
        self.token_ids, self.finish_reason = ids, reason


class _E18Output:
    __slots__ = ("request_id", "outputs", "finished")

    def __init__(self, rid, ids, reason):
        self.request_id, self.outputs, self.finished = rid, [_E18Completion(ids, reason)], reason is not None


class E18Engine:
    """add_request starts one streaming /v1/completions per request in a thread; step() returns the token deltas that
    arrived (merged per request, order kept), raises when the server died; abort_request closes the streams (vLLM aborts
    a request whose client disconnects) and drops their undelivered deltas, as an in-process abort does."""

    def __init__(self, server):
        self.server, self.q, self.aborted, self.streams = server, _queue.Queue(), set(), {}
        self.lock = threading.Lock()
        self.stats = {"added": 0, "finished": 0, "by_reason": {}, "errors": [], "completion_tokens": 0, "aborts": 0,
                      "missing_token_ids": 0, "relaunches": []}
        self.req_log = OUT / "vllm_requests.jsonl"

    def _ensure_server(self):
        """rev 2 (B-M1): the server died -> relaunch it once with the same command, while >= 25 min remain before the
        hard stop. The streams of the dead server end as 'error' by themselves (connection errors): the solver closes
        those units as in S1 (phase 2 / requeue) on the new server. Otherwise raise, as S1's in-process engine did."""
        global E18_SERVER
        if self.server.alive():
            return
        rc = self.server.proc.returncode
        if self.stats["relaunches"] or time.perf_counter() > HARD_STOP - 25 * 60:
            raise RuntimeError(f"vLLM server exited (rc={rc}); see {self.server.log_path}")
        t0 = time.perf_counter()
        rec = {"rc": rc, "t_sec": round(t0 - NOTEBOOK_T0, 1), "old_log": str(self.server.log_path)}
        print(f"E18: vLLM server exited (rc={rc}); relaunching once", flush=True)
        self.server.kill(fast=True)
        rec["gpu_pids_killed"] = _e18_kill_gpu_pids()
        rec["gpu_release"] = _wait_gpu_release() if not E18_FAKE else {"released": True, "fake": True}
        new = E18Server(self.server.name + "_relaunch", self.server.cmd, self.server.env)
        new.start()
        ok, msg = new.wait_ready(20 * 60)
        rec["ready"] = msg
        if ok:
            E18_SERVER, self.server = new, new
            try:
                rec["smoke"] = e18_smoke()
            except Exception as e:
                ok, rec["smoke_error"] = False, f"{type(e).__name__}: {e}"[:400]
        rec["sec"] = round(time.perf_counter() - t0, 1)
        self.stats["relaunches"].append(rec)
        print("E18 relaunch:", json.dumps(rec, default=str)[:800], flush=True)
        if not ok:
            new.kill(fast=True)
            raise RuntimeError(f"vLLM server relaunch failed: {msg} {rec.get('smoke_error', '')}")

    def add_request(self, rid, prompt, params):
        self._ensure_server()
        body = {"model": "q27", "prompt": [int(x) for x in prompt["prompt_token_ids"]], "max_tokens": int(params.max_tokens),
                "temperature": float(params.temperature), "top_p": float(params.top_p), "top_k": int(params.top_k),
                "stream": True, "return_token_ids": True, "skip_special_tokens": False}
        if getattr(params, "seed", None) is not None:
            body["seed"] = int(params.seed)
        st = {"rid": rid, "resp": None, "halt": False}
        with self.lock:
            self.aborted.discard(rid)
            self.streams[rid] = st
            self.stats["added"] += 1
        threading.Thread(target=self._stream, args=(st, json.dumps(body).encode(), time.perf_counter()), daemon=True).start()

    def _stream(self, st, body, t0):
        rid, reason, err, n = st["rid"], None, None, 0
        try:
            req = _urlreq.Request(E18_URL + "/v1/completions", data=body, method="POST",
                                  headers={"Content-Type": "application/json"})
            st["resp"] = _urlreq.urlopen(req, timeout=E18_STREAM_IDLE_SEC)
            if st["halt"]:                       # rev 2 (A-L3 / B-L2): aborted while the request was being opened
                raise ConnectionAbortedError("aborted before the first chunk")
            for raw in st["resp"]:
                if st["halt"]:
                    break
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
                d = json.loads(data)
                if "error" in d or d.get("object") == "error":
                    err = ("stream error: " + json.dumps(d.get("error") or d.get("message") or d))[:300]
                    break
                for c in d.get("choices") or []:
                    ids = c.get("token_ids")
                    if ids is None:
                        if c.get("text"):
                            with self.lock:
                                self.stats["missing_token_ids"] += 1
                        ids = []
                    ids = [int(x) for x in ids]
                    fr = c.get("finish_reason")
                    if fr is not None and reason is None:
                        reason = str(fr)
                    n += len(ids)
                    if ids or fr is not None:
                        self.q.put((rid, ids, None if fr is None else str(fr)))
                if reason is not None:
                    break
        except Exception as e:
            if not st["halt"]:
                err = f"{type(e).__name__}: {e}"[:300]
        finally:                                 # rev 2 (A-L3 / B-L1): close on every exit (vLLM aborts on disconnect)
            try:
                if st["resp"] is not None:
                    st["resp"].close()
            except Exception:
                pass
        if reason is None and not st["halt"]:
            reason = "error"
            self.q.put((rid, [], "error"))
        with self.lock:
            s = self.stats
            s["finished"] += 1
            rr = "abort" if st["halt"] else reason
            s["by_reason"][rr] = s["by_reason"].get(rr, 0) + 1
            s["completion_tokens"] += n
            if err and len(s["errors"]) < 50:
                s["errors"].append(f"{rid}: {err}")
            self.streams.pop(rid, None)
            try:
                with open(self.req_log, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"rid": rid, "finish": rr, "tokens": n, "sec": round(time.perf_counter() - t0, 1),
                                        "error": err}) + "\n")
            except Exception:
                pass

    def step(self):
        self._ensure_server()
        items = []
        try:
            items.append(self.q.get(timeout=0.05))
        except _queue.Empty:
            return []
        while True:
            try:
                items.append(self.q.get_nowait())
            except _queue.Empty:
                break
        merged = {}
        with self.lock:
            aborted = set(self.aborted)
        for rid, ids, reason in items:
            if rid in aborted:
                continue
            m = merged.setdefault(rid, [[], None])
            if m[1] is None:
                m[0].extend(ids)
                m[1] = reason
        return [_E18Output(rid, ids, reason) for rid, (ids, reason) in merged.items()]

    def abort_request(self, ids):
        for rid in list(ids):
            with self.lock:
                self.aborted.add(rid)
                st = self.streams.get(rid)
                self.stats["aborts"] += 1
            if st is None:
                continue
            st["halt"] = True
            try:   # shut the socket first so a reader blocked in recv returns, then close (vLLM aborts on disconnect)
                st["resp"].fp.raw._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                if st["resp"] is not None:
                    st["resp"].close()
            except Exception:
                pass


_E18_PROM_RE = re.compile(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)")


class E18LLM:
    """llm.get_metrics(): the server's Prometheus metrics as objects with name / labels / value (what S1 reads)."""
    def get_metrics(self):
        txt = _urlreq.urlopen(E18_URL + "/metrics", timeout=2).read().decode("utf-8", "replace")   # rev 2 (B-L9)
        out = []
        for line in txt.splitlines():
            if not line or line.startswith("#"):
                continue
            m = _E18_PROM_RE.match(line)
            if not m or m.group(1).endswith(("_bucket", "_created")):
                continue
            try:
                v = float(m.group(3))
            except ValueError:
                continue
            out.append(_types.SimpleNamespace(name=m.group(1), labels=m.group(2), value=v))
        return out


def e18_smoke():
    """A real 16-token greedy decode through the shim (fp8 KV / MTP / attention kernels), as S1's _engine_smoke."""
    ids = tokenizer.encode("Reply with the digits 1 2 3 4 5 6 7 8:", add_special_tokens=False)
    eng = E18Engine(E18_SERVER)
    eng.add_request("e18-smoke", {"prompt_token_ids": ids}, SamplingParams(max_tokens=16, temperature=0.0, top_p=1.0,
                                                                         top_k=1, seed=SEED))
    toks, reason, t_end = [], None, time.perf_counter() + 120         # rev 2 (B-M3): smoke v1 took ~1 s
    while reason is None and time.perf_counter() < t_end:
        for o in eng.step():
            toks.extend(o.outputs[0].token_ids)
            if o.finished:
                reason = o.outputs[0].finish_reason
    if not toks or reason in (None, "error"):
        raise RuntimeError(f"engine decode smoke failed: {len(toks)} tokens, finish {reason}, errors {eng.stats['errors'][:2]}")
    return {"tokens": len(toks), "finish": reason, "text": tokenizer.decode(toks)[:80]}


def _wait_gpu_release(max_sec=float(os.environ.get("Q27X_GPU_WAIT_MAX", 180))):
    """S1's cell 5 function, verbatim."""
    w0 = time.perf_counter(); used = []
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


os.environ.setdefault("MAX_JOBS", "4")
CUDA_LINK = ensure_libcuda_linkable() if not E18_FAKE else {"probe": "PASS", "fake": True}
CONFIG["cuda_link"] = CUDA_LINK
print("CUDA/FlashInfer linker preflight:", json.dumps(CUDA_LINK))
ENGINE_INFO = {"engine": "vllm-0.32.0-server", "attempts": [], "speculative": None, "kv_cache_dtype": None,
               "attention_backend": None, "load_sec": None, "kv_tokens": None, "concurrency": None, "fatal": None,
               "cuda_link": CUDA_LINK, "install": None}
llm = engine = None
ENGINE_READY = False
_e18_first = "FLASHINFER" if CUDA_LINK.get("probe") == "PASS" else "TRITON_ATTN"
_E18_LADDER = [("s1", _e18_first, "s1"), ("vms", _e18_first, "vms"), ("vms_triton", "TRITON_ATTN", "vms")]
if _e18_first == "TRITON_ATTN":
    _E18_LADDER = _E18_LADDER[:2]
if not E10_SANDBOX_SELFTEST.get('ok', False):
    _E18_LADDER = []
_t_engine = time.perf_counter()
try:
    if _E18_LADDER:
        _inst = e18_install()
        ENGINE_INFO["install"] = {k: v for k, v in _inst.items() if k != "env"}
        print("vLLM 0.32 install:", json.dumps(ENGINE_INFO["install"]), flush=True)
    for _i, (_name, _backend, _mode) in enumerate(_E18_LADDER):
        _t0 = time.perf_counter()
        _att = {"name": _name, "attention_backend": _backend, "flag_mode": _mode, "flags": e18_flags(_backend, _mode)}
        print(f"ENGINE ATTEMPT {_name}: vllm serve {' '.join(_att['flags'])}", flush=True)
        E18_SERVER = E18Server(_name, _inst["launch"] + _att["flags"], _inst["env"])
        E18_SERVER.start()
        _ok, _msg = E18_SERVER.wait_ready((30 if _i == 0 else 15) * 60)       # rev 2 (B-M3): 20 -> 15 min
        _att["ready"] = _msg
        if _ok:
            try:
                _att["smoke"] = e18_smoke()
            except Exception as _se:
                _ok, _att["smoke_error"] = False, f"{type(_se).__name__}: {_se}"[:600]
        _att["sec"] = round(time.perf_counter() - _t0, 1)
        if _ok:
            _att["ok"] = True
            ENGINE_INFO["attempts"].append(_att)
            break
        _att["ok"] = False
        _att["log_tail"] = E18_SERVER.log_text()[-2500:]
        print(f"ENGINE ATTEMPT {_name} FAILED: {_msg} {_att.get('smoke_error', '')} | {_att['log_tail'][-800:]}", flush=True)
        E18_SERVER.kill()
        _att["gpu_pids_killed"] = _e18_kill_gpu_pids()                        # rev 2 (B-M2)
        _att["gpu_release"] = _wait_gpu_release() if not E18_FAKE else {"released": True, "fake": True}
        ENGINE_INFO["attempts"].append(_att)
        E18_SERVER = None
        if not _att["gpu_release"]["released"]:
            ENGINE_INFO["fatal"] = "gpu_memory_not_released_after_failed_engine"
            break
    if E18_SERVER is not None:
        engine, llm, ENGINE_READY = E18Engine(E18_SERVER), E18LLM(), True
except Exception as _e:
    ENGINE_INFO["attempts"].append({"name": "install/launch", "ok": False, "error": f"{type(_e).__name__}: {_e}"[:2000]})
    print("vLLM 0.32 engine error:", ENGINE_INFO["attempts"][-1]["error"][:1500], flush=True)
    if E18_SERVER is not None:
        E18_SERVER.kill()
        E18_SERVER = None
    _e18_kill_gpu_pids()                                                      # rev 2 (B-L3)
    _rel = _wait_gpu_release() if not E18_FAKE else {"released": True, "fake": True}
    if not _rel["released"]:
        ENGINE_INFO["fatal"] = "gpu_memory_not_released_after_failed_engine"

if ENGINE_READY:
    _log = E18_SERVER.log_text()
    _kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", _log)
    _ver = re.findall(r"version (\d+\.\d+\.\d+)", _log)
    _a = ENGINE_INFO["attempts"][-1]
    ENGINE_INFO.update(speculative=SPECULATIVE_CONFIG, kv_cache_dtype=KV_CACHE_DTYPE, attention_backend=_a["attention_backend"],
                       prefix_caching=ENABLE_PREFIX_CACHING, flag_mode=_a["flag_mode"],
                       kv_tokens=int(_kv[-1].replace(",", "")) if _kv else None, vllm=_ver[0] if _ver else None,
                       server_log=str(E18_SERVER.log_path))
    CONCURRENCY = MAX_NUM_SEQS
    if ENGINE_INFO["kv_tokens"]:
        CONCURRENCY = max(4, min(MAX_NUM_SEQS, ENGINE_INFO["kv_tokens"] // EXPECTED_LIVE_TOKENS_PER_SEQ))
    ENGINE_INFO["load_sec"] = round(time.perf_counter() - _t_engine, 1)
    ENGINE_INFO["concurrency"] = int(CONCURRENCY)

    def make_params(max_tokens, seed):
        return SamplingParams(max_tokens=int(max_tokens), seed=int(seed), detokenize=False, skip_special_tokens=False,
                              output_kind=RequestOutputKind.DELTA, **SAMPLING)

    def metrics_snapshot():
        """S1's metric filter over the server's /metrics."""
        if not ENGINE_READY:
            return []
        try:
            snap = []
            for m in llm.get_metrics():
                name = getattr(m, "name", "")
                if not any(k in name for k in ("spec_decode", "preempt", "kv_cache", "generation_tokens", "prompt_tokens",
                                               "num_requests", "prefix_cache")):
                    continue
                snap.append({"name": name, "type": "prometheus", "labels": m.labels, "value": m.value})
            return snap
        except Exception as e:
            return [{"error": repr(e)[:300]}]
    print("engine:", json.dumps(ENGINE_INFO, default=str)[:3000], flush=True)
elif E10_SANDBOX_SELFTEST.get('ok', False) and ENGINE_INFO.get("fatal") is None and os.environ.get("E18_NO_FALLBACK") != "1":
    # every vLLM 0.32 attempt failed: S1-60K's own engine cell (in-process vLLM 0.19), verbatim
    print("E18: vLLM 0.32 unavailable -> S1-60K's vLLM 0.19 engine (in-process)", flush=True)
    _e18_attempts = list(ENGINE_INFO["attempts"])
    MAX_NUM_SEQS = 16                       # rev 2 (B-L5): S1's own engine with S1's 16 sequences
    try:
        exec(compile(S1_ENGINE_SRC, "<S1 engine cell>", "exec"), globals())
        ENGINE_INFO["engine"] = "vllm-0.19.0-in-process (E18 fallback)"
    except BaseException as _fe:
        ENGINE_READY = False; engine = None; CONCURRENCY = 0
        ENGINE_INFO["fatal"] = f"fallback vLLM 0.19 failed: {type(_fe).__name__}: {_fe}"[:600]
        ENGINE_INFO["concurrency"] = 0
        print("ENGINE INVALID - keeping the identity submission:", ENGINE_INFO["fatal"], flush=True)
    ENGINE_INFO["e18_attempts"] = _e18_attempts
else:
    ENGINE_INFO["fatal"] = ENGINE_INFO["fatal"] or ("sandbox_selftest_failed" if not E10_SANDBOX_SELFTEST.get("ok", False)
                                                    else "all_engine_attempts_failed")
    engine = None; CONCURRENCY = 0
    ENGINE_INFO["concurrency"] = 0
    print("ENGINE INVALID - keeping the identity submission:", ENGINE_INFO["fatal"], flush=True)
