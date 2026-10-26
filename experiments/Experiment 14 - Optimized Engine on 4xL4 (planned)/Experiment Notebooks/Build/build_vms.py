"""VMS = Engine Matrix on vLLM 0.32.0 (user 2026-10-25: one notebook per engine; the latest vLLM). Engine only, no solver.
Twin of build_ems.py (SGLang): the same task prompts (8k and 40k bands), the same arm protocol (a correctness gate first,
then steady windows at the level the server really runs), the same report shape.

Base cells (read only) from the E15P rev 2 notebook (56a22f7f): CELL 0 mode, CELL 1 config (+ the VMS profile; staging
off: vLLM reads the Kaggle Models directly), CELL 1b E14 facts / phase record / modules, CELL 2 install (the kernel only
tokenizes: transformers 4.57.6 pins), the task-order cell. New: CELL V1 (the vLLM 0.32.0 venv from
foysalemonshanto/vllm-0-32-0-cu13-wheelhouse, -lcuda shim, server launcher, OpenAI-API measurement), CELL V2 (arms),
CELL V3 (report + bundle).

Measurement (vLLM has no SGLang decode lines): /metrics counters. tok/s = delta vllm:generation_tokens_total / time over
the window while vllm:num_requests_running holds the level; accept length = 1 + delta accepted / delta drafts
(vllm:spec_decode_*); KV usage max; preemptions delta. Streams are streaming /v1/completions with prompt token ids and
ignore_eos, so every stream decodes the whole window; closing the stream aborts the request. Gate: greedy 256-token
completions (temperature 0) of 12 prompts serial and all at once, compared with the reference's own A/A by first text
divergence (different weights = diagnostic only).

Arms (TP4 unless noted): ref = S1-60K's engine (FP8 block-128, fp8 KV, MTP 3, FLASHINFER, prefix caching, 16 seqs,
4,096 batched tokens, mem 0.90, custom all-reduce off) | FP8 target-only | MTP 2 / 4 | 20 / 24 seqs | SSM state fp16 |
FP8 Marlin (VLLM_TEST_FORCE_FP8_MARLIN=1) | prefix caching off | INT4 (RedHat W4A16 compressed-tensors) + MTP 3 at 16 and
20 | LAST: 2 x TP2 = --data-parallel-size 2 --tensor-parallel-size 2 (FP8 MTP 3, then INT4 MTP 3), 8 seqs per rank, from a
reserved slot.
rev 2 (GPT pre-GPU serving audit 2026-10-25, every claim checked in the vLLM 0.32.0 source; Experiment 14 "Bug ang fix"):
FP8 target-only with prefix caching OFF = the identity oracle (#53912: hybrid MTP + prefix caching can return empty /
punctuation output), every later arm gated against it, the reference judged from its own rows; a degenerate-output count
in every gate and window; a gate FAIL is recorded, never adopted, but still measured; the Marlin arm uses
--linear-backend (VLLM_TEST_FORCE_FP8_MARLIN is not in v0.32.0 envs.py) plus a Triton arm, with the selected kernels read
from the log; a static INT4 MTP check (config ignore list + safetensors index, #58807) before any INT4 MTP launch.
rev 2b (4-agent audit 2026-10-26): logprob-margin gate (large-margin flips, as EMS) instead of whole-text equality;
one deadline per arm (the 2 x TP2 reserve can no longer be eaten after a launch); windows stop on a dead server, accept a
KV plateau, and start only after the prefill settled; DP metrics per engine (KV max, every KV line); stream error events;
socket shutdown on stop; greedy rows saved; the lb_marlin arm dropped (SM 8.9 auto already = Marlin W8A16); the 2 x TP2
arms with one API server (custom all-reduce off: P2P measured 5.5x slower here); the INT4 static check exact-name like vLLM; FP8 prefetch.
rev 2c: the VGen stop flag renamed _halt (Thread._stop is a method join() calls on Py3.12: Kaggle CPU kit run 1);
nvidia-smi topo -m / -p2p r recorded and NCCL_DEBUG=INFO on the 2 x TP2 arms (GPT audit Gate D).
rev 2d (user 2026-10-26, "take 1 and 2"): the XL band (~85k-token prompts) on every TP4 arm; the FlashInfer
PCIe-IPC all-reduce arm fp8_mtp3_pcie (VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=1); budget 260 min.
rev 2e (two independent GPT audits of rev 2c, 2026-10-26): integer token ids (return_token_ids; logprobs.tokens are
strings); [DONE] + finish_reason required for every greedy row; a post-divergence tail check; PASS_EXACT vs BOUNDARY;
oracle_ready (no certification without the target-only oracle); healthy windows (no stream error / empty / degenerate,
held level, running share); idle gate after every window; prefill-settle required; end-to-end tok/s, prefix-cache hits;
a cross-request sentinel canary; an XL (~85k) greedy parity check; measurement_ok / correctness_ok / capacity_ok /
adoptable kept apart; the order of the audit (no-prefix and MTP width before the INT4 and SSM arms).
usage: python build_vms.py   -> ../Q27-XHIGH-VMS_ENGINE_MATRIX_VLLM.ipynb"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NBDIR = HERE.parent
E15 = NBDIR.parents[1] / "Experiment 15 - Duck 2x35k independent attempts (planned)" / "Experiment Notebooks"
SRC = E15 / "Q27-XHIGH-E15P_EVAL120_S60_S15_POETIQ_SGLANG.ipynb"
SRC_SHA = "56a22f7f"
OUT = NBDIR / "Q27-XHIGH-VMS_ENGINE_MATRIX_VLLM.ipynb"
PUSHED: set = set()
WHEELHOUSE_SLUG = "foysalemonshanto/vllm-0-32-0-cu13-wheelhouse"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rep(text, a, b, n=1):
    assert text.count(a) == n, (text.count(a), a[:100])
    return text.replace(a, b)


assert sha(SRC).startswith(SRC_SHA), sha(SRC)
S = ["".join(c["source"]) for c in json.load(open(SRC, encoding="utf-8"))["cells"] if c["cell_type"] == "code"]
assert len(S) == 11 and "CELL 1b" in S[2] and "task-order" in S[4]
G4B = ["136b0064", "898e7135", "135a2760", "b5ca7ac4", "4c3d4a41", "62593bfd", "247ef758", "bf45cf4b", "f931b4a8",
       "65b59efc", "9385bd28", "cbebaa4b"]
c1 = rep(S[1], 'os.environ.get("Q27X_PROFILE", "E15PEVAL120")', 'os.environ.get("Q27X_PROFILE", "VMS")')
c1 = rep(c1, '''    "E14SEVAL120G4B": {''', '''    # VMS (user 2026-10-25): the vLLM 0.32.0 engine-config matrix; the 12 G4B tasks only feed real ARC prompts; 260 min (rev 2d)
    "VMS":       {"source": "eval", "task_list": ''' + json.dumps(G4B) + ''',
                  "time_budget_sec": 260 * 60, "admit_frac": 0.90, "drain_frac": 0.94, "hard_margin_sec": 2 * 60,
                  "verify_margin_sec": 1 * 60, "drain_margin_sec": 8 * 60, "admit_margin_sec": 2 * 60, "vms": True},   # rev 2d: 260 min
    "E14SEVAL120G4B": {''')
c1 += '''

# ---- VMS (user 2026-10-25): the vLLM 0.32.0 engine matrix. vLLM reads the Kaggle Models directly (no /dev/shm staging:
# two checkpoints, FP8 + INT4, would not both fit); the rerun of this notebook only writes the cell-0 fallback.
VMS_ON = bool(P.get("vms")) and not RERUN
os.environ["E14_LOADPATH"] = "0"
VLLM_WHEELHOUSE = Path(os.environ.get("VMS_WHEELHOUSE", "/kaggle/input/datasets/''' + WHEELHOUSE_SLUG + '''"))
CONFIG["vms"] = {"on": VMS_ON, "wheelhouse": str(VLLM_WHEELHOUSE)}
print("VMS:", json.dumps(CONFIG["vms"]))
'''
c1b = rep(S[2], '"variant": "E15P"', '"variant": "VMS"')

V1 = r'''# ==================================================================== CELL V1 — vLLM 0.32.0 venv, launcher, measurement
import shlex, socket, statistics, threading
VMS = {"install": None, "arms": {}, "launches": [], "order": []}
E14_RESULTS["vms"] = VMS
VMS_FAKE = os.environ.get("VMS_FAKE_SERVER")          # local CPU harness only: a fake vLLM OpenAI server script
VMS_PORT = int(os.environ.get("VMS_PORT", 8011))
VMS_URL = f"http://127.0.0.1:{VMS_PORT}"
VMS_SERVER = None
VMS_DIR = OUT / "vms"
VMS_DIR.mkdir(parents=True, exist_ok=True)


def vms_install():
    """The dataset's installer (restores the '+' Kaggle drops, venv, --no-deps), then a -lcuda shim for FlashInfer JIT."""
    if VMS_FAKE:
        return {"fake": True, "launch": [sys.executable, VMS_FAKE], "env": dict(os.environ)}
    inst = next(iter(sorted(VLLM_WHEELHOUSE.rglob("install_wheelhouse.py"))), None)
    if inst is None:
        raise FileNotFoundError(f"install_wheelhouse.py not under {VLLM_WHEELHOUSE}: attach the vLLM 0.32.0 wheelhouse")
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, str(inst), "/tmp/vllm-venv"], capture_output=True, text=True, timeout=2400)
    if r.returncode != 0:
        raise RuntimeError("vLLM wheelhouse install failed: " + (r.stdout + r.stderr)[-3000:])
    kv = dict(l.split("=", 1) for l in r.stdout.splitlines() if l.startswith(("VENV_PYTHON=", "CUDA_HOME=")))
    shim = Path("/tmp/vms_libcuda")
    shim.mkdir(exist_ok=True)
    for d in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/compat", "/usr/lib64"):
        c = sorted(Path(d).glob("libcuda.so.1")) if Path(d).is_dir() else []
        if c and not (shim / "libcuda.so").exists():
            (shim / "libcuda.so").symlink_to(c[0])
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_PATH")}
    env.update(CUDA_HOME=kv["CUDA_HOME"], PATH=f"/tmp/vllm-venv/bin:{kv['CUDA_HOME']}/bin:{env.get('PATH', '')}",
               LIBRARY_PATH=f"{shim}:{env.get('LIBRARY_PATH', '')}", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_NO_USAGE_STATS="1", VLLM_LOGGING_LEVEL="INFO", VLLM_CACHE_ROOT="/tmp/vllm-cache",
               VLLM_ALLOW_LONG_MAX_MODEL_LEN="1")
    return {"sec": round(time.perf_counter() - t0, 1), "venv_python": kv["VENV_PYTHON"],
            "launch": ["/tmp/vllm-venv/bin/vllm", "serve"], "env": env}


def vms_flags(model, *, tp=4, dp=1, seqs=16, spec=3, kv="fp8_e4m3", backend="FLASHINFER", prefix=True, mem=0.90,
              batched=4096, max_len=106496, extra=None, car_off=True):
    f = [str(model), "--tensor-parallel-size", str(tp), "--kv-cache-dtype", kv, "--max-model-len", str(max_len),
         "--max-num-seqs", str(seqs), "--max-num-batched-tokens", str(batched), "--gpu-memory-utilization", str(mem)]
    f += ["--disable-custom-all-reduce"] if car_off else []
    f += ["--seed", str(SEED), "--host", "127.0.0.1", "--port", str(VMS_PORT),
          "--served-model-name", "q27", "--attention-backend", backend, "--trust-remote-code"]
    if dp > 1:
        f += ["--data-parallel-size", str(dp)]
    f += ["--enable-prefix-caching"] if prefix else ["--no-enable-prefix-caching"]
    if spec:
        f += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": int(spec)})]
    for k, v in (extra or {}).items():
        f += [k] + ([] if v is None else [str(v)])
    return f


class VServer:
    def __init__(self, name, flags, env):
        self.name, self.log_path = name, VMS_DIR / f"vllm_server_{name}.log"
        self.cmd, self.env, self.proc = list(VMS_INST["launch"]) + list(flags), dict(env), None

    def start(self):
        self.log_f = open(self.log_path, "w", encoding="utf-8")
        self.t0 = time.perf_counter()
        self.proc = subprocess.Popen(self.cmd, env=self.env, stdout=self.log_f, stderr=subprocess.STDOUT,
                                     start_new_session=True, cwd=None if VMS_FAKE else "/tmp")

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def log_text(self):
        try:
            return self.log_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return ""

    def wait_ready(self, timeout):
        while time.perf_counter() - self.t0 < timeout:
            if not self.alive():
                return False, f"exited rc={self.proc.returncode}"
            try:
                _urlreq.urlopen(VMS_URL + "/health", timeout=10).read()
                return True, f"ready in {time.perf_counter() - self.t0:.0f}s"
            except Exception:
                time.sleep(3)
        return False, "timeout"

    def kill(self):
        if self.proc is None:
            return
        for sig, wait in ((_signal.SIGTERM, 30), (getattr(_signal, "SIGKILL", _signal.SIGTERM), 60)):   # rev 2b: TERM first
            try:
                if hasattr(os, "killpg"):
                    os.killpg(self.proc.pid, sig)
                else:
                    self.proc.kill()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=wait)
                return
            except Exception:
                pass


def vms_prom():
    txt = _urlreq.urlopen(VMS_URL + "/metrics", timeout=15).read().decode("utf-8", "replace")
    out = {}
    for line in txt.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)", line)
        if m:
            try:
                v = float(m.group(3))
            except ValueError:
                continue
            out[m.group(1)] = out.get(m.group(1), 0.0) + v
            out[m.group(1) + "@max"] = max(out.get(m.group(1) + "@max", v), v)     # rev 2b: per-engine max (DP)
    return out


def _met(p, *names):
    return sum(p.get(n, 0.0) for n in names)


class VGen(threading.Thread):
    """One streaming /v1/completions request with prompt token ids; stop() closes the stream (vLLM aborts it).
    rev 2e (GPT audits E1, HIGH-2): integer token ids from return_token_ids (completion/protocol.py:171-180,
    serving.py:417-421; logprobs.tokens are decoded strings), the terminal [DONE] and the finish_reason recorded; a request
    we did not stop that lacks either is an error, never a valid row."""
    def __init__(self, ids, max_new, temperature=1.0, ignore_eos=True, seed=None, logprobs=0):
        super().__init__(daemon=True)
        self.body = {"model": "q27", "prompt": list(ids), "max_tokens": int(max_new), "temperature": temperature,
                     "top_p": 0.95 if temperature else 1.0, "stream": True,
                     "ignore_eos": ignore_eos, "return_token_ids": True}
        if temperature:
            self.body["top_k"] = 20          # rev 2g-b: greedy sends no top_k (temperature 0 = argmax)
        if seed is not None:
            self.body["seed"] = seed
        if logprobs:
            self.body["logprobs"] = int(logprobs)      # rev 2b: the gate compares tokens and top-1 / top-2 margins
        self.text, self.err, self._resp, self._halt = [], None, None, False   # NOT _stop: Thread._stop() is a method join() calls (Py3.12)
        self.tokens, self.margins = [], []
        self.done_seen, self.finish_reason, self.first_token_at = False, None, None

    def run(self):
        try:
            req = _urlreq.Request(VMS_URL + "/v1/completions", data=json.dumps(self.body).encode(), method="POST",
                                  headers={"Content-Type": "application/json"})
            self._resp = _urlreq.urlopen(req, timeout=3600)
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
                if "error" in d or d.get("object") == "error":     # rev 2b: an in-stream error event (serving.py:497-503)
                    self.err = ("stream error: " + json.dumps(d.get("error") or d.get("message") or d))[:200]
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
                        if ids_ and self.first_token_at is None:
                            self.first_token_at = time.perf_counter()
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
                self.err = repr(e)[:200]
        if not self._halt and not self.err and not (self.done_seen and self.finish_reason):
            self.err = "INCOMPLETE_SSE: [DONE] seen %s, finish_reason %s" % (self.done_seen, self.finish_reason)

    def stop(self):
        self._halt = True
        try:      # rev 2b: shut the socket down first, so a reader blocked in recv returns (close() alone can wait on it)
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
                "done_seen": self.done_seen, "finish_reason": self.finish_reason}


def _vms_dead():
    return VMS_SERVER is not None and not VMS_SERVER.alive()


def vms_idle(wait=60.0):
    """rev 2e (GPT audit E6): 0 running and 0 waiting in 3 consecutive samples (2 s apart) within `wait`."""
    end, hits = time.perf_counter() + wait, 0
    while time.perf_counter() < end:
        try:
            p = vms_prom()
            hits = hits + 1 if (_met(p, "vllm:num_requests_running") == 0 and _met(p, "vllm:num_requests_waiting") == 0) else 0
        except Exception:
            hits = 0
        if hits >= 3:
            return True
        time.sleep(2)
    return False


def vms_window(prompts, level, secs, tag, deadline=None):
    """`level` streams; wait until vllm:num_requests_running reaches level, a stable level - 1, or a KV plateau (requests
    waiting while running stays flat for 30 s); then wait for the prefill to settle; then measure the counters.
    rev 2b: every wait ends at `deadline` (the arm's own: the 2 x TP2 reserve stays intact) and on a dead server.
    rev 2e (GPT audits E5-E7, HIGH-5, E9): the prefill must settle (3 equal prompt-token readings) or the window is
    PREFILL_NOT_SETTLED; the streams are joined and the server must be idle afterwards (else DIRTY_SERVER_AFTER_WINDOW);
    every started stream is checked for empty / degenerate text; steady decode tok/s and end-to-end output tok/s (from
    submission) are both kept; prefix-cache hits / queries / cached prompt tokens are recorded."""
    end = min(HARD_STOP - 120, deadline or HARD_STOP)
    if end - time.perf_counter() < secs + 60:
        return {"skipped": "NOT_TESTED_BUDGET: deadline"}
    # rev 2g: max_tokens up to 32k inside max-model-len 106,496 (a stream that finished inside the window lowered
    # running_mean: a false reject)
    gens = [VGen(prompts[i % len(prompts)], max(1024, min(32768, 106496 - len(prompts[i % len(prompts)]) - 512)))
            for i in range(level)]
    out = {"tag": tag, "level": level}
    p_sub = p0 = p1 = None
    t_sub = t0 = t1 = None
    try:
        try:
            p_sub = vms_prom()
        except Exception:
            p_sub = {}
        t_sub = time.perf_counter()
        for g in gens:
            g.start()
        t_w, got, best, hist, plateau = time.perf_counter(), None, 0, [], False
        while time.perf_counter() < min(t_w + 900, end - secs - 30):
            if _vms_dead():
                out["error"] = "the server died while the streams were admitted"
                return out
            try:
                p = vms_prom()
                run, wait = _met(p, "vllm:num_requests_running"), _met(p, "vllm:num_requests_waiting")
            except Exception:
                run, wait = 0, 0
            best = max(best, run)
            hist.append((time.perf_counter(), run))
            if run >= level:
                got = level
                break
            alive = sum(1 for g in gens if g.is_alive())
            if alive < level - 1 or (alive < level and run >= alive):      # rev 2e-e / 2g: level (or level - 1) unreachable
                out["streams_ended_before_level"] = level - alive
                break
            flat = [r for t_, r in hist if t_ >= hist[-1][0] - 30]
            if wait > 0 and run >= 1 and hist[-1][0] - hist[0][0] >= 30 and len(set(flat)) == 1:
                got, plateau = int(run), True        # KV-bound: measure what the server really holds
                break
            time.sleep(2)
        if got is None and best >= level - 1:
            got = int(best)
        out.update(measured_level=got, plateau=plateau, capacity_limited=plateau, wait_sec=round(time.perf_counter() - t_w, 1))
        if got is None:
            out["error"] = f"the server never ran {level} requests (max {best})"
            return out
        # a request counts as running once its first prefill chunk is scheduled: start only when nothing waits (or the
        # plateau) and 3 consecutive prompt-token readings are equal. Up to 300 s: a no-prefix-cache arm prefills
        # 16 x 40k = 640k tokens (minutes); a window that still did not settle is measured but labelled mixed_load
        # (excluded from measurement_ok), never thrown away
        t_s, last, stable, settled = time.perf_counter(), None, 0, False
        while time.perf_counter() - t_s < 300 and time.perf_counter() < end - secs - 15:
            try:
                p = vms_prom()
                pt, wait = _met(p, "vllm:prompt_tokens_total"), _met(p, "vllm:num_requests_waiting")
            except Exception:
                break
            stable = stable + 1 if (last is not None and pt == last and (wait == 0 or plateau)) else 0
            if stable >= 2:
                settled = True
                break
            last = pt
            time.sleep(2)
        out.update(settle_sec=round(time.perf_counter() - t_s, 1), prefill_settled=settled, mixed_load=not settled)
        p0, t0 = vms_prom(), time.perf_counter()
        runs, kvs, runs_mx = [], [], []
        while time.perf_counter() - t0 < secs:
            time.sleep(2)
            if _vms_dead():
                out["error"] = "the server died inside the window"
                return out
            try:
                p = vms_prom()
                runs.append(_met(p, "vllm:num_requests_running"))
                runs_mx.append(p.get("vllm:num_requests_running@max", 0.0))      # rev 2g-b: per engine (DP skew)
                kvs.append(p.get("vllm:kv_cache_usage_perc@max", 0.0))      # rev 2b: per engine, never the DP sum
            except Exception:
                pass
        p1, t1 = vms_prom(), time.perf_counter()
        out["_runs"], out["_kvs"], out["_runs_mx"] = runs, kvs, runs_mx
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"[:300]
        return out
    finally:
        for g in gens:
            g.stop()
        for g in gens:
            if g.ident is not None:
                g.join(timeout=5)
        out["threads_not_joined"] = sum(1 for g in gens if g.is_alive())
        out["idle_after"] = (not _vms_dead()) and vms_idle(60)
        if out["threads_not_joined"] or not out["idle_after"]:
            out["error"] = out.get("error") or "DIRTY_SERVER_AFTER_WINDOW"
        started = [g for g in gens if g.first_token_at is not None]
        texts = ["".join(g.text) for g in started]
        out.update(finished_empty=sum(1 for g in gens if g.done_seen and not g.tokens),      # rev 2g-c (agent 3 B2)
                   no_first_token=len(gens) - len(started), empty_outputs=sum(1 for x in texts if not x.strip()),
                   degenerate=vms_degenerate(texts), stream_errors=sum(1 for g in gens if g.err),
                   stream_error_samples=[g.err for g in gens if g.err][:3])
    runs, kvs = out.pop("_runs", []), out.pop("_kvs", [])
    out["_runs_mx"] = out.get("_runs_mx") or []
    dt = t1 - t0
    gen = _met(p1, "vllm:generation_tokens_total") - _met(p0, "vllm:generation_tokens_total")
    acc = _met(p1, "vllm:spec_decode_num_accepted_tokens_total") - _met(p0, "vllm:spec_decode_num_accepted_tokens_total")
    drf = _met(p1, "vllm:spec_decode_num_drafts_total") - _met(p0, "vllm:spec_decode_num_drafts_total")
    gen_all = _met(p1, "vllm:generation_tokens_total") - _met(p_sub, "vllm:generation_tokens_total")
    out.update(window_sec=round(dt, 1), tok_s=round(gen / dt, 1) if dt > 0 else None,
               end_to_end_tok_s=round(gen_all / (t1 - t_sub), 1) if t1 and t_sub and t1 > t_sub else None,
               accept=round(1 + acc / drf, 2) if drf > 0 else None, running_mean=round(statistics.mean(runs), 2) if runs else None,
               running_engine_max_mean=round(statistics.mean(out.get("_runs_mx") or [0]), 2),
               kv_max=round(max(kvs), 3) if kvs else None,
               preemptions=_met(p1, "vllm:num_preemptions_total") - _met(p0, "vllm:num_preemptions_total"),
               prefix_cache_hits=_met(p1, "vllm:prefix_cache_hits_total") - _met(p_sub, "vllm:prefix_cache_hits_total"),
               prefix_cache_queries=_met(p1, "vllm:prefix_cache_queries_total") - _met(p_sub, "vllm:prefix_cache_queries_total"),
               prompt_tokens_cached=_met(p1, "vllm:prompt_tokens_cached_total") - _met(p_sub, "vllm:prompt_tokens_cached_total"))
    return out


def vms_window_healthy(w, level):
    """rev 2e (2nd audit HIGH-3): speed is evidence only from a clean window that held its level."""
    return bool(w) and not w.get("error") and not w.get("skipped") and (w.get("tok_s") or 0) > 0 \
        and w.get("prefill_settled") is True and w.get("stream_errors", 0) == 0 and w.get("empty_outputs", 0) == 0 and w.get("degenerate", 0) == 0 \
        and (w.get("measured_level") or 0) >= level and (w.get("running_mean") or 0) >= 0.80 * level \
        and w.get("no_first_token", level) == 0 and w.get("threads_not_joined", 1) == 0 and w.get("idle_after") is True \
        and (w.get("preemptions") or 0) == 0 and w.get("finished_empty", 0) == 0


def vms_greedy(prompts, n_tok=256, concurrent=False, deadline=None):
    """Greedy rows {text, token ids, top-1 / top-2 margins, error, [DONE], finish_reason}; rev 2b: one deadline for the
    whole pass; a row still running at the deadline is stopped and marked."""
    end = min(HARD_STOP - 120, deadline or HARD_STOP)
    gens = [VGen(p, n_tok, temperature=0.0, ignore_eos=False, logprobs=2) for p in prompts]
    if concurrent:
        for g in gens:
            g.start()
        for g in gens:
            g.join(max(0.0, min(900, end - time.perf_counter())))
    else:
        for g in gens:
            if end - time.perf_counter() < 30:
                g.err = "skipped: arm deadline"
                continue
            g.start()
            g.join(max(0.0, min(900, end - time.perf_counter())))
    for g in gens:
        if g.is_alive():
            g.stop()
            g.err = g.err or "stopped: arm deadline"
    return [g.row() for g in gens]


def vms_degenerate(texts):
    """rev 2 (GPT audit V1, vLLM #53912: silent empty / repeated-punctuation output with a normal finish): empty text, a
    run of 32+ identical characters, or >= 200 characters with < 8 distinct characters or < 10 % letters / digits.
    (A distinct-character RATIO would flag every long normal text: 8k characters of prose use ~100 distinct ones.)"""
    return sum(1 for x in texts if vms_text_bad(x))


def vms_grid_like(s):
    """rev 2g (GPT R2F-06): an ARC grid by structure ("0 0 1\n0 0 0", "[[0, 1], ...]"): only digits / brackets / commas /
    whitespace and every row <= 30 cells (the ARC maximum). "0" * 250 or "0 0 0 ..." x 200 is not a grid."""
    s = (s or "").strip()
    return bool(re.fullmatch(r"[0-9,\[\]\s]+", s)) and len(re.findall(r"\d", s)) >= 2 \
        and all(len(re.findall(r"\d", r_)) <= 30 for r_ in re.split(r"\]|\n", s))


def vms_text_bad(x):
    """rev 2e-c: one text is degenerate when empty, or (not grid-like) a 32-character run of one character, < 8 distinct
    characters over 200, or < 10 % letters / digits. A grid is never degenerate (it repeats by nature)."""
    s = (x or "").strip()
    if not s:
        return True
    if vms_grid_like(s):
        return False
    if re.search(r"(.)\1{31,}", s):
        return True
    return len(s) >= 200 and (len(set(s)) < 8 or sum(ch.isalnum() for ch in s) / len(s) < 0.10)


def vms_kernels(txt):
    """rev 2 (GPT audit V6): the GEMM / attention kernels vLLM says it selected (never assume the default)."""
    seen = []
    for ln in txt.splitlines():
        m = re.search(r"((?:Selected|Using|Chose)\b.{0,60}?(?:kernel|backend).{0,80}|--linear-backend=.{0,160})", ln, re.I)
        if m and m.group(1).strip() not in seen:
            seen.append(m.group(1).strip())
    return seen[:30]


def vms_tail_bad(ids):
    """rev 2e (GPT audits E2 / HIGH-10), rev 2e-c: the continuation after the first divergence, decoded, is degenerate by
    the shared text rule (a grid is not; empty, repeated punctuation, near-alphabet-free text is)."""
    ids = list(ids or [])
    if len(ids) < 16:
        return False          # too short to judge
    try:
        return vms_text_bad(tokenizer.decode(ids, skip_special_tokens=True))
    except Exception:
        return False


def vms_diverge(a, b):
    """rev 2b (agent 3 F10): rows compared by integer token id (rev 2e) with the EMS compare_passes rule. A large-margin
    flip = a divergence where either run was confident (top-1 minus top-2 > 1 nat); a divergence without both margins is
    missing evidence, never "no flip"; an errored, incomplete (no [DONE] / finish_reason) or empty row is an error."""
    expected = max(len(a), len(b))
    per = []
    for i in range(expected):
        ra = a[i] if i < len(a) else {"error": "missing row"}
        rb = b[i] if i < len(b) else {"error": "missing row"}
        ia, ib = list(ra.get("tokens") or []), list(rb.get("tokens") or [])
        inc = [x for x in (ra, rb) if "done_seen" in x and not (x.get("done_seen") and x.get("finish_reason"))]
        if ra.get("error") or rb.get("error") or inc or not ia or not ib:
            per.append({"error": ra.get("error") or rb.get("error") or ("incomplete stream" if inc else "no tokens")})
            continue
        if ia == ib:
            per.append({"identical": True})
            continue
        k = next((j for j, (u, v) in enumerate(zip(ia, ib)) if u != v), min(len(ia), len(ib)))
        ma = (ra.get("margins") or [None] * (k + 1))[k] if k < len(ra.get("margins") or []) else None
        mb = (rb.get("margins") or [None] * (k + 1))[k] if k < len(rb.get("margins") or []) else None
        per.append({"identical": False, "first_divergence": k, "margin_a": ma, "margin_b": mb,
                    "missing_margin": ma is None or mb is None,
                    "large_margin_flip": ma is not None and mb is not None and max(ma, mb) > 1.0,
                    "tail_bad": vms_tail_bad(ib[k:])})
    ok = [x for x in per if "identical" in x]
    div = [x for x in ok if not x["identical"]]
    return {"expected": expected, "compared": len(ok), "errors": expected - len(ok),
            "identical": sum(1 for x in ok if x["identical"]), "differ": len(div),
            "large_margin_flips": sum(1 for x in div if x["large_margin_flip"]),
            "missing_margins": sum(1 for x in div if x["missing_margin"]),
            "tail_bad": sum(1 for x in div if x.get("tail_bad")),
            "first_divergence": [x.get("first_divergence") for x in per]}


VMS_INST = None
if VMS_ON:
    try:
        VMS_INST = vms_install()
        VMS["install"] = {k: v for k, v in VMS_INST.items() if k != "env"}
        print("VMS install:", json.dumps(VMS["install"])[:600], flush=True)
    except Exception as _e:
        VMS["install"] = {"error": f"{type(_e).__name__}: {_e}"[:2000]}
        print("VMS install FAILED:", VMS["install"]["error"][:1500], flush=True)
e14_save()
'''

V2 = r'''# ==================================================================== CELL V2 — the vLLM arms
VMS_INT4 = SGL_MODEL_PATH          # RedHat INT4 (compressed-tensors) Kaggle Model; MODEL_PATH = the FP8 repack
VMS_RESERVE = 0.0


def vms_launch(name, flags, env_extra=None, force=False):
    global VMS_SERVER
    rec = {"arm": name, "flags": list(flags), "env": env_extra or {}}
    if HARD_STOP - time.perf_counter() - (0 if force else VMS_RESERVE) - 8 * 60 < 12 * 60:
        rec["skipped"] = "NOT_TESTED_BUDGET: reserved 2 x TP2 slot / hard stop"
        VMS["launches"].append(rec)
        return False, rec
    if VMS_SERVER is not None:
        VMS_SERVER.kill()
        VMS_SERVER = None
    if not VMS_FAKE:
        rec["gpu_free"] = _e11_gpu_free()
    left = HARD_STOP - time.perf_counter() - (0 if force else VMS_RESERVE) - 8 * 60     # rev 2b: after the kill + GPU wait
    if left < 10 * 60:
        rec["skipped"] = "NOT_TESTED_BUDGET: reserved 2 x TP2 slot / hard stop (after the GPU wait)"
        VMS["launches"].append(rec)
        return False, rec
    env = dict(VMS_INST["env"])
    env.update(env_extra or {})
    srv = VServer(name, flags, env)
    srv.start()
    try:          # rev 2g: a crash while this server starts never leaves it holding the GPUs
        ok, why = srv.wait_ready(min(30 * 60, left))
    except BaseException:
        try:
            srv.kill()
        except Exception:
            pass
        raise
    rec.update(ok=ok, why=why, sec=round(time.perf_counter() - srv.t0, 1))
    txt = srv.log_text()
    m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", txt)
    rec["kv_tokens_all"] = [int(x.replace(",", "")) for x in m]          # rev 2b: DP = one line per engine
    rec["kv_tokens"] = min(rec["kv_tokens_all"]) if m else None
    if ok and not m:      # rev 2g-b: a /health 200 without this server's own KV line = another listener on the port
        ok, why = False, "ready, but no 'GPU KV cache size' line in this server's log (another process on the port?)"
    m = re.findall(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", txt)
    rec["max_concurrency_at_max_len"] = float(m[-1]) if m else None
    rec["kernels"] = vms_kernels(txt)
    if ok:
        VMS_SERVER = srv
    else:
        rec["log_tail"] = txt[-4000:]
        srv.kill()
    VMS["launches"].append(rec)
    e14_save()
    print(f"VMS launch {name}: {why} | KV {rec.get('kv_tokens')}", flush=True)
    return ok, rec


VMS_REF = {}


def vms_int4_mtp_static(model):
    """rev 2 (GPT audit V3, vLLM #58807: a W4A16 checkpoint with dense BF16 MTP crashed at load because MTP was not in
    quantization_config.ignore, so the predictor was built quantized). vLLM 0.32.0 builds the predictor under prefix
    'mtp' (qwen3_5_mtp.py:233-235) with the target's quant config (:93-105), so every dense mtp.* weight must be
    exempted by the ignore list. CPU only: config.json + the safetensors index, no weights read."""
    out = {"model": str(model)}
    try:
        cfg = json.loads((Path(model) / "config.json").read_text(encoding="utf-8"))
        q = cfg.get("quantization_config") or (cfg.get("text_config") or {}).get("quantization_config") or {}
        ign = [str(x) for x in (q.get("ignore") or [])]

        def ignored(name):
            # rev 2b: vLLM 0.32.0 matches the fused name by exact name or re.match (config_utils.py:344-362, 392-412)
            return any(re.match(x[3:], name) if x.startswith("re:") else name == x for x in ign)
        wm = json.loads((Path(model) / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
        mtp = [k for k in wm if k.startswith("mtp.")]
        packed = [k for k in mtp if any(s in k for s in ("weight_packed", "weight_scale", "weight_zero_point"))]
        mods = sorted({k.rsplit(".", 1)[0] for k in mtp if k.endswith(".weight") and ("proj" in k or k.startswith("mtp.fc"))})
        _fuse = {"q_proj": "qkv_proj", "k_proj": "qkv_proj", "v_proj": "qkv_proj", "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj"}
        mods = sorted(set(mods) | {m.rsplit(".", 1)[0] + "." + _fuse[m.rsplit(".", 1)[1]] for m in mods if m.rsplit(".", 1)[1] in _fuse})
        not_ign = [m for m in mods if not ignored(m)]
        out.update(format=q.get("format"), method=q.get("quant_method"), mtp_keys=len(mtp), packed=len(packed),
                   dense_linear_modules=len(mods), not_ignored=not_ign[:10],
                   mtp_ignore_entries=[x for x in ign if "mtp" in x])
        # rev 2g (GPT R4): a packed MTP module that the ignore list exempts would be built dense (wrong shapes)
        packed_mods = sorted({k.rsplit(".", 1)[0] for k in packed})
        packed_ign = [m for m in packed_mods if ignored(m)]
        out.update(packed_modules=len(packed_mods), packed_but_ignored=packed_ign[:10])
        out["ok"] = bool(mtp) and not not_ign and not packed_ign        # rev 2b: un-ignored dense linears = #58807
        out["why"] = "MTP dense and exempted by the ignore list" if out["ok"] and not packed else (
            "MTP stored quantized" if out["ok"] else "dense MTP linears missing from the ignore list (#58807 crash)")
    except Exception as e:
        out.update(ok=False, why=f"static check failed: {type(e).__name__}: {e}"[:300])
    VMS["int4_mtp_static"] = out
    e14_save()
    print("VMS INT4 MTP static check:", json.dumps(out)[:600], flush=True)
    return out["ok"]


VMS_ROWS = {}          # rev 2: each A/A arm's own greedy rows (the production reference and the oracle)


def _texts(rows):
    return [r.get("text") or "" for r in rows]


def vms_save_rows(tag, rows):
    """rev 2b (agent 3 F10): every greedy row kept, so any verdict can be re-judged after the run."""
    try:
        with open(VMS_DIR / "vms_greedy_rows.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"tag": tag, "rows": rows}) + "\n")
    except Exception:
        pass


def vms_aa(name, deadline=None):
    rows = {"serial_a": vms_greedy(_S[:6], deadline=deadline), "serial_b": vms_greedy(_S[:6], deadline=deadline),
            "conc_a": vms_greedy(_S, concurrent=True, deadline=deadline), "conc_b": vms_greedy(_S, concurrent=True, deadline=deadline)}
    if _X:      # rev 2e (GPT audits MEDIUM-7 / E11): a greedy parity reference at ~85k context too
        rows["xl_a"] = vms_greedy(_X[:4], concurrent=True, deadline=deadline)
        rows["xl_b"] = vms_greedy(_X[:4], concurrent=True, deadline=deadline)
    vms_save_rows(name + "_aa", rows)
    s, c = vms_diverge(rows["serial_a"], rows["serial_b"]), vms_diverge(rows["conc_a"], rows["conc_b"])
    complete = all(x["expected"] and x["compared"] == x["expected"] and not x["missing_margins"] for x in (s, c)) \
        and s["expected"] >= min(6, len(_S)) and c["expected"] >= len(_S)       # rev 2g (GPT R5): the planned rows
    return rows, {"serial": s, "conc": c, "complete": complete,
                  "xl": vms_diverge(rows["xl_a"], rows["xl_b"]) if _X else None,
                  "degenerate": vms_degenerate(_texts(rows["serial_a"] + rows["serial_b"] + rows["conc_a"] + rows["conc_b"]))}


def vms_judge(s, c, base, gate=True):
    """The EMS parity rule on integer-token rows against `base` (the oracle once it exists): large-margin flips may not
    exceed base's own A/A flips (serial and all at once); no more degenerate outputs than base's A/A; rev 2e: no more
    degenerated continuations after a divergence than base's A/A; INCOMPLETE when any compared row errored, was
    incomplete or a divergence has no margins (fail closed). gate_class: PASS_EXACT = every sequence identical,
    BOUNDARY = a PASS with only near-tie divergences (adoptable under the documented tolerance, not exact: confirm on ARC
    tasks; certified_exact = adoptable and PASS_EXACT)."""
    cs, aa_s = vms_diverge(base["serial_a"], s), vms_diverge(base["serial_a"], base["serial_b"])
    cc, aa_c = vms_diverge(base["conc_a"], c), vms_diverge(base["conc_a"], base["conc_b"])
    dg, dg_aa = vms_degenerate(_texts(list(s) + list(c))), vms_degenerate(_texts(base["serial_a"] + base["conc_a"]))
    d = {"serial": cs, "conc": cc, "aa_serial": aa_s, "aa_conc": aa_c, "degenerate": dg, "aa_degenerate": dg_aa}
    if not gate:
        return "diagnostic", d
    if any(x["errors"] or x["missing_margins"] or not x["compared"] for x in (cs, aa_s, cc, aa_c)) \
            or cs["expected"] < min(6, len(_S)) or cc["expected"] < len(_S):       # rev 2g (GPT R5): the planned rows
        d["gate_class"] = "INCOMPLETE"
        return "INCOMPLETE", d
    passed = cs["large_margin_flips"] <= aa_s["large_margin_flips"] and cc["large_margin_flips"] <= aa_c["large_margin_flips"] \
        and dg <= dg_aa and cs["tail_bad"] + cc["tail_bad"] <= aa_s["tail_bad"] + aa_c["tail_bad"]
    d["gate_class"] = ("PASS_EXACT" if cs["identical"] == cs["compared"] and cc["identical"] == cc["compared"] else "BOUNDARY") \
        if passed else "FAIL"
    return ("PASS" if passed else "FAIL"), d


def vms_judge_xl(x, base):
    """rev 2e: the ~85k greedy rows against the oracle's own XL A/A (flips and tails decide, as above)."""
    if not base.get("xl_a") or not x:
        return None
    cx, aa = vms_diverge(base["xl_a"], x), vms_diverge(base["xl_a"], base["xl_b"])
    if any(r["errors"] or r["missing_margins"] or not r["compared"] for r in (cx, aa)):
        return {"verdict": "INCOMPLETE", "cmp": cx, "aa": aa}
    ok = cx["large_margin_flips"] <= aa["large_margin_flips"] and cx["tail_bad"] <= aa["tail_bad"]
    return {"verdict": "PASS" if ok else "FAIL", "cmp": cx, "aa": aa}


VMS_CANARY = {}


def vms_canary(level, deadline=None):
    """rev 2e (GPT audit MEDIUM-9, SGLang #36548 cross-request bleed): `level` concurrent greedy requests, each asked to
    repeat its own random sentinel (thinking off); a sentinel found in ANOTHER request's output is a leak."""
    import random as _rnd
    rng = _rnd.Random(SEED + level)
    sent = ["Z%04d-%s" % (i, "".join(rng.choice("QWXJVK") for _ in range(6))) for i in range(level)]
    ids = [tokenizer.encode(tokenizer.apply_chat_template([{"role": "user", "content":
           f"Repeat this code exactly once and nothing else: {s_}"}], tokenize=False, add_generation_prompt=True,
           enable_thinking=False), add_special_tokens=False) for s_ in sent]
    rows = vms_greedy(ids, n_tok=32, concurrent=True, deadline=deadline)
    own = sum(1 for s_, r in zip(sent, rows) if s_ in (r.get("text") or ""))
    leaks = sum(1 for i, r in enumerate(rows) for j, s_ in enumerate(sent) if j != i and s_ in (r.get("text") or ""))
    c_ = {"level": level, "own_hits": own, "foreign_leaks": leaks,
          "errors": sum(1 for r in rows if r.get("error")) + max(0, level - len(rows))}
    # rev 2g (GPT R2F-07): 0 own hits = the detector showed nothing: INCONCLUSIVE, not proof of isolation
    c_["informative"] = own > 0 and c_["errors"] == 0
    c_["classification"] = ("FAIL" if leaks else "INCOMPLETE" if c_["errors"] else
                            "INCONCLUSIVE" if not c_["informative"] else "PASS_NO_OBSERVED_LEAK")
    return c_


def vms_classify(a):
    """rev 2e (GPT audits E8 / HIGH-3): speed, correctness, capacity and adoption kept apart. Re-run for every arm at the
    end (the production reference is judged only once the oracle exists)."""
    level = a.get("level_wanted") or 16
    a["measurement_ok"] = all(vms_window_healthy(a.get(b), level) for b in ("short", "long"))
    a["capacity_ok"] = all(((a.get(b) or {}).get("measured_level") or 0) >= level and not (a.get(b) or {}).get("capacity_limited")
                           for b in ("short", "long"))
    # rev 2g (GPT R1 / R2F-01): missing evidence never counts as a pass: XL PASS whenever XL prompts exist; the canary
    # present, at this level, 0 errors / 0 leaks; DP rows diagnostic; a PCIe-IPC arm must show the init line
    _c = a.get("canary")
    _canary_ok = isinstance(_c, dict) and _c.get("level") == level and _c.get("errors") == 0 and _c.get("foreign_leaks") == 0
    _xl_ok = a.get("xl_gate") == "PASS" if _X else a.get("xl_gate") in (None, "PASS")
    a["backend_unverified"] = bool((a.get("env") or {}).get("VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC")
                                   and not (a.get("pcie_ipc") or {}).get("initialized"))
    a["correctness_ok"] = a.get("gate") == "PASS" and _xl_ok and _canary_ok
    a["adoptable"] = bool(VMS.get("oracle_ready") and a["measurement_ok"] and a["capacity_ok"] and a["correctness_ok"]
                          and not a.get("diagnostic_only") and not a["backend_unverified"]
                          and a.get("server_alive_after") and not a.get("log_errors") and not a.get("error"))
    a["candidate_for_arc_validation"] = a["adoptable"]
    a["production_approved"] = False         # rev 2g: needs the 60k+ generation and ARC task gates (not in this matrix)
    a["certified_exact"] = a["adoptable"] and a.get("gate_class") == "PASS_EXACT"     # rev 2e-d
    a["ok"] = a["measurement_ok"]        # "speed data from clean windows exists"; adoption is a["adoptable"]
    return a


def vms_arm(name, flags, **kw):
    """rev 2f: one arm's own crash never ends the remaining arms (recorded, the server killed, the results saved)."""
    try:
        return _vms_arm(name, flags, **kw)
    except Exception as _e:
        a = dict(VMS["arms"].get(name) or {}, ok=False, adoptable=False, certified_exact=False,
                 error=f"arm crashed: {type(_e).__name__}: {_e}"[:500])
        VMS["arms"][name] = a
        if name not in VMS["order"]:
            VMS["order"].append(name)
        print(f"VMS arm {name} crashed:", traceback.format_exc()[-1500:], flush=True)
        try:
            if VMS_SERVER is not None:
                VMS_SERVER.kill()
        except Exception:
            pass
        e14_save()
        return a


def _vms_arm(name, flags, *, level=16, gate=True, env_extra=None, cap=14 * 60):
    ok, rec = vms_launch(name, flags, env_extra)
    a = {"ok": False, "launch_sec": rec.get("sec"), "kv_tokens": rec.get("kv_tokens"), "skipped": rec.get("skipped"),
         "flags": flags, "env": env_extra or {}, "level_wanted": level, "kernels": rec.get("kernels")}
    VMS["arms"][name] = a          # rev 2g: a crash in the post-processing keeps the measured windows
    if ok:
        # rev 2b (agent 3 F1): one deadline for the whole arm: its cap, and never into the 2 x TP2 reserve
        t_end = min(time.perf_counter() + cap, HARD_STOP - VMS_RESERVE - 120)
        a["deadline_in_sec"] = round(t_end - time.perf_counter())
        _dp = "--data-parallel-size" in [str(x) for x in flags]
        a["diagnostic_only"] = bool(_dp)       # rev 2g (GPT R2F-01): no XL / per-rank canary on DP: never adoptable
        try:
            if gate in ("ref", "oracle"):
                rows, aa = vms_aa(name, deadline=t_end)
                VMS_ROWS[name] = rows
                a["aa"] = aa
                a["canary"] = vms_canary(level, deadline=t_end)
                if gate == "ref":
                    VMS_REF.update(rows)
                    a["gate"] = "reference" if aa["complete"] else "reference (A/A incomplete: every later gate INCOMPLETE)"
                elif aa["degenerate"] == 0 and aa["complete"] and a["canary"].get("foreign_leaks") == 0 \
                        and a["canary"].get("errors") == 0 and ((not _X) or (isinstance(aa.get("xl"), dict)
                        and aa["xl"].get("expected", 0) > 0 and aa["xl"].get("compared") == aa["xl"].get("expected")
                        and not aa["xl"].get("missing_margins"))):      # rev 2g (GPT R2F-10): XL A/A + canary clean
                    # rev 2: the oracle replaces the gate reference; the production reference is judged from its own rows
                    _r = VMS_ROWS.get("ref_fp8_mtp3_s16")
                    VMS_REF.clear()
                    VMS_REF.update(rows)
                    VMS["oracle_ready"] = True       # rev 2e (GPT audits HIGH-1 / E3): certification needs this
                    VMS["gate_reference"] = name + " (FP8, no speculation, prefix caching off)"
                    a["gate"] = "oracle"
                    if _r:
                        _v, _d = vms_judge(_r["serial_a"], _r["conc_a"], rows)
                        # rev 2g-b (agent 2 V1-d): the reference's warm (prefix-hit) A/A rows judged too; the worse counts
                        _v2, _d2 = vms_judge(_r["serial_b"], _r["conc_b"], rows)
                        _rank = {"PASS": 0, "INCOMPLETE": 1, "FAIL": 2}
                        _d["warm_verdict"], _d["warm_gate_class"] = _v2, _d2.get("gate_class")
                        if _rank.get(_v2, 2) > _rank.get(_v, 2):
                            _v, _d = _v2, dict(_d2, cold_verdict=_v, warm_verdict=_v2)
                        elif _v == "PASS" and _d2.get("gate_class") == "BOUNDARY":
                            _d["gate_class"] = "BOUNDARY"
                        VMS["ref_vs_oracle"] = {"verdict": _v, "detail": _d, "xl": vms_judge_xl(_r.get("xl_a"), rows)}
                        if VMS["arms"].get("ref_fp8_mtp3_s16"):
                            VMS["arms"]["ref_fp8_mtp3_s16"].update(gate=_v, gate_class=_d.get("gate_class"),
                                                                  xl_gate=(VMS["ref_vs_oracle"]["xl"] or {}).get("verdict"))
                else:
                    a["gate"] = "oracle unusable (incomplete or degenerate A/A, or a canary leak)"
            elif VMS_REF:
                s, c = vms_greedy(_S[:6], deadline=t_end), vms_greedy(_S, concurrent=True, deadline=t_end)
                xg = vms_greedy(_X[:4], concurrent=True, deadline=t_end) if (_X and not _dp) else None
                vms_save_rows(name + "_gate", {"serial": s, "conc": c, "xl": xg})
                a["gate"], a["gate_detail"] = vms_judge(s, c, VMS_REF, gate=bool(gate))
                a["gate_class"] = a["gate_detail"].get("gate_class")
                a["xl_gate"] = (vms_judge_xl(xg, VMS_REF) or {}).get("verdict") if xg else None
                a["canary"] = vms_canary(level, deadline=t_end)
            # rev 2: a FAIL is recorded and never adopted, but the speed is still measured
            a["short"] = vms_window(_S, level, _ts(75), name + "_short", deadline=t_end)
            # the next band runs unless the server is dead or dirty (a contaminated window would follow)
            _stop_next = lambda w: (w or {}).get("error") == "DIRTY_SERVER_AFTER_WINDOW" or (w or {}).get("idle_after") is False
            if time.perf_counter() < t_end and not _vms_dead() and not _stop_next(a["short"]):
                a["long"] = vms_window(_L, level, _ts(75), name + "_long", deadline=t_end)
            # rev 2d: the XL band (~85k prompts + up to 8k generated, ~100k per stream), at the streams the KV holds
            if _X and not _dp and time.perf_counter() < t_end and not _vms_dead() and not _stop_next(a.get("long")):
                _lx = max(1, min(level, (rec.get("kv_tokens") or 0) // 100_000)) if rec.get("kv_tokens") else min(level, 6)
                a["xl"] = vms_window(_X, _lx, _ts(75), name + "_xl", deadline=t_end)
            a["server_alive_after"] = not _vms_dead()
        except BaseException as _e:
            a["error"] = f"{type(_e).__name__}: {_e}"[:500]
        sig = VMS_SERVER.log_text() if VMS_SERVER is not None else ""
        # rev 2g-c (agent 3 A5): no whole-log 'Preempt' scan (the stats line carries it; one preemption in the diagnostic
        # XL window blocked adoption); the short / long windows require 0 preemptions instead (vms_window_healthy)
        a["log_errors"] = {k: sig.count(k) for k in ("Traceback", "out of memory", "OutOfMemory") if k in sig}
        a["preempt_log_lines"] = sig.count("Preempt")
    if VMS_SERVER is not None and "VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC" in (env_extra or {}):
        _lt = VMS_SERVER.log_text()        # rev 2d: did the PCIe-IPC all-reduce really engage?
        a["pcie_ipc"] = {"initialized": "Initialized FlashInfer PCIe IPC all-reduce" in _lt,
                         "not_provided": "does not provide PcieIpcAllReduceWorkspace" in _lt,
                         "other_warning": [l[-200:] for l in _lt.splitlines() if "PCIe IPC" in l and "Initialized" not in l][:4]}
    for band in ("short", "long", "xl"):
        w = a.get(band) or {}
        a[band + "_tok_s"], a["accept_" + band], a["level_" + band] = w.get("tok_s"), w.get("accept"), w.get("measured_level")
        a[band + "_e2e_tok_s"] = w.get("end_to_end_tok_s")
    vms_classify(a)
    VMS["arms"][name] = a
    VMS["order"].append(name)
    e14_save()
    print("VMS arm", name, json.dumps({k: v for k, v in a.items() if k not in ("flags", "short", "long", "xl")}, default=str)[:700], flush=True)
    return a


if VMS_ON and VMS_INST and not VMS["install"].get("error"):
    _ask = "Find the transformation rule of this ARC-AGI-2 task and describe it, then give the test output grids."
    _rows = lambda g: "\n".join(" ".join(str(v) for v in r) for r in g)
    _tt = lambda t: "".join(f"\nExample {i + 1}\nInput:\n{_rows(p['input'])}\nOutput:\n{_rows(p['output'])}\n"
                            for i, p in enumerate(t["train"])) + \
        "".join(f"\nTest input {j + 1}:\n{_rows(q['input'])}\n" for j, q in enumerate(t["test"]))
    _ids = lambda msgs: tokenizer.encode(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                                        enable_thinking=True, reasoning_effort=REASONING_EFFORT),
                                         add_special_tokens=False)
    _S = [_ids([{"role": "user", "content": _ask + "\n" + _tt(CHALLENGES[t])}]) for t in TASK_IDS]
    _all = _load_json(ARC_ROOT / "arc-agi_evaluation_challenges.json")
    _bg, _n = [], 0
    for _t in sorted(_all):
        if _t in CHALLENGES:
            continue
        _bg.append(_tt(_all[_t]))
        _n += len(tokenizer.encode(_bg[-1], add_special_tokens=False))
        if _n >= int(os.environ.get("E14_LONG_TOKENS", 36000)):
            break
    _block = "Reference tasks (not to be solved):\n" + "\n".join(_bg)
    _L = [_ids([{"role": "system", "content": _block}, {"role": "user", "content": _ask + "\n" + _tt(CHALLENGES[t])}])
          for t in TASK_IDS]
    # rev 2d: the XL band, a shared ~85k-token block (vLLM max_model_len 106,496: 85k + 8k generated fits)
    _bgx, _nx = list(_bg), _n
    for _t in sorted(_all):
        if _nx >= int(os.environ.get("E14_XL_TOKENS", 85000)):
            break
        if _t in CHALLENGES or _tt(_all[_t]) in _bgx:
            continue
        _bgx.append(_tt(_all[_t]))
        _nx += len(tokenizer.encode(_bgx[-1], add_special_tokens=False))
    _blockx = "Reference tasks (not to be solved):\n" + "\n".join(_bgx)
    _X = [_ids([{"role": "system", "content": _blockx}, {"role": "user", "content": _ask + "\n" + _tt(CHALLENGES[t])}])
          for t in TASK_IDS]
    VMS["prompts"] = {"short_p50": statistics.median(len(x) for x in _S), "long_p50": statistics.median(len(x) for x in _L),
                      "xl_p50": statistics.median(len(x) for x in _X)}
    E14_STATE.enter("VMS_ARMS")
    vms_arm("ref_fp8_mtp3_s16", vms_flags(MODEL_PATH), gate="ref")
    VMS_RESERVE = _ts(35 * 60)            # the 2 x TP2 slot: no earlier arm may eat it
    # rev 2: the oracle first (FP8, no speculation, prefix caching off), then the arms gated against it
    vms_arm("fp8_target_only_noprefix", vms_flags(MODEL_PATH, spec=0, prefix=False), gate="oracle")
    VMS.setdefault("gate_reference", "ref_fp8_mtp3_s16 (the oracle A/A was not usable)")
    _INT4_OK = vms_int4_mtp_static(VMS_INT4)
    try:          # rev 2g-c (checklist 9): the exact packages and prompts of this run
        VMS["pip_freeze"] = subprocess.run([(VMS.get("install") or {}).get("launch", [sys.executable])[0], "-m", "pip", "freeze"],
                                           capture_output=True, text=True, timeout=180).stdout[-20000:]
    except Exception as _e:
        VMS["pip_freeze"] = repr(_e)[:200]
    VMS["prompt_sha"] = {k_: [__import__("hashlib").sha256(json.dumps(list(x_)).encode()).hexdigest()[:16] for x_ in v_]
                         for k_, v_ in (("short", _S), ("long", _L), ("xl", _X or []))}
    PLAN = [
        # rev 2d: FlashInfer PCIe-IPC all-reduce (cuda_communicator.py: dispatched after custom / quick all-reduce, both
        # off on 4 PCIe GPUs, before NCCL; bf16 / fp16 only)
        ("fp8_mtp3_pcie", vms_flags(MODEL_PATH), dict(env_extra={"VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC": "1"})),
        # rev 2e (GPT audit order): the prefix-cache isolation, MTP width, 20 / 24 streams, then INT4 and SSM variants
        ("fp8_mtp3_noprefix", vms_flags(MODEL_PATH, prefix=False), {}),
        ("fp8_mtp2", vms_flags(MODEL_PATH, spec=2), {}),
        ("fp8_mtp4", vms_flags(MODEL_PATH, spec=4), {}),
        ("fp8_mtp3_s20", vms_flags(MODEL_PATH, seqs=20), dict(level=20)),
        ("fp8_mtp3_s24", vms_flags(MODEL_PATH, seqs=24), dict(level=24)),
        # rev 2 (GPT audit V6): VLLM_TEST_FORCE_FP8_MARLIN is gone (not in v0.32.0 envs.py); --linear-backend replaces it
        # (arg_utils.py:1808-1810, config/kernel.py:244-266). A layer type without that backend falls back to auto, so
        # the arm records the kernels the log says were selected.
        # rev 2b (agent 4 F1, checked in kernels/linear/__init__.py:478-489 + cutlass_scaled_mm_entry.cu:161-174): on SM 8.9
        # the automatic FP8 block-128 choice is already MarlinFP8ScaledMM (weight-only W8A16): CUTLASS block FP8 needs
        # SM 90+, DeepGEMM / B12x too. So "marlin" = the reference; "triton" = the only real W8A8 FP8 GEMM on the L4.
        ("fp8_mtp3_lb_triton", vms_flags(MODEL_PATH, extra={"--linear-backend": "triton"}), {}),
        ("fp8_mtp3_ssm_fp16", vms_flags(MODEL_PATH, extra={"--mamba-ssm-cache-dtype": "float16"}), {}),
        ("int4_mtp3_s16", vms_flags(VMS_INT4), dict(gate=False)),
        ("int4_mtp3_s20", vms_flags(VMS_INT4, seqs=20), dict(level=20, gate=False)),
    ]
    VMS["dropped"] = {"fp8_mtp3_marlin (VLLM_TEST_FORCE_FP8_MARLIN=1)": "the env var is not in vLLM 0.32.0 envs.py "
                      "(removed by PR #52182): the arm would have run the reference unchanged; replaced by --linear-backend",
                      "fp8_mtp3_lb_marlin (--linear-backend marlin)": "on SM 8.9 the automatic FP8 block-128 kernel is "
                      "already MarlinFP8ScaledMM (kernels/linear/__init__.py:478-489; CUTLASS block FP8 needs SM 90+, "
                      "cutlass_scaled_mm_entry.cu:161-174): the arm = the reference"}
    for _name, _fl, _kw in PLAN:
        if _name.startswith("int4_") and not _INT4_OK:
            VMS["arms"][_name] = {"ok": False, "skipped": "INT4 MTP static check failed: " + VMS["int4_mtp_static"].get("why", "")}
            VMS["order"].append(_name)
            continue
        vms_arm(_name, _fl, **_kw)
    # LAST: 2 x TP2 inside one vLLM server (data parallel 2 x tensor parallel 2), 8 seqs per rank = 16 in all
    # (GPT audit V5: --max-num-seqs is per DP rank)
    VMS_RESERVE = 0.0
    # rev 2b (agent 4 F4b, cli/serve.py:107-130): one API server (DP defaults to one per rank, each balancing on its own
    # counts). Custom all-reduce stays OFF: a TP2 group would use it when the P2P test passes (custom_all_reduce.py:
    # 281-291), and on this VM the P2P/IPC path measured 5.5x slower than NCCL (E14A nccl_bench p2p_sys 1,880 us vs 344 us)
    _TP2X = {"--api-server-count": 1}
    _NCCLX = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT"}       # rev 2c (GPT audit Gate D): the NCCL transport in the log
    try:          # rev 2c (Gate D / section 7): the topology and P2P matrices
        VMS["topology"] = {"matrix": subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True, timeout=30).stdout[-3000:],
                           "p2p_read": subprocess.run(["nvidia-smi", "topo", "-p2p", "r"], capture_output=True, text=True, timeout=30).stdout[-2000:],
                           "p2p_prop": subprocess.run(["nvidia-smi", "topo", "-p2p", "p"], capture_output=True, text=True, timeout=30).stdout[-2000:]}
    except Exception as _e:
        VMS["topology"] = {"error": repr(_e)[:200]}
    vms_arm("tp2x2_fp8_mtp3", vms_flags(MODEL_PATH, tp=2, dp=2, seqs=8, extra=_TP2X), level=16, gate=True, cap=16 * 60, env_extra=_NCCLX)
    if _INT4_OK:
        vms_arm("tp2x2_int4_mtp3", vms_flags(VMS_INT4, tp=2, dp=2, seqs=8, extra=_TP2X), level=16, gate=False,
                cap=16 * 60, env_extra=_NCCLX)
    else:
        VMS["arms"]["tp2x2_int4_mtp3"] = {"ok": False, "skipped": "INT4 MTP static check failed: " + VMS["int4_mtp_static"].get("why", "")}
        VMS["order"].append("tp2x2_int4_mtp3")
    for _a in VMS["arms"].values():       # rev 2e: the final classification (the reference is judged after the oracle)
        if "level_wanted" in _a:
            vms_classify(_a)
    E14_STATE.enter("VMS_DONE")
    e14_save()
else:
    VMS["skipped"] = "rerun" if RERUN else ("install failed" if VMS_INST is None or (VMS["install"] or {}).get("error") else "VMS off")
    print("VMS skipped:", VMS["skipped"], flush=True)
'''

V3 = r'''# ==================================================================== CELL V3 — report, bundle
try:
    E14_STATE.enter("REPORT")
    _ref = VMS["arms"].get("ref_fp8_mtp3_s16") or {}
    _rs, _rl = _ref.get("short_tok_s") or 0, _ref.get("long_tok_s") or 0
    f = lambda v, d=0: "—" if v is None else (f"{v:.{d}f}" if isinstance(v, (int, float)) else str(v))
    xr = lambda v, r: "—" if not v or not r else f"{v / r:.2f}"
    L = ["# VMS: vLLM 0.32.0 engine-config matrix", "", f"profile {PROFILE}, rerun {RERUN}, install {VMS.get('install')}",
         f"prompts {VMS.get('prompts')}",
         # rev 2 (GPT pre-GPU audit): the oracle, the production reference vs it, the INT4 MTP static check, dropped arms
         f"gate reference: {VMS.get('gate_reference')} | oracle_ready (certification possible): {bool(VMS.get('oracle_ready'))}",
         "tok/s = steady decode after the prefill settled; e2e = output tokens from submission to the window end / wall (a "
         "submission-to-window-end rate, not completed-request task throughput); "
         "BOUNDARY = PASS with only near-tie divergences (adoptable under the documented tolerance, not exact: confirm "
         "on ARC tasks before production); certified_exact = adoptable and PASS_EXACT",
         f"reference (MTP 3 + prefix caching) vs the oracle: {(VMS.get('ref_vs_oracle') or {}).get('verdict')} "
         f"{json.dumps((VMS.get('ref_vs_oracle') or {}).get('detail'))[:400]}",
         f"INT4 MTP static check: {json.dumps(VMS.get('int4_mtp_static'))[:500]}",
         f"dropped arms: {json.dumps(VMS.get('dropped') or {})}",
         "note: the 40k band shares a ~36k system block; prefix-cache arms reuse it, the oracle and fp8_mtp3_noprefix "
         "prefill it again per request: tok/s is measured after the prefill settled (decode only), the recompute shows "
         "in the e2e column", "",
         "| arm | ok | gate | streams (8k / 40k) | 8k tok/s | x ref | 40k tok/s | x ref | accept 8k / 40k | KV tokens | KV max | preemptions | degenerate 8k / 40k | log errors | XL tok/s (streams) | x ref | accept XL | PCIe-IPC initialized | gate class / XL gate | canary leaks | e2e tok/s 8k / 40k | prefix hits / queries 40k | measurement / capacity / correctness | adoptable |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    _rx = _ref.get("xl_tok_s") or 0
    for k in VMS.get("order") or []:
        a = VMS["arms"][k]
        L.append(f"| {k} | {a.get('ok')} | {a.get('gate')} | {f(a.get('level_short'))} / {f(a.get('level_long'))} | "
                 f"{f(a.get('short_tok_s'))} | {xr(a.get('short_tok_s'), _rs)} | {f(a.get('long_tok_s'))} | {xr(a.get('long_tok_s'), _rl)} | "
                 f"{f(a.get('accept_short'), 2)} / {f(a.get('accept_long'), 2)} | {f(a.get('kv_tokens'))} | "
                 f"{f(((a.get('short') or {}).get('kv_max')), 2)} / {f(((a.get('long') or {}).get('kv_max')), 2)} | "
                 f"{f((a.get('short') or {}).get('preemptions'))} / {f((a.get('long') or {}).get('preemptions'))} | "
                 f"{f((a.get('short') or {}).get('degenerate'))} / {f((a.get('long') or {}).get('degenerate'))} | "
                 f"{a.get('log_errors') or a.get('skipped') or a.get('error') or (a.get('short') or {}).get('error') or (a.get('long') or {}).get('error') or ''} | "
                 f"{f(a.get('xl_tok_s'))} ({f(a.get('level_xl'))}){' ' + str((a.get('xl') or {}).get('error')) if (a.get('xl') or {}).get('error') else ''} | "
                 f"{xr(a.get('xl_tok_s'), _rx)} | {f(a.get('accept_xl'), 2)} | {(a.get('pcie_ipc') or {}).get('initialized', '')} | "
                 f"{a.get('gate_class') or ''} / {a.get('xl_gate') or ''} | {(a.get('canary') or {}).get('foreign_leaks', '')} | "
                 f"{f(a.get('short_e2e_tok_s'))} / {f(a.get('long_e2e_tok_s'))} | "
                 f"{f((a.get('long') or {}).get('prefix_cache_hits'))} / {f((a.get('long') or {}).get('prefix_cache_queries'))} | "
                 f"{a.get('measurement_ok')} / {a.get('capacity_ok')} / {a.get('correctness_ok')} | {a.get('adoptable')}{' certified_exact' if a.get('certified_exact') else ''} |")
    L += ["", "selected kernels per arm (from the server log):"] + [
        f"- {k}: {(VMS['arms'][k].get('kernels') or [])[:6]}" for k in VMS.get("order") or []]
    (VMS_DIR / "vms_report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L), flush=True)
    E14_STATE.end("done")
    e14_save()
    _pack = WORKING / "Q27X_VMS_BUNDLE.zip"
    with zipfile.ZipFile(_pack, "w", zipfile.ZIP_DEFLATED) as z:
        for fp in sorted(VMS_DIR.rglob("*")):
            if fp.is_file() and fp.stat().st_size < 300e6:
                z.write(fp, arcname="vms/" + fp.relative_to(VMS_DIR).as_posix())
        for fp in sorted(E14_OUT.rglob("*")):
            if fp.is_file() and fp.stat().st_size < 300e6:
                z.write(fp, arcname="e14/" + fp.relative_to(E14_OUT).as_posix())
        for fp in (WORKING / "e14_state.json", WORKING / "e14_heartbeat.log"):
            if fp.exists():
                z.write(fp, arcname=fp.name)
    print("VMS bundle:", _pack, round(_pack.stat().st_size / 1e6, 1), "MB", flush=True)
except BaseException:
    print("VMS report error:", traceback.format_exc()[-2000:], flush=True)
try:
    if VMS_SERVER is not None:
        VMS_SERVER.kill()
except Exception:
    pass
'''

pre = '''import re, signal as _signal, urllib.request as _urlreq      # VMS: the helpers the dropped SGLang engine cell provided
def _e11_gpu_free(max_sec=180):
    """Wait until every GPU shows < 1 GiB used (nvidia-smi); returns the last reading."""
    t0, used = time.perf_counter(), None
    while time.perf_counter() - t0 < max_sec:
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=30).stdout
            used = [int(x) for x in out.split() if x.strip().isdigit()]
            if used and max(used) < 1024:
                return {"free": True, "used_mb": used, "sec": round(time.perf_counter() - t0, 1)}
        except Exception:
            return {"free": None, "error": "nvidia-smi unavailable"}
        time.sleep(3)
    return {"free": False, "used_mb": used}
'''
c3 = rep(S[3], '_wdirs = (SGL_MODEL_PATH, SGL_DRAFT_PATH) if ENGINE_BACKEND == "sglang" else (MODEL_PATH,)   # E11-05',
         "_wdirs = (MODEL_PATH,)   # VMS rev 2b: the FP8 reference loads first (the SGLang default read INT4 + DFlash2)")
cells = [S[0], c1, c1b, c3, S[4], pre + V1, V2, V3]
md = """# VMS — vLLM 0.32.0 engine-config matrix (Qwen3.8-27B on 4x L4), engine only

The vLLM twin of the EMS (SGLang) notebook: every vLLM configuration worth testing for 600-800 tok/s, each its own
server launch. Reference = the S1-60K engine (LB 34.58): FP8 block-128, fp8 KV, MTP 3, FLASHINFER, prefix caching,
16 seqs. Arms: the oracle (FP8 target-only, prefix caching off); MTP 2 / 4; 20 / 24 seqs; SSM state fp16; prefix
caching off; --linear-backend triton (Marlin = the auto default on SM 8.9); INT4 (RedHat compressed-tensors) + MTP at 16 and 20 after a static MTP
check; last, 2 x TP2 (data parallel 2 x tensor parallel 2) from a reserved slot.
Each arm: a greedy correctness gate vs the oracle's own A/A (rev 2; INT4 arms diagnostic: other weights) with a
degenerate-output count, then 8k and 40k steady windows measured from /metrics. Report `q27x/vms/vms_report.md`, bundle `Q27X_VMS_BUNDLE.zip`.

**Inputs:** competition data; the FP8 repack and the RedHat INT4 models; the vLLM 0.32.0 wheelhouse
(`foysalemonshanto/vllm-0-32-0-cu13-wheelhouse`); the vLLM 0.19 wheelhouse (only the kernel's tokenizer pins); the
SGLang wheelhouse is NOT needed; the payload dataset (E14 modules).
"""
nb = json.load(open(SRC, encoding="utf-8"))
nb["cells"] = [{"cell_type": "markdown", "metadata": {}, "source": md.splitlines(keepends=True)}] + [
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.splitlines(keepends=True)}
    for c in cells]
for i, c in enumerate(cells):
    compile(c, f"<cell{i}>", "exec")
out = OUT
if out.exists() and sha(out) in PUSHED:
    out = out.with_name(out.stem + "_REBUILD.ipynb")
data = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
out.write_bytes(data.encode("utf-8"))
print(out.name, sha(out)[:8], f"{len(data.encode('utf-8')) / 1024:.0f} kB", len(cells), "code cells")
