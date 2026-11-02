# ==================================================================== CELL 0 — mode + fallback submission (stdlib only)
import os, sys, json, time, random, hashlib, re, shutil, subprocess, threading, traceback, zipfile, math
from pathlib import Path
from datetime import datetime, timezone
import concurrent.futures as cf

NOTEBOOK_T0 = time.perf_counter()
# S1 (CLOCK): Kaggle's 12 h counts from kernel start; R5's cell 0 began 7 s later. Anchor the clock at the kernel
# process start (capped at 10 min in case the value is odd).
try:
    import psutil as _ps
    _since_kernel_start = max(0.0, min(600.0, time.time() - _ps.Process(os.getpid()).create_time()))
except Exception:
    _since_kernel_start = 0.0
NOTEBOOK_T0 -= _since_kernel_start
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes"}
ARC_ROOT = Path(os.environ.get("Q27X_ARC_ROOT", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"))
if not ARC_ROOT.exists():
    for _alt in ("/kaggle/input/arc-prize-2026-arc-agi-2",):
        if Path(_alt).exists():
            ARC_ROOT = Path(_alt)
WORKING = Path(os.environ.get("Q27X_WORKING", "/kaggle/working")); WORKING.mkdir(parents=True, exist_ok=True)
OUT = WORKING / "q27x"; OUT.mkdir(parents=True, exist_ok=True)
SUBMISSION_PATH = WORKING / "submission.json"

def _load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

test_challenges = _load_json(ARC_ROOT / "arc-agi_test_challenges.json")

def write_json_atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)

def identity_submission(challenges):
    return {tid: [{"attempt_1": q["input"], "attempt_2": q["input"]} for q in task["test"]]
            for tid, task in challenges.items()}

write_json_atomic(SUBMISSION_PATH, identity_submission(test_challenges))
print("MODE:", "COMPETITION RERUN (hidden test)" if RERUN else "COMMIT / INTERACTIVE (evaluation)",
      "| fallback submission.json:", len(test_challenges), "tasks | RUN_ID", RUN_ID,
      "| clock anchored", round(_since_kernel_start, 1), "s before cell 0")


# ==================================================================== CELL 1 — config, inputs, lab module (RC2 verbatim + E25 NB1 / NB4S / NB4T marked edits)
import os, sys, json, math, time, re, shutil, subprocess, hashlib, tarfile, threading, traceback
from pathlib import Path
T_START = time.time()
MODE = "FULL"
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes"}
FAKE = os.environ.get("E23_FAKE_SERVER")                          # local CPU harness only
BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 120 * 60)     # E25 NB4T: hard stop 120 min (Q > I8 > P26 > P27 > C12)
HARD_STOP = T_START + BUDGET
MAX_LEN = 131072                                                  # user 2026-10-29: thinking up to 120k
OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/nb4t"))   # E25 NB4T
OUT.mkdir(parents=True, exist_ok=True)
LOGS = OUT / "logs"
LOGS.mkdir(exist_ok=True)
TRACES = OUT / "traces"                                           # memory rule: the full text of every long generation
TRACES.mkdir(exist_ok=True)
TMP = Path(os.environ.get("E23_TMP", "/tmp"))
INP = Path(os.environ.get("E23_INPUT", "/kaggle/input"))
R = {"what": "E25 NB4T LEVERS: INT8 draft head / PP partition 26,22 + 27,21 / 12 streams on the Q-defer base (Flash-Next on Merlin)",   # E25 NB4T
      "budget_min": BUDGET // 60, "max_len": MAX_LEN,
     "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "setup": {}, "sections": {}, "errors": [], "order": []}
_SAVE_LOCK = threading.Lock()
NB1_EXTRA, NB1_ENV, NB1_KV = {}, {}, {}   # E25 NB1: per-boot trailing vllm flags, env and KV pin (filled by the boot plan)


def save():
    with _SAVE_LOCK:
        tmp = OUT / "nb4t_results.json.tmp"                      # E25 NB4T
        tmp.write_text(json.dumps(R, indent=1, default=str))
        tmp.replace(OUT / "nb4t_results.json")                   # E25 NB4T


def left():
    return HARD_STOP - time.time()


def log(*a):
    print(f"[{(time.time() - T_START) / 60:6.1f} min]", *a, flush=True)


COMP = INP / "competitions" / "arc-prize-2026-arc-agi-2"


def write_fallback_submission():
    """E25 NB4T: the proven E18 R2 format (CELL 0's writer): the identity submission of every test task (attempt_1 =
    attempt_2 = the test input), written atomically to CELL 0's SUBMISSION_PATH; never a [[0]] placeholder."""
    write_json_atomic(SUBMISSION_PATH, identity_submission(test_challenges))


if RERUN:                                                         # E25 NB4T: CELL 0 already wrote the identity file
    print("E25 NB4T speed levers: competition rerun -> CELL 0's identity fallback submission.json only (E18 R2 format)")
else:
    LAB_SRC = "\"\"\"E23 lab, rev 3: port-parameterized server lifecycle and measurement for the six-lever notebook (stdlib only).\n\nBase: the Exp 14 VMS harness (VServer / VGen: token ids from return_token_ids, [DONE] + finish_reason required, socket\nshutdown on stop, top-1/top-2 margins, garbage checks on every window). Adopted from the consolidated v3 notebook:\nsame-weight greedy oracle classified by the oracle's logprob margin at the first differing token, a steady decode window\nthat starts only after prefill settled, a GPU-release gate, fail-closed statuses, checkpoint identity, a runtime\n--help flag check, KV + GPU memory samples.\nrev 3 (validation agents 2026-10-29, measurement + robustness):\n  - port check by connect (a TIME_WAIT socket of the previous server is not \"busy\"), with a 90 s wait;\n  - kill(): SIGKILL the whole session after SIGTERM, even when the leader already exited;\n  - steady(): idle gate at entry, admission cap from the prefill to do (700 tok/s assumed), settle needs no preemption\n    and every running stream gaining tokens; in the window: stalled streams, a changing running/waiting set (KV-bound),\n    preemptions and prefills make it INCONCLUSIVE (numbers kept); garbage under load fails it; steps from the\n    vllm:iteration_tokens_total histogram count; replicas measured in one shared window (barrier);\n  - profiled_load(): settle as steady(), bounded profile (delay / max iterations), with_stack off, frontend ignored;\n  - trace_summary(): categories on the kernel base name (attention before GDN: \"fla\" is inside \"flashinfer\"), per-forward\n    graph / kernel launch counts inside the gpu_model_runner forward ranges (VLLM_CUSTOM_SCOPES_FOR_PROFILING=1) and the\n    CUPTI correlation fallback -> FULL / PIECEWISE / EAGER / UNKNOWN; startup cudagraph lines parsed explicitly;\n  - teacher_forced(): prompt logprobs of the last N positions of a long prompt (NVFP4 vs FP8 KV at 40k / 90k);\n  - logprobs keyed by token id (return_tokens_as_token_ids); stop() race closed; compare() flags garbage only when the\n    oracle itself is clean.\n\"\"\"\nimport gzip\nimport hashlib\nimport json\nimport os\nimport re\nimport signal\nimport socket\nimport statistics\nimport subprocess\nimport threading\nimport time\nimport urllib.request as U\nfrom pathlib import Path\n\nFAIL_MARGIN = 0.5          # first differing token: oracle top-1 minus top-2 logprob at or above this = real divergence\nLIVE = set()               # every running Server, for the hard-stop watchdog\n\n\n# ------------------------------------------------------------------ server lifecycle\ndef port_listening(port):\n    \"\"\"True when something accepts connections on 127.0.0.1:port (TIME_WAIT leftovers do not).\"\"\"\n    s = socket.socket()\n    s.settimeout(1.0)\n    try:\n        s.connect((\"127.0.0.1\", int(port)))\n        return True\n    except OSError:\n        return False\n    finally:\n        s.close()\n\n\nclass Server:\n    def __init__(self, name, argv, env, port, log_dir, gpus=\"0,1,2,3\"):\n        self.name, self.argv, self.env, self.port = name, [str(a) for a in argv], dict(env), int(port)\n        self.gpus = str(gpus)\n        self.log_path = Path(log_dir) / f\"server_{name}.log\"\n        self.proc, self.t0, self.ready_sec = None, None, None\n\n    @property\n    def url(self):\n        return f\"http://127.0.0.1:{self.port}\"\n\n    def start(self):\n        end = time.time() + 90\n        while port_listening(self.port):\n            if time.time() > end:\n                raise RuntimeError(f\"PORT_BUSY {self.port} (a listener is still there)\")\n            time.sleep(3)\n        self.log_path.parent.mkdir(parents=True, exist_ok=True)\n        self._fh = open(self.log_path, \"w\", encoding=\"utf-8\")\n        self.t0 = time.perf_counter()\n        self.proc = subprocess.Popen(self.argv, env=self.env, stdout=self._fh, stderr=subprocess.STDOUT,\n                                     start_new_session=True, cwd=\"/tmp\" if os.path.isdir(\"/tmp\") else None)\n        LIVE.add(self)\n        return self\n\n    def alive(self):\n        return self.proc is not None and self.proc.poll() is None\n\n    def log(self):\n        try:\n            return self.log_path.read_text(encoding=\"utf-8\", errors=\"replace\")\n        except Exception:\n            return \"\"\n\n    def rss_mb(self):\n        \"\"\"Host RAM of the server's whole session (engine + workers), MB; None off Linux.\"\"\"\n        try:\n            r = subprocess.run([\"ps\", \"-o\", \"rss=\", \"--sid\", str(self.proc.pid)], capture_output=True, text=True, timeout=20)\n            return round(sum(int(x) for x in r.stdout.split() if x.isdigit()) / 1024, 1)\n        except Exception:\n            return None\n\n    def wait_ready(self, timeout):\n        while time.perf_counter() - self.t0 < timeout:\n            if not self.alive():\n                return False, f\"exited rc={self.proc.returncode}\"\n            try:\n                U.urlopen(self.url + \"/health\", timeout=10).read()\n                self.ready_sec = round(time.perf_counter() - self.t0, 1)\n                return True, f\"ready in {self.ready_sec}s\"\n            except Exception:\n                time.sleep(3)\n        return False, \"startup timeout\"\n\n    def kill(self):\n        if self.proc is None:\n            return\n        pg = self.proc.pid\n        try:\n            os.killpg(pg, signal.SIGTERM) if hasattr(os, \"killpg\") else self.proc.terminate()\n        except Exception:\n            pass\n        try:\n            self.proc.wait(timeout=40)\n        except Exception:\n            pass\n        # rev 3: the leader may exit while EngineCore / workers of the session still hold GPUs: always SIGKILL the group\n        try:\n            os.killpg(pg, signal.SIGKILL) if hasattr(os, \"killpg\") else self.proc.kill()\n        except Exception:\n            pass\n        try:\n            self.proc.wait(timeout=30)\n        except Exception:\n            pass\n        try:\n            self._fh.close()\n        except Exception:\n            pass\n        LIVE.discard(self)\n\n    def facts(self):\n        \"\"\"What vLLM logged at startup: KV pool, max concurrency, selected kernels, graph mode, backend, KV dtype.\"\"\"\n        t = self.log()\n        kv = [int(x.replace(\",\", \"\")) for x in re.findall(r\"GPU KV cache size: ([\\d,]+) tokens\", t)]\n        mc = re.findall(r\"Maximum concurrency for [\\d,]+ tokens per request: ([\\d.]+)x\", t)\n        kern = sorted(set(re.findall(r\"Selected (\\w+Kernel) for\", t)))\n        errs = [l[:300] for l in t.splitlines() if re.search(r\"Traceback|CUDA error|illegal memory|out of memory|\"\n                                                               r\"RuntimeError|AssertionError\", l)][:12]\n        pick = lambda pat, n=8: [l.strip()[:240] for l in t.splitlines() if re.search(pat, l, re.I)][:n]\n        # rev 3 (measurement H5): the decisive startup evidence for the graph mode actually used with spec decode\n        downgrade = pick(r\"not supported with spec-decode.*setting cudagraph_mode|setting cudagraph_mode\\s*=|\"\n                         r\"[Oo]verriding cudagraph_mode to\", 4)       # rev 3e: + the 0.32 wording\n        capture = sorted(set(re.findall(r\"Capturing CUDA graphs \\(([^)]+)\\)\", t)))\n        configured = sorted(set(re.findall(r\"CUDAGraphMode\\.(\\w+)\", t)))\n        return {\"kv_tokens\": min(kv) if kv else None, \"kv_tokens_all\": kv, \"max_concurrency\": float(mc[-1]) if mc else None,\n                \"kernels\": kern, \"error_lines\": errs, \"ready_sec\": self.ready_sec,\n                \"graph_capture_modes\": capture, \"graph_downgrade_lines\": downgrade, \"cudagraph_modes_logged\": configured,\n                \"backend_lines\": pick(r\"attention backend|Using \\w+ backend|backend=|AttentionBackend\"),\n                \"kv_dtype_lines\": pick(r\"kv_cache_dtype|kv cache dtype|nvfp4\"),\n                \"host_embed_lines\": pick(r\"host[- _]?(resident|embed)|HOST_EMBED|embed.*(cpu|host)\", 6)}\n\n\ndef gpu_pids(gpus=None):\n    \"\"\"Compute PIDs on all GPUs, or only on the GPU indices in `gpus` (a bench on another GPU must not fail the release\n    gate of a server that used GPU 0 only). None when nvidia-smi is unavailable.\"\"\"\n    try:\n        a = subprocess.run([\"nvidia-smi\", \"--query-compute-apps=pid,gpu_uuid\", \"--format=csv,noheader,nounits\"],\n                           capture_output=True, text=True, timeout=30)\n        g = subprocess.run([\"nvidia-smi\", \"--query-gpu=index,uuid\", \"--format=csv,noheader,nounits\"],\n                           capture_output=True, text=True, timeout=30)\n    except Exception:\n        return None\n    if a.returncode or g.returncode:\n        return None\n    idx = {u.strip(): int(i) for i, u in (l.split(\",\", 1) for l in g.stdout.strip().splitlines() if \",\" in l)}\n    want = None if gpus is None else {int(x) for x in str(gpus).split(\",\") if str(x).strip() != \"\"}\n    out = set()\n    for l in a.stdout.strip().splitlines():\n        if \",\" not in l:\n            continue\n        pid, uuid = (x.strip() for x in l.split(\",\", 1))\n        if pid.isdigit() and (want is None or idx.get(uuid) in want):\n            out.add(int(pid))\n    return sorted(out)\n\n\ndef gpu_mem():\n    try:\n        r = subprocess.run([\"nvidia-smi\", \"--query-gpu=index,memory.used,memory.total\", \"--format=csv,noheader,nounits\"],\n                           capture_output=True, text=True, timeout=30)\n        return [[int(v) for v in l.split(\",\")] for l in r.stdout.strip().splitlines() if l.strip()]\n    except Exception:\n        return None\n\n\ndef wait_gpu_release(timeout=180, fake=False, gpus=None, free_mib=1024):\n    \"\"\"True when the GPUs in `gpus` (all when None) hold no compute PID AND use < free_mib MiB each (rev 3: compute-app\n    PIDs are often invisible inside Kaggle containers; memory.used is the proven E18 gate); None without nvidia-smi.\"\"\"\n    if fake:\n        return True\n    want = None if gpus is None else {int(x) for x in str(gpus).split(\",\") if str(x).strip() != \"\"}\n    end = time.time() + timeout\n    while time.time() < end:\n        p, mem = gpu_pids(gpus), gpu_mem()\n        if p is None or mem is None:\n            return None\n        used = [m[1] for m in mem if want is None or m[0] in want]\n        if not p and all(u < free_mib for u in used):\n            return True\n        time.sleep(3)\n    return False\n\n\ndef identity(model_dir):\n    \"\"\"Checkpoint identity: sha256 of config / tokenizer / index files (an oracle is only compared with the same weights).\"\"\"\n    root = Path(model_dir)\n    files = [p for n in (\"config.json\", \"tokenizer.json\", \"tokenizer_config.json\", \"generation_config.json\")\n             for p in [root / n] if p.is_file()] + sorted(root.glob(\"*.safetensors.index.json\"))\n    h = hashlib.sha256()\n    for p in files:\n        h.update(p.name.encode())\n        h.update(hashlib.sha256(p.read_bytes()).digest())\n    return {\"dir\": str(root), \"files\": [p.name for p in files], \"sha256\": h.hexdigest()[:16]}\n\n\n_HELP = {}\n\n\ndef help_check(key, argv_prefix, env, flags, timeout=300):\n    \"\"\"The engine's own `serve --help=all` (run once per engine `key`, cached) must list every flag of THIS launch\n    (rev 3: every launch is checked, not only the first); returns the missing ones.\"\"\"\n    if key not in _HELP:\n        try:\n            r = subprocess.run(list(argv_prefix) + [\"--help=all\"], capture_output=True, text=True, timeout=timeout, env=env)\n            _HELP[key] = r.stdout + r.stderr\n        except Exception as e:\n            _HELP[key] = \"ERROR \" + repr(e)[:200]\n    txt = _HELP[key]\n    if len(txt) < 2000:\n        return {\"ran\": False, \"error\": \"help output too short or failed\", \"tail\": txt[-400:]}\n    want = sorted({f for f in flags if isinstance(f, str) and f.startswith(\"--\")})\n    return {\"ran\": True, \"missing\": [f for f in want if f not in txt]}\n\n\ndef start_watchdog(hard_stop_epoch, on_fire):\n    \"\"\"A real hard stop: at `hard_stop_epoch` every live Server is killed and on_fire() is called.\"\"\"\n    def run():\n        while time.time() < hard_stop_epoch:\n            time.sleep(5)\n        try:\n            on_fire()\n        finally:\n            for s in list(LIVE):\n                try:\n                    s.kill()\n                except Exception:\n                    pass\n    t = threading.Thread(target=run, daemon=True, name=\"e23-watchdog\")\n    t.start()\n    return t\n\n\ndef prefetch(paths, threads=16, chunk=16 << 20):\n    \"\"\"Read files once with many readers (page cache warm-up of a cold /kaggle/input mount); returns bytes and seconds.\"\"\"\n    files = [p for root in paths for p in sorted(Path(root).rglob(\"*\")) if p.is_file()]\n    t0, done, lock = time.time(), [0], threading.Lock()\n\n    def worker(fs):\n        for p in fs:\n            try:\n                with open(p, \"rb\") as f:\n                    while True:\n                        b = f.read(chunk)\n                        if not b:\n                            break\n                        with lock:\n                            done[0] += len(b)\n            except Exception:\n                pass\n    big = sorted(files, key=lambda p: -p.stat().st_size)\n    parts = [big[i::threads] for i in range(threads)]\n    ths = [threading.Thread(target=worker, args=(x,), daemon=True) for x in parts if x]\n    for t in ths:\n        t.start()\n    for t in ths:\n        t.join()\n    dt = time.time() - t0\n    return {\"files\": len(files), \"bytes\": done[0], \"sec\": round(dt, 1), \"MBps\": round(done[0] / 1e6 / max(dt, 1e-3), 1)}\n\n\n# ------------------------------------------------------------------ metrics\ndef prom(port):\n    txt = U.urlopen(f\"http://127.0.0.1:{port}/metrics\", timeout=15).read().decode(\"utf-8\", \"replace\")\n    out = {}\n    for line in txt.splitlines():\n        if not line or line.startswith(\"#\"):\n            continue\n        m = re.match(r\"^([A-Za-z_:][A-Za-z0-9_:]*)(\\{[^}]*\\})?\\s+(\\S+)\", line)\n        if not m:\n            continue\n        try:\n            v = float(m.group(3))\n        except ValueError:\n            continue\n        if v != v:\n            continue\n        out[m.group(1)] = out.get(m.group(1), 0.0) + v\n        out[m.group(1) + \"@max\"] = max(out.get(m.group(1) + \"@max\", v), v)\n    return out\n\n\nGEN, PROMPT, RUN, WAIT = (\"vllm:generation_tokens_total\", \"vllm:prompt_tokens_total\", \"vllm:num_requests_running\",\n                          \"vllm:num_requests_waiting\")\nPREEMPT, KVU, ITER = \"vllm:num_preemptions_total\", \"vllm:kv_cache_usage_perc\", \"vllm:iteration_tokens_total_count\"\nACC, DRF, DTOK = (\"vllm:spec_decode_num_accepted_tokens_total\", \"vllm:spec_decode_num_drafts_total\",\n                  \"vllm:spec_decode_num_draft_tokens_total\")\n\n\n# ------------------------------------------------------------------ one streaming request (Exp 14 VGen, port-aware)\nclass Stream(threading.Thread):\n    def __init__(self, port, ids, max_new, temperature=1.0, seed=None, logprobs=0, ignore_eos=True, model=\"q27\"):\n        super().__init__(daemon=True)\n        self.port = int(port)\n        self.body = {\"model\": model, \"prompt\": list(ids), \"max_tokens\": int(max_new), \"temperature\": temperature,\n                     \"top_p\": 0.95 if temperature else 1.0, \"stream\": True, \"ignore_eos\": ignore_eos, \"return_token_ids\": True}\n        if temperature:\n            self.body[\"top_k\"] = 20\n        if seed is not None:\n            self.body[\"seed\"] = int(seed)\n        if logprobs:\n            self.body[\"logprobs\"] = int(logprobs)\n            self.body[\"return_tokens_as_token_ids\"] = True      # rev 3: logprob keys by id (no string collisions)\n        self.text, self.tokens, self.margins = [], [], []\n        self.err, self.done_seen, self.finish_reason = None, False, None\n        self.t_start = self.t_first = self.t_end = None\n        self._resp, self._halt = None, False\n\n    def run(self):\n        self.t_start = time.perf_counter()\n        try:\n            req = U.Request(f\"http://127.0.0.1:{self.port}/v1/completions\", data=json.dumps(self.body).encode(),\n                            method=\"POST\", headers={\"Content-Type\": \"application/json\"})\n            self._resp = U.urlopen(req, timeout=7200)\n            for raw in self._resp:\n                if self._halt:\n                    break\n                line = raw.strip()\n                if not line.startswith(b\"data:\"):\n                    continue\n                payload = line[5:].strip()\n                if payload == b\"[DONE]\":\n                    self.done_seen = True\n                    break\n                d = json.loads(payload)\n                if \"error\" in d or d.get(\"object\") == \"error\":\n                    self.err = (\"stream error: \" + json.dumps(d.get(\"error\") or d.get(\"message\") or d))[:300]\n                    break\n                for c in d.get(\"choices\") or []:\n                    self.text.append(c.get(\"text\") or \"\")\n                    ids_ = c.get(\"token_ids\")\n                    lp = c.get(\"logprobs\") or {}\n                    if ids_ is None:\n                        if c.get(\"text\") or lp.get(\"tokens\"):\n                            self.err = \"MISSING_STREAM_TOKEN_IDS\"\n                    elif any(not isinstance(v, int) or isinstance(v, bool) for v in ids_):\n                        self.err = \"INVALID_STREAM_TOKEN_IDS\"\n                    else:\n                        if ids_ and self.t_first is None:\n                            self.t_first = time.perf_counter()\n                        self.tokens.extend(ids_)\n                    for top in lp.get(\"top_logprobs\") or []:\n                        vals = sorted((v for v in (top or {}).values() if isinstance(v, (int, float))), reverse=True)\n                        self.margins.append(round(vals[0] - vals[1], 4) if len(vals) >= 2 else None)\n                    if c.get(\"finish_reason\") is not None:\n                        self.finish_reason = c[\"finish_reason\"]\n                if self.err:\n                    break\n        except Exception as e:\n            if not self._halt:\n                self.err = repr(e)[:300]\n        finally:                         # rev 3: a stop() before the response existed still closes the socket here\n            try:\n                if self._resp is not None:\n                    self._resp.close()\n            except Exception:\n                pass\n        self.t_end = time.perf_counter()\n        if not self._halt and not self.err and not (self.done_seen and self.finish_reason):\n            self.err = f\"INCOMPLETE_SSE done={self.done_seen} finish={self.finish_reason}\"\n\n    def stop(self):\n        self._halt = True\n        try:\n            self._resp.fp.raw._sock.shutdown(socket.SHUT_RDWR)\n        except Exception:\n            pass\n        try:\n            if self._resp is not None:\n                self._resp.close()\n        except Exception:\n            pass\n\n    def row(self):\n        return {\"text\": \"\".join(self.text), \"tokens\": list(self.tokens), \"margins\": list(self.margins), \"error\": self.err,\n                \"done\": self.done_seen, \"finish_reason\": self.finish_reason}\n\n\n# ------------------------------------------------------------------ correctness\ndef greedy(port, prompts, n_tok=128, logprobs=2, timeout=900, model=\"q27\"):\n    \"\"\"Greedy rows for the same-weight oracle (one request at a time: batch-size effects stay out of the comparison).\"\"\"\n    rows = []\n    for ids in prompts:\n        g = Stream(port, ids, n_tok, temperature=0.0, logprobs=logprobs, ignore_eos=True, model=model)\n        g.start()\n        g.join(timeout)\n        if g.is_alive():\n            g.stop()\n            g.join(10)\n            r = g.row()\n            r[\"error\"] = r[\"error\"] or \"GREEDY_TIMEOUT\"\n        else:\n            r = g.row()\n        rows.append(r)\n    return rows\n\n\ndef degenerate(text):\n    \"\"\"Objective garbage checks: empty, a run of one char, '!!!!' (the E3 / E13 garbage), very low variety.\"\"\"\n    t = text or \"\"\n    if not t.strip():\n        return \"EMPTY\"\n    if re.search(r\"(\\S)\\1{200,}\", t, re.S):      # rev 3: grid rows / separators are not garbage\n        return \"REPEATED_CHAR\"\n    if t.count(\"!\") > 0.5 * max(1, len(t)):\n        return \"BANG_GARBAGE\"\n    words = re.findall(r\"[A-Za-z]{3,}\", t)            # rev 3c: words only; ARC grid dumps (digits) are not garbage\n    if len(words) > 60 and len(set(words)) < 0.08 * len(words):\n        return \"LOW_VARIETY\"\n    return None\n\n\ndef compare(ref, cand):\n    \"\"\"Same-weight oracle: at the first differing token the ORACLE's top-1 - top-2 logprob margin decides:\n    >= FAIL_MARGIN -> FAIL_HIGH_MARGIN, below -> NEAR_TIE; identical -> EXACT; missing data -> INCONCLUSIVE.\"\"\"\n    if not ref or not cand or len(ref) != len(cand):\n        return {\"status\": \"INCONCLUSIVE_PAIRS_MISSING\"}\n    per, worst = [], \"EXACT\"\n    rank = {\"EXACT\": 0, \"NEAR_TIE\": 1, \"INCONCLUSIVE\": 2, \"FAIL_HIGH_MARGIN\": 3}\n    for a, b in zip(ref, cand):\n        if a.get(\"error\") or b.get(\"error\") or not a.get(\"tokens\") or not b.get(\"tokens\"):\n            st, d, m = \"INCONCLUSIVE\", None, None\n        else:\n            x, y = a[\"tokens\"], b[\"tokens\"]\n            d = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)\n            if d is None:\n                st, m = (\"EXACT\" if len(x) == len(y) else \"INCONCLUSIVE\"), None\n            else:\n                mm = a.get(\"margins\") or []\n                m = mm[d] if d < len(mm) else None\n                st = \"INCONCLUSIVE\" if m is None else (\"FAIL_HIGH_MARGIN\" if m >= FAIL_MARGIN else \"NEAR_TIE\")\n        g = degenerate(b.get(\"text\")) if (b.get(\"tokens\") and not b.get(\"error\")) else None\n        if g and degenerate(a.get(\"text\")) is None:           # rev 3c: a request error is not corruption\n            st = \"FAIL_HIGH_MARGIN\"\n        per.append({\"first_diff\": d, \"oracle_margin\": m, \"status\": st, \"garbage\": g})\n        worst = st if rank[st] > rank[worst] else worst\n    return {\"status\": worst, \"rows\": per}\n\n\ndef chat_probes(port, model=\"q27\", cases=((\"Reply with exactly one number: 7 * 6 =\", \"42\"),\n                                          (\"Reply with exactly one number: 2 + 2 =\", \"4\"),\n                                          (\"Reply with exactly one number: 9 - 3 =\", \"6\")), timeout=300):\n    \"\"\"Arithmetic through /v1/chat/completions with thinking off. vLLM 0.32 returns the answer in message.content and\n    the thinking in message.reasoning (NOT reasoning_content: the consolidated notebook's blocker).\"\"\"\n    out = []\n    for q, want in cases:\n        body = {\"model\": model, \"messages\": [{\"role\": \"user\", \"content\": q}], \"temperature\": 0, \"max_tokens\": 32,\n                \"chat_template_kwargs\": {\"enable_thinking\": False}}\n        try:\n            r = json.loads(U.urlopen(U.Request(f\"http://127.0.0.1:{port}/v1/chat/completions\", data=json.dumps(body).encode(),\n                                               headers={\"Content-Type\": \"application/json\"}), timeout=timeout).read())\n            msg = r[\"choices\"][0][\"message\"]\n            got = (msg.get(\"content\") or \"\").strip()\n            out.append({\"q\": q, \"want\": want, \"content\": got[:80], \"reasoning_chars\": len(msg.get(\"reasoning\") or \"\"),\n                        \"ok\": re.search(r\"(?<!\\d)\" + want + r\"(?!\\d)\", got) is not None})\n        except Exception as e:\n            out.append({\"q\": q, \"want\": want, \"error\": repr(e)[:200], \"ok\": False})\n    return {\"ok\": all(x[\"ok\"] for x in out), \"cases\": out}\n\n\ndef template_probe(port, messages, want_ids, model=\"q27\", timeout=120):\n    \"\"\"rev 3h (three GPT SMOKE audits): what the SERVER's chat path renders for `messages` (its own template and default\n    chat-template kwargs, no per-request kwargs) against the production xhigh render `want_ids`. MATCH / MISMATCH /\n    ERROR; the load requests themselves use pre-rendered ids, so this shows chat-path equivalence only.\"\"\"\n    body = {\"model\": model, \"messages\": messages, \"add_generation_prompt\": True}\n    try:\n        r = json.loads(U.urlopen(U.Request(f\"http://127.0.0.1:{port}/tokenize\", data=json.dumps(body).encode(),\n                                           headers={\"Content-Type\": \"application/json\"}), timeout=timeout).read())\n        got = list(r.get(\"tokens\") or [])\n    except Exception as e:\n        return {\"status\": \"ERROR\", \"error\": repr(e)[:200]}\n    want = list(want_ids)\n    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None if len(got) == len(want) else min(len(got), len(want)))\n    return {\"status\": \"MATCH\" if got == want else \"MISMATCH\", \"server_tokens\": len(got), \"want_tokens\": len(want),\n            \"first_diff\": first}\n\n\ndef teacher_forced(port, ids, last=256, model=\"q27\", timeout=1800):\n    \"\"\"rev 3 (measurement M3): prompt logprobs of the last `last` positions of a long prompt (one prefill, 1 token out),\n    for a lossy-KV comparison (NVFP4 vs FP8 KV) at the context lengths that matter. Returns per-position\n    [logprob of the actual token, its rank, the engine's own rank-1 token id] or an error. rev 3h (three GPT SMOKE\n    audits): the rank-1 token id is kept, so top-1 agreement compares the two engines' predictions.\"\"\"\n    body = {\"model\": model, \"prompt\": list(ids), \"max_tokens\": 1, \"temperature\": 0, \"prompt_logprobs\": 1,\n            \"return_tokens_as_token_ids\": True}\n    try:\n        r = json.loads(U.urlopen(U.Request(f\"http://127.0.0.1:{port}/v1/completions\", data=json.dumps(body).encode(),\n                                           headers={\"Content-Type\": \"application/json\"}), timeout=timeout).read())\n        pl = r[\"choices\"][0].get(\"prompt_logprobs\") or []\n    except Exception as e:\n        return {\"error\": repr(e)[:300]}\n    out = []\n    for pos in range(max(1, len(ids) - last), len(ids)):\n        d = pl[pos] if pos < len(pl) else None\n        if not isinstance(d, dict):\n            out.append(None)\n            continue\n        e = d.get(str(ids[pos])) or d.get(f\"token_id:{ids[pos]}\")     # rev 3c: no fallback to another token\n        top1 = [_tok_id(k) for k, v in d.items() if isinstance(v, dict) and v.get(\"rank\") == 1]\n        out.append([e.get(\"logprob\"), e.get(\"rank\"), top1[0] if len(top1) == 1 else None] if isinstance(e, dict) else None)\n    return {\"positions\": out, \"n\": len(out)}\n\n\ndef _tok_id(k):\n    \"\"\"A prompt-logprob key ('123' or 'token_id:123') as an int id; anything else stays a tagged string.\"\"\"\n    m = re.fullmatch(r\"(?:token_id:)?(-?\\d+)\", str(k))\n    return int(m.group(1)) if m else \"str:\" + str(k)\n\n\ndef compare_tf(ref, cand):\n    \"\"\"Teacher-forced comparison: mean / max |delta logprob| of the actual tokens and rank-1 agreement. rev 3d (four\n    GPT audits): MEASURED only when EVERY expected position is valid on both sides (1 of 256 used to be MEASURED);\n    MEASURED is a measurement, not a PASS (the caller applies the A/A-derived tolerance).\"\"\"\n    if not ref or not cand or ref.get(\"error\") or cand.get(\"error\"):\n        return {\"status\": \"INCONCLUSIVE\"}\n    a_, b_ = ref.get(\"positions\") or [], cand.get(\"positions\") or []\n    if not a_ or len(a_) != len(b_):\n        return {\"status\": \"INCONCLUSIVE_INCOMPLETE_TF\", \"ref_n\": len(a_), \"cand_n\": len(b_)}\n    if any(not a or not b or a[0] is None or b[0] is None for a, b in zip(a_, b_)):\n        return {\"status\": \"INCONCLUSIVE_INCOMPLETE_TF\", \"valid\": sum(1 for a, b in zip(a_, b_) if a and b and a[0] is not None\n                                                                     and b[0] is not None), \"expected\": len(a_)}\n    pairs = list(zip(a_, b_))\n    if any(len(a) < 3 or len(b) < 3 or a[2] is None or b[2] is None for a, b in pairs):\n        return {\"status\": \"INCONCLUSIVE_TOP1_MISSING\", \"expected\": len(pairs),\n                \"with_top1\": sum(1 for a, b in pairs if len(a) >= 3 and len(b) >= 3 and a[2] is not None and b[2] is not None)}\n    d = sorted(abs(a[0] - b[0]) for a, b in pairs)\n    # rev 3h (three GPT SMOKE audits): top-1 agreement = both engines predict the same rank-1 token id. rev 3d-3g counted\n    # \"the forced token is rank 1 on both sides or on neither\", which reads 1.0 for ranks 2 vs 3.\n    r1 = sum(1 for a, b in pairs if a[2] == b[2]) / len(pairs)\n    both = sum(1 for a, b in pairs if (a[1] == 1) == (b[1] == 1)) / len(pairs)\n    p99 = d[min(len(d) - 1, int(round(0.99 * (len(d) - 1))))]          # rev 3e: robust to one outlier position\n    return {\"status\": \"MEASURED\", \"n\": len(pairs), \"mean_abs_dlogprob\": round(statistics.mean(d), 4),\n            \"p99_abs_dlogprob\": round(p99, 3), \"max_abs_dlogprob\": round(d[-1], 3), \"rank1_agreement\": round(r1, 4),\n            \"forced_token_rank1_both_or_neither\": round(both, 4)}\n\n\ndef tf_complete(rows):\n    \"\"\"rev 3e: a teacher-forced row set counts as a reference only when every position of every band is valid\n    (rev 3h: including the rank-1 token id).\"\"\"\n    return bool(rows) and all(isinstance(r, dict) and not r.get(\"error\") and r.get(\"positions\") and\n                              all(p and p[0] is not None and len(p) >= 3 and p[2] is not None for p in r[\"positions\"])\n                              for r in rows.values())\n\n\n# ------------------------------------------------------------------ steady decode\ndef steady(port, prompts, level, secs=60.0, admit=None, max_new=8192, temperature=1.0, seed=1234, model=\"q27\",\n           sample=2.0, prefill_tok_s=700.0, gpu_sample=True, barrier=None):\n    \"\"\"`level` distinct prompts, then a decode window that starts only once prefill has settled. Status:\n    STEADY / CAPACITY_LIMITED (fewer running than asked: KV) / INCONCLUSIVE_* / FAIL_*; numbers are kept on an\n    INCONCLUSIVE window so a KV census survives; never a made-up number. `barrier` (steady_multi) starts every replica's\n    window at the same moment.\"\"\"\n    plen = len(prompts[0]) if prompts else 0\n    admit = admit or max(420.0, 120.0 + level * plen / prefill_tok_s)\n    r = {\"level\": level, \"status\": \"INCONCLUSIVE_NOT_STARTED\", \"port\": port, \"prompt_tokens\": plen, \"max_new\": max_new,\n         \"admit_limit_sec\": round(admit)}\n    if not wait_idle(port, 60):\n        r[\"status\"] = \"INCONCLUSIVE_NOT_IDLE_AT_START\"\n        if barrier:\n            barrier.abort()\n        return r\n    try:\n        p0 = prom(port)\n    except Exception as e:\n        r.update(status=\"INCONCLUSIVE_METRICS_UNAVAILABLE\", error=repr(e)[:200])\n        if barrier:\n            barrier.abort()\n        return r\n    if GEN not in p0 or PROMPT not in p0:\n        r[\"status\"] = \"INCONCLUSIVE_COUNTERS_MISSING\"\n        if barrier:\n            barrier.abort()\n        return r\n    ss = [Stream(port, prompts[i % len(prompts)], max_new, temperature, seed + i, model=model) for i in range(level)]\n    for g in ss:\n        g.start()\n    try:\n        t_adm, best, settled, prev, samples, flat_since = time.perf_counter(), 0, 0, None, [], None\n        last_tok = [0] * len(ss)\n        while time.perf_counter() - t_adm < admit:\n            time.sleep(sample)\n            if all(not g.is_alive() for g in ss):          # rev 3: checked before the metrics read (a dead server)\n                r.update(status=\"INCONCLUSIVE_STREAMS_ENDED_IN_ADMISSION\", errors=[g.err for g in ss if g.err][:3])\n                if barrier:\n                    barrier.abort()                       # rev 3c: the other replicas must not wait for this one\n                return r\n            try:\n                m = prom(port)\n            except Exception:\n                continue\n            run, wait, pt, pre = int(m.get(RUN, 0)), int(m.get(WAIT, 0)), m.get(PROMPT), m.get(PREEMPT, 0)\n            now_tok = [len(g.tokens) for g in ss]\n            first = sum(g.t_first is not None for g in ss)\n            gaining = sum(1 for a, b in zip(last_tok, now_tok) if b > a)\n            last_tok = now_tok\n            best = max(best, run)\n            samples.append([round(time.perf_counter() - t_adm, 1), run, wait, first, gaining])\n            # rev 3: no preemption in the sample, every running request emitted and is gaining tokens\n            calm = run > 0 and prev is not None and pt == prev[0] and pre == prev[1] and first >= run and gaining >= run\n            if calm and run >= level and wait == 0:\n                settled += 1\n            elif calm and run < level and wait > 0:       # KV-bound: the rest wait; accept a plateau held 30 s\n                flat_since = flat_since or time.perf_counter()\n                settled = settled + 1 if time.perf_counter() - flat_since > 30 else 0\n            else:\n                settled, flat_since = 0, None\n            prev = (pt, pre)\n            if settled >= 3:\n                break\n        r[\"admission\"] = {\"sec\": round(time.perf_counter() - t_adm, 1), \"best_running\": best, \"tail\": samples[-6:],\n                          \"prefill_tok_s\": round((prom(port).get(PROMPT, 0) - p0.get(PROMPT, 0)) /\n                                                 max(1e-3, time.perf_counter() - t_adm), 1)}\n        if settled < 3:\n            r[\"status\"] = \"INCONCLUSIVE_NO_SETTLE\"\n            if barrier:\n                barrier.abort()\n            return r\n        if barrier:\n            try:\n                barrier.wait(timeout=max(60.0, admit))\n            except threading.BrokenBarrierError:\n                r[\"status\"] = \"INCONCLUSIVE_REPLICA_BARRIER\"\n                return r\n        m0, t0 = prom(port), time.perf_counter()\n        runs, waits, kvs, mids, gmem, stalls = [], [], [], [], [], [0] * len(ss)\n        last_tok, track, misses = [len(g.tokens) for g in ss], None, 0\n        while time.perf_counter() - t0 < secs:\n            time.sleep(min(sample, max(0.1, secs - (time.perf_counter() - t0))))\n            try:\n                m = prom(port)\n            except Exception:                              # rev 3c: one failed metrics read is not a crash\n                misses += 1\n                if misses >= 5:\n                    break\n                continue\n            mids.append(m)\n            runs.append(m.get(RUN, 0))\n            waits.append(m.get(WAIT, 0))\n            if KVU in m:\n                kvs.append(m[KVU])\n            now_tok = [len(g.tokens) for g in ss]\n            if track is None:                              # rev 3c: only streams decoding at window start can stall\n                track = {i for i in range(len(ss)) if now_tok[i] > last_tok[i]}\n            for i, g in enumerate(ss):\n                if i in track and g.is_alive():\n                    stalls[i] = stalls[i] + 1 if now_tok[i] <= last_tok[i] else 0\n            last_tok = now_tok\n            if any(x >= 3 for x in stalls):\n                break\n            if gpu_sample and len(mids) % 5 == 1:\n                g_ = gpu_mem()\n                if g_:\n                    gmem.append(g_)\n        m1, t1 = prom(port), time.perf_counter()\n        gen = m1.get(GEN, 0) - m0.get(GEN, 0)\n        dt = t1 - t0\n        acc, drf, dtk, it = (m1.get(k, 0) - m0.get(k, 0) for k in (ACC, DRF, DTOK, ITER))\n        rm = statistics.mean(runs) if runs else 0\n        steps = it if it else ((drf / rm) if (drf and rm) else ((gen / rm) if rm else 0))\n        r.update(tok_s=round(gen / dt, 1) if dt else None, gen_tokens=int(gen), window_sec=round(dt, 1),\n                 running_mean=round(rm, 2), running_min=min(runs or [0]), running_max=max(runs or [0]),\n                 waiting_min=min(waits or [0]), waiting_max=max(waits or [0]),\n                 tok_s_per_stream=round(gen / dt / rm, 2) if rm and dt else None,\n                 accept_per_draft=round(acc / drf, 3) if drf else None,\n                 mean_acceptance_length=round(1 + acc / drf, 3) if drf else None,\n                 accept_rate=round(acc / dtk, 3) if dtk else None,\n                 steps=int(steps), steps_source=\"iteration_histogram\" if it else \"estimate\",\n                 step_ms=round(dt * 1000 / steps, 1) if steps else None,\n                 tokens_per_step=round(gen / steps, 2) if steps else None,\n                 kv_usage_mean=round(statistics.mean(kvs), 3) if kvs else None, kv_usage_max=max(kvs) if kvs else None,\n                 gpu_mem_peak_mib=[max(row[i][1] for row in gmem if len(row) > i) for i in range(max(map(len, gmem)))] if gmem else None,\n                 preemptions=m1.get(PREEMPT, 0) - m0.get(PREEMPT, 0), t0=t0, t1=t1)\n        if misses >= 5:\n            r[\"status\"] = \"INCONCLUSIVE_METRICS_LOST\"\n        elif any(x >= 3 for x in stalls):\n            r[\"status\"] = \"INCONCLUSIVE_STALLED_STREAM\"\n        elif sum(g.is_alive() for g in ss) < min(level, best):\n            r[\"status\"] = \"INCONCLUSIVE_STREAMS_ENDED_IN_WINDOW\"\n        elif any(x.get(PROMPT) != m0.get(PROMPT) for x in mids + [m1]):\n            r[\"status\"] = \"INCONCLUSIVE_PREFILL_IN_WINDOW\"\n        elif r[\"preemptions\"] > 0:\n            r[\"status\"] = \"INCONCLUSIVE_PREEMPTION\"\n        elif gen <= 0:\n            r[\"status\"] = \"FAIL_NO_DECODE\"\n        elif min(runs or [0]) < level and (len(set(runs)) > 1 or len(set(waits)) > 1):\n            r[\"status\"] = \"INCONCLUSIVE_RUNNING_SET_CHANGED\"      # KV-bound window must hold one fixed set\n        elif min(runs or [0]) < 0.8 * max(runs or [1]):\n            r[\"status\"] = \"INCONCLUSIVE_RUNNING_UNSTABLE\"\n        else:\n            r[\"status\"] = (\"STEADY\" if min(runs or [0]) == max(runs or [0]) == level else\n                           \"CAPACITY_LIMITED\" if max(runs or [0]) < level else \"INCONCLUSIVE_RUNNING_ABOVE_LEVEL\")\n        return r\n    finally:\n        errs = sorted({g.err for g in ss if g.err})[:3]           # rev 3: snapshot before stop() sets _halt\n        for g in ss:\n            g.stop()\n        for g in ss:\n            g.join(10)\n        r[\"client_leak\"] = any(g.is_alive() for g in ss)\n        r[\"stream_errors\"] = errs\n        gb = [degenerate(\"\".join(g.text)[-4000:]) if g.tokens else None for g in ss]\n        r[\"garbage\"] = {\"streams\": len(ss), \"with_output\": sum(1 for g in ss if g.tokens),\n                        \"garbage\": [x for x in gb if x], \"sample_tail\": \"\".join(ss[0].text)[-200:] if ss and ss[0].tokens else \"\"}\n        if r[\"garbage\"][\"garbage\"] and r.get(\"status\") in (\"STEADY\", \"CAPACITY_LIMITED\"):\n            r[\"status\"] = \"FAIL_GARBAGE_UNDER_LOAD\"\n        r[\"idle_after\"] = wait_idle(port)\n\n\ndef steady_multi(jobs, secs=60.0, **kw):\n    \"\"\"Several servers at once (TP1 replicas): jobs = [(port, prompts, level)]; one shared window (barrier). The aggregate\n    counts only when every window is STEADY / CAPACITY_LIMITED and the windows overlap >= 80 % of the shortest.\"\"\"\n    out = [None] * len(jobs)\n    bar = threading.Barrier(len(jobs))\n\n    def one(i, j):\n        try:\n            out[i] = steady(j[0], j[1], j[2], secs=secs, barrier=bar, **kw)\n        except Exception as e:          # rev 3: a dead replica is recorded, not printed to stderr only\n            out[i] = {\"status\": \"ERROR_REPLICA\", \"error\": f\"{type(e).__name__}: {e}\"[:300], \"port\": j[0]}\n            try:\n                bar.abort()\n            except Exception:\n                pass\n    th = [threading.Thread(target=one, args=(i, j), daemon=True) for i, j in enumerate(jobs)]\n    for t in th:\n        t.start()\n    for t in th:\n        t.join()\n    good = all(o and o.get(\"status\") in (\"STEADY\", \"CAPACITY_LIMITED\") for o in out)\n    ov = (min(o[\"t1\"] for o in out) - max(o[\"t0\"] for o in out)) if good else 0\n    aligned = good and ov > 0.8 * min(o[\"window_sec\"] for o in out)\n    return {\"replicas\": out, \"aligned\": aligned, \"overlap_sec\": round(ov, 1),\n            \"tok_s\": round(sum(o[\"tok_s\"] for o in out), 1) if aligned else None,\n            \"running_mean\": round(sum(o[\"running_mean\"] for o in out), 2) if aligned else None,\n            \"status\": \"AGGREGATE_STEADY\" if aligned else \"INCONCLUSIVE_REPLICA_WINDOWS\"}\n\n\ndef wait_idle(port, wait=90.0):\n    end, hits = time.perf_counter() + wait, 0\n    while time.perf_counter() < end:\n        try:\n            m = prom(port)\n            hits = hits + 1 if m.get(RUN, 0) == 0 and m.get(WAIT, 0) == 0 else 0\n        except Exception:\n            hits = 0\n        if hits >= 3:\n            return True\n        time.sleep(2)\n    return False\n\n\n# ------------------------------------------------------------------ profiler (FULL-graph proof + per-category time)\n# rev 3: matched on the kernel BASE name, attention before GDN (\"fla\" is a substring of \"flashinfer\"), real 0.32 GDN names\nCATS = [(\"nccl\", (\"nccl\", \"allreduce\", \"all_reduce\", \"allgather\", \"all_gather\")),\n        (\"attention\", (\"flashinfer\", \"flash_fwd\", \"unified_attention\", \"paged\", \"attn\", \"attention\", \"batchdecode\", \"batchprefill\",\n                   \"reshape_and_cache\")),\n        (\"gdn_linear_attn\", (\"gdn\", \"delta_rule\", \"chunk_\", \"recurrent\", \"causal_conv\", \"l2norm\", \"solve_tril\", \"merge_16x16\",\n                             \"recompute_w_u\", \"kkt\", \"cumsum\", \"post_conv\", \"layer_norm_gated\", \"gated_delta\")),\n        (\"gemm_marlin\", (\"marlin\",)),\n        (\"gemm_other\", (\"gemm\", \"cutlass\", \"xmma\", \"cublas\", \"sm89_\", \"ampere_\", \"sm80_\")),\n        (\"sampling\", (\"sampl\", \"topk\", \"top_k\", \"argmax\", \"softmax\", \"rejection\"))]\n\n\ndef _cat(name):\n    # rev 3h: \"void (anonymous namespace)::gdn_decode_post_conv_mtp_kernel<...>(...)\" used to reduce to \"void \" -> other\n    s = re.sub(r\"\\(anonymous namespace\\)::\", \"\", str(name))\n    s = re.sub(r\"^\\s*void\\s+\", \"\", s)\n    base = s.split(\"<\")[0].split(\"(\")[0].lower()\n    return next((k for k, pats in CATS if any(p in base for p in pats)), \"other\")\n\n\nCOMPUTE_CATS = (\"nccl\", \"attention\", \"gdn_linear_attn\", \"gemm_marlin\", \"gemm_other\")\n# rev 3h: vLLM 0.32 marks each model execution as execute_context_N(N)_generation_N(N) (no gpu_model_runner range)\nSCOPE_RE = re.compile(r\"^execute_context_\\d+\\(\\d+\\)_generation_\\d+\\(\\d+\\)$\")\n\n\ndef _is_scope(e):\n    n = str(e.get(\"name\", \"\"))\n    return SCOPE_RE.match(n) is not None or (\"forward\" in n.lower() and \"gpu_model_runner\" in n.lower())\n\n\ndef _scope_kind(g, compute_eager, unresolved):\n    \"\"\"FULL = graph-launched with no eager model compute (attention / GDN / GEMM / NCCL) inside the execution scope;\n    eager input-preparation kernels are counted apart and allowed. Unresolved eager launches never read FULL.\"\"\"\n    if g == 0:\n        return \"EAGER\"\n    if compute_eager:\n        return \"PIECEWISE\"\n    if unresolved:\n        return \"PIECEWISE\" if g > 1 else \"MIXED\"\n    return \"FULL\"\n\n\ndef trace_summary(pdir):\n    files = sorted(Path(pdir).rglob(\"*.pt.trace.json*\"), key=lambda p: p.stat().st_mtime)\n    if not files:\n        return {\"error\": \"NO_TRACE_FILE\", \"listing\": [str(p) for p in Path(pdir).rglob(\"*\")][:20]}\n    f = next((p for p in files if re.search(r\"rank0\\.|rank[-_]?0[._]|tp0_\", p.name)), files[0])\n    with (gzip.open if f.name.endswith(\".gz\") else open)(f, \"rt\", encoding=\"utf-8\") as fh:\n        tr = json.load(fh)\n    ev = tr.get(\"traceEvents\", [])\n    kern = [e for e in ev if e.get(\"ph\") == \"X\" and str(e.get(\"cat\", \"\")).lower() in (\"kernel\", \"gpu_kernel\")]\n    rt = [e for e in ev if e.get(\"ph\") == \"X\" and str(e.get(\"cat\", \"\")).lower() in (\"cuda_runtime\", \"cuda_driver\")]\n    is_graph = lambda e: \"GraphLaunch\" in str(e.get(\"name\", \"\"))\n    is_kl = lambda e: re.search(r\"LaunchKernel|cuLaunchKernel\", str(e.get(\"name\", \"\"))) is not None\n    # per-scope counts inside the execution annotations (rev 3h: execute_context_*_generation_*; the older\n    # gpu_model_runner forward ranges still count). rev 3c: user_annotation only, not the gpu_user_annotation copies.\n    fw = sorted([e for e in ev if e.get(\"ph\") == \"X\" and str(e.get(\"cat\", \"\")) == \"user_annotation\" and _is_scope(e)],\n                key=lambda e: e.get(\"ts\", 0))\n    kname = {(e.get(\"args\") or {}).get(\"correlation\"): str(e.get(\"name\", \"\")) for e in kern\n             if (e.get(\"args\") or {}).get(\"correlation\") is not None}\n    per_fw, aux = [], {}\n    for a in fw[:400]:\n        lo, hi, tid = a.get(\"ts\", 0), a.get(\"ts\", 0) + a.get(\"dur\", 0), a.get(\"tid\")\n        inside = [e for e in rt if e.get(\"tid\") == tid and lo <= e.get(\"ts\", -1) <= hi]\n        kls = [e for e in inside if is_kl(e)]\n        ce = un = 0\n        for e in kls:\n            kn = kname.get((e.get(\"args\") or {}).get(\"correlation\"))\n            if kn is None:\n                un += 1\n            elif _cat(kn) in COMPUTE_CATS:\n                ce += 1\n            else:\n                b = re.sub(r\"^\\s*void\\s+\", \"\", re.sub(r\"\\(anonymous namespace\\)::\", \"\", kn)).split(\"<\")[0].split(\"(\")[0][:60]\n                aux[b] = aux.get(b, 0) + 1\n        per_fw.append([sum(1 for e in inside if is_graph(e)), len(kls), ce, un])\n    corr = {(e.get(\"args\") or {}).get(\"correlation\"): str(e.get(\"name\", \"\")) for e in rt\n            if (e.get(\"args\") or {}).get(\"correlation\") is not None}\n    per, tot, names = {k: 0.0 for k, _ in CATS + [(\"other\", ())]}, 0.0, {}\n    src = {k: {\"graph\": 0, \"eager\": 0, \"unknown\": 0} for k in per}\n    for e in kern:\n        d = float(e.get(\"dur\", 0.0))\n        tot += d\n        names[e.get(\"name\", \"\")] = names.get(e.get(\"name\", \"\"), 0.0) + d\n        k = _cat(e.get(\"name\", \"\"))\n        per[k] += d\n        launcher = corr.get((e.get(\"args\") or {}).get(\"correlation\"))\n        src[k][\"graph\" if launcher and \"GraphLaunch\" in launcher else (\"eager\" if launcher else \"unknown\")] += 1\n    tot_k = sum(v[\"graph\"] + v[\"eager\"] for v in src.values())\n    eager_share = round(sum(v[\"eager\"] for v in src.values()) / tot_k, 3) if tot_k else None\n    fw_g = statistics.median([x[0] for x in per_fw]) if per_fw else None\n    fw_k = statistics.median([x[1] for x in per_fw]) if per_fw else None\n    kinds_l = [_scope_kind(g, ce, un) for g, k, ce, un in per_fw]\n    if per_fw:                   # rev 3f (GPT rev 3e audit: 2 FULL + 1 EAGER forwards read FULL by the median)\n        kinds = set(kinds_l)\n        verdict = kinds.pop() if len(kinds) == 1 else \"MIXED\"       # every scope must agree\n        basis = \"per_scope_annotation (all scopes; eager input-prep kernels allowed, eager model compute is not)\"\n    elif tot_k:                  # rev 3d (GPT-2 reproduced FULL with 0 forwards): CUPTI alone never proves FULL\n        att = src[\"attention\"]\n        hint = (\"FULL\" if att[\"eager\"] == 0 and (eager_share or 0) < 0.05 else\n                \"PIECEWISE\" if att[\"eager\"] > 0 and any(v[\"graph\"] for v in src.values()) else\n                \"EAGER\" if not any(v[\"graph\"] for v in src.values()) else \"MIXED\")\n        verdict, basis = \"UNPROVEN\", f\"no forward annotations (cupti hint: {hint})\"\n    else:\n        verdict, basis = \"UNKNOWN\", \"none\"\n    return {\"file\": f.name, \"files\": len(files), \"kernels\": len(kern), \"kernel_us_sum\": round(tot),\n            \"graph_launches\": sum(1 for e in rt if is_graph(e)), \"kernel_launches\": sum(1 for e in rt if is_kl(e)),\n            \"forwards\": len(per_fw), \"forward_kinds\": {k_: kinds_l.count(k_) for k_ in (\"FULL\", \"PIECEWISE\", \"EAGER\", \"MIXED\")},\n            \"per_forward_median_graph_launches\": fw_g, \"per_forward_median_kernel_launches\": fw_k,\n            \"per_forward_median_compute_eager\": statistics.median([x[2] for x in per_fw]) if per_fw else None,\n            \"per_forward_unresolved_eager\": sum(x[3] for x in per_fw),\n            \"scope_names\": sorted({str(a.get(\"name\", \"\")) for a in fw})[:5],\n            \"aux_eager_kernels_in_scopes\": dict(sorted(aux.items(), key=lambda x: -x[1])[:20]),\n            \"launch_source\": src, \"eager_kernel_share\": eager_share, \"graph_verdict\": verdict, \"verdict_basis\": basis,\n            \"by_category_pct\": {k: round(100 * v / tot, 1) for k, v in per.items()} if tot else {},\n            \"top_kernels_us\": dict(sorted(((k, round(v)) for k, v in names.items()), key=lambda x: -x[1])[:25]),\n            \"note\": \"kernel time sums overlapping streams; NCCL time includes rank-skew waiting\"}\n\n\ndef profile(port, pdir, secs=6.0, delay_iters=20, max_iters=40):\n    \"\"\"POST /start_profile (bounded by iterations; falls back to a bare POST), decode, POST /stop_profile, summary.\"\"\"\n    rec = {}\n    try:\n        U.urlopen(U.Request(f\"http://127.0.0.1:{port}/start_profile\", method=\"POST\",\n                            data=json.dumps({\"delay_iterations\": delay_iters, \"max_iterations\": max_iters}).encode(),\n                            headers={\"Content-Type\": \"application/json\"}), timeout=900).read()\n        rec[\"start_profile\"] = \"ok_bounded\"\n    except Exception as e:\n        rec[\"start_profile_bounded_error\"] = repr(e)[:200]\n        try:\n            U.urlopen(U.Request(f\"http://127.0.0.1:{port}/start_profile\", data=b\"\", method=\"POST\"), timeout=900).read()\n            rec[\"start_profile\"] = \"ok\"\n        except Exception as e2:\n            rec[\"start_profile\"] = repr(e2)[:200]\n    time.sleep(secs)\n    try:\n        U.urlopen(U.Request(f\"http://127.0.0.1:{port}/stop_profile\", data=b\"\", method=\"POST\"), timeout=900).read()\n        rec[\"stop_profile\"] = \"ok\"\n    except Exception as e:\n        rec[\"stop_profile\"] = repr(e)[:200]\n    time.sleep(20)\n    try:\n        rec[\"trace\"] = trace_summary(pdir)\n    except Exception as e:\n        rec[\"trace\"] = {\"error\": f\"TRACE_PARSE {type(e).__name__}: {e}\"[:300]}\n    return rec\n\n\ndef profiled_load(port, prompts, level, pdir, secs=6.0, max_new=8192, model=\"q27\", prefill_tok_s=700.0, spec=False):\n    \"\"\"`level` decoding streams (own window: the profiler slows decode), profile() only after prefill settled exactly\n    as in steady() (prompt counter + preemptions flat, every running request emitted and gaining tokens). rev 3d: no\n    profile when unsettled; the profile window must be decode-only (prompt counter flat) and, with `spec`, show\n    speculative drafts (else the trace cannot speak for multi-token verification).\"\"\"\n    plen = len(prompts[0]) if prompts else 0\n    settle = max(300.0, 120.0 + level * plen / prefill_tok_s)\n    ss = [Stream(port, prompts[i % len(prompts)], max_new, 1.0, 4321 + i, model=model) for i in range(level)]\n    for g in ss:\n        g.start()\n    rec, t0, flat, ok, prev, last_tok = {\"level\": level}, time.perf_counter(), None, 0, None, [0] * level\n    try:\n        while time.perf_counter() - t0 < settle:\n            time.sleep(2)\n            try:\n                m = prom(port)\n            except Exception:\n                continue\n            run, wait, pt, pre = int(m.get(RUN, 0)), int(m.get(WAIT, 0)), m.get(PROMPT), m.get(PREEMPT, 0)\n            now_tok = [len(g.tokens) for g in ss]\n            gaining = sum(1 for a, b in zip(last_tok, now_tok) if b > a)\n            last_tok = now_tok\n            first = sum(g.t_first is not None for g in ss)\n            calm = run > 0 and prev is not None and (pt, pre) == prev and first >= run and gaining >= run\n            if calm and run >= level and wait == 0:\n                ok += 1\n            elif calm and wait > 0:\n                flat = flat or time.perf_counter()\n                ok = ok + 1 if time.perf_counter() - flat > 30 else 0\n            else:\n                ok, flat = 0, None\n            prev = (pt, pre)\n            if ok >= 3:\n                break\n        rec[\"settled\"] = ok >= 3\n        if not rec[\"settled\"]:\n            rec[\"status\"] = \"INCONCLUSIVE_PREFILL_NOT_SETTLED\"\n            return rec\n        try:\n            m0 = prom(port)\n            rec[\"running_at_start\"] = m0.get(RUN)\n        except Exception:\n            m0 = {}\n        rec.update(profile(port, pdir, secs))\n        try:\n            m1 = prom(port)\n        except Exception:\n            m1 = {}\n        rec[\"window_deltas\"] = {\"prompt_tokens\": m1.get(PROMPT, 0) - m0.get(PROMPT, 0), \"drafts\": m1.get(DRF, 0) - m0.get(DRF, 0),\n                                \"accepted\": m1.get(ACC, 0) - m0.get(ACC, 0), \"generated\": m1.get(GEN, 0) - m0.get(GEN, 0)}\n        if not m0 or not m1:\n            rec[\"status\"] = \"INCONCLUSIVE_METRICS_UNAVAILABLE\"\n        elif rec[\"window_deltas\"][\"prompt_tokens\"]:\n            rec[\"status\"] = \"INCONCLUSIVE_PREFILL_IN_PROFILE\"\n        elif spec and not (rec[\"window_deltas\"][\"drafts\"] and rec[\"window_deltas\"][\"accepted\"]):\n            rec[\"status\"] = \"INCONCLUSIVE_NO_SPEC_DECODE\"            # rev 3e: drafts AND accepted tokens in the window\n        return rec\n    finally:\n        for g in ss:\n            g.stop()\n        for g in ss:\n            g.join(10)\n        rec[\"idle_after\"] = wait_idle(port)\n"
    (TMP / "e23_lab.py").write_text(LAB_SRC)
    sys.path.insert(0, str(TMP))
    import e23_lab as L
    REQ_LOG = TRACES / "requests.jsonl"                           # memory rule: the full text of EVERY request
    _stream_run = L.Stream.run

    def _traced_run(self):
        try:
            _stream_run(self)
        finally:
            if not self.err and any(m is not None and not (isinstance(m, (int, float)) and math.isfinite(m))
                                    for m in self.margins):
                self.err = "NONFINITE_LOGPROB"           # final audit H2: NaN / Inf logprobs are never data
            try:
                r = self.row()
                pr = self.body.get("prompt") or []
                r.update(port=self.port, sampling={k: v for k, v in self.body.items() if k != "prompt"}, prompt_len=len(pr),
                         prompt_sha=hashlib.sha256(json.dumps(pr).encode()).hexdigest()[:16],
                         ttft=round(self.t_first - self.t_start, 2) if self.t_first and self.t_start else None,
                         sec=round(self.t_end - self.t_start, 2) if self.t_end and self.t_start else None, t=time.time())
                with _SAVE_LOCK, open(REQ_LOG, "a") as fh:
                    fh.write(json.dumps(r) + "\n")
            except Exception:
                pass
    L.Stream.run = _traced_run
    PLE_EXPECTED = json.loads(r"""{"model-plefp8-00000.safetensors": [5200027199, "dc2e845b7edd35bda92834fba3626bf7d199e28d6aceac99fee654aade390cfd"], "model-plefp8-00001.safetensors": [5200027202, "899eaa0716e28594468a1389ee58cb065c23907f1270de3831f5ecb0a4f82d56"], "model-plefp8-00002.safetensors": [5200027198, "06fd8a11abf0419a669f89397b8d70dd6ff42d401e6b2a037c65e49704faaf71"], "model-plefp8-00003.safetensors": [5200027189, "c6aaa1fc08e84eced3c8151ac8679ed943888eba0fcef2556963693430f95bd9"], "model-plefp8-00004.safetensors": [5200027190, "d94e97c96d3ea09208614da016960f3f4b429f47a044c898a51b493f42f74ba2"], "model-plefp8-00005.safetensors": [5200027190, "03d5d4792e14a4ab55bae50bb459624b01786f818bbcfcaed1a5d0235af484c6"], "model-plefp8-00006.safetensors": [5200027190, "586cccbc12383021bc9bc02f206d9b19fbd1672373ed9a5d91e3b0ce34c2418f"], "model-plefp8-00007.safetensors": [5200027190, "c4a23bcc10f3cde6b633e82c282a9a518e3a64d433628283e1bc592c94cf3d6c"], "model-plefp8-00008.safetensors": [5200027190, "2dc8098c0d020bff277c9cf499a6b908e17836b70a5f949dfa24793371c9a87e"], "model-plefp8-00009.safetensors": [4400023163, "61de98b89bb79f386795787d7a76827a26f1e292c26edbb0a1b613da455f5a9c"]}""")                     # 10 FP8 PLE shards: [size, sha256]
    # exact Kaggle mount paths (foysalemonshanto Models, checked through the API 2026-10-31; no recursive search)
    MD = INP / "models" / "foysalemonshanto"
    PART_DIRS = {1: MD / "qwen3-8-flash-next-w4a16-merlin-p1/transformers/default/1",
                 2: MD / "qwen3-8-flash-next-w4a16-merlin-p2/transformers/default/1",
                 3: MD / "qwen3-8-flash-next-w4a16-merlin-p3/transformers/default/1",
                 4: MD / "qwen3-8-flash-next-w4a16-merlin-p4/transformers/default/1",
                 5: MD / "qwen3-8-flash-next-w4a16-merlin-p5/transformers/default/1"}
    PARTS = {}
    for k, d in PART_DIRS.items():                                  # the five checkpoint parts (Models from the kernel outputs)
        mf = d / f"q27_fn_merlin_part{k}_manifest.json"
        PARTS[k] = (d, json.loads(mf.read_text())) if mf.is_file() else None
    PLE_DIR = MD / "qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1"      # our FP8 PLE overlay = albucino's shards
    PLE_DIR = PLE_DIR if (PLE_DIR / "model-plefp8-00000.safetensors").is_file() else None
    MERLIN_PAY = MD / "e23-merlin-env/other/v1/1/merlin_offline_tar.bin"
    MERLIN_PAY = MERLIN_PAY if MERLIN_PAY.is_file() else None
    INTEL = INP / "models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"
    INTEL = INTEL if (INTEL / "config.json").is_file() else None
    WHEELHOUSE = INP / "datasets/foysalemonshanto/vllm-0-32-0-cu13-wheelhouse/install_wheelhouse.py"
    WHEELHOUSE = WHEELHOUSE if WHEELHOUSE.is_file() else None
    ARC = COMP / "arc-agi_evaluation_challenges.json"
    if not ARC.is_file():
        raise FileNotFoundError(f"{ARC}: attach the competition data")
    MERLIN_OK = all(PARTS.values()) and PLE_DIR is not None and MERLIN_PAY is not None
    R["inputs"] = {"parts": {k: (str(v[0]) if v else None) for k, v in PARTS.items()}, "ple_dir": str(PLE_DIR),
                   "merlin_pay": str(MERLIN_PAY), "intel": str(INTEL), "wheelhouse": str(WHEELHOUSE), "arc": str(ARC),
                   "merlin_inputs_ok": MERLIN_OK}
    ABORT = {"why": None}
    NVSMI_OK = bool(L.gpu_mem()) and not FAKE
    R["setup"]["gpus_at_start"] = L.gpu_mem()
    if not FAKE and not os.environ.get("RC2_CPU_KIT") and len(R["setup"]["gpus_at_start"] or []) != 4:   # CPU kit: no GPUs
        R["errors"].append(f"expected 4 GPUs, nvidia-smi shows {R['setup']['gpus_at_start']}")
        save()
        raise RuntimeError("GPU_PREFLIGHT_FAILED: " + repr(R["setup"]["gpus_at_start"]))
    L.start_watchdog(HARD_STOP, lambda: ABORT.update(why="HARD_STOP reached: watchdog killed live servers"))
    log("RC2 | budget", BUDGET // 60, "min |", json.dumps(R["inputs"])[:900])
    save()


if not RERUN:
    # ==================================================================== CELL 2 — host facts, checkpoint merge, engines
    def run(name, cmd, timeout=3600, env=None):
        t = time.time()
        try:
            with open(LOGS / f"{name}.log", "w") as f:
                r = subprocess.run(["bash", "-c", "set -o pipefail; " + cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                                   env={**os.environ, **(env or {})})
            rc = r.returncode
        except subprocess.TimeoutExpired:
            rc = -9
        txt = (LOGS / f"{name}.log").read_text(errors="replace")
        R["setup"][name] = {"rc": rc, "sec": round(time.time() - t, 1), "tail": txt[-1500:]}
        log(name, "rc", rc, f"{time.time() - t:.0f}s")
        save()
        return rc, txt


    SHIM = TMP / "e23_libcuda"          # FlashInfer JIT links -lcuda; Kaggle has only libcuda.so.1
    SHIM.mkdir(exist_ok=True)
    for d in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/local/cuda/compat", "/usr/lib64"):
        c = sorted(Path(d).glob("libcuda.so.1")) if Path(d).is_dir() else []
        if c and not (SHIM / "libcuda.so").exists():
            (SHIM / "libcuda.so").symlink_to(c[0])


    def host_facts():
        """A0: what the Kaggle box gives us (glibc, driver, GPUs, PCIe, peer access, cgroup memory, /dev/shm, disk)."""
        f = {}
        for k, cmd in (("glibc", "ldd --version | head -1"), ("uname", "uname -a"),
                       ("nvidia_smi", "nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader"),
                       ("cuda_version", "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"), ("shm", "df -h /dev/shm | tail -1"),
                       ("tmp_disk", f"df -h {TMP} | tail -1"), ("nproc", "nproc"),
                       ("cgroup_max", "cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes"),
                       ("cgroup_cur", "cat /sys/fs/cgroup/memory.current 2>/dev/null || cat /sys/fs/cgroup/memory/memory.usage_in_bytes"),
                       ("meminfo", "grep -E 'MemTotal|MemAvailable' /proc/meminfo"), ("ulimit_l", "ulimit -l"),
                       ("omp", "echo OMP_NUM_THREADS=$OMP_NUM_THREADS"), ("topo", "nvidia-smi topo -m; nvidia-smi topo -p2p r"),
                       ("pcie_now", "nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.width.current,power.limit --format=csv,noheader")):
            try:
                f[k] = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60).stdout.strip()[:600]
            except Exception as e:
                f[k] = repr(e)[:200]
        try:
            import torch
            n = torch.cuda.device_count()
            f["peer_access"] = [[torch.cuda.can_device_access_peer(i, j) if i != j else None for j in range(n)] for i in range(n)]
        except Exception as e:
            f["peer_access"] = repr(e)[:200]
        return f


    def host_mem():
        """MiB. avail_mib = max - (current - inactive_file), Merlin's own rule (RC1's guard counted page cache as used and
        skipped arm 3 with MemAvailable 55 GB); /dev/shm counts against the cgroup."""
        out = {}
        try:
            cur = int(Path("/sys/fs/cgroup/memory.current").read_text())
            mx = Path("/sys/fs/cgroup/memory.max").read_text().strip()
            st = dict(l.split() for l in Path("/sys/fs/cgroup/memory.stat").read_text().splitlines() if len(l.split()) == 2)
            out.update(cgroup_cur_mib=cur >> 20, cgroup_max_mib=None if mx == "max" else int(mx) >> 20,
                       **{k + "_mib": int(st[k]) >> 20 for k in ("anon", "file", "shmem", "inactive_file") if k in st})
            if mx != "max":
                out["avail_mib"] = (int(mx) - (cur - int(st.get("inactive_file", 0)))) >> 20
        except Exception:
            pass
        try:
            for l in open("/proc/meminfo"):
                if l.startswith(("MemAvailable", "MemTotal")):
                    out[l.split(":")[0] + "_mib"] = int(l.split()[1]) >> 10
        except Exception:
            pass
        try:
            out["dev_shm_used_mib"] = shutil.disk_usage("/dev/shm").used >> 20
        except Exception:
            pass
        return out


    TELEMETRY = OUT / "telemetry.jsonl"


    def _telemetry():
        """Host RAM + GPU memory every 10 s for the whole run (Merlin's floor ~69 GiB inside a ~173 GiB cgroup)."""
        while True:
            try:
                with open(TELEMETRY, "a") as fh:
                    fh.write(json.dumps({"t": round(time.time() - T_START, 1), "host": host_mem(), "gpu": L.gpu_mem()}) + "\n")
            except Exception:
                pass
            time.sleep(10)


    threading.Thread(target=_telemetry, daemon=True).start()


    SHM = Path(os.environ.get("RC2_SHM", "/dev/shm"))


    def shm_names():
        try:
            return {p.name for p in SHM.iterdir()}
        except Exception:
            return set()


    def shm_cleanup(before):
        """After a boot's server is dead: only /dev/shm entries that appeared during THIS boot (absent from `before`) and
        match the engine's prefixes are unlinked; everything else is recorded, never touched."""
        out = {"host_before": host_mem(), "removed": [], "kept": []}
        try:
            for p in sorted(SHM.iterdir()):
                try:
                    sz = p.stat().st_size >> 20
                except Exception:
                    sz = None
                if p.name not in before and re.match(r"(torch_|psm_|vllm|ple|nccl|sem\.mp-|cuda\.shm)", p.name, re.I):
                    try:
                        p.unlink() if not p.is_dir() else shutil.rmtree(p)
                        out["removed"].append([p.name, sz])
                    except Exception as e:
                        out["kept"].append([p.name, sz, repr(e)[:80]])
                else:
                    out["kept"].append([p.name, sz])
        except Exception as e:
            out["error"] = repr(e)[:200]
        out["host_after"] = host_mem()
        out["removed"], out["kept"] = out["removed"][:60], out["kept"][:60]
        return out


    class MServer(L.Server):
        """Merlin passes --shutdown-timeout 60 and the PLE transport unlinks its /dev/shm storages only on a graceful
        teardown: SIGTERM, wait 75 s, then E23's kill (SIGTERM + 40 s + SIGKILL of the whole session)."""
        def kill(self):
            if self.proc is not None and self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, 15) if hasattr(os, "killpg") else self.proc.terminate()
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=75)
                except Exception:
                    pass
            super().kill()


    def boot_gate():
        """Merlin's pin leaves ~0.9 GiB free per card: every boot needs near-empty GPUs (< 150 MiB, not E23's 1024)."""
        g = L.wait_gpu_release(300, fake=bool(FAKE), gpus="0,1,2,3", free_mib=150)
        return {"ok": g, "gpus": L.gpu_mem(), "host": host_mem()}       # callers launch only on ok is True


    CKPT = TMP / "fn_merlin"
    SHA = {"thread": None, "ok": None, "bad": [], "sec": None}


    def merge_ckpt():
        """Symlink the 5 parts + 10 PLE shards into one directory; names and sizes now, sha256 in a background thread."""
        SHA.update(thread=None, ok=None, bad=[], sec=None)        # a stale result never gates a new merge
        shutil.rmtree(CKPT, ignore_errors=True)
        CKPT.mkdir(parents=True)
        expected = {}
        for k, (d, man) in PARTS.items():
            for n, (size, sha) in man["files"].items():
                expected[n] = (d / man.get("model_subdir", ".") / n, size, sha)
        for n, (size, sha) in PLE_EXPECTED.items():
            expected[n] = (PLE_DIR / n, size, sha)
        bad = []
        for n, (src, size, sha) in sorted(expected.items()):
            if not src.is_file():
                bad.append(f"missing {n}")
                continue
            if not FAKE and src.stat().st_size != size:
                bad.append(f"{n}: {src.stat().st_size} bytes, expected {size}")
            try:
                (CKPT / n).symlink_to(src)
            except OSError:
                if not FAKE:
                    raise
                shutil.copyfile(src, CKPT / n)               # local Windows harness only (no symlink privilege)
        idx = json.loads((CKPT / "model.safetensors.index.json").read_text())
        need = set(idx["weight_map"].values())
        bad += [f"index needs {n}, not merged" for n in sorted(need - set(expected))]
        cfg = json.loads((CKPT / "config.json").read_text())
        if (cfg.get("text_config") or {}).get("ple_embedding_dtype") != "float8_e4m3fn":
            bad.append("config.json: text_config.ple_embedding_dtype is not float8_e4m3fn")
        rec = {"files": len(expected), "index_files": len(need), "bad": bad[:20], "ok": not bad}

        def hash_all():
            t = time.time()
            import concurrent.futures as CF

            def one(item):
                n, (src, size, sha) = item
                if FAKE:
                    return n, True
                with open(src, "rb") as fh:
                    return n, hashlib.file_digest(fh, "sha256").hexdigest() == sha
            with CF.ThreadPoolExecutor(6) as ex:
                res = list(ex.map(one, sorted(expected.items())))
            SHA.update(ok=all(ok for _, ok in res), bad=[n for n, ok in res if not ok], sec=round(time.time() - t, 1))
            R["setup"]["sha256"] = {k: SHA[k] for k in ("ok", "bad", "sec")}
            save()
        if rec["ok"]:
            SHA["thread"] = threading.Thread(target=hash_all, daemon=True)
            SHA["thread"].start()
        return rec


    def sha_gate(timeout):
        """The first launch waits for the full sha256 check (Merlin itself never hashes shards)."""
        th = SHA["thread"]
        if th is not None:
            th.join(max(1, timeout))
        if th is None or th.is_alive():
            return False, "sha256 check not finished"
        return bool(SHA["ok"]), ("ok" if SHA["ok"] else f"sha256 mismatch: {SHA['bad'][:5]}")


    def prefetch_bg(paths, tag):
        """rev 3: warm the page cache of a cold 17-30 GB /kaggle/input checkpoint in the background (many readers)."""
        def go():
            R["setup"]["prefetch_" + tag] = L.prefetch([p for p in paths if p]) if not FAKE else {"fake": True}
            save()
        th = threading.Thread(target=go, daemon=True)
        th.start()
        return th


    MERLIN_ENG = None


    def merlin_install():
        """P2 payload (.bin = a tar Kaggle does not unpack) -> install_merlin_offline.sh (UV_OFFLINE); payload deleted after;
        a failure is cached so later Merlin steps fail fast (rev 3)."""
        global MERLIN_ENG
        if MERLIN_ENG is not None:
            if MERLIN_ENG.get("failed"):
                raise RuntimeError("Merlin install failed earlier: " + MERLIN_ENG["failed"])
            return MERLIN_ENG
        if not MERLIN_OK or MERLIN_PAY is None:
            MERLIN_ENG = {"failed": "Merlin checkpoint / env not attached or incomplete"}
            raise RuntimeError(MERLIN_ENG["failed"])
        if FAKE:
            MERLIN_ENG = {"fake": True}
            return MERLIN_ENG
        try:
            pay_root = TMP / "merlin_pay"
            shutil.rmtree(pay_root, ignore_errors=True)
            pay_root.mkdir()
            t = time.time()
            with tarfile.open(MERLIN_PAY) as tf:
                tf.extractall(pay_root, filter="data")
            R["setup"]["merlin_extract_sec"] = round(time.time() - t, 1)
            pay = pay_root / "merlin_offline"
            shutil.rmtree(TMP / "merlin_off", ignore_errors=True)
            rc, txt = run("install_merlin", f"bash {pay}/install_merlin_offline.sh {pay} {TMP}/merlin_off", timeout=5400)
            kv = dict(re.findall(r"^(MERLIN_[A-Z]+)=(.*)$", txt, re.M))
            shutil.rmtree(pay_root, ignore_errors=True)
            if rc or "INSTALL OK" not in txt:
                raise RuntimeError("Merlin offline install failed (logs/install_merlin.log)")
            sys.path.insert(0, kv["MERLIN_TREE"] + "/tools/merlin")
            import launch as ML
            site = subprocess.run([kv["MERLIN_VENV"] + "/bin/python", "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                                  capture_output=True, text=True).stdout.strip()
            MERLIN_ENG = {"kv": kv, "ML": ML, "cuda_home": site + "/nvidia/cu13"}
            return MERLIN_ENG
        except Exception as e:
            MERLIN_ENG = {"failed": f"{type(e).__name__}: {e}"[:300]}
            raise


    MERLIN_CC = {"mode": "gcc13"}       # rev 3c: Merlin README:229 / install-env.sh: FlashInfer JIT at first start needs GCC >= 13



    KV_FP8, KV_BF16 = 2479838700, 2464415744      # Merlin's own L4 pins (23,034 MiB cards; preflight.kv_bytes_for_vram, 150 / 79 blocks)
    PROFILE = "fn-quad-24g-sm89"
    XHIGH = json.dumps({"enable_thinking": True, "reasoning_effort": "xhigh"})
    HOST_PHASE = {}


    def merlin_fn_server(name, kv, port, kv_bytes):
        """Merlin's launch.build() for Flash-Next on the L4 profile: KV pin lowered for 22.49 GiB cards, the checkpoint's own
        chat template at xhigh (Merlin passes medium otherwise), Merlin's host_phase checks recorded, E23's env overrides."""
        eng = merlin_install()
        extra = ["--served-model-name", "q27", "--max-model-len", str(MAX_LEN), "--port", str(port),
                 "--default-chat-template-kwargs", XHIGH] + list(NB1_EXTRA.get(name) or [])   # E25 NB1: per-boot flags (last wins)
        if FAKE:
            return MServer(name, [sys.executable, FAKE, str(CKPT) + "-w4a16", "--tensor-parallel-size", "2", "--max-num-seqs",
                                  "8" if kv == "fp8" else "3", "--port", str(port), "--max-model-len", str(MAX_LEN),
                                  "--kv-cache-memory-bytes", str(kv_bytes)] + list(NB1_EXTRA.get(name) or []),   # E25 NB1 (fake only)
                           {**os.environ, **(NB1_ENV.get(name) or {})}, port, LOGS)
        kvv = eng["kv"]
        gcc_bin = str(Path(kvv["MERLIN_CC"]).parent) if kvv.get("MERLIN_CC") else ""
        base = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_PATH", "CUDA_HOME", "LD_PRELOAD", "CC", "CXX")}
        base.update(GPUS="0,1,2,3", PORT=str(port), KV_BYTES=str(kv_bytes), P2P="off",
                    VENV=kvv["MERLIN_VENV"], TREE=kvv["MERLIN_TREE"], CHAT_TEMPLATE="", WARM="0", CACHE_ROOT=f"{TMP}/merlin-cache",
                    PATH=f"{kvv['MERLIN_VENV']}/bin:{gcc_bin}:{eng['cuda_home']}/bin:{os.environ.get('PATH', '')}",
                    CUDA_HOME=eng["cuda_home"])
        if kv == "fp8":
            base["SCALES"] = str(CKPT / "qsa_kv_scales_262k.json")
        base.update((globals().get("NB4T_BUILD_ENV") or {}).get(name) or {})   # E25 NB4T: launcher knobs (DRAFT_HEAD) into build()
        plan = eng["ML"].build("flash-next", PROFILE, str(CKPT), base, extra=extra, kv=kv, checkpoint="flash-next", status="untested")
        try:                                               # audits: tree products, venv, checkpoint marker, PLE dtype, toolchain,
            eng["ML"].host_phase(plan)                     # scales file: a failure stops this boot before any server starts
            HOST_PHASE[name] = "ok"
        except BaseException as e:
            HOST_PHASE[name] = f"{type(e).__name__}: {e}"[:600]
            R["setup"].setdefault("host_phase_error", {})[name] = HOST_PHASE[name]
            save()
            raise RuntimeError("MERLIN_HOST_PHASE_FAILED: " + HOST_PHASE[name])
        env = dict(plan.env)
        env.update(LD_PRELOAD=kvv["MERLIN_SHIM"], CUDA_HOME=eng["cuda_home"], VLLM_USE_RUST_FRONTEND="0", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", LIBRARY_PATH=f"{SHIM}:{os.environ.get('LIBRARY_PATH', '')}",
                   PATH=base["PATH"], VLLM_CACHE_ROOT=f"{TMP}/merlin-cache", TRITON_CACHE_DIR=f"{TMP}/merlin-triton",
                   FLASHINFER_WORKSPACE_BASE=f"{TMP}/merlin-fi", CUDA_DEVICE_ORDER="PCI_BUS_ID", VLLM_PLE_OFFLOAD_READY_TIMEOUT="3600")
        for k in ("CC", "CXX", "CUDAHOSTCXX"):
            env.pop(k, None)
        if MERLIN_CC["mode"] == "gcc13" and kvv.get("MERLIN_CC") and kvv.get("MERLIN_CXX"):
            env.update(CC=kvv["MERLIN_CC"], CXX=kvv["MERLIN_CXX"], CUDAHOSTCXX=kvv["MERLIN_CXX"])
            lib13 = Path(kvv["MERLIN_CC"]).parents[1] / "lib"
            if (lib13 / "libstdc++.so.6").exists():
                env["LD_LIBRARY_PATH"] = f"{lib13}:{env.get('LD_LIBRARY_PATH', '')}"
        try:                                               # Merlin's own scheme: PORT 8000..8355 -> 29500+d / 30000+100d
            mp, vp = eng["ML"].distributed_ports(str(port))
            env.update(MASTER_PORT=str(mp), VLLM_PORT=str(vp))
        except Exception:
            env.update(MASTER_PORT=str(29500 + port - 8000), VLLM_PORT=str(30000 + 100 * (port - 8000)))
        env.update(NB1_ENV.get(name) or {})              # E25 NB1: after build(), which strips every MERLIN_* variable
        argv = plan.argv()
        HELP[name] = L.help_check("merlin", argv[:argv.index("serve") + 1], env, argv)
        R["setup"].setdefault("merlin_plans", {})[name] = {"argv_tail": argv[-60:], "status": plan.status, "host_phase": HOST_PHASE[name],
                                                           "env_keys": {k: env[k] for k in sorted(env) if k.startswith(("VLLM_", "NCCL_", "KV_", "P2P", "SCALES", "CUDA_DEVICE_MAX_CONNECTIONS", "DRAFT_HEAD"))}}   # E25 NB4S / NB4T
        save()
        return MServer(name, argv, env, port, LOGS)


    STOCK = None


    def stock_install():
        """Fallback only: stock vLLM 0.32 venv (E18 / E23 installer) with the -lcuda shim on LIBRARY_PATH (RC1's sampler crash)."""
        global STOCK
        if STOCK is not None:
            return STOCK
        if FAKE:
            STOCK = {"vllm": [sys.executable, FAKE], "env": dict(os.environ)}
            return STOCK
        rc, txt = run("install_stock", f"{sys.executable} {WHEELHOUSE} {TMP}/vllm-venv", timeout=2400)
        kv = dict(l.split("=", 1) for l in txt.splitlines() if l.startswith(("VENV_PYTHON=", "CUDA_HOME=")))
        if rc or "CUDA_HOME" not in kv:
            raise RuntimeError("stock vLLM 0.32 install failed (logs/install_stock.log)")
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_PATH", "LD_PRELOAD")}
        env.update(CUDA_HOME=kv["CUDA_HOME"], PATH=f"{TMP}/vllm-venv/bin:{kv['CUDA_HOME']}/bin:{env.get('PATH', '')}",
                   LIBRARY_PATH=f"{SHIM}:{env.get('LIBRARY_PATH', '')}", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   VLLM_NO_USAGE_STATS="1", VLLM_CACHE_ROOT=f"{TMP}/vllm-cache", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        STOCK = {"vllm": [f"{TMP}/vllm-venv/bin/vllm", "serve"], "env": env}
        return STOCK


    def stock_fn_server(name, port):
        eng = stock_install()
        flags = [str(INTEL), "--served-model-name", "q27", "--host", "127.0.0.1", "--port", str(port),
                 "--tensor-parallel-size", "4", "--enable-expert-parallel", "--max-model-len", "32768", "--max-num-seqs", "4",
                 "--max-num-batched-tokens", "4096", "--gpu-memory-utilization", "0.88", "--disable-custom-all-reduce",
                 "--language-model-only", "--engram-config", json.dumps({"cpu_offload": True}), "--load-format", "safetensors",
                 "--dtype", "bfloat16", "--kv-cache-dtype", "fp8_e4m3", "--default-chat-template-kwargs", XHIGH]
        # diagnostic only: RC1 had a 30,208-token BF16 pool at TP4; no MTP here, so prefix caching (default ON) is unaffected
        if FAKE:
            flags = [str(INTEL) + "-w4a16", "--tensor-parallel-size", "4", "--max-num-seqs", "4", "--port", str(port), "--max-model-len", "32768"]
        return L.Server(name, list(eng["vllm"]) + flags, eng["env"], port, LOGS)
    save()


if not RERUN:
    # ==================================================================== CELL 3 — prompts (Flash-Next tokenizer + its own template at xhigh)
    R["sections"]["A0"] = {}
    _t = time.time()
    R["sections"]["A0"]["host"] = host_facts()
    R["sections"]["A0"]["merge"] = merge_ckpt() if MERLIN_OK else {"ok": False, "why": "Merlin inputs missing", "inputs": R["inputs"]}
    R["sections"]["A0"]["host_mem"] = host_mem()
    log("A0 host", json.dumps(R["sections"]["A0"]["host"])[:1200])
    log("A0 merge", json.dumps(R["sections"]["A0"]["merge"])[:800])
    save()
    TOKDIR = TMP / "fn_tok"
    shutil.rmtree(TOKDIR, ignore_errors=True)
    TOKDIR.mkdir()
    # the Merlin checkpoint's tokenizer when the merge is OK; otherwise the Intel (stock fallback) checkpoint's own
    MERLIN_PROMPTS = bool(R["sections"]["A0"]["merge"].get("ok")) and (CKPT / "tokenizer.json").exists()
    _src = CKPT if MERLIN_PROMPTS else INTEL
    if _src is None or not (_src / "tokenizer.json").exists():
        raise FileNotFoundError("no tokenizer: the Merlin merge failed and the Intel fallback checkpoint is not attached")
    for _n in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (_src / _n).exists():
            shutil.copyfile(_src / _n, TOKDIR / _n)
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer
    TOK = Tokenizer.from_file(str(TOKDIR / "tokenizer.json"))
    _HF = AutoTokenizer.from_pretrained(str(TOKDIR))
    _tpl = ((TOKDIR / "chat_template.jinja").read_text() if (TOKDIR / "chat_template.jinja").exists()
            else json.loads((TOKDIR / "tokenizer_config.json").read_text())["chat_template"])
    _tasks = json.load(open(ARC))
    _ids = sorted(_tasks)


    def _grid(g):
        return "\n".join("".join(str(v) for v in row) for row in g)


    def _task_text(k):
        t = _tasks[k]
        parts = [f"## Task {k}"]
        for i, ex in enumerate(t["train"]):
            parts.append(f"Example {i + 1} input:\n{_grid(ex['input'])}\nExample {i + 1} output:\n{_grid(ex['output'])}")
        for ex in t["test"]:
            parts.append(f"Test input:\n{_grid(ex['input'])}")
        return "\n\n".join(parts)


    _INTRO = "Find the transformation rule of each ARC task below and solve the test inputs.\n\n"
    _PH = "E13_BODY"


    def _render(effort):
        return _HF.apply_chat_template([{"role": "user", "content": _INTRO + _PH}], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True, reasoning_effort=effort, chat_template=_tpl)


    _RX, _RM = _render("xhigh"), _render("medium")
    assert _RX.count(_PH) == 1
    _head_txt, _tail_txt = _RX.split(_PH)
    _HEAD = TOK.encode(_head_txt, add_special_tokens=False).ids
    _TAIL = TOK.encode(_tail_txt, add_special_tokens=False).ids
    TPL_MSG = [{"role": "user", "content": _INTRO + _task_text(_ids[0])}]
    TPL_IDS = TOK.encode(_HF.apply_chat_template(TPL_MSG, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                                 reasoning_effort="xhigh", chat_template=_tpl), add_special_tokens=False).ids
    R["prompt_render"] = {"head_text": _head_txt[-600:], "tail_text": _tail_txt, "head_tokens": len(_HEAD), "tail_tokens": len(_TAIL),
                          "xhigh_differs_from_medium": _RX != _RM, "xhigh_sentence": "xhigh" in _head_txt,
                          "template_sha256": hashlib.sha256(_tpl.encode()).hexdigest(),
                          "tokenizer_json_sha256": hashlib.sha256((TOKDIR / "tokenizer.json").read_bytes()).hexdigest()}
    _TASK_IDS = {k: TOK.encode(_task_text(k) + "\n\n", add_special_tokens=False).ids for k in _ids}


    def prompt(band, variant):
        body, i = [], (variant * 7) % len(_ids)
        need = band - len(_HEAD) - len(_TAIL)
        while len(body) < need:
            body += _TASK_IDS[_ids[i % len(_ids)]]
            i += 1
        return _HEAD + body[:need] + _TAIL


    TOK_SHA = "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"   # = the 27B Merlin tokenizer.json
    TPL_SHA = "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"   # = the 27B FP8 chat_template.jinja
    R["prompt_render"]["source"] = "merlin" if MERLIN_PROMPTS else "intel_fallback"
    R["prompt_render"]["hashes_ok"] = (R["prompt_render"]["tokenizer_json_sha256"] == TOK_SHA and
                                       R["prompt_render"]["template_sha256"] == TPL_SHA)
    if MERLIN_PROMPTS:                    # the pinned hashes belong to the Merlin checkpoint only (Intel's are recorded)
        assert R["prompt_render"]["hashes_ok"], "Flash-Next tokenizer / template differ from the pinned Merlin checkpoint"
    assert R["prompt_render"]["xhigh_sentence"] and R["prompt_render"]["xhigh_differs_from_medium"], "xhigh render missing"
    BANDS = {"2k": 2048, "8k": 8192, "40k": 40000, "90k": 90000, "116k": 116000}
    NSTREAM = 8
    PROMPTS = {b: [prompt(n, v) for v in range(NSTREAM)] for b, n in BANDS.items()}
    MAX_NEW = {b: MAX_LEN - n - 512 for b, n in BANDS.items()}
    ORACLE_P = [PROMPTS["2k"][0], PROMPTS["2k"][1], PROMPTS["8k"][2]]
    LONG_P = [PROMPTS["40k"][3]]
    TF_P = {"40k": PROMPTS["40k"][5]}
    R["prompts"] = {b: {"n": len(p), "len": len(p[0])} for b, p in PROMPTS.items()}
    log("prompts", json.dumps(R["prompts"]), "| render", json.dumps({k: v for k, v in R["prompt_render"].items() if "text" not in k}))
    save()


if not RERUN:
    # ==================================================================== CELL 4 — the step runner (E23, verbatim)
    HELP = HELP if "HELP" in globals() else {}
    WIN = float(os.environ.get("E23_WIN_SEC") or 90.0)   # 120 s FULL windows
    EST_SCALE = float(os.environ.get("E23_EST_SCALE") or 1.0) if FAKE else 1.0   # local harness only: walk the whole order
    SM_TF = MODE == "SMOKE"                                            # SMOKE teacher-forced: the 40k row only
    ROWS = {}            # oracle greedy rows by key (same weights)
    TF = {}              # teacher-forced rows by key
    NTOK = 128 if MODE == "FULL" else 64
    LIVE = {"rec": None}                 # rev 3c: windows land in the step record before every save (a hard kill keeps them)
    COMPILE_ERR = re.compile(r"ninja: build stopped|Error building extension|error: command '.*(nvcc|g\+\+|gcc|c\+\+)' failed|"
                             r"GLIBCXX_[0-9.]+' not found|unsupported GNU version|fatal error: ")
    OOM_ERR = re.compile(r"OutOfMemoryError|CUDA out of memory|out of memory")
    XID = re.compile(r"NVRM: Xid|\bXid \d+|CUDA error: an illegal memory access|illegal memory access was encountered|"
                     r"unspecified launch failure|CUDA error: misaligned address")       # rev 3e: device faults, not any CUDA error


    def keep_after_error(rec, e):
        """E25 NB1 R2: an exception after the measurements (post-processing) keeps the whole record + postprocess_error /
        postprocess_tb. A clean boot (status None / NOT_RUN / DONE / TF_OOM, correctness CLEAN, quality PASS) keeps its status
        (None / NOT_RUN -> DONE), so its dependent boots still run; any other record -> POSTPROCESS_ERROR (status_before kept).
        R2 audit N1-3: postprocess_ok False; whether the dependents run is decided by core_measurements_complete (core_check)."""
        rec["postprocess_ok"] = False
        rec["postprocess_error"] = f"{type(e).__name__}: {e}"[:800]
        rec["postprocess_tb"] = traceback.format_exc()[-2500:]
        if (rec.get("status") in (None, "NOT_RUN", "DONE", "TF_OOM") and rec.get("correctness") == "CLEAN"
                and rec.get("quality_gate") == "PASS"):
            if rec.get("status") in (None, "NOT_RUN"):
                rec["status"] = "DONE"
        else:
            rec["status_before_postprocess"] = rec.get("status")
            rec["status"] = "POSTPROCESS_ERROR"
        return rec


    def step(sec, name, est_min, fn):
        """One measured step: skipped when the budget cannot hold it, recorded on any exception, never PASS by default."""
        rec = R["sections"].setdefault(sec, {}).setdefault(name, {"status": "NOT_RUN"})
        R["order"].append(f"{sec}/{name}")
        if ABORT["why"]:
            rec.update(status="SKIPPED_ABORT", why=ABORT["why"])
            log("---", sec, name, "SKIPPED_ABORT")
            save()
            return rec
        if left() < est_min * EST_SCALE * 60 + 300:
            rec.update(status="SKIPPED_BUDGET", left_min=round(left() / 60, 1), est_min=est_min)
            log("---", sec, name, "SKIPPED_BUDGET", f"(est {est_min} min, left {left() / 60:.0f} min)")
            save()
            return rec
        LIVE["rec"] = rec
        LIVE["out"] = None                       # E25 NB1 R2: the body's record once it returned (set by watched() / run_boot())
        t = time.time()
        log(">>>", sec, name, f"(est {est_min} min, left {left() / 60:.0f} min)")
        out = None                               # E25 NB1 R2
        try:
            out = fn() or {}
            rec.update(out)
            if rec.get("status") in (None, "NOT_RUN"):
                rec["status"] = "DONE"
        except Exception as e:
            part = out if isinstance(out, dict) and out else LIVE.get("out")   # E25 NB1 R2: never discard what the body produced
            if isinstance(part, dict) and part:                               # E25 NB1 R2
                rec.update(part)                                              # E25 NB1 R2
                keep_after_error(rec, e)                                      # E25 NB1 R2: clean -> DONE + postprocess_error
            else:                                                             # E25 NB1 R2
                rec.update(status="ERROR", error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2500:])
        rec["sec"] = round(time.time() - t, 1)
        LIVE["rec"] = None
        LIVE["out"] = None                       # E25 NB1 R2
        if ABORT["why"] and rec["status"] in ("DONE", "SERVER_DIED", "LAUNCH_FAILED", "BODY_ERROR"):
            rec["status"] = "PARTIAL_HARD_STOP"
        log("<<<", sec, name, rec.get("status"), f"{rec['sec']:.0f}s")
        save()
        return rec


    def with_servers(servers, timeout, body):
        """Start, wait ready, record facts + memory, run body(servers), always kill; then the GPU-release gate on these
        servers' GPUs only; abort the lab if they stay held."""
        out = {"servers": {}}
        timeout = max(60, min(timeout, left() - 120))
        try:
            for s in servers:
                s.start()
            for s in servers:
                ok, why = s.wait_ready(timeout)
                out["servers"][s.name] = {"ready": ok, "why": why, **s.facts(), "argv_tail": s.argv[-44:], "gpus": s.gpus,
                                          "help_check": HELP.get(s.name)}
                if not ok:
                    out["status"] = "LAUNCH_FAILED"
                    out["servers"][s.name]["log_tail"] = s.log()[-4000:]
                    _xl = [l for l in (s.log() or "").splitlines() if XID.search(l)][:8]   # rev 3e/3f: start-up / capture faults
                    out["servers"][s.name].update(xid=bool(_xl), xid_lines=_xl)
                    return out
            mem0 = L.gpu_mem()
            for s in servers:
                out["servers"][s.name].update(gpu_mem_after_ready=mem0, host_rss_mb=s.rss_mb())
            try:                             # rev 3: an exception in the body keeps everything measured so far
                out.update(body(servers) or {})
            except Exception as e:
                out.update(status="BODY_ERROR", body_error=f"{type(e).__name__}: {e}"[:800], body_tb=traceback.format_exc()[-2000:])
            for s in servers:
                out["servers"][s.name].update(s.facts(), gpu_mem_end=L.gpu_mem(), host_rss_mb_end=s.rss_mb(),
                                              alive_after_body=s.alive(), rc=None if s.alive() else s.proc.returncode,
                                              xid_lines=[l for l in (s.log() or "").splitlines() if XID.search(l)][:8])
                out["servers"][s.name]["xid"] = bool(out["servers"][s.name]["xid_lines"])
                if not s.alive():
                    out["status"] = "SERVER_DIED"
                    out["servers"][s.name]["log_tail"] = s.log()[-4000:]
            return out
        finally:
            for s in servers:
                s.kill()
            gpus = ",".join(sorted({g for s in servers for g in s.gpus.split(",")}))
            rel = L.wait_gpu_release(180, fake=bool(FAKE), gpus=gpus)
            if rel is False:                 # rev 3c: one more kill + 300 s before the lab aborts
                for s in servers:
                    s.kill()
                rel = L.wait_gpu_release(300, fake=bool(FAKE), gpus=gpus)
            if rel is None and NVSMI_OK:     # nvidia-smi worked at start: a failure now is not "released"
                rel = False
            out["gpu_released"] = rel
            if rel is False:
                ABORT["why"] = f"GPUs {gpus} still held after {[s.name for s in servers]}"
                out["status"] = "FAIL_GPU_RELEASE"


    def oracle(port, key=None, refs=()):
        """Greedy rows (2k, 2k, 8k) + one 40k row; saved under `key` (a target-only oracle) or compared with ROWS[ref]."""
        rows = L.greedy(port, ORACLE_P + LONG_P, n_tok=NTOK, timeout=max(120, min(900, left() / 6)))
        clean = all(not r.get("error") and r.get("tokens") and not L.degenerate(r.get("text")) for r in rows)
        if key and clean:            # rev 3d: an errored / empty / garbage reference is never stored (consumers: NO_ORACLE)
            ROWS[key] = rows
        elif key:
            R["errors"].append(f"oracle {key}: reference not stored (error / empty / garbage row)")
        res = {"garbage": [L.degenerate(r["text"]) for r in rows], "errors": [r["error"] for r in rows],
               "text_head": [r["text"][:120] for r in rows]}
        if not FAKE:                 # rev 3h: chat-path render vs the production xhigh render (recorded per arm)
            res["template_probe"] = L.template_probe(port, TPL_MSG, TPL_IDS)
        for ref in refs:
            res["vs_" + ref] = L.compare(ROWS.get(ref), rows) if ROWS.get(ref) else {"status": "INCONCLUSIVE_NO_ORACLE"}
        return res


    def tforced(port, key=None, ref=None):
        """Teacher-forced 40k / 90k prompt logprobs (last 256 positions); saved under `key` or compared with TF[ref].
        rev 3c: on stock vLLM only on its own short launch (prompt logprobs = chunk x 248k-vocab logits: OOM at 0.90)."""
        rows = {b: L.teacher_forced(port, ids) for b, ids in TF_P.items() if not (SM_TF and b != "40k")}
        if key and L.tf_complete(rows):                                 # rev 3e: complete references only
            TF[key] = rows
        elif key:
            R["errors"].append(f"teacher-forced {key}: reference not stored (incomplete band)")
        if ref:
            return {b: L.compare_tf((TF.get(ref) or {}).get(b), rows[b]) for b in rows}
        return {b: {"n": (rows[b] or {}).get("n"), "error": (rows[b] or {}).get("error")} for b in rows}


    def levels(port, band, lv, server=None, **kw):
        """rev 3c: the guard includes admission (prefill) time; a band stops once KV-bound (the fit is recorded); a dead
        server stops the band; every window is written into the live step record before save()."""
        out, fit = {}, None
        for n in lv:
            k = f"{band}@{n}"
            need = WIN + 90 + min(3600, n * BANDS[band] / 1500)
            if ABORT["why"] or left() < need + 120:
                out[k] = {"status": "SKIPPED_BUDGET", "need_sec": round(need)}
            elif fit is not None:
                out[k] = {"status": "SKIPPED_KV_BOUND", "streams_fit": fit}
            elif server is not None and not server.alive():
                out[k] = {"status": "SKIPPED_SERVER_DEAD"}
            else:
                try:                         # rev 3: one failing level never loses the others
                    out[k] = L.steady(port, PROMPTS[band], n, secs=WIN, max_new=MAX_NEW[band], **kw)
                except Exception as e:
                    out[k] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:300]}
                rm = out[k].get("running_max")
                if out[k].get("status") == "CAPACITY_LIMITED" or (rm is not None and 0 < rm < n and
                                                                  str(out[k].get("status")).startswith("INCONCLUSIVE")):
                    fit = rm
            if LIVE["rec"] is not None:
                LIVE["rec"].setdefault("live_windows", {})[k] = out[k]
            save()
        return out
    save()


if not RERUN:
    # ==================================================================== CELL 5 — the arms
    LADDER = [1, 2, 4, 5]                    # 40k: 8 streams would need 8 x (10 + 7) blocks of 4096 > 149 FP8 blocks (preempts)
    TEMP = 0.6                               # production sampling (Merlin's --override-generation-config is T 1.0)
    E23_TF_SCALE = {"tp4_vs_tp1_layout": 0.0772, "nvfp4_vs_bf16_kv": 0.3787}   # E23 SMOKE rev3g mean |d logprob|, for scale
    PREFILL = {"tok_s": 700.0, "measured": False}
    OOM_STEP = 268435456                     # Merlin's note: lower KV_BYTES in 0.25 GiB steps if the boot OOMs
    # Pre-run capacity (Merlin's rule: blocks per request = ceil(L / 4096) + 7; FP8 149 usable blocks, BF16 ~78)
    R["capacity_plan"] = {
        "fp8": {"pool_blocks": 149, "40k@5": 5 * (math.ceil(130560 / 4096) + 7), "8k@8": 8 * (math.ceil(32000 / 4096) + 7),
                "90k@4": 4 * (math.ceil(131072 / 4096) + 7), "116k@3": 3 * (math.ceil(131072 / 4096) + 7),
                "116k@4": 4 * (math.ceil(131072 / 4096) + 7),
                "note": "blocks if every stream grew to max length; the ladder windows are 90 s, so real use is far lower; "
                        "90k@4 and 116k@4 are capacity-limit probes"},
        "bf16": {"pool_blocks": 78, "40k@3": 3 * (math.ceil(130560 / 4096) + 7)},
        "long_rows": "1 natural xhigh ARC row (max 120,000 new) + 1 forced 60,000-token row, temperature 0.6, run CONCURRENTLY "
                     "(concurrency 2): not a single-stream latency number",
    }
    R["correctness_meaning"] = ("Both Merlin boots run MTP K3: CLEAN = BF16_MTP3 vs FP8_MTP3 parity plus probes / sampled / greedy "
                                "evidence; it is NOT target-only (MTP off) correctness. A production GO needs an MTP-off oracle and "
                                "a forced-rejection test.")
    CLEAN = ("EXACT", "NEAR_TIE")


    def _windows(o):
        """Every steady window of a step record (levels dicts + steady_multi replicas)."""
        out = []
        for v in (o.get("steady") or {}).values():
            if isinstance(v, dict):
                out.append(v)
        for k in ("40k", "90k"):
            for r in ((o.get(k) or {}).get("replicas") or []) if isinstance(o.get(k), dict) else []:
                if isinstance(r, dict):
                    out.append(r)
        return out


    def oracle_gate(o, need=(), extra=()):
        """None when clean; otherwise the NO_GO / INCONCLUSIVE status. `need` = refs that must compare clean; `extra` =
        further comparison dicts (replica greedy, 90k greedy) that must compare clean too. rev 3e (audits B-O1, C-F2):
        garbage under load in any window and failed arithmetic probes (real servers; the fake has no chat) are NO_GO too."""
        orc = o.get("oracle") or {}
        if any(orc.get("errors") or []):
            return "NO_GO_ORACLE_REQUEST_ERROR"
        if any(orc.get("garbage") or []):
            return "NO_GO_ORACLE_GARBAGE"
        if any(w.get("status") == "FAIL_GARBAGE_UNDER_LOAD" for w in _windows(o)):
            return "NO_GO_GARBAGE_UNDER_LOAD"
        if not FAKE and isinstance(o.get("probes"), dict) and o["probes"].get("ok") is False:
            return "NO_GO_SHORT_PROBE"
        sts = [(orc.get("vs_" + k) or {}).get("status") for k in need] + [(x or {}).get("status") for x in extra]
        if any(s == "FAIL_HIGH_MARGIN" for s in sts):
            return "NO_GO_ORACLE"
        if any(s not in CLEAN for s in sts):
            return "INCONCLUSIVE_NO_ORACLE" if any(s in (None, "INCONCLUSIVE_NO_ORACLE") for s in sts) else "INCONCLUSIVE_ORACLE"
        return None


    def gated(o, need=(), extra=()):
        g = oracle_gate(o, need, extra)
        o["correctness"] = g or "CLEAN"
        if g and o.get("status") in (None, "DONE"):
            o["status"] = g
        return o




    def _finite(x):
        return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


    _compare_e23, _compare_tf_e23 = L.compare, L.compare_tf


    def _compare_rc2(ref, cand):
        """Final audit H2: a non-finite margin on either side makes the comparison INCONCLUSIVE before any token logic
        (E23's compare read a NaN reference margin as NEAR_TIE)."""
        for rows in (ref or [], cand or []):
            for r in rows:
                if any(m is not None and not _finite(m) for m in (r.get("margins") or [])):
                    return {"status": "INCONCLUSIVE_NONFINITE"}
        return _compare_e23(ref, cand)


    def _compare_tf_rc2(ref, cand):
        """Final audit H2: a non-finite teacher-forced logprob is never MEASURED."""
        for side in (ref, cand):
            for p in ((side or {}).get("positions") or []):
                if p and p[0] is not None and not _finite(p[0]):
                    return {"status": "INCONCLUSIVE_NONFINITE"}
        return _compare_tf_e23(ref, cand)


    L.compare, L.compare_tf = _compare_rc2, _compare_tf_rc2


    def short_gate_failed(o):
        """Final audit H1: a failed probe (real servers; the fake has no chat) or a failed / garbage sampled request stops
        the boot's body at once; gated() then gives the NO_GO verdict."""
        p, s = o.get("probes") or {}, o.get("sampled") or {}
        return (p.get("ok") is not True and not FAKE) or s.get("ok") is not True or bool(s.get("garbage"))


    def oracle_ok(orc):
        return bool(orc) and orc.get("complete") is True and not any(orc.get("errors") or []) and not any(orc.get("garbage") or [])


    def _windows(o):
        """Every load window of a record: the ladder AND the long census (audit: census_long was not gated)."""
        out = []
        for key in ("steady", "census_long"):
            for v in (o.get(key) or {}).values():
                if isinstance(v, dict):
                    out.append(v)
        return out


    def evidence_gate(o, expected_rows, require_sampled=True):
        """Fail-closed evidence (three pre-GPU audits): probes, the sampled path and a COMPLETE oracle are mandatory;
        a missing piece is never CLEAN. expected_rows None = no oracle in this arm (diagnostic arms)."""
        pr = o.get("probes")
        if not isinstance(pr, dict):
            return "INCONCLUSIVE_PROBES_MISSING"
        if pr.get("ok") is not True and not FAKE:          # the fake server has no chat endpoint (E23 rule)
            return "NO_GO_SHORT_PROBE"
        if require_sampled:
            sm = o.get("sampled")
            if not isinstance(sm, dict):
                return "INCONCLUSIVE_NO_SAMPLED_TEST"
            if sm.get("ok") is not True or sm.get("garbage"):
                return "NO_GO_SAMPLED_PATH"
        if expected_rows is None:
            return None
        orc = o.get("oracle")
        if not isinstance(orc, dict):
            return "INCONCLUSIVE_NO_ORACLE"
        if any(orc.get("errors") or []):
            return "NO_GO_ORACLE_REQUEST_ERROR"
        if any(orc.get("garbage") or []):
            return "NO_GO_ORACLE_GARBAGE"
        if orc.get("row_count") != expected_rows or orc.get("complete") is not True:
            return "INCONCLUSIVE_ORACLE_INCOMPLETE"
        return None


    _gated_e23 = gated


    def gated(o, need=(), extra=(), expected_rows=4, require_sampled=True):
        g = evidence_gate(o, expected_rows, require_sampled)
        if g:
            o["correctness"] = g
            if o.get("status") in (None, "DONE"):
                o["status"] = g
            return o
        return _gated_e23(o, need, extra)


    def oracle(port, key=None, refs=(), prompts=None):
        """E23's oracle plus completeness: every expected row present, no error, exactly NTOK tokens (greedy rows run with
        ignore_eos, so a short row is a defect); a reference is stored only when complete and garbage-free."""
        prompts = list(prompts) if prompts is not None else ORACLE_P + LONG_P
        rows = L.greedy(port, prompts, n_tok=NTOK, timeout=max(120, min(900, left() / 6)))
        garbage = [L.degenerate(r.get("text")) for r in rows]
        complete = len(rows) == len(prompts) and all(not r.get("error") and len(r.get("tokens") or []) == NTOK for r in rows)
        if key and complete and not any(garbage):
            ROWS[key] = rows
        elif key:
            R["errors"].append(f"oracle {key}: reference not stored (incomplete / error / garbage)")
        res = {"garbage": garbage, "errors": [r.get("error") for r in rows], "text_head": [(r.get("text") or "")[:120] for r in rows],
               "row_count": len(rows), "expected_rows": len(prompts), "tokens": [len(r.get("tokens") or []) for r in rows],
               "complete": complete}
        if not FAKE:
            res["template_probe"] = L.template_probe(port, TPL_MSG, TPL_IDS)
        for ref in refs:
            res["vs_" + ref] = L.compare(ROWS.get(ref), rows) if ROWS.get(ref) else {"status": "INCONCLUSIVE_NO_ORACLE"}
        return res


    def kv_census(s):
        log_t = s.log() or ""
        lines = log_t.splitlines()
        kv = re.findall(r"GPU KV cache size: ([\d,]+) tokens", log_t)
        conc = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", log_t)
        pick = lambda rx, n=24: [l[:240] for l in lines if re.search(rx, l, re.I)][:n]
        return {"kv_tokens": [int(x.replace(",", "")) for x in kv][:2], "max_concurrency": conc[:2],
                "scale_lines": pick(r"K/V scale|kv scale|scaling factor|DEFAULT 1\.0|static K/V"),
                "engine_lines": pick(r"WNA16 MoE backend|MarlinExperts|PLE offload|engram|cudagraph_mode|CUDAGraphMode|PIECEWISE"),
                "startup_lines": pick(r"Loading weights took|Model loading took|torch\.compile takes|torch\.compile took|"
                                      r"Graph capturing finished|init engine|PLE.*ready|Available KV cache memory"),
                "oom_lines": [l[:240] for l in lines if OOM_ERR.search(l)][:6], "host_mem": host_mem()}


    def sampled_check(s):
        """RC1 died at its first sampled request (FlashInfer sampling JIT: ld cannot find -lcuda): one short sampled request
        right after ready, then the engine must still be alive."""
        g = L.Stream(s.port, PROMPTS["2k"][1], 64, temperature=TEMP, seed=5)
        g.start()
        g.join(900)
        if g.is_alive():
            g.stop()
            g.join(10)
        r = g.row()
        return {"tokens": len(r["tokens"]), "error": r["error"], "finish": r["finish_reason"], "alive_after": s.alive(),
                "garbage": L.degenerate(r["text"]), "ok": len(r["tokens"]) == 64 and not r["error"] and s.alive()}


    def prefill_probe(port):
        """Time to first token of one cold 40k prompt (1 new token): the measured prefill rate the ladder admission uses."""
        ids = PROMPTS["40k"][7]
        g = L.Stream(port, ids, 1, temperature=0.0)
        g.start()
        g.join(1800)
        if g.is_alive():
            g.stop()
            g.join(10)
        sec = (g.t_first or g.t_end or time.perf_counter()) - (g.t_start or time.perf_counter())
        ok = not g.err and g.t_first is not None
        if ok:
            PREFILL.update(tok_s=round(len(ids) / max(1e-3, sec), 1), measured=True)
        return {"prompt": len(ids), "ttft_sec": round(sec, 1), "tok_s": round(len(ids) / max(1e-3, sec), 1), "error": g.err, "ok": ok}


    def window_garbage(ids, win=2000, min_len=200):
        """Final audit H3: the first garbage 2,000-token window of a row (None if every window is clean). A whole-text check
        would read 100k tokens of healthy reasoning as LOW_VARIETY (unique words < 8 %), so windows are checked one by one."""
        for i in range(0, len(ids), win):
            seg = ids[i:i + win]
            if len(seg) < min_len:
                break
            v = L.degenerate(TOK.decode(seg))
            if v:
                return {"start_token": i, "verdict": v}
        return None


    def long_quality(name, r):
        if not isinstance(r, dict):
            return "MISSING_LONG"
        if r.get("error"):
            return "FAIL_LONG_REQUEST"
        if r.get("window_garbage"):
            return "FAIL_LONG_GARBAGE"
        if name == "forced60k" and not r.get("reached_60k"):
            return "INCOMPLETE_LONG"
        if name == "natural" and r.get("cut_by_cap"):
            return "INCOMPLETE_LONG"
        return "PASS_LONG"


    def long_gate(o):
        """None when the record has no long rows or both rows PASS_LONG; otherwise the verdict for the top-level status.
        Priority: request error > natural garbage > forced garbage (synthetic past its first EOS) > incomplete."""
        lo = o.get("long")
        if not isinstance(lo, dict):
            return None
        q = {n: long_quality(n, lo.get(n)) for n in ("natural", "forced60k")}
        lo["long_quality"] = q
        if "MISSING_LONG" in q.values():
            return "INCONCLUSIVE_LONG_MISSING"
        if "FAIL_LONG_REQUEST" in q.values():
            return "NO_GO_LONG_REQUEST_ERROR"
        if q["natural"] == "FAIL_LONG_GARBAGE":
            return "NO_GO_LONG_GARBAGE"
        if q["forced60k"] == "FAIL_LONG_GARBAGE":
            return "NO_GO_FORCED60K_GARBAGE"
        if "INCOMPLETE_LONG" in q.values():
            return "INCONCLUSIVE_LONG_INCOMPLETE"
        return None


    def long_rows(port, secs_cap=3000):
        """One natural xhigh ARC request (its own stop, up to 120,000 new tokens) and one forced 60,000-token row
        (ignore_eos: 'synthetic' past its first EOS), CONCURRENTLY (concurrency 2). Full text in traces/ + requests.jsonl.
        A row cut by the time cap is incomplete, not a correctness failure."""
        cap = max(60, min(secs_cap, left() - 900))
        jobs = {"natural": L.Stream(port, TPL_IDS, min(120000, MAX_LEN - len(TPL_IDS) - 512), temperature=TEMP, seed=11,
                                    ignore_eos=False),
                "forced60k": L.Stream(port, PROMPTS["8k"][0], 60000, temperature=TEMP, seed=12, ignore_eos=True)}
        t = time.time()
        for g in jobs.values():
            g.start()
        for g in jobs.values():
            g.join(max(1, cap - (time.time() - t)))
        for g in jobs.values():
            if g.is_alive():
                g.stop()
                g.join(10)
        out = {"concurrency": 2, "cap_sec": round(cap)}
        for name, g in jobs.items():
            row = g.row()
            (TRACES / f"long_{port}_{name}.txt").write_text(row["text"])
            ids = row["tokens"]
            n_tok = len(ids)
            sec = (g.t_end or time.perf_counter()) - (g.t_start or time.perf_counter())
            sl = {}
            for k, end in (("to_8k", 8000), ("to_32k", 32000), ("to_60k", 60000), ("to_100k", 100000)):
                sl[k] = L.degenerate(TOK.decode(ids[end - 2000:end])) if n_tok >= end else "NOT_REACHED"
            out[name] = {"tokens": n_tok, "sec": round(sec, 1), "tok_s": round(n_tok / max(1e-6, sec), 1), "error": row["error"],
                         "finish": row["finish_reason"], "garbage": sl, "window_garbage": window_garbage(ids),
                         "think_closed": "</think>" in row["text"], "cut_by_cap": row["finish_reason"] is None,
                         "tail": row["text"][-400:]}
        out["forced60k"]["reached_60k"] = out["forced60k"]["tokens"] >= 60000
        out["forced60k"]["label"] = "synthetic past the first EOS (ignore_eos)"
        try:
            m = {k: v for k, v in L.prom(port).items() if k.startswith(("vllm:spec_decode", "vllm:num_preemptions", "vllm:kv_cache_usage"))}
        except Exception as e:
            m = {"error": repr(e)[:200]}
        acc, drf = m.get(L.ACC), m.get(L.DRF)
        out["metrics_end"] = m
        out["mtp_accept_per_draft"] = round(acc / drf, 3) if acc and drf else None
        return out


    def tf_last(s, key=None, ref=None, aa=False):
        """Teacher-forced 40k LAST in a boot (chunk x 248k-vocab prompt logits can OOM the last PP rank); free memory first.
        How a server death here is classified is decided from its log in merlin_boot (TF_OOM vs TF_SERVER_DIED)."""
        o = {"gpu_before": L.gpu_mem()}
        if key:
            o["ref"] = tforced(s.port, key=key)
            if aa and s.alive():
                o["aa"] = tforced(s.port, ref=key)              # A/A: the same boot, the same prompt (the noise floor)
                TF["aa"] = o["aa"]
        if ref:
            o["vs_" + ref] = tforced(s.port, ref=ref)
            o["aa_" + ref] = TF.get("aa")
            o["e23_scale"] = E23_TF_SCALE
        o["alive_after"] = s.alive()
        return o


    def merlin_boot(name, kv, port, timeout, body, need=(), expected_rows=4, require_sampled=True):
        """Boot gate (ok must be True), one Merlin boot, cleanup of the /dev/shm entries this boot created; a boot OOM
        retries once 0.25 GiB lower. A server death during TF is TF_OOM only with an OOM line and no device fault."""
        kvb = NB1_KV.get(name) or (KV_FP8 if kv == "fp8" else KV_BF16)   # E25 NB1: the B3 pin step
        tries = []
        while True:
            gate = boot_gate()
            if gate["ok"] is not True:
                return {"status": "GPU_NOT_EMPTY", "boot_gate": gate, "tries": tries}
            shm0 = shm_names()
            o = with_servers([merlin_fn_server(name, kv, port, kvb)], timeout, body)
            o.update(kv_bytes=kvb, boot_gate=gate, shm_cleanup=shm_cleanup(shm0))
            srv = (o.get("servers") or {}).get(name) or {}
            oom = o.get("status") == "LAUNCH_FAILED" and bool(OOM_ERR.search(srv.get("log_tail") or ""))
            tries.append({"kv_bytes": kvb, "status": o.get("status"), "oom": oom})
            if not oom or len(tries) > 1 or left() < timeout + 900:
                break
            kvb -= OOM_STEP
            log("boot OOM:", name, "retry with KV_BYTES", kvb)
        o["tries"] = tries
        if o.get("tf_died") and o.get("status") == "SERVER_DIED":
            srv = (o.get("servers") or {}).get(name) or {}
            if OOM_ERR.search(srv.get("log_tail") or "") and not srv.get("xid"):
                o["status"] = "TF_OOM"
                o["status_note"] = "the server died of OOM in the teacher-forced call (run last); everything before it is valid"
            else:
                o["status"] = "TF_SERVER_DIED"
                o["status_note"] = "the server died in the teacher-forced call without an OOM line (or with a device fault)"
        g = gated(o, need=need, expected_rows=expected_rows, require_sampled=require_sampled)
        if g.get("status") == "TF_OOM" and g.get("correctness") != "CLEAN":
            g["status"] = g["correctness"]
        lg = long_gate(g)                                  # final audit H3: the long rows have their own verdict
        g["long_verdict"] = lg or ("PASS_LONG" if isinstance(g.get("long"), dict) else None)
        if lg and g.get("status") in ("DONE", "TF_OOM"):
            g["status"] = lg
        return g


    def arm_bf16():
        def body(ss):
            s = ss[0]
            p = s.port
            o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
            if short_gate_failed(o):
                o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s))
                return o
            o["prefill"] = prefill_probe(p)
            o["oracle"] = oracle(p, key="bf16")
            o["census_end"] = kv_census(s)
            if not oracle_ok(o["oracle"]):
                o["early_stop"] = "ORACLE_INCOMPLETE"          # no reference: the TF A/A would measure nothing usable
                return o
            o["tf"] = tf_last(s, key="bf16", aa=True)
            o["tf_died"] = not o["tf"]["alive_after"]
            return o
        ok, why = sha_gate(min(1500, left() - 900))
        R["setup"]["sha_gate"] = {"ok": ok, "why": why}
        if not ok:
            return {"status": "SKIPPED_SHA_GATE", "why": why}
        return merlin_boot("merlin_bf16", "bf16", 8000, 45 * 60, body)


    def fp8_gate(o):
        vs = (o["oracle"].get("vs_bf16") or {}).get("status")
        if (o.get("sampled") or {}).get("ok") is not True:
            return "FAIL_SAMPLED"
        if (o.get("probes") or {}).get("ok") is not True and not FAKE:
            return "FAIL_PROBES"
        if any(o["oracle"].get("errors") or []) or any(o["oracle"].get("garbage") or []) or vs == "FAIL_HIGH_MARGIN":
            return "FAIL"
        if o["oracle"].get("complete") is True and vs in CLEAN:
            return "PASS"
        return "INCONCLUSIVE"


    def arm_fp8():
        def body(ss):
            s = ss[0]
            p = s.port
            o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
            if short_gate_failed(o):
                o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s),
                         fp8_gate="FAIL_SAMPLED" if (o["sampled"] or {}).get("ok") is not True else "FAIL_PROBES")
                return o
            if not PREFILL["measured"]:
                o["prefill"] = prefill_probe(p)
            o["oracle"] = oracle(p, refs=("bf16",))
            o["fp8_gate"] = fp8_gate(o)
            if o["fp8_gate"] == "PASS":                      # load tests only on a PASS (audits: INCONCLUSIVE never runs them)
                kw = {"server": s, "temperature": TEMP, "prefill_tok_s": PREFILL["tok_s"]}
                o["steady"] = levels(p, "40k", LADDER, **kw)
                o["steady"].update(levels(p, "8k", [8], **kw))
                o["census_long"] = {**levels(p, "90k", [3, 4], **kw), **levels(p, "116k", [3, 4], **kw)}
                o["long"] = long_rows(p)
            else:
                o["load_tests_skipped"] = f"FP8 comparison gate: {o['fp8_gate']}"
                o["status"] = "NO_GO_FP8_KV" if o["fp8_gate"].startswith("FAIL") else "INCONCLUSIVE_FP8_KV"
            o["census_end"] = kv_census(s)
            if o["fp8_gate"] == "PASS" or os.environ.get("RC2_TF_ON_FP8_FAIL") == "1":   # default: fail fast
                o["tf"] = tf_last(s, ref="bf16")
                o["tf_died"] = not o["tf"]["alive_after"]
            return o
        return merlin_boot("merlin_fp8", "fp8", 8001, 40 * 60, body, need=("bf16",))


    def arm_bf16_load():
        """Only when FP8 fails: the ladder, the census and the long rows on BF16 KV (seqs 3). DIAGNOSTIC_ONLY: this arm
        has no oracle of its own (the A1 boot is the BF16 reference)."""
        def body(ss):
            s = ss[0]
            p = s.port
            o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
            if short_gate_failed(o):
                o["early_stop"] = "SHORT_CORRECTNESS_GATE_FAILED"
            else:
                kw = {"server": s, "temperature": TEMP, "prefill_tok_s": PREFILL["tok_s"]}
                o["steady"] = levels(p, "40k", [1, 2, 3], **kw)
                o["census_long"] = levels(p, "116k", [1, 2], **kw)
                o["long"] = long_rows(p)
            o["census_end"] = kv_census(s)
            o["certification"] = "DIAGNOSTIC_ONLY"
            return o
        return merlin_boot("merlin_bf16_load", "bf16", 8002, 30 * 60, body, expected_rows=None)


    def arm_stock():
        """Fallback when Merlin cannot serve: stock vLLM 0.32 + Intel W4A16 (RC1's correct engine), -lcuda shim, FP8 KV."""
        if INTEL is None or WHEELHOUSE is None:
            return {"status": "FALLBACK_UNAVAILABLE", "why": f"intel={INTEL} wheelhouse={WHEELHOUSE}"}

        def body(ss):
            s = ss[0]
            p = s.port
            o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
            if short_gate_failed(o):
                o["early_stop"] = "SHORT_CORRECTNESS_GATE_FAILED"
                return o
            o["oracle"] = oracle(p, key="stock", prompts=ORACLE_P)          # 2k / 2k / 8k (32k max length: no 40k row)
            if oracle_ok(o["oracle"]):
                o["steady"] = levels(p, "8k", [1, 2, 3], server=s, temperature=TEMP)
            return o
        gate = boot_gate()
        if gate["ok"] is not True:
            return {"status": "GPU_NOT_EMPTY", "boot_gate": gate}
        shm0 = shm_names()
        o = gated(with_servers([stock_fn_server("stock_fn", 8003)], 40 * 60, body), expected_rows=len(ORACLE_P))
        o.update(boot_gate=gate, shm_cleanup=shm_cleanup(shm0))
        return o


    save()


if not RERUN:
    # ==================================================================== CELL 6 — E25 NB1: measurement tools (NB1 R2 verbatim; E25 NB4S dropped the unused capacity / prefix / routing / PP2-trace functions: commit_mode, commit_ab, commit_counter_delta, pool_blocks_from_usage, reset_cache, _hits, alloc_sequence, reset_succeeded, fit_alloc, kv_accounting, n_fit, admission_test, residency_bound, residency_capacity, _f_valid, resolve_F, f_from_log, residency_probes, block_layout, nb1_complete, nb1_tf_req, lp_compare, fp8_self_check, nb1_ladder, _prefix_counts, _reuse, prefix_tests, _merge, _clip, _inter, microbatch_costs, pp2_trace, merge_routing, pp2_profile)
    import gzip
    import statistics
    from collections import Counter
    NB1 = R.setdefault("nb1", {})
    SCORECARD = OUT / "lever_scorecard.json"
    TF_RAW = OUT / "tf_raw"
    TF_RAW.mkdir(exist_ok=True)
    HOSTMEM = OUT / "hostmem"
    HOSTMEM.mkdir(exist_ok=True)
    BLOCK = 4096
    PIN_STEPS_GIB = (1.0, 0.75, 0.5)             # B3 pin ladder (addendum 5): the largest step predicted safe first, then down
    PIN_FREE_MIN_MIB = 512                       # a step must leave >= 0.5 GiB free per rank after warm-up + a 90k prefill
    KV_STEP = 512 << 20                          # the smallest step (+0.5 GiB per card)
    RAM_NEED_MIB = (69 + 4) * 1024               # Merlin's ~69 GiB resident floor + 4 GiB margin, on its own rule
    BOOT_READY_SEC = 30 * 60                     # readiness gate per boot
    SEQS_HI = 16
    CAPTURE_HI = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64]   # Merlin registry.capture_sizes_for(16, K=3): seqs x (1 + K)
    COMPILE_HI = json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": CAPTURE_HI})
    PROF_DIR = TMP / "nb1_profile"
    PROFILER_CFG = json.dumps({"profiler": "torch", "torch_profiler_dir": str(PROF_DIR), "torch_profiler_with_stack": False,
                               "ignore_frontend": True})      # idle unless /start_profile (annotations only while running)
    EXT_FLAG = ["--worker-extension-cls", "nb1_worker_ext.NB1WorkerExt"]
    DIAG_ENV = {"VLLM_SERVER_DEV_MODE": "1"}     # /collective_rpc (read-only extension) + /reset_prefix_cache; 127.0.0.1 only
    NCCL_DIAG = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET"}   # B1 only: the real NCCL transport per group
    COMMIT_GRAPH_ENV = {"VLLM_RECOVERSSM_COMMIT_GRAPH": "graph"}         # B2 boot: commit graphs captured; flipped to eager at once
    RC2_FP8 = json.loads(r"""{"argv_tail": ["127.0.0.1", "--port", "8001", "--tensor-parallel-size", "2", "--pipeline-parallel-size", "2", "--enable-expert-parallel", "--disable-custom-all-reduce", "--speculative-config", "{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"probabilistic\"}", "--max-model-len", "262144", "--max-num-seqs", "8", "--max-num-batched-tokens", "1024", "--gpu-memory-utilization", "0.96", "--kv-cache-memory-bytes", "2479838700", "--kv-cache-dtype", "fp8_e4m3", "--limit-mm-per-prompt", "{\"image\":42,\"video\":0}", "--mm-processor-cache-gb", "0", "--mm-processor-kwargs", "{\"size\":{\"longest_edge\":8388608,\"shortest_edge\":65536}}", "--compilation-config", "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[1,2,3,4,6,8,12,16,24,32]}", "--enable-prefix-caching", "--mamba-cache-mode", "align", "--enable-prompt-tokens-details", "--no-enable-flashinfer-autotune", "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder", "--reasoning-parser", "qwen3", "--override-generation-config", "{\"temperature\":1.0,\"top_p\":0.95,\"top_k\":20,\"min_p\":0.0,\"presence_penalty\":0.0,\"repetition_penalty\":1.0}", "--default-chat-template-kwargs", "{\"enable_thinking\": true, \"reasoning_effort\": \"medium\"}", "--shutdown-timeout", "60", "--block-size", "4096", "--use-replayssm", "--prefix-match-unit", "64", "--served-model-name", "q27", "--max-model-len", "131072", "--port", "8001", "--default-chat-template-kwargs", "{\"enable_thinking\": true, \"reasoning_effort\": \"xhigh\"}"], "env": {"KV_BYTES": "2479838700", "NCCL_P2P_DISABLE": "1", "NCCL_VERSION": "2.25.1-1", "P2P": "off", "SCALES": "/tmp/fn_merlin/qsa_kv_scales_262k.json", "VLLM_CACHE_ROOT": "/tmp/merlin-cache", "VLLM_E2_GUARD_MODE": "warn", "VLLM_E2_PLE_EARLY_REQUEST": "1", "VLLM_E2_PLE_FAST_IDS": "1", "VLLM_E2_QSA_LOGITS_GRANULE_MB": "256", "VLLM_HOST_EMBED_TABLE": "1", "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "0", "VLLM_PLE_OFFLOAD_HOME_DEVICE": "0", "VLLM_PLE_OFFLOAD_READY_TIMEOUT": "3600", "VLLM_PORT": "30100", "VLLM_PP_LAYER_PARTITION": "25,23", "VLLM_QSA_KV_SCALES": "/tmp/fn_merlin/qsa_kv_scales_262k.json", "VLLM_SKIP_P2P_CHECK": "1", "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB": "256", "VLLM_USE_BREAKABLE_CUDAGRAPH": "0", "VLLM_USE_RUST_FRONTEND": "0", "VLLM_USE_V2_MODEL_RUNNER": "1"}, "kv_bytes": 2479838700, "kv_tokens": 513940, "max_concurrency": 3.92, "pool_blocks": 149, "F_rule": 6, "F_engine": 6, "seqs": 8, "gpu_mem_after_ready": [[0, 20903, 23034], [1, 20903, 23034], [2, 20863, 23034], [3, 21389, 23034]], "gpu_mem_end": [[0, 21427, 23034], [1, 21427, 23034], [2, 21387, 23034], [3, 21401, 23034]], "windows": {"40k@1": {"status": "STEADY", "tok_s": 70.0, "mal": 3.086, "step_ms": 44.1, "running": [1.0, 1.0]}, "40k@2": {"status": "STEADY", "tok_s": 107.4, "mal": 3.251, "step_ms": 30.3, "running": [2.0, 2.0]}, "40k@4": {"status": "STEADY", "tok_s": 161.0, "mal": 3.017, "step_ms": 37.5, "running": [4.0, 4.0]}, "40k@5": {"status": "STEADY", "tok_s": 190.5, "mal": 3.073, "step_ms": 40.3, "running": [5.0, 5.0]}, "8k@8": {"status": "STEADY", "tok_s": 261.8, "mal": 3.154, "step_ms": 48.2, "running": [8.0, 8.0]}, "90k@3": {"status": "INCONCLUSIVE_PREFILL_IN_WINDOW", "tok_s": 48.7, "mal": 2.93, "step_ms": 60.1, "running": [1.0, 3.0]}, "90k@4": {"status": "STEADY", "tok_s": 159.8, "mal": 3.022, "step_ms": 37.8, "running": [4.0, 4.0]}, "116k@3": {"status": "STEADY", "tok_s": 140.5, "mal": 3.223, "step_ms": 34.4, "running": [3.0, 3.0]}, "116k@4": {"status": "STEADY", "tok_s": 160.3, "mal": 3.016, "step_ms": 37.6, "running": [4.0, 4.0]}}, "tf_40k_vs_bf16": {"status": "MEASURED", "n": 256, "mean_abs_dlogprob": 1.8764, "p99_abs_dlogprob": 5.464, "max_abs_dlogprob": 7.15, "rank1_agreement": 0.3672, "forced_token_rank1_both_or_neither": 0.457}, "ready_sec": 999.1, "bf16": {"kv_tokens": 269042, "max_concurrency": 2.05, "pool_blocks": 78, "seqs": 3, "gpu_mem_after_ready": [[0, 20905, 23034], [1, 20905, 23034], [2, 20693, 23034], [3, 20693, 23034]], "gpu_mem_end": [[0, 21425, 23034], [1, 21425, 23034], [2, 21215, 23034], [3, 21215, 23034]], "ready_sec": 411.1}, "rank_load_gib": {"2": 16.87, "1": 17.33, "0": 17.33, "3": 16.87}, "rank_graph_gib": {"0": 0.33, "1": 0.33, "2": 0.43, "3": 0.43}}""")      # RC2's real boots: argv tail, env keys, pool, memory, windows (rc2_results.json)
    RC2_ORACLE = json.loads(r"""{"bf16": [{"text": "We need solve ARC task. Need infer transformation from examples. User gives Task 0934a4d8 with Example 1 input/output, Example 2 input/output, Example 3 input only? Need solve test inputs? They provided Example 3 input: 194499276 maybe incomplete? Need output? Need find transformation rule and solve test inputs. We need analyze.\n\nWe have ARC tasks represented as strings of digits? Example 1 input is 31 rows? Let's count. Example 1 input lines:\n1 3533665414", "tokens": [1596, 1144, 11290, 67964, 3274, 13, 14235, 22846, 17454, 494, 9989, 13, 2570, 6486, 5261, 220, 15, 24, 18, 19, 64, 19, 67, 23, 440, 12996, 220, 16, 1879, 45447, 11, 12996, 220, 17, 1879, 45447, 11, 12996, 220, 18, 1879, 1132, 30, 14235, 11290, 1228, 10804, 30, 2302, 3766, 12996, 220, 18, 1879, 25, 220, 16, 24, 19, 19, 24, 24, 17, 22, 21, 6970, 31084, 30, 14235, 2468, 30, 14235, 1423, 17454, 5728, 321, 11290, 1228, 10804, 13, 1165, 1144, 22903, 13, 271, 1596, 599, 67964, 8806, 14807, 430, 8796, 314, 17955, 30, 12996, 220, 16, 1879, 369, 220, 18, 16, 6761, 30, 6558, 579, 1698, 13, 12996, 220, 16, 1879, 4965, 25, 198, 16, 220, 18, 20, 18, 18, 21, 21, 20, 19, 16, 19], "margins": [11.375, 10.75, 0.25, 3.5, 0.125, 1.75, 2.375, 0.625, 4.375, 0.125, 4.125, 1.375, 0.125, 1.0, 1.25, 4.875, 8.6875, 10.375, 10.375, 9.6875, 8.25, 10.1875, 14.0, 10.4375, 0.125, 0.875, 1.25, 8.375, 3.875, 1.0, 3.75, 7.375, 3.75, 12.0, 2.625, 8.375, 6.75, 2.5, 8.75, 10.5, 6.625, 0.125, 0.375, 0.5, 0.75, 8.25, 4.125, 1.0, 1.0, 0.125, 0.25, 4.625, 6.375, 9.25, 2.875, 0.375, 4.125, 4.25, 10.9375, 11.5625, 10.625, 10.5625, 9.5625, 11.5, 10.6875, 0.5, 2.25, 4.875, 2.125, 1.25, 0.125, 0.5, 0.625, 0.625, 4.25, 1.125, 2.875, 3.125, 3.125, 4.125, 0.75, 1.375, 1.75, 0.75, 1.0, 4.25, 1.5, 0.5, 1.125, 0.625, 4.625, 1.625, 0.25, 4.5, 1.0, 0.0, 1.25, 8.5, 5.0, 1.125, 2.75, 3.375, 0.625, 0.125, 1.125, 1.875, 7.25, 1.875, 2.125, 0.25, 0.875, 6.25, 1.875, 1.625, 2.625, 2.875, 2.125, 2.375, 8.125, 10.9375, 10.0625, 9.0625, 9.875, 9.375, 11.25, 11.375, 11.875, 12.9375], "error": null, "prompt_sha": "8df9ed2c640ef5bd", "prompt_len": 2048, "finish_reason": "length"}, {"text": "We need solve ARC tasks. Need infer transformation rules from examples and apply to test inputs. We have two tasks: 1818057f and 195c6913. Need output transformed test grids. Need analyze carefully.\n\nWe need produce final answer likely grids. Need infer rules.\n\nTask 1818057f:\nExamples: inputs/outputs with colors 2,4,8. Output changes some 4s to 8? Let's inspect.\n\nExample1 input 10x12? rows:\n1 22224", "tokens": [1596, 1144, 11290, 67964, 8806, 13, 14235, 22846, 17454, 5426, 494, 9989, 321, 3668, 310, 1228, 10804, 13, 1165, 599, 1330, 8806, 25, 220, 16, 23, 16, 23, 15, 20, 22, 69, 321, 220, 16, 24, 20, 66, 21, 24, 16, 18, 13, 14235, 2468, 22773, 1228, 54455, 13, 14235, 22903, 15060, 13, 271, 1596, 1144, 7936, 1534, 4087, 4222, 54455, 13, 14235, 22846, 5426, 13, 271, 6065, 220, 16, 23, 16, 23, 15, 20, 22, 69, 25, 198, 39044, 25, 10804, 14, 39652, 440, 7736, 220, 17, 11, 19, 11, 23, 13, 8984, 4203, 1010, 220, 19, 82, 310, 220, 23, 30, 6558, 579, 23331, 13, 271, 12929, 16, 1879, 220, 16, 15, 87, 16, 17, 30, 6761, 25, 198, 16, 220, 17, 17, 17, 17, 19], "margins": [11.875, 11.375, 0.75, 3.375, 2.625, 3.0, 3.25, 0.5, 3.125, 0.25, 0.625, 4.75, 1.125, 3.25, 2.0, 5.5, 0.875, 3.75, 0.375, 0.5, 1.375, 6.5, 1.125, 2.25, 12.625, 12.0625, 11.3125, 12.9375, 12.4375, 8.875, 10.8125, 10.75, 1.75, 7.375, 13.5625, 13.875, 11.6875, 11.25, 10.75, 12.375, 12.375, 10.75, 2.25, 2.25, 1.625, 0.875, 1.0, 0.375, 0.875, 1.375, 0.125, 1.875, 6.5, 1.5, 1.0, 1.375, 1.0, 4.75, 4.625, 1.75, 1.0, 1.625, 0.875, 0.625, 0.25, 3.5, 1.0, 1.25, 4.625, 11.625, 5.75, 11.125, 12.3125, 10.8125, 10.9375, 11.0, 11.8125, 0.125, 0.0, 2.625, 0.375, 0.25, 0.875, 7.875, 0.75, 0.875, 1.875, 8.125, 0.75, 7.375, 2.875, 3.875, 2.125, 0.25, 2.0, 3.25, 2.625, 0.625, 2.625, 2.125, 10.75, 10.625, 0.75, 1.75, 7.5, 2.75, 6.375, 0.25, 1.75, 1.375, 0.75, 3.25, 8.375, 4.625, 3.75, 4.375, 2.625, 1.5, 0.25, 0.5, 5.375, 1.375, 0.5, 9.75, 4.625, 12.625, 11.4375, 8.375], "error": null, "prompt_sha": "8750743e9dcadca9", "prompt_len": 2048, "finish_reason": "length"}, {"text": "We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We have multiple tasks: 247ef758, 269e22fb, 271d71e2, 28a6681f, 291dc1e1. Need output transformed grids for test inputs. Need infer rules from examples. Need be careful. We need produce final answer likely grids. Need analyze each task.\n\nWe need parse tasks. There are two test inputs for 247ef758? It says Test input: ...", "tokens": [1596, 1144, 11290, 67964, 8806, 13, 14235, 1423, 17454, 5728, 364, 1754, 3274, 321, 11290, 1228, 10804, 13, 1165, 599, 5081, 8806, 25, 220, 17, 19, 22, 797, 22, 20, 23, 11, 220, 17, 21, 24, 68, 17, 17, 10481, 11, 220, 17, 22, 16, 67, 22, 16, 68, 17, 11, 220, 17, 23, 64, 21, 21, 23, 16, 69, 11, 220, 17, 24, 16, 7385, 16, 68, 16, 13, 14235, 2468, 22773, 54455, 364, 1228, 10804, 13, 14235, 22846, 5426, 494, 9989, 13, 14235, 381, 16097, 13, 1165, 1144, 7936, 1534, 4087, 4222, 54455, 13, 14235, 22903, 1754, 3274, 13, 271, 1596, 1144, 4563, 8806, 13, 2532, 513, 1330, 1228, 10804, 364, 220, 17, 19, 22, 797, 22, 20, 23, 30, 1049, 2640, 3284, 1879, 25, 2423], "margins": [11.75, 9.875, 1.125, 1.25, 3.875, 2.0, 2.875, 0.375, 5.25, 0.875, 1.5, 5.625, 0.875, 3.0, 2.875, 3.125, 7.75, 6.75, 0.5, 0.5, 1.5, 5.625, 1.375, 3.0, 11.75, 13.375, 12.625, 11.3125, 12.0, 13.5, 12.625, 0.625, 8.4375, 11.625, 11.0, 12.5625, 10.5625, 11.0, 10.375, 8.375, 7.375, 9.75, 11.5625, 9.125, 10.0, 9.75, 9.4375, 10.3125, 10.4375, 9.0, 8.1875, 9.4375, 12.625, 9.75, 11.0, 11.3125, 10.9375, 9.625, 9.5625, 10.0, 8.1875, 7.625, 12.375, 11.125, 11.375, 10.625, 10.25, 10.375, 10.0, 2.5, 3.0, 2.125, 0.125, 0.125, 1.875, 0.125, 4.125, 3.125, 2.0, 0.75, 3.0, 0.75, 8.75, 4.125, 0.125, 1.25, 3.875, 2.75, 0.125, 0.25, 0.625, 3.875, 1.625, 1.875, 0.75, 1.0, 0.375, 2.375, 2.125, 0.5, 0.5, 2.5, 1.875, 0.125, 0.75, 0.375, 3.625, 0.125, 1.375, 0.0, 4.125, 7.375, 2.5, 0.375, 12.75, 6.75, 12.375, 4.625, 8.75, 12.1875, 10.25, 2.0, 0.625, 0.25, 1.125, 7.625, 1.75, 2.125], "error": null, "prompt_sha": "fc63e297d382cd8f", "prompt_len": 8192, "finish_reason": "length"}, {"text": "We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We need output likely transformed grids. Need analyze many tasks. Need be careful. We have multiple tasks with examples and test inputs. Need infer rules.\n\nWe need produce final answer with transformed test inputs for each task. There are tasks: 2c181942, 2d0172a1 (two test inputs?), 31f7f899, 332f06d7, 35ab12c3, 36a08778", "tokens": [1596, 1144, 11290, 67964, 8806, 13, 14235, 1423, 17454, 5728, 364, 1754, 3274, 321, 11290, 1228, 10804, 13, 1165, 1144, 2468, 4222, 22773, 54455, 13, 14235, 22903, 1599, 8806, 13, 14235, 381, 16097, 13, 1165, 599, 5081, 8806, 440, 9989, 321, 1228, 10804, 13, 14235, 22846, 5426, 13, 271, 1596, 1144, 7936, 1534, 4087, 440, 22773, 1228, 10804, 364, 1754, 3274, 13, 2532, 513, 8806, 25, 220, 17, 66, 16, 23, 16, 24, 19, 17, 11, 220, 17, 67, 15, 16, 22, 17, 64, 16, 318, 19186, 1228, 10804, 66926, 220, 18, 16, 69, 22, 69, 23, 24, 24, 11, 220, 18, 18, 17, 69, 15, 21, 67, 22, 11, 220, 18, 20, 365, 16, 17, 66, 18, 11, 220, 18, 21, 64, 15, 23, 22, 22, 23], "margins": [10.125, 11.5, 1.125, 0.25, 3.5, 2.125, 2.625, 0.5, 5.25, 1.0, 1.75, 5.75, 0.375, 2.375, 3.625, 4.25, 6.625, 7.375, 0.25, 0.125, 1.25, 0.5, 0.25, 2.0, 0.5, 1.625, 1.0, 1.0, 4.375, 2.875, 0.75, 2.125, 4.125, 4.0, 0.375, 0.5, 1.625, 3.875, 0.375, 2.125, 2.125, 2.125, 2.875, 1.625, 2.875, 1.125, 0.75, 1.375, 0.125, 1.625, 2.25, 1.0, 2.875, 2.5, 3.0, 0.125, 1.125, 0.75, 1.25, 2.375, 1.75, 1.625, 0.625, 3.0, 0.125, 4.5, 0.5, 11.625, 10.625, 12.0625, 13.0625, 9.5625, 10.9375, 10.375, 10.875, 1.0, 6.5, 9.8125, 10.25, 11.9375, 13.5, 12.3125, 11.625, 11.25, 11.25, 0.375, 2.5, 4.125, 7.25, 0.625, 5.375, 9.75, 9.625, 12.4375, 11.25, 10.4375, 9.0, 11.4375, 10.8125, 3.25, 8.375, 11.875, 7.375, 11.5, 11.0, 10.5625, 8.8125, 10.875, 10.375, 5.625, 10.125, 13.125, 8.625, 7.375, 11.1875, 10.5, 10.25, 10.9375, 5.25, 9.75, 10.1875, 7.25, 11.4375, 12.0, 8.9375, 11.0625, 9.5, 10.3125], "error": null, "prompt_sha": "49b3958bb234d4fd", "prompt_len": 40000, "finish_reason": "length"}], "fp8": [{"text": "We need solve ARC task. Need infer transformation from examples. User gives Task 0934a4d8. Example 1 input 30x30? output 9x4? Example 2 input 30x30? output 4x5? Example 3 input: \"194499276\" only one row? Maybe truncated? Need find transformation rule and solve test inputs. But only one test input? It says Example 3 input: 194499276. Maybe that's the test input? Need output. Need infer from", "tokens": [1596, 1144, 11290, 67964, 3274, 13, 14235, 22846, 17454, 494, 9989, 13, 2570, 6486, 5261, 220, 15, 24, 18, 19, 64, 19, 67, 23, 13, 12996, 220, 16, 1879, 220, 18, 15, 87, 18, 15, 30, 2468, 220, 24, 87, 19, 30, 12996, 220, 17, 1879, 220, 18, 15, 87, 18, 15, 30, 2468, 220, 19, 87, 20, 30, 12996, 220, 18, 1879, 25, 328, 16, 24, 19, 19, 24, 24, 17, 22, 21, 1, 1132, 799, 2713, 30, 10380, 57739, 30, 14235, 1423, 17454, 5728, 321, 11290, 1228, 10804, 13, 1921, 1132, 799, 1228, 1879, 30, 1049, 2640, 12996, 220, 18, 1879, 25, 220, 16, 24, 19, 19, 24, 24, 17, 22, 21, 13, 10380, 421, 579, 279, 1228, 1879, 30, 14235, 2468, 13, 14235, 22846, 494], "margins": [11.5, 11.25, 0.25, 3.625, 0.375, 2.25, 2.25, 0.875, 4.125, 0.25, 4.25, 1.25, 0.125, 1.0, 1.75, 4.625, 9.0, 9.9375, 10.375, 9.5, 8.875, 10.6875, 14.0, 10.625, 0.125, 0.75, 0.875, 7.5, 4.5, 1.25, 3.5, 0.75, 0.0, 1.75, 4.0, 2.625, 1.25, 9.375, 5.5, 4.25, 2.0, 0.25, 2.625, 0.875, 8.875, 3.75, 4.375, 5.75, 5.5, 4.25, 3.5, 5.5, 0.0, 8.0, 7.25, 7.375, 4.625, 5.875, 1.5, 3.75, 4.875, 9.125, 6.75, 0.25, 0.625, 8.25, 8.25, 12.3125, 11.5, 10.3125, 11.0, 10.1875, 11.625, 9.875, 4.375, 2.25, 0.375, 1.0, 4.875, 0.0, 0.25, 5.75, 2.125, 0.0, 0.25, 2.375, 2.875, 6.75, 9.125, 5.125, 1.875, 0.75, 2.375, 0.0, 0.875, 1.875, 2.125, 0.75, 2.25, 1.625, 9.1875, 7.125, 9.125, 5.625, 1.125, 7.9375, 7.0625, 10.375, 11.375, 10.8125, 10.5625, 11.1875, 11.5, 11.625, 1.0, 1.625, 1.375, 2.625, 0.5, 2.375, 1.75, 2.625, 2.0, 2.625, 0.125, 1.5, 0.25, 1.5], "error": null, "prompt_sha": "8df9ed2c640ef5bd", "prompt_len": 2048, "finish_reason": "length"}, {"text": "We need solve ARC tasks. Need infer transformation rules from examples and apply to test inputs. We have two tasks: 1818057f and 195c6913. Need output transformed test grids. Need analyze carefully.\n\nWe need produce final answer likely grids. Need infer rules.\n\nTask 1818057f examples:\nInput/output grids with colors 2,4,8. Output changes some 4s to 8? Let's inspect.\n\nExample1 10x12? Input rows:\n1 222244", "tokens": [1596, 1144, 11290, 67964, 8806, 13, 14235, 22846, 17454, 5426, 494, 9989, 321, 3668, 310, 1228, 10804, 13, 1165, 599, 1330, 8806, 25, 220, 16, 23, 16, 23, 15, 20, 22, 69, 321, 220, 16, 24, 20, 66, 21, 24, 16, 18, 13, 14235, 2468, 22773, 1228, 54455, 13, 14235, 22903, 15060, 13, 271, 1596, 1144, 7936, 1534, 4087, 4222, 54455, 13, 14235, 22846, 5426, 13, 271, 6065, 220, 16, 23, 16, 23, 15, 20, 22, 69, 9989, 25, 198, 2425, 45447, 54455, 440, 7736, 220, 17, 11, 19, 11, 23, 13, 8984, 4203, 1010, 220, 19, 82, 310, 220, 23, 30, 6558, 579, 23331, 13, 271, 12929, 16, 220, 16, 15, 87, 16, 17, 30, 5396, 6761, 25, 198, 16, 220, 17, 17, 17, 17, 19, 19], "margins": [11.875, 11.75, 0.5, 3.125, 2.625, 3.0, 3.125, 0.5, 3.0, 0.5, 0.625, 4.875, 1.0, 3.25, 2.0, 5.625, 0.625, 3.875, 0.375, 0.75, 1.375, 6.75, 1.375, 2.125, 12.8125, 11.9375, 11.6875, 13.1875, 12.0625, 8.4375, 10.3125, 10.5, 1.625, 7.25, 13.5625, 12.625, 10.5625, 9.375, 12.0, 12.6875, 11.3125, 11.0625, 2.0, 2.375, 1.625, 0.875, 1.0, 0.125, 0.5, 1.25, 0.125, 1.875, 6.625, 1.625, 1.5, 1.5, 1.0, 4.75, 4.625, 2.0, 0.875, 1.5, 1.0, 0.625, 0.5, 3.375, 1.0, 1.125, 4.5, 12.375, 5.625, 11.125, 12.625, 10.0625, 9.875, 10.125, 11.625, 0.125, 1.375, 0.5, 1.0, 0.75, 0.375, 1.25, 1.0, 1.75, 7.75, 0.75, 7.75, 2.375, 3.625, 1.625, 0.25, 2.0, 3.5, 2.75, 1.125, 2.25, 2.375, 10.8125, 10.75, 0.75, 1.875, 7.5, 2.75, 5.25, 0.25, 2.125, 1.25, 0.0, 7.125, 3.625, 5.5, 4.5, 2.0, 1.5, 0.75, 0.875, 0.0, 6.375, 1.0, 0.75, 10.0, 4.75, 12.625, 11.5625, 8.375, 11.4375], "error": null, "prompt_sha": "8750743e9dcadca9", "prompt_len": 2048, "finish_reason": "length"}, {"text": "We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We have multiple tasks: 247ef758, 269e22fb, 271d71e2, 28a6681f, 291dc1e1. Need output transformed test inputs. Need infer rules from examples. Need be careful. We can analyze each.\n\nWe need produce final answer likely grids for each test input. Need determine exact transformations.\n\nLet's parse tasks.\n\nTask 247ef758: examples show", "tokens": [1596, 1144, 11290, 67964, 8806, 13, 14235, 1423, 17454, 5728, 364, 1754, 3274, 321, 11290, 1228, 10804, 13, 1165, 599, 5081, 8806, 25, 220, 17, 19, 22, 797, 22, 20, 23, 11, 220, 17, 21, 24, 68, 17, 17, 10481, 11, 220, 17, 22, 16, 67, 22, 16, 68, 17, 11, 220, 17, 23, 64, 21, 21, 23, 16, 69, 11, 220, 17, 24, 16, 7385, 16, 68, 16, 13, 14235, 2468, 22773, 1228, 10804, 13, 14235, 22846, 5426, 494, 9989, 13, 14235, 381, 16097, 13, 1165, 628, 22903, 1754, 13, 271, 1596, 1144, 7936, 1534, 4087, 4222, 54455, 364, 1754, 1228, 1879, 13, 14235, 7995, 4581, 51192, 13, 271, 9764, 579, 4563, 8806, 13, 271, 6065, 220, 17, 19, 22, 797, 22, 20, 23, 25, 9989, 1420], "margins": [11.875, 9.75, 1.25, 1.5, 3.625, 2.125, 3.0, 0.125, 5.25, 0.875, 1.625, 5.5, 0.875, 3.125, 2.875, 2.875, 8.5, 7.125, 0.375, 0.375, 1.875, 5.375, 1.625, 3.0, 12.125, 13.3125, 12.9375, 11.8125, 11.8125, 13.5, 12.5, 0.375, 8.875, 11.6875, 10.9375, 12.25, 11.0625, 10.6875, 10.375, 8.3125, 7.375, 10.4375, 11.1875, 9.0, 9.8125, 10.125, 10.25, 10.8125, 10.125, 9.3125, 8.0625, 9.5, 12.4375, 10.4375, 11.0, 11.1875, 11.1875, 9.6875, 10.375, 9.0, 7.75, 7.625, 12.875, 10.8125, 11.375, 9.625, 10.625, 11.0625, 9.875, 2.375, 2.875, 2.125, 0.125, 0.0, 0.875, 0.75, 2.0, 0.625, 3.125, 0.875, 8.375, 3.875, 0.125, 1.25, 4.125, 2.875, 0.375, 0.125, 0.5, 0.125, 0.875, 1.375, 2.5, 2.25, 0.75, 4.0, 1.75, 2.0, 0.875, 0.625, 1.125, 3.375, 4.25, 3.75, 0.875, 0.125, 0.5, 0.875, 6.25, 1.5, 3.125, 10.25, 2.625, 2.25, 3.25, 3.625, 3.625, 6.75, 10.375, 7.625, 12.5, 11.75, 11.25, 11.875, 13.4375, 1.5, 0.0, 0.25], "error": null, "prompt_sha": "fc63e297d382cd8f", "prompt_len": 8192, "finish_reason": "length"}, {"text": "We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We have multiple tasks with examples and test inputs. Need output transformed grids likely. Need infer rules. This is complex. Need analyze each task carefully. Need produce final answer with outputs for each test input. Need be accurate.\n\nWe have tasks:\n2c181942 (one test input)\n2d0172a1 (two test inputs? It lists Test input twice)\n31f7f899 (one test)\n332f06d7 (one test)", "tokens": [1596, 1144, 11290, 67964, 8806, 13, 14235, 1423, 17454, 5728, 364, 1754, 3274, 321, 11290, 1228, 10804, 13, 1165, 599, 5081, 8806, 440, 9989, 321, 1228, 10804, 13, 14235, 2468, 22773, 54455, 4222, 13, 14235, 22846, 5426, 13, 1061, 369, 6150, 13, 14235, 22903, 1754, 3274, 15060, 13, 14235, 7936, 1534, 4087, 440, 15796, 364, 1754, 1228, 1879, 13, 14235, 381, 12995, 13, 271, 1596, 599, 8806, 25, 198, 17, 66, 16, 23, 16, 24, 19, 17, 318, 588, 1228, 1879, 8, 198, 17, 67, 15, 16, 22, 17, 64, 16, 318, 19186, 1228, 10804, 30, 1049, 11140, 3284, 1879, 10598, 8, 198, 18, 16, 69, 22, 69, 23, 24, 24, 318, 588, 1228, 8, 198, 18, 18, 17, 69, 15, 21, 67, 22, 318, 588, 1228, 8], "margins": [10.0, 11.25, 1.0, 0.625, 3.375, 2.5, 2.875, 0.375, 5.125, 0.875, 1.75, 5.875, 0.375, 2.875, 3.625, 4.125, 6.375, 7.5, 0.25, 0.0, 0.25, 6.25, 0.25, 3.75, 2.0, 2.625, 3.125, 2.875, 4.125, 2.75, 0.5, 1.375, 0.625, 0.5, 1.875, 0.125, 1.375, 0.25, 0.0, 9.625, 0.125, 1.375, 1.0, 0.625, 3.125, 4.125, 0.25, 1.875, 0.125, 0.875, 3.875, 1.5, 2.75, 0.0, 2.5, 0.875, 3.0, 4.75, 1.25, 1.0, 0.125, 1.75, 10.625, 1.875, 2.625, 0.125, 3.0, 3.5, 1.125, 0.25, 10.5, 13.375, 13.3125, 12.0625, 10.375, 9.625, 11.0, 0.125, 1.625, 7.5, 1.0, 3.5, 7.75, 7.0, 10.375, 13.0625, 13.1875, 12.0, 11.1875, 11.875, 11.1875, 7.125, 5.75, 5.25, 8.5, 0.625, 0.75, 0.75, 1.375, 7.375, 0.75, 0.0, 4.375, 8.9375, 9.875, 12.8125, 11.8125, 10.6875, 10.1875, 10.5625, 12.125, 4.625, 5.25, 1.125, 0.0, 10.875, 11.6875, 6.875, 13.0, 11.125, 12.0, 10.0625, 11.375, 10.5, 9.25, 9.0, 2.625, 5.25], "error": null, "prompt_sha": "49b3958bb234d4fd", "prompt_len": 40000, "finish_reason": "length"}]}""")   # RC2's recorded greedy oracle rows (traces/requests.jsonl), BF16 + FP8
    GEN_RATE = 45.0                              # tok/s per stream used only to project window growth (RC2: 33 at 8, 70 at 1)
    PP = 2                                       # pipeline stages of the profile: one engine step = one micro-batch (C/2 requests)
    TF_BANDS = ("2k", "8k")                      # FP8 self-consistency only (< 2 min); the BF16 / FP8 isolation is NB0's
    TF_P = {b: PROMPTS[b][5] for b in TF_BANDS}  # RC2's TF prompt (variant 5)
    RETENTION_KNOBS = {   # Merlin source (ad42e37), recorded, not changed
        "MAX_GRACE_PINS": "16 finished turns keep their cached prefix referenced (kv_cache_manager.py:735, class constant)",
        "MAX_PINNED_WAITERS": "16 waiting requests' hits pinned until admission (kv_cache_manager.py:940, class constant)",
        "enable_grace_pins": "enable_caching and partial hash hits (prefix-match-unit 64), kv_cache_manager.py:234",
        "VLLM_E2_SCHED_MISS_DEFER_S": "120 s default (merlin/envs.py:40): a cold miss may wait for protected waiters",
        "VLLM_E2_SCHED_COHERENT_YIELD": "on by default (merlin/envs.py:41): need-sized, session-coherent yields",
        "reset": "/reset_prefix_cache (VLLM_SERVER_DEV_MODE=1) drops grace and waiter pins when they alone block the reset "
                 "(kv_cache_manager.py:1107-1113); it fails while requests run"}
    for _b, _n in (("8k", 34), ("40k", 16)):     # distinct prompts (RC2 has 8 per band)
        PROMPTS[_b] += [prompt(BANDS[_b], v) for v in range(len(PROMPTS[_b]), _n)]
    BANDS["60k"] = 60000                         # residency probe band (addendum 7); RC2's PROMPTS bands are unchanged
    PROMPTS["60k"] = [prompt(BANDS["60k"], v) for v in range(8)]
    MAX_NEW["60k"] = MAX_LEN - BANDS["60k"] - 512
    RESIDENCY_BANDS = ("8k", "40k", "60k", "90k", "116k")
    V_CENSUS = {"8k": list(range(24, 32)), "40k": list(range(9, 13))}   # never used by a steady window in the same boot
    V_PREFIX = {"8k": 32, "40k": 8}
    V_ABORT = 33
    SUFFIX_TXT = ("\n</think>\n\nI have a first rule.<|im_end|>\n<|im_start|>user\nCheck the rule on every training pair, "
                  "then give the final test grid." + _tail_txt)
    SUFFIX_IDS = TOK.encode(SUFFIX_TXT, add_special_tokens=False).ids
    R["prompts"] = {b: {"n": len(p), "len": len(p[0])} for b, p in PROMPTS.items()}


    def psha(ids):
        return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


    R["nb1_prompt_sha256"] = {b: psha(TF_P[b]) for b in TF_BANDS}
    _osha = [psha(p)[:16] for p in ORACLE_P + LONG_P]          # RC2's own prompt_sha formula (16 hex)
    NB1["rc2_oracle_prompts_match"] = {k: [r["prompt_sha"] for r in v] == _osha for k, v in RC2_ORACLE.items()}
    for _k, _v in RC2_ORACLE.items():                          # RC2's recorded rows become references only for identical prompts
        if NB1["rc2_oracle_prompts_match"][_k] and not FAKE:    # (the fake server's text differs: it builds its own reference)
            ROWS["rc2_" + _k] = [{"text": r["text"], "tokens": r["tokens"], "margins": r["margins"], "error": r["error"]} for r in _v]
    NB1["retention_knobs"] = RETENTION_KNOBS
    NB1["sampling"] = ("every request passes its own sampling (greedy: temperature 0; load: T 0.6, top_p 0.95, top_k 20; "
                       "Merlin's --override-generation-config is T 1.0)")
    R["capacity_plan"] = {
        "rule": "blocks per running request = ceil(L / 4096) + F, F = 6 (3 mamba groups x 2; RC2 analysis: 149 / 3.92 = 38 = 32 + 6, "
                "live transitions 6.0 / 6.1); a finished session keeps floor(L / 4096) + 3 blocks (up to 16) until re-admitted or yielded",
        "rc2_pool_blocks": {"fp8": 149, "bf16": 78},
        "worst_case_F6": {"8k@8": 8 * (math.ceil((8192 + 8000) / 4096) + 6), "40k@5": 5 * (math.ceil((40000 + 8000) / 4096) + 6),
                          "8k x 8 hold (cold sequence)": 8 * (math.ceil(9000 / 4096) + 6), "40k x 4 hold": 4 * (math.ceil(41000 / 4096) + 6),
                          "90k x 5 admission (max_new 3000)": 5 * (math.ceil(93000 / 4096) + 6), "pool": 149},
        "retention": "every capacity test starts from /reset_prefix_cache (retained sessions released); windows record the idle "
                     "(retained) blocks before them, so running = used - retained",
        "highest_levels": "computed at run time from the measured pool and F, never above max_num_seqs read from the server log",
        "tf": "B1 only: 2k + 8k FP8 self-consistency (teacher-forced A/A + greedy cached-repeat), < 2 min",
        "c3_note": "at BLOCK_SIZE 4096 the attention page is larger than the mamba page (RC2 log: 'Padding mamba page size by "
                   "21.01%'), so bf16 SSM state cannot change the pool's block count; B2 records the smaller minimum block it allows"}

    # ---- trace classifier fix (A7): QSA / Marlin MoE / top-k before the generic patterns; GDN '(anonymous namespace)' is
    # already handled by the lab's rev 3h _cat
    L.CATS = [("nccl_sendrecv", ("sendrecv", "send_recv", "nccldevkernel_send", "nccldevkernel_recv")), L.CATS[0],
              ("recoverssm_commit", ("recoverssm", "commit_", "boundary_fold", "final_fold", "conv_compaction")),
              ("qsa", ("qsa",)), ("moe_marlin", ("marlin_moe", "fused_marlin_moe", "moe_wna16")),
              ("topk", ("persistent_topk", "cooperative_topk"))] + list(L.CATS[1:])
    L.COMPUTE_CATS = tuple(L.COMPUTE_CATS) + ("nccl_sendrecv", "recoverssm_commit", "qsa", "moe_marlin", "topk")

    # ---- RC2 long-row gate, stricter (audit 2): a natural row needs finish == stop and a closed think; forced rows
    # (ignore_eos) are stability tests only. NB1 runs no long rows; the reused gate is fixed for any later reuse.
    _long_quality_rc2, _long_gate_rc2 = long_quality, long_gate


    def long_quality(name, r):
        q = _long_quality_rc2(name, r)
        if q == "PASS_LONG" and name == "natural":
            if r.get("finish") != "stop":
                return "INCOMPLETE_NATURAL_FINISH"
            if r.get("think_closed") is not True:
                return "INCOMPLETE_THINKING"
        return q


    def long_gate(o):
        g = _long_gate_rc2(o)
        lo = o.get("long") if isinstance(o.get("long"), dict) else None
        if lo and isinstance(lo.get("forced60k"), dict):
            lo["forced60k"]["counts_as"] = "stability test only (ignore_eos)"
        if g is None and lo and any(v in ("INCOMPLETE_NATURAL_FINISH", "INCOMPLETE_THINKING") for v in (lo.get("long_quality") or {}).values()):
            return "INCONCLUSIVE_LONG_INCOMPLETE"
        return g


    # ---- the #60490 backport + the worker extension (written before B1)
    P60490 = "# vLLM PR #60490 (issue #60489) rebased onto Merlin v3.0.0 (commit ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5,\n# tree 07c16a79d077f1e06c19d0eea84efd64f0b63540 = the tree the Kaggle payload installs, install_merlin.log).\n# Same code as upstream; only the import context differs (Merlin's import block). One marker comment added.\n# pre-patch sha256  a23d5f90507ad5185f8b8e8d6ba13ae0cb53b2506264fdfbbf6a60b1171dc653\n# post-patch sha256 88fe0a8f4dd2ef90fc6aefb5771b5b48e40c0b1d575039283c52aeca19b8bd6b\n--- a/vllm/models/qwen4_exp/nvidia/qsa.py\n+++ b/vllm/models/qwen4_exp/nvidia/qsa.py\n@@ -18,8 +18,9 @@\n from torch import nn\n from transformers import Qwen4ExpTextConfig\n \n+import vllm.envs as envs\n from vllm.compilation.breakable_cudagraph import eager_break_during_capture\n-from vllm.config import VllmConfig\n+from vllm.config import VllmConfig, get_current_vllm_config_or_none\n from vllm.config.cache import CacheConfig, CacheDType\n from vllm.distributed import get_tensor_model_parallel_world_size\n from vllm.forward_context import get_forward_context\n@@ -45,6 +46,7 @@\n )\n from vllm.platforms import current_platform\n from vllm.platforms.interface import DeviceCapability\n+from vllm.utils.math_utils import cdiv\n from vllm.utils.torch_utils import (\n     LayerNameType,\n     _encode_layer_name,\n@@ -75,6 +77,7 @@\n from . import model\n from .indexer_qsa import QSAIndexer\n from .ops.qsa import qsa_e4m3_uses_native_cast\n+from .ops.qsa_indexer import _TOPK_WORKSPACE_BYTES\n \n logger = init_logger(__name__)\n \n@@ -782,6 +785,31 @@\n         if isinstance(metadata, list):\n             metadata = metadata[0]\n         if not isinstance(metadata, dict):\n+            # E25 NB1: backport of vLLM PR #60490 (issue #60489).\n+            # Profiling/dummy run: the indexer allocates its selection scratch\n+            # on demand, so touch its peak here for the memory profiler (as\n+            # sparse_attn_indexer does): the prefill logits budget or the\n+            # worst-case decode logits [decode tokens, compressed columns],\n+            # plus the top-k workspace.\n+            logits_bytes = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024\n+            cfg = get_current_vllm_config_or_none()\n+            if cfg is not None:\n+                spec = cfg.speculative_config\n+                num_spec = spec.num_speculative_tokens if spec is not None else 0\n+                decode_tokens = min(\n+                    cfg.scheduler_config.max_num_seqs * (num_spec + 1),\n+                    cfg.scheduler_config.max_num_batched_tokens,\n+                )\n+                block_size = cfg.cache_config.block_size\n+                columns = cdiv(cfg.model_config.max_model_len, block_size) * cdiv(\n+                    block_size, self.indexer.compress_ratio\n+                )\n+                logits_bytes = max(logits_bytes, decode_tokens * columns * 4)\n+            _ = torch.empty(\n+                logits_bytes + _TOPK_WORKSPACE_BYTES,\n+                dtype=torch.uint8,\n+                device=output.device,\n+            )\n             output.zero_()\n             return\n         main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])\n"
    P60490_REL = "vllm/models/qwen4_exp/nvidia/qsa.py"
    P60490_PRE, P60490_POST = "a23d5f90507ad5185f8b8e8d6ba13ae0cb53b2506264fdfbbf6a60b1171dc653", "88fe0a8f4dd2ef90fc6aefb5771b5b48e40c0b1d575039283c52aeca19b8bd6b"
    P60490_MARKER = "# E25 NB1: backport of vLLM PR #60490 (issue #60489)."
    NB1_EXT_SRC = "\"\"\"E25 NB1 worker extension (read-only): reached through POST /collective_rpc {\"method\": \"nb1_cache_info\"} with\nVLLM_SERVER_DEV_MODE=1 and --worker-extension-cls nb1_worker_ext.NB1WorkerExt. Reports the ACTUAL cache tensors bound\nto every attention / Mamba layer (dtype, shape, bytes), the KV cache specs (padded vs unpadded Mamba page bytes), the\nphysical block count and this rank's free memory. It never raises: every part returns its error text instead.\"\"\"\nimport json\nimport math\n\n\ndef _t(x):\n    try:\n        import torch\n        if isinstance(x, torch.Tensor):\n            return {\"dtype\": str(x.dtype).replace(\"torch.\", \"\"), \"shape\": list(x.shape), \"bytes\": int(x.numel() * x.element_size())}\n    except Exception:\n        pass\n    return None\n\n\ndef _flat(kc):\n    if _t(kc):\n        return [kc]\n    if isinstance(kc, (list, tuple)):\n        if kc and isinstance(kc[0], (list, tuple)) and not _t(kc[0]):\n            kc = kc[0]                              # one entry per virtual engine\n        return [x for x in kc if _t(x)]\n    return []\n\n\n_RT = {\"on\": False, \"installed\": False, \"rows\": [], \"passes\": 0, \"calls\": 0, \"max_steps\": 200, \"bytes\": 0, \"errors\": [],\n       \"first\": None, \"skipped_capturing\": 0, \"skipped_prefill\": 0}\n_RT_MAX_BYTES = 16 << 20\n_RT_DECODE_MAX_ROWS = 128          # a decode / verify micro-batch: requests x (1 + K) tokens; prefill chunks are larger\n\n\ndef _rt_layer(name):\n    import re as _r\n    m = _r.search(r\"layers\\.(\\d+)\", str(name))\n    return int(m.group(1)) if m else -1\n\n\ndef _rt_wrap(orig):\n    def select_experts(self, hidden_states, router_logits, *a, **k):\n        out = orig(self, hidden_states, router_logits, *a, **k)\n        if _RT[\"on\"] and _RT[\"passes\"] < 2 * _RT[\"max_steps\"] and _RT[\"bytes\"] < _RT_MAX_BYTES:\n            try:\n                import torch\n                if torch.cuda.is_current_stream_capturing():\n                    _RT[\"skipped_capturing\"] += 1\n                else:\n                    ids = out[1]\n                    lay = _rt_layer(getattr(self, \"nb1_layer\", None))\n                    if _RT[\"first\"] is None:\n                        _RT[\"first\"] = lay\n                    n = int(ids.shape[0])\n                    if n > _RT_DECODE_MAX_ROWS:\n                        _RT[\"skipped_prefill\"] += 1\n                    else:\n                        if lay == _RT[\"first\"]:\n                            _RT[\"passes\"] += 1     # one pass of a micro-batch through this stage's first MoE layer\n                        keep = ids.to(torch.int16).cpu().numpy().reshape(n, -1)\n                        _RT[\"rows\"].append((lay, max(0, _RT[\"passes\"] - 1), n, keep))\n                        _RT[\"bytes\"] += keep.nbytes\n                        _RT[\"calls\"] += 1\n            except Exception as e:\n                if len(_RT[\"errors\"]) < 5:\n                    _RT[\"errors\"].append(repr(e)[:160])\n        return out\n    return select_experts\n\n\nclass NB1WorkerExt:\n    def nb1_routing(self, action, arg=\"\"):\n        \"\"\"Router top-k expert ids, ONLY between 'start' and 'stop' (the profiling window): FusedMoERouter.select_experts is\n        wrapped once at class level; every decode / verify-sized call (<= 128 token rows) that runs in Python (not during\n        a CUDA-graph capture) keeps its ids (int16) with the layer index and the pass index (counted at this stage's first\n        MoE layer; NB1 maps pass -> step = pass // 2, mb_index = pass % 2 under PP2); <= max_steps steps, <= 16 MiB per\n        rank. FULL-graph replays run no Python, so graph-replayed decode steps are NOT recorded (vLLM's\n        --enable-return-routed-experts writes the ids from inside the graph; it changes the forward graph and memory, so it\n        is not used on a measured boot). 'stop' writes the rank's part file <arg>_rank<r>.npz.\"\"\"\n        out = {\"action\": action}\n        try:\n            import torch.distributed as dist\n            rank = dist.get_rank() if dist.is_initialized() else 0\n            out[\"rank\"] = rank\n            if action == \"start\":\n                from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter\n                if not _RT[\"installed\"]:\n                    FusedMoERouter.select_experts = _rt_wrap(FusedMoERouter.select_experts)\n                    _RT[\"installed\"] = True\n                    model = getattr(getattr(self, \"model_runner\", None), \"model\", None)\n                    named = 0\n                    if model is not None:\n                        for name, mod in model.named_modules():\n                            for attr in (\"router\", \"_router\"):\n                                r = getattr(mod, attr, None)\n                                if isinstance(r, FusedMoERouter):\n                                    r.nb1_layer = name\n                                    named += 1\n                    out[\"routers_named\"] = named\n                _RT.update(on=True, rows=[], passes=0, calls=0, bytes=0, errors=[], first=None, skipped_capturing=0,\n                           skipped_prefill=0, max_steps=int(arg or 200))\n                return out\n            _RT[\"on\"] = False\n            out.update(passes=_RT[\"passes\"], calls=_RT[\"calls\"], bytes=_RT[\"bytes\"], errors=_RT[\"errors\"],\n                       skipped_capturing=_RT[\"skipped_capturing\"], skipped_prefill=_RT[\"skipped_prefill\"])\n            if action == \"stop\" and arg and _RT[\"rows\"]:\n                import numpy as np\n                rows = _RT[\"rows\"]\n                offs, o_ = [], 0\n                for r in rows:\n                    offs.append(o_)\n                    o_ += r[2]\n                path = f\"{arg}_rank{rank}.npz\"\n                np.savez_compressed(path, ids=np.concatenate([r[3] for r in rows], axis=0), layer=np.array([r[0] for r in rows]),\n                                    pass_index=np.array([r[1] for r in rows]), mb_size=np.array([r[2] for r in rows]),\n                                    offset=np.array(offs), rank=np.array([rank]))\n                out[\"path\"] = path\n            _RT[\"rows\"] = []\n        except Exception as e:\n            out[\"error\"] = repr(e)[:300]\n        return out\n\n    def nb1_cache_info(self):\n        out = {}\n        try:\n            import torch.distributed as dist\n            out[\"rank\"] = dist.get_rank() if dist.is_initialized() else None\n        except Exception as e:\n            out[\"rank_error\"] = repr(e)[:120]\n        try:\n            ctx = self.vllm_config.compilation_config.static_forward_context\n            sig = {}\n            for name, mod in ctx.items():\n                ts = [_t(x) for x in _flat(getattr(mod, \"kv_cache\", None))]\n                if not ts:\n                    continue\n                key = type(mod).__name__ + \"|\" + json.dumps([(x[\"dtype\"], x[\"shape\"]) for x in ts])\n                g = sig.setdefault(key, {\"module\": type(mod).__name__, \"tensors\": ts, \"layers\": 0, \"example\": name})\n                g[\"layers\"] += 1\n            out[\"cache_tensors\"] = list(sig.values())[:40]\n        except Exception as e:\n            out[\"cache_tensors_error\"] = repr(e)[:300]\n        try:\n            agg = {}\n            for name, sp in self.get_kv_cache_spec().items():\n                d = {\"type\": type(sp).__name__}\n                for k in (\"block_size\", \"page_size_bytes\", \"page_size_padded\", \"dtype\", \"num_kv_heads\", \"head_size\", \"mamba_type\"):\n                    try:\n                        v = getattr(sp, k, None)\n                        d[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)\n                    except Exception as e:\n                        d[k] = \"ERR \" + repr(e)[:60]\n                shapes, dtypes = getattr(sp, \"shapes\", None), getattr(sp, \"dtypes\", None)\n                if shapes and dtypes:\n                    try:\n                        d[\"state_dtypes\"] = [str(x).replace(\"torch.\", \"\") for x in dtypes]\n                        d[\"state_shapes\"] = [list(s) for s in shapes]\n                        d[\"unpadded_page_bytes\"] = int(sum(math.prod(s) * x.itemsize for s, x in zip(shapes, dtypes)))\n                    except Exception as e:\n                        d[\"unpadded_error\"] = repr(e)[:80]\n                key = json.dumps(d, sort_keys=True, default=str)\n                a = agg.setdefault(key, dict(d, layers=0, example=name))\n                a[\"layers\"] += 1\n            out[\"kv_cache_specs\"] = list(agg.values())[:40]\n        except Exception as e:\n            out[\"kv_cache_specs_error\"] = repr(e)[:300]\n        try:\n            out[\"num_gpu_blocks\"] = getattr(self.cache_config, \"num_gpu_blocks\", None)\n            out[\"mamba_page_size_padded\"] = getattr(self.cache_config, \"mamba_page_size_padded\", None)\n            out[\"mamba_ssm_cache_dtype\"] = str(getattr(self.cache_config, \"mamba_ssm_cache_dtype\", None))\n            out[\"cache_block_size\"] = getattr(self.cache_config, \"block_size\", None)\n        except Exception as e:\n            out[\"cache_config_error\"] = repr(e)[:200]\n        try:\n            for obj in (getattr(self, \"model_runner\", None), self):\n                cfg = getattr(obj, \"kv_cache_config\", None)\n                if cfg is not None:\n                    out[\"kv_cache_config\"] = {\n                        \"num_blocks\": getattr(cfg, \"num_blocks\", None),\n                        \"groups\": [{\"spec\": type(g.kv_cache_spec).__name__, \"layers\": len(g.layer_names),\n                                    \"block_size\": getattr(g.kv_cache_spec, \"block_size\", None),\n                                    \"page_size_bytes\": getattr(g.kv_cache_spec, \"page_size_bytes\", None)}\n                                   for g in getattr(cfg, \"kv_cache_groups\", [])][:16],\n                        \"tensors\": [{\"size\": getattr(t, \"size\", None), \"shared_by\": len(getattr(t, \"shared_by\", None) or [])}\n                                    for t in getattr(cfg, \"kv_cache_tensors\", [])][:24]}\n                    break\n        except Exception as e:\n            out[\"kv_cache_config_error\"] = repr(e)[:300]\n        try:\n            import torch\n            fr, tot = torch.cuda.mem_get_info()\n            out[\"mem\"] = {\"free_mib\": fr >> 20, \"total_mib\": tot >> 20, \"allocated_mib\": torch.cuda.memory_allocated() >> 20,\n                          \"reserved_mib\": torch.cuda.memory_reserved() >> 20, \"max_allocated_mib\": torch.cuda.max_memory_allocated() >> 20}\n        except Exception as e:\n            out[\"mem_error\"] = repr(e)[:200]\n        return out\n\n\n    def nb4s_env(self, keys=\"\"):\n        \"\"\"E25 NB4S (read-only): this worker's os.environ for the comma-separated `keys` (default CUDA_DEVICE_MAX_CONNECTIONS),\n        all NCCL_* / CUDA_* variables, pid, rank and device. CUDA reads CUDA_DEVICE_MAX_CONNECTIONS at context creation, so\n        the process env (inherited at spawn, before CUDA init) is what is in effect. Never raises.\"\"\"\n        import os as _os\n        out = {\"pid\": _os.getpid()}\n        try:\n            ks = [k for k in str(keys or \"CUDA_DEVICE_MAX_CONNECTIONS\").split(\",\") if k]\n            out[\"env\"] = {k: _os.environ[k] for k in ks if k in _os.environ}\n            out[\"nccl_cuda_env\"] = {k: v for k, v in sorted(_os.environ.items()) if k.startswith((\"NCCL_\", \"CUDA_\"))}\n        except Exception as e:\n            out[\"error\"] = repr(e)[:200]\n        try:\n            import torch.distributed as dist\n            out[\"rank\"] = dist.get_rank() if dist.is_initialized() else None\n        except Exception as e:\n            out[\"rank_error\"] = repr(e)[:120]\n        try:\n            import torch\n            out[\"device\"] = torch.cuda.current_device() if torch.cuda.is_available() else None\n        except Exception as e:\n            out[\"device_error\"] = repr(e)[:120]\n        return out\n"

    def apply_unified(text, patch):
        """Strict unified-diff applier (no fuzz, no offset): every context / removed line must match at its stated line,
        else ValueError. Lines before the first hunk (comments, ---/+++ headers) are ignored."""
        src = text.splitlines(True)
        hunks, cur = [], None
        for ln in patch.splitlines(True):
            if ln.startswith("@@"):
                m = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", ln)
                if not m:
                    raise ValueError("bad hunk header: " + ln[:80])
                cur = [int(m.group(1)), int(m.group(2) or 1), []]
                hunks.append(cur)
            elif cur is not None and ln[:1] in (" ", "-", "+"):
                cur[2].append(ln)
            elif cur is not None and ln.startswith("\\"):
                raise ValueError("'No newline at end of file' hunks are not supported")
        if not hunks:
            raise ValueError("patch has no hunks")
        out, pos = [], 0
        for start, n_old, body in hunks:
            i = start - 1
            if i < pos:
                raise ValueError(f"overlapping hunk at line {start}")
            out += src[pos:i]
            old_seen = 0
            for ln in body:
                tag, s = ln[0], ln[1:]
                if tag in " -":
                    if i >= len(src) or src[i] != s:
                        raise ValueError(f"context mismatch at line {i + 1}: {s[:60]!r}")
                    i += 1
                    old_seen += 1
                if tag in " +":
                    out.append(s)
            if old_seen != n_old:
                raise ValueError(f"hunk at line {start}: {old_seen} old lines, header says {n_old}")
            pos = i
        return "".join(out + src[pos:])


    def apply_60490():
        """After merlin_install, before B1: patch the installed tree's qsa.py (pre-sha must be Merlin ad42e37's), verify the
        post-sha, the marker, a py_compile with the Merlin venv and that `vllm` resolves to the tree; write the read-only
        worker extension next to the tree's vllm/ (on the server's PYTHONPATH). Any miss raises: no boot is started."""
        if FAKE:
            tree = os.environ.get("NB1_FAKE_TREE")
            if not tree:
                raise RuntimeError("PATCH_60490_NO_TREE: NB1_FAKE_TREE is not set (local harness)")
            tree, py = Path(tree), sys.executable
        else:
            eng = merlin_install()
            tree, py = Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python"
        f = tree / P60490_REL
        raw = f.read_bytes()
        h0 = hashlib.sha256(raw).hexdigest()
        rec = {"file": str(f), "sha_before": h0, "pre_expected": P60490_PRE, "post_expected": P60490_POST}
        if h0 == P60490_POST:
            rec["applied"] = "already_applied"
        elif h0 != P60490_PRE:
            raise RuntimeError(f"PATCH_60490_PRE_SHA_MISMATCH: {h0} (expected Merlin ad42e37 {P60490_PRE})")
        else:
            new = apply_unified(raw.decode("utf-8"), P60490).encode("utf-8")
            if hashlib.sha256(new).hexdigest() != P60490_POST:
                raise RuntimeError("PATCH_60490_POST_SHA_MISMATCH: " + hashlib.sha256(new).hexdigest())
            tmp = f.with_name(f.name + ".nb1tmp")
            tmp.write_bytes(new)
            os.replace(tmp, f)
            rec["applied"] = "applied"
        for pyc in (f.parent / "__pycache__").glob("qsa.*.pyc"):
            pyc.unlink()
        data = f.read_bytes()
        rec.update(sha_after=hashlib.sha256(data).hexdigest(), marker=P60490_MARKER.encode() in data)
        ext = tree / "nb1_worker_ext.py"
        ext.write_text(NB1_EXT_SRC)
        rec["worker_ext"] = {"file": str(ext), "sha256": hashlib.sha256(ext.read_bytes()).hexdigest()}
        cp = subprocess.run([py, "-c", "import py_compile, sys\nfor f in sys.argv[1:]: py_compile.compile(f, doraise=True)", str(f), str(ext)],
                            capture_output=True, text=True, timeout=300)
        rec["py_compile"] = {"rc": cp.returncode, "err": cp.stderr[-400:]}
        origin_ok = True
        if not FAKE:
            envp = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
            envp["PYTHONPATH"] = str(tree)            # what Merlin's build() exports for the server
            cp2 = subprocess.run([py, "-c", "import importlib.util as u; print(u.find_spec('vllm').origin); print(u.find_spec('nb1_worker_ext').origin)"],
                                 capture_output=True, text=True, timeout=300, env=envp)
            rec["vllm_origin"] = cp2.stdout.strip()[-400:]
            origin_ok = all(l.startswith(str(tree)) for l in cp2.stdout.strip().splitlines()[-2:]) and len(cp2.stdout.strip().splitlines()) >= 2
        if rec["sha_after"] != P60490_POST or not rec["marker"] or cp.returncode or not origin_ok:
            raise RuntimeError("PATCH_60490_VERIFY_FAILED: " + json.dumps({k: rec.get(k) for k in ("sha_after", "marker", "py_compile", "vllm_origin")})[:600])
        rec["ok"] = True
        rec["forward_test"] = "B1's greedy oracle vs RC2's recorded FP8 rows (unpatched RC2 FP8; B1 adds text-only): B1 oracle.vs_rc2_fp8"
        return rec


    # ---- host memory: previous engine gone, our leaked shm removed, page cache dropped (measured), Merlin's rule
    def cgroup_snapshot(tag):
        """memory.current / stat / events / pressure (first 12 kB each) to hostmem/<tag>.txt; key numbers returned."""
        rec, txt = {"tag": tag}, []
        for n in ("memory.current", "memory.stat", "memory.events", "memory.pressure"):
            try:
                s = (Path("/sys/fs/cgroup") / n).read_text()[:12000]
            except Exception:
                continue
            txt.append(f"--- {n} ---\n{s}")
            if n == "memory.current":
                rec["current_mib"] = int(s.strip()) >> 20
            elif n == "memory.events":
                rec["events"] = {a: int(b) for a, b in (l.split() for l in s.splitlines() if len(l.split()) == 2)}
            elif n == "memory.pressure":
                rec["pressure"] = s.strip().splitlines()[:2]
        try:
            (HOSTMEM / f"{tag}.txt").write_text("\n".join(txt))
        except Exception:
            pass
        rec["host_mem"] = host_mem()
        return rec


    ENGINE_RX = re.compile(r"vllm|PleOffload|ple_offload|EngineCore|merlin_off/venv", re.I)


    def engine_procs():
        """Processes of an engine (vLLM server, workers, PLE offload) other than this notebook; None off Linux."""
        if not Path("/proc/self/stat").exists():
            return None
        me, out = os.getpid(), []
        for d in Path("/proc").glob("[0-9]*"):
            try:
                pid = int(d.name)
                if pid == me:
                    continue
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                if ENGINE_RX.search(cmd) and "py_compile" not in cmd:
                    out.append([pid, cmd[:160]])
            except Exception:
                pass
        return out


    def clear_engine_procs():
        left0 = engine_procs()
        rec = {"found": left0}
        if left0:
            for pid, _ in left0:
                try:
                    os.kill(pid, 9)
                except Exception:
                    pass
            time.sleep(10)
            rec["after_kill"] = engine_procs()
        rec["ok"] = not (rec.get("after_kill") if left0 else left0)
        return rec


    def drop_page_cache():
        """posix_fadvise(DONTNEED) on every checkpoint file (merged dir symlinks resolved + the Model mounts); only after the
        previous server is dead (never on the pinned PLE pages of a live server). A measured attempt, not an assumed fix."""
        if not hasattr(os, "posix_fadvise"):
            return {"available": False}
        t, seen, n, b, err = time.time(), set(), 0, 0, 0
        roots = [CKPT] + [v[0] for v in PARTS.values() if v] + ([PLE_DIR] if PLE_DIR else [])
        for root in roots:
            for dp, _, fns in os.walk(str(root)):
                for x in fns:
                    p = os.path.realpath(os.path.join(dp, x))
                    if p in seen:
                        continue
                    seen.add(p)
                    try:
                        fd = os.open(p, os.O_RDONLY)
                        try:
                            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                            b += os.fstat(fd).st_size
                            n += 1
                        finally:
                            os.close(fd)
                    except OSError:
                        err += 1
        return {"available": True, "files": n, "gib": round(b / 2 ** 30, 1), "errors": err, "sec": round(time.time() - t, 1)}


    def host_ram_gate(tag):
        before = cgroup_snapshot(tag + "_before_drop")
        drops, tries = [drop_page_cache()], 0
        after = host_mem()
        while True:
            av = after.get("avail_mib", after.get("MemAvailable_mib"))
            if av is None:
                ok = bool(FAKE)               # local harness only; on Kaggle an unreadable host is never "enough"
                why = "no cgroup / meminfo reading" + (" (local harness: skipped)" if FAKE else ": HOST_RAM_GATE fails closed")
                break
            if av >= RAM_NEED_MIB:
                ok, why = True, f"avail {av} MiB >= {RAM_NEED_MIB}"
                break
            if FAKE:                          # local / CPU-kit harness only (a 30 GiB CPU box): recorded, never enforced
                ok, why = True, f"avail {av} MiB < {RAM_NEED_MIB} MiB (local harness: recorded, not enforced)"
                break
            tries += 1
            if tries > 3:
                ok, why = False, f"HOST_RAM_GATE: avail {av} MiB < {RAM_NEED_MIB} MiB after {tries} page-cache drops"
                break
            time.sleep(20)
            drops.append(drop_page_cache())
            after = host_mem()
        return {"ok": ok, "why": why, "need_mib": RAM_NEED_MIB, "before": before, "after_drop": cgroup_snapshot(tag + "_after_drop"),
                "drops": drops}


    _boot_gate_rc2, _shm_cleanup_rc2 = boot_gate, shm_cleanup
    BOOT = {"deadline": None, "fired": None, "tag": "boot"}


    def boot_gate():
        """RC2's GPU release gate (< 150 MiB per card), the previous engine's processes gone, then the host-RAM gate."""
        g = _boot_gate_rc2()
        if g.get("ok") is not True:
            return g
        g["engine_procs"] = clear_engine_procs()
        if not g["engine_procs"]["ok"]:
            g["ok"], g["why"] = False, "ENGINE_PROCS_LEFT: " + json.dumps(g["engine_procs"].get("after_kill"))[:300]
            return g
        h = host_ram_gate(BOOT["tag"])
        g["host_ram"] = h
        if not h["ok"]:
            g["ok"], g["why"] = False, h["why"]
        return g


    def shm_cleanup(before):
        """RC2's /dev/shm cleanup (this boot's torch_/psm_/vllm/ple/nccl/sem.mp-/cuda.shm entries) + the loky semaphores the
        RC2 log reported leaked at shutdown (sem.loky-*), only when created during this boot."""
        out = _shm_cleanup_rc2(before)
        out["removed_loky"] = []
        try:
            for p in sorted(SHM.iterdir()):
                if p.name not in before and re.match(r"sem\.loky-", p.name):
                    try:
                        p.unlink()
                        out["removed_loky"].append(p.name)
                    except Exception:
                        pass
        except Exception:
            pass
        return out


    class LoadMonitor(threading.Thread):
        """Every 10 s from launch until ready: per process of the server's session, major faults and /proc/<pid>/io
        (rchar, read_bytes) + the cgroup current; a slow worker (RC2: 873 s vs 125 s) becomes diagnosable."""
        def __init__(self, name):
            super().__init__(daemon=True)
            self.sname, self.stop_ev, self.path, self.last = name, threading.Event(), OUT / f"load_monitor_{name}.jsonl", None

        def sample(self, sid):
            rows = []
            for d in Path("/proc").glob("[0-9]*"):
                try:
                    st = (d / "stat").read_text()
                    rp = st.rindex(")")
                    f = st[rp + 2:].split()
                    if int(f[3]) != sid:
                        continue
                    io = dict(l.split(": ", 1) for l in (d / "io").read_text().splitlines() if ": " in l)
                    rows.append([int(d.name), st[st.index("(") + 1:rp][:24], int(f[9]), int(io.get("rchar", 0)) >> 20,
                                 int(io.get("read_bytes", 0)) >> 20])
                except Exception:
                    pass
            return rows

        def run(self):
            while not self.stop_ev.is_set():
                srv = next((x for x in list(L.LIVE) if x.name == self.sname), None)
                if srv is not None and srv.proc is not None and Path("/proc/self/stat").exists():
                    rec = {"t": round(time.time() - T_START, 1), "procs[pid,comm,majflt,rchar_mib,read_mib]": self.sample(srv.proc.pid),
                           "cgroup_mib": (host_mem() or {}).get("cgroup_cur_mib")}
                    self.last = rec
                    try:
                        with open(self.path, "a") as fh:
                            fh.write(json.dumps(rec) + "\n")
                    except Exception:
                        pass
                    if srv.ready_sec is not None:
                        break
                self.stop_ev.wait(10)


    # ---- per-boot watchdog
    def boot_left():
        d = BOOT["deadline"]
        return min(left(), (d - time.time()) if d else float("inf"))


    def due(sec):
        return boot_left() > sec and not ABORT["why"]


    def watched(body, cap_sec, name):
        """The boot body gets `cap_sec`; every step checks due(); at cap + 300 s the server is killed (hard backstop)."""
        def b(ss):
            BOOT.update(deadline=time.time() + cap_sec, fired=None)

            def fire():
                BOOT["fired"] = name
                for x in ss:
                    try:
                        x.kill()
                    except Exception:
                        pass
            tm = threading.Timer(cap_sec + 300, fire)
            tm.daemon = True
            tm.start()
            try:
                o = body(ss)
                LIVE["out"] = o if isinstance(o, dict) else None   # R2: step() keeps it if anything after the body raises
            finally:
                tm.cancel()
                BOOT["deadline"] = None
            if BOOT["fired"]:
                o["boot_watchdog"] = f"FIRED after {cap_sec + 300:.0f} s: server killed"
            return o
        return b


    # ---- server facts read from the log and from the workers
    def _pick(t, rx, n=6):
        return [l.strip()[:300] for l in t.splitlines() if re.search(rx, l)][:n]


    def server_caps(t, argv=()):
        na = next((l for l in t.splitlines() if "non-default args:" in l), "")
        caps = {"source": "log" if na else "argv"}
        for k in ("max_num_seqs", "max_num_batched_tokens", "block_size", "max_model_len", "kv_cache_memory_bytes", "num_speculative_tokens"):
            m = re.search(rf"'{k}': (\d+)", na)
            caps[k] = int(m.group(1)) if m else None
        for k in ("kv_cache_dtype", "mamba_ssm_cache_dtype", "mamba_cache_mode", "worker_extension_cls"):
            m = re.search(rf"'{k}': '([^']+)'", na)
            caps[k] = m.group(1) if m else None
        m = re.search(r"'language_model_only': (True|False)", na)
        caps["language_model_only"] = m.group(1) == "True" if m else False
        m = re.search(r"'cudagraph_capture_sizes': \[([\d, ]+)\]", na)
        caps["capture_sizes"] = [int(x) for x in m.group(1).split(",")] if m else None
        m = re.findall(r"Setting attention block size to (\d+) tokens", t)
        if m:
            caps["block_size"] = int(m[-1])
        if not na:                                     # fallback (the E23 fake has no args line): the argv, last wins
            a = [str(x) for x in argv]
            for flag, key in (("--max-num-seqs", "max_num_seqs"), ("--block-size", "block_size")):
                idx = [i for i, x in enumerate(a) if x == flag]
                if idx and idx[-1] + 1 < len(a):
                    caps[key] = int(a[idx[-1] + 1])
        caps["block_size"] = caps.get("block_size") or BLOCK
        sizes = caps.get("capture_sizes") or []
        k = caps.get("num_speculative_tokens") or 3
        caps["captures_cover_seqs_x_1pk"] = bool(sizes) and caps.get("max_num_seqs") is not None and max(sizes) >= caps["max_num_seqs"] * (1 + k)
        return caps


    def pool_facts(t):
        f = {}
        m = re.findall(r"GPU KV cache size: ([\d,]+) tokens", t)
        f["kv_tokens"] = int(m[-1].replace(",", "")) if m else None
        m = re.findall(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", t)
        f["max_concurrency"] = float(m[-1][1]) if m else None
        f["max_concurrency_tokens"] = int(m[-1][0].replace(",", "")) if m else None     # R2: M of the line (F_log)
        m = re.findall(r"Padding mamba page size by ([\d.]+)%", t)
        f["mamba_pad_pct"] = float(m[-1]) if m else None
        f["min_block_for_mamba"] = math.ceil(BLOCK / (1 + f["mamba_pad_pct"] / 100)) if m else None
        f["kv_group_sizes"] = (re.findall(r"kv cache group sizes \(([^)]*)\)", t) or [None])[-1]
        f["side_cache_layout"] = _pick(t, r"Side-cache layout", 2)
        f["initial_free_reserved_gib"] = [[float(a), float(b)] for a, b in re.findall(r"Initial free memory ([\d.]+) GiB, reserved ([\d.]+) GiB", t)][:4]
        f["model_load_gib_sec"] = [[float(a), float(b)] for a, b in re.findall(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds", t)][:4]
        f["graph_capture_sec_gib"] = [[int(a), float(b)] for a, b in re.findall(r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB", t)][:4]
        f["rank_load_gib"] = {int(p) * 2 + int(q): float(g) for p, q, g in re.findall(
            r"\(Worker_PP(\d)_TP(\d)\S*(?: pid=\d+)?\) INFO [^\n]*?Model loading took ([\d.]+) GiB memory", t)}
        f["rank_load_sec"] = {int(p) * 2 + int(q): float(g) for p, q, g in re.findall(
            r"\(Worker_PP(\d)_TP(\d)\S*(?: pid=\d+)?\) INFO [^\n]*?Model loading took [\d.]+ GiB memory and ([\d.]+) seconds", t)}
        f["rank_graph_gib"] = {int(p) * 2 + int(q): float(g) for p, q, g in re.findall(
            r"\(Worker_PP(\d)_TP(\d)\S*(?: pid=\d+)?\) INFO [^\n]*?Graph capturing finished in \d+ secs, took ([\d.]+) GiB", t)}
        f["ssm_dtype_lines"] = _pick(t, r"mamba_ssm_dtype|mamba-ssm-cache-dtype")
        f["gdn_fallback_lines"] = _pick(t, r"Falling back to the Triton GDN spec decode path")
        f["share_workspace_lines"] = _pick(t, r"MERLIN_SHARE_DRAFT_WORKSPACE")
        f["mm_warmup_lines"] = _pick(t, r"Multi-modal warmup completed")
        f["host_ram_lines"] = _pick(t, r"host memory available now|Cgroup memory preflight|Skipping auto-prefetch", 6)
        f["sched_warn_lines"] = _pick(t, r"max_num_scheduled_tokens", 2)
        f["oom_lines"] = _pick(t, r"OutOfMemoryError|CUDA out of memory", 4)
        f["static_scale_lines"] = _pick(t, r"applied static K/V scales", 8)
        f["default_scale_lines"] = _pick(t, r"DEFAULT 1\.0|no static calibration", 4)    # audit 2: the MTP layer's QSA FP8 KV
        f["ext_lines"] = _pick(t, r"Injected .*NB1WorkerExt|nb1_worker_ext", 2)
        f["shm_leak_lines"] = _pick(t, r"leaked (semaphore|shared_memory) objects", 4)
        return f


    def graph_facts(t):
        caps = sorted({(a, int(c)) for a, b, c in re.findall(r"Capturing CUDA graphs \((\w+)\):\s*100%\|[^|]*\|\s*(\d+)/(\d+)", t)})
        return {"captured": caps, "resolved_modes": _pick(t, r"resolved cudagraph modes before capture", 8),
                "draft_identity": _pick(t, r"e2-identity\[.*role=draft", 4), "recoverssm_commit": _pick(t, r"\[recoverssm-commit\]"),
                "recoverssm_path": _pick(t, r"RecoverSSM path taken", 4), "jit": jit_kernels(t)}


    def jit_kernels(t):
        return re.findall(r"JIT compilation during inference: (\w+)", t)


    def nccl_transport(t):
        via = Counter(re.findall(r"NCCL INFO Channel \d+/\d+ ?: ?\d+\[\d+\] -> \d+\[\d+\] (?:\[\w+\] )?via (\S+)", t))
        return {"channel_via": dict(via), "net": sorted(set(re.findall(r"NCCL INFO (Using network \w+|NET/\w+ : Using [^\n]{0,60})", t)))[:6],
                "comm_inits": len(re.findall(r"NCCL INFO comm 0x[0-9a-f]+ rank \d+ nranks \d+", t)),
                "pynccl_groups": _pick(t, r"all-reduce backends .* for group", 6),
                "debug_on": "NCCL INFO" in t}


    def rpc_cache_info(port):
        """The read-only worker extension over /collective_rpc (dev mode): actual cache tensor dtypes, specs, free memory."""
        body = {"method": "nb1_cache_info", "timeout": 120}
        try:
            r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/collective_rpc", data=json.dumps(body).encode(),
                                                   headers={"Content-Type": "application/json"}), timeout=180).read())
            res = r.get("results") or []
        except Exception as e:
            return {"error": repr(e)[:300]}
        out = {"ranks": res}
        dts = sorted({x["dtype"] for w in res if isinstance(w, dict) for g in (w.get("cache_tensors") or [])
                      if "mamba" in g.get("module", "").lower() or "gdn" in g.get("module", "").lower() or "linear" in g.get("module", "").lower()
                      for x in g.get("tensors") or []})
        out["recurrent_tensor_dtypes"] = dts
        out["num_gpu_blocks"] = sorted({w.get("num_gpu_blocks") for w in res if isinstance(w, dict)} - {None})
        out["free_mib_per_rank"] = [((w or {}).get("mem") or {}).get("free_mib") for w in res if isinstance(w, dict)]
        pads = []
        for w in res:
            for s in ((w or {}).get("kv_cache_specs") or []) if isinstance(w, dict) else []:
                if s.get("unpadded_page_bytes") and s.get("page_size_bytes"):
                    pads.append({"rank": w.get("rank"), "type": s.get("type"), "layers": s.get("layers"), "state_dtypes": s.get("state_dtypes"),
                                 "padded": s["page_size_bytes"], "unpadded": s["unpadded_page_bytes"],
                                 "pad_pct": round(100 * (s["page_size_bytes"] - s["unpadded_page_bytes"]) / s["unpadded_page_bytes"], 2)})
        out["mamba_padding"] = pads[:16]
        return out


    def rpc(port, method, *args, timeout=120):
        """POST /collective_rpc (dev mode): a worker method by name with string args; the per-rank results or the error."""
        body = {"method": method, "args": [str(a) for a in args], "timeout": timeout}
        try:
            r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/collective_rpc", data=json.dumps(body).encode(),
                                                   headers={"Content-Type": "application/json"}), timeout=timeout + 60).read())
            return {"results": r.get("results")}
        except Exception as e:
            return {"error": repr(e)[:300]}








    def scales_dump():
        """The static QSA K/V scales file (all values) + which layers it covers; the MTP layer is not in it (default 1.0)."""
        p = CKPT / "qsa_kv_scales_262k.json"
        try:
            d = json.loads(p.read_text())
        except Exception as e:
            return {"error": repr(e)[:200]}
        lay = sorted(int(m.group(1)) for k in d for m in [re.search(r"layers\.(\d+)\.", k)] if m)
        ks = [v.get("k_scale") for v in d.values() if isinstance(v, dict)]
        vs = [v.get("v_scale") for v in d.values() if isinstance(v, dict)]
        return {"file": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "values": d, "layers": lay, "n": len(d),
                "k_scale_min_max": [min(ks), max(ks)] if ks else None, "v_scale_min_max": [min(vs), max(vs)] if vs else None,
                "mtp_layer_in_file": any("mtp" in k for k in d)}


    def free_per_card(rec, name):
        """MiB free per card = total - the highest used reading of this boot (after ready, end, every window peak)."""
        srv = ((rec or {}).get("servers") or {}).get(name) or {}
        rows = [r for r in (srv.get("gpu_mem_after_ready"), srv.get("gpu_mem_end")) if r]
        if not rows:
            return None
        tot = {i: t for row in rows for i, _, t in row}
        used = {i: max(u for row in rows for j, u, _ in row if j == i) for i in tot}
        st = (rec or {}).get("steady")             # R2: nb1_ladder's steady holds "_plan" (list), "_fit" (dict), "_seqs_cap" (int)
        ws = list(st.values()) if isinstance(st, dict) else []
        for w in ws + [(rec or {}).get("profile")]:
            if not isinstance(w, dict):            # R2: only window dicts carry gpu_mem_peak_mib
                continue
            pk = w.get("gpu_mem_peak_mib")
            for i, u in enumerate(pk if isinstance(pk, list) else []):
                if i in used and isinstance(u, (int, float)):
                    used[i] = max(used[i], u)
        return [tot[i] - used[i] for i in sorted(tot)]


    def free_after_ready(rec, name):
        srv = ((rec or {}).get("servers") or {}).get(name) or {}
        row = srv.get("gpu_mem_after_ready")
        return [t - u for _, u, t in row] if row else None


    # ---- KV accounting: pool blocks from usage denominators; cold-cache cumulative admission (physical snapshots)
    def kvu(port):
        return L.prom(port).get(L.KVU)
























    F_MAX_BLOCKS = 40                                # R2: a per-request fixed-block F above 40 blocks is not physical here
    FIT_MIN_SAMPLES = 3                              # R2 audit N1-6: a fit is trusted only with >= 3 all-resident snapshots,
    FIT_MAX_RESID_BLOCKS = 2.0                       # a max |residual| <= 2 blocks
    FIT_LOG_TOL_BLOCKS = 1                           # and |ceil(fit F) - log F| <= 1 block
    F_LOG_SOURCE = "server max-concurrency line (this boot)"












    def profiler_tools():
        """ncu / nsys availability (DRAM bytes per stage need a server LAUNCHED under one of them: a separate boot)."""
        out = {}
        cands = ["ncu", "nsys", "/usr/local/cuda/bin/ncu", "/usr/local/cuda/bin/nsys"]
        try:
            eng = MERLIN_ENG if isinstance(MERLIN_ENG, dict) else {}
            if eng.get("cuda_home"):
                cands += [eng["cuda_home"] + "/bin/ncu", eng["cuda_home"] + "/bin/nsys"]
        except Exception:
            pass
        for c in cands:
            path = shutil.which(c) if "/" not in c else (c if os.path.exists(c) else None)
            if path:
                try:
                    v = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=60).stdout.strip()[:200]
                except Exception as e:
                    v = repr(e)[:100]
                out[c] = {"path": path, "version": v}
        return {"found": out, "dram_bytes": "not measured in NB1: ncu / nsys cannot attach to a running vLLM server; DRAM bytes per "
                "stage need a boot launched under ncu (dram__bytes_read.sum on the Marlin MoE + dense kernels) or nsys: NB3 / NB4 item",
                "torch_profiler_window": "the PP2 trace window runs after the ladder, on its own streams; no benchmark window overlaps it"}


    # ---- requests with usage (cached tokens), greedy logprobs and teacher-forced rows with the cache counters








    # ---- JIT warm-up (excluded from every metric) and per-position acceptance
    def warmup(port, n, secs=None):
        """Multi-stream 2k decode, ramped 1 -> n streams, excluded from every metric: compiles the kernels RC2 saw compiled
        during inference (_fill_num_accepted_kernel, _causal_conv1d_update_kernel, _gdn_recoverssm_verify_kernel,
        _ple_conv_kernel, _ple_conv_writeback_kernel) before any measured window."""
        secs = secs or (45.0 if not FAKE else 4.0)
        srv = next((x for x in list(L.LIVE) if x.port == port), None)
        j0 = jit_kernels(srv.log()) if srv else []
        ss, t0 = [], time.time()
        try:
            for i in range(n):
                g = L.Stream(port, PROMPTS["2k"][i % len(PROMPTS["2k"])], 4096, temperature=TEMP, seed=900 + i, ignore_eos=True)
                g.start()
                ss.append(g)
                time.sleep(min(3.0, secs / max(1, 2 * n)))
            while time.time() - t0 < secs:
                time.sleep(1)
        finally:
            for g in ss:
                g.stop()
            for g in ss:
                g.join(10)
        L.wait_idle(port, 60)
        j1 = jit_kernels(srv.log()) if srv else []
        return {"streams": n, "sec": round(time.time() - t0, 1), "jit_before": j0, "jit_compiled_in_warmup": j1[len(j0):],
                "errors": sorted({g.err for g in ss if g.err})[:3]}


    def accept_pos(port):
        try:
            txt = L.U.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=15).read().decode("utf-8", "replace")
        except Exception:
            return None
        pos = {}
        for m in re.finditer(r"^vllm:spec_decode_num_accepted_tokens_per_pos_total\{([^}]*)\}\s+(\S+)", txt, re.M):
            p = re.search(r'position="(\d+)"', m.group(1))
            if p:
                pos[int(p.group(1))] = pos.get(int(p.group(1)), 0.0) + float(m.group(2))
        return {"pos": pos, "drafts": L.prom(port).get(L.DRF, 0.0)}


    def _prefill_done(r, n):
        tail = ((r.get("admission") or {}).get("tail") or [])
        t = tail[-1] if tail else None
        return bool(t) and t[1] == n and t[2] == 0 and t[3] >= n


    def _req_lines():
        try:
            with open(REQ_LOG) as fh:
                return sum(1 for _ in fh)
        except Exception:
            return 0


    def _ttfts(port, start_line):
        out = []
        try:
            with open(REQ_LOG) as fh:
                for i, l in enumerate(fh):
                    if i >= start_line:
                        r = json.loads(l)
                        if r.get("port") == port and r.get("ttft") is not None:
                            out.append(r["ttft"])
        except Exception:
            pass
        return out


    def _counters(port):
        try:
            m = L.prom(port)
            return m.get(L.PROMPT, 0.0), m.get("vllm:prefix_cache_hits_total", 0.0)
        except Exception:
            return None, None


    def window(port, s, band, n, max_new=None, secs=None):
        """One steady window, timed only after every stream finished admission + prefill (the lab's settle needs all running
        streams to have emitted and be gaining, then an explicit check of the last admission sample; Merlin's cost-aware
        admission may hold a cold miss up to 120 s, inside the lab's >= 420 s admission limit); a window with prefill inside is
        retried once. Recorded: tick_ms (one engine step = one micro-batch of C/2 requests under PP2) AND cycle_ms (= PP x
        tick, the per-request advance time), per-request TTFT, uncached prefill tok/s (prompt minus prefix-hit tokens),
        retained (idle) blocks before the window, per-position acceptance."""
        retained_before = None
        if L.wait_idle(port, 60):
            retained_before = kvu(port)
        def run():
            if max_new is None:                 # RC2 / E23 levels(): MAX_NEW[band], WIN, the budget / KV-bound guards
                return levels(port, band, [n], server=s, temperature=TEMP, prefill_tok_s=PREFILL["tok_s"]).get(f"{band}@{n}") or {}
            return L.steady(port, PROMPTS[band], n, secs=secs or WIN, max_new=max_new, temperature=TEMP, prefill_tok_s=PREFILL["tok_s"])
        a0, c0, l0 = accept_pos(port), _counters(port), _req_lines()
        r = run()
        if r.get("status") == "INCONCLUSIVE_PREFILL_IN_WINDOW" and due(WIN + 120 + n * BANDS[band] / max(500.0, PREFILL["tok_s"])):
            c0, l0 = _counters(port), _req_lines()
            r = dict(run(), retried_after="INCONCLUSIVE_PREFILL_IN_WINDOW")
        a1, c1 = accept_pos(port), _counters(port)
        r["prefill_done_before_window"] = _prefill_done(r, n)
        r["running_eq_level"] = r.get("running_min") == r.get("running_max") == n
        if r.get("status") == "STEADY" and not (r["running_eq_level"] and r["prefill_done_before_window"]):
            r["status"] = "INCONCLUSIVE_RUNNING_NE_LEVEL" if not r["running_eq_level"] else "INCONCLUSIVE_PREFILL_NOT_DONE"
        if r.get("step_ms"):
            r["tick_ms"] = r["step_ms"]
            r["cycle_ms"] = round(PP * r["step_ms"], 1)
            r["requests_per_tick"] = round(n / PP, 2)
            r["microbatches_per_s"] = round(1000.0 / r["step_ms"], 2)
            r["wall_clock_tok_s"] = r.get("tok_s")
            r["tick_note"] = ("tick_ms = step_ms = one PP tick (one completed micro-batch of C/2 requests); cycle_ms = 2 ticks = the "
                              "per-request advance time; microbatches_per_s = completed micro-batches per second; wall_clock_tok_s = "
                              "generated tokens / window seconds")
        adm = (r.get("admission") or {}).get("sec")
        if None not in (c0[0], c1[0]) and adm:
            unc = (c1[0] - c0[0]) - (c1[1] - c0[1])
            r["prefill_uncached_tokens"] = int(unc)
            r["prefill_tok_s_uncached"] = round(unc / adm, 1)
            r["prefill_note"] = "admission.prefill_tok_s counts cached prompt tokens; prefill_tok_s_uncached does not (incl. a retry)"
        r["ttft_per_request"] = _ttfts(port, l0)
        if retained_before is not None:
            r["retained_usage_before"] = retained_before
        if a0 and a1 and a1["drafts"] > a0["drafts"]:
            r["accept_rate_per_position_incl_admission"] = {p: round((a1["pos"].get(p, 0) - a0["pos"].get(p, 0)) / (a1["drafts"] - a0["drafts"]), 3)
                                                            for p in sorted(a1["pos"])}
        return r




    # ---- prefix continuation (2-turn replay) and after an abort






    # ---- PP2 trace window: every rank's trace, stage overlap, idle, verify rows per micro-batch
    SCOPE_ROWS = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$")












    CAPTURE_SIZES = None
    ROUTING_FILE = OUT / "nb1_routing_ids.npz"   # NB3 reads it (dataset foysalemonshanto/e25-nb1-routing-ids after a run)






    def quality_gate(o, ref):
        """RC2's fp8_gate with the reference as a parameter (B1 vs RC2's recorded FP8 rows, B2 / B3 vs B1)."""
        vs = ((o.get("oracle") or {}).get("vs_" + ref) or {}).get("status")
        if (o.get("sampled") or {}).get("ok") is not True:
            return "FAIL_SAMPLED"
        if (o.get("probes") or {}).get("ok") is not True and not FAKE:
            return "FAIL_PROBES"
        orc = o.get("oracle") or {}
        if any(orc.get("errors") or []) or any(orc.get("garbage") or []) or vs == "FAIL_HIGH_MARGIN":
            return "FAIL"
        if orc.get("complete") is True and vs in CLEAN:
            return "PASS"
        return "INCONCLUSIVE"


    def flag_map(argv):
        m, a, i = {}, [str(x) for x in argv], 0
        while i < len(a):
            if a[i].startswith("--"):
                k, v = a[i], True
                if i + 1 < len(a) and not a[i + 1].startswith("--"):
                    v = a[i + 1]
                    i += 1
                m[k] = v
            i += 1
        return m


    def config_diff(argv, env, kv_bytes):
        """Effective flags (last occurrence wins) and env vs RC2's real FP8 boot."""
        ref, new = flag_map(RC2_FP8["argv_tail"]), flag_map(argv)
        skip = {"--port", "--served-model-name", "--host"}
        flags = {k: [ref.get(k), new.get(k)] for k in sorted(set(ref) | set(new)) if k not in skip and ref.get(k) != new.get(k)}
        keys = set(RC2_FP8["env"]) | {k for k in env if k.startswith(("VLLM_", "NCCL_", "KV_", "P2P", "SCALES", "MERLIN_"))}
        envd = {k: [RC2_FP8["env"].get(k), env.get(k)] for k in sorted(keys)
                if k not in ("VLLM_PORT", "MASTER_PORT") and RC2_FP8["env"].get(k) != env.get(k)}
        return {"flags": flags, "env": envd, "kv_bytes": [RC2_FP8["kv_bytes"], kv_bytes]}


    def boot_facts(s, o, name):
        t = s.log() or ""
        o["caps"] = server_caps(t, s.argv)
        o["pool"] = pool_facts(t)
        o["graphs"] = graph_facts(t)
        o["nccl"] = nccl_transport(t)
        o["workers"] = rpc_cache_info(s.port)
        o["host_after_ready"] = cgroup_snapshot(name + "_after_ready")
        env = dict((R["setup"].get("merlin_plans") or {}).get(name, {}).get("env_keys") or {})
        env.update(NB1_ENV.get(name) or {})
        o["config_diff"] = config_diff(s.argv, env, NB1_KV.get(name))
        return o
    save()


if not RERUN:
    # ==================================================================== CELL 7 — E25 NB4T: the arms Q / I8 / P26 / P27 / C12, the plan (NEW code)
    import importlib, base64, zlib
    NB4T_LIB_SHA = "11e39802758b5e9472c9d341f8a909f4d170f23d4fb8e756ae2751c914c2365e"                 # nb4t_lib.py (new), nb4_timeline.py (agent C) and nb4_ppack.py (rev 2) = NB4S's:
    NB4T_LIB_SRC = zlib.decompress(base64.b64decode("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")).decode("utf-8")   # verbatim, zlib + base64, sha-checked
    TL_SHA = "43a0848def257e70565285b6e172566a0480dc523c76eab100ed536a5a575f5e"
    TL_SRC = zlib.decompress(base64.b64decode("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")).decode("utf-8")
    PPACK_SHA = "8dadf0763d43d22966603745198ad327ed6ed36001cc60a79925ea18818f2dea"
    PPACK_SRC = zlib.decompress(base64.b64decode("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")).decode("utf-8")
    for _src, _sha, _n in ((NB4T_LIB_SRC, NB4T_LIB_SHA, "nb4t_lib"), (TL_SRC, TL_SHA, "nb4_timeline"), (PPACK_SRC, PPACK_SHA, "nb4_ppack")):
        if hashlib.sha256(_src.encode("utf-8")).hexdigest() != _sha:
            raise RuntimeError(f"EMBED_SHA_MISMATCH: {_n}")
        (TMP / f"{_n}.py").write_text(_src, encoding="utf-8")
    if str(TMP) not in sys.path:
        sys.path.insert(0, str(TMP))
    N = importlib.import_module("nb4t_lib")
    TL = importlib.import_module("nb4_timeline")
    P = importlib.import_module("nb4_ppack")
    NB4T = R.setdefault("nb4t", {})
    NB4T.update(question="on top of Q (NB4S's B + Q-defer), which next lever speeds decode: the INT8 draft LM head at K3 (I8), the PP "
                         "layer partition 26,22 / 27,21 (P26 / P27), and what do 12 streams give (C12, measure only)?",
                lib=N.LIB_VERSION, lib_sha256=NB4T_LIB_SHA, timeline_sha256=TL_SHA, ppack_sha256=PPACK_SHA,
                ppack_file_shas={"pp_pre": P.PP_PRE, "pp_post": P.PP_POST, "mr_pre": P.MR_PRE, "mr_post": P.MR_POST,
                                 "gw_pre": P.GW_PRE, "gw_post": P.GW_POST},
                nb4s_notebook_sha256="fa9f35ca43ec0c8108327a840a9f6c28d2891e1313c131b3ec56b379048f964a", e18_r2_cell0_sha256="abe694be2a35b4dc00bfbf18fe2fdb0916bff15d74f502b5f5836db30cd46ec0", merlin_flags=json.loads(r"""{"Q": {"change": "nb4_ppack rev 2 patch + NB4_PPDEFER=1 (NB4S QD)", "language_model_only": "already in Q: NB4S B = NB1 R2 B3 carries --language-model-only (caps.language_model_only true in the NB4S run), so P26 / P27 change the partition only"}, "I8": {"knob": "env DRAFT_HEAD=int8, an input of Merlin's build()", "merlin": ["tools/merlin/launch.py:319-322 (DRAFT_HEAD read: bf16 | int8 | int4; draft_spec = ',\"draft_lm_head_quant\":\"int8\"')", "tools/merlin/launch.py:441 (--speculative-config {method mtp, K, probabilistic + draft_spec})", "tools/merlin/registry.py:405 (DRAFT_HEAD_DEFAULT = 'bf16')"], "engine": ["vllm/config/speculative.py:610 (draft_lm_head_quant: 'int4' | 'int8' | None)", "vllm/v1/worker/gpu/spec_decode/speculator.py:345 + 547-603 (_maybe_quantize_draft_lm_head: last PP stage only; refuses local argmax / synthetic / adaptive verification)", "vllm/model_executor/layers/draft_lm_head_quant.py:75-125 (QuantizedDraftLMHead; logs 'Draft lm_head: INT8 weight-only copy of the NxK shard (M MiB; the target keeps its torch.bfloat16 head)')", "vllm/models/qwen4_exp/nvidia/mtp.py:497 (supports_draft_lm_head_quant = True)"], "repack_preflight": "not needed: the INT8 copy is packed on the CPU by marlin_quantize (model_executor/layers/quantization/utils/marlin_utils_test.py:127-161: gptq_quantize_weights + marlin_weights + marlin_permute_scales, pure torch) and run by apply_gptq_marlin_linear; draft_lm_head_quant.py never calls gptq_marlin_repack. NB2 SMOKE ran K2 + INT8 on GPU (head read-back PASS, E25_GPU_wave1_NB3_NB2SMOKE). The arm's read-back is the proof it is used.", "readback": "argv spec draft_lm_head_quant=int8 (launch check) + non-default args spec + 'Draft lm_head: INT8' lines"}, "P26": {"knob": "env VLLM_PP_LAYER_PARTITION=26,22 exported after build() (NB1_ENV)", "merlin": ["tools/merlin/launch.py:373-374 (the profile layout sets 25,23; flash-next has no PP_PARTITION knob)", "vllm/envs.py:895", "vllm/distributed/utils.py:126-170 (get_pp_indices; the sum must be 48)", "vllm/model_executor/models/config.py:1030-1046 (every PLE layer must live on PP rank 0)", "vllm/compilation/caching.py:572-576 (the partition is a compile-cache factor: a recompile boot)"], "readback": "worker env (nb4s_env RPC) + 'pipeline layers a-b' per stage + pool blocks, per-rank free, timeline"}, "P27": {"knob": "env VLLM_PP_LAYER_PARTITION=27,21", "skip_rule": "skipped when P26 OOMs (any launch try, or BODY_OOM)"}, "C12": {"knob": "trailing --max-num-seqs 12 (Merlin's profile passes 8 at launch.py:453; extra args come last, launch.py:111, so the last occurrence wins)", "readback": "launch check + non-default args max_num_seqs 12 + running == 12 in every window"}}"""))
    B3_PIN = KV_FP8 + (1 << 30)                 # NB1 R2 B3 = NB4S B: 3,553,580,524 B (FP8 KV, 215 blocks in the NB4S run)
    if B3_PIN != 3553580524:
        raise RuntimeError(f"B3_PIN {B3_PIN} != 3553580524")
    Q_EXTRA = EXT_FLAG + ["--language-model-only", "--profiler-config", PROFILER_CFG]   # = NB4S B_EXTRA (NB1 R2 B3)
    Q_BASE_ENV = dict(DIAG_ENV)                                                      # = NB4S B_ENV
    ARMS = N.arm_plan(Q_EXTRA, Q_BASE_ENV, B3_PIN, q_env=P.server_env("Q-defer"))
    N.check_arm_plan(ARMS)                      # never-do list + "each arm = Q + exactly one change" (raises: no boot)
    NB4T["arms"] = {k: {x: a[x] for x in ("name", "port", "env", "build_env", "change", "expect", "expect_cfg", "extra", "kv_bytes",
                                          "needs_patch", "streams", "windows", "est_min", "cap_min", "title", "lever", "measure_only")}
                    for k, a in ARMS.items()}
    NB4T["order"] = list(N.ORDER)
    NB4T["oom_policy"] = ("NB4S's one policy for every arm: a launch OOM (OOM line anywhere in the launch's full log, no device fault) -> "
                          "one retry 0.25 GiB lower on a fresh port and launch name; every later arm starts at Q's final pin; a body "
                          "OOM never retries (BODY_OOM_<item>); a P26 OOM (any try or BODY_OOM) skips P27")
    NB4T["never_do"] = {"env": sorted(N.NEVER_DO_ENV), "argv": list(N.NEVER_DO_ARGS), "argv_values": N.NEVER_DO_ARG_VALUES}
    NB4T["budget"] = {"hard_stop_min": BUDGET / 60, "start_reserve_sec": N.START_RESERVE_SEC, "body_reserve_sec": N.BODY_RESERVE_SEC,
                      "priority": "Q > I8 > P26 > P27 > C12 (an arm starts only with est_min x 60 + 300 s left)"}
    ARM_OF = {}                                 # launch name (retries included) -> arm
    NB4T_BUILD_ENV = {}                         # launch name -> Merlin build() knobs (CELL 2's E25 NB4T edit reads it)
    PIN = {"kv_bytes": B3_PIN, "source": "NB1 R2 B3 = NB4S B"}
    NB4T_READY_SEC = 20 * 60                    # readiness gate per boot (NB4S: 630 s for the first, compiling boot; 231-249 s later)
    CUR = {"item": None, "kv_bytes": None}
    _HIST = {}


    def _tree_py():
        eng = merlin_install()
        return Path(eng["kv"]["MERLIN_TREE"]), eng["kv"]["MERLIN_VENV"] + "/bin/python"


    def _ppack_apply():
        tree, py = _tree_py()
        return P.apply_ppack_patch(tree, py)


    def _ppack_revert():
        tree, py = _tree_py()
        rec = P.revert_ppack_patch(tree, py)
        v = P.verify_ppack_patch(tree)              # the pre-shas, verified again after the revert
        if v["state"] != "not_applied":
            raise RuntimeError("REVERT_NOT_PRISTINE: " + json.dumps(v)[:400])
        rec["verify_after"] = v
        return rec


    PSEQ = N.PatchSeq(_ppack_apply, _ppack_revert)


    def _prom_nb4t(port):
        """e23_lab.prom (same dict, one HTTP read) + the iteration-token histogram buckets with their read time (NB4S)."""
        txt = L.U.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=15).read().decode("utf-8", "replace")
        agg, buckets = N.prom_parse(txt)
        h = _HIST.setdefault(port, [])
        h.append((time.perf_counter(), buckets))
        del h[:-600]
        return agg


    L.prom = _prom_nb4t


    def _hist_at(port, t):
        """The latest histogram snapshot read at or before perf_counter time t (steady() reads m0 then sets t0)."""
        best = None
        for ts, b in _HIST.get(port, []):
            if ts <= t + 0.01:
                best = (ts, b)
        return best if best and t - best[0] < 1.0 else None


    def ensure_prompts(n):
        """C12: n distinct prompt variants per measured band (L.steady / profiled_load take prompts[i % len]); variants 0-7 stay
        NB4S's, so the 8-stream windows of every other arm see exactly the NB4S prompts."""
        for b in ("8k", "40k"):
            while len(PROMPTS[b]) < n:
                PROMPTS[b].append(prompt(BANDS[b], len(PROMPTS[b])))
        return {b: len(PROMPTS[b]) for b in ("8k", "40k")}


    _merlin_fn_server_nb1 = merlin_fn_server       # RC2 CELL 2 + NB1's per-boot hooks + NB4T's build-knob edit


    def merlin_fn_server(name, kv, port, kv_bytes):
        """E25 NB4T: NB1's Merlin plan, then (fail closed: no server is started) nb4t_lib.launch_check on the FULL final env /
        argv of THIS launch: never-do items, every watched env key (NB4_PPDEFER=1, the partition), the effective
        --speculative-config draft head, the last --max-num-seqs, --language-model-only, the pin."""
        s = _merlin_fn_server_nb1(name, kv, port, kv_bytes)
        arm = ARM_OF.get(name)
        popped = {}
        for w, want in ((arm or {}).get("expect") or {}).items():
            if want is None and w in s.env:              # inherited from the image env, not planned: removed (recorded)
                popped[w] = s.env.pop(w)
        bad = N.launch_check(arm, s.env, s.argv, kv_bytes) if arm else [f"launch {name} has no arm"]
        NB4T.setdefault("launch", {})[name] = {
            "arm": (arm or {}).get("key"), "port": port, "kv_bytes": kv_bytes, "never_do_and_plan": bad,
            "build_env": NB4T_BUILD_ENV.get(name), "spec": N.spec_from_argv(s.argv), "max_num_seqs": N.last_value(s.argv, "--max-num-seqs"),
            "env": {k: s.env[k] for k in sorted(s.env) if k.startswith(("CUDA_", "NCCL_", "NB4_", "VLLM_SERVER_DEV", "VLLM_USE_V2",
                                                                        "VLLM_PP_LAYER_PARTITION", "DRAFT_HEAD", "P2P"))},
            "argv_tail": s.argv[-14:], "help": HELP.get(name), "popped_unplanned_env": popped}
        save()
        if bad:
            raise RuntimeError("NB4T_LAUNCH_CHECK: " + "; ".join(bad))
        return s


    def free_min(port):
        """Free MiB per card = min(nvidia-smi, the worker's own torch free memory of the same rank); both recorded."""
        smi = L.gpu_mem() or []
        w = rpc_cache_info(port)
        wf = w.get("free_mib_per_rank") if isinstance(w, dict) else None
        sf = [t_ - u for _, u, t_ in smi] if smi else None
        mins = None
        if sf and wf and len(sf) == len(wf):
            mins = [min(a, b) if isinstance(b, (int, float)) else a for a, b in zip(sf, wf)]
        return {"nvidia_smi": sf, "worker": wf, "min": mins or sf or wf}


    def proc_environ():
        """Side record: the watched keys in /proc/<pid>/environ of every engine process (server, workers, PLE)."""
        blobs = {}
        for pid, _ in (engine_procs() or [])[:24]:
            try:
                blobs[pid] = Path(f"/proc/{pid}/environ").read_bytes()
            except Exception:
                blobs[pid] = None
        return blobs


    def _live(o, key, val):
        """Every item lands in the record and in the live step record before save() (a hard kill keeps it)."""
        o[key] = val
        rec = LIVE.get("rec")
        if rec is not None:
            rec.setdefault("live", {})[key] = val
        save()
        return val


    def q_readback(s, arm):
        """The 4 'NB4_PPQ init' lines (every rank must agree) + the Q-defer flush ratios, via nb4_ppack.readback_label on the
        launch's whole log (every NB4T arm runs Q-defer)."""
        try:
            lab, det = P.readback_label(s.log() or "", arm["q_label"])
        except Exception as e:
            lab, det = "NOT_INTEGRATED", {"error": f"{type(e).__name__}: {e}"[:300]}
        return {"label": lab, "detail": det}


    def hist_window(port, s, band, n):
        """NB1's window() (RC2 levels / lab steady; running == level, prefill done, per-position acceptance, tick / cycle) + the
        window's micro-batch split from the iteration-token histogram at steady()'s own t0 / t1 (recorded, not gating)."""
        r = window(port, s, band, n)
        h0, h1 = (_hist_at(port, r["t0"]) if r.get("t0") else None), (_hist_at(port, r["t1"]) if r.get("t1") else None)
        d = N.hist_delta(h0[1], h1[1]) if h0 and h1 else None
        r["parity"] = N.parity_from_hist(d, n, PP, 3)
        return r


    def profile_window(port, arm):
        """One profiled 8k window at the arm's stream count: lab profiled_load (settled decode, then the bounded POST
        /start_profile {"delay_iterations": 20, "max_iterations": 40}, 6 s, /stop_profile); the traces are kept per arm."""
        n = arm["streams"]
        shutil.rmtree(PROF_DIR, ignore_errors=True)
        PROF_DIR.mkdir(parents=True, exist_ok=True)
        rec = L.profiled_load(port, PROMPTS["8k"], n, PROF_DIR, secs=6.0, max_new=MAX_NEW["8k"], spec=True, prefill_tok_s=PREFILL["tok_s"])
        rec["profile_request"] = {"delay_iterations": 20, "max_iterations": 40, "secs": 6.0, "level": n, "band": "8k"}
        if "trace" in rec:
            rec["lab_trace_summary"] = rec.pop("trace")
            rec["lab_trace_note"] = "e23_lab rule: its PIECEWISE verdict is a parser artefact here (graphs research §0.1); nb4_timeline decides"
        g = L.gpu_mem()
        if g:
            rec["gpu_mem_peak_mib"] = [x[1] for x in g]
        kept = TRACES / f"profile_{arm['key'].lower()}"
        shutil.rmtree(kept, ignore_errors=True)
        kept.mkdir(parents=True, exist_ok=True)
        rec["trace_files"] = []
        for f in sorted(PROF_DIR.rglob("*.pt.trace.json*")):
            if f.stat().st_size < 120e6:
                shutil.copy2(f, kept / f.name)
                rec["trace_files"].append([f.name, f.stat().st_size])
        rec["trace_dir"] = str(kept)
        shutil.rmtree(PROF_DIR, ignore_errors=True)
        return rec


    def win_need(wk, n):
        """Seconds a window needs (NB4S's WIN_NEED at n streams)."""
        if wk.startswith("profile"):
            return 360 + 10 * max(0, n - 8)
        return WIN + 120 + n * BANDS[wk.split("@")[0]] / max(500.0, PREFILL["tok_s"])


    RPC_KEYS = ",".join(N.WATCHED) + ",NCCL_P2P_DISABLE,P2P,DRAFT_HEAD"


    def arm_body(arm, ref, store):
        """Every arm, the same items in the same order at its stream count; every item gated by due() on the boot's deadline."""
        def body(ss):
            s = ss[0]
            p = s.port
            n = arm["streams"]
            o = {"arm": arm["key"], "items_skipped": [], "streams": n}
            CUR["item"] = "facts"
            o = boot_facts(s, dict(o, census=kv_census(s), probes=L.chat_probes(p), sampled=sampled_check(s)), s.name)
            LIVE["out"] = o
            CUR["item"] = "readback"
            _live(o, "env_readback_raw", rpc(p, "nb4s_env", RPC_KEYS))
            _live(o, "env_readback", N.parse_env_readback(o["env_readback_raw"], arm["expect"]))
            _live(o, "proc_environ", N.parse_proc_environ(proc_environ(), arm["expect"]))
            _live(o, "config_readback", N.config_readback(arm, s.log() or "", CUR["kv_bytes"]))
            _live(o, "q_readback", q_readback(s, arm))
            _live(o, "free_after_ready_min", free_min(p))
            _live(o, "capacity_after_ready", N.capacity_facts(dict(o, pin=CUR["kv_bytes"])))
            if short_gate_failed(o):
                o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s),
                         quality_gate="FAIL_SAMPLED" if (o["sampled"] or {}).get("ok") is not True else "FAIL_PROBES")
                return o
            if o["env_readback"].get("definitive"):
                o.update(status="NOT_ENGAGED", early_stop="the workers do not hold the planned env: " + str(o["env_readback"].get("why")))
                return o                          # definitive: the arm's change is not in effect, nothing to measure
            if o["config_readback"].get("definitive"):
                o.update(status="NOT_ENGAGED", early_stop="the server log contradicts the plan: " + str(o["config_readback"].get("why")))
                return o
            if (o["q_readback"] or {}).get("label") == "NOT_INTEGRATED":
                o.update(status="NOT_INTEGRATED", not_integrated=True, early_stop="nb4_ppack init lines: " + json.dumps(o["q_readback"])[:300])
                return o
            if due(300):                          # every arm (the 40k probe prompt is variant 7 of the 40k window: same cache state)
                CUR["item"] = "prefill_probe"
                _live(o, "prefill", prefill_probe(p))
            if not due(240):
                o["items_skipped"].append("oracle")
                o["status"] = "SKIPPED_BOOT_BUDGET_ORACLE"
                return o
            CUR["item"] = "oracle"
            _live(o, "oracle", oracle(p, key=store, refs=(ref,)))
            o["oracle_vs"] = (o["oracle"].get("vs_" + ref) or {}).get("status")
            o["quality_gate"] = quality_gate(o, ref)
            if o["quality_gate"] != "PASS":
                o["load_tests_skipped"] = f"oracle gate vs {ref}: {o['quality_gate']}"
                o["status"] = "NO_GO_QUALITY" if o["quality_gate"].startswith("FAIL") else "INCONCLUSIVE_QUALITY"
                o["census_end"] = kv_census(s)
                return o
            _live(o, "q_readback", q_readback(s, arm))         # the counters moved by now (probes + sampled + oracle decode)
            if (o["q_readback"] or {}).get("label") not in ("ENGAGED", "ENGAGED_FALLBACK"):   # NB4S: a FALLBACK is not final here
                o.update(status=o["q_readback"]["label"], not_integrated=o["q_readback"]["label"] == "NOT_INTEGRATED",
                         early_stop="nb4_ppack read-back: " + json.dumps(o["q_readback"])[:300])
                return o
            if n > len(PROMPTS["8k"]) or n > len(PROMPTS["40k"]):
                o["prompts"] = ensure_prompts(n)
            if due(150):
                CUR["item"] = "warmup"
                _live(o, "warmup", warmup(p, n))
            else:
                o["items_skipped"].append("warmup")
            cap = o["capacity_after_ready"]
            pool = cap["num_gpu_blocks"][0] if len(cap.get("num_gpu_blocks") or []) == 1 else None
            o["steady"] = {}                      # RC2's _windows() gate reads "steady": garbage under load is NO_GO here too
            for wk in arm["windows"]:
                need = win_need(wk, n)
                if not due(need):
                    o["steady"][wk] = {"status": "SKIPPED_BOOT_BUDGET", "need_sec": round(need), "boot_left_sec": round(boot_left())}
                    o["items_skipped"].append(wk)
                    continue
                if not s.alive():
                    o["steady"][wk] = {"status": "SKIPPED_SERVER_DEAD"}
                    continue
                CUR["item"] = wk
                band = wk.split("@")[0].replace("profile_", "")
                gen_est = int((WIN + n * BANDS[band] / max(500.0, PREFILL["tok_s"])) * 300.0 / n)   # ~300 tok/s aggregate (NB4S QD)
                cp = N.capacity_predict(pool, cap.get("F_blocks"), n, BANDS[band], gen_est)     # recorded, never gating
                try:
                    r = profile_window(p, arm) if wk.startswith("profile") else hist_window(p, s, band, n)
                except Exception as e:
                    r = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1200:]}
                r["capacity_predict"] = cp
                o["steady"][wk] = r
                _live(o, "steady", o["steady"])
            CUR["item"] = "end"
            o["census_end"] = kv_census(s)
            o["workers_end"] = rpc_cache_info(p)
            o["free_end_min"] = free_min(p)
            o["q_readback_end"] = q_readback(s, arm)
            o["ppq_stats_last"] = [e for e in P.parse_log(s.log() or "") if e.get("event") == "stats"][-8:]
            o["jit_after_warmup"] = jit_kernels(s.log())[-12:]
            o["config_readback_end"] = N.config_readback(arm, s.log() or "", CUR["kv_bytes"])
            return o
        return body


    def arm_watched(body, cap_min, name):
        """NB4S's watched(): the cap fixed at body START = min(cap, left - BODY_RESERVE_SEC); at cap + 300 s the server is killed."""
        def b(ss):
            cap = N.body_cap_sec(cap_min, left())
            BOOT.update(deadline=time.time() + cap, fired=None)

            def fire():
                BOOT["fired"] = name
                for x in ss:
                    try:
                        x.kill()
                    except Exception:
                        pass
            tm = threading.Timer(cap + 300, fire)
            tm.daemon = True
            tm.start()
            try:
                o = body(ss)
                LIVE["out"] = o if isinstance(o, dict) else None
            finally:
                tm.cancel()
                BOOT["deadline"] = None
            o["body_cap_sec"] = round(cap)
            if BOOT["fired"]:
                o["boot_watchdog"] = f"FIRED after {cap + 300:.0f} s: server killed"
            return o
        return b


    def arm_boot(arm, body):
        """One arm: the boot gate, one Merlin launch (its build knobs set for this launch name), the /dev/shm cleanup; NB4S's
        ONE OOM policy (launch OOM in the launch's full log, no device fault -> one retry 0.25 GiB lower on a fresh port)."""
        kvb, port, tries, o = PIN["kv_bytes"], arm["port"], [], {}
        ref = "rc2_fp8" if arm["key"] == "Q" else "q"
        while True:
            lname = arm["name"] if not tries else f"{arm['name']}_t{len(tries)}"
            NB1_EXTRA[lname], NB1_ENV[lname], NB1_KV[lname] = list(arm["extra"]), dict(arm["env"]), kvb
            NB4T_BUILD_ENV[lname] = dict(arm["build_env"])
            ARM_OF[lname] = arm
            CUR["kv_bytes"] = kvb
            BOOT["tag"] = lname
            gate = boot_gate()
            if gate["ok"] is not True:
                return {"status": "GPU_NOT_EMPTY", "boot_gate": gate, "tries": tries, "launch_name": lname, "pin": kvb}
            shm0 = shm_names()
            mon = LoadMonitor(lname)
            mon.start()
            srv, CUR["item"] = None, "launch"
            try:
                srv = merlin_fn_server(lname, "fp8", port, kvb)
                o = with_servers([srv], NB4T_READY_SEC, arm_watched(body, arm["cap_min"], lname))
            finally:
                mon.stop_ev.set()
            try:
                full = srv.log() or ""
            except Exception:
                full = ""
            lines = full.splitlines()
            o.update(kv_bytes=kvb, pin=kvb, launch_name=lname, port=port, boot_gate=gate, shm_cleanup=shm_cleanup(shm0),
                     ready_sec=getattr(srv, "ready_sec", None), load_monitor_last=mon.last, oom_in_log=bool(OOM_ERR.search(full)),
                     xid_in_log=bool(XID.search(full)), oom_lines=[l[:240] for l in lines if OOM_ERR.search(l)][:6],
                     xid_lines=[l[:240] for l in lines if XID.search(l)][:6], died_in=CUR["item"])
            if o.get("q_readback") is None and srv is not None:
                o["q_readback"] = q_readback(srv, arm)        # a launch that failed: the init lines (if any) are still in the log
            if srv is not None:
                o["q_readback_final"] = q_readback(srv, arm)  # the whole log after the server stopped (F1's final stats line)
                o.setdefault("config_readback", N.config_readback(arm, full, kvb))
            oom = o.get("status") == "LAUNCH_FAILED" and o["oom_in_log"]
            tries.append({"launch_name": lname, "port": port, "kv_bytes": kvb, "status": o.get("status"), "oom": oom,
                          "device_fault": o["xid_in_log"], "ready_sec": o["ready_sec"]})
            if not oom or o["xid_in_log"] or len(tries) > 1 or left() < NB4T_READY_SEC + arm["cap_min"] * 60:
                break
            kvb -= OOM_STEP
            port += 1
            log("boot OOM:", lname, "-> one retry with KV_BYTES", kvb, "port", port)
        o["tries"] = tries
        if o.get("status") == "SERVER_DIED":
            o["death"] = ("BOOT_CAP_KILL" if o.get("boot_watchdog") else
                          f"BODY_OOM_{o.get('died_in')}" if o["oom_in_log"] and not o["xid_in_log"] else
                          f"DEVICE_FAULT_{o.get('died_in')}" if o["xid_in_log"] else f"SERVER_DIED_IN_{o.get('died_in')}")
            o["status"] = o["death"]
        o["capacity"] = N.capacity_facts(o)
        return gated(o, need=(ref,), expected_rows=4)


    def analyze_arm(o, key):
        """After the boot (GPUs idle): nb4t_lib.graph_stalls + nb4_timeline.analyze on the arm's profiled traces."""
        pk = ARMS[key]["windows"][2]
        d = ((o.get("steady") or {}).get(pk) or {}).get("trace_dir")
        if not d or not list(Path(d).glob("*.pt.trace.json*")):
            o["stall"] = {"error": "NO_TRACE_FILE"}
            o["timeline"] = {"error": "NO_TRACE_FILE"}
            return o
        t = time.time()
        try:
            st = N.graph_stalls(d)
            (OUT / f"stall_{key.lower()}.json").write_text(json.dumps(st, indent=1, default=str))
            o["stall"] = {k: v for k, v in st.items() if k != "by_rank"}
            o["stall"]["by_rank"] = {r: {x: y for x, y in v.items() if x != "rows"} for r, v in (st.get("by_rank") or {}).items()}
        except Exception as e:
            o["stall"] = {"error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1200:]}
        try:
            rep = TL.strip_private(TL.analyze(d))
            (OUT / f"timeline_{key.lower()}.json").write_text(json.dumps(rep, indent=1, default=str))
            o["timeline"] = N.summarize_timeline(rep)
            o["stage_compute"] = N.stage_compute(o["timeline"])
            try:
                (OUT / f"timeline_{key.lower()}.md").write_text(TL.format_report(rep))
            except Exception as e:
                o["timeline"]["format_report_error"] = repr(e)[:200]
        except Exception as e:
            o["timeline"] = {"error": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1200:]}
        o["parity_record"] = N.parity_of(o)
        o["analysis_sec"] = round(time.time() - t, 1)
        return o


    def run_arm(key):
        arm = ARMS[key]
        o = arm_boot(arm, arm_body(arm, "rc2_fp8" if key == "Q" else "q", "q" if key == "Q" else None))
        LIVE["out"] = o
        try:
            o["arm"], o["planned"] = key, NB4T["arms"][key]
            o["free_mib_per_card"] = free_per_card(o, o.get("launch_name"))
            if left() > 120:
                analyze_arm(o, key)
        except Exception as e:
            keep_after_error(o, e)
        return o


    def _arm_rec(key):
        r = (R["sections"].get(key) or {}).get(ARMS[key]["name"])
        return r if isinstance(r, dict) else None


    def arm_clean(o):
        return isinstance(o, dict) and o.get("status") == "DONE" and o.get("correctness") == "CLEAN" and o.get("quality_gate") == "PASS"


    def skip_arm(key, why, status="SKIPPED_DEPENDENCY", **kw):
        R["sections"].setdefault(key, {})[ARMS[key]["name"]] = dict({"status": status, "why": why}, **kw)
        save()


    def finalize():
        """Always (finally): analysis of any arm with traces but no analysis yet, then the verdict + nb4t_verdict.json."""
        try:
            recs = {}
            for k in ARMS:
                r = _arm_rec(k)
                if r is not None and "stall" not in r and ((r.get("steady") or {}).get(ARMS[k]["windows"][2]) or {}).get("trace_dir"):
                    try:
                        analyze_arm(r, k)
                    except Exception as e:
                        r["stall"] = {"error": repr(e)[:300]}
                recs[k] = r
            v = N.verdict({k: r for k, r in recs.items() if r is not None}, ARMS)
        except Exception as e:
            v = {"verdict": "ERROR_IN_VERDICT", "why": f"{type(e).__name__}: {e}"[:400], "tb": traceback.format_exc()[-1500:]}
        v["arms_status"] = {k: (_arm_rec(k) or {}).get("status") for k in ARMS}
        v["patch"] = NB4T.get("patch")
        v["minutes"] = round((time.time() - T_START) / 60, 1)
        NB4T["verdict"] = v
        try:
            (OUT / "nb4t_verdict.json").write_text(json.dumps(v, indent=1, default=str))
        except Exception:
            pass
        save()
        log("NB4T VERDICT", v.get("verdict"), "|", str(v.get("why"))[:400])
        return v


    def _patch_rec(pseq):
        return {"state": pseq.state, "log": pseq.log, "error": pseq.error, "apply": pseq.apply_rec, "revert": pseq.revert_rec}


    def nb4t_main(runner=None, pseq=None):
        """install -> #60490 + extension -> sha gate -> nb4_ppack patch -> Q -> I8 -> P26 -> P27 -> C12 (each later arm needs a
        clean, engaged Q and its full minimum of time; P27 needs P26 without an OOM). The patch is applied before Q, kept for
        every arm, reverted at the end (pre-shas verified); the verdict is written in the finally block whatever happens."""
        runner = runner or run_arm
        pseq = pseq or PSEQ
        try:
            m = R["sections"]["A0"]["merge"]
            if not m.get("ok"):
                NB4T["stop"] = "MERGE_FAILED: no boot started"
                for k in ARMS:
                    skip_arm(k, NB4T["stop"])
                return
            step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
            pt = step("A0", "patch_60490", 3, apply_60490)
            if pt.get("status") != "DONE" or not pt.get("ok"):
                for k in ARMS:
                    skip_arm(k, f"#60490 patch / worker extension not verified: {pt.get('error') or pt.get('status')}")
                return
            if "rc2_fp8" not in ROWS and not FAKE:
                for k in ARMS:
                    skip_arm(k, "RC2's recorded FP8 oracle rows do not match this notebook's prompts (no gate reference)")
                return
            ok, why = sha_gate(min(1500, left() - 900))
            R["setup"]["sha_gate"] = {"ok": ok, "why": why}
            if not ok:
                for k in ARMS:
                    skip_arm(k, "sha gate: " + why, "SKIPPED_SHA_GATE")
                return
            ok_p, why_p = pseq.before_arm(ARMS["Q"])
            NB4T["patch"] = _patch_rec(pseq)
            if not ok_p:
                skip_arm("Q", why_p, "NOT_INTEGRATED", not_integrated=True)
                for k in N.ORDER[1:]:
                    skip_arm(k, "no base: Q " + why_p)
                return
            q = step("Q", ARMS["Q"]["name"], ARMS["Q"]["est_min"], lambda: runner("Q"))
            bq = N.base_check(q)
            if not arm_clean(q) or not bq[0]:
                for k in N.ORDER[1:]:
                    skip_arm(k, f"Q not a clean base (status {q.get('status')}, correctness {q.get('correctness')}, gate "
                                f"{q.get('quality_gate')}, base {bq[1]}: {bq[2]})")
                return
            PIN.update(kv_bytes=q.get("pin") or PIN["kv_bytes"], source="Q's final pin")
            for k in N.ORDER[1:]:
                a = ARMS[k]
                if k == "P27":
                    run_ok, why_d = N.p27_dependency(_arm_rec("P26"))
                    if not run_ok:
                        skip_arm(k, why_d, "SKIPPED_P26_OOM")
                        continue
                if not N.arm_fits(left(), a):
                    skip_arm(k, f"left {left() / 60:.1f} min < its full minimum {a['est_min']} + {N.START_RESERVE_SEC // 60} min "
                                "(priority Q > I8 > P26 > P27 > C12)", "SKIPPED_BUDGET")
                    continue
                ok_p, why_p = pseq.before_arm(a)
                NB4T["patch"] = _patch_rec(pseq)
                if not ok_p:
                    skip_arm(k, why_p, "NOT_INTEGRATED", not_integrated=True)
                    continue
                step(k, a["name"], a["est_min"], lambda k=k: runner(k))
        finally:
            try:
                pseq.finish()
            except Exception as e:
                pseq.error = f"finish: {type(e).__name__}: {e}"[:300]
            NB4T["patch"] = _patch_rec(pseq)
            finalize()


    def _gpu_util():
        while not _gpu_util_stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=20)
                if r.returncode == 0:
                    with open(OUT / "nb4t_gpu_util.jsonl", "a") as fh:
                        fh.write(json.dumps({"t": round(time.time() - T_START, 1), "gpus": r.stdout.strip().splitlines()}) + "\n")
            except Exception:
                pass
            _gpu_util_stop.wait(5)


    _gpu_util_stop = threading.Event()
    if not RERUN and not os.environ.get("RC2_UNIT"):
        if NVSMI_OK:
            threading.Thread(target=_gpu_util, daemon=True).start()
        nb4t_main()
        _gpu_util_stop.set()
        R["host_phase"] = HOST_PHASE
        R["prefill"] = PREFILL
        R["minutes"] = round((time.time() - T_START) / 60, 1)
        save()
    save()


if not RERUN:
    # ==================================================================== CELL 8 — report + the identity fallback submission (E18 R2 format)
    if not RERUN and not os.environ.get("RC2_UNIT"):
        write_fallback_submission()
        def _arm_brief(k):
            r = _arm_rec(k) or {}
            w = r.get("steady") or {}
            return {"status": r.get("status"), "why": r.get("why"), "correctness": r.get("correctness"), "quality_gate": r.get("quality_gate"),
                    "oracle_vs": r.get("oracle_vs"), "pin": r.get("pin"), "tries": r.get("tries"), "ready_sec": r.get("ready_sec"),
                    "env_readback": (r.get("env_readback") or {}).get("why"), "config_readback": (r.get("config_readback") or {}).get("why"),
                    "config_checks": (r.get("config_readback") or {}).get("checks"), "q_readback_final": (r.get("q_readback_final") or {}).get("label"),
                    "capacity": r.get("capacity"), "stage_compute": r.get("stage_compute"), "parity": r.get("parity_record"),
                    "windows": {k2: N.window_brief(v) for k2, v in w.items() if isinstance(v, dict) and not k2.startswith("profile")},
                    "stall_stage0": {x: ((r.get("stall") or {}).get("stage0") or {}).get(x) for x in ("stall_ms_median", "share_below_1ms")},
                    "stall_stage1": {x: ((r.get("stall") or {}).get("stage1") or {}).get(x) for x in ("stall_ms_median", "sendrecv_ms_per_tick_mean")},
                    "timeline": {x: (r.get("timeline") or {}).get(x) for x in ("tick_ms_mean", "stages", "parity_verdict", "error")},
                    "error": r.get("error")}
        R["summary"] = {k: _arm_brief(k) for k in ARMS}
        save()
        _v = NB4T.get("verdict") or {}
        print("NB4T THROUGHPUT WINNER:", _v.get("best_arm"), _v.get("best_verdict"), _v.get("best_gain_8k"), "|", _v.get("why"))
        print("C12 (measure only):", json.dumps(_v.get("c12"), default=str)[:1500])
        print(json.dumps(_v, default=str)[:6000])
        print(json.dumps(R["summary"], default=str)[:14000])
