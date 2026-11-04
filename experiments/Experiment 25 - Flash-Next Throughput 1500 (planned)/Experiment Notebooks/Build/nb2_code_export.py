# ==================================================================== CELL 1 — config, inputs, lab module
import os, sys, json, math, time, re, shutil, subprocess, hashlib, tarfile, threading, traceback
from pathlib import Path
T_START = time.time()
MODE = "FULL"
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes"}
FAKE = os.environ.get("E23_FAKE_SERVER")                          # local CPU harness only
NB2_DEFAULT_PROFILE = "FULL"                         # baked per notebook file (FULL / SMOKE / SMOKE_SEED2 / SMOKE_CTRL)
NB2_PROFILE = (os.environ.get("NB2_PROFILE") or NB2_DEFAULT_PROFILE).strip().upper()
NB2_PROFILES = ("FULL", "SMOKE", "SMOKE_SEED2", "SMOKE_CTRL")
if NB2_PROFILE not in NB2_PROFILES and not RERUN:
    raise ValueError(f"unknown NB2_PROFILE {NB2_PROFILE}; choose one of {NB2_PROFILES}")
SMOKE = NB2_PROFILE in ("SMOKE", "SMOKE_SEED2", "SMOKE_CTRL")   # 35-min hard stop, one boot (B; CTRL in SMOKE_CTRL)
DIAG = NB2_PROFILE in ("SMOKE_SEED2", "SMOKE_CTRL")             # diagnostics of the SMOKE 8k@8 loop: verdict DIAG_DONE
BUDGET = int(os.environ.get("NB2_BUDGET_SEC") or (35 if SMOKE else 265) * 60)     # two cold Merlin boots + ladder + 90k/116k census + long rows
HARD_STOP = T_START + BUDGET
MAX_LEN = 131072                                                  # user 2026-10-29: thinking up to 120k
OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/nb2"))
OUT.mkdir(parents=True, exist_ok=True)
LOGS = OUT / "logs"
LOGS.mkdir(exist_ok=True)
TRACES = OUT / "traces"                                           # memory rule: the full text of every long generation
TRACES.mkdir(exist_ok=True)
TMP = Path(os.environ.get("E23_TMP", "/tmp"))
INP = Path(os.environ.get("E23_INPUT", "/kaggle/input"))
R = {"what": "E25 NB2 decode / MTP (Flash-Next on Merlin)", "profile": NB2_PROFILE, "budget_min": BUDGET // 60, "max_len": MAX_LEN,
     "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "setup": {}, "sections": {}, "errors": [], "order": []}
_SAVE_LOCK = threading.Lock()


def save():
    with _SAVE_LOCK:
        tmp = OUT / "nb2_results.json.tmp"
        tmp.write_text(json.dumps(R, indent=1, default=str))
        tmp.replace(OUT / "nb2_results.json")


def left():
    return HARD_STOP - time.time()


def log(*a):
    print(f"[{(time.time() - T_START) / 60:6.1f} min]", *a, flush=True)


COMP = INP / "competitions" / "arc-prize-2026-arc-agi-2"


def write_fallback_submission():
    tp = COMP / "arc-agi_test_challenges.json"
    if tp.is_file():
        tasks = json.load(open(tp))
        sub = {k: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in v["test"]] for k, v in tasks.items()}
        Path(os.environ.get("NB2_SUBMISSION", "/kaggle/working/submission.json")).write_text(json.dumps(sub))


if RERUN:
    write_fallback_submission()
    print("E25 NB2 speed notebook: competition rerun -> fallback submission.json only")
else:
    LAB_SRC = "\"\"\"E23 lab, rev 3: port-parameterized server lifecycle and measurement for the six-lever notebook (stdlib only).\n\nBase: the Exp 14 VMS harness (VServer / VGen: token ids from return_token_ids, [DONE] + finish_reason required, socket\nshutdown on stop, top-1/top-2 margins, garbage checks on every window). Adopted from the consolidated v3 notebook:\nsame-weight greedy oracle classified by the oracle's logprob margin at the first differing token, a steady decode window\nthat starts only after prefill settled, a GPU-release gate, fail-closed statuses, checkpoint identity, a runtime\n--help flag check, KV + GPU memory samples.\nrev 3 (validation agents 2026-10-29, measurement + robustness):\n  - port check by connect (a TIME_WAIT socket of the previous server is not \"busy\"), with a 90 s wait;\n  - kill(): SIGKILL the whole session after SIGTERM, even when the leader already exited;\n  - steady(): idle gate at entry, admission cap from the prefill to do (700 tok/s assumed), settle needs no preemption\n    and every running stream gaining tokens; in the window: stalled streams, a changing running/waiting set (KV-bound),\n    preemptions and prefills make it INCONCLUSIVE (numbers kept); garbage under load fails it; steps from the\n    vllm:iteration_tokens_total histogram count; replicas measured in one shared window (barrier);\n  - profiled_load(): settle as steady(), bounded profile (delay / max iterations), with_stack off, frontend ignored;\n  - trace_summary(): categories on the kernel base name (attention before GDN: \"fla\" is inside \"flashinfer\"), per-forward\n    graph / kernel launch counts inside the gpu_model_runner forward ranges (VLLM_CUSTOM_SCOPES_FOR_PROFILING=1) and the\n    CUPTI correlation fallback -> FULL / PIECEWISE / EAGER / UNKNOWN; startup cudagraph lines parsed explicitly;\n  - teacher_forced(): prompt logprobs of the last N positions of a long prompt (NVFP4 vs FP8 KV at 40k / 90k);\n  - logprobs keyed by token id (return_tokens_as_token_ids); stop() race closed; compare() flags garbage only when the\n    oracle itself is clean.\n\"\"\"\nimport gzip\nimport hashlib\nimport json\nimport os\nimport re\nimport signal\nimport socket\nimport statistics\nimport subprocess\nimport threading\nimport time\nimport urllib.request as U\nfrom pathlib import Path\n\nFAIL_MARGIN = 0.5          # first differing token: oracle top-1 minus top-2 logprob at or above this = real divergence\nLIVE = set()               # every running Server, for the hard-stop watchdog\n\n\n# ------------------------------------------------------------------ server lifecycle\ndef port_listening(port):\n    \"\"\"True when something accepts connections on 127.0.0.1:port (TIME_WAIT leftovers do not).\"\"\"\n    s = socket.socket()\n    s.settimeout(1.0)\n    try:\n        s.connect((\"127.0.0.1\", int(port)))\n        return True\n    except OSError:\n        return False\n    finally:\n        s.close()\n\n\nclass Server:\n    def __init__(self, name, argv, env, port, log_dir, gpus=\"0,1,2,3\"):\n        self.name, self.argv, self.env, self.port = name, [str(a) for a in argv], dict(env), int(port)\n        self.gpus = str(gpus)\n        self.log_path = Path(log_dir) / f\"server_{name}.log\"\n        self.proc, self.t0, self.ready_sec = None, None, None\n\n    @property\n    def url(self):\n        return f\"http://127.0.0.1:{self.port}\"\n\n    def start(self):\n        end = time.time() + 90\n        while port_listening(self.port):\n            if time.time() > end:\n                raise RuntimeError(f\"PORT_BUSY {self.port} (a listener is still there)\")\n            time.sleep(3)\n        self.log_path.parent.mkdir(parents=True, exist_ok=True)\n        self._fh = open(self.log_path, \"w\", encoding=\"utf-8\")\n        self.t0 = time.perf_counter()\n        self.proc = subprocess.Popen(self.argv, env=self.env, stdout=self._fh, stderr=subprocess.STDOUT,\n                                     start_new_session=True, cwd=\"/tmp\" if os.path.isdir(\"/tmp\") else None)\n        LIVE.add(self)\n        return self\n\n    def alive(self):\n        return self.proc is not None and self.proc.poll() is None\n\n    def log(self):\n        try:\n            return self.log_path.read_text(encoding=\"utf-8\", errors=\"replace\")\n        except Exception:\n            return \"\"\n\n    def rss_mb(self):\n        \"\"\"Host RAM of the server's whole session (engine + workers), MB; None off Linux.\"\"\"\n        try:\n            r = subprocess.run([\"ps\", \"-o\", \"rss=\", \"--sid\", str(self.proc.pid)], capture_output=True, text=True, timeout=20)\n            return round(sum(int(x) for x in r.stdout.split() if x.isdigit()) / 1024, 1)\n        except Exception:\n            return None\n\n    def wait_ready(self, timeout):\n        while time.perf_counter() - self.t0 < timeout:\n            if not self.alive():\n                return False, f\"exited rc={self.proc.returncode}\"\n            try:\n                U.urlopen(self.url + \"/health\", timeout=10).read()\n                self.ready_sec = round(time.perf_counter() - self.t0, 1)\n                return True, f\"ready in {self.ready_sec}s\"\n            except Exception:\n                time.sleep(3)\n        return False, \"startup timeout\"\n\n    def kill(self):\n        if self.proc is None:\n            return\n        pg = self.proc.pid\n        try:\n            os.killpg(pg, signal.SIGTERM) if hasattr(os, \"killpg\") else self.proc.terminate()\n        except Exception:\n            pass\n        try:\n            self.proc.wait(timeout=40)\n        except Exception:\n            pass\n        # rev 3: the leader may exit while EngineCore / workers of the session still hold GPUs: always SIGKILL the group\n        try:\n            os.killpg(pg, signal.SIGKILL) if hasattr(os, \"killpg\") else self.proc.kill()\n        except Exception:\n            pass\n        try:\n            self.proc.wait(timeout=30)\n        except Exception:\n            pass\n        try:\n            self._fh.close()\n        except Exception:\n            pass\n        LIVE.discard(self)\n\n    def facts(self):\n        \"\"\"What vLLM logged at startup: KV pool, max concurrency, selected kernels, graph mode, backend, KV dtype.\"\"\"\n        t = self.log()\n        kv = [int(x.replace(\",\", \"\")) for x in re.findall(r\"GPU KV cache size: ([\\d,]+) tokens\", t)]\n        mc = re.findall(r\"Maximum concurrency for [\\d,]+ tokens per request: ([\\d.]+)x\", t)\n        kern = sorted(set(re.findall(r\"Selected (\\w+Kernel) for\", t)))\n        errs = [l[:300] for l in t.splitlines() if re.search(r\"Traceback|CUDA error|illegal memory|out of memory|\"\n                                                               r\"RuntimeError|AssertionError\", l)][:12]\n        pick = lambda pat, n=8: [l.strip()[:240] for l in t.splitlines() if re.search(pat, l, re.I)][:n]\n        # rev 3 (measurement H5): the decisive startup evidence for the graph mode actually used with spec decode\n        downgrade = pick(r\"not supported with spec-decode.*setting cudagraph_mode|setting cudagraph_mode\\s*=|\"\n                         r\"[Oo]verriding cudagraph_mode to\", 4)       # rev 3e: + the 0.32 wording\n        capture = sorted(set(re.findall(r\"Capturing CUDA graphs \\(([^)]+)\\)\", t)))\n        configured = sorted(set(re.findall(r\"CUDAGraphMode\\.(\\w+)\", t)))\n        return {\"kv_tokens\": min(kv) if kv else None, \"kv_tokens_all\": kv, \"max_concurrency\": float(mc[-1]) if mc else None,\n                \"kernels\": kern, \"error_lines\": errs, \"ready_sec\": self.ready_sec,\n                \"graph_capture_modes\": capture, \"graph_downgrade_lines\": downgrade, \"cudagraph_modes_logged\": configured,\n                \"backend_lines\": pick(r\"attention backend|Using \\w+ backend|backend=|AttentionBackend\"),\n                \"kv_dtype_lines\": pick(r\"kv_cache_dtype|kv cache dtype|nvfp4\"),\n                \"host_embed_lines\": pick(r\"host[- _]?(resident|embed)|HOST_EMBED|embed.*(cpu|host)\", 6)}\n\n\ndef gpu_pids(gpus=None):\n    \"\"\"Compute PIDs on all GPUs, or only on the GPU indices in `gpus` (a bench on another GPU must not fail the release\n    gate of a server that used GPU 0 only). None when nvidia-smi is unavailable.\"\"\"\n    try:\n        a = subprocess.run([\"nvidia-smi\", \"--query-compute-apps=pid,gpu_uuid\", \"--format=csv,noheader,nounits\"],\n                           capture_output=True, text=True, timeout=30)\n        g = subprocess.run([\"nvidia-smi\", \"--query-gpu=index,uuid\", \"--format=csv,noheader,nounits\"],\n                           capture_output=True, text=True, timeout=30)\n    except Exception:\n        return None\n    if a.returncode or g.returncode:\n        return None\n    idx = {u.strip(): int(i) for i, u in (l.split(\",\", 1) for l in g.stdout.strip().splitlines() if \",\" in l)}\n    want = None if gpus is None else {int(x) for x in str(gpus).split(\",\") if str(x).strip() != \"\"}\n    out = set()\n    for l in a.stdout.strip().splitlines():\n        if \",\" not in l:\n            continue\n        pid, uuid = (x.strip() for x in l.split(\",\", 1))\n        if pid.isdigit() and (want is None or idx.get(uuid) in want):\n            out.add(int(pid))\n    return sorted(out)\n\n\ndef gpu_mem():\n    try:\n        r = subprocess.run([\"nvidia-smi\", \"--query-gpu=index,memory.used,memory.total\", \"--format=csv,noheader,nounits\"],\n                           capture_output=True, text=True, timeout=30)\n        return [[int(v) for v in l.split(\",\")] for l in r.stdout.strip().splitlines() if l.strip()]\n    except Exception:\n        return None\n\n\ndef wait_gpu_release(timeout=180, fake=False, gpus=None, free_mib=1024):\n    \"\"\"True when the GPUs in `gpus` (all when None) hold no compute PID AND use < free_mib MiB each (rev 3: compute-app\n    PIDs are often invisible inside Kaggle containers; memory.used is the proven E18 gate); None without nvidia-smi.\"\"\"\n    if fake:\n        return True\n    want = None if gpus is None else {int(x) for x in str(gpus).split(\",\") if str(x).strip() != \"\"}\n    end = time.time() + timeout\n    while time.time() < end:\n        p, mem = gpu_pids(gpus), gpu_mem()\n        if p is None or mem is None:\n            return None\n        used = [m[1] for m in mem if want is None or m[0] in want]\n        if not p and all(u < free_mib for u in used):\n            return True\n        time.sleep(3)\n    return False\n\n\ndef identity(model_dir):\n    \"\"\"Checkpoint identity: sha256 of config / tokenizer / index files (an oracle is only compared with the same weights).\"\"\"\n    root = Path(model_dir)\n    files = [p for n in (\"config.json\", \"tokenizer.json\", \"tokenizer_config.json\", \"generation_config.json\")\n             for p in [root / n] if p.is_file()] + sorted(root.glob(\"*.safetensors.index.json\"))\n    h = hashlib.sha256()\n    for p in files:\n        h.update(p.name.encode())\n        h.update(hashlib.sha256(p.read_bytes()).digest())\n    return {\"dir\": str(root), \"files\": [p.name for p in files], \"sha256\": h.hexdigest()[:16]}\n\n\n_HELP = {}\n\n\ndef help_check(key, argv_prefix, env, flags, timeout=300):\n    \"\"\"The engine's own `serve --help=all` (run once per engine `key`, cached) must list every flag of THIS launch\n    (rev 3: every launch is checked, not only the first); returns the missing ones.\"\"\"\n    if key not in _HELP:\n        try:\n            r = subprocess.run(list(argv_prefix) + [\"--help=all\"], capture_output=True, text=True, timeout=timeout, env=env)\n            _HELP[key] = r.stdout + r.stderr\n        except Exception as e:\n            _HELP[key] = \"ERROR \" + repr(e)[:200]\n    txt = _HELP[key]\n    if len(txt) < 2000:\n        return {\"ran\": False, \"error\": \"help output too short or failed\", \"tail\": txt[-400:]}\n    want = sorted({f for f in flags if isinstance(f, str) and f.startswith(\"--\")})\n    return {\"ran\": True, \"missing\": [f for f in want if f not in txt]}\n\n\ndef start_watchdog(hard_stop_epoch, on_fire):\n    \"\"\"A real hard stop: at `hard_stop_epoch` every live Server is killed and on_fire() is called.\"\"\"\n    def run():\n        while time.time() < hard_stop_epoch:\n            time.sleep(5)\n        try:\n            on_fire()\n        finally:\n            for s in list(LIVE):\n                try:\n                    s.kill()\n                except Exception:\n                    pass\n    t = threading.Thread(target=run, daemon=True, name=\"e23-watchdog\")\n    t.start()\n    return t\n\n\ndef prefetch(paths, threads=16, chunk=16 << 20):\n    \"\"\"Read files once with many readers (page cache warm-up of a cold /kaggle/input mount); returns bytes and seconds.\"\"\"\n    files = [p for root in paths for p in sorted(Path(root).rglob(\"*\")) if p.is_file()]\n    t0, done, lock = time.time(), [0], threading.Lock()\n\n    def worker(fs):\n        for p in fs:\n            try:\n                with open(p, \"rb\") as f:\n                    while True:\n                        b = f.read(chunk)\n                        if not b:\n                            break\n                        with lock:\n                            done[0] += len(b)\n            except Exception:\n                pass\n    big = sorted(files, key=lambda p: -p.stat().st_size)\n    parts = [big[i::threads] for i in range(threads)]\n    ths = [threading.Thread(target=worker, args=(x,), daemon=True) for x in parts if x]\n    for t in ths:\n        t.start()\n    for t in ths:\n        t.join()\n    dt = time.time() - t0\n    return {\"files\": len(files), \"bytes\": done[0], \"sec\": round(dt, 1), \"MBps\": round(done[0] / 1e6 / max(dt, 1e-3), 1)}\n\n\n# ------------------------------------------------------------------ metrics\ndef prom(port):\n    txt = U.urlopen(f\"http://127.0.0.1:{port}/metrics\", timeout=15).read().decode(\"utf-8\", \"replace\")\n    out = {}\n    for line in txt.splitlines():\n        if not line or line.startswith(\"#\"):\n            continue\n        m = re.match(r\"^([A-Za-z_:][A-Za-z0-9_:]*)(\\{[^}]*\\})?\\s+(\\S+)\", line)\n        if not m:\n            continue\n        try:\n            v = float(m.group(3))\n        except ValueError:\n            continue\n        if v != v:\n            continue\n        out[m.group(1)] = out.get(m.group(1), 0.0) + v\n        out[m.group(1) + \"@max\"] = max(out.get(m.group(1) + \"@max\", v), v)\n    return out\n\n\nGEN, PROMPT, RUN, WAIT = (\"vllm:generation_tokens_total\", \"vllm:prompt_tokens_total\", \"vllm:num_requests_running\",\n                          \"vllm:num_requests_waiting\")\nPREEMPT, KVU, ITER = \"vllm:num_preemptions_total\", \"vllm:kv_cache_usage_perc\", \"vllm:iteration_tokens_total_count\"\nACC, DRF, DTOK = (\"vllm:spec_decode_num_accepted_tokens_total\", \"vllm:spec_decode_num_drafts_total\",\n                  \"vllm:spec_decode_num_draft_tokens_total\")\n\n\n# ------------------------------------------------------------------ one streaming request (Exp 14 VGen, port-aware)\nclass Stream(threading.Thread):\n    def __init__(self, port, ids, max_new, temperature=1.0, seed=None, logprobs=0, ignore_eos=True, model=\"q27\"):\n        super().__init__(daemon=True)\n        self.port = int(port)\n        self.body = {\"model\": model, \"prompt\": list(ids), \"max_tokens\": int(max_new), \"temperature\": temperature,\n                     \"top_p\": 0.95 if temperature else 1.0, \"stream\": True, \"ignore_eos\": ignore_eos, \"return_token_ids\": True}\n        if temperature:\n            self.body[\"top_k\"] = 20\n        if seed is not None:\n            self.body[\"seed\"] = int(seed)\n        if logprobs:\n            self.body[\"logprobs\"] = int(logprobs)\n            self.body[\"return_tokens_as_token_ids\"] = True      # rev 3: logprob keys by id (no string collisions)\n        self.text, self.tokens, self.margins = [], [], []\n        self.err, self.done_seen, self.finish_reason = None, False, None\n        self.t_start = self.t_first = self.t_end = None\n        self._resp, self._halt = None, False\n\n    def run(self):\n        self.t_start = time.perf_counter()\n        try:\n            req = U.Request(f\"http://127.0.0.1:{self.port}/v1/completions\", data=json.dumps(self.body).encode(),\n                            method=\"POST\", headers={\"Content-Type\": \"application/json\"})\n            self._resp = U.urlopen(req, timeout=7200)\n            for raw in self._resp:\n                if self._halt:\n                    break\n                line = raw.strip()\n                if not line.startswith(b\"data:\"):\n                    continue\n                payload = line[5:].strip()\n                if payload == b\"[DONE]\":\n                    self.done_seen = True\n                    break\n                d = json.loads(payload)\n                if \"error\" in d or d.get(\"object\") == \"error\":\n                    self.err = (\"stream error: \" + json.dumps(d.get(\"error\") or d.get(\"message\") or d))[:300]\n                    break\n                for c in d.get(\"choices\") or []:\n                    self.text.append(c.get(\"text\") or \"\")\n                    ids_ = c.get(\"token_ids\")\n                    lp = c.get(\"logprobs\") or {}\n                    if ids_ is None:\n                        if c.get(\"text\") or lp.get(\"tokens\"):\n                            self.err = \"MISSING_STREAM_TOKEN_IDS\"\n                    elif any(not isinstance(v, int) or isinstance(v, bool) for v in ids_):\n                        self.err = \"INVALID_STREAM_TOKEN_IDS\"\n                    else:\n                        if ids_ and self.t_first is None:\n                            self.t_first = time.perf_counter()\n                        self.tokens.extend(ids_)\n                    for top in lp.get(\"top_logprobs\") or []:\n                        vals = sorted((v for v in (top or {}).values() if isinstance(v, (int, float))), reverse=True)\n                        self.margins.append(round(vals[0] - vals[1], 4) if len(vals) >= 2 else None)\n                        if getattr(self, \"tops\", None) is not None:   # E25 NB2 DIAG: top-logprob ids per token\n                            self.tops.append(sorted(([k, round(v, 4)] for k, v in (top or {}).items()\n                                                     if isinstance(v, (int, float))), key=lambda kv: -kv[1]))\n                    if c.get(\"finish_reason\") is not None:\n                        self.finish_reason = c[\"finish_reason\"]\n                if self.err:\n                    break\n        except Exception as e:\n            if not self._halt:\n                self.err = repr(e)[:300]\n        finally:                         # rev 3: a stop() before the response existed still closes the socket here\n            try:\n                if self._resp is not None:\n                    self._resp.close()\n            except Exception:\n                pass\n        self.t_end = time.perf_counter()\n        if not self._halt and not self.err and not (self.done_seen and self.finish_reason):\n            self.err = f\"INCOMPLETE_SSE done={self.done_seen} finish={self.finish_reason}\"\n\n    def stop(self):\n        self._halt = True\n        try:\n            self._resp.fp.raw._sock.shutdown(socket.SHUT_RDWR)\n        except Exception:\n            pass\n        try:\n            if self._resp is not None:\n                self._resp.close()\n        except Exception:\n            pass\n\n    def row(self):\n        return {\"text\": \"\".join(self.text), \"tokens\": list(self.tokens), \"margins\": list(self.margins), \"error\": self.err,\n                \"done\": self.done_seen, \"finish_reason\": self.finish_reason}\n\n\n# ------------------------------------------------------------------ correctness\ndef greedy(port, prompts, n_tok=128, logprobs=2, timeout=900, model=\"q27\"):\n    \"\"\"Greedy rows for the same-weight oracle (one request at a time: batch-size effects stay out of the comparison).\"\"\"\n    rows = []\n    for ids in prompts:\n        g = Stream(port, ids, n_tok, temperature=0.0, logprobs=logprobs, ignore_eos=True, model=model)\n        g.start()\n        g.join(timeout)\n        if g.is_alive():\n            g.stop()\n            g.join(10)\n            r = g.row()\n            r[\"error\"] = r[\"error\"] or \"GREEDY_TIMEOUT\"\n        else:\n            r = g.row()\n        rows.append(r)\n    return rows\n\n\ndef degenerate(text):\n    \"\"\"Objective garbage checks: empty, a run of one char, '!!!!' (the E3 / E13 garbage), very low variety.\"\"\"\n    t = text or \"\"\n    if not t.strip():\n        return \"EMPTY\"\n    if re.search(r\"(\\S)\\1{200,}\", t, re.S):      # rev 3: grid rows / separators are not garbage\n        return \"REPEATED_CHAR\"\n    if t.count(\"!\") > 0.5 * max(1, len(t)):\n        return \"BANG_GARBAGE\"\n    words = re.findall(r\"[A-Za-z]{3,}\", t)            # rev 3c: words only; ARC grid dumps (digits) are not garbage\n    if len(words) > 60 and len(set(words)) < 0.08 * len(words):\n        return \"LOW_VARIETY\"\n    return None\n\n\ndef compare(ref, cand):\n    \"\"\"Same-weight oracle: at the first differing token the ORACLE's top-1 - top-2 logprob margin decides:\n    >= FAIL_MARGIN -> FAIL_HIGH_MARGIN, below -> NEAR_TIE; identical -> EXACT; missing data -> INCONCLUSIVE.\"\"\"\n    if not ref or not cand or len(ref) != len(cand):\n        return {\"status\": \"INCONCLUSIVE_PAIRS_MISSING\"}\n    per, worst = [], \"EXACT\"\n    rank = {\"EXACT\": 0, \"NEAR_TIE\": 1, \"INCONCLUSIVE\": 2, \"FAIL_HIGH_MARGIN\": 3}\n    for a, b in zip(ref, cand):\n        if a.get(\"error\") or b.get(\"error\") or not a.get(\"tokens\") or not b.get(\"tokens\"):\n            st, d, m = \"INCONCLUSIVE\", None, None\n        else:\n            x, y = a[\"tokens\"], b[\"tokens\"]\n            d = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)\n            if d is None:\n                st, m = (\"EXACT\" if len(x) == len(y) else \"INCONCLUSIVE\"), None\n            else:\n                mm = a.get(\"margins\") or []\n                m = mm[d] if d < len(mm) else None\n                st = \"INCONCLUSIVE\" if m is None else (\"FAIL_HIGH_MARGIN\" if m >= FAIL_MARGIN else \"NEAR_TIE\")\n        g = degenerate(b.get(\"text\")) if (b.get(\"tokens\") and not b.get(\"error\")) else None\n        if g and degenerate(a.get(\"text\")) is None:           # rev 3c: a request error is not corruption\n            st = \"FAIL_HIGH_MARGIN\"\n        per.append({\"first_diff\": d, \"oracle_margin\": m, \"status\": st, \"garbage\": g})\n        worst = st if rank[st] > rank[worst] else worst\n    return {\"status\": worst, \"rows\": per}\n\n\ndef chat_probes(port, model=\"q27\", cases=((\"Reply with exactly one number: 7 * 6 =\", \"42\"),\n                                          (\"Reply with exactly one number: 2 + 2 =\", \"4\"),\n                                          (\"Reply with exactly one number: 9 - 3 =\", \"6\")), timeout=300):\n    \"\"\"Arithmetic through /v1/chat/completions with thinking off. vLLM 0.32 returns the answer in message.content and\n    the thinking in message.reasoning (NOT reasoning_content: the consolidated notebook's blocker).\"\"\"\n    out = []\n    for q, want in cases:\n        body = {\"model\": model, \"messages\": [{\"role\": \"user\", \"content\": q}], \"temperature\": 0, \"max_tokens\": 32,\n                \"chat_template_kwargs\": {\"enable_thinking\": False}}\n        try:\n            r = json.loads(U.urlopen(U.Request(f\"http://127.0.0.1:{port}/v1/chat/completions\", data=json.dumps(body).encode(),\n                                               headers={\"Content-Type\": \"application/json\"}), timeout=timeout).read())\n            msg = r[\"choices\"][0][\"message\"]\n            got = (msg.get(\"content\") or \"\").strip()\n            out.append({\"q\": q, \"want\": want, \"content\": got[:80], \"reasoning_chars\": len(msg.get(\"reasoning\") or \"\"),\n                        \"ok\": re.search(r\"(?<!\\d)\" + want + r\"(?!\\d)\", got) is not None})\n        except Exception as e:\n            out.append({\"q\": q, \"want\": want, \"error\": repr(e)[:200], \"ok\": False})\n    return {\"ok\": all(x[\"ok\"] for x in out), \"cases\": out}\n\n\ndef template_probe(port, messages, want_ids, model=\"q27\", timeout=120):\n    \"\"\"rev 3h (three GPT SMOKE audits): what the SERVER's chat path renders for `messages` (its own template and default\n    chat-template kwargs, no per-request kwargs) against the production xhigh render `want_ids`. MATCH / MISMATCH /\n    ERROR; the load requests themselves use pre-rendered ids, so this shows chat-path equivalence only.\"\"\"\n    body = {\"model\": model, \"messages\": messages, \"add_generation_prompt\": True}\n    try:\n        r = json.loads(U.urlopen(U.Request(f\"http://127.0.0.1:{port}/tokenize\", data=json.dumps(body).encode(),\n                                           headers={\"Content-Type\": \"application/json\"}), timeout=timeout).read())\n        got = list(r.get(\"tokens\") or [])\n    except Exception as e:\n        return {\"status\": \"ERROR\", \"error\": repr(e)[:200]}\n    want = list(want_ids)\n    first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None if len(got) == len(want) else min(len(got), len(want)))\n    return {\"status\": \"MATCH\" if got == want else \"MISMATCH\", \"server_tokens\": len(got), \"want_tokens\": len(want),\n            \"first_diff\": first}\n\n\ndef teacher_forced(port, ids, last=256, model=\"q27\", timeout=1800):\n    \"\"\"rev 3 (measurement M3): prompt logprobs of the last `last` positions of a long prompt (one prefill, 1 token out),\n    for a lossy-KV comparison (NVFP4 vs FP8 KV) at the context lengths that matter. Returns per-position\n    [logprob of the actual token, its rank, the engine's own rank-1 token id] or an error. rev 3h (three GPT SMOKE\n    audits): the rank-1 token id is kept, so top-1 agreement compares the two engines' predictions.\"\"\"\n    body = {\"model\": model, \"prompt\": list(ids), \"max_tokens\": 1, \"temperature\": 0, \"prompt_logprobs\": 1,\n            \"return_tokens_as_token_ids\": True}\n    try:\n        r = json.loads(U.urlopen(U.Request(f\"http://127.0.0.1:{port}/v1/completions\", data=json.dumps(body).encode(),\n                                           headers={\"Content-Type\": \"application/json\"}), timeout=timeout).read())\n        pl = r[\"choices\"][0].get(\"prompt_logprobs\") or []\n    except Exception as e:\n        return {\"error\": repr(e)[:300]}\n    out = []\n    for pos in range(max(1, len(ids) - last), len(ids)):\n        d = pl[pos] if pos < len(pl) else None\n        if not isinstance(d, dict):\n            out.append(None)\n            continue\n        e = d.get(str(ids[pos])) or d.get(f\"token_id:{ids[pos]}\")     # rev 3c: no fallback to another token\n        top1 = [_tok_id(k) for k, v in d.items() if isinstance(v, dict) and v.get(\"rank\") == 1]\n        out.append([e.get(\"logprob\"), e.get(\"rank\"), top1[0] if len(top1) == 1 else None] if isinstance(e, dict) else None)\n    return {\"positions\": out, \"n\": len(out)}\n\n\ndef _tok_id(k):\n    \"\"\"A prompt-logprob key ('123' or 'token_id:123') as an int id; anything else stays a tagged string.\"\"\"\n    m = re.fullmatch(r\"(?:token_id:)?(-?\\d+)\", str(k))\n    return int(m.group(1)) if m else \"str:\" + str(k)\n\n\ndef compare_tf(ref, cand):\n    \"\"\"Teacher-forced comparison: mean / max |delta logprob| of the actual tokens and rank-1 agreement. rev 3d (four\n    GPT audits): MEASURED only when EVERY expected position is valid on both sides (1 of 256 used to be MEASURED);\n    MEASURED is a measurement, not a PASS (the caller applies the A/A-derived tolerance).\"\"\"\n    if not ref or not cand or ref.get(\"error\") or cand.get(\"error\"):\n        return {\"status\": \"INCONCLUSIVE\"}\n    a_, b_ = ref.get(\"positions\") or [], cand.get(\"positions\") or []\n    if not a_ or len(a_) != len(b_):\n        return {\"status\": \"INCONCLUSIVE_INCOMPLETE_TF\", \"ref_n\": len(a_), \"cand_n\": len(b_)}\n    if any(not a or not b or a[0] is None or b[0] is None for a, b in zip(a_, b_)):\n        return {\"status\": \"INCONCLUSIVE_INCOMPLETE_TF\", \"valid\": sum(1 for a, b in zip(a_, b_) if a and b and a[0] is not None\n                                                                     and b[0] is not None), \"expected\": len(a_)}\n    pairs = list(zip(a_, b_))\n    if any(len(a) < 3 or len(b) < 3 or a[2] is None or b[2] is None for a, b in pairs):\n        return {\"status\": \"INCONCLUSIVE_TOP1_MISSING\", \"expected\": len(pairs),\n                \"with_top1\": sum(1 for a, b in pairs if len(a) >= 3 and len(b) >= 3 and a[2] is not None and b[2] is not None)}\n    d = sorted(abs(a[0] - b[0]) for a, b in pairs)\n    # rev 3h (three GPT SMOKE audits): top-1 agreement = both engines predict the same rank-1 token id. rev 3d-3g counted\n    # \"the forced token is rank 1 on both sides or on neither\", which reads 1.0 for ranks 2 vs 3.\n    r1 = sum(1 for a, b in pairs if a[2] == b[2]) / len(pairs)\n    both = sum(1 for a, b in pairs if (a[1] == 1) == (b[1] == 1)) / len(pairs)\n    p99 = d[min(len(d) - 1, int(round(0.99 * (len(d) - 1))))]          # rev 3e: robust to one outlier position\n    return {\"status\": \"MEASURED\", \"n\": len(pairs), \"mean_abs_dlogprob\": round(statistics.mean(d), 4),\n            \"p99_abs_dlogprob\": round(p99, 3), \"max_abs_dlogprob\": round(d[-1], 3), \"rank1_agreement\": round(r1, 4),\n            \"forced_token_rank1_both_or_neither\": round(both, 4)}\n\n\ndef tf_complete(rows):\n    \"\"\"rev 3e: a teacher-forced row set counts as a reference only when every position of every band is valid\n    (rev 3h: including the rank-1 token id).\"\"\"\n    return bool(rows) and all(isinstance(r, dict) and not r.get(\"error\") and r.get(\"positions\") and\n                              all(p and p[0] is not None and len(p) >= 3 and p[2] is not None for p in r[\"positions\"])\n                              for r in rows.values())\n\n\n# ------------------------------------------------------------------ steady decode\ndef steady(port, prompts, level, secs=60.0, admit=None, max_new=8192, temperature=1.0, seed=1234, model=\"q27\",\n           sample=2.0, prefill_tok_s=700.0, gpu_sample=True, barrier=None):\n    \"\"\"`level` distinct prompts, then a decode window that starts only once prefill has settled. Status:\n    STEADY / CAPACITY_LIMITED (fewer running than asked: KV) / INCONCLUSIVE_* / FAIL_*; numbers are kept on an\n    INCONCLUSIVE window so a KV census survives; never a made-up number. `barrier` (steady_multi) starts every replica's\n    window at the same moment.\"\"\"\n    plen = len(prompts[0]) if prompts else 0\n    admit = admit or max(420.0, 120.0 + level * plen / prefill_tok_s)\n    r = {\"level\": level, \"status\": \"INCONCLUSIVE_NOT_STARTED\", \"port\": port, \"prompt_tokens\": plen, \"max_new\": max_new,\n         \"admit_limit_sec\": round(admit)}\n    if not wait_idle(port, 60):\n        r[\"status\"] = \"INCONCLUSIVE_NOT_IDLE_AT_START\"\n        if barrier:\n            barrier.abort()\n        return r\n    try:\n        p0 = prom(port)\n    except Exception as e:\n        r.update(status=\"INCONCLUSIVE_METRICS_UNAVAILABLE\", error=repr(e)[:200])\n        if barrier:\n            barrier.abort()\n        return r\n    if GEN not in p0 or PROMPT not in p0:\n        r[\"status\"] = \"INCONCLUSIVE_COUNTERS_MISSING\"\n        if barrier:\n            barrier.abort()\n        return r\n    ss = [Stream(port, prompts[i % len(prompts)], max_new, temperature, seed + i, model=model) for i in range(level)]\n    for g in ss:\n        g.start()\n    try:\n        t_adm, best, settled, prev, samples, flat_since = time.perf_counter(), 0, 0, None, [], None\n        last_tok = [0] * len(ss)\n        while time.perf_counter() - t_adm < admit:\n            time.sleep(sample)\n            if all(not g.is_alive() for g in ss):          # rev 3: checked before the metrics read (a dead server)\n                r.update(status=\"INCONCLUSIVE_STREAMS_ENDED_IN_ADMISSION\", errors=[g.err for g in ss if g.err][:3])\n                if barrier:\n                    barrier.abort()                       # rev 3c: the other replicas must not wait for this one\n                return r\n            try:\n                m = prom(port)\n            except Exception:\n                continue\n            run, wait, pt, pre = int(m.get(RUN, 0)), int(m.get(WAIT, 0)), m.get(PROMPT), m.get(PREEMPT, 0)\n            now_tok = [len(g.tokens) for g in ss]\n            first = sum(g.t_first is not None for g in ss)\n            gaining = sum(1 for a, b in zip(last_tok, now_tok) if b > a)\n            last_tok = now_tok\n            best = max(best, run)\n            samples.append([round(time.perf_counter() - t_adm, 1), run, wait, first, gaining])\n            # rev 3: no preemption in the sample, every running request emitted and is gaining tokens\n            calm = run > 0 and prev is not None and pt == prev[0] and pre == prev[1] and first >= run and gaining >= run\n            if calm and run >= level and wait == 0:\n                settled += 1\n            elif calm and run < level and wait > 0:       # KV-bound: the rest wait; accept a plateau held 30 s\n                flat_since = flat_since or time.perf_counter()\n                settled = settled + 1 if time.perf_counter() - flat_since > 30 else 0\n            else:\n                settled, flat_since = 0, None\n            prev = (pt, pre)\n            if settled >= 3:\n                break\n        r[\"admission\"] = {\"sec\": round(time.perf_counter() - t_adm, 1), \"best_running\": best, \"tail\": samples[-6:],\n                          \"prefill_tok_s\": round((prom(port).get(PROMPT, 0) - p0.get(PROMPT, 0)) /\n                                                 max(1e-3, time.perf_counter() - t_adm), 1)}\n        if settled < 3:\n            r[\"status\"] = \"INCONCLUSIVE_NO_SETTLE\"\n            if barrier:\n                barrier.abort()\n            return r\n        if barrier:\n            try:\n                barrier.wait(timeout=max(60.0, admit))\n            except threading.BrokenBarrierError:\n                r[\"status\"] = \"INCONCLUSIVE_REPLICA_BARRIER\"\n                return r\n        m0, t0 = prom(port), time.perf_counter()\n        runs, waits, kvs, mids, gmem, stalls = [], [], [], [], [], [0] * len(ss)\n        last_tok, track, misses = [len(g.tokens) for g in ss], None, 0\n        while time.perf_counter() - t0 < secs:\n            time.sleep(min(sample, max(0.1, secs - (time.perf_counter() - t0))))\n            try:\n                m = prom(port)\n            except Exception:                              # rev 3c: one failed metrics read is not a crash\n                misses += 1\n                if misses >= 5:\n                    break\n                continue\n            mids.append(m)\n            runs.append(m.get(RUN, 0))\n            waits.append(m.get(WAIT, 0))\n            if KVU in m:\n                kvs.append(m[KVU])\n            now_tok = [len(g.tokens) for g in ss]\n            if track is None:                              # rev 3c: only streams decoding at window start can stall\n                track = {i for i in range(len(ss)) if now_tok[i] > last_tok[i]}\n            for i, g in enumerate(ss):\n                if i in track and g.is_alive():\n                    stalls[i] = stalls[i] + 1 if now_tok[i] <= last_tok[i] else 0\n            last_tok = now_tok\n            if any(x >= 3 for x in stalls):\n                break\n            if gpu_sample and len(mids) % 5 == 1:\n                g_ = gpu_mem()\n                if g_:\n                    gmem.append(g_)\n        m1, t1 = prom(port), time.perf_counter()\n        gen = m1.get(GEN, 0) - m0.get(GEN, 0)\n        dt = t1 - t0\n        acc, drf, dtk, it = (m1.get(k, 0) - m0.get(k, 0) for k in (ACC, DRF, DTOK, ITER))\n        rm = statistics.mean(runs) if runs else 0\n        steps = it if it else ((drf / rm) if (drf and rm) else ((gen / rm) if rm else 0))\n        r.update(tok_s=round(gen / dt, 1) if dt else None, gen_tokens=int(gen), window_sec=round(dt, 1),\n                 running_mean=round(rm, 2), running_min=min(runs or [0]), running_max=max(runs or [0]),\n                 waiting_min=min(waits or [0]), waiting_max=max(waits or [0]),\n                 tok_s_per_stream=round(gen / dt / rm, 2) if rm and dt else None,\n                 accept_per_draft=round(acc / drf, 3) if drf else None,\n                 mean_acceptance_length=round(1 + acc / drf, 3) if drf else None,\n                 accept_rate=round(acc / dtk, 3) if dtk else None,\n                 steps=int(steps), steps_source=\"iteration_histogram\" if it else \"estimate\",\n                 step_ms=round(dt * 1000 / steps, 1) if steps else None,\n                 tokens_per_step=round(gen / steps, 2) if steps else None,\n                 kv_usage_mean=round(statistics.mean(kvs), 3) if kvs else None, kv_usage_max=max(kvs) if kvs else None,\n                 gpu_mem_peak_mib=[max(row[i][1] for row in gmem if len(row) > i) for i in range(max(map(len, gmem)))] if gmem else None,\n                 preemptions=m1.get(PREEMPT, 0) - m0.get(PREEMPT, 0), t0=t0, t1=t1)\n        if misses >= 5:\n            r[\"status\"] = \"INCONCLUSIVE_METRICS_LOST\"\n        elif any(x >= 3 for x in stalls):\n            r[\"status\"] = \"INCONCLUSIVE_STALLED_STREAM\"\n        elif sum(g.is_alive() for g in ss) < min(level, best):\n            r[\"status\"] = \"INCONCLUSIVE_STREAMS_ENDED_IN_WINDOW\"\n        elif any(x.get(PROMPT) != m0.get(PROMPT) for x in mids + [m1]):\n            r[\"status\"] = \"INCONCLUSIVE_PREFILL_IN_WINDOW\"\n        elif r[\"preemptions\"] > 0:\n            r[\"status\"] = \"INCONCLUSIVE_PREEMPTION\"\n        elif gen <= 0:\n            r[\"status\"] = \"FAIL_NO_DECODE\"\n        elif min(runs or [0]) < level and (len(set(runs)) > 1 or len(set(waits)) > 1):\n            r[\"status\"] = \"INCONCLUSIVE_RUNNING_SET_CHANGED\"      # KV-bound window must hold one fixed set\n        elif min(runs or [0]) < 0.8 * max(runs or [1]):\n            r[\"status\"] = \"INCONCLUSIVE_RUNNING_UNSTABLE\"\n        else:\n            r[\"status\"] = (\"STEADY\" if min(runs or [0]) == max(runs or [0]) == level else\n                           \"CAPACITY_LIMITED\" if max(runs or [0]) < level else \"INCONCLUSIVE_RUNNING_ABOVE_LEVEL\")\n        return r\n    finally:\n        errs = sorted({g.err for g in ss if g.err})[:3]           # rev 3: snapshot before stop() sets _halt\n        for g in ss:\n            g.stop()\n        for g in ss:\n            g.join(10)\n        r[\"client_leak\"] = any(g.is_alive() for g in ss)\n        r[\"stream_errors\"] = errs\n        gb = [degenerate(\"\".join(g.text)[-4000:]) if g.tokens else None for g in ss]\n        r[\"garbage\"] = {\"streams\": len(ss), \"with_output\": sum(1 for g in ss if g.tokens),\n                        \"garbage\": [x for x in gb if x], \"sample_tail\": \"\".join(ss[0].text)[-200:] if ss and ss[0].tokens else \"\"}\n        if r[\"garbage\"][\"garbage\"] and r.get(\"status\") in (\"STEADY\", \"CAPACITY_LIMITED\"):\n            r[\"status\"] = \"FAIL_GARBAGE_UNDER_LOAD\"\n        r[\"idle_after\"] = wait_idle(port)\n\n\ndef steady_multi(jobs, secs=60.0, **kw):\n    \"\"\"Several servers at once (TP1 replicas): jobs = [(port, prompts, level)]; one shared window (barrier). The aggregate\n    counts only when every window is STEADY / CAPACITY_LIMITED and the windows overlap >= 80 % of the shortest.\"\"\"\n    out = [None] * len(jobs)\n    bar = threading.Barrier(len(jobs))\n\n    def one(i, j):\n        try:\n            out[i] = steady(j[0], j[1], j[2], secs=secs, barrier=bar, **kw)\n        except Exception as e:          # rev 3: a dead replica is recorded, not printed to stderr only\n            out[i] = {\"status\": \"ERROR_REPLICA\", \"error\": f\"{type(e).__name__}: {e}\"[:300], \"port\": j[0]}\n            try:\n                bar.abort()\n            except Exception:\n                pass\n    th = [threading.Thread(target=one, args=(i, j), daemon=True) for i, j in enumerate(jobs)]\n    for t in th:\n        t.start()\n    for t in th:\n        t.join()\n    good = all(o and o.get(\"status\") in (\"STEADY\", \"CAPACITY_LIMITED\") for o in out)\n    ov = (min(o[\"t1\"] for o in out) - max(o[\"t0\"] for o in out)) if good else 0\n    aligned = good and ov > 0.8 * min(o[\"window_sec\"] for o in out)\n    return {\"replicas\": out, \"aligned\": aligned, \"overlap_sec\": round(ov, 1),\n            \"tok_s\": round(sum(o[\"tok_s\"] for o in out), 1) if aligned else None,\n            \"running_mean\": round(sum(o[\"running_mean\"] for o in out), 2) if aligned else None,\n            \"status\": \"AGGREGATE_STEADY\" if aligned else \"INCONCLUSIVE_REPLICA_WINDOWS\"}\n\n\ndef wait_idle(port, wait=90.0):\n    end, hits = time.perf_counter() + wait, 0\n    while time.perf_counter() < end:\n        try:\n            m = prom(port)\n            hits = hits + 1 if m.get(RUN, 0) == 0 and m.get(WAIT, 0) == 0 else 0\n        except Exception:\n            hits = 0\n        if hits >= 3:\n            return True\n        time.sleep(2)\n    return False\n\n\n# ------------------------------------------------------------------ profiler (FULL-graph proof + per-category time)\n# rev 3: matched on the kernel BASE name, attention before GDN (\"fla\" is a substring of \"flashinfer\"), real 0.32 GDN names\nCATS = [(\"nccl\", (\"nccl\", \"allreduce\", \"all_reduce\", \"allgather\", \"all_gather\")),\n        (\"attention\", (\"flashinfer\", \"flash_fwd\", \"unified_attention\", \"paged\", \"attn\", \"attention\", \"batchdecode\", \"batchprefill\",\n                   \"reshape_and_cache\")),\n        (\"gdn_linear_attn\", (\"gdn\", \"delta_rule\", \"chunk_\", \"recurrent\", \"causal_conv\", \"l2norm\", \"solve_tril\", \"merge_16x16\",\n                             \"recompute_w_u\", \"kkt\", \"cumsum\", \"post_conv\", \"layer_norm_gated\", \"gated_delta\")),\n        (\"gemm_marlin\", (\"marlin\",)),\n        (\"gemm_other\", (\"gemm\", \"cutlass\", \"xmma\", \"cublas\", \"sm89_\", \"ampere_\", \"sm80_\")),\n        (\"sampling\", (\"sampl\", \"topk\", \"top_k\", \"argmax\", \"softmax\", \"rejection\"))]\n\n\ndef _cat(name):\n    # rev 3h: \"void (anonymous namespace)::gdn_decode_post_conv_mtp_kernel<...>(...)\" used to reduce to \"void \" -> other\n    s = re.sub(r\"\\(anonymous namespace\\)::\", \"\", str(name))\n    s = re.sub(r\"^\\s*void\\s+\", \"\", s)\n    base = s.split(\"<\")[0].split(\"(\")[0].lower()\n    return next((k for k, pats in CATS if any(p in base for p in pats)), \"other\")\n\n\nCOMPUTE_CATS = (\"nccl\", \"attention\", \"gdn_linear_attn\", \"gemm_marlin\", \"gemm_other\")\n# rev 3h: vLLM 0.32 marks each model execution as execute_context_N(N)_generation_N(N) (no gpu_model_runner range)\nSCOPE_RE = re.compile(r\"^execute_context_\\d+\\(\\d+\\)_generation_\\d+\\(\\d+\\)$\")\n\n\ndef _is_scope(e):\n    n = str(e.get(\"name\", \"\"))\n    return SCOPE_RE.match(n) is not None or (\"forward\" in n.lower() and \"gpu_model_runner\" in n.lower())\n\n\ndef _scope_kind(g, compute_eager, unresolved):\n    \"\"\"FULL = graph-launched with no eager model compute (attention / GDN / GEMM / NCCL) inside the execution scope;\n    eager input-preparation kernels are counted apart and allowed. Unresolved eager launches never read FULL.\"\"\"\n    if g == 0:\n        return \"EAGER\"\n    if compute_eager:\n        return \"PIECEWISE\"\n    if unresolved:\n        return \"PIECEWISE\" if g > 1 else \"MIXED\"\n    return \"FULL\"\n\n\ndef trace_summary(pdir):\n    files = sorted(Path(pdir).rglob(\"*.pt.trace.json*\"), key=lambda p: p.stat().st_mtime)\n    if not files:\n        return {\"error\": \"NO_TRACE_FILE\", \"listing\": [str(p) for p in Path(pdir).rglob(\"*\")][:20]}\n    f = next((p for p in files if re.search(r\"rank0\\.|rank[-_]?0[._]|tp0_\", p.name)), files[0])\n    with (gzip.open if f.name.endswith(\".gz\") else open)(f, \"rt\", encoding=\"utf-8\") as fh:\n        tr = json.load(fh)\n    ev = tr.get(\"traceEvents\", [])\n    kern = [e for e in ev if e.get(\"ph\") == \"X\" and str(e.get(\"cat\", \"\")).lower() in (\"kernel\", \"gpu_kernel\")]\n    rt = [e for e in ev if e.get(\"ph\") == \"X\" and str(e.get(\"cat\", \"\")).lower() in (\"cuda_runtime\", \"cuda_driver\")]\n    is_graph = lambda e: \"GraphLaunch\" in str(e.get(\"name\", \"\"))\n    is_kl = lambda e: re.search(r\"LaunchKernel|cuLaunchKernel\", str(e.get(\"name\", \"\"))) is not None\n    # per-scope counts inside the execution annotations (rev 3h: execute_context_*_generation_*; the older\n    # gpu_model_runner forward ranges still count). rev 3c: user_annotation only, not the gpu_user_annotation copies.\n    fw = sorted([e for e in ev if e.get(\"ph\") == \"X\" and str(e.get(\"cat\", \"\")) == \"user_annotation\" and _is_scope(e)],\n                key=lambda e: e.get(\"ts\", 0))\n    kname = {(e.get(\"args\") or {}).get(\"correlation\"): str(e.get(\"name\", \"\")) for e in kern\n             if (e.get(\"args\") or {}).get(\"correlation\") is not None}\n    per_fw, aux = [], {}\n    for a in fw[:400]:\n        lo, hi, tid = a.get(\"ts\", 0), a.get(\"ts\", 0) + a.get(\"dur\", 0), a.get(\"tid\")\n        inside = [e for e in rt if e.get(\"tid\") == tid and lo <= e.get(\"ts\", -1) <= hi]\n        kls = [e for e in inside if is_kl(e)]\n        ce = un = 0\n        for e in kls:\n            kn = kname.get((e.get(\"args\") or {}).get(\"correlation\"))\n            if kn is None:\n                un += 1\n            elif _cat(kn) in COMPUTE_CATS:\n                ce += 1\n            else:\n                b = re.sub(r\"^\\s*void\\s+\", \"\", re.sub(r\"\\(anonymous namespace\\)::\", \"\", kn)).split(\"<\")[0].split(\"(\")[0][:60]\n                aux[b] = aux.get(b, 0) + 1\n        per_fw.append([sum(1 for e in inside if is_graph(e)), len(kls), ce, un])\n    corr = {(e.get(\"args\") or {}).get(\"correlation\"): str(e.get(\"name\", \"\")) for e in rt\n            if (e.get(\"args\") or {}).get(\"correlation\") is not None}\n    per, tot, names = {k: 0.0 for k, _ in CATS + [(\"other\", ())]}, 0.0, {}\n    src = {k: {\"graph\": 0, \"eager\": 0, \"unknown\": 0} for k in per}\n    for e in kern:\n        d = float(e.get(\"dur\", 0.0))\n        tot += d\n        names[e.get(\"name\", \"\")] = names.get(e.get(\"name\", \"\"), 0.0) + d\n        k = _cat(e.get(\"name\", \"\"))\n        per[k] += d\n        launcher = corr.get((e.get(\"args\") or {}).get(\"correlation\"))\n        src[k][\"graph\" if launcher and \"GraphLaunch\" in launcher else (\"eager\" if launcher else \"unknown\")] += 1\n    tot_k = sum(v[\"graph\"] + v[\"eager\"] for v in src.values())\n    eager_share = round(sum(v[\"eager\"] for v in src.values()) / tot_k, 3) if tot_k else None\n    fw_g = statistics.median([x[0] for x in per_fw]) if per_fw else None\n    fw_k = statistics.median([x[1] for x in per_fw]) if per_fw else None\n    kinds_l = [_scope_kind(g, ce, un) for g, k, ce, un in per_fw]\n    if per_fw:                   # rev 3f (GPT rev 3e audit: 2 FULL + 1 EAGER forwards read FULL by the median)\n        kinds = set(kinds_l)\n        verdict = kinds.pop() if len(kinds) == 1 else \"MIXED\"       # every scope must agree\n        basis = \"per_scope_annotation (all scopes; eager input-prep kernels allowed, eager model compute is not)\"\n    elif tot_k:                  # rev 3d (GPT-2 reproduced FULL with 0 forwards): CUPTI alone never proves FULL\n        att = src[\"attention\"]\n        hint = (\"FULL\" if att[\"eager\"] == 0 and (eager_share or 0) < 0.05 else\n                \"PIECEWISE\" if att[\"eager\"] > 0 and any(v[\"graph\"] for v in src.values()) else\n                \"EAGER\" if not any(v[\"graph\"] for v in src.values()) else \"MIXED\")\n        verdict, basis = \"UNPROVEN\", f\"no forward annotations (cupti hint: {hint})\"\n    else:\n        verdict, basis = \"UNKNOWN\", \"none\"\n    return {\"file\": f.name, \"files\": len(files), \"kernels\": len(kern), \"kernel_us_sum\": round(tot),\n            \"graph_launches\": sum(1 for e in rt if is_graph(e)), \"kernel_launches\": sum(1 for e in rt if is_kl(e)),\n            \"forwards\": len(per_fw), \"forward_kinds\": {k_: kinds_l.count(k_) for k_ in (\"FULL\", \"PIECEWISE\", \"EAGER\", \"MIXED\")},\n            \"per_forward_median_graph_launches\": fw_g, \"per_forward_median_kernel_launches\": fw_k,\n            \"per_forward_median_compute_eager\": statistics.median([x[2] for x in per_fw]) if per_fw else None,\n            \"per_forward_unresolved_eager\": sum(x[3] for x in per_fw),\n            \"scope_names\": sorted({str(a.get(\"name\", \"\")) for a in fw})[:5],\n            \"aux_eager_kernels_in_scopes\": dict(sorted(aux.items(), key=lambda x: -x[1])[:20]),\n            \"launch_source\": src, \"eager_kernel_share\": eager_share, \"graph_verdict\": verdict, \"verdict_basis\": basis,\n            \"by_category_pct\": {k: round(100 * v / tot, 1) for k, v in per.items()} if tot else {},\n            \"top_kernels_us\": dict(sorted(((k, round(v)) for k, v in names.items()), key=lambda x: -x[1])[:25]),\n            \"note\": \"kernel time sums overlapping streams; NCCL time includes rank-skew waiting\"}\n\n\ndef profile(port, pdir, secs=6.0, delay_iters=20, max_iters=40):\n    \"\"\"POST /start_profile (bounded by iterations; falls back to a bare POST), decode, POST /stop_profile, summary.\"\"\"\n    rec = {}\n    try:\n        U.urlopen(U.Request(f\"http://127.0.0.1:{port}/start_profile\", method=\"POST\",\n                            data=json.dumps({\"delay_iterations\": delay_iters, \"max_iterations\": max_iters}).encode(),\n                            headers={\"Content-Type\": \"application/json\"}), timeout=900).read()\n        rec[\"start_profile\"] = \"ok_bounded\"\n    except Exception as e:\n        rec[\"start_profile_bounded_error\"] = repr(e)[:200]\n        try:\n            U.urlopen(U.Request(f\"http://127.0.0.1:{port}/start_profile\", data=b\"\", method=\"POST\"), timeout=900).read()\n            rec[\"start_profile\"] = \"ok\"\n        except Exception as e2:\n            rec[\"start_profile\"] = repr(e2)[:200]\n    time.sleep(secs)\n    try:\n        U.urlopen(U.Request(f\"http://127.0.0.1:{port}/stop_profile\", data=b\"\", method=\"POST\"), timeout=900).read()\n        rec[\"stop_profile\"] = \"ok\"\n    except Exception as e:\n        rec[\"stop_profile\"] = repr(e)[:200]\n    time.sleep(20)\n    try:\n        rec[\"trace\"] = trace_summary(pdir)\n    except Exception as e:\n        rec[\"trace\"] = {\"error\": f\"TRACE_PARSE {type(e).__name__}: {e}\"[:300]}\n    return rec\n\n\ndef profiled_load(port, prompts, level, pdir, secs=6.0, max_new=8192, model=\"q27\", prefill_tok_s=700.0, spec=False):\n    \"\"\"`level` decoding streams (own window: the profiler slows decode), profile() only after prefill settled exactly\n    as in steady() (prompt counter + preemptions flat, every running request emitted and gaining tokens). rev 3d: no\n    profile when unsettled; the profile window must be decode-only (prompt counter flat) and, with `spec`, show\n    speculative drafts (else the trace cannot speak for multi-token verification).\"\"\"\n    plen = len(prompts[0]) if prompts else 0\n    settle = max(300.0, 120.0 + level * plen / prefill_tok_s)\n    ss = [Stream(port, prompts[i % len(prompts)], max_new, 1.0, 4321 + i, model=model) for i in range(level)]\n    for g in ss:\n        g.start()\n    rec, t0, flat, ok, prev, last_tok = {\"level\": level}, time.perf_counter(), None, 0, None, [0] * level\n    try:\n        while time.perf_counter() - t0 < settle:\n            time.sleep(2)\n            try:\n                m = prom(port)\n            except Exception:\n                continue\n            run, wait, pt, pre = int(m.get(RUN, 0)), int(m.get(WAIT, 0)), m.get(PROMPT), m.get(PREEMPT, 0)\n            now_tok = [len(g.tokens) for g in ss]\n            gaining = sum(1 for a, b in zip(last_tok, now_tok) if b > a)\n            last_tok = now_tok\n            first = sum(g.t_first is not None for g in ss)\n            calm = run > 0 and prev is not None and (pt, pre) == prev and first >= run and gaining >= run\n            if calm and run >= level and wait == 0:\n                ok += 1\n            elif calm and wait > 0:\n                flat = flat or time.perf_counter()\n                ok = ok + 1 if time.perf_counter() - flat > 30 else 0\n            else:\n                ok, flat = 0, None\n            prev = (pt, pre)\n            if ok >= 3:\n                break\n        rec[\"settled\"] = ok >= 3\n        if not rec[\"settled\"]:\n            rec[\"status\"] = \"INCONCLUSIVE_PREFILL_NOT_SETTLED\"\n            return rec\n        try:\n            m0 = prom(port)\n            rec[\"running_at_start\"] = m0.get(RUN)\n        except Exception:\n            m0 = {}\n        rec.update(profile(port, pdir, secs))\n        try:\n            m1 = prom(port)\n        except Exception:\n            m1 = {}\n        rec[\"window_deltas\"] = {\"prompt_tokens\": m1.get(PROMPT, 0) - m0.get(PROMPT, 0), \"drafts\": m1.get(DRF, 0) - m0.get(DRF, 0),\n                                \"accepted\": m1.get(ACC, 0) - m0.get(ACC, 0), \"generated\": m1.get(GEN, 0) - m0.get(GEN, 0)}\n        if not m0 or not m1:\n            rec[\"status\"] = \"INCONCLUSIVE_METRICS_UNAVAILABLE\"\n        elif rec[\"window_deltas\"][\"prompt_tokens\"]:\n            rec[\"status\"] = \"INCONCLUSIVE_PREFILL_IN_PROFILE\"\n        elif spec and not (rec[\"window_deltas\"][\"drafts\"] and rec[\"window_deltas\"][\"accepted\"]):\n            rec[\"status\"] = \"INCONCLUSIVE_NO_SPEC_DECODE\"            # rev 3e: drafts AND accepted tokens in the window\n        return rec\n    finally:\n        for g in ss:\n            g.stop()\n        for g in ss:\n            g.join(10)\n        rec[\"idle_after\"] = wait_idle(port)\n"
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
    if not FAKE and not os.environ.get("NB2_CPU_KIT") and len(R["setup"]["gpus_at_start"] or []) != 4:   # CPU kit: no GPUs
        R["errors"].append(f"expected 4 GPUs, nvidia-smi shows {R['setup']['gpus_at_start']}")
        save()
        raise RuntimeError("GPU_PREFLIGHT_FAILED: " + repr(R["setup"]["gpus_at_start"]))
    L.start_watchdog(HARD_STOP, lambda: ABORT.update(why="HARD_STOP reached: watchdog killed live servers"))
    log("NB2 | profile", NB2_PROFILE, "| budget", BUDGET // 60, "min |", json.dumps(R["inputs"])[:900])
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


    SHM = Path(os.environ.get("NB2_SHM", "/dev/shm"))


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


    def merlin_fn_server(name, kv, port, kv_bytes, cfg=None):
        """Merlin's launch.build() for Flash-Next on the L4 profile: KV pin lowered for 22.49 GiB cards, the checkpoint's own
        chat template at xhigh (Merlin passes medium otherwise), Merlin's host_phase checks recorded, E23's env overrides."""
        eng = merlin_install()
        extra = ["--served-model-name", "q27", "--max-model-len", str(MAX_LEN), "--port", str(port),
                 "--default-chat-template-kwargs", XHIGH]
        rc2_extra = list(extra)                            # NB2: the boot's trailing overrides come last (they win)
        extra += list((cfg or {}).get("extra") or [])
        if FAKE and cfg is not None:                   # NB2 local harness: the real build() argv into the fake
            return nb2_fake_server(name, kv, port, kv_bytes, cfg, extra, rc2_extra)
        if FAKE:
            return MServer(name, [sys.executable, FAKE, str(CKPT) + "-w4a16", "--tensor-parallel-size", "2", "--max-num-seqs",
                                  "8" if kv == "fp8" else "3", "--port", str(port), "--max-model-len", str(MAX_LEN)], os.environ, port, LOGS)
        kvv = eng["kv"]
        gcc_bin = str(Path(kvv["MERLIN_CC"]).parent) if kvv.get("MERLIN_CC") else ""
        base = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_PATH", "CUDA_HOME", "LD_PRELOAD", "CC", "CXX")}
        base.update(GPUS="0,1,2,3", PORT=str(port), KV_BYTES=str(kv_bytes), P2P="off",
                    VENV=kvv["MERLIN_VENV"], TREE=kvv["MERLIN_TREE"], CHAT_TEMPLATE="", WARM="0", CACHE_ROOT=f"{TMP}/merlin-cache",
                    PATH=f"{kvv['MERLIN_VENV']}/bin:{gcc_bin}:{eng['cuda_home']}/bin:{os.environ.get('PATH', '')}",
                    CUDA_HOME=eng["cuda_home"])
        if kv == "fp8":
            base["SCALES"] = str(CKPT / "qsa_kv_scales_262k.json")
        base.update((cfg or {}).get("env") or {})          # NB2: DRAFT_HEAD (Merlin knob)
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
        argv = nb2_argv_edit(plan.argv(), cfg)
        if cfg is not None:
            cfg["argv"] = list(argv)
            cfg["exact_diff"] = nb2_exact_diff(eng["ML"], base, rc2_extra, kv, cfg, argv, plan.env)
        HELP[name] = L.help_check("merlin", argv[:argv.index("serve") + 1], env, argv)
        R["setup"].setdefault("merlin_plans", {})[name] = {"argv_tail": argv[-60:], "status": plan.status, "host_phase": HOST_PHASE[name],
                                                           "env_keys": {k: env[k] for k in sorted(env) if k.startswith(("VLLM_", "NCCL_", "KV_", "P2P", "SCALES"))}}
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
        t = time.time()
        log(">>>", sec, name, f"(est {est_min} min, left {left() / 60:.0f} min)")
        try:
            out = fn() or {}
            rec.update(out)
            if rec.get("status") in (None, "NOT_RUN"):
                rec["status"] = "DONE"
        except Exception as e:
            rec.update(status="ERROR", error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2500:])
        rec["sec"] = round(time.time() - t, 1)
        LIVE["rec"] = None
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


    def merlin_boot(name, kv, port, timeout, body, need=(), expected_rows=4, require_sampled=True, cfg=None):
        """Boot gate (ok must be True), one Merlin boot, cleanup of the /dev/shm entries this boot created; a boot OOM
        retries once 0.25 GiB lower. A server death during TF is TF_OOM only with an OOM line and no device fault."""
        kvb = KV_FP8 if kv == "fp8" else KV_BF16
        tries = []
        while True:
            gate = boot_gate_nb2(name)                      # NB2: GPU < 150 MiB + page-cache drop + host RAM gate
            if gate["ok"] is not True:
                return {"status": gate.get("fail_status") or "GPU_NOT_EMPTY", "boot_gate": gate, "tries": tries}
            shm0 = shm_names()
            o = with_servers([merlin_fn_server(name, kv, port, kvb, cfg)], timeout, body)
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



if not RERUN:
    # ==================================================================== CELL 6 — E25 NB2: knobs, readback, measurements (new code)
    import ast
    import gzip
    import statistics
    NB2_OPT = [x.strip() for x in (os.environ.get("NB2_OPTIONAL", "")).split(",") if x.strip()]   # A-7: K0 / CTRL off by default
    DECAY_MIN = float(os.environ.get("NB2_DECAY_MIN") or 20.0)          # the one long window (boot A only), 2 x 10 min
    DECAY_SUB_MIN = float(os.environ.get("NB2_DECAY_SUB_MIN") or 10.0)  # acceptance per 10-minute sub-window
    DECAY_LEVEL = 5                       # 5 x (ceil(~50k / 4096) + F) blocks stays inside the 149-block FP8 pool
    READY_SEC = 30 * 60                   # boot readiness gate (spec: <= 30 min per boot)
    HOST_FLOOR_MIB = (69 + 4) * 1024      # Merlin's host floor (max - (current - inactive_file) >= 69 GiB) + 4 GiB margin
    POOL_BLOCKS_RC2 = 149                 # RC2 kv_cache_usage denominator (usage 0.97315 = 145/149)
    F_PLAN = 6                            # RC2 analysis: 6 fixed blocks per running request at K3 (3 mamba groups x 2); the
                                          # census of each boot re-measures F (it changes with K). kv_cache_usage also counts
                                          # retained finished sessions (floor(L/4096)+3 each, up to 16): never live demand.
    TRACE_RESERVE = 180 if SMOKE else 300  # seconds of boot time kept for the profiler window
    LADDER_MARGIN = 240 if SMOKE else 400  # admission + settle allowance a ladder band needs beyond its window
    WARM_MARGIN = 300 if SMOKE else 600    # warm-up stops when the boot has less than this (+ the trace reserve) left
    SETUP_EST_MIN = 10                    # A0 Merlin install (RC2: 3.5 min; the sha256 check runs beside it)
    SHIPPED_CAPTURES = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    _M_LADDER = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 40, 48, 64, 80, 96, 112, 128]   # Merlin 3.0.0 registry._LADDER
    PROF = TMP / "nb2_prof"
    SEQS16 = {"accepted": None, "why": "not tried yet"}
    BOOT = {"deadline": None, "why": None, "key": None}
    RC2_REFS = json.loads("{\"source\":\"RC2 v1 FP8 boot greedy oracle rows (traces/requests.jsonl, port 8001): ORACLE_P + LONG_P = 2k v0, 2k v1, 8k v2, 40k v3\",\"rows\":[{\"prompt_sha\":\"8df9ed2c640ef5bd\",\"text\":\"We need solve ARC task. Need infer transformation from examples. User gives Task 0934a4d8. Example 1 input 30x30? output 9x4? Example 2 input 30x30? output 4x5? Example 3 input: \\\"194499276\\\" only one row? Maybe truncated? Need find transformation rule and solve test inputs. But only one test input? It says Example 3 input: 194499276. Maybe that's the test input? Need output. Need infer from\",\"tokens\":[1596,1144,11290,67964,3274,13,14235,22846,17454,494,9989,13,2570,6486,5261,220,15,24,18,19,64,19,67,23,13,12996,220,16,1879,220,18,15,87,18,15,30,2468,220,24,87,19,30,12996,220,17,1879,220,18,15,87,18,15,30,2468,220,19,87,20,30,12996,220,18,1879,25,328,16,24,19,19,24,24,17,22,21,1,1132,799,2713,30,10380,57739,30,14235,1423,17454,5728,321,11290,1228,10804,13,1921,1132,799,1228,1879,30,1049,2640,12996,220,18,1879,25,220,16,24,19,19,24,24,17,22,21,13,10380,421,579,279,1228,1879,30,14235,2468,13,14235,22846,494],\"margins\":[11.5,11.25,0.25,3.625,0.375,2.25,2.25,0.875,4.125,0.25,4.25,1.25,0.125,1.0,1.75,4.625,9.0,9.9375,10.375,9.5,8.875,10.6875,14.0,10.625,0.125,0.75,0.875,7.5,4.5,1.25,3.5,0.75,0.0,1.75,4.0,2.625,1.25,9.375,5.5,4.25,2.0,0.25,2.625,0.875,8.875,3.75,4.375,5.75,5.5,4.25,3.5,5.5,0.0,8.0,7.25,7.375,4.625,5.875,1.5,3.75,4.875,9.125,6.75,0.25,0.625,8.25,8.25,12.3125,11.5,10.3125,11.0,10.1875,11.625,9.875,4.375,2.25,0.375,1.0,4.875,0.0,0.25,5.75,2.125,0.0,0.25,2.375,2.875,6.75,9.125,5.125,1.875,0.75,2.375,0.0,0.875,1.875,2.125,0.75,2.25,1.625,9.1875,7.125,9.125,5.625,1.125,7.9375,7.0625,10.375,11.375,10.8125,10.5625,11.1875,11.5,11.625,1.0,1.625,1.375,2.625,0.5,2.375,1.75,2.625,2.0,2.625,0.125,1.5,0.25,1.5],\"error\":null,\"done\":true,\"finish_reason\":\"length\"},{\"prompt_sha\":\"8750743e9dcadca9\",\"text\":\"We need solve ARC tasks. Need infer transformation rules from examples and apply to test inputs. We have two tasks: 1818057f and 195c6913. Need output transformed test grids. Need analyze carefully.\\n\\nWe need produce final answer likely grids. Need infer rules.\\n\\nTask 1818057f examples:\\nInput/output grids with colors 2,4,8. Output changes some 4s to 8? Let's inspect.\\n\\nExample1 10x12? Input rows:\\n1 222244\",\"tokens\":[1596,1144,11290,67964,8806,13,14235,22846,17454,5426,494,9989,321,3668,310,1228,10804,13,1165,599,1330,8806,25,220,16,23,16,23,15,20,22,69,321,220,16,24,20,66,21,24,16,18,13,14235,2468,22773,1228,54455,13,14235,22903,15060,13,271,1596,1144,7936,1534,4087,4222,54455,13,14235,22846,5426,13,271,6065,220,16,23,16,23,15,20,22,69,9989,25,198,2425,45447,54455,440,7736,220,17,11,19,11,23,13,8984,4203,1010,220,19,82,310,220,23,30,6558,579,23331,13,271,12929,16,220,16,15,87,16,17,30,5396,6761,25,198,16,220,17,17,17,17,19,19],\"margins\":[11.875,11.75,0.5,3.125,2.625,3.0,3.125,0.5,3.0,0.5,0.625,4.875,1.0,3.25,2.0,5.625,0.625,3.875,0.375,0.75,1.375,6.75,1.375,2.125,12.8125,11.9375,11.6875,13.1875,12.0625,8.4375,10.3125,10.5,1.625,7.25,13.5625,12.625,10.5625,9.375,12.0,12.6875,11.3125,11.0625,2.0,2.375,1.625,0.875,1.0,0.125,0.5,1.25,0.125,1.875,6.625,1.625,1.5,1.5,1.0,4.75,4.625,2.0,0.875,1.5,1.0,0.625,0.5,3.375,1.0,1.125,4.5,12.375,5.625,11.125,12.625,10.0625,9.875,10.125,11.625,0.125,1.375,0.5,1.0,0.75,0.375,1.25,1.0,1.75,7.75,0.75,7.75,2.375,3.625,1.625,0.25,2.0,3.5,2.75,1.125,2.25,2.375,10.8125,10.75,0.75,1.875,7.5,2.75,5.25,0.25,2.125,1.25,0.0,7.125,3.625,5.5,4.5,2.0,1.5,0.75,0.875,0.0,6.375,1.0,0.75,10.0,4.75,12.625,11.5625,8.375,11.4375],\"error\":null,\"done\":true,\"finish_reason\":\"length\"},{\"prompt_sha\":\"fc63e297d382cd8f\",\"text\":\"We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We have multiple tasks: 247ef758, 269e22fb, 271d71e2, 28a6681f, 291dc1e1. Need output transformed test inputs. Need infer rules from examples. Need be careful. We can analyze each.\\n\\nWe need produce final answer likely grids for each test input. Need determine exact transformations.\\n\\nLet's parse tasks.\\n\\nTask 247ef758: examples show\",\"tokens\":[1596,1144,11290,67964,8806,13,14235,1423,17454,5728,364,1754,3274,321,11290,1228,10804,13,1165,599,5081,8806,25,220,17,19,22,797,22,20,23,11,220,17,21,24,68,17,17,10481,11,220,17,22,16,67,22,16,68,17,11,220,17,23,64,21,21,23,16,69,11,220,17,24,16,7385,16,68,16,13,14235,2468,22773,1228,10804,13,14235,22846,5426,494,9989,13,14235,381,16097,13,1165,628,22903,1754,13,271,1596,1144,7936,1534,4087,4222,54455,364,1754,1228,1879,13,14235,7995,4581,51192,13,271,9764,579,4563,8806,13,271,6065,220,17,19,22,797,22,20,23,25,9989,1420],\"margins\":[11.875,9.75,1.25,1.5,3.625,2.125,3.0,0.125,5.25,0.875,1.625,5.5,0.875,3.125,2.875,2.875,8.5,7.125,0.375,0.375,1.875,5.375,1.625,3.0,12.125,13.3125,12.9375,11.8125,11.8125,13.5,12.5,0.375,8.875,11.6875,10.9375,12.25,11.0625,10.6875,10.375,8.3125,7.375,10.4375,11.1875,9.0,9.8125,10.125,10.25,10.8125,10.125,9.3125,8.0625,9.5,12.4375,10.4375,11.0,11.1875,11.1875,9.6875,10.375,9.0,7.75,7.625,12.875,10.8125,11.375,9.625,10.625,11.0625,9.875,2.375,2.875,2.125,0.125,0.0,0.875,0.75,2.0,0.625,3.125,0.875,8.375,3.875,0.125,1.25,4.125,2.875,0.375,0.125,0.5,0.125,0.875,1.375,2.5,2.25,0.75,4.0,1.75,2.0,0.875,0.625,1.125,3.375,4.25,3.75,0.875,0.125,0.5,0.875,6.25,1.5,3.125,10.25,2.625,2.25,3.25,3.625,3.625,6.75,10.375,7.625,12.5,11.75,11.25,11.875,13.4375,1.5,0.0,0.25],\"error\":null,\"done\":true,\"finish_reason\":\"length\"},{\"prompt_sha\":\"49b3958bb234d4fd\",\"text\":\"We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We have multiple tasks with examples and test inputs. Need output transformed grids likely. Need infer rules. This is complex. Need analyze each task carefully. Need produce final answer with outputs for each test input. Need be accurate.\\n\\nWe have tasks:\\n2c181942 (one test input)\\n2d0172a1 (two test inputs? It lists Test input twice)\\n31f7f899 (one test)\\n332f06d7 (one test)\",\"tokens\":[1596,1144,11290,67964,8806,13,14235,1423,17454,5728,364,1754,3274,321,11290,1228,10804,13,1165,599,5081,8806,440,9989,321,1228,10804,13,14235,2468,22773,54455,4222,13,14235,22846,5426,13,1061,369,6150,13,14235,22903,1754,3274,15060,13,14235,7936,1534,4087,440,15796,364,1754,1228,1879,13,14235,381,12995,13,271,1596,599,8806,25,198,17,66,16,23,16,24,19,17,318,588,1228,1879,8,198,17,67,15,16,22,17,64,16,318,19186,1228,10804,30,1049,11140,3284,1879,10598,8,198,18,16,69,22,69,23,24,24,318,588,1228,8,198,18,18,17,69,15,21,67,22,318,588,1228,8],\"margins\":[10.0,11.25,1.0,0.625,3.375,2.5,2.875,0.375,5.125,0.875,1.75,5.875,0.375,2.875,3.625,4.125,6.375,7.5,0.25,0.0,0.25,6.25,0.25,3.75,2.0,2.625,3.125,2.875,4.125,2.75,0.5,1.375,0.625,0.5,1.875,0.125,1.375,0.25,0.0,9.625,0.125,1.375,1.0,0.625,3.125,4.125,0.25,1.875,0.125,0.875,3.875,1.5,2.75,0.0,2.5,0.875,3.0,4.75,1.25,1.0,0.125,1.75,10.625,1.875,2.625,0.125,3.0,3.5,1.125,0.25,10.5,13.375,13.3125,12.0625,10.375,9.625,11.0,0.125,1.625,7.5,1.0,3.5,7.75,7.0,10.375,13.0625,13.1875,12.0,11.1875,11.875,11.1875,7.125,5.75,5.25,8.5,0.625,0.75,0.75,1.375,7.375,0.75,0.0,4.375,8.9375,9.875,12.8125,11.8125,10.6875,10.1875,10.5625,12.125,4.625,5.25,1.125,0.0,10.875,11.6875,6.875,13.0,11.125,12.0,10.0625,11.375,10.5,9.25,9.0,2.625,5.25],\"error\":null,\"done\":true,\"finish_reason\":\"length\"}]}")
    RC2_CONTROL = json.loads("{\"source\":\"foysalemonshanto/q27-xhigh-e13-flashnext-rc2 v1 (rc2_results.json A3/merlin_fp8), MTP K3, BF16 draft head, seqs 8\",\"windows\":{\"40k@1\":{\"status\":\"STEADY\",\"level\":1,\"tok_s\":70.0,\"running_mean\":1.0,\"gen_tokens\":6305,\"running_min\":1.0,\"running_max\":1.0,\"accept_per_draft\":2.086,\"mean_acceptance_length\":3.086,\"step_ms\":44.1,\"tokens_per_step\":3.09,\"kv_usage_mean\":0.297,\"kv_usage_max\":0.30201342281879195,\"preemptions\":0.0,\"window_sec\":90.0,\"steps_source\":\"iteration_histogram\",\"prompt_tokens\":40000},\"40k@2\":{\"status\":\"STEADY\",\"level\":2,\"tok_s\":107.4,\"running_mean\":2.0,\"gen_tokens\":9665,\"running_min\":2.0,\"running_max\":2.0,\"accept_per_draft\":2.251,\"mean_acceptance_length\":3.251,\"step_ms\":30.3,\"tokens_per_step\":3.25,\"kv_usage_mean\":0.409,\"kv_usage_max\":0.4161073825503355,\"preemptions\":0.0,\"window_sec\":90.0,\"steps_source\":\"iteration_histogram\",\"prompt_tokens\":40000},\"40k@4\":{\"status\":\"STEADY\",\"level\":4,\"tok_s\":161.0,\"running_mean\":4.0,\"gen_tokens\":14496,\"running_min\":4.0,\"running_max\":4.0,\"accept_per_draft\":2.017,\"mean_acceptance_length\":3.017,\"step_ms\":37.5,\"tokens_per_step\":6.03,\"kv_usage_mean\":0.546,\"kv_usage_max\":0.5503355704697986,\"preemptions\":0.0,\"window_sec\":90.0,\"steps_source\":\"iteration_histogram\",\"prompt_tokens\":40000},\"40k@5\":{\"status\":\"STEADY\",\"level\":5,\"tok_s\":190.5,\"running_mean\":5.0,\"gen_tokens\":17145,\"running_min\":5.0,\"running_max\":5.0,\"accept_per_draft\":2.073,\"mean_acceptance_length\":3.073,\"step_ms\":40.3,\"tokens_per_step\":7.68,\"kv_usage_mean\":0.652,\"kv_usage_max\":0.6778523489932886,\"preemptions\":0.0,\"window_sec\":90.0,\"steps_source\":\"iteration_histogram\",\"prompt_tokens\":40000},\"8k@8\":{\"status\":\"STEADY\",\"level\":8,\"tok_s\":261.8,\"running_mean\":8.0,\"gen_tokens\":23563,\"running_min\":8.0,\"running_max\":8.0,\"accept_per_draft\":2.154,\"mean_acceptance_length\":3.154,\"step_ms\":48.2,\"tokens_per_step\":12.62,\"kv_usage_mean\":0.973,\"kv_usage_max\":0.9731543624161074,\"preemptions\":0.0,\"window_sec\":90.0,\"steps_source\":\"iteration_histogram\",\"prompt_tokens\":8192},\"90k@3\":{\"status\":\"INCONCLUSIVE_PREFILL_IN_WINDOW\",\"level\":3,\"tok_s\":48.7,\"running_mean\":1.62,\"gen_tokens\":4385,\"running_min\":1.0,\"running_max\":3.0,\"accept_per_draft\":1.93,\"mean_acceptance_length\":2.93,\"step_ms\":60.1,\"tokens_per_step\":2.93,\"kv_usage_mean\":0.937,\"kv_usage_max\":0.9932885906040269,\"preemptions\":0.0,\"window_sec\":90.0,\"steps_source\":\"iteration_histogram\",\"prompt_tokens\":90000},\"90k@4\":{\"status\":\"STEADY\",\"level\":4,\"tok_s\":159.8,\"running_mean\":4.0,\"gen_tokens\":14387,\"running_min\":4.0,\"running_max\":4.0,\"accept_per_draft\":2.022,\"mean_acceptance_length\":3.022,\"step_ms\":37.8,\"tokens_per_step\":6.04,\"kv_usage_mean\":0.978,\"kv_usage_max\":0.9798657718120806,\"preemptions\":0.0,\"window_sec\":90.0,\"steps_source\":\"iteration_histogram\",\"prompt_tokens\":90000},\"116k@3\":{\"status\":\"STEADY\",\"level\":3,\"tok_s\":140.5,\"running_mean\":3.0,\"gen_tokens\":12648,\"running_min\":3.0,\"running_max\":3.0,\"accept_per_draft\":2.223,\"mean_acceptance_length\":3.223,\"step_ms\":34.4,\"tokens_per_step\":4.83,\"kv_usage_mean\":0.881,\"kv_usage_max\":0.8926174496644296,\"preemptions\":0.0,\"window_sec\":90.0,\"steps_source\":\"iteration_histogram\",\"prompt_tokens\":116000},\"116k@4\":{\"status\":\"STEADY\",\"level\":4,\"tok_s\":160.3,\"running_mean\":4.0,\"gen_tokens\":14427,\"running_min\":4.0,\"running_max\":4.0,\"accept_per_draft\":2.016,\"mean_acceptance_length\":3.016,\"step_ms\":37.6,\"tokens_per_step\":6.03,\"kv_usage_mean\":0.957,\"kv_usage_max\":0.9731543624161074,\"preemptions\":0.0,\"window_sec\":90.0,\"steps_source\":\"iteration_histogram\",\"prompt_tokens\":116000}},\"kv_bytes\":2479838700,\"kv_tokens\":[513940],\"max_concurrency\":[[\"131,072\",\"3.92\"]],\"usage_denominator_blocks\":149,\"spec_log_whole_boot\":{\"lines\":315,\"drafts\":102238.0,\"per_position\":[0.8344,0.6601,0.5156],\"accept_per_draft\":2.0102,\"mean_acceptance_length\":3.0102},\"log_facts\":{\"non_default_found\":true,\"max_num_seqs\":8,\"max_num_batched_tokens\":1024,\"speculative_config\":{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"draft_sample_method\":\"probabilistic\"},\"capture_sizes\":[1,2,3,4,6,8,12,16,24,32],\"engine_spec\":{\"method\":\"mtp\",\"num_spec_tokens\":3},\"engine_spec_none\":false,\"draft_head_lines\":[],\"spec_query_len\":[4],\"replayssm_off_lines\":[],\"captures_done\":[[\"PIECEWISE\",10],[\"FULL\",6]],\"capture_modes_resolved\":[\"target=FULL_AND_PIECEWISE\",\"target=FULL_AND_PIECEWISE draft.prefill_cudagraph_manager=FULL_AND_PIECEWISE draft.decode_cudagraph_manager=FULL_DECODE_ONLY\"],\"draft_identity_modes\":[\"prefill:FULL_AND_PIECEWISE,decode:FULL_DECODE_ONLY\"],\"speculator_capture\":true,\"commit_graph_lines\":[],\"recoverssm_path_lines\":[\"GDN RecoverSSM path taken: 8 spec rows, align=True\",\"PLE RecoverSSM path taken: 8 spec rows, align=True\"],\"initial_free_gib\":{\"Worker_PP1_TP1_EP1\":[21.67,2.31],\"Worker_PP1_TP0_EP0\":[21.67,2.31],\"Worker_PP0_TP1_EP1\":[21.67,2.31],\"Worker_PP0_TP0_EP0\":[21.67,2.31]},\"graph_capture\":{\"Worker_PP0_TP0_EP0\":[4,0.33],\"Worker_PP0_TP1_EP1\":[4,0.33],\"Worker_PP1_TP0_EP0\":[5,0.43],\"Worker_PP1_TP1_EP1\":[5,0.43]},\"model_loading\":{\"Worker_PP1_TP0_EP0\":[16.87,124.662817],\"Worker_PP0_TP1_EP1\":[17.33,126.924117],\"Worker_PP0_TP0_EP0\":[17.33,126.751756],\"Worker_PP1_TP1_EP1\":[16.87,873.474944]},\"block_bytes\":[14680064,16515072],\"kv_tokens\":513940,\"max_concurrency\":[131072,3.92],\"jit_during_inference\":5,\"jit_kernels\":[\"_causal_conv1d_update_kernel\",\"_fill_num_accepted_kernel\",\"_gdn_recoverssm_verify_kernel\",\"_ple_conv_kernel\",\"_ple_conv_writeback_kernel\"],\"sched_tokens_warning\":true,\"max_num_scheduled_tokens\":1024,\"kv_cache_group_sizes\":[4096,4096,4096,8,4096],\"mamba_page_pad_pct\":21.01,\"side_cache_line\":\"Side-cache layout: on; folded replicated Mamba group 3 (language_model.model.layers.1.ple) into Mamba group 1 (now 13 layers); per-worker block bytes unchanged [14680064, 16515072]; QSA ring group 3 packs 1024 requests per pool block\",\"draft_head_copy\":null,\"duplicate_keys_line\":\"Found duplicate keys --max-model-len, --port, --served-model-name, --default-chat-template-kwargs\",\"host_floor_lines\":[],\"mtp_kv_scale_lines\":[\"Qwen4Exp QSA E4M3 KV active on mtp.layers.48.self_attn.attn: k_scale=1 v_scale=1  (DEFAULT 1.0 -- no static calibration loaded)\"],\"mtp_kv_default_scale_1\":true,\"target_kv_scale_layers\":12,\"single_mtp_layer_warning\":true}}")
    R["control_rc2_fp8"] = RC2_CONTROL
    R["sources"] = json.loads("{\"build_e25_nb2.py\": \"ac97850081a354015f50338b7a91358c09d0775fec2b75bcd6d30073f82f9c95\", \"build_e13_rc2.py (RC2 cells)\": \"1c3bc236953408fc489428922f9e891a2d56d4ded80df8a0d5ac1f16ece8cf4f\", \"e23_lab.py (embedded LAB, LF)\": \"ef393fbb2ff7669ee4c2c99ee841b60da66514a3b86b7c0ce81fd5f64689ab5e\", \"e23_lab.py + NB2 DIAG top-logprob patch (embedded, LF)\": \"e9c4c57d734a2b88354f91e53490a8710d99556ea9eb5e4db77d4d0908aa7140\", \"build_e23_notebook.py\": \"07303d50e94a6e5a6cd5a26fc29eddfc969b29006bafcf19823d47ea92e18ba3\", \"rc2_results.json (control)\": \"6a5410db9f42fd8f4a47d7e3b089ae7ac048b3972e483eb4e80489cbf7c1ade5\", \"requests.jsonl (RC2 FP8 greedy rows)\": \"a0da2b10c23cbe994af4e319ce864334851c914beb767efff67dc870aa5e5921\", \"server_merlin_fp8.log (RC2 control log)\": \"d9e8019171887d59f25834d6dda4a5873a0057db1bbc9f475aa629c00693b983\", \"merlin launch.py (local source used by the harness)\": \"87d1c5b75654fdec040735f1cf7f1d80c6a4656a6ffcd0248243c7eefe61cb46\", \"requests.jsonl (failed NB2 SMOKE ad8a28a5, 8k@8 rows 37-44)\": \"a6b266a7cba4da748079267aa1a75a17eb14a1b4ed862c877a51032ed3133f28\"}")          # sha256 of every file this notebook was built from (control preserved)
    R["correctness_meaning"] = ("Greedy oracle vs the RC2 FP8 rows (K3, BF16 draft head) and vs boot A. With temperature 0 a "
                                "draft change may only move proposals, so any FAIL_HIGH_MARGIN is a defect; NEAR_TIE = numerics "
                                "of a different verify batch. CLEAN is not an ARC accuracy result: a speed winner still needs a "
                                "paired ARC run (PLAN section 0).")
    LADDER16 = [("40k", 5), ("40k", 8), ("8k", 8), ("8k", 12), ("8k", 16)]
    BOOTS = {      # A-7 (lead decision 2026-10-31): A / B / C all at seqs 16 with one capture list (CAPS16) -> they differ only in K
        "A": {"key": "A", "port": 8011, "k": 3, "head": "int8", "seqs": 16, "est_min": 72, "long_window": True,
              "ladder": list(LADDER16), "lever": "D2"},
        "B": {"key": "B", "port": 8012, "k": 2, "head": "int8", "seqs": 16, "est_min": 45, "ladder": list(LADDER16), "lever": "D1"},
        "C": {"key": "C", "port": 8013, "k": 1, "head": "int8", "seqs": 16, "est_min": 45, "ladder": list(LADDER16), "lever": "D1"},
        "BH": {"key": "BH", "port": 8016, "k": 2, "head": "bf16", "seqs": 16, "est_min": 45, "ladder": list(LADDER16), "lever": "D1",
               "fallback": True, "note": "BF16-head fallback of B: runs only when A failed for an INT8-head-specific reason"},
        "CH": {"key": "CH", "port": 8017, "k": 1, "head": "bf16", "seqs": 16, "est_min": 45, "ladder": list(LADDER16), "lever": "D1",
               "fallback": True, "note": "BF16-head fallback of C: runs only when A failed for an INT8-head-specific reason"},
        "K0": {"key": "K0", "port": 8014, "k": 0, "head": None, "seqs": 16, "est_min": 40, "optional": True,
               "ladder": list(LADDER16), "lever": "D1",
               "note": "optional speed arm (spec: only if budget remains): target-only decode, no MTP; the MTP-off correctness "
                       "reference and the FP8-vs-BF16 teacher-forced isolation live in NB0 (lead decision)"},
        "CTRL": {"key": "CTRL", "port": 8015, "k": 3, "head": "bf16", "seqs": 8, "est_min": 25, "optional": True, "exact_rc2": True,
                 "ladder": [("8k", 8)], "lever": "control",
                 "note": "the exact RC2 FP8 config (no profiler flag), one 8k@8 window: session-to-session drift of the control"},
    }
    SMOKE_EST_MIN = 24     # SMOKE: boot B inside the 35-min hard stop (step() needs est + 5 min left at its start)
    if SMOKE:              # decision 2026-10-31: the riskiest config (K2 + seqs 16 + INT8 head + profiler), short work only
        BOOTS["B"].update(est_min=SMOKE_EST_MIN, ladder=[("8k", 8)], smoke=True,
                          note="SMOKE: probes, sampled, short oracle, one 90-s 8k@8 window, one 8k@16 admission check, trace")
    if NB2_PROFILE == "SMOKE_CTRL":   # the exact RC2 FP8 control (make_cfg: no env, no trailing flag) through the SMOKE body
        BOOTS["CTRL"].update(est_min=SMOKE_EST_MIN, ladder=[("8k", 8)], smoke=True,
                             note="SMOKE_CTRL: the exact RC2 FP8 config, probes, sampled, short oracle, one 90-s 8k@8 window with "
                                  "the failed SMOKE's prompts / seed 1234 / sampling (no dev-mode reset, no trace: no profiler flag)")
    # 8k@8 window sampling seed (steady(): stream i gets seed + i; its default 1234 = the failed SMOKE). SMOKE_SEED2 only: 4321.
    WINDOW_SEED = 4321 if NB2_PROFILE == "SMOKE_SEED2" else 1234
    NB2_FAILED_SMOKE = json.loads("{\"source\":\"SMOKE ad8a28a5 (2026-10-31) traces/requests.jsonl rows 37-44 + nb2_results.json S2/B_k2_int8_s16 8k@8\",\"rows\":[{\"row\":37,\"stream\":0,\"seed\":1234,\"prompt_sha\":\"f4212f34e7a2cc3a\",\"n_tokens\":2857,\"finish_reason\":null,\"error\":null,\"tail_verdict\":\"LOW_VARIETY\",\"label\":\"MODEL_REPETITION_OBSERVED\",\"numeric_cause\":\"NUMERIC_CAUSE_UNDETERMINED\",\"last_500_token_ids\":[[2713,250],[30,250]],\"margins_kept\":0,\"evidence\":\"degenerate tail (window gate); no logprobs were requested in that run, so no margin evidence\"},{\"row\":38,\"stream\":1,\"seed\":1235,\"prompt_sha\":\"452f4eeeffdffcba\",\"n_tokens\":3017,\"finish_reason\":null,\"error\":null,\"tail_verdict\":null},{\"row\":39,\"stream\":2,\"seed\":1236,\"prompt_sha\":\"fc63e297d382cd8f\",\"n_tokens\":2897,\"finish_reason\":null,\"error\":null,\"tail_verdict\":null},{\"row\":40,\"stream\":3,\"seed\":1237,\"prompt_sha\":\"cabba475611f1ca9\",\"n_tokens\":2920,\"finish_reason\":null,\"error\":null,\"tail_verdict\":null},{\"row\":41,\"stream\":4,\"seed\":1238,\"prompt_sha\":\"b3074e57ef93918d\",\"n_tokens\":3031,\"finish_reason\":null,\"error\":null,\"tail_verdict\":null},{\"row\":42,\"stream\":5,\"seed\":1239,\"prompt_sha\":\"5c55ca1649204598\",\"n_tokens\":2972,\"finish_reason\":null,\"error\":null,\"tail_verdict\":null},{\"row\":43,\"stream\":6,\"seed\":1240,\"prompt_sha\":\"e5f0513c6123c379\",\"n_tokens\":2887,\"finish_reason\":null,\"error\":null,\"tail_verdict\":null},{\"row\":44,\"stream\":7,\"seed\":1241,\"prompt_sha\":\"2395ccdb274f065a\",\"n_tokens\":3024,\"finish_reason\":null,\"error\":null,\"tail_verdict\":null}],\"window\":{\"status\":\"FAIL_GARBAGE_UNDER_LOAD\",\"window_sec\":90.0,\"tok_s\":242.6,\"accept_per_draft\":1.592,\"gen_tokens\":21838}}")   # the failed SMOKE's 8k@8 rows (trace rows 37-44): prompt sha, seed, tail verdict
    # real single-task xhigh ARC prompts for the decay window (RC2's own head / task / tail pieces)
    DECAY_P = [_HEAD + _TASK_IDS[k] + _TAIL for k in _ids[:48]]
    PROMPTS["8k"] = PROMPTS["8k"] + [prompt(BANDS["8k"], v) for v in range(len(PROMPTS["8k"]), 16)]   # 16 distinct 8k prompts
    R["prompts"]["8k"]["n"] = len(PROMPTS["8k"])
    R["prompts"]["decay"] = {"n": len(DECAY_P), "len_min": min(map(len, DECAY_P)), "len_max": max(map(len, DECAY_P))}


    def install_rc2_refs():
        """The RC2 FP8 greedy rows become ROWS['rc2_fp8'] only when every prompt is token-identical (prompt sha) to RC2's.
        On the fake server they are not installed (real-model rows vs a fake: NB2_FAKE_RC2_REFS=1 forces them for tests)."""
        want = [hashlib.sha256(json.dumps(p).encode()).hexdigest()[:16] for p in ORACLE_P + LONG_P]
        rows = RC2_REFS["rows"]
        got = [r["prompt_sha"] for r in rows]
        ok = want == got and all(len(r["tokens"]) == NTOK and len(r["margins"]) == NTOK for r in rows)
        rec = {"prompt_sha_match": want == got, "want": want, "got": got, "ntok": NTOK, "source": RC2_REFS["source"]}
        if FAKE and os.environ.get("NB2_FAKE_RC2_REFS") != "1":
            rec.update(installed=False, why="fake server: RC2 rows are real-model text")
        elif ok:
            ROWS["rc2_fp8"] = [{k: r[k] for k in ("text", "tokens", "margins", "error", "done", "finish_reason")} for r in rows]
            rec["installed"] = True
        else:
            rec.update(installed=False, why="prompt sha / length mismatch: boot A becomes the self-reference")
        R["rc2_refs"] = rec
        return rec


    install_rc2_refs()


    # ---------------------------------------------------------------- DIAG profiles: prompt check + per-stream loop table
    def _psha(ids):
        return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]        # = the prompt_sha of traces/requests.jsonl


    def diag_prompt_check(n=8):
        """The 8k@8 window's prompts (PROMPTS['8k'][:n], stream i = prompt i) vs the failed SMOKE's trace rows 37-44 (stream
        order by seed). Recorded and printed at load; diag_verdict() requires same_set_and_order (prompt_identity, N2-1): a
        different ARC copy / tokenizer changes the shas, and then the run is no repeat of the failed window (DIAG_INCOMPLETE)."""
        want = [r["prompt_sha"] for r in sorted(NB2_FAILED_SMOKE["rows"], key=lambda r: r["stream"])]
        got = [_psha(p) for p in PROMPTS["8k"][:n]]
        rec = {"same_set": sorted(want) == sorted(got), "same_set_and_order": want == got, "failed_run_rows_37_44": want, "this_run": got,
               "mismatch_streams": [i for i, (a, b) in enumerate(zip(want, got)) if a != b], "source": NB2_FAILED_SMOKE["source"]}
        R["diag_prompt_check"] = rec
        return rec


    if DIAG:
        log("NB2 DIAG prompt check (8k@8 prompts vs the failed SMOKE trace rows 37-44):",
            "SAME SET AND ORDER" if diag_prompt_check()["same_set_and_order"] else "MISMATCH " + json.dumps(R["diag_prompt_check"]))


    DIAG_SAMPLING = {"top_p": 0.95, "top_k": 20, "ignore_eos": True}   # LAB Stream at temperature > 0 (= the failed SMOKE's rows)
    # N2-4: the DIAG window requests also ask for the top-5 logprobs. Sampling is unchanged (Merlin source, vllm/v1/sample/
    # sampler.py + rejection_sampler.py and v1/worker/gpu/sample/sampler.py + spec_decode/rejection_sampler.py: logprobs are
    # computed from the raw logits on a separate tensor / after the token is sampled; the per-request logprobs flag changes
    # the sampler choice only for processed_* logprobs modes (server default raw_logprobs, not overridden by Merlin), and the
    # window's explicit seeds already rule out the fused sampler). Logprobs do add response bytes and per-step CPU work.
    DIAG_LP = {"port": None, "n": 5 if DIAG else 0, "basis": "logprobs are read off the raw logits (logprobs_mode raw_logprobs) "
               "after / beside sampling in the Merlin vLLM 0.32 source; the sampled token and the seeded generator are unchanged"}
    _LAB_STREAM_INIT, _LAB_STREAM_ROW = L.Stream.__init__, L.Stream.row
    L.Stream.tops = None                  # per-token top-logprob lists; only DIAG window streams record them (the LAB parser)


    def _nb2_stream_init(self, *a, **kw):
        _LAB_STREAM_INIT(self, *a, **kw)
        if (DIAG_LP["n"] and DIAG_LP["port"] is not None and self.port == DIAG_LP["port"] and self.body.get("temperature")
                and "logprobs" not in self.body):
            self.body["logprobs"] = DIAG_LP["n"]
            self.body["return_tokens_as_token_ids"] = True
            self.tops = []


    def _nb2_stream_row(self):
        r = _LAB_STREAM_ROW(self)
        if self.tops is not None:
            r["top_logprobs"] = [list(x) for x in self.tops]
        return r


    L.Stream.__init__, L.Stream.row = _nb2_stream_init, _nb2_stream_row


    def loop_onset(ids, max_period=16, min_run=64):
        """The start of the periodic tail of a token list (a loop such as ids 2713 / 30 alternating): the earliest position from
        which ids[i] == ids[i + p] holds to the end for some period p <= max_period, with at least min_run tokens; None if no
        such tail."""
        best, n = None, len(ids)
        for p in range(1, max_period + 1):
            if n < p + min_run:
                break
            i = n - p - 1
            while i >= 0 and ids[i] == ids[i + p]:
                i -= 1
            start = i + 1
            if n - start >= min_run and (best is None or start < best["onset_token"]):
                best = {"onset_token": start, "period": p, "periodic_tail_tokens": n - start, "ids": list(ids[start:start + p])}
        return best


    def _top_id(k):
        s = str(k)
        return int(s.rsplit(":", 1)[1]) if s.startswith("token_id:") and s.rsplit(":", 1)[1].lstrip("-").isdigit() else s


    def logprobs_near(r, at, before=32, after=32):
        """Top-logprob ids + margins of the window positions [at - before, at + after) from a request row (when the server
        returned logprobs): [{pos, token, margin, top: [[id, logprob], ...]}]."""
        toks, mg, tops = r.get("tokens") or [], r.get("margins") or [], r.get("top_logprobs") or []
        out = []
        for j in range(max(0, at - before), min(len(toks), at + after)):
            out.append({"pos": j, "token": toks[j], "margin": mg[j] if j < len(mg) else None,
                        "top": [[_top_id(k), v] for k, v in tops[j]] if j < len(tops) else None})
        return out


    def diag_stream_table(port, n, seed, plen, max_new, wall_lo, wall_hi):
        """One row per window stream from traces/requests.jsonl (the full text + token ids of every request): this port, the
        band's prompt length, the window's max_tokens / temperature / top_p / top_k / ignore_eos, seed in [seed, seed + n)
        (stream = seed offset), written between wall_lo and wall_hi. Two matching rows for one seed are never overwritten: the
        table is AMBIGUOUS_STREAM_ROWS with duplicate_streams listed. first_garbage_window = the first garbage 2,000-token
        window (window_garbage, None if every window is clean); tail_verdict = the window gate's own check (degenerate() on the
        last 4,000 characters); a looped stream also gets loop_onset (periodic tail) and, when the server returned logprobs,
        the top-logprob ids / margins around the onset (the full lists stay in the trace row)."""
        rows = {}
        try:
            with open(REQ_LOG) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    sp = r.get("sampling") or {}
                    sd = sp.get("seed")
                    if (r.get("port") == port and r.get("prompt_len") == plen and sp.get("max_tokens") == max_new
                            and sp.get("temperature") == TEMP and sp.get("top_p") == DIAG_SAMPLING["top_p"]
                            and sp.get("top_k") == DIAG_SAMPLING["top_k"] and sp.get("ignore_eos") is DIAG_SAMPLING["ignore_eos"]
                            and isinstance(sd, int) and not isinstance(sd, bool) and 0 <= sd - seed < n
                            and wall_lo <= (r.get("t") or 0) <= wall_hi):
                        rows.setdefault(sd - seed, []).append(r)
        except Exception as e:
            return {"status": "NO_REQUEST_LOG", "error": repr(e)[:200], "streams": [], "rows_found": 0, "loop_count": None,
                    "duplicate_streams": []}
        dup = sorted(i for i, v in rows.items() if len(v) > 1)
        failed = {x["stream"]: x for x in NB2_FAILED_SMOKE["rows"]}
        table = []
        for i in range(n):
            rr, f = rows.get(i) or [], failed.get(i) or {}
            if not rr:
                table.append({"stream": i, "seed": seed + i, "missing": True, "failed_run_prompt_sha": f.get("prompt_sha")})
                continue
            r = rr[0]
            toks = r.get("tokens") or []
            x = {"stream": i, "seed": (r.get("sampling") or {}).get("seed"), "prompt_sha": r.get("prompt_sha"), "n_tokens": len(toks),
                 "first_garbage_window": window_garbage(toks) if toks else None,
                 "tail_verdict": L.degenerate((r.get("text") or "")[-4000:]) if toks else None,
                 "finish_reason": r.get("finish_reason"), "error": r.get("error"),
                 "prompt_sha_matches_failed_run": r.get("prompt_sha") == f.get("prompt_sha"),
                 "failed_run_tail_verdict": f.get("tail_verdict"), "failed_run_n_tokens": f.get("n_tokens"),
                 "logprobs": {"requested": (r.get("sampling") or {}).get("logprobs"), "n_margins": len(r.get("margins") or []),
                              "n_top": len(r.get("top_logprobs") or []),
                              "aligned": bool(toks) and len(r.get("margins") or []) == len(toks)}}
            if len(rr) > 1:
                x["duplicate_rows"] = len(rr)
            if x["first_garbage_window"] or x["tail_verdict"]:
                on = loop_onset(toks)
                at = on["onset_token"] if on else (x["first_garbage_window"] or {}).get("start_token")
                x["loop"] = {"label": "MODEL_REPETITION_OBSERVED", "numeric_cause": "UNDETERMINED", "onset": on,
                             "onset_source": "periodic_tail" if on else ("first_garbage_window" if at is not None else None),
                             "around_onset": logprobs_near(r, at) if at is not None and (r.get("margins") or r.get("top_logprobs")) else None,
                             "around_onset_note": None if r.get("margins") or r.get("top_logprobs") else
                             "the server returned no logprobs for this request (margins empty)"}
            table.append(x)
        loops = [x["stream"] for x in table if x.get("first_garbage_window") or x.get("tail_verdict")]
        return {"status": "AMBIGUOUS_STREAM_ROWS" if dup else "TABLE", "seed": seed, "rows_found": len(rows),
                "matching_rows": sum(len(v) for v in rows.values()), "duplicate_streams": dup, "streams": table,
                "looped_streams": loops, "loop_count": len(loops),
                "loop_count_windowed": sum(1 for x in table if x.get("first_garbage_window")),
                "loop_count_tail_gate": sum(1 for x in table if x.get("tail_verdict")),
                "failed_run_looped_streams": [x["stream"] for x in NB2_FAILED_SMOKE["rows"] if x.get("tail_verdict")],
                "match": dict(DIAG_SAMPLING, port=port, prompt_len=plen, max_tokens=max_new, temperature=TEMP,
                              seeds=[seed, seed + n - 1], wall=[round(wall_lo, 1), round(wall_hi, 1)]),
                "source": "traces/requests.jsonl (full text + token ids of every window stream)"}


    def capture_sizes_for(seqs, k):
        """Merlin's registry.capture_sizes_for (shipped ladder covering seqs x (k + 1) tokens); the installed tree's own
        function when Merlin is installed, else a copy of the 3.0.0 function."""
        ml = MERLIN_ENG.get("ML") if isinstance(MERLIN_ENG, dict) else None
        f = getattr(getattr(ml, "R", None), "capture_sizes_for", None)
        if f is not None:
            return list(f(int(seqs), int(k))), "merlin_registry"
        top = int(seqs) * (int(k) + 1)
        s = [x for x in _M_LADDER if x <= top]
        return (s if top in s else s + [top]), "copy_of_registry_3.0.0"


    def graph_plan(caps, k, seqs, max_batched=1024):
        """Uniform-decode FULL graphs as the 0.32 dispatcher builds them (v1/worker/gpu/cudagraph_utils.py _init_candidates):
        every capture size rounded up to a multiple of (1 + K), kept when <= seqs x (1 + K) and <= the largest capture.
        Per batch request count r: the FULL graph it pads into (None = no FULL decode graph)."""
        q = int(k) + 1
        caps = sorted({int(c) for c in caps if int(c) <= max_batched})
        mx = max(caps) if caps else 0
        full = sorted({-(-c // q) * q for c in caps if -(-c // q) * q <= min(int(seqs) * q, mx)})
        per = {}
        for r in range(1, int(seqs) + 1):
            t = r * q
            d = next((x for x in full if x >= t), None)
            per[r] = {"tokens": t, "full_graph": d, "pad_pct": round(100.0 * (d - t) / d, 1) if d else None}
        return {"query_len": q, "piecewise_sizes": caps, "full_decode_sizes": full, "full_count": len(full),
                "max_decode_tokens": int(seqs) * q, "covers_all_request_counts": all(v["full_graph"] for v in per.values()),
                "per_request_count": per}


    def microbatch_cover(gp, n, k):
        """Addendum 5: under PP2 one micro-batch holds ~C/2 requests, i.e. ceil(C/2) x (1 + K) verify tokens; the FULL decode
        graph it pads into (None = not covered: PIECEWISE / eager for that tick)."""
        r = max(1, math.ceil(n / 2))
        pr = (gp.get("per_request_count") or {}).get(r) or {}
        return {"requests": r, "tokens": r * (k + 1), "full_graph": pr.get("full_graph"), "pad_pct": pr.get("pad_pct"),
                "covered": pr.get("full_graph") is not None}


    def make_cfg(b, seqs=None):
        """One boot's knobs (Merlin names only: DRAFT_HEAD env; trailing --speculative-config / --max-num-seqs /
        --compilation-config, which come after the profile's flags and win; --profiler-config for the trace window)."""
        k, head, seqs = b["k"], b["head"], int(seqs or b["seqs"])
        env, extra, diff = {}, [], []
        if head in ("int8", "int4"):
            env["DRAFT_HEAD"] = head
            diff.append(f"env DRAFT_HEAD={head}: launch.py adds \"draft_lm_head_quant\":\"{head}\" to the profile's --speculative-config")
        spec = None
        if k:
            spec = {"method": "mtp", "num_speculative_tokens": k, "draft_sample_method": "probabilistic"}
            if head in ("int8", "int4"):
                spec["draft_lm_head_quant"] = head
            if k != 3:
                extra += ["--speculative-config", json.dumps(spec, separators=(",", ":"))]
                diff.append(f"trailing --speculative-config num_speculative_tokens={k} (restates probabilistic + the head)")
        else:
            diff.append("--speculative-config removed from the argv (target only; --use-replayssm falls back to stock GDN)")
        # A-7: every seqs > 8 boot takes ONE list, shipped U capture_sizes_for(seqs, 3) (covers K <= 3), so A / B / C (and K0)
        # differ only in K; seqs 8 keeps shipped U capture_sizes_for(8, K) (= shipped for K <= 3, RC2's list)
        capl, src = capture_sizes_for(seqs, 3 if seqs > 8 else k)
        caps = SHIPPED_CAPTURES if (seqs == 8 and k == 3) else sorted(set(SHIPPED_CAPTURES) | set(capl))
        if seqs != 8:
            extra += ["--max-num-seqs", str(seqs)]
            diff.append(f"trailing --max-num-seqs {seqs} (profile: 8)")
        if caps != SHIPPED_CAPTURES:
            extra += ["--compilation-config", json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": caps},
                                                         separators=(",", ":"))]
            diff.append(f"trailing --compilation-config captures {caps} (covers {seqs} x {k + 1} = {seqs * (k + 1)} tokens"
                        + (f"; the one seqs-{seqs} list of every boot = shipped U capture_sizes_for({seqs}, 3))" if seqs > 8 else ")"))
        pdir = PROF / b["key"]
        if not b.get("exact_rc2"):
            env["VLLM_SERVER_DEV_MODE"] = "1"
            diff.append("env VLLM_SERVER_DEV_MODE=1 (exposes POST /reset_prefix_cache for the cleared-cache block census; no compute change)")
            pdir.mkdir(parents=True, exist_ok=True)
            extra += ["--profiler-config", json.dumps({"profiler": "torch", "torch_profiler_dir": str(pdir),
                                                       "torch_profiler_with_stack": False, "ignore_frontend": True})]
            diff.append("trailing --profiler-config (torch; inactive until the trace window, which runs last)")
        name = f"nb2_{b['key']}_k{k}_{head or 'nohead'}_s{seqs}"
        return {"key": b["key"], "name": name, "port": b["port"], "k": k, "head": head or "none", "seqs": seqs, "spec": spec,
                "drop_spec": k == 0, "captures": caps, "capture_source": src, "env": env, "extra": extra,
                "diff_vs_rc2_fp8": diff or ["none (exact RC2 FP8)"], "prof_dir": str(pdir), "ladder": list(b["ladder"]),
                "exact_rc2": bool(b.get("exact_rc2")), "lever": b["lever"], "note": b.get("note"),
                "long_window": bool(b.get("long_window")), "smoke": bool(b.get("smoke")), "fallback": bool(b.get("fallback"))}


    def nb2_argv_edit(argv, cfg):
        """K0 only: drop every --speculative-config (and its value) from the final argv."""
        if not (cfg or {}).get("drop_spec"):
            return list(argv)
        out, skip = [], False
        for a in argv:
            if skip:
                skip = False
            elif a == "--speculative-config":
                skip = True
            elif not a.startswith("--speculative-config="):
                out.append(a)
        return out


    def flag_map(argv):
        """vllm serve flags -> last value (argparse: a repeated flag keeps the last one); a bare flag maps to True."""
        try:
            i = argv.index("serve") + 2
        except ValueError:
            i = 0
        out, a = {}, list(argv[i:])
        j = 0
        while j < len(a):
            if a[j].startswith("--"):
                if j + 1 < len(a) and not a[j + 1].startswith("--"):
                    out[a[j]] = a[j + 1]
                    j += 2
                    continue
                out[a[j]] = True
            j += 1
        return out


    def nb2_exact_diff(ML, base, rc2_extra, kv, cfg, argv, env):
        """The exact diff vs RC2 FP8: Merlin's build() of the RC2 config (same base env minus the boot's env, RC2's extra)
        against this boot's final argv / env."""
        try:
            b0 = {k: v for k, v in base.items() if k not in (cfg.get("env") or {})}
            p0 = ML.build("flash-next", PROFILE, str(CKPT), b0, extra=list(rc2_extra), kv=kv, checkpoint="flash-next", status="untested")
            f0, f1 = flag_map(p0.argv()), flag_map(argv)
            e0, e1 = dict(p0.env), dict(env)
            cfg["recoverssm_commit_graph_env"] = e1.get("VLLM_RECOVERSSM_COMMIT_GRAPH") or "unset (engine default: eager)"
            changed, benign = {}, {}
            for k in sorted(set(e0) | set(e1)):
                if e0.get(k) == e1.get(k) or k in ("PORT", "MASTER_PORT", "VLLM_PORT"):
                    continue
                dup = path_dup_prepend(e0.get(k), e1.get(k)) if k == "PATH" else None
                if dup:            # KIT-9: host_phase() (_toolchain) prepends {CUDA_HOME}/bin again; already on PATH = no change
                    benign[k] = {"duplicate_prepend": dup, "why": "entries already on the RC2 PATH (Merlin host_phase _toolchain "
                                                                  "re-prepends CUDA_HOME/bin); the launched PATH is base PATH in both"}
                else:
                    changed[k] = [e0.get(k), e1.get(k)]
            return {"flags_changed": {k: [f0[k], f1[k]] for k in f0 if k in f1 and f0[k] != f1[k]},
                    "flags_added": {k: f1[k] for k in f1 if k not in f0}, "flags_removed": {k: f0[k] for k in f0 if k not in f1},
                    "env_changed": changed, "env_benign": benign,
                    "duplicate_flags_in_argv": sorted({a for a in argv if a.startswith("--") and argv.count(a) > 1})}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"[:300]}


    def path_dup_prepend(p0, p1):
        """KIT-9: the leading PATH entries p1 adds in front of p0 when every one of them is already on p0 (a duplicate
        prepend: the same set of directories); None for any other difference. The server itself never sees this PATH:
        merlin_fn_server() launches with PATH = base PATH, identical for RC2 and NB2."""
        if not isinstance(p0, str) or not isinstance(p1, str) or p0 == p1:
            return None
        a, b = p0.split(":"), p1.split(":")
        n = len(b) - len(a)
        if n <= 0 or b[n:] != a:
            return None
        extra = b[:n]
        return extra if all(x in a for x in extra) else None


    def _local_ml():
        """Local harness only (NB2_MERLIN_SRC = a Merlin 3.0.0 source tree): the real launch.build() for the fake server."""
        src = os.environ.get("NB2_MERLIN_SRC")
        if not src:
            return None
        p = str(Path(src) / "tools" / "merlin")
        if p not in sys.path:
            sys.path.insert(0, p)
        import launch as _ML
        return _ML


    def nb2_fake_server(name, kv, port, kv_bytes, cfg, extra, rc2_extra):
        ml = _local_ml()
        if ml is not None:
            base = {"GPUS": "0,1,2,3", "PORT": str(port), "KV_BYTES": str(kv_bytes), "P2P": "off", "VENV": str(TMP / "fake_venv"),
                    "TREE": str(TMP / "fake_tree"), "CHAT_TEMPLATE": "", "WARM": "0", "CACHE_ROOT": str(TMP / "merlin-cache"),
                    "SCALES": str(CKPT / "qsa_kv_scales_262k.json")}
            base.update(cfg.get("env") or {})
            plan = ml.build("flash-next", PROFILE, str(CKPT), base, extra=list(extra), kv=kv, checkpoint="flash-next", status="untested")
            argv = nb2_argv_edit(plan.argv(), cfg)
            cfg["argv"] = list(argv)
            cfg["exact_diff"] = nb2_exact_diff(ml, base, rc2_extra, kv, cfg, argv, plan.env)
            flags, src = argv[argv.index("serve") + 2:], "merlin_build_local"
        else:
            flags = ["--max-num-seqs", "8", "--port", str(port), "--max-model-len", str(MAX_LEN)]
            if cfg["spec"]:
                flags += ["--speculative-config", json.dumps(cfg["spec"])]
            flags += list(cfg.get("extra") or [])
            src = "fake_flags"
        R["setup"].setdefault("merlin_plans", {})[name] = {"argv_tail": flags[-60:], "source": src}
        return MServer(name, [sys.executable, FAKE, str(CKPT) + "-w4a16"] + flags, {**os.environ, **(cfg.get("env") or {})}, port, LOGS)


    # ---------------------------------------------------------------- server-log facts + config readback
    SPEC_LINE = re.compile(r"SpecDecoding metrics: Mean acceptance length: ([\d.]+), .*?Accepted: (\d+) tokens, Drafted: (\d+) tokens, "
                           r"Per-position acceptance rate: ([\d., ]+?), Avg Draft")


    def spec_log_parse(t):
        """vLLM's 10-s SpecDecoding log lines -> drafted-weighted per-position acceptance (drafts = drafted tokens / K)."""
        n, acc_pos, drafts, acc = 0, None, 0.0, 0.0
        for m in SPEC_LINE.finditer(t or ""):
            rates = [float(x) for x in m.group(4).replace(" ", "").split(",") if x]
            k = len(rates)
            if not k:
                continue
            d = int(m.group(3)) / k
            acc_pos = acc_pos or [0.0] * k
            if len(acc_pos) != k:
                continue
            for i, r in enumerate(rates):
                acc_pos[i] += r * d
            drafts += d
            acc += int(m.group(2))
            n += 1
        if not n or not drafts:
            return {"lines": n}
        return {"lines": n, "drafts": round(drafts, 1), "per_position": [round(a / drafts, 4) for a in acc_pos],
                "accept_per_draft": round(acc / drafts, 4), "mean_acceptance_length": round(1 + acc / drafts, 4)}


    def log_facts_text(t):
        t = t or ""
        lines = t.splitlines()
        nd = next((l for l in lines if "non-default args:" in l), "")
        f = {"non_default_found": bool(nd)}
        m = re.search(r"'max_num_seqs': (\d+)", nd)
        f["max_num_seqs"] = int(m.group(1)) if m else None
        m = re.search(r"'max_num_batched_tokens': (\d+)", nd)
        f["max_num_batched_tokens"] = int(m.group(1)) if m else None
        m = re.search(r"'speculative_config': (\{[^{}]*\})", nd)
        try:
            f["speculative_config"] = ast.literal_eval(m.group(1)) if m else None
        except Exception:
            f["speculative_config"] = "UNPARSED " + m.group(1)[:200]
        m = re.search(r"'cudagraph_capture_sizes': \[([\d, ]*)\]", nd)
        f["capture_sizes"] = [int(x) for x in m.group(1).replace(" ", "").split(",") if x] if m else None
        m = re.search(r"SpeculativeConfig\(method='(\w+)', model='[^']*', num_spec_tokens=(\d+)\)", t)
        f["engine_spec"] = {"method": m.group(1), "num_spec_tokens": int(m.group(2))} if m else None
        f["engine_spec_none"] = "speculative_config=None" in t
        f["draft_head_lines"] = [l[-200:] for l in lines if "Draft lm_head:" in l][:4]
        f["spec_query_len"] = sorted({int(x) for x in re.findall(r"spec_query_len (\d+)", t)})
        f["replayssm_off_lines"] = [l[-200:] for l in lines if "needs speculative decoding; num_speculative_tokens=0" in l][:2]
        caps = re.findall(r"Capturing CUDA graphs \((\w+)\):\s*100%\|[^|\n\r]*\|\s*(\d+)/(\d+)", t)
        done = []
        for mode, a, b in caps:
            if not done or done[-1] != [mode, int(b)] or a != b:
                done.append([mode, int(b)])
        f["captures_done"] = done[:24]
        f["capture_modes_resolved"] = sorted(set(re.findall(r"resolved cudagraph modes before capture: ([^\r\n]+?)\.?\s*$", t, re.M)))[:6]
        f["draft_identity_modes"] = sorted(set(re.findall(r"cudagraph_mode=(\S+) role=draft", t)))
        f["speculator_capture"] = "Capturing model for speculator" in t
        f["commit_graph_lines"] = [l[-220:] for l in lines if "[recoverssm-commit]" in l][:6]
        f["recoverssm_path_lines"] = sorted(set(re.findall(r"((?:GDN|PLE) RecoverSSM path taken: \d+ spec rows, align=\w+)", t)))[:6]
        f["initial_free_gib"] = {w: [float(a), float(b)] for w, a, b in
                                 re.findall(r"\((Worker_\w+) pid=\d+\)[^\r\n]*Initial free memory ([\d.]+) GiB, reserved ([\d.]+) GiB", t)}
        f["graph_capture"] = {w: [int(a), float(b)] for w, a, b in
                              re.findall(r"\((Worker_\w+) pid=\d+\)[^\r\n]*Graph capturing finished in (\d+) secs, took ([\d.]+) GiB", t)}
        f["model_loading"] = {w: [float(a), float(b)] for w, a, b in
                              re.findall(r"\((Worker_\w+) pid=\d+\)[^\r\n]*Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds", t)}
        m = re.search(r"per-worker block bytes unchanged \[([\d, ]+)\]", t)
        f["block_bytes"] = [int(x) for x in m.group(1).replace(" ", "").split(",") if x] if m else None
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", t)
        f["kv_tokens"] = int(m.group(1).replace(",", "")) if m else None
        m = re.search(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x", t)
        f["max_concurrency"] = [int(m.group(1).replace(",", "")), float(m.group(2))] if m else None
        f["jit_during_inference"] = t.count("JIT compilation during inference")
        f["jit_kernels"] = sorted(set(re.findall(r"JIT compilation during inference: (\w+)", t)))[:20]
        f["sched_tokens_warning"] = "max_num_scheduled_tokens is set to" in t
        m = re.search(r"max_num_scheduled_tokens is set to (\d+)", t)
        f["max_num_scheduled_tokens"] = int(m.group(1)) if m else None
        m = re.search(r"kv cache group sizes \(([\d, ]+)\)", t)
        f["kv_cache_group_sizes"] = [int(x) for x in m.group(1).replace(" ", "").split(",") if x] if m else None
        m = re.search(r"Padding mamba page size by ([\d.]+)%", t)
        f["mamba_page_pad_pct"] = float(m.group(1)) if m else None
        f["side_cache_line"] = next((l[l.find("Side-cache layout"):][:300] for l in lines if "Side-cache layout" in l), None)
        m = re.search(r"Draft lm_head: (\w+) weight-only copy of the (\d+)x(\d+) shard \(([\d.]+) MiB", t)
        f["draft_head_copy"] = {"mode": m.group(1), "shard": [int(m.group(2)), int(m.group(3))], "mib": float(m.group(4))} if m else None
        f["duplicate_keys_line"] = next((l[-200:] for l in lines if "Found duplicate keys" in l), None)
        f["host_floor_lines"] = [l[-200:] for l in lines if re.search(r"69 GiB|host (RAM|memory).*floor", l)][:3]
        mtp_kv = [l[l.find("Qwen4Exp QSA"):][:220] for l in lines if "QSA E4M3 KV active on mtp." in l]
        f["mtp_kv_scale_lines"] = sorted(set(mtp_kv))[:2]
        f["mtp_kv_default_scale_1"] = any("DEFAULT 1.0" in l for l in mtp_kv)
        f["target_kv_scale_layers"] = len({re.sub(r"^.*KV active on (\S+):.*$", r"\1", l) for l in lines
                                           if "QSA E4M3 KV active on language_model" in l})
        f["single_mtp_layer_warning"] = "run multiple times of forward on same MTP layer" in t
        f["spec_log"] = spec_log_parse(t)
        return f


    def log_facts(s):
        return log_facts_text(s.log())


    def readback(f, cfg):
        """The server's own log must show the intended knobs (fail closed: a mismatch stops the boot's load tests)."""
        chk = {"non_default_args_line": bool(f.get("non_default_found")), "max_num_seqs": f.get("max_num_seqs") == cfg["seqs"],
               "capture_sizes": f.get("capture_sizes") == cfg["captures"]}
        sc = f.get("speculative_config")
        if cfg["k"] == 0:
            chk["no_mtp"] = sc is None and f.get("engine_spec") is None
        else:
            want_head = None if cfg["head"] in ("bf16", "none") else cfg["head"]
            chk["k"] = isinstance(sc, dict) and sc.get("num_speculative_tokens") == cfg["k"] and \
                (f.get("engine_spec") or {}).get("num_spec_tokens") == cfg["k"]
            chk["head_in_spec"] = isinstance(sc, dict) and sc.get("draft_lm_head_quant") == want_head
            chk["head_built_line"] = (any(want_head.upper() in l for l in f.get("draft_head_lines") or []) if want_head
                                      else not f.get("draft_head_lines"))
            chk["probabilistic"] = isinstance(sc, dict) and sc.get("draft_sample_method") == "probabilistic"
            chk["spec_query_len"] = (cfg["k"] + 1) in (f.get("spec_query_len") or [])
        return {"ok": all(chk.values()), "checks": chk}


    # ---------------------------------------------------------------- per-position acceptance (labelled Prometheus counter)
    _PROM_E23 = L.prom
    SNAP = {"lock": threading.Lock(), "by_port": {}}
    POS_RE = re.compile(r"^vllm:spec_decode_num_accepted_tokens_per_pos(?:_total)?\{([^}]*)\}\s+(\S+)")
    _SNAP_KEYS = (L.ACC, L.DRF, L.DTOK, L.GEN, L.ITER, L.PREEMPT, L.PROMPT, L.KVU, L.RUN, L.WAIT)


    def prom_pos_text(txt):
        out = {}
        for line in (txt or "").splitlines():
            m = POS_RE.match(line)
            if not m:
                continue
            pm = re.search(r'position="(\d+)"', m.group(1))
            try:
                v = float(m.group(2))
            except ValueError:
                continue
            if pm and v == v:
                out[int(pm.group(1))] = out.get(int(pm.group(1)), 0.0) + v
        return out


    def prom_pos(port):
        return prom_pos_text(L.U.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=15).read().decode("utf-8", "replace"))


    def _prom_nb2(port):
        """E23's prom() unchanged, plus a snapshot (with the per-position counters) kept for window_pos()."""
        d = _PROM_E23(port)
        try:
            pos = prom_pos(port)
        except Exception:
            pos = None
        with SNAP["lock"]:
            lst = SNAP["by_port"].setdefault(int(port), [])
            lst.append((time.perf_counter(), {k: d.get(k) for k in _SNAP_KEYS}, pos))
            del lst[:-5000]
        return d


    L.prom = _prom_nb2


    def window_pos(w, k):
        """Per-position acceptance of one window from the snapshots taken by steady()'s own m0 / m1 reads (prom() returns
        just before t0 / t1 are taken)."""
        if not k:
            return {"status": "N/A_NO_MTP"}
        port, t0, t1 = w.get("port"), w.get("t0"), w.get("t1")
        if not (port and t0 and t1):
            return {"status": "INCONCLUSIVE_NO_WINDOW"}
        with SNAP["lock"]:
            lst = list(SNAP["by_port"].get(int(port), []))
        a = max((s for s in lst if s[0] <= t0), key=lambda s: s[0], default=None)
        b = max((s for s in lst if s[0] <= t1), key=lambda s: s[0], default=None)
        if a is None or b is None or a is b:
            return {"status": "INCONCLUSIVE_NO_SNAPSHOT"}
        if a[2] is None or b[2] is None or not b[2]:
            return {"status": "INCONCLUSIVE_NO_PER_POS_METRIC"}
        drf = (b[1].get(L.DRF) or 0) - (a[1].get(L.DRF) or 0)
        if drf <= 0:
            return {"status": "INCONCLUSIVE_NO_DRAFTS"}
        accp = [(b[2].get(i, 0.0) - a[2].get(i, 0.0)) for i in range(k)]
        return {"status": "MEASURED", "drafts": int(drf), "accepted_by_pos": [int(x) for x in accp],
                "per_position": [round(x / drf, 4) for x in accp], "accept_per_draft": round(sum(accp) / drf, 4),
                "snap_dt": [round(t0 - a[0], 3), round(t1 - b[0], 3)]}


    # ---------------------------------------------------------------- host RAM: page-cache drop + Merlin's floor
    def drop_page_cache():
        """posix_fadvise(DONTNEED) on every checkpoint file (the merged dir's symlink targets = the Model mounts)."""
        t = time.time()
        rec = {"supported": hasattr(os, "posix_fadvise"), "files": 0, "bytes": 0, "errors": []}
        paths = set()
        roots = [CKPT] + [v[0] for v in PARTS.values() if v] + ([PLE_DIR] if PLE_DIR else [])
        for root in roots:
            try:
                for p in Path(root).rglob("*"):
                    try:
                        rp = p.resolve()
                        if rp.is_file():
                            paths.add(str(rp))
                    except Exception:
                        pass
            except Exception as e:
                rec["errors"].append(f"{root}: {e!r}"[:160])
        rec["files_seen"] = len(paths)
        if rec["supported"]:
            for p in sorted(paths):
                try:
                    fd = os.open(p, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                        rec["files"] += 1
                        rec["bytes"] += os.fstat(fd).st_size
                    finally:
                        os.close(fd)
                except Exception as e:
                    rec["errors"].append(f"{Path(p).name}: {e!r}"[:160])
        rec["errors"] = rec["errors"][:10]
        rec["sec"] = round(time.time() - t, 1)
        return rec


    def host_floor():
        """KIT-1: the host RAM floor. Real runs: always Merlin's 69 GiB + 4 GiB. FAKE mode only (the local / CPU-kit fake
        server; the Kaggle CPU kit has a 30 GiB cgroup): NB2_FAKE_HOST_FLOOR_MIB overrides it. Ignored without FAKE."""
        v = os.environ.get("NB2_FAKE_HOST_FLOOR_MIB")
        if FAKE and v:
            try:
                return int(v), "NB2_FAKE_HOST_FLOOR_MIB (fake server only)"
            except ValueError:
                pass
        return HOST_FLOOR_MIB, "Merlin 69 GiB + 4 GiB margin"


    def host_gate(wait_sec=180):
        """Merlin's own rule (max - (current - inactive_file)) >= 69 GiB + 4 GiB margin; MemAvailable when the cgroup has
        no limit; ok None when neither is readable (recorded, not blocking)."""
        t, hist = time.time(), []
        floor, floor_src = host_floor()
        while True:
            h = host_mem()
            av, basis = h.get("avail_mib"), "cgroup max - (current - inactive_file)"
            if av is None:
                av, basis = h.get("MemAvailable_mib"), "MemAvailable"
            hist.append(av)
            out = {"avail_mib": av, "basis": basis, "floor_mib": floor, "floor_source": floor_src, "waited_sec": round(time.time() - t),
                   "host": h, "samples": hist[-6:]}
            if av is None:
                return dict(out, ok=None, why="no cgroup / meminfo numbers")
            if av >= floor:
                return dict(out, ok=True)
            if time.time() - t >= wait_sec:
                return dict(out, ok=False)
            time.sleep(10)


    def boot_gate_nb2(name):
        """RC2's GPU gate (< 150 MiB per card), then the page-cache drop and the host RAM gate (RC2: 873 s PP1 load + the
        68 GiB warning under page-cache pressure). A failed host gate never launches the server."""
        g = boot_gate()
        h0 = host_mem()
        pc = drop_page_cache()
        h1 = host_mem()
        hg = host_gate(30 if FAKE else 180)
        g.update(page_cache=pc, host_before_drop=h0, host_after_drop=h1, host_gate=hg)
        if hg.get("ok") is False:
            g["ok"], g["fail_status"] = False, "HOST_RAM_GATE"
        R["setup"].setdefault("boot_gates", {})[name] = {k: g.get(k) for k in ("ok", "fail_status", "page_cache", "host_gate")}
        save()
        return g


    # ---------------------------------------------------------------- per-boot watchdog
    def bleft():
        d = BOOT["deadline"]
        return min(left(), d - time.time()) if d else left()


    def _deadline_killer(s, key):
        def fire():
            BOOT["why"] = f"BOOT_DEADLINE {key}: the boot's {round((BOOT['deadline'] or 0) - BOOT.get('t0', 0))} s were used up"
            log("!!!", BOOT["why"])
            try:
                s.kill()
            except Exception:
                pass
        t = threading.Timer(max(1.0, (BOOT["deadline"] or time.time()) + 120 - time.time()), fire)
        t.daemon = True
        t.start()
        return t


    # ---------------------------------------------------------------- F census (fixed blocks per request)
    def _usage(port):
        m = L.prom(port)
        return m.get(L.KVU), int(m.get(L.RUN, 0)), int(m.get(L.WAIT, 0)), m.get(L.PROMPT)


    def f_census(port, n=4, max_new=1000):
        """F from kv_cache_usage deltas (Merlin keeps finished turns referenced, so only deltas over an idle baseline count):
        n requests at 2 blocks each (6,000-token prompts) and at 6 blocks each (22,000), < 1,000 new tokens so no block
        boundary is crossed. B = n x 4 / (d6 - d2) blocks, F = d2 x B / n - 2. Prompts with variants 100.. share no block
        with any earlier request."""
        rec = {"method": "kv_cache_usage delta over an idle baseline, n=4 at 2 and 6 blocks per request", "probes": {}}
        if L.wait_idle(port, 120):                      # idle usage with the retained turns, then with the caches cleared
            u_ret = _usage(port)[0]
            rs = reset_prefix_cache(port)
            time.sleep(1)
            u_clr = _usage(port)[0]
            rec["idle"] = {"usage_retained": u_ret, "reset_prefix_cache": rs, "usage_cleared": u_clr,
                           "blocks_retained_rc2_pool": round(u_ret * POOL_BLOCKS_RC2, 1) if u_ret is not None else None,
                           "blocks_cleared_rc2_pool": round(u_clr * POOL_BLOCKS_RC2, 1) if u_clr is not None else None}
        for tag, plen, blocks in (("2_blocks", 6000, 2), ("6_blocks", 22000, 6)):
            if not L.wait_idle(port, 120):
                rec["probes"][tag] = {"status": "NOT_IDLE"}
                continue
            u0 = []
            for _ in range(3):
                u0.append(_usage(port)[0])
                time.sleep(0.5)
            ss = [L.Stream(port, prompt(plen, 100 + i + (0 if blocks == 2 else 10)), max_new, temperature=TEMP, seed=777 + i,
                           ignore_eos=True) for i in range(n)]
            for g in ss:
                g.start()
            samples, prev, t = [], None, time.time()
            try:
                while time.time() - t < 420:
                    time.sleep(0.5)
                    u, run, wait, pt = _usage(port)
                    if run == n and wait == 0 and all(g.t_first for g in ss) and prev is not None and pt == prev and u is not None:
                        samples.append(u)
                    prev = pt
                    if len(samples) >= 5 or all(not g.is_alive() for g in ss):
                        break
            finally:
                for g in ss:
                    g.stop()
                for g in ss:
                    g.join(10)
            ok = len(samples) >= 2 and all(x is not None for x in u0)
            rec["probes"][tag] = {"plen": plen, "blocks_each": blocks, "u0": u0, "u_run": samples,
                                  "delta": round(statistics.mean(samples) - statistics.mean(u0), 6) if ok else None,
                                  "errors": sorted({g.err for g in ss if g.err})[:2]}
        return f_from_probes(rec, n)


    def f_from_probes(rec, n=4):
        d2 = (rec["probes"].get("2_blocks") or {}).get("delta")
        d6 = (rec["probes"].get("6_blocks") or {}).get("delta")
        if d2 is None or d6 is None or d6 - d2 <= 0:
            rec["status"] = "INCONCLUSIVE_CENSUS"
            return rec
        B = n * 4 / (d6 - d2)
        F = d2 * B / n - 2
        rec.update(pool_blocks=round(B, 1), F=round(F, 2), F_int=int(round(F)), F_with_rc2_pool=round(d2 * POOL_BLOCKS_RC2 / n - 2, 2),
                   status="MEASURED" if 60 <= B <= 600 and -0.5 <= F <= 40 else "INCONCLUSIVE_CENSUS_RANGE")
        return rec


    def reset_ok(status, body):
        """The dev endpoint answers HTTP 200 with {"success": false} while blocks are still held (running requests or
        retained turns): only success true counts (vllm/entrypoints/serve/dev/cache/api_router.py)."""
        try:
            return status == 200 and json.loads(body or "{}").get("success") is True
        except Exception:
            return False


    def reset_prefix_cache(port):
        """POST /reset_prefix_cache (vLLM dev endpoint, VLLM_SERVER_DEV_MODE=1); the result is recorded, never required."""
        try:
            r = L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/reset_prefix_cache", data=b"", method="POST"), timeout=60)
            body = r.read()[:200].decode("utf-8", "replace")
            return {"ok": reset_ok(r.status, body), "http": r.status, "body": body}
        except Exception as e:
            return {"ok": False, "error": repr(e)[:200]}


    def jit_count(s):
        t = s.log() or ""
        return {"n": t.count("JIT compilation during inference"),
                "kernels": sorted(set(re.findall(r"JIT compilation during inference: (\w+)", t)))[:20]}


    def warmup(s, cfg, passes=3, max_new=192):
        """Audit addendum 3: the kernels RC2 JIT-compiled during inference (_fill_num_accepted_kernel,
        _causal_conv1d_update_kernel, _gdn_recoverssm_verify_kernel, _ple_conv_kernel, _ple_conv_writeback_kernel) compile
        here, before any measured window: concurrent sampled decodes at 1 / 2 / 4 / 8 (/ 16) requests (staggered starts
        give mixed prefill + decode batches); a pass is repeated while it still triggered a new JIT. Metrics are deltas over
        windows, so the warm-up never enters a measurement."""
        rec = {"jit_before": jit_count(s), "passes": []}
        lv = [1, 2, 4, 8] + ([cfg["seqs"]] if cfg["seqs"] > 8 else [])
        pool = PROMPTS["2k"] + PROMPTS["8k"][:8]
        for p_i in range(passes):
            j0, t = jit_count(s)["n"], time.time()
            for n in lv:
                if bleft() < TRACE_RESERVE + WARM_MARGIN or not s.alive():
                    break
                ss = [L.Stream(s.port, pool[(i + 3 * p_i) % len(pool)], max_new, temperature=TEMP, seed=31000 + 100 * p_i + i)
                      for i in range(n)]
                for i, g in enumerate(ss):
                    g.start()
                    if i == n // 2:
                        time.sleep(0.5)
                for g in ss:
                    g.join(600)
                for g in ss:
                    if g.is_alive():
                        g.stop()
                        g.join(10)
            time.sleep(2)
            j1 = jit_count(s)
            rec["passes"].append({"sec": round(time.time() - t, 1), "new_jit": j1["n"] - j0, "kernels": j1["kernels"]})
            if j1["n"] == j0:
                break
        rec["jit_after"] = jit_count(s)
        rec["clean"] = bool(rec["passes"]) and rec["passes"][-1]["new_jit"] == 0
        return rec


    def plan_window(n, plen, F, B, secs=None, tok_s_stream=None):
        """Pre-run capacity check (memory rule): blocks if every stream grows for admission + window (WIN + 150 s) at the
        RC2-like per-stream rate min(40, max(15, 300 / n)) tok/s (RC2: 70 at 1 stream, 33 at 8)."""
        secs = secs or (WIN + 150)
        tok_s_stream = tok_s_stream or min(40.0, max(15.0, 300.0 / max(1, n)))
        per = math.ceil((plen + int(tok_s_stream * secs)) / 4096) + F
        need = n * per
        return {"n": n, "plen": plen, "F": F, "pool_blocks": B, "blocks_per_request": per, "blocks_needed": need,
                "fits": need <= B, "run": need <= 1.05 * B}


    def fit_step_model(windows):
        """RC2 analysis form: step_ms = floor + slope x r, r = requests per completed microbatch = tokens_per_step / mean
        acceptance length (1 without MTP); single-stream windows are left out (no second microbatch to interleave). Least
        squares over the boot's STEADY windows; needs two distinct r."""
        pts = []
        for w in windows.values():
            if isinstance(w, dict) and w.get("status") == "STEADY" and (w.get("running_mean") or 0) >= 2 and w.get("step_ms")                 and w.get("tokens_per_step"):
                pts.append((w["tokens_per_step"] / (w.get("mean_acceptance_length") or 1.0), w["step_ms"]))
        if len({x for x, _ in pts}) < 2:
            return {"status": "INCONCLUSIVE_FIT", "points": pts}
        n = len(pts)
        mx, my = sum(x for x, _ in pts) / n, sum(y for _, y in pts) / n
        sxx = sum((x - mx) ** 2 for x, _ in pts)
        b = sum((x - mx) * (y - my) for x, y in pts) / sxx
        a = my - b * mx
        ss = sum((y - my) ** 2 for _, y in pts)
        r2 = 1 - sum((y - (a + b * x)) ** 2 for x, y in pts) / ss if ss else None
        return {"status": "FITTED", "floor_ms": round(a, 2), "per_request_ms": round(b, 2), "r2": round(r2, 3) if r2 is not None else None,
                "points": [[round(x, 2), y] for x, y in pts], "form": "step_ms = floor + per_request_ms x r, r = tokens_per_step / mean_acceptance_length"}


    _WALL_OFF = time.time() - time.perf_counter()


    def ttft_summary(port, plen, wall_lo, wall_hi, n):
        """TTFT of the n streams of one window from traces/requests.jsonl (cost-aware admission can hold a cold miss up to
        120 s): the last n rows of this port / prompt length that ended by the window's stop + 60 s."""
        out = []
        try:
            with open(REQ_LOG) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("port") == port and r.get("prompt_len") == plen and wall_lo <= (r.get("t") or 0) <= wall_hi and r.get("ttft") is not None:
                        out.append((r["t"], r["ttft"]))
        except Exception:
            return None
        out = [x for _, x in sorted(out)[-n:]]
        return {"n": len(out), "max": max(out), "median": statistics.median(out)} if out else {"n": 0}


    # ---------------------------------------------------------------- ladder (RC2/E23 levels(), per-position added)
    def ladder(s, cfg, o):
        fc = o.get("f_census") or {}
        measured = fc.get("status") == "MEASURED"
        F = fc["F_int"] if measured else F_PLAN
        B = int(fc["pool_blocks"]) if measured else POOL_BLOCKS_RC2
        seqs_rb = (o.get("log_facts") or {}).get("max_num_seqs")
        kw = {"server": s, "temperature": TEMP, "prefill_tok_s": PREFILL["tok_s"], "seed": WINDOW_SEED}   # 1234 = steady()'s default
        out, bands = {}, {}
        for band, n in cfg["ladder"]:
            bands.setdefault(band, []).append(n)
        for band, lv in bands.items():
            run_lv = []
            for n in lv:
                key = f"{band}@{n}"
                p = plan_window(n, BANDS[band], F, B)
                if n > (seqs_rb or 0):
                    out[key] = {"status": "SKIPPED_SEQS_CAP", "server_max_num_seqs": seqs_rb, "plan": p}
                elif not p["run"]:
                    out[key] = {"status": "SKIPPED_KV_PREDICTED", "plan": p}
                    fit = max((m for m in range(9, n) if m <= (seqs_rb or 0) and plan_window(m, BANDS[band], F, B)["run"]), default=None)
                    if n > 8 and fit and fit not in lv and fit not in run_lv:     # the highest level the pool admits
                        run_lv.append(fit)
                        out[key]["substituted_level"] = fit
                else:
                    run_lv.append(n)
            if run_lv and bleft() < WIN + LADDER_MARGIN + TRACE_RESERVE:
                for n in run_lv:
                    out[f"{band}@{n}"] = {"status": "SKIPPED_BOOT_DEADLINE"}
                continue
            if run_lv:
                out.update(levels(s.port, band, run_lv, **kw))
                for n in run_lv:
                    w = out.get(f"{band}@{n}") or {}
                    if w.get("t1"):
                        end = w["t1"] + _WALL_OFF
                        w["ttft"] = ttft_summary(s.port, BANDS[band], end - 3600, end + 60, n)
            for n in run_lv:
                w = out.get(f"{band}@{n}") or {}
                w["plan"] = plan_window(n, BANDS[band], F, B)
                w["per_pos"] = window_pos(w, cfg["k"])
                w["server_max_num_seqs"] = seqs_rb
                w["real_running_eq_level"] = w.get("running_min") == w.get("running_max") == n
                tail = ((w.get("admission") or {}).get("tail") or [[None] * 5])[-1]
                w["all_prefilled_at_start"] = len(tail) >= 4 and tail[3] == n and tail[2] == 0   # [t, run, wait, first, gaining]
                if w.get("step_ms"):
                    w["tick_ms"] = w["step_ms"]                       # one PP tick = one completed micro-batch (~C/2 requests)
                    w["cycle_ms"] = round(2 * w["step_ms"], 1)        # PP2: a request advances once per 2 ticks
                w["microbatch"] = microbatch_cover(o.get("graph_plan") or {}, n, cfg["k"])
                if w.get("steps") and w.get("window_sec"):
                    w["completed_microbatch_rate_per_s"] = round(w["steps"] / w["window_sec"], 2)   # iteration histogram
                w["wall_aggregate_tok_s"] = w.get("tok_s")                  # generation tokens / wall window (all streams)
                w["phase"] = ("early_reasoning: the first ~" + str(int((w.get("gen_tokens") or 0) / max(1, n))) +
                              " generated tokens per stream (planning scenario a = 2.72 is the long-trace value)")
                tps = w.get("tokens_per_step")
                if tps and w.get("running_mean"):
                    a_out = tps * 2 / w["running_mean"]                   # output tokens per request per verify cycle
                    w["a_output_per_cycle"] = round(a_out, 3)
                    w["accepted_per_verify_token"] = round(a_out / (cfg["k"] + 1), 3)   # MoE weight reads are per verify token
                w["running_blocks_est"] = n * (math.ceil((BANDS[band] + (w.get("gen_tokens") or 0) / max(1, n)) / 4096) + F)
                w["kv_usage_note"] = "kv_cache_usage includes retained finished sessions: running_blocks_est is the live demand"
                a_len = w.get("mean_acceptance_length") or (1.0 if not cfg["k"] else None)
                if w.get("running_mean") and a_len:
                    exp = w["running_mean"] / 2 * a_len
                    w["tokens_per_step_pp2_expected"] = round(exp, 2)              # PP2: ~half the requests per iteration
                    w["tokens_per_step_ratio"] = round(w["tokens_per_step"] / exp, 3) if w.get("tokens_per_step") else None
                if n > 8 and cfg["seqs"] > 8 and w.get("running_max") is not None:
                    if w["running_max"] > 8:
                        SEQS16.update(accepted=True, why=f"boot {cfg['key']} {band}@{n}: {w['running_max']} running at once")
                    elif w.get("waiting_max", 0) > 0 and w["plan"]["fits"] and SEQS16["accepted"] is not True:
                        SEQS16.update(accepted=False, why=f"boot {cfg['key']} {band}@{n}: <= 8 running with waiters and KV room")
        return out


    def admission_check(s, cfg, o, n=16, band="8k", max_new=600, timeout=240):
        """SMOKE: n distinct 8k prompts at once; does the server admit all n (real running == n with every stream gaining
        its first token)? No measured window; the streams stop as soon as the answer is known (or at the timeout)."""
        seqs_rb = (o.get("log_facts") or {}).get("max_num_seqs")
        p = plan_window(n, BANDS[band], F_PLAN, POOL_BLOCKS_RC2, secs=max_new, tok_s_stream=1.0)    # each stream ends at max_new
        rec = {"level": n, "band": band, "max_new": max_new, "server_max_num_seqs": seqs_rb, "plan": p}
        if n > (seqs_rb or 0):
            return dict(rec, status="SKIPPED_SEQS_CAP")
        if not p["run"]:
            return dict(rec, status="SKIPPED_KV_PREDICTED")
        if not L.wait_idle(s.port, 120):
            return dict(rec, status="INCONCLUSIVE_NOT_IDLE")
        m0 = L.prom(s.port)
        ss = [L.Stream(s.port, PROMPTS[band][i % len(PROMPTS[band])], max_new, temperature=TEMP, seed=4400 + i, ignore_eos=True)
              for i in range(n)]
        t0 = time.perf_counter()
        runs, waits, t_all = [], [], None
        for g in ss:
            g.start()
        try:
            while time.perf_counter() - t0 < timeout and s.alive():
                time.sleep(1)
                m = L.prom(s.port)
                runs.append(int(m.get(L.RUN, 0)))
                waits.append(int(m.get(L.WAIT, 0)))
                if runs[-1] >= n and all(g.t_first for g in ss):
                    t_all = time.perf_counter() - t0
                    break
                if all(not g.is_alive() for g in ss):
                    break
            m1 = L.prom(s.port)
        finally:
            for g in ss:
                g.stop()
            for g in ss:
                g.join(10)
        rec.update(running_max=max(runs) if runs else None, waiting_last=waits[-1] if waits else None,
                   sec_to_all_admitted=round(t_all, 1) if t_all is not None else None,
                   preemptions=(m1.get(L.PREEMPT, 0) - m0.get(L.PREEMPT, 0)), errors=sorted({g.err for g in ss if g.err})[:3],
                   first_token_streams=sum(1 for g in ss if g.t_first), alive_after=s.alive(),
                   status="ADMITTED_ALL" if t_all is not None and not any(g.err for g in ss) else "NOT_ALL_ADMITTED")
        if (rec["running_max"] or 0) > 8:
            SEQS16.update(accepted=True, why=f"boot {cfg['key']} admission check: {rec['running_max']} running at once")
        return rec


    # ---------------------------------------------------------------- decay window (acceptance per 10-min sub-window)
    def decay_window(port, cfg, minutes, sub_min=None, level=DECAY_LEVEL):
        sub_min = sub_min or DECAY_SUB_MIN
        rec = {"level": level, "minutes": round(minutes, 2), "sub_min": sub_min, "prompts": "real single-task xhigh ARC prompts",
               "temperature": TEMP, "windows": [], "refills": 0, "finished": []}
        if not L.wait_idle(port, 120):
            rec["status"] = "INCONCLUSIVE_NOT_IDLE"
            return rec
        nxt = [0]

        def new_stream():
            j = nxt[0] % len(DECAY_P)
            nxt[0] += 1
            g = L.Stream(port, DECAY_P[j], MAX_LEN - len(DECAY_P[j]) - 512, temperature=TEMP, seed=9000 + nxt[0], ignore_eos=False)
            g.start()
            return [j, g]
        live = [new_stream() for _ in range(level)]
        done_rows = []
        try:
            t_adm = time.perf_counter()
            while time.perf_counter() - t_adm < 600 and not all(g.t_first for _, g in live):
                time.sleep(2)
            rec["admission_sec"] = round(time.perf_counter() - t_adm, 1)
            t_end = time.perf_counter() + minutes * 60
            while time.perf_counter() < t_end - 5 and not ABORT["why"]:
                w_end = min(t_end, time.perf_counter() + sub_min * 60)
                m0 = L.prom(port)
                t0 = time.perf_counter()
                runs, waits, refills = [], [], 0
                len0 = statistics.mean(len(DECAY_P[j]) + len(g.tokens) for j, g in live)
                while time.perf_counter() < w_end:
                    time.sleep(2)
                    try:
                        m = L.prom(port)
                        runs.append(int(m.get(L.RUN, 0)))
                        waits.append(int(m.get(L.WAIT, 0)))
                    except Exception:
                        pass
                    for slot in live:
                        if not slot[1].is_alive():
                            r = slot[1].row()
                            done_rows.append((slot[0], r))
                            rec["finished"].append({"prompt": slot[0], "tokens": len(r["tokens"]), "finish": r["finish_reason"],
                                                    "error": r["error"], "think_closed": "</think>" in r["text"],
                                                    "quality": long_quality("natural", {"tokens": len(r["tokens"]), "error": r["error"],
                                                                                        "finish": r["finish_reason"], "think_closed": "</think>" in r["text"],
                                                                                        "window_garbage": window_garbage(r["tokens"]),
                                                                                        "cut_by_cap": r["finish_reason"] is None})})
                            slot[:] = new_stream()
                            refills += 1
                    if bleft() < TRACE_RESERVE or ABORT["why"]:
                        break
                m1 = L.prom(port)
                t1 = time.perf_counter()
                dt = t1 - t0
                gen, acc, drf, pre, pt, it = (m1.get(x, 0) - m0.get(x, 0) for x in (L.GEN, L.ACC, L.DRF, L.PREEMPT, L.PROMPT, L.ITER))
                pp = window_pos({"port": port, "t0": t0, "t1": t1}, cfg["k"])
                w = {"i": len(rec["windows"]), "sec": round(dt, 1), "tok_s": round(gen / dt, 1) if dt else None,
                     "accept_per_draft": round(acc / drf, 4) if drf else None, "per_position": pp.get("per_position"),
                     "per_pos_status": pp.get("status"), "running_min": min(runs) if runs else None,
                     "running_max": max(runs) if runs else None, "waiting_max": max(waits) if waits else None,
                     "preemptions": pre, "prompt_tokens": pt, "refills": refills, "mean_context_start": round(len0),
                     "tick_ms": round(dt * 1000 / it, 1) if it else None, "cycle_ms": round(2 * dt * 1000 / it, 1) if it else None,
                     "tokens_per_tick": round(gen / it, 2) if it else None,
                     "completed_microbatch_rate_per_s": round(it / dt, 2) if it and dt else None,
                     "mean_generated_per_stream_start": round(len0 - statistics.mean(len(DECAY_P[j]) for j, _ in live)),
                     "phase": "long_xhigh_trace (natural xhigh rows, refilled on finish)"}
                w["running_mean"] = round(statistics.mean(runs), 2) if runs else None
                if it and w["running_mean"]:
                    w["a_output_per_cycle"] = round(gen / it * 2 / w["running_mean"], 3)
                    w["accepted_per_verify_token"] = round(w["a_output_per_cycle"] / (cfg["k"] + 1), 3)
                # valid: the level held on average (a natural finish is refilled within one 2-s poll) and no preemption
                w["valid"] = bool(runs) and w["running_mean"] >= level - 0.5 and not pre
                rec["windows"].append(w)
                rec["refills"] += refills
                if bleft() < TRACE_RESERVE:
                    rec["cut"] = "boot deadline (trace reserve)"
                    break
        finally:
            for _, g in live:
                g.stop()
            for _, g in live:
                g.join(10)
            done_rows += [(j, g.row()) for j, g in live]
        gb = []
        for j, r in done_rows:
            wg = window_garbage(r["tokens"]) if r["tokens"] else None
            if wg:
                gb.append({"prompt": j, **wg})
        rec["streams_total"] = len(done_rows)
        rec["garbage"] = gb
        rec.update(decay_verdict(rec["windows"]))
        return rec


    def decay_verdict(windows):
        ap = [w["accept_per_draft"] for w in windows if w.get("valid") and w.get("accept_per_draft") is not None]
        if len(ap) < 2:
            return {"status": "INCONCLUSIVE_DECAY_WINDOWS", "valid_windows": len(ap)}
        ratio = ap[-1] / ap[0] if ap[0] else None
        st = "COLLAPSE" if min(ap) < 0.3 else ("DECAY_SUSPECT" if ratio is not None and ratio < 0.85 else "STABLE")
        return {"status": st, "valid_windows": len(ap), "accept_first": ap[0], "accept_last": ap[-1],
                "last_over_first": round(ratio, 3) if ratio else None}


    COHORT_N, COHORT_NEW = 5, 1536


    def cohort_replay(port, cfg, tag):
        """Audit 2 item 3: the SAME frozen cohort (5 real xhigh ARC prompts, fixed seeds, 1,536 tokens each, ignore_eos) at
        the same concurrency, replayed at boot start and again after the ladder / long window: acceptance is compared on
        identical work, so a drop is uptime decay, not context length or concurrency. The window runs from 'all prefilled'
        to the first stream's end (constant concurrency)."""
        rec = {"tag": tag, "n": COHORT_N, "max_new": COHORT_NEW}
        if not cfg["k"]:
            return dict(rec, status="N/A_NO_MTP")
        if not L.wait_idle(port, 120):
            return dict(rec, status="INCONCLUSIVE_NOT_IDLE")
        ss = [L.Stream(port, DECAY_P[i], COHORT_NEW, temperature=TEMP, seed=5000 + i, ignore_eos=True) for i in range(COHORT_N)]
        for g in ss:
            g.start()
        t0 = t1 = None
        try:
            t_s, prev = time.perf_counter(), None
            while time.perf_counter() - t_s < 600:
                time.sleep(1)
                m = L.prom(port)
                if t0 is None:
                    if all(g.t_first for g in ss) and int(m.get(L.RUN, 0)) == COHORT_N and m.get(L.PROMPT) == prev:
                        t0 = time.perf_counter()
                    prev = m.get(L.PROMPT)
                elif any(not g.is_alive() for g in ss):
                    break
            L.prom(port)
            t1 = time.perf_counter()
        finally:
            for g in ss:
                g.stop()
            for g in ss:
                g.join(10)
        if t0 is None:
            return dict(rec, status="INCONCLUSIVE_NO_SETTLE")
        pp = window_pos({"port": port, "t0": t0, "t1": t1}, cfg["k"])
        with SNAP["lock"]:
            lst = list(SNAP["by_port"].get(int(port), []))
        a = max((s for s in lst if s[0] <= t0), key=lambda s: s[0], default=None)
        b = max((s for s in lst if s[0] <= t1), key=lambda s: s[0], default=None)
        gen = ((b[1].get(L.GEN) or 0) - (a[1].get(L.GEN) or 0)) if a and b else None
        return dict(rec, status=pp.get("status"), sec=round(t1 - t0, 1), per_position=pp.get("per_position"),
                    accept_per_draft=pp.get("accept_per_draft"), tok_s=round(gen / (t1 - t0), 1) if gen and t1 > t0 else None,
                    errors=sorted({g.err for g in ss if g.err})[:2], t_boot_min=round((time.time() - BOOT.get("t0", time.time())) / 60, 1))


    def cohort_verdict(reps):
        ok = [r for r in reps if r.get("status") == "MEASURED" and r.get("per_position")]
        if len(ok) < 2:
            return {"status": "INCONCLUSIVE_COHORT", "measured": len(ok)}
        a, b = ok[0], ok[-1]
        ratio = [round(y / x, 3) if x else None for x, y in zip(a["per_position"], b["per_position"])]
        worst = min([r for r in ratio if r is not None] or [1.0])
        return {"status": "UPTIME_DECAY_SUSPECT" if worst < 0.9 else "STABLE", "first": a["per_position"], "last": b["per_position"],
                "last_over_first_per_position": ratio, "minutes_apart": round(b["t_boot_min"] - a["t_boot_min"], 1)}


    _long_quality_rc2 = long_quality


    def long_quality(name, r):
        """Audit 2 item 5: a natural row passes only with finish == 'stop' and a closed think block (RC2 rule plus)."""
        q = _long_quality_rc2(name, r)
        if q == "PASS_LONG" and name == "natural" and not (isinstance(r, dict) and r.get("finish") == "stop" and r.get("think_closed") is True):
            return "INCOMPLETE_LONG"
        return q


    DRAFT_KV_CANDIDATE = ("candidate, no arm: speculative_config.kv_cache_dtype exists (vllm/config/speculative.py:430, 'when None, "
                          "the draft inherits --kv-cache-dtype'); the MTP layer's FP8 QSA KV runs with DEFAULT scale 1.0 (RC2 log "
                          "318-319), so a bf16 draft KV is a possible acceptance lever")


    # ---------------------------------------------------------------- trace window: PP2 overlap, verify rows, graph modes
    NB2_CATS = [("nccl", ("nccl", "allreduce", "all_reduce", "allgather", "all_gather", "sendrecv", "reduce_scatter")),
                ("recoverssm_commit", ("commit", "fold", "compact", "align_post", "recoverssm_plan")),
                ("qsa", ("qsa", "sparse_attn", "indexer")),
                ("persistent_topk", ("persistent_topk",)),
                ("moe_router", ("topk_softmax", "moe_align", "align_block_size")),
                ("sampling", ("sampl", "rejection", "argmax", "softmax", "topk", "top_k", "top_p")),
                ("gdn", ("gdn", "recoverssm", "delta_rule", "chunk_", "recurrent", "causal_conv", "l2norm", "solve_tril",
                         "merge_16x16", "recompute_w_u", "kkt", "cumsum", "post_conv", "layer_norm_gated", "gated_delta")),
                ("marlin_moe", ("marlin_moe", "moe_wna16", "fused_moe", "moe_marlin")),
                ("marlin_dense", ("marlin",)),
                ("attention", ("flashinfer", "flash_fwd", "unified_attention", "paged", "attn", "attention", "reshape_and_cache")),
                ("moe_other", ("moe", "expert")),
                ("gemm_other", ("gemm", "cutlass", "xmma", "cublas", "gemv", "sm89_", "ampere_", "sm80_")),
                ("ple", ("ple", "ngram", "engram"))]
    NB2_COMPUTE = ("nccl", "qsa", "gdn", "marlin_moe", "marlin_dense", "attention", "gemm_other", "moe_other")
    SCOPE_NB2 = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$|"
                           r"^execute_\d+_context_(\d+)\(sq(\d+)[^)]*\)_generation_(\d+)\(sq(\d+)[^)]*\)$")


    def nb2_cat(name):
        s = re.sub(r"\(anonymous namespace\)::", "", str(name))
        s = re.sub(r"^\s*void\s+", "", s)
        base = s.split("<")[0].split("(")[0].lower()
        return next((k for k, pats in NB2_CATS if any(p in base for p in pats)), "other")


    def _union(iv):
        out = []
        for a, b in sorted(iv):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return out


    def _clip(iv, lo, hi):
        return [[max(a, lo), min(b, hi)] for a, b in iv if b > lo and a < hi]


    def _ilen(iv):
        return sum(b - a for a, b in iv)


    def _inter(x, y):
        i = j = 0
        out = []
        while i < len(x) and j < len(y):
            a, b = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
            if b > a:
                out.append([a, b])
            if x[i][1] < y[j][1]:
                i += 1
            else:
                j += 1
        return out


    def _rank_trace(f):
        with (gzip.open if f.name.endswith(".gz") else open)(f, "rt", encoding="utf-8") as fh:
            tr = json.load(fh)
        base = tr.get("baseTimeNanoseconds")
        off = base / 1000.0 if isinstance(base, (int, float)) else 0.0
        ev = tr.get("traceEvents", [])
        kern = [e for e in ev if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("kernel", "gpu_kernel")]
        rt = [e for e in ev if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("cuda_runtime", "cuda_driver")]
        scopes = []
        for e in ev:
            if e.get("ph") == "X" and str(e.get("cat", "")) == "user_annotation":
                m = SCOPE_NB2.match(str(e.get("name", "")))
                if m:
                    g = [int(x) for x in m.groups() if x is not None]
                    scopes.append((e, g))
        scopes.sort(key=lambda x: x[0].get("ts", 0))
        corr_launch = {(e.get("args") or {}).get("correlation"): str(e.get("name", "")) for e in rt
                       if (e.get("args") or {}).get("correlation") is not None}
        kname = {(e.get("args") or {}).get("correlation"): str(e.get("name", "")) for e in kern
                 if (e.get("args") or {}).get("correlation") is not None}
        busy = _union([[e["ts"] + off, e["ts"] + off + e.get("dur", 0)] for e in kern if "ts" in e])
        cat_us, cat_src, names = {}, {}, {}
        for e in kern:
            c = nb2_cat(e.get("name", ""))
            cat_us[c] = cat_us.get(c, 0.0) + float(e.get("dur", 0))
            ln = corr_launch.get((e.get("args") or {}).get("correlation")) or ""
            src = "graph" if "GraphLaunch" in ln else ("eager" if ln else "unknown")
            cat_src.setdefault(c, {"graph": 0, "eager": 0, "unknown": 0})[src] += 1
            names[e.get("name", "")] = names.get(e.get("name", ""), 0.0) + float(e.get("dur", 0))
        per_scope = []
        for a, g in scopes[:600]:
            lo, hi, tid = a.get("ts", 0), a.get("ts", 0) + a.get("dur", 0), a.get("tid")
            inside = [e for e in rt if e.get("tid") == tid and lo <= e.get("ts", -1) <= hi]
            gl = sum(1 for e in inside if "GraphLaunch" in str(e.get("name", "")))
            eager = {}
            for e in inside:
                if re.search(r"LaunchKernel|cuLaunchKernel", str(e.get("name", ""))):
                    c = nb2_cat(kname.get((e.get("args") or {}).get("correlation"), "unresolved"))
                    eager[c] = eager.get(c, 0) + 1
            per_scope.append({"ctx_reqs": g[0], "ctx_tok": g[1], "gen_reqs": g[2], "gen_tok": g[3], "graph_launches": gl,
                              "eager": eager, "dur_us": a.get("dur", 0)})
        tot = sum(cat_us.values()) or 1.0
        return {"file": f.name, "busy": busy, "span": [busy[0][0], busy[-1][1]] if busy else None, "scopes": per_scope,
                "by_category_pct": {k: round(100 * v / tot, 1) for k, v in sorted(cat_us.items(), key=lambda x: -x[1])},
                "launch_source": cat_src, "top_kernels_us": dict(sorted(((k[:90], round(v)) for k, v in names.items()),
                                                                        key=lambda x: -x[1])[:15])}


    def _mode(scopes):
        """Graph mode of a stage over its decode-only scopes: FULL (graph launches, no eager model compute), PIECEWISE
        (graph + eager compute), EAGER (no graph launch)."""
        dec = [s for s in scopes if s["ctx_reqs"] == 0 and s["gen_reqs"] > 0]
        if not dec:
            return {"mode": "NO_DECODE_SCOPES"}
        kinds = []
        for s in dec:
            ce = sum(v for c, v in s["eager"].items() if c in NB2_COMPUTE)
            kinds.append("EAGER" if s["graph_launches"] == 0 else ("PIECEWISE" if ce else "FULL"))
        ks = set(kinds)
        return {"mode": ks.pop() if len(ks) == 1 else "MIXED", "counts": {k: kinds.count(k) for k in ("FULL", "PIECEWISE", "EAGER")},
                "median_graph_launches": statistics.median(s["graph_launches"] for s in dec),
                "commit_eager_per_scope": statistics.median(s["eager"].get("recoverssm_commit", 0) for s in dec)}


    def nb2_trace(pdir, k=3):
        files = sorted(Path(pdir).rglob("*.pt.trace.json*"))
        if not files:
            return {"status": "NO_TRACE_FILE", "listing": [str(p) for p in Path(pdir).rglob("*")][:10]}
        ranks, out = {}, {"files": [[f.name, f.stat().st_size] for f in files]}
        for f in files:
            m = re.search(r"pp(\d+)_tp(\d+)", f.name)
            if not m:
                continue
            key = (int(m.group(1)), int(m.group(2)))
            if key not in ranks:
                try:
                    ranks[key] = _rank_trace(f)
                except Exception as e:
                    out.setdefault("errors", []).append(f"{f.name}: {type(e).__name__}: {e}"[:200])
        out["ranks"] = {f"pp{a}_tp{b}": {x: v[x] for x in ("file", "by_category_pct", "launch_source", "top_kernels_us")} | {
            "graph": _mode(v["scopes"]), "scopes": len(v["scopes"])} for (a, b), v in sorted(ranks.items())}
        s0 = ranks.get((0, 0)) or next((v for (a, _), v in ranks.items() if a == 0), None)
        s1 = ranks.get((1, 0)) or next((v for (a, _), v in ranks.items() if a == 1), None)
        if s0 and s1 and s0["span"] and s1["span"]:
            lo, hi = max(s0["span"][0], s1["span"][0]), min(s0["span"][1], s1["span"][1])
            if hi > lo:
                b0, b1 = _clip(s0["busy"], lo, hi), _clip(s1["busy"], lo, hi)
                win, i01 = hi - lo, _ilen(_inter(b0, b1))
                out["pp2"] = {"window_us": round(win), "stage0_busy_pct": round(100 * _ilen(b0) / win, 1),
                              "stage1_busy_pct": round(100 * _ilen(b1) / win, 1), "both_busy_pct": round(100 * i01 / win, 1),
                              "both_idle_pct": round(100 * (win - _ilen(_union(b0 + b1))) / win, 1),
                              "overlap_of_stage1_pct": round(100 * i01 / max(1e-9, _ilen(b1)), 1),
                              "note": "per-process clocks aligned through baseTimeNanoseconds; ms-level skew possible"}
            else:
                out["pp2"] = {"status": "NO_COMMON_WINDOW"}
        src = s0 or s1
        if src:
            dec = [s for s in src["scopes"] if s["ctx_reqs"] == 0 and s["gen_reqs"] > 0]
            hist = {}
            for s in dec:
                hist[s["gen_reqs"]] = hist.get(s["gen_reqs"], 0) + 1
            out["verify_rows"] = {"decode_scopes": len(dec), "mixed_scopes": sum(1 for s in src["scopes"] if s["ctx_reqs"] > 0),
                                  "gen_reqs_hist": dict(sorted(hist.items())),
                                  "mean_gen_reqs": round(statistics.mean(s["gen_reqs"] for s in dec), 2) if dec else None,
                                  "tokens_per_req": sorted({round(s["gen_tok"] / s["gen_reqs"], 2) for s in dec})[:6],
                                  "expected_tokens_per_req": k + 1}
        if s0 and s1:
            g0, g1 = _mode(s0["scopes"]), _mode(s1["scopes"])
            out["graph_modes"] = {"target_stage0": g0, "stage1_target_plus_draft": g1,
                                  "draft_graph_launches_est": (g1.get("median_graph_launches") or 0) - (g0.get("median_graph_launches") or 0)
                                  if "median_graph_launches" in g0 and "median_graph_launches" in g1 else None,
                                  "recoverssm_commit": "EAGER" if max(g0.get("commit_eager_per_scope") or 0,
                                                                      g1.get("commit_eager_per_scope") or 0) > 0 else "NOT_EAGER_OR_NOT_SEEN"}
        return out


    def trace_window(s, cfg):
        if cfg["exact_rc2"]:
            return {"status": "SKIPPED_CTRL_HAS_NO_PROFILER_FLAG"}
        pdir = Path(cfg["prof_dir"])
        rec = L.profiled_load(s.port, PROMPTS["8k"][:8], 8, str(pdir), secs=6.0, max_new=MAX_NEW["8k"], spec=bool(cfg["k"]),
                              prefill_tok_s=PREFILL["tok_s"])
        try:
            rec["nb2"] = nb2_trace(pdir, cfg["k"])
        except Exception as e:
            rec["nb2"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        try:
            files = sorted(pdir.rglob("*.pt.trace.json*"))
            if files and sum(f.stat().st_size for f in files) < 200 << 20:
                dst = OUT / "profiles" / cfg["key"]
                dst.mkdir(parents=True, exist_ok=True)
                for f in files:
                    shutil.copyfile(f, dst / f.name)
                rec["copied_to"] = str(dst)
        except Exception as e:
            rec["copy_error"] = repr(e)[:200]
        return rec


    # ---------------------------------------------------------------- one boot
    def nb2_body(cfg):
        def body(ss):
            s = ss[0]
            p = s.port
            killer = _deadline_killer(s, cfg["key"])
            try:
                facts = log_facts(s)
                o = {"census": kv_census(s), "log_facts": facts, "graph_plan": graph_plan(cfg["captures"], cfg["k"], cfg["seqs"]),
                     "probes": L.chat_probes(p), "sampled": sampled_check(s)}
                o["config_readback"] = readback(facts, cfg)
                o["graph_coverage"] = {f"{b}@{n}": microbatch_cover(o["graph_plan"], n, cfg["k"]) for b, n in cfg["ladder"]}
                o["graph_memory_gib"] = facts.get("graph_capture")
                if short_gate_failed(o):
                    o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s))
                    return o
                if o["config_readback"]["ok"] is not True:
                    o.update(early_stop="CONFIG_READBACK_MISMATCH", status="NO_GO_CONFIG_READBACK", census_end=kv_census(s))
                    return o
                if not PREFILL["measured"]:
                    o["prefill"] = prefill_probe(p)
                o["oracle"] = oracle(p, key=cfg["key"], refs=cfg["oracle_refs"])
                o["oracle_basis"] = list(cfg["oracle_refs"]) or "SELF_REFERENCE (no earlier reference: this boot's rows are it)"
                og = evidence_gate(o, 4) or oracle_gate(o, need=cfg["oracle_refs"])
                if og:
                    o.update(early_stop="ORACLE_GATE " + og, census_end=kv_census(s))
                    return o
                o["warmup"] = warmup(s, cfg, passes=1 if cfg.get("smoke") else 3)
                o["mtp_notes"] = {"mtp_kv_default_scale_1": facts.get("mtp_kv_default_scale_1"),
                                  "mtp_kv_scale_lines": facts.get("mtp_kv_scale_lines"),
                                  "single_mtp_layer_reused_for_k_gt_1": facts.get("single_mtp_layer_warning"),
                                  "draft_kv_dtype_option": DRAFT_KV_CANDIDATE,
                                  "recoverssm_commit_graph_env": cfg.get("recoverssm_commit_graph_env"),
                                  "recoverssm_commit_log_lines": facts.get("commit_graph_lines"),
                                  "sampling": "every request passes temperature / top_p / top_k / seed (server default T 1.0 never used)"}
                if cfg.get("smoke"):           # SMOKE: dev-mode endpoint, one 8k@8 window, one 8k@16 admission check, trace
                    if cfg["exact_rc2"]:       # SMOKE_CTRL: the RC2 config has no VLLM_SERVER_DEV_MODE (no reset endpoint)
                        o["dev_mode_reset"] = {"ok": None, "status": "N/A_EXACT_RC2_NO_DEV_MODE"}
                    else:
                        o["dev_mode_reset"] = reset_prefix_cache(p) if L.wait_idle(p, 60) else {"ok": None, "why": "not idle"}
                    o["jit"] = {"after_warmup": jit_count(s)["n"]}
                    t_l0 = time.time()
                    if DIAG:                   # N2-4: the window streams ask for the top-5 logprobs (sampling unchanged: DIAG_LP)
                        DIAG_LP["port"] = p
                        o["diag_logprobs"] = {"requested": DIAG_LP["n"], "scope": "the 8k@8 window streams only",
                                              "sampling_unchanged_basis": DIAG_LP["basis"]}
                    try:
                        o["steady"] = ladder(s, cfg, o)
                    finally:
                        DIAG_LP["port"] = None
                    if DIAG:                   # every stream's verdict (the window is never stopped by garbage)
                        o["diag_streams"] = diag_stream_table(p, 8, WINDOW_SEED, BANDS["8k"], MAX_NEW["8k"], t_l0 - 5, time.time() + 5)
                        log("NB2 DIAG", cfg["key"], "seed", WINDOW_SEED, "| loops", o["diag_streams"].get("loop_count"),
                            o["diag_streams"].get("looped_streams"), "| window", ((o["steady"] or {}).get("8k@8") or {}).get("status"))
                    save()
                    o["admission_16"] = (admission_check(s, cfg, o, n=16) if bleft() > TRACE_RESERVE + 120
                                         else {"status": "SKIPPED_BOOT_DEADLINE"})
                    o["decay"] = {"status": "N/A_SMOKE"}
                    o["trace"] = trace_window(s, cfg) if bleft() > 150 else {"status": "SKIPPED_BOOT_DEADLINE"}
                    lf = log_facts(s)
                    o["log_facts_end"] = {k: lf.get(k) for k in ("spec_log", "jit_during_inference", "jit_kernels", "captures_done")}
                    o["census_end"] = kv_census(s)
                    return o
                o["f_census"] = f_census(p)
                save()
                o["cohort"] = [cohort_replay(p, cfg, "start")] if cfg["k"] and not cfg["exact_rc2"] else []
                o["jit"] = {"after_warmup": jit_count(s)["n"]}
                o["steady"] = ladder(s, cfg, o)
                o["step_model"] = fit_step_model(o["steady"])
                if o["cohort"] and bleft() > TRACE_RESERVE + 600:
                    o["cohort"].append(cohort_replay(p, cfg, "after_ladder"))
                o["jit"]["after_ladder"] = jit_count(s)["n"]
                o["jit"]["in_ladder"] = o["jit"]["after_ladder"] - o["jit"]["after_warmup"]
                save()
                if cfg["k"] and not cfg["exact_rc2"] and cfg.get("long_window"):     # A-7: the long window runs in boot A only
                    mins = min(DECAY_MIN, (bleft() - TRACE_RESERVE - 120) / 60)
                    o["decay"] = (decay_window(p, cfg, mins) if mins >= min(DECAY_MIN, 2 * DECAY_SUB_MIN) * 0.5
                                  else {"status": "SKIPPED_BOOT_DEADLINE", "minutes_left": round(mins, 1)})
                else:
                    o["decay"] = {"status": "N/A_NO_MTP" if not cfg["k"] else ("N/A_CTRL_ONE_WINDOW" if cfg["exact_rc2"]
                                                                              else "N/A_NOT_THE_LONG_WINDOW_BOOT")}
                save()
                if o["cohort"] and (o.get("decay") or {}).get("windows") and bleft() > TRACE_RESERVE + 300:
                    o["cohort"].append(cohort_replay(p, cfg, "after_long_window"))
                o["cohort_verdict"] = cohort_verdict(o["cohort"]) if o["cohort"] else {"status": "N/A"}
                o["jit"]["after_decay"] = jit_count(s)["n"]
                o["trace"] = trace_window(s, cfg) if bleft() > 150 else {"status": "SKIPPED_BOOT_DEADLINE"}
                lf = log_facts(s)
                o["log_facts_end"] = {k: lf.get(k) for k in ("spec_log", "jit_during_inference", "jit_kernels", "commit_graph_lines",
                                                             "recoverssm_path_lines", "captures_done")}
                o["census_end"] = kv_census(s)
                return o
            finally:
                killer.cancel()
        return body


    def nb2_arm(bk, seqs=None):
        b = BOOTS[bk]
        cfg = make_cfg(b, seqs)
        cfg["oracle_refs"] = tuple(k for k in ("rc2_fp8", "A") if k in ROWS and k != cfg["key"])
        BOOT.update(deadline=time.time() + b["est_min"] * 60, why=None, key=cfg["key"], t0=time.time())
        try:
            o = merlin_boot(cfg["name"], "fp8", cfg["port"], READY_SEC, nb2_body(cfg), need=cfg["oracle_refs"], cfg=cfg)
        finally:
            why = BOOT["why"]
            BOOT.update(deadline=None, why=None, key=None)
        o["cfg"] = cfg
        if why:
            o["boot_deadline"] = why
            if o.get("status") in ("DONE", "SERVER_DIED", "BODY_ERROR"):
                o["status"] = "BOOT_DEADLINE"
        dec = o.get("decay") or {}
        if dec.get("garbage") and o.get("status") == "DONE":
            o["status"] = "NO_GO_DECAY_GARBAGE"
        elif dec.get("status") == "COLLAPSE" and o.get("status") == "DONE":
            o["status"] = "NO_GO_ACCEPTANCE_COLLAPSE"
        o["seqs16"] = dict(SEQS16)
        return o


    # ---------------------------------------------------------------- lever scorecard (PLAN 4.1), written after every boot
    def _win(rec, key):
        w = ((rec or {}).get("steady") or {}).get(key) or {}
        return w if w.get("status") == "STEADY" else None


    def steady_formula(w):
        """PLAN section 1: steady = C x a / t_cycle. Audit 2: a = OUTPUT (generation) tokens per request per verify cycle,
        never the engine's accepted-draft count: a = tokens_per_step (generation tokens / engine iterations) x 2 / C, since
        under PP2 a request advances once per 2 iterations (t_cycle = 2 x step_ms). steady_tok_s is the measured generation
        rate; the formula is shown as its decomposition, with the engine's own mean acceptance length beside it."""
        if not w:
            return None
        C, st, tps = w.get("running_mean"), w.get("step_ms"), w.get("tokens_per_step")
        if not (C and st and tps):
            return None
        a = round(tps * 2 / C, 3)
        return {"C": C, "a_output_tokens_per_cycle": a, "tick_ms": st, "cycle_ms": round(2 * st, 1),
                "accepted_per_verify_token": w.get("accepted_per_verify_token"),
                "formula_tok_s": round(C * a / (2 * st / 1000.0), 1), "steady_tok_s": w.get("tok_s"),
                "engine_mean_acceptance_length": w.get("mean_acceptance_length")}


    def _correct(rec):
        if not rec:
            return "UNTESTED"
        st, c = str(rec.get("status")), rec.get("correctness")
        if st.startswith("NO_GO") or str(c).startswith("NO_GO"):
            return "FAIL"
        return "PASS" if st == "DONE" and c == "CLEAN" else "UNTESTED"


    CONTROL_CFG = {"key": "rc2_control", "k": 3, "head": "bf16", "seqs": 8, "captures": SHIPPED_CAPTURES}


    def comparison_label(cfg, base):
        """A-7: what differs between an arm and its base. K_ONLY / HEAD_ONLY = a clean one-knob pair; a K change together
        with seqs / captures = K_CHANGE_PLUS_GRAPH_CONFIG; any other mix = UNMATCHED_CONFIG."""
        cfg, base = cfg or {}, base or {}
        norm = lambda c, x: sorted(c.get(x) or []) if x == "captures" else c.get(x)
        diffs = [x for x in ("k", "head", "seqs", "captures") if norm(cfg, x) != norm(base, x)]
        graph = bool({"seqs", "captures"} & set(diffs))
        if not diffs:
            lab = "SAME_CONFIG"
        elif diffs == ["k"]:
            lab = "K_ONLY"
        elif diffs == ["head"]:
            lab = "HEAD_ONLY"
        elif "k" in diffs and graph and "head" not in diffs:
            lab = "K_CHANGE_PLUS_GRAPH_CONFIG"
        else:
            lab = "UNMATCHED_CONFIG"
        return {"label": lab, "differs_in": diffs, "base": base.get("key"),
                "arm": {x: cfg.get(x) for x in ("k", "head", "seqs")}, "base_cfg": {x: base.get(x) for x in ("k", "head", "seqs")}}


    def score_row(lever, rec, base_windows, base_label, adoption, notes="", base_cfg=None):
        cfg = (rec or {}).get("cfg") or {}
        m = {"boot": cfg.get("key"), "status": (rec or {}).get("status"), "config": cfg.get("diff_vs_rc2_fp8"),
             "F": ((rec or {}).get("f_census") or {}).get("F"), "pool_blocks": ((rec or {}).get("f_census") or {}).get("pool_blocks"),
             "k": cfg.get("k"), "head": cfg.get("head"), "seqs": cfg.get("seqs"), "windows": {}}
        best, gains = None, []
        for key, w in (((rec or {}).get("steady")) or {}).items():
            sw = _win(rec, key)
            m["windows"][key] = {"status": w.get("status"), "tok_s": w.get("tok_s"), "accept_per_draft": w.get("accept_per_draft"),
                                 "per_position": (w.get("per_pos") or {}).get("per_position"), "step_ms": w.get("step_ms"),
                                 "tokens_per_step": w.get("tokens_per_step"), "running": [w.get("running_min"), w.get("running_max")],
                                 "tick_ms": w.get("tick_ms"), "cycle_ms": w.get("cycle_ms"),
                                 "accepted_per_verify_token": w.get("accepted_per_verify_token"), "microbatch": w.get("microbatch")}
            if sw:
                f = steady_formula(sw)
                m["windows"][key]["formula"] = f
                if f and (best is None or f["steady_tok_s"] > best["steady_tok_s"]):
                    best = dict(f, window=key)
                bw = base_windows.get(key)
                fb = steady_formula(bw) if bw and bw.get("status") == "STEADY" else None
                if f and fb:
                    gains.append({"window": key, "arm_steady": f["steady_tok_s"], "base_steady": fb["steady_tok_s"],
                                  "run_gain_tok_s": round(0.9 * (f["steady_tok_s"] - fb["steady_tok_s"]), 1),
                                  "run_gain_pct": round(100 * (f["steady_tok_s"] - fb["steady_tok_s"]) / fb["steady_tok_s"], 1),
                                  "arm_inputs": f, "base_inputs": fb})
        m["resident_streams_max"] = max([w.get("running_max") or 0 for w in (((rec or {}).get("steady")) or {}).values()
                                         if w.get("status") in ("STEADY", "CAPACITY_LIMITED")] or [0]) or None
        lf = (rec or {}).get("log_facts") or {}
        m["memory_per_k"] = {k: lf.get(k) for k in ("kv_tokens", "max_concurrency", "block_bytes", "kv_cache_group_sizes",
                                                    "mamba_page_pad_pct", "draft_head_copy", "initial_free_gib", "max_num_scheduled_tokens",
                                                    "graph_capture")}
        m["idle_blocks"] = ((rec or {}).get("f_census") or {}).get("idle")
        m["step_model"] = (rec or {}).get("step_model")
        m["cohort_verdict"] = (rec or {}).get("cohort_verdict")
        m["best_window"] = best
        dw = [w for w in (((rec or {}).get("decay") or {}).get("windows") or []) if w.get("valid") and w.get("a_output_per_cycle")]
        a_long = dw[-1]["a_output_per_cycle"] if dw else None
        m["long_trace"] = {"planning_a_k3": PLANNING_A, "a_long_last_valid_window": a_long,
                           "a_series": [w.get("a_output_per_cycle") for w in dw],
                           "projected_long_tok_s": round(best["steady_tok_s"] * a_long / best["a_output_tokens_per_cycle"], 1)
                           if best and a_long and best.get("a_output_tokens_per_cycle") and best.get("steady_tok_s") else None,
                           "note": "ladder windows are early reasoning; RC2's long natural row fell 3.18 -> 2.69 (planning a = 2.72)"}
        m["decay"] = {k: ((rec or {}).get("decay") or {}).get(k) for k in ("status", "last_over_first", "valid_windows")}
        m["graph"] = {k: ((rec or {}).get("graph_plan") or {}).get(k) for k in ("full_decode_sizes", "covers_all_request_counts")}
        proj = {"basis": f"0.9 x (steady_arm - steady_{base_label}) at matched windows; steady = C x a / (2 x step_ms)",
                "matched": gains,
                "gain_tok_s": round(statistics.mean(g["run_gain_tok_s"] for g in gains), 1) if gains else None}
        conf = ("low: one window per level, no repeats" + (", windows disagree in sign" if gains and len({g["run_gain_tok_s"] > 0 for g in gains}) > 1
                                                          else "")) if gains else "none: no matched STEADY window"
        cmp_ = comparison_label(cfg, base_cfg) if base_cfg is not None else None
        if cmp_:
            proj["comparison"] = cmp_["label"]
        return {"lever": lever, "measured": m, "comparison": cmp_, "projected_run_tok_s_gain": proj, "correctness": _correct(rec), "confidence": conf,
                "adoption": adoption, "cost_gpu_h": round(4 * ((rec or {}).get("sec") or 0) / 3600, 2),
                "notes": (notes + "; " if notes else "") + NB1_DEP}


    PLANNING_A = 2.72          # PLAN_1500 1.1 v3.5: the planning scenario (RC2 natural long row, K3)
    NB1_DEP = ("DIAGNOSTIC until NB0's FP8 gate passes (RC2 TF FP8 vs BF16 mean |dlogprob| 1.88, rank-1 0.37; NB0 owns the MTP-off "
               "reference and the TF isolation); a production candidate also needs a paired ARC accuracy run (PLAN section 0); "
               "cost_gpu_h = 4 GPUs x this boot's wall time (one repeat)")


    def scorecard():
        S = R["sections"].get("S2") or {}
        rec = {k[: k.index("_")] if "_" in k else k: v for k, v in S.items() if isinstance(v, dict) and v.get("cfg")}
        ctrl = RC2_CONTROL["windows"]
        a = rec.get("A")
        a_win = (a or {}).get("steady") or {}
        rows = []
        if a:
            rows.append(score_row("D2 draft head int8 (K3)", a, ctrl, "rc2_control", "config",
                                  f"A (seqs {a['cfg']['seqs']}, captures {a['cfg']['captures']}) vs the RC2 FP8 control (K3, BF16 head, "
                                  "seqs 8, shipped captures): see comparison (DRAFT_HEAD + the seqs-16 graph config when A ran at 16)",
                                  base_cfg=CONTROL_CFG))
        for bk, lv in (("B", "D1 MTP K2 + int8"), ("C", "D1 MTP K1 + int8"), ("BH", "D1 MTP K2 + bf16 head (fallback)"),
                       ("CH", "D1 MTP K1 + bf16 head (fallback)"), ("K0", "D1 MTP off (K0, target only)")):
            if rec.get(bk):
                base, lbl, bcfg = ((a_win, "boot_A", a["cfg"]) if a_win and _correct(a) == "PASS"      # a failed A is no base
                                   else (ctrl, "rc2_control", CONTROL_CFG))
                rows.append(score_row(lv, rec[bk], base, lbl, "config",
                                      f"vs {lbl} at matched windows; seqs {rec[bk]['cfg']['seqs']} (comparison label: K_ONLY = the "
                                      "pair differs only in K; K_CHANGE_PLUS_GRAPH_CONFIG / UNMATCHED_CONFIG = not a clean K pair)",
                                      base_cfg=bcfg))
        gp = {k: {"full_decode_sizes": (v.get("graph_plan") or {}).get("full_decode_sizes"),
                  "covers": (v.get("graph_plan") or {}).get("covers_all_request_counts"),
                  "log_captures": (v.get("log_facts") or {}).get("captures_done"),
                  "trace_modes": ((v.get("trace") or {}).get("nb2") or {}).get("graph_modes")} for k, v in rec.items()}
        if gp:
            rows.append({"lever": "D3 graph coverage (FULL decode graphs per K / seqs)", "measured": gp,
                         "projected_run_tok_s_gain": {"gain_tok_s": None, "basis": "no eager-vs-graph A/B in NB2"},
                         "correctness": "UNTESTED", "confidence": "coverage is exact (config + log); speed effect unmeasured",
                         "adoption": "config", "cost_gpu_h": None,
                         "notes": "capture lists cover seqs x (1+K) by construction; a gap shows as covers=false or a PIECEWISE/EAGER "
                                  "trace mode; " + NB1_DEP})
        out = {"notebook": "E25 NB2", "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "rows": rows,
               "rule": "PLAN_1500 4.1: score = gain x (correctness PASS) x confidence / adoption cost (config 1, patch 2, research 4)"}
        with _SAVE_LOCK:
            tmp = OUT / "lever_scorecard.json.tmp"
            tmp.write_text(json.dumps(out, indent=1, default=str))
            tmp.replace(OUT / "lever_scorecard.json")
        return out
    save()


if not RERUN:
    # ==================================================================== CELL 7 — E25 NB2: the boots
    def nb2_step(bk, seqs=None):
        cfg_name = f"{bk}_k{BOOTS[bk]['k']}_{BOOTS[bk]['head'] or 'nohead'}_s{seqs or BOOTS[bk]['seqs']}"
        r = step("S2", cfg_name, BOOTS[bk]["est_min"], lambda: nb2_arm(bk, seqs))
        try:
            scorecard()
        except Exception as e:
            R["errors"].append(f"scorecard: {type(e).__name__}: {e}"[:300])
        save()
        return r


    def nb2_plan():
        """The planned boot list and minutes (printed at start). FULL: A / B / C (+ NB2_OPTIONAL); SMOKE: boot B only. The
        BF16-head fallback (BH / CH) is conditional and not counted."""
        if NB2_PROFILE == "SMOKE_CTRL":
            keys = ["CTRL"]
        elif SMOKE:
            keys = ["B"]
        else:
            keys = ["A", "B", "C"] + [k for k in ("K0", "CTRL") if k in NB2_OPT]
        boots = [{"key": k, "k": BOOTS[k]["k"], "head": BOOTS[k]["head"] or "none", "seqs": BOOTS[k]["seqs"],
                  "est_min": BOOTS[k]["est_min"], "long_window": bool(BOOTS[k].get("long_window")),
                  "ladder": [f"{b}@{n}" for b, n in BOOTS[k]["ladder"]] + (["8k@16 admission check"] if BOOTS[k].get("smoke")
                                                                           and BOOTS[k]["seqs"] > 8 else [])}
                 for k in keys]
        bm = sum(b["est_min"] for b in boots)
        return {"profile": NB2_PROFILE, "diag": DIAG, "window_seed": WINDOW_SEED,
                "boots": boots, "boot_min": bm, "setup_min": SETUP_EST_MIN, "total_est_min": bm + SETUP_EST_MIN,
                "budget_min": BUDGET // 60, "optional_selected": [k for k in NB2_OPT if not SMOKE],
                "conditional": ("none (SMOKE)" if SMOKE else "BH / CH (K2 / K1, BF16 head, seqs 16, 45 min each) only when A fails for "
                                "an INT8-head-specific reason, each budget-checked; a seqs-16 launch failure (no OOM) retries "
                                "that boot at seqs 8")}


    HEAD_RX = re.compile(r"draft_lm_head|Draft lm_head|lm_head_quant|DRAFT_HEAD", re.I)
    ERR_RX = re.compile(r"Error|Exception|Traceback|assert|failed|not supported|unsupported", re.I)


    def int8_head_failure(a):
        """Why boot A (K3 + INT8 draft head) failed, when the reason is specific to the INT8 head (then B / C run with the
        BF16 head instead of being skipped); None for any other failure (host / GPU gate, budget, OOM, generic launch
        failure, K / seqs / capture readback, deadline), where B / C stay skipped."""
        a = a or {}
        st, cor = str(a.get("status")), str(a.get("correctness"))
        rb = (a.get("config_readback") or {}).get("checks") or {}
        head_keys = ("head_in_spec", "head_built_line")
        if rb and any(rb.get(k) is False for k in head_keys) and all(v is True for k, v in rb.items() if k not in head_keys):
            return "INT8_HEAD_READBACK: the server log does not show the INT8 draft head (" + \
                   ", ".join(k for k in head_keys if rb.get(k) is False) + "); every other knob read back"
        if st in ("LAUNCH_FAILED", "SERVER_DIED", "BODY_ERROR"):
            tails = [v.get("log_tail") or "" for v in (a.get("servers") or {}).values() if isinstance(v, dict)]
            tails.append(str(a.get("body_error") or ""))
            hit = [l.strip()[-240:] for t in tails for l in t.splitlines() if HEAD_RX.search(l) and ERR_RX.search(l)]
            if hit:
                return f"INT8_HEAD_{st}: " + hit[0]
        if st == "NO_GO_ACCEPTANCE_COLLAPSE":
            return "INT8_HEAD_ACCEPTANCE_COLLAPSE: draft acceptance collapsed with the INT8 head"
        if (st.startswith("NO_GO_ORACLE") or cor.startswith("NO_GO_ORACLE") or str(a.get("early_stop") or "").startswith("ORACLE_GATE")
                or st == "NO_GO_DECAY_GARBAGE") and (a.get("config_readback") or {}).get("ok") is True:
            return (f"INT8_HEAD_OR_GRAPH_SUSPECT ({st} / {cor}): A's numerics differ from RC2 only by the INT8 head and the seqs-16 "
                    "graph config; the BF16-head boot decides which")
        return None


    def nb2_seq_step(bk):
        """One boot at seqs 16 (8 once seqs 16 was refused); a seqs-16 launch failure without an OOM line retries it at 8."""
        seqs = 16 if SEQS16["accepted"] is not False else 8
        r = nb2_step(bk, seqs)
        if seqs == 16 and r.get("status") == "LAUNCH_FAILED" and not any(t.get("oom") for t in r.get("tries") or []):
            SEQS16.update(accepted=False, why=f"boot {bk} failed to launch at --max-num-seqs 16 (no OOM line)")
            r = nb2_step(bk, 8)
        return r


    def _ok(r):
        return (r or {}).get("status") == "DONE" and (r or {}).get("correctness") == "CLEAN"


    def nb2_fallback(a, why):
        """Decision 2026-10-31: A failed because of the INT8 head -> B and C run with the BF16 head (BH, CH), each only when
        the budget holds it (checked first) and CH only after a clean BH."""
        S2 = R["sections"].setdefault("S2", {})
        fb = S2.setdefault("fallback", {"reason": why, "a_status": [a.get("status"), a.get("correctness")], "boots": {}})
        for bk, orig in (("BH", "B"), ("CH", "C")):
            need = BOOTS[bk]["est_min"] * EST_SCALE * 60 + 300
            rec = {"left_min": round(left() / 60, 1), "need_min": round(need / 60, 1)}
            if bk == "CH" and not _ok(fb.get("bh_result")):
                rec["decision"] = "SKIPPED_DEPENDENCY (BH not DONE / CLEAN)"
            elif left() < need:
                rec["decision"] = "SKIPPED_FALLBACK_NO_BUDGET"
            else:
                rec["decision"] = "RUN_BF16_HEAD"
            fb["boots"][bk] = rec
            S2[f"{orig}_skipped"] = {"status": "REPLACED_BY_BF16_HEAD_FALLBACK" if rec["decision"] == "RUN_BF16_HEAD"
                                     else "SKIPPED_DEPENDENCY", "why": f"boot A: {why}", "fallback": rec["decision"]}
            log("NB2 fallback", bk, json.dumps(rec), "|", why[:200])
            save()
            if rec["decision"] == "RUN_BF16_HEAD":
                r = nb2_seq_step(bk)
                rec["result"] = [r.get("status"), r.get("correctness")]
                if bk == "BH":
                    fb["bh_result"] = {"status": r.get("status"), "correctness": r.get("correctness")}
        save()
        return fb


    def smoke_verdict(r):
        """SMOKE gate for the FULL run: boot B reached DONE / CLEAN with its config read back, the 8k@8 window STEADY, all 16
        streams admitted at once, and a parsed trace. The dev-mode reset is recorded, not gated (it answers success false
        while retained turns hold blocks)."""
        r = r or {}
        tr = ((r.get("trace") or {}).get("nb2")) or {}
        chk = {"boot_done_clean": _ok(r), "config_readback": (r.get("config_readback") or {}).get("ok") is True,
               "no_early_stop": not r.get("early_stop"),
               "window_8k@8_steady": ((r.get("steady") or {}).get("8k@8") or {}).get("status") == "STEADY",
               "admission_16": (r.get("admission_16") or {}).get("status") == "ADMITTED_ALL",
               "trace_parsed": bool(tr.get("ranks")) and not tr.get("error") and tr.get("status") != "NO_TRACE_FILE"}
        bad = [k for k, v in chk.items() if not v]
        return {"verdict": "SMOKE_PASS" if not bad else "SMOKE_FAIL", "checks": chk, "failed": bad,
                "status": [r.get("status"), r.get("correctness"), r.get("early_stop")],
                "dev_mode_reset": r.get("dev_mode_reset"), "minutes": round((time.time() - T_START) / 60, 1),
                "rule": "the trimmed FULL NB2 runs only after SMOKE_PASS"}


    EXACT_DIFF_KEYS = ("flags_changed", "flags_added", "flags_removed", "env_changed")


    DIAG_B_VS_CTRL = ["MTP K 2 vs 3", "draft head INT8 vs BF16", "--max-num-seqs 16 vs 8", "captures CAPS16 vs shipped",
                      "dev-mode prefix-cache reset vs none (VLLM_SERVER_DEV_MODE)", "--profiler-config vs none",
                      "window seed 4321 (SEED2) vs 1234 (CTRL = the failed SMOKE)"]
    DIAG_INTERPRETATION = [   # N2-3 / N2-5 / N2-6: every row "suggests", never "proves"; it needs BOTH diagnostic results
        {"ctrl": "clean", "seed2": "loop",
         "reading": "suggests the B configuration GROUP is associated with the loop (K2, INT8 draft head, seqs 16, CAPS16 "
                    "captures and the cache reset all differ from CTRL, and the window seed differs too); it does not single out "
                    "one factor",
         "next_arms": ["K2 + BF16 head + seqs 16 + CAPS16, same prompts / seeds (separates the head)",
                       "K2 + INT8 head + seqs 8 (separates seqs / captures)"]},
        {"ctrl": "loop", "seed2": "clean",
         "reading": "suggests target- / prompt-level repetition is plausible (the exact RC2 control loops on the failed run's "
                    "prompts and seed); the B group is not implicated by this pair",
         "next_arms": ["K0 (target only, no MTP) on the same prompts / seed 1234"]},
        {"ctrl": "clean", "seed2": "clean",
         "reading": "suggests the failed SMOKE's loop was not reproduced (a rare sampled event stays possible); no attribution",
         "next_arms": ["none required by this pair; the FULL decision stays with the lead"]},
        {"ctrl": "loop", "seed2": "loop",
         "reading": "suggests no single attribution: both configurations loop (target- / prompt-level repetition stays "
                    "plausible because CTRL loops)",
         "next_arms": ["K0 (target only) on the same prompts / seed 1234", "K2 + BF16 head + seqs 16 + CAPS16"]},
    ]


    def diag_verdict(r):
        """SMOKE_SEED2 / SMOKE_CTRL verdict: DIAG_DONE once the boot read its config back, reached the 8k@8 window without an
        early stop, ran the window to its full length, and the per-stream table is complete and unambiguous: prompt_identity
        (the 8k@8 prompts = the failed run's rows 37-44 in order, and every stream's prompt sha matches), stream_integrity
        (exactly 8 streams, seeds exactly WINDOW_SEED .. WINDOW_SEED + 7, none missing, no request error, n_tokens an int > 0),
        trace_status TABLE and no_duplicate_streams (SMOKE_CTRL also: the exact diff vs RC2 FP8 is empty); else DIAG_INCOMPLETE
        with the failed items and their details. A genuine loop with complete records stays DIAG_DONE (the loop is the
        observation). The boot's own status (e.g. NO_GO_GARBAGE_UNDER_LOAD) stays as gated; SMOKE_PASS / SMOKE_FAIL belong to
        the SMOKE profile only. The interpretation table needs both diagnostic results; every row only suggests."""
        r = r or {}
        w = ((r.get("steady") or {}).get("8k@8")) or {}
        t = r.get("diag_streams") or {}
        cfg = r.get("cfg") or {}
        st = t.get("streams") or []
        pc = R.get("diag_prompt_check") or {}
        want_seeds = list(range(WINDOW_SEED, WINDOW_SEED + 8))
        missing = [x.get("stream") for x in st if x.get("missing")]
        errors = {x.get("stream"): x.get("error") for x in st if not x.get("missing") and x.get("error") is not None}
        bad_tok = [x.get("stream") for x in st if not x.get("missing")
                   and not (isinstance(x.get("n_tokens"), int) and not isinstance(x.get("n_tokens"), bool) and x["n_tokens"] > 0)]
        sha_bad = [x.get("stream") for x in st if x.get("prompt_sha_matches_failed_run") is not True]
        detail = {
            "prompt_identity": {"same_set_and_order": pc.get("same_set_and_order"), "mismatch_streams": pc.get("mismatch_streams"),
                                "streams_sha_not_matching": sha_bad},
            "stream_integrity": {"n_streams": len(st), "table_seed": t.get("seed"), "seeds": [x.get("seed") for x in st],
                                 "want_seeds": [want_seeds[0], want_seeds[-1]], "missing": missing, "errors": errors,
                                 "bad_n_tokens": bad_tok},
            "trace_status": t.get("status"), "no_duplicate_streams": t.get("duplicate_streams")}
        chk = {"config_readback": (r.get("config_readback") or {}).get("ok") is True, "no_early_stop": not r.get("early_stop"),
               "window_ran_full": bool(w.get("window_sec")) and w["window_sec"] >= 0.95 * WIN,
               "all_8_streams_recorded": t.get("rows_found") == 8 and len(st) == 8,
               "prompt_identity": pc.get("same_set_and_order") is True and len(st) == 8 and not sha_bad,
               "stream_integrity": (len(st) == 8 and t.get("seed") == WINDOW_SEED and [x.get("seed") for x in st] == want_seeds
                                    and [x.get("stream") for x in st] == list(range(8)) and not missing and not errors and not bad_tok),
               "trace_status": t.get("status") == "TABLE",
               "no_duplicate_streams": isinstance(t.get("duplicate_streams"), list) and not t["duplicate_streams"]
                                       and not any(x.get("duplicate_rows") for x in st)}
        if NB2_PROFILE == "SMOKE_CTRL":
            ed = cfg.get("exact_diff") or {}
            chk["ctrl_exact_rc2_diff_empty"] = (cfg.get("exact_rc2") is True and isinstance(cfg.get("exact_diff"), dict)
                                                and "error" not in ed and not any(ed.get(k) for k in EXACT_DIFF_KEYS))
        bad = [k for k, v in chk.items() if not v]
        keep = ("stream", "seed", "prompt_sha", "n_tokens", "first_garbage_window", "tail_verdict", "finish_reason", "error",
                "prompt_sha_matches_failed_run", "failed_run_tail_verdict", "missing", "duplicate_rows", "logprobs", "loop")
        side = "ctrl" if NB2_PROFILE == "SMOKE_CTRL" else "seed2"
        outcome = None if bad else ("loop" if t.get("loop_count") else "clean")
        interp = {"rule": "every row suggests, none proves; a row applies only once BOTH SMOKE_SEED2 and SMOKE_CTRL are DIAG_DONE",
                  "b_vs_ctrl_differences": DIAG_B_VS_CTRL, "this_result": {"profile": NB2_PROFILE, "side": side, "outcome": outcome},
                  "rows": DIAG_INTERPRETATION,
                  "rows_consistent_with_this_result": [i for i, x in enumerate(DIAG_INTERPRETATION) if outcome and x[side] == outcome],
                  "note": None if outcome else "DIAG_INCOMPLETE: this result is not an interpretation input"}
        return {"verdict": "DIAG_DONE" if not bad else "DIAG_INCOMPLETE", "profile": NB2_PROFILE, "checks": chk, "failed": bad,
                "failed_detail": {k: detail[k] for k in bad if k in detail}, "interpretation": interp,
                "logprobs": r.get("diag_logprobs"), "duplicate_streams": t.get("duplicate_streams"), "trace_status": t.get("status"),
                "loop_count": t.get("loop_count"), "looped_streams": t.get("looped_streams"),
                "loop_count_windowed": t.get("loop_count_windowed"), "loop_count_tail_gate": t.get("loop_count_tail_gate"),
                "streams": [{k: x.get(k) for k in keep if k in x} for x in t.get("streams") or []],
                "window": {k: w.get(k) for k in ("status", "window_sec", "tok_s", "accept_per_draft", "running_min", "running_max",
                                                 "preemptions", "garbage")},
                "seed": WINDOW_SEED, "sampling": {"temperature": TEMP, "top_p": 0.95, "top_k": 20, "ignore_eos": True,
                                                  "max_tokens": MAX_NEW["8k"], "seeds": [WINDOW_SEED, WINDOW_SEED + 7]},
                "boot": {"key": cfg.get("key"), "k": cfg.get("k"), "head": cfg.get("head"), "seqs": cfg.get("seqs"),
                         "diff_vs_rc2_fp8": cfg.get("diff_vs_rc2_fp8")},
                "prompt_check": {k: (R.get("diag_prompt_check") or {}).get(k) for k in ("same_set", "same_set_and_order", "mismatch_streams")},
                "failed_smoke": {"looped_streams": t.get("failed_run_looped_streams"), "seed": 1234, "window": NB2_FAILED_SMOKE.get("window"),
                                 "looped_rows": [{k: x.get(k) for k in ("row", "stream", "seed", "tail_verdict", "label", "numeric_cause",
                                                                        "last_500_token_ids", "margins_kept", "evidence")}
                                                 for x in NB2_FAILED_SMOKE["rows"] if x.get("tail_verdict")]},
                "admission_16": (r.get("admission_16") or {}).get("status"), "dev_mode_reset": r.get("dev_mode_reset"),
                "status": [r.get("status"), r.get("correctness"), r.get("early_stop")], "minutes": round((time.time() - T_START) / 60, 1),
                "rule": "diagnostic: loop count per config / seed; SMOKE_PASS / SMOKE_FAIL stay the SMOKE profile's gate"}


    def nb2_main():
        R["plan"] = nb2_plan()
        log("NB2 plan", json.dumps(R["plan"]))
        m = R["sections"]["A0"]["merge"]
        if not m.get("ok"):
            R["sections"]["S2"] = {"skipped": {"status": "SKIPPED_MERGE_FAILED", "merge": m}}
            return
        step("A0", "merlin_install", SETUP_EST_MIN, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE
                                                                         else {"fake": True})(merlin_install())})
        ok, why = sha_gate(min(1500, left() - (300 if SMOKE else 900)))
        R["setup"]["sha_gate"] = {"ok": ok, "why": why}
        if not ok:
            R["sections"]["S2"] = {"skipped": {"status": "SKIPPED_SHA_GATE", "why": why}}
            return
        if SMOKE:
            r = nb2_step("CTRL") if NB2_PROFILE == "SMOKE_CTRL" else nb2_step("B", 16)
            R["smoke"] = diag_verdict(r) if DIAG else smoke_verdict(r)
            log("NB2", NB2_PROFILE, R["smoke"]["verdict"], json.dumps(R["smoke"]["checks"]))
            if DIAG:
                log("NB2 DIAG loops", R["smoke"]["loop_count"], "streams", R["smoke"]["looped_streams"], "| failed SMOKE",
                    R["smoke"]["failed_smoke"]["looped_streams"])
            R["seqs16"] = dict(SEQS16)
            return
        a = nb2_seq_step("A")
        if _ok(a):
            for bk in ("B", "C"):
                nb2_seq_step(bk)
        else:
            ROWS.pop("A", None)                     # a failed A is never an oracle reference
            why = int8_head_failure(a)
            if why:
                nb2_fallback(a, why)
            else:
                for bk in ("B", "C"):
                    R["sections"].setdefault("S2", {})[f"{bk}_skipped"] = {
                        "status": "SKIPPED_DEPENDENCY", "why": f"boot A (int8 head) is {a.get('status')} / {a.get('correctness')}; "
                                                              "not an INT8-head-specific failure, so no BF16-head fallback"}
        for bk in ("K0", "CTRL"):
            if bk in NB2_OPT:
                nb2_step(bk, (16 if SEQS16["accepted"] is not False else 8) if bk == "K0" else None)
            else:
                R["sections"].setdefault("S2", {})[f"{bk}_skipped"] = {"status": "SKIPPED_NOT_SELECTED"}
        R["seqs16"] = dict(SEQS16)


    if not RERUN and not os.environ.get("NB2_UNIT"):
        nb2_main()
        R["host_phase"] = HOST_PHASE
        R["prefill"] = PREFILL
        R["minutes"] = round((time.time() - T_START) / 60, 1)
        try:
            scorecard()
        except Exception as e:
            R["errors"].append(f"scorecard: {type(e).__name__}: {e}"[:300])
        save()
        log("NB2 done", json.dumps({f"{s}/{n}": r.get("status") for s, st in R["sections"].items() if isinstance(st, dict)
                                    for n, r in st.items() if isinstance(r, dict) and "status" in r}))


if not RERUN:
    # ==================================================================== CELL 8 — E25 NB2: table
    if not RERUN and not os.environ.get("NB2_UNIT"):
        print("profile", NB2_PROFILE, "| plan", json.dumps(R.get("plan"))[:600])
        if R.get("smoke"):
            print("SMOKE verdict", R["smoke"]["verdict"], json.dumps(R["smoke"]["checks"]))
        if DIAG and R.get("smoke"):
            print("DIAG prompt check vs the failed SMOKE rows 37-44:", json.dumps(R.get("diag_prompt_check"))[:900])
            print(f"DIAG {NB2_PROFILE}: seed {R['smoke']['seed']} | loops {R['smoke']['loop_count']} streams {R['smoke']['looped_streams']} "
                  f"| failed SMOKE (seed 1234, K2 + INT8 + seqs 16) looped {R['smoke']['failed_smoke']['looped_streams']}")
            for _x in R["smoke"]["streams"]:
                print(f"    stream {_x.get('stream')} seed {_x.get('seed')} prompt {_x.get('prompt_sha')} tokens {_x.get('n_tokens')} "
                      f"first_garbage_window {_x.get('first_garbage_window')} tail {_x.get('tail_verdict')} "
                      f"(failed SMOKE tail {_x.get('failed_run_tail_verdict')})")
        for _n, _r in (R["sections"].get("S2") or {}).items():
            if not isinstance(_r, dict):
                continue
            print(f"--- {_n}: {_r.get('status')} / {_r.get('correctness')} | F {(_r.get('f_census') or {}).get('F')} | "
                  f"readback {(_r.get('config_readback') or {}).get('ok')} | decay {(_r.get('decay') or {}).get('status')}")
            for _k, _w in (_r.get("steady") or {}).items():
                print(f"    {_k:8s} {str(_w.get('status')):32s} tok/s {_w.get('tok_s')} acc/draft {_w.get('accept_per_draft')} "
                      f"pos {(_w.get('per_pos') or {}).get('per_position')} step {_w.get('step_ms')} ms "
                      f"run {_w.get('running_min')}-{_w.get('running_max')}")


if not RERUN:
    # ==================================================================== CELL 6 — report + fallback submission
    if not RERUN and not os.environ.get("NB2_UNIT"):
        write_fallback_submission()
        def _brief(r):
            if not isinstance(r, dict):
                return r
            keep = ("status", "correctness", "fp8_gate", "probes", "sampled", "prefill", "census", "steady", "census_long", "long",
                    "tf", "tries", "kv_bytes", "boot_gate", "why", "error", "status_note", "load_tests_skipped",
                    "certification", "early_stop", "long_verdict", "cfg", "config_readback", "f_census", "decay", "boot_deadline",
                        "admission_16", "dev_mode_reset")
            return {k: r.get(k) for k in keep if k in r}
        R["summary"] = {f"{s}/{n}": _brief(r) for s, st in R["sections"].items() if isinstance(st, dict) for n, r in st.items()
                        if isinstance(r, dict) and "status" in r}
        save()
        print(json.dumps(R["summary"], default=str)[:6000])
