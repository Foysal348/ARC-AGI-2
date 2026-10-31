# ==================================================================== CELL 1 — config, inputs, lab module
import os, sys, json, math, time, re, shutil, subprocess, hashlib, tarfile, threading, traceback
from pathlib import Path
T_START = time.time()
MODE = "FULL"
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes"}
FAKE = os.environ.get("E23_FAKE_SERVER")                          # local CPU harness only
BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 240 * 60)     # two cold Merlin boots + ladder + 90k/116k census + long rows
HARD_STOP = T_START + BUDGET
MAX_LEN = 131072                                                  # user 2026-10-29: thinking up to 120k
OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/rc2"))
OUT.mkdir(parents=True, exist_ok=True)
LOGS = OUT / "logs"
LOGS.mkdir(exist_ok=True)
TRACES = OUT / "traces"                                           # memory rule: the full text of every long generation
TRACES.mkdir(exist_ok=True)
TMP = Path(os.environ.get("E23_TMP", "/tmp"))
INP = Path(os.environ.get("E23_INPUT", "/kaggle/input"))
R = {"what": "E13 RC2 Flash-Next on Merlin", "budget_min": BUDGET // 60, "max_len": MAX_LEN,
     "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "setup": {}, "sections": {}, "errors": [], "order": []}
_SAVE_LOCK = threading.Lock()


def save():
    with _SAVE_LOCK:
        tmp = OUT / "rc2_results.json.tmp"
        tmp.write_text(json.dumps(R, indent=1, default=str))
        tmp.replace(OUT / "rc2_results.json")


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
        Path(os.environ.get("RC2_SUBMISSION", "/kaggle/working/submission.json")).write_text(json.dumps(sub))


if RERUN:
    write_fallback_submission()
    print("E13 RC2 diagnostic: competition rerun -> fallback submission.json only")
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
                 "--default-chat-template-kwargs", XHIGH]
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
        argv = plan.argv()
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
                         "finish": row["finish_reason"], "garbage": sl, "think_closed": "</think>" in row["text"],
                         "cut_by_cap": row["finish_reason"] is None, "tail": row["text"][-400:]}
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
        kvb = KV_FP8 if kv == "fp8" else KV_BF16
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
        return g


    def arm_bf16():
        def body(ss):
            s = ss[0]
            p = s.port
            o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
            o["prefill"] = prefill_probe(p)
            o["oracle"] = oracle(p, key="bf16")
            o["census_end"] = kv_census(s)
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
            if o["sampled"]["ok"] is True:
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
            o["oracle"] = oracle(p, key="stock", prompts=ORACLE_P)          # 2k / 2k / 8k (32k max length: no 40k row)
            if o["sampled"]["ok"] is True:
                o["steady"] = levels(p, "8k", [1, 2, 3], server=s, temperature=TEMP)
            return o
        gate = boot_gate()
        if gate["ok"] is not True:
            return {"status": "GPU_NOT_EMPTY", "boot_gate": gate}
        shm0 = shm_names()
        o = gated(with_servers([stock_fn_server("stock_fn", 8003)], 40 * 60, body), expected_rows=len(ORACLE_P))
        o.update(boot_gate=gate, shm_cleanup=shm_cleanup(shm0))
        return o


    if not RERUN and not os.environ.get("RC2_UNIT"):
        m = R["sections"]["A0"]["merge"]
        if m.get("ok"):
            rec = step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
            a1 = step("A1", "merlin_bf16", 45, arm_bf16)
            if a1.get("status") in ("DONE", "TF_OOM") and a1.get("correctness") == "CLEAN":
                a3 = step("A3", "merlin_fp8", 110, arm_fp8)
                if str(a3.get("fp8_gate", "")).startswith("FAIL"):
                    step("A3", "merlin_bf16_load", 60, arm_bf16_load)
            else:
                step("F", "stock_fn", 40, arm_stock)
        else:
            step("F", "stock_fn", 40, arm_stock)
        R["host_phase"] = HOST_PHASE
        R["prefill"] = PREFILL
        R["minutes"] = round((time.time() - T_START) / 60, 1)
        save()
        log("RC2 done", json.dumps({f"{s}/{n}": r.get("status") for s, st in R["sections"].items() if isinstance(st, dict)
                                    for n, r in st.items() if isinstance(r, dict) and "status" in r}))


if not RERUN:
    # ==================================================================== CELL 6 — report + fallback submission
    if not RERUN and not os.environ.get("RC2_UNIT"):
        write_fallback_submission()
        def _brief(r):
            if not isinstance(r, dict):
                return r
            keep = ("status", "correctness", "fp8_gate", "probes", "sampled", "prefill", "census", "steady", "census_long", "long",
                    "tf", "tries", "kv_bytes", "boot_gate", "why", "error", "status_note", "load_tests_skipped",
                    "certification")
            return {k: r.get(k) for k in keep if k in r}
        R["summary"] = {f"{s}/{n}": _brief(r) for s, st in R["sections"].items() if isinstance(st, dict) for n, r in st.items()
                        if isinstance(r, dict) and "status" in r}
        save()
        print(json.dumps(R["summary"], default=str)[:6000])
