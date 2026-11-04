# ==================================================================== CELL 1 — config, inputs, lab module (RC2; NB0 edits marked)
import os, sys, json, math, time, re, shutil, subprocess, hashlib, tarfile, threading, traceback
from pathlib import Path
T_START = time.time()
MODE = "FULL"
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes"}
FAKE = os.environ.get("E23_FAKE_SERVER")                          # local CPU harness only
BUDGET = int(os.environ.get("NB0_BUDGET_SEC") or 100 * 60)     # NB0: 5 short TF boots (lead: ~100 min)
HARD_STOP = T_START + BUDGET
MAX_LEN = 131072                                                  # user 2026-10-29: thinking up to 120k
OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/nb0"))                 # NB0
OUT.mkdir(parents=True, exist_ok=True)
LOGS = OUT / "logs"
LOGS.mkdir(exist_ok=True)
TRACES = OUT / "traces"                                           # memory rule: the full text of every long generation
TRACES.mkdir(exist_ok=True)
TMP = Path(os.environ.get("E23_TMP", "/tmp"))
INP = Path(os.environ.get("E23_INPUT", "/kaggle/input"))
R = {"what": "E25 NB0 FP8 quality isolation (Flash-Next on Merlin)", "budget_min": BUDGET // 60, "max_len": MAX_LEN,
     "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "setup": {}, "sections": {}, "errors": [], "order": []}
_SAVE_LOCK = threading.Lock()


def save():
    with _SAVE_LOCK:
        tmp = OUT / "nb0_results.json.tmp"                 # NB0
        tmp.write_text(json.dumps(R, indent=1, default=str))
        tmp.replace(OUT / "nb0_results.json")


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
    log("NB0 | budget", BUDGET // 60, "min |", json.dumps(R["inputs"])[:900])
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


    # NB0: RC2's arms (A1 / A3 / F) and its main block are not used; NB0's boots follow in CELL 7.


if not RERUN:
    # ==================================================================== CELL 6 — NB0 library (new code)
    import textwrap
    import numpy as np
    import urllib.request as UR
    import urllib.error as UE
    R.pop("capacity_plan", None)          # RC2-only records written by RC2's CELL 5 definitions
    R.pop("correctness_meaning", None)
    RC2_REF = json.loads("{\"fp8\": {\"argv_tail\": [\"127.0.0.1\", \"--port\", \"8001\", \"--tensor-parallel-size\", \"2\", \"--pipeline-parallel-size\", \"2\", \"--enable-expert-parallel\", \"--disable-custom-all-reduce\", \"--speculative-config\", \"{\\\"method\\\":\\\"mtp\\\",\\\"num_speculative_tokens\\\":3,\\\"draft_sample_method\\\":\\\"probabilistic\\\"}\", \"--max-model-len\", \"262144\", \"--max-num-seqs\", \"8\", \"--max-num-batched-tokens\", \"1024\", \"--gpu-memory-utilization\", \"0.96\", \"--kv-cache-memory-bytes\", \"2479838700\", \"--kv-cache-dtype\", \"fp8_e4m3\", \"--limit-mm-per-prompt\", \"{\\\"image\\\":42,\\\"video\\\":0}\", \"--mm-processor-cache-gb\", \"0\", \"--mm-processor-kwargs\", \"{\\\"size\\\":{\\\"longest_edge\\\":8388608,\\\"shortest_edge\\\":65536}}\", \"--compilation-config\", \"{\\\"cudagraph_mode\\\":\\\"FULL_AND_PIECEWISE\\\",\\\"cudagraph_capture_sizes\\\":[1,2,3,4,6,8,12,16,24,32]}\", \"--enable-prefix-caching\", \"--mamba-cache-mode\", \"align\", \"--enable-prompt-tokens-details\", \"--no-enable-flashinfer-autotune\", \"--enable-auto-tool-choice\", \"--tool-call-parser\", \"qwen3_coder\", \"--reasoning-parser\", \"qwen3\", \"--override-generation-config\", \"{\\\"temperature\\\":1.0,\\\"top_p\\\":0.95,\\\"top_k\\\":20,\\\"min_p\\\":0.0,\\\"presence_penalty\\\":0.0,\\\"repetition_penalty\\\":1.0}\", \"--default-chat-template-kwargs\", \"{\\\"enable_thinking\\\": true, \\\"reasoning_effort\\\": \\\"medium\\\"}\", \"--shutdown-timeout\", \"60\", \"--block-size\", \"4096\", \"--use-replayssm\", \"--prefix-match-unit\", \"64\", \"--served-model-name\", \"q27\", \"--max-model-len\", \"131072\", \"--port\", \"8001\", \"--default-chat-template-kwargs\", \"{\\\"enable_thinking\\\": true, \\\"reasoning_effort\\\": \\\"xhigh\\\"}\"], \"env\": {\"KV_BYTES\": \"2479838700\", \"NCCL_P2P_DISABLE\": \"1\", \"NCCL_VERSION\": \"2.25.1-1\", \"P2P\": \"off\", \"SCALES\": \"/tmp/fn_merlin/qsa_kv_scales_262k.json\", \"VLLM_CACHE_ROOT\": \"/tmp/merlin-cache\", \"VLLM_E2_GUARD_MODE\": \"warn\", \"VLLM_E2_PLE_EARLY_REQUEST\": \"1\", \"VLLM_E2_PLE_FAST_IDS\": \"1\", \"VLLM_E2_QSA_LOGITS_GRANULE_MB\": \"256\", \"VLLM_HOST_EMBED_TABLE\": \"1\", \"VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS\": \"0\", \"VLLM_PLE_OFFLOAD_HOME_DEVICE\": \"0\", \"VLLM_PLE_OFFLOAD_READY_TIMEOUT\": \"3600\", \"VLLM_PORT\": \"30100\", \"VLLM_PP_LAYER_PARTITION\": \"25,23\", \"VLLM_QSA_KV_SCALES\": \"/tmp/fn_merlin/qsa_kv_scales_262k.json\", \"VLLM_SKIP_P2P_CHECK\": \"1\", \"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB\": \"256\", \"VLLM_USE_BREAKABLE_CUDAGRAPH\": \"0\", \"VLLM_USE_RUST_FRONTEND\": \"0\", \"VLLM_USE_V2_MODEL_RUNNER\": \"1\"}, \"kv_bytes\": 2479838700, \"kv_tokens\": 513940, \"seqs\": 8}, \"bf16\": {\"argv_tail\": [\"127.0.0.1\", \"--port\", \"8000\", \"--tensor-parallel-size\", \"2\", \"--pipeline-parallel-size\", \"2\", \"--enable-expert-parallel\", \"--disable-custom-all-reduce\", \"--speculative-config\", \"{\\\"method\\\":\\\"mtp\\\",\\\"num_speculative_tokens\\\":3,\\\"draft_sample_method\\\":\\\"probabilistic\\\"}\", \"--max-model-len\", \"262144\", \"--max-num-seqs\", \"3\", \"--max-num-batched-tokens\", \"1024\", \"--gpu-memory-utilization\", \"0.96\", \"--kv-cache-memory-bytes\", \"2464415744\", \"--kv-cache-dtype\", \"bfloat16\", \"--limit-mm-per-prompt\", \"{\\\"image\\\":42,\\\"video\\\":0}\", \"--mm-processor-cache-gb\", \"0\", \"--mm-processor-kwargs\", \"{\\\"size\\\":{\\\"longest_edge\\\":8388608,\\\"shortest_edge\\\":65536}}\", \"--compilation-config\", \"{\\\"cudagraph_mode\\\":\\\"FULL_AND_PIECEWISE\\\",\\\"cudagraph_capture_sizes\\\":[1,2,3,4,6,8,12]}\", \"--enable-prefix-caching\", \"--mamba-cache-mode\", \"align\", \"--enable-prompt-tokens-details\", \"--no-enable-flashinfer-autotune\", \"--enable-auto-tool-choice\", \"--tool-call-parser\", \"qwen3_coder\", \"--reasoning-parser\", \"qwen3\", \"--override-generation-config\", \"{\\\"temperature\\\":1.0,\\\"top_p\\\":0.95,\\\"top_k\\\":20,\\\"min_p\\\":0.0,\\\"presence_penalty\\\":0.0,\\\"repetition_penalty\\\":1.0}\", \"--default-chat-template-kwargs\", \"{\\\"enable_thinking\\\": true, \\\"reasoning_effort\\\": \\\"medium\\\"}\", \"--shutdown-timeout\", \"60\", \"--block-size\", \"4096\", \"--use-replayssm\", \"--prefix-match-unit\", \"64\", \"--served-model-name\", \"q27\", \"--max-model-len\", \"131072\", \"--port\", \"8000\", \"--default-chat-template-kwargs\", \"{\\\"enable_thinking\\\": true, \\\"reasoning_effort\\\": \\\"xhigh\\\"}\"], \"env\": {\"KV_BYTES\": \"2464415744\", \"NCCL_P2P_DISABLE\": \"1\", \"NCCL_VERSION\": \"2.25.1-1\", \"P2P\": \"off\", \"VLLM_CACHE_ROOT\": \"/tmp/merlin-cache\", \"VLLM_E2_GUARD_MODE\": \"warn\", \"VLLM_E2_PLE_EARLY_REQUEST\": \"1\", \"VLLM_E2_PLE_FAST_IDS\": \"1\", \"VLLM_E2_QSA_LOGITS_GRANULE_MB\": \"256\", \"VLLM_HOST_EMBED_TABLE\": \"1\", \"VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS\": \"0\", \"VLLM_PLE_OFFLOAD_HOME_DEVICE\": \"0\", \"VLLM_PLE_OFFLOAD_READY_TIMEOUT\": \"3600\", \"VLLM_PORT\": \"30000\", \"VLLM_PP_LAYER_PARTITION\": \"25,23\", \"VLLM_SKIP_P2P_CHECK\": \"1\", \"VLLM_SPARSE_INDEXER_MAX_LOGITS_MB\": \"256\", \"VLLM_USE_BREAKABLE_CUDAGRAPH\": \"0\", \"VLLM_USE_RUST_FRONTEND\": \"0\", \"VLLM_USE_V2_MODEL_RUNNER\": \"1\"}, \"kv_bytes\": 2464415744, \"kv_tokens\": 269042, \"seqs\": 3}, \"tf40k\": {\"fp8_vs_bf16\": {\"status\": \"MEASURED\", \"n\": 256, \"mean_abs_dlogprob\": 1.8764, \"p99_abs_dlogprob\": 5.464, \"max_abs_dlogprob\": 7.15, \"rank1_agreement\": 0.3672, \"forced_token_rank1_both_or_neither\": 0.457}, \"bf16_aa\": {\"status\": \"MEASURED\", \"n\": 256, \"mean_abs_dlogprob\": 0.0, \"p99_abs_dlogprob\": 0.0, \"max_abs_dlogprob\": 0.0, \"rank1_agreement\": 1.0, \"forced_token_rank1_both_or_neither\": 1.0}}, \"oracle_rows\": {\"rc2_bf16\": [{\"text\": \"We need solve ARC task. Need infer transformation from examples. User gives Task 0934a4d8 with Example 1 input/output, Example 2 input/output, Example 3 input only? Need solve test inputs? They provided Example 3 input: 194499276 maybe incomplete? Need output? Need find transformation rule and solve test inputs. We need analyze.\\n\\nWe have ARC tasks represented as strings of digits? Example 1 input is 31 rows? Let's count. Example 1 input lines:\\n1 3533665414\", \"tokens\": [1596, 1144, 11290, 67964, 3274, 13, 14235, 22846, 17454, 494, 9989, 13, 2570, 6486, 5261, 220, 15, 24, 18, 19, 64, 19, 67, 23, 440, 12996, 220, 16, 1879, 45447, 11, 12996, 220, 17, 1879, 45447, 11, 12996, 220, 18, 1879, 1132, 30, 14235, 11290, 1228, 10804, 30, 2302, 3766, 12996, 220, 18, 1879, 25, 220, 16, 24, 19, 19, 24, 24, 17, 22, 21, 6970, 31084, 30, 14235, 2468, 30, 14235, 1423, 17454, 5728, 321, 11290, 1228, 10804, 13, 1165, 1144, 22903, 13, 271, 1596, 599, 67964, 8806, 14807, 430, 8796, 314, 17955, 30, 12996, 220, 16, 1879, 369, 220, 18, 16, 6761, 30, 6558, 579, 1698, 13, 12996, 220, 16, 1879, 4965, 25, 198, 16, 220, 18, 20, 18, 18, 21, 21, 20, 19, 16, 19], \"margins\": [11.375, 10.75, 0.25, 3.5, 0.125, 1.75, 2.375, 0.625, 4.375, 0.125, 4.125, 1.375, 0.125, 1.0, 1.25, 4.875, 8.6875, 10.375, 10.375, 9.6875, 8.25, 10.1875, 14.0, 10.4375, 0.125, 0.875, 1.25, 8.375, 3.875, 1.0, 3.75, 7.375, 3.75, 12.0, 2.625, 8.375, 6.75, 2.5, 8.75, 10.5, 6.625, 0.125, 0.375, 0.5, 0.75, 8.25, 4.125, 1.0, 1.0, 0.125, 0.25, 4.625, 6.375, 9.25, 2.875, 0.375, 4.125, 4.25, 10.9375, 11.5625, 10.625, 10.5625, 9.5625, 11.5, 10.6875, 0.5, 2.25, 4.875, 2.125, 1.25, 0.125, 0.5, 0.625, 0.625, 4.25, 1.125, 2.875, 3.125, 3.125, 4.125, 0.75, 1.375, 1.75, 0.75, 1.0, 4.25, 1.5, 0.5, 1.125, 0.625, 4.625, 1.625, 0.25, 4.5, 1.0, 0.0, 1.25, 8.5, 5.0, 1.125, 2.75, 3.375, 0.625, 0.125, 1.125, 1.875, 7.25, 1.875, 2.125, 0.25, 0.875, 6.25, 1.875, 1.625, 2.625, 2.875, 2.125, 2.375, 8.125, 10.9375, 10.0625, 9.0625, 9.875, 9.375, 11.25, 11.375, 11.875, 12.9375], \"error\": null, \"prompt_sha\": \"8df9ed2c640ef5bd\", \"prompt_len\": 2048, \"finish_reason\": \"length\"}, {\"text\": \"We need solve ARC tasks. Need infer transformation rules from examples and apply to test inputs. We have two tasks: 1818057f and 195c6913. Need output transformed test grids. Need analyze carefully.\\n\\nWe need produce final answer likely grids. Need infer rules.\\n\\nTask 1818057f:\\nExamples: inputs/outputs with colors 2,4,8. Output changes some 4s to 8? Let's inspect.\\n\\nExample1 input 10x12? rows:\\n1 22224\", \"tokens\": [1596, 1144, 11290, 67964, 8806, 13, 14235, 22846, 17454, 5426, 494, 9989, 321, 3668, 310, 1228, 10804, 13, 1165, 599, 1330, 8806, 25, 220, 16, 23, 16, 23, 15, 20, 22, 69, 321, 220, 16, 24, 20, 66, 21, 24, 16, 18, 13, 14235, 2468, 22773, 1228, 54455, 13, 14235, 22903, 15060, 13, 271, 1596, 1144, 7936, 1534, 4087, 4222, 54455, 13, 14235, 22846, 5426, 13, 271, 6065, 220, 16, 23, 16, 23, 15, 20, 22, 69, 25, 198, 39044, 25, 10804, 14, 39652, 440, 7736, 220, 17, 11, 19, 11, 23, 13, 8984, 4203, 1010, 220, 19, 82, 310, 220, 23, 30, 6558, 579, 23331, 13, 271, 12929, 16, 1879, 220, 16, 15, 87, 16, 17, 30, 6761, 25, 198, 16, 220, 17, 17, 17, 17, 19], \"margins\": [11.875, 11.375, 0.75, 3.375, 2.625, 3.0, 3.25, 0.5, 3.125, 0.25, 0.625, 4.75, 1.125, 3.25, 2.0, 5.5, 0.875, 3.75, 0.375, 0.5, 1.375, 6.5, 1.125, 2.25, 12.625, 12.0625, 11.3125, 12.9375, 12.4375, 8.875, 10.8125, 10.75, 1.75, 7.375, 13.5625, 13.875, 11.6875, 11.25, 10.75, 12.375, 12.375, 10.75, 2.25, 2.25, 1.625, 0.875, 1.0, 0.375, 0.875, 1.375, 0.125, 1.875, 6.5, 1.5, 1.0, 1.375, 1.0, 4.75, 4.625, 1.75, 1.0, 1.625, 0.875, 0.625, 0.25, 3.5, 1.0, 1.25, 4.625, 11.625, 5.75, 11.125, 12.3125, 10.8125, 10.9375, 11.0, 11.8125, 0.125, 0.0, 2.625, 0.375, 0.25, 0.875, 7.875, 0.75, 0.875, 1.875, 8.125, 0.75, 7.375, 2.875, 3.875, 2.125, 0.25, 2.0, 3.25, 2.625, 0.625, 2.625, 2.125, 10.75, 10.625, 0.75, 1.75, 7.5, 2.75, 6.375, 0.25, 1.75, 1.375, 0.75, 3.25, 8.375, 4.625, 3.75, 4.375, 2.625, 1.5, 0.25, 0.5, 5.375, 1.375, 0.5, 9.75, 4.625, 12.625, 11.4375, 8.375], \"error\": null, \"prompt_sha\": \"8750743e9dcadca9\", \"prompt_len\": 2048, \"finish_reason\": \"length\"}, {\"text\": \"We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We have multiple tasks: 247ef758, 269e22fb, 271d71e2, 28a6681f, 291dc1e1. Need output transformed grids for test inputs. Need infer rules from examples. Need be careful. We need produce final answer likely grids. Need analyze each task.\\n\\nWe need parse tasks. There are two test inputs for 247ef758? It says Test input: ...\", \"tokens\": [1596, 1144, 11290, 67964, 8806, 13, 14235, 1423, 17454, 5728, 364, 1754, 3274, 321, 11290, 1228, 10804, 13, 1165, 599, 5081, 8806, 25, 220, 17, 19, 22, 797, 22, 20, 23, 11, 220, 17, 21, 24, 68, 17, 17, 10481, 11, 220, 17, 22, 16, 67, 22, 16, 68, 17, 11, 220, 17, 23, 64, 21, 21, 23, 16, 69, 11, 220, 17, 24, 16, 7385, 16, 68, 16, 13, 14235, 2468, 22773, 54455, 364, 1228, 10804, 13, 14235, 22846, 5426, 494, 9989, 13, 14235, 381, 16097, 13, 1165, 1144, 7936, 1534, 4087, 4222, 54455, 13, 14235, 22903, 1754, 3274, 13, 271, 1596, 1144, 4563, 8806, 13, 2532, 513, 1330, 1228, 10804, 364, 220, 17, 19, 22, 797, 22, 20, 23, 30, 1049, 2640, 3284, 1879, 25, 2423], \"margins\": [11.75, 9.875, 1.125, 1.25, 3.875, 2.0, 2.875, 0.375, 5.25, 0.875, 1.5, 5.625, 0.875, 3.0, 2.875, 3.125, 7.75, 6.75, 0.5, 0.5, 1.5, 5.625, 1.375, 3.0, 11.75, 13.375, 12.625, 11.3125, 12.0, 13.5, 12.625, 0.625, 8.4375, 11.625, 11.0, 12.5625, 10.5625, 11.0, 10.375, 8.375, 7.375, 9.75, 11.5625, 9.125, 10.0, 9.75, 9.4375, 10.3125, 10.4375, 9.0, 8.1875, 9.4375, 12.625, 9.75, 11.0, 11.3125, 10.9375, 9.625, 9.5625, 10.0, 8.1875, 7.625, 12.375, 11.125, 11.375, 10.625, 10.25, 10.375, 10.0, 2.5, 3.0, 2.125, 0.125, 0.125, 1.875, 0.125, 4.125, 3.125, 2.0, 0.75, 3.0, 0.75, 8.75, 4.125, 0.125, 1.25, 3.875, 2.75, 0.125, 0.25, 0.625, 3.875, 1.625, 1.875, 0.75, 1.0, 0.375, 2.375, 2.125, 0.5, 0.5, 2.5, 1.875, 0.125, 0.75, 0.375, 3.625, 0.125, 1.375, 0.0, 4.125, 7.375, 2.5, 0.375, 12.75, 6.75, 12.375, 4.625, 8.75, 12.1875, 10.25, 2.0, 0.625, 0.25, 1.125, 7.625, 1.75, 2.125], \"error\": null, \"prompt_sha\": \"fc63e297d382cd8f\", \"prompt_len\": 8192, \"finish_reason\": \"length\"}, {\"text\": \"We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We need output likely transformed grids. Need analyze many tasks. Need be careful. We have multiple tasks with examples and test inputs. Need infer rules.\\n\\nWe need produce final answer with transformed test inputs for each task. There are tasks: 2c181942, 2d0172a1 (two test inputs?), 31f7f899, 332f06d7, 35ab12c3, 36a08778\", \"tokens\": [1596, 1144, 11290, 67964, 8806, 13, 14235, 1423, 17454, 5728, 364, 1754, 3274, 321, 11290, 1228, 10804, 13, 1165, 1144, 2468, 4222, 22773, 54455, 13, 14235, 22903, 1599, 8806, 13, 14235, 381, 16097, 13, 1165, 599, 5081, 8806, 440, 9989, 321, 1228, 10804, 13, 14235, 22846, 5426, 13, 271, 1596, 1144, 7936, 1534, 4087, 440, 22773, 1228, 10804, 364, 1754, 3274, 13, 2532, 513, 8806, 25, 220, 17, 66, 16, 23, 16, 24, 19, 17, 11, 220, 17, 67, 15, 16, 22, 17, 64, 16, 318, 19186, 1228, 10804, 66926, 220, 18, 16, 69, 22, 69, 23, 24, 24, 11, 220, 18, 18, 17, 69, 15, 21, 67, 22, 11, 220, 18, 20, 365, 16, 17, 66, 18, 11, 220, 18, 21, 64, 15, 23, 22, 22, 23], \"margins\": [10.125, 11.5, 1.125, 0.25, 3.5, 2.125, 2.625, 0.5, 5.25, 1.0, 1.75, 5.75, 0.375, 2.375, 3.625, 4.25, 6.625, 7.375, 0.25, 0.125, 1.25, 0.5, 0.25, 2.0, 0.5, 1.625, 1.0, 1.0, 4.375, 2.875, 0.75, 2.125, 4.125, 4.0, 0.375, 0.5, 1.625, 3.875, 0.375, 2.125, 2.125, 2.125, 2.875, 1.625, 2.875, 1.125, 0.75, 1.375, 0.125, 1.625, 2.25, 1.0, 2.875, 2.5, 3.0, 0.125, 1.125, 0.75, 1.25, 2.375, 1.75, 1.625, 0.625, 3.0, 0.125, 4.5, 0.5, 11.625, 10.625, 12.0625, 13.0625, 9.5625, 10.9375, 10.375, 10.875, 1.0, 6.5, 9.8125, 10.25, 11.9375, 13.5, 12.3125, 11.625, 11.25, 11.25, 0.375, 2.5, 4.125, 7.25, 0.625, 5.375, 9.75, 9.625, 12.4375, 11.25, 10.4375, 9.0, 11.4375, 10.8125, 3.25, 8.375, 11.875, 7.375, 11.5, 11.0, 10.5625, 8.8125, 10.875, 10.375, 5.625, 10.125, 13.125, 8.625, 7.375, 11.1875, 10.5, 10.25, 10.9375, 5.25, 9.75, 10.1875, 7.25, 11.4375, 12.0, 8.9375, 11.0625, 9.5, 10.3125], \"error\": null, \"prompt_sha\": \"49b3958bb234d4fd\", \"prompt_len\": 40000, \"finish_reason\": \"length\"}], \"rc2_fp8\": [{\"text\": \"We need solve ARC task. Need infer transformation from examples. User gives Task 0934a4d8. Example 1 input 30x30? output 9x4? Example 2 input 30x30? output 4x5? Example 3 input: \\\"194499276\\\" only one row? Maybe truncated? Need find transformation rule and solve test inputs. But only one test input? It says Example 3 input: 194499276. Maybe that's the test input? Need output. Need infer from\", \"tokens\": [1596, 1144, 11290, 67964, 3274, 13, 14235, 22846, 17454, 494, 9989, 13, 2570, 6486, 5261, 220, 15, 24, 18, 19, 64, 19, 67, 23, 13, 12996, 220, 16, 1879, 220, 18, 15, 87, 18, 15, 30, 2468, 220, 24, 87, 19, 30, 12996, 220, 17, 1879, 220, 18, 15, 87, 18, 15, 30, 2468, 220, 19, 87, 20, 30, 12996, 220, 18, 1879, 25, 328, 16, 24, 19, 19, 24, 24, 17, 22, 21, 1, 1132, 799, 2713, 30, 10380, 57739, 30, 14235, 1423, 17454, 5728, 321, 11290, 1228, 10804, 13, 1921, 1132, 799, 1228, 1879, 30, 1049, 2640, 12996, 220, 18, 1879, 25, 220, 16, 24, 19, 19, 24, 24, 17, 22, 21, 13, 10380, 421, 579, 279, 1228, 1879, 30, 14235, 2468, 13, 14235, 22846, 494], \"margins\": [11.5, 11.25, 0.25, 3.625, 0.375, 2.25, 2.25, 0.875, 4.125, 0.25, 4.25, 1.25, 0.125, 1.0, 1.75, 4.625, 9.0, 9.9375, 10.375, 9.5, 8.875, 10.6875, 14.0, 10.625, 0.125, 0.75, 0.875, 7.5, 4.5, 1.25, 3.5, 0.75, 0.0, 1.75, 4.0, 2.625, 1.25, 9.375, 5.5, 4.25, 2.0, 0.25, 2.625, 0.875, 8.875, 3.75, 4.375, 5.75, 5.5, 4.25, 3.5, 5.5, 0.0, 8.0, 7.25, 7.375, 4.625, 5.875, 1.5, 3.75, 4.875, 9.125, 6.75, 0.25, 0.625, 8.25, 8.25, 12.3125, 11.5, 10.3125, 11.0, 10.1875, 11.625, 9.875, 4.375, 2.25, 0.375, 1.0, 4.875, 0.0, 0.25, 5.75, 2.125, 0.0, 0.25, 2.375, 2.875, 6.75, 9.125, 5.125, 1.875, 0.75, 2.375, 0.0, 0.875, 1.875, 2.125, 0.75, 2.25, 1.625, 9.1875, 7.125, 9.125, 5.625, 1.125, 7.9375, 7.0625, 10.375, 11.375, 10.8125, 10.5625, 11.1875, 11.5, 11.625, 1.0, 1.625, 1.375, 2.625, 0.5, 2.375, 1.75, 2.625, 2.0, 2.625, 0.125, 1.5, 0.25, 1.5], \"error\": null, \"prompt_sha\": \"8df9ed2c640ef5bd\", \"prompt_len\": 2048, \"finish_reason\": \"length\"}, {\"text\": \"We need solve ARC tasks. Need infer transformation rules from examples and apply to test inputs. We have two tasks: 1818057f and 195c6913. Need output transformed test grids. Need analyze carefully.\\n\\nWe need produce final answer likely grids. Need infer rules.\\n\\nTask 1818057f examples:\\nInput/output grids with colors 2,4,8. Output changes some 4s to 8? Let's inspect.\\n\\nExample1 10x12? Input rows:\\n1 222244\", \"tokens\": [1596, 1144, 11290, 67964, 8806, 13, 14235, 22846, 17454, 5426, 494, 9989, 321, 3668, 310, 1228, 10804, 13, 1165, 599, 1330, 8806, 25, 220, 16, 23, 16, 23, 15, 20, 22, 69, 321, 220, 16, 24, 20, 66, 21, 24, 16, 18, 13, 14235, 2468, 22773, 1228, 54455, 13, 14235, 22903, 15060, 13, 271, 1596, 1144, 7936, 1534, 4087, 4222, 54455, 13, 14235, 22846, 5426, 13, 271, 6065, 220, 16, 23, 16, 23, 15, 20, 22, 69, 9989, 25, 198, 2425, 45447, 54455, 440, 7736, 220, 17, 11, 19, 11, 23, 13, 8984, 4203, 1010, 220, 19, 82, 310, 220, 23, 30, 6558, 579, 23331, 13, 271, 12929, 16, 220, 16, 15, 87, 16, 17, 30, 5396, 6761, 25, 198, 16, 220, 17, 17, 17, 17, 19, 19], \"margins\": [11.875, 11.75, 0.5, 3.125, 2.625, 3.0, 3.125, 0.5, 3.0, 0.5, 0.625, 4.875, 1.0, 3.25, 2.0, 5.625, 0.625, 3.875, 0.375, 0.75, 1.375, 6.75, 1.375, 2.125, 12.8125, 11.9375, 11.6875, 13.1875, 12.0625, 8.4375, 10.3125, 10.5, 1.625, 7.25, 13.5625, 12.625, 10.5625, 9.375, 12.0, 12.6875, 11.3125, 11.0625, 2.0, 2.375, 1.625, 0.875, 1.0, 0.125, 0.5, 1.25, 0.125, 1.875, 6.625, 1.625, 1.5, 1.5, 1.0, 4.75, 4.625, 2.0, 0.875, 1.5, 1.0, 0.625, 0.5, 3.375, 1.0, 1.125, 4.5, 12.375, 5.625, 11.125, 12.625, 10.0625, 9.875, 10.125, 11.625, 0.125, 1.375, 0.5, 1.0, 0.75, 0.375, 1.25, 1.0, 1.75, 7.75, 0.75, 7.75, 2.375, 3.625, 1.625, 0.25, 2.0, 3.5, 2.75, 1.125, 2.25, 2.375, 10.8125, 10.75, 0.75, 1.875, 7.5, 2.75, 5.25, 0.25, 2.125, 1.25, 0.0, 7.125, 3.625, 5.5, 4.5, 2.0, 1.5, 0.75, 0.875, 0.0, 6.375, 1.0, 0.75, 10.0, 4.75, 12.625, 11.5625, 8.375, 11.4375], \"error\": null, \"prompt_sha\": \"8750743e9dcadca9\", \"prompt_len\": 2048, \"finish_reason\": \"length\"}, {\"text\": \"We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We have multiple tasks: 247ef758, 269e22fb, 271d71e2, 28a6681f, 291dc1e1. Need output transformed test inputs. Need infer rules from examples. Need be careful. We can analyze each.\\n\\nWe need produce final answer likely grids for each test input. Need determine exact transformations.\\n\\nLet's parse tasks.\\n\\nTask 247ef758: examples show\", \"tokens\": [1596, 1144, 11290, 67964, 8806, 13, 14235, 1423, 17454, 5728, 364, 1754, 3274, 321, 11290, 1228, 10804, 13, 1165, 599, 5081, 8806, 25, 220, 17, 19, 22, 797, 22, 20, 23, 11, 220, 17, 21, 24, 68, 17, 17, 10481, 11, 220, 17, 22, 16, 67, 22, 16, 68, 17, 11, 220, 17, 23, 64, 21, 21, 23, 16, 69, 11, 220, 17, 24, 16, 7385, 16, 68, 16, 13, 14235, 2468, 22773, 1228, 10804, 13, 14235, 22846, 5426, 494, 9989, 13, 14235, 381, 16097, 13, 1165, 628, 22903, 1754, 13, 271, 1596, 1144, 7936, 1534, 4087, 4222, 54455, 364, 1754, 1228, 1879, 13, 14235, 7995, 4581, 51192, 13, 271, 9764, 579, 4563, 8806, 13, 271, 6065, 220, 17, 19, 22, 797, 22, 20, 23, 25, 9989, 1420], \"margins\": [11.875, 9.75, 1.25, 1.5, 3.625, 2.125, 3.0, 0.125, 5.25, 0.875, 1.625, 5.5, 0.875, 3.125, 2.875, 2.875, 8.5, 7.125, 0.375, 0.375, 1.875, 5.375, 1.625, 3.0, 12.125, 13.3125, 12.9375, 11.8125, 11.8125, 13.5, 12.5, 0.375, 8.875, 11.6875, 10.9375, 12.25, 11.0625, 10.6875, 10.375, 8.3125, 7.375, 10.4375, 11.1875, 9.0, 9.8125, 10.125, 10.25, 10.8125, 10.125, 9.3125, 8.0625, 9.5, 12.4375, 10.4375, 11.0, 11.1875, 11.1875, 9.6875, 10.375, 9.0, 7.75, 7.625, 12.875, 10.8125, 11.375, 9.625, 10.625, 11.0625, 9.875, 2.375, 2.875, 2.125, 0.125, 0.0, 0.875, 0.75, 2.0, 0.625, 3.125, 0.875, 8.375, 3.875, 0.125, 1.25, 4.125, 2.875, 0.375, 0.125, 0.5, 0.125, 0.875, 1.375, 2.5, 2.25, 0.75, 4.0, 1.75, 2.0, 0.875, 0.625, 1.125, 3.375, 4.25, 3.75, 0.875, 0.125, 0.5, 0.875, 6.25, 1.5, 3.125, 10.25, 2.625, 2.25, 3.25, 3.625, 3.625, 6.75, 10.375, 7.625, 12.5, 11.75, 11.25, 11.875, 13.4375, 1.5, 0.0, 0.25], \"error\": null, \"prompt_sha\": \"fc63e297d382cd8f\", \"prompt_len\": 8192, \"finish_reason\": \"length\"}, {\"text\": \"We need solve ARC tasks. Need find transformation rule for each task and solve test inputs. We have multiple tasks with examples and test inputs. Need output transformed grids likely. Need infer rules. This is complex. Need analyze each task carefully. Need produce final answer with outputs for each test input. Need be accurate.\\n\\nWe have tasks:\\n2c181942 (one test input)\\n2d0172a1 (two test inputs? It lists Test input twice)\\n31f7f899 (one test)\\n332f06d7 (one test)\", \"tokens\": [1596, 1144, 11290, 67964, 8806, 13, 14235, 1423, 17454, 5728, 364, 1754, 3274, 321, 11290, 1228, 10804, 13, 1165, 599, 5081, 8806, 440, 9989, 321, 1228, 10804, 13, 14235, 2468, 22773, 54455, 4222, 13, 14235, 22846, 5426, 13, 1061, 369, 6150, 13, 14235, 22903, 1754, 3274, 15060, 13, 14235, 7936, 1534, 4087, 440, 15796, 364, 1754, 1228, 1879, 13, 14235, 381, 12995, 13, 271, 1596, 599, 8806, 25, 198, 17, 66, 16, 23, 16, 24, 19, 17, 318, 588, 1228, 1879, 8, 198, 17, 67, 15, 16, 22, 17, 64, 16, 318, 19186, 1228, 10804, 30, 1049, 11140, 3284, 1879, 10598, 8, 198, 18, 16, 69, 22, 69, 23, 24, 24, 318, 588, 1228, 8, 198, 18, 18, 17, 69, 15, 21, 67, 22, 318, 588, 1228, 8], \"margins\": [10.0, 11.25, 1.0, 0.625, 3.375, 2.5, 2.875, 0.375, 5.125, 0.875, 1.75, 5.875, 0.375, 2.875, 3.625, 4.125, 6.375, 7.5, 0.25, 0.0, 0.25, 6.25, 0.25, 3.75, 2.0, 2.625, 3.125, 2.875, 4.125, 2.75, 0.5, 1.375, 0.625, 0.5, 1.875, 0.125, 1.375, 0.25, 0.0, 9.625, 0.125, 1.375, 1.0, 0.625, 3.125, 4.125, 0.25, 1.875, 0.125, 0.875, 3.875, 1.5, 2.75, 0.0, 2.5, 0.875, 3.0, 4.75, 1.25, 1.0, 0.125, 1.75, 10.625, 1.875, 2.625, 0.125, 3.0, 3.5, 1.125, 0.25, 10.5, 13.375, 13.3125, 12.0625, 10.375, 9.625, 11.0, 0.125, 1.625, 7.5, 1.0, 3.5, 7.75, 7.0, 10.375, 13.0625, 13.1875, 12.0, 11.1875, 11.875, 11.1875, 7.125, 5.75, 5.25, 8.5, 0.625, 0.75, 0.75, 1.375, 7.375, 0.75, 0.0, 4.375, 8.9375, 9.875, 12.8125, 11.8125, 10.6875, 10.1875, 10.5625, 12.125, 4.625, 5.25, 1.125, 0.0, 10.875, 11.6875, 6.875, 13.0, 11.125, 12.0, 10.0625, 11.375, 10.5, 9.25, 9.0, 2.625, 5.25], \"error\": null, \"prompt_sha\": \"49b3958bb234d4fd\", \"prompt_len\": 40000, \"finish_reason\": \"length\"}]}, \"prompt_shas\": {\"8750743e9dcadca9\": 2048, \"cfc35f3fcb1e4c3c\": 40000, \"8df9ed2c640ef5bd\": 2048, \"fc63e297d382cd8f\": 8192, \"49b3958bb234d4fd\": 40000, \"23e0dd6cf518ffd6\": 40000, \"860a315bad6a77cd\": 40000, \"d9b32f9b8dccc17e\": 40000, \"cde642a1f2a6c19b\": 40000, \"f4212f34e7a2cc3a\": 8192, \"452f4eeeffdffcba\": 8192, \"cabba475611f1ca9\": 8192, \"b3074e57ef93918d\": 8192, \"e5f0513c6123c379\": 8192, \"5c55ca1649204598\": 8192, \"2395ccdb274f065a\": 8192, \"29ccfbd3ae41464d\": 90000, \"6d6f401b8828e42d\": 90000, \"0fc6beedaad76444\": 90000, \"44546bcef138a34a\": 90000, \"057ef8fde4806648\": 116000, \"6e42532a0cbaf769\": 116000, \"eced77cf1dce0ac9\": 116000, \"f581c9630300ae10\": 116000, \"d445d5b5d3d7c21c\": 4894}}")       # RC2 GPU v1 facts (argv / env / TF anchor / oracle rows / prompt shas), build-time
    QSA_PATCH = json.loads("{\"ops_rel\": \"vllm/models/qwen4_exp/nvidia/ops/qsa.py\", \"qsa_rel\": \"vllm/models/qwen4_exp/nvidia/qsa.py\", \"ops_sha\": \"23c01a2069097129ae927bdf4623a3837b7403d4a1cad548505769c05eb6792d\", \"qsa_sha\": \"a23d5f90507ad5185f8b8e8d6ba13ae0cb53b2506264fdfbbf6a60b1171dc653\", \"ops_old\": \"    capability = current_platform.get_device_capability()\\n    return capability is not None and capability.to_int() >= 89\\n\", \"ops_new\": \"    if __import__(\\\"os\\\").environ.get(\\\"NB0_QSA_FORCE_BYTE_DECODER\\\") == \\\"1\\\":  # NB0-E25 boot E: Merlin's sm_86 byte decoder on sm_89\\n        print(f\\\"NB0 boot E: QSA E4M3 byte decoder forced (qsa_e4m3_uses_native_cast -> False) pid={__import__('os').getpid()}\\\", flush=True)\\n        return False\\n    capability = current_platform.get_device_capability()\\n    return capability is not None and capability.to_int() >= 89\\n\", \"qsa_old\": \"        capability.to_int() != 86 and not qsa_e4m3_uses_native_cast()\\n    ):\\n\", \"qsa_new\": \"        capability.to_int() != 86 and not qsa_e4m3_uses_native_cast()\\n        and not (__import__(\\\"os\\\").environ.get(\\\"NB0_QSA_FORCE_BYTE_DECODER\\\") == \\\"1\\\" and capability.to_int() >= 89)  # NB0-E25 boot E\\n    ):\\n\", \"marker\": \"NB0-E25 boot E\", \"audit\": {\"ops\": [\"@lru_cache(maxsize=1)\\ndef qsa_e4m3_uses_native_cast() -> bool:\\n\", \"        if qsa_e4m3_uses_native_cast():\\n            # SM89+: upstream's native E4M3 path (vllm#55557).\\n            kv_dtype = 3\\n\", \"        else:\\n            kv_dtype = 1\\n\", \"        if qsa_e4m3_uses_native_cast():\\n            warm_kv_dtype = 3\\n\", \"            keys = _decode_e4m3_to_bf16(key_bytes)\\n\", \"            values = _decode_e4m3_to_bf16(value_bytes)\\n\"], \"qsa\": [\"from .ops.qsa import qsa_e4m3_uses_native_cast\\n\", \"            and (not fp8_layer or qsa_e4m3_uses_native_cast())\\n\", \"        if not self.use_fused_qsa_prepare:\\n\", \"            impl.do_kv_cache_update(\\n\"], \"name_count\": {\"ops\": 3, \"qsa\": 3}}}")   # boot E: audited Merlin ad42e37 source shas, anchors and the env-gated patch
    NB0 = {"topk": int(os.environ.get("NB0_TF_TOPK") or 5), "sc_tok": int(os.environ.get("NB0_SC_TOKENS") or 256),
           "tf_variant": 5, "sc_variant": {"2k": 6, "8k": 7, "40k": 6}, "tf_bands": ["2k", "8k", "40k", "90k"], "sc_bands": ["2k", "8k", "40k"],
           "last": 256, "boot_watchdog_sec": float(os.environ.get("NB0_BOOT_WATCHDOG_SEC") or 1800),
           "ready_sec": float(os.environ.get("NB0_READY_SEC") or 1500), "host_floor_mib": (69 + 4) * 1024}
    SCREEN = {"rank1_min": 0.90, "mean_max": 0.15}       # external audit 2 (verified): conservative screening, not a published limit
    TRIAGE = {"mean_max": 1.0, "rank1_min": 0.5}         # external audit 3 (verified): severe-discrepancy triage
    _FP8_FLAGS = None


    def flags(argv):
        """--flag -> value, the LAST occurrence wins (argparse); a flag followed by another flag is a switch (True)."""
        out = {}
        for i, a in enumerate(argv):
            if isinstance(a, str) and a.startswith("--"):
                nxt = argv[i + 1] if i + 1 < len(argv) else None
                out[a] = nxt if (nxt is not None and not str(nxt).startswith("--")) else True
        return out


    PORT_BASE = int(os.environ.get("NB0_PORT_BASE") or 8010)       # Kaggle 8010-8014 (Merlin's 8000-8355 port scheme)
    FP8_CAPTURES = flags(RC2_REF["fp8"]["argv_tail"])["--compilation-config"]     # RC2 FP8's exact JSON (captures to 32)
    BOOTS = {
        "D": {"name": "nb0_D_fp8_mtp3", "kv": "fp8", "mtp": True, "port": PORT_BASE + 0, "seqs": None, "captures": None, "byte": False,
              "title": "FP8 KV, MTP K3 (= RC2 FP8)"},
        "C": {"name": "nb0_C_bf16_mtp3", "kv": "bf16", "mtp": True, "port": PORT_BASE + 1, "seqs": None, "captures": None, "byte": False,
              "title": "BF16 KV, MTP K3 (= RC2 BF16)"},
        "E": {"name": "nb0_E_fp8_mtp3_bytedec", "kv": "fp8", "mtp": True, "port": PORT_BASE + 2, "seqs": None, "captures": None,
              "byte": True, "title": "FP8 KV, MTP K3, Merlin sm_86 byte decoder forced on sm_89"},
        "B": {"name": "nb0_B_fp8_mtp0", "kv": "fp8", "mtp": False, "port": PORT_BASE + 3, "seqs": None, "captures": None, "byte": False,
              "title": "FP8 KV, MTP off (target only)"},
        "A": {"name": "nb0_A_bf16_mtp0", "kv": "bf16", "mtp": False, "port": PORT_BASE + 4, "seqs": 8, "captures": FP8_CAPTURES,
              "byte": False, "title": "BF16 KV, MTP off (target only), seqs 8 + FP8 captures"},
    }
    for _k, _b in BOOTS.items():
        _b["key"] = _k
        _b["step"] = f"{_k}_{_b['name']}"
        _b["want_seqs"] = _b["seqs"] or RC2_REF[_b["kv"]]["seqs"]
    BOOT_BY_NAME = {b["name"]: b for b in BOOTS.values()}
    # Order (decided, see the report): D first = the production candidate on the RC2-proven config (its own A/A,
    # self-consistency, shift and chunk tests settle "path artefact?" alone); C next = the RC2 pair with raw arrays; E third
    # = the decisive read-path test (lead: priority over A's 90k band); the two never-run MTP-off boots last, B before A
    # (B-D is the MTP effect on the production KV dtype; if B cannot launch, A is skipped: same MTP-off path).
    ORDER = ["D", "C", "E", "B", "A"]
    EST_MIN = {"D": 15, "C": 13, "E": 15, "B": 12, "A": 12}     # step() needs est*60 + 300 s left
    PAIRS = {"AB": ("A", "B", "KV dtype only, target only (MTP off)"),
             "CD": ("C", "D", "KV dtype under MTP K3 (the RC2 pair)"),
             "AC": ("A", "C", "MTP effect, BF16 KV"),
             "BD": ("B", "D", "MTP effect, FP8 KV"),
             "CE": ("C", "E", "KV dtype under MTP K3, FP8 read through Merlin's byte decoder"),
             "DE": ("D", "E", "FP8 read path: sm_89 native cast vs byte decoder")}
    ORACLE_REFS = {"D": ("rc2_fp8", "rc2_bf16"), "C": ("rc2_bf16", "rc2_fp8", "D"), "E": ("rc2_fp8", "D", "C"),
                   "B": ("rc2_fp8", "D", "C"), "A": ("rc2_bf16", "C", "B", "D")}
    for _k, _rows in RC2_REF["oracle_rows"].items():
        ROWS[_k] = _rows                    # RC2's recorded greedy rows (2k v0, 2k v1, 8k v2, 40k v3; 128 tokens)
    RAW_DIR = OUT / "tf_raw"
    RAW_DIR.mkdir(exist_ok=True)
    NB0_REQ = TRACES / "nb0_requests.jsonl"         # every non-streaming request (teacher-forced, decode) with its full text
    TFR, SCR, LOADMON, WD, SIDS, DROP = {}, {}, {}, {}, set(), set()
    PROC = Path(os.environ.get("NB0_PROC", "/proc"))
    CG = Path(os.environ.get("NB0_CGROUP", "/sys/fs/cgroup"))
    SHM0 = shm_names()                               # /dev/shm at NB0 start: anything later and psm_* is ours
    ENV_KEYS = ("VLLM_", "NCCL_", "KV_", "P2P", "SCALES", "NB0_")


    def _set_value(a, flag, value):
        idx = [i for i, x in enumerate(a) if x == flag]
        if len(idx) != 1:
            raise ValueError(f"{flag} occurs {len(idx)} times in the Merlin argv (expected 1)")
        a[idx[0] + 1] = value


    def nb0_argv(argv, boot):
        """Merlin's argv with this boot's edits. MTP off = the same command without --speculative-config (Merlin
        models/flash-next/CHANGELOG.md: "the no-MTP line is the same command without --speculative-config"; launch.py:441/452
        always emits it, R.MTP_K = 3 has no env knob). seqs / captures: the value of the existing flag (no new flags)."""
        a = [str(x) for x in argv]
        if not boot["mtp"]:
            idx = [i for i, x in enumerate(a) if x == "--speculative-config"]
            if len(idx) != 1:
                raise ValueError(f"--speculative-config occurs {len(idx)} times (expected 1)")
            del a[idx[0]:idx[0] + 2]
        if boot.get("seqs"):
            _set_value(a, "--max-num-seqs", str(boot["seqs"]))
        if boot.get("captures"):
            _set_value(a, "--compilation-config", boot["captures"])
        if "--enable-prefix-caching" not in a:
            raise ValueError("prefix caching must stay on in every NB0 boot")
        return a


    def config_diff(argv, env, kv_bytes, ref="fp8"):
        """Flag / env / KV-pin diff of a boot vs RC2's recorded plan (last occurrence wins on both sides)."""
        rf, cf = flags(RC2_REF[ref]["argv_tail"]), flags(argv)      # RC2's tail starts at --host's value (127.0.0.1)
        head = set(flags(argv[:-len(RC2_REF[ref]["argv_tail"]) + 2])) - set(rf)   # Merlin flags before RC2's recorded window
        fl = {k: [rf.get(k), cf.get(k)] for k in sorted(set(rf) | set(cf))
              if k not in ("--port", "--host") and k not in head and rf.get(k) != cf.get(k)}
        renv = RC2_REF[ref]["env"]
        cenv = {k: v for k, v in env.items() if k.startswith(ENV_KEYS)}
        skip = ("VLLM_PORT", "MASTER_PORT", "NCCL_VERSION")
        ev = {k: [renv.get(k), cenv.get(k)] for k in sorted(set(renv) | set(cenv)) if k not in skip and renv.get(k) != cenv.get(k)}
        out = {"flags": fl, "env": ev}
        if kv_bytes != RC2_REF[ref]["kv_bytes"]:
            out["kv_bytes"] = [RC2_REF[ref]["kv_bytes"], kv_bytes]
        return out


    _merlin_fn_server_rc2 = merlin_fn_server


    def merlin_fn_server(name, kv, port, kv_bytes):
        """RC2's server (Merlin launch.build + host_phase, verbatim) + NB0's per-boot argv / env edits, the load monitor and
        the per-boot watchdog. FAKE: RC2's recorded Kaggle argv for this KV dtype (port / pin substituted) into the fake."""
        b = BOOT_BY_NAME[name]
        s = _merlin_fn_server_rc2(name, kv, port, kv_bytes)
        if FAKE:
            tail, out = RC2_REF[kv]["argv_tail"], []
            for i, x in enumerate(tail):
                out.append(str(port) if i and tail[i - 1] == "--port" else str(kv_bytes) if i and tail[i - 1] == "--kv-cache-memory-bytes" else x)
            s.argv = [sys.executable, FAKE, str(CKPT), "--host"] + out
            env_basis = dict(RC2_REF[kv]["env"])
            if kv == "fp8":
                s.env["VLLM_QSA_KV_SCALES"] = str(CKPT / "qsa_kv_scales_262k.json")
        s.argv = nb0_argv(s.argv, b)
        if b["byte"]:
            s.env.update(NB0_QSA_FORCE_BYTE_DECODER="1", VLLM_CACHE_ROOT=f"{TMP}/merlin-cache-bytedec")   # never D's AOT graphs
        else:
            s.env.pop("NB0_QSA_FORCE_BYTE_DECODER", None)
        if FAKE:
            env_basis.update({k: s.env[k] for k in ("NB0_QSA_FORCE_BYTE_DECODER", "VLLM_CACHE_ROOT") if b["byte"] and k in s.env})
        envd = env_basis if FAKE else s.env
        R["setup"].setdefault("nb0_plans", {})[name] = {
            "boot": b["key"], "title": b["title"], "argv_tail": s.argv[-64:], "kv_bytes": kv_bytes,
            "env": {k: envd[k] for k in sorted(envd) if k.startswith(ENV_KEYS)},
            "diff_vs_rc2_fp8": config_diff(s.argv, envd, kv_bytes, "fp8"), "diff_vs_rc2_bf16": config_diff(s.argv, envd, kv_bytes, "bf16")}
        save()
        deadline = time.time() + max(120, min(NB0["boot_watchdog_sec"], left() - 240))
        threading.Thread(target=load_monitor, args=(s, b["key"], deadline), daemon=True).start()
        threading.Thread(target=boot_watchdog, args=(s, b["key"], deadline), daemon=True).start()
        return s


    # ---------------------------------------------------------------- host memory, processes, page cache (new)
    def cg_snapshot(full=True):
        """cgroup v2 memory files (MiB for sizes). avail_mib = max - (current - inactive_file), Merlin's own rule."""
        o, kv = {}, {}
        for n in ("memory.current", "memory.max", "memory.peak"):
            try:
                o[n] = (CG / n).read_text().strip()
            except Exception:
                pass
        try:
            st = (CG / "memory.stat").read_text()
            kv = dict(l.split() for l in st.splitlines() if len(l.split()) == 2)
            o["stat_mib"] = {k: int(kv[k]) >> 20 for k in ("anon", "file", "active_file", "inactive_file", "shmem", "file_mapped",
                                                           "file_dirty", "file_writeback") if k in kv}
            o["stat_count"] = {k: int(kv[k]) for k in ("pgmajfault", "workingset_refault_file", "pgscan", "pgsteal") if k in kv}
            if full:
                o["memory.stat"] = st[:6000]
        except Exception:
            pass
        for n in ("memory.events", "memory.pressure"):
            try:
                o[n] = (CG / n).read_text().strip()[:800]
            except Exception:
                pass
        try:
            mx = o.get("memory.max")
            if mx and mx != "max":
                o["avail_mib"] = (int(mx) - (int(o["memory.current"]) - int(kv.get("inactive_file", 0)))) >> 20
        except Exception:
            pass
        return o


    def _pids():
        try:
            return [int(p.name) for p in PROC.iterdir() if p.name.isdigit()]
        except Exception:
            return None


    def _stat(pid):
        try:
            t = (PROC / str(pid) / "stat").read_text()
        except Exception:
            return None
        i = t.rfind(")")
        f = t[i + 2:].split()
        try:
            return {"pid": pid, "comm": t[t.find("(") + 1:i], "state": f[0], "ppid": int(f[1]), "session": int(f[3]),
                    "minflt": int(f[7]), "majflt": int(f[9]), "utime": int(f[11]), "stime": int(f[12]), "rss_pages": int(f[21])}
        except Exception:
            return None


    def _io(pid):
        try:
            return {k: int(v) for k, v in (l.split(": ") for l in (PROC / str(pid) / "io").read_text().splitlines())
                    if k in ("rchar", "read_bytes", "syscr")}
        except Exception:
            return {}


    def session_procs(sids):
        out = []
        for pid in _pids() or []:
            st = _stat(pid)
            if st and st["session"] in sids and st["state"] != "Z":
                st.update(_io(pid))
                out.append(st)
        return out


    _KILL = os.kill                                  # unit tests replace it (never signal a real pid)


    def reap_leftovers(wait=30):
        """Every process of an NB0 server session (we start each server with start_new_session, so sid = server pid) must
        be gone before the next boot: SIGKILL the leftovers (PLE offload, workers), wait, re-check; fail closed."""
        if _pids() is None:
            return {"ok": bool(FAKE), "why": "no /proc" + (" (local harness)" if FAKE else "")}
        found = session_procs(SIDS)
        killed = []
        for p in found:
            try:
                _KILL(p["pid"], 9)
                killed.append([p["pid"], p["comm"]])
            except Exception as e:
                killed.append([p["pid"], p["comm"], repr(e)[:80]])
        t = time.time()
        while session_procs(SIDS) and time.time() - t < wait:
            time.sleep(1)
        still = session_procs(SIDS)
        sus = []
        for pid in _pids() or []:
            if pid == os.getpid():
                continue
            try:
                cl = (PROC / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
            except Exception:
                continue
            if re.search(r"vllm|ple_offload|EngineCore|fn_merlin", cl) and not re.search(r"ipykernel|jupyter", cl):
                sus.append([pid, cl[:160]])
        return {"ok": not still, "found": [[p["pid"], p["comm"]] for p in found][:20], "killed": killed[:20],
                "still": [[p["pid"], p["comm"]] for p in still][:20], "other_engine_processes": sus[:10], "sessions": sorted(SIDS)}


    def unlink_own_psm():
        out = {"removed": [], "kept": []}
        try:
            for p in sorted(SHM.iterdir()):
                if p.name.startswith("psm_") and p.name not in SHM0:
                    try:
                        p.unlink()
                        out["removed"].append(p.name)
                    except Exception as e:
                        out["kept"].append([p.name, repr(e)[:80]])
        except Exception as e:
            out["error"] = repr(e)[:200]
        return out


    def ckpt_files():
        fs = set()
        for d in [CKPT] + [v[0] for v in PARTS.values() if v] + ([PLE_DIR] if PLE_DIR else []):
            try:
                for p in Path(d).rglob("*"):
                    try:
                        rp = p.resolve()
                        if rp.is_file():
                            fs.add(rp)
                    except Exception:
                        pass
            except Exception:
                pass
        return sorted(fs)


    def drop_ckpt_cache():
        """posix_fadvise(DONTNEED) on every merged file's target and every file of the Model mounts (a successful call is not
        proof of reclaim: the cgroup file / inactive_file before and after are recorded by the caller)."""
        fad = getattr(os, "posix_fadvise", None)
        if fad is None:
            return {"supported": False}
        t, n, b, errs = time.time(), 0, 0, []
        for p in ckpt_files():
            try:
                fd = os.open(p, os.O_RDONLY)
                try:
                    fad(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                    n += 1
                    b += os.fstat(fd).st_size
                finally:
                    os.close(fd)
            except Exception as e:
                errs.append([str(p)[-80:], repr(e)[:80]])
        return {"supported": True, "files": n, "gib": round(b / 2 ** 30, 2), "errors": errs[:5], "sec": round(time.time() - t, 1)}


    def host_prep():
        """Before every boot: previous server + PLE workers gone, our leaked psm_* unlinked, checkpoint page cache dropped,
        Merlin's host rule max - (current - inactive_file) >= 69 + 4 GiB. Fail closed."""
        t = time.time()
        out = {"cg_before": cg_snapshot(False), "host_mem_before": host_mem()}
        out["procs"] = reap_leftovers()
        out["shm"] = unlink_own_psm() if out["procs"]["ok"] else {"skipped": "server processes still alive"}
        out["fadvise"] = drop_ckpt_cache()
        if not FAKE:
            time.sleep(3)
        out["cg_after"] = cg_snapshot(True)
        floor = NB0["host_floor_mib"]

        def avail():
            a = cg_snapshot(False).get("avail_mib")
            return (a, "cgroup") if a is not None else ((host_mem() or {}).get("MemAvailable_mib"), "MemAvailable")
        a, src = avail()
        if a is not None and a < floor:
            time.sleep(10)
            out["fadvise_2"] = drop_ckpt_cache()
            out["cg_after_2"] = cg_snapshot(False)
            a, src = avail()
        gate_ok = (a >= floor) if a is not None else False
        out["host_gate"] = {"avail_mib": a, "source": src, "floor_mib": floor, "ok": gate_ok,
                            "rule": "cgroup memory.max - (memory.current - inactive_file) >= 69 GiB (Merlin) + 4 GiB"}
        if FAKE and not gate_ok:            # local / CPU-kit harness: the fake server needs no host RAM; recorded, not applied
            out["host_gate"]["fake_not_applied"] = True
            gate_ok = True
        out["ok"] = bool(out["procs"]["ok"] and gate_ok)
        out["why"] = None if out["ok"] else ("server processes still alive" if not out["procs"]["ok"]
                                             else f"host memory {a} MiB < {floor} MiB ({src})")
        out["sec"] = round(time.time() - t, 1)
        return out


    _boot_gate_rc2 = boot_gate


    def boot_gate():
        """NB0 host preparation first, then RC2's gate (GPUs < 150 MiB each); `ok` is True only when both pass."""
        prep = host_prep()
        g = _boot_gate_rc2()
        g["host_prep"] = prep
        if prep.get("ok") is not True:
            g["ok"] = False
        return g


    def _roles(text):
        return {int(p): n for n, p in re.findall(r"\((\w+) pid=(\d+)\)", text or "")}


    def load_monitor(s, key, deadline):
        """Every 5 s from launch to ready: cgroup current / avail / stat / pressure and every process of the server session
        (majflt, minflt, rchar, read_bytes, rss, cpu) -> logs/load_monitor_<boot>.jsonl + a per-process summary."""
        path = LOGS / f"load_monitor_{key}.jsonl"
        while s.proc is None and time.time() < deadline:
            time.sleep(0.5)
        if s.proc is None:
            return
        sid = s.proc.pid
        SIDS.add(sid)
        tck = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
        t0, first, last, n = time.time(), {}, {}, 0
        peak = {"cg_current_mib_max": None, "avail_mib_min": None, "pressure_full_avg10_max": None}
        while s.alive() and s.ready_sec is None and time.time() < deadline:
            cg = cg_snapshot(False)
            ps = session_procs({sid})
            try:
                with open(path, "a") as fh:
                    fh.write(json.dumps({"t": round(time.time() - t0, 1), "current": cg.get("memory.current"),
                                         "avail_mib": cg.get("avail_mib"), "stat_mib": cg.get("stat_mib"),
                                         "stat_count": cg.get("stat_count"), "pressure": cg.get("memory.pressure"),
                                         "procs": ps}) + "\n")
            except Exception:
                pass
            for p in ps:
                first.setdefault(p["pid"], p)
                last[p["pid"]] = p
            try:
                c = int(cg["memory.current"]) >> 20
                peak["cg_current_mib_max"] = max(peak["cg_current_mib_max"] or 0, c)
            except Exception:
                pass
            if cg.get("avail_mib") is not None:
                peak["avail_mib_min"] = min(x for x in (peak["avail_mib_min"], cg["avail_mib"]) if x is not None)
            m = re.search(r"full avg10=([\d.]+)", cg.get("memory.pressure") or "")
            if m:
                peak["pressure_full_avg10_max"] = max(peak["pressure_full_avg10_max"] or 0.0, float(m.group(1)))
            n += 1
            time.sleep(0.5 if FAKE else 5)
        roles = _roles(s.log())
        per = {}
        for pid, a in first.items():
            z = last[pid]
            per[str(pid)] = {"role": roles.get(pid, a["comm"]), "majflt": z["majflt"] - a["majflt"],
                             "minflt": z["minflt"] - a["minflt"],
                             "read_gib": round((z.get("read_bytes", 0) - a.get("read_bytes", 0)) / 2 ** 30, 2),
                             "rchar_gib": round((z.get("rchar", 0) - a.get("rchar", 0)) / 2 ** 30, 2),
                             "rss_gib_end": round(z["rss_pages"] * 4096 / 2 ** 30, 2),
                             "cpu_s": round((z["utime"] + z["stime"] - a["utime"] - a["stime"]) / tck, 1)}
        loads = re.findall(r"\((\w+) pid=(\d+)\)[^\n]*Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds", s.log() or "")
        LOADMON[key] = {"samples": n, "ready_sec": s.ready_sec, "alive": s.alive(), "per_process": per, "peak": peak,
                        "model_loading": [[w, int(p), float(g), float(sec)] for w, p, g, sec in loads], "file": path.name}


    def boot_watchdog(s, key, deadline):
        """Per-boot hard bound (<= 30 min from launch, and always before the global hard stop): kill the server."""
        while time.time() < deadline:
            if s.proc is not None and not s.alive():
                return
            time.sleep(2)
        if s.alive():
            WD[key] = {"fired_at_min": round((time.time() - T_START) / 60, 1), "bound_sec": NB0["boot_watchdog_sec"]}
            log("BOOT WATCHDOG:", key, "killing the server")
            try:
                s.kill()
            except Exception as e:
                WD[key]["kill_error"] = repr(e)[:200]


    # ---------------------------------------------------------------- requests and raw arrays (new)
    def _post(port, body, timeout=1800):
        t = time.perf_counter()
        try:
            r = json.loads(UR.urlopen(UR.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                                 headers={"Content-Type": "application/json"}), timeout=timeout).read())
            return r, None, round(time.perf_counter() - t, 2)
        except UE.HTTPError as e:
            try:
                msg = e.read().decode("utf-8", "replace")[:400]
            except Exception:
                msg = ""
            return None, f"HTTP {e.code}: {msg}", round(time.perf_counter() - t, 2)
        except Exception as e:
            return None, repr(e)[:300], round(time.perf_counter() - t, 2)


    def _req_log(rec):
        try:
            with _SAVE_LOCK, open(NB0_REQ, "a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
        except Exception:
            pass


    def ids_sha(ids):
        return hashlib.sha256(json.dumps([int(x) for x in ids]).encode()).hexdigest()


    def _cached(r):
        return (((r or {}).get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")


    def pl_arrays(pl, ids, k):
        """Every prompt position p >= 1 of an OpenAI prompt_logprobs list: the actual token's logprob and rank, the unique
        rank-1 id (E23's rule: an ambiguous rank 1 is -1), the best logprob, the top-k ids / logprobs by rank. NaN is kept
        (a non-finite value is never data); a missing position is valid=False."""
        n = len(ids)
        a = {"ids": np.asarray(ids, np.int32), "lp": np.full(n, np.nan, np.float32), "rank": np.full(n, -1, np.int32),
             "top1": np.full(n, -1, np.int32), "top1_lp": np.full(n, np.nan, np.float32),
             "topk_ids": np.full((n, k), -1, np.int32), "topk_lp": np.full((n, k), np.nan, np.float32), "valid": np.zeros(n, bool)}
        amb = 0
        for pos in range(1, n):
            d = pl[pos] if pos < len(pl) else None
            if not isinstance(d, dict):
                continue
            e = d.get(str(ids[pos])) or d.get(f"token_id:{ids[pos]}")
            if not isinstance(e, dict):
                continue
            a["valid"][pos] = True
            v = e.get("logprob")
            a["lp"][pos] = float(v) if isinstance(v, (int, float)) else np.nan
            a["rank"][pos] = int(e.get("rank") or -1)
            ent = []
            for kk, vv in d.items():
                if isinstance(vv, dict) and isinstance(vv.get("logprob"), (int, float)):
                    t = L._tok_id(kk)
                    if isinstance(t, int):
                        ent.append((vv.get("rank") or 10 ** 9, -vv["logprob"], t, vv["logprob"]))
            ent.sort()
            r1 = [x for x in ent if x[0] == 1]
            if len(r1) == 1:
                a["top1"][pos] = r1[0][2]
            else:
                amb += 1
            if ent:
                a["top1_lp"][pos] = max(x[3] for x in ent)
            for j, x in enumerate([x for x in ent if x[0] <= k][:k]):
                a["topk_ids"][pos, j], a["topk_lp"][pos, j] = x[2], x[3]
        a["top1_ambiguous"] = amb
        return a


    def e23_rows(a, last=256):
        """The same per-position rows E23's teacher_forced() keeps ([logprob, rank, rank-1 id]) for the last `last` positions,
        so RC2's compare_tf (with its non-finite guard) computes the RC2-comparable metric."""
        n, out = len(a["ids"]), []
        for pos in range(max(1, n - last), n):
            if not a["valid"][pos]:
                out.append(None)
                continue
            out.append([float(a["lp"][pos]), int(a["rank"][pos]), int(a["top1"][pos]) if a["top1"][pos] >= 0 else None])
        return {"positions": out, "n": len(out)}


    def tf_request(key, port, ids, band, run, k=None):
        """One teacher-forced request (max_tokens 1, T 0, prompt_logprobs k, ids as ids). k > 1 refused by Merlin's prompt-
        scoring budget (prompt_logprobs.py validate) -> retried with k 1 = RC2's value, recorded. Raw arrays -> tf_raw/*.npz."""
        k = k or NB0["topk"]
        body = {"model": "q27", "prompt": [int(x) for x in ids], "max_tokens": 1, "temperature": 0, "prompt_logprobs": k,
                "return_tokens_as_token_ids": True}
        r, err, sec = _post(port, body)
        note = None
        if err and k > 1 and "prompt-scoring tensor bound" in err:
            note = f"prompt_logprobs {k} refused ({err[:160]}); retried with 1 (RC2's value)"
            k = body["prompt_logprobs"] = 1
            r, err, sec = _post(port, body)
        meta = {"boot": key, "band": band, "run": run, "k": k, "note": note, "sec": sec, "error": err, "prompt_len": len(ids),
                "prompt_sha256": ids_sha(ids), "cached_tokens": _cached(r), "t": round(time.time() - T_START, 1)}
        arr = None
        if r is not None:
            try:
                ch = r["choices"][0]
                pl = ch.get("prompt_logprobs") or []
                meta.update(prompt_logprobs_len=len(pl), gen_token_ids=ch.get("token_ids"), gen_text=ch.get("text"))
                arr = pl_arrays(pl, ids, k)
                vv = arr["valid"]
                meta.update(valid=int(vv.sum()), expected=len(ids) - 1, nonfinite=int((~np.isfinite(arr["lp"][vv])).sum()),
                            top1_ambiguous=arr["top1_ambiguous"])
                f = RAW_DIR / f"{key}_{band}_{run}.npz"
                np.savez_compressed(f, **{x: arr[x] for x in ("ids", "lp", "rank", "top1", "top1_lp", "topk_ids", "topk_lp", "valid")})
                meta["raw"] = f.name
            except Exception as e:
                meta["error"] = meta["error"] or f"PARSE {type(e).__name__}: {e}"[:300]
                arr = None
        _req_log({**meta, "kind": "teacher_forced", "body": {kk: v for kk, v in body.items() if kk != "prompt"}})
        return {"meta": meta, "arr": arr}


    def decode(key, port, ids, n, band, run):
        """Greedy decode (T 0, ignore_eos, logprobs k): tokens, chosen-token logprobs, rank-1 id / logprob, rank-2 logprob."""
        body = {"model": "q27", "prompt": [int(x) for x in ids], "max_tokens": n, "temperature": 0, "logprobs": NB0["topk"],
                "return_tokens_as_token_ids": True, "return_token_ids": True, "ignore_eos": True}
        r, err, sec = _post(port, body, timeout=900)
        meta = {"boot": key, "band": band, "run": run, "sec": sec, "error": err, "prompt_len": len(ids), "prompt_sha256": ids_sha(ids),
                "cached_tokens": _cached(r), "t": round(time.time() - T_START, 1)}
        toks, lps, t1, t1lp, t2lp = [], [], [], [], []
        if r is not None:
            try:
                ch = r["choices"][0]
                lp = ch.get("logprobs") or {}
                toks = ch.get("token_ids") or [L._tok_id(t) for t in (lp.get("tokens") or [])]
                lps = [x if isinstance(x, (int, float)) else float("nan") for x in (lp.get("token_logprobs") or [])]
                for top in lp.get("top_logprobs") or []:
                    it = sorted(((v, L._tok_id(kk)) for kk, v in (top or {}).items() if isinstance(v, (int, float))), reverse=True)
                    t1.append(it[0][1] if it and isinstance(it[0][1], int) else -1)
                    t1lp.append(it[0][0] if it else float("nan"))
                    t2lp.append(it[1][0] if len(it) > 1 else float("nan"))
                meta.update(tokens=len(toks), finish=ch.get("finish_reason"), text=ch.get("text"))
                if any(not isinstance(x, int) or isinstance(x, bool) for x in toks):
                    meta["error"] = "INVALID_TOKEN_IDS"
                elif len(toks) != n or len(lps) != n or len(t1) != n:
                    meta["error"] = f"INCOMPLETE_DECODE tokens {len(toks)} logprobs {len(lps)} top {len(t1)} of {n}"
            except Exception as e:
                meta["error"] = meta["error"] or f"PARSE {type(e).__name__}: {e}"[:300]
        out = {"meta": meta, "tok": [int(x) for x in toks if isinstance(x, int)], "lp": np.asarray(lps, np.float64),
               "top1": np.asarray(t1, np.int64), "top1_lp": np.asarray(t1lp, np.float64), "top2_lp": np.asarray(t2lp, np.float64)}
        if not meta["error"]:
            f = RAW_DIR / f"{key}_dec{band}_{run}.npz"
            np.savez_compressed(f, ids=np.asarray(ids, np.int32), tok=np.asarray(out["tok"], np.int32), lp=out["lp"],
                                top1=out["top1"], top1_lp=out["top1_lp"], top2_lp=out["top2_lp"])
            meta["raw"] = f.name
        _req_log({**meta, "kind": "decode", "body": {kk: v for kk, v in body.items() if kk != "prompt"}})
        return out


    # ---------------------------------------------------------------- comparisons (new; the last-256 metric is RC2's compare_tf)
    def cmp_last256(ra, rb):
        if not ra or not rb or ra.get("arr") is None or rb.get("arr") is None:
            return {"status": "INCONCLUSIVE_MISSING_TF"}
        if ra["meta"]["prompt_sha256"] != rb["meta"]["prompt_sha256"]:
            return {"status": "INCONCLUSIVE_PROMPT_MISMATCH"}
        return L.compare_tf(e23_rows(ra["arr"], NB0["last"]), e23_rows(rb["arr"], NB0["last"]))


    def chunk_check(pos, d, edge=8):
        """|dlogprob| at the first / last `edge` positions of each 1024-token scheduler chunk (--max-num-batched-tokens) and
        each 256-row prompt-logprob chunk (VLLM_MERLIN_PROMPT_LOGPROBS_CHUNK_SIZE) vs mid-chunk."""
        out = {}
        for unit in (1024, 256):
            off = pos % unit
            e = (off < edge) | (off >= unit - edge)
            if e.any() and (~e).any():
                me, mm = float(d[e].mean()), float(d[~e].mean())
                out[str(unit)] = {"edge_mean": round(me, 4), "mid_mean": round(mm, 4), "edge_n": int(e.sum()),
                                  "localised": bool(me > 3 * mm + 0.05)}
        return out


    def cmp_full(a, b, bins=True):
        """Every prompt position: mean / p99 / max |dlogprob|, rank-1 agreement, 4096-token bins (does it grow with context?),
        the first 256-position window over the screening mean, chunk-edge localisation."""
        if a["ids"].shape != b["ids"].shape or not np.array_equal(a["ids"], b["ids"]):
            return {"status": "INCONCLUSIVE_PROMPT_MISMATCH"}
        va, vb = a["valid"] & np.isfinite(a["lp"]), b["valid"] & np.isfinite(b["lp"])
        if (a["valid"] & ~np.isfinite(a["lp"])).any() or (b["valid"] & ~np.isfinite(b["lp"])).any():
            return {"status": "INCONCLUSIVE_NONFINITE"}
        pos = np.nonzero(va & vb)[0]
        n = len(a["ids"])
        if not len(pos):
            return {"status": "INCONCLUSIVE_NO_VALID"}
        d = np.abs(a["lp"][pos].astype(np.float64) - b["lp"][pos].astype(np.float64))
        t1 = (a["top1"][pos] == b["top1"][pos]) & (a["top1"][pos] >= 0)
        ds = np.sort(d)
        out = {"status": "MEASURED" if len(pos) == n - 1 else "PARTIAL", "n": int(len(pos)), "expected": n - 1,
               "mean_abs_dlogprob": round(float(d.mean()), 4), "p99_abs_dlogprob": round(float(ds[min(len(ds) - 1, int(round(0.99 * (len(ds) - 1))))]), 3),
               "max_abs_dlogprob": round(float(d.max()), 3), "rank1_agreement": round(float(t1.mean()), 4)}
        if bins:
            bl = []
            for lo in range(0, n, 4096):
                m = (pos >= lo) & (pos < lo + 4096)
                if m.any():
                    bl.append([lo, int(m.sum()), round(float(d[m].mean()), 4), round(float(t1[m].mean()), 4)])
            out["bins_4096"] = bl                        # [start, n, mean |dlogprob|, rank-1 agreement]
            if len(d) >= 256:
                cs = np.convolve(d, np.ones(256) / 256, "valid")
                idx = np.nonzero(cs > SCREEN["mean_max"])[0]
                out["first_window_over_screen_mean"] = int(pos[idx[0]]) if len(idx) else None
            out["chunk"] = chunk_check(pos, d)
        return out


    def shift_check(a, b, window=2048, shifts=(-2, -1, 0, 1, 2)):
        """Alignment test on the last `window` positions: reference rank-1 id at p vs candidate rank-1 id at p + s. A shifted
        agreement >= 0.9 that beats s = 0 by > 0.2 = a position-alignment artefact."""
        n = min(len(a["top1"]), len(b["top1"]))
        i0 = np.arange(max(1, n - window), n)
        res = {}
        for s in shifts:
            j = i0 + s
            m = (j >= 1) & (j < n)
            i, j = i0[m], j[m]
            v = a["valid"][i] & b["valid"][j] & (a["top1"][i] >= 0) & (b["top1"][j] >= 0)
            if not v.any():
                res[str(s)] = None
                continue
            dl = np.abs(a["top1_lp"][i][v].astype(np.float64) - b["top1_lp"][j][v].astype(np.float64))
            res[str(s)] = {"n": int(v.sum()), "top1_agree": round(float((a["top1"][i][v] == b["top1"][j][v]).mean()), 4),
                           "top1_lp_mean_abs_d": round(float(np.nanmean(dl)), 4) if np.isfinite(dl).any() else None}
        ok = {k: x for k, x in res.items() if x}
        best = max(ok, key=lambda k: (ok[k]["top1_agree"], k == "0")) if ok else None
        a0 = (ok.get("0") or {}).get("top1_agree")
        art = bool(best and best != "0" and ok[best]["top1_agree"] >= 0.9 and (a0 is None or ok[best]["top1_agree"] > a0 + 0.2))
        return {"shifts": res, "best_shift": int(best) if best is not None else None, "alignment_artefact": art}


    def screen(c):
        if not isinstance(c, dict) or c.get("status") != "MEASURED":
            return "INCONCLUSIVE"
        m, r1 = c.get("mean_abs_dlogprob"), c.get("rank1_agreement")
        if not (_finite(m) and _finite(r1)):
            return "INCONCLUSIVE_NONFINITE"
        return "DIVERGENT" if (r1 < SCREEN["rank1_min"] or m > SCREEN["mean_max"]) else "CLEAN"


    def screening_gate(ref_aa, cand):
        """External audit 2 (verified): conservative screening; 0.90 / 0.15 are not published acceptance limits."""
        if (ref_aa or {}).get("status") != "MEASURED":
            return "INCONCLUSIVE_REFERENCE"
        if (cand or {}).get("status") != "MEASURED":
            return "INCONCLUSIVE_FP8"
        s = screen(cand)
        if s == "INCONCLUSIVE_NONFINITE":
            return s
        return "HOLD_FP8_NUMERICAL_DIVERGENCE" if s == "DIVERGENT" else "SCREENING_PASS"


    def triage_gate(cand, short_ok=True):
        """External audit 3 (verified): severe-discrepancy triage. It can never give GO: TF similarity alone cannot certify
        ARC correctness, so the best outcome is PENDING_LONG_ARC_ACCURACY."""
        if short_ok is False:
            return "NO_GO_SHORT_ORACLE"
        if short_ok is not True:
            return "INCONCLUSIVE_SHORT_ORACLE"
        if (cand or {}).get("status") != "MEASURED":
            return "INCONCLUSIVE_TF"
        m, r1 = cand.get("mean_abs_dlogprob"), cand.get("rank1_agreement")
        if not (_finite(m) and _finite(r1)):
            return "INCONCLUSIVE_NONFINITE"
        if m > TRIAGE["mean_max"] or r1 < TRIAGE["rank1_min"]:
            return "NO_GO_FP8_TF_DIVERGENCE"
        return "PENDING_LONG_ARC_ACCURACY"


    def _gate_rank(g):
        g = str(g)
        return 3 if g.startswith(("HOLD", "NO_GO")) else 2 if g.endswith("NONFINITE") else 1 if g.startswith("INCONCLUSIVE") else 0


    def worst(gs):
        gs = list(gs)
        return max(gs, key=_gate_rank) if gs else "INCONCLUSIVE_NO_DATA"


    def decode_compare(ref, cand):
        """Greedy rows token for token: first divergence, the reference's top-1 - top-2 margin there (E23's 0.5 rule:
        FAIL_HIGH_MARGIN at or above, NEAR_TIE below), mean |dlogprob| over the common prefix."""
        if not ref or not cand or ref["meta"].get("error") or cand["meta"].get("error") or not ref["tok"] or not cand["tok"]:
            return {"status": "INCONCLUSIVE_MISSING"}
        x, y = ref["tok"], cand["tok"]
        d = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)
        c = len(x) if d is None else d
        dl = np.abs(ref["lp"][:c] - cand["lp"][:c]) if c else np.zeros(0)
        if not np.isfinite(dl).all() or (d is not None and not math.isfinite(float(ref["top1_lp"][d] - ref["top2_lp"][d]))):
            return {"status": "INCONCLUSIVE_NONFINITE", "first_diff": d}
        out = {"n": len(x), "first_diff": d, "common_prefix_mean_abs_dlogprob": round(float(dl.mean()), 4) if c else None,
               "common_prefix_max_abs_dlogprob": round(float(dl.max()), 4) if c else None}
        if d is None:
            out["status"] = "EXACT" if len(x) == len(y) else "INCONCLUSIVE_LENGTH"
        else:
            m = float(ref["top1_lp"][d] - ref["top2_lp"][d])
            out.update(ref_margin_at_diff=round(m, 4), status="FAIL_HIGH_MARGIN" if m >= L.FAIL_MARGIN else "NEAR_TIE")
        return out


    def sc_compare(dec, tf, P):
        """Decode-vs-prompt-logprob self-consistency on ONE boot: the greedy decode's chosen-token logprob at step i vs the
        teacher-forced logprob of the same token at position P + i (prompt + those tokens), rank-1 ids, +-2 shift."""
        if not dec or dec["meta"].get("error") or not tf or tf.get("arr") is None:
            return {"status": "INCONCLUSIVE_MISSING"}
        a, n = tf["arr"], len(dec["tok"])
        if n == 0:                                             # A-4: nothing decoded, nothing to compare (no IndexError)
            return {"status": "INCONCLUSIVE_EMPTY_DECODE"}
        pos = np.arange(P, P + n)
        if pos[-1] >= len(a["ids"]):                           # the TF prompt does not cover every decoded position
            return {"status": "INCONCLUSIVE_TF_TOO_SHORT", "need_len": int(P + n), "tf_len": int(len(a["ids"]))}
        if not np.array_equal(a["ids"][pos], np.asarray(dec["tok"], np.int32)):
            return {"status": "INCONCLUSIVE_PROMPT_MISMATCH"}
        dl, tl, v = dec["lp"], a["lp"][pos].astype(np.float64), a["valid"][pos]
        if not np.isfinite(dl).all() or not np.isfinite(tl[v]).all():
            return {"status": "INCONCLUSIVE_NONFINITE"}
        if not v.all():
            return {"status": "INCONCLUSIVE_INCOMPLETE_TF", "valid": int(v.sum()), "expected": n}
        d = np.sort(np.abs(dl - tl))
        r1 = float((dec["top1"] == a["top1"][pos]).mean())
        sh = {}
        for s in (-2, -1, 0, 1, 2):
            j = pos + s
            m = (j >= 1) & (j < len(a["ids"]))
            if m.any():
                sh[str(s)] = round(float((dec["top1"][m] == a["top1"][j[m]]).mean()), 4)
        out = {"status": "MEASURED", "n": n, "mean_abs_dlogprob": round(float(d.mean()), 4),
               "p99_abs_dlogprob": round(float(d[min(n - 1, int(round(0.99 * (n - 1))))]), 3), "max_abs_dlogprob": round(float(d[-1]), 3),
               "rank1_agreement": round(r1, 4), "greedy_self_top1": round(float((np.asarray(dec["tok"]) == dec["top1"]).mean()), 4),
               "shift_top1_agree": sh}
        out["screen"] = screen(out)
        return out


    def prefix_consistency(key):
        """One boot, two bands: the shared prompt prefix (RC2's prompt() starts every band of a variant with the same tokens)
        must score the same (cross-request determinism within a boot)."""
        runs = {b: ((TFR.get(key) or {}).get(b) or {}).get("fresh") for b in NB0["tf_bands"]}
        runs = {b: r for b, r in runs.items() if r and r.get("arr") is not None}
        out, bl = {}, list(runs)
        for i in range(len(bl)):
            for j in range(i + 1, len(bl)):
                a, c = runs[bl[i]]["arr"], runs[bl[j]]["arr"]
                n = min(len(a["ids"]), len(c["ids"]))
                eq = a["ids"][:n] == c["ids"][:n]
                common = n if eq.all() else int(np.argmin(eq))
                p = np.arange(1, common)
                p = p[a["valid"][p] & c["valid"][p] & np.isfinite(a["lp"][p]) & np.isfinite(c["lp"][p])]
                if len(p):
                    d = np.abs(a["lp"][p].astype(np.float64) - c["lp"][p].astype(np.float64))
                    out[f"{bl[i]}~{bl[j]}"] = {"shared": common, "n": int(len(p)), "mean_abs_dlogprob": round(float(d.mean()), 5),
                                               "max_abs_dlogprob": round(float(d.max()), 4),
                                               "top1_agree": round(float((a["top1"][p] == c["top1"][p]).mean()), 4)}
        vals = [x for x in out.values() if isinstance(x, dict)]
        out["consistent"] = all(x["mean_abs_dlogprob"] <= 0.01 and x["top1_agree"] >= 0.99 for x in vals) if vals else None
        return out


    # ---------------------------------------------------------------- per-boot suites (new)
    TF_EST = {"2k": 15, "8k": 25, "40k": 60, "90k": 150}      # seconds per teacher-forced request, upper estimates
    SC_EST = {"2k": 40, "8k": 60, "40k": 120}
    # SC variants: RC2's prompt() starts every band of a variant with the same tokens, so 8k v6 would hand 40k v6 a 4096-token
    # prefix hit; 2k v6 / 8k v7 / 40k v6 share < 4096 tokens with each other and with every other prompt of a boot (cold = cold).


    def tf_suite(key, s, bands):
        out = {}
        for band in bands:
            ids = PROMPTS[band][NB0["tf_variant"]]
            rec = TFR.setdefault(key, {}).setdefault(band, {})
            o = {"prompt_len": len(ids), "prompt_sha256": ids_sha(ids), "variant": NB0["tf_variant"]}
            if band == "90k" and f"{key}_90k" in DROP:
                o["status"] = "SKIPPED_BUDGET_PRIORITY"
                o["why"] = "lead rule: under budget pressure the MTP-off BF16 boot's 90k band is dropped first"
                out[band] = o
                continue
            for run in ("fresh", "repeat"):
                if not s.alive():
                    o[run] = {"status": "SKIPPED_SERVER_DEAD"}
                elif key in WD or ABORT["why"] or left() < TF_EST[band] + 240:
                    o[run] = {"status": "SKIPPED_BUDGET"}
                else:
                    rec[run] = tf_request(key, s.port, ids, band, run)
                    o[run] = rec[run]["meta"]
            o["aa"] = cmp_last256(rec.get("fresh"), rec.get("repeat"))
            out[band] = o
            save()
        return out


    def sc_suite(key, s):
        out = {}
        for band in NB0["sc_bands"]:
            ids = PROMPTS[band][NB0["sc_variant"][band]]
            if not s.alive():
                out[band] = {"status": "SKIPPED_SERVER_DEAD"}
                continue
            if key in WD or ABORT["why"] or left() < SC_EST[band] + 240:
                out[band] = {"status": "SKIPPED_BUDGET"}
                continue
            cold = decode(key, s.port, ids, NB0["sc_tok"], band, "cold")
            cached = decode(key, s.port, ids, NB0["sc_tok"], band, "cached")
            tf = tf_request(key, s.port, list(ids) + list(cold["tok"]), "sc" + band, "tf") if not cold["meta"].get("error") else None
            SCR.setdefault(key, {})[band] = {"cold": cold, "cached": cached, "tf": tf, "P": len(ids)}
            out[band] = {"cold": cold["meta"], "cached": cached["meta"], "tf": (tf or {}).get("meta"),
                         "decode_vs_tf": sc_compare(cold, tf, len(ids)), "cold_vs_cached": decode_compare(cold, cached),
                         "cold_was_cold": cold["meta"].get("cached_tokens") in (0, None)}
            save()
        return out


    def scales_facts(t):
        """The static scales sidecar's values, the per-stage 'applied ... to N of 12' lines, every 'E4M3 KV active on <layer>'
        line matched to the file (logged %.6g), and the MTP layer's DEFAULT 1.0 line."""
        path = CKPT / "qsa_kv_scales_262k.json"
        try:
            raw = path.read_bytes()
            file, sha = json.loads(raw), hashlib.sha256(raw).hexdigest()
        except Exception as e:
            file, sha = {}, f"unreadable: {e!r}"[:120]
        applied = re.findall(r"\((\w+) pid=(\d+)\)[^\n]*applied static K/V scales from (.+?) to (\d+) of (\d+) layer\(s\) "
                             r"\(pipeline layers (\d+)-(\d+)\)", t)
        act = {}
        for name, k, v, dflt in re.findall(r"QSA E4M3 KV active on (\S+): k_scale=([-\d.eE+]+) v_scale=([-\d.eE+]+)(\s+\(DEFAULT 1\.0)?", t):
            act.setdefault(name, set()).add((float(k), float(v), bool(dflt)))
        tol = lambda a, b: abs(a - b) <= 1e-5 * max(1e-12, abs(b))
        matched, mismatched = [], []
        for name, sc in sorted(file.items()):
            got = act.get(name)
            if got:
                (matched if all(tol(k, sc["k_scale"]) and tol(v, sc["v_scale"]) and not d for k, v, d in got) else mismatched).append(name)
        stages = sorted({f"{a}-{b}" for *_, a, b in applied})
        return {"file": str(path), "sha256": sha, "values": file, "applied_lines": len(applied), "applied_stages": stages,
                "applied_counts": sorted({f"{n} of {m}" for _, _, _, n, m, _, _ in applied}),
                "applied_paths": sorted({p for _, _, p, *_ in applied}), "active_layers": sorted(act), "matched": len(matched),
                "mismatched": mismatched, "missing": sorted(set(file) - set(act)), "not_in_file": sorted(set(act) - set(file)),
                "mtp_default_scale": sorted(n for n, g in act.items() if any(d for _, _, d in g)),
                "ok": bool(file) and not mismatched and not (set(file) - set(act)) and stages == ["0-25", "25-48"]}


    def boot_facts(s, key):
        """What the server itself logged for this boot (non-default args, engine config line, MTP-off warning, RecoverSSM
        path, byte-decoder marker, scales): config_ok only when every expected property holds (else CONFIG_MISMATCH)."""
        b, t = BOOTS[key], s.log() or ""
        lines = t.splitlines()
        nd = next((l for l in lines if "non-default args:" in l), "")
        eng = next((l for l in lines if "Initializing a V1 LLM engine" in l), "")
        g = lambda rx, txt: (re.search(rx, txt) or [None, None])[1]
        want_kv = "fp8_e4m3" if b["kv"] == "fp8" else "bfloat16"
        f = {"kv_cache_dtype": g(r"'kv_cache_dtype': '(\w+)'", nd), "max_num_seqs": g(r"'max_num_seqs': (\d+)", nd),
             "spec_in_args": "'speculative_config':" in nd, "spec_k_args": g(r"'num_speculative_tokens': (\d+)", nd),
             "captures": g(r"'cudagraph_capture_sizes': \[([\d, ]*)\]", nd), "prefix_caching": "'enable_prefix_caching': True" in nd,
             "engine_spec": g(r"speculative_config=(None|SpeculativeConfig\([^)]*\))", eng), "engine_kv": g(r"kv_cache_dtype=(\w+)", eng),
             "mtp_off_warning": bool(re.search(r"needs speculative decoding; num_speculative_tokens=0, so the stock GDN path is used", t)),
             "recoverssm_path_lines": len(re.findall(r"GDN RecoverSSM path taken", t)),
             "spec_metric_lines": len(re.findall(r"SpecDecoding metrics", t)),
             "byte_marker_pids": sorted(set(re.findall(r"NB0 boot E: QSA E4M3 byte decoder forced[^\n]*?pid=(\d+)", t))),
             "e4m3_active_lines": len(re.findall(r"QSA E4M3 KV active on", t))}
        f["scales"] = scales_facts(t) if b["kv"] == "fp8" else None
        es = f["engine_spec"] or ""
        ck = {"kv_args": f["kv_cache_dtype"] == want_kv, "kv_engine": f["engine_kv"] == want_kv,
              "seqs": f["max_num_seqs"] == str(b["want_seqs"]), "prefix_caching": f["prefix_caching"],
              "mtp": (f["spec_in_args"] and f["spec_k_args"] == "3" and es.startswith("SpeculativeConfig") and "num_spec_tokens=3" in es)
              if b["mtp"] else (not f["spec_in_args"] and es == "None" and f["mtp_off_warning"]),
              "byte_decoder": (len(f["byte_marker_pids"]) >= 4) if b["byte"] else not f["byte_marker_pids"],
              "bf16_no_e4m3": True if b["kv"] == "fp8" else f["e4m3_active_lines"] == 0}
        if b["captures"]:
            ck["captures"] = f["captures"] == "1, 2, 3, 4, 6, 8, 12, 16, 24, 32"
        f["checks"] = ck
        f["config_ok"] = all(ck.values())
        return f


    def nb0_body(key):
        def body(ss):
            s = ss[0]
            p = s.port
            o = {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}
            o["boot_facts"] = boot_facts(s, key)
            if short_gate_failed(o):
                o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s))
                return o
            if o["boot_facts"]["config_ok"] is not True:
                bad = [k for k, v in o["boot_facts"]["checks"].items() if not v]
                o.update(early_stop="CONFIG_MISMATCH", status="CONFIG_MISMATCH", config_mismatch=bad, census_end=kv_census(s))
                return o
            o["oracle"] = oracle(p, key=key, refs=ORACLE_REFS[key])
            o["tf"] = tf_suite(key, s, ["2k", "8k", "40k"])
            o["sc"] = sc_suite(key, s)
            o["tf"].update(tf_suite(key, s, ["90k"]))                # 90k last: the largest prompt-logits request
            o["census_end"] = kv_census(s)
            o["boot_facts_end"] = boot_facts(s, key)
            o["tf_died"] = not s.alive()
            return o
        return body


    # ---------------------------------------------------------------- boot E: byte decoder (new; audited, fail closed)
    DECODER_TEST = r"""
    import json, torch, triton
    import triton.language as tl
    __DECODER_SRC__

    @triton.jit
    def _nb0_decode_all(x_ptr, y_ptr):
        o = tl.arange(0, 256)
        tl.store(y_ptr + o, _decode_e4m3_to_bf16(tl.load(x_ptr + o)))


    x = torch.arange(256, dtype=torch.uint8, device="cuda")
    y = torch.empty(256, dtype=torch.bfloat16, device="cuda")
    _nb0_decode_all[(1,)](x, y)
    torch.cuda.synchronize()
    ref = x.view(torch.float8_e4m3fn).to(torch.bfloat16)
    yn, rn = torch.isnan(y), torch.isnan(ref)
    same = (y.view(torch.int16) == ref.view(torch.int16)) | (yn & rn)
    bad = [int(i) for i in torch.nonzero(~same).flatten().tolist()]
    sub = [i for i in range(256) if (i & 0x78) == 0 and (i & 0x7) != 0]
    print("NB0_DECODER_RESULT " + json.dumps({"ok": not bad, "n_bad": len(bad), "bad_codepoints": bad[:32],
          "subnormals_ok": all(i not in bad for i in sub), "n_subnormals": len(sub), "nan_ok": bool((yn == rn).all()),
          "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "triton": triton.__version__}))
    """
    BYTE = {}


    def decoder_script(src):
        """The device-check script (dedented: the cell wrapper indents this string literal)."""
        return textwrap.dedent(DECODER_TEST).replace("__DECODER_SRC__", src)


    def byte_decoder_device_check(ops_txt):
        """The installed `_decode_e4m3_to_bf16` (Triton) vs torch's native E4M3 -> BF16 cast on all 256 codepoints, on
        GPU 0 of the L4s, bit for bit (NaN as NaN). Merlin verified the byte decoder's FP16-subnormal trick on sm_86 only."""
        m = re.search(r"@triton\.jit\ndef _decode_e4m3_to_bf16\(b\):\n(?:(?:    [^\n]*)?\n)+", ops_txt)
        if not m:
            return {"ok": False, "why": "_decode_e4m3_to_bf16 not found in the installed ops/qsa.py"}
        d = TMP / "nb0_decoder_test"
        d.mkdir(exist_ok=True)
        f = d / "nb0_decoder_test.py"
        f.write_text(decoder_script(m.group(0)), encoding="utf-8")
        eng = MERLIN_ENG or {}
        venv = (eng.get("kv") or {}).get("MERLIN_VENV")
        if not venv:
            return {"ok": False, "why": "Merlin venv unknown"}
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "LD_PRELOAD")}
        env.update(CUDA_VISIBLE_DEVICES="0", TRITON_CACHE_DIR=str(d / "triton"), CUDA_HOME=eng.get("cuda_home", ""),
                   PATH=f"{venv}/bin:{eng.get('cuda_home', '')}/bin:{os.environ.get('PATH', '')}")
        try:
            r = subprocess.run([f"{venv}/bin/python", str(f)], capture_output=True, text=True, timeout=600, env=env, cwd=str(d))
        except Exception as e:
            return {"ok": False, "why": f"device check did not run: {e!r}"[:300]}
        line = next((l for l in r.stdout.splitlines() if l.startswith("NB0_DECODER_RESULT ")), None)
        if r.returncode or not line:
            return {"ok": False, "why": f"device check rc {r.returncode}", "tail": (r.stdout + r.stderr)[-1500:]}
        res = json.loads(line.split(" ", 1)[1])
        res["source_sha256"] = hashlib.sha256(m.group(0).encode()).hexdigest()
        return res


    def byte_decoder_prepare():
        """Boot E (lead addendum): qsa_e4m3_uses_native_cast() -> False on sm_89, i.e. Merlin's sm_86-qualified path (Triton
        byte decoder `_decode_e4m3_to_bf16` for K and V, unfused QSA prepare + reshape_and_cache_flash), and the qsa.py:231
        hardware gate still admitting E4M3 then. The source of the installed tree is patched (qsa.py binds the function by
        name at import, ops/qsa.py calls its module global: one body patch covers ops 639 / 840 and qsa 231 / 710); the patch
        is env-gated (NB0_QSA_FORCE_BYTE_DECODER=1, set in boot E's server env only); restored after the boot. Fail closed."""
        out = {"ok": False}
        if FAKE:
            out.update(ok=True, fake=True)
            return out
        if BYTE.get("applied"):                # a previous patch is still pending restore: never stack a second one
            out["why"] = f"a previous boot-E patch is still pending restore: {BYTE['applied']}"
            return out
        eng = MERLIN_ENG or {}
        tree = Path(eng["kv"]["MERLIN_TREE"]) if eng.get("kv") else None
        if tree is None:
            out["why"] = "Merlin tree unknown (install failed?)"
            return out
        files, audit = {}, {}
        for k in ("ops", "qsa"):
            p = tree / QSA_PATCH[k + "_rel"]
            try:
                raw = p.read_bytes()
                txt = raw.decode("utf-8")
            except Exception as e:
                out["why"] = f"{QSA_PATCH[k + '_rel']}: {e!r}"[:300]
                return out
            sha = hashlib.sha256(raw).hexdigest()
            anchors = [a for a in QSA_PATCH["audit"][k] + [QSA_PATCH[k + "_old"]] if txt.count(a) != 1]
            names = txt.count("qsa_e4m3_uses_native_cast")
            audit[k] = {"sha256": sha, "sha_matches_audited": sha == QSA_PATCH[k + "_sha"], "anchors_failed": len(anchors),
                        "name_count": names, "marker_already": QSA_PATCH["marker"] in txt}
            if anchors or names != QSA_PATCH["audit"]["name_count"][k] or QSA_PATCH["marker"] in txt:
                out.update(audit=audit, why=f"{QSA_PATCH[k + '_rel']}: the audited call sites differ in the installed tree")
                return out
            files[k] = (p, raw, txt)
        out["audit"] = audit
        unreviewed = [QSA_PATCH[k + "_rel"] for k in files if not audit[k]["sha_matches_audited"]]
        if unreviewed:                         # A-1: exact audited bytes only (anchors alone are not a review)
            out.update(status="UNREVIEWED_SOURCE_SHA", why=f"installed source sha256 != the audited Merlin ad42e37 sha: {unreviewed}")
            return out
        try:
            prod = (tree / ".build-products.sha256").read_text()
        except Exception:
            prod = ""
        listed = [QSA_PATCH[k + "_rel"] for k in files if QSA_PATCH[k + "_rel"] in prod]
        if listed:
            out["why"] = f"patched files are verified build products (host_phase would refuse the tree): {listed}"
            return out
        out["device_check"] = byte_decoder_device_check(files["ops"][2])
        if out["device_check"].get("ok") is not True:
            out["why"] = "on-device decoder check failed on the L4: " + json.dumps(out["device_check"])[:300]
            return out
        venv = (eng.get("kv") or {}).get("MERLIN_VENV")
        try:                                   # any failure from here on restores the tree before returning (fail closed)
            for k, (p, raw, txt) in files.items():
                bak = p.with_name(p.name + ".nb0_orig")
                # A-3: backup (read back = the audited sha) -> register -> write. A file is touched only after it is
                # registered, so a write that fails half-way is restored from its verified backup too.
                bak.write_bytes(raw)
                if hashlib.sha256(bak.read_bytes()).hexdigest() != QSA_PATCH[k + "_sha"]:
                    raise OSError(f"backup {bak.name} does not read back as the audited source")
                BYTE.setdefault("applied", []).append(str(p))
                BYTE.setdefault("want", {})[str(p)] = QSA_PATCH[k + "_sha"]
                p.write_bytes(txt.replace(QSA_PATCH[k + "_old"], QSA_PATCH[k + "_new"]).encode("utf-8"))
                cp = subprocess.run([f"{venv}/bin/python", "-m", "py_compile", str(p)], capture_output=True, text=True, timeout=120)
                if cp.returncode or p.read_text(encoding="utf-8").count(QSA_PATCH["marker"]) != 1:
                    out["why"] = f"patched {QSA_PATCH[k + '_rel']} does not compile / marker missing: {cp.stderr[-300:]}"
                    out["restore"] = byte_decoder_restore()
                    return out
        except Exception as e:
            out["why"] = f"patch step failed: {type(e).__name__}: {e}"[:300]
            out["restore"] = byte_decoder_restore()
            return out
        out.update(ok=True, patched=[QSA_PATCH["ops_rel"], QSA_PATCH["qsa_rel"]])
        return out


    def byte_decoder_restore():
        """A-2: every pending file is restored from its backup, read back and compared with the audited original sha; it
        leaves the pending list only after that check, so a failed restore stays pending (a second call fails again)."""
        out, failed, want = {"files": {}}, [], BYTE.get("want") or {}
        for x in list(BYTE.get("applied") or []):
            p = Path(x)
            bak = p.with_name(p.name + ".nb0_orig")
            try:
                if not want.get(x):
                    raise ValueError("no audited original sha recorded for this file")
                raw = bak.read_bytes()
                if hashlib.sha256(raw).hexdigest() != want[x]:
                    raise ValueError("backup sha256 != the audited original")
                p.write_bytes(raw)
                got = hashlib.sha256(p.read_bytes()).hexdigest()
                if got != want[x]:
                    raise ValueError(f"restored file reads back as {got[:16]}, not the audited original")
                rec = {"restored": True, "sha256": got}
                try:
                    bak.unlink()
                except Exception as e:
                    rec["backup_left"] = repr(e)[:200]
                out["files"][x] = rec
            except Exception as e:
                failed.append(x)
                out["files"][x] = {"restored": False, "error": repr(e)[:200]}
        BYTE["applied"] = failed
        BYTE["want"] = {x: want[x] for x in failed if x in want}
        out.update(ok=not failed, pending=failed)
        return out


    # ---------------------------------------------------------------- one boot (RC2's merlin_boot) + its NB0 records
    def boot_record(key):
        return (R["sections"].get("NB0") or {}).get(BOOTS[key]["step"]) or {}


    def qsa_restore_gate(key, rs):
        """A-3: a restore that did not verify (or any file still pending) leaves the Merlin tree patched: ABORT, so no later
        Merlin boot (B, A) runs on patched source."""
        if FAKE or ((rs or {}).get("ok") is not False and not BYTE.get("applied")):
            return
        ABORT["why"] = ABORT["why"] or f"QSA_PATCH_RESTORE_FAILED: boot {key}: still pending {BYTE.get('applied')}"
        save()


    def run_boot(key):
        b = BOOTS[key]
        pre = None
        if b["byte"]:
            pre = byte_decoder_prepare()
            if pre.get("ok") is not True:
                qsa_restore_gate(key, pre.get("restore"))
                return {"status": pre.get("status") or "SKIPPED_BYTE_DECODER_UNSAFE", "byte_decoder": pre}
        try:
            o = merlin_boot(b["name"], b["kv"], b["port"], NB0["ready_sec"], nb0_body(key), need=(), expected_rows=4)
        finally:
            if b["byte"]:
                pre["restore"] = byte_decoder_restore() if not FAKE else {"fake": True}
                qsa_restore_gate(key, pre["restore"])
        gate = o.get("boot_gate") or {}
        if (gate.get("host_prep") or {}).get("ok") is False:
            o["status"] = "HOST_GATE_FAILED"
            ABORT["why"] = ABORT["why"] or f"host preparation failed before boot {key}: {gate['host_prep'].get('why')}"
        if key in WD:
            o["status"] = "BOOT_WATCHDOG"
        o.update(boot=key, title=b["title"], plan=(R["setup"].get("nb0_plans") or {}).get(b["name"]), load_monitor=LOADMON.get(key),
                 watchdog=WD.get(key), byte_decoder=pre)
        return o


    def boot_short_ok(key):
        r = boot_record(key)
        if not r:
            return None
        return r.get("correctness") == "CLEAN" and r.get("status") in ("DONE", "TF_OOM", "PARTIAL_HARD_STOP")


    def prompt_check():
        """NB0's prompt ids = RC2's: every prompt RC2 sent on Kaggle (47 requests, sha256[:16] of the id list) must come out
        of RC2's prompt() unchanged here; plus the full sha256 of every NB0 teacher-forced / decode prompt."""
        have = {}
        for band, ps in PROMPTS.items():
            for v, ids in enumerate(ps):
                have[hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16]] = [band, v, len(ids)]
        have[hashlib.sha256(json.dumps(TPL_IDS).encode()).hexdigest()[:16]] = ["TPL_IDS", None, len(TPL_IDS)]
        rec = RC2_REF["prompt_shas"]
        miss = [s for s, n in rec.items() if s not in have or have[s][2] != n]
        return {"ok": not miss and bool(MERLIN_PROMPTS), "rc2_recorded": len(rec), "reproduced": len(rec) - len(miss),
                "missing": miss[:10], "matched_as": {s: have[s][:2] for s in rec if s in have},
                "tf_prompts": {b: {"variant": NB0["tf_variant"], "len": len(PROMPTS[b][NB0["tf_variant"]]),
                                   "sha256": ids_sha(PROMPTS[b][NB0["tf_variant"]])} for b in NB0["tf_bands"]},
                "sc_prompts": {b: {"variant": NB0["sc_variant"][b], "len": len(PROMPTS[b][NB0["sc_variant"][b]]),
                                   "sha256": ids_sha(PROMPTS[b][NB0["sc_variant"][b]])} for b in NB0["sc_bands"]},
                "rc2_tf_prompt_is_40k_v5": PROMPTS["40k"][5] == TF_P["40k"], "source": R["prompt_render"].get("source")}


    # ---------------------------------------------------------------- analysis, gates, decision table (new)
    DECISION_TABLE = [
        ["KV pair (A-B, C-D or C-E) divergent already at 2k", "quantisation / scales / kernel"],
        ["KV pair clean at 2k (and 8k), divergent only at 40k / 90k", "long-context accumulation / recurrent or cache state"],
        ["Only a repeat differs (per-boot fresh vs repeat A/A, or decode cold vs cached)", "cache / state"],
        ["A-B clean but C-D divergent; or A-C / B-D divergent (target logits change with MTP)", "speculative decoding / PP / RecoverSSM"],
        ["Decode logprobs disagree with prompt logprobs on one boot", "prompt-logprob (prefill-output) path artefact or a prefill defect on that boot"],
        ["A +-1 / +-2 position shift restores the rank-1 agreement", "position-alignment artefact"],
        ["Divergence concentrated at 1024-token scheduler or 256-row logits chunk edges", "chunking artefact"],
        ["C-D divergent while C-E is clean", "the sm_89 native E4M3 read path is the defect; keep FP8 capacity via Merlin's byte decoder"],
        ["C-D and C-E both divergent", "not the native cast: FP8 KV itself (scales / quantisation) or the shared prefill path"],
        ["MTP-on vs MTP-off greedy rows diverge at a high margin (cf. upstream #63593)", "GDN state on rejected drafts (RecoverSSM)"],
    ]


    def analyze():
        an = {"per_boot": {}, "pairs": {}, "rc2_anchor": RC2_REF["tf40k"], "decision_table": DECISION_TABLE,
              "thresholds": {"screening": SCREEN, "triage": TRIAGE, "decode_fail_margin": L.FAIL_MARGIN}}
        for key in ORDER:
            if key not in TFR and key not in SCR:
                continue
            pb = {"aa": {}, "aa_full": {}, "cached_tokens": {}, "prefix_consistency": prefix_consistency(key), "sc": {}}
            for band, runs in (TFR.get(key) or {}).items():
                f, r = runs.get("fresh"), runs.get("repeat")
                pb["aa"][band] = cmp_last256(f, r)
                pb["aa_full"][band] = (cmp_full(f["arr"], r["arr"], bins=False) if f and r and f.get("arr") is not None
                                       and r.get("arr") is not None else {"status": "INCONCLUSIVE_MISSING_TF"})
                pb["cached_tokens"][band] = [(runs.get(x) or {}).get("meta", {}).get("cached_tokens") for x in ("fresh", "repeat")]
            for band, sc in (SCR.get(key) or {}).items():
                pb["sc"][band] = {"decode_vs_tf": sc_compare(sc["cold"], sc["tf"], sc["P"]),
                                  "cold_vs_cached": decode_compare(sc["cold"], sc["cached"]),
                                  "cached_tokens": [sc["cold"]["meta"].get("cached_tokens"), sc["cached"]["meta"].get("cached_tokens")]}
            pb["scales"] = ((boot_record(key).get("boot_facts_end") or boot_record(key).get("boot_facts") or {}).get("scales"))
            an["per_boot"][key] = pb
        for pk, (ra, rb, what) in PAIRS.items():
            pr = {"what": what, "ref": ra, "cand": rb, "bands": {}, "parity": {}}
            for band in NB0["tf_bands"]:
                fa = ((TFR.get(ra) or {}).get(band) or {}).get("fresh")
                fb = ((TFR.get(rb) or {}).get(band) or {}).get("fresh")
                c = cmp_last256(fa, fb)
                e = {"last256": c, "screen": screen(c)}
                if fa and fb and fa.get("arr") is not None and fb.get("arr") is not None:
                    e["full"] = cmp_full(fa["arr"], fb["arr"])
                    e["shift"] = shift_check(fa["arr"], fb["arr"])
                ref_aa = ((an["per_boot"].get(ra) or {}).get("aa") or {}).get(band)
                e["screening_gate"] = screening_gate(ref_aa, c)
                e["triage_gate"] = triage_gate(c, short_ok=boot_short_ok(rb))
                pr["bands"][band] = e
            for band in NB0["sc_bands"]:
                xa, xb = ((SCR.get(ra) or {}).get(band) or {}).get("cold"), ((SCR.get(rb) or {}).get(band) or {}).get("cold")
                pr["parity"][band] = decode_compare(xa, xb) if xa and xb else {"status": "INCONCLUSIVE_MISSING"}
            orc = (boot_record(rb).get("oracle") or {}).get("vs_" + ra)
            pr["oracle_128"] = orc
            an["pairs"][pk] = pr
        an["gates"] = fp8_verdict(an)
        an["findings"] = decide(an)
        return an


    def fp8_verdict(an):
        out = {}
        for pk, label in (("CD", "FP8 native sm_89 read path, MTP K3 (production default) vs BF16 MTP K3"),
                          ("AB", "FP8 vs BF16, target only (MTP off)"),
                          ("CE", "FP8 through Merlin's byte decoder, MTP K3 vs BF16 MTP K3")):
            bands = ((an["pairs"].get(pk) or {}).get("bands")) or {}
            sc = {b: (bands.get(b) or {}).get("screening_gate", "INCONCLUSIVE_FP8") for b in NB0["tf_bands"]}
            tr = {b: (bands.get(b) or {}).get("triage_gate", "INCONCLUSIVE_TF") for b in NB0["tf_bands"]}
            out[pk] = {"what": label, "screening": worst(sc.values()), "triage": worst(tr.values()), "per_band": {b: [sc[b], tr[b]] for b in sc}}
        out["fp8_native_production"] = out["CD"]["triage"]
        out["fp8_byte_decoder"] = out["CE"]["triage"]
        out["note"] = ("No gate here gives GO: TF similarity cannot certify ARC correctness; the best outcome is "
                       "PENDING_LONG_ARC_ACCURACY (a paired ARC accuracy run on the same tasks decides).")
        return out


    def decide(an):
        """The decision table applied to the measured comparisons; every finding names its evidence."""
        F, P, B = [], an.get("pairs") or {}, an.get("per_boot") or {}

        def st(pk, band):
            return (((P.get(pk) or {}).get("bands") or {}).get(band) or {}).get("screen", "INCONCLUSIVE")

        def bands_with(pk, s):
            return [b for b in NB0["tf_bands"] if st(pk, b) == s]

        for key, pb in B.items():
            for band, c in (pb.get("aa") or {}).items():
                s = screen(c)
                if s == "DIVERGENT":
                    F.append({"finding": "REPEAT_DIVERGENCE", "boot": key, "band": band, "evidence": c,
                              "meaning": "the same teacher-forced request twice on one boot differs: cache / state"})
                elif s != "CLEAN":
                    F.append({"finding": "AA_INCONCLUSIVE", "boot": key, "band": band, "evidence": c.get("status")})
            for band, sc in (pb.get("sc") or {}).items():
                if (sc.get("decode_vs_tf") or {}).get("screen") == "DIVERGENT":
                    F.append({"finding": "DECODE_VS_PROMPT_LOGPROB_MISMATCH", "boot": key, "band": band, "evidence": sc["decode_vs_tf"],
                              "meaning": "prompt-logprob (prefill-output) path artefact or a prefill-vs-decode defect on this boot"})
                cc = sc.get("cold_vs_cached") or {}
                if cc.get("status") == "FAIL_HIGH_MARGIN" or (cc.get("common_prefix_mean_abs_dlogprob") or 0) > SCREEN["mean_max"]:
                    F.append({"finding": "CACHED_PREFIX_DIVERGENCE", "boot": key, "band": band, "evidence": cc,
                              "meaning": "decode on a cached prefix differs from the cold decode: cache / state"})
            pc = pb.get("prefix_consistency") or {}
            if pc.get("consistent") is False:
                F.append({"finding": "CROSS_REQUEST_NONDETERMINISM", "boot": key, "evidence": pc,
                          "meaning": "the shared prompt prefix scores differently in two requests of one boot"})
            scl = pb.get("scales")
            if isinstance(scl, dict) and not scl.get("ok"):
                F.append({"finding": "STATIC_SCALES_NOT_FULLY_APPLIED", "boot": key,
                          "evidence": {k: scl.get(k) for k in ("applied_stages", "matched", "mismatched", "missing")}})
        for pk, pr in P.items():
            for band, e in (pr.get("bands") or {}).items():
                if e.get("screen") != "DIVERGENT":
                    continue
                if (e.get("shift") or {}).get("alignment_artefact"):
                    F.append({"finding": "POSITION_SHIFT_ARTEFACT", "pair": pk, "band": band, "evidence": e["shift"]})
                if any(v.get("localised") for v in ((e.get("full") or {}).get("chunk") or {}).values()):
                    F.append({"finding": "CHUNK_BOUNDARY_LOCALISED", "pair": pk, "band": band, "evidence": e["full"]["chunk"]})
        for pk in ("AB", "CD", "CE"):
            meas = [b for b in NB0["tf_bands"] if st(pk, b) in ("CLEAN", "DIVERGENT")]
            div = bands_with(pk, "DIVERGENT")
            if not meas:
                continue
            if "2k" in div:
                f, m = "KV_DIVERGENCE_FROM_2K", "quantisation / scales / kernel"
            elif div and set(div) <= {"40k", "90k"}:
                f, m = "KV_DIVERGENCE_LONG_ONLY", "long-context accumulation / recurrent or cache state"
            elif div:
                f, m = "KV_DIVERGENCE_FROM_8K", "grows with context: accumulation / state"
            else:
                f, m = "KV_CLEAN", "no KV-dtype divergence in the measured bands"
            F.append({"finding": f, "pair": pk, "divergent_bands": div, "measured_bands": meas, "meaning": m,
                      "what": P[pk]["what"]})
        ab_clean = all(st("AB", b) == "CLEAN" for b in NB0["tf_bands"] if st("AB", b) != "INCONCLUSIVE") and any(
            st("AB", b) == "CLEAN" for b in NB0["tf_bands"])
        if ab_clean and bands_with("CD", "DIVERGENT"):
            F.append({"finding": "MTP_ONLY", "evidence": {"AB": "clean", "CD": bands_with("CD", "DIVERGENT")},
                      "meaning": "speculative decoding / PP / RecoverSSM"})
        for pk, f in (("AC", "MTP_CHANGES_TARGET_LOGITS_BF16"), ("BD", "MTP_CHANGES_TARGET_LOGITS_FP8")):
            if bands_with(pk, "DIVERGENT"):
                F.append({"finding": f, "pair": pk, "divergent_bands": bands_with(pk, "DIVERGENT"),
                          "meaning": "the MTP path (RecoverSSM / spec verify / PP) changes the target's prompt logits"})
        cd, ce = bands_with("CD", "DIVERGENT"), bands_with("CE", "DIVERGENT")
        ce_meas = [b for b in NB0["tf_bands"] if st("CE", b) in ("CLEAN", "DIVERGENT")]
        if cd and ce_meas and not ce:
            F.append({"finding": "NATIVE_SM89_E4M3_PATH_DEFECT", "evidence": {"CD_divergent": cd, "CE_clean": ce_meas},
                      "meaning": "the sm_89 native float8 read path is the defect; keep FP8 capacity via Merlin's byte decoder (boot E config)"})
        elif cd and ce:
            F.append({"finding": "FP8_DIVERGENCE_NOT_THE_NATIVE_CAST", "evidence": {"CD": cd, "CE": ce},
                      "meaning": "FP8 KV itself (scales / quantisation) or the shared prefill path"})
        if bands_with("DE", "DIVERGENT"):
            F.append({"finding": "READ_PATH_CHANGES_LOGITS", "pair": "DE", "divergent_bands": bands_with("DE", "DIVERGENT")})
        for pk in ("BD", "AC"):
            for band, x in ((P.get(pk) or {}).get("parity") or {}).items():
                if x.get("status") == "FAIL_HIGH_MARGIN":
                    F.append({"finding": "MTP_GREEDY_DIVERGENCE_HIGH_MARGIN", "pair": pk, "band": band, "evidence": x,
                              "meaning": "MTP-on greedy output leaves the MTP-off greedy output at a confident token (GDN state on rejected drafts, cf. #63593)"})
        c40 = (((P.get("CD") or {}).get("bands") or {}).get("40k") or {}).get("last256") or {}
        F.append({"finding": "RC2_ANCHOR", "rc2_40k": RC2_REF["tf40k"]["fp8_vs_bf16"],
                  "nb0_CD_40k": {k: c40.get(k) for k in ("status", "mean_abs_dlogprob", "p99_abs_dlogprob", "rank1_agreement")}})
        return F


if not RERUN:
    # ==================================================================== CELL 7 — NB0 main: boots, analysis, report
    if not RERUN and not os.environ.get("NB0_UNIT"):
        R["nb0_design"] = {"boots": {k: {x: v for x, v in b.items() if x != "captures"} | {"captures": bool(b["captures"])}
                                     for k, b in BOOTS.items()}, "order": ORDER, "pairs": PAIRS, "params": NB0,
                           "screening": SCREEN, "triage": TRIAGE, "decision_table": DECISION_TABLE}
        R["prompt_check"] = prompt_check()
        log("NB0 prompt check", json.dumps({k: R["prompt_check"][k] for k in ("ok", "rc2_recorded", "reproduced", "missing")}))
        if not R["prompt_check"]["ok"]:
            ABORT["why"] = "PROMPT_MISMATCH: NB0's prompt ids are not RC2's"
        save()
        if R["sections"]["A0"]["merge"].get("ok") and not ABORT["why"]:
            rec = step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
            if rec.get("status") != "DONE":
                ABORT["why"] = ABORT["why"] or "Merlin install failed"
            ok, why = sha_gate(min(1500, max(60, left() - 900)))
            R["setup"]["sha_gate"] = {"ok": ok, "why": why}
            if not ok:
                ABORT["why"] = ABORT["why"] or f"SHA_GATE: {why}"
            save()
            for key in ORDER:
                b = BOOTS[key]
                if key == "A":
                    rb = boot_record("B")
                    if rb.get("status") not in ("DONE", "TF_OOM", "PARTIAL_HARD_STOP"):
                        R["sections"].setdefault("NB0", {})[b["step"]] = {"status": "SKIPPED_DEPENDENT",
                            "why": f"boot B (the MTP-off path) ended {rb.get('status')}: A uses the same path"}
                        save()
                        continue
                    if left() < (EST_MIN["A"] + 3) * 60 + 300:
                        DROP.add("A_90k")
                r = step("NB0", b["step"], EST_MIN[key] - (2 if f"{key}_90k" in DROP else 0), lambda key=key: run_boot(key))
                if key == "D" and r.get("status") in ("LAUNCH_FAILED", "SERVER_DIED", "ERROR", "BODY_ERROR", "GPU_NOT_EMPTY"):
                    ABORT["why"] = ABORT["why"] or f"boot D (the RC2-proven FP8 config) ended {r.get('status')}: environment suspect"
                if r.get("status") in ("HOST_GATE_FAILED", "GPU_NOT_EMPTY", "FAIL_GPU_RELEASE"):
                    ABORT["why"] = ABORT["why"] or f"boot {key}: {r.get('status')}"
        else:
            R["errors"].append("NB0 did not boot: " + (ABORT["why"] or "Merlin inputs missing / merge failed"))
        try:
            R["analysis"] = analyze()
        except Exception as e:
            R["analysis"] = {"error": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-2500:]}
        R["minutes"] = round((time.time() - T_START) / 60, 1)
        R["host_phase"] = HOST_PHASE
        save()
        an = R["analysis"]
        verdict = {"gates": an.get("gates"), "findings": [{k: v for k, v in f.items() if k != "evidence"} for f in an.get("findings") or []],
                   "boots": {k: boot_record(k).get("status") for k in ORDER}, "abort": ABORT["why"], "minutes": R["minutes"]}
        (OUT / "nb0_verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
        write_fallback_submission()
        print(json.dumps(verdict, default=str)[:8000])
