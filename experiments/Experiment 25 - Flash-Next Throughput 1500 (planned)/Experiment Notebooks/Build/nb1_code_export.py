# ==================================================================== CELL 1 — config, inputs, lab module (RC2 verbatim + E25 NB1 marked edits)
import os, sys, json, math, time, re, shutil, subprocess, hashlib, tarfile, threading, traceback
from pathlib import Path
T_START = time.time()
MODE = "FULL"
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes"}
FAKE = os.environ.get("E23_FAKE_SERVER")                          # local CPU harness only
BUDGET = int(os.environ.get("RC2_BUDGET_SEC") or 240 * 60)     # E25 NB1: 3 cold Merlin boots (~13-17 min each) + ~2.9 h of measurements
HARD_STOP = T_START + BUDGET
MAX_LEN = 131072                                                  # user 2026-10-29: thinking up to 120k
OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/nb1"))   # E25 NB1
OUT.mkdir(parents=True, exist_ok=True)
LOGS = OUT / "logs"
LOGS.mkdir(exist_ok=True)
TRACES = OUT / "traces"                                           # memory rule: the full text of every long generation
TRACES.mkdir(exist_ok=True)
TMP = Path(os.environ.get("E23_TMP", "/tmp"))
INP = Path(os.environ.get("E23_INPUT", "/kaggle/input"))
R = {"what": "E25 NB1 FN-S1 Capacity: Flash-Next on Merlin",   # E25 NB1
      "budget_min": BUDGET // 60, "max_len": MAX_LEN,
     "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "setup": {}, "sections": {}, "errors": [], "order": []}
_SAVE_LOCK = threading.Lock()
NB1_EXTRA, NB1_ENV, NB1_KV = {}, {}, {}   # E25 NB1: per-boot trailing vllm flags, env and KV pin (filled by the boot plan)


def save():
    with _SAVE_LOCK:
        tmp = OUT / "nb1_results.json.tmp"                       # E25 NB1
        tmp.write_text(json.dumps(R, indent=1, default=str))
        tmp.replace(OUT / "nb1_results.json")                    # E25 NB1


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
    print("E25 NB1 capacity diagnostic: competition rerun -> fallback submission.json only")
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
    # ==================================================================== CELL 6 — E25 NB1: measurement tools (NEW code)
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
    NB1_EXT_SRC = "\"\"\"E25 NB1 worker extension (read-only): reached through POST /collective_rpc {\"method\": \"nb1_cache_info\"} with\nVLLM_SERVER_DEV_MODE=1 and --worker-extension-cls nb1_worker_ext.NB1WorkerExt. Reports the ACTUAL cache tensors bound\nto every attention / Mamba layer (dtype, shape, bytes), the KV cache specs (padded vs unpadded Mamba page bytes), the\nphysical block count and this rank's free memory. It never raises: every part returns its error text instead.\"\"\"\nimport json\nimport math\n\n\ndef _t(x):\n    try:\n        import torch\n        if isinstance(x, torch.Tensor):\n            return {\"dtype\": str(x.dtype).replace(\"torch.\", \"\"), \"shape\": list(x.shape), \"bytes\": int(x.numel() * x.element_size())}\n    except Exception:\n        pass\n    return None\n\n\ndef _flat(kc):\n    if _t(kc):\n        return [kc]\n    if isinstance(kc, (list, tuple)):\n        if kc and isinstance(kc[0], (list, tuple)) and not _t(kc[0]):\n            kc = kc[0]                              # one entry per virtual engine\n        return [x for x in kc if _t(x)]\n    return []\n\n\n_RT = {\"on\": False, \"installed\": False, \"rows\": [], \"passes\": 0, \"calls\": 0, \"max_steps\": 200, \"bytes\": 0, \"errors\": [],\n       \"first\": None, \"skipped_capturing\": 0, \"skipped_prefill\": 0}\n_RT_MAX_BYTES = 16 << 20\n_RT_DECODE_MAX_ROWS = 128          # a decode / verify micro-batch: requests x (1 + K) tokens; prefill chunks are larger\n\n\ndef _rt_layer(name):\n    import re as _r\n    m = _r.search(r\"layers\\.(\\d+)\", str(name))\n    return int(m.group(1)) if m else -1\n\n\ndef _rt_wrap(orig):\n    def select_experts(self, hidden_states, router_logits, *a, **k):\n        out = orig(self, hidden_states, router_logits, *a, **k)\n        if _RT[\"on\"] and _RT[\"passes\"] < 2 * _RT[\"max_steps\"] and _RT[\"bytes\"] < _RT_MAX_BYTES:\n            try:\n                import torch\n                if torch.cuda.is_current_stream_capturing():\n                    _RT[\"skipped_capturing\"] += 1\n                else:\n                    ids = out[1]\n                    lay = _rt_layer(getattr(self, \"nb1_layer\", None))\n                    if _RT[\"first\"] is None:\n                        _RT[\"first\"] = lay\n                    n = int(ids.shape[0])\n                    if n > _RT_DECODE_MAX_ROWS:\n                        _RT[\"skipped_prefill\"] += 1\n                    else:\n                        if lay == _RT[\"first\"]:\n                            _RT[\"passes\"] += 1     # one pass of a micro-batch through this stage's first MoE layer\n                        keep = ids.to(torch.int16).cpu().numpy().reshape(n, -1)\n                        _RT[\"rows\"].append((lay, max(0, _RT[\"passes\"] - 1), n, keep))\n                        _RT[\"bytes\"] += keep.nbytes\n                        _RT[\"calls\"] += 1\n            except Exception as e:\n                if len(_RT[\"errors\"]) < 5:\n                    _RT[\"errors\"].append(repr(e)[:160])\n        return out\n    return select_experts\n\n\nclass NB1WorkerExt:\n    def nb1_routing(self, action, arg=\"\"):\n        \"\"\"Router top-k expert ids, ONLY between 'start' and 'stop' (the profiling window): FusedMoERouter.select_experts is\n        wrapped once at class level; every decode / verify-sized call (<= 128 token rows) that runs in Python (not during\n        a CUDA-graph capture) keeps its ids (int16) with the layer index and the pass index (counted at this stage's first\n        MoE layer; NB1 maps pass -> step = pass // 2, mb_index = pass % 2 under PP2); <= max_steps steps, <= 16 MiB per\n        rank. FULL-graph replays run no Python, so graph-replayed decode steps are NOT recorded (vLLM's\n        --enable-return-routed-experts writes the ids from inside the graph; it changes the forward graph and memory, so it\n        is not used on a measured boot). 'stop' writes the rank's part file <arg>_rank<r>.npz.\"\"\"\n        out = {\"action\": action}\n        try:\n            import torch.distributed as dist\n            rank = dist.get_rank() if dist.is_initialized() else 0\n            out[\"rank\"] = rank\n            if action == \"start\":\n                from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter\n                if not _RT[\"installed\"]:\n                    FusedMoERouter.select_experts = _rt_wrap(FusedMoERouter.select_experts)\n                    _RT[\"installed\"] = True\n                    model = getattr(getattr(self, \"model_runner\", None), \"model\", None)\n                    named = 0\n                    if model is not None:\n                        for name, mod in model.named_modules():\n                            for attr in (\"router\", \"_router\"):\n                                r = getattr(mod, attr, None)\n                                if isinstance(r, FusedMoERouter):\n                                    r.nb1_layer = name\n                                    named += 1\n                    out[\"routers_named\"] = named\n                _RT.update(on=True, rows=[], passes=0, calls=0, bytes=0, errors=[], first=None, skipped_capturing=0,\n                           skipped_prefill=0, max_steps=int(arg or 200))\n                return out\n            _RT[\"on\"] = False\n            out.update(passes=_RT[\"passes\"], calls=_RT[\"calls\"], bytes=_RT[\"bytes\"], errors=_RT[\"errors\"],\n                       skipped_capturing=_RT[\"skipped_capturing\"], skipped_prefill=_RT[\"skipped_prefill\"])\n            if action == \"stop\" and arg and _RT[\"rows\"]:\n                import numpy as np\n                rows = _RT[\"rows\"]\n                offs, o_ = [], 0\n                for r in rows:\n                    offs.append(o_)\n                    o_ += r[2]\n                path = f\"{arg}_rank{rank}.npz\"\n                np.savez_compressed(path, ids=np.concatenate([r[3] for r in rows], axis=0), layer=np.array([r[0] for r in rows]),\n                                    pass_index=np.array([r[1] for r in rows]), mb_size=np.array([r[2] for r in rows]),\n                                    offset=np.array(offs), rank=np.array([rank]))\n                out[\"path\"] = path\n            _RT[\"rows\"] = []\n        except Exception as e:\n            out[\"error\"] = repr(e)[:300]\n        return out\n\n    def nb1_cache_info(self):\n        out = {}\n        try:\n            import torch.distributed as dist\n            out[\"rank\"] = dist.get_rank() if dist.is_initialized() else None\n        except Exception as e:\n            out[\"rank_error\"] = repr(e)[:120]\n        try:\n            ctx = self.vllm_config.compilation_config.static_forward_context\n            sig = {}\n            for name, mod in ctx.items():\n                ts = [_t(x) for x in _flat(getattr(mod, \"kv_cache\", None))]\n                if not ts:\n                    continue\n                key = type(mod).__name__ + \"|\" + json.dumps([(x[\"dtype\"], x[\"shape\"]) for x in ts])\n                g = sig.setdefault(key, {\"module\": type(mod).__name__, \"tensors\": ts, \"layers\": 0, \"example\": name})\n                g[\"layers\"] += 1\n            out[\"cache_tensors\"] = list(sig.values())[:40]\n        except Exception as e:\n            out[\"cache_tensors_error\"] = repr(e)[:300]\n        try:\n            agg = {}\n            for name, sp in self.get_kv_cache_spec().items():\n                d = {\"type\": type(sp).__name__}\n                for k in (\"block_size\", \"page_size_bytes\", \"page_size_padded\", \"dtype\", \"num_kv_heads\", \"head_size\", \"mamba_type\"):\n                    try:\n                        v = getattr(sp, k, None)\n                        d[k] = v if isinstance(v, (int, float, str, bool, type(None))) else str(v)\n                    except Exception as e:\n                        d[k] = \"ERR \" + repr(e)[:60]\n                shapes, dtypes = getattr(sp, \"shapes\", None), getattr(sp, \"dtypes\", None)\n                if shapes and dtypes:\n                    try:\n                        d[\"state_dtypes\"] = [str(x).replace(\"torch.\", \"\") for x in dtypes]\n                        d[\"state_shapes\"] = [list(s) for s in shapes]\n                        d[\"unpadded_page_bytes\"] = int(sum(math.prod(s) * x.itemsize for s, x in zip(shapes, dtypes)))\n                    except Exception as e:\n                        d[\"unpadded_error\"] = repr(e)[:80]\n                key = json.dumps(d, sort_keys=True, default=str)\n                a = agg.setdefault(key, dict(d, layers=0, example=name))\n                a[\"layers\"] += 1\n            out[\"kv_cache_specs\"] = list(agg.values())[:40]\n        except Exception as e:\n            out[\"kv_cache_specs_error\"] = repr(e)[:300]\n        try:\n            out[\"num_gpu_blocks\"] = getattr(self.cache_config, \"num_gpu_blocks\", None)\n            out[\"mamba_page_size_padded\"] = getattr(self.cache_config, \"mamba_page_size_padded\", None)\n            out[\"mamba_ssm_cache_dtype\"] = str(getattr(self.cache_config, \"mamba_ssm_cache_dtype\", None))\n            out[\"cache_block_size\"] = getattr(self.cache_config, \"block_size\", None)\n        except Exception as e:\n            out[\"cache_config_error\"] = repr(e)[:200]\n        try:\n            for obj in (getattr(self, \"model_runner\", None), self):\n                cfg = getattr(obj, \"kv_cache_config\", None)\n                if cfg is not None:\n                    out[\"kv_cache_config\"] = {\n                        \"num_blocks\": getattr(cfg, \"num_blocks\", None),\n                        \"groups\": [{\"spec\": type(g.kv_cache_spec).__name__, \"layers\": len(g.layer_names),\n                                    \"block_size\": getattr(g.kv_cache_spec, \"block_size\", None),\n                                    \"page_size_bytes\": getattr(g.kv_cache_spec, \"page_size_bytes\", None)}\n                                   for g in getattr(cfg, \"kv_cache_groups\", [])][:16],\n                        \"tensors\": [{\"size\": getattr(t, \"size\", None), \"shared_by\": len(getattr(t, \"shared_by\", None) or [])}\n                                    for t in getattr(cfg, \"kv_cache_tensors\", [])][:24]}\n                    break\n        except Exception as e:\n            out[\"kv_cache_config_error\"] = repr(e)[:300]\n        try:\n            import torch\n            fr, tot = torch.cuda.mem_get_info()\n            out[\"mem\"] = {\"free_mib\": fr >> 20, \"total_mib\": tot >> 20, \"allocated_mib\": torch.cuda.memory_allocated() >> 20,\n                          \"reserved_mib\": torch.cuda.memory_reserved() >> 20, \"max_allocated_mib\": torch.cuda.max_memory_allocated() >> 20}\n        except Exception as e:\n            out[\"mem_error\"] = repr(e)[:200]\n        return out\n"

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


    def commit_mode(port, mode):
        """Merlin's runtime RecoverSSM commit switch (gpu_worker.set_recoverssm_commit_mode: eager | staged | graph); returns
        the previous arm's counters (eager_steps, staged_steps, replays, captures, fallbacks) per rank."""
        return rpc(port, "set_recoverssm_commit_mode", mode)


    def commit_ab(port, s):
        """Separately labelled lever (B2 only): RecoverSSM commit graph. The boot captured commit graphs
        (VLLM_RECOVERSSM_COMMIT_GRAPH=graph) and ran every standard window eager; here 8k@8 eager vs graph on the same prompts,
        the per-arm commit counters (a PP-deferred commit always runs eager: replays may stay 0), and a greedy row under each
        mode (the docstring promises bit-identical state)."""
        out = {"lever": "RecoverSSM commit graph (runtime A/B/C)", "windows": {}, "counters": {}}
        g_eager = nb1_complete(port, PROMPTS["2k"][2], 64, temperature=0.0, logprobs=2)
        for mode in ("eager", "staged", "graph"):
            if not due(WIN + 240):
                out["windows"][mode] = {"status": "SKIPPED_BOOT_BUDGET"}
                continue
            out.setdefault("set", {})[mode] = commit_mode(port, mode)         # returns the counters before this arm
            out["windows"][mode] = window(port, s, "8k", 8)
            if mode == "graph":
                out["greedy_graph_vs_eager"] = lp_compare(g_eager, nb1_complete(port, PROMPTS["2k"][2], 64, temperature=0.0, logprobs=2))
            out["counters"][mode] = commit_counter_delta((out["set"][mode] or {}).get("results"),
                                                         (rpc(port, "get_recoverssm_commit_stats") or {}).get("results"))
        out["stats_after"] = rpc(port, "get_recoverssm_commit_stats")
        out["reset_to_eager"] = commit_mode(port, "eager")
        w = out["windows"]
        out["graph_replay_share"] = (out["counters"].get("graph") or {}).get("replay_share")
        if all((w.get(m) or {}).get("status") == "STEADY" for m in ("eager", "graph")):
            out["delta_tok_s"] = round(w["graph"]["tok_s"] - w["eager"]["tok_s"], 1)
            out["delta_tick_ms"] = round((w["graph"].get("tick_ms") or 0) - (w["eager"].get("tick_ms") or 0), 2)
        return out


    def commit_counter_delta(before, after):
        """Per arm, summed over ranks: commits graph-replayed vs run eager / staged, and the fallbacks by reason."""
        try:
            b = {k: sum((x or {}).get(k, 0) for x in before) for k in ("eager_steps", "staged_steps", "replays")}
            a = {k: sum((x or {}).get(k, 0) for x in after) for k in ("eager_steps", "staged_steps", "replays")}
        except Exception as e:
            return {"error": repr(e)[:200]}
        d = {k: a[k] - b[k] for k in a}
        tot = sum(d.values())
        fb = {}
        for x in after or []:
            for k, v in ((x or {}).get("fallbacks") or {}).items():
                fb[k] = fb.get(k, 0) + v
        return dict(d, replay_share=round(d["replays"] / tot, 3) if tot else None, fallbacks_total_after=fb)


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


    def pool_blocks_from_usage(us):
        vals = sorted({round(u, 9) for u in us if isinstance(u, (int, float)) and 0 < u < 1})
        if len(vals) < 2:
            return None
        for p in range(20, 3001):
            if all(abs(u * p - round(u * p)) < 0.02 for u in vals):
                return p
        return None


    def reset_cache(port):
        """POST /reset_prefix_cache (dev mode): the 'caches cleared' state of the audits; fails while requests run."""
        try:
            r = L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/reset_prefix_cache", data=b"", method="POST"), timeout=120).read()
            return {"ok": True, "body": r[:200].decode(errors="replace")}
        except Exception as e:
            return {"ok": False, "error": repr(e)[:200]}


    def _hits(port):
        try:
            return L.prom(port).get("vllm:prefix_cache_hits_total", 0.0)
        except Exception:
            return None


    def alloc_sequence(port, band, counts):
        """Cold cache: reset, idle snapshot, then add distinct-token requests (no shared prefix) to 1, 2, 4 (, 8) resident
        requests, snapshot after each step once all are past prefill, release, snapshot, reset, snapshot. Block numbers
        follow from vllm:kv_cache_usage_perc x P (P recovered from the usage denominators)."""
        vs = list(V_CENSUS[band])
        rec = {"band": band, "counts": list(counts), "snapshots": []}
        if not L.wait_idle(port, 120):
            rec["status"] = "INCONCLUSIVE_NOT_IDLE"
            return rec
        rec["reset_before"] = reset_cache(port)
        time.sleep(1)
        rec["idle_usage"] = [kvu(port) for _ in range(3)]
        h0, pre0 = _hits(port), L.prom(port).get(L.PREEMPT, 0)
        ss = []
        try:
            for c in counts:
                if c > len(vs):
                    rec["snapshots"].append({"requests": c, "status": "SKIPPED_NOT_ENOUGH_PROMPTS"})
                    break
                if not due(60 + (c - len(ss)) * BANDS[band] / max(500.0, PREFILL["tok_s"])):
                    rec["snapshots"].append({"requests": c, "status": "SKIPPED_BOOT_BUDGET"})
                    break
                while len(ss) < c:
                    g = L.Stream(port, PROMPTS[band][vs[len(ss)]], 8192, temperature=TEMP, seed=700 + len(ss), ignore_eos=True)
                    g.start()
                    ss.append(g)
                t0 = time.time()
                while time.time() - t0 < 600:
                    time.sleep(1.5)
                    m = L.prom(port)
                    if int(m.get(L.RUN, 0)) >= c and all(g.t_first for g in ss):
                        break
                    if any(not g.is_alive() for g in ss):
                        break
                tk0 = [len(g.tokens) for g in ss]
                m = L.prom(port)
                tk1 = [len(g.tokens) for g in ss]
                rec["snapshots"].append({"requests": c, "usage": m.get(L.KVU), "running": m.get(L.RUN), "waiting": m.get(L.WAIT),
                                         "tokens": [len(PROMPTS[band][v]) + (a + b) / 2 for v, a, b in zip(vs, tk0, tk1)],
                                         "tokens_spread": max((b - a for a, b in zip(tk0, tk1)), default=0),
                                         "prefix_hits_delta": (_hits(port) or 0) - (h0 or 0), "admit_sec": round(time.time() - t0, 1)})
        finally:
            for g in ss:
                g.stop()
            for g in ss:
                g.join(10)
            rec["stream_errors"] = sorted({g.err for g in ss if g.err})[:3]
        L.wait_idle(port, 60)
        rec["released_usage"] = kvu(port)
        rec["reset_after"] = reset_cache(port)
        time.sleep(1)
        rec["after_reset_usage"] = kvu(port)
        rec["preemptions"] = L.prom(port).get(L.PREEMPT, 0) - pre0
        rec["status"] = "MEASURED" if any("usage" in s for s in rec["snapshots"]) else "INCONCLUSIVE"
        return rec


    def reset_succeeded(r):
        """R2: POST /reset_prefix_cache answered {"success": true}. Merlin's block_pool.reset_prefix_cache refuses unless the
        pool holds no block but the null block, so a success proves 0 used blocks at that moment."""
        if not isinstance(r, dict) or r.get("ok") is not True:
            return False
        try:
            return json.loads(r.get("body") or "").get("success") is True
        except Exception:
            return False


    def fit_alloc(seqs, P, B):
        """Per snapshot: F = (used - idle - sum ceil(t / B)) / n (blocks beyond the token blocks, per request); least squares
        used - idle = a * sum ceil(t / B) + F * n over every snapshot (a = 1 when singular).
        R2: idle = 0 blocks when the reset before the sequence succeeded: vllm:kv_cache_usage_perc is refreshed only on engine
        steps, so the idle reading right after the reset is the stale value of the last step (B1: 45.6 % / 26.8 % while the
        pool was empty -> F -5.32). Without a successful reset the gauge reading is used (recorded as idle_source)."""
        xs, per = [], []
        for sq in seqs:
            if not sq.get("idle_usage") or sq.get("status") != "MEASURED":
                continue
            gauge_idle = round(statistics.median([u for u in sq["idle_usage"] if u is not None]) * P)
            sq["idle_gauge_blocks"] = gauge_idle
            if reset_succeeded(sq.get("reset_before")):
                idle, sq["idle_source"] = 0, "reset_prefix_cache success (0 used blocks; the idle gauge is stale until the next step)"
            else:
                idle, sq["idle_source"] = gauge_idle, "kv_cache_usage gauge (reset not confirmed)"
            sq["idle_blocks"] = idle
            sq["released_blocks"] = round((sq.get("released_usage") or 0) * P)
            sq["retained_after_release_blocks"] = sq["released_blocks"] - idle
            sq["after_reset_blocks"] = round((sq.get("after_reset_usage") or 0) * P)
            prev = None
            for s in sq["snapshots"]:
                if "usage" not in s or s.get("usage") is None:
                    continue
                used = round(s["usage"] * P)
                tb = sum(math.ceil(t / B) for t in s["tokens"])
                s.update(used_blocks=used, free_blocks=P - used, token_blocks=tb, F=round((used - idle - tb) / s["requests"], 2),
                         all_resident=s.get("running") == s["requests"] and not s.get("waiting"))
                if prev:
                    s["marginal_blocks_per_added_request"] = round((used - prev[0]) / (s["requests"] - prev[1]), 2)
                prev = (used, s["requests"])
                if s["all_resident"]:
                    xs.append((tb, s["requests"], used - idle))
                    per.append(s["F"])
        if not xs:
            return None
        s11 = sum(a * a for a, _, _ in xs); s12 = sum(a * n for a, n, _ in xs); s22 = sum(n * n for _, n, _ in xs)
        b1 = sum(a * y for a, _, y in xs); b2 = sum(n * y for _, n, y in xs)
        det = s11 * s22 - s12 * s12
        if abs(det) > 1e-9:
            a, F = (b1 * s22 - b2 * s12) / det, (s11 * b2 - s12 * b1) / det
        else:
            a, F = 1.0, statistics.mean(per)
        res = [y - (a * x + F * n) for x, n, y in xs]
        return {"a": round(a, 3), "F": round(F, 2), "F_at_a1": round(statistics.mean(per), 2), "F_per_snapshot": per,
                "samples": len(xs), "max_abs_residual_blocks": round(max(abs(r) for r in res), 2),
                "residuals_blocks": [round(r, 2) for r in res],                 # R2 audit N1-6: the fit's residual per snapshot
                "singular": abs(det) <= 1e-9}


    def kv_accounting(port, caps, plan):
        B = caps.get("block_size") or BLOCK
        seqs = []
        for band, counts in plan:
            try:
                seqs.append(alloc_sequence(port, band, counts))
            except Exception as e:
                seqs.append({"band": band, "status": "ERROR", "error": f"{type(e).__name__}: {e}"[:300]})
        us = [u for sq in seqs for u in (sq.get("idle_usage") or []) + [s.get("usage") for s in sq.get("snapshots") or []]
              + [sq.get("released_usage"), sq.get("after_reset_usage")]]
        P = pool_blocks_from_usage(us)
        out = {"sequences": seqs, "pool_blocks": P, "block_size": B, "method": "cold cache (reset_prefix_cache) + distinct "
               "prompts; blocks = kv_cache_usage x P, P recovered from the usage denominators (cross-check: worker num_gpu_blocks)"}
        out["fit"] = fit_alloc(seqs, P, B) if P else None
        if not P:
            out["why"] = "pool blocks not recoverable from the usage readings"
        return out


    def n_fit(P, F, B, band, idle_blocks=0, seqs=None, n_prompts=None):
        """Streams of `band` whose final length (prompt + ~(admission + window + 60 s) x GEN_RATE) all fit in the pool."""
        if P is None or F is None:
            return None
        L0 = BANDS[band]
        final = L0 + (L0 * 8 / max(1.0, PREFILL["tok_s"]) + WIN + 60) * GEN_RATE
        n = int((P - idle_blocks) // (math.ceil(final / B) + F))
        for cap in (seqs, n_prompts):
            if cap:
                n = min(n, cap)
        return max(0, n)


    def admission_test(port, s, band="90k", n=5, max_new=4096):
        """Fresh-cache admission of n distinct long prompts (audit 2: 90k x 5 before any pin change). Reset the cache, start
        n streams, wait until every one has its first token (admission + cold prefill, ~450k tokens: Merlin's cost-aware
        admission may hold a cold miss up to 120 s), then: a resident snapshot (running == n, waiting 0, no preemption, blocks)
        and a short decode window measured on the counters while all n stay resident. max_new 4096 keeps 5 x (ceil(94,096 /
        4096) + 6) = 145 of 149 blocks: the streams cannot outgrow the pool (a lab steady window cannot be used: the first
        stream would finish its max_new inside it, since the cold prefills are staggered by ~20 s each)."""
        if not due(240 + n * BANDS[band] / max(2000.0, PREFILL["tok_s"])):
            return {"status": "SKIPPED_BOOT_BUDGET"}
        if not L.wait_idle(port, 120):
            return {"status": "INCONCLUSIVE_NOT_IDLE"}
        rec = {"band": band, "n": n, "max_new": max_new, "reset": reset_cache(port)}
        time.sleep(1)
        rec["idle_usage"] = kvu(port)
        m0, l0, t0 = L.prom(port), _req_lines(), time.time()
        ss = [L.Stream(port, PROMPTS[band][i], max_new, temperature=TEMP, seed=500 + i, ignore_eos=True) for i in range(n)]
        for g in ss:
            g.start()
        try:
            limit = max(420.0, 180.0 + n * BANDS[band] / max(1000.0, PREFILL["tok_s"]))
            while time.time() - t0 < limit:
                time.sleep(2)
                if all(g.t_first for g in ss) or any(not g.is_alive() for g in ss):
                    break
            m1 = L.prom(port)
            rec.update(admission_sec=round(time.time() - t0, 1), first_token=[round(g.t_first - g.t_start, 1) if g.t_first else None for g in ss],
                       snapshot={"running": m1.get(L.RUN), "waiting": m1.get(L.WAIT), "usage": m1.get(L.KVU),
                                 "tokens": [len(PROMPTS[band][i]) + len(g.tokens) for i, g in enumerate(ss)]},
                       preemptions=m1.get(L.PREEMPT, 0) - m0.get(L.PREEMPT, 0),
                       prefill_uncached_tok_s=round(((m1.get(L.PROMPT, 0) - m0.get(L.PROMPT, 0)) - (m1.get("vllm:prefix_cache_hits_total", 0)
                                                     - m0.get("vllm:prefix_cache_hits_total", 0))) / max(1e-3, time.time() - t0), 1))
            rec["all_resident"] = (all(g.t_first for g in ss) and rec["snapshot"]["running"] == n and not rec["snapshot"]["waiting"]
                                   and not rec["preemptions"])
            if rec["all_resident"]:
                secs = min(20.0, WIN)
                g0, w0, runs = L.prom(port), time.time(), []
                while time.time() - w0 < secs:
                    time.sleep(min(2.0, secs))
                    runs.append(L.prom(port).get(L.RUN))
                g1 = L.prom(port)
                dt = time.time() - w0
                steps = g1.get(L.ITER, 0) - g0.get(L.ITER, 0)
                rec["window"] = {"sec": round(dt, 1), "tok_s": round((g1.get(L.GEN, 0) - g0.get(L.GEN, 0)) / dt, 1), "running_min": min(runs or [0]),
                                 "running_max": max(runs or [0]), "tick_ms": round(dt * 1000 / steps, 1) if steps else None,
                                 "preemptions": g1.get(L.PREEMPT, 0) - g0.get(L.PREEMPT, 0)}
                if rec["window"]["tick_ms"]:
                    rec["window"]["cycle_ms"] = round(PP * rec["window"]["tick_ms"], 1)
                rec["status"] = ("ALL_RESIDENT" if rec["window"]["running_min"] == n and not rec["window"]["preemptions"]
                                 else "INCONCLUSIVE_RUNNING_CHANGED")
            else:
                rec["status"] = "NOT_ALL_RESIDENT"
        finally:
            for g in ss:
                g.stop()
            for g in ss:
                g.join(10)
            rec["stream_errors"] = sorted({g.err for g in ss if g.err})[:3]
        rec["ttft_per_request"] = _ttfts(port, l0)
        return rec


    def residency_bound(P, F, band, max_new=0, seqs=None):
        """Arithmetic bound: running requests of total context band + max_new that fit P blocks at F fixed blocks each."""
        if not P or F is None:
            return None
        n = int(P // (math.ceil((BANDS[band] + max_new) / BLOCK) + F))
        return min(n, seqs) if seqs else n


    def residency_capacity(o):
        """E25 NB1 C-6: this boot's own measured pool P (kv_accounting: the kv_cache_usage gauge denominators) and F (resolve_F:
        the engine's max-concurrency line cross-checked with the cold-sequence fit). Never RC2's 149 / 6.
        R2 audit N1-2: a certified capacity needs pool_check == MATCH (worker num_gpu_blocks - 1 == the gauge pool on every
        rank); MISMATCH or no worker reading -> ok False (INCONCLUSIVE_CAPACITY_UNKNOWN, the reason in why, no probes).
        R2 audit N1-4 / N1-6: F from resolve_F; a fit / line conflict -> ok False (F_FIT_LOG_CONFLICT)."""
        kv = o.get("kv") or {}
        P = kv.get("pool_blocks")
        wbs = (o.get("workers") or {}).get("num_gpu_blocks") or []
        wbs = wbs if isinstance(wbs, list) else []
        usable = sorted({w - 1 for w in wbs if isinstance(w, int) and not isinstance(w, bool) and w > 1})
        if not usable or not isinstance(P, int) or isinstance(P, bool):
            pc = "UNKNOWN"
        else:
            pc = "MATCH" if usable == [P] else "MISMATCH"
        cap = {"pool_blocks": P, "worker_num_gpu_blocks": wbs, "worker_usable_blocks": usable,
               "pool_check": pc, "pool_check_rule": "worker num_gpu_blocks - 1 (null block) == gauge pool on every rank; "
               "a certified capacity needs MATCH (R2 audit N1-2)"}
        if pc == "MISMATCH":
            cap["pool_mismatch"] = f"gauge pool {P} vs worker usable {usable}"
        why, codes = [], []
        P_ok = isinstance(P, int) and not isinstance(P, bool) and P >= 1
        if not P_ok:
            why.append("pool_blocks not measured in this boot (kv.pool_blocks)")
            codes.append("POOL_NOT_MEASURED")
        elif pc == "MISMATCH":
            why.append(f"POOL_MISMATCH: {cap['pool_mismatch']} (capacity not certified)")
            codes.append("POOL_MISMATCH")
        elif pc != "MATCH":
            why.append("POOL_UNVERIFIED: no worker num_gpu_blocks reading to check the gauge pool against "
                       f"({str((o.get('workers') or {}).get('error') or 'no num_gpu_blocks')[:120]})")
            codes.append("POOL_UNVERIFIED")
        fr = resolve_F(o, P if P_ok else None)
        cap.update({k: v for k, v in fr.items() if k not in ("F", "F_source", "why", "codes")})
        cap["F_resolved"] = fr["F"]                   # F by the fit / line rule alone (core_check), before the pool gate
        if fr["F"] is None:
            why.extend(fr["why"] or [])
            codes.extend(fr["codes"] or [])
        ok = not why
        cap.update(ok=ok, F=fr["F"] if ok else None, F_source=fr["F_source"] if ok else None, why=why or None,
                   reason_codes=codes or None, status="CERTIFIED" if ok else "INCONCLUSIVE_CAPACITY_UNKNOWN")
        return cap


    F_MAX_BLOCKS = 40                                # R2: a per-request fixed-block F above 40 blocks is not physical here
    FIT_MIN_SAMPLES = 3                              # R2 audit N1-6: a fit is trusted only with >= 3 all-resident snapshots,
    FIT_MAX_RESID_BLOCKS = 2.0                       # a max |residual| <= 2 blocks
    FIT_LOG_TOL_BLOCKS = 1                           # and |ceil(fit F) - log F| <= 1 block
    F_LOG_SOURCE = "server max-concurrency line (this boot)"


    def _f_valid(F):
        return isinstance(F, (int, float)) and not isinstance(F, bool) and math.isfinite(F) and 0 <= F <= F_MAX_BLOCKS


    def resolve_F(o, P):
        """R2 audit N1-4 / N1-6: F from this boot's two sources. The fit (kv.fit, fit_alloc) counts only when valid (finite,
        0..40); it is trusted only when samples >= 3, max |residual| <= 2 blocks and |ceil(fit) - log F| <= 1.
          both valid, |ceil(fit) - log F| > 1     -> F None, F_FIT_LOG_CONFLICT (no probes);
          both valid, agree, fit trusted          -> the log F (engine-derived), fit_check AGREES;
          both valid, agree, fit not trusted      -> the log F, fit_check NOT_TRUSTED (samples / residual);
          only the log F (fit missing / invalid)  -> the log F;
          only a fit, trusted (samples, residual) -> ceil(fit), labelled 'fit only, no line to cross-check';
          otherwise                               -> F None."""
        kv = o.get("kv") or {}
        fit = kv.get("fit") if isinstance(kv.get("fit"), dict) else {}
        F_fit = fit.get("F")
        smp, res = fit.get("samples"), fit.get("max_abs_residual_blocks")
        r = {"F_fit": F_fit, "fit_a": fit.get("a"), "fit_samples": smp, "fit_max_abs_residual_blocks": res,
             "fit_residuals_blocks": fit.get("residuals_blocks"),
             "fit_trust_rule": f"samples >= {FIT_MIN_SAMPLES}, max |residual| <= {FIT_MAX_RESID_BLOCKS:g} blocks, "
                               f"|ceil(fit F) - log F| <= {FIT_LOG_TOL_BLOCKS}"}
        fit_valid = _f_valid(F_fit)
        if F_fit is not None and not fit_valid:
            r["F_fit_invalid"] = F_fit
        fit_q = (fit_valid and isinstance(smp, int) and not isinstance(smp, bool) and smp >= FIT_MIN_SAMPLES
                 and isinstance(res, (int, float)) and not isinstance(res, bool) and math.isfinite(res) and res <= FIT_MAX_RESID_BLOCKS)
        qwhy = None if fit_q else f"fit samples {smp!r} (need >= {FIT_MIN_SAMPLES}), max |residual| {res!r} (need <= {FIT_MAX_RESID_BLOCKS:g})"
        F_c = int(math.ceil(F_fit - 1e-9)) if fit_valid else None
        r["F_fit_ceil"] = F_c
        fwhy = "F not measured in this boot (kv.fit.F)" if F_fit is None else (None if fit_valid else f"F fit invalid ({F_fit!r})")
        lg = f_from_log(o, P) if P else None          # R2: this boot's own max-concurrency line (nb5_lib.capacity_plan rule)
        r["F_log"] = lg
        log_ok = bool(lg and lg.get("ok"))
        F, src, why, codes = None, None, [], []
        if fit_valid and log_ok:
            d = abs(F_c - lg["F"])
            r["fit_log_diff_blocks"] = d
            if d > FIT_LOG_TOL_BLOCKS:
                r["fit_check"] = "CONFLICT"
                r["F_conflict"] = f"F_FIT_LOG_CONFLICT: ceil(fit F {F_fit}) = {F_c} vs log F {lg['F']} (differ by {d} > {FIT_LOG_TOL_BLOCKS})"
                why.append(r["F_conflict"])
                codes.append("F_FIT_LOG_CONFLICT")
            else:
                F, src = lg["F"], F_LOG_SOURCE
                r["fit_check"] = "AGREES" if fit_q else "NOT_TRUSTED"
                r["fit_agrees"] = fit_q
                if not fit_q:
                    r["fit_not_trusted_reason"] = qwhy
        elif log_ok:
            F, src = lg["F"], F_LOG_SOURCE
            r["fit_check"] = "MISSING" if F_fit is None else "INVALID"
            r["F_fallback_reason"] = fwhy
        elif fit_q:
            F = F_c
            src = ("measured in this boot: kv.fit.F (cold-sequence fit) rounded up to an integer; fit only (trusted: samples / "
                   "residual), no max-concurrency line to cross-check")
            r["fit_check"] = "FIT_ONLY_TRUSTED"
        else:
            r["fit_check"] = ("MISSING" if F_fit is None else "INVALID" if not fit_valid else "NOT_TRUSTED")
            if fwhy:
                why.append(fwhy)
            elif not fit_q:
                why.append("F fit not trusted without a max-concurrency line to cross-check: " + qwhy)
            if P:
                why.append("F from the server max-concurrency line unavailable: " + str((lg or {}).get("why")))
            codes.append("F_UNAVAILABLE")
        r.update(F=F, F_source=src, why=why or None, codes=codes or None)
        return r


    def f_from_log(o, P):
        """R2: F_log = P / X - ceil(M / B) from this boot's line 'GPU KV cache size: N tokens, Maximum concurrency for M tokens
        per request: X x' (X = N / M when N is there: the printed X has 2 decimals); F = ceil(F_log - 0.05) like
        nb5_lib.capacity_plan (RC2 / B1: 149 x 131,072 / 513,940 - 32 = 5.999 -> 6)."""
        pool = o.get("pool") or {}
        N, X, M = pool.get("kv_tokens"), pool.get("max_concurrency"), pool.get("max_concurrency_tokens")
        B = (o.get("caps") or {}).get("block_size") or BLOCK
        if not (isinstance(M, int) and M > 0 and ((isinstance(N, int) and N > 0) or (isinstance(X, (int, float)) and X > 0))):
            return {"ok": False, "why": f"no max-concurrency line in this boot's log (N {N}, M {M}, X {X})"}
        xx = N / M if isinstance(N, int) and N > 0 else float(X)
        f_log = P / xx - math.ceil(M / B)
        F = max(0, int(math.ceil(f_log - 0.05)))
        ok = math.isfinite(f_log) and -0.05 <= f_log <= F_MAX_BLOCKS
        return {"ok": ok, "F": F if ok else None, "F_log": round(f_log, 3), "kv_tokens": N, "max_concurrency": X, "tokens_per_request": M,
                "block_size": B, "x_used": round(xx, 5), "why": None if ok else f"F_log {f_log:.3f} outside 0..{F_MAX_BLOCKS}"}


    def residency_probes(port, s, o, bands=RESIDENCY_BANDS):
        """max_safe_running_by_context (addendum 7): per context, cold cache + distinct prompts, n = the arithmetic bound at
        this boot's measured pool and F (residency_capacity; capped by max_num_seqs and the distinct prompts), max_new sized
        so n requests cannot outgrow the pool; an n whose max_new is below the decode growth expected during the staggered
        cold admission + the window is SKIPPED_HEADROOM (the arithmetic bound is reported apart); achieved = n if every stream
        is resident after prefill with real running == n and 0 preemptions, else one retry at n - 1. The 90k probe reuses the
        90k x 5 admission when it ran at the bound. Pool or F unknown -> INCONCLUSIVE_CAPACITY_UNKNOWN, no probe requests."""
        cap = residency_capacity(o)                    # E25 NB1 C-6: the boot's own P and F, no RC2 fallback
        P, F = cap["pool_blocks"], cap["F"]
        seqs = (o.get("caps") or {}).get("max_num_seqs")
        out = {"pool_blocks": P, "F": F, "F_fit": cap["F_fit"], "capacity": cap, "bands": {},
               "F_source": cap.get("F_source")}         # R2: the fit or this boot's max-concurrency line
        if not cap["ok"]:
            out.update(status="INCONCLUSIVE_CAPACITY_UNKNOWN", F_source=None, why=cap["why"])
            return out
        out["status"] = "MEASURED"
        for band in bands:
            nb = residency_bound(P, F, band, 0, seqs)
            n = min(nb or 0, len(PROMPTS[band]))
            rec = {"bound_prompt_only": residency_bound(P, F, band), "bound_capped": nb, "tries": []}
            adm = o.get("admission_90k@5") or {}
            if band == "90k" and n == 5 and adm.get("status") in ("ALL_RESIDENT", "NOT_ALL_RESIDENT", "INCONCLUSIVE_RUNNING_CHANGED"):
                rec["tries"].append({"n": 5, "status": adm.get("status"), "from": "admission_90k@5"})
                ok5 = adm.get("status") == "ALL_RESIDENT"
                rec["achieved"] = 5 if ok5 else None
                if ok5:
                    out["bands"][band] = rec
                    continue
                n = 4
            rate = float(os.environ["FAKE_SPEED"]) if FAKE and os.environ.get("FAKE_SPEED") else GEN_RATE
            while n >= 1:
                mn = int((P // n - F) * BLOCK - BANDS[band])
                mn = min(8192, mn)
                # the cold prefills are staggered: the first stream decodes during the others' admission + the window
                need = int((n * BANDS[band] / max(500.0, PREFILL["tok_s"]) + min(20.0, WIN) + 10) * rate)
                if mn < max(1024, need):
                    rec["tries"].append({"n": n, "status": "SKIPPED_HEADROOM", "max_new": mn, "need": need,
                                         "why": "the pool leaves too little decode headroom for a valid cold admission at this n"})
                    n -= 1
                    continue
                if not due(240 + n * BANDS[band] / max(2000.0, PREFILL["tok_s"])):
                    rec["tries"].append({"n": n, "status": "SKIPPED_BOOT_BUDGET"})
                    break
                a = admission_test(port, s, band, n, max_new=mn)
                rec["tries"].append({"n": n, "max_new": mn, "status": a.get("status"), "first_token_s": a.get("first_token"),
                                     "window": a.get("window"), "preemptions": a.get("preemptions"), "snapshot": a.get("snapshot")})
                if a.get("status") == "ALL_RESIDENT":
                    rec["achieved"] = n
                    break
                if len([x for x in rec["tries"] if x.get("status") not in ("SKIPPED_BOOT_BUDGET", "SKIPPED_HEADROOM")]) >= 2:
                    break
                n -= 1
            rec.setdefault("achieved", None)
            out["bands"][band] = rec
            save()
        return out


    def block_layout(workers, F=6):
        """The per-request fixed blocks against the kv cache groups the worker reports (spec type, layers, block size). F is
        ENGINE-DERIVED (concurrency line); the split per group is an inference from the group list, never a claim that the
        blocks are reclaimable."""
        ranks = (workers or {}).get("ranks") or []
        g = next(((w or {}).get("kv_cache_config") or {} for w in ranks if isinstance(w, dict) and (w.get("kv_cache_config") or {}).get("groups")), {})
        groups = g.get("groups") or []
        mamba = [x for x in groups if "mamba" in str(x.get("spec", "")).lower()]
        attn = [x for x in groups if x not in mamba]
        return {"groups": groups, "num_blocks": g.get("num_blocks"), "mamba_groups": len(mamba), "attention_groups": len(attn),
                "F_engine": F, "F_per_mamba_group_inferred": round(F / len(mamba), 2) if mamba else None,
                "note": "F from the engine's concurrency line; the per-group split is inferred from the group list; reclaimability "
                        "is not shown (no evidence)"}


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
    def nb1_complete(port, ids, max_new, temperature=0.0, seed=None, logprobs=0, timeout=1800):
        """Non-streaming /v1/completions with token ids: ids, usage, cached tokens (--enable-prompt-tokens-details) and, with
        logprobs, the chosen-token logprobs + top-1/top-2 margins. Every request is saved in traces/requests.jsonl."""
        body = {"model": "q27", "prompt": list(ids), "max_tokens": int(max_new), "temperature": temperature,
                "top_p": 0.95 if temperature else 1.0, "ignore_eos": True, "return_token_ids": True}
        if temperature:
            body["top_k"] = 20
        if seed is not None:
            body["seed"] = int(seed)
        if logprobs:
            body["logprobs"] = int(logprobs)
            body["return_tokens_as_token_ids"] = True
        t = time.time()
        try:
            r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                                   headers={"Content-Type": "application/json"}), timeout=timeout).read())
            ch = r["choices"][0]
            u = r.get("usage") or {}
            out = {"ids": list(ch.get("token_ids") or []), "text": ch.get("text") or "", "usage": u, "error": None,
                   "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens")}
            lp = ch.get("logprobs") or {}
            if logprobs:
                out["lps"] = list(lp.get("token_logprobs") or [])
                out["margins"] = []
                for top in lp.get("top_logprobs") or []:
                    v = sorted((x for x in (top or {}).values() if isinstance(x, (int, float))), reverse=True)
                    out["margins"].append(round(v[0] - v[1], 4) if len(v) >= 2 else None)
        except Exception as e:
            out = {"ids": [], "text": "", "usage": None, "cached": None, "error": repr(e)[:300]}
        out["sec"] = round(time.time() - t, 2)
        try:
            with _SAVE_LOCK, open(REQ_LOG, "a") as fh:
                fh.write(json.dumps({"kind": "nb1_complete", "port": port, "prompt_len": len(ids), "max_new": max_new,
                                     "prompt_sha": psha(ids)[:16], "text": out["text"], "tokens": out["ids"], "lps": out.get("lps"),
                                     "usage": out["usage"], "error": out["error"], "sec": out["sec"], "t": time.time()}) + "\n")
        except Exception:
            pass
        return out


    def nb1_tf_req(port, ids, last=256, timeout=1800):
        """e23_lab.teacher_forced's request and parsing (same body, same per-position [logprob, rank, top-1 id]) + the
        response's cached-token count. vLLM sets skip_reading_prefix_cache for prompt_logprobs requests
        (sampling_params.py), so a repeat recomputes; the cached path is tested in decode space (greedy fresh vs repeat)."""
        body = {"model": "q27", "prompt": list(ids), "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1,
                "return_tokens_as_token_ids": True}
        try:
            r = json.loads(L.U.urlopen(L.U.Request(f"http://127.0.0.1:{port}/v1/completions", data=json.dumps(body).encode(),
                                                   headers={"Content-Type": "application/json"}), timeout=timeout).read())
            pl = r["choices"][0].get("prompt_logprobs") or []
            cached = ((r.get("usage") or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
        except Exception as e:
            return {"error": repr(e)[:300]}
        out = []
        for pos in range(max(1, len(ids) - last), len(ids)):
            d = pl[pos] if pos < len(pl) else None
            if not isinstance(d, dict):
                out.append(None)
                continue
            e = d.get(str(ids[pos])) or d.get(f"token_id:{ids[pos]}")
            top1 = [L._tok_id(k) for k, v in d.items() if isinstance(v, dict) and v.get("rank") == 1]
            out.append([e.get("logprob"), e.get("rank"), top1[0] if len(top1) == 1 else None] if isinstance(e, dict) else None)
        return {"positions": out, "n": len(out), "cached_tokens": cached, "first_position": max(1, len(ids) - last)}


    def lp_compare(a, b):
        """Greedy rows a (reference) vs b: first differing token, the reference's top-1 - top-2 margin there (L.FAIL_MARGIN),
        mean / max |delta chosen-token logprob| over the identical prefix."""
        if not a or not b or a.get("error") or b.get("error") or not a.get("ids") or not b.get("ids"):
            return {"status": "INCONCLUSIVE"}
        x, y = a["ids"], b["ids"]
        d = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), None)
        la, lb = a.get("lps") or [], b.get("lps") or []
        k = d if d is not None else min(len(x), len(y))
        dl = [abs(p - q) for p, q in zip(la[:k], lb[:k]) if isinstance(p, (int, float)) and isinstance(q, (int, float))]
        if d is None:
            st, m = ("EXACT" if len(x) == len(y) else "INCONCLUSIVE"), None
        else:
            mm = a.get("margins") or []
            m = mm[d] if d < len(mm) else None
            st = "INCONCLUSIVE" if m is None else ("FAIL_HIGH_MARGIN" if m >= L.FAIL_MARGIN else "NEAR_TIE")
        return {"status": st, "first_diff": d, "ref_margin": m, "mean_abs_dlogprob_prefix": round(statistics.mean(dl), 5) if dl else None,
                "max_abs_dlogprob_prefix": round(max(dl), 5) if dl else None, "tokens": [len(x), len(y)],
                "cached": [a.get("cached"), b.get("cached")]}


    def fp8_self_check(s, key="b1", bands=TF_BANDS):
        """< 2 min FP8 self-consistency on the serving boot (the BF16 / FP8 isolation is NB0's): per band (2k, 8k) a greedy
        64-token row on a fresh prompt, the same request again (cached repeat; cached tokens recorded), and two
        teacher-forced passes (an FP8 A/A; prompt_logprobs never read the prefix cache). Raw rows -> tf_raw/<key>_<band>.json."""
        out = NB1.setdefault("fp8_self_check", {"bands": {}, "note": "BF16 / FP8 x MTP off / K3 teacher-forced isolation: NB0"})
        for b in bands:
            if not s.alive():
                out["bands"][b] = {"status": "SKIPPED_SERVER_DEAD"}
                continue
            if not due(120 + 3 * BANDS[b] / max(1000.0, PREFILL["tok_s"])):
                out["bands"][b] = {"status": "SKIPPED_BOOT_BUDGET"}
                continue
            ids = TF_P[b]
            rec = {"prompt_tokens": len(ids), "prompt_sha256": psha(ids)}
            da = nb1_complete(s.port, ids, 64, temperature=0.0, logprobs=2)
            db = nb1_complete(s.port, ids, 64, temperature=0.0, logprobs=2)
            rec["decode_cached_vs_fresh"] = lp_compare(da, db)
            t1 = nb1_tf_req(s.port, ids)
            t2 = nb1_tf_req(s.port, ids) if s.alive() else {"error": "server dead"}
            rec["tf_fresh"] = {"n": t1.get("n"), "error": t1.get("error"), "cached_tokens": t1.get("cached_tokens")}
            rec["tf_repeat"] = {"n": t2.get("n"), "error": t2.get("error"), "cached_tokens": t2.get("cached_tokens")}
            rec["tf_aa"] = L.compare_tf(t1, t2)
            try:
                (TF_RAW / f"{key}_{b}.json").write_text(json.dumps({"key": key, "band": b, "prompt_sha256": rec["prompt_sha256"],
                                                                    "prompt_tokens": len(ids), "tf_fresh": t1, "tf_repeat": t2,
                                                                    "decode_fresh": da, "decode_cached_repeat": db}))
            except Exception as e:
                R["errors"].append(f"tf_raw {key}/{b}: {e!r}"[:200])
            out["bands"][b] = rec
            save()
        return out


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


    def nb1_ladder(port, s, o, extra=()):
        caps = o.get("caps") or {}
        seqs = caps.get("max_num_seqs")
        cap = residency_capacity(o)                # R2: the validated F (fit, else this boot's max-concurrency line), never an invalid fit
        # R2 audit N1-2 / N1-4: the same rule as the probes: F is None unless the capacity is certified (pool MATCH, no
        # fit / line conflict), so an uncertified boot adds no capacity-sized level (the fixed 8k@3 / 8k@8 / 40k@5 still run)
        P, F, B = cap["pool_blocks"], cap["F"], caps.get("block_size") or BLOCK
        plan = [("8k", 3), ("8k", 8), ("40k", 5)] + list(extra)
        hi = {}
        for band, base in (("8k", 8), ("40k", 5)):
            nf = n_fit(P, F, B, band, 0, seqs, 24 if band == "8k" else 8)
            hi[band] = nf
            if nf and nf > base:
                plan.append((band, nf))
        out = {"_plan": [f"{b}@{n}" for b, n in plan], "_fit": hi, "_seqs_cap": seqs, "_F": F, "_F_source": cap.get("F_source"),
               "_capacity_status": cap.get("status"), "_F_check": cap.get("fit_check"),          # R2 audit N1-2 / N1-4
               "_capacity_why": cap.get("why")}
        for band, n in plan:
            k = f"{band}@{n}"
            if seqs and n > seqs:
                out[k] = {"status": "SKIPPED_ABOVE_SERVER_CAP", "max_num_seqs": seqs}
                continue
            if not due(WIN + 120 + n * BANDS[band] / max(500.0, PREFILL["tok_s"])):
                out[k] = {"status": "SKIPPED_BOOT_BUDGET"}
                continue
            out[k] = window(port, s, band, n)
        return out


    # ---- prefix continuation (2-turn replay) and after an abort
    def _prefix_counts(port):
        try:
            m = L.prom(port)
            return {k: m.get(k) for k in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total", L.PROMPT)}
        except Exception:
            return {}


    def _reuse(turn1_len, reply_len, turn2_len, cached, B):
        exact = turn1_len + reply_len
        return {"turn1_prompt": turn1_len, "reply": reply_len, "turn2_prompt": turn2_len, "cached_turn2": cached,
                "token_exact_max": exact, "unit64_floor": exact // 64 * 64, "block_floor": exact // B * B,
                "prompt_only_block_floor": turn1_len // B * B,
                "reuse_pct_of_turn2": round(100 * cached / turn2_len, 1) if isinstance(cached, int) and turn2_len else None,
                "reuse_pct_of_token_exact": round(100 * cached / exact, 1) if isinstance(cached, int) and exact else None}


    def prefix_tests(port, B):
        out = {}
        for band in ("8k", "40k"):
            if not due(240):
                out[band] = {"status": "SKIPPED_BOOT_BUDGET"}
                continue
            t1 = PROMPTS[band][V_PREFIX[band]]
            c0 = _prefix_counts(port)
            r1 = nb1_complete(port, t1, 512, temperature=TEMP, seed=31)
            if r1["error"] or not r1["ids"]:
                out[band] = {"status": "ERROR", "error": r1["error"]}
                continue
            t2 = t1 + r1["ids"] + SUFFIX_IDS
            r2 = nb1_complete(port, t2, 16, temperature=TEMP, seed=32)
            rs = nb1_complete(port, t1, 1, temperature=0.0)          # control: the same turn-1 prompt again
            out[band] = {"status": "MEASURED" if not r2["error"] else "ERROR", **_reuse(len(t1), len(r1["ids"]), len(t2), r2["cached"], B),
                         "cached_turn1": r1["cached"], "cached_resend_turn1": rs["cached"], "errors": [r2["error"], rs["error"]],
                         "counters_before": c0, "counters_after": _prefix_counts(port)}
        if due(240):
            t1 = PROMPTS["8k"][V_ABORT]
            g = L.Stream(port, t1, 2048, temperature=TEMP, seed=33)
            g.start()
            t0 = time.time()
            while time.time() - t0 < 300 and len(g.tokens) < 256 and g.is_alive():
                time.sleep(0.5)
            g.stop()
            g.join(10)
            part = list(g.tokens)
            time.sleep(2)
            r2 = nb1_complete(port, t1 + part + SUFFIX_IDS, 16, temperature=TEMP, seed=34)
            rs = nb1_complete(port, t1, 1, temperature=0.0)
            out["abort_8k"] = {"status": "MEASURED" if part and not r2["error"] else "INCONCLUSIVE",
                               **_reuse(len(t1), len(part), len(t1) + len(part) + len(SUFFIX_IDS), r2["cached"], B),
                               "aborted_after_tokens": len(part), "cached_resend_turn1": rs["cached"],
                               "errors": [g.err, r2["error"], rs["error"]]}
        else:
            out["abort_8k"] = {"status": "SKIPPED_BOOT_BUDGET"}
        return out


    # ---- PP2 trace window: every rank's trace, stage overlap, idle, verify rows per micro-batch
    SCOPE_ROWS = re.compile(r"^execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)$")


    def _merge(iv):
        out = []
        for a, b in sorted(iv):
            if out and a <= out[-1][1]:
                out[-1][1] = max(out[-1][1], b)
            else:
                out.append([a, b])
        return out


    def _clip(iv, lo, hi):
        return [[max(a, lo), min(b, hi)] for a, b in iv if b > lo and a < hi]


    def _inter(x, y):
        i = j = 0
        tot = 0.0
        while i < len(x) and j < len(y):
            lo, hi = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
            if hi > lo:
                tot += hi - lo
            if x[i][1] < y[j][1]:
                i += 1
            else:
                j += 1
        return tot


    def microbatch_costs(ev, scopes):
        """Per execute_context scope (one micro-batch on this rank): the GPU kernels it launched (eager launches and graph
        launches inside the scope on its thread, joined to kernels by correlation id), summed per category; then grouped by
        verify tokens per micro-batch. Answers 'MoE ms per micro-batch vs tokens per micro-batch' (both PP micro-batches
        re-read their stage's experts). The draft / vocab heads are cuBLAS GEMMs: inside gemm_other, not separable by name."""
        rt = {}
        for e in ev:
            if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("cuda_runtime", "cuda_driver"):
                c = (e.get("args") or {}).get("correlation")
                if c is not None:
                    rt.setdefault(e.get("tid"), []).append((e.get("ts", 0), c))
        kern = {}
        for e in ev:
            if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("kernel", "gpu_kernel"):
                c = (e.get("args") or {}).get("correlation")
                if c is not None:
                    kern.setdefault(c, []).append((L._cat(e.get("name", "")), float(e.get("dur", 0.0)), float(e.get("ts", 0.0))))
        names = {}
        for e in ev:
            if e.get("ph") == "X" and str(e.get("cat", "")).lower() in ("cuda_runtime", "cuda_driver"):
                c = (e.get("args") or {}).get("correlation")
                if c is not None:
                    names[c] = str(e.get("name", ""))
        per = []
        for e, nctx, nreq, ntok in scopes[:600]:
            lo, hi, tid = e.get("ts", 0), e.get("ts", 0) + e.get("dur", 0), e.get("tid")
            cats, gl, kl, ivl = {}, 0, 0, []
            for ts, c in rt.get(tid, []):
                if lo <= ts <= hi:
                    nm = names.get(c, "")
                    gl += "GraphLaunch" in nm
                    kl += bool(re.search(r"LaunchKernel|cuLaunchKernel", nm))
                    for k, d, kts in kern.get(c, []):
                        cats[k] = cats.get(k, 0.0) + d
                        ivl.append([kts, kts + d])
            mi = _merge(ivl)
            cap = next((x for x in sorted(CAPTURE_SIZES or []) if x >= ntok), None)
            per.append({"ctx_tokens": nctx, "gen_requests": nreq, "gen_tokens": ntok, "scope_ms": round(e.get("dur", 0) / 1000, 3),
                        "graph_launches": gl, "eager_kernel_launches": kl, "resolved_capture_size": cap,
                        "gpu_wall_ms": round((mi[-1][1] - mi[0][0]) / 1000, 3) if mi else 0.0,
                        "gpu_busy_union_ms": round(sum(b - a for a, b in mi) / 1000, 3),
                        "kernel_ms": {k: round(v / 1000, 3) for k, v in sorted(cats.items())}})
        by = {}
        for x in per:
            if x["ctx_tokens"] == 0:
                by.setdefault(x["gen_tokens"], []).append(x)
        agg = {}
        for ntok, xs in sorted(by.items()):
            keys = sorted({k for x in xs for k in x["kernel_ms"]})
            agg[ntok] = {"microbatches": len(xs), "scope_ms_mean": round(statistics.mean(x["scope_ms"] for x in xs), 3),
                         "kernel_ms_mean": {k: round(statistics.mean(x["kernel_ms"].get(k, 0.0) for x in xs), 3) for k in keys},
                         "kernel_ms_sum_mean": round(statistics.mean(sum(x["kernel_ms"].values()) for x in xs), 3),
                         "gpu_wall_ms_mean": round(statistics.mean(x["gpu_wall_ms"] for x in xs), 3),
                         "gpu_busy_union_ms_mean": round(statistics.mean(x["gpu_busy_union_ms"] for x in xs), 3),
                         "resolved_capture_size": xs[0]["resolved_capture_size"],
                         "graph_launches_mean": round(statistics.mean(x["graph_launches"] for x in xs), 2),
                         "eager_fallback_scopes": sum(1 for x in xs if x["graph_launches"] == 0)}
        return {"decode_only_by_verify_tokens": agg, "scopes": len(per), "examples": per[:6],
                "distinct_experts": "not reachable from the trace (router top-k ids need an engine hook: NB3 / NB4 item)",
                "note": "kernel time sums overlapping streams; nccl includes rank-skew waiting; nccl_sendrecv = the PP hand-off"}


    def pp2_trace(pdir, tp=2):
        files = sorted(Path(pdir).rglob("*.pt.trace.json*"))
        if not files:
            return {"error": "NO_TRACE_FILE"}
        per, ivs = {}, {}
        for k, f in enumerate(files[:8]):
            try:
                with (gzip.open if f.name.endswith(".gz") else open)(f, "rt", encoding="utf-8") as fh:
                    tr = json.load(fh)
            except Exception as e:
                per[f"file{k}"] = {"file": f.name, "error": repr(e)[:200]}
                continue
            di = tr.get("distributedInfo") or {}
            m = re.search(r"rank[-_]?(\d+)", f.name)
            rank = di.get("rank") if isinstance(di.get("rank"), int) else (int(m.group(1)) if m else k)
            base = (tr.get("baseTimeNanoseconds") or 0) / 1000.0
            ev = tr.get("traceEvents") or []
            iv = _merge([[e["ts"] + base, e["ts"] + base + e.get("dur", 0)] for e in ev if e.get("ph") == "X"
                         and str(e.get("cat", "")).lower() in ("kernel", "gpu_kernel") and "ts" in e])
            ivs[rank] = iv
            busy = sum(b - a for a, b in iv)
            span = (iv[-1][1] - iv[0][0]) if iv else 0.0
            gaps = [iv[i + 1][0] - iv[i][1] for i in range(len(iv) - 1)]
            rows, reqs, ctx = Counter(), Counter(), Counter()
            scopes = []
            for e in ev:
                if e.get("ph") == "X" and str(e.get("cat", "")) == "user_annotation":
                    mm = SCOPE_ROWS.match(str(e.get("name", "")))
                    if mm:
                        rows[int(mm.group(4))] += 1
                        reqs[int(mm.group(3))] += 1
                        ctx[int(mm.group(2))] += 1
                        scopes.append((e, int(mm.group(2)), int(mm.group(3)), int(mm.group(4))))
            microbatch = microbatch_costs(ev, scopes)
            one = TMP / f"nb1_trace_rank{rank}"
            shutil.rmtree(one, ignore_errors=True)
            one.mkdir(parents=True)
            shutil.copyfile(f, one / f.name)
            try:
                summ = L.trace_summary(one)
            except Exception as e:
                summ = {"error": repr(e)[:200]}
            shutil.rmtree(one, ignore_errors=True)
            per[rank] = {"file": f.name, "stage": rank // tp, "clock": "baseTimeNanoseconds" if base else "raw_ts",
                         "kernel_busy_ms": round(busy / 1000, 1), "span_ms": round(span / 1000, 1),
                         "idle_frac": round(1 - busy / span, 3) if span else None,
                         "gaps_over_1ms": sum(1 for g in gaps if g > 1000), "gap_ms_total": round(sum(g for g in gaps if g > 0) / 1000, 1),
                         "verify_rows_per_microbatch": dict(sorted(rows.items())), "gen_requests_per_microbatch": dict(sorted(reqs.items())),
                         "ctx_tokens_per_microbatch": dict(sorted(ctx.items())), "microbatch_costs": microbatch, "summary": summ}
        out = {"ranks": per, "files": len(files)}
        try:                                   # stage-imbalance test (RC2 hypothesis: heavy ~30 ms, light ~14 ms per tick)
            st = {}
            for r in (0, tp):
                agg = (((per.get(r) or {}).get("microbatch_costs") or {}).get("decode_only_by_verify_tokens") or {})
                if agg:
                    k = max(agg, key=lambda x: agg[x]["microbatches"])
                    st[f"stage{r // tp}"] = {"verify_tokens": k, "critical_path_ms_per_microbatch": agg[k]["gpu_wall_ms_mean"],
                                             "gpu_busy_union_ms": agg[k]["gpu_busy_union_ms_mean"],
                                             "summed_kernel_ms": agg[k]["kernel_ms_sum_mean"], "scope_ms": agg[k]["scope_ms_mean"]}
            if len(st) == 2:
                hv = max(st, key=lambda x: st[x]["critical_path_ms_per_microbatch"])
                lt = min(st, key=lambda x: st[x]["critical_path_ms_per_microbatch"])
                st["binding_stage"] = hv
                st["heavy_stage"] = hv
                st["tick_critical_path_ms"] = st[hv]["critical_path_ms_per_microbatch"]
                st["light_stage_idle_ms_per_tick_est"] = round(st[hv]["critical_path_ms_per_microbatch"] - st[lt]["gpu_busy_union_ms"], 2)
                st["note"] = ("critical path = the GPU wall span of a micro-batch on the binding stage (timeline); summed_kernel_ms "
                              "adds overlapping streams and is reported apart; PP0 and PP1 run concurrently, their times are never "
                              "added; partition 25/23 unchanged in NB1")
            out["stage_imbalance"] = st
        except Exception as e:
            out["stage_imbalance"] = {"error": repr(e)[:200]}
        a, b = ivs.get(0), ivs.get(tp)
        if a and b:
            lo, hi = max(a[0][0], b[0][0]), min(a[-1][1], b[-1][1])
            if hi > lo:
                ca, cb = _clip(a, lo, hi), _clip(b, lo, hi)
                ba, bb = sum(y - x for x, y in ca), sum(y - x for x, y in cb)
                both = _inter(ca, cb)
                frac = both / min(ba, bb) if min(ba, bb) else None
                out["stage_overlap"] = {"window_ms": round((hi - lo) / 1000, 1), "stage0_busy_frac": round(ba / (hi - lo), 3),
                                        "stage1_busy_frac": round(bb / (hi - lo), 3), "both_busy_ms": round(both / 1000, 1),
                                        "overlap_frac_of_less_busy": round(frac, 3) if frac is not None else None,
                                        "verdict": None if frac is None else ("OVERLAPPED" if frac >= 0.3 else "SERIAL" if frac < 0.1 else "PARTIAL"),
                                        "aligned": all((per.get(r) or {}).get("clock") == "baseTimeNanoseconds" for r in (0, tp))}
            else:
                out["stage_overlap"] = {"verdict": "NO_COMMON_WINDOW"}
        else:
            out["stage_overlap"] = {"verdict": "MISSING_STAGE_TRACE", "ranks": sorted(ivs)}
        return out


    CAPTURE_SIZES = None
    ROUTING_FILE = OUT / "nb1_routing_ids.npz"   # NB3 reads it (dataset foysalemonshanto/e25-nb1-routing-ids after a run)


    def merge_routing(stop, tp=2, max_steps=200):
        """The NB3 schema (addendum 9): nb1_routing_ids.npz with ids [total_tokens, 10] int and per-record layer, stage,
        mb_size (token rows of the micro-batch = requests x (1 + K)), step, offset, mb_index (0 / 1: the PP2 micro-batch of
        that step). One record = one MoE layer call for one micro-batch in one decode step. TP rank 0 of each stage only (the
        TP partner routes the same tokens). numpy is required (the workers write npz parts; the Kaggle image has numpy)."""
        parts = []
        for res in ((stop or {}).get("results") or []):
            if isinstance(res, dict) and res.get("path") and res.get("rank") is not None and res["rank"] % tp == 0:
                parts.append((res["rank"] // tp, res["path"]))
        if not parts:
            return {"written": False, "why": "no decode-sized router calls ran in Python in the window (FULL-graph replay) or "
                                            "the capture failed: " + json.dumps([(r or {}).get("error") for r in ((stop or {}).get("results") or [])
                                                                                 if isinstance(r, dict)])[:300]}
        recs = []
        try:
            import numpy as np
            for stage, path in sorted(parts):
                z = np.load(path)
                for i in range(len(z["layer"])):
                    ps = int(z["pass_index"][i])
                    if ps // 2 >= max_steps:
                        continue
                    o, n = int(z["offset"][i]), int(z["mb_size"][i])
                    recs.append((int(z["layer"][i]), stage, n, ps // 2, ps % 2, z["ids"][o:o + n]))
            if not recs:
                return {"written": False, "why": "part files empty"}
            offs, o_ = [], 0
            for r in recs:
                offs.append(o_)
                o_ += r[2]
            ids = np.concatenate([r[5] for r in recs], axis=0).astype(np.int32)
            np.savez_compressed(ROUTING_FILE, ids=ids, layer=np.array([r[0] for r in recs], dtype=np.int32),
                                stage=np.array([r[1] for r in recs], dtype=np.int32), mb_size=np.array([r[2] for r in recs], dtype=np.int32),
                                step=np.array([r[3] for r in recs], dtype=np.int32), offset=np.array(offs, dtype=np.int64),
                                mb_index=np.array([r[4] for r in recs], dtype=np.int32))
            return {"written": True, "file": str(ROUTING_FILE), "records": len(recs), "total_tokens": int(ids.shape[0]),
                    "top_k": int(ids.shape[1]) if ids.ndim == 2 else None, "steps": len({(r[1], r[3]) for r in recs}),
                    "ids_in_range": bool(ids.size == 0 or (ids.min() >= 0 and ids.max() < 512))}
        except ImportError:
            return {"written": False, "why": "numpy unavailable in the notebook kernel"}
        except Exception as e:
            return {"written": False, "error": f"{type(e).__name__}: {e}"[:300]}


    def pp2_profile(port, caps=None):
        global CAPTURE_SIZES
        CAPTURE_SIZES = (caps or {}).get("capture_sizes")
        shutil.rmtree(PROF_DIR, ignore_errors=True)
        PROF_DIR.mkdir(parents=True, exist_ok=True)
        rt0 = rpc(port, "nb1_routing", "start", "200")
        rec = L.profiled_load(port, PROMPTS["8k"], 8, PROF_DIR, secs=6.0, max_new=MAX_NEW["8k"], spec=True,
                              prefill_tok_s=PREFILL["tok_s"])
        stop = rpc(port, "nb1_routing", "stop", str(TMP / "nb1_routing_part"))
        rec["routing_ids"] = {"start": rt0, "stop": stop, "merged": merge_routing(stop),
                              "note": "router top-k ids of the steps that ran in Python inside this window (prefill chunks, "
                                      "uncaptured sizes); FULL-graph decode replays are not visible to a Python hook: decode "
                                      "routing needs --enable-return-routed-experts in its own boot (NB3 / NB4 item)"}
        g = L.gpu_mem()
        if g:
            rec["gpu_mem_peak_mib"] = [x[1] for x in g]
        try:
            rec["pp2"] = pp2_trace(PROF_DIR)
        except Exception as e:
            rec["pp2"] = {"error": f"{type(e).__name__}: {e}"[:300]}
        kept = TRACES / "profile_b1"
        kept.mkdir(exist_ok=True)
        for f in sorted(PROF_DIR.rglob("*.pt.trace.json*")):
            if f.stat().st_size < 80e6:
                shutil.copy2(f, kept / f.name)
        shutil.rmtree(PROF_DIR, ignore_errors=True)
        return rec


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
    # ==================================================================== CELL 7 — E25 NB1: the boots, the scorecard, the plan (NEW code)
    def _cfg(sec, name, kv, port, kv_bytes, extra, env, cap_min, est_min, levers):
        return {"sec": sec, "name": name, "kv": kv, "port": port, "kv_bytes": kv_bytes, "extra": list(extra), "env": dict(env),
                "cap_min": cap_min, "est_min": est_min, "levers": levers}


    FP8_EXTRA = EXT_FLAG + ["--language-model-only", "--profiler-config", PROFILER_CFG]
    B2_EXTRA = FP8_EXTRA + ["--mamba-ssm-cache-dtype", "bfloat16", "--max-num-seqs", str(SEQS_HI), "--compilation-config", COMPILE_HI]
    B2_ENV = {**DIAG_ENV, "MERLIN_SHARE_DRAFT_WORKSPACE": "1", **COMMIT_GRAPH_ENV}
    BOOTS = {"B1": _cfg("B1", "nb1_b1_fp8", "fp8", 8001, KV_FP8, FP8_EXTRA, {**DIAG_ENV, **NCCL_DIAG}, 70, 92,
                        ["D9 vLLM #60490 (source patch)", "C4 --language-model-only", "--profiler-config (idle unless profiling)",
                         "diagnostics: dev mode + read-only worker extension + NCCL_DEBUG INIT,NET (this boot only)"]),
             "B2": _cfg("B2", "nb1_b2_fp8", "fp8", 8002, KV_FP8, B2_EXTRA, B2_ENV, 50, 68,
                        ["B1 (without NCCL debug)", "C3 --mamba-ssm-cache-dtype bfloat16", "MERLIN_SHARE_DRAFT_WORKSPACE=1",
                         f"--max-num-seqs {SEQS_HI} + captures to {CAPTURE_HI[-1]}",
                         "VLLM_RECOVERSSM_COMMIT_GRAPH=graph at boot, eager for every standard window (runtime A/B at the end)",
                         "combined configuration"]),
             "B3": _cfg("B3", "nb1_b3_fp8_pin", "fp8", 8003, KV_FP8 + KV_STEP, B2_EXTRA, {**DIAG_ENV, "MERLIN_SHARE_DRAFT_WORKSPACE": "1"}, 36, 54,
                        ["B2 without the commit-graph boot env", "C5 KV_BYTES + 1.0 / 0.75 / 0.5 GiB per card (descending ladder, "
                         "the largest predicted safe step first; the step is set at run time by pin_rule)"])}
    for _c in BOOTS.values():
        NB1_EXTRA[_c["name"]], NB1_ENV[_c["name"]], NB1_KV[_c["name"]] = _c["extra"], _c["env"], _c["kv_bytes"]
    NB1["boot_plan"] = {k: {"name": c["name"], "levers": c["levers"], "argv_added": c["extra"], "env_added": c["env"],
                            "kv_bytes": [RC2_FP8["kv_bytes"], c["kv_bytes"]], "patch_60490": True, "cap_min": c["cap_min"]}
                        for k, c in BOOTS.items()}
    NB1["not_measured"] = {"W6_chunk": "CHUNK (Merlin launch.py _int_knob 'CHUNK' -> --max-num-batched-tokens) is a startup "
                                       "flag; vLLM has no runtime override, so no separate window here (left for a later boot)",
                           "fp8_quality_isolation": "NB0 (BF16 / FP8 x MTP off / K3 teacher-forced, raw arrays): production use of "
                                                    "any NB1 FP8 number depends on NB0's gate",
                           "C3_capacity": R["capacity_plan"]["c3_note"]}
    NB1["attribution"] = ("C4 = the drop of 'Model loading took X GiB' per rank vs RC2 FP8 (weights only; the patch does not touch "
                          "loading); D9 = the change of free VRAM after ready per rank vs RC2 FP8 minus the C4 weight drop (the patch "
                          "touches its scratch in the profiling run, so the allocator holds it from ready on); the patch's own "
                          "forward test = B1's greedy oracle vs RC2's recorded FP8 rows. B2 is one combined configuration "
                          "(C3 + share + seqs 16) except the commit graph (runtime A/B inside B2); B3 = C5 alone vs B2.")


    def d9_reserve_mib(seqs=8, k=3, mnbt=1024, max_len=MAX_LEN, block=BLOCK, compress=4, logits_mb=256):
        """What the #60490 code touches in the profiling run: max(prefill logits budget, decode logits) + the 1 MiB top-k."""
        cols = math.ceil(max_len / block) * math.ceil(block / compress)
        return round((max(logits_mb << 20, min(seqs * (k + 1), mnbt) * cols * 4) + (1 << 20)) / 2 ** 20, 1)


    def pin_free_check():
        """B3: free MiB per rank after warm-up + the 90k admission (the lazy QSA workspace and graphs are allocated by then)."""
        g = L.gpu_mem()
        if not g and FAKE and os.environ.get("NB1_FAKE_PIN_FREE_MIB"):        # local harness only
            g = [[i, 23034 - int(os.environ["NB1_FAKE_PIN_FREE_MIB"]), 23034] for i in range(4)]
        if not g:
            return {"ok": bool(FAKE), "why": "no nvidia-smi reading" + (" (local harness)" if FAKE else ": fail closed")}
        free = [t_ - u for _, u, t_ in g]
        return {"ok": min(free) >= PIN_FREE_MIN_MIB, "free_mib_per_rank": free, "need_mib": PIN_FREE_MIN_MIB}


    def body_fp8(cfg, ref, store=None, refs_extra=(), self_check=False, prefix=False, profile=False, commit=False, pin_check=False,
                 alloc=(("8k", (1, 2, 4, 8)), ("40k", (1, 2, 4))), residency=RESIDENCY_BANDS):
        def body(ss):
            s = ss[0]
            p = s.port
            o = boot_facts(s, {"census": kv_census(s), "probes": L.chat_probes(p), "sampled": sampled_check(s)}, cfg["name"])
            o["scales"] = scales_dump()
            if commit:                                 # every standard window runs the RC2 commit path (eager)
                o["commit_mode_at_start"] = commit_mode(p, "eager")
            if short_gate_failed(o):
                o.update(early_stop="SHORT_CORRECTNESS_GATE_FAILED", census_end=kv_census(s),
                         quality_gate="FAIL_SAMPLED" if (o["sampled"] or {}).get("ok") is not True else "FAIL_PROBES")
                return o
            if not PREFILL["measured"]:
                o["prefill"] = prefill_probe(p)
            if FAKE and ref not in ROWS:               # local harness only: the fake server's text is not RC2's
                o["fake_reference"] = oracle(p, key=ref)
            o["oracle"] = oracle(p, key=store, refs=(ref,) + tuple(refs_extra))
            o["quality_gate"] = quality_gate(o, ref)
            if o["quality_gate"] != "PASS":
                o["load_tests_skipped"] = f"oracle gate vs {ref}: {o['quality_gate']}"
                o["status"] = "NO_GO_QUALITY" if o["quality_gate"].startswith("FAIL") else "INCONCLUSIVE_QUALITY"
                o["census_end"] = kv_census(s)
                return o
            o["warmup"] = warmup(p, min(8, (o["caps"].get("max_num_seqs") or 8)))
            if self_check:
                o["fp8_self_check"] = fp8_self_check(s)
            o["block_layout"] = block_layout(o.get("workers"))
            if pin_check:                              # the 90k prefill first: the stop rule reads free VRAM after it
                o["admission_90k@5"] = admission_test(p, s)
                o["pin_check"] = pin_free_check()
                if not o["pin_check"]["ok"]:
                    o.update(status="PIN_FREE_BELOW_0P5_GIB", census_end=kv_census(s))
                    return o
            if alloc:
                o["kv"] = kv_accounting(p, o["caps"], alloc)
            if not pin_check:
                o["admission_90k@5"] = admission_test(p, s)
            if residency:
                o["max_safe_running_by_context"] = msr = residency_probes(p, s, o, bands=residency)
                NB1.setdefault("max_safe_running_by_context", {})[cfg["sec"]] = {      # E25 NB1 C-6: the boot's own P / F
                    "status": msr.get("status"), "pool_blocks": msr.get("pool_blocks"), "F": msr.get("F"), "F_source": msr.get("F_source"),
                    "pool_check": (msr.get("capacity") or {}).get("pool_check"), "why": msr.get("why"),
                    "fit_check": (msr.get("capacity") or {}).get("fit_check"),                   # R2 audit N1-4 / N1-6
                    "reason_codes": (msr.get("capacity") or {}).get("reason_codes"),
                    "bands": {b: {"achieved": v.get("achieved"), "bound_at_measured_pool_F": v.get("bound_prompt_only"),
                                  "bound_capped_by_seqs": v.get("bound_capped")} for b, v in msr["bands"].items()}}
            if prefix:
                o["prefix"] = prefix_tests(p, o["caps"].get("block_size") or BLOCK)
            o["steady"] = nb1_ladder(p, s, o)
            if commit:
                o["commit_graph_ab"] = commit_ab(p, s)
            if profile:
                o["profiler_tools"] = profiler_tools()
            if profile and due(420):
                o["profile"] = pp2_profile(p, o["caps"])
            o["census_end"] = kv_census(s)
            o["workers_end"] = rpc_cache_info(p)
            o["jit_after_warmup"] = jit_kernels(s.log())[len(o["warmup"]["jit_before"]) + len(o["warmup"]["jit_compiled_in_warmup"]):]
            return o
        return body


    def run_boot(sec, body, need=()):
        c = BOOTS[sec]
        BOOT["tag"] = c["name"]
        mon = LoadMonitor(c["name"])
        mon.start()
        try:
            o = merlin_boot(c["name"], c["kv"], c["port"], BOOT_READY_SEC, watched(body, c["cap_min"] * 60, c["name"]), need=need)
        finally:
            mon.stop_ev.set()
        tail = (((o.get("servers") or {}).get(c["name"]) or {}).get("log_tail") or "")
        if (o.get("status") == "LAUNCH_FAILED" and "VLLM_RECOVERSSM_COMMIT_GRAPH" in NB1_ENV[c["name"]]
                and re.search(r"recoverssm|commit", tail, re.I) and left() > c["est_min"] * 60):
            NB1_ENV[c["name"]] = {k: v for k, v in NB1_ENV[c["name"]].items() if k != "VLLM_RECOVERSSM_COMMIT_GRAPH"}
            first = {k: o.get(k) for k in ("status", "tries", "boot_gate")}
            log("commit-graph boot failed:", c["name"], "-> one retry without VLLM_RECOVERSSM_COMMIT_GRAPH")
            o = merlin_boot(c["name"], c["kv"], c["port"], BOOT_READY_SEC, watched(body, c["cap_min"] * 60, c["name"]), need=need)
            o["commit_graph_boot_failed"] = first
        LIVE["out"] = o                            # R2: the full boot record, kept by step() whatever happens below
        try:                                       # R2: post-processing never discards the measurements (keep_after_error)
            o["load_monitor_last"] = mon.last
            why = str((o.get("boot_gate") or {}).get("why") or "")
            if o.get("status") == "GPU_NOT_EMPTY" and why.startswith(("HOST_RAM", "ENGINE_PROCS")):
                o["status"] = "HOST_RAM_GATE_FAILED" if why.startswith("HOST_RAM") else "ENGINE_PROCS_LEFT"
            o["planned"] = NB1["boot_plan"][sec]
            o["free_mib_per_card"] = free_per_card(o, c["name"])
            if o["free_mib_per_card"] is None and FAKE and os.environ.get("NB1_FAKE_FREE_MIB"):     # local harness only (no nvidia-smi)
                o["free_mib_per_card"] = [float(os.environ["NB1_FAKE_FREE_MIB"])] * 4
                o["free_mib_source"] = "NB1_FAKE_FREE_MIB (local harness only)"
            o["free_after_ready_mib"] = free_after_ready(o, c["name"])
        except Exception as e:
            keep_after_error(o, e)
        ensure_core(o)                             # R2 audit N1-3 / N1-7: postprocess_ok, core_measurements_complete, headroom
        return o


    def boot_ok(o):
        """Clean boot: DONE / TF_OOM + CLEAN + PASS. R2 audit N1-7: TF_OOM still lets the dependents run, but the config is
        UNSAFE_HEADROOM (headroom_flag): never a recommended pin or a capacity winner."""
        return (isinstance(o, dict) and o.get("status") in ("DONE", "TF_OOM") and o.get("correctness") == "CLEAN"
                and o.get("quality_gate") == "PASS")


    def headroom_flag(o):
        """R2 audit N1-7: a boot whose teacher-forced call died of OOM (TF_OOM, also when post-processing failed after it) has
        no safe memory headroom at that configuration."""
        o = o if isinstance(o, dict) else {}
        if "TF_OOM" in (o.get("status"), o.get("status_before_postprocess")) or o.get("headroom") == "UNSAFE_HEADROOM":
            return "UNSAFE_HEADROOM"
        return None


    def free_valid(fr):
        """R2 audit N1-3: a valid per-rank free memory reading = 4 finite numbers in 0..24,576 MiB (one L4 card)."""
        return (isinstance(fr, list) and len(fr) == 4 and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
                                                              and 0 <= x <= 24576 for x in fr))


    def core_check(o):
        """R2 audit N1-3: the core measurements a dependent boot builds on: the kv census (KV size line), the gauge pool, F
        (resolve_F: line / fit rule), free MiB per card (4 valid ranks) and the residency probes (MEASURED: pool MATCH + F)."""
        o = o if isinstance(o, dict) else {}
        kv = o.get("kv") if isinstance(o.get("kv"), dict) else {}
        P = kv.get("pool_blocks")
        cen = o.get("census") if isinstance(o.get("census"), dict) else {}
        msr = o.get("max_safe_running_by_context") if isinstance(o.get("max_safe_running_by_context"), dict) else {}
        try:
            F_ok = residency_capacity(o).get("F_resolved") is not None
        except Exception:
            F_ok = False
        parts = {"kv_census": bool(cen.get("kv_tokens")), "pool": isinstance(P, int) and not isinstance(P, bool) and P >= 1,
                 "F": F_ok, "free_mib_per_card": free_valid(o.get("free_mib_per_card")), "residency": msr.get("status") == "MEASURED"}
        return {"complete": all(parts.values()), "parts": parts, "missing": [k for k, v in parts.items() if not v]}


    def ensure_core(o):
        """R2 audit N1-3 / N1-7: every boot record carries postprocess_ok, core_measurements_complete (+ the parts) and the
        headroom flag; never raises, never removes a measurement."""
        if not isinstance(o, dict):
            return o
        o["postprocess_ok"] = "postprocess_error" not in o
        try:
            cc = core_check(o)
        except Exception as e:
            cc = {"complete": False, "parts": {}, "missing": [f"core_check error: {type(e).__name__}: {e}"[:200]]}
        o["core_measurements_complete"], o["core_measurements"] = cc["complete"], cc
        hf = headroom_flag(o)
        if hf:
            o["headroom"] = hf
        return o


    def core_complete(o):
        """(complete, missing) of a boot record; a record without the fields (an exception path in step) is checked here."""
        if not isinstance(o, dict):
            return False, ["no record"]
        if not isinstance(o.get("core_measurements_complete"), bool):
            ensure_core(o)
        return o["core_measurements_complete"], (o.get("core_measurements") or {}).get("missing")


    def boot1():
        ok, why = sha_gate(min(1500, left() - 900))
        R["setup"]["sha_gate"] = {"ok": ok, "why": why}
        if not ok:
            return {"status": "SKIPPED_SHA_GATE", "why": why}
        return run_boot("B1", body_fp8(BOOTS["B1"], "rc2_fp8", store="b1", refs_extra=("rc2_bf16",), self_check=True,
                                       prefix=True, profile=True), need=("rc2_fp8",))


    def _rec(sec):
        st = R["sections"].get(sec) or {}
        r = st.get(BOOTS[sec]["name"])
        return r if isinstance(r, dict) else {}


    def pin_rule(base="B2"):
        """The B3 pin ladder (addendum 5): from the smallest free MiB per rank measured while serving (after #60490, the lazy
        QSA workspace, graphs, warm-up and the 90k admission), the largest step of +1.0 / +0.75 / +0.5 GiB that is predicted
        to leave >= 0.5 GiB free; then each lower step in turn if the in-boot check fails (a boot OOM: merlin_boot already
        retries 0.25 GiB lower once, Merlin's rule).
        R2 audit N1-3: the reading must come from the boot the pin builds on: base B2 needs B2's own valid per-rank free
        memory (B1's is added when valid: the smaller per rank counts); B2's missing -> fail closed, never decided from B1
        alone. Base B1 (B2 not clean) uses B1's own reading."""
        srcs = ("B2", "B1") if base == "B2" else ("B1",)
        rd = {s_: _rec(s_).get("free_mib_per_card") for s_ in srcs}
        if FAKE and os.environ.get("NB1_FAKE_FREE_MIB") and not any(free_valid(v) for v in rd.values()):    # local harness only
            rd = {s_: [float(os.environ["NB1_FAKE_FREE_MIB"])] * 4 for s_ in srcs}
        if not free_valid(rd[base]):
            return {"ok": False, "base": base, "free_mib": rd,
                    "why": f"free VRAM per card of {base} (the boot the pin builds on) not measured / invalid ({rd[base]!r}): fail closed"
                           + (", never decided from B1 alone" if base == "B2" else "")}
        fr = [v for v in rd.values() if free_valid(v)]
        mins = [min(x[i] for x in fr) for i in range(4)]
        steps = [g for g in PIN_STEPS_GIB if min(mins) - g * 1024 >= PIN_FREE_MIN_MIB]
        return {"ok": bool(steps), "base": base, "free_sources": [s_ for s_ in srcs if free_valid(rd[s_])],
                "min_free_mib_per_card": mins, "steps_gib": steps, "need_mib": PIN_FREE_MIN_MIB,
                "kv_bytes": [KV_FP8 + int(g * 2 ** 30) for g in steps], "blocks_per_gib_est": round(1024 / 15.75, 1),
                "why": (f"start +{steps[0]} GiB (predicted {min(mins) - steps[0] * 1024:.0f} MiB free on the tightest rank)" if steps
                        else f"no step leaves >= {PIN_FREE_MIN_MIB} MiB: min free {min(mins):.0f} MiB")}


    def pin_ladder(base="B2"):
        """B3 boots, one per step of the pin rule, highest first; stop at the first clean boot whose in-boot check keeps
        >= 0.5 GiB free per rank. Every try keeps its own record (sections B3/nb1_b3_fp8_pin<GiB x 100>). `base` = the clean
        configuration the pin is added to: B2 (without its commit-graph boot env), or B1 (without its NCCL debug) when B2
        was not clean.
        R2 audit N1-7: a TF_OOM step is UNSAFE_HEADROOM: kept, never recommended; the ladder goes on to the next lower step.
        NB1["recommended_pin"] = the first clean, not-UNSAFE step with the in-boot check ok and correctness DIAGNOSTIC_PASS."""
        tries = []
        NB1["recommended_pin"] = None
        extra, env = (B2_EXTRA, {**DIAG_ENV, "MERLIN_SHARE_DRAFT_WORKSPACE": "1"}) if base == "B2" else (FP8_EXTRA, dict(DIAG_ENV))
        ref = "b1"
        for g in NB1["pin_rule"]["steps_gib"]:
            nm = f"nb1_b3_fp8_pin{int(round(g * 100))}"
            BOOTS["B3"] = dict(BOOTS["B3"], name=nm, kv_bytes=KV_FP8 + int(g * 2 ** 30), pin_gib=g, extra=list(extra), env=dict(env), base=base)
            NB1_EXTRA[nm], NB1_ENV[nm], NB1_KV[nm] = BOOTS["B3"]["extra"], BOOTS["B3"]["env"], BOOTS["B3"]["kv_bytes"]
            NB1["boot_plan"]["B3"] = dict(NB1["boot_plan"]["B3"], name=nm, kv_bytes=[RC2_FP8["kv_bytes"], BOOTS["B3"]["kv_bytes"]])
            r = step("B3", nm, BOOTS["B3"]["est_min"],
                     lambda: run_boot("B3", body_fp8(BOOTS["B3"], ref, pin_check=True, alloc=(("8k", (1, 4, 8)), ("40k", (1, 2, 4)))),
                                      need=(ref,)))
            ensure_core(r)
            unsafe = headroom_flag(r)
            tries.append({"gib": g, "name": nm, "status": r.get("status"), "kv_bytes": r.get("kv_bytes"), "pin_check": r.get("pin_check"),
                          "headroom": unsafe, "correctness": _corr(r), "core_measurements_complete": r.get("core_measurements_complete")})
            if boot_ok(r) and not unsafe:
                if (r.get("pin_check") or {}).get("ok") is True and _corr(r) == "DIAGNOSTIC_PASS":
                    NB1["recommended_pin"] = {"gib": g, "name": nm, "kv_bytes": r.get("kv_bytes"), "base": base,
                                              "label": "DIAGNOSTIC (NB1 capacity only; production use depends on NB0's FP8 gate)"}
                else:
                    NB1["recommended_pin_why_none"] = f"{nm}: clean boot but pin_check {r.get('pin_check')} / correctness {_corr(r)}"
            scorecard_save()
            if (boot_ok(r) and not unsafe) or str(r.get("status", "")).startswith("SKIPPED") or r.get("status") == "PARTIAL_HARD_STOP":
                break
        NB1["pin_tries"] = tries
        save()


    def skip(sec, why):
        R["sections"].setdefault(sec, {})[BOOTS[sec]["name"]] = {"status": "SKIPPED_DEPENDENCY", "why": why}
        save()


    # ---- lever scorecard (PLAN v3.2 §4.1), rewritten after every boot
    def _free(rows):
        return [r[2] - r[1] for r in rows] if rows else None


    RC2_REF = {"pool_blocks": RC2_FP8["pool_blocks"], "F": RC2_FP8["F_rule"], "seqs": RC2_FP8["seqs"], "kv_tokens": RC2_FP8["kv_tokens"],
               "max_concurrency": RC2_FP8["max_concurrency"], "free_after_ready": _free(RC2_FP8["gpu_mem_after_ready"]),
               "free_mib": [r[2] - max(r[1], e[1]) for r, e in zip(RC2_FP8["gpu_mem_after_ready"], RC2_FP8["gpu_mem_end"])],
               "rank_load_gib": {int(k): v for k, v in RC2_FP8["rank_load_gib"].items()},
               "windows": RC2_FP8["windows"], "label": "RC2 FP8 (rc2_results.json; F = 6 from the RC2 analysis, not cold-measured)"}


    def boot_metrics(sec):
        r = _rec(sec)
        if not r or str(r.get("status", "")).startswith("SKIPPED"):
            return None
        name = BOOTS[sec]["name"]
        srv = (r.get("servers") or {}).get(name) or {}
        kv = r.get("kv") or {}
        st = {k: v for k, v in (r.get("steady") or {}).items() if isinstance(v, dict) and "@" in k and not k.startswith("_")}
        win = {k: {"status": v.get("status"), "tok_s": v.get("tok_s"), "mal": v.get("mean_acceptance_length"),
                   "tick_ms": v.get("tick_ms"), "cycle_ms": v.get("cycle_ms"), "running": [v.get("running_min"), v.get("running_max")]}
               for k, v in st.items()}
        best = {}
        for k, v in win.items():
            b, n = k.split("@")
            if v["status"] == "STEADY":
                best[b] = max(best.get(b, 0), int(n))
        adm = r.get("admission_90k@5") or {}
        try:                                       # R2 audit N1-2 / N1-4: only a certified capacity (pool MATCH, F rule) gives F
            cap = residency_capacity(r)
        except Exception as e:
            cap = {"status": "INCONCLUSIVE_CAPACITY_UNKNOWN", "why": [f"{type(e).__name__}: {e}"[:200]]}
        return {"status": r.get("status"), "correctness": r.get("correctness"), "quality_gate": r.get("quality_gate"),
                "status_before_postprocess": r.get("status_before_postprocess"), "postprocess_error": r.get("postprocess_error"),
                "postprocess_ok": r.get("postprocess_ok"), "core_measurements_complete": r.get("core_measurements_complete"),
                "core_missing": (r.get("core_measurements") or {}).get("missing"), "headroom": headroom_flag(r),
                "capacity_status": cap.get("status"), "capacity_why": cap.get("why"), "pool_check": cap.get("pool_check"),
                "fit_check": cap.get("fit_check"), "F_fit": (kv.get("fit") or {}).get("F"), "F_source": cap.get("F_source"),
                "kv_tokens": srv.get("kv_tokens") or (r.get("pool") or {}).get("kv_tokens"),
                "max_concurrency": srv.get("max_concurrency") or (r.get("pool") or {}).get("max_concurrency"),
                "pool_blocks": kv.get("pool_blocks"), "F": cap.get("F"), "seqs": (r.get("caps") or {}).get("max_num_seqs"),
                "free_mib": r.get("free_mib_per_card"), "free_after_ready": r.get("free_after_ready_mib"), "windows": win,
                "highest_steady": best, "sec": r.get("sec"),
                "admission_90k@5": {"status": adm.get("status"), "all_resident": adm.get("all_resident"),
                                    "first_token_s": adm.get("first_token"), **{k: (adm.get("window") or {}).get(k) for k in ("tok_s", "tick_ms", "cycle_ms")}},
                "rank_load_gib": (r.get("pool") or {}).get("rank_load_gib"),
                "mamba_pad_pct": (r.get("pool") or {}).get("mamba_pad_pct"), "min_block_for_mamba": (r.get("pool") or {}).get("min_block_for_mamba"),
                "recurrent_tensor_dtypes": (r.get("workers") or {}).get("recurrent_tensor_dtypes"),
                "worker_num_gpu_blocks": (r.get("workers") or {}).get("num_gpu_blocks")}


    def resident_at(m, L_ref=70000):
        if not m or m.get("pool_blocks") is None or m.get("F") is None:
            return None
        return min(m.get("seqs") or 10 ** 6, int(m["pool_blocks"] // (math.ceil(L_ref / BLOCK) + m["F"])))


    def projection(old, new):
        w = ((new or {}).get("windows") or {}).get("8k@8") or {}
        if w.get("status") == "STEADY" and w.get("tok_s"):
            a, src = w.get("mal") or 3.154, "this boot's 8k@8 window"
            t = 8 * a / w["tok_s"]
        else:
            a, src = 3.154, "RC2 FP8 8k@8 (261.8 tok/s, mean acceptance length 3.154)"
            t = 8 * a / 261.8
        c0, c1 = resident_at(old), resident_at(new)
        inp = {"formula": "steady = C * a / t_cycle (PLAN §1); run = 0.9 * steady", "C_old_at_70k": c0, "C_new_at_70k": c1,
               "a_tokens_per_verify_cycle": a, "t_cycle_s": round(t, 4),
               "t_cycle_source": src + " (t_cycle = 8 * a / tok_s = 2 ticks under PP2)",
               "assumption": "t_cycle held at the 8-stream value; at more streams it rises (PLAN per-tick table), so this is an upper bound"}
        if c0 is None or c1 is None:
            return None, {**inp, "why": "resident streams not measurable (pool / F missing)"}
        return round(0.9 * (c1 - c0) * a / t, 1), inp


    CORR_FAIL_STATUSES = ("ERROR", "BENCH_ERROR", "POSTPROCESS_ERROR")


    def _corr(m, key="quality_gate"):
        """R2 audit N1-1: status-aware correctness label of a boot (record or boot_metrics). PASS is never a production
        label: the best NB1 label is DIAGNOSTIC_PASS (production use depends on NB0's FP8 gate).
          status ERROR / BENCH_ERROR / POSTPROCESS_ERROR / FAIL* / NO_GO* (or a FAIL* / NO_GO* gate / correctness) -> FAIL;
          status != DONE -> UNTESTED; a postprocess_error -> INCOMPLETE_DIAGNOSTIC;
          correctness != CLEAN or the gate != PASS -> UNTESTED; otherwise DIAGNOSTIC_PASS."""
        if not m:
            return "UNTESTED"
        st = str(m.get("status") or "")
        g = m.get(key) if key else None
        if (st in CORR_FAIL_STATUSES or st.startswith(("FAIL", "NO_GO"))
                or str(g or "").startswith(("FAIL", "NO_GO")) or str(m.get("correctness") or "").startswith(("FAIL", "NO_GO"))):
            return "FAIL"
        if st != "DONE":
            return "UNTESTED"
        if m.get("postprocess_error"):
            return "INCOMPLETE_DIAGNOSTIC"
        if m.get("correctness") != "CLEAN" or (key is not None and g != "PASS"):
            return "UNTESTED"
        return "DIAGNOSTIC_PASS"


    def _corr_and(m, status, ok=("EXACT",), fail=("FAIL_HIGH_MARGIN",)):
        """A lever row's label from its own check (status) within its boot's label (_corr): DIAGNOSTIC_PASS only when both pass."""
        if status in fail:
            return "FAIL"
        b = _corr(m)
        if b != "DIAGNOSTIC_PASS":
            return b
        return "DIAGNOSTIC_PASS" if status in ok else "UNTESTED"


    def capacity_winner():
        """R2 audit N1-7 / N1-1 / N1-2: the boot with the most resident streams at 70k among the boots whose correctness is
        DIAGNOSTIC_PASS, headroom not UNSAFE_HEADROOM, capacity CERTIFIED and core measurements complete (earlier boot on a tie).
        A diagnostic selection: production use depends on NB0's FP8 gate."""
        cands, excluded = [], {}
        for i, sec in enumerate(("B1", "B2", "B3")):
            m = boot_metrics(sec)
            if not m:
                excluded[sec] = ["not run"]
                continue
            why = []
            c = _corr(m)
            if c != "DIAGNOSTIC_PASS":
                why.append(f"correctness {c}")
            if m.get("headroom") == "UNSAFE_HEADROOM":
                why.append("UNSAFE_HEADROOM (TF_OOM)")
            if m.get("capacity_status") != "CERTIFIED":
                why.append(f"capacity {m.get('capacity_status')}: {m.get('capacity_why')}")
            if m.get("core_measurements_complete") is not True:
                why.append(f"core measurements incomplete: {m.get('core_missing')}")
            ra = resident_at(m)
            if ra is None:
                why.append("resident streams at 70k not computable")
            if why:
                excluded[sec] = why
            else:
                cands.append((ra, -i, sec))
        best = max(cands) if cands else None
        return {"winner": best[2] if best else None, "resident_streams_at_70k": best[0] if best else None,
                "candidates": {s_: r_ for r_, _, s_ in cands}, "excluded": excluded,
                "rule": "max resident streams at 70k over DIAGNOSTIC_PASS + not UNSAFE_HEADROOM + capacity CERTIFIED + core complete",
                "label": "DIAGNOSTIC (" + NB0_NOTE + ")"}


    def _vdiff(a, b):
        return [y - x for x, y in zip(a, b)] if a and b and len(a) == len(b) else None


    def _delta(old, new):
        if not old or not new:
            return None
        d = {"kv_tokens": [old.get("kv_tokens"), new.get("kv_tokens")], "pool_blocks": [old.get("pool_blocks"), new.get("pool_blocks")],
             "F": [old.get("F"), new.get("F")], "free_mib_per_card": [old.get("free_mib"), new.get("free_mib")],
             "free_after_ready_delta_mib": _vdiff(old.get("free_after_ready"), new.get("free_after_ready")),
             "resident_streams_at_70k": [resident_at(old), resident_at(new)], "highest_steady": [old.get("highest_steady"), new.get("highest_steady")],
             "admission_90k@5": [old.get("admission_90k@5"), new.get("admission_90k@5")]}
        d["steady_tok_s_matched"] = {k: [((old.get("windows") or {}).get(k) or {}).get("tok_s"), ((new.get("windows") or {}).get(k) or {}).get("tok_s")]
                                     for k in ("8k@3", "8k@8", "40k@5")}
        d["cycle_ms_matched"] = {k: [((old.get("windows") or {}).get(k) or {}).get("cycle_ms"), ((new.get("windows") or {}).get(k) or {}).get("cycle_ms")]
                                 for k in ("8k@8", "40k@5")}
        return d


    def _conf(m):
        if not m:
            return "none (not run)"
        n = sum(1 for v in (m.get("windows") or {}).values() if isinstance(v, dict) and v.get("status") == "STEADY")   # R2
        return f"LOW: {n} STEADY window(s) of {WIN:.0f} s, no repeat (window noise not measured; read deltas below ~5 % as noise)"


    def _cost(*ms):
        secs = [m.get("sec") for m in ms if m and m.get("sec")]
        return round(sum(secs) * 4 / 3600, 2) if secs else None


    NB0_NOTE = "production use of any FP8 number here depends on NB0's FP8 quality gate"


    def attribution_c4_d9(m1):
        """Per rank: C4 weight drop (GiB -> MiB) from 'Model loading took', D9 = free-after-ready change minus that drop."""
        if not m1:
            return None
        lo, l1 = RC2_REF["rank_load_gib"], {int(k): v for k, v in (m1.get("rank_load_gib") or {}).items()}
        fr = _vdiff(RC2_REF["free_after_ready"], m1.get("free_after_ready"))
        c4 = [round((lo[r] - l1[r]) * 1024) if r in lo and r in l1 else None for r in range(4)]
        d9 = [round(f - c) if f is not None and c is not None else None for f, c in zip(fr or [None] * 4, c4)]
        return {"c4_weight_drop_mib_per_rank": c4, "free_after_ready_change_mib_per_rank": fr, "d9_mib_per_rank": d9,
                "d9_expected_reserve_mib": d9_reserve_mib(), "method": NB1["attribution"]}


    def scorecard():
        m1, m2, m3 = (boot_metrics(s) for s in ("B1", "B2", "B3"))
        rows = []

        def row(lever, old, new, adoption, notes, scope, corr_m=None, cost=()):
            g, inp = projection(old, new) if new else (None, None)
            if headroom_flag(new) or (new or {}).get("headroom") == "UNSAFE_HEADROOM":     # R2 audit N1-7: never adopted
                adoption = "NOT_RECOMMENDED: UNSAFE_HEADROOM (TF_OOM in this boot; measurements kept)"
            rows.append({"lever": lever, "gain_scope": scope, "measured": _delta(old, new), "projected_run_tok_s_gain": g,
                         "projection_inputs": inp, "correctness": _corr(corr_m if corr_m is not None else new),
                         "confidence": _conf(new), "adoption": adoption, "cost_gpu_h": _cost(*cost) if cost else _cost(new),
                         "notes": notes + "; " + NB0_NOTE})
        att = attribution_c4_d9(m1)
        p = (R["sections"].get("A0") or {}).get("patch_60490") or {}
        fwd = ((_rec("B1").get("oracle") or {}).get("vs_rc2_fp8") or {}).get("status")
        rows.append({"lever": "D9 vLLM #60490 (QSA indexer workspace in the profiling run)", "gain_scope": "separable (per-rank attribution in B1)",
                     "measured": {"attribution": att, "patch": {k: p.get(k) for k in ("applied", "sha_after", "marker", "ok")},
                                  "forward_test_vs_rc2_fp8": fwd, "kv_tokens": [RC2_REF["kv_tokens"], (m1 or {}).get("kv_tokens")]},
                     "projected_run_tok_s_gain": 0.0, "projection_inputs": {"why": "KV_BYTES pin skips memory profiling: the pool "
                     "cannot change; the lever is first-long-prefill safety (RC2: +518 MiB per rank at the first requests)"},
                     "correctness": _corr_and(m1, fwd, ok=CLEAN),               # R2 audit N1-1: within B1's own label
                     "confidence": "one boot vs RC2's boot (cross-session noise included)", "adoption": "patch", "cost_gpu_h": _cost(m1),
                     "notes": NB1["attribution"] + "; " + NB0_NOTE})
        row("C4 text-only (--language-model-only)", RC2_REF, m1, "config",
            f"weight drop per rank {(att or {}).get('c4_weight_drop_mib_per_rank')} MiB; with the KV pin the pool cannot grow, the "
            "gain is free VRAM (vision tower not loaded on PP0) and needs C5 to become KV", "separable (per-rank attribution in B1)")
        w2 = (_rec("B2").get("workers") or {})
        row("C3 SSM state bf16 (--mamba-ssm-cache-dtype bfloat16)", m1, m2, "config",
            f"actual recurrent tensor dtypes B1 {(m1 or {}).get('recurrent_tensor_dtypes')} -> B2 {(m2 or {}).get('recurrent_tensor_dtypes')}; "
            f"mamba padding {(m1 or {}).get('mamba_pad_pct')} % -> {(m2 or {}).get('mamba_pad_pct')} %, minimum block "
            f"{(m1 or {}).get('min_block_for_mamba')} -> {(m2 or {}).get('min_block_for_mamba')}; worker padding {json.dumps(w2.get('mamba_padding'))[:300]}; "
            "at block 4096 the pool block count cannot change (attention page larger); the capacity gain needs a smaller BLOCK_SIZE boot",
            "combined configuration gain (B2 = C3 + MERLIN_SHARE_DRAFT_WORKSPACE + seqs 16)")
        row("MERLIN_SHARE_DRAFT_WORKSPACE=1", m1, m2, "config",
            "shares the target FlashInfer float workspace with the MTP drafter (vllm/merlin/draft_workspace.py); Flash-Next's QSA runs on "
            "FlashAttention, so the log line decides whether anything is shared: " + json.dumps((_rec("B2").get("pool") or {}).get("share_workspace_lines") or [])[:300],
            "combined configuration gain (B2)")
        c2 = _rec("B2").get("caps") or {}
        g1, g2 = (_rec("B1").get("pool") or {}).get("rank_graph_gib") or {}, (_rec("B2").get("pool") or {}).get("rank_graph_gib") or {}
        gdelta = {r: round((g2[r] - g1[r]) * 1024) for r in g2 if r in g1}
        row("seqs > 8 (max-num-seqs 16, captures to 64)", m1, m2, "config",
            f"graph memory B1 -> B2 per rank {g1} -> {g2} GiB (delta MiB {gdelta}, the cost of sizes 40 / 48 / 64 + the larger "
            "batch; it comes out of the free VRAM the pin step needs); under PP2 a micro-batch holds C/2 requests, so FULL graphs "
            "need (C/2)(1+K) tokens: 32 covers 16 streams, 64 covers one micro-batch of 16; "
            f"log max_num_seqs={c2.get('max_num_seqs')} captures={c2.get('capture_sizes')} cover seqs x (1+K): "
            f"{c2.get('captures_cover_seqs_x_1pk')}; highest STEADY per band {(m2 or {}).get('highest_steady')}; the registry's 1..8 "
            "limit is a profile-validation rule, not an engine gate", "combined configuration gain (B2)")
        ab = _rec("B2").get("commit_graph_ab") or {}
        rows.append({"lever": "RecoverSSM commit graph (VLLM_RECOVERSSM_COMMIT_GRAPH=graph, runtime A/B)", "gain_scope": "separable (runtime A/B in B2)",
                     "measured": {k: ab.get(k) for k in ("delta_tok_s", "delta_tick_ms", "greedy_graph_vs_eager", "stats_after")} if ab else None,
                     "projected_run_tok_s_gain": (round(0.9 * ab["delta_tok_s"], 1) if ab.get("delta_tok_s") is not None
                                                  and (ab.get("graph_replay_share") or 0) > 0 else None),
                     "projection_inputs": {"formula": "run = 0.9 x measured steady delta at 8k@8 (matched streams); only when the graph "
                                                      "arm replayed commits (replay share > 0)", "graph_replay_share": ab.get("graph_replay_share"),
                                           "counters": ab.get("counters")} if ab else None,
                     "correctness": _corr_and(m2, (ab.get("greedy_graph_vs_eager") or {}).get("status")),   # R2 audit N1-1
                     "confidence": _conf(m2), "adoption": "config", "cost_gpu_h": None,
                     "notes": "PP-deferred commits always run eager (docstring): replays may be 0 under PP2; the counters decide. "
                              + ("boot retried without it: " + json.dumps(_rec("B2").get("commit_graph_boot_failed"))[:200]
                                 if _rec("B2").get("commit_graph_boot_failed") else "") + "; " + NB0_NOTE})
        pr = NB1.get("pin_rule") or {}
        b3base = (NB1.get("b3_base") or {}).get("base", "B2")
        row(f"C5 KV pin +{BOOTS['B3'].get('pin_gib', 0.5)} GiB per card (descending ladder, on {b3base})", m2 if b3base == "B2" else m1, m3, "config",
            (f"tries {json.dumps(NB1.get('pin_tries'))[:400]}" if m3 or NB1.get("pin_tries") else f"not run: {pr.get('why') or (_rec('B3').get('why'))}"),
            "separable (B3 vs its base B2 or B1; B3 boots without the commit-graph env and without NCCL debug)")
        rows.append({"lever": "max_safe_running_by_context (residency, not a lever)", "gain_scope": "measurement",
                     "measured": NB1.get("max_safe_running_by_context"), "projected_run_tok_s_gain": None, "projection_inputs": None,
                     "correctness": "UNTESTED", "confidence": "one cold admission per context and boot", "adoption": "config",
                     "cost_gpu_h": None, "notes": "achieved = all streams resident after prefill, real running == n, 0 preemptions; "
                     "bound = floor(P / (ceil(L / 4096) + 6)) at the measured pool; " + NB0_NOTE})
        rows.append({"lever": "W6 chunk 2048 (--max-num-batched-tokens)", "gain_scope": "not measured", "measured": None,
                     "projected_run_tok_s_gain": None, "projection_inputs": None, "correctness": "UNTESTED",
                     "confidence": "none (not measured)", "adoption": "config", "cost_gpu_h": 0.0, "notes": NB1["not_measured"]["W6_chunk"]})
        sc = NB1.get("fp8_self_check") or {}
        rows.append({"lever": "FP8 KV quality (gate row: decided by NB0)", "gain_scope": "gate",
                     "measured": {"fp8_self_check_2k_8k": {b: {k: (v or {}).get(k) for k in ("tf_aa", "decode_cached_vs_fresh")}
                                                           for b, v in (sc.get("bands") or {}).items()},
                                  "oracle_vs_rc2_fp8": fwd,
                                  "oracle_vs_rc2_bf16": ((_rec("B1").get("oracle") or {}).get("vs_rc2_bf16") or {}).get("status"),
                                  "RC2_40k_tf_vs_bf16": RC2_FP8["tf_40k_vs_bf16"],
                                  "mtp_default_scale_lines": (_rec("B1").get("pool") or {}).get("default_scale_lines")},
                     "projected_run_tok_s_gain": None, "projection_inputs": None, "correctness": "UNTESTED",
                     "confidence": "NB1 measures only FP8 self-consistency; the BF16 / FP8 x MTP off / K3 isolation is NB0's",
                     "adoption": "config", "cost_gpu_h": None, "notes": NB0_NOTE})
        return rows


    def scorecard_save():
        try:
            rows = scorecard()
            SCORECARD.write_text(json.dumps({"what": "E25 NB1 lever scorecard (PLAN v3.2 §4.1)", "t_min": round((time.time() - T_START) / 60, 1),
                                             "boots": {s: (_rec(s) or {}).get("status") for s in BOOTS},
                                             "boot_labels": {s: {"correctness": _corr(boot_metrics(s)), "headroom": headroom_flag(_rec(s)),
                                                                 "core_measurements_complete": _rec(s).get("core_measurements_complete"),
                                                                 "postprocess_ok": _rec(s).get("postprocess_ok")} for s in BOOTS},
                                             "recommended_pin": NB1.get("recommended_pin"),          # R2 audit N1-7: never a TF_OOM step
                                             "capacity_winner": capacity_winner(),                   # R2 audit N1-1 / N1-2 / N1-7
                                             "rows": rows}, indent=1, default=str))
            return rows
        except Exception as e:
            R["errors"].append(f"scorecard: {type(e).__name__}: {e}"[:300])
            save()
            return None


    _gpu_util_stop = threading.Event()


    def _gpu_util():
        """utilization / memory / power every 5 s (per-stage idle cross-check for the PP2 trace)."""
        while not _gpu_util_stop.is_set():
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=20)
                if r.returncode == 0:
                    with open(OUT / "nb1_gpu_util.jsonl", "a") as fh:
                        fh.write(json.dumps({"t": round(time.time() - T_START, 1), "gpus": r.stdout.strip().splitlines()}) + "\n")
            except Exception:
                pass
            _gpu_util_stop.wait(5)


    if NVSMI_OK:
        threading.Thread(target=_gpu_util, daemon=True).start()

    def nb1_main():
        """The plan: install -> patch -> B1 -> B2 -> B3 ladder; every failure stops the dependent boots (fail closed)."""
        m = R["sections"]["A0"]["merge"]
        if not m.get("ok"):
            NB1["stop"] = "MERGE_FAILED: no boot started"
        else:
            step("A0", "merlin_install", 10, lambda: {"install": (lambda e: {"ok": True, "tree": e["kv"].get("MERLIN_TREE")} if not FAKE else {"fake": True})(merlin_install())})
            pt = step("A0", "patch_60490", 3, apply_60490)
            if pt.get("status") != "DONE" or not pt.get("ok"):
                for s_ in BOOTS:
                    skip(s_, f"#60490 patch / worker extension not verified: {pt.get('error') or pt.get('status')}")
            elif "rc2_fp8" not in ROWS and not FAKE:
                for s_ in BOOTS:
                    skip(s_, "RC2's recorded FP8 oracle rows do not match this notebook's prompts (no gate reference)")
            else:
                b1 = step("B1", BOOTS["B1"]["name"], BOOTS["B1"]["est_min"], boot1)
                c1, miss1 = core_complete(b1)                 # R2 audit N1-3 (an exception path in step has no fields yet)
                scorecard_save()
                if not boot_ok(b1):
                    for s_ in ("B2", "B3"):
                        skip(s_, f"B1 not clean (status {b1.get('status')}, quality {b1.get('quality_gate')}, vs RC2 FP8 "
                                 f"{((b1.get('oracle') or {}).get('vs_rc2_fp8') or {}).get('status')})")
                else:
                    if c1:                                    # R2 audit N1-3: B2 builds on B1's complete core measurements
                        b2 = step("B2", BOOTS["B2"]["name"], BOOTS["B2"]["est_min"],
                                  lambda: run_boot("B2", body_fp8(BOOTS["B2"], "b1", commit=True, residency=("8k",)), need=("b1",)))
                        core_complete(b2)
                    else:
                        skip("B2", f"B1 core measurements incomplete (missing {miss1}; postprocess_ok {b1.get('postprocess_ok')}): "
                                   "B2 runs only on a complete B1 core (kv census, pool, F, free MiB per card, residency)")
                        b2 = _rec("B2")
                    scorecard_save()
                    base = "B2" if boot_ok(b2) else "B1"     # the pin ladder needs a clean base; B1 passed, B2 may not have
                    NB1["b3_base"] = {"base": base, "why": "B2 clean" if base == "B2" else
                                      f"B2 not clean (status {b2.get('status')}, quality {b2.get('quality_gate')}): the pin ladder runs on B1's configuration"}
                    NB1["pin_rule"] = pin_rule(base)          # R2 audit N1-3: free memory of the boot the pin builds on
                    if NB1["pin_rule"]["ok"]:
                        pin_ladder(base)
                    else:
                        skip("B3", "pin rule: " + NB1["pin_rule"]["why"])
        return {s: (_rec(s) or {}).get("status") for s in BOOTS}


    if not RERUN and not os.environ.get("RC2_UNIT"):
        nb1_main()
        _gpu_util_stop.set()
        R["host_phase"] = HOST_PHASE
        R["prefill"] = PREFILL
        R["minutes"] = round((time.time() - T_START) / 60, 1)
        save()
        scorecard_save()
        log("NB1 done", json.dumps({f"{s}/{n}": r.get("status") for s, st in R["sections"].items() if isinstance(st, dict)
                                    for n, r in st.items() if isinstance(r, dict) and "status" in r}))
    save()


if not RERUN:
    # ==================================================================== CELL 8 — report + fallback submission
    if not RERUN and not os.environ.get("RC2_UNIT"):
        write_fallback_submission()
        def _brief(r):
            if not isinstance(r, dict):
                return r
            keep = ("status", "correctness", "quality_gate", "probes", "sampled", "prefill", "caps", "pool", "kv", "admission_90k@5",
                    "prefix", "steady", "free_mib_per_card", "free_after_ready_mib", "workers", "nccl", "config_diff", "tries", "kv_bytes",
                    "boot_gate", "why", "error", "status_note", "load_tests_skipped", "early_stop", "boot_watchdog", "applied",
                    "sha_after", "ok", "warmup", "jit_after_warmup", "commit_graph_ab", "fp8_self_check")
            return {k: r.get(k) for k in keep if k in r}
        R["summary"] = {f"{s}/{n}": _brief(r) for s, st in R["sections"].items() if isinstance(st, dict) for n, r in st.items()
                        if isinstance(r, dict) and "status" in r}
        save()
        print(json.dumps(R["summary"], default=str)[:6000])
        print(json.dumps([{k: x.get(k) for k in ("lever", "gain_scope", "projected_run_tok_s_gain", "correctness", "adoption")}
                          for x in (scorecard_save() or [])]))
