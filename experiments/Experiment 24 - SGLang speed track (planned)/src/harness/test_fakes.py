"""E24 WP2 tests for the fake SGLang 0.5.21 / vLLM 0.32 servers (stdlib unittest; /opt/homebrew/bin/python3.12).

Every server runs on a free port, time-compressed (FAKE_TIME_SCALE 0.02, short startups), with HOME and every cache dir
inside a temp dir. The checks are the 0.5.21 / 0.32 facts the E24 lab relies on (SCR/understand/sglang_source_0521.md,
SCR/review/engine_api.md F2/F4/F6/F7/F8, the EMS logs, the E11-04 /metrics dump, the E14A traces).

run:  /opt/homebrew/bin/python3.12 -m unittest -v test_fakes        (from src/harness)
"""
import ast
import glob
import gzip
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request as U

HERE = os.path.dirname(os.path.abspath(__file__))
SG = os.path.join(HERE, "fake_sglang_server.py")
VL = os.path.join(HERE, "fake_vllm_server.py")
PY = sys.executable
E14_MODS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Base Engine", "src_e14_modules")
sys.path.insert(0, HERE)
import fake_common as C  # noqa: E402

TMP_ROOT = os.environ.get("E24_TEST_TMP") or None          # temp dirs go here (default: the system temp dir)
if TMP_ROOT:
    os.makedirs(TMP_ROOT, exist_ok=True)


def mkdtemp(prefix):
    return tempfile.mkdtemp(prefix=prefix, dir=TMP_ROOT)


def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read()


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)

# real 0.5.21 log regexes (SCR/understand/sglang_source_0521.md §10, validated on the 7 EMS logs)
PFX = r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d{3})?)(?: DP\d+)?(?: TP(?P<tp>\d+))?\] "
DECODE_RE = re.compile(PFX + r"Decode batch(?: \[(?P<iter>\d+)\])?, #running-req: (?P<running>\d+), "
                       r"(?:#full token: (?P<full_tok>-?\d+), full token usage: (?P<full_use>-?[\d.]+), mamba num: (?P<mamba_num>-?\d+), mamba usage: (?P<mamba_use>-?[\d.]+), "
                       r"|#token: (?P<tok>-?\d+), token usage: (?P<tok_use>-?[\d.]+), )"
                       r"(?:accept len: (?P<acc_len>[\d.]+), accept rate: (?P<acc_rate>[\d.]+), )?(?:cap len: [\d.]+, )?(?:block accept len: [\d.]+, )?"
                       r".*?(?:cuda|cpu|npu|musa) graph: (?P<graph>True|False), gen throughput \(token/s\): (?P<gen_tps>[\d.]+), #queue-req: (?P<queue>\d+)", re.M)
PREFILL_RE = re.compile(PFX + r"Prefill batch(?: \[(?P<iter>\d+)\])?, #new-seq: (?P<new_seq>\d+), #new-token: (?P<new_tok>\d+), #cached-token: (?P<cached>\d+), "
                        r"(?:full token usage: (?P<full_use>-?[\d.]+), )?(?:swa token usage: [\d.]+, )?(?:mamba usage: (?P<mamba_use>-?[\d.]+), )?(?:token usage: (?P<tok_use>-?[\d.]+), )?"
                        r"#running-req: (?P<running>\d+), #queue-req: (?P<queue>\d+), #pending-token: (?P<pending>\d+), "
                        r"(?:#replay-token: (?P<replay>\d+), )?.*?(?:cuda|cpu|npu|musa) graph: (?P<graph>True|False), input throughput \(token/s\): (?P<in_tps>[\d.]+)", re.M)
RETRACT_RE = re.compile(r"(?:KV cache pool is full\. Retract requests\.|Testing retraction\.) #retracted_reqs: (?P<n>\d+), "
                        r"#new_tokens_gained: (?P<tok>-?\d+)(?:, #mamba_num_gained: (?P<mamba>-?\d+))?(?:, #new_token_ratio: (?P<r0>[\d.]+) -> (?P<r1>[\d.]+))?")
CAP_RE = re.compile(r"max_total_num_tokens=(\d+), chunked_prefill_size=(\d+), max_prefill_tokens=(\d+), max_running_requests=(\d+), "
                    r"context_len=(\d+), available_gpu_mem=([\d.]+) GB")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def ems_argv(port, streams=20, width=6, extra=(), drop=(), model="/m/qwen3-8-27b-int4-redhat/w4a16-g128", tp=4):
    """The DESIGN §2 BASE argv (EMS dflash_w6_s20) with _streams(n)."""
    n = streams + 1
    a = ["--model-path", model, "--tp-size", str(tp), "--kv-cache-dtype", "fp8_e4m3", "--context-length", "155648",
         "--chunked-prefill-size", "4096", "--mamba-ssm-dtype", "float16", "--enable-metrics", "--disable-flashinfer-autotune",
         "--random-seed", "20261005", "--host", "127.0.0.1", "--port", str(port), "--log-level", "info",
         "--watchdog-timeout", "1800", "--decode-log-interval", "40", "--speculative-algorithm", "DFLASH",
         "--speculative-draft-model-path", "/m/qwen3-8-27b-dflash2/bf16", "--speculative-draft-model-quantization", "unquant",
         "--disable-prefill-cuda-graph", "--mamba-radix-cache-strategy", "extra_buffer_lazy", "--mem-fraction-static", "0.83",
         "--incremental-streaming-output", "--stream-interval", "1", "--reasoning-parser", "qwen3", "--tool-call-parser",
         "qwen3_coder", "--default-chat-template-kwargs", '{"preserve_thinking": true}', "--schedule-policy", "lpm",
         "--enable-cache-report", "--speculative-num-draft-tokens", str(width), "--min-free-slots-delay", "1",
         "--max-running-requests", str(n), "--cuda-graph-bs-decode"] + [str(i) for i in range(1, n + 1)] + \
        ["--max-mamba-cache-size", str(4 * n)]
    spec = {"--speculative-algorithm", "--speculative-draft-model-path", "--speculative-draft-model-quantization",
            "--speculative-num-draft-tokens", "--min-free-slots-delay"}
    if "target_only" in drop:
        drop = tuple(drop) + tuple(spec)
    out, i = [], 0
    while i < len(a):
        if a[i] in drop:
            i += 1
            while i < len(a) and not a[i].startswith("--"):
                i += 1
            continue
        out.append(a[i])
        i += 1
    return out + list(extra)


class Fake:
    """One fake server process (own session), log file, HTTP helpers."""

    def __init__(self, engine, argv, env=None, tmp=None, name="srv"):
        self.engine, self.argv = engine, argv
        self.tmp = tmp or mkdtemp(prefix="e24fake_")
        self.port = int(argv[argv.index("--port") + 1])
        e = {k: v for k, v in os.environ.items() if not k.startswith(("FAKE_", "E24_", "SGLANG_", "NCCL_", "VLLM_"))}
        e.update(HOME=os.path.join(self.tmp, "home"), FAKE_TIME_SCALE="0.02", FAKE_SG_STARTUP_SEC="0.6",
                 FAKE_VLLM_STARTUP_SEC="0.6", SGLANG_CACHE_DIR=os.path.join(self.tmp, "cache", "sglang"),
                 SGLANG_JIT_CACHE_DIR=os.path.join(self.tmp, "cache", "jit"), TRITON_CACHE_DIR=os.path.join(self.tmp, "cache", "triton"),
                 TORCHINDUCTOR_CACHE_DIR=os.path.join(self.tmp, "cache", "inductor"), CUDA_CACHE_PATH=os.path.join(self.tmp, "cache", "nv"),
                 VLLM_CACHE_ROOT=os.path.join(self.tmp, "cache", "vllm"), SGLANG_LOGPROB_CHUNK_SIZE="128", PYTHONUNBUFFERED="1")
        e.update(env or {})
        self.env = {k: v for k, v in e.items() if v is not None}
        self.log_path = os.path.join(self.tmp, "%s.log" % name)
        self.proc = None

    def start(self):
        cmd = [PY, SG] + self.argv if self.engine == "sglang" else [PY, VL, "serve"] + self.argv
        self.fh = open(self.log_path, "w")
        self.t0 = time.time()
        self.proc = subprocess.Popen(cmd, env=self.env, stdout=self.fh, stderr=subprocess.STDOUT, start_new_session=True)
        return self

    def url(self, path):
        return "http://127.0.0.1:%d%s" % (self.port, path)

    def wait_ready(self, timeout=60):
        path = "/ready" if self.engine == "sglang" else "/health"
        while time.time() - self.t0 < timeout:
            if self.proc.poll() is not None:
                return False
            try:
                if U.urlopen(self.url(path), timeout=2).status == 200:
                    self.ready_sec = time.time() - self.t0
                    return True
            except Exception:
                pass
            time.sleep(0.05)
        return False

    def get(self, path, timeout=30):
        return U.urlopen(self.url(path), timeout=timeout)

    def post(self, path, body, timeout=120):
        return U.urlopen(U.Request(self.url(path), data=json.dumps(body).encode(),
                                   headers={"Content-Type": "application/json"}), timeout=timeout)

    def pj(self, path, body, timeout=120):
        return json.loads(self.post(path, body, timeout).read())

    def log(self):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return ""

    def prom(self):
        """[(name, labels, value)] label-aware (quoted values may contain commas and slashes)."""
        out = []
        for line in self.get("/metrics").read().decode().splitlines():
            if not line or line.startswith("#"):
                continue
            m = re.match(r'^([A-Za-z_:][A-Za-z0-9_:]*)(\{.*\})?\s+(\S+)$', line)
            labels = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', m.group(2) or ""))
            out.append((m.group(1), labels, float(m.group(3))))
        return out

    def stop(self):
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except OSError:
            pass
        try:
            self.proc.wait(10)
        except Exception:
            pass
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except OSError:
            pass
        try:
            self.fh.close()
        except Exception:
            pass


def pv(snap, name, tp="0", **lab):
    for n, l, v in snap:
        if n == name and (tp is None or l.get("tp_rank", "0") == tp) and all(l.get(k) == x for k, x in lab.items()):
            return v
    return None


def sse(resp):
    out = []
    for raw in resp:
        line = raw.strip()
        if line.startswith(b"data:"):
            pl = line[5:].strip()
            if pl == b"[DONE]":
                out.append("[DONE]")
                break
            out.append(json.loads(pl))
    return out


def ids_of(n, salt=0):
    return [200 + (i * 7919 + salt * 104729) % 240000 for i in range(n)]


def quiet(fn):
    """Thread target that swallows the client-side error of a request whose server is meant to die."""
    def run():
        try:
            fn()
        except Exception:
            pass
    return run


def gen_body(ids, n, **kw):
    b = {"input_ids": ids, "sampling_params": {"max_new_tokens": n, "temperature": 0, "ignore_eos": True},
         "return_logprob": True, "top_logprobs_num": 2, "logprob_start_len": -1, "return_text_in_logprobs": False}
    b.update(kw)
    return b


# ================================================================================================ pure functions
class TestToyLM(unittest.TestCase):
    def test_quantized_margins_and_ties(self):
        lm = C.ToyLM("int4", "k", 0.004, 0.004)
        ties = qerr = 0
        N = 20000
        for p in range(N):
            d = lm.dist_h(C.mix64(p * 31 + 7), "decode")
            m = d.vals[0] - d.vals[1]
            ties += m == 0
            qerr += abs(m / C.QUANT - round(m / C.QUANT)) > 1e-9
        self.assertEqual(qerr, 0, "margins are multiples of 1/8 nat")
        self.assertTrue(0.004 < ties / N < 0.02, "about 1 %% exact ties, got %.4f" % (ties / N))

    def test_greedy_tie_lowest_id_and_full_vocab_normalized(self):
        lm = C.ToyLM("int4", "k")
        for p in range(3000):
            d = lm.dist_h(C.mix64(p), "decode")
            if d.vals[0] == d.vals[1]:
                self.assertLess(d.ids[0], d.ids[1])
                break
        d = lm.dist_h(C.mix64(5), "prefill")
        import math
        tot = sum(math.exp(v - d.logz) for v in d.vals) + (C.LM_VOCAB - len(d.ids)) * math.exp(d.bg_lp)
        self.assertAlmostEqual(tot, 1.0, places=9)

    def test_engine_noise_moves_only_near_ties(self):
        a, b = C.ToyLM("int4", "sglang|float16"), C.ToyLM("int4", "vllm|float32")
        flips = []
        for p in range(20000):
            h = C.mix64(p * 13)
            da, db = a.dist_h(h, "decode"), b.dist_h(h, "decode")
            if da.argmax() != db.argmax():
                flips.append(da.vals[0] - da.vals[1] + db.vals[0] - db.vals[1])
        self.assertTrue(flips, "different engine keys flip some near ties")
        self.assertLess(max(flips), 0.5, "same weights: only NEAR_TIE flips")

    def test_cross_weights_flip_more_but_no_state_flip(self):
        a, b = C.ToyLM("fp8", "vllm"), C.ToyLM("int4", "vllm")
        ms = []
        for p in range(20000):
            h = C.mix64(p * 17)
            da, db = a.dist_h(h, "decode"), b.dist_h(h, "decode")
            if da.argmax() != db.argmax():
                ms.append(da.vals[0] - da.vals[1] + db.vals[0] - db.vals[1])
        self.assertTrue(any(m >= 0.5 for m in ms), "cross-weights pairs produce FLIPs")
        self.assertLess(max(ms), 4.0, "no STATE_FLIP from weights alone")

    def test_diverge_at(self):
        clean, bug = C.ToyLM("int4", "k"), C.ToyLM("int4", "k", diverge_at=5)
        h = C.mix64(99)
        self.assertEqual(clean.dist_h(h, "decode", 4).argmax(), bug.dist_h(h, "decode", 4).argmax())
        d = bug.dist_h(h, "decode", 5)
        self.assertNotEqual(clean.dist_h(h, "decode", 5).argmax(), d.argmax())
        self.assertGreaterEqual(d.vals[0] - d.vals[1], 4.0)
        self.assertEqual(clean.dist_h(h, "prefill", None).argmax(), bug.dist_h(h, "prefill", None).argmax(),
                         "teacher forcing (prefill path) stays clean")

    def test_accept_means(self):
        for w, mu in ((4, 3.3), (6, 4.2), (8, 5.0)):
            p = C.accept_p(w)
            got = sum(C.draw_accept(w, p, C.mix64(i)) for i in range(20000)) / 20000
            self.assertAlmostEqual(got, mu, delta=0.08)


class TestCLI(unittest.TestCase):
    def test_help_lists_every_option_string(self):
        r = subprocess.run([PY, SG, "--help"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 0)
        opts = set(re.findall(r"(?<![\w-])(--[a-z0-9][a-z0-9-]*)", r.stdout))
        self.assertGreaterEqual(len(opts), 525)
        for f in ("--cuda-graph-bs-decode", "--cuda-graph-bs-prefill", "--min-free-slots-delay", "--prefill-decode-interval",
                  "--max-mamba-cache-size", "--speculative-draft-model", "--tensor-parallel-size", "--no-lora-strict-loading",
                  "--enable-mixed-chunk", "--disable-decode-cuda-graph", "--disable-cuda-graph", "--config"):
            self.assertIn(f, opts)
        self.assertNotIn("--cuda-graph-bs", opts, "renamed in 0.5.21")
        r = subprocess.run([PY, SG, "--help=all"], capture_output=True, text=True, timeout=60)
        self.assertEqual(r.returncode, 2)
        self.assertIn("ignored explicit argument 'all'", r.stderr)

    def flag_check(self, argv, *extra):
        r = subprocess.run([PY, SG, "--e24-flag-check", json.dumps(argv)] + list(extra), capture_output=True, text=True, timeout=60)
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_flag_check(self):
        ok = self.flag_check(ems_argv(30000))
        self.assertEqual(set(ok), {"ran", "ok", "unknown", "ambiguous", "parse_error", "method"})
        self.assertTrue(ok["ran"] and ok["ok"], ok)
        bad = self.flag_check(["--model-path", "/m", "--cuda-graph-bs", "1", "--speculative-num-draft", "6", "--e24-not-an-option"])
        self.assertFalse(bad["ok"])
        self.assertEqual(bad["ambiguous"], ["--cuda-graph-bs"])
        self.assertEqual(sorted(bad["unknown"]), ["--e24-not-an-option", "--speculative-num-draft"], "unique prefixes refused")
        self.assertIn("ambiguous option: --cuda-graph-bs could match", bad["parse_error"])
        ch = self.flag_check(["--model-path", "/m", "--mamba-ssm-dtype", "fp16"])
        self.assertFalse(ch["ok"])
        self.assertIn("invalid choice", ch["parse_error"])
        req = self.flag_check(["--tp-size", "4"])
        self.assertIn("required", req["parse_error"])
        rs = self.flag_check(["--model-path", "/m", "--speculative-algorithm", "DFLASH"], "--resolve")
        self.assertFalse(rs["ok"])
        self.assertIn("speculative-draft-model-path", rs["resolve_error"])

    def test_vllm_flag_check_and_help(self):
        stock = ["serve", "/m", "--tensor-parallel-size", "4", "--kv-cache-dtype", "fp8_e4m3", "--max-model-len", "131072",
                 "--max-num-seqs", "16", "--max-num-batched-tokens", "4096", "--gpu-memory-utilization", "0.9",
                 "--disable-custom-all-reduce", "--seed", "20261005", "--host", "127.0.0.1", "--port", "8011",
                 "--served-model-name", "q27", "--attention-backend", "FLASHINFER", "--trust-remote-code",
                 "--language-model-only", "--no-enable-prefix-caching", "--mamba-ssm-cache-dtype", "float16"]
        r = subprocess.run([PY, VL, "--e24-flag-check", json.dumps(stock)], capture_output=True, text=True, timeout=60)
        res = json.loads(r.stdout.strip())
        self.assertTrue(res["ok"], res)
        r = subprocess.run([PY, VL, "--e24-flag-check", json.dumps(["--max-num-seq", "16", "--bogus"])], capture_output=True, text=True)
        res = json.loads(r.stdout.strip())
        self.assertEqual(sorted(res["unknown"]), ["--bogus", "--max-num-seq"])
        full = subprocess.run([PY, VL, "serve", "--help=all"], capture_output=True, text=True).stdout
        short = subprocess.run([PY, VL, "serve", "--help"], capture_output=True, text=True).stdout
        self.assertRegex(full, r"(?<![\w-])--no-enable-prefix-caching(?![\w-])")
        self.assertNotRegex(short, r"(?<![\w-])--max-num-batched-tokens(?![\w-])", "--help alone lists groups only")

    def test_unknown_flag_exits_2_like_real(self):
        port = free_port()
        f = Fake("sglang", ems_argv(port, extra=["--bogus-flag"])).start()
        try:
            f.proc.wait(30)
            self.assertEqual(f.proc.returncode, 2)
            self.assertIn("unrecognized arguments: --bogus-flag", f.log())
        finally:
            f.stop()


# ================================================================================================ SGLang server, W6 S20
class TestSglangW6(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = mkdtemp(prefix="e24w6_")
        site = os.path.join(cls.tmp, "site")
        os.makedirs(site)
        write_text(os.path.join(site, "sitecustomize.py"),
            "import os, sys, threading, time, multiprocessing\n"
            "d = os.environ.get('E24T_HOOK')\n"
            "def _late():\n"
            "    time.sleep(1.0)\n"
            "    m, t = sys.modules.get('setproctitle'), None\n"
            "    try:\n"
            "        t = m.getproctitle() if m is not None else None\n"
            "    except Exception:\n"
            "        pass\n"
            "    open(os.path.join(d, 'late_%d.json' % os.getpid()), 'w').write(repr({'title': t, "
            "'orig': list(getattr(sys, 'orig_argv', []))[:3], 'ppid': os.getppid()}))\n"
            "if d:\n"
            "    os.makedirs(d, exist_ok=True)\n"
            "    open(os.path.join(d, 'env_%d.json' % os.getpid()), 'w').write(repr({'argv': sys.argv[:2], 'ppid': os.getppid(), "
            "'tri': os.environ.get('TRITON_CACHE_DIR')}))\n"
            "    threading.Thread(target=_late, daemon=True).start()\n")
        cls.hook = os.path.join(cls.tmp, "hook")
        cls.port = free_port()
        cls.f = Fake("sglang", ems_argv(cls.port, extra=["--decode-log-interval", "10"]), tmp=cls.tmp,
                     env={"PYTHONPATH": site, "E24T_HOOK": cls.hook, "NCCL_DEBUG": "INFO", "NCCL_MIN_CTAS": "4",
                          "NCCL_MAX_CTAS": "4", "NCCL_PROTO": "Simple"}).start()
        assert cls.f.wait_ready(60), cls.f.log()[-3000:]

    @classmethod
    def tearDownClass(cls):
        cls.f.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_startup_log_formats(self):
        log = self.f.log()
        m = re.search(r"server_args=(\{.*\})$", log, re.M)
        self.assertIsNotNone(m, "server_args={...} dict line")
        self.assertNotIn("server_args=ServerArgs(", log)
        d = ast.literal_eval(m.group(1))
        self.assertEqual(len(d), 519)
        self.assertEqual(d["speculative_num_draft_tokens"], 6)
        self.assertEqual(d["cuda_graph_bs_decode"], list(range(1, 22)))
        self.assertEqual(d["default_chat_template_kwargs"], {"preserve_thinking": True})
        self.assertEqual(d["watchdog_timeout"], 1800.0)
        self.assertEqual(d["decode_log_interval"], 10, "the last occurrence wins, as argparse")
        self.assertIsNone(d["speculative_draft_model_quantization"], "unquant -> None")
        self.assertTrue(d["_speculative_draft_quantization_explicitly_set"])
        self.assertEqual(d["port"], self.port)
        cap = CAP_RE.search(log)
        self.assertEqual(int(cap.group(4)), 21)
        self.assertTrue(700000 < int(cap.group(1)) < 860000, "W6 S20 0.83 pool ~779k")
        self.assertIn("Mamba Cache is allocated. max_mamba_cache_size: 84, conv_state size:", log)
        self.assertIn("intermediate_ssm_state_cache size:", log)
        self.assertIn("Capture target verify CUDA graph begin. backend=full, num_tokens_per_req=6, bs=[1, 2, 3", log)
        self.assertIn("DFLASH block size mismatch: using speculative_num_draft_tokens=6 but draft config block_size=8.", log)
        for r in range(4):
            self.assertIn("TP%d] Init parallel begin." % r, log)
        self.assertLess(log.index("Uvicorn running on"), log.index("The server is fired up and ready to roll!"))
        self.assertIn('"POST /v1/chat/completions HTTP/1.1" 200 OK', log, "startup warmup request")
        self.assertIn("Engine startup timings (s): load_weight=", log)

    def test_nccl_info_lines(self):
        log = self.f.log()
        self.assertIn("NCCL INFO NCCL_MIN_CTAS set by environment to 4.", log)
        self.assertIn("NCCL INFO NCCL_PROTO set by environment to Simple.", log)
        self.assertIn("NCCL INFO NCCL_CUMEM_ENABLE set by environment to 0.", log, "set by the launcher (engine.py)")
        self.assertRegex(log, r"\[0\] NCCL INFO Channel 03/04 : 0 1 2 3")
        self.assertRegex(log, r"(?m)^[^\s:]+:\d+:\d+ \[3\] NCCL INFO ")

    def test_server_info(self):
        si = json.loads(self.f.get("/server_info").read())
        self.assertEqual(si["internal_states"][0]["effective_max_running_requests_per_dp"], 21)
        self.assertEqual(si["version"], "0.5.21")
        self.assertIn("--speculative-algorithm DFLASH", si["launch_command"])
        self.assertEqual(si["max_req_input_len"], 155648 - 6)
        self.assertEqual(si["prefill_decode_interval"], 0)
        self.assertFalse(si["enable_mixed_chunk"])
        mi = json.loads(self.f.get("/model_info").read())
        self.assertEqual(mi["reasoning_parser"], "qwen3")
        models = json.loads(self.f.get("/v1/models").read())
        self.assertEqual(models["data"][0]["id"], "/m/qwen3-8-27b-int4-redhat/w4a16-g128")

    def test_processes_spawned_and_sitecustomize_ran_in_each(self):
        files = glob.glob(os.path.join(self.hook, "env_*.json"))
        recs = [ast.literal_eval(read_text(p)) for p in files]
        main = [r for r in recs if r["ppid"] != self.f.proc.pid]
        kids = [r for r in recs if r["ppid"] == self.f.proc.pid and r["argv"] == ["-c", "--multiprocessing-fork"]]
        tracker = [r for r in recs if r["ppid"] == self.f.proc.pid and r["argv"] == ["-c"]]
        self.assertEqual(len(kids), 5, "4 schedulers + 1 detokenizer, spawn: sitecustomize runs in each (%s)" % recs)
        self.assertEqual(len(tracker), 1, "+ the multiprocessing resource tracker (the real EMS logs have 7 sitecustomize "
                                          "error lines: launcher + 4 schedulers + detokenizer + tracker)")
        self.assertGreaterEqual(len(main), 1)
        # what a role classifier reads later in each process: setproctitle titles (children) and sys.orig_argv (launcher)
        end, late = time.time() + 10, []
        while time.time() < end and len(late) < 6:
            time.sleep(0.2)
            late = [ast.literal_eval(read_text(p)) for p in glob.glob(os.path.join(self.hook, "late_*.json"))]
        titles = sorted(r["title"] for r in late if r["title"])
        self.assertEqual(titles, ["sglang::detokenizer"] + ["sglang::scheduler_TP%d" % i for i in range(4)], late)
        self.assertTrue(any(r["orig"][1:3] == ["-m", "sglang.launch_server"] and r["ppid"] != self.f.proc.pid for r in late),
                        "launcher orig_argv looks like `python -m sglang.launch_server` (%s)" % late)
        self.assertTrue(glob.glob(os.path.join(self.tmp, "cache", "triton", "*", "*.cubin")), "Triton cache written")
        self.assertTrue(os.listdir(os.path.join(self.tmp, "cache", "inductor")), "Inductor cache written on DFlash")
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "home", ".cache", "sglang")), "no default-dir writes")

    def test_generate_logprob_layouts_and_teacher_forcing(self):
        ids = ids_of(3000, 1)
        r = self.f.pj("/generate", gen_body(ids, 24))
        mi = r["meta_info"]
        self.assertEqual(len(r["output_ids"]), 24)
        self.assertEqual(mi["finish_reason"], {"type": "length", "length": 24})
        self.assertEqual(mi["input_token_logprobs"], [], "logprob_start_len -1: output only")
        e = mi["output_token_logprobs"][0]
        self.assertEqual(len(e), 3)
        self.assertIsNone(e[2], "text null unless return_text_in_logprobs")
        self.assertEqual([x[1] for x in mi["output_token_logprobs"]], r["output_ids"])
        top = mi["output_top_logprobs"][3]
        self.assertEqual(len(top), 2)
        self.assertGreaterEqual(top[0][0], top[1][0])
        self.assertIn("spec_accept_length", mi)
        self.assertAlmostEqual(mi["spec_accept_length"], 24 / mi["spec_verify_ct"])
        for k in ("id", "prompt_tokens", "weight_version", "num_retractions", "completion_tokens", "cached_tokens",
                  "e2e_latency", "queue_time", "decode_throughput"):
            self.assertIn(k, mi)
        cont = r["output_ids"][:16]
        tf = self.f.pj("/generate", {"input_ids": ids + cont, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                                      "return_logprob": True, "top_logprobs_num": 2, "logprob_start_len": len(ids) - 1,
                                      "return_text_in_logprobs": False})
        itl, itop = tf["meta_info"]["input_token_logprobs"], tf["meta_info"]["input_top_logprobs"]
        self.assertEqual(len(itl), len(cont) + 1)
        self.assertEqual(itl[0], [None, ids[-1], None], "entry 0 = [null, prompt[-1], null]")
        self.assertIsNone(itop[0])
        self.assertEqual([x[1] for x in itl[1:]], cont, "entry k+1 scores cont[k]")
        for k in range(len(cont)):
            self.assertLess(abs(itl[k + 1][0] - mi["output_token_logprobs"][k][0]), 0.05, "decode vs prefill: path noise only")
            self.assertEqual(len(itop[k + 1]), 2)
        self.assertGreater(tf["meta_info"]["cached_tokens"], 0, "radix hit of the first request (64-aligned)")
        self.assertEqual(tf["meta_info"]["cached_tokens"] % 64, 0)

    def test_token_ids_logprob_full_vocab(self):
        ids = ids_of(500, 2)
        r = self.f.pj("/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                                     "return_logprob": True, "logprob_start_len": -1, "token_ids_logprob": list(range(C.LM_VOCAB))}, 300)
        row = r["meta_info"]["output_token_ids_logprobs"][0]
        self.assertEqual(len(row), C.LM_VOCAB)
        import math
        self.assertAlmostEqual(sum(math.exp(x[0]) for x in row), 1.0, places=6)
        self.assertEqual(r["meta_info"]["input_token_ids_logprobs"], [None], "origin_input_ids[-1:] -> one null entry")
        self.assertEqual(r["meta_info"]["input_token_logprobs"], [[None, ids[-1], None]])

    def test_streaming_incremental_matches_nonstream(self):
        ids = ids_of(2000, 3)
        ref = self.f.pj("/generate", gen_body(ids, 40))["output_ids"]
        evs = sse(self.f.post("/generate", dict(gen_body(ids, 40), stream=True)))
        self.assertEqual(evs[-1], "[DONE]")
        got = [t for ev in evs[:-1] for t in ev["output_ids"]]
        self.assertEqual(got, ref, "greedy is deterministic; incremental chunks are disjoint deltas")
        self.assertGreater(len(evs) - 1, 5, "about one chunk per verify step")
        self.assertEqual(evs[-2]["meta_info"]["output_token_logprobs_length"], 40)
        self.assertEqual(sum(len(ev["meta_info"]["output_token_logprobs"]) for ev in evs[:-1]), 40)

    def test_completions_sse_token_ids_usage_cache_report(self):
        ids = ids_of(3000, 4)
        body = {"model": "default", "prompt": ids, "max_tokens": 12, "temperature": 0, "stream": True,
                "return_token_ids": True, "logprobs": 2}
        evs = sse(self.f.post("/v1/completions", body))
        self.assertEqual(evs[-1], "[DONE]")
        self.assertEqual(evs[0]["choices"][0]["prompt_token_ids"], ids)
        self.assertNotIn("prompt_token_ids", evs[1]["choices"][0])
        self.assertTrue(all(e["usage"] is None for e in evs[:-1]), "no usage without stream_options.include_usage")
        self.assertEqual(sum(len(e["choices"][0]["token_ids"]) for e in evs[:-1]), 12)
        self.assertEqual(evs[-2]["choices"][0]["finish_reason"], "length")
        tl = evs[0]["choices"][0]["logprobs"]["top_logprobs"][0]
        self.assertTrue(all(isinstance(k, str) and not k.startswith("token_id:") for k in tl), "keyed by token TEXT")
        evs = sse(self.f.post("/v1/completions", dict(body, stream_options={"include_usage": True})))
        u = evs[-2]
        self.assertEqual(u["choices"], [])
        self.assertEqual(u["usage"]["prompt_tokens_details"]["cached_tokens"], (3000 - 1) // 64 * 64, "64-aligned prompt-end hit")

    def test_chat_reasoning_and_tool_call(self):
        q = {"model": "x", "messages": [{"role": "user", "content": "Reply with exactly one number: 7 * 6 ="}], "max_tokens": 64}
        r = self.f.pj("/v1/chat/completions", dict(q, chat_template_kwargs={"enable_thinking": False}))
        self.assertEqual(r["choices"][0]["message"]["content"], "42")
        self.assertIsNone(r["choices"][0]["message"]["reasoning_content"])
        r = self.f.pj("/v1/chat/completions", q)
        self.assertTrue(r["choices"][0]["message"]["reasoning_content"])
        tools = [{"type": "function", "function": {"name": "run_python", "parameters": {"type": "object",
                  "properties": {"code": {"type": "string"}}, "required": ["code"]}}}]
        r = self.f.pj("/v1/chat/completions", dict(q, tools=tools))
        ch = r["choices"][0]
        self.assertEqual(ch["finish_reason"], "tool_calls")
        self.assertEqual(ch["message"]["tool_calls"][0]["function"]["name"], "run_python")
        self.assertEqual(json.loads(ch["message"]["tool_calls"][0]["function"]["arguments"]), {"code": "x"})

    def test_metrics_semantics(self):
        snap0 = self.f.prom()
        names = {n for n, _l, _v in snap0}
        self.assertNotIn("sglang:num_retracted_requests_total", names, "absent until the first retraction")
        self.assertIn("sglang:num_retracted_reqs", names)
        self.assertEqual(sorted(l["tp_rank"] for n, l, _v in snap0 if n == "sglang:cuda_graph_passes_total"
                                and l.get("mode") == "decode_cuda_graph"), ["0", "1", "2", "3"])
        self.assertEqual({l["tp_rank"] for n, l, _v in snap0 if n == "sglang:realtime_tokens_total"}, {"0"})
        self.assertTrue(any(l.get("model_name", "").startswith("/m/") for _n, l, _v in snap0))
        ids = ids_of(2500, 5)
        evs = []
        t = threading.Thread(target=lambda: evs.extend(sse(self.f.post("/generate", {
            "input_ids": ids, "stream": True, "sampling_params": {"max_new_tokens": 3000, "temperature": 1.0, "top_p": 0.95,
                                                                  "top_k": 20, "ignore_eos": True}}))))
        t.start()
        time.sleep(0.4)
        s1 = self.f.prom()
        time.sleep(0.4)
        s2 = self.f.prom()
        t.join(60)
        time.sleep(0.2)
        s3 = self.f.prom()
        d12 = pv(s2, "sglang:realtime_tokens_total", mode="decode") - pv(s1, "sglang:realtime_tokens_total", mode="decode")
        self.assertGreater(d12, 0, "realtime decode counter moves every step")
        self.assertEqual(pv(s2, "sglang:generation_tokens_total", tp=None, is_streaming="true"),
                         pv(s1, "sglang:generation_tokens_total", tp=None, is_streaming="true"), "finish-only counter")
        self.assertEqual((pv(s3, "sglang:generation_tokens_total", tp=None, is_streaming="true") or 0)
                         - (pv(s1, "sglang:generation_tokens_total", tp=None, is_streaming="true") or 0), 3000)
        steps = [pv(s3, "sglang:cuda_graph_passes_total", tp=r, mode="decode_cuda_graph") for r in "0123"]
        self.assertEqual(len(set(steps)), 1, "identical per-rank step counters")
        self.assertEqual(pv(s3, "sglang:num_running_reqs"), 0.0, "idle republish")

    def test_profile_traces_spans_and_nccl_kernels(self):
        out = os.path.join(self.tmp, "prof")
        ids = ids_of(1500, 6)
        th = threading.Thread(target=lambda: sse(self.f.post("/generate", {
            "input_ids": ids, "stream": True, "sampling_params": {"max_new_tokens": 600, "temperature": 0, "ignore_eos": True}})))
        th.start()
        time.sleep(0.4)
        body = self.f.post("/start_profile", {"output_dir": out, "num_steps": 8, "activities": ["CPU", "GPU"],
                                              "with_stack": False, "record_shapes": False}).read()
        self.assertEqual(body, b"Start profiling.\n")
        for _ in range(100):
            if len(glob.glob(os.path.join(out, "*-TP-*.trace.json.gz"))) == 4:
                break
            time.sleep(0.1)
        th.join(60)
        files = sorted(glob.glob(os.path.join(out, "*.trace.json.gz")))
        self.assertEqual([re.sub(r"^.*-TP-", "TP-", f) for f in files], ["TP-%d.trace.json.gz" % r for r in range(4)])
        tr = json.load(gzip.open(files[0], "rt"))
        self.assertEqual(tr["distributedInfo"]["rank"], 0)
        ev = tr["traceEvents"]
        ver = [e for e in ev if e.get("cat") == "user_annotation" and e["name"].startswith("step[VERIFY")]
        self.assertGreaterEqual(len(ver), 8)
        self.assertTrue(any(e["name"].startswith("step[DRAFT") for e in ev))
        ar = [e for e in ev if e.get("cat") == "kernel" and "AllReduce" in e["name"]]
        # NCCL >= 2.19 names the specialised variant: _RING_LL even under NCCL_PROTO=Simple (the real E14A trace)
        self.assertTrue(ar and all(e["name"] == "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)"
                                   for e in ar), ar[:1])
        self.assertEqual(ar[0]["args"]["grid"], [4, 1, 1], "grid = channel count (NCCL_MIN/MAX_CTAS 4)")
        self.assertEqual(sum(1 for e in ar if ver[1]["ts"] <= e["ts"] < ver[2]["ts"]), 140, "140 all-reduces per step")
        try:
            sys.path.insert(0, E14_MODS)
            import e14_trace_summary as T
            s = T.summarize(out, 8)
            self.assertIsInstance(s, dict)
            js = json.dumps(s)
            self.assertIn("nccl_allreduce", js)
        except ImportError:
            pass

    def test_flush_cache_busy_then_idle(self):
        ids = ids_of(1000, 7)
        th = threading.Thread(target=lambda: self.f.pj("/generate", {"input_ids": ids, "sampling_params": {
            "max_new_tokens": 300, "temperature": 0, "ignore_eos": True}}))
        th.start()
        time.sleep(0.3)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.f.post("/flush_cache", {})
        self.assertEqual(cm.exception.code, 400)
        self.assertEqual(cm.exception.read(), b"Flush cache failed.\n")
        th.join(60)
        time.sleep(0.2)
        body = self.f.post("/flush_cache", {}).read()
        self.assertTrue(body.startswith(b"Cache flushed.\nPlease check backend logs"))
        log = self.f.log()
        self.assertIn("Cache not flushed because there are pending requests. #queue-req: 0, #running-req: 1", log)
        self.assertIn("TP3] Reset HybridReqToTokenPool", log)
        self.assertIn("TP0] Cache flushed successfully!", log)
        r = self.f.pj("/generate", gen_body(ids, 2))
        self.assertEqual(r["meta_info"]["cached_tokens"], 0, "flush cleared the radix cache")

    def test_abort_request_and_health(self):
        rid = "e24-abort-%d" % time.time_ns()
        evs = []
        th = threading.Thread(target=lambda: evs.extend(sse(self.f.post("/generate", {
            "rid": rid, "input_ids": ids_of(800, 8), "stream": True, "sampling_params": {"max_new_tokens": 5000, "temperature": 0,
                                                                                       "ignore_eos": True}}))))
        th.start()
        time.sleep(0.5)
        self.f.post("/abort_request", {"rid": rid})
        th.join(30)
        self.assertEqual(evs[-1], "[DONE]")
        self.assertEqual(evs[-2]["meta_info"]["finish_reason"]["type"], "abort")
        t = time.time()
        self.assertEqual(self.f.get("/health").status, 200)
        self.assertGreaterEqual(time.time() - t, 0.95, "/health sleeps 1 s and runs a 1-token generate")
        self.assertIn('"GET /health HTTP/1.1" 200 OK', self.f.log())

    def test_decode_log_lines_parse(self):
        th = threading.Thread(target=lambda: self.f.pj("/generate", {"input_ids": ids_of(900, 9), "sampling_params": {
            "max_new_tokens": 200, "temperature": 0, "ignore_eos": True}}))
        th.start()
        th.join(60)
        log = self.f.log()
        dec = [m.groupdict() for m in DECODE_RE.finditer(log)]
        pre = [m.groupdict() for m in PREFILL_RE.finditer(log)]
        self.assertTrue(dec and pre)
        self.assertTrue(all(d["tp"] == "0" and d["acc_len"] for d in dec))
        self.assertTrue(all(p["tp"] == "0" for p in pre))
        n_dec = sum(1 for l in log.splitlines() if "Decode batch" in l)
        self.assertEqual(n_dec, len(dec), "every Decode batch line matches the validated regex")


# ================================================================================================ SGLang scheduling sim
class TestSglangScheduling(unittest.TestCase):
    def test_target_only_tp1_cumulative_stream_and_prefix(self):
        port = free_port()
        f = Fake("sglang", ["--model-path", "/m/int4", "--tp-size", "1", "--enable-metrics", "--port", str(port),
                            "--random-seed", "1", "--chunked-prefill-size", "2048", "--mamba-ssm-dtype", "float16",
                            "--max-running-requests", "4", "--max-mamba-cache-size", "16", "--decode-log-interval", "5"]).start()
        try:
            self.assertTrue(f.wait_ready(60), f.log()[-2000:])
            ids = ids_of(1000, 11)
            evs = sse(f.post("/generate", {"input_ids": ids, "stream": True, "sampling_params": {
                "max_new_tokens": 10, "temperature": 0, "ignore_eos": True}}))
            lens = [len(e["output_ids"]) for e in evs[:-1]]
            self.assertEqual(lens, sorted(lens))
            self.assertEqual(lens[-1], 10)
            self.assertTrue(all(evs[i + 1]["output_ids"][:len(evs[i]["output_ids"])] == evs[i]["output_ids"]
                                for i in range(len(evs) - 2)), "cumulative: every chunk repeats all output_ids")
            log = f.log()
            self.assertRegex(log, r"(?m)^\[[\d: -]+\] Prefill batch, #new-seq: 1, #new-token: 1000,", "tp 1: no TP prefix")
            self.assertNotIn("accept len", "".join(l for l in log.splitlines() if "Decode batch" in l), "target-only")
            si = json.loads(f.get("/server_info").read())
            self.assertIsNone(si["speculative_algorithm"])
            self.assertIsNone(si["speculative_num_steps"])
            self.assertIsNone(si["min_free_slots_delay"])
            self.assertEqual(si["speculative_boundary_reduction"], "rs+rsv")
            self.assertIn("max_running_requests=4,", log)
        finally:
            f.stop()

    def test_mamba_cap_line(self):
        port = free_port()
        a = ems_argv(port, streams=20)
        a[a.index("--max-mamba-cache-size") + 1] = "68"
        f = Fake("sglang", a).start()
        try:
            self.assertTrue(f.wait_ready(60))
            log = f.log()
            self.assertIn("max_running_requests is capped to 17 by the mamba state cache (max_mamba_cache_size=68, 4 state "
                          "slots per request)", log)
            self.assertIn("max_running_requests=17,", log)
            si = json.loads(f.get("/server_info").read())
            self.assertEqual(si["max_running_requests"], 21, "the logged dict keeps the user's value")
            self.assertEqual(si["internal_states"][0]["effective_max_running_requests_per_dp"], 17)
        finally:
            f.stop()

    def test_chunked_prefill_preempts_decode_and_pdi(self):
        """A cold long prompt freezes residents (chunks back to back); --prefill-decode-interval 1 lets decode interleave."""
        res = {}
        for pdi in (0, 1):
            port = free_port()
            extra = ["--prefill-decode-interval", str(pdi)] if pdi else []
            f = Fake("sglang", ems_argv(port, streams=4, drop=("target_only",), extra=extra + ["--decode-log-interval", "4"]),
                     env={"FAKE_SG_PREFILL_TPS": "40000", "FAKE_SG_STEP_MS": "8"}).start()
            try:
                self.assertTrue(f.wait_ready(60))
                gaps, times = [], []
                stop = threading.Event()

                def resident():
                    resp = f.post("/generate", {"input_ids": ids_of(800, 21), "stream": True,
                                                "sampling_params": {"max_new_tokens": 100000, "temperature": 0, "ignore_eos": True}})
                    for raw in resp:
                        if stop.is_set():
                            break
                        if raw.startswith(b"data:"):
                            times.append(time.time())
                th = threading.Thread(target=resident, daemon=True)
                th.start()
                time.sleep(1.0)
                s0 = f.prom()
                t_inj = time.time()
                inj = f.pj("/generate", {"input_ids": ids_of(40960, 22), "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
                t_done = time.time()
                s1 = f.prom()
                time.sleep(0.4)                     # let the first post-prefill chunk arrive before stopping the reader
                stop.set()
                win = [t for t in times if t_inj <= t <= t_done]
                gaps = [b - a for a, b in zip(times, times[1:]) if t_inj <= b <= t_done + 0.1]
                res[pdi] = {"max_gap": max(gaps) if gaps else None, "in_window": len(win), "prefill_sec": t_done - t_inj,
                            "inj_cached": inj["meta_info"]["cached_tokens"],
                            "run_gauge": [pv(s0, "sglang:num_running_reqs"), pv(s1, "sglang:num_running_reqs")]}
                log = f.log()
                pl = [m for m in PREFILL_RE.finditer(log) if m.group("new_tok") == "4096"]
                self.assertTrue(len(pl) >= 9 and all(m.group("running") == "1" and m.group("queue") == "0" for m in pl),
                                "chunk lines: the chunked request is neither running nor queued")
            finally:
                f.stop()
        self.assertGreater(res[0]["max_gap"], 0.8 * res[0]["prefill_sec"], res)
        self.assertLess(res[1]["max_gap"], 0.5 * res[1]["prefill_sec"], res)
        self.assertGreater(res[1]["in_window"], res[0]["in_window"], res)

    def test_retraction_counter_appears_on_all_ranks(self):
        port = free_port()
        f = Fake("sglang", ems_argv(port, streams=8, extra=["--max-total-tokens", "30000", "--decode-log-interval", "5"])).start()
        try:
            self.assertTrue(f.wait_ready(60))
            self.assertNotIn("sglang:num_retracted_requests_total", {n for n, _l, _v in f.prom()})
            stop = threading.Event()

            def stream(i):
                try:
                    for raw in f.post("/generate", {"input_ids": ids_of(2500, 30 + i), "stream": True, "sampling_params": {
                            "max_new_tokens": 3000, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "ignore_eos": True}}):
                        if stop.is_set():
                            break
                except Exception:
                    pass
            th = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(8)]
            for t in th:
                t.start()
            deadline = time.time() + 40
            found = False
            while time.time() < deadline and not found:
                time.sleep(0.5)
                found = "Retract requests" in f.log()
            stop.set()
            f.post("/abort_request", {"abort_all": True})
            self.assertTrue(found, f.log()[-2000:])
            m = RETRACT_RE.search(f.log())
            self.assertIsNotNone(m)
            snap = f.prom()
            ranks = sorted(l["tp_rank"] for n, l, _v in snap if n == "sglang:num_retracted_requests_total")
            self.assertEqual(ranks, ["0", "1", "2", "3"])
        finally:
            f.stop()

    def test_census_kv_bound_with_available_evictable_gauges(self):
        port = free_port()
        f = Fake("sglang", ems_argv(port, streams=20, extra=["--max-total-tokens", "100000"])).start()
        try:
            self.assertTrue(f.wait_ready(60))
            stop = threading.Event()

            def stream(i):
                try:
                    for raw in f.post("/generate", {"input_ids": ids_of(20000, 50 + i), "stream": True, "sampling_params": {
                            "max_new_tokens": 50000, "temperature": 0, "ignore_eos": True}}):
                        if stop.is_set():
                            break
                except Exception:
                    pass
            th = [threading.Thread(target=stream, args=(i,), daemon=True) for i in range(8)]
            for t in th:
                t.start()
            time.sleep(4)
            snap = f.prom()
            stop.set()
            f.post("/abort_request", {"abort_all": True})
            run, q = pv(snap, "sglang:num_running_reqs"), pv(snap, "sglang:num_queue_reqs")
            self.assertTrue(1 <= run < 8 and q >= 1, (run, q))
            avail, ev = pv(snap, "sglang:kv_available_tokens"), pv(snap, "sglang:kv_evictable_tokens")
            self.assertLess(avail + ev, 20000 + 0.7 * 4096, "KV-bound by the lab's census rule")
            mav, mev, mus = (pv(snap, "sglang:mamba_available_tokens"), pv(snap, "sglang:mamba_evictable_tokens"),
                             pv(snap, "sglang:mamba_used_tokens"))
            self.assertEqual(mav + mev + mus, 84)
            self.assertGreaterEqual(mav + mev, 4, "not Mamba-bound")
        finally:
            f.stop()


# ================================================================================================ numerics knobs
class TestNumericsKnobs(unittest.TestCase):
    def rows(self, env, extra=(), drop=(), n=48, prompts=3):
        port = free_port()
        f = Fake("sglang", ems_argv(port, streams=4, drop=drop, extra=list(extra)), env=env).start()
        try:
            self.assertTrue(f.wait_ready(60), f.log()[-2000:])
            out = []
            for i in range(prompts):
                r = f.pj("/generate", gen_body(ids_of(1500, 80 + i), n))
                tops = r["meta_info"]["output_top_logprobs"]
                out.append((r["output_ids"], [t[0][0] - t[1][0] for t in tops], r["meta_info"]))
            return out
        finally:
            f.stop()

    @staticmethod
    def first_diff(a, b):
        return next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)

    def test_aa_exact_diverge_and_dflash_bug(self):
        base = self.rows({})
        again = self.rows({})
        self.assertEqual([r[0] for r in base], [r[0] for r in again], "same arm twice: EXACT")
        bug = self.rows({"FAKE_SG_DIVERGE_AT": "10"})
        for (a, ma, _), (b, mb, _) in zip(base, bug):
            d = self.first_diff(a, b)
            self.assertEqual(d, 10)
            self.assertGreaterEqual(mb[d], 4.0, "state flip margin in the candidate")
        db = self.rows({"FAKE_SG_DFLASH_BUG": "1", "FAKE_SG_DFLASH_BUG_AT": "12"})
        self.assertTrue(any(self.first_diff(a, b) is not None for (a, _, _), (b, _, _) in zip(base, db)))
        fr = self.rows({"FAKE_SG_DFLASH_BUG": "1", "FAKE_SG_DFLASH_BUG_AT": "12", "SGLANG_SIMULATE_ACC_LEN": "1",
                        "SGLANG_SIMULATE_ACC_TOKEN_MODE": "real-draft-token"})
        self.assertEqual([r[0] for r in fr], [r[0] for r in base], "forced reject: no accepted drafts, bug not exercised")
        self.assertTrue(all(abs(m["spec_accept_length"] - 1.0) <= 1.0 / m["spec_verify_ct"] + 1e-9 for _, _, m in fr),
                        "accept len 1 per verify step under forced reject (+ the prefill token)")
        self.assertTrue(all(m["spec_num_correct_drafts"] == 0 for _, _, m in fr))

    def test_target_only_vs_dflash_near_ties_only(self):
        to = self.rows({}, drop=("target_only",), n=200, prompts=4)
        d6 = self.rows({}, n=200, prompts=4)
        for (a, ma, _), (b, mb, _) in zip(to, d6):
            d = self.first_diff(a, b)
            if d is not None:
                self.assertLess(ma[d] + mb[d], 0.5, "verify vs decode numerics: NEAR_TIE only")


# ================================================================================================ faults
class TestFaults(unittest.TestCase):
    def launch(self, fault=None, env=None, argv=None):
        port = free_port()
        e = dict(env or {})
        if fault:
            e["FAKE_SG_FAULT"] = fault
        return Fake("sglang", argv or ems_argv(port, streams=4), env=e).start()

    def test_crash_start(self):
        f = self.launch("crash_start")
        try:
            self.assertFalse(f.wait_ready(30))
            f.proc.wait(30)
            log = f.log()
            self.assertIn("TP0] Scheduler hit an exception: Traceback (most recent call last):", log)
            self.assertIn("SIGQUIT received. signum=None, frame=None. It usually means one child failed.", log)
            self.assertNotEqual(f.proc.returncode, 0)
        finally:
            f.stop()

    def test_slow_start(self):
        f = self.launch("slow_start", env={"FAKE_SG_SLOW_START_SEC": "3.0"})
        try:
            self.assertTrue(f.wait_ready(60))
            self.assertGreaterEqual(f.ready_sec, 2.8)
        finally:
            f.stop()

    def test_oom_log_keeps_running(self):
        f = self.launch("oom_log")
        try:
            self.assertTrue(f.wait_ready(60))
            f.pj("/generate", gen_body(ids_of(500, 1), 4))
            self.assertIn("CUDACachingAllocator.cpp:3933] memory allocation failed with OOM on device 0", f.log())
            self.assertIsNone(f.proc.poll())
            n1 = f.log().count("CUDACachingAllocator.cpp:3933]")
            f.post("/flush_cache", {}).read()                       # EMS: the warning came again under the next load
            f.pj("/generate", gen_body(ids_of(500, 2), 4))
            self.assertEqual(f.log().count("CUDACachingAllocator.cpp:3933]"), n1 + 1)
            self.assertIsNone(f.proc.poll())
        finally:
            f.stop()

    def test_restore_divergence_only_on_a_cached_prefix(self):
        f = self.launch(env={"FAKE_SG_RESTORE_DIVERGE_AT": "8"})
        try:
            self.assertTrue(f.wait_ready(60))
            body = gen_body(ids_of(700, 3), 32)
            cold = f.pj("/generate", body)
            warm = f.pj("/generate", body)                          # radix hit: resumes from a cached prefix
            self.assertGreater(warm["meta_info"]["cached_tokens"], 0)
            self.assertEqual(warm["output_ids"][:8], cold["output_ids"][:8])
            self.assertNotEqual(warm["output_ids"][8:], cold["output_ids"][8:])
            f.post("/flush_cache", {}).read()
            again = f.pj("/generate", body)                         # cold again: identical to the first run
            self.assertEqual(again["meta_info"]["cached_tokens"], 0)
            self.assertEqual(again["output_ids"], cold["output_ids"])
        finally:
            f.stop()

    def test_grid_text_style(self):
        f = self.launch(env={"FAKE_SG_TEXT_STYLE": "grids"})
        try:
            self.assertTrue(f.wait_ready(60))
            txt = f.pj("/generate", gen_body(ids_of(400, 4), 64))["text"]
            self.assertTrue(txt and set(txt) <= set("0123456789 \n"), txt[:80])
        finally:
            f.stop()

    def test_mamba_assert_kills_server(self):
        f = self.launch("mamba_assert", env={"FAKE_SG_FAULT_AFTER": "2"})
        try:
            self.assertTrue(f.wait_ready(60))
            f.pj("/generate", gen_body(ids_of(300, 1), 2))
            with self.assertRaises(Exception):
                f.pj("/generate", gen_body(ids_of(300, 2), 2), timeout=30)
            f.proc.wait(30)
            self.assertIn("AssertionError: Can not alloc mamba cache", f.log())
        finally:
            f.stop()

    def test_hang_decode_metrics_alive_tokens_flat(self):
        f = self.launch("hang_decode", env={"FAKE_SG_FAULT_AFTER": "5"})
        try:
            self.assertTrue(f.wait_ready(60))
            th = threading.Thread(target=quiet(lambda: f.post("/generate", {"input_ids": ids_of(300, 3), "sampling_params": {
                "max_new_tokens": 5000, "temperature": 0, "ignore_eos": True}}, timeout=8).read()), daemon=True)
            th.start()
            time.sleep(1.5)
            a = pv(f.prom(), "sglang:realtime_tokens_total", mode="decode")
            time.sleep(1.0)
            b = pv(f.prom(), "sglang:realtime_tokens_total", mode="decode")
            self.assertEqual(a, b)
            self.assertIsNone(f.proc.poll())
        finally:
            f.stop()

    def test_logprob_oom_needs_chunk_env(self):
        body = {"input_ids": ids_of(300, 4), "return_logprob": True, "logprob_start_len": 0, "top_logprobs_num": 2,
                "sampling_params": {"max_new_tokens": 1, "temperature": 0}}
        f = self.launch("logprob_oom", env={"SGLANG_LOGPROB_CHUNK_SIZE": None})
        try:
            self.assertTrue(f.wait_ready(60))
            with self.assertRaises(Exception):
                f.pj("/generate", body, timeout=30)
            f.proc.wait(30)
            self.assertIn("torch.OutOfMemoryError: CUDA out of memory.", f.log())
        finally:
            f.stop()
        f = self.launch("logprob_oom", env={"SGLANG_LOGPROB_CHUNK_SIZE": "128"})
        try:
            self.assertTrue(f.wait_ready(60))
            r = f.pj("/generate", body)
            self.assertEqual(len(r["meta_info"]["input_token_logprobs"]), 300)
            self.assertIsNone(f.proc.poll())
        finally:
            f.stop()

    def test_nccl_env_ignored_and_subsys_filter(self):
        f = self.launch("nccl_env_ignored", env={"NCCL_DEBUG": "INFO", "NCCL_MIN_CTAS": "8", "NCCL_MAX_CTAS": "8"})
        try:
            self.assertTrue(f.wait_ready(60))
            log = f.log()
            self.assertNotIn("NCCL_MIN_CTAS set by environment", log)
            self.assertIn("Channel 01/02 : 0 1 2 3", log)
        finally:
            f.stop()
        f = self.launch(None, env={"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,GRAPH", "NCCL_MIN_CTAS": "2"})
        try:
            self.assertTrue(f.wait_ready(60))
            log = f.log()
            self.assertNotIn("set by environment", log, "ENV subsystem off -> no env lines")
            self.assertIn("Pattern 4, crossNic 0", log)
        finally:
            f.stop()

    def test_graph_list_shorter_than_running_crashes_dflash(self):
        port = free_port()
        a = ems_argv(port, streams=4)
        i = a.index("--cuda-graph-bs-decode")
        a = a[:i] + ["--cuda-graph-bs-decode", "1", "2"] + a[i + 6:]
        f = Fake("sglang", a).start()
        try:
            self.assertTrue(f.wait_ready(60), f.log()[-2000:])
            ths = [threading.Thread(target=quiet(lambda k=k: f.post("/generate", {"input_ids": ids_of(200, 60 + k), "sampling_params": {
                "max_new_tokens": 200, "temperature": 0, "ignore_eos": True}}, timeout=20).read()), daemon=True) for k in range(3)]
            for t in ths:
                t.start()
            f.proc.wait(30)
            self.assertIn("The size of tensor a (2) must match the size of tensor b (3)", f.log())
        finally:
            f.stop()

    def test_leak_default_cache_and_redirect(self):
        f = self.launch(None, env={"FAKE_SG_LEAK_DEFAULT_CACHE": "1"})
        try:
            self.assertTrue(f.wait_ready(60))
            self.assertTrue(glob.glob(os.path.join(f.tmp, "home", ".cache", "sglang", "triton", "*", "*.cubin")))
        finally:
            f.stop()
        f = self.launch(None, env={"SGLANG_CACHE_DIR": None, "TRITON_CACHE_DIR": None, "TORCHINDUCTOR_CACHE_DIR": None})
        try:
            self.assertTrue(f.wait_ready(60))
            self.assertTrue(glob.glob(os.path.join(f.tmp, "home", ".cache", "sglang", "triton", "*", "*.cubin")),
                            "import sglang setdefault: TRITON_CACHE_DIR -> ~/.cache/sglang/triton")
        finally:
            f.stop()

    def test_time_scale_cold_vs_warm_startup(self):
        tmp = mkdtemp(prefix="e24ts_")
        port = free_port()
        cold = Fake("sglang", ems_argv(port, streams=4), tmp=tmp, env={"FAKE_SG_STARTUP_SEC": None,
                                                                      "FAKE_TIME_SCALE": "0.008"}).start()
        try:
            self.assertTrue(cold.wait_ready(60))
            self.assertTrue(3.0 < cold.ready_sec < 8.0, cold.ready_sec)
        finally:
            cold.stop()
        warm = Fake("sglang", ems_argv(free_port(), streams=4), tmp=tmp, env={"FAKE_SG_STARTUP_SEC": None,
                                                                             "FAKE_TIME_SCALE": "0.008"}, name="warm").start()
        try:
            self.assertTrue(warm.wait_ready(60))
            self.assertLess(warm.ready_sec, 2.5, "warm Triton cache: about 62 s modeled")
        finally:
            warm.stop()
            shutil.rmtree(tmp, ignore_errors=True)


class TestRealHookOnFake(unittest.TestCase):
    """The real e24_hook.py as sitecustomize on the fake: every SGLang target is stubbed in the 4 scheduler ranks, so the
    notebook's rank-only self-test reaches GO in FAKE (exactly 4 counters files, TP ranks 0..3, none from the launcher /
    detokenizer) and the post-GO code paths run locally."""
    ALL = ("Scheduler", "ScheduleBatch", "SchedulerBatchResultProcessor", "MambaComponent", "DFlashWorkerV2",
           "_grouped_conv", "CandidateSelector")

    def test_hook_self_test_ok_on_exactly_the_tp_ranks(self):
        hook_src = os.path.join(os.path.dirname(HERE), "e24_hook.py")
        tmp = mkdtemp(prefix="e24rh_")
        site, hdir = os.path.join(tmp, "site"), os.path.join(tmp, "hook")
        os.makedirs(site)
        shutil.copy2(hook_src, os.path.join(site, "sitecustomize.py"))
        f = Fake("sglang", ems_argv(free_port(), streams=4), tmp=tmp,
                 env={"PYTHONPATH": site, "E24_HOOK_DIR": hdir, "E24_HOOK_COUNTERS": "1", "E24_HOOK_PROBE": "1",
                      "E24_HOOK_EAGER": "1", "E24_INJECT_DRAFT_NAN": "1", "E24_INJECT_SELECTOR_NAN": "1",
                      "E24_HOOK_TARGETS": ",".join(self.ALL), "E24_HOOK_FLUSH_SEC": "0.2", "E24_HOOK_CHAIN": "0"}).start()
        try:
            self.assertTrue(f.wait_ready(60), f.log()[-3000:])
            end, recs = time.time() + 15, []
            while time.time() < end:
                recs = [json.loads(read_text(p)) for p in glob.glob(os.path.join(hdir, "counters_*.json"))]
                if len(recs) >= 4 and all(r.get("tp_rank") is not None and r["self_test"]["ok"] for r in recs):
                    break
                time.sleep(0.3)
            self.assertEqual(sorted(r.get("tp_rank") for r in recs), [0, 1, 2, 3], recs)
            self.assertEqual({r.get("role_tag") for r in recs}, {"scheduler"})
            for r in recs:
                self.assertTrue(r["self_test"]["ok"], r["self_test"])
                self.assertLessEqual(set(self.ALL), set(r["patched"]))
            envs = [json.loads(read_text(p)) for p in glob.glob(os.path.join(hdir, "env_*.json"))]
            tags = sorted(e.get("role_tag") for e in envs)
            self.assertIn("main", tags)
            self.assertIn("detokenizer", tags)
            self.assertEqual(tags.count("scheduler"), 4, tags)
            self.assertEqual(len(glob.glob(os.path.join(hdir, "probe_*.json"))), 4, "one probe per rank (run_event_loop)")
        finally:
            f.stop()
            shutil.rmtree(tmp, ignore_errors=True)


# ================================================================================================ vLLM 0.32 fake
class TestVllm(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = mkdtemp(prefix="e24vl_")
        site = os.path.join(cls.tmp, "site")
        os.makedirs(site)
        write_text(os.path.join(site, "sitecustomize.py"),
            "import os\n"
            "d = os.environ.get('E24T_HOOK')\n"
            "def _rec(tag):\n"
            "    open(os.path.join(d, '%s_%d.txt' % (tag, os.getpid())), 'w').write(str(os.getppid()))\n"
            "if d:\n"
            "    os.makedirs(d, exist_ok=True)\n"
            "    _rec('import')\n"
            "    os.register_at_fork(after_in_child=lambda: _rec('fork'))\n")
        cls.hook = os.path.join(cls.tmp, "hook")
        cls.port = free_port()
        argv = ["/m/qwen3-8-27b-fp8-repacked-v1/hf-fp8", "--tensor-parallel-size", "4", "--kv-cache-dtype", "fp8_e4m3",
                "--max-model-len", "131072", "--max-num-seqs", "16", "--max-num-batched-tokens", "4096",
                "--gpu-memory-utilization", "0.9", "--disable-custom-all-reduce", "--seed", "20261005", "--host", "127.0.0.1",
                "--port", str(cls.port), "--served-model-name", "q27", "--attention-backend", "FLASHINFER",
                "--trust-remote-code", "--language-model-only", "--no-enable-prefix-caching"]
        cls.f = Fake("vllm", argv, tmp=cls.tmp, env={"PYTHONPATH": site, "E24T_HOOK": cls.hook}).start()
        assert cls.f.wait_ready(60), cls.f.log()[-3000:]

    @classmethod
    def tearDownClass(cls):
        cls.f.stop()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_startup_and_forked_children(self):
        log = self.f.log()
        self.assertRegex(log, r"GPU KV cache size: [\d,]+ tokens, Maximum concurrency for 131,072 tokens per request: [\d.]+x")
        self.assertIn("Selected MarlinFP8ScaledMMLinearKernel for Fp8LinearMethod", log)
        self.assertIn("Using KV cache scaling factor 1.0 for fp8_e4m3", log)
        self.assertRegex(log, r"\(EngineCore pid=\d+\) INFO \d\d-\d\d [\d:]+ \[core.py:133\] Initializing a V1 LLM engine \(v0.32.0\)")
        self.assertEqual(len(set(re.findall(r"\(Worker_TP(\d) pid=", log))), 4)
        forks = glob.glob(os.path.join(self.hook, "fork_*.txt"))
        imports = glob.glob(os.path.join(self.hook, "import_*.txt"))
        self.assertEqual(len(forks), 5, "EngineCore + 4 workers forked: register_at_fork ran in each")
        self.assertEqual(len(imports), 1, "fork: sitecustomize does not run again in the children")
        self.assertTrue(glob.glob(os.path.join(self.tmp, "cache", "vllm", "torch_compile_cache", "*", "rank_0_0", "backbone", "*")))

    def test_prompt_logprobs_layout(self):
        ids = ids_of(300, 1)
        r = self.f.pj("/v1/completions", {"model": "q27", "prompt": ids, "max_tokens": 1, "temperature": 0,
                                          "prompt_logprobs": 2, "return_tokens_as_token_ids": True})
        pl = r["choices"][0]["prompt_logprobs"]
        self.assertEqual(len(pl), 300)
        self.assertIsNone(pl[0])
        self.assertIn(str(ids[5]), pl[5])
        self.assertEqual(set(pl[5][str(ids[5])]), {"logprob", "rank", "decoded_token"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.f.post("/v1/completions", {"model": "q27", "prompt": ids, "max_tokens": 1, "stream": True, "prompt_logprobs": 1})
        self.assertEqual(cm.exception.code, 400)

    def test_stream_token_ids_and_id_keyed_logprobs(self):
        ids = ids_of(400, 2)
        evs = sse(self.f.post("/v1/completions", {"model": "q27", "prompt": ids, "max_tokens": 6, "temperature": 0.0,
                                                  "top_p": 1.0, "stream": True, "ignore_eos": True, "return_token_ids": True,
                                                  "logprobs": 2, "return_tokens_as_token_ids": True}))
        self.assertEqual(evs[-1], "[DONE]")
        ch = [e["choices"][0] for e in evs[:-1]]
        self.assertEqual(ch[0]["prompt_token_ids"], ids)
        self.assertIsNone(ch[1]["prompt_token_ids"])
        self.assertEqual(sum(len(c["token_ids"]) for c in ch), 6)
        self.assertTrue(all(k.startswith("token_id:") for k in ch[0]["logprobs"]["top_logprobs"][0]))
        self.assertEqual(ch[-1]["finish_reason"], "length")

    def test_metrics_names(self):
        txt = self.f.get("/metrics").read().decode()
        for n in ("vllm:num_requests_running{", "vllm:num_requests_waiting{", "vllm:generation_tokens_total{",
                  "vllm:generation_tokens_created{", "vllm:prompt_tokens_total{", "vllm:num_preemptions_total{",
                  "vllm:kv_cache_usage_perc{", "vllm:iteration_tokens_total_count{", 'engine="0",model_name="q27"'):
            self.assertIn(n, txt)

    def test_same_weights_cross_engine_agrees_except_near_ties(self):
        port = free_port()
        sg = Fake("sglang", ems_argv(port, streams=4, drop=("target_only",), model="/m/qwen3-8-27b-int4-redhat/w4a16-g128",
                                     extra=["--mamba-ssm-dtype", "float32"])).start()
        port2 = free_port()
        vb = Fake("vllm", ["/m/qwen3-8-27b-int4-redhat/w4a16-g128", "--tensor-parallel-size", "4", "--port", str(port2),
                           "--host", "127.0.0.1", "--served-model-name", "q27", "--no-enable-prefix-caching"]).start()
        try:
            self.assertTrue(sg.wait_ready(60) and vb.wait_ready(60))
            for i in range(3):
                ids = ids_of(1200, 90 + i)
                a = sg.pj("/generate", gen_body(ids, 120))
                b = vb.pj("/v1/completions", {"model": "q27", "prompt": ids, "max_tokens": 120, "temperature": 0,
                                              "ignore_eos": True, "logprobs": 2, "return_tokens_as_token_ids": True,
                                              "return_token_ids": True})
                ta, tb = a["output_ids"], b["choices"][0]["token_ids"]
                d = next((k for k in range(120) if ta[k] != tb[k]), None)
                if d is not None:
                    t = a["meta_info"]["output_top_logprobs"][d]
                    self.assertLess(t[0][0] - t[1][0], 0.5, "b <-> c32: NEAR_TIE only")
        finally:
            sg.stop()
            vb.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
