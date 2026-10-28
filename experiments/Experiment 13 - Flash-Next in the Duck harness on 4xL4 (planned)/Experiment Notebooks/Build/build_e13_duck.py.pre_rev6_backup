"""E13 (simple path, user 2026-10-20): the E12 Duck SMOKE1 notebook unchanged, only the engine swapped to Qwen3.8-Flash-Next.

Base: E12 SGLang SMOKE1 notebook (sha 0fea4d0c, PASSED on Kaggle 2026-10-19). Unchanged: CELL 0, CELL 2 (install/tokenizer;
the Flash-Next tokenizer.json is byte-identical to our RedHat INT4 one, sha 06b95093, and the EOS ids match), CELL 3 (Duck
package + adapter + sandbox), CELL 4 (tasks), CELL 6 (Duck solver), CELL 7 (report).
CELL 1: only the engine inputs and limits (model / draft path, context 139,264, running 4, SMOKE1 time budget).
CELL 5 (engine): new and Duck-only (= the E12-V cell structure):
  - e11_install (verbatim from E12);
  - + FA2 wheel (+ FA4 restore) + Franzen patch #3 (his SHAs);
  - + the user-mandated top-k test (recorded, never gating) + a host-RAM guard;
  - Franzen's launch (cell 12), verbatim except: D1 0.5.21, D4 TP4+EP4, D5 triton GDN, D6 arch 8.9, D7 no
    --gdn-mtp-cache-mode, and the memory items NEXTN off / chunk 2048 / mem 0.93 / 4 running (PLAN 3b);
  - a smoke ("!!!!" check);
  - Duck streams = min(running, KV // 60k) (pre-run capacity rule).
Run: python build_e13_duck.py [--eval30]  ->  ../Q27-XHIGH-E13_DUCK_FLASHNEXT_{SMOKE1|EVAL30}.ipynb
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E12DIR = Path(r"D:/ARC AGI 2/experiments/Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)/Experiment Notebooks")
EVAL30 = "--eval30" in sys.argv
BASE = E12DIR / ("Q27-XHIGH-E12_DUCK_REPL_SGLANG_EVAL30.ipynb" if EVAL30 else "Q27-XHIGH-E12_DUCK_REPL_SGLANG_SMOKE1.ipynb")
BASE_SHA = ("b65fb9dcc80b3652dc18b37d43355743311e2b55db8308ebc766d9d0b23de456" if EVAL30
            else "0fea4d0cb50fe85ec76a4ac27d6de32997aa580b0f5a2d7b36400c0be645c186")
OUT_NB = HERE.parent / f"Q27-XHIGH-E13_DUCK_FLASHNEXT_{'EVAL30' if EVAL30 else 'SMOKE1'}.ipynb"
TOPK_SCRIPT = (HERE / "e13_scripts" / "mp_topk.py").read_text(encoding="utf-8")


def src(c):
    return "".join(c["source"]) if isinstance(c["source"], list) else c["source"]


def once(text, old, new, label):
    assert text.count(old) == 1, f"{label}: {text.count(old)}"
    return text.replace(old, new)


def between(text, start, end, label):
    assert text.count(start) == 1 and (end is None or text.count(end) >= 1), label
    i = text.index(start); j = text.index(end, i) if end else len(text)
    return text[i:j]


raw = BASE.read_bytes(); assert hashlib.sha256(raw).hexdigest() == BASE_SHA, "E12 base notebook changed"
nb = json.loads(raw)
cells = [src(c) for c in nb["cells"]]
ENG12 = cells[6]

# ---------------------------------------------------------------- CELL 1: engine inputs and limits only
c1 = cells[2]
c1 = once(c1, 'SGL_MODEL_PATH = Path(os.environ.get("E11_INT4", "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1"))',
          'SGL_MODEL_PATH = Path(os.environ.get("E11_INT4", "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1"))   # E13',
          "model path")
c1 = once(c1, 'SGL_DRAFT_PATH = Path(os.environ.get("E11_DFLASH", "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-dflash2/pytorch/bf16/1"))',
          'SGL_DRAFT_PATH = Path(os.environ.get("E11_DFLASH", "/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1"))   # E13: precache only (NEXTN off)',
          "draft path")
c1 = once(c1, 'and (c.get("quantization_config") or {}).get("quant_method") == "compressed-tensors")',
          'and (c.get("quantization_config") or c.get("text_config", {}).get("quantization_config") or {}).get("quant_method") == "auto-round")   # E13',
          "model predicate")
c1 = once(c1, 'and c.get("architectures") == ["DFlash2DraftModel"])',
          'and "mtp_routed_experts" in ((c.get("quantization_config") or {}).get("config_groups") or {}))   # E13: Franzen\'s draft test',
          "draft predicate")
c1 = once(c1, "SGL_CONTEXT_LEN = 155648 ", "SGL_CONTEXT_LEN = 139264 ", "context")          # E13: Franzen (116+12+8)k
c1 = once(c1, "SGL_MAX_RUNNING = 12 ", "SGL_MAX_RUNNING = 4  ", "running")               # E13: PLAN 3b memory item
c1 = once(c1, "SGL_SERVER_MAX_RUNNING = 13 ", "SGL_SERVER_MAX_RUNNING = 4  ", "server running")
if not EVAL30:   # E13: Flash-Next load (about 25 min incl. the top-k test) + 15 min solving does not fit 50 min
    c1 = once(c1, '"SMOKE1":    {"source": "eval", "task_list": ["2ba387bc"], "time_budget_sec": 50 * 60,',
              '"SMOKE1":    {"source": "eval", "task_list": ["2ba387bc"], "time_budget_sec": 90 * 60,', "smoke budget")
c1 += r'''
# ---- E13 (Flash-Next on 4xL4, PLAN rev 4.1 simple path): engine constants. Everything else in this notebook = E12.
E13_FA2_WHEEL = "flash_attn-2.8.3.post1+cu13torch2.13cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
E13_FA2_SHA = "74984f628eaa9b22de271c787d82c4ef16daffcfb9684286a7a15ff6ad41ae84"
E13_FA2_DIR = _e11_discover(Path("/kaggle/input/datasets/hosen42/flash-attn-2-8-3-post1-cu13-torch2-13-cp312"), "SHA256SUMS_AND_SOURCE.txt",
                            lambda d, c: True) if not os.environ.get("E13_FA2_DIR") else Path(os.environ["E13_FA2_DIR"])
E13_PATCH3 = ("135e004f7613a59aa0765cbe3d779c9bb0c459f3a199b868358759e020aaeb5d",   # Franzen BEFORE_SHA (= 0.5.21 file)
              "47e65140e2480a8637b9ea5bb746ff77e9c09c01e448fd2056c0b485e4f3a10a")   # Franzen AFTER_SHA (= Pennyroyal wheel)
E13_MEM_FRACTION = "0.93"        # PLAN 3b: about 1.5 GiB outside the static pools (Franzen 0.96 on a 96 GB card)
E13_CHUNK = 2048                 # PLAN 3b: E3-proven on 4xL4 (Franzen 8192 on 96 GB)
E13_KV_PER_STREAM = 60000        # pre-run capacity rule: resident Duck contexts <= KV / 60k (E12 EVAL30 prompts p50 47.9k)
E13_TOPK_TEST = os.environ.get("E13_TOPK_TEST", "1") == "1"   # user: the fast_topk<512> correctness test always runs
CONFIG["e13"] = {k: globals()[k] for k in ("E13_FA2_SHA", "E13_MEM_FRACTION", "E13_CHUNK", "E13_KV_PER_STREAM", "E13_TOPK_TEST",
                                           "SGL_CONTEXT_LEN", "SGL_MAX_RUNNING")}
CONFIG["e13"]["fa2_dir"] = str(E13_FA2_DIR)
'''

# ---------------------------------------------------------------- CELL 5: Flash-Next engine (Duck-only)
REUSED = (between(ENG12, "class E11Fail(Exception):", "# ---- launch: the E11-04 W1 command", "install block") + "\n"
          + between(ENG12, "class E11Server:", "# ---- the shim:", "E11Server + kill") + "\n"
          + between(ENG12, "_E11_KV_RE = re.compile", "def e11_smoke", "decode summary"))
for must in ("def e11_install():", "class E11Server:", "def e11_kill_server(", "def e11_decode_summary(", "_E11_KV_RE"):
    assert must in REUSED, must

C5 = r'''# ==================================================================== CELL 5 — E13 engine: SGLang 0.5.21 + Qwen3.8-Flash-Next (Franzen's launch)
# Duck-only (no Poetiq shim). Reused verbatim from E12 (Kaggle-proven): E11Fail .. e11_install, E11Server, e11_kill_server,
# e11_decode_summary. New: FA2 + patch #3, the top-k test, a host-RAM guard, Franzen's launch with the L4 deltas, a smoke.
import queue as _queue, signal as _signal, statistics as _statistics, types as _types, urllib.request as _urlreq, base64 as _b64
E11_HARNESS = os.environ.get("E11_HARNESS") == "1"            # local CPU harness only: fake server, no install, no GPU
SGL_PORT = int(os.environ.get("E11_PORT", 30000))
SGL_URL = f"http://127.0.0.1:{SGL_PORT}"
E11_SERVER = None
ENGINE_READY = False; llm = None; engine = None; CONCURRENCY = 0
ENGINE_INFO = {"backend": None, "e13": {"stage": "not started"}, "kv_tokens": None, "concurrency": None, "fatal": None,
               "load_sec": None, "tokenizer_check": E11_TOKENIZER_CHECK}

''' + REUSED + r'''

def metrics_snapshot():                                  # SGLang /metrics (CELL 6's heartbeat matches name substrings)
    if not ENGINE_READY:
        return []
    try:
        txt = _urlreq.urlopen(SGL_URL + "/metrics", timeout=10).read().decode("utf-8", "replace")
        rows = []
        for line in txt.splitlines():
            m = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)", line) if line and not line.startswith("#") else None
            if m and not m.group(1).endswith(("_bucket", "_created")) and any(
                    k in m.group(1) for k in ("num_running_reqs", "num_queue_reqs", "gen_throughput", "token_usage", "cache_hit_rate")):
                try:
                    rows.append({"name": m.group(1), "labels": m.group(2), "value": float(m.group(3))})
                except ValueError:
                    pass
        return rows
    except Exception as e:
        return [{"error": repr(e)[:300]}]


def _e13_sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()


def e13_extras(vpy):
    """FA2 (QSA decode on sm89, upstream PR #36968) + FA4 restore + Franzen patch #3 with his SHAs."""
    if E11_HARNESS:
        return {"harness": True}
    w = E13_FA2_DIR / E13_FA2_WHEEL
    if not w.exists():
        w = E13_FA2_DIR / E13_FA2_WHEEL.replace("+", "")              # Kaggle drops '+' from file names
    if _e13_sha(w) != E13_FA2_SHA:
        raise E11Fail("FA2 wheel sha256 mismatch")
    fl = Path("/tmp/e13-fa2"); fl.mkdir(exist_ok=True)
    if not (fl / E13_FA2_WHEEL).exists():
        (fl / E13_FA2_WHEEL).symlink_to(w)
    pip = [sys.executable, "-m", "pip", "--python", str(vpy), "install", "--no-index", "--no-deps", "-q", "--disable-pip-version-check"]
    r = subprocess.run(pip + ["--find-links", "/tmp/e11-wheels", str(fl / E13_FA2_WHEEL)], capture_output=True, text=True)
    if r.returncode:
        raise E11Fail("FA2 install failed: " + (r.stderr or r.stdout)[-800:])
    fa4 = sorted(Path("/tmp/e11-wheels").glob("flash_attn_4-*.whl"))   # FA2 ships 21 old flash_attn/cute files: restore FA4's
    if fa4:
        r = subprocess.run(pip + ["--force-reinstall", str(fa4[0])], capture_output=True, text=True)
        if r.returncode:
            raise E11Fail("FA4 restore failed: " + (r.stderr or r.stdout)[-800:])
    site = Path(subprocess.run([str(vpy), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                               capture_output=True, text=True, check=True).stdout.strip())
    g = site / "sglang/srt/layers/quantization/gptq/schemes/gptq_moe.py"
    h = _e13_sha(g)
    if h == E13_PATCH3[0]:
        t = g.read_bytes(); assert t.count(b"dtype=torch.half") == 2
        tmp = g.with_name(g.name + ".e13tmp"); tmp.write_bytes(t.replace(b"dtype=torch.half", b"dtype=params_dtype")); os.replace(tmp, g)
        h = _e13_sha(g)
    if h != E13_PATCH3[1]:
        raise E11Fail(f"patch #3 after-sha mismatch {h[:12]}")
    probe = ("import json, torch, flash_attn\nfrom sglang.srt.layers.attention.qwen_sparse_attn_backend import "
             "_resolve_flash_attn_varlen_func as r\nprint(json.dumps({'fa2': flash_attn.__version__, 'resolves_to': r().__module__,"
             "'cxx11abi': torch.compiled_with_cxx11_abi()}))")
    r = subprocess.run([str(vpy), "-c", probe], capture_output=True, text=True, env=_E13_ENV, timeout=900)
    info = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"probe_error": (r.stderr or "")[-800:]}
    if not str(info.get("resolves_to", "")).startswith("flash_attn.") or "cute" in str(info.get("resolves_to")):
        raise E11Fail("QSA decode does not resolve to classic FA2: " + json.dumps(info)[:600])
    return info


E13_TOPK_SRC = _b64.b64decode("TOPK_B64").decode()

def e13_topk_test(vpy):
    """User-mandated: SGLang's JIT fast_topk<512> (the QSA block top-k) vs a tie-aware torch oracle (#36807). Recorded only."""
    if not E13_TOPK_TEST or E11_HARNESS:
        return {"skipped": True}
    p = OUT / "mp_topk.py"; p.write_text(E13_TOPK_SRC, encoding="utf-8")
    try:
        r = subprocess.run([str(vpy), "-I", str(p), "--out", str(OUT / "e13_topk.json"), "--fixtures", str(OUT / "topk_fixtures")],
                           capture_output=True, text=True, env=_E13_ENV, timeout=1500)
        (OUT / "e13_topk.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
        rep = json.loads((OUT / "e13_topk.json").read_text())
        return {k: (rep.get(k) or {}).get("summary") if isinstance(rep.get(k), dict) else rep.get(k) for k in ("t1", "t2")} | {
            "graph": rep.get("graph"), "errors": [e[:300] for e in rep.get("errors", [])]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"[:600]}


def e13_host_ram_guard():
    ple = sum(f.stat().st_size for f in SGL_MODEL_PATH.glob("model-ple-*.safetensors"))
    avail = None
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable"):
                avail = int(line.split()[1]) * 1024
    except Exception:
        pass
    need = ple + 20 * 10 ** 9
    return {"ple_gb": round(ple / 1e9, 1), "mem_available_gb": round(avail / 1e9, 1) if avail else None,
            "need_gb": round(need / 1e9, 1), "ok": E11_HARNESS or (avail is not None and avail >= need)}


def e13_flags():
    """Franzen's launch (his notebook cell 12) with the L4 deltas only (PLAN rev 4.1 sections 2 and 3b)."""
    run = SGL_SERVER_MAX_RUNNING
    gb = [n for n in sorted({1, 2, 4, 7, 8, 9, 10, run}) if n <= run]
    f = ["--model-path", str(SGL_MODEL_PATH), "--load-format", "safetensors",
         "--model-loader-extra-config", json.dumps({"enable_multithread_load": False}),
         "--served-model-name", "flashnext", "--host", "127.0.0.1", "--port", str(SGL_PORT),
         "--tensor-parallel-size", "4", "--ep-size", "4",                                               # D4
         "--dtype", "bfloat16", "--quantization", "auto-round", "--kv-cache-dtype", "fp8_e4m3",
         "--mem-fraction-static", E13_MEM_FRACTION, "--context-length", str(SGL_CONTEXT_LEN), "--page-size", "64",
         "--max-running-requests", str(run), "--chunked-prefill-size", str(E13_CHUNK), "--max-prefill-tokens", "16384",
         "--cuda-graph-max-bs-decode", str(run), "--cuda-graph-bs-decode", *map(str, gb),
         "--mamba-ssm-dtype", "bfloat16", "--max-mamba-cache-size", str(run * 5),                     # extra_buffer: 5 slots
         "--mamba-radix-cache-strategy", "extra_buffer", "--mamba-track-interval", "64",
         "--mamba-backend", "triton", "--linear-attn-decode-backend", "triton", "--linear-attn-prefill-backend", "triton",   # D5
         "--moe-runner-backend", "auto", "--ple-offload-embedding", "--trust-remote-code",
         "--mm-feature-transport", "cpu", "--image-processor-backend", "pil",
         "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
         "--default-chat-template-kwargs", json.dumps({"preserve_thinking": True}),
         "--watchdog-timeout", "1800", "--schedule-policy", "lpm", "--warmups", "structured_output",
         "--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging",
         "--weight-loader-prefetch-checkpoints", "--weight-loader-drop-cache-after-load"]
    if (SGL_MODEL_PATH / "chat_template.jinja").exists():
        f += ["--chat-template", str(SGL_MODEL_PATH / "chat_template.jinja")]
    return f


def _e13_caps(log):
    tp0 = "\n".join(l for l in log.splitlines() if "TP0" in l) or log
    kv = re.findall(r"KV Cache is allocated\. dtype: ([\w.]+), #tokens: (\d+), K size: ([\d.]+) GB, V size: ([\d.]+) GB", tp0)
    lw = re.findall(r"Load weight end.*?avail mem=([\d.]+) GB, mem usage=([\d.]+) GB", tp0)
    cap = re.findall(r"max_running_requests is capped to (\d+)", log)
    gr = re.findall(r"Capture [\w ]*?(?:cuda|CUDA) graph end\..*?mem usage=([\d.]+) GB, avail mem=([\d.]+) GB", tp0)
    return {"kv_dtype": kv[0][0] if kv else None, "kv_tokens": int(kv[0][1]) if kv else None,
            "post_weight_avail_gb": float(lw[0][0]) if lw else None, "weight_gb": float(lw[0][1]) if lw else None,
            "capped_running": min(int(x) for x in cap) if cap else None,
            "post_graph_avail_gb": float(gr[-1][1]) if gr else None, "oom": "out of memory" in log.lower()}


def e13_smoke():
    """'!!!!' (the E3 symptom) or a wrong continuation = not ready: raw greedy counting + one chat answer."""
    def post(path, body, timeout):
        req = _urlreq.Request(SGL_URL + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with _urlreq.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    g = post("/generate", {"text": "1, 2, 3, 4, 5, 6, 7, 8,", "sampling_params": {"max_new_tokens": 32, "temperature": 0}}, 600)
    text = g.get("text") or ""
    count_ok = text.strip().startswith("9, 10, 11")
    c = post("/v1/chat/completions", {"model": "flashnext", "max_tokens": 4096,
                                      "messages": [{"role": "user", "content": "What is 2 + 2? Answer with the number only."}]}, 1200)
    m = (c.get("choices") or [{}])[0].get("message") or {}
    ans = (m.get("content") or "") + " " + (m.get("reasoning_content") or "")
    garbage = lambda t: bool(t.strip()) and max(t.count(ch) for ch in set(t.strip())) / len(t.strip()) > 0.5 and len(t.strip()) > 8
    out = {"count_text": text[:80], "count_ok": count_ok, "chat_content": (m.get("content") or "")[:120],
           "chat_has_4": "4" in ans, "garbage": garbage(text) or garbage(m.get("content") or ""), "usage": c.get("usage")}
    out["ok"] = count_ok and out["chat_has_4"] and not out["garbage"]
    return out


def e13_bring_up():
    global E11_SERVER, _E13_ENV
    info = ENGINE_INFO["e13"]
    info["stage"] = "install"
    vpy, _E13_ENV, info["install"] = e11_install()
    _E13_ENV.update({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
                     "SGLANG_MAMBA_CONV_DTYPE": "bfloat16", "SGLANG_NUMA_BIND_V2": "false", "SGLANG_MM_PREPROCESS_DEVICE": "cpu",
                     "OMP_NUM_THREADS": "8", "TORCH_CUDA_ARCH_LIST": "8.9"})                          # Franzen env; D6
    info["stage"] = "fa2_patch3"; info["extras"] = e13_extras(vpy)
    info["stage"] = "topk_test"; info["topk"] = e13_topk_test(vpy)
    print("E13 top-k test:", json.dumps(info["topk"], default=str)[:1500], flush=True)
    info["stage"] = "host_ram"; info["host_ram"] = e13_host_ram_guard()
    if not info["host_ram"]["ok"]:
        raise E11Fail("host RAM guard: " + json.dumps(info["host_ram"]))
    info["gpu_free_before"] = _e11_gpu_free(180)
    info["stage"] = "launch"; info["flags"] = e13_flags()
    launch = [str(vpy), "-m", "sglang.launch_server"] if not E11_HARNESS else [sys.executable, os.environ["E11_FAKE_SERVER"]]
    E11_SERVER = E11Server(launch + info["flags"], _E13_ENV, "e13")
    E11_SERVER.start()
    print("E13: starting SGLang Flash-Next:", " ".join(info["flags"]), flush=True)
    deadline = time.perf_counter() + max(300, min(45 * 60, ADMIT_DEADLINE - time.perf_counter() - 20 * 60))
    last, last_t = -1, time.perf_counter()
    while True:                                          # /health only while the launcher lives; 15 min log silence = stall
        if not E11_SERVER.alive():
            raise E11Fail(f"server exited rc={E11_SERVER.proc.returncode}: " + E11_SERVER.log_text()[-3000:])
        try:
            _urlreq.urlopen(SGL_URL + "/health", timeout=10).read(); break
        except Exception:
            pass
        sz = E11_SERVER.log_path.stat().st_size if E11_SERVER.log_path.exists() else 0
        if sz != last:
            last, last_t = sz, time.perf_counter()
        if time.perf_counter() > deadline or time.perf_counter() - last_t > 15 * 60:
            raise E11Fail("server not ready (deadline or 15 min log silence): " + E11_SERVER.log_text()[-3000:])
        time.sleep(2 if E11_HARNESS else 5)
    info["ready_sec"] = round(time.perf_counter() - E11_SERVER.t0, 1)
    info["caps"] = _e13_caps(E11_SERVER.log_text())
    print("E13 caps:", json.dumps(info["caps"]), flush=True)
    info["stage"] = "smoke"; info["smoke"] = e13_smoke()
    print("E13 smoke:", json.dumps(info["smoke"], default=str)[:800], flush=True)
    if not info["smoke"]["ok"]:
        raise E11Fail("smoke failed: " + json.dumps(info["smoke"], default=str)[:600])
    kv = info["caps"].get("kv_tokens") or 0
    run = min(SGL_MAX_RUNNING, info["caps"].get("capped_running") or SGL_MAX_RUNNING)
    return max(1, min(run, kv // E13_KV_PER_STREAM))          # pre-run capacity rule: resident contexts <= KV / 60k


_E13_ENV = None
_t_engine = time.perf_counter()
if not E10_SANDBOX_SELFTEST.get("ok", False):
    ENGINE_INFO["e13"]["stage"] = "skipped: sandbox self-test failed"
else:
    try:
        CONCURRENCY = e13_bring_up()
        ENGINE_READY = True
        ENGINE_INFO.update(backend="sglang-0.5.21-flashnext-w4a16-ep4-fa2", kv_tokens=ENGINE_INFO["e13"]["caps"].get("kv_tokens"),
                           max_model_len=SGL_CONTEXT_LEN)
        ENGINE_INFO["e13"]["stage"] = "ready"
    except Exception as _e:
        ENGINE_INFO["e13"]["error"] = f"{type(_e).__name__}: {_e}"[:3000]
        ENGINE_INFO["fatal"] = ENGINE_INFO["e13"]["error"][:300]
        print("E13 engine NOT ready:", ENGINE_INFO["e13"]["stage"], "|", ENGINE_INFO["e13"]["error"][:1500], flush=True)
        if E11_SERVER is not None:
            ENGINE_INFO["e13"]["server_log_tail"] = E11_SERVER.log_text()[-4000:]
        ENGINE_READY = False; CONCURRENCY = 0
        # keep E11_SERVER (CELL 7 bundles its log); the process is stopped by CELL 7's e11_kill_server()
ENGINE_INFO["load_sec"] = round(time.perf_counter() - _t_engine, 1)
ENGINE_INFO["concurrency"] = int(CONCURRENCY)
print("engine:", json.dumps(ENGINE_INFO, default=str)[:3000], flush=True)
'''
C5 = C5.replace("TOPK_B64", __import__("base64").b64encode(TOPK_SCRIPT.encode()).decode())

nb["cells"][0]["source"] = ("# E13 — Duck REPL agent for ARC-AGI-2 on SGLang 0.5.21 + Qwen3.8-Flash-Next (Intel W4A16, BF16 PLE) "
                            f"({'EVAL30' if EVAL30 else 'SMOKE1'})\n"
                            "The E12 Duck notebook unchanged except the engine (CELL 5) and its inputs/limits in CELL 1. "
                            "Engine = Franzen's ARC-AGI-3 milestone-2 launch with the 4xL4 deltas (PLAN rev 4.1 simple path): "
                            "0.5.21, TP4+EP4, Triton GDN, arch 8.9, classic FA2 for QSA decode, Franzen patch #3; memory items "
                            "NEXTN off, chunk 2048, mem 0.93, 4 running; Duck streams = min(4, KV // 60k).")
nb["cells"][2]["source"] = c1
nb["cells"][6]["source"] = C5
for c in nb["cells"]:
    if c["cell_type"] == "code":
        c["outputs"] = []; c["execution_count"] = None
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        compile(src(c), f"cell{i}", "exec")
unchanged = [i for i in range(len(cells)) if src(nb["cells"][i]) == cells[i]]
data = json.dumps(nb, indent=1).encode("utf-8")
OUT_NB.write_bytes(data)
print("wrote", OUT_NB.name, "sha256", hashlib.sha256(data).hexdigest(), "| unchanged cells", unchanged, "of", len(cells))
