"""Build the E13 Stage 1 notebook (FN-PROBE-2): Qwen3.8-Flash-Next on SGLang 0.5.21, 4xL4, PLAN rev 4.1.

Reused verbatim (text-extracted with exactly-once asserts) from the E12 SGLang EVAL30 notebook (sha b65fb9dc, ran on Kaggle):
  CELL 0 (mode + fallback submission), and from its engine cell: E11Fail, _e11_post, _e11_gpu_used, _e11_gpu_free,
  e11_install (0.5.21 wheelhouse venv + CUDA 13 pip runtime + link shims), E11Server (own process group, killpg).
New: E13 config, FA2 wheel install, Franzen patch #3 (his SHAs), optional P1/P2 patches, static asserts, the micro-preflight
(e13_scripts/mp_*.py), Launch 1 (target-only) + memory gate + Gate C + speed stage, the pre-declared Launch 2, the report.
Run: python build_e13_probe.py  ->  ../E13-FN-PROBE2-4XL4.ipynb
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
E12NB = Path(r"D:/ARC AGI 2/experiments/Experiment 12 - Duck REPL agent for ARC-AGI-2 (planned)/Experiment Notebooks/"
             r"Q27-XHIGH-E12_DUCK_REPL_SGLANG_EVAL30.ipynb")
E12NB_SHA = "b65fb9dcc80b3652dc18b37d43355743311e2b55db8308ebc766d9d0b23de456"
FRANZEN_NB = Path(r"D:/ARC AGI 2/_scratch/ext/kaggle_dfranzen_versions/v3_run354920985.ipynb")
OUT_NB = HERE.parent / "E13-FN-PROBE2-4XL4.ipynb"
SCRIPTS = ["mp_topk.py", "mp_decode.py", "mp_mqa.py", "mp_pinned.py", "e13_gate.py"]


def src(cell):
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def between(text, start, end, label):
    assert text.count(start) == 1, f"{label}: start marker count {text.count(start)}"
    i = text.index(start)
    j = text.index(end, i) if end else len(text)
    assert end is None or text.count(end) == 1, f"{label}: end marker count {text.count(end)}"
    return text[i:j]


raw = E12NB.read_bytes()
assert hashlib.sha256(raw).hexdigest() == E12NB_SHA, "E12 EVAL30 notebook changed; refusing to reuse"
e12 = json.loads(raw)
cells = [src(c) for c in e12["cells"]]
CELL0 = cells[1]
assert CELL0.startswith("# ==================================================================== CELL 0"), "CELL 0 not found"
ENGINE = cells[6]
REUSED = (between(ENGINE, "class E11Fail(Exception):", "# ---- launch: the E11-04 W1 command", "install block")
          + "\n" + between(ENGINE, "class E11Server:", "def e11_kill_server", "E11Server"))
for must in ("def e11_install():", "def _e11_gpu_free(", "class E11Server:", "def kill(self, wait=True):"):
    assert must in REUSED, must

# Franzen's draft-view helper (cell 12 of his notebook, verbatim): only used if Launch 2 turns NEXTN on
fr = json.loads(FRANZEN_NB.read_text(encoding="utf-8"))
FR12 = src(fr["cells"][12])
FR_DRAFT = between(FR12, "TOKEN_MAP_SHA = ", "# ---- validate notebook inputs / identify the offline bundle", "Franzen draft helpers")
for must in ("def prepare_draft_view(", "def indexed_shards(", "def find_unique(", "EXPERT_TARGET"):
    assert must in FR_DRAFT, must

SCRIPT_SRC = {n: (HERE / "e13_scripts" / n).read_text(encoding="utf-8") for n in SCRIPTS}
SCRIPT_SHA = {n: hashlib.sha256(s.encode()).hexdigest()[:16] for n, s in SCRIPT_SRC.items()}

C1 = r'''# ==================================================================== CELL 1 — E13 configuration (PLAN rev 4.1)
import glob, base64, copy, shlex, signal as _signal, urllib.request as _urlreq
E13_HARNESS = os.environ.get("E13_HARNESS") == "1"            # local CPU harness: fake server, no install, no GPU kernels
E11_HARNESS = E13_HARNESS                                     # names used by the reused E11 helpers
OUT = WORKING / "e13"; OUT.mkdir(parents=True, exist_ok=True)
T0 = time.perf_counter()
HARD_STOP_SEC = float(os.environ.get("E13_HARD_STOP_SEC", 2.5 * 3600))      # whole notebook
L1_SEC, L2_SEC = 75 * 60, 60 * 60                                            # per launch (load + gates + speed)
READY_CAP_SEC, READY_STALL_SEC = 45 * 60, 15 * 60                            # progress-based server wait
SGL_PORT = int(os.environ.get("E13_PORT", 30000)); SGL_URL = f"http://127.0.0.1:{SGL_PORT}"
SERVED = "flashnext"
REPORT = {"run_id": RUN_ID, "plan": "E13 PLAN rev 4.1", "stages": {}, "decision": {}, "launches": []}

def elapsed():
    return time.perf_counter() - T0

def left():
    return HARD_STOP_SEC - elapsed()

def stage(name, ok, **info):
    REPORT["stages"][name] = dict(ok=ok, t_sec=round(elapsed(), 1), **info)
    write_json_atomic(OUT / "e13_probe_report.json", REPORT)
    print(f"[{elapsed():7.0f}s] {name}: {'OK' if ok else 'FAIL'} " + json.dumps(info, default=str)[:600], flush=True)

def _find_dir(pattern, test):
    for d in sorted(glob.glob(pattern, recursive=True)):
        try:
            if os.path.isdir(d) and test(Path(d)):
                return Path(d)
        except Exception:
            pass
    return None

def _qc(d):
    try:
        c = json.loads((d / "config.json").read_text())
        return c.get("quantization_config") or c.get("text_config", {}).get("quantization_config") or {}
    except Exception:
        return {}

INP = Path(os.environ.get("E13_INPUT_ROOT", "/kaggle/input"))
SGL_WHEELHOUSE = _find_dir(str(INP / "**" / "*sglang-0-5-21-cu13-wheelhouse*"), lambda d: (d / "SHA256SUMS").exists())
FA2_DIR = _find_dir(str(INP / "**" / "*flash-attn-2-8-3-post1*"), lambda d: (d / "SHA256SUMS_AND_SOURCE.txt").exists())
MODEL_DIR = _find_dir(str(INP / "**" / "*intel-qwen3*8-flash-next-w4a16-autoround*" / "**"),
                      lambda d: (d / "config.json").exists() and _qc(d).get("quant_method") == "auto-round")
DRAFT_DIR = _find_dir(str(INP / "**" / "*albucino-qwen3-8-flash-next-drafter*"), lambda d: d.exists())
PENNY_DIR = _find_dir(str(INP / "**" / "*pennyroyal-v253*"), lambda d: d.exists())
FA2_WHEEL = "flash_attn-2.8.3.post1+cu13torch2.13cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
FA2_SHA = "74984f628eaa9b22de271c787d82c4ef16daffcfb9684286a7a15ff6ad41ae84"
D2_BEFORE = "135e004f7613a59aa0765cbe3d779c9bb0c459f3a199b868358759e020aaeb5d"   # = Franzen patch #3 BEFORE_SHA = 0.5.21
D2_AFTER = "47e65140e2480a8637b9ea5bb746ff77e9c09c01e448fd2056c0b485e4f3a10a"    # = Franzen AFTER_SHA = Pennyroyal wheel
# Launch 1 (PLAN 3b): Franzen-verbatim except the pre-declared memory items (NEXTN off, chunk 2048, mamba = running x 5,
# mem 0.93 so that >= 1.5 GiB stays outside the static pools, PLAN M6)
LAUNCH1 = {"tag": "L1", "mem": "0.93", "ctx": 139264, "running": 8, "chunk": 2048, "mamba_cache": 40, "nextn": False,
           "no_cuda_graph": False, "decode": None, "tilelang_off": False}
REPORT["inputs"] = {k: str(v) if v else None for k, v in dict(wheelhouse=SGL_WHEELHOUSE, fa2=FA2_DIR, model=MODEL_DIR,
                                                              draft=DRAFT_DIR, penny=PENNY_DIR).items()}
stage("0_inputs", all([SGL_WHEELHOUSE, FA2_DIR, MODEL_DIR]) or E13_HARNESS, **REPORT["inputs"])
'''

C2 = r'''# ==================================================================== CELL 2 — install, patches, static asserts
# ---- reused verbatim from the E12 EVAL30 notebook engine cell (Kaggle-proven): install + server process helpers
''' + REUSED + r'''

E13_SCRIPTS = json.loads(base64.b64decode("BASE64_SCRIPTS").decode())
SCRIPT_DIR = OUT / "scripts"; SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
for _n, _s in E13_SCRIPTS.items():
    (SCRIPT_DIR / _n).write_text(_s, encoding="utf-8")
sys.path.insert(0, str(SCRIPT_DIR))
import e13_gate

def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(16 << 20), b""):
            h.update(b)
    return h.hexdigest()

def _replace_once_atomic(path, old, new, label):
    t = path.read_text(encoding="utf-8")
    if t.count(old) != 1:
        raise E11Fail(f"{label}: anchor found {t.count(old)} times")
    tmp = path.with_name(path.name + ".e13tmp"); tmp.write_text(t.replace(old, new), encoding="utf-8")
    os.replace(tmp, path)

VPY, VENV_ENV, SITE = None, None, None
try:
    VPY, VENV_ENV, _shims = e11_install()
    SITE = Path(subprocess.run([str(VPY), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                               capture_output=True, text=True, check=True).stdout.strip())
    stage("1_install", True, shims=_shims, site=str(SITE))
    if not E13_HARNESS:
        # FA2 (R16): Kaggle may drop '+' from file names; restore the name as a link after the SHA check, then --no-deps
        _w = FA2_DIR / FA2_WHEEL if (FA2_DIR / FA2_WHEEL).exists() else FA2_DIR / FA2_WHEEL.replace("+", "")
        if _sha(_w) != FA2_SHA:
            raise E11Fail("FA2 wheel sha256 mismatch")
        _fl = Path("/tmp/e13-fa2"); _fl.mkdir(exist_ok=True)
        if not (_fl / FA2_WHEEL).exists():
            (_fl / FA2_WHEEL).symlink_to(_w)
        _r = subprocess.run([sys.executable, "-m", "pip", "--python", str(VPY), "install", "--no-index", "--no-deps",
                             "--find-links", "/tmp/e11-wheels", str(_fl / FA2_WHEEL), "-q", "--disable-pip-version-check"],
                            capture_output=True, text=True)
        if _r.returncode:
            raise E11Fail("FA2 install failed: " + (_r.stderr or _r.stdout)[-800:])
        # The FA2 wheel also ships 21 older flash_attn/cute/* files that overwrite FA4's (flash_attn_4 has no
        # flash_attn/__init__.py). Reinstall the wheelhouse FA4 so its cute files are restored; FA2's own files stay.
        _fa4 = sorted(Path("/tmp/e11-wheels").glob("flash_attn_4-*.whl"))
        if _fa4:
            _r = subprocess.run([sys.executable, "-m", "pip", "--python", str(VPY), "install", "--no-index", "--no-deps",
                                 "--force-reinstall", str(_fa4[0]), "-q", "--disable-pip-version-check"],
                                capture_output=True, text=True)
            if _r.returncode:
                raise E11Fail("FA4 restore failed: " + (_r.stderr or _r.stdout)[-800:])
        # D2 = Franzen patch #3: two substitutions, his own BEFORE/AFTER SHAs (the 0.5.21 file equals his BEFORE)
        _g = SITE / "sglang/srt/layers/quantization/gptq/schemes/gptq_moe.py"
        _h = _sha(_g)
        if _h == D2_BEFORE:
            _t = _g.read_bytes()
            assert _t.count(b"dtype=torch.half") == 2
            _tmp = _g.with_name(_g.name + ".e13tmp"); _tmp.write_bytes(_t.replace(b"dtype=torch.half", b"dtype=params_dtype"))
            os.replace(_tmp, _g); _h = _sha(_g)
        if _h != D2_AFTER:
            raise E11Fail(f"patch #3 after-sha mismatch {_h[:12]}")
    stage("2_fa2_and_patch3", True)
except Exception as e:
    stage("2_fa2_and_patch3", False, error=f"{type(e).__name__}: {e}"[:1500])

def apply_p1():
    """PLAN D3 fallback: route sm89 decode to SGLang's own Triton packed-decode branch (the is_hip() branch)."""
    p = SITE / "sglang/srt/layers/attention/qwen_sparse_attn_backend.py"
    _replace_once_atomic(p, "from sglang.srt.utils import is_hip\n",
                         "from sglang.srt.utils import is_hip\nimport os as _e13_os\n"
                         "_E13_SM89_TRITON_DECODE = _e13_os.environ.get('E13_SM89_TRITON_DECODE') == '1'\n", "P1 import")
    _replace_once_atomic(p, "        if is_hip():\n", "        if is_hip() or _E13_SM89_TRITON_DECODE:\n", "P1 branch")

def apply_p2():
    """PLAN B2 option: SGLang's torch reference for the QSA indexer MQA instead of TileLang."""
    p = SITE / "sglang/srt/layers/attention/qsa/mqa.py"
    _replace_once_atomic(p, "    HAS_TILELANG = True\n",
                         "    HAS_TILELANG = __import__('os').environ.get('E13_DISABLE_TILELANG') != '1'\n", "P2")

# ---- static asserts (PLAN 3c): version, GPUs, FA2 ABI, patch, model config, flag existence
def static_asserts():
    info = {}
    probe = ("import json, torch, sglang; d={'sglang': sglang.__version__, 'torch': torch.__version__,"
             "'cxx11abi': torch.compiled_with_cxx11_abi(), 'caps': [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]};"
             "\ntry:\n import flash_attn; d['flash_attn']=flash_attn.__version__\nexcept Exception as e: d['flash_attn']='ERR '+str(e)[:200]"
             "\ntry:\n from sglang.srt.layers.attention.qwen_sparse_attn_backend import _resolve_flash_attn_varlen_func as r"
             "\n d['qsa_decode_resolves_to']=r().__module__\nexcept Exception as e: d['qsa_decode_resolves_to']='ERR '+str(e)[:300]"
             "\nprint(json.dumps(d))")
    if E13_HARNESS:
        info.update(sglang="0.5.21", caps=[[8, 9]] * 4, flash_attn="2.8.3.post1", cxx11abi=True, harness=True,
                    qsa_decode_resolves_to="flash_attn.flash_attn_interface")
    else:
        r = subprocess.run([str(VPY), "-c", probe], capture_output=True, text=True, env=VENV_ENV, timeout=900)
        info.update(json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"probe_error": r.stderr[-800:]})
    cfg = json.loads((MODEL_DIR / "config.json").read_text()) if MODEL_DIR else {}
    tc = cfg.get("text_config", cfg); qc = _qc(MODEL_DIR) if MODEL_DIR else {}
    info.update(desc_act=qc.get("desc_act"), quant=qc.get("quant_method"), bits=qc.get("bits"), group=qc.get("group_size"),
                indexer=(tc.get("indexer_budget"), tc.get("indexer_compress_ratio")), kv_heads=tc.get("num_key_value_heads"),
                moe_inter=tc.get("moe_intermediate_size"), experts=tc.get("num_experts"))
    # effective MoE shapes per rank (PLAN 3c Marlin): EP4 keeps whole experts -> w13 N = 2 x inter, w2 K = inter
    inter = tc.get("moe_intermediate_size") or 0
    info["moe_shapes_ep4"] = {"w13_N": 2 * inter, "w2_K": inter, "experts_per_rank": (tc.get("num_experts") or 0) // 4,
                              "multiple_of_group128": (2 * inter) % 128 == 0 and inter % 128 == 0}
    ok = (str(info.get("sglang", "")).startswith("0.5.21") and len(info.get("caps") or []) == 4
          and all(tuple(c) == (8, 9) for c in info["caps"]) and info.get("cxx11abi") is True
          and not info.get("desc_act") and info["indexer"] == (2048, 4)
          and info["moe_shapes_ep4"]["multiple_of_group128"]) or E13_HARNESS
    # FA2 version / resolver are recorded, not gated here: if FA2 is broken, the micro-preflight picks P1 (PLAN 1b)
    info["fa2_resolved"] = str(info.get("flash_attn", "")).startswith("2.8.3") and \
        str(info.get("qsa_decode_resolves_to", "")).startswith("flash_attn.") and "cute" not in str(info.get("qsa_decode_resolves_to"))
    return ok, info

try:
    _ok, _info = static_asserts()
    stage("3_static_asserts", _ok, **_info)
except Exception as e:
    stage("3_static_asserts", False, error=f"{type(e).__name__}: {e}"[:1000])
'''

C3 = r'''# ==================================================================== CELL 3 — micro-preflight (GPU kernels, no model)
MICRO = {}
def _run_script(name, args, timeout):
    cmd = [str(VPY) if not E13_HARNESS else sys.executable, "-I", str(SCRIPT_DIR / name)] + args
    t = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, env=VENV_ENV if not E13_HARNESS else None, timeout=timeout)
    (OUT / f"{name}.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    return r.returncode, round(time.perf_counter() - t, 1)

_pre_ok = all(REPORT["stages"].get(s, {}).get("ok") for s in ("1_install", "2_fa2_and_patch3", "3_static_asserts"))
if _pre_ok:
    _self = ["--cpu-selftest"] if E13_HARNESS else []
    for _name, _extra, _to in (("mp_decode.py", _self, 1800), ("mp_topk.py", _self + ["--fixtures", str(OUT / "topk_fixtures")], 2400),
                               ("mp_mqa.py", [], 1800),
                               ("mp_pinned.py", ["--model-dir", str(MODEL_DIR)] + (["--per-rank-bytes", "1000000"] if E13_HARNESS else []), 900)):
        if E13_HARNESS and _name in ("mp_mqa.py", "mp_pinned.py"):           # need a GPU; plumbing only locally
            MICRO[_name[3:-3]] = {"pass": True, "harness": True}; continue
        try:
            _rc, _sec = _run_script(_name, ["--out", str(OUT / (_name[:-3] + ".json"))] + _extra, _to)
            MICRO[_name[3:-3]] = json.loads((OUT / (_name[:-3] + ".json")).read_text())
            MICRO[_name[3:-3]]["_rc"], MICRO[_name[3:-3]]["_sec"] = _rc, _sec
        except Exception as e:
            MICRO[_name[3:-3]] = {"error": f"{type(e).__name__}: {e}"[:800]}
    _d = MICRO.get("decode", {})
    _g = {g.get("impl"): g.get("pass") for g in _d.get("graph", [])}
    fa2_ok = bool((_d.get("fa2") or {}).get("pass")) and (_g.get("fa2") is not False) \
        and bool(REPORT["stages"].get("3_static_asserts", {}).get("fa2_resolved"))      # the server must resolve to FA2 too
    p1_ok = bool((_d.get("p1") or {}).get("pass")) and (_g.get("p1_triton_packed") is not False)
    if E13_HARNESS:
        fa2_ok, p1_ok = True, False
    DECODE = "fa2" if fa2_ok else ("p1" if p1_ok else None)          # PLAN 4 1b pre-declared choice
    _t1 = (MICRO.get("topk", {}).get("t1") or {}).get("summary"); _t2 = (MICRO.get("topk", {}).get("t2") or {}).get("summary")
    REPORT["decision"]["micro"] = {"decode": DECODE, "fa2_pass": fa2_ok, "p1_pass": p1_ok, "graph": _g,
                                   "mqa_pass": MICRO.get("mqa", {}).get("pass"), "topk_t1": _t1, "topk_t2": _t2,
                                   "pinned_pass": MICRO.get("pinned", {}).get("pass")}
    MICRO_FLAGS = {"mqa_pass": MICRO.get("mqa", {}).get("pass"),
                   "decode_graph_pass": _g.get("fa2" if DECODE == "fa2" else "p1_triton_packed"),
                   "alt_decode_pass": p1_ok if DECODE == "fa2" else fa2_ok}
    stage("4_micro_preflight", DECODE is not None and MICRO.get("pinned", {}).get("pass") is True,
          **REPORT["decision"]["micro"])
else:
    DECODE, MICRO_FLAGS = None, {}
    stage("4_micro_preflight", False, error="skipped: install / patch / static asserts failed")
'''

C4 = r'''# ==================================================================== CELL 4 — Launch 1 (target-only), gates, Launch 2
def e13_flags(cfg):
    gb = [n for n in sorted({1, 2, 4, 7, 8, 9, 10, cfg["running"]}) if n <= cfg["running"]]   # Franzen's graph sizes
    f = ["--model-path", str(MODEL_DIR), "--load-format", "safetensors",
         "--model-loader-extra-config", json.dumps({"enable_multithread_load": False}),
         "--served-model-name", SERVED, "--host", "127.0.0.1", "--port", str(SGL_PORT),
         "--tensor-parallel-size", "4", "--ep-size", "4",                                       # D4
         "--dtype", "bfloat16", "--quantization", "auto-round", "--kv-cache-dtype", "fp8_e4m3",
         "--mem-fraction-static", cfg["mem"], "--context-length", str(cfg["ctx"]), "--page-size", "64",
         "--max-running-requests", str(cfg["running"]), "--chunked-prefill-size", str(cfg["chunk"]),
         "--max-prefill-tokens", "16384",
         "--mamba-ssm-dtype", "bfloat16", "--max-mamba-cache-size", str(cfg["mamba_cache"]),
         "--mamba-radix-cache-strategy", "extra_buffer", "--mamba-track-interval", "64",
         "--mamba-backend", "triton", "--linear-attn-decode-backend", "triton", "--linear-attn-prefill-backend", "triton",  # D5
         "--moe-runner-backend", "auto", "--ple-offload-embedding", "--trust-remote-code",
         "--mm-feature-transport", "cpu", "--image-processor-backend", "pil",
         "--reasoning-parser", "qwen3", "--tool-call-parser", "qwen3_coder",
         "--default-chat-template-kwargs", json.dumps({"preserve_thinking": True}),
         "--watchdog-timeout", "1800", "--schedule-policy", "lpm", "--warmups", "structured_output",
         "--enable-cache-report", "--enable-metrics", "--enable-request-time-stats-logging",
         "--weight-loader-prefetch-checkpoints", "--weight-loader-drop-cache-after-load"]
    if cfg.get("no_cuda_graph"):
        f += ["--disable-cuda-graph"]
    else:
        f += ["--cuda-graph-max-bs-decode", str(cfg["running"]), "--cuda-graph-bs-decode", *map(str, gb)]
    if MODEL_DIR and (MODEL_DIR / "chat_template.jinja").exists():
        f += ["--chat-template", str(MODEL_DIR / "chat_template.jinja")]
    if cfg.get("nextn"):
        f += ["--speculative-algorithm", "NEXTN", "--speculative-num-steps", "3", "--speculative-eagle-topk", "1",
              "--speculative-num-draft-tokens", "4", "--speculative-draft-model-path", str(cfg["draft_view"]),
              "--speculative-draft-model-quantization", "compressed-tensors", "--speculative-moe-runner-backend", "auto",
              "--speculative-draft-kv-cache-dtype", "fp8_e4m3", "--speculative-accept-threshold-single", "1.0",
              "--speculative-accept-threshold-acc", "1.0", "--speculative-token-map", str(cfg["token_map"])]
    for bad in ("--gdn-mtp-cache-mode", "--language-model-only"):
        assert bad not in f
    return f

def e13_env(cfg):
    env = dict(VENV_ENV or os.environ)
    env.update({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
                "SGLANG_MAMBA_CONV_DTYPE": "bfloat16", "SGLANG_NUMA_BIND_V2": "false", "SGLANG_MM_PREPROCESS_DEVICE": "cpu",
                "OMP_NUM_THREADS": "8", "TOKENIZERS_PARALLELISM": "false", "TORCH_CUDA_ARCH_LIST": "8.9"})          # D6
    if cfg.get("decode") == "p1":
        env["E13_SM89_TRITON_DECODE"] = "1"
    if cfg.get("tilelang_off"):
        env["E13_DISABLE_TILELANG"] = "1"
    return env

class GpuSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.max_used = None; self.min_avail = None; self.stop_ = False
    def run(self):
        while not self.stop_:
            u = _e11_gpu_used()
            if u:
                self.max_used = max(u) if self.max_used is None else max(self.max_used, max(u))
            try:
                for line in open("/proc/meminfo"):
                    if line.startswith("MemAvailable"):
                        v = int(line.split()[1]) * 1024 / 1e9
                        self.min_avail = v if self.min_avail is None else min(self.min_avail, v)
            except Exception:
                pass
            time.sleep(2)

def wait_ready_progress(srv, cap):
    """/health only while the launcher lives; fail on 15 min of log silence (progress-based, R6)."""
    last_size, last_change = -1, time.perf_counter()
    while time.perf_counter() - srv.t0 < cap and left() > 60:
        if not srv.alive():
            return False, f"server exited rc={srv.proc.returncode}"
        try:
            _urlreq.urlopen(SGL_URL + "/health", timeout=10).read()
            return True, f"ready in {time.perf_counter() - srv.t0:.0f}s"
        except Exception:
            pass
        sz = srv.log_path.stat().st_size if srv.log_path.exists() else 0
        if sz != last_size:
            last_size, last_change = sz, time.perf_counter()
        elif time.perf_counter() - last_change > READY_STALL_SEC:
            return False, "stalled: no log progress for 15 min"
        time.sleep(5)
    return False, "timeout"

def run_launch(cfg, budget_sec):
    rec = {"cfg": {k: v for k, v in cfg.items() if k not in ("draft_view", "token_map")}, "t_start": round(elapsed())}
    t_launch = time.perf_counter()
    if cfg.get("decode") == "p1" and not E13_HARNESS:
        apply_p1()
    if cfg.get("tilelang_off") and not E13_HARNESS:
        apply_p2()
    rec["gpu_free_before"] = _e11_gpu_free(180)
    cmd = ([str(VPY), "-m", "sglang.launch_server"] if not E13_HARNESS else [sys.executable, os.environ["E13_FAKE_SERVER"]])
    flags = e13_flags(cfg); rec["flags"] = flags
    srv = E11Server(cmd + flags, e13_env(cfg), cfg["tag"]); srv.log_path = OUT / f"sglang_server_{cfg['tag']}.log"
    sampler = GpuSampler(); sampler.start()
    try:
        srv.start()
        ok, why = wait_ready_progress(srv, min(READY_CAP_SEC, budget_sec - 300))
        rec["ready"] = ok; rec["ready_note"] = why
        mem = e13_gate.memory_gate(srv.log_text()); rec["memory"] = mem
        if not ok:
            rec["gate_c"] = {"pass": False, "signature": "oom_or_cuda_error" if mem.get("oom") else "load_failed",
                             "hard_failures": [why]}
            rec["log_tail"] = srv.log_text()[-6000:]
            return rec
        gate = e13_gate.Gate(SGL_URL, SERVED, str(OUT), log=lambda m: print(m, flush=True))
        running = mem.get("capped_running") or cfg["running"]
        rec["gate_c"] = gate.run_gate_c(mem.get("kv_tokens"), running)
        rec["peak_gpu_used_mb_gate"] = sampler.max_used
        if rec["gate_c"]["pass"] and time.perf_counter() - t_launch < budget_sec - 900:
            kv = mem.get("kv_tokens") or 0
            levels = [n for n in (1, 2, 4, 8) if n <= running and n * 16600 <= max(kv, 1)] or [1]
            rec["speed_16k"] = gate.speed(levels, 16000)
            if kv >= 2 * 50000 and time.perf_counter() - t_launch < budget_sec - 600:
                rec["speed_48k"] = gate.speed([n for n in (1, 2) if n * 50000 <= kv], 48000)
        rec["peak_gpu_used_mb"] = sampler.max_used; rec["min_host_avail_gb"] = sampler.min_avail
        rec["peak_free_gb"] = round((23034 - sampler.max_used) / 1024, 2) if sampler.max_used else None   # L4 = 23034 MiB
        return rec
    except Exception as e:
        rec["error"] = traceback.format_exc()[-2000:]
        rec.setdefault("gate_c", {"pass": False, "signature": "exception", "hard_failures": [str(e)[:300]]})
        return rec
    finally:
        sampler.stop_ = True
        rec["memory_after"] = e13_gate.memory_gate(srv.log_text())       # the decision uses the log at the END (OOM later)
        if rec["memory_after"].get("oom") and rec.get("gate_c"):
            rec["gate_c"]["signature"] = "oom_or_cuda_error"
        rec["server_log_tail"] = srv.log_text()[-4000:]
        srv.kill(wait=True)
        rec["t_end"] = round(elapsed())
        REPORT["launches"].append(rec); write_json_atomic(OUT / "e13_probe_report.json", REPORT)

if REPORT["stages"].get("4_micro_preflight", {}).get("ok") and left() > L1_SEC:
    L1 = dict(LAUNCH1, decode=DECODE, tilelang_off=(MICRO_FLAGS.get("mqa_pass") is False))   # PLAN 1b: indexer rule
    r1 = run_launch(L1, L1_SEC)
    stage("5_launch1", bool(r1.get("gate_c", {}).get("pass")), ready=r1.get("ready"), note=r1.get("ready_note"),
          memory=r1.get("memory"), gate=r1.get("gate_c", {}).get("hard_failures"), signature=r1.get("gate_c", {}).get("signature"),
          speed=r1.get("speed_16k"))
    kind, why = e13_gate.launch2_decision(r1.get("gate_c", {}), r1.get("memory_after") or r1.get("memory", {}),
                                          r1.get("peak_free_gb"), MICRO_FLAGS)
    REPORT["decision"]["launch2"] = {"kind": kind, "why": why}
    stage("6_launch2_decision", True, kind=kind, why=why)
    L2 = None
    if kind == "nextn_on":
        try:
            dv = prepare_draft_view(MODEL_DIR, DRAFT_DIR, Path("/tmp/e13-draft"))
            tok = find_unique(PENNY_DIR, "hot_tokens_64k.pt")
            if sha256(tok) != TOKEN_MAP_SHA or sha256(MODEL_DIR / "tokenizer.json") != TOKENIZER_SHA:
                raise E11Fail("FR-Spec map or tokenizer sha differs from Franzen's")
            L2 = dict(L1, tag="L2_nextn", nextn=True, draft_view=dv, token_map=tok)
        except Exception as e:
            stage("6b_draft_view", False, error=f"{type(e).__name__}: {e}"[:800])
    elif kind == "chunk_4096":
        L2 = dict(L1, tag="L2_chunk4096", chunk=4096)
    elif kind == "b2_tilelang_off":
        L2 = dict(L1, tag="L2_b2_tilelang_off", tilelang_off=True)
    elif kind == "b2_no_cuda_graph":
        L2 = dict(L1, tag="L2_b2_no_graph", no_cuda_graph=True)
    elif kind == "b2_other_decode":
        L2 = dict(L1, tag="L2_b2_decode", decode=("p1" if DECODE == "fa2" else "fa2"))
    elif kind == "b2_diagnostic":
        L2 = dict(L1, tag="L2_b2_diagnostic", tilelang_off=True, no_cuda_graph=True)
    if L2 is not None and left() > L2_SEC:
        r2 = run_launch(L2, L2_SEC)
        stage("7_launch2", bool(r2.get("gate_c", {}).get("pass")), tag=L2["tag"], ready=r2.get("ready"),
              memory=r2.get("memory"), gate=r2.get("gate_c", {}).get("hard_failures"), speed=r2.get("speed_16k"))
    elif L2 is not None:
        stage("7_launch2", False, error=f"not enough time left ({left():.0f}s)")
else:
    stage("5_launch1", False, error="skipped: micro-preflight failed or not enough time")
'''

C5 = r'''# ==================================================================== CELL 5 — report and bundle
REPORT["elapsed_sec"] = round(elapsed()); REPORT["script_sha"] = ''' + json.dumps(SCRIPT_SHA) + r'''
write_json_atomic(OUT / "e13_probe_report.json", REPORT)
with zipfile.ZipFile(WORKING / "E13_PROBE_BUNDLE.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        if p.is_file() and p.stat().st_size < 200 * 1024 * 1024:
            z.write(p, p.relative_to(WORKING))
print("E13 PROBE REPORT:", OUT / "e13_probe_report.json", "| elapsed", REPORT["elapsed_sec"], "s")
for k, v in REPORT["stages"].items():
    print(f"  {k:22s} {'OK' if v['ok'] else 'FAIL'}")
print("decision:", json.dumps(REPORT["decision"], default=str)[:2000])
'''

payload = __import__("base64").b64encode(json.dumps(SCRIPT_SRC).encode()).decode()


def build():
    assert C2.count("BASE64_SCRIPTS") == 1
    c2 = C2.replace("BASE64_SCRIPTS", payload)
    head = "# E13 FN-PROBE-2: Qwen3.8-Flash-Next on 4xL4 (SGLang 0.5.21), Stage 1 of PLAN rev 4.1\n" \
           "No model is used for the submission: CELL 0 writes the identity fallback submission.json (dual-mode safe)."
    nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": head}]
          + [{"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
             for s in (CELL0, C1, c2 + "\n" + FR_DRAFT, C3, C4, C5)],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"][1:], 1):
        compile(c["source"], f"cell{i}", "exec")
    data = json.dumps(nb, indent=1).encode("utf-8")
    OUT_NB.write_bytes(data)                                   # bytes: no Windows CRLF translation, sha = file sha
    assert hashlib.sha256(OUT_NB.read_bytes()).hexdigest() == hashlib.sha256(data).hexdigest()
    print("wrote", OUT_NB, "sha256", hashlib.sha256(data).hexdigest(), "scripts", SCRIPT_SHA)


if __name__ == "__main__":
    build()
