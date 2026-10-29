"""E23 six-lever GPU notebook builder, rev 3 (user plan 2026-10-29; 5 validation agents 2026-10-29).
Base: our vLLM 0.32 harness pattern + the E23 prep artifacts. Adopted from the consolidated v3 notebook (src/e23_lab.py):
margin oracle, settled steady window, GPU-release gate, fail-closed statuses, checkpoint identity, runtime --help flag
check (every launch), KV + GPU memory samples, runtime proof (graph verdict, backend / KV-dtype / capture log lines).
User decisions: thinking up to 120k (max-model-len 131072, 120k band); FULL uses 11.5 h of the 12 h session (spare
time -> A/A repeats); SMOKE before FULL.
rev 3 fixes: port TIME_WAIT (connect check), Merlin -lcuda shim + system GCC at serve + per-replica ports / caches +
TP1 memory fallback + checkpoint prefetch, PP4 last with an --enforce-eager arm (#63505), EAGLE own env + BF16-KV oracle,
NVFP4 teacher-forced long-context check, patch dry-run / revert gate, profiler bounded, RERUN without an exception,
stock install checked, kit sha checks, install failures cached, mode baked per notebook (Kaggle commits cannot set env).
Outputs (identical code, only the default MODE / PART differ):
  Q27-XHIGH-E23_SIX_LEVERS_SMOKE_A.ipynb  (Merlin: P5, TP1 + P4, TP4)
  Q27-XHIGH-E23_SIX_LEVERS_SMOKE_B.ipynb  (stock vLLM 0.32 + EAGLE + PP4)
  Q27-XHIGH-E23_SIX_LEVERS_FULL.ipynb     (the six sections, 11.5 h)"""
import hashlib
import json
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
E23 = HERE.parents[1]
LAB = (E23 / "src" / "e23_lab.py").read_text(encoding="utf-8").replace("\r\n", "\n")
P60532 = (E23 / "patches" / "pr60532.diff").read_text(encoding="utf-8").replace("\r\n", "\n")

MD0 = """# E23 — six levers on 4×L4 (vLLM 0.32, Qwen3.8-27B), target 1500+ tok/s · rev 3b · __VARIANT__

The user's six experiments, each with only its own test sequence (DESIGN.md), at the production context
(thinking up to 120k: `--max-model-len 131072`, bands 2k / 8k / 40k / 90k / 120k):
1. TP1×4 / PP4 (Merlin W4A16 TP1 replicas; PP4 branch on stock vLLM + RedHat INT4, eager + compiled)
2. Four-rank host-staged all-reduce (P5 bench)
3. FULL-graph speculative verification (G0 FlashInfer PIECEWISE, G1/G2 TRITON FULL, G3 FlashInfer + #60532) with a graph verdict
4. SM89 FP8 GEMM (#61224 CUTLASS vs Marlin, P4 bench) + an INT4 TP4 control row
5. NVFP4 KV: correctness (greedy + teacher-forced 40k / 90k), 90k / 120k residency, + MTP3
6. MTP OFF / K1 / K3 at 8/12/16/20 active streams (step time from the iteration histogram)
plus EAGLE-4 feasibility (SMOKE B; FULL only with E23_EAGLE_FULL=1).

This copy runs `MODE=__MODE__`, `PART=__PART__` by default (the code is identical in the three E23 notebooks).
Every number is a measured counter; a missing measurement is INCONCLUSIVE / SKIPPED, never PASS or zero.
"""

CELL1 = r'''# ==================================================================== CELL 1 — config, inputs, lab module
import os, sys, json, time, re, shutil, subprocess, hashlib, tarfile, threading, traceback
from pathlib import Path
T_START = time.time()
MODE = os.environ.get("Q27X_E23_MODE", "__MODE__").upper()        # SMOKE | FULL (baked per notebook)
PART = os.environ.get("Q27X_E23_PART", "__PART__").upper()        # SMOKE: A (Merlin) | B (stock + EAGLE) | ALL
assert MODE in ("SMOKE", "FULL") and PART in ("A", "B", "ALL"), (MODE, PART)
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes"}
FAKE = os.environ.get("E23_FAKE_SERVER")                          # local CPU harness only
BUDGET = int(os.environ.get("E23_BUDGET_SEC") or {"SMOKE": {"A": 120, "B": 150, "ALL": 250}[PART] * 60, "FULL": 690 * 60}[MODE])
HARD_STOP = T_START + BUDGET
MAX_LEN = 131072                                                  # user 2026-10-29: thinking up to 120k
OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/e23"))
OUT.mkdir(parents=True, exist_ok=True)
LOGS = OUT / "logs"
LOGS.mkdir(exist_ok=True)
TMP = Path(os.environ.get("E23_TMP", "/tmp"))
INP = Path(os.environ.get("E23_INPUT", "/kaggle/input"))
R = {"mode": MODE, "part": PART, "budget_min": BUDGET // 60, "max_len": MAX_LEN, "rev": "3b",
     "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "setup": {}, "sections": {}, "errors": [], "order": []}
_SAVE_LOCK = threading.Lock()


def save():
    with _SAVE_LOCK:
        tmp = OUT / "e23_results.json.tmp"
        tmp.write_text(json.dumps(R, indent=1, default=str))
        tmp.replace(OUT / "e23_results.json")


def left():
    return HARD_STOP - time.time()


def log(*a):
    print(f"[{(time.time() - T_START) / 60:6.1f} min]", *a, flush=True)


def find(pattern, root=INP, must=True):
    hits = sorted(Path(root).rglob(pattern)) if Path(root).exists() else []
    if not hits and must:
        raise FileNotFoundError(f"{pattern} not under {root}: attach the input")
    return hits[0] if hits else None


if RERUN:      # rev 3: a competition rerun writes the minimal valid submission and skips cells 2-6 (no exception)
    tasks = json.load(open(find("arc-agi_test_challenges.json")))
    sub = {k: [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in v["test"]] for k, v in tasks.items()}
    Path("/kaggle/working/submission.json").write_text(json.dumps(sub))
    print("E23 lab notebook: competition rerun -> fallback submission.json only")
else:
    LAB_SRC = __LAB__
    (TMP / "e23_lab.py").write_text(LAB_SRC)
    sys.path.insert(0, str(TMP))
    import e23_lab as L
    P60532 = __P60532__
    M = INP / "models" / "foysalemonshanto"
    FP8 = find("config.json", M / "qwen3-8-27b-fp8-repacked-v1").parent
    _i4 = find("config.json", M / "qwen3-8-27b-int4-redhat", must=False)
    INT4 = _i4.parent if _i4 else None
    _mm = find("q27_e23_p1_manifest.json", M / "qwen3-8-27b-w4a16-merlin", must=False)
    MERLIN = _mm.parent if _mm else None
    MERLIN_PAY = find("merlin_offline_tar.bin", M / "e23-merlin-env", must=False)
    EAGLE_PAY = find("eagle_offline_tar.bin", M / "e23-eagle4-env", must=False)
    _ed = find("config.json", M / "qwen3-8-27b-eagle4", must=False)
    EAGLE_DRAFT = _ed.parent if _ed else None
    KITS = find("q27_e23_kernels_manifest.json", INP).parent
    WHEELHOUSE = find("install_wheelhouse.py", INP / "datasets" / "foysalemonshanto" / "vllm-0-32-0-cu13-wheelhouse")
    ARC = find("arc-agi_evaluation_challenges.json")
    R["inputs"] = {k: str(v) for k, v in dict(FP8=FP8, INT4=INT4, MERLIN=MERLIN, MERLIN_PAY=MERLIN_PAY, EAGLE_PAY=EAGLE_PAY,
                                              EAGLE_DRAFT=EAGLE_DRAFT, KITS=KITS, WHEELHOUSE=WHEELHOUSE, ARC=ARC).items()}
    MERLIN_OK = False
    if MERLIN:
        mt = json.loads((MERLIN / "q27_e23_p1_manifest.json").read_text())
        MERLIN_OK = all((MERLIN / n).is_file() and (MERLIN / n).stat().st_size == s
                        for n, (s, _) in mt["files"].items() if n != ".gitattributes")
    R["inputs"]["merlin_files_ok"] = MERLIN_OK
    km = json.loads((KITS / "q27_e23_kernels_manifest.json").read_text())       # rev 3: the kit files are what we built
    R["inputs"]["kits_sha_ok"] = {n: (KITS / n).is_file() and hashlib.sha256((KITS / n).read_bytes()).hexdigest() == v.get("sha256")
                                  for n, v in km.items() if isinstance(v, dict) and n.endswith((".so", ".py"))}
    R["identity"] = {k: L.identity(v) for k, v in dict(fp8=FP8, int4=INT4, merlin=MERLIN, eagle_draft=EAGLE_DRAFT).items() if v}
    ABORT = {"why": None}
    L.start_watchdog(HARD_STOP, lambda: ABORT.update(why="HARD_STOP reached: watchdog killed live servers"))
    log("MODE", MODE, "PART", PART, "| budget", BUDGET // 60, "min |", json.dumps(R["inputs"])[:900])
    save()
'''

CELL2 = r'''# ==================================================================== CELL 2 — engines (stock vLLM 0.32, Merlin, EAGLE fork)
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
if FAKE:
    STOCK = {"vllm": [sys.executable, FAKE], "env": dict(os.environ), "py": sys.executable, "name": "stock"}
else:
    rc, txt = run("install_stock", f"{sys.executable} {WHEELHOUSE} {TMP}/vllm-venv", timeout=2400)
    kv = dict(l.split("=", 1) for l in txt.splitlines() if l.startswith(("VENV_PYTHON=", "CUDA_HOME=")))
    if rc or "CUDA_HOME" not in kv:                                   # rev 3: a clear failure, not a KeyError
        raise RuntimeError("stock vLLM 0.32 install failed (logs/install_stock.log)")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_PATH", "LD_PRELOAD")}
    env.update(CUDA_HOME=kv["CUDA_HOME"], PATH=f"{TMP}/vllm-venv/bin:{kv['CUDA_HOME']}/bin:{env.get('PATH', '')}",
               LIBRARY_PATH=f"{SHIM}:{env.get('LIBRARY_PATH', '')}", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_NO_USAGE_STATS="1", VLLM_CACHE_ROOT=f"{TMP}/vllm-cache", VLLM_ALLOW_LONG_MAX_MODEL_LEN="1")
    STOCK = {"vllm": [f"{TMP}/vllm-venv/bin/vllm", "serve"], "env": env, "py": kv["VENV_PYTHON"], "cuda_home": kv["CUDA_HOME"],
             "name": "stock"}
    R["setup"]["stock_version"] = subprocess.run([kv["VENV_PYTHON"], "-c", "import vllm; print(vllm.__version__)"],
                                                 capture_output=True, text=True, env=env).stdout.strip()
HELP = {}


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


def merlin_server(name, profile, gpus, port, kv_bytes, seqs=None, capture=None):
    """Merlin's own launch.build() (pure: the profile's flags + env, no host refusal), our max len / port last, and
    seqs / capture sizes only when given (None = the profile's own, the TP1 memory fallback)."""
    eng = merlin_install()
    extra = ["--served-model-name", "q27", "--max-model-len", str(MAX_LEN), "--port", str(port)]
    if seqs:
        extra += ["--max-num-seqs", str(seqs)]
    if capture:
        extra += ["--compilation-config", json.dumps({"cudagraph_mode": "FULL_AND_PIECEWISE", "cudagraph_capture_sizes": capture,
                                                      "inductor_compile_config": {"test_configs.force_filter_reduction_configs": True,
                                                                                  "dynamic_scale_rblock": False}})]
    if FAKE:
        return L.Server(name, [sys.executable, FAKE, str(MERLIN), "--tensor-parallel-size", str(len(gpus.split(","))),
                               "--max-num-seqs", str(seqs or 2), "--port", str(port)], os.environ, port, LOGS, gpus=gpus)
    kvv = eng["kv"]
    base = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_PATH", "CUDA_HOME", "LD_PRELOAD", "CC", "CXX")}
    base.update(GPUS=gpus, PORT=str(port), KV_BYTES=str(kv_bytes), P2P="off", VENV=kvv["MERLIN_VENV"])
    plan = eng["ML"].build("27b", profile, str(MERLIN), base, extra=extra, checkpoint="27b", status="untested")
    env = dict(plan.env)
    # rev 3: -lcuda shim (FlashInfer JIT), system GCC at serve time (GCC 13 was only the install rule), per-port caches
    env.update(LD_PRELOAD=kvv["MERLIN_SHIM"], CUDA_HOME=eng["cuda_home"], VLLM_USE_RUST_FRONTEND="0", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", LIBRARY_PATH=f"{SHIM}:{os.environ.get('LIBRARY_PATH', '')}",
               PATH=f"{kvv['MERLIN_VENV']}/bin:{eng['cuda_home']}/bin:{os.environ.get('PATH', '')}",
               VLLM_CACHE_ROOT=f"{TMP}/merlin-cache", TRITON_CACHE_DIR=f"{TMP}/merlin-triton-{port}",
               FLASHINFER_WORKSPACE_BASE=f"{TMP}/merlin-fi")
    for k in ("CC", "CXX", "CUDAHOSTCXX"):
        env.pop(k, None)
    dp = getattr(eng["ML"], "distributed_ports", None)                   # Merlin's own --instances port scheme
    if dp:
        try:
            ports = dp(port)
            env.update({k: str(v) for k, v in (ports.items() if isinstance(ports, dict) else
                                               zip(("MASTER_PORT", "VLLM_PORT"), ports))})
        except Exception:
            env.update(MASTER_PORT=str(29500 + 10 * (port - 8100)), VLLM_PORT=str(29505 + 10 * (port - 8100)))
    else:
        env.update(MASTER_PORT=str(29500 + 10 * (port - 8100)), VLLM_PORT=str(29505 + 10 * (port - 8100)))
    argv = plan.argv()
    HELP[name] = L.help_check("merlin", argv[:argv.index("serve") + 1], env, argv)
    return L.Server(name, argv, env, port, LOGS, gpus=gpus)


EAGLE_ENG = None


def eagle_install():
    global EAGLE_ENG
    if EAGLE_ENG is not None:
        if EAGLE_ENG.get("failed"):
            raise RuntimeError("EAGLE install failed earlier: " + EAGLE_ENG["failed"])
        return EAGLE_ENG
    if FAKE or EAGLE_PAY is None or EAGLE_DRAFT is None:
        return None
    try:
        pay_root = TMP / "eagle_pay"
        shutil.rmtree(pay_root, ignore_errors=True)
        pay_root.mkdir()
        with tarfile.open(EAGLE_PAY) as tf:
            tf.extractall(pay_root, filter="data")
        pay = pay_root / "eagle_offline"
        rc, txt = run("install_eagle", f"bash {pay}/install_eagle_offline.sh {pay} {TMP}/eagle-off", timeout=2400)
        shutil.rmtree(pay_root, ignore_errors=True)
        if rc or "INSTALL OK" not in txt:
            raise RuntimeError("EAGLE offline install failed (logs/install_eagle.log)")
        env = dict(STOCK["env"])                                           # rev 3: the fork's own PATH / caches
        env.update(PATH=f"{TMP}/eagle-off/bin:{env.get('PATH', '')}", VLLM_CACHE_ROOT=f"{TMP}/eagle-cache",
                   VLLM_USE_V2_MODEL_RUNNER="1", TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4")
        EAGLE_ENG = {"vllm": [f"{TMP}/eagle-off/bin/vllm", "serve"], "env": env, "name": "eagle"}
        return EAGLE_ENG
    except Exception as e:
        EAGLE_ENG = {"failed": f"{type(e).__name__}: {e}"[:300]}
        raise


PROFILER = lambda d: json.dumps({"profiler": "torch", "torch_profiler_dir": str(d), "torch_profiler_with_stack": False,
                                 "ignore_frontend": True})


def stock_flags(model, port, *, tp=4, pp=1, seqs=20, spec=3, kv="fp8_e4m3", backend="FLASHINFER", prefix=True, mem=0.90,
                batched=4096, max_len=MAX_LEN, extra=(), profile_dir=None, spec_json=None):
    f = [str(model), "--tensor-parallel-size", str(tp), "--kv-cache-dtype", kv, "--max-model-len", str(max_len),
         "--max-num-seqs", str(seqs), "--max-num-batched-tokens", str(batched), "--gpu-memory-utilization", str(mem),
         "--disable-custom-all-reduce", "--seed", "20261005", "--host", "127.0.0.1", "--port", str(port),
         "--served-model-name", "q27", "--attention-backend", backend, "--trust-remote-code", "--language-model-only"]
    if pp > 1:
        f += ["--pipeline-parallel-size", str(pp)]
    f += ["--enable-prefix-caching"] if prefix else ["--no-enable-prefix-caching"]
    if spec_json:
        f += ["--speculative-config", spec_json]
    elif spec:
        f += ["--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": int(spec)})]
    if profile_dir:
        f += ["--profiler-config", PROFILER(profile_dir)]
    return f + list(extra)


def stock_server(name, flags, port, env_extra=None, engine=None, gpus="0,1,2,3"):
    eng = engine or STOCK
    env = dict(eng.get("env") or STOCK["env"])
    env.update(env_extra or {})
    if "--profiler-config" in flags:
        env["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"                    # forward ranges for the per-forward graph verdict
    if not FAKE:
        HELP[name] = L.help_check(eng["name"], list(eng["vllm"]), env, flags)
    return L.Server(name, list(eng["vllm"]) + list(flags), env, port, LOGS, gpus=gpus)
save()
'''

CELL3 = r'''# ==================================================================== CELL 3 — distinct ARC prompts per stream (token ids)
from tokenizers import Tokenizer
TOK = Tokenizer.from_file(str(FP8 / "tokenizer.json"))
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


_HEAD = TOK.encode("<|im_start|>user\nFind the transformation rule of each ARC task below and solve the test inputs.\n\n",
                   add_special_tokens=False).ids
_TAIL = TOK.encode("\n<|im_end|>\n<|im_start|>assistant\n<think>\n", add_special_tokens=False).ids
for _t in ("<|im_start|>", "<|im_end|>", "<think>"):             # rev 3: the chat markers are single special tokens
    _id = TOK.token_to_id(_t)
    assert _id is not None and _id in (_HEAD + _TAIL), (_t, _id)
_TASK_IDS = {k: TOK.encode(_task_text(k) + "\n\n", add_special_tokens=False).ids for k in _ids}


def prompt(band, variant):
    """~band tokens of DISTINCT ARC tasks (stream `variant` starts at its own task offset), chat-formatted, thinking on."""
    body, i = [], (variant * 7) % len(_ids)
    need = band - len(_HEAD) - len(_TAIL)
    while len(body) < need:
        body += _TASK_IDS[_ids[i % len(_ids)]]
        i += 1
    return _HEAD + body[:need] + _TAIL


# The 120k band starts at 116k: the earliest-admitted streams keep decoding while the last 120k prompts prefill, and
# 120k + 512 would leave only 10.5k new tokens (the window context is ~116k + decoded tokens ~= 120k; rev 3b).
BANDS = {"2k": 2048, "8k": 8192, "40k": 40000, "90k": 90000, "120k": 116000}
NSTREAM = 24
PROMPTS = {b: [prompt(n, v) for v in range(NSTREAM)] for b, n in BANDS.items()}
MAX_NEW = {b: MAX_LEN - n - 512 for b, n in BANDS.items()}       # a stream never ends inside a window
ORACLE_P = [PROMPTS["2k"][0], PROMPTS["2k"][1], PROMPTS["8k"][2]]
LONG_P = [PROMPTS["40k"][3]]                                      # one long-context greedy row (40k)
TF_P = {"40k": PROMPTS["40k"][5], "90k": PROMPTS["90k"][5]}        # teacher-forced long-context rows (NVFP4 vs FP8 KV)
R["prompts"] = {b: {"n": len(p), "len": len(p[0]), "sha": hashlib.sha256(json.dumps(p[:4]).encode()).hexdigest()[:12],
                    "distinct_first_task": len({tuple(x[len(_HEAD):len(_HEAD) + 50]) for x in p})} for b, p in PROMPTS.items()}
log("prompts", json.dumps(R["prompts"]))
save()
'''

CELL4 = r'''# ==================================================================== CELL 4 — the step runner (budget, GPU gate, fail-closed)
WIN = {"SMOKE": 30.0, "FULL": 120.0}[MODE]                         # 120 s FULL windows (noise)
ROWS = {}            # oracle greedy rows by key (same weights)
TF = {}              # teacher-forced rows by key
NTOK = 128 if MODE == "FULL" else 64


def step(sec, name, est_min, fn):
    """One measured step: skipped when the budget cannot hold it, recorded on any exception, never PASS by default."""
    rec = R["sections"].setdefault(sec, {}).setdefault(name, {"status": "NOT_RUN"})
    R["order"].append(f"{sec}/{name}")
    if ABORT["why"]:
        rec.update(status="SKIPPED_ABORT", why=ABORT["why"])
        save()
        return rec
    if left() < est_min * 60 + 300:
        rec.update(status="SKIPPED_BUDGET", left_min=round(left() / 60, 1), est_min=est_min)
        save()
        return rec
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
    if ABORT["why"] and rec["status"] == "DONE":
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
                                          alive_after_body=s.alive(), rc=None if s.alive() else s.proc.returncode)
            if not s.alive():
                out["status"] = "SERVER_DIED"
                out["servers"][s.name]["log_tail"] = s.log()[-4000:]
        return out
    finally:
        for s in servers:
            s.kill()
        gpus = ",".join(sorted({g for s in servers for g in s.gpus.split(",")}))
        rel = L.wait_gpu_release(180, fake=bool(FAKE), gpus=gpus)
        out["gpu_released"] = rel
        if rel is False:
            ABORT["why"] = f"GPUs {gpus} still held after {[s.name for s in servers]}"
            out["status"] = "FAIL_GPU_RELEASE"


def oracle(port, key=None, refs=()):
    """Greedy rows (2k, 2k, 8k) + one 40k row; saved under `key` (a target-only oracle) or compared with ROWS[ref]."""
    rows = L.greedy(port, ORACLE_P + LONG_P, n_tok=NTOK, timeout=max(120, min(900, left() / 6)))
    if key:
        ROWS[key] = rows
    res = {"garbage": [L.degenerate(r["text"]) for r in rows], "errors": [r["error"] for r in rows],
           "text_head": [r["text"][:120] for r in rows]}
    for ref in refs:
        res["vs_" + ref] = L.compare(ROWS.get(ref), rows) if ROWS.get(ref) else {"status": "INCONCLUSIVE_NO_ORACLE"}
    return res


def tforced(port, key=None, ref=None):
    """Teacher-forced 40k / 90k prompt logprobs (last 256 positions); saved under `key` or compared with TF[ref]."""
    rows = {b: L.teacher_forced(port, ids) for b, ids in TF_P.items()}
    if key:
        TF[key] = rows
    if ref:
        return {b: L.compare_tf((TF.get(ref) or {}).get(b), rows[b]) for b in rows}
    return {b: {"n": (rows[b] or {}).get("n"), "error": (rows[b] or {}).get("error")} for b in rows}


def levels(port, band, lv, **kw):
    out = {}
    for n in lv:
        if ABORT["why"] or left() < WIN + 300:
            out[f"{band}@{n}"] = {"status": "SKIPPED_BUDGET"}
            continue
        try:                             # rev 3: one failing level never loses the others
            out[f"{band}@{n}"] = L.steady(port, PROMPTS[band], n, secs=WIN, max_new=MAX_NEW[band], **kw)
        except Exception as e:
            out[f"{band}@{n}"] = {"status": "ERROR", "error": f"{type(e).__name__}: {e}"[:300]}
        save()
    return out
save()
'''

CELL5 = r'''# ==================================================================== CELL 5 — the plan (dependency order; report = user numbering)
import concurrent.futures as CF
SM = MODE == "SMOKE"
DO_A = (not SM) or PART in ("A", "ALL")          # Merlin, P5, P4
DO_B = (not SM) or PART in ("B", "ALL")          # stock vLLM + EAGLE + PP4
CAP_TP1 = [1, 2, 5, 10, 15, 20]                                  # seqs 4 x (1 + MTP 4) = 20 (V2 runner rounds to x5)
CAP_TP4 = [1, 2, 4, 5, 8, 10, 15, 20, 25, 30, 35, 40, 60, 80, 100, 120]   # seqs 24 x 5 = 120
KV_TP1, KV_TP4 = 8055029760, 16301686784                         # Merlin preflight's L4 (22.5 GiB) pins
P4_FUT = None
TP1_CFG = {"seqs": 4, "capture": CAP_TP1}


def kit(name):
    return str(KITS / name)


# ---- #2 host-staged all-reduce: all four GPUs free, before any server
def s2():
    out = str(OUT / "p5_bench.json")
    Path(out).unlink(missing_ok=True)
    cmd = [STOCK["py"], kit("bench_host_ar.py"), "--so", kit("p5_host_ar.so"), "--out", out,
           "--timeout-s", "900" if SM else "3000"] + (["--quick"] if SM else [])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1000 if SM else 3300, env=STOCK["env"])
        rc, tail = p.returncode, (p.stdout + p.stderr)[-2000:]
    except subprocess.TimeoutExpired as e:
        rc, tail = -9, str(e)[-500:]
    res = json.loads(Path(out).read_text()) if Path(out).is_file() else {}
    return {"rc": rc, "summary": res.get("summary"), "tail": tail,
            "status": "DONE" if rc == 0 else ("CHECK_FAILED" if rc == 1 else "BENCH_ERROR")}


def s4_thread():
    out = str(OUT / "p4_bench.json")
    Path(out).unlink(missing_ok=True)
    cmd = [STOCK["py"], kit("bench_fp8_gemm.py"), "--so", kit("p4_sm89_fp8.so"), "--out", out, "--gpu", "3",
           "--weights", "auto", "--model-dir", str(FP8)] + (["--m", "1,16,80", "--shapes", "gate_up_proj,down_proj"] if SM else [])
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1500 if SM else 3600, env=STOCK["env"],
                           start_new_session=True)
        rc, tail = p.returncode, (p.stdout + p.stderr)[-2000:]
    except subprocess.TimeoutExpired as e:
        rc, tail = -9, str(e)[-500:]
    res = json.loads(Path(out).read_text()) if Path(out).is_file() else {}
    return {"rc": rc, "summary_per_m": res.get("summary_per_m"), "gates": res.get("gates") or res.get("gate"),
            "tail": tail, "status": "DONE" if rc == 0 else "BENCH_ERROR"}


def join_p4():
    """P4 holds GPU 3; join it (bounded) before any step that needs GPU 3."""
    global P4_FUT
    if P4_FUT is None:
        return
    try:
        res = P4_FUT.result(timeout=max(60, min(3600, left() - 600)))
    except Exception as e:
        res = {"status": "ERROR", "error": repr(e)[:300]}
    R["sections"].setdefault("4_sm89_fp8_gemm", {})["p4_bench_gpu3"] = res
    P4_FUT = None
    L.wait_gpu_release(180, fake=bool(FAKE), gpus="3")
    save()


# ---- #1 Merlin TP1 (GPU 0) with #4 P4 on GPU 3 meanwhile; memory fallback; replicas; Merlin TP4
def _tp1_body(ss):
    port = ss[0].port
    o = {"probes": L.chat_probes(port), "oracle": oracle(port, key="merlin")}
    o["steady"] = levels(port, "40k", [2, 4] if SM else [2, 3, 4])
    o["steady"].update(levels(port, "90k", [1, 2]))
    o["steady"].update(levels(port, "120k", [1] if SM else [1, 2]))
    return o


def s1_tp1():
    o = with_servers([merlin_server("merlin_tp1", "27b-tp1-sm89-24g", "0", 8100, KV_TP1, **TP1_CFG)], 2400, _tp1_body)
    if o.get("status") == "LAUNCH_FAILED":           # rev 3: memory edge -> Merlin's own TP1 profile values (seqs 2)
        TP1_CFG.update(seqs=None, capture=None)
        o2 = with_servers([merlin_server("merlin_tp1_fallback", "27b-tp1-sm89-24g", "0", 8100, KV_TP1)], 2400, _tp1_body)
        o2["first_attempt"] = {k: o[k] for k in ("status", "servers") if k in o}
        o2["config"] = "profile defaults (seqs 2, capture [1,2,5,10])"
        return o2
    o["config"] = "seqs 4, capture " + str(CAP_TP1)
    return o


def _tp1_level(band, default):
    rec = R["sections"]["1_tp1x4_pp4"].get("a_merlin_tp1_gpu0") or {}
    st = rec.get("steady") or {}
    ok = [int(k.split("@")[1]) for k, v in st.items() if k.startswith(band + "@") and v.get("status") in ("STEADY", "CAPACITY_LIMITED")]
    return max(ok) if ok else default


def s1_rep(n):
    def f():
        lv40, lv90 = _tp1_level("40k", 2), _tp1_level("90k", 1)
        ss = [merlin_server(f"merlin_tp1_rep{i}", "27b-tp1-sm89-24g", str(i), 8100 + i, KV_TP1, **TP1_CFG) for i in range(n)]

        def body(ss):
            o = {"per_replica_level": {"40k": lv40, "90k": lv90},
                 "replica_greedy": {s.name: L.compare(ROWS.get("merlin")[:1], L.greedy(s.port, ORACLE_P[:1], n_tok=NTOK))
                                    if ROWS.get("merlin") else {"status": "INCONCLUSIVE_NO_ORACLE"} for s in ss}}
            o["40k"] = L.steady_multi([(s.port, PROMPTS["40k"][i * 6:], lv40) for i, s in enumerate(ss)], secs=WIN,
                                      max_new=MAX_NEW["40k"])
            if not SM:
                o["90k"] = L.steady_multi([(s.port, PROMPTS["90k"][i * 6:], lv90) for i, s in enumerate(ss)], secs=WIN,
                                          max_new=MAX_NEW["90k"])
            return o
        return with_servers(ss, 2400, body)
    return f


def s1_tp4():
    def body(ss):
        port = ss[0].port
        o = {"probes": L.chat_probes(port), "oracle": oracle(port, refs=("merlin",))}
        o["steady"] = levels(port, "40k", [8, 16] if SM else [8, 16, 24])
        o["steady"].update(levels(port, "90k", [8] if SM else [8, 12, 16]))
        o["steady"].update(levels(port, "120k", [8] if SM else [8, 12]))
        return o
    o = with_servers([merlin_server("merlin_tp4", "27b-tp4-sm89-24g", "0,1,2,3", 8100, KV_TP4, seqs=24, capture=CAP_TP4)], 2400, body)
    if o.get("status") == "LAUNCH_FAILED":           # the profile's own capture list (up to 40)
        o2 = with_servers([merlin_server("merlin_tp4_fallback", "27b-tp4-sm89-24g", "0,1,2,3", 8100, KV_TP4, seqs=24)], 2400, body)
        o2["first_attempt"] = {k: o[k] for k in ("status", "servers") if k in o}
        return o2
    return o


# ---- #6 MTP OFF / K1 / K3 (FP8, stock vLLM 0.32, seqs 20); K3 = G0 of #3 (FlashInfer, PIECEWISE pinned), profiled
def s6(depth, profile=False, tag=None):
    def f():
        pdir = TMP / f"e23_profile_mtp{depth}_{tag or 'main'}" if profile else None
        extra = ["--compilation-config", json.dumps({"cudagraph_mode": "PIECEWISE"})] if depth == 3 else []
        srv = stock_server(tag or f"fp8_mtp{depth}", stock_flags(FP8, 8200, spec=depth, extra=extra, profile_dir=pdir), 8200)

        def body(ss):
            port = ss[0].port
            o = {"probes": L.chat_probes(port),
                 "oracle": oracle(port, key=("fp8_mtp0" if depth == 0 else ("fp8_g0" if depth == 3 else None)) if not tag else None,
                                  refs=() if depth == 0 and not tag else ("fp8_mtp0",))}
            if depth == 0 and not tag:
                o["teacher_forced"] = tforced(port, key="fp8_mtp0")
            o["steady"] = levels(port, "40k", [16] if SM else [8, 12, 16, 20])
            if not SM and depth in (0, 3):
                o["steady"].update(levels(port, "90k", [8, 12, 20]))
                o["steady"].update(levels(port, "120k", [8, 12]))
            if profile:
                o["profile"] = L.profiled_load(port, PROMPTS["40k"], 16, pdir, max_new=MAX_NEW["40k"])
                shutil.rmtree(pdir, ignore_errors=True)
            return o
        return with_servers([srv], 1800, body)
    return f


def _site():
    return subprocess.run([STOCK["py"], "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                          capture_output=True, text=True).stdout.strip()


def _patch(reverse=False):
    """rev 3: forward = back up flashinfer.py, dry-run, apply (restore the backup on a failed apply); reverse = restore
    the backup by copy and verify its sha (not patch -R)."""
    target = Path(_site()) / "vllm/v1/attention/backends/flashinfer.py"
    bak = TMP / "flashinfer.py.e23_orig"
    if reverse:
        if not bak.is_file():
            return {"applied": False, "error": "no backup"}
        shutil.copy2(bak, target)
        ok = hashlib.sha256(target.read_bytes()).hexdigest() == hashlib.sha256(bak.read_bytes()).hexdigest()
        return {"applied": ok, "restored_sha": hashlib.sha256(target.read_bytes()).hexdigest()[:16]}
    shutil.copy2(target, bak)
    diff = P60532.split("diff --git a/vllm/v1/attention/backends/flashinfer.py")[1]
    Path(TMP / "pr60532_vllm_only.diff").write_text("diff --git a/vllm/v1/attention/backends/flashinfer.py" + diff.split("\ndiff --git")[0] + "\n")
    base = ["patch", "-p1", "-d", _site(), "--batch", "--forward", "--no-backup-if-mismatch", "-i", str(TMP / "pr60532_vllm_only.diff")]
    dry = subprocess.run(base + ["--dry-run"], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    if dry.returncode:
        return {"rc": dry.returncode, "dry_run": (dry.stdout + dry.stderr)[-600:], "applied": False}
    p = subprocess.run(base, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    if p.returncode:
        shutil.copy2(bak, target)
    return {"rc": p.returncode, "out": (p.stdout + p.stderr)[-600:], "applied": p.returncode == 0,
            "orig_sha": hashlib.sha256(bak.read_bytes()).hexdigest()[:16]}


def s3(tag, backend, graph, patch=False):
    def f():
        pre = _patch() if patch and not FAKE else None
        if pre and not pre["applied"]:
            return {"status": "PATCH_FAILED", "patch": pre}
        try:
            pdir = TMP / f"e23_profile_{tag}"
            extra = ["--compilation-config", json.dumps({"cudagraph_mode": graph})] if graph else []
            srv = stock_server(f"fp8_{tag}", stock_flags(FP8, 8200, backend=backend, extra=extra, profile_dir=pdir), 8200)

            def body(ss):
                port = ss[0].port
                o = {"probes": L.chat_probes(port), "oracle": oracle(port, refs=("fp8_mtp0", "fp8_g0")), "patch": pre}
                o["steady"] = levels(port, "40k", [16] if SM else [16, 20])
                if not SM:
                    o["steady"].update(levels(port, "90k", [12]))
                o["profile"] = L.profiled_load(port, PROMPTS["40k"], 16, pdir, max_new=MAX_NEW["40k"])
                shutil.rmtree(pdir, ignore_errors=True)
                return o
            return with_servers([srv], 1800, body)
        finally:
            if pre and pre.get("applied"):
                rv = _patch(reverse=True)
                R["sections"].setdefault("3_full_graph", {})[f"{tag}_patch_revert"] = rv
                if not rv.get("applied"):
                    ABORT["why"] = "PR #60532 revert failed: the stock venv is no longer pristine"
    return f


# ---- #5 NVFP4 KV: greedy + teacher-forced correctness vs FP8-KV target-only, 90k / 120k census, + MTP3 (not on NO_GO)
def s5(spec, tag=None):
    def f():
        prior = (R["sections"].get("5_nvfp4_kv") or {}).get("nvfp4_target_only") or {}
        if spec and prior.get("status") in ("NO_GO_CORRUPT", "LAUNCH_FAILED", "ERROR", "FAIL_GPU_RELEASE", "SKIPPED_BUDGET"):
            return {"status": "SKIPPED_TARGET_ONLY_NO_GO", "target_only_status": prior.get("status")}
        srv = stock_server(tag or f"nvfp4_mtp{spec}", stock_flags(FP8, 8200, spec=spec, kv="nvfp4", seqs=24), 8200)

        def body(ss):
            port = ss[0].port
            o = {"probes": L.chat_probes(port), "oracle": oracle(port, refs=("fp8_mtp0",) if spec == 0 else ("fp8_g0", "fp8_mtp0"))}
            if spec == 0:
                o["teacher_forced_vs_fp8kv"] = tforced(port, ref="fp8_mtp0")
                if (o["oracle"].get("vs_fp8_mtp0") or {}).get("status") == "FAIL_HIGH_MARGIN":
                    o["status"] = "NO_GO_CORRUPT"
                    return o
            o["steady"] = levels(port, "90k", [12] if SM else [12, 16, 20, 24])
            o["steady"].update(levels(port, "120k", [12] if SM else [12, 16, 20]))
            if not SM:
                o["steady"].update(levels(port, "40k", [16, 20]))
            return o
        return with_servers([srv], 1800, body)
    return f


# ---- #4 INT4 control (stock vLLM, RedHat W4A16, MTP3, s20): the INT4 path against Merlin's low-M INT4 engine
def s4_int4():
    srv = stock_server("int4_mtp3_s20", stock_flags(INT4, 8200, spec=3, seqs=20), 8200)

    def body(ss):
        port = ss[0].port
        o = {"probes": L.chat_probes(port), "oracle": oracle(port, key="int4_mtp3")}
        o["steady"] = levels(port, "40k", [8, 16, 20])
        o["steady"].update(levels(port, "90k", [12]))
        return o
    return with_servers([srv], 1800, body)


# ---- #1e PP4 branch (stock vLLM, INT4 RedHat): eager first (#63505's only reported pass), then compiled; stop on garbage
def s1_pp4(eager):
    def f():
        extra = ["--enforce-eager"] if eager else []
        srv = stock_server("int4_pp4_eager" if eager else "int4_pp4_compiled",
                           stock_flags(INT4, 8200, tp=1, pp=4, spec=0, seqs=16, extra=extra), 8200)

        def body(ss):
            port = ss[0].port
            o = {"probes": L.chat_probes(port), "oracle": oracle(port, refs=("int4_mtp3",)),
                 "note": "PP4 uses stock vLLM + RedHat INT4 (Merlin's fork has no PP profile); #63505 open on 0.32"}
            if not o["probes"]["ok"] or any(o["oracle"]["garbage"]) or any(o["oracle"]["errors"]):
                o["status"] = "NO_GO_GARBAGE_OR_ERROR"
                return o
            o["steady"] = levels(port, "40k", [8] if SM else [8, 16])
            return o
        return with_servers([srv], 1800, body)
    return f


# ---- EAGLE-4 feasibility: a BF16-KV FP8-weight target-only oracle first (same KV dtype as tree verify), then EAGLE
def s7_kvauto_oracle():
    srv = stock_server("fp8_kvauto_mtp0", stock_flags(FP8, 8300, spec=0, kv="auto", seqs=16), 8300)
    return with_servers([srv], 1800, lambda ss: {"oracle": oracle(ss[0].port, key="fp8_kvauto_mtp0")})


def s7_eagle():
    eng = eagle_install()
    if eng is None:
        return {"status": "SKIPPED_REQUIRED_ARTIFACT", "why": "EAGLE env / drafter not attached (or fake mode)"}
    spec = json.dumps({"method": "eagle_harness", "model": str(EAGLE_DRAFT),
                       "method_config": {"mode": "static", "node_top_k": 4, "tree_depth": 14, "max_verify_width": 16}})
    fl = stock_flags(FP8, 8300, seqs=16, kv="auto", backend="CUSTOM", prefix=False, spec_json=spec)
    srv = stock_server("eagle4_fp8", fl, 8300, engine=eng)

    def body(ss):
        port = ss[0].port
        o = {"probes": L.chat_probes(port), "oracle": oracle(port, refs=("fp8_kvauto_mtp0", "fp8_mtp0"))}
        o["steady"] = levels(port, "8k", [8, 16])
        o["steady"].update(levels(port, "40k", [8]))
        return o
    return with_servers([srv], 2400, body)


PREF = []
if DO_A and MERLIN:
    PREF.append(prefetch_bg([MERLIN], "merlin"))
if DO_B:
    PREF.append(prefetch_bg([FP8] + ([INT4] if INT4 and not SM else []), "fp8_int4"))

# ---- run order. SMOKE A: P5, Merlin TP1 (+P4), Merlin TP4. SMOKE B: FP8 mtp0 -> mtp3/G0 -> G3 -> NVFP4 -> EAGLE -> PP4.
# FULL: P5, Merlin block, section 6, section 3, section 5, INT4 control, A/A repeats (time left), PP4 last (#63505).
if DO_A:
    step("2_host_allreduce", "p5_bench", 12 if SM else 30, s2)
    P4_FUT = CF.ThreadPoolExecutor(1).submit(s4_thread)
    step("1_tp1x4_pp4", "a_merlin_tp1_gpu0", 45 if SM else 75, s1_tp1)
    join_p4()
    if not SM:
        step("1_tp1x4_pp4", "b_merlin_tp1_x2", 30, s1_rep(2))
        step("1_tp1x4_pp4", "c_merlin_tp1_x4", 35, s1_rep(4))
    step("1_tp1x4_pp4", "d_merlin_tp4", 40 if SM else 60, s1_tp4)
if DO_B:
    step("6_mtp_depth", "mtp0_target_only", 20 if SM else 50, s6(0))
    if not SM:
        step("6_mtp_depth", "mtp1", 35, s6(1))
    step("6_mtp_depth", "mtp3_eq_G0_piecewise", 25 if SM else 60, s6(3, profile=True))
    if not SM:
        step("3_full_graph", "G1_triton_full_and_piecewise", 35, s3("G1", "TRITON_ATTN", "FULL_AND_PIECEWISE"))
        step("3_full_graph", "G2_triton_full_decode_only", 35, s3("G2", "TRITON_ATTN", "FULL_DECODE_ONLY"))
    step("3_full_graph", "G3_flashinfer_pr60532", 20 if SM else 40, s3("G3", "FLASHINFER", "FULL_AND_PIECEWISE", patch=True))
    step("5_nvfp4_kv", "nvfp4_target_only", 25 if SM else 55, s5(0))
    if not SM:
        step("5_nvfp4_kv", "nvfp4_mtp3", 45, s5(3))
        step("4_sm89_fp8_gemm", "int4_mtp3_s20_control", 35, s4_int4)
    if SM or os.environ.get("E23_EAGLE_FULL") == "1":
        step("7_eagle4", "fp8_kvauto_target_only_oracle", 15, s7_kvauto_oracle)
        step("7_eagle4", "eagle4_fp8_tp4", 30, s7_eagle)
if not SM:      # A/A repeats of the decisive comparisons while 70 min stay reserved for PP4
    for i in range(1, 4):
        if left() < (70 + 50) * 60:
            break
        step("8_aa_repeats", f"mtp3_control_rep{i}", 50, s6(3, tag=f"fp8_mtp3_rep{i}"))
        step("8_aa_repeats", f"merlin_tp1_x4_rep{i}", 35, s1_rep(4))
        step("8_aa_repeats", f"nvfp4_mtp3_rep{i}", 45, s5(3, tag=f"nvfp4_mtp3_rep{i}"))
if DO_B:
    step("1_tp1x4_pp4", "e_pp4_int4_eager", 20 if SM else 30, s1_pp4(True))
    step("1_tp1x4_pp4", "f_pp4_int4_compiled", 20 if SM else 30, s1_pp4(False))
'''

CELL6 = r'''# ==================================================================== CELL 6 — report
def _flat(sec):
    rows = []
    for name, rec in (R["sections"].get(sec) or {}).items():
        if not isinstance(rec, dict):
            continue
        for k, v in (rec.get("steady") or {}).items():
            rows.append([sec, name, k, v.get("status"), v.get("tok_s"), v.get("running_mean"), v.get("mean_acceptance_length"),
                         v.get("step_ms"), v.get("kv_usage_max"), v.get("preemptions")])
        for k in ("40k", "90k"):
            if isinstance(rec.get(k), dict) and "replicas" in rec[k]:
                rows.append([sec, name, k + " aggregate", rec[k]["status"], rec[k].get("tok_s"), rec[k].get("running_mean"),
                             None, None, None, None])
    return rows


lines = ["| section | step | window | status | tok/s | running | accept len | step ms | kv max | preempt |",
         "|---|---|---|---|---|---|---|---|---|---|"]
for sec in sorted(R["sections"]):
    for r_ in _flat(sec):
        lines.append("| " + " | ".join("-" if x is None else str(x) for x in r_) + " |")
st = ["| section | step | status | windows steady/all | oracle | probes | KV tokens | ready s | graph capture | trace verdict |",
      "|---|---|---|---|---|---|---|---|---|---|"]
for sec in sorted(R["sections"]):
    for name, rec in R["sections"][sec].items():
        if not isinstance(rec, dict):
            continue
        o = rec.get("oracle") or {}
        vs = ", ".join(f"{k[3:]}={v.get('status')}" for k, v in o.items() if k.startswith("vs_") and isinstance(v, dict))
        sv = next(iter((rec.get("servers") or {}).values()), {}) if rec.get("servers") else {}
        tr_ = (rec.get("profile") or {}).get("trace") or {}
        gv = tr_.get("graph_verdict") or tr_.get("error") or "-"
        if tr_.get("graph_verdict"):
            gv += f" (g/fw {tr_.get('per_forward_median_graph_launches')}, k/fw {tr_.get('per_forward_median_kernel_launches')})"
        wins = list((rec.get("steady") or {}).values())
        ws = f"{sum(1 for w in wins if w.get('status') in ('STEADY', 'CAPACITY_LIMITED'))}/{len(wins)}" if wins else "-"
        st.append(f"| {sec} | {name} | {rec.get('status')} | {ws} | {vs or ('saved' if o else '-')} | "
                  f"{(rec.get('probes') or {}).get('ok', '-')} | {sv.get('kv_tokens', '-')} | {sv.get('ready_sec', '-')} | "
                  f"{','.join(sv.get('graph_capture_modes') or []) or '-'} | {gv} |")
R["help_check"] = HELP
R["report_md"] = "\n".join(st) + "\n\n" + "\n".join(lines)
R["minutes"] = round((time.time() - T_START) / 60, 1)
save()
print(R["report_md"])
print("E23", MODE, PART, "done in", R["minutes"], "min | abort:", ABORT["why"])
'''

VARIANTS = {"SMOKE_A": ("SMOKE", "A"), "SMOKE_B": ("SMOKE", "B"), "FULL": ("FULL", "ALL")}
for variant, (mode, part) in VARIANTS.items():
    c1 = CELL1.replace("__MODE__", mode).replace("__PART__", part).replace("__LAB__", repr(LAB)).replace("__P60532__", repr(P60532))
    md = MD0.replace("__VARIANT__", variant).replace("__MODE__", mode).replace("__PART__", part)
    guarded = [("code", "if not RERUN:\n" + textwrap.indent(c, "    ")) for c in (CELL2, CELL3, CELL4, CELL5, CELL6)]
    cells = [("markdown", md), ("code", c1)] + guarded
    for kind, src in cells:
        if kind == "code":
            compile(src, "<cell>", "exec")
    nb = {"cells": [{"cell_type": k, "metadata": {}, "source": s.splitlines(True), **({"execution_count": None, "outputs": []} if k == "code" else {})}
                    for k, s in cells],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1)
    out = HERE.parent / f"Q27-XHIGH-E23_SIX_LEVERS_{variant}.ipynb"
    out.write_text(data, encoding="utf-8", newline="\n")
    print(out.name, hashlib.sha256(data.encode()).hexdigest()[:8], f"{len(data) / 1024:.0f} kB")
old = HERE.parent / "Q27-XHIGH-E23_SIX_LEVERS.ipynb"
if old.exists():
    old.unlink()          # rev 3: the three variants replace the single rev 1 / 2 file (never run on Kaggle)
