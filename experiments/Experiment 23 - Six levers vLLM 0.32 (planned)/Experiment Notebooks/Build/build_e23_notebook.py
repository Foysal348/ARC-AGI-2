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

MD0 = """# E23 — six levers on 4×L4 (vLLM 0.32, Qwen3.8-27B), target 1500+ tok/s · rev 3f · __VARIANT__

The user's six experiments, each with only its own test sequence (DESIGN.md), at the production context
(thinking up to 120k: `--max-model-len 131072`, bands 2k / 8k / 40k / 90k / 120k):
1. TP1×4 / PP4 (Merlin W4A16 TP1 replicas; PP4 branch on stock vLLM + RedHat INT4, eager + compiled)
2. Four-rank host-staged all-reduce (P5 bench)
3. FULL-graph speculative verification (G0 FlashInfer PIECEWISE, G1/G2 TRITON FULL, G3 FlashInfer + #60532) with a graph verdict
4. SM89 FP8 GEMM (#61224 CUTLASS vs Marlin, P4 bench) + an INT4 TP4 control row
5. NVFP4 KV: correctness (greedy + teacher-forced 40k / 90k on separate short launches), 90k / 120k residency, + MTP3
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
BUDGET = int(os.environ.get("E23_BUDGET_SEC") or {"SMOKE": {"A": 135, "B": 215, "ALL": 345}[PART] * 60, "FULL": 690 * 60}[MODE])
HARD_STOP = T_START + BUDGET
MAX_LEN = 131072                                                  # user 2026-10-29: thinking up to 120k
OUT = Path(os.environ.get("E23_OUT", "/kaggle/working/e23"))
OUT.mkdir(parents=True, exist_ok=True)
LOGS = OUT / "logs"
LOGS.mkdir(exist_ok=True)
TMP = Path(os.environ.get("E23_TMP", "/tmp"))
INP = Path(os.environ.get("E23_INPUT", "/kaggle/input"))
R = {"mode": MODE, "part": PART, "budget_min": BUDGET // 60, "max_len": MAX_LEN, "rev": "3f",
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
    if MODE == "SMOKE" and PART in ("A", "ALL") and (not MERLIN_OK or MERLIN_PAY is None):
        save()                   # rev 3c: SMOKE A exists for Merlin; never a silent rc 0 with only P4 / P5 data
        raise RuntimeError("SMOKE A needs the Merlin checkpoint (qwen3-8-27b-w4a16-merlin/Transformers/default/1) and env "
                           "(e23-merlin-env/Other/v1/1): " + json.dumps({"MERLIN": str(MERLIN), "MERLIN_PAY": str(MERLIN_PAY),
                                                                         "merlin_files_ok": MERLIN_OK}))
    km = json.loads((KITS / "q27_e23_kernels_manifest.json").read_text())       # rev 3: the kit files are what we built
    R["inputs"]["kits_sha_ok"] = {n: (KITS / n).is_file() and hashlib.sha256((KITS / n).read_bytes()).hexdigest() == v.get("sha256")
                                  for n, v in km.items() if isinstance(v, dict) and n.endswith((".so", ".py"))}
    _need = ("p4_sm89_fp8.so", "p5_host_ar.so", "bench_fp8_gemm.py", "bench_host_ar.py")
    _bad = sorted(n for n in _need if R["inputs"]["kits_sha_ok"].get(n) is not True)
    if _bad and (MODE == "FULL" or PART in ("A", "ALL")):   # rev 3d (#9) / 3e: only the runs that use P4 / P5
        R["errors"].append("KERNEL_KIT_HASH_FAILED: " + repr(_bad))
        save()
        raise RuntimeError("KERNEL_KIT_HASH_FAILED: " + repr(_bad))
    R["identity"] = {k: L.identity(v) for k, v in dict(fp8=FP8, int4=INT4, merlin=MERLIN, eagle_draft=EAGLE_DRAFT).items() if v}
    ABORT = {"why": None}
    NVSMI_OK = bool(L.gpu_mem()) and not FAKE                       # rev 3c: a later nvidia-smi failure is not "released"
    R["setup"]["gpus_at_start"] = L.gpu_mem()
    if not FAKE and len(R["setup"]["gpus_at_start"] or []) != 4:          # rev 3d (#8): a hard stop, not a note
        R["errors"].append(f"expected 4 GPUs, nvidia-smi shows {R['setup']['gpus_at_start']}")
        save()
        raise RuntimeError("GPU_PREFLIGHT_FAILED: " + repr(R["setup"]["gpus_at_start"]))
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
               VLLM_NO_USAGE_STATS="1", VLLM_CACHE_ROOT=f"{TMP}/vllm-cache", VLLM_ALLOW_LONG_MAX_MODEL_LEN="1",
               CUDA_DEVICE_ORDER="PCI_BUS_ID")
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


MERLIN_CC = {"mode": "gcc13"}       # rev 3c: Merlin README:229 / install-env.sh: FlashInfer JIT at first start needs GCC >= 13


def merlin_server(name, profile, gpus, port, kv_bytes, seqs=None, capture=None):
    """Merlin's own launch.build() (pure: the profile's flags + env, no host refusal), our max len / port last, and
    seqs / capture sizes only when given (None = the profile's own, the TP1 memory fallback). Compiler: the payload's
    GCC 13 (+ its libstdc++ when present) unless a GCC 13 compile failed earlier (then the system GCC 11)."""
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
    # -lcuda shim (FlashInfer JIT), per-port caches; compiler set below (rev 3c: GCC 13 first, see MERLIN_CC)
    env.update(LD_PRELOAD=kvv["MERLIN_SHIM"], CUDA_HOME=eng["cuda_home"], VLLM_USE_RUST_FRONTEND="0", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", LIBRARY_PATH=f"{SHIM}:{os.environ.get('LIBRARY_PATH', '')}",
               PATH=f"{kvv['MERLIN_VENV']}/bin:{eng['cuda_home']}/bin:{os.environ.get('PATH', '')}",
               VLLM_CACHE_ROOT=f"{TMP}/merlin-cache", TRITON_CACHE_DIR=f"{TMP}/merlin-triton-{port}",
               FLASHINFER_WORKSPACE_BASE=f"{TMP}/merlin-fi", CUDA_DEVICE_ORDER="PCI_BUS_ID")
    for k in ("CC", "CXX", "CUDAHOSTCXX"):
        env.pop(k, None)
    if MERLIN_CC["mode"] == "gcc13" and kvv.get("MERLIN_CC") and kvv.get("MERLIN_CXX"):
        env.update(CC=kvv["MERLIN_CC"], CXX=kvv["MERLIN_CXX"], CUDAHOSTCXX=kvv["MERLIN_CXX"])
        lib13 = Path(kvv["MERLIN_CC"]).parents[1] / "lib"
        if (lib13 / "libstdc++.so.6").exists():
            env["LD_LIBRARY_PATH"] = f"{lib13}:{env.get('LD_LIBRARY_PATH', '')}"
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
WIN = float(os.environ.get("E23_WIN_SEC") or {"SMOKE": 30.0, "FULL": 120.0}[MODE])   # 120 s FULL windows (noise)
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
'''

CELL5 = r'''# ==================================================================== CELL 5 — the plan (dependency order; report = user numbering)
import concurrent.futures as CF
import signal
SM = MODE == "SMOKE"
DO_A = (not SM) or PART in ("A", "ALL")          # Merlin, P5, P4
DO_B = (not SM) or PART in ("B", "ALL")          # stock vLLM + EAGLE + PP4
CAP_TP1 = [1, 2, 5, 10, 15, 20]                                  # seqs 4 x (1 + MTP 4) = 20 (V2 runner rounds to x5)
CAP_TP4 = [1, 2, 4, 5, 8, 10, 15, 20, 25, 30, 35, 40, 60, 80, 100, 120]   # seqs 24 x 5 = 120
KV_TP1, KV_TP4 = 8055029760, 16301686784                         # Merlin preflight's L4 (22.5 GiB) pins
P4_FUT = None
P4_PROC = {"p": None}
TP1_CFG = {"seqs": 4, "capture": CAP_TP1}
TP1_KW = {"prefill_tok_s": 250.0}                                # rev 3c: one L4 prefills ~250-400 tok/s (not TP4's 700)


def kit(name):
    return str(KITS / name)


# ---- rev 3d (four GPT audits, REV3C_GPT4_PRIORITY_FIX_LIST #3): every arm's status reflects its correctness evidence.
# A request error / garbage / missing or non-clean comparison is never DONE; _done() (A/A) needs DONE.
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


# rev 3f (GPT rev 3e audit): Xid by recovery class (NVIDIA XID catalog). These need a GPU reset (driver / bus / GSP /
# uncontained ECC / NVLink / channel faults): the campaign stops. Application-class faults (an illegal memory access,
# Xid 13 / 31 / 43 ...) are recovered by the process restart every arm already does, and are continued only after the
# GPU health probe passes (all 4 GPUs, no reset pending, a clean CUDA matmul on each).
XID_RESET = {48, 61, 62, 63, 64, 68, 69, 74, 79, 92, 94, 95, 109, 110, 119, 120, 121, 136, 140, 143, 144, 145, 146, 147,
             148, 149, 150, 151, 154, 155, 156, 157}
HEALTH_CODE = ("import torch\nn = torch.cuda.device_count()\nok = n == 4\nfor i in range(n):\n"
               "    a = torch.ones(1024, 1024, device=f'cuda:{i}')\n"
               "    ok = ok and float((a @ a).sum().item()) == float(1024 ** 3)\n"
               "torch.cuda.synchronize()\nprint('HEALTH_OK' if ok else 'HEALTH_BAD', n)\n")


def gpu_health():
    """rev 3f: nvidia-smi sees 4 GPUs with no reset pending, then a fresh process runs a checked matmul on each."""
    if FAKE:
        return {"ok": True, "fake": True}
    out = {}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,reset_status.reset_required,memory.used", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=60)
        rows = [l for l in q.stdout.strip().splitlines() if l.strip()]
        out["nvidia_smi_rows"] = rows
        out["reset_required"] = any("yes" in l.lower() for l in rows) if q.returncode == 0 else None
        if q.returncode != 0:              # an old driver without reset_status: fall back to the plain query
            out["reset_required"] = None
            rows = [l for l in subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], capture_output=True,
                                              text=True, timeout=60).stdout.strip().splitlines() if l.strip()]
        out["gpus"] = len(rows)
        h = subprocess.run([STOCK["py"], "-c", HEALTH_CODE], capture_output=True, text=True, timeout=300, env=STOCK["env"])
        out["cuda_probe"] = (h.stdout + h.stderr)[-300:]
        out["ok"] = out["gpus"] == 4 and out["reset_required"] is not True and h.returncode == 0 and "HEALTH_OK" in h.stdout
    except Exception as e:
        out.update(ok=False, error=f"{type(e).__name__}: {e}"[:200])
    return out


def fault_gate(texts, arm):
    """rev 3f: classify GPU fault lines in `texts` (server logs, bench output). Returns None (no fault), 'CONTINUE'
    (application fault + healthy GPUs: this arm is NO_GO, the run goes on) or 'ABORT' (a reset-class Xid, or the GPUs
    are not healthy after the fault: the campaign stops)."""
    lines = [l for t in texts for l in (t or "").splitlines() if XID.search(l)]
    if not lines:
        return None
    # "NVRM: Xid (PCI:0000:00:04): 79, pid=..." (the PCI address comes first) or a bare "Xid 79"
    nums = {int(x) for l in lines for x in re.findall(r"Xid \([^)]*\):\s*(\d+)|\bXid\s+(\d+)\b", l) for x in x if x}
    rec = {"arm": arm, "lines": lines[:5], "xid": sorted(nums)}
    if nums & XID_RESET:
        rec["class"] = "RESET_REQUIRED"
        ABORT["why"] = f"GPU fault needing a reset (Xid {sorted(nums & XID_RESET)}) in {arm}"
    else:
        rec["health"] = gpu_health()
        rec["class"] = "APP_RESTART" if rec["health"].get("ok") else "UNHEALTHY_AFTER_FAULT"
        if not rec["health"].get("ok"):
            ABORT["why"] = f"GPUs unhealthy after a GPU fault in {arm}"
    R["setup"].setdefault("gpu_faults", []).append(rec)
    return "ABORT" if ABORT["why"] and rec["class"] != "APP_RESTART" else "CONTINUE"


def taint_check(o):
    """rev 3e/3f: a GPU fault line in a server log (with_servers records `xid_lines`, at launch too) marks THIS arm
    NO_GO_GPU_FAULT; a reset-class Xid or unhealthy GPUs afterwards abort the campaign (fault_gate); an application-class
    fault with healthy GPUs lets the run go on, every later record labelled `after_gpu_fault`."""
    texts = {n: "\n".join((v or {}).get("xid_lines") or []) for n, v in (o.get("servers") or {}).items() if (v or {}).get("xid")}
    prior = list(R["setup"].get("gpu_fault_arms") or [])
    if prior:
        o["after_gpu_fault"] = prior
    if texts:
        R["setup"]["gpu_fault_arms"] = prior + list(texts)
        o["gpu_fault"] = list(texts)
        o["gpu_fault_class"] = fault_gate(list(texts.values()), ",".join(texts))
        o["correctness"] = "NO_GO_GPU_FAULT"
        if o.get("status") in (None, "DONE"):
            o["status"] = "NO_GO_GPU_FAULT"
    return o


def _killpg(p):
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except Exception:
        pass


# ---- #2 host-staged all-reduce: all four GPUs free, before any server
def s2():
    out = str(OUT / "p5_bench.json")
    Path(out).unlink(missing_ok=True)
    cmd = [STOCK["py"], kit("bench_host_ar.py"), "--so", kit("p5_host_ar.so"), "--out", out,
           "--timeout-s", "900" if SM else "3000"] + (["--quick"] if SM else [])
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=STOCK["env"],
                         start_new_session=True)
    try:
        so, _ = p.communicate(timeout=1000 if SM else 3300)
        rc, tail = p.returncode, (so or "")[-2000:]
    except subprocess.TimeoutExpired:
        _killpg(p)                                    # rev 3c: the bench's children die with it
        so, _ = p.communicate()
        rc, tail = -9, (so or "")[-1000:]
    res = json.loads(Path(out).read_text()) if Path(out).is_file() else {}
    fg = fault_gate([so or ""], "p5_bench")          # rev 3f: the standalone benches go through the same fault gate
    if fg:
        return {"rc": rc, "summary": res.get("summary"), "tail": tail, "status": "NO_GO_GPU_FAULT", "gpu_fault_class": fg}
    rel = L.wait_gpu_release(180, fake=bool(FAKE), gpus="0,1,2,3")
    if rel is None and NVSMI_OK:
        rel = False
    o = {"rc": rc, "summary": res.get("summary"), "tail": tail, "gpu_released": rel}
    if rel is False:                  # rev 3d (#7): Merlin must never start on GPUs P5 still holds
        ABORT["why"] = f"P5 GPU release unverified: {rel}"
        return {**o, "status": "FAIL_GPU_RELEASE"}
    passed = rc == 0 and ((res.get("summary") or {}).get("all_checks_pass") is True or bool(FAKE))
    return {**o, "status": "DONE" if passed else ("CHECK_FAILED" if rc in (0, 1) else "BENCH_ERROR")}


def s4_thread():
    out = str(OUT / "p4_bench.json")
    Path(out).unlink(missing_ok=True)
    per_shape = 420 if SM else 600
    cmd = [STOCK["py"], kit("bench_fp8_gemm.py"), "--so", kit("p4_sm89_fp8.so"), "--out", out, "--gpu", "3",
           "--weights", "real", "--model-dir", str(FP8), "--timeout-per-shape", str(per_shape)] + \
          (["--m", "1,16,80", "--shapes", "gate_up_proj,down_proj"] if SM else [])
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=STOCK["env"],
                         start_new_session=True)
    P4_PROC["p"] = p
    try:
        so, _ = p.communicate(timeout=2 * per_shape + 300 if SM else 5 * per_shape + 300)
        rc, tail = p.returncode, (so or "")[-2000:]
    except subprocess.TimeoutExpired:
        _killpg(p)                                    # rev 3c: per-shape children hold GPU 3 otherwise
        so, _ = p.communicate()
        rc, tail = -9, (so or "")[-1000:]
    res = json.loads(Path(out).read_text()) if Path(out).is_file() else {}
    fg = fault_gate([so or ""], "p4_bench_gpu3")     # rev 3f: same fault gate as the servers
    if fg:
        return {"rc": rc, "tail": tail, "status": "NO_GO_GPU_FAULT", "gpu_fault_class": fg}
    kinds = {n: ((v or {}).get("weights_source") or {}).get("kind") for n, v in (res.get("shapes") or {}).items()}
    synth = [n for n, k in kinds.items() if k == "synthetic"]
    no_result = [n for n, k in kinds.items() if k is None]
    o = {"rc": rc, "summary_per_m": res.get("summary_per_m"), "gates": res.get("gates") or res.get("gate"), "tail": tail,
         "weights_kind": {n: ((v or {}).get("weights_source") or {}).get("kind") for n, v in (res.get("shapes") or {}).items()}}
    if synth:                          # rev 3d (#10): only the real Qwen3.8-27B FP8 weights + scales count
        return {**o, "status": "CHECK_FAILED", "why": "SYNTHETIC_WEIGHTS_NOT_ALLOWED", "shapes": synth}
    if no_result:                      # rev 3e: a crashed / timed-out shape is not "synthetic"
        return {**o, "status": "CHECK_FAILED" if rc in (0, 1) else "BENCH_ERROR", "why": "SHAPE_NO_RESULT", "shapes": no_result}
    passed = rc == 0 and res.get("all_ok") is True
    return {**o, "status": "DONE" if passed else ("CHECK_FAILED" if rc in (0, 1) else "BENCH_ERROR")}


def join_p4():
    """P4 holds GPU 3; join it (bounded) before any step that needs GPU 3; a GPU 3 that stays held aborts the lab."""
    global P4_FUT
    if P4_FUT is None:
        return
    try:
        res = P4_FUT.result(timeout=max(60, min(3600, left() - 600)))
    except Exception as e:
        res = {"status": "ERROR", "error": repr(e)[:300]}
        if P4_PROC["p"] is not None:
            _killpg(P4_PROC["p"])
    rel = L.wait_gpu_release(300, fake=bool(FAKE), gpus="3")
    if rel is None and NVSMI_OK:
        rel = False
    res["gpu3_released"] = rel
    R["sections"].setdefault("4_sm89_fp8_gemm", {})["p4_bench_gpu3"] = res
    P4_FUT = None
    if rel is False:
        ABORT["why"] = "GPU 3 still held after the P4 bench"
    save()


# ---- Merlin launch with the two recoveries the audits asked for: a GCC 13 compile failure -> the system GCC once;
# an OOM (launch or mid-body) on TP1 -> Merlin's own TP1 profile values. Never after the hard stop.
def _logs_of(o):
    return " ".join((v or {}).get("log_tail", "") for v in (o.get("servers") or {}).values())


def merlin_run(make, timeout, body, tp1=False):
    o = with_servers(make(), timeout, body)
    tries = []
    for _ in range(2):
        if ABORT["why"] or o.get("status") not in ("LAUNCH_FAILED", "SERVER_DIED"):
            break
        logs = _logs_of(o)
        if COMPILE_ERR.search(logs) and MERLIN_CC["mode"] == "gcc13":
            MERLIN_CC["mode"] = "system"
            why = "gcc13_compile_error -> system GCC"
        elif tp1 and (OOM_ERR.search(logs) or o.get("status") == "LAUNCH_FAILED") and TP1_CFG["seqs"] is not None:
            TP1_CFG.update(seqs=None, capture=None)
            why = "TP1 OOM / launch failure -> profile defaults (seqs 2, capture [1,2,5,10])"
        else:
            break
        tries.append({"why": why, "status": o.get("status"), "servers": o.get("servers")})
        o = with_servers(make(), timeout, body)
    if tries:
        o["earlier_attempts"] = tries
    o["merlin_compiler"] = MERLIN_CC["mode"]
    return taint_check(o)


# ---- #1 Merlin TP1 (GPU 0) with #4 P4 on GPU 3 meanwhile; replicas; Merlin TP4
def _tp1_body(ss):
    s = ss[0]
    port = s.port
    o = {"probes": L.chat_probes(port), "oracle": oracle(port, key="merlin")}
    g90 = L.greedy(port, [PROMPTS["90k"][4]], n_tok=NTOK, timeout=max(300, min(1800, left() / 6)))
    if all(not r.get("error") and r.get("tokens") and not L.degenerate(r.get("text")) for r in g90):
        ROWS["merlin_90k"] = g90                                     # rev 3e: clean references only
    o["greedy_90k"] = {"garbage": [L.degenerate(r["text"]) for r in g90], "errors": [r["error"] for r in g90]}
    o["steady"] = levels(port, "40k", [2, 4] if SM else [2, 3, 4], server=s, **TP1_KW)
    o["steady"].update(levels(port, "90k", [1, 2] if SM else [1, 2, 3, 4], server=s, **TP1_KW))
    o["steady"].update(levels(port, "120k", [1] if SM else [1, 2], server=s, **TP1_KW))
    if s.alive():                    # Merlin's fork budgets prompt-logprob memory; still last (nothing measured after it)
        o["teacher_forced"] = tforced(port, key="merlin")
    g90 = {"status": "EXACT"} if not (any(o["greedy_90k"]["garbage"]) or any(o["greedy_90k"]["errors"])) else \
        {"status": "FAIL_HIGH_MARGIN" if any(o["greedy_90k"]["garbage"]) else "INCONCLUSIVE"}
    return gated(o, extra=(g90,))


def s1_tp1():
    g = merlin_sha_gate()
    if g:
        return g
    o = merlin_run(lambda: [merlin_server("merlin_tp1", "27b-tp1-sm89-24g", "0", 8100, KV_TP1, **TP1_CFG)], 2400,
                   _tp1_body, tp1=True)
    o["config"] = ("profile defaults (seqs 2, capture [1,2,5,10])" if TP1_CFG["seqs"] is None
                   else "seqs 4, capture " + str(CAP_TP1))
    return o


def _tp1_level(band, default):
    rec = R["sections"]["1_tp1x4_pp4"].get("a_merlin_tp1_gpu0") or {}
    st = rec.get("steady") or {}
    ok = []
    for k, v in st.items():
        if k.startswith(band + "@") and v.get("status") in ("STEADY", "CAPACITY_LIMITED"):
            ok.append(int(k.split("@")[1]) if v["status"] == "STEADY" else int(v.get("running_max") or 0))
    ok = [x for x in ok if x > 0]
    return max(ok) if ok else default


def s1_rep(n, bands=None):
    def f():
        g = merlin_sha_gate()
        if g:
            return g
        lv40, lv90 = _tp1_level("40k", 2), _tp1_level("90k", 1)

        def make():
            return [merlin_server(f"merlin_tp1_rep{i}", "27b-tp1-sm89-24g", str(i), 8100 + i, KV_TP1, **TP1_CFG)
                    for i in range(n)]

        def body(ss):
            o = {"per_replica_level": {"40k": lv40, "90k": lv90},
                 "replica_greedy": {s.name: L.compare(ROWS.get("merlin")[:1], L.greedy(s.port, ORACLE_P[:1], n_tok=NTOK))
                                    if ROWS.get("merlin") else {"status": "INCONCLUSIVE_NO_ORACLE"} for s in ss}}
            o["40k"] = L.steady_multi([(s.port, PROMPTS["40k"][i * 6:], lv40) for i, s in enumerate(ss)], secs=WIN,
                                      max_new=MAX_NEW["40k"], **TP1_KW)
            if "90k" in (bands or ("40k", "90k")):
                o["90k"] = L.steady_multi([(s.port, PROMPTS["90k"][i * 6:], lv90) for i, s in enumerate(ss)], secs=WIN,
                                          max_new=MAX_NEW["90k"], **TP1_KW)
            return gated(o, extra=tuple(o["replica_greedy"].values()))
        return merlin_run(make, 2400, body)
    return f


def s1_tp4():
    g = merlin_sha_gate()
    if g:
        return g

    def body(ss):
        s = ss[0]
        port = s.port
        o = {"probes": L.chat_probes(port), "oracle": oracle(port, refs=("merlin",))}
        g90 = L.greedy(port, [PROMPTS["90k"][4]], n_tok=NTOK, timeout=max(300, min(1800, left() / 6)))
        o["greedy_90k_vs_tp1"] = L.compare(ROWS.get("merlin_90k"), g90) if ROWS.get("merlin_90k") else \
            {"status": "INCONCLUSIVE_NO_ORACLE"}
        o["steady"] = levels(port, "40k", [8, 16] if SM else [8, 16, 24], server=s)
        o["steady"].update(levels(port, "90k", [8] if SM else [8, 16], server=s))
        o["steady"].update(levels(port, "120k", [8] if SM else [8, 12], server=s))
        if s.alive():
            o["teacher_forced_vs_tp1"] = tforced(port, ref="merlin")
            o["teacher_forced_vs_tp1_note"] = "info only: same engine, same weights (TP4 vs TP1 numerics)"
        return gated(o, need=("merlin",), extra=(o["greedy_90k_vs_tp1"],))
    o = merlin_run(lambda: [merlin_server("merlin_tp4", "27b-tp4-sm89-24g", "0,1,2,3", 8100, KV_TP4, seqs=24,
                                          capture=CAP_TP4)], 2400, body)
    if o.get("status") == "LAUNCH_FAILED" and not ABORT["why"]:      # the profile's own capture list (validated to 40)
        o2 = merlin_run(lambda: [merlin_server("merlin_tp4_fallback", "27b-tp4-sm89-24g", "0,1,2,3", 8100, KV_TP4,
                                               seqs=8)], 2400, body)
        o2["first_attempt"] = {k: o[k] for k in ("status", "servers") if k in o}
        o2["config"] = "fallback seqs 8 with the profile's capture list (graphs cover every batch)"
        return o2
    return o


# ---- #6 MTP OFF / K1 / K3 (FP8, stock vLLM 0.32, seqs 20); K3 = G0 of #3 (FlashInfer, PIECEWISE pinned), profiled.
# rev 3c: every K3 launch (main + A/A reps) has the same command (profiler config included); only the main one profiles.
def s6(depth, profile=False, tag=None):
    def f():
        pdir = TMP / f"e23_profile_mtp{depth}_{tag or 'main'}" if depth == 3 else None
        extra = ["--compilation-config", json.dumps({"cudagraph_mode": "PIECEWISE"})] if depth == 3 else []
        srv = stock_server(tag or f"fp8_mtp{depth}", stock_flags(FP8, 8200, spec=depth, extra=extra, profile_dir=pdir), 8200)

        def body(ss):
            s = ss[0]
            port = s.port
            o = {"probes": L.chat_probes(port),
                 "oracle": oracle(port, key=("fp8_mtp0" if depth == 0 else ("fp8_g0" if depth == 3 else None)) if not tag else None,
                                  refs=() if depth == 0 and not tag else ("fp8_mtp0",))}
            if depth == 0:
                o["steady"] = {} if SM else levels(port, "40k", [8, 12, 16, 20], server=s)
                if not SM:
                    o["steady"].update(levels(port, "90k", [12, 20], server=s))
            elif depth == 1:
                o["steady"] = levels(port, "40k", [8, 12, 16, 20], server=s)
            else:
                o["steady"] = levels(port, "40k", [16] if SM else [8, 12, 16, 20], server=s)
                if not SM:
                    o["steady"].update(levels(port, "90k", [8, 12, 20], server=s))
                    o["steady"].update(levels(port, "120k", [8], server=s))
            if profile and s.alive():
                o["profile"] = L.profiled_load(port, PROMPTS["40k"], 16, pdir, max_new=MAX_NEW["40k"], spec=True,
                                                 prefill_tok_s=450.0)   # rev 3e: settle limit ~1540 s (cold cache)
                _keep_trace(pdir, tag or f"mtp{depth}")
            if pdir:
                shutil.rmtree(pdir, ignore_errors=True)
            return gated(o, need=() if depth == 0 and not tag else ("fp8_mtp0",))
        o = taint_check(with_servers([srv], 1800, body))
        if profile:
            o["graph_proof"] = graph_proof(o, expect="PIECEWISE")
        return o
    return f


def graph_proof(o, expect=None):
    """rev 3d (#6): a graph verdict counts only from a settled, decode-only, spec-active profile with per-forward
    annotations and no graph-mode downgrade line; `expect` = the mode this arm must show (G0 = PIECEWISE control)."""
    pr = o.get("profile") or {}
    tr = pr.get("trace") or {}
    downgr = [l for v in (o.get("servers") or {}).values() for l in ((v or {}).get("graph_downgrade_lines") or [])]
    if pr.get("status"):                                  # INCONCLUSIVE_PREFILL_NOT_SETTLED / _IN_PROFILE / _NO_SPEC
        return {"status": pr["status"]}
    if not tr or tr.get("error"):
        return {"status": "INCONCLUSIVE_GRAPH", "why": tr.get("error") or "no trace"}
    if tr.get("graph_verdict") in (None, "UNKNOWN", "UNPROVEN"):
        return {"status": "INCONCLUSIVE_GRAPH", "why": f"verdict {tr.get('graph_verdict')} ({tr.get('verdict_basis')})"}
    # rev 3e: a downgrade counts only when vLLM fell back to PIECEWISE / NONE (FULL_AND_PIECEWISE -> FULL_DECODE_ONLY
    # still runs full verify graphs)
    to_low = [l for l in downgr if re.search(r"(?:=|to)\s*['\"]?(?:CUDAGraphMode\.)?(PIECEWISE|NONE)\b", l)]
    if to_low:
        return {"status": "DOWNGRADED", "why": "graph mode downgraded to PIECEWISE / NONE", "lines": to_low[:3],
                "verdict": tr.get("graph_verdict")}
    v = tr["graph_verdict"]
    if expect and v != expect:
        return {"status": "MISMATCH_EXPECTED", "verdict": v, "expected": expect}
    return {"status": "PROVEN_" + v, "verdict": v, "per_forward": [tr.get("per_forward_median_graph_launches"),
                                                                   tr.get("per_forward_median_kernel_launches")]}


def _keep_trace(pdir, tag):
    """rev 3c: the rank-0 trace is kept in OUT (the verdict code has never seen a real 0.32 trace)."""
    try:
        fs = sorted(Path(pdir).rglob("*.pt.trace.json*"), key=lambda p: p.stat().st_size)
        f0 = next((p for p in fs if re.search(r"rank0\.|rank[-_]?0[._]|tp0_", p.name)), fs[0] if fs else None)
        if f0 is not None and f0.stat().st_size < 400e6:
            (OUT / "traces").mkdir(exist_ok=True)
            shutil.copy2(f0, OUT / "traces" / f"{tag}_{f0.name}")
    except Exception as e:
        R["errors"].append(f"keep_trace {tag}: {e!r}"[:200])


def _site():
    return subprocess.run([STOCK["py"], "-c", "import sysconfig;print(sysconfig.get_paths()['purelib'])"],
                          capture_output=True, text=True).stdout.strip()


PATCH_ORIG = {}


def _patch(reverse=False):
    """forward = back up flashinfer.py, dry-run, apply (restore the backup on a failed apply); reverse = restore the
    backup by copy and verify it against the sha recorded BEFORE patching (rev 3c: not against the backup itself)."""
    target = Path(_site()) / "vllm/v1/attention/backends/flashinfer.py"
    bak = TMP / "flashinfer.py.e23_orig"
    if reverse:
        if not bak.is_file():
            return {"applied": False, "error": "no backup"}
        shutil.copy2(bak, target)
        now = hashlib.sha256(target.read_bytes()).hexdigest()
        return {"applied": now == PATCH_ORIG.get("sha"), "restored_sha": now[:16]}
    shutil.copy2(target, bak)
    PATCH_ORIG["sha"] = hashlib.sha256(target.read_bytes()).hexdigest()
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
            "orig_sha": PATCH_ORIG["sha"][:16]}


def s3(tag, backend, graph, patch=False):
    def f():
        pre = None
        try:                          # rev 3c: the patch is inside the try, so any exception still restores the file
            pre = _patch() if patch and not FAKE else None
            if pre and not pre["applied"]:
                return {"status": "PATCH_FAILED", "patch": pre}
            pdir = TMP / f"e23_profile_{tag}"
            extra = ["--compilation-config", json.dumps({"cudagraph_mode": graph})] if graph else []
            srv = stock_server(f"fp8_{tag}", stock_flags(FP8, 8200, backend=backend, extra=extra, profile_dir=pdir), 8200)

            def body(ss):
                s = ss[0]
                port = s.port
                o = {"probes": L.chat_probes(port), "oracle": oracle(port, refs=("fp8_mtp0", "fp8_g0")), "patch": pre}
                o["steady"] = levels(port, "40k", [16], server=s)
                if s.alive():
                    o["profile"] = L.profiled_load(port, PROMPTS["40k"], 16, pdir, max_new=MAX_NEW["40k"], spec=True,
                                                 prefill_tok_s=450.0)   # rev 3e: settle limit ~1540 s (cold cache)
                    _keep_trace(pdir, tag)
                shutil.rmtree(pdir, ignore_errors=True)
                return gated(o, need=("fp8_mtp0",))
            o = taint_check(with_servers([srv], 1800, body))
            o["graph_proof"] = graph_proof(o, expect="FULL")
            g0 = ((R["sections"].get("6_mtp_depth") or {}).get("mtp3_eq_G0_piecewise") or {}).get("graph_proof") or {}
            gs = g0.get("status")
            if gs in ("PROVEN_FULL", "MISMATCH_EXPECTED"):   # the PIECEWISE-pinned control read FULL: the verdict logic lies
                o["graph_proof"] = {"status": "INCONCLUSIVE_GRAPH", "why": f"G0 control broken: {gs}", "own": o["graph_proof"]}
            elif gs != "PROVEN_PIECEWISE" and str(o["graph_proof"].get("status", "")).startswith("PROVEN_"):
                # rev 3f (GPT rev 3e audit): without the PIECEWISE control a FULL verdict is not proven
                o["graph_proof"] = {"status": "INCONCLUSIVE_GRAPH", "why": f"no G0 PIECEWISE control ({gs})",
                                    "own": o["graph_proof"]}
            return o
        finally:
            if patch and not FAKE and (TMP / "flashinfer.py.e23_orig").is_file() and PATCH_ORIG.get("sha"):
                rv = _patch(reverse=True)
                R["sections"].setdefault("3_full_graph", {})[f"{tag}_patch_revert"] = rv
                if not rv.get("applied"):
                    ABORT["why"] = "PR #60532 revert failed: the stock venv is no longer pristine"
    return f


# ---- #5 NVFP4 KV. Teacher-forced on its own short launch per KV dtype (same settings both sides, prompt-logprob
# memory fits); target-only = correctness + one census level; NVFP4 + MTP3 = the residency census + streams increase.
def s5_tf(kv, key=None, ref=None, name=None):
    """Teacher-forced rows on a short launch (same settings for every KV dtype). With `ref`, each band must be a
    complete comparison (compare_tf: every expected position valid on both sides) for DONE."""
    def f():
        srv = stock_server(name or f"tf_{kv}", stock_flags(FP8, 8200, spec=0, kv=kv, seqs=2, batched=512, mem=0.80), 8200)

        def body(ss):
            o = {"teacher_forced": tforced(ss[0].port, key=key, ref=ref)}
            if ref:
                st = [v.get("status") for v in o["teacher_forced"].values()]
                o["status"] = "DONE" if st and all(x == "MEASURED" for x in st) else "INCONCLUSIVE_TF"
            elif key and not L.tf_complete(TF.get(key) or {}):
                o["status"] = "INCONCLUSIVE_TF"                    # rev 3e: every position of every band valid
            return o
        return taint_check(with_servers([srv], 1800, body))
    return f


def _num(x, default):
    return x if isinstance(x, (int, float)) else default


def nvfp4_gate():
    """rev 3e (four agent reports on rev 3d): an FP8-vs-FP8 A/A is exactly 0 on a deterministic engine, so the
    tolerance is calibrated against the production KV dtype instead. BF16 KV (kv auto) is the teacher-forced
    reference; FP8 KV and NVFP4 KV are both measured against it; NVFP4 passes when it is not much worse than FP8 KV:
    mean |dlogprob| <= max(3 x FP8, 0.02), p99 <= max(3 x FP8, 0.2), rank-1 agreement >= FP8 - 0.02 (screening policy,
    not an ARC accuracy proof). Plus: target-only DONE with a clean oracle vs fp8_mtp0, every TF step DONE (complete)."""
    sec = R["sections"].get("5_nvfp4_kv") or {}
    tgt, ref, f8, nv = (sec.get(k) or {} for k in ("nvfp4_target_only", "tf_bf16kv_reference", "tf_fp8kv_vs_bf16",
                                                    "tf_nvfp4_vs_bf16"))
    out = {"target_only": tgt.get("status"), "target_oracle": ((tgt.get("oracle") or {}).get("vs_fp8_mtp0") or {}).get("status"),
           "tf_bf16_reference": ref.get("status"), "tf_fp8": f8.get("status"), "tf_nvfp4": nv.get("status"), "bands": {}}
    if out["target_only"] != "DONE" or out["target_oracle"] not in CLEAN:
        return False, {**out, "why": "TARGET_ONLY_NOT_CLEAN"}
    if out["tf_bf16_reference"] != "DONE" or out["tf_fp8"] != "DONE" or out["tf_nvfp4"] != "DONE":
        return False, {**out, "why": "TEACHER_FORCED_INCOMPLETE"}
    ok = True
    for b, r in (nv.get("teacher_forced") or {}).items():
        a = (f8.get("teacher_forced") or {}).get(b) or {}
        if r.get("status") != "MEASURED" or a.get("status") != "MEASURED":
            ok = False
            out["bands"][b] = {"pass": False, "why": "band not MEASURED", "nvfp4": r.get("status"), "fp8": a.get("status")}
            continue
        lim = {"mean_max": max(3 * _num(a.get("mean_abs_dlogprob"), 9e9), 0.02),
               "p99_max": max(3 * _num(a.get("p99_abs_dlogprob"), 9e9), 0.2),
               "rank1_min": _num(a.get("rank1_agreement"), 2.0) - 0.02}
        good = (_num(r.get("mean_abs_dlogprob"), 9e9) <= lim["mean_max"] and _num(r.get("p99_abs_dlogprob"), 9e9) <= lim["p99_max"]
                and _num(r.get("rank1_agreement"), -1.0) >= lim["rank1_min"])
        keys = ("n", "rank1_agreement", "mean_abs_dlogprob", "p99_abs_dlogprob", "max_abs_dlogprob")
        out["bands"][b] = {"nvfp4": {k: r.get(k) for k in keys}, "fp8": {k: a.get(k) for k in keys}, "limits": lim, "pass": good}
        ok = ok and good
    if not out["bands"]:
        return False, {**out, "why": "NO_TF_BANDS"}
    return ok, {**out, "why": None if ok else "TF_OUTSIDE_TOLERANCE"}


def s5(spec, tag=None):
    def f():
        gate = None
        if spec:
            ok, gate = nvfp4_gate()
            if not ok:
                return {"status": "SKIPPED_CORRECTNESS_GATE", "gate": gate}
        srv = stock_server(tag or f"nvfp4_mtp{spec}", stock_flags(FP8, 8200, spec=spec, kv="nvfp4", seqs=24), 8200)

        def body(ss):
            s = ss[0]
            port = s.port
            o = {"probes": L.chat_probes(port), "oracle": oracle(port, refs=("fp8_mtp0",) if spec == 0 else ("fp8_g0", "fp8_mtp0"))}
            if spec == 0:
                g = oracle_gate(o, need=("fp8_mtp0",))           # rev 3d: not only FAIL_HIGH_MARGIN stops it
                if g:
                    o.update(status=g, correctness=g)
                    return o
                o["steady"] = levels(port, "120k", [8], server=s) if SM else levels(port, "90k", [12], server=s)
            else:
                o["steady"] = levels(port, "90k", [12, 16, 20, 24], server=s)
                o["steady"].update(levels(port, "120k", [12, 16, 20], server=s))
            if gate:
                o["nvfp4_gate"] = gate                               # rev 3e: the admitted evidence stays on record
            return gated(o, need=("fp8_mtp0",))
        return taint_check(with_servers([srv], 1800, body))
    return f


# ---- #4 INT4 (stock vLLM, RedHat W4A16). rev 3d (#4): the same-checkpoint target-only oracle (MTP off) first; the
# INT4 MTP3 control (against Merlin's low-M INT4 engine) and PP4 compare with it, never with themselves or with FP8
def s4_int4_ref():
    if INT4 is None:
        return {"status": "SKIPPED_REQUIRED_ARTIFACT", "why": "RedHat INT4 checkpoint not attached"}
    srv = stock_server("int4_mtp0_ref", stock_flags(INT4, 8200, spec=0, seqs=8), 8200)
    return taint_check(with_servers([srv], 1800, lambda ss: gated({"oracle": oracle(ss[0].port, key="int4_mtp0")})))


def s4_int4():
    if INT4 is None:
        return {"status": "SKIPPED_REQUIRED_ARTIFACT", "why": "RedHat INT4 checkpoint not attached"}
    srv = stock_server("int4_mtp3_s20", stock_flags(INT4, 8200, spec=3, seqs=20), 8200)

    def body(ss):
        s = ss[0]
        port = s.port
        o = {"probes": L.chat_probes(port), "oracle": oracle(port, refs=("int4_mtp0",)),
             "note": "engine + checkpoint comparison vs Merlin (RedHat INT4 on stock vLLM), not a patch-only A/B"}
        o["steady"] = levels(port, "40k", [8, 20], server=s)
        o["steady"].update(levels(port, "90k", [12], server=s))
        return gated(o, need=("int4_mtp0",))
    return taint_check(with_servers([srv], 1800, body))


# ---- #1e PP4 branch (stock vLLM, INT4 RedHat). rev 3d (#5): eager first (#63505: compiled PP faults / gibberish on the
# hybrid GDN model), compiled as a later separate arm only after eager passed; staged fail-fast, each stage against
# int4_mtp0: short probes -> 2k -> 8k -> 40k greedy, then throughput
def s1_pp4(eager):
    def f():
        if INT4 is None:
            return {"status": "SKIPPED_REQUIRED_ARTIFACT", "why": "RedHat INT4 checkpoint not attached"}
        if not ROWS.get("int4_mtp0"):
            return {"status": "SKIPPED_NO_ORACLE", "why": "no clean int4_mtp0 reference"}
        extra = ["--enforce-eager"] if eager else []
        srv = stock_server("int4_pp4_eager" if eager else "int4_pp4_compiled",
                           stock_flags(INT4, 8200, tp=1, pp=4, spec=0, seqs=16, extra=extra), 8200)

        def body(ss):
            s = ss[0]
            port = s.port
            o = {"probes": L.chat_probes(port), "stages": {},
                 "note": "PP4 uses stock vLLM + RedHat INT4 (Merlin's fork has no PP profile); #63505 open on 0.32; "
                         "the V1 runner forces async scheduling off under PP, so these rows understate PP"}
            if not o["probes"]["ok"]:
                o["status"] = "NO_GO_SHORT_PROBE"
                return o
            ref = ROWS["int4_mtp0"]
            for name, idx in (("2k", [0]), ("8k", [2]), ("40k", [3])):      # ORACLE_P + LONG_P order of oracle()
                rows = L.greedy(port, [(ORACLE_P + LONG_P)[i] for i in idx], n_tok=NTOK, timeout=max(120, min(900, left() / 6)))
                cmp_ = L.compare([ref[i] for i in idx], rows)
                bad = any(r.get("error") or not r.get("tokens") or L.degenerate(r.get("text")) for r in rows)
                o["stages"][name] = {"compare": cmp_, "bad_output": bad, "text_head": [r["text"][:120] for r in rows]}
                if bad or cmp_.get("status") not in CLEAN:
                    o["status"] = f"NO_GO_{name}"
                    o["correctness"] = o["status"]
                    return o
            o["steady"] = levels(port, "40k", [8] if SM else [8, 16], server=s)
            return gated(o)                  # rev 3e: garbage under load in the throughput windows is NO_GO too
        return taint_check(with_servers([srv], 1800, body))
    return f


def pp4():
    rec = step("1_tp1x4_pp4", "e_pp4_int4_eager", 15 if SM else 25, s1_pp4(True))
    if rec.get("status") == "DONE":
        step("1_tp1x4_pp4", "f_pp4_int4_compiled", 15 if SM else 25, s1_pp4(False))
    else:
        R["sections"].setdefault("1_tp1x4_pp4", {})["f_pp4_int4_compiled"] = {"status": "SKIPPED_EAGER_NOT_PROVEN",
                                                                              "eager_status": rec.get("status")}


# ---- EAGLE-4 feasibility: a BF16-KV FP8-weight target-only oracle first (same KV dtype as tree verify), then EAGLE
def s7_kvauto_oracle():
    srv = stock_server("fp8_kvauto_mtp0", stock_flags(FP8, 8300, spec=0, kv="auto", seqs=16), 8300)
    return taint_check(with_servers([srv], 1800, lambda ss: gated({"oracle": oracle(ss[0].port, key="fp8_kvauto_mtp0")})))


def s7_eagle():
    eng = eagle_install()
    if eng is None:
        return {"status": "SKIPPED_REQUIRED_ARTIFACT", "why": "EAGLE env / drafter not attached (or fake mode)"}
    spec = json.dumps({"method": "eagle_harness", "model": str(EAGLE_DRAFT),
                       "method_config": {"mode": "static", "node_top_k": 4, "tree_depth": 14, "max_verify_width": 16}})

    def body(ss):
        s = ss[0]
        port = s.port
        o = {"probes": L.chat_probes(port), "oracle": oracle(port, refs=("fp8_kvauto_mtp0", "fp8_mtp0"))}
        o["steady"] = levels(port, "8k", [8, 16], server=s)
        o["steady"].update(levels(port, "40k", [8], server=s))
        return gated(o, need=("fp8_kvauto_mtp0",))       # same KV dtype (BF16) as EAGLE's tree verify
    # rev 3d (#12): the official recipe first (--linear-backend flashinfer_cutedsl; CuTe-DSL may lack sm_89 kernels),
    # then the default linear backend; rev 3c: prefix caching ON first, OFF last and labelled confounded
    tries = [("eagle4_fp8_cutedsl", ["--linear-backend", "flashinfer_cutedsl"], True),
             ("eagle4_fp8", [], True),
             ("eagle4_fp8_noprefix", [], False)]
    attempts = []
    deadline = time.time() + 40 * 60                    # rev 3e: all EAGLE variants share one 40-min window
    for name, extra, prefix in tries:
        if ABORT["why"]:
            return {"status": "SKIPPED_ABORT", "attempts": attempts}
        if time.time() > deadline or left() < 20 * 60:
            return {"status": "SKIPPED_EAGLE_DEADLINE", "attempts": attempts}
        o = taint_check(with_servers([stock_server(name, stock_flags(FP8, 8300, seqs=16, kv="auto", backend="CUSTOM",
                                                                     prefix=prefix, spec_json=spec, extra=extra),
                                                   8300, engine=eng)],
                                     max(300, min(1800, deadline - time.time())), body))
        o.update(variant=name, linear_backend=extra[1] if extra else "default", prefix_caching=prefix)
        if not prefix:
            o["confounded"] = "prefix caching OFF (E19: spec decode + prefix off flips tokens)"
        if o.get("status") in (None, "DONE"):
            o["earlier_attempts"] = attempts
            return o
        # rev 3e: a launch failure, a crash after ready, a body error or a NO_GO all fall through to the next variant
        attempts.append({k: o[k] for k in ("variant", "status", "correctness", "servers") if k in o})
    return {"status": "NO_GO_ALL_EAGLE_VARIANTS", "attempts": attempts}


MERLIN_SHA = {"th": None, "ok": None, "bad": []}


def _merlin_sha():
    """rev 3d (#14): full sha256 of every Merlin checkpoint file against the P1 manifest (size-only is not identity);
    runs after the prefetch so the bytes come from the page cache."""
    mt = json.loads((MERLIN / "q27_e23_p1_manifest.json").read_text())
    bad = []
    for n, (size, sha) in mt["files"].items():
        if n == ".gitattributes" or not sha:
            continue
        h = hashlib.sha256()
        with open(MERLIN / n, "rb") as fh:
            for blk in iter(lambda: fh.read(16 << 20), b""):
                h.update(blk)
        if h.hexdigest() != sha:
            bad.append(n)
    MERLIN_SHA.update(ok=not bad, bad=bad)
    R["inputs"]["merlin_sha_ok"] = not bad
    try:
        save()
    except Exception:
        pass                          # rev 3e: a save race with another thread never turns a good hash into FAIL


def merlin_sha_gate():
    """rev 3e (audits A/B/C/D): every Merlin step. The install runs first (it overlaps the prefetch + hash); a mismatch
    fails every Merlin step; a hash still running after the wait lets the step run, labelled UNVERIFIED."""
    if FAKE or MERLIN_SHA["th"] is None:
        return None
    try:
        merlin_install()
    except Exception:
        pass                          # the step itself reports the install failure
    MERLIN_SHA["th"].join(timeout=max(60, min(900, left() - 900)))
    if MERLIN_SHA["ok"] is False:
        return {"status": "FAIL_CHECKPOINT_SHA", "bad": MERLIN_SHA["bad"]}
    if MERLIN_SHA["ok"] is None:
        R["setup"]["merlin_sha"] = "UNVERIFIED_TIMEOUT"
    return None


def start_prefetch():
    pref = []
    if DO_A and MERLIN:
        th = prefetch_bg([MERLIN], "merlin")
        pref.append(th)
        if not FAKE:
            def _after():
                th.join()
                try:
                    _merlin_sha()
                except Exception as e:
                    MERLIN_SHA.update(ok=False, bad=[f"{type(e).__name__}: {e}"[:200]])
            MERLIN_SHA["th"] = threading.Thread(target=_after, daemon=True)
            MERLIN_SHA["th"].start()
    if DO_B:
        pref.append(prefetch_bg([FP8], "fp8"))
    return pref


# ---- run order (audit5 rev 3b: realistic estimates, trimmed levels the plan does not need, PP4 before the A/A fill).
# SMOKE A: P5, Merlin TP1 (+P4), Merlin TP4, Merlin TP1 x4 (40k). SMOKE B: mtp0 (oracle) -> mtp3/G0 -> G3 -> TF FP8 ref
# -> NVFP4 target-only -> TF NVFP4 -> [kvauto oracle + EAGLE] -> PP4. FULL: P5, Merlin a-d, mtp0, INT4 control, mtp1,
# mtp3/G0, G1, G2, G3, TF FP8 ref, NVFP4 target-only, TF NVFP4, NVFP4 + MTP3, PP4, then A/A repeats fill the time.
PREF = []
if DO_A:
    step("2_host_allreduce", "p5_bench", 12 if SM else 30, s2)      # before the prefetch (host memory + PCIe quiet)
PREF = start_prefetch() if not ABORT["why"] else []
if DO_A:
    if not ABORT["why"] and left() > (40 if SM else 50) * 60 + 300:     # rev 3e: never after the P5 release abort
        P4_FUT = CF.ThreadPoolExecutor(1).submit(s4_thread)
    step("1_tp1x4_pp4", "a_merlin_tp1_gpu0", 40 if SM else 50, s1_tp1)
    join_p4()
    if not SM:
        step("1_tp1x4_pp4", "b_merlin_tp1_x2", 25, s1_rep(2))
        step("1_tp1x4_pp4", "c_merlin_tp1_x4", 25, s1_rep(4))
    step("1_tp1x4_pp4", "d_merlin_tp4", 40 if SM else 62, s1_tp4)
    if SM:
        step("1_tp1x4_pp4", "c_merlin_tp1_x4", 20, s1_rep(4, bands=("40k",)))
if DO_B:
    if INT4:
        PREF.append(prefetch_bg([INT4], "int4"))
    step("6_mtp_depth", "mtp0_target_only", 12 if SM else 45, s6(0))
    step("4_sm89_fp8_gemm", "int4_mtp0_target_only_ref", 12, s4_int4_ref)   # rev 3d (#4): SMOKE_B + FULL, early
    if not SM:
        step("4_sm89_fp8_gemm", "int4_mtp3_s20_control", 35, s4_int4)
        step("6_mtp_depth", "mtp1", 30, s6(1))
    step("6_mtp_depth", "mtp3_eq_G0_piecewise", 20 if SM else 62, s6(3, profile=True))
    if not SM:
        step("3_full_graph", "G1_triton_full_and_piecewise", 25, s3("G1", "TRITON_ATTN", "FULL_AND_PIECEWISE"))
        step("3_full_graph", "G2_triton_full_decode_only", 25, s3("G2", "TRITON_ATTN", "FULL_DECODE_ONLY"))
    step("3_full_graph", "G3_flashinfer_pr60532", 20 if SM else 25, s3("G3", "FLASHINFER", "FULL_AND_PIECEWISE", patch=True))
    # rev 3e: BF16 KV = the teacher-forced reference; FP8 KV (production) and NVFP4 KV are measured against it
    step("5_nvfp4_kv", "tf_bf16kv_reference", 11 if SM else 15, s5_tf("auto", key="bf16kv_tf"))
    step("5_nvfp4_kv", "tf_fp8kv_vs_bf16", 11 if SM else 15, s5_tf("fp8_e4m3", ref="bf16kv_tf"))
    step("5_nvfp4_kv", "nvfp4_target_only", 25 if SM else 30, s5(0))
    step("5_nvfp4_kv", "tf_nvfp4_vs_bf16", 11 if SM else 15, s5_tf("nvfp4", ref="bf16kv_tf"))
    if not SM:
        step("5_nvfp4_kv", "nvfp4_mtp3", 75, s5(3))
    if (SM or os.environ.get("E23_EAGLE_FULL") == "1") and left() >= (11 + 40 + 30) * 60:      # rev 3e: EAGLE window 40 min, PP4 eager 30 min after it
        step("7_eagle4", "fp8_kvauto_target_only_oracle", 11, s7_kvauto_oracle)
        step("7_eagle4", "eagle4_fp8_tp4", 40, s7_eagle)
    pp4()
if not SM:      # A/A repeats of the decisive comparisons fill whatever time is left (greedy; nothing reserved after)
    def _done(path):              # rev 3d (#3): DONE and a clean correctness record (rev 3e: a GPU fault elsewhere only labels)
        sec, name = path.split("/")
        rec = (R["sections"].get(sec) or {}).get(name) or {}
        return rec.get("status") == "DONE" and rec.get("correctness") == "CLEAN"

    def AA(i):       # only configurations whose main run completed (a failed config is not relaunched)
        c = [(f"mtp3_control_rep{i}", 60, s6(3, tag=f"fp8_mtp3_rep{i}"), "6_mtp_depth/mtp3_eq_G0_piecewise"),
             (f"merlin_tp1_x4_rep{i}", 30, s1_rep(4), "1_tp1x4_pp4/c_merlin_tp1_x4"),
             (f"nvfp4_mtp3_rep{i}", 75, s5(3, tag=f"nvfp4_mtp3_rep{i}"), "5_nvfp4_kv/nvfp4_mtp3")]
        return [(n, e, fn) for n, e, fn, main in c if _done(main)]
    for i in range(1, 7):
        todo = [(n, e, fn) for n, e, fn in AA(i) if left() >= e * EST_SCALE * 60 + 300]
        if not todo or ABORT["why"]:
            break
        for n, e, fn in todo:
            step("8_aa_repeats", n, e, fn)
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
st = ["| section | step | status | correctness | windows steady/all | oracle | probes | KV tokens | ready s | graph capture | "
      "trace verdict | graph proof |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
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
        st.append(f"| {sec} | {name} | {rec.get('status')} | {rec.get('correctness', '-')} | {ws} | "
                  f"{vs or ('saved' if o else '-')} | {(rec.get('probes') or {}).get('ok', '-')} | {sv.get('kv_tokens', '-')} | "
                  f"{sv.get('ready_sec', '-')} | {','.join(sv.get('graph_capture_modes') or []) or '-'} | {gv} | "
                  f"{(rec.get('graph_proof') or {}).get('status', '-')} |")
R["help_check"] = HELP
NOTES = ["Notes (rev 3d):",
         "- MEASURED teacher-forced rows are measurements; NVFP4 passes only when its drift from BF16 KV is within 3x the "
         "production FP8 KV drift (mean, p99) and rank-1 within 0.02 (nvfp4_gate; a screening policy, not an ARC accuracy proof).",
         "- The P5 bench is a 983,040 B 4-rank graph microbenchmark against NCCL's 343.88 us; a microbenchmark gain is not "
         "an engine gain until the collective runs inside vLLM.",
         "- The INT4 control vs Merlin is an engine + checkpoint comparison (RedHat INT4 on stock vLLM), not a patch-only A/B.",
         "- tok/s is compared with the PDF baselines (INT4 MTP3 332, FP8 SSM16 312 at 90k) only with the same workload, actual "
         "running streams (running_mean) and accepted-output accounting (generation_tokens_total).",
         "- A graph verdict counts only as PROVEN_* (settled, decode-only, spec-active, annotated forwards, G0 control "
         "PIECEWISE, else UNCONTROLLED); arms with a GPU fault line (NO_GO_GPU_FAULT): " + str(R["setup"].get("gpu_fault_arms")) +
         "; Merlin checkpoint sha: " + str(R["inputs"].get("merlin_sha_ok", R["setup"].get("merlin_sha")))]
R["report_md"] = "\n".join(st) + "\n\n" + "\n".join(lines) + "\n\n" + "\n".join(NOTES)
R["minutes"] = round((time.time() - T_START) / 60, 1)
save()
print(R["report_md"])
print("E23", MODE, PART, "done in", R["minutes"], "min | abort:", ABORT["why"])
'''

import re as _re
_full = 0
for _m in _re.finditer(r'step\("([^"]+)", "([^"]+)", (?:(\d+) if SM else (\d+)|(\d+)),', CELL5):
    if _m.group(1) in ("7_eagle4", "8_aa_repeats") or (_m.group(2) == "c_merlin_tp1_x4" and _m.group(5) == "20"):
        continue
    _full += int(_m.group(4) or _m.group(5))
assert _full <= 670, f"FULL step estimates {_full} min leave no room in the 690-min budget"
print("FULL main-sequence estimate", _full, "min of 690")
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
    kd = HERE / "gpu_kernels" / f"q27-xhigh-e23-six-levers-{variant.lower().replace('_', '-')}"
    kd.mkdir(parents=True, exist_ok=True)
    (kd / f"{kd.name}.ipynb").write_text(data, encoding="utf-8", newline="\n")
    (kd / "kernel-metadata.json").write_text(json.dumps({
        "id": f"foysalemonshanto/{kd.name}", "title": kd.name.replace("-", " ").title(), "code_file": f"{kd.name}.ipynb",
        "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False,
        "enable_internet": False, "keywords": [],
        "dataset_sources": ["foysalemonshanto/q27-e23-sm89-kernels", "foysalemonshanto/vllm-0-32-0-cu13-wheelhouse"],
        "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
        "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1",
                          "foysalemonshanto/qwen3-8-27b-int4-redhat/PyTorch/w4a16-g128/1",
                          "foysalemonshanto/qwen3-8-27b-w4a16-merlin/Transformers/default/1",
                          "foysalemonshanto/e23-merlin-env/Other/v1/1",
                          "foysalemonshanto/e23-eagle4-env/Other/default/1",
                          "foysalemonshanto/qwen3-8-27b-eagle4/Transformers/default/1"],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
        "machine_shape": "NvidiaL4"}, indent=2), encoding="utf-8", newline="\n")
old = HERE.parent / "Q27-XHIGH-E23_SIX_LEVERS.ipynb"
if old.exists():
    old.unlink()          # rev 3: the three variants replace the single rev 1 / 2 file (never run on Kaggle)
