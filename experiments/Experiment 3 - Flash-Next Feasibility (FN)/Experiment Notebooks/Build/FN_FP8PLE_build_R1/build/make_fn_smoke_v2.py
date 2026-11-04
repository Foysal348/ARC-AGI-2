"""Build FN-SMOKE: can Qwen3.8-Flash-Next (Intel W4A16 AutoRound) be served by Pennyroyal/SGLang on Kaggle 4xL4 for
ARC-AGI-2, and how does it do on a few evaluation tasks? Every stage is recorded; the first failing stage stops the run
and the report says exactly where. Dual mode: in the scoring rerun it only writes a valid fallback submission."""
import hashlib
import json
from pathlib import Path

DST = Path(r"D:\ARC AGI 2\Research Base Experiment\FN-SMOKE-FLASHNEXT-4XL4.ipynb")

C_HEAD = r'''# FN-SMOKE — Qwen3.8-Flash-Next (Intel W4A16 AutoRound) on Kaggle 4×L4 for ARC-AGI-2

Feasibility probe before any full run. The model and the Pennyroyal SGLang fork come from dfranzen's ARC-AGI-3
Milestone-2 notebook, which runs on one RTX PRO 6000 (Blackwell, sm120, CUDA 13). Here: 4×L4 (Ada, sm89).

Stages (each recorded in `fn_smoke_report.json`; the first failure stops the run):
0. environment — driver / max CUDA version, RAM, disk, GPUs (stop early if the driver cannot run CUDA 13)
1. inputs — model, wheelhouse, competition data present
2. install Pennyroyal offline (uv venv)
3. torch + CUDA inside the venv on L4
4. launch the server — tries, in order: TP4+EP4 with DP-attention; same with triton linear-attention backends;
   TP4+EP4; TP4 (the experts' INT4 group size 128 does not divide 640/4, so plain TP4 is expected to fail)
5. ARC test — 4 evaluation tasks, full thinking (up to 16k tokens), rule + `solution(grid)`, verified on the
   training pairs and scored on the test outputs; throughput measured

Commit / interactive run: stages 0–5. Competition rerun (`KAGGLE_IS_COMPETITION_RERUN`): only a valid fallback
`submission.json` (this probe is not a solver).

v2 (after review): `LOAD_ONLY = True`: stages 0-4 + one 64-token generation; absolute 45-min deadline, every blocking call clamped to it. Hardware gates (4xL4, sm89, driver >=580 / CUDA 13, available RAM >=110 GB, /tmp >=25 GB); preflight of CLI flags + sgl_kernel/flashinfer/triton imports on all 4 GPUs before the 181 GB load; minimal first-boot profile (TP4+EP4, Triton GDN, BF16 KV, 8K context, 1 request, language-model-only, no CUDA graphs, no DP-attention, no speculative decoding); only the tracked server process group is killed.
'''

C_CFG = r'''# ==================================================================== CELL 1 — config, mode, report, fallback submission
import os, sys, json, time, glob, shutil, socket, zipfile, subprocess, re, shlex, threading, traceback, hashlib
from pathlib import Path

T0 = time.time()
RERUN = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
WORK = Path(os.environ.get("FN_SMOKE_WORK", "/kaggle/working")); WORK.mkdir(parents=True, exist_ok=True)   # override only for local tests
OUT = WORK / "fn_smoke"; OUT.mkdir(parents=True, exist_ok=True)
LOAD_ONLY = True                    # True: stop after the server loads + one 64-token generation (no ARC test)
SMOKE_BUDGET_SEC = float(os.environ.get("FN_SMOKE_BUDGET", (45 if LOAD_ONLY else 80) * 60))   # whole-probe wall budget
DEADLINE = T0 + SMOKE_BUDGET_SEC    # absolute; every blocking call is clamped to it
SAFETY_SEC = 60                     # left for report + bundle after the last blocking call
SERVER_ATTEMPT_TIMEOUT = 18 * 60    # per launch attempt (also clamped to the deadline)
MIN_RAM_AVAILABLE_GB = 110          # PLE pinned host table ~95.4 GB (TP4-sharded, 23.8 GB/rank) + headroom
MIN_TMP_FREE_GB = 25                # Pennyroyal venv + JIT caches
PORT = 8001

def _first(patterns):
    for p in patterns:
        hits = sorted(glob.glob(p))
        if hits:
            return hits[0]
    return None

COMP_DIR = os.environ.get("FN_SMOKE_COMP") or _first(["/kaggle/input/competitions/arc-prize-2026-arc-agi-2", "/kaggle/input/arc-prize-2026-arc-agi-2"])
WHEELHOUSE_DIR = _first(["/kaggle/input/datasets/dfranzen/pennyroyal-v253", "/kaggle/input/pennyroyal-v253"])
MODEL_DIR = _first(["/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
                    "/kaggle/input/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1",
                    "/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/*/*/*"])

REPORT = {"experiment": "FN-SMOKE-FLASHNEXT-4XL4", "rerun": RERUN, "stages": {}, "stopped_at": None,
          "paths": {"competition": COMP_DIR, "wheelhouse": WHEELHOUSE_DIR, "model": MODEL_DIR}}

def stage(name, ok, **info):
    REPORT["stages"][name] = {"ok": bool(ok), "t_sec": round(time.time() - T0, 1), **info}
    if not ok and REPORT["stopped_at"] is None:
        REPORT["stopped_at"] = name
    (OUT / "fn_smoke_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
    print(("PASS " if ok else "FAIL ") + name, json.dumps(info, default=str)[:1500], flush=True)

def remaining():
    return max(0.0, DEADLINE - time.time())

def clamp(t):
    """timeout for a blocking call: never past DEADLINE - SAFETY_SEC"""
    return max(1.0, min(float(t), remaining() - SAFETY_SEC))

def go():
    if REPORT["stopped_at"] is not None:
        return False
    if remaining() <= SAFETY_SEC:
        stage("global_deadline", False, reason="smoke wall-time budget exhausted", budget_sec=SMOKE_BUDGET_SEC)
        return False
    return True

def sh(cmd, timeout=120):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=clamp(timeout))
        return (r.stdout + r.stderr)[-20000:]
    except Exception as e:
        return repr(e)

# valid submission.json first (test inputs as answers) — both modes
def write_fallback_submission():
    test = json.load(open(Path(COMP_DIR) / "arc-agi_test_challenges.json"))
    sub = {t: [{"attempt_1": q["input"], "attempt_2": q["input"]} for q in v["test"]] for t, v in test.items()}
    (WORK / "submission.json").write_text(json.dumps(sub))
    return len(sub)

n_sub = write_fallback_submission() if COMP_DIR else 0
print("MODE:", "COMPETITION RERUN (fallback submission only)" if RERUN else "SMOKE PROBE", "| fallback submission tasks:", n_sub)
if RERUN:
    REPORT["stopped_at"] = "rerun_mode_fallback_only"
'''

C_ENV = r'''# ==================================================================== CELL 2 — stage 0: environment gates
def stage0_check(smi, gpu_csv, meminfo_text, tmp_free_gb):
    m_drv = re.search(r"Driver Version:\s*([\d.]+)", smi or ""); m_cuda = re.search(r"CUDA Version:\s*([\d.]+)", smi or "")
    driver = m_drv.group(1) if m_drv else None; driver_cuda = m_cuda.group(1) if m_cuda else None
    gpus = [g.strip() for g in (gpu_csv or "").strip().splitlines() if g.strip() and "," in g]
    names = [g.split(",")[0].strip() for g in gpus]; ccs = [g.split(",")[-1].strip() for g in gpus]
    mem = {}
    for line in (meminfo_text or "").splitlines():
        k, _, rest = line.partition(":"); p = rest.split()
        if p and p[0].isdigit(): mem[k] = int(p[0]) * 1024
    ram_total = round(mem.get("MemTotal", 0) / 1e9, 1); ram_avail = round(mem.get("MemAvailable", 0) / 1e9, 1)
    checks = {
        "four_gpus": len(gpus) == 4,
        "all_l4": bool(names) and all("L4" in n for n in names),
        "all_sm89": bool(ccs) and all(c == "8.9" for c in ccs),
        "driver_r580_plus": driver is not None and int(driver.split(".")[0]) >= 580,
        "driver_cuda13_plus": driver_cuda is not None and float(driver_cuda.split(".")[0]) >= 13,
        "ram_available_enough": ram_avail >= MIN_RAM_AVAILABLE_GB,
        "tmp_free_enough": tmp_free_gb >= MIN_TMP_FREE_GB,
    }
    info = {"driver": driver, "driver_max_cuda": driver_cuda, "gpus": gpus, "ram_total_gb": ram_total,
            "ram_available_gb": ram_avail, "tmp_free_gb": round(tmp_free_gb, 1), "checks": checks}
    return all(checks.values()), info

if go():
    smi = os.environ.get("FN_SMOKE_FAKE_SMI") or sh("nvidia-smi")
    gpu_csv = os.environ.get("FN_SMOKE_FAKE_GPUS") or sh("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader")
    try:
        meminfo = os.environ.get("FN_SMOKE_FAKE_MEMINFO") or open("/proc/meminfo").read()
    except OSError:
        meminfo = ""
    tmp_free = float(os.environ.get("FN_SMOKE_FAKE_TMPFREE") or shutil.disk_usage("/tmp" if os.path.isdir("/tmp") else ".").free / 1e9)
    ok0, info0 = stage0_check(smi, gpu_csv, meminfo, tmp_free)
    stage("0_environment", ok0, **info0, python=sys.version.split()[0], cpu_count=os.cpu_count(),
          memlock=sh("ulimit -l").strip(), disk_working=sh("df -h /kaggle/working | tail -1").strip())
'''

C_INPUTS = r'''# ==================================================================== CELL 3 — stage 1: inputs
if go():
    ok = all([COMP_DIR, WHEELHOUSE_DIR, MODEL_DIR])
    info = {}
    if MODEL_DIR:
        cfg = json.loads(Path(MODEL_DIR, "config.json").read_text())
        q = cfg.get("quantization_config", {})
        info["model_type"] = cfg.get("model_type"); info["quant"] = {k: q.get(k) for k in ("quant_method", "bits", "group_size")}
        info["model_files"] = len(list(Path(MODEL_DIR).glob("*.safetensors")))
        info["model_gb"] = round(sum(p.stat().st_size for p in Path(MODEL_DIR).glob("*.safetensors")) / 1e9, 1)
    if WHEELHOUSE_DIR:
        info["sglang_wheels"] = [p.name for p in Path(WHEELHOUSE_DIR).rglob("sglang-*.whl")]
    stage("1_inputs", ok, **info)
'''

C_INSTALL = r'''# ==================================================================== CELL 4 — stage 2: install Pennyroyal (offline)
PREFIX = "/tmp/sgl-intel"; VENV = f"{PREFIX}/venv"; PYTHON = f"{VENV}/bin/python"; SGLANG = f"{VENV}/bin/sglang"
if go():
    t = time.time()
    try:
        WHEELS = next(p for p in Path(WHEELHOUSE_DIR).rglob("wheels") if p.is_dir())
        LOCK = next(Path(WHEELHOUSE_DIR).rglob("requirements.lock"))
        wheel = sorted(WHEELS.glob("sglang-*.whl"))[0]
        install_env = dict(os.environ, UV_OFFLINE="1", UV_PYTHON_DOWNLOADS="never", PYTHONNOUSERSITE="1"); install_env.pop("PYTHONPATH", None)
        uv_whl = sorted(WHEELS.glob("uv-*.whl"))[0]
        uv_bin = Path(PREFIX) / "bin/uv"; uv_bin.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(uv_whl) as a:
            member = [n for n in a.namelist() if Path(n).name == "uv" and not n.endswith("/")][0]
            uv_bin.write_bytes(a.read(member))
        uv_bin.chmod(0o755)
        def run(cmd, env=install_env, timeout=1800):
            r = subprocess.run(list(map(str, cmd)), text=True, capture_output=True, env=env, timeout=clamp(timeout))
            if r.returncode:
                raise RuntimeError(f"{shlex.join(map(str, cmd))} -> {r.returncode}\n{(r.stdout + r.stderr)[-4000:]}")
            return r
        run([uv_bin, "venv", "--python", sys.executable, VENV])
        run([uv_bin, "pip", "install", "--python", PYTHON, "--no-index", "--find-links", WHEELS, "-r", LOCK])
        run([uv_bin, "pip", "install", "--python", PYTHON, "--no-index", "--find-links", WHEELS, "--reinstall", "--no-deps", wheel])
        stage("2_install", True, sec=round(time.time() - t, 1), sglang_wheel=wheel.name)
    except Exception as e:
        stage("2_install", False, sec=round(time.time() - t, 1), error=f"{type(e).__name__}: {e}"[:4000])
'''

C_CUDA = r'''# ==================================================================== CELL 5 — stage 3: torch + CUDA toolkit in the venv
if go():
    try:
        cu13 = sorted(Path(VENV).glob("lib/python*/site-packages/nvidia/cu13"))
        CUDA_HOME = str(cu13[0]) if cu13 else None
        env = dict(os.environ); env.pop("PYTHONPATH", None)
        cache = f"{PREFIX}/cache"; Path(cache).mkdir(parents=True, exist_ok=True)
        if CUDA_HOME:
            libdir = Path(CUDA_HOME) / "lib"; lib64 = Path(CUDA_HOME) / "lib64"
            if not lib64.exists() and not lib64.is_symlink(): lib64.symlink_to("lib")
            for so in sorted(libdir.glob("*.so.*")):
                link = libdir / re.sub(r"\.so\..*$", ".so", so.name)
                if not link.exists() and not link.is_symlink(): link.symlink_to(so.name)
            if not (libdir / "libcuda.so").exists():
                cands = []
                for d in ("/usr/local/nvidia/lib64", "/usr/local/nvidia/lib", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
                    cands += sorted(p for p in Path(d).glob("libcuda.so*") if p.is_file())
                if cands: (libdir / "libcuda.so").symlink_to(cands[0].resolve())
        CXX = next((c for c in ("g++-14", "g++-13", "g++-12", "g++-11", "g++") if shutil.which(c)), None)
        env.update({"PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                    "TORCH_CUDA_ARCH_LIST": "8.9",            # L4 (Ada); dfranzen used 12.0 (RTX PRO 6000)
                    "SGLANG_ENABLE_SM120_LOWM_BF16_GEMM": "0", "SGLANG_SM120_ONLINE_MXFP8": "0",
                    "SGLANG_SM120_LOWM_FP8_WEIGHT": "0", "SGLANG_SM120_LM_HEAD_FP8": "0",
                    "SGLANG_MAMBA_CONV_DTYPE": "bfloat16", "SGLANG_MM_PREPROCESS_DEVICE": "cpu",
                    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1", "TOKENIZERS_PARALLELISM": "false",
                    "OMP_NUM_THREADS": "8", "MAX_JOBS": "8",
                    "XDG_CACHE_HOME": cache, "TRITON_CACHE_DIR": f"{cache}/triton", "FLASHINFER_WORKSPACE_BASE": f"{cache}/flashinfer",
                    "SGLANG_CACHE_DIR": f"{cache}/sglang", "SGLANG_JIT_CACHE_DIR": f"{cache}/sglang/jit"})
        if CUDA_HOME:
            env.update({"CUDA_HOME": CUDA_HOME, "CUDACXX": f"{CUDA_HOME}/bin/nvcc",
                        "PATH": f"{CUDA_HOME}/bin:{VENV}/bin:" + env.get("PATH", ""),
                        "LD_LIBRARY_PATH": f"{CUDA_HOME}/lib64:{CUDA_HOME}/lib:" + env.get("LD_LIBRARY_PATH", "")})
        if CXX: env.update({"CC": CXX.replace("g++", "gcc"), "CXX": CXX, "CUDAHOSTCXX": CXX})
        SERVE_ENV = env
        probe = ("import torch, sglang, sgl_kernel, flashinfer, triton;"
                 "print('sglang', sglang.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, 'triton', triton.__version__);"
                 "assert torch.cuda.is_available() and torch.cuda.device_count() == 4, torch.cuda.device_count();"
                 "caps=[torch.cuda.get_device_capability(i) for i in range(4)]; print('caps', caps);"
                 "assert all(c == (8, 9) for c in caps), caps;"
                 "x=torch.randn(1024,1024,device='cuda'); y=x@x; torch.cuda.synchronize(); print('matmul ok', tuple(y.shape))")
        r = subprocess.run([PYTHON, "-c", probe], text=True, capture_output=True, env=env, timeout=clamp(600))
        need = ["--tensor-parallel-size", "--ep-size", "--quantization", "--ple-offload-embedding", "--language-model-only",
                "--mamba-backend", "--linear-attn-decode-backend", "--linear-attn-prefill-backend", "--disable-cuda-graph",
                "--max-mamba-cache-size", "--moe-runner-backend"]
        h = subprocess.run([SGLANG, "serve", "--help"], text=True, capture_output=True, env=env, timeout=clamp(300))
        help_text = h.stdout + h.stderr
        missing = [f for f in need if f not in help_text]
        stage("3_preflight", r.returncode == 0 and not missing, cuda_home=CUDA_HOME, cxx=CXX,
              probe_output=(r.stdout + r.stderr)[-3000:], missing_flags=missing, help_exit=h.returncode)
    except Exception as e:
        stage("3_torch_cuda", False, error=f"{type(e).__name__}: {e}"[:3000])
'''

C_LAUNCH = r'''# ==================================================================== CELL 6 — stage 4: launch the server (ordered attempts)
LOG = OUT / "serve.log"
SERVER = None
def base_args():
    # minimal first-boot profile (review): add one variable at a time later (context, KV dtype, concurrency, ...)
    return [SGLANG, "serve", "--model-path", MODEL_DIR, "--load-format", "safetensors",
            "--served-model-name", "flashnext", "--host", "127.0.0.1", "--port", str(PORT),
            "--dtype", "bfloat16", "--quantization", "auto-round", "--kv-cache-dtype", "auto",
            "--mem-fraction-static", "0.80", "--context-length", "8192", "--page-size", "64",
            "--max-running-requests", "1", "--chunked-prefill-size", "2048", "--max-prefill-tokens", "4096",
            "--mamba-ssm-dtype", "bfloat16", "--max-mamba-cache-size", "4",
            "--ple-offload-embedding", "--language-model-only", "--trust-remote-code",
            "--reasoning-parser", "qwen3", "--watchdog-timeout", "1800", "--disable-cuda-graph",
            "--tensor-parallel-size", "4", "--ep-size", "4",
            "--mamba-backend", "triton", "--linear-attn-decode-backend", "triton", "--linear-attn-prefill-backend", "triton"]
# Not tried: FlashInfer GDN (gdn_flashinfer.py requires SM90+); DP-attention (the PLE would be replicated on every rank,
# 4 x 95 GB pinned host RAM); plain TP4 (experts split to 160 columns, not a multiple of the INT4 group 128).
ATTEMPTS = [
    ("tp4_ep4_triton_moe_auto",   ["--moe-runner-backend", "auto"]),
    ("tp4_ep4_triton_moe_triton", ["--moe-runner-backend", "triton"]),
]
def kill_server():
    global SERVER
    if SERVER is not None and SERVER.poll() is None:
        try: os.killpg(SERVER.pid, 15); SERVER.wait(timeout=60)
        except Exception:
            try: os.killpg(SERVER.pid, 9)
            except Exception: pass
    SERVER = None   # only the tracked process group is killed; nothing else on the machine

if go():
    results = []
    for name, extra in ATTEMPTS:
        if remaining() < 10 * 60 + SAFETY_SEC:   # a load needs >= ~10 min; never start one that cannot finish
            results.append({"attempt": name, "skipped": "smoke budget", "remaining_sec": round(remaining())}); break
        args = base_args() + extra
        off = LOG.stat().st_size if LOG.exists() else 0
        with open(LOG, "ab", buffering=0) as lf:
            lf.write(f"\n\n===== ATTEMPT {name}: {shlex.join(map(str, args))}\n".encode())
            SERVER = subprocess.Popen(list(map(str, args)), env=SERVE_ENV, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        t = time.time(); ready = False
        attempt_limit = clamp(SERVER_ATTEMPT_TIMEOUT)
        while time.time() - t < attempt_limit:
            if SERVER.poll() is not None: break
            try:
                import urllib.request
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                    if r.status == 200: ready = True; break
            except Exception: pass
            time.sleep(5)
        tail = LOG.read_bytes()[off:][-6000:].decode(errors="replace")
        results.append({"attempt": name, "ready": ready, "sec": round(time.time() - t, 1),
                        "exit_code": SERVER.poll(), "log_tail": tail})
        print(f"[launch] {name}: ready={ready} after {time.time() - t:.0f}s exit={SERVER.poll()}", flush=True)
        if ready:
            LAUNCHED = name; break
        kill_server()
    ok = any(r.get("ready") for r in results)
    stage("4_launch", ok, attempts=results, gpu_mem=sh("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader"))
'''

C_ARC = r'''# ==================================================================== CELL 7 — stage 5: ARC test on 4 evaluation tasks
import urllib.request, concurrent.futures as cf
TEST_TASKS = ["4c3d4a41", "2d0172a1", "332f06d7", "c4d067a0"]   # 2 NVARC solves, 2 NVARC misses (all in the EVO12 subset)
def grid_text(g): return "\n".join(" ".join(str(v) for v in r) for r in g)
def puzzle_text(task):
    parts = ["## Training examples"]
    for i, ex in enumerate(task["train"]):
        parts += [f"### Training example {i}", "Input:", grid_text(ex["input"]), "Output:", grid_text(ex["output"])]
    parts.append("## Test")
    for i, ex in enumerate(task["test"]):
        parts += [f"### Test {i}", "Input:", grid_text(ex["input"])]
    return "\n\n".join(parts)
FORMAT = ("Find the transformation that maps every training input to its output. Answer in exactly this format:\n"
          "## Rule\n<the transformation in a few plain sentences>\n## Code\n```python\ndef solution(grid):\n    ...\n```\n"
          "`grid` is a list of lists of ints 0-9; return a new list of lists of ints 0-9. Standard library and numpy only. "
          "Everything the program uses must be defined in the code block.")
VERIFY = r"""
import json, sys, copy
p = json.load(sys.stdin); ns = {}
try:
    exec(p["code"], ns, ns); fn = ns["solution"]
except BaseException as e:
    print(json.dumps({"load_error": type(e).__name__ + ": " + str(e)[:300]})); sys.exit()
def run(g):
    try:
        out = fn(copy.deepcopy(g))
        try:
            import numpy as np
            if isinstance(out, np.ndarray): out = out.tolist()
        except Exception: pass
        return [list(map(int, r)) for r in out]
    except BaseException as e:
        return "ERR " + type(e).__name__
print(json.dumps({"train": [run(ex["input"]) for ex in p["train"]], "test": [run(t["input"]) for t in p["test"]]}))
"""
def verify(code, task):
    r = subprocess.run([sys.executable, "-c", VERIFY], input=json.dumps({"code": code, "train": task["train"], "test": task["test"]}),
                       text=True, capture_output=True, timeout=60)
    try: return json.loads(r.stdout.strip().splitlines()[-1])
    except Exception: return {"load_error": (r.stderr or r.stdout)[-300:]}
def ask(tid, task):
    body = {"model": "flashnext", "messages": [{"role": "user", "content": puzzle_text(task) + "\n\n" + FORMAT}],
            "max_tokens": 16384, "temperature": 0.7, "top_p": 0.95, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": True}}
    t = time.time()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req, timeout=clamp(3600)).read())
    sec = time.time() - t
    ch = resp["choices"][0]; msg = ch["message"]; usage = resp.get("usage", {})
    content = msg.get("content") or ""; reasoning = msg.get("reasoning_content") or ""
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", content, re.S)
    code = next((b for b in reversed(blocks) if "def solution" in b), None)
    v = verify(code, task) if code else {}
    sol = SOLUTIONS[tid]
    train_exact = [o == ex["output"] for o, ex in zip(v.get("train", []), task["train"])] if v.get("train") else []
    test_correct = [o == s for o, s in zip(v.get("test", []), sol)] if v.get("test") else []
    return {"task": tid, "sec": round(sec, 1), "finish": ch.get("finish_reason"), "completion_tokens": usage.get("completion_tokens"),
            "tok_per_sec": round((usage.get("completion_tokens") or 0) / max(sec, 1e-6), 1),
            "reasoning_chars": len(reasoning), "content_chars": len(content), "code": bool(code),
            "load_error": v.get("load_error"), "train_exact": f"{sum(train_exact)}/{len(task['train'])}",
            "all_train_exact": bool(train_exact) and all(train_exact), "test_correct": bool(test_correct) and all(test_correct),
            "rule": (re.search(r"##\s*Rule\s*\n(.*?)(?:\n##|```)", content, re.S) or [None, ""])[1][:600] if content else "",
            "content_tail": content[-1500:], "reasoning_tail": reasoning[-1500:]}
def safe_ask(tid):
    try:
        return ask(tid, EVAL[tid])
    except Exception as e:
        return {"task": tid, "error": f"{type(e).__name__}: {e}"[:1000], "completion_tokens": 0, "test_correct": False,
                "all_train_exact": False}
if go() and LOAD_ONLY:
    # load check only: one short generation proves the loaded model actually produces tokens
    try:
        body = {"model": "flashnext", "messages": [{"role": "user", "content": "Reply with the numbers 1 to 20 separated by spaces."}],
                "max_tokens": 64, "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}
        t = time.time()
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=clamp(600)).read())
        sec = time.time() - t; toks = (resp.get("usage") or {}).get("completion_tokens") or 0
        text = resp["choices"][0]["message"].get("content") or ""
        stage("5_generate_probe", bool(text.strip()), sec=round(sec, 1), completion_tokens=toks,
              tok_per_sec=round(toks / max(sec, 1e-6), 1), text=text[:300])
    except Exception as e:
        stage("5_generate_probe", False, error=f"{type(e).__name__}: {e}"[:2000])
elif go():
    try:
        EVAL = json.load(open(Path(COMP_DIR) / "arc-agi_evaluation_challenges.json"))
        SOLUTIONS = json.load(open(Path(COMP_DIR) / "arc-agi_evaluation_solutions.json"))
        t = time.time()
        with cf.ThreadPoolExecutor(max_workers=4) as ex:
            rows = list(ex.map(safe_ask, TEST_TASKS))
        total_tok = sum(r["completion_tokens"] or 0 for r in rows)
        stage("5_arc_test", True, wall_sec=round(time.time() - t, 1), aggregate_tok_per_sec=round(total_tok / max(time.time() - t, 1e-6), 1),
              solved=sum(r["test_correct"] for r in rows), train_exact_tasks=sum(r["all_train_exact"] for r in rows), rows=rows)
    except Exception as e:
        stage("5_arc_test", False, error=f"{type(e).__name__}: {e}"[:3000], traceback=traceback.format_exc()[-3000:])
'''

C_REPORT = r'''# ==================================================================== CELL 8 — report, cleanup, bundle (always runs)
try: kill_server()
except Exception: pass
REPORT["total_sec"] = round(time.time() - T0, 1)
(OUT / "fn_smoke_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
print("=" * 100)
_expected = ["0_environment", "1_inputs", "2_install", "3_preflight", "4_launch",
             "5_generate_probe" if LOAD_ONLY else "5_arc_test"]
_all_ok = REPORT["stopped_at"] is None and all(REPORT["stages"].get(k, {}).get("ok") for k in _expected)
if not RERUN and REPORT["stopped_at"] is None and not _all_ok:
    REPORT["stopped_at"] = "incomplete_unknown"   # never claim success for stages that did not run
    (OUT / "fn_smoke_report.json").write_text(json.dumps(REPORT, indent=2, default=str))
print("STOPPED AT:", REPORT["stopped_at"] or "none (all stages passed)")
print("VERDICT:", "LOADS AND GENERATES ON 4xL4" if _all_ok else ("RERUN FALLBACK" if RERUN else "NOT SHOWN TO WORK - see stopped_at"))
for k, v in REPORT["stages"].items():
    print(f"  {k}: {'PASS' if v['ok'] else 'FAIL'} at {v['t_sec']}s")
if "5_arc_test" in REPORT["stages"] and REPORT["stages"]["5_arc_test"]["ok"]:
    s = REPORT["stages"]["5_arc_test"]
    print("ARC test:", s["solved"], "/ 4 solved |", s["train_exact_tasks"], "train-exact |", s["aggregate_tok_per_sec"], "tok/s aggregate")
    for r in s["rows"]:
        print("  ", {k: r.get(k) for k in ("task", "test_correct", "train_exact", "code", "completion_tokens", "tok_per_sec", "finish", "error")})
ZIP = WORK / "FN_SMOKE_BUNDLE.zip"
with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob("*")):
        if p.is_file() and p.stat().st_size < 200 * 1024**2: z.write(p, p.relative_to(WORK).as_posix())
    if (WORK / "submission.json").exists(): z.write(WORK / "submission.json", "submission.json")
print("bundle:", ZIP, round(ZIP.stat().st_size / 1e6, 2), "MB | total", REPORT["total_sec"], "s")
'''

cells = [{"cell_type": "markdown", "metadata": {}, "source": C_HEAD.splitlines(keepends=True)}]
for src in (C_CFG, C_ENV, C_INPUTS, C_INSTALL, C_CUDA, C_LAUNCH, C_ARC, C_REPORT):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)})
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
DST.write_bytes((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
print("wrote", DST, hashlib.sha256(DST.read_bytes()).hexdigest())
