"""E12-K1 builder: the FP8 kernel probe notebook for Qwen3.8-27B FP8 on Kaggle 4xL4 (vLLM 0.19.0, R5's exact engine).

Finding behind it (R5 log, 2026-10-16): every main block-FP8 GEMM shape of our model on L4 runs vLLM's Triton
w8a8 block-FP8 kernel with the DEFAULT config ("Using default W8A8 Block FP8 kernel config. Performance might be
sub-optimal!"). FP8 Marlin (weight-only, block-128 supported) is disabled on sm89 unless VLLM_TEST_FORCE_FP8_MARLIN=1.

The notebook, with no change to the weights:
  stage 1  install vLLM 0.19.0 from R5's wheelhouse (R5's code, copied) + libcuda shim (R5's code, copied)
  stage 2  tune the 5 GEMM shapes on the 4 GPUs in parallel (k1_tune_block_fp8.py, vLLM's own kernel)
  stage 3  engine benchmark, one fresh process per variant: default / tuned configs / FP8 Marlin (k1_engine_bench.py)
  stage 4  report: decode tok/s short + long context, MTP acceptance, kernel-path evidence from the logs, greedy agreement
Outputs: /kaggle/working/k1/ (results.json, report.md, configs_l4/*.json ready to ship, all logs).
usage: python build_e12_k1.py
"""
import hashlib, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R5 = (Path(r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Notebooks")
      / "Q27-XHIGH-v1_4_4_E10_86K_POETIQ_R5_EVAL120.ipynb")
OUT_NB = HERE.parent / "E12_K1_fp8_kernel_probe_on_KAGGLE.ipynb"

r5 = json.loads(R5.read_bytes())
r5_src = ["".join(c["source"]) for c in r5["cells"]]


def between(text, start, end, where):
    i = text.find(start); j = text.find(end, i)
    if i < 0 or j < 0 or text.count(start) != 1:
        sys.exit(f"marker not found once in {where}: {start[:60]!r}")
    return text[i:j]


INSTALL = between(r5_src[3], "import importlib, importlib.metadata", "from transformers import AutoTokenizer", "R5 cell 3")
SHIM = between(r5_src[6], "import shutil as _shutil", 'os.environ.setdefault("MAX_JOBS", "4")', "R5 cell 6")
TUNER = (HERE / "k1_tune_block_fp8.py").read_text(encoding="utf-8")
BENCH = (HERE / "k1_engine_bench.py").read_text(encoding="utf-8")

MD = """# E12-K1: FP8 kernel probe for Qwen3.8-27B on 4×L4 (no weight change)

**Why:** R5's log shows that all 5 main block-FP8 GEMM shapes of our model run vLLM's Triton kernel with the **default config** on L4 ("Performance might be sub-optimal!"). vLLM 0.19 has no L4 config files for them. FP8 Marlin (weight-only, block-128 supported) is off on sm89 unless `VLLM_TEST_FORCE_FP8_MARLIN=1`.

**Stages:**
1. Install vLLM 0.19.0 from R5's wheelhouse, plus R5's libcuda shim. The code is copied from R5.
2. Tune the 5 shapes in parallel on the 4 GPUs with vLLM's own kernel. Decode sizes go first, under a time budget, and each best config is checked against the default output.
3. Engine benchmark, one fresh process per variant, all with R5's exact engine settings (TP4, fp8 KV, MTP-3, FLASHINFER, prefix caching, 16 seqs):
   - **default**;
   - **tuned** (L4 config files copied into vLLM);
   - **marlin** (`VLLM_TEST_FORCE_FP8_MARLIN=1`).
4. Report:
   - decode tok/s at about 2–10k and about 36k context;
   - MTP acceptance and preemptions;
   - which kernel path the logs show;
   - greedy token agreement against default.

**Run:** 4×L4, Internet OFF, the pinned Q27 Docker image. Attach the FP8 model, R5's vLLM wheelhouse and the competition data. Expected time: about 1.5–2 h; a guard stops new stages after `K1_BUDGET_SEC` (default 4 h).

**Outputs:** `/kaggle/working/k1/`:
- `results.json`, `report.md`;
- `configs_l4/*.json`: tuned configs, ready to ship;
- `timings_*.json`, `tune_gpu*.log`, `bench_*.json` / `.log`.

This is not a submission.
"""

CELL1 = r'''# ==================================================================== CELL 1 — setup (stdlib only)
import os, sys, json, time, subprocess, shutil, glob, importlib.util
from pathlib import Path
T0 = time.time()
DRYRUN = os.environ.get("K1_DRYRUN") == "1"          # local CPU orchestration test only
WORKING = Path(os.environ.get("K1_WORKING", "/kaggle/working")); OUT = WORKING / "k1"; OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = Path(os.environ.get("Q27X_MODEL", "/kaggle/input/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/pytorch/hf-fp8/1"))
WHEELHOUSE = Path(os.environ.get("Q27X_WHEELHOUSE", "/kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"))
ARC_ROOT = Path(os.environ.get("Q27X_ARC_ROOT", "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"))
if not ARC_ROOT.exists():
    for _alt in ("/kaggle/input/arc-prize-2026-arc-agi-2",):
        if Path(_alt).exists():
            ARC_ROOT = Path(_alt)
os.environ["Q27X_MODEL"] = str(MODEL_PATH); os.environ["Q27X_ARC_ROOT"] = str(ARC_ROOT)
TIME_BUDGET_SEC = float(os.environ.get("K1_BUDGET_SEC", 4 * 3600))      # no new stage after this
TUNE_SEC = float(os.environ.get("K1_TUNE_SEC", 40 * 60))                 # per-GPU tuning budget
BENCH_TIMEOUT_SEC = float(os.environ.get("K1_BENCH_TIMEOUT_SEC", 45 * 60))
# (N, K) per TP4 rank: the 5 shapes of R5's "Using default W8A8 Block FP8 kernel config" warnings
SHAPES = [(8704, 5120), (5120, 4352), (5120, 1536), (4096, 5120), (3584, 5120)]
GPU_SHAPES = [[(8704, 5120)], [(5120, 4352), (5120, 1536)], [(4096, 5120)], [(3584, 5120)]]   # balanced by size
RESULTS = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "dryrun": DRYRUN, "stages": {}}

def save_results():
    (OUT / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str), encoding="utf-8")

def time_left():
    return TIME_BUDGET_SEC - (time.time() - T0)

def sh(cmd, timeout=60):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as e:
        return f"ERR {e!r}"

print("K1 setup | dryrun", DRYRUN, "| model", MODEL_PATH, "| arc", ARC_ROOT, "| out", OUT)
RESULTS["nvidia_smi"] = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
print(RESULTS["nvidia_smi"]); save_results()
'''

CELL2 = r'''# ==================================================================== CELL 2 — offline install (R5's code) + libcuda shim (R5's code)
if DRYRUN:
    os.environ["Q27X_SKIP_INSTALL"] = "1"
__INSTALL__
if not DRYRUN:
__SHIM_INDENTED__
    os.environ.setdefault("MAX_JOBS", "4")
    CUDA_LINK = ensure_libcuda_linkable()
else:
    CUDA_LINK = {"dryrun": True}
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "1"
RESULTS["stages"]["install"] = {"cuda_link": CUDA_LINK, "sec": round(time.time() - T0, 1)}
print("CUDA/FlashInfer linker preflight:", json.dumps(CUDA_LINK)); save_results()
'''

CELL3 = r'''# ==================================================================== CELL 3 — stage 2: tune the 5 block-FP8 shapes (4 GPUs in parallel)
TUNER_SRC = __TUNER_SRC__
(OUT / "k1_tune_block_fp8.py").write_text(TUNER_SRC, encoding="utf-8")
TUNE_DIR = OUT / "tune"; TUNE_DIR.mkdir(exist_ok=True)
_t = time.time(); _procs = []
_budget = max(5 * 60, min(TUNE_SEC, time_left() - 75 * 60))     # keep at least 75 min for the engine benchmarks
_deadline = time.time() + _budget
for _gpu, _shapes in enumerate(GPU_SHAPES):
    _env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(_gpu))
    _log = open(OUT / f"tune_gpu{_gpu}.log", "w")
    _cmd = [sys.executable, str(OUT / "k1_tune_block_fp8.py"), "--shapes", ",".join(f"{n}x{k}" for n, k in _shapes),
            "--out-dir", str(TUNE_DIR), "--deadline", str(_deadline), "--label", f"gpu{_gpu}"]
    _procs.append((_gpu, subprocess.Popen(_cmd, env=_env, stdout=_log, stderr=subprocess.STDOUT), _log))
_rc = {}
for _gpu, _p, _log in _procs:
    try:
        _rc[_gpu] = _p.wait(timeout=max(60, _deadline - time.time() + 10 * 60))
    except subprocess.TimeoutExpired:
        _p.kill(); _rc[_gpu] = "killed_after_deadline"
    _log.close()
CONFIG_DIR = OUT / "configs_l4"; CONFIG_DIR.mkdir(exist_ok=True)
for _f in TUNE_DIR.glob("N=*,K=*,device_name=*.json"):
    shutil.copy(_f, CONFIG_DIR / _f.name)
TUNE_TIMINGS = {}
for _f in sorted(TUNE_DIR.glob("timings_*.json")):
    _d = json.loads(_f.read_text()); TUNE_TIMINGS[f"N={_d['N']},K={_d['K']}"] = _d
RESULTS["stages"]["tune"] = {"sec": round(time.time() - _t, 1), "budget_sec": round(_budget), "returncodes": _rc,
                             "config_files": sorted(p.name for p in CONFIG_DIR.glob("*.json")), "timings": TUNE_TIMINGS}
for _k, _d in TUNE_TIMINGS.items():
    print(_k, "complete" if _d.get("complete") else "PARTIAL", {m: v["speedup"] for m, v in _d["timings"].items()})
save_results()
'''

CELL4 = r'''# ==================================================================== CELL 4 — stage 3: engine benchmark per variant (fresh process each)
BENCH_SRC = __BENCH_SRC__
(OUT / "k1_engine_bench.py").write_text(BENCH_SRC, encoding="utf-8")
if DRYRUN:
    VLLM_DIR = Path(os.environ["K1_VLLM_DIR"])
else:
    VLLM_DIR = Path(importlib.util.find_spec("vllm").submodule_search_locations[0])
VLLM_CFG_DIR = VLLM_DIR / "model_executor" / "layers" / "quantization" / "utils" / "configs"
OUR_CFGS = sorted(CONFIG_DIR.glob("*.json"))

def set_configs(install):
    for f in OUR_CFGS:
        dst = VLLM_CFG_DIR / f.name
        if install:
            shutil.copy(f, dst)
        elif dst.exists():
            dst.unlink()
    return sorted(p.name for p in VLLM_CFG_DIR.glob("*device_name=NVIDIA_L4*fp8_w8a8*"))

def kill_children():
    try:
        import psutil
        for k in psutil.Process(os.getpid()).children(recursive=True):
            try: k.kill()
            except Exception: pass
    except Exception:
        pass

def wait_gpu_free(max_sec=180):
    if DRYRUN:
        return True
    t = time.time()
    while time.time() - t < max_sec:
        try:
            used = [int(x) for x in sh(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).split()]
            if used and max(used) < 1024:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False

def log_evidence(text):
    return {"default_block_fp8_warnings": text.count("Using default W8A8 Block FP8 kernel config"),
            "tuned_config_used": text.count("for W8A8 Block FP8 kernel"),
            "marlin_mentions": text.count("Marlin"),
            "selected_lines": sorted({l.strip()[-110:] for l in text.splitlines() if "Selected" in l and "Kernel" in l})[:6]}

VARIANTS = [("default", False, {}), ("tuned", True, {}), ("marlin", False, {"VLLM_TEST_FORCE_FP8_MARLIN": "1"})]
BENCH = {}
for _name, _install, _extra in VARIANTS:
    if _name == "tuned" and not OUR_CFGS:
        BENCH[_name] = {"skipped": "no tuned config files"}; continue
    if time_left() < 25 * 60:
        BENCH[_name] = {"skipped": f"time guard ({round(time_left() / 60)} min left)"}; continue
    _installed = set_configs(_install)
    _env = dict(os.environ, **_extra)
    _out = OUT / f"bench_{_name}.json"; _logp = OUT / f"bench_{_name}.log"
    _t = time.time()
    with open(_logp, "w") as _log:
        try:
            _rc = subprocess.run([sys.executable, str(OUT / "k1_engine_bench.py"), "--variant", _name, "--out", str(_out)],
                                 env=_env, stdout=_log, stderr=subprocess.STDOUT, timeout=BENCH_TIMEOUT_SEC).returncode
        except subprocess.TimeoutExpired:
            _rc = "timeout"
    kill_children(); _free = wait_gpu_free()
    _res = json.loads(_out.read_text()) if _out.exists() else {"errors": ["no result file"]}
    _res.update(returncode=_rc, sec=round(time.time() - _t, 1), configs_in_vllm=_installed, gpu_released=_free,
                log_evidence=log_evidence(_logp.read_text(errors="replace")))
    BENCH[_name] = _res
    print(_name, "rc", _rc, "| load", _res.get("load_sec"), "| short tok/s", (_res.get("workloads", {}).get("short") or {}).get("tok_s"),
          "| long tok/s", (_res.get("workloads", {}).get("long") or {}).get("tok_s"), "| evidence", _res["log_evidence"])
    RESULTS["stages"]["bench"] = BENCH; save_results()
set_configs(False)            # leave vLLM as it was
RESULTS["stages"]["bench"] = BENCH; save_results()
'''

CELL5 = r'''# ==================================================================== CELL 5 — stage 4: report
def _w(v, w, k):
    return ((BENCH.get(v) or {}).get("workloads") or {}).get(w, {}).get(k)

def _prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y: break
        n += 1
    return n

base = {w: _w("default", w, "tok_s") for w in ("short", "long")}
rows, summary = [], {}
for v in ("default", "tuned", "marlin"):
    b = BENCH.get(v) or {}
    if "skipped" in b:
        rows.append(f"| {v} | skipped: {b['skipped']} |||||||"); continue
    sp = {w: (round(_w(v, w, "tok_s") / base[w], 3) if base.get(w) and _w(v, w, "tok_s") else None) for w in ("short", "long")}
    g0, g1 = (BENCH.get("default") or {}).get("greedy_ids") or [], b.get("greedy_ids") or []
    agree = [_prefix(x, y) for x, y in zip(g0, g1)] if v != "default" else None
    summary[v] = {"short_tok_s": _w(v, "short", "tok_s"), "long_tok_s": _w(v, "long", "tok_s"), "speedup_vs_default": sp,
                  "accept_short": _w(v, "short", "accept_rate"), "accept_long": _w(v, "long", "accept_rate"),
                  "preemptions": (_w(v, "short", "preemptions") or 0) + (_w(v, "long", "preemptions") or 0),
                  "greedy_common_prefix_vs_default": agree, "load_sec": b.get("load_sec"), "errors": b.get("errors"),
                  "log_evidence": b.get("log_evidence")}
    rows.append(f"| {v} | {b.get('load_sec')} | {_w(v, 'short', 'tok_s')} | {sp['short']} | {_w(v, 'long', 'tok_s')} | {sp['long']} | "
                f"{_w(v, 'short', 'accept_rate')} / {_w(v, 'long', 'accept_rate')} | {agree} |")
RESULTS["summary"] = summary
tune_rows = []
for k, d in (RESULTS["stages"].get("tune", {}).get("timings") or {}).items():
    t = d.get("timings", {})
    tune_rows.append(f"| {k} | " + " | ".join(str((t.get(str(m)) or {}).get("speedup")) for m in (16, 64, 128, 512, 4096)) +
                     f" | {'yes' if d.get('complete') else 'partial'} |")
md = ["# E12-K1 FP8 kernel probe: results", "",
      f"Started {RESULTS['started_utc']}, total {round((time.time() - T0) / 60, 1)} min, dryrun {DRYRUN}.", "",
      "## Engine decode speed (R5 engine settings; 16 requests)", "",
      "| Variant | Load s | Short tok/s (2–10k ctx) | × default | Long tok/s (~36k ctx) | × default | MTP accept short / long | Greedy common prefix vs default (4 × 192) |",
      "|---|---|---|---|---|---|---|---|"] + rows + ["",
      "Kernel-path evidence from each log (default warnings / tuned config used / Marlin mentions):", ""] + \
     [f"- {v}: {json.dumps((BENCH.get(v) or {}).get('log_evidence'))}" for v in ("default", "tuned", "marlin")] + ["",
      "## Kernel tuning speedup over the default config (per GEMM shape, at M tokens)", "",
      "| Shape | M=16 | M=64 | M=128 | M=512 | M=4096 | Complete |", "|---|---|---|---|---|---|---|"] + tune_rows + ["",
      "Tuned config files (ship these): `k1/configs_l4/`. Copy them into vLLM's "
      "`model_executor/layers/quantization/utils/configs/` before the engine starts."]
(OUT / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
RESULTS["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()); save_results()
print("\n".join(md))
'''


def indent(text, n=4):
    return "".join((" " * n + line) if line.strip() else line for line in text.splitlines(keepends=True))


cells_src = [
    CELL1,
    CELL2.replace("__INSTALL__", INSTALL.rstrip() + "\n").replace("__SHIM_INDENTED__\n", indent(SHIM)),
    CELL3.replace("__TUNER_SRC__", repr(TUNER)),
    CELL4.replace("__BENCH_SRC__", repr(BENCH)),
    CELL5,
]
for i, s in enumerate(cells_src):
    if "__" in s and ("__INSTALL__" in s or "__SHIM" in s or "__TUNER_SRC__" in s or "__BENCH_SRC__" in s):
        sys.exit(f"placeholder left in cell {i + 1}")
    compile(s, f"<cell{i + 1}>", "exec")
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": MD.splitlines(keepends=True)}] +
      [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)}
       for s in cells_src],
      "metadata": r5.get("metadata", {}), "nbformat": 4, "nbformat_minor": 5}
data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
OUT_NB.write_bytes(data)
print("BUILT", OUT_NB.name, len(data), "bytes sha256", hashlib.sha256(data).hexdigest())
