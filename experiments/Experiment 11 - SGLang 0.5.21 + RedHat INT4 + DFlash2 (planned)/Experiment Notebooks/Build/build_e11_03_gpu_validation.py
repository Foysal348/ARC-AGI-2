"""Experiment 11 Stage 2: GPU validation notebook (Claude, 2026-10-15, user go "ওকে বানানো শুরু করো"). Deterministic builder.

Builds E11_03_sglang_int4_gpu_validation_on_KAGGLE.ipynb (4x L4, Internet OFF is fine). It does not solve the competition;
it answers "does SGLang 0.5.21 + RedHat INT4 (+ MTP / + DFlash2) run on Kaggle's L4s, how fast, and is the INT4 output sane":

  cell 1  venv from the E11-02 wheelhouse + CUDA link shims (E11-02 probe: no libcudart.so / lib64; E10: no libcuda.so)
  cell 2  GPU preflight: torch GEMM, sgl_kernel, NCCL TP4 all-reduce, Triton, FlashInfer decode (head_dim 256 = our
          model at TP4), nvcc JIT torch extension, CuTe DSL import. A fatal failure skips every server stage.
  cell 3  prompts: E10's exact prompt (identical chat template, xhigh), tokenized inside the venv (transformers 5.12.1)
  cell 4  server helpers (launch / ready / stop, /generate, decode-log parsing)
  cell 5  A INT4 | B INT4 + MTP (NEXTN, 3 steps, 4 draft tokens) | C INT4 + DFlash2 (block 8, unquant draft):
          smoke, greedy prefix match vs A, concurrency sweep 1-32 x 4,096 tokens, long run 16 x 24,576 tokens
  cell 6  D = the faster of B / C with the SSM state in float16 (sweep + greedy vs its float32 run)
  cell 7  sanity solve: 8 cheap puzzles FP8 solved in nearly every run, full E10 DIRECT protocol on the faster of B / C
  cell 8  report.md + results.json (written after every stage, so a stopped run keeps what it measured)

Local CPU harness: E11_HARNESS=1 + E11_FAKE_SERVER=<fake server script> skips install / GPU and drives a fake server.
usage: python build_e11_03_gpu_validation.py
"""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent
NAME = "E11_03_sglang_int4_gpu_validation_on_KAGGLE.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


M0 = """
# E11-03 · SGLang 0.5.21 + RedHat Qwen3.8-27B INT4 (+ MTP / + DFlash2) · GPU validation on 4x L4

**Settings:** Accelerator **GPU L4 x4** · Internet can stay **OFF** (everything is offline) · no Secrets needed.
**Attach:** dataset `foysalemonshanto/sglang-0-5-21-cu13-wheelhouse`, models `foysalemonshanto/qwen3-8-27b-int4-redhat`
(w4a16-g128) and `foysalemonshanto/qwen3-8-27b-dflash2` (bf16), competition `arc-prize-2026-arc-agi-2`.
The models are private on foysalemonshanto: run this from that account (or share the models first).

**This is not a submission.** It answers four questions before any production notebook moves to this stack:
1. Does the stack run on Kaggle's L4s? (preflight: GEMM, sgl_kernel, NCCL TP4, Triton, FlashInfer, nvcc JIT)
2. How fast are **A** INT4, **B** INT4 + MTP and **C** INT4 + DFlash2, from 1 to 32 concurrent requests, on our real
   ARC prompts (E10 prompt, xhigh)? Also KV capacity, speculative acceptance, and a long 16 x 24k-token run.
3. Is speculative decoding lossless here? (greedy output of B / C vs A, token by token)
4. Is INT4 output sane? 8 cheap puzzles that FP8 solved in nearly every run, full E10 DIRECT protocol.

Plus **D**: the faster of B / C with the SSM state in float16 instead of the native float32 (speed + drift only).

Expected time about 2.5-3 h; a time guard stops starting new stages after 5 h. Every stage writes
`/kaggle/working/e11_03/results.json`; `report.md` and every server log are in the same folder.
"""

C0 = r'''
# ==================================================================== CELL 0 — configuration, inputs, results file
import os, sys, json, time, re, shutil, signal, subprocess, random, hashlib, statistics
import urllib.request, urllib.error
import concurrent.futures as cf
from pathlib import Path
from datetime import datetime, timezone

T0 = time.time()
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
HARNESS = os.environ.get("E11_HARNESS") == "1"         # local CPU harness only: fake server, no install, no GPU
INPUT = Path(os.environ.get("E11_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("E11_OUT", "/kaggle/working/e11_03")); OUT.mkdir(parents=True, exist_ok=True)
TIME_BUDGET_SEC = float(os.environ.get("E11_TIME_BUDGET", 5.0 * 3600))   # no new stage starts after this

SGLANG_VERSION = "0.5.21"
PORT = 30000
CONTEXT_LEN = 155648            # rev 3b: = L66 (122,880 thinking + 8,192 + prompt); E10's 77,824 cap also fits
MAX_RUNNING = 32
LOAD_TIMEOUT_SEC = 40 * 60      # first launch includes weight load, CUDA-graph capture and any JIT
SWEEP_LEVELS = [1, 4, 8, 12, 16, 24, 32]   # rev 3: 12 added (D66/L66: 12 slots suit long-thinking Poetiq)
SWEEP_TOKENS = 4096             # per request; ARC thinking almost never ends this early
LONG_CONCURRENCY, LONG_TOKENS = 12, 24576       # rev 3b: 12 = production slots (D66/L66); saves ~25 % of the long stage
GREEDY_N, GREEDY_TOKENS = 4, 512
THINK_BUDGET, ANSWER_MAX_TOKENS, CONTEXT_SAFETY = 77824, 8192, 256          # = E10 DIRECT
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}                 # = E10 DIRECT
STOP_THINKING_TEXT = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
SEED = 20261015
# 8 cheap tier-A puzzles with one test input; FP8 solved them in (nearly) every run. (p_hat, FP8 ORDER2 thinking tokens)
SANITY_TASKS = {"2ba387bc": (0.907, 5355), "bf45cf4b": (0.907, 17485), "58f5dbd5": (0.838, 22674),
                "136b0064": (0.834, 23164), "31f7f899": (0.764, 25425), "1818057f": (0.725, 26891),
                "e8686506": (0.892, 27730), "7491f3cf": (0.870, 31874)}
SANITY_CONCURRENCY = 8
SANITY_TIMEOUT_SEC = 60 * 60


def _json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def find_wheelhouse():
    for p in sorted(INPUT.rglob("BUILD_INFO.json")):
        try:
            info = _json(p)
        except Exception:
            continue
        if info.get("sglang") == SGLANG_VERSION and (p.parent / "requirements.lock").is_file():
            return p.parent
    return None


def find_model(pred):
    for p in sorted(INPUT.rglob("config.json")):
        try:
            cfg = _json(p)
        except Exception:
            continue
        if isinstance(cfg, dict) and pred(p.parent, cfg):
            return p.parent
    return None


WH = find_wheelhouse()
INT4 = find_model(lambda d, c: (c.get("quantization_config") or {}).get("quant_method") == "compressed-tensors"
                  and (d / "model_mtp.safetensors").is_file())
DFLASH = find_model(lambda d, c: c.get("architectures") == ["DFlash2DraftModel"])
ARC_ROOT = next((p.parent for p in sorted(INPUT.rglob("arc-agi_evaluation_challenges.json"))), None)
INPUTS = {"wheelhouse": WH, "int4": INT4, "dflash2": DFLASH, "arc": ARC_ROOT}
print(json.dumps({k: str(v) for k, v in INPUTS.items()}, indent=1))
_missing = [k for k, v in INPUTS.items() if v is None]
if _missing:
    raise SystemExit(f"attach the missing inputs: {_missing}")

RESULTS = {"run_id": RUN_ID, "inputs": {k: str(v) for k, v in INPUTS.items()}, "env": {}, "preflight": None,
           "configs": {}, "best_spec": None, "sanity": None, "notes": []}


def save_results():
    tmp = OUT / "results.json.tmp"
    tmp.write_text(json.dumps(RESULTS, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, OUT / "results.json")


def time_left():
    return TIME_BUDGET_SEC - (time.time() - T0)


def stamp(msg):
    print(f"[{time.strftime('%H:%M:%S')} +{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


save_results()
'''

C1 = r'''
# ==================================================================== CELL 1 — SGLang venv from the wheelhouse + CUDA link shims
# Install = the E11-02 test: a venv (so .pth files run and console scripts such as ninja are on PATH), the image's pip,
# --no-deps from the lock. The image's own torch / vLLM stay untouched.
VENV = Path("/tmp/sglang-venv")
VPY = VENV / "bin" / "python"
SHIMS = {}

# Kaggle drops '+' from dataset file names: flashinfer_jit_cache-0.6.18+cu130-...whl is stored as ...-0.6.18cu130-...whl,
# which pip cannot match to the lock pin. Rebuild the original names (SHA256SUMS keeps them) as links in a find-links dir
# and check the hash of every renamed wheel.
FIND_LINKS = (OUT / "harness_wheels") if HARNESS else Path("/tmp/e11-wheels")
FIND_LINKS.mkdir(parents=True, exist_ok=True)
_sums = [l.split("  ", 1) for l in (WH / "SHA256SUMS").read_text(encoding="utf-8").splitlines() if l.strip()]
RENAMED = []
for _h, _name in _sums:
    _name = _name.strip()
    if not _name.endswith(".whl"):
        continue
    _src = WH / _name if (WH / _name).exists() else WH / _name.replace("+", "")
    if not _src.exists():
        raise SystemExit(f"wheel missing from the dataset: {_name}")
    if _src.name != _name:
        _d = hashlib.sha256()
        with open(_src, "rb") as _f:
            for _b in iter(lambda: _f.read(16 * 1024 * 1024), b""):
                _d.update(_b)
        if _d.hexdigest() != _h:
            raise SystemExit(f"sha256 mismatch for {_src.name}")
        RENAMED.append(f"{_src.name} -> {_name}")
    _dst = FIND_LINKS / _name
    if not _dst.exists():
        try:
            _dst.symlink_to(_src)
        except OSError:                                 # harness on Windows without symlink rights
            shutil.copy(_src, _dst)
SHIMS["renamed_wheels"] = RENAMED
print(f"find-links {FIND_LINKS}: {len(list(FIND_LINKS.glob('*.whl')))} wheels | restored names: {RENAMED}")

if HARNESS:
    VPY = Path(sys.executable)
    CUDA_HOME = OUT / "harness_cuda_home"
    LIBCUDA_SHIM = OUT / "harness_libcuda"
else:
    _t = time.time()
    if not VPY.exists():
        subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(VENV)], check=True)
        _r = subprocess.run([sys.executable, "-m", "pip", "--python", str(VPY), "install", "--no-index", "--find-links", str(FIND_LINKS),
                             "--requirement", str(WH / "requirements.lock"), "--no-deps", "--only-binary", ":all:",
                             "--no-compile", "--disable-pip-version-check", "--no-warn-conflicts", "-q"],
                            capture_output=True, text=True)
        if _r.returncode != 0:
            print(_r.stdout[-2000:], _r.stderr[-4000:])
            raise SystemExit("offline install failed")
    SITE = Path(subprocess.run([str(VPY), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                               capture_output=True, text=True, check=True).stdout.strip())
    CUDA_HOME = SITE / "nvidia" / "cu13"
    stamp(f"venv ready in {time.time() - _t:.0f}s | site {SITE}")

    # E11-02 BUILD_PROBE: the pip CUDA runtime ships lib/libcudart.so.13 only -> -lcudart (torch / FlashInfer JIT links
    # with -L$CUDA_HOME/lib64 -lcudart) needs lib/libcudart.so and lib64
    if not (CUDA_HOME / "lib" / "libcudart.so").exists():
        (CUDA_HOME / "lib" / "libcudart.so").symlink_to("libcudart.so.13")
        SHIMS["libcudart.so"] = "created"
    if not (CUDA_HOME / "lib64").exists():
        (CUDA_HOME / "lib64").symlink_to("lib")
        SHIMS["lib64"] = "created"

    # E10 run 1 (2026-10-05): FlashInfer's JIT link failed with `ld: cannot find -lcuda` (Kaggle has only libcuda.so.1)
    LIBCUDA_SHIM = Path("/tmp/e11-libcuda")
    _have = [d for d in ("/usr/local/cuda/lib64", "/usr/local/cuda/lib64/stubs", "/usr/lib/x86_64-linux-gnu", "/usr/lib64")
             if Path(d, "libcuda.so").exists()]
    if _have:
        SHIMS["libcuda.so"] = f"exists in {_have[0]}"
    else:
        _real = None
        try:
            for _line in subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=30).stdout.splitlines():
                if "libcuda.so.1" in _line and "=>" in _line:
                    _real = _line.split("=>")[-1].strip()
                    break
        except Exception:
            pass
        for _d in ("/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib64"):
            if _real is None and Path(_d, "libcuda.so.1").exists():
                _real = str(Path(_d, "libcuda.so.1"))
        if _real:
            LIBCUDA_SHIM.mkdir(parents=True, exist_ok=True)
            _link = LIBCUDA_SHIM / "libcuda.so"
            if _link.is_symlink() or _link.exists():
                _link.unlink()
            _link.symlink_to(Path(_real).resolve())
            SHIMS["libcuda.so"] = f"shim -> {_real}"
        else:
            SHIMS["libcuda.so"] = "libcuda.so.1 NOT FOUND"

# every child (preflight, tokenizer, server) gets this environment; the notebook kernel itself never imports torch
ENV = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_HOME", "CUDA_PATH")}
ENV.update({
    "PATH": os.pathsep.join([str(VENV / "bin"), str(CUDA_HOME / "bin"), os.environ.get("PATH", "")]),
    "CUDA_HOME": str(CUDA_HOME), "CUDA_PATH": str(CUDA_HOME),
    "LIBRARY_PATH": os.pathsep.join(x for x in (str(LIBCUDA_SHIM), str(CUDA_HOME / "lib"), os.environ.get("LIBRARY_PATH", "")) if x),
    "LD_LIBRARY_PATH": os.pathsep.join(x for x in (str(CUDA_HOME / "lib"), os.environ.get("LD_LIBRARY_PATH", "")) if x),
    "TORCH_CUDA_ARCH_LIST": "8.9", "MAX_JOBS": "4",
    "SGLANG_CACHE_DIR": "/tmp/e11-cache/sglang", "CUDA_CACHE_PATH": "/tmp/e11-cache/cuda",     # fresh caches (issue #42917)
    "TRITON_CACHE_DIR": "/tmp/e11-cache/triton", "TORCH_EXTENSIONS_DIR": "/tmp/e11-cache/torch_ext",
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "PYTHONUNBUFFERED": "1",
})
RESULTS["env"] = {"shims": SHIMS, "cuda_home": str(CUDA_HOME), "venv_python": str(VPY)}
print(json.dumps(RESULTS["env"], indent=1))
save_results()
'''

C2 = r'''
# ==================================================================== CELL 2 — GPU preflight (inside the venv, before any server)
PREFLIGHT_SRC = r"""
import json, os, sys, time, traceback

R = {}


def step(name, fn, fatal=False):
    t = time.time()
    try:
        R[name] = {"ok": True, "fatal": fatal, "out": fn(), "sec": round(time.time() - t, 1)}
    except Exception as e:
        R[name] = {"ok": False, "fatal": fatal, "err": f"{type(e).__name__}: {str(e)[:1500]}",
                   "tb": traceback.format_exc()[-2500:], "sec": round(time.time() - t, 1)}
    print(name, "OK" if R[name]["ok"] else "FAIL", R[name]["sec"], "s", flush=True)


def t_versions():
    import importlib.metadata as md
    out = {}
    for p in ("torch", "sglang", "sglang-kernel", "flashinfer-python", "flashinfer-jit-cache", "transformers", "triton",
              "nvidia-cuda-nvcc", "nvidia-cuda-runtime", "nvidia-cutlass-dsl", "humming-kernels"):
        try:
            out[p] = md.version(p)
        except Exception as e:
            out[p] = f"missing ({type(e).__name__})"
    return out


def t_cuda():
    import torch
    n = torch.cuda.device_count()
    if n < 4:
        raise RuntimeError(f"{n} GPUs visible, need 4")
    a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    c = a @ b
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(10):
        c = a @ b
    torch.cuda.synchronize()
    dt = (time.time() - t) / 10
    ref = a.float() @ b.float()
    return {"torch": torch.__version__, "cuda": torch.version.cuda, "n_gpu": n,
            "names": [torch.cuda.get_device_name(i) for i in range(n)],
            "capability": [list(torch.cuda.get_device_capability(i)) for i in range(n)],
            "bf16_gemm_tflops": round(2 * 4096 ** 3 / dt / 1e12, 1),
            "rel_err": float((c.float() - ref).abs().max() / ref.abs().max())}


def t_sgl_kernel():
    import torch, sgl_kernel
    torch.cuda.init()
    return {"file": sgl_kernel.__file__, "ops": len([o for o in dir(torch.ops.sgl_kernel) if not o.startswith("_")])}


def _ar_worker(rank, world, port, q):
    import torch, torch.distributed as dist
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    x = torch.full((16 * 1024 * 1024,), float(rank + 1), device="cuda")
    dist.all_reduce(x)
    torch.cuda.synchronize()
    ok = bool((x == float(sum(range(1, world + 1)))).all().item())
    y = torch.ones(16 * 1024 * 1024, device="cuda")
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(10):
        dist.all_reduce(y)
    torch.cuda.synchronize()
    q.put({"rank": rank, "ok": ok, "ms_per_64MB": round((time.time() - t) / 10 * 1000, 2)})
    dist.destroy_process_group()


def t_nccl():
    import torch, torch.multiprocessing as mp
    world = torch.cuda.device_count()
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_ar_worker, args=(r, world, 29517, q)) for r in range(world)]
    for p in procs:
        p.start()
    out = [q.get(timeout=600) for _ in procs]
    for p in procs:
        p.join(timeout=120)
    if not all(o["ok"] for o in out):
        raise RuntimeError(f"all-reduce result wrong: {out}")
    return sorted(out, key=lambda o: o["rank"])


def t_triton():
    import torch, triton, triton.language as tl

    @triton.jit
    def _add(x_ptr, y_ptr, o_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m) + tl.load(y_ptr + offs, mask=m), mask=m)

    x = torch.randn(100000, device="cuda")
    y = torch.randn(100000, device="cuda")
    o = torch.empty_like(x)
    _add[(triton.cdiv(x.numel(), 1024),)](x, y, o, x.numel(), BLOCK=1024)
    torch.cuda.synchronize()
    return {"triton": triton.__version__, "max_err": float((o - (x + y)).abs().max())}


def t_flashinfer():
    import torch, flashinfer
    torch.manual_seed(0)
    q = torch.randn(6, 256, device="cuda", dtype=torch.bfloat16)          # our model at TP4: 6 query heads, 1 KV head, d=256
    k = torch.randn(4096, 1, 256, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(4096, 1, 256, device="cuda", dtype=torch.bfloat16)

    def ref(kk, vv):
        s = (q.float() @ kk[:, 0].float().T) / 16.0
        return s.softmax(-1) @ vv[:, 0].float()

    o = flashinfer.single_decode_with_kv_cache(q, k, v)
    out = {"flashinfer": flashinfer.__version__, "bf16_max_abs_err": float((o.float() - ref(k, v)).abs().max())}
    try:
        k8, v8 = k.to(torch.float8_e4m3fn), v.to(torch.float8_e4m3fn)
        o8 = flashinfer.single_decode_with_kv_cache(q, k8, v8)
        out["fp8_kv_max_abs_err"] = float((o8.float() - ref(k8, v8)).abs().max())
    except Exception as e:
        out["fp8_kv_error"] = f"{type(e).__name__}: {str(e)[:400]}"
    return out


def t_jit():
    import torch
    from torch.utils.cpp_extension import load_inline
    cuda_src = (
        "__global__ void e11_twice_k(float* x, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] *= 2.f; }\n"
        "torch::Tensor twice(torch::Tensor x) { int n = x.numel(); e11_twice_k<<<(n + 255) / 256, 256>>>(x.data_ptr<float>(), n); return x; }\n"
    )
    bdir = os.path.join(os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp"), "e11_twice")
    os.makedirs(bdir, exist_ok=True)
    m = load_inline("e11_twice", cpp_sources="torch::Tensor twice(torch::Tensor x);", cuda_sources=cuda_src,
                    functions=["twice"], build_directory=bdir, verbose=False)
    x = torch.ones(1000, device="cuda")
    m.twice(x)
    torch.cuda.synchronize()
    s = float(x.sum())
    if s != 2000.0:
        raise RuntimeError(f"kernel result {s}, expected 2000")
    return {"sum": s}


def t_cutlass():
    import cutlass
    import cutlass.cute  # noqa: F401
    return {"cutlass": getattr(cutlass, "__version__", "imported")}


if __name__ == "__main__":
    step("versions", t_versions)
    step("torch_cuda_gemm", t_cuda, fatal=True)
    step("sgl_kernel_import", t_sgl_kernel, fatal=True)
    step("nccl_allreduce_tp4", t_nccl, fatal=True)
    step("triton_kernel", t_triton)
    step("flashinfer_decode_hd256", t_flashinfer)
    step("nvcc_jit_torch_extension", t_jit)
    step("cutlass_dsl_import", t_cutlass)
    print("PREFLIGHT_JSON " + json.dumps(R, default=str))
"""

_smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout if shutil.which("nvidia-smi") else ""
print(_smi[:1500])
_m = re.search(r"Driver Version:\s*([0-9.]+)\s+CUDA Version:\s*([0-9.]+)", _smi)
PREFLIGHT = {"driver": _m.group(1) if _m else None, "driver_cuda": _m.group(2) if _m else None}
if HARNESS:
    PREFLIGHT["steps"] = {"harness": {"ok": True, "fatal": True}}
else:
    _pf = Path("/tmp/e11_preflight.py")
    _pf.write_text(PREFLIGHT_SRC, encoding="utf-8")
    _t = time.time()
    _r = subprocess.run([str(VPY), str(_pf)], env=ENV, capture_output=True, text=True, timeout=2400, cwd="/tmp")
    (OUT / "preflight_stdout.log").write_text(_r.stdout + "\n--- stderr ---\n" + _r.stderr, encoding="utf-8")
    _line = next((l for l in _r.stdout.splitlines() if l.startswith("PREFLIGHT_JSON ")), None)
    PREFLIGHT["steps"] = json.loads(_line[15:]) if _line else {"crash": {"ok": False, "fatal": True, "err": _r.stderr[-3000:]}}
    PREFLIGHT["sec"] = round(time.time() - _t, 1)
PREFLIGHT_OK = all(s.get("ok") for s in PREFLIGHT["steps"].values() if s.get("fatal"))
RESULTS["preflight"] = PREFLIGHT
save_results()
for _k, _s in PREFLIGHT["steps"].items():
    print(f"{_k:28} {'OK  ' if _s.get('ok') else 'FAIL'} {'(fatal)' if _s.get('fatal') else '       '} "
          f"{json.dumps(_s.get('out', _s.get('err')), default=str)[:300]}")
stamp(f"PREFLIGHT {'PASS' if PREFLIGHT_OK else 'FAIL - every server stage is skipped'}")
'''

C3 = r'''
# ==================================================================== CELL 3 — tasks + prompts (E10's prompt, tokenized in the venv)
_CH = _json(ARC_ROOT / "arc-agi_evaluation_challenges.json")
SOLUTIONS = _json(ARC_ROOT / "arc-agi_evaluation_solutions.json")
assert all(t in _CH for t in SANITY_TASKS), "a sanity puzzle is missing from the evaluation file"
_pool = sorted(t for t in _CH if t not in SANITY_TASKS)
SWEEP_UNITS = [f"{t}#0" for t in random.Random(SEED).sample(_pool, max(SWEEP_LEVELS))]
GREEDY_UNITS = SWEEP_UNITS[:GREEDY_N]
SANITY_UNITS = [f"{t}#0" for t in SANITY_TASKS]

TOK_SRC = r"""
import hashlib, json, sys
from transformers import AutoTokenizer

cfg = json.load(open(sys.argv[1], encoding="utf-8"))
tok = AutoTokenizer.from_pretrained(cfg["model"])


# E10 cell 3, verbatim (russcore's LB-10.14 free-form prompt)
def render_grid(g):
    return "\n".join(" ".join(str(int(v)) for v in row) for row in g)


def build_user_message(train, test_input):
    parts = ["You are solving an abstract reasoning puzzle. Each example has an "
             "INPUT grid and its OUTPUT grid (digits 0-9 are colors; 0 is usually "
             "background). Infer the ONE rule that maps every input to its output, "
             "then apply it to the FINAL INPUT.\n"]
    for i, p in enumerate(train):
        parts.append("Example %d INPUT:\n%s\nExample %d OUTPUT:\n%s\n" %
                     (i + 1, render_grid(p["input"]), i + 1, render_grid(p["output"])))
    parts.append("FINAL INPUT:\n%s\n" % render_grid(test_input))
    parts.append("Reason step by step however you like. Then give ONLY the final "
                 "output grid as rows of space-separated digits inside a fenced "
                 "block:\n```grid\n<rows>\n```\nOutput nothing after the block.")
    return "\n".join(parts)


ch = json.load(open(cfg["challenges"], encoding="utf-8"))
units = {}
for uid in cfg["units"]:
    tid, qi = uid.split("#")
    task = ch[tid]
    msgs = [{"role": "user", "content": build_user_message(task["train"], task["test"][int(qi)]["input"])}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True,
                                   reasoning_effort="xhigh")
    units[uid] = tok.encode(text, add_special_tokens=False)
first = cfg["units"][0]
out = {"units": units, "think_close": tok.encode("</think>", add_special_tokens=False),
       "stop_ids": tok.encode(cfg["stop_text"], add_special_tokens=False),
       "smoke_ids": tok.encode("Reply with the digits 1 2 3 4 5 6 7 8:", add_special_tokens=False),
       "template_sha": hashlib.sha256((tok.chat_template or "").encode()).hexdigest()[:16],
       "first_prompt_tail": tok.decode(units[first][-16:])}
json.dump(out, open(cfg["out"], "w", encoding="utf-8"))
print("TOKENIZED", len(units))
"""
_ts, _tc, _to = Path("/tmp/e11_tokenize.py"), OUT / "tokenize_cfg.json", OUT / "prompts.json"
if HARNESS:
    _ts = OUT / "e11_tokenize.py"
_ts.write_text(TOK_SRC, encoding="utf-8")
_tc.write_text(json.dumps({"model": str(INT4), "challenges": str(ARC_ROOT / "arc-agi_evaluation_challenges.json"),
                           "units": SWEEP_UNITS + SANITY_UNITS, "stop_text": STOP_THINKING_TEXT, "out": str(_to)}),
               encoding="utf-8")
_tok_py = os.environ.get("E11_TOKENIZER_PYTHON", str(VPY))            # harness: a local venv with transformers 5.12.1
_r = subprocess.run([_tok_py, str(_ts), str(_tc)], env=ENV, capture_output=True, text=True, timeout=1800)
if _r.returncode != 0:
    print(_r.stdout[-2000:], _r.stderr[-4000:])
    raise SystemExit("tokenization failed")
_P = _json(_to)
PROMPT = _P["units"]
THINK_CLOSE, STOP_IDS, SMOKE_IDS = _P["think_close"], _P["stop_ids"], _P["smoke_ids"]
assert len(THINK_CLOSE) == 1 and THINK_CLOSE[0] in STOP_IDS, (THINK_CLOSE, STOP_IDS)
_gen = _json(INT4 / "generation_config.json")
EOS_IDS = set(_gen["eos_token_id"] if isinstance(_gen["eos_token_id"], list) else [_gen["eos_token_id"]])
_pl = sorted(len(PROMPT[u]) for u in SWEEP_UNITS)
RESULTS["prompts"] = {"template_sha": _P["template_sha"], "first_prompt_tail": _P["first_prompt_tail"],
                      "sweep_prompt_tokens": {"min": _pl[0], "median": _pl[len(_pl) // 2], "max": _pl[-1]},
                      "sweep_units": SWEEP_UNITS, "sanity_units": SANITY_UNITS, "think_close": THINK_CLOSE,
                      "eos": sorted(EOS_IDS)}
save_results()
print(json.dumps(RESULTS["prompts"], indent=1)[:1500])
'''

C4 = r'''
# ==================================================================== CELL 4 — server helpers, workloads, E10 grid parser
LAUNCH = [str(VPY), "-m", "sglang.launch_server"] if not HARNESS else [sys.executable, os.environ["E11_FAKE_SERVER"]]
OPTIONAL_FLAGS = {"--cuda-graph-max-bs-decode": str(MAX_RUNNING), "--watchdog-timeout": "1800",
                  "--decode-log-interval": "40"}
_HELP = None


def server_help():
    global _HELP
    if _HELP is None:
        r = subprocess.run(LAUNCH + ["--help"], env=ENV, capture_output=True, text=True, timeout=900)
        _HELP = r.stdout + r.stderr
    return _HELP


def base_flags(ssm_dtype="float32"):
    f = ["--model-path", str(INT4), "--tp-size", "4", "--kv-cache-dtype", "fp8_e4m3", "--context-length", str(CONTEXT_LEN),
         "--max-running-requests", str(MAX_RUNNING), "--chunked-prefill-size", "4096", "--mamba-ssm-dtype", ssm_dtype,
         "--enable-metrics", "--disable-flashinfer-autotune", "--random-seed", str(SEED),
         "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "info"]
    for k, v in OPTIONAL_FLAGS.items():
        if k in server_help():
            f += [k, v]
    return f


SPEC = {
    "A_int4": [],
    # MTP head (BF16, model_mtp.safetensors). Qwen3.5 is not in SGLang's auto-draft list -> pass the target path.
    "B_int4_mtp": ["--speculative-algorithm", "NEXTN", "--speculative-draft-model-path", str(INT4),
                   "--speculative-draft-model-quantization", "unquant", "--speculative-num-steps", "3",
                   "--speculative-eagle-topk", "1", "--speculative-num-draft-tokens", "4"],
    # DFlash2 card: block size 8; unquant is mandatory (issue #39087: the drafter inherits INT4 otherwise)
    "C_int4_dflash2": ["--speculative-algorithm", "DFLASH", "--speculative-draft-model-path", str(DFLASH),
                       "--speculative-draft-model-quantization", "unquant", "--speculative-num-draft-tokens", "8"],
}
LADDER = [[], ["--attention-backend", "triton"]]       # the 2nd launch only if the 1st fails
SERVER_KEYS = ["version", "max_total_num_tokens", "max_running_requests", "attention_backend", "mamba_radix_cache_strategy",
               "page_size", "mem_fraction_static", "kv_cache_dtype", "chunked_prefill_size", "mamba_ssm_dtype",
               "speculative_algorithm", "speculative_num_draft_tokens", "speculative_num_steps",
               "speculative_draft_attention_backend", "disable_overlap_schedule", "cuda_graph_max_bs_decode",
               "max_mamba_cache_size", "internal_states"]


def http(path, payload=None, timeout=7200):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data, method="GET" if data is None else "POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8", "replace")
    try:
        return json.loads(body)
    except ValueError:
        return body


def generate(ids, max_new, sampling, seed=None, stop_ids=None):
    sp = {"max_new_tokens": int(max_new), "skip_special_tokens": False, "no_stop_trim": True, **sampling}
    if seed is not None:
        sp["sampling_seed"] = int(seed)
    if stop_ids:
        sp["stop_token_ids"] = [int(x) for x in stop_ids]
    t = time.time()
    out = http("/generate", {"input_ids": [int(x) for x in ids], "sampling_params": sp})
    if not isinstance(out, dict):
        raise RuntimeError(f"/generate returned {str(out)[:200]}")
    out["_wall"] = time.time() - t
    return out


def gpu_used_mb():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return [int(x) for x in out.split()]
    except Exception:
        return []


def wait_gpu_free(max_sec=180):
    t = time.time()
    while time.time() - t < max_sec:
        u = gpu_used_mb()
        if HARNESS or (u and max(u) < 1024):
            return {"free": True, "sec": round(time.time() - t), "used_mb": u}
        time.sleep(5)
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "sglang::"], capture_output=True)
    time.sleep(15)
    return {"free": False, "sec": round(time.time() - t), "used_mb": gpu_used_mb(), "pkill": True}


class Server:
    def __init__(self, name, flags):
        self.name, self.flags, self.proc, self.log_f = name, flags, None, None
        self.cmd = LAUNCH + flags

    def start(self, tag):
        self.log_path = OUT / f"server_{self.name}_{tag}.log"
        self.log_f = open(self.log_path, "w", encoding="utf-8")
        self.t_start = time.time()
        self.proc = subprocess.Popen(self.cmd, env=ENV, stdout=self.log_f, stderr=subprocess.STDOUT, cwd="/tmp" if not HARNESS else None,
                                     start_new_session=True)

    def log_size(self):
        try:
            return self.log_path.stat().st_size
        except Exception:
            return 0

    def log_text(self, offset=0):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as f:
                f.seek(offset)
                return f.read()
        except Exception:
            return ""

    def wait_ready(self, timeout):
        while time.time() - self.t_start < timeout:
            if self.proc.poll() is not None:
                return False, f"server exited rc={self.proc.returncode}"
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=10).read()
                return True, f"ready in {time.time() - self.t_start:.0f}s"
            except Exception:
                time.sleep(5)
        return False, "timeout"

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            for sig, wait in ((signal.SIGTERM, 60), (getattr(signal, "SIGKILL", signal.SIGTERM), 30)):
                try:
                    if hasattr(os, "killpg"):
                        os.killpg(self.proc.pid, sig)          # the whole group: TP workers, detokenizer, scheduler
                    else:
                        self.proc.kill()
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=wait)
                    break
                except Exception:
                    continue
        if self.log_f:
            self.log_f.close()
            self.log_f = None
        return wait_gpu_free()


_STARTUP_KEYS = ("max_total_num_tokens", "KV Cache is allocated", "Mamba Cache", "Load weight end", "Capture cuda graph end",
                 "max_running_requests", "avail mem", "speculative", "attention backend", "Using ")


def startup_lines(text):
    return [l[:400] for l in text.splitlines() if any(k in l for k in _STARTUP_KEYS)][:60]


_KV_RE = re.compile(r"([A-Za-z#][A-Za-z#_ ()/\-]*?):\s*(-?\d+(?:\.\d+)?)")


def decode_rows(text):
    rows = []
    for line in text.splitlines():
        if "Decode batch" in line:
            rows.append({k.strip(): float(v) for k, v in _KV_RE.findall(line.split("Decode batch", 1)[1])})
    return rows


def summarize_decode(rows, level=None):
    if not rows:
        return {"n_lines": 0}
    full = [r for r in rows if level is None or r.get("#running-req") == level]   # rev 3b: no fallback to other levels

    def med(key, rs):
        v = [r[key] for r in rs if key in r]
        return round(statistics.median(v), 3) if v else None

    usage = [v for r in rows for k, v in r.items() if "usage" in k]
    return {"n_lines": len(rows), "n_lines_at_level": len(full),
            "gen_tok_s_median": med("gen throughput (token/s)", full),
            "gen_tok_s_max": max((r.get("gen throughput (token/s)", 0) for r in rows), default=None),
            "accept_len_median": med("accept len", full), "running_max": max(r.get("#running-req", 0) for r in rows),
            "usage_max": max(usage) if usage else None}


def run_batch(ids_list, max_new, sampling, seeds, stop_ids=None):
    t0 = time.time()
    res = [None] * len(ids_list)
    with cf.ThreadPoolExecutor(max_workers=len(ids_list)) as ex:
        futs = {ex.submit(generate, ids, max_new, sampling, seeds[i], stop_ids): i for i, ids in enumerate(ids_list)}
        for f in cf.as_completed(futs):
            try:
                res[futs[f]] = f.result()
            except Exception as e:
                res[futs[f]] = {"error": f"{type(e).__name__}: {str(e)[:300]}"}
    return res, time.time() - t0


def batch_stats(res, wall):
    ok = [r for r in res if r and "error" not in r]
    meta = [r.get("meta_info", {}) for r in ok]
    toks = sum(int(m.get("completion_tokens", 0)) for m in meta)
    acc = [float(m["spec_accept_length"]) for m in meta if m.get("spec_accept_length") is not None]
    per = [int(m.get("completion_tokens", 0)) / r["_wall"] for m, r in zip(meta, ok) if r.get("_wall")]
    return {"n": len(res), "ok": len(ok), "errors": [r["error"] for r in res if r and "error" in r][:3],
            "wall_s": round(wall, 1), "completion_tokens": toks, "agg_tok_s": round(toks / wall, 1) if wall > 0 else None,
            "per_req_tok_s_e2e_median": round(statistics.median(per), 1) if per else None,
            "accept_len_mean": round(sum(acc) / len(acc), 3) if acc else None,
            "prompt_tokens": sum(int(m.get("prompt_tokens", 0)) for m in meta),
            "cached_tokens": sum(int(m.get("cached_tokens", 0)) for m in meta),
            "finish": sorted({str((m.get("finish_reason") or {}).get("type")) for m in meta})}


def load_level(srv, name, c, max_new, tag):
    units = [SWEEP_UNITS[i % len(SWEEP_UNITS)] for i in range(c)]
    off = srv.log_size()
    res, wall = run_batch([PROMPT[u] for u in units], max_new, SAMPLING, [SEED + 1000 * c + i for i in range(c)])
    st = batch_stats(res, wall)
    rows = decode_rows(srv.log_text(off))
    st["decode_log"] = summarize_decode(rows, level=c)
    if tag == "long":
        (OUT / f"decode_rows_{name}_long.json").write_text(json.dumps(rows), encoding="utf-8")
    stamp(f"{name} {tag} c={c}: {st['agg_tok_s']} tok/s total | log gen {st['decode_log'].get('gen_tok_s_median')} tok/s | "
          f"accept {st['accept_len_mean']} | wall {st['wall_s']}s | ok {st['ok']}/{c}")
    return st


def sweep(srv, name):
    out = {}
    for c in SWEEP_LEVELS:
        if time_left() < 20 * 60:
            out[str(c)] = {"skipped": "time budget"}
            break
        out[str(c)] = load_level(srv, name, c, SWEEP_TOKENS, "sweep")
        save_results()
        if out[str(c)]["ok"] < c:
            break
    return out


def greedy_outputs():
    return {u: list(generate(PROMPT[u], GREEDY_TOKENS, {"temperature": 0.0}).get("output_ids", [])) for u in GREEDY_UNITS}


def prefix_match(ref, got):
    rows = {}
    for u in GREEDY_UNITS:
        a, b = ref.get(u, []), got.get(u, [])
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        rows[u] = {"common_prefix": n, "len_ref": len(a), "len": len(b), "identical": bool(a) and a == b}   # rev 3b: empty != identical
    return {"units": rows, "identical": sum(r["identical"] for r in rows.values()),
            "mean_prefix": round(sum(r["common_prefix"] for r in rows.values()) / max(1, len(rows)), 1)}


# ---- E10 cell 4, verbatim: answer text -> grid
def valid_grid(g):
    return (isinstance(g, list) and 1 <= len(g) <= 30 and all(isinstance(r, list) and 1 <= len(r) <= 30 for r in g)
            and len({len(r) for r in g}) == 1
            and all(isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 9 for r in g for v in r))


_FENCE_RE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
_SEP_ROW = re.compile(r"\[?\s*\d(?:\s*[, ]\s*\d)*\s*\]?,?")
_JSON_ROW = re.compile(r"\[\s*\d(?:\s*,\s*\d)*\s*\]")


def _parse_row(s):
    s = s.strip()
    if not s:
        return None
    if re.fullmatch(r"\d+", s):
        return [int(c) for c in s]
    if _SEP_ROW.fullmatch(s):
        return [int(c) for c in re.findall(r"\d", s)]
    return None


def _first_row_run(lines):
    rows = []
    for line in lines:
        r = _parse_row(line)
        if r is None:
            if rows:
                break
            continue
        rows.append(r)
    return rows if valid_grid(rows) else None


def _grid_from_obj(obj):
    if isinstance(obj, dict):
        for k in ("output", "grid", "test_output", "answer", "test_outputs"):
            if k in obj:
                return _grid_from_obj(obj[k])
        return None
    if valid_grid(obj):
        return obj
    if isinstance(obj, list) and obj and valid_grid(obj[-1]):
        return obj[-1]
    return None


def _json_rows_grid(b):
    grids, cur = [], []
    for m in _JSON_ROW.finditer(b):
        cur.append(json.loads(m.group(0)))
        if b[m.end():].lstrip().startswith("]"):
            grids.append(cur)
            cur = []
    if cur:
        grids.append(cur)
    for g in reversed(grids):
        if valid_grid(g):
            return g
    return None


def _grid_from_block(body):
    b = body.strip()
    if b[:1] in ("[", "{"):
        try:
            g = _grid_from_obj(json.loads(b))
        except Exception:
            g = None
        if g is None:
            g = _json_rows_grid(b)
        if g is not None:
            return g
    return _first_row_run(b.splitlines())


def parse_grid(text):
    for _lab, body in reversed(_FENCE_RE.findall(text or "")):
        g = _grid_from_block(body)
        if g is not None:
            return g
    runs, cur = [], []
    for line in (text or "").splitlines():
        r = _parse_row(line)
        if r is None:
            if cur:
                runs.append(cur)
                cur = []
        else:
            cur.append(r)
    if cur:
        runs.append(cur)
    for run in reversed(runs):
        if valid_grid(run):
            return run
    return None


assert parse_grid("Done.\n```grid\n1 0 2\n3 4 5\n```") == [[1, 0, 2], [3, 4, 5]] and parse_grid("no grid") is None


def solve_unit(uid, idx):
    """E10 DIRECT protocol: think up to the cap (stop at </think>); natural close -> answer; cap -> forced close + answer"""
    tid = uid.split("#")[0]
    ids = PROMPT[uid]
    cap = max(0, min(THINK_BUDGET, CONTEXT_LEN - len(ids) - ANSWER_MAX_TOKENS - CONTEXT_SAFETY))
    seed = SEED + 100 * idx
    t = time.time()
    r1 = generate(ids, cap, SAMPLING, seed, stop_ids=THINK_CLOSE)
    think = list(r1.get("output_ids", []))
    fin = r1.get("meta_info", {}).get("finish_reason") or {}
    if fin.get("type") == "stop" and (fin.get("matched") == THINK_CLOSE[0] or (think and think[-1] == THINK_CLOSE[0])):
        mode = "natural"
        if not think or think[-1] != THINK_CLOSE[0]:
            think.append(THINK_CLOSE[0])
        cont = ids + think
    else:
        mode = "forced"
        while think and think[-1] in EOS_IDS:
            think.pop()
        cont = ids + think + STOP_IDS
    r2 = generate(cont, ANSWER_MAX_TOKENS, SAMPLING, seed + 1)
    grid = parse_grid(r2.get("text", ""))
    m1, m2 = r1.get("meta_info", {}), r2.get("meta_info", {})
    return {"uid": uid, "mode": mode, "think_tokens": len(think), "answer_tokens": m2.get("completion_tokens"),
            "grid_parsed": grid is not None, "hit": int(grid is not None and grid == SOLUTIONS[tid][0]),
            "accept_len": m1.get("spec_accept_length"), "sec": round(time.time() - t, 1),
            "fp8_p_hat": SANITY_TASKS[tid][0], "fp8_think": SANITY_TASKS[tid][1], "answer_tail": r2.get("text", "")[-200:]}


def sanity(srv, name):
    t0 = time.time()
    res = {}
    ex = cf.ThreadPoolExecutor(max_workers=SANITY_CONCURRENCY)
    futs = {ex.submit(solve_unit, u, i): u for i, u in enumerate(SANITY_UNITS)}
    done, pending = cf.wait(futs, timeout=min(SANITY_TIMEOUT_SEC, max(60, time_left() - 10 * 60)))
    for f in done:
        try:
            res[futs[f]] = f.result()
        except Exception as e:
            res[futs[f]] = {"uid": futs[f], "error": f"{type(e).__name__}: {str(e)[:300]}", "hit": 0}
    for f in pending:
        res[futs[f]] = {"uid": futs[f], "error": "timeout", "hit": 0}
    ex.shutdown(wait=False, cancel_futures=True)
    rows = [res[u] for u in SANITY_UNITS]
    out = {"config": name, "hits": sum(r.get("hit", 0) for r in rows), "n": len(rows),
           "fp8_expected_hits": round(sum(SANITY_TASKS[u.split("#")[0]][0] for u in SANITY_UNITS), 2),
           "wall_s": round(time.time() - t0, 1), "rows": rows}
    stamp(f"SANITY {name}: {out['hits']}/{out['n']} solved (FP8 expectation {out['fp8_expected_hits']})")
    return out


EXTRA_STAGES = {}          # rev 3: name -> fn(srv, name); filled in cell 5b


def run_config(name, spec_flags, ssm_dtype="float32", stages=("greedy", "sweep", "long")):
    rec = {"name": name, "ssm_dtype": ssm_dtype, "spec_flags": spec_flags, "attempts": [], "ok": False}
    RESULTS["configs"][name] = rec
    save_results()
    for i, extra in enumerate(LADDER):
        if time_left() < 25 * 60:
            rec["skipped"] = "time budget"
            break
        srv = Server(name, base_flags(ssm_dtype) + spec_flags + extra)
        srv.start(f"try{i}")
        ok, why = srv.wait_ready(LOAD_TIMEOUT_SEC)
        att = {"try": i, "extra_flags": extra, "ready": ok, "why": why, "cmd": " ".join(srv.cmd)}
        rec["attempts"].append(att)
        if not ok:
            att["log_tail"] = srv.log_text()[-8000:]
            att["gpu_release"] = srv.stop()
            stamp(f"{name} launch {i} FAILED ({why}); log tail:\n{att['log_tail'][-1500:]}")
            save_results()
            continue
        stamp(f"{name} launch {i}: {why}")
        try:
            rec["startup"] = startup_lines(srv.log_text())
            info = http("/get_server_info", timeout=120)
            (OUT / f"server_info_{name}.json").write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
            if isinstance(info, dict):
                rec["server"] = {k: info.get(k) for k in SERVER_KEYS if k in info}
            sm = generate(SMOKE_IDS, 16, {"temperature": 0.0})
            rec["smoke"] = {"tokens": len(sm.get("output_ids", [])), "text": str(sm.get("text"))[:120]}
            if not rec["smoke"]["tokens"]:
                raise RuntimeError("smoke decode produced no tokens")
            if "greedy" in stages:
                rec["greedy_ids"] = greedy_outputs()
            if "sweep" in stages:
                rec["sweep"] = sweep(srv, name)
            if "long" in stages and time_left() > 45 * 60:
                rec["long"] = load_level(srv, name, LONG_CONCURRENCY, LONG_TOKENS, "long")
            if "sanity" in stages:
                rec["sanity"] = sanity(srv, name)
            for _xs, _xfn in EXTRA_STAGES.items():          # rev 3: E11 extra stages (filled in cell 5b)
                if _xs in stages:
                    if time_left() < 15 * 60:
                        rec[_xs] = {"skipped": "time budget"}
                        continue
                    try:
                        rec[_xs] = _xfn(srv, name)
                    except Exception as e:
                        rec[_xs] = {"error": f"{type(e).__name__}: {str(e)[:400]}"}
                    stamp(f"{name} {_xs}: {json.dumps(rec[_xs], default=str)[:300]}")
                    save_results()
            try:
                rec["internal_states_end"] = http("/get_server_info", timeout=120).get("internal_states")
            except Exception:
                pass
            try:
                (OUT / f"metrics_{name}.txt").write_text(str(http("/metrics", timeout=120)), encoding="utf-8")
            except Exception:
                pass
            rec["ok"] = True
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:800]}"
            rec["log_tail"] = srv.log_text()[-8000:]
            stamp(f"{name} FAILED during the workload: {rec['error']}")
        rec["gpu_release"] = srv.stop()
        save_results()
        return rec
    save_results()
    return rec


print("server helpers ready | launch:", " ".join(LAUNCH))
'''

C5 = r'''
# ==================================================================== CELL 5 — A INT4 | B INT4 + MTP | C INT4 + DFlash2
if not PREFLIGHT_OK:
    RESULTS["notes"].append("preflight failed: server stages skipped")
else:
    for _name in ("A_int4", "B_int4_mtp", "C_int4_dflash2"):
        run_config(_name, SPEC[_name])
    _ref = RESULTS["configs"].get("A_int4", {}).get("greedy_ids")
    for _name in ("B_int4_mtp", "C_int4_dflash2"):
        _got = RESULTS["configs"].get(_name, {}).get("greedy_ids")
        if _ref and _got:
            RESULTS["configs"][_name]["greedy_vs_A"] = prefix_match(_ref, _got)


def level_tok_s(rec, level="16"):
    return ((rec or {}).get("sweep") or {}).get(level, {}).get("agg_tok_s") or 0


def _complete(rec, level="12"):
    # rev 3b (GPT audits): a config competes only if every request at the production level succeeded
    st = ((rec or {}).get("sweep") or {}).get(level) or {}
    return bool((rec or {}).get("ok")) and st.get("ok") == int(level) and not st.get("errors")


# rev 3b: A (INT4 alone) competes too; chosen at 12 requests = the production slot count (D66/L66)
_cands = [(level_tok_s(RESULTS["configs"].get(n), "12"), n) for n in ("A_int4", "B_int4_mtp", "C_int4_dflash2")
          if _complete(RESULTS["configs"].get(n))]
RESULTS["best_spec"] = max(_cands)[1] if _cands and max(_cands)[0] > 0 else None
save_results()
stamp(f"best config (total tok/s at 12 requests, complete sweeps only): {RESULTS['best_spec']} {sorted(_cands, reverse=True)}")
'''

C5B = r'''
# ==================================================================== CELL 5b — rev 3 extra tests on the best config
# 1 contcache : forced-close continuation (prompt + 6,144 thinking tokens + stop text) must hit the prefix cache on a
#               hybrid model (mamba radix cache); compared with a cold request of the same length (2nd token changed)
# 2 longctx   : decode speed / KV with ~90k-token contexts (E10 thinking runs to 78k-123k)
# 3 contam    : the same 4 greedy prompts alone vs inside a 16-request batch (cross-request isolation / batch drift)
# 4 temp      : 12 x 4,096 tokens at T 1.0 vs T 0.6 (Poetiq uses 1.0; D66/L66 vLLM data hinted 0.6 is faster)
CONT_THINK, LONGCTX_TOKENS, LONGCTX_DECODE = 6144, 90000, 1024
T10 = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}


def x_contcache(srv, name):
    units = GREEDY_UNITS
    r1, w1 = run_batch([PROMPT[u] for u in units], CONT_THINK, SAMPLING, [SEED + 7 + i for i in range(len(units))])
    rows = []
    for u, r in zip(units, r1):
        if not r or "error" in r:
            rows.append({"unit": u, "error": (r or {}).get("error")}); continue
        cont = PROMPT[u] + list(r.get("output_ids", [])) + STOP_IDS
        warm = generate(cont, 64, SAMPLING, SEED + 11)
        cold_ids = cont[:1] + [(cont[1] + 1) % 240000] + cont[2:]
        cold = generate(cold_ids, 64, SAMPLING, SEED + 11)
        mw, mc = warm.get("meta_info", {}), cold.get("meta_info", {})
        rows.append({"unit": u, "prompt_tokens": mw.get("prompt_tokens"), "cached_warm": mw.get("cached_tokens"),
                     "cached_cold": mc.get("cached_tokens"), "wall_warm_s": round(warm["_wall"], 2), "wall_cold_s": round(cold["_wall"], 2)})
    ok = [r for r in rows if r.get("prompt_tokens")]
    ratio = [r["cached_warm"] / r["prompt_tokens"] for r in ok if r.get("cached_warm") is not None]
    sp = round(sum(r["wall_cold_s"] for r in ok) / max(1e-9, sum(r["wall_warm_s"] for r in ok)), 2) if ok else None
    # rev 3b (GPT audits): a cache hit must also make the continuation faster, and the phase-1 outputs must be non-empty
    nonempty = all((r1i or {}).get("output_ids") for r1i in r1)
    return {"rows": rows, "cache_ratio_min": round(min(ratio), 3) if ratio else None, "speedup_warm_vs_cold": sp,
            "phase1_nonempty": nonempty, "pass": bool(ratio) and min(ratio) >= 0.9 and (sp or 0) >= 1.5 and nonempty}


def _long_prompt(k):
    ids, j = [], k * 5
    while len(ids) < LONGCTX_TOKENS - 12000:
        ids += PROMPT[SWEEP_UNITS[j % len(SWEEP_UNITS)]]; j += 1
    tail = PROMPT[SWEEP_UNITS[(j + 1) % len(SWEEP_UNITS)]]         # ends with the assistant/think start
    return (ids + tail)[-(LONGCTX_TOKENS):] if len(ids + tail) > LONGCTX_TOKENS else ids + tail


def x_longctx(srv, name):
    info = http("/get_server_info", timeout=120) or {}
    kv = info.get("max_total_num_tokens") if isinstance(info, dict) else None
    n = max(1, min(12, int(kv // (LONGCTX_TOKENS + LONGCTX_DECODE + 2000)) if kv else 4))
    prompts = [_long_prompt(k) for k in range(n)]
    off = srv.log_size()
    res, wall = run_batch(prompts, LONGCTX_DECODE, T10, [SEED + 500 + i for i in range(n)])
    st = batch_stats(res, wall)
    st["decode_log"] = summarize_decode(decode_rows(srv.log_text(off)))
    st.update(requests=n, context_tokens_mean=round(sum(map(len, prompts)) / n), server_kv_tokens=kv)
    return st


def x_contam(srv, name):
    solo = {u: list(generate(PROMPT[u], GREEDY_TOKENS, {"temperature": 0.0}).get("output_ids", [])) for u in GREEDY_UNITS}
    others = [u for u in SWEEP_UNITS if u not in GREEDY_UNITS][:12]
    ids = [PROMPT[u] for u in GREEDY_UNITS] + [PROMPT[u] for u in others]
    sps = [{"temperature": 0.0}] * len(GREEDY_UNITS) + [T10] * len(others)
    out = [None] * len(ids)
    with cf.ThreadPoolExecutor(max_workers=len(ids)) as ex:
        futs = {ex.submit(generate, x, GREEDY_TOKENS, sps[i], SEED + 900 + i): i for i, x in enumerate(ids)}
        for f in cf.as_completed(futs):
            try:
                out[futs[f]] = f.result()
            except Exception as e:
                out[futs[f]] = {"error": repr(e)[:200]}
    batched = {u: list((out[i] or {}).get("output_ids", [])) for i, u in enumerate(GREEDY_UNITS)}
    pm = prefix_match(solo, batched)
    pm["note"] = "identical or long common prefix = isolated; an early divergence into another task's content = contamination"
    pm["batched_text_head"] = {u: str((out[i] or {}).get("text", ""))[:160] for i, u in enumerate(GREEDY_UNITS)}
    return pm


def x_temp(srv, name):
    out = {}
    units = SWEEP_UNITS[:12]
    # rev 3b (GPT audits): warm the prefix cache for both arms first, so the second arm gets no prefill advantage
    run_batch([PROMPT[u] for u in units], 1, T10, [SEED + 299 + i for i in range(len(units))])
    for tag, sp in (("T1.0", T10), ("T0.6", SAMPLING)):
        off = srv.log_size()
        res, wall = run_batch([PROMPT[u] for u in units], SWEEP_TOKENS, sp, [SEED + 300 + i for i in range(len(units))])
        st = batch_stats(res, wall)
        st["decode_log"] = summarize_decode(decode_rows(srv.log_text(off)))
        out[tag] = st
    a, b = out["T1.0"].get("agg_tok_s"), out["T0.6"].get("agg_tok_s")
    out["speedup_T06_vs_T10"] = round(b / a, 3) if a and b else None
    return out


EXTRA_STAGES.update(contcache=x_contcache, longctx=x_longctx, contam=x_contam, temp=x_temp)
_xcfg = RESULTS.get("best_spec") or ("A_int4" if (RESULTS["configs"].get("A_int4") or {}).get("ok") else None)
if PREFLIGHT_OK and _xcfg:
    _x = run_config(f"X_{_xcfg}_extra", SPEC[_xcfg], stages=("contcache", "longctx", "contam", "temp"))
    RESULTS["extra"] = {k: _x.get(k) for k in ("contcache", "longctx", "contam", "temp")}
    RESULTS["extra"]["config"] = _xcfg
    save_results()
else:
    stamp("extra tests skipped (no working config)")
'''

C6 = r'''
# ==================================================================== CELL 6 — D: the best of B / C with the SSM state in float16
BEST = RESULTS["best_spec"]
if PREFLIGHT_OK and BEST:
    _d = run_config(f"D_{BEST}_ssm_fp16", SPEC[BEST], ssm_dtype="float16", stages=("greedy", "sweep"))
    _ref = RESULTS["configs"][BEST].get("greedy_ids")
    if _ref and _d.get("greedy_ids"):
        _d["greedy_vs_fp32"] = prefix_match(_ref, _d["greedy_ids"])
    save_results()
else:
    stamp("D skipped (no working speculative config)")
'''

C7 = r'''
# ==================================================================== CELL 7 — sanity solve: 8 cheap puzzles, E10 DIRECT protocol
_san = BEST or ("A_int4" if (RESULTS["configs"].get("A_int4") or {}).get("ok") else None)
if PREFLIGHT_OK and _san:
    _s = run_config(f"S_{_san}_sanity", SPEC[_san], stages=("sanity",))
    RESULTS["sanity"] = _s.get("sanity")
    save_results()
else:
    stamp("sanity skipped (no working config)")
'''

C8 = r'''
# ==================================================================== CELL 8 — report
def _f(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


L = [f"# E11-03 GPU validation · run {RUN_ID}", ""]
_pf = RESULTS.get("preflight") or {}
L += [f"Driver {_pf.get('driver')} (CUDA {_pf.get('driver_cuda')}) · shims {json.dumps(RESULTS['env'].get('shims'))}", "",
      "## Preflight", "", "| step | ok | detail |", "|---|---|---|"]
for k, s in (_pf.get("steps") or {}).items():
    L.append(f"| {k}{' (fatal)' if s.get('fatal') else ''} | {'OK' if s.get('ok') else '**FAIL**'} | "
             f"{json.dumps(s.get('out', s.get('err')), default=str)[:220].replace('|', '/')} |")
L += ["", "## Engines", ""]
for name, rec in RESULTS["configs"].items():
    srv = rec.get("server") or {}
    att = rec.get("attempts") or []
    L += [f"### {name} · {'OK' if rec.get('ok') else '**FAILED**'}",
          f"- launch: {[(a['try'], a['extra_flags'], a['why']) for a in att]}",
          f"- KV tokens {srv.get('max_total_num_tokens')} · max running {srv.get('max_running_requests')} · attention "
          f"{srv.get('attention_backend')} · mamba {srv.get('mamba_radix_cache_strategy')} · SSM {srv.get('mamba_ssm_dtype')}"
          f" · draft attention {srv.get('speculative_draft_attention_backend')}",
          f"- smoke: {rec.get('smoke')}"]
    for key in ("greedy_vs_A", "greedy_vs_fp32"):
        if rec.get(key):
            L.append(f"- {key}: {rec[key]['identical']}/{len(GREEDY_UNITS)} identical, mean common prefix "
                     f"{rec[key]['mean_prefix']} of {GREEDY_TOKENS} tokens")
    if rec.get("error"):
        L.append(f"- error: {rec['error']}")
    if rec.get("sweep"):
        L += ["", "| requests | total tok/s | log gen tok/s | per-request tok/s | accept len | wall s | ok |", "|---|---|---|---|---|---|---|"]
        for c, st in rec["sweep"].items():
            if "skipped" in st:
                L.append(f"| {c} | skipped ({st['skipped']}) | | | | | |")
                continue
            L.append(f"| {c} | {_f(st.get('agg_tok_s'))} | {_f(st['decode_log'].get('gen_tok_s_median'))} | "
                     f"{_f(st.get('per_req_tok_s_e2e_median'))} | {_f(st.get('accept_len_mean'), 2)} | {_f(st.get('wall_s'))} | "
                     f"{st.get('ok')}/{st.get('n')} |")
    if rec.get("long"):
        st = rec["long"]
        L.append(f"- long {LONG_CONCURRENCY} x {LONG_TOKENS}: {_f(st.get('agg_tok_s'))} tok/s total, accept "
                 f"{_f(st.get('accept_len_mean'), 2)}, wall {_f(st.get('wall_s'))} s, usage max {st['decode_log'].get('usage_max')}")
    L.append("")
_a = RESULTS["configs"].get("A_int4") or {}
if _a.get("sweep"):
    L += ["## Speed-up vs A (total tok/s)", "", "| requests | " + " | ".join(n for n in RESULTS["configs"] if n != "A_int4") + " |",
          "|---|" + "---|" * (len(RESULTS["configs"]) - 1)]
    for c in map(str, SWEEP_LEVELS):
        base = (_a["sweep"].get(c) or {}).get("agg_tok_s")
        cells = []
        for n, rec in RESULTS["configs"].items():
            if n == "A_int4":
                continue
            v = ((rec.get("sweep") or {}).get(c) or {}).get("agg_tok_s")
            cells.append(f"{v / base:.2f}x" if v and base else "-")
        L.append(f"| {c} | " + " | ".join(cells) + " |")
    L.append("")
_s = RESULTS.get("sanity")
if _s:
    L += [f"## Sanity solve ({_s['config']}): {_s['hits']}/{_s['n']} solved · FP8 expectation {_s['fp8_expected_hits']}", "",
          "| puzzle | hit | mode | think tokens | FP8 think | FP8 p | accept | sec |", "|---|---|---|---|---|---|---|---|"]
    for r in _s["rows"]:
        L.append(f"| {r['uid']} | {r.get('hit')} | {r.get('mode', r.get('error'))} | {r.get('think_tokens')} | "
                 f"{r.get('fp8_think')} | {r.get('fp8_p_hat')} | {_f(r.get('accept_len'), 2)} | {r.get('sec')} |")
_x = RESULTS.get("extra")
if _x:
    _cc, _lc, _ct, _tp = (_x.get(k) or {} for k in ("contcache", "longctx", "contam", "temp"))
    L += [f"## Extra tests (rev 3) on {_x.get('config')}", "",
          f"- **contcache** (forced-close continuation hits the prefix cache): pass {_cc.get('pass')} · min cached ratio "
          f"{_cc.get('cache_ratio_min')} · warm vs cold speed-up {_cc.get('speedup_warm_vs_cold')}x · {_cc.get('error') or ''}",
          f"- **longctx** ({_lc.get('requests')} x ~{_lc.get('context_tokens_mean')} ctx, {LONGCTX_DECODE} new): {_f(_lc.get('agg_tok_s'))} tok/s · accept "
          f"{_f(_lc.get('accept_len_mean'), 2)} · usage max {(_lc.get('decode_log') or {}).get('usage_max')} · ok {_lc.get('ok')}/{_lc.get('n')} · {_lc.get('error') or ''}",
          f"- **contam** (greedy alone vs in a 16-batch): {_ct.get('identical')}/{len(GREEDY_UNITS)} identical · mean common prefix "
          f"{_ct.get('mean_prefix')} of {GREEDY_TOKENS} · {_ct.get('error') or ''}",
          f"- **temp** (12 x {SWEEP_TOKENS}): T1.0 {_f((_tp.get('T1.0') or {}).get('agg_tok_s'))} tok/s accept "
          f"{_f((_tp.get('T1.0') or {}).get('accept_len_mean'), 2)} · T0.6 {_f((_tp.get('T0.6') or {}).get('agg_tok_s'))} tok/s accept "
          f"{_f((_tp.get('T0.6') or {}).get('accept_len_mean'), 2)} · T0.6/T1.0 {_tp.get('speedup_T06_vs_T10')}", ""]
L += ["", f"Total time {(time.time() - T0) / 60:.0f} min · best config (12 requests, complete sweeps): {RESULTS.get('best_spec')}"]
REPORT = "\n".join(L)
(OUT / "report.md").write_text(REPORT, encoding="utf-8")
RESULTS["total_min"] = round((time.time() - T0) / 60, 1)
save_results()
print(REPORT)
'''

cells = [md(M0), code(C0), code(C1), code(C2), code(C3), code(C4), code(C5), code(C5B), code(C6), code(C7), code(C8)]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
(OUTDIR / NAME).write_bytes(data)
print("BUILT", NAME, len(data), "bytes sha256", hashlib.sha256(data).hexdigest())
