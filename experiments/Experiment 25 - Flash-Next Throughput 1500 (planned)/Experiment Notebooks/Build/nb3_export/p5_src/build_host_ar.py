"""Build the P5 host-staged all-reduce torch extension (p5_host_ar.so) with torch.utils.cpp_extension.

Run with the Python of the venv whose torch will load the library (vLLM 0.32 venv: torch 2.13.0+cu130):

    /tmp/vllm-venv/bin/python build_host_ar.py --out /kaggle/working [--build-dir /tmp/p5_build]

CUDA toolchain: the pip CUDA 13.0 toolkit inside that venv (site-packages/nvidia/cu13: nvcc 13.0.88, crt, nvvm,
cudart headers) unless CUDA_HOME is already set. The CUDA code is compiled for sm_89 SASS only
(-gencode arch=compute_89,code=sm_89; torch's _get_cuda_arch_flags adds nothing when a user flag contains 'arch'). No GPU is needed.
Prints one line `BUILD_JSON {...}` with the paths, flags, ptxas resource usage and the .so sha256.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCES = ["host_ar_binding.cpp", "host_ar.cu"]
HEADERS = ["host_ar_layout.h", "host_ar_check.h"]
CUDA_FLAGS = ["-O3", "-gencode", "arch=compute_89,code=sm_89", "-lineinfo", "-Xptxas", "-v",
              "--expt-relaxed-constexpr"]
CXX_FLAGS = ["-O3"]  # torch adds its own -std=c++20 (needs >= c++17: if constexpr)


def find_cuda_home():
    if os.environ.get("CUDA_HOME") and (Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc").exists():
        return Path(os.environ["CUDA_HOME"])
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    for base in (spec.submodule_search_locations or []) if spec else []:
        cand = Path(base) / "cu13"
        if (cand / "bin" / "nvcc").exists():
            return cand
    raise SystemExit("no nvcc: set CUDA_HOME or install nvidia-cuda-nvcc into this venv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory that receives p5_host_ar.so")
    ap.add_argument("--build-dir", default="/tmp/p5_build")
    ap.add_argument("--src", default=str(HERE), help="directory with the .cpp/.cu/.h sources")
    a = ap.parse_args()
    t0 = time.time()
    src = Path(a.src)
    cuda_home = find_cuda_home()
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    os.environ["PATH"] = f"{cuda_home / 'bin'}:{Path(sys.executable).parent}:{os.environ.get('PATH', '')}"
    lib = cuda_home / "lib"
    if lib.exists() and not (lib / "libcudart.so").exists() and (lib / "libcudart.so.13").exists():
        (lib / "libcudart.so").symlink_to("libcudart.so.13")
    if lib.exists() and not (cuda_home / "lib64").exists():
        (cuda_home / "lib64").symlink_to("lib")
    import torch
    from torch.utils import cpp_extension
    bd = Path(a.build_dir)
    if bd.exists():
        shutil.rmtree(bd)
    bd.mkdir(parents=True)
    for f in SOURCES + HEADERS:
        shutil.copy2(src / f, bd / f)
    # ninja/nvcc output (incl. ptxas -v) is written straight to fd 1 by cpp_extension: capture it at the fd level
    cap = bd / "build_output.log"
    err = None
    sys.stdout.flush()
    saved = os.dup(1)
    with open(cap, "wb") as fcap:
        os.dup2(fcap.fileno(), 1)
        try:
            cpp_extension.load(name="p5_host_ar", sources=[str(bd / f) for f in SOURCES],
                               extra_cflags=CXX_FLAGS, extra_cuda_cflags=CUDA_FLAGS, extra_ldflags=["-lrt"],
                               extra_include_paths=[str(bd)], build_directory=str(bd), verbose=True,
                               with_cuda=True, is_python_module=False)
        except Exception as e:  # noqa: BLE001 - reported in BUILD_JSON
            err = f"{type(e).__name__}: {e}"
        finally:
            sys.stdout.flush()
            os.dup2(saved, 1)
            os.close(saved)
    text = cap.read_text(encoding="utf-8", errors="replace")
    so = bd / "p5_host_ar.so"
    out = {"ok": err is None and so.exists(), "error": (err or "")[-6000:], "sec": round(time.time() - t0, 1),
           "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_home": str(cuda_home),
           "cuda_flags": CUDA_FLAGS, "cxx_flags": CXX_FLAGS, "python": sys.version.split()[0]}
    # ptxas -v lines: registers / spills / shared memory per kernel instantiation
    out["ptxas"] = sorted(set(l.strip() for l in text.splitlines() if "ptxas info" in l and
                              ("Used" in l or "spill" in l or "Compiling entry" in l)))[:200]
    out["spill_lines"] = [l for l in out["ptxas"] if re.search(r"[1-9]\d* bytes spill", l)]
    out["log_tail"] = text[-4000:]
    if out["ok"]:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        dst = Path(a.out) / "p5_host_ar.so"
        shutil.copy2(so, dst)
        out["so"] = str(dst)
        out["so_bytes"] = dst.stat().st_size
        out["so_sha256"] = hashlib.sha256(dst.read_bytes()).hexdigest()
    print("BUILD_JSON " + json.dumps(out), flush=True)
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
