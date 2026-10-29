#!/usr/bin/env python3
"""E23 P4 build: compile the vLLM PR #61224 SM89 block-FP8 CUTLASS GEMM (unchanged PR sources) + p4_binding.cpp into a
standalone torch extension p4_sm89_fp8.so for sm_89, with the vLLM 0.32 venv's torch (2.13.0 cu130) and the venv's pip
nvcc 13.0.88. No GPU needed. Then: SASS check (cuobjdump + an independent fatbin header parse), dynamic-symbol check,
ptxas register/spill summary, and a CPU load test in a fresh interpreter (torch.ops.load_library + op schema + CPU
dispatch negative test). Writes OUT/p4_sm89_fp8.so, OUT/p4_build_report.json, OUT/p4_build.log.

usage (run with the venv python, CUDA_HOME = <venv>/lib/python3.12/site-packages/nvidia/cu13):
  python p4_build.py --src SRC_DIR --cutlass CUTLASS_DIR --out OUT_DIR [--cuobjdump PATH] [--jobs 4]
Flags mirror vLLM 0.32's _C_stable_libtorch target (CMakeLists.txt): C++20, -DUSE_CUDA,
TORCH_TARGET_VERSION=0x020B000000000000ULL, CUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1, -DENABLE_FP8, torch's
COMMON_NVCC_FLAGS with the four __CUDA_NO_HALF*/BFLOAT16 defines removed (CUDA >= 12), -gencode compute_89/sm_89 only.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

NAME = "p4_sm89_fp8"
CU = "csrc/quantization/w8a8/cutlass/scaled_mm_blockwise_sm89_fp8.cu"
BIND = "p4_binding.cpp"
DEFS = ["-DUSE_CUDA", "-DTORCH_TARGET_VERSION=0x020B000000000000ULL", "-DENABLE_FP8", "-DNDEBUG"]
NVCC_EXTRA = ["-O3", "-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1", "-DENABLE_SCALED_MM_SM89_BLOCKWISE=1",
              "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
              "-U__CUDA_NO_HALF2_OPERATORS__", "--expt-relaxed-constexpr", "-gencode=arch=compute_89,code=sm_89",
              "--ptxas-options=-v"]
CXX_EXTRA = ["-O3"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def elf_sections(data):
    assert data[:4] == b"\x7fELF" and data[4] == 2, "not an ELF64 file"
    e_shoff = struct.unpack_from("<Q", data, 0x28)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    secs = []
    for i in range(e_shnum):
        name, typ, _flags, _addr, off, size = struct.unpack_from("<IIQQQQ", data, e_shoff + i * e_shentsize)
        secs.append((name, typ, off, size))
    stroff = secs[e_shstrndx][2]
    out = {}
    for name, typ, off, size in secs:
        end = data.index(b"\0", stroff + name)
        out[data[stroff + name:end].decode(errors="replace")] = (off, size, typ)
    return out


def fatbin_entries(so_path):
    """Independent of cuobjdump: walk the .nv_fatbin section (fatbin magic 0xBA55ED50; entry header kind 1 = PTX,
    2 = ELF/SASS, arch at +28, flags at +40). Returns the entry list."""
    data = Path(so_path).read_bytes()
    secs = elf_sections(data)
    if ".nv_fatbin" not in secs:
        return {"error": "no .nv_fatbin section", "sections": sorted(secs)[:80]}
    off, size, _ = secs[".nv_fatbin"]
    sec = data[off:off + size]
    pos, entries = 0, []
    while pos + 16 <= len(sec):
        magic, _ver, hsize, fsize = struct.unpack_from("<IHHQ", sec, pos)
        if magic != 0xBA55ED50:
            pos += 8
            continue
        p, end = pos + hsize, pos + hsize + fsize
        while p + 64 <= end:
            kind, _u, ehsize, esize = struct.unpack_from("<HHIQ", sec, p)
            arch = struct.unpack_from("<I", sec, p + 28)[0]
            flags = struct.unpack_from("<Q", sec, p + 40)[0]
            entries.append({"kind": {1: "ptx", 2: "elf"}.get(kind, kind), "arch": arch, "size": esize,
                            "flag_0x2000": bool(flags & 0x2000)})
            if ehsize <= 0:
                break
            p += ehsize + esize
        pos = max(end, pos + 8)
    return {"section_bytes": size, "entries": entries}


def find_cuobjdump(arg):
    cands = [arg, shutil.which("cuobjdump"), os.path.join(os.environ.get("CUDA_HOME", ""), "bin", "cuobjdump"),
             "/usr/local/cuda/bin/cuobjdump"] + sorted(str(p) for p in Path("/usr/local").glob("cuda-*/bin/cuobjdump"))
    for c in cands:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--cutlass", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-dir", default="/tmp/p4_build")
    ap.add_argument("--cuobjdump", default=None)
    ap.add_argument("--jobs", default="4")
    ap.add_argument("--cccl-include", default=None,
                    help="use this CCCL include dir (searched first) instead of the venv's nvidia/cu13/include/cccl")
    a = ap.parse_args()
    src, cutlass, out = Path(a.src).resolve(), Path(a.cutlass).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rep = {"name": NAME, "steps": {}, "python": sys.version.split()[0]}
    t0 = time.time()

    def save():
        (out / "p4_build_report.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")

    # ---- 0. toolchain
    import torch
    cuda_home = os.environ.get("CUDA_HOME", "")
    rep["torch"], rep["torch_cuda"], rep["CUDA_HOME"] = torch.__version__, torch.version.cuda, cuda_home
    try:
        r = subprocess.run([os.path.join(cuda_home, "bin", "nvcc"), "--version"], capture_output=True, text=True)
        rep["nvcc_version"] = (r.stdout + r.stderr).strip().splitlines()[-2:] if r.returncode == 0 else f"rc={r.returncode}"
    except OSError as e:
        rep["nvcc_version"] = f"{type(e).__name__}: {e}"
    try:
        r = subprocess.run(["c++", "--version"], capture_output=True, text=True)
        rep["host_cxx"] = (r.stdout or r.stderr).splitlines()[0] if (r.stdout or r.stderr) else "?"
    except OSError as e:
        rep["host_cxx"] = f"{type(e).__name__}: {e}"
    rep["sources"] = {p: sha256(src / p) for p in (CU, BIND, "csrc/torch_utils.h",
                                                   "csrc/cutlass_extensions/gemm/kernel/gemm_blockwise_pertoken.h",
                                                   "csrc/cutlass_extensions/gemm/threadblock/"
                                                   "mma_multistage_blockwise_pertoken.h")}
    save()

    # ---- 1. build (torch.utils.cpp_extension.load, ninja, build_directory)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
    os.environ.setdefault("MAX_JOBS", a.jobs)
    import torch.utils.cpp_extension as ce
    bdir = Path(a.build_dir)
    shutil.rmtree(bdir, ignore_errors=True)
    bdir.mkdir(parents=True)
    cccl = a.cccl_include or os.path.join(cuda_home, "include", "cccl")
    incs = [cccl, str(src / "csrc"), str(cutlass / "include"), str(cutlass / "tools/util/include"),
            os.path.join(cuda_home, "include")]
    ver_h = Path(cccl, "cuda/std/__cccl/version.h")
    rep["cccl"] = {"include": cccl, "from": "--cccl-include" if a.cccl_include else "venv nvidia-cuda-cccl",
                   "CCCL_VERSION": next((l.split()[-1] for l in ver_h.read_text().splitlines()
                                         if l.startswith("#define CCCL_VERSION ")), None) if ver_h.exists() else None}
    rep["flags"] = {"cflags": CXX_EXTRA + DEFS, "cuda_cflags": NVCC_EXTRA + DEFS, "include_paths": incs,
                    "torch_COMMON_NVCC_FLAGS": list(ce.COMMON_NVCC_FLAGS), "TORCH_CUDA_ARCH_LIST": "8.9",
                    "CUDA_HOME_seen_by_cpp_extension": str(ce.CUDA_HOME)}
    save()
    tb = time.time()
    build_exc = None
    # ninja (verbose=True) writes the compiler output, incl. ptxas -v, to fd 1: send fd 1/2 to p4_build.log
    log = out / "p4_build.log"
    sys.stdout.flush(), sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    with open(log, "wb") as lf:
        os.dup2(lf.fileno(), 1), os.dup2(lf.fileno(), 2)
        try:
            ce.load(name=NAME, sources=[str(src / BIND), str(src / CU)], extra_cflags=CXX_EXTRA + DEFS,
                    extra_cuda_cflags=NVCC_EXTRA + DEFS, extra_include_paths=incs, build_directory=str(bdir),
                    verbose=True, with_cuda=True, is_python_module=False)
        except Exception as e:
            build_exc = f"{type(e).__name__}: {str(e)[-3000:]}"
        finally:
            sys.stdout.flush(), sys.stderr.flush()
            os.dup2(saved[0], 1), os.dup2(saved[1], 2)
            os.close(saved[0]), os.close(saved[1])
    ltxt = log.read_text(errors="replace")
    print("---- p4_build.log tail ----\n" + ltxt[-6000:], flush=True)
    so = bdir / f"{NAME}.so"
    rep["steps"]["build"] = {"ok": so.exists(), "sec": round(time.time() - tb), "error": build_exc,
                             "loaded_in_build_process": build_exc is None}
    nj = bdir / "build.ninja"
    if nj.exists():
        shutil.copy(nj, out / "p4_build.ninja")
    save()
    if not so.exists():
        rep["all_ok"] = False
        save()
        print("P4_BUILD " + json.dumps({"all_ok": False, "failed": "build"}))
        return 1
    shutil.copy(so, out / f"{NAME}.so")
    so = out / f"{NAME}.so"
    rep["so"] = {"path": str(so), "bytes": so.stat().st_size, "sha256": sha256(so)}
    save()

    # ---- 2. SASS: cuobjdump (if present) + fatbin header parse
    sass = {}
    cuo = find_cuobjdump(a.cuobjdump)
    sass["cuobjdump"] = cuo
    if cuo:
        r1 = subprocess.run([cuo, "--list-elf", str(so)], capture_output=True, text=True)
        r2 = subprocess.run([cuo, "--list-ptx", str(so)], capture_output=True, text=True)
        sass["list_elf"] = (r1.stdout + r1.stderr).strip().splitlines()[:40]
        sass["list_ptx"] = (r2.stdout + r2.stderr).strip().splitlines()[:40]
        sass["elf_archs"] = sorted(set(re.findall(r"sm_(\d+)", r1.stdout)))
        sass["ptx_archs"] = sorted(set(re.findall(r"(?:sm|compute)_(\d+)", r2.stdout)))
        r3 = subprocess.run([cuo, "-sass", str(so)], capture_output=True, text=True)
        txt = r3.stdout
        sass["sass_rc"] = r3.returncode
        sass["sass_counts"] = {m: len(re.findall(r"\b" + m + r"\b", txt)) for m in ("QMMA", "HMMA", "IMMA", "LDGSTS",
                                                                                     "LDSM", "F2FP")}
        sass["sass_qmma_variants"] = sorted(set(re.findall(r"QMMA\.[A-Z0-9.]+", txt)))[:10]
        sass["sass_functions"] = len(re.findall(r"Function : ", txt))
        del txt
        sass["cuobjdump_ok"] = "89" in sass["elf_archs"]
    try:
        fb = fatbin_entries(so)
        sass["fatbin_parse"] = fb
        ents = fb.get("entries", [])
        sass["fatbin_elf_archs"] = sorted({e["arch"] for e in ents if e["kind"] == "elf"})
        sass["fatbin_ptx_archs"] = sorted({e["arch"] for e in ents if e["kind"] == "ptx"})
        sass["fatbin_ok"] = sass["fatbin_elf_archs"] == [89]
    except Exception as e:
        sass["fatbin_parse_error"] = f"{type(e).__name__}: {e}"
        sass["fatbin_ok"] = False
    sass["ok"] = bool(sass.get("cuobjdump_ok")) if cuo else bool(sass.get("fatbin_ok"))
    if cuo and sass.get("fatbin_ok") is not None and sass.get("cuobjdump_ok") != sass.get("fatbin_ok"):
        sass["note"] = "cuobjdump and the fatbin parse disagree; cuobjdump decides"
    rep["steps"]["sass_sm89"] = sass
    save()

    # ---- 3. dynamic symbols: NEEDED libs, undefined CUDA driver symbols (we do not link -lcuda)
    syms = {}
    try:
        r = subprocess.run(["readelf", "-d", str(so)], capture_output=True, text=True)
        syms["needed"] = re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", r.stdout)
        r = subprocess.run(["nm", "-D", "--undefined-only", str(so)], capture_output=True, text=True)
        und = [l.split()[-1] for l in r.stdout.splitlines() if l.strip()]
        syms["undefined_count"] = len(und)
        syms["undefined_cuda_driver"] = sorted(s for s in und if re.match(r"^cu[A-Z]", s))
        syms["undefined_cudart"] = sorted(s for s in und if s.startswith("cuda"))[:40]
        syms["undefined_torch_shim"] = sorted(s for s in und if s.startswith(("aoti_torch", "torch_")))[:60]
        syms["ok"] = r.returncode == 0 and not syms["undefined_cuda_driver"]
    except Exception as e:
        syms.update({"ok": False, "error": f"{type(e).__name__}: {e}"})
    rep["steps"]["symbols"] = syms
    save()

    # ---- 4. ptxas -v summary (info, not a gate: the PR author reports the mainloop spill-free)
    regs = [int(x) for x in re.findall(r"Used (\d+) registers", ltxt)]
    spills = re.findall(r"(\d+) bytes spill stores, (\d+) bytes spill loads", ltxt)
    rep["steps"]["ptxas"] = {"kernels_reported": len(regs), "max_registers": max(regs) if regs else None,
                             "registers": sorted(set(regs)), "any_spill": any(int(s) or int(l) for s, l in spills),
                             "spills_store_load": sorted(set(f"{s}/{l}" for s, l in spills)),
                             "warnings": len(re.findall(r"warning", ltxt))}
    save()

    # ---- 5. CPU load test in a fresh interpreter
    probe = r'''
import json, sys, torch
torch.ops.load_library(sys.argv[1])
op = torch.ops.p4_sm89.cutlass_scaled_mm_blockwise_sm89_fp8
info = {"schema": str(op.default._schema), "cuda_available": torch.cuda.is_available(), "torch": torch.__version__}
try:   # CPU tensors must dispatch to "no CPU kernel" (the op is registered for CUDA only)
    a = torch.zeros(16, 128, dtype=torch.float8_e4m3fn); b = torch.zeros(128, 16, dtype=torch.float8_e4m3fn).t().contiguous().t()
    op(torch.empty(16, 16, dtype=torch.bfloat16), a, b, torch.ones(16, 1), torch.ones(1, 1))
    info["cpu_call"] = "UNEXPECTED: returned"
except Exception as e:
    info["cpu_call"] = type(e).__name__ + ": " + str(e).splitlines()[0][:300]
print("LOAD_JSON " + json.dumps(info))
'''
    r = subprocess.run([sys.executable, "-c", probe, str(so)], capture_output=True, text=True, timeout=600)
    line = next((l for l in r.stdout.splitlines() if l.startswith("LOAD_JSON ")), None)
    ld = json.loads(line[10:]) if line else {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-2500:]}
    # torch 2.13 prints the mutable alias as "Tensor($0! -> ) out" (kit v1); older ones as "Tensor! out" / "Tensor(a!) out"
    ld["ok"] = bool(line) and bool(re.search(r"Tensor(?:!|\([^)]*!\s*(?:->\s*)?\))\s+out\b", ld.get("schema", "")))         and "NotImplementedError" in ld.get("cpu_call", "") and "'CPU' backend" in ld.get("cpu_call", "")
    rep["steps"]["load_cpu"] = ld
    rep["all_ok"] = all(rep["steps"][k].get("ok") for k in ("build", "sass_sm89", "symbols", "load_cpu"))
    rep["sec"] = round(time.time() - t0)
    save()
    print("P4_BUILD " + json.dumps({"all_ok": rep["all_ok"], "failed": [k for k, v in rep["steps"].items()
                                                                        if isinstance(v, dict) and v.get("ok") is False]}))
    return 0 if rep["all_ok"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        print("P4_BUILD " + json.dumps({"all_ok": False, "fatal": f"{type(e).__name__}: {e}",
                                        "tb": traceback.format_exc()[-3000:]}))
        sys.exit(2)
