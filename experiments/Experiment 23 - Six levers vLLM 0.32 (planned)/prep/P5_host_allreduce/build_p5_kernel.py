"""Build the P5 CPU kernel (kernel/): Kaggle CPU notebook that builds the host-staged all-reduce extension.

Kernel foysalemonshanto/q27-e23-p5-host-ar (CPU, internet on, pinned image, private). The P5 sources (src/, tests/,
bench_host_ar.py) are embedded in the notebook as a sha256-checked zip (about 40 kB, notebook stays far below 400 kB),
so the only dataset is the vLLM 0.32 wheelhouse. Notebook steps:
  1. unpack the sources (sha256 check);
  2. install the vLLM 0.32.0 venv from foysalemonshanto/vllm-0-32-0-cu13-wheelhouse with its install_wheelhouse.py
     (same as Experiment 22 build_cpu_kit_e22_flags.py): torch 2.13.0+cu130, CUDA 13.0.88 pip toolkit (nvcc, crt,
     nvvm, cudart headers) under site-packages/nvidia/cu13, ninja;
  3. toolchain probe (nvcc / g++ / ninja versions, headers, libcudart);
  4. build p5_host_ar.so with src/build_host_ar.py (torch.utils.cpp_extension, -gencode arch=compute_89,code=sm_89);
  5. SASS check with cuobjdump (pip nvidia-cuda-cuobjdump 13.2.51: the nvcc 13.0.88 wheel ships none): sm_89 ELF present, no other
     arch, no PTX, all 18 kernel instantiations, memory-ordering instructions present; ptxas register/spill report;
  6. load the .so on CPU in the venv torch (bench_host_ar.py --cpu-check: ops registered, partition model, init fails
     cleanly without a GPU);
  7. host tests: tests/test_layout.cpp (g++), tests/sim_protocol.py, tests/emulate_numerics.py;
  8. write /kaggle/working/p5_host_ar.so, bench_host_ar.py, p5_src/ (sources), p5_report.json, and print ALL OK / NOT OK.
Run:  python build_p5_kernel.py      (writes kernel/q27-e23-p5-host-ar.ipynb + kernel/kernel-metadata.json)
"""
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_SLUG = "foysalemonshanto/q27-e23-p5-host-ar"
KERNEL_TITLE = "Q27 E23 P5 Host AR"
WH032_SLUG = "foysalemonshanto/vllm-0-32-0-cu13-wheelhouse"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
FILES = ["src/host_ar.cu", "src/host_ar_binding.cpp", "src/host_ar_layout.h", "src/host_ar_check.h",
         "src/build_host_ar.py", "tests/test_layout.cpp", "tests/sim_protocol.py", "tests/emulate_numerics.py",
         "bench_host_ar.py", "README.md"]

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for rel in FILES:
        data = (HERE / rel).read_bytes().replace(b"\r\n", b"\n")  # LF on Linux whatever the Windows checkout did
        zi = zipfile.ZipInfo(rel, date_time=(2026, 10, 29, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, data)
ZIPB = buf.getvalue()
ZIP_SHA = hashlib.sha256(ZIPB).hexdigest()
B64 = base64.b64encode(ZIPB).decode()

CODE = r'''# Q27 E23 P5: build + check the host-staged all-reduce extension (CPU kernel, vLLM 0.32 venv, CUDA 13 pip nvcc)
import base64, glob, hashlib, io, json, os, re, shutil, stat, subprocess, sys, time, zipfile
from pathlib import Path
ZIP_SHA = "''' + ZIP_SHA + r'''"
ZIP_B64 = "''' + B64 + r'''"
WH_SLUG = "''' + WH032_SLUG + r'''"
OUT, W, T0 = Path("/kaggle/working"), Path("/tmp/p5"), time.time()
REPORT = {"kernel": "''' + KERNEL_SLUG + r'''", "src_zip_sha256": ZIP_SHA, "python": sys.version.split()[0], "steps": {}}
def save():
    REPORT["sec"] = round(time.time() - T0)
    (OUT / "p5_report.json").write_text(json.dumps(REPORT, indent=1, default=str), encoding="utf-8")
def run(cmd, timeout=1800, env=None, cwd=None):
    t = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)
        return {"rc": r.returncode, "sec": round(time.time() - t, 1), "out": r.stdout, "err": r.stderr}
    except Exception as e:
        return {"rc": -1, "sec": round(time.time() - t, 1), "out": "", "err": f"{type(e).__name__}: {e}"}
def tail(r, n=3000):
    return ((r.get("out") or "") + (r.get("err") or ""))[-n:]
def jline(r, tag):
    for l in (r.get("out") or "").splitlines():
        if l.startswith(tag + " "):
            return json.loads(l[len(tag) + 1:])
    return None
ok = {}

# 1. sources
b = base64.b64decode(ZIP_B64)
assert hashlib.sha256(b).hexdigest() == ZIP_SHA, "embedded source zip sha256 mismatch"
if W.exists():
    shutil.rmtree(W)
zipfile.ZipFile(io.BytesIO(b)).extractall(W)
REPORT["steps"]["sources"] = sorted(str(p.relative_to(W)) for p in W.rglob("*") if p.is_file())
print("sources:", REPORT["steps"]["sources"], flush=True)

# 2. vLLM 0.32 venv (as Experiment 22 build_cpu_kit_e22_flags.py)
cands = sorted(Path("/kaggle/input").rglob("install_wheelhouse.py"))
inst = next((p for p in cands if "vllm-0-32-0-cu13" in str(p)), cands[0] if cands else None)
assert inst is not None, "attach " + WH_SLUG
r = run([sys.executable, str(inst), "/tmp/vllm-venv"], timeout=2400)
vpy = next((l.split("=", 1)[1] for l in r["out"].splitlines() if l.startswith("VENV_PYTHON=")), "/tmp/vllm-venv/bin/python")
cuda_home = next((l.split("=", 1)[1] for l in r["out"].splitlines() if l.startswith("CUDA_HOME=")), "")
REPORT["steps"]["install"] = {"installer": str(inst), "rc": r["rc"], "sec": r["sec"], "venv_python": vpy,
                              "cuda_home": cuda_home, "tail": tail(r, 1200)}
ok["install"] = r["rc"] == 0 and Path(vpy).exists() and bool(cuda_home)
save()
print("install:", ok["install"], r["sec"], "s", flush=True)
CH = Path(cuda_home)
ENV = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_HOME", "CUDA_PATH")}
ENV.update({"CUDA_HOME": cuda_home, "TORCH_CUDA_ARCH_LIST": "8.9", "MAX_JOBS": "4",
            "PATH": f"{Path(vpy).parent}:{CH / 'bin'}:{os.environ.get('PATH', '')}"})

# 3. toolchain probe
tc = {"nvcc": tail(run([str(CH / "bin" / "nvcc"), "--version"], env=ENV), 400),
      "gxx": tail(run(["g++", "--version"], env=ENV), 300),
      "ninja": tail(run([str(Path(vpy).parent / "ninja"), "--version"], env=ENV), 100),
      "torch": tail(run([vpy, "-c", "import torch;print(torch.__version__, torch.version.cuda, torch._C._GLIBCXX_USE_CXX11_ABI)"], env=ENV), 300),
      "files": {rel: (CH / rel).exists() for rel in ("bin/nvcc", "bin/ptxas", "bin/fatbinary", "nvvm/bin/cicc",
                                                      "include/cuda_runtime.h", "include/cuda_runtime_api.h",
                                                      "include/crt/host_config.h", "lib/libcudart.so", "lib64")}}
REPORT["steps"]["toolchain"] = tc
ok["toolchain"] = "release 13.0" in tc["nvcc"]  # the file map is informational; the build step is the real gate
save()
print("toolchain:", json.dumps(tc)[:1500], flush=True)

# 4. build
r = run([vpy, str(W / "src" / "build_host_ar.py"), "--out", str(OUT), "--build-dir", "/tmp/p5_build",
         "--src", str(W / "src")], timeout=2400, env=ENV)
bj = jline(r, "BUILD_JSON") or {"ok": False, "tail": tail(r, 6000)}
REPORT["steps"]["build"] = {"rc": r["rc"], "sec": r["sec"], **bj}
SO = OUT / "p5_host_ar.so"
ok["build"] = r["rc"] == 0 and bool(bj.get("ok")) and SO.exists()
save()
print("build:", ok["build"], r["sec"], "s |", (bj.get("error") or "")[-2000:], flush=True)
for l in (bj.get("ptxas") or [])[:60]:
    print("   ", l, flush=True)

# 5. SASS check (cuobjdump)
def find_cuobjdump():
    if (CH / "bin" / "cuobjdump").exists():
        return str(CH / "bin" / "cuobjdump"), "toolkit"
    # the nvcc 13.0.88 wheel has no cuobjdump and PyPI starts nvidia-cuda-cuobjdump at 13.2.51 (newer tools read
    # 13.0 cubins); the wheel holds nvidia/cu13/bin/cuobjdump
    d = Path("/tmp/cuobj")
    for spec in ("nvidia-cuda-cuobjdump==13.2.51", "nvidia-cuda-cuobjdump>=13,<14"):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
        rr = run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary", ":all:", "-q", "-d", str(d), spec], timeout=600)
        for whl in d.glob("*.whl"):
            zipfile.ZipFile(whl).extractall(d / "x")
            for c in (d / "x").rglob("cuobjdump"):
                if c.is_file():
                    c.chmod(c.stat().st_mode | stat.S_IEXEC)
                    return str(c), f"pip {whl.name}"
    for c in (Path("/usr/local/cuda/bin/cuobjdump"), Path(shutil.which("cuobjdump") or "/nonexistent")):
        if c.exists():
            return str(c), "system"
    return None, "not found"
sass = {}
if ok["build"]:
    tool, src = find_cuobjdump()
    sass["tool"], sass["tool_source"] = tool, src
    if tool:
        le = run([tool, "--list-elf", str(SO)], env=ENV)
        lp = run([tool, "--list-ptx", str(SO)], env=ENV)
        ds = run([tool, "--dump-sass", str(SO)], env=ENV, timeout=900)
        sass["list_elf"] = tail(le, 3000)
        sass["list_ptx"] = tail(lp, 1000)
        archs = sorted(set(re.findall(r"sm_(\d+)", le["out"])))
        sass["elf_archs"] = archs
        sass["ptx_entries"] = len([l for l in lp["out"].splitlines() if ".ptx" in l])
        txt = ds["out"]
        (OUT / "p5_sass_head.txt").write_text(txt[:400000], encoding="utf-8")
        funcs = re.findall(r"Function : (\S+)", txt)
        sass["kernel_functions"] = len([f for f in funcs if "p5_ar_kernel" in f])
        sass["functions_sample"] = funcs[:6]
        sass["instr_counts"] = {k: len(re.findall(k, txt)) for k in (r"MEMBAR\.SC\.SYS", r"MEMBAR\.ALL\.SYS", r"MEMBAR\S*SYS",
                                                                    r"STRONG\.SYS", r"LDG\.E\S*STRONG\.SYS",
                                                                    r"STG\.E\S*STRONG\.SYS", r"CCTL", r"SHFL\.BFLY",
                                                                    r"F2FP\.BF16", r"F2I")}
        sass["dump_rc"], sass["dump_bytes"] = ds["rc"], len(txt)
        ok["sass"] = (le["rc"] == 0 and archs == ["89"] and sass["ptx_entries"] == 0 and sass["kernel_functions"] >= 18
                      and sass["instr_counts"][r"MEMBAR\S*SYS"] > 0 and sass["instr_counts"][r"STRONG\.SYS"] > 0)
    else:
        ok["sass"] = False
    # independent hint: embedded CUDA ELF headers (EM_CUDA = 190); fatbins may be compressed, so informational only
    data = SO.read_bytes()
    hits = []
    for m in re.finditer(rb"\x7fELF\x02\x01\x01", data):
        o = m.start()
        if o and int.from_bytes(data[o + 18:o + 20], "little") == 190:
            hits.append({"off": o, "e_flags": hex(int.from_bytes(data[o + 48:o + 52], "little"))})
    sass["embedded_cuda_elf_headers"] = hits[:8]
    sass["so_ldd"] = tail(run(["ldd", str(SO)], env=ENV), 2000)
else:
    ok["sass"] = False
REPORT["steps"]["sass"] = sass
REPORT["steps"]["build"]["n_spill_lines"] = len(REPORT["steps"]["build"].get("spill_lines") or [])
save()
print("sass:", ok["sass"], json.dumps({k: v for k, v in sass.items() if k not in ("list_elf", "so_ldd")})[:2500], flush=True)

# 6. CPU load + op registration + partition model (venv torch)
if ok["build"]:
    r = run([vpy, str(W / "bench_host_ar.py"), "--cpu-check", "--so", str(SO), "--out", "/tmp/p5_cpu_check.json"], env=ENV, timeout=900)
    cc = jline(r, "CPU_CHECK") or {"ok": False, "tail": tail(r, 3000)}
    REPORT["steps"]["cpu_load"] = {"rc": r["rc"], **cc}
    ok["cpu_load"] = r["rc"] == 0 and bool(cc.get("ok"))
else:
    ok["cpu_load"] = False
save()
print("cpu_load:", ok["cpu_load"], json.dumps(REPORT["steps"].get("cpu_load", {}))[:1500], flush=True)

# 7. host tests
r = run(["g++", "-O2", "-std=c++17", "-Wall", "-I", str(W / "src"), str(W / "tests" / "test_layout.cpp"), "-o", "/tmp/p5_test_layout"], env=ENV, timeout=600)
lay = {"compile_rc": r["rc"], "compile_tail": tail(r, 1500)}
if r["rc"] == 0:
    r = run(["/tmp/p5_test_layout"], timeout=1200)
    lay.update({"rc": r["rc"], "sec": r["sec"], "result": jline(r, "LAYOUT_JSON")})
REPORT["steps"]["test_layout"] = lay
ok["test_layout"] = lay.get("rc") == 0 and bool((lay.get("result") or {}).get("ok"))
r = run([vpy, str(W / "tests" / "sim_protocol.py")], env=ENV, timeout=900)
REPORT["steps"]["sim_protocol"] = {"rc": r["rc"], "result": jline(r, "SIM_JSON"), "tail": tail(r, 800) if r["rc"] else ""}
ok["sim_protocol"] = r["rc"] == 0
r = run([vpy, str(W / "tests" / "emulate_numerics.py")], env=ENV, timeout=1800)
REPORT["steps"]["emulate_numerics"] = {"rc": r["rc"], "sec": r["sec"], "result": jline(r, "EMU_JSON"), "tail": tail(r, 800) if r["rc"] else ""}
ok["emulate_numerics"] = r["rc"] == 0
r = run([vpy, "-m", "py_compile", str(W / "bench_host_ar.py"), str(W / "src" / "build_host_ar.py")], env=ENV)
ok["py_compile_py312"] = r["rc"] == 0
save()
print("tests:", {k: ok.get(k) for k in ("test_layout", "sim_protocol", "emulate_numerics", "py_compile_py312")}, flush=True)

# 8. outputs
shutil.copy2(W / "bench_host_ar.py", OUT / "bench_host_ar.py")
if (OUT / "p5_src").exists():
    shutil.rmtree(OUT / "p5_src")
shutil.copytree(W, OUT / "p5_src")
if SO.exists():
    REPORT["so"] = {"path": str(SO), "bytes": SO.stat().st_size, "sha256": hashlib.sha256(SO.read_bytes()).hexdigest()}
REPORT["ok"] = ok
REPORT["warnings"] = ([f"ptxas spills: {REPORT['steps']['build'].get('spill_lines')}"]
                      if REPORT["steps"]["build"].get("n_spill_lines") else [])
REPORT["all_ok"] = all(ok.values())
save()
print(json.dumps(ok), flush=True)
print("ALL OK" if REPORT["all_ok"] else "NOT OK", "| total", REPORT["sec"], "s", flush=True)
'''
compile(CODE, "<p5 kernel>", "exec")
nbj = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": CODE.splitlines(True)}],
       "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
KD = HERE / "kernel"
KD.mkdir(exist_ok=True)
NB = KD / "q27-e23-p5-host-ar.ipynb"
NB.write_bytes(json.dumps(nbj, indent=1).encode("utf-8"))
META = {"id": KERNEL_SLUG, "title": KERNEL_TITLE, "code_file": NB.name, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False,
        "enable_internet": True, "keywords": [], "dataset_sources": [WH032_SLUG], "kernel_sources": [],
        "competition_sources": [], "model_sources": [], "docker_image": IMAGE, "machine_shape": "None"}
(KD / "kernel-metadata.json").write_bytes(json.dumps(META, indent=2).encode("utf-8"))
print("notebook", NB, NB.stat().st_size, "bytes | src zip", len(ZIPB), "bytes sha256", ZIP_SHA)
