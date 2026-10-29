"""E23 P4 builder: Kaggle CPU kernel (internet ON, pinned image) that builds the vLLM PR #61224 SM89 block-FP8 CUTLASS
GEMM as a standalone torch extension p4_sm89_fp8.so (sm_89 SASS only) with the vLLM 0.32.0 venv's torch 2.13.0 cu130
and pip nvcc 13.0.88 (wheelhouse dataset foysalemonshanto/vllm-0-32-0-cu13-wheelhouse, installed exactly as the E22
flags kit does), then checks: sm_89 SASS present (cuobjdump 13.2.51 from PyPI + an independent fatbin parse), no CUDA
driver symbols, ptxas registers/spills, CPU load (torch.ops.load_library + schema + CPU-dispatch negative test), and
bench_fp8_gemm.py --check-imports against the real vLLM 0.32 (the Marlin / quant functions the GPU bench calls).

Writes kernel/q27-e23-p4-sm89-fp8.ipynb + kernel/kernel-metadata.json and src/SOURCES.json. Run: python build_p4_kernel.py
The vendored sources (src/) are embedded in the notebook as a sha256-checked zip; CUTLASS v4.7.1 (vLLM 0.32.0's
CUTLASS_REVISION) is fetched by the kernel with git at the pinned tag and its commit is verified.
Kernel output (/kaggle/working): p4_sm89_fp8.so, p4_report.json, p4_build_report.json, p4_build.log, p4_build.ninja,
p4_build_driver.log, p4_bench_check_imports.json, include_graph_kaggle.json, bench_fp8_gemm.py, p4_src.zip.
"""
import base64
import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_SLUG = "foysalemonshanto/q27-e23-p4-sm89-fp8"
WH032_SLUG = "foysalemonshanto/vllm-0-32-0-cu13-wheelhouse"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
VLLM_TAG, VLLM_COMMIT = "v0.32.0", "5df63a7203cdb9328d824a27b682e77c7b9928be"
PR, PR_HEAD = 61224, "dbe22a894e66c11e03f47a70867c93b0c26bb410"
PR_DIFF_SHA256 = "083962b3b0030eee199ed7004c4456636b586356703a4949abe27e642b63da87"   # patches/pr61224.diff = gh pr diff
CUTLASS_TAG, CUTLASS_COMMIT = "v4.7.1", "cb4247394dd82148787aed73e5dc7cef33cbf862"     # vLLM 0.32.0 CUTLASS_REVISION
CUOBJDUMP_PIN = "nvidia-cuda-cuobjdump==13.2.51"   # oldest cu13 cuobjdump on PyPI (13.0/13.1 were never published)
CUOBJDUMP_WHEEL = "nvidia_cuda_cuobjdump-13.2.51-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
CUOBJDUMP_SHA256 = "95c69d3ddf0c2c1f535ec44c3433a66aeee19f588d9d22f73356e20b25f12b50"
# build fallback only: the CCCL that cuda-toolkit 13.0.3 (torch 2.13 cu130's toolkit set) pins next to nvcc 13.0.88
CCCL130_PIN = "nvidia-cuda-cccl==13.0.85"
CCCL130_WHEEL = "nvidia_cuda_cccl-13.0.85-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl"
CCCL130_SHA256 = "e0da7ad981f3a8aff08241b5bfc1af868742a63e2762f53a5171c492ef242649"

# vendored file -> provenance. git blob ids: the PR diff's "index 0000000..<blob>" lines; torch_utils.h = v0.32.0 tree.
VENDORED = {
    "csrc/quantization/w8a8/cutlass/scaled_mm_blockwise_sm89_fp8.cu": f"vLLM PR #{PR} (head {PR_HEAD}), new file, unchanged",
    "csrc/cutlass_extensions/gemm/kernel/gemm_blockwise_pertoken.h": f"vLLM PR #{PR} (head {PR_HEAD}), new file, unchanged",
    "csrc/cutlass_extensions/gemm/threadblock/mma_multistage_blockwise_pertoken.h": f"vLLM PR #{PR} (head {PR_HEAD}), new file, unchanged",
    "csrc/torch_utils.h": f"vLLM {VLLM_TAG} ({VLLM_COMMIT}) csrc/torch_utils.h, unchanged",
    "p4_binding.cpp": "E23 P4 (ours): STABLE_TORCH_LIBRARY binding, one op torch.ops.p4_sm89.cutlass_scaled_mm_blockwise_sm89_fp8",
    "p4_build.py": "E23 P4 (ours): torch.utils.cpp_extension.load build + SASS / symbol / load checks",
    "LICENSE.vllm": f"vLLM {VLLM_TAG} LICENSE (Apache-2.0)",
    "LICENSE.cutlass.txt": f"CUTLASS {CUTLASS_TAG} LICENSE.txt (BSD-3-Clause; the PR headers carry the same NVIDIA BSD-3 notice)",
}
EXPECTED_BLOB = {"csrc/quantization/w8a8/cutlass/scaled_mm_blockwise_sm89_fp8.cu": "f8c2a50b4765",
                 "csrc/cutlass_extensions/gemm/kernel/gemm_blockwise_pertoken.h": "d66c61baa1dc",
                 "csrc/cutlass_extensions/gemm/threadblock/mma_multistage_blockwise_pertoken.h": "f78e5ea453d1",
                 "csrc/torch_utils.h": "f02346739e0ac6bb46e8ec3198e012f8798e1599"}


def blob_id(b):
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


src = HERE / "src"
files, sources_json = {}, {}
for rel, why in VENDORED.items():
    b = (src / rel).read_bytes()
    assert b"\r\n" not in b, f"CRLF in {rel} (Windows edit): restore LF before building"
    if rel in EXPECTED_BLOB:
        assert blob_id(b).startswith(EXPECTED_BLOB[rel]), (rel, blob_id(b), EXPECTED_BLOB[rel])
    files["src/" + rel] = b
    sources_json[rel] = {"provenance": why, "sha256": hashlib.sha256(b).hexdigest(), "git_blob": blob_id(b),
                         "bytes": len(b)}
for extra in ("bench_fp8_gemm.py", "tools/check_includes.py"):
    b = (HERE / extra).read_bytes()
    assert b"\r\n" not in b, f"CRLF in {extra}"
    compile(b.decode(), extra, "exec")
    files[extra] = b
    sources_json[extra] = {"provenance": "E23 P4 (ours)", "sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)}
compile((src / "p4_build.py").read_text(encoding="utf-8"), "p4_build.py", "exec")
meta = {"vllm_tag": VLLM_TAG, "vllm_commit": VLLM_COMMIT, "pr": PR, "pr_head": PR_HEAD, "pr_diff_sha256": PR_DIFF_SHA256,
        "pr_url": f"https://github.com/vllm-project/vllm/pull/{PR}", "cutlass_tag": CUTLASS_TAG,
        "cutlass_commit": CUTLASS_COMMIT, "files": sources_json}
(src / "SOURCES.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8", newline="\n")
files["src/SOURCES.json"] = (src / "SOURCES.json").read_bytes()

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for k in sorted(files):
        z.writestr(zipfile.ZipInfo(k, date_time=(2026, 10, 29, 0, 0, 0)), files[k])
kit = buf.getvalue()
KIT_SHA = hashlib.sha256(kit).hexdigest()
KIT_B64 = base64.b64encode(kit).decode()
MANIFEST = {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}

CODE_TEMPLATE = r'''# E23 P4: build the vLLM PR #61224 SM89 block-FP8 CUTLASS GEMM as a standalone sm_89 torch extension (CPU kernel)
import base64, hashlib, io, json, os, shutil, subprocess, sys, tarfile, time, traceback, zipfile
from pathlib import Path
KIT_SHA = "@@KIT_SHA@@"
MANIFEST = @@MANIFEST@@
CUTLASS_TAG, CUTLASS_COMMIT = "@@CUTLASS_TAG@@", "@@CUTLASS_COMMIT@@"
CUOBJDUMP_PIN, CUOBJDUMP_WHEEL = "@@CUOBJDUMP_PIN@@", "@@CUOBJDUMP_WHEEL@@"
CUOBJDUMP_SHA256 = "@@CUOBJDUMP_SHA256@@"
CCCL130_PIN, CCCL130_WHEEL = "@@CCCL130_PIN@@", "@@CCCL130_WHEEL@@"
CCCL130_SHA256 = "@@CCCL130_SHA256@@"
WH_SLUG = "@@WH_SLUG@@"
KIT_B64 = "@@KIT_B64@@"
OUT, K, T0 = Path("/kaggle/working"), Path("/tmp/p4_kit"), time.time()
VENV, CUT, TOOLS = Path("/tmp/vllm-venv"), Path("/tmp/p4_cutlass"), Path("/tmp/p4_tools")
REPORT = {"kernel": "q27-e23-p4-sm89-fp8", "python": sys.version.split()[0], "steps": {}}
CTX = {}


def save():
    (OUT / "p4_report.json").write_text(json.dumps(REPORT, indent=1, default=str), encoding="utf-8")


def run(cmd, timeout=1800, env=None):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired as e:
        return "timeout", str(e.stdout or ""), str(e.stderr or "")
    except OSError as e:
        return "oserror", "", f"{type(e).__name__}: {e}"


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def step(name):
    def deco(fn):
        t = time.time()
        try:
            REPORT["steps"][name] = fn()
        except Exception as e:
            REPORT["steps"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}", "tb": traceback.format_exc()[-2500:]}
        REPORT["steps"][name]["sec"] = round(time.time() - t)
        save()
        print(f"[{name}]", json.dumps(REPORT["steps"][name], default=str)[:5000], flush=True)
        return fn
    return deco


@step("kit")   # vendored PR + vLLM sources, binding, build script, bench; sha256-checked
def _():
    kit = base64.b64decode(KIT_B64)
    assert hashlib.sha256(kit).hexdigest() == KIT_SHA, "kit sha256 mismatch"
    shutil.rmtree(K, ignore_errors=True)
    zipfile.ZipFile(io.BytesIO(kit)).extractall(K)
    bad = [n for n, h in MANIFEST.items() if sha256(K / n) != h]
    return {"ok": not bad, "files": len(MANIFEST), "bad": bad, "sha256": KIT_SHA}


@step("venv")  # vLLM 0.32.0 venv from the wheelhouse dataset, same installer call as the E22 flags kit
def _():
    inst = next(iter(sorted(Path("/kaggle/input").rglob("install_wheelhouse.py"))), None)
    assert inst is not None, "attach " + WH_SLUG
    rc, so, se = run([sys.executable, str(inst), str(VENV)], timeout=2400)
    vpy = next((l.split("=", 1)[1] for l in so.splitlines() if l.startswith("VENV_PYTHON=")), str(VENV / "bin/python"))
    cuda_home = next((l.split("=", 1)[1] for l in so.splitlines() if l.startswith("CUDA_HOME=")), "")
    CTX.update(vpy=vpy, cuda_home=cuda_home)
    probe = ("import json, torch, importlib.metadata as m; print(json.dumps({'torch': torch.__version__, "
             "'torch_cuda': torch.version.cuda, 'vllm': m.version('vllm'), 'nvcc': m.version('nvidia-cuda-nvcc'), "
             "'cccl': m.version('nvidia-cuda-cccl'), 'crt': m.version('nvidia-cuda-crt'), 'ninja': m.version('ninja'), "
             "'cuda_available': torch.cuda.is_available()}))")
    rc2, so2, se2 = run([vpy, "-c", probe], timeout=600)
    try:
        v = json.loads(so2.strip().splitlines()[-1])
    except Exception:
        v = {"rc": rc2, "tail": (so2 + se2)[-1500:]}
    rc3, so3, se3 = run(["c++", "--version"])
    ok = (rc == 0 and str(v.get("torch", "")).startswith("2.13.0") and v.get("torch_cuda") == "13.0"
          and v.get("vllm") == "0.32.0" and Path(cuda_home, "bin", "nvcc").exists())
    return {"ok": ok, "rc": rc, "installer": str(inst), "VENV_PYTHON": vpy, "CUDA_HOME": cuda_home, "versions": v,
            "host_cxx": (so3 or se3).splitlines()[:1], "tail": "" if ok else (so + se)[-2000:]}


@step("cutlass")  # CUTLASS at vLLM 0.32.0's CUTLASS_REVISION (include/ + tools/util/include/), commit verified
def _():
    shutil.rmtree(CUT, ignore_errors=True)
    st = {"tag": CUTLASS_TAG}
    rc, so, se = run(["git", "clone", "--depth", "1", "--branch", CUTLASS_TAG, "--filter=blob:none", "--no-checkout",
                      "https://github.com/NVIDIA/cutlass.git", str(CUT)], timeout=900)
    if rc == 0:
        run(["git", "-C", str(CUT), "sparse-checkout", "set", "--no-cone", "/include/", "/tools/util/include/",
             "/LICENSE.txt"])
        rc, so, se = run(["git", "-C", str(CUT), "checkout", CUTLASS_TAG], timeout=900)
        head = run(["git", "-C", str(CUT), "rev-parse", "HEAD"])[1].strip()
        st.update({"method": "git", "rc": rc, "head": head})
    if not (CUT / "include/cutlass/cutlass.h").exists():   # fallback: GitHub codeload tarball of the same tag
        shutil.rmtree(CUT, ignore_errors=True)
        tgz = Path("/tmp/cutlass.tgz")
        rc, so, se = run(["curl", "-sSL", "-o", str(tgz),
                          f"https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/{CUTLASS_TAG}"], 900)
        with tarfile.open(tgz) as tf:
            pax = (tf.pax_headers or {}).get("comment", "")
            tf.extractall("/tmp/cutlass_tgz", filter="data")
        shutil.move(str(next(Path("/tmp/cutlass_tgz").iterdir())), str(CUT))
        st.update({"method": "codeload tarball", "rc": rc, "tgz_sha256": sha256(tgz), "head": pax})
    st["n_headers"] = sum(1 for _ in (CUT / "include").rglob("*.h*")) if (CUT / "include").exists() else 0
    st["ok"] = st.get("head") == CUTLASS_COMMIT and st["n_headers"] > 500
    if not st["ok"]:
        st["tail"] = (so + se)[-1500:]
    return st


@step("cuobjdump")  # not in the nvcc wheel: pinned PyPI wheel, sha256-checked (info; p4_build also parses the fatbin)
def _():
    TOOLS.mkdir(parents=True, exist_ok=True)
    rc, so, se = run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary", ":all:", CUOBJDUMP_PIN, "-d",
                      str(TOOLS), "--disable-pip-version-check", "-q"], timeout=600)
    whl = TOOLS / CUOBJDUMP_WHEEL
    st = {"rc": rc, "wheel": whl.exists(), "path": None}
    if whl.exists():
        st["sha256_ok"] = sha256(whl) == CUOBJDUMP_SHA256
        if st["sha256_ok"]:
            with zipfile.ZipFile(whl) as z:
                (TOOLS / "cuobjdump").write_bytes(z.read("nvidia/cu13/bin/cuobjdump"))
            os.chmod(TOOLS / "cuobjdump", 0o755)
            st["path"] = CTX["cuobjdump"] = str(TOOLS / "cuobjdump")
            _, vo, ve = run([st["path"], "--version"])
            st["version"] = (vo + ve).strip().splitlines()[-1:]
    st["ok"] = bool(st["path"])
    if not st["ok"]:
        st["tail"] = (so + se)[-800:]
    return st


def build_env():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_PATH")}
    ch = CTX.get("cuda_home", "")
    env.update({"CUDA_HOME": ch, "PATH": f"{VENV / 'bin'}:{ch}/bin:{env.get('PATH', '')}", "TORCH_CUDA_ARCH_LIST": "8.9",
                "MAX_JOBS": "4", "TORCH_EXTENSIONS_DIR": "/tmp/p4_torch_ext"})
    return env


def build_once(tag, extra):
    cmd = [CTX["vpy"], str(K / "src/p4_build.py"), "--src", str(K / "src"), "--cutlass", str(CUT), "--out", str(OUT),
           "--build-dir", f"/tmp/p4_build_{tag}", "--jobs", "4"] + extra + (
               ["--cuobjdump", CTX["cuobjdump"]] if CTX.get("cuobjdump") else [])
    rc, so, se = run(cmd, timeout=5400, env=build_env())
    (OUT / f"p4_build_driver_{tag}.log").write_text(so + "\n---- stderr ----\n" + se, encoding="utf-8")
    line = next((l for l in so.splitlines() if l.startswith("P4_BUILD ")), None)
    try:
        br = json.loads((OUT / "p4_build_report.json").read_text())
    except Exception:
        br = {}
    if (OUT / "p4_build.log").exists():
        shutil.copy(OUT / "p4_build.log", OUT / f"p4_build_{tag}.log")
    s = br.get("steps", {})
    return {"ok": bool(br.get("all_ok")), "rc": rc, "summary": json.loads(line[9:]) if line else None,
            "cccl": br.get("cccl"), "so": br.get("so"), "build_sec": (s.get("build") or {}).get("sec"),
            "build_error": (s.get("build") or {}).get("error"), "nvcc_version": br.get("nvcc_version"),
            "host_cxx": br.get("host_cxx"),
            "sass": {k: (s.get("sass_sm89") or {}).get(k) for k in
                     ("ok", "cuobjdump", "elf_archs", "ptx_archs", "fatbin_elf_archs", "fatbin_ptx_archs", "sass_counts",
                      "sass_qmma_variants", "sass_functions", "note")},
            "symbols": {k: (s.get("symbols") or {}).get(k) for k in ("ok", "needed", "undefined_cuda_driver")},
            "ptxas": s.get("ptxas"), "load_cpu": s.get("load_cpu"),
            "tail": "" if br.get("all_ok") else (so + se)[-4000:]}


@step("build")  # src/p4_build.py with the venv python: build + SASS / symbols / ptxas / CPU-load checks
def _():
    # attempt 1: the venv's own CCCL (nvidia-cuda-cccl from the wheelhouse lock). The lock has 13.3.4.3.1 (CCCL 3.4.3)
    # while torch 2.13's cuda-toolkit 13.0.3 set pins 13.0.85 (CCCL 3.0.1, the CCCL that ships with nvcc 13.0.88):
    # if attempt 1 does not compile, attempt 2 puts that toolkit-matched CCCL first on the include path.
    a1 = build_once("venvcccl", [])
    if a1["ok"] or a1.get("so"):   # retry only when nothing was compiled
        a1["attempt"] = "venv cccl"
        return a1
    rc, so, se = run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary", ":all:", CCCL130_PIN, "-d",
                      str(TOOLS), "--disable-pip-version-check", "-q"], timeout=600)
    whl = TOOLS / CCCL130_WHEEL
    if not (whl.exists() and sha256(whl) == CCCL130_SHA256):
        a1["attempt2"] = {"ok": False, "error": "cccl 13.0.85 wheel download / sha256 failed", "tail": (so + se)[-600:]}
        return a1
    dst = Path("/tmp/p4_cccl130")
    shutil.rmtree(dst, ignore_errors=True)
    with zipfile.ZipFile(whl) as z:
        z.extractall(dst, [n for n in z.namelist() if n.startswith("nvidia/cu13/include/cccl/")])
    a2 = build_once("cccl130", ["--cccl-include", str(dst / "nvidia/cu13/include/cccl")])
    a2["attempt"] = "cccl 13.0.85 (toolkit-matched) after the venv cccl failed"
    a2["attempt1"] = {k: a1.get(k) for k in ("ok", "rc", "cccl", "build_error", "tail")}
    return a2


@step("include_graph_info")  # include closure against the real installed headers (information; nvcc decided above)
def _():
    vpy, ch = CTX["vpy"], CTX["cuda_home"]
    tdir = run([vpy, "-c", "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'include'))"])
    tinc = [tdir[1].strip().splitlines()[-1]]
    tinc.append(os.path.join(tinc[0], "torch", "csrc", "api", "include"))
    rc, so, se = run([vpy, str(K / "tools/check_includes.py"),
                      "--tu", str(K / "src/csrc/quantization/w8a8/cutlass/scaled_mm_blockwise_sm89_fp8.cu"),
                      "--tu", str(K / "src/p4_binding.cpp"), "-I", str(K / "src/csrc"), "-I", str(CUT / "include"),
                      "-I", str(CUT / "tools/util/include"), "-I", f"{ch}/include", "-I", f"{ch}/include/cccl",
                      *[x for d in tinc for x in ("--isystem", d)],
                      "--json", str(OUT / "include_graph_kaggle.json")], timeout=600)
    ig = json.loads((OUT / "include_graph_kaggle.json").read_text())
    return {"ok": ig["ok"], "files_reached": ig["files_reached"], "unresolved_other": ig["unresolved_other"]}


@step("bench_check_imports")  # the GPU bench's vLLM imports + signatures (real vLLM 0.32, CPU); .so loads there too
def _():
    rc, so, se = run([CTX["vpy"], str(K / "bench_fp8_gemm.py"), "--check-imports", "--so", str(OUT / "p4_sm89_fp8.so"),
                      "--out", str(OUT / "p4_bench_check_imports.json")], timeout=1800, env=build_env())
    try:
        ci = json.loads((OUT / "p4_bench_check_imports.json").read_text())
    except Exception:
        ci = {"all_ok": False}
    return {"ok": bool(ci.get("all_ok")), "rc": rc,
            "checks": {k: (v.get("ok"), v.get("info") if v.get("ok") else v.get("error"))
                       for k, v in (ci.get("checks") or {}).items()},
            "tail": "" if ci.get("all_ok") else (so + se)[-2500:]}


@step("handoff")  # files for the GPU notebook (attach this kernel's output through kernel_sources)
def _():
    shutil.copy(K / "bench_fp8_gemm.py", OUT / "bench_fp8_gemm.py")
    shutil.make_archive(str(OUT / "p4_src"), "zip", str(K / "src"))
    return {"ok": (OUT / "bench_fp8_gemm.py").exists() and (OUT / "p4_sm89_fp8.so").exists(),
            "outputs": sorted(p.name for p in OUT.iterdir() if p.is_file())}


GATES = ("kit", "venv", "cutlass", "build", "bench_check_imports", "handoff")
REPORT["all_ok"] = all(REPORT["steps"].get(g, {}).get("ok") for g in GATES)
REPORT["failed"] = [g for g in GATES if not REPORT["steps"].get(g, {}).get("ok")]
REPORT["info_not_gating"] = {g: REPORT["steps"].get(g, {}).get("ok") for g in ("cuobjdump", "include_graph_info")}
REPORT["sec"] = round(time.time() - T0)
save()
print("ALL OK" if REPORT["all_ok"] else "NOT OK " + str(REPORT["failed"]), "| total", REPORT["sec"], "s", flush=True)
'''
CODE = CODE_TEMPLATE
for key, val in {"@@KIT_SHA@@": KIT_SHA, "@@MANIFEST@@": repr(MANIFEST), "@@CUTLASS_TAG@@": CUTLASS_TAG,
                 "@@CUTLASS_COMMIT@@": CUTLASS_COMMIT, "@@CUOBJDUMP_PIN@@": CUOBJDUMP_PIN,
                 "@@CUOBJDUMP_WHEEL@@": CUOBJDUMP_WHEEL, "@@CUOBJDUMP_SHA256@@": CUOBJDUMP_SHA256,
                 "@@CCCL130_PIN@@": CCCL130_PIN, "@@CCCL130_WHEEL@@": CCCL130_WHEEL,
                 "@@CCCL130_SHA256@@": CCCL130_SHA256, "@@WH_SLUG@@": WH032_SLUG, "@@KIT_B64@@": KIT_B64}.items():
    assert CODE.count(key) == 1, key
    CODE = CODE.replace(key, val)
assert "@@" not in CODE
compile(CODE, "<p4 kernel>", "exec")
nbj = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [
    "# Q27 E23 P4: SM89 block-FP8 GEMM (vLLM PR #61224) as a standalone sm_89 torch extension\n",
    "CPU kernel, internet ON, pinned image, wheelhouse `" + WH032_SLUG + "` attached. Output: `p4_sm89_fp8.so`, "
    "`p4_report.json` (ALL OK / NOT OK), build logs, `bench_fp8_gemm.py` for the GPU notebook.\n"]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CODE.splitlines(True)}],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
KD = HERE / "kernel"
KD.mkdir(exist_ok=True)
nb_bytes = (json.dumps(nbj, indent=1) + "\n").encode("utf-8")
assert len(nb_bytes) < 400_000, len(nb_bytes)
(KD / "q27-e23-p4-sm89-fp8.ipynb").write_bytes(nb_bytes)
md = {"id": KERNEL_SLUG, "title": "Q27 E23 P4 SM89 FP8", "code_file": "q27-e23-p4-sm89-fp8.ipynb", "language": "python",
      "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
      "keywords": [], "dataset_sources": [WH032_SLUG], "kernel_sources": [], "competition_sources": [],
      "model_sources": [], "docker_image": IMAGE, "machine_shape": "None"}
(KD / "kernel-metadata.json").write_bytes((json.dumps(md, indent=2) + "\n").encode("utf-8"))
print("kit", len(kit), "bytes sha256", KIT_SHA[:16], "| notebook", len(nb_bytes), "bytes sha256",
      hashlib.sha256(nb_bytes).hexdigest()[:16], "|", KD)
