"""E23 P2 kernel builder: the Merlin 3.0.0 engine payload for the offline GPU notebook, built and tested on a Kaggle CPU
kernel (internet ON, pinned image: Ubuntu 22.04.5, glibc 2.35, GCC 11.4, Python 3.12.13, uv + mamba present; seen by
the q27-e23-envprobe kernel on 2026-10-29). Output /kaggle/working/merlin_offline/:
  merlin-engine.tar (tag merlin-v3.0.0 checkout with .git), vllm-0.32.0+merlin.7002f56537-cp313-...whl (sha256 checked),
  merlin_uvpy.tar.gz (uv CPython 3.13 at /tmp/merlin_uvpy), merlin_gcc13.tar.gz (conda-forge GCC 13 at /tmp/merlin_gcc13),
  wheelhouse/ (every package of release/requirements-0.32.lock, cp313 manylinux, hash-checked),
  glibc235_fix.py, rehash_products.py, glibc238_shim.c + libglibc238_shim.so, install_merlin_offline.sh.
Then, in the same kernel, the fixed-prefix trees are deleted and install_merlin_offline.sh installs from the payload
only (UV_OFFLINE, no index), exactly as the GPU notebook will; the engine import is tested with the shim and a CUDA
driver stub. Kernel foysalemonshanto/q27-e23-p2-merlin-env (Build: kernel/)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_SLUG = "foysalemonshanto/q27-e23-p2-merlin-env"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
FILES = {n: (HERE / n).read_text(encoding="utf-8").replace("\r\n", "\n")
         for n in ("glibc235_fix.py", "rehash_products.py", "glibc238_shim.c", "install_merlin_offline.sh")}

CODE = r'''# E23 P2: Merlin 3.0.0 offline payload + offline install test on the pinned image (CPU, internet ON)
import json, os, re, shutil, subprocess, sys, time, hashlib
from pathlib import Path
T0 = time.time()
FILES = json.loads(''' + repr(json.dumps(FILES)) + r''')
TAG, COMMIT = "merlin-v3.0.0", "ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5"
WHEEL_NAME = "vllm-0.32.0+merlin.7002f56537-cp313-cp313-linux_x86_64.whl"
WHEEL_SHA = "5c33b3dce51b8255b0c1a95a930edb24c1a481a023327d8d2d8f76e8be45c82d"
WHEEL_URL = f"https://github.com/halt95/merlin-engine/releases/download/{TAG}/{WHEEL_NAME}"
W = Path("/kaggle/working"); PAY = W / "merlin_offline"; T = Path("/tmp/e23p2"); LOGS = T / "logs"
for p in (PAY, T, LOGS):
    p.mkdir(parents=True, exist_ok=True)
R = {"steps": {}, "errors": []}
os.environ.pop("CUDA_HOME", None)


def sh(name, cmd, timeout=3600, env=None, ok_rc=(0,)):
    t = time.time()
    log = LOGS / f"{name}.log"
    with open(log, "w") as f:
        r = subprocess.run(["bash", "-c", "set -o pipefail; " + cmd], stdout=f, stderr=subprocess.STDOUT, timeout=timeout,
                           env={**os.environ, **(env or {})})
    txt = log.read_text(errors="replace")
    rec = {"rc": r.returncode, "sec": round(time.time() - t, 1), "tail": txt[-2500:]}
    R["steps"][name] = rec
    print(f"[{time.time() - T0:7.1f}s] {name}: rc {r.returncode} ({rec['sec']} s)", flush=True)
    if r.returncode not in ok_rc:
        print(txt[-2500:], flush=True)
    return r.returncode, txt


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fail(msg):
    R["errors"].append(msg)
    print("ERROR:", msg, flush=True)


try:
    sh("env", "gcc --version | head -1; ldd --version | head -1; uv --version; nproc; free -g | head -2; "
              "df -h /kaggle/working /tmp | cat; python3 --version; git --version")
    for n, s in FILES.items():
        (PAY / n).write_text(s)
    os.chmod(PAY / "install_merlin_offline.sh", 0o755)

    # 1. engine source at the release tag, with .git (install-env.sh checks the checkout's commit / tree)
    shutil.rmtree(T / "src", ignore_errors=True)
    rc, _ = sh("clone", f"git clone --quiet --depth 1 --branch {TAG} https://github.com/halt95/merlin-engine.git {T}/src/merlin-engine "
                        f"&& git -C {T}/src/merlin-engine rev-parse HEAD", timeout=900)
    head = (R["steps"]["clone"]["tail"].strip().splitlines() or [""])[-1]
    R["merlin_commit"] = head
    if rc or head != COMMIT:
        fail(f"merlin checkout: rc {rc}, HEAD {head} != {COMMIT}")
    sh("engine_tar", f"tar -cf {PAY}/merlin-engine.tar -C {T}/src merlin-engine")
    tag_sum = (T / "src/merlin-engine/release/merlin-wheel.sha256").read_text().split()
    R["tag_wheel_sha256"] = tag_sum
    if tag_sum[:2] != [WHEEL_SHA, WHEEL_NAME]:
        fail(f"release/merlin-wheel.sha256 at the tag is {tag_sum}, expected {WHEEL_SHA} {WHEEL_NAME}")

    # 2. the release wheel
    rc, _ = sh("wheel", f"curl -fsSL --retry 5 -o {PAY}/{WHEEL_NAME} {WHEEL_URL} && ls -la {PAY}/{WHEEL_NAME}", timeout=1800)
    got = sha256(PAY / WHEEL_NAME) if (PAY / WHEEL_NAME).is_file() else None
    R["wheel"] = {"sha256": got, "ok": got == WHEEL_SHA}
    if got != WHEEL_SHA:
        fail(f"wheel sha256 {got} != {WHEEL_SHA}")

    # 3. GCC 13 (conda-forge) at the fixed prefix /tmp/merlin_gcc13: install-env.sh requires CC/CXX >= 13
    shutil.rmtree("/tmp/merlin_gcc13", ignore_errors=True)
    # /usr/local/bin/mamba on this image is NOT conda's mamba (v1 run: a Python test runner's CLI); use micromamba's static
    # release binary (sha256-checked against its own .sha256 asset)
    MM = "https://github.com/mamba-org/micromamba-releases/releases/download/2.10.0-0/micromamba-linux-64"
    rc, _ = sh("gcc13", f"curl -fsSL --retry 5 -o /tmp/micromamba {MM} && curl -fsSL --retry 5 -o /tmp/micromamba.sha256 {MM}.sha256 && "
                        "echo \"$(head -c 64 /tmp/micromamba.sha256)  /tmp/micromamba\" | sha256sum -c - && chmod +x /tmp/micromamba && "
                        "/tmp/micromamba --version && export MAMBA_ROOT_PREFIX=/tmp/mambaroot && "
                        "/tmp/micromamba create -y -q -p /tmp/merlin_gcc13 -c conda-forge "
                        "--override-channels 'gcc_linux-64=13' 'gxx_linux-64=13' && "
                        "/tmp/merlin_gcc13/bin/x86_64-conda-linux-gnu-gcc -dumpversion && "
                        "/tmp/merlin_gcc13/bin/x86_64-conda-linux-gnu-g++ -dumpversion && "
                        "echo 'int main(){return 0;}' > /tmp/t.cc && /tmp/merlin_gcc13/bin/x86_64-conda-linux-gnu-g++ -std=c++20 /tmp/t.cc -o /tmp/t && /tmp/t && echo CXX20 OK",
                        timeout=2400)
    R["gcc13"] = {"ok": rc == 0 and "CXX20 OK" in R["steps"]["gcc13"]["tail"]}
    if not R["gcc13"]["ok"]:
        fail("GCC 13 from conda-forge failed")
    sh("gcc13_tar", f"tar -czf {PAY}/merlin_gcc13.tar.gz -C /tmp merlin_gcc13 && du -sh /tmp/merlin_gcc13 {PAY}/merlin_gcc13.tar.gz")

    # 4. CPython 3.13 (uv-managed) at the fixed prefix /tmp/merlin_uvpy
    shutil.rmtree("/tmp/merlin_uvpy", ignore_errors=True)
    rc, _ = sh("py313", "UV_PYTHON_INSTALL_DIR=/tmp/merlin_uvpy uv python install 3.13 && ls /tmp/merlin_uvpy && "
                        "UV_PYTHON_INSTALL_DIR=/tmp/merlin_uvpy UV_PYTHON_DOWNLOADS=never uv python find 3.13", timeout=900)
    if rc:
        fail("uv python install 3.13 failed")
    sh("py313_tar", f"tar -czf {PAY}/merlin_uvpy.tar.gz -C /tmp merlin_uvpy && du -sh {PAY}/merlin_uvpy.tar.gz")

    # 5. wheelhouse: every pinned package of the lock, cp313 manylinux (<= glibc 2.35), hash-checked by pip
    lock = T / "src/merlin-engine/release/requirements-0.32.lock"
    plats = " ".join(f"--platform manylinux_2_{m}_x86_64" for m in range(17, 36)) + \
        " --platform manylinux2014_x86_64 --platform manylinux1_x86_64 --platform linux_x86_64"
    rc, _ = sh("wheelhouse", f"python3 -m pip download --no-deps --only-binary=:all: --require-hashes --python-version 3.13 "
                             f"--implementation cp --abi cp313 --abi abi3 --abi none {plats} -r {lock} -d {PAY}/wheelhouse "
                             f"--progress-bar off -q", timeout=5400)
    pins = {re.sub(r"[-_.]+", "-", m.group(1)).lower() for m in re.finditer(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==", lock.read_text(), re.M)}
    have = {re.sub(r"[-_.]+", "-", p.name.split("-")[0]).lower() for p in (PAY / "wheelhouse").glob("*.whl")}
    R["wheelhouse"] = {"pins": len(pins), "wheels": len(list((PAY / "wheelhouse").glob("*.whl"))), "missing": sorted(pins - have),
                       "bytes": sum(p.stat().st_size for p in (PAY / "wheelhouse").glob("*"))}
    if rc or R["wheelhouse"]["missing"]:
        fail(f"wheelhouse: rc {rc}, missing {R['wheelhouse']['missing']}")

    # 6. the shim (system GCC 11; plain C, no version script)
    rc, _ = sh("shim", f"gcc -O2 -shared -fPIC -o {PAY}/libglibc238_shim.so {PAY}/glibc238_shim.c && "
                       f"nm -D --defined-only {PAY}/libglibc238_shim.so | grep isoc23")
    if rc:
        fail("shim build failed")

    # 7. diagnosis on the wheel's own .so files (before any install): what needs > 2.35, ldd before / after the fix
    D = T / "diag"
    shutil.rmtree(D, ignore_errors=True)
    D.mkdir()
    sh("diag_unzip", f"cd {D} && python3 -c \"import zipfile; z=zipfile.ZipFile('{PAY}/{WHEEL_NAME}'); "
                     f"[z.extract(n) for n in z.namelist() if n.endswith('.so') or '.so.' in n.split('/')[-1]]\" && find . -name '*.so*' | head -40")
    sos = sorted(str(p) for p in D.rglob("*.so*") if p.is_file())
    rc, txt = sh("diag_scan", f"python3 -I {PAY}/glibc235_fix.py --dry-run " + " ".join(sos), ok_rc=(0, 2))
    m = re.search(r"^GLIBCFIX (.*)$", txt, re.M)
    R["diag_scan"] = json.loads(m.group(1)) if m else None
    if not m or rc:
        fail("glibc scan of the wheel .so files refused or failed (a symbol the shim does not cover?)")
    stub = T / "stub"
    stub.mkdir(exist_ok=True)
    if not (stub / "libcuda.so.1").exists():
        os.symlink("/usr/local/cuda-12.8/lib64/stubs/libcuda.so", stub / "libcuda.so.1")
    R["stub"] = str(stub)

    # 8. offline install from the payload only (the GPU notebook's exact path): fixed-prefix trees removed first
    shutil.rmtree("/tmp/merlin_gcc13", ignore_errors=True)
    shutil.rmtree("/tmp/merlin_uvpy", ignore_errors=True)
    shutil.rmtree("/tmp/merlin_off", ignore_errors=True)
    rc, txt = sh("offline_install", f"bash {PAY}/install_merlin_offline.sh {PAY} /tmp/merlin_off", timeout=5400)
    kv = dict(re.findall(r"^(MERLIN_[A-Z]+)=(.*)$", txt, re.M))
    R["offline_install"] = {"rc": rc, "ok": rc == 0 and "INSTALL OK" in txt, "vars": kv,
                            "glibcfix": [json.loads(x) for x in re.findall(r"^GLIBCFIX (.*)$", txt, re.M)],
                            "rehash": [json.loads(x) for x in re.findall(r"^REHASH (.*)$", txt, re.M)],
                            "verified": "ENVIRONMENT VERIFIED" in txt}
    if not R["offline_install"]["ok"] or not R["offline_install"]["verified"]:
        fail("offline install / --verify failed")

    # 9. load tests on the installed, patched tree: ldd (with / without shim) and the engine import
    if R["offline_install"]["ok"]:
        py, tree, shim = kv["MERLIN_VENV"] + "/bin/python", kv["MERLIN_TREE"], kv["MERLIN_SHIM"]
        rc, txt = sh("libpaths", f"{py} -c \"import torch, sysconfig, os; print(os.path.dirname(torch.__file__) + '/lib'); "
                                 f"print(sysconfig.get_paths()['purelib'] + '/nvidia/cu13/lib')\"")
        lp = ":".join(txt.strip().splitlines()[-2:] + [str(stub)])
        C = f"{tree}/vllm/_C_stable_libtorch.abi3.so"
        # v3: ldd prints "weak version `GLIBC_2.38' not found" for the weakened need (informational, ld.so does not fail on
        # a weak version); the failures are "undefined symbol" and "=> not found" (a missing library)
        sh("ldd_noshim", f"LD_LIBRARY_PATH={lp} ldd -r {C} 2>&1 | grep -E 'GLIBC|isoc23|not found' || echo CLEAN", ok_rc=(0, 1))
        rc, txt = sh("ldd_shim", f"LD_PRELOAD={shim} LD_LIBRARY_PATH={lp} ldd -r {C} 2>&1 | grep -E 'GLIBC|isoc23|not found' || echo CLEAN",
                     ok_rc=(0, 1))
        bad = [l for l in txt.splitlines() if "undefined symbol" in l or "=> not found" in l]
        R["ldd_shim_clean"] = not bad
        R["ldd_shim_lines"] = txt.strip().splitlines()[-10:]
        if not R["ldd_shim_clean"]:
            fail("ldd -r with the shim still reports GLIBC / isoc23 / not-found lines")
        imp = (f"import torch, vllm; import vllm._C_stable_libtorch as c; print('IMPORT OK', vllm.__version__, torch.__version__, "
               f"vllm.__file__)")
        sh("import_noshim", f"cd /tmp && LD_LIBRARY_PATH={stub} VLLM_USE_RUST_FRONTEND=0 {py} -c \"{imp}\"", ok_rc=(0, 1))
        rc, txt = sh("import_shim", f"cd /tmp && LD_PRELOAD={shim} LD_LIBRARY_PATH={stub} VLLM_USE_RUST_FRONTEND=0 {py} -c \"{imp}\"",
                     ok_rc=(0, 1))
        R["import_shim_ok"] = rc == 0 and "IMPORT OK" in txt
        if not R["import_shim_ok"]:
            fail("engine import with the shim failed")
        # launcher facts for the GPU notebook (profiles, dry-run commands); informational
        sh("profiles", f"ls {tree}/tools/merlin/profiles; ls {tree}/models/qwen38-27b", ok_rc=(0, 1, 2))
        for tp in (1, 2, 4):   # v3: merlin_cli serve --dry-run (launch.py has no --dry-run); simulated 4 x L4, no P2P
            sh(f"dryrun_tp{tp}", f"cd {tree} && {py} merlin_cli.py serve 27b --tp {tp} --dry-run --simulate 4xl4:nop2p",
               ok_rc=tuple(range(256)), timeout=300)
except Exception as e:
    import traceback
    fail(f"fatal: {e!r}")
    R["traceback"] = traceback.format_exc()[-3000:]
finally:
    shutil.rmtree(PAY / "__pycache__", ignore_errors=True)
    R["payload"] = {p.name: p.stat().st_size for p in PAY.iterdir() if p.is_file()}
    R["payload_bytes"] = sum(p.stat().st_size for p in PAY.rglob("*") if p.is_file())
    R["output_ok"] = R["payload_bytes"] < 19e9
    if not R["output_ok"]:
        fail(f"payload {R['payload_bytes'] / 1e9:.2f} GB too big")
    R["all_ok"] = not R["errors"]
    R["sec"] = round(time.time() - T0)
    (W / "p2_report.json").write_text(json.dumps(R, indent=1))
    shutil.copytree(LOGS, W / "p2_logs", dirs_exist_ok=True)
    print(json.dumps({k: R.get(k) for k in ("merlin_commit", "wheel", "gcc13", "wheelhouse", "offline_install", "ldd_shim_clean",
                                             "import_shim_ok", "payload_bytes", "errors")}, default=str)[:6000], flush=True)
    print("ALL OK" if R["all_ok"] else "NOT OK", "|", R["sec"], "s", flush=True)
'''
compile(CODE, "<p2 kernel>", "exec")
nbj = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CODE.splitlines(True)}],
       "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
KD = HERE / "kernel"
KD.mkdir(exist_ok=True)
(KD / "q27-e23-p2-merlin-env.ipynb").write_text(json.dumps(nbj, indent=1), encoding="utf-8", newline="\n")
(KD / "kernel-metadata.json").write_text(json.dumps({
    "id": KERNEL_SLUG, "title": "Q27 E23 P2 Merlin Env", "code_file": "q27-e23-p2-merlin-env.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
    "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
    "model_sources": [], "docker_image": IMAGE, "machine_shape": "None"}, indent=2), encoding="utf-8", newline="\n")
print("kernel", KD, len(CODE), "chars")
