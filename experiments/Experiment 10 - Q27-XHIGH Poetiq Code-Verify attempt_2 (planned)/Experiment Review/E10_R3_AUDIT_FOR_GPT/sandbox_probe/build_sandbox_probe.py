"""Build the E10 Kaggle CPU sandbox probe notebook (Claude, 2026-10-14).

It embeds R3's Poetiq core cell (cell 5) VERBATIM, so the probe exercises exactly the sandbox code R3 ships
(poetiq_sandbox_run: util-linux unshare --user --mount --net --pid + bind mounts + chroot + seccomp).
No model, no GPU, no internet. Output: /kaggle/working/sandbox_probe.json and a SANDBOX_VERDICT line.
usage: python build_sandbox_probe.py
"""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R3 = HERE.parents[2] / "Experiment Review" / "R3 bundle (extracted, untrusted)" / "Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R3.ipynb"
R3_SHA = "4418ebbf5ccc1665f9dff9ee1e6c19a649e4bd23e913fa635bbfc4fb0a898853"
OUT = HERE / "E10_SANDBOX_PROBE_CPU.ipynb"

raw = R3.read_bytes()
assert hashlib.sha256(raw).hexdigest() == R3_SHA, "R3 notebook changed"
core = "".join(json.loads(raw)["cells"][5]["source"])
assert "def poetiq_sandbox_run" in core and "def poetiq_verify_code" in core

ENV = r'''# E10 sandbox probe, part 1: environment facts (no model, no GPU)
import os, sys, json, time, shutil, subprocess, platform, ctypes.util, glob
PROBE = {"t0": time.time()}
def _read(p):
    try: return open(p).read().strip()
    except Exception as e: return f"<{type(e).__name__}>"
def _sh(cmd, timeout=20):
    try:
        cp = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return {"rc": cp.returncode, "out": cp.stdout[-600:], "err": cp.stderr[-600:]}
    except Exception as e:
        return {"rc": None, "err": f"{type(e).__name__}: {e}"[:600]}
status = _read("/proc/self/status")
PROBE["env"] = {
    "python": sys.version.split()[0], "executable": sys.executable, "prefix": sys.prefix, "base_prefix": sys.base_prefix,
    "platform": platform.platform(), "proc_version": _read("/proc/version"), "nproc": os.cpu_count(),
    "uid": os.getuid(), "id": _sh(["id"]),
    "status": {l.split(":")[0]: l.split(":", 1)[1].strip() for l in status.splitlines()
               if l.split(":")[0] in ("Seccomp", "Seccomp_filters", "NoNewPrivs", "CapEff", "CapBnd", "CapPrm")},
    "max_user_namespaces": _read("/proc/sys/user/max_user_namespaces"),
    "unprivileged_userns_clone": _read("/proc/sys/kernel/unprivileged_userns_clone"),
    "apparmor_restrict_userns": _read("/proc/sys/kernel/apparmor_restrict_unprivileged_userns"),
    "apparmor_current": _read("/proc/self/attr/current"),
    "which": {b: shutil.which(b) for b in ("unshare", "mount", "chroot", "sh")},
    "libseccomp": ctypes.util.find_library("seccomp"),
    "libseccomp_files": sorted(glob.glob("/usr/lib/x86_64-linux-gnu/libseccomp.so*") + glob.glob("/lib/x86_64-linux-gnu/libseccomp.so*")),
    "root_fs": _sh(["sh", "-c", "grep ' / ' /proc/mounts | head -3"]),
    "kaggle_input": sorted(os.listdir("/kaggle/input"))[:10] if os.path.isdir("/kaggle/input") else None,
}
for m in ("numpy", "scipy", "cv2"):
    try: PROBE["env"][m] = __import__(m).__version__
    except Exception as e: PROBE["env"][m] = f"<{type(e).__name__}>"
print(json.dumps(PROBE["env"], indent=1))
'''

STEPS = r'''# E10 sandbox probe, part 2: the namespace steps one by one (what R3 needs, smallest first)
STEP_CMDS = {
    "userns":            ["unshare", "--user", "--map-root-user", "id"],
    "userns+mount":      ["unshare", "--user", "--map-root-user", "--mount", "true"],
    "userns+mount+net":  ["unshare", "--user", "--map-root-user", "--mount", "--net", "true"],
    "R3 flags (+pid)":   ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "true"],
    "R3 flags + bind ro mount": ["unshare", "--user", "--map-root-user", "--mount", "--net", "--pid", "--fork", "sh", "-c",
        "mount --make-rprivate / && mkdir -p /tmp/e10probe && mount --bind /usr /tmp/e10probe && "
        "mount -o remount,bind,ro /tmp/e10probe && echo BIND_RO_OK"],
}
PROBE["steps"] = {k: _sh(v) for k, v in STEP_CMDS.items()}
for k, v in PROBE["steps"].items():
    print(f"{k:28s} rc={v.get('rc')} out={v.get('out','')[:80]!r} err={v.get('err','')[:200]!r}")
'''

CASES = r'''# E10 sandbox probe, part 3: R3's own poetiq_sandbox_run on real programs (expected result in brackets)
G = [[1, 0], [0, 2]]
CASES = {
    "identity [ok]":            ("import numpy as np\ndef transform(grid):\n    return grid", True),
    "float zeros [ok, floats]": ("import numpy as np\ndef transform(grid):\n    return np.zeros(grid.shape)", True),
    "debug print [ok]":         ("import numpy as np\ndef transform(grid):\n    print('debug', grid.shape)\n    return grid[::-1]", True),
    "scipy.ndimage [ok]":       ("import numpy as np\nfrom scipy import ndimage\ndef transform(grid):\n    lab, n = ndimage.label(grid > 0)\n    return (lab > 0).astype(int)", True),
    "cv2 [ok if installed]":    ("import numpy as np, cv2\ndef transform(grid):\n    return cv2.flip(grid.astype(np.uint8), 1).astype(int)", True),
    "read /kaggle/input [blocked]": ("import os\ndef transform(grid):\n    return [[len(os.listdir('/kaggle/input'))]]", False),
    "write /kaggle/working [blocked]": ("def transform(grid):\n    open('/kaggle/working/E10_SANDBOX_ESCAPE.txt','w').write('x')\n    return grid", False),
    "subprocess [blocked]":     ("import subprocess\ndef transform(grid):\n    subprocess.run(['true'])\n    return grid", False),
    "socket [blocked]":         ("import socket\ndef transform(grid):\n    socket.socket()\n    return grid", False),
    "infinite loop [timeout]":  ("def transform(grid):\n    while True: pass", False),
}
PROBE["cases"] = {}
for name, (code, want_ok) in CASES.items():
    t = time.perf_counter(); ok, payload = poetiq_sandbox_run(code, G, 5.0); dt = round(time.perf_counter() - t, 2)
    PROBE["cases"][name] = {"ok": ok, "as_expected": ok == want_ok, "sec": dt, "payload": str(payload)[-300:]}
    print(f"{name:34s} ok={ok!s:5s} expected={want_ok!s:5s} {dt:5.2f}s  {str(payload)[-140:]!r}")
PROBE["escape_file_exists"] = os.path.exists("/kaggle/working/E10_SANDBOX_ESCAPE.txt")
_t = []
for _ in range(10):
    t = time.perf_counter(); poetiq_sandbox_run("def transform(grid):\n    return grid", G, 5.0); _t.append(time.perf_counter() - t)
PROBE["identity_sec"] = {"mean": round(sum(_t) / len(_t), 3), "max": round(max(_t), 3)}
_t = []
for _ in range(10):
    t = time.perf_counter()
    subprocess.run([sys.executable, "-c", "import numpy, scipy"], capture_output=True, timeout=30); _t.append(time.perf_counter() - t)
PROBE["plain_subprocess_numpy_scipy_sec"] = {"mean": round(sum(_t) / len(_t), 3), "max": round(max(_t), 3)}
print("timing: R3 sandbox per call", PROBE["identity_sec"], "| plain subprocess", PROBE["plain_subprocess_numpy_scipy_sec"])
'''

VERDICT = r'''# E10 sandbox probe, part 4: one real ARC task through poetiq_verify_code, then the verdict
try:
    _root = next(p for p in ("/kaggle/input/competitions/arc-prize-2026-arc-agi-2", "/kaggle/input/arc-prize-2026-arc-agi-2") if os.path.isdir(p))
    _tasks = json.load(open(_root + "/arc-agi_evaluation_challenges.json"))
    _tid = sorted(_tasks)[0]
    t = time.perf_counter()
    _v = poetiq_verify_code("import numpy as np\ndef transform(grid):\n    return grid", _tasks[_tid], 5.0)
    PROBE["verify_real_task"] = {"task": _tid, "sec": round(time.perf_counter() - t, 2), "score": _v["score"],
                                 "test_usable": _v["test_usable"], "train_errors": [r["error"] for r in _v["train_results"]][:2]}
except Exception as e:
    PROBE["verify_real_task"] = {"error": f"{type(e).__name__}: {e}"[:400]}
print("verify on a real task:", PROBE["verify_real_task"])
_c = PROBE["cases"]
_works = _c["identity [ok]"]["ok"] and _c["debug print [ok]"]["ok"]
_contained = all(v["as_expected"] for k, v in _c.items() if "[blocked]" in k or "[timeout]" in k) and not PROBE["escape_file_exists"]
PROBE["verdict"] = "PASS" if (_works and _contained) else ("FAIL_SANDBOX_UNAVAILABLE" if not _works else "FAIL_NOT_CONTAINED")
PROBE["wall_sec"] = round(time.time() - PROBE["t0"], 1)
os.makedirs("/kaggle/working", exist_ok=True)
json.dump(PROBE, open("/kaggle/working/sandbox_probe.json", "w"), indent=1, default=str)
print("SANDBOX_VERDICT:", PROBE["verdict"], "| works", _works, "| contained", _contained, "| wall", PROBE["wall_sec"], "s")
'''

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(True)}

nb = {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# E10 sandbox probe (CPU)\n\nDoes R3's generated-code sandbox (unshare user/mount/net/pid namespaces + bind mounts + "
            "chroot + seccomp) work inside the Kaggle container? R3's Poetiq core cell is embedded verbatim "
            f"(R3 notebook sha256 `{R3_SHA[:16]}…`). No model, no GPU, no internet.\n"]},
        code(ENV), code(STEPS),
        code("# R3 Poetiq core cell 5, verbatim\n" + core), code(CASES), code(VERDICT)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"probe{i}"
data = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
OUT.write_text(data, encoding="utf-8")
print("BUILT", OUT.name, len(data), "bytes sha256", hashlib.sha256(data.encode()).hexdigest()[:16])
