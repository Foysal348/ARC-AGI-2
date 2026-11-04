"""Build the Kaggle CPU probe for R5's sandbox (GPT R5: clone3 -> ENOSYS, non-raising self-test) (Claude, 2026-10-14).
Same cases as the R4 probe; R5's self-test is already non-raising, so cell 5 is embedded fully verbatim.

Embeds R4's Poetiq core cell (cell 5) VERBATIM, except that the final self-test line is wrapped in try/except so the
probe keeps going and reports every case. Extra cases: thread creation (glibc 2.35 pthread_create tries clone3 first,
and R4 makes clone3 return EPERM), OpenCV ops, writes/deletes in /kaggle/working as UID 65534, timing.
No model, no GPU, no internet. Output: /kaggle/working/sandbox_probe_r5.json and a SANDBOX_R5_VERDICT line.
usage: python build_sandbox_probe_r4.py
"""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
R4 = HERE.parents[2] / "Acknowledgement" / "E10_POETIQ_R4T_SAFE_SANDBOX_R5.ipynb"
R4_SHA = "68b0d28f758476336182a5df8a522964982ed5329241f5540ff2ce9a4cd7cfca"
OUT = HERE / "E10_SANDBOX_PROBE_R5_CPU.ipynb"

raw = R4.read_bytes()
assert hashlib.sha256(raw).hexdigest() == R4_SHA, "R5 notebook changed"
core = "".join(json.loads(raw)["cells"][5]["source"])
assert "except BaseException as _e10_sandbox_error" in core and "errno.ENOSYS, clone3_nr" in core, "not the R5 core"

ENV = r'''# R4 sandbox probe, part 1: environment
import os, sys, json, time, stat, subprocess, platform
PROBE = {"t0": time.time()}
def _mode(p):
    try:
        s = os.stat(p); return {"mode": oct(stat.S_IMODE(s.st_mode)), "uid": s.st_uid, "gid": s.st_gid}
    except Exception as e: return f"<{type(e).__name__}>"
PROBE["env"] = {"python": sys.version.split()[0], "executable": sys.executable, "euid": os.geteuid(),
                "glibc": platform.libc_ver(), "nproc": os.cpu_count(),
                "perms": {p: _mode(p) for p in ("/kaggle", "/kaggle/working", "/kaggle/input", "/tmp")}}
print(json.dumps(PROBE["env"], indent=1))
'''

CASES = r'''# R4 sandbox probe, part 3: cases through R4's own poetiq_sandbox_run (expected in brackets)
PROBE["selftest"] = E10_SANDBOX_SELFTEST
root_file = "/kaggle/working/e10_probe_root_owned.txt"
open(root_file, "w").write("original")
G = [[1, 0], [0, 2]]
CASES = {
    "identity [ok]": ("import numpy as np\ndef transform(grid):\n    return grid", True),
    "float zeros [ok]": ("import numpy as np\ndef transform(grid):\n    return np.zeros(grid.shape)", True),
    "scipy.ndimage [ok]": ("import numpy as np\nfrom scipy import ndimage\ndef transform(grid):\n    lab, n = ndimage.label(grid > 0)\n    return (lab > 0).astype(int)", True),
    "cv2 flip [ok]": ("import numpy as np, cv2\ndef transform(grid):\n    return cv2.flip(grid.astype(np.uint8), 1).astype(int)", True),
    "cv2 resize+blur 300x300 [ok]": ("import numpy as np, cv2\ndef transform(grid):\n    a = np.random.randint(0, 9, (300, 300)).astype(np.uint8)\n    b = cv2.GaussianBlur(cv2.resize(a, (600, 600)), (5, 5), 0)\n    return grid", True),
    "python thread [ok]": ("import threading\ndef transform(grid):\n    t = threading.Thread(target=lambda: None); t.start(); t.join()\n    return grid", True),
    "read /kaggle/input [allowed by design]": ("import os\ndef transform(grid):\n    return [[min(9, len(os.listdir('/kaggle/input')))]]", True),
    "write /kaggle/working new file [blocked]": ("def transform(grid):\n    open('/kaggle/working/E10_ESCAPE.txt','w').write('x')\n    return grid", False),
    "overwrite root-owned file in /kaggle/working [blocked]": (f"def transform(grid):\n    open({root_file!r},'w').write('MODIFIED')\n    return grid", False),
    "delete root-owned file in /kaggle/working [blocked]": (f"import os\ndef transform(grid):\n    os.remove({root_file!r})\n    return grid", False),
    "subprocess [blocked]": ("import subprocess\ndef transform(grid):\n    subprocess.run(['/usr/bin/id'])\n    return grid", False),
    "os.fork [blocked]": ("import os\ndef transform(grid):\n    os.fork()\n    return grid", False),
    "socket [blocked]": ("import socket\ndef transform(grid):\n    socket.socket()\n    return grid", False),
    "memory 6 GB [blocked]": ("import numpy as np\ndef transform(grid):\n    a = np.ones((6 * 1024**3,), dtype=np.uint8)\n    return grid", False),
    "infinite loop [timeout]": ("def transform(grid):\n    while True: pass", False),
}
PROBE["cases"] = {}
for name, (code, want_ok) in CASES.items():
    t = time.perf_counter(); ok, payload = poetiq_sandbox_run(code, G, 5.0); dt = round(time.perf_counter() - t, 2)
    PROBE["cases"][name] = {"ok": ok, "as_expected": ok == want_ok, "sec": dt, "payload": str(payload)[-300:]}
    print(f"{name:52s} ok={ok!s:5s} expected={want_ok!s:5s} {dt:5.2f}s  {str(payload)[-150:]!r}")
PROBE["escape_file_exists"] = os.path.exists("/kaggle/working/E10_ESCAPE.txt")
PROBE["root_file_intact"] = os.path.exists(root_file) and open(root_file).read() == "original"
_t = []
for _ in range(10):
    t = time.perf_counter(); poetiq_sandbox_run("def transform(grid):\n    return grid", G, 5.0); _t.append(time.perf_counter() - t)
PROBE["identity_sec"] = {"mean": round(sum(_t) / len(_t), 3), "max": round(max(_t), 3)}
print("self-test:", PROBE["selftest"]); print("timing per call:", PROBE["identity_sec"])
'''

VERDICT = r'''# R4 sandbox probe, part 4: one real task, then the verdict
try:
    _root = next(p for p in ("/kaggle/input/competitions/arc-prize-2026-arc-agi-2", "/kaggle/input/arc-prize-2026-arc-agi-2") if os.path.isdir(p))
    _tasks = json.load(open(_root + "/arc-agi_evaluation_challenges.json")); _tid = sorted(_tasks)[0]
    t = time.perf_counter(); _v = poetiq_verify_code("import numpy as np\ndef transform(grid):\n    return grid.astype(float)", _tasks[_tid], 5.0)
    PROBE["verify_real_task"] = {"task": _tid, "sec": round(time.perf_counter() - t, 2), "score": _v["score"], "test_usable": _v["test_usable"]}
except Exception as e:
    PROBE["verify_real_task"] = {"error": f"{type(e).__name__}: {e}"[:400]}
print("verify on a real task:", PROBE["verify_real_task"])
c = PROBE["cases"]
works = bool(PROBE["selftest"].get("ok")) and c["python thread [ok]"]["ok"] and c["identity [ok]"]["ok"] and c["float zeros [ok]"]["ok"] and c["scipy.ndimage [ok]"]["ok"]
contained = (all(v["as_expected"] for k, v in c.items() if "[blocked]" in k or "[timeout]" in k)
             and not PROBE["escape_file_exists"] and PROBE["root_file_intact"])
PROBE["threads_ok"] = c["python thread [ok]"]["ok"]; PROBE["cv2_parallel_ok"] = c["cv2 resize+blur 300x300 [ok]"]["ok"]
PROBE["verdict"] = "PASS" if works and contained else ("FAIL_NOT_WORKING" if not works else "FAIL_NOT_CONTAINED")
PROBE["wall_sec"] = round(time.time() - PROBE["t0"], 1)
json.dump(PROBE, open("/kaggle/working/sandbox_probe_r5.json", "w"), indent=1, default=str)
print("SANDBOX_R5_VERDICT:", PROBE["verdict"], "| works", works, "| contained", contained,
      "| threads", PROBE["threads_ok"], "| cv2 parallel", PROBE["cv2_parallel_ok"], "| wall", PROBE["wall_sec"], "s")
'''

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(True)}

nb = {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# E10 R5 sandbox probe (CPU)\n\nDoes R5's Kaggle runner (UID 65534 drop + no_new_privs + libseccomp + rlimits + timeout) "
            f"work inside the Kaggle container? R5 core cell embedded verbatim "
            f"(R5 sha256 `{R4_SHA[:16]}…`). No model, no GPU, no internet.\n"]},
        code(ENV), code("# R5 Poetiq core cell 5, verbatim\n" + core), code(CASES), code(VERDICT)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"probe{i}"
data = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
OUT.write_text(data, encoding="utf-8")
print("BUILT", OUT.name, len(data), "bytes sha256", hashlib.sha256(data.encode()).hexdigest()[:16])
