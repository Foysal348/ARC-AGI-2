"""E32 Kaggle CPU kit (units, 2026-11-02): ONE CPU kernel on foysalemonshanto, the notebooks' pinned Py3.12 image, no GPU, no
internet, private. It runs e32_kit_tests.py (= e32_checks.run: P / A / O / D / R / S / K) on the mounted files, found by name +
sha256 exactly as the notebook finds its payload:
  - the E32 payload dataset foysalemonshanto/q27-e32-levers-payload: e32_lib.py + e32_nbx.py (the notebook's pinned shas) and the
    two notebook fixtures kit_q27-xhigh-e32-levers.ipynb (= this build) / kit_q27-xhigh-e29-dflash-d3s.ipynb (= E29 production)
  - nb5_lib.py from the tf-ref dataset's k100q_payload_emb.json (sha fd92a814, the notebook's own K100_EMB source, sha 18efcbc1)
  - e29_lib / e28k100_lib / e29_nbdf from the E32 notebook's own CELL 1 modules literal
The kernel is fail-soft, writes q27x_e32_kit_units_report.json and prints exactly one line `@@KIT e32-units ALL OK@@` or
`@@KIT e32-units NOT OK: <items>@@`. NOTHING is pushed here (the user pushes kits).

  python build_e32_cpu_kit.py  ->  D:/ARC AGI 2/_scratch/push_e25/q27-e32-kit-units__foysalemonshanto/
"""
import base64
import hashlib
import json
import lzma
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
OWNER = "foysalemonshanto"
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
SLUG = "q27-e32-kit-units"
DOCKER = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"   # = E29 kits
TFREF = ("foysalemonshanto/q27-e25-nb0-tf-ref", "k100q_payload_emb.json", "fd92a814040aea6dda12855badf801675580dcd78b2f2135eb0b86121dd80880")
NB5LIB_SHA = "18efcbc1dde9c590a14ac1937b45dec563ff76ed4d92b16632d273dec2c88500"
CHECKS_MIN = 18


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def lf(p):
    b = Path(p).read_bytes()
    assert b"\r" not in b, p
    return b.decode("utf-8")


HEAD = r'''# E32 CPU kit units (pinned image, no GPU): see Build/build_e32_cpu_kit.py for the checks and the verdict rule
import os, sys, json, time, subprocess, shutil, traceback, hashlib, re, base64, lzma
from pathlib import Path
T0 = time.time()
INPUT_ROOT = Path(os.environ.get("KIT_INPUT", "/kaggle/input"))          # env overrides: a local dry run only
K = Path(os.environ.get("KIT_TMP", "/tmp/e32kit")); shutil.rmtree(K, ignore_errors=True); K.mkdir(parents=True)
W = Path(os.environ.get("KIT_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
KF = json.loads(r"""__KF__""")
REP = {"what": "E32 CPU kit units", "notebook_sha256": KF["e32_nb"][1], "python": sys.version, "checks": {}, "kf": KF}
def save():
    (W / "q27x_e32_kit_units_report.json").write_text(json.dumps(REP, indent=1, default=str))
def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
def find(name, want):
    """= the notebook's q27x_payload search: by exact name under the input root (<= 3 directory levels), verified by sha256."""
    seen = []
    for pat in (name, "*/" + name, "*/*/" + name, "*/*/*/" + name):
        for f in sorted(INPUT_ROOT.glob(pat)):
            if hashlib.sha256(f.read_bytes()).hexdigest() == want:
                return f, seen
            seen.append(str(f))
    return None, seen
FILES = json.loads(lzma.decompress(base64.b85decode("__BLOB__")).decode("utf-8"))
assert {n: hashlib.sha256(v.encode("utf-8")).hexdigest() for n, v in FILES.items()} == json.loads(r"""__SHAS__""")
B = K / "build"; B.mkdir()
for n, v in FILES.items():
    (B / n).write_text(v, encoding="utf-8", newline="\n")
V, U = {}, {}
try:
    for key in ("e32_lib", "e32_nbx", "e32_nb", "e29_nb"):
        f, seen = find(*KF[key])
        U[key] = {"path": str(f) if f else None, "seen": seen}
        if f:
            shutil.copyfile(f, B / KF[key][0])
    f, seen = find(KF["tfref"][1], KF["tfref"][2])
    U["tfref"] = {"path": str(f) if f else None, "seen": seen}
    if f:
        emb = json.loads(f.read_text(encoding="utf-8"))
        nb5 = lzma.decompress(base64.b85decode(emb["nb5_lib.py"][1])).decode("utf-8")
        U["nb5_lib_sha256"] = hashlib.sha256(nb5.encode()).hexdigest()
        (B / "nb5_lib.py").write_text(nb5, encoding="utf-8", newline="\n")
except BaseException as e:
    U["find_error"] = f"{type(e).__name__}: {e}"[:600]
REP["checks"]["inputs"] = U; save()
log("inputs", json.dumps(U, default=str)[:2000])
V["inputs_found_by_name_and_sha"] = all((U.get(k) or {}).get("path") for k in ("e32_lib", "e32_nbx", "e32_nb", "e29_nb", "tfref"))
V["nb5_lib_from_tfref_payload_sha"] = U.get("nb5_lib_sha256") == KF["nb5_lib_sha"]
T = {}
if V["inputs_found_by_name_and_sha"] and V["nb5_lib_from_tfref_payload_sha"]:
    try:
        env = dict(os.environ, E32KT_NB=str(B / KF["e32_nb"][0]), E32KT_E29NB=str(B / KF["e29_nb"][0]), E32KT_LIB=str(B / "e32_lib.py"),
                   E32KT_NBX=str(B / "e32_nbx.py"), E32KT_NB5LIB=str(B / "nb5_lib.py"), E32KT_TMP=str(K / "mods"), PYTHONIOENCODING="utf-8")
        t0 = time.time()
        r = subprocess.run([sys.executable, "-B", str(B / "e32_kit_tests.py")], capture_output=True, text=True, env=env, cwd=str(B), timeout=3000)
        out = r.stdout + r.stderr
        m = re.search(r"^(\d+)/(\d+) PASS$", out, re.M)
        T.update(rc=r.returncode, sec=round(time.time() - t0, 1), summary=m.group(0) if m else None, tail=out[-6000:],
                 fails=[l[:600] for l in out.splitlines() if l.startswith("FAIL ")][:40], python=out.splitlines()[0][:200] if out else None)
        T["all_pass"] = bool(m) and r.returncode == 0 and m.group(1) == m.group(2) and int(m.group(2)) >= KF["checks_min"]
        print(out, flush=True)
    except BaseException as e:
        T.update(all_pass=False, fatal=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
REP["checks"]["e32_kit_tests"] = T
V["e32_kit_tests_all_pass"] = T.get("all_pass") is True
REP["verdict"] = {"checks": V, "ALL_OK": all(v is True for v in V.values())}
save()
bad = [k for k, v in V.items() if v is not True]
print("@@KIT e32-units " + ("ALL OK" if not bad else "NOT OK: " + ", ".join(bad)) + "@@", flush=True)
'''


def build():
    facts = json.loads(lf(HERE / "e32_build_facts.json"))
    ds = facts["dataset"]["files"]
    fx = {"e32_nb": "kit_q27-xhigh-e32-levers.ipynb", "e29_nb": "kit_q27-xhigh-e29-dflash-d3s.ipynb"}
    dsdir = Path(facts["dataset"]["dir"])
    assert sha((dsdir / fx["e32_nb"]).read_bytes()) == facts["sha256"], "the payload's E32 notebook fixture != the build"
    assert sha((dsdir / fx["e29_nb"]).read_bytes()) == facts["base"]["e29_sha256"], "the payload's E29 notebook fixture != E29 production"
    files = {n: lf(HERE / n) for n in ("e32_kit_tests.py", "e32_checks.py")}
    kf = {"e32_lib": ["e32_lib.py", ds["e32_lib.py"]["sha256"]], "e32_nbx": ["e32_nbx.py", ds["e32_nbx.py"]["sha256"]],
          "e32_nb": [fx["e32_nb"], facts["sha256"]], "e29_nb": [fx["e29_nb"], facts["base"]["e29_sha256"]],
          "tfref": list(TFREF), "nb5_lib_sha": NB5LIB_SHA, "checks_min": CHECKS_MIN, "build_facts_sha256": sha(lf(HERE / "e32_build_facts.json"))}
    assert kf["e32_lib"] == facts["e32_files"]["lib"] and kf["e32_nbx"] == facts["e32_files"]["nbx"]
    blob = base64.b85encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
    code = (HEAD.replace("__KF__", json.dumps(kf)).replace("__BLOB__", blob)
            .replace("__SHAS__", json.dumps({n: sha(v) for n, v in files.items()})))
    for ph in ("__KF__", "__BLOB__", "__SHAS__"):
        assert ph not in code
    compile(code, "<kit e32-units>", "exec")
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1).encode("utf-8")
    assert b"\r" not in data.replace(b"\\r", b"") and len(data) < 400_000
    out = ROOT / f"{SLUG}__{OWNER}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{SLUG}.ipynb").write_bytes(data)
    meta = {"id": f"{OWNER}/{SLUG}", "title": "Q27 E32 Kit Units", "code_file": f"{SLUG}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": [facts["dataset"]["id"], TFREF[0]], "kernel_sources": [],
            "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": [], "docker_image": DOCKER, "machine_shape": "None"}
    assert meta["docker_image"] == facts["push"]["prod"]["kernel_metadata"]["docker_image"]
    (out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    rec = {"folder": str(out), "sha256": sha(data), "bytes": len(data), "kf": kf, "files": {n: sha(v) for n, v in files.items()}}
    (HERE / "e32_kit_facts.json").write_bytes((json.dumps(rec, indent=1) + "\n").encode("utf-8"))
    print("kit", out.name, "|", len(data), "bytes | sha256", sha(data))
    return rec


if __name__ == "__main__":
    build()
