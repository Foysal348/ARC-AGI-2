"""E34 units CPU kit (pinned Py3.12 image, no GPU, no internet, private): e34_unit_tests.py on the BUILT FULL notebook's body /
export / facts with the real nbdf_lib / nb4t_lib / nb4_ppack / nbd1_lib (byte-identical to the NB-D1 base files), e34_lib, the
NB-S1 GPU fixture (e34_log_fixture.json) and the CPU planner results (e34_capacity_probe.json). The notebook itself is not
shipped: the facts / export checks read e34_code_export.py + e34_build_facts.json; the only allowed SKIP is `SKIP notebook`.
Prints one line `@@KIT units ALL OK@@` or `@@KIT units NOT OK: ...@@` and writes e34_kit_units_report.json.
(The layout follows build_e25_nbd1_units_kit.py; that file is not edited.)

Run: python -B build_e34_units_kit.py [--build-dir DIR] [--out-root DIR] [--sim DIR]
     -> <out-root, default Build/push>/q27-e34-kit-units__foysalemonshanto/  (not pushed)
     --sim DIR: also run the kit's code locally with KIT_WORK=DIR (the kit layout, an empty folder, python -I)."""
import base64
import hashlib
import json
import lzma
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E25B = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")
BDIR = Path(sys.argv[sys.argv.index("--build-dir") + 1]) if "--build-dir" in sys.argv else HERE
NBDIR = BDIR if "--build-dir" in sys.argv else HERE.parent / "Experiment Notebooks"
ROOT = Path(sys.argv[sys.argv.index("--out-root") + 1]) if "--out-root" in sys.argv else HERE / "push"
FACTS = json.loads((BDIR / "e34_build_facts.json").read_text(encoding="utf-8"))
NB = NBDIR / FACTS["notebook"]
NB_SHA = hashlib.sha256(NB.read_bytes()).hexdigest()
assert FACTS["sha256"] == NB_SHA, "build facts are not for this notebook: rebuild"
FILES = {"nbdf_lib.py": E27B / "nbdf_lib.py", "nb4t_lib.py": E25B / "nb4t_lib.py", "nb4_ppack.py": E25B / "nb4_ppack.py",
         "nbd1_lib.py": E25B / "nbd1_lib.py", "e34_lib.py": HERE / "e34_lib.py", "e34_unit_tests.py": HERE / "e34_unit_tests.py",
         "e34_cell7_body.py": BDIR / "e34_cell7_body.py", "e34_code_export.py": BDIR / "e34_code_export.py",
         "e34_build_facts.json": BDIR / "e34_build_facts.json", "e34_log_fixture.json": HERE / "e34_log_fixture.json",
         "e34_capacity_probe.json": HERE.parent / "Experiment Results" / "e34_capacity_probe.json"}
SRC = {k: p.read_bytes().decode("utf-8") for k, p in FILES.items()}
for k, v in SRC.items():
    assert "\r" not in v, f"CR in {k}"
assert hashlib.sha256(SRC["e34_code_export.py"].encode()).hexdigest() == FACTS["export_sha256"], "export != build facts"
assert hashlib.sha256(SRC["e34_lib.py"].encode()).hexdigest() == FACTS["e34_lib_sha256"], "e34_lib != build facts"
assert hashlib.sha256(SRC["nbd1_lib.py"].encode()).hexdigest() == FACTS["nbd1_lib_sha256"], "nbd1_lib != build facts"
assert hashlib.sha256(SRC["e34_cell7_body.py"].encode()).hexdigest() == FACTS["body_file_sha256"], "body != build facts"
assert hashlib.sha256(SRC["nbdf_lib.py"].encode()).hexdigest() == FACTS["nbdf_lib_sha256"], "nbdf_lib != build facts"
assert hashlib.sha256(SRC["nb4_ppack.py"].encode()).hexdigest() == FACTS["ppack_sha256"], "nb4_ppack != build facts"
SHAS = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in SRC.items()}
BLOB = base64.b85encode(lzma.compress(json.dumps(SRC).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
SLUG, OWNER = "q27-e34-kit-units", "foysalemonshanto"
IMAGE = FACTS.get("image")

CODE = r'''import base64, hashlib, json, lzma, os, re, subprocess, sys, time
from pathlib import Path
W = Path(os.environ.get("KIT_WORK") or "/kaggle/working")
K = W / "e34_units"
K.mkdir(parents=True, exist_ok=True)
FILES = json.loads(lzma.decompress(base64.b85decode("__BLOB__")).decode("utf-8"))
SHAS = json.loads(r"""__SHAS__""")
for n, t in FILES.items():
    assert hashlib.sha256(t.encode()).hexdigest() == SHAS[n], n
    (K / n).parent.mkdir(parents=True, exist_ok=True)
    (K / n).write_text(t, encoding="utf-8", newline="\n")
env = {k: v for k, v in os.environ.items() if not k.startswith(("NB1_", "E23_", "RC2_", "NB4_", "Q27X_", "DFLASH_", "NBS2_", "NBD1_", "E34_"))}
t0 = time.time()
r = subprocess.run([sys.executable, str(K / "e34_unit_tests.py"), "--libs", str(K), "--body", str(K / "e34_cell7_body.py"),
                    "--facts", str(K / "e34_build_facts.json"), "--export", str(K / "e34_code_export.py"),
                    "--fixture", str(K / "e34_log_fixture.json"), "--capacity", str(K / "e34_capacity_probe.json"),
                    "--notebook", str(K / "absent.ipynb")], capture_output=True, text=True, cwd=str(K), timeout=1800, env=env)
lines = r.stdout.strip().splitlines()
last = (lines or [""])[-1]
fails = [l for l in lines if l.startswith("FAIL")]
skips = [l for l in lines if l.startswith("SKIP")]
m = re.match(r"(\d+)/(\d+) PASS$", last)
rep = {"what": "E34 units kit", "notebook_sha256": "__NBSHA__", "python": sys.version, "rc": r.returncode, "result": last,
       "fails": fails[:40], "skips": skips, "stderr_tail": r.stderr[-2000:], "sec": round(time.time() - t0, 1), "file_shas": SHAS}
ok_ = (r.returncode == 0 and bool(m) and m.group(1) == m.group(2) and not fails and all(l.startswith("SKIP notebook") for l in skips)
       and (sys.version_info[:2] == (3, 12) or os.environ.get("KIT_LOCAL_SIM") == "1"))
rep["ALL_OK"] = ok_
(W / "e34_kit_units_report.json").write_text(json.dumps(rep, indent=1))
print(r.stdout[-4000:])
print("@@KIT units ALL OK@@" if ok_ else f"@@KIT units NOT OK: {last} {fails[:3]} skips {skips} py {sys.version_info[:2]}@@", flush=True)
'''
code = CODE.replace("__BLOB__", BLOB).replace("__SHAS__", json.dumps(SHAS)).replace("__NBSHA__", NB_SHA)
compile(code, "<e34 units kit>", "exec")
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
data = json.dumps(nb, separators=(",", ":")).encode("utf-8")
assert len(data) < 400_000, len(data)
assert b"\r" not in data.replace(b"\\r", b"")
out = ROOT / f"{SLUG}__{OWNER}"
out.mkdir(parents=True, exist_ok=True)
for f in out.glob("*"):
    if f.is_file():
        f.unlink()
(out / f"{SLUG}.ipynb").write_bytes(data)
meta = {"id": f"{OWNER}/{SLUG}", "title": "Q27 E34 Kit Units", "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [],
        "docker_image": IMAGE}
(out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
print(f"kit {out} | {len(data)} bytes | sha256 {hashlib.sha256(data).hexdigest()} | E34 {NB_SHA[:8]} | files {len(SRC)}")

if "--sim" in sys.argv:                      # the kit layout, locally: the kit's own code in an empty folder (no repo paths)
    sim = Path(sys.argv[sys.argv.index("--sim") + 1])
    sim.mkdir(parents=True, exist_ok=True)
    assert not any(sim.iterdir()), f"--sim needs an empty folder: {sim}"
    (sim / f"{SLUG}.py").write_text(code, encoding="utf-8", newline="\n")
    r = subprocess.run([sys.executable, "-I", str(sim / f"{SLUG}.py")], capture_output=True, text=True, cwd=str(sim), timeout=1800,
                       env=dict(os.environ, KIT_WORK=str(sim / "working"), KIT_LOCAL_SIM="1"))
    print(r.stdout[-1500:], r.stderr[-1500:])
    sys.exit(0 if "@@KIT units ALL OK@@" in r.stdout else 1)
