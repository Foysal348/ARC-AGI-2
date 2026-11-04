"""E25 NB-D1 units CPU kit (pinned Py3.12 image, no GPU, no internet, private): nbd1_unit_tests.py on the BUILT notebook's body /
export / facts with the real nbdf_lib / nb4t_lib / nb4_ppack / nb4_timeline (byte-identical to the NBDF-K100 base files),
the NB-S1 GPU log fixture and Merlin's own vllm/v1/spec_decode/dynamic/utils.py. The notebook itself is not shipped: the
facts / export / switch-line checks read nbd1_code_export.py + nbd1_build_facts.json; the only allowed SKIP is `SKIP notebook`.
Prints one line `@@KIT units ALL OK@@` or `@@KIT units NOT OK: ...@@` and writes nbd1_kit_units_report.json.
(The layout follows build_e25_nbs2_units_kit.py; that file is not edited.)

Run: python build_e25_nbd1_units_kit.py [--build-dir DIR] [--out-root DIR] [--sim DIR]
     -> <out-root, default D:/ARC AGI 2/_scratch/push_e25>/q27-e25-nbd1-kit-units__foysalemonshanto/
     --build-dir DIR: the build outputs (nbd1_* + the notebook) from build_e25_nbd1.py --test-out DIR (default: this folder)
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
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")
MERLIN_SRC = Path(r"D:/ARC AGI 2/_scratch/e23/p2/merlin-src")
BDIR = Path(sys.argv[sys.argv.index("--build-dir") + 1]) if "--build-dir" in sys.argv else HERE
NBDIR = BDIR if "--build-dir" in sys.argv else HERE.parent
ROOT = Path(sys.argv[sys.argv.index("--out-root") + 1]) if "--out-root" in sys.argv else Path(r"D:/ARC AGI 2/_scratch/push_e25")
NB = NBDIR / "Q27-XHIGH-E25_NBD1.ipynb"
FACTS = json.loads((BDIR / "nbd1_build_facts.json").read_text(encoding="utf-8"))
NB_SHA = hashlib.sha256(NB.read_bytes()).hexdigest()
assert FACTS["sha256"] == NB_SHA, "build facts are not for this notebook: rebuild"
FILES = {"nbdf_lib.py": E27B / "nbdf_lib.py", "nb4t_lib.py": HERE / "nb4t_lib.py", "nb4_ppack.py": HERE / "nb4_ppack.py",
         "nb4_timeline.py": HERE / "nb4_timeline.py", "nbd1_lib.py": HERE / "nbd1_lib.py", "nbd1_unit_tests.py": HERE / "nbd1_unit_tests.py",
         "nbd1_cell7_body.py": BDIR / "nbd1_cell7_body.py", "nbd1_code_export.py": BDIR / "nbd1_code_export.py",
         "nbd1_build_facts.json": BDIR / "nbd1_build_facts.json", "nbd1_log_fixture.json": HERE / "nbd1_log_fixture.json",
         "merlin/vllm/v1/spec_decode/dynamic/utils.py": MERLIN_SRC / "vllm/v1/spec_decode/dynamic/utils.py"}
SRC = {k: p.read_bytes().decode("utf-8") for k, p in FILES.items()}
for k, v in SRC.items():
    assert "\r" not in v, f"CR in {k}"
assert hashlib.sha256(SRC["nbd1_code_export.py"].encode()).hexdigest() == FACTS["export_sha256"], "export != build facts"
assert hashlib.sha256(SRC["nbd1_lib.py"].encode()).hexdigest() == FACTS["nbd1_lib_sha256"], "nbd1_lib != build facts"
assert hashlib.sha256(SRC["nbd1_cell7_body.py"].encode()).hexdigest() == FACTS["body_file_sha256"], "body != build facts"
assert hashlib.sha256(SRC["nbdf_lib.py"].encode()).hexdigest() == FACTS["nbdf_lib_sha256"], "nbdf_lib != build facts"
assert hashlib.sha256(SRC["nb4_ppack.py"].encode()).hexdigest() == FACTS["ppack_sha256"], "nb4_ppack != build facts"
SHAS = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in SRC.items()}
BLOB = base64.b85encode(lzma.compress(json.dumps(SRC).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
SLUG, OWNER = "q27-e25-nbd1-kit-units", "foysalemonshanto"
IMAGE = FACTS.get("image") or "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

CODE = r'''import base64, hashlib, json, lzma, os, re, subprocess, sys, time
from pathlib import Path
W = Path(os.environ.get("KIT_WORK") or "/kaggle/working")
K = W / "nbd1_units"
K.mkdir(parents=True, exist_ok=True)
FILES = json.loads(lzma.decompress(base64.b85decode("__BLOB__")).decode("utf-8"))
SHAS = json.loads(r"""__SHAS__""")
for n, t in FILES.items():
    assert hashlib.sha256(t.encode()).hexdigest() == SHAS[n], n
    (K / n).parent.mkdir(parents=True, exist_ok=True)
    (K / n).write_text(t, encoding="utf-8", newline="\n")
env = {k: v for k, v in os.environ.items() if not k.startswith(("NB1_", "E23_", "RC2_", "NB4_", "Q27X_", "DFLASH_", "NBS2_", "NBD1_"))}
t0 = time.time()
r = subprocess.run([sys.executable, str(K / "nbd1_unit_tests.py"), "--libs", str(K), "--merlin-src", str(K / "merlin"),
                    "--body", str(K / "nbd1_cell7_body.py"), "--facts", str(K / "nbd1_build_facts.json"),
                    "--export", str(K / "nbd1_code_export.py"), "--fixture", str(K / "nbd1_log_fixture.json"),
                    "--notebook", str(K / "absent.ipynb")], capture_output=True, text=True, cwd=str(K), timeout=1800, env=env)
lines = r.stdout.strip().splitlines()
last = (lines or [""])[-1]
fails = [l for l in lines if l.startswith("FAIL")]
skips = [l for l in lines if l.startswith("SKIP")]
m = re.match(r"(\d+)/(\d+) PASS$", last)
rep = {"what": "E25 NB-D1 units kit", "notebook_sha256": "__NBSHA__", "python": sys.version, "rc": r.returncode, "result": last,
       "fails": fails[:40], "skips": skips, "stderr_tail": r.stderr[-2000:], "sec": round(time.time() - t0, 1), "file_shas": SHAS}
ok_ = (r.returncode == 0 and bool(m) and m.group(1) == m.group(2) and not fails and all(l.startswith("SKIP notebook") for l in skips)
       and (sys.version_info[:2] == (3, 12) or os.environ.get("KIT_LOCAL_SIM") == "1"))
rep["ALL_OK"] = ok_
(W / "nbd1_kit_units_report.json").write_text(json.dumps(rep, indent=1))
print(r.stdout[-4000:])
print("@@KIT units ALL OK@@" if ok_ else f"@@KIT units NOT OK: {last} {fails[:3]} skips {skips} py {sys.version_info[:2]}@@", flush=True)
'''
code = CODE.replace("__BLOB__", BLOB).replace("__SHAS__", json.dumps(SHAS)).replace("__NBSHA__", NB_SHA)
compile(code, "<nbd1 units kit>", "exec")
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
meta = {"id": f"{OWNER}/{SLUG}", "title": "Q27 E25 NBD1 Kit Units", "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [],
        "docker_image": IMAGE}
(out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
print(f"kit {out} | {len(data)} bytes | sha256 {hashlib.sha256(data).hexdigest()} | NB-D1 {NB_SHA[:8]} | files {len(SRC)}")

if "--sim" in sys.argv:                      # the kit layout, locally: the kit's own code in an empty folder (no repo paths)
    sim = Path(sys.argv[sys.argv.index("--sim") + 1])
    sim.mkdir(parents=True, exist_ok=True)
    assert not any(sim.iterdir()), f"--sim needs an empty folder: {sim}"
    (sim / f"{SLUG}.py").write_text(code, encoding="utf-8", newline="\n")
    r = subprocess.run([sys.executable, "-I", str(sim / f"{SLUG}.py")], capture_output=True, text=True, cwd=str(sim), timeout=1800,
                       env=dict(os.environ, KIT_WORK=str(sim / "working"), KIT_LOCAL_SIM="1"))
    print(r.stdout[-1500:], r.stderr[-1500:])
    sys.exit(0 if "@@KIT units ALL OK@@" in r.stdout else 1)
