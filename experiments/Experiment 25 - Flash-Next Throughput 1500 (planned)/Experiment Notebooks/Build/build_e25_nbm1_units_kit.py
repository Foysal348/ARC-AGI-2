"""E25 NB-M1 units CPU kit (pinned Py3.12 image, no GPU, no internet, private): nbm1_unit_tests.py on the BUILT notebook's
body / export / facts with the real nbdf_lib / nb4t_lib / nb4_ppack / nb4_timeline (byte-identical to the frozen NBDF-K100
files). Prints one line `@@KIT units ALL OK@@` or `@@KIT units NOT OK: ...@@` and writes nbm1_kit_units_report.json.
Run: python build_e25_nbm1_units_kit.py [--local DIR]
   -> D:/ARC AGI 2/_scratch/push_e25/q27-e25-nbm1-kit-units__foysalemonshanto/
   --local DIR: also run the kit's own code in DIR (an empty folder, KIT_WORK=DIR) with this Python (the layout simulation).
   --light: the E25 NB-M1L (light) units kit -> q27-e25-nbm1l-kit-units__foysalemonshanto/ (nbm1l_unit_tests.py on the light
            notebook's body / export / facts + nbm1_lib + nbm1l_overlay; report nbm1l_kit_units_report.json)
   --out-root DIR: write the kit folder under DIR instead of D:/ARC AGI 2/_scratch/push_e25 (reproduction checks)."""
import base64
import hashlib
import json
import lzma
import os
import subprocess
import sys
from pathlib import Path

LIGHT = "--light" in sys.argv
PFX = "nbm1l" if LIGHT else "nbm1"
HERE = Path(__file__).resolve().parent
E27B = Path(r"D:/ARC AGI 2/experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build")
NB = HERE.parent / ("Q27-XHIGH-E25_NBM1L.ipynb" if LIGHT else "Q27-XHIGH-E25_NBM1.ipynb")
FACTS = json.loads((HERE / f"{PFX}_build_facts.json").read_text(encoding="utf-8"))
NB_SHA = hashlib.sha256(NB.read_bytes()).hexdigest()
assert FACTS["sha256"] == NB_SHA, "build facts are not for this notebook: rebuild"
FILES = {"nbdf_lib.py": E27B / "nbdf_lib.py", "nb4t_lib.py": HERE / "nb4t_lib.py", "nb4_ppack.py": HERE / "nb4_ppack.py",
         "nb4_timeline.py": HERE / "nb4_timeline.py", "nbm1_lib.py": HERE / "nbm1_lib.py", "nbm1_unit_tests.py": HERE / "nbm1_unit_tests.py",
         "nbm1_cell7_body.py": HERE / "nbm1_cell7_body.py", "nbm1_code_export.py": HERE / "nbm1_code_export.py",
         "nbm1_build_facts.json": HERE / "nbm1_build_facts.json"}
if LIGHT:
    FILES = {"nbdf_lib.py": E27B / "nbdf_lib.py", "nb4t_lib.py": HERE / "nb4t_lib.py", "nb4_ppack.py": HERE / "nb4_ppack.py",
             "nb4_timeline.py": HERE / "nb4_timeline.py", "nbm1_lib.py": HERE / "nbm1_lib.py", "nbm1l_overlay.py": HERE / "nbm1l_overlay.py",
             "nbm1l_unit_tests.py": HERE / "nbm1l_unit_tests.py", "nbm1l_cell7_body.py": HERE / "nbm1l_cell7_body.py",
             "nbm1l_code_export.py": HERE / "nbm1l_code_export.py", "nbm1l_build_facts.json": HERE / "nbm1l_build_facts.json"}
SRC = {k: p.read_bytes().decode("utf-8") for k, p in FILES.items()}
for k, v in SRC.items():
    assert "\r" not in v, f"CR in {k}"
if LIGHT:
    assert hashlib.sha256(SRC["nbm1l_overlay.py"].encode()).hexdigest() == FACTS["nbm1"]["overlay_sha256"], "nbm1l_overlay != build facts"
EXP_K, BODY_K, CELLS_K = f"{PFX}_code_export.py", f"{PFX}_cell7_body.py", f"{PFX}_export_cells.json"
assert hashlib.sha256(SRC[EXP_K].encode()).hexdigest() == FACTS["export_sha256"], "export != build facts"
# size (< 400 kB): the export's CELL 7 stub (the lzma + b85 module / body blobs, ~190 kB) is omitted; the body ships as
# nbm1_cell7_body.py (sha = the facts' body_file_sha256 = what the stub execs). The other cells go as a list (sha-checked
# against the facts' per-cell shas by the tests); the export itself = these cells joined.
_cells = ["".join(c["source"]) for c in json.loads(NB.read_bytes().decode("utf-8"))["cells"] if c["cell_type"] == "code"]
assert "\n\n".join(_cells) == SRC[EXP_K] and [hashlib.sha256(c.encode()).hexdigest() for c in _cells] == FACTS["nbdf_code_cell_sha256"]
_stub = [i for i, c in enumerate(_cells) if "NBDF_BODY_SRC = _unx(" in c]
assert len(_stub) == 1
SRC[CELLS_K] = json.dumps([None if i == _stub[0] else c for i, c in enumerate(_cells)])
del SRC[EXP_K]
assert hashlib.sha256(SRC["nbm1_lib.py"].encode()).hexdigest() == FACTS["nbm1"]["lib_sha256"], "nbm1_lib != build facts"
assert hashlib.sha256(SRC[BODY_K].encode()).hexdigest() == FACTS["body_file_sha256"], "body != build facts"
assert hashlib.sha256(SRC["nbdf_lib.py"].encode()).hexdigest() == FACTS["lib_sha256"], "nbdf_lib != the notebook's embedded one"
SHAS = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in SRC.items()}
BLOB = base64.b85encode(lzma.compress(json.dumps(SRC).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
SLUG, OWNER = f"q27-e25-{PFX}-kit-units", "foysalemonshanto"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

CODE = r'''import base64, hashlib, json, lzma, os, subprocess, sys, time
from pathlib import Path
W = Path(os.environ.get("KIT_WORK") or "/kaggle/working")
K = W / "nbm1_units"
K.mkdir(parents=True, exist_ok=True)
FILES = json.loads(lzma.decompress(base64.b85decode("__BLOB__")).decode("utf-8"))
SHAS = json.loads(r"""__SHAS__""")
for n, t in FILES.items():
    assert hashlib.sha256(t.encode()).hexdigest() == SHAS[n], n
    (K / n).write_bytes(t.encode("utf-8"))
t0 = time.time()
env = {k: v for k, v in os.environ.items() if not k.startswith(("NB1_", "E23_", "RC2_", "NB4_", "Q27X_", "DFLASH_", "VLLM_", "NBM1_"))}
r = subprocess.run([sys.executable, "-B", str(K / "nbm1_unit_tests.py")], capture_output=True, text=True, cwd=str(K), timeout=1800, env=env)
last = (r.stdout.strip().splitlines() or [""])[-1]
fails = [l for l in r.stdout.splitlines() if l.startswith("FAIL")]
skips = [l for l in r.stdout.splitlines() if l.startswith("SKIP")]
rep = {"what": "E25 NB-M1 units kit", "notebook_sha256": "__NBSHA__", "python": sys.version, "rc": r.returncode, "result": last,
       "fails": fails[:40], "skips": skips, "stderr_tail": r.stderr[-2000:], "sec": round(time.time() - t0, 1), "file_shas": SHAS}
ok_ = r.returncode == 0 and last.endswith("PASS") and not fails and not skips and sys.version_info[:2] == (3, 12)
rep["ALL_OK"] = ok_
(W / "nbm1_kit_units_report.json").write_text(json.dumps(rep, indent=1))
print(r.stdout[-4000:])
print("@@KIT units ALL OK@@" if ok_ else f"@@KIT units NOT OK: {last} {fails[:3]} {skips[:2]} py {sys.version_info[:2]}@@", flush=True)
'''
if LIGHT:                                     # NB-M1L: its own work folder, tests, report; NBM1L_* env never reaches the tests
    for a_, b_ in (('K = W / "nbm1_units"', 'K = W / "nbm1l_units"'), ('str(K / "nbm1_unit_tests.py")', 'str(K / "nbm1l_unit_tests.py")'),
                   ('"Q27X_", "DFLASH_", "VLLM_", "NBM1_"))', '"Q27X_", "DFLASH_", "VLLM_", "NBM1_", "NBM1L_"))'),
                   ('"E25 NB-M1 units kit"', '"E25 NB-M1L units kit"'), ('(W / "nbm1_kit_units_report.json")', '(W / "nbm1l_kit_units_report.json")')):
        assert CODE.count(a_) == 1, a_
        CODE = CODE.replace(a_, b_)
code = CODE.replace("__BLOB__", BLOB).replace("__SHAS__", json.dumps(SHAS)).replace("__NBSHA__", NB_SHA)
compile(code, "<nbm1 units kit>", "exec")
nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
data = json.dumps(nb, separators=(",", ":")).encode("utf-8")
assert len(data) < 400_000, len(data)
out = (Path(sys.argv[sys.argv.index("--out-root") + 1]) if "--out-root" in sys.argv else Path(r"D:/ARC AGI 2/_scratch/push_e25")) / f"{SLUG}__{OWNER}"
out.mkdir(parents=True, exist_ok=True)
(out / f"{SLUG}.ipynb").write_bytes(data)
meta = {"id": f"{OWNER}/{SLUG}", "title": "Q27 E25 NBM1L Kit Units" if LIGHT else "Q27 E25 NBM1 Kit Units", "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
        "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [],
        "docker_image": IMAGE}
(out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
print(f"kit {out} | {len(data)} bytes | sha256 {hashlib.sha256(data).hexdigest()} | {PFX.upper()} {NB_SHA[:8]}")
if "--local" in sys.argv:                      # the kit layout simulated: its own code in an empty folder
    d = Path(sys.argv[sys.argv.index("--local") + 1])
    d.mkdir(parents=True, exist_ok=True)
    assert not any(d.iterdir()), f"--local {d} must be an empty folder"
    (d.parent / f"{d.name}_kit_code.py").write_bytes(code.encode("utf-8"))
    r = subprocess.run([sys.executable, "-I", str(d.parent / f"{d.name}_kit_code.py")], capture_output=True, text=True,
                       env=dict(os.environ, KIT_WORK=str(d)), cwd=str(d), timeout=1800)
    print("LOCAL", r.returncode, (r.stdout.strip().splitlines() or [""])[-1], r.stderr[-1500:])
