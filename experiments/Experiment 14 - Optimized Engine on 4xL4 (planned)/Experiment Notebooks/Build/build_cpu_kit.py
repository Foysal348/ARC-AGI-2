"""E14 CPU kit: the unit tests + the fake-server notebook harness, run on a Kaggle CPU kernel with the SAME pinned image
as the GPU notebook (Python 3.12.13, the same libraries). User 2026-10-23: "CPU run on Kaggle and this problem does not
happen". It does: E14A SLIM run1 crashed at A1-2 on a Python 3.12 Thread._stop clash that the local Python 3.14 harness
could not show (Notes/21).

Writes:
- Build/cpu_kit/q27x_e14_cpu_kit_<sha12>.bin  (a zip: src/, fixtures/, sgl_ref/, nb/ incl. Build/payload, wheels/,
  MANIFEST.json) + dataset-metadata.json  -> the private dataset foysalemonshanto/q27-e14-cpu-kit
- Experiment Notebooks/Q27-E14-CPU-HARNESS.ipynb  (small: finds the kit by name + sha256, installs the same runtime pins
  as the GPU notebook, runs pytest and the harness in both scenarios / modes, writes /kaggle/working/cpu_harness_report.json)
    python build_cpu_kit.py <notebook name under Experiment Notebooks/> [scale ...]
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
E14 = HERE.parents[1]
SRC = E14 / "src"
NBDIR = HERE.parent
FIX = Path(r"D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures")
SGL = Path(r"D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang")
WHEELS = Path(r"D:/ARC AGI 2/_scratch/cpu_kit_wheels")
KIT_SLUG = "foysalemonshanto/q27-e14-cpu-kit"
OUT_DIR = HERE / "cpu_kit"
TARGET = sys.argv[1] if len(sys.argv) > 1 else "Q27-XHIGH-E14A_SLIM_R2_OPTIMIZED_ENGINE_SGLANG.ipynb"
SCALES = [float(x) for x in sys.argv[2:]] or [0.1]
# E14S (the submission notebook) runs its own harness in commit and rerun mode; E14A the measurement harness
RUNS = ([("harness_e14s_nb.py", "E14S_MODE", m) for m in ("e15", "rerun")] if "E15" in TARGET else
        [("harness_e14s_nb.py", "E14S_MODE", m) for m in ("commit", "rerun")] if "E14S" in TARGET else
        [("harness_e14_nb.py", "E14_HARNESS_SCENARIO", m) for m in ("default", "corrupt")])
SGL_FILES = ["_version.py", "srt/dllm/mixin/scheduler.py", "srt/managers/scheduler.py",
             "srt/managers/min_free_slots_delayer.py"]
TESTS = ["test_e14_measure.py", "test_e14_loadpath.py", "test_e14_sglang_patches.py", "test_e14_nccl_bench_and_trace.py"]

files = {}
for p in sorted(SRC.glob("e14_*.py")) + [SRC / t for t in TESTS]:
    files["src/" + p.name] = p.read_bytes()
for p in sorted((SRC / "harness").glob("*.py")):
    files["src/harness/" + p.name] = p.read_bytes()
for p in sorted(FIX.rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts:
        files["fixtures/" + p.relative_to(FIX).as_posix()] = p.read_bytes()
for rel in SGL_FILES:
    files["sgl_ref/" + rel] = (SGL / rel).read_bytes()
nb_path = NBDIR / TARGET
assert nb_path.is_file(), nb_path
files["nb/" + TARGET] = nb_path.read_bytes()
for p in sorted((HERE / "payload").glob("q27x_*.bin")):
    files["nb/Build/payload/" + p.name] = p.read_bytes()
for p in sorted(WHEELS.glob("*.whl")):
    files["wheels/" + p.name] = p.read_bytes()
assert any(k.startswith("wheels/pytest-") for k in files), "pytest wheel missing"
manifest = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())}
files["MANIFEST.json"] = json.dumps(manifest, indent=0).encode()

buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for k in sorted(files):
        zi = zipfile.ZipInfo(k, date_time=(2026, 10, 23, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, files[k])
kit = buf.getvalue()
KIT_SHA = hashlib.sha256(kit).hexdigest()
KIT_NAME = f"q27x_e14_cpu_kit_{KIT_SHA[:12]}.bin"       # .bin: kept out of Kaggle's archive unpacking
OUT_DIR.mkdir(exist_ok=True)
(OUT_DIR / KIT_NAME).write_bytes(kit)
(OUT_DIR / "dataset-metadata.json").write_text(json.dumps(
    {"title": "Q27 E14 CPU kit", "id": KIT_SLUG, "licenses": [{"name": "apache-2.0"}]}, indent=1), encoding="utf-8")

CODE = r'''# E14 CPU harness on Kaggle (pinned image = the GPU notebook's): unit tests + the fake-server notebook harness
import hashlib, io, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
KIT_NAME, KIT_SHA = "''' + KIT_NAME + '''", "''' + KIT_SHA + '''"
TARGET = "nb/''' + TARGET + '''"
SCALES = ''' + json.dumps(SCALES) + r'''
TESTS = ''' + json.dumps(TESTS) + r'''
RUNS = ''' + json.dumps(RUNS) + r'''
WH = "/kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"
OUT = Path("/kaggle/working"); K = Path("/tmp/e14_kit"); T0 = time.time()
REPORT = {"kit": KIT_NAME, "target": TARGET, "python": sys.version, "scales": SCALES, "pytest": None, "harness": {}}


def save():
    (OUT / "cpu_harness_report.json").write_text(json.dumps(REPORT, indent=1), encoding="utf-8")


def find(name, sha):
    for pat in (name, "*/" + name, "*/*/" + name, "*/*/*/" + name):
        for f in sorted(Path("/kaggle/input").glob(pat)):
            b = f.read_bytes()
            if hashlib.sha256(b).hexdigest() == sha:
                return b
    raise FileNotFoundError(f"{name} ({sha[:12]}) not under /kaggle/input: attach foysalemonshanto/q27-e14-cpu-kit")


shutil.rmtree(K, ignore_errors=True)
with zipfile.ZipFile(io.BytesIO(find(KIT_NAME, KIT_SHA))) as z:
    z.extractall(K)
man = json.loads((K / "MANIFEST.json").read_text())
bad = [k for k, h in man.items() if hashlib.sha256((K / k).read_bytes()).hexdigest() != h]
assert not bad, bad
print("kit OK:", len(man), "files |", sys.version.split()[0], flush=True)
# the GPU notebook's runtime pins (its CELL 1): transformers 4.57.6 + huggingface-hub 0.36.2 from the S49 wheelhouse
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", WH,
                    "transformers==4.57.6", "huggingface-hub==0.36.2"], capture_output=True, text=True)
REPORT["pins"] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-800:]}
try:
    import pytest  # noqa: F401
    REPORT["pytest_source"] = "image"
except ImportError:
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", str(K / "wheels"),
                        "pytest"], capture_output=True, text=True)
    REPORT["pytest_source"] = {"kit wheels rc": r.returncode, "tail": (r.stdout + r.stderr)[-500:]}
print("pins:", REPORT["pins"]["rc"], "| pytest:", REPORT["pytest_source"], flush=True)
ENV = dict(os.environ, E14_SGL_SRC=str(K / "sgl_ref"), E14_FIXTURES_DIR=str(K / "fixtures"), PYTHONIOENCODING="utf-8")
t = time.time()
r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--confcutdir", str(K / "src"),
                    "--rootdir", str(K / "src")] + TESTS, cwd=K / "src", env=ENV, capture_output=True, text=True,
                   timeout=3600)
(OUT / "pytest.log").write_text(r.stdout + r.stderr, encoding="utf-8")
REPORT["pytest"] = {"rc": r.returncode, "sec": round(time.time() - t), "summary": (r.stdout.strip().splitlines() or [""])[-1],
                    "failures": [l for l in r.stdout.splitlines() if l.startswith(("FAILED", "ERROR"))][:40]}
save()
print("pytest:", REPORT["pytest"]["summary"], "| rc", r.returncode, flush=True)
for sc in SCALES:
    for hfile, var, scen in RUNS:
        key = f"{scen}@{sc}"
        env = dict(ENV, E14_HARNESS_ROOT=f"/tmp/e14_h_{scen}_{sc}", **{var: scen})
        t = time.time()
        try:
            r = subprocess.run([sys.executable, str(K / "src/harness" / hfile), str(K / TARGET), str(sc)],
                               cwd=K / "src/harness", env=env, capture_output=True, text=True, timeout=7200)
            out = r.stdout + r.stderr
            rc = r.returncode
        except subprocess.TimeoutExpired as e:
            out, rc = (e.stdout or "") if isinstance(e.stdout, str) else "", "timeout"
        (OUT / f"harness_{scen}_{sc}.log").write_text(out, encoding="utf-8")
        lines = out.splitlines()
        REPORT["harness"][key] = {"rc": rc, "sec": round(time.time() - t), "ok": "@@NB_OK@@" in out,
                                  "summary": [l for l in lines if l.startswith("[e14")],
                                  "fails": [l[:1500] for l in lines if l.startswith("   FAIL")],
                                  "traceback_tail": out[-3000:] if "Traceback" in out else ""}
        save()
        print(key, "->", REPORT["harness"][key]["summary"], "ok" if REPORT["harness"][key]["ok"] else "FAIL",
              REPORT["harness"][key]["fails"][:6], flush=True)
REPORT["sec"] = round(time.time() - T0)
REPORT["all_ok"] = REPORT["pytest"]["rc"] == 0 and all(v["ok"] for v in REPORT["harness"].values())
save()
print("ALL OK" if REPORT["all_ok"] else "NOT OK", "| total", REPORT["sec"], "s", flush=True)
'''
compile(CODE, "<cpu harness>", "exec")
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [
          "# E14 CPU harness (Kaggle pinned image)\n\nUnit tests + the fake-server harness of `" + TARGET + "` on the same "
          "image as the GPU notebook. Kit: `" + KIT_SLUG + "` (" + KIT_NAME + ").\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CODE.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
nb_out = NBDIR / "Q27-E14-CPU-HARNESS.ipynb"
nb_out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("kit", OUT_DIR / KIT_NAME, round(len(kit) / 1e6, 1), "MB,", len(manifest), "files | target", TARGET, "| scales", SCALES)
print("notebook", nb_out.name, hashlib.sha256(nb_out.read_bytes()).hexdigest(), round(nb_out.stat().st_size / 1e3, 1), "kB")
