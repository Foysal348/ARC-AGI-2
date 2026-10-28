"""E17 CPU kit (memory: Kaggle CPU validation before any GPU run). On a Kaggle CPU kernel with the SAME pinned image as
the GPU notebooks (Python 3.12), it runs the e17_hook unit tests and the E17 fake-server notebook harness (all 10 code
cells, 20 checks). Derived from Experiment 14 build_cpu_kit_engine.py.

Writes:
- Build/cpu_kit_e17/q27x_e17_cpu_kit_<sha12>.bin (a zip: e17src/ the hook, its tests, the harness + fake server;
  e15p/ the base harness + fake; e14src/ the E14 modules; fixtures/; sgl_ref/; nb/ the notebook; payload/; MANIFEST.json)
  + dataset-metadata.json -> the private dataset foysalemonshanto/q27-e17-cpu-kit
- Build/cpu_kernel_e17/ the kernel (q27-e17-cpu-harness.ipynb + kernel-metadata.json: CPU, no internet, pinned image)
    python build_cpu_kit_e17.py
"""
import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1]
E17SRC = EXP / "src"
E14SRC = EXP.parent / "Experiment 14 - Optimized Engine on 4xL4 (planned)" / "src"
E15H = Path(r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/src/harness")
E15PAY = Path(r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks/Build/payload")
FIX = Path(r"D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures")
SGL = Path(r"D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang")
KIT_SLUG = "foysalemonshanto/q27-e17-cpu-kit"
KERNEL_SLUG = "foysalemonshanto/q27-e17-cpu-harness"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
OUT_DIR = HERE / "cpu_kit_e17"
NB = "Q27-XHIGH-E17_SGLANG_SILENT_ACCURACY.ipynb"
SGL_FILES = ["_version.py", "srt/dllm/mixin/scheduler.py", "srt/managers/scheduler.py", "srt/managers/schedule_batch.py",
             "srt/managers/min_free_slots_delayer.py"]

files = {}
for n in ("e17_hook.py", "test_e17_hook.py"):
    files["e17src/" + n] = (E17SRC / n).read_bytes()
for n in ("harness_e17_nb.py", "e17_fake_server.py"):
    files["e17src/harness/" + n] = (E17SRC / "harness" / n).read_bytes()
for n in ("harness_e15p_nb.py", "e15p_fake_server.py"):
    files["e15p/" + n] = (E15H / n).read_bytes()
for p in sorted(E14SRC.glob("e14_*.py")):
    files["e14src/" + p.name] = p.read_bytes()
for p in sorted(FIX.rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts:
        files["fixtures/" + p.relative_to(FIX).as_posix()] = p.read_bytes()
for rel in SGL_FILES:
    files["sgl_ref/" + rel] = (SGL / rel).read_bytes()
files["nb/" + NB] = (HERE.parent / NB).read_bytes()
for p in sorted(E15PAY.glob("q27x_e14_modules_*.bin")):
    files["payload/" + p.name] = p.read_bytes()
files["run_tests.py"] = b'''import sys, traceback, importlib
sys.path.insert(0, sys.argv[1])
m = importlib.import_module(sys.argv[2])
ok = bad = 0
for k in sorted(vars(m)):
    if k.startswith("test_"):
        try:
            getattr(m, k)(); ok += 1; print("PASS", k)
        except Exception:
            bad += 1; print("FAIL", k); traceback.print_exc()
print(f"{ok} passed, {bad} failed"); sys.exit(1 if bad else 0)
'''
manifest = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())}
files["MANIFEST.json"] = json.dumps(manifest, indent=0).encode()
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for k in sorted(files):
        zi = zipfile.ZipInfo(k, date_time=(2026, 10, 26, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, files[k])
kit = buf.getvalue()
KIT_SHA = hashlib.sha256(kit).hexdigest()
KIT_NAME = f"q27x_e17_cpu_kit_{KIT_SHA[:12]}.bin"
OUT_DIR.mkdir(exist_ok=True)
for old in OUT_DIR.glob("q27x_e17_cpu_kit_*.bin"):
    old.unlink()
(OUT_DIR / KIT_NAME).write_bytes(kit)
(OUT_DIR / "dataset-metadata.json").write_text(json.dumps(
    {"title": "Q27 E17 CPU kit", "id": KIT_SLUG, "licenses": [{"name": "apache-2.0"}]}, indent=1), encoding="utf-8")

CODE = r'''# E17 CPU harness on Kaggle (pinned image = the GPU notebook's): the e17_hook unit tests + the notebook harness
import hashlib, io, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
KIT_NAME, KIT_SHA = "''' + KIT_NAME + '''", "''' + KIT_SHA + '''"
WH = "/kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"
OUT = Path("/kaggle/working"); K = Path("/tmp/e17_kit"); T0 = time.time()
REPORT = {"kit": KIT_NAME, "python": sys.version, "nb_sha": {}, "runs": {}}


def save():
    (OUT / "cpu_harness_report.json").write_text(json.dumps(REPORT, indent=1), encoding="utf-8")


def find(name, sha):
    for pat in (name, "*/" + name, "*/*/" + name, "*/*/*/" + name, "*/*/*/*/" + name):
        for f in sorted(Path("/kaggle/input").glob(pat)):
            b = f.read_bytes()
            if hashlib.sha256(b).hexdigest() == sha:
                return b
    raise FileNotFoundError(f"{name} ({sha[:12]}) not under /kaggle/input: attach foysalemonshanto/q27-e17-cpu-kit")


shutil.rmtree(K, ignore_errors=True)
with zipfile.ZipFile(io.BytesIO(find(KIT_NAME, KIT_SHA))) as z:
    z.extractall(K)
man = json.loads((K / "MANIFEST.json").read_text())
bad = [k for k, h in man.items() if hashlib.sha256((K / k).read_bytes()).hexdigest() != h]
assert not bad, bad
for nb in sorted((K / "nb").glob("*.ipynb")):
    REPORT["nb_sha"][nb.name] = hashlib.sha256(nb.read_bytes()).hexdigest()
print("kit OK:", len(man), "files |", sys.version.split()[0], "|", REPORT["nb_sha"], flush=True)
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", WH,
                    "transformers==4.57.6", "huggingface-hub==0.36.2"], capture_output=True, text=True)
REPORT["pins"] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-800:]}
print("pins:", r.returncode, flush=True)
RUNS = [("hook_tests", [sys.executable, str(K / "run_tests.py"), str(K / "e17src"), "test_e17_hook"], K / "e17src", {}),
        ("e17_short", [sys.executable, str(K / "e17src/harness/harness_e17_nb.py"), str(K / "nb" / "''' + NB + '''"), "0.03"],
         K / "e17src/harness", {"E14_HARNESS_ROOT": "/tmp/e17_hs", "HARNESS_E17_PROFILE": "E17SHORT"}),
        ("e17_full", [sys.executable, str(K / "e17src/harness/harness_e17_nb.py"), str(K / "nb" / "''' + NB + '''"), "0.03"],
         K / "e17src/harness", {"E14_HARNESS_ROOT": "/tmp/e17_hf", "HARNESS_E17_PROFILE": "E17"})]
ENV = dict(os.environ, E14_SGL_SRC=str(K / "sgl_ref"), E14_FIXTURES_DIR=str(K / "fixtures"), PYTHONIOENCODING="utf-8",
           E14_HARNESS_DIR=str(K / "e14src/harness"), E15P_HARNESS_DIR=str(K / "e15p"), E17_PAYLOAD_DIR=str(K / "payload"),
           E15P_MODE="commit")
(K / "e14src/harness").mkdir(parents=True, exist_ok=True)
for name, cmd, cwd, extra in RUNS:
    t = time.time()
    try:
        r = subprocess.run(cmd, cwd=cwd, env=dict(ENV, **extra), capture_output=True, text=True, timeout=3600)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or "") if isinstance(e.stdout, str) else "", "timeout"
    (OUT / f"run_{name}.log").write_text(out, encoding="utf-8")
    lines = out.splitlines()
    REPORT["runs"][name] = {"rc": rc, "sec": round(time.time() - t),
                            "ok": rc == 0 and ("CHECKS PASS" in out or " 0 failed" in out),
                            "summary": [x for x in lines if "CHECKS PASS" in x or "failed of" in x or " passed, " in x],
                            "fails": [x[:1500] for x in lines if x.startswith("FAIL")],
                            "traceback_tail": out[-3000:] if "Traceback" in out else ""}
    save()
    print(name, "->", REPORT["runs"][name]["summary"], REPORT["runs"][name]["fails"][:6], flush=True)
REPORT["sec"] = round(time.time() - T0)
REPORT["all_ok"] = all(v["ok"] for v in REPORT["runs"].values())
save()
print("ALL OK" if REPORT["all_ok"] else "NOT OK", "| total", REPORT["sec"], "s", flush=True)
'''
compile(CODE, "<cpu harness>", "exec")
nbj = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [
          "# E17 CPU harness (Kaggle pinned image)\n\nThe e17_hook unit tests and the E17 fake-server notebook harness "
          "on the same image as the GPU notebook. Kit: `" + KIT_SLUG + "` (" + KIT_NAME + ").\n"]},
                 {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CODE.splitlines(True)}],
       "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
KD = HERE / "cpu_kernel_e17"
KD.mkdir(exist_ok=True)
(KD / "q27-e17-cpu-harness.ipynb").write_text(json.dumps(nbj, indent=1, ensure_ascii=False), encoding="utf-8")
(KD / "kernel-metadata.json").write_text(json.dumps({
    "id": KERNEL_SLUG, "title": "Q27 E17 CPU Harness", "code_file": "q27-e17-cpu-harness.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
    "keywords": [], "dataset_sources": ["mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3", KIT_SLUG], "kernel_sources": [],
    "competition_sources": [], "model_sources": [], "docker_image": IMAGE, "machine_shape": "None"}, indent=2),
    encoding="utf-8")
print("kit", OUT_DIR / KIT_NAME, round(len(kit) / 1e6, 1), "MB,", len(manifest), "files")
print("notebook", NB, hashlib.sha256((HERE.parent / NB).read_bytes()).hexdigest()[:16])
