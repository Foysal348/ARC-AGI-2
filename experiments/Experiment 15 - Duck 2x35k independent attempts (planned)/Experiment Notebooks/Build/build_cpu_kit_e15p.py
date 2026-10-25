"""E15P CPU kit (memory: Kaggle CPU validation before any GPU run): the E15P fake-server notebook harness, all 8 modes
(commit, kvgate, mambagate, drain, rerun, tokfail, smokefail, death), run on a Kaggle CPU kernel with the SAME pinned image
as the GPU notebook (Python 3.12). Derived from build_cpu_kit_e15b.py.

Writes:
- Build/cpu_kit_e15p/q27x_e15p_cpu_kit_<sha12>.bin (a zip: e15p/ harness + fake server, e14src/ the E14 modules + harness,
  e1105/ the E11-05 fake vllm + a path-portable beh_e11_05, fixtures/, sgl_ref/, nb/ the notebook + Build/payload,
  MANIFEST.json) + dataset-metadata.json -> the private dataset foysalemonshanto/q27-e15p-cpu-kit
- Build/cpu_kernel_e15p/ the kernel (q27-e15p-cpu-harness.ipynb + kernel-metadata.json: CPU, no internet, pinned image)
    python build_cpu_kit_e15p.py
"""
import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
E15 = HERE.parents[1]
E14SRC = Path(r"D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/src")
E1105H = Path(r"D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Review/E11_05_AUDIT_FOR_GPT/harness")
FIX = Path(r"D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures")
SGL = Path(r"D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang")
KIT_SLUG = "foysalemonshanto/q27-e15p-cpu-kit"
KERNEL_SLUG = "foysalemonshanto/q27-e15p-cpu-harness"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
OUT_DIR = HERE / "cpu_kit_e15p"
NB = "Q27-XHIGH-E15P_EVAL120_S60_S15_POETIQ_SGLANG.ipynb"
MODES = ["commit", "kvgate", "mambagate", "drain", "rerun", "tokfail", "smokefail", "death"]
SGL_FILES = ["_version.py", "srt/dllm/mixin/scheduler.py", "srt/managers/scheduler.py", "srt/managers/schedule_batch.py",
             "srt/managers/min_free_slots_delayer.py"]

files = {}
for n in ("harness_e15p_nb.py", "e15p_fake_server.py", "make_e15p_fake_server.py"):
    files["e15p/" + n] = (E15 / "src" / "harness" / n).read_bytes()
for p in sorted(E14SRC.glob("e14_*.py")):
    files["e14src/" + p.name] = p.read_bytes()
for p in sorted((E14SRC / "harness").glob("*.py")):
    files["e14src/harness/" + p.name] = p.read_bytes()
for p in sorted((E1105H / "fake_vllm").rglob("*.py")):
    files["e1105/" + p.relative_to(E1105H).as_posix()] = p.read_bytes()
# beh_e11_05 with portable paths: the data and the INT4 tokenizer come from the kit's fixtures (e12_fixtures)
beh = (E1105H / "beh_e11_05.py").read_text(encoding="utf-8")
_a = 'TOKDIR_INT4 = Path(r"D:/ARC AGI 2/_scratch/session_1c6db7aa/e11_03_harness/tok")'
assert beh.count(_a) == 1
beh = beh.replace(_a, _a + '''
try:   # E15P CPU kit: portable paths (the kit's fixtures)
    import e12_fixtures as _F
    DATA, TOKDIR_INT4 = _F.DATA, _F.TOKDIR_INT4
except Exception:
    pass''')
files["e1105/beh_e11_05.py"] = beh.encode("utf-8")
for p in sorted(FIX.rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts:
        files["fixtures/" + p.relative_to(FIX).as_posix()] = p.read_bytes()
for rel in SGL_FILES:
    files["sgl_ref/" + rel] = (SGL / rel).read_bytes()
files["nb/" + NB] = (HERE.parent / NB).read_bytes()
for p in sorted((HERE / "payload").glob("q27x_e14_modules_*.bin")):      # E15P reads only the E14 modules zip
    files["nb/Build/payload/" + p.name] = p.read_bytes()
manifest = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())}
files["MANIFEST.json"] = json.dumps(manifest, indent=0).encode()
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for k in sorted(files):
        zi = zipfile.ZipInfo(k, date_time=(2026, 10, 25, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, files[k])
kit = buf.getvalue()
KIT_SHA = hashlib.sha256(kit).hexdigest()
KIT_NAME = f"q27x_e15p_cpu_kit_{KIT_SHA[:12]}.bin"       # .bin: kept out of Kaggle's archive unpacking
OUT_DIR.mkdir(exist_ok=True)
for old in OUT_DIR.glob("q27x_e15p_cpu_kit_*.bin"):
    old.unlink()
(OUT_DIR / KIT_NAME).write_bytes(kit)
(OUT_DIR / "dataset-metadata.json").write_text(json.dumps(
    {"title": "Q27 E15P CPU kit", "id": KIT_SLUG, "licenses": [{"name": "apache-2.0"}]}, indent=1), encoding="utf-8")

CODE = r'''# E15P CPU harness on Kaggle (pinned image = the GPU notebook's): the fake-server harness, all 8 modes
import hashlib, io, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
KIT_NAME, KIT_SHA = "''' + KIT_NAME + '''", "''' + KIT_SHA + '''"
NB = "''' + NB + '''"
MODES = ''' + json.dumps(MODES) + r'''
WH = "/kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"
OUT = Path("/kaggle/working"); K = Path("/tmp/e15p_kit"); T0 = time.time()
REPORT = {"kit": KIT_NAME, "python": sys.version, "nb_sha": None, "harness": {}}


def save():
    (OUT / "cpu_harness_report.json").write_text(json.dumps(REPORT, indent=1), encoding="utf-8")


def find(name, sha):
    for pat in (name, "*/" + name, "*/*/" + name, "*/*/*/" + name, "*/*/*/*/" + name):
        for f in sorted(Path("/kaggle/input").glob(pat)):
            b = f.read_bytes()
            if hashlib.sha256(b).hexdigest() == sha:
                return b
    raise FileNotFoundError(f"{name} ({sha[:12]}) not under /kaggle/input: attach foysalemonshanto/q27-e15p-cpu-kit")


shutil.rmtree(K, ignore_errors=True)
with zipfile.ZipFile(io.BytesIO(find(KIT_NAME, KIT_SHA))) as z:
    z.extractall(K)
man = json.loads((K / "MANIFEST.json").read_text())
bad = [k for k, h in man.items() if hashlib.sha256((K / k).read_bytes()).hexdigest() != h]
assert not bad, bad
REPORT["nb_sha"] = hashlib.sha256((K / "nb" / NB).read_bytes()).hexdigest()
print("kit OK:", len(man), "files |", sys.version.split()[0], "| notebook", REPORT["nb_sha"], flush=True)
# the GPU notebook's runtime pins (its install cell): transformers 4.57.6 + huggingface-hub 0.36.2 from the S49 wheelhouse
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", WH,
                    "transformers==4.57.6", "huggingface-hub==0.36.2"], capture_output=True, text=True)
REPORT["pins"] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-800:]}
print("pins:", r.returncode, flush=True)
ENV = dict(os.environ, E14_SGL_SRC=str(K / "sgl_ref"), E14_FIXTURES_DIR=str(K / "fixtures"), PYTHONIOENCODING="utf-8",
           E14_HARNESS_DIR=str(K / "e14src/harness"), E1105_HARNESS_DIR=str(K / "e1105"))
for mode in MODES:
    env = dict(ENV, E14_HARNESS_ROOT=f"/tmp/e15p_h_{mode}", E15P_MODE=mode)
    t = time.time()
    try:
        r = subprocess.run([sys.executable, str(K / "e15p/harness_e15p_nb.py"), str(K / "nb" / NB), "0.03"],
                           cwd=K / "e15p", env=env, capture_output=True, text=True, timeout=3600)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or "") if isinstance(e.stdout, str) else "", "timeout"
    (OUT / f"harness_{mode}.log").write_text(out, encoding="utf-8")
    lines = out.splitlines()
    REPORT["harness"][mode] = {"rc": rc, "sec": round(time.time() - t), "ok": rc == 0 and "CHECKS PASS" in out,
                               "summary": [x for x in lines if "CHECKS PASS" in x or "failed of" in x],
                               "fails": [x[:1500] for x in lines if x.startswith("FAIL")],
                               "cell_sec": [x for x in lines if x.startswith("cell seconds")],
                               "traceback_tail": out[-3000:] if "Traceback" in out else ""}
    save()
    print(mode, "->", REPORT["harness"][mode]["summary"], REPORT["harness"][mode]["fails"][:6], flush=True)
REPORT["sec"] = round(time.time() - T0)
REPORT["all_ok"] = all(v["ok"] for v in REPORT["harness"].values())
save()
print("ALL OK" if REPORT["all_ok"] else "NOT OK", "| total", REPORT["sec"], "s", flush=True)
'''
compile(CODE, "<cpu harness>", "exec")
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [
          "# E15P CPU harness (Kaggle pinned image)\n\nThe fake-server harness of the E15P rev 2 notebook, all 8 modes "
          "(commit, kvgate, mambagate, drain, rerun, tokfail, smokefail, death) on the same image as the GPU notebook. Kit: `"
          + KIT_SLUG + "` (" + KIT_NAME + ").\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CODE.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
KD = HERE / "cpu_kernel_e15p"
KD.mkdir(exist_ok=True)
(KD / "q27-e15p-cpu-harness.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
(KD / "kernel-metadata.json").write_text(json.dumps({
    "id": KERNEL_SLUG, "title": "Q27 E15P CPU Harness", "code_file": "q27-e15p-cpu-harness.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
    "keywords": [], "dataset_sources": ["mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3", KIT_SLUG], "kernel_sources": [],
    "competition_sources": [], "model_sources": [], "docker_image": IMAGE, "machine_shape": "None"}, indent=2),
    encoding="utf-8")
print("kit", OUT_DIR / KIT_NAME, round(len(kit) / 1e6, 1), "MB,", len(manifest), "files")
print("notebook", NB, hashlib.sha256((HERE.parent / NB).read_bytes()).hexdigest()[:16])
