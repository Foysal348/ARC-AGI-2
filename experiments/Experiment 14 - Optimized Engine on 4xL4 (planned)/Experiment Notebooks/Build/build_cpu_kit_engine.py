"""Engine-matrix CPU kit (memory: Kaggle CPU validation before any GPU run): the EMS (SGLang) and VMS (vLLM 0.32.0)
fake-server notebook harnesses on a Kaggle CPU kernel with the SAME pinned image as the GPU notebooks (Python 3.12).
Derived from Experiment 15 build_cpu_kit_e15p.py. Runs: ems (commit), vms (commit), vms_bad_int4 (the #58807 ignore list).

Writes:
- Build/cpu_kit_engine/q27x_engine_cpu_kit_<sha12>.bin (a zip: e15p/ the E15P harness + SGLang fake server, e14src/ the
  E14 modules + harness (EMS / VMS harnesses, fake vLLM server), e1105/, fixtures/, sgl_ref/, redhat_index.json, nb/ both
  notebooks + the payload, MANIFEST.json) + dataset-metadata.json -> the private dataset foysalemonshanto/q27-engine-cpu-kit
- Build/cpu_kernel_engine/ the kernel (q27-engine-cpu-harness.ipynb + kernel-metadata.json: CPU, no internet, pinned image)
    python build_cpu_kit_engine.py
"""
import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
E14SRC = HERE.parents[1] / "src"
E15H = Path(r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/src/harness")
E15PAY = Path(r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks/Build/payload")
E1105H = Path(r"D:/ARC AGI 2/experiments/Experiment 11 - SGLang 0.5.21 + RedHat INT4 + DFlash2 (planned)/Experiment Review/E11_05_AUDIT_FOR_GPT/harness")
FIX = Path(r"D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures")
SGL = Path(r"D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang")
REDHAT_INDEX = Path(r"D:/ARC AGI 2/_scratch/kernel_issue/redhat_index.json")
KIT_SLUG = "foysalemonshanto/q27-engine-cpu-kit"
KERNEL_SLUG = "foysalemonshanto/q27-engine-cpu-harness"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
OUT_DIR = HERE / "cpu_kit_engine"
NB_SAFE = "Q27-XHIGH-EMS_SAFE_ENGINE_MATRIX_SGLANG.ipynb"      # rev 2f: EMS split in two (+ VMS = 3 parts)
NB_RISKY = "Q27-XHIGH-EMS_RISKY_ENGINE_MATRIX_SGLANG.ipynb"
NB_MTPX = "Q27-XHIGH-EMS_MTPX_ENGINE_MATRIX_SGLANG.ipynb"       # rev 2h
NB_VMS = "Q27-XHIGH-VMS_ENGINE_MATRIX_VLLM.ipynb"
RUNS = [("ems_all", "e14src/harness/harness_ems_nb.py", NB_SAFE, {"HARNESS_EMS_PART": "ALL"}),
        ("ems_safe", "e14src/harness/harness_ems_nb.py", NB_SAFE, {"HARNESS_EMS_PART": "SAFE"}),
        ("ems_risky", "e14src/harness/harness_ems_nb.py", NB_RISKY, {"HARNESS_EMS_PART": "RISKY"}),
        ("ems_mtpx", "e14src/harness/harness_ems_nb.py", NB_MTPX, {"HARNESS_EMS_PART": "MTPX"}),
        ("vms", "e14src/harness/harness_vms_nb.py", NB_VMS, {}),
        ("vms_bad_int4", "e14src/harness/harness_vms_nb.py", NB_VMS, {"VMS_BAD_INT4": "1"})]
SGL_FILES = ["_version.py", "srt/dllm/mixin/scheduler.py", "srt/managers/scheduler.py", "srt/managers/schedule_batch.py",
             "srt/managers/min_free_slots_delayer.py"]

files = {}
for n in ("harness_e15p_nb.py", "e15p_fake_server.py", "make_e15p_fake_server.py"):
    files["e15p/" + n] = (E15H / n).read_bytes()
for p in sorted(E14SRC.glob("e14_*.py")):
    files["e14src/" + p.name] = p.read_bytes()
for p in sorted((E14SRC / "harness").glob("*.py")):
    files["e14src/harness/" + p.name] = p.read_bytes()
for p in sorted((E1105H / "fake_vllm").rglob("*.py")):
    files["e1105/" + p.relative_to(E1105H).as_posix()] = p.read_bytes()
beh = (E1105H / "beh_e11_05.py").read_text(encoding="utf-8")
_a = 'TOKDIR_INT4 = Path(r"D:/ARC AGI 2/_scratch/session_1c6db7aa/e11_03_harness/tok")'
assert beh.count(_a) == 1
beh = beh.replace(_a, _a + '''
try:   # CPU kit: portable paths (the kit's fixtures)
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
files["redhat_index.json"] = REDHAT_INDEX.read_bytes()
for nb in (NB_SAFE, NB_RISKY, NB_MTPX, NB_VMS):
    files["nb/" + nb] = (HERE.parent / nb).read_bytes()
for p in sorted(E15PAY.glob("q27x_e14_modules_*.bin")):
    files["payload/" + p.name] = p.read_bytes()
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
KIT_NAME = f"q27x_engine_cpu_kit_{KIT_SHA[:12]}.bin"       # .bin: kept out of Kaggle's archive unpacking
OUT_DIR.mkdir(exist_ok=True)
for old in OUT_DIR.glob("q27x_engine_cpu_kit_*.bin"):
    old.unlink()
(OUT_DIR / KIT_NAME).write_bytes(kit)
(OUT_DIR / "dataset-metadata.json").write_text(json.dumps(
    {"title": "Q27 engine matrix CPU kit", "id": KIT_SLUG, "licenses": [{"name": "apache-2.0"}]}, indent=1), encoding="utf-8")

CODE = r'''# Engine-matrix CPU harness on Kaggle (pinned image = the GPU notebooks'): EMS (SGLang) + VMS (vLLM 0.32.0) harnesses
import hashlib, io, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
KIT_NAME, KIT_SHA = "''' + KIT_NAME + '''", "''' + KIT_SHA + '''"
RUNS = ''' + json.dumps(RUNS) + r'''
WH = "/kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"
OUT = Path("/kaggle/working"); K = Path("/tmp/engine_kit"); T0 = time.time()
REPORT = {"kit": KIT_NAME, "python": sys.version, "nb_sha": {}, "harness": {}}


def save():
    (OUT / "cpu_harness_report.json").write_text(json.dumps(REPORT, indent=1), encoding="utf-8")


def find(name, sha):
    for pat in (name, "*/" + name, "*/*/" + name, "*/*/*/" + name, "*/*/*/*/" + name):
        for f in sorted(Path("/kaggle/input").glob(pat)):
            b = f.read_bytes()
            if hashlib.sha256(b).hexdigest() == sha:
                return b
    raise FileNotFoundError(f"{name} ({sha[:12]}) not under /kaggle/input: attach foysalemonshanto/q27-engine-cpu-kit")


shutil.rmtree(K, ignore_errors=True)
with zipfile.ZipFile(io.BytesIO(find(KIT_NAME, KIT_SHA))) as z:
    z.extractall(K)
man = json.loads((K / "MANIFEST.json").read_text())
bad = [k for k, h in man.items() if hashlib.sha256((K / k).read_bytes()).hexdigest() != h]
assert not bad, bad
for nb in sorted((K / "nb").glob("*.ipynb")):
    REPORT["nb_sha"][nb.name] = hashlib.sha256(nb.read_bytes()).hexdigest()
print("kit OK:", len(man), "files |", sys.version.split()[0], "|", REPORT["nb_sha"], flush=True)
# the GPU notebooks' runtime pins (their install cell): transformers 4.57.6 + huggingface-hub 0.36.2 from the S49 wheelhouse
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", WH,
                    "transformers==4.57.6", "huggingface-hub==0.36.2"], capture_output=True, text=True)
REPORT["pins"] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-800:]}
print("pins:", r.returncode, flush=True)
ENV = dict(os.environ, E14_SGL_SRC=str(K / "sgl_ref"), E14_FIXTURES_DIR=str(K / "fixtures"), PYTHONIOENCODING="utf-8",
           E14_HARNESS_DIR=str(K / "e14src/harness"), E1105_HARNESS_DIR=str(K / "e1105"), E15P_HARNESS_DIR=str(K / "e15p"),
           EMS_PAYLOAD_DIR=str(K / "payload"), VMS_REDHAT_INDEX=str(K / "redhat_index.json"))
for name, script, nb, extra in RUNS:
    env = dict(ENV, E14_HARNESS_ROOT=f"/tmp/engine_h_{name}", E15P_MODE="commit", **extra)
    t = time.time()
    try:
        r = subprocess.run([sys.executable, str(K / script), str(K / "nb" / nb), "0.03"],
                           cwd=K / "e14src/harness", env=env, capture_output=True, text=True, timeout=3600)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or "") if isinstance(e.stdout, str) else "", "timeout"
    (OUT / f"harness_{name}.log").write_text(out, encoding="utf-8")
    lines = out.splitlines()
    REPORT["harness"][name] = {"rc": rc, "sec": round(time.time() - t), "ok": rc == 0 and "CHECKS PASS" in out,
                               "summary": [x for x in lines if "CHECKS PASS" in x or "failed of" in x],
                               "fails": [x[:1500] for x in lines if x.startswith("FAIL")],
                               "cell_sec": [x for x in lines if x.startswith("cell seconds")],
                               "traceback_tail": out[-3000:] if "Traceback" in out else ""}
    save()
    print(name, "->", REPORT["harness"][name]["summary"], REPORT["harness"][name]["fails"][:6], flush=True)
REPORT["sec"] = round(time.time() - T0)
REPORT["all_ok"] = all(v["ok"] for v in REPORT["harness"].values())
save()
print("ALL OK" if REPORT["all_ok"] else "NOT OK", "| total", REPORT["sec"], "s", flush=True)
'''
compile(CODE, "<cpu harness>", "exec")
nbj = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [
          "# Engine-matrix CPU harness (Kaggle pinned image)\n\nThe fake-server harnesses of the EMS (SGLang 0.5.21) and VMS "
          "(vLLM 0.32.0) engine-matrix notebooks, plus the VMS #58807 bad-ignore-list case, on the same image as the GPU "
          "notebooks. Kit: `" + KIT_SLUG + "` (" + KIT_NAME + ").\n"]},
                 {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CODE.splitlines(True)}],
       "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
KD = HERE / "cpu_kernel_engine"
KD.mkdir(exist_ok=True)
(KD / "q27-engine-cpu-harness.ipynb").write_text(json.dumps(nbj, indent=1, ensure_ascii=False), encoding="utf-8")
(KD / "kernel-metadata.json").write_text(json.dumps({
    "id": KERNEL_SLUG, "title": "Q27 Engine CPU Harness", "code_file": "q27-engine-cpu-harness.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
    "keywords": [], "dataset_sources": ["mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3", KIT_SLUG], "kernel_sources": [],
    "competition_sources": [], "model_sources": [], "docker_image": IMAGE, "machine_shape": "None"}, indent=2),
    encoding="utf-8")
print("kit", OUT_DIR / KIT_NAME, round(len(kit) / 1e6, 1), "MB,", len(manifest), "files")
for nb in (NB_SAFE, NB_RISKY, NB_MTPX, NB_VMS):
    print("notebook", nb, hashlib.sha256((HERE.parent / nb).read_bytes()).hexdigest()[:16])
