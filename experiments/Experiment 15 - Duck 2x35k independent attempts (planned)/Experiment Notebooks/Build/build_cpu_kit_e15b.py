"""E15B CPU kit (memory: Kaggle CPU validation before any GPU run): the B unit tests + the fake-server notebook harness of
both E15B notebooks, run on a Kaggle CPU kernel with the SAME pinned image as the GPU notebook (Python 3.12). Derived from
Exp 14's build_cpu_kit.py.

Writes:
- Build/cpu_kit/q27x_e15b_cpu_kit_<sha12>.bin  (a zip: e15/ tests + harness, e14src/ the E14 modules + fake server,
  fixtures/, sgl_ref/, nb/ both notebooks + Build/payload, MANIFEST.json) + dataset-metadata.json
  -> the private dataset foysalemonshanto/q27-e15b-cpu-kit
- Build/cpu_kernel/ the kernel (q27-e15b-cpu-harness.ipynb + kernel-metadata.json: CPU, no internet, the pinned image)
    python build_cpu_kit_e15b.py
"""
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
E15 = HERE.parents[1]
E14SRC = Path(r"D:/ARC AGI 2/experiments/Experiment 14 - Optimized Engine on 4xL4 (planned)/src")
FIX = Path(r"D:/ARC AGI 2/_scratch/g4b_bundle_verify/tests/fixtures")
SGL = Path(r"D:/ARC AGI 2/_scratch/session_1c6db7aa/sgl_src/src/sglang")
KIT_SLUG = "foysalemonshanto/q27-e15b-cpu-kit"
KERNEL_SLUG = "foysalemonshanto/q27-e15b-cpu-harness"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
OUT_DIR = HERE / "cpu_kit"
# user 2026-10-25: the EVAL120 notebooks are not needed now; both SUBMIT variants (45k + 30k, 40k + 35k)
NBS = {"s45": "Q27-XHIGH-E15B_SUBMIT_45K30K_S16_R3H_DUCK_SGLANG.ipynb", "s40": "Q27-XHIGH-E15B_SUBMIT_40K35K_S16_R3H_DUCK_SGLANG.ipynb",
       "s50": "Q27-XHIGH-E15B_SUBMIT_50K45K_S16_R3H_DUCK_SGLANG.ipynb"}     # 50K45K: user 2026-10-25
# (label, notebook key, E14S_MODE, E15B_FAULT)
RUNS = [("s45_commit", "s45", "commit", ""), ("s45_rerun", "s45", "rerun", ""), ("s45_fault_kill", "s45", "commit", "kill"),
        ("s45_fault_kill2", "s45", "commit", "kill2"), ("s45_fault_slow", "s45", "commit", "slow"),
        ("s40_commit", "s40", "commit", ""), ("s40_rerun", "s40", "rerun", ""),
        ("s50_commit", "s50", "commit", ""), ("s50_rerun", "s50", "rerun", ""), ("s50_fault_kill", "s50", "commit", "kill")]
if "--r3i" in sys.argv:      # rev 3i (user 2026-10-25): R + stage 3, EVAL120 6-h notebook (R3H passed in kit v5)
    NBS = {"s50i": "Q27-XHIGH-E15B_EVAL120_50K45K_S16_R3I_DUCK_SGLANG.ipynb"}
    RUNS = [("s50i_eval120", "s50i", "eval120", ""), ("s50i_rerun", "s50i", "rerun", "")]
if "--r3j" in sys.argv:      # rev 3j (user 2026-10-25): the two EVAL120 6-h order notebooks
    NBS = {"j1": "Q27-XHIGH-E15B_EVAL120_50K45K_S16_R3K1_DUCK_SGLANG.ipynb",      # rev 3k (was R3J)
           "j2": "Q27-XHIGH-E15B_EVAL120_50K45K_S16_R3K2_DUCK_SGLANG.ipynb"}
    RUNS = [("j1_eval120", "j1", "eval120", ""), ("j1_rerun", "j1", "rerun", ""),
            ("j2_eval120", "j2", "eval120", ""), ("j2_rerun", "j2", "rerun", "")]
SGL_FILES = ["_version.py", "srt/dllm/mixin/scheduler.py", "srt/managers/scheduler.py", "srt/managers/schedule_batch.py",
             "srt/managers/min_free_slots_delayer.py"]

files = {}
for n in ("test_duck_b.py", "verify_items.py"):
    files["e15/" + n] = (E15 / "src" / n).read_bytes()
files["e15/harness/harness_e15b_nb.py"] = (E15 / "src" / "harness" / "harness_e15b_nb.py").read_bytes()
for p in sorted(E14SRC.glob("e14_*.py")):
    files["e14src/" + p.name] = p.read_bytes()
for p in sorted((E14SRC / "harness").glob("*.py")):
    files["e14src/harness/" + p.name] = p.read_bytes()
for p in sorted(FIX.rglob("*")):
    if p.is_file() and "__pycache__" not in p.parts:
        files["fixtures/" + p.relative_to(FIX).as_posix()] = p.read_bytes()
for rel in SGL_FILES:
    files["sgl_ref/" + rel] = (SGL / rel).read_bytes()
for n in NBS.values():
    files["nb/" + n] = (HERE.parent / n).read_bytes()
for p in sorted((HERE / "payload").glob("q27x_*.bin")):
    files["nb/Build/payload/" + p.name] = p.read_bytes()
# the Duck package the target notebooks load (by name in their q27x_payload call): never "the first bin in the folder"
# (kit v2 took the older 19192e8006e0 because two Duck bins were in the payload folder)
import re as _re
_refs = {_re.search(r'q27x_payload\(\\"(q27x_e15b_duck_[0-9a-f]{12}\.bin)', (HERE.parent / n).read_text(encoding="utf-8")).group(1)
         for n in NBS.values()}
assert len(_refs) == 1, _refs
DUCK_BIN = "nb/Build/payload/" + _refs.pop()
assert DUCK_BIN in files, DUCK_BIN
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
KIT_NAME = f"q27x_e15b_cpu_kit_{KIT_SHA[:12]}.bin"       # .bin: kept out of Kaggle's archive unpacking
OUT_DIR.mkdir(exist_ok=True)
for old in OUT_DIR.glob("q27x_e15b_cpu_kit_*.bin"):
    old.unlink()
(OUT_DIR / KIT_NAME).write_bytes(kit)
(OUT_DIR / "dataset-metadata.json").write_text(json.dumps(
    {"title": "Q27 E15B CPU kit", "id": KIT_SLUG, "licenses": [{"name": "apache-2.0"}]}, indent=1), encoding="utf-8")

CODE = r'''# E15B CPU harness on Kaggle (pinned image = the GPU notebook's): B unit tests + the fake-server harness of both notebooks
import hashlib, io, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
KIT_NAME, KIT_SHA = "''' + KIT_NAME + '''", "''' + KIT_SHA + '''"
NBS = ''' + json.dumps(NBS) + r'''
RUNS = ''' + json.dumps(RUNS) + r'''
DUCK_BIN = "''' + DUCK_BIN + r'''"
WH = "/kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"
OUT = Path("/kaggle/working"); K = Path("/tmp/e15b_kit"); T0 = time.time()
REPORT = {"kit": KIT_NAME, "python": sys.version, "nb_sha": {}, "unit": None, "harness": {}}


def save():
    (OUT / "cpu_harness_report.json").write_text(json.dumps(REPORT, indent=1), encoding="utf-8")


def find(name, sha):
    for pat in (name, "*/" + name, "*/*/" + name, "*/*/*/" + name, "*/*/*/*/" + name):
        for f in sorted(Path("/kaggle/input").glob(pat)):
            b = f.read_bytes()
            if hashlib.sha256(b).hexdigest() == sha:
                return b
    raise FileNotFoundError(f"{name} ({sha[:12]}) not under /kaggle/input: attach foysalemonshanto/q27-e15b-cpu-kit")


shutil.rmtree(K, ignore_errors=True)
with zipfile.ZipFile(io.BytesIO(find(KIT_NAME, KIT_SHA))) as z:
    z.extractall(K)
man = json.loads((K / "MANIFEST.json").read_text())
bad = [k for k, h in man.items() if hashlib.sha256((K / k).read_bytes()).hexdigest() != h]
assert not bad, bad
REPORT["nb_sha"] = {k: hashlib.sha256((K / "nb" / v).read_bytes()).hexdigest() for k, v in NBS.items()}
print("kit OK:", len(man), "files |", sys.version.split()[0], "|", REPORT["nb_sha"], flush=True)
# the GPU notebook's runtime pins (its CELL 1): transformers 4.57.6 + huggingface-hub 0.36.2 from the S49 wheelhouse
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", WH,
                    "transformers==4.57.6", "huggingface-hub==0.36.2"], capture_output=True, text=True)
REPORT["pins"] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-800:]}
print("pins:", r.returncode, flush=True)
PKG = Path("/tmp/e15b_duck_pkg"); shutil.rmtree(PKG, ignore_errors=True)
with zipfile.ZipFile(K / DUCK_BIN) as z:          # the B Duck package exactly as the notebooks load it
    z.extractall(PKG)
ENV = dict(os.environ, E14_SGL_SRC=str(K / "sgl_ref"), E14_FIXTURES_DIR=str(K / "fixtures"), PYTHONIOENCODING="utf-8",
           E14_HARNESS_DIR=str(K / "e14src/harness"), DUCK_B_PKG=str(PKG),
           E15B_TASKS=str(K / "fixtures/data/arc-agi_evaluation_challenges.json"))
t = time.time()
r = subprocess.run([sys.executable, str(K / "e15/test_duck_b.py")], cwd=K / "e15", env=ENV, capture_output=True,
                   text=True, timeout=3600)
(OUT / "test_duck_b.log").write_text(r.stdout + r.stderr, encoding="utf-8")
_l = (r.stdout + r.stderr).strip().splitlines()
REPORT["unit"] = {"rc": r.returncode, "sec": round(time.time() - t), "summary": _l[-1] if _l else "",
                  "passes": sum(1 for x in _l if x.startswith("PASS")), "fails": [x for x in _l if x.startswith("FAIL")][:40],
                  "skips": [x for x in _l if x.startswith("SKIP")][:40]}
save()
print("unit:", REPORT["unit"]["summary"], "| PASS", REPORT["unit"]["passes"], "| rc", r.returncode, flush=True)
for label, nbk, mode, fault in RUNS:
    env = dict(ENV, E14_HARNESS_ROOT=f"/tmp/e15b_h_{label}", E14S_MODE=mode)
    env.pop("E15B_FAULT", None)
    if fault:
        env["E15B_FAULT"] = fault
    t = time.time()
    try:
        r = subprocess.run([sys.executable, str(K / "e15/harness/harness_e15b_nb.py"), str(K / "nb" / NBS[nbk]), "0.03"],
                           cwd=K / "e15/harness", env=env, capture_output=True, text=True, timeout=5400)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or "") if isinstance(e.stdout, str) else "", "timeout"
    (OUT / f"harness_{label}.log").write_text(out, encoding="utf-8")
    lines = out.splitlines()
    REPORT["harness"][label] = {"rc": rc, "sec": round(time.time() - t), "ok": "@@NB_OK@@" in out,
                                "summary": [x for x in lines if x.startswith("[e15b")],
                                "fails": [x[:1500] for x in lines if x.startswith("   FAIL")],
                                "traceback_tail": out[-3000:] if "Traceback" in out and "@@NB_OK@@" not in out else ""}
    save()
    print(label, "->", REPORT["harness"][label]["summary"], "ok" if REPORT["harness"][label]["ok"] else "FAIL",
          REPORT["harness"][label]["fails"][:6], flush=True)
REPORT["sec"] = round(time.time() - T0)
REPORT["all_ok"] = REPORT["unit"]["rc"] == 0 and all(v["ok"] for v in REPORT["harness"].values())
save()
print("ALL OK" if REPORT["all_ok"] else "NOT OK", "| total", REPORT["sec"], "s", flush=True)
'''
compile(CODE, "<cpu harness>", "exec")
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [
          "# E15B CPU harness (Kaggle pinned image)\n\nThe B unit tests + the fake-server harness of both E15B notebooks "
          "(45K30K: commit, rerun, fault kill / kill2 / slow; 40K35K: commit, rerun) on the same image as the GPU notebook. Kit: `" + KIT_SLUG +
          "` (" + KIT_NAME + ").\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CODE.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
KD = HERE / "cpu_kernel"
KD.mkdir(exist_ok=True)
(KD / "q27-e15b-cpu-harness.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
(KD / "kernel-metadata.json").write_text(json.dumps({
    "id": KERNEL_SLUG, "title": "Q27 E15B CPU Harness", "code_file": "q27-e15b-cpu-harness.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
    "keywords": [], "dataset_sources": ["mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3", KIT_SLUG], "kernel_sources": [],
    "competition_sources": [], "model_sources": [], "docker_image": IMAGE, "machine_shape": "None"}, indent=2),
    encoding="utf-8")
print("kit", OUT_DIR / KIT_NAME, round(len(kit) / 1e6, 1), "MB,", len(manifest), "files")
print("kernel", KD, {k: hashlib.sha256((HERE.parent / v).read_bytes()).hexdigest()[:16] for k, v in NBS.items()})
