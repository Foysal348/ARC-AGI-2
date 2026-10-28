"""E18 CPU kit (memory: Kaggle CPU validation before any GPU run): the E18 fake-server notebook harness, all 6 scenarios
(mixed, kvgate, hardstop, rerun, lmofail, allfail), on a Kaggle CPU kernel with the SAME pinned image as the GPU notebook
(Python 3.12) and the notebook's tokenizer pins (transformers 4.57.6, tokenizers 0.22.2, huggingface-hub 0.36.2).
Derived from build_cpu_kit_e15p.py (Experiment 15).

Writes:
- Build/cpu_kit_e18/q27x_e18_cpu_kit_<sha12>.bin (a zip: harness/ the harness + fake server, data/ the eval JSON,
  tok/ the Qwen3.8 tokenizer, nb/ the notebook, MANIFEST.json) + dataset-metadata.json -> foysalemonshanto/q27-e18-cpu-kit
- Build/cpu_kernel_e18/ the kernel (q27-e18-cpu-harness.ipynb + kernel-metadata.json: CPU, no internet, pinned image)
    python build_cpu_kit_e18.py
"""
import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
E18 = HERE.parents[1]
DATA = Path(r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/"
            r"E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data")
TOK = Path(r"D:/ARC AGI 2/_scratch/session_1c6db7aa/e11_03_harness/tok")
KIT_SLUG = "foysalemonshanto/q27-e18-cpu-kit"
KERNEL_SLUG = "foysalemonshanto/q27-e18-cpu-harness"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
OUT_DIR = HERE / "cpu_kit_e18"
NB = "Q27-XHIGH-E18_S1_60K_VLLM032_S20_R2.ipynb"
MODES = ["mixed", "kvgate", "hardstop", "rerun", "lmofail", "allfail", "death"]

files = {}
for n in ("harness_e18_nb.py", "fake_vllm_e18_server.py"):
    files["harness/" + n] = (E18 / "src" / "harness" / n).read_bytes()
for n in ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"):
    files["data/" + n] = (DATA / n).read_bytes()
for p in sorted(TOK.iterdir()):
    files["tok/" + p.name] = p.read_bytes()
files["nb/" + NB] = (HERE.parent / NB).read_bytes()
manifest = {k: hashlib.sha256(v).hexdigest() for k, v in sorted(files.items())}
files["MANIFEST.json"] = json.dumps(manifest, indent=0).encode()
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for k in sorted(files):
        zi = zipfile.ZipInfo(k, date_time=(2026, 10, 27, 0, 0, 0))
        zi.compress_type = zipfile.ZIP_DEFLATED
        z.writestr(zi, files[k])
kit = buf.getvalue()
KIT_SHA = hashlib.sha256(kit).hexdigest()
KIT_NAME = f"q27x_e18_cpu_kit_{KIT_SHA[:12]}.bin"       # .bin: kept out of Kaggle's archive unpacking
OUT_DIR.mkdir(exist_ok=True)
for old in OUT_DIR.glob("q27x_e18_cpu_kit_*.bin"):
    old.unlink()
(OUT_DIR / KIT_NAME).write_bytes(kit)
(OUT_DIR / "dataset-metadata.json").write_text(json.dumps(
    {"title": "Q27 E18 CPU kit", "id": KIT_SLUG, "licenses": [{"name": "apache-2.0"}]}, indent=1), encoding="utf-8")

CODE = r'''# E18 CPU harness on Kaggle (pinned image = the GPU notebook's): the fake-server harness, all 6 scenarios
import hashlib, io, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
KIT_NAME, KIT_SHA = "''' + KIT_NAME + '''", "''' + KIT_SHA + '''"
NB = "''' + NB + '''"
MODES = ''' + json.dumps(MODES) + r'''
WH = "/kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"
OUT = Path("/kaggle/working"); K = Path("/tmp/e18_kit"); T0 = time.time()
REPORT = {"kit": KIT_NAME, "python": sys.version, "nb_sha": None, "harness": {}}


def save():
    (OUT / "cpu_harness_report.json").write_text(json.dumps(REPORT, indent=1), encoding="utf-8")


def find(name, sha):
    for pat in (name, "*/" + name, "*/*/" + name, "*/*/*/" + name, "*/*/*/*/" + name):
        for f in sorted(Path("/kaggle/input").glob(pat)):
            b = f.read_bytes()
            if hashlib.sha256(b).hexdigest() == sha:
                return b
    raise FileNotFoundError(f"{name} ({sha[:12]}) not under /kaggle/input: attach foysalemonshanto/q27-e18-cpu-kit")


shutil.rmtree(K, ignore_errors=True)
with zipfile.ZipFile(io.BytesIO(find(KIT_NAME, KIT_SHA))) as z:
    z.extractall(K)
man = json.loads((K / "MANIFEST.json").read_text())
bad = [k for k, h in man.items() if hashlib.sha256((K / k).read_bytes()).hexdigest() != h]
assert not bad, bad
REPORT["nb_sha"] = hashlib.sha256((K / "nb" / NB).read_bytes()).hexdigest()
print("kit OK:", len(man), "files |", sys.version.split()[0], "| notebook", REPORT["nb_sha"], flush=True)
# the GPU notebook's tokenizer pins (its CELL 2), from the S1 wheelhouse
r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--find-links", WH,
                    "transformers==4.57.6", "tokenizers==0.22.2", "huggingface-hub==0.36.2"], capture_output=True, text=True)
REPORT["pins"] = {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-800:]}
print("pins:", r.returncode, flush=True)
ENV = dict(os.environ, E18_DATA=str(K / "data"), E18_TOKDIR=str(K / "tok"), PYTHONIOENCODING="utf-8")
for mode in MODES:
    t = time.time()
    try:
        r = subprocess.run([sys.executable, str(K / "harness/harness_e18_nb.py"), str(K / "nb" / NB), mode],
                           cwd=K / "harness", env=ENV, capture_output=True, text=True, timeout=1800)
        out, rc = r.stdout + r.stderr, r.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = (e.stdout or "") if isinstance(e.stdout, str) else "", "timeout"
    (OUT / f"harness_{mode}.log").write_text(out, encoding="utf-8")
    lines = out.splitlines()
    REPORT["harness"][mode] = {"rc": rc, "sec": round(time.time() - t), "ok": rc == 0 and "ALL CHECKS PASS" in out,
                               "passes": sum(1 for x in lines if x.startswith("PASS")),
                               "summary": [x for x in lines if "ALL CHECKS PASS" in x or " failed in " in x],
                               "fails": [x[:1500] for x in lines if x.startswith("FAIL")],
                               "traceback_tail": out[-3000:] if "Traceback" in out else ""}
    save()
    print(mode, "->", REPORT["harness"][mode]["passes"], "pass", REPORT["harness"][mode]["summary"],
          REPORT["harness"][mode]["fails"][:6], flush=True)
REPORT["sec"] = round(time.time() - T0)
REPORT["all_ok"] = all(v["ok"] for v in REPORT["harness"].values())
save()
print("ALL OK" if REPORT["all_ok"] else "NOT OK", "| total", REPORT["sec"], "s", flush=True)
'''
compile(CODE, "<cpu harness>", "exec")
nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": [
          "# E18 CPU harness (Kaggle pinned image)\n\nThe fake-server harness of the E18 notebook (S1-60K on vLLM 0.32, 20 "
          "streams), all 6 scenarios on the same image as the GPU notebook. Kit: `" + KIT_SLUG + "` (" + KIT_NAME + ").\n"]},
                {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CODE.splitlines(True)}],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
KD = HERE / "cpu_kernel_e18"
KD.mkdir(exist_ok=True)
(KD / "q27-e18-cpu-harness.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
(KD / "kernel-metadata.json").write_text(json.dumps({
    "id": KERNEL_SLUG, "title": "Q27 E18 CPU Harness", "code_file": "q27-e18-cpu-harness.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
    "keywords": [], "dataset_sources": ["mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3", KIT_SLUG], "kernel_sources": [],
    "competition_sources": [], "model_sources": [], "docker_image": IMAGE, "machine_shape": "None"}, indent=2),
    encoding="utf-8")
print("kit", OUT_DIR / KIT_NAME, round(len(kit) / 1e6, 1), "MB,", len(manifest), "files")
print("notebook", NB, hashlib.sha256((HERE.parent / NB).read_bytes()).hexdigest()[:16])
