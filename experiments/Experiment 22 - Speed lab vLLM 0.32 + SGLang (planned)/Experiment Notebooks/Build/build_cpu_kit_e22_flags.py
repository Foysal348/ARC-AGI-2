"""E22 flag kit: every E22 vLLM Speed Lab launch command (src/e22_flags.json, from src/e22_genflags.py) checked on a
Kaggle CPU kernel against the REAL vLLM 0.32.0 (installed from foysalemonshanto/vllm-0-32-0-cu13-wheelhouse into a venv,
as the notebook does): argparse + AsyncEngineArgs (E21's e21_flagcheck.py) and the JSON / choice fields argparse does not
validate (src/e22_fieldcheck.py: use_local_argmax_reduction, cudagraph_mode, mamba_ssm_cache_dtype bfloat16,
prefix_match_unit, enable_mamba_shared_prefix_checkpoint).
Dataset foysalemonshanto/q27-e22-flags-kit, kernel foysalemonshanto/q27-e22-cpu-flags (Build/cpu_kernel_e22_flags/)."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
E22 = HERE.parents[1]
E21SRC = Path(r"D:/ARC AGI 2/experiments/Experiment 21 - Duck config B on vLLM 0.19 + 0.32 (planned)/src")
KIT_SLUG = "foysalemonshanto/q27-e22-flags-kit"
KERNEL_SLUG = "foysalemonshanto/q27-e22-cpu-flags"
WH032_SLUG = "foysalemonshanto/vllm-0-32-0-cu13-wheelhouse"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"

files = {"e21_flagcheck.py": (E21SRC / "e21_flagcheck.py").read_bytes(),
         "e22_fieldcheck.py": (E22 / "src/e22_fieldcheck.py").read_bytes(),
         "e22_flags.json": (E22 / "src/e22_flags.json").read_bytes()}
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for k in sorted(files):
        z.writestr(zipfile.ZipInfo(k, date_time=(2026, 10, 29, 0, 0, 0)), files[k])
kit = buf.getvalue()
KIT_SHA = hashlib.sha256(kit).hexdigest()
KIT_NAME = f"q27x_e22_flags_kit_{KIT_SHA[:12]}.bin"
OUT = HERE / "cpu_kit_e22_flags"
OUT.mkdir(exist_ok=True)
for old in OUT.glob("q27x_e22_flags_kit_*.bin"):
    old.unlink()
(OUT / KIT_NAME).write_bytes(kit)
(OUT / "dataset-metadata.json").write_text(json.dumps({"title": "Q27 E22 flags kit", "id": KIT_SLUG,
                                                       "licenses": [{"name": "apache-2.0"}]}, indent=1), encoding="utf-8")
CODE = r'''# E22 flag kit: the E22 vLLM Speed Lab launch commands against the real vLLM 0.32.0 (CPU, pinned image)
import hashlib, io, json, subprocess, sys, time, zipfile
from pathlib import Path
KIT_NAME, KIT_SHA = "''' + KIT_NAME + '''", "''' + KIT_SHA + '''"
WH032 = "/kaggle/input/datasets/''' + WH032_SLUG + r'''"
OUT, K, T0 = Path("/kaggle/working"), Path("/tmp/e22_flags"), time.time()
REPORT = {"kit": KIT_NAME, "python": sys.version}
f = next(p for pat in (KIT_NAME, "*/" + KIT_NAME, "*/*/" + KIT_NAME, "*/*/*/" + KIT_NAME) for p in Path("/kaggle/input").glob(pat))
b = f.read_bytes(); assert hashlib.sha256(b).hexdigest() == KIT_SHA
zipfile.ZipFile(io.BytesIO(b)).extractall(K)
inst = next(iter(sorted(Path(WH032).rglob("install_wheelhouse.py"))), None)
assert inst is not None, "attach " + WH032
r = subprocess.run([sys.executable, str(inst), "/tmp/vllm-venv"], capture_output=True, text=True, timeout=2400)
REPORT["install"] = {"rc": r.returncode, "sec": round(time.time() - T0), "tail": (r.stdout + r.stderr)[-800:]}
vpy = next((l.split("=", 1)[1] for l in r.stdout.splitlines() if l.startswith("VENV_PYTHON=")), "/tmp/vllm-venv/bin/python")
r = subprocess.run([vpy, str(K / "e21_flagcheck.py"), str(K / "e22_flags.json"), "0.32.0"], capture_output=True, text=True, timeout=1800)
REPORT["flagcheck"] = {"rc": r.returncode, "out": (r.stdout + r.stderr)[-8000:],
                       "summary": next((json.loads(l[10:]) for l in r.stdout.splitlines() if l.startswith("FLAGCHECK ")), None)}
print("flagcheck:", json.dumps(REPORT["flagcheck"]["summary"])[:3000], flush=True)
r = subprocess.run([vpy, str(K / "e22_fieldcheck.py"), str(K / "e22_flags.json")], capture_output=True, text=True, timeout=900)
REPORT["fieldcheck"] = next((json.loads(l[11:]) for l in r.stdout.splitlines() if l.startswith("FIELDCHECK ")),
                            {"rc": r.returncode, "tail": (r.stdout + r.stderr)[-3000:]})
print("fieldcheck:", json.dumps(REPORT["fieldcheck"])[:4000], flush=True)
REPORT["all_ok"] = bool((REPORT["flagcheck"]["summary"] or {}).get("all_ok")) and bool(REPORT["fieldcheck"].get("all_ok"))
REPORT["sec"] = round(time.time() - T0)
(OUT / "e22_flags_report.json").write_text(json.dumps(REPORT, indent=1), encoding="utf-8")
print("ALL OK" if REPORT["all_ok"] else "NOT OK", "| total", REPORT["sec"], "s", flush=True)
'''
compile(CODE, "<flags kit>", "exec")
nbj = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": CODE.splitlines(True)}],
       "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
KD = HERE / "cpu_kernel_e22_flags"
KD.mkdir(exist_ok=True)
(KD / "q27-e22-cpu-flags.ipynb").write_text(json.dumps(nbj, indent=1), encoding="utf-8")
(KD / "kernel-metadata.json").write_text(json.dumps({
    "id": KERNEL_SLUG, "title": "Q27 E22 CPU Flags", "code_file": "q27-e22-cpu-flags.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
    "keywords": [], "dataset_sources": [WH032_SLUG, KIT_SLUG], "kernel_sources": [], "competition_sources": [],
    "model_sources": [], "docker_image": IMAGE, "machine_shape": "None"}, indent=2), encoding="utf-8")
print("kit", OUT / KIT_NAME, len(kit), "bytes | kernel", KD)
