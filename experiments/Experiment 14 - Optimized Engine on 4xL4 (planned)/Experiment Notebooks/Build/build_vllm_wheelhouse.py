"""vLLM 0.32.0 (latest on PyPI, released 2026-10-18) offline wheelhouse -> Kaggle dataset (user 2026-10-25: "একদম লেটেস্ট
টা বানাও", for the vLLM engine-config notebook). Same proven shape as E11-02 (build_e11_assets.py notebook 2): a Kaggle CPU
notebook with Internet ON downloads a pinned lock, tests the offline install in a venv through the dataset's own installer
on a Kaggle-style copy ('+' dropped from file names), runs CPU probes and the launch-flag check, writes checksums + README,
then publishes the dataset with the attached Secrets (secrets are readable only in editor runs: Save & Run All).

Lock (vllm_0_32_0_requirements.lock next to this file, 199 pins): resolved off-Kaggle with
    uv pip compile req.in --python-version 3.12 --python-platform x86_64-manylinux_2_35 --override overrides.txt
        --only-binary :all:        (req.in = vllm==0.32.0; overrides = nvcc / crt / nvvm / nvjitlink 13.0.88)
plus flashinfer-jit-cache==0.7.0.post1+cu130 (precompiled kernels, flashinfer.ai cu130 index). Every pin has a Linux
x86_64 cp312 / abi3 / py3 wheel (wheel-only resolve). torch 2.13.0 + CUDA runtime 13.0.96 + nvcc 13.0.88 + cuda-bindings
13.4.4 = the same set as the SGLang 0.5.21 wheelhouse that runs on the Kaggle L4 driver (CUDA 13.0). glibc 2.35 platform
because llguidance 1.7.x ships only manylinux_2_31 wheels.
usage: python build_vllm_wheelhouse.py  -> ../E14_V1_vllm_0_32_0_wheelhouse_build_on_KAGGLE.ipynb"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent
LOCK = (HERE / "vllm_0_32_0_requirements.lock").read_text(encoding="utf-8")
LOCK = "".join(l + "\n" for l in LOCK.splitlines() if "==" in l and not l.lstrip().startswith("#"))
PINS = dict(l.split("==", 1) for l in LOCK.splitlines())
assert len(PINS) == 199 and PINS["vllm"] == "0.32.0" and PINS["torch"] == "2.13.0", (len(PINS), PINS.get("vllm"))
assert all(PINS[p] == "13.0.88" for p in ("nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-nvvm", "nvidia-nvjitlink"))
assert PINS["nvidia-cuda-runtime"] == "13.0.96" and PINS["flashinfer-jit-cache"] == "0.7.0.post1+cu130"
NB_NAME = "E14_V1_vllm_0_32_0_wheelhouse_build_on_KAGGLE.ipynb"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


W0 = """# E14-V1 · Build the vLLM 0.32.0 offline wheelhouse → Kaggle dataset (runs ON Kaggle)

**Settings:** Accelerator **None** (CPU, no GPU quota) · Internet **ON** · Add-ons → Secrets → attach `KAGGLE_USERNAME`
and `KAGGLE_KEY` · start with **Save Version → Save & Run All** from the editor (secrets are not readable in API-started runs).

**What:** every wheel of vLLM 0.32.0 (latest, 2026-10-18) for this Kaggle environment (CPython 3.12, glibc 2.35, x86_64):
torch 2.13.0 (CUDA 13 runtime wheels), flashinfer-python 0.7.0.post1 + flashinfer-jit-cache 0.7.0.post1+cu130 (precompiled
kernels, no offline JIT stall), transformers 5.19.0, compressed-tensors 0.17.0 (the RedHat INT4 checkpoint) - 199 wheels.

**Why a pinned lock:** resolved off-Kaggle with `uv pip compile --only-binary :all:` for Linux / CPython 3.12 / glibc 2.35,
so every pin is a wheel; the CUDA compiler wheels are pinned to **13.0.88** (torch 2.13's cuda-toolkit 13.0.3 set, the same
as our SGLang 0.5.21 wheelhouse that runs on the Kaggle L4 driver = CUDA 13.0). Installed with `--no-deps`.

**Install in the GPU notebook (a separate venv, the image's own packages stay untouched):**

    python /kaggle/input/<this dataset>/install_wheelhouse.py /tmp/vllm-venv

It restores the '+' Kaggle drops from file names (sha256 checked), creates the venv, installs `requirements.lock` with
`--no-deps`, adds `lib/libcudart.so` + `lib64` under the pip CUDA runtime, and prints `VENV_PYTHON`, `CUDA_HOME`, `PATH`.
Run vLLM as a server process (`VENV_PYTHON -m vllm.entrypoints.openai.api_server ...`) so it can be stopped cleanly.
"""

W1 = f'''
import datetime, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

VLLM_VERSION = "0.32.0"
PUBLIC = False
DATASET_SLUG = "vllm-0-32-0-cu13-wheelhouse"
DATASET_TITLE = "vLLM 0.32.0 cu13 wheelhouse (py3.12)"
VERSION_NOTES = "v1: vLLM 0.32.0 (2026-10-18), torch 2.13.0 cu130, flashinfer 0.7.0.post1 + jit-cache cu130, 199 wheels"
WH = Path("/kaggle/working/vllm-0.32.0-wheelhouse")       # this is what gets uploaded
VENV = Path("/tmp/vllm-venv")                              # offline install test (venv), deleted before upload
VPY = VENV / "bin" / "python"
FLASHINFER_INDEX = "https://flashinfer.ai/whl/cu130/"
LOCK = """{LOCK}"""
WH.mkdir(parents=True, exist_ok=True)
(WH / "requirements.lock").write_text(LOCK, encoding="utf-8")
PINS = dict(l.strip().split("==", 1) for l in LOCK.splitlines() if "==" in l)
print("python", sys.version.split()[0], "| pins", len(PINS))
r = subprocess.run(["ldd", "--version"], capture_output=True, text=True)
print("glibc:", (r.stdout or r.stderr).splitlines()[0] if (r.stdout or r.stderr) else "?")
for root in (Path("/kaggle/working"), Path("/tmp")):
    print(f"  free {{str(root):16}} {{shutil.disk_usage(root).free / 1024**3:7.1f}} GiB")
assert sys.version_info[:2] == (3, 12), "must run on the same Python as the ARC notebook (3.12)"
assert len(PINS) == 199 and PINS["vllm"] == VLLM_VERSION and PINS["torch"] == "2.13.0"
assert PINS["nvidia-cuda-nvcc"] == PINS["nvidia-nvvm"] == PINS["nvidia-nvjitlink"] == "13.0.88"
r = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True)
print(r.stdout.strip())
assert tuple(int(x) for x in r.stdout.split()[1].split(".")[:2]) >= (22, 3), "pip >= 22.3 is needed for pip --python <venv>"
'''

W2 = r'''
# 1. Download every pinned wheel (binary only, no dependency resolution: the lock is already complete)
t0 = time.time()
r = subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:", "-r", str(WH / "requirements.lock"),
                    "-d", str(WH), "--extra-index-url", FLASHINFER_INDEX, "--disable-pip-version-check"],
                   capture_output=True, text=True)
print(r.stdout[-2500:])
if r.returncode != 0:
    print(r.stderr[-5000:])
    raise SystemExit("pip download failed - paste this output back")
from packaging.utils import parse_wheel_filename, canonicalize_name
have = {}
for w in sorted(WH.glob("*.whl")):
    name, ver, _b, _t = parse_wheel_filename(w.name)
    have.setdefault(canonicalize_name(name), []).append(str(ver))
want = {canonicalize_name(k): v for k, v in PINS.items()}
missing = sorted(k for k in want if k not in have)
dupes = {k: v for k, v in have.items() if len(v) > 1}
wrongv = {k: (have[k], want[k]) for k in want if k in have and want[k] not in have[k]}
size = sum(p.stat().st_size for p in WH.glob("*.whl"))
print(f"wheels: {sum(len(v) for v in have.values())} | {size / 1024**3:.2f} GiB | missing {missing} | duplicates {dupes} | "
      f"version mismatches {wrongv} | {time.time() - t0:.0f}s")
assert not missing and not dupes and not wrongv, "the wheelhouse does not match the lock"
'''

W3 = r'''
# 2. Offline install test through the dataset's own installer, on a copy laid out the way Kaggle stores the files
#    (Kaggle drops '+' from file names: flashinfer_jit_cache-0.7.0.post1+cu130-... is stored as ...-0.7.0.post1cu130-...).
INSTALLER_SRC = r"""
# install_wheelhouse.py - offline install of this vLLM 0.32.0 wheelhouse into a fresh venv (CPython 3.12, Linux x86_64)
#
#     python install_wheelhouse.py [VENV_DIR]          (default /tmp/vllm-venv)
#
# Do NOT run `pip install --find-links <this folder>` directly: Kaggle drops '+' from dataset file names, so the pin
# flashinfer-jit-cache==0.7.0.post1+cu130 cannot be matched. This script: 1. links every wheel under its original name
# (SHA256SUMS keeps the names) into VENV_DIR-wheels, checking the sha256 of every renamed wheel; 2. creates the venv
# without pip (the system pip installs into it with `pip --python`), so .pth files run and console scripts land in
# VENV/bin; 3. installs requirements.lock with --no-deps; 4. adds lib/libcudart.so and lib64 under the pip CUDA runtime
# (site-packages/nvidia/cu13). It prints VENV_PYTHON / CUDA_HOME / PATH. On Kaggle also make -lcuda linkable for JIT
# (only libcuda.so.1 exists): symlink it as libcuda.so into a directory on LIBRARY_PATH.
import hashlib, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
venv = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/vllm-venv")
links = Path(os.environ.get("WHEELHOUSE_LINKS", str(venv) + "-wheels"))
links.mkdir(parents=True, exist_ok=True)
restored = []
for line in (HERE / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    digest, name = line.split("  ", 1)
    name = name.strip()
    if not name.endswith(".whl"):
        continue
    src = HERE / name if (HERE / name).exists() else HERE / name.replace("+", "")
    if not src.exists():
        sys.exit(f"wheel missing from this folder: {name}")
    if src.name != name:
        h = hashlib.sha256()
        with open(src, "rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                h.update(block)
        if h.hexdigest() != digest:
            sys.exit(f"sha256 mismatch: {src.name}")
        restored.append(name)
    dst = links / name
    if not dst.exists():
        dst.symlink_to(src)
vpy = venv / "bin" / "python"
if not vpy.exists():
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True)
subprocess.run([sys.executable, "-m", "pip", "--python", str(vpy), "install", "--no-index", "--find-links", str(links),
                "--requirement", str(HERE / "requirements.lock"), "--no-deps", "--only-binary", ":all:", "--no-compile",
                "--disable-pip-version-check", "--no-warn-conflicts", "-q"], check=True)
site = Path(subprocess.run([str(vpy), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                           capture_output=True, text=True, check=True).stdout.strip())
cuda = site / "nvidia" / "cu13"
if (cuda / "lib").exists() and not (cuda / "lib" / "libcudart.so").exists():
    (cuda / "lib" / "libcudart.so").symlink_to("libcudart.so.13")
if (cuda / "lib").exists() and not (cuda / "lib64").exists():
    (cuda / "lib64").symlink_to("lib")
print("restored wheel names:", restored or "none")
print("VENV_PYTHON=" + str(vpy))
print("CUDA_HOME=" + str(cuda))
print("PATH=" + str(venv / "bin") + ":" + str(cuda / "bin") + ":$PATH")
"""
(WH / "install_wheelhouse.py").write_text(INSTALLER_SRC.lstrip("\n"), encoding="utf-8")


def sha256(p, chunk=16 * 1024 * 1024):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


t0 = time.time()
WHEEL_SUMS = {w.name: sha256(w) for w in sorted(WH.glob("*.whl"))}
print(f"sha256 of {len(WHEEL_SUMS)} wheels in {time.time() - t0:.0f}s")
SIM = Path("/tmp/kaggle-layout-sim")
shutil.rmtree(SIM, ignore_errors=True)
SIM.mkdir(parents=True)
for _n in WHEEL_SUMS:
    (SIM / _n.replace("+", "")).symlink_to(WH / _n)
(SIM / "SHA256SUMS").write_text("".join(f"{h}  {n}\n" for n, h in WHEEL_SUMS.items()), encoding="utf-8")
for _f in ("requirements.lock", "install_wheelhouse.py"):
    shutil.copy(WH / _f, SIM / _f)
print("names changed in the simulated Kaggle layout:", [n for n in WHEEL_SUMS if "+" in n])
shutil.rmtree(VENV, ignore_errors=True)
shutil.rmtree(Path(str(VENV) + "-wheels"), ignore_errors=True)
t0 = time.time()
r = subprocess.run([sys.executable, str(SIM / "install_wheelhouse.py"), str(VENV)], capture_output=True, text=True)
print(r.stdout[-2500:])
if r.returncode != 0:
    print(r.stderr[-4000:])
    raise SystemExit("installer failed on the Kaggle-style layout - paste this output back")
assert "flashinfer_jit_cache-0.7.0.post1+cu130" in r.stdout, "the renamed wheel was not restored"
SITE = Path(subprocess.run([str(VPY), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                           capture_output=True, text=True, check=True).stdout.strip())
CUDA_HOME = SITE / "nvidia" / "cu13"
inst = sum(p.stat().st_size for p in VENV.rglob("*") if p.is_file())
print(f"offline install OK in {time.time() - t0:.0f}s | installed size {inst / 1024**3:.1f} GiB | site {SITE}")
print(".pth files:", sorted(p.name for p in SITE.glob("*.pth")))
'''

W4 = r'''
# 3. CPU probes (no GPU here): imports, transformers reads the RedHat INT4 config + tokenizer, the vLLM registry knows
#    the Qwen3.5/3.8 architectures and the MTP head. Recorded in BUILD_PROBE.json; the GPU notebook re-checks on GPU.
from huggingface_hub import hf_hub_download
SMALL = Path("/tmp/redhat-int4-small"); SMALL.mkdir(exist_ok=True)
for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
    try:
        hf_hub_download("RedHatAI/Qwen3.8-27B-INT4", f, revision="91bd022d5b49442a868bc35008f6c21e1860edfa", local_dir=str(SMALL))
    except Exception as e:
        print("hf download", f, repr(e)[:200])
PROBE = r"""
import json, sys, importlib
out = {}
def step(name, fn):
    try:
        out[name] = fn()
    except Exception as e:
        out[name] = f"ERROR {type(e).__name__}: {str(e)[:300]}"
step("torch", lambda: (lambda t: [t.__version__, t.version.cuda])(importlib.import_module("torch")))
step("transformers", lambda: importlib.import_module("transformers").__version__)
step("vllm", lambda: importlib.import_module("vllm").__version__)
step("flashinfer", lambda: importlib.import_module("flashinfer").__version__)
step("compressed_tensors", lambda: importlib.import_module("compressed_tensors").__version__)
def cfg():
    from transformers import AutoConfig, AutoTokenizer
    c = AutoConfig.from_pretrained(sys.argv[1])
    t = AutoTokenizer.from_pretrained(sys.argv[1])
    return {"arch": getattr(c, "architectures", None), "model_type": c.model_type, "vocab": len(t)}
step("redhat_config_tokenizer", cfg)
def reg():
    from vllm.model_executor.models.registry import ModelRegistry
    archs = set(ModelRegistry.get_supported_archs())
    want = ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM", "Qwen3_5MTP", "Qwen3NextForCausalLM", "Qwen3NextMTP")
    return {a: (a in archs) for a in want} | {"n_archs": len(archs), "qwen3_5_like": sorted(a for a in archs if "Qwen3_5" in a)[:12]}
step("vllm_registry", reg)
print("PROBE_JSON " + json.dumps(out))
"""
env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_HOME", "CUDA_PATH")}
env.update({"PATH": f"{VENV / 'bin'}:{CUDA_HOME / 'bin'}:{env.get('PATH', '')}", "CUDA_HOME": str(CUDA_HOME),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "VLLM_TARGET_DEVICE": "cuda"})
r = subprocess.run([str(VPY), "-c", PROBE, str(SMALL)], env=env, capture_output=True, text=True, timeout=1800)
line = next((l for l in r.stdout.splitlines() if l.startswith("PROBE_JSON ")), None)
PROBE_RESULT = json.loads(line[11:]) if line else {"crash": (r.stdout + r.stderr)[-3000:]}
TOOL = {"files": {rel: (CUDA_HOME / rel).exists() for rel in ("bin/nvcc", "bin/ptxas", "include/cuda_runtime.h",
                                                               "lib/libcudart.so.13", "lib/libcudart.so", "lib64")}}
p = subprocess.run([str(CUDA_HOME / "bin" / "nvcc"), "--version"], env=env, capture_output=True, text=True)
TOOL["nvcc_version"] = {"rc": p.returncode, "out": (p.stdout + p.stderr).strip()[-300:]}
PROBE_RESULT["toolchain"] = TOOL
print(json.dumps(PROBE_RESULT, indent=1))
(WH / "BUILD_PROBE.json").write_text(json.dumps(PROBE_RESULT, indent=2), encoding="utf-8")
_bad = [k for k, v in PROBE_RESULT.items() if k == "crash" or (isinstance(v, str) and v.startswith("ERROR"))]
_bad += [f"missing {k}" for k, ok in TOOL["files"].items() if not ok]
print("\nPROBE SUMMARY - not OK on this CPU machine (recorded, reviewed before the GPU notebook):", _bad or "none")
'''

W5 = r'''
# 4. The engine-config notebook's launch flags must exist in this build (help text of the OpenAI server)
FLAGS = ["--tensor-parallel-size", "--data-parallel-size", "--quantization", "--kv-cache-dtype", "--max-num-seqs",
         "--max-model-len", "--gpu-memory-utilization", "--enable-prefix-caching", "--speculative-config",
         "--compilation-config", "--max-num-batched-tokens", "--enable-chunked-prefill", "--mamba-ssm-cache-dtype",
         "--reasoning-parser", "--disable-log-requests", "--seed"]
r = subprocess.run([str(VPY), "-m", "vllm.entrypoints.openai.api_server", "--help=all"], env=env, capture_output=True,
                   text=True, timeout=1800)
helptext = r.stdout + r.stderr
if "usage" not in helptext.lower():
    r = subprocess.run([str(VPY), "-m", "vllm.entrypoints.openai.api_server", "--help"], env=env, capture_output=True,
                       text=True, timeout=1800)
    helptext = r.stdout + r.stderr
missing = [f for f in FLAGS if f not in helptext]
print("usage printed:", "usage" in helptext.lower(), "| flags not listed:", missing or "none")
if "usage" not in helptext.lower():
    print(helptext[-3000:])
(WH / "BUILD_FLAGS.json").write_text(json.dumps({"usage_printed": "usage" in helptext.lower(), "missing": missing}, indent=2),
                                     encoding="utf-8")
'''

W6 = r'''
# 5. Checksums, build info, README (then free the test install)
info = {
    "vllm": VLLM_VERSION, "python": sys.version.split()[0],
    "glibc": subprocess.run(["ldd", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
    "built_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z", "built_on": "Kaggle",
    "wheel_count": len(list(WH.glob("*.whl"))),
    "note": "lock resolved with uv --only-binary for cp312 / glibc 2.35; nvcc/crt/nvvm/nvjitlink pinned 13.0.88 (torch 2.13 "
            "cuda-toolkit 13.0.3 set, Kaggle L4 driver = CUDA 13.0); install with install_wheelhouse.py (venv, --no-deps)",
    "dataset_version_notes": VERSION_NOTES,
    "stamp": f"vllm=={VLLM_VERSION} torch=={PINS['torch']} flashinfer=={PINS['flashinfer-python']} "
             f"jit-cache=={PINS['flashinfer-jit-cache']} transformers=={PINS['transformers']} "
             f"compressed-tensors=={PINS['compressed-tensors']}",
    "probe": PROBE_RESULT, "pins": PINS,
}
(WH / "BUILD_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
(WH / "README.md").write_text(
    f"# vLLM {VLLM_VERSION} offline wheelhouse (built on Kaggle, CPython {info['python']}, CUDA 13 wheels)\n\n"
    "Install (offline, CPython 3.12 Linux x86_64):\n\n"
    "    python /kaggle/input/<this dataset>/install_wheelhouse.py /tmp/vllm-venv\n\n"
    "Do NOT run `pip install --find-links <this dir>` directly: Kaggle drops '+' from file names (flashinfer-jit-cache\n"
    "0.7.0.post1+cu130). install_wheelhouse.py restores the names from SHA256SUMS (sha256 checked), creates the venv,\n"
    "installs requirements.lock with --no-deps and adds lib/libcudart.so + lib64 under the pip CUDA runtime.\n"
    "Made for ARC-AGI-2 Experiment 14 (engine configs on 4x L4: RedHat Qwen3.8-27B INT4 / FP8, MTP).\n",
    encoding="utf-8")
files = sorted(p for p in WH.iterdir() if p.is_file() and p.name != "SHA256SUMS")
(WH / "SHA256SUMS").write_text("".join(f"{WHEEL_SUMS.get(p.name) or sha256(p)}  {p.name}\n" for p in files), encoding="utf-8")
for _d in (VENV, Path(str(VENV) + "-wheels"), SIM):
    shutil.rmtree(_d, ignore_errors=True)
print(info["stamp"])
print("upload size:", round(sum(p.stat().st_size for p in WH.iterdir()) / 1024**3, 2), "GiB")
'''

W7 = r'''
# 6. Publish as a dataset (Kaggle -> Kaggle, uses the attached Secrets)
from kaggle_secrets import UserSecretsClient

s = UserSecretsClient()
KAGGLE_USERNAME = s.get_secret("KAGGLE_USERNAME").strip()
os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = s.get_secret("KAGGLE_KEY").strip()
cred = Path.home() / ".kaggle" / "kaggle.json"
cred.parent.mkdir(parents=True, exist_ok=True)
cred.write_text(json.dumps({"username": KAGGLE_USERNAME, "key": os.environ["KAGGLE_KEY"]}))
cred.chmod(0o600)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle==2.2.4"], check=True)
meta = {"title": DATASET_TITLE, "id": f"{KAGGLE_USERNAME}/{DATASET_SLUG}", "licenses": [{"name": "other"}]}
(WH / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print(json.dumps(meta, indent=1))
REF = f"{KAGGLE_USERNAME}/{DATASET_SLUG}"
_st = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "status", REF], capture_output=True, text=True)
EXISTS = _st.returncode == 0 and "ready" in (_st.stdout + _st.stderr).lower()
if EXISTS:
    cmd = [sys.executable, "-m", "kaggle", "datasets", "version", "-p", str(WH), "-m", VERSION_NOTES, "-r", "skip"]
else:
    cmd = [sys.executable, "-m", "kaggle", "datasets", "create", "-p", str(WH), "-r", "skip"] + (["--public"] if PUBLIC else [])
print("dataset exists:", EXISTS, "->", " ".join(cmd[3:5]))
r = subprocess.run(cmd, text=True)
print("exit code:", r.returncode)
# E11-02 run 1: the create call printed a non-JSON reply and exited 1, yet the dataset appeared later: wait for the files
t0 = time.time()
listed = False
while time.time() - t0 < 25 * 60:
    _ls = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "files", REF, "--page-size", "300"],
                         capture_output=True, text=True)
    if "install_wheelhouse.py" in _ls.stdout:
        listed = True
        break
    time.sleep(30)
print(f"dataset listed: {listed} after {time.time() - t0:.0f}s")
subprocess.run([sys.executable, "-m", "kaggle", "datasets", "status", REF])
if not listed:
    raise SystemExit("the dataset is not visible after 25 min - paste this output back")
print("attach:", REF, "| mount: /kaggle/input/datasets/" + REF)
'''

for name, src in (("W1", W1), ("W2", W2), ("W3", W3), ("W4", W4), ("W5", W5), ("W6", W6), ("W7", W7)):
    compile(src, name, "exec")
cells = [md(W0), code(W1), md("## 1. Download every pinned wheel"), code(W2), md("## 2. Offline install test"), code(W3),
         md("## 3. CPU probes"), code(W4), md("## 4. Launch-flag check"), code(W5), md("## 5. Checksums, README"), code(W6),
         md("## 6. Publish the dataset"), code(W7)]
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
(OUTDIR / NB_NAME).write_bytes(data)
print("BUILT", NB_NAME, len(data), "bytes sha256", hashlib.sha256(data).hexdigest()[:16], "| pins", len(PINS))
