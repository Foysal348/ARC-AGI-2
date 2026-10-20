# install_wheelhouse.py - offline install of this SGLang 0.5.21 wheelhouse into a fresh venv (CPython 3.12, Linux x86_64)
#
#     python install_wheelhouse.py [VENV_DIR]          (default /tmp/sglang-venv)
#
# Do NOT run `pip install --find-links <this folder>` directly: Kaggle drops '+' from dataset file names, so
# flashinfer_jit_cache-0.6.18+cu130-cp39-abi3-manylinux_2_28_x86_64.whl is stored as ...-0.6.18cu130-... and pip cannot
# match it to the lock pin. This script:
#   1. links every wheel under its original name (SHA256SUMS keeps the names) into VENV_DIR-wheels, and checks the
#      sha256 of every wheel whose name it had to restore;
#   2. creates the venv without pip (the system pip installs into it with `pip --python`). A venv, not --target +
#      PYTHONPATH, so .pth files run (nvidia-cutlass-dsl is only a .pth) and console scripts (ninja) land in VENV/bin;
#   3. installs requirements.lock with --no-deps (the lock overrides sglang's sdist-only cuda-tile==1.6.0rc5 pin);
#   4. adds lib/libcudart.so and lib64 under the pip CUDA runtime (site-packages/nvidia/cu13): JIT linking uses -lcudart.
# It prints the VENV_PYTHON / CUDA_HOME / PATH to use. On Kaggle also make -lcuda linkable for FlashInfer's JIT (only
# libcuda.so.1 exists there): symlink it as libcuda.so into a directory on LIBRARY_PATH.
import hashlib, os, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
venv = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sglang-venv")
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
if not (cuda / "lib" / "libcudart.so").exists():
    (cuda / "lib" / "libcudart.so").symlink_to("libcudart.so.13")
if not (cuda / "lib64").exists():
    (cuda / "lib64").symlink_to("lib")
print("restored wheel names:", restored or "none")
print("VENV_PYTHON=" + str(vpy))
print("CUDA_HOME=" + str(cuda))
print("PATH=" + str(venv / "bin") + ":" + str(cuda / "bin") + ":$PATH")
