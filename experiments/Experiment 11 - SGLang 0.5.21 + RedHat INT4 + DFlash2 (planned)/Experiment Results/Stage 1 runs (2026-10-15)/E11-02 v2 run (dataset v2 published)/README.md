# SGLang 0.5.21 offline wheelhouse (built on Kaggle, CPython 3.12.13, CUDA 13 wheels)

Install (one command, offline, CPython 3.12 Linux x86_64):

    python /kaggle/input/<this dataset>/install_wheelhouse.py /tmp/sglang-venv

Do NOT run `pip install --find-links <this dir>` directly: Kaggle drops '+' from file names, so
flashinfer_jit_cache-0.6.18+cu130-...whl is stored as ...-0.6.18cu130-...whl and pip cannot match the lock.
install_wheelhouse.py restores the names from SHA256SUMS (checking their sha256), creates the venv, installs
requirements.lock with --no-deps and adds lib/libcudart.so + lib64 under the pip CUDA runtime. It prints
VENV_PYTHON, CUDA_HOME and PATH: start SGLang with them and fresh SGLANG_CACHE_DIR / CUDA_CACHE_PATH.
A venv (not --target + PYTHONPATH) so that .pth files run (nvidia-cutlass-dsl) and console scripts (ninja) are
on PATH. On Kaggle also symlink libcuda.so.1 as libcuda.so into a LIBRARY_PATH directory (FlashInfer JIT links -lcuda).

Made for ARC-AGI-2 Experiment 11 (RedHat Qwen3.8-27B INT4 + DFlash2 on 4x L4). The lock overrides cuda-tile
1.6.0rc5 -> 1.6.0 because rc5 is sdist-only; --no-deps keeps pip from re-checking that pin. nvcc/crt/nvvm/nvjitlink
are pinned to 13.0.88 (torch 2.13's cuda-toolkit 13.0.3 set; the Kaggle L4 driver supports CUDA 13.0).
BUILD_PROBE.json (imports + CPU nvcc toolchain) / BUILD_FLAGS.json record the CPU-side checks done at build time.
