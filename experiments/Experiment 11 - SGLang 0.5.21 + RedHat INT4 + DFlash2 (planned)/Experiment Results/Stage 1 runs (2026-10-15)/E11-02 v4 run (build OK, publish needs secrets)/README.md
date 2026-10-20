# SGLang 0.5.21 offline wheelhouse (built on Kaggle, CPython 3.12.13, CUDA 13 wheels)

    python -m venv --without-pip /tmp/sglang-venv
    python -m pip --python /tmp/sglang-venv/bin/python install --no-index --find-links <this dir> \
        -r requirements.lock --no-deps --only-binary :all:

Run with /tmp/sglang-venv/bin/python, PATH=/tmp/sglang-venv/bin:$CUDA_HOME/bin:$PATH and
CUDA_HOME=<venv site-packages>/nvidia/cu13. A venv (not --target + PYTHONPATH) so that .pth files run
(nvidia-cutlass-dsl) and console scripts (ninja) are on PATH.

Made for ARC-AGI-2 Experiment 11 (RedHat Qwen3.8-27B INT4 + DFlash2 on 4x L4). The lock overrides cuda-tile
1.6.0rc5 -> 1.6.0 because rc5 is sdist-only; --no-deps keeps pip from re-checking that pin. nvcc/crt/nvvm/nvjitlink
are pinned to 13.0.88 (torch 2.13's cuda-toolkit 13.0.3 set; the Kaggle L4 driver supports CUDA 13.0).
BUILD_PROBE.json (imports + CPU nvcc toolchain) / BUILD_FLAGS.json record the CPU-side checks done at build time.
