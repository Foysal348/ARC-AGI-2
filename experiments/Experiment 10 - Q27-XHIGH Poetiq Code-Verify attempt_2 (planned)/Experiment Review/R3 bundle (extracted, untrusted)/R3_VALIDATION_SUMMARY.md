# R3 Validation Summary

Notebook: `Q27-XHIGH-v1_4_4_E10_86K_POETIQ_PROGRAM_SOLVER_R3.ipynb`
SHA-256: `4418ebbf5ccc1665f9dff9ee1e6c19a649e4bd23e913fa635bbfc4fb0a898853`

## Results

- Notebook schema: PASS (`nbformat.validate`).
- Source diff vs R2: exactly physical cells **5 and 7**.
- Base parity cells **1, 3, 6**: PASS.
- Base SHA-256: PASS.
- E10 CPU fake-engine baseline: PASS.
- R2 bugs regression: PASS.
- R3 approved changes adversarial regression: PASS.
- Generated-code stdout, NumPy/SciPy/OpenCV: PASS on local Linux.
- Test-invalid training-perfect refinement: PASS.
- One-expert history second-attempt diversity: PASS.
- Sandbox rejected outside read/write, read-only runtime writes, subprocess and network: PASS.
- Infinite loop terminated: PASS.
- Huge stdout and excessive memory requests: rejected in separate local probes.
- Kaggle's namespace/mount/seccomp availability: NOT VERIFIED.
- Real 4xL4 GPU Q27 model inference: NOT RUN.
- Known original `MIN_THINK_BUDGET=4096` boundary: UNCHANGED (not one of the three approved changes).
