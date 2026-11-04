# Experiment 26: Flash-Next klee100 3bpw low-bit checkpoint (planned)

**Goal:** load `klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP` (rev 600715b3) on Merlin v3.0.0 (TP2×PP2+EP, 4×L4).
- **Expert layout:** W3 g128 on layers 0-11 / 36-47, W2 g64 on layers 12-35, MTP W4, main projections W8.
- **Why:** to grow the KV pool from 149 to an estimated 463 blocks, and to measure speed and teacher-forced closeness against the W4 base.

Split out of Experiment 25 on 2026-10-31.

## Status (2026-10-31)
- Kaggle Model `foysalemonshanto/qwen38-flash-next-klee100-3bpw-mtp/PyTorch/autoround-3bpw-mtp-nople/1`: 33 files, 50.68 GB.
- CPU compatibility kit v2: ALL OK (CPU_COMPATIBILITY_PASS).
- GPU load notebook `Q27-XHIGH-E25_KLEE100_LOAD.ipynb` (sha 78b94c49):
  - units kit ALL OK (64/64);
  - real kit: every check true except the two known CPU-only limits (`--help`, argparse "Failed to infer device type");
  - GPU run needs the user's go.
- Reference dataset `foysalemonshanto/q27-e25-nb0-tf-ref` (25 files).

## GPT audits 2026-10-31
- **Three audits:** all NO-GO as-is, GO after the fixes.
- **13 verified items** (6 HIGH, 4 MED, 1 doc, 2 LOW) in `Experiment Review/KLEE100_LOAD_AUDIT_FIXES_2026-10-31.md`; the fixes are being applied.
- **Capacity correction:** 463 blocks is FP8-equivalent; BF16 KV gives ≈ 244 usable blocks (est.).

## Where things are
- **`Experiment Review/Audit Zips/`:** the GPT audit zips.
  - `E25_klee100_LOAD_audit_advice_FLAT_2026-10-31.zip` is the current audit + advice bundle (34 files).
- **Code, still in Experiment 25** (the load builder reuses the NB0 / NB1 builders by relative path):
  - notebook `Experiment 25 .../Experiment Notebooks/Q27-XHIGH-E25_KLEE100_LOAD.ipynb`;
  - builders and tests in `Experiment 25 .../Experiment Notebooks/Build/`: `build_e25_klee100_load.py`, `k100load_*`, `build_e25_klee100_cpu_kit.py`, `k100kit_*`, `build_e25_klee100_upload.py`.
- **Results:** `Experiment 25 .../Experiment Results/E25_klee100_cpu_kit_v1_2026-10-31.md` (kit v1, v2 and the load kits).
