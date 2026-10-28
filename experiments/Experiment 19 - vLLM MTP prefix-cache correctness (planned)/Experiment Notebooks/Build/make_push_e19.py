"""Writes (does NOT push) the three Kaggle push folders for E19 under D:/ARC AGI 2/_scratch/push_e19/:
  P1 (arms D, C) -> foysalemonshanto | P2 (arms E, A) -> esfersami50 | P3 (arms B, K) -> hosen42
Each folder: kernel-metadata.json + the notebook (renamed to the kernel slug). Inputs = S1-60K's: the competition, the
vLLM 0.19 wheelhouse dataset and the FP8 repack model; NvidiaL4, the pinned Python 3.12 image, internet off, private.
No credential is read here. Push later (only after the user's go) with: kaggle kernels push -p <folder>
usage: python make_push_e19.py"""
import hashlib
import json
import shutil
from pathlib import Path

NBDIR = Path(__file__).resolve().parent.parent
OUT = Path(r"D:/ARC AGI 2/_scratch/push_e19")
PLAN = [("Q27-XHIGH-E19_MTPPC_P1_D_C.ipynb", "foysalemonshanto", "q27-xhigh-e19-mtppc-p1-d-c", "Q27 XHIGH E19 MTPPC P1 D C"),
        ("Q27-XHIGH-E19_MTPPC_P2_E_A.ipynb", "esfersami50", "q27-xhigh-e19-mtppc-p2-e-a", "Q27 XHIGH E19 MTPPC P2 E A"),
        ("Q27-XHIGH-E19_MTPPC_P3_B_K.ipynb", "hosen42", "q27-xhigh-e19-mtppc-p3-b-k", "Q27 XHIGH E19 MTPPC P3 B K")]
for nb, user, slug, title in PLAN:
    d = OUT / f"{slug}__{user}"
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(NBDIR / nb, d / f"{slug}.ipynb")
    meta = {"id": f"{user}/{slug}", "title": title, "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": ["mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"], "kernel_sources": [],
            "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"],
            "docker_image": "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461",
            "machine_shape": "NvidiaL4"}
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    h = hashlib.sha256((d / f"{slug}.ipynb").read_bytes()).hexdigest()[:8]
    print(f"{d}  notebook sha {h}  ({nb})")
