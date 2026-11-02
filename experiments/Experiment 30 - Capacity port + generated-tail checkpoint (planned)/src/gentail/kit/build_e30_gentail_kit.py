"""Builds the E30 gentail Kaggle CPU kit (GPU_TEST_SPEC_gentail.md section 0): one notebook + kernel-metadata.json in
D:/ARC AGI 2/_scratch/push_e25/q27-e30-gentail-kit__foysalemonshanto/ (the NBDF K100 real kit's metadata fields, minus the
GPU-only Model sources). Run: python -I build_e30_gentail_kit.py"""
import base64
import hashlib
import json
import lzma
from pathlib import Path

HERE = Path(__file__).resolve().parent
G = HERE.parent
DUCK = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Results/prep_0gpu/duck_timeline")
K3V2 = Path(r"D:/ARC AGI 2/_scratch/e25_gpu_out_k3_v2/q27-xhigh-e25-klee100-load-k3only/k100/k100_results.json")
OUT = Path(r"D:/ARC AGI 2/_scratch/push_e25/q27-e30-gentail-kit__foysalemonshanto")
OWNER, SLUG = "foysalemonshanto", "q27-e30-gentail-kit"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
MODELS = ["foysalemonshanto/e23-merlin-env/Other/v1/1",                                         # merlin_offline_tar.bin
          "foysalemonshanto/qwen38-flash-next-klee100-3bpw-mtp/PyTorch/autoround-3bpw-mtp-nople/1"]   # config.json (R9 layer names)
PIN = 7933526016          # the K3_ONLY v2 GPU record's pin (480 blocks), the NBDF K100 kit's R9 q_pin


def blob(files):
    return base64.b85encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")


def lf(p):
    t = Path(p).read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "\r" not in t, p
    return t


def main():
    k3 = json.loads(K3V2.read_text(encoding="utf-8"))["sections"]["K3"]["k100_K3_fp8_mtp3_pin739_r0"]
    assert k3["kv_bytes"] == PIN and k3["capacity"]["pool_blocks"] == 479
    files = {
        "gentail_seam.py": lf(G / "gentail_seam.py"),
        "test_gentail_seam.py": lf(G / "test_gentail_seam.py"),
        "test_gentail_logic.py": lf(G / "test_gentail_logic.py"),
        "gentail_logic.py": lf(G / "gentail_logic.py"),
        "kit_realdrive.py": lf(HERE / "kit_realdrive.py"),
        "platform_preamble.py": lf(HERE / "platform_preamble.py"),
        "k100_fixture.json": json.dumps({"q_workers": k3["workers"], "q_pool": k3.get("pool")}),
        "duck_timeline/out_analysis2/reprefill_per_turn.csv": lf(DUCK / "out_analysis2/reprefill_per_turn.csv"),
        "duck_timeline/out_analysis1/gaps.csv": lf(DUCK / "out_analysis1/gaps.csv"),
    }
    shas = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in files.items()}
    code = lf(HERE / "kit_cell.py")
    for k, v in (("__BLOB__", blob(files)), ("__PIN__", str(PIN)), ("__FILESHA__", json.dumps(shas).replace('"', "'"))):
        assert code.count(k) >= 1, k
        code = code.replace(k, v)
    compile(code, "<gentail kit>", "exec")
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1).encode("utf-8")
    assert len(data) < 400_000, len(data)
    assert b"\r" not in data.replace(b"\\r", b"")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{SLUG}.ipynb").write_bytes(data)
    meta = {"id": f"{OWNER}/{SLUG}", "title": "Q27 E30 Gentail Kit", "code_file": f"{SLUG}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"],
            "model_sources": MODELS, "docker_image": IMAGE, "machine_shape": "None"}
    (OUT / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    print("kit", OUT, "|", len(data), "bytes | ipynb sha256", hashlib.sha256(data).hexdigest())
    print("metadata sha256", hashlib.sha256((OUT / "kernel-metadata.json").read_bytes()).hexdigest())
    for k, v in shas.items():
        print(" ", k, v)


if __name__ == "__main__":
    main()
