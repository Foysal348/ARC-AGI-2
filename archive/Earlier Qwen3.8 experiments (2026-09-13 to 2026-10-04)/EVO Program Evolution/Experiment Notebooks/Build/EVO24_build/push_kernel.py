"""Push (Save & Run All) a notebook to Kaggle as a private 4xL4 kernel. Prints the full plan; pushes only with --push.

usage: python push_kernel.py <notebook.ipynb> <kernel-slug> --account esfersami50|foysalemonshanto --title "..." [--push]
The Kaggle image is pinned to the Python 3.12 build (the default image moved to Python 3.13 on 2026-10-03 and the
cp312 wheelhouse cannot install vLLM there).
"""
import base64
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
nb_path, slug = Path(sys.argv[1]), sys.argv[2]
account = sys.argv[sys.argv.index("--account") + 1]
title = sys.argv[sys.argv.index("--title") + 1] if "--title" in sys.argv else slug
if account == "esfersami50":
    headers = {"Authorization": "Bearer " + Path(r"C:\Users\hp\.kaggle-esfer\access_token").read_text(encoding="utf-8").strip()}
elif account == "foysalemonshanto":
    kj = json.loads(Path(os.path.expanduser(r"~\.kaggle\kaggle.json")).read_text(encoding="utf-8"))
    headers = {"Authorization": "Basic " + base64.b64encode(f"{kj['username']}:{kj['key']}".encode()).decode()}
else:
    raise SystemExit("unknown account")
raw = nb_path.read_bytes()
nb = json.loads(raw)
payload = {
    "slug": f"{account}/{slug}", "newTitle": title[:50], "text": json.dumps(nb, ensure_ascii=False),
    "language": "python", "kernelType": "notebook", "isPrivate": True,
    "enableGpu": True, "enableTpu": False, "enableInternet": False, "machineShape": "NvidiaL4",
    "datasetDataSources": (sys.argv[sys.argv.index("--datasets") + 1].split(",") if "--datasets" in sys.argv
                           else ["mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3"]),
    "competitionDataSources": ["arc-prize-2026-arc-agi-2"],
    "kernelDataSources": [],
    "modelDataSources": (sys.argv[sys.argv.index("--models") + 1].split(",") if "--models" in sys.argv
                         else ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"]),
    "categoryIds": [], "dockerImage": DOCKER_IMAGE, "dockerImagePinningType": "original",
}
print(json.dumps({"account": account, "kernel": payload["slug"], "title": payload["newTitle"],
                  "notebook": str(nb_path), "notebook_sha256": hashlib.sha256(raw).hexdigest(),
                  "machine": "NvidiaL4 (4xL4)", "internet": False, "private": True,
                  "docker_image": DOCKER_IMAGE[-24:], "datasets": payload["datasetDataSources"],
                  "competition": payload["competitionDataSources"], "models": payload["modelDataSources"]}, indent=2))
if "--push" not in sys.argv:
    print("DRY RUN - nothing pushed")
    raise SystemExit(0)
req = urllib.request.Request("https://www.kaggle.com/api/v1/kernels/push", data=json.dumps(payload).encode(),
                             headers={**headers, "Content-Type": "application/json"}, method="POST")
r = json.loads(urllib.request.urlopen(req, timeout=300).read())
print({k: r.get(k) for k in ("url", "versionNumber", "error", "invalidDatasetSources", "invalidCompetitionSources", "invalidModelSources")})
