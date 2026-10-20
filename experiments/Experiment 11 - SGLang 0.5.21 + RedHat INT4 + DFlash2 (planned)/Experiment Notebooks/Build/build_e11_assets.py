"""Experiment 11 asset notebooks (Claude, 2026-10-15, user go "ওকে শুরু করো"). Deterministic builder.

Builds two Kaggle CPU notebooks (Internet ON, no GPU quota), adapted from the ARC-AGI-3 notebooks that are proven on
Kaggle (D:/Machine Learning 1 Year Challenge/ARC-AGI-3/experiments/2026-09-26_nvidia-flashnext-nvfp4/01_model-upload and
2026-09-25_occamy-1.0-nvfp4/02_runtime):

  E11_01_model_mirror_upload_on_KAGGLE.ipynb
      streams RedHatAI/Qwen3.8-27B-INT4 (19.47 GB) and z-lab/Qwen3.8-27B-DFlash2 (3.85 GB) from Hugging Face straight
      into Kaggle's resumable upload (no disk staging), hashes every file against the pinned HF tree, then creates two
      Kaggle models (private by user choice 2026-10-15; PUBLIC = True would make them public).
  E11_02_sglang_0_5_21_wheelhouse_build_on_KAGGLE.ipynb
      downloads the 220-wheel SGLang 0.5.21 lock (torch 2.13.0 cu130, flashinfer 0.6.18 + jit-cache cu130,
      transformers 5.12.1, sglang-kernel 0.4.7) - resolved off-Kaggle with uv for Linux / CPython 3.12 because sglang
      0.5.19-0.5.21 pin cuda-tile==1.6.0rc5, which exists only as an sdist; the lock overrides it to cuda-tile 1.6.0
      (cp312 wheel). Offline install test (--no-deps), CPU probes, checksums, then a Kaggle dataset.

Rev 2 (2026-10-15, after the external audit):
  - nvcc / crt / nvvm / nvjitlink pinned to 13.0.88 = torch 2.13's own cuda-toolkit 13.0.3 set (cudart 13.0.96, Kaggle
    L4 driver = CUDA 13.0). 13.4.92 came only from humming-kernels' unpinned nvcc. The other 216 pins are unchanged.
  - install test into a venv (not --target + PYTHONPATH): .pth files run (nvidia-cutlass-dsl is only a .pth; ARC-AGI-3's
    vLLM log showed "cuteDSL (CUTLASS Python) not available" with --target) and console scripts (ninja) land in bin/.
  - CPU toolchain check of the wheel nvcc (ARC-AGI-3's FlashInfer JIT never found the wheel nvcc), more import probes,
    and the dataset-create exit code is checked.

Inputs next to this file: hf_pins.json (pinned HF revisions + LFS sha256 / git sha1), sglang_0_5_21_requirements.lock.
usage: python build_e11_assets.py
"""
import hashlib, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent
PINS = json.loads((HERE / "hf_pins.json").read_text(encoding="utf-8"))
LOCK = (HERE / "sglang_0_5_21_requirements.lock").read_text(encoding="utf-8")
assert LOCK.count("==") == 220 and "cuda-tile==1.6.0\n" in LOCK and "sglang==0.5.21\n" in LOCK
assert all(f"{p}==13.0.88\n" in LOCK for p in ("nvidia-cuda-nvcc", "nvidia-cuda-crt", "nvidia-nvvm", "nvidia-nvjitlink"))
assert "nvidia-cuda-runtime==13.0.96\n" in LOCK and "torch==2.13.0\n" in LOCK


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


def write(name, cells):
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                       "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    data = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    (OUTDIR / name).write_bytes(data)
    print("BUILT", name, len(data), "bytes sha256", hashlib.sha256(data).hexdigest())


# ============================================================================ notebook 1: model mirror
jobs = {
    "int4": {"hf_repo": "RedHatAI/Qwen3.8-27B-INT4", "slug": "qwen3-8-27b-int4-redhat", "title": "Qwen3.8 27B INT4 W4A16 (RedHat)",
             "instance": "w4a16-g128", "subtitle": "RedHat W4A16 g128 (compressed-tensors) + BF16 MTP head, for offline SGLang/vLLM"},
    "dflash2": {"hf_repo": "z-lab/Qwen3.8-27B-DFlash2", "slug": "qwen3-8-27b-dflash2", "title": "Qwen3.8 27B DFlash2 drafter (z-lab)",
                "instance": "bf16", "subtitle": "DFlash2 BF16 speculative drafter for Qwen3.8-27B (SGLang --speculative-algorithm DFLASH)"},
}
for k, j in jobs.items():
    p = PINS[j["hf_repo"]]
    j["revision"] = p["sha"]
    j["files"] = {n: list(v) for n, v in sorted(p["files"].items())}
    j["total"] = p["total"]
    assert p["license"] == "apache-2.0", (k, p["license"])
JOBS_LITERAL = json.dumps(jobs, indent=1)

M0 = f"""# E11-01 · Mirror RedHat Qwen3.8-27B INT4 + z-lab DFlash2 → Kaggle Models (run this ON Kaggle)

**Notebook settings: Accelerator = None (CPU), Internet = ON.** No GPU quota is used.

### Before running: attach two Kaggle Secrets, start the run from the editor
Add-ons → Secrets → attach `KAGGLE_USERNAME` and `KAGGLE_KEY`, then **Save Version → Save & Run All** from the notebook
editor (a run started through the Kaggle API cannot reach the secrets service - proven in the ARC-AGI-3 project).

### How it works (no disk) - same code as the ARC-AGI-3 uploader that mirrored a 132.7 GB model
Every file streams from Hugging Face straight into Kaggle's resumable upload URL in 256 MiB chunks (RAM only).
Each file is hashed while it streams and compared with the pinned Hugging Face tree (LFS sha256 / git blob sha1); a model
instance is created **only if all its files match**. A dropped connection resumes from the byte the server confirmed.

### What gets published (both Apache-2.0; **private** models, `PUBLIC = False`. Other accounts can attach them only after
they are shared with them, or once `PUBLIC` is flipped)
| Hugging Face | revision | files | Kaggle path afterwards |
|---|---|---|---|
| `RedHatAI/Qwen3.8-27B-INT4` | `{jobs['int4']['revision'][:12]}` | {len(jobs['int4']['files']) - 1} (19.47 GB: model.safetensors 18.6 GB + model_mtp.safetensors 0.85 GB) | `/kaggle/input/models/<user>/qwen3-8-27b-int4-redhat/pytorch/w4a16-g128/1` |
| `z-lab/Qwen3.8-27B-DFlash2` | `{jobs['dflash2']['revision'][:12]}` | {len(jobs['dflash2']['files']) - 2} (3.85 GB BF16 drafter) | `/kaggle/input/models/<user>/qwen3-8-27b-dflash2/pytorch/bf16/1` |

`.gitattributes` and the DFlash2 README figure are not uploaded. The run is idempotent: an instance that already exists
is only listed.
"""

M1 = """
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PUBLIC = False                      # user choice 2026-10-15: private Kaggle models (both checkpoints are Apache-2.0)
LICENSE_NAME = "Apache 2.0"
KAGGLE_FRAMEWORK = "pyTorch"        # appears as "pytorch" in the /kaggle/input path
HF_URL = "https://huggingface.co/{repo}/resolve/{rev}/{name}"
CHUNK = 256 << 20          # bytes per upload request (multiple of 256 KiB, as the resumable protocol requires)
QUANTUM = 256 << 10
READ = 8 << 20             # Hugging Face read size
WORKERS = 3
NOT_UPLOADED = {".gitattributes", "assets/dflash2-figure.png"}

# job -> pinned Hugging Face revision and every file: name -> [bytes, hash kind, hash]
# ("sha256" = LFS oid, "gitsha1" = git blob id of a non-LFS file)
JOBS = __JOBS__
for _k, _j in JOBS.items():
    assert sum(v[0] for v in _j["files"].values()) == _j["total"], _k
print("python", sys.version.split()[0], "| cpus", os.cpu_count())
for _k, _j in JOBS.items():
    print(f"  {_k}: {_j['hf_repo']} @ {_j['revision'][:12]} | {len(_j['files'])} files | {_j['total'] / 1e9:.2f} GB")
""".replace("__JOBS__", JOBS_LITERAL)

M2 = """
# Kaggle Secrets -> Kaggle API credentials. Stops here (before any transfer) if they are missing or wrong.
try:
    from kaggle_secrets import UserSecretsClient
    _secrets = UserSecretsClient()
    KAGGLE_USERNAME = _secrets.get_secret("KAGGLE_USERNAME").strip()
    _key = _secrets.get_secret("KAGGLE_KEY").strip()
except Exception as exc:
    raise RuntimeError(
        f"Could not read Kaggle Secrets KAGGLE_USERNAME / KAGGLE_KEY ({exc!r}). Attach both (Add-ons -> Secrets) "
        "and start the run from the notebook editor (Save Version -> Save & Run All).") from None

cred_dir = Path.home() / ".kaggle"
cred_dir.mkdir(parents=True, exist_ok=True)
(cred_dir / "kaggle.json").write_text(json.dumps({"username": KAGGLE_USERNAME, "key": _key}))
(cred_dir / "kaggle.json").chmod(0o600)
os.environ["KAGGLE_USERNAME"] = KAGGLE_USERNAME
os.environ["KAGGLE_KEY"] = _key
del _key

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle==2.2.4"])

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.models.upload_file import UploadFile
from kagglesdk.blobs.types.blob_api_service import ApiBlobType, ApiStartBlobUploadRequest

api = KaggleApi()
api.authenticate()
# Credential check first: kernels_list(mine=True) answers 401 for a wrong key. (Get-calls cannot be used for this:
# Kaggle answers 403 PERMISSION_DENIED both for a missing model and for someone else's private one.)
try:
    api.kernels_list(mine=True, page_size=1)
except Exception as exc:
    status = getattr(getattr(exc, "response", None), "status_code", None)
    raise RuntimeError(f"Kaggle rejected the KAGGLE_USERNAME / KAGGLE_KEY secrets (HTTP {status}). "
                       "Check the two secret values, then run again.") from None

_mine = {getattr(m, "ref", None) for m in (api.model_list(owner=KAGGLE_USERNAME, page_size=100) or [])}
for _k, _j in JOBS.items():
    _j["model_ref"] = f"{KAGGLE_USERNAME}/{_j['slug']}"
    _j["instance_ref"] = f"{_j['model_ref']}/pytorch/{_j['instance']}"
    _j["model_exists"] = _j["model_ref"] in _mine
    _j["instance_exists"] = False
    if _j["model_exists"]:
        try:
            api.model_instance_get(_j["instance_ref"])
            _j["instance_exists"] = True
        except Exception as exc:
            if getattr(getattr(exc, "response", None), "status_code", None) not in (403, 404):
                raise
    print(f"{_k}: model exists={_j['model_exists']} instance exists={_j['instance_exists']} -> {_j['instance_ref']}")
print("Kaggle API OK as", KAGGLE_USERNAME)
"""

M3 = r'''
# Stream every file Hugging Face -> Kaggle resumable upload, hashing on the fly. Nothing is written to disk.
# (Verbatim from the ARC-AGI-3 uploader; HF_REPO / HF_REVISION are set per job in the next cell.)
class _HFReader:
    """Reads one file from a byte offset in a background thread into a bounded queue (<= 48 x 8 MiB)."""

    def __init__(self, url, start, size):
        self.q = queue.Queue(maxsize=48)
        self.stop = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, args=(url, start, size), daemon=True)
        self.thread.start()

    def _run(self, url, start, size):
        try:
            headers = {"Accept-Encoding": "identity"}
            if start:
                headers["Range"] = f"bytes={start}-"
            with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as r:
                r.raise_for_status()
                if start and r.status_code != 206:
                    raise RuntimeError(f"Hugging Face ignored the Range header (HTTP {r.status_code})")
                got = start
                for piece in r.iter_content(READ):
                    if self.stop.is_set():
                        return
                    got += len(piece)
                    self.q.put(piece)
                if got != size:
                    raise RuntimeError(f"Hugging Face stream ended at {got} of {size} bytes")
        except BaseException as exc:
            self.error = exc
        finally:
            self.q.put(None)

    def get(self):
        piece = self.q.get()
        if piece is None:
            raise self.error or RuntimeError("Hugging Face stream ended early")
        return piece

    def close(self):
        self.stop.set()
        try:
            while True:
                self.q.get_nowait()
        except queue.Empty:
            pass


def _start_blob(name, size):
    req = ApiStartBlobUploadRequest()
    req.type = ApiBlobType.MODEL
    req.name = name
    req.content_length = size
    req.last_modified_epoch_seconds = int(time.time())
    with api.build_kaggle_client() as kaggle:
        resp = api.with_retry(kaggle.blobs.blob_api_client.start_blob_upload)(req)
    return resp.token, resp.create_url


def _persisted(response, size):
    """Bytes the upload server has stored, from a chunk or status response."""
    if response.status_code in (200, 201):
        return size
    if response.status_code == 308:
        rng = response.headers.get("Range")
        return int(rng.rsplit("-", 1)[1]) + 1 if rng else 0
    raise RuntimeError(f"upload server answered HTTP {response.status_code}: {response.text[:300]}")


def _query(session, put_url, size):
    r = session.put(put_url, headers={"Content-Range": f"bytes */{size}", "Content-Length": "0"}, timeout=(30, 120))
    return _persisted(r, size)


def stream_file(name, size, kind, max_stalls=12):
    """Returns (token, sha256 hex, git-sha1 hex or None, seconds, failures). Gives up only after max_stalls
    consecutive failures without progress."""
    token, put_url = _start_blob(name, size)
    session = requests.Session()
    h256 = hashlib.sha256()
    gsha = hashlib.sha1(b"blob %d\0" % size) if kind == "gitsha1" else None
    hashed = persisted = 0          # hashed == bytes confirmed stored and folded into the digests
    failures = stalls = 0
    stall_mark = -1
    next_report = 5e9
    started = time.time()
    while hashed < size or persisted < size:
        reader = _HFReader(HF_URL.format(repo=HF_REPO, rev=HF_REVISION, name=name), hashed, size)
        buf = bytearray()
        buf_start = hashed
        try:
            while buf_start < size:
                while len(buf) < CHUNK and buf_start + len(buf) < size:
                    buf += reader.get()
                if persisted > buf_start:          # stored by an earlier attempt: hash only, do not resend
                    n = min(persisted - buf_start, len(buf))
                else:
                    at_end = buf_start + len(buf) == size
                    send = len(buf) if at_end else (len(buf) // QUANTUM) * QUANTUM
                    r = session.put(put_url, data=bytes(buf[:send]), timeout=(30, 900),
                                    headers={"Content-Range": f"bytes {buf_start}-{buf_start + send - 1}/{size}"})
                    persisted = _persisted(r, size)
                    n = persisted - buf_start
                    if n < 0 or n > len(buf):
                        raise RuntimeError(f"server reports {persisted} stored; chunk started at {buf_start}")
                with memoryview(buf) as mv:
                    h256.update(mv[:n])
                    if gsha is not None:
                        gsha.update(mv[:n])
                del buf[:n]
                buf_start += n
                hashed = buf_start
                if hashed >= next_report:
                    dt = time.time() - started
                    print(f"    {name}: {hashed / 1e9:.1f}/{size / 1e9:.1f} GB ({hashed / 1e6 / dt:.0f} MB/s)", flush=True)
                    next_report += 5e9
        except Exception as exc:
            failures += 1
            stalls = stalls + 1 if hashed == stall_mark else 1
            stall_mark = hashed
            print(f"    {name}: attempt failed at {hashed} bytes (stall {stalls}/{max_stalls}): {exc!r}"[:400], flush=True)
            if stalls >= max_stalls:
                raise
            session.close()
            session = requests.Session()          # drop pooled connections after a failure
            time.sleep(min(120, 10 * stalls))
            for q in range(5):
                try:
                    persisted = _query(session, put_url, size)
                    break
                except Exception as qexc:
                    if q == 4:
                        raise
                    print(f"    {name}: status query failed ({qexc!r}); retrying"[:300], flush=True)
                    time.sleep(15 * (q + 1))
            if persisted < hashed:
                raise RuntimeError(f"{name}: server now reports {persisted} bytes < {hashed} already confirmed")
        finally:
            reader.close()
    return token, h256.hexdigest(), gsha.hexdigest() if gsha is not None else None, time.time() - started, failures


def upload_bytes(name, data):
    token, put_url = _start_blob(name, len(data))
    r = requests.put(put_url, data=data, headers={"Content-Range": f"bytes 0-{len(data) - 1}/{len(data)}"},
                     timeout=(30, 300))
    if _persisted(r, len(data)) != len(data):
        raise RuntimeError(f"{name}: small upload incomplete")
    return token
'''

M4 = r'''
# Per job: stream + verify every file, then create the model (if new) and the instance with the streamed blob tokens.
RESULTS = {}
for JOB, J in JOBS.items():
    print(f"\n==================== {JOB}: {J['hf_repo']} -> {J['instance_ref']}")
    if J["instance_exists"]:
        print("instance already exists: transfer skipped")
        RESULTS[JOB] = "exists"
        continue
    HF_REPO, HF_REVISION = J["hf_repo"], J["revision"]
    EXPECTED = {n: tuple(v) for n, v in J["files"].items()}
    todo = sorted((n for n in EXPECTED if n not in NOT_UPLOADED), key=lambda n: -EXPECTED[n][0])
    TOKENS, HASHES = {}, {}
    t0 = time.time()

    def _one(name):
        size, kind, want = EXPECTED[name]
        print(f"[{time.strftime('%H:%M:%S')}] start {name} ({size / 1e9:.2f} GB)", flush=True)
        for attempt in range(1, 4):      # a fresh upload session per attempt (e.g. if the old one expired)
            try:
                token, s256, g1, dt, fails = stream_file(name, size, kind)
                break
            except Exception as exc:
                print(f"[{time.strftime('%H:%M:%S')}] {name}: file attempt {attempt}/3 failed: {exc!r}"[:400], flush=True)
                if attempt == 3:
                    raise
        got = s256 if kind == "sha256" else g1
        print(f"[{time.strftime('%H:%M:%S')}] done  {name} in {dt:.0f} s ({size / 1e6 / max(dt, 1e-6):.0f} MB/s, "
              f"{fails} retries) {kind} {'OK' if got == want else 'MISMATCH'}", flush=True)
        return name, token, s256, got == want

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for name, token, s256, ok in pool.map(_one, todo):
            TOKENS[name], HASHES[name] = token, (s256, ok)
    bad = sorted(n for n, (_, ok) in HASHES.items() if not ok)
    total = sum(EXPECTED[n][0] for n in todo)
    dt = time.time() - t0
    print(f"streamed {len(todo)} files, {total / 1e9:.2f} GB in {dt / 60:.1f} min ({total / 1e6 / max(dt, 1e-6):.0f} MB/s)")
    if bad:
        raise RuntimeError(f"{JOB}: hash mismatch, instance NOT created: {bad}")
    print(f"all {len(todo)} files match the pinned Hugging Face revision byte-for-byte")
    manifest = {
        "source": {"hf_repo": HF_REPO, "hf_revision": HF_REVISION},
        "verified_against": "Hugging Face tree API at the pinned revision (LFS sha256 / git blob sha1), hashed while streaming",
        "files": {n: {"bytes": EXPECTED[n][0], "sha256": HASHES[n][0]} for n in sorted(todo)},
        "total_bytes": total, "not_uploaded": sorted(NOT_UPLOADED & set(EXPECTED)),
        "built_for": "ARC-AGI-2 Experiment 11: SGLang 0.5.21 + RedHat Qwen3.8-27B INT4 + DFlash2 on 4x L4",
    }
    TOKENS["KAGGLE_MIRROR_MANIFEST.json"] = upload_bytes(
        "KAGGLE_MIRROR_MANIFEST.json", (json.dumps(manifest, indent=1) + "\n").encode("utf-8"))

    if not J["model_exists"]:
        parent_dir = Path(f"/tmp/kaggle-model-parent-{JOB}")
        shutil.rmtree(parent_dir, ignore_errors=True)
        parent_dir.mkdir(parents=True)
        (parent_dir / "model-metadata.json").write_text(json.dumps({
            "ownerSlug": KAGGLE_USERNAME, "title": J["title"], "slug": J["slug"], "subtitle": J["subtitle"],
            "isPrivate": not PUBLIC,
            "description": (f"Byte-identical mirror of {HF_REPO} (revision {HF_REVISION}) for offline Kaggle inference. "
                            "License: Apache-2.0 (see README.md in the files). Hashes in KAGGLE_MIRROR_MANIFEST.json."),
            "licenseName": LICENSE_NAME,
        }, indent=2))
        created = api.model_create_new(str(parent_dir))
        if getattr(created, "error", None):
            raise RuntimeError(f"{JOB}: model create failed: {created.error}")
        print("model created:", getattr(created, "ref", None), getattr(created, "url", None))

    # model_instance_create walks a folder and asks _upload_file for a token per file. The folder holds empty
    # placeholders with the real names; the patched _upload_file returns the token of the streamed blob.
    ph = Path(f"/tmp/kaggle-instance-placeholders-{JOB}")
    shutil.rmtree(ph, ignore_errors=True)
    ph.mkdir(parents=True)
    for name in TOKENS:
        (ph / name).touch()
    (ph / "model-instance-metadata.json").write_text(json.dumps({
        "ownerSlug": KAGGLE_USERNAME, "modelSlug": J["slug"], "instanceSlug": J["instance"],
        "framework": KAGGLE_FRAMEWORK,
        "overview": f"{HF_REPO} at {HF_REVISION}, uploaded byte-identical (hashes in KAGGLE_MIRROR_MANIFEST.json).",
        "usage": "Mount ${PATH}; serve offline with SGLang 0.5.21 (see the Experiment 11 notebooks).",
        "licenseName": LICENSE_NAME, "fineTunable": False, "trainingData": [], "modelInstanceType": "Unspecified",
        "baseModelInstance": "",
        "externalBaseModelUrl": "",
    }, indent=2))
    used = []
    _original_upload_file = KaggleApi._upload_file

    def _token_for(self, file_name, full_path, blob_type, upload_context, quiet, resources, content_type=None):
        if file_name not in TOKENS:
            raise RuntimeError(f"no streamed blob for {file_name}")
        used.append(file_name)
        return UploadFile(token=TOKENS[file_name])

    KaggleApi._upload_file = _token_for
    try:
        response = api.model_instance_create(str(ph), quiet=True, dir_mode="skip")
    finally:
        KaggleApi._upload_file = _original_upload_file
    if sorted(used) != sorted(TOKENS):
        raise RuntimeError(f"{JOB}: token hand-off mismatch: used={sorted(used)}")
    if getattr(response, "error", None):
        raise RuntimeError(f"{JOB}: instance create failed: {response.error}")
    print(f"instance created with {len(used)} files:", getattr(response, "ref", None), getattr(response, "url", None))
    RESULTS[JOB] = "created"
print("\nresults:", RESULTS)
'''

M5 = r'''
# List what Kaggle now holds (processing can lag a few minutes behind the create call).
for JOB, J in JOBS.items():
    expected = {n: J["files"][n][0] for n in J["files"] if n not in NOT_UPLOADED}
    listed = {}
    for attempt in range(20):
        try:
            resp = api.model_instance_version_files(f"{J['instance_ref']}/1", page_size=100)
            listed = {f.name: f.size for f in (getattr(resp, "files", None) or [])}
        except Exception as exc:
            print("listing not ready:", repr(exc)[:200])
        if set(expected) <= set(listed):
            break
        time.sleep(60)
    print(f"\n{JOB}: {len(listed)} files listed for {J['instance_ref']}/1")
    for name in sorted(listed):
        print(f"  {listed[name]:>14,}  {name}")
    missing = sorted(set(expected) - set(listed))
    wrong = {n: (listed[n], expected[n]) for n in expected if n in listed and listed[n] != expected[n]}
    print("missing:", missing or "none", "| size mismatches:", wrong or "none")
    print("attach with:", f'"model_sources": ["{J["instance_ref"]}/1"]', "| mount:", f"/kaggle/input/models/{J['instance_ref']}/1")
'''

write("E11_01_model_mirror_upload_on_KAGGLE.ipynb", [md(M0), code(M1), code(M2), code(M3), code(M4), code(M5)])

# ============================================================================ notebook 2: SGLang wheelhouse
W0 = """# E11-02 · Build the SGLang 0.5.21 offline wheelhouse → Kaggle dataset (runs ON Kaggle)

**Settings:** Accelerator **None** (CPU, no GPU quota) · Internet **ON** · Add-ons → Secrets → attach `KAGGLE_USERNAME`
and `KAGGLE_KEY` · start with **Save Version → Save & Run All** from the editor (secrets are not readable in API-started runs).

**What:** every wheel of the SGLang 0.5.21 stack for this exact Kaggle environment (CPython 3.12, glibc 2.35, x86_64):
torch 2.13.0 (CUDA 13 runtime wheels), flashinfer-python 0.6.18 + flashinfer-jit-cache 0.6.18+cu130 (precompiled kernels,
no offline JIT stall), transformers 5.12.1, sglang-kernel 0.4.7 - 220 wheels, about 5 GB.

**Why a pinned lock instead of `pip download sglang`:** sglang 0.5.19-0.5.21 all pin `cuda-tile==1.6.0rc5`, which exists
on PyPI only as a source tarball, so a binary-only download fails. The lock was resolved off-Kaggle with
`uv pip compile --override cuda-tile==1.6.0` for Linux / CPython 3.12 (cuda-tile 1.6.0 has a cp312 wheel). It is
installed with **`--no-deps`**, so pip never re-checks that one pin.

**CUDA compiler wheels:** `nvidia-cuda-nvcc` / `-crt` / `nvidia-nvvm` / `nvidia-nvjitlink` are pinned to **13.0.88**, the set
torch 2.13's own `cuda-toolkit 13.0.3` defines (cudart 13.0.96). Kaggle's L4 driver supports CUDA 13.0. Without the pin, an
unpinned dependency (humming-kernels) pulls nvcc 13.4.92.

**Version 2 (2026-10-15):** run 1 published the dataset, but Kaggle drops `+` from file names
(`flashinfer_jit_cache-0.6.18+cu130-...whl` became `...-0.6.18cu130-...whl`), so a plain `pip install --find-links` cannot
match the lock. The dataset now ships **`install_wheelhouse.py`**; this notebook tests it on a Kaggle-style copy (`+`
removed) and then publishes a new version of the same dataset.

**Install in the GPU notebook (a separate venv, the image's own torch stays untouched):**

    python /kaggle/input/<this dataset>/install_wheelhouse.py /tmp/sglang-venv

It restores the names (checking their sha256), creates the venv, installs `requirements.lock` with `--no-deps`, adds
`lib/libcudart.so` + `lib64` under the pip CUDA runtime, and prints `VENV_PYTHON`, `CUDA_HOME` and `PATH`. Start SGLang
with them and a fresh `SGLANG_CACHE_DIR` / `CUDA_CACHE_PATH`.
- Why a venv rather than `--target` + `PYTHONPATH`: `.pth` files run only from a site directory. `nvidia-cutlass-dsl` is
  *only* a `.pth` file.
- Console scripts such as `ninja` must be on `PATH`.
"""

W1 = f'''
import datetime, hashlib, json, os, re, shutil, subprocess, sys, time
from pathlib import Path

SGLANG_VERSION = "0.5.21"
PUBLIC = True
DATASET_SLUG = "sglang-0-5-21-cu13-wheelhouse"
DATASET_TITLE = "SGLang 0.5.21 cu13 wheelhouse (py3.12)"
VERSION_NOTES = ("v2: install_wheelhouse.py restores the '+' Kaggle drops from file names "
                 "(flashinfer-jit-cache 0.6.18+cu130) and installs into a venv; README updated")
WH = Path("/kaggle/working/sglang-0.5.21-wheelhouse")      # this is what gets uploaded (~5 GB)
VENV = Path("/tmp/sglang-venv")                              # offline install test (venv), deleted before upload
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
assert len(PINS) == 220 and PINS["sglang"] == SGLANG_VERSION and PINS["cuda-tile"] == "1.6.0"
assert PINS["nvidia-cuda-nvcc"] == PINS["nvidia-nvvm"] == PINS["nvidia-nvjitlink"] == "13.0.88"
r = subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True, text=True)
print(r.stdout.strip())
_pipv = tuple(int(x) for x in r.stdout.split()[1].split(".")[:2])
assert _pipv >= (22, 3), "pip >= 22.3 is needed for pip --python <venv>"
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
#    (Kaggle drops '+' from file names: run 1 stored flashinfer_jit_cache-0.6.18+cu130-... as ...-0.6.18cu130-...).
INSTALLER_SRC = r"""
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
SIM = Path("/tmp/kaggle-layout-sim")                          # what a Kaggle mount looks like: '+' removed from names
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
assert "flashinfer_jit_cache-0.6.18+cu130" in r.stdout, "the renamed wheel was not restored"
SITE = Path(subprocess.run([str(VPY), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                           capture_output=True, text=True, check=True).stdout.strip())
CUDA_HOME = SITE / "nvidia" / "cu13"
inst = sum(p.stat().st_size for p in VENV.rglob("*") if p.is_file())
print(f"offline install OK in {time.time() - t0:.0f}s | installed size {inst / 1024**3:.1f} GiB | site {SITE}")
print(".pth files:", sorted(p.name for p in SITE.glob("*.pth")))
print("bin:", sorted(p.name for p in (VENV / "bin").iterdir())[:40])
'''

W4 = r'''
# 3. CPU probes (no GPU here): imports, transformers 5.12.1 reads the RedHat INT4 config + tokenizer, SGLang registry
#    knows the target and DFlash2 architectures. Recorded in BUILD_PROBE.json; the GPU notebook re-checks on GPU.
from huggingface_hub import hf_hub_download
SMALL = Path("/tmp/redhat-int4-small"); SMALL.mkdir(exist_ok=True)
for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "processor_config.json"):
    hf_hub_download("RedHatAI/Qwen3.8-27B-INT4", f, revision="91bd022d5b49442a868bc35008f6c21e1860edfa", local_dir=str(SMALL))
DF = Path("/tmp/dflash2-small"); DF.mkdir(exist_ok=True)
hf_hub_download("z-lab/Qwen3.8-27B-DFlash2", "config.json", revision="50307d4c4cde6860d4eee73e2547cd786fe8e8a4", local_dir=str(DF))

PROBE = r"""
import json, sys
out = {}
def step(name, fn):
    try:
        out[name] = fn()
    except Exception as e:
        out[name] = f"ERROR {type(e).__name__}: {str(e)[:300]}"
import importlib
step("torch", lambda: (lambda t: [t.__version__, t.version.cuda])(importlib.import_module("torch")))
step("transformers", lambda: importlib.import_module("transformers").__version__)
step("sglang", lambda: importlib.import_module("sglang").__version__)
step("flashinfer", lambda: importlib.import_module("flashinfer").__version__)
step("sgl_kernel_import", lambda: str(importlib.import_module("sgl_kernel"))[:80])
def cfg():
    from transformers import AutoConfig, AutoTokenizer
    c = AutoConfig.from_pretrained(sys.argv[1])
    t = AutoTokenizer.from_pretrained(sys.argv[1])
    msgs = [{"role": "user", "content": "hi"}]
    txt = t.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    return {"arch": getattr(c, "architectures", None), "model_type": c.model_type, "vocab": len(t), "prompt_tail": txt[-40:]}
step("redhat_config_tokenizer", cfg)
def reg():
    from sglang.srt.models.registry import ModelRegistry
    archs = set(ModelRegistry.get_supported_archs())
    return {a: (a in archs) for a in ("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM", "DFlash2DraftModel")} | \
           {"n_archs": len(archs), "dflash_like": sorted(a for a in archs if "lash" in a.lower())[:8]}
step("sglang_registry", reg)
step("cutlass_dsl", lambda: [getattr(importlib.import_module("cutlass"), "__version__", "imported"),
                             any("dsl_packages" in p for p in sys.path)])
step("sglang_models_qwen3_5", lambda: str(importlib.import_module("sglang.srt.models.qwen3_5"))[:100])
step("sglang_models_dflash", lambda: str(importlib.import_module("sglang.srt.models.dflash"))[:100])
step("dflash2_config", lambda: (lambda c: {"arch": c.get("architectures"), "layers": c.get("num_hidden_layers")})(
    json.load(open(sys.argv[2]))))
print("PROBE_JSON " + json.dumps(out))
"""
env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "CUDA_HOME", "CUDA_PATH")}
env.update({"PATH": f"{VENV / 'bin'}:{CUDA_HOME / 'bin'}:{env.get('PATH', '')}", "CUDA_HOME": str(CUDA_HOME),
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "SGLANG_CACHE_DIR": "/tmp/sgl-cache"})
r = subprocess.run([str(VPY), "-c", PROBE, str(SMALL), str(DF / "config.json")], env=env, capture_output=True, text=True,
                   timeout=1800)
line = next((l for l in r.stdout.splitlines() if l.startswith("PROBE_JSON ")), None)
PROBE_RESULT = json.loads(line[11:]) if line else {"crash": r.stderr[-2000:]}

# CPU toolchain check of the wheel nvcc 13.0.88 (compiling needs no GPU). In ARC-AGI-3 FlashInfer's JIT never found the
# wheel nvcc; this records what the GPU notebook must set (CUDA_HOME, include / lib paths, a libcudart.so link).
TOOL = {"files": {rel: (CUDA_HOME / rel).exists() for rel in (
    "bin/nvcc", "bin/ptxas", "bin/fatbinary", "nvvm/bin/cicc", "include/cuda_runtime.h", "include/crt/host_config.h",
    "include/cccl", "lib/libcudart.so.13", "lib/libcudart.so", "lib64")}}
def tool(name, cmd):
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
        TOOL[name] = {"rc": p.returncode, "out": (p.stdout + p.stderr).strip()[-700:]}
    except Exception as e:
        TOOL[name] = {"rc": None, "out": f"{type(e).__name__}: {e}"[:700]}
NVCC = str(CUDA_HOME / "bin" / "nvcc")
TC = Path("/tmp/e11-toolchain"); shutil.rmtree(TC, ignore_errors=True); (TC / "lib").mkdir(parents=True)
(TC / "k.cu").write_text('#include <cuda_runtime.h>\n__global__ void k(float* x) { x[threadIdx.x] *= 2.f; }\n'
                         'extern "C" int launch(float* x) { k<<<1, 32>>>(x); return (int)cudaGetLastError(); }\n')
(TC / "lib" / "libcudart.so").symlink_to(CUDA_HOME / "lib" / "libcudart.so.13")
INC = ["-I", str(CUDA_HOME / "include")]
SM89 = ["-gencode=arch=compute_89,code=sm_89"]
tool("nvcc_version", [NVCC, "--version"])
tool("gcc_version", ["gcc", "--version"])
tool("ninja_version", [str(VENV / "bin" / "ninja"), "--version"])
tool("cubin_sm89", [NVCC, *SM89, "-cubin", *INC, "-o", str(TC / "k.cubin"), str(TC / "k.cu")])
tool("cubin_sm89_no_I", [NVCC, *SM89, "-cubin", "-o", str(TC / "k2.cubin"), str(TC / "k.cu")])
tool("object_sm89", [NVCC, *SM89, "-Xcompiler", "-fPIC", "-c", *INC, "-o", str(TC / "k.o"), str(TC / "k.cu")])
tool("shared_lib_link", [NVCC, *SM89, "-Xcompiler", "-fPIC", "-shared", "-cudart", "shared", *INC,
                         "-L", str(CUDA_HOME / "lib"), "-o", str(TC / "k.so"), str(TC / "k.cu")])
tool("shared_lib_link_with_libcudart_so", [NVCC, *SM89, "-Xcompiler", "-fPIC", "-shared", "-cudart", "shared", *INC,
                                           "-L", str(TC / "lib"), "-o", str(TC / "k3.so"), str(TC / "k.cu")])
PROBE_RESULT["toolchain"] = TOOL
print(json.dumps(PROBE_RESULT, indent=1))
(WH / "BUILD_PROBE.json").write_text(json.dumps(PROBE_RESULT, indent=2), encoding="utf-8")
_bad = [k for k, v in PROBE_RESULT.items() if k == "crash" or (isinstance(v, str) and v.startswith("ERROR"))]
_bad += [k for k, v in TOOL.items() if k != "files" and v.get("rc") != 0]
_bad += [f"missing {k}" for k, ok in TOOL["files"].items() if not ok]
print("\nPROBE SUMMARY - not OK on this CPU machine (recorded, reviewed before the GPU notebook):", _bad or "none")
'''

W5 = r'''
# 4. The Experiment 11 launch flags must exist in this build (help text of sglang.launch_server)
FLAGS = ["--model-path", "--tp-size", "--quantization", "--kv-cache-dtype", "--attention-backend", "--mem-fraction-static",
         "--max-running-requests", "--chunked-prefill-size", "--max-prefill-tokens", "--context-length",
         "--mamba-radix-cache-strategy", "--mamba-ssm-dtype", "--mamba-full-memory-ratio", "--speculative-algorithm",
         "--speculative-draft-model-path", "--speculative-draft-model-quantization", "--speculative-num-draft-tokens",
         "--speculative-num-steps", "--speculative-eagle-topk", "--reasoning-parser", "--enable-metrics",
         "--disable-flashinfer-autotune"]
r = subprocess.run([str(VPY), "-m", "sglang.launch_server", "--help"], env=env, capture_output=True, text=True, timeout=1800)
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
    "sglang": SGLANG_VERSION, "python": sys.version.split()[0],
    "glibc": subprocess.run(["ldd", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
    "built_utc": datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z", "built_on": "Kaggle",
    "wheel_count": len(list(WH.glob("*.whl"))),
    "note": "cuda-tile overridden 1.6.0rc5 -> 1.6.0 (rc5 has no wheel); nvcc/crt/nvvm/nvjitlink pinned 13.0.88 (torch "
            "2.13 cuda-toolkit 13.0.3 set, Kaggle L4 driver = CUDA 13.0); install with install_wheelhouse.py (venv, "
            "--no-deps, restores the '+' Kaggle drops from file names)",
    "dataset_version_notes": VERSION_NOTES,
    "stamp": f"sglang=={SGLANG_VERSION} torch=={PINS['torch']} flashinfer=={PINS['flashinfer-python']} "
             f"jit-cache=={PINS['flashinfer-jit-cache']} transformers=={PINS['transformers']}",
    "probe": PROBE_RESULT, "pins": PINS,
}
(WH / "BUILD_INFO.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
(WH / "README.md").write_text(
    f"# SGLang {SGLANG_VERSION} offline wheelhouse (built on Kaggle, CPython {info['python']}, CUDA 13 wheels)\n\n"
    "Install (one command, offline, CPython 3.12 Linux x86_64):\n\n"
    "    python /kaggle/input/<this dataset>/install_wheelhouse.py /tmp/sglang-venv\n\n"
    "Do NOT run `pip install --find-links <this dir>` directly: Kaggle drops '+' from file names, so\n"
    "flashinfer_jit_cache-0.6.18+cu130-...whl is stored as ...-0.6.18cu130-...whl and pip cannot match the lock.\n"
    "install_wheelhouse.py restores the names from SHA256SUMS (checking their sha256), creates the venv, installs\n"
    "requirements.lock with --no-deps and adds lib/libcudart.so + lib64 under the pip CUDA runtime. It prints\n"
    "VENV_PYTHON, CUDA_HOME and PATH: start SGLang with them and fresh SGLANG_CACHE_DIR / CUDA_CACHE_PATH.\n"
    "A venv (not --target + PYTHONPATH) so that .pth files run (nvidia-cutlass-dsl) and console scripts (ninja) are\n"
    "on PATH. On Kaggle also symlink libcuda.so.1 as libcuda.so into a LIBRARY_PATH directory (FlashInfer JIT links -lcuda).\n\n"
    "Made for ARC-AGI-2 Experiment 11 (RedHat Qwen3.8-27B INT4 + DFlash2 on 4x L4). The lock overrides cuda-tile\n"
    "1.6.0rc5 -> 1.6.0 because rc5 is sdist-only; --no-deps keeps pip from re-checking that pin. nvcc/crt/nvvm/nvjitlink\n"
    "are pinned to 13.0.88 (torch 2.13's cuda-toolkit 13.0.3 set; the Kaggle L4 driver supports CUDA 13.0).\n"
    "BUILD_PROBE.json (imports + CPU nvcc toolchain) / BUILD_FLAGS.json record the CPU-side checks done at build time.\n",
    encoding="utf-8")
files = sorted(p for p in WH.iterdir() if p.is_file() and p.name != "SHA256SUMS")
(WH / "SHA256SUMS").write_text("".join(f"{WHEEL_SUMS.get(p.name) or sha256(p)}  {p.name}\n" for p in files),
                               encoding="utf-8")
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
if EXISTS:      # v2: run 1 created the dataset -> publish a new version of it
    cmd = [sys.executable, "-m", "kaggle", "datasets", "version", "-p", str(WH), "-m", VERSION_NOTES, "-r", "skip"]
else:
    cmd = [sys.executable, "-m", "kaggle", "datasets", "create", "-p", str(WH), "-r", "skip"] + (["--public"] if PUBLIC else [])
print("dataset exists:", EXISTS, "->", " ".join(cmd[3:5]))
r = subprocess.run(cmd, text=True)
print("exit code:", r.returncode)
# Run 1: the create call printed "Expecting value" (a non-JSON reply) and exited 1, yet the dataset appeared a little later.
# So the exit code alone is not trusted: wait until the new file install_wheelhouse.py is listed in the dataset.
t0 = time.time()
listed = False
while time.time() - t0 < 25 * 60:
    _ls = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "files", REF, "--page-size", "300"],
                         capture_output=True, text=True)
    if "install_wheelhouse.py" in _ls.stdout:
        listed = True
        break
    time.sleep(30)
print(f"new version listed: {listed} after {time.time() - t0:.0f}s")
subprocess.run([sys.executable, "-m", "kaggle", "datasets", "status", REF])
if not listed:
    raise SystemExit("the new dataset version is not visible after 25 min - paste this output back")
print("attach:", REF, "| mount: /kaggle/input/datasets/" + REF)
'''

write("E11_02_sglang_0_5_21_wheelhouse_build_on_KAGGLE.ipynb",
      [md(W0), code(W1), md("## 1. Download every pinned wheel"), code(W2), md("## 2. Offline install test"), code(W3),
       md("## 3. CPU probes"), code(W4), md("## 4. Launch-flag check"), code(W5), md("## 5. Checksums, README"), code(W6),
       md("## 6. Publish the dataset"), code(W7)])
