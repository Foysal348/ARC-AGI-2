"""FN-PLEFP8-UPLOAD v2: the tested ARC-AGI-3 streaming uploader (HF -> Kaggle resumable blob upload, no disk staging,
hash-verified, instance created only if every hash matches), adapted to the albucino FP8 PLE overlay.
Only config / descriptive text changes; the transfer, resume, publish and verify code is untouched."""
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).parent
SRC = Path(r"D:\Machine Learning 1 Year Challenge\ARC-AGI-3\experiments\2026-09-26_nvidia-flashnext-nvfp4\01_model-upload"
           r"\NVIDIA_FlashNext_NVFP4_01_upload_on_kaggle.ipynb")
DST = Path(r"D:\ARC AGI 2\Research Base Experiment\FN-PLEFP8-UPLOAD.ipynb")
MAN = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
OVERLAY = {p: v for p, v in MAN["files"].items() if v["role"] == "overlay"}
TOTAL = sum(v["size"] for v in OVERLAY.values())
REV = MAN["revision"]

if DST.exists() and not (HERE / "FN-PLEFP8-UPLOAD.v1-kagglehub.ipynb").exists():
    shutil.copy(DST, HERE / "FN-PLEFP8-UPLOAD.v1-kagglehub.ipynb")
nb = json.loads(SRC.read_text(encoding="utf-8"))
cells = ["".join(c["source"]) for c in nb["cells"]]
assert len(cells) == 6


def rep(i, a, b):
    assert cells[i].count(a) == 1, (i, a[:80], cells[i].count(a))
    cells[i] = cells[i].replace(a, b)


# ---------------------------------------------------------------- cell 0: markdown
cells[0] = f"""# albucino FP8 PLE overlay for Qwen3.8-Flash-Next W4A16 -> private Kaggle Model (run this ON Kaggle)

**Notebook settings: Accelerator = None (CPU), Internet = ON.** No GPU quota is used. Utility notebook: no
competition data, never submitted.

### Before running: attach two Kaggle Secrets, start the run from the editor
Add-ons -> Secrets -> attach `KAGGLE_USERNAME` and `KAGGLE_KEY` (account **foysalemonshanto**), then
**Save Version -> Save & Run All** from the notebook editor (a run started through the Kaggle API could not reach
the secrets service on 2026-09-26).

### How it works (no disk) — same tested code as the ARC-AGI-3 NVIDIA upload (132.7 GB, 2026-09-26)
Every file streams straight from Hugging Face into Kaggle's resumable upload URL in 256 MiB chunks (RAM only,
3 workers). Each file is hashed while it streams and compared with the pinned Hugging Face tree; the model instance
is created **only if all {len(OVERLAY)} files match**. A dropped connection resumes from the byte the server confirmed.

### What gets published ({TOTAL / 1e9:.1f} GB)
Only the files of `{MAN["repo"]}` (revision `{REV[:10]}`) that differ from Intel's checkpoint:
10 FP8 PLE shards (`model-plefp8-*.safetensors`, the official RadixArk FP8 table + scale), `config.json` (Intel's
config + `ple_embedding_dtype: float8_e4m3fn`), `model.safetensors.index.json`, LICENSE, README.md, SHA256SUMS, plus
`KAGGLE_MIRROR_MANIFEST.json`. The 15 main W4A16 shards are byte-identical to
`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound@{MAN["intel_revision"][:7]}` (SHA256 checked) and are taken at run time from
`dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` (merged with this overlay by FN-SMOKE).

Kaggle path afterwards: `/kaggle/input/models/foysalemonshanto/qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1`

The run is idempotent: if the model instance already exists it only lists its files.
"""

# ---------------------------------------------------------------- cell 1: config
i = cells[1].index('HF_REPO = "nvidia/Qwen3.8-Flash-Next-NVFP4"')
j = cells[1].index('PLACEHOLDER_DIR = ')
cells[1] = cells[1][:i] + f'''HF_REPO = "{MAN["repo"]}"
HF_REVISION = "{REV}"   # pinned 2026-10-05
HF_URL = "https://huggingface.co/{{repo}}/resolve/{{rev}}/{{name}}"
KAGGLE_MODEL_SLUG = "qwen38-flash-next-fp8ple-overlay"
KAGGLE_MODEL_TITLE = "Qwen3.8 Flash-Next W4A16 FP8-PLE overlay"
KAGGLE_FRAMEWORK = "pyTorch"          # appears as "pytorch" in the /kaggle/input path
KAGGLE_INSTANCE_SLUG = "albucino-fp8ple"
LICENSE_NAME = "Other (specified in description)"   # Qwen Community License 1.0 (LICENSE file included)
''' + cells[1][j:]
i = cells[1].index("# name -> (bytes, hash kind, hash).")
j = cells[1].index('print("python"')
rows = "\n".join(f"    {n!r}: ({v['size']}, 'sha256', {v['sha256']!r})," for n, v in sorted(OVERLAY.items()))
cells[1] = cells[1][:i] + f'''# name -> (bytes, hash kind, hash). "sha256" = LFS oid for shards, sha256 of the content for the small files
# (the streamer always computes sha256). Taken from the Hugging Face tree API at HF_REVISION.
EXPECTED_FILES = {{
{rows}
}}
EXPECTED_TOTAL_BYTES = {TOTAL:_}
NOT_UPLOADED = set()
assert sum(v[0] for v in EXPECTED_FILES.values()) == EXPECTED_TOTAL_BYTES
assert len(EXPECTED_FILES) == {len(OVERLAY)}
''' + cells[1][j:]
rep(1, 'PLACEHOLDER_DIR = Path("/tmp/nv-instance-placeholders")', 'PLACEHOLDER_DIR = Path("/tmp/fp8ple-instance-placeholders")')

# ---------------------------------------------------------------- cell 2: refuse a wrong account before any transfer
rep(2, 'api = KaggleApi()\napi.authenticate()\n',
    'api = KaggleApi()\napi.authenticate()\nif KAGGLE_USERNAME != "foysalemonshanto":\n'
    '    raise RuntimeError(f"secrets belong to {KAGGLE_USERNAME!r}, expected foysalemonshanto - nothing transferred")\n')

# ---------------------------------------------------------------- cell 3: manifest notes only
i = cells[3].index('        "serving_notes": {')
j = cells[3].index('    TOKENS["KAGGLE_MIRROR_MANIFEST.json"]')
cells[3] = cells[3][:i] + f'''        "serving_notes": {{
            "what": "overlay: merge with the 15 main shards of Intel/Qwen3.8-Flash-Next-W4A16-AutoRound@{MAN["intel_revision"][:7]} "
                    "(byte-identical to albucino's), e.g. dfranzen/intel-qwen3.8-flash-next-w4a16-autoround on Kaggle",
            "use_from_overlay": ["config.json", "model.safetensors.index.json", "model-plefp8-*.safetensors"],
            "ple": "FP8 E4M3FN n-gram/PLE table + weight_scale (RadixArk), text_config.ple_embedding_dtype=float8_e4m3fn",
            "runtime": "Pennyroyal SGLang 0.5.19 (dfranzen/pennyroyal-v253), --quantization auto-round --ple-offload-embedding",
            "license": "Qwen Community License 1.0 (see LICENSE)",
        }},
    }}
''' + cells[3][j:]

# ---------------------------------------------------------------- cell 4: descriptive metadata only
i = cells[4].index('            "subtitle":')
j = cells[4].index('            "licenseName": LICENSE_NAME,\n        }, indent=2))\n        created')
cells[4] = cells[4][:i] + '''            "subtitle": "FP8 PLE overlay (albucino) for Intel W4A16 AutoRound Qwen3.8-Flash-Next",
            "isPrivate": True,
            "description": (
                f"Private mirror of the FP8-PLE files of {HF_REPO} (revision {HF_REVISION}) for offline Kaggle "
                "inference: 10 FP8 PLE shards + config + index. Merge with the main shards of "
                "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound. License: Qwen Community License 1.0 (LICENSE in the files)."
            ),
''' + cells[4][j:]
i = cells[4].index('        "overview": (')
j = cells[4].index('        "licenseName": LICENSE_NAME,\n        "fineTunable"')
cells[4] = cells[4][:i] + '''        "overview": (f"FP8-PLE overlay of {HF_REPO} at {HF_REVISION}, uploaded byte-identical (hashes in "
                     "KAGGLE_MIRROR_MANIFEST.json). 10 model-plefp8 shards + config.json + index."),
        "usage": ("Symlink these files together with the 15 main shards of Intel W4A16 AutoRound into one folder "
                  "and serve it with Pennyroyal SGLang (--quantization auto-round --ple-offload-embedding)."),
''' + cells[4][j:]

for k, src in enumerate(cells):
    nb["cells"][k]["source"] = src.splitlines(keepends=True)
    if nb["cells"][k]["cell_type"] == "code":
        nb["cells"][k]["outputs"] = []; nb["cells"][k]["execution_count"] = None
DST.write_bytes((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
print("wrote", DST, hashlib.sha256(DST.read_bytes()).hexdigest())
