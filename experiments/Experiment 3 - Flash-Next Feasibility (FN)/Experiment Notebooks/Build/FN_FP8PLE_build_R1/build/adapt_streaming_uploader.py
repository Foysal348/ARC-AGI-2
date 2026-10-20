"""FN-PLEFP8-UPLOAD: the tested ARC-AGI-3 streaming uploader (HF -> Kaggle resumable blob upload, no disk staging,
hash-verified, instance created only if every hash matches), adapted to the albucino FP8 PLE overlay.
Changes vs the reference: config, an account check, descriptive text, and (review R1) a fail-closed final verification
that also covers KAGGLE_MIRROR_MANIFEST.json. The stream / resume / publish code is untouched.

usage: python adapt_streaming_uploader.py [reference.ipynb] [out.ipynb]      (defaults: bundle layout)"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reference" / "NVIDIA_FlashNext_NVFP4_01_upload_on_kaggle.ipynb"
DST = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "notebooks" / "FN-PLEFP8-UPLOAD.ipynb"
MAN = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
OVERLAY = {p: v for p, v in MAN["files"].items() if v["role"] == "overlay"}
TOTAL = sum(v["size"] for v in OVERLAY.values())
REV = MAN["revision"]

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
The last cell is fail-closed: it raises unless Kaggle lists every file, including the mirror manifest, with the
expected size.

### What gets published ({TOTAL / 1e9:.1f} GB)
Only the files of `{MAN["repo"]}` (revision `{REV[:10]}`) that differ from Intel's checkpoint, plus albucino's small
runtime files so that the overlay is authoritative for everything except the weights:
10 FP8 PLE shards (`model-plefp8-*.safetensors`, the official RadixArk FP8 table + scale), `config.json` (Intel's
config + `ple_embedding_dtype: float8_e4m3fn`), `model.safetensors.index.json`, tokenizer / chat template /
generation + preprocessor configs (identical to Intel's), LICENSE, README.md, SHA256SUMS, plus
`KAGGLE_MIRROR_MANIFEST.json`. The 15 main W4A16 shards are byte-identical to
`Intel/Qwen3.8-Flash-Next-W4A16-AutoRound@{MAN["intel_revision"][:7]}` (SHA256 checked) and are taken at run time from
`dfranzen/intel-qwen3.8-flash-next-w4a16-autoround` (merged with this overlay by FN-SMOKE).

License: Qwen Community License 1.0 (LICENSE file included). Kaggle label "Other (specified in description)" — the
same label foysalemonshanto's NVIDIA mirror was accepted with (checked 2026-10-05).

Kaggle path afterwards: `/kaggle/input/models/foysalemonshanto/qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1`

The run is idempotent: if the model instance already exists it only lists and verifies its files.
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
LICENSE_NAME = "Other (specified in description)"   # Qwen Community License 1.0; label accepted for the NVIDIA mirror
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

# ---------------------------------------------------------------- cell 3: manifest notes; keep the manifest bytes
i = cells[3].index('        "serving_notes": {')
j = cells[3].index('    TOKENS["KAGGLE_MIRROR_MANIFEST.json"]')
cells[3] = cells[3][:i] + f'''        "serving_notes": {{
            "what": "overlay: merge with the 15 main shards of Intel/Qwen3.8-Flash-Next-W4A16-AutoRound@{MAN["intel_revision"][:7]} "
                    "(byte-identical to albucino's), e.g. dfranzen/intel-qwen3.8-flash-next-w4a16-autoround on Kaggle",
            "use_from_overlay": "everything except the 15 model-000NN-of-00017 shards",
            "ple": "FP8 E4M3FN n-gram/PLE table + weight_scale (RadixArk), text_config.ple_embedding_dtype=float8_e4m3fn",
            "runtime": "Pennyroyal SGLang 0.5.19 (dfranzen/pennyroyal-v253), --quantization auto-round --ple-offload-embedding",
            "license": "Qwen Community License 1.0 (see LICENSE)",
        }},
    }}
''' + cells[3][j:]
rep(3, '''    TOKENS["KAGGLE_MIRROR_MANIFEST.json"] = upload_bytes(
        "KAGGLE_MIRROR_MANIFEST.json", (json.dumps(mirror_manifest, indent=1) + "\\n").encode("utf-8"))''',
    '''    MANIFEST_BYTES = (json.dumps(mirror_manifest, indent=1) + "\\n").encode("utf-8")
    TOKENS["KAGGLE_MIRROR_MANIFEST.json"] = upload_bytes("KAGGLE_MIRROR_MANIFEST.json", MANIFEST_BYTES)''')

# ---------------------------------------------------------------- cell 4: descriptive metadata only
i = cells[4].index('            "subtitle":')
j = cells[4].index('            "licenseName": LICENSE_NAME,\n        }, indent=2))\n        created')
cells[4] = cells[4][:i] + '''            "subtitle": "FP8 PLE overlay (albucino) for Intel W4A16 AutoRound Qwen3.8-Flash-Next",
            "isPrivate": True,
            "description": (
                f"Private mirror of the FP8-PLE files of {HF_REPO} (revision {HF_REVISION}) for offline Kaggle "
                "inference: 10 FP8 PLE shards + config + index + tokenizer files. Merge with the main shards of "
                "Intel/Qwen3.8-Flash-Next-W4A16-AutoRound. License: Qwen Community License 1.0 (LICENSE in the files)."
            ),
''' + cells[4][j:]
i = cells[4].index('        "overview": (')
j = cells[4].index('        "licenseName": LICENSE_NAME,\n        "fineTunable"')
cells[4] = cells[4][:i] + '''        "overview": (f"FP8-PLE overlay of {HF_REPO} at {HF_REVISION}, uploaded byte-identical (hashes in "
                     "KAGGLE_MIRROR_MANIFEST.json). 10 model-plefp8 shards + config.json + index + tokenizer files."),
        "usage": ("Symlink these files together with the 15 main shards of Intel W4A16 AutoRound into one folder "
                  "and serve it with Pennyroyal SGLang (--quantization auto-round --ple-offload-embedding). "
                  "License: Qwen Community License 1.0 (LICENSE)."),
''' + cells[4][j:]

# ---------------------------------------------------------------- cell 5: fail closed (review R1: it only printed)
rep(5, '''expected_upload = {n: EXPECTED_FILES[n][0] for n in EXPECTED_FILES if n not in NOT_UPLOADED}
''', '''expected_upload = {n: EXPECTED_FILES[n][0] for n in EXPECTED_FILES if n not in NOT_UPLOADED}
# the manifest is checked too: exact size when this run uploaded it, presence when the instance already existed
expected_upload["KAGGLE_MIRROR_MANIFEST.json"] = len(MANIFEST_BYTES) if "MANIFEST_BYTES" in globals() else None
''')
rep(5, '''wrong = {n: (listed[n], expected_upload[n]) for n in expected_upload if n in listed and listed[n] != expected_upload[n]}
print("missing:", missing or "none", "| size mismatches:", wrong or "none")
''', '''wrong = {n: (listed[n], expected_upload[n]) for n in expected_upload
         if n in listed and expected_upload[n] is not None and listed[n] != expected_upload[n]}
print("missing:", missing or "none", "| size mismatches:", wrong or "none")
if missing or wrong:
    raise RuntimeError(f"Kaggle mirror verification FAILED: missing={missing} wrong_sizes={wrong} - do not attach this model")
print(f"VERIFIED: all {len(expected_upload)} files listed with the expected sizes")
''')

for k, src in enumerate(cells):
    nb["cells"][k]["source"] = src.splitlines(keepends=True)
    if nb["cells"][k]["cell_type"] == "code":
        nb["cells"][k]["outputs"] = []; nb["cells"][k]["execution_count"] = None
DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_bytes((json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
print("wrote", DST, hashlib.sha256(DST.read_bytes()).hexdigest())
