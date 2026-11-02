"""E25 KLEE100-UPLOAD builder: the tested ARC-AGI-3 streaming uploader (HF -> Kaggle resumable blob upload, no disk
staging, sha256 while streaming, instance created only if every hash matches), adapted to
klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP minus its BF16 PLE shard.

The adaptation follows Experiment 3's FP8 PLE adapter (FN_FP8PLE_build_R1/build/adapt_streaming_uploader.py), which
built the 51 GB Model foysalemonshanto/qwen38-flash-next-fp8ple-overlay. Only these change: the config, the file list,
the account check (refuse any user other than foysalemonshanto), the text, and the fail-closed final verification
(FP8 PLE review R1). The stream / resume / blob / worker / publish / token hand-off code stays byte-identical to the
reference, and this script asserts that.

What is uploaded: every file of the pinned revision EXCEPT
  - model-00001-of-00011.safetensors (102.4 GB): it holds only the BF16 PLE table. The index weight_map and the shard
    header both list exactly 128 tensors `...layers.1.ple.ple_embedding.ngram_embedding.shard_N.weight` (BF16), and
    nothing else is in it. Checked in --fetch and recorded in the manifest.
  - .gitattributes (git metadata; the reference and the FP8 PLE adapter dropped it too).
The files are the original HF bytes: no fixkeys, and the index is unchanged. The GPU notebook merges them with the
10 FP8 PLE shards of the overlay Model and the fixkeys config at run time.

usage
  python build_e25_klee100_upload.py --fetch   network: HF API sha + tree + HEAD per file (x-linked-size /
                                               x-linked-etag = LFS sha256) + small-file bytes (sha256, git blob
                                               sha1 vs tree) + the shard-1 safetensors header -> manifest JSON
  python build_e25_klee100_upload.py           offline: manifest -> KLEE100-UPLOAD.ipynb + asserts + syntax check
"""
import contextlib
import hashlib
import io
import json
import re
import sys
import time
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
NB_DIR = HERE.parent
EXPS = HERE.parents[2]
REF = (EXPS / "Experiment 3 - Flash-Next Feasibility (FN)" / "Experiment Notebooks" / "Build" / "FN_FP8PLE_build_R1"
       / "reference" / "NVIDIA_FlashNext_NVFP4_01_upload_on_kaggle.ipynb")
REF_SHA256 = "c4b7d1bd569e7a56819c0d3cb47118b8465dd8e4f3c5a66851f56144a1463ea9"
MAN_PATH = HERE / "klee100_upload_manifest.json"
DST = NB_DIR / "KLEE100-UPLOAD.ipynb"
FETCH_DIR = Path("D:/ARC AGI 2/_scratch/klee100_upload_fetch_2026-10-31")   # small HF files: hashed, never executed

REPO = "klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
REV = "600715b31bed3833fb4a51d9ea5d684ea2c7f32c"   # HF main on 2026-10-31 (lastModified 2026-09-24); = PLE rows rev
PLE_SHARD = "model-00001-of-00011.safetensors"
PLE_KEY = re.compile(r"^model\.language_model\.layers\.1\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.weight$")
GIT_META = ".gitattributes"
BAD_CHARS = re.compile(r"[\[\]=,&@+]")             # Kaggle rejects these in file names (memory: upload rules)
MAX_FILES = 50                                      # Kaggle cap per version, KAGGLE_MIRROR_MANIFEST.json included
ACCOUNT = "foysalemonshanto"
MODEL_SLUG = "qwen38-flash-next-klee100-3bpw-mtp"
MODEL_TITLE = "Qwen3.8 Flash-Next klee100 AutoRound 3bpw MTP"
INSTANCE_SLUG = "autoround-3bpw-mtp-nople"
OVERLAY_MODEL = "foysalemonshanto/qwen38-flash-next-fp8ple-overlay"


# ===================================================================================================== fetch (network)
def fetch():
    import requests
    s = requests.Session()
    api = s.get(f"https://huggingface.co/api/models/{REPO}", timeout=60).json()
    if api["sha"] != REV:
        raise SystemExit(f"HF main moved: {api['sha']} != pinned {REV}. Review the new revision, then update REV.")
    tree = s.get(f"https://huggingface.co/api/models/{REPO}/tree/{REV}?recursive=1&expand=1", timeout=60).json()
    FETCH_DIR.mkdir(parents=True, exist_ok=True)
    files = {}
    for e in sorted(tree, key=lambda e: e["path"]):
        if e["type"] != "file":
            raise SystemExit(f"unexpected tree entry {e}")
        name, size, lfs = e["path"], e["size"], e.get("lfs")
        url = f"https://huggingface.co/{REPO}/resolve/{REV}/{name}"
        h = s.head(url, allow_redirects=False, timeout=60)
        assert h.headers["X-Repo-Commit"] == REV, (name, h.headers.get("X-Repo-Commit"))
        xsize, xetag = int(h.headers["X-Linked-Size"]), h.headers["X-Linked-ETag"].strip('"')
        assert xsize == size, (name, xsize, size)
        row = {"size": size, "lfs": bool(lfs), "git_oid": e["oid"], "x_linked_size": xsize, "x_linked_etag": xetag}
        if lfs:
            assert lfs["size"] == size and lfs["oid"] == xetag, (name, lfs, xetag)
            row["sha256"] = xetag                   # LFS oid = sha256 of the bytes
            row["sha256_source"] = "x-linked-etag (LFS oid), equal to tree lfs.oid"
        else:
            data = s.get(url, timeout=120).content
            assert len(data) == size, (name, len(data), size)
            assert hashlib.sha1(b"blob %d\0" % size + data).hexdigest() == e["oid"] == xetag, name
            (FETCH_DIR / name).write_bytes(data)
            row["sha256"] = hashlib.sha256(data).hexdigest()
            row["sha256_source"] = "sha256 of the downloaded bytes (git blob sha1 == tree oid == x-linked-etag)"
        files[name] = row
        print(f"  {name:42} {size:>14,}  {row['sha256'][:16]}")

    # the index (LFS) and the shard-1 header decide that shard 1 is the BF16 PLE table and nothing else
    idx_bytes = s.get(f"https://huggingface.co/{REPO}/resolve/{REV}/model.safetensors.index.json", timeout=300).content
    assert hashlib.sha256(idx_bytes).hexdigest() == files["model.safetensors.index.json"]["sha256"]
    (FETCH_DIR / "model.safetensors.index.json").write_bytes(idx_bytes)
    wm = json.loads(idx_bytes)["weight_map"]
    in_index = sorted(k for k, v in wm.items() if v == PLE_SHARD)
    shard_url = f"https://huggingface.co/{REPO}/resolve/{REV}/{PLE_SHARD}"
    n = int.from_bytes(s.get(shard_url, headers={"Range": "bytes=0-7"}, timeout=60).content, "little")
    hdr_bytes = s.get(shard_url, headers={"Range": f"bytes=8-{8 + n - 1}"}, timeout=60).content
    assert len(hdr_bytes) == n
    hdr = json.loads(hdr_bytes)
    hdr.pop("__metadata__", None)
    in_header = sorted(hdr)
    dtypes = sorted({v["dtype"] for v in hdr.values()})
    data_bytes = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in hdr.values())
    non_ple = [k for k in in_header if not PLE_KEY.match(k)]
    ple_elsewhere = sorted({v for k, v in wm.items() if "ngram_embedding" in k and v != PLE_SHARD})
    assert in_index == in_header, "index and header disagree on shard 1"
    assert not non_ple, f"shard 1 holds non-PLE tensors, it must be uploaded: {non_ple[:5]}"
    assert not ple_elsewhere, ple_elsewhere
    assert 8 + n + data_bytes == files[PLE_SHARD]["size"]

    tensor_shards = sorted(set(wm.values()))
    for name, row in files.items():
        if name == PLE_SHARD:
            row["role"] = "excluded_bf16_ple"
        elif name == GIT_META:
            row["role"] = "excluded_git_metadata"
        else:
            row["role"] = "upload"
    up = {k: v for k, v in files.items() if v["role"] == "upload"}
    man = {
        "repo": REPO,
        "revision": REV,
        "hf_last_modified": api.get("lastModified"),
        "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "method": "HF API sha == revision; tree?expand=1 at the revision; HEAD /resolve/<rev>/<file> per file "
                  "(X-Repo-Commit == revision, X-Linked-Size == size, X-Linked-ETag == LFS oid or git blob sha1); "
                  "non-LFS files downloaded and hashed",
        "files": files,
        "ple_shard_check": {
            "shard": PLE_SHARD,
            "tensors_in_index": len(in_index),
            "tensors_in_header": len(in_header),
            "header_equals_index": True,
            "dtypes": dtypes,
            "non_ple_tensors": non_ple,
            "ple_tensors_in_other_shards": ple_elsewhere,
            "header_len": n,
            "tensor_data_bytes": data_bytes,
            "example": in_header[0],
            "index_keys_sha256": hashlib.sha256("\n".join(in_index).encode()).hexdigest(),
        },
        "index_tensor_shards": tensor_shards,
        "upload_file_count": len(up),
        "upload_total_bytes": sum(v["size"] for v in up.values()),
        "hf_total_bytes": sum(v["size"] for v in files.values()),
    }
    MAN_PATH.write_bytes((json.dumps(man, indent=1) + "\n").encode("utf-8"))
    print(f"wrote {MAN_PATH}: {len(up)} upload files, {man['upload_total_bytes']:,} bytes")


# ====================================================================================================== build (offline)
def build():
    assert hashlib.sha256(REF.read_bytes()).hexdigest() == REF_SHA256, "reference uploader changed"
    man = json.loads(MAN_PATH.read_text(encoding="utf-8"))
    assert man["repo"] == REPO and man["revision"] == REV
    UP = {n: v for n, v in man["files"].items() if v["role"] == "upload"}
    TOTAL = sum(v["size"] for v in UP.values())
    roles = {n: v["role"] for n, v in man["files"].items() if v["role"] != "upload"}
    assert roles == {PLE_SHARD: "excluded_bf16_ple", GIT_META: "excluded_git_metadata"}, roles
    pc = man["ple_shard_check"]
    assert pc["tensors_in_index"] == pc["tensors_in_header"] == 128 and not pc["non_ple_tensors"]
    assert not pc["ple_tensors_in_other_shards"] and pc["dtypes"] == ["BF16"]
    # every other tensor shard of the index is uploaded; nothing needed is dropped
    assert set(man["index_tensor_shards"]) - {PLE_SHARD} <= set(UP), set(man["index_tensor_shards"]) - set(UP)
    assert TOTAL == man["upload_total_bytes"] and len(UP) == man["upload_file_count"]
    assert len(UP) + 1 <= MAX_FILES, len(UP)                          # + KAGGLE_MIRROR_MANIFEST.json
    assert not [n for n in UP if BAD_CHARS.search(n) or "/" in n], "unsafe or nested file name"
    assert all(re.fullmatch(r"[0-9a-f]{64}", v["sha256"]) for v in UP.values())
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]{4,49}", MODEL_SLUG) and 5 <= len(MODEL_TITLE) <= 50
    assert re.fullmatch(r"[a-z0-9][a-z0-9-]{1,49}", INSTANCE_SLUG)
    shards = sorted(n for n in UP if n.endswith(".safetensors"))
    assert shards == [f"model-{i:05d}-of-00011.safetensors" for i in range(2, 12)] + [
        "mtp-model-00001-of-00002.safetensors", "mtp-model-00002-of-00002.safetensors"], shards
    PLE_GB = man["files"][PLE_SHARD]["size"] / 1e9
    n_main = sum(n.startswith("model-0") for n in shards)

    nb = json.loads(REF.read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in nb["cells"]]
    ref_cells = list(cells)
    assert len(cells) == 6

    def rep(i, a, b):
        assert cells[i].count(a) == 1, (i, a[:80], cells[i].count(a))
        cells[i] = cells[i].replace(a, b)

    # ------------------------------------------------------------------------------------------- cell 0: markdown
    cells[0] = f"""# klee100 Qwen3.8-Flash-Next AutoRound 3bpw + MTP (without the BF16 PLE shard) -> private Kaggle Model (run this ON Kaggle)

**Notebook settings: Accelerator = None (CPU), Internet = ON.** No GPU quota is used. Utility notebook: no
competition data, never submitted.

### Before running: attach two Kaggle Secrets, start the run from the editor
Add-ons -> Secrets -> attach `KAGGLE_USERNAME` and `KAGGLE_KEY` (account **{ACCOUNT}**), then
**Save Version -> Save & Run All** from the notebook editor (a run started through the Kaggle API could not reach
the secrets service on 2026-09-26). The account check refuses any other user before any transfer.

**Private storage:** the Model is private and needs about {TOTAL / 1e9:.1f} GB of free private Kaggle storage on
{ACCOUNT}. The instance is created only after the transfer, so a full quota fails at that step (E11-01 on
2026-10-15 streamed 19.5 GB and then failed on the quota).

### How it works (no disk): the same tested code as the ARC-AGI-3 NVIDIA upload (132.7 GB, 2026-09-26) and the FP8 PLE overlay (51.2 GB, 2026-10-05)
Every file streams straight from Hugging Face into Kaggle's resumable upload URL in 256 MiB chunks (RAM only,
3 workers). Each file is hashed while it streams and compared with the pinned Hugging Face tree. The model
instance is created **only if all {len(UP)} files match**. A dropped connection resumes from the byte the server
confirmed. The last cell is fail-closed: it raises unless Kaggle lists every file, including the mirror manifest,
with the expected size.

### What gets published ({TOTAL / 1e9:.2f} GB, {len(UP)} files + `KAGGLE_MIRROR_MANIFEST.json`)
`{REPO}` at revision `{REV[:10]}`, byte-identical. **No fixkeys:** config.json, quantization_config.json and the
index are the original files.
- {n_main} main shards `model-00002..00011-of-00011.safetensors`: AutoRound SignRoundV2, auto_round:auto_gptq mixed bits.
  Routed experts are W3A16G128 on layers 0-11 and 36-47 and W2A16G64 on layers 12-35, the main projections are
  W8A16G128, and the sensitive paths are BF16.
- 2 MTP shards `mtp-model-0000{{1,2}}-of-00002.safetensors` (W4A16G128 experts).
- `model.safetensors.index.json`, unchanged. Its 128 PLE `ngram_embedding` entries still point to the shard that is
  not uploaded.
- `config.json`, `quantization_config.json`, `generation_config.json`.
- Tokenizer, processor and template files: `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`,
  `chat_template.jinja`, and the processor / preprocessor / video preprocessor configs.
- `build_info.json`, `mtp_pack_plan.json`, the quality / serving / source-validation / validation reports, LICENSE
  and README.md.

**Not uploaded:**
- `{PLE_SHARD}` ({PLE_GB:.1f} GB): the BF16 PLE table. It holds exactly 128 `ngram_embedding` tensors and nothing
  else, by both the index and the shard header.
- `.gitattributes`.

At run time the GPU notebook merges these files with the 10 FP8 PLE shards of `{OVERLAY_MODEL}` and the fixkeys
config.

License: Qwen Community License 1.0 (LICENSE file included). Kaggle label "Other (specified in description)", the
same label the NVIDIA mirror and the FP8 PLE overlay were accepted with.

Kaggle path afterwards: `/kaggle/input/models/{ACCOUNT}/{MODEL_SLUG}/pytorch/{INSTANCE_SLUG}/1`

The run is idempotent: if the model instance already exists it only lists and verifies its files.
"""

    # -------------------------------------------------------------------------------------------- cell 1: config
    i = cells[1].index('HF_REPO = "nvidia/Qwen3.8-Flash-Next-NVFP4"')
    j = cells[1].index('PLACEHOLDER_DIR = ')
    cells[1] = cells[1][:i] + f'''HF_REPO = "{REPO}"
HF_REVISION = "{REV}"   # pinned 2026-10-31 (HF main, last modified {man.get("hf_last_modified", "")[:10]})
HF_URL = "https://huggingface.co/{{repo}}/resolve/{{rev}}/{{name}}"
KAGGLE_MODEL_SLUG = "{MODEL_SLUG}"
KAGGLE_MODEL_TITLE = "{MODEL_TITLE}"
KAGGLE_FRAMEWORK = "pyTorch"          # appears as "pytorch" in the /kaggle/input path
KAGGLE_INSTANCE_SLUG = "{INSTANCE_SLUG}"
LICENSE_NAME = "Other (specified in description)"   # Qwen Community License 1.0; label accepted for the NVIDIA mirror
''' + cells[1][j:]
    i = cells[1].index("# name -> (bytes, hash kind, hash).")
    j = cells[1].index('print("python"')
    rows = "\n".join(f"    {n!r}: ({v['size']}, 'sha256', {v['sha256']!r})," for n, v in sorted(UP.items()))
    cells[1] = cells[1][:i] + f'''# name -> (bytes, hash kind, hash). "sha256" = LFS oid (x-linked-etag) for LFS files, sha256 of the content for
# the small files (the streamer always computes sha256). Taken from the Hugging Face tree + HEAD at HF_REVISION.
# Not uploaded: {PLE_SHARD} (BF16 PLE table only) and .gitattributes.
EXPECTED_FILES = {{
{rows}
}}
EXPECTED_TOTAL_BYTES = {TOTAL:_}
NOT_UPLOADED = set()
assert sum(v[0] for v in EXPECTED_FILES.values()) == EXPECTED_TOTAL_BYTES
assert len(EXPECTED_FILES) == {len(UP)}
''' + cells[1][j:]
    rep(1, 'PLACEHOLDER_DIR = Path("/tmp/nv-instance-placeholders")',
        'PLACEHOLDER_DIR = Path("/tmp/klee100-instance-placeholders")')

    # ----------------------------------------------------------- cell 2: refuse a wrong account before any transfer
    rep(2, 'api = KaggleApi()\napi.authenticate()\n',
        f'api = KaggleApi()\napi.authenticate()\nif KAGGLE_USERNAME != "{ACCOUNT}":\n'
        f'    raise RuntimeError(f"secrets belong to {{KAGGLE_USERNAME!r}}, expected {ACCOUNT} - nothing transferred")\n')

    # -------------------------------------------------------------------- cell 3: manifest notes; keep the bytes
    i = cells[3].index('        "serving_notes": {')
    j = cells[3].index('    TOKENS["KAGGLE_MIRROR_MANIFEST.json"]')
    ple = man["files"][PLE_SHARD]
    cells[3] = cells[3][:i] + f'''        "serving_notes": {{
            "what": "klee100 3bpw-MTP checkpoint without its BF16 PLE shard; merge at run time with the 10 "
                    "model-plefp8 shards of {OVERLAY_MODEL} and the fixkeys config",
            "not_uploaded": {{
                "{PLE_SHARD}": {{"bytes": {ple["size"]}, "sha256": "{ple["sha256"]}",
                    "why": "BF16 PLE table only: 128 ple.ple_embedding.ngram_embedding tensors, nothing else"}},
                ".gitattributes": "git metadata",
            }},
            "index": "original model.safetensors.index.json; its 128 ngram_embedding entries point to "
                     "{PLE_SHARD}, which the runtime graft replaces with the FP8 PLE overlay",
            "configs": "original config.json and quantization_config.json (no fixkeys applied)",
            "quantization": "AutoRound SignRoundV2, auto_round:auto_gptq mixed: routed experts W3A16G128 (layers "
                            "0-11, 36-47) and W2A16G64 (layers 12-35), main projections W8A16G128, BF16 sensitive "
                            "paths, MTP experts W4A16G128",
            "runtime": "Merlin v3.0.0 (humming kernels for the W2/W3 experts, Marlin for W4/W8), fixkeys overlay",
            "license": "Qwen Community License 1.0 (see LICENSE)",
        }},
    }}
''' + cells[3][j:]
    rep(3, '''    TOKENS["KAGGLE_MIRROR_MANIFEST.json"] = upload_bytes(
        "KAGGLE_MIRROR_MANIFEST.json", (json.dumps(mirror_manifest, indent=1) + "\\n").encode("utf-8"))''',
        '''    MANIFEST_BYTES = (json.dumps(mirror_manifest, indent=1) + "\\n").encode("utf-8")
    TOKENS["KAGGLE_MIRROR_MANIFEST.json"] = upload_bytes("KAGGLE_MIRROR_MANIFEST.json", MANIFEST_BYTES)''')

    # ------------------------------------------------------------------------- cell 4: descriptive metadata only
    i = cells[4].index('            "subtitle":')
    j = cells[4].index('            "licenseName": LICENSE_NAME,\n        }, indent=2))\n        created')
    cells[4] = cells[4][:i] + f'''            "subtitle": "klee100 AutoRound 3bpw MTP of Qwen3.8-Flash-Next, without the BF16 PLE shard",
            "isPrivate": True,
            "description": (
                f"Private mirror of {{HF_REPO}} (revision {{HF_REVISION}}) for offline Kaggle inference, without "
                "{PLE_SHARD} (the BF16 PLE table): {n_main} main shards + 2 MTP shards + index + configs + "
                "tokenizer files + reports, byte-identical. Merge with the FP8 PLE shards of "
                "{OVERLAY_MODEL}. License: Qwen Community License 1.0 (LICENSE in the files)."
            ),
''' + cells[4][j:]
    i = cells[4].index('        "overview": (')
    j = cells[4].index('        "licenseName": LICENSE_NAME,\n        "fineTunable"')
    cells[4] = cells[4][:i] + f'''        "overview": (f"{{HF_REPO}} at {{HF_REVISION}} without {PLE_SHARD} (BF16 PLE), uploaded "
                     "byte-identical (hashes in KAGGLE_MIRROR_MANIFEST.json). {n_main} main + 2 MTP shards, original "
                     "index and configs (no fixkeys)."),
        "usage": ("Symlink these files together with the 10 model-plefp8 shards of {OVERLAY_MODEL} into one "
                  "folder, graft the PLE entries of the index and apply the fixkeys config, then serve with Merlin "
                  "v3.0.0. License: Qwen Community License 1.0 (LICENSE)."),
''' + cells[4][j:]

    # ------------------------------------------------------------------------- cell 5: fail closed (FP8 PLE R1)
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
            nb["cells"][k]["outputs"] = []
            nb["cells"][k]["execution_count"] = None
    out = (json.dumps(nb, indent=1, ensure_ascii=False) + "\n").encode("utf-8")
    check(out, ref_cells, UP)
    DST.write_bytes(out)
    print(f"wrote {DST}\n  sha256 {hashlib.sha256(out).hexdigest()}  {len(out):,} bytes")
    print(f"  {len(UP)} files + manifest, {TOTAL:,} bytes; model {ACCOUNT}/{MODEL_SLUG}/pytorch/{INSTANCE_SLUG}/1")


# ====================================================================================================== checks (quick)
def check(out, ref, UP):
    new = ["".join(c["source"]) for c in json.loads(out)["cells"]]
    assert b"\r" not in out
    # 1. syntax of every code cell
    for k in range(1, 6):
        compile(new[k], f"cell{k}", "exec")
    # 2. stream / resume / blob / workers / publish / token hand-off code identical to the reference
    def block(src, start, end):
        i = src.index(start)
        return src[i:src.index(end, i)]
    for k, start, end in [(3, "class _HFReader:", "TOKENS, HASHES = {}, {}"),
                          (3, "TOKENS, HASHES = {}, {}", "    mirror_manifest = {"),
                          (4, "# Publish: parent model", '            "subtitle":'),
                          (4, '            "licenseName": LICENSE_NAME,\n        }, indent=2))', '        "overview": ('),
                          (4, '        "licenseName": LICENSE_NAME,\n        "fineTunable"', "\n\x00")]:
        if end == "\n\x00":
            a, b = new[k][new[k].index(start):], ref[k][ref[k].index(start):]
        else:
            a, b = block(new[k], start, end), block(ref[k], start, end)
        assert a == b, (k, start)
    for k, start, end in [(1, "import hashlib", "HF_REPO = "), (1, "CHUNK = ", "# name ->"),
                          (1, 'print("python"', "\n\x01"), (3, "    mirror_manifest = {", '        "serving_notes": {'),
                          (5, "listed = {}", "missing = sorted"), (5, "missing = sorted", "wrong = {")]:
        if end == "\n\x01":
            assert new[k][new[k].index(start):] == ref[k][ref[k].index(start):], (k, start)
        else:
            assert block(new[k], start, end) == block(ref[k], start, end), (k, start)
    assert new[2].replace('if KAGGLE_USERNAME != "foysalemonshanto":\n    raise RuntimeError(f"secrets belong to '
                          '{KAGGLE_USERNAME!r}, expected foysalemonshanto - nothing transferred")\n', "") == ref[2]
    # 3. cell 1 executes; EXPECTED_FILES == manifest upload set
    G = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(new[1].replace("import requests", "requests = None")
             .replace('shutil.disk_usage("/tmp")', 'shutil.disk_usage(".")'), G)
    assert G["EXPECTED_FILES"] == {n: (v["size"], "sha256", v["sha256"]) for n, v in UP.items()}
    assert G["HF_REVISION"] == REV and G["KAGGLE_MODEL_SLUG"] == MODEL_SLUG and G["NOT_UPLOADED"] == set()
    assert PLE_SHARD not in G["EXPECTED_FILES"] and GIT_META not in G["EXPECTED_FILES"]
    assert G["LICENSE_NAME"] == "Other (specified in description)"
    # 4. final cell is fail-closed (fake Kaggle listing)
    def verify(listing, manifest_bytes=b"x" * 100):
        g = {"EXPECTED_FILES": G["EXPECTED_FILES"], "NOT_UPLOADED": set(), "INSTANCE_REF": "u/m/pytorch/i",
             "time": types.SimpleNamespace(sleep=lambda s: None)}
        if manifest_bytes is not None:
            g["MANIFEST_BYTES"] = manifest_bytes
        files = [types.SimpleNamespace(name=n, size=s) for n, s in listing.items()]
        g["api"] = types.SimpleNamespace(
            model_instance_version_files=lambda ref, page_size: types.SimpleNamespace(files=files))
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(new[5], g)
            return "PASS"
        except RuntimeError:
            return "RAISE"
    full = {n: v[0] for n, v in G["EXPECTED_FILES"].items()} | {"KAGGLE_MIRROR_MANIFEST.json": 100}
    drop = lambda k: {a: b for a, b in full.items() if a != k}
    cases = [(full, b"x" * 100, "PASS"), (drop("model-00007-of-00011.safetensors"), b"x" * 100, "RAISE"),
             (full | {"mtp-model-00002-of-00002.safetensors": 1}, b"x" * 100, "RAISE"),
             (drop("tokenizer.json"), b"x" * 100, "RAISE"), (drop("KAGGLE_MIRROR_MANIFEST.json"), b"x" * 100, "RAISE"),
             (full | {"KAGGLE_MIRROR_MANIFEST.json": 7}, b"x" * 100, "RAISE"),
             (full | {"KAGGLE_MIRROR_MANIFEST.json": 4321}, None, "PASS"),
             (drop("KAGGLE_MIRROR_MANIFEST.json"), None, "RAISE")]
    for listing, mb, want in cases:
        assert verify(listing, mb) == want, (sorted(set(full) - set(listing)), want)
    print("checks: syntax OK; stream/resume/publish code == reference; EXPECTED_FILES == manifest; "
          f"final cell fail-closed ({len(cases)} fake listings)")


if __name__ == "__main__":
    if "--fetch" in sys.argv:
        fetch()
    build()
