"""Validation of FN-PLEFP8-UPLOAD, runnable from an extracted bundle:  python build/validate_upload.py [--network]
1. the notebook is reproducible from reference/ + adapt_streaming_uploader.py (same sha256)
2. the stream / resume / publish code is identical to the reference (per-function comparison) + per-cell diff stats
3. cell 1 executes; EXPECTED_FILES == manifest overlay
4. the final cell is fail-closed (fake Kaggle listing): missing shard / wrong size / manifest missing or wrong -> raise
5. (--network) the real HF reader returns correct sha256 for the small files and resumes with a Range request"""
import difflib, hashlib, json, subprocess, sys, tempfile, types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "FN-PLEFP8-UPLOAD.ipynb"
REF = ROOT / "reference" / "NVIDIA_FlashNext_NVFP4_01_upload_on_kaggle.ipynb"
cells = lambda p: ["".join(c["source"]) for c in json.load(open(p, encoding="utf-8"))["cells"]]
new, ref = cells(NB), cells(REF)

# 1. reproducible
out = Path(tempfile.mkdtemp()) / "rebuilt.ipynb"
subprocess.run([sys.executable, str(ROOT / "build" / "adapt_streaming_uploader.py"), str(REF), str(out)], check=True, capture_output=True)
assert hashlib.sha256(out.read_bytes()).hexdigest() == hashlib.sha256(NB.read_bytes()).hexdigest()
print("1. notebook rebuilt from reference/ + adapt script: identical sha256")

# 2. untouched code
def block(src, start, end):
    i = src.index(start); return src[i:src.index(end, i)]
for k, start, end in [(3, "class _HFReader:", "TOKENS, HASHES = {}, {}"),          # reader, blob start, resume, stream
                      (3, "    def _one(name):", "    mirror_manifest = {"),          # workers + hash check
                      (4, "    # model_instance_create walks a folder", "\n"),       # token hand-off (first line)
                      (4, "    used = []", "print(f\"instance created")]:            # patched _upload_file + create
    assert block(new[k], start, end) == block(ref[k], start, end), (k, start)
for k in range(1, 6):
    d = [l for l in difflib.unified_diff(ref[k].splitlines(), new[k].splitlines(), lineterm="", n=0)
         if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    print(f"   cell {k}: {len(d)} changed lines")
print("2. stream/resume/blob/publish/token-hand-off code identical to the reference")

# 3. config
G = {}
exec(new[1].replace("import requests", "requests = None"), G)
man = json.loads((ROOT / "build" / "manifest.json").read_text())
ov = {n: v for n, v in man["files"].items() if v["role"] == "overlay"}
assert {n: (v["size"], "sha256", v["sha256"]) for n, v in ov.items()} == G["EXPECTED_FILES"]
assert G["LICENSE_NAME"] == "Other (specified in description)"
print(f"3. EXPECTED_FILES == manifest overlay ({len(ov)} files, {sum(v['size'] for v in ov.values()) / 1e9:.2f} GB)")

# 4. fail-closed verification
def verify(listing, manifest_bytes=b"x" * 100):
    g = {"EXPECTED_FILES": G["EXPECTED_FILES"], "NOT_UPLOADED": set(), "INSTANCE_REF": "u/m/pytorch/i",
         "time": types.SimpleNamespace(sleep=lambda s: None)}
    if manifest_bytes is not None: g["MANIFEST_BYTES"] = manifest_bytes
    files = [types.SimpleNamespace(name=n, size=s) for n, s in listing.items()]
    g["api"] = types.SimpleNamespace(model_instance_version_files=lambda ref, page_size: types.SimpleNamespace(files=files))
    try:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            exec(new[5], g)
        return "PASS"
    except RuntimeError as e:
        return "RAISE"

full = {n: v[0] for n, v in G["EXPECTED_FILES"].items()} | {"KAGGLE_MIRROR_MANIFEST.json": 100}
cases = {
    "all 22 files ok": (full, b"x" * 100, "PASS"),
    "one PLE shard missing": ({k: v for k, v in full.items() if k != "model-plefp8-00007.safetensors"}, b"x" * 100, "RAISE"),
    "PLE shard wrong size": (full | {"model-plefp8-00003.safetensors": 1}, b"x" * 100, "RAISE"),
    "tokenizer missing": ({k: v for k, v in full.items() if k != "tokenizer.json"}, b"x" * 100, "RAISE"),
    "manifest missing": ({k: v for k, v in full.items() if k != "KAGGLE_MIRROR_MANIFEST.json"}, b"x" * 100, "RAISE"),
    "manifest wrong size": (full | {"KAGGLE_MIRROR_MANIFEST.json": 7}, b"x" * 100, "RAISE"),
    "rerun, instance existed, manifest present": (full | {"KAGGLE_MIRROR_MANIFEST.json": 4321}, None, "PASS"),
    "rerun, instance existed, manifest missing": ({k: v for k, v in full.items() if k != "KAGGLE_MIRROR_MANIFEST.json"}, None, "RAISE"),
}
for name, (listing, mb, want) in cases.items():
    got = verify(listing, mb); print(f"   {name:45} -> {got}"); assert got == want, name
print("4. final verification is fail-closed")

# 5. real Hugging Face reader (network)
if "--network" in sys.argv:
    import requests  # noqa: F401
    exec(new[1], G); G["instance_exists"] = True
    sys.modules.setdefault("kagglesdk", types.ModuleType("kagglesdk"))
    exec(new[3], G)
    url = lambda n: G["HF_URL"].format(repo=G["HF_REPO"], rev=G["HF_REVISION"], name=n)
    for name in ["LICENSE", "config.json", "tokenizer_config.json", "chat_template.jinja", "model.safetensors.index.json"]:
        size, _, want = G["EXPECTED_FILES"][name]; r = G["_HFReader"](url(name), 0, size); h = hashlib.sha256(); n = 0
        while n < size:
            b = r.get(); h.update(b); n += len(b)
        assert h.hexdigest() == want, name
        print(f"   {name:32} {n:>10} bytes sha256 OK")
    name = "model-plefp8-00009.safetensors"; size = G["EXPECTED_FILES"][name][0]; start = size - 3 * (8 << 20)
    r = G["_HFReader"](url(name), start, size); n = start
    while n < size: n += len(r.get())
    assert n == size; print(f"   {name}: Range resume from byte {start} reached {n} == size")
    print("5. real HF reader OK")
print("ALL UPLOADER VALIDATION TESTS PASS")
