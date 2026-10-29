"""E23 P1: Kaggle CPU kernel (internet ON, foysal) that downloads the Merlin Qwen3.8-27B W4A16 checkpoint
(HF halt95/Qwen3.8-27B-W4A16-Merlin at the revision release/download-sizes.json of merlin-v3.0.0 pins) and the
merlin-engine source at tag merlin-v3.0.0 into /kaggle/working, verifying every file by size + sha256.
A GPU notebook then attaches the kernel output through kernel_sources (see README.md for the finder).

Writes kernel/q27-e23-p1-merlin-ckpt.ipynb + kernel/kernel-metadata.json. Run: python build_p1_kernel.py
Local test (small files only, real HF + GitHub downloads, no Kaggle):
  P1_OUT=<dir> P1_TMP=<dir> P1_MAX_FILE_BYTES=300000000 python D:/ARC AGI 2/_scratch/e23/p1/p1_code_export.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KERNEL_SLUG = "foysalemonshanto/q27-e23-p1-merlin-ckpt"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
REPO = "halt95/Qwen3.8-27B-W4A16-Merlin"
REV = "7a0af39f63dd4fb0733c23d3ec126762074a503f"
MERLIN_TAG = "merlin-v3.0.0"
MERLIN_TAG_OBJECT = "72780a84c34035d9ec79bd32fea1ff0a9634631f"   # annotated tag object (git ls-remote)
MERLIN_COMMIT = "ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5"       # refs/tags/merlin-v3.0.0^{} (= tarball pax comment)
MERLIN_TGZ_SHA256_SEEN = "1cd121b60ae3f348f0a0416f91385a77a8f5fe9fb645f7e61b4b42e672a48b06"  # codeload, 2026-10-29 (info)
TOTAL_BYTES = 17049970203   # release/download-sizes.json checkpoints["27b"]["bytes"], 15 files

# name -> [size, sha256]. Sizes: HF API tree at REV. sha256: LFS oid (4 LFS files) = checkpoint.sha256 for the
# 11 files that list covers; .gitattributes / LICENSE / NOTICE / README.md hashed from the HF resolve URL at REV.
EXPECTED = {
    ".gitattributes": [1570, "34448b82c17d60fec9b65b1f093c115ddbaadc04beb1b0140b6bfed2e012a930"],
    "LICENSE": [11358, "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"],
    "NOTICE": [817, "53f4fec1e15d3672f7af686374aaa0fbd4521daf8dd53145ce1260f4e26d1e56"],
    "README.md": [10401, "020ef027d13479762e19e9ef3c05ca035d040ddc9e1fc4cf56360a277b1777b0"],
    "chat_template.jinja": [8952, "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041"],
    "config.json": [21743, "c927c334e6d1482c72dec23ec1391c463756871769a3496917e5ed5bf9afab71"],
    "generation_config.json": [214, "56147698c439ec2da22f5820fcf76fa14788b6c0ebdd2b2e073c6cddfdd1ea30"],
    "kv_scales.safetensors": [3584, "bc51185517b9faacee3668faf2ede65c9c0f2725bac21c7d95f1d089c90c4709"],
    "model-mtp.safetensors": [220687640, "fd754bb62c7a653d41f89e1c1c36caaebe4e26630f76355e1c6d8636ff6bcb31"],
    "model.safetensors": [16816202144, "b798cb4bf37790fb923e6c427c56c31769015a47f850e5a6ed4fa65a4628aeed"],
    "model.safetensors.index.json": [210561, "d839722e3d6830f012ec5810eedbb5e2c65e39bf091336b40711febd75339bb8"],
    "preprocessor_config.json": [390, "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"],
    "tokenizer.json": [12809320, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"],
    "tokenizer_config.json": [1124, "66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b"],
    "video_preprocessor_config.json": [385, "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"],
}
# models/qwen38-27b/checkpoint.sha256 at tag merlin-v3.0.0 (raw.githubusercontent.com), verbatim
MERLIN_CKPT_SHA256 = """\
c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041  chat_template.jinja
c927c334e6d1482c72dec23ec1391c463756871769a3496917e5ed5bf9afab71  config.json
56147698c439ec2da22f5820fcf76fa14788b6c0ebdd2b2e073c6cddfdd1ea30  generation_config.json
bc51185517b9faacee3668faf2ede65c9c0f2725bac21c7d95f1d089c90c4709  kv_scales.safetensors
fd754bb62c7a653d41f89e1c1c36caaebe4e26630f76355e1c6d8636ff6bcb31  model-mtp.safetensors
b798cb4bf37790fb923e6c427c56c31769015a47f850e5a6ed4fa65a4628aeed  model.safetensors
d839722e3d6830f012ec5810eedbb5e2c65e39bf091336b40711febd75339bb8  model.safetensors.index.json
27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516  preprocessor_config.json
0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3  tokenizer.json
66e427c470fe580fe8c7b5725d857af23d8417e37fae62667ec698306a19987b  tokenizer_config.json
7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13  video_preprocessor_config.json
"""

# build-time consistency: the Merlin list must agree with EXPECTED, and the sizes must add up to download-sizes.json
_ml = dict(reversed(l.split()) for l in MERLIN_CKPT_SHA256.splitlines() if l.strip())
assert len(_ml) == 11 and all(EXPECTED[k][1] == v for k, v in _ml.items()), "checkpoint.sha256 vs EXPECTED"
assert len(EXPECTED) == 15 and sum(v[0] for v in EXPECTED.values()) == TOTAL_BYTES, "sizes vs download-sizes.json"
assert all(len(v[1]) == 64 for v in EXPECTED.values())

CODE = r'''# E23 P1: Merlin Qwen3.8-27B W4A16 checkpoint + merlin-engine source (tag) into /kaggle/working, every file verified
# (size + sha256, embedded at build time). CPU kernel, internet ON, public HF repo (no token, no secrets).
import hashlib, importlib.metadata, json, os, re, shutil, struct, subprocess, sys, tarfile, time, traceback
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_ENABLE_HF_TRANSFER"):
    os.environ.pop(_k, None)            # public repo: never use (or print) a token; no offline flags
OUT = Path(os.environ.get("P1_OUT", "/kaggle/working"))
TMP = Path(os.environ.get("P1_TMP", "/tmp/e23_p1"))
MAX_FILE = int(os.environ.get("P1_MAX_FILE_BYTES", "0"))   # 0 = full run; >0 = local test, skip bigger files
os.environ.update({"HF_HOME": str(TMP / "hf_home"), "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                   "HF_XET_HIGH_PERFORMANCE": "1", "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
                   "HF_HUB_DOWNLOAD_TIMEOUT": "120", "HF_HUB_ETAG_TIMEOUT": "60"})
REPO, REV = "__REPO__", "__REV__"
EXPECTED = json.loads(r"""__EXPECTED__""")          # name -> [size, sha256]
MERLIN_CKPT_SHA256 = """__MERLIN_CKPT_SHA256__"""
TOTAL_BYTES = __TOTAL_BYTES__
MERLIN_TAG, MERLIN_COMMIT, MERLIN_TGZ_SHA256_SEEN = "__MERLIN_TAG__", "__MERLIN_COMMIT__", "__MERLIN_TGZ_SHA__"
MODEL_SUBDIR = "."   # v3: flat output (the v2 Model creation from a subfolder failed; our working Models are flat)
SKIP = {".gitattributes"}   # v2: not downloaded (the v1 Model creation failed); still checked in the remote tree
MD = OUT
# v4: the source tarball stays in /tmp (checked, not in the output): Kaggle unpacks .tar.gz files on Model creation and
# refuses the GitHub tarball's PaxHeader / pax_global_header entries (the v1-v3 "File name contains invalid character"
# failures); P2 ships the engine source
TGZ = TMP / f"merlin-engine-{MERLIN_TAG}.tar.gz"
MANIFEST = OUT / "q27_e23_p1_manifest.json"
REPORT_PATH = OUT / "p1_report.json"
UA = "q27-e23-p1/1.0"
T0 = time.time()
R = {"kernel": "q27-e23-p1-merlin-ckpt", "repo": REPO, "revision": REV, "model_subdir": MODEL_SUBDIR,
     "local_test_max_file_bytes": MAX_FILE, "files": {}, "errors": [], "warnings": []}


def log(*a):
    print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)


def disk():
    out = {}
    for p in (OUT, TMP):
        try:
            u = shutil.disk_usage(p)
            out[str(p)] = {"free_gb": round(u.free / 1e9, 2), "total_gb": round(u.total / 1e9, 2)}
        except Exception as e:
            out[str(p)] = repr(e)
    return out


def ver(m):
    try:
        return importlib.metadata.version(m)
    except Exception:
        return None


def sha256_file(p):
    with open(p, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def retry(fn, tries, what, pause=5):
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            log(f"retry {what} {i + 1}/{tries}: {e!r}"[:400])
            time.sleep(pause * (i + 1))
    raise last


def http_get(url, headers=None, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": UA, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)   # follows the HF -> CDN / github -> codeload redirects


def dl_hf(name, force):
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(REPO, name, revision=REV, local_dir=str(MD), force_download=force))


def resolve_cdn(name):
    """One /resolve/ call (HF anonymous limit: 3,000 resolver calls / 5 min per IP, and Kaggle egress IPs are shared)
    -> the signed CDN URL the ranges then use. On 429 wait for the RateLimit header's reset time."""
    url = f"https://huggingface.co/{REPO}/resolve/{REV}/{name}"
    for i in range(8):
        try:
            with http_get(url, {"Range": "bytes=0-0"}, timeout=60) as r:
                return r.geturl()
        except urllib.error.HTTPError as e:
            if e.code != 429:
                raise
            hdr = e.headers.get("RateLimit", "") or ""
            wait = next((int(x[2:]) for x in hdr.replace(",", ";").split(";") if x.strip().startswith("t=")
                         and x.strip()[2:].isdigit()), 60)
            log(f"HF 429 on resolve {name}; waiting {wait + 5} s")
            time.sleep(min(wait + 5, 330))
    raise IOError(f"resolve {name}: rate limited 8 times")


def dl_http(name, size):
    """Parallel HTTP Range download (pinned revision, CDN URL resolved once) into a .p1part file, then rename."""
    cdn = {"url": resolve_cdn(name)}
    part = MD / (name + ".p1part")
    part.parent.mkdir(parents=True, exist_ok=True)
    with open(part, "wb") as f:
        f.truncate(size)
    CH = int(os.environ.get("P1_CHUNK_BYTES", str(64 << 20)))   # env override only for the local test

    def get(off):
        end = min(off + CH, size) - 1

        def once():
            try:
                r = http_get(cdn["url"], {"Range": f"bytes={off}-{end}"}, timeout=180)
            except urllib.error.HTTPError as e:
                if e.code in (401, 403, 410):          # signed CDN URL expired: resolve again
                    cdn["url"] = resolve_cdn(name)
                raise
            with r:
                if r.status != 206 and not (r.status == 200 and off == 0 and end == size - 1):
                    raise IOError(f"HTTP {r.status} for range {off}-{end}")
                data = r.read()
            if len(data) != end - off + 1:
                raise IOError(f"short read {len(data)} != {end - off + 1} at {off}")
            with open(part, "r+b") as f:
                f.seek(off)
                f.write(data)
        retry(once, 6, f"{name}@{off}", pause=3)

    with ThreadPoolExecutor(16) as ex:
        list(ex.map(get, range(0, size, CH)))
    final = MD / name
    os.replace(part, final)
    return final


def st_check(p):
    """safetensors header: parse, tensor names, data end == file size."""
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    end = max((v["data_offsets"][1] for v in h.values()), default=0)
    return set(h), 8 + n + end == p.stat().st_size, len(h)


def fetch_merlin_src():
    TMP.mkdir(parents=True, exist_ok=True)
    rec = {"tag": MERLIN_TAG, "commit_expected": MERLIN_COMMIT, "file": TGZ.name}
    urls = [f"https://codeload.github.com/halt95/merlin-engine/tar.gz/refs/tags/{MERLIN_TAG}",
            f"https://github.com/halt95/merlin-engine/archive/refs/tags/{MERLIN_TAG}.tar.gz"]
    got = False
    for u in urls:
        def once():
            with http_get(u, timeout=300) as r, open(TGZ, "wb") as f:
                shutil.copyfileobj(r, f, 8 << 20)
        try:
            retry(once, 4, "merlin tarball")
            rec["method"], got = u, True
            break
        except Exception as e:
            rec.setdefault("url_errors", []).append(f"{u}: {e!r}"[:400])
    if not got:   # last resort: shallow git clone of the tag, then git archive (same pax commit comment)
        g = TMP / "merlin_git"
        shutil.rmtree(g, ignore_errors=True)
        subprocess.run(["git", "clone", "--quiet", "--depth", "1", "--branch", MERLIN_TAG,
                        "https://github.com/halt95/merlin-engine.git", str(g)], check=True, timeout=900)
        subprocess.run(["git", "-C", str(g), "archive", "--format=tar.gz", f"--prefix=merlin-engine-{MERLIN_TAG}/",
                        "-o", str(TGZ), "HEAD"], check=True, timeout=600)
        rec["method"] = "git clone + git archive"
    rec["size"] = TGZ.stat().st_size
    rec["sha256"] = sha256_file(TGZ)
    if rec["sha256"] != MERLIN_TGZ_SHA256_SEEN:
        R["warnings"].append("merlin tarball sha256 differs from the 2026-10-29 codeload bytes (GitHub archives are "
                             "not byte-stable); the commit id check below is the authority")
    with tarfile.open(TGZ) as t:
        rec["commit"] = t.pax_headers.get("comment")
        members = t.getmembers()
        rec["members"] = len(members)
        x = TMP / "merlin_x"
        shutil.rmtree(x, ignore_errors=True)
        t.extractall(x, filter="data")
    rec["commit_ok"] = rec["commit"] == MERLIN_COMMIT
    roots = [p for p in x.iterdir() if p.is_dir()]
    root = roots[0] if len(roots) == 1 else x
    need = ["release/install-env.sh", "release/requirements-0.32.lock", "release/download-sizes.json",
            "release/merlin-wheel.sha256", "merlin_cli.py", "models/qwen38-27b/checkpoint.sha256",
            "models/qwen38-27b/serve.sh", "models/qwen38-27b/serve-tp4.sh", "models/qwen38-27b/VERSION"]
    rec["missing"] = [n for n in need if not (root / n).is_file()]
    norm = lambda s: [l.split() for l in s.replace("\r", "").splitlines() if l.strip()]
    ck = root / "models/qwen38-27b/checkpoint.sha256"
    rec["checkpoint_sha256_matches_embedded"] = ck.is_file() and norm(ck.read_text()) == norm(MERLIN_CKPT_SHA256)
    try:
        ds = json.loads((root / "release/download-sizes.json").read_text())["checkpoints"]["27b"]
        rec["download_sizes_27b"] = ds
        rec["download_sizes_ok"] = ds.get("revision") == REV and ds.get("bytes") == TOTAL_BYTES and ds.get("files") == 15
    except Exception as e:
        rec["download_sizes_ok"] = False
        rec["download_sizes_error"] = repr(e)
    try:
        rec["version"] = (root / "models/qwen38-27b/VERSION").read_text().strip()
        rec["merlin_wheel_sha256"] = (root / "release/merlin-wheel.sha256").read_text().strip()
    except Exception:
        pass
    rec["ok"] = bool(rec["commit_ok"] and not rec["missing"] and rec["checkpoint_sha256_matches_embedded"]
                     and rec["download_sizes_ok"])
    shutil.rmtree(x, ignore_errors=True)
    return rec


try:
    OUT.mkdir(parents=True, exist_ok=True)
    TMP.mkdir(parents=True, exist_ok=True)
    MD.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.unlink(missing_ok=True)
    MANIFEST.unlink(missing_ok=True)
    mem = {}
    try:
        for l in open("/proc/meminfo"):
            k, v = l.split(":", 1)
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = round(int(v.split()[0]) / 1e6, 1)
    except Exception:
        pass
    R["env"] = {"python": sys.version, "huggingface_hub": ver("huggingface_hub"), "hf_xet": ver("hf_xet"),
                "hf_transfer": ver("hf_transfer"), "requests": ver("requests"), "nproc": os.cpu_count(),
                "mem_gb": mem, "disk_start": disk()}
    log("env", json.dumps(R["env"]))

    # 0. capacity: the whole checkpoint lands in OUT (no second copy anywhere)
    # v2: Kaggle Models refuse dot-files (the v1 Model creation failed on .gitattributes, a HF git setting the engine never
    # reads), and the upload page cannot deselect one file: it stays in the remote-tree check but is not downloaded
    want = {n: v for n, v in EXPECTED.items() if (not MAX_FILE or v[0] <= MAX_FILE) and n not in SKIP}
    R["skipped_files"] = sorted(SKIP)
    need = sum(v[0] for v in want.values()) + 100_000_000
    free = shutil.disk_usage(OUT).free
    R["capacity"] = {"need_bytes": need, "free_bytes": free, "ok": free >= need}
    if not R["capacity"]["ok"]:
        raise RuntimeError(f"not enough space in {OUT}: free {free / 1e9:.2f} GB < need {need / 1e9:.2f} GB")

    # 1. remote tree at the pinned revision must equal the embedded list (names, sizes, LFS sha256)
    tree = retry(lambda: json.load(http_get(f"https://huggingface.co/api/models/{REPO}/tree/{REV}?recursive=true",
                                            timeout=60)), 5, "HF tree")
    remote = {e["path"]: [e.get("size"), (e.get("lfs") or {}).get("oid")] for e in tree if e.get("type") == "file"}
    drift = sorted(n for n in set(remote) | set(EXPECTED)
                   if n not in remote or n not in EXPECTED or remote[n][0] != EXPECTED[n][0]
                   or (remote[n][1] is not None and remote[n][1] != EXPECTED[n][1]))
    R["remote_tree"] = {"files": len(remote), "drift": drift, "ok": not drift}
    log("HF tree at", REV[:12], ":", len(remote), "files, drift:", drift)
    if drift:
        raise RuntimeError(f"HF tree at {REV} differs from the embedded list: {drift}")

    # 2. download + verify each file (small first). hf_hub_download (hf_xet) x2, then parallel HTTP Range x2.
    for name in sorted(want, key=lambda n: want[n][0]):
        size, sha = want[name]
        rec = {"size_expected": size, "sha256_expected": sha, "attempts": [], "ok": False}
        R["files"][name] = rec
        for i, method in enumerate(("hf", "hf", "http", "http")):
            t = time.time()
            try:
                p = dl_hf(name, force=i > 0) if method == "hf" else dl_http(name, size)
                dt = time.time() - t
                got_size = p.stat().st_size
                t2 = time.time()
                got = sha256_file(p)
                a = {"method": method, "sec": round(dt, 1), "MBps": round(size / 1e6 / max(dt, 1e-3), 1),
                     "hash_sec": round(time.time() - t2, 1), "size": got_size, "sha256": got,
                     "size_ok": got_size == size, "sha_ok": got == sha}
                rec["attempts"].append(a)
                if got_size == size and got == sha:
                    rec.update(size=got_size, sha256=got, ok=True, method=method, path=str(p.relative_to(OUT)))
                    break
                p.unlink(missing_ok=True)   # never leave a wrong file where the GPU notebook could load it
                log(f"MISMATCH {name} ({method}): size {got_size} sha {got[:16]}; deleted, retrying")
            except Exception as e:
                rec["attempts"].append({"method": method, "sec": round(time.time() - t, 1), "error": repr(e)[:600]})
                log(f"download error {name} ({method}): {e!r}"[:400])
        log(f"{'OK ' if rec['ok'] else 'BAD'} {name} {size} B", json.dumps(rec["attempts"][-1])[:300])
        if not rec["ok"]:
            R["errors"].append(f"{name}: download/verify failed after {len(rec['attempts'])} attempts")

    # 3. structural checks: safetensors headers vs index weight_map; final size re-check
    try:
        idx = json.loads((MD / "model.safetensors.index.json").read_text())
        wm = idx["weight_map"]
        st = {}
        for fn in sorted(set(wm.values())):
            if (MD / fn).is_file():
                names, end_ok, n = st_check(MD / fn)
                mapped = {k for k, v in wm.items() if v == fn}
                st[fn] = {"tensors": n, "data_end_eq_size": end_ok, "names_eq_weight_map": names == mapped}
            else:
                st[fn] = {"skipped": "not downloaded (local test)" if MAX_FILE else "missing"}
        R["safetensors"] = st
        bad = [k for k, v in st.items() if ("skipped" in v and not MAX_FILE)
               or ("tensors" in v and not (v["data_end_eq_size"] and v["names_eq_weight_map"]))]
        if bad:
            R["errors"].append(f"safetensors/index mismatch: {bad}")
        log("safetensors", json.dumps(st))
    except Exception as e:
        R["errors"].append(f"safetensors/index check failed: {e!r}"[:600])
    for name, rec in R["files"].items():
        p = MD / name
        if rec["ok"] and (not p.is_file() or p.stat().st_size != rec["size_expected"]):
            rec["ok"] = False
            R["errors"].append(f"{name}: missing or resized after download")

    # 4. merlin-engine source at the release tag (P2 needs it): tarball, commit id + content checks
    try:
        R["merlin_src"] = fetch_merlin_src()
        log("merlin src", json.dumps(R["merlin_src"])[:1500])
        if not R["merlin_src"]["ok"]:
            R["errors"].append("merlin source checks failed")
    except Exception as e:
        R["merlin_src"] = {"ok": False, "error": repr(e)[:600]}
        R["errors"].append(f"merlin source: {e!r}"[:600])
except Exception as e:
    R["errors"].append(f"fatal: {e!r}"[:800])
    R["traceback"] = traceback.format_exc()[-3000:]
finally:
    # 5. cleanup (hf metadata / partial files), totals, manifest only when everything verified, report always
    shutil.rmtree(MD / ".cache", ignore_errors=True)
    for p in MD.glob("*.p1part"):
        p.unlink(missing_ok=True)
    files_ok = bool(R["files"]) and all(r["ok"] for r in R["files"].values()) and len(R["files"]) == (
        sum(1 for n, v in EXPECTED.items() if (not MAX_FILE or v[0] <= MAX_FILE) and n not in SKIP))
    R["checkpoint_bytes"] = sum(r.get("size", 0) for r in R["files"].values() if r["ok"])
    want_bytes = TOTAL_BYTES - sum(EXPECTED[n][0] for n in SKIP)
    R["checkpoint_bytes_expected"] = want_bytes
    if not MAX_FILE and R["checkpoint_bytes"] != want_bytes:
        R["errors"].append(f"checkpoint bytes {R['checkpoint_bytes']} != {want_bytes} ({TOTAL_BYTES} minus {sorted(SKIP)})")
    known_out = {TGZ.name, MANIFEST.name, REPORT_PATH.name}
    extra = sorted(str(p.relative_to(OUT)) for p in MD.rglob("*") if p.is_file() and p.name not in known_out
                   and (p.name not in EXPECTED or p.name in SKIP))
    if extra:
        R["warnings"].append(f"unexpected files in model dir: {extra[:20]}")
    # v2: every output path must be a name Kaggle Models accept (no dot-files, only [A-Za-z0-9._-] per path part)
    # v3: and no subfolders at all (flat, like our working FP8 / INT4 Models)
    odd = sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file() and (len(p.relative_to(OUT).parts) > 1 or any(
        part.startswith(".") or re.search(r"[^A-Za-z0-9._-]", part) for part in p.relative_to(OUT).parts)))
    # v4: and no archive Kaggle would unpack (it checks every name inside; see TGZ above)
    odd += sorted(str(p.relative_to(OUT)) for p in OUT.rglob("*") if p.is_file()
                  and re.search(r"\.(zip|tar|gz|tgz|bz2|xz|7z)$", p.name, re.I))
    R["odd_names"] = odd
    if odd:
        R["errors"].append(f"output names Kaggle Models refuse: {odd[:20]}")
    R["all_ok"] = files_ok and not R["errors"] and bool((R.get("merlin_src") or {}).get("ok"))
    if R["all_ok"]:
        MANIFEST.write_text(json.dumps({
            "what": "E23 P1 Merlin checkpoint", "repo": REPO, "revision": REV, "model_subdir": MODEL_SUBDIR,
            "files": {n: [r["size"], r["sha256"]] for n, r in sorted(R["files"].items())},
            "total_bytes": R["checkpoint_bytes"], "merlin_src": {"file": TGZ.name, "tag": MERLIN_TAG,
                                                                 "commit": MERLIN_COMMIT}}, indent=1))
    R["output_files"] = len([p for p in OUT.rglob("*") if p.is_file()]) + 1
    R["output_bytes"] = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    R["output_limit_ok"] = R["output_bytes"] < 19.5e9
    if not R["output_limit_ok"]:
        R["all_ok"] = False
        R["errors"].append(f"output {R['output_bytes'] / 1e9:.2f} GB is too close to the 20 GB /kaggle/working cap")
    R["disk_end"] = disk()
    R["seconds"] = round(time.time() - T0, 1)
    REPORT_PATH.write_text(json.dumps(R, indent=1))
    for n, r in sorted(R["files"].items(), key=lambda kv: kv[1]["size_expected"]):
        print(f"{'ok ' if r['ok'] else 'BAD'} {r['size_expected']:>12} {r.get('sha256', '-')[:16]} {n}", flush=True)
    print("merlin_src:", json.dumps({k: (R.get("merlin_src") or {}).get(k) for k in
                                     ("ok", "commit", "commit_ok", "missing", "checkpoint_sha256_matches_embedded",
                                      "download_sizes_ok", "size", "sha256", "method")}), flush=True)
    print("errors:", json.dumps(R["errors"]), "| warnings:", json.dumps(R["warnings"]), flush=True)
    print("ALL OK" if R["all_ok"] else "NOT OK", f"| {R['checkpoint_bytes']} checkpoint bytes |",
          f"output {R['output_bytes'] / 1e9:.2f} GB in {R['output_files']} files |", R["seconds"], "s", flush=True)
if not R["all_ok"]:
    raise RuntimeError("E23 P1 NOT OK: " + "; ".join(R["errors"])[:2000])
'''

subs = {"__REPO__": REPO, "__REV__": REV, "__EXPECTED__": json.dumps(EXPECTED, indent=0, sort_keys=True),
        "__MERLIN_CKPT_SHA256__": MERLIN_CKPT_SHA256, "__TOTAL_BYTES__": str(TOTAL_BYTES),
        "__MERLIN_TAG__": MERLIN_TAG, "__MERLIN_COMMIT__": MERLIN_COMMIT, "__MERLIN_TGZ_SHA__": MERLIN_TGZ_SHA256_SEEN}
for k, v in subs.items():
    assert k in CODE, k
    CODE = CODE.replace(k, v)
assert "__" + "REPO" not in CODE and "__EXP" not in CODE
compile(CODE, "<e23 p1 kernel>", "exec")

nbj = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": CODE.splitlines(True)}],
       "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
KD = HERE / "kernel"
KD.mkdir(exist_ok=True)
NB = KD / "q27-e23-p1-merlin-ckpt.ipynb"
NB.write_bytes(json.dumps(nbj, indent=1).encode("utf-8"))
(KD / "kernel-metadata.json").write_bytes(json.dumps({
    "id": KERNEL_SLUG, "title": "Q27 E23 P1 Merlin Ckpt", "code_file": NB.name, "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
    "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": [],
    "model_sources": [], "docker_image": IMAGE, "machine_shape": "None"}, indent=2).encode("utf-8"))
SCRATCH = Path(r"D:/ARC AGI 2/_scratch/e23/p1")
if SCRATCH.is_dir():   # same code as the notebook cell, for the local small-file test (scratch only, not a deliverable)
    (SCRATCH / "p1_code_export.py").write_bytes(CODE.encode("utf-8"))
assert NB.stat().st_size < 400_000
print("kernel", KD, "| notebook", NB.stat().st_size, "bytes | code", len(CODE), "chars")
