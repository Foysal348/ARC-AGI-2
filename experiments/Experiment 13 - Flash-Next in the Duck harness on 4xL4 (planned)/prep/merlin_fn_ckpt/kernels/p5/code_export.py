# E13 FN Merlin part 5/5: halt95/Qwen3.8-Flash-Next-W4A16-Merlin files into /kaggle/working, every file verified
# (size + sha256, embedded at build time). CPU kernel, internet ON, public HF repo (no token, no secrets).
import hashlib, importlib.metadata, json, os, re, shutil, struct, subprocess, sys, tarfile, time, traceback
import urllib.error, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

for _k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_ENABLE_HF_TRANSFER"):
    os.environ.pop(_k, None)            # public repo: never use (or print) a token; no offline flags
OUT = Path(os.environ.get("P1_OUT", "/kaggle/working"))
TMP = Path(os.environ.get("P1_TMP", "/tmp/e13_fn_part5"))
MAX_FILE = int(os.environ.get("P1_MAX_FILE_BYTES", "0"))   # 0 = full run; >0 = local test, skip bigger files
os.environ.update({"HF_HOME": str(TMP / "hf_home"), "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                   "HF_XET_HIGH_PERFORMANCE": "1", "HF_XET_CHUNK_CACHE_SIZE_BYTES": "0",
                   "HF_HUB_DOWNLOAD_TIMEOUT": "120", "HF_HUB_ETAG_TIMEOUT": "60"})
REPO, REV = "halt95/Qwen3.8-Flash-Next-W4A16-Merlin", "cec53e9f90e5995b81691604ec0b32191debf797"
EXPECTED = json.loads(r"""{
"model-00014-of-00017.safetensors": [
5175581416,
"d4a5cc91fd8d156fdd5b429c46341802226bb772b7f3bc8e9be3360b7ad77bcd"
],
"model-00015-of-00017.safetensors": [
4988975312,
"0d053ba20af9958ece7e93d3503c32a1e7f8b52a5ab950711ea584480d3b9bc6"
],
"model-00017-of-00017.safetensors": [
1284526608,
"fdbe0ca6b1bfab1dece276c48efadbc7d24ebb4c2619497c19db47b400f31782"
]
}""")          # name -> [size, sha256]
MERLIN_CKPT_SHA256 = """# SHA-256 of the 38 checkpoint files of halt95/Qwen3.8-Flash-Next-W4A16-Merlin (Hugging Face revision cec53e9f90e5995b81691604ec0b32191debf797, 2026-10-12);
# run 'sha256sum -c' inside the checkpoint directory. README.md (the model card) is not listed: card edits change it.
# Generated from the Hugging Face file metadata (large files: the LFS sha256; small files: downloaded and hashed); the method reproduces the
# Swift 1.5 list of that package on all files except the card.
465dcafcac7d3542a0cfc150fc550e0b7228c22ee68c8f1f81db942e7c658cb4  LICENSE
c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041  chat_template.jinja
64f990cd49a91c58c2aa77ff0ba671ac2c6e348d49418f0b65acfd2e8f37a442  config.json
6ef70d6670f24f03a9041724c7663ef26bc1e678d10f79c93fe11c192ba14b7d  generation_config.json
6497ab40abcba6aa6fa39d9bf185f3cabf243cd855ada8cbcf618c07aa79fa77  model-00001-of-00017.safetensors
78d84bc0eb7357f9294d9bf022cfc981d5ca507ef72c06acb71f7c445fa3fb05  model-00003-of-00017.safetensors
496b86777197c54a08a855c67177b79d7ff4c8aeb3288adb54c2971125a301bf  model-00004-of-00017.safetensors
d526e53a56da19a58e88951f22d2405b98535d83932a767b1b858c073787f32e  model-00005-of-00017.safetensors
286592cfc24271527614a66d94d5d3f7fa3bfa7577ba218f19789791fa4db942  model-00006-of-00017.safetensors
e9a4ef70da6b3a161711c7ce8572988928fb1eaeb7c562c9392d06af6d602e9f  model-00007-of-00017.safetensors
6c4f6034c549827c83f41f5675a97fbe343b54a292c3cf833351aa46086dacb8  model-00008-of-00017.safetensors
23f08c53d136da449dc5251656eb1bde4d5a2fd92d7ec034190004a7e01ec213  model-00009-of-00017.safetensors
4c6f54068213a339381734f744e95e2747079c9efe0e6bb73dc61a0e956b33b6  model-00010-of-00017.safetensors
b992dd3c832457c9940a434e583753de92381b72d129fd1eaf542b97088e42c1  model-00011-of-00017.safetensors
7750e0493f1a0f331883582c6ea691b3e3ac67b9f38a995871de411991914505  model-00012-of-00017.safetensors
5ce3024a8654fd742200d3808128d371db1e8cdfb9c2a93967d6c2a3f27680a2  model-00013-of-00017.safetensors
d4a5cc91fd8d156fdd5b429c46341802226bb772b7f3bc8e9be3360b7ad77bcd  model-00014-of-00017.safetensors
0d053ba20af9958ece7e93d3503c32a1e7f8b52a5ab950711ea584480d3b9bc6  model-00015-of-00017.safetensors
fdbe0ca6b1bfab1dece276c48efadbc7d24ebb4c2619497c19db47b400f31782  model-00017-of-00017.safetensors
dc2e845b7edd35bda92834fba3626bf7d199e28d6aceac99fee654aade390cfd  model-plefp8-00000.safetensors
899eaa0716e28594468a1389ee58cb065c23907f1270de3831f5ecb0a4f82d56  model-plefp8-00001.safetensors
06fd8a11abf0419a669f89397b8d70dd6ff42d401e6b2a037c65e49704faaf71  model-plefp8-00002.safetensors
c6aaa1fc08e84eced3c8151ac8679ed943888eba0fcef2556963693430f95bd9  model-plefp8-00003.safetensors
d94e97c96d3ea09208614da016960f3f4b429f47a044c898a51b493f42f74ba2  model-plefp8-00004.safetensors
03d5d4792e14a4ab55bae50bb459624b01786f818bbcfcaed1a5d0235af484c6  model-plefp8-00005.safetensors
586cccbc12383021bc9bc02f206d9b19fbd1672373ed9a5d91e3b0ce34c2418f  model-plefp8-00006.safetensors
c4a23bcc10f3cde6b633e82c282a9a518e3a64d433628283e1bc592c94cf3d6c  model-plefp8-00007.safetensors
2dc8098c0d020bff277c9cf499a6b908e17836b70a5f949dfa24793371c9a87e  model-plefp8-00008.safetensors
61de98b89bb79f386795787d7a76827a26f1e292c26edbb0a1b613da455f5a9c  model-plefp8-00009.safetensors
218c67d0c052737b18cc7e82550e5f82a55ab246135c3128248f2910a9dc713e  model.safetensors.index.json
70935e2c3a327d35e947524fa19b2ac007772af75827bd8eedc647e2e508bdb6  model_extra_tensors.safetensors
7f046b00fe9599aa048d99d718d61505b6825c45113cdb3a1d7f51cd994e5da0  mtp-routed-experts-int4.safetensors
3a159dfec9978a186a72ba085e0ad6a050f3968d8b364218d7bd13f5c89381f2  preprocessor_config.json
bfbc24af59a3e73a9cd0653b8d4ae758dfaec4e3e6c15dfb1c6c8ee8d5683c85  processor_config.json
5cbe6ae87ed9d2dc97453a9a7b4967d0e0d32a0e6992ec260dc2c54b96305f67  qsa_kv_scales_262k.json
2371a9d0bed1ff619622ac57aa0e6e5b39688bb834b48d761e90d5834f4cab4b  quantization_config.json
0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3  tokenizer.json
792fa3f0cb88b111e54ef3134c873531008c4df471d108da17903426e308aa7b  tokenizer_config.json
"""
TOTAL_BYTES = 11449083336
MERLIN_TAG, MERLIN_COMMIT, MERLIN_TGZ_SHA256_SEEN = "merlin-v3.0.0", "ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5", ""
MODEL_SUBDIR = "."   # v3: flat output (the v2 Model creation from a subfolder failed; our working Models are flat)
SKIP = {".gitattributes"}   # v2: not downloaded (the v1 Model creation failed); still checked in the remote tree
MD = OUT
# v4: the source tarball stays in /tmp (checked, not in the output): Kaggle unpacks .tar.gz files on Model creation and
# refuses the GitHub tarball's PaxHeader / pax_global_header entries (the v1-v3 "File name contains invalid character"
# failures); P2 ships the engine source
TGZ = TMP / f"merlin-engine-{MERLIN_TAG}.tar.gz"
MANIFEST = OUT / "q27_fn_merlin_part5_manifest.json"
REPORT_PATH = OUT / "q27_fn_merlin_part5_report.json"
UA = "q27-e13-fn-merlin/1.0"
T0 = time.time()
R = {"kernel": "q27-fn-merlin-ckpt-p5", "part": 5, "repo": REPO, "revision": REV, "model_subdir": MODEL_SUBDIR,
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
    drift = sorted(n for n in EXPECTED
                   if n not in remote or remote[n][0] != EXPECTED[n][0]
                   or (remote[n][1] is not None and remote[n][1] != EXPECTED[n][1]))
    R["remote_tree_total_files"] = len(remote)
    if len(remote) != 40:
        R["errors"].append(f"HF tree at the pinned revision has {len(remote)} files, expected 40")
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
        ip = MD / "model.safetensors.index.json"
        wm = json.loads(ip.read_text())["weight_map"] if ip.is_file() else None
        st = {}
        for fn in sorted(n for n in EXPECTED if n.endswith(".safetensors")):
            if (MD / fn).is_file():
                names, end_ok, n = st_check(MD / fn)
                mapped = {k for k, v in wm.items() if v == fn} if wm is not None else set()
                st[fn] = {"tensors": n, "data_end_eq_size": end_ok, "names_eq_weight_map": (names == mapped) if mapped else True,
                          "in_index": bool(mapped)}
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
        R["merlin_src"] = {"ok": True, "skipped": "engine source and wheel come from foysalemonshanto/e23-merlin-env (E23 P2)"}
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
    want_bytes = TOTAL_BYTES - sum(EXPECTED[n][0] for n in SKIP if n in EXPECTED)
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
            "what": "E13 Merlin Flash-Next checkpoint part 5 of 5", "part": 5, "parts": 5, "repo": REPO, "revision": REV, "model_subdir": MODEL_SUBDIR,
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
    raise RuntimeError("E13 FN Merlin part 5 NOT OK: " + "; ".join(R["errors"])[:2000])
