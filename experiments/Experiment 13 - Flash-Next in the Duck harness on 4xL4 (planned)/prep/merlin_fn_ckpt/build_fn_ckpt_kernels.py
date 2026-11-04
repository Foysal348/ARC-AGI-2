"""E13 Merlin Flash-Next checkpoint on Kaggle (user 2026-10-31): five CPU kernels (foysal, internet ON) download
halt95/Qwen3.8-Flash-Next-W4A16-Merlin at the revision merlin-v3.0.0 pins (cec53e9f, 2026-10-12) in parts that fit the
20 GB /kaggle/working output cap; the GPU notebook attaches all five as kernel_sources and merges them with symlinks.

Not downloaded: the 10 model-plefp8-*.safetensors shards (52.0 GB). Their sha256 in Merlin's checkpoint.sha256 equals
the LFS sha256 of albucino/Qwen3.8-Flash-Next-W4A16-FP8PLE, which our Kaggle overlay
foysalemonshanto/qwen38-flash-next-fp8ple-overlay holds (same sizes); the GPU notebook verifies them by sha256 before use.
Also not downloaded: README.md (not in Merlin's sha list: card edits change it) and .gitattributes (Kaggle refuses dot-files).

The kernel code is E23 P1's (prep/P1_merlin_checkpoint/build_p1_kernel.py, Kaggle-proven v4: flat output, no dot-files,
no archives, HF 429 handling, size + sha256 per file, wrong files deleted), with the part-mode changes below asserted
one by one. Run: python build_fn_ckpt_kernels.py  ->  kernels/p1 .. p5 (notebook + kernel-metadata.json each).
"""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
P1_BUILDER = Path(r"D:/ARC AGI 2/experiments/Experiment 23 - Six levers vLLM 0.32 (planned)/prep/P1_merlin_checkpoint/build_p1_kernel.py")
P1_BUILDER_SHA = "2ff414c33d24333a"   # first 16 hex of the template we adapted (checked below; update deliberately)
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
REPO = "halt95/Qwen3.8-Flash-Next-W4A16-Merlin"
REV = "cec53e9f90e5995b81691604ec0b32191debf797"      # merlin-v3.0.0 release/download-sizes.json checkpoints["flash-next"]
SRC = Path(r"D:/ARC AGI 2/_scratch/fn_public")
CKPT_SHA256 = (SRC / "v3_checkpoint.sha256").read_text(encoding="utf-8")     # models/flash-next/checkpoint.sha256 @ tag
TREE = json.loads((SRC / "tree_fn_merlin_cec53e9f.json").read_text(encoding="utf-8"))   # HF API tree at REV

sha = {l.split()[1]: l.split()[0] for l in CKPT_SHA256.splitlines() if l.strip() and not l.startswith("#")}
size = {e["path"]: e["size"] for e in TREE if e.get("type") == "file"}
lfs = {e["path"]: (e.get("lfs") or {}).get("oid") for e in TREE if e.get("type") == "file"}
assert len(sha) == 38 and set(sha) <= set(size), "sha list vs tree"
assert all(lfs[n] in (None, sha[n]) for n in sha), "LFS sha256 vs Merlin's list"
assert sum(size.values()) == 123754055850 and len(size) == 40, "tree vs release/download-sizes.json"
PLE = sorted(n for n in sha if n.startswith("model-plefp8-"))
assert len(PLE) == 10

SMALL = sorted(n for n in sha if not n.endswith(".safetensors"))
PARTS = {
    1: SMALL + ["model-00001-of-00017.safetensors", "model-00003-of-00017.safetensors", "model-00004-of-00017.safetensors",
                "mtp-routed-experts-int4.safetensors", "model_extra_tensors.safetensors"],
    2: ["model-00005-of-00017.safetensors", "model-00006-of-00017.safetensors", "model-00007-of-00017.safetensors"],
    3: ["model-00008-of-00017.safetensors", "model-00009-of-00017.safetensors", "model-00010-of-00017.safetensors"],
    4: ["model-00011-of-00017.safetensors", "model-00012-of-00017.safetensors", "model-00013-of-00017.safetensors"],
    5: ["model-00014-of-00017.safetensors", "model-00015-of-00017.safetensors", "model-00017-of-00017.safetensors"],
}
allp = [n for v in PARTS.values() for n in v]
assert len(allp) == len(set(allp)) and set(allp) == set(sha) - set(PLE), "parts must cover every non-PLE listed file once"
for k, v in PARTS.items():
    assert sum(size[n] for n in v) < 16.0e9, f"part {k} too big for the 20 GB output cap"

# ---- the E23 P1 kernel code, adapted for part mode (every change asserted)
src = P1_BUILDER.read_text(encoding="utf-8")
assert hashlib.sha256(src.encode("utf-8")).hexdigest()[:16] == P1_BUILDER_SHA, "E23 P1 builder changed: re-check the adaptation"
CODE = src[src.index("CODE = r'''") + len("CODE = r'''"): src.index("\n'''\n\nsubs = ")] + "\n"


def once(old, new, label):
    global CODE
    assert CODE.count(old) == 1, f"{label}: {CODE.count(old)}"
    CODE = CODE.replace(old, new)


once("# E23 P1: Merlin Qwen3.8-27B W4A16 checkpoint + merlin-engine source (tag) into /kaggle/working, every file verified",
     "# E13 FN Merlin part __PART__/5: halt95/Qwen3.8-Flash-Next-W4A16-Merlin files into /kaggle/working, every file verified",
     "header")
once('TMP = Path(os.environ.get("P1_TMP", "/tmp/e23_p1"))', 'TMP = Path(os.environ.get("P1_TMP", "/tmp/e13_fn_part__PART__"))', "tmp")
once('MANIFEST = OUT / "q27_e23_p1_manifest.json"', 'MANIFEST = OUT / "q27_fn_merlin_part__PART___manifest.json"', "manifest")
once('REPORT_PATH = OUT / "p1_report.json"', 'REPORT_PATH = OUT / "q27_fn_merlin_part__PART___report.json"', "report")
once('UA = "q27-e23-p1/1.0"', 'UA = "q27-e13-fn-merlin/1.0"', "ua")
once('R = {"kernel": "q27-e23-p1-merlin-ckpt",', 'R = {"kernel": "q27-fn-merlin-ckpt-p__PART__", "part": __PART__,', "kernel name")
# the remote tree holds all 40 files; a part checks only its own names (size + LFS sha256 at the pinned revision)
once("""    drift = sorted(n for n in set(remote) | set(EXPECTED)
                   if n not in remote or n not in EXPECTED or remote[n][0] != EXPECTED[n][0]
                   or (remote[n][1] is not None and remote[n][1] != EXPECTED[n][1]))""",
     """    drift = sorted(n for n in EXPECTED
                   if n not in remote or remote[n][0] != EXPECTED[n][0]
                   or (remote[n][1] is not None and remote[n][1] != EXPECTED[n][1]))
    R["remote_tree_total_files"] = len(remote)
    if len(remote) != __TREE_FILES__:
        R["errors"].append(f"HF tree at the pinned revision has {len(remote)} files, expected __TREE_FILES__")""", "drift")
# safetensors check: only this part's files; names vs the index weight_map when the index is in this part, else header only
once("""        idx = json.loads((MD / "model.safetensors.index.json").read_text())
        wm = idx["weight_map"]
        st = {}
        for fn in sorted(set(wm.values())):
            if (MD / fn).is_file():
                names, end_ok, n = st_check(MD / fn)
                mapped = {k for k, v in wm.items() if v == fn}
                st[fn] = {"tensors": n, "data_end_eq_size": end_ok, "names_eq_weight_map": names == mapped}
            else:
                st[fn] = {"skipped": "not downloaded (local test)" if MAX_FILE else "missing"}""",
     """        ip = MD / "model.safetensors.index.json"
        wm = json.loads(ip.read_text())["weight_map"] if ip.is_file() else None
        st = {}
        for fn in sorted(n for n in EXPECTED if n.endswith(".safetensors")):
            if (MD / fn).is_file():
                names, end_ok, n = st_check(MD / fn)
                mapped = {k for k, v in wm.items() if v == fn} if wm is not None else set()
                st[fn] = {"tensors": n, "data_end_eq_size": end_ok, "names_eq_weight_map": (names == mapped) if mapped else True,
                          "in_index": bool(mapped)}
            else:
                st[fn] = {"skipped": "not downloaded (local test)" if MAX_FILE else "missing"}""", "safetensors")
# the engine source comes from foysalemonshanto/e23-merlin-env (E23 P2): no tarball here
once('        R["merlin_src"] = fetch_merlin_src()',
     '        R["merlin_src"] = {"ok": True, "skipped": "engine source and wheel come from foysalemonshanto/e23-merlin-env (E23 P2)"}',
     "no merlin src")
once("    want_bytes = TOTAL_BYTES - sum(EXPECTED[n][0] for n in SKIP)",
     "    want_bytes = TOTAL_BYTES - sum(EXPECTED[n][0] for n in SKIP if n in EXPECTED)", "skip bytes")
once('            "what": "E23 P1 Merlin checkpoint", "repo": REPO,',
     '            "what": "E13 Merlin Flash-Next checkpoint part __PART__ of 5", "part": __PART__, "parts": 5, "repo": REPO,', "manifest what")
once('    raise RuntimeError("E23 P1 NOT OK: "', '    raise RuntimeError("E13 FN Merlin part __PART__ NOT OK: "', "raise")

out_root = HERE / "kernels"
for k, names in PARTS.items():
    exp = {n: [size[n], sha[n]] for n in names}
    total = sum(v[0] for v in exp.values())
    code = CODE
    subs = {"__PART__": str(k), "__REPO__": REPO, "__REV__": REV, "__EXPECTED__": json.dumps(exp, indent=0, sort_keys=True),
            "__MERLIN_CKPT_SHA256__": CKPT_SHA256.replace('"""', ''), "__TOTAL_BYTES__": str(total),
            "__TREE_FILES__": str(len(size)), "__MERLIN_TAG__": "merlin-v3.0.0",
            "__MERLIN_COMMIT__": "ad42e37feb7611c4ffd24017bf20a7e2dc94c6e5", "__MERLIN_TGZ_SHA__": ""}
    for a, b in subs.items():
        assert a in code, a
        code = code.replace(a, b)
    assert not re.search(r"__[A-Z_]+__", code), re.findall(r"__[A-Z_]+__", code)[:5]
    compile(code, f"<fn merlin part {k}>", "exec")
    kd = out_root / f"p{k}"
    kd.mkdir(parents=True, exist_ok=True)
    slug = f"q27-fn-merlin-ckpt-p{k}"
    nb = kd / f"{slug}.ipynb"
    nbj = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
           "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                        "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    nb.write_bytes(json.dumps(nbj, indent=1).encode("utf-8"))
    (kd / "kernel-metadata.json").write_bytes(json.dumps({
        "id": f"foysalemonshanto/{slug}", "title": f"Q27 FN Merlin Ckpt P{k}", "code_file": nb.name, "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": True,
        "keywords": [], "dataset_sources": [], "kernel_sources": [], "competition_sources": [], "model_sources": [],
        "docker_image": IMAGE, "machine_shape": "None"}, indent=2).encode("utf-8"))
    (kd / "code_export.py").write_bytes(code.encode("utf-8"))      # same code, for the local small-file test
    assert nb.stat().st_size < 400_000
    print(f"part {k}: {len(names)} files, {total / 1e9:.2f} GB | {nb.name} {nb.stat().st_size} B")
print("total downloaded:", round(sum(size[n] for n in allp) / 1e9, 2), "GB | PLE reused from the overlay:",
      round(sum(size[n] for n in PLE) / 1e9, 2), "GB")
(out_root / "parts.json").write_text(json.dumps({"repo": REPO, "revision": REV, "parts": PARTS,
                                                 "ple_from_overlay": {n: [size[n], sha[n]] for n in PLE}}, indent=1), encoding="utf-8")
