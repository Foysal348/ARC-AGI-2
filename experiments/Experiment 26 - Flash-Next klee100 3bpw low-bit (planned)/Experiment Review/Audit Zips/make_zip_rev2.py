"""Build the FLAT GPT audit zip for the revised klee100 GPU load notebook from MANIFEST_rev2.txt.

  python make_zip_rev2.py --dry-run         list every manifest entry (OK / MISSING / STALE), the consistency checks; no zip
  python make_zip_rev2.py                   write E26_klee100_LOAD_audit_rev2_FLAT_<date>.zip next to this script
  python make_zip_rev2.py --allow-inconsistent   write even when a consistency check fails (recorded in the zip)

Every source must exist (REQ) or is reported (OPT). The sha256 of every file goes into 00_ZIP_MANIFEST_SHA256.txt inside the
zip and into <zip>.sha256.json beside it. The zip has no folders. An existing zip of the same name is never overwritten
(a _b / _c ... suffix is used). Standard library only.
"""
import argparse
import datetime
import glob
import hashlib
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "MANIFEST_rev2.txt"
KITOUT_GLOB = "D:/ARC AGI 2/_scratch/e25_kits_out_k100load*"
NB_FLAT = "Q27-XHIGH-E25_KLEE100_LOAD.ipynb"
PUSH_NB = Path("D:/ARC AGI 2/_scratch/push_e25/q27-xhigh-e25-klee100-load__foysalemonshanto/q27-xhigh-e25-klee100-load.ipynb")


def sha(b):
    return hashlib.sha256(b).hexdigest()


def parse_manifest():
    rows = []
    for n, line in enumerate(MANIFEST.read_text(encoding="utf-8").replace("\r\n", "\n").split("\n"), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 3 or parts[0] not in ("REQ", "OPT"):
            raise SystemExit(f"MANIFEST line {n}: expected KIND<TAB>FLAT<TAB>SOURCE, got {line!r}")
        rows.append({"kind": parts[0], "flat": parts[1].strip(), "src": parts[2].strip(), "line": n})
    return rows


def read_zip_member(spec):
    zpath, member = spec[len("zip:"):].split("!", 1)
    zp = Path(zpath)
    if not zp.is_file():
        return None, f"zip not found: {zp}"
    with zipfile.ZipFile(zp) as z:
        if member not in z.namelist():
            return None, f"member {member} not in {zp.name}"
        return z.read(member), None


def kit_folder(kit, nb_sha):
    """The newest e25_kits_out_k100load*/q27-e25-k100load-kit-<kit>/ whose report tested nb_sha; also the stale ones seen."""
    found, stale = [], []
    for d in glob.glob(f"{KITOUT_GLOB}/q27-e25-k100load-kit-{kit}"):
        rp = Path(d) / f"k100load_kit_{kit}_report.json"
        if not rp.is_file():
            continue
        try:
            got = json.loads(rp.read_text(encoding="utf-8")).get("notebook_sha256")
        except Exception as e:  # noqa: BLE001
            stale.append(f"{rp} (unreadable: {e})")
            continue
        (found if got == nb_sha else stale).append((rp.stat().st_mtime, Path(d), got) if got == nb_sha else f"{rp} tested {str(got)[:8]}")
    found.sort(key=lambda t: t[0], reverse=True)
    return (found[0][1] if found else None), stale


def resolve(rows, nb_sha):
    """-> list of {kind, flat, src, data|None, status, note}."""
    explicit = {str(Path(r["src"]).resolve()).lower() for r in rows if not r["src"].startswith(("zip:", "kitout:", "glob:"))}
    kitdirs, out = {}, []
    for r in rows:
        src = r["src"]
        if src.startswith("glob:"):
            hits = [Path(p) for p in sorted(glob.glob(src[len("glob:"):])) if str(Path(p).resolve()).lower() not in explicit]
            if not hits:
                out.append(dict(r, data=None, status="NONE", note="no extra file beyond the explicit entries (fine)"))
            for p in hits:
                out.append(dict(r, flat=p.name if r["flat"] == "*" else r["flat"], src=str(p), data=p.read_bytes(), status="OK", note="glob"))
            continue
        if src.startswith("zip:"):
            data, why = read_zip_member(src)
            out.append(dict(r, data=data, status="OK" if data is not None else "MISSING", note=why or ""))
            continue
        if src.startswith("kitout:"):
            kit, fname = src[len("kitout:"):].split("/", 1)
            if kit not in kitdirs:
                kitdirs[kit] = kit_folder(kit, nb_sha) if nb_sha else (None, ["notebook missing: kit match impossible"])
            d, stale = kitdirs[kit]
            if d is None:
                out.append(dict(r, data=None, status="STALE" if stale else "MISSING",
                                note=("only results for other notebooks: " + "; ".join(stale)) if stale else f"no {KITOUT_GLOB}/q27-e25-k100load-kit-{kit}"))
                continue
            p = d / fname
            out.append(dict(r, src=str(p), data=p.read_bytes() if p.is_file() else None, status="OK" if p.is_file() else "MISSING",
                            note="" if p.is_file() else f"not in {d}"))
            continue
        p = Path(src)
        out.append(dict(r, data=p.read_bytes() if p.is_file() else None, status="OK" if p.is_file() else "MISSING",
                        note="" if p.is_file() else "file not found"))
    return out


def consistency(files):
    """Freshness / identity checks so the zip is not cut from a half-rebuilt state. -> list of (name, ok, detail)."""
    by = {f["flat"]: f["data"] for f in files if f["data"] is not None}
    res = []

    def chk(name, ok, detail=""):
        res.append((name, bool(ok), detail))

    nb, facts = by.get(NB_FLAT), by.get("k100load_build_facts.json")
    fj = json.loads(facts) if facts else {}
    if nb and facts:
        chk("notebook sha == build_facts.notebook_sha256", sha(nb) == fj.get("notebook_sha256"),
            f"{sha(nb)[:12]} vs {str(fj.get('notebook_sha256'))[:12]}")
    if nb and PUSH_NB.is_file():
        chk("notebook == push-folder copy", sha(nb) == sha(PUSH_NB.read_bytes()), f"{sha(nb)[:12]} vs {sha(PUSH_NB.read_bytes())[:12]}")
    if nb and by.get("k100load_code_export.py"):
        cells = ["".join(c["source"]) for c in json.loads(nb)["cells"] if c["cell_type"] == "code"]
        chk("export == the notebook's code cells", "\n\n".join(cells).encode("utf-8") == by["k100load_code_export.py"])
    pin = fj.get("pin") or {}
    if by.get("k100load_lib.py") and pin.get("lib_sha256"):
        chk("lib (LF) sha == build_facts.pin.lib_sha256", sha(by["k100load_lib.py"].replace(b"\r\n", b"\n")) == pin["lib_sha256"])
    pay = fj.get("payload") or {}
    for name in ("k100load_lab.py", "qsa_kv_scales_262k.json"):
        want = (pay.get(name) or {}).get("sha256")
        if want and by.get(name) is not None:
            chk(f"dataset {name} sha == build_facts.payload", sha(by[name]) == want, f"{sha(by[name])[:12]} vs {want[:12]}")
    if by.get("SHA256SUMS.json"):
        sums = json.loads(by["SHA256SUMS.json"])
        ref_dir = Path("D:/ARC AGI 2/_scratch/push_e25/ds_e25_nb0_tf_ref")
        bad = [n for n, s in sums.items() if not (ref_dir / n).is_file() or sha((ref_dir / n).read_bytes()) != s]
        chk("reference SHA256SUMS.json matches every listed file", not bad, ", ".join(bad)[:300])
        for name in ("k100load_lab.py", "qsa_kv_scales_262k.json"):
            chk(f"reference SHA256SUMS.json lists {name}", name in sums)
    if nb:
        nb_m = max(f["mtime"] for f in files if f["flat"] == NB_FLAT)
        for kit in ("q27-e25-k100load-kit-real.ipynb", "q27-e25-k100load-kit-units.ipynb"):
            m = [f["mtime"] for f in files if f["flat"] == kit and f["data"] is not None]
            if m:
                chk(f"{kit} built after the notebook", m[0] >= nb_m, "kit notebook is older than the notebook" if m[0] < nb_m else "")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="list only; write nothing")
    ap.add_argument("--allow-inconsistent", action="store_true", help="write even when a consistency check fails")
    a = ap.parse_args()

    rows = parse_manifest()
    nb_row = next((r for r in rows if r["flat"] == NB_FLAT), None)
    nb_path = Path(nb_row["src"]) if nb_row else None
    nb_sha = sha(nb_path.read_bytes()) if nb_path and nb_path.is_file() else None
    files = resolve(rows, nb_sha)
    for f in files:
        p = Path(f["src"]) if not f["src"].startswith(("zip:", "kitout:", "glob:")) else None
        f["mtime"] = p.stat().st_mtime if p is not None and p.is_file() else 0.0

    names = [f["flat"] for f in files if f["data"] is not None]
    dups = sorted({n for n in names if names.count(n) > 1})
    bad_names = [n for n in names if "/" in n or "\\" in n]
    req_missing = [f for f in files if f["kind"] == "REQ" and f["data"] is None]
    opt_missing = [f for f in files if f["kind"] == "OPT" and f["data"] is None and f["status"] != "NONE"]
    checks = consistency(files)
    failed = [c for c in checks if not c[1]]

    print(f"manifest: {MANIFEST}")
    print(f"notebook sha256: {nb_sha}")
    for f in files:
        size = len(f["data"]) if f["data"] is not None else "-"
        print(f"  [{f['status']:7}] {f['kind']} {f['flat']:45} {size:>9}  {f['src']}" + (f"   <- {f['note']}" if f["note"] else ""))
    print("consistency checks:")
    for name, ok, detail in checks:
        print(f"  [{'OK' if ok else 'FAIL':4}] {name}" + (f"   ({detail})" if detail else ""))
    print(f"summary: {len(names)} present, {len(req_missing)} REQ missing, {len(opt_missing)} OPT missing / stale, "
          f"{len(dups)} duplicate names, {len(failed)} consistency checks failing")
    for f in req_missing:
        print(f"  MISSING REQ: {f['flat']}  <- {f['src']}  {f['note']}")
    for f in opt_missing:
        print(f"  {f['status']} OPT: {f['flat']}  <- {f['src']}  {f['note']}")
    if dups or bad_names:
        print("  DUPLICATE / NON-FLAT NAMES:", dups + bad_names)

    if a.dry_run:
        print("dry run: no zip written")
        return 0
    if req_missing or dups or bad_names:
        print("NOT WRITTEN: required files missing or names not unique / flat")
        return 2
    if failed and not a.allow_inconsistent:
        print("NOT WRITTEN: consistency checks failing (rerun with --allow-inconsistent to write anyway)")
        return 3

    date = datetime.date.today().isoformat()
    zp = HERE / f"E26_klee100_LOAD_audit_rev2_FLAT_{date}.zip"
    for suf in "bcdefghij":
        if not zp.exists():
            break
        zp = HERE / f"E26_klee100_LOAD_audit_rev2_FLAT_{date}_{suf}.zip"
    else:
        raise SystemExit("too many zips of this date")
    present = [f for f in files if f["data"] is not None]
    rec = {"zip": zp.name, "created": datetime.datetime.now().isoformat(timespec="seconds"), "notebook_sha256": nb_sha,
           "files": [{"flat": f["flat"], "bytes": len(f["data"]), "sha256": sha(f["data"]), "source": f["src"]} for f in present],
           "missing_optional": [{"flat": f["flat"], "status": f["status"], "source": f["src"], "note": f["note"]} for f in opt_missing],
           "consistency": [{"check": n, "ok": ok, "detail": d} for n, ok, d in checks]}
    lines = [f"# {zp.name}  created {rec['created']}  notebook sha256 {nb_sha}", "# flat_name\tbytes\tsha256\tsource"]
    lines += [f"{x['flat']}\t{x['bytes']}\t{x['sha256']}\t{x['source']}" for x in rec["files"]]
    lines += ["", "# optional files not included"] + [f"{x['flat']}\t{x['status']}\t{x['source']}\t{x['note']}" for x in rec["missing_optional"]]
    lines += ["", "# consistency checks"] + [f"{'OK' if x['ok'] else 'FAIL'}\t{x['check']}\t{x['detail']}" for x in rec["consistency"]]
    with zipfile.ZipFile(zp, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("00_ZIP_MANIFEST_SHA256.txt", "\n".join(lines) + "\n")
        for f in present:
            z.writestr(f["flat"], f["data"])
    with zipfile.ZipFile(zp) as z:                     # read back: flat, complete, byte-identical
        assert z.testzip() is None
        assert all("/" not in n for n in z.namelist())
        assert all(sha(z.read(x["flat"])) == x["sha256"] for x in rec["files"])
    rec["zip_sha256"] = sha(zp.read_bytes())
    Path(str(zp) + ".sha256.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"WROTE {zp} ({zp.stat().st_size} bytes, {len(present) + 1} entries) sha256 {rec['zip_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
