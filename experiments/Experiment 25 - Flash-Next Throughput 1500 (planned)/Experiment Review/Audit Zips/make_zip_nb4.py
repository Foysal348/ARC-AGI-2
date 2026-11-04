"""Build the FLAT GPT audit zip for the E25 NB4 TICK notebook (pattern: Experiment 26 make_zip_rev2.py; the manifest is
embedded below instead of a separate MANIFEST file).

  python make_zip_nb4.py --dry-run              list every entry (OK / MISSING / STALE / NONE) and the consistency checks; no zip
  python make_zip_nb4.py                        write E25_NB4_TICK_audit_FLAT_<date>.zip next to this script
  python make_zip_nb4.py --allow-inconsistent   write even when a consistency check fails (recorded in the zip)

Every REQ source must exist; OPT entries (the Kaggle CPU kit reports) are taken only when the report's notebook_sha256 equals
this notebook's sha (newest matching folder wins), else reported STALE / MISSING. The sha256 of every file goes into
00_ZIP_MANIFEST_SHA256.txt inside the zip. The zip has no folders. An existing zip of the same name is never overwritten
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
E25 = HERE.parents[1]
NBD = E25 / "Experiment Notebooks"
BLD = NBD / "Build"
REV = E25 / "Experiment Review"
RES = E25 / "Experiment Results"
PAY = Path("D:/ARC AGI 2/_scratch/push_e25/ds_e25_nb4_payload")
PUSH = Path("D:/ARC AGI 2/_scratch/push_e25")
NB_FLAT = "Q27-XHIGH-E25_NB4_TICK.ipynb"
KITOUT_GLOB = "D:/ARC AGI 2/_scratch/e25_kits_out_nb4*"
KITS = ("real", "units")

# KIND, flat name in the zip, source (a path, or kitout:<kit>/<file> = the newest matching Kaggle kit output)
MANIFEST = [
    ("REQ", "00_AUDIT_PROMPT_NB4.md", HERE / "00_AUDIT_PROMPT_NB4.md"),
    ("REQ", NB_FLAT, NBD / NB_FLAT),
    ("REQ", "build_e25_nb4.py", BLD / "build_e25_nb4.py"),
    ("REQ", "nb4_lib.py", BLD / "nb4_lib.py"),
    ("REQ", "nb4_code_export.py", BLD / "nb4_code_export.py"),
    ("REQ", "nb4_build_facts.json", BLD / "nb4_build_facts.json"),
    ("REQ", "nb4_unit_tests.py", BLD / "nb4_unit_tests.py"),
    ("REQ", "nb4_timeline.py", BLD / "nb4_timeline.py"),
    ("REQ", "nb4_timeline_tests.py", BLD / "nb4_timeline_tests.py"),
    ("REQ", "nb4_ppack.py", BLD / "nb4_ppack.py"),
    ("REQ", "nb4_ppack_tests.py", BLD / "nb4_ppack_tests.py"),
    ("REQ", "nb4_p5_tests.py", BLD / "nb4_p5_tests.py"),
    # the payload dataset (foysalemonshanto/e25-nb4-payload) as uploaded; p5_src/<name> -> <name> (flat)
    ("REQ", "nb4_p5_engine.py", PAY / "nb4_p5_engine.py"),
    ("REQ", "build_host_ar.py", PAY / "p5_src" / "build_host_ar.py"),
    ("REQ", "host_ar.cu", PAY / "p5_src" / "host_ar.cu"),
    ("REQ", "host_ar_binding.cpp", PAY / "p5_src" / "host_ar_binding.cpp"),
    ("REQ", "host_ar_check.h", PAY / "p5_src" / "host_ar_check.h"),
    ("REQ", "host_ar_layout.h", PAY / "p5_src" / "host_ar_layout.h"),
    ("REQ", "payload_SHA256SUMS.txt", PAY / "SHA256SUMS"),
    ("REQ", "payload_dataset-metadata.json", PAY / "dataset-metadata.json"),
    ("REQ", "NB4_TICK_SPEC_2026-11-01.md", REV / "NB4_TICK_SPEC_2026-11-01.md"),
    ("REQ", "NB4_RESEARCH_graphs_2026-11-01.md", REV / "NB4_RESEARCH_graphs_2026-11-01.md"),
    ("REQ", "NB4_RESEARCH_folders_2026-11-01.md", REV / "NB4_RESEARCH_folders_2026-11-01.md"),
    ("REQ", "NB4_RESEARCH_conversations_2026-11-01.md", REV / "NB4_RESEARCH_conversations_2026-11-01.md"),
    ("REQ", "E25_GPU_NB1_R2_2026-10-31.md", RES / "E25_GPU_NB1_R2_2026-10-31.md"),
    ("REQ", "E25_GPU_NB5_QUALITY_2026-11-01.md", RES / "E25_GPU_NB5_QUALITY_2026-11-01.md"),
    ("REQ", "PLAN_1500.md", E25 / "PLAN_1500.md"),
    ("REQ", "q27-xhigh-e25-nb4-tick_kernel-metadata.json", PUSH / "q27-xhigh-e25-nb4-tick__foysalemonshanto" / "kernel-metadata.json"),
    ("REQ", "build_e25_nb4_cpu_kit.py", BLD / "build_e25_nb4_cpu_kit.py"),
    ("REQ", "q27-e25-nb4-kit-real.ipynb", PUSH / "q27-e25-nb4-kit-real__foysalemonshanto" / "q27-e25-nb4-kit-real.ipynb"),
    ("REQ", "q27-e25-nb4-kit-real_kernel-metadata.json", PUSH / "q27-e25-nb4-kit-real__foysalemonshanto" / "kernel-metadata.json"),
    ("REQ", "q27-e25-nb4-kit-units.ipynb", PUSH / "q27-e25-nb4-kit-units__foysalemonshanto" / "q27-e25-nb4-kit-units.ipynb"),
    ("REQ", "q27-e25-nb4-kit-units_kernel-metadata.json", PUSH / "q27-e25-nb4-kit-units__foysalemonshanto" / "kernel-metadata.json"),
    ("OPT", "nb4_kit_real_report.json", "kitout:real/nb4_kit_real_report.json"),
    ("OPT", "nb4_kit_units_report.json", "kitout:units/nb4_kit_units_report.json"),
    ("OPT", "q27-e25-nb4-kit-real.log", "kitout:real/q27-e25-nb4-kit-real.log"),
    ("OPT", "q27-e25-nb4-kit-units.log", "kitout:units/q27-e25-nb4-kit-units.log"),
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def kit_folder(kit, nb_sha):
    """The newest e25_kits_out_nb4*/q27-e25-nb4-kit-<kit>/ whose report tested nb_sha; also the stale ones seen."""
    found, stale = [], []
    for d in glob.glob(f"{KITOUT_GLOB}/q27-e25-nb4-kit-{kit}"):
        rp = Path(d) / f"nb4_kit_{kit}_report.json"
        if not rp.is_file():
            continue
        try:
            got = json.loads(rp.read_text(encoding="utf-8")).get("notebook_sha256")
        except Exception as e:  # noqa: BLE001
            stale.append(f"{rp} (unreadable: {e})")
            continue
        if got == nb_sha:
            found.append((rp.stat().st_mtime, Path(d)))
        else:
            stale.append(f"{rp} tested {str(got)[:8]}")
    found.sort(key=lambda t: t[0], reverse=True)
    return (found[0][1] if found else None), stale


def resolve(nb_sha):
    kitdirs, out = {}, []
    for kind, flat, src in MANIFEST:
        r = {"kind": kind, "flat": flat, "src": str(src)}
        if isinstance(src, str) and src.startswith("kitout:"):
            kit, fname = src[len("kitout:"):].split("/", 1)
            if kit not in kitdirs:
                kitdirs[kit] = kit_folder(kit, nb_sha) if nb_sha else (None, ["notebook missing: kit match impossible"])
            d, stale = kitdirs[kit]
            if d is None:
                out.append(dict(r, data=None, status="STALE" if stale else "MISSING", mtime=0.0,
                                note=("only results for other notebooks: " + "; ".join(stale)) if stale
                                else f"no {KITOUT_GLOB}/q27-e25-nb4-kit-{kit} yet (the kit has not run on Kaggle)"))
                continue
            p = d / fname
            out.append(dict(r, src=str(p), data=p.read_bytes() if p.is_file() else None, status="OK" if p.is_file() else "MISSING",
                            mtime=p.stat().st_mtime if p.is_file() else 0.0, note="" if p.is_file() else f"not in {d}"))
            continue
        p = Path(src)
        out.append(dict(r, data=p.read_bytes() if p.is_file() else None, status="OK" if p.is_file() else "MISSING",
                        mtime=p.stat().st_mtime if p.is_file() else 0.0, note="" if p.is_file() else "file not found"))
    return out


def consistency(files):
    """Freshness / identity checks so the zip is not cut from a half-rebuilt state. -> list of (name, ok, detail)."""
    by = {f["flat"]: f["data"] for f in files if f["data"] is not None}
    res = []

    def chk(name, ok, detail=""):
        res.append((name, bool(ok), detail))

    nb, facts = by.get(NB_FLAT), by.get("nb4_build_facts.json")
    fj = json.loads(facts) if facts else {}
    if nb and facts:
        chk("notebook sha == build_facts.notebook_sha256", sha(nb) == fj.get("notebook_sha256"),
            f"{sha(nb)[:12]} vs {str(fj.get('notebook_sha256'))[:12]}")
        chk("notebook size == build_facts.bytes", len(nb) == fj.get("bytes"), f"{len(nb)} vs {fj.get('bytes')}")
        chk("notebook < 400 kB and LF only", len(nb) < 400_000 and b"\r" not in nb.replace(b"\\r", b""))
    push_nb = PUSH / "q27-xhigh-e25-nb4-tick__foysalemonshanto" / "q27-xhigh-e25-nb4-tick.ipynb"
    if nb and push_nb.is_file():
        chk("notebook == push-folder copy", sha(nb) == sha(push_nb.read_bytes()), f"{sha(nb)[:12]} vs {sha(push_nb.read_bytes())[:12]}")
    if nb and by.get("nb4_code_export.py"):
        cells = ["".join(c["source"]) for c in json.loads(nb)["cells"] if c["cell_type"] == "code"]
        chk("export == the notebook's code cells", "\n\n".join(cells).encode("utf-8") == by["nb4_code_export.py"])
    pin = fj.get("pin") or {}
    lf = lambda b: b.replace(b"\r\n", b"\n")
    if by.get("nb4_lib.py") and pin.get("lib_sha256"):
        chk("lib (LF) sha == build_facts.pin.lib_sha256", sha(lf(by["nb4_lib.py"])) == pin["lib_sha256"])
    for name in ("nb4_timeline.py", "nb4_ppack.py", "nb4_p5_engine.py"):
        want = (pin.get("hooks") or {}).get(name)
        if want and by.get(name) is not None:
            chk(f"{name} (LF) sha == build_facts.pin.hooks", sha(lf(by[name])) == want, f"{sha(lf(by[name]))[:12]} vs {want[:12]}")
    if (BLD / "nb4_p5_engine.py").is_file() and by.get("nb4_p5_engine.py") is not None:
        chk("Build/nb4_p5_engine.py == the payload copy", sha((BLD / "nb4_p5_engine.py").read_bytes()) == sha(by["nb4_p5_engine.py"]))
    pf = fj.get("payload_files") or {}
    flat_of = {"nb4_p5_engine.py": "nb4_p5_engine.py", **{f"p5_src/{n}": n for n in
               ("build_host_ar.py", "host_ar.cu", "host_ar_binding.cpp", "host_ar_check.h", "host_ar_layout.h")}}
    for path, want in sorted(pf.items()):
        d = by.get(flat_of.get(path, "?"))
        chk(f"payload {path} sha == build_facts.payload_files", d is not None and sha(d) == want,
            "" if d is not None and sha(d) == want else f"{sha(d)[:12] if d else 'absent'} vs {want[:12]}")
    if by.get("payload_SHA256SUMS.txt"):
        sums = dict(reversed(l.split("  ", 1)) for l in by["payload_SHA256SUMS.txt"].decode("utf-8").splitlines() if l.strip())
        chk("payload SHA256SUMS == build_facts.payload_files", sums == pf, f"{sorted(sums)} vs {sorted(pf)}")
    for name, want in ((fj.get("pin") or {}).get("research") or {}).items():
        if by.get(name) is not None:
            chk(f"{name} sha == build_facts.pin.research", sha(by[name]) == want, f"{sha(by[name])[:12]} vs {str(want)[:12]}")
    if nb:
        nbsha = sha(nb).encode()
        nb_m = max(f["mtime"] for f in files if f["flat"] == NB_FLAT)
        for kit in KITS:
            kn = by.get(f"q27-e25-nb4-kit-{kit}.ipynb")
            if kn is None:
                continue
            chk(f"q27-e25-nb4-kit-{kit}.ipynb names this notebook's sha", nbsha in kn)
            chk(f"q27-e25-nb4-kit-{kit}.ipynb < 400 kB", len(kn) < 400_000, str(len(kn)))
            m = [f["mtime"] for f in files if f["flat"] == f"q27-e25-nb4-kit-{kit}.ipynb"]
            chk(f"q27-e25-nb4-kit-{kit}.ipynb built after the notebook", m and m[0] >= nb_m)
            meta = by.get(f"q27-e25-nb4-kit-{kit}_kernel-metadata.json")
            if meta:
                mj = json.loads(meta)
                chk(f"kit {kit} metadata: foysalemonshanto, private, CPU, no internet, pinned image",
                    mj.get("id") == f"foysalemonshanto/q27-e25-nb4-kit-{kit}" and mj.get("is_private") is True
                    and mj.get("enable_gpu") is False and mj.get("enable_internet") is False
                    and mj.get("docker_image") == json.loads(by.get("q27-xhigh-e25-nb4-tick_kernel-metadata.json") or b"{}").get("docker_image"))
    for kit in KITS:
        rep = by.get(f"nb4_kit_{kit}_report.json")
        if rep is not None:
            rj = json.loads(rep)
            chk(f"nb4_kit_{kit}_report.json tested this notebook", nb is not None and rj.get("notebook_sha256") == sha(nb))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="list only; write nothing")
    ap.add_argument("--allow-inconsistent", action="store_true", help="write even when a consistency check fails")
    a = ap.parse_args()

    nb_path = NBD / NB_FLAT
    nb_sha = sha(nb_path.read_bytes()) if nb_path.is_file() else None
    files = resolve(nb_sha)
    names = [f["flat"] for f in files if f["data"] is not None]
    dups = sorted({n for n in names if names.count(n) > 1})
    bad_names = [n for n in names if "/" in n or "\\" in n]
    req_missing = [f for f in files if f["kind"] == "REQ" and f["data"] is None]
    opt_missing = [f for f in files if f["kind"] == "OPT" and f["data"] is None]
    checks = consistency(files)
    failed = [c for c in checks if not c[1]]

    print(f"notebook sha256: {nb_sha}")
    for f in files:
        size = len(f["data"]) if f["data"] is not None else "-"
        print(f"  [{f['status']:7}] {f['kind']} {f['flat']:48} {size:>9}  {f['src']}" + (f"   <- {f['note']}" if f["note"] else ""))
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
    zp = HERE / f"E25_NB4_TICK_audit_FLAT_{date}.zip"
    for suf in "bcdefghij":
        if not zp.exists():
            break
        zp = HERE / f"E25_NB4_TICK_audit_FLAT_{date}_{suf}.zip"
    else:
        raise SystemExit("too many zips of this date")
    present = [f for f in files if f["data"] is not None]
    created = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [f"# {zp.name}  created {created}  notebook sha256 {nb_sha}", "# flat_name\tbytes\tsha256\tsource"]
    lines += [f"{f['flat']}\t{len(f['data'])}\t{sha(f['data'])}\t{f['src']}" for f in present]
    lines += ["", "# optional files not included"] + [f"{f['flat']}\t{f['status']}\t{f['src']}\t{f['note']}" for f in opt_missing]
    lines += ["", "# consistency checks"] + [f"{'OK' if ok else 'FAIL'}\t{n}\t{d}" for n, ok, d in checks]
    with zipfile.ZipFile(zp, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("00_ZIP_MANIFEST_SHA256.txt", "\n".join(lines) + "\n")
        for f in present:
            z.writestr(f["flat"], f["data"])
    with zipfile.ZipFile(zp) as z:                     # read back: flat, complete, byte-identical
        assert z.testzip() is None
        assert all("/" not in n for n in z.namelist())
        assert all(sha(z.read(f["flat"])) == sha(f["data"]) for f in present)
        n_entries = len(z.namelist())
    print(f"WROTE {zp} ({zp.stat().st_size} bytes, {n_entries} entries) sha256 {sha(zp.read_bytes())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
