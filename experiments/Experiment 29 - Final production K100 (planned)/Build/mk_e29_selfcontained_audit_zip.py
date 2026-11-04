"""Builds the SELF-CONTAINED E29_DFLASH_D3S rev 2d audit zip (fix F8: the rev 2c zip lacked e28_sched_patch.py, the Duck payload
and the kit reports) and verifies it: extracted into an EMPTY temp folder, the four test files run there (python -I -X utf8,
every input path pointed into that folder, the adversarial tests with --no-fallback).

usage:
  python mk_e29_selfcontained_audit_zip.py [--date 2026-11-02] [--kit LABEL=DIR ...] [--no-rev2c-kits] [--out ZIP]
                                           [--no-verify] [--verify-dir DIR] [--rev2c-zip ZIP] [--keep-verify-dir]
  --kit LABEL=DIR   a Kaggle CPU kit output folder (every file in it, flattened as KIT_<LABEL>__<path>); repeat per kit.
                    The rev 2c kits (rev2c=D:/ARC AGI 2/_scratch/kits_e29v2c) are always added unless --no-rev2c-kits;
                    add the rev 2d kits after they ran, e.g. --kit rev2d=D:/ARC AGI 2/_scratch/kits_e29v2d
  --out             default: Experiment Review/Audit Zips/E29_DFLASH_D3S_REV2D_audit_FLAT_<date>.zip
  --rev2c-zip       the rev 2c audit zip used for DELTA_rev2c_to_rev2d.diff (default: Audit Zips/E29_DFLASH_D3S_REV2C_audit_FLAT_2026-11-02.zip)
Exit 0 = zip written and (unless --no-verify) all four test files passed in the empty folder; 1 otherwise. Nothing is uploaded.

Zip layout (flat): the notebook + push metadata (PUSH__*), the Build modules + tests, the dependencies (e28_sched_patch.py,
e28k100_lib.py, nb5_lib.py, dflash_seam.py, the Duck payload bin), the payload-dataset files by facts sha, the kit_lib facts
(e29_kit_lib_facts_dflash-d3s.json), records_k100q_w4.zip (branch_result.json only) + the ARC eval challenges (for the real-
record selection tests), the 4 audits (AUDIT_A1__ / AUDIT_A2A3__ / AUDIT_A4__), the fix lists, the agent reports (REPORT__),
the kit reports (KIT_<label>__), the kit notebooks (KITNB__), DELTA_rev2c_to_rev2d.diff, 00_RUN_TESTS.py, 00_GPT_PROMPT.md,
00_MANIFEST_SHA256.json + SHA256SUMS.
"""
import argparse
import difflib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REVIEW = EXP / "Experiment Review"
ZIPS = REVIEW / "Audit Zips"
ROOT = Path(r"D:/ARC AGI 2")
PUSH = ROOT / "_scratch/push_e25"
E28B = ROOT / "experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Build"
E25B = ROOT / "experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build"
E27B = ROOT / "experiments/Experiment 27 - Flash-Next DFlash2 (planned)/Experiment Notebooks/Build"
PAYLOAD_NAME = "q27x_e15b_duck_9e4b89285136.bin"
PAYLOAD_SHA = "9e4b89285136616d523028ac042dcca6afe82175081a1481d584c998ab3ee566"
PAYLOAD_CANDS = [ROOT / "experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks/Build/payload",
                 ROOT / "_scratch/best_engine_zip/payload", ROOT / "_scratch/e15b/hamid_payload", ROOT / "_scratch/e15b/sami_payload"]
NB_NAME = "Q27-XHIGH-E29_DFLASH_D3S.ipynb"
NB = EXP / "Experiment Notebooks" / NB_NAME
PUSH_NB_DIR = PUSH / "q27-xhigh-e29-dflash-d3s__foysalemonshanto"
PUSH_DS_DIR = PUSH / "ds_q27-e29-final-payload"
KIT_NB_DIRS = ["q27-e29-kit-units-dflash-d3s__foysalemonshanto", "q27-e29-kit-real-dflash-d3s__foysalemonshanto",
               "q27-e29-kit-direct__foysalemonshanto"]
K100_PHASE = ROOT / "_scratch/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x/phase_Q_K100_FP8_K3"
W4_PHASE = ROOT / "_scratch/e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x/phase_Q_FP8_K3"
EVAL = ROOT / ("experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review/"
               "E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data/arc-agi_evaluation_challenges.json")
REV2C_KITS = ROOT / "_scratch/kits_e29v2c"
BUILD_FILES = ["build_e29_final.py", "build_e29_cpu_kit.py", "e29_direct_build.py", "e29_lib.py", "e29_lib_v1_0.py", "e29_live.py",
               "e29_sched_patch.py", "e29_dflash_d3s_code_export.py", "e29_dflash_d3s_build_facts.json", "e29_build_facts.json",
               "e29_kit_facts.json", "e29_sched_tests.py", "e29d_direct_tests.py", "e29_kit_lib_tests.py",
               "e29_rev2d_adversarial_tests.py", "mk_e29_selfcontained_audit_zip.py"]
DELTA_FILES = ["e29_lib.py", "e29d_sched_patch.py", "e29_nbx.py", "e29_direct.py", "e29_direct_build.py", "build_e29_final.py",
               "build_e29_cpu_kit.py", "e29_live.py", "e29_kit_lib_tests.py", "e29d_direct_tests.py", "e29_dflash_d3s_code_export.py",
               "e29_dflash_d3s_build_facts.json"]
TESTS = [("e29_sched_tests.py", []), ("e29d_direct_tests.py", []), ("e29_kit_lib_tests.py", []),
         ("e29_rev2d_adversarial_tests.py", ["--no-fallback", "--strict"])]


def sha(b):
    return hashlib.sha256(b).hexdigest()


class Pack:
    def __init__(self):
        self.items, self.warn = {}, []

    def add(self, name, data, source):
        assert "/" not in name and "\\" not in name, name
        if isinstance(data, Path):
            data = data.read_bytes()
        if name in self.items and self.items[name][0] != data:
            raise SystemExit(f"duplicate name with different bytes: {name}")
        self.items[name] = (data, str(source))

    def need(self, name, path, source=None):
        p = Path(path)
        if not p.is_file():
            self.warn.append(f"MISSING {name}: {p}")
            return False
        self.add(name, p, source or p)
        return True


def by_sha(name, want, dirs):
    """The first file `name` with sha256 `want` in dirs (the payload dataset: name + sha, like the notebook's q27x_payload)."""
    for d in dirs:
        p = Path(d) / name
        if p.is_file() and sha(p.read_bytes()) == want:
            return p
    return None


def find_payload():
    for d in PAYLOAD_CANDS:
        p = d / PAYLOAD_NAME
        if p.is_file() and sha(p.read_bytes()) == PAYLOAD_SHA:
            return p
    for base in (ROOT / "experiments", ROOT / "_scratch"):          # a full search only when the known places miss
        for p in base.rglob(PAYLOAD_NAME):
            if sha(p.read_bytes()) == PAYLOAD_SHA:
                return p
    return None


def records_zip():
    buf, n = io.BytesIO(), {"k100q": 0, "w4": 0}
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for tag, ph in (("k100q", K100_PHASE), ("w4", W4_PHASE)):
            for f in sorted(ph.glob("duck/*/branch_result.json")):
                z.writestr(f"{tag}/{ph.name}/duck/{f.parent.name}/branch_result.json", f.read_bytes())
                n[tag] += 1
    return buf.getvalue(), n


def delta(rev2c_zip):
    out = []
    with zipfile.ZipFile(rev2c_zip) as z:
        names = set(z.namelist())
        for f in DELTA_FILES:
            old = z.read(f).decode("utf-8", "replace").replace("\r\n", "\n").splitlines(True) if f in names else []
            p = HERE / f
            new = p.read_bytes().decode("utf-8", "replace").replace("\r\n", "\n").splitlines(True) if p.is_file() else []
            d = list(difflib.unified_diff(old, new, f"rev2c/{f}", f"rev2d/{f}", n=3))
            out.append(f"# ==== {f}: {len([l for l in d if l[:1] in '+-' and l[:3] not in ('+++', '---')])} changed lines\n")
            out.extend(d)
    return "".join(out).encode("utf-8")


RUNNER = r'''"""Runs the four E29 test files of THIS folder only (self-contained check):  python -I -X utf8 00_RUN_TESTS.py
Every input path is pointed into this folder (no D:/ARC AGI 2 path, no Kaggle mount): the Duck payload, e28_sched_patch.py,
nb5_lib.py, the K100Q / W4 branch records (records_k100q_w4.zip -> _records/), the ARC eval challenges, the payload-dataset
files (by name + sha), the kit_lib facts. Logs: _logs/<test>.log; summary: 00_RUN_TESTS_RESULTS.json. Exit 1 on any failure."""
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

H = Path(__file__).resolve().parent
REC, TMP, LOGS = H / "_records", H / "_tmp", H / "_logs"
if not REC.is_dir():
    with zipfile.ZipFile(H / "records_k100q_w4.zip") as z:
        z.extractall(REC)
TMP.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)
ENV = dict(os.environ, E28_TEST_PAYLOAD=str(H / "__PAYLOAD__"), E29_TEST_E28_DIR=str(H), E28_TEST_K100=str(REC / "k100q"),
           E28_TEST_W4=str(REC / "w4"), E28_TEST_EVAL=str(H / "arc-agi_evaluation_challenges.json"), E28_TEST_EXPORTS=str(H),
           E28_TEST_TMP=str(TMP), E29LT_NB=str(H / "__NB__"), E29LT_FACTS=str(H / "e29_dflash_d3s_build_facts.json"),
           E29LT_KF=str(H / "e29_kit_lib_facts_dflash-d3s.json"), E29LT_NB5LIB=str(H / "nb5_lib.py"), E29LT_PAYLOAD_DIR=str(H),
           PYTHONIOENCODING="utf-8")
TESTS = __TESTS__
# python -I (isolated) where possible: the Duck harness needs `requests`; when it lives only in the user site (-I hides it),
# fall back to -E -P (PYTHON* env ignored, the script dir / cwd not prepended; the user site kept), recorded.
FLAGS = ["-I"]
if subprocess.run([sys.executable, "-I", "-c", "import requests"], capture_output=True).returncode != 0:
    FLAGS = ["-E", "-P"] if sys.version_info >= (3, 11) else ["-E"]
print("interpreter flags:", FLAGS, sys.version.split()[0], flush=True)
out, bad = {"_flags": FLAGS}, 0
for name, extra in TESTS:
    t0 = time.time()
    cp = subprocess.run([sys.executable] + FLAGS + ["-X", "utf8", str(H / name)] + list(extra) + (["--dir", str(H)] if "adversarial" in name else []),
                        cwd=str(H), env=ENV, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    (LOGS / (name + ".log")).write_text(cp.stdout + "\n--- STDERR ---\n" + cp.stderr, encoding="utf-8")
    tail = [l for l in (cp.stdout + "\n" + cp.stderr).splitlines() if l.strip()][-6:]
    out[name] = {"rc": cp.returncode, "sec": round(time.time() - t0, 1), "tail": tail,
                 "outside_path_mentions": sum(1 for l in (cp.stdout + cp.stderr).splitlines() if "ARC AGI 2" in l)}
    bad += int(cp.returncode != 0)
    print(f"{'OK  ' if cp.returncode == 0 else 'FAIL'} rc={cp.returncode} {name} ({out[name]['sec']} s)", *("    " + l for l in tail), sep="\n", flush=True)
(H / "00_RUN_TESTS_RESULTS.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("ALL OK" if not bad else f"{bad} test file(s) failed")
sys.exit(1 if bad else 0)
'''


def prompt(date, nb_sha, kits, notes):
    return f"""# GPT audit request: E29_DFLASH_D3S rev 2d (Experiment 29, {date}) -- GO / NO-GO on the rev 2d DELTA only

You audited rev 2c (notebook ed2e23a7) four times (A1 = RISK_AUDIT, A2 = A3 = GPT_AUDIT, A4 = INDEPENDENT_AUDIT; all in this zip
as AUDIT_*). Every finding was verified and turned into the fix list FIX_LIST_E29_REV2C_AUDITS_2026-11-02.md (F1..F14). Rev 2d
(notebook sha256 {nb_sha}) applies them. Please review ONLY the rev 2d delta (DELTA_rev2c_to_rev2d.diff + the files it touches)
and answer:

1. First line: **GO** or **NO-GO** for (a) the GPU smoke and (b) the 6 h / 12 h production run.
2. Then ONLY blockers and severe issues of the delta (a crash, a skipped arm, a wasted GPU run, wrong data / a wrong
   submission, a fix that does not do what the fix list says, a regression of rev 2c behaviour). For each: file:line, the
   trigger, the impact, the minimal fix, and a test that would catch it. No style, naming, documentation, medium or low items.

## The fixes to check (FIX_LIST F1..F14)
- F1 / F1+ late phase C: `arc2_duck.E29D_CAN_START(deadline)` (now + 60 s + drain_min x 60 < deadline, set in e29_direct_setup,
  cleared on OFF / failure). The patched scheduler takes the DIRECT branch only when the hook is callable and True; otherwise
  that same item runs the ORIGINAL Duck C session (one C start / completion / reservation release).
- F2 forced max-size first prefill probe on every DFlash try + worker-side free memory (low free -> one D ladder step).
- F3 / F14 QSA 128/64 read-back required on all 4 workers before the first prefill (/proc, else worker RPC); a mismatch or a
  missing rank -> the SAME rung relaunched without the QSA env (Merlin's 256/256 + EMBED_HOST), DFlash kept; UNVERIFIED_QSA label.
- F4 the in-run DFlash -> MTP relaunch runs the E28 boot gates (C-11, method map, I8 read-back / E28's I8 rule).
- F5 effective_algorithm record + experiment_validity DIRECT_OFF (never stops the run). F6 / F13 host cgroup + KV ledger (record only).
- F9 size (< 400 kB, new code in the payload module). F10 in-run XID -> MTP (device fault first), GPU_DEVICE_FAULT recorded.
- **F11 is decided differently from A4's fail-closed proposal** (on purpose: a fail-closed raise kills the whole production run
  and leaves only the CELL 0 identity submission): the pristine sha of every seam-touched file is snapshotted before the apply;
  after SSEQ.finish() (also when it raised) the files are verified, re-restored from the snapshot and verified again; if still
  not pristine, the next MTP boot needs the q_seam_inert read-back (seam env unset): inert -> continue, recorded
  SEAM_REVERT_UNVERIFIED_INERT; not inert -> that try fails (the E28 ladder continues). Judge this decision only for a real
  hole (a way to produce wrong answers or a crash), not for the policy itself.
- F12 submission_check (record only) after the writes, in the report and the LIVE header. F7 / S1 = smoke measurements.

## Executable evidence (run it yourself)
`python -I -X utf8 00_RUN_TESTS.py` in the extracted folder runs, with every path inside the folder:
e29_sched_tests.py, e29d_direct_tests.py, e29_kit_lib_tests.py and e29_rev2d_adversarial_tests.py (--no-fallback --strict).
The adversarial file turns every audit regression (F1, F2, F3, F4, F5, F10, F11, F12 and A4's 7 audit_repro scenarios) into a
plain-assert case on the REAL modules / the notebook's own functions (AST-extracted from e29_dflash_d3s_code_export.py); its
header records the rev 2c results (25 FAIL = the bugs reproduced, 20 SKIP_NOT_IMPLEMENTED) and the expected rev 2d result
(58/58 PASS). Kit reports included: {', '.join(kits) or 'none'}.

## Contents
Notebook {NB_NAME} + PUSH__ (kernel-metadata, payload dataset metadata / SHA256SUMS), every Build module and test, the
dependencies (e28_sched_patch.py, e28k100_lib.py, nb5_lib.py, dflash_seam.py, the Duck payload {PAYLOAD_NAME}), the payload-
dataset files by facts sha, records_k100q_w4.zip (branch_result.json of the K100Q / W4 GPU runs) + the public ARC eval
challenges, AUDIT_*, FIX_LIST_*, REPORT__ (agent reports), KIT_<label>__ (Kaggle CPU kit outputs), KITNB__ (kit notebooks),
DELTA_rev2c_to_rev2d.diff, 00_MANIFEST_SHA256.json / SHA256SUMS.
{chr(10).join('- NOTE: ' + n for n in notes)}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=time.strftime("%Y-%m-%d"))
    ap.add_argument("--kit", action="append", default=[])
    ap.add_argument("--no-rev2c-kits", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--verify-dir", default=None)
    ap.add_argument("--keep-verify-dir", action="store_true")
    ap.add_argument("--rev2c-zip", default=str(ZIPS / "E29_DFLASH_D3S_REV2C_audit_FLAT_2026-11-02.zip"))
    a = ap.parse_args()
    out = Path(a.out) if a.out else ZIPS / f"E29_DFLASH_D3S_REV2D_audit_FLAT_{a.date}.zip"
    P = Pack()
    facts = json.loads((HERE / "e29_dflash_d3s_build_facts.json").read_text(encoding="utf-8"))
    # ---- the notebook + push metadata
    P.need(NB_NAME, NB)
    nb_sha = sha(NB.read_bytes())
    if facts.get("notebook_sha256") != nb_sha:
        P.warn.append(f"the notebook sha {nb_sha[:12]} != facts notebook_sha256 {str(facts.get('notebook_sha256'))[:12]}")
    pnb = PUSH_NB_DIR / "q27-xhigh-e29-dflash-d3s.ipynb"
    if pnb.is_file() and sha(pnb.read_bytes()) != nb_sha:
        P.warn.append(f"the push copy {pnb.name} differs from the notebook (included as PUSH__ copy)")
        P.add("PUSH__q27-xhigh-e29-dflash-d3s__" + pnb.name, pnb, pnb)
    P.need("PUSH__q27-xhigh-e29-dflash-d3s__kernel-metadata.json", PUSH_NB_DIR / "kernel-metadata.json")
    for f in ("dataset-metadata.json", "SHA256SUMS"):
        P.need("PUSH__q27-e29-final-payload__" + f, PUSH_DS_DIR / f)
    for d in KIT_NB_DIRS:
        for f in sorted((PUSH / d).glob("*")) if (PUSH / d).is_dir() else []:
            if f.is_file():
                P.add(f"KITNB__{d}__{f.name}", f, f)
    # ---- Build modules, tests, dependencies
    for f in BUILD_FILES:
        P.need(f, HERE / f)
    files = facts["e29"]["direct"]["files"]
    pay = [("e29_direct.py", files["direct"][1]), ("e29d_sched_patch.py", files["sched"][1]), ("e29_nbx.py", files["nbx"][1]),
           (facts["e29"]["seam"][0], facts["e29"]["seam"][1]), ("e29_nbs1_log_fixture.json", facts["e29"]["nbs1_fixture_sha256"])]
    for name, want in pay:
        p = by_sha(name, want, [HERE, PUSH_DS_DIR, E27B] + sorted(HERE.glob("_kit_local/*")))
        if p is None:
            P.warn.append(f"no {name} with the facts sha {want[:12]} (Build has {sha((HERE / name).read_bytes())[:12] if (HERE / name).is_file() else 'none'}): "
                          "the Build copy is packed; e29_kit_lib_tests R-checks will not find it by sha")
            p = HERE / name if (HERE / name).is_file() else (E27B / name)
        P.need(name, p)
    P.need("e28_sched_patch.py", E28B / "e28_sched_patch.py")
    P.need("e28k100_lib.py", E28B / "e28k100_lib.py")
    P.need("nb5_lib.py", E25B / "nb5_lib.py")
    kf = HERE / "_kit_local/dflash-d3s/e29_kit_facts.json"
    if kf.is_file():
        miss = {"variants_cfg", "nbdf_q_pin", "nbdf_d_retry", "nbdf_q_captures"} - set(json.loads(kf.read_text(encoding="utf-8")))
        if miss:
            P.warn.append(f"kit_lib facts {kf} lack {sorted(miss)}")
    P.need("e29_kit_lib_facts_dflash-d3s.json", kf)
    pp = find_payload()
    if pp is None:
        P.warn.append(f"MISSING the Duck payload {PAYLOAD_NAME} (sha {PAYLOAD_SHA[:12]}) under {ROOT}")
    else:
        P.add(PAYLOAD_NAME, pp, pp)
    rz, rn = records_zip()
    P.add("records_k100q_w4.zip", rz, f"{K100_PHASE} ({rn['k100q']}) + {W4_PHASE} ({rn['w4']}), branch_result.json only")
    if not rn["k100q"] or not rn["w4"]:
        P.warn.append(f"branch records missing: {rn}")
    P.need("arc-agi_evaluation_challenges.json", EVAL)
    # ---- the audits, fix lists, agent reports
    audits = {"AUDIT_A4__": ("E29_REV2C_INDEPENDENT_AUDIT_2026-11-02.zip",
                             lambda n: n.split("/")[-1][:1].isdigit() or n.split("/")[-1].startswith(("README", "audit_"))),
              "AUDIT_A1__": ("E29_REV2C_INDEPENDENT_RISK_AUDIT_2026-11-02.zip", lambda n: True),
              "AUDIT_A2A3__": ("E29_DFLASH_D3S_REV2C_GPT_AUDIT_2026-11-02.zip", lambda n: True)}
    for pre, (zn, keep) in audits.items():
        zp = REVIEW / zn
        if not zp.is_file():
            P.warn.append(f"MISSING audit {zp}")
            continue
        with zipfile.ZipFile(zp) as z:
            for n in z.namelist():
                if not n.endswith("/") and keep(n):
                    P.add(pre + n.split("/")[-1], z.read(n), f"{zn}:{n}")
    for f in sorted(REVIEW.glob("FIX_LIST_*.md")):
        P.add(f.name, f, f)
    for f in sorted((REVIEW / "Agent Reports").glob("*.md")):
        P.add("REPORT__" + f.name, f, f)
    # ---- kit reports
    kits = ([("rev2c", REV2C_KITS)] if not a.no_rev2c_kits else []) + [tuple(k.split("=", 1)) for k in a.kit]
    for label, d in kits:
        d = Path(d)
        if not d.is_dir():
            P.warn.append(f"MISSING kit dir {label}={d}")
            continue
        for f in sorted(x for x in d.rglob("*") if x.is_file()):
            P.add(f"KIT_{label}__" + "__".join(f.relative_to(d).parts), f, f)
    # ---- delta, runner, prompt, manifest
    if Path(a.rev2c_zip).is_file():
        P.add("DELTA_rev2c_to_rev2d.diff", delta(Path(a.rev2c_zip)), f"difflib: {a.rev2c_zip} -> {HERE}")
    else:
        P.warn.append(f"MISSING the rev 2c zip for the delta: {a.rev2c_zip}")
    P.add("00_RUN_TESTS.py", RUNNER.replace("__PAYLOAD__", PAYLOAD_NAME).replace("__NB__", NB_NAME)
          .replace("__TESTS__", json.dumps(TESTS)).encode("utf-8"), "generated by mk_e29_selfcontained_audit_zip.py")
    P.add("00_GPT_PROMPT.md", prompt(a.date, nb_sha, [k for k, _ in kits], P.warn).encode("utf-8"), "generated")
    man = {n: {"sha256": sha(b), "bytes": len(b), "source": s} for n, (b, s) in sorted(P.items.items())}
    P.add("00_MANIFEST_SHA256.json", json.dumps({"date": a.date, "notebook_sha256": nb_sha, "warnings": P.warn, "files": man},
                                                indent=1).encode("utf-8"), "generated")
    P.add("SHA256SUMS", "".join(f"{v['sha256']}  {n}\n" for n, v in man.items()).encode("utf-8"), "generated")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = out.with_suffix(".zip.tmp")
    with zipfile.ZipFile(tmp_out, "w", zipfile.ZIP_DEFLATED) as z:
        for n, (b, _) in sorted(P.items.items()):
            z.writestr(n, b)
    os.replace(tmp_out, out)
    print(f"wrote {out} ({out.stat().st_size:,} B, {len(P.items)} files)")
    for w in P.warn:
        print("WARNING", w)
    if a.no_verify:
        return 0 if not [w for w in P.warn if w.startswith("MISSING")] else 1
    # ---- verify: an EMPTY folder, nothing but the zip
    vd = Path(a.verify_dir) if a.verify_dir else Path(tempfile.mkdtemp(prefix="e29_rev2d_zipcheck_"))
    if vd.exists() and any(vd.iterdir()):
        raise SystemExit(f"verify dir not empty: {vd}")
    vd.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out) as z:
        z.extractall(vd)
    with zipfile.ZipFile(out) as z:                                   # the manifest matches the extracted bytes
        m = json.loads(z.read("00_MANIFEST_SHA256.json"))["files"]
    bad_sha = [n for n, v in m.items() if sha((vd / n).read_bytes()) != v["sha256"]]
    print(f"verify dir {vd}: {len(m)} manifest entries, {len(bad_sha)} sha mismatches {bad_sha[:5]}")
    cp = subprocess.run([sys.executable, "-I", "-X", "utf8", str(vd / "00_RUN_TESTS.py")], cwd=str(vd), capture_output=True,
                        text=True, encoding="utf-8", errors="replace", timeout=7200)
    print(cp.stdout[-6000:], cp.stderr[-2000:], sep="\n")
    res = json.loads((vd / "00_RUN_TESTS_RESULTS.json").read_text(encoding="utf-8")) if (vd / "00_RUN_TESTS_RESULTS.json").is_file() else {}
    for n, r in res.items():
        if isinstance(r, dict) and r.get("outside_path_mentions"):
            print(f"NOTE {n}: {r['outside_path_mentions']} log lines mention 'ARC AGI 2' (check {vd / '_logs' / (n + '.log')})")
    ok = cp.returncode == 0 and not bad_sha
    print("SELF-CONTAINED CHECK:", "PASS" if ok else "FAIL", "| logs:", vd / "_logs")
    if ok and not a.keep_verify_dir and not a.verify_dir:
        shutil.rmtree(vd, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
