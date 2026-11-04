"""E26 K100 QUALITY Kaggle CPU kits (lead 2026-11-01): independent CPU kernels on foysalemonshanto, the pinned Py3.12 image of
the notebook, no GPU, no internet, private. Each kernel is fail-soft, writes its JSON report and prints exactly one line
`@@KIT <name> ALL OK@@` or `@@KIT <name> NOT OK: <items>@@`. NOTHING is pushed here (the lead / user pushes).

Everything about the notebook under test is read at kit-build time from the builder's outputs, never hard-coded:
e26_k100q_build_facts.json (notebook path / sha / bytes, payload file + sha + entries, K3 pins, pin steps, seqs table,
captures, window, budget, kernel-metadata sources, embedded shas, push folder, staged dataset folder) and the notebook
itself (default profile, the quality phase name, the pinned payload sha). Rebuild the kits after every notebook rebuild.

Patterns: build_e25_nb6_cpu_kit.py (NB6 kits ALL OK: its e2e driver, real-kit cell runner / C0-vs-E18 / submission blocks are
imported from it and edited with exact-once replacements, so NB6's proven dual-mode checks run unchanged on THIS notebook)
and build_e25_klee100_load_cpu_kit.py (klee100 kits ALL OK: venv_py, vLLM argparse, the MM method-map probe on real Merlin
objects are imported from it).

  units         e26_k100q_unit_tests.py on Py3.12 (all PASS, the report-block section on NB5 QUALITY's real FP8 output, not
                skipped) + the dataset version check: every file of the STAGED tf-ref version is on the mount with the staged
                sha256 (the old files unchanged, k100q_payload_emb.json + K100Q_PAYLOAD.json added), the payload sha = facts =
                the sha pinned in the notebook, every blob decodes to its entry sha, K100Q_PAYLOAD.json agrees.
                Kit substitutes (sha-proven, recorded): the K3_ONLY notebook is given as its K100_EMB line, rebuilt from the
                MOUNTED payload blobs and checked against that line's sha256 cut from the sha-pinned K3_ONLY notebook at
                build time; E18 R2 is given as its first two cells (cell 1 sha-checked); NB5 QUALITY's FP8 phase output with
                nb5_requests.jsonl reduced to the keys the report reads (t0 / t1 / task / branch / kind / completion /
                finish / error / verdict / tool-call count); the unit test's two absolute Windows paths (the old tf-ref
                version folder, the push folder) exist relative to its working directory on Linux and hold the mounted
                old files (each sha-checked against the old version first) / the notebook + its kernel-metadata.json.
  units-report  k100q_report_tests.py (all tests run, none skipped) on NB5 QUALITY's real FP8 + BF16 output (same reduction,
                nb5_report.json byte-exact), the 15-task solutions of the NB5 output audit zip. (Split from units: the
                400 kB commit size limit.)
  e2e-rerun     NB6's e2e driver on THIS notebook (every code cell, one namespace, nb5_fake_server.py, the real Duck payload,
                a fake 6-task hidden test set of training tasks), modes rerun + rerun-retry (FAKE_OOM between K3 rung 1 and
                rung 2 -> E18 R2 ladder step, NO eval file mounted). The klee100 path runs for REAL on the attached mounts
                (models/ and the tf-ref dataset symlinked into the fake input tree): fixkeys overlay, model dir, K6, the
                K7/K3 digest thread, the klee100 tokenizer, the K100Q payload from the dataset, the scales sidecar.
                + E26 checks (merge, k100_setup, payload record, prompts, k100 gates recorded, SKIPPED_RERUN report).
  e2e-commit    the same with RERUN unset: the baked commit profile, its fixed window, the eval subset scored, CELL 0's
                fallback kept, the sha gate on the real digests, the k100 quality report BUILT.
  real          the notebook's cells before CELL 6/7 on the REAL mounts in the commit and the rerun namespace, then: payload
                loading from the dataset files, inputs, the klee100 merge (overlay == v2, model dir, K6) + the K7/K3 digests
                (sha gate), the real Merlin offline install, #60490 applied + verified, the worker extension (k100_worker_ext),
                the scales sidecar (tree + dataset + model-dir copy sha), then for EVERY K3 rung of BOTH profiles' boot plans:
                K6 before launch + launch.build() + host_phase() + `vllm serve --help=all` (VLLM_TARGET_DEVICE=cpu) + argv /
                env checks (pin bytes, --max-num-seqs = the seqs table, captures = facts and cover seqs x 4, FP8 KV dtype = RC2's,
                MTP K3, --language-model-only, --worker-extension-cls, VLLM_QSA_KV_SCALES + its sha, VLLM_SERVER_DEV_MODE,
                lever env), no server started; vLLM's own argparse in the Merlin venv on those argvs; the MM method-map probe on
                real Merlin objects (klee100 kit's MM); CELL 0 vs E18 R2 on the real mount; CELL 0's fallback (strict full
                schema) in both namespaces; the rerun namespace never opens the eval solutions.
                Known CPU-only limits (recorded, not gates): the #60490 import probe's circular import; --help / argparse
                need VLLM_TARGET_DEVICE=cpu (set for those subprocesses only, as in the earlier kits).

Prerequisite: the kits read the NEW version of foysalemonshanto/q27-e25-nb0-tf-ref (KIT_PREREQ.md).
Rules: no Kaggle API calls, no git, no local whole-notebook runs (user rule: long tests on Kaggle CPU kits only).
Run: python build_e26_k100q_cpu_kit.py  ->  D:/ARC AGI 2/_scratch/push_e25/q27-e26-k100q-kit-<name>__foysalemonshanto/
"""
import ast
import base64
import hashlib
import json
import lzma
import re
import runpy
import sys
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
SLUG = "q27-e26-k100q-kit"
OWNER = "foysalemonshanto"


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def once(text, old, new, what, count=1):
    assert text.count(old) == count, (what, text.count(old))
    return text.replace(old, new)


def between(text, start, end):
    i = text.index(start)
    return text[i:text.index(end, i)]


def lf_text(p):
    b = Path(p).read_bytes()
    assert b"\r" not in b, f"CR bytes in {p}"
    return b.decode("utf-8")


# ---------------------------------------------------------------- the notebook under test (from the builder's outputs)
FACTS_PATH = HERE / "e26_k100q_build_facts.json"
FACTS_TXT = lf_text(FACTS_PATH)
FACTS = json.loads(FACTS_TXT)
assert FACTS.get("interim") is False, "interim build (no report lib): rebuild the notebook first"
NB_PATH = Path(FACTS["notebook"])
NB_RAW = NB_PATH.read_bytes()
assert b"\r" not in NB_RAW, "notebook has CR bytes"
NB_SHA = sha(NB_RAW)
assert NB_SHA == FACTS["notebook_sha256"] and len(NB_RAW) == FACTS["bytes"], ("facts != notebook on disk", NB_SHA, len(NB_RAW))
NB_TXT = NB_RAW.decode("utf-8")
NB_NAME = NB_PATH.name
CODE = ["".join(c["source"]) for c in json.loads(NB_TXT)["cells"] if c["cell_type"] == "code"]
EXPORT_PATH = Path(FACTS["export"])
assert "\n\n".join(CODE) == lf_text(EXPORT_PATH), "code export != the notebook's code cells"
PUSH = Path(FACTS["push"])
PUSH_META_TXT = lf_text(PUSH / "kernel-metadata.json")
PUSH_META = json.loads(PUSH_META_TXT)
assert (PUSH / PUSH_META["code_file"]).read_bytes() == NB_RAW, "push folder notebook != the notebook"
assert PUSH_META == FACTS["kernel_metadata"], "push kernel-metadata != facts"
COMMIT_PROFILE = re.search(r'^NB5_DEFAULT_PROFILE = "(\w+)"$', CODE[1], re.M).group(1)
_prow = CODE[1][CODE[1].index(f'    "{COMMIT_PROFILE}": dict('):]
PHASE = re.search(r'nb5_phases=\[\["(\w+)"', _prow).group(1)
PAYNAME, PAYSHA = FACTS["payload"]["file"], FACTS["payload"]["sha256"]
assert f'K100Q_PAYLOAD_NAME = "{PAYNAME}"' in CODE[1] and f'K100Q_PAYLOAD_SHA = "{PAYSHA}"' in CODE[1], "payload pin in CELL 1 != facts"
SCALES_SHA = FACTS["dataset_files_read"]["qsa_kv_scales_262k.json"]
LAB_SHA = FACTS["dataset_files_read"]["k100load_lab.py"]
assert f'K100_SCALES_SHA = "{SCALES_SHA}"' in CODE[4] and f'K100_LAB_SHA = "{LAB_SHA}"' in CODE[4]
PINS, GIBS = [int(x) for x in FACTS["pins"]], [float(x) for x in FACTS["pin_steps_gib"]]
SEQS = [int(FACTS["seqs_table"][str(round(g, 2))]) for g in GIBS]
CAPS = [int(x) for x in FACTS["captures"]]
WINDOW, BUDGET_MIN = FACTS["window_sec"], FACTS["budget_min"]
NB5LIB_SHA = FACTS["embedded"]["nb5_lib.py (in the payload file)"]
# the extra sha-pinned files CELL 1 reads from the same tf-ref version (k100q_ds_text): k100q_report.py, nb4_ppack.py (Q-defer)
DS_PINS = dict(re.findall(r'k100q_ds_text\("([^"]+)", "([0-9a-f]{64})"\)', CODE[1]))
assert all(FACTS["dataset_files_read"].get(n) == h for n, h in DS_PINS.items()), ("CELL 1 dataset pins != facts", DS_PINS)
QDEFER = "nb4_ppack.py" in DS_PINS
if QDEFER:
    PPK_SHA = DS_PINS["nb4_ppack.py"]
    assert sha((HERE / "nb4_ppack.py").read_bytes()) == PPK_SHA == FACTS["qdefer"]["sha256"], "Build nb4_ppack.py != the CELL 1 pin"
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("nb4_ppack_build", HERE / "nb4_ppack.py")
    PPKB = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(PPKB)
    QD_PRE = {rel: v[0] for rel, v in PPKB.PATCHES.items()}
    QD_POST = {rel: v[1] for rel, v in PPKB.PATCHES.items()}
    assert QD_POST == {PPKB.PP_REL: PPKB.PP_POST, PPKB.MR_REL: PPKB.MR_POST, PPKB.GW_REL: PPKB.GW_POST}
    assert "k100_qdefer_apply()" in CODE[4] and 'K100_QD = {"applied": False' in CODE[4]
else:
    PPK_SHA, QD_PRE, QD_POST = None, {}, {}
META = FACTS["kernel_metadata"]
IMAGE, MODELS, DATASETS, COMPS = META["docker_image"], META["model_sources"], META["dataset_sources"], META["competition_sources"]
assert "foysalemonshanto/q27-e25-nb0-tf-ref" in DATASETS
# the fake server's capacity (nb5_fake_server.py: FP8 block 16,515,072 B per card, usable = blocks - 1; F 5 with the lever env)
FAKE_BLOCK, FAKE_F = 16515072, 5
FAKE_CAP = [(g, b // FAKE_BLOCK - 1, FAKE_F, s, min((b // FAKE_BLOCK - 1) // (16 + FAKE_F), s)) for g, b, s in zip(GIBS, PINS, SEQS)]
OOM_AT = (PINS[0] + PINS[1]) // 2          # rerun-retry: rung 1 OOMs, rung 2 boots
# the k100 quality report's success status, as this notebook sets it (audit F: k100q_report.notebook_status -> VALID; before: BUILT)
# the commit's absolute deadman (CELL 1): its seconds from the facts == the notebook's K100_DEADMAN_SEC (None: no deadman in this build)
_dm = re.search(r"^K100_DEADMAN_SEC = (\d+)$", CODE[1], re.M)
DEADMAN_SEC = int(_dm.group(1)) if _dm else None
if DEADMAN_SEC is not None:
    assert DEADMAN_SEC == FACTS.get("deadman_sec") and 'name="k100q_deadman"' in CODE[1] and "daemon=True" in CODE[1], ("deadman facts", DEADMAN_SEC)
KQ_OK_STATUS = "VALID" if "K100Q.notebook_status(" in CODE[-1] else "BUILT"
assert f'"{KQ_OK_STATUS}"' in (CODE[-1] + lf_text(HERE / "k100q_report.py")), KQ_OK_STATUS

# ---------------------------------------------------------------- the staged tf-ref dataset version (and the version it replaces)
STAGE = Path(FACTS["payload"]["staged"])
OLD_DS = Path(r"D:/ARC AGI 2/_scratch/push_e25/ds_e25_nb0_tf_ref")
NOT_MOUNTED = ("dataset-metadata.json",)       # Kaggle consumes it at upload: never on the mount (kit run v1: "missing")
STAGE_SHAS = {f.name: sha(f.read_bytes()) for f in sorted(STAGE.iterdir()) if f.is_file() and f.name not in NOT_MOUNTED}
OLD_SHAS = {f.name: sha(f.read_bytes()) for f in sorted(OLD_DS.iterdir()) if f.is_file() and f.name not in NOT_MOUNTED}
assert not any(p.is_dir() for p in STAGE.iterdir()), "a sub-directory (e.g. __pycache__) in the staged dataset folder"
assert STAGE_SHAS[PAYNAME] == PAYSHA and "K100Q_PAYLOAD.json" in STAGE_SHAS
assert all(STAGE_SHAS.get(n) == h for n, h in OLD_SHAS.items()), "staged version changed an old file"
NEW_FILES = sorted(set(STAGE_SHAS) - set(OLD_SHAS))
PAYLOAD = json.loads((STAGE / PAYNAME).read_text(encoding="utf-8"))
assert {n: v[0] for n, v in PAYLOAD.items()} == FACTS["payload"]["entries"]

# ---------------------------------------------------------------- donors (module import only; their build() is not called)
K6 = runpy.run_path(str(HERE / "build_e25_nb6_cpu_kit.py"), run_name="nb6kit")
K5 = K6["K5"]
NB6KIT_TXT = (HERE / "build_e25_nb6_cpu_kit.py").read_text(encoding="utf-8")
KLEE = (HERE / "build_e25_klee100_load_cpu_kit.py").read_text(encoding="utf-8")
assert K5["IMAGE"] == IMAGE, "the notebook's docker image != the NB5 / NB6 kits' image"
E18_FILES = K6["E18_FILES"]
E18_NB = K6["E18_NB"]
E18_CELLS = json.loads(E18_NB.read_bytes().decode("utf-8"))["cells"]
E18_CELL1_SHA = sha("".join(E18_CELLS[1]["source"]).replace("\r\n", "\n"))
E18_REDUCED = json.dumps({"cells": E18_CELLS[:2], "note": "E26 kit: E18 R2 (sha " + K6["E18_SHA"] + ") cells 0-1 only"})
NB6_NB = HERE.parent / "Q27-XHIGH-E25_NB6_SUBMIT.ipynb"
assert sha(NB6_NB.read_bytes()) == FACTS["nb6_sha256"]
K3_NB = HERE.parent / "Q27-XHIGH-E25_KLEE100_LOAD_K3ONLY.ipynb"
assert sha(K3_NB.read_bytes()) == FACTS["k3only_sha256"]
_k3_line = next(l for c in json.loads(K3_NB.read_bytes().decode("utf-8"))["cells"] for l in "".join(c["source"]).splitlines()
                if "K100_EMB = json.loads(" in l)
K3_LINE_SHA = sha(_k3_line)
K3_TEMPLATE = _k3_line
K3_BLOBS = []
for _n, (_s, _b) in sorted(PAYLOAD.items()):
    if _k3_line.count(_b) == 1:
        K3_TEMPLATE = K3_TEMPLATE.replace(_b, f"@@K100BLOB:{_n}@@")
        K3_BLOBS.append(_n)
assert len(K3_TEMPLATE) < 2000 and K3_BLOBS, (len(K3_TEMPLATE), K3_BLOBS)

# ---------------------------------------------------------------- NB5 QUALITY's real output (reduced) + the 15-task solutions
NB5_OUT = Path(r"D:/ARC AGI 2/_scratch/e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x")
REQ_KEYS = ("t0", "t1", "phase", "task", "branch", "kind", "completion", "finish", "error", "verdict")


def nb5_out_files(phases):
    """{relative path: text}: nb5_report.json byte-exact; per phase the run report, generated attempts, sampler rows, every
    duck/<attempt>/branch_result.json; nb5_requests.jsonl reduced to the keys k100q_report / k100_finish read (tool calls
    kept as a list of names: their count / truthiness is what load_phase_records reads)."""
    out = {"nb5_report.json": lf_text(NB5_OUT / "nb5_report.json")}
    for ph in phases:
        d = NB5_OUT / f"phase_{ph}"
        for n in ("e12_run_report.json", "e12_generated_attempts.json", "nb5_sampler.jsonl"):
            out[f"phase_{ph}/{n}"] = lf_text(d / n)
        for f in sorted((d / "duck").glob("*/branch_result.json")):
            out[f"phase_{ph}/duck/{f.parent.name}/branch_result.json"] = lf_text(f)
        for f in sorted((d / "duck").glob("*/nb5_requests.jsonl")):
            rows = []
            for line in lf_text(f).splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                t = {k: r[k] for k in REQ_KEYS if k in r}
                if r.get("tool_calls"):
                    t["tool_calls"] = [{"name": (c or {}).get("name")} for c in r["tool_calls"]]
                rows.append(json.dumps(t))
            out[f"phase_{ph}/duck/{f.parent.name}/nb5_requests.jsonl"] = "\n".join(rows) + "\n"
    return out


NB5_REPORT_SHA = sha((NB5_OUT / "nb5_report.json").read_bytes())
_zip = HERE.parents[1] / "Experiment Review" / "Audit Zips" / "E25_NB5_QUALITY_output_FLAT_2026-11-01.zip"
with zipfile.ZipFile(_zip) as _z:
    SOL15 = _z.read("arc_eval_15tasks_solutions.json").decode("utf-8")
json.loads(SOL15)


def build_text(name):
    return lf_text(HERE / name)


# the unit test's real-Merlin-tree Q-defer section (it SKIPs in the units kit: no Merlin tree there), cut verbatim and run in the
# real kit on a pristine copy of the INSTALLED tree's three files (prelude: the test's own check / sha / fn_src, ENG = code[4])
QD_REAL_SCRIPT, QD_UNIT_CHECKS = None, 0
UT_TXT = build_text("e26_k100q_unit_tests.py")
QD_SKIP_LINE = "SKIP Q-defer apply / revert on the real Merlin source"
if QDEFER:
    import ast as _ast
    _ut = _ast.parse(UT_TXT)
    _helpers = [_ast.get_source_segment(UT_TXT, n) for n in _ut.body if isinstance(n, _ast.FunctionDef) and n.name in ("check", "sha", "fn_src")]
    assert len(_helpers) == 3
    assert "ENG = code[4]\n" in UT_TXT
    _qns = between(UT_TXT, "_QLOG = []\n", '_ks2 = fn_src(ENG, "k100_setup")')
    _qreal = between(UT_TXT, 'MERLIN_SRC = Path(os.environ.get("K100QT_MERLIN_SRC")', '\nelse:\n    print("' + QD_SKIP_LINE)
    # derived from the test file (never hard-coded): every check( call in the cut text (a check inside a loop runs more often;
    # the kit requires all executed checks PASS and at least this many)
    QD_UNIT_CHECKS = _qns.count("check(") + _qreal.count("check(")
    assert _qreal.count("check(") >= 4, QD_UNIT_CHECKS
    QD_REAL_SCRIPT = ('"""E26 kit: e26_k100q_unit_tests.py\'s real-Merlin-tree Q-defer section, cut verbatim (helpers + _qd_ns + the\n'
                      'MERLIN_SRC block). usage: python e26_qd_real_tree.py <notebook.ipynb> <nb4_ppack.py>  (K100QT_MERLIN_SRC = the tree)"""\n'
                      "import hashlib, json, os, shutil, sys, tempfile, ast\nimport importlib.util as _ilu\nfrom pathlib import Path\n"
                      "RESULTS = []\n\n\n" + "\n\n\n".join(_helpers) + "\n\n\n"
                      'code = ["".join(c["source"]) for c in json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]\n'
                      "ENG = code[4]\n"
                      '_spec = _ilu.spec_from_file_location("nb4_ppack_staged", sys.argv[2])\nPPKS = _ilu.module_from_spec(_spec)\n'
                      "_spec.loader.exec_module(PPKS)\n" + _qns + _qreal + "\nelse:\n"
                      '    RESULTS.append(("real Merlin tree present", False))\n    print("FAIL real Merlin tree missing at", MERLIN_SRC, flush=True)\n'
                      "n_bad = sum(1 for _, ok in RESULTS if not ok)\n"
                      'print(f"\\n{len(RESULTS) - n_bad}/{len(RESULTS)} PASS", flush=True)\nsys.exit(1 if n_bad else 0)\n')
    compile(QD_REAL_SCRIPT, "<e26_qd_real_tree>", "exec")


# ---------------------------------------------------------------------------------------------------------- HEAD
HEAD = K5["HEAD"]
HEAD = once(HEAD, 'FILES["nb5.ipynb"]', 'FILES["e26.ipynb"]', "embedded notebook key")
HEAD = once(HEAD, "E25 NB5 CPU kit", "E26 K100Q CPU kit", "titles", 2)
HEAD = once(HEAD, '"/tmp/nb5kit"', '"/tmp/k100qkit"', "tmp dir")
HEAD = once(HEAD, 'f"nb5_kit_{KIT}_report.json"', 'f"k100q_kit_{KIT}_report.json"', "report name")
HEAD = once(HEAD, 'base64.b64decode("__BLOB__")', 'base64.b85decode("__BLOB__")', "blob codec")
_i = NB6KIT_TXT.index("HEAD += r'''") + len("HEAD += r'''")
HEAD += NB6KIT_TXT[_i:NB6KIT_TXT.index("'''", _i)]          # NB6 kit: grid_ok + strict_schema (verbatim)
HEAD += r'''
KF = json.loads(r"""__KF__""")                  # build-time facts of the notebook under test (from e26_k100q_build_facts.json)
def sha_file(p):
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()
def ref_mount():
    """The tf-ref dataset mount (the klee100 kits' order): datasets/<owner>/<slug>, then the legacy /kaggle/input/<slug>."""
    return next((d for d in (Path(INPUT_ROOT) / "datasets/foysalemonshanto/q27-e25-nb0-tf-ref", Path(INPUT_ROOT) / "q27-e25-nb0-tf-ref")
                 if (d / "nb0_reference.json").is_file()), None)
def dataset_version_check():
    """Every file of the STAGED tf-ref version on the mount with the staged sha256 (old files unchanged + the new ones)."""
    ref = ref_mount()
    out = {"dir": str(ref), "files": {}}
    for n, h in sorted(KF["stage_shas"].items()):
        p = ref / n if ref is not None else None
        e = {"exists": bool(p) and p.is_file(), "new_in_this_version": n in KF["new_files"]}
        if e["exists"]:
            hh = sha_file(p)
            e.update(sha_ok=hh == h, sha256=hh[:16])
        out["files"][n] = e
    out["missing"] = sorted(n for n, e in out["files"].items() if not e["exists"])
    out["sha_bad"] = sorted(n for n, e in out["files"].items() if e["exists"] and not e["sha_ok"])
    out["on_mount_not_staged"] = sorted(p.name for p in ref.iterdir() if p.is_file() and p.name not in KF["stage_shas"]) if ref else None
    out["not_mount_targets"] = ["dataset-metadata.json (Kaggle does not mount it)"]
    out["new_version_mounted"] = ref is not None and all(out["files"][n]["exists"] for n in KF["new_files"])
    out["ok"] = ref is not None and not out["missing"] and not out["sha_bad"]
    if not out["new_version_mounted"]:
        out["prereq"] = ("PREREQ NOT MET: the attached q27-e25-nb0-tf-ref version lacks " + ", ".join(KF["new_files"])
                         + " -- push the staged version (KIT_PREREQ.md), then rerun this kit")
        print(out["prereq"], flush=True)
    return out
def copy_pinned(dst):
    """The sha-pinned files CELL 1 reads from the tf-ref version (k100q_report.py, nb4_ppack.py) are taken from the MOUNT
    (sha == the notebook's pin == the Build file, asserted at kit-build time) instead of being embedded (400-kB cap)."""
    ref, out = ref_mount(), {}
    for n, h in KF["ds_pins"].items():
        p = ref / n if ref is not None else None
        out[n] = bool(p) and p.is_file() and sha_file(p) == h
        if out[n]:
            shutil.copyfile(p, Path(dst) / n)
    return out
def payload_check(ref):
    """The K100Q payload on the mount: sha = facts = the notebook's pin; every blob decodes to its entry sha; the
    K100Q_PAYLOAD.json record agrees."""
    out = {}
    try:
        raw = (ref / KF["payname"]).read_bytes()
        out["sha_ok"] = hashlib.sha256(raw).hexdigest() == KF["paysha"]
        pl = json.loads(raw.decode("utf-8"))
        out["entries_equal_facts"] = {n: v[0] for n, v in pl.items()} == KF["entries"]
        out["blobs_ok"] = {n: hashlib.sha256(lzma.decompress(base64.b85decode(b))).hexdigest() == s for n, (s, b) in pl.items()}
        rec = json.loads((ref / "K100Q_PAYLOAD.json").read_text(encoding="utf-8"))
        out["record_ok"] = rec.get("file") == KF["payname"] and rec.get("sha256") == KF["paysha"] and rec.get("bytes") == len(raw) \
            and rec.get("entries") == KF["entries"]
        out["ok"] = bool(out["sha_ok"] and out["entries_equal_facts"] and all(out["blobs_ok"].values()) and out["record_ok"])
        out["_payload"] = pl
    except BaseException as e:
        out.update(ok=False, error=f"{type(e).__name__}: {e}"[:600])
    return out
'''

# ---------------------------------------------------------------------------------------------------------- units
UNITS = r'''
U = {"dataset": dataset_version_check()}
REF = ref_mount()
PC = payload_check(REF) if REF is not None else {"ok": False, "error": "tf-ref dataset not mounted"}
PL = PC.pop("_payload", None)
U["payload"] = PC
# the other sha-pinned files CELL 1 reads from this version (k100q_ds_text: k100q_report.py, nb4_ppack.py): mounted sha == the pin
U["ds_pins"] = {n: {"pin": h, "ok": REF is not None and (REF / n).is_file() and sha_file(REF / n) == h} for n, h in KF["ds_pins"].items()}
log("dataset", json.dumps({k: v for k, v in U["dataset"].items() if k != "files"})[:1500], "| payload", json.dumps(PC)[:800])
REP["checks"]["units"] = U; save()
B = K / "build"; B.mkdir()
for n, v in FILES.items():
    if not n.startswith(("nb5out/", "push/", "k3only/")):
        (B / n).write_text(v, encoding="utf-8", newline="\n")
U["pinned_from_mount"] = copy_pinned(B)
# kit substitute 1: the K3_ONLY notebook = its K100_EMB line rebuilt from the MOUNTED payload blobs (sha of the real line)
K3S = {"blobs": KF["k3_blobs"]}
try:
    line = FILES["k3only/k100_emb_line.tmpl"]
    for n in KF["k3_blobs"]:
        line = line.replace(f"@@K100BLOB:{n}@@", PL[n][1])
    K3S["line_sha_ok"] = hashlib.sha256(line.encode("utf-8")).hexdigest() == KF["k3_line_sha"]
    (K / KF["k3_name"]).write_text(json.dumps({"cells": [{"cell_type": "code", "source": [line + "\n"]}],
                                               "note": "E26 kit substitute: the K100_EMB line of " + KF["k3_name"]}), encoding="utf-8")
except BaseException as e:
    K3S.update(line_sha_ok=False, error=f"{type(e).__name__}: {e}"[:400])
U["k3only_substitute"] = K3S
# kit substitute 2: E18 R2 cells 0-1 (cell 1 sha-checked against the sha-pinned notebook at build time)
E18P = B / "e18r2_cells01.ipynb"
_c1 = "".join(json.loads(E18P.read_text(encoding="utf-8"))["cells"][1]["source"])
U["e18_cell1_sha_ok"] = hashlib.sha256(_c1.encode("utf-8")).hexdigest() == KF["e18_cell1_sha"]
# kit substitute 3: the unit test's absolute Windows paths, relative to its cwd (B) on Linux
OLD = B / "D:/ARC AGI 2/_scratch/push_e25/ds_e25_nb0_tf_ref"
OLD.mkdir(parents=True)
U["old_version_files"] = {}
for n, h in sorted(KF["old_shas"].items()):
    p = REF / n if REF is not None else None
    ok_ = bool(p) and p.is_file() and sha_file(p) == h
    U["old_version_files"][n] = ok_
    if ok_:
        os.symlink(p, OLD / n)
PUSHD = B / KF["push_literal"]
PUSHD.mkdir(parents=True)
(PUSHD / KF["push_code_file"]).write_text(FILES["e26.ipynb"], encoding="utf-8", newline="\n")
(PUSHD / "kernel-metadata.json").write_text(FILES["push/kernel-metadata.json"], encoding="utf-8", newline="\n")
U["push_dir"] = str(PUSHD)
NO = K / "nb5out"
for n, v in FILES.items():
    if n.startswith("nb5out/"):
        (NO / n[7:]).parent.mkdir(parents=True, exist_ok=True)
        (NO / n[7:]).write_text(v, encoding="utf-8", newline="\n")
U["nb5_report_sha_ok"] = sha_file(NO / "nb5_report.json") == KF["nb5_report_sha"]
REP["checks"]["units"] = U; save()
for k_ in [k_ for k_ in os.environ if k_.startswith(("NB5", "NB6", "E12_", "E15B_", "Q27X_", "RC2_", "E23_", "FAKE_", "K100"))] + [
        "E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN"]:
    os.environ.pop(k_, None)
T = {}
try:
    r = sh("e26_k100q_unit_tests", [sys.executable, str(B / "e26_k100q_unit_tests.py")],
           {"K100QT_NB": str(B / KF["nb_name"]), "K100QT_NB6": str(B / "Q27-XHIGH-E25_NB6_SUBMIT.ipynb"), "K100QT_STAGE": str(REF),
            "K100QT_NB5_OUT": str(NO), "K100QT_E18": str(E18P), "K100Q_NB5_OUT": str(NO),
            "K100Q_SOLUTIONS": str(B / "arc_eval_15tasks_solutions.json"), "PYTHONIOENCODING": "utf-8"}, 3600, cwd=B)
    lines = r["_out"].splitlines()
    m = next((re.match(r"(\d+)/(\d+) PASS", l) for l in lines[::-1] if re.match(r"\d+/\d+ PASS", l)), None)
    T.update(rc=r["rc"], sec=r["sec"], summary=m.group(0) if m else None, fails=[l[:600] for l in lines if l.startswith("FAIL ")][:60],
             skips=[l[:300] for l in lines if l.startswith("SKIP ")], tail=r["tail"][-1500:])
    T["all_pass"] = bool(m) and m.group(1) == m.group(2) and r["rc"] == 0 and not T["fails"]
    # the ONE allowed skip: the real-Merlin-tree Q-defer section (no tree in this kit) -- it runs verbatim in the real kit
    T["skips_other"] = [l for l in T["skips"] if not l.startswith(KF["qd_skip_line"])]
    T["qdefer_real_tree_skipped_here_run_in_real_kit"] = any(l.startswith(KF["qd_skip_line"]) for l in T["skips"])
    T["report_block_ran"] = not T["skips_other"] and "PASS report block:" in r["_out"] and "SKIP report block" not in r["_out"]
    # the deadman's Linux killpg path (SKIPs only on Windows): must RUN and PASS here (no psutil -> os.killpg + pkill)
    T["deadman_killpg_ran"] = (not KF["deadman"]) or ("PASS deadman fire (no psutil)" in r["_out"] and "SKIP deadman" not in r["_out"])
except BaseException as e:
    T.update(all_pass=False, fatal=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
U["unit_tests"] = T
REP["checks"]["units"] = U; save()
log("e26 unit tests", json.dumps({k: v for k, v in T.items() if k != "tb"})[:3000])
DSV = U["dataset"]
verdict({"tf_ref_new_version_mounted": DSV.get("new_version_mounted") is True,
         "tf_ref_every_staged_file_sha_ok": DSV.get("ok") is True,
         "payload_sha_entries_blobs_record_ok": PC.get("ok") is True,
         "dataset_pinned_files_sha_eq_notebook_pins": bool(U["ds_pins"]) and all(v["ok"] for v in U["ds_pins"].values())
             and all(U["pinned_from_mount"].values()),
         "k3only_k100_emb_line_rebuilt_from_mounted_blobs_sha_ok": K3S.get("line_sha_ok") is True,
         "e18_cell1_sha_ok": U["e18_cell1_sha_ok"] is True,
         "old_version_files_unchanged_on_mount": bool(U["old_version_files"]) and all(U["old_version_files"].values()),
         "nb5_report_byte_exact": U["nb5_report_sha_ok"] is True,
         "e26_unit_tests_all_pass": T.get("all_pass") is True, "report_block_ran_on_real_nb5_fp8": T.get("report_block_ran") is True,
         "deadman_linux_killpg_path_ran_and_passed": T.get("deadman_killpg_ran") is True},
        recorded={"summary": T.get("summary"), "fails": T.get("fails"), "skips": T.get("skips"), "prereq": DSV.get("prereq"),
                  "qdefer_real_tree_section": "skipped here (no Merlin tree), run verbatim in q27-e26-k100q-kit-real"
                  if T.get("qdefer_real_tree_skipped_here_run_in_real_kit") else "ran here", "ds_pins": U["ds_pins"],
                  "on_mount_not_staged": DSV.get("on_mount_not_staged"), "dataset_dir": DSV.get("dir")})
'''

UNITS_REPORT = r'''
U = {"dataset": dataset_version_check()}
B = K / "build"; B.mkdir()
for n, v in FILES.items():
    if not n.startswith("nb5out/"):
        (B / n).write_text(v, encoding="utf-8", newline="\n")
U["pinned_from_mount"] = copy_pinned(B)
NO = K / "nb5out"
for n, v in FILES.items():
    if n.startswith("nb5out/"):
        (NO / n[7:]).parent.mkdir(parents=True, exist_ok=True)
        (NO / n[7:]).write_text(v, encoding="utf-8", newline="\n")
U["nb5_report_sha_ok"] = sha_file(NO / "nb5_report.json") == KF["nb5_report_sha"]
U["embedded_shas"] = {n: sha_file(B / n) == h for n, h in KF["report_shas"].items()}
T = {}
try:
    r = sh("k100q_report_tests", [sys.executable, str(B / "k100q_report_tests.py")],
           {"K100Q_NB5_OUT": str(NO), "K100Q_SOLUTIONS": str(B / "arc_eval_15tasks_solutions.json"), "PYTHONIOENCODING": "utf-8"},
           1800, cwd=B)
    o = r["_out"]
    m = re.search(r"^Ran (\d+) tests? in", o, re.M)
    T.update(rc=r["rc"], sec=r["sec"], ran=int(m.group(1)) if m else None, ok_line=bool(re.search(r"^OK\s*$", o, re.M)),
             skipped=re.findall(r"\.\.\. skipped (.*)$", o, re.M)[:30], failures=re.findall(r"^(?:FAIL|ERROR): (.*)$", o, re.M)[:30],
             tail=r["tail"][-1500:])
    T["all_pass"] = r["rc"] == 0 and T["ok_line"] and not T["failures"] and T["ran"] is not None and T["ran"] >= KF["report_tests_min"]
    T["none_skipped"] = not T["skipped"]
except BaseException as e:
    T.update(all_pass=False, fatal=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
U["report_tests"] = T
REP["checks"]["units_report"] = U; save()
log("k100q report tests", json.dumps({k: v for k, v in T.items() if k != "tb"})[:3000])
verdict({"nb5_report_byte_exact": U["nb5_report_sha_ok"] is True, "embedded_files_sha_ok": all(U["embedded_shas"].values()),
         "pinned_files_from_mount_sha_ok": all(U["pinned_from_mount"].values()),
         "report_tests_all_pass": T.get("all_pass") is True, "report_tests_none_skipped_real_data_ran": T.get("none_skipped") is True},
        recorded={"ran": T.get("ran"), "skipped": T.get("skipped"), "failures": T.get("failures"),
                  "tf_ref_new_version_mounted": U["dataset"].get("new_version_mounted")})
'''

# ---------------------------------------------------------------------------------------------------------- e2e driver (NB6's, edited)
E2E_DRV = K6["NB6_E2E"]
_doc_end = E2E_DRV.index("import faulthandler")
E2E_DRV = r'''"""E26 K100 QUALITY e2e (Kaggle CPU kit only) = NB6's e2e driver (build_e25_nb6_cpu_kit.NB6_E2E, NB6 kits ALL OK) on the
E26 notebook: every code cell in order in one namespace against nb5_fake_server.py (E11_HARNESS=1, no GPU), the real Duck
payload, a fake 6-task hidden test set; the klee100 model path (overlay, model dir, K6, digests, tokenizer, payload, scales)
runs on the REAL attached mounts (symlinked into the fake input tree). Edits vs NB6: marked "E26" below.
usage: python e26_e2e.py <notebook.ipynb> rerun|rerun-retry|commit
Output: one line per check, then @@E26_E2E_OK@@ (exit 0) or @@E26_E2E_FAIL@@ (exit 1)."""
''' + E2E_DRV[_doc_end:]
E2E_DRV = E2E_DRV.replace("NB6_E2E", "E26_E2E").replace("[nb6 e2e", "[e26 e2e")
E2E_DRV = once(E2E_DRV, "shutil.copytree(SRC_INPUT, INP)\n",
               "shutil.copytree(SRC_INPUT, INP, symlinks=True)   # E26: models/ + the tf-ref dataset are symlinks to the real mounts\n",
               "copytree symlinks")
E2E_DRV = once(E2E_DRV, '    os.environ["FAKE_OOM_ABOVE_BYTES"] = str(2479838700 + int(0.8 * 2 ** 30))         # RC2 FP8 pin + 0.8 GiB\n',
               '    os.environ["FAKE_OOM_ABOVE_BYTES"] = "__OOM__"          # E26: between K3 rung 1 (__PIN1__) and rung 2 (__PIN2__)\n',
               "OOM threshold")
E2E_DRV = once(E2E_DRV, 'for k in [k for k in os.environ if k.startswith(("NB5", "E12_", "E15B_", "FAKE_", "Q27X_"))] + [\n',
               'for k in [k for k in os.environ if k.startswith(("NB5", "E12_", "E15B_", "FAKE_", "Q27X_", "K100"))] + [   # E26: + K100*\n',
               "env pop")
E2E_DRV = once(E2E_DRV, 'want_profile = "RERUN_PROD" if RR else "NB6COMMIT"', 'want_profile = "RERUN_PROD" if RR else "__COMMIT_PROFILE__"   # E26',
               "profile")
E2E_DRV = once(E2E_DRV, '''check("phases run = the profile's phases (['EVAL']), all DONE",
      lambda: want_ph == ["EVAL"] and list(ph) == want_ph''',
               '''check(f"phases run = the profile's phases ({['EVAL'] if RR else ['__PHASE__']}), all DONE",   # E26
      lambda: want_ph == (["EVAL"] if RR else ["__PHASE__"]) and list(ph) == want_ph''', "phases")
_r0 = E2E_DRV.index("if RETRY:\n    check(\"boot retry (NB6 P3")
_r1 = E2E_DRV.index("cap = b0.get(\"capacity\") or {}", _r0)
E2E_DRV = E2E_DRV[:_r0] + r'''if RETRY:          # E26: the K3 ladder (rung 1 OOM -> rung 2), E18 R2's engine ladder
    check("boot retry (NB6 P3 = E18 R2 engine ladder) on the K3 ladder: one boot; attempt 1 (rung 1, pin +__G1__ GiB, lever env) "
          "failed with an OOM line before any phase, attempt 2 (rung 2, pin +__G2__ GiB, the same lever env) booted (FP8, MTP on, "
          "probes ok); the ready limit of the later attempt is 15 min (E18 R2: 30 / 15 / 15)",
          lambda: len(boots) == 1 and len(b0["tries"]) == 2 and all(abs(t_["pin_gib"] - g_) < 1e-6 for t_, g_ in zip(b0["tries"], (__G1__, __G2__)))
          and b0["tries"][0]["ladder"]["failed"] and b0["tries"][0]["ladder"]["oom_lines"] and b0["tries"][1].get("status") in (None, "DONE")
          and abs(b0["pin_used"]["gib"] - __G2__) < 1e-6 and b0["tries"][1]["levers"] == {"MERLIN_SHARE_DRAFT_WORKSPACE": "1"}
          and b0["kv_dtype"] == "fp8" and b0["mtp"] is True and (b0.get("probes") or {}).get("fatal") is None
          and G.get("NB6_READY_SEC") == 15 * 60, _bsum + f" | NB6_READY_SEC {G.get('NB6_READY_SEC')}")
else:
    check("one boot (FP8, MTP on), the first K3 rung booted (ready limit: 30 min = E18 R2 in the rerun, NB5_READY_TIMEOUT in the "
          "commit), probes without a fatal",
          lambda: len(boots) == 1 and b0["kv_dtype"] == "fp8" and b0["mtp"] is True and (b0.get("probes") or {}).get("fatal") is None
          and len(b0["tries"]) == 1 and G.get("NB6_READY_SEC") == (30 * 60 if RR else G.get("NB5_READY_TIMEOUT")), _bsum)
''' + E2E_DRV[_r1:]
E2E_DRV = once(E2E_DRV, "_cw = (0.75, 197, 5, 8, 8) if RETRY else (1.0, 214, 5, 9, 9)     # pin, fake pool, F, launch seqs, workers\n",
               "_cw = __CW2__ if RETRY else __CW1__     # E26: pin, fake pool, F, launch seqs (K100 table), workers\n", "capacity tuple")
E2E_DRV = once(E2E_DRV, 'lambda: b0["pin_used"]["gib"] == _cw[0] and cap["pool_blocks"] == _cw[1]',
               'lambda: abs(b0["pin_used"]["gib"] - _cw[0]) < 1e-6 and cap["pool_blocks"] == _cw[1]', "capacity pin")
E2E_DRV = once(E2E_DRV, 'main = ph.get("EVAL") or {}\n', 'main = ph.get("EVAL" if RR else "__PHASE__") or {}   # E26\n', "main phase")
E2E_DRV = once(E2E_DRV, 'check("commit: the baked 40-min budget (no override)", lambda: G["TOTAL_BUDGET_SEC"] == 2400, G.get("TOTAL_BUDGET_SEC"))',
               'check("commit: the baked __BUDGET_MIN__-min budget (no override)", lambda: G["TOTAL_BUDGET_SEC"] == __BUDGET_SEC__, G.get("TOTAL_BUDGET_SEC"))',
               "budget")
E2E_DRV = once(E2E_DRV, "the eval subset (NB6COMMIT task_list)", "the eval subset (__COMMIT_PROFILE__ task_list)", "label")
E2E_DRV = once(E2E_DRV, 'lambda: [L_["env_levers"] for L_ in launches if L_["outcome"] == "ready"] == [_lv] and b0["server_env_levers"] == _lv,',
               'lambda: [L_["env_levers"] for L_ in launches if L_["outcome"] == "ready"] == [_lv] and all(b0["server_env_levers"].get(k_) == v_ '
               'for k_, v_ in _lv.items())\n      and all(b0["server_env_levers"][k_] == (G.get("NB5_BASE_ENV") or {}).get(k_) for k_ in '
               'set(b0["server_env_levers"]) - set(_lv)),   # E26: the record also carries NB5_BASE_ENV (VLLM_SERVER_DEV_MODE)',
               "lever env check")
E2E_DRV = once(E2E_DRV, "bad = [c for c in checks if not c[1]]\n", r'''

# ---- E26 K100: the klee100 path on the REAL mounts (the engine is the fake server)
REFD = INP / "datasets" / "foysalemonshanto" / "q27-e25-nb0-tf-ref"
_R = G.get("R") or {}
_mg = ((_R.get("sections") or {}).get("A0") or {}).get("merge") or {}
REC["k100_merge"] = {k: _mg.get(k) for k in ("ok", "bad", "why", "k6", "digest_plan", "ple_expected_equals_digests", "digest_manifest_sha_ok")}
check("E26: klee100 merge on the real mounts: fixkeys overlay regenerated == the v2 artifact, model dir, K6 (every indexed shard), "
      "FP8 PLE shards == the digest manifest, the manifest sha pinned",
      lambda: _mg["ok"] is True and _mg["overlay"]["ok"] is True and all(_mg["overlay"]["equal_v2_artifact"].values())
      and _mg["k6"]["ok"] is True and _mg["ple_expected_equals_digests"] is True and _mg["digest_manifest_sha_ok"] is True,
      json.dumps(REC["k100_merge"], default=str)[:1200])
_ks = rep.get("k100_setup") or {}
check("E26: k100_setup before the first boot: OVERLAY_MERGE_K6 ok, worker extension = the fake record (no Merlin tree), the scales "
      "sidecar sha-OK from the tf-ref dataset, copied into the klee100 model dir",
      lambda: _ks["overlay_merge_k6"]["ok"] is True and _ks["worker_ext"].get("fake") is True and _ks["scales"]["ok"] is True
      and _ks["scales"]["sha256"] == "__SCALES__" and Path(_ks["scales"]["src"]).resolve() == (REFD / "qsa_kv_scales_262k.json").resolve()
      and Path(_ks["scales"]["file"]).is_file(), json.dumps(_ks, default=str)[:1200])
KQP = W / "q27x" / "k100_quality_report.json"
KQ = json.loads(KQP.read_text(encoding="utf-8")) if KQP.is_file() else {}
if __QDEFER__:
    _qd = _ks.get("qdefer")
    _QDG = G.get("K100_QD") or {}
    check("E26 Q-defer on the fake server: k100_setup recorded the qdefer field = FAKE_SKIPPED (applied False, no env), no launch "
          "got NB4_PPDEFER / NB4_PPACK, the k100 quality report carries the qdefer field (FAKE_SKIPPED)",
          lambda: isinstance(_qd, dict) and _qd["status"] == "FAKE_SKIPPED" and _qd["applied"] is False and _qd["env"] == {}
          and _QDG["status"] == "FAKE_SKIPPED" and _QDG["env"] == {} and _QDG["launch_env"] and all(v is None for v in _QDG["launch_env"].values())
          and KQ["qdefer"]["status"] == "FAKE_SKIPPED" and KQ["qdefer"]["applied"] is False,
          json.dumps({"setup": _qd, "K100_QD": {k: _QDG.get(k) for k in ("status", "applied", "env", "launch_env")},
                      "report": (KQ.get("qdefer") if KQ else None)}, default=str)[:1500])
_pu = (_R.get("setup") or {}).get("k100q_payload") or {}
check("E26: the K100Q payload came from the tf-ref dataset file (sha pinned), every blob sha-OK (CELL 2b decode), the lab from the "
      "dataset, nb5_lib from the payload (sha 18efcbc1 checked at load)",
      lambda: _pu["sha256"] == "__PAYSHA__" and Path(_pu["file"]).resolve() == (REFD / "__PAYNAME__").resolve()
      and all(G["K1"]["embedded_sha_ok"].values()) and Path(G["K100_LAB_FILE"]).resolve() == (REFD / "k100load_lab.py").resolve()
      and G["CONFIG"]["nb5"]["lib_sha256"] == "__NB5LIBSHA__", json.dumps(_pu, default=str))
check("E26: prompts rendered with the klee100 tokenizer from the merged dir (pinned tokenizer / template hashes)",
      lambda: G["MERLIN_PROMPTS"] is True and (rep.get("prompt_render") or {}).get("hashes_ok") is True, rep.get("prompt_render"))
_sg = (_R.get("setup") or {}).get("sha_gate") or {}
if not RR:
    _dg = (G.get("K1") or {}).get("digests") or {}
    check("commit: the sha gate passed on the real K7 / K3 streaming digests (every planned file checked)",
          lambda: _sg["ok"] is True and _dg["ok"] is True and _dg["n_checked"] == _dg["n_planned"] > 0,
          json.dumps({"gate": _sg, "digests": {k: _dg.get(k) for k in ("ok", "bad", "n_planned", "n_checked", "gb", "sec")}}))
_kg = ((b0.get("tries") or [{}])[-1] or {}).get("k100_gates") or {}
check("E26: the klee100 boot gates ran on the try that booted (method map + static scales recorded; a fake server is never gated)",
      lambda: _kg["gated"] is False and _kg["fatal"] is None and set(_kg["checks"]) == {"GPU_METHOD_MAP_PASS", "STATIC_SCALES_APPLIED"},
      json.dumps({k: _kg.get(k) for k in ("checks", "ok", "fatal", "gated", "method_map_verdict")}, default=str))
_la = [L_ for L_ in launches if L_["outcome"] == "ready"]
def _last(a, f):
    i = len(a) - 1 - a[::-1].index(f) if f in a else -1
    return a[i + 1] if 0 <= i < len(a) - 1 else None
check("E26: the booted launch argv: --max-num-seqs = the K3 rung's seqs, captures = __CAPS__, --language-model-only, "
      "--worker-extension-cls nb1_worker_ext.NB1WorkerExt, FP8 KV, the rung's pin bytes",
      lambda: len(_la) == 1 and _last(_la[0]["argv"], "--max-num-seqs") == str(_cw[3])
      and json.loads(_last(_la[0]["argv"], "--compilation-config"))["cudagraph_capture_sizes"] == __CAPS__
      and "--language-model-only" in _la[0]["argv"] and _last(_la[0]["argv"], "--worker-extension-cls") == "nb1_worker_ext.NB1WorkerExt"
      and _last(_la[0]["argv"], "--kv-cache-dtype").startswith("fp8")
      and _last(_la[0]["argv"], "--kv-cache-memory-bytes") == str(__PIN2__ if RETRY else __PIN1__), json.dumps(_la, default=str)[:1500])
KQP = W / "q27x" / "k100_quality_report.json"
KQ = json.loads(KQP.read_text(encoding="utf-8")) if KQP.is_file() else {}
REC["k100_quality"] = {k: KQ.get(k) for k in ("status", "error", "boot")}
if RR:
    check("rerun: the k100 quality report is SKIPPED_RERUN (written, no scoring)", lambda: KQ["status"] == "SKIPPED_RERUN", REC["k100_quality"])
else:
    _wp = main.get("window_plan") or {}
    check("commit: the fixed solve window = __WINDOW__ s (planned, plan fixed True)",
          lambda: _wp["planned_sec"] == __WINDOW__ and _wp["plan"]["fixed"] is True, json.dumps(_wp, default=str))
    _pc = ((rep.get("quality_ab") or {}).get("per_config") or {}).get("__PHASE__") or {}
    check("commit: quality per-task table: every task of the profile's list, 2 attempts each, outcomes in the NB5 set",
          lambda: set(_pc["tasks"]) == set(P["task_list"]) and all(len(v["attempts"]) == 2 for v in _pc["tasks"].values())
          and all(a["outcome"] in ("ACCEPTED", "BUDGET_END", "INCOMPLETE_WINDOW", "FAILED_OR_INTERRUPTED", "NOT_STARTED")
                  for v in _pc["tasks"].values() for a in v["attempts"]), json.dumps(_pc.get("summary"), default=str)[:1200])
    check("commit: the k100 quality report vs the embedded NB5 FP8 reference: final status __KQOK__ (the notebook's success "
          "status), a finish row per task, paired counts, no report errors, every task with gold, the markdown table",
          lambda: KQ["status"] == "__KQOK__" and KQ["finish"]["tasks"] == len(P["task_list"]) and not KQ["quality"].get("errors")
          and KQ["solutions_check"]["missing"] == []
          and "counts" in KQ["quality"]["paired"] and bool(KQ.get("table_md")),
          json.dumps({k: KQ.get(k) for k in ("status", "error", "finish")}, default=str)[:1500])
if __DEADMAN__:          # the commit's absolute deadman (first code of CELL 1): armed + daemon + fire at T0 + deadman_sec, NOT fired
    import threading as _th
    _DM = G.get("K100_DEADMAN") or {}
    _dmt = [t_ for t_ in _th.enumerate() if t_.name == "k100q_deadman"]
    _dmf = W / "k100q_deadman.txt"
    REC["deadman"] = {"armed": _DM.get("armed"), "fire_in_sec": round(_DM["fire_pc"] - G["NOTEBOOK_T0"], 3) if _DM.get("fire_pc") else None,
                      "threads": [(t_.name, t_.daemon, t_.is_alive()) for t_ in _dmt], "marker_exists": _dmf.exists(),
                      "sec": G.get("K100_DEADMAN_SEC")}
    if RR:
        check("rerun: NO deadman (RERUN_PROD): not armed, no k100q_deadman thread, no marker file",
              lambda: _DM.get("armed") is False and "thread" not in _DM and not _dmt and not _dmf.exists(), REC["deadman"])
    else:
        check("commit: the deadman is armed: one daemon thread 'k100q_deadman', fire time = NOTEBOOK_T0 + __DEADMAN_SEC__ s "
              "(__DEADMAN_MIN__ min), still sleeping at the end (it did NOT fire: alive, no marker file, the driver got here)",
              lambda: _DM["armed"] is True and G["K100_DEADMAN_SEC"] == __DEADMAN_SEC__ and abs(_DM["fire_pc"] - (G["NOTEBOOK_T0"] + __DEADMAN_SEC__)) < 1e-6
              and _DM["thread"] in _dmt and len(_dmt) == 1 and _dmt[0].daemon is True and _dmt[0].is_alive() and not _dmf.exists()
              and T_END < _DM["fire_pc"], REC["deadman"])
import zipfile as _zf
_bz = W / "Q27X_NB5_BUNDLE.zip"
check("E26: the bundle zip carries k100_quality_report.json",
      lambda: any(n_.endswith("k100_quality_report.json") for n_ in _zf.ZipFile(_bz).namelist()), str(_bz))
bad = [c for c in checks if not c[1]]
''', "E26 checks")
for _k, _v in (("__OOM__", str(OOM_AT)), ("__PIN1__", str(PINS[0])), ("__PIN2__", str(PINS[1])), ("__COMMIT_PROFILE__", COMMIT_PROFILE),
               ("__PHASE__", PHASE), ("__G1__", repr(GIBS[0])), ("__G2__", repr(GIBS[1])), ("__CW1__", repr(FAKE_CAP[0])),
               ("__CW2__", repr(FAKE_CAP[1])), ("__BUDGET_MIN__", str(BUDGET_MIN)), ("__BUDGET_SEC__", str(BUDGET_MIN * 60)),
               ("__SCALES__", SCALES_SHA), ("__PAYSHA__", PAYSHA), ("__PAYNAME__", PAYNAME), ("__NB5LIBSHA__", NB5LIB_SHA),
               ("__CAPS__", json.dumps(CAPS)), ("__KQOK__", KQ_OK_STATUS), ("__QDEFER__", repr(QDEFER)),
               ("__DEADMAN__", repr(DEADMAN_SEC is not None)), ("__DEADMAN_SEC__", str(DEADMAN_SEC)),
               ("__DEADMAN_MIN__", str((DEADMAN_SEC or 0) / 60)),
               ("__WINDOW__", repr(float(WINDOW)) if isinstance(WINDOW, float) else str(WINDOW))):
    E2E_DRV = E2E_DRV.replace(_k, _v)
assert "__" + "PHASE__" not in E2E_DRV and "NB6COMMIT" not in E2E_DRV, "unreplaced e2e placeholders"
compile(E2E_DRV, "<e26_e2e>", "exec")

E2E = r'''
PAY = find_payload(); REP["checks"]["payload"] = PAY; save()
log("payload", json.dumps(PAY)[:1200])
U = {"dataset": dataset_version_check()}
REF = ref_mount()
B = K / "build"; B.mkdir()
for n in ("e26.ipynb", "e26_e2e.py", "nb5_fake_server.py", "e18_cell0.py", "e18_writer.py"):
    (B / n).write_text(FILES[n], encoding="utf-8", newline="\n")
for k_ in [k_ for k_ in os.environ if k_.startswith(("NB5", "NB6", "E12_", "E15B_", "Q27X_", "RC2_", "E23_", "FAKE_", "K100"))] + [
        "E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN"]:
    os.environ.pop(k_, None)
U["comp_files"] = {p.name: [p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()[:16]] for p in sorted(COMPD.glob("*.json"))}
U["comp_has_eval_solutions"] = (COMPD / "arc-agi_evaluation_solutions.json").is_file()
# the fake input tree: models/ -> the real model mounts; the tf-ref dataset -> its real mount (the driver adds the competition
# dir with the fake hidden test set and the Duck payload copy)
FIN = K / "fin"
(FIN / "datasets" / "foysalemonshanto").mkdir(parents=True)
os.symlink(Path(INPUT_ROOT) / "models", FIN / "models")
if REF is not None:
    os.symlink(REF, FIN / "datasets" / "foysalemonshanto" / "q27-e25-nb0-tf-ref")
MDR = Path(INPUT_ROOT) / "models" / "foysalemonshanto"
U["mounts"] = {"klee100": sorted(str(p.parent) for p in (MDR / "qwen38-flash-next-klee100-3bpw-mtp").glob("*/*/*/KAGGLE_MIRROR_MANIFEST.json")),
               "fp8ple": (MDR / "qwen38-flash-next-fp8ple-overlay/pytorch/albucino-fp8ple/1/model-plefp8-00000.safetensors").is_file(),
               "merlin_env": (MDR / "e23-merlin-env/other/v1/1/merlin_offline_tar.bin").is_file(), "tf_ref": str(REF)}
log("setup", json.dumps({k: v for k, v in U.items() if k != "dataset"})[:1500], "| dataset", json.dumps({k: v for k, v in U["dataset"].items() if k != "files"})[:800])
REP["checks"]["setup"] = U; save()
EALL = {}
for mode in "__MODE__".split(","):          # one or more e2e modes, one after the other (each its own work dir)
    _max = 3500 if mode != "commit" else __COMMIT_MAX__   # commit: the notebook's own budget + 30 min
    try:
        r = sh("e2e_" + mode, [sys.executable, str(B / "e26_e2e.py"), str(B / "e26.ipynb"), mode],
               {"E26_E2E_INPUT": str(FIN), "E26_E2E_DATA": str(COMPD), "E26_E2E_PAYSRC": str(Path(PAY["used"]).parent),
                "E26_E2E_ROOT": str(K / "e2e"), "E26_E2E_BUDGET": "1800", "E26_E2E_BUDGET_RETRY": "3000", "E26_E2E_MAX_SEC": str(_max),
                "PYTHONIOENCODING": "utf-8"}, _max + 300, cwd=B)
        lines = r["_out"].splitlines()
        E = {"rc": r["rc"], "sec": r["sec"], "ok": r["rc"] == 0 and "@@E26_E2E_OK@@" in r["_out"],
             "summary": next((l for l in lines if re.match(r"\d+/\d+ PASS", l)), None),
             "fails": [l[:900] for l in lines if "e2e FAIL]" in l][:40],
             "passes": [l[:200] for l in lines if "e2e PASS]" in l],
             "record": next((json.loads(l[15:]) for l in lines if l.startswith("E26_E2E_RECORD ")), None),
             "fake_set": next((l[31:] for l in lines if l.startswith("[e26 e2e] fake hidden test set ")), None), "tail": r["tail"][-1500:]}
    except BaseException as e:
        E = {"ok": False, "fatal": f"{type(e).__name__}: {e}"[:600], "tb": traceback.format_exc()[-2000:]}
    EALL[mode] = E
    log("e2e", mode, json.dumps({k: v for k, v in E.items() if k not in ("tail", "passes")}, default=str)[:4000])
    REP["checks"]["e2e"] = EALL; save()
verdict({"payload_sha_ok_under_input": PAY["used"] is not None,
         "tf_ref_new_version_mounted": U["dataset"].get("new_version_mounted") is True,
         "tf_ref_every_staged_file_sha_ok": U["dataset"].get("ok") is True,
         "klee100_ple_merlin_env_mounted": len(U["mounts"]["klee100"]) == 1 and U["mounts"]["fp8ple"] and U["mounts"]["merlin_env"],
         "competition_eval_challenges_and_solutions": bool(U.get("comp_has_eval_solutions")),
         **{f"e2e_{m}": bool(E.get("ok")) for m, E in EALL.items()}},
        recorded={m: {"summary": E.get("summary"), "fails": E.get("fails"), "record": E.get("record")} for m, E in EALL.items()}
        | {"payload_used": PAY.get("used"), "prereq": U["dataset"].get("prereq")})
'''

# ---------------------------------------------------------------------------------------------------------- real
REAL6 = K6["REAL"]
VENV_PY = K6["VENV_PY"]
ARGPARSE = K6["ARGPARSE"]
assert "language_model_only" in ARGPARSE and "worker_extension_cls" in ARGPARSE
R_HEAD = between(REAL6, "PAY = find_payload(); REP[\"checks\"][\"payload\"] = PAY; save()", "def run_ns(label, rerun):")
R_HEAD = once(R_HEAD, 'NB = json.loads(FILES["nb6.ipynb"])', 'NB = json.loads(FILES["e26.ipynb"])', "real nb key")
R_HEAD = once(R_HEAD, '"E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN", "VLLM_TARGET_DEVICE"]:', '"E11_HARNESS", "KAGGLE_IS_COMPETITION_RERUN", "VLLM_TARGET_DEVICE",\n        "K100_PAYLOAD_DIR", "K100_PROFILE", "K100_FAKE_GATES"]:', "real env pop")
R_RUN = between(REAL6, "def run_ns(label, rerun):", "step(\"inputs\"")
R_RUN = once(R_RUN, 'f"<nb6 {label} cell {i}>"', 'f"<e26 {label} cell {i}>"', "cell label")
R_SHARED = between(REAL6, "SHARED = sorted(", "# E18 R2 CELL 0 (LB 31.53)")
R_C0 = between(REAL6, "# E18 R2 CELL 0 (LB 31.53)", "REP[\"checks\"][\"cells_rerun\"] = CELLS_R")
R_C0 = once(R_C0, '(K / "nb6_cell0.py").write_text(CODE[0], encoding="utf-8")', '(K / "nb6_cell0.py").write_text(CODE[0], encoding="utf-8")   # (the E26 CELL 0)', "c0")
R_MERGE_NS = between(REAL6, "REP[\"checks\"][\"cells_rerun\"] = CELLS_R", "# ---------------------------------------------------------------- every boot try")
R_SUB = between(REAL6, "# ---------------------------------------------------------------- dual mode: CELL 0's fallback", "pu = (D3.get")
MM_SEC = between(KLEE, "MM_PROBE = r\"\"\"", "# ---------------------------------------------------------------- R7")
MM_SEC = once(MM_SEC, '(K / "mm_ext.py").write_text(G["NB1_EXT_SRC"])',
              '(K / "mm_ext.py").write_text((G["EMB"] / "nb1_worker_ext.py").read_text(encoding="utf-8"))   # E26: the payload\'s worker extension',
              "mm ext src")

REAL = R_HEAD + R_RUN + r'''
# ---------------------------------------------------------------- E26: the dataset version + payload loading from the dataset files
DSV = dataset_version_check()
REF = ref_mount()
PCK = payload_check(REF) if REF is not None else {"ok": False, "error": "tf-ref dataset not mounted"}
PCK.pop("_payload", None)
_pu = G.get("K100Q_PAYLOAD_USED") or {}
PLD = {"used": _pu, "lab_file": str(G.get("K100_LAB_FILE")), "emb_ok": (G.get("K1") or {}).get("embedded_sha_ok"),
       "nb5_lib_sha256": ((G.get("CONFIG") or {}).get("nb5") or {}).get("lib_sha256"), "check": PCK}
PLD["ok"] = bool(REF is not None and _pu.get("sha256") == KF["paysha"] and _pu.get("file") and Path(_pu["file"]).resolve() == (REF / KF["payname"]).resolve()
                 and G.get("K100_LAB_FILE") is not None and Path(G["K100_LAB_FILE"]).resolve() == (REF / "k100load_lab.py").resolve()
                 and PLD["emb_ok"] and all(PLD["emb_ok"].values()) and PLD["nb5_lib_sha256"] == KF["nb5lib_sha"] and PCK.get("ok"))
REP["checks"]["dataset"] = DSV; REP["checks"]["payload_loading"] = PLD; save()
log("dataset", json.dumps({k: v for k, v in DSV.items() if k != "files"})[:1200], "| payload loading", json.dumps(PLD, default=str)[:1200])
step("inputs", lambda: (G["nb5_rc2_inputs"](), {"inputs": G["R"]["inputs"]})[1])
step("prompts", lambda: (G["nb5_rc2_prompts"](), {"merge": {k: v for k, v in G["R"]["sections"]["A0"]["merge"].items() if k != "overlay"},
                                                   "overlay": {k: v for k, v in (G["R"]["sections"]["A0"]["merge"].get("overlay") or {}).items()
                                                               if k not in ("files", "tb")},
                                                   "host": G["R"]["sections"]["A0"]["host"],
                                                   "prompt_render": {k: v for k, v in G["R"]["prompt_render"].items() if "text" not in k}})[1])
step("install", lambda: (lambda e: {"kv": dict(e["kv"]), "cuda_home": e["cuda_home"]})(G["merlin_install"]()))
step("patch_60490", lambda: G["nb5_apply_60490"]())
step("sha256", lambda: (lambda ok_why: {"sha_ok": ok_why[0], "why": ok_why[1], "detail": G["R"]["setup"].get("sha256")})(G["sha_gate"](3600)))

# ---------------------------------------------------------------- Q-defer: a pristine copy of the INSTALLED tree's three files ->
# nb4_ppack_tests.py and the unit test's real-tree Q-defer section (cut verbatim) run on it, before k100_setup patches the tree
QD = {"enabled": KF["qdefer"]}
if KF["qdefer"]:
    try:
        _eng = G["MERLIN_ENG"]
        QTREE, QPY = Path(_eng["kv"]["MERLIN_TREE"]), _eng["kv"]["MERLIN_VENV"] + "/bin/python"
        PPK = G.get("PPK")
        QD["lib"] = {"loaded": PPK is not None, "error": G.get("PPK_ERROR"), "file": getattr(PPK, "__file__", None)}
        QD["lib"]["sha_ok"] = PPK is not None and sha_file(PPK.__file__) == KF["ppk_sha"]
        QD["installed_before_setup"] = PPK.verify_ppack_patch(QTREE)
        QD["pristine_pre_shas_ok"] = (QD["installed_before_setup"]["state"] == "not_applied"
                                      and {r_: v["sha"] for r_, v in QD["installed_before_setup"]["files"].items()} == KF["qd_pre"])
        QSRC = K / "merlin_src_pristine"
        for rel in PPK.PATCHES:
            (QSRC / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(QTREE / rel, QSRC / rel)
        QT = K / "qdt"; QT.mkdir()
        shutil.copyfile(PPK.__file__, QT / "nb4_ppack.py")
        (QT / "nb4_ppack_tests.py").write_text(FILES["nb4_ppack_tests.py"], encoding="utf-8", newline="\n")
        (QT / "e26_qd_real_tree.py").write_text(FILES["e26_qd_real_tree.py"], encoding="utf-8", newline="\n")
        (QT / "e26.ipynb").write_text(FILES["e26.ipynb"], encoding="utf-8", newline="\n")
        r = sh("nb4_ppack_tests", [sys.executable, str(QT / "nb4_ppack_tests.py"), "--merlin-src", str(QSRC)], {"PYTHONIOENCODING": "utf-8"}, 1800, cwd=QT)
        m = re.search(r'"checks": (\d+),\s*"pass": (\d+),\s*"fail": (\d+)', r["_out"])
        QD["nb4_ppack_tests"] = {"rc": r["rc"], "sec": r["sec"], "checks": int(m.group(1)) if m else None, "pass": int(m.group(2)) if m else None,
                                 "fails": [l[:400] for l in r["_out"].splitlines() if l.startswith("FAIL")][:30], "tail": r["tail"][-1200:]}
        QD["nb4_ppack_tests"]["all_pass"] = r["rc"] == 0 and "ALL PASS" in r["_out"] and bool(m) and m.group(1) == m.group(2) and m.group(3) == "0"
        r = sh("e26_qd_real_tree", [sys.executable, str(QT / "e26_qd_real_tree.py"), str(QT / "e26.ipynb"), str(QT / "nb4_ppack.py")],
               {"K100QT_MERLIN_SRC": str(QSRC), "PYTHONIOENCODING": "utf-8"}, 900, cwd=QT)
        m = next((re.match(r"(\d+)/(\d+) PASS", l) for l in r["_out"].splitlines()[::-1] if re.match(r"\d+/\d+ PASS", l)), None)
        QD["unit_real_tree"] = {"rc": r["rc"], "summary": m.group(0) if m else None, "lines": [l[:300] for l in r["_out"].splitlines()
                                if l.startswith(("PASS", "FAIL"))], "tail": r["tail"][-1200:]}
        QD["unit_real_tree"]["all_pass"] = (r["rc"] == 0 and bool(m) and m.group(1) == m.group(2) and int(m.group(2)) >= KF["qd_unit_checks"]
                                            and not any(l.startswith("FAIL") for l in QD["unit_real_tree"]["lines"]))
    except BaseException as e:
        QD.update(error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
    REP["checks"]["qdefer"] = QD; save()
    log("Q-defer pre-setup", json.dumps({k: v for k, v in QD.items() if k != "tb"}, default=str)[:2500])
# ---------------------------------------------------------------- k100_setup on the real installed tree (worker ext, Q-defer, scales)
G["NB5_REPORT"] = {}   # kit stub: CELL 8's report dict (k100_setup records into it; nb5_write_report fails soft without BASE_OUT)
step("k100_setup", lambda: G["k100_setup"]())
if KF["qdefer"]:
    try:
        KQD = G["K100_QD"]
        ap = KQD.get("apply") or {}
        _post = {r_: v["sha"] for r_, v in ((ap.get("after") or {}).get("files") or {}).items()}
        QD["setup"] = {"applied": KQD.get("applied"), "status": KQD.get("status"), "env": KQD.get("env"), "error": KQD.get("error"),
                       "apply_applied": ap.get("applied"), "before_state": (ap.get("before") or {}).get("state"),
                       "after_state": (ap.get("after") or {}).get("state"), "post_shas": _post, "py_compile": ap.get("py_compile"),
                       "record_in_setup": (R1.get("k100_setup") or {}).get("qdefer")}
        QD["setup"]["verify_tree"] = PPK.verify_ppack_patch(QTREE)["state"]
        QD["setup_ok"] = (KQD.get("applied") is True and KQD.get("status") == "APPLIED" and KQD.get("env") == {"NB4_PPDEFER": "1"}
                          and ap.get("applied") == "applied" and QD["setup"]["before_state"] == "not_applied"
                          and QD["setup"]["after_state"] == "applied" and _post == KF["qd_post"]
                          and (ap.get("py_compile") or {}).get("rc") == 0 and QD["setup"]["verify_tree"] == "applied")
        # the revert path on the real tree: revert -> Merlin's exact pre-shas, then the notebook's own apply again
        rv = PPK.revert_ppack_patch(QTREE, QPY)
        _pre = {r_: v["sha"] for r_, v in rv["after"]["files"].items()}
        QD["revert"] = {"reverted": rv.get("reverted"), "state": rv["after"]["state"], "pre_shas": _pre, "py_compile": rv.get("py_compile")}
        QD["revert"]["ok"] = rv.get("ok") is True and rv.get("reverted") == "reverted" and _pre == KF["qd_pre"]
        if "k100_qd_restore" in G:     # audit G on the REAL tree: snapshot + an injected failure on the 2nd apply write -> restored
            _wa, _calls = PPK._write_atomic, []

            def _wa2(f, data):
                _calls.append(Path(f).name)
                if len(_calls) == 2:
                    raise OSError("injected (kit): 2nd apply write")
                return _wa(f, data)
            PPK._write_atomic = _wa2
            try:
                G["k100_qdefer_apply"]()
            finally:
                PPK._write_atomic = _wa
            _g = G["K100_QD"]
            _rs = _g.get("restore") or {}
            _gs = {r_: v["sha"] for r_, v in PPK.verify_ppack_patch(QTREE)["files"].items()}
            QD["g_inject"] = {"writes": _calls, "status": _g.get("status"), "applied": _g.get("applied"), "env": _g.get("env"),
                              "error": _g.get("error"), "snapshot_taken": "restore" in _g and _g.get("status") != "APPLY_FAILED_NOT_TOUCHED",
                              "restore": {k: _rs.get(k) for k in ("ok", "files", "consistent", "states", "py_compile")}, "tree_shas": _gs}
            QD["g_inject"]["ok"] = (_calls[:2] == [Path(KF["qd_chain"][0]).name, Path(KF["qd_chain"][1]).name]
                                    and _g.get("status") == "APPLY_FAILED_RESTORED" and QD["g_inject"]["snapshot_taken"]
                                    and _gs == KF["qd_pre"] and _rs.get("ok") is True and _rs.get("consistent") is True
                                    and (_rs.get("files") or {}).get(KF["qd_chain"][0]) == "restored"
                                    and (_rs.get("py_compile") or {}).get("rc") == 0 and _g.get("env") == {} and _g.get("applied") is False)
        G["k100_qdefer_apply"]()
        ap2 = G["K100_QD"].get("apply") or {}
        _post2 = {r_: v["sha"] for r_, v in ((ap2.get("after") or {}).get("files") or {}).items()}
        QD["reapply"] = {"status": G["K100_QD"].get("status"), "applied": ap2.get("applied"), "post_shas": _post2, "env": G["K100_QD"].get("env")}
        QD["reapply"]["ok"] = (G["K100_QD"].get("status") == "APPLIED" and ap2.get("applied") == "applied" and _post2 == KF["qd_post"]
                               and G["K100_QD"].get("env") == {"NB4_PPDEFER": "1"} and PPK.verify_ppack_patch(QTREE)["state"] == "applied")
    except BaseException as e:
        QD.update(post_error=f"{type(e).__name__}: {e}"[:800], post_tb=traceback.format_exc()[-2000:])
    REP["checks"]["qdefer"] = QD; save()
    log("Q-defer setup / revert / reapply", json.dumps({k: QD.get(k) for k in ("setup_ok", "setup", "revert", "reapply", "post_error")}, default=str)[:3000])
''' + R_SHARED + R_C0 + R_MERGE_NS + r'''
if KF["qdefer"]:            # the rerun namespace's own K100_QD (CELL 8 runs k100_setup there too): the tree is patched -> already_applied
    try:
        GR["k100_qdefer_apply"]()
        _a3 = GR["K100_QD"].get("apply") or {}
        QD["rerun_ns"] = {"status": GR["K100_QD"].get("status"), "applied": _a3.get("applied"), "env": GR["K100_QD"].get("env"),
                          "after_state": (_a3.get("after") or {}).get("state")}
        QD["rerun_ns"]["ok"] = (GR["K100_QD"].get("status") == "APPLIED" and _a3.get("applied") == "already_applied"
                                and GR["K100_QD"].get("env") == {"NB4_PPDEFER": "1"} and QD["rerun_ns"]["after_state"] == "applied")
    except BaseException as e:
        QD["rerun_ns"] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600]}
    REP["checks"]["qdefer"] = QD; save()
# ---------------------------------------------------------------- scales sidecar: the tree copy, the dataset copy, the model-dir copy
eng = G.get("MERLIN_ENG") or {}
SC = {}
try:
    _tree = (eng.get("kv") or {}).get("MERLIN_TREE")
    for lab_, p in (("tree", Path(_tree) / "models/flash-next/scales/qsa_kv_scales_262k.json" if _tree else None),
                    ("dataset", REF / "qsa_kv_scales_262k.json" if REF is not None else None),
                    ("model_dir", G["CKPT"] / "qsa_kv_scales_262k.json")):
        SC[lab_] = {"path": str(p), "exists": bool(p) and p.is_file()}
        if SC[lab_]["exists"]:
            SC[lab_]["sha_ok"] = sha_file(p) == KF["scales_sha"]
    SC["notebook"] = {k: v for k, v in ((R1.get("k100_setup") or {}).get("scales") or {}).items() if k != "tb"}
    SC["ok"] = (SC["notebook"].get("ok") is True and SC["model_dir"].get("sha_ok") is True and SC["dataset"].get("sha_ok") is True
                and all(v.get("sha_ok") for k, v in SC.items() if k in ("tree", "dataset", "model_dir") and v.get("exists")))
except BaseException as e:
    SC.update(ok=False, error=f"{type(e).__name__}: {e}"[:600])
REP["checks"]["scales"] = SC; save()
log("scales", json.dumps(SC, default=str)[:1200])

# ---------------------------------------------------------------- every K3 rung of both profiles' boot plans (no server started)
os.environ["VLLM_TARGET_DEVICE"] = "cpu"     # kit only: help_check inherits it via the server env (klee100 kit v2 ALL OK)
CPU_ENV = {"VLLM_TARGET_DEVICE": "cpu", "CUDA_VISIBLE_DEVICES": ""}
''' + VENV_PY + r'''
''' + ARGPARSE + r'''
def plans_for(g, label):
    """The tries nb5_boot builds (pin_ladder with the K3 steps when P k100; E18 ladder = the first 3 in the rerun; the lever env;
    --max-num-seqs from K100_SEQS_TABLE) for the profile's (MTP, KV) groups; #60490 taken as verified (the kit applied it)."""
    out, P_, L5 = {}, g["P"], g["NB5L"]
    groups = []
    for sp in P_["nb5_phases"]:
        sp = list(sp) + [None] * (7 - len(sp))
        key = (sp[3], sp[5] or g["NB5_KV_DTYPE"])
        if not groups or groups[-1][0] != key:
            groups.append([key, []])
        groups[-1][1].append(sp[0])
    rc2_fp8_dtype = L5.argv_last(g["NB5_RC2_FP8_ARGV_TAIL"]).get("--kv-cache-dtype")
    for (mtp, kv), phs in groups:
        base_extra = list(g["NB5_EXTRA_ARGS"]) + ([] if mtp == "on" else list(g["NB5_MTP_OFF_ARGS"]))
        tries = (L5.pin_tries(g["NB5_KV_BYTES"][kv], None, steps=g["K100_PIN_STEPS_GIB"]) if P_.get("k100")
                 else L5.pin_tries(g["NB5_KV_BYTES"][kv], None)) if P_.get("pin_ladder") else [{"gib": 0.0, "kv_bytes": g["NB5_KV_BYTES"][kv]}]
        if P_.get("e18_ladder"):
            tries = tries[:3]
        lev0 = dict(g["NB5_LEVER_ENV"])
        for i, tr in enumerate(tries):
            name = f"kit_{label}_{kv}_{'mtp' if mtp == 'on' else 'nomtp'}_pin{int(round(tr['gib'] * 100))}"
            g["NB5_BOOT_KV"], g["NB5_BOOT_LEVERS"] = kv, dict(lev0)
            seqs = (g["K100_SEQS_TABLE"][round(float(tr["gib"]), 2)] if g["NB5_SEQS_RULE"] == "k100"
                    else L5.seqs_for(kv, tr["gib"]) if g["NB5_SEQS_RULE"] == "table" else None)
            g["NB5_BOOT_EXTRA"] = base_extra + (["--max-num-seqs", str(seqs)] if seqs else [])
            b = {"phases": phs, "kv": kv, "mtp": mtp, "pin_gib": tr["gib"], "kv_bytes": tr["kv_bytes"], "seqs": seqs, "levers": lev0}
            try:
                s = g["merlin_fn_server"](name, kv, 8100 + 10 * i + (0 if label == "commit" else 5), tr["kv_bytes"])
                al = L5.argv_last(s.argv)
                caps = json.loads(al.get("--compilation-config") or "{}").get("cudagraph_capture_sizes")
                spec = al.get("--speculative-config")
                wx = s.argv[len(s.argv) - 1 - s.argv[::-1].index("--worker-extension-cls") + 1] if "--worker-extension-cls" in s.argv else None
                scf = s.env.get("VLLM_QSA_KV_SCALES")
                chk = []
                if str(al.get("--kv-cache-memory-bytes")) != str(tr["kv_bytes"]): chk.append(f"kv bytes {al.get('--kv-cache-memory-bytes')}")
                if seqs is None or str(al.get("--max-num-seqs")) != str(seqs): chk.append(f"seqs {al.get('--max-num-seqs')} vs {seqs}")
                if caps != KF["captures"] or not L5.captures_cover(caps or [], seqs or 1): chk.append(f"captures {caps}")
                if any(s.env.get(k) != v for k, v in lev0.items()): chk.append(f"lever env {[s.env.get(k) for k in lev0]}")
                try:              # the argv value is a compact JSON string ('{"method":"mtp","num_speculative_tokens":3,...}'): parse it
                    _spj = json.loads(spec) if spec not in (None, "None") else None
                except Exception:
                    _spj = {"unparsed": str(spec)}
                if (mtp == "on") != (_spj is not None) or (mtp == "on" and not (_spj.get("method") == "mtp" and _spj.get("num_speculative_tokens") == 3)):
                    chk.append(f"spec {spec}")
                if kv == "fp8" and al.get("--kv-cache-dtype") != rc2_fp8_dtype: chk.append(f"kv dtype {al.get('--kv-cache-dtype')} vs RC2 {rc2_fp8_dtype}")
                if str(al.get("--max-model-len")) != "131072": chk.append(f"max-model-len {al.get('--max-model-len')}")
                if "--language-model-only" not in s.argv: chk.append("--language-model-only missing")
                if wx != "nb1_worker_ext.NB1WorkerExt": chk.append(f"worker extension {wx}")
                if s.env.get("VLLM_SERVER_DEV_MODE") != "1": chk.append(f"VLLM_SERVER_DEV_MODE {s.env.get('VLLM_SERVER_DEV_MODE')}")
                if KF["qdefer"]:        # Q-defer: NB4_PPDEFER=1 on every launch of a verified patch, never NB4_PPACK
                    if s.env.get("NB4_PPDEFER") != "1" or "NB4_PPACK" in s.env: chk.append(f"Q-defer env {s.env.get('NB4_PPDEFER')} / NB4_PPACK {s.env.get('NB4_PPACK')}")
                    if (g["K100_QD"].get("launch_env") or {}).get(name) != "1": chk.append(f"K100_QD launch_env {(g['K100_QD'].get('launch_env') or {}).get(name)}")
                if kv == "fp8" and (scf != str(g["CKPT"] / "qsa_kv_scales_262k.json") or not Path(scf).is_file() or sha_file(scf) != KF["scales_sha"]):
                    chk.append(f"VLLM_QSA_KV_SCALES {scf}")
                k6 = (g["K1"].get("k6_before_launch") or {}).get(name) or {}
                cli = (g["K1"].get("cli_flags") or {}).get(name) or {}
                if k6.get("ok") is not True: chk.append(f"K6 before launch {k6}")
                if cli.get("ok") is not True: chk.append(f"cli flags {cli}")
                b.update(ok=True, host_phase=g["HOST_PHASE"].get(name), help=g["HELP"].get(name), argv=s.argv, server_started=s.proc is not None,
                         argv_flags={f: al.get(f) for f in ("--kv-cache-dtype", "--kv-cache-memory-bytes", "--max-num-seqs",
                                                            "--speculative-config", "--max-model-len")}, captures=caps, worker_ext=wx,
                         env={k: s.env.get(k) for k in ("VLLM_QSA_KV_SCALES", "SCALES", "VLLM_SERVER_DEV_MODE", "VLLM_PP_LAYER_PARTITION",
                                                         "NB4_PPDEFER", "NB4_PPACK",
                                                         "PYTHONPATH", "VLLM_TARGET_DEVICE", *lev0)}, k6=k6, cli_flags=cli, argv_check=chk)
            except BaseException as e:
                b.update(ok=False, error=f"{type(e).__name__}: {e}"[:1500], host_phase=(g.get("HOST_PHASE") or {}).get(name),
                         help=(g.get("HELP") or {}).get(name), tb=traceback.format_exc()[-2500:])
            out[name] = b
            log("plan", name, json.dumps({k: v for k, v in b.items() if k not in ("argv", "tb", "help")}, default=str)[:1500])
    g["NB5_BOOT_EXTRA"], g["NB5_BOOT_LEVERS"] = [], {}
    return out
PLANS = {}
for label, g in (("commit", G), ("rerun", GR)):
    try:
        PLANS[label] = plans_for(g, label)
    except BaseException as e:
        PLANS[label] = {"_fatal": {"ok": False, "error": f"{type(e).__name__}: {e}"[:1500], "tb": traceback.format_exc()[-2500:]}}
    REP["checks"]["plans"] = PLANS; save()
try:
    R1["help_text_len"] = len(G["L"]._HELP.get("merlin") or "")
    R1["help_real"] = "--max-num-seqs" in (G["L"]._HELP.get("merlin") or "") and "Failed to infer device" not in (G["L"]._HELP.get("merlin") or "")
except Exception as e:
    R1["help_text_len"] = repr(e)
ALLP = [b for pl in PLANS.values() for b in pl.values()]
def _strip(argv):
    a = list(argv or [])
    if "--port" in a:
        i = a.index("--port"); a[i + 1] = "*"
    return a
PC_, PR_ = PLANS.get("commit") or {}, PLANS.get("rerun") or {}
RUNGS = {"commit": [(round(b["pin_gib"], 6), b["kv_bytes"], b["seqs"]) for b in PC_.values() if "pin_gib" in b],
         "rerun": [(round(b["pin_gib"], 6), b["kv_bytes"], b["seqs"]) for b in PR_.values() if "pin_gib" in b]}
WANT_RUNGS = [(round(g_, 6), b_, s_) for g_, b_, s_ in zip(KF["gibs"], KF["pins"], KF["seqs"])]
ARGV_EQ = [_strip(a.get("argv")) == _strip(b.get("argv")) for a, b in zip(PC_.values(), PR_.values())]
R6 = {}
try:
    argvs = {f"{lab}:{n}": b["argv"] for lab, pl in PLANS.items() for n, b in pl.items() if b.get("argv")}
    (K / "argvs.json").write_text(json.dumps(argvs))
    # never name="argparse": K/argparse.py on the probe's sys.path[0] shadowed the stdlib argparse for this and every later venv
    # probe (kit run v1: torch circular import here, JSONDecodeError in the MM probe); the klee100 kit's r6_argparse passed
    r = venv_py(ARGPARSE, [str(K / "argvs.json")], name="r6_argparse", extra_env=CPU_ENV)
    R6["parsed"] = r["json"] or {"error": r["tail"]}
    fields = {}
    for lab_, argv in argvs.items():
        lab, n = lab_.split(":", 1)
        b = PLANS[lab][n]
        x = (R6["parsed"] or {}).get(lab_) or {}
        bad = []
        if x.get("ok") is not True: bad.append("parse " + str(x.get("error"))[:160])
        if x.get("unknown"): bad.append(f"unknown flags {x.get('unknown')}")
        if x.get("max_num_seqs") != b["seqs"]: bad.append(f"max_num_seqs {x.get('max_num_seqs')}")
        if x.get("kv_cache_memory_bytes") != b["kv_bytes"]: bad.append(f"kv {x.get('kv_cache_memory_bytes')}")
        if x.get("kv_cache_dtype") != (b.get("argv_flags") or {}).get("--kv-cache-dtype"): bad.append(f"kv_cache_dtype {x.get('kv_cache_dtype')}")
        if (x.get("capture_sizes") or []) != KF["captures"]: bad.append(f"captures {x.get('capture_sizes')}")
        if x.get("language_model_only") is not True: bad.append(f"language_model_only {x.get('language_model_only')}")
        if x.get("worker_extension_cls") != "nb1_worker_ext.NB1WorkerExt": bad.append(f"worker_extension_cls {x.get('worker_extension_cls')}")
        if x.get("max_model_len") != 131072: bad.append(f"max_model_len {x.get('max_model_len')}")
        if x.get("enable_prefix_caching") is not True: bad.append(f"prefix caching {x.get('enable_prefix_caching')}")
        if x.get("model") != str(G["CKPT"]): bad.append(f"model {x.get('model')}")
        if (x.get("tensor_parallel_size"), x.get("pipeline_parallel_size"), x.get("enable_expert_parallel")) != (2, 2, True): bad.append("layout")
        spec = x.get("speculative_config")
        if b["mtp"] == "on":
            okm = (spec.get("method") == "mtp" and spec.get("num_speculative_tokens") == 3) if isinstance(spec, dict) else \
                  ("'num_speculative_tokens': 3" in str(spec) and "mtp" in str(spec))
            if not okm: bad.append(f"spec {str(spec)[:80]}")
        elif spec not in (None, "None"): bad.append(f"spec should be None: {str(spec)[:80]}")
        fields[lab_] = bad
    R6["field_check"] = fields
    R6["ok"] = bool(fields) and len(fields) == len(ALLP) and all(not v for v in fields.values())
except BaseException as e:
    R6.update(ok=False, error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
os.environ.pop("VLLM_TARGET_DEVICE", None)
log("argparse", json.dumps({k: (R6.get("parsed") or {}).get(k) for k in ("_env", "_platform", "import_error")}), json.dumps(R6.get("field_check"))[:2000])
REP["checks"]["argparse"] = R6; save()

# ---------------------------------------------------------------- MM: the klee100 kit's method-map probe on REAL Merlin objects
MM = {}
''' + MM_SEC + R_SUB + r'''
P6 = R1.get("patch_60490") or {}
MG = ((G.get("R") or {}).get("sections", {}).get("A0") or {}).get("merge") or {}
DG = (G.get("K1") or {}).get("digests") or {}
WX = (R1.get("k100_setup") or {}).get("worker_ext") or {}
QDV = ({"qdefer_lib_loaded_from_dataset_sha_ok": (QD.get("lib") or {}).get("sha_ok") is True,
        "qdefer_installed_tree_pristine_pre_shas": QD.get("pristine_pre_shas_ok") is True,
        "nb4_ppack_tests_all_pass_on_installed_tree_copy": (QD.get("nb4_ppack_tests") or {}).get("all_pass") is True,
        "e26_unit_qdefer_real_tree_section_all_pass": (QD.get("unit_real_tree") or {}).get("all_pass") is True,
        "k100_setup_qdefer_applied_post_shas_env": QD.get("setup_ok") is True,
        "qdefer_revert_exact_pre_shas": (QD.get("revert") or {}).get("ok") is True,
        "qdefer_reapplied_after_revert": (QD.get("reapply") or {}).get("ok") is True,
        **({"qdefer_G_snapshot_2nd_write_failure_restores_pre_shas_no_env": (QD.get("g_inject") or {}).get("ok") is True}
           if KF["qd_restore"] else {}),
        "qdefer_rerun_namespace_already_applied_env": (QD.get("rerun_ns") or {}).get("ok") is True,
        "qdefer_env_every_launch_both_profiles": bool(ALLP) and all((b.get("env") or {}).get("NB4_PPDEFER") == "1"
                                                                    and (b.get("env") or {}).get("NB4_PPACK") is None for b in ALLP)}
       if KF["qdefer"] else {})
sc, sr = SUB.get("commit") or {}, SUB.get("rerun") or {}
verdict({
    "payload_sha_ok_under_input": PAY["used"] is not None,
    "tf_ref_new_version_mounted": DSV.get("new_version_mounted") is True,
    "tf_ref_every_staged_file_sha_ok": DSV.get("ok") is True,
    "payload_loaded_from_dataset_files": PLD.get("ok") is True,
    "duck_cell_loaded": bool(CELLS_C.get(2, {}).get("ok")) and D3["manifest_files"] > 0,
    "sandbox_selftest_ok_incl_memory_limit": bool((D3.get("sandbox_selftest") or {}).get("ok")),
    "cells_before_6_7_ran_commit_and_rerun": N_PRE == 5 and all(CELLS_C.get(i, {}).get("ok") for i in range(N_PRE))
        and all(CELLS_R.get(i, {}).get("ok") for i in range(N_PRE)),
    "profile_commit": G.get("PROFILE") == KF["commit_profile"] and G.get("RERUN") is False,
    "profile_rerun_RERUN_PROD_source_test_all_tasks": GR.get("PROFILE") == "RERUN_PROD" and GR.get("RERUN") is True
        and sr.get("source") == "test" and sr.get("task_ids_eq_test") is True,
    "inputs_found": bool(((R1.get("inputs") or {}).get("inputs") or {}).get("merlin_inputs_ok")),
    "klee100_merge_overlay_eq_v2_model_dir_k6": MG.get("ok") is True and (MG.get("overlay") or {}).get("ok") is True
        and all(((MG.get("overlay") or {}).get("equal_v2_artifact") or {"x": False}).values()) and (MG.get("k6") or {}).get("ok") is True,
    "k7_k3_digests_all_ok": bool((R1.get("sha256") or {}).get("sha_ok")) and DG.get("ok") is True and DG.get("n_checked") == DG.get("n_planned") > 0,
    "prompt_hashes_ok_klee100_tokenizer": bool(((R1.get("prompts") or {}).get("prompt_render") or {}).get("hashes_ok")) and G.get("MERLIN_PROMPTS") is True,
    "merlin_install_ok": bool((R1.get("install") or {}).get("ok")),
    "patch_60490_applied_verified": bool(P6.get("ok")) and P6.get("sha_after") == G.get("P60490_POST"),
    "worker_ext_installed_compiled_resolves_in_tree": WX.get("ok") is True,
    "scales_sidecar_sha_ok_tree_dataset_model_dir": SC.get("ok") is True,
    "commit_rungs_equal_facts": RUNGS["commit"] == WANT_RUNGS,
    "rerun_rungs_e18_first_3": RUNGS["rerun"] == WANT_RUNGS[:3],
    "plans_every_rung_both_profiles_ok": bool(PC_) and bool(PR_) and all(b.get("ok") for b in ALLP),
    "host_phase_ok_all_plans": bool(ALLP) and all(b.get("host_phase") == "ok" for b in ALLP),
    "help_cpu_every_flag_present_all_plans": bool(ALLP) and all((b.get("help") or {}).get("ran") and not (b.get("help") or {}).get("missing")
                                                                for b in ALLP),
    "argv_env_pin_seqs_caps_fp8_mtp_lmo_ext_scales_devmode_levers_all_plans": bool(ALLP) and all(b.get("argv") and not b.get("argv_check") for b in ALLP),
    "vllm_argparse_fields_all_plans": bool(R6.get("ok")),
    "no_server_started": bool(ALLP) and not any(b.get("server_started") for b in ALLP),
    "method_map_probe_real_merlin_objects_pass": MM.get("ok") is True, "method_map_src_anchors_ok": MM.get("src_ok") is True,
    "submission_commit_strict_full_schema_identity": sc.get("strict_full_schema") is True and sc.get("identity") is True
        and sc.get("profile") == KF["commit_profile"],
    "submission_rerun_strict_full_schema_identity": sr.get("strict_full_schema") is True and sr.get("identity") is True
        and sr.get("profile") == "RERUN_PROD",
    "cell0_same_test_file_and_fallback_as_e18_r2_real_mount": all(C0.get(f"{a}_{m}", {}).get("test_sha") for a in ("e18", "nb6")
        for m in ("rerun", "commit")) and all(C0[f"e18_{m}"][k] == C0[f"nb6_{m}"][k] for m in ("rerun", "commit")
        for k in ("arc_root", "rerun", "test_sha", "n_tasks", "sub_sha", "identity")) and C0["nb6_rerun"]["rerun"] is True
        and C0["nb6_commit"]["rerun"] is False and C0["nb6_rerun"]["identity"] is True,
    "rerun_cells_read_test_file_only_at_COMPD_never_eval_solutions": bool(RERUN_OPENS["test_challenges"])
        and all(Path(p).resolve() == (COMPD / "arc-agi_test_challenges.json").resolve() for p in RERUN_OPENS["test_challenges"])
        and not RERUN_OPENS["evaluation_solutions"],
    "k100_setup_ran_on_real_tree": (R1.get("k100_setup") or {}).get("ok") is True,
    **QDV,
}, recorded={"payload_used": PAY.get("used"), "qdefer": {k: v for k, v in QD.items() if k not in ("tb", "post_tb")}, "prereq": DSV.get("prereq"), "rungs": RUNGS, "want_rungs": WANT_RUNGS,
             "rerun_argv_equal_commit_first_rungs": ARGV_EQ,
             "plans": {lab: {n: {k: b.get(k) for k in ("pin_gib", "kv_bytes", "seqs", "argv_flags", "captures", "worker_ext", "env",
                                                        "host_phase", "cli_flags", "argv_check", "error")} for n, b in pl.items()} for lab, pl in PLANS.items()},
             "patch_60490": {k: P6.get(k) for k in ("applied", "sha_after", "marker", "py_compile", "vllm_origin", "error")},
             "help_text_len": R1.get("help_text_len"), "help_real_on_cpu": R1.get("help_real"),
             "argparse_env": (R6.get("parsed") or {}).get("_env"), "argparse_platform": (R6.get("parsed") or {}).get("_platform"),
             "method_map": {"check": MM.get("check"), "how": (MM.get("probe") or {}).get("how"), "libcuda": MM.get("libcuda")},
             "digests": {k: DG.get(k) for k in ("n_planned", "n_checked", "gb", "sec", "gb_per_s", "threads")},
             "budget_sec": {"commit": sc.get("budget_sec"), "rerun": sr.get("budget_sec")}})
'''

# ---------------------------------------------------------------------------------------------------------- kit table
KF = {"nb_sha": NB_SHA, "nb_name": NB_NAME, "nb_bytes": len(NB_RAW), "commit_profile": COMMIT_PROFILE, "phase": PHASE,
      "payname": PAYNAME, "paysha": PAYSHA, "entries": FACTS["payload"]["entries"], "stage_shas": STAGE_SHAS, "old_shas": OLD_SHAS,
      "new_files": NEW_FILES, "scales_sha": SCALES_SHA, "lab_sha": LAB_SHA, "nb5lib_sha": NB5LIB_SHA, "pins": PINS, "gibs": GIBS,
      "seqs": SEQS, "captures": CAPS, "window_sec": WINDOW, "budget_min": BUDGET_MIN, "k3_name": K3_NB.name, "k3_line_sha": K3_LINE_SHA,
      "k3_blobs": K3_BLOBS, "e18_cell1_sha": E18_CELL1_SHA, "push_literal": FACTS["push"], "push_code_file": PUSH_META["code_file"],
      "nb5_report_sha": NB5_REPORT_SHA}
REPORT_FILES = {n: build_text(n) for n in ("k100q_report.py", "k100q_report_tests.py", "nb5_fp8_reference.json", "nb5_lib.py")}
if FACTS.get("ref27"):                     # the 30-task build: the 27B / ARC Prize reference JSON next to the report lib (sha = facts)
    _r27 = Path(FACTS["ref27"]["file"])
    assert sha(_r27.read_bytes()) == FACTS["ref27"]["sha256"], "ref27 file != facts"
    REPORT_FILES[_r27.name] = lf_text(_r27)
assert sha(REPORT_FILES["k100q_report.py"]) == FACTS["embedded"]["k100q_report.py"], "k100q_report.py != the one embedded in the notebook"
assert sha(REPORT_FILES["nb5_fp8_reference.json"]) == FACTS["embedded"]["nb5_fp8_reference.json"]
for _n, _h in DS_PINS.items():          # pinned dataset files: the Build copy == the pin == the staged copy (taken from the mount)
    assert sha((HERE / _n).read_bytes()) == _h == STAGE_SHAS.get(_n), (_n, "Build / pin / staged differ")
KF["report_shas"] = {n: sha(v) for n, v in REPORT_FILES.items()}
REPORT_FILES = {n: v for n, v in REPORT_FILES.items() if n not in DS_PINS}     # those come from the mount (copy_pinned)
KF.update(ds_pins=DS_PINS, qd_skip_line=QD_SKIP_LINE, ppk_sha=PPK_SHA, qd_pre=QD_PRE, qd_post=QD_POST, qd_unit_checks=QD_UNIT_CHECKS,
          qdefer=QDEFER, deadman=DEADMAN_SEC is not None, deadman_sec=DEADMAN_SEC, qd_restore="def k100_qd_restore(" in CODE[4],
          qd_chain=list(ast.literal_eval(re.search(r"^K100_QD_CHAIN = (\(.*\))$", CODE[4], re.M).group(1))) if "K100_QD_CHAIN = " in CODE[4] else [])
if KF["qd_restore"]:
    assert KF["qd_chain"] == [PPKB.PP_REL, PPKB.MR_REL, PPKB.GW_REL], KF["qd_chain"]
KF["report_tests_min"] = len(re.findall(r"^    def test_", REPORT_FILES["k100q_report_tests.py"], re.M))
UNIT_FILES = {"e26.ipynb": NB_TXT, KF["nb_name"]: NB_TXT, "e26_k100q_unit_tests.py": build_text("e26_k100q_unit_tests.py"),
              "e26_k100q_build_facts.json": FACTS_TXT, "Q27-XHIGH-E25_NB6_SUBMIT.ipynb": lf_text(NB6_NB), "e18r2_cells01.ipynb": E18_REDUCED,
              "arc_eval_15tasks_solutions.json": SOL15, "k3only/k100_emb_line.tmpl": K3_TEMPLATE, "push/kernel-metadata.json": PUSH_META_TXT,
              **REPORT_FILES, **{"nb5out/" + k: v for k, v in nb5_out_files(["Q_FP8_K3"]).items()}}
REPORT_KIT_FILES = {"e26.ipynb": NB_TXT, "arc_eval_15tasks_solutions.json": SOL15, **REPORT_FILES,
                    **{"nb5out/" + k: v for k, v in nb5_out_files(["Q_FP8_K3", "Q_BF16_K3"]).items()}}
E2E_FILES = {"e26.ipynb": NB_TXT, "e26_e2e.py": E2E_DRV, "nb5_fake_server.py": K5["src"]("nb5_fake_server.py"), **E18_FILES}
REAL_FILES = {"e26.ipynb": NB_TXT, "e18_cell0.py": E18_FILES["e18_cell0.py"]}
if QDEFER:                 # nb4_ppack.py itself comes from the dataset mount (the notebook's own sha-pinned load, PPK.__file__)
    REAL_FILES.update({"nb4_ppack_tests.py": build_text("nb4_ppack_tests.py"), "e26_qd_real_tree.py": QD_REAL_SCRIPT})
KITS = {
    "units": {"body": UNITS, "title": "Q27 E26 K100Q Kit Units", "models": [], "files": UNIT_FILES},
    "units-report": {"body": UNITS_REPORT, "title": "Q27 E26 K100Q Kit Units Report", "models": [], "files": REPORT_KIT_FILES},
    "e2e-rerun": {"body": E2E.replace("__MODE__", "rerun,rerun-retry"), "title": "Q27 E26 K100Q Kit E2E Rerun", "models": MODELS, "files": E2E_FILES},
    "e2e-commit": {"body": E2E.replace("__MODE__", "commit"), "title": "Q27 E26 K100Q Kit E2E Commit", "models": MODELS, "files": E2E_FILES},
    "real": {"body": REAL, "title": "Q27 E26 K100Q Kit Real", "models": MODELS, "files": REAL_FILES},
}


def build(name):
    spec = KITS[name]
    files = spec["files"]
    assert all("\r" not in v for v in files.values()), name
    blob = base64.b85encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
    shas = {n: sha(v) for n, v in files.items()}
    code = HEAD + spec["body"]
    for k, v in (("__SHAS__", json.dumps(shas)), ("__NBSHA__", NB_SHA), ("__NAME__", name), ("__COMMIT_MAX__", str(int(BUDGET_MIN * 60 + 1800))),
                 ("__KF__", json.dumps(KF)), ("__BLOB__", blob)):
        code = code.replace(k, v)
    for ph in ("__SHAS__", "__NBSHA__", "__KF__", "__BLOB__", "__MODE__"):
        assert ph not in code, (name, ph)
    compile(code, f"<kit {name}>", "exec")
    slug = f"{SLUG}-{name}"
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1).encode("utf-8")
    assert len(data) < 400_000, (name, len(data))
    assert b"\r" not in data.replace(b"\\r", b"")
    out = ROOT / f"{slug}__{OWNER}"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{slug}.ipynb").write_bytes(data)
    meta = {"id": f"{OWNER}/{slug}", "title": spec["title"], "code_file": f"{slug}.ipynb", "language": "python",
            "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False,
            "keywords": [], "dataset_sources": list(DATASETS), "kernel_sources": [], "competition_sources": list(COMPS),
            "model_sources": list(spec["models"]), "docker_image": IMAGE, "machine_shape": "None"}
    (out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    print("kit", out.name, "|", len(data), "bytes | sha256", sha(data), flush=True)
    return out, sha(data), len(data)


if __name__ == "__main__":
    print("notebook under test", NB_NAME, "sha256", NB_SHA, len(NB_RAW), "bytes | profile", COMMIT_PROFILE, "| phase", PHASE,
          "| window", WINDOW, "| budget", BUDGET_MIN, "min | rungs", list(zip(PINS, SEQS)), flush=True)
    for n in KITS:
        build(n)
