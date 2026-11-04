"""E33 CHAIN Kaggle CPU kits (2026-11-02; DESIGN_E33_ATTEMPT_CHAIN_2026-11-02.md section 6; the kit-must-mirror rule): independent CPU
kernels on foysalemonshanto, the notebook's pinned Py3.12 image, no GPU, no internet, private. Each kernel is fail-soft, writes its
JSON report and prints exactly one line `@@KIT <name> ALL OK@@` or `@@KIT <name> NOT OK: <items>@@`. NOTHING is pushed here
(the user pushes kits), no Kaggle API, no git.

  units   the E33 module tests on Py3.12 (the tests ship in the kit; the modules under test are STAGED BY SHA from the mounted
          E33 payload dataset, = the bytes the notebook loads): S's e33_sched_tests.py (scheduler diff, answer rule, switches
          off parity, negative control; on the REAL arc2_duck.py of the mounted Duck payload), P's Poetiq tests (the sandbox runs
          for real here: root + libseccomp on the Kaggle image), the kit's own e33_kit_switch_tests.py (the notebook's CELL 1
          switch literals + parser: a bad value turns only that item off) and the E29 suites that must still pass on the same
          payload (e29_sched_tests.py = the E29 post file E33 patches; e29d_direct_tests.py = the Direct solver e29_direct.py
          stage E uses), + the Python version = 3.12
  real    E29's real kit (build_e29_cpu_kit.py, ALL OK on Kaggle for E29_DFLASH_D3S) RETARGETED to the E33 notebook through the
          same E26 / E28 / E29 kit chain (cells 0-4 in the commit + rerun namespace on the real mounts, Merlin install, the
          klee100 inputs, every boot try through merlin_fn_server + vLLM's argparse, the seam, CELL 0 vs E18 R2) + the E33
          section: the E33 scheduler diff applied in both namespaces / verified / reverted to the E29 post / re-applied; the
          notebook's own E33 setup (Poetiq + Direct hooks on E12A) and the hooks called on a fake token-id server with the REAL
          klee100 tokenizer; Poetiq's poetiq_sandbox_selftest under the Kaggle CPU limits; the notebook's own submission writer
          (e12_write_submission, cut verbatim from the readable cell) in commit and rerun mode on the patched arc2_duck with E33
          records; a step-coverage table
  chain   e33_kit_chain_e2e.py: a scripted run of 6 tasks through A -> B -> C -> D -> E and attempt 2 on the REAL arc2_duck.py
          (Duck payload) + the REAL E29 + E33 scheduler diffs + the REAL e33_poetiq.py + the REAL e29_direct.py against an
          in-process fake token-id server (mocked GPU); per-stage counts, the stop at the first train pass, attempt-2 order and
          the answer slots asserted
Rules: no Kaggle API calls, no git, no local whole-kit runs (user rule: long tests on Kaggle CPU kits only). Locally: the quick
units (--local-units) and ONE layout simulation (--sim): each kit's own code with KIT_INPUT = a simulated /kaggle/input tree and
E33KIT_LAYOUT_ONLY=1 (it stops after finding / staging every input and writes a layout verdict).
Run: python build_e33_cpu_kit.py [--local-units] [--sim] [--only units,real,chain]
     -> D:/ARC AGI 2/_scratch/push_e25/q27-e33-kit-<name>__foysalemonshanto/
"""
import ast
import base64
import hashlib
import json
import lzma
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
E33 = HERE.parent
NBDIR = E33 / "Experiment Notebooks"
E29B = Path(r"D:/ARC AGI 2/experiments/Experiment 29 - Final production K100 (planned)/Build")
E28B = Path(r"D:/ARC AGI 2/experiments/Experiment 28 - Flash-Next EVAL120 6h W4 vs klee100 (planned)/Experiment Notebooks/Build")
E25B = Path(r"D:/ARC AGI 2/experiments/Experiment 25 - Flash-Next Throughput 1500 (planned)/Experiment Notebooks/Build")
E15_PAYLOAD = Path(r"D:/ARC AGI 2/experiments/Experiment 15 - Duck 2x35k independent attempts (planned)/Experiment Notebooks/Build/payload")
EVAL_JSON = Path(r"D:/ARC AGI 2/experiments/Experiment 10 - Q27-XHIGH Poetiq Code-Verify attempt_2 (planned)/Experiment Review"
                 r"/E10_ORDER2_EVAL120_RESULTS_FOR_GPT/data")
ROOT = Path(r"D:/ARC AGI 2/_scratch/push_e25")
DS_SLUG = "q27-e33-chain-payload"
DS_DIR = ROOT / f"ds_{DS_SLUG}"
SIM = Path(r"D:/ARC AGI 2/_scratch/kits_e33_local")
OWNER = "foysalemonshanto"
PREFIX = "q27-e33-kit-"
NB_NAME = "Q27-XHIGH-E33_CHAIN.ipynb"
DUCK_PAYLOAD = ("q27x_e15b_duck_9e4b89285136.bin", "9e4b89285136616d523028ac042dcca6afe82175081a1481d584c998ab3ee566")


def sha(b):
    return hashlib.sha256(b if isinstance(b, bytes) else b.encode("utf-8")).hexdigest()


def once(text, old, new, what, count=1):
    assert text.count(old) == count, (what, text.count(old))
    return text.replace(old, new)


def lf(p):
    """A text file as LF (the E29 kit rule: no CR bytes)."""
    b = Path(p).read_bytes()
    assert b"\r" not in b, f"CR bytes in {p}"
    return b.decode("utf-8")


def lf_copy_text(p):
    """Files other builders wrote with write_text on Windows (CRLF): the LF text (same content); used for JSON facts only."""
    return Path(p).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")


# ====================================================================== the kit HEAD (units / chain): the E29 kit helpers
KIT_HEAD = r'''# E33 CHAIN CPU kit __NAME__ (pinned image, no GPU): see Build/build_e33_cpu_kit.py for the checks and the verdict rule
import os, sys, json, time, subprocess, shutil, traceback, hashlib, re, base64, lzma
from pathlib import Path
T0 = time.time()
KIT = "__NAME__"
INPUT_ROOT = os.environ.get("KIT_INPUT", "/kaggle/input")          # env overrides: the local layout simulation only
K = Path(os.environ.get("KIT_TMP", "/tmp/e33kit")); shutil.rmtree(K, ignore_errors=True); K.mkdir(parents=True)
W = Path(os.environ.get("KIT_WORK", "/kaggle/working")); W.mkdir(parents=True, exist_ok=True)
LAYOUT_ONLY = os.environ.get("E33KIT_LAYOUT_ONLY") == "1"          # the local layout simulation: inputs only, no test run
REP = {"what": f"E33 CHAIN CPU kit {KIT}", "notebook_sha256": "__NBSHA__", "python": sys.version, "checks": {}}
def save():
    (W / f"e33_kit_{KIT}_report.json").write_text(json.dumps(REP, indent=1, default=str))
def log(*a):
    print(f"[{(time.time() - T0) / 60:5.1f} min]", *a, flush=True)
FILES = json.loads(lzma.decompress(base64.b85decode("__BLOB__")).decode("utf-8"))
FILE_SHAS = json.loads(r"""__SHAS__""")
REP["files_ok"] = {n: hashlib.sha256(FILES[n].encode("utf-8")).hexdigest() == h for n, h in FILE_SHAS.items()}
assert all(REP["files_ok"].values()), REP["files_ok"]
KF = json.loads(r"""__KF__""")
def sh(name, args, env=None, timeout=3600, cwd=None):
    t = time.time()
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})}, cwd=str(cwd or K),
                           encoding="utf-8", errors="replace")
        out, rc = r.stdout + "\n--- stderr ---\n" + r.stderr, r.returncode
    except Exception as e:
        out, rc = f"{type(e).__name__}: {e}", -9
    (W / f"kit_{name}.log").write_text(out, encoding="utf-8")
    return {"rc": rc, "sec": round(time.time() - t), "tail": out[-1500:], "_out": out}
def verdict(ok, recorded=None):
    REP["verdict"] = {"checks": ok, "ALL_OK": bool(ok) and all(ok.values()), "recorded": recorded or {},
                      "minutes": round((time.time() - T0) / 60, 1), "layout_only": LAYOUT_ONLY}
    save()
    bad = [k for k, v in ok.items() if not v] or ([] if ok else ["no checks"])
    print("VERDICT", json.dumps(REP["verdict"], default=str)[:8000], flush=True)
    print(f"@@KIT {KIT} {'LAYOUT ' if LAYOUT_ONLY else ''}ALL OK@@" if not bad else f"@@KIT {KIT} NOT OK: {', '.join(bad)}@@", flush=True)
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()
def find_by_sha(name, sha_):
    """A file of a mounted dataset by its exact name (any depth under the input root: datasets/<owner>/<slug>/..., the legacy
    /kaggle/input/<slug>/...) and sha256; every candidate recorded."""
    seen, hit = [], None
    for f in sorted(Path(INPUT_ROOT).glob("**/" + name)):
        if not f.is_file():
            continue
        ok_ = sha_file(f) == sha_
        seen.append([str(f), ok_])
        if ok_ and hit is None:
            hit = f
    return hit, seen
STAGED = {}
def stage(dst_dir, items):
    """KF items {published name: [sha256, staged name]} from the mounted datasets into dst_dir (the bytes the notebook loads)."""
    for n_, (s_, to_) in items.items():
        hit, seen = find_by_sha(n_, s_)
        STAGED[n_] = {"path": str(hit) if hit else None, "seen": seen, "to": to_}
        if hit:
            (Path(dst_dir) / to_).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(hit, Path(dst_dir) / to_)
    REP["checks"]["staged"] = STAGED; save()
    return all(v["path"] for v in STAGED.values())
COMPD = Path(INPUT_ROOT) / "competitions/arc-prize-2026-arc-agi-2"
if not COMPD.is_dir() and Path(INPUT_ROOT, "arc-prize-2026-arc-agi-2").is_dir():
    COMPD = Path(INPUT_ROOT, "arc-prize-2026-arc-agi-2")
def write_files(dst):
    for n, v in FILES.items():
        (dst / n).parent.mkdir(parents=True, exist_ok=True)
        (dst / n).write_text(v, encoding="utf-8", newline="\n")
def ut_parse(out):
    """unittest's own summary (TextTestRunner, stderr): 'Ran N tests' + 'OK' / 'FAILED (failures=.., errors=.., skipped=..)';
    or the 'n/m PASS' line of a check-style file."""
    r = {"run": None, "status": None, "failures": 0, "errors": 0, "skipped": 0}
    m = list(re.finditer(r"^Ran (\d+) tests? in [\d.]+s", out, re.M))
    if m:
        r["run"] = int(m[-1].group(1))
        tail = out[m[-1].end():]
        s = re.search(r"^(OK|FAILED)(?: \(([^)]*)\))?\s*$", tail, re.M)
        if s:
            r["status"] = s.group(1)
            for kv in (s.group(2) or "").split(","):
                if "=" in kv:
                    k_, v_ = kv.strip().split("=", 1)
                    r[k_] = int(v_)
    else:
        p = [x for x in re.finditer(r"^(\d+)/(\d+) PASS\s*$", out, re.M)]
        if p:
            r.update(run=int(p[-1].group(2)), status="OK" if p[-1].group(1) == p[-1].group(2) else "FAILED",
                     failures=int(p[-1].group(2)) - int(p[-1].group(1)))
        r["skipped"] = len(re.findall(r"^SKIP ", out, re.M))
    r["fail_lines"] = [l[:400] for l in out.splitlines() if l.startswith(("FAIL:", "ERROR:", "FAIL ")) or l.endswith((" ... FAIL", " ... ERROR"))][:40]
    r["skip_lines"] = [l[:300] for l in out.splitlines() if " ... skipped" in l or l.startswith("SKIP ")][:40]
    return r
def run_suite(spec, B, subs):
    """One test file of KF["tests"]: {name, file, args, env (placeholders {B} {K} {PAY} {EVAL} {COMPD}), min, skips_allowed, timeout}."""
    T = {"file": spec["file"]}
    try:
        env = dict(spec.get("env") or {})
        for key, value in env.items():
            for placeholder, replacement in subs.items():
                value = value.replace("{" + placeholder + "}", str(replacement))
            env[key] = value
        env.setdefault("PYTHONIOENCODING", "utf-8")
        r = sh(spec["name"], [sys.executable, str(B / spec["file"])] + list(spec.get("args") or []), env, int(spec.get("timeout") or 3000), cwd=B)
        p = ut_parse(r["_out"])
        T.update(rc=r["rc"], sec=r["sec"], **p, tail=r["tail"][-2500:])
        if spec.get("expect_fail"):          # a negative control: the suite must FAIL (and run)
            T["all_pass"] = (r["rc"] != 0 and (p["run"] or 0) >= int(spec["min"]) and p["failures"] + p["errors"] >= int(spec["min_fail"]))
            T["skips_ok"] = True
        else:
            T["all_pass"] = (r["rc"] == 0 and p["status"] == "OK" and (p["run"] or 0) >= int(spec["min"]) and p["failures"] == 0
                             and p["errors"] == 0)
            T["skips_ok"] = p["skipped"] <= int(spec.get("skips_allowed") or 0)
    except BaseException as e:
        T.update(all_pass=False, skips_ok=False, fatal=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
    log(spec["name"], json.dumps({k_: T.get(k_) for k_ in ("rc", "sec", "run", "status", "failures", "errors", "skipped", "fail_lines", "skip_lines")})[:3000])
    return T
'''

# ====================================================================== units kit body
UNITS_BODY = r'''
U = {"python_312": sys.version_info[:2] == (3, 12)}
B = K / "build"; B.mkdir()
write_files(B)
U["staged_all"] = stage(B, KF["stage"])
_pay, _seen = find_by_sha(KF["duck_payload"][0], KF["duck_payload"][1])
U["duck_payload"] = {"path": str(_pay) if _pay else None, "seen": _seen}
U["eval_challenges"] = (COMPD / "arc-agi_evaluation_challenges.json").is_file()
REP["checks"]["units"] = U; save()
log("inputs", json.dumps(U, default=str)[:3000])
if LAYOUT_ONLY:
    verdict({"duck_payload_found_by_sha": _pay is not None, "e33_modules_staged_by_sha": U["staged_all"],
             "competition_eval_challenges_present": U["eval_challenges"],
             "every_test_file_and_its_inputs_in_place": all((B / s["file"]).is_file() for s in KF["tests"])
             and all((B / v[1]).is_file() for v in KF["stage"].values())}, recorded={"staged": STAGED})
    raise SystemExit(0)
for k_ in [k_ for k_ in os.environ if k_.startswith(("E28_", "E29_", "E33_", "E12_", "ARC3_", "NB5", "Q27X_", "E11_"))] + ["KAGGLE_IS_COMPETITION_RERUN"]:
    os.environ.pop(k_, None)
SUBS = {"B": str(B), "K": str(K), "PAY": str(_pay), "EVAL": str(COMPD / "arc-agi_evaluation_challenges.json"), "COMPD": str(COMPD)}
RES = {}
for spec in KF["tests"]:
    RES[spec["name"]] = run_suite(spec, B, SUBS)
    REP["checks"]["tests"] = RES; save()
ok = {"python_3_12_pinned_image": U["python_312"], "duck_payload_mounted_sha_9e4b8928": _pay is not None,
      "e33_modules_from_the_e33_dataset_by_sha": U["staged_all"], "competition_eval_challenges_present": U["eval_challenges"]}
for spec in KF["tests"]:
    ok[spec["name"] + "_all_pass"] = RES[spec["name"]].get("all_pass") is True
    ok[spec["name"] + "_skips_within_" + str(spec.get("skips_allowed") or 0)] = RES[spec["name"]].get("skips_ok") is True
verdict(ok, recorded={n: {k_: v.get(k_) for k_ in ("run", "status", "failures", "errors", "skipped", "fail_lines", "skip_lines")} for n, v in RES.items()})
'''

# ====================================================================== chain e2e kit body (same shape: one suite)
CHAIN_BODY = UNITS_BODY.replace('U = {"python_312"', 'U = {"chain": True, "python_312"', 1)


# ====================================================================== discovery of the other agents' outputs
def find_first(*cands):
    for c in cands:
        for p in sorted(HERE.glob(c)) if any(ch in c for ch in "*?[") else [HERE / c]:
            if p.is_file():
                return p
    return None


def discover():
    """What the other agents built (their files are read, never edited)."""
    d = {"sched_patch": find_first("e33_sched_patch.py"), "sched_tests": find_first("e33_sched_tests.py"),
         "sched_build": find_first("e33_sched_build.py"), "poetiq": find_first("e33_poetiq.py"),
         "poetiq_tests": find_first("e33_poetiq_tests.py", "e33_poetiq_test*.py", "test_e33_poetiq*.py"),
         "builder": find_first("build_e33.py"), "notebook": NBDIR / NB_NAME if (NBDIR / NB_NAME).is_file() else None,
         "smoke": NBDIR / NB_NAME.replace(".ipynb", "_SMOKE.ipynb") if (NBDIR / NB_NAME.replace(".ipynb", "_SMOKE.ipynb")).is_file() else None,
         "facts": find_first("e33*build_facts.json", "*e33*facts*.json"), "dataset": DS_DIR if DS_DIR.is_dir() else None}
    d["dataset_files"] = ({f.name: sha(f.read_bytes()) for f in sorted(DS_DIR.iterdir()) if f.is_file()
                           and f.name not in ("SHA256SUMS", "dataset-metadata.json")} if d["dataset"] else {})
    return d


# ====================================================================== the E29 kit chain (real kit), retargeted
_K29 = []


def k29():
    """The E29 kit builder's namespace (exec of its source, main() never called; one edit: its LF facts cache lives under this
    Build folder, so nothing is written into the E29 folder)."""
    if _K29:
        return _K29[0]
    p = E29B / "build_e29_cpu_kit.py"
    src = lf(p)
    src = once(src, 'LOCAL = HERE / "_kit_local"                # local unit runs + the LF copies of the variant facts (see facts_lf)\n',
               f'LOCAL = Path(r"{HERE / "_kit_local_e29chain"}")   # E33 kit: the LF facts cache under the E33 Build folder\n', "local dir")
    ns = {"__file__": str(p), "__name__": "e33kit_e29"}
    exec(compile(src, str(p), "exec"), ns)
    ns["_src_sha"] = sha(lf(p))
    _K29.append(ns)                      # the namespace itself (its functions read their globals from it)
    return ns


def kit_notebook(code):
    nb = {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.splitlines(True)}],
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    data = json.dumps(nb, indent=1).encode("utf-8")
    assert b"\r" not in data.replace(b"\\r", b"")
    return data


def chain_kit_code(ns, name, body, files, kf, tag="e33"):
    """= build_e29_cpu_kit.build_kit's code (the E26 / E28 HEAD + body), E33 names (tag "e29" reproduces the E29 kit)."""
    NS = ns["NS"]
    assert all("\r" not in v for v in files.values()), name
    blob = base64.b85encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
    shas = {n: sha(v) for n, v in files.items()}
    code = NS["HEAD"] + body
    code = once(code, '"/tmp/e28k100kit"', f'"/tmp/{tag}kit"', "tmp")
    code = once(code, 'f"e28k100_kit_{KIT}_report.json"', f'f"{tag}_kit_{{KIT}}_report.json"', "report")
    code = code.replace("E28 K100 CPU kit", "E29 FINAL CPU kit" if tag == "e29" else "E33 CHAIN CPU kit (E29 kit chain)")
    if "e26.ipynb" not in files:
        code = once(code, 'assert hashlib.sha256(FILES["e26.ipynb"].encode("utf-8")).hexdigest() ==',
                    'assert "e26.ipynb" not in FILES or hashlib.sha256(FILES["e26.ipynb"].encode("utf-8")).hexdigest() ==', "nb sha optional")
    for k, v in (("__SHAS__", json.dumps(shas)), ("__NBSHA__", ns["NB_SHA"]), ("__NAME__", name), ("__KF__", json.dumps(kf)), ("__BLOB__", blob)):
        code = code.replace(k, v)
    for ph in ("__SHAS__", "__NBSHA__", "__KF__", "__BLOB__", "__MODE__", "__STEPS__"):
        assert ph not in code, (name, ph)
    compile(code, f"<kit {name}>", "exec")
    return code


LOCAL = HERE / "_kit_local_e33"                 # the kit builder's own LF facts / local unit runs (kit-owned)
DRAFT_DIR = Path(r"D:/ARC AGI 2/_scratch/e33_nb")


def readable(code_cells):
    """The notebook's readable code cells (= the build's code export): build_e29_cpu_kit's E28_READABLE (every packed block)."""
    g = {"ast": ast}
    exec(compile(k29()["E29_READABLE_SRC"], "<E29 readable>", "exec"), g)
    return g["E28_READABLE"](code_cells)


def e33_facts(draft=False):
    """The E33 notebook's facts in the E28 / E29 facts format the kit chain reads (E26 -> E28 -> E29 kit builders): E29_DFLASH_D3S's
    facts (the engine E33 keeps verbatim) with the notebook / export / push / metadata / pack records of the E33 build
    (Build/e33_chain_build_facts.json, agent I) + an "e33" key. draft=True: agent I's --draft notebook (local unit runs only;
    a local export + push copy under _kit_local_e33/draft)."""
    base = json.loads(lf_copy_text(E29B / "e29_dflash_d3s_build_facts.json"))
    LOCAL.mkdir(parents=True, exist_ok=True)
    if draft:
        nbp = DRAFT_DIR / ("DRAFT_" + NB_NAME)
        raw = nbp.read_bytes()
        code = ["".join(c["source"]) for c in json.loads(raw)["cells"] if c["cell_type"] == "code"]
        dd = LOCAL / "draft"
        (dd / "push").mkdir(parents=True, exist_ok=True)
        km = dict(base["kernel_metadata"])
        km["dataset_sources"] = list(km["dataset_sources"]) + [f"{OWNER}/{DS_SLUG}"]
        (dd / "push" / km["code_file"]).write_bytes(raw)
        (dd / "push" / "kernel-metadata.json").write_bytes((json.dumps(km, indent=2) + "\n").encode("utf-8"))
        (dd / "export.py").write_bytes("\n\n".join(readable(code)).encode("utf-8"))
        nb = json.loads(raw)
        rec = {"notebook": str(nbp), "notebook_sha256": sha(raw), "bytes": len(raw), "cells": len(nb["cells"]), "json_format": base["json_format"],
               "e21_cell7_packed": base["e21_cell7_packed"], "e21_cell6_packed": base["e21_cell6_packed"],
               "e29_engine_packed": base["e29"]["e29_engine_packed"], "export": str(dd / "export.py"),
               "push": {"push": str(dd / "push"), "kernel_metadata": km}, "payload_pins": None, "draft": True}
    else:
        rec = json.loads(lf_copy_text(HERE / "e33_chain_build_facts.json"))
    f = dict(base)
    f.update(notebook=rec["notebook"], notebook_sha256=rec["notebook_sha256"], bytes=rec["bytes"], cells=rec["cells"],
             json_format=rec["json_format"], e21_cell7_packed=rec["e21_cell7_packed"], e21_cell6_packed=rec["e21_cell6_packed"],
             export=rec["export"], push=rec["push"]["push"], kernel_metadata=rec["push"]["kernel_metadata"])
    sm = dict(base["e29"]["smoke"])                    # the smoke copy of THIS notebook (agent I's record; draft: computed)
    if draft:
        a_, b_ = (json.dumps(x + "\n")[1:-1].encode("utf-8") for x in sm["diff"][1:])
        sm.update(sha256=sha(raw.replace(a_, b_)), bytes=len(raw), slug=None)
    else:
        sm.update(sha256=rec["smoke_sha256"], bytes=rec["smoke_bytes"], slug=rec["push_smoke"]["slug"], diff=rec["smoke_diff"])
    e29 = dict(base["e29"], e29_engine_packed=rec["e29_engine_packed"], direct=None, smoke=sm)
    f["e29"] = e29
    f["e33"] = {k: rec.get(k) for k in ("payload_pins", "smoke_sha256", "smoke_diff", "switch_line", "chain_default", "lib_version",
                                        "edits", "variant", "agent_files", "dataset", "push_smoke", "push_submit", "draft",
                                        "builder_sha256", "nbx_src_sha256")}
    f["e33"]["e29_dflash_d3s_notebook_sha256"] = base["notebook_sha256"]
    out = LOCAL / ("e33_chain_kit_facts_draft.json" if draft else "e33_chain_kit_facts.json")
    data = (json.dumps(f, indent=1, default=str) + "\n").encode("utf-8")
    if not out.is_file() or out.read_bytes() != data:
        out.write_bytes(data)
    return out, f


def load_chain_e33(fpath, slug):
    """= build_e29_cpu_kit.load_chain for the E33 notebook: the E28 kit builder (-> the E26 kit builder) exec'd with the E29 kit's
    exact-once edits, the facts path = the E33 facts (LF), the modules == the notebook's packed literal."""
    K = k29()
    facts = json.loads(lf(fpath))
    mods_nb = K["packed_modules"](lf(facts["notebook"]))

    def mod_text(n):
        if n == "e29_nbdf.py":
            return mods_nb[n]
        t = lf((E28B / n) if n.startswith("e28") else (E29B / n))
        assert t == mods_nb[n], (n, "Build file != the notebook's packed module")
        return t
    s = K["E28_KIT_SRC"]
    s = once(s, 'E28_FACTS = HERE / "e28k100_build_facts.json"\n', f'E28_FACTS = Path(r"{fpath}")   # E33 kit: the E33 facts (LF)\n', "facts")
    s = once(s, 'SLUG, OWNER = "q27-e28-k100-kit", "esfersami50"\n', f'SLUG, OWNER = "{slug}", "{OWNER}"   # E33 kit\n', "slug")
    a = s.index("_E28_READABLE_SRC = r'''")
    b = s.index("'''\n", a + 30) + 4
    s = s[:a] + "_E28_READABLE_SRC = " + repr(K["E29_READABLE_SRC"]) + "   # E29 kit: both packed literals\n" + s[b:]
    s = once(s, 'assert {n: sha((HERE / n).read_bytes()) for n in MODS} == MODS, "the Build modules != the ones packed in the notebook (rebuild)"\n',
             'assert {n: sha(E29_MOD_TEXT(n)) for n in MODS} == MODS, "the Build modules != the ones packed in the notebook (rebuild)"   # E29 kit\n',
             "modules check")
    ns = {"__file__": str(K["E28_KIT"]), "__name__": "e33kit_e28", "E29_MOD_TEXT": mod_text}
    exec(compile(s, str(K["E28_KIT"]), "exec"), ns)
    assert ns["NB_SHA"] == facts["notebook_sha256"]
    return ns, facts, mods_nb, mod_text


def libt_e33(libt):
    """e29_kit_lib_tests.py retargeted to the E33 notebook (exact-once, marked "# E33 kit"): its rev-2 block reads the E29 DIRECT
    nbx file from the facts (E33 has none: W4 loads e33_nbx.py) -> the E33 notebook's own nbx pin (CELL 1 E33D_FILES). Every
    check runs; the checks of the E29 parts E33 replaced by design (W1-W4) are expected to fail and are listed in KF."""
    t = once(libt, '    NBX_P = payload_file(*FACTS["e29"]["direct"]["files"]["nbx"])\n',
             '    _E33F = json.loads(next(l_ for l_ in C1.splitlines() if l_.startswith("E33D_FILES = "))[len("E33D_FILES = "):].split("   #")[0])   # E33 kit\n'
             '    NBX_P = payload_file(*_E33F["nbx"])   # E33 kit: the E33 nbx (E29 LiveLog + e29d_tok verbatim + the chain functions)\n',
             "nbx pin")
    t = once(t, '"""E29 kit unit tests (quick;', '"""E33 kit: e29_kit_lib_tests.py RETARGETED to the E33 notebook (build_e33_cpu_kit.libt_e33). E29 kit unit tests (quick;',
             "doc")
    t = t.replace('C1.splitlines()[1].startswith("E29_QSA =', 'next(l for l in C1.splitlines() if l.startswith("E29_QSA = ")).startswith("E29_QSA =')
    start = t.index('    def _r29():')
    end = t.index('    check("R29', start)
    smoke_test = t[start:end]
    smoke_test = smoke_test.replace('assert sm == {"tasks": sorted(SMOKE12), "window_min": 40, "budget_min": 70}, sm',
        'assert sm == {"tasks": sorted(SMOKE12), "window_min": 120, "budget_min": 150, "probe_min": 95, "probe_sched_min": 20, "probe": {"poetiq": "a47bf94d", "direct": "b6f77b65"}}, sm')
    smoke_test = smoke_test.replace('(4200, 12, sorted(SMOKE12), 2400)', '(9000, 12, sorted(SMOKE12), 7200)').replace('== 65 * 60', '== 145 * 60').replace('("E29SMOKE", 4200)', '("E29SMOKE", 9000)')
    smoke_test = smoke_test.replace('lines[2].startswith(', 'next(l for l in lines if l.startswith("E29_PROFILE = ")).startswith(')
    t = t[:start] + smoke_test + t[end:]
    compile(t, "<e33 retargeted e29 lib tests>", "exec")
    return t


def nb_unit_files(ns, facts, mod_text, ut_text, libt_text):
    """= build_e29_cpu_kit.main's units files (the E29_DFLASH_D3S branch: the K100Q fixture + notebook staged from a dataset)."""
    K = k29()
    uf = dict(ns["UNIT_FILES"])
    uf.pop("e28k100_unit_tests.py")
    uf.update({"e29_e28k100_unit_tests.py": ut_text, "e29_kit_lib_tests.py": libt_text,
               "e29_kit_facts.json": json.dumps({"variants_cfg": K["BUILD_FACTS"]["variants_cfg"], **K["NBDF_K100_CFG"]}),
               "e29_lib.py": mod_text("e29_lib.py"), "e29_sched_patch.py": mod_text("e29_sched_patch.py"),
               "e29_nbdf.py": mod_text("e29_nbdf.py"), "e28k100.ipynb": lf(E28B.parent / "Q27-XHIGH-E28_K100_EVAL120_6H.ipynb")})
    assert sha(uf["e28k100.ipynb"]) == facts["e29"]["e28_notebook_sha256"]
    staged = {}
    for n in ("k100q_fixture.json", "k100q.ipynb"):          # = E29: from the mounted q27-e29-final-payload (kit size)
        b = uf.pop(n).encode("utf-8")
        assert sha((K["DS_DIR"] / n).read_bytes()) == sha(b), n
        staged[n] = sha(b)
    return uf, staged


# the e29_kit_lib_tests checks of the E29 parts E33 REPLACES by design (DESIGN section 1: "E29's DIRECT scheduler diff is NOT
# used"; build_e33.py W1-W4): expected to fail on the E33 notebook; their E33 counterparts are in the units / real / chain kits
LIBT_EXPECTED_FAIL = {
    "R11": "W1: the E29_DIRECT2 switch line -> E33_CHAIN (E33: e33_kit_switch_tests.py)",
    "R12": "W4: e29_direct_setup -> e33_setup (E33: the real kit's e33_setup / hooks checks)",
    "R13": "W2: E29D_FILES / e29d_sched_patch.py -> E33D_FILES / e33_sched_patch.py (E33: units staged by sha, real kit)",
    "R14": "W3: CELL 3's DIRECT diff step -> the E33 chain diff step (E33: real kit apply / verify / revert; S's tests)",
    "R16": "W4: e29_live_setup -> e33_live_setup (the E33Live wrapper; E33: agent I's nbx, real kit LIVE install check)",
    "R17": "W4: e29_live_setup -> e33_live_setup (E33: real kit LIVE install check)",
    "R27": "W4: E29D_CAN_START -> E33_CAN_START = e33_lib.can_start (E33: e33_kit_switch_tests TestAdmission, chain kit)",
    "R28": "RECORD ONLY (finding for agent I): the E29 LIVE 15-min line printed ' | ledger ' + e29_ledger(...); e33_live_setup's "
           "15-min line does not (the engine's ledger at ready / after F2 is unchanged)",
}
LIBT_UNITS_RUN_OLD = ('    L29["all_pass"] = bool(m) and m.group(1) == m.group(2) and r["rc"] == 0 and not L29["fails"] and '
                      'int(m.group(2)) >= KF["e29_lib_checks_min"]\n')
LIBT_UNITS_RUN_NEW = ('    L29["fail_ids"] = sorted(l.split()[1] for l in lines if l.startswith("FAIL "))   # E33 kit: the E29 parts E33 replaced\n'
                      '    L29["unexpected_fails"] = [x for x in L29["fail_ids"] if x not in KF["libt_expected_fail"]]\n'
                      '    L29["expected_now_pass"] = [x for x in KF["libt_expected_fail"] if x not in L29["fail_ids"]]\n'
                      '    L29["all_pass"] = (bool(m) and int(m.group(2)) >= KF["e29_lib_checks_min"] and not L29["unexpected_fails"]\n'
                      '                       and int(m.group(1)) == int(m.group(2)) - len(L29["fail_ids"]) and r["rc"] in (0, 1))\n')


def run_local(d, files, cmds, env):
    d = Path(d)
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for n, v in files.items():
        (d / n).parent.mkdir(parents=True, exist_ok=True)
        (d / n).write_bytes(v.encode("utf-8") if isinstance(v, str) else v)
    out = {}
    for name, args in cmds:
        r = subprocess.run([sys.executable] + args, capture_output=True, text=True, env=dict(os.environ, PYTHONIOENCODING="utf-8", **env),
                           cwd=str(d), timeout=900, encoding="utf-8", errors="replace")
        o = r.stdout + "\n" + r.stderr
        summ = (re.findall(r"^\d+/\d+ PASS\s*$", o, re.M) or re.findall(r"^Ran \d+ tests? in .*$", o, re.M) or [None])[-1]
        st = (re.findall(r"^(?:OK|FAILED).*$", o, re.M) or [None])[-1]
        out[name] = {"rc": r.returncode, "summary": summ, "status": st,
                     "fails": [l[:500] for l in o.splitlines() if l.startswith(("FAIL", "ERROR"))][:40],
                     "skips": [l[:300] for l in o.splitlines() if l.startswith("SKIP") or " ... skipped" in l][:40]}
        (d / f"_{name}.log").write_text(o, encoding="utf-8")
        print("LOCAL", name, json.dumps(out[name])[:3000], flush=True)
    return out


def e29_selfcheck():
    """The wrapper is faithful: the E29 kit chain loaded here rebuilds E29's real-dflash-d3s kit byte for byte (in memory)."""
    K = k29()
    rec = json.loads(lf_copy_text(E29B / "e29_kit_facts.json"))
    want = rec["kits"]["real-dflash-d3s"]["sha256"]
    if rec.get("builder_sha256") != K["_src_sha"]:
        return {"ok": None, "why": "the E29 kit builder changed after its last kit build (no reference)"}
    ns, facts, mods_nb, mod_text = K["load_chain"]("E29_DFLASH_D3S", "q27-e29-kit-dflash-d3s")
    kf = K["variant_kf"](ns, facts, "E29_DFLASH_D3S")
    kf["staged"] = {n: sha((K["DS_DIR"] / n).read_bytes()) for n in ("k100q_fixture.json", "k100q.ipynb")}   # = E29 main()'s df branch
    K["E29_ARGPARSE"] = K["_mk_argparse"](ns)
    code = chain_kit_code(ns, "real-dflash-d3s", K["real_body"](ns, "E29_DFLASH_D3S", facts), dict(ns["REAL_FILES"]), kf, tag="e29")
    got = sha(kit_notebook(code))
    return {"ok": got == want, "got": got, "want": want}


# ====================================================================== the inputs (pins from the E33 notebook + the dataset)
E18_NB = Path(r"D:/ARC AGI 2/experiments/Experiment 18 - S1-60K on vLLM 0.32 (planned)/Experiment Notebooks/Q27-XHIGH-E18_S1_60K_VLLM032_S20_R2.ipynb")
E18_NB_SHA = "116d2d4101a53eb4ff833e9c97e5a779f0f2bae5735129a4663ad1b42ad79a1d"
E29_DS = ROOT / "ds_q27-e29-final-payload"
E33_STAGE_NAMES = ("e33_sched_patch.py", "e33_poetiq.py", "e29_direct.py", "e33_lib.py", "e33_nbx.py")
FIX_DIRS = (("k100", Path(r"D:/ARC AGI 2/_scratch/e26_gpu_out/q27-xhigh-e26-k100-quality/q27x/phase_Q_K100_FP8_K3")),
            ("w4", Path(r"D:/ARC AGI 2/_scratch/e25_gpu_out/q27-xhigh-e25-nb5-quality/q27x/phase_Q_FP8_K3")))


def nb_code(p):
    return ["".join(c["source"]) for c in json.loads(lf(p))["cells"] if c["cell_type"] == "code"]


def e33_pins(nb_path):
    """CELL 1's E33D_FILES literal (the notebook's own pins) == agent I's facts == the dataset files on disk (read back)."""
    c1 = nb_code(nb_path)[1]
    line = next(l for l in c1.splitlines() if l.startswith("E33D_FILES = "))
    pins = json.loads(line[len("E33D_FILES = "):].split("   #")[0])
    for k, (n, s) in pins.items():
        assert sha((DS_DIR / n).read_bytes()) == s, ("the E33 dataset file != the notebook pin", k, n)
    return pins


def stage_spec(pins, extra=None):
    """KF["stage"]: {published name: [sha256, staged name]}: the E33 dataset files the notebook pins (+ extra E29-dataset files)."""
    st = {n: [s, n] for k, (n, s) in pins.items() if n in E33_STAGE_NAMES}
    st.update(extra or {})
    return st


def build_side_agreement(pins):
    """The agents' Build files == the dataset bytes the notebook loads (the tests in the kits test what ships); a difference is
    recorded (the kit always stages the DATASET bytes)."""
    out = {}
    for k, (n, s) in pins.items():
        b = HERE / n
        out[n] = {"pin": s[:16], "build_file": (sha(b.read_bytes())[:16] if b.is_file() else None)}
        out[n]["same"] = b.is_file() and sha(b.read_bytes()) == s
    return out


def count_tests(text):
    return len(re.findall(r"^    def test_\w+\(self", text, re.M))


def fixture_files():
    files, n = {}, {}
    for lab, ph in FIX_DIRS:
        fs = sorted(ph.glob("duck/*/branch_result.json"))
        n[lab] = len(fs)
        for f in fs:
            files[f"{lab}/{ph.name}/duck/{f.parent.name}/branch_result.json"] = lf_copy_text(f)
    return files, n


def cells01(nb_path):
    """A reduced copy of the notebook: its first two code cells (CELL 0 + CELL 1, the switch / pins / parser), verbatim."""
    nb = json.loads(lf(nb_path))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"][:2]
    return json.dumps({"cells": code, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}, indent=1)


# ---------------------------------------------------------------------- kit 1: units (the E33 module tests + E29 suites)
def units_kit(pins, nb_path):
    sched_t, poetiq_t, nbx_t = lf(HERE / "e33_sched_tests.py"), lf(HERE / "e33_poetiq_tests.py"), lf(HERE / "e33_nbx_tests.py")
    sw_t = lf(HERE / "e33_kit_switch_tests.py")
    e29st, e29dt = lf(E29B / "e29_sched_tests.py"), lf(E29B / "e29d_direct_tests.py")
    assert sha(E18_NB.read_bytes()) == E18_NB_SHA
    fx, nfx = fixture_files()
    files = {"e33_sched_tests.py": sched_t, "e33_poetiq_tests.py": poetiq_t, "e33_poetiq_build.py": lf(HERE / "e33_poetiq_build.py"),
             "e33_kit_run_poetiq_tests.py": lf(HERE / "e33_kit_run_poetiq_tests.py"), "e18r2.ipynb": lf(E18_NB),
             "e33_nbx_tests.py": nbx_t, "e33_kit_switch_tests.py": sw_t, "e33_cells01.ipynb": cells01(nb_path),
             "e29_sched_tests.py": e29st, "e29d_direct_tests.py": e29dt, "e29_sched_patch.py": lf(E29B / "e29_sched_patch.py"),
             "e29_lib.py": lf(E29B / "e29_lib.py"), "e28dir/e28_sched_patch.py": lf(E28B / "e28_sched_patch.py"),
             "exports/nb5_lib.py": lf(E25B / "nb5_lib.py"), **fx}
    mods = k29()["packed_modules"](lf(nb_path))
    assert files["e29_sched_patch.py"] == mods["e29_sched_patch.py"] and files["e29_lib.py"] == mods["e29_lib.py"]
    assert files["e28dir/e28_sched_patch.py"] == mods["e28_sched_patch.py"]
    stage = stage_spec(pins, {"e29d_sched_patch.py": [sha((E29_DS / "e29d_sched_patch.py").read_bytes()), "e29d_sched_patch.py"]})
    live = lf(E29B / "e29_live.py")
    files["e28_sched_patch.py"] = lf(E28B / "e28_sched_patch.py")
    generator = lf(HERE / "e33_sched_build.py")
    marker = 'E29_POST_SHA = "d8603109f9530896e9e4a56e52806202fc2cd4c58bd664d9a25222fe6f7592c2"'
    generator = once(generator, marker, marker + '\nimport os\nE28B_DIR = Path(os.environ["E29_TEST_E28_DIR"])\nE29B_DIR = HERE\nDUCK_PAYLOAD = Path(os.environ["E28_TEST_PAYLOAD"])\n', "portable scheduler generator")
    for name, text in (("e33_sched_build.py", generator), ("e29_live.py", live)):
        (DS_DIR / name).write_bytes(text.encode())
        stage[name] = [sha(text), name]
    st_env = {"E33_TEST_E29_DIR": "{B}", "E29_TEST_E28_DIR": "{B}/e28dir", "E28_TEST_PAYLOAD": "{PAY}", "E28_TEST_K100": "{B}/k100",
              "E28_TEST_W4": "{B}/w4", "E28_TEST_EVAL": "{EVAL}", "E28_TEST_EXPORTS": "{B}/exports", "E28_TEST_TMP": "{K}"}
    tests = [
        {"name": "e33_sched_tests", "file": "e33_sched_tests.py", "args": ["-v"], "env": st_env, "min": count_tests(sched_t), "timeout": 3000},
        {"name": "e33_sched_tests_negative_control", "file": "e33_sched_tests.py", "args": ["-v"], "env": dict(st_env, E33_TEST_NEGCTL="1"),
         "min": 1, "expect_fail": True, "min_fail": 5, "timeout": 3000},
        {"name": "e33_poetiq_tests_real_sandbox", "file": "e33_kit_run_poetiq_tests.py", "args": ["-v"], "env": {}, "min": count_tests(poetiq_t),
         "timeout": 1800},
        {"name": "e33_nbx_tests", "file": "e33_nbx_tests.py", "args": ["-v"], "env": {}, "min": count_tests(nbx_t), "timeout": 900},
        {"name": "e33_kit_switch_tests", "file": "e33_kit_switch_tests.py", "args": [], "env": {"E33KT_NB": "{B}/e33_cells01.ipynb"},
         "min": count_tests(sw_t), "timeout": 300},
        {"name": "e29_sched_tests", "file": "e29_sched_tests.py", "args": ["-v"],
         "env": {k: v for k, v in st_env.items() if k != "E33_TEST_E29_DIR"}, "min": count_tests(e29st), "timeout": 3000},
        {"name": "e29d_direct_tests", "file": "e29d_direct_tests.py", "args": ["-v"],
         "env": {"E29_TEST_E28_DIR": "{B}/e28dir", "E28_TEST_PAYLOAD": "{PAY}", "E28_TEST_TMP": "{K}"}, "min": count_tests(e29dt), "timeout": 3000},
    ]
    kf = {"stage": stage, "duck_payload": list(DUCK_PAYLOAD), "tests": tests, "fixtures": nfx}
    return files, kf


# ---------------------------------------------------------------------- kit 4: chain e2e
def chain_kit(pins, nb_path):
    t = lf(HERE / "e33_kit_chain_e2e.py")
    files = {"e33_kit_chain_e2e.py": t, "e29_sched_tests.py": lf(E29B / "e29_sched_tests.py"), "e29_sched_patch.py": lf(E29B / "e29_sched_patch.py"),
             "e29_lib.py": lf(E29B / "e29_lib.py"), "e28dir/e28_sched_patch.py": lf(E28B / "e28_sched_patch.py")}
    lit = next(l for l in nb_code(nb_path)[1].splitlines() if l.startswith("E33_CHAIN = {"))
    switch = ast.literal_eval(ast.parse(lit[len("E33_CHAIN = "):], mode="eval").body)
    env = {"E29_TEST_E28_DIR": "{B}/e28dir", "E28_TEST_PAYLOAD": "{PAY}", "E28_TEST_TMP": "{K}",
           "E33KT_FILES": json.dumps({k: v for k, v in pins.items() if k in ("direct", "poetiq")}), "E33KT_SWITCH": json.dumps(switch)}
    tests = [{"name": "e33_kit_chain_e2e", "file": "e33_kit_chain_e2e.py", "args": ["-v"], "env": env, "min": count_tests(t), "timeout": 2400}]
    return files, {"stage": stage_spec(pins), "duck_payload": list(DUCK_PAYLOAD), "tests": tests}


# ---------------------------------------------------------------------- kit 2: real (the E29 real kit chain + the E33 section)
E33_REVERT_SRC = r"""    _DS33 = G.get("E33DS")   # E33 kit: the E33 chain diff (CELL 3) -> back to the E29 post first, then E28's revert / re-apply
    E28R["e33_state"] = _DS33.tree_state(E12D) if _DS33 is not None else None
    if E28R["e33_state"] == "E33_PATCHED":
        E28R["e33_revert"] = {k: v for k, v in _DS33.revert_e33(E12D).items() if k != "text"}
        E28R["e33_revert_sha"] = hashlib.sha256((E12D / "arc2_duck.py").read_bytes()).hexdigest()
        E28R["state"] = S_.tree_state(E12D)   # = the E29 post file under the E33 diff
"""
E33_REAPPLY_SRC = r"""    if _DS33 is not None:                   # E33 kit: the E33 chain diff again (= the state the run uses)
        E28R["e33_reapply"] = {k: v for k, v in _DS33.apply_e33(E12D).items() if k != "text"}
        E28R["e33_state_after"] = _DS33.tree_state(E12D)
"""

E33_REAL = r'''
# ---------------------------------------------------------------- E33 kit: the chain diff (CELL 3), the notebook's E33 setup and its
# hooks on a fake token-id server (the REAL klee100 tokenizer / template), Poetiq's sandbox self-test, the notebook's writer
E33K, E33V = {}, {}
import threading as _th33, types as _ty33, tempfile as _tf33
def _in_ds33(p):
    return bool(p) and "q27-e33-chain-payload" in Path(p).resolve().parts
try:
    _keys = ("want", "ok", "path", "error", "applied", "after", "sha_after", "verify_why", "env", "revert", "revert_error")
    _c3, _r3 = G.get("E33_SCHED_REC") or {}, GR.get("E33_SCHED_REC") or {}
    E33K["sched"] = {"commit": {k: _c3.get(k) for k in _keys}, "rerun": {k: _r3.get(k) for k in _keys}}
    E33K["chain"] = {"commit": G.get("E33_CHAIN"), "rerun": GR.get("E33_CHAIN")}
    E33K["files_nb"] = G.get("E33D_FILES")
    E33K["used"] = {n: (G.get("Q27X_PAYLOAD_USED") or {}).get(n) for n, _s in (G.get("E33D_FILES") or {}).values()}
    E33K["env_now"] = {k: os.environ.get(k) for k in KF["e33_env_names"]}
    _ch = lambda c: (isinstance(c, dict) and c.get("on") is True and c.get("poetiq") is True and c.get("direct") is True and c.get("attempt2") is True
                     and not c.get("invalid") and not c.get("harness_override")
                     and {k: c.get(k) for k in KF["e33_spec"]} == KF["e33_spec"])
    E33V["e33_cell1_switch_checked_all_on_spec_budgets_no_invalid_commit_rerun"] = _ch(G.get("E33_CHAIN")) and _ch(GR.get("E33_CHAIN"))
    E33V["e33_cell1_pins_eq_build_and_dataset"] = G.get("E33D_FILES") == KF["e33_pins"] == GR.get("E33D_FILES")
    E33V["e33_cell3_chain_diff_applied_verified_env_commit_rerun"] = all(
        x.get("ok") is True and x.get("after") == "E33_PATCHED" and x.get("sha_after") == KF["e33_post"] and x.get("env") == KF["e33_env_want"]
        and not x.get("error") for x in (_c3, _r3))
    E33V["e33_env_in_process_after_cell3"] = E33K["env_now"] == {k: KF["e33_env_want"].get(k) for k in KF["e33_env_names"]}
    _E28R = REP["checks"].get("e28") or {}
    E33K["revert"] = {k: _E28R.get(k) for k in ("e33_state", "e33_revert", "e33_revert_sha", "e33_reapply", "e33_state_after", "state")}
    E33V["e33_revert_to_e29_post_then_reapply_on_payload_arc2_duck"] = (
        _E28R.get("e33_state") == "E33_PATCHED" and (_E28R.get("e33_revert") or {}).get("after") == "E29_POST"
        and _E28R.get("e33_revert_sha") == KF["e29_post"] and (_E28R.get("e33_reapply") or {}).get("after") == "E33_PATCHED"
        and _E28R.get("e33_state_after") == "E33_PATCHED" and sha_file(Path(os.environ.get("E12_DIR") or (K / "e12_duck")) / "arc2_duck.py") == KF["e33_post"])
    _E12 = G.get("E12A")
    E33V["e33_patched_arc2_duck_loaded_select_rule_hooks"] = (_E12 is not None and hasattr(_E12, "select_attempts_e33")
                                                             and hasattr(_E12, "E33_RUN_POETIQ") and _E12._e33_on() is True)
    _rt = G.get("E33_RT") or {}
    E33K["setup"] = json.loads(json.dumps({k: _rt.get(k) for k in ("state", "items", "hooks", "error", "sched", "tokenizer")}, default=str))
    E33V["e33_setup_on_items_on_three_hooks_on_e12a"] = (
        _rt.get("state") == "ON" and str((_rt.get("items") or {}).get("poetiq")).startswith("ON (e33_poetiq")
        and str((_rt.get("items") or {}).get("direct")).startswith("ON (e29_direct") and sorted(_rt.get("hooks") or []) == ["E33_CAN_START", "E33_RUN_DIRECT", "E33_RUN_POETIQ"]
        and _E12.E33_RUN_POETIQ is G.get("e33_run_poetiq") and _E12.E33_RUN_DIRECT is G.get("e33_run_direct") and _E12.E33_CAN_START is G.get("e33_can_start"))
    E33V["e33_live_wrapper_installed"] = getattr(_E12, "_e33_live_orig", None) is not None and getattr(_E12.solve_all_r6, "__name__", "") == "live_solve"
    E33V["e33_payload_files_from_the_e33_dataset_by_sha"] = (
        bool(E33K["used"]) and all((_in_ds33((u or {}).get("path")) or (n == "e29_direct.py" and "q27-e29-final-payload" in Path(u["path"]).resolve().parts))
                                   and (u or {}).get("sha256") == KF["e33_pins_by_name"][n] and sha_file(u["path"]) == KF["e33_pins_by_name"][n]
                                   for n, u in E33K["used"].items()))
except BaseException as e:
    E33K.update(error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
REP["checks"]["e33"] = E33K; save()
log("E33 setup", json.dumps({k: E33K.get(k) for k in ("sched", "setup", "used", "error")}, default=str)[:4000])

# the hooks on a fake streaming token-id server: the notebook's OWN e33_run_poetiq / e33_run_direct / e33_can_start, the real
# tokenizer through the notebook's own CELL 8 statement (NB5_LOCAL_TOK, cut verbatim) and its e29d_tok; no GPU server
FK = {"log": [], "lock": _th33.Lock()}
E33H = {}
try:
    import http.server as _hs33, socketserver as _ss33
    exec(compile(KF["tok_update_src"], "<CELL 8: NB5_LOCAL_TOK.update (verbatim)>", "exec"), G)
    _tk = G["e29d_tok"]()
    _P = G["E33P"]
    E33H["tokenizer"] = json.loads(json.dumps(G["E33_RT"].get("tokenizer"), default=str))
    _ANS = {"poetiq-wrong": "```python\ndef transform(grid):\n    return [[int(v) for v in row] for row in grid]\n```",
            "poetiq-pass": "```python\ndef transform(grid):\n    return [[int(v) for v in reversed(row)] for row in grid]\n```",
            "direct": "```grid\n6 5\n```"}
    class _H33(_hs33.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_GET(self):
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        def _chunk(self, ids, fr=None):
            self.wfile.write(b"data: " + json.dumps({"choices": [{"token_ids": list(ids), "text": "", "finish_reason": fr}]}).encode() + b"\n\n")
            self.wfile.flush()
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
            tag = str(body.get("model") or "").split("|")[-1]
            text = _tk["decode"](body["prompt"], False)
            with FK["lock"]:
                FK["log"].append({"tag": tag, "max_tokens": body.get("max_tokens"), "temperature": body.get("temperature"), "seed": body.get("seed"),
                                  "prompt_tokens": len(body.get("prompt") or []), "feedback": "EXISTING PARTIAL/INCORRECT SOLUTIONS" in text,
                                  "prev_code": "return [list(r) for r in grid]" in text, "think_open": "<think>" in text})
            self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
            self._chunk(_tk["encode"]("Let me look at the examples."))
            self._chunk([G["E33P"].FN_THINK_CLOSE_ID] + _tk["encode"]("\n\n" + _ANS.get(tag, "nothing")))
            self._chunk([_tk["eos_ids"][0]], "stop")
    class _TS33(_ss33.ThreadingMixIn, _hs33.HTTPServer):
        daemon_threads = True
    _srv33 = _TS33(("127.0.0.1", 0), _H33)
    _th33.Thread(target=_srv33.serve_forever, daemon=True).start()
    _url = f"http://127.0.0.1:{_srv33.server_address[1]}"
    G["SGL_URL"], G["E11_SERVER"] = _url, _ty33.SimpleNamespace(url=_url)    # kit: the booted-server globals the hooks check
    _task = {"train": [{"input": [[1, 2]], "output": [[2, 1]]}, {"input": [[3, 4], [5, 6]], "output": [[4, 3], [6, 5]]}],
             "test": [{"input": [[5, 6]]}]}
    _out = K / "e33_hooks"; _out.mkdir()
    _dl = time.monotonic() + 3 * 3600
    _rc = G["e33_run_poetiq"]("kit00001", _task, {"e33_attempt": 1, "branch": 113}, _dl, _url + "/v1", "kit|poetiq-wrong", _out / "poetiq" / "a1_it1", 1, None)
    _rd = G["e33_run_poetiq"]("kit00001", _task, {"e33_attempt": 1, "branch": 114}, _dl, _url + "/v1", "kit|poetiq-pass", _out / "poetiq" / "a1_it2", 2, _rc)
    _re = G["e33_run_direct"]("kit00001", _task, {"e33_attempt": 1, "branch": 115}, _dl, _url + "/v1", "kit|direct", _out / "direct" / "a1")
    _cs = {s_: G["e33_can_start"](s_, _dl) for s_ in ("C", "D", "E")}
    _cs_late = {s_: G["e33_can_start"](s_, time.monotonic() + 600) for s_ in ("C", "E")}
    _pq = lambda r: (r or {}).get("poetiq") or {}
    E33H.update(c=_pq(_rc), d=_pq(_rd), e=(_re or {}).get("e31"), e_sub=(_re or {}).get("submissions"), can_start=_cs, can_start_late=_cs_late,
                fake=list(FK["log"]), runs=json.loads(json.dumps(G["E33_RT"].get("runs"), default=str)),
                files=sorted(str(p.relative_to(_out)) for p in _out.rglob("*") if p.is_file())[:60])
    _lg = FK["log"]
    _p1 = [x for x in _lg if x["tag"].startswith("poetiq")]
    _d1 = [x for x in _lg if x["tag"] == "direct"]
    E33V["e33_tokenizer_real_klee100_stop_ids_eq_spec_think_close_1_id"] = ((E33H["tokenizer"] or {}).get("stop_ids_equal_spec") is True
                                                                           and (E33H["tokenizer"] or {}).get("think_close_ids") == [G["E33P"].FN_THINK_CLOSE_ID])
    E33V["e33_poetiq_hook_c_wrong_d_pass_with_feedback_real_sandbox"] = (
        (_rc or {}).get("status") == "done" and E33H["c"].get("iteration") == 1 and E33H["c"].get("train_pass") is False
        and E33H["c"].get("parse_status") == "ok" and (E33H["c"].get("sandbox") or {}).get("ok") is True
        and (_rd or {}).get("status") == "done" and E33H["d"].get("iteration") == 2 and E33H["d"].get("train_pass") is True
        and E33H["d"].get("history_in") == 1 and (_rd["submissions"][0]["test_outputs"] == [[[6, 5]]])
        and [x["feedback"] for x in _p1] == [False, True] and [x["prev_code"] for x in _p1] == [False, True])
    E33V["e33_cell1_budgets_reach_the_solvers_60k_104k_temperature_1"] = (
        [x["max_tokens"] for x in _p1] == [60000, 60000] and [x["max_tokens"] for x in _d1][:1] == [104000]
        and all(x["temperature"] == 1.0 for x in _lg) and len(_d1) == 1)
    E33V["e33_direct_hook_grid_record"] = ((_re or {}).get("submissions") or [{}])[0].get("test_outputs") == [[[6, 5]]] and isinstance((_re or {}).get("e31"), dict)
    E33V["e33_can_start_60k_floor_far_true_near_false"] = all(_cs.values()) and not any(_cs_late.values())
    E33V["e33_traces_saved_every_request"] = (sum(1 for f in E33H["files"] if f.endswith("transcript.txt")) >= 3
                                              and len(E33H["runs"]) == 3 and not any(r_.get("error") for r_ in E33H["runs"]))
    # the notebook's own report block (CELL 9 calls e33_report / e33_report_text) + the traces zip, on these three runs
    _bo = {k_: G.get(k_, "__unset__") for k_ in ("BASE_OUT", "WORKING")}
    G["BASE_OUT"], G["WORKING"] = _out, K
    try:
        _rp = G["e33_report"]()
        _rt_txt = G["e33_report_text"](_rp)
    finally:
        for k_, v_ in _bo.items():
            if v_ == "__unset__":
                G.pop(k_, None)
            else:
                G[k_] = v_
    _tr = _rp.get("traces") or {}
    E33H["report"] = {"keys": sorted(_rp), "traces": _tr, "calls": _rp.get("calls"), "state": _rp.get("state"), "text_lines": len(str(_rt_txt).splitlines())}
    E33V["e33_report_block_and_traces_zip"] = (_rp.get("state") == "ON" and (_rp.get("calls") or {}).get("poetiq") == 2
                                               and (_rp.get("calls") or {}).get("direct") == 1 and bool(_tr.get("zip")) and Path(_tr["zip"]).is_file()
                                               and ((_tr.get("by_solver") or {}).get("poetiq") or {}).get("transcripts", 0) >= 2
                                               and ((_tr.get("by_solver") or {}).get("direct") or {}).get("files", 0) >= 1
                                               and isinstance(_rt_txt, str) and "algorithm" in _rt_txt)
    _srv33.shutdown()
except BaseException as e:
    E33H.update(error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2500:])
finally:
    try:
        (G.get("E33_RT") or {}).get("_rate_stop") and G["E33_RT"]["_rate_stop"].set()
    except Exception:
        pass
REP["checks"]["e33_hooks"] = E33H; save()
log("E33 hooks", json.dumps({k: E33H.get(k) for k in ("c", "d", "e", "can_start", "can_start_late", "fake", "error")}, default=str)[:5000])

# Poetiq's sandbox self-test (E18 verbatim) under THIS Kaggle CPU kernel's limits (uid, cpus, cgroup memory)
E33S = {}
try:
    _P = G["E33P"]
    E33S["env"] = {"euid": os.geteuid(), "cpus": os.cpu_count(), "affinity": len(os.sched_getaffinity(0))}
    for _cg in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/cpu.max"):
        if os.path.isfile(_cg):
            E33S["env"][_cg] = open(_cg).read().strip()[:60]
    E33S["ready"] = _P.sandbox_ready(force=True)
    E33S["selftest"] = _P.poetiq_sandbox_selftest()
    _ok, _pl = _P.poetiq_sandbox_run("def transform(grid):\n    while True:\n        pass\n", [[1]], 2.0)
    E33S["timeout_probe"] = [_ok, str(_pl)[:120]]
    E33S["bound"] = {k: _P._SANDBOX.get(k) for k in ("workers", "max_live", "runs")}
    E33V["e33_poetiq_sandbox_selftest_real_kaggle_cpu"] = (E33S["ready"].get("ok") is True and E33S["selftest"].get("ok") is True
                                                         and E33S["selftest"].get("mode") == "uid_drop+seccomp+rlimits+timeout"
                                                         and E33S["timeout_probe"] == [False, "sandbox timeout"] and E33S["env"]["euid"] == 0)
except BaseException as e:
    E33S.update(error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-1500:])
    E33V["e33_poetiq_sandbox_selftest_real_kaggle_cpu"] = False
REP["checks"]["e33_sandbox"] = E33S; save()
log("E33 sandbox", json.dumps(E33S, default=str)[:2000])

# the notebook's own writer (e12_write_submission + _e12_write_json_atomic of e21_cell6, cut verbatim at kit build) in commit and
# rerun mode on the patched arc2_duck (E12A.build_submission -> the E33 answer rule) with E33 records on real test tasks
E33W = {}
try:
    _TC = G["test_challenges"]
    _ones = sorted(t for t, v in _TC.items() if len(v["test"]) == 1)
    _twos = sorted(t for t, v in _TC.items() if len(v["test"]) == 2)
    _X, _Y, _Z = _ones[:3]
    _W = _twos[0] if _twos else None
    _g = lambda v, h=2, w=3: [[v] * w for _ in range(h)]
    def _row(a, st, ph, b):
        return {"phase": ph, "e33_attempt": a, "e33_stage": st, "e33_name": ph.split(":")[0], "e33_kind": {"A": "duck", "B": "duck", "C": "poetiq", "D": "poetiq", "E": "direct"}[st], "branch": b}
    def _duck(tid, a, st, ph, b, grid, acc, seq):
        subs = [{"test_outputs": [grid] * len(_TC[tid]["test"]), "accepted": acc, "train_exact": [True, acc], "t": 1.0 + seq, "branch": b}] if grid else []
        return {"task_id": tid, "branch": b, "status": "done", "accepted": acc, "generated_tokens": 1000, "submissions": subs, "checked": [],
                "publish_seq": seq, "e28": _row(a, st, ph, b)}
    def _pq_rec(tid, a, st, ph, b, grids, tp, score, seq):
        return {"task_id": tid, "branch": b, "status": "done", "accepted": tp, "generated_tokens": 900, "publish_seq": seq,
                "submissions": [{"test_outputs": grids, "accepted": tp, "train_exact": [tp, tp], "t": 5.0 + seq, "branch": b}],
                "poetiq": {"train_pass": tp, "score": score, "train_score": score, "iteration": 1 if st == "C" else 2}, "e28": _row(a, st, ph, b), "e33": _row(a, st, ph, b)}
    def _dr_rec(tid, a, b, grids, seq):
        return {"task_id": tid, "branch": b, "status": "done", "accepted": False, "generated_tokens": 800, "publish_seq": seq,
                "submissions": [{"test_outputs": grids, "accepted": False, "train_exact": [], "t": 9.0 + seq, "branch": b}],
                "e31": {"mode": "natural"}, "e28": _row(a, "E", f"E-DIRECT:{a}", b), "e33": _row(a, "E", f"E-DIRECT:{a}", b)}
    _recs = {_X: [_duck(_X, 1, "A", "A:1", 0, _g(1), True, 1), _pq_rec(_X, 2, "D", "D-POETIQ2:2", 124, [_g(2)], True, 1.0, 2)],
             _Y: [_duck(_Y, 1, "A", "A:1", 0, None, False, 1), _pq_rec(_Y, 1, "C", "C-POETIQ1:1", 113, [_g(3)], False, 0.5, 2),
                  _dr_rec(_Y, 1, 115, [_g(4)], 3)]}
    _want = {_X: [{"attempt_1": _g(1), "attempt_2": _g(2)}], _Y: [{"attempt_1": _g(3), "attempt_2": _g(4)}]}
    if _W:
        _recs[_W] = [_duck(_W, 1, "A", "A:1", 0, None, False, 1), _dr_rec(_W, 1, 115, [_g(5), _g(6, 3, 3)], 2)]
        _want[_W] = [{"attempt_1": _g(5), "attempt_2": _TC[_W]["test"][0]["input"]}, {"attempt_1": _g(6, 3, 3), "attempt_2": _TC[_W]["test"][1]["input"]}]
    _slots = {t: [] for t in _recs}
    for t, v in _recs.items():
        G["E12A"].select_attempts_e33(_TC[t], v, _slots[t])
    E33W["slots"] = _slots
    E33W["env"] = {k: os.environ.get(k) for k in ("E28_SCHED", "E33_SCHED", "E12_ARC2_REV5")}
    for _mode, _rr in (("commit", False), ("rerun", True)):
        _wd = K / f"e33_writer_{_mode}"; _wd.mkdir()
        _wns = {"__builtins__": __builtins__, "identity_submission": G["identity_submission"], "test_challenges": _TC, "RERUN": _rr,
                "E12_RECORDS": {t: list(_recs.get(t, [])) for t in _TC}, "E12A": G["E12A"], "CHALLENGES": _TC,
                "SUBMISSION_PATH": _wd / "submission.json", "_E12_WRITE_LOCK": _th33.Lock(), "_E12_LOCK": _th33.Lock(),
                "Path": Path, "os": os, "json": json, "_tempfile": _tf33}
        exec(compile(KF["writer_src"], "<e21_cell6: _e12_write_json_atomic + e12_write_submission (verbatim)>", "exec"), _wns)
        _wns["e12_write_submission"]()
        _s = json.loads((_wd / "submission.json").read_text(encoding="utf-8"))
        _idn = G["identity_submission"](_TC)
        E33W[_mode] = {"strict_schema_bad": strict_schema(_s, _TC)[:10], "identity": _s == _idn,
                       "e33_rows": {t: _s.get(t) == _want[t] for t in _want},
                       "others_identity": all(_s[t] == _idn[t] for t in _TC if t not in _want)}
    E33V["e33_writer_commit_mode_identity_fallback"] = E33W["commit"]["identity"] is True and not E33W["commit"]["strict_schema_bad"]
    E33V["e33_writer_rerun_mode_e33_answer_rule_slots_strict_schema"] = (
        not E33W["rerun"]["strict_schema_bad"] and all(E33W["rerun"]["e33_rows"].values()) and E33W["rerun"]["others_identity"]
        and E33W["env"]["E28_SCHED"] == "1" and E33W["env"]["E33_SCHED"] == "1"
        and [(r_["a1_step"], r_["a2_step"]) for r_ in _slots[_X]] == [("1-pass", "a2:1-pass")]
        and [(r_["a1_step"], r_["a2_step"]) for r_ in _slots[_Y]] == [("2-best", "a1-next")])
except BaseException as e:
    E33W.update(error=f"{type(e).__name__}: {e}"[:800], tb=traceback.format_exc()[-2000:])
    E33V.setdefault("e33_writer_rerun_mode_e33_answer_rule_slots_strict_schema", False)
REP["checks"]["e33_writer"] = E33W; save()
log("E33 writer", json.dumps(E33W, default=str)[:3000])
for _k33 in ("e33_cell1_switch_checked_all_on_spec_budgets_no_invalid_commit_rerun", "e33_cell3_chain_diff_applied_verified_env_commit_rerun",
             "e33_setup_on_items_on_three_hooks_on_e12a", "e33_poetiq_hook_c_wrong_d_pass_with_feedback_real_sandbox",
             "e33_writer_commit_mode_identity_fallback"):
    E33V.setdefault(_k33, False)
log("E33 verdict items", json.dumps(E33V))
'''

STEPS_E33 = [("CELL 1: the E33_CHAIN switch (plain literal) -> e33_lib.check_chain (from the E33 dataset by sha); E33D_FILES pins",
              ["e33_cell1_switch_checked_all_on_spec_budgets_no_invalid_commit_rerun", "e33_cell1_pins_eq_build_and_dataset"]),
             ("CELL 3: the E33 chain diff (e33_sched_patch.cell3 from the E33 dataset) applied + verified + env, commit + rerun; "
              "revert to the E29 post / re-apply on the payload's arc2_duck.py; the patched module loaded",
              ["e33_cell3_chain_diff_applied_verified_env_commit_rerun", "e33_env_in_process_after_cell3",
               "e33_revert_to_e29_post_then_reapply_on_payload_arc2_duck", "e33_patched_arc2_duck_loaded_select_rule_hooks"]),
             ("e29_setup -> e29_extras -> e33_nbx.py (E33 dataset) -> e33_setup (Poetiq / Direct solvers by sha, the three E12A hooks) + "
              "e33_live_setup", ["e33_setup_on_items_on_three_hooks_on_e12a", "e33_live_wrapper_installed",
                                 "e33_payload_files_from_the_e33_dataset_by_sha"]),
             ("the hooks on a fake token server: CELL 8's NB5_LOCAL_TOK (verbatim) + e29d_tok (the real klee100 tokenizer), Poetiq "
              "C (wrong) -> D (feedback, pass), Direct E, E33_CAN_START, traces",
              ["e33_tokenizer_real_klee100_stop_ids_eq_spec_think_close_1_id", "e33_poetiq_hook_c_wrong_d_pass_with_feedback_real_sandbox",
               "e33_cell1_budgets_reach_the_solvers_60k_104k_temperature_1", "e33_direct_hook_grid_record",
               "e33_can_start_60k_floor_far_true_near_false", "e33_traces_saved_every_request", "e33_report_block_and_traces_zip"]),
             ("Poetiq's sandbox self-test (E18 verbatim) under the Kaggle CPU limits", ["e33_poetiq_sandbox_selftest_real_kaggle_cpu"]),
             ("the writer (e12_write_submission, verbatim) commit + rerun with E33 records (the E33 answer rule)",
              ["e33_writer_commit_mode_identity_fallback", "e33_writer_rerun_mode_e33_answer_rule_slots_strict_schema"])]


def cut_nested(src, outer, names):
    """Nested defs of `outer` (a top-level def of the readable cell text) by name, verbatim, dedented by one level."""
    t = ast.parse(src)
    fn = next(n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == outer)
    segs = []
    for nm in names:
        d = next(n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name == nm)
        seg = ast.get_source_segment(src, d, padded=True)
        lines = seg.splitlines(True)
        assert all(l.startswith("    ") or not l.strip() for l in lines), nm
        segs.append("".join(l[4:] if l.startswith("    ") else l for l in lines))
    out = "\n\n".join(segs)
    for s_ in segs:
        assert "\n".join("    " + l if l.strip() else l for l in s_.splitlines()) in src, "not verbatim"
    compile(out, "<cut>", "exec")
    return out


def tok_update_src(runner_cell):
    """CELL 8's NB5_LOCAL_TOK.update(...) statement (inside its try block), verbatim, dedented."""
    t = ast.parse(runner_cell)
    hits = [n for n in ast.walk(t) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == "update"
            and getattr(n.value.func.value, "id", None) == "NB5_LOCAL_TOK"]
    assert len(hits) == 1, len(hits)
    seg = ast.get_source_segment(runner_cell, hits[0], padded=True)
    ind = len(seg) - len(seg.lstrip(" "))
    out = "\n".join(l[ind:] if l.startswith(" " * ind) else l for l in seg.splitlines()) + "\n"
    compile(out, "<tok>", "exec")
    return out


def real_kit(fpath, pins, nb_path):
    K = k29()
    ns, facts, mods_nb, mod_text = load_chain_e33(fpath, PREFIX.rstrip("-"))
    kf = K["variant_kf"](ns, facts, "E29_DFLASH_D3S")
    # The smoked engine starts below the older donor's top D pin; keep its MTP ladder unchanged.
    kf["d_pins"] = [7665090560, 7396655104, 6859784192]
    assert kf["e29_direct"] is False and kf["sched_path"] == "E29" and kf["e29_df"] is True
    kf["staged"] = {n: sha((E29_DS / n).read_bytes()) for n in ("k100q_fixture.json", "k100q.ipynb")}
    rd = readable(nb_code(nb_path))
    _l33 = {"__name__": "e33_lib_kitb"}
    exec(compile(lf(DS_DIR / "e33_lib.py"), "e33_lib.py", "exec"), _l33)                     # the DATASET bytes
    lit = next(l for l in rd[1].splitlines() if l.startswith("E33_CHAIN = {"))
    chain = _l33["check_chain"](ast.literal_eval(ast.parse(lit[len("E33_CHAIN = "):], mode="eval").body))
    order_line = next(l for l in rd[1].splitlines() if l.startswith("E33_A1_ORDER = "))
    chain["a1_order"] = ast.literal_eval(ast.parse(order_line).body[0].value)
    assert chain["a1_order"] == "e28", "release must follow the fixed answer-order design"
    assert sha((DS_DIR / "e33_sched_patch.py").read_bytes()) == pins["sched"][1]
    _ns33 = {"__name__": "e33_sched_patch_kitb"}
    if str(E28B) not in sys.path:
        sys.path.append(str(E28B))                    # its applier import (e28_sched_patch, read only)
    exec(compile(lf(DS_DIR / "e33_sched_patch.py"), "e33_sched_patch.py", "exec"), _ns33)   # the DATASET bytes' constants
    kf.update(e33_pins=pins, e33_pins_by_name={n: s for n, s in pins.values()}, e33_post=_ns33["POST"], e29_post=_ns33["PRE"],
              e33_env_names=list(_ns33["ENV_NAMES"]), e33_env_want=_ns33["server_env"](chain),
              e33_spec={"poetiq_think": 60000, "poetiq_answer": 8192, "direct_think": 104000, "direct_answer": 8192, "temperature": 1.0},
              writer_src=cut_nested(next(c for c in rd if "def e21_cell6():" in c), "e21_cell6", ["_e12_write_json_atomic", "e12_write_submission"]),
              tok_update_src=tok_update_src(next(c for c in rd if "NB5_LOCAL_TOK.update(" in c)))
    assert kf["e29_post"] == facts["e29"]["sched_sha"]["post"]
    K["E29_ARGPARSE"] = K["_mk_argparse"](ns)
    keep = {k: K[k] for k in ("DIRECT_REVERT_SRC", "DIRECT_REAPPLY_SRC", "STEPS_DIRECT")}
    try:
        K["DIRECT_REVERT_SRC"], K["DIRECT_REAPPLY_SRC"], K["STEPS_DIRECT"] = E33_REVERT_SRC, E33_REAPPLY_SRC, STEPS_E33
        body = K["real_body"](ns, "E29_DFLASH_D3S", facts)
    finally:
        K.update(keep)
    v = body.index('sc, sr = SUB.get("commit") or {}, SUB.get("rerun") or {}\nverdict({')
    body = body[:v] + E33_REAL + body[v:]
    body = once(body, "    **QDV, **E28V, **E29V,\n}, recorded={", '    **QDV, **E28V, **E29V, **E33V,\n}, recorded={"e33": {k: v for k, v in E33K.items() if k != "tb"}, ',
                "verdict e33")
    code = chain_kit_code(ns, "real", body, dict(ns["REAL_FILES"]), kf)
    return code, ns, facts, kf


# ---------------------------------------------------------------------- kit 3: units-nb (E29's notebook unit suites on the E33 notebook)
def units_nb_kit(fpath, nb_path):
    K = k29()
    ns, facts, mods_nb, mod_text = load_chain_e33(fpath, PREFIX.rstrip("-"))
    kf = K["variant_kf"](ns, facts, "E29_DFLASH_D3S")
    uf, staged = nb_unit_files(ns, facts, mod_text, K["UT29"], libt_e33(K["LIBT"]))
    pins = e33_pins(nb_path)
    kf["staged"] = dict(staged, **{pins["nbx"][0]: pins["nbx"][1]})   # + the E33 nbx the retargeted lib tests read
    kf["libt_expected_fail"] = LIBT_EXPECTED_FAIL
    body = K["units_body"](ns["UNITS"])
    body = once(body, LIBT_UNITS_RUN_OLD, LIBT_UNITS_RUN_NEW, "libt expected fails")
    body = once(body, '_hit = [f for f in sorted(Path("/kaggle/input").glob("**/" + _sn)) if sha_file(f) == _ss]',
                '_hit = [f for f in sorted(Path(INPUT_ROOT).glob("**/" + _sn)) if sha_file(f) == _ss]   # E33 kit: the input root', "stage root")
    body = once(body, '         "e29_kit_lib_tests_all_pass": L29.get("all_pass") is True},',
                '         "e29_kit_lib_tests_all_pass_except_the_parts_e33_replaced": L29.get("all_pass") is True,\n'
                '         "k100q_fixtures_e29_dataset_and_e33_nbx_e33_dataset_staged_by_sha": len(U.get("staged_from_dataset") or {}) == len(KF["staged"])\n'
                '         and all(U["staged_from_dataset"].values())},',
                "verdict")
    body = once(body, 'recorded={"e29_lib": {k: L29.get(k) for k in ("summary", "fails")},',
                'recorded={"e29_lib": {k: L29.get(k) for k in ("summary", "fail_ids", "unexpected_fails", "expected_now_pass")}, '
                '"libt_expected_fail": KF["libt_expected_fail"],', "recorded")
    return body, uf, kf, ns


def simple_kit_code(name, body, files, kf, nb_sha):
    assert all("\r" not in v for v in files.values()), name
    blob = base64.b85encode(lzma.compress(json.dumps(files).encode("utf-8"), preset=9 | lzma.PRESET_EXTREME)).decode("ascii")
    code = KIT_HEAD + body
    for k, v in (("__SHAS__", json.dumps({n: sha(v) for n, v in files.items()})), ("__NBSHA__", nb_sha), ("__NAME__", name),
                 ("__KF__", json.dumps(kf)), ("__BLOB__", blob)):
        code = code.replace(k, v)
    for ph in ("__SHAS__", "__NBSHA__", "__KF__", "__BLOB__", "__NAME__"):
        assert ph not in code, (name, ph)
    compile(code, f"<kit {name}>", "exec")
    return code


def kit_meta(slug, title, nb_meta, models):
    """The notebook's own kernel-metadata (datasets, competition, image) on a CPU kernel; models only for the real kit."""
    return {"id": f"{OWNER}/{slug}", "title": title, "code_file": f"{slug}.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": False, "enable_tpu": False, "enable_internet": False, "keywords": [],
            "dataset_sources": list(nb_meta["dataset_sources"]), "kernel_sources": [], "competition_sources": list(nb_meta["competition_sources"]),
            "model_sources": list(models), "docker_image": nb_meta["docker_image"], "machine_shape": "None"}


def write_kit(name, code, meta):
    slug = PREFIX + name
    data = kit_notebook(code)
    assert len(data) < 400_000, (name, len(data))
    out = ROOT / f"{slug}__{OWNER}"
    out.mkdir(parents=True, exist_ok=True)
    for f in out.glob("*"):
        if f.is_file():
            f.unlink()
    (out / f"{slug}.ipynb").write_bytes(data)
    (out / "kernel-metadata.json").write_bytes((json.dumps(meta, indent=2) + "\n").encode("utf-8"))
    assert b"\r" not in (out / "kernel-metadata.json").read_bytes() and (out / f"{slug}.ipynb").read_bytes() == data
    print("kit", out.name, "|", len(data), "bytes | sha256", sha(data), flush=True)
    return {"folder": str(out), "sha256": sha(data), "bytes": len(data), "kernel_metadata": meta}


def a7_marked_hunks(nb_path):
    """Build-time check (the E29 kit's A6 rule, one level up): every changed hunk of the E33 notebook vs E29_DFLASH_D3S (readable
    cells) adds a line marked E33 (or E29: an E29 text moved)."""
    import difflib
    a = readable(nb_code(E29B.parent / "Experiment Notebooks" / "Q27-XHIGH-E29_DFLASH_D3S.ipynb"))
    b = readable(nb_code(nb_path))
    bad, n_h = [], 0
    for i, (x, y) in enumerate(zip(a, b)):
        hunk, has = [], False
        for ln in list(difflib.unified_diff(x.splitlines(), y.splitlines(), lineterm="", n=0))[2:] + ["@@"]:
            if ln.startswith("@@"):
                if hunk:
                    n_h += 1
                    if not has:
                        bad.append((i, hunk[:3]))
                hunk, has = [], False
            elif ln.startswith(("+", "-")):
                hunk.append(ln[:140])
                has = has or (ln.startswith("+") and ("E33" in ln.upper()))
    return {"cells": [len(a), len(b)], "hunks": n_h, "unmarked": bad[:12], "n_unmarked": len(bad)}


LAYOUT_PROBE = r'''
# ---- the local layout simulation (build_e33_cpu_kit.py --sim): this kit's HEAD ran (FILES decoded, sha-checked); now only the
# input discovery the body does on Kaggle (no test, no install), then a layout verdict
LY = {"payload": find_payload(), "ref": str(ref_mount()) if ref_mount() is not None else None, "comp": str(COMPD), "staged": {}, "q27x": {}}
for _sn, _ss in (KF.get("staged") or {}).items():
    LY["staged"][_sn] = [str(f) for f in sorted(Path(INPUT_ROOT).glob("**/" + _sn)) if sha_file(f) == _ss][:1]
if KF.get("e33_pins"):            # the real kit: the notebook's own q27x_payload (CELL 1, verbatim) finds every E33 file
    import ast as _ast
    _g = {"Path": Path, "os": os, "hashlib": hashlib, "print": lambda *a, **k: None}
    _c1 = ["".join(c["source"]) for c in json.loads(FILES["e26.ipynb"])["cells"] if c["cell_type"] == "code"][1]
    _t = _ast.parse(_c1)
    _keep = [n for n in _t.body if (isinstance(n, _ast.Assign) and all(isinstance(x, _ast.Name) and x.id.startswith("Q27X_PAYLOAD") for x in n.targets))
             or (isinstance(n, _ast.FunctionDef) and n.name == "q27x_payload")]
    exec(compile(_ast.Module(_keep, []), "<CELL 1: Q27X_PAYLOAD_* + q27x_payload (verbatim)>", "exec"), _g)
    _g["Q27X_PAYLOAD_USED"] = {}
    os.environ["Q27X_PAYLOAD_DIR"] = INPUT_ROOT
    for _k, (_n, _s) in KF["e33_pins"].items():
        try:
            _g["q27x_payload"](_n, _s)
            LY["q27x"][_n] = _g["Q27X_PAYLOAD_USED"][_n]["path"]
        except Exception as e:
            LY["q27x"][_n] = f"MISSING {type(e).__name__}: {e}"[:300]
REP["checks"]["layout"] = LY; save()
_ok = {"duck_payload_found_by_sha": bool(LY["payload"].get("used")), "competition_dir_present": COMPD.is_dir(),
       "tf_ref_mount_found": LY["ref"] is not None or "tf_ref" not in KF.get("_layout_needs", ["tf_ref"]),
       "staged_files_found_by_sha": all(LY["staged"].values())}
if KF.get("e33_pins"):
    _ok["notebook_q27x_payload_finds_every_e33_file_by_sha"] = all(
        "q27-e33-chain-payload" in str(v) or (k == "e29_direct.py" and "q27-e29-final-payload" in str(v)) for k, v in LY["q27x"].items())
print("LAYOUT", json.dumps(LY, default=str)[:3000], flush=True)
REP["verdict"] = {"checks": _ok, "ALL_OK": all(_ok.values()), "layout_only": True}; save()
print(f"@@KIT {KIT} LAYOUT {'ALL OK' if all(_ok.values()) else 'NOT OK: ' + ', '.join(k for k, v in _ok.items() if not v)}@@", flush=True)
raise SystemExit(0)
'''


def make_sim_input():
    """A /kaggle/input-shaped tree with the mounted datasets these kits read (the E33 / E29 payloads, the Duck payload, the tf-ref
    version, the competition files). The Models (klee100, Merlin env, drafter: tens of GB) are not simulated."""
    inp = SIM / "input"
    if inp.exists():
        shutil.rmtree(inp)
    d = inp / "datasets" / OWNER
    for src, slug in ((DS_DIR, DS_SLUG), (E29_DS, "q27-e29-final-payload"), (ROOT / "ds_e25_nb0_tf_ref", "q27-e25-nb0-tf-ref"),
                      (ROOT / "ds_e25_nb0_tf_ref_k100q", "q27-e25-nb0-tf-ref")):   # the tf-ref version: base + the K100Q staging
        (d / slug).mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and f.name != "dataset-metadata.json":   # Kaggle never mounts it
                shutil.copyfile(f, d / slug / f.name)
    (d / "q27-e15b-payload").mkdir(parents=True)
    shutil.copyfile(E15_PAYLOAD / DUCK_PAYLOAD[0], d / "q27-e15b-payload" / DUCK_PAYLOAD[0])
    c = inp / "competitions" / "arc-prize-2026-arc-agi-2"
    c.mkdir(parents=True)
    for n in ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json"):
        shutil.copyfile(EVAL_JSON / n, c / n)
    ev = json.loads((EVAL_JSON / "arc-agi_evaluation_challenges.json").read_text(encoding="utf-8"))
    (c / "arc-agi_test_challenges.json").write_text(json.dumps(ev), encoding="utf-8")   # the test file: the eval tasks' shape
    return inp


def sim_run(name, code, inp):
    d = SIM / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    (d / "kit.py").write_text(code, encoding="utf-8", newline="\n")
    r = subprocess.run([sys.executable, "-I", str(d / "kit.py")], capture_output=True, text=True, cwd=str(d), timeout=900,
                       env=dict(os.environ, KIT_INPUT=str(inp), KIT_TMP=str(d / "tmp"), KIT_WORK=str(d / "working"), E33KIT_LAYOUT_ONLY="1",
                                PYTHONIOENCODING="utf-8"), encoding="utf-8", errors="replace")
    (d / "kit.log").write_text(r.stdout + "\n--- stderr ---\n" + r.stderr, encoding="utf-8")
    line = next((l for l in r.stdout.splitlines()[::-1] if l.startswith("@@KIT")), None)
    print("SIM", name, "rc", r.returncode, "|", line, "|", r.stderr[-600:] if r.returncode else "", flush=True)
    return {"rc": r.returncode, "line": line, "log": str(d / "kit.log")}


def main(only=None, local_units=False, sim=False):
    d = discover()
    nb_path = d["notebook"]
    assert nb_path is not None and d["facts"] is not None, "the E33 notebook / facts are not built yet (agent I)"
    rec = json.loads(lf_copy_text(HERE / "e33_chain_build_facts.json"))
    assert sha(nb_path.read_bytes()) == rec["notebook_sha256"], "agent I's facts are not for the notebook on disk"
    pins = e33_pins(nb_path)
    fpath, facts = e33_facts()
    nb_meta = rec["push"]["kernel_metadata"]
    assert facts["kernel_metadata"] == nb_meta and nb_meta["docker_image"].endswith("37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461")
    out = {"notebook": str(nb_path), "notebook_sha256": rec["notebook_sha256"], "pins": pins, "build_side": build_side_agreement(pins),
           "a7_e33_hunks_vs_e29_dflash": a7_marked_hunks(nb_path), "e29_chain_selfcheck": e29_selfcheck(), "kits": {}}
    print("build side == dataset:", json.dumps(out["build_side"]), "| A7:", json.dumps(out["a7_e33_hunks_vs_e29_dflash"])[:600], flush=True)
    codes = {}
    want = only or ["units", "units-nb", "real", "chain"]
    if "units" in want:
        files, kf = units_kit(pins, nb_path)
        c_ = simple_kit_code("units", UNITS_BODY, files, kf, rec["notebook_sha256"])
        codes["units"] = (c_, [], c_)
    if "chain" in want:
        files, kf = chain_kit(pins, nb_path)
        c_ = simple_kit_code("chain", CHAIN_BODY, files, kf, rec["notebook_sha256"])
        codes["chain"] = (c_, [], c_)
    if "units-nb" in want:
        body, uf, kf, ns = units_nb_kit(fpath, nb_path)
        codes["units-nb"] = (chain_kit_code(ns, "units-nb", body, uf, kf), [], chain_kit_code(ns, "units-nb", LAYOUT_PROBE, uf, kf))
    if "real" in want:
        code, ns, facts_r, kf = real_kit(fpath, pins, nb_path)
        codes["real"] = (code, list(nb_meta["model_sources"]), chain_kit_code(ns, "real", LAYOUT_PROBE, dict(ns["REAL_FILES"]), kf))
    titles = {"units": "Q27 E33 Kit Units", "units-nb": "Q27 E33 Kit Units NB", "real": "Q27 E33 Kit Real", "chain": "Q27 E33 Kit Chain"}
    for n, (code, models, _sc) in codes.items():
        out["kits"][n] = write_kit(n, code, kit_meta(PREFIX + n, titles[n], nb_meta, models))
    if sim:
        inp = make_sim_input()
        out["sim"] = {"input": str(inp)}
        for n, (code, _m, simcode) in codes.items():
            out["sim"][n] = sim_run(n, simcode, inp)
    if local_units:
        out["local"] = run_local_units(nb_path, pins, fpath)
    return out


def run_local_units(nb_path, pins, fpath):
    """The quick units locally (no long e2e): the switch tests, agent I's nbx tests, the E29 notebook suites on the E33 notebook."""
    res = {}
    d = LOCAL / "units_quick"
    files = {"e33_kit_switch_tests.py": lf(HERE / "e33_kit_switch_tests.py"), "e33_cells01.ipynb": cells01(nb_path),
             "e33_nbx_tests.py": lf(HERE / "e33_nbx_tests.py"), "e29_lib.py": lf(E29B / "e29_lib.py")}
    for n in ("e33_lib.py", "e33_nbx.py", "e33_poetiq.py", "e29_direct.py", "e33_sched_patch.py"):
        files[n] = (DS_DIR / n).read_bytes()
    res.update(run_local(d, files, [("switch", ["e33_kit_switch_tests.py"]), ("nbx", ["e33_nbx_tests.py"])],
                         {"E33KT_NB": str(d / "e33_cells01.ipynb")}))
    K = k29()
    ns, facts, mods_nb, mod_text = load_chain_e33(fpath, PREFIX.rstrip("-"))
    uf, staged = nb_unit_files(ns, facts, mod_text, K["UT29"], libt_e33(K["LIBT"]))
    uf["e33_nbx.py"] = (DS_DIR / "e33_nbx.py").read_bytes()
    d2 = LOCAL / "units_nb_quick"
    env = dict(E28KT_NB=facts["notebook"], E28KT_FACTS=str(fpath), E28KT_NB5LIB=str(d2 / "nb5_lib.py"), E29KT_E28_NB=str(d2 / "e28k100.ipynb"),
               E28KT_K100Q_FIXTURE=str(E29_DS / "k100q_fixture.json"), E28KT_K100Q_NB=str(E29_DS / "k100q.ipynb"), E28KT_E18=str(d2 / "e18r2_cells01.ipynb"),
               E29LT_NB=facts["notebook"], E29LT_FACTS=str(fpath), E29LT_KF=str(d2 / "e29_kit_facts.json"), E29LT_NB5LIB=str(d2 / "nb5_lib.py"),
               E29LT_PAYLOAD_DIR=str(E29_DS))
    r = run_local(d2, uf, [("e29_e28k100_unit_tests", ["e29_e28k100_unit_tests.py"]), ("e29_kit_lib_tests", ["e29_kit_lib_tests.py"])], env)
    fails = sorted(l.split()[1] for l in r["e29_kit_lib_tests"]["fails"] if l.startswith("FAIL "))
    r["e29_kit_lib_tests"]["unexpected_fails"] = [x for x in fails if x not in LIBT_EXPECTED_FAIL]
    res.update(r)
    # the chain e2e + the real kit's writer / hooks blocks on the real modules (seconds; LOCAL: the in-process sandbox stand-in,
    # E33KT_LOCAL_SANDBOX=1; the Kaggle kits run the real E18 sandbox and the real klee100 tokenizer)
    files = {n: lf(HERE / n) for n in ("e33_kit_chain_e2e.py", "e33_kit_writer_quick.py", "e33_kit_hooks_quick.py")}
    files.update({n: lf(E29B / n) for n in ("e29_sched_tests.py", "e29_sched_patch.py", "e29_lib.py")})
    for n in ("e33_lib.py", "e33_nbx.py", "e33_poetiq.py", "e29_direct.py", "e33_sched_patch.py"):
        files[n] = (DS_DIR / n).read_bytes()
    env = {"E29_TEST_E28_DIR": str(E28B), "E33KT_LOCAL_SANDBOX": "1", "E33KWQ_BUILDER": str(HERE / "build_e33_cpu_kit.py"),
           "E33KWQ_EVAL": str(EVAL_JSON / "arc-agi_evaluation_challenges.json"),
           "E33KT_FILES": json.dumps({k: v for k, v in pins.items() if k in ("direct", "poetiq")})}
    res.update(run_local(LOCAL / "chain_quick", files, [("chain_e2e_local_sandbox", ["e33_kit_chain_e2e.py"]),
                                                         ("real_writer_block", ["e33_kit_writer_quick.py"]),
                                                         ("real_hooks_block", ["e33_kit_hooks_quick.py"])], env))
    return res


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        print("E29 chain self-check", e29_selfcheck())
        sys.exit(0)
    _only = sys.argv[sys.argv.index("--only") + 1].split(",") if "--only" in sys.argv else None
    res = main(only=_only, local_units="--local-units" in sys.argv, sim="--sim" in sys.argv)
    res["builder_sha256"] = sha((HERE / "build_e33_cpu_kit.py").read_bytes())
    kfp = HERE / "e33_kit_facts.json"
    old = json.loads(kfp.read_text(encoding="utf-8")) if (_only and kfp.is_file()) else {}
    res["kits"] = dict(old.get("kits") or {}, **res["kits"])
    kfp.write_bytes((json.dumps(res, indent=1, default=str) + "\n").encode("utf-8"))
    print(json.dumps({k: (v.get("bytes"), v.get("sha256", "")[:16]) for k, v in res["kits"].items()}, indent=1))
